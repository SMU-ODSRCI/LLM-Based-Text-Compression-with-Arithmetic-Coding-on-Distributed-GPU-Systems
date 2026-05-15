# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

import os
import re
import time
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

# Distributed configuration
def distributed_config(rank, world_size, local_rank):
	host = os.environ.get('MASTER_ADDR', None)
	if host is None:
		host = os.environ['SLURM_NODELIST'].split(',')[0]
		m = re.match(r'([a-zA-Z\-]+)\[(\d+)-(\d+)\]', host)
		if m:
			host = f'{m.group(1)}{m.group(2)}'
		os.environ['MASTER_ADDR'] = host

	os.environ['MASTER_PORT'] = '29504'
	os.environ['NCCL_P2P_LEVEL'] = 'SYS'
	os.environ['OMP_NUM_THREADS'] = '1'
	dist.init_process_group(backend = 'nccl', init_method = f'tcp://{host}:29504', rank = rank, world_size = world_size)
	torch.cuda.set_device(local_rank)
	torch.set_num_threads(1)
	torch.set_num_interop_threads(1)

def clean_process():
	dist.destroy_process_group()

# Dataset loading and preprocessing
class TextDataset(Dataset):
	"""
	TextDataset() class for next-token prediction training with a fixed sliding window.
	For each token position in this class, we consider an overlapping context window of 64 tokens, where:
		- context = tokens[idx:idx + context_window + 1]
        - ground truth token (target) = [-100, ..., -100, input_ids[-1]]
    Windowed training samples are no longer precomputed and stored using Python lists as it can make the dataset construction computationally expensive.
    Instead, the original token sequence is stored once and each window is created only when DataLoader asks for the token sample at position 'idx'.
    This still preserves the same stride-1 overlapping windows, but with lower memory consumption.

    Since inference uses 64 context tokens as Llama's decoder input (no pad token handling required),
    training (fine-tuning) should also use the same decoder contexts, but compute loss only at the last decoder position, predicting exactly one next-token per sliding window.

	For fine-tuning Llama, we use context_window + 1 to slice the sequence of tokens into input IDs because
	Hugging Face decoder-only causal LMs compute loss using an internal shifting rule so that logits at position i is trained to predict the label position i + 1.
    Therefore, we need to include the target token inside the sequence as the final token.
    """
	def __init__(self, tokens, context_window = None):
		super().__init__()
		self.tokens = torch.tensor(tokens, dtype = torch.long)
		self.context_window = context_window

		if self.context_window <= 0:
			raise ValueError('Context window must be a positive integer.')
		if self.tokens.numel() <= self.context_window:
			raise ValueError(f'Not enough tokens {self.tokens.numel()} for context window = {self.context_window}.')

	def __len__(self):
		return self.tokens.numel() - (self.context_window + 1) + 1 # No. of training samples

	def __getitem__(self, idx):
		if idx < 0 or idx >= len(self):
			raise IndexError(f'Index {idx} out of range for dataset of length {len(self)}.')

		# Construct inputs IDs (contexts) and corresponding attention masks for Llama's decoder by using index context window + 1 as the last position because Llama shifts labels internally
		decoder_input_ids = self.tokens[idx:idx + self.context_window + 1]
		decoder_attention_masks = torch.ones(self.context_window + 1, dtype = torch.long)

		# Next-token labels with masking applied to the preceding token positions as we only supervise the last position to match the inference objective
		labels = torch.full((self.context_window + 1,), -100, dtype = torch.long)
		labels[-1] = decoder_input_ids[-1]

		return decoder_input_ids, decoder_attention_masks, labels

# Llama-3.2-3B fine-tuning for next-token predictions
def train_llama(llama_model, dataloader, train_sampler, device, rank, local_rank, epochs = 10, lr = 1e-5, max_norm = 1.0, fused_AdamW = True, torch_compile = True):
	llama_model = DDP(llama_model, device_ids = [local_rank])
	# Apply torch.compile() after wrapping Llama with DDP
	if torch_compile:
		llama_model = torch.compile(llama_model)
	optimizer = AdamW(llama_model.parameters(), lr = lr, weight_decay = 0.01, fused = fused_AdamW) # Setting fused to True allows AdamW to use fused CUDA kernels that combine multiple optimizer operations into fewer GPU kernel launches, reducing memory traffic and improving training throughput

	llama_model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		total_steps = 0.0
		train_sampler.set_epoch(epoch)
		for step, (decoder_input_ids, decoder_attention_masks, labels) in enumerate(dataloader):
			optimizer.zero_grad(set_to_none = True)
			decoder_input_ids, decoder_attention_masks, labels = [x.to(device, non_blocking = False) for x in (decoder_input_ids, decoder_attention_masks, labels)]
			with autocast('cuda', dtype = torch.bfloat16):
				outputs = llama_model(input_ids = decoder_input_ids, attention_mask = decoder_attention_masks, labels = labels, use_cache = False)
				loss = outputs.loss
			loss.backward()
			clip_grad_norm_(llama_model.parameters(), max_norm)
			optimizer.step()
			total_loss += loss.item()
			total_steps += 1.0
		# Aggregrate loss values and total steps from all GPUs, convert them to scalar tensors, keep them on same GPU device, and then stack them together
		metrics = torch.stack([torch.tensor(total_loss, device = device, dtype = torch.float32), torch.tensor(total_steps, device = device, dtype = torch.float32)])
		dist.all_reduce(metrics, op = dist.ReduceOp.SUM)
		global_total_loss, global_total_steps = metrics.tolist()
		avg_loss = global_total_loss / global_total_steps
		if rank == 0:
			print(f'[Epoch: {epoch + 1} / {epochs}], average loss: {avg_loss:.4f}', flush = True)
	return llama_model

# Saving the fine-tuned model checkpoints
def save_model(llama_model, save_path):
	os.makedirs(save_path, exist_ok = True)
	model_to_save = llama_model.module if hasattr(llama_model, 'module') else llama_model
	torch.save(model_to_save.state_dict(), f'{save_path}/pytorch_model_Llama-3.2-3B_6GPUs.bin')
	print(f'Fine-tuned Llama-3.2-3B checkpoints saved to {save_path}', flush = True)

# Main worker function
def main_worker(rank, world_size, local_rank, epochs = 10, batch_size = 128, lr = 1e-5, max_norm = 1.0, fused_AdamW = True, torch_compile = True):
	distributed_config(rank, world_size, local_rank)
	device = torch.device('cuda', local_rank)

	with open('wiki.train.txt', 'r', encoding = 'utf-8') as f:
		lines = []
		for i, line in enumerate(f):
			if i >= 50000:
				break
			lines.append(line.strip())

	text = " ".join(lines)
	tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
	tokens = tokenizer.encode(text, add_special_tokens = False)

	train_dataset = TextDataset(tokens, context_window = 64)
	train_sampler = DistributedSampler(train_dataset, num_replicas = world_size, rank = rank, shuffle = True)
	dataloader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, num_workers = 0, pin_memory = False)
	llama_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B', dtype = torch.bfloat16).to(device)

	start_time = time.time()
	llama_model = train_llama(llama_model, dataloader, train_sampler, device, rank, local_rank, epochs, lr, max_norm, fused_AdamW = fused_AdamW, torch_compile = torch_compile)
	end_time = time.time()
	total_time = end_time - start_time

	# Get the maximum wall-clock time for the distributed job to end
	time_tensor = torch.tensor([total_time], device = device, dtype = torch.float32)
	dist.all_reduce(time_tensor, op = dist.ReduceOp.MAX)
	wall_clock_time = time_tensor.item()

	if rank == 0:
		print(f'Total wall clock time for fine-tuning Llama-3.2-3B: {wall_clock_time:.4f} seconds', flush = True)
		save_model(llama_model, 'Llama-3.2-3B_finetuned_6GPUs')
	clean_process()

if __name__ == '__main__':
	# SLURM environment variables
	rank = int(os.environ.get('SLURM_PROCID'))
	world_size = int(os.environ.get('SLURM_NTASKS'))
	local_rank = int(os.environ.get('SLURM_LOCALID'))
	main_worker(rank = rank, world_size = world_size, local_rank = local_rank, epochs = 10, batch_size = 128, lr = 1e-5, max_norm = 1.0)
