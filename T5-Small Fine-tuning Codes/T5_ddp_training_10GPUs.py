# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

# Importing necessary PyTorch, transformer, and other modules
import os
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration, EncoderDecoderCache
from torch.optim import AdamW
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

# Distributed Configuration
def distributed_config(rank, world_size, local_rank):
	host = os.environ.get('MASTER_ADDR', None)
	if host is None:
		host = os.environ['SLURM_NODELIST'].split(',')[0]
		m = re.match(r'([a-zA-Z\-]+)\[(\d+)-(\d+)\]', host)
		if m:
			host = f'{m.group(1)}{m.group(2)}'
		os.environ['MASTER_ADDR'] = host
	os.environ['MASTER_PORT'] = '29500'
	os.environ['NCCL_P2P_LEVEL'] = 'SYS'
	os.environ['OMP_NUM_THREADS'] = '1'
	dist.init_process_group(backend = 'nccl', init_method = f'tcp://{host}:29500', rank = rank, world_size = world_size)
	torch.cuda.set_device(local_rank)
	torch.set_num_threads(1)
	torch.set_num_interop_threads(1)

def clean_process():
	dist.destroy_process_group()

# T5 neural network predictor for causal prediction
class CausalT5(nn.Module):
	def __init__(self, device = None, model_name = 't5-small'):
		super().__init__()
		self.device = device
		self.t5model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

	def forward(self, encoder_input_ids = None, encoder_attention_masks = None, decoder_input_ids = None, decoder_attention_masks = None, labels = None, past_key_values = None, use_cache = False):
		if past_key_values is not None and not isinstance(past_key_values, EncoderDecoderCache):
			past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
		outputs = self.t5model(input_ids = encoder_input_ids, attention_mask = encoder_attention_masks, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_masks, labels = labels, past_key_values = past_key_values, use_cache = use_cache, return_dict = True)
		return outputs

# Data Loading and Preprocessing for fine-tuning T5-Small in a causal, autoregressive manner
class TextDataset(Dataset):
	"""
	TextDataset() class for next-token prediction training with a fixed sliding window.
	For each token position in this class, we consider an overlapping context window of 64 tokens, where:
		- context = tokens[idx:idx + context_window]
		- ground truth token (target) = tokens[idx + context_window]
	Windowed training samples are no longer precomputed and stored using Python lists as it can make the dataset construction computationally expensive
	Instead, the original token sequence is stored once and each window is created only when DataLoader asks for the token sample at position 'idx'
	This still preserves the same stride-1 overlapping windows, but with lower memory consumption

	Since inference uses 64 context tokens as T5-Small's encoder input and its right-shited, padded version (pad token ID at the start) as its decoder input,
	training (fine-tuning) should also use the same encoder and decoder contexts, but compute loss only at the last decoder position, predicting exactly one next-token per sliding window
	"""
	def __init__(self, tokens, context_window = None, pad_token_id = None):
		super().__init__()
		self.tokens = torch.tensor(tokens, dtype = torch.long) # A tensorized sequence of 1D token IDs from the entire training corpus
		self.context_window = context_window # No. of tokens used as the conditioning context, e.g., 64
		self.pad_token_id = pad_token_id # Token ID used to pad T5-Small's right-shifted decoder context as the starting token.

		if self.context_window <= 0:
			raise ValueError('Context window must be a positive integer.')
		if self.tokens.numel() <= self.context_window:
			raise ValueError(f'Not enough tokens {self.tokens.numel()} for context window = {self.context_window}.')

	def __len__(self):
		return self.tokens.numel() - self.context_window # No. of training samples/windows

	def __getitem__(self, idx):
		if idx < 0 or idx >= len(self):
			raise IndexError(f'Index {idx} out of range for dataset of length {len(self)}.')

		# Construct the encoder contexts for T5-Small and the corresponding attention masks
		encoder_input_ids = self.tokens[idx:idx + self.context_window]
		encoder_attention_masks = torch.ones(self.context_window, dtype = torch.long)

		# Construct the right-shifted decoder contexts for T5-Small with a pad token ID at the start and the corresponding attention masks
		decoder_input_ids = torch.empty(self.context_window, dtype = torch.long)
		decoder_input_ids[0] = self.pad_token_id
		decoder_input_ids[1:] = encoder_input_ids[:-1]
		decoder_attention_masks = torch.ones(self.context_window, dtype = torch.long)

		# Target next token immediately after the context window
		next_token = self.tokens[idx + self.context_window]

		# Next-token labels with masking applied to the preceding token positions as we only supervise the last position to match the inference objective
		labels = torch.full((self.context_window,), -100, dtype = torch.long)
		labels[-1] = next_token

		return encoder_input_ids, encoder_attention_masks, decoder_input_ids, decoder_attention_masks, labels

# Fine-tuning T5-Small for next-token predictions
def train_t5(dataloader, train_sampler, device, rank, local_rank, epochs = 10, lr = 1e-5, max_norm = 1.0):
	t5model = CausalT5(device = device)
	t5model = DDP(t5model, device_ids = [local_rank])
	optimizer = AdamW(t5model.parameters(), lr = lr, weight_decay = 0.01)
	scaler = GradScaler()

	t5model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		total_steps = 0.0
		train_sampler.set_epoch(epoch)
		for step, (encoder_input_ids, encoder_attention_masks, decoder_input_ids, decoder_attention_masks, labels) in enumerate(dataloader):
			optimizer.zero_grad()
			encoder_input_ids, encoder_attention_masks, decoder_input_ids, decoder_attention_masks, labels = [x.to(device, non_blocking = False) for x in (encoder_input_ids, encoder_attention_masks, decoder_input_ids, decoder_attention_masks, labels)]
			with autocast('cuda'):
				outputs = t5model(encoder_input_ids = encoder_input_ids, encoder_attention_masks = encoder_attention_masks, decoder_input_ids = decoder_input_ids, decoder_attention_masks = decoder_attention_masks, labels = labels)
				loss = outputs.loss
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			clip_grad_norm_(t5model.parameters(), max_norm)
			scaler.step(optimizer)
			scaler.update()
			total_loss += loss.item()
			total_steps += 1.0
		# Aggregrate loss values and total steps from all GPUs, convert them to scalar tensors, keep them on same GPU device, and then stack them together
		metrics = torch.stack([torch.tensor(total_loss, device = device, dtype = torch.float32), torch.tensor(total_steps, device = device, dtype = torch.float32)])
		dist.all_reduce(metrics, op = dist.ReduceOp.SUM)
		global_total_loss, global_total_steps = metrics.tolist()
		avg_loss = global_total_loss / global_total_steps
		if rank == 0:
			print(f'[Epoch: {epoch + 1} / {epochs}], average loss: {avg_loss:.4f}', flush = True)
	return t5model

# Main worker function
def main_worker(rank, world_size, local_rank, epochs = 10, batch_size = 128, lr = 1e-5, max_norm = 1.0):
	distributed_config(rank, world_size, local_rank)
	device = torch.device('cuda', local_rank)

	with open('wiki.train.txt', 'r', encoding='utf-8') as f:
		lines = []
		for i, line in enumerate(f):
			if i >= 50000:
				break
			lines.append(line.strip())

	text = ' '.join(lines)
	tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy = False)
	pad_token_id = tokenizer.pad_token_id # Pad token ID is now explicitly defined in main worker before encoding the text lines into tokens
	tokens = tokenizer.encode(text, add_special_tokens = True)

	train_dataset = TextDataset(tokens, context_window = 64, pad_token_id = pad_token_id)
	train_sampler = DistributedSampler(train_dataset, num_replicas = world_size, rank = rank, shuffle = True)
	dataloader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, num_workers = 0, pin_memory = False, drop_last = False)

	start_time = time.time()
	t5model = train_t5(dataloader, train_sampler, device, rank, local_rank, epochs, lr, max_norm)
	end_time = time.time()
	total_time = end_time - start_time

	# Get the maximum wall-clock time for the distributed job to end
	time_tensor = torch.tensor([total_time], device = device, dtype = torch.float32)
	dist.all_reduce(time_tensor, op = dist.ReduceOp.MAX)
	wall_clock_time = time_tensor.item()

	if rank == 0:
		print(f'Total wall clock time for fine-tuning T5-Small: {wall_clock_time:.4f} seconds', flush = True)
		save_model(t5model, 'T5-Small_ddp_trained_model_10GPUs')
	clean_process()

# Saving the model
def save_model(t5model, save_path):
	os.makedirs(save_path, exist_ok = True)
	model_to_save = t5model.module if hasattr(t5model, 'module') else t5model
	torch.save(model_to_save.state_dict(), f'{save_path}/pytorch_model_T5-Small_10GPUs.bin')
	print(f'Fine-tuned T5-Small checkpoints saved to {save_path}')

if __name__ == '__main__':
	# SLURM environment variables
	rank = int(os.environ.get('SLURM_PROCID'))
	world_size = int(os.environ.get('SLURM_NTASKS'))
	local_rank = int(os.environ.get('SLURM_LOCALID'))
	main_worker(rank = rank, world_size = world_size, local_rank = local_rank, epochs = 10, batch_size = 128, lr = 1e-5, max_norm = 1.0)
