# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

# Importing necessary PyTorch, transformer, and other modules
import os
import re
import torch
import time
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch.optim import AdamW
import torch.distributed as dist
from torch import nn
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from types import SimpleNamespace

# Distributed configuration
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

# RoBERTa neural network for causal next-token predictions
class CausalRoBERTa(nn.Module):
	def __init__(self, device = None, model_name = 'roberta-base'):
		super().__init__()
		self.device = device
		config = RobertaConfig.from_pretrained(model_name)
		config.is_decoder = True # Needed for causal masking behavior
		config.use_pooler = False
		self.roberta = RobertaModel.from_pretrained(model_name, config = config, add_pooling_layer = False)
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
		self.to(self.device)

	def forward(self, encoder_input_ids = None, encoder_attention_masks = None, labels = None):
		outputs = self.roberta(input_ids = encoder_input_ids, attention_mask = encoder_attention_masks)
		logits = self.lm_head(outputs.last_hidden_state)

		if labels is None:
			return logits

		# Labels are masked with -100 everywhere, except the last position
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index = -100)

		return SimpleNamespace(loss = loss, logits = logits)

# Data loading and preprocessing
class TextDataset(Dataset):
	"""
	TextDataset() class for next-token prediction training with a fixed sliding window.
	For each token position in this class, we consider an overlapping context window of 64 tokens, where:
		- context = tokens[idx:idx + context_window]
		- ground truth token (target) = tokens[idx + context_window]
	Windowed training samples are no longer precomputed and stored using Python lists as it can make the dataset construction computationally expensive.
	Instead, the original token sequence is stored once and each window is created only when DataLoader asks for the token sample at position 'idx'.
	This still preserves the same stride-1 overlapping windows, but with lower memory consumption.

	Since inference uses 64 context tokens as RoBERTa's encoder input (no pad token handling required),
	training (fine-tuning) should also use the same encoder contexts, but compute loss only at the last decoder position, predicting exactly one next-token per sliding window.
	"""
	def __init__(self, tokens, context_window = 64):
		super().__init__()
		self.tokens = torch.tensor(tokens, dtype = torch.long)
		self.context_window = context_window

		if self.context_window <= 0:
			raise ValueError('Context window must be a positive integer.')
		if self.tokens.numel() <= self.context_window:
			raise ValueError(f'Not enough tokens {self.tokens.numel()} for context window = {self.context_window}.')

	def __len__(self):
		return self.tokens.numel() - self.context_window # No. of training samples

	def __getitem__(self, idx):
		if idx < 0 or idx >= len(self):
			raise IndexError(f'Index {idx} out of range for dataset of length {len(self)}.')

		# Construct input IDs (contexts) and their corresponding attention masks for BERT's encoder
		encoder_input_ids = self.tokens[idx:idx + self.context_window]
		encoder_attention_masks = torch.ones(self.context_window, dtype = torch.long)

		# Target next token immediately after the context window
		next_token = self.tokens[idx + self.context_window]

		# Next-token labels with masking applied to the preceding token positions as we only supervise the last position to match the inference objective
		labels = torch.full((self.context_window,), -100, dtype = torch.long)
		labels[-1] = next_token

		return encoder_input_ids, encoder_attention_masks, labels

# RoBERTa fine-tuning for next-token predictions
def train_roberta(dataloader, train_sampler, device, rank, local_rank, epochs = 10, lr = 1e-5, max_norm = 1.0):
	model = CausalRoBERTa(device = device)
	model = DDP(model, device_ids = [local_rank])
	optimizer = AdamW(model.parameters(), lr = lr, weight_decay = 0.01)
	scaler = GradScaler()

	model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		train_sampler.set_epoch(epoch)
		for step, (encoder_input_ids, encoder_attention_masks, labels) in enumerate(dataloader):
			optimizer.zero_grad()
			encoder_input_ids, encoder_attention_masks, labels = [x.to(device, non_blocking = False) for x in (encoder_input_ids, encoder_attention_masks, labels)]
			with autocast('cuda'):
				outputs = model(encoder_input_ids = encoder_input_ids, encoder_attention_masks = encoder_attention_masks, labels = labels)
				loss = outputs.loss
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			clip_grad_norm_(model.parameters(), max_norm)
			scaler.step(optimizer)
			scaler.update()
			total_loss += loss.item()
		avg_loss = total_loss / len(dataloader)
		if rank == 0:
			print(f'[Epoch: {epoch + 1} / {epochs}], average loss: {avg_loss:.4f}', flush = True)
	return model

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
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
	tokens = tokenizer.encode(text, add_special_tokens = True)

	train_dataset = TextDataset(tokens, context_window = 64)
	train_sampler = DistributedSampler(train_dataset, num_replicas = world_size, rank = rank, shuffle = True)
	dataloader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, num_workers = 0, pin_memory = False)

	start_time = time.time()
	model = train_roberta(dataloader, train_sampler, device, rank, local_rank, epochs, lr, max_norm)
	end_time = time.time()
	total_time = end_time - start_time

	if rank == 0:
		print(f'Total wall clock time for fine-tuning RoBERTa: {total_time:.4f} seconds', flush = True)
		save_model(model, 'roberta_ddp_trained_model_1GPU')
	clean_process()

# Saving the model:
def save_model(model, save_path):
	os.makedirs(save_path, exist_ok = True)
	model_to_save = model.module if hasattr(model, 'module') else model
	torch.save(model_to_save.state_dict(), f'{save_path}/pytorch_model_RoBERTa_1GPU.bin')
	print(f'Fine-tuned RoBERTa checkpoints saved to {save_path}')

if __name__ == '__main__':
	# SLURM environment variables
	rank = int(os.environ.get('SLURM_PROCID'))
	world_size = int(os.environ.get('SLURM_NTASKS'))
	local_rank = int(os.environ.get('SLURM_LOCALID'))
	main_worker(rank = rank, world_size = world_size, local_rank = local_rank, epochs = 10, batch_size = 128, lr = 1e-5, max_norm = 1.0)
