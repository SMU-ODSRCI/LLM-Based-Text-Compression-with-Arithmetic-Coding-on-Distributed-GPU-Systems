# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

import torch
import math, time, io
from transformers import T5Tokenizer
from T5_ddp_training_1GPU import CausalT5

# Enable/disable TF32 on Ampere (A100) GPUs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Timing flags
INFERENCE_TIME_ENC = True
INFERENCE_TIME_DEC = True
CDF_TIME_ENC = True
CDF_TIME_DEC = True
CDF_TO_CPU_ENC_TIME = True
CDF_TO_CPU_DEC_TIME = True
TRUE_TOKEN_TO_CPU_ENC_TIME = True
TRUE_TOKEN_TO_CPU_DEC_TIME = True
ALS_TO_CPU_ENC_TIME = True
ALS_TO_CPU_DEC_TIME = True

# Load the fine-tuned CausalT5 model checkpoints
def load_CausalT5(model_name = 't5-small', device = None):
	"""
	Loads the fine-tuned CausalT5 weights and biases from a .bin checkpoint
	"""

	t5small_model = CausalT5(device = device, model_name = model_name)
	t5small_state = torch.load('T5-Small_ddp_trained_model_1GPU/pytorch_model_T5-Small_1GPU.bin', map_location = device)
	t5small_model.load_state_dict(t5small_state)
	t5small_model.to(device)
	t5small_model.eval()

	return t5small_model

# Extract a range of lines containing the test (unseen) data from enwiki9
def extract_enwiki9_lines(data_path = None, start = 50000, end = 100001):
	"""
	Reads and extracts the test (unseen) data from enwiki9.
	To test with a small sequence, we extract lines 50,001 to 100,000.
	"""

	lines = []
	with open(data_path, 'r', encoding = 'utf-8') as f:
		for idx, line in enumerate(f):
			if idx < start:
				continue
			if idx >= end:
				break
			lines.append(line)
	text = ' '.join(lines)

	return text

# Compute next-token probabilities using t5small_model
def predict_next_token_probs(t5small_model, batch_encoder_context = None, batch_encoder_attention_mask = None, batch_decoder_context = None, batch_decoder_attention_mask = None, temperature = 1.0):
	"""
	Computes next-token probabilities from the sequence using one a set of batched contexts at a time.
	Includes temperature scaling where:
		- temperature > 1.0 --> softer probability distribution
		- temperature < 1.0 --> sharper probability distribution
		- temperature = 1.0 --> T5-Small's true probability distribution as is.
	"""

	with torch.inference_mode():
		outputs = t5small_model(encoder_input_ids = batch_encoder_context, encoder_attention_masks = batch_encoder_attention_mask, decoder_input_ids = batch_decoder_context, decoder_attention_masks = batch_decoder_attention_mask) # Forward pass via CausalT5
		step_logits = outputs.logits[:, -2, :] / temperature # Since CausalT5 predicts the next token autoregressively, we use the last context position
		prob_dist = torch.softmax(step_logits, dim = -1) # Converts logits into probabilities

	return prob_dist.to(dtype = torch.float32)

# Build integer CDF from fixed probabilities
def probs_to_int_cdf_batched(prob_dist_batched = None, total = None):
	"""
	This function converts batched next-token probabilities into batched integer CDF models
	"""
	# ======================================================
	# Normalization using tensor ops
	# ======================================================
	# Old normalization: probs = prob_dist / prob_dist.sum()
	# New (batched) normalization using sum(dim = -1, keepdim = True) that computes row-wise sums for all batched probability streams at once
	probs = prob_dist_batched

	# ======================================================
	# Get the vocabulary size using .size(-1)
	# ======================================================
	# Old way to get vocabulary size: vocab_size = probs.numel()
	# New batched version uses probs.size(-1)
	vocab_size = probs.size(-1)

	# ======================================================
	# Integer frequency computation using tensor ops
	# ======================================================
	# When total is >= vocab_size, reserve one count for every symbol, preventing zero-frequency symbols and eliminating underflow (-diff) issues
	# Distribute the remaining budget proportionally using floor, where floor ensures that we never exceed total
	# Final frequency is at least 1 for every symbol.
	extra = (probs * (total - probs.size(-1))).to(dtype = torch.int32)
	freqs = extra + 1

	# ======================================================
	# Diff computation using tensor ops
	# ======================================================
	# Old diff calculation: diff = total - int(freqs.sum().item())
	# New diff calculation using sum(dim = -1) gives one diff per batched stream
	diff = (total - freqs.sum(dim = -1)).to(dtype = torch.int32)

	# ======================================================
	# Topk selection using tensor ops
	# ======================================================
	# One single topk selection is performed over the entire batched (batch_size, vocab_size) tensor
	# diff.max() gives the symbols with the largest fractional portion
	# Compute diff.max() once per batch, move to CPU with non-blocking, and convert to Python int for usage in topk selection and mask creation
	diff_max = int(diff.max().item()) # Implement .item() for correct integer conversion
	if diff_max > 0: # Removed .item() from this line
		eps = torch.arange(vocab_size, device = probs.device, dtype = torch.float32) * 1e-12 # Deterministic tie-breaker for near-identical probabilities in topk
		probs_for_topk = probs + eps

		# Batched topk selection with a single call that handles all batched streams at once
		top_idx = torch.topk(probs_for_topk, k = diff_max, largest = True).indices
		add_mask = (torch.arange(diff_max, device = probs.device).unsqueeze(0) < diff.unsqueeze(1)).to(dtype = torch.int32) # add_mask builds a boolean mask denoting each batched stream using the first diff[i] indices
		freqs = freqs + torch.zeros_like(freqs).scatter_add(dim = -1, index = top_idx, src = add_mask) # scatter_add applies the leftover +1 counts to the selected symbols from all batched streams at once

	# Build tensorized integer CDF
	cdf_vec = torch.empty((probs.size(0), vocab_size + 1), device = probs.device, dtype = torch.int32)
	cdf_vec[:, 0] = 0
	cdf_vec[:, 1:] = torch.cumsum(freqs, dim = -1) # Now torch.cumsum(dim = -1) builds CDFs for all batched streams at once

	# Sanity check for to ensure that CDF is valid for every batch of active active streams
	# assert int(cdf_vec[:, -1].min().item()) == total
	# assert int(cdf_vec[:, -1].max().item()) == total

	return cdf_vec, freqs

# Bit-oriented I/O streams
# The following classes below accept a stream of bits that can be read and written to
# Bits come from an underlying byte stream, read in big endian order (MSB --> LSB within each byte)
# Total no. of bits is always a multiple of 8
class BitInputStream(object):
    # Construct a bit input stream based on the given byte input stream
	def __init__(self, input):
		self.input = input # Underlying byte stream to read from
		self.current_byte = 0 # Most recently read byte within the range [0, 255] or -1 when EOF is reached
		self.remaining_num_bits = 0 # No. of bits left to consume from current byte
		self.bits = 0 # Counts the no. of bits consumed

    # Read a bit from the current stream
    # Returns 0 or 1 if bit is available or -1 if end-of-stream is reached, which always occurs on a byte boundary
	def read_bit(self):
                # If EOF hit previously, stay at EOF
		if self.current_byte == -1:
			return -1

        # If no bits remaining in buffer, read the next byte from the underlying stream
		if self.remaining_num_bits == 0:
			byte = self.input.read(1) # Read 1 byte from the byte stream
			if len(byte) == 0: # No bytes returned --> EOF
				self.current_byte = -1
				return -1
			self.current_byte = byte[0]
			self.remaining_num_bits = 8 # 8 unread bits in current_byte

        # Consume one bit in big endian order, where we take the next most-significant remaining bit
		assert self.remaining_num_bits > 0
		self.remaining_num_bits -= 1
		bit = (self.current_byte >> self.remaining_num_bits) & 1
		self.bits += 1
		return bit

    # Reads a bit from the current stream
    # Returns 0 or 1 if bit is available or raises an EOFError if end-of-stream is reached
	def read_no_eof(self):
		result = self.read_bit()
		if result != -1:
			return result
		raise EOFError()

    # Close the current and underlying input stream
	def close_stream(self):
		self.input.close()
		self.current_byte = -1
		self.remaining_num_bits = 0

# Accepts a stream, where bit can be written to
# At close(), the stream is padded with 0-bits up to a multiple of 8
# Bits are written in big endian order (MSB --> LSB within each byte)
class BitOutputStream(object):
    # Construct a bit output stream based on the given byte output stream
	def __init__(self, output):
		self.output = output # Underlying byte stream to write to
		self.current_byte = 0 # Accumulator for the current byte being built
		self.filled_num_bits = 0 # No. of bits written into current_byte

    # Writes a bit to the stream, where the given bit must be 0 or 1
	def write_bit(self, bit):
        # Check for bit correctness
		if bit not in (0, 1):
			raise ValueError('Argument must be 0 or 1')

        # Shift left by 1 to make room for the new bit
		self.current_byte = (self.current_byte << 1) | bit
		self.filled_num_bits += 1

        # If accumulated 8 bits, then flush as one byte to the underlying stream
		if self.filled_num_bits == 8:
			self.output.write(bytes((self.current_byte,))) # Writes a single byte
			self.current_byte = 0
			self.filled_num_bits = 0

	# Flush the remaining bits with 0-padding to reach the next byte boundary, then close the stream
	def close_stream(self):
        # Pad with zeros until byte is packed
        # Decode stops after N tokens so byte padding does not matter
		while self.filled_num_bits != 0:
			self.write_bit(0)
		if not isinstance(self.output, io.BytesIO):
			self.output.close()

# Arithmetic Encoder/Decoder
class ArithmeticCoder:
	def __init__(self, mode = None, num_state_bits = None, bitstream = None):
		assert mode in ('encode', 'decode')
		self.mode = mode
		self.n = num_state_bits

		self.full_range = 1 << self.n
		self.half = self.full_range >> 1
		self.quarter = self.half >> 1
		self.three_quarter = self.quarter * 3

		self.low = 0
		self.high = self.full_range - 1

		if self.mode == 'encode':
			self.pending = 0
            # New byte-packed output bitstream using BitOutputStream
			self.byte_out = io.BytesIO() # In-memory byte stream
			self.bit_out = BitOutputStream(self.byte_out) # Bit-level writer that packs bits into bytes
		else:
            # New byte-packed input bitstream using BitInputStream
			if bitstream is None:
				bitstream = b'' # Default empty bytes
			self.byte_in = io.BytesIO(bitstream) # Wrap bytes in a byte-stream interface so that BitInputStream can read bits
			self.bit_in = BitInputStream(self.byte_in)
			self.code = 0
			for _ in range(self.n):
				bit = self.bit_in.read_bit()
				if bit == -1:
					bit = 0
				self.code = (self.code << 1) | bit

	def output_bits(self, bit = None):
		# Write bits via BitOutputStream
		self.bit_out.write_bit(bit)
		while self.pending > 0:
			self.bit_out.write_bit(1 - bit)
			self.pending -= 1

	def encode_symbol(self, cdf, symbol = None):
		assert self.mode == 'encode'
		total = int(cdf[-1]) # Removed .item() call
		if not (0 <= symbol < cdf.numel() - 1):
			raise ValueError('symbol out of range')

		range_width = self.high - self.low + 1
		symbol_low = int(cdf[symbol]) # Removed .item() call
		symbol_high = int(cdf[symbol + 1]) # Removed .item() call
		self.high = self.low + (range_width * symbol_high // total) - 1
		self.low = self.low + (range_width * symbol_low // total)

		# Renormalize to output a real bitstream
		while True:
			if self.high < self.half:
				self.output_bits(0)
			elif self.low >= self.half:
				self.output_bits(1)
				self.low -= self.half
				self.high -= self.half
			elif self.low >= self.quarter and self.high < self.three_quarter:
				self.pending += 1
				self.low -= self.quarter
				self.high -= self.quarter
			else:
				break

			self.low <<= 1
			self.high = (self.high << 1) + 1
			self.low &= (self.full_range - 1)
			self.high &= (self.full_range - 1)

	def finish(self):
		assert self.mode == 'encode'

        # Flush termination bits
		self.pending += 1
		if self.low < self.quarter:
			self.output_bits(0)
		else:
			self.output_bits(1)

        # Close BitOutputStream and return raw bytes from the underlying byte stream
		self.bit_out.close_stream()
		return self.byte_out.getvalue() # Packed byte stream

	def decode_symbol(self, cdf):
		assert self.mode == 'decode'
		total = int(cdf[-1])
		range_width = self.high - self.low + 1

		offset = self.code - self.low
		encoded_value = ((offset + 1) * total - 1) // range_width

		# Find symbol such that cdf[symbol] <= encoded value < cdf[symbol + 1)
		symbol = int(torch.searchsorted(cdf, encoded_value, right = True) - 1) # Removed .item() call
		symbol = max(0, min(symbol, cdf.numel() - 2))

		symbol_low = int(cdf[symbol]) # Removed .item() call
		symbol_high = int(cdf[symbol + 1]) # Removed .item() call
		self.high = self.low + (range_width * symbol_high // total) - 1
		self.low = self.low + (range_width * symbol_low // total)

		# Renormalize like in encode_symbol
		while True:
			if self.high < self.half:
				pass
			elif self.low >= self.half:
				self.low -= self.half
				self.high -= self.half
				self.code -= self.half
			elif self.low >= self.quarter and self.high < self.three_quarter:
				self.low -= self.quarter
				self.high -= self.quarter
				self.code -= self.quarter
			else:
				break

			self.low <<= 1
			self.high = (self.high << 1) + 1

			# Read next bit from BitInputStream
			bit = self.bit_in.read_bit()
			if bit == -1:
				bit = 0
			self.code = (self.code << 1) | bit

			self.low &= (self.full_range - 1)
			self.high &= (self.full_range - 1)
			self.code &= (self.full_range - 1)

		return symbol

def evaluate_ac_t5(device = None, data_path = None, context_window = 64, temperature = 1.0):
	"""
	This function:
		- Builds a token stream from enwiki9
		- Constructs contexts and attention masks for T5-Small's encoder and decoder
		- Encodes tokens at each step using a set of 64 context tokens and predicted next-token probabilities
		- Decodes tokens using the same sliding window strategy, probability distribution, and integer CDF model as in encoding
		- Verifies perfect equality after decoding
	"""

 	# Initialize GPU timing accumulators
	inference_time_enc_ms = 0.0
	inference_time_dec_ms = 0.0
	cdf_time_enc_ms = 0.0
	cdf_time_dec_ms = 0.0
	cdf_to_cpu_enc_time_ms = 0.0
	cdf_to_cpu_dec_time_ms = 0.0
	true_token_to_cpu_enc_time_ms = 0.0
	true_token_to_cpu_dec_time_ms = 0.0
	als_to_cpu_enc_time_ms = 0.0
	als_to_cpu_dec_time_ms = 0.0

    # CUDA event buffer
	event_buff = []
	flush_every = 256

    # Initialize CPU timing accumulators
	ac_enc_time = 0.0
	ac_dec_time = 0.0

	def event_pair():
		return (torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True))

	def flush_events():
		nonlocal inference_time_enc_ms, inference_time_dec_ms
		nonlocal cdf_time_enc_ms, cdf_time_dec_ms
		nonlocal cdf_to_cpu_enc_time_ms, cdf_to_cpu_dec_time_ms
		nonlocal true_token_to_cpu_enc_time_ms, true_token_to_cpu_dec_time_ms
		nonlocal als_to_cpu_enc_time_ms, als_to_cpu_dec_time_ms

		if not event_buff:
			return

		for tag, start, end in event_buff:
			gpu_time = start.elapsed_time(end)
			if tag == 'inf_enc': inference_time_enc_ms += gpu_time
			elif tag == 'inf_dec': inference_time_dec_ms += gpu_time
			elif tag == 'cdf_enc': cdf_time_enc_ms += gpu_time
			elif tag == 'cdf_dec': cdf_time_dec_ms += gpu_time
			elif tag == 'cdf2cpu_enc': cdf_to_cpu_enc_time_ms += gpu_time
			elif tag == 'cdf2cpu_dec': cdf_to_cpu_dec_time_ms += gpu_time
			elif tag == 'tok2cpu_enc': true_token_to_cpu_enc_time_ms += gpu_time
			elif tag == 'tok2cpu_dec': true_token_to_cpu_dec_time_ms += gpu_time
			elif tag == 'als2cpu_enc': als_to_cpu_enc_time_ms += gpu_time
			elif tag == 'als2cpu_dec': als_to_cpu_dec_time_ms += gpu_time
		event_buff.clear()

	# Load CausalT5 model
	t5small_model = load_CausalT5(model_name = 't5-small', device = device)

	# Load the test (unseen) text segment and calculate the original file size
	text = extract_enwiki9_lines(data_path = data_path, start = 50000, end = 100001)
	num_chars = len(text)
	original_bytes = len(text.encode('utf-8'))
	original_MB = original_bytes / 1024 / 1024

	# Tokenize the text and load tokenized data using dataloader with a batch size of 128
	tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy = False)
	tokens = tokenizer.encode(text, add_special_tokens = True)
	pad_token_id = tokenizer.pad_token_id

	# Build independent token streams (contiguous chunks) of batch size B
	def split_into_B_batches(tokens, B):
		n = len(tokens)
		base = n // B
		remaining = n % B
		batches = []
		start = 0
		for i in range(B):
			size = base + (1 if i < remaining else 0)
			end = start + size
			batches.append(tokens[start:end])
			start = end
		return batches

    # Reduce batch size until all streams have enough length
	B_target = 128 # Since we run this case using 1-GPU, setting B_target = 128 guarantees 128 streams as it does not get further split across GPU ranks
	B = min(B_target, max(1, len(tokens) // (context_window + 1)))
	streams = split_into_B_batches(tokens, B)
	while B > 1 and any(len(stream) <= context_window for stream in streams):
		B -= 1
		streams = split_into_B_batches(tokens, B)

	# Tensorize streams once outside the hot loop so that true token can be extracted via tensor indexing instead of Python list indexing
	length = torch.tensor([len(stream) for stream in streams], device = device, dtype = torch.long)
	length_max = int(length.max().item()) # One time control variable and therefore, using .item() here is required and safe
	streams_tensor = torch.full((B, length_max), pad_token_id, device = device, dtype = torch.long)
	for idx, stream in enumerate(streams):
		streams_tensor[idx, :len(stream)] = torch.tensor(stream, device = device, dtype = torch.long)
	stream_ids_tensor = torch.arange(B, device = device, dtype = torch.long) # Stream indices to be used inside the hot loop

    # Initialize encoder performance metrics
	total_tokens = 0
	total_log_likelihood = 0.0
	total_kl_divergence = 0.0

    # Batched encoding across independent token streams
	encoders = [ArithmeticCoder(mode = 'encode', num_state_bits = 32) for _ in range(B)]
	encoder_context_all = torch.tensor([stream[:context_window] for stream in streams], device = device, dtype = torch.long)

    # Next position to encode for each stream
	max_steps = max(len(stream) - context_window for stream in streams)
	torch.cuda.synchronize() # CUDA time synchronization added at the beginning of encoding time boundary
	encoder_start = time.time()
	for step in range(max_steps):
		any_active_stream = False
		position = context_window + step

		for batch in range(0, stream_ids_tensor.numel(), B_target):
			# Removed .tolist() conversion as it creates a Python list and enforces Python list indexing in the hot loop
			# Keep batch_stream_id as a tensor
			batch_stream_id = stream_ids_tensor[batch:batch + B_target]

            # A stream is active at this step if it has token at index position (context_window + step)
			# Tensorize active local stream selection by removing Python list indexing and replacing it with pure tensor operations
			active_mask = position < length[batch_stream_id]
			active_stream = batch_stream_id[active_mask]
			if active_stream.numel() == 0:
				continue
			any_active_stream = True

			context_batch = encoder_context_all[active_stream]

            # Build T5 encoder/decoder contexts and attention masks from previously encoded tokens
			encoder_context = context_batch
			encoder_attention_mask = torch.ones_like(encoder_context, device = device, dtype = torch.long)
			decoder_context = torch.empty_like(encoder_context, device = device, dtype = torch.long)
			decoder_context[:, 0] = pad_token_id
			decoder_context[:, 1:] = encoder_context[:, :-1] # Restored the exact right-shift rule from training to build decoder contexts
			decoder_attention_mask = torch.ones_like(decoder_context, device = device, dtype = torch.long)

			if INFERENCE_TIME_ENC:
				start, end = event_pair()
				start.record()

            # Batched next-token probabilities across independent streams
			prob_dist_batched = predict_next_token_probs(t5small_model, batch_encoder_context = encoder_context, batch_encoder_attention_mask = encoder_attention_mask, batch_decoder_context = decoder_context, batch_decoder_attention_mask = decoder_attention_mask, temperature = temperature)

			if INFERENCE_TIME_ENC:
				end.record()
				event_buff.append(('inf_enc', start, end))

			if CDF_TIME_ENC:
				start, end = event_pair()
				start.record()

			# Build all CDFs for the active batch at once and move to CPU once per batch
			cdf_batched, _  = probs_to_int_cdf_batched(prob_dist_batched, total = 1 << 18)

			if CDF_TIME_ENC:
				end.record()
				event_buff.append(('cdf_enc', start, end))

			if CDF_TO_CPU_ENC_TIME:
				start, end = event_pair()
				start.record()

			cdf_batched_cpu = cdf_batched.to(device = 'cpu', non_blocking = False)

			if CDF_TO_CPU_ENC_TIME:
				end.record()
				event_buff.append(('cdf2cpu_enc', start, end))

            # True next token for each batch of active streams
			# True token extraction is now tensorized with no Python list indexing and moved to CPU once per batch
			true_token = streams_tensor[active_stream, position]

			if TRUE_TOKEN_TO_CPU_ENC_TIME:
				start, end = event_pair()
				start.record()

			true_token_cpu = true_token.to(device = 'cpu', non_blocking = False) # Avoids .item() call inside the inner-most loop

			if TRUE_TOKEN_TO_CPU_ENC_TIME:
				end.record()
				event_buff.append(('tok2cpu_enc', start, end))

            # Compute NLL and KL for each batch of active streams
			batch_idx = torch.arange(active_stream.numel(), device = device, dtype = torch.long)
			prob_true_batched = prob_dist_batched[batch_idx, true_token]
			total_log_likelihood += (-torch.log(prob_true_batched)).sum()
			mask = prob_dist_batched > 0
			log_vocab_size = torch.log(torch.tensor(prob_dist_batched.size(-1), device = device, dtype = torch.float32))
			total_kl_divergence += (prob_dist_batched[mask] * (torch.log(prob_dist_batched[mask]) + log_vocab_size)).sum()

			# Total no. of tokens across all batches
			total_tokens += active_stream.numel()

			# Encode one token per active local stream, then advance that stream by 1
			# Active and local stream selection is now tensorized, except the AC calls

			if ALS_TO_CPU_ENC_TIME:
				start, end = event_pair()
				start.record()

			active_stream_cpu = active_stream.to(device = 'cpu', non_blocking = False) # Move to CPU once per batch of local active stream IDs

			if ALS_TO_CPU_ENC_TIME:
				end.record()
				event_buff.append(('als2cpu_enc', start, end))

			for idx, stream_id in enumerate(active_stream_cpu):
				stream_id = int(stream_id) # Index Python list with Python scalar instead of tensor scalar
				cdf = cdf_batched_cpu[idx]

				token = int(true_token_cpu[idx])

				t_0 = time.perf_counter()
				encoders[stream_id].encode_symbol(cdf, token)
				ac_enc_time += time.perf_counter() - t_0

                # Update rolling encoder context with the previous encoded token
				encoder_context_all[stream_id, :-1] = encoder_context_all[stream_id, 1:].clone()
				encoder_context_all[stream_id, -1] = token

			if len(event_buff) >= flush_every:
				flush_events()

		if not any_active_stream:
			break

    # Finalize per-stream bitstreams
	bitstreams = [encoders[idx].finish() for idx in range(B)]
	torch.cuda.synchronize() # CUDA time synchronization added at the end of encoding time boundary
	encoder_end = time.time()
	flush_events()

	# Convert GPU encoding timings to seconds
	inference_time_enc = inference_time_enc_ms / 1000.00
	cdf_time_enc = cdf_time_enc_ms / 1000.00
	cdf_to_cpu_enc_time = cdf_to_cpu_enc_time_ms / 1000.00
	true_token_to_cpu_enc_time = true_token_to_cpu_enc_time_ms / 1000.00
	als_to_cpu_enc_time = als_to_cpu_enc_time_ms / 1000.00

    # Calculate total compressed bits
	total_compressed_bits = sum(len(bitstream) for bitstream in bitstreams) * 8

    # Batched decoding across independent streams
	total_matches = 0
	decoded_tokens = 0
	decoders = [ArithmeticCoder(mode = 'decode', num_state_bits = 32, bitstream = bitstreams[idx]) for idx in range(B)]
	decoder_context_all = torch.tensor([stream[:context_window] for stream in streams], device = device, dtype = torch.long)

    # Next position to decode for each stream
	torch.cuda.synchronize() # CUDA time synchronization added at the beginning of decoding time boundary
	decoder_start = time.time()
	for step in range(max_steps):
		any_active_stream = False
		position = context_window + step

		for batch in range(0, stream_ids_tensor.numel(), B_target):
 			# Removed .tolist() conversion as it creates a Python list and enforces Python list indexing in the hot loop
			# Keep batch_stream_id as a tensor
			batch_stream_id = stream_ids_tensor[batch:batch + B_target]

            # A stream is active at this step if it has token at index position (context_window + step)
            # Tensorize active local stream selection by removing Python list indexing and replacing it with pure tensor operations
			active_mask = position < length[batch_stream_id]
			active_stream = batch_stream_id[active_mask]
			if active_stream.numel() == 0:
				continue
			any_active_stream = True

			context_batch = decoder_context_all[active_stream]
            # Build T5 encoder/decoder contexts and attention masks from decoded tokens
			encoder_context = context_batch
			encoder_attention_mask = torch.ones_like(encoder_context, device = device, dtype = torch.long)
			decoder_context = torch.empty_like(encoder_context, device = device, dtype = torch.long)
			decoder_context[:, 0] = pad_token_id
			decoder_context[:, 1:] = encoder_context[:, :-1] # Restored the exact right-shift rule from training to build decoder contexts
			decoder_attention_mask = torch.ones_like(decoder_context, device = device, dtype = torch.long)

			if INFERENCE_TIME_DEC:
				start, end = event_pair()
				start.record()

            # Recompute probabilities from the decoded contexts
			prob_dist_batched = predict_next_token_probs(t5small_model, batch_encoder_context = encoder_context, batch_encoder_attention_mask = encoder_attention_mask, batch_decoder_context = decoder_context, batch_decoder_attention_mask = decoder_attention_mask, temperature = temperature)

			if INFERENCE_TIME_DEC:
				end.record()
				event_buff.append(('inf_dec', start, end))

			if CDF_TIME_DEC:
				start, end = event_pair()
				start.record()

			# Build all CDFs for the active batch at once and move to CPU once per batch
			cdf_batched, _  = probs_to_int_cdf_batched(prob_dist_batched, total = 1 << 18)

			if CDF_TIME_DEC:
				end.record()
				event_buff.append(('cdf_dec', start, end))

			if CDF_TO_CPU_DEC_TIME:
				start, end = event_pair()
				start.record()

			cdf_batched_cpu = cdf_batched.to(device = 'cpu', non_blocking = False)

			if CDF_TO_CPU_DEC_TIME:
				end.record()
				event_buff.append(('cdf2cpu_dec', start, end))

 			# True next token for each batch of active streams
            # True token extraction is now tensorized with no Python list indexing and moved to CPU once per batch
			true_token = streams_tensor[active_stream, position]

			if TRUE_TOKEN_TO_CPU_DEC_TIME:
				start, end = event_pair()
				start.record()

			true_token_cpu = true_token.to(device = 'cpu', non_blocking = False) # Avoids .item() call in the inner-most loop

			if TRUE_TOKEN_TO_CPU_DEC_TIME:
				end.record()
				event_buff.append(('tok2cpu_dec', start, end))

			if ALS_TO_CPU_DEC_TIME:
				start, end = event_pair()
				start.record()

			active_stream_cpu = active_stream.to(device = 'cpu', non_blocking = False) # Move to CPU once per batch of local active stream IDs

			if ALS_TO_CPU_DEC_TIME:
				end.record()
				event_buff.append(('als2cpu_dec', start, end))

			for idx, stream_id in enumerate(active_stream_cpu):
				stream_id = int(stream_id) # Index Python list with Python scalar instead of tensor scalar
				cdf = cdf_batched_cpu[idx]

				t_0 = time.perf_counter()
				decoded_token = int(decoders[stream_id].decode_symbol(cdf))
				ac_dec_time += time.perf_counter() - t_0

				actual_token = int(true_token_cpu[idx])

                # Update decoder rolling contexts with the decoded tokens
				decoder_context_all[stream_id, :-1] = decoder_context_all[stream_id, 1:].clone()
				decoder_context_all[stream_id, -1] = decoded_token

				total_matches += int(decoded_token == actual_token)
				decoded_tokens += 1

			if len(event_buff) >= flush_every:
				flush_events()

		if not any_active_stream:
			break

	torch.cuda.synchronize() # CUDA time synchronization added at the end of decoding time boundary
	decoder_end = time.time()
	flush_events()

	# Convert GPU decoding timings to seconds
	inference_time_dec = inference_time_dec_ms / 1000.00
	cdf_time_dec = cdf_time_dec_ms / 1000.00
	cdf_to_cpu_dec_time = cdf_to_cpu_dec_time_ms / 1000.00
	true_token_to_cpu_dec_time = true_token_to_cpu_dec_time_ms / 1000.00
	als_to_cpu_dec_time = als_to_cpu_dec_time_ms / 1000.00

	# Total wall-clock time
	total_time = (encoder_end - encoder_start) + (decoder_end - decoder_start)

	accuracy = total_matches / decoded_tokens
	average_bpc = total_compressed_bits / num_chars
	compressed_MB = total_compressed_bits / 8 / 1024 / 1024
	compression_ratio = original_MB / compressed_MB

	average_cross_entropy = total_log_likelihood / decoded_tokens
	perplexity = math.exp(average_cross_entropy)
	average_bpt = average_cross_entropy / math.log(2)
	average_kl_divergence = total_kl_divergence / decoded_tokens

	# Print performance metrics
	print(f'--- Evaluation Report ---')
	print(f'Reconstruction Accuracy: {accuracy:.4f}', flush = True)
	print(f'Bits per character (BPC): {average_bpc:.4f}', flush = True)
	print(f'Bits per token (BPT): {average_bpt:.4f}', flush = True)
	print(f'Original size: {original_MB:.4f} MB', flush = True)
	print(f'Compressed size: {compressed_MB:.4f} MB', flush = True)
	print(f'Compression ratio: {compression_ratio:.4f}', flush = True)
	print(f'Cross-entropy (nats/token): {average_cross_entropy:.4f}', flush = True)
	print(f'Perplexity: {perplexity:.4f}', flush = True)
	print(f'KL-Divergence: {average_kl_divergence:.4f}', flush = True)
	print(f'Wall clock time for compression and decompression: {total_time:.4f} seconds', flush = True)
	print(f'Inference encoding GPU time: {inference_time_enc:.4f} seconds', flush = True)
	print(f'Inference decoding GPU time: {inference_time_dec:.4f} seconds', flush = True)
	print(f'CDF encoding GPU time: {cdf_time_enc:.4f} seconds', flush = True)
	print(f'CDF decoding GPU time: {cdf_time_dec:.4f} seconds', flush = True)
	print(f'CDF GPU-to-CPU encoding time: {cdf_to_cpu_enc_time:.4f} seconds', flush = True)
	print(f'CDF GPU-to-CPU decoding time: {cdf_to_cpu_dec_time:.4f} seconds', flush = True)
	print(f'True token GPU-to-CPU encoding time: {true_token_to_cpu_enc_time:.4f} seconds', flush = True)
	print(f'True token GPU-to-CPU decoding time: {true_token_to_cpu_dec_time:.4f} seconds', flush = True)
	print(f'Active local stream GPU-to-CPU encoding time: {als_to_cpu_enc_time:.4f} seconds', flush = True)
	print(f'Active local stream GPU-to-CPU decoding time: {als_to_cpu_dec_time:.4f} seconds', flush = True)
	print(f'AC encoding CPU time: {ac_enc_time:.4f} seconds', flush = True)
	print(f'AC decoding CPU time: {ac_dec_time:.4f} seconds', flush = True)

if __name__ == '__main__':
	evaluate_ac_t5(device = 'cuda', data_path = 'wiki.train.txt', context_window = 64, temperature = 1.0)
















