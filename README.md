# LLM-based-Hybrid-Text-Compressors-on-HPC

**Overview**
This work presents the first systems-level characterizations of hybrid text compression pipelines that incorporate transformer-based Large Language Models (LLMs) with Arithmetic Coding (AC) on a modern, state-of-the-art (SOTA) high-performance computing (HPC) platform. Unlike prior research work that primarily focused on either the final compressed sizes of the original files or compression ratio in isolation using resource-constrained hardware, this study investigates how such hybrid compressors behave under distributed (multi-GPU and multi-node) workloads, including their scalability, and resource utilization such as memory consumption during inference. In this framework, LLMs act as probabilistic predictors that generate token-level conditional probability distributions consumed by the AC to produce compressed bitstreams that reach near-entropy limits. While the integration of LLMs with AC improves compression efficiency by leveraging long-range contextual dependencies, it introduces new systems challenges such as high compute intensity, memory usage patterns, and sequential dependencies during inference. Moreover, almost none of the recent existing work on hybrid text compression frameworks report performance metrics, e.g., bits per character (BPC), bits per token (BPT), cross-entropy loss, perplexity, KL-divergence, and reconstruction accuracy that would truly provide a broad overview of their performance on new, unseen data. To address these gaps, this study evaluates four representative transformer-based architectures: 1) BERT, 2) RoBERTa, 3) T5-small, and 4) Llama-3.2-3B with AC, providing the first comprehensive comparison of architecture-driven trade-offs between compression performance, wall-clock times, and distributed workloads.

**Fine-Tuning and Inference across the 4 Models**
The hybrid LLM–AC pipelines across the 4 models consist of two main stages: 1) fine-tuning and 2) inference, both executed across 1, 2, 4, 6, and 8 GPUs on one node and 10, 12, 14, and 16 GPUs on a second node. During fine-tuning, each LLM was trained for causal next-token predictions using the first 50,000 lines of the enwiki9 dataset. Encoder-only models such as BERT and RoBERTa, which are not inherently autoregressive, were fine-tuned using a sliding window approach where a fixed-length context, e.g., 64 tokens is used to predict the next token, allowing them to function as probabilistic predictors. On the other hand, T5-Small is an encoder–decoder model that introduced additional complexity by requiring paired encoder inputs and right-shifted decoder inputs, enabling autoregressive predictions through its decoder while leveraging the encoder for understanding the contextual representations. In contrast, LLaMA-3.2-3B is a decoder-only model that naturally aligned with the fine-tuning objective due to its inherent causal, autoregressive nature. During inference, all models generated token-level probability distributions in an autoregressive manner over the subsequent 50,000 lines (unseen) of the enwiki9 data, which were later converted into integer cumulative distribution functions (CDFs) for AC. Distributed workloads were achieved via data parallelism, where the input token streams were manually sharded across the GPUs and the exact copies of the fine-tuned model checkpoints were loaded in each GPU device. However, the AC part of the inference remained strictly sequential and hence, on the CPU for each token stream processed. Additional optimizations such as disabling TF32 to avoid numerical precision drifts, batched CDF constructions on the GPUs, token streams tensorizations, and careful CPU thread management helped to ensure deterministic executions of the compression pipelines.

**Setup**
Hardware: 
•	All experiments have been conducted on the NVIDIA DGX Ampere 100 (A100) SuperPOD, but they can also be run on any machine with dedicated GPU support
Software:
•	First, create an environment using “environment.yml” and a name of your choice:
conda env create -f environment.yml
conda activate env_name

•	If an environment is already setup, then the required PyTorch, Hugging Face transformers, and CUDA toolkit libraries as well as other dependencies can be installed using “requirements.txt”:
pip install requirements.txt

•	Detailed directory tree:
Main
	environment.yml
	requirements.txt
	ReadMe.md
	BERT Fine-tuning Codes
-	bert_ddp_training_1GPU.py
-	bert_ddp_training_2GPUs.py
-	bert_ddp_training_4GPUs.py
-	bert_ddp_training_6GPUs.py
-	bert_ddp_training_8GPUs.py
-	bert_ddp_training_10GPUs.py
-	bert_ddp_training_12GPUs.py
-	bert_ddp_training_14GPUs.py
-	bert_ddp_training_16GPUs.py
	BERT Fine-tuning SBATCH Script
-	bert_arithmetic_compression_training_1GPU.sbatch
-	bert_arithmetic_compression_training_2GPUs.sbatch
-	bert_arithmetic_compression_training_4GPUs.sbatch
-	bert_arithmetic_compression_training_6GPUs.sbatch
-	bert_arithmetic_compression_training_8GPUs.sbatch
-	bert_arithmetic_compression_training_10GPUs.sbatch
-	bert_arithmetic_compression_training_12GPUs.sbatch
-	bert_arithmetic_compression_training_14GPUs.sbatch
-	bert_arithmetic_compression_training_16GPUs.sbatch
	BERT + AC Inference Codes
-	bert_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	bert_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
	BERT + AC Inference SBATCH Scripts
-	bert_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	bert_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
	RoBERTa Fine-tuning Codes
-	roberta_ddp_training_1GPU.py
-	roberta_ddp_training_2GPUs.py
-	roberta_ddp_training_4GPUs.py
-	roberta_ddp_training_6GPUs.py
-	roberta_ddp_training_8GPUs.py
-	roberta_ddp_training_10GPUs.py
-	roberta_ddp_training_12GPUs.py
-	roberta_ddp_training_14GPUs.py
-	roberta_ddp_training_16GPUs.py
	RoBERTa Fine-tuning SBATCH Scripts
-	roberta_arithmetic_compression_training_1GPU.sbatch
-	roberta_arithmetic_compression_training_2GPUs.sbatch
-	roberta_arithmetic_compression_training_4GPUs.sbatch
-	roberta_arithmetic_compression_training_6GPUs.sbatch
-	roberta_arithmetic_compression_training_8GPUs.sbatch
-	roberta_arithmetic_compression_training_10GPUs.sbatch
-	roberta_arithmetic_compression_training_12GPUs.sbatch
-	roberta_arithmetic_compression_training_14GPUs.sbatch
-	roberta_arithmetic_compression_training_16GPUs.sbatch
	RoBERTa + AC Inference Codes
-	roberta_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	roberta_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
	RoBERTa + AC Inference SBATCH Scripts
-	roberta_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	roberta_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
	T5-Small Fine-tuning Codes
-	T5_ddp_training_1GPU.py
-	T5_ddp_training_2GPUs.py
-	T5_ddp_training_4GPUs.py
-	T5_ddp_training_6GPUs.py
-	T5_ddp_training_8GPUs.py
-	T5_ddp_training_10GPUs.py
-	T5_ddp_training_12GPUs.py
-	T5_ddp_training_14GPUs.py
-	T5_ddp_training_16GPUs.py
	T5-Small Fine-tuning SBATCH Scripts
-	T5_arithmetic_compression_training_1GPU.sbatch
-	T5_arithmetic_compression_training_2GPUs.sbatch
-	T5_arithmetic_compression_training_4GPUs.sbatch
-	T5_arithmetic_compression_training_6GPUs.sbatch
-	T5_arithmetic_compression_training_8GPUs.sbatch
-	T5_arithmetic_compression_training_10GPUs.sbatch
-	T5_arithmetic_compression_training_12GPUs.sbatch
-	T5_arithmetic_compression_training_14GPUs.sbatch
-	T5_arithmetic_compression_training_16GPUs.sbatch
	T5-Small + AC Inference Codes
-	T5-Small_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	T5-Small_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
	T5-Small + AC Inference SBATCH Scripts
-	T5-Small_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	T5-Small_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
	Llama-3.2-3B Fine-tuning Codes
-	Llama_ddp_training_1GPU.py
-	Llama_ddp_training_2GPUs.py
-	Llama_ddp_training_4GPUs.py
-	Llama_ddp_training_6GPUs.py
-	Llama_ddp_training_8GPUs.py
-	Llama_ddp_training_10GPUs.py
-	Llama_ddp_training_12GPUs.py
-	Llama_ddp_training_14GPUs.py
-	Llama_ddp_training_16GPUs.py
	Llama-3.2-3B Fine-tuning SBATCH Scripts
-	Llama-3.2-3B_arithmetic_compression_training_1GPU.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_2GPUs.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_4GPUs.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_6GPUs.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_8GPUs.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_10GPUs.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_12GPUs.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_14GPUs.sbatch
-	Llama-3.2-3B_arithmetic_compression_training_16GPU.sbatch
	Llama-3.2-3B + AC Inference Codes
-	Llama-3.2-3B_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
-	Llama-3.2-3B_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py
	Llama-3.3-3B + AC Inference SBATCH Scripts
-	Llama-3.2_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_4GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_6GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_10GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_12GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_14GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
-	Llama-3.2_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch

•	For fine-tuning BERT, RoBERTa, T5-Small, and Llama-3.2-3B on 1 GPU, please run the corresponding SBATCH scripts with “1GPU” labels
For example:
	Run the “sbatch bert_arithmetic_compression_training_1GPU.sbatch” command for fine-tuning BERT on 1 GPU
	Run the “sbatch roberta_arithmetic_compression_training_1GPU.sbatch” command for fine-tuning RoBERTa on 1 GPU
	Run the “sbatch T5_arithmetic_compression_training_1GPU.sbatch” command for fine-tuning T5-Small on 1 GPU
	Run the “sbatch Llama-3.2-3B_arithmetic_compression_training_1GPU.sbatch” command for fine-tuning Llama-3.2-3B on 1 GPU

•	For fine-tuning BERT, RoBERTa, T5-Small, and Llama-3.2-3B on multiple GPUs, please run the corresponding SBATCH scripts with multi-GPU labels
For example: 
	Run the “sbatch bert_arithmetic_compression_training_8GPUs.sbatch” command for fine-tuning BERT on 8 GPUs
	Run the “sbatch roberta_arithmetic_compression_training_8GPUs.sbatch” command for fine-tuning RoBERTa on 8 GPUs
	Run the “sbatch T5_arithmetic_compression_training_8GPUs.sbatch” command for fine-tuning T5-Small on 8 GPUs
	Run the “sbatch Llama-3.2-3B_arithmetic_compression_training_8GPUs.sbatch” command for fine-tuning Llama-3.2-3B on 8 GPUs

•	For running inference for BERT, RoBERTa, T5-Small, and Llama-3.2-3B with AC on 1 GPU, please run the corresponding SBATCH scripts with “1GPU” labels
For example: 
	Run the “sbatch bert_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for BERT + AC inference on 1 GPU
	Run the “sbatch roberta_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for RoBERTa + AC inference on 1 GPU
	Run the “sbatch T5-Small_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for T5-Small + AC inference on 1 GPU 
	Run the “sbatch Llama-3.2_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for Llama-3.2-3B + AC inference on 1 GPU

•	For running inference for BERT, RoBERTa, T5-Small, and Llama-3.2-3B with AC on multiple GPUs please run the corresponding SBATCH scripts with multi-GPU labels
For example: 
	Run the “sbatch bert_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for BERT + AC inference on 8 GPUs
	Run the “sbatch roberta_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for RoBERTa + AC inference on 8 GPUs
	Run the “sbatch T5-Small_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for T5-Small + AC inference on 8 GPUs
	Run the “sbatch Llama-3.2_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch” command for Llama-3.2-3B + AC inference on 8 GPUs

**Acknowledgement**
This research work was conducted using the O’Donnell Data Science and Research Computing Institute’s (ODSRCI’s) HPC platform at Southern Methodist University (SMU).

