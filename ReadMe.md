## Title: LLM-Based Text Compression with Arithmetic Coding on Distributed GPU Systems

## Overview

This repository provides code for studying hybrid text compression pipelines that combine transformer-based Large Language Models (LLMs) with Arithmetic Coding (AC). In this framework, an LLM is used as a probabilistic predictor to estimate token-level conditional probability distributions, which are then consumed by AC to produce compressed bitstreams. The repository supports four representative transformer architectures: BERT, RoBERTa, T5-Small, and Llama-3.2-3B. The codebase includes fine-tuning and inference workflows designed for multi-GPU and multi-node high-performance computing (HPC) environments. The goal of this release is to support reproducible experimentation with LLM-based text compression, with emphasis on distributed execution, deterministic inference, GPU resource utilization, and the systems-level behavior of hybrid LLM + AC pipelines.

## System Architecture

<div align="center">
![alt text](image.png)
<p><b>Figure 1:</b> End-to-end LLM-based hybrid text compression pipeline with AC</p>
</div>

The system architecture shown above follows an end-to-end, hybrid text compression pipeline that integrates transformer-based LLMs with AC. The first two blocks in the architecture represent the fine-tuning stage, where the training data is loaded, tokenized, and then transformed into context–label pairs using a fixed-length context window of 64 tokens for learning next-token predictions. The LLMs are fine-tuned autoregressively in distributed GPU configurations, where each training step performs a forward pass on the training data, computes the cross-entropy loss, and updates the weights. The remaining blocks represent the inference stage. Here, the new, unseen data (not part of the training data) is loaded, tokenized, and converted into streams of context tokens, which are passed through the fine-tuned LLM checkpoints to generate deterministic next-token probability distributions. These probabilities are further transformed into integer CDFs so that they can be consumed by the AC’s encoder during compression to produce compressed bitstreams. The same probability distributions and compressed bitstreams are later used by the AC’s decoder to perform lossless reconstruction of the original sequence of tokens. Lastly, the performance of the pipeline is evaluated by computing and measuring metrics such as compression ratio, bits per character (BPC), bits per token (BPT), cross-entropy, perplexity, KL-divergence, reconstruction accuracy, wall-clock times, memory usage, and scaling efficiency.

## Key Contributions
- Hybrid text compression pipelines integrating transformer-based LLMs with AC
- Support for BERT, RoBERTa, T5-Small, and Llama-3.2-3B architectures
- Multi-GPU and multi-node distributed fine-tuning and inference workflows
- Deterministic AC inference pipelines for reproducible experimentation
- SLURM-based HPC execution scripts for scalable workloads
- End-to-end implementations for fine-tuning, inference, and AC integration

## Repository Structure

├── environment.yml
├── requirements.txt
├── README.md
├── BERT/
│   ├── BERT Fine-Tuning Codes/
│   ├── BERT Fine-Tuning SBATCH Scripts/
│   ├── BERT + AC Inference Codes/
│   └── BERT + AC Inference SBATCH Scripts/
├── RoBERTa/
│   ├── RoBERTa Fine-Tuning Codes/
│   ├── RoBERTa Fine-Tuning SBATCH Scripts/
│   ├── RoBERTa + AC Inference Codes/
│   └── RoBERTa + AC Inference SBATCH Scripts/
├── T5-Small/
│   ├── T5-Small Fine-Tuning Codes/
│   ├── T5-Small Fine-Tuning SBATCH Scripts/
│   ├── T5-Small + AC Inference Codes/
│   └── T5-Small + AC Inference SBATCH Scripts/
├── Llama-3.2-3B/
      ├── Llama-3.2-3B Fine-Tuning Codes/
      ├── Llama-3.2-3B Fine-Tuning SBATCH Scripts/
      ├── Llama-3.2-3B + AC Inference Codes/
      └── Llama-3.2-3B + AC Inference SBATCH Scripts/

## System Requirements

- Any machine or workstation with access to a dedicated GPU or HPC support
- NVIDIA CUDA Toolkit 12.8
- PyTorch 2.10.0
- Hugging Face transformers 4.57.1
- SLURM workload manager
- Linux environment running the Ubuntu 22.04.5 LTS operating system

All codes have been developed and implemented on the NVIDIA DGX Ampere 100 (A100) SuperPOD. The SuperPOD consists of 20 nodes with 8 A100 Tensor Core GPUs each with 80 GB of video random-access memory (VRAM), 128 CPU cores, and 2 physical CPU sockets on each node. In total, this HPC system provides approximately 1.64 PetaFLOPS (PFLOPS) of computational ability. Additionally, the SuperPOD offers a total storage capacity of 52.5 TB with 2 TB of storage capacity per node and 200 Gb/s of InfiniBand connectivity per node with NVLink support for high-speed GPU-to-GPU communications. For more information on the SuperPOD, please visit the following link: https://www.smu.edu/provost/odonnell-institute/hpc.

## Installation

- First, create an environment using “environment.yml” and a name of your choice:
    ```bash
    conda env create -f environment.yml
    conda activate env_name
    ```

- If an environment is already set up, then the required PyTorch, Hugging Face transformers, and CUDA toolkit libraries as well as other dependencies can be installed using “requirements.txt”:
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

The codes in this repository use the enwiki9 dataset as the input source for running both fine-tuning and inference. The enwiki9 dataset is an English text corpus that covers a broad range of encyclopedic topics derived from the first 1 GB of Wikipedia in XML format. This dataset spans all topics typically found in an encyclopedia, from history, science, and geography to biographies and arts, and is commonly used in benchmarking text compression frameworks. The dataset can be downloaded from the Large Text Compression Benchmark (https://www.mattmahoney.net/dc/text.html), Hugging Face, or Kaggle, where it is provided as a compressed file. After downloading and extracting the dataset into the same project directory as the codes, it should appear as a plain text file that can be loaded directly into the codes. If the dataset is stored in a different location or directory, then the dataset path inside the corresponding Python scripts must be updated accordingly prior to launching the SBATCH scripts.

## Quickstart

- For fine-tuning BERT on 1 GPU, run:
    ```bash
    sbatch projects/username/text_compression/BERT/BERT_Fine-Tuning_SBATCH_Scripts/bert_arithmetic_compression_training_1GPU.sbatch
    ```

- For running BERT + AC inference and compression on 1 GPU, run:
    ```bash
    sbatch projects/username/text_compression/BERT/BERT+AC_Inference_SBATCH_Scripts/bert_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    ```

## Fine-Tuning

The fine-tuning stage trains each LLM - BERT, RoBERTa, T5-Small, and Llama-3.2-3B - to serve as a causal, next-token predictor for the AC pipeline. In this stage, the first 50,000 lines of enwiki9 are tokenized and transformed into fixed-length context windows of 64 tokens, which are later consumed by the LLMs to learn and predict the next token from the preceding tokens in the sequence. The resulting fine-tuned LLM checkpoints are saved and employed during inference and compression. Fine-tuning is supported for all four LLMs on 1, 2, 4, 6, 8, 10, 12, 14, and 16 GPU configurations across two compute nodes on the SuperPOD. In the case of the SuperPOD, one compute node consists of 8 NVIDIA A100 GPUs and therefore, using two compute nodes provides access to a total of 16 GPUs for running distributed fine-tuning experiments.

To launch fine-tuning directly from the login node while inside the project directory, run the SBATCH script that matches the desired LLM and GPU count. For example, to fine-tune each LLM on 1 GPU, run:
    ```bash
    - sbatch bert_arithmetic_compression_training_1GPU.sbatch
    - sbatch roberta_arithmetic_compression_training_1GPU.sbatch
    - sbatch t5-small_arithmetic_compression_training_1GPU.sbatch
    - sbatch llama-3.2-3B_arithmetic_compression_training_1GPU.sbatch
    ```

To launch multi-GPU fine-tuning directly from the login node while inside the project directory, use the same naming convention with the required GPU count. For example, to fine-tune each LLM on 8 GPUs, run:
    ```bash
    - sbatch bert_arithmetic_compression_training_8GPUs.sbatch
    - sbatch roberta_arithmetic_compression_training_8GPUs.sbatch
    - sbatch t5-small_arithmetic_compression_training_8GPUs.sbatch
    - sbatch llama-3.2-3B_arithmetic_compression_training_8GPUs.sbatch
    ```

## Inference

The inference and compression stage uses the saved fine-tuned LLM checkpoints to compress unseen text with AC. In this stage, the selected model checkpoints are loaded in evaluation mode and used to generate deterministic next-token probability distributions for lines 50,001 to 100,000 of the enwiki9 dataset. This unseen data is tokenized using the same tokenizer as in the fine-tuning stage and then divided into multiple independent streams for autoregressive processing. The LLMs generate logits for the next token for every context window of preceding 64 tokens, which are then converted into normalized probability distributions using softmax. The generated token probabilities are then converted into integer cumulative distribution functions (CDFs). Since AC requires integer frequency ranges instead of floating-point probabilities, the CDF construction step creates deterministic integer intervals that preserve valid token ordering and prevent zero-width ranges. The AC step is integrated after probability generation and CDF construction. During encoding, the true next token selects its corresponding CDF interval, and AC updates the interval range to emit compressed bits. During decoding, the same probability distributions and integer CDF ranges are used to reconstruct the original token sequences from the encoded (compressed) bitstreams. This part of the inference and compression pipeline remains CPU-bound and single-threaded because AC is inherently sequential. Like fine-tuning, inference and compression are also supported for all four LLM-based AC compression frameworks on 1, 2, 4, 6, 8, 10, 12, 14, and 16 GPU configurations across two compute nodes on the SuperPOD. As mentioned previously, one compute node in the SuperPOD contains 8 NVIDIA A100 GPUs and therefore, two compute nodes provide access to 16 GPUs for running distributed inference experiments.

To launch inference and compression directly from the login node while inside the project directory, use the SBATCH script that matches the desired framework and GPU count. For example, to launch LLM + AC inference on 1 GPU, run:
    ```bash
    - sbatch bert_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    - sbatch roberta_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    - sbatch t5-small_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    - sbatch llama-3.2-3B_arithmetic_compression_1GPU_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    ```

To launch multi-GPU inference and compression directly from the login node while inside the project directory, use the same naming convention with the required GPU count. For example, to launch LLM + AC inference on 8 GPUs, run:
    ```bash
    - sbatch bert_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    - sbatch roberta_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    - sbatch t5-small_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch
    - sbatch llama-3.2-3B_arithmetic_compression_8GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.sbatch

## Reproducibility Notes

Several implementation details are included to make the compression experiments deterministic and reproducible across GPU configurations. During inference, TF32 is disabled to avoid numerical precision drifts because AC is highly sensitive to small changes in token probability distributions and integer CDFs. Token streams are tensorized inside the encoding and decoding hot loops, Python list indexing and repeated .item() calls are removed, and CDF construction is batched and tensorized on the GPU to reduce computational overhead and maintain consistent probability-to-CDF conversions. The repository also uses deterministic manual sharding instead of DistributedSampler, where the total number of streams is selected to be exactly divisible by the number of GPUs, allowing each GPU to receive an equal workload and exact token coverage. CPU resource usage is controlled using settings such as OMP_NUM_THREADS=1, torch.set_num_threads(1), and torch.set_num_interop_threads(1) to avoid oversubscription during the CPU-bound AC stage. Moreover, timing blocks, CUDA synchronization calls, and SLURM utilities such as sacct are used to measure GPU inference timings, GPU CDF construction timings, GPU-to-CPU transfer timings, CPU AC timings, and memory usage consistently across single-GPU and multi-GPU inference runs.

## Acknowledgement

This repository is currently released for research and evaluation purposes only. Formal license information will be added in a future release.
