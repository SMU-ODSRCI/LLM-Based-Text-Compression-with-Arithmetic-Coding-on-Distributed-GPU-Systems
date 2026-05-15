#!/bin/bash -e

# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

# ===== Shared Job Settings =====
#SBATCH -p batch
#SBATCH -A nimam_ml_project_0002
#SBATCH -J bert_arithmetic_compression
#SBATCH --output=bert_arithmetic_compression_training_10GPUs_v3_%j.out
#SBATCH --error=bert_arithmetic_compression_training_10GPUs_v3_%j.err
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=abrara@mail.smu.edu

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=64G
#SBATCH hetjob
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=64G

module purge
module load conda
conda activate llm_ac_text_compression

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29505
export NCCL_SOCKET_IFNAME=^lo,docker
export OMP_NUM_THREADS=1

srun --cpu-bind=cores --cpus-per-task=4 --het-group=0,1 python3 -u bert_ddp_training_10GPUs.py
