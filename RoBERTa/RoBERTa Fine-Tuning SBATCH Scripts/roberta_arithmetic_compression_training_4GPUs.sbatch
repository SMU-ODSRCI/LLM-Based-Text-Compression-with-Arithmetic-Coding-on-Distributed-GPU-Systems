# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

#!/bin/bash -e
#SBATCH -p batch
#SBATCH -A nimam_ml_project_0002
#SBATCH -J roberta_arithmetic_compression_training
#SBATCH --output=RoBERTa_arithmetic_compression_training_4GPUs_v3_%j.out
#SBATCH --error=RoBERTa_arithmetic_compression_training_4GPUs_v3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=abrara@mail.smu.edu

module purge
module load conda
conda activate zero_shot_text_compression

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29502
export NCCL_SOCKET_IFNAME=^lo,docker
export OMP_NUM_THREADS=1

srun --cpu-bind=cores --cpus-per-task=4 python3 -u roberta_ddp_training_4GPUs.py
