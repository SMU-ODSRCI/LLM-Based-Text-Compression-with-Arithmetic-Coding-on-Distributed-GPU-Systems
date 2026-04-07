# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

#!/bin/bash -e

# ===== Shared Job Settings =====
#SBATCH -p batch
#SBATCH -A nimam_ml_project_0002
#SBATCH -J Llama-3.2_arithmetic_compression
#SBATCH --output=Llama-3.2_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS_v2_%j.out
#SBATCH --error=Llama-3.2_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS_v2_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=abrara@mail.smu.edu

module purge
module load conda
conda activate zero_shot_text_compression

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29507
export NCCL_SOCKET_IFNAME=^lo,docker
export OMP_NUM_THREADS=1

JOBID="${SLURM_JOB_ID}"

echo "====== JOB START ========"
hostname
echo "JobID: ${JOBID}"
echo "----- sstat (start) -----"
sstat -j "${JOBID}" --format=JobID,AveRSS,MaxRSS,MaxVMSize 2>&1 || true
echo "----- sacct (start snapshot; may be sparse early) -----"
sacct -j "${JOBID}" --format=JobID,MaxRSS,ReqMem,Elapsed 2>&1 || true
echo "========================="

srun --cpu-bind=cores --cpus-per-task=4 python3 -u Llama-3.2-3B_arithmetic_compression_16GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py

echo "======= JOB END ========"
hostname
echo "JobID: ${JOBID}"
echo "----- sstat (end) -----"
sstat -j "${JOBID}" --format=JobID,AveRSS,MaxRSS,MaxVMSize 2>&1 || true
echo "----- sacct (end summary) -----"
sleep 2
sacct -j "${JOBID}" --format=JobID,MaxRSS,ReqMem,Elapsed 2>&1 || true
echo "========================"
