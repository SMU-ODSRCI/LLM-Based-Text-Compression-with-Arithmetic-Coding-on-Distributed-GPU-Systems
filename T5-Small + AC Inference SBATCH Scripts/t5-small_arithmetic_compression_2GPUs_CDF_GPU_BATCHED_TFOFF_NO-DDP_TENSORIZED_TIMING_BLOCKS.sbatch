# --------------------------------------------------------------------------------------------
# This code has been written by the O' Donnell Data Science and Research Computing Institute
# at Southern Methodist University. It is intended for research purposes only.
# Do not redistribute or use this code for commercial applications without seeking permission.
# --------------------------------------------------------------------------------------------

#!/bin/bash -e
#SBATCH -p batch
#SBATCH -A nimam_ml_project_0002
#SBATCH -J t5-small_arithmetic_compression
#SBATCH --output=T5-Small_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS_v2_%j.out
#SBATCH --error=T5-Small_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END
#SBATCH --mail-user=abrara@mail.smu.edu

module purge
module load conda
conda activate zero_shot_text_compression

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
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

srun --cpu-bind=cores --cpus-per-task=4 python3 -u t5-small_arithmetic_compression_2GPUs_CDF_GPU_BATCHED_TFOFF_NO-DDP_TENSORIZED_TIMING_BLOCKS.py

echo "======= JOB END ========"
hostname
echo "JobID: ${JOBID}"
echo "----- sstat (end) -----"
sstat -j "${JOBID}.0" --format=JobID,AveRSS,MaxRSS,MaxVMSize 2>&1 || true
echo "----- sacct (end summary) -----"
sleep 2
sacct -j "${JOBID}" --format=JobID,MaxRSS,ReqMem,Elapsed 2>&1 || true
echo "========================"
