#!/bin/bash
#SBATCH --job-name=pixdit-promoe-b
#SBATCH --partition=gpujl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/wangye/projects/PixelDiT-ProMoE/c2i/promoe_b-%j.out
#SBATCH --error=/home/wangye/projects/PixelDiT-ProMoE/c2i/promoe_b-%j.err

set -euo pipefail

REPO_ROOT=/home/wangye/projects/PixelDiT-ProMoE
C2I_ROOT=${REPO_ROOT}/c2i
CONFIG=${C2I_ROOT}/configs/pix256_promoe_b.yaml
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /home/wangye/anaconda3/etc/profile.d/conda.sh
conda activate pixel
cd ${C2I_ROOT}

python -m torch.distributed.run --nproc_per_node=4 main.py fit -c "${CONFIG}" --trainer.devices=4
