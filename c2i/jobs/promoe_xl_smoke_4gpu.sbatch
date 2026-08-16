#!/bin/bash
#SBATCH --job-name=pixdit-promoe-xl-smoke
#SBATCH --partition=gpujl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wangye/projects/PixelDiT-ProMoE/c2i/promoe_xl_smoke-%j.out
#SBATCH --error=/home/wangye/projects/PixelDiT-ProMoE/c2i/promoe_xl_smoke-%j.err

set -euo pipefail

REPO_ROOT=/home/wangye/projects/PixelDiT-ProMoE
C2I_ROOT=${REPO_ROOT}/c2i
CONFIG=${C2I_ROOT}/configs/pix256_promoe_xl_smoke.yaml
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source /home/wangye/anaconda3/etc/profile.d/conda.sh
conda activate pixel
cd ${C2I_ROOT}

torchrun --nproc_per_node=4 main.py fit -c "${CONFIG}" --trainer.devices=4

RUN_DIR=${C2I_ROOT}/train_logs/exp_pixdit_promoe_xl_smoke
CHECKPOINT=$(find "${RUN_DIR}" -maxdepth 1 -name '*.ckpt' -type f | sort | tail -n 1)
test -n "${CHECKPOINT}"
test -f "${CHECKPOINT}"
python tests/verify_promoe_checkpoint.py "${CHECKPOINT}"

# Lightning restores the full module and optimizer state. Since max_steps is
# still one, this verifies checkpoint loading without taking another step.
torchrun --nproc_per_node=4 main.py fit -c "${CONFIG}" --trainer.devices=4 --ckpt_path="${CHECKPOINT}"
