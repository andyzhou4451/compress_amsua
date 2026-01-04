#!/bin/bash
#SBATCH -J amsua_eval_test
#SBATCH -p qgpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e
mkdir -p logs
cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate amsua

# 先找 best（如果存在）
CKPT=$(ls -t runs_amsua/finetune_*/best_finetune_entropy.pt 2>/dev/null | head -n 1)

# 如果 best 不存在，就退化为最新 epoch ckpt
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t runs_amsua/finetune_*/ckpt_finetune_entropy_epoch*.pt 2>/dev/null | head -n 1)
fi

if [ -z "$CKPT" ]; then
  echo "[ERROR] cannot find finetune ckpt (best or epoch ckpt)"
  echo "[HINT] run: ls -lh runs_amsua/finetune_*"
  exit 1
fi

echo "[INFO] CKPT=$CKPT"


srun -n 1 python eval_amsua_split.py \
  --data_dir . \
  --split_json splits_amsua.json \
  --split test \
  --stats_json stats_amsua.json \
  --ckpt $CKPT \
  --stage finetune_entropy \
  --num_samples 4000 \
  --batch_size 4 \
  --amp --device cuda \
  --out_json eval_test_${SLURM_JOB_ID}.json
