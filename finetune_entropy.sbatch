#!/bin/bash
#SBATCH -J amsua_finetune
#SBATCH -p qgpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e
mkdir -p logs
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/.bashrc
conda activate amsua

echo "[INFO] pwd=$(pwd)"
echo "[INFO] listing runs_amsua:"
ls -lh runs_amsua || true

# 自动找最新的 pretrain ckpt
PRETRAIN_CKPT=$(ls -t runs_amsua/pretrain_*/best_pretrain.pt 2>/dev/null | head -n 1)
if [ -z "$PRETRAIN_CKPT" ]; then
  echo "[ERROR] cannot find runs_amsua/pretrain_*/best_pretrain.pt"
  echo "[HINT] try: find runs_amsua -name best_pretrain.pt -print"
  exit 1
fi
echo "[INFO] using PRETRAIN_CKPT=$PRETRAIN_CKPT"
ls -lh "$PRETRAIN_CKPT"

OUTDIR=runs_amsua/finetune_${SLURM_JOB_ID}
mkdir -p $OUTDIR

# 注意：resume 的 ckpt epoch=10，所以 epochs=15 才会跑 5 轮
srun -n 1 python train_amsua_vaeformer_lite.py \
  --data_dir . \
  --stage finetune_entropy \
  --resume $PRETRAIN_CKPT \
  --out_dir $OUTDIR \
  --epochs 15 \
  --steps_per_epoch 1000 \
  --val_steps 200 \
  --batch_size 2 \
  --num_workers 2 \
  --patch_h 256 --patch_w 256 --patch_size 16 \
  --in_channels 40 \
  --stats_json stats_amsua.json \
  --norm_clamp 10 \
  --lr 5e-5 \
  --lambda_rd 0.01 \
  --amp \
  --device cuda

echo "[INFO] finetune out_dir content:"
ls -lh $OUTDIR || true

FINETUNE_CKPT=$OUTDIR/best_finetune_entropy.pt
if [ ! -f "$FINETUNE_CKPT" ]; then
  echo "[ERROR] cannot find finetune ckpt: $FINETUNE_CKPT"
  exit 1
fi
echo "[INFO] finetune ckpt: $FINETUNE_CKPT"
