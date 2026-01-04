#!/bin/bash
#SBATCH -J amsua_pretrain
#SBATCH -p qgpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
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

# 1) 生成 split（单进程）
python train_amsua_vaeformer_lite.py --data_dir . --make_split_only --show_split 3

# 2) 计算训练集统计量（mean/std），得到 stats_amsua.json
python train_amsua_vaeformer_lite.py --data_dir . --compute_stats --stats_out stats_amsua.json

# 3) 两卡 DDP 训练（加载 stats 做标准化，loss 强制 fp32）
torchrun --standalone --nproc_per_node=2 train_amsua_vaeformer_lite.py \
  --data_dir . \
  --stage pretrain \
  --out_dir runs_amsua/pretrain_${SLURM_JOB_ID} \
  --epochs 10 \
  --steps_per_epoch 1000 \
  --val_steps 200 \
  --batch_size 2 \
  --num_workers 2 \
  --patch_h 256 --patch_w 256 --patch_size 16 \
  --in_channels 40 \
  --lr 3e-5 \
  --stats_json stats_amsua.json \
  --norm_clamp 10 \
  --amp
