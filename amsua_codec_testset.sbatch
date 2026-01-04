#!/bin/bash
#SBATCH -J amsua_codec_test
#SBATCH -p qgpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e
mkdir -p logs
cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate amsua

# 先 best，找不到就用最新 epoch ckpt
CKPT=$(ls -t runs_amsua/finetune_*/best_finetune_entropy.pt 2>/dev/null | head -n 1)
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t runs_amsua/finetune_*/ckpt_finetune_entropy_epoch*.pt 2>/dev/null | head -n 1)
fi
if [ -z "$CKPT" ]; then
  echo "[ERROR] cannot find finetune ckpt (best or epoch ckpt)"
  exit 1
fi
echo "[INFO] CKPT=$CKPT"

# ✅ 关键：导出给 Python heredoc
export CKPT

python - <<'PY'
import json, os, subprocess

split = json.load(open("splits_amsua.json", "r"))
test_files = split["test"]

ckpt = os.environ.get("CKPT", None)
assert ckpt and os.path.exists(ckpt), ckpt
print("[PY] using ckpt:", ckpt)
print("[PY] test files:", len(test_files))

for f in test_files:
    code = f.replace(".npz", ".code.npz")
    recon = f.replace(".npz", ".recon.npz")
    metrics = f.replace(".npz", ".metrics.json")

    # 1) encode -> 生成中间压缩表示 code
    subprocess.check_call([
        "python","amsua_codec_tiles.py","--mode","encode",
        "--ckpt",ckpt,"--stats_json","stats_amsua.json",
        "--input_npz",f,"--out_code",code,
        "--tile_h","256","--tile_w","256","--patch_size","16",
        "--estimate_bpp","--amp","--device","cuda"
    ])

    # 2) decode -> 生成还原 recon
    subprocess.check_call([
        "python","amsua_codec_tiles.py","--mode","decode",
        "--ckpt",ckpt,
        "--input_code",code,"--out_recon",recon,
        "--amp","--device","cuda"
    ])

    # 3) compare -> 生成误差 metrics
    subprocess.check_call([
        "python","amsua_codec_tiles.py","--mode","compare",
        "--orig_npz",f,"--recon_npz",recon,"--out_json",metrics
    ])

print("[DONE] processed test files:", len(test_files))
PY
