#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, os, glob, json, random, argparse
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int):
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)

def is_rank0(rank: int) -> bool:
    return rank == 0

def ddp_barrier(distributed: bool):
    if distributed:
        torch.distributed.barrier()

def ddp_allreduce_sum(distributed: bool, x: torch.Tensor) -> torch.Tensor:
    if distributed:
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


# -----------------------------
# DDP init (torchrun / slurm compatible)
# -----------------------------
def _env_int(name: str, default: int):
    v = os.environ.get(name, None)
    return int(v) if v is not None else default

def init_distributed(args):
    world_size = _env_int("WORLD_SIZE", -1)
    rank = _env_int("RANK", -1)
    local_rank = _env_int("LOCAL_RANK", -1)

    # slurm fallback
    if world_size < 0:
        world_size = _env_int("SLURM_NTASKS", 1)
    if rank < 0:
        rank = _env_int("SLURM_PROCID", 0)
    if local_rank < 0:
        local_rank = _env_int("SLURM_LOCALID", 0)

    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        # newer torch supports device_id; keep compatibility
        try:
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=rank,
                device_id=device if backend == "nccl" else None,
            )
        except TypeError:
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
        torch.distributed.barrier()
    else:
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    return distributed, rank, local_rank, world_size, device


# -----------------------------
# split helpers
# -----------------------------
def list_npz_files(data_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    return [os.path.relpath(f, data_dir) for f in files]

def make_split(files: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    files = files.copy()
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {"train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]}

def load_or_create_split(data_dir: str, pattern: str, split_json: str,
                         train_ratio: float, val_ratio: float, test_ratio: float,
                         seed: int, distributed: bool, rank: int):
    path = os.path.join(data_dir, split_json)
    if os.path.exists(path):
        return json.load(open(path, "r", encoding="utf-8"))

    if is_rank0(rank):
        files = list_npz_files(data_dir, pattern)
        if not files:
            raise FileNotFoundError(f"No npz in {data_dir} pattern={pattern}")
        sp = make_split(files, train_ratio, val_ratio, test_ratio, seed)
        json.dump(sp, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[split] created: {path} (train={len(sp['train'])}, val={len(sp['val'])}, test={len(sp['test'])})")

    ddp_barrier(distributed)
    return json.load(open(path, "r", encoding="utf-8"))


# -----------------------------
# stats (mean/std) for normalization
# -----------------------------
def compute_stats_from_train(data_dir: str, train_files: List[str], in_channels: int,
                             stats_out: str, fill_invalid: float = 0.0):
    """
    用训练集所有有效点计算每个 feature 的 mean/std（float64 累加，避免溢出）。
    stats_out: json file path
    """
    sum_ = np.zeros((in_channels,), dtype=np.float64)
    sumsq = np.zeros((in_channels,), dtype=np.float64)
    cnt = np.zeros((in_channels,), dtype=np.float64)
    vmin = np.full((in_channels,), np.inf, dtype=np.float64)
    vmax = np.full((in_channels,), -np.inf, dtype=np.float64)

    feat_names = None

    for i, rel in enumerate(train_files):
        path = os.path.join(data_dir, rel)
        d = np.load(path, allow_pickle=True)
        X = d["X"]  # (T,F,H,W)
        M = d["M"]  # (T,F,H,W)

        if "feature_names" in d and feat_names is None:
            try:
                feat_names = [str(x) for x in d["feature_names"].tolist()]
            except Exception:
                feat_names = None

        if X.ndim != 4:
            raise ValueError(f"{rel}: X must be (T,F,H,W), got {X.shape}")
        T, F, H, W = X.shape
        if F != in_channels:
            raise ValueError(f"{rel}: feature={F} != in_channels={in_channels}")

        X = X.astype(np.float64)
        M = M.astype(np.float64)
        M = np.where(np.isfinite(M), M, 0.0)
        M = (M > 0.5)

        finite = np.isfinite(X)
        valid = M & finite

        # 将 invalid 处置 0 方便 sum/sumsq
        X0 = np.where(valid, X, 0.0)

        # sum/sumsq/count over (T,H,W) -> (F,)
        sum_ += X0.sum(axis=(0, 2, 3))
        sumsq += (X0 * X0).sum(axis=(0, 2, 3))
        cnt += valid.sum(axis=(0, 2, 3))

        # min/max（忽略 invalid）
        x_min = np.where(valid, X, np.inf).min(axis=(0, 2, 3))
        x_max = np.where(valid, X, -np.inf).max(axis=(0, 2, 3))
        vmin = np.minimum(vmin, x_min)
        vmax = np.maximum(vmax, x_max)

        if (i + 1) % 5 == 0:
            print(f"[stats] processed {i+1}/{len(train_files)} files")

    mean = np.zeros((in_channels,), dtype=np.float64)
    std = np.ones((in_channels,), dtype=np.float64)

    for c in range(in_channels):
        if cnt[c] > 0:
            mean[c] = sum_[c] / cnt[c]
            var = sumsq[c] / cnt[c] - mean[c] * mean[c]
            var = max(var, 1e-6)
            std[c] = math.sqrt(var)
        else:
            mean[c] = 0.0
            std[c] = 1.0
            vmin[c] = 0.0
            vmax[c] = 0.0

    out = {
        "in_channels": in_channels,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "count": cnt.astype(float).tolist(),
        "min": vmin.astype(float).tolist(),
        "max": vmax.astype(float).tolist(),
        "feature_names": feat_names,
    }
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[stats] saved -> {stats_out}")


def load_stats(stats_json: str, in_channels: int):
    with open(stats_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    if mean.shape[0] != in_channels or std.shape[0] != in_channels:
        raise ValueError(f"stats feature mismatch: {mean.shape[0]} vs in_channels={in_channels}")
    return mean, std, d.get("feature_names", None)


# -----------------------------
# Crop (lon wrap)
# -----------------------------
def crop_lon_wrap(arr: np.ndarray, top: int, left: int, ph: int, pw: int) -> np.ndarray:
    C, H, W = arr.shape
    patch = arr[:, top:top+ph, :]
    if left + pw <= W:
        return patch[:, :, left:left+pw]
    r = (left + pw) - W
    return np.concatenate([patch[:, :, left:W], patch[:, :, 0:r]], axis=-1)


# -----------------------------
# Dataset (sanitize + optional normalize)
# -----------------------------
class AMSUARandomPatchDataset(Dataset):
    def __init__(self, data_dir: str, files: List[str],
                 patch_h: int, patch_w: int,
                 samples: int,
                 in_channels: int = 40,
                 fill_invalid: float = 0.0,
                 min_valid_frac: float = 0.01,
                 max_resample: int = 30,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None,
                 norm_clamp: float = 10.0):
        self.data_dir = data_dir
        self.files = files
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.samples = samples
        self.in_channels = in_channels
        self.fill_invalid = float(fill_invalid)
        self.min_valid_frac = float(min_valid_frac)
        self.max_resample = int(max_resample)
        self.mean = mean
        self.std = std
        self.norm_clamp = float(norm_clamp)

        self._cache_path = None
        self._cache_X = None
        self._cache_M = None

    def __len__(self):
        return self.samples

    def _load(self, rel: str):
        path = os.path.join(self.data_dir, rel)
        if self._cache_path == path and self._cache_X is not None:
            return self._cache_X, self._cache_M
        d = np.load(path, allow_pickle=True)
        X = d["X"]
        M = d["M"]
        self._cache_path = path
        self._cache_X = X
        self._cache_M = M
        return X, M

    def __getitem__(self, idx):
        for _ in range(self.max_resample):
            rel = self.files[np.random.randint(0, len(self.files))]
            X, M = self._load(rel)

            if X.ndim != 4:
                raise ValueError(f"{rel}: X must be (T,F,H,W), got {X.shape}")
            T, F, H, W = X.shape
            if F != self.in_channels:
                raise ValueError(f"{rel}: feature={F} != in_channels={self.in_channels}")

            t = np.random.randint(0, T)
            x = X[t].astype(np.float32)      # (F,H,W)
            m = M[t].astype(np.float32)

            # mask 清洗
            m = np.where(np.isfinite(m), m, 0.0)
            m = (m > 0.5).astype(np.float32)

            # 保证 mask==1 的地方 x 一定 finite；否则 mask 置 0，并把 x 非有限值填掉
            finite = np.isfinite(x)
            if not finite.all():
                m = m * finite.astype(np.float32)
                x = np.nan_to_num(x, nan=self.fill_invalid, posinf=self.fill_invalid, neginf=self.fill_invalid)

            # 先把 invalid 填成 fill_invalid（保证输入 finite）
            x = np.where(m > 0.5, x, self.fill_invalid).astype(np.float32)

            # 标准化（强烈推荐！）
            if self.mean is not None and self.std is not None:
                x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
                if self.norm_clamp > 0:
                    x = np.clip(x, -self.norm_clamp, self.norm_clamp)
                # invalid 位置固定置 0（对应 mean）
                x = np.where(m > 0.5, x, 0.0).astype(np.float32)

            # crop
            top = np.random.randint(0, H - self.patch_h + 1)
            left = np.random.randint(0, W)
            x_patch = crop_lon_wrap(x, top, left, self.patch_h, self.patch_w)
            m_patch = crop_lon_wrap(m, top, left, self.patch_h, self.patch_w)

            if float(m_patch.mean()) >= self.min_valid_frac:
                return torch.from_numpy(x_patch), torch.from_numpy(m_patch)

        return torch.from_numpy(x_patch), torch.from_numpy(m_patch)


# -----------------------------
# Loss in FP32 (关键：避免 AMP 溢出)
# -----------------------------
def masked_mse_fp32(x_hat: torch.Tensor, x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6):
    x_hat = x_hat.float()
    x = x.float()
    m = m.float()
    diff = x_hat - x
    diff = torch.where(m > 0.5, diff, torch.zeros_like(diff))
    return (diff * diff).sum() / m.sum().clamp_min(eps)


# -----------------------------
# Model blocks
# -----------------------------
def get_2d_sincos_pos_embed(embed_dim: int, gh: int, gw: int, device: torch.device):
    assert embed_dim % 4 == 0
    half = embed_dim // 2
    dim_each = half // 2
    omega = torch.arange(dim_each, device=device, dtype=torch.float32) / dim_each
    omega = 1.0 / (10000 ** omega)

    y = torch.arange(gh, device=device, dtype=torch.float32)
    x = torch.arange(gw, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    out_y = yy * omega.reshape(1, -1)
    out_x = xx * omega.reshape(1, -1)

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    pos = torch.cat([pos_y, pos_x], dim=1)
    return pos.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x

def gaussian_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gaussian_likelihood(x, mean, scale, eps=1e-9):
    scale = torch.clamp(scale, min=1e-6)
    centered = x - mean
    upper = (centered + 0.5) / scale
    lower = (centered - 0.5) / scale
    probs = gaussian_cdf(upper) - gaussian_cdf(lower)
    return torch.clamp(probs, min=eps)

def quantize(x: torch.Tensor, training: bool):
    if training:
        return x + torch.empty_like(x).uniform_(-0.5, 0.5)
    return torch.round(x)

def compute_bpp_fp32(liks: Dict[str, torch.Tensor], H: int, W: int):
    total_bits = 0.0
    for p in liks.values():
        p = p.float()
        total_bits = total_bits + (-torch.log2(p)).sum()
    B = next(iter(liks.values())).shape[0]
    return total_bits / (B * H * W)

class VAEformerLite(nn.Module):
    def __init__(self, in_channels=40, patch_size=16, embed_dim=256, latent_dim=192, depth=6, heads=8):
        super().__init__()
        groups = 8 if in_channels >= 8 else 1
        self.in_norm = nn.GroupNorm(groups, in_channels)

        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, in_channels, patch_size, stride=patch_size)

        self.enc = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.dec = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])

        self.to_y = nn.Conv2d(embed_dim, latent_dim, 1)
        self.y_to_embed = nn.Conv2d(latent_dim, embed_dim, 1)

        self.h_a = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
        )
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(latent_dim, latent_dim, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(latent_dim, 2*latent_dim, 3, padding=1),
        )
        self.z_log_scale = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor, compute_likelihood: bool):
        B, C, H, W = x.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            raise ValueError(f"H,W={H,W} not divisible by patch_size={ps}")

        x = self.in_norm(x)

        e = self.patch_embed(x)
        Hp, Wp = e.shape[-2], e.shape[-1]
        tok = e.flatten(2).transpose(1, 2)
        pos = get_2d_sincos_pos_embed(tok.shape[-1], Hp, Wp, x.device)
        tok = tok + pos
        for blk in self.enc:
            tok = blk(tok)
        e2 = tok.transpose(1, 2).reshape(B, -1, Hp, Wp)

        y = self.to_y(e2)
        y_hat = quantize(y, self.training)

        # decode path
        d = self.y_to_embed(y_hat)
        dtok = d.flatten(2).transpose(1, 2) + pos
        for blk in self.dec:
            dtok = blk(dtok)
        d2 = dtok.transpose(1, 2).reshape(B, -1, Hp, Wp)
        x_hat = self.patch_unembed(d2)

        if not compute_likelihood:
            return {"x_hat": x_hat}

        z = self.h_a(y_hat)
        z_hat = quantize(z, self.training)
        params = self.h_s(z_hat)
        mean_y, log_scale_y = params.chunk(2, dim=1)
        scale_y = F.softplus(log_scale_y) + 1e-6

        y_lik = gaussian_likelihood(y_hat.float(), mean_y.float(), scale_y.float())
        z_scale = (F.softplus(self.z_log_scale).float()[None, :, None, None] + 1e-6)
        z_lik = gaussian_likelihood(z_hat.float(), torch.zeros_like(z_hat).float(), z_scale)

        return {"x_hat": x_hat, "y_likelihood": y_lik, "z_likelihood": z_lik}

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def configure_stage_trainables(model: VAEformerLite, stage: str):
    """
    pretrain：只训练主干，冻结 hyperprior（避免 DDP unused 参数报错）
    finetune_entropy：冻结主干，只训练 hyperprior
    """
    if stage == "pretrain":
        set_requires_grad(model.in_norm, True)
        set_requires_grad(model.patch_embed, True)
        set_requires_grad(model.patch_unembed, True)
        set_requires_grad(model.enc, True)
        set_requires_grad(model.dec, True)
        set_requires_grad(model.to_y, True)
        set_requires_grad(model.y_to_embed, True)

        set_requires_grad(model.h_a, False)
        set_requires_grad(model.h_s, False)
        model.z_log_scale.requires_grad_(False)

    elif stage == "finetune_entropy":
        set_requires_grad(model.in_norm, False)
        set_requires_grad(model.patch_embed, False)
        set_requires_grad(model.patch_unembed, False)
        set_requires_grad(model.enc, False)
        set_requires_grad(model.dec, False)
        set_requires_grad(model.to_y, False)
        set_requires_grad(model.y_to_embed, False)

        set_requires_grad(model.h_a, True)
        set_requires_grad(model.h_s, True)
        model.z_log_scale.requires_grad_(True)
    else:
        raise ValueError(stage)


@torch.no_grad()
def eval_loop(model, loader, stage, lambda_rd, device, distributed):
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    mse_sum = torch.tensor(0.0, device=device)
    bpp_sum = torch.tensor(0.0, device=device)
    n_sum = torch.tensor(0.0, device=device)

    for x, m in loader:
        x = x.to(device, non_blocking=True).float()
        m = m.to(device, non_blocking=True).float()

        # forward under autocast, loss outside
        with autocast(device_type="cuda", enabled=False):
            pass

        out = model(x, compute_likelihood=(stage != "pretrain"))
        x_hat = out["x_hat"]

        mse = masked_mse_fp32(x_hat, x, m)
        if stage == "pretrain":
            bpp = torch.tensor(0.0, device=device)
            loss = mse
        else:
            bpp = compute_bpp_fp32({"y": out["y_likelihood"], "z": out["z_likelihood"]}, x.shape[-2], x.shape[-1])
            loss = lambda_rd * mse + bpp

        loss_sum += loss
        mse_sum += mse
        bpp_sum += bpp
        n_sum += 1.0

    loss_sum = ddp_allreduce_sum(distributed, loss_sum)
    mse_sum = ddp_allreduce_sum(distributed, mse_sum)
    bpp_sum = ddp_allreduce_sum(distributed, bpp_sum)
    n_sum = ddp_allreduce_sum(distributed, n_sum)

    return {"loss": (loss_sum/n_sum).item(),
            "mse": (mse_sum/n_sum).item(),
            "bpp": (bpp_sum/n_sum).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--pattern", type=str, default="*.npz")
    ap.add_argument("--split_json", type=str, default="splits_amsua.json")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--make_split_only", action="store_true")
    ap.add_argument("--show_split", type=int, default=0)

    # stats
    ap.add_argument("--compute_stats", action="store_true", help="compute mean/std on train split then exit")
    ap.add_argument("--stats_out", type=str, default="stats_amsua.json")
    ap.add_argument("--stats_json", type=str, default=None, help="use this stats json to normalize input")
    ap.add_argument("--norm_clamp", type=float, default=10.0)

    # train
    ap.add_argument("--stage", type=str, default="pretrain", choices=["pretrain", "finetune_entropy"])
    ap.add_argument("--out_dir", type=str, default="./runs_amsua")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--val_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5, help="建议先用 3e-5，更稳")
    ap.add_argument("--lambda_rd", type=float, default=0.01)
    ap.add_argument("--resume", type=str, default=None)

    # model/patch
    ap.add_argument("--patch_h", type=int, default=256)
    ap.add_argument("--patch_w", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--in_channels", type=int, default=40)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)

    ap.add_argument("--fill_invalid", type=float, default=0.0)
    ap.add_argument("--min_valid_frac", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    distributed, rank, local_rank, world_size, device = init_distributed(args)
    set_seed(args.seed + rank)

    if is_rank0(rank):
        print(f"[ddp] distributed={distributed} world_size={world_size} rank={rank} local_rank={local_rank} device={device}")

    split = load_or_create_split(
        args.data_dir, args.pattern, args.split_json,
        args.train_ratio, args.val_ratio, args.test_ratio,
        args.seed, distributed, rank
    )

    if is_rank0(rank):
        print(f"[split] train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
        if args.show_split > 0:
            n = args.show_split
            print("[train files]", split["train"][:n])
            print("[val files]  ", split["val"][:n])
            print("[test files] ", split["test"][:n])

    if args.make_split_only:
        if is_rank0(rank):
            print("[split] done (make_split_only). Exit.")
        ddp_barrier(distributed)
        if distributed:
            torch.distributed.destroy_process_group()
        return

    # compute stats then exit (single process建议在 sbatch 里单独跑)
    if args.compute_stats:
        if not is_rank0(rank):
            ddp_barrier(distributed)
            if distributed:
                torch.distributed.destroy_process_group()
            return
        compute_stats_from_train(args.data_dir, split["train"], args.in_channels,
                                 stats_out=os.path.join(args.data_dir, args.stats_out),
                                 fill_invalid=args.fill_invalid)
        return

    # load stats if provided
    mean = std = None
    if args.stats_json is not None:
        mean, std, feat_names = load_stats(args.stats_json, args.in_channels)
        if is_rank0(rank):
            print(f"[stats] loaded {args.stats_json}")
            if feat_names is not None:
                print("[stats] feature_names[0:5] =", feat_names[:5])

    if is_rank0(rank):
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(args.out_dir, "train_args.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    # dataset sizes: ensure each rank has same steps
    train_samples_total = args.steps_per_epoch * args.batch_size * (world_size if distributed else 1)
    val_samples_total = args.val_steps * args.batch_size * (world_size if distributed else 1)

    train_ds = AMSUARandomPatchDataset(args.data_dir, split["train"],
                                      args.patch_h, args.patch_w,
                                      train_samples_total,
                                      in_channels=args.in_channels,
                                      fill_invalid=args.fill_invalid,
                                      min_valid_frac=args.min_valid_frac,
                                      mean=mean, std=std, norm_clamp=args.norm_clamp)
    val_ds = AMSUARandomPatchDataset(args.data_dir, split["val"],
                                    args.patch_h, args.patch_w,
                                    val_samples_total,
                                    in_channels=args.in_channels,
                                    fill_invalid=args.fill_invalid,
                                    min_valid_frac=args.min_valid_frac,
                                    mean=mean, std=std, norm_clamp=args.norm_clamp)

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                              worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, shuffle=False,
                            num_workers=max(0, min(args.num_workers, 2)),
                            pin_memory=(device.type=="cuda"),
                            worker_init_fn=worker_init_fn, drop_last=False)

    # model
    base_model = VAEformerLite(args.in_channels, args.patch_size, args.embed_dim,
                              args.latent_dim, args.depth, args.heads).to(device)
    configure_stage_trainables(base_model, args.stage)

    # resume
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", best_val))
        if is_rank0(rank):
            print(f"[resume] {args.resume} epoch={start_epoch}")

    # ddp wrap
    model = base_model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, base_model.parameters()),
                              lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(args.amp and device.type=="cuda"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for it, (x, m) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).float()
            m = m.to(device, non_blocking=True).float()

            optim.zero_grad(set_to_none=True)

            # 1) forward with autocast
            with autocast(device_type="cuda", enabled=(args.amp and device.type=="cuda")):
                out = model(x, compute_likelihood=(args.stage != "pretrain"))
                x_hat = out["x_hat"]

            # 2) loss in FP32 (关键：避免溢出)
            mse = masked_mse_fp32(x_hat, x, m)
            if args.stage == "pretrain":
                loss = mse
            else:
                bpp = compute_bpp_fp32({"y": out["y_likelihood"], "z": out["z_likelihood"]},
                                      x.shape[-2], x.shape[-1])
                loss = args.lambda_rd * mse + bpp

            if not torch.isfinite(loss):
                if is_rank0(rank):
                    print("[FATAL] loss is non-finite!")
                    print("x finite ratio:", torch.isfinite(x).float().mean().item(),
                          "mask valid frac:", m.mean().item(),
                          "x min/max:", float(x.min().item()), float(x.max().item()))
                    if mean is None:
                        print("TIP: please compute stats and run with --stats_json for normalization.")
                raise RuntimeError("Non-finite loss")

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            if is_rank0(rank) and (it + 1) % 50 == 0:
                if args.stage == "pretrain":
                    print(f"[train][epoch {epoch+1}/{args.epochs} step {it+1}] mse={mse.item():.6f} loss={loss.item():.6f}")
                else:
                    print(f"[train][epoch {epoch+1}/{args.epochs} step {it+1}] mse={mse.item():.6f} bpp={bpp.item():.6f} loss={loss.item():.6f}")

            if (it + 1) >= args.steps_per_epoch:
                break

        valm = eval_loop(model, val_loader, args.stage, args.lambda_rd, device, distributed)
        if is_rank0(rank):
            print(f"[val][epoch {epoch+1}] loss={valm['loss']:.6f} mse={valm['mse']:.6f} bpp={valm['bpp']:.6f}")

            ckpt_path = os.path.join(args.out_dir, f"ckpt_{args.stage}_epoch{epoch+1}.pt")
            torch.save({"model": base_model.state_dict(),
                        "epoch": epoch+1,
                        "best_val": best_val}, ckpt_path)

            if valm["loss"] < best_val:
                best_val = valm["loss"]
                best_path = os.path.join(args.out_dir, f"best_{args.stage}.pt")
                torch.save({"model": base_model.state_dict(),
                            "epoch": epoch+1,
                            "best_val": best_val}, best_path)

        ddp_barrier(distributed)

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
