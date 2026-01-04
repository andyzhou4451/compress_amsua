#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast


def load_stats(stats_json: str, in_channels: int):
    d = json.load(open(stats_json, "r", encoding="utf-8"))
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    names = d.get("feature_names", None)
    if mean.shape[0] != in_channels:
        raise ValueError("stats in_channels mismatch")
    return mean, std, names


def crop_lon_wrap(arr: np.ndarray, top: int, left: int, ph: int, pw: int) -> np.ndarray:
    C, H, W = arr.shape
    patch_lat = arr[:, top:top+ph, :]
    if left + pw <= W:
        return patch_lat[:, :, left:left+pw]
    r = (left + pw) - W
    return np.concatenate([patch_lat[:, :, left:W], patch_lat[:, :, 0:r]], axis=-1)


class AMSUAEvalPatchDataset(Dataset):
    """
    用固定 seed 预生成 (file_idx, t, top, left)，保证评估可复现。
    """
    def __init__(
        self,
        data_dir: str,
        files: List[str],
        patch_h: int,
        patch_w: int,
        num_samples: int,
        seed: int,
        in_channels: int,
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        norm_clamp: float = 10.0,
        fill_invalid: float = 0.0,
    ):
        self.data_dir = data_dir
        self.files = files
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_samples = num_samples
        self.seed = seed
        self.in_channels = in_channels
        self.mean = mean
        self.std = std
        self.norm_clamp = float(norm_clamp)
        self.fill_invalid = float(fill_invalid)

        f0 = os.path.join(data_dir, files[0])
        d0 = np.load(f0, allow_pickle=True)
        X0 = d0["X"]
        T, F, H, W = X0.shape
        if F != in_channels:
            raise ValueError(f"feature mismatch: {F} vs {in_channels}")
        if H < patch_h or W < patch_w:
            raise ValueError("patch larger than data")

        rng = np.random.RandomState(seed)
        self.index = []
        for _ in range(num_samples):
            file_idx = int(rng.randint(0, len(files)))
            t = int(rng.randint(0, T))
            top = int(rng.randint(0, H - patch_h + 1))
            left = int(rng.randint(0, W))  # wrap
            self.index.append((file_idx, t, top, left))

        self._cache_path = None
        self._cache_X = None
        self._cache_M = None

    def __len__(self):
        return self.num_samples

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

    def __getitem__(self, i):
        file_idx, t, top, left = self.index[i]
        rel = self.files[file_idx]
        X, M = self._load(rel)

        x = X[t].astype(np.float32)  # (F,H,W)
        m = M[t].astype(np.float32)

        m = np.where(np.isfinite(m), m, 0.0)
        m = (m > 0.5).astype(np.float32)

        finite = np.isfinite(x)
        if not finite.all():
            m = m * finite.astype(np.float32)
            x = np.nan_to_num(x, nan=self.fill_invalid, posinf=self.fill_invalid, neginf=self.fill_invalid)

        x = np.where(m > 0.5, x, self.fill_invalid).astype(np.float32)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
            x = np.clip(x, -self.norm_clamp, self.norm_clamp)
            x = np.where(m > 0.5, x, 0.0).astype(np.float32)

        x_patch = crop_lon_wrap(x, top, left, self.patch_h, self.patch_w)
        m_patch = crop_lon_wrap(m, top, left, self.patch_h, self.patch_w)

        return torch.from_numpy(x_patch), torch.from_numpy(m_patch)


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

def masked_mse_fp32(x_hat: torch.Tensor, x: torch.Tensor, m: torch.Tensor, eps=1e-6):
    x_hat = x_hat.float()
    x = x.float()
    m = m.float()
    diff = x_hat - x
    diff = torch.where(m > 0.5, diff, torch.zeros_like(diff))
    return (diff * diff).sum() / m.sum().clamp_min(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--split_json", type=str, default="splits_amsua.json")
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--stats_json", type=str, default=None)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--stage", type=str, default="pretrain", choices=["pretrain", "finetune_entropy", "rd_joint"])
    ap.add_argument("--out_json", type=str, default=None)

    ap.add_argument("--num_samples", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--patch_h", type=int, default=256)
    ap.add_argument("--patch_w", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--in_channels", type=int, default=40)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)

    ap.add_argument("--norm_clamp", type=float, default=10.0)
    ap.add_argument("--fill_invalid", type=float, default=0.0)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    split = json.load(open(os.path.join(args.data_dir, args.split_json), "r", encoding="utf-8"))
    files = split[args.split]
    print(f"[split] {args.split} files={len(files)}")

    mean = std = names = None
    if args.stats_json is not None:
        mean, std, names = load_stats(args.stats_json, args.in_channels)
        if names is not None:
            print("[stats] feature_names[0:5] =", names[:5])

    ds = AMSUAEvalPatchDataset(
        data_dir=args.data_dir,
        files=files,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        num_samples=args.num_samples,
        seed=args.seed,
        in_channels=args.in_channels,
        mean=mean,
        std=std,
        norm_clamp=args.norm_clamp,
        fill_invalid=args.fill_invalid,
    )

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    model = VAEformerLite(
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        depth=args.depth,
        heads=args.heads
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    mse_sum = 0.0
    bpp_sum = 0.0
    n = 0

    with torch.no_grad():
        for x, m in dl:
            x = x.to(device, non_blocking=True).float()
            m = m.to(device, non_blocking=True).float()

            with autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                out = model(x, compute_likelihood=(args.stage != "pretrain"))
                x_hat = out["x_hat"]

            mse = masked_mse_fp32(x_hat, x, m).item()
            if args.stage == "pretrain":
                bpp = 0.0
            else:
                bpp = float(compute_bpp_fp32({"y": out["y_likelihood"], "z": out["z_likelihood"]},
                                             x.shape[-2], x.shape[-1]).item())

            mse_sum += mse
            bpp_sum += bpp
            n += 1

    mse_avg = mse_sum / max(n, 1)
    bpp_avg = bpp_sum / max(n, 1)

    print(f"[eval][{args.split}] mse={mse_avg:.6f} bpp={bpp_avg:.6f} batches={n}")

    if args.out_json is not None:
        out = {
            "split": args.split,
            "ckpt": args.ckpt,
            "stage": args.stage,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "mse": mse_avg,
            "bpp": bpp_avg,
            "batches": n
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[eval] saved json -> {args.out_json}")


if __name__ == "__main__":
    main()
