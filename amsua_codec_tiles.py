#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


# -----------------------------
# stats
# -----------------------------
def load_stats(stats_json: str, in_channels: int):
    d = json.load(open(stats_json, "r", encoding="utf-8"))
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    names = d.get("feature_names", None)
    if mean.shape[0] != in_channels:
        raise ValueError("stats in_channels mismatch")
    return mean, std, names


# -----------------------------
# Model (same arch as training)
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
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

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

    @torch.no_grad()
    def encode_yq(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            raise ValueError("tile H/W must be divisible by patch_size")

        x = self.in_norm(x)
        e = self.patch_embed(x)
        Hp, Wp = e.shape[-2], e.shape[-1]
        tok = e.flatten(2).transpose(1, 2)
        pos = get_2d_sincos_pos_embed(self.embed_dim, Hp, Wp, x.device)
        tok = tok + pos
        for blk in self.enc:
            tok = blk(tok)
        e2 = tok.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        y = self.to_y(e2)

        # ✅ 强制 FP32，避免 AMP half 影响 hyperprior conv
        y_q = torch.round(y.float())
        return y_q  # float32 (integer-valued)

    @torch.no_grad()
    def decode_from_yq(self, y_q: torch.Tensor) -> torch.Tensor:
        B, C, Hp, Wp = y_q.shape
        d = self.y_to_embed(y_q.float())
        tok = d.flatten(2).transpose(1, 2)
        pos = get_2d_sincos_pos_embed(self.embed_dim, Hp, Wp, y_q.device)
        tok = tok + pos
        for blk in self.dec:
            tok = blk(tok)
        d2 = tok.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        x_hat = self.patch_unembed(d2)
        return x_hat

    @torch.no_grad()
    def estimate_bpp_from_yq(self, y_q: torch.Tensor) -> float:
        # ✅ 强制 FP32，避免 half/bias mismatch
        y_q = y_q.float()

        z = self.h_a(y_q)
        z_q = torch.round(z)
        params = self.h_s(z_q)
        mean_y, log_scale_y = params.chunk(2, dim=1)
        scale_y = F.softplus(log_scale_y) + 1e-6

        y_lik = gaussian_likelihood(y_q, mean_y.float(), scale_y.float())
        z_scale = (F.softplus(self.z_log_scale).float()[None, :, None, None] + 1e-6)
        z_lik = gaussian_likelihood(z_q.float(), torch.zeros_like(z_q).float(), z_scale)

        return float(compute_bpp_fp32({"y": y_lik, "z": z_lik}, H=y_q.shape[-2], W=y_q.shape[-1]).item())


# -----------------------------
# IO helpers
# -----------------------------
def load_npz_xm(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    M = d["M"]
    names = d["feature_names"].tolist() if "feature_names" in d else None
    return X, M, names

def sanitize_and_normalize(x: np.ndarray, m: np.ndarray,
                           mean: Optional[np.ndarray], std: Optional[np.ndarray],
                           fill_invalid: float = 0.0, norm_clamp: float = 10.0):
    m = np.where(np.isfinite(m), m, 0.0)
    m = (m > 0.5).astype(np.float32)

    finite = np.isfinite(x)
    if not finite.all():
        m = m * finite.astype(np.float32)
        x = np.nan_to_num(x, nan=fill_invalid, posinf=fill_invalid, neginf=fill_invalid).astype(np.float32)

    x = np.where(m > 0.5, x, fill_invalid).astype(np.float32)

    if mean is not None and std is not None:
        x = (x - mean[:, None, None]) / (std[:, None, None] + 1e-6)
        x = np.clip(x, -norm_clamp, norm_clamp)
        x = np.where(m > 0.5, x, 0.0).astype(np.float32)

    return x, m

def denormalize(x_norm: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]):
    if mean is None or std is None:
        return x_norm
    return (x_norm * std[:, None, None] + mean[:, None, None]).astype(np.float32)

def extract_tile_wrap(x: np.ndarray, m: np.ndarray, top: int, left: int, tile_h: int, tile_w: int):
    F, H, W = x.shape
    tile_x = np.zeros((F, tile_h, tile_w), dtype=np.float32)
    tile_m = np.zeros((F, tile_h, tile_w), dtype=np.float32)

    h_avail = max(0, min(tile_h, H - top))
    if h_avail == 0:
        return tile_x, tile_m

    if left + tile_w <= W:
        tile_x[:, :h_avail, :] = x[:, top:top+h_avail, left:left+tile_w]
        tile_m[:, :h_avail, :] = m[:, top:top+h_avail, left:left+tile_w]
    else:
        seg1 = W - left
        seg2 = tile_w - seg1
        tile_x[:, :h_avail, :seg1] = x[:, top:top+h_avail, left:W]
        tile_m[:, :h_avail, :seg1] = m[:, top:top+h_avail, left:W]
        tile_x[:, :h_avail, seg1:] = x[:, top:top+h_avail, 0:seg2]
        tile_m[:, :h_avail, seg1:] = m[:, top:top+h_avail, 0:seg2]

    return tile_x, tile_m

def accumulate_tile_wrap(dst_sum: np.ndarray, dst_w: np.ndarray,
                         tile: np.ndarray, top: int, left: int):
    F, H, W = dst_sum.shape
    tile_h, tile_w = tile.shape[-2], tile.shape[-1]
    h_avail = max(0, min(tile_h, H - top))
    if h_avail == 0:
        return

    if left + tile_w <= W:
        dst_sum[:, top:top+h_avail, left:left+tile_w] += tile[:, :h_avail, :]
        dst_w[:, top:top+h_avail, left:left+tile_w] += 1.0
    else:
        seg1 = W - left
        seg2 = tile_w - seg1
        dst_sum[:, top:top+h_avail, left:W] += tile[:, :h_avail, :seg1]
        dst_w[:, top:top+h_avail, left:W] += 1.0
        dst_sum[:, top:top+h_avail, 0:seg2] += tile[:, :h_avail, seg1:]
        dst_w[:, top:top+h_avail, 0:seg2] += 1.0

def compute_metrics(orig_x: np.ndarray, orig_m: np.ndarray, recon_x: np.ndarray, names=None):
    valid = (orig_m > 0.5) & np.isfinite(orig_x)
    F = orig_x.shape[0]

    rmse, mae, bias, cnt = [], [], [], []
    for c in range(F):
        v = valid[c]
        if v.sum() == 0:
            rmse.append(float("nan")); mae.append(float("nan")); bias.append(float("nan")); cnt.append(0)
            continue
        diff = recon_x[c][v] - orig_x[c][v]
        rmse.append(float(np.sqrt(np.mean(diff * diff))))
        mae.append(float(np.mean(np.abs(diff))))
        bias.append(float(np.mean(diff)))
        cnt.append(int(v.sum()))

    out = {"rmse": rmse, "mae": mae, "bias": bias, "count": cnt, "feature_names": names}
    w = np.asarray(cnt, dtype=np.float64)
    ok = w > 0
    if ok.any():
        out["rmse_mean"] = float(np.nanmean(np.asarray(rmse)[ok]))
        out["mae_mean"] = float(np.nanmean(np.asarray(mae)[ok]))
        out["bias_mean"] = float(np.nanmean(np.asarray(bias)[ok]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["encode", "decode", "compare"])

    # ✅ ckpt 只在 encode/decode 用；compare 不需要
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--stats_json", type=str, default=None)

    ap.add_argument("--in_channels", type=int, default=40)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)

    ap.add_argument("--tile_h", type=int, default=256)
    ap.add_argument("--tile_w", type=int, default=256)

    ap.add_argument("--norm_clamp", type=float, default=10.0)
    ap.add_argument("--fill_invalid", type=float, default=0.0)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")

    # encode
    ap.add_argument("--input_npz", type=str, default=None)
    ap.add_argument("--out_code", type=str, default=None)
    ap.add_argument("--estimate_bpp", action="store_true")

    # decode
    ap.add_argument("--input_code", type=str, default=None)
    ap.add_argument("--out_recon", type=str, default=None)

    # compare
    ap.add_argument("--orig_npz", type=str, default=None)
    ap.add_argument("--recon_npz", type=str, default=None)
    ap.add_argument("--out_json", type=str, default=None)

    args = ap.parse_args()

    # compare：不需要 ckpt / 不加载模型
    if args.mode == "compare":
        if args.orig_npz is None or args.recon_npz is None or args.out_json is None:
            ap.error("compare requires --orig_npz --recon_npz --out_json")

        X, M, names = load_npz_xm(args.orig_npz)
        dr = np.load(args.recon_npz, allow_pickle=True)
        Xr = dr["X_recon"]
        names2 = dr["feature_names"].tolist() if "feature_names" in dr else None
        if names is None:
            names = names2

        if X.shape != Xr.shape:
            raise ValueError(f"shape mismatch: orig {X.shape} vs recon {Xr.shape}")

        metrics = compute_metrics(X[0].astype(np.float32), M[0].astype(np.float32), Xr[0].astype(np.float32), names)

        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"[compare] saved: {args.out_json}")
        print(f"[compare] rmse_mean={metrics.get('rmse_mean')} mae_mean={metrics.get('mae_mean')} bias_mean={metrics.get('bias_mean')}")
        return

    # encode/decode：需要 ckpt
    if args.ckpt is None:
        ap.error("--ckpt is required for encode/decode")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    mean = std = feat_names = None
    if args.stats_json is not None:
        mean, std, feat_names = load_stats(args.stats_json, args.in_channels)

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

    if args.mode == "encode":
        if args.input_npz is None or args.out_code is None:
            ap.error("encode requires --input_npz and --out_code")

        X, M, names = load_npz_xm(args.input_npz)
        if names is None:
            names = feat_names

        T, F, H, W = X.shape
        if F != args.in_channels:
            raise ValueError(f"in_channels mismatch: {F} vs {args.in_channels}")
        if args.tile_h % args.patch_size != 0 or args.tile_w % args.patch_size != 0:
            raise ValueError("tile_h/tile_w must be divisible by patch_size")

        Hp = args.tile_h // args.patch_size
        Wp = args.tile_w // args.patch_size
        n_h = int(math.ceil(H / args.tile_h))
        n_w = int(math.ceil(W / args.tile_w))

        Yq = np.zeros((T, n_h, n_w, args.latent_dim, Hp, Wp), dtype=np.int16)
        bpp_tiles = np.zeros((T, n_h, n_w), dtype=np.float32)

        with torch.no_grad():
            for t in range(T):
                x0 = X[t].astype(np.float32)
                m0 = M[t].astype(np.float32)
                x0, m0 = sanitize_and_normalize(x0, m0, mean, std,
                                                fill_invalid=args.fill_invalid,
                                                norm_clamp=args.norm_clamp)

                for ih in range(n_h):
                    top = ih * args.tile_h
                    for iw in range(n_w):
                        left = iw * args.tile_w
                        tile_x, _ = extract_tile_wrap(x0, m0, top, left, args.tile_h, args.tile_w)
                        xt = torch.from_numpy(tile_x[None]).to(device)

                        with autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                            y_q = model.encode_yq(xt)

                        y_int = y_q.squeeze(0).cpu().to(torch.int32).numpy()
                        if np.max(np.abs(y_int)) <= 32767:
                            Yq[t, ih, iw] = y_int.astype(np.int16)
                        else:
                            Yq = Yq.astype(np.int32)
                            Yq[t, ih, iw] = y_int.astype(np.int32)

                        if args.estimate_bpp:
                            bpp_tiles[t, ih, iw] = model.estimate_bpp_from_yq(y_q)

        np.savez_compressed(
            args.out_code,
            y_q=Yq,
            H=np.int32(H), W=np.int32(W),
            tile_h=np.int32(args.tile_h), tile_w=np.int32(args.tile_w),
            patch_size=np.int32(args.patch_size),
            in_channels=np.int32(args.in_channels),
            embed_dim=np.int32(args.embed_dim),
            latent_dim=np.int32(args.latent_dim),
            depth=np.int32(args.depth),
            heads=np.int32(args.heads),
            mean=mean if mean is not None else np.array([], dtype=np.float32),
            std=std if std is not None else np.array([], dtype=np.float32),
            feature_names=np.array(names if names is not None else [], dtype=object),
            bpp_tiles=bpp_tiles if args.estimate_bpp else np.array([], dtype=np.float32),
        )

        size_bytes = os.path.getsize(args.out_code)
        file_bpp = (size_bytes * 8.0) / (H * W)
        print(f"[encode] saved: {args.out_code}")
        print(f"[encode] code size = {size_bytes/1024/1024:.3f} MB, file_bpp={file_bpp:.4f} bits/gridpoint")
        if args.estimate_bpp:
            print(f"[encode] mean(bpp_tiles) = {float(bpp_tiles.mean()):.6f} (entropy-model estimate)")

    elif args.mode == "decode":
        if args.input_code is None or args.out_recon is None:
            ap.error("decode requires --input_code and --out_recon")

        d = np.load(args.input_code, allow_pickle=True)
        Yq = d["y_q"]
        H = int(d["H"]); W = int(d["W"])
        tile_h = int(d["tile_h"]); tile_w = int(d["tile_w"])
        patch_size = int(d["patch_size"])
        in_channels = int(d["in_channels"])

        mean_code = d["mean"]
        std_code = d["std"]
        names = d["feature_names"].tolist() if "feature_names" in d else None
        if mean_code.size > 0:
            mean = mean_code.astype(np.float32)
            std = std_code.astype(np.float32)

        T, n_h, n_w, _, _, _ = Yq.shape
        recon = np.zeros((T, in_channels, H, W), dtype=np.float32)

        with torch.no_grad():
            for t in range(T):
                sum_ = np.zeros((in_channels, H, W), dtype=np.float32)
                w_ = np.zeros((in_channels, H, W), dtype=np.float32)

                for ih in range(n_h):
                    top = ih * tile_h
                    for iw in range(n_w):
                        left = iw * tile_w
                        y_tile = Yq[t, ih, iw].astype(np.float32)
                        yt = torch.from_numpy(y_tile[None]).to(device)

                        with autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                            x_hat = model.decode_from_yq(yt)

                        tile_rec = x_hat.squeeze(0).cpu().float().numpy()
                        accumulate_tile_wrap(sum_, w_, tile_rec, top, left)

                x_norm = sum_ / np.maximum(w_, 1.0)
                x_rec = denormalize(x_norm,
                                    mean if (mean is not None and mean.size > 0) else None,
                                    std if (std is not None and std.size > 0) else None)
                recon[t] = x_rec

        np.savez_compressed(
            args.out_recon,
            X_recon=recon.astype(np.float32),
            feature_names=np.array(names if names is not None else [], dtype=object)
        )
        print(f"[decode] saved: {args.out_recon}")

    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
