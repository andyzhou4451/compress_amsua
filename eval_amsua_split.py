#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from amsua_codec.data.datasets import AMSUAEvalPatchDataset
from amsua_codec.data.stats import load_stats
from amsua_codec.models.vaeformer_lite import VAEformerLite
from amsua_codec.training.utils import compute_bpp_fp32, masked_mse_fp32


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

    with open(os.path.join(args.data_dir, args.split_json), "r", encoding="utf-8") as f:
        split = json.load(f)
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
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = VAEformerLite(
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        depth=args.depth,
        heads=args.heads,
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
                bpp = float(
                    compute_bpp_fp32({"y": out["y_likelihood"], "z": out["z_likelihood"]}, x.shape[-2], x.shape[-1]).item()
                )

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
            "batches": n,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[eval] saved json -> {args.out_json}")


if __name__ == "__main__":
    main()
