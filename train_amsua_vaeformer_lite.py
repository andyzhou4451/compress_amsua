#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from amsua_codec.data.datasets import AMSUARandomPatchDataset
from amsua_codec.data.splits import load_or_create_split
from amsua_codec.data.stats import compute_stats_from_train, load_stats
from amsua_codec.models.vaeformer_lite import VAEformerLite, configure_stage_trainables
from amsua_codec.training.distributed import init_distributed
from amsua_codec.training.utils import (
    compute_bpp_fp32,
    ddp_allreduce_sum,
    ddp_barrier,
    is_rank0,
    masked_mse_fp32,
    set_seed,
    worker_init_fn,
)


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
        args.seed, distributed, rank, ddp_barrier, is_rank0
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
