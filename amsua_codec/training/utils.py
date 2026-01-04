"""Training helpers for DDP and loss computation."""

from __future__ import annotations

import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def is_rank0(rank: int) -> bool:
    return rank == 0


def ddp_barrier(distributed: bool) -> None:
    if distributed:
        torch.distributed.barrier()


def ddp_allreduce_sum(distributed: bool, x: torch.Tensor) -> torch.Tensor:
    if distributed:
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


def masked_mse_fp32(x_hat: torch.Tensor, x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_hat = x_hat.float()
    x = x.float()
    m = m.float()
    diff = x_hat - x
    diff = torch.where(m > 0.5, diff, torch.zeros_like(diff))
    return (diff * diff).sum() / m.sum().clamp_min(eps)


def compute_bpp_fp32(liks, H: int, W: int) -> torch.Tensor:
    total_bits = 0.0
    for p in liks.values():
        p = p.float()
        total_bits = total_bits + (-torch.log2(p)).sum()
    B = next(iter(liks.values())).shape[0]
    return total_bits / (B * H * W)
