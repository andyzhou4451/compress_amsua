"""Distributed initialization utilities."""

from __future__ import annotations

import os
import torch


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    return int(v) if v is not None else default


def init_distributed(args):
    world_size = _env_int("WORLD_SIZE", -1)
    rank = _env_int("RANK", -1)
    local_rank = _env_int("LOCAL_RANK", -1)

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
