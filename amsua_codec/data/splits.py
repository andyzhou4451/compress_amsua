"""Dataset splitting utilities."""

from __future__ import annotations

import json
import os
import glob
from typing import List, Dict


def list_npz_files(data_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    return [os.path.relpath(f, data_dir) for f in files]


def make_split(files: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[str]]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    import numpy as np

    rng = np.random.RandomState(seed)
    files = files.copy()
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def load_or_create_split(
    data_dir: str,
    pattern: str,
    split_json: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    distributed: bool,
    rank: int,
    ddp_barrier,
    is_rank0,
) -> Dict[str, List[str]]:
    path = os.path.join(data_dir, split_json)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if is_rank0(rank):
        files = list_npz_files(data_dir, pattern)
        if not files:
            raise FileNotFoundError(f"No npz in {data_dir} pattern={pattern}")
        sp = make_split(files, train_ratio, val_ratio, test_ratio, seed)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sp, f, ensure_ascii=False, indent=2)
        print(
            f"[split] created: {path} (train={len(sp['train'])}, val={len(sp['val'])}, test={len(sp['test'])})"
        )

    ddp_barrier(distributed)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
