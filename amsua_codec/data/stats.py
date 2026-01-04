"""Statistics utilities for AMSU-A features."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_stats_from_train(
    data_dir: str,
    train_files: List[str],
    in_channels: int,
    stats_out: str,
    fill_invalid: float = 0.0,
) -> None:
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
        X = d["X"]
        M = d["M"]

        if "feature_names" in d and feat_names is None:
            try:
                feat_names = [str(x) for x in d["feature_names"].tolist()]
            except Exception:
                feat_names = None

        if X.ndim != 4:
            raise ValueError(f"{rel}: X must be (T,F,H,W), got {X.shape}")
        _, F, _, _ = X.shape
        if F != in_channels:
            raise ValueError(f"{rel}: feature={F} != in_channels={in_channels}")

        X = X.astype(np.float64)
        M = M.astype(np.float64)
        M = np.where(np.isfinite(M), M, 0.0)
        M = (M > 0.5)

        finite = np.isfinite(X)
        valid = M & finite

        X0 = np.where(valid, X, fill_invalid)

        sum_ += X0.sum(axis=(0, 2, 3))
        sumsq += (X0 * X0).sum(axis=(0, 2, 3))
        cnt += valid.sum(axis=(0, 2, 3))

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


def load_stats(stats_json: str, in_channels: int) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    with open(stats_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    if mean.shape[0] != in_channels or std.shape[0] != in_channels:
        raise ValueError(f"stats feature mismatch: {mean.shape[0]} vs in_channels={in_channels}")
    return mean, std, d.get("feature_names", None)
