"""Feature tensor construction utilities for AMSU-A gridded data."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


def build_feature_names(ds: xr.Dataset, keep_main_vars: List[str], aux_vars: List[str]) -> List[str]:
    """
    feature 名称顺序：
    tmbrs-1..15, channels-1..15, said..bearaz
    """
    names: List[str] = []

    if "channel" not in ds.coords and "channel" not in ds.dims:
        raise ValueError("Dataset has no 'channel' coordinate/dimension.")

    ch_vals = ds["channel"].values

    for v in keep_main_vars:
        if v not in ds:
            raise KeyError(f"Missing main var '{v}' in file.")
        da = ds[v]
        if "channel" not in da.dims:
            raise ValueError(f"Var '{v}' has no 'channel' dim, dims={da.dims}")
        for ch in ch_vals:
            names.append(f"{v}-{int(ch)}")

    for v in aux_vars:
        if v in ds:
            names.append(v)
        else:
            print(f"  [warn] aux var not found, skip: {v}")

    return names


def to_float32_and_mask(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32, copy=False)
    m = np.isfinite(x).astype(np.uint8)
    return x, m


def fill_nan(x: np.ndarray, m: np.ndarray, mode: str = "mean", constant: float = 0.0) -> np.ndarray:
    if mode == "zero":
        return np.where(m == 1, x, 0.0).astype(np.float32)
    if mode == "constant":
        return np.where(m == 1, x, constant).astype(np.float32)
    if mode == "mean":
        valid = x[m == 1]
        mean_val = float(valid.mean()) if valid.size > 0 else 0.0
        return np.where(m == 1, x, mean_val).astype(np.float32)
    raise ValueError(f"Unknown fill mode: {mode}")


def normalize_per_feature(X: np.ndarray, M: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    X/M: (T,F,H,W)
    逐 feature 只用有效点算 mean/std，再标准化
    """
    _, F, _, _ = X.shape
    mean = np.zeros((F,), dtype=np.float32)
    std = np.ones((F,), dtype=np.float32)

    Xn = X.copy()
    for f in range(F):
        valid = X[:, f][M[:, f] == 1]
        if valid.size == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = float(valid.mean())
            sd = float(valid.std())
            sd = max(sd, eps)
        mean[f] = mu
        std[f] = sd
        Xn[:, f] = (Xn[:, f] - mu) / sd

    stats = {"mean": mean, "std": std}
    return Xn.astype(np.float32), stats


def make_4d_feature_tensor(
    ds: xr.Dataset,
    keep_main_vars: List[str],
    aux_vars: List[str],
    fill_mode: str = "mean",
    fill_constant: float = 0.0,
    do_normalize: bool = False,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, List[str], xr.DataArray, Dict[str, object]]:
    """
    返回：
      X: np.ndarray float32 (time, feature, lat, lon)
      M: np.ndarray uint8   (time, feature, lat, lon)
      feature_names: list[str]
      out_da: xarray.DataArray
      meta: dict
    """
    feature_names = build_feature_names(ds, keep_main_vars, aux_vars)
    pieces = []

    for v in keep_main_vars:
        da = ds[v]
        da2 = da.rename({"channel": "feature"})
        da2 = da2.assign_coords(feature=ds["channel"].values)
        pieces.append(da2)

    main_cat = xr.concat(pieces, dim="feature")

    aux_pieces = []
    aux_names = []
    for v in aux_vars:
        if v not in ds:
            continue
        a = ds[v]
        a2 = a.expand_dims({"feature": [v]})
        aux_pieces.append(a2)
        aux_names.append(v)

    if aux_pieces:
        aux_cat = xr.concat(aux_pieces, dim="feature")
        all_cat = xr.concat([main_cat, aux_cat], dim="feature")
    else:
        all_cat = main_cat

    if all_cat.sizes["feature"] != len(feature_names):
        raise ValueError(
            f"Feature mismatch: tensor={all_cat.sizes['feature']} vs names={len(feature_names)}"
        )

    all_cat = all_cat.assign_coords(feature=np.array(feature_names, dtype=object))

    X = all_cat.values
    X, M = to_float32_and_mask(X)
    X = fill_nan(X, M, mode=fill_mode, constant=fill_constant)

    norm_stats: Optional[Dict[str, np.ndarray]] = None
    if do_normalize:
        X, norm_stats = normalize_per_feature(X, M, eps=eps)

    out_da = xr.DataArray(
        X,
        dims=("time", "feature", "lat", "lon"),
        coords={
            "time": ds["time"].values,
            "feature": np.array(feature_names, dtype=object),
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
        },
        name="X",
        attrs={
            "description": "AMSU-A network input tensor (time, feature, lat, lon)",
            "fill_nan_mode": fill_mode,
            "normalized": str(do_normalize),
        },
    )

    meta = {
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "output_shape": list(X.shape),
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "keep_main_vars": keep_main_vars,
        "aux_vars_requested": aux_vars,
        "aux_vars_found": aux_names,
        "fill_nan_mode": fill_mode,
        "do_normalize": do_normalize,
        "norm_stats": None if norm_stats is None else {
            "mean": norm_stats["mean"].tolist(),
            "std": norm_stats["std"].tolist(),
        },
        "global_attrs": {k: str(v) for k, v in ds.attrs.items()},
    }

    return X, M, feature_names, out_da, meta


def save_meta_json(meta: Dict[str, object], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
