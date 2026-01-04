"""NetCDF to NPZ conversion utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from .feature_tensor import build_feature_names, fill_nan, normalize_per_feature, to_float32_and_mask


def nc_to_npz(
    nc_path: str,
    out_path: str,
    keep_main_vars: List[str],
    aux_vars: List[str],
    fill_mode: str = "mean",
    fill_constant: float = 0.0,
    do_normalize: bool = False,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    单文件：nc -> npz
    返回保存内容 dict 与 norm stats（如果启用）
    """
    ds = xr.open_dataset(
        nc_path,
        engine="netcdf4",
        mask_and_scale=True,
        decode_cf=True,
        chunks="auto",
    )

    try:
        feature_names = build_feature_names(ds, keep_main_vars, aux_vars)

        pieces = []
        for v in keep_main_vars:
            da = ds[v].rename({"channel": "feature"})
            pieces.append(da)

        main_cat = xr.concat(pieces, dim="feature")

        aux_pieces = []
        for v in aux_vars:
            if v not in ds:
                continue
            a = ds[v].expand_dims({"feature": [v]})
            aux_pieces.append(a)

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
            X, norm_stats = normalize_per_feature(X, M)

        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        time = ds["time"].values.astype("datetime64[ns]")

        save_dict = {
            "X": X.astype(np.float32),
            "M": M.astype(np.uint8),
            "feature_names": np.array(feature_names, dtype=object),
            "lat": lat,
            "lon": lon,
            "time": time,
        }
        if norm_stats is not None:
            save_dict["mean"] = norm_stats["mean"].astype(np.float32)
            save_dict["std"] = norm_stats["std"].astype(np.float32)

        return save_dict, norm_stats
    finally:
        ds.close()
