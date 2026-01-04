#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import xarray as xr


KEEP_MAIN_VARS = ["tmbrs", "channels"]  # 两个都保留
AUX_VARS = ["said", "siid", "fovn", "lsql", "saza", "soza", "hols", "hmsl", "solazi", "bearaz"]


def build_feature_names(ds: xr.Dataset):
    """
    feature 名称顺序：
    tmbrs-1..15, channels-1..15, said..bearaz
    """
    names = []

    # 主变量按 channel 展开
    if "channel" not in ds.coords and "channel" not in ds.dims:
        raise ValueError("Dataset has no 'channel' coordinate/dimension.")

    ch_vals = ds["channel"].values  # e.g. 1..15

    for v in KEEP_MAIN_VARS:
        if v not in ds:
            raise KeyError(f"Missing main var '{v}' in file.")
        da = ds[v]
        if "channel" not in da.dims:
            raise ValueError(f"Var '{v}' has no 'channel' dim, dims={da.dims}")
        for ch in ch_vals:
            names.append(f"{v}-{int(ch)}")

    # 辅助变量
    for v in AUX_VARS:
        if v in ds:
            names.append(v)
        else:
            # 文件里缺就跳过（也可以改成 raise）
            print(f"  [warn] aux var not found, skip: {v}")

    return names


def to_float32_and_mask(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    m = np.isfinite(x).astype(np.uint8)  # 1 valid, 0 invalid
    return x, m


def fill_nan(x: np.ndarray, m: np.ndarray, mode="mean", constant=0.0):
    """
    网络通常不接受 NaN：用 mask 把 NaN 填掉
    """
    if mode == "zero":
        return np.where(m == 1, x, 0.0).astype(np.float32)
    if mode == "constant":
        return np.where(m == 1, x, constant).astype(np.float32)
    if mode == "mean":
        valid = x[m == 1]
        mean_val = float(valid.mean()) if valid.size > 0 else 0.0
        return np.where(m == 1, x, mean_val).astype(np.float32)
    raise ValueError(f"Unknown fill mode: {mode}")


def normalize_per_feature(X: np.ndarray, M: np.ndarray, eps=1e-6):
    """
    X/M: (T,F,H,W)
    逐 feature 只用有效点算 mean/std，再标准化
    """
    T, F, H, W = X.shape
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


def nc_to_npz(nc_path: str,
              out_path: str,
              fill_mode: str = "mean",
              fill_constant: float = 0.0,
              do_normalize: bool = False):
    """
    单文件：nc -> npz
    输出 npz 内容：
      X: float32 (T,F,H,W)
      M: uint8   (T,F,H,W)
      feature_names: object array (F,)
      lat, lon, time
      （如果 do_normalize=True，还会存 mean/std）
    """
    print(f"[convert] {nc_path} -> {out_path}")

    ds = xr.open_dataset(
        nc_path,
        engine="netcdf4",
        mask_and_scale=True,
        decode_cf=True,
        chunks="auto",
    )

    try:
        # 生成 feature 名字
        feature_names = build_feature_names(ds)

        pieces = []

        # 1) tmbrs/channels: (time,channel,lat,lon) -> rename channel->feature
        for v in KEEP_MAIN_VARS:
            da = ds[v].rename({"channel": "feature"})
            pieces.append(da)

        main_cat = xr.concat(pieces, dim="feature")  # (time, 30, lat, lon)

        # 2) aux: (time,lat,lon) -> expand to (time,1,lat,lon), feature=string
        aux_pieces = []
        for v in AUX_VARS:
            if v not in ds:
                continue
            a = ds[v].expand_dims({"feature": [v]})
            aux_pieces.append(a)

        if aux_pieces:
            aux_cat = xr.concat(aux_pieces, dim="feature")
            all_cat = xr.concat([main_cat, aux_cat], dim="feature")  # (time, 40, lat, lon)
        else:
            all_cat = main_cat

        # 强制 feature 坐标改成你要的字符串顺序
        if all_cat.sizes["feature"] != len(feature_names):
            raise ValueError(
                f"Feature mismatch: tensor={all_cat.sizes['feature']} vs names={len(feature_names)}"
            )
        all_cat = all_cat.assign_coords(feature=np.array(feature_names, dtype=object))

        # 转 numpy + mask
        X = all_cat.values  # (T,F,H,W)
        X, M = to_float32_and_mask(X)

        # 填 NaN
        X = fill_nan(X, M, mode=fill_mode, constant=fill_constant)

        # 标准化（可选）
        norm_stats = None
        if do_normalize:
            X, norm_stats = normalize_per_feature(X, M)
            # norm_stats["mean/std"] 是 numpy array

        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        time = ds["time"].values.astype("datetime64[ns]")

        # 保存
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

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez_compressed(out_path, **save_dict)

    finally:
        ds.close()


def main():
    parser = argparse.ArgumentParser(description="Batch convert AMSU-A gridded nc files to npz (same basename).")
    parser.add_argument("--in_dir", type=str, default=".", help="Input directory containing .nc files (default: .)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same as in_dir)")
    parser.add_argument("--pattern", type=str, default="*.nc", help="Glob pattern (default: *.nc)")
    parser.add_argument("--fill_mode", type=str, default="mean", choices=["mean", "zero", "constant"],
                        help="How to fill NaNs")
    parser.add_argument("--fill_constant", type=float, default=0.0, help="Used when fill_mode=constant")
    parser.add_argument("--normalize", action="store_true", help="Enable per-feature normalization")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output npz already exists")
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir or in_dir

    nc_files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not nc_files:
        raise FileNotFoundError(f"No files match: {os.path.join(in_dir, args.pattern)}")

    print(f"Found {len(nc_files)} files.")

    n_ok, n_fail, n_skip = 0, 0, 0
    for nc_path in nc_files:
        base = os.path.splitext(os.path.basename(nc_path))[0]
        out_path = os.path.join(out_dir, base + ".npz")

        if args.skip_existing and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            n_skip += 1
            continue

        try:
            nc_to_npz(
                nc_path,
                out_path,
                fill_mode=args.fill_mode,
                fill_constant=args.fill_constant,
                do_normalize=args.normalize,
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[fail] {nc_path}: {repr(e)}")

    print(f"\nDone. ok={n_ok}, fail={n_fail}, skip={n_skip}")


if __name__ == "__main__":
    main()
