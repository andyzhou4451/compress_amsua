#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import xarray as xr

# -----------------------------
# 配置
# -----------------------------
NC_PATH = "1bamua_20250101_t00.nc"

KEEP_MAIN_VARS = ["tmbrs", "channels"]  # 两个都保留
AUX_VARS = ["said", "siid", "fovn", "lsql", "saza", "soza", "hols", "hmsl", "solazi", "bearaz"]

# NaN 填充（网络不能吃 NaN）
FILL_NAN_MODE = "mean"     # "mean" | "zero" | "constant"
FILL_CONSTANT = 0.0

# 是否按 feature 逐通道标准化
DO_NORMALIZE = True
EPS = 1e-6

# 输出文件
OUT_NC = "network_ready_feature.nc"        # 推荐：保留 feature 坐标
OUT_NPZ = "network_ready_feature.npz"      # 训练友好：numpy
OUT_META = "network_ready_feature_meta.json"


# -----------------------------
# 工具函数
# -----------------------------
def print_header(t):
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)


def build_feature_names(ds: xr.Dataset):
    """
    生成 feature 名称列表：
    tmbrs-1..15, channels-1..15, said, ...
    """
    names = []

    # 主变量：展开 channel
    for v in KEEP_MAIN_VARS:
        if v not in ds:
            raise KeyError(f"{v} not found in dataset.")
        da = ds[v]
        if "channel" not in da.dims:
            raise ValueError(f"{v} has no 'channel' dim, dims={da.dims}")
        ch_vals = ds["channel"].values  # [1..15]
        for ch in ch_vals:
            names.append(f"{v}-{int(ch)}")

    # 辅助变量：每个一个 feature
    for v in AUX_VARS:
        if v in ds:
            names.append(v)
        else:
            print(f"[warn] AUX var not found, skip: {v}")

    return names


def to_float32_and_mask(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    m = np.isfinite(x).astype(np.uint8)
    return x, m


def fill_nan(x: np.ndarray, m: np.ndarray, mode="mean", constant=0.0):
    if mode == "zero":
        return np.where(m == 1, x, 0.0).astype(np.float32)
    if mode == "constant":
        return np.where(m == 1, x, constant).astype(np.float32)
    if mode == "mean":
        valid = x[m == 1]
        mean_val = float(valid.mean()) if valid.size > 0 else 0.0
        return np.where(m == 1, x, mean_val).astype(np.float32)
    raise ValueError(f"Unknown FILL_NAN_MODE={mode}")


def normalize_per_feature(X: np.ndarray, M: np.ndarray, eps=1e-6):
    """
    X: (T,F,H,W)
    M: (T,F,H,W)
    逐 feature 用有效点算 mean/std，然后标准化
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

    stats = {
        "mode": "per_feature",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    return Xn.astype(np.float32), stats


# -----------------------------
# 核心：组装 (time, feature, lat, lon)
# -----------------------------
def make_4d_feature_tensor(ds: xr.Dataset):
    """
    返回：
      X: np.ndarray float32 (time, feature, lat, lon)
      M: np.ndarray uint8   (time, feature, lat, lon)
      feature_names: list[str] 长度=feature
      out_da: xarray.DataArray (同 shape，带 feature 坐标，便于保存 nc)
    """
    print_header("Build (time, feature, lat, lon) tensor")

    feature_names = build_feature_names(ds)
    pieces = []

    # 1) tmbrs/channels：把 channel 展开到 feature
    for v in KEEP_MAIN_VARS:
        da = ds[v]  # (time, channel, lat, lon)
        # 转成 (time, feature=channel, lat, lon)
        da2 = da.rename({"channel": "feature"})
        da2 = da2.assign_coords(feature=ds["channel"].values)  # feature=1..15（先用数字）
        pieces.append(da2)

    # concat 后 feature=1..15 + 1..15 会冲突，所以先 concat 再重新赋字符串 feature 坐标
    main_cat = xr.concat(pieces, dim="feature")  # (time, feature=30, lat, lon) 但 feature 值重复无所谓

    # 2) aux：(time, lat, lon) -> (time, feature=1, lat, lon)
    aux_pieces = []
    aux_names = []
    for v in AUX_VARS:
        if v not in ds:
            continue
        a = ds[v]  # (time,lat,lon)
        a2 = a.expand_dims({"feature": [v]})  # feature 直接用字符串
        aux_pieces.append(a2)
        aux_names.append(v)

    if aux_pieces:
        aux_cat = xr.concat(aux_pieces, dim="feature")  # (time, feature=10, lat, lon)
        all_cat = xr.concat([main_cat, aux_cat], dim="feature")  # (time, feature=40, lat, lon)
    else:
        all_cat = main_cat

    # 3) 强制 feature 坐标用你要的名字顺序
    if all_cat.sizes["feature"] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: tensor feature={all_cat.sizes['feature']} "
            f"but names={len(feature_names)}"
        )

    all_cat = all_cat.assign_coords(feature=np.array(feature_names, dtype=object))

    # 4) 转 numpy + mask
    X = all_cat.values  # (T,F,H,W)
    X, M = to_float32_and_mask(X)

    # 5) NaN 填充
    print_header(f"Fill NaNs: {FILL_NAN_MODE}")
    X = fill_nan(X, M, mode=FILL_NAN_MODE, constant=FILL_CONSTANT)

    # 6) 标准化
    norm_stats = None
    if DO_NORMALIZE:
        print_header("Normalize per feature (valid points only)")
        X, norm_stats = normalize_per_feature(X, M, eps=EPS)

    # 7) 做一个带坐标的 DataArray 用于保存 nc
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
            "fill_nan_mode": FILL_NAN_MODE,
            "normalized": str(DO_NORMALIZE),
        }
    )

    meta = {
        "file": NC_PATH,
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "output_shape": list(X.shape),
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "keep_main_vars": KEEP_MAIN_VARS,
        "aux_vars_requested": AUX_VARS,
        "aux_vars_found": aux_names,
        "fill_nan_mode": FILL_NAN_MODE,
        "do_normalize": DO_NORMALIZE,
        "norm_stats": norm_stats,
        "global_attrs": {k: str(v) for k, v in ds.attrs.items()},
    }

    return X, M, feature_names, out_da, meta


# -----------------------------
# 主程序
# -----------------------------
def main():
    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(NC_PATH)

    print_header("Open dataset")
    ds = xr.open_dataset(NC_PATH, engine="netcdf4", mask_and_scale=True, decode_cf=True, chunks="auto")
    print(ds)

    X, M, feature_names, out_da, meta = make_4d_feature_tensor(ds)

    print_header("Final summary")
    print("X:", X.shape, X.dtype)  # (time, feature, lat, lon)
    print("M:", M.shape, M.dtype)
    print("feature[0:10]:", feature_names[:10], "...")

    # 保存 NC（推荐：feature 名字作为坐标保留下来）
    print_header(f"Save NetCDF: {OUT_NC}")
    out_ds = xr.Dataset({"X": out_da, "M": (("time", "feature", "lat", "lon"), M)})
    out_ds["M"].attrs["description"] = "valid mask (1=valid,0=invalid)"
    out_ds.to_netcdf(OUT_NC)

    # 保存 NPZ（训练方便）
    print_header(f"Save NPZ: {OUT_NPZ}")
    np.savez_compressed(
        OUT_NPZ,
        X=X.astype(np.float32),
        M=M.astype(np.uint8),
        feature_names=np.array(feature_names, dtype=object),
        lat=ds["lat"].values.astype(np.float32),
        lon=ds["lon"].values.astype(np.float32),
        time=ds["time"].values.astype("datetime64[ns]"),
    )

    # 保存 meta
    print_header(f"Save META: {OUT_META}")
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
