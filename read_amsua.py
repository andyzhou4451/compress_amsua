#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import xarray as xr

from amsua_codec.data.feature_tensor import make_4d_feature_tensor, save_meta_json

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


# -----------------------------
# 主程序
# -----------------------------

def main():
    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(NC_PATH)

    print_header("Open dataset")
    ds = xr.open_dataset(NC_PATH, engine="netcdf4", mask_and_scale=True, decode_cf=True, chunks="auto")
    print(ds)

    X, M, feature_names, out_da, meta = make_4d_feature_tensor(
        ds,
        keep_main_vars=KEEP_MAIN_VARS,
        aux_vars=AUX_VARS,
        fill_mode=FILL_NAN_MODE,
        fill_constant=FILL_CONSTANT,
        do_normalize=DO_NORMALIZE,
        eps=EPS,
    )

    meta["file"] = NC_PATH

    print_header("Final summary")
    print("X:", X.shape, X.dtype)
    print("M:", M.shape, M.dtype)
    print("feature[0:10]:", feature_names[:10], "...")

    print_header(f"Save NetCDF: {OUT_NC}")
    out_ds = xr.Dataset({"X": out_da, "M": (("time", "feature", "lat", "lon"), M)})
    out_ds["M"].attrs["description"] = "valid mask (1=valid,0=invalid)"
    out_ds.to_netcdf(OUT_NC)

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

    print_header(f"Save META: {OUT_META}")
    save_meta_json(meta, OUT_META)

    print("Done.")


if __name__ == "__main__":
    main()
