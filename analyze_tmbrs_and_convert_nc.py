#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
import argparse
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xarray as xr


def _load_feature_names(npz_obj):
    if "feature_names" in npz_obj:
        arr = npz_obj["feature_names"]
        # allow_pickle=True 时通常是 object array
        try:
            return [str(x) for x in arr.tolist()]
        except Exception:
            return [str(x) for x in arr]
    return None


def _parse_time_from_filename(path: str):
    # e.g. 1bamua_20250105_t18.npz
    base = os.path.basename(path)
    m = re.search(r"_(\d{8})_t(\d{2})", base)
    if not m:
        return None
    ymd = m.group(1)
    hh = m.group(2)
    try:
        return np.array([np.datetime64(datetime.strptime(ymd + hh, "%Y%m%d%H"))])
    except Exception:
        return None


def _get_lat_lon(npz_obj, H: int, W: int):
    # 如果 npz 里有 lat/lon 就用；否则按 0.25deg (721,1440) 兜底
    if "lat" in npz_obj and "lon" in npz_obj:
        lat = np.asarray(npz_obj["lat"], dtype=np.float32)
        lon = np.asarray(npz_obj["lon"], dtype=np.float32)
        return lat, lon

    # fallback
    lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
    lon = (np.arange(W, dtype=np.float32) * (360.0 / W)).astype(np.float32)
    return lat, lon


def _align_recon_to_orig(orig_names, recon_names, Xr):
    """
    将 recon 的 feature 顺序对齐到 orig 的 feature 顺序。
    若 recon_names 缺失，则假设已经对齐。
    """
    if recon_names is None:
        return Xr, orig_names, True

    if orig_names == recon_names:
        return Xr, orig_names, True

    mp = {name: j for j, name in enumerate(recon_names)}
    T, F, H, W = Xr.shape
    Xr2 = np.empty((T, len(orig_names), H, W), dtype=np.float32)

    missing = []
    for i, name in enumerate(orig_names):
        j = mp.get(name, None)
        if j is None:
            missing.append(name)
            Xr2[:, i, :, :] = np.nan
        else:
            Xr2[:, i, :, :] = Xr[:, j, :, :]

    ok = (len(missing) == 0)
    return Xr2, orig_names, ok


def _tmbrs_indices(names):
    """
    返回 tmbrs 通道的索引列表（按通道号排序），兼容 tmbrs- / tmsbrs-
    """
    idx = []
    for i, n in enumerate(names):
        if n.startswith("tmbrs-") or n.startswith("tmsbrs-"):
            try:
                ch = int(n.split("-")[-1])
            except Exception:
                ch = 9999
            idx.append((ch, i, n))
    idx.sort(key=lambda x: x[0])
    return idx  # list of (ch, i, name)


def _masked_err(orig, recon, mask):
    # err only on valid points, else nan
    valid = (mask > 0.5) & np.isfinite(orig) & np.isfinite(recon)
    err = recon - orig
    err = np.where(valid, err, np.nan).astype(np.float32)
    return err, valid


def _stats_1d(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(count=0, bias=np.nan, rmse=np.nan, mae=np.nan, std=np.nan,
                    p05=np.nan, p50=np.nan, p95=np.nan, min=np.nan, max=np.nan)
    bias = float(np.mean(x))
    rmse = float(np.sqrt(np.mean(x * x)))
    mae = float(np.mean(np.abs(x)))
    std = float(np.std(x))
    p05, p50, p95 = [float(v) for v in np.percentile(x, [5, 50, 95])]
    return dict(count=int(x.size), bias=bias, rmse=rmse, mae=mae, std=std,
                p05=p05, p50=p50, p95=p95, min=float(np.min(x)), max=float(np.max(x)))


def plot_hist_overall(err_all, out_png, title):
    e = err_all[np.isfinite(err_all)]
    if e.size == 0:
        print("[WARN] no finite error to plot:", out_png)
        return
    # robust range
    lo, hi = np.percentile(e, [0.5, 99.5])
    plt.figure(figsize=(8, 5))
    plt.hist(np.clip(e, lo, hi), bins=200)
    plt.title(title)
    plt.xlabel("Error (recon - orig)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("[plot] saved:", out_png)


def plot_hist_per_channel(err_ch_list, ch_nums, out_png):
    # 3x5 for 15 channels
    n = len(err_ch_list)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    plt.figure(figsize=(4*ncol, 3*nrow))
    for k, (err2d, ch) in enumerate(zip(err_ch_list, ch_nums), start=1):
        e = err2d[np.isfinite(err2d)]
        ax = plt.subplot(nrow, ncol, k)
        if e.size == 0:
            ax.set_title(f"ch{ch} (no valid)")
            ax.axis("off")
            continue
        lo, hi = np.percentile(e, [1, 99])
        ax.hist(np.clip(e, lo, hi), bins=120)
        ax.set_title(f"tmbrs ch{ch}")
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("[plot] saved:", out_png)


def plot_global_compare(lat, lon, orig2d, recon2d, err2d, out_png, title_prefix=""):
    # pcolormesh with 1D lat/lon
    # robust ranges
    def _robust_vmin_vmax(a):
        v = a[np.isfinite(a)]
        if v.size == 0:
            return (0.0, 1.0)
        return (float(np.percentile(v, 1)), float(np.percentile(v, 99)))

    vmin_o, vmax_o = _robust_vmin_vmax(orig2d)
    vmin_r, vmax_r = _robust_vmin_vmax(recon2d)

    ev = err2d[np.isfinite(err2d)]
    if ev.size > 0:
        emax = float(np.percentile(np.abs(ev), 99))
    else:
        emax = 1.0

    fig = plt.figure(figsize=(16, 4.8))

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.pcolormesh(lon, lat, orig2d, shading="auto", vmin=vmin_o, vmax=vmax_o)
    ax1.set_title(f"{title_prefix}orig")
    ax1.set_xlabel("lon"); ax1.set_ylabel("lat")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.pcolormesh(lon, lat, recon2d, shading="auto", vmin=vmin_r, vmax=vmax_r)
    ax2.set_title(f"{title_prefix}recon")
    ax2.set_xlabel("lon"); ax2.set_ylabel("lat")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.pcolormesh(lon, lat, err2d, shading="auto", vmin=-emax, vmax=emax)
    ax3.set_title(f"{title_prefix}error (recon-orig)")
    ax3.set_xlabel("lon"); ax3.set_ylabel("lat")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle(title_prefix.strip(), y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[plot] saved:", out_png)


def save_stats_csv(stats_rows, out_csv):
    keys = ["channel", "feature_name", "count", "bias", "rmse", "mae", "std", "p05", "p50", "p95", "min", "max"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in stats_rows:
            w.writerow(r)
    print("[stats] saved:", out_csv)


def recon_to_netcdf(out_nc, lat, lon, time, names, Xr_masked):
    """
    把 recon 的 40 个 feature 拆成:
    - tmbrs(time, channel, lat, lon)
    - channels(time, channel, lat, lon)
    - 其他单层变量(time, lat, lon)
    """
    T, F, H, W = Xr_masked.shape
    coords = {
        "time": time if time is not None else np.arange(T),
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
    }

    # 收集 tmbrs/channels
    tm_list = []
    tm_ch = []
    ch_list = []
    ch_ch = []
    other_vars = {}

    for i, n in enumerate(names):
        if n.startswith("tmbrs-") or n.startswith("tmsbrs-"):
            try:
                c = int(n.split("-")[-1])
            except Exception:
                c = len(tm_ch) + 1
            tm_list.append(Xr_masked[:, i, :, :])
            tm_ch.append(c)
        elif n.startswith("channels-"):
            try:
                c = int(n.split("-")[-1])
            except Exception:
                c = len(ch_ch) + 1
            ch_list.append(Xr_masked[:, i, :, :])
            ch_ch.append(c)
        else:
            other_vars[n] = (("time", "lat", "lon"), Xr_masked[:, i, :, :].astype(np.float32))

    data_vars = {}

    if tm_list:
        order = np.argsort(np.array(tm_ch))
        tm_ch_sorted = np.array(tm_ch)[order].astype(np.int32)
        tm_arr = np.stack([tm_list[k] for k in order], axis=1).astype(np.float32)  # (T, C, H, W)
        data_vars["tmbrs"] = (("time", "channel", "lat", "lon"), tm_arr)
        coords["channel"] = tm_ch_sorted

    if ch_list:
        order2 = np.argsort(np.array(ch_ch))
        ch_arr = np.stack([ch_list[k] for k in order2], axis=1).astype(np.float32)
        # 如果 tmbrs 已经定义了 channel 坐标且长度一致，就共用；否则单独建 channels_channel
        if "channel" in coords and ch_arr.shape[1] == len(coords["channel"]):
            data_vars["channels"] = (("time", "channel", "lat", "lon"), ch_arr)
        else:
            coords["channels_channel"] = np.array(ch_ch)[order2].astype(np.int32)
            data_vars["channels"] = (("time", "channels_channel", "lat", "lon"), ch_arr)

    data_vars.update(other_vars)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["source"] = "reconstructed from VAEformerLite codec"
    ds.attrs["note"] = "masked with original valid mask; invalid points set to NaN"

    # 压缩编码（zlib）
    encoding = {}
    for v in ds.data_vars:
        encoding[v] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.nan,
        }

    try:
        ds.to_netcdf(out_nc, encoding=encoding, engine="netcdf4")
    except Exception:
        ds.to_netcdf(out_nc, encoding=encoding)
    print("[nc] saved:", out_nc)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_npz", required=True)
    ap.add_argument("--recon_npz", required=True)
    ap.add_argument("--out_dir", default="tmbrs_eval_out")
    ap.add_argument("--make_all_channel_maps", action="store_true",
                    help="若不加，只画 1/8/15 三个通道；加上则画 1..15 全部通道")
    ap.add_argument("--out_nc", default=None, help="输出 recon nc 文件名（默认同名 .recon.nc）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d0 = np.load(args.orig_npz, allow_pickle=True)
    X = np.asarray(d0["X"], dtype=np.float32)
    M = np.asarray(d0["M"], dtype=np.float32)
    orig_names = _load_feature_names(d0)
    if orig_names is None:
        raise RuntimeError("orig_npz missing feature_names")

    dr = np.load(args.recon_npz, allow_pickle=True)
    Xr = np.asarray(dr["X_recon"], dtype=np.float32)
    recon_names = _load_feature_names(dr)

    # 对齐 feature 顺序
    Xr_aligned, names, ok_align = _align_recon_to_orig(orig_names, recon_names, Xr)
    if not ok_align:
        print("[WARN] recon feature_names not fully matched orig; missing ones filled as NaN")

    T, F, H, W = X.shape
    lat, lon = _get_lat_lon(d0, H, W)
    time = _parse_time_from_filename(args.orig_npz)

    # 取 tmbrs indices
    tm = _tmbrs_indices(names)
    if len(tm) == 0:
        raise RuntimeError("cannot find tmbrs-* (or tmsbrs-*) in feature_names")

    ch_nums = [c for (c, _, _) in tm]
    tm_idx = [i for (_, i, _) in tm]
    tm_names = [n for (_, _, n) in tm]

    # 计算误差：err = recon - orig（只在 mask 有效点）
    err_ch_list = []
    stats_rows = []
    all_err_flat = []

    for c, i, n in tm:
        orig2d = X[0, i, :, :]
        recon2d = Xr_aligned[0, i, :, :]
        mask2d = M[0, i, :, :]

        err2d, valid = _masked_err(orig2d, recon2d, mask2d)
        err_ch_list.append(err2d)
        all_err_flat.append(err2d.reshape(-1))

        st = _stats_1d(err2d.reshape(-1))
        stats_rows.append({
            "channel": c,
            "feature_name": n,
            **st
        })

    all_err_flat = np.concatenate(all_err_flat, axis=0)
    overall = _stats_1d(all_err_flat)

    # 保存 stats
    out_csv = os.path.join(args.out_dir, "tmbrs_error_stats.csv")
    out_json = os.path.join(args.out_dir, "tmbrs_error_stats.json")
    save_stats_csv(stats_rows, out_csv)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_channel": stats_rows, "overall": overall}, f, ensure_ascii=False, indent=2)
    print("[stats] saved:", out_json)
    print("[stats][overall]", overall)

    # 误差分布图
    plot_hist_overall(all_err_flat, os.path.join(args.out_dir, "tmbrs_error_hist_overall.png"),
                      title="tmbrs error distribution (all channels, valid points only)")
    plot_hist_per_channel(err_ch_list, ch_nums, os.path.join(args.out_dir, "tmbrs_error_hist_per_channel.png"))

    # 全球分布对比图（orig/recon/error）
    if args.make_all_channel_maps:
        plot_channels = ch_nums
    else:
        # 默认画 1/8/15（如果存在）
        want = [1, 8, 15]
        plot_channels = [c for c in want if c in ch_nums]
        if not plot_channels:
            plot_channels = [ch_nums[0]]

    for c in plot_channels:
        k = ch_nums.index(c)
        i = tm_idx[k]
        n = tm_names[k]

        orig2d = X[0, i, :, :]
        recon2d = Xr_aligned[0, i, :, :]
        mask2d = M[0, i, :, :]
        err2d, _ = _masked_err(orig2d, recon2d, mask2d)

        out_png = os.path.join(args.out_dir, f"global_tmbrs_ch{c:02d}.png")
        plot_global_compare(lat, lon, orig2d, recon2d, err2d,
                            out_png=out_png,
                            title_prefix=f"tmbrs ch{c:02d} ({n}) ")

    # 将 recon 转 NetCDF（并用原 mask 把无效点设 NaN）
    Xr_masked = Xr_aligned.copy()
    Xr_masked[M <= 0.5] = np.nan

    if args.out_nc is None:
        base = args.recon_npz.replace(".npz", "")
        out_nc = base + ".nc"
    else:
        out_nc = args.out_nc

    recon_to_netcdf(out_nc, lat, lon, time, names, Xr_masked)


if __name__ == "__main__":
    main()
