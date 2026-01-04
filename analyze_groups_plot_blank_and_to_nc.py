#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, csv, argparse
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xarray as xr


def load_feature_names(npz_obj):
    if "feature_names" in npz_obj:
        arr = npz_obj["feature_names"]
        try:
            return [str(x) for x in arr.tolist()]
        except Exception:
            return [str(x) for x in arr]
    return None


def parse_time_from_filename(path: str):
    base = os.path.basename(path)
    m = re.search(r"_(\d{8})_t(\d{2})", base)
    if not m:
        return None
    ymd = m.group(1); hh = m.group(2)
    try:
        return np.array([np.datetime64(datetime.strptime(ymd + hh, "%Y%m%d%H"))])
    except Exception:
        return None


def get_lat_lon_from_ref(ref_nc: str):
    ds = xr.open_dataset(ref_nc)
    lat = np.asarray(ds["lat"].values, dtype=np.float32)
    lon = np.asarray(ds["lon"].values, dtype=np.float32)
    time = ds["time"].values if "time" in ds.coords else None
    ds.close()
    if time is not None:
        time = np.asarray(time)
        if time.ndim == 0:
            time = time.reshape(1)
    return lat, lon, time


def get_lat_lon_fallback(H: int, W: int):
    lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
    lon = (np.arange(W, dtype=np.float32) * (360.0 / W)).astype(np.float32)
    return lat, lon


def align_recon_to_orig(orig_names, recon_names, Xr):
    """
    把 recon 的 feature 顺序对齐到 orig 的 feature 顺序
    """
    if recon_names is None or orig_names == recon_names:
        return Xr, True

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
    if not ok:
        print("[WARN] missing recon features:", missing[:10], "...")
    return Xr2, ok


def group_indices(names, group: str):
    """
    group: 'tmbrs' or 'channels'
    return list of (channel_number, feature_index, feature_name), sorted by channel_number
    """
    out = []
    for i, n in enumerate(names):
        if group == "tmbrs":
            if n.startswith("tmbrs-") or n.startswith("tmsbrs-"):
                ch = int(n.split("-")[-1])
                out.append((ch, i, n))
        elif group == "channels":
            if n.startswith("channels-"):
                ch = int(n.split("-")[-1])
                out.append((ch, i, n))
        else:
            raise ValueError(group)
    out.sort(key=lambda x: x[0])
    return out


def masked_arrays(orig2d, recon2d, mask2d):
    valid = (mask2d > 0.5) & np.isfinite(orig2d) & np.isfinite(recon2d)
    o = np.where(valid, orig2d, np.nan).astype(np.float32)
    r = np.where(valid, recon2d, np.nan).astype(np.float32)
    e = (r - o).astype(np.float32)
    return o, r, e


def stats_1d(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(count=0, bias=np.nan, rmse=np.nan, mae=np.nan, std=np.nan,
                    p05=np.nan, p50=np.nan, p95=np.nan, min=np.nan, max=np.nan)
    bias = float(np.mean(x))
    rmse = float(np.sqrt(np.mean(x * x)))
    mae  = float(np.mean(np.abs(x)))
    std  = float(np.std(x))
    p05, p50, p95 = [float(v) for v in np.percentile(x, [5, 50, 95])]
    return dict(count=int(x.size), bias=bias, rmse=rmse, mae=mae, std=std,
                p05=p05, p50=p50, p95=p95, min=float(np.min(x)), max=float(np.max(x)))


def save_stats(stats_rows, out_csv, out_json):
    keys = ["channel", "feature_name", "count", "bias", "rmse", "mae", "std", "p05", "p50", "p95", "min", "max"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in stats_rows:
            w.writerow(r)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats_rows, f, ensure_ascii=False, indent=2)


def extent_from_latlon(lat, lon):
    # 用中心点推边界，避免半格偏移
    dlat = float(abs(lat[1] - lat[0])) if lat.size > 1 else 1.0
    dlon = float(abs(lon[1] - lon[0])) if lon.size > 1 else 1.0
    x0 = float(np.min(lon) - dlon/2)
    x1 = float(np.max(lon) + dlon/2)
    y0 = float(np.min(lat) - dlat/2)
    y1 = float(np.max(lat) + dlat/2)
    origin = "upper" if lat[0] > lat[-1] else "lower"
    return (x0, x1, y0, y1), origin


def plot_hist(err_all, out_png, title):
    e = err_all[np.isfinite(err_all)]
    if e.size == 0:
        return
    lo, hi = np.percentile(e, [0.5, 99.5])
    plt.figure(figsize=(8, 5))
    plt.hist(np.clip(e, lo, hi), bins=200)
    plt.title(title)
    plt.xlabel("error (recon - orig)")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_hist_per_channel(err_list, ch_nums, out_png, title):
    n = len(err_list)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    plt.figure(figsize=(4*ncol, 3*nrow))
    for k, (err2d, ch) in enumerate(zip(err_list, ch_nums), start=1):
        e = err2d[np.isfinite(err2d)]
        ax = plt.subplot(nrow, ncol, k)
        if e.size == 0:
            ax.set_title(f"ch{ch} (no valid)")
            ax.axis("off")
            continue
        lo, hi = np.percentile(e, [1, 99])
        ax.hist(np.clip(e, lo, hi), bins=120)
        ax.set_title(f"ch{ch}")
        ax.grid(True, alpha=0.2)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_global_triplet(lat, lon, orig2d, recon2d, err2d, out_png, title_prefix):
    extent, origin = extent_from_latlon(lat, lon)

    # colormap：NaN 显示白色
    cmap_main = plt.cm.viridis.copy()
    cmap_main.set_bad(color="white")
    cmap_err  = plt.cm.RdBu_r.copy()
    cmap_err.set_bad(color="white")

    # 用“有效点”计算色标范围
    ov = orig2d[np.isfinite(orig2d)]
    rv = recon2d[np.isfinite(recon2d)]
    if ov.size > 0:
        vmin = float(np.percentile(ov, 1)); vmax = float(np.percentile(ov, 99))
    elif rv.size > 0:
        vmin = float(np.percentile(rv, 1)); vmax = float(np.percentile(rv, 99))
    else:
        vmin, vmax = 0.0, 1.0

    ev = err2d[np.isfinite(err2d)]
    emax = float(np.percentile(np.abs(ev), 99)) if ev.size > 0 else 1.0

    fig = plt.figure(figsize=(16, 4.8))

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(orig2d, extent=extent, origin=origin, vmin=vmin, vmax=vmax,
                     cmap=cmap_main, interpolation="nearest", aspect="auto")
    ax1.set_title(f"{title_prefix} orig")
    ax1.set_xlabel("lon"); ax1.set_ylabel("lat")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(recon2d, extent=extent, origin=origin, vmin=vmin, vmax=vmax,
                     cmap=cmap_main, interpolation="nearest", aspect="auto")
    ax2.set_title(f"{title_prefix} recon")
    ax2.set_xlabel("lon"); ax2.set_ylabel("lat")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(err2d, extent=extent, origin=origin, vmin=-emax, vmax=emax,
                     cmap=cmap_err, interpolation="nearest", aspect="auto")
    ax3.set_title(f"{title_prefix} error (recon-orig)")
    ax3.set_xlabel("lon"); ax3.set_ylabel("lat")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def recon_to_netcdf(out_nc, lat, lon, time, names, Xr_masked):
    T, F, H, W = Xr_masked.shape
    coords = {"time": time if time is not None else np.arange(T),
              "lat": lat.astype(np.float32),
              "lon": lon.astype(np.float32)}

    tm_list, tm_ch = [], []
    ch_list, ch_ch = [], []
    other = {}

    for i, n in enumerate(names):
        if n.startswith("tmbrs-") or n.startswith("tmsbrs-"):
            tm_ch.append(int(n.split("-")[-1]))
            tm_list.append(Xr_masked[:, i, :, :])
        elif n.startswith("channels-"):
            ch_ch.append(int(n.split("-")[-1]))
            ch_list.append(Xr_masked[:, i, :, :])
        else:
            other[n] = (("time", "lat", "lon"), Xr_masked[:, i, :, :].astype(np.float32))

    data_vars = {}
    if tm_list:
        order = np.argsort(np.array(tm_ch))
        coords["channel"] = np.array(tm_ch)[order].astype(np.int32)
        tm = np.stack([tm_list[k] for k in order], axis=1).astype(np.float32)
        data_vars["tmbrs"] = (("time", "channel", "lat", "lon"), tm)

    if ch_list:
        order = np.argsort(np.array(ch_ch))
        ch = np.stack([ch_list[k] for k in order], axis=1).astype(np.float32)
        if "channel" in coords and ch.shape[1] == len(coords["channel"]):
            data_vars["channels"] = (("time", "channel", "lat", "lon"), ch)
        else:
            coords["channels_channel"] = np.array(ch_ch)[order].astype(np.int32)
            data_vars["channels"] = (("time", "channels_channel", "lat", "lon"), ch)

    data_vars.update(other)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["source"] = "reconstructed (mask=0 -> NaN)"
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan}
           for v in ds.data_vars}
    try:
        ds.to_netcdf(out_nc, engine="netcdf4", encoding=enc)
    except Exception:
        ds.to_netcdf(out_nc, encoding=enc)
    print("[nc] saved:", out_nc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_npz", required=True)
    ap.add_argument("--recon_npz", required=True)
    ap.add_argument("--out_dir", default="eval_out")
    ap.add_argument("--ref_nc", default=None, help="可选：用原 nc 的 lat/lon/time（更严格一致）")
    ap.add_argument("--groups", default="tmbrs,channels", help="tmbrs,channels 或只写一个")
    ap.add_argument("--plot_all_maps", action="store_true", help="画所有通道地图（否则默认画 1/8/15）")
    ap.add_argument("--out_nc", default=None, help="输出 recon nc（默认 recon 同名 .recon.nc）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d0 = np.load(args.orig_npz, allow_pickle=True)
    X = np.asarray(d0["X"], dtype=np.float32)
    M = np.asarray(d0["M"], dtype=np.float32)
    orig_names = load_feature_names(d0)
    if orig_names is None:
        raise RuntimeError("orig_npz missing feature_names")

    dr = np.load(args.recon_npz, allow_pickle=True)
    Xr = np.asarray(dr["X_recon"], dtype=np.float32)
    recon_names = load_feature_names(dr)

    Xr_aligned, _ = align_recon_to_orig(orig_names, recon_names, Xr)

    T, F, H, W = X.shape

    if args.ref_nc is not None:
        lat, lon, time = get_lat_lon_from_ref(args.ref_nc)
        if time is None:
            time = parse_time_from_filename(args.orig_npz)
    else:
        lat, lon = get_lat_lon_fallback(H, W)
        time = parse_time_from_filename(args.orig_npz)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    # 先把 recon 按原 mask 置 NaN（用于输出 nc）
    Xr_masked_all = Xr_aligned.copy()
    Xr_masked_all[M <= 0.5] = np.nan

    # 分组处理绘图+统计
    for g in groups:
        gdir = os.path.join(args.out_dir, g)
        os.makedirs(gdir, exist_ok=True)

        gi = group_indices(orig_names, g)
        if len(gi) == 0:
            print(f"[WARN] group {g} not found in feature_names")
            continue

        ch_nums = [c for (c, _, _) in gi]
        err_list = []
        stats_rows = []
        all_err = []

        for c, idx, fname in gi:
            orig2d = X[0, idx, :, :]
            recon2d = Xr_aligned[0, idx, :, :]
            mask2d = M[0, idx, :, :]

            o, r, e = masked_arrays(orig2d, recon2d, mask2d)
            err_list.append(e)
            all_err.append(e.reshape(-1))

            st = stats_1d(e.reshape(-1))
            stats_rows.append({"channel": c, "feature_name": fname, **st})

        all_err = np.concatenate(all_err, axis=0)
        overall = stats_1d(all_err)

        save_stats(stats_rows,
                   out_csv=os.path.join(gdir, f"{g}_error_stats.csv"),
                   out_json=os.path.join(gdir, f"{g}_error_stats.json"))

        with open(os.path.join(gdir, f"{g}_overall.json"), "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)

        plot_hist(all_err, os.path.join(gdir, f"{g}_error_hist_overall.png"),
                  title=f"{g} error distribution (all channels, masked)")
        plot_hist_per_channel(err_list, ch_nums,
                              os.path.join(gdir, f"{g}_error_hist_per_channel.png"),
                              title=f"{g} per-channel error hist (masked)")

        # 地图：默认画 1/8/15（存在则画），或画全部
        if args.plot_all_maps:
            plot_channels = ch_nums
        else:
            want = [1, 8, 15]
            plot_channels = [c for c in want if c in ch_nums]
            if not plot_channels:
                plot_channels = [ch_nums[0]]

        for c in plot_channels:
            k = ch_nums.index(c)
            _, idx, fname = gi[k]

            orig2d = X[0, idx, :, :]
            recon2d = Xr_aligned[0, idx, :, :]
            mask2d = M[0, idx, :, :]

            o, r, e = masked_arrays(orig2d, recon2d, mask2d)
            out_png = os.path.join(gdir, f"global_{g}_ch{c:02d}.png")
            plot_global_triplet(lat, lon, o, r, e, out_png, title_prefix=f"{g} ch{c:02d} ({fname})")

        print(f"[DONE] group={g} overall:", overall)

    # 输出 recon nc
    if args.out_nc is None:
        out_nc = args.recon_npz.replace(".npz", "") + ".nc"
    else:
        out_nc = args.out_nc

    recon_to_netcdf(out_nc, lat, lon, time, orig_names, Xr_masked_all)


if __name__ == "__main__":
    main()
