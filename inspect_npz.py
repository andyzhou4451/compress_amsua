#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_npz.py
读取并尽可能详细地检查 .npz 文件结构（zip 成员 + .npy 头信息 + 可选数组统计）
适用于后续将数据输入深度神经网络前的“摸底”。

用法示例：
  python inspect_npz.py 1bamua_20250101_t00.npz
  python inspect_npz.py 1bamua_20250101_t00.npz --max_load_mb 1024 --sample_elems 200000
  python inspect_npz.py 1bamua_20250101_t00.npz --allow_pickle --report report.json
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter
from pprint import pformat

import numpy as np


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.2f}{u}" if u != "B" else f"{int(f)}B"
        f /= 1024.0
    return f"{f:.2f}PB"


def sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_npy_header_from_fileobj(fp):
    """
    从 .npy 文件流读取 header（不读取整个数组数据）
    返回：dict(version, shape, fortran_order, dtype_str)
    """
    from numpy.lib import format as npformat

    version = npformat.read_magic(fp)
    # 兼容不同 numpy 版本
    if version == (1, 0):
        shape, fortran_order, dtype = npformat.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran_order, dtype = npformat.read_array_header_2_0(fp)
    else:
        # 某些版本会有 (3,0)；内部函数可兜底
        if hasattr(npformat, "read_array_header_3_0") and version == (3, 0):
            shape, fortran_order, dtype = npformat.read_array_header_3_0(fp)
        elif hasattr(npformat, "_read_array_header"):
            shape, fortran_order, dtype = npformat._read_array_header(fp, version)
        else:
            raise RuntimeError(f"无法解析 .npy header：magic version={version}")

    return {
        "version": tuple(version),
        "shape": tuple(shape),
        "fortran_order": bool(fortran_order),
        "dtype": str(dtype),
        "_dtype_obj": dtype,  # 仅内部使用，写报告时转成 str
    }


def safe_to_json(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (tuple, list)):
        return [safe_to_json(i) for i in x]
    if isinstance(x, dict):
        return {str(k): safe_to_json(v) for k, v in x.items()}
    return str(x)


def summarize_object_array(arr, max_items=20):
    # object 数组：统计元素类型分布 + 简单预览
    flat = arr.ravel()
    n = flat.size
    take = min(n, max_items)
    types = [type(flat[i]).__name__ for i in range(take)]
    type_counts = Counter(types)
    samples = []
    for i in range(take):
        v = flat[i]
        s = repr(v)
        if len(s) > 200:
            s = s[:200] + "…"
        samples.append(s)
    return {
        "object_sampled_items": int(take),
        "object_type_counts_in_sample": dict(type_counts),
        "object_samples_repr": samples[: min(10, len(samples))],
    }


def numeric_stats_from_sample(sample: np.ndarray, percentiles):
    # sample 需是一维
    sample = sample.astype(np.float64, copy=False)
    finite = np.isfinite(sample)
    finite_vals = sample[finite]
    out = {
        "sample_size": int(sample.size),
        "finite_count": int(finite_vals.size),
        "nan_count": int(np.isnan(sample).sum()),
        "posinf_count": int(np.isposinf(sample).sum()),
        "neginf_count": int(np.isneginf(sample).sum()),
    }
    if finite_vals.size > 0:
        out.update({
            "min": float(np.min(finite_vals)),
            "max": float(np.max(finite_vals)),
            "mean": float(np.mean(finite_vals)),
            "std": float(np.std(finite_vals)),
            "abs_max": float(np.max(np.abs(finite_vals))),
        })
        qs = np.percentile(finite_vals, percentiles).tolist()
        out["percentiles"] = {str(p): float(q) for p, q in zip(percentiles, qs)}
        # 稀疏度粗略：0 的占比（只对有限值统计）
        out["zero_fraction_in_finite"] = float(np.mean(finite_vals == 0.0))
    return out


def summarize_array(arr: np.ndarray, sample_elems: int, percentiles, preview_slices: int = 3):
    info = {}
    info["type"] = type(arr).__name__
    info["dtype"] = str(arr.dtype)
    info["ndim"] = int(arr.ndim)
    info["shape"] = tuple(arr.shape)
    info["size"] = int(arr.size)
    info["itemsize"] = int(arr.itemsize)
    info["nbytes"] = int(arr.nbytes)
    info["nbytes_human"] = human_bytes(arr.nbytes)

    # contiguous
    try:
        flags = arr.flags
        info["flags"] = {
            "C_CONTIGUOUS": bool(flags["C_CONTIGUOUS"]),
            "F_CONTIGUOUS": bool(flags["F_CONTIGUOUS"]),
            "OWNDATA": bool(flags["OWNDATA"]),
            "WRITEABLE": bool(flags["WRITEABLE"]),
            "ALIGNED": bool(flags["ALIGNED"]),
        }
    except Exception:
        pass

    # structured dtype
    if arr.dtype.fields is not None:
        info["structured_fields"] = {k: str(v[0]) for k, v in arr.dtype.fields.items()}

    kind = arr.dtype.kind
    info["dtype_kind"] = kind

    # 预览（不把巨大数据全打印）
    def preview(a):
        s = repr(a)
        if len(s) > 500:
            s = s[:500] + "…"
        return s

    # 分情况统计
    if kind in ("i", "u", "f", "c"):
        # 数值/复数
        # 采样：对大数组只取部分元素做统计
        flat = arr.ravel()
        if flat.size <= sample_elems:
            sample = flat
            info["sampling"] = {"mode": "all", "sample_elems": int(flat.size)}
        else:
            rng = np.random.default_rng(0)
            idx = rng.choice(flat.size, size=sample_elems, replace=False)
            sample = flat[idx]
            info["sampling"] = {"mode": "random", "sample_elems": int(sample_elems), "rng_seed": 0}

        if kind == "c":
            # 复数：分别对实部/虚部统计
            info["numeric_stats_real"] = numeric_stats_from_sample(np.real(sample), percentiles)
            info["numeric_stats_imag"] = numeric_stats_from_sample(np.imag(sample), percentiles)
        else:
            info["numeric_stats"] = numeric_stats_from_sample(sample, percentiles)

        # 小整数数组：给一个近似 unique 计数（只对 sample）
        if kind in ("i", "u") and sample.size > 0:
            # 若取样过大 unique 也会重
            sub = sample[: min(sample.size, 200000)]
            vals, cnts = np.unique(sub, return_counts=True)
            if vals.size <= 50:
                info["unique_values_in_sample"] = {str(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}
            else:
                info["unique_count_in_sample"] = int(vals.size)

    elif kind == "b":
        # 布尔
        flat = arr.ravel()
        if flat.size <= sample_elems:
            sample = flat
            info["sampling"] = {"mode": "all", "sample_elems": int(flat.size)}
        else:
            rng = np.random.default_rng(0)
            idx = rng.choice(flat.size, size=sample_elems, replace=False)
            sample = flat[idx]
            info["sampling"] = {"mode": "random", "sample_elems": int(sample_elems), "rng_seed": 0}
        info["bool_true_fraction_in_sample"] = float(np.mean(sample))

    elif kind in ("S", "U"):
        # 字符串
        flat = arr.ravel()
        take = min(flat.size, 10)
        samples = [str(flat[i]) for i in range(take)]
        info["string_samples"] = samples

    elif kind in ("M", "m"):
        # datetime64 / timedelta64
        flat = arr.ravel()
        take = min(flat.size, sample_elems)
        sub = flat[:take]
        try:
            info["time_min"] = str(np.min(sub))
            info["time_max"] = str(np.max(sub))
        except Exception as e:
            info["time_stat_error"] = repr(e)

    elif kind == "O":
        # object（可能涉及 pickle）
        info.update(summarize_object_array(arr, max_items=20))

    else:
        info["note"] = f"未特别处理的 dtype kind: {kind}"

    # 形状预览：取前几行/块
    try:
        if arr.ndim == 0:
            info["preview"] = preview(arr.item())
        elif arr.ndim == 1:
            info["preview"] = preview(arr[: min(arr.shape[0], 20)])
        else:
            # 多维：只取前若干 slice
            slicer = tuple(slice(0, min(s, preview_slices)) for s in arr.shape)
            info["preview"] = preview(arr[slicer])
            info["preview_slice"] = str(slicer)
    except Exception as e:
        info["preview_error"] = repr(e)

    return info


def guess_dnn_roles(arr_info: dict):
    """
    基于 dtype/shape 的粗略角色猜测：features / labels / meta
    """
    kind = arr_info.get("dtype_kind")
    ndim = arr_info.get("ndim", 0)
    shape = arr_info.get("shape", ())
    nbytes = arr_info.get("nbytes", 0)

    # 只是一种启发式
    if kind in ("f", "c") and ndim >= 2:
        return "features候选(连续数值，多维)"
    if kind in ("i", "u") and (ndim == 1 or (ndim == 2 and (shape[1] <= 10 if len(shape) > 1 else False))):
        return "labels/索引候选(整数，一维/小二维)"
    if kind == "b" and ndim <= 2:
        return "mask/labels候选(布尔)"
    if kind in ("S", "U", "O"):
        return "meta/辅助信息候选(字符串/对象)"
    if kind in ("M", "m"):
        return "时间戳/时间间隔候选(datetime/timedelta)"
    if nbytes < 1024:
        return "小数组/超参数候选"
    return "未确定"


def main():
    ap = argparse.ArgumentParser(description="Inspect .npz structure in a very detailed way")
    ap.add_argument("1bamua_20250101_t00.npz", help="Path to .npz file, e.g. 1bamua_20250101_t00.npz")
    ap.add_argument("--allow_pickle", action="store_true",
                    help="允许加载 object 数组（可能包含 pickle）。注意：不可信文件不要开！")
    ap.add_argument("--max_load_mb", type=int, default=256,
                    help="每个数组允许实际加载并做统计的最大大小(MB)。超过则只读 header，不加载统计。默认 256MB")
    ap.add_argument("--sample_elems", type=int, default=200000,
                    help="统计时最多采样多少元素(对大数组随机采样)。默认 200000")
    ap.add_argument("--report", type=str, default="",
                    help="可选：把结果保存为 JSON 报告路径")
    ap.add_argument("--no_stats", action="store_true",
                    help="仅输出 zip 成员 + .npy header，不加载数组做统计（更快/更省内存）")
    args = ap.parse_args()

    npz_path = args.npz_path
    if not os.path.isfile(npz_path):
        print(f"[错误] 文件不存在: {npz_path}", file=sys.stderr)
        sys.exit(2)

    file_stat = os.stat(npz_path)
    base_report = {
        "file": {
            "path": os.path.abspath(npz_path),
            "size_bytes": int(file_stat.st_size),
            "size_human": human_bytes(file_stat.st_size),
            "mtime": _dt.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "sha256": sha256_file(npz_path),
        },
        "zip_members": [],
        "npy_headers": {},
        "arrays": {},
        "dnn_role_hints": {},
    }

    print("=" * 90)
    print("[1] NPZ 文件信息")
    print(pformat(base_report["file"]))
    print("=" * 90)

    # 1) zip 成员级别信息 + .npy header
    print("[2] Zip 成员列表 + (若为 .npy) 头信息解析")
    with zipfile.ZipFile(npz_path, "r") as zf:
        members = zf.infolist()
        for zi in members:
            member_info = {
                "filename": zi.filename,
                "compress_type": zi.compress_type,
                "compress_size": int(zi.compress_size),
                "file_size": int(zi.file_size),
                "compress_size_human": human_bytes(zi.compress_size),
                "file_size_human": human_bytes(zi.file_size),
                "CRC": hex(zi.CRC),
                "date_time": str(zi.date_time),
            }
            base_report["zip_members"].append(member_info)

            # 打印成员行
            print(f"- {zi.filename}")
            print(f"  压缩后: {human_bytes(zi.compress_size)} | 解压后: {human_bytes(zi.file_size)} | CRC: {hex(zi.CRC)} | time: {zi.date_time}")

            # 如果是 .npy，读 header（不加载数组数据）
            if zi.filename.endswith(".npy"):
                try:
                    with zf.open(zi, "r") as fp:
                        hdr = read_npy_header_from_fileobj(fp)
                    key = zi.filename[:-4]  # 去掉 .npy；注意如果内部有目录，也会保留目录名
                    base_report["npy_headers"][key] = {
                        "member_filename": zi.filename,
                        "version": hdr["version"],
                        "shape": hdr["shape"],
                        "fortran_order": hdr["fortran_order"],
                        "dtype": hdr["dtype"],
                    }
                    print(f"  .npy header -> key='{key}', shape={hdr['shape']}, dtype={hdr['dtype']}, fortran_order={hdr['fortran_order']}, version={hdr['version']}")
                except Exception as e:
                    print(f"  [警告] 解析 .npy header 失败: {repr(e)}")

    print("=" * 90)

    # 2) 如需：加载数组并做更详细统计
    if args.no_stats:
        print("[3] --no_stats 已开启：跳过数组实际加载与统计。")
    else:
        print("[3] 数组级详细统计（可能耗时/耗内存；超过 --max_load_mb 将自动跳过）")
        max_load_bytes = args.max_load_mb * 1024 * 1024

        try:
            # NpzFile 是懒加载的：访问某个 key 才会真正解压读取
            with np.load(npz_path, allow_pickle=args.allow_pickle) as data:
                keys = list(data.files)
                print(f"发现 keys（{len(keys)} 个）: {keys}")

                for k in keys:
                    # 尝试从 header 里拿解压后大小；否则只能加载后看 nbytes
                    hdr = base_report["npy_headers"].get(k)
                    expected_uncompressed = None
                    if hdr is not None:
                        # 从 zip member 表里找 file_size
                        member_name = hdr.get("member_filename")
                        for m in base_report["zip_members"]:
                            if m["filename"] == member_name:
                                expected_uncompressed = m["file_size"]
                                break

                    if expected_uncompressed is not None and expected_uncompressed > max_load_bytes:
                        print(f"\n[跳过统计] key='{k}' 预计解压后大小 {human_bytes(expected_uncompressed)} > 阈值 {human_bytes(max_load_bytes)}")
                        base_report["arrays"][k] = {
                            "skipped": True,
                            "reason": f"expected_uncompressed {expected_uncompressed} > max_load_bytes {max_load_bytes}",
                            "header": hdr,
                        }
                        base_report["dnn_role_hints"][k] = "跳过(过大，仅header)"
                        continue

                    print(f"\n[加载并统计] key='{k}' ...")
                    try:
                        arr = data[k]  # 这里会真正读取解压
                    except ValueError as e:
                        # 常见：object 数组但 allow_pickle=False
                        print(f"  [无法加载] {repr(e)}")
                        base_report["arrays"][k] = {
                            "skipped": True,
                            "reason": f"load_error: {repr(e)}",
                            "header": hdr,
                        }
                        base_report["dnn_role_hints"][k] = "跳过(加载失败)"
                        continue

                    # 如果实际 nbytes 仍然过大，也可跳过（双保险）
                    if arr.nbytes > max_load_bytes:
                        print(f"  [跳过统计] 实际数组大小 {human_bytes(arr.nbytes)} > 阈值 {human_bytes(max_load_bytes)}")
                        base_report["arrays"][k] = {
                            "skipped": True,
                            "reason": f"arr.nbytes {arr.nbytes} > max_load_bytes {max_load_bytes}",
                            "header": hdr,
                            "basic": {
                                "dtype": str(arr.dtype),
                                "shape": tuple(arr.shape),
                                "nbytes": int(arr.nbytes),
                            }
                        }
                        base_report["dnn_role_hints"][k] = "跳过(过大，仅basic)"
                        continue

                    info = summarize_array(
                        arr,
                        sample_elems=args.sample_elems,
                        percentiles=[0, 1, 5, 25, 50, 75, 95, 99, 100],
                        preview_slices=3
                    )
                    role = guess_dnn_roles(info)

                    base_report["arrays"][k] = info
                    base_report["dnn_role_hints"][k] = role

                    print("  --- 统计摘要 ---")
                    print(f"  dtype={info['dtype']} (kind={info.get('dtype_kind')})")
                    print(f"  shape={info['shape']} ndim={info['ndim']} size={info['size']}")
                    print(f"  nbytes={info['nbytes_human']}")
                    if "numeric_stats" in info:
                        ns = info["numeric_stats"]
                        print(f"  数值统计(sample): min={ns.get('min')}, mean={ns.get('mean')}, std={ns.get('std')}, max={ns.get('max')}")
                        print(f"  NaN={ns.get('nan_count')}, +Inf={ns.get('posinf_count')}, -Inf={ns.get('neginf_count')}")
                        print(f"  分位数: {ns.get('percentiles')}")
                    if "numeric_stats_real" in info:
                        print(f"  复数实部统计: {info['numeric_stats_real']}")
                        print(f"  复数虚部统计: {info['numeric_stats_imag']}")
                    if "bool_true_fraction_in_sample" in info:
                        print(f"  bool True 占比(sample): {info['bool_true_fraction_in_sample']}")
                    if "structured_fields" in info:
                        print(f"  structured fields: {info['structured_fields']}")
                    if "string_samples" in info:
                        print(f"  string samples: {info['string_samples']}")
                    if "object_type_counts_in_sample" in info:
                        print(f"  object types(sample): {info['object_type_counts_in_sample']}")
                    print(f"  预览: {info.get('preview')}")
                    print(f"  [DNN角色猜测] {role}")

        except Exception as e:
            print(f"[错误] np.load 读取失败：{repr(e)}", file=sys.stderr)

    print("=" * 90)
    print("[4] DNN 输入候选提示（启发式，不保证准确）")
    for k, role in base_report["dnn_role_hints"].items():
        hdr = base_report["npy_headers"].get(k, {})
        shape = hdr.get("shape") or base_report["arrays"].get(k, {}).get("shape")
        dtype = hdr.get("dtype") or base_report["arrays"].get(k, {}).get("dtype")
        print(f"- {k}: dtype={dtype}, shape={shape} -> {role}")

    print("=" * 90)

    # 3) 保存 JSON 报告
    if args.report:
        report_path = args.report
        # 去掉不可 json 的对象
        cleaned = safe_to_json(base_report)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        print(f"[已保存] JSON 报告 -> {report_path}")

    print("完成。")


if __name__ == "__main__":
    main()
