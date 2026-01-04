#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np

from amsua_codec.data.nc_to_npz import nc_to_npz

KEEP_MAIN_VARS = ["tmbrs", "channels"]  # 两个都保留
AUX_VARS = ["said", "siid", "fovn", "lsql", "saza", "soza", "hols", "hmsl", "solazi", "bearaz"]


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
            save_dict, _ = nc_to_npz(
                nc_path,
                out_path,
                keep_main_vars=KEEP_MAIN_VARS,
                aux_vars=AUX_VARS,
                fill_mode=args.fill_mode,
                fill_constant=args.fill_constant,
                do_normalize=args.normalize,
            )
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            np.savez_compressed(out_path, **save_dict)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[fail] {nc_path}: {repr(e)}")

    print(f"\nDone. ok={n_ok}, fail={n_fail}, skip={n_skip}")


if __name__ == "__main__":
    main()
