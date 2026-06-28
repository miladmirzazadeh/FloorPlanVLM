#!/usr/bin/env python3
"""One-command synthetic floor-plan dataset generator.

Two resumable stages:
  1. generate **validity-gated** configs  (config_generator.py -> JSON + shards)
  2. render configs -> PNG + rich_json + YOLO labels  (render_dataset.py)

Every emitted plan passes the strict validator (validate_plan.py): watertight
walls, no floating walls, every room reachable through a door, no wall crossing
an opening, sensible opening widths.

Designed for Kaggle / unattended background runs: it is **resumable** -- a PNG
on disk always implies its label exists, so re-running continues where it left
off (e.g. after a 12-hour Kaggle commit times out, just run again).

    python generate_dataset.py --count 30000 --output /kaggle/working/dataset
    python generate_dataset.py --count 30000 --output /kaggle/working/dataset --workers 4
"""

from __future__ import annotations

import argparse
import os
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=30000,
                    help="number of plans to produce (default 30000)")
    ap.add_argument("--output", default="./dataset_valid",
                    help="dataset root (images/, labels/, rich_json/)")
    ap.add_argument("--configs", default=None,
                    help="configs dir (default: <output>/configs)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (0 = auto = cpu count, max 8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0,
                    help="first plan index — for sharding across parallel runs "
                         "(shard k: --start k*count, distinct --output each).")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--max-dpi", type=int, default=0,
                    help="cap render DPI (0 = engine's native 80-220). Capping "
                         "(e.g. 150) roughly halves render time + file size; "
                         "rich_json stays pixel-aligned. Recommended on Kaggle.")
    ap.add_argument("--skip-config", action="store_true",
                    help="reuse existing configs, only (re)render")
    ap.add_argument("--skip-render", action="store_true",
                    help="only generate configs, do not render")
    args = ap.parse_args(argv)

    # Headless + reproducible + no BLAS thread oversubscription across render
    # workers (must be set before matplotlib / numpy / engine import)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    workers = args.workers if args.workers > 0 else min(8, os.cpu_count() or 1)
    out = os.path.abspath(args.output)
    cfg_dir = os.path.abspath(args.configs or os.path.join(out, "configs"))
    os.makedirs(out, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    import config_generator
    import render_dataset

    t0 = time.time()

    # ---- Stage 1: validity-gated configs (skip if shards already present) ----
    batch_dir = os.path.join(cfg_dir, "render_batches")
    have_cfg = os.path.isdir(batch_dir) and any(
        f.endswith(".json") for f in os.listdir(batch_dir))
    if args.skip_config or have_cfg:
        print(f"[1/2] configs already present in {cfg_dir} -> skipping")
    else:
        print(f"[1/2] generating {args.count} validity-gated configs "
              f"({workers} workers) -> {cfg_dir}")
        config_generator.main([
            "--count", str(args.count), "--output", cfg_dir,
            "--workers", str(workers), "--seed", str(args.seed),
            "--start", str(args.start),
            "--shard-size", str(args.shard_size)])
    t1 = time.time()

    # ---- Stage 2: render (resumable) ----
    if args.skip_render:
        print("[2/2] --skip-render set -> configs only")
    else:
        print(f"[2/2] rendering -> {out}  ({workers} workers, resumable, "
              f"max_dpi={args.max_dpi or 'native'})")
        render_dataset.main([
            "--configs", cfg_dir, "--output", out,
            "--workers", str(workers), "--seed", str(args.seed),
            "--max-dpi", str(args.max_dpi)])
    t2 = time.time()

    n_png = 0
    img_root = os.path.join(out, "images")
    for r, _d, files in os.walk(img_root):
        n_png += sum(1 for f in files if f.endswith(".png"))
    print("\n" + "=" * 60)
    print(f"DONE  ->  {out}")
    print(f"  rendered PNGs on disk : {n_png}")
    print(f"  config stage          : {t1 - t0:.0f}s")
    print(f"  render stage          : {t2 - t1:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
