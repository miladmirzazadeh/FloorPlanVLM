"""Batch renderer: converts 10,000 floor-plan configs into a YOLO detection dataset.

Reads scenario batches from ``<configs>/render_batches/`` (10 × batch_XXXX.json,
each a JSON array of scenario objects), renders each plan through the frozen
``generator/`` engine, and writes a YOLO-ready dataset under ``<output>/``.

Output layout
-------------
    dataset_10k/
      images/train/plan_XXXXX.png
      images/val/plan_XXXXX.png
      labels/train/plan_XXXXX.txt
      labels/val/plan_XXXXX.txt
      rich_json/plan_XXXXX_rich.json   # flat, not split
      data.yaml
      render_report.json

Usage
-----
    python render_dataset.py [--configs ./configs] [--output ./dataset_10k]
                             [--workers 8] [--seed 42] [--limit N]

``--limit N`` (default 0 = ALL): controller-authorized testing aid — renders only
the first N configs so the controller can smoke-test before a full 10k run.
"""

from __future__ import annotations

# MUST be set before importing matplotlib or any generator module so that
# spawned child processes (Windows uses 'spawn') inherit the correct backend.
import os
os.environ.setdefault("MPLBACKEND", "Agg")
# Pin the hash seed so the engine's per-plan RNG (it seeds off hash(plan_id)) is
# reproducible across spawned workers and reruns: a given plan always renders
# identically while still varying plan-to-plan. Children inherit this on spawn.
os.environ.setdefault("PYTHONHASHSEED", "0")
# One BLAS/OpenMP thread per process. Rendering is many small numpy ops; without
# this each of the N render workers spawns its own BLAS thread pool and the
# processes oversubscribe the cores (no parallel speedup). MUST precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import multiprocessing
import tempfile
import traceback
from typing import Dict, List, Tuple

from generator.scenario_loader import load_scenarios
from generator.layout import FloorPlan
from generator.renderer import render
from generator.exporter import build_openings, export_rich_json, export_yolo, write_data_yaml
from generator import style


# ---------------------------------------------------------------------------
# Deterministic, cross-process-stable train/val split
# ---------------------------------------------------------------------------

def split_for(plan_id: str, seed: int) -> str:
    """Return 'val' (~15 %) or 'train' (~85 %) deterministically via MD5."""
    h = hashlib.md5(f"{seed}:{plan_id}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if frac < 0.15 else "train"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(root: str) -> Dict[str, str]:
    paths = {
        "img_train": os.path.join(root, "images", "train"),
        "img_val":   os.path.join(root, "images", "val"),
        "lbl_train": os.path.join(root, "labels", "train"),
        "lbl_val":   os.path.join(root, "labels", "val"),
        "rich":      os.path.join(root, "rich_json"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Per-plan worker — MUST be a top-level function for Windows spawn safety
# ---------------------------------------------------------------------------

def _worker(item: Tuple) -> Dict:
    """Process one floor-plan config.  item = (config_dict, seed, root, max_dpi)."""
    config, seed, root, max_dpi = item
    plan_id = config.get("plan_id", "unknown")
    split = split_for(plan_id, seed)

    png_path  = os.path.join(root, "images",   split,           f"{plan_id}.png")
    txt_path  = os.path.join(root, "labels",   split,           f"{plan_id}.txt")
    rich_path = os.path.join(root, "rich_json",                 f"{plan_id}_rich.json")
    # Render the PNG to a sidecar first; it is atomically renamed into place only
    # AFTER its label/rich files are written, so a PNG on disk ALWAYS implies its
    # labels exist. Keep the .png suffix so matplotlib still infers the format.
    png_tmp   = png_path + ".tmp.png"

    # Resume: skip only when BOTH the PNG and its label exist. A plan interrupted
    # after the PNG but before its label is re-rendered, never skipped — otherwise
    # an unlabeled image would silently poison the dataset.
    if os.path.exists(png_path) and os.path.exists(txt_path):
        return {"plan_id": plan_id, "status": "skipped", "split": split}

    fd, temp_dxf = tempfile.mkstemp(suffix=".dxf")
    os.close(fd)  # close the OS-level fd; ezdxf will open it by path

    try:
        plan = FloorPlan(config)

        # 1. Write DXF (returns ezdxf document)
        doc = plan.write_dxf(temp_dxf)

        # 2. Label-leak check
        violations = [
            e.dxf.layer
            for e in doc.modelspace().query("TEXT MTEXT")
            if e.dxf.layer in style.COMPONENT_LAYERS
        ]

        # 3. Render PNG to the sidecar path
        rcfg = plan.render_cfg
        dpi = int(rcfg.get("dpi", 150))
        if max_dpi:
            dpi = min(dpi, int(max_dpi))      # cap to bound canvas size / time
        w, h, transform = render(
            temp_dxf, png_tmp,
            plan=plan,
            dpi=dpi,
            line_weight_style=rcfg.get("line_weight_style", "standard"),
            monochrome=bool(rcfg.get("monochrome", True)),
            noise_std=float(rcfg.get("noise_std", 0.0)),
        )

        # 4. Build openings and export labels FIRST
        openings, _warns = build_openings(plan, transform, w, h)
        export_rich_json(plan, transform, w, h, rich_path, openings=openings)
        export_yolo(plan, transform, w, h, txt_path, openings=openings)

        # 5. Publish the PNG LAST: a finished .png now guarantees its labels exist.
        os.replace(png_tmp, png_path)

        n_doors   = sum(1 for o in openings if o["type"] == "door")
        n_windows = sum(1 for o in openings if o["type"] == "window")

        return {
            "plan_id":  plan_id,
            "status":   "rendered",
            "split":    split,
            "w":        w,
            "h":        h,
            "doors":    n_doors,
            "windows":  n_windows,
            "leak":     violations,
        }

    except Exception as exc:  # noqa: BLE001 — intentional catch-all
        return {
            "plan_id":   plan_id,
            "status":    "failed",
            "split":     split,
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        }

    finally:
        # Always remove the temporary DXF and any orphaned sidecar PNG (a plan
        # that failed after render leaves png_tmp behind). ezdxf has released the
        # DXF after render. Best-effort; never mask the original exception.
        for tmp in (temp_dxf, png_tmp):
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Batch renderer: 10k floor-plan configs -> YOLO dataset"
    )
    ap.add_argument("--configs",  default="./configs",      help="configs root (contains render_batches/)")
    ap.add_argument("--output",   default="./dataset_10k",  help="dataset output root")
    ap.add_argument("--workers",  type=int, default=8,      help="multiprocessing pool size")
    ap.add_argument("--seed",     type=int, default=42,     help="train/val split seed")
    ap.add_argument(
        "--limit", type=int, default=0,
        help="(testing aid) if >0, render only the first N configs and stop",
    )
    ap.add_argument(
        "--max-dpi", type=int, default=0,
        help="cap render DPI (0 = no cap). The engine samples 80-220 DPI; large "
             "plans at 220 make huge, slow canvases. Capping (e.g. 150) cuts "
             "render time/size a lot. rich_json stays pixel-aligned to the PNG.",
    )
    args = ap.parse_args(argv)

    configs_dir = os.path.abspath(args.configs)
    root        = os.path.abspath(args.output)
    batch_dir   = os.path.join(configs_dir, "render_batches")

    # ---- Load all scenarios in the parent process once ----
    print(f"Loading scenarios from {batch_dir} …")
    configs, conv_warnings = load_scenarios(batch_dir)
    print(f"  Loaded {len(configs)} configs, {len(conv_warnings)} conversion warnings.")

    if args.limit > 0:
        configs = configs[: args.limit]
        print(f"  --limit {args.limit}: using first {len(configs)} configs only.")

    total_configs = len(configs)

    # ---- Create output directory tree ----
    _ensure_dirs(root)

    # ---- Dispatch to pool ----
    work_items = [(cfg, args.seed, root, args.max_dpi) for cfg in configs]

    rendered = skipped = failed = 0
    total_doors = total_windows = 0
    leak_plans: List[str] = []
    failures:   List[Dict] = []
    per_split   = {"train": 0, "val": 0}

    print(f"Rendering {total_configs} plans with {args.workers} workers …")

    with multiprocessing.Pool(processes=args.workers, maxtasksperchild=200) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, work_items), start=1):
            status = result["status"]
            split  = result.get("split", "train")

            if status == "rendered":
                rendered += 1
                total_doors   += result.get("doors",   0)
                total_windows += result.get("windows", 0)
                per_split[split] += 1
                if result.get("leak"):
                    leak_plans.append(result["plan_id"])

            elif status == "skipped":
                skipped += 1
                per_split[split] += 1

            else:  # failed
                failed += 1
                failures.append({
                    "plan_id":   result["plan_id"],
                    "error":     result.get("error", ""),
                    "traceback": result.get("traceback", ""),
                })

            if i % 100 == 0 or i == total_configs:
                print(
                    f"  [{i:>5}/{total_configs}]  "
                    f"rendered={rendered}  skipped={skipped}  failed={failed}"
                )

    # ---- Write data.yaml ----
    write_data_yaml(root)

    # ---- Build and write report ----
    report = {
        "total_configs":           total_configs,
        "rendered":                rendered,
        "failed":                  failed,
        "skipped":                 skipped,
        "per_split":               per_split,
        "total_doors":             total_doors,
        "total_windows":           total_windows,
        "label_leak_violations":   len(leak_plans),
        "label_leak_plans":        leak_plans,
        "conversion_warnings":     len(conv_warnings),
        "failures":                failures,
        "seed":                    args.seed,
        "workers":                 args.workers,
    }
    report_path = os.path.join(root, "render_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ---- Human-readable summary ----
    print()
    print("=" * 60)
    print("RENDER REPORT")
    print("=" * 60)
    print(f"  Total configs   : {total_configs}")
    print(f"  Rendered        : {rendered}")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Failed          : {failed}")
    print(f"  Per-split       : train={per_split['train']}  val={per_split['val']}")
    print(f"  Total doors     : {total_doors}")
    print(f"  Total windows   : {total_windows}")
    print(f"  Label-leak violations: {len(leak_plans)} (must be 0)")
    print(f"  Conversion warnings  : {len(conv_warnings)}")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for f in failures[:5]:
            print(f"    {f['plan_id']}: {f['error']}")
        if len(failures) > 5:
            print(f"    … and {len(failures) - 5} more (see render_report.json)")
    print(f"  Report written  : {report_path}")
    print("=" * 60)

    assert rendered + skipped + failed == total_configs, "Counts do not sum to total!"
    return report


if __name__ == "__main__":
    # Windows requires the entry point to be guarded so spawned workers
    # do not re-execute the top-level launch code.
    multiprocessing.freeze_support()
    main()
