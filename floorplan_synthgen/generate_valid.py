#!/usr/bin/env python3
"""Generate N floor plans that each PASS the strict validator, render the same
outputs as the production pipeline (config JSON + rich_json + PNG), and emit a
contact sheet so a human can eyeball that walls close and every room has a door.

Validity is guaranteed by **reject-and-regenerate**: a candidate is built with
``config_generator`` (which no longer injects floating nibs and places doors on
a spanning tree of the room-adjacency graph), then gated through
``validate_plan.validate_plan``; failures are discarded and a fresh layout is
drawn for the same structural identity until one passes (cap = --attempts),
falling back to the guaranteed-simple builder only if needed.

Usage
-----
    python generate_valid.py 50 valid_out
    python generate_valid.py 50 valid_out --seed 20260627 --samples 12

Outputs (under OUT_DIR)
-----------------------
    configs/plan_XXXXX.json          # identical schema to the 10k configs
    rich_json/plan_XXXXX_rich.json   # mm<->px opening records (pixel-aligned)
    images/plan_XXXXX.png            # rendered plan
    valid_samples/contact_sheet.png  # ~12 plans, walls=red, doors=blue overlay
    valid_samples/overlay_XXXXX.png  # per-sample annotated overlays
    generation_report.json           # attempts/plan, pass rate, 50/50 confirmation
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")          # headless render

import json
import random
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import config_generator as C
from validate_plan import validate_plan

from generator.scenario_loader import scenario_to_config
from generator.layout import FloorPlan
from generator.renderer import render
from generator.exporter import build_openings, export_rich_json, export_yolo


# ===========================================================================
# Reject-and-regenerate: produce ONE strictly-valid config
# ===========================================================================
def make_valid_plan(index: int, seed: int, max_attempts: int = 25
                    ) -> Tuple[Optional[dict], int, bool]:
    """Return (config, attempts_used, fell_back).

    ``attempts_used`` counts how many candidate layouts were drawn before one
    passed the strict validator (1 == passed on the first try -> the
    "pre-repair" success case).  ``fell_back`` is True if no random layout
    passed and the guaranteed-simple builder was used."""
    sp = C._structural_params(index, seed)
    for attempt in range(max_attempts):
        rng = random.Random(
            (seed * 7919 + index * 131 + attempt * 101) & 0xFFFFFFFF)
        try:
            d = C._build(index, rng, sp)
        except Exception:
            d = None
        if d is not None and C.validate_plan(d)[0] and validate_plan(d)[0]:
            return d, attempt + 1, False
    # safety net: simple rectangle plan that always validates
    d = C._build_safe(index, seed, sp)
    if d is not None and validate_plan(d)[0]:
        return d, max_attempts, True
    return None, max_attempts, True


# ===========================================================================
# Render a config exactly like the production pipeline
# ===========================================================================
def render_config(config: dict, png_path: str, rich_path: str,
                  txt_path: Optional[str] = None):
    """Render one config to PNG + rich_json (+ optional YOLO txt) through the
    same bridge the 10k used.  Returns (w, h, transform, openings)."""
    warnings: List[str] = []
    eng = scenario_to_config(config, warnings)
    plan = FloorPlan(eng)

    fd, dxf = tempfile.mkstemp(suffix=".dxf")
    os.close(fd)
    try:
        plan.write_dxf(dxf)
        rcfg = plan.render_cfg
        w, h, transform = render(
            dxf, png_path, plan=plan,
            dpi=int(rcfg.get("dpi", 150)),
            line_weight_style=rcfg.get("line_weight_style", "standard"),
            monochrome=bool(rcfg.get("monochrome", True)),
            noise_std=float(rcfg.get("noise_std", 0.0)),
        )
        openings, _ = build_openings(plan, transform, w, h)
        export_rich_json(plan, transform, w, h, rich_path, openings=openings)
        if txt_path:
            export_yolo(plan, transform, w, h, txt_path, openings=openings)
        return w, h, transform, openings
    finally:
        if os.path.exists(dxf):
            os.remove(dxf)


# ===========================================================================
# Contact sheet: walls (red) + doors (blue) overlaid on the rendered PNG
# ===========================================================================
def _overlay_axes(ax, png_path: str, config: dict, transform):
    import matplotlib.image as mpimg
    img = mpimg.imread(png_path)
    ax.imshow(img, cmap="gray")
    # walls -> red outline (polygon traces straight + curved walls alike)
    for w in config["walls"]:
        poly = w.get("polygon") or []
        if len(poly) < 2:
            continue
        pts = [transform.to_px(float(x), float(y)) for x, y in poly]
        pts.append(pts[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="red", linewidth=0.8, alpha=0.9)
    # doors -> blue segment + dot;  windows -> cyan segment (context)
    for o in config["openings"]:
        cat = o.get("category")
        if cat == "door":
            col, lw = "blue", 2.0
        elif cat == "window":
            col, lw = "deepskyblue", 1.2
        else:
            col, lw = "lime", 1.5
        p1 = transform.to_px(float(o["p1"][0]), float(o["p1"][1]))
        p2 = transform.to_px(float(o["p2"][0]), float(o["p2"][1]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col, linewidth=lw)
        if cat == "door":
            c = transform.to_px(float(o["center"][0]), float(o["center"][1]))
            ax.plot([c[0]], [c[1]], marker="o", color="blue", markersize=3)
    ax.set_xticks([])
    ax.set_yticks([])
    nrooms = len(config["rooms"])
    ndoors = sum(1 for o in config["openings"] if o["category"] == "door")
    cw = sum(1 for w in config["walls"] if w.get("arc"))
    ax.set_title(f"{config['id']}  R{nrooms} D{ndoors}"
                 f"{'  +curve' if cw else ''}", fontsize=7)


def build_contact_sheet(samples: List[dict], out_path: str):
    """samples: list of dicts with keys png, config, transform."""
    import math
    import matplotlib.pyplot as plt
    n = len(samples)
    if n == 0:
        return
    cols = 4 if n >= 4 else n
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for k, s in enumerate(samples):
        _overlay_axes(axes[k], s["png"], s["config"], s["transform"])
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Validated synthetic plans  -  walls (red), doors (blue), "
                 "windows (light blue)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_overlay(sample: dict, out_path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    _overlay_axes(ax, sample["png"], sample["config"], sample["transform"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def _dirs(root: str) -> Dict[str, str]:
    paths = {
        "configs": os.path.join(root, "configs"),
        "rich": os.path.join(root, "rich_json"),
        "images": os.path.join(root, "images"),
        "labels": os.path.join(root, "labels"),
        "samples": os.path.join(root, "valid_samples"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate N strictly-valid floor plans + a contact sheet.")
    ap.add_argument("n", type=int, help="number of valid plans to generate")
    ap.add_argument("out_dir", help="output directory")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--start", type=int, default=0,
                    help="first structural index (for distinct identities)")
    ap.add_argument("--attempts", type=int, default=25,
                    help="max regenerate attempts per plan before fallback")
    ap.add_argument("--samples", type=int, default=12,
                    help="how many plans to draw on the contact sheet")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.out_dir)
    paths = _dirs(root)
    t0 = time.time()

    sample_stride = max(1, args.n // max(1, args.samples))
    sample_slots = set(range(0, args.n, sample_stride))
    while len(sample_slots) > args.samples:
        sample_slots.pop()

    records: List[Dict] = []
    samples: List[Dict] = []
    first_try = 0
    fallbacks = 0
    total_attempts = 0
    confirmed_valid = 0

    print(f"Generating {args.n} strictly-valid plans -> {root}")
    for slot in range(args.n):
        index = args.start + slot
        config, attempts, fell_back = make_valid_plan(
            index, args.seed, args.attempts)
        if config is None:
            print(f"  [slot {slot}] FAILED to make a valid plan (index {index})")
            records.append({"slot": slot, "index": index, "status": "failed"})
            continue

        # renumber id to a clean 1..N sequence for this output set
        plan_id = f"plan_{slot + 1:05d}"
        config["id"] = plan_id

        total_attempts += attempts
        if attempts == 1 and not fell_back:
            first_try += 1
        if fell_back:
            fallbacks += 1

        cfg_path = os.path.join(paths["configs"], f"{plan_id}.json")
        png_path = os.path.join(paths["images"], f"{plan_id}.png")
        rich_path = os.path.join(paths["rich"], f"{plan_id}_rich.json")
        txt_path = os.path.join(paths["labels"], f"{plan_id}.txt")
        with open(cfg_path, "w") as fh:
            json.dump(config, fh)

        w = h = None
        transform = None
        try:
            w, h, transform, _ = render_config(config, png_path, rich_path,
                                               txt_path)
        except Exception as e:
            print(f"  [{plan_id}] render error: {e}")

        # FINAL confirmation: re-read the written config and re-validate
        re_ok, _ = validate_plan(json.load(open(cfg_path)))
        confirmed_valid += int(re_ok)

        records.append({
            "slot": slot, "index": index, "plan_id": plan_id,
            "attempts": attempts, "fell_back": fell_back,
            "valid": re_ok, "rooms": len(config["rooms"]),
            "doors": sum(1 for o in config["openings"]
                         if o["category"] == "door"),
            "curved_walls": sum(1 for wl in config["walls"] if wl.get("arc")),
            "footprint_shape": config["metadata"].get("footprint_shape"),
            "w_px": w, "h_px": h,
        })

        if slot in sample_slots and transform is not None:
            samples.append({"png": png_path, "config": config,
                            "transform": transform})

        if (slot + 1) % 10 == 0 or slot + 1 == args.n:
            print(f"  [{slot + 1}/{args.n}]  valid so far: {confirmed_valid}  "
                  f"first-try: {first_try}  fallbacks: {fallbacks}")

    # ---- contact sheet + per-sample overlays ----
    print(f"Building contact sheet from {len(samples)} samples ...")
    cs_path = os.path.join(paths["samples"], "contact_sheet.png")
    try:
        build_contact_sheet(samples, cs_path)
        for s in samples:
            ov = os.path.join(paths["samples"],
                              f"overlay_{s['config']['id']}.png")
            save_overlay(s, ov)
    except Exception as e:
        print(f"  contact-sheet error: {e}")

    made = [r for r in records if r.get("plan_id")]
    elapsed = round(time.time() - t0, 1)
    pre_repair_rate = round(100.0 * first_try / max(1, len(made)), 1)
    report = {
        "requested": args.n,
        "produced": len(made),
        "confirmed_valid": confirmed_valid,
        "all_valid": confirmed_valid == len(made) == args.n,
        "pre_repair_first_try": first_try,
        "pre_repair_pass_rate_pct": pre_repair_rate,
        "fallbacks": fallbacks,
        "avg_attempts": round(total_attempts / max(1, len(made)), 2),
        "max_attempts_used": max((r.get("attempts", 0) for r in made),
                                 default=0),
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "contact_sheet": cs_path,
        "plans": records,
    }
    rep_path = os.path.join(root, "generation_report.json")
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 60)
    print("GENERATE-VALID REPORT")
    print("=" * 60)
    print(f"  Requested            : {args.n}")
    print(f"  Produced             : {len(made)}")
    print(f"  Confirmed valid      : {confirmed_valid}/{len(made)}  "
          f"({'ALL PASS' if report['all_valid'] else 'CHECK FAILURES'})")
    print(f"  Pre-repair pass rate : {pre_repair_rate}%  "
          f"(passed on the first layout, no regenerate)")
    print(f"  Avg attempts/plan    : {report['avg_attempts']}  "
          f"(max {report['max_attempts_used']})")
    print(f"  Fallbacks (safe)     : {fallbacks}")
    print(f"  Elapsed              : {elapsed}s")
    print(f"  Contact sheet        : {cs_path}")
    print(f"  Report               : {rep_path}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    main()
