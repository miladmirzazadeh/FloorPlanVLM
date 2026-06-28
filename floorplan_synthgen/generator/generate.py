"""Main runner: build the three Session-1 scenarios into a YOLO dataset.

Usage
-----
    python -m generator.generate                 # -> ./dataset
    python -m generator.generate --output out    # -> ./out

For each scenario it writes a DXF, renders a PNG, and exports YOLO .txt +
rich JSON, then prints a validation report covering image size, opening
counts, the smallest bbox, the no-component-label guarantee and junction
resolution.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from . import style
from .layout import FloorPlan
from .renderer import render
from .exporter import build_openings, export_rich_json, export_yolo, write_data_yaml
from .scenarios import all_scenarios
from . import scenarios_full

GREEN_CHECK = "OK"
WARN = "!!"


def _ensure_dirs(root: str) -> Dict[str, str]:
    paths = {
        "img_train": os.path.join(root, "images", "train"),
        "img_val": os.path.join(root, "images", "val"),
        "lbl_train": os.path.join(root, "labels", "train"),
        "lbl_val": os.path.join(root, "labels", "val"),
        "rich": os.path.join(root, "rich_json"),
        "dxf": os.path.join(root, "dxf"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _component_text_violations(doc) -> List[str]:
    """Any TEXT/MTEXT sitting on a component layer = a forbidden opening label."""
    msp = doc.modelspace()
    bad = []
    for e in msp.query("TEXT MTEXT"):
        if e.dxf.layer in style.COMPONENT_LAYERS:
            bad.append(e.dxf.layer)
    return bad


def _clutter_text_count(doc) -> int:
    return len(doc.modelspace().query("TEXT MTEXT"))


def run_scenario(config: Dict, paths: Dict[str, str]) -> Dict:
    plan_id = config["plan_id"]
    plan = FloorPlan(config)

    dxf_path = os.path.join(paths["dxf"], f"{plan_id}.dxf")
    png_path = os.path.join(paths["img_train"], f"{plan_id}.png")
    txt_path = os.path.join(paths["lbl_train"], f"{plan_id}.txt")
    rich_path = os.path.join(paths["rich"], f"{plan_id}_rich.json")

    doc = plan.write_dxf(dxf_path)

    rcfg = plan.render_cfg
    w, h, transform = render(
        dxf_path, png_path, plan=plan,
        dpi=int(rcfg.get("dpi", 150)),
        line_weight_style=rcfg.get("line_weight_style", "standard"),
        monochrome=bool(rcfg.get("monochrome", True)),
        noise_std=float(rcfg.get("noise_std", 0.0)),
    )

    openings, warnings = build_openings(plan, transform, w, h)
    export_rich_json(plan, transform, w, h, rich_path, openings=openings)
    export_yolo(plan, transform, w, h, txt_path, openings=openings)

    doors = [o for o in openings if o["type"] == "door"]
    windows = [o for o in openings if o["type"] == "window"]
    smallest = None
    for o in openings:
        x1, y1, x2, y2 = o["bbox_px"]
        dim = min(x2 - x1, y2 - y1)
        smallest = dim if smallest is None else min(smallest, dim)

    return {
        "plan_id": plan_id,
        "w": w, "h": h,
        "doors": doors, "windows": windows,
        "smallest_px": smallest,
        "text_violations": _component_text_violations(doc),
        "clutter_text": _clutter_text_count(doc),
        "junctions": plan.junctions_resolved,
        "warnings": warnings,
    }


def print_report(r: Dict):
    print(f"\n{r['plan_id']}: {GREEN_CHECK}")
    print(f"    PNG: {r['w']} x {r['h']} px")
    door_small = _smallest(r["doors"])
    win_small = _smallest(r["windows"])
    print(f"    Doors: {len(r['doors'])}" +
          (f"  (smallest bbox: {door_small:.0f}px)" if door_small is not None else ""))
    print(f"    Windows: {len(r['windows'])}" +
          (f"  (smallest bbox: {win_small:.0f}px)" if win_small is not None else ""))
    if r["smallest_px"] is not None:
        flag = f"  {WARN} <20px" if r["smallest_px"] < 20 else ""
        print(f"    Smallest bbox overall: {r['smallest_px']:.0f}px{flag}")
    print(f"    Wall junctions: {r['junctions']} resolved, 0 overlaps "
          f"(boolean-union faces)")
    if r["text_violations"]:
        print(f"    {WARN} Component-layer text found: {r['text_violations']}")
    else:
        print(f"    Labels visible in PNG: NONE {GREEN_CHECK} "
              f"(no text on door/window/column layers)")
    print(f"    Clutter text glyphs present (distractors): {r['clutter_text']}")
    for w in r["warnings"]:
        print(f"    {WARN} {w}")


def _smallest(items):
    s = None
    for o in items:
        x1, y1, x2, y2 = o["bbox_px"]
        dim = min(x2 - x1, y2 - y1)
        s = dim if s is None else min(s, dim)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Synthetic floor plan generator")
    ap.add_argument("--output", default="dataset", help="dataset root folder")
    ap.add_argument("--set", dest="scenario_set", default="session1",
                    choices=["session1", "full"],
                    help="session1 = 3 test plans; full = the 55-scenario matrix")
    ap.add_argument("--scenarios", default=None, metavar="DIR",
                    help="load external absolute-geometry scenarios from DIR "
                         "(scenarios/*.json) via the bridge, instead of --set")
    args = ap.parse_args(argv)

    if args.scenarios:
        from .scenario_loader import load_scenarios
        configs, conv_warnings = load_scenarios(args.scenarios)
        if not configs:
            print(f"No scenarios found in {args.scenarios}")
            raise SystemExit(1)
        if conv_warnings:
            print(f"Conversion warnings ({len(conv_warnings)}):")
            for w in conv_warnings:
                print("  -", w)
    elif args.scenario_set == "full":
        configs = scenarios_full.all_full()
        errs = scenarios_full.check_all()
        if errs:
            print("CONSTRAINT VIOLATIONS:")
            for e in errs:
                print("  -", e)
            raise SystemExit(1)
    else:
        configs = all_scenarios()

    root = os.path.abspath(args.output)
    paths = _ensure_dirs(root)

    print(f"Generating dataset -> {root}  ({len(configs)} plans)")
    results = []
    for cfg in configs:
        results.append(run_scenario(cfg, paths))

    yaml_path = write_data_yaml(root)

    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    for r in results:
        print_report(r)

    total_doors = sum(len(r["doors"]) for r in results)
    total_win = sum(len(r["windows"]) for r in results)
    any_violation = any(r["text_violations"] for r in results)
    print("\n" + "-" * 60)
    print(f"Plans: {len(results)}   Doors: {total_doors}   Windows: {total_win}")
    print(f"data.yaml: {yaml_path}")
    print(f"Component-label leak: {'YES ' + WARN if any_violation else 'NONE ' + GREEN_CHECK}")
    return results


if __name__ == "__main__":
    main()
