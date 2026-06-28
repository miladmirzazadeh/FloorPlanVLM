"""Label exporter: YOLO .txt + rich JSON + data.yaml.

The PNG carries no opening labels; everything the detector learns from lives
here. Two products per plan:

* ``<plan_id>.txt``       - Ultralytics YOLO: one line per opening,
                            ``<class_id> <cx> <cy> <w> <h>`` normalised 0..1,
                            class 0 = door, 1 = window. A lossy projection.
* ``<plan_id>_rich.json`` - the exact geometry we already know (model-space mm
                            and pixels): hinge / leaf points for doors, glazing
                            endpoints + angle for windows, plus boxes. This is
                            the training signal kept for the later
                            primitive/keypoint model; it costs nothing now.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from . import geometry as g
from .components import Door, Window

CLASS_ID = {"door": 0, "window": 1}
YOLO_MIN_DIM = 0.03  # warn below this normalised size (too small for YOLO)


def _round_pt(p, nd=2):
    return [round(float(p[0]), nd), round(float(p[1]), nd)]


def build_openings(plan, transform, image_w: int, image_h: int) -> Tuple[List[Dict], List[str]]:
    """Return (opening dicts, warnings)."""
    openings: List[Dict] = []
    warnings: List[str] = []

    for comp, kind in plan.opening_records():
        if comp.bbox_model is None:
            continue
        bbox_px = transform.bbox_to_px(comp.bbox_model)
        xmin, ymin, xmax, ymax = bbox_px
        cx = (xmin + xmax) / 2.0 / image_w
        cy = (ymin + ymax) / 2.0 / image_h
        w = (xmax - xmin) / image_w
        h = (ymax - ymin) / image_h
        center_px = transform.to_px(float(comp.center[0]), float(comp.center[1]))

        rec: Dict = {
            "id": comp.id,
            "type": kind,
            "subtype": comp.subtype,
            "clear_opening_mm": round(float(comp.clear), 2),
            "center_px": _round_pt(center_px),
            "bbox_model": [round(float(v), 2) for v in comp.bbox_model],
            "bbox_px": [round(float(v), 2) for v in bbox_px],
            "bbox_normalized": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
        }

        if isinstance(comp, Door):
            rec["swing_direction"] = comp.swing
            rec["max_swing_angle_deg"] = round(float(comp.max_swing), 2)
            if comp.hinge_point is not None:
                hp = comp.hinge_point
                rec["hinge_point_px"] = _round_pt(transform.to_px(float(hp[0]), float(hp[1])))
                rec["hinge_point_mm"] = _round_pt(hp)
            else:
                rec["hinge_point_px"] = None
            if comp.leaf_end is not None:
                le = comp.leaf_end
                rec["leaf_end_mm"] = _round_pt(le)
                rec["leaf_end_px"] = _round_pt(transform.to_px(float(le[0]), float(le[1])))
        else:  # Window
            rec["swing_direction"] = None
            if comp.p1 is not None and comp.p2 is not None:
                rec["p1_mm"] = _round_pt(comp.p1)
                rec["p2_mm"] = _round_pt(comp.p2)
                rec["p1_px"] = _round_pt(transform.to_px(float(comp.p1[0]), float(comp.p1[1])))
                rec["p2_px"] = _round_pt(transform.to_px(float(comp.p2[0]), float(comp.p2[1])))
                rec["angle_deg"] = round(g.angle_deg(comp.direction), 2)

        if w < YOLO_MIN_DIM or h < YOLO_MIN_DIM:
            warnings.append(
                f"{plan.plan_id}: {kind} {comp.id} bbox {w:.3f}x{h:.3f} < {YOLO_MIN_DIM} "
                f"(small for YOLO)")
        openings.append(rec)

    return openings, warnings


def export_rich_json(plan, transform, image_w: int, image_h: int, path: str,
                     openings: List[Dict] = None) -> Dict:
    if openings is None:
        openings, _ = build_openings(plan, transform, image_w, image_h)
    data = {
        "plan_id": plan.plan_id,
        "scenario": plan.scenario,
        "scale": plan.scale,
        "rotation_deg": plan.rotation_deg,
        "image_w": image_w,
        "image_h": image_h,
        "openings": openings,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return data


def export_yolo(plan, transform, image_w: int, image_h: int, path: str,
                openings: List[Dict] = None) -> List[str]:
    if openings is None:
        openings, _ = build_openings(plan, transform, image_w, image_h)
    lines = []
    for op in openings:
        cls = CLASS_ID[op["type"]]
        cx, cy, w, h = op["bbox_normalized"]
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
    return lines


def write_data_yaml(dataset_root: str, path: str = None) -> str:
    dataset_root = os.path.abspath(dataset_root)
    path = path or os.path.join(dataset_root, "data.yaml")
    content = (
        f"# Ultralytics YOLO dataset descriptor (auto-generated)\n"
        f"path: {dataset_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 2\n"
        f"names: [door, window]\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
