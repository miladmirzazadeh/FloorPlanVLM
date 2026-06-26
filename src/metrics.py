"""Vector floor-plan metrics (the paper's FPBench measures), pure functions.

  validity    : valid JSON + at least one wall
  ext_iou     : IoU of the external wall footprint (paper's headline IoU_ext)
  room_iou    : mean IoU of geometrically matched rooms (>= 0.5)
  room_f1     : F1 where a room is a true positive only if IoU>0.5 AND label matches
  opening_f1  : F1 of doors/windows matched by type + location along the wall
  wall_mae    : |#pred_walls - #gt_walls|

Rooms in our schema are wall-id references, so room polygons are reconstructed by
polygonizing the referenced (arc-aware) wall centerlines. All geometry is in the
1024-normalized space. No model/torch here, so this is unit-tested directly.
"""
import json

import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

from .geometry import wall_polyline
from .rewards import extract_json, walls_to_polygon, poly_iou

OPENING_TOL = 30.0   # px (in 1024 space) for matching opening locations
ROOM_IOU_THR = 0.5


def _faces(plan):
    """Room faces = interior polygons formed by polygonizing all wall centerlines.
    Unambiguous (no reliance on which walls a room lists), so it works even when
    walls span multiple rooms."""
    lines = []
    for w in plan.get("walls", []):
        try:
            lines.append(LineString(wall_polyline(w)))
        except Exception:
            pass
    if len(lines) < 3:
        return []
    try:
        return [p for p in polygonize(unary_union(lines)) if p.area > 1.0]
    except Exception:
        return []


def _room_geom_scores(pred, gt):
    """Geometry-only: match room faces by IoU>=0.5 -> (mean matched IoU, face F1)."""
    P, G = _faces(pred), _faces(gt)
    if not G:
        return 0.0, 0.0
    used, ious, tp = set(), [], 0
    for gp in G:
        best, bj = -1.0, -1
        for j, pp in enumerate(P):
            if j in used:
                continue
            iou = poly_iou(gp, pp)
            if iou > best:
                best, bj = iou, j
        if bj >= 0 and best >= ROOM_IOU_THR:
            used.add(bj)
            ious.append(best)
            tp += 1
    prec = tp / len(P) if P else 0.0
    rec = tp / len(G) if G else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return (float(np.mean(ious)) if ious else 0.0), f1


def _room_label_f1(pred, gt):
    """Semantic-only: multiset F1 of room labels."""
    from collections import Counter
    pc = Counter(r.get("label") for r in pred.get("rooms", []))
    gc = Counter(r.get("label") for r in gt.get("rooms", []))
    if not pc and not gc:
        return None
    tp = sum((pc & gc).values())
    prec = tp / sum(pc.values()) if sum(pc.values()) else 0.0
    rec = tp / sum(gc.values()) if sum(gc.values()) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _opening_points(plan):
    pts = []
    for w in plan.get("walls", []):
        try:
            line = LineString(wall_polyline(w))
        except Exception:
            continue
        for op in w.get("openings", []):
            try:
                pts.append((op.get("type"), line.interpolate(float(op.get("center", 0)))))
            except Exception:
                pass
    return pts


def _opening_f1(pred, gt):
    P, G = _opening_points(pred), _opening_points(gt)
    if not P and not G:
        return None                      # nothing to score
    used, tp = set(), 0
    for gtype, gp in G:
        best, bj = OPENING_TOL + 1, -1
        for j, (ptype, pp) in enumerate(P):
            if j in used or ptype != gtype:
                continue
            d = gp.distance(pp)
            if d < best:
                best, bj = d, j
        if bj >= 0 and best <= OPENING_TOL:
            used.add(bj)
            tp += 1
    prec = tp / len(P) if P else 0.0
    rec = tp / len(G) if G else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def evaluate_pair(pred_text, gt_str):
    """Return a per-sample metric dict. pred_text is raw model output."""
    try:
        gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
    except Exception:
        gt = {"walls": [], "rooms": []}
    pred = extract_json(pred_text)
    valid = bool(pred and isinstance(pred.get("walls"), list) and pred["walls"])
    if not valid:
        return {"valid": 0.0, "ext_iou": 0.0, "room_iou": 0.0, "room_f1": 0.0,
                "room_label_f1": 0.0, "opening_f1": float("nan"),
                "wall_mae": float(len(gt.get("walls", [])))}
    ext = poly_iou(walls_to_polygon(pred.get("walls", [])),
                   walls_to_polygon(gt.get("walls", [])))
    room_iou, room_f1 = _room_geom_scores(pred, gt)
    lf1 = _room_label_f1(pred, gt)
    of = _opening_f1(pred, gt)
    return {"valid": 1.0, "ext_iou": float(ext), "room_iou": room_iou, "room_f1": room_f1,
            "room_label_f1": float("nan") if lf1 is None else float(lf1),
            "opening_f1": float("nan") if of is None else float(of),
            "wall_mae": float(abs(len(pred.get("walls", [])) - len(gt.get("walls", []))))}


def aggregate(rows):
    """Mean each metric across per-sample dicts (nan-safe)."""
    keys = ["valid", "ext_iou", "room_iou", "room_f1", "room_label_f1", "opening_f1", "wall_mae"]
    out = {"n": len(rows)}
    for k in keys:
        vals = np.array([r[k] for r in rows], float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(vals.mean()) if len(vals) else float("nan")
    return out
