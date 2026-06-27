"""Geometric reward for GRPO — FloorPlanVLM Eq. 9.

    R = 0.1 * R_val  +  0.5 * R_ext  +  alpha * 0.4 * R_int

  R_val : JSON validity + structural well-formedness (walls have required keys,
          rooms reference existing wall ids).
  R_ext : IoU of the external wall footprint vs. ground truth.
  R_int : internal-structure agreement (room-label set overlap).
  alpha : gate on R_ext (Eq. 8) — internal detail only rewarded once the global
          boundary is roughly right, so the model fixes the outline first.

Adapted from the community reference; kept deterministic and exception-safe so a
single malformed completion can never crash a training step.
"""
import json
import re

from . import config
from .geometry import walls_union, region_iou


def extract_json(text):
    if isinstance(text, list):
        text = text[0].get("content", "") if text else ""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return _salvage_walls(text)


def _salvage_walls(text):
    """Recover walls from truncated/over-long output by brace-matching each complete
    wall object (handles the case where generation hit the token cap mid-JSON)."""
    i = text.find('"walls"')
    if i < 0:
        return None
    i = text.find("[", i)
    if i < 0:
        return None
    walls, j, n = [], i + 1, len(text)
    while j < n:
        while j < n and text[j] not in "{]":
            j += 1
        if j >= n or text[j] == "]":
            break
        depth, k = 0, j
        while k < n:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if k >= n:  # object truncated -> stop
            break
        try:
            walls.append(json.loads(text[j:k + 1]))
        except Exception:
            pass
        j = k + 1
    return {"walls": walls} if walls else None


def walls_to_polygon(walls):
    """External footprint of the wall set (arc-aware via geometry.walls_union)."""
    if not walls or len(walls) < 3:
        return None
    try:
        combined = walls_union(walls)
        return combined.convex_hull if combined is not None else None
    except Exception:
        return None


def poly_iou(p1, p2):
    if p1 is None or p2 is None:
        return 0.0
    try:
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def floorplan_reward(completions, **kwargs):
    """TRL reward fn. `json_gt` is forwarded automatically from the dataset column."""
    gt_jsons = kwargs.get("json_gt", [])
    rewards = []
    for c, gt_str in zip(completions, gt_jsons):
        text = c[0]["content"] if isinstance(c, list) else c
        pred = extract_json(text)
        if pred is None:
            rewards.append(0.0)
            continue
        try:
            gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
        except Exception:
            rewards.append(0.0)
            continue

        # R_val
        r_val = 0.0
        pw = pred.get("walls")
        if isinstance(pw, list) and pw:
            valid = sum(
                1 for w in pw
                if all(k in w for k in ("id", "start", "end", "thickness"))
                and isinstance(w.get("start"), list) and len(w.get("start", [])) == 2
            )
            r_val = 0.3 + 0.5 * (valid / max(len(pw), 1))
            pr = pred.get("rooms")
            if isinstance(pr, list):
                wids = {w.get("id") for w in pw}
                vr = sum(1 for r in pr if "label" in r and "walls" in r
                         and all(wid in wids for wid in r.get("walls", [])))
                r_val += 0.2 * (vr / max(len(pr), 1))

        # R_ext
        r_ext = poly_iou(walls_to_polygon(pred.get("walls", [])),
                         walls_to_polygon(gt.get("walls", [])))

        if config.WALLS_ONLY:
            # walls-only: reward correct outer boundary + correct space partition
            # (region topology = closure + separation), no room labels needed.
            r_region = region_iou(pred.get("walls", []), gt.get("walls", []))
            rewards.append(float(0.1 * min(r_val, 1.0) + 0.45 * r_ext + 0.45 * r_region))
            continue

        # alpha gate (Eq. 8)
        if r_ext < 0.3:
            alpha = 0.1
        elif r_ext < 0.7:
            alpha = 0.1 + 0.9 * (r_ext - 0.3) / 0.4
        else:
            alpha = 1.0

        # R_int (room-label set overlap)
        r_int = 0.0
        pr, gr = pred.get("rooms", []), gt.get("rooms", [])
        if pr and gr:
            pl = {r.get("label", "") for r in pr}
            gl = {r.get("label", "") for r in gr}
            tot = len(pl | gl)
            r_int = len(pl & gl) / tot if tot > 0 else 0.0

        rewards.append(float(0.1 * min(r_val, 1.0) + 0.5 * r_ext + alpha * 0.4 * r_int))
    return rewards
