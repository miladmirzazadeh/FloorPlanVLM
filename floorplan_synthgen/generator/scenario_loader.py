"""Bridge: load the external absolute-geometry scenario files (``scenarios/*.json``,
schema in ``scenarios/SCHEMA.md``) and convert them to the engine's config-dict
format so they render through the refined component library.

The external format places each opening by absolute jamb points
(``p1``/``p2``/``center``) on absolute wall polygons / arcs; the engine places
openings by ``(wall_id, position_along_wall, clear_opening_mm)``. This module:

* maps wall ``centerline`` -> ``x1,y1,x2,y2`` (+ tessellates ``arc`` walls into
  short straight segments),
* matches each opening's ``center`` to its host wall and projects it to a 0..1
  parameter,
* maps the subtype / hinge / swing enums to the engine's,
* honours the authored jamb gap: the engine cuts ``clear + 2*FRAME_W``, so it
  sets ``clear = width - 2*FRAME_W`` and the cut wall gap equals ``|p1-p2|``,
* clamps-and-warns rather than failing on imperfect inputs (off-wall openings,
  positions out of range), so a large authored batch never halts mid-run.

Anything the engine can't place is collected in the returned ``warnings`` list.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

from .components import FRAME_W

# external subtype enum -> engine subtype enum
_DOOR_SUBTYPE = {
    "SINGLE_HINGED": "SINGLE", "DOUBLE_HINGED": "DOUBLE", "SLIDING": "SLIDING",
    "POCKET": "POCKET", "BIFOLD": "BIFOLD", "GARAGE": "GARAGE", "FRENCH": "FRENCH",
}
_WIN_SUBTYPE = {s: s for s in ("CASEMENT", "SLIDING", "FIXED", "BAY", "AWNING",
                               "LOUVRE", "CORNER", "CLERESTORY")}
# external wall type -> engine wall_type (drives full/continuous vs interior)
_WALL_TYPE = {"exterior": "EXTERIOR", "party": "FIRE", "core": "EXTERIOR",
              "structural": "EXTERIOR", "interior": "INTERIOR",
              "partition": "INTERIOR"}

_GROUP_FILES = ("A.json", "B.json", "C.json", "D.json", "E.json")


# --------------------------------------------------------------------------
def _seg_project(p, a, b) -> Tuple[float, float]:
    """Return (perpendicular distance from p to segment a-b, clamped t in 0..1)."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    tc = max(0.0, min(1.0, t))
    qx, qy = ax + dx * tc, ay + dy * tc
    return math.hypot(px - qx, py - qy), tc


def _straight_wall(w: Dict) -> Dict:
    (x1, y1), (x2, y2) = w["centerline"]
    return {"id": w["id"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "thickness": float(w.get("thickness_mm", 150.0)),
            "wall_type": _WALL_TYPE.get(str(w.get("type", "interior")).lower(),
                                        "INTERIOR")}


def _on_arc(center, arc: Dict, thickness: float) -> bool:
    """True if ``center`` lies on the arc band (near the radius and within the
    swept angle). A tight radius tolerance disambiguates multiple arcs."""
    if center is None:
        return False
    cx, cy = arc["center"]
    r = float(arc["radius"])
    if abs(math.hypot(center[0] - cx, center[1] - cy) - r) > max(thickness, 150.0):
        return False
    th = math.atan2(center[1] - cy, center[0] - cx)
    a0 = math.radians(float(arc["a0"]))
    a1 = math.radians(float(arc["a1"]))
    lo, hi = min(a0, a1), max(a0, a1)
    while th < lo - 1e-9:
        th += 2 * math.pi
    while th > hi + 1e-9:
        th -= 2 * math.pi
    return lo - 1e-6 <= th <= hi + 1e-6


def _arc_segments_with_openings(wall: Dict, ops: List[Dict],
                                warnings: List[str], plan_id: str):
    """Tessellate an arc wall into straight segments, but carve it so each
    opening gets exactly ONE host segment spanning its chord (between its
    jambs, oriented along the chord, with endpoints on the arc so neighbours
    connect cleanly). Returns (segment dicts, {op_id: (seg_id, pos, clear)})."""
    arc = wall["arc"]
    cx, cy = arc["center"]
    r = float(arc["radius"])
    a0 = math.radians(float(arc["a0"]))
    a1 = math.radians(float(arc["a1"]))
    lo, hi = min(a0, a1), max(a0, a1)
    thick = float(wall.get("thickness_mm", 150.0))
    wtype = _WALL_TYPE.get(str(wall.get("type", "interior")).lower(), "INTERIOR")

    def pt(theta):
        return (cx + r * math.cos(theta), cy + r * math.sin(theta))

    spans = []                                  # (theta_lo, theta_hi, opening)
    for op in ops:
        c = op.get("center")
        th = math.atan2(c[1] - cy, c[0] - cx)
        while th < lo - 1e-9:
            th += 2 * math.pi
        while th > hi + 1e-9:
            th -= 2 * math.pi
        width = float(op.get("width_mm", 800.0))
        half = math.asin(min(0.999, (width / 2.0) / r))
        spans.append([max(lo, th - half), min(hi, th + half), op])
    spans.sort(key=lambda s: s[0])

    segs = []
    assigns = {}
    idx = [0]

    def tess(t_start, t_end):                   # curvature segments over a gap
        if t_end - t_start < 1e-6:
            return
        n = max(1, int(abs(math.degrees(t_end - t_start)) / 12.0))
        for k in range(n):
            ta = t_start + (t_end - t_start) * (k / n)
            tb = t_start + (t_end - t_start) * ((k + 1) / n)
            (x1, y1), (x2, y2) = pt(ta), pt(tb)
            segs.append({"id": f"{wall['id']}_s{idx[0]}", "x1": x1, "y1": y1,
                         "x2": x2, "y2": y2, "thickness": thick, "wall_type": wtype})
            idx[0] += 1

    cur = lo
    for t_lo, t_hi, op in spans:
        tess(cur, t_lo)                         # arc up to this opening
        (x1, y1), (x2, y2) = pt(t_lo), pt(t_hi)
        sid = f"{wall['id']}_o{idx[0]}"
        idx[0] += 1
        segs.append({"id": sid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "thickness": thick, "wall_type": wtype})
        chord = math.hypot(x2 - x1, y2 - y1)
        assigns[op["id"]] = (sid, 0.5, max(chord - 2 * FRAME_W, 200.0))
        cur = t_hi
    tess(cur, hi)                               # remaining arc
    if not segs:                                # degenerate arc -> single chord
        (x1, y1), (x2, y2) = pt(lo), pt(hi)
        segs.append({"id": f"{wall['id']}_s0", "x1": x1, "y1": y1, "x2": x2,
                     "y2": y2, "thickness": thick, "wall_type": wtype})
    return segs, assigns


def _swing_string(op: Dict) -> str:
    sw = str(op.get("swing") or "none").lower()
    hg = str(op.get("hinge") or "left").lower()
    if sw == "none":
        return None
    inout = "INWARD" if sw in ("in", "both") else "OUTWARD"
    side = "RIGHT" if hg == "right" else "LEFT"
    return f"{inout}_{side}"


def _opening_to_config(op: Dict, walls_cfg: List[Dict], arc_assigns: Dict,
                       warnings: List[str], plan_id: str) -> Dict:
    cat = str(op.get("category", "door")).lower()
    center = op.get("center")
    if center is None and op.get("p1") and op.get("p2"):
        center = [(op["p1"][0] + op["p2"][0]) / 2.0,
                  (op["p1"][1] + op["p2"][1]) / 2.0]

    # roof-plane opening (skylight): no host wall
    if str(op.get("plane", "wall")).lower() == "roof":
        return {"id": op["id"], "rooflight": True, "x": center[0], "y": center[1],
                "size_mm": float(op.get("width_mm", 1000.0)), "type": "window",
                "subtype": "FIXED"}

    width = float(op.get("width_mm", 800.0))
    if op["id"] in arc_assigns:
        # pre-assigned to a carved arc chord segment
        wall_id, t, clear = arc_assigns[op["id"]]
    else:
        # match to the nearest STRAIGHT wall centreline, project centre to 0..1
        best = None
        for w in walls_cfg:
            d, tt = _seg_project(center, (w["x1"], w["y1"]), (w["x2"], w["y2"]))
            if best is None or d < best[0]:
                best = (d, tt, w)
        dist, t, wall = best
        wall_id = wall["id"]
        tol = wall["thickness"] * 0.5 + 50.0
        if dist > tol:
            warnings.append(f"[{plan_id}] opening {op['id']} is {dist:.0f}mm off "
                            f"the nearest wall {wall_id} (placed anyway at t={t:.2f})")
        # engine cuts clear + 2*FRAME_W; keep the cut gap == authored jamb gap
        clear = max(width - 2 * FRAME_W, 200.0)

    if cat == "window":
        subtype = _WIN_SUBTYPE.get(str(op.get("subtype", "FIXED")).upper(), "FIXED")
        return {"id": op["id"], "type": "window", "wall_id": wall_id,
                "position_along_wall": t, "subtype": subtype,
                "clear_opening_mm": clear}

    # doors and leafless openings (CASED / GAP) both render as doors
    subtype = _DOOR_SUBTYPE.get(str(op.get("subtype", "SINGLE_HINGED")).upper(),
                                "SINGLE")
    return {"id": op["id"], "type": "door", "wall_id": wall_id,
            "position_along_wall": t, "subtype": subtype,
            "clear_opening_mm": clear, "swing": _swing_string(op),
            "max_swing_angle_deg": float(op.get("max_swing_angle_deg", 90.0))}


def _clutter_for(scenario: Dict) -> Dict:
    """A moderate, realistic clutter set; heavier for the group-E stress plans.
    NOTE: ``hatch_walls`` is intentionally left unset so the engine chooses a
    per-plan, scale-appropriate material fill (varied across plans) rather than
    one uniform hatch everywhere."""
    grp = str(scenario.get("group", "")).upper()
    base = {"room_labels": True, "dimensions": True, "title_block": True,
            "furniture": False, "grid": False, "noise_lines": 0}
    if grp == "E":
        base.update({"furniture": True, "grid": True, "noise_lines": 4})
    return base


def _render_for(scenario: Dict) -> Dict:
    r = dict(scenario.get("render", {}) or {})
    lw = str(r.get("line_weight", "")).lower()
    style = "heavy" if "heavy" in lw or "bold" in lw else "standard"
    degr = str(r.get("degradation", "none")).lower()
    noise = 0.0 if degr in ("none", "") else 0.015
    return {"dpi": 150, "line_weight_style": style, "monochrome": True,
            "noise_std": noise}


def scenario_to_config(scenario: Dict, warnings: List[str]) -> Dict:
    plan_id = scenario["id"]
    raw_walls = scenario.get("walls", [])
    raw_ops = scenario.get("openings", [])

    # arc walls are carved around their own openings (each gets a chord segment);
    # straight walls pass through directly.
    walls_cfg = []
    arc_assigns = {}
    used = set()
    for w in raw_walls:
        if w.get("arc"):
            thick = float(w.get("thickness_mm", 150.0))
            ops_on = [o for o in raw_ops if o.get("id") not in used
                      and str(o.get("plane", "wall")).lower() != "roof"
                      and _on_arc(o.get("center"), w["arc"], thick)]
            segs, assigns = _arc_segments_with_openings(w, ops_on, warnings, plan_id)
            walls_cfg.extend(segs)
            arc_assigns.update(assigns)
            used.update(assigns.keys())
        else:
            walls_cfg.append(_straight_wall(w))

    openings_cfg = []
    for op in raw_ops:
        try:
            openings_cfg.append(_opening_to_config(op, walls_cfg, arc_assigns,
                                                    warnings, plan_id))
        except Exception as e:                       # never let one opening halt
            warnings.append(f"[{plan_id}] skipped opening {op.get('id')}: {e}")
    rooms = [{"id": r.get("id", f"r{i}"), "name": r.get("name", "ROOM"),
              "polygon": r.get("polygon")} for i, r in enumerate(scenario.get("rooms", []))
             if r.get("polygon")]
    render = scenario.get("render", {}) or {}
    return {
        "plan_id": plan_id,
        "scenario": scenario.get("name", plan_id),
        "scale": render.get("scale", "1:100"),
        "rotation_deg": float(render.get("rotation_deg", 0.0)),
        "walls": walls_cfg,
        "openings": openings_cfg,
        "rooms": rooms,
        "columns": [],
        "clutter": _clutter_for(scenario),
        "render": _render_for(scenario),
    }


def load_scenarios(directory: str) -> Tuple[List[Dict], List[str]]:
    """Load every scenario in ``directory`` (the group files A..E.json) and return
    (list of engine configs, list of conversion warnings)."""
    configs, warnings = [], []
    files = [os.path.join(directory, f) for f in _GROUP_FILES
             if os.path.exists(os.path.join(directory, f))]
    if not files:  # fall back to any *.json that is a list of scenarios
        for f in sorted(os.listdir(directory)):
            if f.endswith(".json") and f not in ("index.json",):
                files.append(os.path.join(directory, f))
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            continue
        for scenario in data:
            if isinstance(scenario, dict) and scenario.get("walls"):
                configs.append(scenario_to_config(scenario, warnings))
    return configs, warnings
