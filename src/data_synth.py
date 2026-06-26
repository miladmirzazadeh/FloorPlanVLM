"""synth-floorseg (the user's Vitruev synthetic generator) -> FloorPlanVLM JSON.

This is the richest source: SYNTHETIC (pixel-perfect) with EXPLICIT vector geometry —
walls carry a centerline + thickness_mm + an optional `arc` (real curved-wall labels),
rooms carry a polygon + room_type, openings carry category + width + p1/p2. Inputs are
the generator's own rendered PNGs.

Layout (matches the Kaggle `synth-floorseg` zip / dataset_10k):
  configs/plan_XXXXX.json          walls/rooms/openings in millimetres
  rich_json/plan_XXXXX_rich.json   per-opening mm<->px pairs (to solve the transform)
  images/**/plan_XXXXX.png         rendered floor plan

The mm->px affine is solved per plan from rich_json's same-point (mm,px) pairs (the
generator's exact ModelTransform), then we scale to longest-edge 1024. Adapted from the
generator's own floorseg/synth_to_masks.py.
"""
import os
import re
import glob
import json

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon

from . import config
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .taxonomy import SYNTH_ROOM_MAP
from .geometry import fit_curvature, wall_polyline

TARGET = 1024
PLAN_RE = re.compile(r"(plan_\d+)")


def _pid(path):
    m = PLAN_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def _collect_pairs(rich):
    mm, px = [], []
    for o in rich.get("openings", []):
        for a, b in (("hinge_point_mm", "hinge_point_px"),
                     ("leaf_end_mm", "leaf_end_px"),
                     ("p1_mm", "p1_px"), ("p2_mm", "p2_px")):
            if o.get(a) and o.get(b):
                mm.append(o[a])
                px.append(o[b])
    return np.array(mm, float), np.array(px, float)


def _solve_affine(mm, px):
    if len(mm) < 3:
        return None
    M = np.hstack([mm, np.ones((len(mm), 1))])
    sol, *_ = np.linalg.lstsq(M, px, rcond=None)   # [3,2]
    return sol.T                                    # [2,3]  px ~= A @ [x,y,1]


def _mk_tx(A, s):
    def tx(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        h = np.hstack([pts, np.ones((len(pts), 1))])
        return (h @ A.T) * s
    return tx


def _arc_endpoints(arc):
    c = np.array(arc["center"], float)
    R = float(arc["radius"])
    a0, a1 = np.radians(arc["a0"]), np.radians(arc["a1"])
    am = (a0 + a1) / 2.0
    P = lambda a: c + R * np.array([np.cos(a), np.sin(a)])
    return P(a0), P(am), P(a1)


def convert(cfg, rich, img_path):
    A = _solve_affine(*_collect_pairs(rich))
    if A is None:
        return None
    with Image.open(img_path) as im:
        W, H = im.size
    s = TARGET / max(W, H)
    tx = _mk_tx(A, s)
    ppm = float(np.hypot(A[0, 0], A[1, 0])) * s     # mm -> 1024 px scale

    walls = []
    for i, w in enumerate(cfg.get("walls", [])):
        thickness = max(round(float(w.get("thickness_mm", 100)) * ppm), 1)
        if w.get("arc"):
            p0, pm, p1 = _arc_endpoints(w["arc"])
            q0, qm, q1 = tx([p0])[0], tx([pm])[0], tx([p1])[0]
            curv = round(float(fit_curvature([q0, qm, q1])), 3)
            start, end = q0, q1
        else:
            cl = w.get("centerline")
            if not cl or len(cl) < 2:
                continue
            pts = tx([cl[0], cl[-1]])
            start, end, curv = pts[0], pts[1], 0
        walls.append({
            "id": f"wall_{i + 1}",
            "start": [round(float(start[0])), round(float(start[1]))],
            "end": [round(float(end[0])), round(float(end[1]))],
            "thickness": thickness,
            "curvature": curv,
            "openings": [],
        })
    if not walls:
        return None

    lines = [LineString(wall_polyline(w)) for w in walls]

    for o in cfg.get("openings", []):
        p1, p2, ctr = o.get("p1"), o.get("p2"), o.get("center")
        if not (p1 and p2):
            continue
        cat = (o.get("category") or o.get("type") or "").lower()
        typ = "window" if "window" in cat else "door"
        c = tx([ctr if ctr else p1])[0]
        cp = Point(c)
        if o.get("width_mm"):
            width = max(round(float(o["width_mm"]) * ppm), 1)
        else:
            a, b = tx([p1, p2])
            width = max(round(float(np.hypot(b[0] - a[0], b[1] - a[1]))), 1)
        bi = min(range(len(walls)), key=lambda i: lines[i].distance(cp))
        if lines[bi].distance(cp) < walls[bi]["thickness"] * 4 + width:
            walls[bi]["openings"].append({"type": typ,
                                          "center": round(lines[bi].project(cp)),
                                          "width": width})

    rooms = []
    for r in cfg.get("rooms", []):
        poly = r.get("polygon") or r.get("points")
        if not poly or len(poly) < 3:
            continue
        label = SYNTH_ROOM_MAP.get(r.get("room_type") or r.get("name", ""), "room")
        try:
            rp = Polygon([tuple(p) for p in tx(poly)])
            if not rp.is_valid:
                rp = rp.buffer(0)
        except Exception:
            continue
        ids = []
        for w, ln in zip(walls, lines):
            try:
                if rp.boundary.distance(ln) < w["thickness"] * 2 + 4:
                    ids.append(w["id"])
            except Exception:
                pass
        if ids:
            rooms.append({"label": label, "walls": ids})

    result = {"walls": walls, "rooms": rooms}
    img = Image.open(img_path).convert("RGB").resize(
        (max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
    return img, result


def _record(img, js):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": js}]},
        ],
        "images": [img],
    }


def build_synth_records(synth_dir, max_samples=None, want_records=True):
    cfgs = {_pid(p): p for p in glob.glob(os.path.join(synth_dir, "**", "configs", "plan_*.json"), recursive=True)}
    if not cfgs:
        cfgs = {_pid(p): p for p in glob.glob(os.path.join(synth_dir, "**", "plan_*.json"), recursive=True)
                if "rich" not in p}
    riches = {_pid(p): p for p in glob.glob(os.path.join(synth_dir, "**", "*_rich.json"), recursive=True)}
    imgs = {_pid(p): p for p in glob.glob(os.path.join(synth_dir, "**", "*.png"), recursive=True)
            if _pid(p)}
    ids = sorted(i for i in (set(cfgs) & set(riches) & set(imgs)) if i)
    print(f"[synth] {len(ids)} plans with config+rich+image under {synth_dir}")
    if max_samples:
        ids = ids[:max_samples]

    render_dir = config.SYNTH_RENDER_DIR
    os.makedirs(render_dir, exist_ok=True)

    records, annotations, errors = [], [], 0
    for k, pid in enumerate(ids):
        if k % 200 == 0:
            print(f"[synth]   {k}/{len(ids)} ({len(annotations)} ok, {errors} err)")
        try:
            cfg = json.loads(open(cfgs[pid]).read())
            rich = json.loads(open(riches[pid]).read())
            res = convert(cfg, rich, imgs[pid])
            if res is None:
                errors += 1
                continue
            img, jd = res
            js = json.dumps(jd, separators=(",", ":"))
            if len(js) > config.MAX_JSON_CHARS:
                continue
            img_path = os.path.abspath(os.path.join(render_dir, f"{pid}.png"))
            img.save(img_path)
            annotations.append({"image_path": img_path, "json_annotation": js})
            if want_records:
                records.append(_record(img, js))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"[synth]   err {pid}: {e}")
    print(f"[synth] built {len(annotations)} samples ({errors} errors)")
    return records, annotations


def _debug(path):
    cfgs = sorted(glob.glob(os.path.join(path, "**", "configs", "plan_*.json"), recursive=True)) \
        or sorted(p for p in glob.glob(os.path.join(path, "**", "plan_*.json"), recursive=True) if "rich" not in p)
    if not cfgs:
        print("no configs under", path)
        return
    from PIL import ImageDraw
    cfgp = cfgs[0]
    pid = _pid(cfgp)
    rich = next(p for p in glob.glob(os.path.join(path, "**", f"{pid}_rich.json"), recursive=True))
    img = next(p for p in glob.glob(os.path.join(path, "**", f"{pid}.png"), recursive=True))
    res = convert(json.loads(open(cfgp).read()), json.loads(open(rich).read()), img)
    if res is None:
        print("conversion returned None")
        return
    im, jd = res
    nc = sum(1 for w in jd["walls"] if abs(w["curvature"]) > 0.08)
    print(f"{pid}: walls={len(jd['walls'])} ({nc} curved) rooms={len(jd['rooms'])} "
          f"openings={sum(len(w['openings']) for w in jd['walls'])}")
    print("room labels:", sorted({r['label'] for r in jd['rooms']}))
    ov = im.copy()
    d = ImageDraw.Draw(ov)
    for w in jd["walls"]:
        d.line(wall_polyline(w), fill=(255, 0, 0), width=2)
    out = f"{pid}_debug.png"
    ov.save(out)
    print(f"saved overlay -> {out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.data_synth <dataset_dir>")
        raise SystemExit(1)
    _debug(sys.argv[1])
