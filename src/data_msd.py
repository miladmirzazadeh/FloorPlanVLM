"""MSD (Modified Swiss Dwellings, ECCV 2024) -> FloorPlanVLM JSON.

MSD is a floor-plan *generation* benchmark. Its graph representation OMITS walls,
but `full_out/*.npy` is a complete integer segmentation mask (rooms + Structure +
openings). We convert from that single mask so everything lives in one coordinate
space and the (rendered image, JSON) pair is pixel-aligned by construction:

  rooms     : per-class connected-component contours -> simplified polygons
  walls     : decompose room polygons into edges, dedup shared edges
              (wall-centric schema reconstructed from room boundaries; thickness
               estimated from the Structure mask)
  openings  : Door/Window/Entrance components -> center + width -> nearest wall
  input img : neutral line drawing (black walls, gray rooms) — no per-type colour,
              so room *type* is NOT leaked into the input.

This module reads ONLY full_out/*.npy (no torch/networkx/pickle needed).

NOTE: not validated on real MSD arrays in this environment. Before a full run,
eyeball one conversion:  python -m src.data_msd /path/to/some_full_out.npy
"""
import os
import glob
import json

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point

from . import config
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .taxonomy import (
    MSD_ROOM_INDICES, MSD_STRUCTURE_INDEX, MSD_DOOR_INDICES, MSD_WINDOW_INDEX,
)
from .geometry import rooms_to_walls

try:
    import cv2
except Exception:  # pragma: no cover - surfaced clearly at call time
    cv2 = None

TARGET = 1024


def _need_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for MSD parsing "
                           "(pip install opencv-python-headless)")


def _squeeze_mask(mask):
    """Coerce whatever np.load returns into a 2D integer class map."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:          # (1,H,W)
            mask = mask[0]
        elif mask.shape[-1] == 1:       # (H,W,1)
            mask = mask[..., 0]
        else:                            # (H,W,K) one-hot / probs
            mask = mask.argmax(-1)
    return np.rint(mask).astype(np.int32)


def _contours(mask, value, min_area):
    binm = (mask == value).astype(np.uint8)
    if binm.sum() == 0:
        return []
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        if len(approx) >= 3:
            polys.append(approx)
    return polys


def _components(mask, value, min_pix):
    binm = (mask == value).astype(np.uint8)
    if binm.sum() == 0:
        return []
    n, lab = cv2.connectedComponents(binm)
    out = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(xs) < min_pix:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        width = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
        out.append((cx, cy, width))
    return out


def _estimate_thickness(mask):
    binm = (mask == MSD_STRUCTURE_INDEX).astype(np.uint8)
    if binm.sum() == 0:
        return 4.0
    dt = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
    vals = dt[dt > 0]
    return float(np.median(vals) * 2.0) if len(vals) else 4.0


def _assign_openings(walls, openings):
    if not walls:
        return
    lines = [LineString([w["start"], w["end"]]) for w in walls]
    for op in openings:
        c = Point(op["center"])
        best = min(range(len(walls)), key=lambda i: lines[i].distance(c))
        if lines[best].distance(c) < walls[best]["thickness"] * 4 + 6:
            walls[best]["openings"].append({
                "type": op["type"],
                "center": round(lines[best].project(c)),
                "width": op["width"],
            })


def _render(mask):
    """Neutral floor-plan drawing: black walls, light-gray rooms, no type colour."""
    h, w = mask.shape
    img = np.full((h, w, 3), 255, np.uint8)
    room_vals = list(MSD_ROOM_INDICES.keys())
    img[np.isin(mask, room_vals)] = (230, 230, 230)
    img[mask == MSD_WINDOW_INDEX] = (140, 150, 200)
    for d in MSD_DOOR_INDICES:
        img[mask == d] = (250, 250, 250)
    img[mask == MSD_STRUCTURE_INDEX] = (0, 0, 0)
    return Image.fromarray(img)


def mask_to_record(mask):
    """One full_out mask -> (PIL image @ longest-edge 1024, json dict) or None."""
    _need_cv2()
    mask = _squeeze_mask(mask)
    h, w = mask.shape
    scale = TARGET / max(h, w)
    min_area = max(20.0, 0.0002 * h * w)
    min_pix = max(6, int(0.00005 * h * w))

    rooms = []
    for idx, label in MSD_ROOM_INDICES.items():
        for poly in _contours(mask, idx, min_area):
            rooms.append((label, poly * scale))
    if not rooms:
        return None

    thickness = _estimate_thickness(mask) * scale
    walls, room_walls = rooms_to_walls(rooms, thickness, fit_curves=config.FIT_CURVES)
    if not walls:
        return None

    openings = []
    for idx, typ in ([(MSD_WINDOW_INDEX, "window")] +
                     [(d, "door") for d in MSD_DOOR_INDICES]):
        for cx, cy, width in _components(mask, idx, min_pix):
            openings.append({"type": typ,
                             "center": [round(cx * scale), round(cy * scale)],
                             "width": max(round(width * scale), 1)})
    _assign_openings(walls, openings)

    result = {"walls": walls,
              "rooms": [{"label": lab, "walls": ids} for lab, ids in room_walls if ids]}

    img = _render(mask).resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.NEAREST).convert("RGB")
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


def build_msd_records(msd_dir, max_samples=None, want_records=True):
    """Walk full_out/*.npy, convert each, render+save input PNGs to disk.

    Returns (records, annotations). annotations always carry a real image_path
    (the rendered PNG) so the GRPO stage can train on MSD too.
    """
    files = sorted(glob.glob(os.path.join(msd_dir, "**", "full_out", "*.npy"), recursive=True))
    if not files:  # tolerate a flat directory of arrays
        files = sorted(glob.glob(os.path.join(msd_dir, "**", "*.npy"), recursive=True))
    print(f"[msd] found {len(files)} full_out arrays under {msd_dir}")
    if max_samples:
        files = files[:max_samples]

    render_dir = config.MSD_RENDER_DIR
    os.makedirs(render_dir, exist_ok=True)

    records, annotations, errors = [], [], 0
    for i, fp in enumerate(files):
        if i % 200 == 0:
            print(f"[msd]   {i}/{len(files)} ({len(annotations)} ok, {errors} err)")
        try:
            res = mask_to_record(np.load(fp))
            if res is None:
                errors += 1
                continue
            img, jd = res
            js = json.dumps(jd, separators=(",", ":"))
            if len(js) > config.MAX_JSON_CHARS:
                continue
            stem = os.path.splitext(os.path.basename(fp))[0]
            img_path = os.path.abspath(os.path.join(render_dir, f"msd_{stem}.png"))
            img.save(img_path)
            annotations.append({"image_path": img_path, "json_annotation": js})
            if want_records:
                records.append(_record(img, js))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"[msd]   err {fp}: {e}")
    print(f"[msd] built {len(annotations)} samples ({errors} errors)")
    return records, annotations


def _debug(path):
    """Convert one array (or the first in a dir) and save an overlay for eyeballing."""
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "**", "full_out", "*.npy"), recursive=True) \
            or glob.glob(os.path.join(path, "**", "*.npy"), recursive=True)
        if not cands:
            print("no .npy found under", path)
            return
        path = sorted(cands)[0]
    arr = _squeeze_mask(np.load(path))
    print(f"file={path}  shape={arr.shape}  unique values={sorted(set(arr.flatten().tolist()))[:20]}")
    res = mask_to_record(arr)
    if res is None:
        print("conversion returned None (no rooms/walls detected — check class indices in taxonomy.py)")
        return
    img, jd = res
    print(f"walls={len(jd['walls'])}  rooms={len(jd['rooms'])}  "
          f"openings={sum(len(w['openings']) for w in jd['walls'])}")
    print("room labels:", sorted({r['label'] for r in jd['rooms']}))
    overlay = img.copy()
    d = ImageDraw.Draw(overlay)
    for w in jd["walls"]:
        d.line([tuple(w["start"]), tuple(w["end"])], fill=(255, 0, 0), width=2)
        for op in w["openings"]:
            d.ellipse([w["start"][0] - 3, w["start"][1] - 3, w["start"][0] + 3, w["start"][1] + 3],
                      fill=(0, 0, 255))
    out = os.path.splitext(os.path.basename(path))[0] + "_debug.png"
    overlay.save(out)
    print(f"saved overlay -> {out}  (red=walls). Open it to verify the conversion.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.data_msd <full_out.npy | dir>")
        raise SystemExit(1)
    _debug(sys.argv[1])
