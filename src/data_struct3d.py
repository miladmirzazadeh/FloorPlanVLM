"""Structured3D (ECCV 2020) -> FloorPlanVLM JSON.

Structured3D is SYNTHETIC, so its geometry is exact -> rendering a 2D floor plan
from it gives PIXEL-PERFECT (image, JSON) pairs (the paper's HQ-synthetic trick).
We only need the small structure-annotation zip (~39 MB); the giant panorama /
perspective image zips are NOT used (they're 3D renders — we draw our own top-down
floor plan from the vector annotation).

Format of each scene's annotation_3d.json (validated on the real release):
  junctions          : [{"coordinate": [x, y, z]}, ...]   (3D, millimetres; floor z≈0)
  planes             : [{"type": "wall"|"floor"|"ceiling"}, ...]
  semantics          : [{"type": "<room>"|"door"|"window"|"outwall", "planeID": [...]}]
  planeLineMatrix    : [n_planes x n_lines] incidence (plane -> lines)
  lineJunctionMatrix : [n_lines x n_junctions] incidence (line -> its 2 junctions)

Conversion (top-down projection, x/y only):
  rooms     : each room semantic's floor plane -> ordered polygon (room type label)
  walls     : room polygon edges, deduped (shared edges = one wall); nominal thickness
  openings  : door/window semantic planes -> projected segment -> center+width -> wall

`convert_lines_to_vertices` is adapted from the official repo (visualize_floorplan.py).
"""
import os
import json
import glob
import zipfile
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point

from . import config
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .taxonomy import S3D_ROOM_MAP
from .geometry import arc_points, rooms_to_walls

TARGET = 1024


# ── annotation download (only the 39MB structure zip) ─────────────────────────

def download_annotations():
    """Ensure S3D annotation_3d.json files exist under config.S3D_DIR."""
    if glob.glob(os.path.join(config.S3D_DIR, "**", "annotation_3d.json"), recursive=True):
        return config.S3D_DIR
    os.makedirs(config.S3D_DIR, exist_ok=True)
    zip_path = os.path.join(config.S3D_DIR, "annotation_3d.zip")
    if not os.path.exists(zip_path):
        print(f"[s3d] downloading structure annotations (~39MB) from {config.S3D_ANNOT_URL}")
        urllib.request.urlretrieve(config.S3D_ANNOT_URL, zip_path)
    print("[s3d] extracting ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(config.S3D_DIR)
    return config.S3D_DIR


# ── geometry helpers ──────────────────────────────────────────────────────────

def convert_lines_to_vertices(lines):
    """Order (junction-pair) lines into polygon vertex loops. From the S3D repo."""
    polygons, lines = [], np.array(lines)
    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)
        lineID, juncID = np.where(lines == polygon[-1])
        if len(lineID) == 0:
            polygons.append(polygon)
            polygon = None
            continue
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)
        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)
    if polygon is not None:
        polygons.append(polygon)
    return polygons


def _plane_polygon(annos, plane_id):
    line_ids = np.where(np.array(annos["planeLineMatrix"][plane_id]))[0].tolist()
    pairs = []
    for lid in line_ids:
        js = np.where(np.array(annos["lineJunctionMatrix"][lid]))[0].tolist()
        if len(js) == 2:
            pairs.append(js)
    if not pairs:
        return None
    polys = convert_lines_to_vertices(pairs)
    polys = [p for p in polys if len(p) >= 3]
    return max(polys, key=len) if polys else None


def _opening_junctions(annos, plane_ids):
    ids = set()
    for pid in plane_ids:
        for lid in np.where(np.array(annos["planeLineMatrix"][pid]))[0]:
            ids.update(np.where(np.array(annos["lineJunctionMatrix"][lid]))[0].tolist())
    return list(ids)


def _assign_openings(walls, openings):
    if not walls:
        return
    lines = [LineString([w["start"], w["end"]]) for w in walls]
    for op in openings:
        c = Point(op["center"])
        best = min(range(len(walls)), key=lambda i: lines[i].distance(c))
        if lines[best].distance(c) < walls[best]["thickness"] * 4 + op["width"]:
            walls[best]["openings"].append({"type": op["type"],
                                            "center": round(lines[best].project(c)),
                                            "width": op["width"]})


def _render(rooms, walls, W, H):
    """Neutral floor-plan drawing: gray rooms, black walls, openings as wall gaps
    (doors = plain gap, windows = gap + thin blue pane). No room-type colour."""
    img = Image.new("RGB", (max(W, 1), max(H, 1)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for _, poly in rooms:
        if len(poly) >= 3:
            d.polygon([tuple(p) for p in poly], fill=(230, 230, 230))
    for w in walls:
        d.line(arc_points(w["start"], w["end"], w.get("curvature", 0)),
               fill=(0, 0, 0), width=max(int(w["thickness"]), 2))
    for w in walls:
        if not w["openings"]:
            continue
        line = LineString(arc_points(w["start"], w["end"], w.get("curvature", 0)))
        length = line.length or 1.0
        (sx, sy), (ex, ey) = w["start"], w["end"]
        ux, uy = (ex - sx) / length, (ey - sy) / length
        t = max(int(w["thickness"]), 2)
        for op in w["openings"]:
            half = min(op["width"], length) / 2.0
            c = line.interpolate(op["center"])
            a = (c.x - ux * half, c.y - uy * half)
            b = (c.x + ux * half, c.y + uy * half)
            d.line([a, b], fill=(255, 255, 255), width=t + 2)          # gap in wall
            if op["type"] == "window":
                d.line([a, b], fill=(70, 90, 200), width=max(t // 2, 2))  # blue pane
    return img


# ── one scene -> record ───────────────────────────────────────────────────────

def scene_to_record(annos):
    Jxy = np.array([j["coordinate"][:2] for j in annos["junctions"]], dtype=float)

    raw_rooms = []
    for sem in annos["semantics"]:
        t = sem["type"]
        if t in ("door", "window", "outwall"):
            continue
        label = S3D_ROOM_MAP.get(t, "room")
        for pid in sem["planeID"]:
            if annos["planes"][pid]["type"] != "floor":
                continue
            poly_ids = _plane_polygon(annos, pid)
            if poly_ids and len(poly_ids) >= 3:
                raw_rooms.append((label, Jxy[poly_ids]))
    if not raw_rooms:
        return None

    # normalize all coords to longest-edge = 1024
    allpts = np.concatenate([p for _, p in raw_rooms], axis=0)
    mn = allpts.min(0)
    span = max((allpts.max(0) - mn).max(), 1.0)
    scale = TARGET / span

    def norm(arr):
        return (arr - mn) * scale

    rooms = [(label, norm(poly)) for label, poly in raw_rooms]
    thickness = config.S3D_WALL_THICKNESS
    walls, room_walls = rooms_to_walls(rooms, thickness, fit_curves=config.FIT_CURVES)
    if not walls:
        return None

    openings = []
    for sem in annos["semantics"]:
        if sem["type"] not in ("door", "window"):
            continue
        jids = _opening_junctions(annos, sem["planeID"])
        if len(jids) < 2:
            continue
        pts = norm(Jxy[jids])
        lo, hi = pts.min(0), pts.max(0)
        center = [round((lo[0] + hi[0]) / 2), round((lo[1] + hi[1]) / 2)]
        width = max(round(float(np.hypot(hi[0] - lo[0], hi[1] - lo[1]))), 1)
        openings.append({"type": sem["type"], "center": center, "width": width})
    _assign_openings(walls, openings)

    result = {"walls": walls,
              "rooms": [{"label": lab, "walls": ids} for lab, ids in room_walls if ids]}

    hi = norm(allpts).max(0)
    img = _render(rooms, walls, int(round(hi[0])) + 1, int(round(hi[1])) + 1)
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


def build_struct3d_records(s3d_dir, max_samples=None, want_records=True):
    download_annotations()
    files = sorted(glob.glob(os.path.join(s3d_dir, "**", "annotation_3d.json"), recursive=True))
    print(f"[s3d] found {len(files)} scenes under {s3d_dir}")
    if max_samples:
        files = files[:max_samples]
    render_dir = config.S3D_RENDER_DIR
    os.makedirs(render_dir, exist_ok=True)

    records, annotations, errors = [], [], 0
    for i, fp in enumerate(files):
        if i % 200 == 0:
            print(f"[s3d]   {i}/{len(files)} ({len(annotations)} ok, {errors} err)")
        try:
            with open(fp) as f:
                annos = json.load(f)
            res = scene_to_record(annos)
            if res is None:
                errors += 1
                continue
            img, jd = res
            js = json.dumps(jd, separators=(",", ":"))
            if len(js) > config.MAX_JSON_CHARS:
                continue
            scene = os.path.basename(os.path.dirname(fp))
            img_path = os.path.abspath(os.path.join(render_dir, f"{scene}.png"))
            img.save(img_path)
            annotations.append({"image_path": img_path, "json_annotation": js})
            if want_records:
                records.append(_record(img, js))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"[s3d]   err {fp}: {e}")
    print(f"[s3d] built {len(annotations)} samples ({errors} errors)")
    return records, annotations


def _debug(path):
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "**", "annotation_3d.json"), recursive=True)
        if not cands:
            print("no annotation_3d.json under", path)
            return
        path = sorted(cands)[0]
    with open(path) as f:
        annos = json.load(f)
    res = scene_to_record(annos)
    if res is None:
        print("conversion returned None (no room floor planes found)")
        return
    img, jd = res
    print(f"file={path}")
    print(f"image size={img.size}  walls={len(jd['walls'])}  rooms={len(jd['rooms'])}  "
          f"openings={sum(len(w['openings']) for w in jd['walls'])}")
    print("room labels:", sorted({r['label'] for r in jd['rooms']}))
    overlay = img.copy()
    d = ImageDraw.Draw(overlay)
    for w in jd["walls"]:
        d.line([tuple(w["start"]), tuple(w["end"])], fill=(255, 0, 0), width=2)
    out = os.path.basename(os.path.dirname(path)) + "_debug.png"
    overlay.save(out)
    print(f"saved overlay -> {out} (red = walls). Open it to verify.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.data_struct3d <annotation_3d.json | scene_dir | dataset_dir>")
        raise SystemExit(1)
    _debug(sys.argv[1])
