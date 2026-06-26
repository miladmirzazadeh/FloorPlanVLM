"""CubiCasa5K -> FloorPlanVLM JSON dataset.

Geometry parsing (SVG -> walls/openings/rooms) is adapted from the community
reference (manitocross/floorplan-vlm-training). Additions vs. the reference:
  * writes annotations.json (image_path + json) so the GRPO stage can train on
    REAL data instead of silently falling back to synthetic plans;
  * a `want_records` switch so the GRPO box can rebuild annotations cheaply
    (no PIL images held in RAM) after a fresh pod where the volume was lost;
  * a train/eval split for best-model tracking during SFT.
"""
import os
import json
import zipfile
import subprocess
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
from xml.dom import minidom
from shapely.geometry import LineString, Polygon, Point
from datasets import Dataset

from . import config
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .taxonomy import CUBICASA_ROOM_MAP as ROOM_MAP


# ── Download & extract ────────────────────────────────────────────────────────

def download_and_extract():
    """Download CubiCasa5K from Zenodo and extract. Idempotent."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    for d in os.listdir(config.DATA_DIR):
        dp = os.path.join(config.DATA_DIR, d)
        if os.path.isdir(dp) and d != "__MACOSX":
            count = 0
            for _, _, files in os.walk(dp):
                if "model.svg" in files:
                    count += 1
                    if count >= 10:
                        print(f"[data] already extracted at {dp}")
                        return dp

    zip_path = os.path.join(config.DATA_DIR, "cubicasa5k.zip")
    if not os.path.exists(zip_path):
        print("[data] downloading CubiCasa5K (~5GB) from Zenodo ...")
        try:
            subprocess.run(
                ["wget", "-q", "--show-progress", config.ZENODO_URL, "-O", zip_path],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("[data] wget unavailable/failed, using urllib ...")
            urllib.request.urlretrieve(config.ZENODO_URL, zip_path)

    print("[data] extracting ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(config.DATA_DIR)

    for d in os.listdir(config.DATA_DIR):
        dp = os.path.join(config.DATA_DIR, d)
        if os.path.isdir(dp) and d != "__MACOSX":
            return dp
    return config.DATA_DIR


# ── SVG -> JSON ───────────────────────────────────────────────────────────────

def _parse_polygon(element):
    for child in element.childNodes:
        if child.nodeName == "polygon":
            X, Y = [], []
            for p in child.getAttribute("points").split(" "):
                p = p.strip()
                if "," in p:
                    a, b = (p.split(",") + ["", ""])[:2]
                    try:
                        X.append(float(a))
                        Y.append(float(b))
                    except ValueError:
                        pass
            if len(X) >= 3:
                return np.array(X), np.array(Y)
    return None, None


def parse_floorplan(svg_path, img_path):
    """Parse one CubiCasa5K SVG + image -> FloorPlanVLM JSON dict (or None)."""
    img = Image.open(img_path)
    w, h = img.size
    scale = 1024.0 / max(w, h)

    svg = minidom.parse(svg_path)
    walls, openings, rooms = [], [], []

    for e in svg.getElementsByTagName("g"):
        eid = e.getAttribute("id")
        ecls = e.getAttribute("class")

        if eid == "Wall":
            X, Y = _parse_polygon(e)
            if X is None or len(X) < 4:
                continue
            X, Y = X * scale, Y * scale
            dx, dy = abs(max(X) - min(X)), abs(max(Y) - min(Y))
            if dx < 3 and dy < 3:
                continue
            if dx > dy:
                cy = round((min(Y) + max(Y)) / 2)
                start, end = [round(min(X)), cy], [round(max(X)), cy]
                thickness = max(round(dy), 1)
            else:
                cx = round((min(X) + max(X)) / 2)
                start, end = [cx, round(min(Y))], [cx, round(max(Y))]
                thickness = max(round(dx), 1)
            walls.append({"start": start, "end": end, "thickness": thickness,
                          "centerline": LineString([start, end])})
            for child in e.getElementsByTagName("g"):
                cid = child.getAttribute("id")
                if cid in ("Door", "Window"):
                    cX, cY = _parse_polygon(child)
                    if cX is not None and len(cX) >= 3:
                        cX, cY = cX * scale, cY * scale
                        center = [round(float(np.mean(cX))), round(float(np.mean(cY)))]
                        ow = max(round(max(abs(max(cX) - min(cX)), abs(max(cY) - min(cY)))), 1)
                        openings.append({"type": cid.lower(), "center_point": center, "width": ow})

        elif eid in ("Door", "Window"):
            parent_id = e.parentNode.getAttribute("id") if e.parentNode else ""
            if parent_id == "Wall":
                continue
            X, Y = _parse_polygon(e)
            if X is None or len(X) < 3:
                continue
            X, Y = X * scale, Y * scale
            center = [round(float(np.mean(X))), round(float(np.mean(Y)))]
            ow = max(round(max(abs(max(X) - min(X)), abs(max(Y) - min(Y)))), 1)
            openings.append({"type": eid.lower(), "center_point": center, "width": ow})

        elif "Space " in ecls:
            name = ecls.replace("Space ", "").split(" ")[0]
            label = ROOM_MAP.get(name, "room")
            X, Y = _parse_polygon(e)
            if X is not None and len(X) >= 3:
                X, Y = X * scale, Y * scale
                try:
                    poly = Polygon(list(zip(Y, X)))
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    rooms.append({"label": label, "polygon": poly})
                except Exception:
                    pass

    if not walls:
        return None

    for op in openings:
        oc = Point(op["center_point"])
        best_i, best_d = None, float("inf")
        for i, wl in enumerate(walls):
            d = wl["centerline"].distance(oc)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d < walls[best_i]["thickness"] * 3:
            op["wall_idx"] = best_i
            op["center_along"] = round(walls[best_i]["centerline"].project(oc))

    for room in rooms:
        room["wall_ids"] = []
        for i, wl in enumerate(walls):
            try:
                if room["polygon"].boundary.distance(wl["centerline"]) < wl["thickness"] * 2:
                    room["wall_ids"].append(f"wall_{i + 1}")
            except Exception:
                pass

    result = {"walls": [], "rooms": []}
    for i, wl in enumerate(walls):
        entry = {"id": f"wall_{i + 1}", "start": wl["start"], "end": wl["end"],
                 "thickness": wl["thickness"], "curvature": 0, "openings": []}
        for op in openings:
            if op.get("wall_idx") == i:
                entry["openings"].append({"type": op["type"],
                                          "center": op["center_along"],
                                          "width": op["width"]})
        result["walls"].append(entry)
    for room in rooms:
        if room.get("wall_ids"):
            result["rooms"].append({"label": room["label"], "walls": room["wall_ids"]})
    return result


# ── Record building ───────────────────────────────────────────────────────────

def _iter_plan_dirs(data_dir):
    plans = []
    for root, _, files in os.walk(data_dir):
        if "model.svg" in files and "F1_scaled.png" in files:
            plans.append(root)
    plans.sort()
    return plans


def _build(data_dir, max_samples, want_records):
    """Return (records, annotations). records is [] when want_records=False."""
    plans = _iter_plan_dirs(data_dir)
    print(f"[data] found {len(plans)} floor plans")
    if max_samples:
        plans = plans[:max_samples]

    records, annotations, errors = [], [], 0
    for i, pdir in enumerate(plans):
        if i % 200 == 0:
            print(f"[data]   converting {i}/{len(plans)} ({len(annotations)} ok, {errors} err)")
        try:
            jd = parse_floorplan(os.path.join(pdir, "model.svg"),
                                 os.path.join(pdir, "F1_scaled.png"))
            if not jd or not jd["walls"]:
                errors += 1
                continue
            js = json.dumps(jd, separators=(",", ":"))
            if len(js) > config.MAX_JSON_CHARS:
                continue
            img_path = os.path.abspath(os.path.join(pdir, "F1_scaled.png"))
            annotations.append({"image_path": img_path, "json_annotation": js})
            if want_records:
                img = Image.open(img_path).convert("RGB")
                records.append({
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": USER_PROMPT}]},
                        {"role": "assistant", "content": [{"type": "text", "text": js}]},
                    ],
                    "images": [img],
                })
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"[data]   err {pdir}: {e}")

    print(f"[data] built {len(annotations)} samples ({errors} errors)")
    return records, annotations


def _write_annotations(annotations):
    os.makedirs(os.path.dirname(config.ANN_PATH) or ".", exist_ok=True)
    with open(config.ANN_PATH, "w") as f:
        json.dump(annotations, f)
    print(f"[data] wrote {config.ANN_PATH} ({len(annotations)} entries)")


def _load_one(name, want_records):
    """Return (records, annotations) for a single dataset, already harmonized."""
    name = name.lower()
    if name == "cubicasa":
        data_dir = download_and_extract()
        return _build(data_dir, config.MAX_SAMPLES, want_records=want_records)
    if name == "msd":
        from .data_msd import build_msd_records
        return build_msd_records(config.MSD_DIR, config.MSD_MAX_SAMPLES, want_records=want_records)
    if name in ("struct3d", "s3d", "structured3d"):
        from .data_struct3d import build_struct3d_records
        return build_struct3d_records(config.S3D_DIR, config.S3D_MAX_SAMPLES, want_records=want_records)
    raise ValueError(f"unknown dataset '{name}' (supported: cubicasa, msd, struct3d)")


def get_sft_datasets():
    """Build + harmonize + mix all configured datasets; persist annotations; split."""
    all_records, all_anns = [], []
    for name in config.DATASETS:
        try:
            recs, anns = _load_one(name, want_records=True)
            print(f"[data] {name}: {len(anns)} samples")
            all_records += recs
            all_anns += anns
        except Exception as e:
            print(f"[data] dataset '{name}' failed: {e}")

    if len(all_records) < 5:
        print("[data] insufficient real data; using synthetic fallback")
        all_records = _synthetic(20)
        all_anns = []

    _write_annotations(all_anns)
    ds = Dataset.from_list(all_records).shuffle(seed=42)
    print(f"[data] combined SFT dataset: {len(ds)} samples from {config.DATASETS}")
    if config.EVAL_RATIO > 0 and len(ds) >= 40:
        split = ds.train_test_split(test_size=config.EVAL_RATIO, seed=42)
        return split["train"], split["test"]
    return ds, None


def ensure_annotations():
    """GRPO path: rebuild combined annotations.json if missing (fresh pod)."""
    if os.path.exists(config.ANN_PATH):
        return
    all_anns = []
    for name in config.DATASETS:
        try:
            _, anns = _load_one(name, want_records=False)
            all_anns += anns
        except Exception as e:
            print(f"[data] dataset '{name}' failed: {e}")
    _write_annotations(all_anns)


def build_grpo_dataset(ann_path, max_samples):
    """Prompt-only dataset (+ images + json_gt for the reward) for GRPO."""
    ensure_annotations()
    with open(ann_path) as f:
        anns = json.load(f)
    if not anns:
        print("[data] no real annotations; synthetic GRPO fallback")
        return _synthetic_grpo(max_samples or 16)
    if max_samples:
        anns = anns[:max_samples]
    records = []
    for a in anns:
        ip = a.get("image_path")
        if not ip or not os.path.exists(ip):
            continue
        records.append({
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": USER_PROMPT}]},
            ],
            "images": [Image.open(ip).convert("RGB")],
            "json_gt": a["json_annotation"],
        })
    print(f"[data] GRPO dataset: {len(records)} samples")
    return Dataset.from_list(records)


# ── Synthetic fallbacks (only used if the Zenodo download fails) ───────────────

def _synthetic_plan(i):
    size = 256
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    s = 1024.0 / size
    m, wt, mid = 30 + i * 3, 6, size // 2 + i * 2
    d.rectangle([m, m, size - m, size - m], outline="black", width=wt)
    d.line([(m, mid), (size - m, mid)], fill="black", width=wt)
    jd = {"walls": [
        {"id": "wall_1", "start": [round(m * s), round(m * s)], "end": [round((size - m) * s), round(m * s)], "thickness": round(wt * s), "curvature": 0, "openings": []},
        {"id": "wall_2", "start": [round((size - m) * s), round(m * s)], "end": [round((size - m) * s), round((size - m) * s)], "thickness": round(wt * s), "curvature": 0, "openings": []},
        {"id": "wall_3", "start": [round((size - m) * s), round((size - m) * s)], "end": [round(m * s), round((size - m) * s)], "thickness": round(wt * s), "curvature": 0, "openings": []},
        {"id": "wall_4", "start": [round(m * s), round((size - m) * s)], "end": [round(m * s), round(m * s)], "thickness": round(wt * s), "curvature": 0, "openings": []},
        {"id": "wall_5", "start": [round(m * s), round(mid * s)], "end": [round((size - m) * s), round(mid * s)], "thickness": round(wt * s), "curvature": 0, "openings": []},
    ], "rooms": [
        {"label": "bedroom", "walls": ["wall_1", "wall_2", "wall_5", "wall_4"]},
        {"label": "living_room", "walls": ["wall_5", "wall_2", "wall_3", "wall_4"]},
    ]}
    return img, json.dumps(jd, separators=(",", ":"))


def _synthetic(n=20):
    out = []
    for i in range(n):
        img, js = _synthetic_plan(i)
        out.append({"messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": js}]},
        ], "images": [img]})
    return out


def _synthetic_grpo(n=16):
    out = []
    for i in range(n):
        img, js = _synthetic_plan(i)
        out.append({"prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
        ], "images": [img], "json_gt": js})
    return Dataset.from_list(out)
