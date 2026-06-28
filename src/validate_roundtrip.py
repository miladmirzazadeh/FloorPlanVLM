"""DATA PIPELINE CHECKLIST GATE — run before training.

Takes built records (JSONL of {"image": path, "target": minified-json-string}), decodes
each target back to coordinates, un-normalizes from [0,GRID] onto the padded square, and
draws them over the raster. If the lines don't sit on the walls, the converter math is
wrong — fix it before exposing it to the model.

Also reports an ink-alignment score per sample (fraction of drawn wall length landing on
dark pixels) and flags the worst offenders.

    python -m src.validate_roundtrip --built built/train.jsonl --out roundtrip_check --n 40
"""
import os
import sys
import json
import argparse

import numpy as np
from PIL import Image, ImageDraw

from . import config
from .normalize import pad_to_square, arc_polyline
from .schema import decode


def ink_mask(square_img, thresh=190, dilate=2):
    g = np.asarray(square_img.convert("L"))
    m = g < thresh
    out = m.copy()
    for dy in range(-dilate, dilate + 1):
        for dx in range(-dilate, dilate + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def _pts(w, f):
    """Arc-aware polyline of a wall in padded-square pixels."""
    return [(x * f, y * f) for (x, y) in arc_polyline(w["cl"], w.get("cv", 0), n=24)]


def align_frac(walls, mask, side, grid):
    H, W = mask.shape
    f = side / float(grid)
    hit = tot = 0
    for w in walls:
        pts = _pts(w, f)
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            steps = max(2, int(max(abs(xb - xa), abs(yb - ya))))
            for t in np.linspace(0, 1, steps):
                xi, yi = int(round(xa + (xb - xa) * t)), int(round(ya + (yb - ya) * t))
                tot += 1
                if 0 <= yi < H and 0 <= xi < W and mask[yi, xi]:
                    hit += 1
    return hit / tot if tot else 0.0


def render(square_img, walls, side, grid):
    im = square_img.convert("RGB")
    d = ImageDraw.Draw(im)
    f = side / float(grid)
    for w in walls:
        pts = _pts(w, f)
        d.line(pts, fill=(255, 0, 0), width=max(2, int(w["th"] * f)))
        d.line(pts, fill=(0, 120, 255), width=1)  # centerline
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", required=True, help="JSONL of {image, target}")
    ap.add_argument("--out", default="roundtrip_check")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--flag-below", type=float, default=0.6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open(a.built) if l.strip()][: a.n]
    if not rows:
        sys.exit(f"[rt] no records in {a.built}")

    scores, flagged = [], []
    for i, r in enumerate(rows):
        try:
            img = Image.open(r["image"])
        except Exception as e:
            print(f"[rt] {r.get('image')}: open failed {e}")
            continue
        sq, side = pad_to_square(img)
        walls = decode(r["target"])
        mask = ink_mask(sq)
        sc = align_frac(walls, mask, side, config.GRID)
        scores.append(sc)
        name = os.path.splitext(os.path.basename(r["image"]))[0]
        render(sq, walls, side, config.GRID).save(os.path.join(a.out, f"{name}_rt.png"))
        tag = "  <-- LOW" if sc < a.flag_below else ""
        if sc < a.flag_below:
            flagged.append((name, round(sc, 2)))
        print(f"[rt] {i+1}/{len(rows)} {name}: walls={len(walls)} align={sc:.2f}{tag}")

    if scores:
        print(f"\n[rt] mean align={np.mean(scores):.3f}  median={np.median(scores):.3f}  "
              f"flagged(<{a.flag_below})={len(flagged)}/{len(scores)}")
        if flagged:
            print("[rt] worst:", sorted(flagged, key=lambda x: x[1])[:10])
        print(f"[rt] overlays -> {a.out}/  (open a few; lines MUST sit on the walls)")


if __name__ == "__main__":
    main()
