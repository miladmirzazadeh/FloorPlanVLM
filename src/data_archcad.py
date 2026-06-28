"""ArchCAD (HF jackluoluo/ArchCAD, cc-by-nc) -> (image_path, json_annotation) records.

Local layout: ARCHCAD_DIR/json/<uuid>.json  (+ optional ARCHCAD_DIR/png/<uuid>.png).
json = {"entities":[{type:LINE|ARC|CIRCLE, start,end, center,radius,start_angle,end_angle,
        line_width, semantic:<class>, ...}]}.

CLASS MAP (verified): semantic 20 = WALLS (drawn as DOUBLE parallel lines); 100 = grid;
6 = stairs; 19/1 = openings/symbols. ~60% of drawings have NO walls (detail/section sheets)
-> skipped.

IMAGE SOURCE (two modes; PNG preferred — config.ARCHCAD_USE_PNG, auto-on when png/ exists):
  * PNG mode: use the OFFICIAL render ARCHCAD_DIR/png/<uuid>.png. VERIFIED that the json
    coords ARE the png pixels (980x980, 1:1, Y-DOWN, no flip), so walls map directly with no
    transform, and the image carries real CAD line weights / door swings / column grids.
  * render mode (fallback): rasterize all entities (black on white) ourselves with a
    Y-flip CAD->pixel transform — a thin-line sketch; used only when no png/ is present.

Pipeline per drawing:
  1. obtain the image (png or render) and the coordinate space the json walls live in.
  2. pair the semantic-20 double-lines into centerline + thickness (parallel + overlapping
     + small perpendicular gap). Skip drawings with < MIN_WALLS paired walls.
  3. emit raw walls {start,end,thickness,curvature,openings} in PIXEL space -> build_data
     re-encodes to the [0,GRID] schema. (openings: TODO; walls-only for now, which our
     schema allows.)
"""
import os
import glob
import json
import math

from PIL import Image, ImageDraw

from . import config

WALL_SEM = 20
MIN_WALLS = 6                      # skip non-plan sheets
RENDER_LONG_EDGE = 1024


def _dir(s, e):
    dx, dy = e[0] - s[0], e[1] - s[1]
    L = math.hypot(dx, dy)
    return (dx / L, dy / L, L) if L > 1e-9 else (0.0, 0.0, 0.0)


def _pair_walls(segs, span, ang_tol=8.0, min_ov=0.25, return_edges=False):
    """Double-line LINE segments -> [(centerline_start, centerline_end, thickness)].

    return_edges=True also returns, per wall, the two ORIGINAL paired edge segments
    [(seg_i, seg_j), ...] — used by the clean renderer to draw authentic hollow walls
    while skipping every UNpaired sem-20 edge (so the rendered image == the labels)."""
    th_lo, th_hi = max(1.0, 0.002 * span), 0.08 * span
    n = len(segs)
    used = [False] * n
    dirs = [_dir(*s) for s in segs]
    walls = []
    edges = []
    for i in range(n):
        if used[i] or dirs[i][2] < 1e-9:
            continue
        si, ei = segs[i]
        uix, uiy, Li = dirs[i]
        nix, niy = -uiy, uix

        def proj(p, si=si, uix=uix, uiy=uiy):
            return (p[0] - si[0]) * uix + (p[1] - si[1]) * uiy

        best = None
        for j in range(n):
            if j == i or used[j] or dirs[j][2] < 1e-9:
                continue
            ujx, ujy, Lj = dirs[j]
            if abs(uix * ujx + uiy * ujy) < math.cos(math.radians(ang_tol)):
                continue
            mj = ((segs[j][0][0] + segs[j][1][0]) / 2, (segs[j][0][1] + segs[j][1][1]) / 2)
            perp = abs((mj[0] - si[0]) * nix + (mj[1] - si[1]) * niy)
            if perp < th_lo or perp > th_hi:
                continue
            aj0, aj1 = sorted([proj(segs[j][0]), proj(segs[j][1])])
            ov = min(Li, aj1) - max(0.0, aj0)
            if ov <= 0 or ov / min(Li, Lj) < min_ov:
                continue
            if best is None or perp < best[0]:
                best = (perp, j, max(0.0, aj0), min(Li, aj1), mj)
        if best:
            perp, j, o0, o1, mj = best
            used[i] = used[j] = True
            sign = 1 if ((mj[0] - si[0]) * nix + (mj[1] - si[1]) * niy) > 0 else -1
            h = sign * perp / 2
            cs = [si[0] + o0 * uix + h * nix, si[1] + o0 * uiy + h * niy]
            ce = [si[0] + o1 * uix + h * nix, si[1] + o1 * uiy + h * niy]
            if math.hypot(ce[0] - cs[0], ce[1] - cs[1]) > 0.01 * span:
                walls.append((cs, ce, perp))
                edges.append((segs[i], segs[j]))
    return (walls, edges) if return_edges else walls


def _arc_pts(c, r, a0, a1, n=20):
    a0, a1 = math.radians(a0), math.radians(a1)
    if a1 < a0:
        a1 += 2 * math.pi
    return [(c[0] + r * math.cos(a0 + (a1 - a0) * k / n),
             c[1] + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]


def _render(entities):
    """All entities (black) -> (PIL image, to_px, scale). bbox from lines+arcs."""
    pts = []
    for e in entities:
        if e.get("start"):
            pts += [e["start"], e["end"]]
        if e.get("type") in ("ARC", "CIRCLE") and e.get("center") and e.get("radius"):
            c, r = e["center"], e["radius"]
            pts += [[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]]
    if not pts:
        return None, None, 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    m = span * 0.03
    sc = RENDER_LONG_EDGE / (span + 2 * m)
    W = max(1, int((maxx - minx + 2 * m) * sc))
    H = max(1, int((maxy - miny + 2 * m) * sc))

    def to_px(x, y):
        return ((x - minx + m) * sc, (maxy - y + m) * sc)        # Y-flip CAD->image

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for e in entities:
        t = e.get("type")
        if t == "LINE" and e.get("start"):
            d.line([to_px(*e["start"]), to_px(*e["end"])], fill=(0, 0, 0), width=1)
        elif t == "ARC" and e.get("center"):
            p = [to_px(*q) for q in _arc_pts(e["center"], e["radius"],
                                             e.get("start_angle", 0), e.get("end_angle", 360))]
            d.line(p, fill=(0, 0, 0), width=1)
        elif t == "CIRCLE" and e.get("center"):
            p = [to_px(*q) for q in _arc_pts(e["center"], e["radius"], 0, 360)]
            d.line(p, fill=(0, 0, 0), width=1)
    return img, to_px, sc, span


def convert(entities):
    """entities -> (PIL image, raw_walls in px) or (None, None)."""
    segs = [(e["start"], e["end"]) for e in entities
            if e.get("semantic") == WALL_SEM and e.get("type") == "LINE" and e.get("start")]
    if len(segs) < MIN_WALLS:
        return None, None
    rendered = _render(entities)
    if rendered[0] is None:
        return None, None
    img, to_px, sc, span = rendered
    walls_cad = _pair_walls(segs, span)
    if len(walls_cad) < MIN_WALLS:
        return None, None
    walls = []
    for cs, ce, th in walls_cad:
        sx, sy = to_px(*cs)
        ex, ey = to_px(*ce)
        walls.append({"start": [round(sx), round(sy)], "end": [round(ex), round(ey)],
                      "thickness": max(1, round(th * sc)), "curvature": 0, "openings": []})
    return img, walls


def convert_png(entities, png_w, png_h):
    """entities + official png size -> raw_walls in PNG-PIXEL space (json coords ARE png px).

    No render, no transform: pairs the semantic-20 double-lines directly in png pixels.
    Returns walls or None (too few walls)."""
    segs = [(e["start"], e["end"]) for e in entities
            if e.get("semantic") == WALL_SEM and e.get("type") == "LINE" and e.get("start")]
    if len(segs) < MIN_WALLS:
        return None
    span = float(max(png_w, png_h))                  # the thickness band scales off the frame
    walls_px = _pair_walls(segs, span)
    if len(walls_px) < MIN_WALLS:
        return None
    walls = []
    for cs, ce, th in walls_px:
        walls.append({"start": [round(cs[0]), round(cs[1])], "end": [round(ce[0]), round(ce[1])],
                      "thickness": max(1, round(th)), "curvature": 0, "openings": []})
    return walls


def build_archcad_records(archcad_dir, max_samples=None, want_records=False):
    files = sorted(glob.glob(os.path.join(archcad_dir, "json", "*.json")))
    if not files:                                   # robust to HF layout: find entity jsons anywhere
        files = sorted(glob.glob(os.path.join(archcad_dir, "**", "*.json"), recursive=True))
    if not files:
        print(f"[archcad] no json found under {archcad_dir} — set ARCHCAD_DIR.")
        return [], []

    png_dir = os.path.join(archcad_dir, "png")
    use_png = config.ARCHCAD_USE_PNG and os.path.isdir(png_dir)
    img_dir = os.path.join(archcad_dir, "rendered")
    if use_png:
        print(f"[archcad] {len(files)} json to scan — IMAGE: official png/ "
              f"(json coords = png pixels, no render)...", flush=True)
    else:
        os.makedirs(img_dir, exist_ok=True)
        print(f"[archcad] {len(files)} json to scan — IMAGE: rendering from json "
              f"(no png/ dir found)...", flush=True)

    anns, kept, skipped, no_png = [], 0, 0, 0
    for i, f in enumerate(files):
        if max_samples and kept >= max_samples:
            break
        if i and i % 2000 == 0:
            print(f"[archcad]   {i}/{len(files)} scanned ({kept} kept, {skipped} skipped)", flush=True)
        try:
            name = os.path.splitext(os.path.basename(f))[0]
            if use_png:
                p = os.path.join(png_dir, f"{name}.png")
                if not os.path.exists(p):            # check png BEFORE loading json (fast skip)
                    no_png += 1
                    skipped += 1
                    continue
                ents = json.load(open(f))["entities"]
                with Image.open(p) as im:            # header only — no full decode
                    pw, ph = im.size
                walls = convert_png(ents, pw, ph)
                if walls is None:
                    skipped += 1
                    continue
            else:
                ents = json.load(open(f))["entities"]
                img, walls = convert(ents)
                if img is None:
                    skipped += 1
                    continue
                p = os.path.join(img_dir, f"{name}.png")
                img.save(p)
            anns.append({"image_path": os.path.abspath(p),
                         "json_annotation": json.dumps({"walls": walls}, separators=(",", ":"))})
            kept += 1
        except Exception:
            skipped += 1
    extra = f" ({no_png} missing png)" if no_png else ""
    src = "official png" if use_png else f"rendered -> {img_dir}"
    print(f"[archcad] {kept} plans kept, {skipped} skipped (non-plan/too-few-walls{extra}) — image: {src}")
    return [], anns
