"""Coordinate canonicalization — the geometric discipline behind the dataset.

Turns a converter's RAW walls (pixel start/end + thickness + openings) into the
canonical, deterministic form the model trains on:

  1. pad (never distort) the image to a square; scale ALL coords by GRID/longest_edge
     so the [0,GRID] grid maps 1:1 onto the padded square (encoder ↔ labels aligned).
  2. order each centerline endpoints so x1<=x2 (tie: y1<=y2) — one token sequence per
     visual wall (and flip opening offsets to match).
  3. sort walls: exterior boundary clockwise (from top), then interior partitions
     top-left → bottom-right — so autoregressive generation follows a stable order.

Output wall (pre-encode):  {"cl":[x1,y1,x2,y2], "th":T, "op":[{"t":"door"|"window","c":C,"w":W}]}
"""
import math

from PIL import Image

from . import config


def pad_to_square(img, fill=(255, 255, 255)):
    """Pad bottom/right to a square (no distortion; coords keep their origin)."""
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    if w == h:
        return img, side
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, (0, 0))
    return canvas, side


def _clamp(v, g):
    return max(0, min(g, int(round(v))))


def _order(cl, ops, cv):
    """Enforce x1<=x2 (tie y1<=y2). On flip: mirror opening offsets (c->L-c) AND
    negate curvature (an arc bulging one way A->B bulges the other way B->A)."""
    x1, y1, x2, y2 = cl
    if (x1, y1) <= (x2, y2):
        return cl, ops, cv
    L = math.hypot(x2 - x1, y2 - y1)
    ops = [{"t": o["t"], "c": int(round(L - o["c"])), "w": o["w"]} for o in ops]
    return [x2, y2, x1, y1], ops, -cv


def arc_polyline(cl, cv, n=24):
    """Points tracing the wall: straight if |cv|~0, else a circular arc with signed
    sagitta-ratio curvature cv (h = cv*L/2). Pure math (no shapely) for rendering."""
    ax, ay, bx, by = cl
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6 or abs(cv) < 1e-3:
        return [(ax, ay), (bx, by)]
    h = cv * L / 2.0
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    R = (L * L / 4.0 + h * h) / (2.0 * abs(h))
    c = -math.copysign(1.0, h) * math.sqrt(max(R * R - L * L / 4.0, 0.0))
    ox, oy = mx + c * nx, my + c * ny
    th_a = math.atan2(ay - oy, ax - ox)
    th_b = math.atan2(by - oy, bx - ox)
    px, py = mx + h * nx, my + h * ny
    th_p = math.atan2(py - oy, px - ox)

    def wrap(x):
        while x <= -math.pi:
            x += 2 * math.pi
        while x > math.pi:
            x -= 2 * math.pi
        return x

    d = wrap(th_b - th_a) or math.pi
    if not (0.0 <= wrap(th_p - th_a) / d <= 1.0):
        d -= math.copysign(2 * math.pi, d)
    return [(ox + R * math.cos(th_a + d * (i / n)), oy + R * math.sin(th_a + d * (i / n)))
            for i in range(n + 1)]


def _sort(walls, grid):
    """Exterior walls clockwise from the top, then interior walls top-left→bottom-right."""
    if not walls:
        return walls
    xs = [c for w in walls for c in (w["cl"][0], w["cl"][2])]
    ys = [c for w in walls for c in (w["cl"][1], w["cl"][3])]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    margin = 0.04 * grid

    def mid(w):
        return ((w["cl"][0] + w["cl"][2]) / 2.0, (w["cl"][1] + w["cl"][3]) / 2.0)

    def is_ext(w):
        mx, my = mid(w)
        return mx <= minx + margin or mx >= maxx - margin or my <= miny + margin or my >= maxy - margin

    def cw(w):                                  # clockwise angle from 12 o'clock
        mx, my = mid(w)
        a = math.atan2(mx - cx, -(my - cy))
        return a if a >= 0 else a + 2 * math.pi

    ext = sorted([w for w in walls if is_ext(w)], key=cw)
    int_ = sorted([w for w in walls if not is_ext(w)],
                  key=lambda w: (min(w["cl"][1], w["cl"][3]), min(w["cl"][0], w["cl"][2])))
    return ext + int_


def canonicalize(raw_walls, img_w, img_h, grid=None, order=None, sort=None):
    """RAW walls (px start/end/thickness/openings) -> canonical [0,grid] walls."""
    grid = config.GRID if grid is None else grid
    order = config.ORDER_ENDPOINTS if order is None else order
    sort = config.SORT_WALLS if sort is None else sort
    side = max(img_w, img_h) or 1
    s = grid / float(side)

    walls = []
    for w in raw_walls:
        st, en = w.get("start"), w.get("end")
        if not (isinstance(st, (list, tuple)) and isinstance(en, (list, tuple))
                and len(st) == 2 and len(en) == 2):
            continue
        cl = [_clamp(st[0] * s, grid), _clamp(st[1] * s, grid),
              _clamp(en[0] * s, grid), _clamp(en[1] * s, grid)]
        if cl[0] == cl[2] and cl[1] == cl[3]:
            continue                                   # zero-length after rounding
        th = max(1, int(round(max(w.get("thickness", 1), 1) * s)))
        cv = float(w.get("curvature", 0) or 0)   # sagitta ratio: scale-invariant, no *s
        ops = []
        for op in (w.get("openings") or []):
            t = str(op.get("type", "door")).lower()
            ops.append({"t": "window" if t.startswith("w") else "door",
                        "c": _clamp(op.get("center", 0) * s, grid),
                        "w": max(1, int(round(op.get("width", 0) * s)))})
        if order:
            cl, ops, cv = _order(cl, ops, cv)
        walls.append({"cl": cl, "th": th, "cv": cv, "op": ops})

    return _sort(walls, grid) if sort else walls
