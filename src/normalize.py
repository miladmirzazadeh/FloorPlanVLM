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


def _mid(w):
    return ((w["cl"][0] + w["cl"][2]) / 2.0, (w["cl"][1] + w["cl"][3]) / 2.0)


def _sort(walls, grid):
    """Deterministic, identical-for-every-image READING ORDER: top-left wall first, then
    left→right across each horizontal band, then the next band down. y is bucketed into rows
    (band = 4% of grid) so walls at the same height are grouped left→right rather than split
    by a one-pixel y difference. Robust on partial/non-rectangular plans (no exterior split)."""
    band = max(1.0, 0.04 * grid)

    def key(w):
        x1, y1, x2, y2 = w["cl"]
        return (int(min(y1, y2) // band), min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    return sorted(walls, key=key)            # full geometric key -> stable & input-order-independent


def canonicalize(raw_walls, img_w, img_h, grid=None, order=None, sort=None, rooms=None):
    """RAW walls (+ optional rooms referencing border-wall ids) -> (walls, rooms).

      walls = [{cl,th,cv,op}]  in [0,grid], ordered + sorted.
      rooms = [{t, w:[wall ids]}]  FloorplanVLM-style: a room = its set of BORDER walls;
              wall ids are remapped to the final sorted 1..N order (kept if >=3 survive).
    """
    grid = config.GRID if grid is None else grid
    order = config.ORDER_ENDPOINTS if order is None else order
    sort = config.SORT_WALLS if sort is None else sort
    side = max(img_w, img_h) or 1
    s = grid / float(side)

    walls = []
    for idx, w in enumerate(raw_walls):
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
        walls.append({"cl": cl, "th": th, "cv": cv, "op": ops, "_src": w.get("id", idx)})

    if sort:
        walls = _sort(walls, grid)

    out_rooms = []
    if rooms and config.ROOMS:
        id_map = {w["_src"]: i + 1 for i, w in enumerate(walls)}   # source id -> final wall id
        final = {i + 1: w for i, w in enumerate(walls)}            # final id -> wall
        for r in (rooms or []):
            refs = list({id_map[wid] for wid in (r.get("walls") or []) if wid in id_map})
            if len(refs) >= 3:                                     # a real enclosed room
                # ORDER the border walls as a boundary walk: clockwise around the room centroid
                # (FloorplanVLM Eq.5 wants an ORDERED sequence of wall ids, not a sorted set).
                rcx = sum(_mid(final[i])[0] for i in refs) / len(refs)
                rcy = sum(_mid(final[i])[1] for i in refs) / len(refs)

                def _cw(i, rcx=rcx, rcy=rcy):
                    mx, my = _mid(final[i])
                    a = math.atan2(mx - rcx, -(my - rcy))
                    return a if a >= 0 else a + 2 * math.pi

                refs = sorted(refs, key=_cw)
                out_rooms.append({"t": r.get("label") or r.get("room_type") or "room", "w": refs})
    for w in walls:
        w.pop("_src", None)
    return walls, out_rooms
