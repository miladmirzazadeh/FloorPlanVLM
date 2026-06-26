"""Curved-wall geometry, shared across parsers, rewards, and renderers.

A wall is straight when ``curvature == 0`` and a circular-ish arc otherwise.
Encoding (matches the paper's scalar kappa): signed **sagitta ratio**

    curvature = 2 * h / L

where L is the chord length (|end - start|) and h is the signed perpendicular
height of the arc's apex above the chord. 0 = straight, +/-1 = semicircle; the
sign is the bulge direction. It's scale-invariant (coords are already normalized
to 1024), so the same number means the same shape at any size.

We reconstruct the arc as a quadratic Bezier through start, apex, end — a robust,
wraparound-free approximation of a circular arc that is plenty accurate for
rendering, IoU, and the GRPO reward signal.
"""
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

CURVE_EPS = 1e-3


def arc_points(a, b, curvature, n=24):
    """Polyline (list of (x,y)) from a to b bulging by `curvature`."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    chord = b - a
    L = float(np.hypot(*chord))
    if L < 1e-6 or abs(curvature) < CURVE_EPS:
        return [(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))]
    h = curvature * L / 2.0                      # signed sagitta
    u = chord / L
    nrm = np.array([-u[1], u[0]])                # unit perpendicular
    M = (a + b) / 2.0
    C = M + 2.0 * h * nrm                         # Bezier control -> passes through apex at t=0.5
    ts = np.linspace(0.0, 1.0, n + 1)
    pts = [(1 - t) ** 2 * a + 2 * t * (1 - t) * C + t ** 2 * b for t in ts]
    return [(float(p[0]), float(p[1])) for p in pts]


def fit_curvature(points):
    """Signed sagitta ratio of an ordered polyline relative to its endpoints."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return 0.0
    a, b = p[0], p[-1]
    chord = b - a
    L = float(np.hypot(*chord))
    if L < 1e-6:
        return 0.0
    u = chord / L
    nrm = np.array([-u[1], u[0]])
    offs = (p[1:-1] - a) @ nrm
    h = offs[int(np.argmax(np.abs(offs)))]
    return float(2.0 * h / L)


def wall_polyline(wall, n=24):
    return arc_points(wall["start"], wall["end"], wall.get("curvature", 0) or 0, n=n)


def wall_to_polygon(wall):
    """Thickened wall footprint (arc-aware), for IoU."""
    line = LineString(wall_polyline(wall))
    t = max(wall.get("thickness", 10), 1)
    return line.buffer(t / 2.0, cap_style=2)


def walls_union(walls):
    polys = []
    for w in walls:
        try:
            polys.append(wall_to_polygon(w))
        except Exception:
            pass
    if not polys:
        return None
    u = unary_union(polys)
    return None if u.is_empty else u


# ── room polygons -> wall list (shared by MSD + Structured3D) ──────────────────

def _split_runs(poly, corner_deg):
    """Split a closed polygon into vertex-runs at sharp corners."""
    n = len(poly)
    d = []
    for i in range(n):
        v = poly[(i + 1) % n] - poly[i]
        L = np.hypot(*v)
        d.append(v / L if L > 1e-9 else v * 0.0)
    corners = []
    for i in range(n):
        ang = np.degrees(np.arccos(np.clip(float(d[i - 1] @ d[i]), -1.0, 1.0)))
        if ang > corner_deg:
            corners.append(i)
    if not corners:
        return [[poly[i] for i in range(n)] + [poly[0]]]
    runs = []
    for k in range(len(corners)):
        s, e = corners[k], corners[(k + 1) % len(corners)]
        idx, i = [], s
        while True:
            idx.append(i)
            if i == e:
                break
            i = (i + 1) % n
        runs.append([poly[j] for j in idx])
    return runs


def _polygon_segments(poly, fit_curves, tol, corner_deg, curve_thresh):
    n = len(poly)
    if n < 2:
        return []
    if not fit_curves:
        out = []
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            if np.hypot(a[0] - b[0], a[1] - b[1]) >= tol:
                out.append((a, b, 0.0))
        return out
    segs = []
    for run in _split_runs(poly, corner_deg):
        run = [np.asarray(p, float) for p in run]
        if len(run) >= 3:
            k = fit_curvature(run)
            if abs(k) >= curve_thresh:
                segs.append((run[0], run[-1], k))
                continue
        for i in range(len(run) - 1):
            a, b = run[i], run[i + 1]
            if np.hypot(a[0] - b[0], a[1] - b[1]) >= tol:
                segs.append((a, b, 0.0))
    return segs


def rooms_to_walls(rooms, thickness, fit_curves=False, tol=2.0,
                   corner_deg=35.0, curve_thresh=0.08):
    """rooms: list[(label, Nx2 array)] -> (walls, [(label, [wall_id,...])]).

    Decomposes each room polygon into edges, dedups edges shared between rooms.
    With fit_curves=True, smooth multi-vertex runs become a single curved wall.
    """
    def canon(p):
        return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)

    walls, index, room_walls = [], {}, []
    for label, poly in rooms:
        poly = np.asarray(poly, float)
        ids = []
        for a, b, curv in _polygon_segments(poly, fit_curves, tol, corner_deg, curve_thresh):
            key = tuple(sorted([canon(a), canon(b)]))
            if key not in index:
                wid = f"wall_{len(walls) + 1}"
                index[key] = wid
                walls.append({
                    "id": wid,
                    "start": [round(float(a[0])), round(float(a[1]))],
                    "end": [round(float(b[0])), round(float(b[1]))],
                    "thickness": max(round(thickness), 1),
                    "curvature": round(float(curv), 3),
                    "openings": [],
                })
            if index[key] not in ids:
                ids.append(index[key])
        room_walls.append((label, ids))
    return walls, room_walls
