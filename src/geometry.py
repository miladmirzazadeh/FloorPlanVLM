"""Curved-wall geometry, shared across parsers, rewards, and renderers.

A wall is straight when ``curvature == 0`` and a circular-ish arc otherwise.
Encoding (matches the paper's scalar kappa): signed **sagitta ratio**

    curvature = 2 * h / L

where L is the chord length (|end - start|) and h is the signed perpendicular
height of the arc's apex above the chord. 0 = straight, +/-1 = semicircle; the
sign is the bulge direction. It's scale-invariant (coords are already normalized
to 1024), so the same number means the same shape at any size.

We reconstruct a TRUE circular arc from (start, end, curvature): the sagitta ratio
plus the two endpoints uniquely determine a circle, so the rendered/scored curve
matches the real arc (a semicircle is a real semicircle, not a parabola).
"""
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize

CURVE_EPS = 1e-3


def _wrap(x):
    """Wrap angle to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def arc_points(a, b, curvature, n=24):
    """Polyline (list of (x,y)) tracing the circular arc from a to b with the given
    signed sagitta-ratio `curvature`."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    chord = b - a
    L = float(np.hypot(*chord))
    if L < 1e-6 or abs(curvature) < CURVE_EPS:
        return [(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))]
    h = curvature * L / 2.0                       # signed sagitta
    u = chord / L
    nrm = np.array([-u[1], u[0]])                 # unit perpendicular
    M = (a + b) / 2.0
    R = (L * L / 4.0 + h * h) / (2.0 * abs(h))    # circumradius
    c = -np.sign(h) * np.sqrt(max(R * R - L * L / 4.0, 0.0))  # center offset (opposite the apex)
    O = M + c * nrm
    P = M + h * nrm                               # apex (defines which way we sweep)
    th_a = np.arctan2(a[1] - O[1], a[0] - O[0])
    th_b = np.arctan2(b[1] - O[1], b[0] - O[0])
    th_p = np.arctan2(P[1] - O[1], P[0] - O[0])
    d = _wrap(th_b - th_a) or np.pi
    if not (0.0 <= _wrap(th_p - th_a) / d <= 1.0):   # ensure the apex lies on the swept arc
        d -= np.sign(d) * 2.0 * np.pi
    pts = [O + R * np.array([np.cos(th_a + d * t), np.sin(th_a + d * t)])
           for t in np.linspace(0.0, 1.0, n + 1)]
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


def poly_iou(a, b):
    if a is None or b is None:
        return 0.0
    try:
        if not a.is_valid:
            a = a.buffer(0)
        if not b.is_valid:
            b = b.buffer(0)
        inter = a.intersection(b).area
        uni = a.union(b).area
        return inter / uni if uni > 0 else 0.0
    except Exception:
        return 0.0


def wall_faces(walls):
    """Closed regions ('rooms' as geometry) formed by polygonizing the wall network.
    Hanging/unclosed walls produce no face -> this is how topology gets scored."""
    lines = []
    for w in walls:
        try:
            lines.append(LineString(wall_polyline(w)))
        except Exception:
            pass
    if len(lines) < 3:
        return []
    try:
        return [p for p in polygonize(unary_union(lines)) if p.area > 1.0]
    except Exception:
        return []


def region_iou(pred_walls, gt_walls, thr=0.5):
    """Mean IoU of matched wall-enclosed regions (label-free topology score):
    rewards walls that CLOSE and partition space like the ground truth."""
    P, G = wall_faces(pred_walls), wall_faces(gt_walls)
    if not G:
        return 0.0
    used, ious = set(), []
    for gp in G:
        best, bj = -1.0, -1
        for j, pp in enumerate(P):
            if j in used:
                continue
            iou = poly_iou(gp, pp)
            if iou > best:
                best, bj = iou, j
        if bj >= 0 and best >= thr:
            used.add(bj)
            ious.append(best)
    return float(np.mean(ious)) if ious else 0.0


def topology_ok(walls, tol=8.0, min_junction_frac=0.85, min_walls=4):
    """Reject topologically-broken plans (floating walls, unclosed loops).
    A plan passes if (1) most wall endpoints touch another wall's SEGMENT — counts
    both corner and T-junctions, only true free ends are 'hanging' — and (2) the
    walls enclose at least one face. Tol is in normalized-1024 px."""
    from shapely.geometry import Point
    walls = [w for w in walls if isinstance(w.get("start"), list) and isinstance(w.get("end"), list)]
    if len(walls) < min_walls:
        return False
    lines = [LineString(wall_polyline(w)) for w in walls]
    connected = total = 0
    for i, w in enumerate(walls):
        for end in (w["start"], w["end"]):
            total += 1
            p = Point(end)
            if any(j != i and lines[j].distance(p) <= tol for j in range(len(walls))):
                connected += 1
    if total == 0:
        return False
    return (connected / total) >= min_junction_frac and len(wall_faces(walls)) >= 1


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
