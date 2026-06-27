"""Convert model-output JSON (walls) into a DXF for CAD precision-checking.

LOCAL use (your Mac):
    pip install ezdxf
    python scripts/json_to_dxf.py eval_results/plan1.json plan1.dxf      # one file
    python scripts/json_to_dxf.py eval_results/ dxf_out/                 # whole folder

Handles both the batch_infer wrapper ({"prediction": {"walls": …}}) and a raw
{"walls": […]} object. Coordinates are the model's 1024-normalized space (longer
edge = 1024); Y is flipped so the plan is north-up in CAD.

Layers:  WALLS (thick wall body)  ·  WALLS_CL (centerlines)  ·  OPENINGS (doors/windows)
Curved walls (curvature != 0) are reconstructed as true circular arcs.
"""
import os
import sys
import json
import math
import glob

import ezdxf

CANVAS = 1024  # normalization space; used to flip Y for CAD orientation


def arc_points(a, b, curvature, n=28):
    """Polyline tracing the circular arc from a to b with signed sagitta-ratio curvature."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6 or abs(curvature) < 1e-3:
        return [(ax, ay), (bx, by)]
    h = curvature * L / 2.0
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


def _band(pts, half):
    """Offset a polyline by +/- half on each side -> closed wall-body polygon."""
    n = len(pts)
    left, right = [], []
    for i in range(n):
        if i == 0:
            dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        elif i == n - 1:
            dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        x, y = pts[i]
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    return left + right[::-1]


def _point_at(pts, dist):
    acc = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= dist:
            t = (dist - acc) / seg if seg > 0 else 0.0
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        acc += seg
    return pts[-1]


def load_plan(path):
    data = json.load(open(path))
    if isinstance(data, dict) and isinstance(data.get("prediction"), dict):
        data = data["prediction"]
    return data.get("walls", []) if isinstance(data, dict) else []


def to_dxf(walls, out_path):
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    for name, color in (("WALLS", 8), ("WALLS_CL", 1), ("OPENINGS", 5)):
        if name not in doc.layers:
            doc.layers.add(name, color=color)

    def fy(y):
        return CANVAS - y  # flip Y for CAD north-up

    for w in walls:
        s, e = w.get("start"), w.get("end")
        if not (s and e):
            continue
        cv = w.get("curvature", 0) or 0
        th = max(float(w.get("thickness", 8)), 1.0)
        cl = arc_points(s, e, cv)
        clf = [(x, fy(y)) for x, y in cl]
        msp.add_lwpolyline(clf, dxfattribs={"layer": "WALLS_CL"})
        body = [(x, fy(y)) for x, y in _band(cl, th / 2.0)]
        msp.add_lwpolyline(body, close=True, dxfattribs={"layer": "WALLS"})
        for op in w.get("openings", []):
            try:
                cx, cy = _point_at(cl, float(op.get("center", 0)))
                r = max(float(op.get("width", 10)) / 2.0, 2.0)
                msp.add_circle((cx, fy(cy)), r, dxfattribs={"layer": "OPENINGS"})
            except Exception:
                pass

    doc.saveas(out_path)
    return len(walls)


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/json_to_dxf.py <in.json|in_dir> [out.dxf|out_dir]")
        sys.exit(1)
    src = sys.argv[1]
    if os.path.isdir(src):
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "dxf_out"
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in glob.glob(os.path.join(src, "*.json")) if not f.endswith("_summary.json")]
        for f in sorted(files):
            name = os.path.splitext(os.path.basename(f))[0]
            try:
                n = to_dxf(load_plan(f), os.path.join(out_dir, f"{name}.dxf"))
                print(f"{name}.dxf  ({n} walls)")
            except Exception as e:
                print(f"{name}: FAILED {e}")
        print(f"-> {out_dir}/")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".dxf"
        n = to_dxf(load_plan(src), out)
        print(f"wrote {out} ({n} walls)")


if __name__ == "__main__":
    main()
