"""Pure 2D geometry helpers and the model->pixel transform.

All floor-plan geometry lives in millimetres in a y-up coordinate system
(the same convention DXF and matplotlib use). Images are y-down, so the
``ModelTransform`` flips y when mapping to pixels.

Nothing in here touches ezdxf, matplotlib or shapely - it is the small,
exact maths layer that the rest of the engine and the tests rely on.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


def to_np(p: Sequence[float]) -> np.ndarray:
    return np.asarray(p, dtype=float)


def length(v: Sequence[float]) -> float:
    v = to_np(v)
    return float(math.hypot(v[0], v[1]))


def unit(v: Sequence[float]) -> np.ndarray:
    v = to_np(v)
    n = length(v)
    return v / n if n > 1e-12 else v.copy()


def perp(v: Sequence[float]) -> np.ndarray:
    """Left normal: rotate +90 degrees CCW."""
    v = to_np(v)
    return np.array([-v[1], v[0]])


def angle_deg(v: Sequence[float]) -> float:
    v = to_np(v)
    return math.degrees(math.atan2(v[1], v[0]))


def cross(a: Sequence[float], b: Sequence[float]) -> float:
    a = to_np(a)
    b = to_np(b)
    return float(a[0] * b[1] - a[1] * b[0])


def rotate_vec(v: Sequence[float], deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    v = to_np(v)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def rotate_point(p: Sequence[float], deg: float, origin: Sequence[float] = (0.0, 0.0)) -> Point:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    px, py = p[0] - origin[0], p[1] - origin[1]
    return (origin[0] + px * c - py * s, origin[1] + px * s + py * c)


def arc_points(center: Sequence[float], radius: float, a0_deg: float, a1_deg: float,
               segments: int = 48) -> List[Point]:
    """Sample an arc as a polyline. Direction follows the sign of a1-a0."""
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    pts: List[Point] = []
    for i in range(segments + 1):
        t = i / segments
        a = a0 + (a1 - a0) * t
        pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
    return pts


def bbox_of_points(points: Iterable[Sequence[float]]) -> BBox:
    arr = np.asarray(list(points), dtype=float)
    return (float(arr[:, 0].min()), float(arr[:, 1].min()),
            float(arr[:, 0].max()), float(arr[:, 1].max()))


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    """Intersection of the infinite lines (p1,p2) and (p3,p4). None if parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


class ModelTransform:
    """Maps *unrotated* model-space mm to pixel coordinates.

    Construction guarantees the mapping matches the rendered PNG exactly:
    the renderer fills the whole canvas with one axes whose data limits are
    ``[xlim0, xlim0+range_x] x [ylim0, ylim0+range_y]`` and whose figure
    aspect equals the data aspect, so there is no letter-boxing.

    Global plan rotation (if any) is folded in here: a point is first rotated
    about ``rot_origin`` by ``rotation_deg`` (matching the rotation applied to
    the DXF entities), then projected. Components record their boxes in the
    unrotated frame, so the exporter can feed raw corners straight in.
    """

    def __init__(self, *, rotation_deg: float, rot_origin: Point,
                 xlim0: float, ylim0: float, range_x: float, range_y: float,
                 img_w: int, img_h: int):
        self.rotation_deg = rotation_deg
        self.rot_origin = rot_origin
        self.xlim0 = xlim0
        self.ylim0 = ylim0
        self.range_x = range_x
        self.range_y = range_y
        self.img_w = img_w
        self.img_h = img_h

    def to_px(self, x: float, y: float) -> Point:
        if abs(self.rotation_deg) > 1e-9:
            x, y = rotate_point((x, y), self.rotation_deg, self.rot_origin)
        px = (x - self.xlim0) / self.range_x * self.img_w
        py = self.img_h - (y - self.ylim0) / self.range_y * self.img_h
        return (px, py)

    def bbox_to_px(self, bbox: BBox) -> BBox:
        """Axis-aligned pixel box enclosing the (possibly rotated) model box."""
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        pts = [self.to_px(*c) for c in corners]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin = max(0.0, min(xs))
        ymin = max(0.0, min(ys))
        xmax = min(float(self.img_w), max(xs))
        ymax = min(float(self.img_h), max(ys))
        return (xmin, ymin, xmax, ymax)

    def scalar_mm_to_px(self) -> float:
        """Approximate uniform mm->px factor (x and y are equal: aspect locked)."""
        return self.img_w / self.range_x
