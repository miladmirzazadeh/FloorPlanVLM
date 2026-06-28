"""Layout engine: FloorPlan.

Takes a scenario config dict, builds Wall/Door/Window/Column objects,
resolves wall junctions and opening gaps with exact polygon boolean
operations (shapely), draws optional clutter (the stuff a real plan is
full of and which the detector must learn to ignore), and writes a DXF.

Junction rule
-------------
Wall *bodies* (rectangles) are unioned per category. Full / exterior walls
are continuous; interior walls are subtracted by the exterior union so they
terminate exactly at the exterior face. Columns are subtracted from both, so
walls stop at the column face. Because the rendered lines are the boundary of
a single valid polygon, there are by construction no overlapping lines through
any T, L or X junction.

Opening rule
------------
Each opening subtracts a full-thickness gap (clear opening + a frame reveal
each side) from its wall body before the union, so the two wall faces break
and clean jamb lines appear. The Door/Window then draws its symbol into the gap.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import Matrix44
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import geometry as g
from . import style
from .components import Wall, Door, Window, Column, Rooflight, rough_opening

_EPS = 2.0  # mm, used to over-cut gaps so the wall is fully severed


def _polys(geom):
    if geom is None or geom.is_empty:
        return []
    t = geom.geom_type
    if t == "Polygon":
        return [geom]
    if t == "MultiPolygon":
        return list(geom.geoms)
    if t == "GeometryCollection":
        return [gm for gm in geom.geoms if gm.geom_type == "Polygon"]
    return []


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_on_segment(p, a, b, tol=1.0) -> bool:
    """True if p lies on segment a-b (not at its endpoints) within tol."""
    ap = (p[0] - a[0], p[1] - a[1])
    ab = (b[0] - a[0], b[1] - a[1])
    ab_len2 = ab[0] ** 2 + ab[1] ** 2
    if ab_len2 < 1e-9:
        return False
    t = (ap[0] * ab[0] + ap[1] * ab[1]) / ab_len2
    if t <= 0.02 or t >= 0.98:
        return False
    proj = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return _dist(p, proj) <= tol


class FloorPlan:
    def __init__(self, config: Dict):
        self.config = config
        self.plan_id = config.get("plan_id") or config.get("scenario") or "plan"
        self.scenario = config.get("scenario", self.plan_id)
        self.scale = config.get("scale", "1:100")
        self.rotation_deg = float(config.get("rotation_deg", 0.0))
        self.clutter = dict(config.get("clutter", {}))
        self.render_cfg = dict(config.get("render", {}))

        self.rng = random.Random(self._seed())

        self.walls: Dict[str, Wall] = {}
        self.walls_list: List[Wall] = []
        self.columns: List[Column] = []
        self.opening_components: List[object] = []   # Door/Window, in config order
        self._gaps_by_wall: Dict[str, list] = {}

        self.rot_origin = (0.0, 0.0)
        self.junctions_resolved = 0

        self._validate()
        self._build_walls()
        self._build_columns()
        self._place_openings()
        self._compute_rot_origin()

    # ---- setup --------------------------------------------------------
    def _seed(self) -> int:
        return abs(hash(("floorplan", self.plan_id))) & 0xFFFFFFFF

    def _validate(self):
        cfg = self.config
        if not cfg.get("walls"):
            raise ValueError(f"[{self.plan_id}] config has no walls")
        wall_ids = {w["id"] for w in cfg["walls"]}
        for op in cfg.get("openings", []):
            if op.get("rooflight"):
                # free roof-plane opening: no host wall, needs a centre + size
                if "x" not in op or "y" not in op:
                    raise ValueError(
                        f"[{self.plan_id}] rooflight {op.get('id')} needs x,y")
                continue
            if op.get("wall_id") not in wall_ids:
                raise ValueError(
                    f"[{self.plan_id}] opening {op.get('id')} references unknown wall {op.get('wall_id')!r}")
            pos = op.get("position_along_wall", 0.5)
            if not (0.0 <= pos <= 1.0):
                raise ValueError(
                    f"[{self.plan_id}] opening {op.get('id')} position_along_wall {pos} out of 0..1")
            t = op.get("type")
            if t not in ("door", "window"):
                raise ValueError(f"[{self.plan_id}] opening {op.get('id')} bad type {t!r}")
        try:
            style.scale_denominator(self.scale)
        except ValueError as e:
            raise ValueError(f"[{self.plan_id}] {e}")

    def _build_walls(self):
        for w in self.config["walls"]:
            wall = Wall(w["id"], (w["x1"], w["y1"]), (w["x2"], w["y2"]),
                        w.get("thickness", 150.0), w.get("wall_type", "INTERIOR"))
            self.walls[wall.id] = wall
            self.walls_list.append(wall)
            self._gaps_by_wall[wall.id] = []

    def _build_columns(self):
        for c in self.config.get("columns", []):
            self.columns.append(Column(column_id=c["id"], x=c["x"], y=c["y"],
                                       shape=c.get("shape", "SQUARE"),
                                       size=c.get("size_mm", c.get("size", 300.0))))

    def _plan_centroid(self):
        pts = []
        for w in self.walls_list:
            pts.append(tuple((w.start + w.end) / 2.0))
        arr = np.asarray(pts, dtype=float)
        return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    def _compute_rot_origin(self):
        xs, ys = [], []
        for w in self.walls_list:
            for p in w.rectangle():
                xs.append(p[0]); ys.append(p[1])
        self.rot_origin = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    # ---- openings -----------------------------------------------------
    def _place_openings(self):
        centroid = np.asarray(self._plan_centroid())
        for op in self.config.get("openings", []):
            if op.get("rooflight"):
                comp = Rooflight(opening_id=op["id"], x=op["x"], y=op["y"],
                                 size=float(op.get("size_mm", op.get("size", 1000.0))),
                                 subtype=op.get("subtype", "FIXED"), scale=self.scale)
                self.opening_components.append(comp)
                continue
            wall = self.walls[op["wall_id"]]
            t = float(op.get("position_along_wall", 0.5))
            center = wall.point_at(t)
            direction = wall.direction
            normal = wall.normal
            clear = float(op.get("clear_opening_mm", op.get("clear_opening", 800.0)))

            # interior normal points toward the plan centroid
            to_centroid = centroid - center
            interior_normal = normal if float(np.dot(to_centroid, normal)) >= 0 else -normal
            exterior_normal = -interior_normal
            ext_sign = 1.0 if float(np.dot(exterior_normal, normal)) >= 0 else -1.0

            # record gap to cut from the wall body
            self._gaps_by_wall[wall.id].append((center, direction, normal,
                                                rough_opening(clear), wall.half_thick))

            if op["type"] == "door":
                swing = op.get("swing", "INWARD_LEFT")
                inward = swing is None or "INWARD" in (swing or "")
                swing_normal = interior_normal if inward else exterior_normal
                comp = Door(opening_id=op["id"], center=center, direction=direction,
                            normal=normal, half_thick=wall.half_thick,
                            clear_opening=clear, subtype=op.get("subtype", "SINGLE"),
                            swing=swing, swing_normal=swing_normal,
                            max_swing_angle=float(op.get("max_swing_angle_deg",
                                                         op.get("max_swing_angle", 90.0))),
                            scale=self.scale)
            else:
                comp = Window(opening_id=op["id"], center=center, direction=direction,
                              normal=normal, half_thick=wall.half_thick,
                              clear_opening=clear, subtype=op.get("subtype", "FIXED"),
                              ext_sign=ext_sign, scale=self.scale)
            wall.openings.append(comp)
            self.opening_components.append(comp)

    # ---- wall network (shapely) ---------------------------------------
    def _end_connects(self, wall: Wall, pt) -> bool:
        """True if wall endpoint ``pt`` meets another wall (shared corner or a
        T into another wall's span). Such an end is extended so the union fills
        the outer corner square instead of leaving a re-entrant notch."""
        p = (float(pt[0]), float(pt[1]))
        for other in self.walls_list:
            if other is wall:
                continue
            tol = max(wall.thickness, other.thickness) * 0.75 + 1.0
            if (_dist(p, tuple(other.start)) < tol or
                    _dist(p, tuple(other.end)) < tol):
                return True
            if _point_on_segment(p, tuple(other.start), tuple(other.end), tol):
                return True
        return False

    def _wall_rectangle(self, wall: Wall) -> List:
        """Wall body rectangle, with connected ends extended by half_thick along
        the wall so L/T/X junctions miter cleanly (no unfilled corner)."""
        s, e = wall.start, wall.end
        d = wall.direction
        ext = d * wall.half_thick
        if self._end_connects(wall, wall.start):
            s = wall.start - ext
        if self._end_connects(wall, wall.end):
            e = wall.end + ext
        n = wall.normal * wall.half_thick
        return [tuple(s + n), tuple(e + n), tuple(e - n), tuple(s - n)]

    def _wall_body(self, wall: Wall) -> Polygon:
        body = Polygon(self._wall_rectangle(wall))
        for (center, direction, normal, width, ht) in self._gaps_by_wall[wall.id]:
            half_w = width / 2.0
            half_h = ht + _EPS
            d = g.to_np(direction)
            n = g.to_np(normal)
            corners = [center + d * half_w + n * half_h,
                       center - d * half_w + n * half_h,
                       center - d * half_w - n * half_h,
                       center + d * half_w - n * half_h]
            gap = Polygon([tuple(c) for c in corners])
            body = body.difference(gap)
        return body

    def _column_union(self):
        polys = [Polygon(c.polygon()) for c in self.columns]
        return unary_union(polys) if polys else None

    def _build_wall_network(self, msp):
        full = [self._wall_body(w) for w in self.walls_list if w.is_full]
        intr = [self._wall_body(w) for w in self.walls_list if not w.is_full]
        full_u = unary_union(full) if full else None
        intr_u = unary_union(intr) if intr else None
        cols = self._column_union()

        if full_u is not None and intr_u is not None:
            intr_u = intr_u.difference(full_u)  # interior terminates at exterior face
        if cols is not None:
            if full_u is not None:
                full_u = full_u.difference(cols)
            if intr_u is not None:
                intr_u = intr_u.difference(cols)

        ext_fill, intr_fill = self._wall_fills()
        if full_u is not None:
            self._draw_polys(msp, full_u, style.LAY_WALL_FULL, ext_fill)
        if intr_u is not None:
            self._draw_polys(msp, intr_u, style.LAY_WALL_INTR, intr_fill)

    def _wall_fills(self):
        """Pick (exterior_fill, interior_fill) for this plan. Explicit
        ``hatch_walls: False`` forces plain outlines; otherwise a material fill
        is chosen per plan from the scale-appropriate palette, seeded by plan id
        so plans vary and exterior reads differently from interior."""
        if self.clutter.get("hatch_walls") is False:
            return style.PLAIN_FILL, style.PLAIN_FILL
        ext_pal, intr_pal = style.wall_fill_palette(style.detail_for_scale(self.scale))
        ext = ext_pal[self.rng.randrange(len(ext_pal))]
        intr = intr_pal[self.rng.randrange(len(intr_pal))]
        return ext, intr

    def _draw_polys(self, msp, geom, layer, fill):
        kind = fill[0]
        for poly in _polys(geom):
            msp.add_lwpolyline(list(poly.exterior.coords), close=True,
                               dxfattribs={"layer": layer})
            for ring in poly.interiors:
                msp.add_lwpolyline(list(ring.coords), close=True,
                                   dxfattribs={"layer": layer})
            if kind == "plain":
                continue
            if kind == "solid":
                h = msp.add_hatch(color=8, dxfattribs={"layer": style.LAY_WALL_PATT})
                h.paths.add_polyline_path(list(poly.exterior.coords),
                                          is_closed=True, flags=1)
                for ring in poly.interiors:
                    h.paths.add_polyline_path(list(ring.coords), is_closed=True,
                                              flags=0)
                h.set_solid_fill()
            else:  # "lines": clipped hatch segments (renderer-independent)
                _, angles, spacing = fill
                self._draw_hatch_lines(msp, poly, angles, spacing)

    def _draw_hatch_lines(self, msp, poly, angles, spacing):
        """Fill ``poly`` with parallel hatch lines at each angle, generated as
        real LINE entities clipped to the polygon (holes excluded). Robust to
        any backend, unlike pattern-fill hatches."""
        from shapely.geometry import LineString, MultiLineString
        minx, miny, maxx, maxy = poly.bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        diag = math.hypot(maxx - minx, maxy - miny) or 1.0
        n = int(diag / max(spacing, 1.0)) + 2
        segs = []
        for ang in angles:
            a = math.radians(ang)
            dx, dy = math.cos(a), math.sin(a)      # line direction
            px, py = -dy, dx                       # offset (perpendicular) dir
            for i in range(-n, n + 1):
                off = i * spacing
                mx, my = cx + px * off, cy + py * off
                segs.append([(mx - dx * diag, my - dy * diag),
                             (mx + dx * diag, my + dy * diag)])
        if not segs:
            return
        clipped = poly.intersection(MultiLineString(segs))
        self._emit_clip(msp, clipped)

    def _emit_clip(self, msp, geom):
        if geom is None or geom.is_empty:
            return
        gt = geom.geom_type
        if gt == "LineString":
            cs = list(geom.coords)
            if len(cs) >= 2:
                msp.add_line(cs[0], cs[-1],
                             dxfattribs={"layer": style.LAY_WALL_PATT})
        elif gt in ("MultiLineString", "GeometryCollection"):
            for g2 in geom.geoms:
                self._emit_clip(msp, g2)

    # ---- junction accounting ------------------------------------------
    def _count_junctions(self) -> int:
        n = len(self.walls_list)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if self._walls_connect(self.walls_list[i], self.walls_list[j]):
                    count += 1
        return count

    @staticmethod
    def _walls_connect(a: Wall, b: Wall) -> bool:
        tol = max(a.thickness, b.thickness) * 0.75 + 1.0
        pa = [tuple(a.start), tuple(a.end)]
        pb = [tuple(b.start), tuple(b.end)]
        for x in pa:
            for y in pb:
                if _dist(x, y) < tol:
                    return True
        for x in pa:
            if _point_on_segment(x, tuple(b.start), tuple(b.end), tol):
                return True
        for y in pb:
            if _point_on_segment(y, tuple(a.start), tuple(a.end), tol):
                return True
        return False

    # ---- clutter ------------------------------------------------------
    def _model_bounds(self):
        xs, ys = [], []
        for w in self.walls_list:
            for p in w.rectangle():
                xs.append(p[0]); ys.append(p[1])
        return (min(xs), min(ys), max(xs), max(ys))

    def _text(self, msp, s, pos, height, layer, align=TextEntityAlignment.MIDDLE_CENTER,
              rotation=0.0):
        t = msp.add_text(s, height=height, dxfattribs={"layer": layer, "rotation": rotation})
        t.set_placement((float(pos[0]), float(pos[1])), align=align)

    def _draw_clutter(self, msp):
        denom = style.scale_denominator(self.scale)
        th = 2.5 * denom  # text height ~2.5mm on paper
        if self.clutter.get("room_labels", False):
            self._draw_room_labels(msp, th)
        if self.clutter.get("title_block", False):
            self._draw_title_block(msp, th)
        if self.clutter.get("dimensions", False):
            self._draw_dimensions(msp, th)
        if self.clutter.get("furniture", False):
            self._draw_furniture(msp)
        if self.clutter.get("grid", False):
            self._draw_grid(msp, th)
        n_noise = int(self.clutter.get("noise_lines", 0))
        if n_noise > 0:
            self._draw_noise(msp, n_noise)

    @staticmethod
    def _room_geom(r):
        """(cx, cy, bbox_w, bbox_h, is_rect) for rectangle or polygon rooms."""
        poly = r.get("polygon")
        if poly:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            return cx, cy, max(xs) - min(xs), max(ys) - min(ys), False
        rw = r.get("width", r.get("w", 0))
        rh = r.get("height", r.get("h", 0))
        return r["x"] + rw / 2.0, r["y"] + rh / 2.0, rw, rh, True

    def _draw_room_labels(self, msp, th):
        for r in self.config.get("rooms", []):
            cx, cy, rw, rh, is_rect = self._room_geom(r)
            name = r.get("name", "ROOM").upper()
            # shrink the label so it comfortably fits the room width
            fit = 0.70 * rw / (0.85 * max(len(name), 3))
            lh = max(min(th, fit), th * 0.4)
            show_dim = is_rect and rw > 2500 and rh > 2200
            y_off = lh * 0.7 if show_dim else 0.0
            self._text(msp, name, (cx, cy + y_off), lh, style.LAY_ANNO_TEXT)
            if show_dim:
                self._text(msp, f"{rw/1000.0:.1f} x {rh/1000.0:.1f}",
                           (cx, cy - lh * 0.7), lh * 0.8, style.LAY_ANNO_TEXT)

    def _draw_title_block(self, msp, th):
        x0, y0, x1, y1 = self._model_bounds()
        m = (x1 - x0 + y1 - y0) * 0.06 + th * 2
        bx0, by0, bx1, by1 = x0 - m, y0 - m, x1 + m, y1 + m
        # outer border
        msp.add_lwpolyline([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)],
                           close=True, dxfattribs={"layer": style.LAY_ANNO_TTLB})
        # title block panel bottom-right
        pw = (bx1 - bx0) * 0.28
        ph = (by1 - by0) * 0.12
        px0, py0 = bx1 - pw, by0
        msp.add_lwpolyline([(px0, py0), (bx1, py0), (bx1, py0 + ph), (px0, py0 + ph)],
                           close=True, dxfattribs={"layer": style.LAY_ANNO_TTLB})
        self._text(msp, "PLAN", (px0 + pw / 2, py0 + ph * 0.66), th * 1.2,
                   style.LAY_ANNO_TTLB)
        self._text(msp, f"SCALE {self.scale}", (px0 + pw / 2, py0 + ph * 0.33),
                   th * 0.8, style.LAY_ANNO_TTLB)

    def _draw_dimensions(self, msp, th):
        x0, y0, x1, y1 = self._model_bounds()
        off = (y1 - y0) * 0.08 + th * 3
        tick = th * 0.6
        # one overall dimension below, one to the left
        self._dim_line(msp, (x0, y0 - off), (x1, y0 - off), th, tick, horizontal=True,
                       ext_from=y0)
        self._dim_line(msp, (x0 - off, y0), (x0 - off, y1), th, tick, horizontal=False,
                       ext_from=x0)

    def _dim_line(self, msp, a, b, th, tick, horizontal, ext_from):
        lay = style.LAY_ANNO_DIMS
        msp.add_line(a, b, dxfattribs={"layer": lay})
        # extension lines + 45-degree ticks at the ends
        for p in (a, b):
            if horizontal:
                msp.add_line((p[0], ext_from), (p[0], p[1]), dxfattribs={"layer": lay})
            else:
                msp.add_line((ext_from, p[1]), (p[0], p[1]), dxfattribs={"layer": lay})
            msp.add_line((p[0] - tick, p[1] - tick), (p[0] + tick, p[1] + tick),
                         dxfattribs={"layer": lay})
        if horizontal:
            dist = abs(b[0] - a[0])
            mid = ((a[0] + b[0]) / 2, a[1] + th * 0.6)
            rot = 0.0
        else:
            dist = abs(b[1] - a[1])
            mid = (a[0] - th * 0.6, (a[1] + b[1]) / 2)
            rot = 90.0
        self._text(msp, f"{dist/1000.0:.2f}", mid, th, lay, rotation=rot)

    def _draw_furniture(self, msp):
        for r in self.config.get("rooms", []):
            cx, cy, w, h, _is_rect = self._room_geom(r)
            if w < 1500 or h < 1500:
                continue
            fw, fh = w * 0.4, h * 0.3
            msp.add_lwpolyline([(cx - fw / 2, cy - fh / 2), (cx + fw / 2, cy - fh / 2),
                                (cx + fw / 2, cy + fh / 2), (cx - fw / 2, cy + fh / 2)],
                               close=True, dxfattribs={"layer": style.LAY_FURN})
            msp.add_circle((cx + w * 0.25, cy - h * 0.25), min(w, h) * 0.12,
                           dxfattribs={"layer": style.LAY_FURN})

    def _draw_grid(self, msp, th):
        x0, y0, x1, y1 = self._model_bounds()
        spacing = 3000.0
        ext = th * 2
        r = th * 1.2
        i = 0
        x = x0
        while x <= x1 + 1:
            msp.add_line((x, y0 - ext), (x, y1 + ext), dxfattribs={"layer": style.LAY_GRID})
            msp.add_circle((x, y1 + ext + r), r, dxfattribs={"layer": style.LAY_GRID})
            self._text(msp, chr(ord("A") + i), (x, y1 + ext + r), th * 0.9, style.LAY_GRID)
            x += spacing
            i += 1
        j = 1
        y = y0
        while y <= y1 + 1:
            msp.add_line((x0 - ext, y), (x1 + ext, y), dxfattribs={"layer": style.LAY_GRID})
            msp.add_circle((x0 - ext - r, y), r, dxfattribs={"layer": style.LAY_GRID})
            self._text(msp, str(j), (x0 - ext - r, y), th * 0.9, style.LAY_GRID)
            y += spacing
            j += 1

    def _draw_noise(self, msp, n):
        x0, y0, x1, y1 = self._model_bounds()
        span = max(x1 - x0, y1 - y0)
        for _ in range(n):
            px = self.rng.uniform(x0, x1)
            py = self.rng.uniform(y0, y1)
            ang = self.rng.uniform(0, 2 * math.pi)
            ln = self.rng.uniform(span * 0.02, span * 0.12)
            ex = px + ln * math.cos(ang)
            ey = py + ln * math.sin(ang)
            msp.add_line((px, py), (ex, ey), dxfattribs={"layer": style.LAY_MISC})

    # ---- rotation -----------------------------------------------------
    def _apply_rotation(self, msp):
        if abs(self.rotation_deg) < 1e-9:
            return
        cx, cy = self.rot_origin
        m = (Matrix44.translate(-cx, -cy, 0)
             @ Matrix44.z_rotate(math.radians(self.rotation_deg))
             @ Matrix44.translate(cx, cy, 0))
        for e in list(msp):
            try:
                e.transform(m)
            except Exception:
                pass

    # ---- public -------------------------------------------------------
    def _insert_as_block(self, doc, msp, comp, prefix: str) -> str:
        """Author a component into its own block definition and drop a single
        block reference into modelspace, so the whole door / window / column
        selects as ONE object (the DXF equivalent of an AutoCAD block).
        Geometry is in world coordinates and the reference is inserted at the
        origin, so the recorded bbox and the exporter are unchanged."""
        name = base = f"{prefix}_{comp.id}"
        i = 1
        while name in doc.blocks:
            name = f"{base}_{i}"
            i += 1
        blk = doc.blocks.new(name=name)
        comp.draw(blk)
        msp.add_blockref(name, (0.0, 0.0, 0.0),
                         dxfattribs={"layer": getattr(comp, "layer", style.LAY_MISC)})
        return name

    def write_dxf(self, path: str):
        doc = ezdxf.new("R2010", setup=True)
        style.register_layers(doc)
        msp = doc.modelspace()

        self._build_wall_network(msp)
        for comp in self.opening_components:
            self._insert_as_block(doc, msp, comp, "OPN")
        for col in self.columns:
            self._insert_as_block(doc, msp, col, "COL")
        self._draw_clutter(msp)

        self.junctions_resolved = self._count_junctions()

        # scale dashed linetypes (swing arcs are self-dashed, but the grid /
        # pocket / clerestory rely on the linetype) so they read at plan size.
        x0, y0, x1, y1 = self._model_bounds()
        doc.header["$LTSCALE"] = max(1.0, max(x1 - x0, y1 - y0) / 50.0)

        self._apply_rotation(msp)
        doc.set_modelspace_vport(height=max(1.0, self._model_bounds()[3]))
        doc.saveas(path)
        return doc

    def opening_records(self):
        """List of (component, type, subtype) for the exporter, in order."""
        out = []
        for comp in self.opening_components:
            kind = "door" if isinstance(comp, Door) else "window"
            out.append((comp, kind))
        return out
