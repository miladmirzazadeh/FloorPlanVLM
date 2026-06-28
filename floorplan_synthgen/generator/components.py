"""Component library: Wall, Door, Window, Column.

Each component draws itself into an ezdxf modelspace given an insertion
point, a wall direction and a scale, and records its own axis-aligned
bounding box in model space (mm). NO component draws any text, tag bubble
or annotation - openings are identified only in the exported label files.

Coordinate convention for openings
-----------------------------------
* ``center``      : centre of the opening, on the wall centreline (mm).
* ``direction``   : unit vector along the wall (numpy array).
* ``normal``      : unit normal = perp(direction).
* ``half_thick``  : wall half thickness (mm).
* ``clear``       : clear opening width (mm).

The wall *gap* (the break in the two wall faces, including the frame
reveal) is cut by the FloorPlan via a boolean subtraction, so the jamb
lines are produced by the wall body itself. The Door / Window only draws
the symbol that lives inside that gap (frame, leaf, swing arc, glazing,
sills, ...).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from . import geometry as g
from . import style

# ---- fixed real-world dimensions (mm) --------------------------------------
FRAME_W = 70            # door/window frame reveal each jamb
REBATE = 15             # door stop rebate step
LEAF_THICK = 45         # door leaf (shutter) thickness, real-world mm
SHUTTER_FULL = 45       # alias kept for back-compat
SHUTTER_MED = 45
HINGE_DIA = 25          # hinge knuckle diameter at 1:50
EXT_SILL_PROJ = 40      # exterior sill projection beyond outer face
INT_SILL_PROJ = 150     # interior sill board projection into room
DEFAULT_SWING = 90.0


def rough_opening(clear: float) -> float:
    """Wall gap width = clear opening + a frame reveal each side."""
    return clear + 2 * FRAME_W


# ---------------------------------------------------------------------------
class _Drawable:
    """Mixin: accumulates the points it draws and exposes a model bbox."""

    layer = style.LAY_MISC

    def __init__(self):
        self._pts: List[Sequence[float]] = []
        self.bbox_model: Optional[g.BBox] = None

    def _track(self, pts: Sequence[Sequence[float]]):
        for p in pts:
            self._pts.append((float(p[0]), float(p[1])))

    def _finalize_bbox(self):
        if self._pts:
            self.bbox_model = g.bbox_of_points(self._pts)

    def _track_gap(self, center, direction, normal, half_clear, half_thick):
        """Seed the bbox with the opening's gap in the wall (jambs + thickness),
        so even arc-less symbols (sliding doors, fixed windows) enclose the
        visible break in the wall."""
        c = g.to_np(center)
        d = g.to_np(direction)
        n = g.to_np(normal) * half_thick
        w = d * (half_clear + FRAME_W)
        self._track([c + w + n, c - w + n, c - w - n, c + w - n])

    # --- primitive helpers (all confine themselves to self.layer) ----------
    def _poly(self, msp, pts, *, close=False, layer=None, linetype=None):
        attribs = {"layer": layer or self.layer}
        if linetype:
            attribs["linetype"] = linetype
        msp.add_lwpolyline([(float(x), float(y)) for x, y in pts],
                           close=close, dxfattribs=attribs)
        self._track(pts)

    def _line(self, msp, a, b, *, layer=None, linetype=None):
        attribs = {"layer": layer or self.layer}
        if linetype:
            attribs["linetype"] = linetype
        msp.add_line((float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
                     dxfattribs=attribs)
        self._track([a, b])

    def _circle(self, msp, center, radius, *, layer=None):
        msp.add_circle((float(center[0]), float(center[1])), float(radius),
                       dxfattribs={"layer": layer or self.layer})
        self._track([(center[0] - radius, center[1] - radius),
                     (center[0] + radius, center[1] + radius)])

    def _solid_circle(self, msp, center, radius, *, layer=None):
        lay = layer or self.layer
        self._circle(msp, center, radius, layer=lay)
        hatch = msp.add_hatch(color=7, dxfattribs={"layer": lay})
        edge = hatch.paths.add_edge_path()
        edge.add_arc((float(center[0]), float(center[1])), float(radius), 0, 360)
        hatch.set_solid_fill()

    def _arc_poly(self, msp, center, radius, a0, a1, *, layer=None):
        pts = g.arc_points(center, radius, a0, a1)
        self._poly(msp, pts, close=False, layer=layer or self.layer)

    def _arc(self, msp, center, radius, a0, a1, *, layer=None, linetype=None):
        """A single true ARC entity (smooth, one object). Angles in degrees,
        drawn CCW from a0 to a1."""
        attribs = {"layer": layer or self.layer}
        if linetype:
            attribs["linetype"] = linetype
        msp.add_arc((float(center[0]), float(center[1])), float(radius),
                    float(a0), float(a1), dxfattribs=attribs)
        self._track(g.arc_points(center, radius, a0, a1, segments=24))

    def _dashed_arc(self, msp, center, radius, a0, a1, *, layer=None,
                    dash_mm=150.0, gap_ratio=0.45):
        """A hidden/dashed swing arc built from short true ARC segments, so the
        dashes are visible in every renderer regardless of $LTSCALE. Each dash
        is a real ARC entity (smooth), not a faceted polyline."""
        lay = layer or self.layer
        lo, hi = (a0, a1) if a1 >= a0 else (a1, a0)
        if hi - lo < 1e-6:
            return
        dash_deg = max(2.0, math.degrees(dash_mm / max(radius, 1.0)))
        step = dash_deg / max(1e-3, (1.0 - gap_ratio))   # dash + gap
        a = lo
        while a < hi - 1e-6:
            b = min(a + dash_deg, hi)
            self._arc(msp, center, radius, a, b, layer=lay)
            a += step


# ---------------------------------------------------------------------------
class Wall:
    """A straight wall segment defined by its centreline and thickness.

    The actual *clean* rendering (faces that terminate exactly at the face
    of a crossing wall, mitred corners, no overlapping lines) is produced by
    ``FloorPlan`` from the union of wall *bodies*; this class supplies the
    body rectangle and face lines that feed that process.
    """

    def __init__(self, wall_id: str, start, end, thickness: float,
                 wall_type: str = "INTERIOR"):
        self.id = wall_id
        self.start = g.to_np(start)
        self.end = g.to_np(end)
        self.thickness = float(thickness)
        self.wall_type = wall_type
        self.openings: List[object] = []

    # geometry ----------------------------------------------------------
    @property
    def direction(self) -> np.ndarray:
        return g.unit(self.end - self.start)

    @property
    def normal(self) -> np.ndarray:
        return g.perp(self.direction)

    @property
    def length(self) -> float:
        return g.length(self.end - self.start)

    @property
    def half_thick(self) -> float:
        return self.thickness / 2.0

    @property
    def is_full(self) -> bool:
        """Full-thickness (exterior) walls are continuous at junctions."""
        return self.thickness >= 250.0 or self.wall_type in ("EXTERIOR", "FIRE")

    @property
    def layer(self) -> str:
        return style.LAY_WALL_FULL if self.is_full else style.LAY_WALL_INTR

    def point_at(self, t: float) -> np.ndarray:
        """Point on centreline, t in 0..1 from start."""
        return self.start + (self.end - self.start) * t

    def rectangle(self) -> List[g.Point]:
        """Four corners of the wall body rectangle (CCW)."""
        n = self.normal * self.half_thick
        a = self.start + n
        b = self.end + n
        c = self.end - n
        d = self.start - n
        return [tuple(a), tuple(b), tuple(c), tuple(d)]

    def faces(self):
        """The two face line segments ((p1,p2), (p3,p4))."""
        n = self.normal * self.half_thick
        return ((tuple(self.start + n), tuple(self.end + n)),
                (tuple(self.start - n), tuple(self.end - n)))


# ---------------------------------------------------------------------------
class Door(_Drawable):
    layer = style.LAY_DOOR

    SUBTYPES = ("SINGLE", "DOUBLE", "SLIDING", "POCKET", "BIFOLD",
                "FRENCH", "GARAGE")

    def __init__(self, *, opening_id, center, direction, normal, half_thick,
                 clear_opening, subtype="SINGLE", swing="INWARD_LEFT",
                 swing_normal=None, max_swing_angle=DEFAULT_SWING, scale="1:100"):
        super().__init__()
        self.id = opening_id
        self.center = g.to_np(center)
        self.direction = g.unit(direction)
        self.normal = g.unit(normal)
        self.half_thick = float(half_thick)
        self.clear = float(clear_opening)
        self.subtype = subtype.upper()
        self.swing = (swing or "").upper() if swing else None
        # which side the leaf opens toward (defaults to +normal)
        self.swing_normal = g.unit(swing_normal) if swing_normal is not None else self.normal
        self.max_swing = float(max_swing_angle)
        self.detail = style.detail_for_scale(scale)
        # exact geometry, filled in during draw (mm)
        self.hinge_point = None
        self.leaf_end = None

    # convenience ------------------------------------------------------
    @property
    def _hinge_left(self) -> bool:
        return self.swing is None or self.swing.endswith("LEFT")

    def _clear_jambs(self):
        h = self.clear / 2.0
        j1 = self.center - h * self.direction   # "left" / start side
        j2 = self.center + h * self.direction   # "right" / end side
        return j1, j2

    def draw(self, msp):
        self._track_gap(self.center, self.direction, self.normal,
                        self.clear / 2.0, self.half_thick)
        method = {
            "SINGLE": self._draw_single,
            "DOUBLE": self._draw_double,
            "FRENCH": self._draw_double,   # twin glazed leaves, same plan symbol
            "SLIDING": self._draw_sliding,
            "POCKET": self._draw_pocket,
            "BIFOLD": self._draw_bifold,
            "GARAGE": self._draw_garage,
        }.get(self.subtype, self._draw_single)
        method(msp)
        self._finalize_bbox()
        return self.bbox_model

    # -- frame / leaf primitives ---------------------------------------
    def _frame_piece(self, msp, clear_jamb, outward_sign):
        """Frame reveal at a jamb. The reveal box spans from the clear jamb
        (inner frame edge) out to the rough jamb (the wall face), so the door
        frame physically connects to the wall on BOTH faces."""
        n = self.normal * self.half_thick
        rough_jamb = clear_jamb + outward_sign * FRAME_W * self.direction
        if self.detail == style.DETAIL_FULL:
            quad = [clear_jamb + n, rough_jamb + n, rough_jamb - n, clear_jamb - n]
            self._poly(msp, quad, close=True)
        else:
            # inner jamb line + two reveal sides closing to the wall face
            self._line(msp, clear_jamb + n, clear_jamb - n)
            self._line(msp, clear_jamb + n, rough_jamb + n)
            self._line(msp, clear_jamb - n, rough_jamb - n)

    def _both_frames(self, msp):
        """Frame reveal at BOTH jambs, connecting the opening to the wall on
        both faces. Every opening that sits in a cut wall gap needs this so it
        does not float (the wall gap is cut at the rough opening = clear + a
        FRAME_W reveal each side). Skipped only at the lightest scale, where the
        symbol is intentionally minimal."""
        if self.detail == style.DETAIL_MIN:
            return
        j1, j2 = self._clear_jambs()
        self._frame_piece(msp, j1, -1.0)
        self._frame_piece(msp, j2, +1.0)

    def _leaf(self, msp, hinge, leaf_dir, length, thickness, closed_dir=None):
        """Leaf in its open position. Single line for simplified scales, a slim
        rectangle at 1:50/1:100.

        The hinge pin sits on the *back edge* (hinge stile) of the leaf, not on
        its centreline, exactly as a real door pivots. So the leaf's pivot edge
        is rooted on the jamb line and its thickness is offset toward the
        closing direction (into the opening). This keeps the whole leaf on the
        room/opening side of the jamb -- it never straddles the jamb back into
        the frame reveal, which is what made the shutter merge with the frame."""
        end = hinge + leaf_dir * length
        if thickness <= 0 or closed_dir is None:
            self._line(msp, hinge, end)
            return np.array(end, dtype=float)
        off = g.unit(closed_dir) * thickness   # toward the latch / opening interior
        quad = [hinge, end, end + off, hinge + off]
        self._poly(msp, quad, close=True)
        return np.array(end, dtype=float)

    def _swing(self, msp, hinge, closed_dir, length):
        """Dashed (hidden-line) swing arc as smooth true ARC segments, plus the
        open-leaf direction it sweeps to."""
        sign = 1.0 if g.cross(closed_dir, self.swing_normal) > 0 else -1.0
        a0 = g.angle_deg(closed_dir)
        a1 = a0 + sign * self.max_swing
        leaf_dir = g.rotate_vec(closed_dir, sign * self.max_swing)
        self._dashed_arc(msp, hinge, length, a0, a1)
        return leaf_dir

    # -- subtypes -------------------------------------------------------
    def _draw_single(self, msp):
        j1, j2 = self._clear_jambs()
        hinge = j1 if self._hinge_left else j2
        latch = j2 if self._hinge_left else j1
        closed_dir = g.unit(latch - hinge)
        self.hinge_point = np.array(hinge, dtype=float)

        if self.detail == style.DETAIL_MIN:
            # 1:200 - lightweight but still legible: single-line leaf + dashed
            # swing. (Previously only the arc, so the door nearly vanished.)
            leaf_dir = self._swing(msp, hinge, closed_dir, self.clear)
            self.leaf_end = self._leaf(msp, hinge, leaf_dir, self.clear, 0.0)
            self._track([j1, j2])
            return

        # frame reveal each jamb (connects the opening to the wall)
        self._both_frames(msp)
        # dashed swing arc + leaf (shutter) shown open, with real thickness
        leaf_dir = self._swing(msp, hinge, closed_dir, self.clear)
        self.leaf_end = self._leaf(msp, hinge, leaf_dir, self.clear, LEAF_THICK,
                                   closed_dir)
        if self.detail == style.DETAIL_FULL:
            # small open hinge knuckle at the pivot
            self._circle(msp, hinge, HINGE_DIA / 2.0)

    def _draw_double(self, msp):
        j1, j2 = self._clear_jambs()
        half = self.clear / 2.0
        self.hinge_point = np.array(j1, dtype=float)
        self._both_frames(msp)
        # leaf from each jamb meeting at centre
        for hinge, closed_dir, sgn in ((j1, self.direction, +1), (j2, -self.direction, -1)):
            cdir = g.unit(closed_dir)
            leaf_dir = self._swing(msp, hinge, cdir, half)
            thick = 0.0 if self.detail == style.DETAIL_MIN else LEAF_THICK
            end = self._leaf(msp, hinge, leaf_dir, half, thick, cdir)
            if hinge is j1:
                self.leaf_end = end

    def _draw_sliding(self, msp):
        # frame reveal at BOTH jambs (consistent with swing doors and windows),
        # then two sashes living INSIDE the clear opening, offset to opposite
        # faces and overlapping at the centre. Nothing overruns into the wall.
        j1, j2 = self._clear_jambs()
        self.hinge_point = None
        n = self.normal
        off = self.half_thick * 0.45
        ov = self.clear * 0.08                # small overlap past the centre
        mid_a = self.center + self.direction * ov
        mid_b = self.center - self.direction * ov
        thick = 0.0 if self.detail == style.DETAIL_MIN else LEAF_THICK
        self._both_frames(msp)
        self._sash(msp, j1 + n * off, mid_a + n * off, thick)   # sash on +face
        self._sash(msp, mid_b - n * off, j2 - n * off, thick)   # sash on -face
        self._line(msp, j1, j2)               # track / rail across the opening
        self.leaf_end = np.array(j2, dtype=float)

    def _sash(self, msp, a, b, thick):
        """A sliding sash drawn between two already-offset endpoints."""
        a = g.to_np(a); b = g.to_np(b)
        if thick <= 0:
            self._line(msp, a, b)
            return
        ln = g.perp(g.unit(b - a)) * (thick / 2.0)
        self._poly(msp, [a + ln, b + ln, b - ln, a - ln], close=True)

    def _panel(self, msp, a, b, n_offset, thick):
        n = self.normal
        a = g.to_np(a) + n * n_offset
        b = g.to_np(b) + n * n_offset
        ln = n * thick / 2
        quad = [a + ln, b + ln, b - ln, a - ln]
        self._poly(msp, quad, close=True)

    def _draw_pocket(self, msp):
        # one panel + dashed pocket void inside the wall
        j1, j2 = self._clear_jambs()
        self.hinge_point = None
        self._both_frames(msp)
        panel_t = SHUTTER_FULL if self.detail == style.DETAIL_FULL else SHUTTER_MED
        self._panel(msp, j1, j2, 0.0, panel_t)
        # pocket extends past j1 into the wall, drawn dashed
        pocket_end = j1 - self.direction * self.clear
        n = self.normal * (self.half_thick * 0.8)
        quad = [j1 + n, pocket_end + n, pocket_end - n, j1 - n]
        self._poly(msp, quad, close=True, linetype="DASHED")
        self.leaf_end = np.array(pocket_end, dtype=float)

    def _draw_bifold(self, msp):
        # accordion of small panels zig-zagging across the opening
        j1, j2 = self._clear_jambs()
        self.hinge_point = np.array(j1, dtype=float)
        self._both_frames(msp)
        n_panels = 4
        seg = self.clear / n_panels
        amp = min(seg, self.half_thick * 0.9)
        pts = [tuple(j1)]
        for i in range(1, n_panels):
            base = j1 + self.direction * (seg * i)
            sign = 1 if i % 2 == 1 else -1
            pts.append(tuple(base + self.swing_normal * amp * sign))
        pts.append(tuple(j2))
        self._poly(msp, pts, close=False)
        self.leaf_end = np.array(j2, dtype=float)

    def _draw_garage(self, msp):
        # wide sectional overhead door: a thin panel across the gap with
        # horizontal section division lines. No swing arc.
        j1, j2 = self._clear_jambs()
        self.hinge_point = None
        self.leaf_end = None
        self._both_frames(msp)
        panel_t = SHUTTER_FULL if self.detail == style.DETAIL_FULL else SHUTTER_MED
        self._panel(msp, j1, j2, 0.0, panel_t)
        if self.detail != style.DETAIL_MIN:
            n = self.normal
            for k in range(1, 5):                      # sectional panel divisions
                p = j1 + self.direction * (self.clear * k / 5.0)
                self._line(msp, p + n * (panel_t / 2), p - n * (panel_t / 2))


# ---------------------------------------------------------------------------
class Window(_Drawable):
    layer = style.LAY_GLAZ

    SUBTYPES = ("CASEMENT", "SLIDING", "FIXED", "BAY", "AWNING",
                "LOUVRE", "CORNER", "CLERESTORY")

    def __init__(self, *, opening_id, center, direction, normal, half_thick,
                 clear_opening, subtype="FIXED", ext_sign=1.0, scale="1:100"):
        super().__init__()
        self.id = opening_id
        self.center = g.to_np(center)
        self.direction = g.unit(direction)
        self.normal = g.unit(normal)
        self.half_thick = float(half_thick)
        self.clear = float(clear_opening)
        self.subtype = subtype.upper()
        # exterior face is at +normal*ext_sign*half_thick
        self.ext_sign = 1.0 if ext_sign >= 0 else -1.0
        self.detail = style.detail_for_scale(scale)
        self.p1 = None  # glazing endpoints (mm)
        self.p2 = None

    def _jambs(self):
        h = self.clear / 2.0
        return self.center - h * self.direction, self.center + h * self.direction

    def draw(self, msp):
        self._track_gap(self.center, self.direction, self.normal,
                        self.clear / 2.0, self.half_thick)
        j1, j2 = self._jambs()                 # clear jambs = glazing extent
        self.p1 = np.array(j1, dtype=float)
        self.p2 = np.array(j2, dtype=float)
        n = self.normal
        ht = self.half_thick
        # rough jambs sit exactly on the wall gap edges, so every line below
        # runs the full width of the opening and MEETS the wall jamb faces
        # (the wall body draws the perpendicular jamb caps at rj1/rj2). No
        # floating gap between the window and the wall.
        rhw = self.clear / 2.0 + FRAME_W
        rj1 = self.center - self.direction * rhw   # wall gap edges (rough jambs)
        rj2 = self.center + self.direction * rhw

        # frame faces, flush with the wall faces, spanning the whole opening so
        # the window meets the wall on both long sides.
        self._line(msp, rj1 + n * ht, rj2 + n * ht)
        self._line(msp, rj1 - n * ht, rj2 - n * ht)

        # the window's OWN frame jamb on BOTH sides: a cap across the reveal at
        # each clear jamb. This closes the frame left and right itself, instead
        # of relying on the wall body to cap the opening.
        self._line(msp, j1 + n * ht, j1 - n * ht)
        self._line(msp, j2 + n * ht, j2 - n * ht)

        # glazing centre line between the two frame jambs
        self._line(msp, j1, j2)

        if self.detail != style.DETAIL_MIN:
            # inner frame / second glazing line: frame reads with depth at 1:100+
            self._line(msp, j1 + n * (ht * 0.45), j2 + n * (ht * 0.45))
            self._line(msp, j1 - n * (ht * 0.45), j2 - n * (ht * 0.45))

        # BAY / BOW project the building line outward -- a footprint feature,
        # so it is drawn at EVERY scale, not just 1:50.
        if self.subtype in ("BAY", "BOW"):
            self._draw_projection(msp, j1, j2)

        # opening-action indicators (casement arc, awning, sashes, louvre
        # slats, ...) read at 1:50 AND 1:100 so window types stay visually
        # distinct; omitted only at 1:200 where the symbol is minimal.
        if self.detail != style.DETAIL_MIN:
            self._draw_indicator(msp, j1, j2)

        if self.detail == style.DETAIL_FULL:
            ext = self.normal * self.ext_sign
            # exterior sill board sitting ON the exterior face (connected, not
            # floating), spanning the opening and projecting outward.
            self._poly(msp, [rj1 + ext * ht, rj2 + ext * ht,
                             rj2 + ext * (ht + EXT_SILL_PROJ),
                             rj1 + ext * (ht + EXT_SILL_PROJ)], close=True)

        self._finalize_bbox()
        return self.bbox_model

    def _draw_indicator(self, msp, j1, j2):
        """Opening-action indicator distinguishing the window subtype. Drawn at
        1:50 and 1:100. BAY/BOW carry no indicator (they draw a projection);
        FIXED has none by definition."""
        n = self.normal
        ht = self.half_thick
        ext = n * self.ext_sign
        if self.subtype == "CASEMENT":
            # opening indicator: diagonal from one jamb (interior) to mid exterior
            self._line(msp, j1 - ext * ht, self.center + ext * ht)
        elif self.subtype == "AWNING":
            # indicator on the (top) exterior edge from centre
            self._line(msp, self.center - ext * ht, j2 + ext * ht)
        elif self.subtype == "SLIDING":
            # two overlapping sashes
            mid = self.center
            self._line(msp, j1 + n * (ht * 0.3), mid + n * (ht * 0.3))
            self._line(msp, mid - n * (ht * 0.3), j2 - n * (ht * 0.3))
        elif self.subtype == "LOUVRE":
            # row of slat lines across the opening width
            n_slats = max(3, int(self.clear // 200))
            for k in range(1, n_slats):
                p = j1 + self.direction * (self.clear * k / n_slats)
                self._line(msp, p + n * ht, p - n * ht)
        elif self.subtype == "CLERESTORY":
            # high-level window: dashed band indicating it sits above eye level
            self._line(msp, j1, j2, linetype="DASHED")
            self._line(msp, j1 + n * (ht * 0.4), j2 + n * (ht * 0.4), linetype="DASHED")
        elif self.subtype == "CORNER":
            # corner-glazing leg: extra inner glazing line, no opening arc
            # (the L is formed by two CORNER windows meeting at the junction)
            self._line(msp, j1 + n * (ht * 0.25), j2 + n * (ht * 0.25))

    def _draw_projection(self, msp, j1, j2):
        """BAY / BOW: the glazing projects beyond the wall as a splayed footprint
        with a sill following it. A footprint feature, so the projection is
        drawn at every scale (the sill, a fine detail, only at 1:50)."""
        n = self.normal
        ht = self.half_thick
        ext = n * self.ext_sign
        depth = max(self.clear * 0.4, 600.0)        # how far it projects out
        inset = min(depth, self.clear * 0.3)        # splay; clamped so the front
        #                                             edge keeps positive width
        #                                             (never bow-ties on narrow widths)
        fl = j1 + ext * depth + self.direction * inset
        fr = j2 + ext * depth - self.direction * inset
        # splayed reveal from each wall jamb out to the projected front face
        self._poly(msp, [j1 + ext * ht, fl, fr, j2 + ext * ht], close=False)
        if self.detail == style.DETAIL_FULL:
            # sill board following the projected front
            self._poly(msp, [fl, fl + ext * EXT_SILL_PROJ,
                             fr + ext * EXT_SILL_PROJ, fr], close=False)


# ---------------------------------------------------------------------------
class Rooflight(_Drawable):
    """A roof-plane opening (skylight): a dashed square drawn inside a room,
    not cut into any wall. Classified as a window for detection."""

    layer = style.LAY_GLAZ
    SUBTYPES = ("FIXED",)

    def __init__(self, *, opening_id, x, y, size, subtype="FIXED", scale="1:100"):
        super().__init__()
        self.id = opening_id
        self.x = float(x)
        self.y = float(y)
        self.size = float(size)
        self.subtype = subtype.upper()
        self.clear = float(size)
        self.center = g.to_np((x, y))
        self.detail = style.detail_for_scale(scale)
        h = self.size / 2.0
        self.p1 = g.to_np((x - h, y - h))   # for the exporter (window record)
        self.p2 = g.to_np((x + h, y - h))
        self.direction = g.unit(self.p2 - self.p1)

    def draw(self, msp):
        h = self.size / 2.0
        sq = [(self.x - h, self.y - h), (self.x + h, self.y - h),
              (self.x + h, self.y + h), (self.x - h, self.y + h)]
        self._poly(msp, sq, close=True, linetype="DASHED")
        self._line(msp, sq[0], sq[2], linetype="DASHED")   # rooflight cross
        self._line(msp, sq[1], sq[3], linetype="DASHED")
        self._finalize_bbox()
        return self.bbox_model


# ---------------------------------------------------------------------------
class Column(_Drawable):
    layer = style.LAY_COLS

    def __init__(self, *, column_id, x, y, shape="SQUARE", size=300.0):
        super().__init__()
        self.id = column_id
        self.x = float(x)
        self.y = float(y)
        self.shape = shape.upper()
        self.size = float(size)

    def polygon(self):
        """Footprint as a list of points (square) - used to trim walls."""
        if self.shape == "ROUND":
            return g.arc_points((self.x, self.y), self.size / 2.0, 0, 360, segments=32)[:-1]
        h = self.size / 2.0
        return [(self.x - h, self.y - h), (self.x + h, self.y - h),
                (self.x + h, self.y + h), (self.x - h, self.y + h)]

    def draw(self, msp):
        if self.shape == "ROUND":
            r = self.size / 2.0
            self._solid_circle(msp, (self.x, self.y), r)
        else:
            poly = self.polygon()
            self._poly(msp, poly, close=True)
            hatch = msp.add_hatch(color=7, dxfattribs={"layer": self.layer})
            hatch.paths.add_polyline_path([(p[0], p[1]) for p in poly], is_closed=True)
            hatch.set_solid_fill()
            self._track(poly)
        self._finalize_bbox()
        return self.bbox_model
