#!/usr/bin/env python3
"""config_generator.py - procedural floor-plan *config* generator.

Produces N unique, architecturally valid, maximally diverse floor-plan
scenarios as JSON, in the SAME absolute-geometry schema your existing
``generator/`` package already consumes (documented in
``scenarios/SCHEMA.md`` and enforced by ``validate_scenarios.py``).  Each
scenario is additionally enriched with metadata fields (wall material,
building type, complexity, region, clutter level, ...) that the current
renderer ignores but you can use later.

This script produces ONLY JSON configs.  It does NOT draw, render or write
DXF.  The configs feed your existing pipeline, e.g.::

    python config_generator.py --count 10000 --output ./configs --seed 42
    python -m generator.generate --scenarios ./configs/render_batches   # render

Coordinate system (matches the engine): integer millimetres, origin (0,0),
+x right / +y up, angles in degrees CCW from +x.  Geometry is absolute --
walls carry centrelines + band polygons (+ true ``arc`` for curves), openings
carry explicit jamb endpoints ``p1``/``p2``/``center`` and absolute ``symbol``
drawing primitives.  There is no host-wall linkage.

Design notes
------------
* Walls are derived from the room tiling by an exact interval method, so
  junctions are clean and every room is fully enclosed.
* Curved walls are TRUE circular arcs (``arc:{center,radius,a0,a1}``), never
  polyline approximations; the band polygon is a concentric outer+inner arc.
* "Chaotic" plans are visually messy (clutter / rotation / degradation /
  extreme aspect) but never geometrically broken: every emitted plan passes
  the internal validator, which mirrors ``validate_scenarios.py`` plus the
  hard-reject rules in the brief.
* Out-of-enum opening subtypes from the brief (REVOLVING, TILT_TURN,
  SHOPFRONT, FOLDING_PARTITION, ROOF_LIGHT) are canonicalised to a valid
  enum value, with the original kept in ``subtype_detail``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "synthdata-absolute-geometry/1.1"

# ===========================================================================
# SECTION 1 -- vector / geometry helpers  (conventions from generate_scenarios)
# ===========================================================================

Pt = Tuple[float, float]


def _sub(a: Pt, b: Pt) -> Pt: return (a[0] - b[0], a[1] - b[1])
def _add(a: Pt, b: Pt) -> Pt: return (a[0] + b[0], a[1] + b[1])
def _mul(a: Pt, s: float) -> Pt: return (a[0] * s, a[1] * s)
def _len(a: Pt) -> float: return math.hypot(a[0], a[1])


def _unit(a: Pt) -> Pt:
    L = _len(a)
    return (a[0] / L, a[1] / L) if L else (0.0, 0.0)


def _norm(d: Pt) -> Pt:
    """Left normal of a direction vector (90 deg CCW)."""
    return (-d[1], d[0])


def _i(p: Pt) -> List[int]:
    """Round a point to integer mm."""
    return [int(round(p[0])), int(round(p[1]))]


def _angle(d: Pt) -> float:
    return math.degrees(math.atan2(d[1], d[0]))


def _pt_at_angle(center: Pt, radius: float, deg: float) -> Pt:
    r = math.radians(deg)
    return (center[0] + radius * math.cos(r), center[1] + radius * math.sin(r))


def _arc_poly(center: Pt, radius: float, a0: float, a1: float,
              segs: int = 32) -> List[List[int]]:
    pts = []
    for k in range(segs + 1):
        a = a0 + (a1 - a0) * k / segs
        pts.append(_i(_pt_at_angle(center, radius, a)))
    return pts


def _rotate_pt(p: Pt, deg: float, about: Pt = (0, 0)) -> Pt:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y = p[0] - about[0], p[1] - about[1]
    return (x * c - y * s + about[0], x * s + y * c + about[1])


def _poly_area(poly: List[List[float]]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def _point_in_poly(pt: Pt, poly: List[List[float]]) -> bool:
    """Ray-cast point-in-polygon (polygon = list of [x,y])."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# --- interval algebra (used to turn a room tiling into clean walls) --------

def _iv_union(ivs: List[Tuple[float, float]]) -> List[List[float]]:
    if not ivs:
        return []
    s = sorted((min(a, b), max(a, b)) for a, b in ivs)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _iv_intersect(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    out = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > 1e-6:
                out.append([lo, hi])
    return _iv_union(out)


def _iv_subtract(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    out = []
    for a0, a1 in A:
        cuts = [(max(a0, b0), min(a1, b1)) for b0, b1 in B
                if min(a1, b1) - max(a0, b0) > 1e-6]
        cuts.sort()
        cur = a0
        for c0, c1 in cuts:
            if c0 - cur > 1e-6:
                out.append([cur, c0])
            cur = max(cur, c1)
        if a1 - cur > 1e-6:
            out.append([cur, a1])
    return out


# ===========================================================================
# SECTION 2 -- weighted sampling helpers
# ===========================================================================

def wchoice(rng: random.Random, table: Dict) -> object:
    """Pick a key from {key: weight}."""
    keys = list(table.keys())
    weights = [table[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def snap(v: float, grid: int = 50) -> int:
    return int(round(v / grid) * grid)


# ===========================================================================
# SECTION 3 -- distributions / catalogues  (from the brief)
# ===========================================================================

BUILDING_TYPES = {
    "residential_apartment": 25, "residential_house": 15,
    "office_open_plan": 10, "office_cellular": 10, "hotel_floor": 8,
    "hospital_ward": 5, "school_classroom_block": 5, "retail_unit": 5,
    "restaurant_cafe": 5, "mixed_use_ground_floor": 5,
    "industrial_unit": 4, "sports_facility": 3,
}

# building type -> ((w_min,w_max),(d_min,d_max)) in mm
BUILDING_DIMS = {
    "residential_apartment": ((6000, 18000), (7000, 16000)),
    "residential_house": ((8000, 16000), (9000, 18000)),
    "office_open_plan": ((15000, 45000), (12000, 35000)),
    "office_cellular": ((12000, 30000), (10000, 25000)),
    "hotel_floor": ((20000, 60000), (10000, 20000)),
    "hospital_ward": ((15000, 40000), (8000, 16000)),
    "school_classroom_block": ((15000, 35000), (8000, 14000)),
    "retail_unit": ((8000, 25000), (10000, 30000)),
    "restaurant_cafe": ((8000, 20000), (8000, 16000)),
    "mixed_use_ground_floor": ((10000, 30000), (10000, 25000)),
    "industrial_unit": ((20000, 60000), (15000, 40000)),
    "sports_facility": ((20000, 50000), (15000, 40000)),
}

# building type -> (room_min, room_max)
ROOM_COUNTS = {
    "residential_apartment": (3, 9), "residential_house": (5, 13),
    "office_open_plan": (3, 8), "office_cellular": (6, 22),
    "hotel_floor": (8, 28), "hospital_ward": (6, 18),
    "school_classroom_block": (4, 14), "retail_unit": (2, 8),
    "restaurant_cafe": (3, 10), "mixed_use_ground_floor": (3, 9),
    "industrial_unit": (2, 6), "sports_facility": (3, 8),
}

ROOM_TYPES = {
    "residential_apartment": ["living", "kitchen", "dining", "bedroom",
        "master_bedroom", "bathroom", "WC", "hallway", "corridor",
        "utility", "study", "wardrobe", "balcony", "storage"],
    "residential_house": ["living", "kitchen", "dining", "bedroom",
        "master_bedroom", "bathroom", "WC", "hallway", "corridor",
        "utility", "study", "wardrobe", "garage", "storage", "playroom"],
    "office_open_plan": ["open_plan", "meeting_room", "reception",
        "corridor", "WC", "server_room", "breakout", "kitchen_staff"],
    "office_cellular": ["private_office", "meeting_room", "reception",
        "corridor", "WC", "server_room", "breakout", "kitchen_staff",
        "open_plan"],
    "hotel_floor": ["guest_room", "corridor", "lift_lobby", "stair",
        "service_room", "linen_cupboard", "bathroom_en_suite"],
    "hospital_ward": ["ward_bay", "nurse_station", "corridor",
        "single_room", "sluice", "store", "WC", "shower"],
    "school_classroom_block": ["classroom", "corridor", "WC",
        "staff_room", "store", "reception", "office"],
    "retail_unit": ["sales_floor", "stockroom", "WC", "office",
        "changing_room", "loading_bay"],
    "restaurant_cafe": ["dining_area", "kitchen", "WC", "bar",
        "store", "office", "cold_room"],
    "mixed_use_ground_floor": ["retail_unit", "lobby", "corridor",
        "WC", "plant_room", "bin_store", "cycle_store"],
    "industrial_unit": ["workshop", "warehouse", "office", "WC",
        "plant_room", "loading_bay"],
    "sports_facility": ["hall", "changing_room", "WC", "shower",
        "store", "reception", "office", "gym"],
}

# rooms that prefer to be small (carved as closets / wet rooms)
SMALL_ROOMS = {"WC", "wardrobe", "storage", "store", "linen_cupboard",
               "sluice", "server_room", "bathroom", "shower", "utility",
               "cold_room", "plant_room", "bin_store", "cycle_store"}
WET_ROOMS = {"bathroom", "WC", "bathroom_en_suite", "shower", "kitchen",
             "kitchen_staff", "sluice", "utility", "wet_room", "cold_room"}

FOOTPRINT_SHAPES = {
    "rectangle": 30, "L_shape": 18, "T_shape": 8, "U_shape": 7,
    "irregular_polygon": 12, "rectangle_with_bay": 8,
    "rectangle_with_curved_end": 7, "fully_curved_facade": 5,
    "organic_multi_arc": 5,
}

# ---- wall materials: name -> (t_min, t_max, hatch, class) -----------------
WALL_MATERIALS = {
    "CAVITY_BRICK": (270, 340, ["ANSI31", "DOTS", "ANSI31"], "exterior"),
    "SOLID_BRICK": (215, 330, "ANSI31", "exterior"),
    "REINFORCED_CONCRETE": (200, 400, "AR-CONC", "exterior"),
    "CONCRETE_BLOCK": (100, 215, "AR-BRSTD", "interior"),
    "TIMBER_STUD": (89, 140, "ANSI37", "interior"),
    "METAL_STUD": (70, 150, "ANSI31", "interior"),
    "GLASS_PARTITION": (50, 100, None, "interior"),
    "WET_WALL_BLOCK": (150, 215, "AR-BRSTD", "wet"),
    "RAMMED_EARTH": (350, 600, "AR-SAND", "exterior"),
    "ICF_INSULATED": (250, 350, ["ANSI37", "AR-CONC", "ANSI37"], "exterior"),
    "STRUCTURAL_RC_CORE": (250, 500, "AR-CONC", "core"),
}
EXTERIOR_MATERIALS = {"CAVITY_BRICK": 30, "SOLID_BRICK": 15,
                      "REINFORCED_CONCRETE": 25, "TIMBER_STUD": 12,
                      "METAL_STUD": 8, "RAMMED_EARTH": 5, "ICF_INSULATED": 5}
INTERIOR_MATERIALS = {"CONCRETE_BLOCK": 35, "TIMBER_STUD": 35, "METAL_STUD": 30}
OFFICE_PARTITIONS = {"GLASS_PARTITION": 40, "METAL_STUD": 60}

# map material class -> engine wall "type"
CLASS_TO_TYPE = {"exterior": "exterior", "interior": "interior",
                 "wet": "interior", "core": "core", "party": "party",
                 "structural": "structural"}

# ---- opening subtype distributions (detail -> canonical enum) -------------
DOOR_SUBTYPES = {  # detail subtype: weight
    "SINGLE_HINGED": 48, "DOUBLE_HINGED": 10, "SLIDING": 14, "POCKET": 11,
    "BIFOLD": 8, "FRENCH": 5, "GARAGE": 2, "REVOLVING": 1,
    "FOLDING_PARTITION": 1,
}
DOOR_CANON = {  # detail -> valid engine enum
    "SINGLE_HINGED": "SINGLE_HINGED", "DOUBLE_HINGED": "DOUBLE_HINGED",
    "SLIDING": "SLIDING", "POCKET": "POCKET", "BIFOLD": "BIFOLD",
    "FRENCH": "FRENCH", "GARAGE": "GARAGE",
    "REVOLVING": "DOUBLE_HINGED", "FOLDING_PARTITION": "BIFOLD",
}
WINDOW_SUBTYPES = {
    "CASEMENT": 28, "FIXED": 24, "SLIDING": 14, "BAY": 9, "AWNING": 8,
    "TILT_TURN": 5, "CLERESTORY": 5, "LOUVRE": 3, "CORNER": 2,
    "ROOF_LIGHT": 1, "SHOPFRONT": 1,
}
WINDOW_CANON = {
    "CASEMENT": "CASEMENT", "FIXED": "FIXED", "SLIDING": "SLIDING",
    "BAY": "BAY", "AWNING": "AWNING", "TILT_TURN": "CASEMENT",
    "CLERESTORY": "CLERESTORY", "LOUVRE": "LOUVRE", "CORNER": "CORNER",
    "ROOF_LIGHT": "FIXED", "SHOPFRONT": "FIXED",
}

# ---- size pools (real standard sizes) -------------------------------------
DOOR_SIZES = {
    "main_entry": [900, 1000, 1050, 1200],
    "interior_metric": [700, 750, 800, 826, 900],
    "interior_imperial": [610, 686, 711, 762, 813, 838],
    "bathroom": [600, 650, 686, 700, 750],
    "accessible": [900, 950, 1000],
    "double_leaf": [600, 700, 750, 800, 900],     # per leaf
    "sliding_patio": [1600, 1800, 2100, 2400, 3000, 3600],
    "folding_partition": [2400, 3000, 3600, 4800, 6000],
    "garage": [2400, 2700, 3000, 3600, 4800, 5400],
}
WINDOW_SIZE_CATS = {"narrow": 15, "standard": 40, "wide": 30, "full_width": 15}
WINDOW_SIZE_RANGE = {"narrow": (300, 700), "standard": (800, 1400),
                     "wide": (1500, 2200), "full_width": (2300, 4500)}

STD_ANGLES = [22.5, 30, 45, 60, 67.5, 112.5, 120, 135, 150]

SCALES = {"1:20": 5, "1:50": 25, "1:100": 40, "1:200": 25, "1:500": 5}
STANDARDS = {"AIA": 40, "ISO": 25, "BS": 15, "GB": 12, "DIN": 8}
CLUTTER_LEVELS = {"none": 15, "light": 30, "medium": 30, "heavy": 25}
LINEWEIGHT_STYLES = {"standard_layered": 50, "light_all": 20, "heavy_all": 15,
                     "hairline_all": 8, "mixed_random": 7}
ROTATION_BUCKET = {"zero": 50, "small": 20, "medium": 20, "large": 10}
REGIONS = {"us": 40, "eu": 35, "uk": 15, "other": 10}

# column presence probability by building type
COLUMN_PROB = {
    "residential_apartment": 0.20, "residential_house": 0.10,
    "office_open_plan": 0.80, "office_cellular": 0.80, "hotel_floor": 0.60,
    "hospital_ward": 0.70, "school_classroom_block": 0.55,
    "retail_unit": 0.85, "restaurant_cafe": 0.45,
    "mixed_use_ground_floor": 0.70, "industrial_unit": 0.90,
    "sports_facility": 0.90,
}

PLACEMENTS = {"centered": 25, "offset_left": 25, "offset_right": 25,
              "near_corner_left": 12, "near_corner_right": 13}
OPENINGS_PER_WALL = {1: 60, 2: 28, 3: 10, 4: 2}

# global geometry constants
MIN_WALL = 300          # mm, walls shorter than this are rejected
JAMB_CLEAR = 150        # mm, min distance jamb -> wall endpoint
GAP_BETWEEN = 220       # mm, min gap between two openings on a wall
GRID = 50               # mm snapping grid


# ===========================================================================
# SECTION 4 -- dataclasses (engine schema)
# ===========================================================================

@dataclass
class Wall:
    id: str
    type: str
    thickness_mm: int
    length_mm: int
    centerline: list
    polygon: list
    arc: Optional[dict] = None
    material: str = ""
    hatch: object = None
    angle_class: str = "orthogonal"


@dataclass
class Room:
    id: str
    name: str
    shape: str
    polygon: list
    room_type: str = ""


@dataclass
class Opening:
    id: str
    category: str
    subtype: str
    width_mm: int
    p1: list
    p2: list
    center: list
    angle_deg: float
    height_mm: Optional[int] = None
    hinge: str = "none"
    swing: str = "none"
    panels: int = 1
    sill_mm: Optional[int] = None
    head_mm: Optional[int] = None
    plane: str = "wall"
    group: Optional[str] = None
    symbol: dict = field(default_factory=lambda: {
        "lines": [], "arcs": [], "polylines": [], "dashed": []})
    # metadata (ignored by current engine):
    subtype_detail: str = ""
    placement: str = ""
    flush: str = "centred"
    opening_direction: Optional[str] = None
    sill_ext_mm: Optional[int] = None
    sill_int_mm: Optional[int] = None


# ===========================================================================
# SECTION 5 -- Plan builder
# ===========================================================================

class Plan:
    def __init__(self, plan_id: str, rng: random.Random):
        self.id = plan_id
        self.rng = rng
        self.walls: List[Wall] = []
        self.rooms: List[Room] = []
        self.openings: List[Opening] = []
        self.decoys: List[dict] = []
        self._wc = 0
        self._oc = 0
        self.meta: dict = {}
        self.render: dict = {}
        self.clutter: str = "None."
        self.name = plan_id
        self.description = ""
        self.hard_case = ""

    # -- walls --------------------------------------------------------------
    def wall(self, a: Pt, b: Pt, t: int, wtype="interior", material="",
             hatch=None, angle_class="orthogonal", wid=None) -> Wall:
        self._wc += 1
        wid = wid or f"W{self._wc}"
        d = _unit(_sub(b, a))
        n = _norm(d)
        off = _mul(n, t / 2)
        poly = [_i(_add(a, off)), _i(_add(b, off)),
                _i(_sub(b, off)), _i(_sub(a, off))]
        w = Wall(wid, wtype, int(t), int(round(_len(_sub(b, a)))),
                 [_i(a), _i(b)], poly, None, material, hatch, angle_class)
        self.walls.append(w)
        return w

    def arc_wall(self, center: Pt, radius: float, a0: float, a1: float, t: int,
                 wtype="exterior", material="", hatch=None, wid=None,
                 segs=32) -> Wall:
        self._wc += 1
        wid = wid or f"W{self._wc}"
        outer = _arc_poly(center, radius + t / 2, a0, a1, segs)
        inner = _arc_poly(center, radius - t / 2, a1, a0, segs)
        poly = outer + inner
        cl = _arc_poly(center, radius, a0, a1, segs)
        length = int(round(abs(math.radians(a1 - a0)) * radius))
        w = Wall(wid, wtype, int(t), length, [cl[0], cl[-1]], poly,
                 {"center": _i(center), "radius": int(round(radius)),
                  "a0": round(a0, 2), "a1": round(a1, 2)},
                 material, hatch, "curved")
        self.walls.append(w)
        return w

    # -- rooms --------------------------------------------------------------
    def room(self, poly: List[List[int]], name: str, room_type: str,
             shape: Optional[str] = None, rid=None) -> Room:
        rid = rid or f"R{len(self.rooms) + 1}"
        ipoly = [_i(p) for p in poly]
        sh = shape or self._classify(ipoly)
        r = Room(rid, name, sh, ipoly, room_type)
        self.rooms.append(r)
        return r

    @staticmethod
    def _classify(poly) -> str:
        n = len(poly)
        ortho = all(p[0] == q[0] or p[1] == q[1]
                    for p, q in zip(poly, poly[1:] + poly[:1]))
        if n == 4:
            xs = {p[0] for p in poly}
            ys = {p[1] for p in poly}
            if ortho and len(xs) == 2 and len(ys) == 2:
                return "rectangle"
            return "quad"
        if n == 6 and ortho:
            return "l_shape"
        if n == 8 and ortho:
            return "t_or_u_shape"
        return "polygon"

    # -- openings -----------------------------------------------------------
    def _base(self, a: Pt, b: Pt, offset: float, width: float):
        d = _unit(_sub(b, a))
        center = _add(a, _mul(d, offset))
        p1 = _sub(center, _mul(d, width / 2))
        p2 = _add(center, _mul(d, width / 2))
        return d, center, p1, p2

    def opening_on_wall(self, a: Pt, b: Pt, offset: float, width: int,
                        category: str, detail: str, **kw) -> Opening:
        """Place a door/window/opening on a straight wall axis a->b."""
        self._oc += 1
        d, center, p1, p2 = self._base(a, b, offset, width)
        n = _norm(d)
        oid = kw.get("oid") or (
            ("D" if category == "door" else "W" if category == "window"
             else "O") + str(self._oc))
        if category == "door":
            subtype = DOOR_CANON.get(detail, "SINGLE_HINGED")
        elif category == "window":
            subtype = WINDOW_CANON.get(detail, "FIXED")
        else:
            subtype = detail if detail in ("CASED", "GAP") else "CASED"

        hinge = kw.get("hinge", "left")
        swing = kw.get("swing", "in" if category == "door" else "none")
        side = 1 if swing in ("in", "both", "none") else -1
        if kw.get("side") is not None:
            side = kw["side"]
        panels = kw.get("panels", 1)
        sym = {"lines": [], "arcs": [], "polylines": [], "dashed": []}
        plane = kw.get("plane", "wall")
        ip1, ip2 = _i(p1), _i(p2)
        icenter = [int(round((ip1[0] + ip2[0]) / 2)),
                   int(round((ip1[1] + ip2[1]) / 2))]

        if category == "door":
            self._door_symbol(sym, subtype, p1, p2, center, d, n, width,
                              hinge, side)
            if subtype in ("DOUBLE_HINGED", "FRENCH"):
                panels = 2
        elif category == "window":
            self._window_symbol(sym, subtype, p1, p2, center, d, n, width,
                               kw.get("project", 600), side)
        else:
            self._opening_symbol(sym, subtype, p1, p2, d, n)

        o = Opening(
            id=oid, category=category, subtype=subtype, width_mm=int(width),
            p1=ip1, p2=ip2, center=icenter,
            angle_deg=round(_angle(d), 2),
            height_mm=kw.get("height_mm"),
            hinge=hinge if category == "door" else "none",
            swing=swing if category == "door" else "none",
            panels=panels, sill_mm=kw.get("sill_mm"), head_mm=kw.get("head_mm"),
            plane=plane, group=kw.get("group"), symbol=sym,
            subtype_detail=detail, placement=kw.get("placement", ""),
            flush=kw.get("flush", "centred"),
            opening_direction=kw.get("opening_direction"),
            sill_ext_mm=kw.get("sill_ext_mm"), sill_int_mm=kw.get("sill_int_mm"),
        )
        self.openings.append(o)
        return o

    def _door_symbol(self, sym, subtype, p1, p2, center, d, n, width, hinge, side):
        if subtype == "SINGLE_HINGED":
            hp = p1 if hinge == "left" else p2
            jamb = p2 if hinge == "left" else p1
            closed_ang = _angle(_unit(_sub(jamb, hp)))
            open_ang = closed_ang + 90 * side * (1 if hinge == "left" else -1)
            tip = _pt_at_angle(hp, width, open_ang)
            sym["lines"].append([_i(hp), _i(tip)])
            sym["arcs"].append({"center": _i(hp), "radius": int(width),
                                "a0": round(closed_ang, 2),
                                "a1": round(open_ang, 2)})
        elif subtype in ("DOUBLE_HINGED", "FRENCH"):
            half = width / 2
            for hp, jdir, sgn in ((p1, d, 1), (p2, _mul(d, -1), -1)):
                closed_ang = _angle(jdir)
                open_ang = closed_ang + 90 * side * sgn
                tip = _pt_at_angle(hp, half, open_ang)
                sym["lines"].append([_i(hp), _i(tip)])
                sym["arcs"].append({"center": _i(hp), "radius": int(half),
                                    "a0": round(closed_ang, 2),
                                    "a1": round(open_ang, 2)})
        elif subtype == "SLIDING":
            nn = _mul(_norm(d), 60 * side)
            sym["lines"].append([_i(_add(p1, nn)), _i(_add(center, nn))])
            sym["lines"].append([_i(center), _i(p2)])
        elif subtype == "POCKET":
            pocket_end = _add(p1, _mul(d, width))
            sym["dashed"].append([_i(p1), _i(pocket_end)])
        elif subtype == "BIFOLD":
            q = _add(center, _mul(_norm(d), (width / 4) * side))
            sym["polylines"].append([_i(p1), _i(q), _i(p2)])
        elif subtype == "GARAGE":
            nn = _mul(_norm(d), 80 * side)
            sym["polylines"].append([_i(p1), _i(_add(p1, nn)),
                                     _i(_add(p2, nn)), _i(p2)])
            for k in range(1, 4):
                pk = _add(p1, _mul(d, width * k / 4))
                sym["lines"].append([_i(pk), _i(_add(pk, nn))])

    def _window_symbol(self, sym, subtype, p1, p2, center, d, n, width,
                       project, side):
        inn = _mul(n, 40)
        if subtype in ("CASEMENT", "FIXED", "SLIDING", "LOUVRE", "CLERESTORY",
                       "CORNER"):
            sym["lines"].append([_i(_add(p1, inn)), _i(_add(p2, inn))])
            sym["lines"].append([_i(_sub(p1, inn)), _i(_sub(p2, inn))])
            if subtype == "LOUVRE":
                for k in range(1, 5):
                    pk = _add(p1, _mul(d, width * k / 5))
                    sym["lines"].append([_i(_add(pk, inn)), _i(_sub(pk, inn))])
            if subtype == "SLIDING":
                sym["lines"].append([_i(center), _i(_add(center, _mul(n, 60)))])
            if subtype == "CLERESTORY":
                sym["dashed"].append([_i(p1), _i(p2)])
        elif subtype == "AWNING":
            sym["lines"].append([_i(_add(p1, inn)), _i(_add(p2, inn))])
            sym["lines"].append([_i(_sub(p1, inn)), _i(_sub(p2, inn))])
            sym["dashed"].append([_i(p1), _i(center)])
            sym["dashed"].append([_i(p2), _i(center)])
        elif subtype == "BAY":
            proj = _mul(_unit(n), project * side)
            b1 = _add(p1, _mul(d, width * 0.18))
            b2 = _add(p2, _mul(d, -width * 0.18))
            sym["polylines"].append([_i(p1), _i(_add(b1, proj)),
                                     _i(_add(b2, proj)), _i(p2)])

    def _opening_symbol(self, sym, subtype, p1, p2, d, n):
        if subtype == "CASED":
            nn = _mul(_norm(d), 30)
            sym["lines"].append([_i(_add(p1, nn)), _i(_sub(p1, nn))])
            sym["lines"].append([_i(_add(p2, nn)), _i(_sub(p2, nn))])
        # GAP: no symbol

    def rooflight(self, cx: int, cy: int, hw: int, width: int) -> Opening:
        self._oc += 1
        oid = f"RL{self._oc}"
        sym = {"lines": [], "arcs": [], "polylines": [],
               "dashed": [[[cx - hw, cy - hw], [cx + hw, cy - hw]],
                          [[cx + hw, cy - hw], [cx + hw, cy + hw]],
                          [[cx + hw, cy + hw], [cx - hw, cy + hw]],
                          [[cx - hw, cy + hw], [cx - hw, cy - hw]],
                          [[cx - hw, cy - hw], [cx + hw, cy + hw]],
                          [[cx + hw, cy - hw], [cx - hw, cy + hw]]]}
        o = Opening(id=oid, category="window", subtype="FIXED",
                    width_mm=int(width), p1=[cx - width // 2, cy],
                    p2=[cx + width // 2, cy], center=[cx, cy], angle_deg=0.0,
                    plane="roof", symbol=sym, subtype_detail="ROOF_LIGHT")
        self.openings.append(o)
        return o

    def decoy(self, kind: str, geometry: dict):
        self.decoys.append({"kind": kind, "geometry": geometry})

    # -- transforms ---------------------------------------------------------
    def rotate(self, deg: float, about: Pt = (0, 0)):
        def R(p):
            return _i(_rotate_pt(p, deg, about))
        for w in self.walls:
            w.centerline = [R(p) for p in w.centerline]
            w.polygon = [R(p) for p in w.polygon]
            if w.arc:
                w.arc["center"] = R(w.arc["center"])
                w.arc["a0"] = round(w.arc["a0"] + deg, 2)
                w.arc["a1"] = round(w.arc["a1"] + deg, 2)
        for r in self.rooms:
            r.polygon = [R(p) for p in r.polygon]
        for o in self.openings:
            o.p1, o.p2, o.center = R(o.p1), R(o.p2), R(o.center)
            o.angle_deg = round(o.angle_deg + deg, 2)
            for key in ("lines", "polylines", "dashed"):
                o.symbol[key] = [[R(p) for p in seg] for seg in o.symbol[key]]
            for arc in o.symbol["arcs"]:
                arc["center"] = R(arc["center"])
                arc["a0"] = round(arc["a0"] + deg, 2)
                arc["a1"] = round(arc["a1"] + deg, 2)
        for dc in self.decoys:
            _rotate_geometry(dc["geometry"], R)

    # -- serialise ----------------------------------------------------------
    def bbox(self):
        xs, ys = [], []
        for w in self.walls:
            for p in w.polygon:
                xs.append(p[0]); ys.append(p[1])
        for r in self.rooms:
            for p in r.polygon:
                xs.append(p[0]); ys.append(p[1])
        if not xs:
            return [0, 0, 0, 0]
        return [min(xs), min(ys), max(xs), max(ys)]

    def to_dict(self) -> dict:
        bb = self.bbox()
        doors = sum(1 for o in self.openings if o.category == "door")
        windows = sum(1 for o in self.openings if o.category == "window")
        return {
            "id": self.id, "group": self.meta.get("building_type", "gen"),
            "name": self.name, "description": self.description,
            "hard_case": self.hard_case,
            "units": "mm", "origin": [0, 0],
            "bbox": bb,
            "footprint": {"w": bb[2] - bb[0], "h": bb[3] - bb[1]},
            "counts": {"walls": len(self.walls), "rooms": len(self.rooms),
                       "openings": len(self.openings), "doors": doors,
                       "windows": windows, "decoys": len(self.decoys)},
            "walls": [_wall_dict(w) for w in self.walls],
            "rooms": [_room_dict(r) for r in self.rooms],
            "openings": [_opening_dict(o) for o in self.openings],
            "decoys": self.decoys,
            "clutter": self.clutter,
            "render": self.render,
            "metadata": self.meta,
        }


def _wall_dict(w: Wall) -> dict:
    return {"id": w.id, "type": w.type, "thickness_mm": w.thickness_mm,
            "length_mm": w.length_mm, "centerline": w.centerline,
            "polygon": w.polygon, "arc": w.arc,
            "material": w.material, "hatch": w.hatch,
            "angle_class": w.angle_class}


def _room_dict(r: Room) -> dict:
    return {"id": r.id, "name": r.name, "shape": r.shape,
            "polygon": r.polygon, "room_type": r.room_type}


def _opening_dict(o: Opening) -> dict:
    return asdict(o)


def _rotate_geometry(geom: dict, R):
    for key, val in list(geom.items()):
        if key in ("rect", "around") and isinstance(val, list) and len(val) == 4:
            (x0, y0), (x1, y1) = R([val[0], val[1]]), R([val[2], val[3]])
            geom[key] = [x0, y0, x1, y1]
        elif key == "circle" and isinstance(val, list) and len(val) == 3:
            c = R([val[0], val[1]])
            geom[key] = [c[0], c[1], val[2]]
        elif key == "bubble" and isinstance(val, list) and len(val) == 3:
            c = R([val[0], val[1]])
            geom[key] = [c[0], c[1], val[2]]
        elif key == "line" and isinstance(val, list) and len(val) == 2 \
                and isinstance(val[0], list):
            geom[key] = [R(val[0]), R(val[1])]
        elif key == "lines" and isinstance(val, list):
            geom[key] = [[R(p) for p in seg] for seg in val]


# ===========================================================================
# SECTION 6 -- BSP room partitioning
# ===========================================================================

@dataclass
class Cell:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self): return self.x1 - self.x0
    @property
    def h(self): return self.y1 - self.y0
    @property
    def area(self): return self.w * self.h


def bsp_split(region: Cell, target: int, rng: random.Random,
              min_dim: int = 2200, min_small: int = 1200) -> List[Cell]:
    """Recursively split a rectangle into ~target sub-rectangles."""
    cells = [region]
    guard = 0
    while len(cells) < target and guard < target * 6:
        guard += 1
        # pick a splittable cell, biased to large area
        splittable = [c for c in cells
                      if c.w >= 2 * min_small + 100 or c.h >= 2 * min_small + 100]
        if not splittable:
            break
        splittable.sort(key=lambda c: c.area, reverse=True)
        # weighted pick among the largest few
        c = rng.choices(splittable[:max(1, len(splittable) // 2 + 1)],
                        weights=[s.area for s in
                                 splittable[:max(1, len(splittable) // 2 + 1)]],
                        k=1)[0]
        # choose axis: prefer splitting the longer side
        vertical = c.w >= c.h
        if rng.random() < 0.18:
            vertical = not vertical
        span = c.w if vertical else c.h
        lo_min = min_small if rng.random() < 0.32 else min_dim
        if span < 2 * lo_min + 100:
            # try the other axis
            vertical = not vertical
            span = c.w if vertical else c.h
            if span < 2 * lo_min + 100:
                continue
        frac = rng.uniform(0.30, 0.70)
        cut = snap(span * frac, 100)
        cut = max(lo_min, min(span - lo_min, cut))
        cells.remove(c)
        if vertical:
            xm = c.x0 + cut
            cells.append(Cell(c.x0, c.y0, xm, c.y1))
            cells.append(Cell(xm, c.y0, c.x1, c.y1))
        else:
            ym = c.y0 + cut
            cells.append(Cell(c.x0, c.y0, c.x1, ym))
            cells.append(Cell(c.x0, ym, c.x1, c.y1))
    return cells


# ===========================================================================
# SECTION 7 -- footprint generation -> list of room cells (+ feature edges)
# ===========================================================================

@dataclass
class Feature:
    """A non-orthogonal exterior feature to splice in after rectilinear walls."""
    kind: str                    # 'arc' | 'angle'
    # for arc:
    center: Optional[Pt] = None
    radius: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    # the rectilinear edge this feature REPLACES (so we drop that wall):
    drop_edge: Optional[Tuple[str, int, float, float]] = None  # (axis,coord,lo,hi)
    # vertices to graft onto the adjacent room polygon:
    graft_room_idx: Optional[int] = None
    graft_poly: Optional[List[List[int]]] = None
    thickness: int = 300
    material: str = "REINFORCED_CONCRETE"


def _rects_to_polys(cells: List[Cell]) -> List[List[List[int]]]:
    return [[[c.x0, c.y0], [c.x1, c.y0], [c.x1, c.y1], [c.x0, c.y1]]
            for c in cells]


def make_footprint(shape: str, W: int, H: int, target_rooms: int, n_arcs: int,
                   rng: random.Random) -> Tuple[List[List[List[int]]], List[Feature]]:
    """Return (room polygons, arc features)."""
    feats: List[Feature] = []
    if shape in ("L_shape", "T_shape", "U_shape"):
        cells = _rectilinear_footprint(shape, W, H, target_rooms, rng)
        return _rects_to_polys(cells), feats

    cells = bsp_split(Cell(0, 0, W, H), target_rooms, rng)
    polys = _rects_to_polys(cells)

    if shape in ("rectangle_with_curved_end", "fully_curved_facade",
                 "organic_multi_arc"):
        feats = _curved_features(shape, W, H, n_arcs, rng)
        return polys, feats

    if shape == "irregular_polygon":
        polys = _clip_corners(polys, rng)
        return polys, feats

    # rectangle / rectangle_with_bay
    return polys, feats


def _clip_corners(polys, rng) -> List[List[List[int]]]:
    """Clip 1-2 outer building corners diagonally -> pentagon rooms + angled
    exterior walls (added later from the non-axis edges)."""
    BX0 = min(p[0] for poly in polys for p in poly)
    BY0 = min(p[1] for poly in polys for p in poly)
    BX1 = max(p[0] for poly in polys for p in poly)
    BY1 = max(p[1] for poly in polys for p in poly)
    W, H = BX1 - BX0, BY1 - BY0
    corners = {"NE": [BX1, BY1], "NW": [BX0, BY1],
               "SE": [BX1, BY0], "SW": [BX0, BY0]}
    out = [list(p) for p in polys]
    for name in rng.sample(list(corners), rng.randint(1, 2)):
        cv = corners[name]
        for idx, poly in enumerate(out):
            if len(poly) == 4 and cv in poly:
                bx = snap(rng.uniform(1200, max(1300, min(3500, 0.35 * W))), 100)
                by = snap(rng.uniform(1200, max(1300, min(3500, 0.35 * H))), 100)
                cw = poly[1][0] - poly[0][0]
                ch = poly[2][1] - poly[1][1]
                if bx < cw - 900 and by < ch - 900:
                    out[idx] = _clip_corner_vertex(poly, cv, bx, by)
                break
    return out


def _along(frm, to, dist):
    d = _unit(_sub(to, frm))
    return _i(_add(frm, _mul(d, dist)))


def _clip_corner_vertex(poly, corner, bx, by):
    n = len(poly)
    i = poly.index(corner)
    prev = poly[(i - 1) % n]
    nxt = poly[(i + 1) % n]
    pin = _along(corner, prev, bx if prev[1] == corner[1] else by)
    pout = _along(corner, nxt, bx if nxt[1] == corner[1] else by)
    out = []
    for k, p in enumerate(poly):
        if k == i:
            out.append(pin)
            out.append(pout)
        else:
            out.append(list(p))
    return out


def _rectilinear_footprint(shape, W, H, target_rooms, rng) -> List[Cell]:
    """L/T/U by removing a void; BSP the remaining wings."""
    cells: List[Cell] = []
    if shape == "L_shape":
        cw = snap(rng.uniform(0.30, 0.62) * W, 100)
        ch = snap(rng.uniform(0.30, 0.62) * H, 100)
        corner = rng.choice(["NE", "NW", "SE", "SW"])
        # wing A = bottom full strip, wing B = remaining vertical strip
        if corner == "NE":
            wings = [Cell(0, 0, W, H - ch), Cell(0, H - ch, W - cw, H)]
        elif corner == "NW":
            wings = [Cell(0, 0, W, H - ch), Cell(cw, H - ch, W, H)]
        elif corner == "SE":
            wings = [Cell(0, ch, W, H), Cell(0, 0, W - cw, ch)]
        else:  # SW
            wings = [Cell(0, ch, W, H), Cell(cw, 0, W, ch)]
    elif shape == "T_shape":
        sw = snap(rng.uniform(0.34, 0.5) * W, 100)
        sh = snap(rng.uniform(0.4, 0.6) * H, 100)
        x0 = (W - sw) // 2
        wings = [Cell(0, H - max(1500, H - sh), W, H),    # cross bar (top)
                 Cell(x0, 0, x0 + sw, H - (H - sh))]       # stem (down)
        wings = [Cell(0, H - (H - sh), W, H), Cell(x0, 0, x0 + sw, H - (H - sh))]
    else:  # U_shape
        cw = snap(rng.uniform(0.28, 0.44) * W, 100)
        ch = snap(rng.uniform(0.4, 0.66) * H, 100)
        x0 = (W - cw) // 2
        wings = [Cell(0, 0, W, H - ch),          # base
                 Cell(0, H - ch, x0, H),         # left leg
                 Cell(x0 + cw, H - ch, W, H)]    # right leg
    # allocate rooms to wings by area
    total = sum(w.area for w in wings) or 1
    for i, wing in enumerate(wings):
        share = max(1, round(target_rooms * wing.area / total))
        cells.extend(bsp_split(wing, share, rng))
    return cells


def _curved_features(shape, W, H, n_arcs, rng) -> List[Feature]:
    feats = []
    n_arcs = max(1, n_arcs)
    if shape == "rectangle_with_curved_end":
        # semicircular / curved short end on the east side
        r = H / 2
        cy = H / 2
        feats.append(Feature(kind="arc", center=(W, cy), radius=r,
                             a0=-90, a1=90,
                             drop_edge=("v", W, 0, H),
                             thickness=rng.choice([250, 300, 340]),
                             material="REINFORCED_CONCRETE"))
        return feats
    # convex facade(s) on the top edge, splitting the span
    seg = W / n_arcs
    for k in range(n_arcs):
        x0 = k * seg
        x1 = (k + 1) * seg
        bulge = rng.uniform(600, 2200)
        chord = x1 - x0
        # radius from chord + bulge:  r = (c^2/4 + b^2) / (2b)
        b = max(300.0, bulge)
        r = (chord * chord / 4 + b * b) / (2 * b)
        cx = (x0 + x1) / 2
        cy = H - r + b      # center below the wall so it bulges up (north)
        half = math.degrees(math.asin(min(1.0, (chord / 2) / r)))
        feats.append(Feature(kind="arc", center=(cx, cy), radius=r,
                             a0=90 - half, a1=90 + half,
                             drop_edge=("h", H, x0, x1),
                             thickness=rng.choice([250, 300, 340]),
                             material=rng.choice(
                                 ["REINFORCED_CONCRETE", "CAVITY_BRICK"])))
    return feats


# ===========================================================================
# SECTION 8 -- walls from rooms (exact interval method)
# ===========================================================================

@dataclass
class Adjacency:
    a: int          # room index
    b: int          # room index
    axis: str       # 'v' | 'h'
    coord: int
    lo: int
    hi: int


def _room_edges(poly: List[List[int]]):
    """Yield axis-aligned edges as (axis, coord, lo, hi, interior_side).

    interior_side: for vertical edge (+1 room on +x side, -1 on -x side);
    for horizontal edge (+1 room above (+y), -1 below).
    """
    n = len(poly)
    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if x0 == x1 and y0 != y1:        # vertical
            lo, hi = sorted((y0, y1))
            mid = (lo + hi) / 2
            inside_plus = _point_in_poly((x0 + 1.0, mid), poly)
            yield ("v", x0, lo, hi, 1 if inside_plus else -1)
        elif y0 == y1 and x0 != x1:      # horizontal
            lo, hi = sorted((x0, x1))
            mid = (lo + hi) / 2
            inside_plus = _point_in_poly((mid, y0 + 1.0), poly)
            yield ("h", y0, lo, hi, 1 if inside_plus else -1)


def build_walls_from_rooms(plan: Plan, rooms_poly: List[List[List[int]]],
                           ext_mat: str, ext_t: int, ext_hatch,
                           interior_picker) -> List[Adjacency]:
    """Create walls separating rooms / bounding the plan. Returns adjacencies."""
    # gather edges grouped by (axis, coord)
    vlines: Dict[int, dict] = {}
    hlines: Dict[int, dict] = {}
    for idx, poly in enumerate(rooms_poly):
        for axis, coord, lo, hi, side in _room_edges(poly):
            book = vlines if axis == "v" else hlines
            d = book.setdefault(int(round(coord)),
                                {"left": [], "right": []})
            if side > 0:        # room on +side -> contributes to 'right'
                d["right"].append((lo, hi, idx))
            else:
                d["left"].append((lo, hi, idx))

    adjacencies: List[Adjacency] = []

    def emit(axis, coord, book):
        for coord, d in book.items():
            left = d["left"]      # room on -side (its + boundary here)
            right = d["right"]    # room on +side
            left_iv = _iv_union([(lo, hi) for lo, hi, _ in left])
            right_iv = _iv_union([(lo, hi) for lo, hi, _ in right])
            interior = _iv_intersect(left_iv, right_iv)
            union = _iv_union(left_iv + right_iv)
            exterior = _iv_subtract(union, interior)
            # interior walls + adjacency (pair rooms across the segment)
            for lo, hi in interior:
                if hi - lo < MIN_WALL:
                    continue
                a = _segment_endpoints(axis, coord, lo, hi)
                mat, t, hatch, wtype = interior_picker()
                plan.wall(a[0], a[1], t, wtype=wtype, material=mat,
                          hatch=hatch,
                          angle_class="orthogonal")
                # find the two rooms
                ra = _room_covering(left, lo, hi)
                rb = _room_covering(right, lo, hi)
                if ra is not None and rb is not None:
                    adjacencies.append(Adjacency(ra, rb, axis, int(coord),
                                                 int(lo), int(hi)))
            for lo, hi in exterior:
                if hi - lo < MIN_WALL:
                    continue
                a = _segment_endpoints(axis, coord, lo, hi)
                plan.wall(a[0], a[1], ext_t, wtype="exterior", material=ext_mat,
                          hatch=ext_hatch, angle_class="orthogonal")

    emit("v", 0, vlines)
    emit("h", 0, hlines)
    return adjacencies


def _segment_endpoints(axis, coord, lo, hi):
    if axis == "v":
        return ((coord, lo), (coord, hi))
    return ((lo, coord), (hi, coord))


def _room_covering(entries, lo, hi):
    mid = (lo + hi) / 2
    for elo, ehi, idx in entries:
        if elo - 1 <= mid <= ehi + 1:
            return idx
    return None


# ===========================================================================
# SECTION 9 -- opening placement (connectivity-aware)
# ===========================================================================

def _occupied_ok(occ: List[Tuple[float, float]], lo, hi):
    for a, b in occ:
        if hi > a - GAP_BETWEEN and lo < b + GAP_BETWEEN:
            return False
    return True


def sample_placement(rng, length, width):
    """Return (center_offset, placement_label) along a wall of `length`."""
    pl = wchoice(rng, PLACEMENTS)
    margin = JAMB_CLEAR + width / 2
    lo_ok = margin
    hi_ok = length - margin
    if hi_ok <= lo_ok:
        return length / 2, "centered"
    if pl == "centered":
        return length / 2, pl
    if pl == "offset_left":
        return max(lo_ok, min(hi_ok, length * rng.uniform(0.15, 0.40))), pl
    if pl == "offset_right":
        return max(lo_ok, min(hi_ok, length * rng.uniform(0.60, 0.85))), pl
    if pl == "near_corner_left":
        return max(lo_ok, 300 + width / 2), pl
    return min(hi_ok, length - 300 - width / 2), pl


def _junctions_on_segment(plan: "Plan", a: Pt, b: Pt, tol: float = 3.0):
    """Distances along the segment a->b (from a) at which ANOTHER wall meets it
    transversely -- i.e. a wall endpoint lands on this line strictly between the
    ends.  These are the points an opening must not straddle, or the meeting
    wall would cross (block) the opening.  Returns (sorted distances, length)."""
    ax, ay = a
    bx, by = b
    L = math.hypot(bx - ax, by - ay)
    if L < 1:
        return [], 0.0
    ux, uy = (bx - ax) / L, (by - ay) / L          # unit along the segment
    out = []
    for w in plan.walls:
        for e in (w.centerline[0], w.centerline[1]):
            dx, dy = e[0] - ax, e[1] - ay
            t = dx * ux + dy * uy                   # projection (distance from a)
            if tol < t < L - tol:
                perp = abs(-dx * uy + dy * ux)      # distance off the line
                if perp <= tol:
                    out.append(t)
    return sorted(set(round(x, 1) for x in out)), L


def _clear_spans(plan: "Plan", a: Pt, b: Pt, margin: float):
    """Sub-spans (lo,hi distances from a) of segment a->b that are clear of
    transverse wall junctions, each already inset by `margin` (jamb clearance)
    from the junctions / ends."""
    js, L = _junctions_on_segment(plan, a, b)
    if L <= 0:
        return []
    bounds = [0.0] + js + [L]
    spans = []
    for i in range(len(bounds) - 1):
        lo = bounds[i] + margin
        hi = bounds[i + 1] - margin
        if hi > lo:
            spans.append((lo, hi))
    return spans


# standard, sensible opening widths (mm) -- openings are only ever sized from
# these, so a width is always a real catalogue value (never an arbitrary
# shrunk-to-fit number).
STD_DOOR_W = [600, 610, 650, 686, 700, 711, 750, 762, 800, 813, 826, 838,
              900, 950, 1000, 1050, 1200]
STD_WINDOW_W = [400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1500, 1800,
                2000, 2200, 2400, 3000, 3600, 4200, 4800]


def _width_choices(preferred: int, catalogue: List[int]) -> List[int]:
    """[preferred, then catalogue sizes smaller than it, largest-first] -- used
    so an opening keeps its preferred size when it fits, else steps down through
    standard sizes rather than being squashed to a random width."""
    smaller = sorted((w for w in catalogue if w < preferred), reverse=True)
    return [int(preferred)] + smaller


def sample_clear(rng, plan: "Plan", a: Pt, b: Pt, widths, occ,
                 margin: float = JAMB_CLEAR):
    """Choose (offset_from_a, width) for an opening placed inside one
    junction-free span of a->b that also clears existing openings `occ`.

    `widths` is an ordered list of acceptable STANDARD widths (mm), tried in
    order -- typically [preferred, then descending fallbacks].  The first width
    that fits a clear, unoccupied span is used, so the chosen width is always a
    sensible catalogue size.  Returns (None, None) if nothing fits."""
    if isinstance(widths, (int, float)):
        widths = [int(widths)]
    spans = _clear_spans(plan, a, b, margin)
    if not spans:
        return None, None
    spans.sort(key=lambda s: s[1] - s[0], reverse=True)   # widest span first
    for w in widths:
        w = int(w)
        if w <= 0:
            continue
        for lo, hi in spans:
            if hi - lo < w:
                continue
            for _ in range(6):
                c = rng.uniform(lo + w / 2, hi - w / 2)
                if _occupied_ok(occ, c - w / 2, c + w / 2):
                    return c, w
    return None, None


def place_doors(plan: Plan, adjacencies: List[Adjacency], rooms_poly,
                btype, region, rng) -> int:
    """Place interior doors connectivity-first: every adjacency that still
    bridges two disconnected rooms gets a door; extras are added at random.
    Returns the number of connected room components afterwards."""
    nrooms = len(rooms_poly)
    parent = list(range(nrooms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
            return True
        return False

    imperial = rng.random() < 0.30
    door_occ: Dict[tuple, List] = {}
    usable = [adj for adj in adjacencies
              if (adj.hi - adj.lo) >= 760 + 2 * JAMB_CLEAR
              and adj.a < nrooms and adj.b < nrooms]
    rng.shuffle(usable)
    for adj in usable:
        bridges = find(adj.a) != find(adj.b)
        if not bridges and rng.random() >= 0.22:
            continue
        wkey = (adj.axis, adj.coord)
        occ = door_occ.setdefault(wkey, [])
        rt_a = plan.rooms[adj.a].room_type
        rt_b = plan.rooms[adj.b].room_type
        width = pick_interior_door_width(rng, rt_a, rt_b, imperial)
        a = _segment_endpoints(adj.axis, adj.coord, adj.lo, adj.hi)[0]
        b = _segment_endpoints(adj.axis, adj.coord, adj.lo, adj.hi)[1]
        # place the door only within a span clear of transverse wall junctions,
        # so no wall crosses (blocks) the doorway (S10); size from standard
        # widths (preferred, then smaller standards) so the width stays sensible
        off, width = sample_clear(rng, plan, a, b,
                                  _width_choices(width, STD_DOOR_W), occ)
        if off is None:
            continue
        occ.append((off - width / 2, off + width / 2))
        detail = pick_door_subtype(rng, rt_a, rt_b)
        plan.opening_on_wall(a, b, off, width, "door", detail,
                             hinge=rng.choice(["left", "right"]),
                             swing=rng.choice(["in", "out"]),
                             placement="clear_span", height_mm=2100)
        union(adj.a, adj.b)
    return len({find(i) for i in range(nrooms)})


def pick_interior_door_width(rng, rt_a, rt_b, imperial):
    rts = {rt_a, rt_b}
    if rts & {"bathroom", "WC", "shower", "wardrobe", "bathroom_en_suite",
              "linen_cupboard", "storage", "store"}:
        return rng.choice(DOOR_SIZES["bathroom"])
    if rng.random() < 0.08:
        return rng.choice(DOOR_SIZES["accessible"])
    pool = DOOR_SIZES["interior_imperial"] if imperial \
        else DOOR_SIZES["interior_metric"]
    return rng.choice(pool)


def pick_door_subtype(rng, rt_a, rt_b):
    rts = {rt_a, rt_b}
    if rts & {"bathroom", "WC", "wardrobe", "storage"} and rng.random() < 0.4:
        return rng.choice(["POCKET", "SLIDING", "SINGLE_HINGED"])
    return wchoice(rng, {k: v for k, v in DOOR_SUBTYPES.items()
                         if k not in ("GARAGE",)})


def place_exterior_openings(plan: Plan, btype, region, rng,
                            curved_walls: List[Wall]):
    """Entry doors + windows on exterior straight walls; openings on arcs."""
    ext_walls = [w for w in plan.walls if w.type == "exterior" and not w.arc]
    if not ext_walls:
        ext_walls = [w for w in plan.walls if not w.arc]
    imperial = region in ("us", "uk") or rng.random() < 0.3

    # ---- entry door(s) ----
    long_walls = sorted(ext_walls, key=lambda w: w.length_mm, reverse=True)
    n_entries = 1 if btype in ("residential_apartment", "residential_house") \
        else rng.randint(1, 2)
    entry_occ: Dict[str, List] = {}
    for w in long_walls[:n_entries]:
        if w.length_mm < 1100:
            continue
        detail, width = pick_entry_door(rng, btype)
        a, b = w.centerline[0], w.centerline[1]
        occ = entry_occ.setdefault(w.id, [])
        # entrance must sit in a span clear of interior-wall junctions (S10);
        # keep the picked entry width, else step down to standard single-leaf
        # entrance widths (never an arbitrary shrunk value)
        cands = [width] + [x for x in (1200, 1050, 1000, 900) if x < width]
        off, fitted = sample_clear(rng, plan, a, b, cands, occ)
        if off is None:
            continue
        # if a wide leaf (garage/sliding/double) couldn't fit and we stepped
        # down to a single-leaf width, make the subtype match the actual size
        if fitted < width and detail in ("GARAGE", "SLIDING", "REVOLVING",
                                         "DOUBLE_HINGED", "FRENCH"):
            detail = "SINGLE_HINGED"
        width = fitted
        plan.opening_on_wall(a, b, off, width, "door", detail,
                             hinge=rng.choice(["left", "right"]),
                             swing="in" if btype.startswith("res") else
                             rng.choice(["in", "out"]),
                             placement="clear_span", height_mm=2100,
                             flush=wchoice(rng, {"centred": 60,
                                                 "flush_exterior": 20,
                                                 "flush_interior": 20}))
        occ.append((off - width / 2, off + width / 2))

    # ---- windows ----
    allow_bay = btype in ("residential_apartment", "residential_house")
    allow_shop = btype in ("retail_unit", "restaurant_cafe",
                           "mixed_use_ground_floor")
    for w in ext_walls:
        if w.length_mm < 900:
            continue
        nopen = wchoice(rng, _wall_open_dist(w.length_mm))
        occ = entry_occ.get(w.id, [])
        a, b = w.centerline[0], w.centerline[1]
        for _ in range(nopen):
            detail = pick_window_subtype(rng, allow_bay, allow_shop, btype)
            width = pick_window_width(rng, w.length_mm, detail)
            # window must sit in a span clear of interior-wall junctions (S10);
            # size from standard window widths so it stays sensible
            off, fitted = sample_clear(rng, plan, a, b,
                                       _width_choices(width, STD_WINDOW_W), occ)
            if off is None:
                continue
            width = fitted
            occ.append((off - width / 2, off + width / 2))
            sill = rng.choice([600, 750, 900, 1000, 1100])
            plan.opening_on_wall(
                a, b, off, width, "window", detail,
                placement="clear_span", height_mm=rng.choice([900, 1200, 1500]),
                sill_mm=sill, head_mm=sill + rng.choice([900, 1200, 1500]),
                sill_ext_mm=rng.randint(25, 75), sill_int_mm=rng.randint(80, 220),
                opening_direction=rng.choice(
                    ["left_hung", "right_hung", "top_hung", "bottom_hung"]),
                project=rng.uniform(450, 750))

    # ---- openings on curved walls (by angle) ----
    for w in curved_walls:
        _place_on_arc(plan, w, btype, rng)


def _place_on_arc(plan: Plan, w: Wall, btype, rng):
    arc = w.arc
    cx, cy = arc["center"]
    r = arc["radius"]
    a0, a1 = arc["a0"], arc["a1"]
    sweep = abs(a1 - a0)
    if sweep < 8 or r < 400:
        return
    n = rng.randint(1, max(1, min(3, int(sweep // 35))))
    used = []
    for _ in range(n):
        # window width in mm -> angular width
        width = rng.choice([800, 1000, 1200, 1500])
        ang_w = math.degrees(width / r)
        if ang_w > sweep - 6:
            continue
        pos = rng.uniform(a0 + ang_w / 2 + 2, a1 - ang_w / 2 - 2)
        if any(abs(pos - u) < ang_w + 3 for u in used):
            continue
        used.append(pos)
        pa = _pt_at_angle((cx, cy), r, pos - ang_w / 2)
        pb = _pt_at_angle((cx, cy), r, pos + ang_w / 2)
        ip1, ip2 = _i(pa), _i(pb)
        actual_w = int(round(_len(_sub(pb, pa))))
        icenter = [int(round((ip1[0] + ip2[0]) / 2)),
                   int(round((ip1[1] + ip2[1]) / 2))]
        plan._oc += 1
        # radial jamb lines (toward center) + glazing chord
        j1o = _i(_pt_at_angle((cx, cy), r + 40, pos - ang_w / 2))
        j1i = _i(_pt_at_angle((cx, cy), r - 40, pos - ang_w / 2))
        j2o = _i(_pt_at_angle((cx, cy), r + 40, pos + ang_w / 2))
        j2i = _i(_pt_at_angle((cx, cy), r - 40, pos + ang_w / 2))
        sym = {"lines": [[j1i, j1o], [j2i, j2o]],
               "arcs": [{"center": [cx, cy], "radius": int(r),
                         "a0": round(pos - ang_w / 2, 2),
                         "a1": round(pos + ang_w / 2, 2)}],
               "polylines": [], "dashed": []}
        o = Opening(id=f"W{plan._oc}", category="window", subtype="CASEMENT",
                    width_mm=actual_w, p1=ip1, p2=ip2, center=icenter,
                    angle_deg=round(pos, 2), plane="wall", symbol=sym,
                    subtype_detail="CURVED_GLAZING", sill_mm=900)
        plan.openings.append(o)


def _wall_open_dist(length):
    if length > 9000:
        return {1: 20, 2: 30, 3: 35, 4: 15}
    if length > 5000:
        return {1: 40, 2: 38, 3: 20, 4: 2}
    if length > 2600:
        return {1: 62, 2: 30, 3: 8}
    return {1: 80, 2: 20}


def pick_entry_door(rng, btype):
    if btype in ("industrial_unit",) and rng.random() < 0.6:
        return "GARAGE", rng.choice(DOOR_SIZES["garage"])
    if btype in ("retail_unit", "restaurant_cafe", "mixed_use_ground_floor"):
        return rng.choice(["SLIDING", "DOUBLE_HINGED", "REVOLVING"]), \
            rng.choice([1600, 1800, 2100, 1200])
    if btype in ("office_open_plan", "office_cellular", "hotel_floor",
                 "hospital_ward", "school_classroom_block", "sports_facility"):
        return rng.choice(["DOUBLE_HINGED", "SLIDING"]), \
            rng.choice([1600, 1800, 2100])
    return "SINGLE_HINGED", rng.choice(DOOR_SIZES["main_entry"])


def pick_window_subtype(rng, allow_bay, allow_shop, btype):
    table = dict(WINDOW_SUBTYPES)
    if not allow_bay:
        table.pop("BAY", None)
    if not allow_shop:
        table.pop("SHOPFRONT", None)
    if btype in ("office_open_plan", "office_cellular"):
        table["FIXED"] = table.get("FIXED", 0) + 14
    return wchoice(rng, table)


def pick_window_width(rng, wall_len, detail):
    """Pick a sensible, standard window width (mm) for the given category."""
    if detail in ("SHOPFRONT",):
        pool = [w for w in (2400, 3000, 3600, 4200, 4800)
                if w <= wall_len * 0.8] or [2400]
        return rng.choice(pool)
    cat = wchoice(rng, WINDOW_SIZE_CATS)
    lo, hi = WINDOW_SIZE_RANGE[cat]
    pool = [w for w in STD_WINDOW_W if lo <= w <= hi]
    pool = [w for w in pool if w <= wall_len] or [min(STD_WINDOW_W)]
    return rng.choice(pool)


# ===========================================================================
# SECTION 10 -- decoys (columns / furniture / clutter -> hard negatives)
# ===========================================================================

def add_columns(plan: Plan, btype, bb, rng):
    if rng.random() > COLUMN_PROB.get(btype, 0.3):
        return 0
    gx = rng.uniform(4000, 9000)
    gy = rng.uniform(4000, 9000)
    shape = wchoice(rng, {"square": 55, "round": 35, "cruciform": 10})
    size = int(rng.uniform(250, 600)) if shape != "round" \
        else int(rng.uniform(250, 500))
    minx, miny, maxx, maxy = bb
    n = 0
    x = minx + gx
    while x < maxx - 500 and n < 60:
        y = miny + gy
        while y < maxy - 500 and n < 60:
            jx = int(x + rng.uniform(-120, 120))
            jy = int(y + rng.uniform(-120, 120))
            geom = {"shape": shape, "size_mm": size,
                    "rect": [jx - size // 2, jy - size // 2,
                             jx + size // 2, jy + size // 2]}
            if shape == "round":
                geom = {"shape": "round", "size_mm": size,
                        "circle": [jx, jy, size // 2]}
            plan.decoy("xref_column", geom)
            n += 1
            y += gy
        x += gx
    return n


def add_furniture(plan: Plan, density: float, rng):
    n = 0
    for r in plan.rooms:
        if rng.random() > density:
            continue
        poly = r.polygon
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        rw, rh = x1 - x0, y1 - y0
        rt = r.room_type
        items = FURNITURE_BY_ROOM.get(rt, [])
        if not items and "bed" in rt:
            items = FURNITURE_BY_ROOM["bedroom"]
        for item, (iw, ih) in items:
            if iw + 200 > rw or ih + 200 > rh:
                continue
            if rng.random() > density:
                continue
            px = int(rng.uniform(x0 + 80, x1 - iw - 80))
            py = int(rng.uniform(y0 + 80, y1 - ih - 80))
            plan.decoy("furniture", {"item": item,
                                     "rect": [px, py, px + iw, py + ih]})
            n += 1
    return n


FURNITURE_BY_ROOM = {
    "bedroom": [("bed", (1350, 1900)), ("wardrobe", (600, 1800)),
                ("bedside", (500, 500))],
    "master_bedroom": [("bed", (1800, 2000)), ("wardrobe", (600, 2400)),
                       ("desk", (750, 1400))],
    "living": [("sofa", (2200, 850)), ("coffee_table", (1000, 500)),
               ("tv_unit", (1800, 450))],
    "kitchen": [("counter", (600, 3000)), ("island", (1500, 900)),
                ("fridge", (600, 650))],
    "dining": [("dining_table", (1600, 900))],
    "bathroom": [("bath", (1700, 700)), ("basin", (600, 460)),
                 ("wc", (380, 680))],
    "WC": [("wc", (380, 680)), ("basin", (500, 400))],
    "private_office": [("desk", (1600, 800)), ("chair", (650, 650))],
    "meeting_room": [("table", (2400, 1200))],
    "guest_room": [("bed", (1500, 2000)), ("desk", (1200, 600))],
    "classroom": [("desk", (1200, 600)), ("desk", (1200, 600))],
    "dining_area": [("table", (750, 750)), ("table", (900, 900))],
    "ward_bay": [("bed", (900, 2100)), ("locker", (500, 500))],
    "office": [("desk", (1600, 800))],
    "reception": [("desk", (1800, 800))],
}


def add_clutter_decoys(plan: Plan, level, bb, rng):
    minx, miny, maxx, maxy = bb
    if level in ("light", "medium", "heavy"):
        # dimension chains along bottom + left
        off = miny - rng.randint(400, 900)
        ticks = list(range(minx, maxx, max(1000, (maxx - minx) // 6)))
        plan.decoy("dimension_chain",
                   {"line": [[minx, off], [maxx, off]], "tier": 1,
                    "ticks": ticks})
        if rng.random() < 0.3:
            offx = minx - rng.randint(400, 900)
            plan.decoy("dimension_chain",
                       {"line": [[offx, miny], [offx, maxy]], "tier": 1,
                        "ticks": list(range(miny, maxy,
                                            max(1000, (maxy - miny) // 6)))})
    if level == "heavy":
        plan.decoy("title_block", {"rect": [maxx - 4000, miny - 2600,
                                            maxx, miny - 200]})
        plan.decoy("sheet_border", {"rect": [minx - 1500, miny - 3000,
                                             maxx + 1500, maxy + 1500]})
        for _ in range(rng.randint(1, 4)):
            wx = rng.randint(minx, max(minx + 1, maxx - 1500))
            wy = rng.randint(miny, max(miny + 1, maxy - 1500))
            plan.decoy("revision_cloud",
                       {"around": [wx, wy, wx + 1400, wy + 1000],
                        "style": "scalloped", "rev": f"P{rng.randint(1, 4)}"})
        for _ in range(rng.randint(3, 8)):
            bx = rng.randint(minx, maxx)
            by = rng.randint(miny, maxy)
            plan.decoy("keynote", {"bubble": [bx, by, 250],
                                   "leader_to": [bx + 300, by + 300],
                                   "text": f"K{rng.randint(1, 30):02d}"})
        for _ in range(rng.randint(2, 8)):     # noise lines
            sx = rng.randint(minx, maxx); sy = rng.randint(miny, maxy)
            ln = rng.randint(20, 200); ang = rng.uniform(0, 360)
            ex = int(sx + ln * math.cos(math.radians(ang)))
            ey = int(sy + ln * math.sin(math.radians(ang)))
            plan.decoy("noise_line", {"line": [[sx, sy], [ex, ey]]})
        # duplicate-wall digitising artefact
        if plan.walls and rng.random() < 0.6:
            w = rng.choice([w for w in plan.walls if not w.arc] or plan.walls)
            d = rng.randint(2, 4)
            plan.decoy("xref_ghost",
                       {"line": [[w.centerline[0][0] + d, w.centerline[0][1] + d],
                                 [w.centerline[1][0] + d, w.centerline[1][1] + d]],
                        "note": "duplicate wall offset artefact"})


# ===========================================================================
# SECTION 11 -- per-plan orchestration
# ===========================================================================

def apply_arc_features(plan: Plan, feats: List[Feature], rng):
    """Splice arc features in: drop the replaced straight exterior wall, add
    the true arc wall, tag the nearest room as curved."""
    curved = []
    for f in feats:
        if f.kind != "arc":
            continue
        axis, coord, lo, hi = f.drop_edge
        plan.walls = [w for w in plan.walls
                      if not _wall_on_edge(w, axis, coord, lo, hi)]
        w = plan.arc_wall(f.center, f.radius, f.a0, f.a1, f.thickness,
                          wtype="exterior", material=f.material,
                          hatch=WALL_MATERIALS[f.material][2])
        curved.append(w)
        _tag_curved_room(plan, f)
    return curved


def _wall_on_edge(w: Wall, axis, coord, lo, hi):
    if w.arc:
        return False
    (x0, y0), (x1, y1) = w.centerline
    if axis == "v" and x0 == x1 == coord and min(y0, y1) >= lo - 1 \
            and max(y0, y1) <= hi + 1:
        return True
    if axis == "h" and y0 == y1 == coord and min(x0, x1) >= lo - 1 \
            and max(x0, x1) <= hi + 1:
        return True
    return False


def _tag_curved_room(plan: Plan, f: Feature):
    cx, cy = f.center
    best = None
    for r in plan.rooms:
        rxs = [p[0] for p in r.polygon]; rys = [p[1] for p in r.polygon]
        rcx = sum(rxs) / len(rxs); rcy = sum(rys) / len(rys)
        d = math.hypot(rcx - cx, rcy - cy)
        if best is None or d < best[0]:
            best = (d, r)
    if best:
        best[1].shape = "curved"


def _add_nonaxis_walls(plan: Plan, rng):
    """Add an exterior wall for every non-axis-aligned room edge (e.g. the
    diagonal of a clipped/pentagon corner). Deduplicated by edge."""
    seen = set()
    for r in plan.rooms:
        poly = r.polygon
        n = len(poly)
        for k in range(n):
            a = tuple(poly[k]); b = tuple(poly[(k + 1) % n])
            if a[0] == b[0] or a[1] == b[1]:
                continue                       # axis-aligned -> handled already
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            t = rng.choice([215, 250, 300])
            plan.wall(a, b, t, wtype="exterior", material="SOLID_BRICK",
                      hatch="ANSI31", angle_class="angled")


def assign_rooms(plan: Plan, cells_poly, btype, rng):
    types = ROOM_TYPES.get(btype, ["room"])
    # sort cells: smallest -> small rooms
    order = sorted(range(len(cells_poly)),
                   key=lambda i: _poly_area_pts(cells_poly[i]))
    small_pool = [t for t in types if t in SMALL_ROOMS] or types
    big_pool = [t for t in types if t not in SMALL_ROOMS] or types
    # always include a circulation room if many rooms
    names = {}
    used_master = False
    for rank, idx in enumerate(order):
        area = _poly_area_pts(cells_poly[idx])
        if rank < max(1, len(order) // 4):
            rt = rng.choice(small_pool)
        elif not used_master and "master_bedroom" in types and area > 12_000_000:
            rt = "master_bedroom"; used_master = True
        else:
            rt = rng.choice(big_pool)
        names[idx] = rt
    return names


def _poly_area_pts(poly):
    return abs(_poly_area(poly))


_STRICT_VALIDATOR = None
_STRICT_TRIED = False


def _strict_validator():
    """Lazily load the strict topological/semantic validator from
    ``validate_plan.py``.  Returns a ``validate_plan(config)->(ok,errs)``
    callable, or ``None`` if the module is unavailable (so this generator still
    runs standalone).  Imported lazily to avoid an import cycle."""
    global _STRICT_VALIDATOR, _STRICT_TRIED
    if not _STRICT_TRIED:
        _STRICT_TRIED = True
        try:
            from validate_plan import validate_plan as sv
            _STRICT_VALIDATOR = sv
        except Exception:
            _STRICT_VALIDATOR = None
    return _STRICT_VALIDATOR


def _is_valid(d: dict) -> bool:
    """Pass BOTH the fast schema/diversity check and the strict topological /
    semantic validator (watertight walls, every room a door, connected plan)."""
    if d is None or not validate_plan(d)[0]:
        return False
    strict = _strict_validator()
    return strict is None or strict(d)[0]


def generate_one(index: int, seed: int) -> Optional[dict]:
    """Generate a single valid plan dict, or None if it could not be made.

    The structural identity (building type, footprint shape, size, room count,
    region, curved-wall bucket) is fixed per index, so retries only reshuffle
    the interior layout / placement -- this keeps the dataset distributions
    exactly on target regardless of per-attempt validity.

    Each candidate must pass the strict validator (``validate_plan.py``):
    enforce-by-construction (no floating nib; spanning-tree doors) plus this
    reject-and-regenerate gate together guarantee a 100%-valid result.
    """
    sp = _structural_params(index, seed)
    for attempt in range(20):
        rng = random.Random((seed * 7919 + index * 131 + attempt * 101) & 0xFFFFFFFF)
        try:
            d = _build(index, rng, sp)
        except Exception:
            d = None
        if _is_valid(d):
            return d
    return _build_safe(index, seed, sp)


def _build_safe(index: int, seed: int, sp: dict) -> Optional[dict]:
    """Guaranteed fallback: a simple rectangle plan (full-width wall >8m, an
    L-merged non-rect room, modest room count) that always validates -- so
    every index yields a plan even on a pathological footprint draw."""
    sp = dict(sp)
    sp["shape"] = "rectangle"
    sp["n_arcs"] = 0
    sp["target"] = min(sp["target"], 5)
    sp["W"] = max(sp["W"], 8600)
    for attempt in range(15):
        rng = random.Random((seed * 99991 + index * 7 + attempt * 13) & 0xFFFFFFFF)
        try:
            d = _build(index, rng, sp)
        except Exception:
            d = None
        if _is_valid(d):
            return d
    return None


def _structural_params(index: int, seed: int) -> dict:
    rng = random.Random((seed * 1_000_003 + index * 97) & 0xFFFFFFFF)
    btype = wchoice(rng, BUILDING_TYPES)
    (wmin, wmax), (dmin, dmax) = BUILDING_DIMS[btype]
    W = snap(rng.uniform(wmin, wmax), 100)
    H = snap(rng.uniform(dmin, dmax), 100)
    ar = W / H
    if ar > 2.8:
        W = int(H * 2.8)
    elif ar < 0.4:
        W = int(H * 0.4)
    W = max(3000, snap(W, 100)); H = max(3000, snap(H, 100))
    rmin, rmax = ROOM_COUNTS[btype]
    target = rng.randint(rmin, rmax)
    # >3-room plans need a wall >8m. The base width W is the one full-span,
    # unbroken exterior wall in every footprint shape (rectangle / L / T / U /
    # curved), so clamp W -- not just max(W,H) which a T/U junction breaks up.
    if target > 3 and W < 8600:
        W = 8600 + rng.randint(0, 4000)
    W = snap(W, 100); H = snap(H, 100)
    region = wchoice(rng, REGIONS)
    cw_bucket = wchoice(rng, {"none": 55, "one": 25, "few": 15, "many": 5})
    shape, n_arcs = _shape_for_bucket(rng, cw_bucket)
    return {"btype": btype, "W": W, "H": H, "target": target,
            "region": region, "cw_bucket": cw_bucket, "shape": shape,
            "n_arcs": n_arcs}


def _shape_for_bucket(rng, bucket):
    """Map the curved-wall frequency bucket to a footprint shape + arc count,
    so curved-wall presence matches none 55 / 1 25 / 2-3 15 / 4+ 5."""
    if bucket == "none":
        sh = wchoice(rng, {"rectangle": 30, "L_shape": 18, "T_shape": 8,
                           "U_shape": 7, "irregular_polygon": 12,
                           "rectangle_with_bay": 8})
        return sh, 0
    if bucket == "one":
        return rng.choice(["rectangle_with_curved_end",
                           "fully_curved_facade"]), 1
    if bucket == "few":
        return rng.choice(["fully_curved_facade", "organic_multi_arc"]), \
            rng.randint(2, 3)
    return "organic_multi_arc", rng.randint(4, 6)


def _build(index: int, rng: random.Random, sp: dict) -> Optional[dict]:
    btype = sp["btype"]; W = sp["W"]; H = sp["H"]; target = sp["target"]
    shape = sp["shape"]; region = sp["region"]

    cells_poly, feats = make_footprint(shape, W, H, target, sp["n_arcs"], rng)
    cells_poly = [p for p in cells_poly
                  if (max(x for x, _ in p) - min(x for x, _ in p)) >= 600
                  and (max(y for _, y in p) - min(y for _, y in p)) >= 600]
    if not cells_poly:
        return None

    plan = Plan(f"plan_{index + 1:05d}", rng)
    plan.name = _plan_name(btype, shape, rng)
    rtypes = assign_rooms(plan, cells_poly, btype, rng)

    # guarantee at least one non-rectangular room
    has_nonrect = bool(feats) or any(
        Plan._classify(p) != "rectangle" for p in cells_poly)
    if not has_nonrect:
        cells_poly, rtypes = _ensure_nonrect(cells_poly, rtypes, rng)
    rooms_poly = cells_poly

    ext_mat = wchoice(rng, EXTERIOR_MATERIALS)
    ext_t = int(rng.uniform(*WALL_MATERIALS[ext_mat][:2]))
    ext_hatch = WALL_MATERIALS[ext_mat][2]
    office = btype in ("office_open_plan", "office_cellular")

    def interior_picker():
        if office and rng.random() < 0.4:
            m = wchoice(rng, OFFICE_PARTITIONS)
        else:
            m = wchoice(rng, INTERIOR_MATERIALS)
        t = int(rng.uniform(*WALL_MATERIALS[m][:2]))
        return m, t, WALL_MATERIALS[m][2], CLASS_TO_TYPE[WALL_MATERIALS[m][3]]

    for i, poly in enumerate(rooms_poly):
        rt = rtypes.get(i, "room")
        plan.room(poly, _room_label(rt), rt)

    adjacencies = build_walls_from_rooms(plan, rooms_poly, ext_mat, ext_t,
                                         ext_hatch, interior_picker)
    curved = apply_arc_features(plan, feats, rng)
    _add_nonaxis_walls(plan, rng)

    comps = place_doors(plan, adjacencies, rooms_poly, btype, region, rng)
    if len(rooms_poly) > 1 and comps > 1:
        return None                            # disconnected -> not functional
    place_exterior_openings(plan, btype, region, rng, curved)
    _ensure_constraints(plan, rng)

    bb = plan.bbox()
    ncol = add_columns(plan, btype, bb, rng)
    clutter = wchoice(rng, CLUTTER_LEVELS)
    furn = 0
    if clutter in ("medium", "heavy"):
        furn = add_furniture(plan, 0.6 if clutter == "medium" else 0.9, rng)
    add_clutter_decoys(plan, clutter, bb, rng)

    scale = wchoice(rng, SCALES); standard = wchoice(rng, STANDARDS)
    lw_style = wchoice(rng, LINEWEIGHT_STYLES); rot = _sample_rotation(rng)
    if abs(rot) > 0.01:
        plan.rotate(rot)
    # repair AFTER rotation so it acts on the exact geometry the validator sees
    # (rotation's int rounding can nudge a window across a wall junction).
    _repair_blocked_windows(plan)          # drop any window a wall crosses (S10)
    plan.render = {"scale": scale, "rotation_deg": round(rot, 2),
                   "line_weight": _lw_string(lw_style),
                   "line_weight_style": lw_style,
                   "degradation": _degradation(clutter, rng),
                   "dpi": int(rng.uniform(80, 220)), "standard": standard,
                   "monochrome": True}
    plan.clutter = _clutter_string(clutter, furn)
    cw_count = len([w for w in plan.walls if w.arc])
    plan.meta = {"schema_version": SCHEMA_VERSION,
                 "generator": "config_generator.py", "seed_index": index,
                 "building_type": btype, "footprint_shape": shape,
                 "region": region, "standard": standard, "scale": scale,
                 "clutter_level": clutter, "curved_wall_count": cw_count,
                 "complexity": _complexity(len(plan.rooms)),
                 "has_columns": ncol > 0, "has_furniture": furn > 0,
                 "imperial_sizing": region in ("us", "uk")}
    plan.description = (f"{btype.replace('_', ' ')}, "
                        f"{shape.replace('_', ' ')} footprint, "
                        f"{len(plan.rooms)} rooms.")
    plan.hard_case = _hard_case(clutter, cw_count, shape)
    return plan.to_dict()


def _ensure_nonrect(polys, types, rng):
    """Guarantee >=1 non-rectangular room: try merging two cells into an L;
    failing that, notch a building corner cell into an L."""
    m = _maybe_merge(polys, types, rng, force=True)
    if any(Plan._classify(p) != "rectangle" for p in m["polys"]):
        return m["polys"], m["types"]
    polys2 = [list(p) for p in m["polys"]]
    types2 = dict(m["types"])
    BX1 = max(p[0] for poly in polys2 for p in poly)
    BY1 = max(p[1] for poly in polys2 for p in poly)
    idx = None
    for k, poly in enumerate(polys2):
        if len(poly) == 4 and [BX1, BY1] in [list(pt) for pt in poly]:
            idx = k
            break
    if idx is None:
        idx = max(range(len(polys2)),
                  key=lambda k: _poly_area_pts(polys2[k])
                  if len(polys2[k]) == 4 else 0)
    poly = polys2[idx]
    if len(poly) == 4:
        x0, y0 = poly[0]; x1, y1 = poly[2]
        bw = snap(max(700, min(int(0.45 * (x1 - x0)), (x1 - x0) - 700)), 100)
        bh = snap(max(700, min(int(0.45 * (y1 - y0)), (y1 - y0) - 700)), 100)
        if 300 <= bw < (x1 - x0) and 300 <= bh < (y1 - y0):
            polys2[idx] = [[x0, y0], [x1, y0], [x1, y1 - bh],
                           [x1 - bw, y1 - bh], [x1 - bw, y1], [x0, y1]]
    return polys2, types2


def _maybe_merge(cells_poly, rtypes, rng, force):
    """Optionally merge two horizontally/vertically adjacent rectangles that
    share a partial edge into one L-shaped room (guarantees a non-rect room)."""
    polys = [list(p) for p in cells_poly]
    types = dict(rtypes)
    do = force or rng.random() < 0.4
    if not do or len(polys) < 2:
        return {"polys": polys, "types": types}
    # find a pair of rectangles sharing part of an edge but not aligned -> L
    rects = [(_bounds(p), i) for i, p in enumerate(polys)]
    rng.shuffle(rects)
    for (ax0, ay0, ax1, ay1), i in rects:
        for (bx0, by0, bx1, by1), j in rects:
            if i >= j:
                continue
            # share vertical edge ax1==bx0, overlapping y, different heights
            if ax1 == bx0 and min(ay1, by1) - max(ay0, by0) > 500 \
                    and (ay0 != by0 or ay1 != by1):
                uni = _rect_union_L((ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1))
                if uni:
                    return _do_merge(polys, types, i, j, uni)
            if ay1 == by0 and min(ax1, bx1) - max(ax0, bx0) > 500 \
                    and (ax0 != bx0 or ax1 != bx1):
                uni = _rect_union_L((ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1))
                if uni:
                    return _do_merge(polys, types, i, j, uni)
    return {"polys": polys, "types": types}


def _do_merge(polys, types, i, j, uni):
    keep = min(i, j)
    drop = max(i, j)
    newpolys = []
    newtypes = {}
    k = 0
    for idx in range(len(polys)):
        if idx == drop:
            continue
        if idx == keep:
            newpolys.append(uni)
        else:
            newpolys.append(polys[idx])
        newtypes[k] = types.get(idx, "room")
        k += 1
    return {"polys": newpolys, "types": newtypes}


def _bounds(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _rect_union_L(r1, r2):
    """Union of two edge-sharing rectangles as an ortho polygon (CCW)."""
    return _rectilinear_union([r1, r2])


def _rectilinear_union(rects):
    """Outline of a union of axis-aligned rectangles (handles partial edge
    overlaps).  Returns a CCW polygon of corner vertices, or None.

    Method: lay down all x/y grid lines, mark covered cells, then emit each
    covered cell's boundary edges oriented with the covered side on the LEFT
    (so the chained loop is CCW), and walk them head-to-tail.
    """
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
    nx, ny = len(xs) - 1, len(ys) - 1
    if nx < 1 or ny < 1:
        return None
    cov = [[False] * ny for _ in range(nx)]
    for i in range(nx):
        cxv = (xs[i] + xs[i + 1]) / 2
        for j in range(ny):
            cyv = (ys[j] + ys[j + 1]) / 2
            for (x0, y0, x1, y1) in rects:
                if x0 < cxv < x1 and y0 < cyv < y1:
                    cov[i][j] = True
                    break

    def covered(i, j):
        return 0 <= i < nx and 0 <= j < ny and cov[i][j]

    segs = []
    for i in range(nx):
        for j in range(ny):
            if not cov[i][j]:
                continue
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            if not covered(i - 1, j):
                segs.append(((x0, y1), (x0, y0)))   # left edge, going down
            if not covered(i + 1, j):
                segs.append(((x1, y0), (x1, y1)))   # right edge, going up
            if not covered(i, j - 1):
                segs.append(((x0, y0), (x1, y0)))   # bottom edge, going right
            if not covered(i, j + 1):
                segs.append(((x1, y1), (x0, y1)))   # top edge, going left
    if not segs:
        return None
    from collections import defaultdict
    nxt = defaultdict(list)
    for a, b in segs:
        nxt[a].append(b)
    start = segs[0][0]
    poly = [list(start)]
    cur = start
    guard = 0
    while guard < len(segs) + 5:
        guard += 1
        outs = nxt.get(cur)
        if not outs:
            break
        b = outs.pop()
        cur = b
        if cur == start:
            break
        poly.append(list(b))
    poly = _collapse_collinear(poly)
    if _poly_area(poly) < 0:
        poly = poly[::-1]
    return poly if len(poly) >= 4 else None


def _collapse_collinear(poly):
    if len(poly) < 3:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[(i - 1) % n]; b = poly[i]; c = poly[(i + 1) % n]
        # keep b unless a,b,c collinear
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) != 0:
            out.append(b)
    return out if len(out) >= 4 else poly


def _repair_blocked_windows(plan: Plan):
    """Remove any window / cased opening that another wall crosses (S10).

    Used on curved plans, where arc glazing is placed by angle and an interior
    wall meeting the facade can clip a window.  Doors are left untouched -- a
    blocked door is rare and is handled by the regenerate gate (removing it
    could break connectivity).  Reuses the validator's S10 logic so the
    generator and validator never disagree."""
    try:
        from validate_plan import _Geo, check_S10
    except Exception:
        return
    import re
    errs = check_S10(_Geo(plan.to_dict()))
    if not errs:
        return
    blocked = set()
    for e in errs:
        m = re.search(r"crosses opening (\w+)", e)
        if m:
            blocked.add(m.group(1))
    if blocked:
        plan.openings = [o for o in plan.openings
                         if not (o.id in blocked and o.category != "door")]


def _ensure_constraints(plan: Plan, rng):
    """Guarantee >=3 distinct wall thicknesses without rejecting.

    NOTE: a free-floating short interior "nib" used to be injected here to
    satisfy a legacy "wall < 1.5 m" diversity heuristic.  That nib is a
    topologically floating wall (a dangling free end) and violates T2 in
    ``validate_plan.py``, so it has been removed -- the generator now emits
    zero floating walls.  Short walls still occur naturally (partitions,
    closets), so wall-length diversity is preserved."""
    # thickness diversity
    thicks = {w.thickness_mm for w in plan.walls}
    if len(thicks) < 3 and plan.walls:
        extra = [t for t in (90, 150, 215, 250, 300) if t not in thicks]
        for w in plan.walls[:3]:
            if len(thicks) >= 3:
                break
            if extra:
                w.thickness_mm = extra.pop()
                # rebuild band polygon for new thickness
                if not w.arc:
                    (a, b) = w.centerline
                    d = _unit(_sub(b, a)); nn = _norm(d)
                    off = _mul(nn, w.thickness_mm / 2)
                    w.polygon = [_i(_add(a, off)), _i(_add(b, off)),
                                 _i(_sub(b, off)), _i(_sub(a, off))]
                thicks = {x.thickness_mm for x in plan.walls}


# ===========================================================================
# SECTION 12 -- naming / string helpers
# ===========================================================================

def _plan_name(btype, shape, rng):
    base = btype.replace("_", " ").title()
    tag = {"rectangle": "", "L_shape": " (L-plan)", "T_shape": " (T-plan)",
           "U_shape": " (U-plan)", "irregular_polygon": " (irregular)",
           "rectangle_with_bay": " (bay)",
           "rectangle_with_curved_end": " (curved end)",
           "fully_curved_facade": " (curved facade)",
           "organic_multi_arc": " (organic)"}.get(shape, "")
    return base + tag


def _room_label(rt):
    return rt.replace("_", " ").title()


def _complexity(nrooms):
    if nrooms <= 4:
        return "simple"
    if nrooms <= 20:
        return "moderate"
    if nrooms <= 100:
        return "complex"
    return "very_complex"


def _sample_rotation(rng):
    b = wchoice(rng, ROTATION_BUCKET)
    if b == "zero":
        return 0.0
    if b == "small":
        return round(rng.uniform(1, 5), 2)
    if b == "medium":
        return round(rng.uniform(5, 45), 2)
    return round(rng.uniform(45, 89), 2)


def _lw_string(style):
    return {"standard_layered": "layered standard",
            "light_all": "light uniform", "heavy_all": "heavy uniform",
            "hairline_all": "hairline", "mixed_random": "mixed random"}[style]


def _degradation(clutter, rng):
    if clutter == "heavy" and rng.random() < 0.5:
        return rng.choice(["low-contrast scan; speckle; line dropout",
                           "ghost half-tone xref underlay",
                           "hand-drawn wobble; broken lines"])
    return "none"


def _clutter_string(level, furn):
    return {"none": "None.", "light": "Light: dims + room labels.",
            "medium": f"Moderate: {furn} furniture items + dimension chains.",
            "heavy": f"Heavy: {furn} furniture items, title block, "
                     f"revision clouds, keynotes, noise."}[level]


def _hard_case(clutter, cw, shape):
    bits = []
    if clutter == "heavy":
        bits.append("dense annotation clutter / hard negatives")
    if cw:
        bits.append(f"{cw} true-arc curved wall(s) with radial jambs")
    if shape in ("irregular_polygon", "organic_multi_arc"):
        bits.append("non-orthogonal geometry")
    return "; ".join(bits) or "clean orthogonal baseline"


# ===========================================================================
# SECTION 13 -- validation (mirrors validate_scenarios.py + brief)
# ===========================================================================

DOOR_ENUM = {"SINGLE_HINGED", "DOUBLE_HINGED", "SLIDING", "POCKET", "BIFOLD",
             "GARAGE", "FRENCH"}
WIN_ENUM = {"CASEMENT", "SLIDING", "FIXED", "BAY", "AWNING", "LOUVRE",
            "CORNER", "CLERESTORY"}
OPEN_ENUM = {"CASED", "GAP"}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_int_pt(p):
    return isinstance(p, list) and len(p) == 2 and all(isinstance(c, int) for c in p)


def validate_plan(p: dict) -> Tuple[bool, List[str]]:
    errs = []
    for k in ("id", "group", "name", "units", "origin", "bbox", "footprint",
              "walls", "rooms", "openings"):
        if k not in p:
            errs.append(f"missing key {k}")
    if p.get("units") != "mm":
        errs.append("units != mm")
    if not p.get("walls"):
        errs.append("no walls")
    if not p.get("rooms"):
        errs.append("no rooms")

    thicks, lengths = set(), []
    for w in p.get("walls", []):
        if w["thickness_mm"] <= 0:
            errs.append("non-positive thickness")
        thicks.add(w["thickness_mm"])
        lengths.append(w["length_mm"])
        if w["length_mm"] < MIN_WALL:
            errs.append(f"wall {w['id']} shorter than {MIN_WALL}")
        if len(w["polygon"]) < 4:
            errs.append("polygon <4 pts")
        if not all(_is_int_pt(pt) for pt in w["polygon"]):
            errs.append("non-integer polygon pt")
        if not w.get("arc"):
            cl = w["centerline"]
            if abs(_dist(cl[0], cl[1]) - w["length_mm"]) > 2:
                errs.append(f"wall {w['id']} length != |centerline|")
    if len(thicks) < 3:
        errs.append(f"<3 wall thicknesses {sorted(thicks)}")

    nonrect = sum(1 for r in p.get("rooms", []) if r["shape"] != "rectangle")
    if nonrect < 1:
        errs.append("no non-rectangular room")
    for r in p.get("rooms", []):
        if len(r["polygon"]) < 3:
            errs.append("room polygon <3 pts")
        if not all(_is_int_pt(pt) for pt in r["polygon"]):
            errs.append("non-integer room pt")

    if len(p.get("rooms", [])) > 3 and lengths:
        if max(lengths) <= 8000:
            errs.append(f"no wall >8m (max {max(lengths)})")
        # NOTE: the legacy "must have a wall < 1.5 m" rule was dropped -- it had
        # forced a floating nib wall (see _ensure_constraints).  Short walls are
        # still common naturally; we no longer fabricate one.

    # openings
    occ_by_wall = {}
    for o in p.get("openings", []):
        cat, st = o["category"], o["subtype"]
        if cat == "door" and st not in DOOR_ENUM:
            errs.append(f"bad door subtype {st}")
        if cat == "window" and st not in WIN_ENUM:
            errs.append(f"bad window subtype {st}")
        if cat == "opening" and st not in OPEN_ENUM:
            errs.append(f"bad opening subtype {st}")
        if not (_is_int_pt(o["p1"]) and _is_int_pt(o["p2"])
                and _is_int_pt(o["center"])):
            errs.append("non-integer endpoint")
            continue
        if abs(_dist(o["p1"], o["p2"]) - o["width_mm"]) > 3:
            errs.append(f"|p1-p2| != width for {o['id']}")
        mid = [(o["p1"][0] + o["p2"][0]) / 2, (o["p1"][1] + o["p2"][1]) / 2]
        if _dist(mid, o["center"]) > 2:
            errs.append(f"center not midpoint for {o['id']}")

    return (len(errs) == 0), errs


# ===========================================================================
# SECTION 14 -- report aggregation
# ===========================================================================

def new_report():
    return {
        "total_generated": 0, "total_rejected": 0,
        "rejection_reasons": {},
        "building_type_distribution": {},
        "footprint_shape_distribution": {},
        "wall_material_distribution": {},
        "curved_wall_count_distribution": {},
        "door_subtype_distribution": {},
        "window_subtype_distribution": {},
        "clutter_level_distribution": {},
        "scale_distribution": {}, "standard_distribution": {},
        "region_distribution": {}, "complexity_distribution": {},
        "_walls": 0, "_doors": 0, "_windows": 0,
        "_with_columns": 0, "_with_furniture": 0, "_with_curved": 0,
    }


def _bump(d, k):
    d[k] = d.get(k, 0) + 1


def tally(rep, cfg):
    rep["total_generated"] += 1
    m = cfg.get("metadata", {})
    _bump(rep["building_type_distribution"], m.get("building_type", "?"))
    _bump(rep["footprint_shape_distribution"], m.get("footprint_shape", "?"))
    _bump(rep["curved_wall_count_distribution"], str(m.get("curved_wall_count", 0)))
    _bump(rep["clutter_level_distribution"], m.get("clutter_level", "?"))
    _bump(rep["scale_distribution"], m.get("scale", "?"))
    _bump(rep["standard_distribution"], m.get("standard", "?"))
    _bump(rep["region_distribution"], m.get("region", "?"))
    _bump(rep["complexity_distribution"], m.get("complexity", "?"))
    for w in cfg["walls"]:
        if w.get("material"):
            _bump(rep["wall_material_distribution"], w["material"])
    rep["_walls"] += cfg["counts"]["walls"]
    rep["_doors"] += cfg["counts"]["doors"]
    rep["_windows"] += cfg["counts"]["windows"]
    for o in cfg["openings"]:
        if o["category"] == "door":
            _bump(rep["door_subtype_distribution"], o["subtype"])
        elif o["category"] == "window":
            _bump(rep["window_subtype_distribution"], o["subtype"])
    if m.get("has_columns"):
        rep["_with_columns"] += 1
    if m.get("has_furniture"):
        rep["_with_furniture"] += 1
    if m.get("curved_wall_count", 0) > 0:
        rep["_with_curved"] += 1


def finalize_report(rep):
    n = max(1, rep["total_generated"])
    rep["mean_walls_per_plan"] = round(rep.pop("_walls") / n, 2)
    rep["mean_doors_per_plan"] = round(rep.pop("_doors") / n, 2)
    rep["mean_windows_per_plan"] = round(rep.pop("_windows") / n, 2)
    rep["plans_with_columns_pct"] = round(100 * rep.pop("_with_columns") / n, 1)
    rep["plans_with_furniture_pct"] = round(100 * rep.pop("_with_furniture") / n, 1)
    rep["plans_with_curved_walls_pct"] = round(100 * rep.pop("_with_curved") / n, 1)
    return rep


# ===========================================================================
# SECTION 15 -- main / CLI
# ===========================================================================

def _index_entry(cfg):
    return {"id": cfg["id"], "file": f"{cfg['id']}.json",
            "building_type": cfg["metadata"]["building_type"],
            "footprint_shape": cfg["metadata"]["footprint_shape"],
            "footprint": cfg["footprint"], "counts": cfg["counts"]}


def _worker_chunk(task):
    """Generate + write one contiguous chunk [lo,hi); return a compact partial
    report and index entries.  Heavy data stays in the worker (it writes its own
    per-plan files and one render-batch), so only small summaries cross the pipe.
    Deterministic: each plan depends solely on (index, seed)."""
    lo, hi, seed, out, indent, no_per_plan, shard_no, batch_dir = task
    rep = new_report()
    idx = []
    shard = []
    for i in range(lo, hi):
        cfg = generate_one(i, seed)
        if cfg is None:
            rep["total_rejected"] += 1
            _bump(rep["rejection_reasons"], "exceeded_15_attempts")
            continue
        tally(rep, cfg)
        if not no_per_plan:
            with open(os.path.join(out, f"{cfg['id']}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=indent)
        idx.append(_index_entry(cfg))
        if shard_no is not None:
            shard.append(cfg)
    if shard_no is not None and shard:
        _write_shard(batch_dir, shard_no, shard)
    return {"rep": rep, "idx": idx}


def _merge_report(master, part):
    for k, v in part.items():
        if isinstance(v, dict):
            d = master.setdefault(k, {})
            for kk, vv in v.items():
                d[kk] = d.get(kk, 0) + vv
        elif isinstance(v, (int, float)):
            master[k] = master.get(k, 0) + v


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Procedural floor-plan config generator (JSON only).")
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--output", default="./configs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0,
                    help="first structural index (for sharding across runs: "
                         "shard k uses --start k*count). plan ids and batch "
                         "files stay globally unique so shards can be merged.")
    ap.add_argument("--shard-size", type=int, default=1000,
                    help="scenarios per render-batch array (0 = no batches)")
    ap.add_argument("--indent", type=int, default=0,
                    help="JSON indent for per-plan files (0 = compact)")
    ap.add_argument("--no-per-plan", action="store_true",
                    help="skip writing individual plan_*.json files")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes (0 = auto, 1 = serial)")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    batch_dir = os.path.join(out, "render_batches")
    if args.shard_size > 0:
        os.makedirs(batch_dir, exist_ok=True)
    indent = args.indent if args.indent > 0 else None
    workers = args.workers if args.workers > 0 else min(8, os.cpu_count() or 1)
    t0 = time.time()

    # build tasks: one per render-batch shard (so shards align with chunks)
    step = args.shard_size if args.shard_size > 0 else \
        max(1, math.ceil(args.count / max(1, workers)))
    tasks = []
    end = args.start + args.count
    for lo in range(args.start, end, step):
        hi = min(lo + step, end)
        # base the shard number on the absolute index so batch_*.json names are
        # globally unique across shards (mergeable without collision)
        sn = (lo // step) if args.shard_size > 0 else None
        tasks.append((lo, hi, args.seed, out, indent, args.no_per_plan, sn,
                      batch_dir))

    master = new_report()
    all_idx = []
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_worker_chunk, tasks):
                _merge_report(master, res["rep"])
                all_idx.extend(res["idx"])
    else:
        for task in tasks:
            res = _worker_chunk(task)
            _merge_report(master, res["rep"])
            all_idx.extend(res["idx"])

    written = len(all_idx)
    all_idx.sort(key=lambda e: e["id"])
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"count": written, "seed": args.seed,
                   "scenarios": all_idx}, fh)

    finalize_report(master)
    master["seed"] = args.seed
    master["requested"] = args.count
    master["workers"] = workers
    master["elapsed_sec"] = round(time.time() - t0, 2)
    with open(os.path.join(out, "generation_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)

    print(f"Generated {written}/{args.count} configs in "
          f"{master['elapsed_sec']}s using {workers} worker(s) -> {out}")
    print(f"Rejected: {master['total_rejected']}  "
          f"({master['rejection_reasons']})")
    return master


def _write_shard(batch_dir, no, shard):
    fn = os.path.join(batch_dir, f"batch_{no:04d}.json")
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(shard, fh)


if __name__ == "__main__":
    main()
