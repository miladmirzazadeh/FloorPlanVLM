#!/usr/bin/env python3
"""Strict topological + semantic validator for synthetic floor-plan configs.

A *config* is one ``configs/plan_XXXXX.json`` object (schema
``synthdata-absolute-geometry/1.1``).  ``validate_plan(config)`` returns
``(is_valid, violations)`` where ``violations`` is a list of human-readable
strings, each prefixed by the constraint id it failed (``T1``..``T8``,
``S1``..``S10``, ``C1``..``C5``) so failures are diagnosable.  ``S10`` is the
"no wall crosses an opening" rule (a transverse wall through a doorway).

Geometry source of truth
------------------------
Walls in this schema are *derived from room edges* (axis-aligned walls from
``build_walls_from_rooms``, diagonal walls from ``_add_nonaxis_walls``, curved
walls from ``apply_arc_features``).  Therefore the **room polygons** are the
primary topological truth and the **wall centerlines must cover every room
edge** -- a missing wall shows up as an uncovered room edge (an "unclosed wall
loop"), and a room with no door shows up in the door-incidence graph.

Usage
-----
    python validate_plan.py configs/plan_00001.json        # single -> PASS / list
    python validate_plan.py configs/                       # baseline histogram
    python validate_plan.py configs/ --limit 2000 --json report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

# Reuse the engine's grid / enums where available (keeps the validator in lock
# step with the generator).  Falls back to literals so the file is standalone.
try:
    from config_generator import (DOOR_ENUM, GRID, MIN_WALL, OPEN_ENUM,
                                   SCHEMA_VERSION, WIN_ENUM)
except Exception:                                            # pragma: no cover
    GRID = 50
    MIN_WALL = 300
    SCHEMA_VERSION = "synthdata-absolute-geometry/1.1"
    DOOR_ENUM = {"SINGLE_HINGED", "DOUBLE_HINGED", "SLIDING", "POCKET",
                 "BIFOLD", "GARAGE", "FRENCH"}
    WIN_ENUM = {"CASEMENT", "SLIDING", "FIXED", "BAY", "AWNING", "LOUVRE",
                "CORNER", "CLERESTORY"}
    OPEN_ENUM = {"CASED", "GAP"}

# ---------------------------------------------------------------------------
# Tolerances (mm / mm^2).  Coordinates are integer mm snapped to GRID, so
# coincident junctions are exact up to rotation rounding (~1-2 mm).
# ---------------------------------------------------------------------------
TOL_PT = 6.0           # two points "coincide" within this (mm)
TOL_COVER = 30.0       # half-width of the buffer used for edge<->wall coverage
TOL_RESIDUAL = 80.0    # uncovered room-edge length that counts as a real gap
TOL_ON_WALL = 40.0     # perpendicular dist: opening endpoint -> host centerline
TOL_INCIDENT = 45.0    # dist: opening footprint -> a room boundary (incidence)
TOL_EXTERIOR = 60.0    # dist: opening center -> outer shell (entrance/facade)
TOL_JAMB = 30.0        # a wall within this of an opening end is a jamb, not a cross
TOL_AREA = 2.0e4       # 0.02 m^2 -- overlap / hole area below this is rounding
MIN_ROOM_AREA = 2.0e6  # 2 m^2 hard floor (engine min observed ~3.1 m^2)
MAX_ASPECT = 16.0      # extreme sliver guard (engine corridors reach ~18)
MIN_SIDE = 450.0       # narrowest room dimension before it is a sliver
THICK_MIN, THICK_MAX = 40, 600        # realistic wall thickness band
DOOR_W_MIN, DOOR_W_MAX = 550, 6000    # door incl. accessible/garage/commercial
WIN_W_MIN, WIN_W_MAX = 250, 6000

ACCESS_CATS = ("door", "opening")     # categories that grant room access


# ===========================================================================
# Geometry helpers
# ===========================================================================
def _poly(pts) -> Optional[Polygon]:
    """Shapely polygon from a point list; repair minor self-touch via buffer0."""
    if not pts or len(pts) < 3:
        return None
    try:
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            return None
        return p
    except Exception:
        return None


def _arc_polyline(arc: dict, step_deg: float = 4.0) -> Optional[LineString]:
    """Densify an arc dict into a LineString tracing its centerline."""
    try:
        cx, cy = arc["center"]
        r = float(arc["radius"])
        a0, a1 = float(arc["a0"]), float(arc["a1"])
    except Exception:
        return None
    if r <= 0:
        return None
    sweep = a1 - a0
    n = max(2, int(abs(sweep) / step_deg) + 1)
    pts = []
    for i in range(n + 1):
        a = math.radians(a0 + sweep * i / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return LineString(pts)


def _chord_line(w: dict) -> Optional[LineString]:
    """The straight chord of a wall (centerline endpoints)."""
    cl = w.get("centerline")
    if not cl or len(cl) < 2:
        return None
    try:
        ln = LineString([tuple(cl[0]), tuple(cl[1])])
        return ln if ln.length > 1e-6 else None
    except Exception:
        return None


def _wall_geoms(w: dict) -> List[LineString]:
    """All centerline geometry of a wall used for coverage / host tests.

    A curved wall keeps BOTH its straight chord (which coincides with a
    rectangular room's edge) and the densified arc (which coincides with a
    curved room's edge), so either kind of room edge gets covered.
    """
    geoms: List[LineString] = []
    chord = _chord_line(w)
    if chord is not None:
        geoms.append(chord)
    if w.get("arc"):
        arc = _arc_polyline(w["arc"])
        if arc is not None and arc.length > 1e-6:
            geoms.append(arc)
    return geoms


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _room_edges(poly: List[List[float]]):
    """Yield consecutive boundary edges as (a, b) point tuples."""
    n = len(poly)
    for i in range(n):
        a = tuple(poly[i])
        b = tuple(poly[(i + 1) % n])
        if a != b:
            yield a, b


def _aspect_and_minside(poly: Polygon) -> Tuple[float, float]:
    """Aspect ratio and shortest side of the minimum rotated bounding box."""
    try:
        mrr = poly.minimum_rotated_rectangle
        cs = list(mrr.exterior.coords)
        e1 = _dist(cs[0], cs[1])
        e2 = _dist(cs[1], cs[2])
        lo, hi = min(e1, e2), max(e1, e2)
        return (hi / lo if lo > 0 else 999.0), lo
    except Exception:
        return 1.0, 999.0


# ===========================================================================
# Shared derived geometry (built once per plan, passed to every check)
# ===========================================================================
class _Geo:
    def __init__(self, config: dict):
        self.config = config
        self.walls = config.get("walls", []) or []
        self.rooms = config.get("rooms", []) or []
        self.openings = config.get("openings", []) or []

        self.room_polys: List[Optional[Polygon]] = [
            _poly(r.get("polygon")) for r in self.rooms]
        # per-wall list of centerline geometries (chord [+ arc]); and the
        # straight chord alone (used for endpoint / mid-span-crossing tests)
        self.wall_geoms: List[List[LineString]] = [
            _wall_geoms(w) for w in self.walls]
        self.wall_chords: List[Optional[LineString]] = [
            _chord_line(w) for w in self.walls]

        good = [p for p in self.room_polys if p is not None]
        self.union = unary_union(good) if good else None
        # the watertight outer shell (largest polygon's exterior), as a ring
        self.union_boundary = None
        if self.union is not None:
            geom = self.union
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda g: g.area)
            self.union_boundary = geom.exterior

        # one merged line layer of every wall centerline, for coverage tests
        wl = [ln for geoms in self.wall_geoms for ln in geoms]
        self.wall_union = unary_union(wl) if wl else None
        self.wall_cover = (self.wall_union.buffer(TOL_COVER)
                           if self.wall_union is not None else None)

    # -- opening -> host wall / incident rooms -----------------------------
    def host_wall(self, o: dict) -> Optional[int]:
        """Index of the wall an opening sits on.

        Scored by how close BOTH endpoints p1/p2 are to the wall geometry --
        endpoints always lie *on* the wall (the center of an arc opening sits
        ~one sagitta inside the arc, so center distance alone is unreliable).
        Using both endpoints also rejects a perpendicular wall that happens to
        pass near one end.
        """
        p1, p2 = o.get("p1"), o.get("p2")
        if not (p1 and p2):
            c = o.get("center")
            if not c:
                return None
            p1 = p2 = c
        q1, q2 = Point(p1), Point(p2)
        best, bestd = None, TOL_ON_WALL
        for i, geoms in enumerate(self.wall_geoms):
            if not geoms:
                continue
            d = min(max(ln.distance(q1), ln.distance(q2)) for ln in geoms)
            if d <= bestd:
                bestd, best = d, i
        return best

    def _opening_geom(self, o: dict):
        """The opening as a LineString p1->p2 (its footprint on the wall)."""
        p1, p2 = o.get("p1"), o.get("p2")
        if p1 and p2 and p1 != p2:
            try:
                return LineString([tuple(p1), tuple(p2)])
            except Exception:
                pass
        c = o.get("center")
        return Point(c) if c else None

    def incident_rooms(self, o: dict) -> List[int]:
        """Rooms whose boundary contains the opening's footprint (within a tight
        tolerance) -- 2 for an interior door, 1 for an exterior one."""
        geom = self._opening_geom(o)
        if geom is None:
            return []
        hits = []
        for i, p in enumerate(self.room_polys):
            if p is None:
                continue
            d = p.exterior.distance(geom)
            if d <= TOL_INCIDENT:
                hits.append((d, i))
        hits.sort()
        return [i for _, i in hits]

    def door_rooms(self, o: dict, wi: Optional[int]) -> List[int]:
        """Rooms an interior opening actually joins: the nearest room on EACH
        side of the host wall.  Near a T-junction three rooms fall within
        tolerance, but a door joins exactly the two across the wall line -- so
        nearest-2 is wrong; side-of-wall is right."""
        inc = self.incident_rooms(o)
        if wi is None or len(inc) <= 2:
            return inc[:2]
        ch = self.wall_chords[wi]
        c = o.get("center")
        if ch is None or c is None:
            return inc[:2]
        (ax, ay), (bx, by) = ch.coords[0], ch.coords[-1]
        nx_, ny_ = -(by - ay), (bx - ax)        # wall normal
        pos = neg = None
        for ri in inc:                          # inc is nearest-first
            rp = self.room_polys[ri].representative_point()
            s = (rp.x - c[0]) * nx_ + (rp.y - c[1]) * ny_
            if s >= 0 and pos is None:
                pos = ri
            elif s < 0 and neg is None:
                neg = ri
        return [r for r in (pos, neg) if r is not None] or inc[:2]

    def on_exterior(self, o: dict) -> bool:
        """True if the opening sits on the outer shell (an entrance/facade).

        Measured at the center so that an *interior* door whose wall merely
        terminates on the exterior boundary (segment endpoint touching the
        shell) is not misread as exterior."""
        c = o.get("center")
        if not c or self.union_boundary is None:
            return False
        return self.union_boundary.distance(Point(c)) <= TOL_EXTERIOR


# ===========================================================================
# Topological predicates
# ===========================================================================
def check_T1(g: _Geo) -> List[str]:
    """Watertight exterior: the rooms form ONE connected region (a single
    closed outer shell), so there is no second detached cluster / leak.

    Uses a room-adjacency graph (rooms within TOL_PT share an edge) rather than
    the raw ``unary_union`` topology, which is brittle to the ~1mm coordinate
    rounding introduced by plan rotation."""
    polys = [(i, p) for i, p in enumerate(g.room_polys) if p is not None]
    if not polys:
        return ["T1: no valid room polygons to form an exterior"]
    if len(polys) == 1:
        return []
    G = nx.Graph()
    G.add_nodes_from(i for i, _ in polys)
    for a in range(len(polys)):
        ia, pa = polys[a]
        for b in range(a + 1, len(polys)):
            ib, pb = polys[b]
            if pa.distance(pb) <= TOL_PT:
                G.add_edge(ia, ib)
    ncomp = nx.number_connected_components(G)
    if ncomp > 1:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        stray = sum(len(c) for c in comps[1:])
        return [f"T1: exterior is not a single closed loop "
                f"({ncomp} disconnected room clusters; {stray} stray rooms)"]
    return []


def check_T2(g: _Geo) -> List[str]:
    """No floating walls: every wall endpoint coincides with another wall
    endpoint, lies on another wall (T-junction), or on the exterior boundary."""
    errs = []
    # endpoints are the straight-chord endpoints (== arc endpoints for curved
    # walls), which is where junctions to other walls actually occur
    endpoints = []
    idx = []
    for wi, ch in enumerate(g.wall_chords):
        if ch is None:
            continue
        cs = list(ch.coords)
        endpoints.append((cs[0], cs[-1]))
        idx.append(wi)
    for k, (a, b) in enumerate(endpoints):
        wi = idx[k]
        for end in (a, b):
            ep = Point(end)
            # (a) shares with another wall endpoint?
            shared = False
            for kk, (c, d) in enumerate(endpoints):
                if kk == k:
                    continue
                if _dist(end, c) <= TOL_PT or _dist(end, d) <= TOL_PT:
                    shared = True
                    break
            if shared:
                continue
            # (b) lies on another wall's span (T-junction)?
            on_other = False
            for wj, geoms in enumerate(g.wall_geoms):
                if wj == wi:
                    continue
                if any(ln.distance(ep) <= TOL_PT for ln in geoms):
                    on_other = True
                    break
            if on_other:
                continue
            # (c) lies on the exterior boundary?
            if g.union_boundary is not None and \
                    g.union_boundary.distance(ep) <= max(TOL_PT, TOL_COVER):
                continue
            wid = g.walls[wi].get("id", f"#{wi}")  # noqa: E501
            errs.append(f"T2: wall {wid} has a floating free end at "
                        f"({int(end[0])},{int(end[1])})")
    return errs


def check_T3(g: _Geo) -> List[str]:
    """Closed room loops: every room edge is covered by a wall centerline, so
    each room is a closed face of the wall network (no unclosed regions)."""
    if g.wall_cover is None:
        return ["T3: no wall geometry"]
    errs = []
    for ri, (room, poly) in enumerate(zip(g.rooms, g.room_polys)):
        if poly is None:
            continue
        uncovered = 0.0
        for a, b in _room_edges(room.get("polygon", [])):
            seg = LineString([a, b])
            if seg.length < 1:
                continue
            residual = seg.difference(g.wall_cover)
            uncovered = max(uncovered, residual.length)
        if uncovered > TOL_RESIDUAL:
            errs.append(f"T3: room {room.get('id', ri)} has an unclosed edge "
                        f"({uncovered:.0f}mm not covered by any wall)")
    return errs


def check_T4(g: _Geo) -> List[str]:
    """Gap-free, overlap-free partition: rooms tile one region with no internal
    voids and no overlaps."""
    errs = []
    polys = [p for p in g.room_polys if p is not None]
    if not polys or g.union is None:
        return errs
    sum_a = sum(p.area for p in polys)
    overlap = sum_a - g.union.area
    if overlap > TOL_AREA:
        errs.append(f"T4: rooms overlap (~{overlap / 1e6:.2f} m^2 double-counted)")
    geoms = (g.union.geoms if g.union.geom_type == "MultiPolygon"
             else [g.union])
    holes = 0.0
    for geom in geoms:
        for ring in geom.interiors:
            holes += Polygon(ring).area
    if holes > TOL_AREA:
        errs.append(f"T4: interior void(s) between rooms (~{holes / 1e6:.2f} m^2)")
    return errs


def check_T5(g: _Geo) -> List[str]:
    """No degenerate walls: zero/short length, length < thickness, duplicates."""
    errs = []
    seen = []
    for w in g.walls:
        cl = w.get("centerline")
        L = w.get("length_mm", 0)
        if not w.get("arc"):
            if L < MIN_WALL:
                errs.append(f"T5: wall {w.get('id')} shorter than MIN_WALL "
                            f"({L}mm)")
            if L < w.get("thickness_mm", 0):
                errs.append(f"T5: wall {w.get('id')} length {L} < thickness "
                            f"{w.get('thickness_mm')}")
        if cl and len(cl) >= 2:
            key = tuple(sorted([tuple(cl[0]), tuple(cl[1])]))
            for s in seen:
                if _dist(s[0], key[0]) <= TOL_PT and _dist(s[1], key[1]) <= TOL_PT:
                    errs.append(f"T5: wall {w.get('id')} duplicates another wall")
                    break
            seen.append(key)
    return errs


def check_T6(g: _Geo) -> List[str]:
    """Clean junctions: no two straight walls cross mid-span without a node."""
    errs = []
    lines = []
    for i, w in enumerate(g.walls):
        if w.get("arc"):
            continue
        ln = g.wall_chords[i]
        if ln is not None:
            lines.append((i, ln))
    for ai in range(len(lines)):
        i, la = lines[ai]
        for bi in range(ai + 1, len(lines)):
            j, lb = lines[bi]
            if not la.intersects(lb):
                continue
            inter = la.intersection(lb)
            if inter.geom_type != "Point":
                continue                         # collinear overlap handled elsewhere
            pt = (inter.x, inter.y)
            # a crossing is clean if the node is an endpoint of BOTH walls
            ea = [la.coords[0], la.coords[-1]]
            eb = [lb.coords[0], lb.coords[-1]]
            a_end = min(_dist(pt, e) for e in ea) <= TOL_PT
            b_end = min(_dist(pt, e) for e in eb) <= TOL_PT
            if not (a_end or b_end):
                errs.append(f"T6: walls {g.walls[i].get('id')} and "
                            f"{g.walls[j].get('id')} cross mid-span without a "
                            f"junction")
    return errs


def check_T7(g: _Geo) -> List[str]:
    """Valid arcs: consistent center/radius/angles with centerline endpoints."""
    errs = []
    for w in g.walls:
        arc = w.get("arc")
        if not arc:
            continue
        wid = w.get("id")
        try:
            cx, cy = arc["center"]
            r = float(arc["radius"])
            a0, a1 = float(arc["a0"]), float(arc["a1"])
        except Exception:
            errs.append(f"T7: wall {wid} arc missing center/radius/angles")
            continue
        if r <= 0:
            errs.append(f"T7: wall {wid} arc radius <= 0")
            continue
        cl = w.get("centerline")
        if cl and len(cl) >= 2:
            e0 = (cx + r * math.cos(math.radians(a0)),
                  cy + r * math.sin(math.radians(a0)))
            e1 = (cx + r * math.cos(math.radians(a1)),
                  cy + r * math.sin(math.radians(a1)))
            d_match = min(_dist(e0, cl[0]) + _dist(e1, cl[1]),
                          _dist(e0, cl[1]) + _dist(e1, cl[0]))
            if d_match > max(2 * TOL_COVER, 0.02 * r):
                errs.append(f"T7: wall {wid} arc endpoints disagree with "
                            f"centerline (off by {d_match:.0f}mm)")
        if len(w.get("polygon", [])) < 4:
            errs.append(f"T7: wall {wid} curved but polygon < 4 pts")
    return errs


def check_T8(g: _Geo) -> List[str]:
    """Simple room polygons: non-self-intersecting, positive area."""
    errs = []
    for ri, r in enumerate(g.rooms):
        pts = r.get("polygon")
        if not pts or len(pts) < 3:
            errs.append(f"T8: room {r.get('id', ri)} polygon < 3 pts")
            continue
        try:
            raw = Polygon(pts)
        except Exception:
            errs.append(f"T8: room {r.get('id', ri)} polygon unparseable")
            continue
        if raw.area <= 0:
            errs.append(f"T8: room {r.get('id', ri)} has zero area")
        if not raw.is_valid or not raw.exterior.is_simple:
            errs.append(f"T8: room {r.get('id', ri)} polygon self-intersects")
    return errs


# ===========================================================================
# Semantic predicates
# ===========================================================================
def _door_graph(g: _Geo) -> Tuple[nx.Graph, Dict[int, List[str]], bool]:
    """Graph over rooms (+ 'EXT'); accessibility openings are the edges.

    An opening's role is decided by its HOST WALL TYPE (the engine's own
    interior/exterior label), which is far more robust than boundary distance
    at re-entrant L/U corners: a door on an *exterior* wall is an entrance
    (room<->EXT); a door on an *interior* wall joins its two rooms.

    Returns (graph, room_index -> incident opening ids, has_entrance)."""
    G = nx.Graph()
    for i in range(len(g.rooms)):
        G.add_node(i)
    G.add_node("EXT")
    room_doors: Dict[int, List[str]] = defaultdict(list)
    has_entrance = False
    for o in g.openings:
        if o.get("category") not in ACCESS_CATS:
            continue
        oid = o.get("id")
        wi = g.host_wall(o)
        wtype = g.walls[wi].get("type") if wi is not None else None
        exterior = (wtype == "exterior") or (wi is None and g.on_exterior(o))
        if exterior:
            has_entrance = True
            rooms = g.incident_rooms(o)
            if rooms:
                G.add_edge(rooms[0], "EXT")
                room_doors[rooms[0]].append(oid)
        else:                                            # interior wall
            rooms = g.door_rooms(o, wi)
            if len(rooms) >= 2:
                G.add_edge(rooms[0], rooms[1])
                room_doors[rooms[0]].append(oid)
                room_doors[rooms[1]].append(oid)
            elif len(rooms) == 1:
                room_doors[rooms[0]].append(oid)
    return G, room_doors, has_entrance


def check_S1(g: _Geo) -> List[str]:
    """Every room has >= 1 door/opening on its boundary."""
    _, room_doors, _ = _door_graph(g)
    errs = []
    for ri, r in enumerate(g.rooms):
        if not room_doors.get(ri):
            errs.append(f"S1: room {r.get('id', ri)} has no door (sealed)")
    return errs


def check_S2(g: _Geo) -> List[str]:
    """Connected plan: every room reachable from an exterior entrance."""
    G, _, has_entrance = _door_graph(g)
    errs = []
    if not has_entrance:
        errs.append("S2: no exterior entrance door")
    room_nodes = [n for n in G.nodes if n != "EXT"]
    if not room_nodes:
        return errs
    if G.degree("EXT") > 0:
        # every room must be reachable from the entrance
        reach = nx.node_connected_component(G, "EXT")
        unreached = [n for n in room_nodes if n not in reach]
        if unreached:
            ids = ", ".join(str(g.rooms[n].get("id", n)) for n in unreached[:6])
            errs.append(f"S2: {len(unreached)} room(s) unreachable from "
                        f"entrance ({ids})")
    else:
        # no entrance edge resolved: at least require interior connectivity
        comps = nx.number_connected_components(G.subgraph(room_nodes))
        if comps > 1:
            errs.append(f"S2: plan splits into {comps} disconnected components")
    return errs


def check_S3(g: _Geo) -> List[str]:
    """Valid doors: on a wall, within span, connecting 2 rooms or room+EXT."""
    errs = []
    for o in g.openings:
        if o.get("category") != "door":
            continue
        oid = o.get("id")
        wi = g.host_wall(o)
        if wi is None:
            errs.append(f"S3: door {oid} does not lie on any wall")
            continue
        w = g.walls[wi]
        if not w.get("arc") and o.get("width_mm", 0) > w.get("length_mm", 0) + TOL_PT:
            errs.append(f"S3: door {oid} wider ({o.get('width_mm')}) than its "
                        f"wall {w.get('id')} ({w.get('length_mm')})")
        rooms = g.incident_rooms(o)
        if len(rooms) == 0 and not g.on_exterior(o):
            errs.append(f"S3: door {oid} is an orphan (touches no room)")
    return errs


def check_S4(g: _Geo) -> List[str]:
    """Valid windows: on a wall, within span, not overlapping a door."""
    errs = []
    per_wall_doors: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for o in g.openings:
        if o.get("category") == "door":
            wi = g.host_wall(o)
            if wi is not None:
                per_wall_doors[wi].append(_span_on_wall(g, o, wi))
    for o in g.openings:
        if o.get("category") != "window":
            continue
        oid = o.get("id")
        wi = g.host_wall(o)
        if wi is None:
            errs.append(f"S4: window {oid} does not lie on any wall")
            continue
        w = g.walls[wi]
        if not w.get("arc") and o.get("width_mm", 0) > w.get("length_mm", 0) + TOL_PT:
            errs.append(f"S4: window {oid} wider than its wall {w.get('id')}")
        s = _span_on_wall(g, o, wi)
        if s is not None:
            for ds in per_wall_doors.get(wi, []):
                if ds is not None and _overlap(s, ds) > TOL_PT:
                    errs.append(f"S4: window {oid} overlaps a door on wall "
                                f"{w.get('id')}")
                    break
    return errs


def _span_on_wall(g: _Geo, o: dict, wi: int) -> Optional[Tuple[float, float]]:
    """Project an opening onto its host wall centerline as a [t0,t1] interval."""
    geoms = g.wall_geoms[wi]
    if not geoms:
        return None
    c = o.get("center")
    ln = geoms[0]
    if len(geoms) > 1 and c is not None:        # curved wall: pick nearer geom
        ln = min(geoms, key=lambda L: L.distance(Point(c)))
    try:
        t1 = ln.project(Point(o["p1"]))
        t2 = ln.project(Point(o["p2"]))
        return (min(t1, t2), max(t1, t2))
    except Exception:
        return None


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return min(a[1], b[1]) - max(a[0], b[0])


def check_S5(g: _Geo) -> List[str]:
    """Openings on one wall don't overlap and fit within the wall span."""
    errs = []
    spans: Dict[int, List[Tuple[Tuple[float, float], str]]] = defaultdict(list)
    for o in g.openings:
        wi = g.host_wall(o)
        if wi is None:
            continue
        s = _span_on_wall(g, o, wi)
        if s is not None:
            spans[wi].append((s, o.get("id")))
    for wi, items in spans.items():
        w = g.walls[wi]
        L = w.get("length_mm", 0)
        items.sort(key=lambda it: it[0][0])
        total = 0.0
        for k, (s, oid) in enumerate(items):
            total += s[1] - s[0]
            if k > 0:
                prev, pid = items[k - 1]
                if _overlap(prev, s) > TOL_PT:
                    errs.append(f"S5: openings {pid} and {oid} overlap on wall "
                                f"{w.get('id')}")
        if not w.get("arc") and total > L + TOL_PT:
            errs.append(f"S5: openings on wall {w.get('id')} total "
                        f"{total:.0f}mm > wall length {L}mm")
    return errs


def check_S6(g: _Geo) -> List[str]:
    """Minimum room area and sane aspect ratio (no degenerate slivers)."""
    errs = []
    for ri, p in enumerate(g.room_polys):
        if p is None:
            continue
        rid = g.rooms[ri].get("id", ri)
        if p.area < MIN_ROOM_AREA:
            errs.append(f"S6: room {rid} area {p.area / 1e6:.2f} m^2 below "
                        f"{MIN_ROOM_AREA / 1e6:.0f} m^2")
        aspect, side = _aspect_and_minside(p)
        if aspect > MAX_ASPECT and side < MIN_SIDE:
            errs.append(f"S6: room {rid} is a sliver (aspect {aspect:.1f}, "
                        f"min side {side:.0f}mm)")
    return errs


def check_S7(g: _Geo) -> List[str]:
    """Realistic wall thickness band."""
    errs = []
    for w in g.walls:
        t = w.get("thickness_mm", 0)
        if t < THICK_MIN or t > THICK_MAX:
            errs.append(f"S7: wall {w.get('id')} thickness {t}mm outside "
                        f"[{THICK_MIN},{THICK_MAX}]")
    return errs


def check_S8(g: _Geo) -> List[str]:
    """Realistic door width."""
    errs = []
    for o in g.openings:
        if o.get("category") != "door":
            continue
        wd = o.get("width_mm", 0)
        if wd < DOOR_W_MIN or wd > DOOR_W_MAX:
            errs.append(f"S8: door {o.get('id')} width {wd}mm outside "
                        f"[{DOOR_W_MIN},{DOOR_W_MAX}]")
    return errs


def check_S9(g: _Geo) -> List[str]:
    """Room typing present; declared counts match actual."""
    errs = []
    for ri, r in enumerate(g.rooms):
        if not r.get("room_type"):
            errs.append(f"S9: room {r.get('id', ri)} has no room_type")
    counts = g.config.get("counts", {})
    if counts:
        if counts.get("rooms") != len(g.rooms):
            errs.append(f"S9: counts.rooms {counts.get('rooms')} != "
                        f"{len(g.rooms)}")
        if counts.get("walls") != len(g.walls):
            errs.append(f"S9: counts.walls {counts.get('walls')} != "
                        f"{len(g.walls)}")
        n_doors = sum(1 for o in g.openings if o.get("category") == "door")
        n_win = sum(1 for o in g.openings if o.get("category") == "window")
        if counts.get("doors") not in (None, n_doors):
            errs.append(f"S9: counts.doors {counts.get('doors')} != {n_doors}")
        if counts.get("windows") not in (None, n_win):
            errs.append(f"S9: counts.windows {counts.get('windows')} != {n_win}")
    return errs


def check_S10(g: _Geo) -> List[str]:
    """No wall crosses an opening.

    A door/window occupies a clear span p1->p2 on its host wall.  No OTHER wall
    may pass transversely through the interior of that span -- a wall meeting
    the host wall in the middle of a doorway would physically block it.  A wall
    that merely T-junctions at the jamb (at/near an opening endpoint) is fine,
    so only crossings strictly interior to the span count."""
    errs = []
    for o in g.openings:
        seg = g._opening_geom(o)
        if seg is None or seg.geom_type != "LineString":
            continue
        p1, p2 = o.get("p1"), o.get("p2")
        if not (p1 and p2):
            continue
        wi = g.host_wall(o)
        for j, geoms in enumerate(g.wall_geoms):
            if j == wi:
                continue                      # the host wall carries the opening
            crossed = False
            for ln in geoms:
                inter = ln.intersection(seg)
                if inter.is_empty:
                    continue
                # only a TRANSVERSE crossing blocks the opening: the other wall
                # meets the span at a point in its interior.  A collinear wall
                # on the same line (intersection is a segment) is redundant
                # geometry, not a wall through the doorway -- ignore it.
                gt = inter.geom_type
                if gt == "Point":
                    pts = [(inter.x, inter.y)]
                elif gt == "MultiPoint":
                    pts = [(p.x, p.y) for p in inter.geoms]
                else:
                    continue
                for pt in pts:
                    if (_dist(pt, p1) > TOL_JAMB and _dist(pt, p2) > TOL_JAMB):
                        crossed = True
                        break
                if crossed:
                    break
            if crossed:
                wid = g.walls[j].get("id", f"#{j}")
                errs.append(f"S10: wall {wid} crosses opening {o.get('id')} "
                            f"(blocks its span)")
    return errs


# ===========================================================================
# Consistency predicates
# ===========================================================================
def check_C1(g: _Geo) -> List[str]:
    """Output JSON schema unchanged (required top-level keys present)."""
    errs = []
    required = ("id", "group", "name", "units", "origin", "bbox", "footprint",
                "counts", "walls", "rooms", "openings", "render", "metadata")
    for k in required:
        if k not in g.config:
            errs.append(f"C1: missing top-level key '{k}'")
    for w in g.walls:
        for k in ("id", "type", "thickness_mm", "length_mm", "centerline",
                  "polygon", "arc"):
            if k not in w:
                errs.append(f"C1: wall {w.get('id')} missing '{k}'")
                break
    for r in g.rooms:
        for k in ("id", "polygon", "room_type"):
            if k not in r:
                errs.append(f"C1: room {r.get('id')} missing '{k}'")
                break
    for o in g.openings:
        for k in ("id", "category", "p1", "p2", "center", "width_mm"):
            if k not in o:
                errs.append(f"C1: opening {o.get('id')} missing '{k}'")
                break
    return errs


def check_C2(g: _Geo) -> List[str]:
    """centerline <-> length_mm consistent; arc set iff curved."""
    errs = []
    for w in g.walls:
        if w.get("arc"):
            continue
        cl = w.get("centerline")
        if cl and len(cl) >= 2:
            d = _dist(cl[0], cl[1])
            if abs(d - w.get("length_mm", 0)) > 2:
                errs.append(f"C2: wall {w.get('id')} length_mm != |centerline|")
        if len(w.get("polygon", [])) < 4:
            errs.append(f"C2: wall {w.get('id')} polygon < 4 pts")
    return errs


def check_C3(g: _Geo) -> List[str]:
    """openings p1/p2/center on host wall; width == |p1-p2|; valid category."""
    errs = []
    for o in g.openings:
        oid = o.get("id")
        cat = o.get("category")
        if cat not in ("door", "window", "opening"):
            errs.append(f"C3: opening {oid} bad category '{cat}'")
        p1, p2, c = o.get("p1"), o.get("p2"), o.get("center")
        if not (p1 and p2 and c):
            errs.append(f"C3: opening {oid} missing endpoint")
            continue
        if abs(_dist(p1, p2) - o.get("width_mm", -1)) > 3:
            errs.append(f"C3: opening {oid} width_mm != |p1-p2|")
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        if _dist(mid, c) > 2:
            errs.append(f"C3: opening {oid} center not midpoint of p1,p2")
        if g.host_wall(o) is None:
            errs.append(f"C3: opening {oid} center not on any wall")
        # subtype enum sanity
        st = o.get("subtype")
        if cat == "door" and st not in DOOR_ENUM:
            errs.append(f"C3: door {oid} bad subtype '{st}'")
        if cat == "window" and st not in WIN_ENUM:
            errs.append(f"C3: window {oid} bad subtype '{st}'")
    return errs


def check_C5(g: _Geo) -> List[str]:
    """units == mm; bbox / footprint coherent."""
    errs = []
    if g.config.get("units") != "mm":
        errs.append("C5: units != 'mm'")
    bb = g.config.get("bbox")
    fp = g.config.get("footprint")
    if isinstance(bb, list) and len(bb) == 4:
        if bb[2] < bb[0] or bb[3] < bb[1]:
            errs.append("C5: bbox max < min")
        if isinstance(fp, dict):
            if abs((bb[2] - bb[0]) - fp.get("w", -1)) > 2 or \
                    abs((bb[3] - bb[1]) - fp.get("h", -1)) > 2:
                errs.append("C5: footprint w/h disagree with bbox")
    else:
        errs.append("C5: bbox malformed")
    return errs


# ===========================================================================
# Aggregate
# ===========================================================================
_CHECKS = [
    ("T1", check_T1), ("T2", check_T2), ("T3", check_T3), ("T4", check_T4),
    ("T5", check_T5), ("T6", check_T6), ("T7", check_T7), ("T8", check_T8),
    ("S1", check_S1), ("S2", check_S2), ("S3", check_S3), ("S4", check_S4),
    ("S5", check_S5), ("S6", check_S6), ("S7", check_S7), ("S8", check_S8),
    ("S9", check_S9), ("S10", check_S10),
    ("C1", check_C1), ("C2", check_C2), ("C3", check_C3), ("C5", check_C5),
]


def validate_plan(config: dict) -> Tuple[bool, List[str]]:
    """Return (is_valid, violations). Every T*/S*/C* predicate is checked.

    Each violation string is prefixed by the failed constraint id, e.g.
    'S1: room R7 has no door (sealed)'.
    """
    try:
        g = _Geo(config)
    except Exception as e:                                   # pragma: no cover
        return False, [f"FATAL: could not build geometry ({e})"]
    violations: List[str] = []
    for cid, fn in _CHECKS:
        try:
            violations.extend(fn(g))
        except Exception as e:                               # pragma: no cover
            violations.append(f"{cid}: validator error ({e})")
    return (len(violations) == 0), violations


def constraint_of(v: str) -> str:
    return v.split(":", 1)[0].strip()


# ===========================================================================
# CLI
# ===========================================================================
def _iter_config_paths(path: str, limit: int) -> List[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "plan_*.json")))
        if limit > 0:
            files = files[:limit]
        return files
    return [path]


def _run_single(path: str) -> int:
    with open(path) as fh:
        config = json.load(fh)
    ok, violations = validate_plan(config)
    if ok:
        print(f"PASS  {os.path.basename(path)}")
        return 0
    print(f"FAIL  {os.path.basename(path)}  ({len(violations)} violations)")
    for v in violations:
        print(f"   - {v}")
    return 1


def _run_batch(paths: List[str], json_out: Optional[str]) -> int:
    n = len(paths)
    n_valid = 0
    per_constraint = Counter()       # plans failing each constraint at least once
    total_violations = Counter()     # raw violation occurrences
    failing_examples: Dict[str, str] = {}
    print(f"Validating {n} plans ...")
    for i, p in enumerate(paths, 1):
        try:
            with open(p) as fh:
                config = json.load(fh)
            ok, violations = validate_plan(config)
        except Exception as e:
            violations = [f"FATAL: {e}"]
            ok = False
        if ok:
            n_valid += 1
        else:
            hit = set()
            for v in violations:
                c = constraint_of(v)
                total_violations[c] += 1
                hit.add(c)
                failing_examples.setdefault(c, f"{os.path.basename(p)}: {v}")
            for c in hit:
                per_constraint[c] += 1
        if i % 1000 == 0 or i == n:
            print(f"  [{i}/{n}]  valid so far: {n_valid} "
                  f"({100.0 * n_valid / i:.1f}%)")

    n_invalid = n - n_valid
    print("\n" + "=" * 64)
    print("VALIDATION BASELINE")
    print("=" * 64)
    print(f"  Plans            : {n}")
    print(f"  Valid            : {n_valid} ({100.0 * n_valid / max(1, n):.2f}%)")
    print(f"  Invalid          : {n_invalid} "
          f"({100.0 * n_invalid / max(1, n):.2f}%)")
    print("\n  Per-constraint failures (plans failing >=1 time):")
    order = [cid for cid, _ in _CHECKS]
    for c in order:
        if per_constraint.get(c):
            pct = 100.0 * per_constraint[c] / n
            print(f"    {c:>3} : {per_constraint[c]:>6} plans ({pct:5.2f}%)   "
                  f"e.g. {failing_examples.get(c, '')[:90]}")
    other = [c for c in per_constraint if c not in order]
    for c in sorted(other):
        print(f"    {c:>3} : {per_constraint[c]:>6} plans   "
              f"e.g. {failing_examples.get(c, '')[:90]}")

    if json_out:
        with open(json_out, "w") as fh:
            json.dump({
                "plans": n, "valid": n_valid, "invalid": n_invalid,
                "valid_pct": round(100.0 * n_valid / max(1, n), 3),
                "per_constraint_plans": dict(per_constraint),
                "total_violations": dict(total_violations),
                "examples": failing_examples,
            }, fh, indent=2)
        print(f"\n  Report written -> {json_out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strict floor-plan validator")
    ap.add_argument("path", help="a plan_*.json file OR a directory of them")
    ap.add_argument("--limit", type=int, default=0,
                    help="(dir mode) validate only the first N plans")
    ap.add_argument("--json", default=None, metavar="FILE",
                    help="(dir mode) write the histogram report to FILE")
    args = ap.parse_args(argv)

    paths = _iter_config_paths(args.path, args.limit)
    if not paths:
        print(f"No plan_*.json found at {args.path}")
        return 2
    if len(paths) == 1 and not os.path.isdir(args.path):
        return _run_single(paths[0])
    return _run_batch(paths, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
