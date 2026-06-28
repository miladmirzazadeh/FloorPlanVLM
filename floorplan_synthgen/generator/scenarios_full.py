"""The full 55-scenario matrix in the engine's native config format.

Groups: A typology (15), B geometry (10), C opening edge cases (15),
D international standards (7), E clutter/render stress (8).

Authoring model: straight centreline walls + openings placed by (wall_id,
distance-from-start). Curves are tessellated into short straight segments.
Rooms are label anchors (rectangles or polygons). Subtypes use the
standardised enums the engine now supports.

Constraints enforced by ``check_all()`` (mirrors the brief):
  * >=3 distinct wall thicknesses per plan
  * >=1 non-rectangular room per plan
  * plans with >3 rooms: a wall >8 m and a wall <1.5 m
  * unique footprint per group
  * varied opening sizes dataset-wide
"""

from __future__ import annotations

import math
from typing import Dict, List

# realistic, varied clear-opening pools (mm)
DOOR_W = [610, 686, 711, 726, 762, 813, 826, 838, 900, 915, 927, 1000, 1067, 1200]
WIN_W = [450, 540, 600, 720, 750, 900, 1010, 1100, 1200, 1350, 1480, 1500, 1760,
         1810, 2100, 2400]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Plan:
    def __init__(self, pid, group, name, scale="1:100", rotation=0.0,
                 description="", hard_case="", clutter=None, render=None):
        self.pid = pid
        self.group = group
        self.name = name
        self.scale = scale
        self.rotation = float(rotation)
        self.description = description
        self.hard_case = hard_case
        self.walls: Dict[str, dict] = {}
        self.wall_order: List[str] = []
        self.rooms: List[dict] = []
        self.openings: List[dict] = []
        self.columns: List[dict] = []
        self.clutter = clutter or {"room_labels": True, "dimensions": False,
                                   "title_block": False, "furniture": False,
                                   "hatch_walls": False, "grid": False,
                                   "noise_lines": 0}
        self.render = render or {"dpi": 150, "line_weight_style": "standard",
                                 "monochrome": True}
        self.W = 0
        self.H = 0

    # --- walls ----------------------------------------------------------
    def wall(self, wid, x1, y1, x2, y2, t, wtype="INTERIOR"):
        self.walls[wid] = {"id": wid, "x1": float(x1), "y1": float(y1),
                           "x2": float(x2), "y2": float(y2),
                           "thickness": float(t), "wall_type": wtype}
        self.wall_order.append(wid)
        return wid

    def envelope(self, W, H, t):
        self.W, self.H = W, H
        self.wall("EXT_S", 0, 0, W, 0, t, "EXTERIOR")
        self.wall("EXT_N", 0, H, W, H, t, "EXTERIOR")
        self.wall("EXT_W", 0, 0, 0, H, t, "EXTERIOR")
        self.wall("EXT_E", W, 0, W, H, t, "EXTERIOR")

    def arc(self, prefix, cx, cy, r, a0, a1, t, wtype="EXTERIOR", segs=16):
        """Tessellate an arc into straight wall segments; return their ids."""
        pts = []
        for k in range(segs + 1):
            a = math.radians(a0 + (a1 - a0) * k / segs)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        ids = []
        for k in range(segs):
            wid = f"{prefix}{k}"
            self.wall(wid, pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1],
                      t, wtype)
            ids.append(wid)
        return ids

    def wlen(self, wid):
        w = self.walls[wid]
        return math.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"])

    # --- openings -------------------------------------------------------
    def door(self, oid, wid, dist, clear, subtype="SINGLE", swing="INWARD_LEFT",
             max_swing=90):
        pos = _clamp(dist / self.wlen(wid), 0.0, 1.0)
        self.openings.append({"id": oid, "type": "door", "wall_id": wid,
                              "position_along_wall": round(pos, 4),
                              "subtype": subtype, "clear_opening_mm": clear,
                              "swing": swing, "max_swing_angle_deg": max_swing})

    def window(self, oid, wid, dist, clear, subtype="FIXED"):
        pos = _clamp(dist / self.wlen(wid), 0.0, 1.0)
        self.openings.append({"id": oid, "type": "window", "wall_id": wid,
                              "position_along_wall": round(pos, 4),
                              "subtype": subtype, "clear_opening_mm": clear})

    def rooflight(self, oid, x, y, size, subtype="FIXED"):
        self.openings.append({"id": oid, "type": "window", "rooflight": True,
                              "x": float(x), "y": float(y), "size_mm": size,
                              "subtype": subtype})

    # --- rooms / columns ------------------------------------------------
    def room(self, rid, name, x, y, w, h):
        self.rooms.append({"id": rid, "name": name, "x": x, "y": y,
                           "width": w, "height": h})

    def room_poly(self, rid, name, polygon):
        self.rooms.append({"id": rid, "name": name,
                           "polygon": [[float(p[0]), float(p[1])] for p in polygon]})

    def column(self, cid, x, y, shape="SQUARE", size=300):
        self.columns.append({"id": cid, "x": x, "y": y, "shape": shape,
                             "size_mm": size})

    # --- output ---------------------------------------------------------
    def config(self):
        return {"plan_id": self.pid, "scenario": self.pid, "group": self.group,
                "name": self.name, "description": self.description,
                "hard_case": self.hard_case,
                "scale": self.scale, "rotation_deg": self.rotation,
                "rooms": self.rooms,
                "walls": [self.walls[w] for w in self.wall_order],
                "openings": self.openings, "columns": self.columns,
                "clutter": self.clutter, "render": self.render}


REGISTRY: Dict[str, List[Plan]] = {"A": [], "B": [], "C": [], "D": [], "E": []}


def reg(p: Plan) -> Plan:
    REGISTRY[p.group].append(p)
    return p


# ===========================================================================
# constraint checker
# ===========================================================================
def _poly_is_rect(poly):
    if len(poly) != 4:
        return False
    xs = {round(p[0]) for p in poly}
    ys = {round(p[1]) for p in poly}
    ortho = all(p[0] == q[0] or p[1] == q[1]
                for p, q in zip(poly, poly[1:] + poly[:1]))
    return ortho and len(xs) == 2 and len(ys) == 2


def check_all() -> List[str]:
    errs = []
    door_w, win_w = [], []
    for g, plans in REGISTRY.items():
        seen = {}
        for p in plans:
            thk = {w["thickness"] for w in p.walls.values()}
            if len(thk) < 3:
                errs.append(f"{p.pid}: <3 wall thicknesses {sorted(thk)}")
            nonrect = any("polygon" in r and not _poly_is_rect(r["polygon"])
                          for r in p.rooms)
            if not nonrect:
                errs.append(f"{p.pid}: no non-rectangular room")
            lengths = [p.wlen(w) for w in p.walls]
            if len(p.rooms) > 3 and lengths:
                if max(lengths) <= 8000:
                    errs.append(f"{p.pid}: >3 rooms, no wall >8m (max {max(lengths):.0f})")
                if min(lengths) >= 1500:
                    errs.append(f"{p.pid}: >3 rooms, no wall <1.5m (min {min(lengths):.0f})")
            for o in p.openings:
                (door_w if o["type"] == "door" else win_w).append(
                    o.get("clear_opening_mm", o.get("size_mm", 0)))
            # footprint = bbox of wall centrelines
            xs = [w["x1"] for w in p.walls.values()] + [w["x2"] for w in p.walls.values()]
            ys = [w["y1"] for w in p.walls.values()] + [w["y2"] for w in p.walls.values()]
            fp = (round(max(xs) - min(xs)), round(max(ys) - min(ys)))
            if fp in seen:
                errs.append(f"group {g}: {p.pid} shares footprint {fp} with {seen[fp]}")
            seen[fp] = p.pid
    if door_w and len(set(door_w)) < 6:
        errs.append(f"too few distinct door widths ({len(set(door_w))})")
    if win_w and len(set(win_w)) < 6:
        errs.append(f"too few distinct window widths ({len(set(win_w))})")
    return errs


# ===========================================================================
# GROUP A -- typology coverage
# ===========================================================================
def build_A():
    C_LIGHT = {"room_labels": True, "dimensions": False, "title_block": False,
               "furniture": False, "hatch_walls": False, "grid": False, "noise_lines": 0}
    C_MED = {"room_labels": True, "dimensions": True, "title_block": True,
             "furniture": True, "hatch_walls": False, "grid": False, "noise_lines": 2}
    C_HEAVY = {"room_labels": True, "dimensions": True, "title_block": True,
               "furniture": True, "hatch_walls": True, "grid": True, "noise_lines": 4}

    # A01 Studio apartment (2 rooms)
    p = reg(Plan("A01", "A", "Studio apartment", "1:50",
                 description="6.0x4.5m studio: living/sleeping/kitchen + wet room.",
                 hard_case="Micro openings packed tightly at the smallest scale.",
                 clutter=C_LIGHT))
    p.envelope(6000, 4500, 200)
    p.wall("P1", 4200, 3000, 6000, 3000, 75)
    p.wall("P2", 4200, 3000, 4200, 4500, 75)
    p.wall("P3", 600, 0, 600, 1000, 100)
    p.room_poly("living", "Living",
                [[0, 0], [6000, 0], [6000, 3000], [4200, 3000], [4200, 4500], [0, 4500]])
    p.room("wet", "Wet", 4200, 3000, 1800, 1500)
    p.door("D1", "EXT_W", 1300, 826, "SINGLE", "INWARD_RIGHT")
    p.door("D2", "P2", 750, 610, "POCKET", None, 0)
    p.window("W1", "EXT_S", 3000, 1480, "CASEMENT")
    p.window("W2", "EXT_E", 1500, 600, "FIXED")
    p.window("W3", "EXT_N", 5100, 540, "AWNING")

    # A02 Two-bed apartment (4 rooms)
    p = reg(Plan("A02", "A", "Two-bed apartment", "1:50",
                 description="9.2x7.4m unit: open living+kitchen, 2 beds, bath, balcony.",
                 hard_case="Balcony slider beside a window on the same wall.",
                 clutter=C_MED))
    p.envelope(9200, 7400, 250)
    p.wall("PARTY", 0, 4200, 9200, 4200, 200, "INTERIOR")
    p.wall("P1", 3600, 4200, 3600, 7400, 100)
    p.wall("P2", 6400, 4200, 6400, 7400, 100)
    p.wall("P3", 6400, 5600, 9200, 5600, 75)
    p.wall("P4", 1800, 0, 1800, 1200, 100)
    p.room("living", "Living + Kitchen", 0, 0, 9200, 4200)
    p.room("bed1", "Bedroom 1", 0, 4200, 3600, 3200)
    p.room("bed2", "Bedroom 2", 3600, 4200, 2800, 3200)
    p.room_poly("hall", "Hall + Bath",
                [[6400, 4200], [9200, 4200], [9200, 5600], [7800, 5600],
                 [7800, 7400], [6400, 7400]])
    p.door("D1", "EXT_W", 600, 900, "SINGLE", "INWARD_LEFT")
    p.door("D2", "P1", 1600, 762, "SINGLE", "INWARD_RIGHT")
    p.door("D3", "P2", 1600, 762, "SINGLE", "INWARD_LEFT")
    p.door("D4", "P3", 1400, 686, "SINGLE", "INWARD_RIGHT")
    p.door("D5", "EXT_S", 2000, 1810, "SLIDING", None, 0)
    p.window("W1", "EXT_S", 5200, 1200, "CASEMENT")
    p.window("W2", "EXT_E", 5800, 1010, "CASEMENT")
    p.window("W3", "EXT_W", 5900, 900, "CASEMENT")

    # A03 Three-bed apartment (5 rooms)
    p = reg(Plan("A03", "A", "Three-bed apartment", "1:50",
                 description="11.8x8.6m unit with central circulation and many doors.",
                 hard_case="High interior-door density off a central hall.",
                 clutter=C_MED))
    p.envelope(11800, 8600, 250)
    p.wall("PARTY", 0, 4400, 11800, 4400, 200)
    p.wall("P1", 4200, 0, 4200, 4400, 100)
    p.wall("P2", 4200, 4400, 4200, 8600, 100)
    p.wall("P3", 8200, 4400, 8200, 8600, 100)
    p.wall("P4", 8200, 6400, 11800, 6400, 75)
    p.wall("P5", 4200, 2000, 5100, 2000, 90)
    p.room("living", "Living/Dining", 0, 0, 4200, 4400)
    p.room("kitchen", "Kitchen", 4200, 0, 7600, 4400)
    p.room("bed1", "Master", 0, 4400, 4200, 4200)
    p.room("bed2", "Bedroom 2", 4200, 4400, 4000, 4200)
    p.room_poly("bed3", "Bedroom 3 + Baths",
                [[8200, 4400], [11800, 4400], [11800, 6400], [10000, 6400],
                 [10000, 8600], [8200, 8600]])
    p.door("D1", "EXT_W", 6800, 915, "SINGLE", "INWARD_LEFT")
    p.door("D2", "P1", 1200, 826, "SINGLE", "INWARD_LEFT")
    p.door("D3", "P2", 2200, 762, "SINGLE", "INWARD_RIGHT")
    p.door("D4", "P3", 2200, 762, "SINGLE", "INWARD_LEFT")
    p.door("D5", "P4", 900, 686, "SINGLE", "INWARD_RIGHT")
    p.door("D6", "P5", 450, 610, "POCKET", None, 0)
    p.window("W1", "EXT_S", 2400, 1760, "CASEMENT")
    p.window("W2", "EXT_S", 8000, 1480, "FIXED")
    p.window("W3", "EXT_N", 2400, 1350, "CASEMENT")
    p.window("W4", "EXT_N", 6200, 1350, "CASEMENT")
    p.window("W5", "EXT_E", 7400, 750, "AWNING")

    # A04 Terraced house (4 rooms, elongated)
    p = reg(Plan("A04", "A", "Terraced house", "1:50",
                 description="5.2x14.5m narrow plan; rooms along a long corridor.",
                 hard_case="Very long thin plan; openings line up on one axis.",
                 clutter=C_LIGHT))
    p.envelope(5200, 14500, 215)
    p.wall("COR", 3400, 0, 3400, 14500, 100)
    p.wall("P1", 0, 4200, 3400, 4200, 100)
    p.wall("P2", 0, 8400, 3400, 8400, 100)
    p.wall("P3", 3400, 11200, 5200, 11200, 75)
    p.wall("P4", 3400, 12100, 4300, 12100, 90)
    p.room("front", "Living", 0, 0, 3400, 4200)
    p.room("mid", "Dining", 0, 4200, 3400, 4200)
    p.room_poly("kitchen", "Kitchen",
                [[0, 8400], [3400, 8400], [3400, 14500], [1800, 14500],
                 [1800, 12500], [0, 12500]])
    p.room("cor", "Corridor + Stair", 3400, 0, 1800, 14500)
    p.door("D1", "EXT_S", 1700, 900, "SINGLE", "INWARD_LEFT")
    p.door("D2", "COR", 2100, 762, "SINGLE", "INWARD_LEFT")
    p.door("D3", "COR", 6300, 762, "SINGLE", "INWARD_RIGHT")
    p.door("D4", "COR", 11000, 762, "SINGLE", "INWARD_LEFT")
    p.door("D5", "P3", 900, 610, "POCKET", None, 0)
    p.door("D6", "EXT_N", 4300, 838, "SINGLE", "OUTWARD_RIGHT")
    p.window("W1", "EXT_S", 3400, 1500, "BAY")
    p.window("W2", "EXT_E", 6300, 600, "FIXED")
    p.window("W3", "EXT_N", 1700, 1200, "CASEMENT")

    # A05 Open plan office (3 rooms)
    p = reg(Plan("A05", "A", "Open plan office", "1:100",
                 description="24x16m office: open field, glazed pods, service core.",
                 hard_case="Glass partitions as double lines resemble glazing.",
                 clutter=C_HEAVY))
    p.envelope(24000, 16000, 300)
    p.wall("CORE_S", 9000, 6000, 15000, 6000, 200, "INTERIOR")
    p.wall("CORE_N", 9000, 11000, 15000, 11000, 200)
    p.wall("CORE_W", 9000, 6000, 9000, 11000, 200)
    p.wall("CORE_E", 15000, 6000, 15000, 11000, 200)
    p.wall("POD_S", 2000, 12000, 7000, 12000, 50)
    p.wall("POD_E", 7000, 12000, 7000, 16000, 50)
    p.wall("BOOTH", 22000, 0, 22000, 1100, 50)
    p.room_poly("open", "Open Workspace",
                [[0, 0], [24000, 0], [24000, 16000], [0, 16000], [0, 11000],
                 [9000, 11000], [9000, 6000], [0, 6000]])
    p.room("core", "Core", 9000, 6000, 6000, 5000)
    p.room("pod", "Meeting Pod", 2000, 12000, 5000, 4000)
    p.door("D1", "EXT_S", 3000, 1810, "DOUBLE", "INWARD_LEFT")
    p.door("D2", "POD_S", 1500, 1000, "SINGLE", "INWARD_RIGHT")
    p.door("D3", "CORE_W", 2500, 1000, "SINGLE", "INWARD_LEFT")
    p.door("D4", "CORE_E", 2500, 1000, "SINGLE", "INWARD_RIGHT")
    p.window("CW1", "EXT_W", 8000, 2400, "FIXED")
    p.window("CW2", "EXT_E", 8000, 2400, "FIXED")
    p.window("CW3", "EXT_N", 12000, 2400, "SLIDING")

    # A06 Hotel corridor (repetitive rooms)
    p = reg(Plan("A06", "A", "Hotel corridor", "1:100",
                 description="32x13m double-loaded corridor, repetitive guest rooms.",
                 hard_case="High repetition of identical entry doors.",
                 clutter=C_MED))
    p.envelope(32000, 13000, 250)
    p.wall("COR_S", 0, 5800, 32000, 5800, 150)
    p.wall("COR_N", 0, 7200, 32000, 7200, 150)
    xs = [0, 5300, 10600, 15900, 21200, 26500, 32000]
    for i in range(6):
        x0, x1 = xs[i], xs[i + 1]
        if i < 5:
            p.wall(f"VS{i}", x1, 0, x1, 5800, 100)
            p.wall(f"VN{i}", x1, 7200, x1, 13000, 100)
        p.wall(f"BS{i}", x0 + 800, 0, x0 + 800, 1100, 75)
        if i == 0:
            p.room_poly(f"rmS{i}", f"Room S{i+1}",
                        [[x0, 0], [x1, 0], [x1, 5800], [x0 + 1800, 5800],
                         [x0 + 1800, 3800], [x0, 3800]])
        else:
            p.room(f"rmS{i}", f"Room S{i+1}", x0, 0, x1 - x0, 5800)
        p.room(f"rmN{i}", f"Room N{i+1}", x0, 7200, x1 - x0, 5800)
        p.door(f"DS{i}", "COR_S", x0 + 1400, 826, "SINGLE",
               "INWARD_LEFT" if i % 2 == 0 else "INWARD_RIGHT")
        p.door(f"DN{i}", "COR_N", x0 + 1400, 826, "SINGLE",
               "OUTWARD_RIGHT" if i % 2 == 0 else "OUTWARD_LEFT")
        p.window(f"WS{i}", "EXT_S", x0 + 2650, 1480, "CASEMENT")
        p.window(f"WN{i}", "EXT_N", x0 + 2650, 1480, "CASEMENT")
    p.room("cor", "Corridor", 0, 5800, 32000, 1400)
    p.door("DLIFT", "COR_S", 15900, 1100, "SLIDING", None, 0)

    # A07 Hospital ward (wide doors, airlock)
    p = reg(Plan("A07", "A", "Hospital ward", "1:100",
                 description="28x14.5m ward: wide corridor, patient rooms, isolation airlock.",
                 hard_case="Extra-wide bed doors; two airlock doors close together.",
                 clutter=C_HEAVY))
    p.envelope(28000, 14500, 250)
    p.wall("COR_S", 0, 6000, 28000, 6000, 150)
    p.wall("COR_N", 0, 8500, 28000, 8500, 150)
    px = [0, 4800, 9600, 14400, 19200, 24000, 28000]
    for i in range(5):
        x0, x1 = px[i], px[i + 1]
        p.wall(f"PV{i}", x1, 8500, x1, 14500, 120)
        p.room(f"pat{i}", f"Patient {i+1}", x0, 8500, x1 - x0, 6000)
        p.door(f"DP{i}", "COR_N", x0 + 1400, 1100 + (i % 2) * 100, "SINGLE",
               "INWARD_RIGHT" if i % 2 else "INWARD_LEFT")
        p.window(f"WP{i}", "EXT_N", x0 + 2400, 1810, "FIXED")
    p.wall("ISO_W", 19200, 0, 19200, 6000, 120)
    p.wall("ISO_E", 21600, 0, 21600, 6000, 120)
    p.wall("ISO_H", 19200, 2400, 21600, 2400, 90)
    p.wall("STUB", 24000, 0, 24000, 1200, 75)
    p.room_poly("nurse", "Nurse + Utility",
                [[0, 0], [19200, 0], [19200, 2400], [21600, 2400], [21600, 6000],
                 [0, 6000]])
    p.room("ante", "Anteroom", 19200, 0, 2400, 2400)
    p.room("cor", "Corridor", 0, 6000, 28000, 2500)
    p.door("DISOa", "ISO_W", 1200, 1100, "SINGLE", "INWARD_LEFT")
    p.door("DISOb", "ISO_H", 1200, 1067, "SINGLE", "INWARD_LEFT")
    p.door("DSMOKE", "COR_S", 14000, 1700, "DOUBLE", "INWARD_LEFT")
    p.door("DNS", "COR_S", 5000, 2400, "SLIDING", None, 0)
    p.window("WEND", "EXT_E", 10000, 900, "AWNING")

    # A08 School classroom block (high window density)
    p = reg(Plan("A08", "A", "School classroom block", "1:100",
                 description="40x11.5m single-loaded corridor; banded classroom windows.",
                 hard_case="High window density; doors swing outward into corridor.",
                 clutter=C_MED))
    p.envelope(40000, 11500, 300)
    p.wall("COR", 0, 8000, 40000, 8000, 150)
    cx = [0, 7600, 15200, 22800, 30400, 36000, 40000]
    for i in range(4):
        x0, x1 = cx[i], cx[i + 1]
        p.wall(f"CV{i}", x1, 0, x1, 8000, 120)
        p.room(f"cls{i}", f"Classroom {i+1}", x0, 0, x1 - x0, 8000)
        p.door(f"DC{i}", "COR", x0 + 1200, 900, "SINGLE", "OUTWARD_LEFT")
        for k in range(4):
            p.window(f"WC{i}_{k}", "EXT_S", x0 + 1000 + k * 1600, 1200, "CASEMENT")
    p.wall("TV", 36000, 0, 36000, 8000, 120)
    p.wall("CUB", 30400, 1000, 31300, 1000, 75)
    p.room("toilets", "Toilets", 30400, 0, 5600, 8000)
    p.room_poly("stair", "Stair + Lobby",
                [[36000, 0], [40000, 0], [40000, 11500], [0, 11500], [0, 8000],
                 [36000, 8000]])
    p.door("DWC", "COR", 33000, 850, "SINGLE", "OUTWARD_LEFT")
    p.door("DEXIT1", "EXT_N", 2000, 1800, "DOUBLE", "OUTWARD_LEFT")
    p.door("DEXIT2", "EXT_N", 38000, 1800, "DOUBLE", "OUTWARD_RIGHT")
    p.window("WCOR", "EXT_N", 20000, 2100, "CLERESTORY")

    # A09 Retail unit (glazed frontage)
    p = reg(Plan("A09", "A", "Retail unit", "1:100",
                 description="18x13.5m shop: sales floor, stock, fitting rooms, office.",
                 hard_case="Full-width storefront with embedded auto slider.",
                 clutter=C_HEAVY))
    p.envelope(18000, 13500, 250)
    p.wall("ST_W", 12000, 9000, 12000, 13500, 150)
    p.wall("ST_S", 12000, 9000, 18000, 9000, 150)
    p.wall("FIT", 0, 1500, 4500, 1500, 100)
    p.wall("FC1", 1500, 0, 1500, 1500, 75)
    p.wall("FC2", 3000, 0, 3000, 1500, 75)
    p.room_poly("sales", "Sales Floor",
                [[0, 0], [18000, 0], [18000, 9000], [12000, 9000], [12000, 13500], [0, 13500]])
    p.room("stock", "Stock", 12000, 9000, 6000, 4500)
    p.room("office", "Office", 12000, 0, 6000, 9000)
    p.door("D1", "EXT_N", 9000, 2000, "SLIDING", None, 0)
    p.door("D2", "ST_S", 3000, 900, "SINGLE", "INWARD_RIGHT")
    p.door("D3", "ST_W", 4000, 826, "SINGLE", "INWARD_LEFT")
    p.door("D4", "EXT_S", 16000, 1000, "SINGLE", "OUTWARD_LEFT")
    p.window("CW1", "EXT_N", 3500, 2400, "FIXED")
    p.window("CW2", "EXT_N", 14500, 2400, "FIXED")
    p.window("WOFF", "EXT_E", 4500, 1100, "AWNING")

    # A10 Cafe / restaurant (vestibule airlock)
    p = reg(Plan("A10", "A", "Cafe / restaurant", "1:50",
                 description="21x14m: dining, bar, kitchen, cold store, WCs, vestibule.",
                 hard_case="Vestibule = two doors close together; service double-acting.",
                 clutter=C_HEAVY))
    p.envelope(21000, 14000, 250)
    p.wall("KIT_W", 13000, 0, 13000, 14000, 150)
    p.wall("KIT_S", 13000, 6000, 21000, 6000, 150)
    p.wall("COLD", 16000, 0, 16000, 6000, 100)
    p.wall("VS", 0, 0, 2500, 0, 100)  # placeholder no
    p.walls.pop("VS"); p.wall_order.remove("VS")
    p.wall("VEST_E", 2500, 0, 2500, 2200, 100)
    p.wall("VEST_N", 0, 2200, 2500, 2200, 100)
    p.wall("WCSTUB", 18000, 6000, 18900, 6000, 75)
    p.room_poly("dining", "Dining",
                [[0, 2200], [2500, 2200], [2500, 0], [13000, 0], [13000, 14000], [0, 14000]])
    p.room("kitchen", "Kitchen", 13000, 6000, 8000, 8000)
    p.room("boh", "Cold + WC", 13000, 0, 8000, 6000)
    p.room("vest", "Vestibule", 0, 0, 2500, 2200)
    p.door("DV1", "EXT_S", 1250, 900, "SINGLE", "INWARD_LEFT")
    p.door("DV2", "VEST_N", 1250, 900, "SINGLE", "INWARD_RIGHT")
    p.door("DKIT", "KIT_W", 4000, 900, "SINGLE", "INWARD_LEFT")
    p.door("DCOLD", "COLD", 3000, 900, "SINGLE", "INWARD_LEFT")
    p.door("DWC", "KIT_S", 5500, 762, "SINGLE", "INWARD_RIGHT")
    p.window("CW1", "EXT_S", 7000, 2400, "FIXED")
    p.window("CW2", "EXT_S", 10000, 1500, "FIXED")
    p.window("WSIDE", "EXT_W", 7000, 1350, "CASEMENT")

    # A11 Gym / sports facility (large span)
    p = reg(Plan("A11", "A", "Gym / sports facility", "1:200",
                 description="30x22m sports hall + changing strip. Large spans.",
                 hard_case="Huge hall (low signal) beside dense small changing doors.",
                 clutter=C_MED))
    p.envelope(30000, 22000, 350)
    p.wall("SPLIT", 0, 5000, 22000, 5000, 200)
    cxs = [0, 3500, 7000, 10500, 14000]
    for i in range(4):
        x0, x1 = cxs[i], cxs[i + 1]
        p.wall(f"CV{i}", x1, 0, x1, 5000, 100)
        p.room(f"chg{i}", f"Changing {i+1}", x0, 0, x1 - x0, 5000)
        p.door(f"DC{i}", "SPLIT", x0 + 1750, 813, "SINGLE",
               "INWARD_LEFT" if i % 2 else "INWARD_RIGHT")
    p.wall("RDESK", 14000, 0, 14000, 1000, 75)
    p.room("hall", "Sports Hall", 0, 5000, 30000, 17000)
    p.room_poly("anc", "Reception + Store",
                [[15000, 0], [30000, 0], [30000, 5000], [14000, 5000]])
    p.door("DMAIN", "EXT_S", 6000, 1810, "DOUBLE", "INWARD_LEFT")
    p.door("DHALL", "SPLIT", 26000, 1810, "DOUBLE", "INWARD_LEFT")
    p.door("DFIRE", "EXT_E", 18000, 1000, "SINGLE", "OUTWARD_LEFT")
    p.window("WH1", "EXT_N", 8000, 2400, "FIXED")
    p.window("WH2", "EXT_N", 22000, 2400, "FIXED")
    p.window("WREC", "EXT_S", 24000, 1500, "CASEMENT")

    # A12 Basement car park (minimal openings)
    p = reg(Plan("A12", "A", "Basement car park", "1:200",
                 description="46x31m parking level: stair cores, lift, ramp.",
                 hard_case="Almost no doors; stair/lift doors among stall-line noise.",
                 clutter={"room_labels": True, "dimensions": False, "title_block": True,
                          "furniture": False, "hatch_walls": False, "grid": True,
                          "noise_lines": 6}))
    p.envelope(46000, 31000, 300)
    p.wall("S1_S", 2000, 2000, 6000, 2000, 200, "INTERIOR")
    p.wall("S1_N", 2000, 6000, 6000, 6000, 200)
    p.wall("S1_W", 2000, 2000, 2000, 6000, 200)
    p.wall("S1_E", 6000, 2000, 6000, 6000, 200)
    p.wall("S2_S", 40000, 25000, 44000, 25000, 200)
    p.wall("S2_N", 40000, 29000, 44000, 29000, 200)
    p.wall("S2_W", 40000, 25000, 40000, 29000, 200)
    p.wall("S2_E", 44000, 25000, 44000, 29000, 200)
    p.wall("LIFT_W", 20000, 28000, 20000, 31000, 200)
    p.wall("LIFT_E", 23000, 28000, 23000, 31000, 200)
    p.wall("RSTUB", 9000, 0, 10000, 0, 150)
    p.room_poly("deck", "Parking Deck",
                [[0, 0], [46000, 0], [46000, 31000], [0, 31000], [2000, 31000],
                 [2000, 6000], [6000, 6000], [6000, 2000], [0, 2000]])
    p.room("s1", "Stair 1", 2000, 2000, 4000, 4000)
    p.room("s2", "Stair 2", 40000, 25000, 4000, 4000)
    p.door("DS1", "S1_E", 2000, 1000, "SINGLE", "INWARD_LEFT")
    p.door("DS2", "S2_W", 2000, 1000, "SINGLE", "INWARD_RIGHT")
    p.door("DLIFT", "LIFT_W", 1500, 1100, "SLIDING", None, 0)
    p.door("DGATE", "EXT_S", 13500, 3000, "GARAGE", None, 0)
    p.window("WVENT", "EXT_E", 15500, 450, "LOUVRE")

    # A13 Penthouse (irregular + curved)
    p = reg(Plan("A13", "A", "Penthouse", "1:50",
                 description="~15x12m irregular outline with a curved living bay.",
                 hard_case="Mix of orthogonal, splayed and curved walls.",
                 clutter=C_MED))
    p.wall("EXT_S", 0, 0, 15000, 0, 250, "EXTERIOR")
    p.wall("EXT_E", 15000, 0, 15000, 7000, 250, "EXTERIOR")
    p.wall("SPLAY", 15000, 7000, 11000, 12000, 250, "EXTERIOR")
    p.wall("EXT_N", 11000, 12000, 4000, 12000, 250, "EXTERIOR")
    p.wall("EXT_W", 0, 12000, 0, 0, 250, "EXTERIOR")
    arc_ids = p.arc("BAY", 4000, 9000, 3000, 90, 180, 250, "EXTERIOR", segs=10)
    p.wall("P1", 7000, 0, 7000, 6000, 100)
    p.wall("P2", 7000, 6000, 15000, 6000, 100)
    p.wall("P3", 10500, 6000, 10500, 12000, 120)
    p.wall("WC", 12500, 0, 12500, 1200, 90)
    p.room_poly("living", "Living + Bay",
                [[0, 0], [7000, 0], [7000, 6000], [1000, 9000], [1000, 12000],
                 [4000, 12000], [4000, 9000]])
    p.room("kitchen", "Kitchen/Dining", 7000, 0, 8000, 6000)
    p.room_poly("suite1", "Master Suite",
                [[7000, 6000], [10500, 6000], [10500, 12000], [11000, 12000],
                 [15000, 7000], [15000, 6000]])
    p.door("D1", "EXT_S", 2000, 1000, "SINGLE", "INWARD_LEFT")
    p.door("D2", "P1", 3000, 826, "SINGLE", "INWARD_LEFT")
    p.door("D3", "P2", 2000, 826, "SINGLE", "INWARD_RIGHT")
    p.door("D4", "P3", 3000, 762, "SINGLE", "INWARD_LEFT")
    p.door("D5", "EXT_S", 6000, 2400, "SLIDING", None, 0)
    p.window("WBAY", arc_ids[5], p.wlen(arc_ids[5]) * 0.5, 1200, "CASEMENT")
    p.window("W1", "EXT_E", 3500, 1810, "FIXED")
    p.window("W2", "EXT_W", 1500, 900, "AWNING")

    # A14 Old masonry building (thick walls, small windows)
    p = reg(Plan("A14", "A", "Old masonry building", "1:50",
                 description="13x10.5m solid masonry: thick walls, small punched windows.",
                 hard_case="Thick walls (450/350); openings short vs wall depth.",
                 clutter={"room_labels": True, "dimensions": False, "title_block": False,
                          "furniture": False, "hatch_walls": True, "grid": False,
                          "noise_lines": 0}))
    p.envelope(13000, 10500, 450)
    p.wall("SPINE", 6500, 0, 6500, 10500, 350, "INTERIOR")
    p.wall("P1", 0, 5500, 6500, 5500, 250)
    p.wall("P2", 6500, 5500, 13000, 5500, 250)
    p.wall("CHIM", 10000, 0, 11000, 0, 120)
    p.room("hall", "Hall", 0, 0, 6500, 5500)
    p.room("parlour", "Parlour", 6500, 0, 6500, 5500)
    p.room("kitchen", "Kitchen", 0, 5500, 6500, 5000)
    p.room_poly("scullery", "Scullery",
                [[6500, 5500], [13000, 5500], [13000, 10500], [9000, 10500],
                 [9000, 8000], [6500, 8000]])
    p.door("D1", "EXT_S", 2500, 1067, "SINGLE", "INWARD_LEFT")
    p.door("D2", "SPINE", 2750, 838, "SINGLE", "INWARD_RIGHT")
    p.door("D3", "P1", 1500, 762, "SINGLE", "INWARD_LEFT")
    p.door("D4", "P2", 3250, 813, "SINGLE", "INWARD_LEFT")
    p.window("W1", "EXT_S", 5500, 720, "CASEMENT")
    p.window("W2", "EXT_S", 9500, 720, "CASEMENT")
    p.window("W3", "EXT_E", 2750, 600, "FIXED")
    p.window("W4", "EXT_N", 3250, 540, "AWNING")

    # A15 Modern light steel frame (thin walls, big windows)
    p = reg(Plan("A15", "A", "Modern light steel frame", "1:50",
                 description="16.5x9.5m steel-frame house: thin partitions, large glazing.",
                 hard_case="Thin walls (70/90) and very large glazed openings.",
                 clutter=C_LIGHT))
    p.envelope(16500, 9500, 150)
    p.wall("P1", 9000, 0, 9000, 9500, 90)
    p.wall("P2", 9000, 5500, 16500, 5500, 70)
    p.wall("P3", 12500, 5500, 12500, 9500, 70)
    p.wall("UTIL", 6000, 0, 6000, 1100, 90)
    p.room("living", "Living/Kitchen", 0, 0, 9000, 9500)
    p.room("bed1", "Master", 9000, 0, 7500, 5500)
    p.room("bed2", "Bedroom 2", 9000, 5500, 3500, 4000)
    p.room_poly("bath", "Bath",
                [[12500, 5500], [16500, 5500], [16500, 9500], [14000, 9500],
                 [14000, 7500], [12500, 7500]])
    p.door("D1", "EXT_S", 2500, 1000, "SINGLE", "INWARD_LEFT")
    p.door("D2", "P1", 2750, 826, "SINGLE", "INWARD_LEFT")
    p.door("D3", "P2", 1750, 762, "SINGLE", "INWARD_RIGHT")
    p.door("D4", "P3", 2000, 686, "SINGLE", "INWARD_LEFT")
    p.door("DSLIDE", "EXT_N", 4500, 2400, "SLIDING", None, 0)
    p.window("WC_a", "EXT_W", 5000, 2100, "CORNER")
    p.window("WC_b", "EXT_S", 2000, 2100, "CORNER")
    p.window("W1", "EXT_E", 2750, 1810, "FIXED")
    p.window("W2", "P2", 5000, 1100, "AWNING")


def all_full() -> List[Dict]:
    if not any(REGISTRY.values()):
        build_A()
    out = []
    for g in ("A", "B", "C", "D", "E"):
        for p in REGISTRY[g]:
            out.append(p.config())
    return out
