"""The three Session-1 test scenarios.

These are hand-laid coordinate layouts (mm, y-up) used to exercise the engine
end to end. They are NOT a general scenario format - real datasets come later
as separate configs. Every opening listed in the brief is reproduced, and
clutter is dialled differently per scenario so the engine's
"clutter + rendering randomisation" capability is visibly exercised while the
"no opening labels in the image" guarantee still holds.

Wall thickness convention: exterior 300 mm, interior 150 mm, wet (bathroom/WC)
200 mm.
"""

from __future__ import annotations

import copy
from typing import Dict, List

EXT = 300.0
INT = 150.0
WET = 200.0


def _ext(wid, x1, y1, x2, y2):
    return {"id": wid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "thickness": EXT, "wall_type": "EXTERIOR"}


def _wall(wid, x1, y1, x2, y2, thickness=INT, wtype="INTERIOR"):
    return {"id": wid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "thickness": thickness, "wall_type": wtype}


# ---------------------------------------------------------------------------
def simple_studio() -> Dict:
    """6 x 8 m studio. Living spans the south band (touches W/E/S walls);
    kitchen / hallway / bathroom occupy the north band."""
    W, H = 6000.0, 8000.0
    walls = [
        _ext("S", 0, 0, W, 0),
        _ext("E", W, 0, W, H),
        _ext("N", W, H, 0, H),
        _ext("Wt", 0, H, 0, 0),
        _wall("H1", 0, 5000, W, 5000, INT),          # living | north band
        _wall("V1", 3000, 5000, 3000, H, INT),       # kitchen | hall/bath
        _wall("V2", 4500, 5000, 4500, H, WET),       # hallway | bathroom
    ]
    rooms = [
        {"id": "living", "name": "Living Room", "x": 0, "y": 0, "width": 6000, "height": 5000},
        {"id": "kitchen", "name": "Kitchen", "x": 0, "y": 5000, "width": 3000, "height": 3000},
        {"id": "hall", "name": "Hallway", "x": 3000, "y": 5000, "width": 1500, "height": 3000},
        {"id": "bath", "name": "Bathroom", "x": 4500, "y": 5000, "width": 1500, "height": 3000},
    ]
    openings = [
        # doors
        {"id": "D1", "type": "door", "wall_id": "S", "position_along_wall": 0.5,
         "subtype": "SINGLE", "clear_opening_mm": 900, "swing": "INWARD_LEFT",
         "max_swing_angle_deg": 90},
        {"id": "D2", "type": "door", "wall_id": "V2", "position_along_wall": 0.333,
         "subtype": "SINGLE", "clear_opening_mm": 800, "swing": "INWARD_LEFT",
         "max_swing_angle_deg": 90},
        {"id": "D3", "type": "door", "wall_id": "H1", "position_along_wall": 0.25,
         "subtype": "SLIDING", "clear_opening_mm": 1500, "swing": None,
         "max_swing_angle_deg": 0},
        # windows
        {"id": "W1", "type": "window", "wall_id": "Wt", "position_along_wall": 0.8125,
         "subtype": "CASEMENT", "clear_opening_mm": 1200},
        {"id": "W2", "type": "window", "wall_id": "Wt", "position_along_wall": 0.5625,
         "subtype": "CASEMENT", "clear_opening_mm": 1200},
        {"id": "W3", "type": "window", "wall_id": "N", "position_along_wall": 0.75,
         "subtype": "FIXED", "clear_opening_mm": 1500},
        {"id": "W4", "type": "window", "wall_id": "E", "position_along_wall": 0.3125,
         "subtype": "SLIDING", "clear_opening_mm": 1800},
    ]
    return {
        "plan_id": "simple_studio",
        "scenario": "simple_studio",
        "scale": "1:100",
        "rotation_deg": 0.0,
        "rooms": rooms,
        "walls": walls,
        "openings": openings,
        "columns": [],
        "clutter": {"room_labels": True, "dimensions": True, "title_block": True,
                    "furniture": True, "hatch_walls": False, "grid": False,
                    "noise_lines": 2},
        "render": {"dpi": 150, "line_weight_style": "standard", "monochrome": True},
    }


# ---------------------------------------------------------------------------
def office_floor() -> Dict:
    """16 x 8 m office floor. Open office on the left; meeting / reception / WC
    in the right band. Heavier clutter (grid + hatch + noise)."""
    W, H = 16000.0, 8000.0
    walls = [
        _ext("S", 0, 0, W, 0),
        _ext("E", W, 0, W, H),
        _ext("N", W, H, 0, H),
        _ext("Wt", 0, H, 0, 0),
        _wall("PV", 12000, 0, 12000, H, INT),          # open office | right band
        _wall("PH", 12000, 4000, W, 4000, INT),        # meeting | bottom band
        _wall("PV2", 14000, 0, 14000, 4000, INT),      # WC/lobby | reception
        _wall("PH2", 12000, 2000, 14000, 2000, WET),   # WC | lobby
    ]
    rooms = [
        {"id": "office", "name": "Open Office", "x": 0, "y": 0, "width": 12000, "height": 8000},
        {"id": "meeting", "name": "Meeting Room", "x": 12000, "y": 4000, "width": 4000, "height": 4000},
        {"id": "reception", "name": "Reception", "x": 14000, "y": 0, "width": 2000, "height": 4000},
        {"id": "wc", "name": "WC", "x": 12000, "y": 0, "width": 2000, "height": 2000},
    ]
    openings = [
        # doors: 3 single + 1 double (main entry)
        {"id": "D1", "type": "door", "wall_id": "S", "position_along_wall": 0.9375,
         "subtype": "DOUBLE", "clear_opening_mm": 1600, "swing": "INWARD_LEFT",
         "max_swing_angle_deg": 90},
        {"id": "D2", "type": "door", "wall_id": "PV", "position_along_wall": 0.375,
         "subtype": "SINGLE", "clear_opening_mm": 900, "swing": "INWARD_RIGHT",
         "max_swing_angle_deg": 90},
        {"id": "D3", "type": "door", "wall_id": "PH", "position_along_wall": 0.25,
         "subtype": "SINGLE", "clear_opening_mm": 900, "swing": "INWARD_LEFT",
         "max_swing_angle_deg": 90},
        {"id": "D4", "type": "door", "wall_id": "PH2", "position_along_wall": 0.5,
         "subtype": "SINGLE", "clear_opening_mm": 800, "swing": "INWARD_LEFT",
         "max_swing_angle_deg": 90},
        # windows: 6 fixed facade + 1 sliding (meeting)
        {"id": "W1", "type": "window", "wall_id": "S", "position_along_wall": 0.125,
         "subtype": "FIXED", "clear_opening_mm": 1500},
        {"id": "W2", "type": "window", "wall_id": "S", "position_along_wall": 0.3125,
         "subtype": "FIXED", "clear_opening_mm": 1500},
        {"id": "W3", "type": "window", "wall_id": "S", "position_along_wall": 0.5,
         "subtype": "FIXED", "clear_opening_mm": 1500},
        {"id": "W4", "type": "window", "wall_id": "Wt", "position_along_wall": 0.75,
         "subtype": "FIXED", "clear_opening_mm": 1500},
        {"id": "W5", "type": "window", "wall_id": "Wt", "position_along_wall": 0.25,
         "subtype": "FIXED", "clear_opening_mm": 1500},
        {"id": "W6", "type": "window", "wall_id": "N", "position_along_wall": 0.625,
         "subtype": "FIXED", "clear_opening_mm": 1500},
        {"id": "W7", "type": "window", "wall_id": "E", "position_along_wall": 0.75,
         "subtype": "SLIDING", "clear_opening_mm": 1800},
    ]
    return {
        "plan_id": "office_floor",
        "scenario": "office_floor",
        "scale": "1:100",
        "rotation_deg": 0.0,
        "rooms": rooms,
        "walls": walls,
        "openings": openings,
        "columns": [
            {"id": "C1", "x": 4000, "y": 4000, "shape": "SQUARE", "size_mm": 300},
            {"id": "C2", "x": 8000, "y": 4000, "shape": "ROUND", "size_mm": 250},
        ],
        "clutter": {"room_labels": True, "dimensions": True, "title_block": True,
                    "furniture": True, "hatch_walls": True, "grid": True,
                    "noise_lines": 4},
        "render": {"dpi": 150, "line_weight_style": "heavy", "monochrome": True},
    }


# ---------------------------------------------------------------------------
def door_in_corner() -> Dict:
    """simple_studio at 1:50 with the bathroom door 200 mm from the V2/H1
    interior corner and its swing physically constrained to 70 degrees."""
    cfg = copy.deepcopy(simple_studio())
    cfg["plan_id"] = "door_in_corner"
    cfg["scenario"] = "door_in_corner"
    cfg["scale"] = "1:50"
    # V2 runs (4500,5000)->(4500,8000); the interior corner with H1 is at y=5000.
    # clear 800 -> rough opening 940 -> near jamb 200 mm above the corner means
    # centre at 5000 + 200 + 470 = 5670 -> t = 670/3000.
    for op in cfg["openings"]:
        if op["id"] == "D2":
            op["position_along_wall"] = round(670.0 / 3000.0, 4)
            op["max_swing_angle_deg"] = 70
    # light clutter so the constrained swing arc is clearly inspectable
    cfg["clutter"] = {"room_labels": True, "dimensions": False, "title_block": False,
                      "furniture": False, "hatch_walls": False, "grid": False,
                      "noise_lines": 0}
    cfg["render"] = {"dpi": 150, "line_weight_style": "standard", "monochrome": True}
    return cfg


def all_scenarios() -> List[Dict]:
    return [simple_studio(), office_floor(), door_in_corner()]
