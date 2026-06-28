"""Drawing standards: layers, line weights, scale-driven detail levels.

Layer names follow the AIA / US National CAD Standard convention
(discipline-major group). Component geometry is strictly confined to the
A-DOOR / A-GLAZ / A-COLS layers and carries NO text. Everything that a
real drawing is cluttered with - dimensions, room labels, the word "PLAN",
furniture, hatching, grid bubbles, title block - lives on dedicated
annotation layers so it can be toggled and so the "no component label"
guarantee is mechanically checkable (no TEXT on a component layer).
"""

from __future__ import annotations

# ---- layer names -----------------------------------------------------------
LAY_WALL_FULL = "A-WALL-FULL"   # exterior / full-thickness walls
LAY_WALL_INTR = "A-WALL-INTR"   # interior / partition walls
LAY_DOOR = "A-DOOR"
LAY_GLAZ = "A-GLAZ"
LAY_COLS = "A-COLS"
LAY_WALL_PATT = "A-WALL-PATT"   # wall hatching (clutter)
LAY_ANNO_DIMS = "A-ANNO-DIMS"   # dimension lines (clutter)
LAY_ANNO_TEXT = "A-ANNO-TEXT"   # room labels, "PLAN", notes (clutter, distractors)
LAY_ANNO_TTLB = "A-ANNO-TTLB"   # title block (clutter)
LAY_FURN = "A-FURN"             # furniture (clutter)
LAY_GRID = "A-GRID"             # structural grid + bubbles (clutter)
LAY_MISC = "A-MISC"             # stray noise lines (clutter)

# Component layers must never contain text. Used by the validator.
COMPONENT_LAYERS = (LAY_DOOR, LAY_GLAZ, LAY_COLS, LAY_WALL_FULL, LAY_WALL_INTR)

# ACI color 7 = black-on-white / white-on-black (monochrome safe).
# Line weights are in 1/100 mm (ezdxf convention): 50 == 0.50 mm.
# (key, aci_color, lineweight_1_100mm, linetype)
_LAYER_TABLE = [
    (LAY_WALL_FULL, 7, 50, "CONTINUOUS"),
    (LAY_WALL_INTR, 7, 35, "CONTINUOUS"),
    (LAY_DOOR,      7, 25, "CONTINUOUS"),
    (LAY_GLAZ,      7, 18, "CONTINUOUS"),
    (LAY_COLS,      7, 35, "CONTINUOUS"),
    (LAY_WALL_PATT, 8, 9,  "CONTINUOUS"),
    (LAY_ANNO_DIMS, 7, 13, "CONTINUOUS"),
    (LAY_ANNO_TEXT, 7, 18, "CONTINUOUS"),
    (LAY_ANNO_TTLB, 7, 25, "CONTINUOUS"),
    (LAY_FURN,      7, 13, "CONTINUOUS"),
    (LAY_GRID,      7, 9,  "DASHED"),
    (LAY_MISC,      7, 13, "CONTINUOUS"),
]

# global multiplier applied to every lineweight at render time
LINE_WEIGHT_STYLES = {
    "light": 0.6,
    "standard": 1.0,
    "heavy": 1.6,
}


def register_layers(doc) -> None:
    """Create every layer used by the engine on an ezdxf document."""
    needed_linetypes = {lt for *_, lt in _LAYER_TABLE}
    for lt in needed_linetypes:
        if lt != "CONTINUOUS" and lt not in doc.linetypes:
            # setup=True on ezdxf.new already loads the standard linetypes,
            # but guard in case a caller built the doc differently.
            try:
                doc.linetypes.add(lt, pattern="A,.5,-.25")
            except Exception:
                pass
    for name, color, lw, lt in _LAYER_TABLE:
        if name in doc.layers:
            layer = doc.layers.get(name)
        else:
            layer = doc.layers.add(name)
        layer.color = color
        layer.dxf.lineweight = lw
        if lt in doc.linetypes:
            layer.dxf.linetype = lt


# ---- scale / detail level ---------------------------------------------------
DETAIL_FULL = "full"     # 1:50 and finer - frames, shutters, sills, hinges
DETAIL_MED = "med"       # 1:100 - simplified frames + symbol
DETAIL_MIN = "min"       # 1:200 - gap + minimal symbol


def scale_denominator(scale: str) -> int:
    """'1:100' -> 100."""
    try:
        return int(str(scale).split(":")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Bad scale string: {scale!r} (expected '1:50' etc.)")


def detail_for_scale(scale: str) -> str:
    d = scale_denominator(scale)
    if d <= 60:
        return DETAIL_FULL
    if d <= 150:
        return DETAIL_MED
    return DETAIL_MIN


# ---- wall fill (poche) styles ----------------------------------------------
# A wall fill is a tuple (kind, angles, spacing_mm):
#   kind "plain" -> outline only (no fill)
#   kind "solid" -> solid poche
#   kind "lines" -> parallel hatch lines at each angle in ``angles`` (deg),
#                   spaced ``spacing_mm`` apart, generated as real line
#                   segments CLIPPED to the wall polygon (so holes/rooms are
#                   never flooded and it renders identically in every backend).
# The engine picks ONE exterior fill and ONE (lighter) interior fill per plan,
# seeded by plan id, so plans differ and exterior reads differently from
# interior. Patterns are scale-filtered: finer hatch at 1:50, poche/plain at
# 1:200 (fine lines are unreadable when the plan is that small).
PLAIN_FILL = ("plain", None, 0.0)
SOLID_FILL = ("solid", None, 0.0)


def wall_fill_palette(detail: str):
    """Return (exterior_palette, interior_palette) for the given detail level.
    Each palette is a non-empty list of fill tuples to choose from."""
    if detail == DETAIL_MIN:                       # 1:200 - bold and simple
        sp = 90.0
        ext = [SOLID_FILL, SOLID_FILL, ("lines", [45], sp)]
        intr = [PLAIN_FILL, SOLID_FILL]
    elif detail == DETAIL_MED:                     # 1:100
        sp = 70.0
        ext = [("lines", [45], sp), ("lines", [0, 90], sp),
               ("lines", [135], sp), ("lines", [45, 135], sp), SOLID_FILL]
        intr = [PLAIN_FILL, ("lines", [90], sp), ("lines", [45], sp), PLAIN_FILL]
    else:                                          # DETAIL_FULL 1:50 - finest
        sp = 45.0
        ext = [("lines", [0, 90], sp), ("lines", [45], sp),
               ("lines", [45, 135], sp), ("lines", [135], sp), ("lines", [0], sp)]
        intr = [PLAIN_FILL, ("lines", [90], sp), ("lines", [45], sp),
                ("lines", [0], sp)]
    return ext, intr
