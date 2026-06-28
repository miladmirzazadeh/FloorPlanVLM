"""ArchCAD (HF: jackluoluo/ArchCAD) -> (image_path, json_annotation) records.  [STUB]

ArchCAD is a cc-by-nc-4.0 panoptic-CAD dataset (research use only). Extracted locally to
ARCHCAD_DIR with `json/` and `svg/` (~41k drawings each).

CONFIRMED schema (from samples): json = {"entities":[...]}, each entity:
  type: LINE|ARC|CIRCLE ; start[x,y], end[x,y] ; line_width ; rgb ; semantic:<class id>
  (ARC/CIRCLE also have center,radius,start_angle,end_angle ; some have `instance`)
  Coords are CAD units, **Y-up** (flip for image space).
CLASS MAP (verified by rendering): **semantic 20 = WALLS** (drawn as DOUBLE parallel lines
= the wall band). 100 = structural/dimension grid (NOT walls). 6 = stairs. 19/1 = openings/
symbols. Openings carry `instance`.

TODO to implement:
  1. keep entities with semantic==20; they are double lines (the two faces of each wall).
  2. PAIR them: group near-parallel segments separated by a small offset -> centerline =
     midline, thickness = perpendicular gap. (Singletons / unpaired -> drop or treat as
     thin walls.) Handle ARC walls via center/radius -> curvature.
  3. map opening classes (19, etc.) to op on the nearest wall (need the exact opening ids).
  4. align to the matching png in ARCHCAD_DIR/png (or render the lines) -> emit raw walls
     {start,end,thickness,curvature,openings} in PIXEL space -> build_data re-encodes.

Not yet implemented (double-line pairing + opening-class confirmation pending).
"""
import os


def build_archcad_records(archcad_dir, max_samples=None, want_records=False):
    raise NotImplementedError(
        "ArchCAD converter not implemented yet. Format is known (semantic 20 = walls, double "
        f"lines, in {archcad_dir}/json). Implementing double-line->centerline pairing next. "
        "cc-by-nc: research use only."
    )
