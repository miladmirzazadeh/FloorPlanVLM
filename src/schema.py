"""The training target format — one place that defines, emits, and parses it.

Default (ABBREVIATE=True) — minified, abbreviated, nested, count-anchored:

  {"n":14,"walls":[{"id":1,"cl":[x1,y1,x2,y2],"th":12,"op":[{"t":"door","c":126,"w":27}]}, ...]}

  n   total wall count (lightweight CoT — declared before the array)
  id  1..N, in canonical order (exterior clockwise, then interior TL→BR)
  cl  centerline [x1,y1,x2,y2], integers in [0,GRID], x1<=x2 (tie y1<=y2)
  th  wall thickness (same [0,GRID] scale)
  op  openings nested in their wall; t=door|window, c=offset along cl from (x1,y1), w=width
      (omitted entirely when the wall has none)

Negative/empty sample target:  {"n":0,"walls":[]}

decode() accepts both the abbreviated keys and the verbose ones (cl|centerline,
th|thickness, op|openings, t|type, c|center, w|width) so inference output is parsed
robustly regardless of minor drift.
"""
import json

from . import config


def encode(walls):
    """Canonical walls (from normalize.canonicalize) -> minified target string."""
    ab = config.ABBREVIATE
    k_cl, k_th, k_op = ("cl", "th", "op") if ab else ("centerline", "thickness", "openings")
    k_cv = "cv" if ab else "curvature"
    k_t, k_c, k_w = ("t", "c", "w") if ab else ("type", "center", "width")
    k_n = "n" if ab else "total_walls"

    objs = []
    for i, w in enumerate(walls, 1):
        o = {"id": i, k_cl: [int(v) for v in w["cl"]], k_th: int(w["th"])}
        cv = float(w.get("cv", 0) or 0)
        if config.CURVATURE and abs(cv) >= config.CURVE_EPS:
            o[k_cv] = round(cv, 3)                       # curved walls only
        if config.NEST_OPENINGS and w.get("op"):
            o[k_op] = [{k_t: op["t"], k_c: int(op["c"]), k_w: int(op["w"])} for op in w["op"]]
        objs.append(o)

    doc = {}
    if config.COUNT_ANCHOR:
        doc[k_n] = len(objs)
    doc["walls"] = objs
    return json.dumps(doc, separators=(",", ":"))   # minified: no spaces/newlines


EMPTY = None  # set lazily so it follows config at call time


def empty_target():
    """Target for negative/empty samples."""
    return encode([])


def decode(text):
    """Parse a target/prediction string back to canonical walls (robust to key style)."""
    text = (text or "").strip()
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group())
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return []

    out = []
    for w in obj.get("walls", []):
        cl = w.get("cl", w.get("centerline"))
        if not (isinstance(cl, list) and len(cl) == 4):
            continue
        th = w.get("th", w.get("thickness", 1))
        cv = w.get("cv", w.get("curvature", 0)) or 0
        ops = []
        for op in (w.get("op", w.get("openings")) or []):
            ops.append({
                "t": op.get("t", op.get("type", "door")),
                "c": op.get("c", op.get("center", 0)),
                "w": op.get("w", op.get("width", 0)),
            })
        out.append({"cl": [int(round(v)) for v in cl], "th": int(round(th)),
                    "cv": float(cv), "op": ops})
    return out


def schema_doc():
    """Exact schema text embedded in the (static) system prompt — kept in sync with encode()."""
    g = config.GRID
    if config.ABBREVIATE:
        body = (
            '{"n":N,"walls":[{"id":1,"cl":[x1,y1,x2,y2],"th":T,"cv":0,'
            '"op":[{"t":"door"|"window","c":C,"w":W}]}]}'
        )
        keys = ("n=number of walls; id=1..N; cl=centerline [x1,y1,x2,y2]; th=thickness; "
                "cv=curvature (0 for straight walls, a small signed value for curved walls; "
                "omit when 0); op=openings (omit if none); t=type; "
                "c=offset along cl from the first point; w=width.")
    else:
        body = ('{"total_walls":N,"walls":[{"id":1,"centerline":[x1,y1,x2,y2],"thickness":T,'
                '"curvature":0,"openings":[{"type":"door"|"window","center":C,"width":W}]}]}')
        keys = "centerline=[x1,y1,x2,y2]; curvature 0 unless curved; openings omitted if none."
    return (
        f"Output ONLY one line of minified JSON (no spaces, no newlines) with this schema:\n{body}\n"
        f"{keys}\n"
        f"All coordinates are integers in [0,{g}], normalized so the longer image edge = {g}.\n"
        f"Centerlines are ordered x1<=x2 (if vertical, y1<=y2). List the exterior boundary "
        f"walls first (clockwise), then interior walls top-left to bottom-right. "
        f'If the image is not a readable floor plan, output {{"n":0,"walls":[]}}.'
    )
