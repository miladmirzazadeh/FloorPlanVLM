"""Shared prompt + schema (identical for SFT and GRPO).

Two modes:
  * full       — walls (with nested openings) + rooms (label + wall refs)
  * WALLS_ONLY — only walls (a downstream VLM does rooms/openings/semantics).
                 Shorter outputs -> faster/cheaper, higher validity, better walls.
Keep the prompt identical across SFT and GRPO so the RL stage matches what it was
supervised on.
"""
from . import config
from .taxonomy import UNIFIED_LABELS

if config.WALLS_ONLY:
    SYSTEM_PROMPT = (
        "You are a floor plan wall-extraction expert. From the floor plan image, extract "
        "ONLY the walls as structured JSON.\n\n"
        "Output ONLY valid JSON with this schema:\n"
        '{"walls":[{"id":"wall_N","start":[x,y],"end":[x,y],"thickness":T,"curvature":0}]}\n\n'
        "curvature is 0 for a straight wall and a nonzero signed value for a curved wall. "
        "Coordinates are normalized so the longer image edge = 1024."
    )
    USER_PROMPT = "Extract all walls from this floor plan as structured JSON."
else:
    _ALLOWED = ", ".join(UNIFIED_LABELS)
    SYSTEM_PROMPT = (
        "You are a floor plan vectorization expert. Extract wall, door, window geometry "
        "from floor plan images into structured JSON.\n\n"
        "Output ONLY valid JSON with this schema:\n"
        '{"walls":[{"id":"wall_N","start":[x,y],"end":[x,y],"thickness":T,"curvature":0,'
        '"openings":[{"type":"door"|"window","center":D,"width":W}]}],'
        '"rooms":[{"label":"room_type","walls":["wall_N",...]}]}\n\n'
        "Coordinates normalized so longer image edge = 1024.\n"
        f"Use only these room labels: {_ALLOWED}."
    )
    USER_PROMPT = (
        "Vectorize this floor plan into structured JSON with all walls, doors, windows, and rooms."
    )
