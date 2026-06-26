"""Shared prompt + schema (identical for SFT and GRPO).

Keeping the prompt identical across stages matters: GRPO must see the same prompt
distribution it was supervised on. We also enumerate the allowed room labels so the
model produces the unified taxonomy (lower entropy, faster/cleaner learning).
"""
from .taxonomy import UNIFIED_LABELS

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
