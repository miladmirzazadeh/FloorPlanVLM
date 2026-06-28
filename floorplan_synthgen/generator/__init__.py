"""Synthetic floor plan generator (core engine).

Session 1: core infrastructure. Produces architectural-looking floor plan
PNGs with NO door/window annotations in the image, plus separate label
files (YOLO .txt + rich JSON) that carry the exact opening geometry.
"""

__all__ = [
    "geometry",
    "components",
    "layout",
    "renderer",
    "exporter",
    "scenarios",
]
