"""ArchCAD (HF: jackluoluo/ArchCAD) -> (image_path, json_annotation) records.  [STUB]

ArchCAD is a GATED, cc-by-nc-4.0 panoptic-CAD dataset (research use only). Files are zips:
  data/png.zip   raster drawings
  data/svg.zip   vector drawings (line/arc primitives)
  data/json.zip  per-drawing annotations (primitives + semantic class incl. a Wall class)
  data/point.zip, data/caption.zip

TODO once access is granted (request on the HF dataset page, then `huggingface-cli login`):
  1. read one json + svg to learn the exact schema (primitive geometry + how 'Wall' is tagged);
  2. extract wall primitives; CAD walls are usually DOUBLE lines -> pair parallel segments
     into a centerline + thickness (or take the json's wall instances directly if provided);
  3. map doors/windows (their own classes) -> openings on the nearest wall;
  4. emit raw walls {start,end,thickness,curvature,openings} in the PNG's pixel space and
     return [{image_path, json_annotation}] — build_data re-encodes to the [0,GRID] schema.

Until then this raises with instructions rather than silently producing nothing.
"""
import os


def build_archcad_records(archcad_dir, max_samples=None, want_records=False):
    raise NotImplementedError(
        "ArchCAD converter not implemented yet — the dataset is gated. Request access at "
        "https://huggingface.co/datasets/jackluoluo/ArchCAD, `huggingface-cli login`, unzip "
        f"data/*.zip under ARCHCAD_DIR ({archcad_dir}), then share one json+svg so the wall "
        "schema can be wired. (cc-by-nc: research use only.)"
    )
