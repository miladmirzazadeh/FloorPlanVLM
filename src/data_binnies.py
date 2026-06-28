"""BinniesHK-AI/floorplan-vlm-sft-dataset-flat -> (image_path, json_annotation) records.

This HF dataset is ALREADY in the FloorplanVLM schema:
  columns: image / instruction / json_string
  json_string = {"walls":[{id,start,end,thickness,curvature,openings}],"rooms":[...]}
                with coords normalized so the LONGER image edge = 1024.

So we just materialize each image to disk (resized to longest-edge 1024 so its pixel
space matches the coord space) and hand the json straight to build_data's re-encoder,
which runs it through canonicalize -> the [0,GRID] minified schema like every other source.

License: cc-by-4.0 (images are likely CubiCasa-derived -> treat as NC for COMMERCIAL use).
"""
import os

from . import config

REPO = "BinniesHK-AI/floorplan-vlm-sft-dataset-flat"


def build_binnies_records(out_dir, max_samples=None, want_records=False):
    from datasets import load_dataset
    ds = load_dataset(REPO, split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    img_dir = os.path.join(out_dir, "binnies_images")
    os.makedirs(img_dir, exist_ok=True)
    anns = []
    for i, row in enumerate(ds):
        try:
            img = row["image"].convert("RGB")
            s = 1024.0 / max(img.size)                       # pixel space == coord space (longest=1024)
            if abs(s - 1.0) > 1e-3:
                img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
            p = os.path.join(img_dir, f"binnies_{i:05d}.png")
            img.save(p)
            anns.append({"image_path": os.path.abspath(p), "json_annotation": row["json_string"]})
        except Exception:
            continue
    print(f"[binnies] {len(anns)} samples -> {img_dir}")
    return [], anns
