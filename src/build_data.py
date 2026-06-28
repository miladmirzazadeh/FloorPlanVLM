"""Build the SFT corpus.

Reuses the validated CubiCasa/synth/MSD wall parsers, then RE-ENCODES each plan's walls
through the new pipeline — pad-to-square + [0,GRID] normalization, endpoint ordering,
exterior→interior sort, curvature, minified/abbreviated/nested/count-anchored target —
and writes one JSONL line per sample:

    {"image": "/abs/path.png", "target": "{\\"n\\":14,\\"walls\\":[...]}", "source": "cubicasa"}

    python -m src.build_data                 # -> built/train.jsonl, built/val.jsonl
    python -m src.validate_roundtrip --built built/train.jsonl --out rt_check --n 40

Negative samples and augmentation are off by default (config.NEG_SAMPLE_FRAC / AUGMENT).
"""
import os
import sys
import json
import random

from PIL import Image

from . import config, schema
from .normalize import canonicalize


def _annotations(name):
    """(image_path, json_annotation) per plan from the reused, validated parsers."""
    name = name.lower()
    if name == "cubicasa":
        from .data import _build, download_and_extract
        _, anns = _build(download_and_extract(),
                         config.CUBICASA_MAX_SAMPLES or config.MAX_SAMPLES, want_records=False)
    elif name == "msd":
        from .data_msd import build_msd_records
        _, anns = build_msd_records(config.MSD_DIR, config.MSD_MAX_SAMPLES, want_records=False)
    elif name in ("synth", "synth-floorseg", "synthfloorseg"):
        from .data_synth import build_synth_records
        _, anns = build_synth_records(config.SYNTH_DIR, config.SYNTH_MAX_SAMPLES, want_records=False)
    else:
        raise ValueError(f"unknown dataset '{name}' (cubicasa, msd, synth)")
    return anns


def _encode(ann):
    """Old verbose annotation -> new minified target; returns {image,target} or None."""
    try:
        with Image.open(ann["image_path"]) as im:
            w, h = im.size
        jd = json.loads(ann["json_annotation"])
    except Exception:
        return None
    walls = canonicalize(jd.get("walls", []), w, h)
    if not walls:
        return None
    tgt = schema.encode(walls)
    # self-check: every target must decode to the same wall count
    if len(schema.decode(tgt)) != len(walls):
        return None
    return {"image": os.path.abspath(ann["image_path"]), "target": tgt}


def _negatives(n, like):
    """Optional empty/garbage samples -> empty target. Off unless NEG_SAMPLE_FRAC>0."""
    if n <= 0 or not like:
        return []
    import numpy as np
    os.makedirs(os.path.join(config.BUILT_DATA, "neg"), exist_ok=True)
    out = []
    rng = random.Random(1)
    for i in range(n):
        side = 768
        kind = i % 3
        if kind == 0:
            arr = np.full((side, side, 3), 255, np.uint8)                 # blank page
        elif kind == 1:
            arr = (np.clip(np.random.default_rng(i).normal(127, 60, (side, side, 3)),
                           0, 255)).astype("uint8")                       # noise
        else:
            arr = np.full((side, side, 3), rng.randint(0, 255), np.uint8)  # solid color
        p = os.path.join(config.BUILT_DATA, "neg", f"neg_{i:05d}.png")
        Image.fromarray(arr).save(p)
        out.append({"image": os.path.abspath(p), "target": schema.empty_target(), "source": "neg"})
    return out


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build():
    recs = []
    for name in config.DATASETS:
        n0 = len(recs)
        for ann in _annotations(name):
            r = _encode(ann)
            if r:
                r["source"] = name
                recs.append(r)
        print(f"[build] {name}: +{len(recs) - n0} samples")

    if not recs:
        sys.exit("[build] no samples — check dataset paths (DATA_DIR/MSD_DIR/SYNTH_DIR).")

    if config.NEG_SAMPLE_FRAC > 0:
        nneg = int(len(recs) * config.NEG_SAMPLE_FRAC / (1 - config.NEG_SAMPLE_FRAC))
        recs += _negatives(nneg, like=recs)
        print(f"[build] negatives: +{nneg}")

    random.Random(0).shuffle(recs)
    k = max(1, int(len(recs) * config.EVAL_RATIO))
    val, train = recs[:k], recs[k:]

    os.makedirs(config.BUILT_DATA, exist_ok=True)
    _write(os.path.join(config.BUILT_DATA, "train.jsonl"), train)
    _write(os.path.join(config.BUILT_DATA, "val.jsonl"), val)

    # stats
    nwalls = [len(schema.decode(r["target"])) for r in recs]
    tgtlen = [len(r["target"]) for r in recs]
    nwalls.sort(); tgtlen.sort()
    med = lambda a: a[len(a) // 2] if a else 0
    print(f"\n[build] total {len(recs)}  ->  train {len(train)} / val {len(val)}  in {config.BUILT_DATA}/")
    print(f"[build] walls/plan: median {med(nwalls)}  max {nwalls[-1]}")
    print(f"[build] target chars: median {med(tgtlen)}  max {tgtlen[-1]}  (≈chars/4 tokens; keep < MAX_SEQ_LEN minus image)")
    print(f"[build] NEXT: python -m src.validate_roundtrip --built {config.BUILT_DATA}/train.jsonl --out rt_check --n 40")
    return train, val


if __name__ == "__main__":
    config.banner("build_data")
    build()
