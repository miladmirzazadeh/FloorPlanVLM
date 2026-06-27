"""Fast batch inference over a folder of floor-plan images.

For each image -> writes <name>.json (parsed prediction) and, if walls were produced,
<name>_overlay.png (predicted walls drawn on the 1024-normalized image). Minimal deps
(no trl/datasets/opencv) so it installs/runs quickly on a fresh pod.

  python -m src.batch_infer --images test_images --out eval_results \
      [--adapter mudasir13cs/qwen25-vl-3b-floorplan-grpo]   # default = community GRPO model

Notes:
- Uses the prompt from src/prompts.py — leave WALLS_ONLY unset to match the community
  full-schema adapter; set WALLS_ONLY=1 when testing your own walls-only model.
- Coords are model-normalized to longest-edge 1024, so the overlay resizes the image to
  longest-edge 1024 before drawing.
"""
import os
import sys
import json
import re
import glob
import time
import argparse

import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from . import config
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .geometry import wall_polyline
from .rewards import extract_json   # robust (salvages truncated/looping output)

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples")
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--adapter", default="mudasir13cs/qwen25-vl-3b-floorplan-grpo",
                    help="HF repo or local path; 'base' for no adapter")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--repetition-penalty", type=float, default=1.1,
                    help="soft logit penalty on repeats; 1.1 breaks the greedy loop without "
                         "banning the legitimately-repeated JSON key tokens")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    imgs = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                  if p.lower().endswith(EXTS))
    if not imgs:
        print(f"[batch] no images found under {a.images}/")
        sys.exit(1)
    print(f"[batch] {len(imgs)} images | base={config.MODEL_ID} | adapter={a.adapter}")

    proc = AutoProcessor.from_pretrained(
        config.MODEL_ID, min_pixels=config.IMG_MIN_PIXELS, max_pixels=config.IMG_MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID, torch_dtype="auto", device_map="auto")
    if a.adapter and a.adapter.lower() not in ("none", "base"):
        model = PeftModel.from_pretrained(model, a.adapter, token=config.HF_TOKEN or None)
    model.eval()

    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    summary = []
    for i, p in enumerate(imgs):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            img = Image.open(p).convert("RGB")
            inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                    repetition_penalty=a.repetition_penalty)
            gen = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            jd = extract_json(gen)
            nwalls = len(jd.get("walls", [])) if jd else 0
            dt = time.time() - t0

            with open(os.path.join(a.out, f"{name}.json"), "w") as f:
                json.dump({"image": os.path.basename(p), "valid_json": jd is not None,
                           "n_walls": nwalls, "prediction": jd,
                           "raw": None if jd is not None else gen}, f, indent=2)

            if jd and jd.get("walls"):
                s = 1024.0 / max(img.size)
                rs = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s)))).convert("RGB")
                d = ImageDraw.Draw(rs)
                for w in jd["walls"]:
                    try:
                        d.line(wall_polyline(w), fill=(255, 0, 0), width=3)
                    except Exception:
                        pass
                rs.save(os.path.join(a.out, f"{name}_overlay.png"))

            summary.append({"image": name, "valid": jd is not None, "n_walls": nwalls, "sec": round(dt, 1)})
            print(f"[batch] {i + 1}/{len(imgs)} {name}: {'OK' if jd else 'INVALID-JSON'} "
                  f"walls={nwalls} ({dt:.1f}s)")
        except Exception as e:
            print(f"[batch] {i + 1}/{len(imgs)} {name}: FAILED {e}")
            summary.append({"image": name, "valid": False, "n_walls": 0, "sec": 0})

    with open(os.path.join(a.out, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ok = sum(1 for s in summary if s["valid"])
    print(f"\n[batch] DONE — valid {ok}/{len(summary)}. Results + overlays in {a.out}/")


if __name__ == "__main__":
    main()
