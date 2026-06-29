"""Run the trained SFT adapter on a FOLDER of floor-plan images (no ground truth).

For each image it draws the model's PREDICTED walls (red) + openings (green=door,
blue=window) on top of the original plan, so you can eyeball accuracy. Also saves the
raw JSON the model produced.

  python scripts/infer.py --adapter miladmirza/floorplan-vlm-sft2 --images samples --out infer_out
"""
import argparse
import glob
import json
import os
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, schema                       # noqa: E402
from src.prompts import SYSTEM_PROMPT, USER_PROMPT   # noqa: E402
from src.normalize import arc_polyline               # noqa: E402

try:
    from transformers import Qwen3VLForConditionalGeneration as VLM
except Exception:
    from transformers import AutoModelForImageTextToText as VLM
from transformers import AutoProcessor               # noqa: E402
from peft import PeftModel                           # noqa: E402


def load_model(adapter):
    proc = AutoProcessor.from_pretrained(
        config.MODEL_ID, min_pixels=config.IMG_MIN_PIXELS, max_pixels=config.IMG_MAX_PIXELS)
    model = VLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter and adapter.lower() not in ("none", "base"):
        print(f"[infer] adapter {adapter}")
        model = PeftModel.from_pretrained(model, adapter, token=config.HF_TOKEN or None)
    return model.eval(), proc


@torch.no_grad()
def generate(model, proc, img, max_new_tokens):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def overlay(base, walls, scale):
    """Draw predicted walls (red) + openings (green door / blue window) on a copy of base."""
    from shapely.geometry import LineString
    im = base.copy()
    d = ImageDraw.Draw(im)
    for w in walls:
        pts = arc_polyline(w["cl"], w.get("cv", 0))
        spts = [(x * scale, y * scale) for x, y in pts]
        if len(spts) >= 2:
            d.line(spts, fill=(220, 0, 0), width=max(2, int(w.get("th", 4) * scale)))
        if w.get("op") and len(pts) >= 2:
            try:
                ln = LineString(pts)
                for op in w["op"]:
                    pt = ln.interpolate(float(op.get("c", 0)))
                    ox, oy = pt.x * scale, pt.y * scale
                    col = (0, 160, 0) if op.get("t") == "door" else (0, 90, 255)
                    d.ellipse([ox - 6, oy - 6, ox + 6, oy + 6], fill=col)
            except Exception:
                pass
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=config.REPO_SFT, help="HF repo or local path; 'base' = no adapter")
    ap.add_argument("--images", default="samples", help="folder of plan images")
    ap.add_argument("--out", default="infer_out")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG")
                   for p in glob.glob(os.path.join(a.images, ext)))
    print(f"[infer] {len(paths)} images in '{a.images}'  adapter={a.adapter}")
    model, proc = load_model(a.adapter)

    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img = Image.open(p).convert("RGB")
        W, H = img.size
        scale = max(W, H) / config.GRID
        raw = generate(model, proc, img, a.max_new_tokens)
        walls = schema.decode(raw)
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            f.write(raw)
        overlay(img, walls, scale).save(os.path.join(a.out, f"{name}_pred.png"))
        try:
            obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except Exception:
            obj = {}
        nop = sum(len(w.get("op", w.get("openings", []))) for w in obj.get("walls", []))
        print(f"  {name}: walls={len(walls)}  rooms={len(obj.get('rooms', []))}  openings={nop}")

    print(f"[infer] done -> {a.out}/   (<name>_pred.png = walls on your plan, <name>.json = raw output)")


if __name__ == "__main__":
    main()
