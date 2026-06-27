"""EXACT reproduction of the model-card README inference snippet, looped over a folder.

This is verbatim what mudasir13cs's README tells you to run — no deviations in the parts
that matter:
  * processor = AutoProcessor.from_pretrained(BASE)        # README: NO min/max_pixels
  * model     = base Qwen  +  GRPO adapter ONLY            # README loading snippet
                (NOT stacked with SFT — that was my mistake)
  * SYSTEM_PROMPT = the schema-explicit string (README's explicit recommendation)
  * USER_PROMPT   = "Vectorize this floor plan ..."
  * out = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
  * raw -> re.search(r"\\{[\\s\\S]*\\}", raw) -> json.loads

Only additions: folder loop + save <name>.json and <name>_overlay.png. Nothing else.

    python scripts/infer_readme.py --images samples --out eval_readme
    python scripts/infer_readme.py --images samples --out eval_readme_sft \\
        --adapter mudasir13cs/qwen25-vl-3b-floorplan-sft   # the SFT card's snippet
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

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "mudasir13cs/qwen25-vl-3b-floorplan-grpo"

# verbatim from the README (== train_floorplan_vlm.py SYSTEM_PROMPT)
SYSTEM_PROMPT = (
    "You are a floor plan vectorization expert. Extract wall, door, window geometry "
    "from floor plan images into structured JSON.\n\n"
    "Output ONLY valid JSON with this schema:\n"
    '{"walls":[{"id":"wall_N","start":[x,y],"end":[x,y],"thickness":T,"curvature":0,'
    '"openings":[{"type":"door"|"window","center":D,"width":W}]}],'
    '"rooms":[{"label":"room_type","walls":["wall_N",...]}]}\n\n'
    "Coordinates normalized so longer image edge = 1024."
)
USER_PROMPT = "Vectorize this floor plan into structured JSON with all walls, doors, windows, and rooms."

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples")
    ap.add_argument("--out", default="eval_readme")
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    imgs = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                  if p.lower().endswith(EXTS))
    if not imgs:
        print(f"[readme] no images under {a.images}/")
        sys.exit(1)

    # ── verbatim README loading ────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(BASE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    # ───────────────────────────────────────────────────────────────────────────
    print(f"[readme] {len(imgs)} imgs | base={BASE} | adapter={a.adapter} | "
          f"processor=AutoProcessor.from_pretrained(BASE) [default res] | greedy 4096")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    summary = []
    for i, p in enumerate(imgs):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            image = Image.open(p).convert("RGB")
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            raw = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            m = re.search(r"\{[\s\S]*\}", raw)
            jd = None
            if m:
                try:
                    jd = json.loads(m.group())
                except Exception:
                    jd = None
            nwalls = len(jd.get("walls", [])) if isinstance(jd, dict) else 0
            dt = time.time() - t0

            with open(os.path.join(a.out, f"{name}.json"), "w") as f:
                json.dump({"image": os.path.basename(p), "valid_json": jd is not None,
                           "n_walls": nwalls, "prediction": jd, "raw": raw}, f, indent=2)

            if isinstance(jd, dict) and jd.get("walls"):
                s = 1024.0 / max(image.size)
                rs = image.resize((max(1, round(image.width * s)),
                                   max(1, round(image.height * s)))).convert("RGB")
                d = ImageDraw.Draw(rs)
                for w in jd["walls"]:
                    try:
                        (x0, y0), (x1, y1) = w["start"], w["end"]
                        d.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=3)
                    except Exception:
                        pass
                rs.save(os.path.join(a.out, f"{name}_overlay.png"))

            summary.append({"image": name, "valid": jd is not None, "n_walls": nwalls, "sec": round(dt, 1)})
            print(f"[readme] {i + 1}/{len(imgs)} {name}: {'OK' if jd else 'INVALID-JSON'} "
                  f"walls={nwalls} ({dt:.1f}s)")
        except Exception as e:
            print(f"[readme] {i + 1}/{len(imgs)} {name}: FAILED {e}")
            summary.append({"image": name, "valid": False, "n_walls": 0, "sec": 0})

    with open(os.path.join(a.out, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ok = sum(1 for s in summary if s["valid"])
    print(f"\n[readme] DONE — valid {ok}/{len(summary)} in {a.out}/")


if __name__ == "__main__":
    main()
