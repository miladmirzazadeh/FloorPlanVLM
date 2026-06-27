"""Character-for-character faithful reproduction of the community model's OWN
inference snippet (manitocross/floorplan-vlm-training -> train_floorplan_vlm.py /
mudasir13cs/qwen25-vl-3b-floorplan-grpo model card).

Why this exists: our src/batch_infer.py is wired to our training config (WALLS_ONLY,
our prompts.py with an extra room-label line, max_pixels=1024*1024). To fairly judge
the *pretrained* community adapter we must feed it EXACTLY what it was trained on —
same system/user prompt, same processor min/max_pixels, same greedy decode. Any
deviation puts the input off-distribution and a GRPO model degenerates (e.g. the
infinite-wall loop we saw).

Run on the pod (minimal deps: transformers, peft, torch, pillow):
    python scripts/infer_community.py --images samples --out community_results
    python scripts/infer_community.py --images cubi_samples --out cubi_results

Outputs per image: <name>.json (parsed + raw) and <name>_overlay.png.
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

# ── verbatim from manitocross/floorplan-vlm-training : train_floorplan_vlm.py ──
BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER = "mudasir13cs/qwen25-vl-3b-floorplan-grpo"

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

# their GPU processor settings (NOT 1024*1024)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
# ──────────────────────────────────────────────────────────────────────────────

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def extract_json(text):
    """Their note: 'extracting JSON via regex from raw output.'"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples")
    ap.add_argument("--out", default="community_results")
    ap.add_argument("--adapter", default=ADAPTER, help="'base' to run the raw base model")
    ap.add_argument("--max-new-tokens", type=int, default=4096)  # adapter card value
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    imgs = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                  if p.lower().endswith(EXTS))
    if not imgs:
        print(f"[infer] no images under {a.images}/")
        sys.exit(1)
    print(f"[infer] {len(imgs)} imgs | base={BASE} | adapter={a.adapter} | "
          f"min_px={MIN_PIXELS} max_px={MAX_PIXELS} | greedy, no rep-penalty (faithful)")

    processor = AutoProcessor.from_pretrained(BASE, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype="auto", device_map="auto")
    if a.adapter.lower() not in ("base", "none"):
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    summary = []
    for i, p in enumerate(imgs):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            image = Image.open(p).convert("RGB")
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            jd = extract_json(gen)
            nwalls = len(jd.get("walls", [])) if isinstance(jd, dict) else 0
            dt = time.time() - t0

            with open(os.path.join(a.out, f"{name}.json"), "w") as f:
                json.dump({"image": os.path.basename(p), "valid_json": jd is not None,
                           "n_walls": nwalls, "prediction": jd,
                           "raw": gen}, f, indent=2)

            if isinstance(jd, dict) and jd.get("walls"):
                s = 1024.0 / max(image.size)
                rs = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s)))).convert("RGB")
                d = ImageDraw.Draw(rs)
                for w in jd["walls"]:
                    try:
                        (x0, y0), (x1, y1) = w["start"], w["end"]
                        d.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=3)
                    except Exception:
                        pass
                rs.save(os.path.join(a.out, f"{name}_overlay.png"))

            summary.append({"image": name, "valid": jd is not None, "n_walls": nwalls, "sec": round(dt, 1)})
            print(f"[infer] {i + 1}/{len(imgs)} {name}: {'OK' if jd else 'INVALID-JSON'} "
                  f"walls={nwalls} ({dt:.1f}s)")
        except Exception as e:
            print(f"[infer] {i + 1}/{len(imgs)} {name}: FAILED {e}")
            summary.append({"image": name, "valid": False, "n_walls": 0, "sec": 0})

    with open(os.path.join(a.out, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ok = sum(1 for s in summary if s["valid"])
    print(f"\n[infer] DONE — valid {ok}/{len(summary)} in {a.out}/")


if __name__ == "__main__":
    main()
