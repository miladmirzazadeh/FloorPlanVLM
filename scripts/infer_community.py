"""Inference for the community floorplan model.

Two LoRA adapters exist:
  * SFT  : mudasir13cs/qwen25-vl-3b-floorplan-sft
  * GRPO : mudasir13cs/qwen25-vl-3b-floorplan-grpo

EMPIRICAL FINDING (don't repeat my mistake): I first assumed GRPO was trained on top of
SFT and that you must stack them (--mode full). The data refutes that:
  * --mode grpo  (base + GRPO)            -> coherent real JSON, then repetition loops
  * --mode full  (base + SFT-merged+GRPO) -> COMPLETE GIBBERISH (rare-token salad)
If GRPO sat on SFT, stacking would FIX it and grpo-only would be broken — we see the
reverse. So mudasir's GRPO was trained on PLAIN Qwen (matching its adapter_config
base=Qwen); merging SFT underneath shifts the weights out from under the GRPO delta and
corrupts the model. => correct loading is base + GRPO (the model card's way).

Bottom line: this checkpoint is undertrained (49% token acc, 5K imgs, 1234 steps). Loaded
correctly it loops; loaded any other way it's garbage. Kept here only for the record.

Modes:
  --mode grpo   base + GRPO only            <- correct loading per the card  (default)
  --mode sft    base + SFT only             <- what stage-1/2 alone produces
  --mode full   base + SFT(merged) + GRPO   <- DO NOT USE: corrupts the model (see above)

Run on the pod (deps: transformers, peft, torch, pillow; HF_HUB_DISABLE_XET=1):
    python scripts/infer_community.py --images samples --out eval_results_grpo --mode grpo

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

# ── identities ────────────────────────────────────────────────────────────────
BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
SFT_ADAPTER = "mudasir13cs/qwen25-vl-3b-floorplan-sft"
GRPO_ADAPTER = "mudasir13cs/qwen25-vl-3b-floorplan-grpo"

# ── verbatim prompt from manitocross/floorplan-vlm-training:train_floorplan_vlm.py ──
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
# ──────────────────────────────────────────────────────────────────────────────

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def extract_json(text):
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


def build_model(mode, sft_adapter, grpo_adapter):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype="auto", device_map="auto")
    if mode == "grpo":                                   # broken card snippet (SFT dropped)
        return PeftModel.from_pretrained(base, grpo_adapter)
    if mode == "sft":                                    # stage-1/2 only
        return PeftModel.from_pretrained(base, sft_adapter)
    # mode == "full": bake SFT into the weights, then apply the GRPO delta on top
    merged = PeftModel.from_pretrained(base, sft_adapter).merge_and_unload()
    return PeftModel.from_pretrained(merged, grpo_adapter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples")
    ap.add_argument("--out", default="eval_results_full")
    ap.add_argument("--mode", choices=["full", "sft", "grpo"], default="grpo")
    ap.add_argument("--sft-adapter", default=SFT_ADAPTER)
    ap.add_argument("--grpo-adapter", default=GRPO_ADAPTER)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    imgs = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                  if p.lower().endswith(EXTS))
    if not imgs:
        print(f"[infer] no images under {a.images}/")
        sys.exit(1)

    # processor from the SFT repo -> exact training min/max_pixels + chat template.
    # Fall back to base Qwen with the training-script's pixel budget if the adapter repo
    # is missing image-processor files (so a pod run never dies on processor load).
    try:
        processor = AutoProcessor.from_pretrained(a.sft_adapter)
    except Exception as e:
        print(f"[infer] processor from {a.sft_adapter} failed ({e}); using base + training pixels")
        processor = AutoProcessor.from_pretrained(
            BASE, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    model = build_model(a.mode, a.sft_adapter, a.grpo_adapter)
    model.eval()
    dev = next(model.parameters()).device
    print(f"[infer] mode={a.mode} | {len(imgs)} imgs | base={BASE} | "
          f"sft={a.sft_adapter} grpo={a.grpo_adapter} | processor from SFT repo | greedy")

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
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(dev)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            jd = extract_json(gen)
            nwalls = len(jd.get("walls", [])) if isinstance(jd, dict) else 0
            dt = time.time() - t0

            with open(os.path.join(a.out, f"{name}.json"), "w") as f:
                json.dump({"image": os.path.basename(p), "mode": a.mode,
                           "valid_json": jd is not None, "n_walls": nwalls,
                           "prediction": jd, "raw": gen}, f, indent=2)

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
        json.dump({"mode": a.mode, "results": summary}, f, indent=2)
    ok = sum(1 for s in summary if s["valid"])
    print(f"\n[infer] DONE mode={a.mode} — valid {ok}/{len(summary)} in {a.out}/")


if __name__ == "__main__":
    main()
