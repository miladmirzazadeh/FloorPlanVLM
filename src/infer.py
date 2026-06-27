"""Run a trained adapter on one floor-plan image.

    python -m src.infer path/to/floorplan.png
    python -m src.infer path/to/floorplan.png miladmirzazadeh/floorplan-vlm-sft

Defaults to the GRPO adapter (config.REPO_GRPO).
"""
import sys
import json
import re

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from . import config
from .prompts import SYSTEM_PROMPT, USER_PROMPT


def run(image_path, adapter=None):
    adapter = adapter or config.REPO_GRPO
    print(f"[infer] base={config.MODEL_ID}  adapter={adapter}")
    proc = AutoProcessor.from_pretrained(
        config.MODEL_ID, min_pixels=config.IMG_MIN_PIXELS, max_pixels=config.IMG_MAX_PIXELS
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter, token=config.HF_TOKEN or None).eval()

    img = Image.open(image_path).convert("RGB")
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=3072, do_sample=False)
    gen = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print("\n=== RAW OUTPUT ===\n" + gen)

    m = re.search(r"\{[\s\S]*\}", gen)
    if m:
        try:
            j = json.loads(m.group())
            print(f"\n✅ valid JSON — walls: {len(j.get('walls', []))}, rooms: {len(j.get('rooms', []))}")
        except Exception:
            print("\n⚠ JSON parse failed (model may need more training)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.infer <image> [adapter_repo_or_path]")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
