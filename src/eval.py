"""Evaluate a trained adapter on the held-out split with the paper's metrics.

    python -m src.eval                                  # GRPO adapter, 100 samples
    python -m src.eval --adapter <user>/floorplan-vlm-sft --limit 200
    DATASETS=synth python -m src.eval                   # eval on one dataset's holdout

Uses the SAME eval split as training (get_sft_datasets, seed 42), so it's genuinely
held-out. Reports validity, external-wall IoU, room IoU/F1, room-label F1, opening F1,
and wall-count MAE. Metric math lives in src/metrics.py (unit-tested separately).
"""
import os
import sys
import json
import argparse

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from . import config
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .data import get_sft_datasets
from .metrics import evaluate_pair, aggregate


def _load(adapter):
    print(f"[eval] base={config.MODEL_ID}  adapter={adapter}")
    proc = AutoProcessor.from_pretrained(config.MODEL_ID,
                                         min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID, torch_dtype="auto", device_map="auto")
    if adapter and adapter.lower() not in ("none", "base"):
        model = PeftModel.from_pretrained(model, adapter, token=config.HF_TOKEN or None)
    return model.eval(), proc


def _generate(model, proc, img, max_new_tokens):
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=config.REPO_GRPO,
                    help="HF repo or local path; 'base' for no adapter")
    ap.add_argument("--limit", type=int, default=100, help="# held-out samples to score")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--out", default="eval_results.json")
    a = ap.parse_args()

    _, eval_ds = get_sft_datasets()
    if eval_ds is None:
        print("[eval] no eval split (dataset too small) — set EVAL_RATIO higher or add data.")
        sys.exit(1)
    n = min(a.limit, len(eval_ds))
    print(f"[eval] scoring {n} of {len(eval_ds)} held-out samples on {config.DATASETS}")

    model, proc = _load(a.adapter)
    rows = []
    for i in range(n):
        s = eval_ds[i]
        img = s["images"][0]
        gt = s["messages"][2]["content"][0]["text"]
        try:
            pred = _generate(model, proc, img, a.max_new_tokens)
            rows.append(evaluate_pair(pred, gt))
        except Exception as e:
            print(f"[eval]   sample {i} failed: {e}")
            rows.append(evaluate_pair("", gt))
        if (i + 1) % 10 == 0:
            print(f"[eval]   {i + 1}/{n} ...")

    agg = aggregate(rows)
    print("\n" + "=" * 52)
    print(f"  RESULTS  ({agg['n']} samples | {config.DATASETS} | {a.adapter})")
    print("-" * 52)
    print(f"  Validity rate     : {agg['valid'] * 100:6.2f} %")
    print(f"  Ext-wall IoU      : {agg['ext_iou']:6.4f}")
    print(f"  Room IoU (geom)   : {agg['room_iou']:6.4f}")
    print(f"  Room F1 (geom)    : {agg['room_f1']:6.4f}")
    print(f"  Room label F1     : {agg['room_label_f1']:6.4f}")
    print(f"  Opening F1        : {agg['opening_f1']:6.4f}")
    print(f"  Wall-count MAE    : {agg['wall_mae']:6.2f}")
    print("=" * 52)

    result = {"adapter": a.adapter, "datasets": config.DATASETS, "model": config.MODEL_ID, **agg}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] wrote {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
