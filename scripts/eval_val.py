"""Evaluate the trained SFT adapter on the held-out VAL split.

Generates predictions for N val samples, decodes them with OUR schema, scores them
(valid-JSON rate, wall-count MAE, external-wall IoU), and writes side-by-side overlays
(left = ground truth in green, right = prediction in red) so you can eyeball them.

Single GPU. Run:
  python scripts/eval_val.py --adapter miladmirza/floorplan-vlm-sft2 \
      --built /data/dataset_export --n 30 --out eval_out
"""
import argparse
import json
import os
import sys
import statistics

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
        print(f"[eval] loading adapter {adapter}")
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


def footprint(walls):
    """Filled floor area from polygonizing all wall centerlines (arc-aware) -> shapely geom."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union
    lines = []
    for w in walls:
        pts = arc_polyline(w["cl"], w.get("cv", 0))
        if len(pts) >= 2:
            lines.append(LineString(pts))
    if len(lines) < 3:
        return None
    try:
        faces = [p for p in polygonize(unary_union(lines)) if p.area > 1.0]
        return unary_union(faces) if faces else None
    except Exception:
        return None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    try:
        u = a.union(b).area
        return a.intersection(b).area / u if u > 0 else 0.0
    except Exception:
        return 0.0


def draw_walls(base, walls, color, scale):
    im = base.copy()
    d = ImageDraw.Draw(im)
    for w in walls:
        pts = [(x * scale, y * scale) for x, y in arc_polyline(w["cl"], w.get("cv", 0))]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=max(2, int(w.get("th", 4) * scale)))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=config.REPO_SFT, help="HF repo or local path; 'base' = no adapter")
    ap.add_argument("--built", default=config.BUILT_DATA, help="dir with val.jsonl + images/")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    val = [json.loads(l) for l in open(os.path.join(a.built, "val.jsonl")) if l.strip()]
    n = min(a.n, len(val))
    print(f"[eval] adapter={a.adapter}  val={len(val)}  scoring {n}")
    model, proc = load_model(a.adapter)

    nvalid, cnt_err, ious = 0, [], []
    for i in range(n):
        r = val[i]
        p = r["image"]
        if not os.path.isabs(p):
            p = os.path.join(a.built, p)
        img = Image.open(p).convert("RGB")
        W, H = img.size
        scale = max(W, H) / config.GRID
        gt = schema.decode(r["target"])
        try:
            raw = generate(model, proc, img, a.max_new_tokens)
            pred = schema.decode(raw)
        except Exception as e:
            print(f"  [{i}] generate failed: {e}")
            pred = []
        nvalid += int(len(pred) > 0)
        cnt_err.append(abs(len(pred) - len(gt)))
        ii = iou(footprint(pred), footprint(gt))
        ious.append(ii)
        combo = Image.new("RGB", (W * 2 + 20, H), "white")
        combo.paste(draw_walls(img, gt, (0, 180, 0), scale), (0, 0))
        combo.paste(draw_walls(img, pred, (220, 0, 0), scale), (W + 20, 0))
        combo.save(os.path.join(a.out, f"{i:03d}_iou{int(ii*100):02d}_gt{len(gt)}_pred{len(pred)}.png"))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n}  valid={nvalid}  meanIoU={statistics.mean(ious):.2f}")

    print("\n" + "=" * 50)
    print(f"  VAL RESULTS  ({n} samples | {a.adapter})")
    print("-" * 50)
    print(f"  valid-JSON rate : {nvalid / n * 100:5.1f} %")
    print(f"  wall-count MAE  : {statistics.mean(cnt_err):5.2f}")
    print(f"  ext-wall IoU    : {statistics.mean(ious):.3f}  (median {statistics.median(ious):.3f})")
    print("=" * 50)
    print(f"  overlays -> {a.out}/   (left = GT green, right = prediction red)")
    json.dump({"adapter": a.adapter, "n": n, "valid_rate": nvalid / n,
               "wall_count_mae": statistics.mean(cnt_err),
               "ext_iou_mean": statistics.mean(ious), "ext_iou_median": statistics.median(ious)},
              open(os.path.join(a.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
