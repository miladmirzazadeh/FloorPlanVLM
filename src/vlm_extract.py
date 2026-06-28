"""Verifier-guided floorplan wall extraction with a frontier VLM — no training.

Instead of GRPO on weights (impossible for a closed API model), we put the reward in the
loop at inference time:

    generate walls  ->  score (REFERENCE-FREE)  ->  feed failures back  ->  refine
                         keep the best of N samples per round

Reference-free signals (we have no ground truth at inference):
  * validity      JSON parses and walls are well-formed
  * anti-repeat   fraction of near-duplicate segments  (kills the repetition collapse)
  * topology      geometry.topology_ok: endpoints connect + walls enclose rooms
  * alignment     fraction of each wall lying on dark "ink" pixels in the image
                  (catches hallucinated walls not on any visible line)

Providers: OpenAI (GPT-5.x) or Anthropic (Claude). Set OPENAI_API_KEY or ANTHROPIC_API_KEY.

    pip install pillow numpy shapely openai          # and/or: anthropic
    python -m src.vlm_extract --images samples --out vlm_results \
        --provider openai --model gpt-5.5 --rounds 3 --n 2

Per image -> <name>.json (best prediction + score + history) and <name>_overlay.png.
"""
import os
import io
import sys
import json
import re
import glob
import time
import base64
import argparse

import numpy as np
from PIL import Image, ImageDraw

from .geometry import wall_polyline, topology_ok

CANVAS = 1024

SYSTEM_PROMPT = (
    "You are an expert at vectorizing architectural floor plans. Extract the WALLS as "
    "structured JSON.\n\n"
    "Output ONLY valid JSON with this schema:\n"
    '{"walls":[{"id":"wall_N","start":[x,y],"end":[x,y],"thickness":T,"curvature":0,'
    '"openings":[{"type":"door"|"window","center":D,"width":W}]}]}\n\n'
    "Rules:\n"
    "- Coordinates are pixels, normalized so the longer image edge = 1024.\n"
    "- Put each wall endpoint exactly on the visible wall line in the image.\n"
    "- Output EACH wall exactly once. Never repeat or near-duplicate a segment.\n"
    "- Walls must connect at corners/junctions so rooms form CLOSED loops.\n"
    "- A typical plan has ~10-40 walls. If you are emitting hundreds, you are repeating.\n"
    "- curvature is 0 for straight walls; a small signed value for curved walls.\n"
    "- openings is optional; omit if unsure."
)
USER_PROMPT = "Extract all walls from this floor plan as JSON following the schema."


# ── image helpers ───────────────────────────────────────────────────────────────

def load_1024(path):
    img = Image.open(path).convert("RGB")
    s = CANVAS / max(img.size)
    if s < 1.0:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
    return img


def ink_mask(img, thresh=190, dilate=3):
    """Boolean mask of dark 'ink' pixels (walls/lines are dark on light paper)."""
    g = np.asarray(img.convert("L"))
    m = g < thresh
    if dilate > 0:                       # grow a few px so near-misses still count as on-line
        from scipy.ndimage import binary_dilation  # optional; fall back if missing
        m = binary_dilation(m, iterations=dilate)
    return m


def ink_mask_safe(img, thresh=190, dilate=3):
    try:
        return ink_mask(img, thresh, dilate)
    except Exception:
        g = np.asarray(img.convert("L"))            # no scipy -> manual box dilation
        m = g < thresh
        out = m.copy()
        for dy in range(-dilate, dilate + 1):
            for dx in range(-dilate, dilate + 1):
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
        return out


def b64_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── parsing + reference-free scoring ─────────────────────────────────────────────

def parse_walls(text):
    text = (text or "").strip()
    cands = [text]
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        cands.append(m.group())
    for cand in cands:
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and isinstance(d.get("walls"), list):
                return d["walls"]
        except Exception:
            pass
    return None


def _seg_key(w, tol=6):
    a = (round(w["start"][0] / tol), round(w["start"][1] / tol))
    b = (round(w["end"][0] / tol), round(w["end"][1] / tol))
    return tuple(sorted([a, b]))


def dup_fraction(walls):
    if not walls:
        return 1.0
    seen, dups = set(), 0
    for w in walls:
        try:
            k = _seg_key(w)
        except Exception:
            continue
        if k in seen:
            dups += 1
        seen.add(k)
    return dups / len(walls)


def alignment(walls, mask, r=3):
    """Per-wall fraction of sampled centerline points sitting on ink; returns
    (mean_fraction, [(id, frac), ...]) so we can name the hallucinated walls."""
    H, W = mask.shape
    per = []
    for w in walls:
        try:
            pts = wall_polyline(w, n=40)
        except Exception:
            continue
        hit = tot = 0
        for (x, y) in pts:
            xi, yi = int(round(x)), int(round(y))
            tot += 1
            lo_y, hi_y = max(0, yi - r), min(H, yi + r + 1)
            lo_x, hi_x = max(0, xi - r), min(W, xi + r + 1)
            if lo_y < hi_y and lo_x < hi_x and mask[lo_y:hi_y, lo_x:hi_x].any():
                hit += 1
        if tot:
            per.append((w.get("id", "?"), hit / tot))
    mean = float(np.mean([f for _, f in per])) if per else 0.0
    return mean, per


def score(walls, mask):
    if not walls or len(walls) < 4:
        return {"total": 0.0, "valid": False, "n": len(walls or []), "dup": 1.0,
                "align": 0.0, "topo": False, "weak": []}
    valid = sum(1 for w in walls
                if all(k in w for k in ("id", "start", "end"))
                and isinstance(w.get("start"), list) and len(w["start"]) == 2) / len(walls)
    dup = dup_fraction(walls)
    al, per = alignment(walls, mask)
    topo = topology_ok(walls)
    weak = [wid for wid, f in per if f < 0.5]
    total = 0.45 * al + 0.35 * (1 - dup) + 0.10 * valid + 0.10 * (1.0 if topo else 0.0)
    return {"total": round(total, 3), "valid": valid > 0.9, "n": len(walls),
            "dup": round(dup, 3), "align": round(al, 3), "topo": topo, "weak": weak}


def feedback(sc):
    msgs = []
    if sc["dup"] > 0.05:
        msgs.append(f"You emitted {int(sc['dup'] * sc['n'])} duplicate/near-identical walls out "
                    f"of {sc['n']}. Output each distinct wall exactly once — do NOT repeat segments.")
    if sc["n"] > 120:
        msgs.append(f"{sc['n']} walls is far too many; you are looping. Real plans have ~10-40 walls.")
    if not sc["topo"]:
        msgs.append("The walls do not form closed rooms — make endpoints meet at shared corners/"
                    "junctions so every room is enclosed.")
    if sc["weak"]:
        ex = ", ".join(sc["weak"][:8])
        msgs.append(f"These walls are not on any visible line in the image (likely hallucinated): "
                    f"{ex}. Remove them or move their endpoints onto real wall lines.")
    if sc["align"] < 0.6:
        msgs.append("Many walls are off the drawn lines; align every endpoint to the actual walls.")
    return " ".join(msgs) or "Improve coordinate precision and ensure all rooms are closed."


# ── VLM clients ──────────────────────────────────────────────────────────────────

def call_openai(model, img_b64, system, user, effort="medium"):
    from openai import OpenAI
    client = OpenAI()
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]},
        ],
        max_completion_tokens=16000,   # reasoning + output; avoids truncating the JSON
    )
    if effort:
        kwargs["reasoning_effort"] = effort   # GPT-5.x: minimal|low|medium|high
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content


def call_anthropic(model, img_b64, system, user):
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=model, max_tokens=8192, system=system,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": user},
        ]}],
    )
    return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")


def pick_provider(arg):
    if arg:
        return arg
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    sys.exit("[vlm] set OPENAI_API_KEY or ANTHROPIC_API_KEY (or pass --provider).")


def make_caller(provider, model, effort="medium"):
    if provider == "openai":
        return lambda b, s, u: call_openai(model or "gpt-5.5", b, s, u, effort)
    if provider == "anthropic":
        return lambda b, s, u: call_anthropic(model or "claude-opus-4-8", b, s, u)
    sys.exit(f"[vlm] unknown provider {provider}")


# ── loop ─────────────────────────────────────────────────────────────────────────

def extract_one(caller, img, mask, rounds, n):
    best, best_sc, history = None, {"total": -1.0}, []
    convo_extra = ""
    for rnd in range(rounds):
        cands = []
        for _ in range(max(1, n)):
            try:
                txt = caller(b64_png(img), SYSTEM_PROMPT, USER_PROMPT + convo_extra)
            except Exception as e:
                history.append({"round": rnd, "error": str(e)[:200]})
                continue
            walls = parse_walls(txt) or []
            cands.append((walls, score(walls, mask)))
        if not cands:
            break
        walls, sc = max(cands, key=lambda c: c[1]["total"])
        history.append({"round": rnd, **sc})
        if sc["total"] > best_sc["total"]:
            best, best_sc = walls, sc
        if best_sc["dup"] < 0.02 and best_sc["align"] >= 0.8 and best_sc["topo"]:
            break                                   # good enough; stop refining
        convo_extra = ("\n\nYour previous attempt had problems. " + feedback(best_sc) +
                       " Re-extract the walls correctly as JSON.")
    return best or [], best_sc, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="samples")
    ap.add_argument("--out", default="vlm_results")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="medium", help="GPT-5.x reasoning effort: minimal|low|medium|high")
    ap.add_argument("--rounds", type=int, default=3, help="max refine iterations")
    ap.add_argument("--n", type=int, default=2, help="samples per round (best-of-N)")
    a = ap.parse_args()

    provider = pick_provider(a.provider)
    caller = make_caller(provider, a.model, a.effort)
    os.makedirs(a.out, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    imgs = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                  if p.lower().endswith(exts))
    if not imgs:
        sys.exit(f"[vlm] no images under {a.images}/")
    print(f"[vlm] {len(imgs)} imgs | provider={provider} model={a.model or '(default)'} "
          f"| rounds={a.rounds} n={a.n}")

    rows = []
    for i, p in enumerate(imgs):
        name = os.path.splitext(os.path.basename(p))[0]
        img = load_1024(p)
        mask = ink_mask_safe(img)
        t0 = time.time()
        walls, sc, hist = extract_one(caller, img, mask, a.rounds, a.n)
        dt = time.time() - t0

        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump({"image": os.path.basename(p), "prediction": {"walls": walls},
                       "score": sc, "history": hist}, f, indent=2)
        if walls:
            rs = img.convert("RGB")
            d = ImageDraw.Draw(rs)
            for w in walls:
                try:
                    d.line([tuple(map(float, w["start"])), tuple(map(float, w["end"]))],
                           fill=(255, 0, 0), width=3)
                except Exception:
                    pass
            rs.save(os.path.join(a.out, f"{name}_overlay.png"))
        rows.append({"image": name, "n": sc.get("n"), "score": sc.get("total"),
                     "dup": sc.get("dup"), "align": sc.get("align"), "topo": sc.get("topo"),
                     "sec": round(dt, 1)})
        print(f"[vlm] {i + 1}/{len(imgs)} {name}: score={sc.get('total')} n={sc.get('n')} "
              f"dup={sc.get('dup')} align={sc.get('align')} topo={sc.get('topo')} ({dt:.0f}s)")

    with open(os.path.join(a.out, "_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[vlm] DONE -> {a.out}/  (json + overlay per image)")


if __name__ == "__main__":
    main()
