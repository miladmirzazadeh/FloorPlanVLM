"""Grab a few CubiCasa5K sample images from the HF mirror (no 5GB Zenodo download).

Streams `Claudio9701/cubicasa5k` (the dataset the model card lists) and saves the first
N raster floor-plan images as PNGs — so we can test the community model on its OWN
training distribution.

    pip install datasets
    python scripts/get_cubicasa_samples.py --n 2 --out samples   # lands next to plan1..6
"""
import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--dataset", default="Claudio9701/cubicasa5k")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from datasets import load_dataset
    print(f"[cubi] streaming {a.dataset} [{a.split}] ...")
    ds = load_dataset(a.dataset, split=a.split, streaming=True)

    saved = 0
    for rec in ds:
        img = None
        for k, v in rec.items():
            if hasattr(v, "save") and hasattr(v, "convert"):   # a PIL image field
                img = v
                break
        if img is None:
            continue
        path = os.path.join(a.out, f"cubicasa_{saved + 1}.png")
        img.convert("RGB").save(path)
        print(f"[cubi] saved {path}  ({img.size[0]}x{img.size[1]})")
        saved += 1
        if saved >= a.n:
            break

    if saved == 0:
        print("[cubi] no image column found — inspect the dataset schema:")
        print("       python -c \"from datasets import load_dataset as L; "
              "print(next(iter(L('%s', split='%s', streaming=True))).keys())\"" % (a.dataset, a.split))
    else:
        print(f"[cubi] done — {saved} CubiCasa image(s) in {a.out}/")


if __name__ == "__main__":
    main()
