"""Grab a few real CubiCasa5K images WITHOUT downloading the full 5GB Zenodo zip.

Uses HTTP range requests (remotezip) to read the archive's central directory and then
pull only the bytes of the few F1_original.png files we want — typically ~10-20MB total
instead of 5GB. Ideal for a slow-network pod.

    pip install remotezip
    python scripts/get_cubicasa_remote.py --n 2 --out cubi_samples

Then run the README-exact inference on them:
    python scripts/infer_readme.py --images cubi_samples --out eval_cubi
"""
import argparse
import os

from remotezip import RemoteZip

# CubiCasa5K on Zenodo (same record used by src/config.py ZENODO_URL)
URL = "https://zenodo.org/record/2613548/files/cubicasa5k.zip?download=1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="cubi_samples")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--name", default="F1_original.png",
                    help="which per-plan raster to pull (F1_original.png or F1_scaled.png)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print(f"[cubi] opening remote zip (range requests) ...\n       {a.url}")
    with RemoteZip(a.url) as z:
        names = [n for n in z.namelist() if n.endswith(a.name)]
        print(f"[cubi] {len(names)} '{a.name}' entries found in archive")
        saved = 0
        for n in names:
            if saved >= a.n:
                break
            try:
                data = z.read(n)
            except Exception as e:
                print(f"[cubi] skip {n}: {e}")
                continue
            out = os.path.join(a.out, f"cubicasa_{saved + 1}.png")
            with open(out, "wb") as f:
                f.write(data)
            print(f"[cubi] saved {out}  ({len(data) // 1024} KB)  <- {n}")
            saved += 1

    if saved == 0:
        print("[cubi] nothing saved — the server may not support range requests; "
              "try --name F1_scaled.png or download the full zip on a fast pod.")
    else:
        print(f"[cubi] done — {saved} CubiCasa image(s) in {a.out}/")


if __name__ == "__main__":
    main()
