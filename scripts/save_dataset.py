"""Package the BUILT SFT dataset into a self-contained, portable folder so it survives pod
loss and can be reused WITHOUT rebuilding (no re-download / no ~30-min ArchCAD re-render).

It reads built/{train,val}.jsonl, copies every referenced image into <out>/images/ with
flat names, and rewrites the jsonl `image` fields to RELATIVE paths (images/NNN.png). The
result is movable anywhere. Optionally pushes it to an HF dataset repo.

    python scripts/save_dataset.py --out /workspace/dataset_export
    python scripts/save_dataset.py --out /workspace/dataset_export --hf-repo miladmirza/floorplan-built
    # tarball is written too: <out>.tgz

Reuse later (skip the whole build):
    # download/extract the export to e.g. /workspace/dataset_export, then:
    export BUILT_DATA=/workspace/dataset_export SKIP_BUILD=1
    bash scripts/run_sft.sh
"""
import argparse
import json
import os
import shutil
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", default="built")
    ap.add_argument("--out", default="/workspace/dataset_export")
    ap.add_argument("--hf-repo", default="", help="optional HF dataset repo to push to (private)")
    ap.add_argument("--tar", action="store_true", default=True)
    a = ap.parse_args()

    imgdir = os.path.join(a.out, "images")
    os.makedirs(imgdir, exist_ok=True)
    seen = {}
    for split in ("train", "val"):
        src = os.path.join(a.built, f"{split}.jsonl")
        if not os.path.exists(src):
            continue
        rows = []
        for line in open(src):
            r = json.loads(line)
            p = r["image"]
            if p not in seen:
                ext = os.path.splitext(p)[1] or ".png"
                name = f"{len(seen):06d}{ext}"
                try:
                    shutil.copy(p, os.path.join(imgdir, name))
                except Exception as e:
                    print("[save] skip missing image", p, e)
                    continue
                seen[p] = name
            r = dict(r)
            r["image"] = f"images/{seen[p]}"
            rows.append(r)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[save] {split}: {len(rows)} rows")
    print(f"[save] {len(seen)} unique images -> {a.out}")

    if a.tar:
        tgz = a.out.rstrip("/") + ".tgz"
        parent, base = os.path.split(a.out.rstrip("/"))
        subprocess.run(["tar", "czf", tgz, "-C", parent or ".", base], check=False)
        print(f"[save] tarball -> {tgz}")

    if a.hf_repo:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN")
        api = HfApi(token=tok)
        api.create_repo(a.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"[save] uploading to https://huggingface.co/datasets/{a.hf_repo} ...")
        api.upload_folder(repo_id=a.hf_repo, repo_type="dataset", folder_path=a.out)
        print("[save] upload done")


if __name__ == "__main__":
    main()
