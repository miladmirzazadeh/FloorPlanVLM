#!/usr/bin/env bash
# FAST pod-side test: minimal deps -> FAITHFUL community inference on samples/ -> zip
# results -> (optionally) upload the zip to HF so you can STOP THE POD IMMEDIATELY and
# download the results later from anywhere. No HF login needed (base + adapter public).
#
#   bash scripts/quick_eval.sh                 # community GRPO model (default)
#   bash scripts/quick_eval.sh <ADAPTER>       # your own model, e.g. miladmirza/floorplan-walls-grpo
#
# Total pod time ~5-10 min (deps + 7.5GB model download + inference).
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER="${1:-mudasir13cs/qwen25-vl-3b-floorplan-grpo}"

# keep caches on the big volume (avoid container-disk quota)
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"

echo "[quick_eval] installing minimal inference deps..."
pip install -q transformers peft accelerate pillow huggingface_hub

echo "[quick_eval] running FAITHFUL community inference (adapter=$ADAPTER)..."
# infer_community.py = exact upstream snippet (their prompt + processor max_pixels=1280*28*28
# + plain greedy). Apples-to-apples test of the pretrained adapter.
python scripts/infer_community.py --images "${IMAGES:-samples}" --out eval_results --adapter "$ADAPTER"

echo "[quick_eval] zipping results..."
rm -f eval_results.zip
( cd eval_results && zip -rq ../eval_results.zip . )
echo "[quick_eval] -> eval_results.zip ($(du -h eval_results.zip | cut -f1))"

# Best path to 'stop pod immediately': push the small zip to HF, download later.
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_USER:-}" ]; then
  echo "[quick_eval] uploading results to HF (so you can stop the pod now)..."
  python - "$HF_USER" <<'PY' || echo "[quick_eval] HF upload skipped (check token); use the zip instead."
import sys
from huggingface_hub import HfApi
user = sys.argv[1]
repo = f"{user}/floorplan-eval-results"
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj="eval_results.zip", path_in_repo="eval_results.zip",
                repo_id=repo, repo_type="dataset")
print(f"[quick_eval] uploaded -> https://huggingface.co/datasets/{repo} (file: eval_results.zip)")
PY
  echo "[quick_eval] DONE. Download eval_results.zip from the HF dataset above, then STOP THE POD."
else
  echo "[quick_eval] DONE. Set HF_TOKEN+HF_USER to auto-upload, or download eval_results.zip"
  echo "             from the RunPod file browser / 'runpodctl send eval_results.zip', then STOP THE POD."
fi
