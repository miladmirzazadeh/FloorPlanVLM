#!/usr/bin/env bash
# FAST pod-side test: minimal deps -> run the community model 3 ways (sft / grpo / full)
# on samples/ -> zip results -> (optionally) upload the zip to HF so you can STOP THE POD
# IMMEDIATELY and download later. No HF login needed (base + adapters are public).
#
#   bash scripts/quick_eval.sh                 # runs modes: sft grpo full
#   MODES="full" bash scripts/quick_eval.sh    # just the correct stacked model
#   IMAGES=cubi_samples bash scripts/quick_eval.sh
#
# Why 3 modes: the GRPO LoRA was trained ON TOP of the SFT LoRA, but the card loads GRPO
# on bare Qwen (drops SFT). 'full' = base+SFT(merged)+GRPO is the correct final model;
# 'grpo' reproduces the broken card path; 'sft' shows the stage-1/2 baseline.
# Total pod time ~15-20 min (deps + 7.5GB download + 3 passes).
set -euo pipefail
cd "$(dirname "$0")/.."
MODES="${MODES:-sft grpo full}"

# keep caches on the big volume (avoid container-disk quota) + xet off (adapters are xet-backed)
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"

echo "[quick_eval] installing minimal inference deps..."
pip install -q transformers peft accelerate pillow huggingface_hub

for m in $MODES; do
  echo "[quick_eval] ===== mode=$m ====="
  python scripts/infer_community.py --images "${IMAGES:-samples}" --out "eval_results_$m" --mode "$m"
done

echo "[quick_eval] zipping results..."
rm -f eval_results.zip
zip -rq eval_results.zip eval_results_* 2>/dev/null || true
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
