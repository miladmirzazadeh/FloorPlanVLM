#!/usr/bin/env bash
# One command on RunPod: deps -> build dataset -> round-trip gate -> SFT Qwen3-VL-8B.
# Datasets are read from config paths (DATA_DIR / MSD_DIR / SYNTH_DIR); set HF_USER+HF_TOKEN
# to autosave checkpoints to the Hub. Launch detached so it survives terminal close:
#   setsid bash scripts/run_sft.sh > sft.log 2>&1 < /dev/null &   ; tail -f sft.log
set -euo pipefail
cd "$(dirname "$0")/.."

# caches on the big volume; xet off (avoids the slow/blocked backend)
export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$HF_HOME"

echo "[run_sft] installing deps..."
pip install -q "transformers>=4.57" peft accelerate pillow numpy shapely \
    opencv-python-headless huggingface_hub

echo "[run_sft] === build dataset ==="
python -m src.build_data

echo "[run_sft] === round-trip gate (sample; open a few overlays) ==="
python -m src.validate_roundtrip --built built/train.jsonl --out rt_check --n 40 || true

echo "[run_sft] === train ==="
python -m src.train_sft
echo "[run_sft] DONE."
