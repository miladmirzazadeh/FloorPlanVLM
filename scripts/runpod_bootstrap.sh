#!/usr/bin/env bash
# One command to set up the box and launch training DETACHED, so it keeps running
# after you close the RunPod terminal / shut your laptop.
#
#   bash scripts/runpod_bootstrap.sh
#
# Requires HF_TOKEN and HF_USER in the environment or in ./.env
# (set them as RunPod template env vars for zero-typing resumes).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Load .env if present (RunPod env vars also work without it)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

: "${HF_TOKEN:?Set HF_TOKEN (export HF_TOKEN=hf_... or add to .env)}"
: "${HF_USER:?Set HF_USER (your HuggingFace username) or add to .env}"
export HF_TOKEN HF_USER

mkdir -p logs state outputs

# ── deps (install once; marker on the persistent volume) ──
if [ ! -f state/.deps_installed ]; then
  echo "[bootstrap] installing dependencies (first run only)..."
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  pip install -q flash-attn --no-build-isolation 2>/dev/null \
    && echo "[bootstrap] flash-attn installed" \
    || echo "[bootstrap] flash-attn skipped (optional)"
  touch state/.deps_installed
else
  echo "[bootstrap] deps already installed (state/.deps_installed)"
fi

# ── HF auth (non-interactive) ──
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 \
  && echo "[bootstrap] HF login ok" || echo "[bootstrap] HF login warning (continuing)"

# ── launch the watchdog fully detached ──
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/latest.log
echo "[bootstrap] launching pipeline -> $LOG"
setsid nohup bash scripts/run_pipeline.sh > "$LOG" 2>&1 < /dev/null &
echo $! > state/pipeline.pid

sleep 2
echo
echo "  ✅ Training is running in the background (PID $(cat state/pipeline.pid))."
echo "     Watch it:   bash scripts/status.sh"
echo "     Stop it:    bash scripts/stop.sh"
echo "     You can safely close this terminal / shut your laptop now."
