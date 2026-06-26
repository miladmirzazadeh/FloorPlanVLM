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

# Keep everything that grows on the LARGE volume, not the small container disk —
# otherwise '/' hits its quota and writes fail with "Disk quota exceeded".
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"          # base model (~7GB) + tokens
export TMPDIR="${TMPDIR:-$ROOT/.tmp}"                 # temp files (unzip, downloads)
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"   # triton/matplotlib/etc.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
echo "[bootstrap] caches on volume: HF_HOME=$HF_HOME TMPDIR=$TMPDIR"

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
# `huggingface-cli` is deprecated -> prefer `hf auth login`, fall back to the old CLI.
# Either way the Python pipeline also reads HF_TOKEN directly, so this is best-effort.
if hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 \
   || huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1; then
  echo "[bootstrap] HF login ok"
else
  echo "[bootstrap] HF CLI login skipped (Python pipeline uses HF_TOKEN directly)"
fi

# ── launch the watchdog fully detached ──
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" logs/latest.log
echo "[bootstrap] launching pipeline -> $LOG"
setsid nohup bash scripts/run_pipeline.sh > "$LOG" 2>&1 < /dev/null &
sleep 1
# record the real watchdog PID (setsid detaches, so $! can be wrong/empty)
pgrep -f "scripts/run_pipeline.sh" | tail -1 > state/pipeline.pid 2>/dev/null || echo "$!" > state/pipeline.pid

sleep 2
echo
echo "  ✅ Training is running in the background (PID $(cat state/pipeline.pid))."
echo "     Watch it:   bash scripts/status.sh"
echo "     Stop it:    bash scripts/stop.sh"
echo "     You can safely close this terminal / shut your laptop now."
