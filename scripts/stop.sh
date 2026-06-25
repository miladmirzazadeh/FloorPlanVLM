#!/usr/bin/env bash
# Stop the background pipeline (does NOT delete checkpoints; resume with bootstrap).
cd "$(dirname "$0")/.."
PID="$(cat state/pipeline.pid 2>/dev/null || true)"
if [ -n "${PID:-}" ]; then
  kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
  echo "stopped process group $PID"
fi
pkill -f "src.train_sft"  2>/dev/null || true
pkill -f "src.train_grpo" 2>/dev/null || true
pkill -f "run_pipeline.sh" 2>/dev/null || true
echo "done. Re-run 'bash scripts/runpod_bootstrap.sh' to resume."
