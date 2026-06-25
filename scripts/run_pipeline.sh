#!/usr/bin/env bash
# Watchdog: run SFT, then GRPO. If a stage crashes (OOM, eviction, spot kill,
# network blip), it is restarted and RESUMES from the latest Hub checkpoint.
# Each stage is also a no-op once its FINISHED marker exists on the Hub, so this
# script is safe to re-run on a brand-new pod.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export PYTHONUNBUFFERED=1
mkdir -p state logs
RETRY_WAIT="${RETRY_WAIT:-30}"

run_stage () {
  local name="$1"; shift
  local done_marker="state/${name}.done"
  if [ -f "$done_marker" ]; then
    echo "[pipeline] $name already complete (local marker) — skipping"
    return 0
  fi
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "[pipeline] ===== $name : attempt $attempt @ $(date) ====="
    if "$@"; then
      touch "$done_marker"
      echo "[pipeline] ===== $name : COMPLETED ====="
      return 0
    fi
    echo "[pipeline] $name exited non-zero; resuming in ${RETRY_WAIT}s ..."
    sleep "$RETRY_WAIT"
  done
}

echo "[pipeline] start @ $(date)"
run_stage sft  python -m src.train_sft

if [ "${RUN_GRPO:-true}" = "true" ] || [ "${RUN_GRPO:-1}" = "1" ]; then
  run_stage grpo python -m src.train_grpo
else
  echo "[pipeline] RUN_GRPO disabled — stopping after SFT"
fi

echo "[pipeline] 🎉 ALL STAGES COMPLETE @ $(date)"
