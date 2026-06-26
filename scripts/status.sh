#!/usr/bin/env bash
# Show pipeline state and follow the live log.
cd "$(dirname "$0")/.."

echo "== process =="
WPIDS=$(pgrep -f "scripts/run_pipeline.sh" | tr '\n' ' ')
TPIDS=$(pgrep -f "src\.train_" | tr '\n' ' ')
if [ -n "$WPIDS" ]; then
  echo "running — watchdog PID(s): $WPIDS | trainer PID(s): ${TPIDS:-none (data prep or between retries)}"
  WCOUNT=$(echo $WPIDS | wc -w)
  [ "$WCOUNT" -gt 1 ] && echo "  ⚠ $WCOUNT watchdogs running — should be 1. Run: bash scripts/stop.sh"
else
  echo "not running"
fi

echo "== stage markers =="
ls -1 state/*.done 2>/dev/null || echo "  (none yet)"

echo "== following logs/latest.log (Ctrl-C to stop watching; training keeps running) =="
tail -n "${1:-60}" -f logs/latest.log
