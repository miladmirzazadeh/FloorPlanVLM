#!/usr/bin/env bash
# Show pipeline state and follow the live log.
cd "$(dirname "$0")/.."

echo "== process =="
if [ -f state/pipeline.pid ] && kill -0 "$(cat state/pipeline.pid)" 2>/dev/null; then
  echo "running (PID $(cat state/pipeline.pid))"
else
  echo "not running"
fi

echo "== stage markers =="
ls -1 state/*.done 2>/dev/null || echo "  (none yet)"

echo "== following logs/latest.log (Ctrl-C to stop watching; training keeps running) =="
tail -n "${1:-60}" -f logs/latest.log
