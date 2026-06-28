#!/usr/bin/env bash
# Prepare the SFT dataset on a CHEAP CPU pod, then push it to an HF dataset — so the
# expensive 2xH100 pod only ever trains (download + train, no gen/build).
#
# Run detached (survives terminal close):
#   export HF_TOKEN=hf_xxx HF_USER=miladmirza
#   setsid nohup bash scripts/prep_dataset.sh > prep.log 2>&1 < /dev/null &
#   tail -f prep.log
#
# Tunables (env): SYNTH_COUNT (10000), SYNTH_MAX_DPI (150), DATASETS (binnies,synth),
#   SYNTH_DIR, BUILT_DATA, EXPORT_DIR, HF_REPO_BUILT (default <HF_USER>/floorplan-built), VENV.
set -euo pipefail
cd "$(dirname "$0")/.."

COUNT="${SYNTH_COUNT:-10000}"
MAXDPI="${SYNTH_MAX_DPI:-150}"
DATASETS="${DATASETS:-binnies,synth}"
SYNTH_DIR="${SYNTH_DIR:-/workspace/synth_data}"
BUILT_DATA="${BUILT_DATA:-/workspace/built}"
EXPORT_DIR="${EXPORT_DIR:-/workspace/dataset_export}"
HF_REPO_BUILT="${HF_REPO_BUILT:-${HF_USER:-miladmirza}/floorplan-built}"
VENV="${VENV:-/workspace/venv}"
NPROC="$(nproc)"
export DATASETS SYNTH_DIR BUILT_DATA

# --- venv so `python` and the installed deps are the SAME interpreter (pods often ship
#     a python3.8 default while pip targets python3.13 -> ModuleNotFoundError) ---
if [ ! -x "$VENV/bin/python" ]; then
  PY313="$(command -v python3.13 || command -v python3 || command -v python)"
  echo "[prep] creating venv at $VENV using $PY313"
  "$PY313" -m venv "$VENV"
fi
PY="$VENV/bin/python"
echo "[prep] installing deps (generator + build; no torch)..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r floorplan_synthgen/requirements.txt \
    datasets pillow numpy shapely opencv-python-headless huggingface_hub
"$PY" -c "import numpy,PIL,shapely,datasets,cv2,huggingface_hub,ezdxf,matplotlib,networkx; print('[prep] deps OK')"

echo "[prep] === 1/4 generate $COUNT synth  (workers=$NPROC, max-dpi=$MAXDPI) ==="
"$PY" floorplan_synthgen/generate_dataset.py --count "$COUNT" \
    --output "$SYNTH_DIR" --workers "$NPROC" --max-dpi "$MAXDPI"

echo "[prep] === 2/4 build [$DATASETS] -> $BUILT_DATA ==="
"$PY" -m src.build_data

echo "[prep] === 3/4 round-trip gate (sanity; non-fatal) ==="
"$PY" -m src.validate_roundtrip --built "$BUILT_DATA/train.jsonl" --out rt_check --n 40 || true

echo "[prep] === 4/4 package + push -> HF dataset $HF_REPO_BUILT ==="
"$PY" scripts/save_dataset.py --built "$BUILT_DATA" --out "$EXPORT_DIR" --hf-repo "$HF_REPO_BUILT"

echo ""
echo "[prep] ============================================================"
echo "[prep] DONE -> https://huggingface.co/datasets/$HF_REPO_BUILT"
echo "[prep] Kill this CPU pod. On the 2xH100, train with:"
echo "[prep]   export BUILT_DATA=/workspace/dataset_export SKIP_BUILD=1"
echo "[prep]   huggingface-cli download $HF_REPO_BUILT --repo-type dataset --local-dir /workspace/dataset_export"
echo "[prep]   bash scripts/run_sft.sh"
echo "[prep] ============================================================"
