#!/usr/bin/env bash
# Pod-side: download MSD and unzip MSD + synth into the layout the loaders expect.
# CubiCasa + Structured3D auto-download during training, so they're not handled here.
#
# Run this AFTER you've transferred synth_floorseg.zip onto the pod (e.g. runpodctl
# receive). Idempotent — safe to re-run.
set -euo pipefail
cd "${WORKSPACE:-/workspace}"

MSD_URL="https://data.4tu.nl/file/e1d89cb5-6872-48fc-be63-aadd687ee6f9/279ef4b4-d3bd-41f4-b0c9-5e9af8cce6f6"

# ── MSD (~4.8 GB) ──
if [ -z "$(find msd_data -path '*full_out*' -name '*.npy' 2>/dev/null | head -1)" ]; then
  [ -f msd_train.zip ] || { echo "[fetch] downloading MSD (~4.8 GB)…"; \
    wget -q --show-progress -O msd_train.zip "$MSD_URL"; }
  echo "[fetch] unzipping MSD…"; mkdir -p msd_data && unzip -q -o msd_train.zip -d msd_data
else
  echo "[fetch] MSD already present"
fi

# ── synth (transferred zip) ──
if [ -f synth_floorseg.zip ] && [ ! -d synth_data/configs ]; then
  echo "[fetch] unzipping synth…"; mkdir -p synth_data && unzip -q -o synth_floorseg.zip -d synth_data
fi

echo "[fetch] MSD full_out arrays : $(find msd_data -path '*full_out*' -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
echo "[fetch] synth configs       : $(find synth_data -path '*configs*' -name 'plan_*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "[fetch] done. Verify with:  python -m src.data_msd $PWD/msd_data ; python -m src.data_synth $PWD/synth_data"
