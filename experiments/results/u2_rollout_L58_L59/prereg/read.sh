#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
D="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/u2_rollout_L58_L59"
P="$REPO/experiments/results/u2_rollout_L58_L59/prereg"
R="$REPO/experiments/results/u2_rollout_L58_L59"
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
LAYERS=${LAYERS:?give LAYERS}

mkdir -p "$R"
(cd "$P" && sha256sum -c --quiet SHA256SUMS)
(cd "$REPO" && sha256sum -c --quiet "$P/CODE_SHA256SUMS")

for L in $LAYERS; do
  [ -e "$R/L${L}_verdict.json" ] && { echo "L$L: already read"; continue; }
  $PY "$REPO/scripts/u2_rollout_read.py" \
    --prereg-dir "$P" \
    --trial-dir "$D" \
    --layer "$L" \
    --out-dir "$R" 2>&1 | tee "$R/L${L}_read.log"
done
