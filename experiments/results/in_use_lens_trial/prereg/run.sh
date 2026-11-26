#!/usr/bin/env bash
set -euo pipefail
REPO="/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief"
P="$REPO/experiments/results/in_use_lens_trial/prereg"
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

(cd "$P" && sha256sum -c --quiet SHA256SUMS)
(cd "$REPO" && sha256sum -c --quiet "$P/CODE_SHA256SUMS")

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
[ "${free:-0}" -ge 23900 ] || { echo "Need >=23,900 MiB free GPU VRAM; got ${free:-0}"; exit 5; }

$PY "$REPO/scripts/in_use_lens_trial.py" \
  --prereg-dir "$P" \
  --layers "59,58,57" \
  --out-dir "/home/eric/bin/ai/theodolite2/atlas_data/census/trials/in_use_lens" \
  --results-dir "$REPO/experiments/results/in_use_lens_trial" \
  --batch-windows 8 \
  --batch-events 4
