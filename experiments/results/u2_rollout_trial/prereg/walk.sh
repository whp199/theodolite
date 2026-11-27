#!/usr/bin/env bash
set -uo pipefail
REPO="/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief"
D="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/u2_rollout"
P="$REPO/experiments/results/u2_rollout_trial/prereg"
R="$REPO/experiments/results/u2_rollout_trial"
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"

for L in 55 56 57; do
  if [ ! -f "$R/L${L}_verdict.json" ]; then
    echo "== $(date -u +%FT%TZ) U2 upward walk L$L: GPU rollout =="
    LAYERS="$L" bash "$P/night.sh" || { echo "night.sh stopped at L$L (rerun to resume)"; exit 6; }
    echo "== $(date -u +%FT%TZ) U2 upward walk L$L: dual read =="
    LAYERS="$L" bash "$P/read.sh" || exit 7
  fi
  res=$($PY -c 'import json, sys; d=json.load(open(sys.argv[1])); print("pass" if d.get("pass_either") else "fail")' "$R/L${L}_verdict.json")
  echo "== $(date -u +%FT%TZ) L$L verdict: $res =="
  if [ "$res" != "pass" ]; then
    echo "$L" > "$D/walk.up.stopped_at"
    echo "$L" > "$R/walk.up.stopped_at"
    echo "Stopping U2 upward walk at L$L"
    break
  fi
done
date -u +%FT%TZ > "$R/walk.done"
