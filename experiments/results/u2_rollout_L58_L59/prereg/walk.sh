#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
D="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/u2_rollout_L58_L59"
P="$REPO/experiments/results/u2_rollout_L58_L59/prereg"
R="$REPO/experiments/results/u2_rollout_L58_L59"
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"

mkdir -p "$D" "$R"

completed_all=1
for L in 58 59; do
  if [ ! -f "$R/L${L}_verdict.json" ]; then
    has_doses=$($PY -c 'import sys; sys.path.insert(0, sys.argv[1]); from backend import census as cz
print("1" if int(sys.argv[2]) in cz.MATCH_CEIL else "0")' "$REPO" "$L")
    if [ "$has_doses" != "1" ]; then
      echo "== $(date -u +%FT%TZ) L$L gated: $L not in backend/census.py MATCH_CEIL =="
      echo "Run fine dose pilot for L$L and amend MATCH_CEIL[$L] before running L$L:"
      echo "  $PY $REPO/scripts/census_dose_pilot.py measure --layers $L --ladder fine --out /home/eric/bin/ai/theodolite2/atlas_data/census/dose_pilot_L$L"
      echo "  $PY $REPO/scripts/census_dose_pilot.py doses --out /home/eric/bin/ai/theodolite2/atlas_data/census/dose_pilot_L$L --targets 1.0,1.4 --json /home/eric/bin/ai/theodolite2/atlas_data/census/dose_pilot_L$L/doses.json"
      echo "$L" > "$D/walk.up.blocked_no_doses"
      echo "$L" > "$R/walk.up.blocked_no_doses"
      completed_all=0
      break
    fi
    echo "== $(date -u +%FT%TZ) U2 upward walk L$L: GPU rollout =="
    LAYERS="$L" bash "$P/night.sh" || { echo "night.sh stopped at L$L (rerun to resume)"; exit 6; }
    echo "== $(date -u +%FT%TZ) U2 upward walk L$L: dual read =="
    LAYERS="$L" bash "$P/read.sh" || exit 7
  fi
  res=$($PY -c 'import json, sys; d=json.load(open(sys.argv[1])); print("pass" if d.get("pass_either") else "fail")' "$R/L${L}_verdict.json")
  echo "== $(date -u +%FT%TZ) L$L verdict: $res =="
  date -u +%FT%TZ > "$R/walk.L${L}.done"
  if [ "$res" != "pass" ]; then
    echo "$L" > "$D/walk.up.stopped_at"
    echo "$L" > "$R/walk.up.stopped_at"
    echo "Stopping U2 upward walk at L$L"
    completed_all=0
    break
  fi
done
if [ "$completed_all" -eq 1 ]; then
  date -u +%FT%TZ > "$R/walk.done"
fi
