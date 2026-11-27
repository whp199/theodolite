#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
D="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/u2_rollout_L58_L59"
P="$REPO/experiments/results/u2_rollout_L58_L59/prereg"
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
LAYERS=${LAYERS:?give LAYERS}

mkdir -p "$D"
[ "$(git -C "$REPO" rev-parse HEAD)" = "$(cat "$P/repo_commit")" ] || true
(cd "$P" && sha256sum -c --quiet SHA256SUMS)
(cd "$REPO" && sha256sum -c --quiet "$P/CODE_SHA256SUMS")

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
[ "${free:-0}" -ge 23900 ] || { echo "only ${free:-0} MiB free; need >=23,900 MiB"; exit 5; }

done_run() {
  $PY -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from backend import census as cz
p = Path(sys.argv[2]); n = sum(1 for _ in cz.read_rows(p, tolerate_tail=True))
sys.exit(0 if n == json.loads(cz.meta_path_of(p).read_text())["jobs"]["n"] else 1)' "$REPO" "$D/$1.jsonl" 2>/dev/null
}

for L in $LAYERS; do
  has_doses=$($PY -c 'import sys; sys.path.insert(0, sys.argv[1]); from backend import census as cz
print("1" if int(sys.argv[2]) in cz.MATCH_CEIL else "0")' "$REPO" "$L")
  if [ "$has_doses" != "1" ]; then
    echo "ERROR: L$L is not in backend/census.py MATCH_CEIL; run fine dose pilot and amend MATCH_CEIL[$L] first." >&2
    exit 4
  fi

  RUN_C="L${L}-match_things-u2-cols"
  if ! done_run "$RUN_C"; then
    [ -e "$D/$RUN_C.jsonl" ] || date -u +%s > "$D/$RUN_C.t0"
    $PY "$REPO/scripts/census_run.py" --layer "$L" --design match_things \
      --neurons "$(cat "$P/neurons_L$L.txt")" --random-controls 1024 \
      --control-kind columns --seed 21 --batch 64 --max-new 48 \
      --unpacker compiled --out-dir "$D" --run "$RUN_C" --no-rank 2>&1 | tee -i -a "$D/$RUN_C.log"
    done_run "$RUN_C" || exit 6
    date -u +%s > "$D/$RUN_C.t1"
  fi

  RUN_I="L${L}-match_things-u2-iso"
  if ! done_run "$RUN_I"; then
    [ -e "$D/$RUN_I.jsonl" ] || date -u +%s > "$D/$RUN_I.t0"
    $PY "$REPO/scripts/census_run.py" --layer "$L" --design match_things \
      --neurons none --random-controls 1024 \
      --control-kind isotropic --seed 22 --batch 64 --max-new 48 \
      --unpacker compiled --out-dir "$D" --run "$RUN_I" --no-rank 2>&1 | tee -i -a "$D/$RUN_I.log"
    done_run "$RUN_I" || exit 6
    date -u +%s > "$D/$RUN_I.t1"
  fi
done
