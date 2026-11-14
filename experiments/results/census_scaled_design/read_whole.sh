#!/usr/bin/env bash
# Read a whole layer taken with a design that counts its push in write ceilings (match_ceil, soft_ceil) the way that
# layer's trial was read: the three checks N1, N2 and N3 of ceil_read.py, the reader and gates hashed before any answer
# of the trial existed, on the CPU; then, reported and gating nothing, the upper dose's answers alone. The record is
# kept in experiments/results/census_L<NN>_whole/ beside the other layers', each reading's named_words left out to
# keep it small. A whole layer is about 20 minutes and 6 GB (at nice 15, so a census run beside it keeps its pace).
#
# The reader is the layer's own trial's: its three files (ceil_read.py, trial_read.py, readlib.py) are the same bytes
# in every one of these folders, and each folder's SHA256SUMS is checked before anything is read.
#   match_ceil  layers 30-36  census/trials/match_walk/prereg     (the walk of 13 November 2026)
#               layer 37      census/trials/match/prereg
#   soft_ceil   layer 38      census/trials/soft/prereg
#               layer 52      census/trials/soft/prereg_L52
# A layer with no trial here (53-59, should one pass later) is read with PREREG=<its trial's folder>.
# For break_rel (39, 51, and 40-50) the reader is experiments/results/census_neighbour_trials/read_whole.sh.
#
#     bash read_whole.sh 36                              # atlas_data/census/runs/L36-match_ceil.jsonl
#     RUN=<a run> OUT=<a folder> bash read_whole.sh 36   # another run of that layer (a trial, a volunteer's, a copy)
#     PY=<a python> THEODOLITE_ATLAS_DIR=<the data folder> bash read_whole.sh 36    # from a worktree, which has neither
set -euo pipefail
L=${1:?a layer, counted from 0}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PY=${PY:-$ROOT/.venv/bin/python}
ATLAS=${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}
T=$ATLAS/census/trials
R=${RUN:-}
if [ -z "$R" ]; then
    for d in match_ceil soft_ceil; do
        [ -f "$ATLAS/census/runs/L$L-$d.jsonl" ] && R=$ATLAS/census/runs/L$L-$d.jsonl
    done
fi
[ -n "$R" ] && [ -f "$R" ] || { echo "no run of layer $L taken with match_ceil or soft_ceil under $ATLAS/census/runs (give RUN=)"; exit 2; }
META=${R%.jsonl}.meta.json
[ -f "$META" ] || { echo "no meta beside $R"; exit 2; }
# the run's own layer, design and upper dose (the greater of its two sizes), from its meta
read -r RL DESIGN UPPER < <("$PY" -I - "$META" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
sizes = sorted({abs(float(d)) for _, d in m["design"]})
print(m.get("layer"), m.get("design_name"), f"{sizes[-1]:g}" if len(sizes) == 2 else "none")
PY
)
[ "$RL" = "$L" ] || { echo "$R is a run of layer $RL, not $L"; exit 2; }
[ "$UPPER" != none ] || { echo "$R: a design of two doses each way is read here"; exit 2; }
P=${PREREG:-}
if [ -z "$P" ]; then
    case "$DESIGN:$L" in
        match_ceil:3[0-6]) P=$T/match_walk/prereg ;;
        match_ceil:37)     P=$T/match/prereg ;;
        soft_ceil:38)      P=$T/soft/prereg ;;
        soft_ceil:52)      P=$T/soft/prereg_L52 ;;
        break_rel:*)       echo "layer $L was taken with break_rel: read it with experiments/results/census_neighbour_trials/read_whole.sh"; exit 2 ;;
        *)                 echo "no trial of $DESIGN at layer $L is known here: give PREREG=<its trial's folder of registered files>"; exit 2 ;;
    esac
fi
[ -f "$P/ceil_read.py" ] && [ -f "$P/SHA256SUMS" ] || { echo "no registered reader at $P"; exit 2; }
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || { echo "$P differs from its SHA256SUMS: see AMENDMENTS.txt there"; exit 2; }
HERE=${OUT:-experiments/results/census_L${L}_whole}
mkdir -p "$HERE"
# the registered reader takes backend/census.py from this checkout: the one that knows the design's doses
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$ROOT"
nice -n 15 "$PY" -I "$P/ceil_read.py" "$R" --out "$HERE/L${L}_verdict.json" > "$HERE/L${L}_verdict.txt" 2>&1 || true
[ -f "$HERE/L${L}_verdict.json" ] || { echo "the reader wrote no verdict: see $HERE/L${L}_verdict.txt"; tail -n 5 "$HERE/L${L}_verdict.txt"; exit 3; }
# reported, gating nothing: the upper dose's answers alone
nice -n 15 "$PY" -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 "--doses=-$UPPER,$UPPER" --out "$HERE/L${L}_upper.json" > "$HERE/L${L}_upper.txt" 2>&1 || true
"$PY" -I - "$HERE" "$L" <<'PY'
import json, sys
from pathlib import Path
here, layer = Path(sys.argv[1]), sys.argv[2]
for part in ("whole", "pm1"):
    p = here / f"L{layer}_verdict.{part}.json"
    if p.exists():
        d = json.loads(p.read_text())
        for rd in (d.get("readings") or {}).values():
            rd.pop("named_words", None)
        p.write_text(json.dumps(d, indent=1))
p = here / f"L{layer}_upper.json"
if p.exists():
    d = json.loads(p.read_text())
    for rd in (d.get("readings") or {}).values():
        rd.pop("named_words", None)
    p.write_text(json.dumps(d, indent=1))
PY
echo "layer $L, $DESIGN, read by $P"
grep -E '^(===|pass  N|FAIL  N)' "$HERE/L${L}_verdict.txt"
grep -E '^  (pass|FAIL)  G[1-4]' "$HERE/L${L}_upper.txt" 2>/dev/null | sed "s/^/the upper dose alone, +-$UPPER (reported): /" || true
