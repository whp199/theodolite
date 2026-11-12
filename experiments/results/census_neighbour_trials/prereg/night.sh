#!/usr/bin/env bash
# The neighbour trials on the home RTX 5090: break_rel at layers 39, 51, 38 and 52 (counted from 0), about 70 minutes
# each. Run from a clean checkout of the commit in prereg/repo_commit (R). Stops on the first failure; run again it
# continues where it stopped (a finished run is kept as it is). Nothing here reads a verdict (read.sh does, afterwards),
# and nothing here commits, pushes or merges.
#     LAYERS="39 51 38 52" bash night.sh
set -euo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/neighbour-trials}
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data   # the stats index's p50, as every census reads it
D=$THEODOLITE_ATLAS_DIR/census/trials/neighbours                        # not census/runs: no page reads a trial as a layer's census
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
LAYERS=${LAYERS:-"39 51 38 52"}
[ "$(git -C "$R" rev-parse HEAD)" = "$(cat "$P/repo_commit")" ] || { echo "$R is not at the commit in $P/repo_commit"; exit 2; }
[ -z "$(git -C "$R" --no-optional-locks status --porcelain --untracked-files=no)" ] || { echo "$R has uncommitted changes"; exit 2; }
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || { echo "$P differs from its SHA256SUMS: record why in AMENDMENTS.txt"; exit 2; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then
    echo "something answers on :8765: stop the Theodolite server first (one GPU process at a time)"; exit 5
fi
done_run() {  # every answer of the run's job list is on disk
    $PY -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from backend import census as cz
p = Path(sys.argv[2]); n = sum(1 for _ in cz.read_rows(p, tolerate_tail=True))
sys.exit(0 if n == json.loads(cz.meta_path_of(p).read_text())["jobs"]["n"] else 1)' "$R" "$D/$1.jsonl" 2>/dev/null
}
for L in $LAYERS; do
    RUN=L$L-break_rel-trial
    if done_run "$RUN"; then echo "$RUN: every answer already written, kept as it is"; continue; fi
    [ -e "$D/$RUN.jsonl" ] || date -u +%s > "$D/$RUN.t0"   # the first start (a continued run keeps it)
    $PY "$R/scripts/census_run.py" --layer "$L" --design break_rel --neurons "$(cat "$P/neurons_L$L.txt")" --random-controls 1024 \
        --seed 3 --batch 64 --max-new 48 --unpacker compiled --out-dir "$D" --run "$RUN" 2>&1 | tee -i -a "$D/$RUN.log"
    done_run "$RUN" || { echo "$RUN stopped before its last answer: run this again to continue it row for row"; exit 6; }
    date -u +%s > "$D/$RUN.t1"
done
date -u +%FT%TZ > "$D/night.done"
echo "written in $D: $LAYERS; not read here (read.sh)"
