#!/usr/bin/env bash
# The break_ceil trial on the home RTX 5090: layer 38 (then, with LAYERS, the next layers down), about 75 minutes each.
# Run from a clean checkout of the commit in prereg/repo_commit (R). Stops on a failure; run again it continues where it
# stopped. Nothing here reads a verdict (read.sh does), and nothing here commits, pushes or merges.
#     LAYERS="38" bash night.sh
set -euo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/ceil-trials}
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data
D=$THEODOLITE_ATLAS_DIR/census/trials/ceil
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
LAYERS=${LAYERS:-"38"}
[ "$(git -C "$R" rev-parse HEAD)" = "$(cat "$P/repo_commit")" ] || { echo "$R is not at the commit in $P/repo_commit"; exit 2; }
[ -z "$(git -C "$R" --no-optional-locks status --porcelain --untracked-files=no)" ] || { echo "$R has uncommitted changes"; exit 2; }
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || { echo "$P differs from its SHA256SUMS: record why in AMENDMENTS.txt"; exit 2; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then
    echo "something answers on :8765: stop the Theodolite server first (one GPU process at a time)"; exit 5
fi
done_run() {
    $PY -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from backend import census as cz
p = Path(sys.argv[2]); n = sum(1 for _ in cz.read_rows(p, tolerate_tail=True))
sys.exit(0 if n == json.loads(cz.meta_path_of(p).read_text())["jobs"]["n"] else 1)' "$R" "$D/$1.jsonl" 2>/dev/null
}
for L in $LAYERS; do
    RUN=L$L-break_ceil-trial
    if done_run "$RUN"; then echo "$RUN: every answer already written, kept as it is"; continue; fi
    [ -e "$D/$RUN.jsonl" ] || date -u +%s > "$D/$RUN.t0"
    $PY "$R/scripts/census_run.py" --layer "$L" --design break_ceil --neurons "$(cat "$P/neurons_L$L.txt")" --random-controls 1024 \
        --seed 3 --batch 64 --max-new 48 --unpacker compiled --out-dir "$D" --run "$RUN" 2>&1 | tee -i -a "$D/$RUN.log"
    done_run "$RUN" || { echo "$RUN stopped before its last answer: run this again to continue it row for row"; exit 6; }
    date -u +%s > "$D/$RUN.t1"
done
echo "written in $D: $LAYERS; not read here (read.sh)"
