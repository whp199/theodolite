#!/usr/bin/env bash
# The edge_things trial at layer 29 (RULES.md) on the home RTX 5090: 1,022 neurons (neurons_L29.txt, drawn by
# draw_neurons.py, none asked by any census run before), 1,024 controls (seed 13, drawn afresh), at the two doses in
# EDGE_THINGS at 29, twelve questions, 48 word-pieces, batch 64; 98,220 answers, about 3 hours. Run from the clean
# checkout of prereg/repo_commit (R), whose HEAD nothing here moves. Stops on a failure; run again it continues where
# it stopped. Nothing here reads a verdict (read.sh does), and nothing here commits, pushes or merges.
#     bash night.sh
set -euo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/edge-trial}
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data
D=$THEODOLITE_ATLAS_DIR/census/trials/edge
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
L=29
[ "$(git -C "$R" rev-parse HEAD)" = "$(cat "$P/repo_commit")" ] || { echo "$R is not at the commit in $P/repo_commit"; exit 2; }
[ -z "$(git -C "$R" --no-optional-locks status --porcelain --untracked-files=no)" ] || { echo "$R has uncommitted changes"; exit 2; }
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || { echo "$P differs from its SHA256SUMS: record why in AMENDMENTS.txt"; exit 2; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then
    echo "something answers on :8765: stop the Theodolite server first (one GPU process at a time)"; exit 5
fi
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ') || free=
[ "${free:-0}" -ge 23900 ] 2>/dev/null || { echo "only ${free:-an unknown number of} MiB of the card are free; a layer at batch 64 asks 23,900: close what holds it"; exit 5; }
done_run() {
    $PY -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from backend import census as cz
p = Path(sys.argv[2]); n = sum(1 for _ in cz.read_rows(p, tolerate_tail=True))
sys.exit(0 if n == json.loads(cz.meta_path_of(p).read_text())["jobs"]["n"] else 1)' "$R" "$D/$1.jsonl" 2>/dev/null
}
RUN=L$L-edge_things-trial
if done_run "$RUN"; then echo "$RUN: every answer already written, kept as it is"; [ -e "$D/$RUN.t1" ] || date -u +%s > "$D/$RUN.t1"; exit 0; fi
[ -e "$D/$RUN.jsonl" ] || date -u +%s > "$D/$RUN.t0"
$PY "$R/scripts/census_run.py" --layer "$L" --design edge_things --neurons "$(cat "$P/neurons_L$L.txt")" --random-controls 1024 \
    --seed 13 --batch 64 --max-new 48 --unpacker compiled --out-dir "$D" --run "$RUN" 2>&1 | tee -i -a "$D/$RUN.log"
done_run "$RUN" || { echo "$RUN stopped before its last answer: run this again to continue it row for row"; exit 6; }
date -u +%s > "$D/$RUN.t1"
echo "written in $D: layer $L; not read here (read.sh)"
