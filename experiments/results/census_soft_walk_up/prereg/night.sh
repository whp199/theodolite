#!/usr/bin/env bash
# One soft_ceil trial of the walk upward on the home RTX 5090, at each layer of LAYERS (53-59): the layer's 1,022 neurons
# (neurons_L<NN>.txt: 53-56 drawn on 12 November, 57-59 drawn on 16 November by the same script and rule), 1,024
# controls (seed 6, as layer 52's soft_ceil trial), +-1.6 and +-2.3 of each neuron's write ceiling, about 85 minutes a
# layer. Run from the clean kit-4 worktree at prereg/repo_commit (R, census-kit-4: the code a volunteer runs), whose
# HEAD nothing here moves. Stops on a failure; run again it continues where it stopped. Nothing here reads a verdict
# (read.sh does), and nothing here commits, pushes or merges.
#     LAYERS="53" bash night.sh
set -euo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/kit-4}
export THEODOLITE_ATLAS_DIR=/home/eric/bin/ai/theodolite2/atlas_data
D=$THEODOLITE_ATLAS_DIR/census/trials/soft_walk_up
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
LAYERS=${LAYERS:?give LAYERS, the layers to take}
[ "$(git -C "$R" rev-parse HEAD)" = "$(cat "$P/repo_commit")" ] || { echo "$R is not at the commit in $P/repo_commit"; exit 2; }
[ -z "$(git -C "$R" --no-optional-locks status --porcelain --untracked-files=no)" ] || { echo "$R has uncommitted changes"; exit 2; }
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || { echo "$P differs from its SHA256SUMS: record why in AMENDMENTS.txt"; exit 2; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then
    echo "something answers on :8765: stop the Theodolite server first (one GPU process at a time)"; exit 5
fi
# what kit 4 asks free of a card for a layer at batch 64 (KIT_MIN_FREE_VRAM_MIB in volunteers/kit.env); a failed query
# is an unknown, refused with the message below rather than ended silently by set -e
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ') || free=
[ "${free:-0}" -ge 23900 ] 2>/dev/null || { echo "only ${free:-an unknown number of} MiB of the card are free; kit 4 asks 23,900 at batch 64: close what holds it"; exit 5; }
done_run() {
    $PY -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from backend import census as cz
p = Path(sys.argv[2]); n = sum(1 for _ in cz.read_rows(p, tolerate_tail=True))
sys.exit(0 if n == json.loads(cz.meta_path_of(p).read_text())["jobs"]["n"] else 1)' "$R" "$D/$1.jsonl" 2>/dev/null
}
for L in $LAYERS; do
    case $L in 53|54|55|56|57|58|59) ;; *) echo "layer $L is not in this walk"; exit 2;; esac
    RUN=L$L-soft_ceil-trial
    if done_run "$RUN"; then echo "$RUN: every answer already written, kept as it is"; [ -e "$D/$RUN.t1" ] || date -u +%s > "$D/$RUN.t1"; continue; fi
    [ -e "$D/$RUN.jsonl" ] || date -u +%s > "$D/$RUN.t0"
    $PY "$R/scripts/census_run.py" --layer "$L" --design soft_ceil --neurons "$(cat "$P/neurons_L$L.txt")" --random-controls 1024 \
        --seed 6 --batch 64 --max-new 48 --unpacker compiled --out-dir "$D" --run "$RUN" 2>&1 | tee -i -a "$D/$RUN.log"
    done_run "$RUN" || { echo "$RUN stopped before its last answer: run this again to continue it row for row"; exit 6; }
    date -u +%s > "$D/$RUN.t1"
done
echo "written in $D: $LAYERS; not read here (read.sh)"
