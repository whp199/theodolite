#!/usr/bin/env bash
# Read the neighbour trials that are written, on the CPU, with the reader and gates hashed before any was taken.
#     LAYERS="39 51 38 52" bash read.sh
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/neighbour-trials}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/neighbours
P=$D/prereg
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
for L in ${LAYERS:-"39 51 38 52"}; do
    RUN=$D/L$L-break_rel-trial
    [ -e "$RUN.t1" ] || { echo "layer $L: not written to its end yet"; continue; }
    nice -n 15 /home/eric/bin/ai/theodolite2/.venv/bin/python -I "$P/neighbour_read.py" "$RUN.jsonl" --out "$D/L${L}_verdict.json" > "$D/L${L}_verdict.txt" 2>&1
    grep -E '^(===|pass  N|FAIL  N)' "$D/L${L}_verdict.txt"
done
