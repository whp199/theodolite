#!/usr/bin/env bash
# Read the walk's soft_ceil trial at each layer of LAYERS once it is written, on the CPU, with the reader and gates
# hashed before any answer of the walk existed: ceil_read.py (the three gates), then, gating nothing, the upper dose's
# answers alone. A layer already read is not read again.
#     LAYERS="37" bash read.sh
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/soft-trials}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/soft_walk
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
LAYERS=${LAYERS:?give LAYERS, the layers to read}
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
for L in $LAYERS; do
    RUN=$D/L$L-soft_ceil-trial
    [ -e "$RUN.t1" ] || { echo "layer $L: not written to its end yet"; continue; }
    [ -e "$D/L${L}_verdict.json" ] && { echo "layer $L: already read, kept as it is"; continue; }
    nice -n 15 $PY "$P/ceil_read.py" "$RUN.jsonl" --out "$D/L${L}_verdict.json" > "$D/L${L}_verdict.txt" 2>&1
    grep -E '^(===|pass  N|FAIL  N)' "$D/L${L}_verdict.txt"
    # reported, gating nothing: the +-2.3 answers alone
    nice -n 15 $PY "$P/trial_read.py" "$RUN.jsonl" --gates L45 --cuts 48 --doses=-2.3,2.3 --out "$D/L${L}_upper.json" > "$D/L${L}_upper.txt" 2>&1
    grep -E '^  (pass|FAIL)  G[1-4]' "$D/L${L}_upper.txt" | sed 's/^/the upper dose alone (reported): /'
done
