#!/usr/bin/env bash
# Read the match_wide trial at each layer of LAYERS once it is written, on the CPU, with the reader and gates hashed
# before any answer of the design existed: wide_read.py (two gates, N1 and N3, and the readings it reports). A layer
# already read is not read again.
#     LAYERS="31" bash read.sh
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/wide-trials}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/wide
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
LAYERS=${LAYERS:?give LAYERS, the layers to read}
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
for L in $LAYERS; do
    case $L in 31|30) ;; *) echo "layer $L is not in this walk"; continue;; esac
    RUN=$D/L$L-match_wide-trial
    [ -e "$RUN.t1" ] || { echo "layer $L: not written to its end yet"; continue; }
    [ -e "$D/L${L}_verdict.json" ] && { echo "layer $L: already read, kept as it is"; continue; }
    nice -n 15 $PY "$P/wide_read.py" "$RUN.jsonl" --out "$D/L${L}_verdict.json" > "$D/L${L}_verdict.txt" 2>&1
    grep -E '^(===|pass  N|FAIL  N|reported, three)' "$D/L${L}_verdict.txt"
done
