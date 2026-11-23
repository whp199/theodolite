#!/usr/bin/env bash
# Read the edge_things trial at layer 29 once it is written, on the CPU, with the reader and gates hashed before any
# answer of the trial existed: things_read.py, byte for byte the reader of match_things's trial at 30 and of its walk
# (N1 and N3 gate; the readings it reports gate nothing). Not read again once read.
#     bash read.sh
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/edge-trial}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/edge
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
L=29
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
RUN=$D/L$L-edge_things-trial
[ -e "$RUN.t1" ] || { echo "layer $L: not written to its end yet"; exit 3; }
[ -e "$D/L${L}_verdict.json" ] && { echo "layer $L: already read, kept as it is"; exit 0; }
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 $PY "$P/things_read.py" "$RUN.jsonl" --out "$D/L${L}_verdict.json" > "$D/L${L}_verdict.txt" 2>&1
grep -E '^(===|pass  N|FAIL  N|reported, (the|three))' "$D/L${L}_verdict.txt"
