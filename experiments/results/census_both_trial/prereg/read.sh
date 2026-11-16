#!/usr/bin/env bash
# Read match_both's trial at layer 30 once it is written, on the CPU, with the reader and gates hashed before any answer
# of the design existed: both_read.py (two gates, N1 and N3, and the readings it reports). Not read again once read.
#     bash read.sh
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/both-trial}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/both
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
L=30
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
RUN=$D/L$L-match_both-trial
[ -e "$RUN.t1" ] || { echo "layer $L: not written to its end yet"; exit 3; }
[ -e "$D/L${L}_verdict.json" ] && { echo "layer $L: already read, kept as it is"; exit 0; }
nice -n 15 $PY "$P/both_read.py" "$RUN.jsonl" --out "$D/L${L}_verdict.json" > "$D/L${L}_verdict.txt" 2>&1
grep -E '^(===|pass  N|FAIL  N|reported, (the|three))' "$D/L${L}_verdict.txt"
date -u +%FT%TZ > "$D/trial.done"
