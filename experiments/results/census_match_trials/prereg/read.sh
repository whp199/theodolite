#!/usr/bin/env bash
# Read the match_ceil trial at layer 37 once it is written, on the CPU, with the reader and gates hashed before any
# answer of it existed: ceil_read.py (the three gates), then, gating nothing, the upper dose's answers alone.
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/match-trials}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/match
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
RUN=$D/L37-match_ceil-trial
[ -e "$RUN.t1" ] || { echo "layer 37: not written to its end yet"; exit 3; }
nice -n 15 $PY "$P/ceil_read.py" "$RUN.jsonl" --out "$D/L37_verdict.json" > "$D/L37_verdict.txt" 2>&1
grep -E '^(===|pass  N|FAIL  N)' "$D/L37_verdict.txt"
# reported, gating nothing: the upper dose's answers alone
nice -n 15 $PY "$P/trial_read.py" "$RUN.jsonl" --gates L45 --cuts 48 --doses=-2.55,2.55 --out "$D/L37_upper.json" > "$D/L37_upper.txt" 2>&1
grep -E '^  (pass|FAIL)  G[1-4]' "$D/L37_upper.txt" | sed 's/^/the upper dose alone (reported): /'
