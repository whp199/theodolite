#!/usr/bin/env bash
# Read the soft_ceil trial at layer 38 once it is written, on the CPU, with the reader and gates hashed before any
# answer of it existed: ceil_read.py (the three gates), then, gating nothing, the upper dose's answers alone.
set -uo pipefail
R=${R:-/home/eric/bin/ai/theodolite2/.claude/worktrees/soft-trials}
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/soft
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || echo "the pre-registered files differ from SHA256SUMS: see AMENDMENTS.txt"
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$R"
RUN=$D/L38-soft_ceil-trial
[ -e "$RUN.t1" ] || { echo "layer 38: not written to its end yet"; exit 3; }
nice -n 15 $PY "$P/ceil_read.py" "$RUN.jsonl" --out "$D/L38_verdict.json" > "$D/L38_verdict.txt" 2>&1
grep -E '^(===|pass  N|FAIL  N)' "$D/L38_verdict.txt"
# reported, gating nothing: the +-2.3 answers alone, the first trial's lower dose asked again of new neurons
nice -n 15 $PY "$P/trial_read.py" "$RUN.jsonl" --gates L45 --cuts 48 --doses=-2.3,2.3 --out "$D/L38_upper.json" > "$D/L38_upper.txt" 2>&1
grep -E '^  (pass|FAIL)  G[1-4]' "$D/L38_upper.txt" | sed 's/^/the upper dose alone (reported): /'
