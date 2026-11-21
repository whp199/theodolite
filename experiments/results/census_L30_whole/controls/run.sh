#!/usr/bin/env bash
# Why layer 30, read whole, missed the replication bar its trial passed: the trial's own 1,022 neurons read again with
# the registered reader (census/trials/things/prereg/things_read.py, unchanged), swapping one thing at a time between
# the trial's run (the home card's answers, 1,024 controls drawn with seed 10) and the volunteer's whole run (another
# card class's answers, the kit's 2,048 controls drawn with seed 7). Reported, gating nothing, and not registered:
# written on 21 November 2026 after the whole layer's verdict was seen. CPU, a few minutes; the runs it builds go to
# OUT (default atlas_data/census/diagnostic/L30_controls), the readings' text beside this file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
HERE=experiments/results/census_L30_whole/controls
PY=${PY:-$ROOT/.venv/bin/python}
OUT=${OUT:-${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}/census/diagnostic/L30_controls}
P=${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}/census/trials/things/prereg
T=${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}/census/trials/things/L30-match_things-trial
W=${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}/census/runs/L30-match_things
(cd "$P" && sha256sum -c --quiet SHA256SUMS)
mkdir -p "$OUT"/{sub30,mixA,mixB,half0,half1024}
# the two cards' answers to the same 49,056 jobs (the trial's neurons, each question and dose)
nice -n 15 "$PY" -I $HERE/cmp_trial_neurons.py "$T.jsonl" "$W.jsonl" | tee $HERE/cards.txt
# sub30: the volunteer's answers for the trial's neurons, the kit's 2,048 controls
nice -n 15 "$PY" -I $HERE/make_sub30.py "$W.jsonl" "$W.meta.json" "$T.meta.json" "$OUT/sub30"
# mixA: the home card's answers for them, the kit's 2,048 controls; mixB: the volunteer's, the trial's 1,024
"$PY" -I $HERE/make_mix.py "$T.jsonl" "$T.meta.json" "$OUT/sub30/L30-match_things.jsonl" 2048 "$OUT/mixA"
"$PY" -I $HERE/make_mix.py "$OUT/sub30/L30-match_things.jsonl" "$OUT/sub30/L30-match_things.meta.json" "$T.jsonl" 1024 "$OUT/mixB"
# half0, half1024: the volunteer's answers for them, each half of the kit's 2,048 controls alone
for lo in 0 1024; do
    "$PY" -I $HERE/make_half.py "$OUT/sub30/L30-match_things.jsonl" "$OUT/sub30/L30-match_things.meta.json" $lo "$OUT/half$lo"
done
export CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO="$ROOT"
for X in sub30 mixA mixB half0 half1024; do
    nice -n 15 "$PY" -I "$P/things_read.py" "$OUT/$X/L30-match_things.jsonl" --out "$OUT/$X/verdict.json" > $HERE/$X.txt 2>&1 || true
    echo "== $X"; grep -E '^(pass|FAIL)  N|    (pass|FAIL)  G[13]' $HERE/$X.txt
done
# G5b (the run's units file against the reading) fails on every one of these: they have no units file of their own
# the controls' push sizes in the trial's run and in the kit's (on the trial's neurons)
nice -n 15 "$PY" -I $HERE/ceilings.py "$T.jsonl" "$OUT/sub30/L30-match_things.jsonl" | tee $HERE/ceilings.txt
