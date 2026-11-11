#!/usr/bin/env bash
# Layer 50 (L50-break_rel), taken by a volunteer (anon, claim b153da82, who also took layers 42 and 48) on an RTX PRO
# 6000 Blackwell, a card class of its own, checked and spot-checked at home (its spot run on the home card) and
# accepted on 11 November 2026 (census_intake.py; its provenance beside the run; written in one session from
# 10 November 16:13Z, at the kit's commit from a clean tree). Read that day with the trial's own reader, unchanged
# (atlas_data/census/trials/prereg, its SHA256SUMS checked first), on the CPU, as layers 45, 40, 47, 46, 41, 42, 43,
# 44, 48 and 49 were. No gates were fixed for layer 50 before it ran: it is read against layer 45's, after the fact.
# Layer 50 (page 51) is the last of the layers the doses were calibrated for (41-51 counted from 1). It was ranked at
# acceptance with the collapse rule.
# Peak resident memory 5.5 GB (time -v), 15 minutes (at nice 15).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L50-break_rel.jsonl
HERE=experiments/results/census_L50_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L50_verdict.json" > "$HERE/L50_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L50_pm1_verdict.json" > "$HERE/L50_pm1_verdict.txt" || true
# (the copies kept here leave out each reading's named_words, to keep the record small)
