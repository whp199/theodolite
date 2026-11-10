#!/usr/bin/env bash
# Layer 42 (L42-break_rel), taken by a volunteer (anon, claim a56863f2) on an RTX PRO 6000 Blackwell, a card class of
# its own, checked and spot-checked at home and accepted on 10 November 2026 (census_intake.py; its provenance beside
# the run; written in two sessions, resumed once, the compiled unpacker's self-test the reference's bits in both).
# Read that day with the trial's own reader, unchanged (atlas_data/census/trials/prereg, its SHA256SUMS
# checked first), on the CPU, as layers 45, 40, 47, 46 and 41 were. No gates were fixed for layer 42 before it ran: it
# is read against layer 45's, after the fact. It was ranked at acceptance with the collapse rule.
# Peak memory 6.0 GB, 20 minutes, beside layer 43's read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L42-break_rel.jsonl
HERE=experiments/results/census_L42_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L42_verdict.json" > "$HERE/L42_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L42_pm1_verdict.json" > "$HERE/L42_pm1_verdict.txt" || true
# (the copies kept here leave out each reading's named_words, to keep the record small)
