#!/usr/bin/env bash
# Layer 44 (L44-break_rel), taken by a volunteer (Kimi-chan, claim 5c3e7396, which replaced their claim b81c805c when
# its secret was lost) on an RTX PRO 6000 Blackwell, a card class of its own, checked and spot-checked at home and
# accepted on 10 November 2026 (census_intake.py; its provenance beside the run). Read that day with the trial's own
# reader, unchanged (atlas_data/census/trials/prereg, its SHA256SUMS checked first), on the CPU, as layers 45, 40, 47,
# 46, 41, 42 and 43 were, beside the home card's run of layer 49. No gates were fixed for layer 44 before it ran: it is
# read against layer 45's, after the fact. It was ranked at acceptance with the collapse rule.
# Peak memory 6.0 GB, 25 minutes (at nice 15 after its first minutes, to leave the census run its core).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L44-break_rel.jsonl
HERE=experiments/results/census_L44_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L44_verdict.json" > "$HERE/L44_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L44_pm1_verdict.json" > "$HERE/L44_pm1_verdict.txt" || true
# (the copies kept here leave out each reading's named_words, to keep the record small)
