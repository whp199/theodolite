#!/usr/bin/env bash
# Layer 48 (L48-break_rel), taken by a volunteer (anon, claim 5cd76843, who also took layers 42 and 50) on an RTX PRO
# 6000 Blackwell, a card class of its own, checked and spot-checked at home (its spot run on the home card once home
# layer 49 had ended) and accepted on 11 November 2026 (census_intake.py; its provenance beside the run; written in
# one session from 10 November 02:32Z, at the kit's commit from a clean tree). Read that day with the trial's own
# reader, unchanged (atlas_data/census/trials/prereg, its SHA256SUMS checked first), on the CPU, as layers 45, 40, 47,
# 46, 41, 42, 43 and 44 were, beside layer 49's read. No gates were fixed for layer 48 before it ran: it is read
# against layer 45's, after the fact. It was ranked at acceptance with the collapse rule.
# Peak resident memory 5.5 GB (time -v), 17 minutes (at nice 15), beside layer 49's read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L48-break_rel.jsonl
HERE=experiments/results/census_L48_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L48_verdict.json" > "$HERE/L48_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L48_pm1_verdict.json" > "$HERE/L48_pm1_verdict.txt" || true
# (the copies kept here leave out each reading's named_words, to keep the record small)
