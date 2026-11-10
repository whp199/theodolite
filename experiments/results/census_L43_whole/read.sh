#!/usr/bin/env bash
# The fourth whole break_rel layer on the home card (L43-break_rel, the home 5090, 9-10 November 2026, the kit's
# settings with --unpacker compiled; written at 601b77d, its first batch the layer-43 home canary bit for bit; its
# meta's git_clean is false, the checkout's uncommitted change at its start being the board's own files), read
# on 10 November with the trial's own reader, unchanged (atlas_data/census/trials/prereg, its SHA256SUMS checked
# first), on the CPU, as layers 45, 40, 47, 46 and 41 were. No gates were fixed for layer 43 before it ran: it is read
# against layer 45's, after the fact. The run ranked itself with the collapse rule, so its ranking is read as it stands.
# Peak memory 6.0 GB, 24 minutes, beside layer 42's read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L43-break_rel.jsonl
HERE=experiments/results/census_L43_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L43_verdict.json" > "$HERE/L43_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L43_pm1_verdict.json" > "$HERE/L43_pm1_verdict.txt" || true
# (the copies kept here leave out each reading's named_words, to keep the record small)
