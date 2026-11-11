#!/usr/bin/env bash
# The fifth whole break_rel layer on the home card (L49-break_rel, the home 5090, 10-11 November 2026, the kit's
# settings with --unpacker compiled; written at c155448 from a clean tree, its first batch the layer-49 home canary bit
# for bit; moved into its own systemd scope (census-L49) while it ran, and it kept writing), read on 11 November with
# the trial's own reader, unchanged (atlas_data/census/trials/prereg, its SHA256SUMS checked first), on the CPU, as
# layers 45, 40, 47, 46, 41, 42, 43 and 44 were. No gates were fixed for layer 49 before it ran: it is read against
# layer 45's, after the fact. Layer 49 (page 50) lies inside the layers the doses were calibrated for (41-51 counted
# from 1). The run ranked itself with the collapse rule, so its ranking is read as it stands.
# Peak resident memory 5.5 GB (time -v), 16 minutes (at nice 15), beside layer 48's read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/prereg
R=atlas_data/census/runs/L49-break_rel.jsonl
HERE=experiments/results/census_L49_whole
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c SHA256SUMS)
# the whole layer, as the trial was read (48 pieces, then the same answers cut to 32)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48,32 --out "$HERE/L49_verdict.json" > "$HERE/L49_verdict.txt" || true
# the +-1 answers alone (its G5b compares with the units file, which ranks all four doses: it fails by construction)
.venv/bin/python -I "$P/trial_read.py" "$R" --gates L45 --cuts 48 --doses=-1,1 --out "$HERE/L49_pm1_verdict.json" > "$HERE/L49_pm1_verdict.txt" || true
# (the copies kept here leave out each reading's named_words, to keep the record small)
