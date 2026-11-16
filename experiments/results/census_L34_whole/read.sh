#!/usr/bin/env bash
# Layer 34 (L34-match_ceil), taken by a volunteer (anon, claim eaef4eff, who took layers 42, 48 and 50 with kit 1) on an
# RTX PRO 6000 Blackwell Server Edition, a card class of its own, with kit 4 (census-kit-4 at 81c13628, a clean tree;
# begun 14 November 2026 at 05:26Z by the run's record). Checked and spot-checked at home and accepted on 16 November
# 2026 (census_intake.py; its provenance beside the run): spot chunks 47, 168, 384, 565 and 727 of 737, the first
# word-piece the home card's in 0.990 of answers, log-probabilities apart by a median 0.0051 and 0.115 at the 99th
# percentile, against match_ceil's bars of 0.90, 0.012 and 0.25. Ranked at acceptance with the collapse rule.
# Read that day (the files here written 02:56Z to 02:59Z) on the CPU by the registered reader for its trial, with its
# defaults: census_scaled_design/read_whole.sh checks atlas_data/census/trials/match_walk/prereg against its SHA256SUMS
# (hashed 13 November 2026 at 07:25Z, before any answer of the walk existed), reads N1, N2 and N3 with ceil_read.py,
# then the upper dose's answers alone (L34_upper.*), reported and gating nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 34
# Added with the write-up of 16 November 2026 (ROADMAP §vi), gating nothing, seconds: from the run's units file (the
# ranking the page shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above
# the controls' median in their standard deviations, and what they write; how many neurons stand more than five of
# those deviations above it (the count layers 48-50's write-up used; it gives 1,408, 1,407 and 1,353 there); the count
# with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049), and with the five
# strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L34-match_ceil.units.jsonl <<'PY'
import json, sys
import numpy as np
U = [json.loads(s) for s in open(sys.argv[1])]
neu = np.array([u["mass"] for u in U if u["kind"] == "neuron"])
ctl = sorted((u for u in U if u["kind"] == "random"), key=lambda u: -u["mass"])
c = np.array([u["mass"] for u in ctl]); med, sd = float(np.median(c)), float(np.std(c))
def q05(c):
    s = np.sort(c); p = np.sort((1 + len(s) - np.searchsorted(s, neu, side="left")) / (len(s) + 1))
    return int((np.minimum.accumulate((p * len(p) / np.arange(1, len(p) + 1))[::-1]) <= 0.05).sum())
for u in ctl[:5]:
    print(f"control {(u['mass'] - med) / sd:5.2f} sd over the controls' median:", " ".join(w[0] for w in u["words"][:6]))
print("neurons more than five sd over it:", int((neu > med + 5 * sd).sum()), "of", len(neu))
print("q <= 0.05:", q05(c), "; with the five strongest controls set aside:", q05(c[5:]))
PY
