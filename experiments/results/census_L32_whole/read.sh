#!/usr/bin/env bash
# Layer 32 (L32-match_ceil), taken by a volunteer (anon, claim 943783a3, who took layers 33, 34 and 35 with kit 4
# and 42, 48 and 50 with kit 1) on an RTX PRO 6000 Blackwell Server Edition, a card class of its own, with kit 5
# (census-kit-5 at 3943703, a clean tree; begun 17 November 2026 at 11:11Z by the run's record, sent home 18 November
# at 03:35Z). Checked, spot-checked at home and accepted on 18 November 2026 (census_intake.py; its provenance beside
# the run): spot chunks 91, 193, 432, 493 and 654 of 737, the first word-piece the home card's in 0.986 of answers,
# log-probabilities apart by a median 0.0052 and 0.117 at the 99th percentile, against match_ceil's bars of 0.90, 0.012
# and 0.25. Ranked at acceptance with the collapse rule.
# Read that day on the CPU by the registered reader for its trial, with its defaults:
# census_scaled_design/read_whole.sh checks atlas_data/census/trials/match_walk/prereg against its SHA256SUMS (hashed 13
# November 2026 at 07:25Z, before any answer of the walk existed), reads N1, N2 and N3 with ceil_read.py, then the
# upper dose's answers alone (L32_upper.*), reported and gating nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 32
# As layers 33-37 were written up (ROADMAP §vi), gating nothing, seconds: from the run's units file (the ranking the page
# shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above the controls'
# median in their standard deviations, and what they write; how many neurons stand more than five of those deviations
# above it; the count with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049),
# and with the five strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L32-match_ceil.units.jsonl <<'PY'
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
print("above every control:", int((neu > c.max()).sum()))
print("q <= 0.05:", q05(c), "; with the five strongest controls set aside:", q05(c[5:]))
PY
