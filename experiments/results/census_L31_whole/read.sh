#!/usr/bin/env bash
# Layer 31 (L31-match_wide), taken by a volunteer (anon, claim eda6b586, who took layers 32 to 35 with kits 4 and 5 and
# 42, 48 and 50 with kit 1) on an RTX PRO 6000 Blackwell Server Edition, a card class of its own, with kit 5
# (census-kit-5 at 3943703, a clean tree; begun 18 November 2026 at 03:41Z by the run's record, sent home that day at
# 22:25Z). Checked under kit 5's own file, read from its tag (census_volunteer.earlier_kit: kit 6 was cut while it
# ran), spot-checked at home and accepted on 19 November 2026 (census_intake.py; its provenance beside the run): spot
# chunks 47, 121, 328, 340 and 541 of 737, the first word-piece the home card's in 0.988 of answers, log-probabilities
# apart by a median 0.0051 and 0.117 at the 99th percentile, against match_wide's bars at 31 of 0.90, 0.013 and 0.26.
# Ranked at acceptance with the collapse rule. Read that day on the CPU by the registered reader for its trial, with
# its defaults: census_scaled_design/read_whole.sh checks atlas_data/census/trials/wide/prereg against its SHA256SUMS
# (hashed 16 November 2026 at 15:22Z, before any answer of the design existed), and reads N1 and N3 with wide_read.py,
# and its two readings on three questions a half, reported. match_wide has one dose, so there is no upper dose to read
# alone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 31
# As layers 33-37 were written up (ROADMAP §vi), gating nothing, seconds: from the run's units file (the ranking the page
# shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above the controls'
# median in their standard deviations, and what they write; how many neurons stand more than five of those deviations
# above it; the count with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049),
# and with the five strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L31-match_wide.units.jsonl <<'PY'
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
