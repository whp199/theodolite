#!/usr/bin/env bash
# Layer 38 (L38-soft_ceil), taken by a volunteer (Zlorthishen, anon on the board, claim 3853d507, who took layers 39
# and 37 before it) on an RTX 5090, the home card's bits, with kit 5 (census-kit-5 at 3943703, a clean tree; begun 17
# November 2026 at 22:36Z by the run's record, sent home 18 November at 22:16Z). Checked under kit 5's own file, read
# from its tag (census_volunteer.earlier_kit: kit 6 was cut while it ran), spot-checked at home and accepted on 19
# November 2026 (census_intake.py; its provenance beside the run): spot chunks 9, 22, 324, 382 and 423 of 737, 3,840 of
# 3,840 answers identical bit for bit. Ranked at acceptance with the collapse rule. Read that day on the CPU by the
# registered reader for its trial, with its defaults: census_scaled_design/read_whole.sh checks
# atlas_data/census/trials/soft/prereg against its SHA256SUMS (hashed 12 November 2026 at 22:34Z, before any answer of
# that trial existed), reads N1, N2 and N3 with ceil_read.py, then the upper dose's answers alone (L38_upper.*),
# reported and gating nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 38
# As layers 33-37 were written up (ROADMAP §vi), gating nothing, seconds: from the run's units file (the ranking the page
# shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above the controls'
# median in their standard deviations, and what they write; how many neurons stand more than five of those deviations
# above it; the count with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049),
# and with the five strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L38-soft_ceil.units.jsonl <<'PY'
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
