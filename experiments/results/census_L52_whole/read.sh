#!/usr/bin/env bash
# Layer 52 (L52-soft_ceil), taken by a volunteer (eigentactics, claim c40b4d3a, who took layer 51 and wrote kit 4's
# change for 24 GB cards) on an RTX 3090 24 GB, a card class of its own, with kit 4 (census-kit-4 at 81c13628, a clean
# tree; begun 16 November 2026 at 05:00Z by the run's record, sent home 17 November at 17:44Z). Checked under kit 4's
# own file, read from its tag (census_volunteer.earlier_kit: kit 5 was cut while it ran), spot-checked at home and
# accepted on 17 November 2026 (census_intake.py; its provenance beside the run): spot chunks 291, 379, 642, 716 and
# 719 of 737, the first word-piece the home card's in 0.995 of answers, log-probabilities apart by a median 0.0077 and
# 0.132 at the 99th percentile, against soft_ceil's bars of 0.90, 0.013 and 0.29. Ranked at acceptance with the
# collapse rule. Its neurons and its controls were pushed on the same 3090; its trial was taken on the home card.
# Read that day on the CPU by the registered reader for its trial, with its defaults:
# census_scaled_design/read_whole.sh checks atlas_data/census/trials/soft/prereg_L52 against its SHA256SUMS (hashed 13
# November 2026 at 02:50Z, before any answer of the trial existed), reads N1, N2 and N3 with ceil_read.py, then the
# upper dose's answers alone (L52_upper.*), reported and gating nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 52
# As layers 33-37 were written up (ROADMAP §vi), gating nothing, seconds: from the run's units file (the ranking the page
# shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above the controls'
# median in their standard deviations, and what they write; how many neurons stand more than five of those deviations
# above it; the count with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049),
# and with the five strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L52-soft_ceil.units.jsonl <<'PY'
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
