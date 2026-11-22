#!/usr/bin/env bash
# Layer 54 (L54-match_things), taken by a volunteer (anon, claim 0471bd29, who took layers 30 to 35 with kits 4 to 6
# and 42, 48 and 50 with kit 1) on an RTX PRO 6000 Blackwell Server Edition, a card class of its own, with kit 7
# (census-kit-7 at fbcd44f, a clean tree; begun 20 November 2026 at 23:29Z by the run's record, in one session, sent
# home 22 November at 01:03Z). Checked, spot-checked at home and accepted on 22 November 2026 (census_intake.py; its
# provenance beside the run): spot chunks 198, 526, 762, 990 and 1267 of 1,473, the first word-piece the home card's in
# 0.993 of answers, log-probabilities apart by a median 0.0037 and 0.112 at the 99th percentile, against
# match_things's bars at 53 and 54 of 0.90, 0.0091 and 0.26. Ranked at acceptance with the collapse rule. Read that day
# on the CPU by the registered reader for its walk's trial, with its defaults: census_scaled_design/read_whole.sh
# checks atlas_data/census/trials/things_walk/prereg against its SHA256SUMS (hashed 18 November 2026 at 16:18:23Z,
# before any answer of the walk existed), and reads with things_read.py: N1 and N3 gate (N2, the lower dose alone,
# gates only a design of one dose), and each dose alone and two readings on three questions a half, reported.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 54
# As layers 33-37 were written up (ROADMAP §vi), gating nothing, seconds: from the run's units file (the ranking the page
# shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above the controls'
# median in their standard deviations, and what they write; how many neurons stand more than five of those deviations
# above it; the count with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049),
# and with the five strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L54-match_things.units.jsonl <<'PY'
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
