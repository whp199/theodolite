#!/usr/bin/env bash
# Layer 53 (L53-match_things), taken by a volunteer (Zlorthishen, anon on the board, claim 7ce4c92d, who took layers
# 37, 38 and 39 before it) on an RTX 5090, the home card's bits, with kit 7 (census-kit-7 at fbcd44f, a clean tree;
# begun 19 November 2026 at 21:35Z by the run's record, in one session, sent home 20 November at 23:07Z). Its first
# PROGRESS line, posted 21:39Z with its RESERVE and START, said 21% done, which the run's own record does not bear out;
# the spot-check is what vouches for the answers. Checked, spot-checked at home and accepted on 21 November 2026
# (census_intake.py; its provenance beside the run): spot chunks 164, 174, 403, 1112 and 1258 of 1,473, 3,840 of 3,840
# answers identical bit for bit. Ranked at acceptance with the collapse rule. Read that day on the CPU by the
# registered reader for its walk's trial, with its defaults: census_scaled_design/read_whole.sh checks
# atlas_data/census/trials/things_walk/prereg against its SHA256SUMS (hashed 18 November 2026 at 16:18:23Z, before any
# answer of the walk existed), and reads with things_read.py: N1 and N3 gate (N2, the lower dose alone, gates only a
# design of one dose), and each dose alone and two readings on three questions a half, reported.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 53
# As layers 33-37 were written up (ROADMAP §vi), gating nothing, seconds: from the run's units file (the ranking the page
# shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above the controls'
# median in their standard deviations, and what they write; how many neurons stand more than five of those deviations
# above it; the count with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049),
# and with the five strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L53-match_things.units.jsonl <<'PY'
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
