#!/usr/bin/env bash
# Layer 30 (L30-match_things), taken by a volunteer (anon, claim 6413bb68, who took layers 31 to 35 with kits 4 and 5
# and 42, 48 and 50 with kit 1) on an RTX PRO 6000 Blackwell Server Edition, a card class of its own, with kit 6
# (census-kit-6 at 6574729, a clean tree; begun 18 November 2026 at 22:32Z by the run's record, in four sessions, sent
# home 20 November at 23:22Z). Checked under kit 6's own file, read from its tag (census_volunteer.earlier_kit: kit 7
# was cut while it ran), spot-checked at home and accepted on 21 November 2026 (census_intake.py; its provenance beside
# the run): spot chunks 147, 336, 362, 552 and 577 of 1,473, the first word-piece the home card's in 0.988 of answers,
# log-probabilities apart by a median 0.0047 and 0.112 at the 99th percentile, against match_things's bars at 30 of
# 0.90, 0.013 and 0.26. Ranked at acceptance with the collapse rule. Read that day on the CPU by the registered reader
# for its trial, with its defaults: census_scaled_design/read_whole.sh checks atlas_data/census/trials/things/prereg
# against its SHA256SUMS (hashed 18 November 2026 at 00:20:38Z, before any answer of the design existed), and reads with
# things_read.py: N1 and N3 gate (N2, the lower dose alone, gates only a design of one dose), and each dose alone and
# two readings on three questions a half, reported.
# The first check of this bundle (and of layer 53's) refused it: the intake capped the answers unpacked at a flat 4 GB,
# a layer of 565,254 answers, and this layer has twice that. The cap is now counted from the answers the kit's plan at
# the layer asks for (census_intake.MAX_ANSWER_BYTES); the bundle was checked again under it and passed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_scaled_design/read_whole.sh 30
# As layers 33-37 were written up (ROADMAP §vi), gating nothing, seconds: from the run's units file (the ranking the page
# shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above the controls'
# median in their standard deviations, and what they write; how many neurons stand more than five of those deviations
# above it; the count with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049),
# and with the five strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L30-match_things.units.jsonl <<'PY'
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
