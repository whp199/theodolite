#!/usr/bin/env bash
# Layer 39 (L39-break_rel), taken by a volunteer new to the census (anon, claim 1f217635) on an RTX 5090 that writes the
# home card's bits, with kit 4 (census-kit-4 at 81c13628, a clean tree; begun 15 November 2026 at 05:42Z by the run's
# record; its RESERVE, START and DONE were posted in one comment at 18:50Z, so the board never held the layer for it).
# Checked and spot-checked at home and accepted on 16 November 2026 (census_intake.py; its provenance beside the run):
# spot chunks 77, 208, 295, 483 and 525 of 737, all 3,840 answers the home card's bit for bit. Ranked at acceptance
# with the collapse rule.
# Read that day (the files here written at 03:29Z) on the CPU by the registered reader for its trial, with its
# defaults: census_neighbour_trials/read_whole.sh checks atlas_data/census/trials/neighbours/prereg against its
# SHA256SUMS (hashed 12 November 2026 at 15:14Z, before any answer of the trials existed) and reads N1, N2 and N3 with
# neighbour_read.py, as layer 39's trial was read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_neighbour_trials/read_whole.sh 39
# Added with the write-up of 16 November 2026 (ROADMAP §vi), gating nothing, seconds: from the run's units file (the
# ranking the page shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above
# the controls' median in their standard deviations, and what they write; how many neurons stand more than five of
# those deviations above it (the count layers 48-50's write-up used; it gives 1,408, 1,407 and 1,353 there); the count
# with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049), and with the five
# strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L39-break_rel.units.jsonl <<'PY'
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
