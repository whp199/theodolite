#!/usr/bin/env bash
# Layer 51 (L51-break_rel), taken by a volunteer (eigentactics, claim 6d708804, who wrote kit 4's change for 24 GB cards)
# on an RTX 3090 24 GB, a card class of its own and the first layer taken on a 24 GB card, with kit 4 (census-kit-4 at
# 81c13628, a clean tree; begun 14 November 2026 at 03:28Z by the run's record). Checked and spot-checked at home and
# accepted on 16 November 2026 (census_intake.py; its provenance beside the run): spot chunks 2, 218, 356, 612 and 615
# of 737, the first word-piece the home card's in 0.989 of answers, log-probabilities apart by a median 0.0111 and
# 0.150 at the 99th percentile, against break_rel's bars of 0.90, 0.028 and 0.37. Ranked at acceptance with the
# collapse rule. Its neurons and its controls were pushed on the same 3090; its trial was taken on the home card.
# Read that day (the files here written at 03:26Z) on the CPU by the registered reader for its trial, with its
# defaults: census_neighbour_trials/read_whole.sh checks atlas_data/census/trials/neighbours/prereg against its
# SHA256SUMS (hashed 12 November 2026 at 15:14Z, before any answer of the trials existed) and reads N1, N2 and N3 with
# neighbour_read.py, as layer 51's trial was read.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
bash experiments/results/census_neighbour_trials/read_whole.sh 51
# Added with the write-up of 16 November 2026 (ROADMAP §vi), gating nothing, seconds: from the run's units file (the
# ranking the page shows; its mass is the reader's mass_c to 0.0005, G5b) how far the strongest controls stand above
# the controls' median in their standard deviations, and what they write; how many neurons stand more than five of
# those deviations above it (the count layers 48-50's write-up used; it gives 1,408, 1,407 and 1,353 there); the count
# with q <= 0.05 as readlib.py makes it (Benjamini-Hochberg on (1 + controls at or above) / 2,049), and with the five
# strongest controls set aside.
.venv/bin/python -I - atlas_data/census/runs/L51-break_rel.units.jsonl <<'PY'
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
