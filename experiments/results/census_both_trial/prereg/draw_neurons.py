#!/usr/bin/env python3
"""The neurons match_both's trial at layer 30 asks: 1,022 of the layer's 21,504, drawn without replacement by a seed
fixed here (2026111699), sorted, from the neurons no census run has asked at layer 30: the 1,022 of match_wide's
trial there (atlas_data/census/trials/wide/L30-match_wide-trial.meta.json) and the 64 of the two ladders of 28 October
(atlas_data/census/runs/L30-calib_rel and L30-calib_ceil). The lists so left out are read from those runs' metas and
their sha256 printed with the list to stderr. Prints the neurons comma-joined, as census_run.py --neurons takes them.

    THEODOLITE_TRIAL_REPO=<checkout> python draw_neurons.py 30 > neurons_L30.txt
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get('THEODOLITE_TRIAL_REPO') or '/home/eric/bin/ai/theodolite2')
from backend import census as cz  # noqa: E402

ATLAS = Path('/home/eric/bin/ai/theodolite2/atlas_data/census')
ASKED = {30: [ATLAS / 'trials/wide/L30-match_wide-trial.meta.json',
              ATLAS / 'runs/L30-calib_rel.meta.json', ATLAS / 'runs/L30-calib_ceil.meta.json']}
D_FF, N, SEED = 21504, 1022, 2026111699
layer = int(sys.argv[1])
if layer not in ASKED:
    raise SystemExit(f'layer {layer}: this registration draws neurons at layer 30 alone')
asked = set()
for p in ASKED[layer]:
    m = json.loads(p.read_text())
    assert int(m['layer']) == layer and int(m['d_ff']) == D_FF, p
    asked |= set(cz.parse_neurons(m['neuron_ids'], D_FF))
    print(f'left out: {p.name}, sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}', file=sys.stderr)
pool = np.array(sorted(set(range(D_FF)) - asked))
ids = np.sort(np.random.default_rng(SEED).choice(pool, N, replace=False))
print(f'layer {layer}: {len(asked)} asked before left out, {len(pool)} to draw from, {N} drawn', file=sys.stderr)
print(",".join(str(int(i)) for i in ids))
