#!/usr/bin/env python3
"""The neurons a match_wide trial asks: 1,022 of a layer's 21,504, drawn without replacement by a seed fixed here
(20261116 * 100 + the layer, counted from 0), sorted, from the neurons no census run has asked at that layer: at
layer 31 the 1,022 of match_ceil's trial there (atlas_data/census/trials/match_walk/L31-match_ceil-trial.meta.json);
at layer 30 the 64 of the two ladders of 28 October (atlas_data/census/runs/L30-calib_rel and L30-calib_ceil). The
lists so left out are read from those runs' metas and their sha256 printed with the list to stderr. Prints the
neurons comma-joined, as census_run.py --neurons takes them.

    THEODOLITE_TRIAL_REPO=<checkout> python draw_neurons.py 31 > neurons_L31.txt
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
ASKED = {31: [ATLAS / 'trials/match_walk/L31-match_ceil-trial.meta.json'],
         30: [ATLAS / 'runs/L30-calib_rel.meta.json', ATLAS / 'runs/L30-calib_ceil.meta.json']}
D_FF, N = 21504, 1022
layer = int(sys.argv[1])
if layer not in ASKED:
    raise SystemExit(f'layer {layer}: this registration draws neurons at layers {sorted(ASKED)} alone')
asked = set()
for p in ASKED[layer]:
    m = json.loads(p.read_text())
    assert int(m['layer']) == layer and int(m['d_ff']) == D_FF, p
    asked |= set(cz.parse_neurons(m['neuron_ids'], D_FF))
    print(f'left out: {p.name}, sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}', file=sys.stderr)
pool = np.array(sorted(set(range(D_FF)) - asked))
ids = np.sort(np.random.default_rng(20261116 * 100 + layer).choice(pool, N, replace=False))
print(f'layer {layer}: {len(asked)} asked before left out, {len(pool)} to draw from, {N} drawn', file=sys.stderr)
print(",".join(str(int(i)) for i in ids))
