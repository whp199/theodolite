#!/usr/bin/env python3
"""The neurons the edge_things trial asks at layer 29: 1,022 of the layer's 21,504, drawn without replacement by a seed
fixed here (20261122 x 100 + 29 = 2026112229), sorted, from the neurons no census run has asked at 29. The runs so left
out are named in ASKED, read from their metas, and their sha256 printed with the list to stderr: the 1,022 of
match_things's walk trial at 29 (18 November 2026) and the 32 of the calib_edge ladder that set this design's doses
(22 November 2026). The script refuses to draw if a census run at 29 is not in ASKED (a meta under atlas_data/census,
volunteers' folders aside, whose layer is 29). Prints the neurons comma-joined, as census_run.py --neurons takes them.

    THEODOLITE_TRIAL_REPO=<checkout> python draw_neurons.py 29 > neurons_L29.txt
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
ASKED = {29: ['trials/things_walk/L29-match_things-trial.meta.json', 'edge_pilot/L29-calib_edge.meta.json']}
D_FF, N = 21504, 1022
layer = int(sys.argv[1])
if layer not in ASKED:
    raise SystemExit(f'layer {layer}: this registration draws neurons at 29 alone')
named = {ATLAS / p for p in ASKED[layer]}
found = set()
for p in ATLAS.rglob('*.meta.json'):
    if 'volunteer' in p.relative_to(ATLAS).parts:
        continue
    try:
        m = json.loads(p.read_text())
    except (OSError, ValueError):
        continue
    if isinstance(m, dict) and m.get('layer') is not None and int(m['layer']) == layer and m.get('neuron_ids'):
        found.add(p)
if found != named:
    raise SystemExit(f'layer {layer}: the census runs found there ({sorted(map(str, found))}) are not the ones named '
                     f'({sorted(map(str, named))}): nothing drawn')
asked = set()
for p in sorted(named):
    m = json.loads(p.read_text())
    assert int(m['layer']) == layer and int(m['d_ff']) == D_FF, p
    asked |= set(cz.parse_neurons(m['neuron_ids'], D_FF))
    print(f'left out: {p.relative_to(ATLAS)}, sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}', file=sys.stderr)
seed = 20261122 * 100 + layer
pool = np.array(sorted(set(range(D_FF)) - asked))
ids = np.sort(np.random.default_rng(seed).choice(pool, N, replace=False))
print(f'layer {layer}: seed {seed}; {len(asked)} asked before left out, {len(pool)} to draw from, {N} drawn', file=sys.stderr)
print(",".join(str(int(i)) for i in ids))
