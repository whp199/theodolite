#!/usr/bin/env python3
"""The neurons match_things's walk asks at one of its layers (25 to 29 and 53 to 58): 1,022 of the layer's 21,504,
drawn without replacement by a seed fixed here (20261118 x 100 + the layer: 2026111825 ... 2026111858), sorted, from
the neurons no census run has asked at that layer. The runs so left out are named in ASKED, read from their metas,
and their sha256 printed with the list to stderr: at 53 the 1,022 of soft_ceil's trial (16 November); at 57 the
507 of break_rel's probe and the 64 of the two ladders before it (29 October). No census run has asked a neuron at
25 to 29, 54, 55, 56 or 58. The home canaries (volunteers/home_canaries, each plan batch 0 under break_rel: the first
few neurons of a layer) are not left out, as no earlier draw left them out. The script refuses to draw if a census run
at the layer is not in ASKED (a meta under atlas_data/census, volunteers' folders aside, whose layer is the layer).
Prints the neurons comma-joined, as census_run.py --neurons takes them.

    THEODOLITE_TRIAL_REPO=<checkout> python draw_neurons.py 53 > neurons_L53.txt
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
WALK = (25, 26, 27, 28, 29, 53, 54, 55, 56, 57, 58)
ASKED = {53: ['trials/soft_walk_up/L53-soft_ceil-trial.meta.json'],
         57: ['trials/L57-break_rel-probe.meta.json', 'runs/L57-calib_rel.meta.json', 'runs/L57-calib_ceil.meta.json']}
D_FF, N = 21504, 1022
layer = int(sys.argv[1])
if layer not in WALK:
    raise SystemExit(f'layer {layer}: this registration draws neurons at {", ".join(map(str, WALK))} alone')
named = {ATLAS / p for p in ASKED.get(layer, [])}
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
seed = 20261118 * 100 + layer
pool = np.array(sorted(set(range(D_FF)) - asked))
ids = np.sort(np.random.default_rng(seed).choice(pool, N, replace=False))
print(f'layer {layer}: seed {seed}; {len(asked)} asked before left out, {len(pool)} to draw from, {N} drawn', file=sys.stderr)
print(",".join(str(int(i)) for i in ids))
