#!/usr/bin/env python3
"""Report-only readings of the match_things walk (RULES.md, "Reported, gating nothing"), computed after the verdicts
(RESULT.md) from the runs in atlas_data/census/trials (not in the repository):
1. each trial's twelve unpushed answers against the trial at 30's, bit for bit (token_ids and logprob strings);
2. the strongest controls, and q <= 0.05 with the five strongest set aside, as census_L34_whole/read.sh computes them.

    .venv/bin/python -I experiments/results/census_things_walk/reported.py > experiments/results/census_things_walk/reported.txt
"""
import json
from pathlib import Path
import numpy as np
T = str(Path(__file__).resolve().parents[3] / 'atlas_data' / 'census' / 'trials')
def baselines(path):
    out = {}
    with open(path) as f:
        for line in f:
            if '"baseline"' not in line[:80]:
                continue
            r = json.loads(line)
            if r.get('kind') == 'baseline':
                out[r['prompt']] = (r['token_ids'], r['logprob'], r['n_gen'], r['finish'])
    return out
ref = baselines(f'{T}/things/L30-match_things-trial.jsonl')
print('L30 baselines:', len(ref), sorted(ref))
for L in (53, 29, 54, 55):
    b = baselines(f'{T}/things_walk/L{L}-match_things-trial.jsonl')
    same_tok = sum(b[p][0] == ref[p][0] for p in ref if p in b)
    same_lp = sum(b[p][1] == ref[p][1] for p in ref if p in b)
    diff = [p for p in ref if p in b and b[p][:2] != ref[p][:2]]
    print(f'L{L}: {len(b)} unpushed answers; token_ids equal to L30 {same_tok}/12, logprobs equal {same_lp}/12; differ: {diff}')
for L in (53, 29, 54, 55):
    U = [json.loads(s) for s in open(f'{T}/things_walk/L{L}-match_things-trial.units.jsonl')]
    neu = np.array([u['mass'] for u in U if u['kind'] == 'neuron'])
    ctl = sorted((u for u in U if u['kind'] == 'random'), key=lambda u: -u['mass'])
    c = np.array([u['mass'] for u in ctl]); med, sd = float(np.median(c)), float(np.std(c))
    def q05(c):
        s = np.sort(c); p = np.sort((1 + len(s) - np.searchsorted(s, neu, side='left')) / (len(s) + 1))
        return int((np.minimum.accumulate((p * len(p) / np.arange(1, len(p) + 1))[::-1]) <= 0.05).sum())
    print(f'--- L{L}: {len(neu)} neurons, {len(c)} controls')
    for u in ctl[:5]:
        print(f'  control {(u["mass"] - med) / sd:5.2f} sd over the controls\' median:', ' '.join(w[0][:40] for w in u['words'][:6]))
    print('  neurons more than five sd over it:', int((neu > med + 5 * sd).sum()), 'of', len(neu))
    print('  q <= 0.05:', q05(c), '; with the five strongest controls set aside:', q05(c[5:]))
