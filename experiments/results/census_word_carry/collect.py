#!/usr/bin/env python3
"""Summarise cells/*.json (carry.py's output) into summary.json and summary.txt. CPU, seconds."""
import collections
import glob
import json
from pathlib import Path

import numpy as np

D = Path(__file__).parent
PASS = {('L30', 'match_wide'): False, ('L31', 'match_wide'): True, ('L30', 'match_both'): False,
        ('L53', 'soft_ceil'): False, ('L52', 'soft_ceil'): True, ('L38', 'soft_ceil'): True, ('L37', 'soft_ceil'): False,
        ('L31', 'match_ceil'): False, ('L32', 'match_ceil'): True, ('L33', 'match_ceil'): True,
        ('L34', 'match_ceil'): True, ('L35', 'match_ceil'): True, ('L36', 'match_ceil'): True,
        ('L52', 'break_rel'): False, ('L51', 'break_rel'): True, ('L39', 'break_rel'): True,
        ('L38', 'break_rel'): False, ('L45', 'break_rel'): True}
out, lines = [], []
lines.append('trial               pass  arm    kind  within (n, mean)   between (n, mean)  diff   paired diff (score q held fixed)')
for f in sorted(glob.glob(str(D / 'cells' / '*.json')), key=lambda p: (int(Path(p).name.split('-')[0][1:]), p)):
    d = json.load(open(f))
    key = (f"L{d['layer']}", d['design'])
    rec = {'trial': f"{key[0]} {key[1]}", 'pass': PASS.get(key), 'units': d['units'], 'counts': d['counts'], 'arms': {}}
    for arm in d['arms']:
        for kind in ('2to1', '3to3'):
            cs = [c for c in d['cells'] if c['arm'] == arm and c['kind'] == kind]
            if not cs:
                continue
            w = [c['auc'] for c in cs if c['type'] == 'within']
            b = [c['auc'] for c in cs if c['type'] == 'between']
            # paired: for each scored question, its within cells' mean minus its between cells' mean
            by_q = collections.defaultdict(lambda: {'within': [], 'between': []})
            for c in cs:
                by_q[c['score'][0] if kind == '2to1' else tuple(c['score'])][c['type']].append(c['auc'])
            pd = [np.mean(v['within']) - np.mean(v['between']) for v in by_q.values() if v['within'] and v['between']]
            r = {'within_n': len(w), 'within': round(float(np.mean(w)), 4), 'between_n': len(b),
                 'between': round(float(np.mean(b)), 4), 'diff': round(float(np.mean(w) - np.mean(b)), 4),
                 'paired_diff': round(float(np.mean(pd)), 4) if pd else None}
            if kind == '2to1':
                sq = collections.defaultdict(list)
                for c in cs:
                    sq[c['score'][0]].append(c['auc'])
                r['by_scored_question'] = {q: round(float(np.mean(v)), 4) for q, v in sq.items()}
            rec['arms'].setdefault(arm, {})[kind] = r
            lines.append(f"{rec['trial']:19s} {str(rec['pass']):5s} {arm:6s} {kind}  {r['within_n']:3d} {r['within']:.3f}"
                         f"           {r['between_n']:3d} {r['between']:.3f}       {r['diff']:+.3f}  "
                         f"{'' if r['paired_diff'] is None else format(r['paired_diff'], '+.3f')}")
    out.append(rec)
lines.append('')
lines.append('per neuron (all doses; chance read off the controls, leave-one-out, scaled to 1,022)')
lines.append('trial               pass  pA,pB<=.05 (chance)  above every ctl  q05  q05 five ctl aside  >5sd  strongest controls (sd)')
for rec in out:
    c, u = rec['counts']['all doses'], rec['units']
    lines.append(f"{rec['trial']:19s} {str(rec['pass']):5s} {c['pA05_and_pB05']:4d} ({c['chance_scaled']:4.1f})        "
                 f"{u['above_every_control']:4d}           {u['q05']:4d}  {u['q05_five_set_aside']:4d}               "
                 f"{u['above_5sd']:4d}  {' '.join(format(x, '.1f') for x in u['top_controls_sd'])}")
lines.append('')
lines.append('2to1 by scored question (mean AUC over every cell scored on it), the gate\'s dose levels together where two')
for rec in out:
    arm = 'both' if 'both' in rec['arms'] else 'lower'
    bq = rec['arms'][arm]['2to1']['by_scored_question']
    lines.append(f"{rec['trial']:19s} {arm:5s} " + ' '.join(f"{q} {v:.3f}" for q, v in bq.items()))
(D / 'summary.json').write_text(json.dumps(out, indent=1))
(D / 'summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
