#!/usr/bin/env python3
"""From subsample.py's cells: each figure's AUC by questions a half (mean over the matched subsets), as d' (sqrt 2 x
probit), the growth exponent b (d' ~ q^b) from q = 2 to 3 and from 1 to 3, and the figure projected to 6 questions a
half (twelve questions) as d'(3) x 2^b. Exploratory."""
import json
import math
import sys
from pathlib import Path

from scipy.stats import norm

BARS = {'mass_c': 0.60, 'rep_AB': 0.57, 'rep_BA': 0.57, 'mass_B': 0.56}
d = lambda a: math.sqrt(2) * norm.ppf(a)  # noqa: E731
A = lambda x: float(norm.cdf(x / math.sqrt(2)))  # noqa: E731

out_rows = []
for f in sys.argv[1:]:
    j = json.loads(Path(f).read_text())
    name = Path(f).stem
    print(f'\n== {name} (layer {j["layer"]}, {j["design"]}; lower {j["lower"][-1]:g}, upper {j["upper"][-1]:g})')
    print(f'{"arm":6s}{"figure":8s}  q=1    q=2    q=3  | b(2-3) b(1-3) | proj q=6 (low..high)  bar')
    for arm in ('lower', 'upper', 'both'):
        for k, bar in BARS.items():
            by = {}
            for c in j['cells']:
                if c['arm'] == arm:
                    by.setdefault(c['q'], []).append(c[k])
            m = {q: sum(v) / len(v) for q, v in by.items()}
            dq = {q: d(x) for q, x in m.items()}
            b23 = math.log(dq[3] / dq[2]) / math.log(1.5) if dq[2] > 0 and dq[3] > 0 else float('nan')
            b13 = math.log(dq[3] / dq[1]) / math.log(3) if dq[1] > 0 and dq[3] > 0 else float('nan')
            bs = [b for b in (b23, b13) if b == b]
            # the growth is clipped to [0.5, 1.0]: no slower than averaging noise, no faster than linear in q
            bs = [min(max(b, 0.5), 1.0) for b in bs] or [0.5]
            lo, hi = A(dq[3] * 2 ** min(bs)), A(dq[3] * 2 ** max(bs))
            flag = 'PASS' if lo >= bar else ('maybe' if hi >= bar else 'fail')
            print(f'{arm:6s}{k:8s} {m[1]:.3f}  {m[2]:.3f}  {m[3]:.3f} | {b23:5.2f}  {b13:5.2f} | {lo:.3f}..{hi:.3f} {flag:5s} {bar}')
            out_rows.append({'run': name, 'layer': j['layer'], 'arm': arm, 'fig': k, 'q1': m[1], 'q2': m[2], 'q3': m[3],
                             'b23': b23, 'b13': b13, 'proj_lo': lo, 'proj_hi': hi, 'bar': bar})
Path('projections.json').write_text(json.dumps(out_rows, indent=1))
