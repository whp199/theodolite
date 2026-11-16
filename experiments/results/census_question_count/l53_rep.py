import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/eric/bin/ai/theodolite2/experiments/results/census_match_walk/prereg')
import readlib as R
from backend import census as cz
sys.path.insert(0, '/home/eric/bin/ai/theodolite2/experiments/results/census_question_count')
from subsample import load
run = sys.argv[1]; arm = float(sys.argv[2])
rows = load(run)
A, B = {'who','orange','free'}, {'bicycle','walk','chair'}
keep = lambda r: abs(abs(r['dose']) - arm) < 1e-9
kA = lambda r: keep(r) and r['prompt'] in A
kB = lambda r: keep(r) and r['prompt'] in B
cA, cB = R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
mA = R.mass_from_counts(cA, 'controls')
rep = R.z_of(cB, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
# d4 and repetition by kind
d4 = collections.defaultdict(list)
for r in rows:
    if r['kind'] != 'baseline' and keep(r):
        d4[r['kind']].append(cz.distinct4(r['token_ids']))
print({k: round(float(np.mean(v)), 3) for k, v in d4.items()})
for kind in ('neuron', 'random'):
    us = [u for u in rep if u[0] == kind]
    v = np.array([rep[u] for u in us])
    print(kind, 'rep mean %.2f median %.2f p90 %.2f' % (v.mean(), np.median(v), np.percentile(v, 90)))
    top = sorted(us, key=lambda u: -rep[u])[:6]
    for u in top:
        print('   ', u, round(rep[u], 1), [w for z, w in mA[u][1]])
# per-unit: rep vs that unit's own mean d4 at this dose
ud4 = collections.defaultdict(list)
for r in rows:
    if r['kind'] != 'baseline' and keep(r):
        ud4[(r['kind'], r['neuron'])].append(cz.distinct4(r['token_ids']))
from scipy import stats
for kind in ('neuron', 'random'):
    us = [u for u in rep if u[0] == kind]
    print(kind, 'spearman(rep, mean d4) = %.3f' % stats.spearmanr([rep[u] for u in us], [np.mean(ud4[u]) for u in us])[0])
