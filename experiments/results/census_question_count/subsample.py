#!/usr/bin/env python3
"""How the trial's gate figures grow with the answers a neuron gets: read a finished trial again on subsets of its
questions (1, 2 or 3 of each half) and of its doses (the lower pair, the upper pair, both), with the reading's own
statistics (readlib: mass_c, mass_B, rep_AB, rep_BA, against the controls). CPU only, reads the run, writes --out.
Exploratory: on data already read; it decides nothing by itself."""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

P = Path('/home/eric/bin/ai/theodolite2/experiments/results/census_match_walk/prereg')
sys.path.insert(0, str(P))
import readlib as R  # noqa: E402
from backend import census as cz  # noqa: E402


def load(path):
    rows = []
    for r in cz.read_rows(Path(path), tolerate_tail=True):
        rows.append({'kind': r['kind'], 'neuron': int(r['neuron']), 'prompt': r['prompt'], 'dose': float(r['dose']),
                     'finish': r['finish'], 'text': r['text'], 'token_ids': r['token_ids'], 'lp': r['logprob']})
    return rows


def light(rows, keep, A, B):
    A, B = set(A), set(B)
    kA = lambda r: keep(r) and r['prompt'] in A  # noqa: E731
    kB = lambda r: keep(r) and r['prompt'] in B  # noqa: E731
    cnt = R.counts(rows, keep, 'own')
    m_c = R.mass_from_counts(cnt, 'controls')
    cA, cB = R.counts(rows, kA, 'own'), R.counts(rows, kB, 'own')
    mA, mB = R.mass_from_counts(cA, 'controls'), R.mass_from_counts(cB, 'controls')
    repAB = R.z_of(cB, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
    repBA = R.z_of(cA, {u: [w for z, w in mB[u][1] if z > 0] for u in mB})
    stat = {'mass_c': {u: v[0] for u, v in m_c.items()}, 'mass_A': {u: v[0] for u, v in mA.items()},
            'mass_B': {u: v[0] for u, v in mB.items()}, 'rep_AB': repAB, 'rep_BA': repBA}
    return {k: round(R.auc(*R.split(d))[0], 4) for k, d in stat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    meta = json.loads(cz.meta_path_of(Path(a.run)).read_text())
    qs = list(meta.get('prompts') or {})
    asked = [q for q in qs if any(p == q for p, _ in meta['design'])]
    HA, HB = asked[:3], asked[3:6]
    doses = sorted({float(d) for _, d in meta['design']})
    least = min(abs(d) for d in doses)
    lower = [d for d in doses if abs(abs(d) - least) < 1e-9]
    upper = [d for d in doses if d not in lower]
    t0 = time.time()
    rows = load(a.run)
    for r in rows:
        R._words(r)
    print(f'{Path(a.run).name}: {len(rows)} rows in {time.time() - t0:.0f}s; halves {HA} | {HB}; lower {lower} upper {upper}',
          flush=True)
    arms = {'lower': lower, 'upper': upper, 'both': lower + upper}
    # question subsets: q of each half; for q < 3 the three matched drops (A's i-th with B's i-th, and so on)
    subsets = {3: [(tuple(HA), tuple(HB))]}
    for q in (2, 1):
        combA = list(itertools.combinations(HA, q))
        combB = list(itertools.combinations(HB, q))
        subsets[q] = list(zip(combA, combB))
    out = {'run': Path(a.run).name, 'layer': meta.get('layer'), 'design': meta.get('design_name'), 'half_a': HA,
           'half_b': HB, 'lower': lower, 'upper': upper, 'cells': []}
    for arm, ds in arms.items():
        for q, pairs in subsets.items():
            for sa, sb in pairs:
                t = time.time()
                keep = lambda r, ds=ds, qs_=set(sa) | set(sb): r['prompt'] in qs_ and any(abs(r['dose'] - d) < 1e-9 for d in ds)  # noqa: E731
                fig = light(rows, keep, sa, sb)
                cell = {'arm': arm, 'q': q, 'A': sa, 'B': sb, 'answers_per_half': q * len(ds), **fig}
                out['cells'].append(cell)
                print(f'  {arm:5s} q={q} {"+".join(sa)} | {"+".join(sb)}: lead {fig["mass_c"]:.3f} rep {fig["rep_AB"]:.3f}/'
                      f'{fig["rep_BA"]:.3f} B {fig["mass_B"]:.3f} ({time.time() - t:.0f}s)', flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
