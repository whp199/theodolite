#!/usr/bin/env python3
"""Do a census trial's words carry from question to question within a half of its questions better than between the
halves? Read again from a finished trial, on the CPU; it writes --out and nothing else.

The trials' gate G3 picks a unit's top five words on one half of the questions and scores them on the other half
(readlib.rep_AB / rep_BA, an AUC of neurons over controls). The halves are not alike: half A asks about the model and
about colours (who, orange, free; at twelve questions also self, green, interesting), half B about objects and
everyday tasks (bicycle, walk, chair; also kettle, sleep, lamp). If a neuron's words carry from question to question
inside a half and not across, its push brings words that depend on the kind of question, and more questions or
another dose would not mend G3. If they carry no better inside a half, the halves are not what stops them.

The same statistic is taken on smaller, matched sets of questions:
- 2to1 (every trial): words picked on two questions of a half, scored on one other question, either of the same half
  ("within") or of the other half ("between"). Every such cell is read.
- 3to3 (twelve-question trials only): picked on three questions, scored on three others. Within: the two halves of a
  half, both ways (40 cells). Between: 40 cells drawn at random (seed 0), 20 each way.
Each is read on each dose level alone (the lower pair of doses, the upper pair, where a design has two).

Also per trial, for the count of neurons that can be read one at a time (its chance level read off the controls, not
assumed): how many neurons have p <= 0.05 against the controls on each half's mass alone, and how many controls would
(leave-one-out, scaled to the neurons' count); from the run's units file, how far its strongest controls stand out,
and the q <= 0.05 count with the five strongest controls set aside (as the whole-layer write-ups read it).
Exploratory: it decides nothing by itself."""
import argparse
import collections
import itertools
import json
import random
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
                     'text': r['text']})
    return rows


def per_question(rows, doses):
    """{question: {unit: Counter of word -> answers holding it}}, the question's own unpushed words left out."""
    base = R.baselines(rows)
    units = sorted({(r['kind'], r['neuron']) for r in rows if r['kind'] != 'baseline'})
    out = {q: {u: collections.Counter() for u in units} for q in base}
    for r in rows:
        if r['kind'] == 'baseline' or not any(abs(r['dose'] - d) < 1e-9 for d in doses):
            continue
        out[r['prompt']][(r['kind'], r['neuron'])].update(R._words(r) - R._words(base[r['prompt']]))
    return out, units


def summed(pq, units, qs):
    out = {}
    for u in units:
        c = collections.Counter()
        for q in qs:
            c.update(pq[q][u])
        out[u] = c
    return out


def carry(pq, units, pick, score):
    """AUC of neurons over controls: summed z on `score` of each unit's top-5 words picked on `pick`."""
    cP, cS = summed(pq, units, pick), summed(pq, units, score)
    mP = R.mass_from_counts(cP, 'controls')
    rep = R.z_of(cS, {u: [w for z, w in mP[u][1] if z > 0] for u in units})
    return R.auc(*R.split(rep))[0]


def cells(HA, HB):
    out = []
    for H, O in ((HA, HB), (HB, HA)):
        for pick in itertools.combinations(H, 2):
            for q in H + O:
                if q in pick:
                    continue
                out.append(('2to1', 'within' if q in H else 'between', pick, (q,)))
    if len(HA) == 6:
        for H in (HA, HB):
            for P3 in itertools.combinations(H, 3):
                S3 = tuple(q for q in H if q not in P3)
                out.append(('3to3', 'within', P3, S3))
        rng = random.Random(0)
        for H, O in ((HA, HB), (HB, HA)):
            seen = set()
            while len(seen) < 20:
                seen.add((tuple(sorted(rng.sample(H, 3), key=H.index)), tuple(sorted(rng.sample(O, 3), key=O.index))))
            out += [('3to3', 'between', p, s) for p, s in sorted(seen)]
    return out


def chance_counts(rows, HA, HB, doses):
    """Neurons with p <= 0.05 on each half's mass alone, against the same count among the controls (leave-one-out)."""
    pq, units = per_question(rows, doses)
    mA = R.mass_from_counts(summed(pq, units, HA), 'controls')
    mB = R.mass_from_counts(summed(pq, units, HB), 'controls')
    pA, pB = R.emp_p({u: v[0] for u, v in mA.items()}), R.emp_p({u: v[0] for u, v in mB.items()})
    N = sum(1 for u in units if u[0] == 'neuron')
    C = sum(1 for u in units if u[0] == 'random')
    both = lambda kind: sum(1 for u in units if u[0] == kind and pA[u] <= 0.05 and pB[u] <= 0.05)  # noqa: E731
    return {'neurons': N, 'controls': C, 'pA05_and_pB05': both('neuron'),
            'controls_pA05_and_pB05': both('random'), 'chance_scaled': round(both('random') * N / C, 1)}


def units_file(path):
    U = [json.loads(s) for s in open(path)]
    neu = np.array([u['mass'] for u in U if u['kind'] == 'neuron'])
    ctl = sorted((u for u in U if u['kind'] == 'random'), key=lambda u: -u['mass'])
    c = np.array([u['mass'] for u in ctl])
    med, sd = float(np.median(c)), float(np.std(c))

    def q05(c):
        s = np.sort(c)
        p = np.sort((1 + len(s) - np.searchsorted(s, neu, side='left')) / (len(s) + 1))
        return int((np.minimum.accumulate((p * len(p) / np.arange(1, len(p) + 1))[::-1]) <= 0.05).sum())
    return {'top_controls_sd': [round((u['mass'] - med) / sd, 2) for u in ctl[:5]],
            'top_controls_words': [' '.join(w[0] for w in u['words'][:6]) for u in ctl[:5]],
            'above_5sd': int((neu > med + 5 * sd).sum()), 'above_every_control': int((neu > c.max()).sum()),
            'q05': q05(c), 'q05_five_set_aside': q05(c[5:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads(cz.meta_path_of(run).read_text())
    qs = list(meta['prompts'])
    n = len(qs) // 2
    HA, HB = qs[:n], qs[n:]
    doses = sorted({float(d) for _, d in meta['design']})
    least = min(abs(d) for d in doses)
    arms = {'lower': [d for d in doses if abs(abs(d) - least) < 1e-9]}
    if len(arms['lower']) < len(doses):
        arms['upper'] = [d for d in doses if d not in arms['lower']]
        arms['both'] = doses
    t0 = time.time()
    rows = load(run)
    for r in rows:
        R._words(r)
    print(f'{run.name}: {len(rows)} rows in {time.time() - t0:.0f}s; halves {HA} | {HB}; arms {arms}', flush=True)
    out = {'run': run.name, 'layer': meta['layer'], 'design': meta['design_name'], 'half_a': HA, 'half_b': HB,
           'arms': arms, 'cells': [], 'counts': {}}
    out['units'] = units_file(str(run).replace('.jsonl', '.units.jsonl'))
    out['counts']['all doses'] = chance_counts(rows, HA, HB, doses)
    print('  units', out['units'], '\n  counts', out['counts'], flush=True)
    todo = cells(HA, HB)
    for arm, ds in arms.items():
        t = time.time()
        pq, units = per_question(rows, ds)
        for kind, typ, pick, score in todo:
            out['cells'].append({'arm': arm, 'kind': kind, 'type': typ, 'pick': pick, 'score': score,
                                 'auc': round(carry(pq, units, pick, score), 4)})
        for kind in ('2to1', '3to3'):
            for typ in ('within', 'between'):
                v = [c['auc'] for c in out['cells'] if c['arm'] == arm and c['kind'] == kind and c['type'] == typ]
                if v:
                    print(f'  {arm:5s} {kind} {typ:7s} n={len(v):3d} mean {np.mean(v):.3f} sd {np.std(v):.3f}', flush=True)
        print(f'  {arm} read in {time.time() - t:.0f}s', flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
