#!/usr/bin/env python3
"""Step 0 of match_things's trial at layer 30, its second round: the place of suitcase. CPU only; reads, prints, writes
nothing.

The first round (step0.py on atlas_data/census/trials/things/floor_match_things.json, 17 November 2026, 18:09Z; its
output in step0.txt) kept eleven of the twelve questions. suitcase, a question of the three-tips genre, parted from
itself written alone after 28 word-pieces in the batch of 61 its run writes it in (and in every other shape), and both
spares of its genre parted there too (garden after 3, interview after 16), so by that round's rule no trial could be
run with those questions. No neuron had been read. This round's rule, written before its check was taken: four more
questions of the genre, fixed in this order before the check asked them (census.MORE_PROMPTS: money, language, winter,
tidy), are read in suitcase's place, a batch of 61, by the same test as the first round's (the unpushed row 0 and a
neuron's push at dose 0 in the last row each hold 32 word-pieces with the answer written alone, or end before parting);
the first that holds takes the place. If none holds, no trial is run under this registration. The eleven other
questions are the first round's, read from its record; the questions so found must be
census.DESIGN_QUESTIONS['match_things'] at the checkout read, in the same order, or this fails.

    THEODOLITE_TRIAL_REPO=<checkout> python step0_round2.py <floor_match_things.json> <floor_match_things_round2.json>
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get('THEODOLITE_TRIAL_REPO') or '/home/eric/bin/ai/theodolite2')
from backend import census as cz  # noqa: E402

FLOOR = 32
LAYER = 30
r1 = json.load(open(sys.argv[1]))['results']
r2 = json.load(open(sys.argv[2]))['results']
CANDIDATES = ['bicycle', 'walk', 'chair', 'kettle', 'sleep', 'lamp', 'clock', 'rice', 'vase', 'lock', 'suitcase', 'blanket']
ROUND2 = ['money', 'language', 'winter', 'tidy']
cells = [(p, s * d) for p in CANDIDATES for d in cz.MATCH_CEIL[LAYER] for s in (-1.0, 1.0)]  # design_at's order
jobs = cz.census_jobs(list(range(1022)), 1024, cells, CANDIDATES)
place_bs = [len(idx) for _, _, idx in cz.plan_batches(jobs, 64, '12 x batch') for i in idx if jobs[i]['kind'] == 'baseline']
assert len(place_bs) == 12, place_bs


def holds(res, q, b):
    rows = [res[q][k] for k in (f'census {b}, baseline (row 0)', f'census {b}, dose 0 (row {b - 1})')]
    return all(r['same'] or r['shared'] >= FLOOR for r in rows), [r['shared'] for r in rows], res[q]['alone']['n']


final, ok = [], True
for q, b in zip(CANDIDATES, place_bs):
    if q != 'suitcase':
        h, shared, n = holds(r1, q, b)
        print(f'{q:10s} {"kept" if h else "PARTS"} (first round): in a batch of {b}, {min(shared)} of {n} word-pieces ({shared})')
        ok = ok and h
        final.append(q)
        continue
    took = None
    for s in ROUND2:
        hs, sshared, sn = holds(r2, s, b)
        print(f'suitcase   its place, second round: {s} in a batch of {b}: {"holds" if hs else "parts, passed over"} '
              f'({sshared} of {sn})')
        if hs:
            took = s
            break
    if took is None:
        print('suitcase   no question of the second round holds in its place')
        ok = False
        final.append(q)
    else:
        final.append(took)
print('the questions asked:', ', '.join(final))
print('halves:', final[:6], '|', final[6:])
in_code = list(cz.DESIGN_QUESTIONS.get('match_things') or ())
same = final == in_code
print(f"census.DESIGN_QUESTIONS['match_things'] at the checkout: {', '.join(in_code) or 'none'} ({'the same' if same else 'NOT the same'})")
ok = ok and same
for q in final:  # reported, gating nothing: every other batch shape a trial writes in
    res = r2 if q in ROUND2 else r1
    for b in (56, 60, 61, 64):
        k = f'census {b}, baseline (row 0)'
        if k in res[q] and not (res[q][k]['same'] or res[q][k]['shared'] >= FLOOR):
            print(f'reported: {q} parts after {res[q][k]["shared"]} word-pieces unpushed in a batch of {b}')
print('OK' if ok else 'NOT RUN: a question parts, no second-round question holds, or the code asks other questions')
sys.exit(0 if ok else 1)
