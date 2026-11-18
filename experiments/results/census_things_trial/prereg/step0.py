#!/usr/bin/env python3
"""Step 0 of match_things's trial at layer 30: which questions are asked, read from the questions' check on the home card
(scripts/verify_atlas_writer.py --layer 30 --max-new 48 --batches 56,60,61,64; its record
atlas_data/census/trials/things/floor_match_things.json). CPU only; reads, prints, writes nothing.

match_things asks twelve questions of three genres at both of match_ceil's doses: how a thing works (bicycle, kettle,
clock, lock), three tips (walk, sleep, rice, suitcase) and an object (chair, lamp, vase, blanket). Each question's
unpushed answer is written in the batch census.plan_batches gives it over the trial's job list (the twelve unpushed
answers come first, then 48 answers a unit): 64 for the first nine, 61 for the last three. The rule is the one
match_wide's and match_both's step 0 kept, written here before the check was taken: a question gives its place to a
spare of its genre where its unpushed answer, written in the batch its run writes it in, parts from itself written
alone before 32 word-pieces. Both rows the check writes in that shape are read: the unpushed one (row 0) and a
neuron's push at dose 0 (the last row). An answer that ends before 32 word-pieces without parting holds. Each genre has
two spares, tried in order (windmill then camera; garden then interview; boot then mug); a spare that parts in the
place's batch is passed over and used up. If a question parts and no spare of its genre is left that holds, no trial
is run under this registration. The questions so found must be census.DESIGN_QUESTIONS['match_things'] at the checkout
read, in the same order, or this fails.

    THEODOLITE_TRIAL_REPO=<checkout> python step0.py <floor_match_things.json>
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get('THEODOLITE_TRIAL_REPO') or '/home/eric/bin/ai/theodolite2')
from backend import census as cz  # noqa: E402

FLOOR = 32
LAYER = 30
fl = json.load(open(sys.argv[1]))
res = fl['results']
CANDIDATES = ['bicycle', 'walk', 'chair', 'kettle', 'sleep', 'lamp', 'clock', 'rice', 'vase', 'lock', 'suitcase', 'blanket']
GENRE = {q: g for g, qs in (('works', ('bicycle', 'kettle', 'clock', 'lock')), ('tips', ('walk', 'sleep', 'rice', 'suitcase')),
                            ('object', ('chair', 'lamp', 'vase', 'blanket'))) for q in qs}
SPARES = {'works': ['windmill', 'camera'], 'tips': ['garden', 'interview'], 'object': ['boot', 'mug']}
# the batch each question's unpushed answer is written in, from the plan of the trial at layer 30
cells = [(p, s * d) for p in CANDIDATES for d in cz.MATCH_CEIL[LAYER] for s in (-1.0, 1.0)]  # design_at's order
jobs = cz.census_jobs(list(range(1022)), 1024, cells, CANDIDATES)
place_bs = [len(idx) for _, _, idx in cz.plan_batches(jobs, 64, '12 x batch') for i in idx if jobs[i]['kind'] == 'baseline']
assert len(place_bs) == 12, place_bs


def holds(q, b):
    rows = [res[q][k] for k in (f'census {b}, baseline (row 0)', f'census {b}, dose 0 (row {b - 1})')]
    return all(r['same'] or r['shared'] >= FLOOR for r in rows), [r['shared'] for r in rows], res[q]['alone']['n']


final, ok = [], True
for q, b in zip(CANDIDATES, place_bs):
    h, shared, n = holds(q, b)
    if h:
        final.append(q)
        print(f'{q:10s} kept: in a batch of {b} it holds {min(shared)} of {n} word-pieces with itself alone ({shared})')
        continue
    took = None
    while SPARES[GENRE[q]]:
        s = SPARES[GENRE[q]].pop(0)
        hs, sshared, sn = holds(s, b)
        print(f'{q:10s} parts after {min(shared)} word-pieces in a batch of {b} ({shared}); its genre\'s spare {s}: '
              f'{"holds" if hs else "parts too, passed over"} ({sshared} of {sn})')
        if hs:
            took = s
            break
    if took is None:
        print(f'{q:10s} has no spare of its genre left that holds')
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
# reported, gating nothing: every other batch shape a trial writes in, for each question asked
for q in final:
    for b in (56, 60, 61, 64):
        k = f'census {b}, baseline (row 0)'
        if k in res[q] and not (res[q][k]['same'] or res[q][k]['shared'] >= FLOOR):
            print(f'reported: {q} parts after {res[q][k]["shared"]} word-pieces unpushed in a batch of {b}')
print('OK' if ok else 'NOT RUN: a question parts with no spare of its genre left that holds, or the code asks other questions')
sys.exit(0 if ok else 1)
