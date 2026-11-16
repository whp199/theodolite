#!/usr/bin/env python3
"""Step 0 of match_wide's walk: which questions are asked, by the rule in census.py (MORE_PROMPTS' note on match_wide's
questions) read from the questions' check on the home card (scripts/verify_atlas_writer.py, its record
atlas_data/census/trials/wide/floor_match_wide.json). CPU only; reads, prints, writes nothing.

The questions first chosen (CANDIDATES, in the order a run asks them) and a spare a genre (SPARES), all in
census.MORE_PROMPTS before the check was taken. The rule: a question gives its place to its genre's spare where its
unpushed answer, written
in the census batch its run writes it in, parts from itself written alone before 32 word-pieces. That batch is read
from the design's own plan (census.plan_batches over a trial's job list): 64 for the first half's six, 63 for the
second half's, the twelve unpushed answers coming first. Both rows the check writes in that shape are read: the
unpushed one (row 0) and a neuron's push at dose 0 (the last row). A spare that would take a place must hold by the
same rule in the same shape; if it does not, or two questions of one genre fail, no trial is run under this
registration. An answer that ends before 32 word-pieces without parting holds. The questions so found must be
census.DESIGN_QUESTIONS['match_wide'] at the checkout read, in the same order, or this fails.

    THEODOLITE_TRIAL_REPO=<checkout> python step0.py <floor_match_wide.json>
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get('THEODOLITE_TRIAL_REPO') or '/home/eric/bin/ai/theodolite2')
from backend import census as cz  # noqa: E402

FLOOR = 32
fl = json.load(open(sys.argv[1]))
res = fl['results']
CANDIDATES = ['who', 'orange', 'free', 'skills', 'blue', 'interesting', 'bicycle', 'walk', 'chair', 'kettle', 'sleep', 'bridge']
SPARES = {'who': 'self', 'skills': 'self', 'orange': 'green', 'blue': 'green', 'free': 'lines', 'interesting': 'lines',
          'bicycle': 'clock', 'kettle': 'clock', 'walk': 'rice', 'sleep': 'rice', 'chair': 'lamp', 'bridge': 'lamp'}
qs = CANDIDATES
spares = SPARES
# the batch each question's unpushed answer is written in, from the plan of a trial at layer 31 (the same at 30, and
# for a whole layer: checked when this was written)
cells = [(p, s * d) for p in qs for d in cz.MATCH_WIDE[31] for s in (-1.0, 1.0)]  # design_at's order, on the candidates
jobs = cz.census_jobs(list(range(1022)), 1024, cells, qs)
base_bs = {jobs[i]['prompt']: len(idx) for _, _, idx in cz.plan_batches(jobs, 64, cz.DESIGN_PLAN['match_wide'])
           for i in idx if jobs[i]['kind'] == 'baseline'}


def holds(q, b):
    rows = [res[q][k] for k in (f'census {b}, baseline (row 0)', f'census {b}, dose 0 (row {b - 1})')]
    return all(r['same'] or r['shared'] >= FLOOR for r in rows), [r['shared'] for r in rows], res[q]['alone']['n']


final, failed_genres, ok = [], {}, True
for q in qs:
    b = base_bs[q]
    h, shared, n = holds(q, b)
    if h:
        final.append(q)
        print(f'{q:12s} kept: in a batch of {b} it holds {min(shared)} of {n} word-pieces with itself alone ({shared})')
        continue
    s = spares[q]
    hs, sshared, sn = holds(s, b)
    g = s  # a genre is named by its spare
    failed_genres[g] = failed_genres.get(g, 0) + 1
    print(f'{q:12s} gives its place: in a batch of {b} it parts after {min(shared)} word-pieces ({shared}); '
          f'its spare {s}: {"holds" if hs else "parts too"} ({sshared} of {sn})')
    if not hs or failed_genres[g] > 1:
        ok = False
    final.append(s)
print('the questions asked:', ', '.join(final))
print('halves:', final[:6], '|', final[6:])
in_code = list(cz.DESIGN_QUESTIONS['match_wide'])
same = final == in_code
print(f"census.DESIGN_QUESTIONS['match_wide'] at the checkout: {', '.join(in_code)} ({'the same' if same else 'NOT the same'})")
ok = ok and same
print('OK' if ok else 'NOT RUN: a spare parts too, two questions of one genre failed, or the code asks other questions')
sys.exit(0 if ok else 1)
