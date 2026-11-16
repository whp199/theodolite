#!/usr/bin/env python3
"""Step 0 of match_both's trial at layer 30: which questions are asked, read from the questions' check on the home card
(scripts/verify_atlas_writer.py --layer 30 --max-new 48 --batches 56,60,61,64; its record
atlas_data/census/trials/both/floor_match_both.json). CPU only; reads, prints, writes nothing.

match_both asks match_wide's twelve questions (census.DESIGN_QUESTIONS['match_wide'], fixed by match_wide's own step 0
of 16 November 2026) at both of match_ceil's doses, so each question's unpushed answer is written in another batch
shape: 64 for the first nine, 61 for kettle, sleep and lamp (census.plan_batches over a trial's job list; the twelve
unpushed answers come first, then 48 answers a unit). The rule is match_wide's, written here before the check was
taken: a question gives its place to its genre's spare where its unpushed answer, written in the batch its run writes
it in, parts from itself written alone before 32 word-pieces. Both rows the check writes in that shape are read: the
unpushed one (row 0) and a neuron's push at dose 0 (the last row). An answer that ends before 32 word-pieces without
parting holds. Three genres still have a spare (lines, clock, rice); the other three gave theirs to match_wide (self,
green, lamp). If a question of a genre with no spare left parts, or a spare parts too, no trial is run under this
registration. The questions so found must be census.DESIGN_QUESTIONS['match_both'] at the checkout read, in the same
order, or this fails.

    THEODOLITE_TRIAL_REPO=<checkout> python step0.py <floor_match_both.json>
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
CANDIDATES = ['who', 'orange', 'free', 'self', 'green', 'interesting', 'bicycle', 'walk', 'chair', 'kettle', 'sleep', 'lamp']
SPARES = {'free': 'lines', 'interesting': 'lines', 'bicycle': 'clock', 'kettle': 'clock', 'walk': 'rice', 'sleep': 'rice'}
# the batch each question's unpushed answer is written in, from the plan of the trial at layer 30
cells = [(p, s * d) for p in CANDIDATES for d in cz.MATCH_CEIL[LAYER] for s in (-1.0, 1.0)]  # design_at's order
jobs = cz.census_jobs(list(range(1022)), 1024, cells, CANDIDATES)
base_bs = {jobs[i]['prompt']: len(idx) for _, _, idx in cz.plan_batches(jobs, 64, '12 x batch')
           for i in idx if jobs[i]['kind'] == 'baseline'}


def holds(q, b):
    rows = [res[q][k] for k in (f'census {b}, baseline (row 0)', f'census {b}, dose 0 (row {b - 1})')]
    return all(r['same'] or r['shared'] >= FLOOR for r in rows), [r['shared'] for r in rows], res[q]['alone']['n']


final, used, ok = [], set(), True
for q in CANDIDATES:
    b = base_bs[q]
    h, shared, n = holds(q, b)
    if h:
        final.append(q)
        print(f'{q:12s} kept: in a batch of {b} it holds {min(shared)} of {n} word-pieces with itself alone ({shared})')
        continue
    s = SPARES.get(q)
    if s is None or s in used:
        print(f'{q:12s} parts after {min(shared)} word-pieces in a batch of {b} ({shared}) and its genre has no spare left')
        ok = False
        final.append(q)
        continue
    hs, sshared, sn = holds(s, b)
    used.add(s)
    print(f'{q:12s} gives its place: in a batch of {b} it parts after {min(shared)} word-pieces ({shared}); '
          f'its spare {s}: {"holds" if hs else "parts too"} ({sshared} of {sn})')
    ok = ok and hs
    final.append(s)
print('the questions asked:', ', '.join(final))
print('halves:', final[:6], '|', final[6:])
in_code = list(cz.DESIGN_QUESTIONS.get('match_both') or ())
same = final == in_code
print(f"census.DESIGN_QUESTIONS['match_both'] at the checkout: {', '.join(in_code) or 'none'} ({'the same' if same else 'NOT the same'})")
ok = ok and same
print('OK' if ok else 'NOT RUN: a question parts with no spare left, a spare parts too, or the code asks other questions')
sys.exit(0 if ok else 1)
