#!/usr/bin/env python3
"""A test of both_read.py before any answer of match_both existed (16 November 2026): a mock two-dose run made of
match_wide's trial at layer 30, each pushed answer written twice, once at its own dose and once as if at the upper dose
(x 1.1033: +-2.42 to +-2.67), the meta's design made match_both's. Nothing in it is new: the upper dose's answers are
the lower's. It shows the reader runs to its end on 98,220 rows of four doses and twelve questions, and what the
reading gives for answers that are only repeated. G5b fails there by its making (the units file is match_wide's).
CPU only; writes the mock and its verdict into OUT, a scratch folder.

    CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO=<checkout> python mock_read.py OUT
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
W = Path('/home/eric/bin/ai/theodolite2/atlas_data/census/trials/wide')
meta = json.loads((W / 'L30-match_wide-trial.meta.json').read_text())
n = 0
with open(OUT / 'L30-mock-trial.jsonl', 'w') as out:
    for line in open(W / 'L30-match_wide-trial.jsonl'):
        r = json.loads(line)
        out.write(json.dumps(r) + '\n')
        n += 1
        if r['kind'] != 'baseline':
            out.write(json.dumps({**r, 'dose': round(r['dose'] * 1.1033, 2), 'push': r['push'] * 1.1033}) + '\n')
            n += 1
meta['design'] = [[p, s * d] for p in dict.fromkeys(p for p, _ in meta['design']) for d in (2.42, 2.67) for s in (-1.0, 1.0)]
meta['jobs']['n'], meta['design_name'] = n, 'match_both'
(OUT / 'L30-mock-trial.meta.json').write_text(json.dumps(meta))
for ext in ('units.jsonl', 'collapse.json'):
    shutil.copyfile(W / f'L30-match_wide-trial.{ext}', OUT / f'L30-mock-trial.{ext}')
sys.exit(subprocess.call([sys.executable, '-I', str(HERE / 'both_read.py'), str(OUT / 'L30-mock-trial.jsonl'),
                          '--out', str(OUT / 'L30_mock_verdict.json')]))
