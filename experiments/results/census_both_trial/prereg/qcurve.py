"""Exploratory, on data already read (16 November 2026, after match_wide's verdict at layer 30): the lead (mass_c) and
the words carrying between the halves (rep_AB, rep_BA) of match_wide's trials at layers 30 and 31, re-read on 1 to 6
questions of each half (six random draws of each count, seeded, averaged; all six once), and on each sign of the push
alone. By census_question_count/subsample.py's own reading (readlib.py of the match_ceil walk). CPU only; writes
qcurve.json beside this file. It decides nothing by itself."""
import itertools, json, random, sys, time
from pathlib import Path
sys.path.insert(0, '/home/eric/bin/ai/theodolite2/experiments/results/census_question_count')
sys.path.insert(0, '/home/eric/bin/ai/theodolite2')
import subsample as S
from backend import census as cz
out = {}
for L in (30, 31):
    run = Path(f'/home/eric/bin/ai/theodolite2/atlas_data/census/trials/wide/L{L}-match_wide-trial.jsonl')
    meta = json.loads(cz.meta_path_of(run).read_text())
    qs = list(meta.get('prompts') or {})
    asked = [q for q in qs if any(p == q for p, _ in meta['design'])]
    HA, HB = asked[:6], asked[6:12]
    rows = S.load(run)
    for r in rows: S.R._words(r)
    rng = random.Random(0)
    res = {}
    for q in (1, 2, 3, 4, 5, 6):
        cells = []
        n = 1 if q == 6 else 6
        for _ in range(n):
            sa = tuple(rng.sample(HA, q)); sb = tuple(rng.sample(HB, q))
            keep = lambda r, s=set(sa) | set(sb): r['prompt'] in s
            cells.append(S.light(rows, keep, sa, sb))
        avg = {k: round(sum(c[k] for c in cells) / len(cells), 4) for k in cells[0]}
        res[q] = avg
        print(L, q, avg, flush=True)
    # sign halves: + only and - only, all twelve questions
    for sg in (1, -1):
        keep = lambda r, sg=sg: r['dose'] * sg > 0 or r['kind'] == 'unpushed'
        print(L, 'sign', sg, S.light(rows, keep, HA, HB), flush=True)
    out[L] = res
(Path(__file__).resolve().parent / 'qcurve.json').write_text(json.dumps(out, indent=1))
