#!/usr/bin/env python3
"""Read a match_wide trial (match_ceil's lower dose alone, +-d ceilings, on twelve questions read as two halves of six)
against the gates fixed for it on 16 November 2026 before any answer of the design was written. CPU only; reads the
run, writes only --out and the readings beside it. It is ceil_read.py of the match_ceil walk with two changes: the
halves are the run's twelve questions split six and six (its meta's order: the first six, the reference's genres;
the last six, break_rel's added genres), and with one dose there is no gate on a lower dose's answers alone. The
reader under it is the first trial's of 29 October (trial_read.py and readlib.py, copied here unchanged: their sha256
are the ones the first trial's SHA256SUMS holds), its gates layer 45's as they were (GATES['L45']: 0.60, 0.58, 0.57
and 0.57, 0.56).

A trial passes when both of these pass, each read at 48 word-pieces:

  N1  the whole reading. Every answer, the one dose both ways: G1 to G4, G5 (the sanity checks) and G5b (the code
      ranks as read) of trial_read.py --gates L45, on the halves of six.
  N3  the neurons the controls cover: G1 to G4 on the neurons every one of whose six text-survival covariates
      (readlib.COVS) lies inside the controls' own range, least to greatest. A trial where fewer than half the neurons
      are covered fails N3 whatever they read. (Named N3, as in the earlier registrations; there is no N2.)

Reported and gating nothing: the same answers cut to 32 word-pieces and the length rule; the share covered; how much of
its distinct 4-grams the text loses, neurons and controls; how many neurons beat every control; and two readings on
three questions a half, each by trial_read.py: break_rel's six (who, orange, free | bicycle, walk, chair), as the walk
with match_ceil read them, and the six this design adds (the others).

    CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO=<the checkout the trial ran from> .venv/bin/python -I wide_read.py \
        RUN.jsonl --out L31_verdict.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import readlib as R  # noqa: E402
import trial_read as TR  # noqa: E402
from backend import census as cz  # noqa: E402

GATES = TR.GATES['L45']
MIN_COVERED = 0.5  # N3: at least this share of the neurons read must lie inside the controls' covariate range
CUT = 48
OLD_SIX = (('who', 'orange', 'free'), ('bicycle', 'walk', 'chair'))


def judge(fig):
    """G1 to G4 on a reading's figures: {gate: {values, pass}}."""
    out = {}
    for name, bars in GATES.items():
        vals = [(s, fig[s]['auc'], bar) for s, bar in bars]
        out[name] = {'values': vals, 'pass': all(v >= bar for _, v, bar in vals)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    run, out_path = Path(a.run), Path(a.out)
    stem = str(out_path)[:-len('.json')] if str(out_path).endswith('.json') else str(out_path)
    meta = json.loads(cz.meta_path_of(run).read_text())
    doses = tuple(sorted({float(d) for _, d in meta['design']}))
    if len(doses) != 2 or abs(doses[0] + doses[1]) > 1e-9:
        raise SystemExit(f'{run.name}: a design of one dose each way is read here, not {doses}')
    asked = [q for q in list(meta.get('prompts') or {}) if any(p == q for p, _ in meta['design'])]
    if len(asked) != 12:
        raise SystemExit(f'{run.name}: twelve questions are read here, not {len(asked)}')
    A, B = asked[:6], asked[6:]
    as_arg = lambda ds: '--doses=' + ','.join(f'{d:g}' for d in ds)  # noqa: E731
    halves = lambda a_, b_: ['--half-a', ','.join(a_), '--half-b', ','.join(b_)]  # noqa: E731

    # N1: the whole reading (48 pieces, then the same answers cut to 32)
    whole_path = stem + '.whole.json'
    TR.main([str(run), '--gates', 'L45', '--cuts', f'{CUT},32', as_arg(doses), *halves(A, B), '--out', whole_path])
    whole = json.loads(Path(whole_path).read_text())
    w = whole['readings'][str(CUT)]
    n1 = {'pass': bool(w['pass']), 'gates': {g: v for g, v in w['gates'].items()}}

    # N3: the neurons whose text survival the controls cover
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TR.TOK)
    keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in doses)  # noqa: E731
    rows = TR.load(run, CUT, tok)
    cov = R.covariates(rows, keep)
    ctl = [u for u in cov if u[0] == 'random']
    neu = [u for u in cov if u[0] == 'neuron']
    lo = {k: min(cov[u][k] for u in ctl) for k in R.COVS}
    hi = {k: max(cov[u][k] for u in ctl) for k in R.COVS}
    outside = lambda u: any(cov[u][k] < lo[k] or cov[u][k] > hi[k] for k in R.COVS)  # noqa: E731
    covered = {u[1] for u in neu if not outside(u)}
    share = len(covered) / len(neu) if neu else 0.0
    fig3 = R.read(rows, keep, A, B, units_only=covered, null_splits=50)[0] if covered else None
    g3 = judge(fig3) if fig3 else {}
    n3 = {'pass': bool(fig3) and share >= MIN_COVERED and all(v['pass'] for v in g3.values()), 'gates': g3,
          'covered': len(covered), 'neurons': len(neu), 'share_covered': round(share, 4), 'min_share': MIN_COVERED,
          'outside_by_covariate': {k: int(sum(cov[u][k] < lo[k] or cov[u][k] > hi[k] for u in neu)) for k in R.COVS}}

    # reported, gating nothing: three questions a half, break_rel's six and the six added
    three = {}
    new_a = [q for q in A if q not in OLD_SIX[0]]
    new_b = [q for q in B if q not in OLD_SIX[1]]
    for name, (ha, hb) in (('six', OLD_SIX), ('added', (new_a, new_b))):
        if len(ha) == 3 and len(hb) == 3 and set(ha) <= set(A) and set(hb) <= set(B):
            path = f'{stem}.{name}.json'
            TR.main([str(run), '--gates', 'L45', '--cuts', str(CUT), as_arg(doses), *halves(ha, hb), '--out', path])
            rd = json.loads(Path(path).read_text())['readings'][str(CUT)]
            three[name] = {'half_a': list(ha), 'half_b': list(hb), 'pass_G1_G4': all(v['pass'] for g, v in rd['gates'].items() if g[:2] in ('G1', 'G2', 'G3', 'G4')),
                           'figures': {k: rd['figures'][k]['auc'] for k in ('mass_c', 'mass_c_res', 'rep_AB', 'rep_BA', 'mass_B')}}
    m = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, keep, 'own'), 'controls').items()}
    top_ctl = max(v for u, v in m.items() if u[0] == 'random')
    reported = {'above_every_control': int(sum(1 for u in neu if m[u] > top_ctl)),
                'by_chance': round(len(neu) / (len(ctl) + 1), 2), 'text': w['text'], 'figures': w['figures'].get('counts'),
                'named': len(w.get('named') or []), 'max_new_rule': whole.get('max_new_rule'),
                'three_a_half': three, 'throughput': whole.get('writing') or whole.get('throughput')}
    out = {'run': run.name, 'layer': meta.get('layer'), 'design': meta.get('design_name'), 'max_new': meta.get('max_new'),
           'batch': meta.get('batch'), 'seed': meta.get('seed'), 'cut': CUT, 'half_a': A, 'half_b': B,
           'dose_unit': meta.get('dose_unit'), 'doses': list(doses),
           'N1 whole': n1, 'N3 the neurons the controls cover': n3,
           'pass': bool(n1['pass'] and n3['pass']), 'reported': reported}
    cz.write_json_atomic(out_path, out)
    print(f'=== {run.name} (layer {meta.get("layer")}, counted from 0): {"PASS" if out["pass"] else "FAIL"}')
    for name in ('N1 whole', 'N3 the neurons the controls cover'):
        part = out[name]
        extra = f' ({part["covered"]} of {part["neurons"]} covered, {part["share_covered"]:.1%}; at least {MIN_COVERED:.0%})' if name.startswith('N3') else ''
        print(f'{"pass" if part["pass"] else "FAIL"}  {name}{extra}')
        for g, v in part['gates'].items():
            vals = ', '.join(f'{s} {x}' + (f' (bar {b})' if not isinstance(b, bool) else '') for s, x, b in v['values'])
            print(f'    {"pass" if v["pass"] else "FAIL"}  {g}: {vals}')
    for name, t in three.items():
        print(f'reported, three questions a half ({name}: {"+".join(t["half_a"])} | {"+".join(t["half_b"])}): '
              + ', '.join(f'{k} {v}' for k, v in t['figures'].items()))
    print(f'reported: {json.dumps({k: v for k, v in reported.items() if k != "three_a_half"})}')
    print(f'-> {out_path}')
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
