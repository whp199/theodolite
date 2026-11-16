#!/usr/bin/env python3
"""Read a break_ceil trial (break_rel with its doses counted in write ceilings, +-2.3 and +-4.6) against the gates the
neighbour trials are read by, fixed for this design on 12 November 2026 before any trial of it was taken. CPU only;
reads the run, writes only --out and the two readings beside it. It is neighbour_read.py of that day with one change:
the doses are the run's own, read from its meta, where that reader had break_rel's -2, -1, +1 and +2 written in. The
reader under it is the first trial's of 29 October (trial_read.py and readlib.py, copied here unchanged: their sha256
are the ones the first trial's SHA256SUMS holds), its gates layer 45's as they were (GATES['L45']: 0.60, 0.58, 0.57
and 0.57, 0.56).

A trial passes when all three of these pass, each read at 48 word-pieces:

  N1  the whole reading. Every answer, the four doses: G1 to G4, G5 (the sanity checks) and G5b (the code ranks as
      read) of trial_read.py --gates L45.
  N2  the lower dose's answers alone (+-2.3 ceilings for break_ceil; +-1 of the stream for break_rel): G1 to G4 on
      them, the upper dose's answers left out. (G5 is read and must pass; G5b compares with the units file, which
      ranks all four doses, so it is not read here.) The review of the first trial found layer 45's lead in these
      answers (0.706 alone) and layer 57's absent from them (0.52).
  N3  the neurons the controls cover: G1 to G4 on the neurons every one of whose six text-survival covariates
      (readlib.COVS, over the four doses) lies inside the controls' own range, least to greatest. The residual of G2 is
      a fit made on the controls; outside their range it is an extrapolation. At layer 45, 893 of 1,022 neurons were
      covered and cleared every gate; at layer 57, 354 of 507, and they failed every one. A trial where fewer than
      half the neurons are covered fails N3 whatever they read: too few to stand for the layer.

Reported and gating nothing: the same answers cut to 32 word-pieces and the length rule; the share covered; how much
of its distinct 4-grams the text loses at each dose, neurons and controls; how many neurons beat every control.

    CUDA_VISIBLE_DEVICES= THEODOLITE_TRIAL_REPO=<the checkout the trial ran from> .venv/bin/python -I ceil_read.py \
        RUN.jsonl --out L38_verdict.json

--fresh: the neurons the gates are read on (the first trial's own 1,022, to read that trial this way; a neighbour
trial reads every neuron it asked).
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
    ap.add_argument('--fresh', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    run, out_path = Path(a.run), Path(a.out)
    stem = str(out_path)[:-len('.json')] if str(out_path).endswith('.json') else str(out_path)
    fresh_args = ['--fresh', a.fresh] if a.fresh else []
    # the run's own doses: every one, and the lower pair (the two of least size)
    meta0 = json.loads(cz.meta_path_of(run).read_text())
    doses = tuple(sorted({float(d) for _, d in meta0['design']}))
    least = min(abs(d) for d in doses)
    lower = tuple(d for d in doses if abs(abs(d) - least) < 1e-9)
    if len(doses) != 4 or len(lower) != 2:
        raise SystemExit(f'{run.name}: a design of two doses each way is read here, not {doses}')
    as_arg = lambda ds: '--doses=' + ','.join(f'{d:g}' for d in ds)  # noqa: E731

    # N1: the whole reading, as the first trial was read (48 pieces, then the same answers cut to 32)
    whole_path = stem + '.whole.json'
    TR.main([str(run), '--gates', 'L45', '--cuts', f'{CUT},32', as_arg(doses), '--out', whole_path, *fresh_args])
    whole = json.loads(Path(whole_path).read_text())
    w = whole['readings'][str(CUT)]
    n1 = {'pass': bool(w['pass']), 'gates': {g: v for g, v in w['gates'].items()}}

    # N2: the lower dose's answers alone (G5b left out: the units file ranks all four doses)
    pm1_path = stem + '.pm1.json'
    TR.main([str(run), '--gates', 'L45', '--cuts', str(CUT), as_arg(lower), '--out', pm1_path, *fresh_args])
    pm1 = json.loads(Path(pm1_path).read_text())['readings'][str(CUT)]
    g2 = {g: v for g, v in pm1['gates'].items() if not g.startswith('G5b')}
    n2 = {'pass': all(v['pass'] for v in g2.values()), 'gates': g2}

    # N3: the neurons whose text survival the controls cover, all four doses
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TR.TOK)
    meta = json.loads(cz.meta_path_of(run).read_text())
    asked = [q for q in list(meta.get('prompts') or {}) if any(p == q for p, _ in meta['design'])]
    A, B = asked[:3], asked[3:6]
    keep = lambda r: r['prompt'] in set(A) | set(B) and any(abs(r['dose'] - d) < 1e-9 for d in doses)  # noqa: E731
    rows = TR.load(run, CUT, tok)
    cov = R.covariates(rows, keep)
    fresh = TR.ids_of(run, a.fresh)
    ctl = [u for u in cov if u[0] == 'random']
    neu = [u for u in cov if u[0] == 'neuron' and (fresh is None or u[1] in fresh)]
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

    # reported, gating nothing
    m = {u: v[0] for u, v in R.mass_from_counts(R.counts(rows, keep, 'own'), 'controls').items()}
    top_ctl = max(v for u, v in m.items() if u[0] == 'random')
    reported = {'above_every_control': int(sum(1 for u in neu if m[u] > top_ctl)),
                'by_chance': round(len(neu) / (len(ctl) + 1), 2), 'text': w['text'], 'figures': w['figures'].get('counts'),
                'named': len(w.get('named') or []), 'max_new_rule': whole.get('max_new_rule'),
                'pm1_mass_c': pm1['figures']['mass_c']['auc'], 'throughput': whole.get('writing') or whole.get('throughput')}
    out = {'run': run.name, 'layer': meta.get('layer'), 'design': meta.get('design_name'), 'max_new': meta.get('max_new'),
           'batch': meta.get('batch'), 'seed': meta.get('seed'), 'cut': CUT, 'half_a': A, 'half_b': B,
           'dose_unit': meta.get('dose_unit'), 'doses': list(doses), 'lower_doses': list(lower),
           'N1 whole': n1, 'N2 the +-1 answers alone': n2, 'N3 the neurons the controls cover': n3,
           'pass': bool(n1['pass'] and n2['pass'] and n3['pass']), 'reported': reported}
    cz.write_json_atomic(out_path, out)
    print(f'=== {run.name} (layer {meta.get("layer")}, counted from 0): {"PASS" if out["pass"] else "FAIL"}')
    for name in ('N1 whole', 'N2 the +-1 answers alone', 'N3 the neurons the controls cover'):
        part = out[name]
        extra = f' ({part["covered"]} of {part["neurons"]} covered, {part["share_covered"]:.1%}; at least {MIN_COVERED:.0%})' if name.startswith('N3') else ''
        shown = name if not name.startswith('N2') else "N2 the lower dose's answers alone (" + ", ".join(f"{d:+g}" for d in lower) + ")"
        print(f'{"pass" if part["pass"] else "FAIL"}  {shown}{extra}')
        for g, v in part['gates'].items():
            vals = ', '.join(f'{s} {x}' + (f' (bar {b})' if not isinstance(b, bool) else '') for s, x, b in v['values'])
            print(f'    {"pass" if v["pass"] else "FAIL"}  {g}: {vals}')
    print(f'reported: {json.dumps(reported)}')
    print(f'-> {out_path}')
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
