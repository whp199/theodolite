#!/usr/bin/env python3
"""Where layer 38's lead lies, dose by dose. Gates nothing: of these figures only the rho between the two designs was
named in the registration (RULES.md, "Reported"); the rest were looked at after the verdict was read, to say what the
verdict means, and are not a second reading of it. CPU only; reads the trials' runs in atlas_data, writes by_dose.json.

  - each dose pair read alone by the registered reader (trial_read.py, layer 45's gates, 48 word-pieces): layer 38
    under break_ceil and under break_rel, and layer 39 under break_rel beside them, each with its push in ceilings;
  - Spearman's rho, over the 1,022 neurons both trials asked, of a neuron's mass under break_ceil with its mass under
    break_rel at layer 38;
  - how firmly the whole lead sits on its bar: a bootstrap of the lead over neurons and controls (4,000 draws).

    CUDA_VISIBLE_DEVICES= nice -n 15 .venv/bin/python -I experiments/results/census_ceil_trials/by_dose.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
TRIALS = ROOT / "atlas_data" / "census" / "trials"
os.environ.setdefault("THEODOLITE_TRIAL_REPO", str(ROOT / ".claude" / "worktrees" / "ceil-trials"))
sys.path.insert(0, str(HERE / "prereg"))
import trial_read as TR  # noqa: E402

CEILINGS = json.loads((HERE.parent / "census_scaled_design" / "ceilings.json").read_text())["layers"]
RUNS = (("L38 break_ceil", TRIALS / "ceil" / "L38-break_ceil-trial.jsonl", 38, "ceiling", (2.3, 4.6)),
        ("L38 break_rel", TRIALS / "neighbours" / "L38-break_rel-trial.jsonl", 38, "stream", (1.0, 2.0)),
        ("L39 break_rel", TRIALS / "neighbours" / "L39-break_rel-trial.jsonl", 39, "stream", (1.0, 2.0)))


def alone(run, dose, tmp):
    out = tmp / "r.json"
    TR.main([str(run), "--gates", "L45", "--cuts", "48", f"--doses={-dose:g},{dose:g}", "--out", str(out)])
    r = json.loads(out.read_text())["readings"]["48"]
    g = {k[:2]: [x for _, x, _ in v["values"]] for k, v in r["gates"].items() if k[:2] in ("G1", "G2", "G3", "G4")}
    return {"lead": g["G1"][0], "beyond_text": g["G2"][0], "rep_AB": g["G3"][0], "rep_BA": g["G3"][1], "new_questions": g["G4"][0],
            "text_lost": {k: v["d4_minus_unpushed"] for k, v in r["text"].items()}}


def masses(units):
    neu, ctl = {}, []
    for line in open(units):
        u = json.loads(line)
        if u["kind"] == "neuron":
            neu[int(u["neuron"])] = float(u["mass"])
        else:
            ctl.append(float(u["mass"]))
    return neu, np.array(ctl)


def auc(x, y):
    r = rankdata(np.concatenate([x, y]))
    return float((r[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def main():
    doses = []
    with tempfile.TemporaryDirectory() as t:
        for name, run, layer, unit, pair in RUNS:
            per = float(CEILINGS[str(layer)]["ceilings_in_one_stream"])
            for d in pair:
                fig = alone(run, d, Path(t))
                doses.append({"trial": name, "layer": layer, "dose": d, "unit": unit,
                              "ceilings": round(d if unit == "ceiling" else d * per, 2), **fig})
    a, ca = masses(TRIALS / "ceil" / "L38-break_ceil-trial.units.jsonl")
    b, _ = masses(TRIALS / "neighbours" / "L38-break_rel-trial.units.jsonl")
    ids = sorted(set(a) & set(b))
    rho = spearmanr([a[i] for i in ids], [b[i] for i in ids])
    x = np.array([a[i] for i in sorted(a)])
    rng = np.random.default_rng(0)
    bs = np.array([auc(rng.choice(x, len(x)), rng.choice(ca, len(ca))) for _ in range(4000)])
    out = {"what": "report only; gates nothing (by_dose.py)", "gates": "layer 45's: 0.60, 0.58, 0.57 and 0.57, 0.56",
           "each_dose_pair_alone": doses,
           "between_designs_L38": {"neurons": len(ids), "spearman_rho": round(float(rho.statistic), 4), "p": float(rho.pvalue)},
           "whole_lead_L38_break_ceil": {"lead": round(auc(x, ca), 4), "bar": 0.60, "bootstrap_draws": len(bs),
                                         "p2_5": round(float(np.percentile(bs, 2.5)), 4), "p97_5": round(float(np.percentile(bs, 97.5)), 4),
                                         "share_of_draws_at_or_over_the_bar": round(float(np.mean(bs >= 0.60)), 3)}}
    (HERE / "by_dose.json").write_text(json.dumps(out, indent=1) + "\n")
    print("trial            dose  ceilings | lead   resid  repAB  repBA  newQ")
    for d in doses:
        print(f"{d['trial']:<15} {d['dose']:>5g}  {d['ceilings']:>7.2f}  | " + "  ".join(
            f"{d[k]:.3f}" for k in ("lead", "beyond_text", "rep_AB", "rep_BA", "new_questions")))
    print("bars:                           | 0.600  0.580  0.570  0.570  0.560")
    print(json.dumps({k: out[k] for k in ("between_designs_L38", "whole_lead_L38_break_ceil")}))


if __name__ == "__main__":
    main()
