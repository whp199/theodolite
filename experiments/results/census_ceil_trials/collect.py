#!/usr/bin/env python3
"""Copy the break_ceil trials' verdicts that are read into this folder (atlas_data is not in the repository): each
layer's L<NN>_verdict.json and .txt, its two readings (.whole.json, .pm1.json) with their named_words left out to keep
the record small, and each run's meta, which says how it was taken. Then one line a layer.

    .venv/bin/python -I experiments/results/census_ceil_trials/collect.py
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent.parent / "atlas_data" / "census" / "trials" / "ceil"
PARTS = (("N1 whole", "whole"), ("N2 the +-1 answers alone", "the lower dose alone"), ("N3 the neurons the controls cover", "covered"))


rows = []
for v in sorted(SRC.glob("L*_verdict.json"), key=lambda p: int(re.match(r"L(\d+)_", p.name).group(1))):
    layer = re.match(r"L(\d+)_", v.name).group(1)
    shutil.copyfile(v, HERE / v.name)
    shutil.copyfile(v.with_suffix(".txt"), HERE / v.with_suffix(".txt").name)
    for part in ("whole", "pm1"):
        p = SRC / f"L{layer}_verdict.{part}.json"
        d = json.loads(p.read_text())
        for rd in (d.get("readings") or {}).values():
            rd.pop("named_words", None)
        (HERE / p.name).write_text(json.dumps(d, indent=1) + "\n")
    meta = SRC / f"L{layer}-break_ceil-trial.meta.json"
    if meta.exists():
        shutil.copyfile(meta, HERE / meta.name)
    d = json.loads(v.read_text())
    row = {"layer": d["layer"], "pass": d["pass"], "doses": d.get("doses"), "dose_unit": d.get("dose_unit")}
    for name, key in PARTS:
        g = {k[:2]: [x for _, x, _ in val["values"]] for k, val in d[name]["gates"].items() if k[:2] in ("G1", "G2", "G3", "G4")}
        row[key] = {"pass": d[name]["pass"], "lead": g["G1"][0], "beyond_text": g["G2"][0], "rep_AB": g["G3"][0],
                    "rep_BA": g["G3"][1], "new_questions": g["G4"][0]}
    row["covered"].update({"neurons": d[PARTS[2][0]]["covered"], "of": d[PARTS[2][0]]["neurons"]})
    rows.append(row)
(HERE / "summary.json").write_text(json.dumps({"bars": {"lead": 0.60, "beyond_text": 0.58, "rep_AB": 0.57, "rep_BA": 0.57, "new_questions": 0.56},
                                               "layers": rows}, indent=1) + "\n")
print("layer  verdict  | whole: lead  resid  repAB  repBA  newQ | lower dose: lead  resid  repAB  repBA  newQ | covered: lead  resid  repAB  repBA  newQ")
for r in rows:
    cells = [("ok " if r[k]["pass"] else "NO ") + "  ".join(f"{r[k][s]:.3f}" for s in ("lead", "beyond_text", "rep_AB", "rep_BA", "new_questions"))
             for _, k in PARTS]
    print(f"{r['layer']:>5}  {'PASS' if r['pass'] else 'FAIL':<7}  | " + " | ".join(cells) + f"  ({r['covered']['neurons']}/{r['covered']['of']})")
print("bars:                    0.600  0.580  0.570  0.570  0.560")
