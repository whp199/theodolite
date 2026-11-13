#!/usr/bin/env python3
"""Copy the verdicts of the match_ceil walk downward that are read into this folder (atlas_data is not in the repository): each
layer's L<NN>_verdict.json and .txt, its two readings (.whole.json, .pm1.json) and the report-only reading of the
upper dose alone (L<NN>_upper.json, .txt) with their named_words left out to keep the record small, and each run's
meta, which says how it was taken. Then one line a layer.

    .venv/bin/python -I experiments/results/census_match_walk/collect.py
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent.parent / "atlas_data" / "census" / "trials" / "match_walk"
# the reader is the break_ceil trials' own, byte for byte, so its second gate keeps that reader's name for it
PARTS = (("N1 whole", "whole"), ("N2 the +-1 answers alone", "the lower dose alone"), ("N3 the neurons the controls cover", "covered"))
FIGS = ("lead", "beyond_text", "rep_AB", "rep_BA", "new_questions")


def figures(gates):
    g = {k[:2]: [x for _, x, _ in val["values"]] for k, val in gates.items() if k[:2] in ("G1", "G2", "G3", "G4")}
    return {"lead": g["G1"][0], "beyond_text": g["G2"][0], "rep_AB": g["G3"][0], "rep_BA": g["G3"][1], "new_questions": g["G4"][0]}


def slim(src, dst):
    d = json.loads(src.read_text())
    for rd in (d.get("readings") or {}).values():
        rd.pop("named_words", None)
    dst.write_text(json.dumps(d, indent=1) + "\n")
    return d


rows = []
for v in sorted(SRC.glob("L*_verdict.json"), key=lambda p: int(re.match(r"L(\d+)_", p.name).group(1))):
    layer = re.match(r"L(\d+)_", v.name).group(1)
    shutil.copyfile(v, HERE / v.name)
    shutil.copyfile(v.with_suffix(".txt"), HERE / v.with_suffix(".txt").name)
    for part in ("whole", "pm1"):
        p = SRC / f"L{layer}_verdict.{part}.json"
        slim(p, HERE / p.name)
    meta = SRC / f"L{layer}-match_ceil-trial.meta.json"
    if meta.exists():
        shutil.copyfile(meta, HERE / meta.name)
    d = json.loads(v.read_text())
    row = {"layer": d["layer"], "pass": d["pass"], "doses": d.get("doses"), "lower_doses": d.get("lower_doses"), "dose_unit": d.get("dose_unit")}
    for name, key in PARTS:
        row[key] = {"pass": d[name]["pass"], **figures(d[name]["gates"])}
    n3 = d[PARTS[2][0]]
    row["covered"].update({"neurons": n3["covered"], "of": n3["neurons"], "outside_by_covariate": n3.get("outside_by_covariate")})
    row["text"] = (d.get("reported") or {}).get("text")
    row["above_every_control"] = (d.get("reported") or {}).get("above_every_control")
    up = SRC / f"L{layer}_upper.json"
    if up.exists():  # report-only: the upper dose's answers read alone; gates nothing
        u = slim(up, HERE / up.name)
        if up.with_suffix(".txt").exists():
            shutil.copyfile(up.with_suffix(".txt"), HERE / up.with_suffix(".txt").name)
        row["the upper dose alone (reported)"] = {"doses": u.get("doses"), **figures(u["readings"]["48"]["gates"])}
    rows.append(row)
(HERE / "summary.json").write_text(json.dumps({"bars": {"lead": 0.60, "beyond_text": 0.58, "rep_AB": 0.57, "rep_BA": 0.57, "new_questions": 0.56},
                                               "layers": rows}, indent=1) + "\n")
print("layer  verdict  | whole: lead  resid  repAB  repBA  newQ | lower dose: lead  resid  repAB  repBA  newQ | covered: lead  resid  repAB  repBA  newQ")
for r in rows:
    cells = [("ok " if r[k]["pass"] else "NO ") + "  ".join(f"{r[k][s]:.3f}" for s in FIGS) for _, k in PARTS]
    print(f"{r['layer']:>5}  {'PASS' if r['pass'] else 'FAIL':<7}  | " + " | ".join(cells) + f"  ({r['covered']['neurons']}/{r['covered']['of']})")
    u = r.get("the upper dose alone (reported)")
    if u:
        print("       upper dose alone, reported: " + "  ".join(f"{u[s]:.3f}" for s in FIGS))
print("bars:                    0.600  0.580  0.570  0.570  0.560")
