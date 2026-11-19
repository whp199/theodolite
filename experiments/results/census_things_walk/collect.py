#!/usr/bin/env python3
"""Copy the verdicts of the match_things walk into this folder (atlas_data is not in the repository): each layer's
L<NN>_verdict.json and .txt, its whole reading (.whole.json), and the report-only readings, each dose alone (.lower.json,
.upper.json) and three questions a half (.first.json, .last.json), with their named_words left out to keep the record
small, and each run's meta, which says how it was taken. Then one line a reading, in the order the walk read them, and
the walk's own log and stop files as they were written. census_things_trial/collect.py with the folder, the order and
the log added.

    .venv/bin/python -I experiments/results/census_things_walk/collect.py
"""
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent.parent / "atlas_data" / "census" / "trials" / "things_walk"
ORDER = (53, 29, 54, 28, 55, 27, 56, 26, 57, 25, 58)  # the walk's order (RULES.md)
PARTS = (("N1 whole", "whole"), ("N3 the neurons the controls cover", "covered"))
FIGS = ("lead", "beyond_text", "rep_AB", "rep_BA", "new_questions")
EACH = ("lower", "upper")  # reported, gating nothing: each dose alone
THREE = ("first", "last")  # reported, gating nothing: three questions a half


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
for v in sorted(SRC.glob("L*_verdict.json"), key=lambda p: ORDER.index(int(re.match(r"L(\d+)_", p.name).group(1)))):
    layer = re.match(r"L(\d+)_", v.name).group(1)
    shutil.copyfile(v, HERE / v.name)
    shutil.copyfile(v.with_suffix(".txt"), HERE / v.with_suffix(".txt").name)
    for part in ("whole",) + EACH + THREE:
        p = SRC / f"L{layer}_verdict.{part}.json"
        if p.exists():
            slim(p, HERE / p.name)
    for ext in ("meta.json", "pauses"):
        f = SRC / f"L{layer}-match_things-trial.{ext}"
        if f.exists():
            shutil.copyfile(f, HERE / f.name)
    d = json.loads(v.read_text())
    row = {"layer": d["layer"], "pass": d["pass"], "doses": d.get("doses"), "dose_unit": d.get("dose_unit"),
           "half_a": d.get("half_a"), "half_b": d.get("half_b")}
    for name, key in PARTS:
        row[key] = {"pass": d[name]["pass"], **figures(d[name]["gates"])}
    n3 = d[PARTS[1][0]]
    row["covered"].update({"neurons": n3["covered"], "of": n3["neurons"], "outside_by_covariate": n3.get("outside_by_covariate")})
    rep = d.get("reported") or {}
    row["text"] = rep.get("text")
    row["above_every_control"] = rep.get("above_every_control")
    row["counts"] = rep.get("figures")
    for name in EACH:  # report-only: each dose's answers alone, every question
        t = (rep.get("each_dose") or {}).get(name)
        if t:
            f = t["figures"]
            sub = json.loads((SRC / f"L{layer}_verdict.{name}.json").read_text())["readings"]["48"]["figures"]["counts"]
            row[f"{name} (reported)"] = {"doses": t["doses"], "lead": f["mass_c"], "beyond_text": f["mass_c_res"],
                                         "rep_AB": f["rep_AB"], "rep_BA": f["rep_BA"], "new_questions": f["mass_B"],
                                         "above_every_control": sub["above_every_control"], "named": sub["named"]}
    for name in THREE:  # report-only: the first three of each half, and the last three
        t = (rep.get("three_a_half") or {}).get(name)
        if t:
            f = t["figures"]
            sub = json.loads((SRC / f"L{layer}_verdict.{name}.json").read_text())["readings"]["48"]["figures"]["counts"]
            row[f"{name} (reported)"] = {"half_a": t["half_a"], "half_b": t["half_b"], "lead": f["mass_c"],
                                         "beyond_text": f["mass_c_res"], "rep_AB": f["rep_AB"], "rep_BA": f["rep_BA"],
                                         "new_questions": f["mass_B"], "above_every_control": sub["above_every_control"],
                                         "named": sub["named"]}
    tp = rep.get("throughput") or {}
    row["throughput"] = tp
    rows.append(row)
(HERE / "summary.json").write_text(json.dumps({"bars": {"lead": 0.60, "beyond_text": 0.58, "rep_AB": 0.57, "rep_BA": 0.57, "new_questions": 0.56},
                                               "layers": rows}, indent=1) + "\n")
print("layer  verdict  | whole: lead  resid  repAB  repBA  newQ | covered: lead  resid  repAB  repBA  newQ")
for r in rows:
    cells = [("ok " if r[k]["pass"] else "NO ") + "  ".join(f"{r[k][s]:.3f}" for s in FIGS) for _, k in PARTS]
    print(f"{r['layer']:>5}  {'PASS' if r['pass'] else 'FAIL':<7}  | " + " | ".join(cells) + f"  ({r['covered']['neurons']}/{r['covered']['of']})")
    for name in EACH:
        t = r.get(f"{name} (reported)")
        if t:
            print(f"       {name} dose alone (reported):  "
                  + "  ".join(f"{t[s]:.3f}" for s in FIGS) + f"  ({t['above_every_control']} above every control, {t['named']} named)")
    for name in THREE:
        t = r.get(f"{name} (reported)")
        if t:
            print(f"       three a half, {name:5s} (reported): " + "  ".join(f"{t[s]:.3f}" for s in FIGS)
                  + f"  ({t['above_every_control']} above every control, {t['named']} named)")
print("bars:                     0.600  0.580  0.570  0.570  0.560")
# the walk's own log (its stops, and where each arm ended) and its stop files, as they were written
for name in ("chain.log", "walk.up.stopped_at", "walk.down.stopped_at", "walk.done"):
    if (SRC / name).exists():
        shutil.copyfile(SRC / name, HERE / name)
