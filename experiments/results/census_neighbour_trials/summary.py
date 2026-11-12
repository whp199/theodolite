#!/usr/bin/env python3
"""The neighbour trials' verdicts gathered: one line a layer, and summary.json beside this file. Reads the verdicts
read.sh wrote (atlas_data/census/trials/neighbours/L<NN>_verdict.json, or another folder given); writes nothing else.

    .venv/bin/python -I experiments/results/census_neighbour_trials/summary.py [--from FOLDER] [--no-write]
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
PARTS = (("N1 whole", "whole"), ("N2 the +-1 answers alone", "pm1"), ("N3 the neurons the controls cover", "covered"))
GATES = (("G1", "mass_c"), ("G2", "mass_c_res"), ("G3", "rep_AB"), ("G3", "rep_BA"), ("G4", "mass_B"))


def figures(part):
    """{statistic: its AUC} over a part's gates."""
    out = {}
    for g in part.get("gates", {}).values():
        for s, x, _ in g["values"]:
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                out[s] = x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=str(ROOT / "atlas_data" / "census" / "trials" / "neighbours"))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.src).glob("L*_verdict.json"), key=lambda p: int(re.match(r"L(\d+)_", p.name).group(1))):
        v = json.loads(p.read_text())
        row = {"layer": v["layer"], "pass": v["pass"]}
        for name, key in PARTS:
            part = v[name]
            row[key] = {"pass": part["pass"], **{s: figures(part).get(s) for _, s in GATES}}
        row["covered"].update({"neurons": v[PARTS[2][0]]["covered"], "of": v[PARTS[2][0]]["neurons"]})
        rep = v.get("reported") or {}
        text = rep.get("text") or {}
        row["reported"] = {"above_every_control": rep.get("above_every_control"), "by_chance": rep.get("by_chance"),
                           "named": rep.get("named"), "max_new_rule": rep.get("max_new_rule"),
                           "d4_loss_at_1": {k: text.get(f"{d} {k}", {}).get("d4_minus_unpushed") for k in ("neuron", "random") for d in ("+1",)},
                           "d4_loss_at_minus_1": {k: text.get(f"-1 {k}", {}).get("d4_minus_unpushed") for k in ("neuron", "random")},
                           "answers_per_s": (rep.get("throughput") or {}).get("answers_per_s")}
        rows.append(row)
    f = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
    print("layer  verdict  | whole: lead  resid  repAB  repBA  newQ | +-1: lead  resid  repAB  repBA  newQ | covered: lead  resid  repAB  repBA  newQ  (share)")
    for r in rows:
        cells = []
        for _, key in PARTS:
            p = r[key]
            cells.append(("ok " if p["pass"] else "NO ") + "  ".join(f(p[s]) for _, s in GATES))
        c = r["covered"]
        print(f"{r['layer']:>5}  {'PASS' if r['pass'] else 'FAIL':<7}  | " + " | ".join(cells) + f"  ({c['neurons']}/{c['of']})")
    print("bars:                    0.600  0.580  0.570  0.570  0.560")
    if not a.no_write:
        (HERE / "summary.json").write_text(json.dumps({"gates": {"mass_c": 0.60, "mass_c_res": 0.58, "rep_AB": 0.57, "rep_BA": 0.57, "mass_B": 0.56},
                                                       "min_share_covered": 0.5, "layers": rows}, indent=1) + "\n")
        print(f"-> {HERE / 'summary.json'}")


if __name__ == "__main__":
    main()
