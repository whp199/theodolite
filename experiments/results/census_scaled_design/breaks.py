#!/usr/bin/env python3
"""Where text breaks, layer by layer, from every run already taken that says (CPU only; reads the runs, writes
breaks.json here): the two calibration ladders of 28 October 2026 (layers 20, 30, 40, 45, 50, 57; 64 neurons and 64
controls; calib_rel in the stream's median length, calib_ceil in each neuron's write ceiling), the first trial and its
probe (45, 57) and the neighbour trials read so far (38, 39, 51, ...; 1,022 neurons and 1,024 controls at +-1 and +-2 of
the stream).

For each run, kind of unit and dose (both signs pooled): the push as a share of the stream's median length (`rel`),
the push over the unit's write ceiling (`ceilings`; a control's is the ceiling of the neuron it is paired with), and
how much of its distinct 4-grams the answer loses against the same question's unpushed answer (`d4_loss`, the mean;
answers cut to 48 word-pieces, the census's length, so the ladders' 64 and the trials' 48 compare).

Then, per layer and kind, the push at which the loss crosses 0.10 (census.BREAKS_BY, "the break") and 0.04 (what the
+-1 answers lose at layers 39 and 45, where the design passes), read off the ladder by a straight line in log push
between the two doses that bracket it, in both units.

    .venv/bin/python -I experiments/results/census_scaled_design/breaks.py
"""
import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
from backend import census as cz  # noqa: E402

CUT = 48
RUNS = sorted((ROOT / "atlas_data" / "census" / "runs").glob("L*-calib_*.jsonl")) \
    + sorted((ROOT / "atlas_data" / "census" / "trials").glob("L*-break_rel-*.jsonl")) \
    + sorted((ROOT / "atlas_data" / "census" / "trials" / "neighbours").glob("L*-break_rel-trial.jsonl"))


def cross(xs, ys, at):
    """The x at which y first rises through `at`, by a straight line in log x between neighbours; None if it never does."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 < at <= y1:
            t = (at - y0) / (y1 - y0)
            return round(math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0))), 4)
    return None


out = {"cut": CUT, "runs": {}, "layers": {}}
for run in RUNS:
    if not cz.meta_path_of(run).exists():
        continue
    if "trials" in run.parts and not Path(str(run)[: -len(".jsonl")] + ".t1").exists():
        continue  # a trial still being written
    meta = json.loads(cz.meta_path_of(run).read_text())
    layer, design = int(meta["layer"]), meta.get("design_name")
    base, acc = {}, collections.defaultdict(lambda: collections.defaultdict(list))
    rows = list(cz.read_rows(run, tolerate_tail=True))
    for r in rows:
        if r["kind"] == "baseline":
            base[r["prompt"]] = cz.distinct4(r["token_ids"][:CUT])
    for r in rows:
        if r["kind"] == "baseline" or r["prompt"] not in base:
            continue
        key = (r["kind"], round(abs(float(r["dose"])), 4))
        acc[key]["d4"].append(base[r["prompt"]] - cz.distinct4(r["token_ids"][:CUT]))
        acc[key]["rel"].append(abs(float(r["rel_push"])))
        if r.get("ceiling"):
            acc[key]["ceil"].append(abs(float(r["push"])) / float(r["ceiling"]))
    table = []
    for (kind, dose), a in sorted(acc.items()):
        table.append({"kind": kind, "dose": dose, "n": len(a["d4"]), "rel": round(float(np.median(a["rel"])), 4),
                      "ceilings": round(float(np.median(a["ceil"])), 3) if a["ceil"] else None, "d4_loss": round(float(np.mean(a["d4"])), 4)})
    out["runs"][run.name] = {"layer": layer, "design": design, "p50": meta.get("p50"), "table": table}
    for kind in ("neuron", "random"):
        t = [x for x in table if x["kind"] == kind]
        if len(t) < 2:
            continue
        rec = out["layers"].setdefault(str(layer), {}).setdefault(run.name, {})
        rec[kind] = {f"{unit}_at_loss_{int(round(at * 100)):02d}": cross([x[unit] for x in t if x[unit]], [x["d4_loss"] for x in t if x[unit]], at)
                     for unit in ("rel", "ceilings") for at in (0.04, 0.10)}
(HERE / "breaks.json").write_text(json.dumps(out, indent=1) + "\n")
for name, r in out["runs"].items():
    print(f"\n{name} (layer {r['layer']}, {r['design']})")
    for kind in ("neuron", "random"):
        row = [f"{x['dose']:g}: rel {x['rel']:.2f}" + (f" ({x['ceilings']:.1f} c)" if x["ceilings"] else "") + f" loss {x['d4_loss']:+.3f}" for x in r["table"] if x["kind"] == kind]
        print(f"  {kind:7s} " + " | ".join(row))
print("\nwhere the loss crosses 0.04 and 0.10, by layer (rel = share of the stream, c = ceilings):")
for layer in sorted(out["layers"], key=int):
    for name, rec in out["layers"][layer].items():
        for kind, v in rec.items():
            print(f"  L{layer:>2} {name:32s} {kind:7s} 0.04 at rel {v['rel_at_loss_04']} / {v['ceilings_at_loss_04']} c;  0.10 at rel {v['rel_at_loss_10']} / {v['ceilings_at_loss_10']} c")
