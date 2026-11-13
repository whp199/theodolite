#!/usr/bin/env python3
"""The steer lens's dose pilot, kept here (atlas_data is not in the repository) and set beside the census's doses.

The pilot (atlas_build.py steer_dose, taken on the home card on 13 November 2026) pushes 128 neurons of every layer at
nine doses, each a share of the stream's median length at that layer, and records how far the next word-piece moves
(the median KL from the unpushed one, in nats). The lens's dose for a layer is where that median meets layer 45's at
0.8 of the stream. This script writes pilot.json:

- the pilot as measured (its ladder, each layer's three curves, each layer's dose);
- each layer's dose counted in write ceilings (census_scaled_design/ceilings.json: how many median write ceilings one
  stream length is, a layer at a time);
- how far the census's doses move the next word-piece at each layer, read off the pilot's curve between its rungs
  (a straight line in log dose and log KL: an estimate, coarse where the curve is steep);
- every dose a census trial has read alone, with the lead it gave and the movement the pilot puts on it.

Nothing here gates anything. The last table was made after its first fourteen readings were known; the rows said to
be read after the span was drawn came later and are a test of it.

    .venv/bin/python -I experiments/results/atlas_steer_dose/pilot_in_ceilings.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
PILOT = ROOT / "atlas_data" / "index" / "steer_dose.json"
CEILINGS = HERE.parent / "census_scaled_design" / "ceilings.json"

# Each dose read alone by a census trial at 48 word-pieces: layer, the push in the trial's own unit, the lead (how
# often a neuron's mass beats a control's; 0.5 is chance), and where the figure is from.
READ_ALONE = (
    (38, "ceiling", 1.6, 0.654, "soft_ceil lower dose, registered gate (census_soft_trials)"),
    (38, "ceiling", 2.3, 0.655, "soft_ceil upper dose, reported (census_soft_trials)"),
    (38, "ceiling", 2.3, 0.682, "break_ceil lower dose, registered gate (census_ceil_trials)"),
    (38, "ceiling", 4.6, 0.506, "break_ceil upper dose, looked at afterwards (census_ceil_trials/by_dose.json)"),
    (38, "stream", 1.0, 0.496, "break_rel lower dose, registered gate (census_neighbour_trials)"),
    (38, "stream", 2.0, 0.506, "break_rel upper dose, looked at afterwards (by_dose.json)"),
    (39, "stream", 1.0, 0.690, "break_rel lower dose, registered gate"),
    (39, "stream", 2.0, 0.595, "break_rel upper dose, looked at afterwards (by_dose.json)"),
    (45, "stream", 1.0, 0.706, "break_rel lower dose, the first trial's review"),
    (51, "stream", 1.0, 0.723, "break_rel lower dose, registered gate"),
    (52, "stream", 1.0, 0.603, "break_rel lower dose, registered gate"),
    (52, "stream", 2.0, 0.797, "break_rel upper dose, looked at afterwards (by_dose.json)"),
    (52, "ceiling", 2.3, 0.790, "break_ceil lower dose, registered gate"),
    (52, "ceiling", 4.6, 0.485, "break_ceil upper dose, looked at afterwards (by_dose.json)"),
    # read after the span below was drawn on the fourteen rows above: the walk downward (census_soft_walk)
    (37, "ceiling", 1.6, 0.633, "soft_ceil lower dose, registered gate, read after the span was drawn (census_soft_walk)"),
    (37, "ceiling", 2.3, 0.657, "soft_ceil upper dose, reported, read after the span was drawn (census_soft_walk)"),
    (52, "ceiling", 1.6, 0.633, "soft_ceil lower dose, registered gate, read after the span was drawn (census_soft_trials)"),
    (52, "ceiling", 2.3, 0.783, "soft_ceil upper dose, reported, a third draw of neurons, read after the span was drawn (census_soft_trials)"),
    # match_ceil at layer 37 (census_match_trials). The census's fine pilot puts these two at 1.0 and 1.4 nats; the kl
    # column here is this coarser pilot's line
    (37, "ceiling", 2.41, 0.647, "match_ceil lower dose, registered gate; 1.0 nats by the fine pilot (census_match_trials)"),
    (37, "ceiling", 2.55, 0.616, "match_ceil upper dose, reported; 1.4 nats by the fine pilot (census_match_trials)"),
    # the match_ceil walk downward (census_match_walk), a layer at a time; 1.0 and 1.4 nats by the fine pilot
    (36, "ceiling", 2.27, 0.615, "match_ceil lower dose, registered gate; 1.0 nats by the fine pilot (census_match_walk)"),
    (36, "ceiling", 2.46, 0.601, "match_ceil upper dose, reported; 1.4 nats by the fine pilot (census_match_walk)"),
    (35, "ceiling", 2.27, 0.667, "match_ceil lower dose, registered gate; 1.0 nats by the fine pilot (census_match_walk)"),
    (35, "ceiling", 2.41, 0.645, "match_ceil upper dose, reported; 1.4 nats by the fine pilot (census_match_walk)"),
)
CENSUS_CEILINGS = (1.6, 2.3, 4.6)   # soft_ceil's two doses and break_ceil's upper one
CENSUS_STREAM = (1.0, 2.0)          # break_rel's


def kl_at(ladder, kl, dose):
    """The curve's KL at a dose in the stream's median length: a line between rungs in log dose and log KL."""
    xs = [math.log(x) for x in ladder]
    ys = [math.log(max(k, 1e-9)) for k in kl]
    x = math.log(dose)
    if x <= xs[0]:
        return kl[0]
    if x >= xs[-1]:
        return kl[-1]
    i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))


def main():
    pilot = json.loads(PILOT.read_text())
    ceil = json.loads(CEILINGS.read_text())["layers"]
    ladder = pilot["ladder"]
    layers = {}
    for l in sorted(pilot["layers"], key=int):
        per_stream = ceil[l]["ceilings_in_one_stream"]
        dose = pilot["layers"][l]["dose"]
        m = pilot["measured"][l]
        layers[l] = {
            "dose": dose, "note": pilot["layers"][l].get("note"), "ceilings_in_one_stream": per_stream,
            "dose_in_ceilings": round(dose * per_stream, 3),
            "kl": m["kl"], "kl_p10": m["kl_p10"], "kl_p90": m["kl_p90"],
            "kl_at_ceilings": {f"{c:g}": round(kl_at(ladder, m["kl"], c / per_stream), 4) for c in CENSUS_CEILINGS},
            "kl_at_stream": {f"{s:g}": round(kl_at(ladder, m["kl"], s), 4) for s in CENSUS_STREAM},
        }
    read = []
    for layer, unit, dose, lead, src in READ_ALONE:
        per_stream = ceil[str(layer)]["ceilings_in_one_stream"]
        stream = dose / per_stream if unit == "ceiling" else dose
        read.append({"layer": layer, "dose": dose, "unit": unit, "in_ceilings": round(stream * per_stream, 2),
                     "of_the_stream": round(stream, 3), "kl": round(kl_at(ladder, pilot["measured"][str(layer)]["kl"], stream), 3),
                     "lead": lead, "from": src})
    read.sort(key=lambda r: r["kl"])
    out = {"what": __doc__.split("\n\n")[0], "written_at": pilot.get("written_at"), "model": pilot.get("model"),
           "reference": pilot["reference"], "ladder": ladder, "neurons": pilot.get("neurons"), "seed": pilot.get("seed"),
           "missing": pilot.get("missing"), "layers": layers, "read_alone": read}
    (HERE / "pilot.json").write_text(json.dumps(out, indent=1) + "\n")

    ref = pilot["reference"]
    print(f"the reference: layer {ref['layer']} at {ref['dose']} of the stream moves the next word-piece by a median {ref['kl']:.3f} nats")
    print("layer  dose, of the stream  in ceilings | nats at 1.6  2.3  4.6 ceilings | at 1  2 of the stream")
    for l, r in layers.items():
        k, s = r["kl_at_ceilings"], r["kl_at_stream"]
        print(f"{int(l):>5}  {r['dose']:>8.4f}  {r['dose_in_ceilings']:>14.2f}   | {k['1.6']:>9.3g} {k['2.3']:>6.3g} {k['4.6']:>6.3g}"
              f"         | {s['1']:>6.3g} {s['2']:>6.3g}" + (f"   {r['note']}" if r["note"] else ""))
    print("\nevery dose a trial has read alone, by how far the pilot says it moves the next word-piece")
    print("layer  push            in ceilings   nats    lead")
    for r in read:
        push = f"{r['dose']:g} {'ceilings' if r['unit'] == 'ceiling' else 'of the stream'}"
        print(f"{r['layer']:>5}  {push:<17} {r['in_ceilings']:>6.2f}   {r['kl']:>7.3g}   {r['lead']:.3f}   {r['from']}")


if __name__ == "__main__":
    main()
