#!/usr/bin/env python3
"""The steer lens as built, every layer of it, read on the CPU (writes lens.json here; nothing is run on the model).

The lens was built on the home card between 13 November 2026 01:31Z and 23:43Z, each layer at the dose its pilot chose
(pilot.json here). `against_census.py` set it against the whole censuses, layers 40 to 50, as the build's last step.
This reads three things more, the first two at all sixty layers:

1. Where the push landed. The pilot chose each layer's dose from 128 of its neurons, to move the next word-piece as
   far as layer 45's 0.8 does (a median KL of 0.355 nats over that sample). Here: the median over the whole layer.
2. Whether the neurons read apart. At 0.8 of the stream layer 5 gave nearly one reading whichever neuron was pushed
   (two_layers.json: four word-pieces each listed by 40 to 49 in 100 of its neurons). Here, for each sign: the share of
   neurons that list the layer's most listed word-piece, and the share whose first word-piece is one of the layer's
   ten commonest first word-pieces. Layer 45 at 0.8, where the lens agrees with the census: 0.056 and 0.058.
3. The lens against a trial's census, at the layers where no whole census is taken but a trial of 1,022 neurons
   passed and its doses are counted calibrated (backend/census.py DOSES_CALIBRATED). The measure is
   against_census.py's: for how many neurons a steer-lens word-piece (the twelve of either sign, three letters or
   more, case and marks dropped) is one of the neuron's five census words, or begins one, or is begun by one; and the
   same with the neurons shuffled (five shuffles, seed 0). A trial has 1,024 controls where a whole layer has 2,048,
   and its q never reaches 0.05, so the groups here are by the controls alone: above every control, above 95 in 100
   of them, and the rest. The whole layers are read into the same groups so the two can be set side by side.

    .venv/bin/python -I experiments/results/atlas_steer_dose/read_lens.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
from backend import census as cz  # noqa: E402

INDEX = ROOT / "atlas_data" / "index"
TRIALS = ROOT / "atlas_data" / "census" / "trials"
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalpha())  # noqa: E731
hit = lambda ws, ts: any(w == s or w.startswith(s) or s.startswith(w) for w in ws for s in ts)  # noqa: E731


def apart(a, vocab):
    """How alike a layer's readings are on one sign (`a`: neurons x 12 word-pieces)."""
    first = collections.Counter(a[:, 0].tolist())
    every = collections.Counter(a.ravel().tolist())
    return {"most_listed_piece": round(every.most_common(1)[0][1] / len(a), 4),
            "first_piece_among_ten_commonest": round(sum(v for _, v in first.most_common(10)) / len(a), 4),
            "distinct_pieces": len(every),
            "commonest": [[vocab[int(t)].strip(), round(v / len(a), 3)] for t, v in every.most_common(4)]}


def against(words, up, down, vocab, rng):
    """`words`: neuron -> (its five census words, the share of controls it stands above). The measure, by group."""
    steer = {n: {w for w in (norm(vocab[int(x)]) for x in list(up[n]) + list(down[n])) if len(w) >= 3} for n in words}
    groups = {"above every control": [n for n, (w, p) in words.items() if p == 1.0 and w],
              "above 95 in 100 controls": [n for n, (w, p) in words.items() if p >= 0.95 and w],
              "the rest": [n for n, (w, p) in words.items() if p < 0.95 and w]}
    out = {}
    for name, ns in groups.items():
        if len(ns) < 2:
            continue
        got = float(np.mean([hit(words[n][0], steer[n]) for n in ns]))
        base = float(np.mean([np.mean([hit(words[n][0], steer[m]) for n, m in zip(ns, rng.permutation(ns))]) for _ in range(5)]))
        out[name] = {"neurons": len(ns), "matched": round(got, 4), "neurons_shuffled": round(base, 4)}
    return out


def main():
    vocab = json.loads((INDEX / "vocab.json").read_text())["vocab"]
    z = np.load(INDEX / "steer_lens.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    pilot = json.loads((HERE / "pilot.json").read_text())
    target = float(pilot["reference"]["kl"])
    up, down, dose = z["up"], z["down"], np.asarray(z["dose"], dtype=float)
    kl = (z["up_kl"].astype(float) + z["down_kl"].astype(float)) / 2
    L = up.shape[0]
    out = {"what": __doc__.split("\n\n")[0], "built_at": meta.get("built_at"), "weights": (meta.get("weights") or {}).get("id"),
           "dose_rule": meta.get("dose_rule"), "target_kl": target, "layers": {}}
    for l in range(L):
        out["layers"][str(l)] = {
            "dose": round(float(dose[l]), 4), "dose_in_ceilings": pilot["layers"][str(l)].get("dose_in_ceilings"),
            "kl_p50": round(float(np.median(kl[l])), 3), "kl_p10": round(float(np.percentile(kl[l], 10)), 3),
            "kl_p90": round(float(np.percentile(kl[l], 90)), 3), "kl_p50_over_target": round(float(np.median(kl[l])) / target, 3),
            "up": apart(up[l], vocab), "down": apart(down[l], vocab)}

    rng = np.random.default_rng(0)
    whole = cz.layer_runs()
    for layer in sorted(whole):
        if layer >= L or not cz.units_path_of(whole[layer]).exists():
            continue
        t = cz.load_table(whole[layer])
        words = {int(t.neuron[i]): (t.shown(i, 5), float(t.null_pct[i])) for i in range(len(t)) if t.is_neuron[i]}
        out["layers"][str(layer)]["census"] = {"kind": "whole", "run": whole[layer].name, "neurons": len(words),
                                               "groups": against(words, up[layer], down[layer], vocab, rng)}
    for units in sorted(TRIALS.glob("**/L*-trial.units.jsonl")):
        run = Path(str(units)[: -len(".units.jsonl")] + ".jsonl")
        m = json.loads(cz.meta_path_of(run).read_text())
        layer, design = int(m["layer"]), m["design_name"]
        if layer in whole or not cz.doses_calibrated(design, layer, m.get("max_new")):
            continue
        rows = [json.loads(line) for line in open(units) if line.strip()]
        words = {int(u["neuron"]): (cz.shown_words(u, 5), float(u["null_pct"])) for u in rows if u.get("kind") == "neuron"}
        out["layers"][str(layer)]["census"] = {"kind": "trial", "run": str(run.relative_to(TRIALS)), "design": design,
                                               "neurons": len(words), "controls": sum(u.get("kind") != "neuron" for u in rows),
                                               "groups": against(words, up[layer], down[layer], vocab, rng)}

    (HERE / "lens.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"target: a median KL of {target:.3f} nats (layer 45 at 0.8, over the pilot's 128 neurons)")
    print("layer  dose  ceilings  KL p50 (x target)  most-listed piece up/down  first-in-ten up/down   census: above every control, matched (shuffled) | above 95 in 100")
    for l in range(L):
        r = out["layers"][str(l)]
        c = r.get("census")
        line = (f"{l:>5} {r['dose']:>6.3f} {r['dose_in_ceilings'] or float('nan'):>8.2f}  {r['kl_p50']:>6.3f} ({r['kl_p50_over_target']:.2f})"
                f"      {r['up']['most_listed_piece']:.3f} / {r['down']['most_listed_piece']:.3f}"
                f"            {r['up']['first_piece_among_ten_commonest']:.3f} / {r['down']['first_piece_among_ten_commonest']:.3f}")
        if c:
            a, b = c["groups"].get("above every control"), c["groups"].get("above 95 in 100 controls")
            line += f"    {c['kind']:<5}"
            line += f" {a['neurons']:>5}: {a['matched']:.3f} ({a['neurons_shuffled']:.3f})" if a else "     none"
            line += f" | {b['neurons']:>5}: {b['matched']:.3f} ({b['neurons_shuffled']:.3f})" if b else ""
        print(line)
    print(f"-> {HERE / 'lens.json'}")


if __name__ == "__main__":
    main()
