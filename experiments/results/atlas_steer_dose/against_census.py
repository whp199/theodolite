#!/usr/bin/env python3
"""The steer lens set against the census, layer by layer, where both exist (CPU only). For each layer with a ranked
census and a built steer-lens layer: for how many neurons a steer-lens word-piece (the twelve of either sign, three
letters or more, case and marks dropped) is one of the neuron's five census words, or begins one, or is begun by one;
the same with the neurons shuffled (five shuffles, seed 0); and how hard the layer's push lands (median KL) at its
dose. Layer 45 at 0.8 of the stream, read on 12 November 2026 from the layer built on 29 October: 98.3% of the 1,580
neurons above every control, 2.1% shuffled.

    .venv/bin/python -I experiments/results/atlas_steer_dose/against_census.py                    # the index's steer_lens.d
    .venv/bin/python -I experiments/results/atlas_steer_dose/against_census.py --steer DIR --out FILE
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
from backend import census as cz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steer", default=str(ROOT / "atlas_data" / "index" / "steer_lens.d"), help="a folder of L<NN>.npz steer-lens layers")
ap.add_argument("--out", default=str(HERE / "against_census.json"))
a = ap.parse_args()
vocab = json.loads((ROOT / "atlas_data" / "index" / "vocab.json").read_text())["vocab"]
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalpha())  # noqa: E731
hit = lambda ws, ts: any(w == s or w.startswith(s) or s.startswith(w) for w in ws for s in ts)  # noqa: E731
rng = np.random.default_rng(0)
out = {"steer": a.steer, "layers": {}}
runs = cz.layer_runs()
for shard in sorted(Path(a.steer).glob("L*.npz"), key=lambda p: int(re.match(r"L(\d+)", p.name).group(1))):
    layer = int(re.match(r"L(\d+)", shard.name).group(1))
    if layer not in runs or not cz.units_path_of(runs[layer]).exists():
        continue
    z = np.load(shard, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    t = cz.load_table(runs[layer])
    words = {int(t.neuron[i]): (t.shown(i, 5), float(t.null_pct[i]), float(t.q[i]) if not np.isnan(t.q[i]) else 1.0)
             for i in range(len(t)) if t.is_neuron[i]}
    steer = {n: {w for w in (norm(vocab[int(x)]) for x in list(z["up"][n]) + list(z["down"][n])) if len(w) >= 3} for n in words}
    groups = {"above every control": [n for n, (w, p, q) in words.items() if p == 1.0 and w],
              "q <= 0.05": [n for n, (w, p, q) in words.items() if q <= 0.05 and w],
              "q > 0.05": [n for n, (w, p, q) in words.items() if q > 0.05 and w]}
    rec = {"dose": meta.get("dose"), "kl_p50": round(float(np.median((z["up_kl"].astype(float) + z["down_kl"].astype(float)) / 2)), 3), "groups": {}}
    for name, ns in groups.items():
        if not ns:
            continue
        got = float(np.mean([hit(words[n][0], steer[n]) for n in ns]))
        base = float(np.mean([np.mean([hit(words[n][0], steer[m]) for n, m in zip(ns, rng.permutation(ns))]) for _ in range(5)]))
        rec["groups"][name] = {"neurons": len(ns), "matched": round(got, 4), "neurons_shuffled": round(base, 4)}
    out["layers"][str(layer)] = rec
    g = rec["groups"].get("above every control", {})
    print(f"layer {layer}: dose {rec['dose']}, median KL {rec['kl_p50']}; above every control {g.get('neurons')}: matched {g.get('matched')}, shuffled {g.get('neurons_shuffled')}")
Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
print(f"-> {a.out}")
