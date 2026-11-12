#!/usr/bin/env python3
"""What the two steer-lens layers built on 29 October 2026 at 0.8 of the stream say about that dose (CPU only; reads
atlas_data/offline/steer_lens_L45 and _L5, the vocabulary, and layer 45's census ranking; writes two_layers.json here).

1. How hard the push lands: KL(pushed || unpushed) of the next word-piece, per neuron, its median and 90th percentile.
2. How alike the neurons' readings are: the share of neurons whose first word-piece is one of the layer's ten
   commonest, and the word-pieces the most neurons list.
3. At layer 45, where the census's doses are calibrated: for how many neurons a steer-lens word-piece (the twelve of
   either sign, three letters or more, case and marks dropped) is one of the neuron's five census words, or begins
   one, or is begun by one; and the same with the neurons shuffled (five shuffles, seed 0).

    .venv/bin/python -I experiments/results/atlas_steer_dose/read_two_layers.py
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

vocab = json.loads((ROOT / "atlas_data" / "index" / "vocab.json").read_text())["vocab"]
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalpha())  # noqa: E731
out = {"dose": 0.8, "unit": "the stream's median length at the layer", "layers": {}}
for L in (45, 5):
    z = np.load(ROOT / "atlas_data" / "offline" / f"steer_lens_L{L}" / f"L{L}.npz", allow_pickle=False)
    rec = {}
    for sign in ("up", "down"):
        a, kl = z[sign], z[f"{sign}_kl"].astype(float)
        first = collections.Counter(a[:, 0].tolist())
        every = collections.Counter(a.ravel().tolist())
        rec[sign] = {"kl_p50": round(float(np.median(kl)), 3), "kl_p90": round(float(np.percentile(kl, 90)), 3),
                     "first_piece_among_ten_commonest": round(sum(v for _, v in first.most_common(10)) / len(a), 4),
                     "distinct_pieces": len(every),
                     "commonest": [[vocab[int(t)].strip(), round(v / len(a), 3)] for t, v in every.most_common(6)]}
    out["layers"][str(L)] = rec
t = cz.load_table(ROOT / "atlas_data" / "census" / "runs" / "L45-break_rel.jsonl")
z = np.load(ROOT / "atlas_data" / "offline" / "steer_lens_L45" / "L45.npz", allow_pickle=False)
words = {int(t.neuron[i]): (t.shown(i, 5), float(t.null_pct[i]), float(t.q[i]) if not np.isnan(t.q[i]) else 1.0)
         for i in range(len(t)) if t.is_neuron[i]}
steer = {n: {w for w in (norm(vocab[int(x)]) for x in list(z["up"][n]) + list(z["down"][n])) if len(w) >= 3} for n in words}
hit = lambda ws, ts: any(w == s or w.startswith(s) or s.startswith(w) for w in ws for s in ts)  # noqa: E731
rng = np.random.default_rng(0)
groups = {"above every control": [n for n, (w, p, q) in words.items() if p == 1.0 and w],
          "q <= 0.05": [n for n, (w, p, q) in words.items() if q <= 0.05 and w],
          "q > 0.05": [n for n, (w, p, q) in words.items() if q > 0.05 and w]}
out["layer_45_against_its_census"] = {}
for name, ns in groups.items():
    got = float(np.mean([hit(words[n][0], steer[n]) for n in ns]))
    base = float(np.mean([np.mean([hit(words[n][0], steer[m]) for n, m in zip(ns, rng.permutation(ns))]) for _ in range(5)]))
    out["layer_45_against_its_census"][name] = {"neurons": len(ns), "matched": round(got, 4), "neurons_shuffled": round(base, 4)}
(HERE / "two_layers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False))
