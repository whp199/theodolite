#!/usr/bin/env python3
"""Each layer's write ceiling against its stream, for all sixty layers (CPU only; reads the checkpoint's tensors a
layer at a time and the stats index; writes ceilings.json here). The ceiling is the most a neuron can write by itself
(`census.checkpoint_ceilings`, as `atlas_ops.Weights.write_ceiling` reads it on the model); p50 is the stream's median
length leaving the layer (the stats index's, the census's push unit). For a sample of 512 of each layer's neurons
(seed 0 + the layer): the median ceiling, its spread, and the median as a share of p50, which is how many ceilings a
push of 1 of the stream is the inverse of.

    .venv/bin/python -I experiments/results/census_scaled_design/ceilings.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from backend import census as cz  # noqa: E402
import export_control_vector as ecv  # noqa: E402

files = ecv.checkpoint_files(None)
z = np.load(ROOT / "atlas_data" / "index" / "act_scale.npz", allow_pickle=False)
p50 = [float(x) for x in z["resid_q"][:, 1]]
out = {"sample": 512, "p50_source": "atlas_data/index/act_scale.npz (the stats index)", "layers": {}}
for layer in range(len(p50)):
    width = cz.checkpoint_ceilings(files, layer, [0])[1]
    idx = sorted(int(i) for i in np.random.default_rng(layer).choice(width, 512, replace=False))
    c = np.array(list(cz.checkpoint_ceilings(files, layer, idx)[0].values()))
    med = float(np.median(c))
    out["layers"][str(layer)] = {"p50": round(p50[layer], 3), "ceiling": round(med, 3), "ceiling_p10": round(float(np.percentile(c, 10)), 3),
                                 "ceiling_p90": round(float(np.percentile(c, 90)), 3), "ceiling_over_p50": round(med / p50[layer], 4),
                                 "ceilings_in_one_stream": round(p50[layer] / med, 3)}
    print(f"L{layer:>2}: p50 {p50[layer]:7.1f}  ceiling {med:6.1f} ({np.percentile(c, 10):.1f}-{np.percentile(c, 90):.1f})  ceiling/p50 {med / p50[layer]:.3f}  1 stream = {p50[layer] / med:5.2f} ceilings", flush=True)
(HERE / "ceilings.json").write_text(json.dumps(out, indent=1) + "\n")
