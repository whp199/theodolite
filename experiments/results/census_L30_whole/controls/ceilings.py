"""The write ceilings the controls' pushes are scaled by (each that of the neuron it is paired with, ref_neuron), in a
run against those of the run's neurons: whether two draws of controls differ in push size as well as direction."""
import json, sys
import numpy as np
for p in sys.argv[1:]:
    c, n = {}, {}
    for s in open(p):
        if '"kind": "baseline"' in s[:60]:
            continue
        r = json.loads(s)
        if r["kind"] == "random":
            c[r["neuron"]] = (r.get("ceiling"), r.get("ref_neuron"))
        elif r["kind"] == "neuron":
            n[r["neuron"]] = r.get("ceiling")
    cc = np.array([v[0] for v in c.values()], float); nn = np.array(list(n.values()), float)
    print(p.split("/")[-1], f"controls {len(cc)}: ceiling median {np.median(cc):.2f} p10 {np.quantile(cc, .1):.2f} "
          f"p90 {np.quantile(cc, .9):.2f}; neurons {len(nn)}: median {np.median(nn):.2f} p10 {np.quantile(nn, .1):.2f} "
          f"p90 {np.quantile(nn, .9):.2f}; controls paired with one of these neurons: {np.mean([v[1] in n for v in c.values()]):.2f}")
