"""A run of one source's baselines and neurons with another source's controls, for things_read.py (diagnostic)."""
import json, sys
from pathlib import Path
nrun, nmeta, crun, n_ctl, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
m = json.load(open(nmeta)); m["random_controls"] = n_ctl
k = {"n": 0, "c": 0}
with open(out / "L30-match_things.jsonl", "w") as f:
    for s in open(nrun):
        if '"kind": "random"' not in s[:60]:
            f.write(s); k["n"] += 1
    for s in open(crun):
        if '"kind": "random"' in s[:60]:
            f.write(s); k["c"] += 1
(out / "L30-match_things.meta.json").write_text(json.dumps(m))
print(out.name, k)
