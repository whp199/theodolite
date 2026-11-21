"""The subset run with only one half of its 2,048 controls (renumbered from 0), for things_read.py (diagnostic)."""
import json, sys
from pathlib import Path
src, meta, lo, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
m = json.load(open(meta)); m["random_controls"] = 1024
with open(out / "L30-match_things.jsonl", "w") as f:
    for s in open(src):
        if '"kind": "random"' in s[:60]:
            r = json.loads(s)
            if not lo <= r["neuron"] < lo + 1024:
                continue
            r["neuron"] -= lo
            s = json.dumps(r, ensure_ascii=False) + "\n"
        f.write(s)
(out / "L30-match_things.meta.json").write_text(json.dumps(m))
