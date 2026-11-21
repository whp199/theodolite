"""The volunteer's whole layer-30 run cut to the trial's 1,022 neurons, with every control and baseline kept, and a
meta naming those neurons, for things_read.py (reported, unregistered: a diagnostic of the whole layer's miss)."""
import json, sys
whole_run, whole_meta, trial_meta, out_dir = sys.argv[1:5]
from pathlib import Path
out = Path(out_dir)
t = json.load(open(trial_meta))
m = json.load(open(whole_meta))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from backend import census as cz
keep = set(cz.parse_neurons(t["neuron_ids"], int(t["d_ff"])))
m["neurons"], m["neuron_ids"] = t["neurons"], t["neuron_ids"]
n = 0
with open(out / "L30-match_things.jsonl", "w") as f:
    for s in open(whole_run):
        if '"kind": "neuron"' in s[:60]:
            if json.loads(s)["neuron"] not in keep:
                continue
        f.write(s); n += 1
m["jobs"] = cz.jobs_digest(cz.census_jobs(sorted(keep), int(m["random_controls"]), [tuple(x) for x in m["design"]], cz.meta_questions(m)))
(out / "L30-match_things.meta.json").write_text(json.dumps(m))
print(f"{n:,} rows; {len(keep)} neurons kept")
