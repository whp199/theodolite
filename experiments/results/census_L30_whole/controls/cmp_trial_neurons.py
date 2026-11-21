"""The volunteer's whole-run answers for the trial's neurons against the home card's trial answers (same jobs)."""
import json, sys
import numpy as np
trial_p, whole_p = sys.argv[1], sys.argv[2]
def key(r): return (r["kind"], r["neuron"], r["prompt"], r["dose"])
home = {}
for s in open(trial_p):
    r = json.loads(s)
    if r["kind"] == "neuron":
        home[key(r)] = r
n = same = first = 0; dl = []; pref = []
for s in open(whole_p):
    if '"kind": "neuron"' not in s[:60]:
        continue
    r = json.loads(s)
    h = home.get(key(r))
    if h is None:
        continue
    n += 1
    a, b = h["token_ids"], r["token_ids"]
    same += a == b and h.get("logprob") == r.get("logprob")
    first += bool(a) and bool(b) and a[0] == b[0]
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    pref.append(k)
    la, lb = h.get("logprob") or [], r.get("logprob") or []
    dl += [abs(x - y) for x, y in zip(la[:k], lb[:k])]
print(f"trial neuron answers: {len(home):,}; found in the whole run: {n:,}; identical bit for bit: {same:,}")
print(f"first word-piece the same: {first / n:.4f}; shared prefix median {np.median(pref):.0f}, mean {np.mean(pref):.1f} of 48; "
      f"whole text the same: {np.mean([p >= 48 for p in pref]):.3f}")
d = np.array(dl); print(f"|d log-prob| on shared prefixes: median {np.median(d):.4f}, p99 {np.quantile(d, 0.99):.3f} over {len(d):,} pieces")
