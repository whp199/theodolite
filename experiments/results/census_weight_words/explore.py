"""Exploratory, unregistered (22 November 2026; the owner's "Let's try both of those recommendations"): read a census
trial on words fixed before any answer is read, each neuron's own weight-lens words, instead of words picked from its
answers.

The census picks a neuron's words from its pushed answers and asks whether they carry from one half of the questions
to the other (G3). At layer 55 they did not (0.518 and 0.539 against 0.57), though the lead passed. A volunteer
suggested that at the top layers a reading from the weights alone might be the better instrument. This reads that
suggestion as a test of behaviour, not a label: does pushing a neuron bring out the words its write direction names
when read straight through the final norm and the unembedding (the atlas's "naked eye", index/lens.npz eye_pos and
eye_neg, first ten each), more than pushing a random direction of the same size does?

For each neuron n: W+(n) and W-(n), its naked eye's first ten word-pieces up and down, as plain lower-case words. An
answer's score on a word set is the share of its word-pieces that are in it. A unit's score for n is the mean over its
pushed-up answers on W+(n) and its pushed-down answers on W-(n). The neuron's own score is set against all controls'
scores on the same two sets: its share of controls below it (ties half) and p = (1 + controls at or above) / (1 + C).
Reported for the whole run, each half of the questions alone, and each dose alone:
- lead: the mean over neurons of that share (0.5 where the neurons do nothing the controls do not);
- q05: neurons with q <= 0.05 (Benjamini-Hochberg over the neurons' p);
- rho: across neurons, how the share on half A ranks with the share on half B (the controls' the same, beside it).
Nothing is picked from the answers, so the two halves are two separate tests of the same fixed claim.

What it would not show: what a neuron does beyond moving its own unembedding words. At the top layers a push along a
direction raises the logits of the word-pieces that direction names almost by construction; a pass there says the
labels describe what the push does, which is a narrower claim than the census's.

    python explore.py <trial run .jsonl> [...]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "atlas_data"
HALVES = (("bicycle", "walk", "chair", "kettle", "sleep", "lamp"), ("clock", "rice", "vase", "lock", "money", "blanket"))


def norm(s):
    return re.sub(r"[^\w]", "", str(s).replace("▁", " ").strip().lower())


V = json.load(open(A / "index" / "vocab.json"))["vocab"]
VN = [norm(t) for t in V]
LENS = np.load(A / "index" / "lens.npz")
EYE = {+1: LENS["eye_pos"], -1: LENS["eye_neg"]}


def read(path):
    rows = []
    with open(path) as f:
        for s in f:
            r = json.loads(s)
            if r["kind"] == "baseline":
                continue
            rows.append((r["kind"], int(r["neuron"]), r["prompt"], float(r["dose"]), r.get("token_ids") or []))
    return rows


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0)
    return out


def main(path):
    rows = read(path)
    layer = int(re.search(r"L(\d+)-", Path(path).name).group(1))
    neurons = sorted({n for k, n, *_ in rows if k == "neuron"})
    controls = sorted({n for k, n, *_ in rows if k == "random"})
    # each neuron's two fixed word sets, and one index over every word in any of them
    W = {s: [set(VN[t] for t in EYE[s][layer, n, :10] if VN[t]) for n in neurons] for s in (+1, -1)}
    vocab = sorted(set().union(*W[+1], *W[-1]))
    vi = {w: i for i, w in enumerate(vocab)}
    M = {s: sparse.csr_matrix((np.ones(sum(len(x) for x in W[s])),
                               ([vi[w] for x in W[s] for w in x], [j for j, x in enumerate(W[s]) for _ in x])),
                              shape=(len(vocab), len(neurons))) for s in (+1, -1)}
    mags = sorted({abs(d) for *_, d, _ in rows})
    subsets = {"whole": lambda q, d: True, "half A": lambda q, d: q in HALVES[0], "half B": lambda q, d: q in HALVES[1],
               "lower dose": lambda q, d: abs(d) == mags[0], "upper dose": lambda q, d: abs(d) == mags[-1]}
    # per unit, per sign, per subset: the mean over its answers of each indexed word's share of the answer
    units = [("neuron", n) for n in neurons] + [("random", c) for c in controls]
    ui = {u: i for i, u in enumerate(units)}
    acc = {(name, s): np.zeros((len(units), len(vocab)), np.float32) for name in subsets for s in (+1, -1)}
    cnt = {(name, s): np.zeros(len(units), np.float32) for name in subsets for s in (+1, -1)}
    for kind, n, q, d, ids in rows:
        s = 1 if d > 0 else -1
        hits = [vi[w] for w in (VN[t] for t in ids) if w in vi]
        vec = np.bincount(hits, minlength=len(vocab)).astype(np.float32) / max(len(ids), 1) if hits else None
        u = ui[(kind, n)]
        for name, keep in subsets.items():
            if keep(q, d):
                cnt[(name, s)][u] += 1
                if vec is not None:
                    acc[(name, s)][u] += vec
    nn = len(neurons)
    print(f"{Path(path).name}: layer {layer}, {nn} neurons, {len(controls)} controls, doses {mags}, "
          f"{len(vocab)} words in the neurons' naked-eye sets")
    share = {}
    for name in subsets:
        S = sum(sparse.csr_matrix(acc[(name, s)] / np.maximum(cnt[(name, s)], 1)[:, None]) @ M[s] for s in (+1, -1)) / 2
        S = np.asarray(S.todense())  # units x neurons: each unit's score on each neuron's words
        own = S[np.arange(nn), np.arange(nn)]
        ctl = S[nn:, :]  # controls x neurons
        below = ((ctl < own).sum(0) + 0.5 * (ctl == own).sum(0)) / ctl.shape[0]
        p = (1 + (ctl >= own).sum(0)) / (1 + ctl.shape[0])
        q = bh(p)
        share[name] = (below, ctl)
        # the null for "its own words": each neuron's answers read on another neuron's words (a fixed shuffle)
        perm = np.random.default_rng(0).permutation(nn)
        sh = S[perm, np.arange(nn)]
        below_sh = ((ctl < sh).sum(0) + 0.5 * (ctl == sh).sum(0)) / ctl.shape[0]
        print(f"  {name:10s} lead {below.mean():.3f} (shuffled {below_sh.mean():.3f})  q<=0.05 {int((q <= 0.05).sum()):5d}  "
              f"above every control {int((own > ctl.max(0)).sum()):5d}  own score > 0 in {np.mean(own > 0):.2f} of neurons, "
              f"controls' {np.mean(ctl > 0):.2f}")
    a, b = share["half A"][0], share["half B"][0]
    ca, cb = share["half A"][1], share["half B"][1]

    def rank_corr(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    # the controls' own ranks across halves: each control's share of the others below it, on each neuron's words
    ctl_rho = np.mean([rank_corr(ca[:, j], cb[:, j]) for j in range(0, nn, max(1, nn // 64))])
    print(f"  across halves: neurons' shares rank together rho {rank_corr(a, b):.3f}; a control's score on the same "
          f"words, half A against half B, rho {ctl_rho:.3f} (mean over 64 neurons' words)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
