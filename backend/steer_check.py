"""The steer lens's own check: how much of a reading its layer shares, and how the lens fares beside the census.

CPU only; no model is loaded. It reads the built steer lens (`steer_lens.npz`), the vocabulary beside it, the whole
censuses (`census.layer_runs`) and, where a layer has no whole census, a trial of it whose doses are counted
calibrated there (`census.doses_calibrated`). It writes `steer_check.json` beside the index, which the server reads
(`AtlasIndex.steer_check`) so that a page can say how far a steer-lens reading at a layer is to be trusted.

Why it exists. The lens lists the word-pieces whose odds rose most when a neuron is pushed. Nothing in it sets the
neuron against other pushes, as the census's controls do. Built on the 31B on 13 November 2026, each layer at the dose
its pilot chose, it agrees with the census at layers 39 to 52 and less and less below: of the neurons that stand above
95 in 100 controls, 70 to 91 in 100 share a word with it at layers 41 to 52, 60 to 62 at 39 and 40, 33 at 38 and 7 to
20 at 32 to 37 (1 to 6 with the neurons shuffled). And below layer 40 part of every reading is the layer's own: sentence
openers such as "Here", "The" and "You", listed by a tenth to a half of a layer's neurons whichever is pushed
(experiments/results/atlas_steer_dose/read_lens.py, whose measure this is).

The measure, for one layer: a neuron "shares a word" with its census when one of its steer-lens word-pieces (the
twelve of either sign, three letters or more, case and marks dropped) is one of its five census words, or begins one,
or is begun by one. It is counted over the neurons that stand above 95 in 100 of the run's controls (GROUP), which a
trial of 1,022 neurons fills with a few hundred, and beside it the same count with the neurons shuffled among
themselves (five shuffles, seed 0): the lens agrees with the layer's census when it shares a word with STEER_AGREES
of them or more. A layer with no census is "unchecked", and is given the nearest layer that has one.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend import census as cz
from backend.atlas_index import FILES, STEER_AGREES, STEER_CHECK_FILE, STEER_COMMON, index_dir, steer_common_shares

VERSION = 1
GROUP = "above 95 in 100 controls"
GROUP_FLOOR = 0.95
SHUFFLES = 5


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalpha())


def _hit(census_words: Sequence[str], pieces: Sequence[str]) -> bool:
    return any(w == s or w.startswith(s) or s.startswith(w) for w in census_words for s in pieces)


def against(words: Dict[int, Tuple[List[str], float]], up: np.ndarray, down: np.ndarray, vocab: Sequence[str],
            rng: np.random.Generator) -> Dict[str, Dict[str, Any]]:
    """`words`: neuron -> (its five census words, the share of the run's controls it stands above). For each group of
    neurons with a census word to show: how many share a word with their own steer-lens reading (`matched`), and how
    many with another neuron's of the same group (`neurons_shuffled`)."""
    steer = {n: {w for w in (_norm(vocab[int(x)]) for x in list(up[n]) + list(down[n])) if len(w) >= 3} for n in words}
    groups = {"above every control": [n for n, (w, p) in words.items() if p == 1.0 and w],
              GROUP: [n for n, (w, p) in words.items() if p >= GROUP_FLOOR and w],
              "the rest": [n for n, (w, p) in words.items() if p < GROUP_FLOOR and w]}
    out: Dict[str, Dict[str, Any]] = {}
    for name, ns in groups.items():
        if len(ns) < 2:
            continue
        got = float(np.mean([_hit(words[n][0], steer[n]) for n in ns]))
        base = float(np.mean([np.mean([_hit(words[n][0], steer[m]) for n, m in zip(ns, rng.permutation(ns))]) for _ in range(SHUFFLES)]))
        out[name] = {"neurons": len(ns), "matched": round(got, 4), "neurons_shuffled": round(base, 4)}
    return out


def shared_count(a: np.ndarray) -> float:
    """Of a neuron's word-pieces on one sign (`a`: neurons x k), how many the layer shares, on average: those listed by
    STEER_COMMON of the layer's neurons or more."""
    shares = steer_common_shares(a)
    if not shares:
        return 0.0
    return float(np.isin(a, np.fromiter(shares, dtype=np.int64)).sum(-1).mean())


def trial_runs(trials_dir: Path, model: Optional[str], calibrated: Callable[..., bool]) -> Dict[int, Path]:
    """layer -> a ranked trial of it whose design's doses are counted calibrated there (the larger, where two are)."""
    best: Dict[int, Tuple[int, Path]] = {}
    for units in sorted(Path(trials_dir).glob("**/*.units.jsonl")):
        run = Path(str(units)[: -len(".units.jsonl")] + ".jsonl")
        try:
            m = json.loads(cz.meta_path_of(run).read_text(encoding="utf-8"))
            layer, design = int(m["layer"]), str(m["design_name"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (model is not None and m.get("model") != model) or not calibrated(design, layer, m.get("max_new")):
            continue
        size = units.stat().st_size
        if layer not in best or size > best[layer][0]:
            best[layer] = (size, run)
    return {layer: run for layer, (_, run) in best.items()}


def measure(root: Optional[Path] = None, runs_dir: Optional[Path] = None, trials_dir: Optional[Path] = None,
            vocab: Optional[Sequence[str]] = None, calibrated: Callable[..., bool] = cz.doses_calibrated) -> Dict[str, Any]:
    """The check of the steer lens in `root` (the index folder), as `steer_check.json` keeps it."""
    root = Path(root) if root is not None else index_dir()
    path = root / FILES["steer_lens"]
    if not path.exists():
        raise ValueError(f"{path} is not built: the steer lens is merged when its last layer is (atlas_build.py steer).")
    with np.load(path, allow_pickle=False) as z:
        up, down = z["up"], z["down"]
        kl = (z["up_kl"].astype(np.float64) + z["down_kl"].astype(np.float64)) / 2
        dose = np.asarray(z["dose"], dtype=np.float64).reshape(-1)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if vocab is None:
        vp = root / "vocab.json"
        if not vp.exists():
            raise ValueError(f"{vp} is not there: the server writes it when it first needs the vocabulary. Start it once.")
        vocab = json.loads(vp.read_text(encoding="utf-8"))["vocab"]
    L = up.shape[0]
    model = meta.get("model")
    layers: Dict[str, Dict[str, Any]] = {}
    for l in range(L):
        layers[str(l)] = {"dose": round(float(dose[l] if dose.size > 1 else dose[0]), 4), "kl_p50": round(float(np.median(kl[l])), 3),
                          "shared": round((shared_count(up[l]) + shared_count(down[l])) / 2, 2), "of": int(up.shape[-1])}
    rng = np.random.default_rng(0)
    whole = {l: p for l, p in cz.layer_runs(runs_dir, model=model).items() if l < L and cz.units_path_of(p).exists()}
    for layer in sorted(whole):
        t = cz.load_table(whole[layer])
        words = {int(t.neuron[i]): (t.shown(i, 5), float(t.null_pct[i])) for i in range(len(t)) if t.is_neuron[i]}
        m = json.loads(cz.meta_path_of(whole[layer]).read_text(encoding="utf-8"))
        layers[str(layer)]["census"] = {"kind": "whole", "run": whole[layer].name, "design": m.get("design_name"), "neurons": len(words),
                                        "controls": int((~t.is_neuron).sum()), "groups": against(words, up[layer], down[layer], vocab, rng)}
    trials_dir = Path(trials_dir) if trials_dir is not None else cz.census_dir().parent / "trials"
    if trials_dir.is_dir():
        for layer, run in sorted(trial_runs(trials_dir, model, calibrated).items()):
            if layer in whole or layer >= L:
                continue
            m = json.loads(cz.meta_path_of(run).read_text(encoding="utf-8"))
            with open(cz.units_path_of(run), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            words = {int(u["neuron"]): (cz.shown_words(u, 5), float(u["null_pct"])) for u in rows if u.get("kind") == "neuron"}
            layers[str(layer)]["census"] = {"kind": "trial", "run": run.name, "design": m.get("design_name"), "neurons": len(words),
                                            "controls": sum(u.get("kind") != "neuron" for u in rows),
                                            "groups": against(words, up[layer], down[layer], vocab, rng)}
    checked = {}
    for l in range(L):
        row = layers[str(l)]
        g = ((row.get("census") or {}).get("groups") or {}).get(GROUP)
        if g:
            row["matched"], row["shuffled"], row["neurons"] = g["matched"], g["neurons_shuffled"], g["neurons"]
            row["standing"] = "agrees" if g["matched"] >= STEER_AGREES else "weak"
            checked[l] = row
        else:
            row.pop("census", None)  # a census with too few neurons standing out to count checks nothing
            row["standing"] = "unchecked"
    for l in range(L):
        row = layers[str(l)]
        if row["standing"] == "unchecked" and checked:
            near = min(checked, key=lambda c: (abs(c - l), c))
            row["nearest"] = {"layer": near, "matched": checked[near]["matched"], "standing": checked[near]["standing"]}
    return {"version": VERSION,
            "what": "the steer lens set beside the census, layer by layer, and how much of each reading its layer shares (backend/steer_check.py)",
            "lens": {"built_at": meta.get("built_at"), "model": model, "weights": meta.get("weights"), "dose_rule": meta.get("dose_rule")},
            "group": GROUP, "agrees_at": STEER_AGREES, "common_floor": STEER_COMMON,
            "written_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "layers": layers}


def write(root: Optional[Path] = None, **kw: Any) -> Path:
    """Measure the check and keep it beside the index (written whole, then renamed, so the server never reads half)."""
    root = Path(root) if root is not None else index_dir()
    d = measure(root, **kw)
    out = root / STEER_CHECK_FILE
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def said(d: Dict[str, Any]) -> List[str]:
    """The check a line a layer, for a terminal."""
    out = []
    for l, row in sorted(d["layers"].items(), key=lambda kv: int(kv[0])):
        c = row.get("census")
        tail = (f"{c['kind']} census ({c.get('design')}): shares a word with {row['matched']:.3f} of the {row['neurons']} {d['group']}"
                f" ({row['shuffled']:.3f} shuffled)" if c else
                "no census" + (f"; nearest is layer {row['nearest']['layer']} ({row['nearest']['matched']:.3f}, {row['nearest']['standing']})"
                               if row.get("nearest") else ""))
        out.append(f"layer {int(l):>2}: dose {row['dose']:<6} median KL {row['kl_p50']:<5} {row['shared']:>5.2f} of {row['of']} shared   "
                   f"{row['standing']:<9} {tail}")
    return out
