#!/usr/bin/env python3
"""Experiment 1A (CPU): Calibrate Design 2 (T_dec + empirical FDP) across finished trials & whole layers.

Tests:
  - Baseline reproduction of L55 trial against L55_verdict.whole.json (sanity check).
  - `raw` vs `no_thought` (stripping any word containing "thought" before counting).
  - Residualized cross-half replication `rep_AB_res`, `rep_BA_res` and composite score
    `T_dec(u) = min(rz(rep_AB_res), rz(rep_BA_res))` with empirical FDP <= 0.05 / 0.10.
  - 50 random half-control null splits (full re-estimation within each split).
  - Out-of-sample 3-way split on 12-question runs (fit on q[0:4] vs q[4:8], test replicated words on q[8:12]).
  - Lower-dose-only calibration on L53, L54, L55 trials.
"""
from __future__ import annotations

import collections
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "experiments/results/census_gap_pilots"

sys.path.insert(0, str(REPO_ROOT))
from backend import atlas_index as ai  # noqa: E402
from backend import census as cz  # noqa: E402
from backend.gap_stats import mannwhitney_greater  # noqa: E402

COVS = ("lnw", "d4", "cut", "lp", "dep", "lost")


def auc(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    return mannwhitney_greater(a, b)


def emp_p_map(d: Dict[Tuple[str, int], float]) -> Dict[Tuple[str, int], float]:
    ctl = np.sort([v for u, v in d.items() if u[0] == "random"])
    C = len(ctl)
    out: Dict[Tuple[str, int], float] = {}
    for u, v in d.items():
        ge = C - int(np.searchsorted(ctl, v, side="left"))
        out[u] = (1 + ge) / (C + 1) if u[0] == "neuron" else (ge / C if C > 0 else 1.0)
    return out


def bh_map(ps: Dict[Tuple[str, int], float]) -> Dict[Tuple[str, int], float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    n = len(items)
    q, out = 1.0, {}
    for i in range(n - 1, -1, -1):
        q = min(q, items[i][1] * n / (i + 1))
        out[items[i][0]] = q
    return out


def residualise(stat: Dict[Tuple[str, int], float], cov: Dict[Tuple[str, int], Dict[str, float]],
                ctl_pool: Optional[Sequence[Tuple[str, int]]] = None) -> Dict[Tuple[str, int], float]:
    units = [u for u in stat if u in cov]
    ctl = list(ctl_pool) if ctl_pool is not None else [u for u in units if u[0] == "random"]
    X = np.array([[1.0] + [cov[u][c] for c in COVS] for u in ctl], dtype=np.float64)
    y = np.array([stat[u] for u in ctl], dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {u: float(stat[u] - np.dot([1.0] + [cov[u][c] for c in COVS], beta)) for u in units}


def robust_z(stat: Dict[Tuple[str, int], float], ctl_pool: Optional[Sequence[Tuple[str, int]]] = None) -> Dict[Tuple[str, int], float]:
    ctl = np.array([stat[u] for u in (ctl_pool if ctl_pool is not None else [u for u in stat if u[0] == "random"])], dtype=np.float64)
    med = float(np.median(ctl))
    mad = float(np.median(np.abs(ctl - med)))
    std = float(np.std(ctl))
    scale = max(1.4826 * mad, 0.25 * std, 1e-6)
    return {u: float((v - med) / scale) for u, v in stat.items()}


def z_top(c: Dict[int, int], tot: Dict[int, int], den: int, loo: bool, k: int = 5, min_count: int = 2) -> List[Tuple[float, int]]:
    zs = []
    for w, cnt in c.items():
        if cnt < min_count:
            continue
        mu = (tot.get(w, 0) - (cnt if loo else 0)) / den
        zs.append(((cnt - mu) / math.sqrt(mu + 0.5), w))
    zs.sort(reverse=True)
    return zs[:k]


def mass_from_counts(cnt: Dict[Tuple[str, int], collections.Counter], ctl_pool: Optional[Sequence[Tuple[str, int]]] = None,
                     k: int = 5) -> Dict[Tuple[str, int], Tuple[float, List[Tuple[float, int]]]]:
    ctl = list(ctl_pool) if ctl_pool is not None else [u for u in cnt if u[0] == "random"]
    tot: collections.Counter = collections.Counter()
    for u in ctl:
        tot.update(cnt[u])
    C = len(ctl)
    out = {}
    for u in cnt:
        loo = u[0] == "random"
        den = (C - 1) if loo else C
        top = z_top(cnt[u], tot, den, loo, k=k)
        out[u] = (float(sum(max(z, 0.0) for z, _ in top)), top)
    return out


def z_of(cnt: Dict[Tuple[str, int], collections.Counter], words_of: Dict[Tuple[str, int], List[int]],
         ctl_pool: Optional[Sequence[Tuple[str, int]]] = None) -> Dict[Tuple[str, int], float]:
    ctl = list(ctl_pool) if ctl_pool is not None else [u for u in cnt if u[0] == "random"]
    tot: collections.Counter = collections.Counter()
    for u in ctl:
        tot.update(cnt[u])
    C = len(ctl)
    out = {}
    for u, ws in words_of.items():
        loo = u[0] == "random"
        den = (C - 1) if loo else C
        s = 0.0
        cu = cnt[u]
        for w in ws:
            c = cu.get(w, 0)
            mu = (tot.get(w, 0) - (c if loo else 0)) / den
            s += (c - mu) / math.sqrt(mu + 0.5)
        out[u] = float(s)
    return out


def emp_fdp_select(scores: Dict[Tuple[str, int], float], positive_ok: Dict[Tuple[str, int], bool],
                   alpha: float = 0.05) -> Dict[str, Any]:
    """Step-up empirical FDP: FDP(t) = (N/(C+1)) * (1 + #{c >= t}) / max(1, #{n >= t})."""
    neu_vals = np.array([scores[u] for u in scores if u[0] == "neuron" and positive_ok[u]], dtype=np.float64)
    neu_units = [u for u in scores if u[0] == "neuron" and positive_ok[u]]
    N = sum(1 for u in scores if u[0] == "neuron")
    ctl_vals = np.sort(np.array([scores[u] for u in scores if u[0] == "random"], dtype=np.float64))
    C = len(ctl_vals)
    if len(neu_vals) == 0 or C == 0:
        return {"count": 0, "threshold": None, "fdp": None, "named": [], "above_max_ctl": 0}
    order = np.argsort(-neu_vals)
    sorted_vals = neu_vals[order]
    # For each unique descending threshold among candidate neurons
    best_idx = -1
    best_fdp = 1.0
    n_tot = len(sorted_vals)
    i = 0
    while i < n_tot:
        j = i
        t = sorted_vals[i]
        while j + 1 < n_tot and sorted_vals[j + 1] == t:
            j += 1
        n_ge = j + 1
        c_ge = C - int(np.searchsorted(ctl_vals, t, side="left"))
        fdp = (N / (C + 1)) * (1.0 + c_ge) / max(1, n_ge)
        if fdp <= alpha:
            best_idx = j
            best_fdp = fdp
        i = j + 1
    max_ctl = float(ctl_vals[-1])
    above_max = int(np.sum(neu_vals > max_ctl))
    if best_idx < 0:
        return {"count": 0, "threshold": None, "fdp": None, "named": [], "above_max_ctl": above_max}
    thresh = float(sorted_vals[best_idx])
    named = [neu_units[k] for k in order[: best_idx + 1]]
    return {"count": len(named), "threshold": round(thresh, 4), "fdp": round(best_fdp, 4),
            "named": named, "above_max_ctl": above_max}


class StreamedRun:
    """Compact memory-bounded representation of a census JSONL run aggregated over requested prompt-subsets."""

    def __init__(self, path: Path, subsets: Dict[str, Callable[[str, float], bool]]):
        t0 = time.time()
        self.path = Path(path)
        meta = json.loads(cz.meta_path_of(self.path).read_text(encoding="utf-8"))
        self.layer = int(meta["layer"])
        self.design = str(meta.get("design_name"))
        self.prompts = [q for q in (meta.get("prompts") or {}) if any(p == q for p, _ in meta["design"])]
        self.doses = sorted({abs(float(d)) for _, d in meta["design"]})

        # Pass 1: fast binary scan for baseline rows (~2s on 5.6 GB)
        self.vocab: List[str] = []
        self._wid: Dict[str, int] = {}
        self.thought_wids: Set[int] = set()
        bases: Dict[str, Dict[str, Any]] = {}
        with open(self.path, "rb") as f:
            for line in f:
                if b'"baseline"' in line and b'"kind": "baseline"' in line:
                    r = json.loads(line)
                    if r.get("kind") == "baseline" and r["prompt"] not in bases:
                        ws = self._to_wids(cz.words(r["text"]))
                        bases[r["prompt"]] = {
                            "wids": set(ws),
                            "ids": [int(x) for x in (r.get("token_ids") or [])],
                            "d4": cz.distinct4(r.get("token_ids") or []),
                        }

        # Pass 2: stream non-baseline rows and accumulate directly into subset buckets
        self.sub_cnt: Dict[str, Dict[Tuple[str, int], collections.Counter]] = {
            name: collections.defaultdict(collections.Counter) for name in subsets
        }
        self.sub_cov_acc: Dict[str, Dict[Tuple[str, int], List[float]]] = {
            name: collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for name in subsets
        }
        n_rows = 0
        sub_items = list(subsets.items())
        with open(self.path, "rb") as f:
            for line in f:
                if b'"kind": "baseline"' in line:
                    continue
                r = json.loads(line)
                if r.get("kind") == "baseline":
                    continue
                n_rows += 1
                u = (str(r["kind"]), int(r["neuron"]))
                p = str(r["prompt"])
                d = float(r["dose"])
                b = bases[p]
                bw = b["wids"]
                rw_set = set(self._to_wids(cz.words(r["text"])))
                new_wids = rw_set - bw
                ids = r.get("token_ids") or []
                lp = r.get("logprob") or []
                nw = float(len(new_wids))
                d4 = cz.distinct4(ids)
                cut = 1.0 if r.get("finish") == "length" else 0.0
                lp_m = float(np.mean(lp)) if lp else 0.0
                dep_idx = cz.first_departure(ids, b["ids"])
                dep = (len(ids) if dep_idx is None else dep_idx) / max(len(ids), 1)
                lost = len(bw - rw_set) / max(len(bw), 1)

                for sname, pred in sub_items:
                    if pred(p, d):
                        cnt_u = self.sub_cnt[sname][u]
                        for wi in new_wids:
                            cnt_u[wi] += 1
                        acc = self.sub_cov_acc[sname][u]
                        acc[0] += nw
                        acc[1] += d4
                        acc[2] += cut
                        acc[3] += lp_m
                        acc[4] += dep
                        acc[5] += lost
                        acc[6] += 1.0

        self.sub_cov: Dict[str, Dict[Tuple[str, int], Dict[str, float]]] = {}
        for sname, acc_map in self.sub_cov_acc.items():
            cdict = {}
            for u, acc in acc_map.items():
                n = max(acc[6], 1.0)
                cdict[u] = {
                    "lnw": math.log1p(acc[0] / n),
                    "d4": acc[1] / n,
                    "cut": acc[2] / n,
                    "lp": acc[3] / n,
                    "dep": acc[4] / n,
                    "lost": acc[5] / n,
                }
            self.sub_cov[sname] = cdict
        del self.sub_cov_acc
        self.units = sorted(self.sub_cnt[next(iter(subsets))].keys())
        self.neurons = [u for u in self.units if u[0] == "neuron"]
        self.controls = [u for u in self.units if u[0] == "random"]
        print(f"Loaded {self.path.name}: {n_rows:,} rows, {len(self.neurons)} neurons, {len(self.controls)} controls in {time.time()-t0:.1f}s", flush=True)

    def _to_wids(self, ws: Iterable[str]) -> List[int]:
        out = []
        for w in ws:
            wi = self._wid.get(w)
            if wi is None:
                wi = self._wid[w] = len(self.vocab)
                self.vocab.append(w)
                if "thought" in w:
                    self.thought_wids.add(wi)
            out.append(wi)
        return out

    def filtered_counts(self, subset: str, strip_thought: bool) -> Dict[Tuple[str, int], collections.Counter]:
        src = self.sub_cnt[subset]
        if not strip_thought or not self.thought_wids:
            return src
        tw = self.thought_wids
        return {u: collections.Counter({wi: c for wi, c in cnt.items() if wi not in tw}) for u, cnt in src.items()}


def eval_pair(run: StreamedRun, half_a: str, half_b: str, all_sub: str, strip_thought: bool,
              run_null_splits: int = 50, seed: int = 0) -> Dict[str, Any]:
    cnt_all = run.filtered_counts(all_sub, strip_thought)
    cnt_A = run.filtered_counts(half_a, strip_thought)
    cnt_B = run.filtered_counts(half_b, strip_thought)
    cov_all = run.sub_cov[all_sub]

    m_all = mass_from_counts(cnt_all)
    mA = mass_from_counts(cnt_A)
    mB = mass_from_counts(cnt_B)
    mass_c = {u: v[0] for u, v in m_all.items()}
    rep_AB = z_of(cnt_B, {u: [w for z, w in mA[u][1] if z > 0] for u in mA})
    rep_BA = z_of(cnt_A, {u: [w for z, w in mB[u][1] if z > 0] for u in mB})
    rep = {u: rep_AB[u] + rep_BA[u] for u in rep_AB}

    mass_c_res = residualise(mass_c, cov_all)
    rep_AB_res = residualise(rep_AB, cov_all)
    rep_BA_res = residualise(rep_BA, cov_all)
    rep_res = residualise(rep, cov_all)

    # Readlib's baseline named rule
    p_all = emp_p_map(mass_c)
    p_r = emp_p_map(mass_c_res)
    p_AB = emp_p_map(rep_AB)
    p_BA = emp_p_map(rep_BA)
    q_all = bh_map({u: v for u, v in p_all.items() if u[0] == "neuron"})
    readlib_named = [u for u in q_all if q_all[u] <= 0.05 and p_AB[u] <= 0.05 and p_BA[u] <= 0.05 and p_r[u] <= 0.05]

    # Design 2 composite score: T_dec = min(rz(rep_AB_res), rz(rep_BA_res))
    rz_AB = robust_z(rep_AB_res)
    rz_BA = robust_z(rep_BA_res)
    T_dec = {u: min(rz_AB[u], rz_BA[u]) for u in rep_AB_res}
    pos_ok = {u: (rep_AB_res[u] > 0.0 and rep_BA_res[u] > 0.0) for u in T_dec}
    pos_ok_mr = {u: (pos_ok[u] and p_r[u] <= 0.05) for u in T_dec}

    sel_05 = emp_fdp_select(T_dec, pos_ok, 0.05)
    sel_10 = emp_fdp_select(T_dec, pos_ok, 0.10)
    sel_05_mr = emp_fdp_select(T_dec, pos_ok_mr, 0.05)

    res = {
        "mass_c_auc": round(auc([mass_c[u] for u in run.neurons], [mass_c[u] for u in run.controls])[0], 4),
        "mass_c_res_auc": round(auc([mass_c_res[u] for u in run.neurons], [mass_c_res[u] for u in run.controls])[0], 4),
        "rep_AB_auc": round(auc([rep_AB[u] for u in run.neurons], [rep_AB[u] for u in run.controls])[0], 4),
        "rep_BA_auc": round(auc([rep_BA[u] for u in run.neurons], [rep_BA[u] for u in run.controls])[0], 4),
        "rep_auc": round(auc([rep[u] for u in run.neurons], [rep[u] for u in run.controls])[0], 4),
        "rep_AB_res_auc": round(auc([rep_AB_res[u] for u in run.neurons], [rep_AB_res[u] for u in run.controls])[0], 4),
        "rep_BA_res_auc": round(auc([rep_BA_res[u] for u in run.neurons], [rep_BA_res[u] for u in run.controls])[0], 4),
        "rep_res_auc": round(auc([rep_res[u] for u in run.neurons], [rep_res[u] for u in run.controls])[0], 4),
        "T_dec_auc": round(auc([T_dec[u] for u in run.neurons], [T_dec[u] for u in run.controls])[0], 4),
        "readlib_q05": int(sum(1 for v in q_all.values() if v <= 0.05)),
        "readlib_named": len(readlib_named),
        "fdp05": {k: v for k, v in sel_05.items() if k != "named"},
        "fdp10": {k: v for k, v in sel_10.items() if k != "named"},
        "fdp05_plus_pr": {k: v for k, v in sel_05_mr.items() if k != "named"},
    }

    # 50 random half-control null splits (re-estimating totals, OLS, and robust z strictly on reference half)
    if run_null_splits > 0:
        rng = np.random.default_rng(seed)
        ctls = list(run.controls)
        null_aucs, null_fd05_1x, null_fd05_scaled = [], [], []
        ratio = len(run.neurons) / max(len(ctls), 1)
        for _ in range(run_null_splits):
            perm = rng.permutation(len(ctls))
            half = len(ctls) // 2
            fake = [ctls[i] for i in perm[:half]]
            ref = [ctls[i] for i in perm[half:]]
            fake_set = set(fake)
            # Re-label fake controls as ('neuron', id) cleanly inside dict copies
            subA = {((("neuron", u[1]) if u in fake_set else u)): cnt_A[u] for u in ctls}
            subB = {((("neuron", u[1]) if u in fake_set else u)): cnt_B[u] for u in ctls}
            subCov = {((("neuron", u[1]) if u in fake_set else u)): cov_all[u] for u in ctls}
            mA_n = mass_from_counts(subA)
            mB_n = mass_from_counts(subB)
            rAB_n = z_of(subB, {u: [w for z, w in mA_n[u][1] if z > 0] for u in mA_n})
            rBA_n = z_of(subA, {u: [w for z, w in mB_n[u][1] if z > 0] for u in mB_n})
            rAB_r = residualise(rAB_n, subCov)
            rBA_r = residualise(rBA_n, subCov)
            rzA = robust_z(rAB_r)
            rzB = robust_z(rBA_r)
            Tn = {u: min(rzA[u], rzB[u]) for u in rzA}
            pok = {u: (rAB_r[u] > 0.0 and rBA_r[u] > 0.0) for u in Tn}
            null_aucs.append(auc([Tn[u] for u in Tn if u[0] == "neuron"], [Tn[u] for u in Tn if u[0] == "random"])[0])
            null_fd05_1x.append(emp_fdp_select(Tn, pok, 0.05)["count"])
            # Also check how often a single fake-half split passes an effective alpha/ratio bar matching whole-run multiplicity
            null_fd05_scaled.append(emp_fdp_select(Tn, pok, min(0.05 / max(ratio / 2.0, 1.0), 0.05))["count"])
        res["null_splits"] = {
            "n": run_null_splits,
            "T_dec_auc_mean": round(float(np.mean(null_aucs)), 4),
            "T_dec_auc_max": round(float(np.max(null_aucs)), 4),
            "fdp05_false_mean": round(float(np.mean(null_fd05_1x)), 2),
            "fdp05_false_max": int(np.max(null_fd05_1x)),
            "fdp05_scaled_false_mean": round(float(np.mean(null_fd05_scaled)), 2),
        }

    return res


def out_of_sample_3way(run: StreamedRun, strip_thought: bool) -> Dict[str, Any]:
    """On 12-question runs: fit T_dec on A4=q[0:4] vs B4=q[4:8], test replicated words on unseen C4=q[8:12]."""
    cnt_A4 = run.filtered_counts("A4", strip_thought)
    cnt_B4 = run.filtered_counts("B4", strip_thought)
    cnt_C4 = run.filtered_counts("C4", strip_thought)
    # Approximate AB covariates as mean of A4 and B4? Better: we pre-accumulated 'AB8' and 'C4'!
    cov_AB8 = run.sub_cov["AB8"]
    mA4 = mass_from_counts(cnt_A4)
    mB4 = mass_from_counts(cnt_B4)
    rAB4 = residualise(z_of(cnt_B4, {u: [w for z, w in mA4[u][1] if z > 0] for u in mA4}), cov_AB8)
    rBA4 = residualise(z_of(cnt_A4, {u: [w for z, w in mB4[u][1] if z > 0] for u in mB4}), cov_AB8)
    rzA, rzB = robust_z(rAB4), robust_z(rBA4)
    T4 = {u: min(rzA[u], rzB[u]) for u in rzA}
    pok4 = {u: (rAB4[u] > 0.0 and rBA4[u] > 0.0) for u in T4}
    sel = emp_fdp_select(T4, pok4, 0.05)
    named = sel["named"]
    if not named:
        return {"fit_named": 0, "c4_auc": None, "frac_z_pos": None, "frac_p05": None}

    # Build each unit's replicated word list strictly from A4 and B4
    totA4, totB4 = collections.Counter(), collections.Counter()
    for c in run.controls:
        totA4.update(cnt_A4[c])
        totB4.update(cnt_B4[c])
    C = len(run.controls)

    def rep_words_of(u: Tuple[str, int]) -> List[int]:
        loo = u[0] == "random"
        den = (C - 1) if loo else C
        ws: List[int] = []
        seen: Set[int] = set()
        for _, w in mA4[u][1]:
            c = cnt_B4[u].get(w, 0)
            mu = (totB4.get(w, 0) - (c if loo else 0)) / den
            if (c - mu) / math.sqrt(mu + 0.5) > 0 and w not in seen:
                seen.add(w)
                ws.append(w)
        for _, w in mB4[u][1]:
            c = cnt_A4[u].get(w, 0)
            mu = (totA4.get(w, 0) - (c if loo else 0)) / den
            if (c - mu) / math.sqrt(mu + 0.5) > 0 and w not in seen:
                seen.add(w)
                ws.append(w)
        return ws

    # Score every named neuron's replicated words on C4, against all controls scored on (a) a paired named neuron's words
    # (clean test of whether the named words themselves fire on unseen C4) and (b) controls' own rep_words.
    totC4 = collections.Counter()
    for c in run.controls:
        totC4.update(cnt_C4[c])

    def score_on_C4(u: Tuple[str, int], ws: Sequence[int]) -> float:
        loo = u[0] == "random"
        den = (C - 1) if loo else C
        s = 0.0
        for w in ws:
            c = cnt_C4[u].get(w, 0)
            mu = (totC4.get(w, 0) - (c if loo else 0)) / den
            s += (c - mu) / math.sqrt(mu + 0.5)
        return float(s)

    named_words = {u: rep_words_of(u) for u in named}
    z_c4_named = {u: score_on_C4(u, named_words[u]) for u in named}
    # Controls paired cycling over named neurons' word sets (fair null for fixed word sets on unseen C4)
    z_c4_ctl_paired = [score_on_C4(c, named_words[named[i % len(named)]]) for i, c in enumerate(run.controls)]
    # Precompute inverted postings over controls' C4 word counts for O(nnz) exact LOO scoring
    postings_c4: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for ci, cu_id in enumerate(run.controls):
        for w, c in cnt_C4[cu_id].items():
            postings_c4[w].append((ci, c))
    base_zero_c4: Dict[int, float] = {}
    for w in set().union(*(named_words[u] for u in named)):
        mu0 = totC4.get(w, 0) / (C - 1)
        base_zero_c4[w] = -mu0 / math.sqrt(mu0 + 0.5)

    p_c4 = []
    for u in named:
        ws = named_words[u]
        if not ws:
            p_c4.append(1.0)
            continue
        tot_w = 0.0
        ctl_scores = np.zeros(C, dtype=np.float64)
        for w in ws:
            bz = base_zero_c4[w]
            tot_w += bz
            tw = totC4.get(w, 0)
            for ci, c in postings_c4.get(w, ()):
                mu_loo = (tw - c) / (C - 1)
                ctl_scores[ci] += (c - mu_loo) / math.sqrt(mu_loo + 0.5) - bz
        ctl_scores += tot_w
        ge = int(np.sum(ctl_scores >= z_c4_named[u]))
        p_c4.append((1.0 + ge) / (C + 1.0))

    vals_n = [z_c4_named[u] for u in named]
    return {
        "fit_named": len(named),
        "fit_above_max_ctl": sel["above_max_ctl"],
        "c4_auc_vs_paired_ctl": round(auc(vals_n, z_c4_ctl_paired)[0], 4),
        "frac_z_gt_0": round(float(np.mean([v > 0 for v in vals_n])), 4),
        "frac_z_gt_0_ctl": round(float(np.mean([v > 0 for v in z_c4_ctl_paired])), 4),
        "frac_p05_on_C4": round(float(np.mean([p <= 0.05 for p in p_c4])), 4),
        "median_z_C4": round(float(np.median(vals_n)), 3),
    }


def analyze_file(path: Path, is_trial: bool) -> Dict[str, Any]:
    t0 = time.time()
    meta = json.loads(cz.meta_path_of(path).read_text(encoding="utf-8"))
    prompts = [q for q in (meta.get("prompts") or {}) if any(p == q for p, _ in meta["design"])]
    doses = sorted({abs(float(d)) for _, d in meta["design"]})
    half_n = len(prompts) // 2
    A_set, B_set = set(prompts[:half_n]), set(prompts[half_n:])
    subsets: Dict[str, Callable[[str, float], bool]] = {
        "all": lambda p, d: p in A_set or p in B_set,
        "A": lambda p, d: p in A_set,
        "B": lambda p, d: p in B_set,
    }
    if len(prompts) == 12:
        A4, B4, C4 = set(prompts[:4]), set(prompts[4:8]), set(prompts[8:12])
        subsets.update({
            "AB8": lambda p, d: p in A4 or p in B4,
            "A4": lambda p, d: p in A4,
            "B4": lambda p, d: p in B4,
            "C4": lambda p, d: p in C4,
        })
    if is_trial and len(doses) == 2:
        d_lo = doses[0]
        subsets.update({
            "lo_all": lambda p, d, dl=d_lo: abs(abs(d) - dl) < 1e-6 and (p in A_set or p in B_set),
            "lo_A": lambda p, d, dl=d_lo: abs(abs(d) - dl) < 1e-6 and p in A_set,
            "lo_B": lambda p, d, dl=d_lo: abs(abs(d) - dl) < 1e-6 and p in B_set,
        })

    run = StreamedRun(path, subsets)
    out: Dict[str, Any] = {
        "file": path.name,
        "layer": run.layer,
        "design": run.design,
        "neurons": len(run.neurons),
        "controls": len(run.controls),
        "doses": run.doses,
    }
    for tag, strip in (("raw", False), ("no_thought", True)):
        entry = eval_pair(run, "A", "B", "all", strip_thought=strip, run_null_splits=50, seed=run.layer)
        if len(prompts) == 12:
            entry["oos_3way"] = out_of_sample_3way(run, strip_thought=strip)
        out[tag] = entry

    if is_trial and len(doses) == 2:
        out["lower_dose_no_thought"] = eval_pair(run, "lo_A", "lo_B", "lo_all", strip_thought=True, run_null_splits=20, seed=100 + run.layer)

    del run
    gc.collect()
    print(f"  -> {path.name} done in {time.time()-t0:.1f}s", flush=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = ai.atlas_dir()
    jobs = [
        (data_dir / "census/trials/things_walk/L55-match_things-trial.jsonl", True),
        (data_dir / "census/trials/things_walk/L54-match_things-trial.jsonl", True),
        (data_dir / "census/trials/things_walk/L53-match_things-trial.jsonl", True),
        (data_dir / "census/trials/things_walk/L29-match_things-trial.jsonl", True),
        (data_dir / "census/runs/L30-match_things.jsonl", False),
        (data_dir / "census/runs/L34-match_ceil.jsonl", False),
        (data_dir / "census/runs/L53-match_things.jsonl", False),
        (data_dir / "census/runs/L54-match_things.jsonl", False),
    ]
    results = []
    for p, is_t in jobs:
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        res = analyze_file(p, is_t)
        results.append(res)
        # Incremental save
        (OUT_DIR / "design2_calibration.json").write_text(json.dumps(results, indent=1) + "\n")

    lines = [
        "Design 2 CPU Calibration (raw vs no_thought, T_dec = min(rz(rep_AB_res), rz(rep_BA_res)), FDP <= 0.05)",
        "-" * 118,
        f"{'Run':<28} {'Var':<10} {'mass_c':>6} {'repAB':>6} {'repBA':>6} {'repABr':>6} {'repBAr':>6} "
        f"{'T_auc':>6} {'rlib':>5} {'FDP05':>6} {'FDP10':>6} {'Null05':>7} {'C4_p05':>7}",
    ]
    for r in results:
        for var in ("raw", "no_thought"):
            e = r[var]
            ns = e.get("null_splits", {})
            oos = e.get("oos_3way", {})
            c4 = f"{oos['frac_p05_on_C4']:.3f}" if oos.get("frac_p05_on_C4") is not None else "-"
            lines.append(
                f"{r['file'][:27]:<28} {var:<10} {e['mass_c_auc']:6.3f} {e['rep_AB_auc']:6.3f} {e['rep_BA_auc']:6.3f} "
                f"{e['rep_AB_res_auc']:6.3f} {e['rep_BA_res_auc']:6.3f} {e['T_dec_auc']:6.3f} "
                f"{e['readlib_named']:5d} {e['fdp05']['count']:6d} {e['fdp10']['count']:6d} "
                f"{ns.get('fdp05_false_mean', -1):7.2f} {c4:>7}"
            )
        if "lower_dose_no_thought" in r:
            e = r["lower_dose_no_thought"]
            ns = e.get("null_splits", {})
            lines.append(
                f"{r['file'][:27]:<28} {'lo_noth':<10} {e['mass_c_auc']:6.3f} {e['rep_AB_auc']:6.3f} {e['rep_BA_auc']:6.3f} "
                f"{e['rep_AB_res_auc']:6.3f} {e['rep_BA_res_auc']:6.3f} {e['T_dec_auc']:6.3f} "
                f"{e['readlib_named']:5d} {e['fdp05']['count']:6d} {e['fdp10']['count']:6d} "
                f"{ns.get('fdp05_false_mean', -1):7.2f} {'-':>7}"
            )
    txt = "\n".join(lines) + "\n"
    (OUT_DIR / "design2_calibration.txt").write_text(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
