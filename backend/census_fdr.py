"""Design 2 per-neuron empirical-FDR reader (`T_dec` + `no_thought`) for decoding census runs.

Streams `.jsonl` census files with bounded memory, filters out `"thought"`-glued tokens,
fits 6 survival covariates on controls alone, scores residualized cross-half replication
`T_dec = min(rz(rep_AB_res), rz(rep_BA_res))`, applies step-up empirical FDP <= 0.05 / 0.10,
checks N3 covered-neuron subset, runs 50 half-control null splits, and validates on unseen
4-question splits (12-question runs) via an inverted posting index.
"""
from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats

from backend import census as cz

COVS: Tuple[str, ...] = ("lnw", "d4", "cut", "lp", "dep", "lost")


def auc(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a_arr, b_arr, alternative="greater")
    return float(u.statistic / (len(a_arr) * len(b_arr))), float(u.pvalue)


def emp_p_map(d: Dict[Tuple[str, int], float]) -> Dict[Tuple[str, int], float]:
    ctl = np.sort([v for u, v in d.items() if u[0] == "random"])
    C = len(ctl)
    out: Dict[Tuple[str, int], float] = {}
    for u, v in d.items():
        ge = C - int(np.searchsorted(ctl, v, side="left"))
        out[u] = (1.0 + ge) / (C + 1.0) if u[0] == "neuron" else (ge / C if C > 0 else 1.0)
    return out


def bh_map(ps: Dict[Tuple[str, int], float]) -> Dict[Tuple[str, int], float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    n = len(items)
    q, out = 1.0, {}
    for i in range(n - 1, -1, -1):
        q = min(q, items[i][1] * n / (i + 1))
        out[items[i][0]] = q
    return out


def residualise(
    stat: Dict[Tuple[str, int], float],
    cov: Dict[Tuple[str, int], Dict[str, float]],
    ctl_pool: Optional[Sequence[Tuple[str, int]]] = None,
) -> Dict[Tuple[str, int], float]:
    units = [u for u in stat if u in cov]
    ctl = list(ctl_pool) if ctl_pool is not None else [u for u in units if u[0] == "random"]
    X = np.array([[1.0] + [cov[u][c] for c in COVS] for u in ctl], dtype=np.float64)
    y = np.array([stat[u] for u in ctl], dtype=np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {u: float(stat[u] - np.dot([1.0] + [cov[u][c] for c in COVS], beta)) for u in units}


def robust_z(
    stat: Dict[Tuple[str, int], float],
    ctl_pool: Optional[Sequence[Tuple[str, int]]] = None,
) -> Dict[Tuple[str, int], float]:
    ctl = np.array(
        [stat[u] for u in (ctl_pool if ctl_pool is not None else [u for u in stat if u[0] == "random"])],
        dtype=np.float64,
    )
    med = float(np.median(ctl))
    mad = float(np.median(np.abs(ctl - med)))
    std = float(np.std(ctl))
    scale = max(1.4826 * mad, 0.25 * std, 1e-6)
    return {u: float((v - med) / scale) for u, v in stat.items()}


def z_top(
    c: Dict[int, int],
    tot: Dict[int, int],
    den: int,
    loo: bool,
    k: int = 5,
    min_count: int = 2,
) -> List[Tuple[float, int]]:
    zs: List[Tuple[float, int]] = []
    for w, cnt in c.items():
        if cnt < min_count:
            continue
        mu = (tot.get(w, 0) - (cnt if loo else 0)) / den
        zs.append(((cnt - mu) / math.sqrt(mu + 0.5), w))
    zs.sort(reverse=True)
    return zs[:k]


def mass_from_counts(
    cnt: Dict[Tuple[str, int], collections.Counter],
    ctl_pool: Optional[Sequence[Tuple[str, int]]] = None,
    k: int = 5,
) -> Dict[Tuple[str, int], Tuple[float, List[Tuple[float, int]]]]:
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


def z_of(
    cnt: Dict[Tuple[str, int], collections.Counter],
    words_of: Dict[Tuple[str, int], List[int]],
    ctl_pool: Optional[Sequence[Tuple[str, int]]] = None,
) -> Dict[Tuple[str, int], float]:
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


def emp_fdp_select(
    scores: Dict[Tuple[str, int], float],
    positive_ok: Dict[Tuple[str, int], bool],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Step-up empirical FDP: FDP(t) = (N/(C+1)) * (1 + #{c >= t}) / max(1, #{n >= t})."""
    neu_units = [u for u in scores if u[0] == "neuron" and positive_ok[u]]
    neu_vals = np.array([scores[u] for u in neu_units], dtype=np.float64)
    N = sum(1 for u in scores if u[0] == "neuron")
    ctl_vals = np.sort(np.array([scores[u] for u in scores if u[0] == "random"], dtype=np.float64))
    C = len(ctl_vals)
    if len(neu_vals) == 0 or C == 0 or N == 0:
        return {"count": 0, "threshold": None, "fdp": None, "named": [], "above_max_ctl": 0}
    max_ctl = float(ctl_vals[-1])
    above_max = int(np.sum(neu_vals > max_ctl))
    order = np.argsort(-neu_vals)
    sorted_vals = neu_vals[order]
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
        fdp = (N / (C + 1.0)) * (1.0 + c_ge) / max(1, n_ge)
        if fdp <= alpha:
            best_idx = j
            best_fdp = fdp
        i = j + 1
    if best_idx < 0:
        return {"count": 0, "threshold": None, "fdp": None, "named": [], "above_max_ctl": above_max}
    thresh = float(sorted_vals[best_idx])
    named = [neu_units[k] for k in order[: best_idx + 1]]
    return {
        "count": len(named),
        "threshold": round(thresh, 4),
        "fdp": round(best_fdp, 4),
        "named": named,
        "above_max_ctl": above_max,
    }


class StreamedRun:
    """Memory-bounded representation of a census JSONL run over prompt/dose subsets."""

    def __init__(self, path: Path, subsets: Dict[str, Callable[[str, float], bool]], max_new: int = 48):
        t0 = time.time()
        self.path = Path(path)
        meta_path = cz.meta_path_of(self.path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.layer = int(meta["layer"])
        self.design = str(meta.get("design_name"))
        self.prompts = [q for q in (meta.get("prompts") or {}) if any(p == q for p, _ in meta["design"])]
        self.doses = sorted({abs(float(d)) for _, d in meta["design"]})

        self.vocab: List[str] = []
        self._wid: Dict[str, int] = {}
        self.thought_wids: Set[int] = set()
        bases: Dict[str, Dict[str, Any]] = {}

        # Pass 1: full scan for interleaved baseline rows (~2s on 5.6 GB)
        with open(self.path, "rb") as f:
            for line in f:
                if b'"baseline"' in line and b'"kind": "baseline"' in line:
                    r = json.loads(line)
                    if r.get("kind") == "baseline" and r["prompt"] not in bases:
                        ids = [int(x) for x in (r.get("token_ids") or [])][:max_new]
                        ws = self._to_wids(cz.words(r["text"]))
                        bases[r["prompt"]] = {
                            "wids": set(ws),
                            "ids": ids,
                            "d4": cz.distinct4(ids),
                        }

        # Pass 2: stream pushed rows into subsets
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
                ids = (r.get("token_ids") or [])[:max_new]
                lp = (r.get("logprob") or [])[:max_new]
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
        self.elapsed_load_s = round(time.time() - t0, 2)

    def _to_wids(self, ws: Iterable[str]) -> List[int]:
        out: List[int] = []
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


def _score_subset_dict(
    cnt_all: Dict[Tuple[str, int], collections.Counter],
    cnt_A: Dict[Tuple[str, int], collections.Counter],
    cnt_B: Dict[Tuple[str, int], collections.Counter],
    cov_all: Dict[Tuple[str, int], Dict[str, float]],
    units_keep: Optional[Set[Tuple[str, int]]] = None,
) -> Dict[str, Any]:
    if units_keep is not None:
        cnt_all = {u: c for u, c in cnt_all.items() if u[0] == "random" or u in units_keep}
        cnt_A = {u: c for u, c in cnt_A.items() if u[0] == "random" or u in units_keep}
        cnt_B = {u: c for u, c in cnt_B.items() if u[0] == "random" or u in units_keep}
        cov_all = {u: c for u, c in cov_all.items() if u[0] == "random" or u in units_keep}

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

    p_all = emp_p_map(mass_c)
    p_r = emp_p_map(mass_c_res)
    p_AB = emp_p_map(rep_AB)
    p_BA = emp_p_map(rep_BA)
    q_all = bh_map({u: v for u, v in p_all.items() if u[0] == "neuron"})
    readlib_named = [u for u in q_all if q_all[u] <= 0.05 and p_AB[u] <= 0.05 and p_BA[u] <= 0.05 and p_r[u] <= 0.05]

    rz_AB = robust_z(rep_AB_res)
    rz_BA = robust_z(rep_BA_res)
    T_dec = {u: min(rz_AB[u], rz_BA[u]) for u in rep_AB_res}
    pos_ok = {u: (rep_AB_res[u] > 0.0 and rep_BA_res[u] > 0.0) for u in T_dec}
    sel05 = emp_fdp_select(T_dec, pos_ok, 0.05)
    sel10 = emp_fdp_select(T_dec, pos_ok, 0.10)

    neu = [u for u in T_dec if u[0] == "neuron"]
    ctl = [u for u in T_dec if u[0] == "random"]
    return {
        "mass_c_auc": round(auc([mass_c[u] for u in neu], [mass_c[u] for u in ctl])[0], 4),
        "mass_c_res_auc": round(auc([mass_c_res[u] for u in neu], [mass_c_res[u] for u in ctl])[0], 4),
        "rep_AB_auc": round(auc([rep_AB[u] for u in neu], [rep_AB[u] for u in ctl])[0], 4),
        "rep_BA_auc": round(auc([rep_BA[u] for u in neu], [rep_BA[u] for u in ctl])[0], 4),
        "rep_AB_res_auc": round(auc([rep_AB_res[u] for u in neu], [rep_AB_res[u] for u in ctl])[0], 4),
        "rep_BA_res_auc": round(auc([rep_BA_res[u] for u in neu], [rep_BA_res[u] for u in ctl])[0], 4),
        "rep_res_auc": round(auc([rep_res[u] for u in neu], [rep_res[u] for u in ctl])[0], 4),
        "T_dec_auc": round(auc([T_dec[u] for u in neu], [T_dec[u] for u in ctl])[0], 4),
        "readlib_q05": int(sum(1 for v in q_all.values() if v <= 0.05)),
        "readlib_named": len(readlib_named),
        "fdp05": {k: v for k, v in sel05.items() if k != "named"},
        "fdp10": {k: v for k, v in sel10.items() if k != "named"},
        "_named05": sel05["named"],
        "_named10": sel10["named"],
        "_T_dec": T_dec,
        "_rep_AB_res": rep_AB_res,
        "_rep_BA_res": rep_BA_res,
        "_mA": mA,
        "_mB": mB,
    }


def eval_pair(
    run: StreamedRun,
    half_a: str,
    half_b: str,
    all_sub: str,
    strip_thought: bool = True,
    run_null_splits: int = 50,
    seed: int = 0,
) -> Dict[str, Any]:
    cnt_all = run.filtered_counts(all_sub, strip_thought)
    cnt_A = run.filtered_counts(half_a, strip_thought)
    cnt_B = run.filtered_counts(half_b, strip_thought)
    cov_all = run.sub_cov[all_sub]

    n1 = _score_subset_dict(cnt_all, cnt_A, cnt_B, cov_all, units_keep=None)

    # N3: neurons inside controls' [min, max] range on all 6 covariates
    lo = {k: min(cov_all[u][k] for u in run.controls) for k in COVS}
    hi = {k: max(cov_all[u][k] for u in run.controls) for k in COVS}
    covered = {u for u in run.neurons if not any(cov_all[u][k] < lo[k] or cov_all[u][k] > hi[k] for k in COVS)}
    n3_raw = _score_subset_dict(cnt_all, cnt_A, cnt_B, cov_all, units_keep=covered) if covered else None
    n3_cov_named05 = sum(1 for u in n1["_named05"] if u in covered)

    # 50 random half-control null splits
    null_res = None
    if run_null_splits > 0 and len(run.controls) >= 32:
        rng = np.random.default_rng(int(seed))
        ctls = list(run.controls)
        null_aucs, null_fd05 = [], []
        half = len(ctls) // 2
        for _ in range(int(run_null_splits)):
            perm = rng.permutation(len(ctls))
            fake_set = {ctls[i] for i in perm[:half]}
            subA = {(("neuron", u[1]) if u in fake_set else u): cnt_A[u] for u in ctls}
            subB = {(("neuron", u[1]) if u in fake_set else u): cnt_B[u] for u in ctls}
            subCov = {(("neuron", u[1]) if u in fake_set else u): cov_all[u] for u in ctls}
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
            null_fd05.append(emp_fdp_select(Tn, pok, 0.05)["count"])
        null_res = {
            "n": int(run_null_splits),
            "T_dec_auc_mean": round(float(np.mean(null_aucs)), 4),
            "T_dec_auc_max": round(float(np.max(null_aucs)), 4),
            "fdp05_false_mean": round(float(np.mean(null_fd05)), 2),
            "fdp05_false_max": int(np.max(null_fd05)),
        }

    # Build replicated words list for individually named (FDP <= 0.05) neurons
    totA, totB = collections.Counter(), collections.Counter()
    for c in run.controls:
        totA.update(cnt_A[c])
        totB.update(cnt_B[c])
    C = len(run.controls)
    named_records: List[Dict[str, Any]] = []
    for u in n1["_named05"]:
        words_AB: List[Tuple[str, float]] = []
        for _, w in n1["_mA"][u][1]:
            c = cnt_B[u].get(w, 0)
            mu = totB.get(w, 0) / max(C, 1)
            z = (c - mu) / math.sqrt(mu + 0.5)
            if z > 0:
                words_AB.append((run.vocab[w], round(float(z), 2)))
        words_BA: List[Tuple[str, float]] = []
        for _, w in n1["_mB"][u][1]:
            c = cnt_A[u].get(w, 0)
            mu = totA.get(w, 0) / max(C, 1)
            z = (c - mu) / math.sqrt(mu + 0.5)
            if z > 0:
                words_BA.append((run.vocab[w], round(float(z), 2)))
        named_records.append(
            {
                "neuron": int(u[1]),
                "T_dec": round(float(n1["_T_dec"][u]), 4),
                "rep_AB_res": round(float(n1["_rep_AB_res"][u]), 4),
                "rep_BA_res": round(float(n1["_rep_BA_res"][u]), 4),
                "covered": u in covered,
                "rep_words_AB": words_AB,
                "rep_words_BA": words_BA,
            }
        )

    out = {k: v for k, v in n1.items() if not k.startswith("_")}
    out["n3_covered"] = {
        "covered": len(covered),
        "neurons": len(run.neurons),
        "share_covered": round(len(covered) / max(len(run.neurons), 1), 4),
        "n1_named_inside_covered": n3_cov_named05,
        **({k: v for k, v in n3_raw.items() if not k.startswith("_")} if n3_raw else {}),
    }
    out["null_splits"] = null_res
    out["_named_records"] = named_records
    return out


def out_of_sample_3way(run: StreamedRun, strip_thought: bool = True) -> Dict[str, Any]:
    """Fit T_dec on A4=q[0:4] vs B4=q[4:8], test replicated words on unseen C4=q[8:12] via inverted postings."""
    cnt_A4 = run.filtered_counts("A4", strip_thought)
    cnt_B4 = run.filtered_counts("B4", strip_thought)
    cnt_C4 = run.filtered_counts("C4", strip_thought)
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
        return {"fit_named": 0, "fit_above_max_ctl": sel["above_max_ctl"], "c4_auc": None, "frac_p05_on_C4": None}

    totA4, totB4, totC4 = collections.Counter(), collections.Counter(), collections.Counter()
    for c in run.controls:
        totA4.update(cnt_A4[c])
        totB4.update(cnt_B4[c])
        totC4.update(cnt_C4[c])
    C = len(run.controls)

    def rep_words_of(u: Tuple[str, int]) -> List[int]:
        loo = u[0] == "random"
        den = max((C - 1) if loo else C, 1)
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

    def score_on_C4(u: Tuple[str, int], ws: Sequence[int]) -> float:
        loo = u[0] == "random"
        den = max((C - 1) if loo else C, 1)
        s = 0.0
        for w in ws:
            c = cnt_C4[u].get(w, 0)
            mu = (totC4.get(w, 0) - (c if loo else 0)) / den
            s += (c - mu) / math.sqrt(mu + 0.5)
        return float(s)

    named_words = {u: rep_words_of(u) for u in named}
    z_c4_named = {u: score_on_C4(u, named_words[u]) for u in named}
    z_c4_ctl_paired = [score_on_C4(c, named_words[named[i % len(named)]]) for i, c in enumerate(run.controls)]

    postings_c4: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
    for ci, cu_id in enumerate(run.controls):
        for w, c in cnt_C4[cu_id].items():
            postings_c4[w].append((ci, c))
    base_zero_c4: Dict[int, float] = {}
    for w in set().union(*(named_words[u] for u in named)):
        mu0 = totC4.get(w, 0) / max(C - 1, 1)
        base_zero_c4[w] = -mu0 / math.sqrt(mu0 + 0.5)

    p_c4: List[float] = []
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
                mu_loo = (tw - c) / max(C - 1, 1)
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
        "frac_p05_on_C4": round(float(np.mean([p <= 0.05 for p in p_c4])), 4),
        "median_z_C4": round(float(np.median(vals_n)), 3),
    }
