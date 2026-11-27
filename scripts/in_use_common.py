"""Shared CPU/GPU helpers for exploratory on-manifold ('in-use') kill-pilots.

Provides strict <eos> document-half partitioning over passages.d, vectorized
face eligibility auditing, like-for-like 'columns' control construction at
exact post_feedforward_layernorm scale, leave-one-out control centering, and
probability-weighted directional carry statistics. Never stores corpus text.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from backend import atlas_index as ai
from backend import atlas_ops as ao
from backend import input_census as ic
from backend.gap_stats import mannwhitney_greater, wilcoxon_greater


def load_passages_and_doc_halves(
    data_dir: Path,
    *,
    eos_id: int = 1,
    seed: int = 20261124,
) -> Dict[str, Any]:
    """Load passages.d, act_scale.npz, vocab.json, and strict <eos> document halves."""
    idx_dir = Path(data_dir) / "index"
    P = ai.load_folder(idx_dir / "passages.d/meta.json")
    act_scale = np.load(idx_dir / "act_scale.npz")
    vocab = json.loads((idx_dir / "vocab.json").read_text(encoding="utf-8"))["vocab"]
    windows = np.asarray(P["windows"])
    unigram = np.asarray(P["unigram"])
    bos_id = int(windows[0, 0]) if windows.shape[0] > 0 and windows[0, 0] >= 0 else 2
    doc_id, half_id, n_docs = ic.build_doc_halves(windows, eos_id=eos_id, seed=seed)
    pure_h0_mask = np.all(half_id == 0, axis=1) & (np.sum(windows >= 0, axis=1) >= 8)
    pure_h1_mask = np.all(half_id == 1, axis=1) & (np.sum(windows >= 0, axis=1) >= 8)
    return {
        "P": P,
        "act_scale": act_scale,
        "vocab": vocab,
        "windows": windows,
        "unigram": unigram,
        "eos_id": int(eos_id),
        "bos_id": int(bos_id),
        "doc_id": doc_id,
        "half_id": half_id,
        "n_docs": int(n_docs),
        "pure_h0_mask": pure_h0_mask,
        "pure_h1_mask": pure_h1_mask,
    }


def layer_face_counts_vectorized(
    P: Dict[str, Any],
    windows: np.ndarray,
    act_scale: Any,
    half_id: np.ndarray,
    pure_h0_mask: np.ndarray,
    pure_h1_mask: np.ndarray,
    layer: int,
    *,
    eos_id: int = 1,
    bos_id: int = 2,
    min_rel: float = 0.5,
    min_extreme: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Vectorized count of unique-window events per (face, neuron) across half0/half1."""
    W_tot, seq_len = windows.shape
    N_ff = int(P["pos_win"].shape[1])
    out: Dict[str, np.ndarray] = {}
    max_ext = np.maximum(
        np.abs(np.asarray(act_scale["pos"][layer], dtype=np.float32)),
        np.abs(np.asarray(act_scale["neg"][layer], dtype=np.float32)),
    )
    out["nondead_mask"] = max_ext >= float(min_extreme)

    for face, sgn in (("pos", 1.0), ("neg", -1.0)):
        ext = np.abs(np.asarray(act_scale[face][layer], dtype=np.float32))
        ws_parts = [np.asarray(P[f"{face}_win"][layer], dtype=np.int32)]
        ats_parts = [np.asarray(P[f"{face}_at"][layer], dtype=np.int32)]
        acts_parts = [np.asarray(P[f"{face}_act"][layer], dtype=np.float32)]
        if f"{face}_smp_win" in P:
            ws_parts.append(np.asarray(P[f"{face}_smp_win"][layer], dtype=np.int32))
            ats_parts.append(np.asarray(P[f"{face}_smp_at"][layer], dtype=np.int32))
            acts_parts.append(np.asarray(P[f"{face}_smp_act"][layer], dtype=np.float32))
        ws = np.concatenate(ws_parts, axis=1)
        ats = np.concatenate(ats_parts, axis=1)
        acts = np.concatenate(acts_parts, axis=1)
        S = int(ws.shape[1])

        valid = (ext[:, None] >= float(min_extreme)) & (ws >= 0) & (ws < W_tot) & (ats >= 1) & (ats < seq_len)
        ws_c = np.where(valid, ws, 0)
        ats_c = np.where(valid, ats, 1)
        tok0 = windows[ws_c, ats_c]
        valid &= (tok0 >= 0) & (tok0 != int(eos_id)) & (tok0 != int(bos_id))
        valid &= (sgn * acts) >= float(min_rel) * np.maximum(ext[:, None], 1e-6)

        # Deduplicate repeated windows across (top, smp), keeping the first valid slot.
        tril = np.tril(np.ones((S, S), dtype=bool), k=-1)
        eq = ws[:, :, None] == ws[:, None, :]
        dup = np.any(eq & valid[:, None, :] & tril[None, :, :], axis=-1)
        uval = valid & (~dup)

        h0_p = np.sum(pure_h0_mask[ws_c] & uval, axis=1)
        h1_p = np.sum(pure_h1_mask[ws_c] & uval, axis=1)
        hat = half_id[ws_c, ats_c]
        h0_d = np.sum((hat == 0) & uval, axis=1)
        h1_d = np.sum((hat == 1) & uval, axis=1)

        out[f"{face}_pure0"] = h0_p
        out[f"{face}_pure1"] = h1_p
        out[f"{face}_doc0"] = h0_d
        out[f"{face}_doc1"] = h1_d
        out[f"{face}_pure44"] = (h0_p >= 4) & (h1_p >= 4)
        out[f"{face}_pure66"] = (h0_p >= 6) & (h1_p >= 6)
        out[f"{face}_doc44"] = (h0_d >= 4) & (h1_d >= 4)

    out["neu_pure44"] = out["pos_pure44"] | out["neg_pure44"]
    out["neu_pure66"] = out["pos_pure66"] | out["neg_pure66"]
    out["neu_doc44"] = out["pos_doc44"] | out["neg_doc44"]
    out["face_pure44_count"] = int(out["pos_pure44"].sum() + out["neg_pure44"].sum())
    assert out["neu_pure44"].shape == (N_ff,)
    return out


def collect_face_events(
    P: Dict[str, Any],
    windows: np.ndarray,
    act_scale: Any,
    pure_h0_mask: np.ndarray,
    pure_h1_mask: np.ndarray,
    doc_id: np.ndarray,
    half_id: np.ndarray,
    layer: int,
    n: int,
    face: str,
    *,
    eos_id: int = 1,
    bos_id: int = 2,
    min_rel: float = 0.5,
    min_extreme: float = 0.1,
) -> Dict[str, Any]:
    """Collect deduplicated events for one (layer, n, face), partitioned into disjoint doc halves."""
    assert face in ("pos", "neg")
    sgn = 1.0 if face == "pos" else -1.0
    ext = float(abs(act_scale[face][layer, n]))
    if ext < float(min_extreme):
        return {"extreme": ext, "pure_h0": [], "pure_h1": [], "doc_h0": [], "doc_h1": [], "own_windows": set()}

    W_tot, seq_len = windows.shape
    raw: List[Tuple[int, int, float]] = []
    for wi, ai_, ac in zip(P[f"{face}_win"][layer, n], P[f"{face}_at"][layer, n], P[f"{face}_act"][layer, n]):
        raw.append((int(wi), int(ai_), float(ac)))
    if f"{face}_smp_win" in P:
        smp = [
            (int(wi), int(ai_), float(ac))
            for wi, ai_, ac in zip(
                P[f"{face}_smp_win"][layer, n], P[f"{face}_smp_at"][layer, n], P[f"{face}_smp_act"][layer, n]
            )
        ]
        smp.sort(key=lambda x: -(sgn * x[2]))
        raw.extend(smp)

    seen_win: Set[int] = set()
    own_windows: Set[int] = set()
    pure_h0: List[Dict[str, Any]] = []
    pure_h1: List[Dict[str, Any]] = []
    doc_h0: List[Dict[str, Any]] = []
    doc_h1: List[Dict[str, Any]] = []

    for wi, ai_, ac in raw:
        if 0 <= wi < W_tot:
            own_windows.add(wi)
        if wi in seen_win or not (0 <= wi < W_tot) or not (1 <= ai_ < seq_len):
            continue
        tok = int(windows[wi, ai_])
        if tok < 0 or tok in (int(eos_id), int(bos_id)):
            continue
        rel = (sgn * ac) / max(ext, 1e-6)
        if rel < float(min_rel):
            continue
        seen_win.add(wi)
        did = int(doc_id[wi, ai_])
        ev = {"win": wi, "at": ai_, "act": ac, "rel": rel, "doc_id": did, "tok0": tok}
        if bool(pure_h0_mask[wi]):
            pure_h0.append(ev)
        elif bool(pure_h1_mask[wi]):
            pure_h1.append(ev)
        if int(half_id[wi, ai_]) == 0:
            doc_h0.append(ev)
        else:
            doc_h1.append(ev)

    d0_p = {e["doc_id"] for e in pure_h0}
    d1_p = {e["doc_id"] for e in pure_h1}
    d0_d = {e["doc_id"] for e in doc_h0}
    d1_d = {e["doc_id"] for e in doc_h1}
    if d0_p & d1_p:
        raise AssertionError(f"Document leakage in pure halves for L{layer} N{n} {face}: {d0_p & d1_p}")
    if d0_d & d1_d:
        raise AssertionError(f"Document leakage in doc halves for L{layer} N{n} {face}: {d0_d & d1_d}")

    return {
        "extreme": ext,
        "pure_h0": pure_h0,
        "pure_h1": pure_h1,
        "doc_h0": doc_h0,
        "doc_h1": doc_h1,
        "own_windows": own_windows,
    }


def build_event_sequence(
    windows: np.ndarray,
    w: int,
    at: int,
    *,
    bos_id: int = 2,
    pad_tail: int = 4,
    doc_id: Optional[np.ndarray] = None,
    single_doc_slice: bool = False,
) -> Tuple[List[int], int]:
    """Build [bos_id] + tokens up to min(seq, at + 1 + pad_tail), matching passages.d context."""
    row = windows[int(w)]
    seq_len = int(row.shape[0])
    en_pos = min(seq_len, int(at) + 1 + int(pad_tail))
    st_pos = 1
    if single_doc_slice and doc_id is not None:
        target_doc = int(doc_id[int(w), int(at)])
        s_i = int(at)
        while s_i > 1 and int(row[s_i - 1]) >= 0 and int(doc_id[int(w), s_i - 1]) == target_doc:
            s_i -= 1
        e_i = int(at) + 1
        while e_i < en_pos and int(row[e_i]) >= 0 and int(doc_id[int(w), e_i]) == target_doc:
            e_i += 1
        st_pos, en_pos = s_i, e_i

    tail: List[int] = []
    pk = -1
    for pos in range(st_pos, en_pos):
        t = int(row[pos])
        if t < 0:
            continue
        if pos == int(at):
            pk = 1 + len(tail)
        tail.append(t)
    seq = [int(bos_id)] + tail
    if not (1 <= pk < len(seq)) or seq[pk] != int(row[int(at)]):
        raise AssertionError(f"Bad sequence peak reconstruction at w={w} at={at}: pk={pk}")
    return seq, pk


@torch.no_grad()
def build_columns_control_dirs(
    w_down: torch.Tensor,
    w_post: torch.Tensor,
    target_neurons: Sequence[int],
    seed: int,
    *,
    mix_k: int = 256,
) -> torch.Tensor:
    """For each target neuron n_i, build a random W_down-column mixture matched in ||w_post * v_c||_2.

    Matches both the layer's W_down column covariance structure ('columns' family) and
    the exact stream write norm ||w_post * W_down[:, n_i]||_2 on device.
    Returns [M, D] float32 tensor on w_down.device.
    """
    dev = w_down.device
    D, N_ff = int(w_down.shape[0]), int(w_down.shape[1])
    M = len(target_neurons)
    if M == 0:
        return torch.empty((0, D), dtype=torch.float32, device=dev)

    wp = w_post.to(device=dev, dtype=torch.float32)
    rng = np.random.default_rng(int(seed))
    k_use = min(int(mix_k), N_ff)
    idx_np = np.stack([rng.choice(N_ff, size=k_use, replace=False) for _ in range(M)], axis=0)
    idx_t = torch.from_numpy(idx_np).to(device=dev, dtype=torch.long)
    coeffs = torch.from_numpy(rng.standard_normal((M, k_use)).astype(np.float32)).to(device=dev)

    tgt_idx = torch.tensor([int(n) for n in target_neurons], dtype=torch.long, device=dev)
    tgt_cols = w_down.index_select(1, tgt_idx).float().T  # [M, D]
    tgt_post_norm = (tgt_cols * wp[None, :]).norm(dim=-1, keepdim=True).clamp(min=1e-30)

    out = torch.empty((M, D), dtype=torch.float32, device=dev)
    for i in range(M):
        cols_i = w_down.index_select(1, idx_t[i]).float()  # [D, k_use]
        v_raw = cols_i @ coeffs[i]  # [D]
        v_post_norm = (v_raw * wp).norm().clamp(min=1e-30)
        out[i] = v_raw * (tgt_post_norm[i, 0] / v_post_norm)
    return out


def loo_center(neu: np.ndarray, ctl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-out control-mean centering matching backend/forced_odd.py:_normalize_half_c."""
    neu_f = np.asarray(neu, dtype=np.float32)
    ctl_f = np.asarray(ctl, dtype=np.float32)
    C = int(ctl_f.shape[0])
    sum_c = ctl_f.sum(axis=0, keepdims=True)
    mu_all = sum_c / max(C, 1)
    mu_loo = (sum_c - ctl_f) / max(C - 1, 1) if C > 1 else np.zeros_like(ctl_f)
    return neu_f - mu_all, ctl_f - mu_loo


def emp_p_values(neu_scores: Sequence[float], ctl_scores: Sequence[float], *, greater: bool = True) -> Dict[str, Any]:
    """Empirical one-sided p-values, LOO control false-positive rate, BH q-values, and AUC."""
    ns = np.asarray(neu_scores, dtype=np.float64).ravel()
    cs = np.asarray(ctl_scores, dtype=np.float64).ravel()
    N, C = int(ns.size), int(cs.size)
    if N == 0 or C == 0:
        return {"auc": float("nan"), "p": float("nan"), "emp_p": np.array([]), "bh_q": np.array([])}

    sign = 1.0 if greater else -1.0
    ns_s = sign * ns
    cs_s = sign * cs
    cs_sorted = np.sort(cs_s)
    ge_neu = np.array([C - int(np.searchsorted(cs_sorted, v, side="left")) for v in ns_s], dtype=np.float64)
    emp_p = (1.0 + ge_neu) / (C + 1.0)

    # LOO empirical p among controls themselves to check null calibration
    ge_ctl_loo = np.array([int(np.sum(cs_s >= v)) - 1 for v in cs_s], dtype=np.float64)
    ctl_emp_p = (1.0 + np.maximum(ge_ctl_loo, 0.0)) / max(C, 1)

    order = np.argsort(emp_p, kind="mergesort")
    bh_q = np.empty(N, dtype=np.float64)
    q_running = 1.0
    for r in range(N - 1, -1, -1):
        idx = int(order[r])
        q_running = min(q_running, float(emp_p[idx]) * N / (r + 1))
        bh_q[idx] = min(1.0, q_running)

    auc, mw_p = mannwhitney_greater(ns_s, cs_s)
    paired_p = wilcoxon_greater(ns_s, cs_s) if N == C else None
    max_ctl = float(np.max(cs_s))
    return {
        "auc": round(float(auc), 4),
        "mw_p": float(f"{mw_p:.3e}"),
        "wilcoxon_p": None if paired_p is None else float(f"{paired_p:.3e}"),
        "paired_win_rate": None if N != C else round(float(np.mean(ns_s > cs_s) + 0.5 * np.mean(ns_s == cs_s)), 4),
        "neu_median": float(np.median(ns)),
        "ctl_median": float(np.median(cs)),
        "neu_p90": float(np.percentile(ns, 90)),
        "ctl_p90": float(np.percentile(cs, 90)),
        "n_emp_p05": int(np.sum(emp_p <= 0.05)),
        "frac_emp_p05": round(float(np.mean(emp_p <= 0.05)), 4),
        "ctl_loo_p05": int(np.sum(ctl_emp_p <= 0.05)),
        "frac_ctl_loo_p05": round(float(np.mean(ctl_emp_p <= 0.05)), 4),
        "n_above_max_ctl": int(np.sum(ns_s > max_ctl)),
        "n_bh_q05": int(np.sum(bh_q <= 0.05)),
        "n_bh_q10": int(np.sum(bh_q <= 0.10)),
        "emp_p": emp_p,
        "bh_q": bh_q,
    }


def score_directional_carry(
    fit_neu: np.ndarray,
    fit_ctl: np.ndarray,
    test_neu: np.ndarray,
    test_ctl: np.ndarray,
    *,
    top_k: int = 5,
    exclude_mask: Optional[np.ndarray] = None,
    allow_negative: bool = False,
) -> Dict[str, Any]:
    """LOO-centered cross-half directional carry: pick top-K on fit, score on test, and vice versa."""
    fn, fc = loo_center(fit_neu, fit_ctl)
    tn, tc = loo_center(test_neu, test_ctl)

    def _one_dir(src_u: np.ndarray, dst_u: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        M = src_u.shape[0]
        scores = np.zeros(M, dtype=np.float64)
        picks: List[List[int]] = []
        for i in range(M):
            row = src_u[i].copy()
            if exclude_mask is not None:
                row[exclude_mask] = 0.0 if allow_negative else -np.inf
            if allow_negative:
                idx = np.argsort(-np.abs(row))[: int(top_k)]
                sel = [int(j) for j in idx if row[j] != 0.0]
                if sel:
                    scores[i] = float(np.sum(np.sign(row[sel]) * dst_u[i, sel]))
            else:
                if exclude_mask is None:
                    pass
                cand = np.flatnonzero(row > 0.0)
                if cand.size > 0:
                    if cand.size > int(top_k):
                        top_sub = cand[np.argpartition(-row[cand], int(top_k) - 1)[: int(top_k)]]
                        sel_arr = top_sub[np.argsort(-row[top_sub])]
                    else:
                        sel_arr = cand[np.argsort(-row[cand])]
                    sel = [int(j) for j in sel_arr.tolist()]
                    scores[i] = float(np.sum(dst_u[i, sel]))
                else:
                    sel = []
            picks.append(sel)
        return scores, picks

    ft_neu, picks_ft_neu = _one_dir(fn, tn)
    ft_ctl, _ = _one_dir(fc, tc)
    tf_neu, picks_tf_neu = _one_dir(tn, fn)
    tf_ctl, _ = _one_dir(tc, fc)

    mean_neu = 0.5 * (ft_neu + tf_neu)
    mean_ctl = 0.5 * (ft_ctl + tf_ctl)
    min_neu = np.minimum(ft_neu, tf_neu)
    min_ctl = np.minimum(ft_ctl, tf_ctl)

    st_ft = emp_p_values(ft_neu, ft_ctl)
    st_tf = emp_p_values(tf_neu, tf_ctl)
    st_mean = emp_p_values(mean_neu, mean_ctl)
    st_min = emp_p_values(min_neu, min_ctl)
    return {
        "ft": st_ft,
        "tf": st_tf,
        "mean": st_mean,
        "min": st_min,
        "ft_neu": ft_neu,
        "tf_neu": tf_neu,
        "mean_neu": mean_neu,
        "picks_ft_neu": picks_ft_neu,
        "picks_tf_neu": picks_tf_neu,
        "fit_centered_neu": fn,
        "test_centered_neu": tn,
    }


def _strip_array_stats(st: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in st.items() if not isinstance(v, np.ndarray)}


def summarize_stats(st: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-serializable summary of emp_p_values() output."""
    out = _strip_array_stats(st)
    for k in ("neu_median", "ctl_median", "neu_p90", "ctl_p90"):
        if k in out and out[k] is not None and math.isfinite(out[k]):
            out[k] = round(float(out[k]), 6)
    return out
