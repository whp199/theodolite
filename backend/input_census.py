"""Exploratory split-half Input-Side Token-Trigger Table (`input_census`) for layers 0-29.

Uses precomputed 2M-token ``passages.d`` top-16 firing events partitioned by **document-disjoint**
halves (splitting at ``<eos>`` boundaries in ``windows.npy`` with deterministic PRNG half assignment),
strictly causal offsets ``(-1, 0)`` (never ``+1``; any ``+1`` next-piece replication is stored separately),
raw ``>=2+2`` counts before FDP, same-layer frequency-matched other-face null checks, and unseen-document
held-out prediction compared against frequency-matched wrong pieces. Never stores corpus passage text.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from backend import atlas_index as ai

# Triggers are strictly causal: -1 (previous token) and 0 (peak token). +1 is next-piece only.
CAUSAL_OFFSETS: Tuple[int, ...] = (-1, 0)


def decode_key(key: int) -> Tuple[int, int]:
    """Decode ``tok * 2 + (off + 1)`` back into ``(off, tok)`` for ``off in (-1, 0)``."""
    tok, rem = divmod(int(key), 2)
    return rem - 1, tok


def encode_key(off: int, tok: int) -> int:
    if off not in (-1, 0):
        raise ValueError(f"Causal trigger offset must be -1 or 0, got {off}")
    return int(tok) * 2 + (int(off) + 1)


def build_doc_halves(
    windows: np.ndarray,
    eos_id: int = 1,
    seed: int = 20261124,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Assign each ``(w, a)`` position a ``doc_id`` from ``<eos>`` boundaries and random doc-level half (0 or 1).

    When no ``eos_id`` appears in ``windows`` (e.g. small unit tests without ``<eos>``), falls back to
    one document per window, while still using deterministic random doc-half assignment (never parity).
    """
    W_tot, seq = windows.shape
    flat = windows.reshape(-1)
    eos_flags = (flat == int(eos_id))
    if np.any(eos_flags):
        doc_flat = np.zeros(flat.shape[0], dtype=np.int32)
        if flat.shape[0] > 1:
            doc_flat[1:] = np.cumsum(eos_flags[:-1], dtype=np.int32)
        doc_id = doc_flat.reshape(W_tot, seq)
        n_docs = int(doc_flat[-1]) + 1
    else:
        doc_id = np.broadcast_to(np.arange(W_tot, dtype=np.int32)[:, None], (W_tot, seq)).copy()
        n_docs = W_tot
    rng = np.random.default_rng(seed)
    doc_half = rng.integers(0, 2, size=max(n_docs, 1), dtype=np.int8)
    half_id = doc_half[doc_id]
    return doc_id, half_id, n_docs


def empirical_fdp_1d(
    neu_scores: np.ndarray,
    ctl_scores: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Step-up empirical FDP over non-negative neuron/face scores against ``ctl_scores``."""
    N = int(len(neu_scores))
    C = int(len(ctl_scores))
    if N == 0 or C == 0:
        return {"count": 0, "threshold": None, "fdp": None, "above_max_ctl": 0, "selected_mask": np.zeros(N, dtype=bool)}

    ctl_sorted = np.sort(ctl_scores.astype(np.float64))
    max_ctl = float(ctl_sorted[-1])
    pos_idx = np.nonzero(neu_scores > 0.0)[0]
    above_max = int(np.sum(neu_scores[pos_idx] > max_ctl)) if len(pos_idx) else 0
    if len(pos_idx) == 0:
        return {"count": 0, "threshold": None, "fdp": None, "above_max_ctl": 0, "selected_mask": np.zeros(N, dtype=bool)}

    cand_s = neu_scores[pos_idx]
    order = np.argsort(-cand_s)
    sorted_s = cand_s[order]
    best_j = -1
    best_fdp = 1.0
    m = len(sorted_s)
    i = 0
    while i < m:
        j = i
        t = sorted_s[i]
        while j + 1 < m and sorted_s[j + 1] == t:
            j += 1
        c_ge = C - int(np.searchsorted(ctl_sorted, t, side="left"))
        fdp = (N / (C + 1.0)) * (1.0 + c_ge) / (j + 1.0)
        if fdp <= alpha:
            best_j = j
            best_fdp = fdp
        i = j + 1

    sel = np.zeros(N, dtype=bool)
    if best_j < 0:
        return {"count": 0, "threshold": None, "fdp": None, "above_max_ctl": above_max, "selected_mask": sel}
    sel[pos_idx[order[: best_j + 1]]] = True
    return {
        "count": int(best_j + 1),
        "threshold": float(sorted_s[best_j]),
        "fdp": round(float(best_fdp), 4),
        "above_max_ctl": above_max,
        "selected_mask": sel,
    }


def _extract_tokens_for_events(
    windows: np.ndarray,
    win: np.ndarray,
    at: np.ndarray,
    eos_id: int = 1,
    bos_id: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract tokens at (-1, 0, +1) respecting window bounds and excluding BOS/EOS at -1/+1."""
    W_tot, seq = windows.shape
    valid = (win >= 0) & (win < W_tot) & (at >= 1) & (at < seq)
    w_safe = np.where(valid, win, 0)
    a_safe = np.where(valid, at, 1)

    t0 = np.where(valid, windows[w_safe, a_safe], -1)
    t0 = np.where(t0 >= 0, t0, -1)

    has_m1 = valid & (a_safe > 1)
    tm1 = np.where(has_m1, windows[w_safe, np.maximum(a_safe - 1, 0)], -1)
    tm1 = np.where((tm1 >= 0) & (tm1 != bos_id), tm1, -1)

    has_p1 = valid & (a_safe + 1 < seq)
    tp1 = np.where(has_p1, windows[w_safe, np.minimum(a_safe + 1, seq - 1)], -1)
    tp1 = np.where((tp1 >= 0) & (tp1 != eos_id) & (tp1 != bos_id), tp1, -1)
    return t0, tm1, tp1


def _score_causal_split(
    h0_idxs: Sequence[int],
    h1_idxs: Sequence[int],
    t0_row: np.ndarray,
    tm1_row: np.ndarray,
    bg_freq: np.ndarray,
    top_k: int = 5,
    compute_details: bool = True,
) -> Tuple[float, bool, List[Tuple[int, int, int, int, float]]]:
    """Score one document-disjoint split using strictly causal offsets ``(-1, 0)``."""
    n0 = len(h0_idxs)
    n1 = len(h1_idxs)
    if n0 < 4 or n1 < 4:
        return 0.0, False, []

    ce: Dict[int, int] = {}
    co: Dict[int, int] = {}
    bge: Dict[int, int] = {}
    bgo: Dict[int, int] = {}

    for j in h0_idxs:
        x0 = int(t0_row[j])
        if x0 < 0:
            continue
        k0 = x0 * 2 + 1
        ce[k0] = ce.get(k0, 0) + 1
        xm = int(tm1_row[j])
        if xm >= 0:
            km = xm * 2
            ce[km] = ce.get(km, 0) + 1
            if compute_details:
                bk = (xm << 20) | x0
                bge[bk] = bge.get(bk, 0) + 1

    for j in h1_idxs:
        x0 = int(t0_row[j])
        if x0 < 0:
            continue
        k0 = x0 * 2 + 1
        co[k0] = co.get(k0, 0) + 1
        xm = int(tm1_row[j])
        if xm >= 0:
            km = xm * 2
            co[km] = co.get(km, 0) + 1
            if compute_details:
                bk = (xm << 20) | x0
                bgo[bk] = bgo.get(bk, 0) + 1

    bigram_rep = False
    if compute_details:
        for bk, cnt_e in bge.items():
            if cnt_e >= 2 and bgo.get(bk, 0) >= 2:
                bigram_rep = True
                break

    best_sc = 0.0
    cands: List[Tuple[float, int, int, int, int, float]] = []
    inv_n0 = 1.0 / n0
    inv_n1 = 1.0 / n1
    for k, cnt_e in ce.items():
        if cnt_e < 2:
            continue
        cnt_o = co.get(k, 0)
        if cnt_o < 2:
            continue
        tok, rem = divmod(k, 2)
        off = rem - 1  # -1 or 0 strictly
        rate = min(cnt_e * inv_n0, cnt_o * inv_n1)
        bg = float(bg_freq[tok]) if tok < len(bg_freq) else 1e-6
        if bg < 1e-6:
            bg = 1e-6
        lift = rate / bg
        sc = rate * math.log1p(lift)
        if sc > best_sc:
            best_sc = sc
        if compute_details:
            cands.append((sc, off, tok, cnt_e, cnt_o, lift))

    if compute_details and cands:
        cands.sort(key=lambda x: -x[0])
        top = [(off, tok, ce0, co0, lift0) for _, off, tok, ce0, co0, lift0 in cands[:top_k]]
        return float(best_sc), bigram_rep, top
    return float(best_sc), bigram_rep, []


def _score_next_pieces(
    h0_idxs: Sequence[int],
    h1_idxs: Sequence[int],
    tp1_row: np.ndarray,
    bg_freq: np.ndarray,
    top_k: int = 3,
) -> List[Tuple[int, int, int, float]]:
    """Separately tally ``+1`` next pieces (never mixed into causal triggers or score)."""
    n0, n1 = len(h0_idxs), len(h1_idxs)
    if n0 < 4 or n1 < 4:
        return []
    ce: Dict[int, int] = {}
    co: Dict[int, int] = {}
    for j in h0_idxs:
        xp = int(tp1_row[j])
        if xp >= 0:
            ce[xp] = ce.get(xp, 0) + 1
    for j in h1_idxs:
        xp = int(tp1_row[j])
        if xp >= 0:
            co[xp] = co.get(xp, 0) + 1
    inv_n0, inv_n1 = 1.0 / n0, 1.0 / n1
    cands: List[Tuple[float, int, int, int, float]] = []
    for tok, cnt_e in ce.items():
        if cnt_e < 2:
            continue
        cnt_o = co.get(tok, 0)
        if cnt_o < 2:
            continue
        rate = min(cnt_e * inv_n0, cnt_o * inv_n1)
        bg = max(float(bg_freq[tok]) if tok < len(bg_freq) else 1e-6, 1e-6)
        lift = rate / bg
        cands.append((rate * math.log1p(lift), tok, cnt_e, cnt_o, lift))
    cands.sort(key=lambda x: -x[0])
    return [(tok, cnt_e, cnt_o, lift) for _, tok, cnt_e, cnt_o, lift in cands[:top_k]]


def _build_freq_bins(unigram: np.ndarray, n_bins: int = 20, special_max: int = 4) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Partition tokens by log-unigram frequency rank into ``n_bins`` buckets for matched nulls."""
    V = len(unigram)
    tok_bin = np.full(V, -1, dtype=np.int32)
    valid = np.nonzero((unigram > 0) & (np.arange(V) > special_max))[0]
    if len(valid) == 0:
        valid = np.nonzero(unigram > 0)[0]
    if len(valid) == 0:
        return tok_bin, []
    order = valid[np.argsort(unigram[valid], kind="mergesort")]
    splits = np.array_split(order, min(n_bins, len(order)))
    bins: List[np.ndarray] = []
    for bi, arr in enumerate(splits):
        tok_bin[arr] = bi
        bins.append(arr.astype(np.int32))
    return tok_bin, bins


def analyze_layer_arrays(
    windows: np.ndarray,
    unigram: np.ndarray,
    pos_win: np.ndarray,
    pos_at: np.ndarray,
    neg_win: np.ndarray,
    neg_at: np.ndarray,
    pos_smp_win: Optional[np.ndarray] = None,
    pos_smp_at: Optional[np.ndarray] = None,
    neg_smp_win: Optional[np.ndarray] = None,
    neg_smp_at: Optional[np.ndarray] = None,
    n_null: int = 4096,
    n_null_splits: int = 20,
    seed: int = 2026112450,
    doc_split_seed: int = 20261124,
    eos_id: int = 1,
    bos_id: int = 2,
    top_k: int = 5,
    next_k: int = 3,
) -> Dict[str, Any]:
    """Compute document-disjoint causal-only (-1, 0) trigger tables and matched nulls for one layer."""
    N_ff, K = pos_win.shape
    W_tot, seq = windows.shape
    unigram_f = np.asarray(unigram, dtype=np.float64)
    tot_tok = max(float(unigram_f.sum()), 1.0)
    bg_freq = (unigram_f / tot_tok).astype(np.float64)
    V = len(bg_freq)
    top100_set = set(int(x) for x in np.argsort(-unigram_f)[: min(100, V)].tolist())

    doc_id, half_id, n_docs = build_doc_halves(windows, eos_id=eos_id, seed=doc_split_seed)
    tok_bin, freq_bins = _build_freq_bins(unigram_f)

    win_valid_len = np.sum(windows >= 0, axis=1).astype(np.int32)
    elig_wins = np.array([w for w in range(W_tot) if win_valid_len[w] >= 3], dtype=np.int32)

    eligible = np.zeros((2, N_ff), dtype=bool)
    raw_2plus2 = np.zeros((2, N_ff), dtype=bool)
    score = np.zeros((2, N_ff), dtype=np.float32)
    top_token_id = np.full((2, N_ff, top_k), -1, dtype=np.int32)
    top_offset = np.zeros((2, N_ff, top_k), dtype=np.int8)
    top_cnt_e = np.zeros((2, N_ff, top_k), dtype=np.uint8)
    top_cnt_o = np.zeros((2, N_ff, top_k), dtype=np.uint8)
    top_lift = np.zeros((2, N_ff, top_k), dtype=np.float32)
    next_token_id = np.full((2, N_ff, next_k), -1, dtype=np.int32)
    next_cnt_e = np.zeros((2, N_ff, next_k), dtype=np.uint8)
    next_cnt_o = np.zeros((2, N_ff, next_k), dtype=np.uint8)
    next_lift = np.zeros((2, N_ff, next_k), dtype=np.float32)
    bigram_rep = np.zeros((2, N_ff), dtype=bool)

    face_events: List[Tuple[np.ndarray, np.ndarray, List[int], List[int], Set[int]]] = []
    # Store per face across both polarities for cross-face frequency-matched null
    bins_to_faces: Dict[int, List[int]] = {}
    all_elig_flat: List[int] = []

    for s_idx, (f_win, f_at) in enumerate(((pos_win, pos_at), (neg_win, neg_at))):
        t0_all, tm1_all, tp1_all = _extract_tokens_for_events(windows, f_win, f_at, eos_id=eos_id, bos_id=bos_id)
        for n in range(N_ff):
            t0_r, tm1_r, tp1_r = t0_all[n], tm1_all[n], tp1_all[n]
            h0_idxs: List[int] = []
            h1_idxs: List[int] = []
            doc_set: set[int] = set()
            for j in range(K):
                w0, a0 = int(f_win[n, j]), int(f_at[n, j])
                if w0 < 0 or w0 >= W_tot or a0 < 1 or a0 >= seq or int(t0_r[j]) < 0:
                    continue
                doc_set.add(int(doc_id[w0, a0]))
                if int(half_id[w0, a0]) == 0:
                    h0_idxs.append(j)
                else:
                    h1_idxs.append(j)
            flat_f = s_idx * N_ff + n
            face_events.append((t0_r, tm1_r, h0_idxs, h1_idxs, doc_set))
            if len(h0_idxs) >= 4 and len(h1_idxs) >= 4:
                eligible[s_idx, n] = True
                all_elig_flat.append(flat_f)
                sc, bg, top = _score_causal_split(h0_idxs, h1_idxs, t0_r, tm1_r, bg_freq, top_k=top_k, compute_details=True)
                score[s_idx, n] = sc
                bigram_rep[s_idx, n] = bg
                if sc > 0.0:
                    raw_2plus2[s_idx, n] = True
                for k_i, (off, tok, ce0, co0, lift0) in enumerate(top):
                    top_token_id[s_idx, n, k_i] = tok
                    top_offset[s_idx, n, k_i] = off
                    top_cnt_e[s_idx, n, k_i] = min(ce0, 255)
                    top_cnt_o[s_idx, n, k_i] = min(co0, 255)
                    top_lift[s_idx, n, k_i] = lift0
                if top:
                    bi = int(tok_bin[top[0][1]])
                    if bi >= 0:
                        bins_to_faces.setdefault(bi, []).append(flat_f)
                nxt = _score_next_pieces(h0_idxs, h1_idxs, tp1_r, bg_freq, top_k=next_k)
                for k_i, (tok, ce0, co0, lift0) in enumerate(nxt):
                    next_token_id[s_idx, n, k_i] = tok
                    next_cnt_e[s_idx, n, k_i] = min(ce0, 255)
                    next_cnt_o[s_idx, n, k_i] = min(co0, 255)
                    next_lift[s_idx, n, k_i] = lift0

    # Random corpus position null (legacy reference comparison, causal offsets only, document-disjoint)
    rng = np.random.default_rng(seed)
    elig_pair_counts = [
        (len(face_events[flat_f][2]), len(face_events[flat_f][3]))
        for flat_f in all_elig_flat
    ]
    n_elig = len(all_elig_flat)
    elig_h0_wins = elig_wins[half_id[elig_wins, 1] == 0]
    elig_h1_wins = elig_wins[half_id[elig_wins, 1] == 1]
    if len(elig_h0_wins) == 0:
        elig_h0_wins = elig_wins
    if len(elig_h1_wins) == 0:
        elig_h1_wins = elig_wins

    ctl_scores = np.zeros(n_null, dtype=np.float32)
    if n_null > 0 and len(elig_wins) > 0:
        null_win = np.zeros((n_null, K), dtype=np.int32)
        null_at = np.zeros((n_null, K), dtype=np.int32)
        for c in range(n_null):
            ne, no = elig_pair_counts[c % n_elig] if n_elig else (8, 8)
            if ne + no > K:
                ne = K // 2
                no = K - ne
            null_win[c, :ne] = rng.choice(elig_h0_wins, size=ne, replace=True)
            null_win[c, ne : ne + no] = rng.choice(elig_h1_wins, size=no, replace=True)
            for j in range(ne + no):
                hi = int(win_valid_len[null_win[c, j]])
                null_at[c, j] = int(rng.integers(1, max(hi, 2)))
        nt0, ntm1, _ = _extract_tokens_for_events(windows, null_win, null_at, eos_id=eos_id, bos_id=bos_id)
        for c in range(n_null):
            ne, no = elig_pair_counts[c % n_elig] if n_elig else (8, 8)
            h0_c = [j for j in range(ne) if int(nt0[c, j]) >= 0]
            h1_c = [j for j in range(ne, ne + no) if int(nt0[c, j]) >= 0]
            sc_c, _, _ = _score_causal_split(h0_c, h1_c, nt0[c], ntm1[c], bg_freq, compute_details=False)
            ctl_scores[c] = sc_c

    scores_flat = score.reshape(-1)
    elig_flat = eligible.reshape(-1)
    ctl_elig_scores = np.where(elig_flat, scores_flat, 0.0)
    sel05 = empirical_fdp_1d(ctl_elig_scores, ctl_scores, alpha=0.05)
    sel10 = empirical_fdp_1d(ctl_elig_scores, ctl_scores, alpha=0.10)
    named05 = sel05["selected_mask"].reshape(2, N_ff) & raw_2plus2
    named10 = sel10["selected_mask"].reshape(2, N_ff) & raw_2plus2

    # Null #1 (stronger): same-layer frequency-matched other-face firing null.
    # For each raw_2plus2 face f, test f's top (tok, off) on another eligible face g of the SAME layer
    # chosen from the same unigram-frequency bin (falling back to all_elig_flat if bin has <=1 face).
    other_face_null = np.zeros((2, N_ff), dtype=bool)
    raw_indices = np.argwhere(raw_2plus2)
    other_face_hits = 0
    for s_idx, n in raw_indices:
        tok = int(top_token_id[s_idx, n, 0])
        off = int(top_offset[s_idx, n, 0])
        if tok < 0 or n_elig < 2:
            continue
        flat_f = int(s_idx) * N_ff + int(n)
        bi = int(tok_bin[tok])
        pool = bins_to_faces.get(bi, [])
        g_flat = flat_f
        if len(pool) >= 2:
            for _ in range(8):
                cand = int(pool[int(rng.integers(0, len(pool)))])
                if cand != flat_f:
                    g_flat = cand
                    break
        if g_flat == flat_f:
            for _ in range(8):
                cand = int(all_elig_flat[int(rng.integers(0, n_elig))])
                if cand != flat_f:
                    g_flat = cand
                    break
        if g_flat != flat_f:
            gt0, gtm1, gh0, gh1, _ = face_events[g_flat]
            arr = gtm1 if off == -1 else gt0
            c0 = sum(1 for j in gh0 if int(arr[j]) == tok)
            c1 = sum(1 for j in gh1 if int(arr[j]) == tok)
            if c0 >= 2 and c1 >= 2:
                other_face_null[s_idx, n] = True
                other_face_hits += 1

    # Null #2 (stronger): held-out unseen-document prediction vs frequency-matched wrong piece & shuffled
    smp_hits = np.zeros((2, N_ff), dtype=np.uint8)
    smp_freq_hits = np.zeros((2, N_ff), dtype=np.uint8)
    smp_total = np.zeros((2, N_ff), dtype=np.uint8)
    has_smp = pos_smp_win is not None and pos_smp_at is not None and neg_smp_win is not None and neg_smp_at is not None

    smp_named_hits_tot = 0
    smp_freq_hits_tot = 0
    smp_shuf_hits_tot = 0
    smp_named_denom = 0
    top1_common_cnt = 0
    has_next_rep = 0

    if has_smp:
        raw_pool = [(int(top_token_id[si, ni, 0]), int(top_offset[si, ni, 0])) for si, ni in raw_indices if int(top_token_id[si, ni, 0]) >= 0]
        for s_idx, (sw_arr, sa_arr) in enumerate(((pos_smp_win, pos_smp_at), (neg_smp_win, neg_smp_at))):
            st0, stm1, _ = _extract_tokens_for_events(windows, sw_arr, sa_arr, eos_id=eos_id, bos_id=bos_id)
            S_k = sw_arr.shape[1]
            for n in range(N_ff):
                tok = int(top_token_id[s_idx, n, 0])
                off = int(top_offset[s_idx, n, 0])
                if tok < 0:
                    continue
                flat_f = s_idx * N_ff + n
                top_docs = face_events[flat_f][4]
                bi = int(tok_bin[tok])
                wrong_tok = -1
                if bi >= 0 and len(freq_bins[bi]) >= 2:
                    fb = freq_bins[bi]
                    for _ in range(8):
                        wt = int(fb[int(rng.integers(0, len(fb)))])
                        if wt != tok:
                            wrong_tok = wt
                            break
                if wrong_tok < 0:
                    fb_any = np.nonzero(unigram_f > 0)[0]
                    if len(fb_any) >= 2:
                        for _ in range(8):
                            wt = int(fb_any[int(rng.integers(0, len(fb_any)))])
                            if wt != tok:
                                wrong_tok = wt
                                break
                tot_s = 0
                hit_s = 0
                fhit_s = 0
                shuf_s = 0
                if bool(raw_2plus2[s_idx, n]) and raw_pool:
                    shuf_tok, shuf_off = raw_pool[int(rng.integers(0, len(raw_pool)))]
                else:
                    shuf_tok, shuf_off = -1, 0

                for j in range(S_k):
                    w0, a0 = int(sw_arr[n, j]), int(sa_arr[n, j])
                    if w0 < 0 or w0 >= W_tot or a0 < 1 or a0 >= seq or int(st0[n, j]) < 0:
                        continue
                    if int(doc_id[w0, a0]) in top_docs:
                        continue  # Strictly unseen documents!
                    tot_s += 1
                    val = int(stm1[n, j] if off == -1 else st0[n, j])
                    if val == tok:
                        hit_s += 1
                    if wrong_tok >= 0 and val == wrong_tok:
                        fhit_s += 1
                    if shuf_tok >= 0:
                        sval = int(stm1[n, j] if shuf_off == -1 else st0[n, j])
                        if sval == shuf_tok:
                            shuf_s += 1

                smp_total[s_idx, n] = min(tot_s, 255)
                smp_hits[s_idx, n] = min(hit_s, 255)
                smp_freq_hits[s_idx, n] = min(fhit_s, 255)
                if bool(raw_2plus2[s_idx, n]) and tot_s > 0:
                    smp_named_hits_tot += hit_s
                    smp_freq_hits_tot += fhit_s
                    smp_shuf_hits_tot += shuf_s
                    smp_named_denom += tot_s

    for s_idx, n in raw_indices:
        if int(top_token_id[s_idx, n, 0]) in top100_set:
            top1_common_cnt += 1
        if int(next_token_id[s_idx, n, 0]) >= 0:
            has_next_rep += 1

    # Also record standalone next-piece (+1) count across ALL eligible faces for transparency
    next_any_elig = int(np.sum((next_token_id[:, :, 0] >= 0) & eligible))
    raw_cnt = int(raw_2plus2.sum())
    elig_cnt = int(eligible.sum())
    named05_cnt = int(named05.sum())
    named10_cnt = int(named10.sum())

    stats = {
        "total_faces": int(2 * N_ff),
        "eligible_faces": elig_cnt,
        "n_docs": int(n_docs),
        "raw_2plus2_count": raw_cnt,
        "raw_2plus2_pct_elig": round(100.0 * raw_cnt / max(elig_cnt, 1), 2),
        "named_fdp05_count": named05_cnt,
        "named_fdp05_pct_elig": round(100.0 * named05_cnt / max(elig_cnt, 1), 2),
        "named_fdp05_pct_total": round(100.0 * named05_cnt / max(2 * N_ff, 1), 2),
        "named_fdp10_count": named10_cnt,
        "fdp_vs_raw_diff": raw_cnt - named05_cnt,
        "threshold_05": sel05["threshold"],
        "null_random_corpus_pos_pct": round(100.0 * float(np.mean(ctl_scores > 0.0)), 2) if n_null > 0 else None,
        "other_face_null_hits": other_face_hits,
        "other_face_null_rate": round(other_face_hits / max(raw_cnt, 1), 4),
        "top1_in_top100_common_share": round(top1_common_cnt / max(raw_cnt, 1), 4),
        "bigram_rep_count": int(bigram_rep.sum()),
        "bigram_rep_pct_elig": round(100.0 * int(bigram_rep.sum()) / max(elig_cnt, 1), 2),
        "next_p1_rep_elig_count": next_any_elig,
        "next_p1_rep_elig_pct": round(100.0 * next_any_elig / max(elig_cnt, 1), 2),
        "smp_named_hits": smp_named_hits_tot,
        "smp_freq_hits": smp_freq_hits_tot,
        "smp_shuf_hits": smp_shuf_hits_tot,
        "smp_named_tot": smp_named_denom,
        "smp_named_rate": round(smp_named_hits_tot / max(smp_named_denom, 1), 4) if smp_named_denom else None,
        "smp_freq_rate": round(smp_freq_hits_tot / max(smp_named_denom, 1), 4) if smp_named_denom else None,
        "smp_shuf_rate": round(smp_shuf_hits_tot / max(smp_named_denom, 1), 4) if smp_named_denom else None,
    }

    return {
        "eligible": eligible,
        "raw_2plus2": raw_2plus2,
        "named_fdp05": named05,
        "named_fdp10": named10,
        "score": score,
        "top_token_id": top_token_id,
        "top_offset": top_offset,
        "top_cnt_e": top_cnt_e,
        "top_cnt_o": top_cnt_o,
        "top_lift": top_lift,
        "next_token_id": next_token_id,
        "next_cnt_e": next_cnt_e,
        "next_cnt_o": next_cnt_o,
        "next_lift": next_lift,
        "bigram_rep": bigram_rep,
        "other_face_null": other_face_null,
        "smp_hits": smp_hits,
        "smp_freq_hits": smp_freq_hits,
        "smp_total": smp_total,
        "stats": stats,
    }


def save_layer_input_census(out_path: Path, layer_res: Dict[str, Any]) -> None:
    """Save numeric trigger tables only (strictly causal offsets -1/0 in top_offset; never stores text)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        eligible=layer_res["eligible"],
        raw_2plus2=layer_res["raw_2plus2"],
        named_fdp05=layer_res["named_fdp05"],
        named_fdp10=layer_res["named_fdp10"],
        score=layer_res["score"],
        top_token_id=layer_res["top_token_id"],
        top_offset=layer_res["top_offset"],
        top_cnt_e=layer_res["top_cnt_e"],
        top_cnt_o=layer_res["top_cnt_o"],
        top_lift=layer_res["top_lift"],
        next_token_id=layer_res["next_token_id"],
        next_cnt_e=layer_res["next_cnt_e"],
        next_cnt_o=layer_res["next_cnt_o"],
        next_lift=layer_res["next_lift"],
        bigram_rep=layer_res["bigram_rep"],
        other_face_null=layer_res["other_face_null"],
        smp_hits=layer_res["smp_hits"],
        smp_freq_hits=layer_res["smp_freq_hits"],
        smp_total=layer_res["smp_total"],
    )
