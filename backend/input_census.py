"""Deterministic split-half Input-Side Firing Census (`input_census`) for layers 0-29.

Uses precomputed 2M-token `passages.d` top-16 firing events (even vs odd window split)
and matched empirical random-window pseudo-faces to identify replicated token/offset
triggers at empirical step-up FDP <= 0.05 (never saving raw corpus window text).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend import atlas_index as ai

OFFSETS: Tuple[int, ...] = (-1, 0, 1)


def decode_key(key: int) -> Tuple[int, int]:
    """Decode `tok * 3 + (off + 1)` back into `(off, tok)`."""
    tok, rem = divmod(int(key), 3)
    return rem - 1, tok


def encode_key(off: int, tok: int) -> int:
    return int(tok) * 3 + (int(off) + 1)


def empirical_fdp_1d(
    neu_scores: np.ndarray,
    ctl_scores: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Step-up empirical FDP over non-negative neuron/face scores against `ctl_scores`."""
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


def _extract_tokens_for_events(windows: np.ndarray, win: np.ndarray, at: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized extraction of tokens at offsets (-1, 0, +1) respecting BOS (pos 0) and padding (< 0)."""
    W_tot, seq = windows.shape
    valid = (win >= 0) & (win < W_tot) & (at >= 1) & (at < seq)
    w_safe = np.where(valid, win, 0)
    a_safe = np.where(valid, at, 1)

    t0 = np.where(valid, windows[w_safe, a_safe], -1)
    t0 = np.where(t0 >= 0, t0, -1)

    has_m1 = valid & (a_safe > 1)  # exclude BOS at index 0
    tm1 = np.where(has_m1, windows[w_safe, np.maximum(a_safe - 1, 0)], -1)
    tm1 = np.where(tm1 >= 0, tm1, -1)

    has_p1 = valid & (a_safe + 1 < seq)
    tp1 = np.where(has_p1, windows[w_safe, np.minimum(a_safe + 1, seq - 1)], -1)
    tp1 = np.where(tp1 >= 0, tp1, -1)
    return t0, tm1, tp1


def _score_single_event_split(
    ev_idxs: Sequence[int],
    od_idxs: Sequence[int],
    t0_row: np.ndarray,
    tm1_row: np.ndarray,
    tp1_row: np.ndarray,
    bg_freq: np.ndarray,
    top_k: int = 5,
    compute_details: bool = True,
) -> Tuple[float, bool, List[Tuple[int, int, int, int, float]]]:
    """Score one split-half event set with `>=2` replication in both even and odd halves."""
    ne = len(ev_idxs)
    no = len(od_idxs)
    if ne < 4 or no < 4:
        return 0.0, False, []

    ce: Dict[int, int] = {}
    co: Dict[int, int] = {}
    bge: Dict[int, int] = {}
    bgo: Dict[int, int] = {}

    for j in ev_idxs:
        x0 = int(t0_row[j])
        if x0 < 0:
            continue
        k0 = x0 * 3 + 1
        ce[k0] = ce.get(k0, 0) + 1
        xm = int(tm1_row[j])
        if xm >= 0:
            km = xm * 3
            ce[km] = ce.get(km, 0) + 1
            if compute_details:
                bk = (xm << 20) | x0
                bge[bk] = bge.get(bk, 0) + 1
        xp = int(tp1_row[j])
        if xp >= 0:
            kp = xp * 3 + 2
            ce[kp] = ce.get(kp, 0) + 1
            if compute_details:
                bk = (1 << 40) | (x0 << 20) | xp
                bge[bk] = bge.get(bk, 0) + 1

    for j in od_idxs:
        x0 = int(t0_row[j])
        if x0 < 0:
            continue
        k0 = x0 * 3 + 1
        co[k0] = co.get(k0, 0) + 1
        xm = int(tm1_row[j])
        if xm >= 0:
            km = xm * 3
            co[km] = co.get(km, 0) + 1
            if compute_details:
                bk = (xm << 20) | x0
                bgo[bk] = bgo.get(bk, 0) + 1
        xp = int(tp1_row[j])
        if xp >= 0:
            kp = xp * 3 + 2
            co[kp] = co.get(kp, 0) + 1
            if compute_details:
                bk = (1 << 40) | (x0 << 20) | xp
                bgo[bk] = bgo.get(bk, 0) + 1

    bigram_rep = False
    if compute_details:
        for bk, cnt_e in bge.items():
            if cnt_e >= 2 and bgo.get(bk, 0) >= 2:
                bigram_rep = True
                break

    best_sc = 0.0
    cands: List[Tuple[float, int, int, int, int, float]] = []
    inv_ne = 1.0 / ne
    inv_no = 1.0 / no
    for k, cnt_e in ce.items():
        if cnt_e < 2:
            continue
        cnt_o = co.get(k, 0)
        if cnt_o < 2:
            continue
        tok, rem = divmod(k, 3)
        rate = min(cnt_e * inv_ne, cnt_o * inv_no)
        bg = float(bg_freq[tok]) if tok < len(bg_freq) else 1e-6
        if bg < 1e-6:
            bg = 1e-6
        lift = rate / bg
        sc = rate * math.log1p(lift)
        if sc > best_sc:
            best_sc = sc
        if compute_details:
            cands.append((sc, rem - 1, tok, cnt_e, cnt_o, lift))

    if compute_details and cands:
        cands.sort(key=lambda x: -x[0])
        top = [(off, tok, ce0, co0, lift0) for _, off, tok, ce0, co0, lift0 in cands[:top_k]]
        return float(best_sc), bigram_rep, top
    return float(best_sc), bigram_rep, []


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
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute full split-half input-side census arrays and empirical FDP stats for one layer."""
    N_ff, K = pos_win.shape
    W_tot, seq = windows.shape
    unigram_f = np.asarray(unigram, dtype=np.float64)
    tot_tok = max(float(unigram_f.sum()), 1.0)
    bg_freq = (unigram_f / tot_tok).astype(np.float64)

    # Precompute valid lengths per corpus window for clean random pseudo-face sampling
    win_valid_len = np.sum(windows >= 0, axis=1).astype(np.int32)
    even_wins_all = np.array([w for w in range(0, W_tot, 2) if win_valid_len[w] >= 3], dtype=np.int32)
    odd_wins_all = np.array([w for w in range(1, W_tot, 2) if win_valid_len[w] >= 3], dtype=np.int32)

    eligible = np.zeros((2, N_ff), dtype=bool)
    score = np.zeros((2, N_ff), dtype=np.float32)
    top_token_id = np.full((2, N_ff, top_k), -1, dtype=np.int32)
    top_offset = np.zeros((2, N_ff, top_k), dtype=np.int8)
    top_cnt_e = np.zeros((2, N_ff, top_k), dtype=np.uint8)
    top_cnt_o = np.zeros((2, N_ff, top_k), dtype=np.uint8)
    top_lift = np.zeros((2, N_ff, top_k), dtype=np.float32)
    bigram_rep = np.zeros((2, N_ff), dtype=bool)
    smp_hits = np.zeros((2, N_ff), dtype=np.uint8)
    smp_total = np.zeros((2, N_ff), dtype=np.uint8)

    elig_counts: List[Tuple[int, int]] = []

    sides_data = [
        (0, np.asarray(pos_win), np.asarray(pos_at), np.asarray(pos_smp_win) if pos_smp_win is not None else None, np.asarray(pos_smp_at) if pos_smp_at is not None else None),
        (1, np.asarray(neg_win), np.asarray(neg_at), np.asarray(neg_smp_win) if neg_smp_win is not None else None, np.asarray(neg_smp_at) if neg_smp_at is not None else None),
    ]

    for s_idx, w_arr, a_arr, sw_arr, sa_arr in sides_data:
        t0_all, tm1_all, tp1_all = _extract_tokens_for_events(windows, w_arr, a_arr)
        has_smp = (sw_arr is not None) and (sa_arr is not None)
        if has_smp:
            st0_all, stm1_all, stp1_all = _extract_tokens_for_events(windows, sw_arr, sa_arr)

        for n in range(N_ff):
            ws = w_arr[n]
            ats = a_arr[n]
            valid = (ws >= 0) & (ats >= 1) & (ats < seq)
            ev_idxs = [j for j in range(K) if valid[j] and (int(ws[j]) & 1 == 0)]
            od_idxs = [j for j in range(K) if valid[j] and (int(ws[j]) & 1 == 1)]
            if len(ev_idxs) < 4 or len(od_idxs) < 4:
                continue
            eligible[s_idx, n] = True
            elig_counts.append((len(ev_idxs), len(od_idxs)))
            sc, bg_flag, top = _score_single_event_split(
                ev_idxs, od_idxs, t0_all[n], tm1_all[n], tp1_all[n], bg_freq, top_k=top_k, compute_details=True
            )
            score[s_idx, n] = sc
            bigram_rep[s_idx, n] = bg_flag
            for r_i, (off, tok, ce0, co0, lift0) in enumerate(top):
                top_token_id[s_idx, n, r_i] = tok
                top_offset[s_idx, n, r_i] = off
                top_cnt_e[s_idx, n, r_i] = min(255, ce0)
                top_cnt_o[s_idx, n, r_i] = min(255, co0)
                top_lift[s_idx, n, r_i] = lift0

            if has_smp and top:
                top_win_set = {int(ws[j]) for j in range(K) if int(ws[j]) >= 0}
                best_off, best_tok = top[0][0], top[0][1]
                arr_check = st0_all[n] if best_off == 0 else (stm1_all[n] if best_off == -1 else stp1_all[n])
                sw_row = sw_arr[n]
                sa_row = sa_arr[n]
                tot_s, hit_s = 0, 0
                for sj in range(len(sw_row)):
                    w_s = int(sw_row[sj])
                    a_s = int(sa_row[sj])
                    if w_s < 0 or a_s < 1 or a_s >= seq or w_s in top_win_set:
                        continue
                    tot_s += 1
                    if int(arr_check[sj]) == best_tok:
                        hit_s += 1
                smp_total[s_idx, n] = min(255, tot_s)
                smp_hits[s_idx, n] = min(255, hit_s)

    # Matched empirical random-window null
    elig_flat = np.nonzero(eligible.reshape(-1))[0]
    n_elig = len(elig_flat)
    rng = np.random.default_rng(int(seed))
    null_scores = np.zeros(n_null, dtype=np.float64)
    if n_elig > 0 and len(even_wins_all) >= 4 and len(odd_wins_all) >= 4:
        pair_choices = rng.integers(0, len(elig_counts), size=n_null)
        max_e = max(elig_counts[i][0] for i in pair_choices)
        max_o = max(elig_counts[i][1] for i in pair_choices)
        K_null = max_e + max_o
        null_win = np.full((n_null, K_null), -1, dtype=np.int32)
        null_at = np.full((n_null, K_null), -1, dtype=np.int32)
        for ci in range(n_null):
            ne, no = elig_counts[pair_choices[ci]]
            ew = rng.choice(even_wins_all, size=min(ne, len(even_wins_all)), replace=False)
            ow = rng.choice(odd_wins_all, size=min(no, len(odd_wins_all)), replace=False)
            null_win[ci, : len(ew)] = ew
            null_at[ci, : len(ew)] = rng.integers(1, win_valid_len[ew] - 1)
            null_win[ci, max_e : max_e + len(ow)] = ow
            null_at[ci, max_e : max_e + len(ow)] = rng.integers(1, win_valid_len[ow] - 1)

        nt0, ntm1, ntp1 = _extract_tokens_for_events(windows, null_win, null_at)
        ev_template = list(range(max_e))
        od_template = list(range(max_e, K_null))
        for ci in range(n_null):
            ne, no = elig_counts[pair_choices[ci]]
            sc_n, _, _ = _score_single_event_split(
                ev_template[: min(ne, len(even_wins_all))],
                od_template[: min(no, len(odd_wins_all))],
                nt0[ci],
                ntm1[ci],
                ntp1[ci],
                bg_freq,
                top_k=1,
                compute_details=False,
            )
            null_scores[ci] = sc_n

    scores_elig = score.reshape(-1)[elig_flat].astype(np.float64)
    sel05 = empirical_fdp_1d(scores_elig, null_scores, alpha=0.05)
    sel10 = empirical_fdp_1d(scores_elig, null_scores, alpha=0.10)

    named_fdp05 = np.zeros(2 * N_ff, dtype=bool)
    named_fdp10 = np.zeros(2 * N_ff, dtype=bool)
    if n_elig > 0:
        named_fdp05[elig_flat[sel05["selected_mask"]]] = True
        named_fdp10[elig_flat[sel10["selected_mask"]]] = True
    named_fdp05_2d = named_fdp05.reshape(2, N_ff)
    named_fdp10_2d = named_fdp10.reshape(2, N_ff)

    # Half-null splits calibration check
    null_split_counts: List[int] = []
    if n_null_splits > 0 and n_null >= 64:
        half = n_null // 2
        for _ in range(n_null_splits):
            perm = rng.permutation(n_null)
            f_sc = null_scores[perm[:half]]
            r_sc = null_scores[perm[half:]]
            null_split_counts.append(empirical_fdp_1d(f_sc, r_sc, alpha=0.05)["count"])

    # Mid-range sample confirmation hit rate on named vs shuffled wrong-face triggers & non-named rep faces
    smp_named_hits, smp_named_tot = 0, 0
    smp_shuf_hits, smp_shuf_tot = 0, 0
    smp_other_hits, smp_other_tot = 0, 0

    named_coords = list(zip(*np.nonzero(named_fdp05_2d)))
    if len(named_coords) >= 2 and pos_smp_win is not None and neg_smp_win is not None:
        smp_w_pair = (np.asarray(pos_smp_win), np.asarray(neg_smp_win))
        smp_a_pair = (np.asarray(pos_smp_at), np.asarray(neg_smp_at))
        w_pair = (np.asarray(pos_win), np.asarray(neg_win))
        shuf_perm = rng.permutation(len(named_coords))
        for idx_c, (si, ni) in enumerate(named_coords):
            smp_named_hits += int(smp_hits[si, ni])
            smp_named_tot += int(smp_total[si, ni])
            # Wrong-face control: evaluate shuffled named face's trigger on THIS face's held-out samples
            sj, nj = named_coords[int(shuf_perm[idx_c])]
            if idx_c == int(shuf_perm[idx_c]):
                sj, nj = named_coords[(idx_c + 1) % len(named_coords)]
            sh_off = int(top_offset[sj, nj, 0])
            sh_tok = int(top_token_id[sj, nj, 0])
            top_win_set = {int(w) for w in w_pair[si][ni] if int(w) >= 0}
            sw_row = smp_w_pair[si][ni]
            sa_row = smp_a_pair[si][ni]
            for sk in range(len(sw_row)):
                ws0, as0 = int(sw_row[sk]), int(sa_row[sk])
                if ws0 < 0 or as0 < 1 or as0 >= seq or ws0 in top_win_set:
                    continue
                smp_shuf_tot += 1
                chk_pos = as0 + sh_off
                if sh_off == -1 and as0 <= 1:
                    continue
                if 1 <= chk_pos < seq and int(windows[ws0, chk_pos]) == sh_tok:
                    smp_shuf_hits += 1

    non_named_mask = eligible & (~named_fdp05_2d) & (top_cnt_e[:, :, 0] > 0)
    smp_other_hits = int(smp_hits[non_named_mask].sum())
    smp_other_tot = int(smp_total[non_named_mask].sum())

    return {
        "eligible": eligible,
        "named_fdp05": named_fdp05_2d,
        "named_fdp10": named_fdp10_2d,
        "score": score,
        "top_token_id": top_token_id,
        "top_offset": top_offset,
        "top_cnt_e": top_cnt_e,
        "top_cnt_o": top_cnt_o,
        "top_lift": top_lift,
        "bigram_rep": bigram_rep,
        "smp_hits": smp_hits,
        "smp_total": smp_total,
        "stats": {
            "total_faces": int(2 * N_ff),
            "eligible_faces": int(n_elig),
            "named_fdp05_count": int(sel05["count"]),
            "named_fdp05_pct_elig": round(100.0 * sel05["count"] / max(n_elig, 1), 2),
            "named_fdp05_pct_total": round(100.0 * sel05["count"] / max(2 * N_ff, 1), 2),
            "named_fdp10_count": int(sel10["count"]),
            "named_fdp10_pct_elig": round(100.0 * sel10["count"] / max(n_elig, 1), 2),
            "threshold_05": round(float(sel05["threshold"]), 5) if sel05["threshold"] is not None else None,
            "bigram_rep_pct_elig": round(100.0 * int(bigram_rep.sum()) / max(n_elig, 1), 2),
            "null_split_false_mean": round(float(np.mean(null_split_counts)), 2) if null_split_counts else None,
            "null_split_false_max": int(np.max(null_split_counts)) if null_split_counts else None,
            "smp_named_hits": smp_named_hits,
            "smp_named_tot": smp_named_tot,
            "smp_named_rate": round(smp_named_hits / max(smp_named_tot, 1), 4) if smp_named_tot else None,
            "smp_shuf_rate": round(smp_shuf_hits / max(smp_shuf_tot, 1), 4) if smp_shuf_tot else None,
            "smp_non_named_rate": round(smp_other_hits / max(smp_other_tot, 1), 4) if smp_other_tot else None,
        },
    }


def save_layer_input_census(out_path: Path, layer_res: Dict[str, Any]) -> None:
    """Save compact numeric arrays for one layer (strictly NO raw corpus text)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        eligible=layer_res["eligible"],
        named_fdp05=layer_res["named_fdp05"],
        named_fdp10=layer_res["named_fdp10"],
        score=layer_res["score"],
        top_token_id=layer_res["top_token_id"],
        top_offset=layer_res["top_offset"],
        top_cnt_e=layer_res["top_cnt_e"],
        top_cnt_o=layer_res["top_cnt_o"],
        top_lift=layer_res["top_lift"],
        bigram_rep=layer_res["bigram_rep"],
        smp_hits=layer_res["smp_hits"],
        smp_total=layer_res["smp_total"],
    )
