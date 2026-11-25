#!/usr/bin/env python3
"""Experiments 1B & 2 (GPU):
  Part 2A: Corrected Weight-Word Null (W(u_n) vs control's OWN W(u_c)) at L29, L53, L54, L55.
  Part 2B: Controlled Antisymmetric Multi-Position Forced Logit Census (`forced_odd`) pilot across
           L45 (anchor), L59, L55, L29 (+ ablations), L25, L20 (and L14 if L20 holds).
"""
from __future__ import annotations

import collections
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse, stats

WORKTREE = Path("/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief")
MAIN_DATA = Path("/home/eric/bin/ai/theodolite2/atlas_data")
OUT_DIR = WORKTREE / "experiments/results/census_gap_pilots"

sys.path.insert(0, str(WORKTREE))
from backend import atlas_ops as ao  # noqa: E402
from backend import census as cz  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.label_gate import label_top_ids  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402
from backend.theodolite_math import unembed_gemma4  # noqa: E402
from backend import w4a16_fast  # noqa: E402

HALVES = (
    ("bicycle", "walk", "chair", "kettle", "sleep", "lamp"),
    ("clock", "rice", "vase", "lock", "money", "blanket"),
)
HALF_C = ("bridge", "windmill", "camera", "boot", "mug", "winter")
L29_EXPLORE_NEURONS = [
    20164, 15902, 19529, 9747, 554, 453, 2503, 10902, 20632, 9733,
    3109, 3700, 15288, 17213, 19588, 12023, 21411, 4590, 9604, 17846,
]


def auc(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan")
    u = stats.mannwhitneyu(a_arr, b_arr, alternative="greater")
    return float(u.statistic / (len(a_arr) * len(b_arr)))


def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def norm_word(s: str) -> str:
    return re.sub(r"[^\w]", "", str(s).replace("▁", " ").strip().lower())


def rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    rx = np.argsort(np.argsort(np.asarray(x)))
    ry = np.argsort(np.argsort(np.asarray(y)))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


# ================================================================================================
# Part 2A: Corrected Weight-Word Null
# ================================================================================================

def run_part2a(rt: Any, lm: nn.Module, w: ao.Weights) -> None:
    print("\n=== Part 2A: Corrected Weight-Word Null ===", flush=True)
    vocab_list = json.loads((MAIN_DATA / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    VN = [norm_word(t) for t in vocab_list]
    lens_npz = np.load(MAIN_DATA / "index/lens.npz")
    eye_pos_all = lens_npz["eye_pos"]
    eye_neg_all = lens_npz["eye_neg"]
    fnw, W_U = rt.final_norm_weight, rt.lm_head_weight
    dev = model_device(lm)

    # Sanity check: verify W_U @ (w_final * u_n) matches lens.npz eye_pos/eye_neg on 20 neurons at L55
    test_ns = list(range(0, 2000, 100))[:20]
    w_post_55 = w.norm(55, "post_feedforward").to(dev)
    vecs = (w.down_cols(55, test_ns) * w_post_55).to(W_U.dtype)
    logits = unembed_gemma4(vecs, fnw, W_U)
    got_pos = label_top_ids(vecs, fnw, W_U, logits, k=10)
    got_neg = label_top_ids(-vecs, fnw, W_U, -logits, k=10)
    del vecs, logits
    exact_p, exact_n, overlap_p, overlap_n = 0, 0, 0, 0
    for idx, n in enumerate(test_ns):
        ref_p = [int(x) for x in eye_pos_all[55, n, :10]]
        ref_n = [int(x) for x in eye_neg_all[55, n, :10]]
        if got_pos[idx] == ref_p:
            exact_p += 1
        if got_neg[idx] == ref_n:
            exact_n += 1
        overlap_p += len(set(got_pos[idx]) & set(ref_p))
        overlap_n += len(set(got_neg[idx]) & set(ref_n))
    print(f"Lens sanity (20 L55 neurons): exact_pos={exact_p}/20, exact_neg={exact_n}/20, "
          f"overlap_pos={overlap_p}/200, overlap_neg={overlap_n}/200", flush=True)
    if overlap_p < 180 or overlap_n < 180:
        raise RuntimeError(f"Weight-word sanity check failed: overlap {overlap_p}/200, {overlap_n}/200")

    results = []
    txt_lines = [
        f"Part 2A: Corrected Weight-Word Null (sanity L55 top-10 overlap: +{overlap_p}/200, -{overlap_n}/200)",
        "-" * 116,
        f"{'L':>3} {'Subset':<7} {'Old_lead':>8} {'Old_q05':>7} {'Corr_lead':>9} {'Corr_q05':>8} "
        f"{'AboveMax':>8} {'Neu>0':>6} {'Ctl>0':>6} {'RhoN':>6} {'RhoC_own':>8}",
    ]

    for layer in (29, 53, 54, 55):
        trial_path = MAIN_DATA / f"census/trials/things_walk/L{layer}-match_things-trial.jsonl"
        meta = json.loads(cz.meta_path_of(trial_path).read_text(encoding="utf-8"))
        rows = []
        with open(trial_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "baseline":
                    continue
                rows.append((r["kind"], int(r["neuron"]), r["prompt"], float(r["dose"]), r.get("token_ids") or []))
        neurons = sorted({n for k, n, *_ in rows if k == "neuron"})
        controls = sorted({n for k, n, *_ in rows if k == "random"})
        C = len(controls)
        nn = len(neurons)
        seed = int(meta["seed"])
        ckind = cz.control_kind_of(meta)
        w_post = w.norm(layer, "post_feedforward").cpu().numpy()
        writes = None
        if ckind == "columns":
            writes = np.concatenate([w.write_dirs(layer, list(range(a, min(a + 4096, w.d_ff)))).cpu().numpy()
                                     for a in range(0, w.d_ff, 4096)])
        ctrl_dirs = cz.control_dirs(seed, C, w_post, ckind, writes)
        del writes

        # Compute naked-eye top-10 for all controls
        ctl_eye_pos, ctl_eye_neg = [], []
        for s in range(0, C, 128):
            cv = torch.from_numpy(ctrl_dirs[s : s + 128]).to(device=dev, dtype=W_U.dtype)
            cl = unembed_gemma4(cv, fnw, W_U)
            ctl_eye_pos.extend(label_top_ids(cv, fnw, W_U, cl, k=10))
            ctl_eye_neg.extend(label_top_ids(-cv, fnw, W_U, -cl, k=10))
            del cv, cl

        # Word sets for neurons (1022) and controls (1024)
        W_neu = {s: [set(VN[t] for t in (eye_pos_all if s == 1 else eye_neg_all)[layer, n, :10] if VN[t])
                     for n in neurons] for s in (+1, -1)}
        W_ctl = {s: [set(VN[t] for t in (ctl_eye_pos if s == 1 else ctl_eye_neg)[c][:10] if VN[t])
                     for c in range(C)] for s in (+1, -1)}

        all_words = sorted(set().union(*W_neu[1], *W_neu[-1], *W_ctl[1], *W_ctl[-1]))
        vi = {wd: i for i, wd in enumerate(all_words)}
        Vn = len(all_words)

        def build_mat(wdict):
            mats = {}
            for s in (+1, -1):
                rows_i, cols_i = [], []
                for j, wset in enumerate(wdict[s]):
                    for wd in wset:
                        rows_i.append(vi[wd])
                        cols_i.append(j)
                mats[s] = sparse.csr_matrix((np.ones(len(rows_i), np.float32), (rows_i, cols_i)),
                                            shape=(Vn, len(wdict[s])))
            return mats

        M_neu = build_mat(W_neu)
        M_ctl = build_mat(W_ctl)

        mags = sorted({abs(d) for *_, d, _ in rows})
        subsets = {
            "whole": lambda q, d: True,
            "half A": lambda q, d: q in HALVES[0],
            "half B": lambda q, d: q in HALVES[1],
            "lower": lambda q, d: abs(d) == mags[0],
            "upper": lambda q, d: abs(d) == mags[-1],
        }
        units = [("neuron", n) for n in neurons] + [("random", c) for c in controls]
        ui = {u: i for i, u in enumerate(units)}
        acc = {(nm, s): np.zeros((len(units), Vn), np.float32) for nm in subsets for s in (+1, -1)}
        cnt = {(nm, s): np.zeros(len(units), np.float32) for nm in subsets for s in (+1, -1)}

        for kind, n, q, d, ids in rows:
            s = 1 if d > 0 else -1
            hits = [vi[w0] for w0 in (VN[t] for t in ids) if w0 in vi]
            u = ui[(kind, n)]
            inv_l = 1.0 / max(len(ids), 1)
            vec = np.bincount(hits, minlength=Vn).astype(np.float32) * inv_l if hits else None
            for nm, pred in subsets.items():
                if pred(q, d):
                    cnt[(nm, s)][u] += 1.0
                    if vec is not None:
                        acc[(nm, s)][u] += vec

        layer_entry: Dict[str, Any] = {"layer": layer, "neurons": nn, "controls": C, "control_kind": ckind, "subsets": {}}
        neu_shares, ctl_own_shares = {}, {}

        for nm in subsets:
            F_mat = {s: sparse.csr_matrix(acc[(nm, s)] / np.maximum(cnt[(nm, s)], 1.0)[:, None]) for s in (+1, -1)}
            # Score every unit on every neuron's words (exact explore.py)
            S_neu = 0.5 * (F_mat[1] @ M_neu[1] + F_mat[-1] @ M_neu[-1])
            S_neu = np.asarray(S_neu.todense())
            own_n = S_neu[np.arange(nn), np.arange(nn)]
            ctl_orth = S_neu[nn:, :]  # [C, nn] controls scored on neurons' words
            below_orth = ((ctl_orth < own_n).sum(0) + 0.5 * (ctl_orth == own_n).sum(0)) / C
            p_orth = (1.0 + (ctl_orth >= own_n).sum(0)) / (1.0 + C)
            q_orth = bh(p_orth)

            # Corrected null: score controls on their OWN naked-eye words
            S_ctl_own = 0.5 * (F_mat[1][nn:] @ M_ctl[1] + F_mat[-1][nn:] @ M_ctl[-1])
            S_ctl_own = np.asarray(S_ctl_own.todense())
            own_c = S_ctl_own[np.arange(C), np.arange(C)]

            corr_lead = auc(own_n, own_c)
            sorted_oc = np.sort(own_c)
            ge_oc = C - np.searchsorted(sorted_oc, own_n, side="left")
            p_corr = (1.0 + ge_oc) / (1.0 + C)
            q_corr = bh(p_corr)
            above_max_corr = int(np.sum(own_n > float(sorted_oc[-1])))

            neu_shares[nm] = own_n
            ctl_own_shares[nm] = own_c
            sub_res = {
                "old_orth_lead": round(float(below_orth.mean()), 4),
                "old_orth_q05": int(np.sum(q_orth <= 0.05)),
                "old_orth_above_max": int(np.sum(own_n > ctl_orth.max(0))),
                "corr_lead_auc": round(corr_lead, 4),
                "corr_q05": int(np.sum(q_corr <= 0.05)),
                "corr_above_max": above_max_corr,
                "mean_own_n": round(float(own_n.mean()), 5),
                "mean_own_c": round(float(own_c.mean()), 5),
                "frac_gt0_n": round(float(np.mean(own_n > 0)), 4),
                "frac_gt0_c_own": round(float(np.mean(own_c > 0)), 4),
                "frac_gt0_c_orth": round(float(np.mean(ctl_orth > 0)), 4),
            }
            layer_entry["subsets"][nm] = sub_res

        rho_n = rank_corr(neu_shares["half A"], neu_shares["half B"])
        rho_c_own = rank_corr(ctl_own_shares["half A"], ctl_own_shares["half B"])
        layer_entry["rho_AB_neurons"] = round(rho_n, 4)
        layer_entry["rho_AB_controls_own"] = round(rho_c_own, 4)
        results.append(layer_entry)

        for nm in ("whole", "half A", "half B"):
            s = layer_entry["subsets"][nm]
            txt_lines.append(
                f"{layer:3d} {nm:<7} {s['old_orth_lead']:8.3f} {s['old_orth_q05']:7d} "
                f"{s['corr_lead_auc']:9.3f} {s['corr_q05']:8d} {s['corr_above_max']:8d} "
                f"{s['frac_gt0_n']:6.2f} {s['frac_gt0_c_own']:6.2f} "
                f"{rho_n if nm == 'whole' else float('nan'):6.3f} {rho_c_own if nm == 'whole' else float('nan'):8.3f}"
            )
        del rows, acc
        gc.collect()

    out_json = {
        "sanity_L55_20_neurons": {"exact_pos": exact_p, "exact_neg": exact_n, "overlap_pos": overlap_p, "overlap_neg": overlap_n},
        "layers": results,
    }
    (OUT_DIR / "weight_words_corrected.json").write_text(json.dumps(out_json, indent=1) + "\n")
    (OUT_DIR / "weight_words_corrected.txt").write_text("\n".join(txt_lines) + "\n")
    print("\n".join(txt_lines), flush=True)


# ================================================================================================
# Part 2B: Controlled Antisymmetric Multi-Position Forced Logit Census (forced_odd)
# ================================================================================================

def build_vocab_masks(rt: Any) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    tok = rt.tokenizer
    V = int(rt.lm_head_weight.shape[0])
    special_ids = set(int(i) for i in (getattr(tok, "all_special_ids", None) or []))
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    word_re = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
    all_ns = np.ones(V, dtype=bool)
    lex = np.zeros(V, dtype=bool)
    clean_str = [""] * V
    raw_str = [""] * V
    for i, p in enumerate(pieces):
        s = "" if p is None else str(p)
        raw_str[i] = s
        if i in special_ids or s.startswith("<"):
            all_ns[i] = False
            continue
        cleaned = s.replace("▁", "").strip()
        if word_re.fullmatch(cleaned):
            lex[i] = True
            clean_str[i] = cleaned.lower()
        else:
            clean_str[i] = cleaned.lower()
    return torch.from_numpy(lex), torch.from_numpy(all_ns), clean_str, raw_str


def build_fixed_texts(rt: Any, lm: nn.Module) -> Dict[str, Any]:
    meta29 = json.loads((MAIN_DATA / "census/trials/things_walk/L29-match_things-trial.meta.json").read_text(encoding="utf-8"))
    prompt_ids_map: Dict[str, List[int]] = dict(meta29["prompt_ids"])
    bases29: Dict[str, List[int]] = {}
    with open(MAIN_DATA / "census/trials/things_walk/L29-match_things-trial.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "baseline":
                bases29.setdefault(r["prompt"], [int(t) for t in r["token_ids"]])

    # Generate Half C unpushed baseline answers once on GPU
    stop_ids = rt.end_of_turn_ids()
    bos_id = rt.tokenizer.bos_token_id
    softcap = ao.softcap_of(rt.hf_model)
    c_prompts = [rt.tokenize_prompt(rt.compose_prompt(cz.MORE_PROMPTS[q], chat=True)) for q in HALF_C]
    c_answers: List[List[int]] = [[] for _ in HALF_C]
    c_ended = [False] * len(HALF_C)
    for st in ao.write_rows(lm, rt.lm_head_weight, c_prompts, None, max_new=48, stop_ids=stop_ids,
                            bos_id=bos_id, spare_bos=True, softcap=softcap, top_k=0):
        c_answers[st.row].append(st.token_id)
        c_ended[st.row] = c_ended[st.row] or st.end
    for q, pids, ans, end in zip(HALF_C, c_prompts, c_answers, c_ended):
        prompt_ids_map[q] = pids
        bases29[q] = ans[:-1] if end else ans

    # Build 18 sequences truncated to T <= 64 and select 16 mid-answer logit positions (predicting answer[4..34])
    seqs: Dict[str, List[int]] = {}
    prompt_lens: Dict[str, int] = {}
    scored_pos: Dict[str, List[int]] = {}
    for q in (*HALVES[0], *HALVES[1], *HALF_C):
        pids = list(prompt_ids_map[q])
        aids = list(bases29[q])
        # Trim trailing stop token if present in match_things baseline
        if aids and aids[-1] in stop_ids:
            aids = aids[:-1]
        full = (pids + aids)[:64]
        p_len = len(pids)
        L_q = len(full)
        cand = list(range(p_len + 3, min(L_q - 1, p_len + 35), 2))
        if len(cand) < 16:
            cand = sorted(set(int(round(x)) for x in np.linspace(p_len + 2, L_q - 2, 16)))[:16]
        seqs[q] = full
        prompt_lens[q] = p_len
        scored_pos[q] = cand[:16]

    return {
        "seqs": seqs,
        "prompt_lens": prompt_lens,
        "scored_pos": scored_pos,
        "groups": {"A": list(HALVES[0]), "B": list(HALVES[1]), "C": list(HALF_C)},
    }


def empirical_fdp(scores: np.ndarray, pok: np.ndarray, n_neu: int, ctl_scores: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    N = int(n_neu)
    C = len(ctl_scores)
    ctl_sorted = np.sort(ctl_scores.astype(np.float64))
    max_ctl = float(ctl_sorted[-1])
    above_max = int(np.sum(scores[pok] > max_ctl))
    cand_idx = np.nonzero(pok)[0]
    if len(cand_idx) == 0 or C == 0:
        return {"count": 0, "above_max_ctl": above_max, "threshold": None, "fdp": None, "named_idx": np.array([], np.int64)}
    cand_s = scores[cand_idx]
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
    if best_j < 0:
        return {"count": 0, "above_max_ctl": above_max, "threshold": None, "fdp": None, "named_idx": np.array([], np.int64)}
    return {
        "count": int(best_j + 1),
        "above_max_ctl": above_max,
        "threshold": round(float(sorted_s[best_j]), 4),
        "fdp": round(float(best_fdp), 4),
        "named_idx": cand_idx[order[: best_j + 1]],
    }


def ols_resid(y_all: np.ndarray, X_all: np.ndarray, ctl_mask: np.ndarray) -> np.ndarray:
    Xc = X_all[ctl_mask]
    yc = y_all[ctl_mask]
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return y_all - (X_all @ beta)


def rz_vec(vals: np.ndarray, ctl_vals: np.ndarray) -> np.ndarray:
    med = float(np.median(ctl_vals))
    mad = float(np.median(np.abs(ctl_vals - med)))
    std = float(np.std(ctl_vals))
    scale = max(1.4826 * mad, 0.25 * std, 1e-6)
    return (vals - med) / scale


def run_forward_half(
    lm: nn.Module, W_U: torch.Tensor, softcap: Optional[float],
    layer: int, h_l: torch.Tensor, att: torch.Tensor, pos: torch.Tensor,
    keep_mask_grp: torch.Tensor, scored_idx: torch.Tensor, base_lsm: torch.Tensor,
    deltas: torch.Tensor, batch_units: int, return_raw_signs: bool = False,
) -> Dict[str, Any]:
    """Run +/-deltas on a 6-text group [H=6, T, D] and return half-mean odd shift [M, V] + stats."""
    dev = h_l.device
    M, D = deltas.shape
    H, T, _ = h_l.shape
    S = scored_idx.shape[1]
    V = int(W_U.shape[0])

    odd_out = torch.empty((M, V), dtype=torch.float16, device="cpu")
    raw_pos_out = torch.empty((M, V), dtype=torch.float16, device="cpu") if return_raw_signs else None
    raw_neg_out = torch.empty((M, V), dtype=torch.float16, device="cpu") if return_raw_signs else None
    kl_pos_out = np.zeros(M, np.float32)
    kl_neg_out = np.zeros(M, np.float32)
    even_l2_out = np.zeros(M, np.float32)
    odd_l2_out = np.zeros(M, np.float32)

    h_pos_gather = torch.arange(H, device=dev)[:, None].expand(H, S)

    for s in range(0, M, batch_units):
        db = deltas[s : s + batch_units]  # [b, D]
        b = db.shape[0]
        # Construct [b, 2, H, T, D] flattened to [2*b*H, T, D]
        d_exp = db.to(h_l.dtype)[:, None, None, :] * keep_mask_grp[None, :, :, None]  # wait: [b, H, T, D]
        hp = (h_l[None, :, :, :] + d_exp).reshape(b * H, T, D)
        hn = (h_l[None, :, :, :] - d_exp).reshape(b * H, T, D)
        hs = torch.cat([hp, hn], dim=0)  # [2*b*H, T, D]
        del hp, hn, d_exp

        R = hs.shape[0]
        att_r = att.repeat(2 * b, 1)
        pos_r = pos.repeat(2 * b, 1)
        out = ao.forward_from_layer(lm, layer + 1, hs, att_r, pos_r)  # [2*b*H, T, D]
        del hs, att_r, pos_r

        # Gather only the H x S scored positions per row -> [2*b, H, S, D]
        out_v = out.view(2 * b, H, T, D)
        sel = out_v[:, h_pos_gather, scored_idx, :].reshape(2 * b * H * S, D)
        del out, out_v

        # fp32 logits + log_softmax in chunks of 256 scored positions (<= 268 MB VRAM)
        lsm = torch.empty((2 * b * H * S, V), dtype=torch.float32, device=dev)
        for c0 in range(0, sel.shape[0], 256):
            lsm[c0 : c0 + 256] = F.log_softmax(logits_fp32(sel[c0 : c0 + 256], W_U), dim=-1)
        del sel

        lsm = lsm.view(2, b, H, S, V)
        lsm_p, lsm_n = lsm[0], lsm[1]  # [b, H, S, V]
        dp = lsm_p - base_lsm[None, :, :, :]
        dn = lsm_n - base_lsm[None, :, :, :]
        kl_p = (lsm_p.exp() * dp).sum(-1).mean(dim=(1, 2))
        kl_n = (lsm_n.exp() * dn).sum(-1).mean(dim=(1, 2))
        odd = torch.clamp(0.5 * (lsm_p - lsm_n), -4.0, 4.0).mean(dim=(1, 2))  # [b, V]
        even = (0.5 * (dp + dn)).mean(dim=(1, 2))  # [b, V]

        odd_out[s : s + b] = odd.half().cpu()
        kl_pos_out[s : s + b] = kl_p.cpu().numpy()
        kl_neg_out[s : s + b] = kl_n.cpu().numpy()
        odd_l2_out[s : s + b] = odd.norm(dim=-1).cpu().numpy()
        even_l2_out[s : s + b] = even.norm(dim=-1).cpu().numpy()
        if return_raw_signs:
            raw_pos_out[s : s + b] = torch.clamp(dp, -4.0, 4.0).mean(dim=(1, 2)).half().cpu()
            raw_neg_out[s : s + b] = torch.clamp(dn, -4.0, 4.0).mean(dim=(1, 2)).half().cpu()
        del lsm, lsm_p, lsm_n, dp, dn, odd, even

    return {
        "odd": odd_out,
        "raw_pos": raw_pos_out,
        "raw_neg": raw_neg_out,
        "kl_pos": kl_pos_out,
        "kl_neg": kl_neg_out,
        "odd_l2": odd_l2_out,
        "even_l2": even_l2_out,
    }


def score_odd_pair_gpu(
    A_cpu: torch.Tensor, B_cpu: torch.Tensor, is_ctl: np.ndarray,
    cov: np.ndarray, mask_gpu: torch.Tensor, zscore_var: bool, dev: torch.device,
    ctl_pool_mask: Optional[np.ndarray] = None, keep_words: bool = False,
) -> Dict[str, Any]:
    """Center/normalize [U, V] fp16 Half A/B shifts against reference controls, compute R_AB/R_BA, resid, FDP, and null splits."""
    U, V = A_cpu.shape
    ref_m = ctl_pool_mask if ctl_pool_mask is not None else is_ctl
    C = int(ref_m.sum())
    mask_f = mask_gpu.float()
    M_mask = float(mask_f.sum().item())

    # Accumulate ref sums on GPU in chunks
    ref_idx = np.nonzero(ref_m)[0]
    sumA = torch.zeros(V, dtype=torch.float32, device=dev)
    sum2A = torch.zeros(V, dtype=torch.float32, device=dev)
    sumB = torch.zeros(V, dtype=torch.float32, device=dev)
    sum2B = torch.zeros(V, dtype=torch.float32, device=dev)
    for c0 in range(0, len(ref_idx), 128):
        ii = ref_idx[c0 : c0 + 128]
        xa = A_cpu[ii].to(device=dev, dtype=torch.float32)
        xb = B_cpu[ii].to(device=dev, dtype=torch.float32)
        sumA += xa.sum(0)
        sum2A += (xa * xa).sum(0)
        sumB += xb.sum(0)
        sum2B += (xb * xb).sum(0)
        del xa, xb

    muA_all = sumA / C
    varA_all = ((sum2A - C * muA_all * muA_all) / max(C - 1, 1)).clamp(min=0.0)
    muB_all = sumB / C
    varB_all = ((sum2B - C * muB_all * muB_all) / max(C - 1, 1)).clamp(min=0.0)

    r_ab = np.zeros(U, np.float32)
    r_ba = np.zeros(U, np.float32)
    topA_p = np.zeros((U, 5), np.int32) if keep_words else None
    topA_n = np.zeros((U, 5), np.int32) if keep_words else None
    topB_p = np.zeros((U, 5), np.int32) if keep_words else None
    topB_n = np.zeros((U, 5), np.int32) if keep_words else None
    posB_onA = np.zeros((U, 5), np.float32) if keep_words else None
    negB_onA = np.zeros((U, 5), np.float32) if keep_words else None
    posA_onB = np.zeros((U, 5), np.float32) if keep_words else None
    negA_onB = np.zeros((U, 5), np.float32) if keep_words else None

    masked_big_neg = torch.where(mask_gpu, torch.zeros(V, device=dev), torch.full((V,), -1e9, device=dev))
    masked_big_pos = torch.where(mask_gpu, torch.zeros(V, device=dev), torch.full((V,), 1e9, device=dev))

    for s in range(0, U, 64):
        e = min(U, s + 64)
        xa = A_cpu[s:e].to(device=dev, dtype=torch.float32)
        xb = B_cpu[s:e].to(device=dev, dtype=torch.float32)
        ref_b = torch.from_numpy(ref_m[s:e]).to(dev)

        # Exact LOO for ref controls, exact ref mean/var for non-ref
        if ref_b.any():
            muA_u = torch.where(ref_b[:, None], (sumA[None] - xa) / max(C - 1, 1), muA_all[None])
            muB_u = torch.where(ref_b[:, None], (sumB[None] - xb) / max(C - 1, 1), muB_all[None])
            if zscore_var:
                vA_loo = ((sum2A[None] - xa * xa) - (C - 1) * muA_u * muA_u) / max(C - 2, 1)
                vB_loo = ((sum2B[None] - xb * xb) - (C - 1) * muB_u * muB_u) / max(C - 2, 1)
                varA_u = torch.where(ref_b[:, None], vA_loo.clamp(min=0.0), varA_all[None])
                varB_u = torch.where(ref_b[:, None], vB_loo.clamp(min=0.0), varB_all[None])
        else:
            muA_u, muB_u = muA_all[None], muB_all[None]
            varA_u, varB_u = varA_all[None], varB_all[None]

        ca = xa - muA_u
        cb = xb - muB_u
        if zscore_var:
            ca = ca / torch.sqrt(varA_u + 0.0025)
            cb = cb / torch.sqrt(varB_u + 0.0025)

        ca = ca * mask_f[None]
        cb = cb * mask_f[None]
        rmsA = torch.sqrt((ca * ca).sum(-1, keepdim=True) / M_mask + 1e-8)
        rmsB = torch.sqrt((cb * cb).sum(-1, keepdim=True) / M_mask + 1e-8)
        na = ca / rmsA
        nb = cb / rmsB

        pa_idx = (na + masked_big_neg[None]).topk(5, dim=-1).indices
        na_idx = (na + masked_big_pos[None]).topk(5, dim=-1, largest=False).indices
        pb_idx = (nb + masked_big_neg[None]).topk(5, dim=-1).indices
        nb_idx = (nb + masked_big_pos[None]).topk(5, dim=-1, largest=False).indices

        s_ab = 0.1 * (nb.gather(-1, pa_idx).sum(-1) - nb.gather(-1, na_idx).sum(-1))
        s_ba = 0.1 * (na.gather(-1, pb_idx).sum(-1) - na.gather(-1, nb_idx).sum(-1))
        r_ab[s:e] = s_ab.cpu().numpy()
        r_ba[s:e] = s_ba.cpu().numpy()

        if keep_words:
            topA_p[s:e] = pa_idx.cpu().numpy()
            topA_n[s:e] = na_idx.cpu().numpy()
            topB_p[s:e] = pb_idx.cpu().numpy()
            topB_n[s:e] = nb_idx.cpu().numpy()
            posA_onB[s:e] = nb.gather(-1, pa_idx).cpu().numpy()
            negA_onB[s:e] = nb.gather(-1, na_idx).cpu().numpy()
            posB_onA[s:e] = na.gather(-1, pb_idx).cpu().numpy()
            negB_onA[s:e] = na.gather(-1, nb_idx).cpu().numpy()

        del xa, xb, ca, cb, na, nb

    # Residualize R_AB and R_BA on cov fitted on ref_m
    r_ab_res = ols_resid(r_ab, cov, ref_m)
    r_ba_res = ols_resid(r_ba, cov, ref_m)
    T_vec = np.minimum(rz_vec(r_ab_res, r_ab_res[ref_m]), rz_vec(r_ba_res, r_ba_res[ref_m]))

    out = {
        "r_ab": r_ab, "r_ba": r_ba,
        "r_ab_res": r_ab_res, "r_ba_res": r_ba_res,
        "T": T_vec,
    }
    if keep_words:
        out.update({
            "topA_p": topA_p, "topA_n": topA_n, "topB_p": topB_p, "topB_n": topB_n,
            "posA_onB": posA_onB, "negA_onB": negA_onB, "posB_onA": posB_onA, "negB_onA": negB_onA,
        })
    return out


def eval_split_metrics(scores_dict: Dict[str, Any], neu_mask: np.ndarray, ctl_mask: np.ndarray) -> Dict[str, Any]:
    r_ab, r_ba = scores_dict["r_ab"], scores_dict["r_ba"]
    r_abr, r_bar, T = scores_dict["r_ab_res"], scores_dict["r_ba_res"], scores_dict["T"]
    n_neu = int(neu_mask.sum())
    pok_neu = (r_abr[neu_mask] > 0) & (r_bar[neu_mask] > 0)
    sel05 = empirical_fdp(T[neu_mask], pok_neu, n_neu, T[ctl_mask], 0.05)
    sel10 = empirical_fdp(T[neu_mask], pok_neu, n_neu, T[ctl_mask], 0.10)
    return {
        "auc_r_AB": round(auc(r_ab[neu_mask], r_ab[ctl_mask]), 4),
        "auc_r_BA": round(auc(r_ba[neu_mask], r_ba[ctl_mask]), 4),
        "auc_r_AB_res": round(auc(r_abr[neu_mask], r_abr[ctl_mask]), 4),
        "auc_r_BA_res": round(auc(r_bar[neu_mask], r_bar[ctl_mask]), 4),
        "auc_T": round(auc(T[neu_mask], T[ctl_mask]), 4),
        "fdp05": {k: v for k, v in sel05.items() if k != "named_idx"},
        "fdp10": {k: v for k, v in sel10.items() if k != "named_idx"},
        "_named_local_idx": sel05["named_idx"],
    }


def extract_trial_neurons(layer: int) -> List[int]:
    p = MAIN_DATA / f"census/trials/things_walk/L{layer}-match_things-trial.jsonl"
    ns: Set[int] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            if '"neuron"' in line:
                r = json.loads(line)
                if r["kind"] == "neuron":
                    ns.add(int(r["neuron"]))
    return sorted(ns)


def run_layer_forced_odd(
    rt: Any, lm: nn.Module, w: ao.Weights, texts: Dict[str, Any],
    lex_mask_gpu: torch.Tensor, all_ns_mask_gpu: torch.Tensor, clean_str: List[str], raw_str: List[str],
    layer: int, initial_dose_ceil: float, neurons: List[int], n_cols: int, n_iso: int,
    seed: int, do_half_c: bool = False, do_null_splits: int = 50,
    cross_check_l45: Optional[Dict[int, Set[str]]] = None,
    cross_check_l55_named: Optional[Set[int]] = None,
    check_explore29: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    dev = model_device(lm)
    N_ff, D = w.d_ff, w.d_model
    softcap = ao.softcap_of(rt.hf_model)
    W_U = rt.lm_head_weight
    bos_id = rt.tokenizer.bos_token_id
    batch_units = 8 if layer >= 40 else 4

    # Cache h_l and base_lsm for groups A, B (and C if needed)
    groups = {}
    grp_names = ("A", "B", "C") if do_half_c else ("A", "B")
    for gnm in grp_names:
        qlist = texts["groups"][gnm]
        seq_list = [texts["seqs"][q] for q in qlist]
        inp, att, _, _ = pad_right(seq_list, dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        keep_m = ao.keep_mask(inp, att, bos_id, True).to(rt.lm_head_weight.dtype)
        seen: Dict[str, torch.Tensor] = {}
        hk = lm.layers[layer].register_forward_hook(lambda _m, _a, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            full_out = lm(input_ids=inp, attention_mask=att, use_cache=False)
        finally:
            hk.remove()
        h_l = seen["h"]
        h_final = full_out.last_hidden_state
        sc_idx = torch.tensor([texts["scored_pos"][q] for q in qlist], dtype=torch.long, device=dev)  # [6, 16]
        h_rows = torch.arange(len(qlist), device=dev)[:, None].expand(len(qlist), sc_idx.shape[1])
        h_sc = h_final[h_rows, sc_idx]  # [6, 16, D]
        V = int(W_U.shape[0])
        base_lsm = torch.empty((len(qlist), sc_idx.shape[1], V), dtype=torch.float32, device=dev)
        h_flat = h_sc.reshape(-1, D)
        for c0 in range(0, h_flat.shape[0], 64):
            base_lsm.view(-1, V)[c0 : c0 + 64] = F.log_softmax(logits_fp32(h_flat[c0 : c0 + 64], W_U), dim=-1)
        groups[gnm] = {"h_l": h_l, "att": att, "pos": pos, "keep": keep_m, "sc_idx": sc_idx, "base_lsm": base_lsm, "inp": inp}
        del full_out, h_final, h_sc, h_flat

    # Verify dose on 64 neurons (group A) targeting median KL in [0.18, 0.55] (target 0.35)
    sample64 = neurons[:: max(1, len(neurons) // 64)][:64]
    u64 = w.write_dirs(layer, sample64)
    c64 = w.write_ceiling(layer, sample64)
    dose_ceil = float(initial_dose_ceil)
    for attempt in range(3):
        res64 = run_forward_half(lm, W_U, softcap, layer, groups["A"]["h_l"], groups["A"]["att"],
                                 groups["A"]["pos"], groups["A"]["keep"], groups["A"]["sc_idx"],
                                 groups["A"]["base_lsm"], (dose_ceil * c64)[:, None] * u64, batch_units)
        kl64 = 0.5 * (res64["kl_pos"] + res64["kl_neg"])
        med_kl = float(np.median(kl64))
        print(f"L{layer} dose check try {attempt+1}: dose={dose_ceil:.3f} ceil -> med KL={med_kl:.4f} (p10={np.percentile(kl64,10):.4f}, p90={np.percentile(kl64,90):.4f})", flush=True)
        if 0.18 <= med_kl <= 0.55 or attempt == 2:
            break
        factor = min(2.0, max(0.3, (0.35 / max(med_kl, 1e-4)) ** 0.5))
        dose_ceil = round(dose_ceil * factor, 3)
    del u64, c64, res64

    # Build unit directions & ceilings: [columns_controls, isotropic_controls, neurons]
    w_post_np = w.norm(layer, "post_feedforward").cpu().numpy()
    writes_all = np.concatenate([w.write_dirs(layer, list(range(a, min(a + 4096, N_ff)))).cpu().numpy()
                                 for a in range(0, N_ff, 4096)])
    col_dirs = cz.control_dirs(seed, n_cols, w_post_np, "columns", writes_all)
    iso_dirs = cz.control_dirs(seed + 999, n_iso, w_post_np, "isotropic")
    del writes_all
    refs_col = cz.control_refs(seed, n_cols, N_ff)
    refs_iso = cz.control_refs(seed + 999, n_iso, N_ff)
    ref_ceils_col = w.write_ceiling(layer, refs_col).cpu().numpy()
    ref_ceils_iso = w.write_ceiling(layer, refs_iso).cpu().numpy()

    u_neu = w.write_dirs(layer, neurons)
    c_neu = w.write_ceiling(layer, neurons)

    delta_cols = torch.from_numpy((dose_ceil * ref_ceils_col)[:, None] * col_dirs).to(dev)
    delta_iso = torch.from_numpy((dose_ceil * ref_ceils_iso)[:, None] * iso_dirs).to(dev)
    delta_neu = (dose_ceil * c_neu)[:, None] * u_neu
    del u_neu, c_neu, col_dirs, iso_dirs

    all_deltas = torch.cat([delta_cols, delta_iso, delta_neu], dim=0)
    U_tot = all_deltas.shape[0]
    col_mask = np.zeros(U_tot, dtype=bool); col_mask[:n_cols] = True
    iso_mask = np.zeros(U_tot, dtype=bool); iso_mask[n_cols : n_cols + n_iso] = True
    neu_mask = np.zeros(U_tot, dtype=bool); neu_mask[n_cols + n_iso :] = True

    resA = run_forward_half(lm, W_U, softcap, layer, groups["A"]["h_l"], groups["A"]["att"],
                            groups["A"]["pos"], groups["A"]["keep"], groups["A"]["sc_idx"],
                            groups["A"]["base_lsm"], all_deltas, batch_units)
    resB = run_forward_half(lm, W_U, softcap, layer, groups["B"]["h_l"], groups["B"]["att"],
                            groups["B"]["pos"], groups["B"]["keep"], groups["B"]["sc_idx"],
                            groups["B"]["base_lsm"], all_deltas, batch_units)

    cov = np.column_stack([
        np.ones(U_tot, np.float64),
        0.5 * (resA["kl_pos"] + resB["kl_pos"]),
        0.5 * (resA["kl_neg"] + resB["kl_neg"]),
        np.log(0.5 * (resA["odd_l2"] + resB["odd_l2"]) + 1e-8),
        0.5 * (resA["even_l2"] / (resA["odd_l2"] + 1e-8) + resB["even_l2"] / (resB["odd_l2"] + 1e-8)),
    ])

    # Run all 4 mask/norm combinations against columns controls (plus check vs isotropic controls)
    configs = {}
    primary_sc = None
    for m_name, m_gpu in (("LEX", lex_mask_gpu), ("ALL", all_ns_mask_gpu)):
        for n_name, zv in (("L2_cent", False), ("L2_zsc", True)):
            key = f"{m_name}_{n_name}"
            kw = (key == "LEX_L2_cent")
            sc = score_odd_pair_gpu(resA["odd"], resB["odd"], col_mask, cov, m_gpu, zv, dev, keep_words=kw)
            ev_col = eval_split_metrics(sc, neu_mask, col_mask)
            # Also evaluate vs isotropic controls (residualized against isotropic)
            sc_iso = score_odd_pair_gpu(resA["odd"], resB["odd"], col_mask, cov, m_gpu, zv, dev, ctl_pool_mask=iso_mask)
            ev_iso = eval_split_metrics(sc_iso, neu_mask, iso_mask)
            configs[key] = {
                "vs_columns": {k: v for k, v in ev_col.items() if k != "_named_local_idx"},
                "vs_isotropic": {k: v for k, v in ev_iso.items() if k != "_named_local_idx"},
            }
            if kw:
                primary_sc = sc
                primary_named_local = ev_col["_named_local_idx"]

    # 50 random half-control null splits on primary (LEX_L2_cent) and ALL_L2_cent
    null_splits_res: Dict[str, Any] = {}
    if do_null_splits > 0 and n_cols >= 64:
        rng = np.random.default_rng(seed + layer)
        ctlA = resA["odd"][:n_cols]
        ctlB = resB["odd"][:n_cols]
        cov_c = cov[:n_cols]
        for key, m_gpu in (("LEX_L2_cent", lex_mask_gpu),):
            aucs, fdp05_cnts = [], []
            for sp in range(do_null_splits):
                perm = rng.permutation(n_cols)
                half = n_cols // 2
                fake_m = np.zeros(n_cols, dtype=bool); fake_m[perm[:half]] = True
                ref_m = ~fake_m
                sc_sp = score_odd_pair_gpu(ctlA, ctlB, ref_m, cov_c, m_gpu, False, dev, ctl_pool_mask=ref_m)
                r_abr, r_bar, Ts = sc_sp["r_ab_res"], sc_sp["r_ba_res"], sc_sp["T"]
                pok_f = (r_abr[fake_m] > 0) & (r_bar[fake_m] > 0)
                aucs.append(auc(Ts[fake_m], Ts[ref_m]))
                fdp05_cnts.append(empirical_fdp(Ts[fake_m], pok_f, int(fake_m.sum()), Ts[ref_m], 0.05)["count"])
            null_splits_res[key] = {
                "n": do_null_splits,
                "auc_mean": round(float(np.mean(aucs)), 4),
                "auc_min": round(float(np.min(aucs)), 4),
                "auc_max": round(float(np.max(aucs)), 4),
                "fdp05_false_mean": round(float(np.mean(fdp05_cnts)), 2),
                "fdp05_false_max": int(np.max(fdp05_cnts)),
            }

    # Build top replicated piece summaries & examples
    T_neu = primary_sc["T"][neu_mask]
    top_order = np.argsort(-T_neu)
    named_ids = [neurons[int(i)] for i in primary_named_local]
    named_set = set(named_ids)

    def unit_pieces_summary(global_u: int) -> Dict[str, Any]:
        # Replicated positive & negative pieces from A->B and B->A
        rep_pos, rep_neg = [], []
        seen_p, seen_n = set(), set()
        for k in range(5):
            pid = int(primary_sc["topA_p"][global_u, k])
            if primary_sc["posA_onB"][global_u, k] > 0 and pid not in seen_p:
                seen_p.add(pid); rep_pos.append((raw_str[pid], round(float(primary_sc["posA_onB"][global_u, k]), 2), pid))
            pid2 = int(primary_sc["topB_p"][global_u, k])
            if primary_sc["posB_onA"][global_u, k] > 0 and pid2 not in seen_p:
                seen_p.add(pid2); rep_pos.append((raw_str[pid2], round(float(primary_sc["posB_onA"][global_u, k]), 2), pid2))
            nid = int(primary_sc["topA_n"][global_u, k])
            if primary_sc["negA_onB"][global_u, k] < 0 and nid not in seen_n:
                seen_n.add(nid); rep_neg.append((raw_str[nid], round(float(-primary_sc["negA_onB"][global_u, k]), 2), nid))
            nid2 = int(primary_sc["topB_n"][global_u, k])
            if primary_sc["negB_onA"][global_u, k] < 0 and nid2 not in seen_n:
                seen_n.add(nid2); rep_neg.append((raw_str[nid2], round(float(-primary_sc["negB_onA"][global_u, k]), 2), nid2))
        rep_pos.sort(key=lambda x: -x[1])
        rep_neg.sort(key=lambda x: -x[1])
        return {"pos": rep_pos[:5], "neg": rep_neg[:5]}

    examples = []
    for loc_i in top_order[:25]:
        gu = n_cols + n_iso + int(loc_i)
        nid = neurons[int(loc_i)]
        ps = unit_pieces_summary(gu)
        examples.append({
            "neuron": nid, "named": nid in named_set, "T": round(float(T_neu[loc_i]), 3),
            "r_ab_res": round(float(primary_sc["r_ab_res"][gu]), 3),
            "r_ba_res": round(float(primary_sc["r_ba_res"][gu]), 3),
            "pos": [(w0, sc) for w0, sc, _ in ps["pos"]],
            "neg": [(w0, sc) for w0, sc, _ in ps["neg"]],
        })
    ctl_order = np.argsort(-primary_sc["T"][col_mask])[:10]
    ctl_examples = []
    for ci in ctl_order:
        gu = int(ci)
        ps = unit_pieces_summary(gu)
        ctl_examples.append({
            "control": gu, "T": round(float(primary_sc["T"][gu]), 3),
            "pos": [(w0, sc) for w0, sc, _ in ps["pos"]],
            "neg": [(w0, sc) for w0, sc, _ in ps["neg"]],
        })

    # Optional Half C out-of-sample replication
    half_c_res = None
    if do_half_c:
        eval_local = primary_named_local[:250] if len(primary_named_local) >= 20 else top_order[:50]
        if len(eval_local) > 0:
            sel_gu = n_cols + n_iso + eval_local
            c_ctl_idx = np.arange(min(256, n_cols))
            sub_deltas = torch.cat([all_deltas[c_ctl_idx], all_deltas[sel_gu]], dim=0)
            resC = run_forward_half(lm, W_U, softcap, layer, groups["C"]["h_l"], groups["C"]["att"],
                                    groups["C"]["pos"], groups["C"]["keep"], groups["C"]["sc_idx"],
                                    groups["C"]["base_lsm"], sub_deltas, batch_units)
            oddC = resC["odd"].to(device=dev, dtype=torch.float32)  # [C_c + K, V]
            nc_c = len(c_ctl_idx)
            muC = oddC[:nc_c].mean(0)
            mask_f = lex_mask_gpu.float()
            M_mask = float(mask_f.sum().item())
            oddC_cent = (oddC - muC[None]) * mask_f[None]
            normC = oddC_cent / torch.sqrt((oddC_cent * oddC_cent).sum(-1, keepdim=True) / M_mask + 1e-8)
            normC_ctl = normC[:nc_c].cpu().numpy()
            normC_neu = normC[nc_c:].cpu().numpy()

            sc_c_vals, sc_c_paired_ctl, p_c_list = [], [], []
            for j, loc_i in enumerate(eval_local):
                gu = n_cols + n_iso + int(loc_i)
                ps = unit_pieces_summary(gu)
                p_ids = [pid for _, _, pid in ps["pos"]]
                n_ids = [nid0 for _, _, nid0 in ps["neg"]]
                denom = max(len(p_ids) + len(n_ids), 1)
                val_n = float((normC_neu[j, p_ids].sum() - normC_neu[j, n_ids].sum()) / denom) if denom > 0 else 0.0
                ctl_scores = (normC_ctl[:, p_ids].sum(1) - normC_ctl[:, n_ids].sum(1)) / denom
                sc_c_vals.append(val_n)
                sc_c_paired_ctl.append(float(ctl_scores[j % nc_c]))
                ge = int(np.sum(ctl_scores >= val_n))
                p_c_list.append((1.0 + ge) / (nc_c + 1.0))
            half_c_res = {
                "evaluated": len(eval_local),
                "were_fdp05_named": len(primary_named_local) >= 20,
                "half_c_auc": round(auc(sc_c_vals, sc_c_paired_ctl), 4),
                "frac_gt_0": round(float(np.mean([v > 0 for v in sc_c_vals])), 4),
                "frac_p05_vs_ctl": round(float(np.mean([p <= 0.05 for p in p_c_list])), 4),
                "median_score_C": round(float(np.median(sc_c_vals)), 3),
            }
            del oddC, oddC_cent, normC, resC

    # Cross-checks
    cross = {}
    if cross_check_l45 is not None:
        matched_words, matched_units, shuf_matched_words = 0, 0, 0
        rng_cc = np.random.default_rng(0)
        cand_idx_list = list(primary_named_local) if len(primary_named_local) > 0 else list(top_order[:100])
        shuf_order = rng_cc.permutation(len(cand_idx_list))
        for j, loc_i in enumerate(cand_idx_list):
            nid = neurons[int(loc_i)]
            gu = n_cols + n_iso + int(loc_i)
            ps = unit_pieces_summary(gu)
            p_words = {clean_str[pid] for _, _, pid in ps["pos"] + ps["neg"]}
            dec_words = cross_check_l45.get(nid, set())
            if dec_words and (p_words & dec_words):
                matched_units += 1
                matched_words += len(p_words & dec_words)
            sh_nid = neurons[int(cand_idx_list[shuf_order[j]])]
            sh_dec = cross_check_l45.get(sh_nid, set())
            if sh_dec and (p_words & sh_dec):
                shuf_matched_words += 1
        cross["l45_decoding_overlap"] = {
            "checked": len(cand_idx_list),
            "units_with_word_match": matched_units,
            "frac_units_match": round(matched_units / max(len(cand_idx_list), 1), 4),
            "shuffled_units_match": shuf_matched_words,
            "frac_shuffled_match": round(shuf_matched_words / max(len(cand_idx_list), 1), 4),
        }

    if cross_check_l55_named is not None:
        overlap = len(named_set & cross_check_l55_named)
        cross["l55_decoding_overlap"] = {
            "forced_named": len(named_set),
            "decoding_named": len(cross_check_l55_named),
            "overlap": overlap,
            "recall_of_decoding_named": round(overlap / max(len(cross_check_l55_named), 1), 4),
        }

    if check_explore29:
        rank_of = {neurons[int(loc_i)]: idx + 1 for idx, loc_i in enumerate(top_order)}
        l29_rows = []
        nid_to_loc = {nid: idx for idx, nid in enumerate(neurons)}
        for nid in L29_EXPLORE_NEURONS:
            loc_i = nid_to_loc.get(nid)
            if loc_i is None:
                continue
            gu = n_cols + n_iso + loc_i
            ps = unit_pieces_summary(gu)
            l29_rows.append({
                "neuron": nid, "rank_of_1022": rank_of[nid], "named_fdp05": nid in named_set,
                "T": round(float(T_neu[loc_i]), 3),
                "pos": [w0 for w0, _, _ in ps["pos"]],
                "neg": [w0 for w0, _, _ in ps["neg"]],
            })
        cross["l29_explore20"] = l29_rows

    # Save example file for this layer
    ex_lines = [f"Layer {layer} forced_odd (dose={dose_ceil:.3f} ceil, KL p50={float(np.median(0.5*(resA['kl_pos'][neu_mask]+resA['kl_neg'][neu_mask]))):.3f}), "
                f"FDP<=0.05 named: {configs['LEX_L2_cent']['vs_columns']['fdp05']['count']}/{len(neurons)}"]
    ex_lines.append("\nTop 25 neurons:")
    for e in examples:
        ex_lines.append(f"  n={e['neuron']:6d} named={str(e['named']):<5} T={e['T']:6.2f} "
                        f"+{e['pos'][:4]}  -{e['neg'][:4]}")
    ex_lines.append("\nTop 10 columns controls:")
    for e in ctl_examples:
        ex_lines.append(f"  c={e['control']:6d} T={e['T']:6.2f} +{e['pos'][:4]}  -{e['neg'][:4]}")
    (OUT_DIR / f"forced_odd_L{layer}_examples.txt").write_text("\n".join(ex_lines) + "\n")

    kl_n = 0.5 * (0.5 * (resA["kl_pos"][neu_mask] + resA["kl_neg"][neu_mask]) + 0.5 * (resB["kl_pos"][neu_mask] + resB["kl_neg"][neu_mask]))
    kl_c = 0.5 * (0.5 * (resA["kl_pos"][col_mask] + resA["kl_neg"][col_mask]) + 0.5 * (resB["kl_pos"][col_mask] + resB["kl_neg"][col_mask]))
    eo_n = cov[neu_mask, 4]
    eo_c = cov[col_mask, 4]

    summary = {
        "layer": layer,
        "neurons": len(neurons),
        "columns_controls": n_cols,
        "isotropic_controls": n_iso,
        "dose_ceil": dose_ceil,
        "elapsed_s": round(time.time() - t0, 1),
        "kl_neurons": {"p10": round(float(np.percentile(kl_n, 10)), 4), "p50": round(float(np.percentile(kl_n, 50)), 4),
                       "p90": round(float(np.percentile(kl_n, 90)), 4)},
        "kl_columns": {"p10": round(float(np.percentile(kl_c, 10)), 4), "p50": round(float(np.percentile(kl_c, 50)), 4),
                       "p90": round(float(np.percentile(kl_c, 90)), 4)},
        "even_odd_ratio": {"neurons_p50": round(float(np.median(eo_n)), 3), "columns_p50": round(float(np.median(eo_c)), 3)},
        "configs": configs,
        "null_splits": null_splits_res,
        "half_c": half_c_res,
        "cross_check": cross,
    }
    del resA, resB, all_deltas, groups
    torch.cuda.empty_cache()
    gc.collect()
    return summary


def run_l29_ablations(
    rt: Any, lm: nn.Module, w: ao.Weights, texts: Dict[str, Any],
    lex_mask_gpu: torch.Tensor, raw_str: List[str], l29_trial_neurons: List[int], seed: int = 11,
) -> Dict[str, Any]:
    print("\n=== L29 Mini-Ablations (256 neurons + 256 columns controls) ===", flush=True)
    layer = 29
    dev = model_device(lm)
    N_ff = w.d_ff
    W_U = rt.lm_head_weight
    softcap = ao.softcap_of(rt.hf_model)
    bos_id = rt.tokenizer.bos_token_id
    rest = [n for n in l29_trial_neurons if n not in set(L29_EXPLORE_NEURONS)]
    sub_neu = list(L29_EXPLORE_NEURONS) + rest[: (256 - len(L29_EXPLORE_NEURONS))]
    n_neu, n_ctl = len(sub_neu), 256

    w_post_np = w.norm(layer, "post_feedforward").cpu().numpy()
    writes_all = np.concatenate([w.write_dirs(layer, list(range(a, min(a + 4096, N_ff)))).cpu().numpy()
                                 for a in range(0, N_ff, 4096)])
    col_dirs = cz.control_dirs(seed + 77, n_ctl, w_post_np, "columns", writes_all)
    del writes_all
    ref_ceils = w.write_ceiling(layer, cz.control_refs(seed + 77, n_ctl, N_ff)).cpu().numpy()
    u_neu = w.write_dirs(layer, sub_neu)
    c_neu = w.write_ceiling(layer, sub_neu)

    # Precompute h_29 & masks for A, B
    groups = {}
    for gnm in ("A", "B"):
        qlist = texts["groups"][gnm]
        inp, att, _, _ = pad_right([texts["seqs"][q] for q in qlist], dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        keep_all = ao.keep_mask(inp, att, bos_id, True).to(W_U.dtype)
        # Answer-only mask: clear positions < prompt_len
        keep_ans = keep_all.clone()
        for bi, q in enumerate(qlist):
            keep_ans[bi, : texts["prompt_lens"][q]] = 0
        seen: Dict[str, torch.Tensor] = {}
        hk = lm.layers[layer].register_forward_hook(lambda _m, _a, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            full_out = lm(input_ids=inp, attention_mask=att, use_cache=False)
        finally:
            hk.remove()
        h_l = seen["h"]
        sc_idx = torch.tensor([texts["scored_pos"][q] for q in qlist], dtype=torch.long, device=dev)
        h_rows = torch.arange(len(qlist), device=dev)[:, None].expand(len(qlist), sc_idx.shape[1])
        h_sc = full_out.last_hidden_state[h_rows, sc_idx]
        V = int(W_U.shape[0])
        base_lsm = torch.empty((len(qlist), sc_idx.shape[1], V), dtype=torch.float32, device=dev)
        hf = h_sc.reshape(-1, h_l.shape[-1])
        for c0 in range(0, hf.shape[0], 64):
            base_lsm.view(-1, V)[c0 : c0 + 64] = F.log_softmax(logits_fp32(hf[c0 : c0 + 64], W_U), dim=-1)
        groups[gnm] = {"h_l": h_l, "att": att, "pos": pos, "keep_all": keep_all, "keep_ans": keep_ans, "sc_idx": sc_idx, "base_lsm": base_lsm}
        del full_out, h_sc, hf

    U_tot = n_ctl + n_neu
    ctl_mask = np.zeros(U_tot, dtype=bool); ctl_mask[:n_ctl] = True
    neu_mask = ~ctl_mask

    ablation_specs = [
        ("dose_1.67_all", 1.67, "keep_all", True),
        ("dose_1.67_ans_only", 1.67, "keep_ans", False),
        ("dose_1.15_all", 1.15, "keep_all", False),
        ("dose_2.05_all", 2.05, "keep_all", False),
    ]
    ablation_out = {}
    for nm, dose_c, keep_key, keep_raw in ablation_specs:
        dc = torch.from_numpy((dose_c * ref_ceils)[:, None] * col_dirs).to(dev)
        dn = (dose_c * c_neu)[:, None] * u_neu
        deltas = torch.cat([dc, dn], dim=0)
        rA = run_forward_half(lm, W_U, softcap, layer, groups["A"]["h_l"], groups["A"]["att"], groups["A"]["pos"],
                              groups["A"][keep_key], groups["A"]["sc_idx"], groups["A"]["base_lsm"], deltas, 4, return_raw_signs=keep_raw)
        rB = run_forward_half(lm, W_U, softcap, layer, groups["B"]["h_l"], groups["B"]["att"], groups["B"]["pos"],
                              groups["B"][keep_key], groups["B"]["sc_idx"], groups["B"]["base_lsm"], deltas, 4, return_raw_signs=keep_raw)
        cov = np.column_stack([
            np.ones(U_tot, np.float64),
            0.5 * (rA["kl_pos"] + rB["kl_pos"]),
            0.5 * (rA["kl_neg"] + rB["kl_neg"]),
            np.log(0.5 * (rA["odd_l2"] + rB["odd_l2"]) + 1e-8),
            0.5 * (rA["even_l2"] / (rA["odd_l2"] + 1e-8) + rB["even_l2"] / (rB["odd_l2"] + 1e-8)),
        ])
        sc_odd = score_odd_pair_gpu(rA["odd"], rB["odd"], ctl_mask, cov, lex_mask_gpu, False, dev)
        ev_odd = eval_split_metrics(sc_odd, neu_mask, ctl_mask)
        kl_n = 0.5 * (0.5 * (rA["kl_pos"][neu_mask] + rA["kl_neg"][neu_mask]) + 0.5 * (rB["kl_pos"][neu_mask] + rB["kl_neg"][neu_mask]))
        entry = {
            "dose_ceil": dose_c, "kl_p50": round(float(np.median(kl_n)), 4),
            "even_odd_ratio_p50": round(float(np.median(cov[neu_mask, 4])), 3),
            "odd": {k: v for k, v in ev_odd.items() if k != "_named_local_idx"},
        }
        if keep_raw:
            sc_pos = score_odd_pair_gpu(rA["raw_pos"], rB["raw_pos"], ctl_mask, cov, lex_mask_gpu, False, dev, keep_words=True)
            sc_neg = score_odd_pair_gpu(rA["raw_neg"], rB["raw_neg"], ctl_mask, cov, lex_mask_gpu, False, dev, keep_words=True)
            ev_pos = eval_split_metrics(sc_pos, neu_mask, ctl_mask)
            ev_neg = eval_split_metrics(sc_neg, neu_mask, ctl_mask)
            sc_odd_w = score_odd_pair_gpu(rA["odd"], rB["odd"], ctl_mask, cov, lex_mask_gpu, False, dev, keep_words=True)
            def shared_stats(sc_obj):
                cnt5 = collections.Counter(int(sc_obj["topA_p"][n_ctl + i, j]) for i in range(n_neu) for j in range(5))
                shared = {k for k, v in cnt5.items() if v >= max(5, int(0.02 * n_neu))}
                mean_in_top5 = float(np.mean([sum(1 for j in range(5) if int(sc_obj["topA_p"][n_ctl + i, j]) in shared) for i in range(n_neu)]))
                return mean_in_top5, [(raw_str[k], v) for k, v in cnt5.most_common(5)]
            raw_sh_n, raw_comm = shared_stats(sc_pos)
            odd_sh_n, odd_comm = shared_stats(sc_odd_w)
            entry["raw_plus"] = {k: v for k, v in ev_pos.items() if k != "_named_local_idx"}
            entry["raw_minus"] = {k: v for k, v in ev_neg.items() if k != "_named_local_idx"}
            entry["raw_pos_shared_top5_of5"] = round(raw_sh_n, 3)
            entry["odd_shared_top5_of5"] = round(odd_sh_n, 3)
            entry["raw_pos_top5_commonest"] = raw_comm
            entry["odd_top5_commonest"] = odd_comm
        ablation_out[nm] = entry
        del rA, rB, deltas
        torch.cuda.empty_cache()

    del groups
    gc.collect()
    return ablation_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    w = ao.Weights(lm, rt)

    # Part 2A
    run_part2a(rt, lm, w)

    # Part 2B setup + preflight
    print("\n=== Part 2B: forced_odd Setup & Preflight ===", flush=True)
    texts = build_fixed_texts(rt, lm)
    lex_cpu, all_ns_cpu, clean_str, raw_str = build_vocab_masks(rt)
    dev = model_device(lm)
    lex_gpu = lex_cpu.to(dev)
    all_ns_gpu = all_ns_cpu.to(dev)
    print(f"Masks: LEX={int(lex_cpu.sum()):,}, ALL_NONSPECIAL={int(all_ns_cpu.sum()):,}", flush=True)

    # Pre-flight: check base log-prob on all 18 texts & forward_from_layer exactness at L29 and L59
    W_U = rt.lm_head_weight
    for gnm in ("A", "B", "C"):
        qlist = texts["groups"][gnm]
        seq_list = [texts["seqs"][q] for q in qlist]
        inp, att, _, _ = pad_right(seq_list, dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        cap: Dict[int, torch.Tensor] = {}
        hks = [
            lm.layers[29].register_forward_hook(lambda _m, _a, o: cap.__setitem__(29, (o[0] if isinstance(o, tuple) else o).detach())),
            lm.layers[59].register_forward_hook(lambda _m, _a, o: cap.__setitem__(59, (o[0] if isinstance(o, tuple) else o).detach())),
        ]
        try:
            full = lm(input_ids=inp, attention_mask=att, use_cache=False)
        finally:
            for hk in hks:
                hk.remove()
        h_ref = full.last_hidden_state
        # Check base answer logprob across answer positions
        min_lp = 0.0
        for bi, q in enumerate(qlist):
            p_len = texts["prompt_lens"][q]
            L_q = len(texts["seqs"][q])
            l_ans = logits_fp32(h_ref[bi, p_len - 1 : L_q - 1], W_U)
            lp_ans = F.log_softmax(l_ans, dim=-1).gather(-1, inp[bi, p_len:L_q, None]).mean().item()
            min_lp = min(min_lp, lp_ans)
        # Check forward_from_layer exactness at L29 and L59
        r29 = ao.forward_from_layer(lm, 30, cap[29], att, pos)
        r59 = ao.forward_from_layer(lm, 60, cap[59], att, pos)
        diff29 = float((r29 - h_ref).abs().max().item())
        diff59 = float((r59 - h_ref).abs().max().item())
        print(f"Preflight group {gnm}: min_ans_lp={min_lp:.3f}, max_diff(L29)={diff29:.6f}, max_diff(L59)={diff59:.6f}", flush=True)
        if min_lp < -0.5 or diff29 > 1e-2 or diff59 > 1e-2:
            raise RuntimeError(f"Preflight failed on group {gnm}: lp={min_lp}, diff29={diff29}, diff59={diff59}")
        del full, h_ref, r29, r59, cap

    # Load cross-check references (L45 words, L55 named, L55/L29 trial neurons)
    l45_words: Dict[int, Set[str]] = {}
    with open(MAIN_DATA / "census/runs/L45-break_rel.units.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "neuron":
                ws = set(cz.shown_words(r, k=5))
                if ws:
                    l45_words[int(r["neuron"])] = ws
    l55_verdict = json.loads((WORKTREE / "experiments/results/census_things_walk/L55_verdict.whole.json").read_text())
    l55_named_set = set(int(x) for x in l55_verdict["readings"]["48"]["named"])
    l55_trial_ns = extract_trial_neurons(55)
    l29_trial_ns = extract_trial_neurons(29)
    rng_n = np.random.default_rng(11)
    l59_ns = sorted(int(x) for x in rng_n.choice(w.d_ff, 1022, replace=False))
    l45_ns = sorted(int(x) for x in np.random.default_rng(12).choice(w.d_ff, 1022, replace=False))

    summaries: List[Dict[str, Any]] = []

    def save_progress(abl: Optional[Dict[str, Any]] = None) -> None:
        out = {"elapsed_total_s": round(time.time() - t_start, 1), "layers": summaries}
        if abl is not None:
            out["l29_ablations"] = abl
        (OUT_DIR / "forced_odd_summary.json").write_text(json.dumps(out, indent=1) + "\n")

    # 1. Anchor layer L45 FIRST (fail-fast guard)
    s45 = run_layer_forced_odd(rt, lm, w, texts, lex_gpu, all_ns_gpu, clean_str, raw_str,
                               layer=45, initial_dose_ceil=2.03, neurons=l45_ns, n_cols=1024, n_iso=256,
                               seed=1045, do_half_c=True, do_null_splits=50, cross_check_l45=l45_words)
    summaries.append(s45)
    save_progress()
    ov45 = s45["cross_check"]["l45_decoding_overlap"]
    c45 = s45["configs"]["LEX_L2_cent"]["vs_columns"]
    print(f"Anchor L45: T_auc={c45['auc_T']}, FDP05={c45['fdp05']['count']}, L45 match={ov45['frac_units_match']} (shuf {ov45['frac_shuffled_match']})", flush=True)
    if c45["auc_T"] < 0.55 or ov45["frac_units_match"] <= ov45["frac_shuffled_match"] * 1.3:
        raise RuntimeError(f"Re-plan guard triggered at L45 anchor: {c45} / {ov45}")

    # 2. L59 and L55 (top layers)
    s59 = run_layer_forced_odd(rt, lm, w, texts, lex_gpu, all_ns_gpu, clean_str, raw_str,
                               layer=59, initial_dose_ceil=0.55, neurons=l59_ns, n_cols=1024, n_iso=256,
                               seed=1059, do_half_c=False, do_null_splits=50)
    summaries.append(s59); save_progress()

    s55 = run_layer_forced_odd(rt, lm, w, texts, lex_gpu, all_ns_gpu, clean_str, raw_str,
                               layer=55, initial_dose_ceil=1.34, neurons=l55_trial_ns, n_cols=1024, n_iso=256,
                               seed=1055, do_half_c=True, do_null_splits=50, cross_check_l55_named=l55_named_set)
    summaries.append(s55); save_progress()

    # 3. L29 (exact trial neurons) + L29 Mini-Ablations
    s29 = run_layer_forced_odd(rt, lm, w, texts, lex_gpu, all_ns_gpu, clean_str, raw_str,
                               layer=29, initial_dose_ceil=1.67, neurons=l29_trial_ns, n_cols=1024, n_iso=256,
                               seed=1029, do_half_c=True, do_null_splits=50, check_explore29=True)
    summaries.append(s29); save_progress()

    abl29 = run_l29_ablations(rt, lm, w, texts, lex_gpu, raw_str, l29_trial_ns)
    save_progress(abl29)

    # 4. Downward floor probes: L25, L20 (and L14 if L20 stays strong)
    l25_ns = sorted(int(x) for x in np.random.default_rng(1025).choice(w.d_ff, 256, replace=False))
    s25 = run_layer_forced_odd(rt, lm, w, texts, lex_gpu, all_ns_gpu, clean_str, raw_str,
                               layer=25, initial_dose_ceil=1.55, neurons=l25_ns, n_cols=256, n_iso=128,
                               seed=2025, do_half_c=False, do_null_splits=30)
    summaries.append(s25); save_progress(abl29)

    l20_ns = sorted(int(x) for x in np.random.default_rng(1020).choice(w.d_ff, 256, replace=False))
    s20 = run_layer_forced_odd(rt, lm, w, texts, lex_gpu, all_ns_gpu, clean_str, raw_str,
                               layer=20, initial_dose_ceil=2.00, neurons=l20_ns, n_cols=256, n_iso=128,
                               seed=2020, do_half_c=False, do_null_splits=30)
    summaries.append(s20); save_progress(abl29)

    c20 = s20["configs"]["LEX_L2_cent"]["vs_columns"]
    if c20["auc_T"] >= 0.57 or c20["fdp05"]["count"] >= 10:
        l14_ns = sorted(int(x) for x in np.random.default_rng(1014).choice(w.d_ff, 256, replace=False))
        s14 = run_layer_forced_odd(rt, lm, w, texts, lex_gpu, all_ns_gpu, clean_str, raw_str,
                                   layer=14, initial_dose_ceil=2.18, neurons=l14_ns, n_cols=256, n_iso=128,
                                   seed=2014, do_half_c=False, do_null_splits=30)
        summaries.append(s14); save_progress(abl29)

    # Human-readable summary table
    tlines = [
        "Part 2B: forced_odd Summary (LEX_L2_cent vs columns controls)",
        "-" * 116,
        f"{'L':>3} {'N/C':>9} {'DoseC':>6} {'KLp50':>6} {'EO_n':>5} {'rABr':>6} {'rBAr':>6} {'T_auc':>6} "
        f"{'FDP05':>6} {'FDP10':>6} {'Null05':>6} {'IsoT':>6} {'HalfC_p05':>9}",
    ]
    for s in summaries:
        vc = s["configs"]["LEX_L2_cent"]["vs_columns"]
        vi = s["configs"]["LEX_L2_cent"]["vs_isotropic"]
        ns = s["null_splits"].get("LEX_L2_cent", {})
        hc = s.get("half_c") or {}
        hc_str = f"{hc['frac_p05_vs_ctl']:.3f}" if hc.get("frac_p05_vs_ctl") is not None else "-"
        tlines.append(
            f"{s['layer']:3d} {s['neurons']:4d}/{s['columns_controls']:4d} {s['dose_ceil']:6.3f} {s['kl_neurons']['p50']:6.3f} "
            f"{s['even_odd_ratio']['neurons_p50']:5.2f} {vc['auc_r_AB_res']:6.3f} {vc['auc_r_BA_res']:6.3f} "
            f"{vc['auc_T']:6.3f} {vc['fdp05']['count']:6d} {vc['fdp10']['count']:6d} "
            f"{ns.get('fdp05_false_mean', -1):6.2f} {vi['auc_T']:6.3f} {hc_str:>9}"
        )
    txt_summary = "\n".join(tlines) + "\n"
    (OUT_DIR / "forced_odd_summary.txt").write_text(txt_summary)
    print("\n" + txt_summary, flush=True)


if __name__ == "__main__":
    main()
