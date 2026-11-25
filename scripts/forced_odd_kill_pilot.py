#!/usr/bin/env python3
"""Exploratory Design-A fresh-text forced_odd kill-pilot (L45, L54, L58, L59).

Generates 18 brand-new unseen fixed-continuation texts (A': 6, B': 6, C': 6),
disjoint from HALVES/HALF_C/ALL_PROMPTS. Pushes N=256 sampled neurons plus C_cols=512
and C_iso=512 controls at calibrated kl_p90~0.25 ceiling doses, measuring:
  - cent/LEX_L2_cent and proj/LEX_L2_cent total odd logit shifts;
  - Bidirectional cross-pool contamination (iso->cols and cols->iso at FDP<=0.05);
  - Like-for-like matched Half-C' replication (score_half_c_matched);
  - Probability-weighted Delta p = 0.5*(p(+d)-p(-d)) and base p on picked C' words;
  - Direct-vs-carried decomposition at L45/L54/L58 (subtracting direct unembed shift
    from h_59+/-d BEFORE clamping; None/0 at L59 by construction);
  - Incremental accuracy vs lens.npz at predicting L45/L54 decoding census words.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_ops as ao  # noqa: E402
from backend import forced_odd as fo  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402
from scripts.forced_odd_census import load_decoding_overlap_ref  # noqa: E402

FRESH_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "A": [
        ("compass", "Explain how a compass works."),
        ("turbine", "Explain how a steam turbine works."),
        ("bread", "Give me three tips for baking bread."),
        ("piano", "Give me three tips for practicing piano."),
        ("harbor", "Describe a quiet harbor."),
        ("violin", "Describe a wooden violin."),
    ],
    "B": [
        ("elevator", "Explain how an elevator works."),
        ("telescope", "Explain how a telescope works."),
        ("sketch", "Give me three tips for sketching landscapes."),
        ("swim", "Give me three tips for swimming smoothly."),
        ("lantern", "Describe a paper lantern."),
        ("glacier", "Describe a mountain glacier."),
    ],
    "C": [
        ("barometer", "Explain how a barometer works."),
        ("submarine", "Explain how a submarine works."),
        ("speaking", "Give me three tips for speaking clearly."),
        ("bookshelf", "Give me three tips for organizing books."),
        ("cottage", "Describe a thatched cottage."),
        ("marble", "Describe a striped glass marble."),
    ],
}

FALLBACK_DOSES: Dict[int, float] = {45: 1.225, 54: 1.421, 58: 0.5818, 59: 0.2482}


def _dose_for_layer(layer: int) -> Tuple[float, str]:
    fo_dir = REPO_ROOT / "experiments/results/census_forced_odd"
    for rel, key in (
        (f"whole/L{layer}_forced_odd.meta.json", "dose_ceil"),
        (f"calibration_p90/calib_L{layer}.json", "dose_ceil"),
    ):
        p = fo_dir / rel
        if p.is_file():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if key in obj and obj[key] is not None:
                return float(obj[key]), f"{rel}"
    return float(FALLBACK_DOSES[layer]), "fallback"


@torch.no_grad()
def build_fresh_texts(rt: Any, lm: Any) -> Dict[str, Any]:
    """Generate 18 fresh unpushed greedy continuations for A'/B'/C' prompts."""
    stop_ids = rt.end_of_turn_ids()
    bos_id = rt.tokenizer.bos_token_id
    softcap = ao.softcap_of(rt.hf_model)

    seqs: Dict[str, List[int]] = {}
    prompt_lens: Dict[str, int] = {}
    scored_pos: Dict[str, List[int]] = {}
    groups: Dict[str, List[str]] = {}

    for gnm, items in FRESH_GROUPS.items():
        qnames = [k for k, _ in items]
        prompts = [rt.tokenize_prompt(rt.compose_prompt(txt, chat=True)) for _, txt in items]
        ans: List[List[int]] = [[] for _ in items]
        ended = [False] * len(items)
        for st in ao.write_rows(
            lm,
            rt.lm_head_weight,
            prompts,
            None,
            max_new=48,
            stop_ids=stop_ids,
            bos_id=bos_id,
            spare_bos=True,
            softcap=softcap,
            top_k=0,
        ):
            ans[st.row].append(st.token_id)
            ended[st.row] = ended[st.row] or st.end
        for q, pids, aids, end in zip(qnames, prompts, ans, ended):
            a_clean = aids[:-1] if end else aids
            if a_clean and a_clean[-1] in stop_ids:
                a_clean = a_clean[:-1]
            full = (list(pids) + list(a_clean))[:64]
            p_len = len(pids)
            L_q = len(full)
            cand = list(range(p_len + 3, min(L_q - 1, p_len + 35), 2))
            if len(cand) < 16:
                cand = sorted(set(int(round(x)) for x in np.linspace(p_len + 2, L_q - 2, 16)))[:16]
            seqs[q] = full
            prompt_lens[q] = p_len
            scored_pos[q] = cand[:16]
        groups[gnm] = qnames

    return {"seqs": seqs, "prompt_lens": prompt_lens, "scored_pos": scored_pos, "groups": groups}


@torch.no_grad()
def preflight_with_h59(rt: Any, lm: Any, texts: Dict[str, Any]) -> Dict[str, Any]:
    """Verify answer fluency, forward_from_layer exactness, and h_59 -> final norm equivalence."""
    pf = fo.run_preflight(rt, lm, texts, check_layers=(29, 59))
    dev = model_device(lm)
    for gnm, qlist in texts["groups"].items():
        seq_list = [texts["seqs"][q] for q in qlist]
        inp, att, _, _ = pad_right(seq_list, dev)
        sc_idx = torch.tensor([texts["scored_pos"][q] for q in qlist], dtype=torch.long, device=dev)
        h_rows = torch.arange(len(qlist), device=dev)[:, None].expand(len(qlist), sc_idx.shape[1])
        seen: Dict[str, torch.Tensor] = {}
        hk = lm.layers[59].register_forward_hook(
            lambda _m, _a, o: seen.__setitem__("h59", (o[0] if isinstance(o, tuple) else o).detach())
        )
        try:
            full = lm(input_ids=inp, attention_mask=att, use_cache=False)
        finally:
            hk.remove()
        h59_sc = seen["h59"][h_rows, sc_idx].reshape(1, -1, seen["h59"].shape[-1])
        ones = torch.ones((1, h59_sc.shape[1]), dtype=torch.long, device=dev)
        zeros = torch.zeros((1, h59_sc.shape[1]), dtype=torch.long, device=dev)
        norm_out = ao.forward_from_layer(lm, 60, h59_sc, ones, zeros).reshape(-1, h59_sc.shape[-1])
        ref_out = full.last_hidden_state[h_rows, sc_idx].reshape(-1, h59_sc.shape[-1])
        h59_diff = float((norm_out - ref_out).abs().max().item())
        pf[gnm]["h59_norm_diff"] = round(h59_diff, 6)
        if h59_diff >= 1e-3:
            raise RuntimeError(f"h_59 final norm check failed on {gnm}: diff={h59_diff}")
        del full, seen, inp, att
    return pf


@torch.no_grad()
def prepare_group_cache_split(rt: Any, lm: Any, layer: int, texts: Dict[str, Any], gnm: str) -> Dict[str, Any]:
    """Cache h_l leaving `layer`, h_59 pre-norm at scored positions (when layer < 59), and unpushed stats."""
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    bos_id = rt.tokenizer.bos_token_id
    D = int(W_U.shape[1])
    V = int(W_U.shape[0])

    qlist = texts["groups"][gnm]
    seq_list = [texts["seqs"][q] for q in qlist]
    inp, att, _, _ = pad_right(seq_list, dev)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    keep_m = ao.keep_mask(inp, att, bos_id, True).to(W_U.dtype)
    sc_idx = torch.tensor([texts["scored_pos"][q] for q in qlist], dtype=torch.long, device=dev)
    if not bool(torch.all(keep_m.gather(1, sc_idx) == 1)):
        raise RuntimeError(f"Scored position unexpectedly masked out on group {gnm}")
    h_rows = torch.arange(len(qlist), device=dev)[:, None].expand(len(qlist), sc_idx.shape[1])

    seen: Dict[str, torch.Tensor] = {}
    hks = [
        lm.layers[layer].register_forward_hook(
            lambda _m, _a, o: seen.__setitem__("h_l", (o[0] if isinstance(o, tuple) else o).detach())
        )
    ]
    if layer < 59:
        hks.append(
            lm.layers[59].register_forward_hook(
                lambda _m, _a, o: seen.__setitem__("h_59", (o[0] if isinstance(o, tuple) else o).detach())
            )
        )
    try:
        full_out = lm(input_ids=inp, attention_mask=att, use_cache=False)
    finally:
        for hk in hks:
            hk.remove()

    h_l = seen["h_l"]
    h_59_sc = seen["h_59"][h_rows, sc_idx].detach() if layer < 59 else None
    h_sc = full_out.last_hidden_state[h_rows, sc_idx]
    base_lsm = torch.empty((len(qlist), sc_idx.shape[1], V), dtype=torch.float32, device=dev)
    h_flat = h_sc.reshape(-1, D)
    for c0 in range(0, h_flat.shape[0], 64):
        base_lsm.view(-1, V)[c0 : c0 + 64] = F.log_softmax(logits_fp32(h_flat[c0 : c0 + 64], W_U), dim=-1)
    z0, z0_norm2, base_mean = fo.precompute_base_stats(base_lsm)
    base_p_mean = base_lsm.exp().mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
    del full_out, h_sc, h_flat, inp, seen
    return {
        "h_l": h_l,
        "h_59_sc": h_59_sc,
        "att": att,
        "pos": pos,
        "keep": keep_m,
        "sc_idx": sc_idx,
        "base_lsm": base_lsm,
        "z0": z0,
        "z0_norm2": z0_norm2,
        "base_mean": base_mean,
        "base_p_mean": base_p_mean,
    }


@torch.no_grad()
def run_forward_half_split(
    lm: Any,
    W_U: torch.Tensor,
    layer: int,
    grp: Dict[str, Any],
    deltas: torch.Tensor,
    batch_units: int,
    direct_scale: float = 1.0,
    clamp_val: float = fo.CLAMP_ODD_DEFAULT,
    keep_dp: bool = False,
) -> Dict[str, Any]:
    """Run +/-deltas on one group, computing total (cent+proj), direct, carried, and Delta p."""
    h_l = grp["h_l"]
    h_59_sc = grp["h_59_sc"]
    att = grp["att"]
    pos = grp["pos"]
    keep_m = grp["keep"]
    sc_idx = grp["sc_idx"]
    base_lsm = grp["base_lsm"]
    z0 = grp["z0"]
    z0_norm2 = grp["z0_norm2"]
    base_mean = grp["base_mean"]

    dev = h_l.device
    M, D = deltas.shape
    H, T, _ = h_l.shape
    S = sc_idx.shape[1]
    V = int(W_U.shape[0])
    has_split = layer < 59 and h_59_sc is not None

    odd_cent = torch.empty((M, V), dtype=torch.float16, device="cpu")
    odd_proj = torch.empty((M, V), dtype=torch.float16, device="cpu")
    odd_car = torch.empty((M, V), dtype=torch.float16, device="cpu") if has_split else None
    dp_mean = np.empty((M, V), dtype=np.float32) if keep_dp else None
    p_pos_mean = np.empty((M, V), dtype=np.float32) if keep_dp else None

    kl_pos = np.zeros(M, np.float32)
    kl_neg = np.zeros(M, np.float32)
    even_l2 = np.zeros(M, np.float32)
    odd_l2_cent = np.zeros(M, np.float32)
    odd_l2_proj = np.zeros(M, np.float32)
    odd_l2_car = np.zeros(M, np.float32) if has_split else None
    alpha_abs = np.zeros(M, np.float32)
    carried_l2_frac = np.full(M, np.nan, dtype=np.float32)
    cos_tot_dir = np.full(M, np.nan, dtype=np.float32)
    cos_tot_car = np.full(M, np.nan, dtype=np.float32)

    h_rows = torch.arange(H, device=dev)[:, None].expand(H, S)

    for s in range(0, M, batch_units):
        db = deltas[s : s + batch_units]
        b = db.shape[0]
        d_exp = db.to(h_l.dtype)[:, None, None, :] * keep_m[None, :, :, None]
        hp = (h_l[None] + d_exp).reshape(b * H, T, D)
        hn = (h_l[None] - d_exp).reshape(b * H, T, D)
        hs = torch.cat([hp, hn], dim=0)
        del hp, hn, d_exp

        att_r = att.repeat(2 * b, 1)
        pos_r = pos.repeat(2 * b, 1)
        out = ao.forward_from_layer(lm, layer + 1, hs, att_r, pos_r)
        del hs, att_r, pos_r

        sel = out.view(2 * b, H, T, D)[:, h_rows, sc_idx, :].reshape(2 * b * H * S, D)
        del out
        lsm_t = torch.empty((2 * b * H * S, V), dtype=torch.float32, device=dev)
        for c0 in range(0, sel.shape[0], 256):
            lsm_t[c0 : c0 + 256] = F.log_softmax(logits_fp32(sel[c0 : c0 + 256], W_U), dim=-1)
        del sel
        lsm_t = lsm_t.view(2, b, H, S, V)
        lt_p, lt_n = lsm_t[0], lsm_t[1]

        for ui in range(b):
            lp_u, ln_u = lt_p[ui], lt_n[ui]
            kl_pos[s + ui] = float((lp_u.exp() * (lp_u - base_lsm)).sum(-1).mean().item())
            kl_neg[s + ui] = float((ln_u.exp() * (ln_u - base_lsm)).sum(-1).mean().item())

        if keep_dp and dp_mean is not None and p_pos_mean is not None:
            p_p = lt_p.exp()
            p_n = lt_n.exp()
            p_pos_mean[s : s + b] = p_p.mean(dim=(1, 2)).cpu().numpy()
            dp_mean[s : s + b] = (0.5 * (p_p - p_n)).mean(dim=(1, 2)).cpu().numpy()
            del p_p, p_n

        even = 0.5 * (lt_p.mean(dim=(1, 2)) + lt_n.mean(dim=(1, 2))) - base_mean[None]
        even.sub_(even.mean(dim=-1, keepdim=True))
        even_l2[s : s + b] = even.norm(dim=-1).cpu().numpy().astype(np.float32)
        del even

        # Raw centered total antisymmetric shift [b, H, S, V] before clamping
        lt_p.sub_(lt_n).mul_(0.5)
        lt_p.sub_(lt_p.mean(dim=-1, keepdim=True))
        del lt_n

        alpha = (lt_p * z0[None]).sum(dim=-1, keepdim=True) / z0_norm2[None]
        alpha_abs[s : s + b] = alpha.abs().mean(dim=(1, 2, 3)).cpu().numpy().astype(np.float32)

        for ui in range(b):
            oc = lt_p[ui].clamp(-clamp_val, clamp_val).mean(dim=(0, 1))
            odd_cent[s + ui] = oc.half().cpu()
            odd_l2_cent[s + ui] = float(oc.norm().item())
            op_raw = lt_p[ui] - alpha[ui] * z0
            op = op_raw.clamp_(-clamp_val, clamp_val).mean(dim=(0, 1))
            odd_proj[s + ui] = op.half().cpu()
            odd_l2_proj[s + ui] = float(op.norm().item())
            del op_raw

        del alpha

        if has_split and h_59_sc is not None and odd_car is not None and odd_l2_car is not None:
            d_sc = (db * float(direct_scale)).to(h_59_sc.dtype)[:, None, None, :]
            hp59 = (h_59_sc[None] + d_sc).reshape(b, H * S, D)
            hn59 = (h_59_sc[None] - d_sc).reshape(b, H * S, D)
            h59_cat = torch.cat([hp59, hn59], dim=0)
            del hp59, hn59, d_sc
            ones_d = torch.ones((2 * b, H * S), dtype=torch.long, device=dev)
            zeros_d = torch.zeros((2 * b, H * S), dtype=torch.long, device=dev)
            out_dir = ao.forward_from_layer(lm, 60, h59_cat, ones_d, zeros_d).reshape(-1, D)
            del h59_cat, ones_d, zeros_d
            lsm_d = torch.empty((2 * b * H * S, V), dtype=torch.float32, device=dev)
            for c0 in range(0, out_dir.shape[0], 256):
                lsm_d[c0 : c0 + 256] = F.log_softmax(logits_fp32(out_dir[c0 : c0 + 256], W_U), dim=-1)
            del out_dir
            lsm_d = lsm_d.view(2, b, H, S, V)
            raw_d = 0.5 * (lsm_d[0] - lsm_d[1])
            raw_d.sub_(raw_d.mean(dim=-1, keepdim=True))
            del lsm_d

            # Carried residual = raw_total - raw_direct BEFORE clamp & averaging
            m_tot = lt_p.mean(dim=(1, 2))
            m_dir = raw_d.mean(dim=(1, 2))
            raw_car = lt_p - raw_d
            m_car = raw_car.mean(dim=(1, 2))

            n_tot = m_tot.norm(dim=-1).clamp(min=1e-8)
            n_dir = m_dir.norm(dim=-1).clamp(min=1e-8)
            n_car = m_car.norm(dim=-1).clamp(min=1e-8)
            carried_l2_frac[s : s + b] = (n_car / (n_dir + n_car + 1e-8)).cpu().numpy()
            cos_tot_dir[s : s + b] = ((m_tot * m_dir).sum(-1) / (n_tot * n_dir)).cpu().numpy()
            cos_tot_car[s : s + b] = ((m_tot * m_car).sum(-1) / (n_tot * n_car)).cpu().numpy()

            for ui in range(b):
                oc_car = raw_car[ui].clamp(-clamp_val, clamp_val).mean(dim=(0, 1))
                odd_car[s + ui] = oc_car.half().cpu()
                odd_l2_car[s + ui] = float(oc_car.norm().item())
            del raw_d, raw_car, m_tot, m_dir, m_car

        del lt_p, lsm_t
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    res: Dict[str, Any] = {
        "kl_pos": kl_pos,
        "kl_neg": kl_neg,
        "even_l2": even_l2,
        "odd_cent": odd_cent,
        "odd_proj": odd_proj,
        "odd_l2_cent": odd_l2_cent,
        "odd_l2_proj": odd_l2_proj,
        "alpha_abs": alpha_abs,
    }
    if has_split:
        res["odd_car"] = odd_car
        res["odd_l2_car"] = odd_l2_car
        res["carried_l2_frac"] = carried_l2_frac
        res["cos_tot_dir"] = cos_tot_dir
        res["cos_tot_car"] = cos_tot_car
    if keep_dp:
        res["dp_mean"] = dp_mean
        res["p_pos_mean"] = p_pos_mean
    return res


def _extract_unit_pairs(sc: Dict[str, Any], u_idx: int, raw_str: Sequence[str], k: int = 5) -> Tuple[List[int], List[int]]:
    ps = fo.extract_replicated_pieces(sc, u_idx, raw_str, k=k)
    return [int(pid) for _, _, pid in ps["pos"]], [int(pid) for _, _, pid in ps["neg"]]


def _top_clean_face_words(pos_row: np.ndarray, neg_row: np.ndarray, clean_str: Sequence[str], k_per_face: int = 5) -> Set[str]:
    """Extract up to k_per_face clean words from + and k_per_face from - (fair 5+5 budget matching forced_odd)."""
    out: Set[str] = set()
    for arr in (np.asarray(pos_row, dtype=np.int64), np.asarray(neg_row, dtype=np.int64)):
        picked = 0
        seen_face: Set[str] = set()
        for pid_raw in arr:
            pid = int(pid_raw)
            if 0 <= pid < len(clean_str):
                w = clean_str[pid]
                if w and len(w) >= 2 and w not in seen_face:
                    seen_face.add(w)
                    out.add(w)
                    picked += 1
                    if picked >= k_per_face:
                        break
    return out


def _eval_incremental_vs_lens(
    dec_ref: Optional[Dict[str, Dict[int, Set[str]]]],
    eye_pos_l: np.ndarray,
    eye_neg_l: np.ndarray,
    lens_pos_l: Optional[np.ndarray],
    lens_neg_l: Optional[np.ndarray],
    sc: Dict[str, Any],
    neurons: Sequence[int],
    offset_neu: int,
    raw_str: Sequence[str],
    clean_str: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if dec_ref is None:
        return None
    out: Dict[str, Any] = {}
    for sub in ("q05", "all"):
        wmap = dec_ref[sub]
        in_ref = [loc_i for loc_i, nid in enumerate(neurons) if int(nid) in wmap]
        if not in_ref:
            out[sub] = {"ref_in_sample": 0}
            continue
        eye_hits, lens_hits, forced_hits = 0, 0, 0
        union_eye_hits, union_lens_hits = 0, 0
        added_over_eye, added_over_lens = 0, 0
        for loc_i in in_ref:
            nid = int(neurons[loc_i])
            dw = wmap[nid]
            ew = _top_clean_face_words(eye_pos_l[nid], eye_neg_l[nid], clean_str, k_per_face=5)
            lw = (
                _top_clean_face_words(lens_pos_l[nid], lens_neg_l[nid], clean_str, k_per_face=5)
                if lens_pos_l is not None and lens_neg_l is not None
                else ew
            )
            ps = fo.extract_replicated_pieces(sc, offset_neu + loc_i, raw_str, k=5)
            fw = {clean_str[pid] for _, _, pid in ps["pos"] + ps["neg"] if 0 <= pid < len(clean_str) and len(clean_str[pid]) >= 2}
            e_ok = bool(ew & dw)
            l_ok = bool(lw & dw)
            f_ok = bool(fw & dw)
            eye_hits += int(e_ok)
            lens_hits += int(l_ok)
            forced_hits += int(f_ok)
            union_eye_hits += int(e_ok or f_ok)
            union_lens_hits += int(l_ok or f_ok)
            if not e_ok and f_ok:
                added_over_eye += 1
            if not l_ok and f_ok:
                added_over_lens += 1
        d_n = len(in_ref)
        eye_acc = eye_hits / d_n
        lens_acc = lens_hits / d_n
        u_eye = union_eye_hits / d_n
        u_lens = union_lens_hits / d_n
        out[sub] = {
            "ref_in_sample": d_n,
            "eye_only_hits": eye_hits,
            "eye_only_acc": round(eye_acc, 4),
            "lens_only_hits": lens_hits,
            "lens_only_acc": round(lens_acc, 4),
            "forced_only_hits": forced_hits,
            "forced_only_acc": round(forced_hits / d_n, 4),
            "union_eye_forced_acc": round(u_eye, 4),
            "union_lens_forced_acc": round(u_lens, 4),
            "added_by_forced_over_eye": added_over_eye,
            "added_by_forced_over_lens": added_over_lens,
            "incremental_over_eye": round(u_eye - eye_acc, 4),
            "incremental_over_lens": round(u_lens - lens_acc, 4),
        }
    return out


def _score_and_eval_matrix(
    A_mat: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    cov: np.ndarray,
    col_mask: np.ndarray,
    iso_mask: np.ndarray,
    neu_mask: np.ndarray,
    lex_gpu: torch.Tensor,
    raw_str: Sequence[str],
    dev: torch.device,
    dp_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n_cols = int(col_mask.sum())
    n_iso = int(iso_mask.sum())
    n_neu = int(neu_mask.sum())
    offset = n_cols + n_iso

    sc_col = fo.score_odd_pair_gpu(A_mat, B_mat, col_mask, cov, lex_gpu, False, dev, keep_words=True, top_k=5)
    sc_iso = fo.score_odd_pair_gpu(A_mat, B_mat, iso_mask, cov, lex_gpu, False, dev, keep_words=True, top_k=5)

    ev_col = fo.eval_split_metrics(sc_col, neu_mask, col_mask)
    ev_iso = fo.eval_split_metrics(sc_iso, neu_mask, iso_mask)
    ev_iso_vs_cols = fo.eval_split_metrics(sc_col, iso_mask, col_mask)
    ev_cols_vs_iso = fo.eval_split_metrics(sc_iso, col_mask, iso_mask)

    dual_named = sorted(set(int(x) for x in ev_col["_named_local_idx"]) & set(int(x) for x in ev_iso["_named_local_idx"]))

    # Like-for-like matched Half-C' (evaluated on ALL n_neu and cols/iso)
    C_dev = C_mat.to(device=dev, dtype=torch.float32)
    normC_col, normC_neu_c = fo._normalize_half_c(C_dev[:n_cols], C_dev[offset:], lex_gpu, zscore_var=False)
    normC_iso, normC_neu_i = fo._normalize_half_c(C_dev[n_cols:offset], C_dev[offset:], lex_gpu, zscore_var=False)
    col_pairs = [_extract_unit_pairs(sc_col, u, raw_str, k=5) for u in range(n_cols)]
    iso_pairs = [_extract_unit_pairs(sc_iso, n_cols + u, raw_str, k=5) for u in range(n_iso)]
    neu_pairs_c = [_extract_unit_pairs(sc_col, offset + li, raw_str, k=5) for li in range(n_neu)]
    neu_pairs_i = [_extract_unit_pairs(sc_iso, offset + li, raw_str, k=5) for li in range(n_neu)]

    hc_vs_cols = fo.score_half_c_matched(normC_col, col_pairs, normC_neu_c, neu_pairs_c)
    hc_vs_iso = fo.score_half_c_matched(normC_iso, iso_pairs, normC_neu_i, neu_pairs_i)

    # Relabelled control check on matched Half-C'
    hc_relabel_ctl = fo.score_half_c_matched(normC_col[: n_cols // 2], col_pairs[: n_cols // 2], normC_col[n_cols // 2 :], col_pairs[n_cols // 2 :])

    dp_metrics = None
    if dp_info is not None:
        dp_c = dp_info["dp_mean"]
        p_pos_c = dp_info["p_pos_mean"]
        base_p = dp_info["base_p_mean"]
        top20_cutoff = float(np.partition(base_p, -20)[-20])

        def _dp_for(u_idx: int, pairs: Tuple[List[int], List[int]]) -> Tuple[float, float, float]:
            pids, nids = pairs
            if not pids and not nids:
                return 0.0, 0.0, 0.0
            pb_vals = [float(base_p[p]) for p in pids + nids]
            dp_pos = [float(dp_c[u_idx, p]) for p in pids]
            dp_neg = [-float(dp_c[u_idx, p]) for p in nids]
            signed_dp = float(np.mean(dp_pos + dp_neg)) if (dp_pos or dp_neg) else 0.0
            entered_t20 = float(np.mean([1.0 if (base_p[p] < top20_cutoff and p_pos_c[u_idx, p] >= top20_cutoff) else 0.0 for p in pids])) if pids else 0.0
            return float(np.mean(pb_vals)), signed_dp, entered_t20

        neu_dp_rows = [_dp_for(offset + li, neu_pairs_c[li]) for li in range(n_neu)]
        col_dp_rows = [_dp_for(c, col_pairs[c]) for c in range(n_cols)]
        dp_metrics = {
            "neu_p_base_p50": round(float(np.median([r[0] for r in neu_dp_rows])), 6),
            "neu_p_base_p90": round(float(np.percentile([r[0] for r in neu_dp_rows], 90)), 6),
            "neu_signed_dp_p50": round(float(np.median([r[1] for r in neu_dp_rows])), 6),
            "neu_signed_dp_p90": round(float(np.percentile([r[1] for r in neu_dp_rows], 90)), 6),
            "ctl_signed_dp_p50": round(float(np.median([r[1] for r in col_dp_rows])), 6),
            "dp_auc_vs_cols": round(fo.auc([r[1] for r in neu_dp_rows], [r[1] for r in col_dp_rows]), 4),
            "top20_entry_rate_mean": round(float(np.mean([r[2] for r in neu_dp_rows])), 5),
        }

    return {
        "sc_col": sc_col,
        "vs_columns": {k: v for k, v in ev_col.items() if k != "_named_local_idx"},
        "vs_isotropic": {k: v for k, v in ev_iso.items() if k != "_named_local_idx"},
        "dual05_count": len(dual_named),
        "dual05_frac": round(len(dual_named) / max(n_neu, 1), 4),
        "iso_false_vs_cols": int(ev_iso_vs_cols["fdp05"]["count"]),
        "iso_false_vs_cols_frac": round(int(ev_iso_vs_cols["fdp05"]["count"]) / max(n_iso, 1), 4),
        "cols_false_vs_iso": int(ev_cols_vs_iso["fdp05"]["count"]),
        "cols_false_vs_iso_frac": round(int(ev_cols_vs_iso["fdp05"]["count"]) / max(n_cols, 1), 4),
        "half_c_matched_vs_cols": hc_vs_cols,
        "half_c_matched_vs_iso": hc_vs_iso,
        "half_c_relabelled_ctl_check": hc_relabel_ctl,
        "delta_p_on_C": dp_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Design-A fresh-text forced_odd kill-pilot")
    ap.add_argument("--layers", default="45,54,58,59")
    ap.add_argument("--n-neurons", type=int, default=256)
    ap.add_argument("--n-cols", type=int, default=512)
    ap.add_argument("--n-iso", type=int, default=512)
    ap.add_argument("--batch-units", type=int, default=4)
    ap.add_argument("--out-dir", default="experiments/results/census_gap_pilots")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = fo.resolve_data_dir(Path(args.data_dir) if args.data_dir else None)

    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    w_ops = ao.Weights(lm, rt)
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    N_ff = w_ops.d_ff

    lex_gpu, _, clean_str, raw_str = fo.build_vocab_masks(rt.tokenizer, int(W_U.shape[0]))
    lex_gpu = lex_gpu.to(dev)
    texts = build_fresh_texts(rt, lm)
    pf = preflight_with_h59(rt, lm, texts)
    print(f"Fresh-text preflight OK: {pf}", flush=True)

    lens_path = data_dir / "index/lens.npz"
    lens_npz = np.load(lens_path) if lens_path.exists() else None

    layer_reports: List[Dict[str, Any]] = []
    neuron_ids_meta: Dict[str, List[int]] = {}
    for layer in layers:
        tl0 = time.time()
        dose_ceil, dose_src = _dose_for_layer(layer)
        rng = np.random.default_rng(2026112520 + layer)
        neurons = sorted(int(i) for i in rng.choice(N_ff, size=args.n_neurons, replace=False))
        neuron_ids_meta[str(layer)] = neurons

        units = fo.make_units(rt, lm, w_ops, layer, neurons, n_cols=args.n_cols, seed=9100 + layer, n_iso=args.n_iso, iso_seed=9600 + layer)
        d_cols = torch.from_numpy((dose_ceil * units["ref_ceils_col"])[:, None] * units["col_dirs"]).to(dev)
        d_iso = torch.from_numpy((dose_ceil * units["ref_ceils_iso"])[:, None] * units["iso_dirs"]).to(dev)
        d_neu = (dose_ceil * units["c_neu"])[:, None] * units["u_neu"]
        all_deltas = torch.cat([d_cols, d_iso, d_neu], dim=0)
        del d_cols, d_iso, d_neu

        U = all_deltas.shape[0]
        offset = args.n_cols + args.n_iso
        col_mask = np.zeros(U, dtype=bool)
        col_mask[: args.n_cols] = True
        iso_mask = np.zeros(U, dtype=bool)
        iso_mask[args.n_cols : offset] = True
        neu_mask = np.zeros(U, dtype=bool)
        neu_mask[offset:] = True

        direct_scale = float(w_ops.scalars[layer + 1 :].prod().item()) if layer < 59 else 1.0
        grpA = prepare_group_cache_split(rt, lm, layer, texts, "A")
        resA = run_forward_half_split(lm, W_U, layer, grpA, all_deltas, args.batch_units, direct_scale=direct_scale, keep_dp=False)
        del grpA
        grpB = prepare_group_cache_split(rt, lm, layer, texts, "B")
        resB = run_forward_half_split(lm, W_U, layer, grpB, all_deltas, args.batch_units, direct_scale=direct_scale, keep_dp=False)
        del grpB
        grpC = prepare_group_cache_split(rt, lm, layer, texts, "C")
        resC = run_forward_half_split(lm, W_U, layer, grpC, all_deltas, args.batch_units, direct_scale=direct_scale, keep_dp=True)
        base_p_c = grpC["base_p_mean"]
        del grpC, all_deltas

        combos_out: Dict[str, Any] = {}
        dp_payload = {"dp_mean": resC["dp_mean"], "p_pos_mean": resC["p_pos_mean"], "base_p_mean": base_p_c}
        for rmode in ("cent", "proj"):
            cov = fo.build_cov_matrix(resA, resB, rmode)
            scored = _score_and_eval_matrix(
                resA[f"odd_{rmode}"],
                resB[f"odd_{rmode}"],
                resC[f"odd_{rmode}"],
                cov,
                col_mask,
                iso_mask,
                neu_mask,
                lex_gpu,
                raw_str,
                dev,
                dp_info=dp_payload,
            )
            inc = None
            if lens_npz is not None and layer in (45, 54):
                dec_ref = load_decoding_overlap_ref(layer, data_dir)
                inc = _eval_incremental_vs_lens(
                    dec_ref,
                    lens_npz["eye_pos"][layer],
                    lens_npz["eye_neg"][layer],
                    lens_npz["lens_pos"][layer] if "lens_pos" in lens_npz else None,
                    lens_npz["lens_neg"][layer] if "lens_neg" in lens_npz else None,
                    scored["sc_col"],
                    neurons,
                    offset,
                    raw_str,
                    clean_str,
                )
            scored["incremental_vs_lens"] = inc
            scored.pop("sc_col", None)
            combos_out[f"{rmode}/LEX_L2_cent"] = scored

        carried_out = None
        if layer < 59 and "odd_car" in resA:
            cov_car = np.column_stack(
                [
                    np.ones(U, dtype=np.float64),
                    np.log(0.5 * (resA["kl_pos"] + resB["kl_pos"]) + 1e-4),
                    np.log(0.5 * (resA["kl_neg"] + resB["kl_neg"]) + 1e-4),
                    np.log(0.5 * (resA["odd_l2_car"] + resB["odd_l2_car"]) + 1e-8),
                    0.5 * (resA["even_l2"] / (resA["odd_l2_car"] + 1e-8) + resB["even_l2"] / (resB["odd_l2_car"] + 1e-8)),
                ]
            )
            car_scored = _score_and_eval_matrix(
                resA["odd_car"],
                resB["odd_car"],
                resC["odd_car"],
                cov_car,
                col_mask,
                iso_mask,
                neu_mask,
                lex_gpu,
                raw_str,
                dev,
                dp_info=None,
            )
            car_scored.pop("sc_col", None)
            cf_neu = np.concatenate([resA["carried_l2_frac"][neu_mask], resB["carried_l2_frac"][neu_mask]])
            ctd_neu = np.concatenate([resA["cos_tot_dir"][neu_mask], resB["cos_tot_dir"][neu_mask]])
            ctc_neu = np.concatenate([resA["cos_tot_car"][neu_mask], resB["cos_tot_car"][neu_mask]])
            carried_out = {
                "carried_l2_frac_p50": round(float(np.nanmedian(cf_neu)), 4),
                "carried_l2_frac_p90": round(float(np.nanpercentile(cf_neu, 90)), 4),
                "cos_tot_dir_p50": round(float(np.nanmedian(ctd_neu)), 4),
                "cos_tot_car_p50": round(float(np.nanmedian(ctc_neu)), 4),
                **car_scored,
            }

        kl_n = 0.25 * (resA["kl_pos"][neu_mask] + resA["kl_neg"][neu_mask] + resB["kl_pos"][neu_mask] + resB["kl_neg"][neu_mask])
        rep = {
            "layer": layer,
            "dose_ceil": dose_ceil,
            "dose_source": dose_src,
            "direct_scale": round(direct_scale, 6),
            "n_neurons": args.n_neurons,
            "n_cols": args.n_cols,
            "n_iso": args.n_iso,
            "kl_p50": round(float(np.median(kl_n)), 4),
            "kl_p90": round(float(np.percentile(kl_n, 90)), 4),
            "combos": combos_out,
            "carried": carried_out,
            "elapsed_s": round(time.time() - tl0, 1),
        }
        layer_reports.append(rep)
        p_rep = combos_out["proj/LEX_L2_cent"]
        print(
            f"L{layer:2d} ({rep['elapsed_s']}s): proj T_col={p_rep['vs_columns']['auc_T']:.3f} Dual05={p_rep['dual05_count']}/{args.n_neurons} "
            f"iso->col={p_rep['iso_false_vs_cols_frac']:.3f} col->iso={p_rep['cols_false_vs_iso_frac']:.3f} "
            f"HC'_col={p_rep['half_c_matched_vs_cols']['half_c_auc']} | "
            f"carDual={carried_out['dual05_count'] if carried_out else '-'}",
            flush=True,
        )
        del resA, resB, resC
        torch.cuda.empty_cache()
        gc.collect()

    lines = [
        "Exploratory Design-A Fresh-Text forced_odd Kill-Pilot (N=256, C_cols=512, C_iso=512, 18 unseen prompts A'/B'/C')",
        "-" * 134,
        f"{'L':>3} {'Mode':<5} {'DoseC':>6} {'KL50/90':>11} {'Tcol':>6} {'Tiso':>6} {'Dual05':>7} {'Iso->Col':>9} {'Col->Iso':>9} {'HC_col':>7} {'CarFrac':>8} {'CarTcol':>8} {'CarDual':>8}",
    ]
    for r in layer_reports:
        kl_s = f"{r['kl_p50']:.3f}/{r['kl_p90']:.3f}"
        car = r["carried"] or {}
        for mk in ("cent", "proj"):
            c = r["combos"][f"{mk}/LEX_L2_cent"]
            hc = c["half_c_matched_vs_cols"]
            car_dual_s = f"{car['dual05_count']}/{r['n_neurons']}" if mk == "cent" and car else "-"
            lines.append(
                f"{r['layer']:3d} {mk:<5} {r['dose_ceil']:6.3f} {kl_s:>11} "
                f"{c['vs_columns']['auc_T']:6.3f} {c['vs_isotropic']['auc_T']:6.3f} "
                f"{c['dual05_count']:3d}/{r['n_neurons']:<3d} "
                f"{c['iso_false_vs_cols_frac']:9.3f} {c['cols_false_vs_iso_frac']:9.3f} "
                f"{(hc.get('half_c_auc') or 0):7.3f} "
                f"{(car.get('carried_l2_frac_p50') if mk == 'cent' and car else float('nan')):8.3f} "
                f"{(car.get('vs_columns', {}).get('auc_T') if mk == 'cent' and car else float('nan')):8.3f} "
                f"{car_dual_s:>8}"
            )
    by_layer_rep = {int(r["layer"]): r for r in layer_reports}
    kc1_layers = {}
    for l_chk in (58, 59):
        if l_chk in by_layer_rep:
            pr = by_layer_rep[l_chk]["combos"]["proj/LEX_L2_cent"]
            kc1_layers[str(l_chk)] = {
                "iso_false_vs_cols_frac": pr["iso_false_vs_cols_frac"],
                "cols_false_vs_iso_frac": pr["cols_false_vs_iso_frac"],
                "triggered": bool(pr["iso_false_vs_cols_frac"] > 0.05 or pr["cols_false_vs_iso_frac"] > 0.05),
            }
    kc2_inc = {}
    for l_chk in (45, 54):
        if l_chk in by_layer_rep:
            inc_q05 = ((by_layer_rep[l_chk]["combos"]["proj/LEX_L2_cent"].get("incremental_vs_lens") or {}).get("q05")) or {}
            inc_val = inc_q05.get("incremental_over_lens")
            kc2_inc[str(l_chk)] = {
                "incremental_over_lens_q05": inc_val,
                "incremental_over_eye_q05": inc_q05.get("incremental_over_eye"),
                "triggered_lt_0_03": (inc_val < 0.03) if inc_val is not None else None,
            }
    kc2_l58_car = None
    if 58 in by_layer_rep and by_layer_rep[58].get("carried"):
        c58 = by_layer_rep[58]["carried"]
        kc2_l58_car = {
            "carried_dual05_frac": c58["dual05_frac"],
            "carried_l2_frac_p50": c58["carried_l2_frac_p50"],
            "triggered_lt_0_05": bool(c58["dual05_frac"] < 0.05),
        }
    kill_criteria = {
        "kc1_cross_pool_gt_0_05": kc1_layers,
        "kc2_incremental_over_lens": kc2_inc,
        "kc2_l58_carried_lt_0_05": kc2_l58_car,
    }
    lines.extend(["", f"Kill Criteria: {json.dumps(kill_criteria)}"])
    txt = "\n".join(lines) + "\n"
    prov = gp.build_provenance(
        REPO_ROOT,
        [Path(__file__), REPO_ROOT / "backend" / "forced_odd.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(),
        model_id=getattr(rt, "model_id", None),
        weights=weights_card(rt),
        neuron_ids=neuron_ids_meta,
    )
    payload = {
        **prov,
        "exploratory": True,
        "registered": False,
        "gates_nothing": True,
        "prompt_names": {g: [k for k, _ in v] for g, v in FRESH_GROUPS.items()},
        "preflight": pf,
        "kill_criteria": kill_criteria,
        "elapsed_total_s": round(time.time() - t0, 1),
        "layers": layer_reports,
    }
    (out_dir / "forced_odd_kill_pilot.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (out_dir / "forced_odd_kill_pilot.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
