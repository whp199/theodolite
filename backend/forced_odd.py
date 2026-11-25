"""Controlled antisymmetric fixed-context forced logit census (`forced_odd`).

Implements fixed-text residual-stream perturbation analysis without autoregressive
decoding noise, using vocab-mean centering (removing RMSNorm Δ log Z offsets) and
optional projection orthogonal to unpushed log-logits `z0` (removing RMSNorm
temperature scaling at late layers such as L59).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import atlas_index as ai
from backend import atlas_ops as ao
from backend import census as cz
from backend.gap_stats import mannwhitney_greater
from backend.causal_check import language_model, logits_fp32, model_device, pad_right
from backend.label_gate import label_top_ids
from backend.theodolite_math import unembed_gemma4

HALVES: Tuple[Tuple[str, ...], Tuple[str, ...]] = (
    ("bicycle", "walk", "chair", "kettle", "sleep", "lamp"),
    ("clock", "rice", "vase", "lock", "money", "blanket"),
)
HALF_C: Tuple[str, ...] = ("bridge", "windmill", "camera", "boot", "mug", "winter")

LADDER_RUNGS: Tuple[float, ...] = (0.08, 0.12, 0.18, 0.26, 0.38, 0.55, 0.80, 1.15, 1.65, 2.35)
TARGET_KL_DEFAULT: float = 0.25
CLAMP_ODD_DEFAULT: float = 6.0
VAR_FLOOR: float = 0.05  # variance stabilisation (VAR_FLOOR**2 = 0.0025)
READOUT_MODES: Tuple[str, ...] = ("cent", "proj", "both")

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_dir(data_dir: Optional[Path] = None) -> Path:
    """Resolve read-only atlas_data root from explicit argument or backend.atlas_index.atlas_dir()."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    return ai.atlas_dir()


def norm_word(s: str) -> str:
    return re.sub(r"[^\w]", "", str(s).replace("▁", " ").strip().lower())


def auc(a: Sequence[float], b: Sequence[float]) -> float:
    return float(mannwhitney_greater(a, b)[0])


def bh(p: Sequence[float]) -> np.ndarray:
    p_arr = np.asarray(p, float)
    n = len(p_arr)
    if n == 0:
        return p_arr
    order = np.argsort(p_arr)
    q = np.minimum.accumulate((p_arr[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return out


def ols_resid(y_all: np.ndarray, X_all: np.ndarray, fit_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xc = X_all[fit_mask]
    yc = y_all[fit_mask]
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return y_all - (X_all @ beta), beta


def robust_stats(vals: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    std = float(np.std(vals))
    scale = max(1.4826 * mad, 0.25 * std, 1e-6)
    return med, scale


def empirical_fdp(
    scores: np.ndarray,
    pok: np.ndarray,
    n_neu: int,
    ctl_scores: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Step-up empirical FDP: FDP(t) = (N/(C+1)) * (1 + #{c >= t}) / max(1, #{n >= t})."""
    N = int(n_neu)
    C = len(ctl_scores)
    if C == 0 or N == 0:
        return {"count": 0, "above_max_ctl": 0, "threshold": None, "fdp": None, "named_idx": np.array([], np.int64)}
    ctl_sorted = np.sort(ctl_scores.astype(np.float64))
    max_ctl = float(ctl_sorted[-1])
    cand_idx = np.nonzero(pok)[0]
    if len(cand_idx) == 0:
        return {"count": 0, "above_max_ctl": 0, "threshold": None, "fdp": None, "named_idx": np.array([], np.int64)}
    above_max = int(np.sum(scores[pok] > max_ctl))
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


def loglog_interpolate_dose(rungs: Sequence[float], kls: Sequence[float], target: float = TARGET_KL_DEFAULT) -> Tuple[float, Optional[str]]:
    """Interpolate the ceiling dose giving `target` median KL cleanly in log-dose vs log-KL."""
    ladder = [float(d) for d in rungs]
    kl = [float(x) for x in kls]
    if not ladder or len(ladder) != len(kl):
        raise ValueError("rungs and kls must be non-empty and the same length")
    if target <= kl[0]:
        note = None if target == kl[0] else "below the ladder: smallest rung already exceeds target KL"
        return ladder[0], note
    for i in range(len(ladder) - 1):
        a, b = kl[i], kl[i + 1]
        if a <= target <= b and b > a > 0:
            t = (math.log(target) - math.log(a)) / (math.log(b) - math.log(a))
            val = math.exp(math.log(ladder[i]) + t * (math.log(ladder[i + 1]) - math.log(ladder[i])))
            return float(f"{val:.4g}"), None
        if a <= target <= b:
            return ladder[i + 1], None
    return ladder[-1], "above the ladder: largest rung stays below target KL"


def dose_from_ladder_entry(entry: Dict[str, Any], stat: str = "kl_p90", target: float = TARGET_KL_DEFAULT) -> Tuple[float, Optional[str]]:
    """Interpolate a layer's target ceiling dose from one ladder entry using `stat` (e.g. `kl_p90` or `kl_p50`)."""
    rungs = [float(rr["dose_ceil"]) for rr in entry["rungs"]]
    vals = [float(rr[stat]) for rr in entry["rungs"]]
    return loglog_interpolate_dose(rungs, vals, target=target)


def build_vocab_masks(tokenizer: Any, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Build LEX (3+-letter non-special lexical pieces) and ALL (all non-special pieces) masks."""
    V = int(vocab_size)
    special_ids = set(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
    pieces = tokenizer.convert_ids_to_tokens(list(range(V)))
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
        clean_str[i] = cleaned.lower()
        if word_re.fullmatch(cleaned):
            lex[i] = True
    return torch.from_numpy(lex), torch.from_numpy(all_ns), clean_str, raw_str


def build_fixed_texts(rt: Any, lm: nn.Module, data_dir: Optional[Path] = None, include_half_c: bool = True) -> Dict[str, Any]:
    """Load the 12 match_things chat prompts & unpushed greedy answers (plus optional Half C)."""
    dd = resolve_data_dir(data_dir)
    meta29_path = dd / "census/trials/things_walk/L29-match_things-trial.meta.json"
    trial29_path = dd / "census/trials/things_walk/L29-match_things-trial.jsonl"
    meta29 = json.loads(meta29_path.read_text(encoding="utf-8"))
    prompt_ids_map: Dict[str, List[int]] = dict(meta29["prompt_ids"])
    bases: Dict[str, List[int]] = {}
    with open(trial29_path, encoding="utf-8") as f:
        for line in f:
            if '"baseline"' in line:
                r = json.loads(line)
                if r.get("kind") == "baseline":
                    bases.setdefault(r["prompt"], [int(t) for t in r["token_ids"]])

    stop_ids = rt.end_of_turn_ids()
    bos_id = rt.tokenizer.bos_token_id
    softcap = ao.softcap_of(rt.hf_model)

    q_all: List[str] = [*HALVES[0], *HALVES[1]]
    groups: Dict[str, List[str]] = {"A": list(HALVES[0]), "B": list(HALVES[1])}

    if include_half_c:
        c_prompts = [rt.tokenize_prompt(rt.compose_prompt(cz.MORE_PROMPTS[q], chat=True)) for q in HALF_C]
        c_answers: List[List[int]] = [[] for _ in HALF_C]
        c_ended = [False] * len(HALF_C)
        for st in ao.write_rows(
            lm,
            rt.lm_head_weight,
            c_prompts,
            None,
            max_new=48,
            stop_ids=stop_ids,
            bos_id=bos_id,
            spare_bos=True,
            softcap=softcap,
            top_k=0,
        ):
            c_answers[st.row].append(st.token_id)
            c_ended[st.row] = c_ended[st.row] or st.end
        for q, pids, ans, end in zip(HALF_C, c_prompts, c_answers, c_ended):
            prompt_ids_map[q] = pids
            bases[q] = ans[:-1] if end else ans
        q_all.extend(HALF_C)
        groups["C"] = list(HALF_C)

    seqs: Dict[str, List[int]] = {}
    prompt_lens: Dict[str, int] = {}
    scored_pos: Dict[str, List[int]] = {}
    for q in q_all:
        pids = list(prompt_ids_map[q])
        aids = list(bases[q])
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

    return {"seqs": seqs, "prompt_lens": prompt_lens, "scored_pos": scored_pos, "groups": groups}


def run_preflight(rt: Any, lm: nn.Module, texts: Dict[str, Any], check_layers: Sequence[int] = (29, 59)) -> Dict[str, Any]:
    """Verify base answer log-probabilities and exactness of `forward_from_layer`."""
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    report: Dict[str, Any] = {}
    for gnm, qlist in texts["groups"].items():
        seq_list = [texts["seqs"][q] for q in qlist]
        inp, att, _, _ = pad_right(seq_list, dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        cap: Dict[int, torch.Tensor] = {}
        hks = [
            lm.layers[l].register_forward_hook(
                lambda _m, _a, o, ll=l: cap.__setitem__(ll, (o[0] if isinstance(o, tuple) else o).detach())
            )
            for l in check_layers
        ]
        try:
            with torch.no_grad():
                full = lm(input_ids=inp, attention_mask=att, use_cache=False)
        finally:
            for hk in hks:
                hk.remove()
        h_ref = full.last_hidden_state
        min_lp = 0.0
        for bi, q in enumerate(qlist):
            p_len = texts["prompt_lens"][q]
            L_q = len(texts["seqs"][q])
            l_ans = logits_fp32(h_ref[bi, p_len - 1 : L_q - 1], W_U)
            lp_ans = float(F.log_softmax(l_ans, dim=-1).gather(-1, inp[bi, p_len:L_q, None]).mean().item())
            min_lp = min(min_lp, lp_ans)
        diffs: Dict[str, float] = {}
        for l in check_layers:
            r_l = ao.forward_from_layer(lm, l + 1, cap[l], att, pos)
            diffs[str(l)] = float((r_l - h_ref).abs().max().item())
            del r_l
        max_diff = max(diffs.values()) if diffs else 0.0
        report[gnm] = {"min_ans_lp": round(min_lp, 4), "max_diffs": diffs}
        if min_lp < -0.5 or max_diff > 1e-2:
            raise RuntimeError(f"Preflight failed on group {gnm}: lp={min_lp:.4f}, diffs={diffs}")
        del full, h_ref, cap, inp, att, pos
    return report


def precompute_base_stats(base_lsm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute unpushed mean-centered logits `z0`, `||z0||^2`, and mean across scored positions."""
    z0 = base_lsm - base_lsm.mean(dim=-1, keepdim=True)
    z0_norm2 = (z0 * z0).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    base_mean = base_lsm.mean(dim=(0, 1))
    return z0, z0_norm2, base_mean


@torch.no_grad()
def extract_odd_from_lsm(
    lsm: torch.Tensor,
    base_lsm: torch.Tensor,
    z0: torch.Tensor,
    z0_norm2: torch.Tensor,
    base_mean: torch.Tensor,
    readout_mode: str = "both",
    clamp_val: float = CLAMP_ODD_DEFAULT,
) -> Dict[str, torch.Tensor | np.ndarray]:
    """Extract KL, centered even norm, RMSNorm temperature alpha, and odd_cent / odd_proj from `lsm [2, b, H, S, V]`.

    Mutates `lsm[0]` in-place to minimize GPU peak memory.
    """
    if readout_mode not in READOUT_MODES:
        raise ValueError(f"Unknown readout_mode={readout_mode!r}; expected one of {READOUT_MODES}")

    lsm_p, lsm_n = lsm[0], lsm[1]  # [b, H, S, V]
    b = lsm_p.shape[0]
    kl_p = np.zeros(b, dtype=np.float32)
    kl_n = np.zeros(b, dtype=np.float32)
    for ui in range(b):
        lp_ui = lsm_p[ui]
        ln_ui = lsm_n[ui]
        kl_p[ui] = float((lp_ui.exp() * (lp_ui - base_lsm)).sum(-1).mean().item())
        kl_n[ui] = float((ln_ui.exp() * (ln_ui - base_lsm)).sum(-1).mean().item())

    # Centered even norm computed on [b, V] mean vectors (avoids [b, H, S, V] allocation)
    even = 0.5 * (lsm_p.mean(dim=(1, 2)) + lsm_n.mean(dim=(1, 2))) - base_mean[None]
    even.sub_(even.mean(dim=-1, keepdim=True))
    even_l2 = even.norm(dim=-1).cpu().numpy().astype(np.float32)
    del even

    # In-place antisymmetric shift & vocab-mean centering per (h, s) to strip Δ log Z
    lsm_p.sub_(lsm_n).mul_(0.5)
    lsm_p.sub_(lsm_p.mean(dim=-1, keepdim=True))

    alpha = (lsm_p * z0[None]).sum(dim=-1, keepdim=True) / z0_norm2[None]  # [b, H, S, 1]
    alpha_temp = alpha.mean(dim=(1, 2, 3)).cpu().numpy().astype(np.float32)
    alpha_abs = alpha.abs().mean(dim=(1, 2, 3)).cpu().numpy().astype(np.float32)

    out: Dict[str, Any] = {
        "kl_pos": kl_p,
        "kl_neg": kl_n,
        "even_l2": even_l2,
        "alpha_temp": alpha_temp,
        "alpha_abs": alpha_abs,
    }

    V = lsm_p.shape[-1]
    if readout_mode in ("cent", "both"):
        odd_cent = torch.empty((b, V), dtype=torch.float16, device="cpu")
        odd_l2_cent = np.zeros(b, dtype=np.float32)
        for ui in range(b):
            oc = lsm_p[ui].clamp(-clamp_val, clamp_val).mean(dim=(0, 1))
            odd_cent[ui] = oc.half().cpu()
            odd_l2_cent[ui] = float(oc.norm().item())
        out["odd_cent"] = odd_cent
        out["odd_l2_cent"] = odd_l2_cent

    if readout_mode in ("proj", "both"):
        odd_proj = torch.empty((b, V), dtype=torch.float16, device="cpu")
        odd_l2_proj = np.zeros(b, dtype=np.float32)
        for ui in range(b):
            op_raw = lsm_p[ui]
            op_raw.sub_(alpha[ui] * z0)
            op = op_raw.clamp_(-clamp_val, clamp_val).mean(dim=(0, 1))
            odd_proj[ui] = op.half().cpu()
            odd_l2_proj[ui] = float(op.norm().item())
        out["odd_proj"] = odd_proj
        out["odd_l2_proj"] = odd_l2_proj

    del alpha
    return out


@torch.no_grad()
def prepare_group_cache(rt: Any, lm: nn.Module, layer: int, texts: Dict[str, Any], gnm: str) -> Dict[str, Any]:
    """Cache residual stream leaving `layer` and unpushed `base_lsm`/`z0` for a prompt group."""
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

    seen: Dict[str, torch.Tensor] = {}
    hk = lm.layers[layer].register_forward_hook(
        lambda _m, _a, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    try:
        full_out = lm(input_ids=inp, attention_mask=att, use_cache=False)
    finally:
        hk.remove()
    h_l = seen["h"]
    h_final = full_out.last_hidden_state
    sc_idx = torch.tensor([texts["scored_pos"][q] for q in qlist], dtype=torch.long, device=dev)
    h_rows = torch.arange(len(qlist), device=dev)[:, None].expand(len(qlist), sc_idx.shape[1])
    h_sc = h_final[h_rows, sc_idx]
    base_lsm = torch.empty((len(qlist), sc_idx.shape[1], V), dtype=torch.float32, device=dev)
    h_flat = h_sc.reshape(-1, D)
    for c0 in range(0, h_flat.shape[0], 64):
        base_lsm.view(-1, V)[c0 : c0 + 64] = F.log_softmax(logits_fp32(h_flat[c0 : c0 + 64], W_U), dim=-1)
    z0, z0_norm2, base_mean = precompute_base_stats(base_lsm)
    del full_out, h_final, h_sc, h_flat, inp
    return {
        "h_l": h_l,
        "att": att,
        "pos": pos,
        "keep": keep_m,
        "sc_idx": sc_idx,
        "base_lsm": base_lsm,
        "z0": z0,
        "z0_norm2": z0_norm2,
        "base_mean": base_mean,
    }


@torch.no_grad()
def run_forward_half(
    lm: nn.Module,
    W_U: torch.Tensor,
    layer: int,
    group_cache: Dict[str, Any],
    deltas: torch.Tensor,
    batch_units: int,
    readout_mode: str = "both",
    clamp_val: float = CLAMP_ODD_DEFAULT,
) -> Dict[str, Any]:
    """Run +/-deltas on one 6-text group and return odd shift matrix `[M, V]` fp16 + KL/norm metrics."""
    if readout_mode not in READOUT_MODES:
        raise ValueError(f"Unknown readout_mode={readout_mode!r}")
    h_l = group_cache["h_l"]
    att = group_cache["att"]
    pos = group_cache["pos"]
    keep_mask_grp = group_cache["keep"]
    scored_idx = group_cache["sc_idx"]
    base_lsm = group_cache["base_lsm"]
    z0 = group_cache["z0"]
    z0_norm2 = group_cache["z0_norm2"]
    base_mean = group_cache["base_mean"]

    dev = h_l.device
    M, D = deltas.shape
    H, T, _ = h_l.shape
    S = scored_idx.shape[1]
    V = int(W_U.shape[0])

    odd_cent_out = torch.empty((M, V), dtype=torch.float16, device="cpu") if readout_mode in ("cent", "both") else None
    odd_proj_out = torch.empty((M, V), dtype=torch.float16, device="cpu") if readout_mode in ("proj", "both") else None
    kl_pos_out = np.zeros(M, np.float32)
    kl_neg_out = np.zeros(M, np.float32)
    even_l2_out = np.zeros(M, np.float32)
    odd_l2_cent_out = np.zeros(M, np.float32) if readout_mode in ("cent", "both") else None
    odd_l2_proj_out = np.zeros(M, np.float32) if readout_mode in ("proj", "both") else None
    alpha_temp_out = np.zeros(M, np.float32)
    alpha_abs_out = np.zeros(M, np.float32)

    h_pos_gather = torch.arange(H, device=dev)[:, None].expand(H, S)

    for s in range(0, M, batch_units):
        db = deltas[s : s + batch_units]
        b = db.shape[0]
        d_exp = db.to(h_l.dtype)[:, None, None, :] * keep_mask_grp[None, :, :, None]
        hp = (h_l[None, :, :, :] + d_exp).reshape(b * H, T, D)
        hn = (h_l[None, :, :, :] - d_exp).reshape(b * H, T, D)
        hs = torch.cat([hp, hn], dim=0)
        del hp, hn, d_exp

        att_r = att.repeat(2 * b, 1)
        pos_r = pos.repeat(2 * b, 1)
        out = ao.forward_from_layer(lm, layer + 1, hs, att_r, pos_r)
        del hs, att_r, pos_r

        out_v = out.view(2 * b, H, T, D)
        sel = out_v[:, h_pos_gather, scored_idx, :].reshape(2 * b * H * S, D)
        del out, out_v

        lsm = torch.empty((2 * b * H * S, V), dtype=torch.float32, device=dev)
        for c0 in range(0, sel.shape[0], 256):
            lsm[c0 : c0 + 256] = F.log_softmax(logits_fp32(sel[c0 : c0 + 256], W_U), dim=-1)
        del sel

        ext = extract_odd_from_lsm(
            lsm.view(2, b, H, S, V),
            base_lsm=base_lsm,
            z0=z0,
            z0_norm2=z0_norm2,
            base_mean=base_mean,
            readout_mode=readout_mode,
            clamp_val=clamp_val,
        )
        del lsm

        kl_pos_out[s : s + b] = ext["kl_pos"]
        kl_neg_out[s : s + b] = ext["kl_neg"]
        even_l2_out[s : s + b] = ext["even_l2"]
        alpha_temp_out[s : s + b] = ext["alpha_temp"]
        alpha_abs_out[s : s + b] = ext["alpha_abs"]
        if odd_cent_out is not None:
            odd_cent_out[s : s + b] = ext["odd_cent"]
            odd_l2_cent_out[s : s + b] = ext["odd_l2_cent"]
        if odd_proj_out is not None:
            odd_proj_out[s : s + b] = ext["odd_proj"]
            odd_l2_proj_out[s : s + b] = ext["odd_l2_proj"]

    res: Dict[str, Any] = {
        "kl_pos": kl_pos_out,
        "kl_neg": kl_neg_out,
        "even_l2": even_l2_out,
        "alpha_temp": alpha_temp_out,
        "alpha_abs": alpha_abs_out,
    }
    if odd_cent_out is not None:
        res["odd_cent"] = odd_cent_out
        res["odd_l2_cent"] = odd_l2_cent_out
    if odd_proj_out is not None:
        res["odd_proj"] = odd_proj_out
        res["odd_l2_proj"] = odd_l2_proj_out
    return res


def build_cov_matrix(resA: Dict[str, Any], resB: Dict[str, Any], mode: str) -> np.ndarray:
    """5-column OLS covariate matrix [1, log(kl_pos+1e-4), log(kl_neg+1e-4), log_odd_l2, even_odd_ratio]."""
    l2_key = f"odd_l2_{mode}"
    odd_a = resA[l2_key]
    odd_b = resB[l2_key]
    U = len(odd_a)
    return np.column_stack(
        [
            np.ones(U, dtype=np.float64),
            np.log(0.5 * (resA["kl_pos"] + resB["kl_pos"]) + 1e-4),
            np.log(0.5 * (resA["kl_neg"] + resB["kl_neg"]) + 1e-4),
            np.log(0.5 * (odd_a + odd_b) + 1e-8),
            0.5 * (resA["even_l2"] / (odd_a + 1e-8) + resB["even_l2"] / (odd_b + 1e-8)),
        ]
    )


@dataclass
class ControlModel:
    """Fitted reference-control normalization, OLS covariate betas, and robust scaling."""

    C: int
    zscore_var: bool
    sumA: torch.Tensor
    sum2A: torch.Tensor
    sumB: torch.Tensor
    sum2B: torch.Tensor
    muA: torch.Tensor
    varA: torch.Tensor
    muB: torch.Tensor
    varB: torch.Tensor
    beta_AB: np.ndarray
    beta_BA: np.ndarray
    med_AB: float
    scale_AB: float
    med_BA: float
    scale_BA: float
    r_ab_ctl: np.ndarray
    r_ba_ctl: np.ndarray
    r_ab_res_ctl: np.ndarray
    r_ba_res_ctl: np.ndarray
    T_ctl: np.ndarray
    words_ctl: Optional[Dict[str, np.ndarray]] = None


def _top_bottom_scores(
    na: torch.Tensor,
    nb: torch.Tensor,
    big_neg: torch.Tensor,
    big_pos: torch.Tensor,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    pa_idx = (na + big_neg[None]).topk(5, dim=-1).indices
    na_idx = (na + big_pos[None]).topk(5, dim=-1, largest=False).indices
    pb_idx = (nb + big_neg[None]).topk(5, dim=-1).indices
    nb_idx = (nb + big_pos[None]).topk(5, dim=-1, largest=False).indices
    s_ab = 0.1 * (nb.gather(-1, pa_idx).sum(-1) - nb.gather(-1, na_idx).sum(-1))
    s_ba = 0.1 * (na.gather(-1, pb_idx).sum(-1) - na.gather(-1, nb_idx).sum(-1))
    words: Optional[Dict[str, np.ndarray]] = None
    if top_k > 0:
        pa_k = (na + big_neg[None]).topk(top_k, dim=-1).indices
        na_k = (na + big_pos[None]).topk(top_k, dim=-1, largest=False).indices
        pb_k = (nb + big_neg[None]).topk(top_k, dim=-1).indices
        nb_k = (nb + big_pos[None]).topk(top_k, dim=-1, largest=False).indices
        comb = 0.5 * (na + nb)
        cp_k = (comb + big_neg[None]).topk(top_k, dim=-1).indices
        cn_k = (comb + big_pos[None]).topk(top_k, dim=-1, largest=False).indices
        words = {
            "topA_p": pa_k.cpu().numpy().astype(np.int32),
            "topA_n": na_k.cpu().numpy().astype(np.int32),
            "topB_p": pb_k.cpu().numpy().astype(np.int32),
            "topB_n": nb_k.cpu().numpy().astype(np.int32),
            "posA_onB": nb.gather(-1, pa_k).cpu().numpy().astype(np.float32),
            "negA_onB": nb.gather(-1, na_k).cpu().numpy().astype(np.float32),
            "posB_onA": na.gather(-1, pb_k).cpu().numpy().astype(np.float32),
            "negB_onA": na.gather(-1, nb_k).cpu().numpy().astype(np.float32),
            "topC_p": cp_k.cpu().numpy().astype(np.int32),
            "topC_n": cn_k.cpu().numpy().astype(np.int32),
            "valC_p": comb.gather(-1, cp_k).cpu().numpy().astype(np.float32),
            "valC_n": comb.gather(-1, cn_k).cpu().numpy().astype(np.float32),
        }
    return s_ab.cpu().numpy().astype(np.float32), s_ba.cpu().numpy().astype(np.float32), words


@torch.no_grad()
def fit_control_model(
    A_ctl: torch.Tensor,
    B_ctl: torch.Tensor,
    cov_ctl: np.ndarray,
    mask_gpu: torch.Tensor,
    zscore_var: bool,
    dev: torch.device,
    keep_words: bool = False,
    top_k: int = 5,
    chunk_size: int = 64,
) -> ControlModel:
    """Fit exact leave-one-out control stats, OLS covariate model, and robust normalization."""
    C, V = A_ctl.shape
    mask_f = mask_gpu.float()
    M_mask = float(mask_f.sum().item())
    vfloor2 = VAR_FLOOR * VAR_FLOOR

    sumA = torch.zeros(V, dtype=torch.float32, device=dev)
    sum2A = torch.zeros(V, dtype=torch.float32, device=dev)
    sumB = torch.zeros(V, dtype=torch.float32, device=dev)
    sum2B = torch.zeros(V, dtype=torch.float32, device=dev)
    for c0 in range(0, C, 128):
        xa = A_ctl[c0 : c0 + 128].to(device=dev, dtype=torch.float32)
        xb = B_ctl[c0 : c0 + 128].to(device=dev, dtype=torch.float32)
        sumA += xa.sum(0)
        sum2A += (xa * xa).sum(0)
        sumB += xb.sum(0)
        sum2B += (xb * xb).sum(0)
        del xa, xb

    muA_all = sumA / C
    varA_all = ((sum2A - C * muA_all * muA_all) / max(C - 1, 1)).clamp(min=0.0)
    muB_all = sumB / C
    varB_all = ((sum2B - C * muB_all * muB_all) / max(C - 1, 1)).clamp(min=0.0)

    big_neg = torch.where(mask_gpu, torch.zeros(V, device=dev), torch.full((V,), -1e9, device=dev))
    big_pos = torch.where(mask_gpu, torch.zeros(V, device=dev), torch.full((V,), 1e9, device=dev))

    r_ab = np.zeros(C, np.float32)
    r_ba = np.zeros(C, np.float32)
    k_word = top_k if keep_words else 0
    w_bufs: Optional[Dict[str, np.ndarray]] = None

    for s in range(0, C, chunk_size):
        e = min(C, s + chunk_size)
        xa = A_ctl[s:e].to(device=dev, dtype=torch.float32)
        xb = B_ctl[s:e].to(device=dev, dtype=torch.float32)
        muA_u = (sumA[None] - xa) / max(C - 1, 1)
        muB_u = (sumB[None] - xb) / max(C - 1, 1)
        ca = xa - muA_u
        cb = xb - muB_u
        if zscore_var:
            vA_loo = (((sum2A[None] - xa * xa) - (C - 1) * muA_u * muA_u) / max(C - 2, 1)).clamp(min=0.0)
            vB_loo = (((sum2B[None] - xb * xb) - (C - 1) * muB_u * muB_u) / max(C - 2, 1)).clamp(min=0.0)
            ca = ca / torch.sqrt(vA_loo + vfloor2)
            cb = cb / torch.sqrt(vB_loo + vfloor2)
        ca = ca * mask_f[None]
        cb = cb * mask_f[None]
        na = ca / torch.sqrt((ca * ca).sum(-1, keepdim=True) / M_mask + 1e-8)
        nb = cb / torch.sqrt((cb * cb).sum(-1, keepdim=True) / M_mask + 1e-8)

        s_ab, s_ba, w_part = _top_bottom_scores(na, nb, big_neg, big_pos, k_word)
        r_ab[s:e] = s_ab
        r_ba[s:e] = s_ba
        if w_part is not None:
            if w_bufs is None:
                w_bufs = {key: np.empty((C, k_word), dtype=arr.dtype) for key, arr in w_part.items()}
            for key, arr in w_part.items():
                w_bufs[key][s:e] = arr
        del xa, xb, ca, cb, na, nb

    full_fit = np.ones(C, dtype=bool)
    r_ab_res, beta_AB = ols_resid(r_ab, cov_ctl, full_fit)
    r_ba_res, beta_BA = ols_resid(r_ba, cov_ctl, full_fit)
    med_AB, scale_AB = robust_stats(r_ab_res)
    med_BA, scale_BA = robust_stats(r_ba_res)
    T_ctl = np.minimum((r_ab_res - med_AB) / scale_AB, (r_ba_res - med_BA) / scale_BA).astype(np.float32)

    return ControlModel(
        C=C,
        zscore_var=zscore_var,
        sumA=sumA,
        sum2A=sum2A,
        sumB=sumB,
        sum2B=sum2B,
        muA=muA_all,
        varA=varA_all,
        muB=muB_all,
        varB=varB_all,
        beta_AB=beta_AB,
        beta_BA=beta_BA,
        med_AB=med_AB,
        scale_AB=scale_AB,
        med_BA=med_BA,
        scale_BA=scale_BA,
        r_ab_ctl=r_ab,
        r_ba_ctl=r_ba,
        r_ab_res_ctl=r_ab_res.astype(np.float32),
        r_ba_res_ctl=r_ba_res.astype(np.float32),
        T_ctl=T_ctl,
        words_ctl=w_bufs,
    )


@torch.no_grad()
def score_chunk_against_controls(
    A_chunk: torch.Tensor,
    B_chunk: torch.Tensor,
    cov_chunk: np.ndarray,
    model: ControlModel,
    mask_gpu: torch.Tensor,
    dev: torch.device,
    keep_words: bool = False,
    top_k: int = 5,
    chunk_size: int = 64,
) -> Dict[str, Any]:
    """Score arbitrary non-reference units against a pre-fitted `ControlModel`."""
    M, V = A_chunk.shape
    mask_f = mask_gpu.float()
    M_mask = float(mask_f.sum().item())
    vfloor2 = VAR_FLOOR * VAR_FLOOR
    big_neg = torch.where(mask_gpu, torch.zeros(V, device=dev), torch.full((V,), -1e9, device=dev))
    big_pos = torch.where(mask_gpu, torch.zeros(V, device=dev), torch.full((V,), 1e9, device=dev))

    r_ab = np.zeros(M, np.float32)
    r_ba = np.zeros(M, np.float32)
    k_word = top_k if keep_words else 0
    w_bufs: Optional[Dict[str, np.ndarray]] = None

    inv_sdA = torch.rsqrt(model.varA + vfloor2) if model.zscore_var else None
    inv_sdB = torch.rsqrt(model.varB + vfloor2) if model.zscore_var else None

    for s in range(0, M, chunk_size):
        e = min(M, s + chunk_size)
        xa = A_chunk[s:e].to(device=dev, dtype=torch.float32)
        xb = B_chunk[s:e].to(device=dev, dtype=torch.float32)
        ca = xa - model.muA[None]
        cb = xb - model.muB[None]
        if model.zscore_var:
            ca = ca * inv_sdA[None]
            cb = cb * inv_sdB[None]
        ca = ca * mask_f[None]
        cb = cb * mask_f[None]
        na = ca / torch.sqrt((ca * ca).sum(-1, keepdim=True) / M_mask + 1e-8)
        nb = cb / torch.sqrt((cb * cb).sum(-1, keepdim=True) / M_mask + 1e-8)

        s_ab, s_ba, w_part = _top_bottom_scores(na, nb, big_neg, big_pos, k_word)
        r_ab[s:e] = s_ab
        r_ba[s:e] = s_ba
        if w_part is not None:
            if w_bufs is None:
                w_bufs = {key: np.empty((M, k_word), dtype=arr.dtype) for key, arr in w_part.items()}
            for key, arr in w_part.items():
                w_bufs[key][s:e] = arr
        del xa, xb, ca, cb, na, nb

    r_ab_res = (r_ab - (cov_chunk @ model.beta_AB)).astype(np.float32)
    r_ba_res = (r_ba - (cov_chunk @ model.beta_BA)).astype(np.float32)
    T = np.minimum((r_ab_res - model.med_AB) / model.scale_AB, (r_ba_res - model.med_BA) / model.scale_BA).astype(np.float32)
    out: Dict[str, Any] = {
        "r_ab": r_ab,
        "r_ba": r_ba,
        "r_ab_res": r_ab_res,
        "r_ba_res": r_ba_res,
        "T": T,
    }
    if w_bufs is not None:
        out.update(w_bufs)
    return out


def score_odd_pair_gpu(
    A_cpu: torch.Tensor,
    B_cpu: torch.Tensor,
    ctl_pool_mask: np.ndarray,
    cov: np.ndarray,
    mask_gpu: torch.Tensor,
    zscore_var: bool,
    dev: torch.device,
    keep_words: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Score all units in `A_cpu, B_cpu` with exact LOO on `ctl_pool_mask` and non-LOO on all others."""
    U = A_cpu.shape[0]
    ref_m = np.asarray(ctl_pool_mask, dtype=bool)
    ref_idx = np.nonzero(ref_m)[0]
    oth_idx = np.nonzero(~ref_m)[0]

    model = fit_control_model(
        A_cpu[ref_idx],
        B_cpu[ref_idx],
        cov[ref_idx],
        mask_gpu=mask_gpu,
        zscore_var=zscore_var,
        dev=dev,
        keep_words=keep_words,
        top_k=top_k,
    )

    out: Dict[str, Any] = {
        "r_ab": np.zeros(U, np.float32),
        "r_ba": np.zeros(U, np.float32),
        "r_ab_res": np.zeros(U, np.float32),
        "r_ba_res": np.zeros(U, np.float32),
        "T": np.zeros(U, np.float32),
        "_control_model": model,
    }
    out["r_ab"][ref_idx] = model.r_ab_ctl
    out["r_ba"][ref_idx] = model.r_ba_ctl
    out["r_ab_res"][ref_idx] = model.r_ab_res_ctl
    out["r_ba_res"][ref_idx] = model.r_ba_res_ctl
    out["T"][ref_idx] = model.T_ctl

    if keep_words and model.words_ctl is not None:
        for key, arr in model.words_ctl.items():
            out[key] = np.empty((U, arr.shape[1]), dtype=arr.dtype)
            out[key][ref_idx] = arr

    if len(oth_idx) > 0:
        oth_res = score_chunk_against_controls(
            A_cpu[oth_idx],
            B_cpu[oth_idx],
            cov[oth_idx],
            model=model,
            mask_gpu=mask_gpu,
            dev=dev,
            keep_words=keep_words,
            top_k=top_k,
        )
        for key in ("r_ab", "r_ba", "r_ab_res", "r_ba_res", "T"):
            out[key][oth_idx] = oth_res[key]
        if keep_words:
            for key in model.words_ctl or ():
                out[key][oth_idx] = oth_res[key]

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


def extract_replicated_pieces(scores_dict: Dict[str, Any], global_u: int, raw_str: Sequence[str], k: int = 5) -> Dict[str, Any]:
    """Return replicated positive & negative pieces from A->B and B->A for unit `global_u`."""
    rep_pos: List[Tuple[str, float, int]] = []
    rep_neg: List[Tuple[str, float, int]] = []
    seen_p: Set[int] = set()
    seen_n: Set[int] = set()
    width = scores_dict["topA_p"].shape[1]
    for j in range(width):
        pid = int(scores_dict["topA_p"][global_u, j])
        if scores_dict["posA_onB"][global_u, j] > 0 and pid not in seen_p:
            seen_p.add(pid)
            rep_pos.append((raw_str[pid], round(float(scores_dict["posA_onB"][global_u, j]), 2), pid))
        pid2 = int(scores_dict["topB_p"][global_u, j])
        if scores_dict["posB_onA"][global_u, j] > 0 and pid2 not in seen_p:
            seen_p.add(pid2)
            rep_pos.append((raw_str[pid2], round(float(scores_dict["posB_onA"][global_u, j]), 2), pid2))
        nid = int(scores_dict["topA_n"][global_u, j])
        if scores_dict["negA_onB"][global_u, j] < 0 and nid not in seen_n:
            seen_n.add(nid)
            rep_neg.append((raw_str[nid], round(float(-scores_dict["negA_onB"][global_u, j]), 2), nid))
        nid2 = int(scores_dict["topB_n"][global_u, j])
        if scores_dict["negB_onA"][global_u, j] < 0 and nid2 not in seen_n:
            seen_n.add(nid2)
            rep_neg.append((raw_str[nid2], round(float(-scores_dict["negB_onA"][global_u, j]), 2), nid2))
    rep_pos.sort(key=lambda x: -x[1])
    rep_neg.sort(key=lambda x: -x[1])
    return {"pos": rep_pos[:k], "neg": rep_neg[:k]}


def run_null_splits(
    A_ctl: torch.Tensor,
    B_ctl: torch.Tensor,
    cov_ctl: np.ndarray,
    mask_gpu: torch.Tensor,
    zscore_var: bool,
    dev: torch.device,
    n_splits: int = 50,
    seed: int = 0,
) -> Dict[str, Any]:
    """Random half-control null splits re-fitting all centering, OLS, and robust scaling inside each split."""
    C = A_ctl.shape[0]
    rng = np.random.default_rng(int(seed))
    aucs: List[float] = []
    fdp05_cnts: List[int] = []
    half = C // 2
    for _ in range(int(n_splits)):
        perm = rng.permutation(C)
        fake_m = np.zeros(C, dtype=bool)
        fake_m[perm[:half]] = True
        ref_m = ~fake_m
        sc_sp = score_odd_pair_gpu(A_ctl, B_ctl, ref_m, cov_ctl, mask_gpu, zscore_var, dev, keep_words=False)
        r_abr, r_bar, Ts = sc_sp["r_ab_res"], sc_sp["r_ba_res"], sc_sp["T"]
        pok_f = (r_abr[fake_m] > 0) & (r_bar[fake_m] > 0)
        aucs.append(auc(Ts[fake_m], Ts[ref_m]))
        fdp05_cnts.append(empirical_fdp(Ts[fake_m], pok_f, int(fake_m.sum()), Ts[ref_m], 0.05)["count"])
    return {
        "n": int(n_splits),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_min": round(float(np.min(aucs)), 4),
        "auc_max": round(float(np.max(aucs)), 4),
        "fdp05_false_mean": round(float(np.mean(fdp05_cnts)), 2),
        "fdp05_false_max": int(np.max(fdp05_cnts)),
    }


def serialize_control_model(model: ControlModel) -> Dict[str, Any]:
    """Convert a `ControlModel` to CPU tensors/arrays safe for `torch.save`."""
    return {
        "C": int(model.C),
        "zscore_var": bool(model.zscore_var),
        "sumA": model.sumA.detach().cpu(),
        "sum2A": model.sum2A.detach().cpu(),
        "sumB": model.sumB.detach().cpu(),
        "sum2B": model.sum2B.detach().cpu(),
        "muA": model.muA.detach().cpu(),
        "varA": model.varA.detach().cpu(),
        "muB": model.muB.detach().cpu(),
        "varB": model.varB.detach().cpu(),
        "beta_AB": np.asarray(model.beta_AB, dtype=np.float64),
        "beta_BA": np.asarray(model.beta_BA, dtype=np.float64),
        "med_AB": float(model.med_AB),
        "scale_AB": float(model.scale_AB),
        "med_BA": float(model.med_BA),
        "scale_BA": float(model.scale_BA),
        "r_ab_ctl": np.asarray(model.r_ab_ctl, dtype=np.float32),
        "r_ba_ctl": np.asarray(model.r_ba_ctl, dtype=np.float32),
        "r_ab_res_ctl": np.asarray(model.r_ab_res_ctl, dtype=np.float32),
        "r_ba_res_ctl": np.asarray(model.r_ba_res_ctl, dtype=np.float32),
        "T_ctl": np.asarray(model.T_ctl, dtype=np.float32),
        "words_ctl": model.words_ctl,
    }


def deserialize_control_model(d: Dict[str, Any], dev: torch.device) -> ControlModel:
    """Restore a `ControlModel` onto target device `dev` from serialized dict."""
    return ControlModel(
        C=int(d["C"]),
        zscore_var=bool(d["zscore_var"]),
        sumA=d["sumA"].to(device=dev, dtype=torch.float32),
        sum2A=d["sum2A"].to(device=dev, dtype=torch.float32),
        sumB=d["sumB"].to(device=dev, dtype=torch.float32),
        sum2B=d["sum2B"].to(device=dev, dtype=torch.float32),
        muA=d["muA"].to(device=dev, dtype=torch.float32),
        varA=d["varA"].to(device=dev, dtype=torch.float32),
        muB=d["muB"].to(device=dev, dtype=torch.float32),
        varB=d["varB"].to(device=dev, dtype=torch.float32),
        beta_AB=np.asarray(d["beta_AB"], dtype=np.float64),
        beta_BA=np.asarray(d["beta_BA"], dtype=np.float64),
        med_AB=float(d["med_AB"]),
        scale_AB=float(d["scale_AB"]),
        med_BA=float(d["med_BA"]),
        scale_BA=float(d["scale_BA"]),
        r_ab_ctl=np.asarray(d["r_ab_ctl"], dtype=np.float32),
        r_ba_ctl=np.asarray(d["r_ba_ctl"], dtype=np.float32),
        r_ab_res_ctl=np.asarray(d["r_ab_res_ctl"], dtype=np.float32),
        r_ba_res_ctl=np.asarray(d["r_ba_res_ctl"], dtype=np.float32),
        T_ctl=np.asarray(d["T_ctl"], dtype=np.float32),
        words_ctl=d.get("words_ctl"),
    )


def _normalize_half_c(
    oddC_ctl: torch.Tensor,
    oddC_neu: torch.Tensor,
    mask_gpu: torch.Tensor,
    zscore_var: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """LOO-center controls and non-LOO-center units on Half C, returning masked RMS-normalized vectors."""
    nc = int(oddC_ctl.shape[0])
    mask_f = mask_gpu.float()
    M_mask = float(mask_f.sum().item())
    vfloor2 = VAR_FLOOR * VAR_FLOOR

    sumC = oddC_ctl.sum(0)
    muC = sumC / nc
    muC_loo = (sumC[None] - oddC_ctl) / max(nc - 1, 1)
    c_ctl = oddC_ctl - muC_loo
    c_neu = oddC_neu - muC[None]
    if zscore_var:
        sum2C = (oddC_ctl * oddC_ctl).sum(0)
        varC_all = ((sum2C - nc * muC * muC) / max(nc - 1, 1)).clamp(min=0.0)
        varC_loo = (((sum2C[None] - oddC_ctl * oddC_ctl) - (nc - 1) * muC_loo * muC_loo) / max(nc - 2, 1)).clamp(min=0.0)
        c_ctl = c_ctl / torch.sqrt(varC_loo + vfloor2)
        c_neu = c_neu / torch.sqrt(varC_all[None] + vfloor2)

    c_ctl = c_ctl * mask_f[None]
    c_neu = c_neu * mask_f[None]
    normC_ctl = (c_ctl / torch.sqrt((c_ctl * c_ctl).sum(-1, keepdim=True) / M_mask + 1e-8)).cpu().numpy()
    normC_neu = (c_neu / torch.sqrt((c_neu * c_neu).sum(-1, keepdim=True) / M_mask + 1e-8)).cpu().numpy()
    return normC_ctl, normC_neu


def _piece_pair_score(vec_1d: np.ndarray, p_ids: Sequence[int], n_ids: Sequence[int]) -> float:
    p_idx = [int(x) for x in p_ids]
    n_idx = [int(x) for x in n_ids]
    denom = max(len(p_idx) + len(n_idx), 1)
    pos_s = float(vec_1d[p_idx].sum()) if p_idx else 0.0
    neg_s = float(vec_1d[n_idx].sum()) if n_idx else 0.0
    return (pos_s - neg_s) / denom


def score_half_c_matched(
    normC_ctl: np.ndarray,
    ctl_piece_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    normC_neu: np.ndarray,
    neu_piece_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
) -> Dict[str, Any]:
    """Like-for-like Half-C scoring: each unit AND each control is scored on its own A/B chosen pieces."""
    nc = int(normC_ctl.shape[0])
    nn = int(normC_neu.shape[0])
    if nc == 0 or nn == 0 or len(ctl_piece_pairs) != nc or len(neu_piece_pairs) != nn:
        return {
            "evaluated": 0,
            "matched": True,
            "half_c_auc": None,
            "frac_gt_0": None,
            "frac_p05_vs_ctl": None,
            "median_score_C": None,
            "ctl_median_score_C": None,
        }
    ctl_scores = np.array([_piece_pair_score(normC_ctl[c], *ctl_piece_pairs[c]) for c in range(nc)], dtype=np.float64)
    neu_scores = np.array([_piece_pair_score(normC_neu[j], *neu_piece_pairs[j]) for j in range(nn)], dtype=np.float64)
    ctl_sorted = np.sort(ctl_scores)
    ge = nc - np.searchsorted(ctl_sorted, neu_scores, side="left")
    p_vals = (1.0 + ge) / (nc + 1.0)
    return {
        "evaluated": nn,
        "matched": True,
        "half_c_auc": round(auc(neu_scores, ctl_scores), 4),
        "frac_gt_0": round(float(np.mean(neu_scores > 0.0)), 4),
        "frac_p05_vs_ctl": round(float(np.mean(p_vals <= 0.05)), 4),
        "median_score_C": round(float(np.median(neu_scores)), 3),
        "ctl_median_score_C": round(float(np.median(ctl_scores)), 3),
    }


def _eval_half_c_against_ctl(
    oddC_ctl: torch.Tensor,
    oddC_neu: torch.Tensor,
    piece_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    mask_gpu: torch.Tensor,
    zscore_var: bool,
) -> Dict[str, Any]:
    """Legacy (asymmetric) Half-C evaluator kept for backward compatibility; prefer ``score_half_c_matched``."""
    nc = int(oddC_ctl.shape[0])
    if nc == 0 or len(piece_pairs) == 0:
        return {"evaluated": 0, "half_c_auc": None, "frac_gt_0": None, "frac_p05_vs_ctl": None, "median_score_C": None}
    normC_ctl, normC_neu = _normalize_half_c(oddC_ctl, oddC_neu, mask_gpu, zscore_var)

    sc_c_vals: List[float] = []
    sc_c_ctl: List[float] = []
    p_c_list: List[float] = []
    for j, (p_ids, n_ids) in enumerate(piece_pairs):
        p_idx = [int(x) for x in p_ids]
        n_idx = [int(x) for x in n_ids]
        denom = max(len(p_idx) + len(n_idx), 1)
        val_n = _piece_pair_score(normC_neu[j], p_idx, n_idx)
        ctl_sc = (normC_ctl[:, p_idx].sum(1) - normC_ctl[:, n_idx].sum(1)) / denom
        sc_c_vals.append(val_n)
        sc_c_ctl.append(float(ctl_sc[j % nc]))
        ge = int(np.sum(ctl_sc >= val_n))
        p_c_list.append((1.0 + ge) / (nc + 1.0))

    return {
        "evaluated": len(piece_pairs),
        "matched": False,
        "half_c_auc": round(auc(sc_c_vals, sc_c_ctl), 4),
        "frac_gt_0": round(float(np.mean([v > 0.0 for v in sc_c_vals])), 4),
        "frac_p05_vs_ctl": round(float(np.mean([p <= 0.05 for p in p_c_list])), 4),
        "median_score_C": round(float(np.median(sc_c_vals)), 3),
    }


@torch.no_grad()
def eval_half_c_dual(
    rt: Any,
    lm: nn.Module,
    layer: int,
    texts: Dict[str, Any],
    lex_gpu: torch.Tensor,
    batch_units: int,
    readout_mode: str,
    deltas_cols_c: torch.Tensor,
    deltas_iso_c: torch.Tensor,
    deltas_neu_c: torch.Tensor,
    piece_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    zscore_var: bool = False,
) -> Dict[str, Any]:
    """Legacy Half-C evaluator (asymmetric control scoring); prefer ``eval_half_c_matched``."""
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    n_cols = int(deltas_cols_c.shape[0])
    n_iso = int(deltas_iso_c.shape[0])
    n_neu = int(deltas_neu_c.shape[0])
    if n_neu == 0 or len(piece_pairs) == 0:
        return {"evaluated": 0, "vs_columns": None, "vs_isotropic": None}
    if n_neu != len(piece_pairs):
        raise ValueError(f"deltas_neu_c rows ({n_neu}) must match len(piece_pairs) ({len(piece_pairs)})")

    sub_deltas = torch.cat([deltas_cols_c.to(dev), deltas_iso_c.to(dev), deltas_neu_c.to(dev)], dim=0)
    grpC = prepare_group_cache(rt, lm, layer, texts, "C")
    resC = run_forward_half(lm, W_U, layer, grpC, sub_deltas, batch_units, readout_mode=readout_mode)
    del grpC, sub_deltas

    oddC = resC[f"odd_{readout_mode}"].to(device=dev, dtype=torch.float32)
    del resC
    oddC_cols = oddC[:n_cols]
    oddC_iso = oddC[n_cols : n_cols + n_iso]
    oddC_neu = oddC[n_cols + n_iso :]

    res_cols = _eval_half_c_against_ctl(oddC_cols, oddC_neu, piece_pairs, lex_gpu, zscore_var=zscore_var)
    res_iso = _eval_half_c_against_ctl(oddC_iso, oddC_neu, piece_pairs, lex_gpu, zscore_var=zscore_var)
    del oddC
    return {"evaluated": len(piece_pairs), "matched": False, "vs_columns": res_cols, "vs_isotropic": res_iso}


@torch.no_grad()
def eval_half_c_matched(
    rt: Any,
    lm: nn.Module,
    layer: int,
    texts: Dict[str, Any],
    lex_gpu: torch.Tensor,
    batch_units: int,
    readout_mode: str,
    deltas_cols_c: torch.Tensor,
    deltas_iso_c: torch.Tensor,
    deltas_neu_c: torch.Tensor,
    neu_piece_pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    ctl_piece_pairs_col: Sequence[Tuple[Sequence[int], Sequence[int]]],
    ctl_piece_pairs_iso: Sequence[Tuple[Sequence[int], Sequence[int]]],
    zscore_var: bool = False,
) -> Dict[str, Any]:
    """Evaluate Half C like-for-like against both control pools (each control scored on its own A/B pieces)."""
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    n_cols = int(deltas_cols_c.shape[0])
    n_iso = int(deltas_iso_c.shape[0])
    n_neu = int(deltas_neu_c.shape[0])
    if n_neu == 0 or len(neu_piece_pairs) == 0:
        return {"evaluated": 0, "matched": True, "vs_columns": None, "vs_isotropic": None}
    if n_neu != len(neu_piece_pairs):
        raise ValueError(f"deltas_neu_c rows ({n_neu}) must match len(neu_piece_pairs) ({len(neu_piece_pairs)})")
    if n_cols != len(ctl_piece_pairs_col) or n_iso != len(ctl_piece_pairs_iso):
        raise ValueError("control piece pairs length must match control delta rows")

    sub_deltas = torch.cat([deltas_cols_c.to(dev), deltas_iso_c.to(dev), deltas_neu_c.to(dev)], dim=0)
    grpC = prepare_group_cache(rt, lm, layer, texts, "C")
    resC = run_forward_half(lm, W_U, layer, grpC, sub_deltas, batch_units, readout_mode=readout_mode)
    del grpC, sub_deltas

    oddC = resC[f"odd_{readout_mode}"].to(device=dev, dtype=torch.float32)
    del resC
    oddC_cols = oddC[:n_cols]
    oddC_iso = oddC[n_cols : n_cols + n_iso]
    oddC_neu = oddC[n_cols + n_iso :]

    nc_col, nn_col = _normalize_half_c(oddC_cols, oddC_neu, lex_gpu, zscore_var=zscore_var)
    res_cols = score_half_c_matched(nc_col, ctl_piece_pairs_col, nn_col, neu_piece_pairs)
    nc_iso, nn_iso = _normalize_half_c(oddC_iso, oddC_neu, lex_gpu, zscore_var=zscore_var)
    res_iso = score_half_c_matched(nc_iso, ctl_piece_pairs_iso, nn_iso, neu_piece_pairs)
    del oddC
    return {"evaluated": len(neu_piece_pairs), "matched": True, "vs_columns": res_cols, "vs_isotropic": res_iso}


@torch.no_grad()
def _eval_own_words(
    A_sub: torch.Tensor,
    B_sub: torch.Tensor,
    pos_ids_list: Sequence[Sequence[int]],
    neg_ids_list: Sequence[Sequence[int]],
    model: ControlModel,
    mask_gpu: torch.Tensor,
    dev: torch.device,
    loo: bool,
) -> np.ndarray:
    """Evaluate normalized A/B odd shift agreement with each unit's own top-10 positive minus negative pieces."""
    mask_f = mask_gpu.float()
    M_mask = float(mask_f.sum().item())
    vfloor2 = VAR_FLOOR * VAR_FLOOR
    C = int(model.C)
    inv_sdA = torch.rsqrt(model.varA + vfloor2) if model.zscore_var else None
    inv_sdB = torch.rsqrt(model.varB + vfloor2) if model.zscore_var else None

    M_u = int(A_sub.shape[0])
    scores = np.zeros(M_u, dtype=np.float32)
    for s in range(0, M_u, 128):
        e = min(M_u, s + 128)
        xa = A_sub[s:e].to(device=dev, dtype=torch.float32)
        xb = B_sub[s:e].to(device=dev, dtype=torch.float32)
        if loo:
            muA_u = (model.sumA[None] - xa) / max(C - 1, 1)
            muB_u = (model.sumB[None] - xb) / max(C - 1, 1)
            ca, cb = xa - muA_u, xb - muB_u
            if model.zscore_var:
                vA_loo = (((model.sum2A[None] - xa * xa) - (C - 1) * muA_u * muA_u) / max(C - 2, 1)).clamp(min=0.0)
                vB_loo = (((model.sum2B[None] - xb * xb) - (C - 1) * muB_u * muB_u) / max(C - 2, 1)).clamp(min=0.0)
                ca = ca / torch.sqrt(vA_loo + vfloor2)
                cb = cb / torch.sqrt(vB_loo + vfloor2)
        else:
            ca = xa - model.muA[None]
            cb = xb - model.muB[None]
            if model.zscore_var:
                ca = ca * inv_sdA[None]
                cb = cb * inv_sdB[None]
        ca = ca * mask_f[None]
        cb = cb * mask_f[None]
        na = ca / torch.sqrt((ca * ca).sum(-1, keepdim=True) / M_mask + 1e-8)
        nb = cb / torch.sqrt((cb * cb).sum(-1, keepdim=True) / M_mask + 1e-8)
        vec_cpu = (0.5 * (na + nb)).cpu().numpy()
        for li in range(e - s):
            gi = s + li
            pids = [int(t) for t in pos_ids_list[gi][:10]]
            nids = [int(t) for t in neg_ids_list[gi][:10]]
            scores[gi] = float(vec_cpu[li, pids].mean() - vec_cpu[li, nids].mean())
        del xa, xb, ca, cb, na, nb
    return scores


@torch.no_grad()
def compute_weight_word_ctl_null(
    rt: Any,
    col_dirs: np.ndarray,
    A_ctl: torch.Tensor,
    B_ctl: torch.Tensor,
    model: ControlModel,
    mask_gpu: torch.Tensor,
    dev: torch.device,
) -> np.ndarray:
    """Compute exact LOO own-word unembed agreement null distribution `W(u_c)` over columns control directions."""
    fnw, W_U = rt.final_norm_weight, rt.lm_head_weight
    C = int(col_dirs.shape[0])
    ctl_eye_pos: List[List[int]] = []
    ctl_eye_neg: List[List[int]] = []
    for s in range(0, C, 128):
        cv = torch.from_numpy(col_dirs[s : s + 128]).to(device=dev, dtype=W_U.dtype)
        cl = unembed_gemma4(cv, fnw, W_U)
        ctl_eye_pos.extend(label_top_ids(cv, fnw, W_U, cl, k=10))
        ctl_eye_neg.extend(label_top_ids(-cv, fnw, W_U, -cl, k=10))
        del cv, cl
    return _eval_own_words(A_ctl, B_ctl, ctl_eye_pos, ctl_eye_neg, model=model, mask_gpu=mask_gpu, dev=dev, loo=True)


@torch.no_grad()
def score_weight_word_chunk(
    eye_pos_layer: np.ndarray,
    eye_neg_layer: np.ndarray,
    neurons: Sequence[int],
    A_chunk: torch.Tensor,
    B_chunk: torch.Tensor,
    model: ControlModel,
    mask_gpu: torch.Tensor,
    dev: torch.device,
) -> np.ndarray:
    """Score normalized forced-odd agreement `W(u_n)` with naked-eye unembed top-10 pieces for one chunk."""
    pos_neu = [eye_pos_layer[int(n), :10].tolist() for n in neurons]
    neg_neu = [eye_neg_layer[int(n), :10].tolist() for n in neurons]
    return _eval_own_words(A_chunk, B_chunk, pos_neu, neg_neu, model=model, mask_gpu=mask_gpu, dev=dev, loo=False)


def finalize_weight_word_tag(sw_neu: Optional[np.ndarray], sw_ctl: Optional[np.ndarray]) -> Dict[str, Any]:
    """Compute BH-corrected own-word direct-unembed diagnostic stats from neuron and control scores."""
    if sw_neu is None or sw_ctl is None or len(sw_ctl) == 0:
        return {"available": False}
    C = int(len(sw_ctl))
    sorted_c = np.sort(np.asarray(sw_ctl, dtype=np.float64))
    sw_neu_f = np.asarray(sw_neu, dtype=np.float32)
    ge = C - np.searchsorted(sorted_c, sw_neu_f.astype(np.float64), side="left")
    p_direct = (1.0 + ge) / (C + 1.0)
    q_direct = bh(p_direct)
    is_direct = (q_direct <= 0.05) & (sw_neu_f > 0.0)
    return {
        "available": True,
        "auc_own_word_vs_own_ctl": round(auc(sw_neu_f, sw_ctl), 4),
        "ctl_frac_gt0": round(float(np.mean(np.asarray(sw_ctl) > 0.0)), 4),
        "neu_frac_gt0": round(float(np.mean(sw_neu_f > 0.0)), 4),
        "direct_q05_count": int(np.sum(is_direct)),
        "p_direct": p_direct,
        "q_direct": q_direct,
        "is_direct": is_direct,
        "sw_neu": sw_neu_f,
    }


@torch.no_grad()
def corrected_weight_word_tag(
    rt: Any,
    layer: int,
    neurons: Sequence[int],
    col_dirs: np.ndarray,
    A_ctl: torch.Tensor,
    B_ctl: torch.Tensor,
    A_neu: torch.Tensor,
    B_neu: torch.Tensor,
    model: ControlModel,
    mask_gpu: torch.Tensor,
    dev: torch.device,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Corrected W(u_n) vs own-word W(u_c) diagnostic checking own weight-lens agreement."""
    dd = resolve_data_dir(data_dir)
    lens_path = dd / "index/lens.npz"
    if not lens_path.exists():
        return {"available": False}
    lens_npz = np.load(lens_path)
    eye_pos_l = lens_npz["eye_pos"][layer]
    eye_neg_l = lens_npz["eye_neg"][layer]
    sw_ctl = compute_weight_word_ctl_null(rt, col_dirs, A_ctl, B_ctl, model, mask_gpu, dev)
    sw_neu = score_weight_word_chunk(eye_pos_l, eye_neg_l, neurons, A_neu, B_neu, model, mask_gpu, dev)
    return finalize_weight_word_tag(sw_neu, sw_ctl)


def make_units(
    rt: Any,
    lm: nn.Module,
    w: ao.Weights,
    layer: int,
    neurons: Sequence[int],
    n_cols: int,
    seed: int,
    n_iso: int = 0,
    iso_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate unit write directions for columns controls, optional isotropic controls, and neurons."""
    dev = model_device(lm)
    N_ff = w.d_ff
    D = w.d_model
    w_post_np = w.norm(layer, "post_feedforward").cpu().numpy()
    writes_all = np.concatenate(
        [w.write_dirs(layer, list(range(a, min(a + 4096, N_ff)))).cpu().numpy() for a in range(0, N_ff, 4096)]
    )
    col_dirs = cz.control_dirs(seed, n_cols, w_post_np, "columns", writes_all)
    del writes_all
    refs_col = cz.control_refs(seed, n_cols, N_ff)
    ref_ceils_col = w.write_ceiling(layer, refs_col).cpu().numpy()

    iso_dirs = None
    ref_ceils_iso = None
    if n_iso > 0:
        isd = iso_seed if iso_seed is not None else seed + 999
        iso_dirs = cz.control_dirs(isd, n_iso, w_post_np, "isotropic")
        refs_iso = cz.control_refs(isd, n_iso, N_ff)
        ref_ceils_iso = w.write_ceiling(layer, refs_iso).cpu().numpy()

    if len(neurons) > 0:
        u_neu = w.write_dirs(layer, list(neurons))
        c_neu = w.write_ceiling(layer, list(neurons))
    else:
        u_neu = torch.empty((0, D), dtype=torch.float32, device=dev)
        c_neu = torch.empty((0,), dtype=torch.float32, device=dev)

    return {
        "col_dirs": col_dirs,
        "ref_ceils_col": ref_ceils_col,
        "iso_dirs": iso_dirs,
        "ref_ceils_iso": ref_ceils_iso,
        "u_neu": u_neu,
        "c_neu": c_neu,
        "neurons": [int(x) for x in neurons],
    }
