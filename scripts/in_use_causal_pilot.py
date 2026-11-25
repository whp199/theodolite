#!/usr/bin/env python3
"""Probe 1 (GPU): On-manifold ('in-use') natural-site causal ablation vs norm-matched 'columns' controls.

Evaluates L=[2,14,25,29,55,58,59] (N=64 random pure_4_4 neurons/layer x 8 pure doc-disjoint events:
4 fit docs, 4 test docs). At exact natural peaks, silences neuron n over [pk..min(seq-1, pk+3)] and
pairs against subtracting a norm-matched random W_down-column mixture ('columns') at the exact same
passages and activation scales.

Measures:
  M1: Norm-matched causal necessity KL(pk) and KL(span) (AUC, Wilcoxon p, p<=0.05 fraction)
  M2: Probability-weighted Delta p = p_base - p_perturbed directional carry (fit->test, test->fit)
  M3: Pre-fixed lens.npz eye_10 Delta p drop validation (vs paired control's unembedded eye_10)
  M4: (L < 30) On-manifold downstream bridge to all 29,265 page-described L30-L54 census neurons
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.label_gate import label_top_ids  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402
from backend.theodolite_math import gemma4_rms_norm  # noqa: E402
from scripts.downstream_pilot import _load_target_universe  # noqa: E402
from scripts.in_use_common import (  # noqa: E402
    build_event_sequence,
    collect_face_events,
    emp_p_values,
    layer_face_counts_vectorized,
    load_passages_and_doc_halves,
    score_directional_carry,
    summarize_stats,
)


class _StopAfterLayer(Exception):
    pass


def _special_vocab_mask(V: int, tokenizer: Any) -> np.ndarray:
    mask = np.zeros(V, dtype=bool)
    for sid in getattr(tokenizer, "all_special_ids", []) or []:
        if 0 <= int(sid) < V:
            mask[int(sid)] = True
    for fixed in (0, 1, 2, 3, 4):
        if fixed < V:
            mask[fixed] = True
    return mask


def _pick_layer_neurons(
    P: Dict[str, Any],
    windows: np.ndarray,
    act_scale: Any,
    doc_id: np.ndarray,
    half_id: np.ndarray,
    pure_h0_mask: np.ndarray,
    pure_h1_mask: np.ndarray,
    layer: int,
    n_neurons: int,
    *,
    eos_id: int,
    bos_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Select N distinct pure_4_4 neurons at `layer`, picking the stronger face when both qualify."""
    cnt = layer_face_counts_vectorized(
        P, windows, act_scale, half_id, pure_h0_mask, pure_h1_mask, layer, eos_id=eos_id, bos_id=bos_id
    )
    elig_idx = np.flatnonzero(cnt["neu_pure44"])
    n_elig = int(len(elig_idx))
    if n_elig < n_neurons:
        raise RuntimeError(f"L{layer}: only {n_elig} pure_4_4 neurons (< {n_neurons})")

    rng = np.random.default_rng(2026112501 + int(layer))
    chosen = sorted(int(elig_idx[int(i)]) for i in rng.choice(n_elig, size=int(n_neurons), replace=False))
    picked: List[Dict[str, Any]] = []
    for n in chosen:
        opts = []
        for face in ("pos", "neg"):
            if not bool(cnt[f"{face}_pure44"][n]):
                continue
            ev = collect_face_events(
                P, windows, act_scale, pure_h0_mask, pure_h1_mask, doc_id, half_id, layer, n, face, eos_id=eos_id, bos_id=bos_id
            )
            if len(ev["pure_h0"]) >= 4 and len(ev["pure_h1"]) >= 4:
                score = (len(ev["pure_h0"]) + len(ev["pure_h1"]), ev["extreme"], 1 if face == "pos" else 0)
                opts.append((score, face, ev))
        opts.sort(key=lambda x: x[0], reverse=True)
        if not opts:
            raise RuntimeError(f"L{layer} n={n}: failed event verification after pure44 filter")
        _, face, ev = opts[0]
        picked.append({"n": int(n), "face": face, "fit": ev["pure_h0"][:4], "test": ev["pure_h1"][:4], "extreme": ev["extreme"]})
    return picked, n_elig


@torch.no_grad()
def _control_eye_ids(
    ctl_dirs: torch.Tensor,
    faces: Sequence[str],
    w_post: torch.Tensor,
    fnw: torch.Tensor,
    W_U: torch.Tensor,
    k: int = 10,
) -> np.ndarray:
    """Compute exact fp32 softcapped eye top-k vocab IDs for paired signed 'columns' control directions."""
    sgn = torch.tensor([1.0 if f == "pos" else -1.0 for f in faces], dtype=torch.float32, device=ctl_dirs.device)
    vecs = (sgn[:, None] * ctl_dirs.float()) * w_post.to(ctl_dirs.device).float()[None, :]
    hn_bf16 = gemma4_rms_norm(vecs.to(W_U.dtype), fnw)
    bf16_logits = ao.softcap_of_tensor(hn_bf16, W_U) if hasattr(ao, "softcap_of_tensor") else logits_fp32(hn_bf16, W_U)
    ids = label_top_ids(vecs, fnw, W_U, bf16_logits, k=k)
    return np.asarray(ids, dtype=np.int32)


@torch.no_grad()
def _run_batch_0_sanity(
    lm: Any,
    W_U: torch.Tensor,
    layer: int,
    ib: torch.Tensor,
    ab: torch.Tensor,
    pb: torch.Tensor,
    pos_mat: torch.Tensor,
    in_span_eval: torch.Tensor,
    span_mask: torch.Tensor,
    y_mlp: torch.Tensor,
    y_abl: torch.Tensor,
    y_ctl: torch.Tensor,
    w_post_l: torch.Tensor,
    rb: torch.Tensor,
    pt_i: torch.Tensor,
    post_rec: torch.Tensor,
    post_abl: torch.Tensor,
    target_ns: Sequence[int],
    lsm_rec_combined: torch.Tensor,
    lsm_abl_combined: torch.Tensor,
) -> Dict[str, float]:
    """Verify bit-exact F.linear activation-space ablation vs down_proj zeroing hook and control norm match."""
    b, T = ib.shape
    D = int(post_rec.shape[-1])
    V = int(W_U.shape[0])
    br = torch.arange(b, device=ib.device)[:, None].expand(b, 4)

    cap_in: Dict[str, torch.Tensor] = {}
    hk_in = lm.layers[layer].register_forward_pre_hook(lambda _m, a: cap_in.__setitem__("h_in", a[0].detach()))
    try:
        out_full = lm(input_ids=ib, attention_mask=ab, use_cache=False)
    finally:
        hk_in.remove()

    h_full = out_full.last_hidden_state[br, pos_mat].reshape(b * 4, D)
    lsm_full = F.log_softmax(logits_fp32(h_full, W_U), dim=-1).view(b, 4, V)

    out_rec_b = ao.forward_from_layer(lm, layer + 1, post_rec, ab, pb)
    h_rec_b = out_rec_b[br, pos_mat].reshape(b * 4, D)
    lsm_rec_b = F.log_softmax(logits_fp32(h_rec_b, W_U), dim=-1).view(b, 4, V)
    kl_noop_same_b = float((lsm_full.exp() * (lsm_full - lsm_rec_b)).sum(-1).mean().item())
    kl_noop_comb = float((lsm_full.exp() * (lsm_full - lsm_rec_combined)).sum(-1).mean().item())
    max_lsm_noop = float((lsm_full - lsm_rec_b).abs().max().item())

    t_idx = torch.tensor(target_ns, dtype=torch.long, device=ib.device)
    rows_i = torch.arange(b, device=ib.device)
    cap_y: Dict[str, torch.Tensor] = {}

    def _zero_hook(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> Tuple[Any, ...]:
        x = f_args[0].clone()
        orig = x[rows_i, :, t_idx]
        x[rows_i, :, t_idx] = torch.where(span_mask, torch.zeros_like(orig), orig)
        return (x,) + tuple(f_args[1:])

    hk_z = lm.layers[layer].mlp.down_proj.register_forward_pre_hook(_zero_hook)
    hk_y = lm.layers[layer].mlp.down_proj.register_forward_hook(lambda _m, _a, o: cap_y.__setitem__("y_hook", o.detach()))
    try:
        out_hook = ao.forward_from_layer(lm, layer, cap_in["h_in"], ab, pb)
    finally:
        hk_z.remove()
        hk_y.remove()

    y_abl_vs_hook_max_abs = float((y_abl - cap_y["y_hook"]).abs().max().item())
    wp_f = w_post_l.to(device=y_mlp.device, dtype=torch.float32)
    ctl_norm = ((y_mlp[rb, pt_i].float() - y_ctl[rb, pt_i].float()) * wp_f).norm(dim=-1)
    abl_norm = ((y_mlp[rb, pt_i].float() - y_abl[rb, pt_i].float()) * wp_f).norm(dim=-1)
    achieved_ratio_med = float((ctl_norm / abl_norm.clamp(min=1e-6)).median().item())

    out_abl_b = ao.forward_from_layer(lm, layer + 1, post_abl, ab, pb)
    h_abl_b = out_abl_b[br, pos_mat].reshape(b * 4, D)
    lsm_abl_b = F.log_softmax(logits_fp32(h_abl_b, W_U), dim=-1).view(b, 4, V)
    h_hook = out_hook[br, pos_mat].reshape(b * 4, D)
    lsm_hook = F.log_softmax(logits_fp32(h_hook, W_U), dim=-1).view(b, 4, V)

    kl_alg_same_b = (lsm_rec_b.exp() * (lsm_rec_b - lsm_abl_b)).sum(-1)
    kl_hook_same_b = (lsm_full.exp() * (lsm_full - lsm_hook)).sum(-1)
    kl_diff_abs = (kl_alg_same_b - kl_hook_same_b).abs()
    mean_kl_diff_same_b = float(kl_diff_abs.mean().item())
    max_kl_diff_same_b = float(kl_diff_abs.max().item())
    max_kl_comb_diff = float(((lsm_rec_combined.exp() * (lsm_rec_combined - lsm_abl_combined)).sum(-1) - kl_hook_same_b).abs().max().item())
    del out_full, out_rec_b, out_abl_b, out_hook, h_full, h_rec_b, h_abl_b, h_hook, lsm_full, lsm_rec_b, lsm_abl_b, lsm_hook, cap_y
    return {
        "noop_kl_same_batch": round(kl_noop_same_b, 8),
        "noop_kl_combined_3b": round(kl_noop_comb, 8),
        "noop_max_lsm_diff_same_batch": round(max_lsm_noop, 6),
        "y_abl_vs_hook_max_abs": round(y_abl_vs_hook_max_abs, 6),
        "achieved_ctl_over_abl_post_norm_median": round(achieved_ratio_med, 6),
        "mean_kl_alg_vs_hook_same_batch": round(mean_kl_diff_same_b, 8),
        "max_kl_alg_vs_hook_same_batch": round(max_kl_diff_same_b, 8),
        "max_kl_alg_3b_vs_hook_b": round(max_kl_comb_diff, 8),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe 1 GPU: In-use natural-site ablation vs columns controls")
    ap.add_argument("--layers", default="2,14,25,29,55,58,59")
    ap.add_argument("--n-neurons", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default="experiments/results/in_use_kill_pilots")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()

    loaded = load_passages_and_doc_halves(data_dir)
    P = loaded["P"]
    act_scale = loaded["act_scale"]
    vocab = loaded["vocab"]
    windows = loaded["windows"]
    doc_id = loaded["doc_id"]
    half_id = loaded["half_id"]
    pure_h0_mask = loaded["pure_h0_mask"]
    pure_h1_mask = loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]

    lens = np.load(data_dir / "index/lens.npz")
    target_uni = _load_target_universe(data_dir, act_scale)
    M_targets = int(target_uni["M_targets"])
    target_meta = target_uni["target_meta"]
    print(f"Loaded target universe: {M_targets} L30-L54 page-described neurons", flush=True)

    rt = make_runtime(load_jlens=False).load()
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    w_ops = ao.Weights(lm, rt)
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    fnw = rt.final_norm_weight
    V, D = int(W_U.shape[0]), int(W_U.shape[1])
    bad_tok_mask = _special_vocab_mask(V, rt.tokenizer)

    # Stage L30-L54 target indices on GPU once
    targets_gpu: Dict[int, Dict[str, Any]] = {}
    for t_l, info in target_uni["by_layer"].items():
        targets_gpu[t_l] = {
            "idx": torch.tensor(info["nids"], dtype=torch.long, device=dev),
            "inv_ext": torch.from_numpy(info["inv_ext"]).to(dev),
            "offset": int(info["offset"]),
            "count": int(info["count"]),
        }

    sanity_report: Dict[str, Any] = {}
    layer_results: List[Dict[str, Any]] = []
    neuron_ids_meta: Dict[str, Any] = {}

    for li, l in enumerate(layers):
        tl0 = time.time()
        picked, n_elig = _pick_layer_neurons(
            P, windows, act_scale, doc_id, half_id, pure_h0_mask, pure_h1_mask, l, args.n_neurons, eos_id=eos_id, bos_id=bos_id
        )
        N = len(picked)
        nids = [p["n"] for p in picked]
        faces = [p["face"] for p in picked]
        neuron_ids_meta[str(l)] = [f"n{p['n']}{'p' if p['face']=='pos' else 'm'}" for p in picked]

        w_down_l = w_ops.w_down(l).to(dev)
        w_post_l = w_ops.norm(l, "post_feedforward").to(dev)
        s_l = float(w_ops.scalars[l])
        post_ffn_ln = lm.layers[l].post_feedforward_layernorm

        mix_k = 64
        rng_ctl = np.random.default_rng(2026112501 + 1000 + l)
        ctl_col_idx = torch.from_numpy(
            np.stack([rng_ctl.choice(w_ops.d_ff, size=mix_k, replace=False) for _ in range(N)], axis=0)
        ).to(device=dev, dtype=torch.long)  # [N, 64]
        ctl_coeffs = torch.from_numpy(rng_ctl.standard_normal((N, mix_k)).astype(np.float32)).to(device=dev)
        w_ctl_dirs = torch.empty((N, D), dtype=torch.float32, device=dev)
        wp_f = w_post_l.float()
        for i in range(N):
            tn = (w_down_l[:, nids[i]].float() * wp_f).norm().clamp(min=1e-30)
            v_raw = w_down_l.index_select(1, ctl_col_idx[i]).float() @ ctl_coeffs[i]
            scale_i = tn / (v_raw * wp_f).norm().clamp(min=1e-30)
            ctl_coeffs[i] = ctl_coeffs[i] * scale_i
            w_ctl_dirs[i] = v_raw * scale_i

        # Pre-fixed eye_10 IDs for real neurons and exact matched control directions
        eye_neu = np.stack([lens[f"eye_{p['face']}"][l, p["n"], :10] for p in picked], axis=0).astype(np.int32)
        eye_ctl = _control_eye_ids(w_ctl_dirs, faces, w_post_l, fnw, W_U, k=10)

        # Build 8 sequences per neuron: first 4 fit (pure_h0), next 4 test (pure_h1)
        seqs: List[List[int]] = []
        peak_pos: List[int] = []
        row_meta: List[Tuple[int, int, str]] = []  # (local_neuron_idx, split_0fit_1test, face)
        for ni, p in enumerate(picked):
            for split_idx, split_key in ((0, "fit"), (1, "test")):
                for ev in p[split_key]:
                    sq, pk = build_event_sequence(windows, ev["win"], ev["at"], bos_id=bos_id, pad_tail=3)
                    seqs.append(sq)
                    peak_pos.append(pk)
                    row_meta.append((ni, split_idx, p["face"]))

        inp, att, _, _ = pad_right(seqs, dev)
        pos_t = (att.cumsum(-1) - 1).clamp(min=0)
        B_tot = len(seqs)
        eval_offsets = torch.arange(4, device=dev)[None, :]

        # Accumulators across fit (split=0) and test (split=1) for N=64 neurons and C=64 paired controls
        kl_abl_pk = np.zeros((N, 8), dtype=np.float64)
        kl_ctl_pk = np.zeros((N, 8), dtype=np.float64)
        kl_abl_span = np.zeros((N, 8), dtype=np.float64)
        kl_ctl_span = np.zeros((N, 8), dtype=np.float64)
        cnt_row = np.zeros(N, dtype=np.int32)

        dp_pk_neu = np.zeros((2, N, V), dtype=np.float32)
        dp_pk_ctl = np.zeros((2, N, V), dtype=np.float32)
        dp_sp_neu = np.zeros((2, N, V), dtype=np.float32)
        dp_sp_ctl = np.zeros((2, N, V), dtype=np.float32)

        eye_pk_neu = np.zeros((N, 8), dtype=np.float64)
        eye_pk_ctl = np.zeros((N, 8), dtype=np.float64)
        eye_sp_neu = np.zeros((N, 8), dtype=np.float64)
        eye_sp_ctl = np.zeros((N, 8), dtype=np.float64)

        has_bridge = l < 30
        if has_bridge:
            da_pk_neu = np.zeros((2, N, M_targets), dtype=np.float32)
            da_pk_ctl = np.zeros((2, N, M_targets), dtype=np.float32)
            da_sp_neu = np.zeros((2, N, M_targets), dtype=np.float32)
            da_sp_ctl = np.zeros((2, N, M_targets), dtype=np.float32)

        max_rec_diff = 0.0
        max_act_diff = 0.0
        for s in range(0, B_tot, int(args.batch_size)):
            e = min(B_tot, s + int(args.batch_size))
            ib, ab, pb = inp[s:e], att[s:e], pos_t[s:e]
            b = e - s
            pk_t = torch.tensor(peak_pos[s:e], dtype=torch.long, device=dev)
            seq_lens = ab.sum(dim=1)
            end_t = torch.minimum(pk_t + 3, seq_lens - 1)
            T_b = int(ib.shape[1])
            t_grid = torch.arange(T_b, device=dev)[None, :]
            span_mask = (t_grid >= pk_t[:, None]) & (t_grid <= end_t[:, None]) & (ab > 0)

            cap: Dict[str, torch.Tensor] = {}
            hks1 = [
                lm.layers[l].pre_feedforward_layernorm.register_forward_pre_hook(
                    lambda _m, a: cap.__setitem__("resid_mid", a[0].detach())
                ),
                lm.layers[l].mlp.down_proj.register_forward_pre_hook(lambda _m, a: cap.__setitem__("act_all", a[0].detach())),
                lm.layers[l].mlp.down_proj.register_forward_hook(lambda _m, _a, o: cap.__setitem__("y_mlp", o.detach())),
            ]

            def _after_l(_m: Any, _a: Any, o: Any) -> None:
                cap["layer_out"] = (o[0] if isinstance(o, tuple) else o).detach()
                raise _StopAfterLayer()

            hks1.append(lm.layers[l].register_forward_hook(_after_l))
            try:
                with torch.no_grad():
                    try:
                        lm(input_ids=ib, attention_mask=ab, use_cache=False)
                    except _StopAfterLayer:
                        pass
            finally:
                for hk in hks1:
                    hk.remove()

            resid_mid = cap["resid_mid"]
            act_all = cap["act_all"]
            y_mlp = cap["y_mlp"]
            layer_out = cap["layer_out"]

            loc_ns = [row_meta[s + i][0] for i in range(b)]
            tgt_ns = [nids[loc] for loc in loc_ns]
            t_idx = torch.tensor(tgt_ns, dtype=torch.long, device=dev)
            rows_i = torch.arange(b, device=dev)
            act_sel = act_all[rows_i, :, t_idx]  # [b, T]

            # Record activation ratio fidelity at peak
            pk_act_measured = act_sel[rows_i, pk_t].float()
            for i in range(b):
                ni = loc_ns[i]
                split_idx = row_meta[s + i][1]
                k_slot = cnt_row[ni]
                evs = picked[ni]["fit"] if split_idx == 0 else picked[ni]["test"]
                ev_act = evs[k_slot % 4]["act"]
                max_act_diff = max(max_act_diff, abs(float(pk_act_measured[i].item()) - ev_act) / max(abs(ev_act), 1e-3))

            loc_t = torch.tensor(loc_ns, dtype=torch.long, device=dev)

            # Bit-exact activation-space perturbations through the same F.linear(..., w_down_l) kernel
            flat_idx = torch.nonzero(span_mask, as_tuple=False)  # [M_tok, 2]
            rb, pt_i = flat_idx[:, 0], flat_idx[:, 1]
            loc_rb = loc_t.index_select(0, rb)  # [M_tok]
            tgt_rb = t_idx.index_select(0, rb)  # [M_tok]
            a_tok = act_all[rb, pt_i, tgt_rb].float()  # [M_tok]

            x_abl = act_all.clone()
            x_abl[rb, pt_i, tgt_rb] = 0
            y_abl = F.linear(x_abl, w_down_l)
            del x_abl

            x_ctl = act_all.clone()
            c_cols_tok = ctl_col_idx.index_select(0, loc_rb)  # [M_tok, 64]
            c_coef_tok = ctl_coeffs.index_select(0, loc_rb)  # [M_tok, 64]
            cur_vals = x_ctl[rb[:, None], pt_i[:, None], c_cols_tok].float()
            x_ctl[rb[:, None], pt_i[:, None], c_cols_tok] = (cur_vals - a_tok[:, None] * c_coef_tok).to(act_all.dtype)
            y_ctl = F.linear(x_ctl, w_down_l)
            del x_ctl

            post_rec = (resid_mid + post_ffn_ln(y_mlp)) * s_l
            max_rec_diff = max(max_rec_diff, float((layer_out - post_rec).abs().max().item()))
            if max_rec_diff >= 0.05:
                raise RuntimeError(f"L{l} layer output reconstruction mismatch max_abs={max_rec_diff:.4f}")
            post_abl = (resid_mid + post_ffn_ln(y_abl)) * s_l
            post_ctl = (resid_mid + post_ffn_ln(y_ctl)) * s_l

            pos_mat = torch.minimum(pk_t[:, None] + eval_offsets, (seq_lens - 1)[:, None])  # [b, 4]
            in_span_eval = ((pk_t[:, None] + eval_offsets) < seq_lens[:, None]).float()  # [b, 4]
            br = torch.arange(b, device=dev)[:, None].expand(b, 4)

            # Combined batch [3*b, T, D] so rec, abl, ctl share the exact same forward batch & unpacking pass
            post_3 = torch.cat([post_rec, post_abl, post_ctl], dim=0)
            ab_3 = ab.repeat(3, 1)
            pb_3 = pb.repeat(3, 1)

            da_buf = torch.empty((3 * b, 4, M_targets), dtype=torch.float32, device=dev) if has_bridge else None
            br3 = torch.arange(3 * b, device=dev)[:, None].expand(3 * b, 4)
            pos_mat3 = pos_mat.repeat(3, 1)

            hks2 = []
            if has_bridge:
                for t_l in range(30, 55):
                    if t_l not in targets_gpu:
                        continue
                    info = targets_gpu[t_l]
                    ig = info["idx"]
                    ie = info["inv_ext"]
                    os_i = info["offset"]
                    oe_i = os_i + info["count"]

                    def _make_read(ig_: torch.Tensor, ie_: torch.Tensor, os_: int, oe_: int):
                        def _hk(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> None:
                            xs = f_args[0].index_select(-1, ig_)
                            da_buf[:, :, os_:oe_] = xs[br3, pos_mat3, :].float() * ie_[None, None, :]

                        return _hk

                    hks2.append(lm.layers[t_l].mlp.down_proj.register_forward_pre_hook(_make_read(ig, ie, os_i, oe_i)))

            try:
                with torch.no_grad():
                    h_final = ao.forward_from_layer(lm, l + 1, post_3, ab_3, pb_3)
            finally:
                for hk in hks2:
                    hk.remove()

            h_sel = h_final.view(3, b, T_b, D)[:, br, pos_mat, :].reshape(3 * b * 4, D)
            lsm = F.log_softmax(logits_fp32(h_sel, W_U), dim=-1).view(3, b, 4, V)

            if li == 0 and s == 0:
                sanity_report = _run_batch_0_sanity(
                    lm, W_U, l, ib, ab, pb, pos_mat, in_span_eval, span_mask, y_mlp, y_abl, y_ctl, w_post_l, rb, pt_i, post_rec, post_abl, tgt_ns, lsm[0], lsm[1]
                )
                sanity_report["layer"] = int(l)
                print(f"Batch-0 sanity at L{l}: {sanity_report}", flush=True)
                if sanity_report["noop_kl_same_batch"] >= 1e-5 or sanity_report["max_kl_alg_vs_hook_same_batch"] >= 1e-5:
                    raise RuntimeError(f"Sanity check failed at L{l}: {sanity_report}")

            del y_mlp, y_abl, y_ctl, rb, pt_i, flat_idx, loc_rb, tgt_rb, a_tok, c_cols_tok, c_coef_tok, cur_vals

            p_all = lsm.exp()
            p_rec, p_abl, p_ctl = p_all[0], p_all[1], p_all[2]
            kl_a = (p_rec * (lsm[0] - lsm[1])).sum(dim=-1)  # [b, 4]
            kl_c = (p_rec * (lsm[0] - lsm[2])).sum(dim=-1)
            span_den = in_span_eval.sum(dim=-1).clamp(min=1.0)
            kl_a_sp = ((kl_a * in_span_eval).sum(dim=-1) / span_den).cpu().numpy()
            kl_c_sp = ((kl_c * in_span_eval).sum(dim=-1) / span_den).cpu().numpy()
            kl_a_pk = kl_a[:, 0].cpu().numpy()
            kl_c_pk = kl_c[:, 0].cpu().numpy()

            dp_a = p_rec - p_abl  # [b, 4, V]
            dp_c = p_rec - p_ctl
            dp_a_pk_cpu = dp_a[:, 0, :].cpu().numpy()
            dp_c_pk_cpu = dp_c[:, 0, :].cpu().numpy()
            span_w = (in_span_eval / span_den[:, None])[:, :, None]
            dp_a_sp_cpu = (dp_a * span_w).sum(dim=1).cpu().numpy()
            dp_c_sp_cpu = (dp_c * span_w).sum(dim=1).cpu().numpy()

            if has_bridge and da_buf is not None:
                da_v = da_buf.view(3, b, 4, M_targets)
                da_a_pk_cpu = (da_v[0, :, 0, :] - da_v[1, :, 0, :]).cpu().numpy()
                da_c_pk_cpu = (da_v[0, :, 0, :] - da_v[2, :, 0, :]).cpu().numpy()
                da_a_sp_cpu = ((da_v[0] - da_v[1]) * span_w).sum(dim=1).cpu().numpy()
                da_c_sp_cpu = ((da_v[0] - da_v[2]) * span_w).sum(dim=1).cpu().numpy()

            for i in range(b):
                ni, split_idx, _ = row_meta[s + i]
                slot_k = int(cnt_row[ni])
                cnt_row[ni] += 1
                kl_abl_pk[ni, slot_k] = float(kl_a_pk[i])
                kl_ctl_pk[ni, slot_k] = float(kl_c_pk[i])
                kl_abl_span[ni, slot_k] = float(kl_a_sp[i])
                kl_ctl_span[ni, slot_k] = float(kl_c_sp[i])

                dp_pk_neu[split_idx, ni] += dp_a_pk_cpu[i] * 0.25
                dp_pk_ctl[split_idx, ni] += dp_c_pk_cpu[i] * 0.25
                dp_sp_neu[split_idx, ni] += dp_a_sp_cpu[i] * 0.25
                dp_sp_ctl[split_idx, ni] += dp_c_sp_cpu[i] * 0.25

                eye_pk_neu[ni, slot_k] = float(np.sum(dp_a_pk_cpu[i, eye_neu[ni]]))
                eye_pk_ctl[ni, slot_k] = float(np.sum(dp_c_pk_cpu[i, eye_ctl[ni]]))
                eye_sp_neu[ni, slot_k] = float(np.sum(dp_a_sp_cpu[i, eye_neu[ni]]))
                eye_sp_ctl[ni, slot_k] = float(np.sum(dp_c_sp_cpu[i, eye_ctl[ni]]))

                if has_bridge:
                    da_pk_neu[split_idx, ni] += da_a_pk_cpu[i] * 0.25
                    da_pk_ctl[split_idx, ni] += da_c_pk_cpu[i] * 0.25
                    da_sp_neu[split_idx, ni] += da_a_sp_cpu[i] * 0.25
                    da_sp_ctl[split_idx, ni] += da_c_sp_cpu[i] * 0.25

            del cap, resid_mid, act_all, layer_out, post_rec, post_abl, post_ctl, post_3, h_final, h_sel, lsm, p_all
            if has_bridge:
                del da_buf
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # M1: Norm-matched KL necessity
        m1_pk = summarize_stats(emp_p_values(kl_abl_pk.mean(axis=1), kl_ctl_pk.mean(axis=1)))
        m1_sp = summarize_stats(emp_p_values(kl_abl_span.mean(axis=1), kl_ctl_span.mean(axis=1)))

        # M2: Probability-weighted Delta p directional carry (fit <-> test)
        m2_pk_res = score_directional_carry(
            dp_pk_neu[0], dp_pk_ctl[0], dp_pk_neu[1], dp_pk_ctl[1], top_k=5, exclude_mask=bad_tok_mask
        )
        m2_sp_res = score_directional_carry(
            dp_sp_neu[0], dp_sp_ctl[0], dp_sp_neu[1], dp_sp_ctl[1], top_k=5, exclude_mask=bad_tok_mask
        )

        # Top-5 replicated Delta p examples (never storing passage text)
        dp_top_ex: List[Dict[str, Any]] = []
        order_dp = np.argsort(-m2_sp_res["mean_neu"])[:5]
        for idx_u in order_dp:
            idx_i = int(idx_u)
            toks_ft = m2_sp_res["picks_ft_neu"][idx_i]
            dp_top_ex.append(
                {
                    "neuron": int(nids[idx_i]),
                    "face": faces[idx_i],
                    "mean_carry_sp": round(float(m2_sp_res["mean_neu"][idx_i]), 5),
                    "kl_sp_neu": round(float(kl_abl_span[idx_i].mean()), 5),
                    "kl_sp_ctl": round(float(kl_ctl_span[idx_i].mean()), 5),
                    "picked_fit_words": [
                        {
                            "tok": vocab[t],
                            "fit_dp": round(float(m2_sp_res["fit_centered_neu"][idx_i, t]), 5),
                            "test_dp": round(float(m2_sp_res["test_centered_neu"][idx_i, t]), 5),
                        }
                        for t in toks_ft
                    ],
                }
            )

        # M3: Pre-fixed weight-lens eye_10 Delta p drop
        m3_pk = summarize_stats(emp_p_values(eye_pk_neu.mean(axis=1), eye_pk_ctl.mean(axis=1)))
        m3_sp = summarize_stats(emp_p_values(eye_sp_neu.mean(axis=1), eye_sp_ctl.mean(axis=1)))
        m3_pk_test = summarize_stats(emp_p_values(eye_pk_neu[:, 4:].mean(axis=1), eye_pk_ctl[:, 4:].mean(axis=1)))

        # M4: On-manifold L30-L54 bridge carry (L < 30 only)
        m4_summary: Any = None
        bridge_top_ex: List[Dict[str, Any]] = []
        if has_bridge:
            m4_pos = score_directional_carry(
                da_sp_neu[0], da_sp_ctl[0], da_sp_neu[1], da_sp_ctl[1], top_k=5, allow_negative=False
            )
            m4_signed = score_directional_carry(
                da_sp_neu[0], da_sp_ctl[0], da_sp_neu[1], da_sp_ctl[1], top_k=5, allow_negative=True
            )
            m4_pk_signed = score_directional_carry(
                da_pk_neu[0], da_pk_ctl[0], da_pk_neu[1], da_pk_ctl[1], top_k=5, allow_negative=True
            )
            m4_summary = {
                "span_pos_mean": summarize_stats(m4_pos["mean"]),
                "span_signed_ft": summarize_stats(m4_signed["ft"]),
                "span_signed_tf": summarize_stats(m4_signed["tf"]),
                "span_signed_mean": summarize_stats(m4_signed["mean"]),
                "pk_signed_mean": summarize_stats(m4_pk_signed["mean"]),
            }
            for idx_u in np.argsort(-m4_signed["mean_neu"])[:5]:
                idx_i = int(idx_u)
                picks = m4_signed["picks_ft_neu"][idx_i]
                bridge_top_ex.append(
                    {
                        "src_neuron": int(nids[idx_i]),
                        "src_face": faces[idx_i],
                        "mean_signed_carry": round(float(m4_signed["mean_neu"][idx_i]), 4),
                        "targets": [
                            {
                                "layer": int(target_meta[ti][0]),
                                "neuron": int(target_meta[ti][1]),
                                "words": target_meta[ti][2],
                                "fit_da": round(float(m4_signed["fit_centered_neu"][idx_i, ti]), 4),
                                "test_da": round(float(m4_signed["test_centered_neu"][idx_i, ti]), 4),
                            }
                            for ti in picks
                        ],
                    }
                )
            del da_pk_neu, da_pk_ctl, da_sp_neu, da_sp_ctl

        row_l = {
            "layer": int(l),
            "pure44_eligible_neurons": n_elig,
            "n_neurons": N,
            "elapsed_s": round(time.time() - tl0, 2),
            "max_layer_rec_diff": round(max_rec_diff, 6),
            "max_rel_act_diff": round(max_act_diff, 4),
            "m1_kl_pk": m1_pk,
            "m1_kl_span": m1_sp,
            "m2_dp_pk": {
                "ft": summarize_stats(m2_pk_res["ft"]),
                "tf": summarize_stats(m2_pk_res["tf"]),
                "mean": summarize_stats(m2_pk_res["mean"]),
            },
            "m2_dp_span": {
                "ft": summarize_stats(m2_sp_res["ft"]),
                "tf": summarize_stats(m2_sp_res["tf"]),
                "mean": summarize_stats(m2_sp_res["mean"]),
            },
            "m3_eye10": {"pk_all8": m3_pk, "span_all8": m3_sp, "pk_test4": m3_pk_test},
            "m4_bridge_l30_54": m4_summary,
            "examples_dp": dp_top_ex,
            "examples_bridge": bridge_top_ex,
        }
        layer_results.append(row_l)
        print(
            f"L{l:2d} ({row_l['elapsed_s']:.1f}s): M1 KL_sp AUC={m1_sp['auc']} p05={m1_sp['n_emp_p05']}/{N}; "
            f"M2 dp_sp mean AUC={row_l['m2_dp_span']['mean']['auc']} p05={row_l['m2_dp_span']['mean']['n_emp_p05']}/{N}; "
            f"M3 eye10_pk AUC={m3_pk['auc']} p05={m3_pk['n_emp_p05']}/{N}"
            + (f"; M4 bridge AUC={m4_summary['span_signed_mean']['auc']} p05={m4_summary['span_signed_mean']['n_emp_p05']}/{N}" if has_bridge else ""),
            flush=True,
        )

        del dp_pk_neu, dp_pk_ctl, dp_sp_neu, dp_sp_ctl, eye_neu, eye_ctl, w_down_l, ctl_col_idx, ctl_coeffs, w_ctl_dirs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t1 = time.time()
    prov = gp.build_provenance(
        REPO_ROOT,
        [__file__, REPO_ROOT / "scripts/in_use_common.py", REPO_ROOT / "scripts/downstream_pilot.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(t1),
        model_id=rt.model_id,
        weights=weights_card(rt),
        neuron_ids=neuron_ids_meta,
    )
    out = {
        **prov,
        "elapsed_s": round(t1 - t0, 2),
        "config": {"layers": layers, "n_neurons": int(args.n_neurons), "batch_size": int(args.batch_size), "split": "pure_doc_half_4fit_4test"},
        "batch0_sanity": sanity_report,
        "layers": layer_results,
    }
    out_path = out_dir / "probe1_in_use_causal.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} in {t1 - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
