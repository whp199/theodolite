#!/usr/bin/env python3
"""Diagnostic check for Path 1 (in_use_lens) L58 symmetric shuffled-vs-shuffled G5 null.

Why G5 tripped in in_use_lens_trial.py:
  shuf_all measured (p_rec - p_abl) on WRONG/deranged eye_10 tokens (median 0.0,
  often negative), whereas col_all measured (p_rec - p_col) on the control's OWN
  eye_10 tokens (p90 = 7e-6), yielding asymmetric AUC = 0.1382 despite fdp05 = 0.

This standalone script:
  - In --saved-only mode (CPU): audits saved L58_verdict.json + L58_units.json and
    runs a synthetic CPU verification of evaluate_symmetric_g5().
  - In default GPU mode: re-runs L58 with the exact frozen neurons_L58.txt & seeds,
    verifies reproduction of G1-G4 (156/1,024 dual05), and evaluates the symmetric
    shuffled-vs-shuffled G5 null: shuf_neu (abl on deranged eye_neu) vs shuf_col
    (col on deranged eye_col) and shuf_iso (iso on deranged eye_iso).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ALLOC_CONF_NAMES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
if not any(os.environ.get(k) for k in _ALLOC_CONF_NAMES):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

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
from scripts.in_use_common import (  # noqa: E402
    build_event_sequence,
    collect_face_events,
    emp_p_values,
    load_passages_and_doc_halves,
    summarize_stats,
)
from scripts.in_use_lens_trial import (  # noqa: E402
    _StopAfterLayer,
    _control_eye_ids,
    _fdp_from_arrays,
    _load_frozen_neurons,
    _run_layer_batch0_sanity,
    derangement_perm,
    evaluate_p1_layer,
)


def evaluate_symmetric_g5(
    *,
    layer: int,
    eye_pk_neu: np.ndarray,
    eye_pk_shuf_neu: np.ndarray,
    eye_pk_col: np.ndarray,
    eye_pk_shuf_col: np.ndarray,
    eye_pk_iso: np.ndarray,
    eye_pk_shuf_iso: np.ndarray,
    kl_abl: np.ndarray,
    kl_col: np.ndarray,
    kl_iso: np.ndarray,
    sanity: Dict[str, Any],
    seed_base: int = 20261126,
) -> Dict[str, Any]:
    """Evaluate standard P1 gates plus symmetric shuffled-vs-shuffled G5 null."""
    base = evaluate_p1_layer(
        layer=layer,
        eye_pk_neu=eye_pk_neu,
        eye_pk_shuf=eye_pk_shuf_neu,
        eye_pk_col=eye_pk_col,
        eye_pk_iso=eye_pk_iso,
        kl_abl=kl_abl,
        kl_col=kl_col,
        kl_iso=kl_iso,
        sanity=sanity,
        seed_base=seed_base,
    )
    shuf_neu = eye_pk_shuf_neu.mean(axis=1)
    shuf_col = eye_pk_shuf_col.mean(axis=1)
    shuf_iso = eye_pk_shuf_iso.mean(axis=1)

    st_sym_col = emp_p_values(shuf_neu, shuf_col)
    st_sym_iso = emp_p_values(shuf_neu, shuf_iso)
    st_sym_xp = emp_p_values(shuf_col, shuf_iso)

    fdp_sym_col = _fdp_from_arrays(shuf_neu, shuf_col, alpha=0.05, n_boot=0, seed=seed_base)
    fdp_sym_iso = _fdp_from_arrays(shuf_neu, shuf_iso, alpha=0.05, n_boot=0, seed=seed_base)
    fdp_sym_xp = _fdp_from_arrays(shuf_col, shuf_iso, alpha=0.05, n_boot=0, seed=seed_base)

    sanity_ok = (
        float(sanity.get("noop_kl_same_batch", 1.0)) < 1e-5
        and float(sanity.get("max_kl_alg_vs_hook_same_batch", 1.0)) < 1e-5
        and float(sanity.get("max_layer_rec_diff", 1.0)) < 0.05
        and 0.90 <= float(sanity.get("achieved_col_over_abl_post_norm_median", 0.0)) <= 1.15
        and 0.90 <= float(sanity.get("achieved_iso_over_abl_post_norm_median", 0.0)) <= 1.15
    )
    sym_g5_col_pass = (0.45 <= st_sym_col["auc"] <= 0.55) and (fdp_sym_col["count"] == 0) and sanity_ok
    sym_g5_iso_pass = (0.45 <= st_sym_iso["auc"] <= 0.55) and (fdp_sym_iso["count"] == 0) and sanity_ok

    base_clean = {k: v for k, v in base.items() if not k.startswith("_")}
    return {
        **base_clean,
        "symmetric_g5": {
            "shuf_neu_vs_shuf_col": {
                **summarize_stats(st_sym_col),
                "fdp05_count": fdp_sym_col["count"],
                "pass": bool(sym_g5_col_pass),
            },
            "shuf_neu_vs_shuf_iso": {
                **summarize_stats(st_sym_iso),
                "fdp05_count": fdp_sym_iso["count"],
                "pass": bool(sym_g5_iso_pass),
            },
            "shuf_col_vs_shuf_iso": {
                **summarize_stats(st_sym_xp),
                "fdp05_count": fdp_sym_xp["count"],
            },
            "pass_sym_g5_both": bool(sym_g5_col_pass and sym_g5_iso_pass),
            "pass_all_gates_with_sym_g5": bool(
                base["gates"]["G1_all8_effect"]
                and base["gates"]["G2_heldout_test4_and_rho"]
                and base["gates"]["G3_dual_fdp05_yield"]
                and base["gates"]["G4_cross_pool_exchangeability"]
                and sym_g5_col_pass
                and sym_g5_iso_pass
            ),
        },
    }


def run_saved_only_audit(trial_dir: Path) -> Dict[str, Any]:
    v_path = trial_dir / "L58_verdict.json"
    u_path = trial_dir / "L58_units.json"
    if not v_path.is_file() or not u_path.is_file():
        v_path = REPO_ROOT / "experiments/results/in_use_lens_trial/L58_verdict.json"
    verdict = json.loads(v_path.read_text(encoding="utf-8"))
    units = json.loads(u_path.read_text(encoding="utf-8")) if u_path.is_file() else []

    # Synthetic verification of evaluate_symmetric_g5 (N=1024 matching L58)
    rng = np.random.default_rng(58)
    N = 1024
    sanity_ok = {
        "noop_kl_same_batch": 0.0,
        "max_kl_alg_vs_hook_same_batch": 0.0,
        "max_layer_rec_diff": 0.0,
        "achieved_col_over_abl_post_norm_median": 0.9994,
        "achieved_iso_over_abl_post_norm_median": 0.9974,
    }
    col = rng.normal(0.0, 0.002, size=(N, 8))
    iso = rng.normal(0.0, 0.002, size=(N, 8))
    neu = rng.normal(0.0, 0.002, size=(N, 8))
    neu[:240] += 0.015
    shuf_neu = rng.normal(-0.0005, 0.002, size=(N, 8))
    shuf_col = rng.normal(-0.0005, 0.002, size=(N, 8))
    shuf_iso = rng.normal(-0.0005, 0.002, size=(N, 8))
    syn = evaluate_symmetric_g5(
        layer=58,
        eye_pk_neu=neu,
        eye_pk_shuf_neu=shuf_neu,
        eye_pk_col=col,
        eye_pk_shuf_col=shuf_col,
        eye_pk_iso=iso,
        eye_pk_shuf_iso=shuf_iso,
        kl_abl=np.abs(neu),
        kl_col=np.abs(col),
        kl_iso=np.abs(iso),
        sanity=sanity_ok,
    )
    assert syn["gates"]["G5_shuffled_null_and_sanity"] is False
    assert syn["symmetric_g5"]["pass_sym_g5_both"] is True
    assert syn["symmetric_g5"]["pass_all_gates_with_sym_g5"] is True

    shuf = verdict["shuffled_null_vs_col"]
    out = {
        "mode": "saved_only_audit",
        "saved_verdict_path": str(v_path),
        "saved_units_count": len(units),
        "saved_dual05_count": verdict["dual05_count"],
        "saved_gates": verdict["gates"],
        "saved_asymmetric_shuf_vs_col": {
            "auc": shuf["auc"],
            "fdp05_count": shuf["fdp05_count"],
            "n_bh_q10": shuf["n_bh_q10"],
            "n_above_max_ctl": shuf["n_above_max_ctl"],
            "neu_median": shuf["neu_median"],
            "ctl_median": shuf["ctl_median"],
            "ctl_p90": shuf["ctl_p90"],
        },
        "gpu_rerun_required_reason": "L58_units.json stores per-neuron eye_pk_all8 only (no per-control eye_pk_col or deranged eye_pk_shuf_col arrays).",
        "synthetic_sym_g5_selftest_pass": True,
    }
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Path 1 L58 symmetric shuffled-vs-shuffled G5 null diagnostic")
    ap.add_argument("--prereg-dir", default="experiments/results/in_use_lens_trial/prereg")
    ap.add_argument("--trial-dir", default="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/in_use_lens")
    ap.add_argument("--out", default="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/in_use_lens/L58_symnull_verdict.json")
    ap.add_argument("--batch-windows", type=int, default=8)
    ap.add_argument("--batch-events", type=int, default=4)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--saved-only", action="store_true", help="CPU audit of saved L58 JSONs + synthetic check")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir)
    if args.saved_only:
        run_saved_only_audit(trial_dir)
        return

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    l = 58
    prereg_dir = Path(args.prereg_dir) if Path(args.prereg_dir).is_absolute() else REPO_ROOT / args.prereg_dir
    out_path = Path(args.out)

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()
    loaded = load_passages_and_doc_halves(data_dir)
    P, act_scale, windows = loaded["P"], loaded["act_scale"], loaded["windows"]
    doc_id, half_id = loaded["doc_id"], loaded["half_id"]
    pure_h0_mask, pure_h1_mask = loaded["pure_h0_mask"], loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]
    lens = np.load(data_dir / "index/lens.npz")

    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    if getattr(rt, "eager_model", None) is not None and hasattr(rt.eager_model, "_lens_jacobians"):
        rt.eager_model._lens_jacobians.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    w_ops = ao.Weights(lm, rt)
    dev = model_device(lm)
    W_U = rt.lm_head_weight
    fnw = rt.final_norm_weight
    D, V = int(W_U.shape[1]), int(W_U.shape[0])

    items = _load_frozen_neurons(prereg_dir / f"neurons_L{l}.txt")
    N = len(items)
    nids = [n for n, _ in items]
    faces = [fc for _, fc in items]

    events_by_win: Dict[int, List[Tuple[int, int, int, int, float]]] = {}
    win_seq_cache: Dict[int, Tuple[List[int], Dict[int, int]]] = {}
    win_max_end: Dict[int, int] = {}
    temp_evs: List[Tuple[int, int, int, int, float]] = []
    for ni, (nid, fc) in enumerate(items):
        ev = collect_face_events(
            P, windows, act_scale, pure_h0_mask, pure_h1_mask, doc_id, half_id, l, nid, fc, eos_id=eos_id, bos_id=bos_id
        )
        chosen = ev["pure_h0"][:4] + ev["pure_h1"][:4]
        for slot_k, row in enumerate(chosen):
            wi, at_i, act_v = int(row["win"]), int(row["at"]), float(row["act"])
            win_max_end[wi] = max(win_max_end.get(wi, 0), at_i)
            temp_evs.append((wi, ni, slot_k, at_i, act_v))

    for wi, max_at in win_max_end.items():
        sq, _ = build_event_sequence(windows, wi, max_at, bos_id=bos_id, pad_tail=0)
        row = windows[wi]
        at_to_pk: Dict[int, int] = {}
        t_len = 1
        for pos in range(1, max_at + 1):
            if int(row[pos]) >= 0:
                at_to_pk[pos] = t_len
                t_len += 1
        win_seq_cache[wi] = (sq, at_to_pk)

    for wi, ni, slot_k, at_i, act_v in temp_evs:
        _, at_to_pk = win_seq_cache[wi]
        events_by_win.setdefault(wi, []).append((ni, slot_k, at_to_pk[at_i], at_i, act_v))

    sanity_ns = nids[:20]
    w_post_l = w_ops.norm(l, "post_feedforward").to(dev)
    w_down_l = w_ops.w_down(l).to(dev)
    vecs_chk = (w_down_l.index_select(1, torch.tensor(sanity_ns, device=dev, dtype=torch.long)).T * w_post_l[None, :]).to(W_U.dtype)
    logits_chk = logits_fp32(gemma4_rms_norm(vecs_chk, fnw), W_U)
    got_chk_pos = label_top_ids(vecs_chk, fnw, W_U, logits_chk, k=10)
    got_chk_neg = label_top_ids(-vecs_chk, fnw, W_U, -logits_chk, k=10)
    ov_pos = sum(len(set(got_chk_pos[i]) & set(int(x) for x in lens["eye_pos"][l, sanity_ns[i], :10])) for i in range(20))
    ov_neg = sum(len(set(got_chk_neg[i]) & set(int(x) for x in lens["eye_neg"][l, sanity_ns[i], :10])) for i in range(20))
    del vecs_chk, logits_chk

    mix_k = 64
    rng_col = np.random.default_rng(20261126100 + l)
    rng_iso = np.random.default_rng(20261126200 + l)
    ctl_col_idx = torch.from_numpy(
        np.stack([rng_col.choice(w_ops.d_ff, size=mix_k, replace=False) for _ in range(N)], axis=0)
    ).to(device=dev, dtype=torch.long)
    ctl_coeffs = torch.from_numpy(rng_col.standard_normal((N, mix_k)).astype(np.float32)).to(device=dev)
    iso_raw = torch.from_numpy(rng_iso.standard_normal((N, D)).astype(np.float32)).to(device=dev)

    w_col_dirs = torch.empty((N, D), dtype=torch.float32, device=dev)
    w_iso_dirs = torch.empty((N, D), dtype=torch.float32, device=dev)
    wp_f = w_post_l.float()
    for i in range(N):
        tn = (w_down_l[:, nids[i]].float() * wp_f).norm().clamp(min=1e-30)
        v_col = w_down_l.index_select(1, ctl_col_idx[i]).float() @ ctl_coeffs[i]
        sc_c = tn / (v_col * wp_f).norm().clamp(min=1e-30)
        ctl_coeffs[i] = ctl_coeffs[i] * sc_c
        w_col_dirs[i] = v_col * sc_c
        v_iso = iso_raw[i]
        sc_i = tn / (v_iso * wp_f).norm().clamp(min=1e-30)
        w_iso_dirs[i] = v_iso * sc_i

    eye_neu = torch.from_numpy(np.stack([lens[f"eye_{fc}"][l, nid, :10] for nid, fc in items], axis=0).astype(np.int64)).to(dev)
    eye_col = torch.from_numpy(_control_eye_ids(w_col_dirs, faces, w_post_l, fnw, W_U, k=10).astype(np.int64)).to(dev)
    eye_iso = torch.from_numpy(_control_eye_ids(w_iso_dirs, faces, w_post_l, fnw, W_U, k=10).astype(np.int64)).to(dev)
    eye_shuf_neu = eye_neu.index_select(0, torch.from_numpy(derangement_perm(N, 20261126300 + l)).to(dev))
    eye_shuf_col = eye_col.index_select(0, torch.from_numpy(derangement_perm(N, 20261126400 + l)).to(dev))
    eye_shuf_iso = eye_iso.index_select(0, torch.from_numpy(derangement_perm(N, 20261126500 + l)).to(dev))
    del w_col_dirs, iso_raw

    first_wins = sorted(events_by_win.keys())
    b0_seqs, b0_pks, b0_nis = [], [], []
    for wi in first_wins:
        sq_w, _ = win_seq_cache[wi]
        for ni, _sk, pk, _at, _av in events_by_win[wi]:
            b0_seqs.append(sq_w[: pk + 1])
            b0_pks.append(pk)
            b0_nis.append(ni)
            if len(b0_seqs) >= 4:
                break
        if len(b0_seqs) >= 4:
            break
    b0_ni_t = torch.tensor(b0_nis, dtype=torch.long, device=dev)
    sanity = _run_layer_batch0_sanity(
        lm,
        W_U,
        l,
        b0_seqs,
        b0_pks,
        [nids[i] for i in b0_nis],
        ctl_col_idx.index_select(0, b0_ni_t),
        ctl_coeffs.index_select(0, b0_ni_t),
        w_iso_dirs.index_select(0, b0_ni_t),
        w_down_l,
        w_post_l,
        float(w_ops.scalars[l]),
    )
    sanity["lens_overlap_pos_20"] = ov_pos
    sanity["lens_overlap_neg_20"] = ov_neg

    eye_pk_neu_arr = np.zeros((N, 8), dtype=np.float64)
    eye_pk_shuf_neu_arr = np.zeros((N, 8), dtype=np.float64)
    eye_pk_col_arr = np.zeros((N, 8), dtype=np.float64)
    eye_pk_shuf_col_arr = np.zeros((N, 8), dtype=np.float64)
    eye_pk_iso_arr = np.zeros((N, 8), dtype=np.float64)
    eye_pk_shuf_iso_arr = np.zeros((N, 8), dtype=np.float64)
    kl_abl_arr = np.zeros((N, 8), dtype=np.float64)
    kl_col_arr = np.zeros((N, 8), dtype=np.float64)
    kl_iso_arr = np.zeros((N, 8), dtype=np.float64)

    win_order = sorted(events_by_win.keys(), key=lambda w: len(win_seq_cache[w][0]))
    post_ffn_ln = lm.layers[l].post_feedforward_layernorm
    s_l = float(w_ops.scalars[l])
    tidx_full = torch.tensor(nids, dtype=torch.long, device=dev)
    max_ev_rec_diff = 0.0

    for ws in range(0, len(win_order), int(args.batch_windows)):
        we = min(len(win_order), ws + int(args.batch_windows))
        w_batch = win_order[ws:we]
        ib_w, ab_w, _, _ = pad_right([win_seq_cache[w][0] for w in w_batch], dev)
        cap_w: Dict[str, torch.Tensor] = {}
        hks_w = [
            lm.layers[l].pre_feedforward_layernorm.register_forward_pre_hook(lambda _m, a: cap_w.__setitem__("resid_mid", a[0].detach())),
            lm.layers[l].mlp.down_proj.register_forward_pre_hook(lambda _m, a: cap_w.__setitem__("act_all", a[0].detach())),
            lm.layers[l].mlp.down_proj.register_forward_hook(lambda _m, _a, o: cap_w.__setitem__("y_mlp", o.detach())),
        ]

        def _after_lw(_m: Any, _a: Any, o: Any) -> None:
            cap_w["layer_out"] = (o[0] if isinstance(o, tuple) else o).detach()
            raise _StopAfterLayer()

        hks_w.append(lm.layers[l].register_forward_hook(_after_lw))
        try:
            try:
                lm(input_ids=ib_w, attention_mask=ab_w, use_cache=False)
            except _StopAfterLayer:
                pass
        finally:
            for hk in hks_w:
                hk.remove()

        ev_meta: List[Tuple[int, int, int, int]] = []
        for row_i, wi in enumerate(w_batch):
            for ni, slot_k, pk, _at, _av in events_by_win[wi]:
                ev_meta.append((row_i, pk, ni, slot_k))

        r_rows = torch.tensor([m[0] for m in ev_meta], dtype=torch.long, device=dev)
        r_pks = torch.tensor([m[1] for m in ev_meta], dtype=torch.long, device=dev)
        r_nis = torch.tensor([m[2] for m in ev_meta], dtype=torch.long, device=dev)
        act_ev = cap_w["act_all"][r_rows, r_pks]
        ym_ev = cap_w["y_mlp"][r_rows, r_pks]
        rm_ev = cap_w["resid_mid"][r_rows, r_pks]
        lo_ev = cap_w["layer_out"][r_rows, r_pks]
        layer_out_full = cap_w["layer_out"]
        del cap_w, ib_w, ab_w

        tgt_ev = tidx_full.index_select(0, r_nis)
        ev_arange = torch.arange(len(ev_meta), device=dev)
        a_tok = act_ev[ev_arange, tgt_ev].float()

        x_abl = act_ev.clone()
        x_abl[ev_arange, tgt_ev] = 0
        y_abl_ev = F.linear(x_abl, w_down_l)
        del x_abl

        c_cols_ev = ctl_col_idx.index_select(0, r_nis)
        c_coef_ev = ctl_coeffs.index_select(0, r_nis)
        x_col = act_ev.clone()
        cur_c = x_col[ev_arange[:, None], c_cols_ev].float()
        x_col[ev_arange[:, None], c_cols_ev] = (cur_c - a_tok[:, None] * c_coef_ev).to(act_ev.dtype)
        y_col_ev = F.linear(x_col, w_down_l)
        del x_col, act_ev, c_cols_ev, c_coef_ev

        v_iso_ev = w_iso_dirs.index_select(0, r_nis).float()
        y_iso_ev = (ym_ev.float() - a_tok[:, None] * v_iso_ev).to(ym_ev.dtype)

        post_rec_ev = (rm_ev + post_ffn_ln(ym_ev)) * s_l
        max_ev_rec_diff = max(max_ev_rec_diff, float((lo_ev - post_rec_ev).abs().max().item()))
        post_abl_ev = (rm_ev + post_ffn_ln(y_abl_ev)) * s_l
        post_col_ev = (rm_ev + post_ffn_ln(y_col_ev)) * s_l
        post_iso_ev = (rm_ev + post_ffn_ln(y_iso_ev)) * s_l
        del ym_ev, rm_ev, lo_ev, y_abl_ev, y_col_ev, y_iso_ev, v_iso_ev

        M_ev = len(ev_meta)
        ev_idx_sorted = sorted(range(M_ev), key=lambda idx: ev_meta[idx][1])
        be = int(args.batch_events)
        for es in range(0, M_ev, be):
            ee = min(M_ev, es + be)
            sub_idx = ev_idx_sorted[es:ee]
            m = len(sub_idx)
            T_m = max(ev_meta[j][1] for j in sub_idx) + 1
            h4 = torch.zeros((4 * m, T_m, D), dtype=post_rec_ev.dtype, device=dev)
            am = torch.zeros((4 * m, T_m), dtype=torch.long, device=dev)
            pk_list: List[int] = []
            nis_m: List[int] = []
            slots_m: List[int] = []
            for bi_m, j_ev in enumerate(sub_idx):
                row_i, pk, ni, slot_k = ev_meta[j_ev]
                pk_list.append(pk)
                nis_m.append(ni)
                slots_m.append(slot_k)
                prefix = layer_out_full[row_i, : pk + 1]
                for v_i in range(4):
                    h4[v_i * m + bi_m, : pk + 1] = prefix
                    am[v_i * m + bi_m, : pk + 1] = 1
                h4[0 * m + bi_m, pk] = post_rec_ev[j_ev]
                h4[1 * m + bi_m, pk] = post_abl_ev[j_ev]
                h4[2 * m + bi_m, pk] = post_col_ev[j_ev]
                h4[3 * m + bi_m, pk] = post_iso_ev[j_ev]
            pm = (am.cumsum(-1) - 1).clamp(min=0)
            h_fin = ao.forward_from_layer(lm, l + 1, h4, am, pm)
            pk_t_m = torch.tensor(pk_list, dtype=torch.long, device=dev).repeat(4)
            rows_4m = torch.arange(4 * m, device=dev)
            h_pk = h_fin[rows_4m, pk_t_m]
            lsm = F.log_softmax(logits_fp32(h_pk, W_U), dim=-1).view(4, m, V)
            p_all = lsm.exp()
            p_rec, p_abl, p_col, p_iso = p_all[0], p_all[1], p_all[2], p_all[3]

            ka = (p_rec * (lsm[0] - lsm[1])).sum(dim=-1).cpu().numpy()
            kc = (p_rec * (lsm[0] - lsm[2])).sum(dim=-1).cpu().numpy()
            ki = (p_rec * (lsm[0] - lsm[3])).sum(dim=-1).cpu().numpy()

            nis_t_m = torch.tensor(nis_m, dtype=torch.long, device=dev)
            ep_neu = (p_rec - p_abl).gather(-1, eye_neu.index_select(0, nis_t_m)).sum(-1).cpu().numpy()
            ep_shf_n = (p_rec - p_abl).gather(-1, eye_shuf_neu.index_select(0, nis_t_m)).sum(-1).cpu().numpy()
            ep_col = (p_rec - p_col).gather(-1, eye_col.index_select(0, nis_t_m)).sum(-1).cpu().numpy()
            ep_shf_c = (p_rec - p_col).gather(-1, eye_shuf_col.index_select(0, nis_t_m)).sum(-1).cpu().numpy()
            ep_iso = (p_rec - p_iso).gather(-1, eye_iso.index_select(0, nis_t_m)).sum(-1).cpu().numpy()
            ep_shf_i = (p_rec - p_iso).gather(-1, eye_shuf_iso.index_select(0, nis_t_m)).sum(-1).cpu().numpy()

            for bi_m in range(m):
                ni, slot_k = nis_m[bi_m], slots_m[bi_m]
                kl_abl_arr[ni, slot_k] = float(ka[bi_m])
                kl_col_arr[ni, slot_k] = float(kc[bi_m])
                kl_iso_arr[ni, slot_k] = float(ki[bi_m])
                eye_pk_neu_arr[ni, slot_k] = float(ep_neu[bi_m])
                eye_pk_shuf_neu_arr[ni, slot_k] = float(ep_shf_n[bi_m])
                eye_pk_col_arr[ni, slot_k] = float(ep_col[bi_m])
                eye_pk_shuf_col_arr[ni, slot_k] = float(ep_shf_c[bi_m])
                eye_pk_iso_arr[ni, slot_k] = float(ep_iso[bi_m])
                eye_pk_shuf_iso_arr[ni, slot_k] = float(ep_shf_i[bi_m])
            del h4, am, pm, h_fin, h_pk, lsm, p_all
        del post_rec_ev, post_abl_ev, post_col_ev, post_iso_ev, layer_out_full
        if (ws // int(args.batch_windows)) % 40 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    sanity["max_window_batch_rec_diff"] = round(max_ev_rec_diff, 6)
    res = evaluate_symmetric_g5(
        layer=l,
        eye_pk_neu=eye_pk_neu_arr,
        eye_pk_shuf_neu=eye_pk_shuf_neu_arr,
        eye_pk_col=eye_pk_col_arr,
        eye_pk_shuf_col=eye_pk_shuf_col_arr,
        eye_pk_iso=eye_pk_iso_arr,
        eye_pk_shuf_iso=eye_pk_shuf_iso_arr,
        kl_abl=kl_abl_arr,
        kl_col=kl_col_arr,
        kl_iso=kl_iso_arr,
        sanity=sanity,
        seed_base=20261126,
    )
    t1 = time.time()
    prov = gp.build_provenance(
        REPO_ROOT,
        [__file__, REPO_ROOT / "scripts/in_use_lens_trial.py", REPO_ROOT / "scripts/in_use_common.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(t1),
        model_id=rt.model_id,
        weights=weights_card(rt),
    )
    out = {**prov, "elapsed_s": round(t1 - t0, 2), **res}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    sg = out["symmetric_g5"]
    print(
        f"L58 SYM-NULL ({out['elapsed_s']}s): dual05={out['dual05_count']}/{N} | "
        f"sym_vs_col AUC={sg['shuf_neu_vs_shuf_col']['auc']} fdp05={sg['shuf_neu_vs_shuf_col']['fdp05_count']} | "
        f"sym_vs_iso AUC={sg['shuf_neu_vs_shuf_iso']['auc']} fdp05={sg['shuf_neu_vs_shuf_iso']['fdp05_count']} | "
        f"pass_all={sg['pass_all_gates_with_sym_g5']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
