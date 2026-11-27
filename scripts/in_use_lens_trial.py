#!/usr/bin/env python3
"""Confirmatory Path 1 (P1): Window-cached in_use_lens trial at L59, L58, L57.

Evaluates pre-registered frozen neuron lists (N=1,024 pure_4_4 neurons per layer,
4 pure_h0 fit + 4 pure_h1 test events each = 8,192 events/layer) against TWO paired
norm-matched control families (C_col=1,024 columns mixtures and C_iso=1,024 isotropic
directions) plus a shuffled-label derangement null.

Verifies clean git tree, SHA256SUMS, and CODE_SHA256SUMS before any GPU work.
Never writes raw corpus text to disk.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from backend import census_fdr as cf  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.gap_stats import _mid_ranks_and_ties  # noqa: E402
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


class _StopAfterLayer(Exception):
    pass


def verify_prereg_integrity(prereg_dir: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Refuse to run if git tree is dirty or SHA256SUMS / CODE_SHA256SUMS fail."""
    st = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if st.stdout.strip():
        raise RuntimeError(f"Refusing confirmatory run on dirty git tree:\n{st.stdout}")
    head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    for fname in ("SHA256SUMS", "CODE_SHA256SUMS"):
        fpath = prereg_dir / fname
        if not fpath.is_file():
            raise RuntimeError(f"Missing {fpath}")
        cwd = prereg_dir if fname == "SHA256SUMS" else repo_root
        for line in fpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            rel = rel.lstrip("*").strip()
            target = cwd / rel
            if not target.is_file():
                raise RuntimeError(f"Missing hashed file {target}")
            got = hashlib.sha256(target.read_bytes()).hexdigest()
            if got != digest:
                raise RuntimeError(f"SHA256 mismatch for {rel}: expected {digest}, got {got}")
    return {
        "git_head": head,
        "repo_commit": (prereg_dir / "repo_commit").read_text(encoding="utf-8").strip() if (prereg_dir / "repo_commit").is_file() else None,
        "prereg_at": (prereg_dir / "prereg_at").read_text(encoding="utf-8").strip() if (prereg_dir / "prereg_at").is_file() else None,
        "prereg_sha256": hashlib.sha256((prereg_dir / "SHA256SUMS").read_bytes()).hexdigest(),
    }


def derangement_perm(n: int, seed: int) -> np.ndarray:
    """Return a deterministic permutation with perm[i] != i for all i."""
    if n < 2:
        raise ValueError("Need at least 2 items for derangement")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    assert np.all(perm != np.arange(n))
    return perm


@torch.no_grad()
def _control_eye_ids(
    ctl_dirs: torch.Tensor,
    faces: Sequence[str],
    w_post: torch.Tensor,
    fnw: torch.Tensor,
    W_U: torch.Tensor,
    k: int = 10,
) -> np.ndarray:
    """Compute exact fp32 softcapped eye_k vocab IDs for signed control directions."""
    out: List[List[int]] = []
    dev = ctl_dirs.device
    sgn = torch.tensor([1.0 if f == "pos" else -1.0 for f in faces], dtype=torch.float32, device=dev)
    wp = w_post.to(dev).float()
    for s in range(0, len(faces), 128):
        e = min(len(faces), s + 128)
        vecs = (sgn[s:e, None] * ctl_dirs[s:e].float()) * wp[None, :]
        hn_bf16 = gemma4_rms_norm(vecs.to(W_U.dtype), fnw)
        logits = logits_fp32(hn_bf16, W_U)
        out.extend(label_top_ids(vecs, fnw, W_U, logits, k=k))
        del vecs, hn_bf16, logits
    return np.asarray(out, dtype=np.int32)


def _load_frozen_neurons(path: Path) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ns, fc = line.split(",")
        assert fc in ("pos", "neg"), f"Invalid face {fc!r}"
        items.append((int(ns), fc))
    return items


def _fdp_from_arrays(neu_scores: np.ndarray, ctl_scores: np.ndarray, *, alpha: float = 0.05, n_boot: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Run step-up empirical FDP (score > 0 required) + 200-boot stability matching backend.census_fdr."""
    scores: Dict[Tuple[str, int], float] = {}
    pos_ok: Dict[Tuple[str, int], bool] = {}
    for i, v in enumerate(neu_scores):
        fv = float(v)
        scores[("neuron", i)] = fv
        pos_ok[("neuron", i)] = fv > 0.0
    for i, v in enumerate(ctl_scores):
        fv = float(v)
        scores[("random", i)] = fv
        pos_ok[("random", i)] = fv > 0.0
    sel = cf.emp_fdp_select(scores, pos_ok, alpha=alpha)
    boot = cf.bootstrap_fdp_stability(scores, pos_ok, alpha=alpha, n_boot=n_boot, seed=seed)
    return {
        "count": int(sel["count"]),
        "threshold": sel["threshold"],
        "fdp": sel["fdp"],
        "above_max_ctl": int(sel["above_max_ctl"]),
        "controls_above_thresh": int(sel["controls_above_thresh"]),
        "named_idx": sorted(int(u[1]) for u in sel["named"]),
        "boot": boot,
    }


def evaluate_p1_layer(
    *,
    layer: int,
    eye_pk_neu: np.ndarray,  # [N, 8] (cols 0..3 fit, 4..7 test)
    eye_pk_shuf: np.ndarray,
    eye_pk_col: np.ndarray,
    eye_pk_iso: np.ndarray,
    kl_abl: np.ndarray,
    kl_col: np.ndarray,
    kl_iso: np.ndarray,
    sanity: Dict[str, Any],
    seed_base: int = 0,
) -> Dict[str, Any]:
    """Compute G1-G5 confirmatory metrics and pass/fail verdict for one layer."""
    N = int(eye_pk_neu.shape[0])
    neu_all = eye_pk_neu.mean(axis=1)
    shuf_all = eye_pk_shuf.mean(axis=1)
    col_all = eye_pk_col.mean(axis=1)
    iso_all = eye_pk_iso.mean(axis=1)

    neu_fit = eye_pk_neu[:, :4].mean(axis=1)
    neu_test = eye_pk_neu[:, 4:].mean(axis=1)
    col_test = eye_pk_col[:, 4:].mean(axis=1)
    iso_test = eye_pk_iso[:, 4:].mean(axis=1)

    st_all_col = emp_p_values(neu_all, col_all)
    st_all_iso = emp_p_values(neu_all, iso_all)
    st_test_col = emp_p_values(neu_test, col_test)
    st_test_iso = emp_p_values(neu_test, iso_test)
    st_shuf_col = emp_p_values(shuf_all, col_all)

    rx, _ = _mid_ranks_and_ties(neu_fit)
    ry, _ = _mid_ranks_and_ties(neu_test)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        rho_fit_test, rho_fit_test_p = 0.0, 1.0
    else:
        rho_fit_test = float(np.corrcoef(rx, ry)[0, 1])
        z_r = rho_fit_test * np.sqrt(max(N - 1, 1))
        import math
        rho_fit_test_p = float(math.erfc(abs(z_r) / math.sqrt(2.0)))

    fdp_col = _fdp_from_arrays(neu_all, col_all, alpha=0.05, n_boot=200, seed=seed_base + layer)
    fdp_iso = _fdp_from_arrays(neu_all, iso_all, alpha=0.05, n_boot=200, seed=seed_base + 100 + layer)
    dual05_idx = sorted(set(fdp_col["named_idx"]) & set(fdp_iso["named_idx"]))

    # Bidirectional cross-pool FDP<=0.05 exchangeability
    xp_iso_vs_col = _fdp_from_arrays(iso_all, col_all, alpha=0.05, n_boot=0, seed=seed_base)
    xp_col_vs_iso = _fdp_from_arrays(col_all, iso_all, alpha=0.05, n_boot=0, seed=seed_base)
    iso_false_vs_col = xp_iso_vs_col["count"] / max(len(iso_all), 1)
    col_false_vs_iso = xp_col_vs_iso["count"] / max(len(col_all), 1)

    fdp_shuf_vs_col = _fdp_from_arrays(shuf_all, col_all, alpha=0.05, n_boot=0, seed=seed_base)

    st_kl_col = summarize_stats(emp_p_values(kl_abl.mean(axis=1), kl_col.mean(axis=1)))
    st_kl_iso = summarize_stats(emp_p_values(kl_abl.mean(axis=1), kl_iso.mean(axis=1)))

    g1_pass = (
        st_all_col["auc"] >= 0.58
        and st_all_col["mw_p"] <= 1e-4
        and st_all_iso["auc"] >= 0.58
        and st_all_iso["mw_p"] <= 1e-4
    )
    g2_pass = (
        st_test_col["auc"] >= 0.57
        and st_test_col["mw_p"] <= 1e-3
        and st_test_iso["auc"] >= 0.57
        and st_test_iso["mw_p"] <= 1e-3
        and float(rho_fit_test) >= 0.30
    )
    min_dual = max(1, int(np.ceil(0.10 * N)))  # 103 for N=1,024
    g3_pass = (
        len(dual05_idx) >= min_dual
        and (fdp_col["boot"]["p10"] if fdp_col["boot"] else 0) >= 60
        and (fdp_col["boot"]["frac_zero"] if fdp_col["boot"] else 1.0) == 0.0
        and (fdp_iso["boot"]["p10"] if fdp_iso["boot"] else 0) >= 60
        and (fdp_iso["boot"]["frac_zero"] if fdp_iso["boot"] else 1.0) == 0.0
    )
    g4_pass = iso_false_vs_col <= 0.02 and col_false_vs_iso <= 0.02
    sanity_ok = (
        float(sanity.get("noop_kl_same_batch", 1.0)) < 1e-5
        and float(sanity.get("max_kl_alg_vs_hook_same_batch", 1.0)) < 1e-5
        and float(sanity.get("max_layer_rec_diff", 1.0)) < 0.05
        and 0.90 <= float(sanity.get("achieved_col_over_abl_post_norm_median", 0.0)) <= 1.15
        and 0.90 <= float(sanity.get("achieved_iso_over_abl_post_norm_median", 0.0)) <= 1.15
    )
    g5_pass = (0.45 <= st_shuf_col["auc"] <= 0.55) and (fdp_shuf_vs_col["count"] == 0) and sanity_ok
    all_pass = bool(g1_pass and g2_pass and g3_pass and g4_pass and g5_pass)

    return {
        "layer": int(layer),
        "n_neurons": N,
        "pass": all_pass,
        "gates": {
            "G1_all8_effect": g1_pass,
            "G2_heldout_test4_and_rho": g2_pass,
            "G3_dual_fdp05_yield": g3_pass,
            "G4_cross_pool_exchangeability": g4_pass,
            "G5_shuffled_null_and_sanity": g5_pass,
        },
        "eye_pk_all8_vs_col": summarize_stats(st_all_col),
        "eye_pk_all8_vs_iso": summarize_stats(st_all_iso),
        "eye_pk_test4_vs_col": summarize_stats(st_test_col),
        "eye_pk_test4_vs_iso": summarize_stats(st_test_iso),
        "rho_fit4_test4": round(float(rho_fit_test), 4),
        "rho_fit4_test4_p": float(f"{rho_fit_test_p:.3e}"),
        "fdp05_vs_col": {k: v for k, v in fdp_col.items() if k != "named_idx"},
        "fdp05_vs_iso": {k: v for k, v in fdp_iso.items() if k != "named_idx"},
        "dual05_count": len(dual05_idx),
        "dual05_frac": round(len(dual05_idx) / max(N, 1), 4),
        "dual05_min_bar": min_dual,
        "cross_pool": {
            "iso_false_vs_col_count": xp_iso_vs_col["count"],
            "iso_false_vs_col_frac": round(iso_false_vs_col, 4),
            "col_false_vs_iso_count": xp_col_vs_iso["count"],
            "col_false_vs_iso_frac": round(col_false_vs_iso, 4),
        },
        "shuffled_null_vs_col": {
            **summarize_stats(st_shuf_col),
            "fdp05_count": fdp_shuf_vs_col["count"],
        },
        "kl_all8_vs_col": st_kl_col,
        "kl_all8_vs_iso": st_kl_iso,
        "sanity": sanity,
        "_dual05_idx": dual05_idx,
        "_col_p": st_all_col["emp_p"].tolist(),
        "_col_q": st_all_col["bh_q"].tolist(),
        "_iso_p": st_all_iso["emp_p"].tolist(),
        "_iso_q": st_all_iso["bh_q"].tolist(),
    }


@torch.no_grad()
def _run_layer_batch0_sanity(
    lm: Any,
    W_U: torch.Tensor,
    layer: int,
    seqs: Sequence[Sequence[int]],
    pks: Sequence[int],
    tgt_ns: Sequence[int],
    ctl_col_idx_b: torch.Tensor,
    ctl_coeffs_b: torch.Tensor,
    v_iso_b: torch.Tensor,
    w_down_l: torch.Tensor,
    w_post_l: torch.Tensor,
    s_l: float,
) -> Dict[str, Any]:
    """Verify bit-exact activation-space F.linear ablation vs eager down_proj zero hook on 4 events."""
    dev = model_device(lm)
    w_ops_dev = dev
    b = len(seqs)
    ib, ab, _, _ = pad_right(seqs, w_ops_dev)
    pb = (ab.cumsum(-1) - 1).clamp(min=0)
    pk_t = torch.tensor(pks, dtype=torch.long, device=dev)
    rows_i = torch.arange(b, device=dev)
    t_idx = torch.tensor(tgt_ns, dtype=torch.long, device=dev)
    post_ffn_ln = lm.layers[layer].post_feedforward_layernorm
    D = int(w_down_l.shape[0])
    V = int(W_U.shape[0])

    # 1) Full unperturbed forward
    cap: Dict[str, torch.Tensor] = {}
    hks = [
        lm.layers[layer].register_forward_pre_hook(lambda _m, a: cap.__setitem__("h_in", a[0].detach())),
        lm.layers[layer].pre_feedforward_layernorm.register_forward_pre_hook(lambda _m, a: cap.__setitem__("resid_mid", a[0].detach())),
        lm.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda _m, a: cap.__setitem__("act_all", a[0].detach())),
        lm.layers[layer].mlp.down_proj.register_forward_hook(lambda _m, _a, o: cap.__setitem__("y_mlp", o.detach())),
        lm.layers[layer].register_forward_hook(lambda _m, _a, o: cap.__setitem__("layer_out", (o[0] if isinstance(o, tuple) else o).detach())),
    ]
    try:
        out_full = lm(input_ids=ib, attention_mask=ab, use_cache=False)
    finally:
        for hk in hks:
            hk.remove()
    lsm_full = F.log_softmax(logits_fp32(out_full.last_hidden_state[rows_i, pk_t], W_U), dim=-1)

    # 2) Same-batch reconstruction + ablation + col + iso
    act_all = cap["act_all"]
    y_mlp = cap["y_mlp"]
    resid_mid = cap["resid_mid"]
    layer_out = cap["layer_out"]

    a_tok = act_all[rows_i, pk_t, t_idx].float()
    x_abl = act_all.clone()
    x_abl[rows_i, pk_t, t_idx] = 0
    y_abl = F.linear(x_abl, w_down_l)

    x_col = act_all.clone()
    cur_c = x_col[rows_i[:, None], pk_t[:, None], ctl_col_idx_b].float()
    x_col[rows_i[:, None], pk_t[:, None], ctl_col_idx_b] = (cur_c - a_tok[:, None] * ctl_coeffs_b).to(act_all.dtype)
    y_col = F.linear(x_col, w_down_l)

    y_iso = y_mlp.clone()
    y_iso[rows_i, pk_t] = (y_iso[rows_i, pk_t].float() - a_tok[:, None] * v_iso_b.float()).to(y_mlp.dtype)

    post_rec = (resid_mid + post_ffn_ln(y_mlp)) * s_l
    post_abl = (resid_mid + post_ffn_ln(y_abl)) * s_l
    max_layer_rec_diff = float((layer_out - post_rec).abs().max().item())

    wp = w_post_l.float()
    abl_norm = ((y_mlp[rows_i, pk_t].float() - y_abl[rows_i, pk_t].float()) * wp[None, :]).norm(dim=-1).clamp(min=1e-6)
    col_norm = ((y_mlp[rows_i, pk_t].float() - y_col[rows_i, pk_t].float()) * wp[None, :]).norm(dim=-1)
    iso_norm = ((y_mlp[rows_i, pk_t].float() - y_iso[rows_i, pk_t].float()) * wp[None, :]).norm(dim=-1)
    col_ratio_med = float((col_norm / abl_norm).median().item())
    iso_ratio_med = float((iso_norm / abl_norm).median().item())

    # 3) Eager zeroing pre-hook on full forward from layer
    cap_y: Dict[str, torch.Tensor] = {}

    def _zero_hook(_m: Any, f_args: Tuple[torch.Tensor, ...]) -> Tuple[Any, ...]:
        x = f_args[0].clone()
        x[rows_i, pk_t, t_idx] = 0
        return (x,) + tuple(f_args[1:])

    hk_z = lm.layers[layer].mlp.down_proj.register_forward_pre_hook(_zero_hook)
    hk_y = lm.layers[layer].mlp.down_proj.register_forward_hook(lambda _m, _a, o: cap_y.__setitem__("y_hook", o.detach()))
    try:
        out_hook = ao.forward_from_layer(lm, layer, cap["h_in"], ab, pb)
    finally:
        hk_z.remove()
        hk_y.remove()
    y_abl_vs_hook = float((y_abl - cap_y["y_hook"]).abs().max().item())
    lsm_hook = F.log_softmax(logits_fp32(out_hook[rows_i, pk_t], W_U), dim=-1)

    h_rec = ao.forward_from_layer(lm, layer + 1, post_rec, ab, pb)
    h_abl = ao.forward_from_layer(lm, layer + 1, post_abl, ab, pb)
    lsm_rec = F.log_softmax(logits_fp32(h_rec[rows_i, pk_t], W_U), dim=-1)
    lsm_abl = F.log_softmax(logits_fp32(h_abl[rows_i, pk_t], W_U), dim=-1)

    noop_kl = float((lsm_full.exp() * (lsm_full - lsm_rec)).sum(-1).mean().item())
    kl_alg = (lsm_rec.exp() * (lsm_rec - lsm_abl)).sum(-1)
    kl_hk = (lsm_full.exp() * (lsm_full - lsm_hook)).sum(-1)
    max_kl_diff = float((kl_alg - kl_hk).abs().max().item())

    del cap, cap_y, ib, ab, pb, out_full, out_hook, h_rec, h_abl, lsm_full, lsm_hook, lsm_rec, lsm_abl
    del act_all, x_abl, x_col, y_mlp, y_abl, y_col, y_iso, resid_mid, layer_out, post_rec, post_abl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "noop_kl_same_batch": round(noop_kl, 8),
        "max_kl_alg_vs_hook_same_batch": round(max_kl_diff, 8),
        "y_abl_vs_hook_max_abs": round(y_abl_vs_hook, 6),
        "max_layer_rec_diff": round(max_layer_rec_diff, 6),
        "achieved_col_over_abl_post_norm_median": round(col_ratio_med, 6),
        "achieved_iso_over_abl_post_norm_median": round(iso_ratio_med, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirmatory Path 1: Window-cached in_use_lens trial")
    ap.add_argument("--prereg-dir", default="experiments/results/in_use_lens_trial/prereg")
    ap.add_argument("--layers", default="59,58,57")
    ap.add_argument("--out-dir", default="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/in_use_lens")
    ap.add_argument("--results-dir", default="experiments/results/in_use_lens_trial")
    ap.add_argument("--batch-windows", type=int, default=8)
    ap.add_argument("--batch-events", type=int, default=4)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    prereg_dir = Path(args.prereg_dir) if Path(args.prereg_dir).is_absolute() else REPO_ROOT / args.prereg_dir
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else REPO_ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    prereg_info = verify_prereg_integrity(prereg_dir, REPO_ROOT)
    print(f"Pre-registration verified: HEAD={prereg_info['git_head'][:12]}", flush=True)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
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

    verdicts: Dict[str, Any] = {}
    for l in layers:
        tl0 = time.time()
        nfile = prereg_dir / f"neurons_L{l}.txt"
        items = _load_frozen_neurons(nfile)
        N = len(items)
        nids = [n for n, _ in items]
        faces = [fc for _, fc in items]

        # Collect 4 fit pure_h0 + 4 test pure_h1 events per frozen neuron
        events_by_win: Dict[int, List[Tuple[int, int, int, int, float]]] = {}
        # tuple: (ni, slot_k, pk_pos_in_seq, at_in_win, ev_act)
        win_seq_cache: Dict[int, Tuple[List[int], int]] = {}  # win -> (seq up to max_end, max_at)
        win_max_end: Dict[int, int] = {}
        temp_evs: List[Tuple[int, int, int, int, float]] = []
        for ni, (nid, fc) in enumerate(items):
            ev = collect_face_events(
                P, windows, act_scale, pure_h0_mask, pure_h1_mask, doc_id, half_id, l, nid, fc, eos_id=eos_id, bos_id=bos_id
            )
            assert len(ev["pure_h0"]) >= 4 and len(ev["pure_h1"]) >= 4, f"L{l} n={nid} {fc} lacks pure_4_4"
            chosen = ev["pure_h0"][:4] + ev["pure_h1"][:4]
            for slot_k, row in enumerate(chosen):
                wi, at_i, act_v = int(row["win"]), int(row["at"]), float(row["act"])
                win_max_end[wi] = max(win_max_end.get(wi, 0), at_i)
                temp_evs.append((wi, ni, slot_k, at_i, act_v))

        for wi, max_at in win_max_end.items():
            sq, _ = build_event_sequence(windows, wi, max_at, bos_id=bos_id, pad_tail=0)
            # Map window index at_i -> index in sq (1-to-1 since windows has no <0 before valid tokens)
            row = windows[wi]
            at_to_pk: Dict[int, int] = {}
            t_len = 1
            for pos in range(1, max_at + 1):
                if int(row[pos]) >= 0:
                    at_to_pk[pos] = t_len
                    t_len += 1
            win_seq_cache[wi] = (sq, at_to_pk)  # type: ignore[assignment]

        for wi, ni, slot_k, at_i, act_v in temp_evs:
            _, at_to_pk = win_seq_cache[wi]
            pk = at_to_pk[at_i]
            events_by_win.setdefault(wi, []).append((ni, slot_k, pk, at_i, act_v))

        # Lens sanity overlap check on first 20 neurons at layer l
        sanity_ns = nids[:20]
        w_post_l = w_ops.norm(l, "post_feedforward").to(dev)
        w_down_l = w_ops.w_down(l).to(dev)
        vecs_chk = (w_down_l.index_select(1, torch.tensor(sanity_ns, device=dev, dtype=torch.long)).T * w_post_l[None, :]).to(W_U.dtype)
        logits_chk = logits_fp32(gemma4_rms_norm(vecs_chk, fnw), W_U)
        got_chk_pos = label_top_ids(vecs_chk, fnw, W_U, logits_chk, k=10)
        got_chk_neg = label_top_ids(-vecs_chk, fnw, W_U, -logits_chk, k=10)
        ov_pos = sum(len(set(got_chk_pos[i]) & set(int(x) for x in lens["eye_pos"][l, sanity_ns[i], :10])) for i in range(20))
        ov_neg = sum(len(set(got_chk_neg[i]) & set(int(x) for x in lens["eye_neg"][l, sanity_ns[i], :10])) for i in range(20))
        if ov_pos < 190 or ov_neg < 190:
            raise RuntimeError(f"L{l} lens.npz sanity check failed: pos={ov_pos}/200 neg={ov_neg}/200")
        del vecs_chk, logits_chk

        # Build paired columns & isotropic controls (exact w_post-weighted write norm match)
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

        # Pre-fixed eye_10 IDs
        eye_neu = torch.from_numpy(
            np.stack([lens[f"eye_{fc}"][l, nid, :10] for nid, fc in items], axis=0).astype(np.int64)
        ).to(dev)
        perm_shuf = torch.from_numpy(derangement_perm(N, 20261126300 + l)).to(dev)
        eye_shuf = eye_neu.index_select(0, perm_shuf)
        eye_col = torch.from_numpy(_control_eye_ids(w_col_dirs, faces, w_post_l, fnw, W_U, k=10).astype(np.int64)).to(dev)
        eye_iso = torch.from_numpy(_control_eye_ids(w_iso_dirs, faces, w_post_l, fnw, W_U, k=10).astype(np.int64)).to(dev)
        del w_col_dirs, iso_raw

        # Batch-0 bit-exact eager hook sanity on first 4 events
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
        print(f"L{l} Batch-0 sanity: {sanity}", flush=True)
        if sanity["noop_kl_same_batch"] >= 1e-5 or sanity["max_kl_alg_vs_hook_same_batch"] >= 1e-5:
            raise RuntimeError(f"L{l} batch-0 sanity failed: {sanity}")

        # Accumulators
        eye_pk_neu_arr = np.zeros((N, 8), dtype=np.float64)
        eye_pk_shuf_arr = np.zeros((N, 8), dtype=np.float64)
        eye_pk_col_arr = np.zeros((N, 8), dtype=np.float64)
        eye_pk_iso_arr = np.zeros((N, 8), dtype=np.float64)
        kl_abl_arr = np.zeros((N, 8), dtype=np.float64)
        kl_col_arr = np.zeros((N, 8), dtype=np.float64)
        kl_iso_arr = np.zeros((N, 8), dtype=np.float64)

        win_order = sorted(events_by_win.keys(), key=lambda w: len(win_seq_cache[w][0]))
        post_ffn_ln = lm.layers[l].post_feedforward_layernorm
        s_l = float(w_ops.scalars[l])
        need_tail = (l + 1) < len(lm.layers)
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

            # Flatten events inside this window batch
            ev_meta: List[Tuple[int, int, int, int]] = []  # (w_local_row, pk, ni, slot_k)
            for row_i, wi in enumerate(w_batch):
                for ni, slot_k, pk, _at, _av in events_by_win[wi]:
                    ev_meta.append((row_i, pk, ni, slot_k))

            r_rows = torch.tensor([m[0] for m in ev_meta], dtype=torch.long, device=dev)
            r_pks = torch.tensor([m[1] for m in ev_meta], dtype=torch.long, device=dev)
            r_nis = torch.tensor([m[2] for m in ev_meta], dtype=torch.long, device=dev)

            act_ev = cap_w["act_all"][r_rows, r_pks]  # [M_ev, N_ff]
            ym_ev = cap_w["y_mlp"][r_rows, r_pks]  # [M_ev, D]
            rm_ev = cap_w["resid_mid"][r_rows, r_pks]  # [M_ev, D]
            lo_ev = cap_w["layer_out"][r_rows, r_pks]  # [M_ev, D]
            layer_out_full = cap_w["layer_out"] if need_tail else None
            del cap_w, ib_w, ab_w

            tgt_ev = tidx_full.index_select(0, r_nis)
            ev_arange = torch.arange(len(ev_meta), device=dev)
            a_tok = act_ev[ev_arange, tgt_ev].float()  # [M_ev]

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
            if not need_tail:
                # L59: post_norm -> lm.norm -> logits at peak position ONLY (no transformer layers)
                p4 = torch.cat([post_rec_ev, post_abl_ev, post_col_ev, post_iso_ev], dim=0)
                hn = lm.norm(p4)
                lsm = F.log_softmax(logits_fp32(hn, W_U), dim=-1).view(4, M_ev, V)
                p_all = lsm.exp()
                p_rec, p_abl, p_col, p_iso = p_all[0], p_all[1], p_all[2], p_all[3]

                ka = (p_rec * (lsm[0] - lsm[1])).sum(dim=-1).cpu().numpy()
                kc = (p_rec * (lsm[0] - lsm[2])).sum(dim=-1).cpu().numpy()
                ki = (p_rec * (lsm[0] - lsm[3])).sum(dim=-1).cpu().numpy()

                en_idx = eye_neu.index_select(0, r_nis)
                es_idx = eye_shuf.index_select(0, r_nis)
                ec_idx = eye_col.index_select(0, r_nis)
                ei_idx = eye_iso.index_select(0, r_nis)

                ep_neu = (p_rec - p_abl).gather(-1, en_idx).sum(-1).cpu().numpy()
                ep_shf = (p_rec - p_abl).gather(-1, es_idx).sum(-1).cpu().numpy()
                ep_col = (p_rec - p_col).gather(-1, ec_idx).sum(-1).cpu().numpy()
                ep_iso = (p_rec - p_iso).gather(-1, ei_idx).sum(-1).cpu().numpy()

                for idx_e, (_row_i, _pk, ni, slot_k) in enumerate(ev_meta):
                    kl_abl_arr[ni, slot_k] = float(ka[idx_e])
                    kl_col_arr[ni, slot_k] = float(kc[idx_e])
                    kl_iso_arr[ni, slot_k] = float(ki[idx_e])
                    eye_pk_neu_arr[ni, slot_k] = float(ep_neu[idx_e])
                    eye_pk_shuf_arr[ni, slot_k] = float(ep_shf[idx_e])
                    eye_pk_col_arr[ni, slot_k] = float(ep_col[idx_e])
                    eye_pk_iso_arr[ni, slot_k] = float(ep_iso[idx_e])
                del p4, hn, lsm, p_all
            else:
                # L58 / L57: run only layers l+1..59 on exact [0..pk] prefixes in micro-batches of batch_events
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
                    ep_shf = (p_rec - p_abl).gather(-1, eye_shuf.index_select(0, nis_t_m)).sum(-1).cpu().numpy()
                    ep_col = (p_rec - p_col).gather(-1, eye_col.index_select(0, nis_t_m)).sum(-1).cpu().numpy()
                    ep_iso = (p_rec - p_iso).gather(-1, eye_iso.index_select(0, nis_t_m)).sum(-1).cpu().numpy()

                    for bi_m in range(m):
                        ni, slot_k = nis_m[bi_m], slots_m[bi_m]
                        kl_abl_arr[ni, slot_k] = float(ka[bi_m])
                        kl_col_arr[ni, slot_k] = float(kc[bi_m])
                        kl_iso_arr[ni, slot_k] = float(ki[bi_m])
                        eye_pk_neu_arr[ni, slot_k] = float(ep_neu[bi_m])
                        eye_pk_shuf_arr[ni, slot_k] = float(ep_shf[bi_m])
                        eye_pk_col_arr[ni, slot_k] = float(ep_col[bi_m])
                        eye_pk_iso_arr[ni, slot_k] = float(ep_iso[bi_m])
                    del h4, am, pm, h_fin, h_pk, lsm, p_all

            del post_rec_ev, post_abl_ev, post_col_ev, post_iso_ev, layer_out_full
            if (ws // int(args.batch_windows)) % 40 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        sanity["max_window_batch_rec_diff"] = round(max_ev_rec_diff, 6)
        v_res = evaluate_p1_layer(
            layer=l,
            eye_pk_neu=eye_pk_neu_arr,
            eye_pk_shuf=eye_pk_shuf_arr,
            eye_pk_col=eye_pk_col_arr,
            eye_pk_iso=eye_pk_iso_arr,
            kl_abl=kl_abl_arr,
            kl_col=kl_col_arr,
            kl_iso=kl_iso_arr,
            sanity=sanity,
            seed_base=20261126,
        )
        v_res["elapsed_s"] = round(time.time() - tl0, 2)
        v_res["unique_windows_forward"] = len(win_order)

        # Save full compact units file in isolated out_dir + compact verdict in results_dir
        dual_set = set(v_res["_dual05_idx"])
        units_rows = []
        for idx_i, (nid, fc) in enumerate(items):
            units_rows.append(
                {
                    "neuron": nid,
                    "face": fc,
                    "eye_pk_all8": round(float(eye_pk_neu_arr[idx_i].mean()), 6),
                    "eye_pk_fit4": round(float(eye_pk_neu_arr[idx_i, :4].mean()), 6),
                    "eye_pk_test4": round(float(eye_pk_neu_arr[idx_i, 4:].mean()), 6),
                    "kl_all8": round(float(kl_abl_arr[idx_i].mean()), 6),
                    "col_p": round(float(v_res["_col_p"][idx_i]), 6),
                    "col_q": round(float(v_res["_col_q"][idx_i]), 6),
                    "iso_p": round(float(v_res["_iso_p"][idx_i]), 6),
                    "iso_q": round(float(v_res["_iso_q"][idx_i]), 6),
                    "dual_fdp05": idx_i in dual_set,
                }
            )
        (out_dir / f"L{l}_units.json").write_text(json.dumps(units_rows, indent=1) + "\n", encoding="utf-8")

        verdict_clean = {k: v for k, v in v_res.items() if not k.startswith("_")}
        verdict_clean["dual05_neurons"] = [f"n{items[i][0]}{'p' if items[i][1]=='pos' else 'm'}" for i in v_res["_dual05_idx"]]
        (out_dir / f"L{l}_verdict.json").write_text(json.dumps(verdict_clean, indent=1) + "\n", encoding="utf-8")
        (results_dir / f"L{l}_verdict.json").write_text(json.dumps(verdict_clean, indent=1) + "\n", encoding="utf-8")
        verdicts[str(l)] = verdict_clean
        print(
            f"L{l} VERDICT ({v_res['elapsed_s']:.1f}s): pass={verdict_clean['pass']} gates={verdict_clean['gates']} "
            f"dual05={verdict_clean['dual05_count']}/{N} all8_col_auc={verdict_clean['eye_pk_all8_vs_col']['auc']} "
            f"test4_col_auc={verdict_clean['eye_pk_test4_vs_col']['auc']} rho={verdict_clean['rho_fit4_test4']} "
            f"xp={verdict_clean['cross_pool']}",
            flush=True,
        )
        del w_down_l, w_post_l, w_iso_dirs, ctl_col_idx, ctl_coeffs, eye_neu, eye_shuf, eye_col, eye_iso
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t1 = time.time()
    prov = gp.build_provenance(
        REPO_ROOT,
        [__file__, REPO_ROOT / "scripts/in_use_common.py", prereg_dir / "SHA256SUMS"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(t1),
        model_id=rt.model_id,
        weights=weights_card(rt),
        neuron_ids={lk: [f"n{n}{'p' if f=='pos' else 'm'}" for n, f in _load_frozen_neurons(prereg_dir / f"neurons_L{lk}.txt")] for lk in layers},
    )
    summary = {
        **prov,
        "prereg": prereg_info,
        "elapsed_s": round(t1 - t0, 2),
        "layers": verdicts,
    }
    (results_dir / "verdict.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out_dir / "verdict.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {results_dir / 'verdict.json'} in {t1 - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
