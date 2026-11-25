#!/usr/bin/env python3
"""CPU-only unit tests for backend/forced_odd.py."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORKTREE = Path("/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from backend import forced_odd as fo  # noqa: E402


def test_mean_centering_invariance() -> None:
    rng = np.random.default_rng(7)
    b, H, S, V = 3, 2, 4, 256
    base_logits = torch.from_numpy(rng.normal(0.0, 2.0, size=(H, S, V)).astype(np.float32))
    base_lsm = F.log_softmax(base_logits, dim=-1)
    z0, z0_norm2, base_mean = fo.precompute_base_stats(base_lsm)

    dp = torch.from_numpy(rng.normal(0.0, 0.25, size=(b, H, S, V)).astype(np.float32))
    dn = torch.from_numpy(rng.normal(0.0, 0.25, size=(b, H, S, V)).astype(np.float32))
    lsm0 = torch.stack([F.log_softmax(base_logits[None] + dp, dim=-1), F.log_softmax(base_logits[None] + dn, dim=-1)], dim=0)
    ext0 = fo.extract_odd_from_lsm(lsm0.clone(), base_lsm, z0, z0_norm2, base_mean, readout_mode="both", clamp_val=6.0)

    # Add arbitrary per-(b, h, s) constant shifts to raw logits & directly to lsm_p BEFORE extraction
    shift = torch.from_numpy(rng.normal(0.0, 5.0, size=(b, H, S, 1)).astype(np.float32))
    lsm1 = lsm0.clone()
    lsm1[0] = lsm1[0] + shift
    ext1 = fo.extract_odd_from_lsm(lsm1, base_lsm, z0, z0_norm2, base_mean, readout_mode="both", clamp_val=6.0)

    diff_cent = float((ext1["odd_cent"].float() - ext0["odd_cent"].float()).abs().max().item())
    diff_proj = float((ext1["odd_proj"].float() - ext0["odd_proj"].float()).abs().max().item())
    diff_alpha = float(np.max(np.abs(ext1["alpha_temp"] - ext0["alpha_temp"])))
    assert diff_cent < 5e-4, f"odd_cent not invariant to Δ log Z: {diff_cent}"
    assert diff_proj < 5e-4, f"odd_proj not invariant to Δ log Z: {diff_proj}"
    assert diff_alpha < 1e-5, f"alpha_temp not invariant to Δ log Z: {diff_alpha}"
    print(f"[PASS] mean-centering invariance (diff_cent={diff_cent:.2e}, diff_proj={diff_proj:.2e})")


def test_temperature_projection_invariance() -> None:
    rng = np.random.default_rng(13)
    b, H, S, V = 3, 2, 4, 256
    base_logits = torch.from_numpy(rng.normal(0.0, 2.5, size=(H, S, V)).astype(np.float32))
    base_lsm = F.log_softmax(base_logits, dim=-1)
    z0, z0_norm2, base_mean = fo.precompute_base_stats(base_lsm)

    # 1) Exact additive +c * z0 on odd log-odds
    dp = torch.from_numpy(rng.normal(0.0, 0.2, size=(b, H, S, V)).astype(np.float32))
    dn = torch.from_numpy(rng.normal(0.0, 0.2, size=(b, H, S, V)).astype(np.float32))
    lsm_orig = torch.stack([base_lsm[None] + dp, base_lsm[None] + dn], dim=0)
    ext_orig = fo.extract_odd_from_lsm(lsm_orig.clone(), base_lsm, z0, z0_norm2, base_mean, readout_mode="both", clamp_val=6.0)

    c = torch.tensor([0.35, -0.42, 0.78], dtype=torch.float32)[:, None, None, None]
    lsm_scaled = lsm_orig.clone()
    lsm_scaled[0] = lsm_scaled[0] + c * z0[None]
    lsm_scaled[1] = lsm_scaled[1] - c * z0[None]
    ext_scaled = fo.extract_odd_from_lsm(lsm_scaled, base_lsm, z0, z0_norm2, base_mean, readout_mode="both", clamp_val=6.0)

    # Note float16 quantization of odd_proj has eps ~ 1e-3, so check fp16 noise bound <= 5e-3 and exact alpha_temp shift
    diff_proj = float((ext_scaled["odd_proj"].float() - ext_orig["odd_proj"].float()).abs().max().item())
    diff_alpha = float(np.max(np.abs((ext_scaled["alpha_temp"] - ext_orig["alpha_temp"]) - c.squeeze().numpy())))
    assert diff_proj < 5e-3, f"odd_proj changed under +c*z0: {diff_proj}"
    assert diff_alpha < 1e-5, f"alpha_temp failed to recover +c: {diff_alpha}"

    # 2) Real nonlinear softmax temperature scaling: logits((1 +/- c) * base_logits) -> odd_proj must be ~0 and alpha_temp == c
    base_centered = base_logits - base_logits.mean(dim=-1, keepdim=True)
    c_pure = torch.tensor([0.25, -0.30, 0.50], dtype=torch.float32)[:, None, None, None]
    lsm_temp = torch.stack(
        [
            F.log_softmax((1.0 + c_pure) * base_centered[None], dim=-1),
            F.log_softmax((1.0 - c_pure) * base_centered[None], dim=-1),
        ],
        dim=0,
    )
    ext_temp = fo.extract_odd_from_lsm(lsm_temp, base_lsm, z0, z0_norm2, base_mean, readout_mode="both", clamp_val=6.0)
    max_resid = float(ext_temp["odd_proj"].float().abs().max().item())
    alpha_err = float(np.max(np.abs(ext_temp["alpha_temp"] - c_pure.squeeze().numpy())))
    assert max_resid < 1e-5, f"nonlinear softmax temperature residual not zeroed: {max_resid}"
    assert alpha_err < 1e-5, f"nonlinear softmax alpha mismatch: {alpha_err}"
    print(f"[PASS] temperature-projection invariance (add_diff={diff_proj:.2e}, temp_resid={max_resid:.2e}, alpha_err={alpha_err:.2e})")


def test_streaming_exactness() -> None:
    rng = np.random.default_rng(42)
    C, N, V = 64, 40, 256
    U = C + N
    dev = torch.device("cpu")
    mask = torch.ones(V, dtype=torch.bool)
    mask[:16] = False

    A = torch.from_numpy(rng.normal(0.0, 0.5, size=(U, V)).astype(np.float16))
    B = torch.from_numpy(rng.normal(0.0, 0.5, size=(U, V)).astype(np.float16))
    # Implant strong replicating signals on first 6 neurons
    for i in range(6):
        A[C + i, 20 + i * 3 : 23 + i * 3] += 3.0
        B[C + i, 20 + i * 3 : 23 + i * 3] += 3.0
        A[C + i, 120 + i * 3 : 123 + i * 3] -= 3.0
        B[C + i, 120 + i * 3 : 123 + i * 3] -= 3.0

    cov = np.column_stack(
        [
            np.ones(U),
            rng.uniform(0.05, 0.5, size=U),
            rng.uniform(0.05, 0.5, size=U),
            rng.normal(-1.0, 0.3, size=U),
            rng.uniform(0.3, 1.5, size=U),
        ]
    )
    ctl_mask = np.zeros(U, dtype=bool)
    ctl_mask[:C] = True

    for zv in (False, True):
        full = fo.score_odd_pair_gpu(A, B, ctl_mask, cov, mask, zv, dev, keep_words=True, top_k=5)
        model = fo.fit_control_model(A[:C], B[:C], cov[:C], mask, zv, dev, keep_words=True, top_k=5, chunk_size=16)
        chunked = fo.score_chunk_against_controls(A[C:], B[C:], cov[C:], model, mask, dev, keep_words=True, top_k=5, chunk_size=13)
        for key in ("r_ab", "r_ba", "r_ab_res", "r_ba_res", "T"):
            diff = float(np.max(np.abs(full[key][C:] - chunked[key])))
            assert diff < 1e-4, f"streaming mismatch (zv={zv}, key={key}): {diff}"
        for key in ("topA_p", "topA_n", "topB_p", "topB_n"):
            assert np.array_equal(full[key][C:], chunked[key]), f"top-5 piece IDs mismatch (zv={zv}, key={key})"
    print("[PASS] streaming chunk exactness vs score_odd_pair_gpu (both L2_cent and L2_zsc)")


def test_fdp_and_ladder() -> None:
    # Step-up FDP k_min and tie handling:
    # N=100, C=100, alpha=0.05 -> single count requires (100/101)*(1+0)/j <= 0.05 -> j >= 20
    ctl = np.linspace(0.0, 1.0, 100)
    neu = np.concatenate([np.full(15, 2.0), np.full(10, 1.5), np.linspace(0.0, 0.9, 75)])
    pok = np.ones(100, dtype=bool)
    sel = fo.empirical_fdp(neu, pok, 100, ctl, 0.05)
    assert sel["count"] == 25, f"expected all 25 tied/top discoveries above ctl max, got {sel}"
    assert sel["above_max_ctl"] == 25

    # Only 10 above ctl max -> k_min=20 blocks naming (< 20 above 1.0, 20th sits inside ctl)
    neu_small = np.concatenate([np.full(10, 2.0), np.zeros(90)])
    sel_small = fo.empirical_fdp(neu_small, pok, 100, ctl, 0.05)
    assert sel_small["count"] == 0 and sel_small["above_max_ctl"] == 10, f"expected k_min=0, got {sel_small}"

    # Tie at threshold: 19 at 2.0 plus 2 tied at 1.5 > 1.0 -> tie grouping jumps j=20 (fdp <= 0.05), includes both -> 21
    neu_tie = np.concatenate([np.full(19, 2.0), np.full(2, 1.5), np.zeros(79)])
    sel_tie = fo.empirical_fdp(neu_tie, pok, 100, ctl, 0.05)
    assert sel_tie["count"] == 21, f"expected tie-grouped count=21, got {sel_tie}"

    # Log-log ladder interpolation
    rungs = [0.1, 0.2, 0.4, 0.8]
    kls = [0.04, 0.16, 0.64, 2.56]  # kl = 4 * d^2 -> d(0.25) = 0.25
    d25, note = fo.loglog_interpolate_dose(rungs, kls, target=0.25)
    assert note is None and abs(d25 - 0.25) < 1e-3, f"expected 0.25, got {d25}, {note}"
    d_lo, note_lo = fo.loglog_interpolate_dose(rungs, kls, target=0.01)
    assert d_lo == 0.1 and note_lo is not None
    print(f"[PASS] empirical_fdp ties/k_min & loglog_interpolate_dose (d25={d25})")


def test_weight_word_streaming_exactness() -> None:
    rng = np.random.default_rng(101)
    C, N, V, D = 100, 40, 256, 16
    dev = torch.device("cpu")
    mask = torch.ones(V, dtype=torch.bool)
    mask[:16] = False

    class _FakeRT:
        def __init__(self, fnw: torch.Tensor, W_U: torch.Tensor) -> None:
            self.final_norm_weight = fnw
            self.lm_head_weight = W_U

    fnw = torch.ones(D, dtype=torch.float32)
    W_U = torch.from_numpy(rng.normal(0.0, 0.25, size=(V, D)).astype(np.float32))
    rt = _FakeRT(fnw, W_U)

    col_raw = rng.normal(0.0, 1.0, size=(C, D)).astype(np.float32)
    col_dirs = col_raw / np.linalg.norm(col_raw, axis=1, keepdims=True)
    A_ctl = torch.from_numpy(rng.normal(0.0, 0.4, size=(C, V)).astype(np.float16))
    B_ctl = torch.from_numpy(rng.normal(0.0, 0.4, size=(C, V)).astype(np.float16))
    A_neu = torch.from_numpy(rng.normal(0.0, 0.4, size=(N, V)).astype(np.float16))
    B_neu = torch.from_numpy(rng.normal(0.0, 0.4, size=(N, V)).astype(np.float16))

    eye_pos = np.zeros((1, N, 12), dtype=np.int32)
    eye_neg = np.zeros((1, N, 12), dtype=np.int32)
    for n in range(N):
        perm = rng.choice(np.arange(16, V), size=24, replace=False)
        eye_pos[0, n] = perm[:12]
        eye_neg[0, n] = perm[12:24]
    # Implant direct own-word signal on first 10 neurons
    for n in range(10):
        A_neu[n, eye_pos[0, n, :10]] += 2.5
        B_neu[n, eye_pos[0, n, :10]] += 2.5
        A_neu[n, eye_neg[0, n, :10]] -= 2.5
        B_neu[n, eye_neg[0, n, :10]] -= 2.5

    cov_ctl = np.column_stack([np.ones(C), rng.uniform(-1, 1, size=(C, 4))])
    model = fo.fit_control_model(A_ctl, B_ctl, cov_ctl, mask, zscore_var=False, dev=dev)
    ser = fo.serialize_control_model(model)
    model2 = fo.deserialize_control_model(ser, dev=dev)
    assert float(np.max(np.abs(model.T_ctl - model2.T_ctl))) == 0.0
    assert float((model.muA - model2.muA).abs().max().item()) == 0.0

    with tempfile.TemporaryDirectory() as td:
        dd = Path(td)
        (dd / "index").mkdir()
        np.savez(dd / "index/lens.npz", eye_pos=eye_pos, eye_neg=eye_neg)
        direct_res = fo.corrected_weight_word_tag(
            rt, 0, list(range(N)), col_dirs, A_ctl, B_ctl, A_neu, B_neu, model, mask, dev, data_dir=dd
        )

    sw_ctl = fo.compute_weight_word_ctl_null(rt, col_dirs, A_ctl, B_ctl, model2, mask, dev)
    sw0 = fo.score_weight_word_chunk(eye_pos[0], eye_neg[0], list(range(20)), A_neu[:20], B_neu[:20], model2, mask, dev)
    sw1 = fo.score_weight_word_chunk(eye_pos[0], eye_neg[0], list(range(20, N)), A_neu[20:], B_neu[20:], model2, mask, dev)
    stream_res = fo.finalize_weight_word_tag(np.concatenate([sw0, sw1]), sw_ctl)

    diff_sw = float(np.max(np.abs(direct_res["sw_neu"] - stream_res["sw_neu"])))
    assert diff_sw < 1e-5, f"weight-word streaming mismatch: {diff_sw}"
    assert np.array_equal(direct_res["is_direct"], stream_res["is_direct"])
    assert direct_res["direct_q05_count"] >= 10
    print(f"[PASS] weight-word streaming & ControlModel round-trip (max_diff={diff_sw:.2e}, direct={direct_res['direct_q05_count']})")


def test_dual_null_assembly() -> None:
    rng = np.random.default_rng(202)
    C_col, C_iso, N, V = 100, 100, 40, 256  # k_min=8 at alpha=0.05
    dev = torch.device("cpu")
    mask = torch.ones(V, dtype=torch.bool)
    mask[:16] = False

    A_col = torch.from_numpy(rng.normal(0.0, 0.4, size=(C_col, V)).astype(np.float16))
    B_col = torch.from_numpy(rng.normal(0.0, 0.4, size=(C_col, V)).astype(np.float16))
    A_iso = torch.from_numpy(rng.normal(0.0, 0.4, size=(C_iso, V)).astype(np.float16))
    B_iso = torch.from_numpy(rng.normal(0.0, 0.4, size=(C_iso, V)).astype(np.float16))
    A_neu = torch.from_numpy(rng.normal(0.0, 0.4, size=(N, V)).astype(np.float16))
    B_neu = torch.from_numpy(rng.normal(0.0, 0.4, size=(N, V)).astype(np.float16))

    # First 10 neurons beat BOTH col and iso; neurons 10..17 beat ONLY col (iso controls share pieces 180..199!)
    for i in range(10):
        A_neu[i, 20 + i * 2 : 22 + i * 2] += 3.5
        B_neu[i, 20 + i * 2 : 22 + i * 2] += 3.5
        A_neu[i, 80 + i * 2 : 82 + i * 2] -= 3.5
        B_neu[i, 80 + i * 2 : 82 + i * 2] -= 3.5
    for i in range(10, 18):
        A_neu[i, 180 + (i - 10)] += 3.0
        B_neu[i, 180 + (i - 10)] += 3.0
    for c in range(25):
        A_iso[c, 180 + (c % 8)] += 4.5
        B_iso[c, 180 + (c % 8)] += 4.5

    cov_col = np.column_stack([np.ones(C_col), rng.uniform(-0.5, 0.5, size=(C_col, 4))])
    cov_iso = np.column_stack([np.ones(C_iso), rng.uniform(-0.5, 0.5, size=(C_iso, 4))])
    cov_neu = np.column_stack([np.ones(N), rng.uniform(-0.5, 0.5, size=(N, 4))])

    model_col = fo.fit_control_model(A_col, B_col, cov_col, mask, False, dev)
    model_iso = fo.fit_control_model(A_iso, B_iso, cov_iso, mask, False, dev)

    # Stream in 2 chunks of 20 and compare against monolithic score_odd_pair_gpu
    sc_col_c = [fo.score_chunk_against_controls(A_neu[a:b], B_neu[a:b], cov_neu[a:b], model_col, mask, dev, keep_words=True, top_k=5) for a, b in ((0, 20), (20, 40))]
    sc_iso_c = [fo.score_chunk_against_controls(A_neu[a:b], B_neu[a:b], cov_neu[a:b], model_iso, mask, dev, keep_words=False) for a, b in ((0, 20), (20, 40))]

    T_col = np.concatenate([x["T"] for x in sc_col_c])
    T_iso = np.concatenate([x["T"] for x in sc_iso_c])
    r_ab_c = np.concatenate([x["r_ab_res"] for x in sc_col_c])
    r_ba_c = np.concatenate([x["r_ba_res"] for x in sc_col_c])
    r_ab_i = np.concatenate([x["r_ab_res"] for x in sc_iso_c])
    r_ba_i = np.concatenate([x["r_ba_res"] for x in sc_iso_c])

    mono_col = fo.score_odd_pair_gpu(torch.cat([A_col, A_neu]), torch.cat([B_col, B_neu]), np.arange(C_col + N) < C_col, np.vstack([cov_col, cov_neu]), mask, False, dev, keep_words=True)
    diff_t = float(np.max(np.abs(T_col - mono_col["T"][C_col:])))
    assert diff_t < 1e-4, f"Dual chunk mismatch: {diff_t}"

    sel_col = fo.empirical_fdp(T_col, (r_ab_c > 0) & (r_ba_c > 0), N, model_col.T_ctl, 0.05)
    sel_iso = fo.empirical_fdp(T_iso, (r_ab_i > 0) & (r_ba_i > 0), N, model_iso.T_ctl, 0.05)
    dual_set = set(sel_col["named_idx"].tolist()) & set(sel_iso["named_idx"].tolist())
    assert set(range(10)).issubset(dual_set) and not (dual_set & set(range(10, 18))), f"Unexpected dual set: col={sel_col['named_idx']}, iso={sel_iso['named_idx']}"

    # Pure null neuron matrix must produce 0 dual discoveries
    A_null = torch.from_numpy(rng.normal(0.0, 0.4, size=(N, V)).astype(np.float16))
    B_null = torch.from_numpy(rng.normal(0.0, 0.4, size=(N, V)).astype(np.float16))
    sc_nc = fo.score_chunk_against_controls(A_null, B_null, cov_neu, model_col, mask, dev)
    sc_ni = fo.score_chunk_against_controls(A_null, B_null, cov_neu, model_iso, mask, dev)
    n_dual_null = len(
        set(fo.empirical_fdp(sc_nc["T"], (sc_nc["r_ab_res"] > 0) & (sc_nc["r_ba_res"] > 0), N, model_col.T_ctl, 0.05)["named_idx"].tolist())
        & set(fo.empirical_fdp(sc_ni["T"], (sc_ni["r_ab_res"] > 0) & (sc_ni["r_ba_res"] > 0), N, model_iso.T_ctl, 0.05)["named_idx"].tolist())
    )
    assert n_dual_null == 0, f"Pure null yielded false dual discoveries: {n_dual_null}"
    print(f"[PASS] dual-null streaming assembly (col={sel_col['count']}, iso={sel_iso['count']}, dual={len(dual_set)}, null_dual={n_dual_null})")


def main() -> None:
    test_mean_centering_invariance()
    test_temperature_projection_invariance()
    test_streaming_exactness()
    test_fdp_and_ladder()
    test_weight_word_streaming_exactness()
    test_dual_null_assembly()
    print("\nAll forced_odd CPU checks passed.")


if __name__ == "__main__":
    main()
