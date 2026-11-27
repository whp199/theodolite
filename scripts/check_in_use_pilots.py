#!/usr/bin/env python3
"""CPU verification tests for the exploratory on-manifold ('in-use') pilot scripts.

Verifies without GPU:
  1. Strict <eos> document-half disjointness in collect_face_events and build_event_sequence.
  2. Vectorized layer_face_counts_vectorized matches per-face collect_face_events bit-for-bit.
  3. 'columns' control directions match ||w_post * W_down[:, n]||_2 within 1e-5 relative error.
  4. LOO centering and score_directional_carry chance calibration under null noise.
  5. Prompt construction, delimiter parsing (_parse_behind_endings), and blind scoring math.
  6. Recorded JSON outputs (when present) have valid provenance, git_dirty=False, and zero raw corpus leaks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import input_census as ic  # noqa: E402
from scripts.in_use_common import (  # noqa: E402
    build_columns_control_dirs,
    build_event_sequence,
    collect_face_events,
    emp_p_values,
    layer_face_counts_vectorized,
    loo_center,
    score_directional_carry,
)
from scripts.in_use_pattern_pilot import _evaluate_ab_scores, _parse_behind_endings  # noqa: E402


def test_doc_halves_and_vectorized_counts() -> None:
    rng = np.random.default_rng(7)
    W, seq, N = 24, 32, 6
    windows = rng.integers(10, 500, size=(W, seq), dtype=np.int32)
    windows[:, 0] = 2
    # Add <eos>=1 boundaries every ~20 tokens across concatenated corpus
    for w in (2, 5, 9, 14, 19, 22):
        windows[w, 12] = 1
    doc_id, half_id, n_docs = ic.build_doc_halves(windows, eos_id=1, seed=42)
    pure_h0 = np.all(half_id == 0, axis=1)
    pure_h1 = np.all(half_id == 1, axis=1)

    act_scale = {"pos": np.full((1, N), 2.0, np.float32), "neg": np.full((1, N), 2.0, np.float32)}
    pos_win = rng.integers(0, W, size=(1, N, 8), dtype=np.int32)
    pos_at = rng.integers(1, seq - 4, size=(1, N, 8), dtype=np.int32)
    pos_act = rng.uniform(0.5, 2.0, size=(1, N, 8)).astype(np.float32)
    # Inject a duplicate window and a weak slot below 0.5*ext
    pos_win[0, 0, 1] = pos_win[0, 0, 0]
    pos_act[0, 0, 2] = 0.2
    neg_win = rng.integers(0, W, size=(1, N, 8), dtype=np.int32)
    neg_at = rng.integers(1, seq - 4, size=(1, N, 8), dtype=np.int32)
    neg_act = -rng.uniform(0.5, 2.0, size=(1, N, 8)).astype(np.float32)

    P = {
        "pos_win": pos_win,
        "pos_at": pos_at,
        "pos_act": pos_act,
        "neg_win": neg_win,
        "neg_at": neg_at,
        "neg_act": neg_act,
    }
    cnt = layer_face_counts_vectorized(P, windows, act_scale, half_id, pure_h0, pure_h1, 0, eos_id=1, bos_id=2)
    for n in range(N):
        for face in ("pos", "neg"):
            ev = collect_face_events(P, windows, act_scale, pure_h0, pure_h1, doc_id, half_id, 0, n, face, eos_id=1, bos_id=2)
            assert len(ev["pure_h0"]) == int(cnt[f"{face}_pure0"][n])
            assert len(ev["pure_h1"]) == int(cnt[f"{face}_pure1"][n])
            assert len(ev["doc_h0"]) == int(cnt[f"{face}_doc0"][n])
            assert len(ev["doc_h1"]) == int(cnt[f"{face}_doc1"][n])
            for e in ev["pure_h0"] + ev["pure_h1"]:
                sq, pk = build_event_sequence(windows, e["win"], e["at"], bos_id=2, pad_tail=3)
                assert sq[0] == 2 and sq[pk] == int(windows[e["win"], e["at"]])


def test_columns_control_norm_match() -> None:
    torch.manual_seed(0)
    D, N_ff = 64, 512
    w_down = torch.randn(D, N_ff, dtype=torch.float32)
    w_post = torch.rand(D, dtype=torch.float32) + 0.1
    tgts = [3, 17, 128, 300]
    ctl = build_columns_control_dirs(w_down, w_post, tgts, seed=99, mix_k=64)
    assert ctl.shape == (4, D)
    tgt_norms = (w_down[:, tgts].T * w_post[None, :]).norm(dim=-1)
    ctl_norms = (ctl * w_post[None, :]).norm(dim=-1)
    rel_err = float(((tgt_norms - ctl_norms).abs() / tgt_norms.clamp(min=1e-8)).max().item())
    assert rel_err < 1e-5, f"Norm mismatch rel_err={rel_err}"


def test_loo_and_null_carry() -> None:
    rng = np.random.default_rng(123)
    N, C, V = 64, 64, 128
    # Null data with shared background drift across all units
    drift_f = rng.normal(0.0, 0.2, size=(1, V))
    drift_t = rng.normal(0.0, 0.2, size=(1, V))
    fit_neu = drift_f + rng.normal(0.0, 0.05, size=(N, V))
    fit_ctl = drift_f + rng.normal(0.0, 0.05, size=(C, V))
    test_neu = drift_t + rng.normal(0.0, 0.05, size=(N, V))
    test_ctl = drift_t + rng.normal(0.0, 0.05, size=(C, V))

    c_n, c_c = loo_center(fit_neu, fit_ctl)
    assert np.allclose(c_c.mean(axis=0), 0.0, atol=1e-5)
    assert np.abs(c_n.mean()) < 0.05

    res_null = score_directional_carry(fit_neu, fit_ctl, test_neu, test_ctl, top_k=5)
    assert 0.35 <= res_null["mean"]["auc"] <= 0.65, f"Null AUC out of range: {res_null['mean']['auc']}"
    assert res_null["mean"]["n_emp_p05"] <= 10

    # Planted signal on first 16 neurons: sparse ~25% signal -> AUC ~0.625, n_emp_p05 >= 14
    fit_sig = fit_neu.copy()
    test_sig = test_neu.copy()
    fit_sig[:16, :5] += 0.3
    test_sig[:16, :5] += 0.3
    res_sig = score_directional_carry(fit_sig, fit_ctl, test_sig, test_ctl, top_k=5)
    assert res_sig["mean"]["auc"] > 0.59
    assert res_sig["mean"]["n_emp_p05"] >= 14

    # Broad signal on first 32 neurons -> AUC > 0.72
    fit_broad = fit_neu.copy()
    test_broad = test_neu.copy()
    fit_broad[:32, :5] += 0.3
    test_broad[:32, :5] += 0.3
    res_broad = score_directional_carry(fit_broad, fit_ctl, test_broad, test_ctl, top_k=5)
    assert res_broad["mean"]["auc"] > 0.72
    assert res_broad["mean"]["n_emp_p05"] >= 28

    ep = emp_p_values(np.arange(10, dtype=np.float64), np.zeros(20, dtype=np.float64))
    assert abs(ep["auc"] - 0.95) < 1e-6, f"Expected 0.95 tie-aware AUC, got {ep['auc']}"


def test_behind_parsing_and_ab_eval() -> None:
    raw = (
        "FIRE 1: <<< plays a crucial role>>>\n"
        "FIRE 2: <<< plays a vital part>>>\n"
        "QUIET 1: <<< plays the piano>>>\n"
        "QUIET 2: <<< has a red bicycle>>>\n"
    )
    f, q = _parse_behind_endings(raw)
    assert f == [" plays a crucial role", " plays a vital part"]
    assert q == [" plays the piano", " has a red bicycle"]

    items = [
        {"face_key": "L2-n1p", "layer": 2, "truth": "a", "match": "phrase"},
        {"face_key": "L2-n1p", "layer": 2, "truth": "b", "match": "phrase"},
        {"face_key": "L2-n1p", "layer": 2, "truth": "a", "match": "before"},
        {"face_key": "L2-n1p", "layer": 2, "truth": "b", "match": "word"},
    ]
    own = [{"pick": x["truth"], "p_ab": 0.98} for x in items]
    oth = [{"pick": "a", "p_ab": 0.95} for _ in items]
    ev = _evaluate_ab_scores(items, own, oth)
    assert ev["own_share"] == 1.0 and ev["other_share"] == 0.5


def test_recorded_outputs_if_present() -> None:
    out_dir = REPO_ROOT / "experiments/results/in_use_kill_pilots"
    for fname in ("probe2_cpu_context.json", "probe1_in_use_causal.json", "probe3_local_pattern.json"):
        p = out_dir / fname
        if not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8")
        obj = json.loads(raw)
        assert obj.get("git_dirty") is False, f"{fname} recorded while git_dirty={obj.get('git_dirty')}"
        assert obj.get("code_sha256"), f"{fname} missing code_sha256"
        assert "kept_tail" not in raw and "fit_snippets" not in raw, f"{fname} leaked corpus snippets"


def main() -> None:
    tests = [
        test_doc_halves_and_vectorized_counts,
        test_columns_control_norm_match,
        test_loo_and_null_carry,
        test_behind_parsing_and_ab_eval,
        test_recorded_outputs_if_present,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(tests)} checks passed.")


if __name__ == "__main__":
    main()
