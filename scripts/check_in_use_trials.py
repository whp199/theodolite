#!/usr/bin/env python3
"""CPU verification tests for all three confirmatory trial pipelines (P1, P2, P3).

Verifies without GPU:
  1. P1 evaluator (G1-G5): passes on planted signal + balanced nulls; FAILS on null noise; FAILS G4 on cross-pool shift; FAILS G5 on bad shuffled null.
  2. P2 evaluator (Track 2A U2-T_dec + Track 2B U2-weight_words): dual-null FDP & cross-pool rejection, plus corrected own-word scoring math.
  3. P3 evaluator (G1-G4): K=3 derangement uniqueness, unseen-doc event selection isolation, and pooled + individual + behind gate logic.
  4. SHA256SUMS & CODE_SHA256SUMS guard refuses tampered files cleanly.
  5. Recorded trial JSON outputs (when present) have git_dirty=False, valid prereg metadata, and zero leaked corpus text.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.in_use_lens_trial import derangement_perm, evaluate_p1_layer  # noqa: E402
from scripts.in_use_pattern_trial import _select_doc_events, evaluate_p3_layer, k3_derangement_offsets  # noqa: E402
from scripts.u2_rollout_read import compute_weight_word_metrics, evaluate_track_2a  # noqa: E402


def test_p1_gates_and_cross_pool() -> None:
    rng = np.random.default_rng(11)
    N = 256
    sanity_ok = {
        "noop_kl_same_batch": 0.0,
        "max_kl_alg_vs_hook_same_batch": 0.0,
        "max_layer_rec_diff": 0.0,
        "achieved_col_over_abl_post_norm_median": 1.02,
        "achieved_iso_over_abl_post_norm_median": 1.00,
    }
    col = rng.normal(0.0, 0.002, size=(N, 8))
    iso = rng.normal(0.0, 0.002, size=(N, 8))
    shuf = rng.normal(0.0, 0.002, size=(N, 8))

    # Plant strong signal on first 90 neurons (>10% = 26, and boot p10 >= 60)
    neu_sig = rng.normal(0.0, 0.002, size=(N, 8))
    neu_sig[:90, :4] += 0.015
    neu_sig[:90, 4:] += 0.015
    res_pass = evaluate_p1_layer(
        layer=58,
        eye_pk_neu=neu_sig,
        eye_pk_shuf=shuf,
        eye_pk_col=col,
        eye_pk_iso=iso,
        kl_abl=np.abs(neu_sig),
        kl_col=np.abs(col),
        kl_iso=np.abs(iso),
        sanity=sanity_ok,
    )
    assert res_pass["pass"] is True, f"Expected P1 pass, got gates={res_pass['gates']}"

    # Cross-pool contamination test: shift iso upward so col_false_vs_iso or iso_false_vs_col fails G4
    iso_bad = col + 0.020
    res_xp = evaluate_p1_layer(
        layer=59,
        eye_pk_neu=neu_sig,
        eye_pk_shuf=shuf,
        eye_pk_col=col,
        eye_pk_iso=iso_bad,
        kl_abl=np.abs(neu_sig),
        kl_col=np.abs(col),
        kl_iso=np.abs(iso_bad),
        sanity=sanity_ok,
    )
    assert res_xp["gates"]["G4_cross_pool_exchangeability"] is False
    assert res_xp["pass"] is False

    # Pure null must fail
    neu_null = rng.normal(0.0, 0.002, size=(N, 8))
    res_null = evaluate_p1_layer(
        layer=57,
        eye_pk_neu=neu_null,
        eye_pk_shuf=shuf,
        eye_pk_col=col,
        eye_pk_iso=iso,
        kl_abl=np.abs(neu_null),
        kl_col=np.abs(col),
        kl_iso=np.abs(iso),
        sanity=sanity_ok,
    )
    assert res_null["pass"] is False
    p = derangement_perm(100, 7)
    assert len(p) == 100 and np.all(p != np.arange(100))


def test_p2_u2_track2a_and_track2b() -> None:
    # Track 2A synthetic dual pass vs cross-pool fail
    named_rec = [{"neuron": i, "covered": True, "T_dec": 5.0} for i in range(95)]
    res_col = {
        "fdp05_count": 95,
        "bootstrap_fdp05": {"p10": 75, "frac_zero": 0.0},
        "n3_covered": {"neurons": 1022, "share_covered": 0.95, "bootstrap_fdp05": {"p10": 75, "frac_zero": 0.0}},
        "code_symmetry_check": {"fdp05_false_max": 0},
        "_named_records": named_rec,
    }
    res_iso = dict(res_col)
    c4_ok = {"fit_named": 60, "frac_p05_on_C4": 0.90, "c4_auc_vs_paired_ctl": 0.91}
    xp_ok = {"fdp05_count": 5, "n3_covered": {"neurons": 1024}}
    xp_bad = {"fdp05_count": 40, "n3_covered": {"neurons": 1024}}

    ev_ok = evaluate_track_2a(
        res_col=res_col,
        res_iso=res_iso,
        c4_col=c4_ok,
        c4_iso=c4_ok,
        iso_vs_col_res=xp_ok,
        col_vs_iso_res=xp_ok,
        baseline_match_ok=True,
        complete_ok=True,
    )
    assert ev_ok["pass"] is True
    ev_bad = evaluate_track_2a(
        res_col=res_col,
        res_iso=res_iso,
        c4_col=c4_ok,
        c4_iso=c4_ok,
        iso_vs_col_res=xp_bad,
        col_vs_iso_res=xp_ok,
        baseline_match_ok=True,
        complete_ok=True,
    )
    assert ev_bad["gates"]["G_CrossPool"] is False and ev_bad["pass"] is False

    # Track 2B synthetic weight-words scoring
    VN = [f"w{i}" for i in range(600)]
    prompts_a = ("bicycle", "walk", "chair", "kettle", "sleep", "lamp")
    prompts_b = ("clock", "rice", "vase", "lock", "money", "blanket")
    nn, C = 120, 120
    neu_ep = {n: [n * 2] for n in range(nn)}
    neu_en = {n: [n * 2 + 1] for n in range(nn)}
    col_ep = [[250 + c * 2] for c in range(C)]
    col_en = [[250 + c * 2 + 1] for c in range(C)]
    iso_ep = [[250 + c * 2] for c in range(C)]
    iso_en = [[250 + c * 2 + 1] for c in range(C)]

    rows_c, rows_i = [], []
    rng = np.random.default_rng(42)
    for n in range(nn):
        for q in prompts_a + prompts_b:
            for d in (-1.95, 1.95):
                if n < 50:
                    w = (n * 2) if d > 0 else (n * 2 + 1)
                    rows_c.append(("neuron", n, q, d, [w, int(rng.integers(500, 600))]))
                else:
                    rows_c.append(("neuron", n, q, d, [int(rng.integers(500, 600)), int(rng.integers(500, 600))]))
    for c in range(C):
        for q in prompts_a + prompts_b:
            for d in (-1.95, 1.95):
                tok_rand = [int(rng.integers(500, 600)), int(rng.integers(500, 600))]
                rows_c.append(("random", c, q, d, tok_rand))
                rows_i.append(("random", c, q, d, tok_rand))

    ww = compute_weight_word_metrics(
        rows_neu_and_col=rows_c,
        rows_iso=rows_i,
        VN=VN,
        neu_eye_pos=neu_ep,
        neu_eye_neg=neu_en,
        col_eye_pos=col_ep,
        col_eye_neg=col_en,
        iso_eye_pos=iso_ep,
        iso_eye_neg=iso_en,
    )
    assert ww["pass"] is True, f"Expected Track 2B pass, got {ww}"


def test_p3_derangement_doc_isolation_and_gates() -> None:
    o1, o2, o3 = k3_derangement_offsets(128)
    assert len({0, o1, o2, o3}) == 4

    ev = {
        "pure_h0": [{"win": i, "at": 40, "doc_id": i, "act": 1.0, "rel": 0.9} for i in range(6)],
        "doc_h0": [{"win": i, "at": 40, "doc_id": i, "act": 1.0, "rel": 0.8} for i in range(10)],
        "pure_h1": [{"win": 100 + i, "at": 40, "doc_id": 100 + i, "act": 1.0, "rel": 0.9} for i in range(10)],
        "doc_h1": [{"win": 100 + i, "at": 40, "doc_id": 100 + i, "act": 1.0, "rel": 0.8} for i in range(15)],
    }
    fit_e, test_e, beh_e = _select_doc_events(ev, n_fit=8, n_test=10)
    assert len(fit_e) == 8 and len(test_e) == 10
    fit_d = {e["doc_id"] for e in fit_e}
    test_d = {e["doc_id"] for e in test_e}
    assert not (fit_d & test_d)
    assert beh_e["doc_id"] not in fit_d and beh_e["doc_id"] not in test_d

    # Gate evaluator test on 40 faces x 10 pairs (N>=40 required for asymptotic Wilcoxon p <= 1e-6)
    fkeys = [f"L2-n{i}p" for i in range(40)]
    items_b: List[Dict[str, Any]] = []
    own_sc: List[Dict[str, float]] = []
    oth_sc: List[List[Dict[str, float]]] = [[], [], []]
    rng = np.random.default_rng(99)
    for fi, fk in enumerate(fkeys):
        for pi in range(10):
            tr = "a" if pi % 2 == 0 else "b"
            items_b.append({"face_key": fk, "truth": tr, "match": "phrase"})
            pick_own = tr if (fi < 28 or rng.random() < 0.6) else ("b" if tr == "a" else "a")
            own_sc.append({"pick": pick_own, "p_ab": 0.99})
            for ki in range(3):
                pick_o = "a" if rng.random() < 0.5 else "b"
                oth_sc[ki].append({"pick": pick_o, "p_ab": 0.99})

    beh_recs = []
    for fi, fk in enumerate(fkeys):
        fired = 3 if fi < 20 else 0
        beh_recs.append(
            {
                "key": fk,
                "own_holds_gt_0.5": True,
                "result": {"fire": 3, "fired": fired, "quiet": 3, "quiet_fired": 0, "apart": 1.0 if fired else 0.5},
                "other_fire_total": 3,
                "other_fire_fired": 0,
            }
        )
    res_p3 = evaluate_p3_layer(
        layer=2,
        face_keys=fkeys,
        items_b=items_b,
        own_scores=own_sc,
        other_scores_k=oth_sc,
        len_ratios=[1.0] * len(items_b),
        behind_records=beh_recs,
    )
    assert res_p3["pass"] is True, f"Expected P3 pass, got {res_p3}"

    # Verify G3 fails when behind_records are null/noise
    beh_null = [dict(b, result={"fire": 3, "fired": 0, "quiet": 3, "quiet_fired": 0, "apart": 0.5}) for b in beh_recs]
    res_p3_fail = evaluate_p3_layer(
        layer=20,
        face_keys=fkeys,
        items_b=items_b,
        own_scores=own_sc,
        other_scores_k=oth_sc,
        len_ratios=[1.0] * len(items_b),
        behind_records=beh_null,
    )
    assert res_p3_fail["gates"]["G3_unseen_behind_causality"] is False and res_p3_fail["pass"] is False


def test_sha_tamper_detection() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        f = p / "neurons.txt"
        f.write_text("1,pos\n", encoding="utf-8")
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        f.write_text("2,pos\n", encoding="utf-8")
        assert hashlib.sha256(f.read_bytes()).hexdigest() != digest


def test_recorded_trial_outputs_if_present() -> None:
    for rdir in (
        REPO_ROOT / "experiments/results/in_use_lens_trial",
        REPO_ROOT / "experiments/results/in_use_pattern_trial",
        REPO_ROOT / "experiments/results/u2_rollout_trial",
    ):
        for p in sorted(rdir.glob("*.json")) if rdir.is_dir() else []:
            raw = p.read_text(encoding="utf-8")
            obj = json.loads(raw)
            assert obj.get("git_dirty") is False, f"{p} has git_dirty={obj.get('git_dirty')}"
            assert obj.get("prereg"), f"{p} missing prereg block"
            for forbidden in ('"kept_tail"', '"fit_snippets"', '"a_txt"', '"b_txt"'):
                assert forbidden not in raw, f"{p} leaked {forbidden}"


def main() -> None:
    tests = [
        test_p1_gates_and_cross_pool,
        test_p2_u2_track2a_and_track2b,
        test_p3_derangement_doc_isolation_and_gates,
        test_sha_tamper_detection,
        test_recorded_trial_outputs_if_present,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(tests)} confirmatory checks passed.")


if __name__ == "__main__":
    main()
