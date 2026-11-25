#!/usr/bin/env python3
"""CPU unit tests for backend/input_census.py."""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path("/home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from backend import input_census as ic  # noqa: E402


def test_input_census_synthetic() -> None:
    rng = np.random.default_rng(123)
    W, seq, V = 680, 32, 500
    N_ff, K, S = 225, 16, 6

    # Background tokens concentrated in 10..200; special trigger tokens 250..499 are rare (~1e-4)
    probs = np.zeros(V, dtype=np.float64)
    probs[10:200] = 1.0
    probs[200:500] = 0.05
    probs /= probs.sum()

    windows = rng.choice(np.arange(10, 500), size=(W, seq), p=probs[10:] / probs[10:].sum()).astype(np.int32)
    windows[:, 0] = 2  # BOS mark at index 0
    unigram = np.bincount(windows[:, 1:].reshape(-1), minlength=V).astype(np.int64)

    pos_win = np.full((N_ff, K), -1, dtype=np.int32)
    pos_at = np.full((N_ff, K), -1, dtype=np.int32)
    neg_win = np.full((N_ff, K), -1, dtype=np.int32)
    neg_at = np.full((N_ff, K), -1, dtype=np.int32)
    pos_smp_win = np.full((N_ff, S), -1, dtype=np.int32)
    pos_smp_at = np.full((N_ff, S), -1, dtype=np.int32)
    neg_smp_win = np.full((N_ff, S), -1, dtype=np.int32)
    neg_smp_at = np.full((N_ff, S), -1, dtype=np.int32)

    even_pool = np.arange(0, W, 2)
    odd_pool = np.arange(1, W, 2)

    # Plant 30 strong positive-face neurons (10 @0, 10 @-1, 10 @+1) on disjoint 16-window blocks
    for n in range(30):
        ws = np.arange(16 * n, 16 * (n + 1), dtype=np.int32)  # 8 even, 8 odd
        ats = rng.integers(3, 18, size=16)
        tok = 300 + n
        off_mode = n % 3  # 0->0, 1->-1, 2->+1
        for j in range(16):
            w0, a0 = int(ws[j]), int(ats[j])
            if off_mode == 0:
                windows[w0, a0] = tok
            elif off_mode == 1:
                windows[w0, a0 - 1] = tok
            else:
                windows[w0, a0 + 1] = tok
            # Also plant a bigram partner on neuron 0 (with extra background hits so tok=300 has strictly higher lift)
            if n == 0 and off_mode == 0:
                windows[w0, a0 - 1] = 280
        if n == 0:
            windows[600:615, 29] = 280
        pos_win[n] = ws
        pos_at[n] = ats

        # Held-out mid-range samples: 2 overlapping with top-16 (MUST be excluded), 4 disjoint fresh windows (3 matching, 1 miss)
        base_f = 480 + 4 * n
        smp_w = [int(ws[0]), int(ws[1]), base_f, base_f + 1, base_f + 2, base_f + 3]
        smp_a = [int(ats[0]), int(ats[1]), 22, 24, 26, 28]
        for k_s in range(2, 5):  # plant trigger in 3 of the 4 fresh sample windows
            if off_mode == 0:
                windows[smp_w[k_s], smp_a[k_s]] = tok
            elif off_mode == 1:
                windows[smp_w[k_s], smp_a[k_s] - 1] = tok
            else:
                windows[smp_w[k_s], smp_a[k_s] + 1] = tok
        # Ensure fresh[3] doesn't match by accident
        windows[smp_w[5], smp_a[5] - 1 : smp_a[5] + 2] = 15
        pos_smp_win[n] = smp_w
        pos_smp_at[n] = smp_a

    # Plant BOS trap at neuron 31: a0 = 1 where index 0 is BOS (tok=2) -> must NOT count BOS as offset -1
    ev31 = rng.choice(even_pool, size=8, replace=False)
    od31 = rng.choice(odd_pool, size=8, replace=False)
    pos_win[31] = np.concatenate([ev31, od31])
    pos_at[31] = 1
    # Random pure noise faces for remaining neurons
    for n in range(32, N_ff):
        pos_win[n] = np.concatenate([rng.choice(even_pool, 8, replace=False), rng.choice(odd_pool, 8, replace=False)])
        pos_at[n] = rng.integers(3, seq - 3, size=16)
    for n in range(N_ff):
        neg_win[n] = np.concatenate([rng.choice(even_pool, 8, replace=False), rng.choice(odd_pool, 8, replace=False)])
        neg_at[n] = rng.integers(3, seq - 3, size=16)

    # Recompute unigram after planting
    unigram = np.bincount(windows[:, 1:].reshape(-1), minlength=V).astype(np.int64)

    res = ic.analyze_layer_arrays(
        windows=windows,
        unigram=unigram,
        pos_win=pos_win,
        pos_at=pos_at,
        neg_win=neg_win,
        neg_at=neg_at,
        pos_smp_win=pos_smp_win,
        pos_smp_at=pos_smp_at,
        neg_smp_win=neg_smp_win,
        neg_smp_at=neg_smp_at,
        n_null=512,
        n_null_splits=20,
        seed=99,
    )

    planted_named = int(res["named_fdp05"][0, :30].sum())
    assert planted_named >= 26, f"Expected >=26/30 planted faces named, got {planted_named}"
    # Verify exact recovered token & offset on planted neurons
    exact_hits = 0
    for n in range(30):
        exp_tok = 300 + n
        exp_off = [0, -1, 1][n % 3]
        if int(res["top_token_id"][0, n, 0]) == exp_tok and int(res["top_offset"][0, n, 0]) == exp_off:
            exact_hits += 1
        assert int(res["smp_total"][0, n]) == 4, f"top-16 overlap not excluded at n={n}: total={res['smp_total'][0, n]}"
        assert int(res["smp_hits"][0, n]) == 3, f"expected 3/4 smp hits at n={n}: hits={res['smp_hits'][0, n]}"
    assert exact_hits >= 26, f"Expected >=26/30 exact token+offset matches, got {exact_hits}"
    assert bool(res["bigram_rep"][0, 0]), "Expected bigram replication on planted neuron 0"

    # Verify BOS trap (neuron 31): offset=-1 must never match BOS token 2
    for k in range(5):
        if int(res["top_token_id"][0, 31, k]) == 2:
            assert int(res["top_offset"][0, 31, k]) != -1, "BOS at index 0 leaked into offset=-1!"

    # Random neg faces should have very low false rate
    neg_false = int(res["named_fdp05"][1].sum())
    assert neg_false <= 3, f"Too many false discoveries on pure random neg faces: {neg_false}"
    assert res["stats"]["null_split_false_mean"] <= 1.0, f"Null split mean too high: {res['stats']}"

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "L00_input_census.npz"
        ic.save_layer_input_census(p, res)
        loaded = np.load(p)
        assert "windows" not in loaded.files
        assert np.array_equal(loaded["named_fdp05"], res["named_fdp05"])
        assert np.array_equal(loaded["top_token_id"], res["top_token_id"])
    print(f"[PASS] input_census synthetic checks (planted={planted_named}/30, exact={exact_hits}/30, neg_false={neg_false}, null_mean={res['stats']['null_split_false_mean']})")


if __name__ == "__main__":
    test_input_census_synthetic()
    print("\nAll input_census CPU checks passed.")
