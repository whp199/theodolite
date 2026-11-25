#!/usr/bin/env python3
"""CPU unit tests for backend/input_census.py (document-disjoint, causal offsets, nulls)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import input_census as ic  # noqa: E402
from scripts import publish_gap_census as pgc  # noqa: E402


def test_input_census_synthetic() -> None:
    rng = np.random.default_rng(123)
    W, seq, V = 700, 32, 500
    N_ff, K, S = 225, 16, 6

    probs = np.zeros(V, dtype=np.float64)
    probs[10:200] = 1.0
    probs[200:500] = 0.05
    probs /= probs.sum()

    windows = rng.choice(np.arange(10, 500), size=(W, seq), p=probs[10:] / probs[10:].sum()).astype(np.int32)
    windows[:, 0] = 2  # BOS at pos 0
    # Put <eos> (id=1) at the end of every window BY DEFAULT, EXCEPT straddling pairs below.
    windows[:, seq - 1] = 1

    # Find doc halves upfront so synthetic planting places true cross-doc triggers across BOTH halves!
    # Also test a straddling document pair: windows 640 & 641 have NO <eos> between them!
    windows[640, seq - 1] = 25
    doc_id, half_id, n_docs = ic.build_doc_halves(windows, eos_id=1, seed=20261124)
    assert int(doc_id[640, 10]) == int(doc_id[641, 10]), "Straddling pair without <eos> must share doc_id"
    assert int(half_id[640, 10]) == int(half_id[641, 10]), "Straddling doc must stay in ONE half"

    h0_wins = [w for w in range(480) if int(half_id[w, 10]) == 0]
    h1_wins = [w for w in range(480) if int(half_id[w, 10]) == 1]
    assert len(h0_wins) >= 180 and len(h1_wins) >= 180

    pos_win = np.full((N_ff, K), -1, dtype=np.int32)
    pos_at = np.full((N_ff, K), -1, dtype=np.int32)
    neg_win = np.full((N_ff, K), -1, dtype=np.int32)
    neg_at = np.full((N_ff, K), -1, dtype=np.int32)
    pos_smp_win = np.full((N_ff, S), -1, dtype=np.int32)
    pos_smp_at = np.full((N_ff, S), -1, dtype=np.int32)
    neg_smp_win = np.full((N_ff, S), -1, dtype=np.int32)
    neg_smp_at = np.full((N_ff, S), -1, dtype=np.int32)

    # Plant 20 strong causal faces (10 @offset 0, 10 @offset -1) and 10 @offset +1 ONLY (next piece, must NOT be trigger!)
    for n in range(30):
        w_e = h0_wins[8 * (n % 20) : 8 * ((n % 20) + 1)]
        w_o = h1_wins[8 * (n % 20) : 8 * ((n % 20) + 1)]
        ws = np.array(w_e + w_o, dtype=np.int32)
        ats = rng.integers(3, 18, size=16)
        tok = 300 + n
        off_mode = 0 if n < 10 else (-1 if n < 20 else 1)
        for j in range(16):
            w0, a0 = int(ws[j]), int(ats[j])
            if off_mode == 0:
                windows[w0, a0] = tok
            elif off_mode == -1:
                windows[w0, a0 - 1] = tok
            else:
                windows[w0, a0 + 1] = tok
            if n == 0:
                windows[w0, a0 - 1] = 280
        if n == 0:
            windows[600:615, 29] = 280
        pos_win[n] = ws
        pos_at[n] = ats

        # Held-out: 2 from same document as top-16 (MUST be excluded), 4 from unseen documents (3 match, 1 miss)
        fresh_ws = [490 + 4 * (n % 25) + k for k in range(4)]
        smp_w = [int(ws[0]), int(ws[1])] + fresh_ws
        smp_a = [int(ats[0]), int(ats[1]), 22, 24, 26, 28]
        for k_s in range(2, 5):
            if off_mode == 0:
                windows[smp_w[k_s], smp_a[k_s]] = tok
            elif off_mode == -1:
                windows[smp_w[k_s], smp_a[k_s] - 1] = tok
            else:
                windows[smp_w[k_s], smp_a[k_s] + 1] = tok
        windows[smp_w[5], smp_a[5] - 1 : smp_a[5] + 2] = 15
        pos_smp_win[n] = smp_w
        pos_smp_at[n] = smp_a

    # Neuron 30: straddling single-document artifact! Trigger planted ONLY in docs straddling windows 640/641
    # (which would have split 8+8 under even/odd window parity, but must stay single-half under doc split -> 0/uneligible!)
    w640_pos = np.full(8, 640, dtype=np.int32)
    w641_pos = np.full(8, 641, dtype=np.int32)
    for j in range(8):
        windows[640, 4 + 3 * j] = 450
        windows[641, 4 + 3 * j] = 450
    pos_win[30] = np.concatenate([w640_pos, w641_pos])
    pos_at[30] = np.concatenate([4 + 3 * np.arange(8), 4 + 3 * np.arange(8)]).astype(np.int32)

    # BOS trap at neuron 31: a0 = 1 where pos 0 is BOS (tok=2) -> must NOT count BOS at offset -1
    pos_win[31] = np.array(h0_wins[-8:] + h1_wins[-8:], dtype=np.int32)
    pos_at[31] = 1

    # Pure random noise faces
    for n in range(32, N_ff):
        pos_win[n] = rng.choice(W, size=16, replace=False)
        pos_at[n] = rng.integers(3, seq - 3, size=16)
    for n in range(N_ff):
        neg_win[n] = rng.choice(W, size=16, replace=False)
        neg_at[n] = rng.integers(3, seq - 3, size=16)

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
        doc_split_seed=20261124,
        eos_id=1,
        bos_id=2,
    )

    # 1. All 20 causal planted neurons (0..19) must recover exact tok and offset (-1 or 0)
    exact_causal = 0
    for n in range(20):
        exp_tok = 300 + n
        exp_off = 0 if n < 10 else -1
        if int(res["top_token_id"][0, n, 0]) == exp_tok and int(res["top_offset"][0, n, 0]) == exp_off:
            exact_causal += 1
        assert int(res["smp_total"][0, n]) == 4, f"same-doc smp not excluded at n={n}: total={res['smp_total'][0, n]}"
        assert int(res["smp_hits"][0, n]) == 3, f"expected 3/4 smp hits at n={n}: hits={res['smp_hits'][0, n]}"
    assert exact_causal >= 18, f"Expected >=18/20 causal exact matches, got {exact_causal}"
    assert bool(res["bigram_rep"][0, 0]), "Expected bigram replication (-1, 0) on neuron 0"

    # 2. +1 planted tokens (neurons 20..29) MUST NEVER enter top_offset / triggers and MUST appear in next_token_id
    assert not np.any(res["top_offset"] > 0) and not np.any(res["top_offset"] < -1), "top_offset outside (-1, 0)!"
    next_recovered = sum(1 for n in range(20, 30) if int(res["next_token_id"][0, n, 0]) == 300 + n)
    assert next_recovered >= 9, f"Expected +1 pieces stored in next_token_id separately, got {next_recovered}/10"
    for n in range(20, 30):
        assert int(res["top_token_id"][0, n, 0]) != 300 + n, f"+1 token {300+n} leaked into causal top_token_id!"

    # 3. Straddling single-doc trap (neuron 30) must NOT become eligible / named across document halves
    assert not bool(res["eligible"][0, 30]) and not bool(res["raw_2plus2"][0, 30]), "Straddling doc falsely split across halves!"

    # 4. BOS trap (neuron 31): offset=-1 must never match BOS token 2
    for k in range(5):
        if int(res["top_token_id"][0, 31, k]) == 2:
            assert int(res["top_offset"][0, 31, k]) != -1, "BOS at index 0 leaked into offset=-1!"

    # 5. Held-out causal rate beats frequency-matched wrong piece
    assert res["stats"]["smp_named_rate"] is not None and res["stats"]["smp_freq_rate"] is not None
    assert res["stats"]["smp_named_rate"] > res["stats"]["smp_freq_rate"] + 0.1

    # 6. Validate saved NPZ schema & causal-offset enforcement via publish_gap_census.validate_input_census_npz
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "L00_input_census.npz"
        ic.save_layer_input_census(p, res)
        pgc.validate_input_census_npz(p)
        loaded = np.load(p)
        assert "windows" not in loaded.files
        assert "raw_2plus2" in loaded.files and "other_face_null" in loaded.files and "smp_freq_hits" in loaded.files
        assert np.array_equal(loaded["top_offset"], res["top_offset"])
    print(
        f"[PASS] input_census CPU checks (causal={exact_causal}/20, next_sep={next_recovered}/10, "
        f"straddling_blocked=True, smp={res['stats']['smp_named_rate']} vs freq={res['stats']['smp_freq_rate']})"
    )


if __name__ == "__main__":
    test_input_census_synthetic()
    print("\nAll input_census CPU checks passed.")
