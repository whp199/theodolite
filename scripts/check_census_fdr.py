#!/usr/bin/env python3
"""CPU-only unit tests for backend/census_fdr.py."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import census as cz  # noqa: E402
from backend import census_fdr as fdr  # noqa: E402


def test_emp_fdp_hand_and_ties() -> None:
    # Hand-computed test: N=10 neurons, C=19 controls -> N/(C+1) = 0.5.
    # Suppose neuron scores (descending) are:
    # 5.0, 5.0 (tie of 2), 4.0, 4.0, 3.0 (tie of 3 above/at 3.0 -> wait let's be exact):
    # n_scores: n0=6.0, n1=5.0, n2=5.0, n3=4.0, n4=4.0, n5=3.0 (all positive_ok=True), n6..n9=-1.0 (pos_ok=False)
    # ctl_vals: c0=5.5 (1 ctl >= 6.0? No, >= 5.0 is 1), wait let's construct exact threshold:
    # Let ctl_vals have c0=6.0 (so 1 ctl >= 6.0 -> FDP at n_ge=1 is 0.5*2/1=1.0),
    # c1=4.0 (so at t=5.0: c_ge=1 -> FDP at n_ge=3 (6.0,5.0,5.0) is 0.5*(1+1)/3 = 0.3333; wait we want <= 0.05!)
    # With N=10, C=19, N/(C+1)=0.5 -> minimum FDP with c_ge=0 is 0.5*1/n_ge <= 0.05 iff n_ge >= 10!
    # Wait: to get FDP <= 0.05 with small counts, use N=10, C=199 -> N/(C+1) = 10/200 = 0.05!
    # Then FDP(t) = 0.05 * (1 + c_ge) / n_ge!
    N, C = 10, 199
    scores = {}
    pos_ok = {}
    # Neurons:
    # n0=10.0 (1) -> 1 ctl >= 10 -> wait if 0 ctls >= 10: FDP = 0.05 * 1 / 1 = 0.05!
    # Put 0 ctls at >= 8.0, 1 ctl tied AT 5.0, and 1 ctl at 4.0.
    # Neuron values: n0=9.0, n1=8.0, n2=5.0, n3=5.0, n4=5.0, n5=4.0, n6=3.0, n7..n9=-1.0 (pos_ok=False)
    neu_vals = [9.0, 8.0, 5.0, 5.0, 5.0, 4.0, 3.0, -1.0, -1.0, -1.0]
    for i, v in enumerate(neu_vals):
        scores[("neuron", i)] = v
        pos_ok[("neuron", i)] = v > 0.0
    ctl_vals = np.zeros(C, dtype=np.float64)
    ctl_vals[0] = 5.0   # tied right at neuron score 5.0 -> c_ge(5.0) MUST include this control (side='left')!
    ctl_vals[1] = 4.0   # tied at 4.0 -> c_ge(4.0) = 2
    for c in range(C):
        scores[("random", c)] = float(ctl_vals[c])
        pos_ok[("random", c)] = True

    # Expected step-up evaluation (N/(C+1) = 0.05):
    # t=9.0: n_ge=1, c_ge=0 -> FDP = 0.05*(1+0)/1 = 0.0500 <= 0.05
    # t=8.0: n_ge=2, c_ge=0 -> FDP = 0.05*(1+0)/2 = 0.0250 <= 0.05
    # t=5.0 (3-way neuron tie grouped at j=4): n_ge=5, c_ge=1 (ctl_vals[0]==5.0) -> FDP = 0.05*(1+1)/5 = 0.0200 <= 0.05
    # t=4.0: n_ge=6, c_ge=2 -> FDP = 0.05*(1+2)/6 = 0.0250 <= 0.05
    # t=3.0: n_ge=7, c_ge=2 -> FDP = 0.05*(1+2)/7 = 0.021428... <= 0.05!
    # Wait: make ctl_vals[2]..ctl_vals[8] = 3.0 so t=3.0 fails and t=5.0 or t=4.0 is the exact boundary!
    # Put 5 controls at 3.5 -> at t=4.0 (n_ge=6, c_ge=2) -> 0.025, wait put 7 controls at 3.9 -> at t=3.0 c_ge=9 -> 0.05*10/7=0.0714 (>0.05)!
    ctl_vals[2:9] = 3.5
    for c in range(2, 9):
        scores[("random", c)] = float(ctl_vals[c])

    # Now at t=5.0: n_ge=5, c_ge=1 -> 0.05*2/5 = 0.0200 <= 0.05
    # At t=4.0: n_ge=6, c_ge=2 -> 0.05*3/6 = 0.0250 <= 0.05
    # Let's make t=4.0 ALSO fail so tie boundary at t=5.0 is tested directly: move ctl_vals[1] = 4.5, add 4 ctls at 4.0 -> at t=4.0: c_ge=1+1+4=6 -> FDP = 0.05*7/6 = 0.0583 > 0.05!
    ctl_vals[1] = 4.5
    ctl_vals[2:6] = 4.0
    ctl_vals[6:9] = 3.0
    for c in range(1, 9):
        scores[("random", c)] = float(ctl_vals[c])

    res = fdr.emp_fdp_select(scores, pos_ok, alpha=0.05)
    # At t=5.0: n_ge=5 (all 3 tied neurons at 5.0 included together!), c_ge=1 (strict tie at 5.0 counted!), FDP=0.05*(1+1)/5 = 0.02.
    # Wait: why is above_max_ctl 2 (only 9.0 and 8.0 > 5.0)? Exact tie check!
    assert res["count"] == 5, f"Expected 5 selected (grouped ties at 5.0), got {res}"
    assert res["threshold"] == 5.0, f"Expected threshold 5.0, got {res}"
    assert res["fdp"] == 0.02, f"Expected exact FDP 0.02, got {res}"
    assert res["controls_above_thresh"] == 1, f"Expected 1 control >= 5.0 (tie-inclusive), got {res}"
    assert res["above_max_ctl"] == 2, f"Expected 2 neurons strictly > 5.0, got {res}"

    # Also test tie failure when one too many controls tie AT 5.0:
    # Add 2 more controls at 5.0 -> c_ge(5.0) = 3 -> FDP at 5.0 = 0.05*(1+3)/5 = 0.04 (still passes);
    # Add 3rd extra (total 4 at 5.0): FDP = 0.05*(1+4)/5 = 0.05 (passes); add 4th (total 5 at 5.0): FDP=0.05*6/5=0.06 > 0.05 -> drops back to t=8.0 (n_ge=2, c_ge=0, FDP=0.025).
    for c in range(10, 14):
        scores[("random", c)] = 5.0
    res2 = fdr.emp_fdp_select(scores, pos_ok, alpha=0.05)
    assert res2["count"] == 2 and res2["threshold"] == 8.0, f"Expected drop to t=8.0 when 5 ctls tie at 5.0, got {res2}"
    print("[PASS] emp_fdp_select hand calculation and boundary tie handling verified")


def test_exchangeable_null_montecarlo() -> None:
    rng = np.random.default_rng(12345)
    N, C, n_trials = 500, 1000, 200
    n_sig = 75
    alpha = 0.05
    fdps = []
    powers = []
    neu_keys = [("neuron", i) for i in range(N)]
    ctl_keys = [("random", i) for i in range(C)]
    n_tot = N + C

    for _ in range(n_trials):
        # Heavy-tailed Student-t(df=5) draws: exchangeable between null neurons and controls
        z = rng.normal(0.0, 1.0, size=n_tot)
        chi = rng.chisquare(5.0, size=n_tot)
        t_draws = z / np.sqrt(chi / 5.0)
        t_draws[:n_sig] += 4.8
        scores = {u: float(t_draws[i]) for i, u in enumerate(neu_keys)}
        for j, u in enumerate(ctl_keys):
            scores[u] = float(t_draws[N + j])
        pos_ok = {u: (scores[u] > 0.0) for u in scores}
        sel = fdr.emp_fdp_select(scores, pos_ok, alpha=alpha)
        named = sel["named"]
        if named:
            n_false = sum(1 for _, idx in named if idx >= n_sig)
            fdps.append(n_false / len(named))
            powers.append((len(named) - n_false) / n_sig)
        else:
            fdps.append(0.0)
            powers.append(0.0)

    mean_fdp = float(np.mean(fdps))
    mean_pwr = float(np.mean(powers))
    assert mean_fdp <= alpha + 0.015, f"Exchangeable heavy-tail FDP exceeded bound: {mean_fdp:.4f}"
    assert mean_pwr >= 0.4, f"Monte Carlo signal power unexpectedly low: {mean_pwr:.4f}"
    print(f"[PASS] exchangeable Student-t(5) MC FDR verified (mean_fdp={mean_fdp:.4f} <= {alpha+0.015:.4f}, mean_power={mean_pwr:.4f})")


def _make_synthetic_run(root: Path) -> Path:
    rng = np.random.default_rng(2026)
    prompts = [f"q{i}" for i in range(12)]
    doses = (-1.0, 1.0)
    run_path = root / "L99-match_things.jsonl"
    meta_path = cz.meta_path_of(run_path)
    collapse_path = cz.collapse_path_of(run_path)
    design = [[p, d] for p in prompts for d in doses]
    meta_path.write_text(
        json.dumps(
            {
                "layer": 99,
                "design_name": "match_things",
                "prompts": {p: p for p in prompts},
                "design": design,
            }
        ),
        encoding="utf-8",
    )
    collapse_path.write_text(
        json.dumps({"words": ["collapsefoo", "collapsebar"]}),
        encoding="utf-8",
    )

    # 0..19: strong signal neurons
    # 20..27: glue-only neurons ("gluedthought", "thoughtglued")
    # 28..33: collapse-only neurons ("collapsefoo", "collapsebar")
    # 34..37: genuine English "thought" signal neurons ("thought", "thoughtful", "afterthought")
    N, C = 46, 120
    base_words = ["alpha", "beta", "gamma", "forethought"]
    rows = []
    for pi, p in enumerate(prompts):
        rows.append(
            {
                "kind": "baseline",
                "neuron": -1,
                "prompt": p,
                "dose": 0.0,
                "text": " ".join(base_words),
                "token_ids": [10, 11, 12, 13, 14, 15],
                "logprob": [-0.1] * 6,
                "finish": "stop",
            }
        )
        for n in range(N):
            for d in doses:
                ws = ["common", "noise" + str(int(rng.integers(0, 40))), "noise" + str(int(rng.integers(0, 40)))]
                if n < 20:
                    ws.extend([f"sigword{n:02d}", f"sibword{n:02d}"])
                elif 20 <= n < 28:
                    ws.extend(["gluedthought", "thoughtglued"])
                elif 28 <= n < 34:
                    ws.extend(["collapsefoo", "collapsebar"])
                elif 34 <= n < 38:
                    ws.extend([f"thought", "thoughtful", "afterthought", f"realmind{n:02d}"])
                ids = [10, 11, 12, 13, 20 + int(rng.integers(0, 50)), 30 + int(rng.integers(0, 50))]
                rows.append(
                    {
                        "kind": "neuron",
                        "neuron": n,
                        "prompt": p,
                        "dose": d,
                        "text": " ".join(ws),
                        "token_ids": ids,
                        "logprob": [-0.2] * 6,
                        "finish": "stop",
                    }
                )
        for c in range(C):
            for d in doses:
                ws = ["common", "gluedthought" if c % 3 == 0 else "fillerword", "noise" + str(int(rng.integers(0, 40)))]
                if c % 5 == 0:
                    ws.append("collapsefoo")
                ids = [10, 11, 12, 13, 20 + int(rng.integers(0, 50)), 30 + int(rng.integers(0, 50))]
                rows.append(
                    {
                        "kind": "random",
                        "neuron": c,
                        "prompt": p,
                        "dose": d,
                        "text": " ".join(ws),
                        "token_ids": ids,
                        "logprob": [-0.2] * 6,
                        "finish": "stop",
                    }
                )

    with open(run_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return run_path


def test_census_fdr() -> None:
    assert fdr.is_thought_glue("gluedthought")
    assert fdr.is_thought_glue("thoughtglued")
    assert not fdr.is_thought_glue("thought")
    assert not fdr.is_thought_glue("thoughtful")
    assert not fdr.is_thought_glue("afterthought")
    assert not fdr.is_thought_glue("customthought", {"customthought"})

    with tempfile.TemporaryDirectory() as td:
        p = _make_synthetic_run(Path(td))
        prompts = [f"q{i}" for i in range(12)]
        A, B = set(prompts[:6]), set(prompts[6:])
        A4, B4, C4 = set(prompts[:4]), set(prompts[4:8]), set(prompts[8:12])
        subsets = {
            "all": lambda q, d: q in A or q in B,
            "A": lambda q, d: q in A,
            "B": lambda q, d: q in B,
            "AB8": lambda q, d: q in A4 or q in B4,
            "A4": lambda q, d: q in A4,
            "B4": lambda q, d: q in B4,
            "C4": lambda q, d: q in C4,
        }
        run = fdr.StreamedRun(p, subsets, load_collapse=True)
        cnt_clean = run.filtered_counts("all", strip_thought=True, strip_collapse=True)
        cnt_raw = run.filtered_counts("all", strip_thought=False, strip_collapse=False)

        # Verify ALL units have glue & collapse words stripped in clean mode, while English thought words survive.
        forbidden = {"gluedthought", "thoughtglued", "collapsefoo", "collapsebar"}
        for u, cnt in cnt_clean.items():
            words_u = {run.vocab[w] for w in cnt}
            hit = words_u & forbidden
            assert not hit, f"Excluded token survived in {u}: {hit}"
        for n in range(34, 38):
            words_n = {run.vocab[w] for w in cnt_clean[("neuron", n)]}
            assert {"thought", "thoughtful", "afterthought"} <= words_n, f"Real English thought words dropped for {n}: {words_n}"

        # Verify excluded words are ALSO removed from the lnw covariate!
        cov_clean = run.covariates("all", strip_thought=True, strip_collapse=True)
        cov_raw = run.covariates("all", strip_thought=False, strip_collapse=False)
        for n in range(20, 34):
            assert cov_clean[("neuron", n)]["lnw"] < cov_raw[("neuron", n)]["lnw"], (
                f"Expected cleaned lnw < raw lnw for excluded-word neuron {n}"
            )
        assert len({run.vocab[w] for w in cnt_raw[("neuron", 20)]} & {"gluedthought", "thoughtglued"}) == 2

        res = fdr.eval_pair(run, "A", "B", "all", strip_thought=True, strip_collapse=True, alpha=0.05, run_null_splits=20, n_boot=50, seed=42)
        assert "fdp10" not in res, "Primary result should use fixed alpha=0.05 only"
        assert res["bootstrap_fdp05"]["n_boot"] == 50
        assert res["code_symmetry_check"]["fdp05_false_mean"] <= 1.0, f"Code symmetry check unexpected: {res['code_symmetry_check']}"

        oos = fdr.out_of_sample_3way(run, strip_thought=True, strip_collapse=True, alpha=0.05)
        named_ids = {r["neuron"] for r in res["_named_records"]}
        assert len(named_ids & set(range(20))) >= 18, f"Expected >=18/20 signal neurons named, got {named_ids}"
        assert len(named_ids & set(range(34, 38))) == 4, f"Expected all 4 real-English-thought neurons named, got {named_ids}"
        assert not (named_ids & set(range(20, 34))), f"Glue/collapse-only neurons were falsely named: {named_ids & set(range(20, 34))}"
        assert not (named_ids - set(range(20)) - set(range(34, 38))), f"Expected 0 false positives, got {named_ids}"
        assert oos["frac_p05_on_C4"] >= 0.9, f"Expected >=0.9 C4 replication, got {oos}"
        print(
            f"[PASS] census_fdr synthetic checks (named={len(named_ids)}, "
            f"boot_p50={res['bootstrap_fdp05']['p50']}, C4_p05={oos['frac_p05_on_C4']}, "
            f"sym_mean={res['code_symmetry_check']['fdp05_false_mean']})"
        )


if __name__ == "__main__":
    test_emp_fdp_hand_and_ties()
    test_exchangeable_null_montecarlo()
    test_census_fdr()
    print("\nAll census_fdr CPU checks passed.")
