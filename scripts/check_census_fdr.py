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

from backend import census_fdr as fdr  # noqa: E402


def _make_synthetic_run(root: Path) -> Path:
    rng = np.random.default_rng(2026)
    prompts = [f"q{i}" for i in range(12)]
    doses = (-1.0, 1.0)
    run_path = root / "L99-match_things.jsonl"
    meta_path = root / "L99-match_things.meta.json"
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

    N, C = 40, 100
    base_words = ["alpha", "beta", "gamma"]
    rows = []
    # Interleave baselines across prompt groups
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
        # Neurons: first 20 inject their specific signal word across ALL 12 questions (even C4!)
        for n in range(N):
            for d in doses:
                ws = ["common", "noise" + str(int(rng.integers(0, 40))), "noise" + str(int(rng.integers(0, 40)))]
                if n < 20:
                    ws.extend([f"sigword{n:02d}", f"sibword{n:02d}"])
                if n >= 20 and n < 28:
                    # "thought"-glued nuisance token ONLY on half A + B — stripped by no_thought!
                    ws.append("gluedthought")
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
        run = fdr.StreamedRun(p, subsets)
        assert "gluedthought" not in [run.vocab[w] for w in next(iter(run.filtered_counts("all", True).values()))]
        res = fdr.eval_pair(run, "A", "B", "all", strip_thought=True, run_null_splits=20, seed=42)
        oos = fdr.out_of_sample_3way(run, strip_thought=True)
        named_ids = {r["neuron"] for r in res["_named_records"]}
        assert len(named_ids & set(range(20))) >= 18, f"Expected >=18/20 signal neurons named, got {named_ids}"
        assert len(named_ids - set(range(20))) == 0, f"Expected 0 false positives, got {named_ids - set(range(20))}"
        assert res["null_splits"]["fdp05_false_mean"] <= 1.0, f"Null split mean too high: {res['null_splits']}"
        assert oos["frac_p05_on_C4"] >= 0.9, f"Expected >=0.9 C4 replication, got {oos}"
        print(f"[PASS] census_fdr synthetic checks (named={len(named_ids)}, C4_p05={oos['frac_p05_on_C4']}, null_mean={res['null_splits']['fdp05_false_mean']})")


if __name__ == "__main__":
    test_census_fdr()
    print("\nAll census_fdr CPU checks passed.")
