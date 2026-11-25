#!/usr/bin/env python3
"""Run Design 2 (T_dec + no_thought + step-up empirical FDP) sweep across finished layers and trials."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import census_fdr as fdr  # noqa: E402


DEFAULT_RUNS = [
    ("L30_whole", "census/runs/L30-match_things.jsonl"),
    ("L31_whole", "census/runs/L31-match_wide.jsonl"),
    ("L34_whole", "census/runs/L34-match_ceil.jsonl"),
    ("L35_whole", "census/runs/L35-match_ceil.jsonl"),
    ("L36_whole", "census/runs/L36-match_ceil.jsonl"),
    ("L37_whole", "census/runs/L37-match_ceil.jsonl"),
    ("L53_whole", "census/runs/L53-match_things.jsonl"),
    ("L54_whole", "census/runs/L54-match_things.jsonl"),
    ("L29_trial", "census/trials/things_walk/L29-match_things-trial.jsonl"),
    ("L55_trial", "census/trials/things_walk/L55-match_things-trial.jsonl"),
]


def analyze_one(tag: str, path: Path, out_dir: Path, n_null_splits: int = 50) -> Dict[str, Any]:
    t0 = time.time()
    meta = json.loads(cz.meta_path_of(path).read_text(encoding="utf-8"))
    prompts = [q for q in (meta.get("prompts") or {}) if any(p == q for p, _ in meta["design"])]
    half_n = len(prompts) // 2
    A_set, B_set = set(prompts[:half_n]), set(prompts[half_n:])
    subsets: Dict[str, Callable[[str, float], bool]] = {
        "all": lambda p, d: p in A_set or p in B_set,
        "A": lambda p, d: p in A_set,
        "B": lambda p, d: p in B_set,
    }
    if len(prompts) == 12:
        A4, B4, C4 = set(prompts[:4]), set(prompts[4:8]), set(prompts[8:12])
        subsets.update(
            {
                "AB8": lambda p, d: p in A4 or p in B4,
                "A4": lambda p, d: p in A4,
                "B4": lambda p, d: p in B4,
                "C4": lambda p, d: p in C4,
            }
        )

    run = fdr.StreamedRun(path, subsets)
    entry = fdr.eval_pair(run, "A", "B", "all", strip_thought=True, run_null_splits=n_null_splits, seed=1000 + run.layer)
    named_records = entry.pop("_named_records")
    if len(prompts) == 12:
        entry["oos_3way"] = fdr.out_of_sample_3way(run, strip_thought=True)

    if named_records:
        named_out = {
            "tag": tag,
            "file": path.name,
            "layer": run.layer,
            "design": run.design,
            "count": len(named_records),
            "neurons": named_records,
        }
        (out_dir / f"{tag}_named.json").write_text(json.dumps(named_out, indent=1) + "\n", encoding="utf-8")

    out = {
        "tag": tag,
        "file": path.name,
        "layer": run.layer,
        "design": run.design,
        "prompts": len(prompts),
        "neurons": len(run.neurons),
        "controls": len(run.controls),
        "elapsed_s": round(time.time() - t0, 1),
        **entry,
    }
    print(
        f"{tag:<10} L{run.layer:2d} ({out['elapsed_s']}s): T_auc={out['T_dec_auc']:.3f} "
        f"rlib={out['readlib_named']} FDP05={out['fdp05']['count']} (N3={out['n3_covered'].get('fdp05',{}).get('count', 0)}, "
        f"cov={out['n3_covered']['share_covered']:.3f}) FDP10={out['fdp10']['count']} "
        f"Null05={out['null_splits']['fdp05_false_mean']:.2f}",
        flush=True,
    )
    del run
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Design 2 empirical-FDR rescue sweep")
    ap.add_argument("--out-dir", default="experiments/results/census_fdr_sweep")
    ap.add_argument("--n-null-splits", type=int, default=50)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()

    results: List[Dict[str, Any]] = []
    for tag, rel in DEFAULT_RUNS:
        p = data_dir / rel
        if not p.exists():
            print(f"MISSING: {p}", flush=True)
            continue
        res = analyze_one(tag, p, out_dir, n_null_splits=args.n_null_splits)
        results.append(res)
        (out_dir / "summary.json").write_text(
            json.dumps({"elapsed_total_s": round(time.time() - t0, 1), "runs": results}, indent=1) + "\n",
            encoding="utf-8",
        )

    lines = [
        "Design 2 Empirical-FDR Rescue Sweep (T_dec = min(rz(rep_AB_res), rz(rep_BA_res)), no_thought, FDP <= 0.05)",
        "-" * 130,
        f"{'Tag':<10} {'Design':<13} {'L':>2} {'N/C':>11} {'rABr':>6} {'rBAr':>6} {'T_auc':>6} "
        f"{'rlib':>5} {'N1_05':>6} {'N3_05':>6} {'N3cov%':>6} {'FDP10':>6} {'Null05':>7} {'C4_p05':>7}",
    ]
    for r in results:
        nc = f"{r['neurons']}/{r['controls']}"
        n3 = r["n3_covered"]
        n3_c = n3.get("fdp05", {}).get("count", 0)
        ns = r.get("null_splits") or {}
        oos = r.get("oos_3way") or {}
        c4 = f"{oos['frac_p05_on_C4']:.3f}" if oos.get("frac_p05_on_C4") is not None else "-"
        lines.append(
            f"{r['tag']:<10} {r['design']:<13} {r['layer']:2d} {nc:>11} "
            f"{r['rep_AB_res_auc']:6.3f} {r['rep_BA_res_auc']:6.3f} {r['T_dec_auc']:6.3f} "
            f"{r['readlib_named']:5d} {r['fdp05']['count']:6d} {n3_c:6d} {100.0*n3['share_covered']:6.1f} "
            f"{r['fdp10']['count']:6d} {ns.get('fdp05_false_mean', -1):7.2f} {c4:>7}"
        )
    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
