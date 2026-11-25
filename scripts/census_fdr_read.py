#!/usr/bin/env python3
"""Exploratory empirical-FDR decoding reader across all 25 whole census layers (L30-L54).

Post-hoc, unregistered proposal (gates nothing, no page integration). Uses fixed
alpha = 0.05 against isotropic controls, exact thinking-channel glue filtering,
census.COLLAPSE_RULE word exclusion (from both counts and lnw covariates), control
bootstrap stability, and an exchangeable half-control code-symmetry check.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import census_fdr as fdr  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402

ALPHA = 0.05


def _load_registered_verdict(layer: int, runs_dir: Path) -> Dict[str, Any]:
    vpath = REPO_ROOT / f"experiments/results/census_L{layer}_whole/L{layer}_verdict.json"
    v = json.loads(vpath.read_text(encoding="utf-8"))
    run_name = str(v["run"])
    run_path = runs_dir / run_name
    if "reported" in v:
        rep = v["reported"]
        figs = rep.get("figures") or {}
        reg_q05 = figs.get("q05")
        reg_named = figs.get("named")
        reg_above = figs.get("above_every_control")
        take_shorter = bool((rep.get("max_new_rule") or {}).get("take_shorter", False))
    else:
        r48 = v["readings"]["48"]
        cnt48 = (r48.get("figures") or {}).get("counts") or {}
        reg_q05 = cnt48.get("q05")
        reg_named = len(r48.get("named") or [])
        reg_above = cnt48.get("above_every_control")
        take_shorter = bool((v.get("max_new_rule") or {}).get("take_shorter", False))

    page_described: Optional[int] = None
    upath = cz.units_path_of(run_path)
    if upath.is_file():
        neus: List[Tuple[float, float]] = []
        ctls: List[Tuple[float, float]] = []
        with upath.open("r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                k = o.get("kind")
                pair = (float(o.get("mass", 0.0)), float(o.get("mass_shown", 0.0)))
                if k == "neuron":
                    neus.append(pair)
                elif k == "random":
                    ctls.append(pair)
        if ctls:
            max_m = max(m for m, _ in ctls)
            max_ms = max(ms for _, ms in ctls)
            page_described = sum(1 for m, ms in neus if m > max_m and ms > max_ms)

    return {
        "layer": int(layer),
        "run": run_name,
        "run_path": run_path,
        "half_a": [str(q) for q in v["half_a"]],
        "half_b": [str(q) for q in v["half_b"]],
        "registered": {
            "q05": int(reg_q05) if reg_q05 is not None else None,
            "named": int(reg_named) if reg_named is not None else None,
            "above_every_control": int(reg_above) if reg_above is not None else None,
            "page_described": int(page_described) if page_described is not None else None,
            "take_shorter_rule": take_shorter,
        },
    }


def _weights_from_index(data_dir: Path) -> Optional[Dict[str, Any]]:
    p = data_dir / "index" / "passages.d" / "meta.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        w = d.get("weights")
        return w if isinstance(w, dict) else None
    except (OSError, ValueError):
        return None


def analyze_one(
    tag: str,
    path: Path,
    *,
    half_a: Optional[List[str]] = None,
    half_b: Optional[List[str]] = None,
    registered: Optional[Dict[str, Any]] = None,
    collapse_words: Optional[Iterable[str]] = None,
    out_dir: Path,
    n_null_splits: int = 50,
    n_boot: int = 200,
    write_named: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    meta = json.loads(cz.meta_path_of(path).read_text(encoding="utf-8"))
    prompts = [q for q in (meta.get("prompts") or {}) if any(p == q for p, _ in meta["design"])]
    if half_a is not None and half_b is not None:
        A_set, B_set = set(half_a), set(half_b)
    else:
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

    run = fdr.StreamedRun(path, subsets, max_new=48, collapse_words=collapse_words, load_collapse=True)
    entry = fdr.eval_pair(
        run,
        "A",
        "B",
        "all",
        strip_thought=True,
        strip_collapse=True,
        alpha=ALPHA,
        run_null_splits=n_null_splits,
        n_boot=n_boot,
        seed=1000 + run.layer,
    )
    named_records = entry.pop("_named_records")

    # Measure COLLAPSE_RULE impact using the already-streamed run (no extra disk scan).
    if run.collapse_wids:
        no_col = fdr.eval_pair(
            run,
            "A",
            "B",
            "all",
            strip_thought=True,
            strip_collapse=False,
            alpha=ALPHA,
            run_null_splits=0,
            n_boot=0,
            seed=1000 + run.layer,
        )
        no_col_recs = no_col.pop("_named_records")
        clean_ids = {int(r["neuron"]) for r in named_records}
        unfilt_ids = {int(r["neuron"]) for r in no_col_recs}
        collapse_impact = {
            "unfiltered_fdp05": int(no_col["fdp05"]["count"]),
            "filtered_fdp05": int(entry["fdp05"]["count"]),
            "dropped_only_by_collapse": int(len(unfilt_ids - clean_ids)),
            "gained_after_collapse_filter": int(len(clean_ids - unfilt_ids)),
        }
    else:
        n_clean = int(entry["fdp05"]["count"])
        collapse_impact = {
            "unfiltered_fdp05": n_clean,
            "filtered_fdp05": n_clean,
            "dropped_only_by_collapse": 0,
            "gained_after_collapse_filter": 0,
        }

    if len(prompts) == 12:
        entry["oos_3way"] = fdr.out_of_sample_3way(run, strip_thought=True, strip_collapse=True, alpha=ALPHA)

    if write_named and named_records:
        named_out = {
            "exploratory": True,
            "registered": False,
            "gates_nothing": True,
            "alpha": ALPHA,
            "tag": tag,
            "file": path.name,
            "layer": run.layer,
            "design": run.design,
            "model": meta.get("model"),
            "count": len(named_records),
            "neurons": named_records,
        }
        (out_dir / f"{tag}_named.json").write_text(json.dumps(named_out, separators=(",", ":")) + "\n", encoding="utf-8")

    out: Dict[str, Any] = {
        "tag": tag,
        "file": path.name,
        "layer": run.layer,
        "design": run.design,
        "model": meta.get("model"),
        "prompts": len(prompts),
        "neurons": len(run.neurons),
        "controls": len(run.controls),
        "elapsed_s": round(time.time() - t0, 1),
        **entry,
        "collapse_impact": collapse_impact,
    }
    if registered is not None:
        out["registered"] = registered

    boot = out.get("bootstrap_fdp05") or {}
    sym = out.get("code_symmetry_check") or {}
    n3 = out["n3_covered"]
    print(
        f"{tag:<12} L{run.layer:2d} ({out['elapsed_s']:4.1f}s): "
        f"T_auc={out['T_dec_auc']:.3f} reg={registered.get('named') if registered else '-'}(page={registered.get('page_described') if registered else '-'}) "
        f"N1_05={out['fdp05']['count']} [boot {boot.get('p10', 0)}-{boot.get('p90', 0)}, ctl>={out['fdp05']['controls_above_thresh']}] "
        f"N3_05={n3.get('fdp05', {}).get('count', 0)} ({100.0*n3['share_covered']:.1f}%) "
        f"col_drop={collapse_impact['dropped_only_by_collapse']} sym_max={sym.get('fdp05_false_max', '-')}",
        flush=True,
    )
    del run
    gc.collect()
    return out


def _format_summary_txt(results: List[Dict[str, Any]], extras: List[Dict[str, Any]]) -> str:
    lines = [
        "Exploratory Empirical-FDR Proposal Sweep Across All 25 Whole Census Layers (L30-L54)",
        "Status: post-hoc, unregistered, exploratory (gates nothing, no page integration). Fixed alpha=0.05 vs isotropic controls.",
        "-" * 148,
        (
            f"{'Layer':>5} {'Design':<13} {'Q':>2} {'Reg_q05':>7} {'RegNmd':>7} {'PageDsc':>7} "
            f"{'rABr':>6} {'rBAr':>6} {'T_auc':>6} {'N1_05':>6} {'Boot10-90':>11} {'Ctl>=t':>6} "
            f"{'N3_05':>6} {'Cov%':>5} {'NoCol05':>7} {'ColDrop':>7} {'C4_p05':>6} {'SymMax':>6}"
        ),
    ]
    for r in results:
        reg = r.get("registered") or {}
        n3 = r["n3_covered"]
        n3_c = n3.get("fdp05", {}).get("count", 0)
        boot = r.get("bootstrap_fdp05") or {}
        boot_rng = f"{boot.get('p10', 0)}-{boot.get('p90', 0)}"
        col = r.get("collapse_impact") or {}
        oos = r.get("oos_3way") or {}
        c4 = f"{oos['frac_p05_on_C4']:.3f}" if oos.get("frac_p05_on_C4") is not None else "-"
        sym = r.get("code_symmetry_check") or {}
        lines.append(
            f"L{r['layer']:2d}   {r['design']:<13} {r['prompts']:2d} "
            f"{str(reg.get('q05')):>7} {str(reg.get('named')):>7} {str(reg.get('page_described')):>7} "
            f"{r['rep_AB_res_auc']:6.3f} {r['rep_BA_res_auc']:6.3f} {r['T_dec_auc']:6.3f} "
            f"{r['fdp05']['count']:6d} {boot_rng:>11} {r['fdp05']['controls_above_thresh']:6d} "
            f"{n3_c:6d} {100.0*n3['share_covered']:5.1f} {col.get('unfiltered_fdp05', 0):7d} "
            f"{col.get('dropped_only_by_collapse', 0):7d} {c4:>6} {sym.get('fdp05_false_max', -1):6d}"
        )
    if extras:
        lines.extend(["", "Additional Diagnostic Checks (Not Whole Census Layers)", "-" * 100])
        for r in extras:
            boot = r.get("bootstrap_fdp05") or {}
            sym = r.get("code_symmetry_check") or {}
            oos = r.get("oos_3way") or {}
            c4 = f"{oos['frac_p05_on_C4']:.3f}" if oos.get("frac_p05_on_C4") is not None else "-"
            lines.append(
                f"{r['tag']:<20} L{r['layer']:2d} ({r['neurons']}/{r['controls']}) "
                f"T_auc={r['T_dec_auc']:.4f} N1_05={r['fdp05']['count']} "
                f"boot10-90={boot.get('p10', 0)}-{boot.get('p90', 0)} ctl>={r['fdp05']['controls_above_thresh']} "
                f"N3_05={r['n3_covered'].get('fdp05', {}).get('count', 0)} C4_p05={c4} sym_max={sym.get('fdp05_false_max', '-')}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run exploratory empirical-FDR proposal across all 25 whole census layers (L30-L54)")
    ap.add_argument("--out-dir", default="experiments/results/census_fdr_sweep")
    ap.add_argument("--n-null-splits", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--write-named", action="store_true", help="Write compact *_named.json files into out-dir (default: False)")
    ap.add_argument("--whole-only", action="store_true", help="Skip L45 column-mix and L29/L55 trial checks")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    started_iso = gp.utc_iso(t0)
    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.write_named:
        for stale in out_dir.glob("*_named.json"):
            stale.unlink()

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()
    runs_dir = data_dir / "census" / "runs"
    weights_card = _weights_from_index(data_dir)
    code_files = [
        REPO_ROOT / "scripts/census_fdr_read.py",
        REPO_ROOT / "backend/census_fdr.py",
        REPO_ROOT / "backend/gap_stats.py",
        REPO_ROOT / "backend/gap_provenance.py",
    ]

    results: List[Dict[str, Any]] = []
    l45_info: Optional[Dict[str, Any]] = None
    for layer in range(30, 55):
        spec = _load_registered_verdict(layer, runs_dir)
        if layer == 45:
            l45_info = spec
        p = spec["run_path"]
        if not p.exists():
            raise FileNotFoundError(f"Missing whole-layer run for L{layer}: {p}")
        res = analyze_one(
            f"L{layer:02d}_whole",
            p,
            half_a=spec["half_a"],
            half_b=spec["half_b"],
            registered=spec["registered"],
            out_dir=out_dir,
            n_null_splits=args.n_null_splits,
            n_boot=args.n_boot,
            write_named=args.write_named,
        )
        results.append(res)
        model_id = res.get("model")
        payload = {
            "exploratory": True,
            "registered": False,
            "gates_nothing": True,
            "status_note": "Exploratory post-hoc proposal across all 25 whole census layers (L30-L54); fixed alpha=0.05; no page integration.",
            "alpha": ALPHA,
            "elapsed_total_s": round(time.time() - t0, 1),
            "provenance": gp.build_provenance(
                REPO_ROOT,
                code_files,
                started_at=started_iso,
                finished_at=gp.utc_iso(),
                model_id=model_id,
                weights=weights_card,
                neuron_ids="all",
            ),
            "runs": results,
            "extra_checks": [],
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    extras: List[Dict[str, Any]] = []
    if not args.whole_only:
        l45_mix = data_dir / "census" / "nulls" / "L45-mixes-as-neurons.jsonl"
        if l45_mix.is_file() and l45_info is not None:
            cpath = cz.collapse_path_of(l45_info["run_path"])
            cwords = json.loads(cpath.read_text(encoding="utf-8")).get("words", []) if cpath.is_file() else []
            extras.append(
                analyze_one(
                    "L45_cols_mix_null",
                    l45_mix,
                    half_a=l45_info["half_a"],
                    half_b=l45_info["half_b"],
                    collapse_words=cwords,
                    out_dir=out_dir,
                    n_null_splits=args.n_null_splits,
                    n_boot=args.n_boot,
                    write_named=False,
                )
            )
        for tag, rel in [
            ("L29_trial", "census/trials/things_walk/L29-match_things-trial.jsonl"),
            ("L55_trial", "census/trials/things_walk/L55-match_things-trial.jsonl"),
        ]:
            tp = data_dir / rel
            if tp.is_file():
                extras.append(
                    analyze_one(
                        tag,
                        tp,
                        out_dir=out_dir,
                        n_null_splits=args.n_null_splits,
                        n_boot=args.n_boot,
                        write_named=False,
                    )
                )

    final_model = results[0].get("model") if results else None
    payload = {
        "exploratory": True,
        "registered": False,
        "gates_nothing": True,
        "status_note": "Exploratory post-hoc proposal across all 25 whole census layers (L30-L54); fixed alpha=0.05; no page integration.",
        "alpha": ALPHA,
        "elapsed_total_s": round(time.time() - t0, 1),
        "provenance": gp.build_provenance(
            REPO_ROOT,
            code_files,
            started_at=started_iso,
            finished_at=gp.utc_iso(),
            model_id=final_model,
            weights=weights_card,
            neuron_ids="all",
        ),
        "runs": results,
        "extra_checks": extras,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    txt = _format_summary_txt(results, extras)
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
