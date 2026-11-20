#!/usr/bin/env python3
"""Owner's command line for Levelling Phase 6: driver evaluation ("Whether agents help"; `scripts/levelling_eval.py`).

Subcommands:
- `plan`: build, validate, and estimate a Phase 6 multi-session comparison study (reach ladder + matched-budget
  cheaper-knobs competitor + optional second driver family and `--bare` ablation).
- `run`: execute a study plan JSON (`--stand-in` on CPU or live 31B on the card with `gpu.lock`).
- `knobs`: run one owner cheaper-knob fresh claim reading (prompts, decoding, and/or transient steering).
- `rebuild`: emit a <= 200-word rebuild description from a recorded paired reading, and optionally verify it.
- `cold-page`: print the <= 1-page cold-driver brief naming no site or layer.
- `report`: model-free verification and report rendering from `<study_dir>/study_book.jsonl`.

Zero kept-out words appear here or in printed outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import level_atlas as la  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_drivers as ldrv  # noqa: E402
from backend import level_eval as lev  # noqa: E402
from backend import level_knobs as lkn  # noqa: E402
from backend import level_mcp as lmcp  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend.weight_reader import export_stand_in_safetensors  # noqa: E402


def _load_handle(stand_in: bool, out_dir: Path, *, port: int = lmcp.SERVER_PORT) -> tuple[Any, Optional[Path], Optional[Path], Any]:
    """Load either the CPU stand-in or the live 31B runtime (taking gpu.lock on the card)."""
    if stand_in:
        import torch
        from scripts.atlas_stand_in import stand_in as make_stand_in, use_stand_in_atlas

        torch.set_num_threads(1)
        use_stand_in_atlas()
        rt, _vault = make_stand_in()
        h = lv.handle(rt)
        wdir = export_stand_in_safetensors(h.lm, out_dir / "_stand_in_weights", model_id=h.model_id)
        adir = la.build_stand_in_atlas_index(h.lm, h.W_U, h.tokenizer, out_dir / "_stand_in_atlas", model_id=h.model_id)
        return h, wdir, adir, None

    lock_f = lmcp.hold_card()
    why_not = lmcp.card_refusal(int(port))
    if why_not:
        lock_f.close()
        raise SystemExit(why_not)
    from backend import atlas_index as ai
    from backend import w4a16_fast
    from backend.runtime_factory import checkpoint_snapshot, make_runtime

    rt = make_runtime(load_jlens=False).load()
    if hasattr(rt, "jpp_lenses"):
        rt.jpp_lenses.clear()
    w4a16_fast.install(rt, "compiled")
    h = lv.handle(rt)
    wdir = checkpoint_snapshot(h.model_id.split("@", 1)[0])
    adir: Optional[Path] = None
    try:
        adir = la.resolve_atlas_index_dir(ai.atlas_dir())
    except la.AtlasReaderError:
        adir = None
    return h, wdir, adir, lock_f


def _cmd_plan(args: argparse.Namespace) -> int:
    sb = lses.SessionBudget(
        max_gpu_seconds=float(args.max_gpu_seconds),
        max_forwards=int(args.max_forwards),
        max_cells=int(args.max_cells),
        max_wall_seconds=float(args.max_wall_seconds),
        min_controls=int(args.min_controls),
    )
    db = ldrv.DriverBudget(
        max_turns=int(args.max_turns),
        max_tokens=int(args.max_tokens),
        max_dollars=(float(args.max_dollars) if args.max_dollars is not None else (0.0 if args.driver == "scripted" else None)),
        max_wall_seconds=float(args.max_wall_seconds),
    )
    claim_design = {}
    if args.claim_controls is not None:
        claim_design["controls_override"] = int(args.claim_controls)
    if args.claim_n_target is not None:
        claim_design["n_target"] = int(args.claim_n_target)

    plan = lev.default_phase6_study_plan(
        name=args.name,
        brief_name=args.brief,
        sessions_per_arm=int(args.sessions),
        include_atlas=not args.no_atlas,
        include_bare_ablation=bool(args.bare_ablation),
        driver_family=args.driver,
        driver_model=args.model,
        second_driver_family=args.second_driver,
        second_driver_model=args.second_model,
        session_budget=sb,
        driver_budget=db,
        claim_design=claim_design,
    )
    doc = plan.canonical_dict()
    doc["study_sha256"] = plan.sha256
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote study plan ({len(plan.arms)} arms x {plan.sessions_per_arm} sessions, sha256 {plan.sha256[:12]}) to {out_p}")
    else:
        print(json.dumps(doc, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    plan = lev.study_plan_from_dict(raw)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    h, auto_w, auto_a, lock_f = _load_handle(bool(args.stand_in), out_dir, port=int(args.port))
    wdir = Path(args.weights_dir).resolve() if args.weights_dir else auto_w
    adir = Path(args.atlas_dir).resolve() if args.atlas_dir else auto_a
    ks = ldrv.KillSwitch(stop_file=Path(args.stop_file) if args.stop_file else (out_dir / "STOP"))
    try:
        res = lev.run_eval_study(
            h,
            plan,
            out_dir,
            weights_dir=wdir,
            atlas_dir=adir,
            kill_switch=ks,
            probe_sandbox=not args.no_probe,
        )
    except (lev.EvalRefused, ldrv.DriverRefused, lses.SessionRefused) as exc:
        raise SystemExit(f"refused: {exc}") from None
    finally:
        if lock_f is not None:
            lock_f.close()
    print(res["report"])
    return 0


def _cmd_knobs(args: argparse.Namespace) -> int:
    knob_raw = json.loads(Path(args.knob).read_text(encoding="utf-8"))
    out_dir = Path(args.book).resolve().parent if args.book else Path.cwd()
    h, _w, _a, lock_f = _load_handle(bool(args.stand_in), out_dir, port=int(args.port))
    try:
        starters = lbr.starter_briefs(h.tokenizer)
        b_path = Path(args.brief)
        brief = lbr.load_brief(b_path) if b_path.is_file() else starters[args.brief]
        bars = lcl.load_bars(Path(args.bars))
        res = lkn.run_fresh_knob_claim(
            h,
            brief,
            bars,
            knob_raw,
            base_dir=Path(args.knob).resolve().parent,
            book_path=Path(args.book) if args.book else None,
            controls_override=args.controls,
        )
    except (lkn.KnobRefused, lbr.BriefError, lcl.ClaimRefused, KeyError) as exc:
        raise SystemExit(f"refused: {exc}") from None
    finally:
        if lock_f is not None:
            lock_f.close()
    print(res["sheet"])
    return 0


def _cmd_rebuild(args: argparse.Namespace) -> int:
    try:
        desc = lev.make_rebuild_description(Path(args.book), int(args.seq))
    except lev.EvalRefused as exc:
        raise SystemExit(f"refused: {exc}") from None
    print(desc["description"])
    print(f"\n[{desc['word_count']} words, sha256={desc['sha256'][:12]}]")
    if args.check:
        out_book = Path(args.out_book) if args.out_book else (Path(args.book).resolve().parent / "rebuild_check_book.jsonl")
        h, _w, _a, lock_f = _load_handle(bool(args.stand_in), out_book.parent, port=int(args.port))
        try:
            chk = lev.run_rebuild_check(h, desc["description"], Path(args.book), int(args.seq), out_book=out_book)
        finally:
            if lock_f is not None:
                lock_f.close()
        print(json.dumps(chk, indent=2))
        return 0 if chk["ok"] else 1
    return 0


def _cmd_cold_page(args: argparse.Namespace) -> int:
    p = Path(args.brief)
    if p.is_file():
        brief = lbr.load_brief(p)
    else:
        starters = lbr.starter_briefs(None)
        if args.brief not in starters:
            raise SystemExit(f"Unknown brief {args.brief!r}; choose from {', '.join(sorted(starters))}.")
        brief = starters[args.brief]
    page = lev.render_cold_driver_page(brief)
    print(page["text"])
    print(f"\n[{page['word_count']} words, prompt_named_site={page['prompt_named_site']}, sha256={page['sha256'][:12]}]")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        loaded = lev.load_study_from_book(Path(args.study_dir))
    except lev.EvalRefused as exc:
        raise SystemExit(f"refused: {exc}") from None
    if args.json:
        print(json.dumps(loaded["summary"], indent=2))
    else:
        print(loaded["report"])
    if loaded["session_books_bad"]:
        raise SystemExit(f"Session book verification failures: {loaded['session_books_bad']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Levelling Phase 6 owner CLI: driver evaluation, reach ladders, and cheaper knobs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Build and estimate a Phase 6 multi-session study plan JSON.")
    p_plan.add_argument("--name", default="phase6_study")
    p_plan.add_argument("--brief", default="pronoun_pull")
    p_plan.add_argument("--sessions", type=int, default=3, help="Independent sessions per arm (>= 3 recommended).")
    p_plan.add_argument("--driver", default="scripted", choices=list(ldrv.DRIVER_FAMILIES))
    p_plan.add_argument("--model", default=None)
    p_plan.add_argument("--second-driver", default=None, choices=list(ldrv.DRIVER_FAMILIES))
    p_plan.add_argument("--second-model", default=None)
    p_plan.add_argument("--no-atlas", action="store_true")
    p_plan.add_argument("--bare-ablation", action="store_true")
    p_plan.add_argument("--max-gpu-seconds", type=float, default=600.0)
    p_plan.add_argument("--max-forwards", type=int, default=200)
    p_plan.add_argument("--max-cells", type=int, default=60)
    p_plan.add_argument("--max-wall-seconds", type=float, default=1800.0)
    p_plan.add_argument("--min-controls", type=int, default=3)
    p_plan.add_argument("--max-turns", type=int, default=24)
    p_plan.add_argument("--max-tokens", type=int, default=200000)
    p_plan.add_argument("--max-dollars", type=float, default=None)
    p_plan.add_argument("--claim-controls", type=int, default=None)
    p_plan.add_argument("--claim-n-target", type=int, default=None)
    p_plan.add_argument("--out", type=Path, default=None)
    p_plan.set_defaults(func=_cmd_plan)

    p_run = sub.add_parser("run", help="Run a Phase 6 study plan JSON.")
    p_run.add_argument("plan", type=Path)
    p_run.add_argument("--out", required=True, type=Path)
    p_run.add_argument("--weights-dir", type=Path, default=None)
    p_run.add_argument("--atlas-dir", type=Path, default=None)
    p_run.add_argument("--stop-file", type=Path, default=None)
    p_run.add_argument("--stand-in", action="store_true")
    p_run.add_argument("--no-probe", action="store_true")
    p_run.add_argument("--port", type=int, default=lmcp.SERVER_PORT)
    p_run.set_defaults(func=_cmd_run)

    p_kn = sub.add_parser("knobs", help="Run one owner fresh cheaper-knob claim reading.")
    p_kn.add_argument("--brief", required=True)
    p_kn.add_argument("--knob", required=True, type=Path)
    p_kn.add_argument("--bars", required=True, type=Path)
    p_kn.add_argument("--book", type=Path, default=None)
    p_kn.add_argument("--controls", type=int, default=None)
    p_kn.add_argument("--stand-in", action="store_true")
    p_kn.add_argument("--port", type=int, default=lmcp.SERVER_PORT)
    p_kn.set_defaults(func=_cmd_knobs)

    p_rb = sub.add_parser("rebuild", help="Emit and optionally verify a <= 200-word rebuild description.")
    p_rb.add_argument("--book", required=True, type=Path)
    p_rb.add_argument("--seq", required=True, type=int)
    p_rb.add_argument("--check", action="store_true")
    p_rb.add_argument("--out-book", type=Path, default=None)
    p_rb.add_argument("--stand-in", action="store_true")
    p_rb.add_argument("--port", type=int, default=lmcp.SERVER_PORT)
    p_rb.set_defaults(func=_cmd_rebuild)

    p_cp = sub.add_parser("cold-page", help="Render the <= 1-page cold-driver brief.")
    p_cp.add_argument("--brief", default="pronoun_pull")
    p_cp.set_defaults(func=_cmd_cold_page)

    p_rep = sub.add_parser("report", help="Model-free verification and report regeneration from study_book.jsonl.")
    p_rep.add_argument("study_dir", type=Path)
    p_rep.add_argument("--json", action="store_true")
    p_rep.set_defaults(func=_cmd_report)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
