#!/usr/bin/env python3
"""Owner's command line for Levelling Phase 6: driver evaluation ("Whether agents help"; `scripts/levelling_eval.py`).

Subcommands:
- `plan`: build and validate a Phase 6 multi-session study plan (reach ladder + matched-budget cheaper-knobs arm +
  optional second driver family and `--bare` ablation), and print its bounds. A plan that does not validate (a claim
  design that would read nothing, such as `--claim-controls 0`, among the rest) is refused and nothing is written.
- `run`: run a study plan JSON (`--stand-in` on CPU, or the live 31B on the card under `gpu.lock`). The plan is read
  and validated, its brief found (a starter brief by name, or the brief file it names, read), and the study folder
  checked unused and free of a STOP file, before the model is loaded or the lock taken.
- `knobs`: one owner cheaper-knob fresh claim reading (prompts, decoding, and/or transient steering), entered in the
  book named by `--book` (required: it never falls back to the default book). The knob, the bars and the brief are
  read, and `--controls` below 1 refused, before the model is loaded. With `--stand-in` it writes nothing in the
  working folder, and the default book is refused: a stand-in reading is never entered there.
- `rebuild`: print a rebuild description of a recorded paired reading (at most 200 words, each word and number of its
  spec counted, and at most 2,400 characters), and with `--replay` read it again in this process and compare. That
  is a same-process replay, not the ruling's fresh-session rebuild test, which has not been run. A `--stand-in`
  replay into the default book is refused.
- `cold-page`: print the one-page sheet a cold driver would be given, and whether it names a site. The ruling's
  cold-driver test has not been run.
- `report`: reload a study from `<study_dir>/study_book.jsonl` with no model, check every side book against it, and
  print its report.

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
from backend import level_book as lb  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_drivers as ldrv  # noqa: E402
from backend import level_eval as lev  # noqa: E402
from backend import level_knobs as lkn  # noqa: E402
from backend import level_mcp as lmcp  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend.weight_reader import export_stand_in_safetensors  # noqa: E402


def _load_handle(stand_in: bool, export_dir: Optional[Path], *, port: int = lmcp.SERVER_PORT) -> tuple[Any, Optional[Path], Optional[Path], Any]:
    """Load either the CPU stand-in or the live 31B runtime (taking gpu.lock on the card). The stand-in's weights and
    atlas index are exported into `export_dir` only when one is given (a study's reaches need them); with None nothing
    is written."""
    if stand_in:
        import torch
        from scripts.atlas_stand_in import stand_in as make_stand_in, use_stand_in_atlas

        torch.set_num_threads(1)
        use_stand_in_atlas()
        rt, _vault = make_stand_in()
        h = lv.handle(rt)
        if export_dir is None:
            return h, None, None, None
        wdir = export_stand_in_safetensors(h.lm, export_dir / "_stand_in_weights", model_id=h.model_id)
        adir = la.build_stand_in_atlas_index(h.lm, h.W_U, h.tokenizer, export_dir / "_stand_in_atlas", model_id=h.model_id)
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


def _refuse_default_book_on_stand_in(stand_in: bool, book: Path, what: str) -> None:
    """A stand-in reading is never entered in the default book: that book counts the owner's readings of the model."""
    if stand_in and Path(book).expanduser().resolve() == lb.default_book_path().expanduser().resolve():
        raise SystemExit(f"refused: {what} {book} is the default book, and a --stand-in reading is never entered there; "
                         "name a side book.")


def _brief_before_load(brief_arg: str) -> Optional[lbr.Brief]:
    """The brief `brief_arg` names, found before anything is loaded: a brief file, read (returned), or a starter
    brief's name, checked (None: a starter brief is built with the model's tokenizer once it is loaded)."""
    p = Path(brief_arg)
    if p.is_file():
        try:
            return lbr.load_brief(p)
        except lbr.BriefError as exc:
            raise SystemExit(f"refused: {exc}") from None
    names = sorted(lbr.starter_briefs(None))
    if brief_arg not in names:
        raise SystemExit(f"refused: {brief_arg!r} is neither a brief file nor a starter brief ({', '.join(names)}).")
    return None


def _validation_lines(val: dict) -> List[str]:
    lines = [
        f"{val['arms']} arms x {val['sessions_per_arm']} sessions = {val['total_sessions']} sessions; "
        f"study sha256 {val['study_sha256'][:12]}",
        f"upper bounds: {val['upper_bound_gpu_seconds']:g} GPU-s, {val['upper_bound_practice_forwards']} practice forwards, "
        f"{val['upper_bound_wall_seconds']:g} wall s (cap {val['max_total_wall_seconds']:g}), "
        f"dollars {val['upper_bound_dollars'] if val['upper_bound_dollars'] is not None else 'not capped'}",
    ]
    if not val.get("paths_checked"):
        lines.append("reach folders not checked here (no --weights-dir or --atlas-dir); `run` checks them")
    lines += [f"warning: {w}" for w in val.get("warnings") or []]
    return lines


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
        max_total_dollars=args.max_total_dollars,
        max_total_wall_seconds=args.max_total_wall_seconds,
    )
    check_paths = args.weights_dir is not None or args.atlas_dir is not None
    try:
        val = lev.validate_study_plan(plan, weights_dir=args.weights_dir, atlas_dir=args.atlas_dir, check_paths=check_paths)
    except (lev.EvalRefused, lses.SessionRefused) as exc:
        raise SystemExit(f"refused: {exc}") from None
    doc = plan.canonical_dict()
    doc["study_sha256"] = plan.sha256
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote study plan to {out_p}")
    else:
        print(json.dumps(doc, indent=2))
    for line in _validation_lines(val):
        print(line, file=sys.stderr if not args.out else sys.stdout)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        raw = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refused: cannot read the plan {args.plan}: {exc}") from None
    out_dir = Path(args.out).resolve()
    stop_file = Path(args.stop_file) if args.stop_file else (out_dir / "STOP")
    wdir_arg = Path(args.weights_dir).resolve() if args.weights_dir else None
    adir_arg = Path(args.atlas_dir).resolve() if args.atlas_dir else None
    # Everything that can be checked is checked before the model is loaded or the card's lock taken
    try:
        plan = lev.study_plan_from_dict(raw)
        lev.validate_study_plan(plan, weights_dir=wdir_arg, atlas_dir=adir_arg, check_paths=False)
        if plan.brief_path:
            lbr.load_brief(Path(plan.brief_path))  # read now: a missing or broken brief file is refused here
        elif plan.brief_name not in lbr.starter_briefs(None):
            raise lev.EvalRefused(f"Unknown brief_name {plan.brief_name!r}; supply brief_path or choose from "
                                  f"{', '.join(sorted(lbr.starter_briefs(None)))}.")
        if wdir_arg is not None and not wdir_arg.is_dir():
            raise lev.EvalRefused(f"--weights-dir {wdir_arg} is not a folder.")
        if adir_arg is not None:
            la.resolve_atlas_index_dir(adir_arg)
        lev.refuse_used_study_dir(out_dir)
    except (lev.EvalRefused, lses.SessionRefused, la.AtlasReaderError, lbr.BriefError) as exc:
        raise SystemExit(f"refused: {exc}") from None
    if stop_file.exists():
        raise SystemExit(f"refused: a STOP file is at {stop_file}; remove it to run the study (nothing was started).")
    out_dir.mkdir(parents=True, exist_ok=True)
    h, auto_w, auto_a, lock_f = _load_handle(bool(args.stand_in), out_dir if args.stand_in else None, port=int(args.port))
    wdir = wdir_arg or auto_w
    adir = adir_arg or auto_a
    ks = ldrv.KillSwitch(stop_file=stop_file)
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
    except (lev.EvalRefused, ldrv.DriverRefused, lses.SessionRefused, lbr.BriefError, lcl.ClaimRefused) as exc:
        raise SystemExit(f"refused: {exc}") from None
    finally:
        if lock_f is not None:
            lock_f.close()
    print(res["report"])
    return 0


def _cmd_knobs(args: argparse.Namespace) -> int:
    # What can be read and refused without the model is, before it is loaded or the card's lock taken
    try:
        knob_raw = json.loads(Path(args.knob).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refused: cannot read the knob {args.knob}: {exc}") from None
    if args.controls is not None and int(args.controls) < 1:
        raise SystemExit(f"refused: --controls is a whole number of 1 or more control edits a family; {args.controls} "
                         "is not (a claim with no control edits is refused).")
    book = Path(args.book).resolve()
    _refuse_default_book_on_stand_in(bool(args.stand_in), book, "--book")
    try:
        bars = lcl.load_bars(Path(args.bars))
    except lcl.ClaimRefused as exc:
        raise SystemExit(f"refused: {exc}") from None
    brief_file = _brief_before_load(str(args.brief))
    if brief_file is not None and bars.brief_id != brief_file.id:
        raise SystemExit(f"refused: BarSet {bars.short()} is for brief {bars.brief_id[:12]}, not {brief_file.short()}.")
    h, _w, _a, lock_f = _load_handle(bool(args.stand_in), None, port=int(args.port))
    try:
        brief = brief_file if brief_file is not None else lbr.starter_briefs(h.tokenizer)[args.brief]
        res = lkn.run_fresh_knob_claim(
            h,
            brief,
            bars,
            knob_raw,
            base_dir=Path(args.knob).resolve().parent,
            book_path=book,
            controls_override=args.controls,
        )
    except (lkn.KnobRefused, lbr.BriefError, lcl.ClaimRefused, KeyError, OSError) as exc:
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
    print(f"\n[{desc['word_count']} words counted with every word and number of the spec (at most "
          f"{lev.MAX_REBUILD_WORDS}), {desc['characters']} characters (at most {lev.MAX_REBUILD_CHARS}), "
          f"sha256={desc['sha256'][:12]}]")
    if args.replay:
        out_book = Path(args.out_book) if args.out_book else (Path(args.book).resolve().parent / "rebuild_replay_book.jsonl")
        _refuse_default_book_on_stand_in(bool(args.stand_in), out_book, "--out-book")
        h, _w, _a, lock_f = _load_handle(bool(args.stand_in), None, port=int(args.port))
        try:
            chk = lev.run_rebuild_replay(h, desc["description"], Path(args.book), int(args.seq), out_book=out_book)
        except lev.EvalRefused as exc:
            raise SystemExit(f"refused: {exc}") from None
        finally:
            if lock_f is not None:
                lock_f.close()
        print(json.dumps({k: v for k, v in chk.items() if k != "comparisons"}, indent=2))
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
    if page["prompt_named_site"]:
        print("This page names a site (" + (", ".join(page["sites_named_in_page"]) or "the brief says so")
              + "): it is not a cold page.")
    print(page["site_finder"])
    print(page["ruling_test"])
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
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Levelling Phase 6 owner CLI: driver evaluation, reach ladders, and cheaper knobs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Build, validate and bound a Phase 6 multi-session study plan JSON.")
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
    p_plan.add_argument("--max-total-dollars", type=float, default=None)
    p_plan.add_argument("--max-total-wall-seconds", type=float, default=None,
                        help="The study's wall cap; by default what every planned session needs at its wall cap.")
    p_plan.add_argument("--claim-controls", type=int, default=None)
    p_plan.add_argument("--claim-n-target", type=int, default=None)
    p_plan.add_argument("--weights-dir", type=Path, default=None, help="Check the weights reaches against this folder.")
    p_plan.add_argument("--atlas-dir", type=Path, default=None, help="Check the atlas reach against this index.")
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
    p_kn.add_argument("--book", required=True, type=Path, help="The book the reading is entered in (required).")
    p_kn.add_argument("--controls", type=int, default=None)
    p_kn.add_argument("--stand-in", action="store_true")
    p_kn.add_argument("--port", type=int, default=lmcp.SERVER_PORT)
    p_kn.set_defaults(func=_cmd_knobs)

    p_rb = sub.add_parser("rebuild", help="Print a <= 200-word rebuild description, and optionally replay it here.")
    p_rb.add_argument("--book", required=True, type=Path)
    p_rb.add_argument("--seq", required=True, type=int)
    p_rb.add_argument("--replay", action="store_true",
                      help="Read the description again in this process and compare (not the ruling's fresh-session test).")
    p_rb.add_argument("--out-book", type=Path, default=None)
    p_rb.add_argument("--stand-in", action="store_true")
    p_rb.add_argument("--port", type=int, default=lmcp.SERVER_PORT)
    p_rb.set_defaults(func=_cmd_rebuild)

    p_cp = sub.add_parser("cold-page", help="Render the one-page sheet a cold driver would be given.")
    p_cp.add_argument("--brief", default="pronoun_pull")
    p_cp.set_defaults(func=_cmd_cold_page)

    p_rep = sub.add_parser("report", help="Model-free check and report of a study from its study_book.jsonl.")
    p_rep.add_argument("study_dir", type=Path)
    p_rep.add_argument("--json", action="store_true")
    p_rep.set_defaults(func=_cmd_report)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
