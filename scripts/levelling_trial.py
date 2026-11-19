#!/usr/bin/env python3
"""The owner's command line for a trial of record (`scripts/levelling_trial.py`; `backend/level_trial.py`).

    .venv/bin/python -I scripts/levelling_trial.py reworded pronoun_pull --from my_items.json --out reworded.json
    .venv/bin/python -I scripts/levelling_trial.py register pronoun_pull --bars bars.json --edit edit.json \\
        --name she_ten --reworded reworded.json --out experiments/results/levelling_trials/she_ten/prereg
    .venv/bin/python -I scripts/levelling_trial.py verify experiments/results/levelling_trials/she_ten/prereg
    git add experiments/results/levelling_trials/she_ten/prereg && git commit    # a trial of record's folder is committed
    .venv/bin/python -I scripts/levelling_trial.py run experiments/results/levelling_trials/she_ten/prereg \\
        --out ~/levelling/she_ten_result

`reworded`: writes a reworded set with its hash. `--from FILE` takes the owner's own items (a JSON object with
"name", "items": [{"item_id", "half", "question", "answer"}, ...]) and checks them against the brief's own wording;
with no `--from` it writes the starter set that stands in the instrument's source, which is for checks and dry runs.

`register`: writes the registration folder. The bars must be those a calibration reading wrote into the book the trial
will be read in (`scripts/levelling.py calibrate`; `--book`, else the level book). Nothing is entered in the book.

`run`: reads the trial, once. It is of record only when this process has read nothing else (it is a new process),
the checkout is clean, the folder is committed as it stands, and the runner and reader are the files hashed at
registration; otherwise it is refused, in words that say which did not hold. `--dry-run` reads it all the same, as a
dry run: judged and counted as a trial is, spending a fresh reading of the brief, and never a verdict of record. On
the card it takes the card's lock and refuses while the page server answers.

With `--stand-in` (the float32 stand-in) or `--packed-stand-in` (the packed one, which has a folded form) the model is
a small random one on the CPU and `--book` must be given: a stand-in's readings do not go into the level book. A
trial on a stand-in shows that the plumbing runs; its figures mean nothing.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backend import level_book as lb  # noqa: E402

PORT = 8765


def _is_stand_in(a: argparse.Namespace) -> bool:
    return bool(getattr(a, "stand_in", False) or getattr(a, "packed_stand_in", False))


def _open_handle(stand_in: bool, port: int, *, packed_stand_in: bool = False) -> Tuple[Any, int, Any]:
    """`(handle, exit code, the card's lock)`: a stand-in on the CPU, or the 31B on the card with the card's lock taken
    (kept for the life of the process) and nothing answering on the page server's port."""
    if stand_in or packed_stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch

        torch.set_num_threads(1)
        from backend import levelling as lv
        from backend import weight_edits as we

        if packed_stand_in:
            import check_dequant as cd

            rt = cd.packed_stand_in()[0]
        else:
            from atlas_stand_in import stand_in as make_stand_in

            rt, _vault = make_stand_in()
        h = lv.handle(rt)
        we.assert_base(h.lm, h.hooks_at_load)
        return h, 0, None

    from backend import level_mcp as lmcp

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    words = lmcp.card_refusal(port)
    if words:
        print(f"Refused: {words}", file=sys.stderr)
        return None, 2, None
    try:
        lock = lmcp.hold_card()
    except SystemExit as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return None, 2, None
    import torch

    from backend import levelling as lv
    from backend import w4a16_fast
    from backend import weight_edits as we
    from backend.runtime_factory import make_runtime

    torch.set_grad_enabled(False)
    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    gc.collect()
    torch.cuda.empty_cache()
    w4a16_fast.install(rt, "compiled")
    h = lv.handle(rt)
    we.assert_base(h.lm, h.hooks_at_load)
    return h, 0, lock


def _resolve_brief(spec: str, tokenizer: Any) -> Any:
    from backend import level_briefs as lbr

    p = Path(spec)
    if p.is_file():
        return lbr.load_brief(p)
    starters = lbr.starter_briefs(tokenizer)
    if spec in starters:
        return starters[spec]
    raise lbr.BriefError(f'Unknown brief "{spec}". Give a JSON path or one of: {", ".join(sorted(starters))}.')


def _book(a: argparse.Namespace) -> Optional[Path]:
    """The book a command reads and writes: `--book`, else the level book; with a stand-in, `--book` or the
    environment's, never the level book by default."""
    if a.book:
        return Path(a.book).expanduser()
    if _is_stand_in(a) and not os.environ.get(lb.BOOK_ENV):
        raise ValueError("With a stand-in give --book: the book the bars were calibrated in. A stand-in's readings do "
                         "not go into the level book.")
    return None


def _refusals() -> Tuple[type, ...]:
    from backend import level_briefs as lbr
    from backend import level_claim as lcl
    from backend import level_trial as lt
    from backend import levelling as lv
    from backend import weight_edits as we

    return (lt.TrialRefused, lcl.ClaimRefused, lbr.BriefError, we.Refused, lv.Refused, lb.BookBroken, OSError, ValueError)


def cmd_reworded(a: argparse.Namespace) -> int:
    import levelling as cli
    from backend import level_trial as lt

    try:
        brief = _resolve_brief(a.brief, cli._resolve_tokenizer(_is_stand_in(a)))
        if a.source:
            raw = json.loads(Path(a.source).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise lt.TrialRefused(f"{a.source} is not a JSON object with \"name\" and \"items\".")
            raw = {**raw, "brief_id": brief.id, "brief_name": brief.name}
            raw.pop("reworded_id", None)  # the hash is worked here, from the items as they stand
            rset = lt.RewordedSet.from_dict(raw)
            lt.validate_reworded_against_brief(rset, brief)
        else:
            make = lt.STARTER_SETS.get(brief.name)
            if make is None:
                print(f'No starter reworded set is written for "{brief.name}": write your own items and give them '
                      "with --from.", file=sys.stderr)
                return 2
            rset = make(brief)
        where = lt.reworded_origin(rset, brief)
        if a.out:
            lt.save_reworded(rset, Path(a.out))
            print(f"wrote {a.out} (reworded_id={rset.short()}, {len(rset.items)} items; {where})")
        else:
            print(json.dumps({**rset.to_dict(), "reworded_id": rset.id}, indent=2, ensure_ascii=False))
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_register(a: argparse.Namespace) -> int:
    from backend import level_claim as lcl
    from backend import level_trial as lt
    from backend import weight_edits as we

    try:
        book_p = _book(a)
        if not a.reworded:
            raise lt.TrialRefused("A trial reads a reworded set: give --reworded FILE (see the reworded command).")
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    h, rc, _lock = _open_handle(bool(a.stand_in), int(a.port), packed_stand_in=bool(a.packed_stand_in))
    if rc != 0 or h is None:
        return rc
    try:
        brief = _resolve_brief(a.brief, h.tokenizer)
        bars = lcl.load_bars(Path(a.bars))
        edit_path = Path(a.edit)
        saved_dir: Optional[Path] = None
        if edit_path.is_dir():
            edit = we.load(edit_path, h.lm, model_id=h.model_id, weights_id=h.weights_id)
            saved_dir = edit_path
        else:
            recipe = json.loads(edit_path.read_text(encoding="utf-8"))
            got = we.from_recipe(recipe, edit_path.parent)
            if isinstance(got, (str, Path)):
                saved_dir = we._find_saved(got)
                edit = we.load(saved_dir, h.lm, model_id=h.model_id, weights_id=h.weights_id)
            else:
                edit = we.build(h.lm, got, model_id=h.model_id, weights_id=h.weights_id)
        shams = json.loads(Path(a.shams).read_text(encoding="utf-8")) if a.shams else None
        if shams is not None and not isinstance(shams, list):
            raise lt.TrialRefused("--shams is a JSON list of edit recipes.")
        reg = lt.register_trial(
            Path(a.out), h, brief, bars, edit,
            trial_name=str(a.name),
            reworded=lt.load_reworded(Path(a.reworded)),
            book_path=book_p,
            main_dose=float(a.dose),
            selection_mode=str(a.selection_mode),
            declared_rule=a.declared_rule,
            sham_recipes=shams,
            base_dir=Path(a.shams).parent if a.shams else Path("."),
            controls_per_family=int(a.controls),
            reworded_controls=a.reworded_controls,
            n_target=int(a.n_target),
            n_neutral=int(a.n_neutral),
            n_near=int(a.n_near),
            n_panel=int(a.n_panel),
            rows={"small": int(a.rows_small), "large": int(a.rows_large)},
            match_tolerance=float(a.match_tolerance),
            saved_edit_dir=saved_dir,
        )
        print(json.dumps(
            {"prereg_dir": reg["prereg_dir"], "trial_sha256": reg["trial_sha256"], "files": reg["files"],
             "edit_provenance": reg["trial"]["edit_provenance"], "reworded": reg["trial"]["reworded_origin"],
             "bars_calibration": reg["trial"]["bars_calibration"]},
            indent=2,
        ))
        if reg["trial"]["edit_provenance"] == "carried":
            print("note: this edit carries arrays. The folder holds their hashes only: keep the saved edit, give it "
                  "to the run with --payload, and expect no volunteer kit for it.", file=sys.stderr)
        print("next: commit the folder as it stands, then run it from a clean checkout (a trial of record), or read "
              "it now with --dry-run.", file=sys.stderr)
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_verify(a: argparse.Namespace) -> int:
    from backend import level_trial as lt

    try:
        v = lt.verify_registration(Path(a.prereg))
        cond = lt.record_conditions(Path(a.prereg), v["trial"])
        print(json.dumps({
            "ok": True,
            "prereg_dir": v["prereg_dir"],
            "trial_sha256": v["trial_sha256"],
            "trial_name": v["trial"]["trial_name"],
            "edit_id": v["trial"]["edit_id"],
            "edit_provenance": v["trial"]["edit_provenance"],
            "files": v["files"],
            # what a run from here would find, the freshness of its own process aside
            "would_not_be_of_record_because": [r for r in cond["reasons"] if not r.startswith("this process")],
        }, indent=2))
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1


def cmd_run(a: argparse.Namespace) -> int:
    from backend import level_trial as lt

    try:
        book_p = _book(a)
        lt.verify_registration(Path(a.prereg))
    except _refusals() as exc:
        print(f"Refused before load: {exc}", file=sys.stderr)
        return 2

    h, rc, _lock = _open_handle(bool(a.stand_in), int(a.port), packed_stand_in=bool(a.packed_stand_in))
    if rc != 0 or h is None:
        return rc
    try:
        res = lt.run_trial_of_record(
            h,
            Path(a.prereg),
            out_dir=Path(a.out),
            payload_dir=Path(a.payload) if a.payload else None,
            book_path=book_p,
            dry_run=bool(a.dry_run),
        )
        print(res["sheet"])
        return 0 if res["verdict"] != "void" else 1
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="levelling_trial.py", description="Register and read a trial of record.")
    sub = ap.add_subparsers(dest="command", required=True)

    rw = sub.add_parser("reworded", help="write a reworded set with its hash: your own items (--from) or the starter set")
    rw.add_argument("brief", help="starter brief name or JSON path")
    rw.add_argument("--from", dest="source", default=None, help="your own items, a JSON file with no hash in it yet")
    rw.add_argument("--out", default=None, help="output JSON path")
    rw.add_argument("--stand-in", action="store_true")
    rw.add_argument("--packed-stand-in", action="store_true")
    rw.set_defaults(fn=cmd_reworded)

    rg = sub.add_parser("register", help="write a trial's registration folder")
    rg.add_argument("brief", help="starter brief name or JSON path")
    rg.add_argument("--bars", required=True, help="the bar set a calibration reading wrote (JSON path)")
    rg.add_argument("--edit", required=True, help="edit recipe JSON path or saved edit folder")
    rg.add_argument("--name", required=True, help="trial name")
    rg.add_argument("--reworded", default=None, help="the reworded set (JSON path); a trial has one")
    rg.add_argument("--out", required=True, help="registration folder path (must not exist or be empty)")
    rg.add_argument("--book", default=None, help="the book the bars were calibrated in and the trial will be read in")
    rg.add_argument("--dose", type=float, default=1.0)
    rg.add_argument("--selection-mode", default="free_pick", choices=["free_pick", "declared_rule"])
    rg.add_argument("--declared-rule", default=None)
    rg.add_argument("--shams", default=None, help="JSON list of sham edit recipes (with --selection-mode declared_rule)")
    rg.add_argument("--controls", type=int, default=99)
    rg.add_argument("--reworded-controls", type=int, default=None)
    rg.add_argument("--n-target", type=int, default=16)
    rg.add_argument("--n-neutral", type=int, default=8)
    rg.add_argument("--n-near", type=int, default=8)
    rg.add_argument("--n-panel", type=int, default=8)
    rg.add_argument("--rows-small", type=int, default=12)
    rg.add_argument("--rows-large", type=int, default=32)
    rg.add_argument("--match-tolerance", type=float, default=2.0)
    rg.add_argument("--stand-in", action="store_true")
    rg.add_argument("--packed-stand-in", action="store_true")
    rg.add_argument("--port", type=int, default=PORT)
    rg.set_defaults(fn=cmd_register)

    vf = sub.add_parser("verify", help="check a registration folder against itself, with no model")
    vf.add_argument("prereg", help="registration folder path")
    vf.set_defaults(fn=cmd_verify)

    rn = sub.add_parser("run", help="read a registered trial, once")
    rn.add_argument("prereg", help="registration folder path")
    rn.add_argument("--out", required=True, help="output folder for trial_result.json and TRIAL_SHEET.md")
    rn.add_argument("--payload", default=None, help="saved edit folder (for an edit that carries arrays)")
    rn.add_argument("--book", default=None, help="the book the trial was registered against")
    rn.add_argument("--dry-run", action="store_true",
                    help="read it as a dry run: counted as a trial is, never a verdict of record")
    rn.add_argument("--stand-in", action="store_true")
    rn.add_argument("--packed-stand-in", action="store_true")
    rn.add_argument("--port", type=int, default=PORT)
    rn.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
