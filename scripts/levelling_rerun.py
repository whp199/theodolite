#!/usr/bin/env python3
"""A volunteer's re-run of a trial, and its intake (`scripts/levelling_rerun.py`; `backend/level_rerun.py`).

At home, after a trial has been read:
    .venv/bin/python -I scripts/levelling_rerun.py pack PREREG --home-result RESULT/trial_result.json --out KIT
    .venv/bin/python -I scripts/levelling_rerun.py verify-kit KIT
On the volunteer's machine (the 31B on the card, the page server stopped):
    .venv/bin/python -I scripts/levelling_rerun.py run KIT --out my_return.tar --volunteer NAME
At home again:
    .venv/bin/python -I scripts/levelling_rerun.py intake my_return.tar --prereg PREREG --book HOME_BOOK --out STAGE

`pack` needs the home trial's result: the kit carries the seed of the home draw, so that the volunteer reads the home
items. It also carries the brief and the reworded set in clear; a set that has gone out in a kit is spent as a thing
kept from a reader. `run` reads the kit into a book of its own (a re-run is never of record) and packs four files into
a plain tar. `intake` judges the returned BOOK against the home book, both judged again by the home's code, and
writes `intake_verdict.json`: rejected, disagreed, or accepted (same_figures or within_tolerance). `--book` is the
book the home trial was read in (the level book when not given).

What an intake cannot tell: a re-run from the home book's own lines copied into a new book. Keep the home book
unpublished until the returns are in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PORT = 8765


def _refusals() -> tuple:
    from backend import level_book as lb
    from backend import level_rerun as lr
    from backend import level_trial as lt

    return (lr.RerunRefused, lt.TrialRefused, lb.BookBroken, OSError, ValueError, KeyError, TypeError)


def cmd_pack(a: argparse.Namespace) -> int:
    from backend import level_rerun as lr

    try:
        home_res = json.loads(Path(a.home_result).read_text(encoding="utf-8"))
        out = lr.make_rerun_kit(Path(a.prereg), Path(a.out), home_res)
        print(json.dumps({k: v for k, v in out.items() if k != "kit"}, indent=2))
        print("note: the kit carries the brief and the reworded set in clear.", file=sys.stderr)
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_verify_kit(a: argparse.Namespace) -> int:
    from backend import level_rerun as lr

    try:
        v = lr.verify_rerun_kit(Path(a.kit))
        print(json.dumps({
            "ok": True,
            "kit_dir": v["kit_dir"],
            "kit_sha256": v["kit_sha256"],
            "trial_sha256": v["trial_sha256"],
            "trial_name": v["kit"]["trial_name"],
            "edit_id": v["kit"]["edit_id"],
        }, indent=2))
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1


def cmd_run(a: argparse.Namespace) -> int:
    import levelling_trial as lt_cli
    from backend import level_rerun as lr

    try:
        lr.verify_rerun_kit(Path(a.kit))
    except _refusals() as exc:
        print(f"Refused before load: {exc}", file=sys.stderr)
        return 2

    h, rc, _lock = lt_cli._open_handle(bool(a.stand_in), int(a.port), packed_stand_in=bool(a.packed_stand_in))
    if rc != 0 or h is None:
        return rc
    try:
        res = lr.run_volunteer_rerun(h, Path(a.kit), Path(a.out), volunteer_name=str(a.volunteer))
        print(json.dumps({
            "bundle_tar": res["bundle_tar"],
            "bundle_sha256": res["bundle_sha256"],
            "kit_sha256": res["kit_sha256"],
            "verdict": res["verdict"],
        }, indent=2))
        return 0 if res["verdict"] != "void" else 1
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_intake(a: argparse.Namespace) -> int:
    from backend import level_book as lb
    from backend import level_rerun as lr

    try:
        book = Path(a.book).expanduser() if a.book else lb.book_path()
        verdict = lr.intake_rerun_bundle(Path(a.bundle), Path(a.prereg), book, Path(a.out),
                                         allow_dirty=bool(a.allow_dirty),
                                         allow_other_runner=bool(a.allow_other_runner))
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["outcome"] == "accepted" else 1
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="levelling_rerun.py", description="A volunteer's re-run of a trial, and its intake.")
    sub = ap.add_subparsers(dest="command", required=True)

    pk = sub.add_parser("pack", help="pack a registration that has been read at home into a kit (rebuilt edits only)")
    pk.add_argument("prereg", help="the home registration folder")
    pk.add_argument("--home-result", required=True, help="the home trial's trial_result.json (it holds the draw's seed)")
    pk.add_argument("--out", required=True, help="output kit folder")
    pk.set_defaults(fn=cmd_pack)

    vk = sub.add_parser("verify-kit", help="check a kit against itself, with no model")
    vk.add_argument("kit", help="re-run kit folder")
    vk.set_defaults(fn=cmd_verify_kit)

    rn = sub.add_parser("run", help="read a kit on this machine's model and pack the return")
    rn.add_argument("kit", help="re-run kit folder")
    rn.add_argument("--out", required=True, help="output .tar bundle path")
    rn.add_argument("--volunteer", default="anonymous")
    rn.add_argument("--stand-in", action="store_true")
    rn.add_argument("--packed-stand-in", action="store_true")
    rn.add_argument("--port", type=int, default=PORT)
    rn.set_defaults(fn=cmd_run)

    ik = sub.add_parser("intake", help="unpack a returned bundle and judge its book against the home book")
    ik.add_argument("bundle", help="returned .tar bundle path")
    ik.add_argument("--prereg", required=True, help="the home registration folder")
    ik.add_argument("--book", default=None, help="the book the home trial was read in (the level book when not given)")
    ik.add_argument("--out", required=True, help="a fresh, empty staging folder")
    ik.add_argument("--allow-dirty", action="store_true",
                    help="take a re-run whose checkout had a tracked file changed (the verdict says leave was given)")
    ik.add_argument("--allow-other-runner", action="store_true",
                    help="take a re-run whose runner or reader was not the one registered (the verdict says so)")
    ik.set_defaults(fn=cmd_intake)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
