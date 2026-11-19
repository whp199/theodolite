#!/usr/bin/env python3
"""Owner CLI for Levelling Phase 5: Trial of Record (`scripts/levelling_trial.py`).

Creates hash-verified house registration folders (`prereg/`), verifies `SHA256SUMS`, and executes one
fresh-process Trial of Record with live base checksums, folded weight verification, and owner-written reworded sets.

    .venv/bin/python -I scripts/levelling_trial.py reworded pronoun_pull --out reworded.json --stand-in
    .venv/bin/python -I scripts/levelling_trial.py register pronoun_pull --bars bars.json --edit edit.json \\
        --name pronoun_pull_she10 --reworded reworded.json --out /tmp/trial/prereg --stand-in
    .venv/bin/python -I scripts/levelling_trial.py verify /tmp/trial/prereg
    .venv/bin/python -I scripts/levelling_trial.py run /tmp/trial/prereg --out /tmp/trial/result --stand-in
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backend import level_book as lb  # noqa: E402

PORT = 8765
MIN_FREE_GIB = 22.0


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _open_handle(stand_in: bool, port: int, *, packed_stand_in: bool = False) -> Tuple[Any, int]:
    if stand_in:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch

        torch.set_num_threads(1)
        from backend import levelling as lv
        from backend import weight_edits as we

        if packed_stand_in:
            import check_dequant as cd

            rt = cd.packed_stand_in()[0]
            h = lv.handle(rt)
            we.assert_base(h.lm, h.hooks_at_load)
            return h, 0
        from atlas_stand_in import stand_in as make_stand_in

        rt, _vault = make_stand_in()
        h = lv.handle(rt)
        we.assert_base(h.lm, h.hooks_at_load)
        return h, 0

    if _port_open(port):
        print(f"Refused: Theodolite server answers on :{port}; stop it before running a Trial of Record.", file=sys.stderr)
        return None, 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    if not torch.cuda.is_available():
        print("Refused: CUDA is unavailable (use --stand-in for a CPU dry run).", file=sys.stderr)
        return None, 2
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    if free_gib < MIN_FREE_GIB:
        print(f"Refused: only {free_gib:.2f} GiB free on GPU; Trial of Record requires >= {MIN_FREE_GIB:g} GiB.", file=sys.stderr)
        return None, 2
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
    return h, 0


def _resolve_brief(spec: str, tokenizer: Any) -> Any:
    from backend import level_briefs as lbr

    p = Path(spec)
    if p.is_file():
        return lbr.load_brief(p)
    starters = lbr.starter_briefs(tokenizer)
    if spec in starters:
        return starters[spec]
    raise lbr.BriefError(f'Unknown brief "{spec}". Give a JSON path or one of: {", ".join(sorted(starters))}.')


def cmd_reworded(a: argparse.Namespace) -> int:
    from backend import level_trial as lt

    h, rc = _open_handle(bool(a.stand_in), int(a.port))
    if rc != 0 or h is None:
        return rc
    brief = _resolve_brief(a.brief, h.tokenizer)
    if brief.name != "pronoun_pull":
        print(f'No built-in starter reworded set for "{brief.name}"; write a JSON file matching RewordedSet.', file=sys.stderr)
        return 2
    rset = lt.starter_reworded_pronoun_pull(brief)
    if a.out:
        lt.save_reworded(rset, Path(a.out))
        print(f"wrote {a.out} (reworded_id={rset.short()}, {len(rset.items)} items)")
    else:
        print(json.dumps({**rset.to_dict(), "reworded_id": rset.id}, indent=2, ensure_ascii=False))
    return 0


def cmd_register(a: argparse.Namespace) -> int:
    from backend import level_claim as lcl
    from backend import level_trial as lt
    from backend import weight_edits as we

    h, rc = _open_handle(bool(a.stand_in), int(a.port), packed_stand_in=bool(a.packed_stand_in))
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

        shams = None
        if a.shams:
            raw_sh = json.loads(Path(a.shams).read_text(encoding="utf-8"))
            shams = []
            for r in raw_sh:
                g = we.from_recipe(r, Path(a.shams).parent)
                se = we.load(g, h.lm, model_id=h.model_id, weights_id=h.weights_id) if isinstance(g, (str, Path)) else we.build(
                    h.lm, g, model_id=h.model_id, weights_id=h.weights_id,
                )
                shams.append(se)

        reworded = lt.load_reworded(Path(a.reworded)) if a.reworded else None
        rows_cfg = {"small": int(a.rows_small), "large": int(a.rows_large)}
        reg = lt.register_trial(
            Path(a.out), brief, bars, edit,
            trial_name=str(a.name),
            main_dose=float(a.dose),
            selection_mode=str(a.selection_mode),
            declared_rule=a.declared_rule,
            sham_edits=shams,
            reworded=reworded,
            controls_per_family=int(a.controls),
            reworded_controls=a.reworded_controls,
            n_target=int(a.n_target),
            n_neutral=int(a.n_neutral),
            n_near=int(a.n_near),
            n_panel=int(a.n_panel),
            rows=rows_cfg,
            folded_required=not bool(a.no_folded_required),
            saved_edit_dir=saved_dir,
        )
        print(json.dumps(
            {"prereg_dir": reg["prereg_dir"], "trial_sha256": reg["trial_sha256"], "files": reg["files"]},
            indent=2,
        ))
        return 0
    except (lt.TrialRefused, lcl.ClaimRefused, we.Refused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_verify(a: argparse.Namespace) -> int:
    from backend import level_trial as lt

    try:
        v = lt.verify_registration(Path(a.prereg))
        print(json.dumps({
            "ok": True,
            "prereg_dir": v["prereg_dir"],
            "trial_sha256": v["trial_sha256"],
            "trial_name": v["trial"]["trial_name"],
            "edit_id": v["trial"]["edit_id"],
            "edit_provenance": v["trial"]["edit_provenance"],
            "files": v["files"],
        }, indent=2))
        return 0
    except lt.TrialRefused as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1


def cmd_run(a: argparse.Namespace) -> int:
    from backend import level_trial as lt

    try:
        lt.verify_registration(Path(a.prereg))
    except lt.TrialRefused as exc:
        print(f"Refused before load: {exc}", file=sys.stderr)
        return 2

    h, rc = _open_handle(bool(a.stand_in), int(a.port), packed_stand_in=bool(a.packed_stand_in))
    if rc != 0 or h is None:
        return rc
    book_p = Path(a.book) if a.book else (
        Path(tempfile.mkdtemp(prefix="trial-standin-")) / "trial_book.jsonl"
        if (a.stand_in and not os.environ.get(lb.BOOK_ENV)) else None
    )
    try:
        res = lt.run_trial_of_record(
            h,
            Path(a.prereg),
            out_dir=Path(a.out),
            payload_dir=Path(a.payload) if a.payload else None,
            book_path=book_p,
            require_book_calibration=bool(a.require_book_calibration),
        )
        print(res["sheet"])
        return 0 if res["verdict"] != "void" else 1
    except (lt.TrialRefused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="levelling_trial.py", description="Phase 5 Trial of Record registration and fresh-process execution.")
    sub = ap.add_subparsers(dest="command", required=True)

    rw = sub.add_parser("reworded", help="export the built-in starter reworded set for pronoun_pull")
    rw.add_argument("brief", help="starter brief name or JSON path")
    rw.add_argument("--out", default=None, help="output JSON path")
    rw.add_argument("--stand-in", action="store_true")
    rw.add_argument("--port", type=int, default=PORT)
    rw.set_defaults(fn=cmd_reworded)

    rg = sub.add_parser("register", help="create a hash-verified prereg/ registration folder")
    rg.add_argument("brief", help="starter brief name or JSON path")
    rg.add_argument("--bars", required=True, help="hashed BarSet JSON path")
    rg.add_argument("--edit", required=True, help="edit recipe JSON path or saved edit folder")
    rg.add_argument("--name", required=True, help="trial name")
    rg.add_argument("--out", required=True, help="registration folder path (must not exist or be empty)")
    rg.add_argument("--dose", type=float, default=1.0)
    rg.add_argument("--selection-mode", default="free_pick", choices=["free_pick", "declared_rule"])
    rg.add_argument("--declared-rule", default=None)
    rg.add_argument("--shams", default=None, help="JSON list of sham edit recipes")
    rg.add_argument("--reworded", default=None, help="owner-written RewordedSet JSON path")
    rg.add_argument("--controls", type=int, default=99)
    rg.add_argument("--reworded-controls", type=int, default=None)
    rg.add_argument("--n-target", type=int, default=16)
    rg.add_argument("--n-neutral", type=int, default=8)
    rg.add_argument("--n-near", type=int, default=8)
    rg.add_argument("--n-panel", type=int, default=8)
    rg.add_argument("--rows-small", type=int, default=12)
    rg.add_argument("--rows-large", type=int, default=32)
    rg.add_argument("--no-folded-required", action="store_true")
    rg.add_argument("--stand-in", action="store_true")
    rg.add_argument("--packed-stand-in", action="store_true")
    rg.add_argument("--port", type=int, default=PORT)
    rg.set_defaults(fn=cmd_register)

    vf = sub.add_parser("verify", help="verify SHA256SUMS and integrity of a prereg/ folder off-GPU")
    vf.add_argument("prereg", help="registration folder path")
    vf.set_defaults(fn=cmd_verify)

    rn = sub.add_parser("run", help="run a registered Trial of Record in a fresh process")
    rn.add_argument("prereg", help="registration folder path")
    rn.add_argument("--out", required=True, help="output folder for trial_result.json and TRIAL_SHEET.md")
    rn.add_argument("--payload", default=None, help="saved edit folder (required only for carried edits)")
    rn.add_argument("--book", default=None, help="level book path override")
    rn.add_argument("--require-book-calibration", action="store_true")
    rn.add_argument("--stand-in", action="store_true")
    rn.add_argument("--packed-stand-in", action="store_true")
    rn.add_argument("--port", type=int, default=PORT)
    rn.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
