#!/usr/bin/env python3
"""The owner's command line for exporting an edit (`scripts/levelling_export.py`; `backend/level_export.py`).

Writes an edit as a llama.cpp LoRA `.gguf` adapter with a `.manifest.json` beside it, outside every git checkout
and never through an agent's tools. It runs on the CPU and loads no model.

    .venv/bin/python -I scripts/levelling_export.py lora --trial PREREG --trial-result RESULT/trial_result.json
    .venv/bin/python -I scripts/levelling_export.py lora --trial PREREG --gguf MODEL.gguf -o ~/exports/she_ten.gguf
    .venv/bin/python -I scripts/levelling_export.py lora --saved EDIT_ID_OR_FOLDER --weights-dir CHECKPOINT
    .venv/bin/python -I scripts/levelling_export.py verify ADAPTER.gguf --trial PREREG --gguf MODEL.gguf
    .venv/bin/python -I scripts/levelling_export.py show ADAPTER.gguf
    .venv/bin/python -I scripts/levelling_export.py smoke ADAPTER.gguf --gguf MODEL.gguf --llama-server BIN --prompt "..."

The edit: a trial's registration folder (`--trial`, with `--payload FOLDER` when it carries arrays) or a saved edit
(`--saved`), each rebuilt with no model loaded and kept to the id it names. That shows the terms are the id's and no
more, so an edit rebuilt this way is exported only when it also meets the kernel's caps that need no weights and a
level book holds a reading of it (`--book`, or the level book): the kernel has then gated it on a model once. A bare
recipe file (`--edit`) is taken only with `--stand-in`, where there is a model to build and gate it on.

The base the adapter is compiled against: `--gguf FILE` (the GGUF it will be served over), `--weights-dir FOLDER` (a
checkpoint), or, with neither, this machine's cached W4A16 checkpoint. llama.cpp serves a GGUF: an adapter compiled
against a checkpoint is exact to weights it will not be served over, its keys say so (`levelling.serving`), and what
it does over a GGUF is to be measured there. Give `--gguf` to compile against the file that will be served.

`--trial-result FILE` puts a trial's verdict into the adapter. The verdict is read from the trial's BOOK (`--book`,
or the path the result names), which must verify and hold the trial's verdict for this edit at this dose; the result
file only says where to look. With `--trial PREREG` the result must be that registration's, and the book's readings
are judged again with it before the verdict is written. The adapter says whether the trial was of record, a dry run
or a re-run, and whether the base it read is the base served. The implant label is the one worked at the trial's
registration (the model was at hand then), taken only for the registered edit at the registered dose; an export
with no registration says "not checked".

`verify` reads an adapter back; with the edit and the base it also shows the file is that edit on that base. `smoke`
loads it in a local llama-server and writes one prompt with it off and on: that llama.cpp takes the file, and no
measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _refusals() -> Tuple[type, ...]:
    from backend import level_book as lb
    from backend import level_export as lex
    from backend import level_trial as lt
    from backend import levelling as lv
    from backend import weight_edits as we

    return (lex.ExportRefused, lt.TrialRefused, we.Refused, lv.Refused, lb.BookBroken, OSError, ValueError, TypeError,
            KeyError)


def _load_edit_and_target(a: argparse.Namespace) -> Tuple[Any, Any, Optional[Path], Any, Any, Any]:
    """`(edit, target, saved_dir, lm, W_U, registration)`: the edit as the kernel hashed it, the base to compile
    against, and, with `--stand-in`, the stand-in model (for the implant label)."""
    import torch

    torch.set_num_threads(1)
    from backend import level_export as lex
    from backend import level_trial as lt
    from backend import levelling as lv
    from backend import weight_edits as we

    lm_ref = W_U_ref = h = None
    if a.stand_in:
        from atlas_stand_in import stand_in as make_stand_in

        rt, _vault = make_stand_in()
        h = lv.handle(rt)
        lm_ref, W_U_ref = h.lm, h.W_U

    if a.gguf:
        target = lex.TargetBase(gguf_path=Path(a.gguf))
    elif a.weights_dir:
        target = lex.TargetBase(weights_dir=Path(a.weights_dir))
    elif lm_ref is not None:
        target = lex.TargetBase(lm=lm_ref, label="stand-in", weights_id=h.weights_id)
    else:
        target = lex.TargetBase.default_checkpoint()

    given = [x for x in (a.trial, a.saved, a.edit) if x]
    if len(given) != 1:
        raise lex.ExportRefused("Give one of --trial PREREG, --saved EDIT, or (with --stand-in) --edit RECIPE.")
    payload = Path(a.payload) if getattr(a, "payload", None) else None
    if a.trial:
        reg = lt.verify_registration(Path(a.trial))
        if h is not None:
            edit = lt.load_registered_edit(h, reg["edit_manifest"], payload_dir=payload)
        else:
            edit = lex.edit_from_registration(reg["edit_manifest"], payload_dir=payload)
        saved_dir = we._find_saved(payload if payload is not None else edit.id) if not reg["edit_manifest"].get("all_rebuilt") else None
        return edit, target, saved_dir, lm_ref, W_U_ref, reg
    if a.saved:
        saved_dir = we._find_saved(a.saved)
        edit = (we.load(saved_dir, lm_ref, model_id=h.model_id, weights_id=h.weights_id) if h is not None
                else lex.edit_from_saved(saved_dir))
        return edit, target, saved_dir, lm_ref, W_U_ref, None
    if h is None:
        raise lex.ExportRefused(
            "A bare recipe is built, and gated, on a model: off the model give the trial's registration (--trial) or a "
            "saved edit (--saved), which the kernel built and hashed. (--edit is taken with --stand-in.)"
        )
    ep = Path(a.edit)
    got = we.from_recipe(json.loads(ep.read_text(encoding="utf-8")), ep.parent)
    edit = (we.load(got, lm_ref, model_id=h.model_id, weights_id=h.weights_id) if isinstance(got, (str, Path))
            else we.build(lm_ref, got, model_id=h.model_id, weights_id=h.weights_id))
    return edit, target, None, lm_ref, W_U_ref, None


def cmd_lora(a: argparse.Namespace) -> int:
    from backend import level_export as lex

    try:
        edit, target, saved_dir, lm_ref, W_U_ref, reg = _load_edit_and_target(a)
        trial_res = json.loads(Path(a.trial_result).read_text(encoding="utf-8")) if a.trial_result else None
        dose = float(a.dose) if a.dose is not None else float(reg["trial"]["main_dose"]) if reg is not None else 1.0
        out = lex.export_lora_adapter(
            edit, target,
            out_path=Path(a.out) if a.out else None,
            dose=dose,
            overwrite=bool(a.force),
            trial_result=trial_res,
            book_path=Path(a.book).expanduser() if a.book else None,
            saved_edit_dir=saved_dir,
            lm_for_implant=lm_ref,
            W_U_for_implant=W_U_ref,
            registration=reg,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "manifest"}, indent=2))
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_verify(a: argparse.Namespace) -> int:
    from backend import level_export as lex

    try:
        edit = target = None
        if a.edit or a.saved or a.trial:
            edit, target, _, _, _, _ = _load_edit_and_target(a)
        res = lex.verify_exported_lora(Path(a.adapter), edit=edit, target=target, dose=a.dose)
        print(json.dumps({k: v for k, v in res.items() if k != "meta_summary"}, indent=2))
        if edit is None:
            print("note: keys and manifest only. Give the edit (--trial or --saved) and the base (--gguf or "
                  "--weights-dir) to show the file is that edit on that base.", file=sys.stderr)
        return 0 if res["ok"] else 1
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_show(a: argparse.Namespace) -> int:
    from backend import level_export as lex

    try:
        res = lex.verify_exported_lora(Path(a.adapter))
        print(json.dumps({"tensors": res["tensors"], "meta": res["meta_summary"]}, indent=2, default=str))
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_smoke(a: argparse.Namespace) -> int:
    from backend import level_export as lex

    try:
        res = lex.llama_server_smoke(
            Path(a.llama_server), Path(a.gguf), Path(a.adapter), str(a.prompt),
            scale=float(a.scale), n_predict=int(a.n_predict), port=int(a.port), timeout_s=float(a.load_timeout),
        )
        print(json.dumps(res, indent=2))
        return 0
    except _refusals() as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def _edit_and_base_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trial", default=None, help="a trial's registration folder")
    p.add_argument("--saved", default=None, help="a saved edit: its id or its folder")
    p.add_argument("--edit", default=None, help="an edit recipe JSON path (only with --stand-in)")
    p.add_argument("--payload", default=None, help="the saved edit folder of a registered edit that carries arrays")
    p.add_argument("--gguf", default=None, help="the Gemma 4 GGUF the adapter will be served over")
    p.add_argument("--weights-dir", default=None, help="a W4A16 or dense checkpoint folder")
    p.add_argument("--stand-in", action="store_true")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="levelling_export.py", description="Export an edit as a llama.cpp LoRA adapter.")
    sub = ap.add_subparsers(dest="command", required=True)

    lo = sub.add_parser("lora", help="write an edit as a llama.cpp LoRA .gguf with its .manifest.json")
    _edit_and_base_flags(lo)
    lo.add_argument("--dose", type=float, default=None, help="the dose (a registration's own when not given, else 1)")
    lo.add_argument("-o", "--out", default=None, help="the adapter file to write (outside every git checkout)")
    lo.add_argument("--trial-result", default=None, help="the trial_result.json of this edit's trial at this dose")
    lo.add_argument("--book", default=None,
                    help="the book the trial was read in, and where a reading of an edit rebuilt off the model is "
                         "looked for (the path in the result, else the level book, when not given)")
    lo.add_argument("--force", action="store_true", help="write over an adapter and manifest already there")
    lo.set_defaults(fn=cmd_lora)

    vf = sub.add_parser("verify", help="read an adapter back; with the edit and the base, show it is that edit there")
    vf.add_argument("adapter", help="the .gguf adapter")
    _edit_and_base_flags(vf)
    vf.add_argument("--dose", type=float, default=None)
    vf.set_defaults(fn=cmd_verify)

    sh = sub.add_parser("show", help="print an adapter's general.* and theodolite.* keys")
    sh.add_argument("adapter", help="the .gguf adapter")
    sh.set_defaults(fn=cmd_show)

    sm = sub.add_parser("smoke", help="a smoke test in a local llama-server: one prompt with the adapter off and on")
    sm.add_argument("adapter", help="the .gguf adapter")
    sm.add_argument("--gguf", required=True, help="the Gemma 4 GGUF to serve")
    sm.add_argument("--llama-server", required=True, help="path to the llama-server binary")
    sm.add_argument("--prompt", required=True, help="the prompt to write with the adapter at 0 and at --scale")
    sm.add_argument("--scale", type=float, default=1.0)
    sm.add_argument("--n-predict", type=int, default=16)
    sm.add_argument("--port", type=int, default=18089)
    sm.add_argument("--load-timeout", type=float, default=600.0, help="seconds to wait for the server to load the model")
    sm.set_defaults(fn=cmd_smoke)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
