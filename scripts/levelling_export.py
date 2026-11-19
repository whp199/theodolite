#!/usr/bin/env python3
"""Owner-only CLI for Levelling Phase 5: Adapter Export & Verification (`scripts/levelling_export.py`).

Exports a Levelling edit as a llama.cpp LoRA `.gguf` adapter with Apache-2.0 licence & origin keys and a companion
`.manifest.json`, recompiled against the served base (`--gguf` or checkpoint), into an owner-controlled directory
outside the checkout and outside tool-writable folders. Never exposed to agents.

    .venv/bin/python -I scripts/levelling_export.py lora --edit edit.json --gguf MODEL.gguf -o /tmp/exports/out.gguf
    .venv/bin/python -I scripts/levelling_export.py lora --trial /tmp/trial/prereg --weights-dir /path/to/w4a16
    .venv/bin/python -I scripts/levelling_export.py verify /tmp/exports/out.gguf
    .venv/bin/python -I scripts/levelling_export.py show /tmp/exports/out.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_edit_and_target(a: argparse.Namespace) -> Any:
    import torch

    torch.set_num_threads(1)
    from backend import level_export as lex
    from backend import level_trial as lt
    from backend import levelling as lv
    from backend import weight_edits as we

    lm_ref = None
    W_U_ref = None
    saved_dir = None

    if a.stand_in:
        from atlas_stand_in import stand_in as make_stand_in

        rt, _vault = make_stand_in()
        h = lv.handle(rt)
        lm_ref, W_U_ref = h.lm, h.W_U
        model_id, weights_id = h.model_id, h.weights_id
    else:
        lm_ref = None
        model_id = "google/gemma-4-31B-it-qat-w4a16-ct"
        weights_id = None

    # Build the TargetBase for recompilation (GGUF > weights-dir > stand-in module)
    if a.gguf:
        target = lex.TargetBase(gguf_path=Path(a.gguf))
    elif a.weights_dir:
        target = lex.TargetBase(weights_dir=Path(a.weights_dir))
    elif lm_ref is not None:
        target = lex.TargetBase(lm=lm_ref, label="stand-in")
    else:
        from backend.census import cached_checkpoint

        snap = cached_checkpoint()
        target = lex.TargetBase(weights_dir=Path(snap))

    # Load Edit (from --trial prereg, --saved dir, or --edit recipe)
    if a.trial:
        reg = lt.verify_registration(Path(a.trial))
        man = reg["edit_manifest"]
        if lm_ref is not None:
            edit = lt.load_registered_edit(lv.handle(rt), man, payload_dir=Path(a.payload) if a.payload else None)
        else:
            # Off-GPU build from manifest without loading full GPU model: use lightweight Edit built from TargetBase
            edit = _edit_from_manifest_cpu(man, target, payload_dir=Path(a.payload) if a.payload else None)
        return edit, target, saved_dir, lm_ref, W_U_ref

    if a.saved:
        saved_dir = we._find_saved(a.saved)
        if lm_ref is not None:
            edit = we.load(saved_dir, lm_ref, model_id=model_id, weights_id=weights_id)
        else:
            man = json.loads((saved_dir / "manifest.json").read_text(encoding="utf-8"))
            edit = _edit_from_saved_cpu(saved_dir, man)
        return edit, target, saved_dir, lm_ref, W_U_ref

    if a.edit:
        ep = Path(a.edit)
        recipe = json.loads(ep.read_text(encoding="utf-8"))
        if lm_ref is not None:
            got = we.from_recipe(recipe, ep.parent)
            edit = (we.load(got, lm_ref, model_id=model_id, weights_id=weights_id)
                    if isinstance(got, (str, Path))
                    else we.build(lm_ref, got, model_id=model_id, weights_id=weights_id))
        else:
            edit = _edit_from_recipe_cpu(recipe, ep.parent, target)
        return edit, target, saved_dir, lm_ref, W_U_ref

    raise lex.ExportRefused("Give one of --trial, --saved, or --edit.")


def _edit_from_saved_cpu(folder: Path, man: dict) -> Any:
    """Reconstruct a `weight_edits.Edit` on CPU from a saved edit folder without loading the full model."""
    import numpy as np
    from backend import weight_edits as we

    payload = {}
    if (folder / "payload.safetensors").is_file():
        from safetensors.numpy import load_file

        payload = load_file(str(folder / "payload.safetensors"))
    terms = []
    for t in man["terms"]:
        layer, which = we._parse(t["linear"])
        if t["kind"] in ("cols", "rows"):
            idx_key = "columns" if t["kind"] == "cols" else "rows"
            idx = we._ro([int(n) for n in t[idx_key]], np.int64)
            op = we._ro([we.OPS.index(o) for o in t["ops"]], np.uint8)
            fac = we._ro([float(f) for f in t["factors"]], np.float32)
            vec = we._ro(payload[t["vectors"]], np.float32) if t.get("vectors") else None
            terms.append(we.Term(t["linear"], t["kind"], layer, which, idx=idx, op=op, factor=fac, vectors=vec))
        else:
            A = we._ro(payload[t["A"]], np.float32)
            B = we._ro(payload[t["B"]], np.float32)
            terms.append(we.Term(t["linear"], "lowrank", layer, which, A=A, B=B))
    return we.Edit(man["id"], tuple(terms), man["base"], str(man.get("note") or ""), dict(man.get("report") or {}))


def _edit_from_manifest_cpu(man: dict, target: Any, *, payload_dir: Optional[Path] = None) -> Any:
    from backend import level_export as lex
    from backend import weight_edits as we

    if not man.get("all_rebuilt"):
        where = payload_dir if payload_dir is not None else we._find_saved(man["edit_id"])
        saved = we._find_saved(where)
        raw = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
        return _edit_from_saved_cpu(saved, raw)
    recipe = man.get("rebuilt_recipe")
    if not isinstance(recipe, dict):
        raise lex.ExportRefused("edit_manifest lacks rebuilt_recipe.")
    return _edit_from_recipe_cpu(recipe, Path("."), target, edit_id=man["edit_id"], base=man.get("base"))


def _edit_from_recipe_cpu(
    recipe: dict,
    base_dir: Path,
    target: Any,
    *,
    edit_id: Optional[str] = None,
    base: Optional[dict] = None,
) -> Any:
    import numpy as np
    from backend import weight_edits as we

    got = we.from_recipe(recipe, base_dir)
    if isinstance(got, (str, Path)):
        saved = we._find_saved(got)
        return _edit_from_saved_cpu(saved, json.loads((saved / "manifest.json").read_text(encoding="utf-8")))
    terms = []
    for s in got:
        lname = we.linear_name(int(s.layer), s.which)
        if s.kind in ("cols", "rows"):
            idx = we._ro([int(n) for n in s.neurons], np.int64)
            order = np.argsort(idx, kind="stable")
            idx = we._ro(idx[order], np.int64)
            op = we._ro([we.OPS.index(s.op)] * len(idx), np.uint8)
            if s.op == "scale":
                f = s.factor
                fac = np.asarray([float(f)] * len(idx) if not isinstance(f, (list, tuple)) else [float(x) for x in f], dtype=np.float32)[order]
                vec = None
            else:
                fac = np.ones(len(idx), dtype=np.float32)
                vec = we._ro(np.asarray(s.vectors, dtype=np.float32)[order], np.float32)
            terms.append(we.Term(lname, s.kind, int(s.layer), s.which, idx=idx, op=op, factor=we._ro(fac, np.float32), vectors=vec))
        elif s.kind == "lowrank":
            terms.append(we.Term(lname, "lowrank", int(s.layer), s.which, A=we._ro(s.A, np.float32), B=we._ro(s.B, np.float32)))
        else:
            raise we.Refused(f'CPU off-model export does not lower "{s.kind}" directly; save the built edit first.')
    base_d = dict(base or {"model": "google/gemma-4-31B-it-qat-w4a16-ct", "weights_id": target.record, "tensors": {}})
    eid = edit_id or we._edit_id(base_d, terms)
    return we.Edit(eid, tuple(terms), base_d, str(recipe.get("note") or ""), {})


def cmd_lora(a: argparse.Namespace) -> int:
    from backend import level_export as lex
    from backend import level_trial as lt
    from backend import weight_edits as we

    try:
        edit, target, saved_dir, lm_ref, W_U_ref = _load_edit_and_target(a)
        trial_res = json.loads(Path(a.trial_result).read_text(encoding="utf-8")) if a.trial_result else None
        out = lex.export_lora_adapter(
            edit, target,
            out_path=Path(a.out) if a.out else None,
            dose=float(a.dose),
            overwrite=bool(a.force),
            book_seq=a.book_seq,
            book_hash=a.book_hash,
            trial_result=trial_res,
            prior_attempts=a.prior_attempts,
            saved_edit_dir=saved_dir,
            lm_for_implant=lm_ref,
            W_U_for_implant=W_U_ref,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "manifest"}, indent=2))
        return 0
    except (lex.ExportRefused, lt.TrialRefused, we.Refused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_verify(a: argparse.Namespace) -> int:
    from backend import level_export as lex

    try:
        edit = target = None
        if a.edit or a.saved or a.trial:
            edit, target, _, _, _ = _load_edit_and_target(a)
        res = lex.verify_exported_lora(Path(a.adapter), edit=edit, target=target, dose=a.dose)
        print(json.dumps({k: v for k, v in res.items() if k != "meta_summary"}, indent=2))
        return 0 if res["ok"] else 1
    except (lex.ExportRefused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_show(a: argparse.Namespace) -> int:
    from backend import level_export as lex

    try:
        res = lex.verify_exported_lora(Path(a.adapter))
        print(json.dumps({"tensors": res["tensors"], "meta": res["meta_summary"]}, indent=2, default=str))
        return 0
    except (lex.ExportRefused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def cmd_remeasure(a: argparse.Namespace) -> int:
    from backend import level_export as lex

    try:
        res = lex.remeasure_with_llama_server(
            Path(a.llama_server), Path(a.gguf), Path(a.adapter), str(a.prompt),
            scale=float(a.scale), n_predict=int(a.n_predict), port=int(a.port),
        )
        print(json.dumps(res, indent=2))
        return 0
    except (lex.ExportRefused, OSError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="levelling_export.py", description="Owner-only Phase 5 LoRA adapter export and verification.")
    sub = ap.add_subparsers(dest="command", required=True)

    lo = sub.add_parser("lora", help="export an edit to a llama.cpp LoRA .gguf + .manifest.json")
    lo.add_argument("--trial", default=None, help="Phase 5 prereg/ registration folder")
    lo.add_argument("--saved", default=None, help="saved edit id or folder")
    lo.add_argument("--edit", default=None, help="edit recipe JSON path")
    lo.add_argument("--payload", default=None, help="saved edit folder for carried terms")
    lo.add_argument("--gguf", default=None, help="served Gemma 4 GGUF to recompile rebuilt terms against")
    lo.add_argument("--weights-dir", default=None, help="W4A16/dense checkpoint folder")
    lo.add_argument("--stand-in", action="store_true")
    lo.add_argument("--dose", type=float, default=1.0)
    lo.add_argument("-o", "--out", default=None, help="output .gguf path (outside checkout/tool folders)")
    lo.add_argument("--trial-result", default=None, help="optional trial_result.json to embed verdict & book hash")
    lo.add_argument("--book-seq", type=int, default=None)
    lo.add_argument("--book-hash", default=None)
    lo.add_argument("--prior-attempts", type=int, default=None)
    lo.add_argument("--force", action="store_true", help="overwrite an existing export file")
    lo.set_defaults(fn=cmd_lora)

    vf = sub.add_parser("verify", help="verify headers, licence keys, manifest sha256, and optional math equivalence")
    vf.add_argument("adapter", help="exported .gguf adapter path")
    vf.add_argument("--trial", default=None)
    vf.add_argument("--saved", default=None)
    vf.add_argument("--edit", default=None)
    vf.add_argument("--payload", default=None)
    vf.add_argument("--gguf", default=None)
    vf.add_argument("--weights-dir", default=None)
    vf.add_argument("--stand-in", action="store_true")
    vf.add_argument("--dose", type=float, default=None)
    vf.set_defaults(fn=cmd_verify)

    sh = sub.add_parser("show", help="print all general.* and theodolite.* keys of an exported adapter")
    sh.add_argument("adapter", help="exported .gguf adapter path")
    sh.set_defaults(fn=cmd_show)

    rm = sub.add_parser("remeasure", help="live re-measurement via llama-server --lora-init-without-apply")
    rm.add_argument("adapter", help="exported .gguf adapter path")
    rm.add_argument("--gguf", required=True, help="served Gemma 4 GGUF model path")
    rm.add_argument("--llama-server", required=True, help="path to llama-server binary")
    rm.add_argument("--prompt", required=True, help="prompt text to compare scale 0 vs scale")
    rm.add_argument("--scale", type=float, default=1.0)
    rm.add_argument("--n-predict", type=int, default=16)
    rm.add_argument("--port", type=int, default=18089)
    rm.set_defaults(fn=cmd_remeasure)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
