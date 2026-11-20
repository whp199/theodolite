#!/usr/bin/env python3
"""CPU checks of Levelling Phase 5: the trial of record, the owner's export, a volunteer's re-run and its intake
(`scripts/check_level_trial.py`).

No card (`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, one thread). The models are the packed bf16 stand-in (which
has a folded form) and the float32 stand-in, and for the export a small Gemma 4 GGUF written into a temporary folder
with weights of its own. A stand-in's figures mean nothing: what is checked is what is refused, counted, written and
compared.

1. The kernel's additions: `rebuilt_recipe`, `edit_provenance`, `base_checksum` (which must notice a change, on packed
   and on plain linears), `implant_shape`, and the ignore rules for weight files.
2. Reworded sets: validation, the hash, overlap with the brief's own wording, uses counted by an item's wording (so a
   renamed or re-noted set is the same set), where a set's items stand (`reworded_origin`), practice on a set.
3. The registration folder: what `register_trial` refuses before it writes anything, what `verify_registration`
   refuses (a file changed, added or missing, a weight file, a folder, a link, and any `SHA256SUMS` that is not byte
   for byte the one a registration writes).
4. The judgement, by planted cases: the folded tolerance and its cap, `passed`, `failed`, and every `void`.
5. A trial read on the packed stand-in: refused when it would not be of record, read as a dry run when asked, judged
   again from its book; read once; counted in the brief's budget beside claims; the claim gate's refusals applied; a
   trial of record read in a process and a repository of its own; the sentence; the command line.
6. The export: where it may go, what its keys say (a trial's verdict only for its own edit and dose; the implant label
   in four words), that the file is the edit on the target by arithmetic that does not go through the compiler, an
   edit rebuilt off the model and kept to its id, the command line, and that no agent surface reaches any of it.
7. The re-run: the kit, the volunteer's run on the home items, and an intake that judges the returned book: a bundle
   typed by hand, the home book handed back, a summary that is not its book's, another card's figures, odd JSON, and
   unsafe tars.
8. No kept-out word in any Levelling file or in any sheet or rules file written here.

Run:
    .venv/bin/python -I scripts/check_level_trial.py
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

import atlas_stand_in as asi  # noqa: E402
import check_dequant as cd  # noqa: E402
import export_lora as el  # noqa: E402
import levelling_export as lex_cli  # noqa: E402
import levelling_rerun as lr_cli  # noqa: E402
import levelling_trial as lt_cli  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_export as lex  # noqa: E402
from backend import level_mcp as lmcp  # noqa: E402
from backend import level_rerun as lr  # noqa: E402
from backend import level_routes as lroutes  # noqa: E402
from backend import level_trial as lt  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend import weight_reader as wr  # noqa: E402

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)

PASSED = 0
FAILED = 0
ROWS = {"small": 10, "large": 16}
SHEETS: List[str] = []


def check(cond: bool, msg: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}", flush=True)


def section(title: str) -> None:
    print(f"\n[{title}]", flush=True)


def refuses(fn: Callable[[], Any], kinds: Any, says: Optional[str] = None) -> bool:
    """Whether `fn` raises one of `kinds`, and (when `says`) with those words in its message."""
    try:
        fn()
    except kinds as exc:
        return says is None or says.lower() in str(exc).lower()
    return False


class DrawnStub:
    """A drawn item, as far as the folded reader looks at one."""

    def __init__(self, name: str) -> None:
        self.item = type("I", (), {"name": name, "id": name})()
        self.frame_id, self.half = name, 0


def _write_synthetic_gemma4_gguf(path: Path, n_layers: int, d_model: int, d_ff: int, seed: int = 99,
                                 arch: str = "gemma4") -> Dict[str, np.ndarray]:
    """A small F32 GGUF of the given architecture with weights of its own, to compile an adapter against."""
    import gguf

    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), arch)
    w.add_name("synthetic-gemma4-standin")
    w.add_uint32(f"{arch}.block_count", int(n_layers))
    w.add_uint32(f"{arch}.embedding_length", int(d_model))
    w.add_uint32(f"{arch}.feed_forward_length", int(d_ff))
    stored: Dict[str, np.ndarray] = {}
    for L in range(n_layers):
        for which, shape in (
            ("ffn_down", (d_model, d_ff)),
            ("ffn_gate", (d_ff, d_model)),
            ("ffn_up", (d_ff, d_model)),
        ):
            arr = rng.standard_normal(shape).astype(np.float32) * 0.25
            tname = f"blk.{L}.{which}.weight"
            w.add_tensor(tname, arr)
            stored[tname] = arr
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return stored


def _own_items(name: str = "own") -> Dict[str, Any]:
    """A reworded set of the owner's own writing, as a file with no hash in it yet."""
    return {"name": name, "items": [
        {"item_id": "a", "half": 0, "question": "Tell what the miller Ottilie did when the sluice jammed at dusk.",
         "answer": " Ottilie fetched the iron bar because"},
        {"item_id": "b", "half": 0, "question": "Say how the glazier Henrike spent the frost-bound morning.",
         "answer": " Henrike warmed the putty while"},
        {"item_id": "c", "half": 1, "question": "Describe the ferrywoman Liesel at the turn of the tide.",
         "answer": " Liesel coiled the wet rope before"},
        {"item_id": "d", "half": 1, "question": "Relate what the cooper Margit noticed about the oak staves.",
         "answer": " Margit set the hoop aside since"}]}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=check", "-c", "user.email=check@example.invalid", *args],
                   check=True, capture_output=True)


def _pack(folder: Path, dst: Path) -> None:
    with tarfile.open(dst, "w") as tf:
        for n in lr.PACKED_FILES:
            tf.add(folder / n, arcname=n)


def _bundle(folder: Path, book: Path, ret: Dict[str, Any], sheet: str = "no sheet\n") -> None:
    """A returned bundle's four files in `folder`, with a manifest that names them truly."""
    folder.mkdir(parents=True, exist_ok=True)
    if book.resolve() != (folder / "volunteer_book.jsonl").resolve():
        shutil.copy2(book, folder / "volunteer_book.jsonl")
    (folder / "return.json").write_text(json.dumps(ret), encoding="utf-8")
    (folder / "TRIAL_SHEET.md").write_text(sheet, encoding="utf-8")
    man = {"v": 1, "files": {n: {"sha256": hashlib.sha256((folder / n).read_bytes()).hexdigest()}
                             for n in ("return.json", "volunteer_book.jsonl", "TRIAL_SHEET.md")}}
    (folder / "manifest.json").write_text(json.dumps(man), encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="check-level-trial-") as td:
        tmp = Path(td)
        prev_edits = os.environ.get(we.EDITS_ENV)
        prev_exp = os.environ.get(lex.EXPORT_ENV)
        prev_bk = os.environ.get(lb.BOOK_ENV)
        os.environ[we.EDITS_ENV] = str(tmp / "saved_edits")
        os.environ[lex.EXPORT_ENV] = str(tmp / "owner_exports")
        os.environ[lb.BOOK_ENV] = str(tmp / "default_book.jsonl")

        try:
            hP = lv.handle(cd.packed_stand_in()[0])
            we.assert_base(hP.lm, hP.hooks_at_load)
            n_layers = hP.n_layers
            D, N = we.linear_shape(hP.lm.layers[2].mlp.down_proj)
            rng = np.random.default_rng(7)

            def build(specs: Any, note: str = "") -> we.Edit:
                return we.build(hP.lm, specs, model_id=hP.model_id, weights_id=hP.weights_id, note=note)

            # ------------------------------------------------------------------ 1. the kernel's additions
            section("1. Recipes, provenance, the base checksum, the implant label, the ignore rules")
            gi_txt = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
            check("*.safetensors" in gi_txt and "*.gguf" in gi_txt, "1: .gitignore ignores *.safetensors and *.gguf")

            e_reb = build([we.silence(2, [3, 7]), we.scale_rows(3, "up_proj", [1, 4], 0.5)], "rebuilt test")
            rec = we.rebuilt_recipe(e_reb)
            check(rec is not None and len(rec["terms"]) == 2, "1: rebuilt_recipe emits pure JSON recipe for uncarried edit")
            e_reb_again = we.build(hP.lm, we.from_recipe(rec, tmp), model_id=hP.model_id, weights_id=hP.weights_id,
                                   note=rec.get("note", ""))
            check(e_reb_again.id == e_reb.id, "1: edit rebuilt from rebuilt_recipe hashes to identical edit.id")
            prov_reb = we.edit_provenance(e_reb)
            check(prov_reb["all_rebuilt"] is True and prov_reb["provenance"] == "rebuilt" and prov_reb["payload_sha256"] is None
                  and all(t["provenance"] == "rebuilt" and "recipe_term" in t for t in prov_reb["terms"]),
                  "1: edit_provenance marks every uncarried term 'rebuilt' with recipe_term and no payload")
            e_car = build([we.silence(2, [1, 2]),
                           we.add_to_columns(3, [5], rng.standard_normal((1, D)).astype(np.float32) * 0.05)], "mixed carried")
            check(we.rebuilt_recipe(e_car) is None, "1: rebuilt_recipe returns None when any term carries vectors")
            saved_car = we.save(e_car)
            prov_car = we.edit_provenance(e_car, saved_dir=saved_car)
            check(prov_car["all_rebuilt"] is False and prov_car["provenance"] == "carried"
                  and isinstance(prov_car["payload_sha256"], str) and len(prov_car["payload_sha256"]) == 64
                  and prov_car["terms"][0]["provenance"] == "rebuilt" and prov_car["terms"][1]["provenance"] == "carried",
                  "1: edit_provenance marks mixed terms 'rebuilt' vs 'carried' and hashes payload.safetensors")

            ck0 = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)
            ck1 = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)
            check(ck0["sha256"] == ck1["sha256"] and ck0["linears_checked"] >= 18 and ck0["overlays_in_force"] == 0,
                  f"1: base_checksum deterministic over {ck0['linears_checked']} packed linears")
            mod_ck = hP.lm.layers[2].mlp.down_proj
            with torch.no_grad():
                w0 = int(mod_ck.weight_packed.view(-1)[0])
                mod_ck.weight_packed.view(-1)[0] = w0 ^ 1
            ck_bit = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)["sha256"]
            with torch.no_grad():
                mod_ck.weight_packed.view(-1)[0] = w0
                s0 = mod_ck.weight_scale.view(-1)[0].clone()
                mod_ck.weight_scale.view(-1)[0] = s0 * 2
            ck_scale = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)["sha256"]
            with torch.no_grad():
                mod_ck.weight_scale.view(-1)[0] = s0
                emb = hP.lm.embed_tokens.weight
                e0 = emb.view(-1)[0].clone()
                emb.view(-1)[0] = e0 + 1
            ck_emb = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)["sha256"]
            with torch.no_grad():
                emb.view(-1)[0] = e0
            ck_back = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)["sha256"]
            check(len({ck0["sha256"], ck_bit, ck_scale, ck_emb}) == 4 and ck_back == ck0["sha256"],
                  "1: base_checksum changes with one bit of a packed word, one scale and one embedding entry, and comes back")
            hF = lv.handle(asi.stand_in()[0])
            ckf0 = we.base_checksum(hF.lm, hooks_at_load=hF.hooks_at_load)
            wF = hF.lm.layers[2].mlp.down_proj.weight
            with torch.no_grad():
                f0 = wF[0, 0].clone()
                wF[0, 0] = f0 + 1
            ckf1 = we.base_checksum(hF.lm, hooks_at_load=hF.hooks_at_load)
            with torch.no_grad():
                wF[0, 0] = f0
            check(ckf0["linears_checked"] >= 18 and ckf0["tensors_checked"] > ckf0["linears_checked"]
                  and ckf0["sha256"] != ckf1["sha256"]
                  and ckf0["sha256"] == we.base_checksum(hF.lm, hooks_at_load=hF.hooks_at_load)["sha256"],
                  f"1: base_checksum hashes plain linears too ({ckf0['linears_checked']} linears, {ckf0['tensors_checked']} "
                  "tensors on the float stand-in) and notices a weight written in place")

            # the implant label: a write along one unembedding row in the last layer, as a column and as a low-rank term
            last_L = n_layers - 1
            w_post_last = we._w_post(hP.lm, last_L)
            u = (hP.W_U[5].detach().float().cpu().numpy() / np.where(np.abs(w_post_last) > 1e-3, w_post_last, 1.0)).astype(np.float32)
            u = u / (float(np.linalg.norm(u)) + 1e-8) * 0.05
            e_imp = build([we.add_to_columns(last_L, [0], u[None, :])])
            e_imp_lr = build([we.lowrank(last_L, np.eye(N, dtype=np.float32)[0:1], u[:, None].astype(np.float32))])
            imp_c, imp_l = we.implant_shape(e_imp, hP.lm, hP.W_U), we.implant_shape(e_imp_lr, hP.lm, hP.W_U)
            check(imp_c["flagged"] and imp_c["examined"] == 1 and imp_c["hits"][0]["top_piece_id"] == 5,
                  f"1: implant_shape flags a last-layer column written along one unembedding row ({imp_c['hits']})")
            check(imp_l["flagged"] and imp_l["examined"] == 1,
                  "1: implant_shape looks at a low-rank term's write too, and flags the same write there")
            imp_rows = we.implant_shape(build([we.silence_rows(last_L, "up_proj", [1])]), hP.lm, hP.W_U)
            imp_low = we.implant_shape(build([we.silence(2, [1])]), hP.lm, hP.W_U)
            check(not imp_rows["flagged"] and imp_rows["examined"] == 0 and len(imp_rows["not_examined"]) == 1
                  and not imp_low["flagged"] and imp_low["examined"] == 0 and len(imp_low["not_examined"]) == 1,
                  "1: implant_shape says what it did not look at (a row term; a term below the last tenth of the layers)")
            check(lex.implant_words(imp_c) == "true" and lex.implant_words(imp_low) == "nothing in reach of the check"
                  and lex.implant_words(lex.implant_check(e_imp)) == "not checked"
                  and lex.implant_words(we.implant_shape(build([we.scale(last_L, [1], 0.5)]), hP.lm, hP.W_U)) == "false",
                  "1: the implant label is one of four words, and 'false' only when something in reach was looked at")
            imp_part = we.implant_shape(build([we.scale(last_L, [1], 0.5), we.silence(2, [1])]), hP.lm, hP.W_U)
            check(lex.implant_words(imp_part) == "false for the part in reach of the check (1 term(s) were not)"
                  and imp_part["examined"] == 1 and len(imp_part["not_examined"]) == 1,
                  "1: and when part of an edit was out of the check's reach the label says 'false' of the part it looked at alone")
            chunk_was = we.IMPLANT_CHUNK
            try:
                we.IMPLANT_CHUNK = 7
                imp_c7, imp_l7 = we.implant_shape(e_imp, hP.lm, hP.W_U), we.implant_shape(e_imp_lr, hP.lm, hP.W_U)
            finally:
                we.IMPLANT_CHUNK = chunk_was
            check(int(hP.W_U.shape[0]) > 3 * 7 and imp_c7["hits"] == imp_c["hits"] and imp_l7["hits"] == imp_l["hits"],
                  "1: the implant label reads the unembedding in parts, and finds the same rows whatever the size of a part")

            # weights a module keeps where it does not save them (a GGUF linear's blocks, an int4 linear's words) are
            # hashed too; a cache it may fill as it runs is not
            mod_un = hP.lm.layers[1].mlp.up_proj
            mod_un.register_buffer("weight_blocks", torch.arange(24, dtype=torch.uint8).reshape(4, 6), persistent=False)
            mod_un.register_buffer("scratch_cache", torch.zeros(3), persistent=False)
            try:
                ck_blk = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)
                with torch.no_grad():
                    mod_un.weight_blocks.view(-1)[5] += 1
                ck_blk2 = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)
                with torch.no_grad():
                    mod_un.scratch_cache += 1.0
                ck_blk3 = we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)
            finally:
                for nm in ("weight_blocks", "scratch_cache"):
                    del mod_un._buffers[nm]
                    mod_un._non_persistent_buffers_set.discard(nm)
            check(ck_blk["sha256"] != ck0["sha256"] and ck_blk2["sha256"] != ck_blk["sha256"]
                  and ck_blk3["sha256"] == ck_blk2["sha256"] and ck_blk["tensors_checked"] == ck0["tensors_checked"] + 1
                  and we.base_checksum(hP.lm, hooks_at_load=hP.hooks_at_load)["sha256"] == ck0["sha256"],
                  "1: base_checksum hashes weights kept in a buffer a module does not save, and no cache beside them")
            check(lv.asked_in_process()["forwards"] == 0,
                  "1: the process counts the forwards the kernel has run, and has run none before the first reading")

            # ------------------------------------------------------------------ 2. reworded sets
            section("2. Reworded sets: validation, the hash, overlap, uses by wording, where the items stand")
            brief = lbr.pronoun_pull_brief(hP.tokenizer, max_fresh_readings=2)
            rset = lt.starter_reworded_pronoun_pull(brief)
            check(len(rset.items) == 8 and sum(i.half == 0 for i in rset.items) == 4 and sum(i.half == 1 for i in rset.items) == 4,
                  "2: starter_reworded_pronoun_pull has 4 half-0 and 4 half-1 items")
            rw_file = lt.save_reworded(rset, tmp / "reworded.json")
            rset_back = lt.load_reworded(rw_file)
            check(rset_back.id == rset.id, "2: save_reworded / load_reworded round-trip exact id")
            bad_rw = json.loads(rw_file.read_text())
            bad_rw["items"][0]["question"] += " extra"
            (tmp / "reworded_bad.json").write_text(json.dumps(bad_rw))
            check(refuses(lambda: lt.load_reworded(tmp / "reworded_bad.json"), lt.TrialRefused), "2: tampered reworded.json refused")
            bad_items = (
                lt.RewordedItem("b1", 0, "Finish the story: Across the studio, Claire worked quietly as a pianist. Soon",
                                " Claire set down the notebook and"),
                lt.RewordedItem("b2", 0, "A distinct question two?", "Answer two"),
                lt.RewordedItem("b3", 1, "A distinct question three?", "Answer three"),
                lt.RewordedItem("b4", 1, "A distinct question four?", "Answer four"),
            )
            check(refuses(lambda: lt.validate_reworded_against_brief(
                lt.RewordedSet("overlap_test", brief.id, brief.name, bad_items), brief), lt.TrialRefused),
                "2: reworded item overlapping brief kept-back frame refused")
            check(refuses(lambda: lt.RewordedSet("thin", brief.id, brief.name, bad_items[:3]), lt.TrialRefused),
                  "2: a reworded set needs two items in each half")

            own = lt.RewordedSet.from_dict({**_own_items(), "brief_id": brief.id, "brief_name": brief.name})
            lt.validate_reworded_against_brief(own, brief)
            check("starter set" in lt.reworded_origin(rset, brief) and "starter set" in lt.reworded_origin(rset_back, brief)
                  and lt.reworded_origin(own, brief) == "from a file given at registration",
                  "2: a sheet says the starter set's items are the starter set's, by their wording, whatever file they came in by")
            mixed = lt.RewordedSet("mixed", brief.id, brief.name,
                                   own.items[:3] + (lt.RewordedItem("z", 1, rset.items[4].question, rset.items[4].answer),))
            check("1 of its 4 items are the starter set's" in lt.reworded_origin(mixed, brief),
                  "2: a set that borrows one starter item says so")
            brief_alias = lbr.Brief.from_dict({**brief.to_dict(), "name": "pronoun_pull_under_another_name"})
            near_set = lt.RewordedSet("starter_reworked", brief_alias.id, brief_alias.name, tuple(
                lt.RewordedItem(i.item_id, i.half, i.question.replace("brief scene", "short scene").replace("plain prose", "simple prose"),
                                i.answer + " then", "a system turn of its own") for i in rset.items))
            check(near_set.wordings() != rset.wordings() and brief_alias.id != brief.id
                  and lt.reworded_origin(near_set, brief_alias).startswith("the starter set")
                  and lt.reworded_origin(near_set).startswith("the starter set"),
                  "2: starter items under another brief's name, with a word changed and another system turn, are still the starter set's")

            marks_book = tmp / "marks_book.jsonl"
            for i in range(lt.MAX_REWORDED_USES):
                lb.append("note", {"note_type": "trial_registration", "reworded_marks": rset.marks(), "trial_sha256": f"m{i}"},
                          path=marks_book)
                lb.append("paired", {"name": f"trial_m{i}_reworded", "items": [
                    {"item": {"question": it.question, "answer": it.answer}} for it in rset.items]}, path=marks_book)
            # every question with a word added and every answer changed: no mark of the starter set's is left, and
            # only the text a trial's own reworded reading keeps can show these are its items
            near_all = lt.RewordedSet("near_all", brief.id, brief.name, tuple(
                lt.RewordedItem(i.item_id, i.half, i.question + " Please.", i.answer + " then") for i in rset.items))
            marks_only = tmp / "marks_only_book.jsonl"  # registrations that never came to a reworded reading
            for i in range(2):
                lb.append("note", {"note_type": "trial_registration", "reworded_marks": rset.marks(), "trial_sha256": f"o{i}"},
                          path=marks_only)
            resys = lt.RewordedSet("resys", brief.id, brief.name, tuple(
                lt.RewordedItem(i.item_id, i.half, i.question, i.answer, "a system turn of its own") for i in rset.items))
            reans = lt.RewordedSet("reans", brief.id, brief.name, tuple(
                lt.RewordedItem(i.item_id, i.half, i.question, " She") for i in rset.items))
            check(set(resys.wordings()).isdisjoint(rset.wordings())
                  and lt.reworded_uses_in_book(resys, marks_book) == lt.MAX_REWORDED_USES
                  and lt.reworded_uses_in_book(reans, marks_book) == lt.MAX_REWORDED_USES
                  and lt.reworded_uses_in_book(own, marks_book) == 0 and len(rset.marks()) == 16
                  and set(near_all.marks()).isdisjoint(rset.marks())
                  and lt.reworded_uses_in_book(near_all, marks_only) == 0 and lt.reworded_uses_in_book(rset, marks_only) == 2
                  and lt.reworded_uses_in_book(near_all, marks_book) == lt.MAX_REWORDED_USES,
                  "2: an item given another system turn or another answer, or with a word changed, is the same item when "
                  "its uses are counted (by its marks in a registration, or by a near wording in a trial's reworded reading)")
            prac_near = tmp / "prac_near_book.jsonl"
            lb.append("paired", {"name": "practice_near", "items": [{"item": {
                "question": rset.items[0].question.replace("brief scene", "short scene"), "answer": " Somebody else did"}}]},
                path=prac_near)
            lb.append("paired", {"name": "cal_the_harness_own", "items": [{"item": {
                "question": rset.items[1].question, "answer": rset.items[1].answer}}]}, path=prac_near)
            lb.append("paired", {"name": "practice_unlike", "items": [{"item": {
                "question": "Which river runs through the old town?", "answer": " She"}}]}, path=prac_near)
            lb.append("paired", {"name": "practice_answer", "items": [{"item": {
                "question": "Go on from here.", "answer": rset.items[2].answer + " the storm"}}]}, path=prac_near)
            check(lt.reworded_practised(rset, prac_near) == [0, 3] and lt.reworded_practised(own, prac_near) == [],
                  f"2: a practice reading holds a reworded item by a near wording of its question, or by the start of its "
                  f"answer; the harness's own readings and unlike items do not count ({lt.reworded_practised(rset, prac_near)})")

            rot_book = tmp / "rot_book.jsonl"
            for i in range(lt.MAX_REWORDED_USES):
                lb.append("note", {"note_type": "trial_registration", "reworded_id": rset.id,
                                   "reworded_wordings": rset.wordings(), "trial_sha256": f"t{i}"}, path=rot_book)
            renamed = lt.RewordedSet("another_name", brief.id, brief.name, rset.items, note="another note")
            one_new = lt.RewordedSet("one_new", brief.id, brief.name, rset.items[:7] + (own.items[3],))
            check(lt.reworded_uses_in_book(rset, rot_book) == lt.MAX_REWORDED_USES and renamed.id != rset.id
                  and lt.reworded_uses_in_book(renamed, rot_book) == lt.MAX_REWORDED_USES
                  and lt.reworded_uses_in_book(one_new, rot_book) == lt.MAX_REWORDED_USES
                  and lt.reworded_uses_in_book(own, rot_book) == 0
                  and lt.reworded_uses_in_book(rset, tmp / "no_such_book.jsonl") == 0,
                  "2: uses are counted by an item's wording: a renamed, re-noted or part-changed set does not start again")

            # ------------------------------------------------------------------ 3. the registration folder
            section("3. The registration folder: what is refused before it is written, and what verification refuses")
            cal_book = tmp / "trial_book.jsonl"
            bars = lcl.calibrate_bars(hP, brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
                                      rows=ROWS, seed=0, book_path=cal_book)
            e_trial = build([we.silence(2, [3, 7])], "trial edit")
            KW: Dict[str, Any] = dict(reworded=rset, book_path=cal_book, controls_per_family=39, reworded_controls=19,
                                      n_target=4, n_neutral=2, n_near=2, n_panel=2, rows=ROWS, match_tolerance=20.0)
            n_refused = [0]

            def reg_refused(says: str, *, edit: Any = None, brief_: Any = None, bars_: Any = None, **over: Any) -> bool:
                n_refused[0] += 1
                folder = tmp / f"refused_{n_refused[0]}"
                ok = refuses(lambda: lt.register_trial(folder, hP, brief_ or brief, bars_ or bars,
                                                       edit or e_trial, trial_name="x", **{**KW, **over}),
                             lt.TrialRefused, says)
                return ok and not folder.exists()

            check(reg_refused("reworded set", reworded=None), "3: a trial with no reworded set is refused, and no folder is written")
            wbrief = lbr.enumeration_loop_brief(hP.tokenizer)
            check(reg_refused("written brief", brief_=wbrief, bars_=lcl.make_bar_set(wbrief.id)),
                  "3: a written brief has no trial of record, and is told so at registration")
            e_lr = build([we.lowrank(2, rng.standard_normal((2, N)).astype(np.float32) * 0.02,
                                     rng.standard_normal((D, 2)).astype(np.float32) * 0.02), we.silence(2, [1])])
            check(reg_refused("folded form", edit=e_lr), "3: an edit with no folded form (a low-rank term) has no trial")
            check(reg_refused("calibration reading", bars_=lcl.make_bar_set(brief.id, near_miss_abs_max=50.0)),
                  "3: bars that no calibration in the book produced are refused at registration")
            check(reg_refused("calibration reading", book_path=tmp / "another_book.jsonl"),
                  "3: bars calibrated in another book are refused: a trial is registered against the book it is read in")
            check(reg_refused("could not pass", controls_per_family=3)
                  and reg_refused("could not pass", reworded_controls=3)
                  and reg_refused("at most", controls_per_family=lv.MAX_CONTROLS + 1),
                  "3: a design that could not reach the level, or asks more control edits than a reading takes, is refused")
            check(reg_refused("sham", selection_mode="declared_rule", declared_rule="the rule",
                              sham_recipes=[{"terms": [{"layer": 2, "silence": [3, 7]}]}])
                  and reg_refused("sham", selection_mode="declared_rule", declared_rule="the rule",
                                  sham_recipes=[{"terms": [{"layer": 2, "silence": [10, 11]}]}] * 2)
                  and reg_refused("declared_rule", sham_recipes=[{"terms": [{"layer": 2, "silence": [10, 11]}]}]),
                  "3: sham controls are checked as the claim gate checks them (the edit itself, a repeat, shams with a free pick)")
            check(we.dose_refusal(e_trial, 5.0, hP.dtype) is None and reg_refused("cannot be read as registered", main_dose=5.0)
                  and reg_refused(f"1 to {lv.MAX_ITEMS} items", n_target=lv.MAX_ITEMS + 4),
                  "3: a reading the kernel would refuse as planned (a dose past its reach, more items than a reading "
                  "takes) is found before a folder is written")
            # bars whose calibration note does not follow from the book: a note written in by hand, with bars of choice
            real_cal = lcl.calibration_note(bars, cal_book)
            forged_book = tmp / "forged_cal_book.jsonl"
            shutil.copy2(cal_book, forged_book)
            bars_forged = lcl.make_bar_set(brief.id, near_miss_abs_max=50.0, calibration_book_seqs=bars.calibration_book_seqs)
            lb.append("note", {**{k: v for k, v in real_cal.items() if k not in ("seq", "at")},
                               "bars": bars_forged.to_dict(), "bars_sha256": bars_forged.id}, path=forged_book)
            forged_note = lcl.calibration_note(bars_forged, forged_book)
            check(lcl.calibration_follows(real_cal, cal_book, brief, bars) is None
                  and "other bars" in str(lcl.calibration_follows(forged_note, forged_book, brief, bars_forged))
                  and "settings" in str(lcl.calibration_follows({k: v for k, v in real_cal.items() if k != "settings"},
                                                                cal_book, brief, bars))
                  and "not the reading" in str(lcl.calibration_follows({**real_cal, "book_seqs": [1, 1, 2]}, cal_book, brief, bars))
                  and reg_refused("does not follow", bars_=bars_forged, book_path=forged_book),
                  "3: a calibration note follows from its book's own three readings, or its bars are not a trial's "
                  f"({lcl.calibration_follows(forged_note, forged_book, brief, bars_forged)})")

            make_was = lv.make_item

            def no_item(*a: Any, **k: Any) -> Any:
                raise lv.Refused("an item this tokenizer cannot make")

            lv.make_item = no_item
            try:
                check(reg_refused("cannot be made into items"),
                      "3: a reworded set the tokenizer cannot make items of is refused at registration, not after the trial is entered")
            finally:
                lv.make_item = make_was

            prereg_dir = tmp / "my_trial" / "prereg"
            reg = lt.register_trial(prereg_dir, hP, brief, bars, e_trial, trial_name="standin_rebuilt_trial", **KW)
            check(sorted(p.name for p in prereg_dir.iterdir()) == sorted(lt.REG_FILES + ("SHA256SUMS",)),
                  "3: register_trial writes exactly the files a registration holds")
            check(not list(prereg_dir.glob("*.safetensors")) and not list(prereg_dir.glob("*.gguf")),
                  "3: prereg/ contains zero .safetensors or .gguf weight files")
            v_ok = lt.verify_registration(prereg_dir)
            check(v_ok["ok"] and v_ok["trial_sha256"] == reg["trial_sha256"], "3: verify_registration passes clean folder")
            tc = reg["trial"]
            check(tc["bars_calibration"]["book"] == cal_book.name and isinstance(tc["bars_calibration"]["seq"], int)
                  and tc["bars_calibration"]["settings"]["mult"] == 2.0 and tc["bars_calibration"]["settings"]["layer"] == 2
                  and set(tc["sources"]) == set(lt.source_hashes()) and all(tc["sources"].values())
                  and {"backend/atlas_ops.py", "backend/engine_runtime.py", "backend/level_trial.py",
                       "scripts/levelling_trial.py", "scripts/levelling_rerun.py"} <= set(tc["sources"])
                  and tc["implant_check"]["checked"] is True and "starter set" in tc["reworded_origin"]
                  and tc["max_fresh_readings"] == 2 and tc["priced"]["forwards"] > 0
                  and tc["reworded_marks"] == rset.marks(),
                  "3: the registration records its calibration and how it was taken, the hashes of every backend file and "
                  "the two command lines, the price of its readings, the implant label and where the reworded set stands")
            rules_txt = (prereg_dir / "RULES.md").read_text()
            for needle in ("OF RECORD only when", "spends one of the brief's fresh readings",
                           "the frames every fresh reading of the", "largest gap of any one item",
                           "within a factor of `20`", "`passed, matched by a cheaper knob`", "facts about ONE book",
                           "as public as that commit", "it is read in the level book", "only if that claim read",
                           "its registered least and `2.0` times the largest figure they gave: neutral KL",
                           "of the 4 are the least a bar can be", "at layer `2` (page layer `3`)",
                           "on what a base pre-pass expects each to deliver", "is counted as moving as much",
                           "the folded forward's own repeated rows agree to the bit",
                           "in size or as a share of its own effect", "matches on its effect alone",
                           "with as few repeats as its eligible sites allow",
                           "no more separate draws than the fewest different stand-ins any one neuron had",
                           "A look taken at registration, with the base alone, counted",
                           "also at the last position, where this brief's metric is read",
                           "A control edit, or the edit, with no such figure is counted the same way",
                           "and no code where the runner",
                           "the look taken at registration did not say it could not pass"):
                check(needle in rules_txt, f"3: RULES.md says \"{needle}\"")
            sz_reg = reg["trial"]["sizing"]
            check(sz_reg["matched"] == {"control_edits": 39, "found": 39, "fewest_eligible": sz_reg["matched"]["fewest_eligible"],
                                        "fewest_distinct": 39, "counted_as": 39, "needs": 39, "could_pass": True}
                  and sz_reg["matched"]["fewest_eligible"] >= 39 and sz_reg["reworded"]["counted_as"] == 19
                  and sz_reg["reworded"]["needs"] == 19 and sz_reg["reworded"]["could_pass"] is True and reg["warnings"] == []
                  and "BY THAT LOOK" not in rules_txt
                  and f"`39` of `39` (fewest eligible sites a neuron `{sz_reg['matched']['fewest_eligible']}`)" in rules_txt,
                  f"3: a registration records the look taken at how many separate draws its control sets count as, and "
                  f"its rules print it ({sz_reg['matched']})")
            # a design that by that look could not pass: registered, said so, and not read of record
            pre_cant = tmp / "cannot_pass" / "prereg"
            reg_cant = lt.register_trial(pre_cant, hP, brief, bars, e_trial, trial_name="cannot_pass",
                                         **{**KW, "match_tolerance": 1.0})
            cond_cant = lt.record_conditions(pre_cant, reg_cant["trial"], book=cal_book)
            cond_unlooked = lt.record_conditions(prereg_dir, {k: v for k, v in reg["trial"].items() if k != "sizing"},
                                                 book=cal_book)
            n_cant = lb.verify(cal_book)["entries"]
            check(reg_cant["trial"]["sizing"]["matched"]["could_pass"] is False
                  and reg_cant["trial"]["sizing"]["matched"]["counted_as"] < 39 and len(reg_cant["warnings"]) >= 1
                  and "could not pass" in reg_cant["warnings"][0] and "wider tolerance" in reg_cant["warnings"][0]
                  and "BY THAT LOOK THIS TRIAL COULD NOT PASS" in (pre_cant / "RULES.md").read_text()
                  and lt.verify_registration(pre_cant)["ok"]
                  and any("by the look taken at registration it could not pass" in r for r in cond_cant["reasons"])
                  and not any("could not pass" in r for r in lt.record_conditions(prereg_dir, reg["trial"], book=cal_book)["reasons"])
                  and any("records no look" in r for r in cond_unlooked["reasons"])
                  and refuses(lambda: lt.run_trial_of_record(hP, pre_cant, out_dir=tmp / "cannot_pass" / "out", book_path=cal_book),
                              lt.TrialRefused, "by the look taken at registration it could not pass")
                  and lb.verify(cal_book)["entries"] == n_cant,
                  f"3: a trial whose control sets could not count as enough separate draws is registered with a warning, its "
                  f"rules say so, and it is not read of record; nor is one whose registration records no look "
                  f"({reg_cant['trial']['sizing']['matched']})")
            check(refuses(lambda: lt.register_trial(prereg_dir, hP, brief, bars, e_trial, trial_name="dup", **KW), lt.TrialRefused),
                  "3: register_trial refuses overwriting existing prereg/")

            sums = (prereg_dir / "SHA256SUMS").read_bytes()
            first = sums.split(b"\n")[0]
            mutators = {
                "a file changed": lambda d: (d / "trial.json").write_text((d / "trial.json").read_text() + " "),
                "a file added": lambda d: (d / "sneaky.txt").write_text("hi"),
                "a file added and listed in the sums": lambda d: (
                    (d / "columns.npy").write_bytes(b"x"),
                    (d / "SHA256SUMS").write_bytes(lt._sums_bytes(d, list(lt.REG_FILES) + ["columns.npy"]))),
                "a weight file": lambda d: (d / "payload.safetensors").write_bytes(b"fake"),
                "a folder inside": lambda d: ((d / "sub").mkdir(), (d / "sub" / "x").write_text("x")),
                "a link inside": lambda d: os.symlink("/etc/hostname", d / "link"),
                "a file missing": lambda d: (d / "RULES.md").unlink(),
                "a newline added to the sums": lambda d: (d / "SHA256SUMS").write_bytes(sums + b"\n"),
                "the sums in CRLF": lambda d: (d / "SHA256SUMS").write_bytes(sums.replace(b"\n", b"\r\n")),
                "an upper-case digest": lambda d: (d / "SHA256SUMS").write_bytes(sums.replace(first, first[:64].upper() + first[64:])),
                "a line repeated": lambda d: (d / "SHA256SUMS").write_bytes(sums + first + b"\n"),
                # the sums written again after the change: only what the files say of one another can refuse these
                "RULES.md changed by hand, its sums written again": lambda d: (
                    (d / "RULES.md").write_text((d / "RULES.md").read_text().replace("in BOTH batch classes", "in one batch class")),
                    lt._write_sha256sums(d)),
                "registered_at changed, its sums written again": lambda d: (
                    (d / "registered_at").write_text("2020-01-01T00:00:00Z\n"), lt._write_sha256sums(d)),
                "trial.json's tolerance changed, its sums written again": lambda d: (
                    (d / "trial.json").write_text((d / "trial.json").read_text().replace('"match_tolerance": 20.0', '"match_tolerance": 50.0')),
                    lt._write_sha256sums(d)),
            }
            for label, mutator in mutators.items():
                twin = tmp / ("bad_prereg_" + re.sub(r"\W+", "_", label))
                shutil.copytree(prereg_dir, twin)
                mutator(twin)
                check(refuses(lambda: lt.verify_registration(twin), lt.TrialRefused), f"3: verify_registration refuses {label}")

            prereg_car = tmp / "carried_trial" / "prereg"
            lt.register_trial(prereg_car, hP, brief, bars, e_car, trial_name="carried_trial", saved_edit_dir=saved_car, **KW)
            v_car = lt.verify_registration(prereg_car)
            check(v_car["trial"]["edit_provenance"] == "carried" and v_car["trial"]["payload_sha256"] == prov_car["payload_sha256"]
                  and not (prereg_car / "payload.safetensors").exists(),
                  "3: carried-edit prereg/ stores payload_sha256 only, never payload.safetensors")
            e_car_loaded = lt.load_registered_edit(hP, v_car["edit_manifest"], payload_dir=saved_car)
            check(e_car_loaded.id == e_car.id, "3: load_registered_edit verifies and loads carried edit from external payload_dir")

            # ------------------------------------------------------------------ 4. the judgement, by planted cases
            section("4. The judgement: the folded tolerance, passed, failed, and every void")
            b_unit = lbr.pronoun_pull_brief(hP.tokenizer, max_fresh_readings=1)
            bars_unit = lcl.BarSet(brief_id=b_unit.id)

            def _good_fresh_classes() -> Dict[str, Any]:
                effs = [
                    {"frame_id": f"pr_k0{i}", "half": 0 if i <= 4 else 1, "item_id": f"i{i}", "value": -1.5}
                    for i in range(1, 9)
                ]
                return {
                    cls: {
                        "item_effects": effs,
                        "floor": 0.05,
                        "reversed_mean": 1.4,
                        "controls": {"other_sites_matched": [-0.05] * 39, "same_site": [0.02] * 39},
                    }
                    for cls in ("small", "large")
                }

            good_bats = {
                "neutral_text": {"mean_kl": 0.001},
                "near_misses": {"mean_effect": -0.02, "abs_effect": 0.02},
                "labelled_panel": {"own_drop": 0.0, "mean_kl": 0.001},
                "habitat": {"available": False},
            }
            knobs_none = {"prompt_arms": [], "prompt_arms_state": lcl.PROMPT_ARMS_NONE}
            fj_good = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=_good_fresh_classes(), batteries=good_bats,
                                      cheaper_knobs=knobs_none)
            check(fj_good["verdict"] == "lead", f"4: synthetic good fresh claim returns lead (got {fj_good['verdict']})")

            def folded_in(value: float = -1.5, gap: Optional[float] = 0.0, n: int = 8) -> Dict[str, Any]:
                return {
                    cls: {
                        "available": True, "taken_count": n, "expected_count": n,
                        "rows_identical_all": True, "same_bits_count": n if gap == 0.0 else 0,
                        "item_effects": [{**x, "value": value} for x in _good_fresh_classes()[cls]["item_effects"]],
                        "max_abs_gap": gap, "mean_abs_gap": gap, "between_kl_max": 0.0,
                    }
                    for cls in ("small", "large")
                }

            def rw_in(value: float = -1.5, controls: Any = None) -> Dict[str, Any]:
                return {cls: {"item_effects": [{**x, "value": value} for x in _good_fresh_classes()[cls]["item_effects"]],
                              "floor": 0.05, "controls": {"other_sites_matched": [-0.05] * 19 if controls is None else controls}}
                        for cls in ("small", "large")}

            fe_good = lt.judge_folded(b_unit, bars_unit, fj_good, folded_in())
            rw_good = lt.judge_reworded(b_unit, bars_unit, rw_in(), folded_in())
            tj_pass = lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, reworded_eval=rw_good)
            check(tj_pass["verdict"] == "passed", f"4: all gates passing yields 'passed' (got {tj_pass['verdict']})")

            # the tolerance: the card's own figures (floor 0.1534, effect 1.4057, bar 0.3392), and its cap
            tol_card = lt.folded_tolerance(0.1534, -1.4057, 0.3392, floor_mult=lt.DEFAULT_FOLDED_FLOOR_MULT,
                                           rel=lt.DEFAULT_FOLDED_REL_TOL, least=lt.DEFAULT_FOLDED_MIN_TOL)
            check(abs(tol_card - 0.1534) < 1e-9 and tol_card <= 0.5 * 0.3392
                  and lt.folded_tolerance(0.5, -1.0, 0.3, floor_mult=3.0, rel=0.05, least=0.02) == 0.15,
                  f"4: the folded tolerance is one floor on the card's figures ({tol_card:.4f}) and never above half the house bar")
            fe_half = lt.judge_folded(b_unit, bars_unit, fj_good, folded_in(value=-0.8, gap=0.7))
            check(not fe_half["ok"] and fe_half["available"]
                  and lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_half, reworded_eval=rw_good)["verdict"] == "failed",
                  "4: a folded form that delivers half the per-row effect is not agreement: 'failed'")
            fe_nogap = lt.judge_folded(b_unit, bars_unit, fj_good, folded_in(gap=None))
            check(not fe_nogap["ok"], "4: a folded form with no item gap recorded is not called agreement")
            rw_bad = lt.judge_reworded(b_unit, bars_unit, rw_in(value=-0.01), folded_in(value=-0.01))
            check(lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, reworded_eval=rw_bad)["verdict"] == "failed"
                  and not rw_bad["ok"], "4: planted reworded failure yields 'failed'")
            rw_noctl = lt.judge_reworded(b_unit, bars_unit, rw_in(controls=[]), folded_in())
            check(not rw_noctl["ok"], "4: a reworded set with no control edits to stand against does not pass")

            fe_unavail = lt.judge_folded(b_unit, bars_unit, fj_good, {"small": {"available": False}, "large": {"available": False}})
            fe_part = lt.judge_folded(b_unit, bars_unit, fj_good, {"small": folded_in()["small"], "large": {"available": False, "why": "x"}})
            voids = {
                "the base checksum changed": lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, reworded_eval=rw_good, base_checksum_unchanged=False),
                "the unpacker fell back": lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, reworded_eval=rw_good, unpacker_fallbacks=1),
                "the folded form was not read": lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_unavail, reworded_eval=rw_good),
                "the folded form was read in one class only": lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_part, reworded_eval=rw_good),
                "the reworded set was not read": lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, reworded_eval=None),
            }
            for label, tj in voids.items():
                check(tj["verdict"] == "void", f"4: void when {label} (got {tj['verdict']})")
            twin_ok = {"taken": True, "same_bits": True, "metrics": {"odds": {"mean": -1.0}}, "gap": {"odds": {"abs": 0.0}}}

            def ex_of(second: Dict[str, Any]) -> Dict[str, Any]:
                return lt.extract_folded_class({"items": [
                    {"status": "ok", "item": {"name": "n0"}, "classes": {"small": {"folded": twin_ok}}}, second]},
                    [DrawnStub("n0"), DrawnStub("n1")], "small", "odds")

            ex_part = ex_of({"status": "ok", "item": {"name": "n1"}, "classes": {"small": {"folded": {"taken": False, "why": "no twin"}}}})
            ex_none = ex_of({"status": "ok", "item": {"name": "n1"}, "classes": {"small": {"arms": []}}})
            ex_nan = ex_of({"status": "ok", "item": {"name": "n1"}, "classes": {"small": {"folded": {
                **twin_ok, "metrics": {"odds": {"mean": None}}}}}})
            check(ex_part["available"] is False and ex_part["why"] == "no twin" and ex_part["taken_count"] == 1
                  and ex_part["expected_count"] == 2 and ex_none["available"] is False and ex_none["why"] == "folded twin missing"
                  and ex_nan["available"] is False and ex_nan["why"] == "the folded form gave a finite reading on 1 of the 2 items that read",
                  "4: a folded form read on some of the items that read is not the folded form of the reading "
                  f"({ex_part['why']}; {ex_none['why']}; {ex_nan['why']})")
            ex_unstable = ex_of({"status": "unstable", "item": {"name": "n1"}, "classes": {}})
            check(ex_unstable["available"] is True and ex_unstable["expected_count"] == 1 and ex_unstable["drawn_count"] == 2
                  and lt.extract_folded_class({"items": []}, [DrawnStub("n0")], "small", "odds")["why"]
                  == "none of the 1 items read in this class",
                  "4: an item that did not read is in neither form: it does not make the folded form 'not read'")

            # with no bar (the floor too loud) the folded form is said to have none to clear, and nothing else of a bar
            loud = _good_fresh_classes()
            for cls in loud:
                loud[cls]["floor"] = 0.5
            fj_loud = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=loud, batteries=good_bats, cheaper_knobs=knobs_none)
            fe_loud = lt.judge_folded(b_unit, bars_unit, fj_loud, folded_in())
            rw_loud_in = rw_in()
            for cls in rw_loud_in:
                rw_loud_in[cls]["floor"] = 0.5
            rw_loud = lt.judge_reworded(b_unit, bars_unit, rw_loud_in, folded_in())
            said = " ".join(fe_loud["reasons"] + rw_loud["reasons"])
            check(fj_loud["verdict"] == "kept the base" and not fe_loud["ok"] and fe_loud["classes"]["small"]["no_bar"] is True
                  and "floor too loud" in said and "(None)" not in said and "did not replicate" not in said
                  and "did not clear" not in said,
                  f"4: with the floor too loud the folded form and the reworded set are said to have no bar, and are not "
                  f"said to have missed one ({said[:160]})")

            # a pass that one of the brief's prompt arms matched is called that, with the arm's own figure
            knobs_arm = {"prompt_arms": [{"index": 0, "prompt_sha256": "ab12", "directed_effect": 1.6, "near_miss_abs": 0.01}],
                         "prompt_arms_state": lcl.PROMPT_ARMS_RUN}
            fj_knob = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=_good_fresh_classes(), batteries=good_bats,
                                      cheaper_knobs=knobs_arm)
            tj_knob = lt.judge_trial(b_unit, bars_unit, fj_knob, folded_eval=fe_good, reworded_eval=rw_good)
            knobs_strong = {"prompt_arms": [{"index": 0, "prompt_sha256": "cd34", "directed_effect": 8.72, "near_miss_abs": 1.38}],
                            "prompt_arms_state": lcl.PROMPT_ARMS_RUN}
            knobs_blunt = {"prompt_arms": [{"index": 0, "prompt_sha256": "ef56", "directed_effect": 8.72, "near_miss_abs": 6.0}],
                           "prompt_arms_state": lcl.PROMPT_ARMS_RUN}
            fj_strong = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=_good_fresh_classes(), batteries=good_bats,
                                        cheaper_knobs=knobs_strong)
            fj_blunt = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=_good_fresh_classes(), batteries=good_bats,
                                       cheaper_knobs=knobs_blunt)
            s_strong = lcl.claim_sentence(b_unit, fj_strong, {"main_dose": 1.0, "design": {"match_tolerance": 2.0}}) or ""
            s_blunt = lcl.claim_sentence(b_unit, fj_blunt, {"main_dose": 1.0, "design": {"match_tolerance": 2.0}}) or ""
            check(fj_strong["verdict"] == "lead, matched by a cheaper knob" and fj_blunt["verdict"] == "lead"
                  and "matched by prompt arm 0 (it moved odds by 8.7200 in the brief's direction, near-miss 1.3800)" in s_strong
                  and "no author prompt arm matched it (the author's strongest prompt arm moved odds by 8.7200 in the "
                      "brief's direction, near-miss 6.0000)" in s_blunt
                  and fj_strong["cheaper_knobs"]["matched_arm"]["near_miss"].endswith("as a share of its own effect")
                  and fj_blunt["cheaper_knobs"]["matched_arm"] is None,
                  "4: a prompt arm that moves the metric six times as far with a near-miss effect past the bar in size and "
                  f"within it as a share of its own effect matches the edit; the sentence gives the matched arm's own "
                  f"figures, or the strongest arm's when none matched ({fj_strong['verdict']})")
            # two arms, and the weaker is the one that matched: the figures said are its own, not the strongest arm's.
            # And an arm whose near-miss effect was not read matches on its effect alone, and is said so.
            knobs_two = {"prompt_arms": [{"index": 0, "prompt_sha256": "aa11", "directed_effect": 1.6, "near_miss_abs": 0.01},
                                         {"index": 1, "prompt_sha256": "bb22", "directed_effect": 8.72, "near_miss_abs": 6.0}],
                         "prompt_arms_state": lcl.PROMPT_ARMS_RUN}
            knobs_unread = {"prompt_arms": [{"index": 0, "prompt_sha256": "cc33", "directed_effect": 8.72, "near_miss_abs": None}],
                            "prompt_arms_state": lcl.PROMPT_ARMS_RUN}
            fj_two = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=_good_fresh_classes(), batteries=good_bats,
                                     cheaper_knobs=knobs_two)
            fj_unread = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=_good_fresh_classes(), batteries=good_bats,
                                        cheaper_knobs=knobs_unread)
            meta_k = {"main_dose": 1.0, "design": {"match_tolerance": 2.0}}
            s_two = lcl.claim_sentence(b_unit, fj_two, meta_k) or ""
            s_unread = lcl.claim_sentence(b_unit, fj_unread, meta_k) or ""
            sheet_unread = lcl.render_sheet(b_unit, bars_unit, fj_unread, meta_k, None)
            tj_unread = lt.judge_trial(b_unit, bars_unit, fj_unread, folded_eval=fe_good, reworded_eval=rw_good)
            check(fj_two["cheaper_knobs"]["matched_by"] == "prompt arm 0"
                  and "matched by prompt arm 0 (it moved odds by 1.6000 in the brief's direction, near-miss 0.0100)" in s_two
                  and "8.7200" not in s_two
                  and fj_unread["verdict"] == "lead, matched by a cheaper knob"
                  and fj_unread["cheaper_knobs"]["matched_arm"]["near_miss"] == "not read"
                  and "near-miss not read)" in s_unread and "near-miss none" not in s_unread
                  and "near_miss_abs=not read" in sheet_unread
                  and "a prompt arm matches when it moved the metric at least 0.95 times as far as the edit" in sheet_unread
                  and "an arm whose near-miss effect was not read matches on its effect alone" in sheet_unread
                  and tj_unread["cheaper_knob"]["near_miss"] == "not read"
                  and "was not read, and it is taken to match on its effect alone" in lt._knob_near_miss(tj_unread["cheaper_knob"]),
                  "4: the arm that matched is said with its own figures (not the strongest arm's), an arm whose near-miss "
                  "effect was not read matches on its effect alone and is said so, and a sheet states the rule an arm "
                  "matches by")
            check(fj_knob["verdict"] == "lead, matched by a cheaper knob" and tj_knob["verdict"] == "passed, matched by a cheaper knob"
                  and tj_knob["verdict"] in lt.PASSED
                  and {k: tj_knob["cheaper_knob"][k] for k in ("name", "directed_effect", "near_miss_abs",
                                                                "edit_directed_effect", "edit_near_miss_abs")} == {
                      "name": "prompt arm 0", "directed_effect": 1.6, "near_miss_abs": 0.01, "edit_directed_effect": 1.5,
                      "edit_near_miss_abs": 0.02}
                  and tj_knob["cheaper_knob"]["near_miss"].endswith("in size")
                  and "its near-miss effect was 0.0100 against the edit's 0.0200" in lt._knob_near_miss(tj_knob["cheaper_knob"])
                  and tj_pass["cheaper_knob"] is None and lt.TRIAL_VERDICTS[:2] == lt.PASSED,
                  f"4: a pass matched by a cheaper knob reads 'passed, matched by a cheaper knob' and carries the arm's "
                  f"figures beside the edit's, its near-miss effect among them ({tj_knob['verdict']})")

            # the sentence: only a trial of record that passed has one, and it says what was found
            meta_s = {
                "date": "2026-11-19T00:00:00Z", "model_id": "m", "weights_id": "w" * 64, "unpacker": "compiled",
                "edit_id": e_trial.id, "edit_linears": e_trial.linears(), "main_dose": 2.0,
                "bars_sha256": bars_unit.id, "bars_calibration_seq": 3, "reading_index": 2, "max_fresh_readings": 2,
                "kept_back": {"frames": 8, "published": 8}, "book": {"name": "level_book.jsonl", "default": True},
                "design": {"match_tolerance": 2.0, "rows": {"small": 12, "large": 32}}, "prompt_named_site": False,
                "trial_sha256": "f" * 64, "of_record": True, "not_of_record": [], "rerun_of": None,
                "repo_commit": "c" * 40, "reworded_use_index": 1, "registration_seq": 41, "earlier_fresh_readings": 1,
                "source_claim": {"seq": 10, "verdict": "lead", "reading_index": 1},
                "shams": None,
            }
            cfg_s = {"trial_name": "t", "reworded_id": rset.id, "reworded_origin": lt.reworded_origin(rset, b_unit)}
            sent = lt.trial_sentence(b_unit, cfg_s, fj_good, tj_pass, meta_s) or ""
            for needle in ("at dose 2", "fresh reading 2 of 2", "no author prompt arm was compared with it (none in the brief)",
                           "starter set in the instrument's source", "claim reading of this edit at book line 10",
                           "selection not matched", "in the level book of the environment it ran in, at line 41",
                           "which 1 earlier fresh reading of the brief in this book had already read",
                           "in a process that had asked nothing of the model before it",
                           "Those figures are of the large batch class (rows 32); in the small one (rows 12)",
                           "in the small class", "It was read once in that book", " Passed. "):
                check(needle in sent, f"4: a passed trial of record's sentence says \"{needle}\"")
            check("no author prompt arm matched it" not in sent and "owner-written" not in sent.lower()
                  and "This is a lead" not in sent and "Driver prompt" not in sent and "side book" not in sent
                  and KEPT_OUT_RE.search(sent) is None,
                  "4: and does not say a prompt arm was matched against when none ran, who wrote the reworded set, or "
                  "anything of a driver's prompt")
            sent_knob = lt.trial_sentence(b_unit, cfg_s, fj_knob, tj_knob, {**meta_s, "earlier_fresh_readings": 0}) or ""
            check("Passed, matched by a cheaper knob." in sent_knob and " Passed. " not in sent_knob
                  and "The cheaper knob: prompt arm 0" in sent_knob and "moved odds by 1.6000" in sent_knob
                  and "against the edit's 1.5000" in sent_knob and "matched by prompt arm 0" in sent_knob
                  and "which no earlier fresh reading in this book had read" in sent_knob,
                  "4: a pass a cheaper knob matched says so where it says 'passed', with the arm's figure beside the edit's")
            sham_meta = {**meta_s, "shams": {"count": 39, "sha256": "ab" * 32, "source": "registered with the trial"}}
            loud_sham = _good_fresh_classes()
            for cls in loud_sham:
                loud_sham[cls]["controls"]["sham_target"] = [0.01] * 39
            fj_sham = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=loud_sham, batteries=good_bats,
                                      cheaper_knobs=knobs_none, selection_mode="declared_rule")
            sent_sham = lcl.claim_sentence(b_unit, fj_sham, sham_meta, tail=False) or ""
            check("sham-target control edits, registered with the trial and checked" in sent_sham
                  and "supplied with the claim" not in sent_sham
                  and "supplied with the claim" in (lcl.claim_sentence(b_unit, fj_sham, {**meta_s, "shams": {"count": 39}}) or "")
                  and "sham-target controls: 39 registered with the trial" in lcl.render_sheet(b_unit, bars_unit, fj_sham, sham_meta, None)
                  and "Driver prompt did not name a suspected site. This is a lead, not a finding of record."
                  in (lcl.claim_sentence(b_unit, fj_sham, sham_meta) or ""),
                  f"4: a trial's sham controls are said to be registered with the trial; a claim's, supplied with the claim "
                  f"({fj_sham['verdict']})")
            SHEETS.append(sent)
            SHEETS.append(sent_knob)
            check(lt.trial_sentence(b_unit, cfg_s, fj_good, tj_pass, {**meta_s, "of_record": False}) is None
                  and lt.trial_sentence(b_unit, cfg_s, fj_good, tj_pass, {**meta_s, "rerun_of": "x" * 64}) is None
                  and lt.trial_sentence(b_unit, cfg_s, fj_good, {**tj_pass, "verdict": "failed"}, meta_s) is None,
                  "4: a dry run, a re-run and a trial that did not pass have no sentence")

            # ------------------------------------------------------------------ 5. a trial read on the packed stand-in
            section("5. A trial read: of record or refused, a dry run, read once, counted, judged again from its book")
            trial_out = tmp / "my_trial" / "result"
            check(lv.asked_in_process()["readings"] > 0 and lv.asked_in_process()["forwards"] > 0,
                  "5: the process counts the readings it has begun and the forwards the kernel has run")
            msg: List[str] = [""]

            def run_refused(says: str, pre: Path, out: str, *, book: Optional[Path] = None, **kw: Any) -> bool:
                try:
                    lt.run_trial_of_record(hP, pre, out_dir=tmp / out, book_path=book or cal_book, **kw)
                except lt.TrialRefused as exc:
                    msg.append(str(exc))
                    return says.lower() in str(exc).lower()
                return False

            n_before = lb.verify(cal_book)["entries"]
            check(run_refused("would not be of record", prereg_dir, "refused_out")
                  and "first thing its process asks of the model" in msg[-1] and "registration folder" in msg[-1]
                  and "a side book and not the level book" in msg[-1]
                  and "--dry-run" in msg[-1] and lb.verify(cal_book)["entries"] == n_before
                  and not (tmp / "refused_out").exists(),
                  "5: a trial that would not be of record is refused, says each reason, and enters nothing in the book")
            (tmp / "a_file").write_text("not a folder\n")
            check(run_refused("cannot be written to", prereg_dir, "a_file/out", dry_run=True)
                  and run_refused("inside the registration folder", prereg_dir, "my_trial/prereg/out", dry_run=True)
                  and lb.verify(cal_book)["entries"] == n_before
                  and sorted(p.name for p in prereg_dir.iterdir()) == sorted(lt.REG_FILES + ("SHA256SUMS",)),
                  "5: a result folder that cannot be written to, or that is inside the registration folder, is refused "
                  "before anything is entered in the book")

            t_res = lt.run_trial_of_record(hP, prereg_dir, out_dir=trial_out, book_path=cal_book, dry_run=True,
                                           _seed_override=b"trial-standin-seed-32bytes-00001")
            SHEETS.append(t_res["sheet"])
            check(t_res["verdict"] in ("passed", "failed") and t_res["of_record"] is False and t_res["claim_sentence"] is None
                  and t_res["sheet"].startswith("=== DRY RUN OF A TRIAL (not of record)")
                  and "(a dry run: not a verdict of record)" in t_res["sheet"] and "not of record: this process" in t_res["sheet"]
                  and "not of record: the seed of its draw was handed in" in t_res["sheet"]
                  and "not of record: it is read in trial_book.jsonl, a side book" in t_res["sheet"],
                  f"5: read as a dry run it is judged ({t_res['verdict']}), says it is not of record and why, and has no sentence")
            notes = [e["body"] for e in lb.entries(cal_book) if e["kind"] == "note"]
            reg_note = next(n for n in notes if n.get("note_type") == "trial_registration")
            v_note = notes[-1]
            check(v_note["note_type"] == lt.NOTE_DRY_RUN and reg_note["of_record"] is False and reg_note["reading_index"] == 1
                  and reg_note["max_fresh_readings"] == 2 and reg_note["main_dose"] == 1.0 and v_note["main_dose"] == 1.0
                  and isinstance(reg_note["edit_provenance"], dict) and reg_note["edit_provenance"]["rebuilt_recipe"]["terms"]
                  and len(reg_note["kept_back"]["template_ids"]) == 8 and reg_note["process"]["readings"] > 0,
                  "5: the book's notes say dry run, which fresh reading it was, the dose, the edit's recipe and the frames read")
            check(t_res["folded"]["available"] is True
                  and all(t_res["folded"]["classes"][c]["same_bits_count"] == t_res["folded"]["classes"][c]["taken_count"] == 4
                          for c in ("small", "large")),
                  "5: the folded form of a silence is the per-row form to the bit, on every item, in both classes")
            check(t_res["reworded"] is not None and t_res["reworded"]["folded"]["available"] is True
                  and "the reworded set" in t_res["sheet"] and "starter set in the instrument's source" in t_res["sheet"]
                  and "read 1 of at most 5 times in this book" in t_res["sheet"],
                  "5: the reworded set is read with its folded form, and the sheet says where it stands and how often it has been read")
            check(t_res["meta"]["base_checksum_unchanged"] is True and t_res["meta"]["base_linears_checked"] >= 18
                  and lv._fell_back() is None and t_res["meta"]["unpacker_fallbacks"] is None
                  and "unpacker fallbacks not counted" in t_res["sheet"] and "unpacker fallbacks 0" not in t_res["sheet"],
                  "5: the base checksum is unchanged over the stand-in's linears; with no compiled unpacker to count "
                  "fallbacks the sheet says they were not counted, and not that there were none")
            seed_file = json.loads((trial_out / "trial_seed.json").read_text())
            check((trial_out / "trial_result.json").is_file() and (trial_out / "TRIAL_SHEET.md").is_file()
                  and len(t_res["seed_hex"]) == 64 and t_res["seed_hex"] not in cal_book.read_text()
                  and seed_file["seed_hex"] == t_res["seed_hex"] and seed_file["trial_sha256"] == t_res["trial_sha256"]
                  and seed_file["registration_seq"] == t_res["book"]["registration_seq"],
                  "5: the result and the sheet are written; the draw's seed is kept in its own file from the draw on, "
                  "and in the result, and not in the book")
            check(lb.verify(cal_book)["ok"], "5: level book chain whole after the trial")
            for needle in ("CLAIM SHEET", "earlier fresh readings on this brief (0)", "earlier trials on this brief (0)",
                           "habitat (info): not run", "fresh items: new fillings of the brief's kept-back frames, which no "
                           "earlier fresh reading in this book had read",
                           "follows: no claim reading of this edit stands in this book",
                           "book: trial_book.jsonl (a side book, not the level book); the trial was entered at line",
                           "Read once and counted in this book: the harness cannot see another book",
                           "a probe of 2 columns at layer 2 (page layer 3) against 2 control edits (seed 0); each bar is "
                           "the larger of its registered least and 2.0 times the largest figure they gave: neutral KL",
                           "practice readings in this book: 0 | practice best, as reported with the edit's claim reading: none",
                           "other sites matched on what a base pre-pass expects each to deliver, within a factor of 20",
                           "agrees with the per-row form: True",
                           "process: had begun"):
                check(needle in t_res["sheet"], f"5: the sheet prints \"{needle}\"")
            dv_small = t_res["fresh_judged"]["classes"]["small"]["delivered"]["other_sites_matched"]
            cs_small = t_res["fresh_judged"]["classes"]["small"]["controls"]["other_sites_matched"]
            matched_plan = next(lb.entries(cal_book, start=t_res["book"]["all_seqs"][1], limit=1))["body"]
            check("driver prompt named site" not in t_res["sheet"] and "as reported with the claim:" not in t_res["sheet"]
                  and dv_small["within_tolerance"] + dv_small["not_comparable"] == dv_small["of_controls"] == 39
                  and dv_small["tolerance"] == 20.0 and isinstance(dv_small["fewest_eligible"], int)
                  and dv_small["read_at_end"] is True and dv_small["edit_end"] is not None and dv_small["edit_end"] > 0.0
                  and abs(dv_small["edit_end"] - dv_small["edit"]) > 1e-6
                  and dv_small["draw"] == "spread" and dv_small["fewest_distinct"] == min(39, dv_small["fewest_eligible"])
                  and cs_small["effective_count"] == min(39, dv_small["fewest_distinct"])
                  and matched_plan["plan"]["controls"].get("draw") == "spread" and matched_plan["controls"]["draw"] == "spread"
                  and "per_control" not in dv_small and "apart" not in dv_small
                  and f"{dv_small['within_tolerance']} of 39 within a factor of 20 of the edit's over the answer and at the "
                      "last position" in t_res["sheet"]
                  and "at the last position, where the metric is read: edit " in t_res["sheet"]
                  and (dv_small["within_tolerance"] == 0 or "(those a median " in t_res["sheet"])
                  and f"over the sets each neuron had at least {dv_small['fewest_distinct']} different stand-ins of "
                      f"{dv_small['fewest_eligible']} eligible sites, drawn with as few repeats as the sites allow" in t_res["sheet"]
                  and t_res["fresh_judged"]["batteries"]["labelled_panel"]["items"]["given"] == 2,
                  "5: the sheet of a trial has no line that is a claim's alone, says what the edit and its matched control "
                  f"edits delivered over the answer and at the last position, where this brief's metric is read "
                  f"({dv_small['within_tolerance']} of 39 within the factor on both), how many different stand-ins a neuron "
                  f"had ({dv_small['fewest_distinct']} of {dv_small['fewest_eligible']} eligible, drawn spread), and how "
                  f"many of a battery's items read")
            check(t_res["meta"]["book_head_before"] == next(lb.entries(cal_book, start=t_res["book"]["registration_seq"] - 1,
                                                                        limit=1))["hash"]
                  and t_res["meta"]["book_head_before"][:12] in t_res["sheet"] and t_res["meta"]["priced"]["forwards"] > 0,
                  "5: the sheet names the line of the book the trial was entered after, by its hash")
            check(KEPT_OUT_RE.search(t_res["sheet"]) is None and KEPT_OUT_RE.search(rules_txt) is None,
                  "5: the sheet and RULES.md contain zero kept-out words")

            rj = lt.rejudge_trial(cal_book, t_res["trial_sha256"], v_ok)
            check(rj["as_noted"] and rj["verdict"] == t_res["verdict"] and len(rj["items"]["target"]["small"]) == 4
                  and len(rj["items"]["reworded"]["large"]) == 8 and rj["readings"]["matched"]["edit_id"] == e_trial.id,
                  "5: judged again from the lines its note names, the trial gives the verdict its note records")
            fig = rj["figures"]
            check(sum(k.startswith("target:small:control:other_sites_matched:") for k in fig) == 39
                  and sum(k.startswith("target:large:control:same_site:") for k in fig) == 39
                  and sum(k.startswith("reworded:small:control:") for k in fig) == 19
                  and sum(k.startswith("target:small:folded:") for k in fig) == 4
                  and {"target:small:floor", "target:large:reversed", "reworded:large:floor", "battery:neutral_text:mean_kl",
                       "battery:near_misses:abs_effect", "battery:labelled_panel:own_drop"} <= set(fig)
                  and rj["drawn"]["target"] == v_note["drawn"]["target"] and rj["registration"]["trial_sha256"] == t_res["trial_sha256"]
                  and rj["readings"]["matched"]["plan"]["controls"]["count"] == 39,
                  f"5: and it gives every figure the judgement was made from, by name ({len(fig)} of them)")

            v_ent = next(lb.entries(cal_book, start=t_res["book"]["verdict_seq"], limit=1))
            card = lroutes.build_replay_card(v_ent)
            check(card["note_type"] == lt.NOTE_DRY_RUN and card["status"] == f"dry run: {t_res['verdict']}"
                  and card["of_record"] is False and card["edit_linears"] == ["layers.2.mlp.down_proj"]
                  and card["folded_summary"] is not None and card["reworded_summary"] is not None,
                  "5: the page's replay card of a dry run says dry run, and takes no verdict's status")

            # read once: the same folder, the same out folder, the same edit registered again
            check(run_refused("read once", prereg_dir, "my_trial/result", dry_run=True)
                  and run_refused("read once", prereg_dir, "my_trial/result_2", dry_run=True),
                  "5: read-once refuses both existing trial_result.json and duplicate trial_sha256 in book")
            again = tmp / "again" / "prereg"
            lt.register_trial(again, hP, brief, bars, e_trial, trial_name="again", **KW)
            check(run_refused("already had a trial", again, "again_out", dry_run=True),
                  "5: the same edit registered again is refused: an edit has one trial on a brief in a book")

            # counted: a trial spends a fresh reading; a claim after it lists it; the budget is claims and trials together
            hist1 = lcl.brief_history(brief.id, cal_book, name=brief.name)
            check(hist1["fresh_count"] == 1 and len(hist1["registrations"]) == 0 and len(hist1["trials"]) == 1
                  and hist1["book_fresh_readings"] == {brief.name: 1}
                  and lcl.prior_trials(hist1)[0]["kind"] == "dry run" and lcl.prior_trials(hist1)[0]["verdict"] == t_res["verdict"],
                  "5: the book's history of the brief counts the trial as a fresh reading")
            CK: Dict[str, Any] = dict(rows=ROWS, n_target=4, n_neutral=2, n_near=2, n_panel=2, controls_override=4,
                                      match_tolerance=20.0, book_path=cal_book)
            check(refuses(lambda: lcl.run_fresh_claim(hP, brief, bars, {"terms": [{"layer": 2, "silence": [3, 7]}]}, **CK),
                          lcl.ClaimRefused, "already had a trial"),
                  "5: nothing is read of an edit on a brief after its trial: a claim of it is refused")
            c2 = lcl.run_fresh_claim(hP, brief, bars, {"terms": [{"layer": 2, "silence": [10, 11]}]}, **CK)
            SHEETS.append(c2["sheet"])
            check(c2["meta"]["reading_index"] == 2 and "earlier trials on this brief (1)" in c2["sheet"]
                  and 'trial "standin_rebuilt_trial" (dry run)' in c2["sheet"],
                  "5: a claim after a trial is the brief's second fresh reading, and its sheet lists the trial")
            third = tmp / "third" / "prereg"
            e_third = build([we.silence(2, [20, 21])])
            lt.register_trial(third, hP, brief, bars, e_third, trial_name="third", **KW)
            check(run_refused("already spent all 2", third, "third_out", dry_run=True)
                  and "A trial of record is a fresh reading and spends one" in msg[-1],
                  "5: a brief's budget is its claims and its trials together: a third fresh reading is refused")

            # the claim gate's other refusals reach a trial too (a book of its own for each)
            def fresh_book(tag: str, brief_: Any = None) -> Any:
                bk = tmp / f"book_{tag}.jsonl"
                return bk, lcl.calibrate_bars(hP, brief_ or brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2,
                                              n_panel=2, rows=ROWS, seed=0, book_path=bk)

            def practice(book: Path, name: str, items: Dict[str, Any]) -> None:
                os.environ[lb.BOOK_ENV] = str(book)
                try:
                    lv.read(hP, {"name": name, "kind": "paired", "edit": {"terms": [{"layer": 2, "silence": [3, 7]}]},
                                 "controls": {"count": 1, "seed": 0}, "rows": ROWS, "readouts": dict(brief.readouts)}, items)
                finally:
                    os.environ[lb.BOOK_ENV] = str(tmp / "default_book.jsonl")

            bk_p, bars_p = fresh_book("practised")
            kb_items = brief.generator.draw("kept_back", 2, seed_bytes=b"x" * 32, tokenizer=hP.tokenizer)
            practice(bk_p, "practice_look", {d.item.name: d.item for d in kb_items})
            pre_p = tmp / "practised" / "prereg"
            lt.register_trial(pre_p, hP, brief, bars_p, e_trial, trial_name="practised", **{**KW, "book_path": bk_p})
            check(run_refused("read in practice in this book", pre_p, "practised_out", book=bk_p, dry_run=True),
                  "5: kept-back wording read in practice in the book refuses a trial, as it refuses a claim")

            bk_r, bars_r = fresh_book("rw_practised")
            practice(bk_r, "practice_on_the_set", {d.item.name: d.item for d in rset.to_drawn_items(hP.tokenizer)[:2]})
            pre_r = tmp / "rw_practised" / "prereg"
            lt.register_trial(pre_r, hP, brief, bars_r, e_trial, trial_name="rw_practised", **{**KW, "book_path": bk_r})
            check(lt.reworded_practised(rset, bk_r) != []
                  and run_refused("was read in practice in this book", pre_r, "rw_practised_out", book=bk_r, dry_run=True),
                  "5: a reworded set read in practice in the book refuses the trial that would read it")

            bk_w, bars_w = fresh_book("worn")
            for i in range(lt.MAX_REWORDED_USES):
                lb.append("note", {"note_type": "trial_registration", "reworded_wordings": rset.wordings(),
                                   "trial_sha256": f"w{i}", "brief_name": f"other_{i}", "brief_id": f"id{i}"}, path=bk_w)
            pre_w = tmp / "worn" / "prereg"
            lt.register_trial(pre_w, hP, brief, bars_w, e_trial, trial_name="worn", **{**KW, "book_path": bk_w})
            check(run_refused("at most 5", pre_w, "worn_out", book=bk_w, dry_run=True),
                  "5: a reworded set read five times in the book is refused: write a new set")

            bk_o, _bars_o = fresh_book("other")
            check(run_refused("calibration reading", prereg_car, "other_out", book=bk_o, dry_run=True),
                  "5: a trial is not read in a book that holds no calibration of its bars")

            def recopied(tag: str, one_more: bool) -> Path:
                """The home book's calibration, written again into another book: its three readings, a line more
                when asked, and its note. The note follows from that book as it did from the first."""
                bk = tmp / f"book_{tag}.jsonl"
                first = list(lb.entries(cal_book, limit=4))
                for ent in first[:3]:
                    lb.append(ent["kind"], ent["body"], path=bk)
                if one_more:
                    lb.append("note", {"note_type": "hello"}, path=bk)
                lb.append("note", first[3]["body"], path=bk)
                return bk

            bk_shift, bk_again = recopied("shifted", True), recopied("written_again", False)
            check(lcl.calibration_follows(lcl.calibration_note(bars, bk_shift), bk_shift, brief, bars) is None
                  and run_refused("registered against", third, "shift_out", book=bk_shift, dry_run=True)
                  and "is another line (line 4)" in msg[-1]
                  and run_refused("registered against", third, "again_book_out", book=bk_again, dry_run=True)
                  and lb.verify(bk_shift)["entries"] == 5 and lb.verify(bk_again)["entries"] == 4,
                  "5: a trial is read in the book it was registered against: the same calibration at another line, or "
                  "written again at the same one, is another book")

            # a trial follows a lead: an edit whose claim reading in the book read anything else has no trial there
            def claimed_book(tag: str, said: Optional[str]) -> Any:
                bk, bars_x = fresh_book(tag)
                reg_c = lb.append("note", {"note_type": "claim_registration", "brief_id": brief.id, "brief_name": brief.name,
                                           "edit_id": e_trial.id, "reading_index": 1,
                                           "main_dose": 2.0 if tag == "after_lead_other_dose" else 1.0}, path=bk)
                if said is not None:
                    lb.append("note", {"note_type": "claim_verdict", "brief_id": brief.id, "brief_name": brief.name,
                                       "edit_id": e_trial.id, "verdict": said, "registration_seq": reg_c["seq"],
                                       "reading_index": 1}, path=bk)
                pre = tmp / tag / "prereg"
                lt.register_trial(pre, hP, brief, bars_x, e_trial, trial_name=tag, **{**KW, "book_path": bk})
                return bk, pre

            bk_nl, pre_nl = claimed_book("after_no_lead", "did not replicate")
            bk_ne, pre_ne = claimed_book("after_unended", None)
            bk_ld, pre_ld = claimed_book("after_lead", "lead")
            bk_od, pre_od = claimed_book("after_lead_other_dose", "lead")
            check(run_refused("read it at dose 2", pre_od, "after_other_dose_out", book=bk_od, dry_run=True)
                  and "this trial is registered at dose 1" in msg[-1],
                  "5: a trial follows its edit's claim at the dose the claim read: a lead at one dose does not promote another")
            n_nl = lb.verify(bk_nl)["entries"]
            check(run_refused("follows a lead", pre_nl, "after_no_lead_out", book=bk_nl, dry_run=True)
                  and 'read "did not replicate"' in msg[-1] and lb.verify(bk_nl)["entries"] == n_nl
                  and run_refused("was registered and never ended", pre_ne, "after_unended_out", book=bk_ne, dry_run=True),
                  "5: a trial of an edit whose claim reading in the book did not read a lead, or never ended, is refused: "
                  "a second reading after anything but a lead would be a second draw at one question")
            t_ld = lt.run_trial_of_record(hP, pre_ld, out_dir=tmp / "after_lead_out", book_path=bk_ld, dry_run=True)
            SHEETS.append(t_ld["sheet"])
            check("follows: the claim reading of this edit at book line 5 (lead, read at dose 1)" in t_ld["sheet"]
                  and t_ld["meta"]["reading_index"] == 2 and t_ld["meta"]["earlier_fresh_readings"] == 1
                  and "which 1 earlier fresh reading of the brief in this book had already read" in t_ld["sheet"]
                  and "earlier fresh readings on this brief (1)" in t_ld["sheet"],
                  "5: after a claim that read a lead the trial is read, as the brief's second fresh reading, and its sheet "
                  "says which claim it follows and that the frames had been read")

            # a prompt arm too long for an item: found when the trial is registered, not after every reading of it
            brief_arm = lbr.Brief.from_dict({**brief.to_dict(), "name": "pronoun_pull_long_arm",
                                             "prompt_arms": [" ".join(["she"] * (lv.MAX_PIECES + 40))]})
            bk_arm, bars_arm = fresh_book("long_arm", brief_arm)
            rset_arm = lt.RewordedSet("for_long_arm", brief_arm.id, brief_arm.name, rset.items)
            check(refuses(lambda: lt.register_trial(tmp / "long_arm" / "prereg", hP, brief_arm, bars_arm, e_trial,
                                                    trial_name="long_arm", **{**KW, "book_path": bk_arm, "reworded": rset_arm}),
                          lt.TrialRefused, "cannot be read as registered") and not (tmp / "long_arm" / "prereg").exists(),
                  "5: a prompt arm too long to make an item of is found when the trial is registered, not after its readings")
            # an arm that alone makes an item, and behind the brief's own system turn (as it is read) does not
            sys_turn_x = " ".join(["quiet"] * 80)
            its_x = [d.item for d in brief.generator.draw("kept_back", 4, seed_bytes=b"y" * 32, tokenizer=hP.tokenizer)]

            def fits(n_words: int, system: Optional[str]) -> bool:
                b_x = lbr.Brief.from_dict({**brief.to_dict(), "name": "pronoun_pull_fit",
                                           "chat_settings": {**brief.chat_settings, "system": system},
                                           "prompt_arms": [" ".join(["she"] * n_words)]})
                try:
                    return lt._arm_items(hP, b_x, its_x) == len(its_x)
                except lv.Refused:
                    return False

            n_fit = next(n for n in range(lv.MAX_PIECES, 0, -1) if fits(n, None)) - 20
            brief_sys = lbr.Brief.from_dict({**brief.to_dict(), "name": "pronoun_pull_sys_arm",
                                             "chat_settings": {**brief.chat_settings, "system": sys_turn_x},
                                             "prompt_arms": [" ".join(["she"] * n_fit)]})
            bk_sys, bars_sys = fresh_book("sys_arm", brief_sys)
            rset_sys = lt.RewordedSet("for_sys_arm", brief_sys.id, brief_sys.name, rset.items)
            check(n_fit > 100 and fits(n_fit, None) and not fits(n_fit, sys_turn_x)
                  and lcl.prompt_arm_system(brief_sys, "x") == sys_turn_x + "\nx" and lcl.prompt_arm_system(brief, "x") == "x"
                  and refuses(lambda: lt.register_trial(tmp / "sys_arm" / "prereg", hP, brief_sys, bars_sys, e_trial,
                                                        trial_name="sys_arm", **{**KW, "book_path": bk_sys, "reworded": rset_sys}),
                              lt.TrialRefused, "cannot be read as registered") and not (tmp / "sys_arm" / "prereg").exists(),
                  f"5: a prompt arm of {n_fit} words that alone makes an item, and behind the brief's own system turn does "
                  "not, is found when the trial is registered: an arm is asked as it will be read")

            # a trial that is entered in the book always gets a verdict note, whatever ends it
            bk_err, bars_err = fresh_book("errors")
            pre_err = tmp / "errors" / "prereg"
            lt.register_trial(pre_err, hP, brief, bars_err, e_trial, trial_name="errors", **{**KW, "book_path": bk_err})
            read_was = lv.read

            def fell_over(*a: Any, **k: Any) -> Any:
                raise RuntimeError("the card fell over")

            lv.read = fell_over
            try:
                err_raised = refuses(lambda: lt.run_trial_of_record(hP, pre_err, out_dir=tmp / "errors_out", book_path=bk_err,
                                                                    dry_run=True), RuntimeError, "fell over")
            finally:
                lv.read = read_was
            err_notes = [e["body"] for e in lb.entries(bk_err) if e["kind"] == "note"]
            err_hist = lcl.brief_history(brief.id, bk_err, name=brief.name)
            check(err_raised and err_notes[-2]["note_type"] == "trial_registration"
                  and err_notes[-1]["note_type"] == lt.NOTE_DRY_RUN and err_notes[-1]["verdict"] == "void"
                  and err_notes[-1]["ended"] == "error" and "RuntimeError: the card fell over" in err_notes[-1]["reasons"][0]
                  and lcl.prior_trials(err_hist)[0]["verdict"] == "void" and err_hist["fresh_count"] == 1
                  and (tmp / "errors_out" / "trial_seed.json").is_file() and not (tmp / "errors_out" / "trial_result.json").exists()
                  and lb.verify(bk_err)["ok"]
                  and run_refused("read once", pre_err, "errors_out_2", book=bk_err, dry_run=True),
                  "5: a trial that ends on an error after it was entered is written void in the book, the error is raised "
                  "again, its seed is kept, and it is not read a second time")

            # the conditions of record, looked at in a small repository that holds a registration folder
            bk_rec = tmp / "book_of_record.jsonl"
            brief_rec = lbr.Brief.from_dict({**brief.to_dict(), "name": "pronoun_pull_of_record"})
            bars_rec = lcl.calibrate_bars(hP, brief_rec, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
                                          rows=ROWS, seed=0, book_path=bk_rec)
            own_rec = lt.RewordedSet.from_dict({**_own_items(), "brief_id": brief_rec.id, "brief_name": brief_rec.name})
            repo = tmp / "repo"
            pre_rec = repo / "trials" / "rec" / "prereg"
            lt.register_trial(pre_rec, hP, brief_rec, bars_rec, e_trial, trial_name="rec",
                              **{**KW, "book_path": bk_rec, "reworded": own_rec})
            _git(repo, "init", "-q")
            (repo / "README").write_text("a repository for a trial\n")
            _git(repo, "add", "README")
            _git(repo, "commit", "-q", "-m", "first")
            cfg_rec = lt.verify_registration(pre_rec)["trial"]
            level_book = lb.default_book_path()

            def cond(**kw: Any) -> Dict[str, Any]:
                return lt.record_conditions(pre_rec, cfg_rec, repo_root=repo, **{"book": level_book, **kw})

            def why(c: Dict[str, Any]) -> List[str]:
                return [r for r in c["reasons"] if not r.startswith("this process")]

            cond_uncommitted = cond()
            _git(repo, "add", "-A")
            cond_staged = cond()
            _git(repo, "commit", "-q", "-m", "registration")
            cond_now = cond()
            (pre_rec / "RULES.md").write_text((pre_rec / "RULES.md").read_text() + " ")
            cond_touched = cond()
            _git(repo, "checkout", "-q", "--", ".")
            check(any("uncommitted" in r for r in cond_uncommitted["reasons"])
                  and any("in no commit" in r for r in cond_uncommitted["reasons"])
                  and any("in no commit" in r for r in cond_staged["reasons"])
                  and any("differs from what is committed" in r for r in cond_touched["reasons"])
                  and why(cond_now) == []
                  and cond_now["checkout"]["dirty"] is False and cond_now["checkout"]["folder_committed"] is True
                  and cond_now["checkout"]["sources_match"] is True and cond_now["checkout"]["default_book"] is True
                  and cond_now["checkout"]["tracked_changed"] == 0 and cond_now["checkout"]["hidden"] == 0,
                  "5: the conditions of record see a folder that is not committed, only staged, or changed after, and a "
                  f"clean checkout that holds it committed ({cond_uncommitted['reasons']}; {cond_touched['reasons']}; {cond_now['reasons']})")
            # what a plain `git status` would not show: a file git is told to overlook, and a file it does not know
            # where the configuration tells status to leave such files out
            _git(repo, "update-index", "--assume-unchanged", "README")
            (repo / "README").write_text("changed, and git told to overlook it\n")
            cond_hidden = cond()
            _git(repo, "update-index", "--no-assume-unchanged", "README")
            _git(repo, "checkout", "-q", "--", "README")
            _git(repo, "update-index", "--skip-worktree", "trials/rec/prereg/RULES.md")
            (pre_rec / "RULES.md").write_text("other rules\n")
            cond_skipped = cond()
            _git(repo, "update-index", "--no-skip-worktree", "trials/rec/prereg/RULES.md")
            _git(repo, "checkout", "-q", "--", ".")
            _git(repo, "config", "status.showUntrackedFiles", "no")
            (repo / "sitecustomize.py").write_text("print('run before anything else')\n")
            plain_status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout
            cond_untracked = cond()
            (repo / "sitecustomize.py").unlink()
            _git(repo, "config", "--unset", "status.showUntrackedFiles")
            check(any("told to overlook changes to 1 file(s) of the checkout (README)" in r for r in cond_hidden["reasons"])
                  and cond_hidden["checkout"]["dirty"] is False
                  and any("told to overlook" in r for r in cond_skipped["reasons"])
                  and any("differs from what is committed" in r for r in cond_skipped["reasons"])
                  and plain_status == "" and any("uncommitted path(s) (sitecustomize.py" in r for r in cond_untracked["reasons"])
                  and cond_untracked["checkout"]["tracked_changed"] == 0 and cond_untracked["checkout"]["changed"] == 1
                  and why(cond()) == [],
                  "5: a file git is told to overlook, and a file git does not know where status is set to leave those "
                  f"out, are both found ({why(cond_hidden)}; {why(cond_untracked)})")
            (repo / ".git" / "info" / "exclude").write_text("numpy.py\nscripts/\n")
            (repo / "numpy.py").write_text("raise SystemExit('not the numpy the registration ran')\n")
            (repo / "scripts").mkdir()
            (repo / "scripts" / "levelling_trial.py").write_text("print('another runner')\n")
            plain_ignored = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                                           capture_output=True, text=True).stdout
            cond_ignored = cond()
            (repo / "numpy.py").unlink()
            shutil.rmtree(repo / "scripts")
            (repo / ".git" / "info" / "exclude").write_text("")
            hidden_was = lt.hidden_from_status
            lt.hidden_from_status = lambda root=None: None
            try:
                cond_unasked = cond()
            finally:
                lt.hidden_from_status = hidden_was
            check(plain_ignored == "" and cond_ignored["checkout"]["dirty"] is False
                  and cond_ignored["checkout"]["unlisted_code"] == 2
                  and any("2 file(s) of code the runner would import that are in no commit (numpy.py, scripts/levelling_trial.py)" in r
                          for r in cond_ignored["reasons"])
                  and any("could not be asked" in r for r in cond_unasked["reasons"]) and cond_unasked["checkout"]["hidden"] is None
                  and why(cond()) == [] and cond()["checkout"]["unlisted_code"] == 0,
                  "5: Python where the runner imports from that git was told to ignore is found (a status lists none of it), "
                  f"and a question git could not be asked is not a condition met ({why(cond_ignored)}; {why(cond_unasked)})")
            # the same in the forms Python imports that are not source: compiled Python where a source file would
            # stand, an extension module, a package whose `__init__` is compiled. This repository's own .gitignore
            # ignores `*.pyc`, so a status lists none of them.
            so_name = "torch" + next(s for s in lt._CODE_SUFFIXES if s.endswith(".so") or s.endswith(".pyd"))
            (repo / ".git" / "info" / "exclude").write_text("*.pyc\n*.so\n*.pyd\nscripts/\ncolorsys/\nbackend/__pycache__/\n")
            (repo / "numpy.pyc").write_bytes(b"not python")
            (repo / "scripts").mkdir()
            (repo / "scripts" / so_name).write_bytes(b"not a module")
            (repo / "colorsys").mkdir()
            (repo / "colorsys" / "__init__.pyc").write_bytes(b"not python")
            had_backend = (repo / "backend").is_dir()
            (repo / "backend" / "__pycache__").mkdir(parents=True, exist_ok=True)
            (repo / "backend" / "__pycache__" / "levelling.cpython-312.pyc").write_bytes(b"a cache beside its source")
            plain_compiled = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                                            capture_output=True, text=True).stdout
            cond_compiled = cond()
            (repo / "numpy.pyc").unlink()
            shutil.rmtree(repo / "scripts")
            shutil.rmtree(repo / "colorsys")
            shutil.rmtree(repo / "backend" / "__pycache__" if had_backend else repo / "backend")
            (repo / ".git" / "info" / "exclude").write_text("")
            check(plain_compiled == "" and cond_compiled["checkout"]["dirty"] is False
                  and cond_compiled["checkout"]["unlisted_code"] == 3
                  and any(f"3 file(s) of code the runner would import that are in no commit (colorsys/__init__.pyc, numpy.pyc, "
                          f"scripts/{so_name})" in r for r in cond_compiled["reasons"])
                  and why(cond()) == [] and cond()["checkout"]["unlisted_code"] == 0,
                  f"5: compiled Python and an extension module where the runner imports from, which a status does not "
                  f"list, are found as source is, and a cache beside its source is not counted ({why(cond_compiled)})")
            outside = lt.record_conditions(prereg_dir, v_ok["trial"], repo_root=repo, book=level_book)
            check(any("outside the checkout" in r for r in outside["reasons"]),
                  "5: a registration folder outside the checkout is in no commit, and is not of record")
            moved = lt.record_conditions(pre_rec, {**cfg_rec, "sources": {k: "0" * 64 for k in cfg_rec["sources"]}},
                                         repo_root=repo, book=level_book)
            added = lt.record_conditions(pre_rec, {**cfg_rec, "sources": {**cfg_rec["sources"], "backend/gone_since.py": "0" * 64}},
                                         repo_root=repo, book=level_book)
            check(any("runner or reader is not the one registered" in r for r in moved["reasons"])
                  and any("runner or reader is not the one registered (backend/gone_since.py)" in r for r in added["reasons"])
                  and added["checkout"]["sources_moved"] == ["backend/gone_since.py"],
                  "5: a runner or reader changed, added or gone since registration is not of record")
            check(any("a side book and not the level book" in r for r in cond(book=bk_rec)["reasons"])
                  and cond(book=bk_rec)["checkout"]["default_book"] is False
                  and any("handed in, not drawn" in r for r in cond(seed_given=True)["reasons"])
                  and any("run the model forward" in r for r in cond()["reasons"]),
                  "5: a side book, a seed handed in, and a process that has already run the model are each a reason a "
                  "trial is not of record")

            # a trial of record, read whole: a repository of its own that holds the code as it stands here, a level
            # book of its own, the registration made by one process and committed, the trial read by another
            real = tmp / "repo_of_record"
            shutil.copytree(ROOT_DIR / "backend", real / "backend", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (real / "scripts").mkdir()
            for src_py in sorted((ROOT_DIR / "scripts").glob("*.py")):
                shutil.copy2(src_py, real / "scripts" / src_py.name)
            shutil.copy2(ROOT_DIR / ".gitignore", real / ".gitignore")
            _git(real, "init", "-q")
            _git(real, "add", "-A")
            _git(real, "commit", "-q", "-m", "the code as it stands")
            pre_real = real / "trials" / "rec" / "prereg"
            head = (
                "import os, sys, json\n"
                "os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['HF_HUB_OFFLINE'] = '1'\n"
                f"sys.path.insert(0, {str(real)!r}); sys.path.insert(0, {str(real / 'scripts')!r})\n"
                "import torch; torch.set_num_threads(1)\n"
                "from pathlib import Path\n"
                "from backend import levelling as lv, level_trial as lt, level_claim as lcl, level_briefs as lbr, weight_edits as we\n"
                "import check_dequant as cd\n"
                "h = lv.handle(cd.packed_stand_in()[0])\n"
                f"pre = Path({str(pre_real)!r})\n"
            )
            code_reg = head + (
                "brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2)\n"
                f"rows = {ROWS!r}\n"
                "bars = lcl.calibrate_bars(h, brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2, "
                "rows=rows, seed=0)\n"
                "edit = we.build(h.lm, [we.silence(2, [3, 7])], model_id=h.model_id, weights_id=h.weights_id)\n"
                f"own = lt.RewordedSet.from_dict({{**{_own_items()!r}, 'brief_id': brief.id, 'brief_name': brief.name}})\n"
                "reg = lt.register_trial(pre, h, brief, bars, edit, trial_name='rec', reworded=own, controls_per_family=39, "
                "reworded_controls=19, n_target=4, n_neutral=2, n_near=2, n_panel=2, rows=rows, match_tolerance=20.0)\n"
                "print('REG ' + reg['trial_sha256'])\n"
            )
            code_run = head + (
                "try:\n"
                f"    r = lt.run_trial_of_record(h, pre, out_dir=Path({str(tmp / 'rec_out')!r}))\n"
                "except lt.TrialRefused as exc:\n"
                "    print('REFUSED ' + str(exc)); sys.exit(0)\n"
                "print('RESULT ' + json.dumps({'of_record': r['of_record'], 'verdict': r['verdict'], "
                "'why': r['not_of_record'], 'sheet': r['sheet'], 'book': r['book']['path'], 'sentence': r['claim_sentence']}))\n"
            )
            env = {k: v for k, v in os.environ.items() if k != lb.BOOK_ENV}
            env.update({"XDG_DATA_HOME": str(tmp / "xdg"), we.EDITS_ENV: str(tmp / "saved_edits")})

            def in_real(code: str) -> subprocess.CompletedProcess:
                return subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, env=env,
                                      cwd=str(real), timeout=900)

            p_reg = in_real(code_reg)
            reg_line = next((ln for ln in p_reg.stdout.splitlines() if ln.startswith("REG ")), None)
            check(reg_line is not None and pre_real.is_dir(),
                  f"5: a process of its own calibrates and registers a trial in the repository ({p_reg.stderr.strip()[-300:]})")
            p_early = in_real(code_run)  # the folder is there and not yet committed
            early = next((ln for ln in p_early.stdout.splitlines() if ln.startswith("REFUSED ")), "")
            _git(real, "add", "-A")
            _git(real, "commit", "-q", "-m", "the registration")
            real_book = tmp / "xdg" / "theodolite" / "level_book.jsonl"
            n_real = lb.verify(real_book)["entries"]
            check("would not be of record" in early and "in no commit" in early and "uncommitted path" in early
                  and "first thing its process" not in early and "side book" not in early and n_real == 4,
                  f"5: before its folder is committed the trial is refused, for that and nothing else, and enters nothing ({early[:200]})")
            proc = in_real(code_run)
            got_line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
            got = json.loads(got_line[7:]) if got_line else {}
            rec_sheet = str(got.get("sheet", ""))
            check(got.get("of_record") is True and got.get("why") == [] and got.get("verdict") in lt.PASSED + ("failed",)
                  and rec_sheet.startswith("=== TRIAL OF RECORD: rec ===")
                  and "process: had asked nothing of the model before the trial" in rec_sheet
                  and "checkout clean | registration folder committed: yes | runner and reader as registered: yes" in rec_sheet
                  and "book: level_book.jsonl (the level book of the environment this ran in)" in rec_sheet
                  and "from a file given at registration" in rec_sheet
                  and Path(str(got.get("book"))) == real_book,
                  "5: read first in a process of its own, from a clean checkout of the code that holds its folder "
                  "committed, in the level book, a trial is of record "
                  f"({proc.stderr.strip()[-300:] if not got else rec_sheet.splitlines()[0]})")
            SHEETS.append(rec_sheet)
            rec_entries = list(lb.entries(real_book))
            check(rec_entries[-1]["body"].get("note_type") == lt.NOTE_OF_RECORD and lb.verify(real_book)["ok"]
                  and rec_entries[4]["body"].get("of_record") is True and rec_entries[4]["body"]["process"]["forwards"] == 0,
                  "5: and its verdict is entered in the book as a trial of record")
            rec_card = lroutes.build_replay_card(rec_entries[-1])
            check(rec_card["of_record"] is True and rec_card["status"] == got.get("verdict"),
                  "5: only a trial of record's card shows its verdict as a verdict")
            again_run = in_real(code_run.replace("rec_out", "rec_out_2"))
            check(any(ln.startswith("REFUSED Read once") for ln in again_run.stdout.splitlines())
                  and lb.verify(real_book)["entries"] == len(rec_entries),
                  "5: and it is read once: a second process is refused, and enters nothing")

            # the pass, whole. No stand-in reads `passed`, so the book of the trial of record above is written again
            # with figures that would pass, judged again by the one path a trial is judged by, and the sentence and
            # the sheet of record are made from that judgement and the run's own account of itself.
            def would_pass(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if ent["kind"] != "paired" or not str(body.get("name", "")).startswith("trial_"):
                    return
                for term in ((body.get("controls") or {}).get("coverage") or {}).get("terms") or ():
                    if "min_eligible" in term:
                        term["min_eligible"] = 500
                for it in body["items"]:
                    for cb in it["classes"].values():
                        if isinstance((cb.get("floor") or {}).get("odds"), dict):
                            cb["floor"]["odds"]["mean_abs"] = 0.01
                        for arm in cb["arms"]:
                            key = str(arm["key"])
                            if "delivered" in arm and (key.startswith("control:") or key == "edit:1.0"):
                                arm["delivered"] = 0.01
                                arm["delivered_end"] = 0.01
                            if "odds" in arm["metrics"] and (key in ("edit:1.0", "edit:-1.0") or key.startswith("control:")):
                                arm["metrics"]["odds"]["mean"] = -1.5 if key == "edit:1.0" else 1.4 if key == "edit:-1.0" else 0.01
                        fl = cb.get("folded")
                        if isinstance(fl, dict) and fl.get("taken") and "odds" in (fl.get("metrics") or {}):
                            fl["metrics"]["odds"]["mean"] = -1.5
                            fl["gap"]["odds"]["abs"] = 0.0
                for cb in body["summary"]["classes"].values():
                    for key, arm in cb["arms"].items():
                        for metric, val in (("kl", 0.001), ("own", 0.0), ("odds", -0.02)):
                            if isinstance(arm.get(metric), dict) and "mean_over_items" in arm[metric] \
                                    and not str(body["name"]).endswith(("_matched", "_same_site", "_reworded")):
                                arm[metric]["mean_over_items"] = val if key.startswith("edit:1.0") else 0.0

            reg_real = lt.verify_registration(pre_real)
            res_real = json.loads((tmp / "rec_out" / "trial_result.json").read_text())
            pass_book = tmp / "would_pass.jsonl"
            for ent in lb.entries(real_book):
                body = json.loads(json.dumps(ent["body"]))
                would_pass(ent, body)
                lb.append(ent["kind"], body, path=pass_book)
            rj_pass = lt.rejudge_trial(pass_book, reg_real["trial_sha256"], reg_real)
            jp = rj_pass["judged"]
            sent_live = lt.trial_sentence(reg_real["brief"], reg_real["trial"], jp["fresh_judged"], jp["trial"], res_real["meta"]) or ""
            sheet_live = lt.render_trial_sheet(reg_real["brief"], reg_real["bars"], reg_real["trial"], jp["fresh_judged"],
                                               jp["trial"], res_real["meta"], sent_live or None)
            check(rj_pass["verdict"] in lt.PASSED and jp["fresh_judged"]["verdict"] in lcl.LEADS
                  and rj_pass["as_noted"] is (rj_pass["verdict"] == got.get("verdict")),
                  f"5: a book written again with figures that would pass, judged again, reads '{rj_pass['verdict']}' "
                  f"({(jp['trial'].get('reasons') or [''])[:2]})")
            check(sent_live.startswith('Trial of record "rec" (registration ' + reg_real["trial_sha256"][:12])
                  and "in the level book of the environment it ran in, at line 4" in sent_live and "on 4 fresh items" in sent_live
                  and "as read, 39 of 39 changed the stream within that factor of what the edit changed it, over the answer "
                      "and at the position the metric is read" in sent_live
                  and "over the sets each neuron had at least 39 different stand-ins" in sent_live
                  and "gating batteries passed (neutral text 2 of 2; near misses 2 of 2; labelled panel 2 of 2 items read)" in sent_live
                  and sent_live.count("-1.5000") >= 6 and "fresh reading 1 of 2" in sent_live
                  and "0 of 39 matched other-site" in sent_live and "0 of 19 matched control edits moved as much" in sent_live
                  and "the same bits on 4 of 4 items" in sent_live and "No claim reading of this edit stands in this book" in sent_live
                  and "which no earlier fresh reading in this book had read" in sent_live
                  and (" Passed. " in sent_live or " Passed, matched by a cheaper knob. " in sent_live)
                  # a figure that was not there prints as "none" or "None" (a stand-in has no weights id, and says so)
                  and " none" not in sent_live.replace("(none in the brief)", "")
                  and "None" not in sent_live.replace("(weights None)", ""),
                  f"5: and the sentence of record made from that judgement holds every figure it speaks of ({sent_live})")
            check(sheet_live.startswith("=== TRIAL OF RECORD: rec ===") and f"verdict: {rj_pass['verdict']}\n" in sheet_live
                  and "--- the sentence of record ---" in sheet_live and "not of record" not in sheet_live
                  and "agrees with the per-row form: True | clears the bar: True, both halves: True | ok=True" in sheet_live
                  and KEPT_OUT_RE.search(sheet_live) is None,
                  "5: and so does its sheet, with the sentence under its own heading")
            SHEETS.append(sheet_live)

            # Each rule that sets a control edit aside, or counts the sets as fewer draws, decides a verdict: the book
            # that would pass, written again with one thing changed, no longer passes, and says why.
            def judged_with(tag: str, more: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> Dict[str, Any]:
                bk_x = tmp / f"would_pass_{tag}.jsonl"
                for ent in lb.entries(real_book):
                    body = json.loads(json.dumps(ent["body"]))
                    would_pass(ent, body)
                    if ent["kind"] == "paired" and str(body.get("name", "")).endswith("_matched"):
                        more(ent, body)
                    lb.append(ent["kind"], body, path=bk_x)
                return lt.rejudge_trial(bk_x, reg_real["trial_sha256"], reg_real)

            def arms_in(body: Dict[str, Any], what: str) -> Any:
                for it in body["items"]:
                    for cb in it["classes"].values():
                        for arm in cb["arms"]:
                            if str(arm["key"]).startswith(what):
                                yield arm

            def quiet_at_end(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                for arm in arms_in(body, "control:"):
                    arm["delivered_end"] = 0.0001  # a hundredth of the edit's at the last position; the same over the answer

            def few_stand_ins(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                for term in body["controls"]["coverage"]["terms"]:
                    term["min_distinct"] = 18

            def edit_no_figure(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                for arm in arms_in(body, "edit:1.0"):
                    arm["delivered"] = None

            def edit_delivered_nothing(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                for arm in arms_in(body, "edit:1.0"):
                    arm["delivered"] = 0.0

            def old_reading(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                for arm in arms_in(body, ""):
                    arm.pop("delivered_end", None)

            in_pass = jp["inputs"]["classes"]["small"]
            got_x = {tag: judged_with(tag, fn) for tag, fn in (
                ("quiet_at_end", quiet_at_end), ("few_stand_ins", few_stand_ins), ("edit_no_figure", edit_no_figure),
                ("edit_delivered_nothing", edit_delivered_nothing), ("old_reading", old_reading))}

            def small_of(tag: str) -> Dict[str, Any]:
                return got_x[tag]["judged"]["fresh_judged"]["classes"]["small"]

            def why_x(tag: str) -> str:
                return " | ".join(got_x[tag]["judged"]["trial"]["reasons"])

            check(all(isinstance(v, float) for v in in_pass["controls_as_read"]["other_sites_matched"])
                  and in_pass["controls_aside"]["other_sites_matched"] == [False] * 39
                  and in_pass["controls"]["other_sites_matched"] == in_pass["controls_as_read"]["other_sites_matched"]
                  and got_x["quiet_at_end"]["verdict"] == "failed"
                  and got_x["quiet_at_end"]["fresh_verdict"] == "no better than control edits"
                  and small_of("quiet_at_end")["delivered"]["other_sites_matched"]["not_comparable"] == 39
                  and small_of("quiet_at_end")["delivered"]["other_sites_matched"]["within_tolerance"] == 0
                  and small_of("quiet_at_end")["controls"]["other_sites_matched"]["at_least"] == 39
                  and got_x["quiet_at_end"]["judged"]["inputs"]["classes"]["small"]["controls_aside"]["other_sites_matched"] == [True] * 39
                  and got_x["quiet_at_end"]["figures"] == rj_pass["figures"]
                  and "did not change the stream as the edit did" in why_x("quiet_at_end"),
                  "5: control edits that changed the stream as the edit did over the answer and a hundredth as far at the "
                  "last position, where the metric is read, are each counted as moving as much: a book that would pass "
                  f"then fails, with every figure as read the same ({got_x['quiet_at_end']['verdict']}: {why_x('quiet_at_end')[:150]})")
            check(got_x["few_stand_ins"]["verdict"] == "failed"
                  and small_of("few_stand_ins")["controls"]["other_sites_matched"]["effective_count"] == 18
                  and small_of("few_stand_ins")["controls"]["other_sites_matched"]["at_least"] == 0
                  and small_of("few_stand_ins")["controls"]["other_sites_matched"]["p_fw"] > 0.05
                  and "the 39 control edits are counted as 18 separate draws, too few to reach the level whatever they read"
                      in why_x("few_stand_ins")
                  and jp["fresh_judged"]["classes"]["small"]["controls"]["other_sites_matched"]["effective_count"] == 39,
                  "5: control sets in which one neuron had 18 different stand-ins are counted as 18 draws: with none moving "
                  f"as much the level is not reached, and the reason says the count alone settled it ({why_x('few_stand_ins')[:170]})")
            check(all(got_x[tag]["verdict"] == "failed"
                      and small_of(tag)["delivered"]["other_sites_matched"]["not_comparable"] == 39
                      and small_of(tag)["delivered"]["other_sites_matched"]["within_tolerance"] == 0
                      for tag in ("edit_no_figure", "edit_delivered_nothing", "old_reading"))
                  and "outside the matching tolerance of the edit's, or with no figure" in lcl.delivered_words(
                      small_of("edit_no_figure")["delivered"]["other_sites_matched"],
                      small_of("edit_no_figure")["controls"]["other_sites_matched"])
                  and lcl.delivered_apart(None, 0.5, 2.0) and lcl.delivered_apart(0.5, None, 2.0)
                  and lcl.delivered_apart(0.0, 0.5, 2.0) and not lcl.delivered_apart(0.0, 0.0, 2.0)
                  and not lcl.delivered_apart(1.0, 0.5, 2.0) and not lcl.delivered_apart(1.0, 2.0, 2.0)
                  and lcl.delivered_apart(1.0, 0.4999, 2.0) and lcl.delivered_apart(1.0, 2.0001, 2.0),
                  "5: when the edit has no delivered figure, when it delivered nothing and its control edits something, and "
                  "when a reading records no figure at the last position for a metric read there, no control edit is taken "
                  "to have changed the stream as the edit did: each is counted as moving as much, and the trial fails")


            # the command line
            check(lt_cli.main(["verify", str(prereg_dir)]) == 0, "5: scripts/levelling_trial.py verify exits 0")
            rw_cli_out = tmp / "cli_reworded.json"
            check(lt_cli.main(["reworded", "pronoun_pull", "--out", str(rw_cli_out), "--stand-in"]) == 0
                  and rw_cli_out.is_file(),
                  "5: scripts/levelling_trial.py reworded writes the starter reworded set")
            brief_cli = lbr.pronoun_pull_brief(hP.tokenizer, max_fresh_readings=1)
            bk_cli, bars_cli = fresh_book("cli", brief_cli)
            (tmp / "own_items.json").write_text(json.dumps(_own_items()))
            lbr.save_brief(brief_cli, tmp / "cli_brief.json")
            lcl.save_bars(bars_cli, tmp / "cli_bars.json")
            (tmp / "cli_edit.json").write_text(json.dumps({"terms": [{"layer": 2, "silence": [3, 7]}]}))
            cli_pre, cli_out = tmp / "cli_prereg", tmp / "cli_out"
            rc_rw = lt_cli.main(["reworded", str(tmp / "cli_brief.json"), "--from", str(tmp / "own_items.json"),
                                 "--out", str(tmp / "cli_own.json"), "--packed-stand-in"])
            reg_args = ["register", str(tmp / "cli_brief.json"), "--bars", str(tmp / "cli_bars.json"),
                        "--edit", str(tmp / "cli_edit.json"), "--book", str(bk_cli), "--controls", "19", "--n-target", "4",
                        "--n-neutral", "2", "--n-near", "2", "--n-panel", "2", "--rows-small", "10", "--rows-large", "16",
                        "--match-tolerance", "20", "--packed-stand-in"]
            rc_reg = lt_cli.main(reg_args + ["--name", "cli", "--reworded", str(tmp / "cli_own.json"), "--out", str(cli_pre)])
            rc_noreg = lt_cli.main(reg_args + ["--name", "cli2", "--out", str(tmp / "cli_prereg2")])
            rc_nobook = lt_cli.main(["run", str(cli_pre), "--out", str(cli_out), "--packed-stand-in"])
            rc_run = lt_cli.main(["run", str(cli_pre), "--out", str(cli_out), "--book", str(bk_cli), "--packed-stand-in"])
            rc_dry = lt_cli.main(["run", str(cli_pre), "--out", str(cli_out), "--book", str(bk_cli), "--packed-stand-in",
                                  "--dry-run"])
            check(rc_rw == 0 and lt.load_reworded(tmp / "cli_own.json").id and rc_reg == 0 and rc_noreg == 2,
                  "5: the command line hashes the owner's own items, registers with them, and refuses a trial without a set")
            check(rc_nobook == 2 and rc_run == 2 and rc_dry == 0 and (cli_out / "TRIAL_SHEET.md").is_file(),
                  "5: the command line's run refuses a stand-in with no --book, refuses what would not be of record, and reads a dry run when asked")

            # ------------------------------------------------------------------ 6. the export
            section("6. The export: where it may go, what its keys say, that the file is the edit, off the model, no agent's reach")
            sess_fake = tmp / "fake_session"
            sess_fake.mkdir()
            (sess_fake / "session.json").write_text("{}")
            other_repo = tmp / "another_checkout"
            other_repo.mkdir()
            _git(other_repo, "init", "-q")
            for label, bad_path in (
                ("inside this checkout", ROOT_DIR / "evil.gguf"),
                ("inside another git checkout", other_repo / "sub" / "evil.gguf"),
                ("inside prereg/", prereg_dir / "evil.gguf"),
                ("inside a session's folder", sess_fake / "evil.gguf"),
            ):
                check(refuses(lambda: lex.refuse_unsafe_export_path(bad_path), lex.ExportRefused),
                      f"6: export refuses a path {label}")

            synth_gguf = tmp / "synthetic-gemma4.gguf"
            gguf_weights = _write_synthetic_gemma4_gguf(synth_gguf, n_layers, D, N, seed=1234)
            target_gguf = lex.TargetBase(gguf_path=synth_gguf)
            target_mod = lex.TargetBase(lm=hP.lm, label="packed-standin")
            other_arch = tmp / "not-gemma.gguf"
            _write_synthetic_gemma4_gguf(other_arch, 1, D, N, arch="llama")
            check(refuses(lambda: lex.TargetBase(gguf_path=other_arch), lex.ExportRefused, "gemma 4"),
                  "6: a GGUF of another architecture is not a base an adapter of a Gemma 4 edit is compiled against")
            check(target_mod.is_base_of(e_trial) is True and target_gguf.is_base_of(e_trial) is None,
                  "6: a base is known to be the edit's own by its pins; a GGUF is another object")
            e_w1 = we.Edit(e_trial.id, e_trial.terms, {**e_trial.base, "weights_id": "w1"}, "", {})
            st_same, st_other = (lex.TargetBase(lm=hP.lm, weights_id=w).standing(e_w1) for w in ("w1", "w2"))
            check(st_same == (True, "yes: the same weights id, and every touched linear by its pin")
                  and st_other[0] is False and "the weights id is another" in st_other[1]
                  and target_mod.standing(e_trial)[0] is None and target_mod.standing(e_trial)[1].startswith("its touched linears are, by their pins")
                  and "not compared" in target_mod.standing(e_trial)[1]
                  and target_gguf.standing(e_trial) == (None, "another object (a GGUF)")
                  and lex.TargetBase(lm=hF.lm).standing(e_trial)[0] is False,
                  "6: a served base is called the edit's base only when the whole of it was shown the same (its weights "
                  "id); touched linears that agree by their pins are called that and no more")

            e_mix = build([we.silence(2, [3, 7]), we.scale_rows(3, "up_proj", [1, 4], 0.5)], "col+row export")
            out_gguf = tmp / "owner_exports" / "recompiled_mix.gguf"
            exp_res = lex.export_lora_adapter(e_mix, target_gguf, out_path=out_gguf, dose=1.0,
                                              lm_for_implant=hP.lm, W_U_for_implant=hP.W_U)
            check(Path(exp_res["gguf_path"]).is_file() and Path(exp_res["manifest_path"]).is_file(),
                  "6: export_lora_adapter writes .gguf and companion .manifest.json")
            check(exp_res["recompiled_from_recipe"] is True
                  and "blk.3.ffn_up.weight" in exp_res["unverified_llama_cpp_tensors"],
                  "6: rebuilt export marks recompiled_from_recipe=True and flags ffn_up as unverified in llama.cpp")
            check(refuses(lambda: lex.export_lora_adapter(e_mix, target_gguf, out_path=out_gguf), lex.ExportRefused),
                  "6: export_lora_adapter refuses overwriting existing file without --force")
            check(refuses(lambda: lex.export_lora_adapter(e_mix, target_gguf, out_path=tmp / "owner_exports"), lex.ExportRefused, "folder"),
                  "6: a folder given as the adapter's path is refused, not written beside")

            ver_gguf = lex.verify_exported_lora(out_gguf, edit=e_mix, target=target_gguf, dose=1.0)
            check(ver_gguf["ok"] and (ver_gguf["max_abs_err"] or 0.0) < 1e-5,
                  f"6: verify_exported_lora: the file is the edit on the target (err={ver_gguf['max_abs_err']})")
            lora_ab, meta_ab = el.read_lora(out_gguf)
            A_down, B_down = lora_ab["blk.2.ffn_down.weight"]
            W_down_gguf = gguf_weights["blk.2.ffn_down.weight"]  # [D, N]
            W_down_after = W_down_gguf + (B_down @ A_down)
            check(float(np.max(np.abs(W_down_after[:, [3, 7]]))) < 1e-6
                  and float(np.max(np.abs(W_down_after[:, 0] - W_down_gguf[:, 0]))) == 0.0,
                  "6: recompiled LoRA zeroes silenced columns of the SERVED GGUF exactly while leaving other columns untouched")
            A_up, B_up = lora_ab["blk.3.ffn_up.weight"]
            W_up_gguf = gguf_weights["blk.3.ffn_up.weight"]  # [N, D]
            W_up_after = W_up_gguf + (B_up @ A_up)
            check(float(np.max(np.abs(W_up_after[[1, 4], :] - 0.5 * W_up_gguf[[1, 4], :]))) < 1e-6,
                  "6: recompiled row-scale LoRA halves scaled rows of the SERVED GGUF exactly")
            derived = str(meta_ab.get("theodolite.derived_from"))
            check("blk.3.ffn_up.weight: lora_a holds the change to 2 row(s)" in derived
                  and "blk.2.ffn_down.weight: lora_b holds the change to 2 column(s)" in derived
                  and "each lora_a holds only ones and noughts" not in derived,
                  "6: the file says of each tensor which half holds the change (a row adapter's lora_a is not ones and noughts)")
            check(meta_ab.get("theodolite.levelling.trial_verdict") == "none given"
                  and meta_ab.get("theodolite.levelling.trial_judged_again") == "no trial given"
                  and str(meta_ab.get("theodolite.levelling.gate")).startswith("built and gated by the kernel")
                  and meta_ab.get("theodolite.levelling.compiled_against_is_edit_base") == "another object (a GGUF)"
                  and meta_ab.get("theodolite.levelling.implant_flagged") == "nothing in reach of the check"
                  and meta_ab.get("theodolite.levelling.implant_checked_at") == "export",
                  "6: with no trial given the file says none; a GGUF is another object; the implant label says what it looked at")

            # every kind of term against the kernel's own change, and a compiler that writes the wrong thing
            def vc(k: int, w: int) -> np.ndarray:
                return (rng.standard_normal((k, w)) * 0.05).astype(np.float32)

            kinds = {
                "columns scaled and silenced": [we.scale(2, [3, 7], [1.5, -3.0]), we.silence(2, [9])],
                "columns set, added to and scaled": [we.set_columns(2, [4], vc(1, D)), we.add_to_columns(2, [5, 6], vc(2, D)),
                                                     we.scale(2, [8], 0.25)],
                "rows of gate and up": [we.silence_rows(3, "gate_proj", [1, 2]), we.scale_rows(3, "up_proj", [4, 5], [0.5, 2.0])],
                "a low-rank term beside a column": [we.lowrank(4, vc(2, N), vc(2, D).T.copy()), we.silence(4, [1])],
            }
            for label, specs in kinds.items():
                e_k = build(specs)
                worst_k, ok_k = 0.0, True
                for dose_k in (1.0, -0.5):
                    out_k = lex.export_lora_adapter(e_k, target_mod, dose=dose_k,
                                                    out_path=tmp / "owner_exports" / f"k_{e_k.short()}_{dose_k}.gguf")
                    ok_k = ok_k and lex.verify_exported_lora(Path(out_k["gguf_path"]), edit=e_k, target=target_mod, dose=dose_k)["ok"]
                    tens_k, _m = el.read_lora(Path(out_k["gguf_path"]))
                    if not e_k.has_lowrank():
                        for t in e_k.terms:
                            A_k, B_k = tens_k[lex.GGUF_TENSOR_MAP[t.which].format(t.layer)]
                            full = B_k.astype(np.float64) @ A_k.astype(np.float64)
                            mine = full[:, t.idx].T if t.kind == "cols" else full[t.idx, :]
                            worst_k = max(worst_k, float(np.abs(mine - dose_k * we._term_delta_f64(hP.lm.get_submodule(t.linear), t)).max()))
                check(ok_k and worst_k < 1e-6,
                      f"6: {label}: the adapter is the kernel's own change at dose 1 and -0.5 (largest gap {worst_k:.1e})")
            good_compile = lex.build_lora_tensors

            def halved(edit: Any, tgt: Any, *, dose: float = 1.0) -> Dict[str, Any]:
                r = good_compile(edit, tgt, dose=dose)
                r["tensors"] = {k: (A, 0.5 * B) for k, (A, B) in r["tensors"].items()}
                return r

            lex.build_lora_tensors = halved
            try:
                out_bad = lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "halved.gguf")
                ver_bad = lex.verify_exported_lora(Path(out_bad["gguf_path"]), edit=e_trial, target=target_mod)
            finally:
                lex.build_lora_tensors = good_compile
            check(not ver_bad["ok"] and ver_bad["max_abs_err"] > 1e-3,
                  "6: a compiler that writes half the change is seen: the check does not go through the compiler")

            # a file of numbers that are not numbers, and a small change against a file of noughts
            def not_a_number(edit: Any, tgt: Any, *, dose: float = 1.0) -> Dict[str, Any]:
                r = good_compile(edit, tgt, dose=dose)
                r["tensors"] = {k: (A, np.full_like(B, np.nan)) for k, (A, B) in r["tensors"].items()}
                return r

            def noughts(edit: Any, tgt: Any, *, dose: float = 1.0) -> Dict[str, Any]:
                r = good_compile(edit, tgt, dose=dose)
                r["tensors"] = {k: (A, 0.0 * B) for k, (A, B) in r["tensors"].items()}
                return r

            e_small = build([we.scale(2, [3], 1.0 + 2.0 ** -12)])
            small_change = float(np.abs(lex.term_change(e_small.terms[0], target_mod, 1.0)).max())
            vers: Dict[str, Any] = {}
            for tag, wrong, e_w in (("nan", not_a_number, e_trial), ("noughts", noughts, e_small)):
                lex.build_lora_tensors = wrong
                try:
                    out_w = lex.export_lora_adapter(e_w, target_mod, out_path=tmp / "owner_exports" / f"wrong_{tag}.gguf")
                finally:
                    lex.build_lora_tensors = good_compile
                vers[tag] = lex.verify_exported_lora(Path(out_w["gguf_path"]), edit=e_w, target=target_mod)
            out_small = lex.export_lora_adapter(e_small, target_mod, out_path=tmp / "owner_exports" / "small_right.gguf")
            ver_small = lex.verify_exported_lora(Path(out_small["gguf_path"]), edit=e_small, target=target_mod)
            check(not vers["nan"]["ok"] and vers["nan"]["max_abs_err"] is None
                  and next(c for c in vers["nan"]["checks"] if c["name"] == "file_is_the_edit_on_the_target")["detail"]["all_finite"] is False,
                  "6: a file whose numbers are not numbers is not the edit (a largest difference of NaN is not a small one)")
            check(0.0 < small_change < 1e-4 and not vers["noughts"]["ok"] and vers["noughts"]["max_rel_err"] > 0.5
                  and ver_small["ok"] and ver_small["max_rel_err"] < 1e-6,
                  f"6: a change of {small_change:.1e} is not shown by a file of noughts: the tolerance goes with the size "
                  "of the change, and the true file of the same edit passes")
            # a small change beside a large one in the same tensor: each column is set against its own size
            e_two = build([we.silence(2, [3]), we.scale(2, [7], 0.99995)])

            def one_column_dropped(edit: Any, tgt: Any, *, dose: float = 1.0) -> Dict[str, Any]:
                r = good_compile(edit, tgt, dose=dose)
                out_t = {}
                for k, (A, B) in r["tensors"].items():
                    B2 = B.copy()
                    B2[:, int(np.argmax(A[:, 7]))] = 0.0 if A[:, 7].any() else B2[:, 0]
                    out_t[k] = (A, B2)
                r["tensors"] = out_t
                return r

            lex.build_lora_tensors = one_column_dropped
            try:
                out_two = lex.export_lora_adapter(e_two, target_mod, out_path=tmp / "owner_exports" / "one_dropped.gguf")
            finally:
                lex.build_lora_tensors = good_compile
            ver_two = lex.verify_exported_lora(Path(out_two["gguf_path"]), edit=e_two, target=target_mod)
            out_two_ok = lex.export_lora_adapter(e_two, target_mod, out_path=tmp / "owner_exports" / "two_right.gguf")
            check(not ver_two["ok"] and ver_two["max_rel_err"] > 0.5
                  and lex.verify_exported_lora(Path(out_two_ok["gguf_path"]), edit=e_two, target=target_mod)["ok"],
                  "6: a file that leaves out a column's small change beside another column's large one is not the edit: "
                  "every column is set against its own size")
            twice = we.Edit("x" * 64, (e_trial.terms[0], e_trial.terms[0]), dict(e_trial.base), "", {})
            check(refuses(lambda: lex.build_lora_tensors(twice, target_mod), lex.ExportRefused, "named twice")
                  and refuses(lambda: lex.export_lora_adapter(twice, target_mod, out_path=tmp / "owner_exports" / "twice.gguf"),
                              lex.ExportRefused, "hash to another id")
                  and refuses(lambda: lex.export_lora_adapter(e_w1, target_mod, out_path=tmp / "owner_exports" / "other_id.gguf"),
                              lex.ExportRefused, "hash to another id"),
                  "6: an edit that names a column twice is not compiled, and an edit whose id is not its terms' is not exported")

            out_lr = tmp / "owner_exports" / "carried_lr.gguf"
            exp_lr = lex.export_lora_adapter(e_lr, target_gguf, out_path=out_lr)
            ver_lr = lex.verify_exported_lora(out_lr, edit=e_lr, target=target_gguf)
            check(exp_lr["recompiled_from_recipe"] is False and any("re-measure" in w for w in exp_lr["warnings"]) and ver_lr["ok"],
                  "6: carried lowrank export warns to re-measure on served GGUF and passes verify_exported_lora")

            # a trial's verdict goes into the file for its own edit at its own dose, read from its book
            check(refuses(lambda: lex.export_lora_adapter(e_mix, target_mod, out_path=tmp / "owner_exports" / "wrong_edit.gguf",
                                                          trial_result=t_res), lex.ExportRefused, "not of the edit exported")
                  and refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "wrong_dose.gguf",
                                                              dose=2.0, trial_result=t_res), lex.ExportRefused, "dose"),
                  "6: another edit's trial, or this edit's trial at another dose, is refused")
            exp_own = lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "own.gguf", trial_result=t_res,
                                              registration=v_ok)
            _t, m_own = el.read_lora(Path(exp_own["gguf_path"]))
            check(m_own["theodolite.levelling.trial_verdict"] == t_res["verdict"]
                  and m_own["theodolite.levelling.trial_fresh_verdict"] == t_res["fresh_verdict"]
                  and m_own["theodolite.levelling.trial_kind"] == "a dry run"
                  and m_own["theodolite.levelling.trial_brief"] == brief.name
                  and m_own["theodolite.levelling.trial_judged_again"].startswith("yes: the book's readings, judged again")
                  and m_own["theodolite.levelling.trial_applies_to_compiled_base"].startswith("not shown for the whole base")
                  and m_own["theodolite.levelling.compiled_against_is_edit_base"].startswith("its touched linears are, by their pins")
                  and m_own["theodolite.levelling.book"] == f"line {t_res['book']['verdict_seq']} of {cal_book.name} is the trial's verdict; the book verifies"
                  and "earlier trials 0" in m_own["theodolite.levelling.book_before_it"]
                  and m_own["theodolite.levelling.implant_checked_at"] == "the trial's registration"
                  and m_own["theodolite.levelling.serving"].startswith("compiled against a checkpoint llama.cpp does not serve")
                  and any("llama.cpp does not serve" in w for w in exp_own["warnings"])
                  and el.read_lora(out_gguf)[1]["theodolite.levelling.serving"].startswith("compiled against gguf synthetic-gemma4.gguf")
                  and not any("llama.cpp does not serve" in w for w in exp_res["warnings"]),
                  "6: its own trial's verdict goes in from the book, with its kind, its brief, that it was judged again, how "
                  "far the base it was compiled against was compared, where the implant label was worked, and that a "
                  "checkpoint is not what llama.cpp serves")
            exp_gg = lex.export_lora_adapter(e_trial, target_gguf, out_path=tmp / "owner_exports" / "own_gguf.gguf", trial_result=t_res)
            _t, m_gg = el.read_lora(Path(exp_gg["gguf_path"]))
            check(m_gg["theodolite.levelling.trial_applies_to_compiled_base"].startswith("no: the trial read")
                  and m_gg["theodolite.levelling.trial_judged_again"].startswith("no: no registration was given")
                  and m_gg["theodolite.levelling.implant_flagged"] == "not checked",
                  "6: over a GGUF the file says the trial read another base; with no registration the verdict is the book's "
                  "note, said so; with no model at hand the implant label is 'not checked'")
            bad_line = {**t_res, "book": {**t_res["book"], "verdict_hash": "0" * 64}}
            check(refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "bad_book.gguf",
                                                          trial_result=bad_line), lex.ExportRefused, "do not belong together"),
                  "6: a trial result whose book line is not its verdict is refused")
            # the result file is where to look, not what to believe: one changed by hand to say more is not taken at its word
            said_more = {**t_res, "verdict": "passed", "fresh_verdict": "lead", "of_record": True, "not_of_record": []}
            exp_more = lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "said_more.gguf",
                                               trial_result=said_more, registration=v_ok)
            m_more = el.read_lora(Path(exp_more["gguf_path"]))[1]
            check(m_more["theodolite.levelling.trial_verdict"] == t_res["verdict"] == v_note["verdict"]
                  and m_more["theodolite.levelling.trial_kind"] == "a dry run"
                  and (t_res["verdict"] != "passed" or t_res["of_record"] is False),
                  "6: a result file changed by hand to say 'passed' and 'of record' changes nothing the adapter says: the "
                  "verdict and the kind are the book's")
            no_book = {**t_res, "book": {**t_res["book"], "path": str(tmp / "no_such_book.jsonl")}}
            cut_book = tmp / "cut_book.jsonl"
            cut_book.write_bytes(b"".join(cal_book.read_bytes().splitlines(keepends=True)[:t_res["book"]["verdict_seq"]]))
            torn_book = tmp / "torn_book.jsonl"
            torn_book.write_bytes(cal_book.read_bytes().replace(b'"verdict":"', b'"verdict":"x', 1))
            check(refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "nb.gguf",
                                                          trial_result=no_book), lex.ExportRefused, "is not at hand")
                  and refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "cb.gguf",
                                                              trial_result=t_res, book_path=cut_book), lex.ExportRefused, "no ended trial")
                  and refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "tb.gguf",
                                                              trial_result=t_res, book_path=torn_book), lex.ExportRefused, "does not verify")
                  and refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "or.gguf",
                                                              trial_result=t_res, registration=v_car), lex.ExportRefused,
                              "goes with its own registration"),
                  "6: with its book not at hand, cut short of its verdict or not whole, or with another trial's registration, "
                  "no verdict is written")
            exp_dose = lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "other_dose.gguf", dose=0.5,
                                               registration=v_ok)
            check(el.read_lora(Path(exp_dose["gguf_path"]))[1]["theodolite.levelling.implant_flagged"] == "not checked",
                  "6: the implant label a registration recorded is of the registered dose: at another dose the file says 'not checked'")
            far_term = SimpleNamespace(kind="cols", linear="layers.2.mlp.down_proj", op=np.zeros(1, dtype=np.uint8),
                                       factor=np.array([1.0 + we.MAX_REL_CHANGE / 2.0], dtype=np.float32))
            check(lex.dose_caps(e_trial, 1.0) is None and lex.dose_caps(e_trial, -we.MAX_DOSE) is None
                  and lex.dose_caps(e_trial, 1.0, hP.dtype) is None
                  and "within +/-4" in str(lex.dose_caps(e_trial, 50.0)) and "finite number" in str(lex.dose_caps(e_trial, float("nan")))
                  and "finite number" in str(lex.dose_caps(e_trial, True))
                  and "times its own length" in str(lex.dose_caps(SimpleNamespace(terms=(far_term,)), 3.0))
                  and lex.dose_caps(SimpleNamespace(terms=(far_term,)), 1.0) is None
                  and refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "dose_50.gguf",
                                                              dose=50.0), lex.ExportRefused, "not exported at this dose")
                  and not (tmp / "owner_exports" / "dose_50.gguf").exists(),
                  f"6: an adapter is not written for a dose the kernel would not read the edit at ({lex.dose_caps(e_trial, 50.0)})")
            exp_imp = lex.export_lora_adapter(e_imp, target_mod, out_path=tmp / "owner_exports" / "imp.gguf",
                                              lm_for_implant=hP.lm, W_U_for_implant=hP.W_U)
            check(el.read_lora(Path(exp_imp["gguf_path"]))[1]["theodolite.levelling.implant_flagged"] == "true",
                  "6: an edit of the implant's shape is labelled so in the file")

            # off the model: an edit is rebuilt as the kernel hashed it, or not at all
            check(lex.edit_from_registration(v_ok["edit_manifest"]).id == e_trial.id
                  and lex.edit_from_registration(prov_reb).id == e_reb.id
                  and lex.edit_from_saved(saved_car).id == e_car.id
                  and lex.edit_from_registration(v_car["edit_manifest"], payload_dir=saved_car).id == e_car.id,
                  "6: a registered recipe, a saved edit and a registered carried edit are rebuilt off the model to their ids")
            tampered_prov = json.loads(json.dumps(prov_reb))
            tampered_prov["rebuilt_recipe"]["terms"][0]["silence"] = [3, 8]
            saved_twin = tmp / "saved_twin"
            shutil.copytree(saved_car, saved_twin)
            from safetensors.numpy import load_file, save_file

            pl = load_file(str(saved_twin / "payload.safetensors"))
            save_file({k: v * 10 for k, v in pl.items()}, str(saved_twin / "payload.safetensors"))
            check(refuses(lambda: lex.edit_from_registration(tampered_prov), lex.ExportRefused, "does not rebuild")
                  and refuses(lambda: lex.edit_from_saved(saved_twin), lex.ExportRefused, "does not rebuild")
                  and refuses(lambda: lex.edit_from_registration(v_car["edit_manifest"], payload_dir=saved_twin), lex.ExportRefused),
                  "6: a recipe or a payload changed since the kernel hashed it is refused off the model")
            # off the model an id shows the terms are its own, and no more: the caps that need no weights, and a
            # reading of the edit in the book, are asked as well
            off_trial, off_never = lex.edit_from_registration(v_ok["edit_manifest"]), lex.edit_from_registration(prov_reb)
            ckpt_target = lex.TargetBase(weights_dir=wr.export_stand_in_safetensors(hP.lm, tmp / "ckpt_off", model_id=hP.model_id))
            exp_off = lex.export_lora_adapter(off_trial, ckpt_target, out_path=tmp / "owner_exports" / "off_ok.gguf", book_path=cal_book)
            gate_said = el.read_lora(Path(exp_off["gguf_path"]))[1]["theodolite.levelling.gate"]
            seen_at = lex.gate_evidence(e_trial.id, cal_book)
            check(off_trial.report.get("made") == lex.OFF_THE_MODEL and seen_at is not None
                  and gate_said.startswith("rebuilt off the model") and f"line {seen_at['seq']} of {cal_book.name}" in gate_said
                  and lex.gate_evidence(e_reb.id, cal_book) is None
                  and refuses(lambda: lex.export_lora_adapter(off_never, ckpt_target, out_path=tmp / "owner_exports" / "off_never.gguf",
                                                              book_path=cal_book), lex.ExportRefused, "holds no reading of it")
                  and refuses(lambda: lex.export_lora_adapter(off_trial, ckpt_target, out_path=tmp / "owner_exports" / "off_nobook.gguf"),
                              lex.ExportRefused, "holds no reading of it"),
                  "6: an edit rebuilt off the model is exported only when a level book holds a reading of it, and the file says "
                  f"which line ({gate_said[:90]})")
            big_f = we.Term(e_trial.terms[0].linear, "cols", 2, "down_proj", idx=we._ro([3], np.int64), op=we._ro([0], np.uint8),
                            factor=we._ro([1000.0], np.float32))
            many = [we.Term(we.linear_name(L, w), "rows", L, w, idx=we._ro([1], np.int64), op=we._ro([0], np.uint8),
                            factor=we._ro([0.5], np.float32)) for L in range(6) for w in ("gate_proj", "up_proj")] \
                + [we.Term(we.linear_name(L, "down_proj"), "cols", L, "down_proj", idx=we._ro([1], np.int64),
                           op=we._ro([0], np.uint8), factor=we._ro([0.5], np.float32)) for L in range(5)]

            def by_hand(terms: Any) -> we.Edit:
                ordered = sorted(terms, key=lambda t: (t.linear, t.kind))
                return lex._kept_to_id(list(terms), dict(e_trial.base), we._edit_id(dict(e_trial.base), ordered), "an edit typed by hand")

            by_hand_edit = by_hand

            far_f = we.Term(e_trial.terms[0].linear, "cols", 2, "down_proj", idx=we._ro([3], np.int64), op=we._ro([0], np.uint8),
                            factor=we._ro([-64.0], np.float32))
            low40 = [we.Term(we.linear_name(4, "down_proj"), "lowrank", 4, "down_proj",
                             A=we._ro(np.zeros((40, N), np.float32), np.float32), B=we._ro(np.zeros((D, 40), np.float32), np.float32))
                     for _ in range(2)]
            typed_book = tmp / "typed_gate_book.jsonl"
            typed_book.write_text(json.dumps({"v": 1, "seq": 0, "at": "2026-11-20T00:00:00Z", "kind": "paired", "prev": "not a hash",
                                              "body": {"status": "ok", "name": "typed", "edit": {"id": e_reb.id}},
                                              "hash": "not a hash"}) + "\n")
            check("65 times its own length" in str(we.static_caps(by_hand_edit([far_f])))
                  and "rank at most 64" in str(we.static_caps(by_hand_edit(low40)))
                  and lb.verify(typed_book)["ok"] is False
                  and refuses(lambda: lex.gate_evidence(e_reb.id, typed_book), lex.ExportRefused, "does not verify")
                  and refuses(lambda: lex.export_lora_adapter(off_never, ckpt_target, book_path=typed_book,
                                                              out_path=tmp / "owner_exports" / "typed_gate.gguf"),
                              lex.ExportRefused, "does not verify"),
                  "6: off the model a scaling the gate would refuse for its size, and low-rank terms that join past the "
                  "cap on rank, are refused; and a reading is not taken as the kernel's from a book that does not verify")
            check(we.static_caps(e_trial) is None and we.static_caps(e_car) is None and we.static_caps(e_lr) is None
                  and "at most 64" in str(we.static_caps(by_hand([big_f])))
                  and len(many) == 17 and "at most 16 terms" in str(we.static_caps(by_hand(many)))
                  and refuses(lambda: lex.export_lora_adapter(by_hand([big_f]), ckpt_target, book_path=cal_book,
                                                              out_path=tmp / "owner_exports" / "big_f.gguf"),
                              lex.ExportRefused, "would not pass the kernel's gate"),
                  "6: an edit typed by hand with an id that is its terms' own is still asked the kernel's caps that need no "
                  f"weights ({we.static_caps(by_hand([big_f]))})")

            ckpt = wr.export_stand_in_safetensors(hP.lm, tmp / "ckpt", model_id=hP.model_id)
            cli_gguf = tmp / "owner_exports" / "cli_export.gguf"
            rc_lora = lex_cli.main(["lora", "--trial", str(prereg_dir), "--weights-dir", str(ckpt), "-o", str(cli_gguf),
                                    "--trial-result", str(trial_out / "trial_result.json")])
            m_cli = el.read_lora(cli_gguf)[1] if cli_gguf.is_file() else {}
            check(rc_lora == 0 and m_cli.get("theodolite.levelling.edit_id") == e_trial.id
                  and str(m_cli.get("theodolite.levelling.compiled_against_is_edit_base")).startswith("its touched linears are, by their pins")
                  and m_cli.get("theodolite.levelling.implant_checked_at") == "the trial's registration"
                  and m_cli.get("theodolite.levelling.trial_kind") == "a dry run"
                  and str(m_cli.get("theodolite.levelling.trial_judged_again")).startswith("yes")
                  and str(m_cli.get("theodolite.levelling.gate")).startswith("rebuilt off the model"),
                  "6: the command line exports a registration off the model with its trial's verdict read from the book the "
                  "result names, and says how far the checkpoint was shown to be the edit's base")
            check(lex_cli.main(["lora", "--saved", str(saved_car), "--weights-dir", str(ckpt),
                                "-o", str(tmp / "owner_exports" / "cli_never_read.gguf")]) == 2
                  and not (tmp / "owner_exports" / "cli_never_read.gguf").exists(),
                  "6: the command line does not export off the model a saved edit that no book shows the kernel has read")
            check(lex_cli.main(["verify", str(cli_gguf), "--trial", str(prereg_dir), "--weights-dir", str(ckpt)]) == 0
                  and lex_cli.main(["verify", str(cli_gguf), "--trial", str(prereg_dir), "--gguf", str(synth_gguf)]) == 1
                  and lex_cli.main(["show", str(cli_gguf)]) == 0,
                  "6: the command line's verify passes on the base the file was compiled against, and fails on another")
            (tmp / "twice.json").write_text(json.dumps({"terms": [{"layer": 2, "silence": [3, 3]}]}))
            check(lex_cli.main(["lora", "--edit", str(tmp / "twice.json"), "--weights-dir", str(ckpt),
                                "-o", str(tmp / "owner_exports" / "bare.gguf")]) == 2
                  and not (tmp / "owner_exports" / "bare.gguf").exists(),
                  "6: a bare recipe is not exported off the model: nothing there would gate it")
            from_default_was = wr.WeightReader.from_default
            try:
                wr.WeightReader.from_default = classmethod(lambda cls: wr.WeightReader(ckpt))
                default_found = lex.TargetBase.default_checkpoint()

                def not_cached(cls: Any) -> Any:
                    raise wr.WeightReadError("no such checkpoint in the cache")

                wr.WeightReader.from_default = classmethod(not_cached)
                default_missing = refuses(lambda: lex.TargetBase.default_checkpoint(), lex.ExportRefused, "--gguf FILE or --weights-dir")
            finally:
                wr.WeightReader.from_default = from_default_was
            check(default_found.kind == "checkpoint" and default_found.is_base_of(e_trial) is True and default_missing,
                  "6: with no base named, the cached checkpoint is the base when it is there, and its absence is refused in "
                  "words that say what to give")

            mcp_tool_names = list(lmcp.TOOL_SPECS.keys())
            route_paths = [getattr(r, "path", "") for r in lroutes.router.routes]
            check(all("export" not in n and "trial" not in n for n in mcp_tool_names)
                  and all("export" not in p and "trial" not in p for p in route_paths),
                  f"6: zero export/trial tools exposed on MCP ({mcp_tool_names}) or HTTP ({route_paths})")
            surfaces = ["backend/level_mcp.py", "backend/level_session.py", "backend/level_routes.py",
                        "backend/scientist_tools.py", "backend/scientist_mcp.py", "backend/server.py"]
            check(all(not re.search(r"level_(export|trial|rerun)", (ROOT_DIR / s).read_text(encoding="utf-8"))
                      for s in surfaces if (ROOT_DIR / s).is_file()),
                  "6: no agent or page surface names the trial, the export or the re-run")

            # ------------------------------------------------------------------ 7. a volunteer's re-run and its intake
            section("7. The re-run: the kit, the home items, and an intake that judges the returned book")
            check(refuses(lambda: lr.make_rerun_kit(prereg_car, tmp / "kit_carried_fail", t_res), lr.RerunRefused, "rebuildable"),
                  "7: make_rerun_kit refuses carried-payload edits (rebuildable recipes only)")
            check(refuses(lambda: lr.make_rerun_kit(prereg_dir, tmp / "kit_no_result", {}), lr.RerunRefused, "not this registration")
                  and refuses(lambda: lr.make_rerun_kit(prereg_dir, tmp / "kit_void", {**t_res, "verdict": "void"}), lr.RerunRefused, "void")
                  and refuses(lambda: lr.make_rerun_kit(again, tmp / "kit_other", t_res), lr.RerunRefused, "not this registration"),
                  "7: a kit is made from a registration and the result of its own home trial, which ended")
            kit_dir = tmp / "rerun_kit"
            kit_info = lr.make_rerun_kit(prereg_dir, kit_dir, t_res)
            kv = lr.verify_rerun_kit(kit_dir)
            check(kv["ok"] and kv["trial_sha256"] == reg["trial_sha256"] and kv["kit_sha256"] == kit_info["kit_sha256"]
                  and kv["kit"]["seed_hex"] == t_res["seed_hex"]
                  and sorted(p.name for p in kit_dir.iterdir()) == sorted(lt.REG_FILES + lt.KIT_EXTRA_FILES + ("SHA256SUMS",)),
                  "7: the kit is the registration's files, the home sums, and the seed of the home draw")
            kit_twin, kit_rules = tmp / "kit_twin", tmp / "kit_rules"
            shutil.copytree(kit_dir, kit_twin)
            shutil.copytree(kit_dir, kit_rules)
            # a setting the files do not say of one another: only the home registration's own sums can tell
            (kit_twin / "trial.json").write_text((kit_twin / "trial.json").read_text().replace('"n_neutral": 2', '"n_neutral": 3'))
            lt._write_sha256sums(kit_twin)
            (kit_rules / "RULES.md").write_text((kit_rules / "RULES.md").read_text() + "\nedited\n")
            lt._write_sha256sums(kit_rules)
            check('"n_neutral": 3' in (kit_twin / "trial.json").read_text()
                  and refuses(lambda: lr.verify_rerun_kit(kit_twin), lr.RerunRefused, "home registration")
                  and refuses(lambda: lr.verify_rerun_kit(kit_rules), lr.RerunRefused, "RULES.md is not the rules"),
                  "7: a kit whose registration files are not the home registration's is refused, its own sums rewritten or not")

            vol_tar = tmp / "vol_alice.tar"
            vol_res = lr.run_volunteer_rerun(hP, kit_dir, vol_tar, volunteer_name="alice", work_dir=tmp / "vol_work")
            with tarfile.open(vol_tar, "r:") as tf:
                members = sorted(m.name for m in tf.getmembers())
            vol_book = tmp / "vol_work" / "volunteer_book.jsonl"
            vol_notes = [e["body"] for e in lb.entries(vol_book) if e["kind"] == "note"]
            check(members == sorted(lr.PACKED_FILES) and vol_res["verdict"] == t_res["verdict"]
                  and vol_notes[0]["rerun_of"] == reg["trial_sha256"] and vol_notes[-1]["note_type"] == lt.NOTE_RERUN
                  and vol_notes[0]["of_record"] is False
                  and sorted(d["name"] for d in vol_notes[-1]["drawn"]["target"]) == sorted(d["name"] for d in v_note["drawn"]["target"])
                  and sorted(d["item_id"] for d in vol_notes[-1]["drawn"]["target"]) == sorted(d["item_id"] for d in v_note["drawn"]["target"]),
                  "7: the volunteer's run reads the home trial's own items, into a book of its own, as a re-run and never of record")
            vol_sheet = (tmp / "vol_work" / "trial_out" / "TRIAL_SHEET.md").read_text()
            SHEETS.append(vol_sheet)
            check(vol_sheet.startswith("=== RE-RUN OF A TRIAL (a volunteer's; not of record)"),
                  "7: the re-run's sheet says it is a re-run")

            # the checkout this check runs in is whatever it is (changed while the code is worked on, clean once it is
            # committed), so the bundle as it came back is taken with both leaves, and what each leave is for is shown
            # below on books written again with a checkout of the check's choosing
            stage_ok = tmp / "intake_ok"
            iv_ok = lr.intake_rerun_bundle(vol_tar, prereg_dir, cal_book, stage_ok, allow_dirty=True, allow_other_runner=True)
            check(iv_ok["outcome"] == "accepted" and iv_ok["match_class"] == "same_figures"
                  and (stage_ok / "intake_verdict.json").is_file()
                  and iv_ok["class_diffs"]["target:small"]["items"] == 4 and iv_ok["class_diffs"]["reworded:large"]["items"] == 8
                  and iv_ok["class_diffs"]["target:small:controls"]["items"] == 78
                  and iv_ok["class_diffs"]["reworded:large:controls"]["items"] == 19
                  and iv_ok["class_diffs"]["target:large:folded"]["items"] == 4 and iv_ok["class_diffs"]["batteries"]["items"] >= 4
                  and all(v["largest_difference"] == 0.0 and v["missing"] == 0 for v in iv_ok["class_diffs"].values())
                  and iv_ok["figures_compared"] + iv_ok["figures_neither_trial_has"] == len(rj["figures"])
                  and iv_ok["figures_compared"] == sum(1 for v in rj["figures"].values() if v is not None)
                  and all(rj["figures"][k] is not None for k in rj["figures"] if ":control:" in k)
                  and iv_ok["control_edits_set_aside_differently"] == {}
                  and iv_ok["judged_from"].startswith("the returned book") and len(iv_ok["on_the_volunteers_word"]) == 3
                  and any("copied into a new book" in w for w in iv_ok["on_the_volunteers_word"]),
                  f"7: the same card re-running its own kit is accepted with the same figures, every one of "
                  f"{iv_ok.get('figures_compared')} ({iv_ok['outcome']}, {iv_ok['match_class']}; {iv_ok['reject_reasons'][:1]})")

            ret_ok = {"v": 1, "volunteer": "nobody", "trial_sha256": reg["trial_sha256"], "verdict": t_res["verdict"],
                      "fresh_verdict": t_res["fresh_verdict"]}

            def intake_of(tag: str, folder: Path, *, dirty: bool = True, other_runner: bool = True) -> Dict[str, Any]:
                _pack(folder, tmp / f"{tag}.tar")
                return lr.intake_rerun_bundle(tmp / f"{tag}.tar", prereg_dir, cal_book, tmp / f"stage_{tag}",
                                              allow_dirty=dirty, allow_other_runner=other_runner)

            def written_again(tag: str, change: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> Path:
                """The volunteer's book written again line by line, `change(entry, body)` altering what it will, as a
                bundle whose summary states the home verdict. A book is a chain with no secret in it: anyone can."""
                dst = tmp / tag / "volunteer_book.jsonl"
                dst.parent.mkdir()
                for ent in lb.entries(vol_book):
                    body = json.loads(json.dumps(ent["body"]))
                    change(ent, body)
                    lb.append(ent["kind"], body, path=dst)
                _bundle(dst.parent, dst, ret_ok)
                return dst.parent

            def checkout_says(**over: Any) -> Callable[[Dict[str, Any], Dict[str, Any]], None]:
                def change(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                    if body.get("note_type") == "trial_registration":
                        body["checkout"] = {**body["checkout"], "dirty": False, "changed": 0, "tracked_changed": 0, "hidden": 0,
                                            "unlisted_code": 0, "sources_match": True, "sources_moved": [], **over}
                return change

            def failed(iv: Dict[str, Any], name: str) -> bool:
                return any(c["name"] == name and not c["ok"] for c in iv["checks"])

            # what each leave is for: a tracked file changed, and a runner or reader that is not the registered one
            co_clean = written_again("co_clean", checkout_says())
            co_untracked = written_again("co_untracked", checkout_says(dirty=True, changed=3))
            co_dirty = written_again("co_dirty", checkout_says(dirty=True, changed=2, tracked_changed=2))
            co_hidden = written_again("co_hidden", checkout_says(hidden=1))
            co_runner = written_again("co_runner", checkout_says(sources_match=False, sources_moved=["backend/levelling.py"]))
            iv_clean = intake_of("co_clean", co_clean, dirty=False, other_runner=False)
            iv_untracked = intake_of("co_untracked", co_untracked, dirty=False, other_runner=False)
            check(iv_clean["outcome"] == "accepted" and iv_clean["with_leave_for"] == []
                  and iv_untracked["outcome"] == "accepted" and iv_untracked["with_leave_for"] == [],
                  f"7: a re-run from a checkout with no tracked file changed is accepted with no leave given, whatever files "
                  f"git does not know lie in it ({iv_clean['reject_reasons'][:1]}; {iv_untracked['reject_reasons'][:1]})")
            iv_dirty = intake_of("co_dirty", co_dirty, dirty=False, other_runner=True)
            iv_dirty_ok = intake_of("co_dirty_leave", co_dirty, dirty=True, other_runner=False)
            iv_hidden = intake_of("co_hidden", co_hidden, dirty=False, other_runner=True)
            check(iv_dirty["outcome"] == "rejected" and failed(iv_dirty, "no_tracked_file_changed")
                  and not failed(iv_dirty, "runner_and_reader_as_registered")
                  and iv_hidden["outcome"] == "rejected" and failed(iv_hidden, "no_tracked_file_changed")
                  and iv_dirty_ok["outcome"] == "accepted" and len(iv_dirty_ok["with_leave_for"]) == 1
                  and "allow_dirty" in iv_dirty_ok["with_leave_for"][0],
                  "7: a re-run whose checkout had a tracked file changed, or a file git was told to overlook, is rejected "
                  "unless that leave is given, and the verdict then says it was")
            # code no commit holds where the volunteer's runner imports from: its own check, its own count shown, and
            # a count that could not be asked is not one that was found to be nought
            co_code = written_again("co_code", checkout_says(unlisted_code=1))
            co_code_none = written_again("co_code_none", checkout_says(unlisted_code=None))
            iv_code = intake_of("co_code", co_code, dirty=False, other_runner=False)
            iv_code_none = intake_of("co_code_none", co_code_none, dirty=False, other_runner=False)
            iv_code_ok = intake_of("co_code_leave", co_code, dirty=True, other_runner=False)
            code_said = next(c for c in iv_code["checks"] if c["name"] == "no_code_outside_the_commit")
            check(iv_code["outcome"] == "rejected" and failed(iv_code, "no_code_outside_the_commit")
                  and not failed(iv_code, "no_tracked_file_changed") and code_said["detail"]["code_no_commit_holds"] == 1
                  and iv_code_none["outcome"] == "rejected" and failed(iv_code_none, "no_code_outside_the_commit")
                  and iv_code_ok["outcome"] == "accepted" and len(iv_code_ok["with_leave_for"]) == 1
                  and "no commit holds" in iv_code_ok["with_leave_for"][0],
                  f"7: a re-run whose checkout had code no commit holds where its runner imports from, or could not say, is "
                  f"rejected by a check that shows that count ({code_said['detail']})")
            iv_runner = intake_of("co_runner", co_runner, dirty=True, other_runner=False)
            iv_runner_ok = intake_of("co_runner_leave", co_runner, dirty=False, other_runner=True)
            check(iv_runner["outcome"] == "rejected" and failed(iv_runner, "runner_and_reader_as_registered")
                  and not failed(iv_runner, "no_tracked_file_changed")
                  and iv_runner_ok["outcome"] == "accepted" and "allow_other_runner" in iv_runner_ok["with_leave_for"][0],
                  "7: a re-run by another runner or reader is rejected unless that leave is given: the leave for a changed "
                  "checkout does not cover it")

            typed = tmp / "typed"
            typed.mkdir()
            lb.append("note", {"note_type": "hello", "text": "no reading was taken"}, path=typed / "volunteer_book.jsonl")
            _bundle(typed, typed / "volunteer_book.jsonl", ret_ok)
            iv_typed = intake_of("typed", typed)
            check(iv_typed["outcome"] == "rejected" and any("re-run of" in r for r in iv_typed["reject_reasons"]),
                  "7: a bundle typed by hand, with a book that holds no re-run, is rejected whatever its summary says")
            handed = tmp / "handed_back"
            _bundle(handed, cal_book, ret_ok)
            check(intake_of("handed_back", handed)["outcome"] == "rejected",
                  "7: the home book handed back as a volunteer's is rejected: it holds no re-run")
            other_word = tmp / "other_word"
            _bundle(other_word, vol_book, {**ret_ok, "verdict": "passed" if t_res["verdict"] == "failed" else "failed"})
            iv_word = intake_of("other_word", other_word)
            check(iv_word["outcome"] == "rejected" and any("bundle_states_what_its_book_gives" in r for r in iv_word["reject_reasons"]),
                  "7: a summary that states another verdict than its book gives is rejected")

            # another card's figures: the volunteer's book written again with every fresh effect moved by half a unit
            moved_book = tmp / "moved" / "volunteer_book.jsonl"
            moved_book.parent.mkdir()
            for ent in lb.entries(vol_book):
                body = json.loads(json.dumps(ent["body"]))
                if ent["kind"] == "paired" and str(body.get("name", "")).endswith("_matched"):
                    for it in body["items"]:
                        for cls in ("small", "large"):
                            for arm in it["classes"][cls]["arms"]:
                                if str(arm.get("key", "")).startswith("edit:1.0"):
                                    arm["metrics"]["odds"]["mean"] += 0.5
                lb.append(ent["kind"], body, path=moved_book)
            _bundle(moved_book.parent, moved_book, ret_ok)
            iv_moved = intake_of("moved", moved_book.parent)
            check(iv_moved["outcome"] == "disagreed" and any("items_agree_target" in r for r in iv_moved["disagree_reasons"])
                  and iv_moved["class_diffs"]["target:small"]["outside_tolerance"] == 4,
                  f"7: figures that differ from the home's by more than the tolerance are a disagreement ({iv_moved['outcome']}: "
                  f"{(iv_moved['disagree_reasons'] or iv_moved['reject_reasons'])[:1]})")

            # more than the items: the control edits' means, the plan, which item is which frame and half, the base
            def named(end: str, ent: Dict[str, Any], body: Dict[str, Any]) -> bool:
                return ent["kind"] == "paired" and str(body.get("name", "")).endswith(end)

            def controls_moved(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if named("_matched", ent, body):
                    for it in body["items"]:
                        for cls in ("small", "large"):
                            for arm in it["classes"][cls]["arms"]:
                                if str(arm.get("key", "")).startswith("control:"):
                                    arm["metrics"]["odds"]["mean"] += 0.5

            def other_plan(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if named("_same_site", ent, body):
                    body["plan"]["null_e"] = [0.25, -0.25]

            def halves_swapped(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if body.get("note_type") == lt.NOTE_RERUN:
                    for row in body["drawn"]["target"]:
                        row["half"] = 1 - row["half"]

            def other_base(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if body.get("note_type") == "trial_registration":
                    body["base_checksum_before"] = "0" * 64

            iv_ctl = intake_of("controls_moved", written_again("controls_moved", controls_moved))
            # every matched control edit's mean is set beside the home's as read, whether or not the judgement then
            # set it aside (39 of the 78 in this row are the same-site ones, which were not moved)
            check(iv_ctl["outcome"] != "accepted"
                  and iv_ctl["class_diffs"]["target:small:controls"]["outside_tolerance"] == 39
                  and iv_ctl["class_diffs"]["target:small:controls"]["missing"] == 0
                  and iv_ctl["class_diffs"]["target:small"]["outside_tolerance"] == 0
                  and failed(iv_ctl, "figures_agree_target_small_controls"),
                  f"7: control edits that read otherwise than at home are seen, with every item's own effect the same "
                  f"({iv_ctl['outcome']}: {(iv_ctl['disagree_reasons'] or iv_ctl['reject_reasons'])[:1]})")
            def other_controls(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if named("_matched", ent, body):
                    body["controls"]["ids"][0] = "0" * 64

            iv_ids = intake_of("other_controls", written_again("other_controls", other_controls))
            check(iv_ids["outcome"] == "disagreed" and failed(iv_ids, "control_edits_of_matched_drawn_as_at_home")
                  and not failed(iv_ids, "reading_matched_is_the_home_reading"),
                  f"7: a card that drew other control edits from the same seed is a disagreement the intake names "
                  f"({iv_ids['outcome']}: {iv_ids['disagree_reasons'][:1]})")
            # a control edit at the bound of the matching tolerance: set aside on one card and not on the other. Its
            # mean as read is the home's, so no figure is "missing"; the intake says which trials set aside what.
            aside_home = [rj["judged"]["inputs"]["classes"][c]["controls_aside"]["other_sites_matched"] for c in ("small", "large")]
            c_in = next((c for c in range(39) if not aside_home[0][c] and not aside_home[1][c]), None)

            def one_control_quiet(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if named("_matched", ent, body) and c_in is not None:
                    for it in body["items"]:
                        for cls in ("small", "large"):
                            for arm in it["classes"][cls]["arms"]:
                                if arm.get("key") == f"control:{c_in}":
                                    arm["delivered"], arm["delivered_end"] = 1e-9, 1e-9

            iv_bound = intake_of("ctl_at_bound", written_again("ctl_at_bound", one_control_quiet))
            check(c_in is not None and iv_bound["class_diffs"]["target:small:controls"]["missing"] == 0
                  and iv_bound["class_diffs"]["target:small:controls"]["largest_difference"] == 0.0
                  and iv_bound["class_diffs"]["target:small:controls"]["items"] == 78
                  and iv_bound["control_edits_set_aside_differently"] == {"target:small:other_sites_matched": 1,
                                                                           "target:large:other_sites_matched": 1}
                  and iv_bound["match_class"] != "same_figures"
                  and (iv_bound["outcome"] == "accepted" or any("verdict_matches_home" in r or "book_note_follows" in r
                                                                 for r in iv_bound["disagree_reasons"] + iv_bound["reject_reasons"])),
                  f"7: a control edit set aside on the volunteer's card and not at home is no figure gone missing: its mean "
                  f"as read is compared, and the intake says the two trials set aside other control edits "
                  f"({iv_bound['outcome']}, {iv_bound['control_edits_set_aside_differently']})")
            # a value nested deeper than Python will walk, and a number that is no JSON number: a verdict in words
            deep = tmp / "deep_return"
            _bundle(deep, vol_book, ret_ok)
            (deep / "return.json").write_text(json.dumps(ret_ok)[:-1] + ', "says": ' + "[" * 3000 + "]" * 3000 + "}", encoding="utf-8")
            deep_v = tmp / "deep_verdict"
            _bundle(deep_v, vol_book, ret_ok)
            (deep_v / "return.json").write_text(json.dumps({**ret_ok, "verdict": "@@V@@"}).replace('"@@V@@"', "[" * 3000 + "]" * 3000),
                                                encoding="utf-8")
            nan_v = tmp / "nan_verdict"
            _bundle(nan_v, vol_book, ret_ok)
            (nan_v / "return.json").write_text(json.dumps({**ret_ok, "verdict": "@@V@@"}).replace('"@@V@@"', "NaN"), encoding="utf-8")
            for folder_x in (deep, deep_v, nan_v):
                man_x = json.loads((folder_x / "manifest.json").read_text())
                man_x["files"]["return.json"]["sha256"] = hashlib.sha256((folder_x / "return.json").read_bytes()).hexdigest()
                (folder_x / "manifest.json").write_text(json.dumps(man_x), encoding="utf-8")
            try:
                iv_deep, iv_deep_v, iv_nan = intake_of("deep_return", deep), intake_of("deep_verdict", deep_v), intake_of("nan_verdict", nan_v)

                def strict(tag: str) -> Any:
                    def no_constant(name: str) -> Any:
                        raise ValueError(f"{name} is no JSON number")
                    return json.loads((tmp / f"stage_{tag}" / "intake_verdict.json").read_text(), parse_constant=no_constant)

                check(iv_deep["outcome"] == "accepted" and iv_deep_v["outcome"] == "rejected" and iv_nan["outcome"] == "rejected"
                      and any("bundle_states_what_its_book_gives" in r for r in iv_deep_v["reject_reasons"] + iv_nan["reject_reasons"])
                      and strict("deep_return")["outcome"] == "accepted" and strict("deep_verdict")["outcome"] == "rejected"
                      and strict("nan_verdict")["outcome"] == "rejected"
                      and "nested too deep to write" in json.dumps(strict("deep_verdict"))
                      and "not a finite number" in json.dumps(strict("nan_verdict"))
                      and lr._writable(float("inf")) == "inf (not a finite number)" and lr._writable(1.5) == 1.5
                      and lr._s([["x"]], 8) == "(list)" and lr._s("abcdefghij", 4) == "abcd" and lr._s(None, 4) == "",
                      "7: a return that nests a value three thousand deep, or states a verdict that is no JSON number, is "
                      "judged and its verdict written as a file any JSON reader takes")
            except Exception as exc:  # noqa: BLE001
                check(False, f"7: a return nested three thousand deep, or holding NaN, is judged, not an error ({type(exc).__name__}: {exc})")
            walk_was = lr._intake
            lr._intake = lambda *a, **k: (_ for _ in ()).throw(RecursionError("maximum recursion depth exceeded"))
            try:
                iv_rec = lr.intake_rerun_bundle(vol_tar, prereg_dir, cal_book, tmp / "stage_recursion")
            finally:
                lr._intake = walk_was
            check(iv_rec["outcome"] == "rejected" and "nested deeper than can be read" in iv_rec["reject_reasons"][0]
                  and json.loads((tmp / "stage_recursion" / "intake_verdict.json").read_text())["outcome"] == "rejected",
                  "7: and where a value is nested past what Python itself will walk, the intake ends in a rejection that "
                  "is written, not in an error")
            odd_name = tmp / "odd_name"
            _bundle(odd_name, vol_book, {**ret_ok, "volunteer": "Ren\udce9", "says": "\ud800"})
            try:
                iv_name = intake_of("odd_name", odd_name)
                check(iv_name["outcome"] == "accepted" and (tmp / "stage_odd_name" / "intake_verdict.json").is_file()
                      and "\\udce9" in iv_name["volunteer"],
                      "7: a return whose words a file cannot hold (half of a surrogate pair) is judged and written all the same")
            except Exception as exc:  # noqa: BLE001
                check(False, f"7: a return with half a surrogate pair in it is judged, not an error ({type(exc).__name__}: {exc})")
            torn_home = tmp / "torn_home_book.jsonl"
            torn_home.write_bytes(cal_book.read_bytes().replace(b'"note_type":"brief_calibration"', b'"note_type":"brief_calibratioN"', 1))
            check(lb.verify(torn_home)["ok"] is False
                  and refuses(lambda: lr.intake_rerun_bundle(vol_tar, prereg_dir, torn_home, tmp / "stage_torn_home"),
                              lr.RerunRefused, "does not verify"),
                  "7: an intake against a home book that does not verify does not begin")
            iv_plan = intake_of("other_plan", written_again("other_plan", other_plan))
            iv_half = intake_of("halves_swapped", written_again("halves_swapped", halves_swapped))
            iv_base = intake_of("other_base", written_again("other_base", other_base))
            check(iv_plan["outcome"] == "rejected" and failed(iv_plan, "reading_same_site_is_the_home_reading")
                  and iv_half["outcome"] == "rejected" and failed(iv_half, "drawn_items_recorded_as_at_home")
                  and iv_base["outcome"] == "rejected" and failed(iv_base, "base_checksum_is_the_home_base"),
                  "7: a reading taken by another plan, a note that calls other items the two halves, and a base with "
                  "another checksum are each not the home trial's re-run")

            # a book typed by hand ends an intake in words, whatever it holds
            def a_billion_controls(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if named("_matched", ent, body):
                    body["controls"]["count"] = 10 ** 9

            def odd_shapes(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if named("_reworded", ent, body):
                    body["summary"], body["items"][0]["classes"] = [1, 2, 3], [7]
                if named("_panel", ent, body):
                    body["base"] = "a string"

            def items_a_string(ent: Dict[str, Any], body: Dict[str, Any]) -> None:
                if named("_neutral", ent, body):
                    body["items"] = "all of them"

            for label, change in (("a billion control edits", a_billion_controls), ("values of other kinds", odd_shapes),
                                  ("items that are a string", items_a_string)):
                tag = "typed_" + re.sub(r"\W+", "_", label)
                try:
                    iv_t = intake_of(tag, written_again(tag, change))
                    check(iv_t["outcome"] == "rejected" and failed(iv_t, "rerun_can_be_judged_from_its_book"),
                          f"7: a returned book that holds {label} is rejected ({iv_t['reject_reasons'][:1]})")
                except Exception as exc:  # noqa: BLE001
                    check(False, f"7: a returned book that holds {label} is rejected, not an error ({type(exc).__name__}: {exc})")
            past_reach = tmp / "past_reach"
            past_reach.mkdir()
            raw_book = vol_book.read_bytes()
            (past_reach / "volunteer_book.jsonl").write_bytes(raw_book.replace(b'"seconds":', b'"a_number":1e999,"seconds":', 1))
            _bundle(past_reach, past_reach / "volunteer_book.jsonl", ret_ok)
            v_past = lb.verify(past_reach / "volunteer_book.jsonl")
            try:
                iv_past = intake_of("past_reach", past_reach)
                check(v_past["ok"] is False and "not finite" in str(v_past["why"]) and iv_past["outcome"] == "rejected"
                      and failed(iv_past, "returned_book_whole"),
                      f"7: a book that holds a number past a float's reach does not verify, and is rejected ({v_past['why']})")
            except Exception as exc:  # noqa: BLE001
                check(False, f"7: a number past a float's reach is a rejection, not an error ({type(exc).__name__}: {exc})")
            (tmp / "lines.txt").write_bytes(b"abc\n" + b"x" * 10 + b"\nfive5")
            long_line = tmp / "long_line"
            _bundle(long_line, vol_book, ret_ok)
            cap_was = lr.MAX_LINE_BYTES
            lr.MAX_LINE_BYTES = 2000
            try:
                iv_long = intake_of("long_line", long_line)
            finally:
                lr.MAX_LINE_BYTES = cap_was
            check(lr._longest_line(tmp / "lines.txt") == 10 and lr._longest_line(vol_book) > 2000
                  and iv_long["outcome"] == "rejected" and failed(iv_long, "returned_book_lines_under_the_cap")
                  and not any(c["name"] == "book_holds_a_rerun_of_this_registration" for c in iv_long["checks"]),
                  "7: a returned book with a line longer than a reading's is rejected before any line of it is parsed")

            # odd JSON is a rejection, not an error
            for label, files in (("a manifest that is a list", {"manifest.json": "[]"}),
                                 ("a manifest whose files are null", {"manifest.json": json.dumps({"files": None})}),
                                 ("a return that is a number", {"return.json": "7"})):
                odd = tmp / ("odd_" + re.sub(r"\W+", "_", label))
                _bundle(odd, vol_book, ret_ok)
                for name, text in files.items():
                    (odd / name).write_text(text)
                try:
                    check(intake_of(odd.name, odd)["outcome"] == "rejected", f"7: {label} is rejected")
                except Exception as exc:  # noqa: BLE001
                    check(False, f"7: {label} is rejected, not an error ({type(exc).__name__}: {exc})")

            def _make_evil_tar(dst: Path, mode: str) -> None:
                with tarfile.open(dst, "w") as tf:
                    if mode == "traversal":
                        ti = tarfile.TarInfo(name="../escape.txt")
                        ti.size = 3
                        tf.addfile(ti, io.BytesIO(b"bad"))
                    elif mode == "symlink":
                        ti = tarfile.TarInfo(name="return.json")
                        ti.type = tarfile.SYMTYPE
                        ti.linkname = "/etc/passwd"
                        tf.addfile(ti)
                    elif mode == "extra":
                        with tarfile.open(vol_tar, "r") as src_tf:
                            for m in src_tf.getmembers():
                                tf.addfile(m, src_tf.extractfile(m))
                        ti = tarfile.TarInfo(name="evil.py")
                        ti.size = 7
                        tf.addfile(ti, io.BytesIO(b"print()"))
                    elif mode == "oversize":
                        with tarfile.open(vol_tar, "r") as src_tf:
                            for m in src_tf.getmembers():
                                if m.name == "return.json":
                                    ti = tarfile.TarInfo(name="return.json")
                                    ti.size = lr.MAX_SMALL_BYTES + 1
                                    tf.addfile(ti, io.BytesIO(b"\0" * ti.size))
                                else:
                                    tf.addfile(m, src_tf.extractfile(m))
                    elif mode == "tamper_sheet":
                        with tarfile.open(vol_tar, "r") as src_tf:
                            for m in src_tf.getmembers():
                                data = src_tf.extractfile(m).read()
                                if m.name == "TRIAL_SHEET.md":
                                    data += b"\ntampered\n"
                                ti = tarfile.TarInfo(name=m.name)
                                ti.size = len(data)
                                tf.addfile(ti, io.BytesIO(data))

            for mode in ("traversal", "symlink", "extra", "oversize"):
                bad_tar = tmp / f"evil_{mode}.tar"
                _make_evil_tar(bad_tar, mode)
                check(refuses(lambda: lr.intake_rerun_bundle(bad_tar, prereg_dir, cal_book, tmp / f"stage_{mode}", allow_dirty=True, allow_other_runner=True),
                              lr.RerunRefused), f"7: the unpacking refuses a bundle with {mode}")
            (tmp / "vol.tar.gz").write_bytes(gzip.compress(vol_tar.read_bytes()))
            check(refuses(lambda: lr.intake_rerun_bundle(tmp / "vol.tar.gz", prereg_dir, cal_book, tmp / "stage_gz", allow_dirty=True, allow_other_runner=True),
                          lr.RerunRefused, "plain tar"),
                  "7: a compressed bundle is refused: nothing is inflated to learn its size")
            tamp_tar = tmp / "evil_tamper.tar"
            _make_evil_tar(tamp_tar, "tamper_sheet")
            iv_tamp = lr.intake_rerun_bundle(tamp_tar, prereg_dir, cal_book, tmp / "stage_tamper", allow_dirty=True, allow_other_runner=True)
            check(iv_tamp["outcome"] == "rejected" and any("TRIAL_SHEET.md" in r for r in iv_tamp["reject_reasons"]),
                  f"7: intake rejects tampered file sha256 ({iv_tamp['reject_reasons'][:1]})")
            check(refuses(lambda: lr.intake_rerun_bundle(vol_tar, prereg_dir, bk_cli, tmp / "stage_wrong_book", allow_dirty=True, allow_other_runner=True),
                          lr.RerunRefused, "home trial"),
                  "7: an intake against a book that does not hold the home trial does not begin")

            kit_cli = tmp / "kit_cli"
            stage_cli = tmp / "stage_cli"
            cli_tar = tmp / "cli_return.tar"
            check(
                lr_cli.main(["pack", str(prereg_dir), "--out", str(kit_cli), "--home-result", str(trial_out / "trial_result.json")]) == 0
                and lr_cli.main(["verify-kit", str(kit_cli)]) == 0
                and lr_cli.main(["run", str(kit_cli), "--out", str(cli_tar), "--volunteer", "bob", "--packed-stand-in"]) == 0
                and lr_cli.main(["intake", str(cli_tar), "--prereg", str(prereg_dir), "--book", str(cal_book),
                                 "--out", str(stage_cli), "--allow-dirty", "--allow-other-runner"]) == 0,
                "7: scripts/levelling_rerun.py pack, verify-kit, run and intake exit 0",
            )

            # ------------------------------------------------------------------ 8. no kept-out word
            section("8. No kept-out word in any Levelling file, sheet, sentence or rules file")
            audit_files = [
                "backend/level_book.py", "backend/weight_edits.py", "backend/levelling.py",
                "backend/weight_reader.py", "backend/level_sandbox.py", "backend/level_session.py",
                "backend/level_mcp.py", "backend/level_briefs.py", "backend/level_claim.py",
                "backend/level_routes.py", "backend/level_trial.py", "backend/level_export.py",
                "backend/level_rerun.py",
                "scripts/levelling.py", "scripts/level_recorded_cards.py",
                "scripts/levelling_trial.py", "scripts/levelling_export.py", "scripts/levelling_rerun.py",
                "frontend/src/screens/vault/compare.ts",
                "frontend/src/screens/vault/compareRecorded.ts",
                "frontend/src/screens/vault/ComparePanel.tsx",
            ]
            for rel in audit_files:
                txt = (ROOT_DIR / rel).read_text(encoding="utf-8")
                hits = [m.group(0) for m in KEPT_OUT_RE.finditer(txt)]
                check(not hits, f"8: {rel} has zero kept-out words (found: {hits})")
            written = "\n".join(SHEETS) + "\n" + rules_txt + "\n" + json.dumps(iv_ok) + "\n" + json.dumps(exp_own["manifest"])
            hits = [m.group(0) for m in KEPT_OUT_RE.finditer(written)]
            check(len(SHEETS) >= 5 and not hits,
                  f"8: every sheet, sentence, rules file, intake verdict and manifest written here has zero kept-out words (found: {hits[:5]})")
        finally:
            for env_k, prev in ((we.EDITS_ENV, prev_edits), (lex.EXPORT_ENV, prev_exp), (lb.BOOK_ENV, prev_bk)):
                if prev is None:
                    os.environ.pop(env_k, None)
                else:
                    os.environ[env_k] = prev

    elapsed = time.perf_counter() - t0
    print(f"\nPhase 5 CPU verification complete in {elapsed:.1f} s: {PASSED} passed, {FAILED} failed [ran on CPU stand-ins].")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
