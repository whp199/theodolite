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
                  and set(tc["sources"]) == set(lt.SOURCE_FILES) and all(tc["sources"].values())
                  and tc["implant_check"]["checked"] is True and "starter set" in tc["reworded_origin"]
                  and tc["max_fresh_readings"] == 2,
                  "3: the registration records its calibration, the runner's and reader's hashes, the implant label and where the reworded set stands")
            rules_txt = (prereg_dir / "RULES.md").read_text()
            check("OF RECORD only when" in rules_txt and "spends one of the brief's fresh readings" in rules_txt
                  and "kept-back frames, which a claim reading of the brief has" in rules_txt,
                  "3: RULES.md says what makes a trial of record, that it spends a fresh reading, and what its fresh items are")
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
            ex_part = lt.extract_folded_class({"items": [{"status": "ok", "item": {"name": "n0"}, "classes": {"small": {"folded": {
                "taken": True, "same_bits": True, "metrics": {"odds": {"mean": -1.0}}, "gap": {"odds": {"abs": 0.0}}}}}}]},
                [DrawnStub("n0"), DrawnStub("n1")], "small", "odds")
            check(ex_part["available"] is False and "1 of 2" in str(ex_part["why"]),
                  "4: a folded form read on some of the items is not the folded form of the reading")

            # the sentence: only a trial of record that passed has one, and it says what was found
            meta_s = {
                "date": "2026-11-19T00:00:00Z", "model_id": "m", "weights_id": "w" * 64, "unpacker": "compiled",
                "edit_id": e_trial.id, "edit_linears": e_trial.linears(), "main_dose": 2.0,
                "bars_sha256": bars_unit.id, "bars_calibration_seq": 3, "reading_index": 2, "max_fresh_readings": 2,
                "kept_back": {"frames": 8, "published": 8}, "book": {"name": "side.jsonl", "default": False},
                "design": {"match_tolerance": 2.0}, "prompt_named_site": False,
                "trial_sha256": "f" * 64, "of_record": True, "not_of_record": [], "rerun_of": None,
                "repo_commit": "c" * 40, "reworded_use_index": 1,
                "source_claim": {"seq": 10, "verdict": "lead", "reading_index": 1},
            }
            cfg_s = {"trial_name": "t", "reworded_id": rset.id, "reworded_origin": lt.reworded_origin(rset, b_unit)}
            sent = lt.trial_sentence(b_unit, cfg_s, fj_good, tj_pass, meta_s) or ""
            for needle in ("at dose 2", "fresh reading 2 of 2", "no author prompt arm was compared with it (none in the brief)",
                           "starter set in the instrument's source", "claim reading of this edit at book line 10",
                           "selection not matched", "taken in a side book", "kept-back frames, which a claim reading",
                           "in a process that had read nothing before it", "Passed."):
                check(needle in sent, f"4: a passed trial of record's sentence says \"{needle}\"")
            check("no author prompt arm matched it" not in sent and "owner-written" not in sent.lower()
                  and "This is a lead" not in sent and KEPT_OUT_RE.search(sent) is None,
                  "4: and does not say a prompt arm was matched against when none ran, nor who wrote the reworded set")
            SHEETS.append(sent)
            check(lt.trial_sentence(b_unit, cfg_s, fj_good, tj_pass, {**meta_s, "of_record": False}) is None
                  and lt.trial_sentence(b_unit, cfg_s, fj_good, tj_pass, {**meta_s, "rerun_of": "x" * 64}) is None
                  and lt.trial_sentence(b_unit, cfg_s, fj_good, {**tj_pass, "verdict": "failed"}, meta_s) is None,
                  "4: a dry run, a re-run and a trial that did not pass have no sentence")

            # ------------------------------------------------------------------ 5. a trial read on the packed stand-in
            section("5. A trial read: of record or refused, a dry run, read once, counted, judged again from its book")
            trial_out = tmp / "my_trial" / "result"
            check(lv.asked_in_process()["readings"] > 0, "5: the process counts the readings it has begun")
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
                  and "first thing its process reads" in msg[-1] and "registration folder" in msg[-1]
                  and "--dry-run" in msg[-1] and lb.verify(cal_book)["entries"] == n_before,
                  "5: a trial that would not be of record is refused, says each reason, and enters nothing in the book")

            t_res = lt.run_trial_of_record(hP, prereg_dir, out_dir=trial_out, book_path=cal_book, dry_run=True,
                                           _seed_override=b"trial-standin-seed-32bytes-00001")
            SHEETS.append(t_res["sheet"])
            check(t_res["verdict"] in ("passed", "failed") and t_res["of_record"] is False and t_res["claim_sentence"] is None
                  and t_res["sheet"].startswith("=== DRY RUN OF A TRIAL (not of record)")
                  and "(a dry run: not a verdict of record)" in t_res["sheet"] and "not of record: this process" in t_res["sheet"],
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
            check(t_res["meta"]["base_checksum_unchanged"] is True and t_res["meta"]["unpacker_fallbacks"] == 0
                  and t_res["meta"]["base_linears_checked"] >= 18,
                  "5: the base checksum is unchanged over the stand-in's linears, with no unpacker fallback")
            check((trial_out / "trial_result.json").is_file() and (trial_out / "TRIAL_SHEET.md").is_file()
                  and len(t_res["seed_hex"]) == 64 and t_res["seed_hex"] not in cal_book.read_text(),
                  "5: the result and the sheet are written; the draw's seed is in the result file and not in the book")
            check(lb.verify(cal_book)["ok"], "5: level book chain whole after the trial")
            for needle in ("CLAIM SHEET", "earlier fresh readings on this brief (0)", "earlier trials on this brief (0)",
                           "habitat (info): not run", "fresh items: new fillings of the brief's kept-back frames",
                           "follows: no claim reading of this edit stands in this book"):
                check(needle in t_res["sheet"], f"5: the sheet prints \"{needle}\"")
            check(KEPT_OUT_RE.search(t_res["sheet"]) is None and KEPT_OUT_RE.search(rules_txt) is None,
                  "5: the sheet and RULES.md contain zero kept-out words")

            rj = lt.rejudge_trial(cal_book, t_res["trial_sha256"], v_ok)
            check(rj["as_noted"] and rj["verdict"] == t_res["verdict"] and len(rj["items"]["target"]["small"]) == 4
                  and len(rj["items"]["reworded"]["large"]) == 8 and rj["readings"]["matched"]["edit_id"] == e_trial.id,
                  "5: judged again from the lines its note names, the trial gives the verdict its note records")

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
            check(run_refused("calibration reading", prereg_car, "other_out", book=bk_o, dry_run=True)
                  or "registered against" in msg[-1],
                  "5: a trial is read in the book it was registered against, and in no other")

            # a trial of record: a process of its own, and a repository of its own that holds the folder, committed
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
            cond_uncommitted = lt.record_conditions(pre_rec, cfg_rec, repo_root=repo)
            _git(repo, "add", "-A")
            _git(repo, "commit", "-q", "-m", "registration")
            cond_now = lt.record_conditions(pre_rec, cfg_rec, repo_root=repo)
            (pre_rec / "RULES.md").write_text((pre_rec / "RULES.md").read_text() + " ")
            cond_touched = lt.record_conditions(pre_rec, cfg_rec, repo_root=repo)
            _git(repo, "checkout", "-q", "--", ".")
            check(any("uncommitted" in r for r in cond_uncommitted["reasons"])
                  and any("in no commit" in r for r in cond_uncommitted["reasons"])
                  and any("differs from what is committed" in r for r in cond_touched["reasons"])
                  and [r for r in cond_now["reasons"] if not r.startswith("this process")] == []
                  and cond_now["checkout"]["dirty"] is False and cond_now["checkout"]["folder_committed"] is True
                  and cond_now["checkout"]["sources_match"] is True,
                  "5: the conditions of record see a folder that is not committed or was changed after, and a clean checkout "
                  f"that holds it committed ({cond_uncommitted['reasons']}; {cond_touched['reasons']}; {cond_now['reasons']})")
            outside = lt.record_conditions(prereg_dir, v_ok["trial"], repo_root=repo)
            check(any("outside the checkout" in r for r in outside["reasons"]),
                  "5: a registration folder outside the checkout is in no commit, and is not of record")
            moved = lt.record_conditions(pre_rec, {**cfg_rec, "sources": {k: "0" * 64 for k in lt.SOURCE_FILES}}, repo_root=repo)
            check(any("runner or reader is not the one registered" in r for r in moved["reasons"]),
                  "5: a runner or reader changed since registration is not of record")
            code = (
                "import os, sys, json\n"
                "os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['HF_HUB_OFFLINE'] = '1'\n"
                f"sys.path.insert(0, {str(ROOT_DIR)!r}); sys.path.insert(0, {str(ROOT_DIR / 'scripts')!r})\n"
                "import torch; torch.set_num_threads(1)\n"
                "from pathlib import Path\n"
                "from backend import levelling as lv, level_trial as lt\n"
                "import check_dequant as cd\n"
                "h = lv.handle(cd.packed_stand_in()[0])\n"
                f"r = lt.run_trial_of_record(h, Path({str(pre_rec)!r}), out_dir=Path({str(tmp / 'rec_out')!r}), "
                f"book_path=Path({str(bk_rec)!r}), _repo_root=Path({str(repo)!r}))\n"
                "print('RESULT ' + json.dumps({'of_record': r['of_record'], 'verdict': r['verdict'], "
                "'why': r['not_of_record'], 'sheet': r['sheet']}))\n"
            )
            env = {**os.environ, lb.BOOK_ENV: str(bk_rec), we.EDITS_ENV: str(tmp / "saved_edits")}
            proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=600)
            got_line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
            got = json.loads(got_line[7:]) if got_line else {}
            rec_sheet = str(got.get("sheet", ""))
            check(got.get("of_record") is True and got.get("why") == [] and got.get("verdict") in ("passed", "failed")
                  and rec_sheet.startswith("=== TRIAL OF RECORD: rec ===")
                  and "process: had read nothing before the trial" in rec_sheet
                  and "registration folder committed: yes" in rec_sheet
                  and "from a file given at registration" in rec_sheet,
                  "5: read first in a process of its own, from a clean checkout that holds its folder, a trial is of record "
                  f"({proc.stderr.strip()[-300:] if not got else rec_sheet.splitlines()[0]})")
            SHEETS.append(rec_sheet)
            rec_entries = list(lb.entries(bk_rec))
            check(rec_entries[-1]["body"].get("note_type") == lt.NOTE_OF_RECORD and lb.verify(bk_rec)["ok"],
                  "5: and its verdict is entered in the book as a trial of record")
            rec_card = lroutes.build_replay_card(rec_entries[-1])
            check(rec_card["of_record"] is True and rec_card["status"] == got.get("verdict"),
                  "5: only a trial of record's card shows its verdict as a verdict")

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
                  and meta_ab.get("theodolite.levelling.served_base_is_edit_base") == "another object (a GGUF)"
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
            twice = we.Edit("x" * 64, (e_trial.terms[0], e_trial.terms[0]), dict(e_trial.base), "", {})
            check(refuses(lambda: lex.build_lora_tensors(twice, target_mod), lex.ExportRefused, "named twice"),
                  "6: an edit that names a column twice is not compiled (as the kernel builds none)")

            out_lr = tmp / "owner_exports" / "carried_lr.gguf"
            exp_lr = lex.export_lora_adapter(e_lr, target_gguf, out_path=out_lr)
            ver_lr = lex.verify_exported_lora(out_lr, edit=e_lr, target=target_gguf)
            check(exp_lr["recompiled_from_recipe"] is False and any("re-measure" in w for w in exp_lr["warnings"]) and ver_lr["ok"],
                  "6: carried lowrank export warns to re-measure on served GGUF and passes verify_exported_lora")

            # a trial's verdict goes into the file for its own edit at its own dose, and says what kind it was
            check(refuses(lambda: lex.export_lora_adapter(e_mix, target_mod, out_path=tmp / "owner_exports" / "wrong_edit.gguf",
                                                          trial_result=t_res), lex.ExportRefused, "not of the edit exported")
                  and refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "wrong_dose.gguf",
                                                              dose=2.0, trial_result=t_res), lex.ExportRefused, "dose"),
                  "6: another edit's trial, or this edit's trial at another dose, is refused")
            exp_own = lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "own.gguf", trial_result=t_res,
                                              implant_at_registration=v_ok["trial"]["implant_check"])
            _t, m_own = el.read_lora(Path(exp_own["gguf_path"]))
            check(m_own["theodolite.levelling.trial_verdict"] == t_res["verdict"]
                  and m_own["theodolite.levelling.trial_kind"] == "a dry run"
                  and m_own["theodolite.levelling.trial_applies_to_served_base"].startswith("yes")
                  and m_own["theodolite.levelling.book"].endswith("is the trial's verdict")
                  and "earlier trials 0" in m_own["theodolite.levelling.book_before_it"]
                  and m_own["theodolite.levelling.implant_checked_at"] == "the trial's registration",
                  "6: its own trial's verdict goes in, with its kind, the book line checked, and where the implant label was worked")
            exp_gg = lex.export_lora_adapter(e_trial, target_gguf, out_path=tmp / "owner_exports" / "own_gguf.gguf", trial_result=t_res)
            _t, m_gg = el.read_lora(Path(exp_gg["gguf_path"]))
            check(m_gg["theodolite.levelling.trial_applies_to_served_base"].startswith("no: the trial read")
                  and m_gg["theodolite.levelling.implant_flagged"] == "not checked",
                  "6: over a GGUF the file says the trial read another base; with no model at hand the implant label is 'not checked'")
            bad_line = {**t_res, "book": {**t_res["book"], "verdict_hash": "0" * 64}}
            check(refuses(lambda: lex.export_lora_adapter(e_trial, target_mod, out_path=tmp / "owner_exports" / "bad_book.gguf",
                                                          trial_result=bad_line), lex.ExportRefused, "do not belong together"),
                  "6: a trial result whose book line is not its verdict is refused")
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

            ckpt = wr.export_stand_in_safetensors(hP.lm, tmp / "ckpt", model_id=hP.model_id)
            cli_gguf = tmp / "owner_exports" / "cli_export.gguf"
            rc_lora = lex_cli.main(["lora", "--trial", str(prereg_dir), "--weights-dir", str(ckpt), "-o", str(cli_gguf),
                                    "--trial-result", str(trial_out / "trial_result.json")])
            m_cli = el.read_lora(cli_gguf)[1] if cli_gguf.is_file() else {}
            check(rc_lora == 0 and m_cli.get("theodolite.levelling.edit_id") == e_trial.id
                  and m_cli.get("theodolite.levelling.served_base_is_edit_base") == "yes"
                  and m_cli.get("theodolite.levelling.implant_checked_at") == "the trial's registration"
                  and m_cli.get("theodolite.levelling.trial_kind") == "a dry run",
                  "6: the command line exports a registration off the model, against a checkpoint it shows to be the edit's base")
            check(lex_cli.main(["verify", str(cli_gguf), "--trial", str(prereg_dir), "--weights-dir", str(ckpt)]) == 0
                  and lex_cli.main(["verify", str(cli_gguf), "--trial", str(prereg_dir), "--gguf", str(synth_gguf)]) == 1
                  and lex_cli.main(["show", str(cli_gguf)]) == 0,
                  "6: the command line's verify passes on the base the file was compiled against, and fails on another")
            (tmp / "twice.json").write_text(json.dumps({"terms": [{"layer": 2, "silence": [3, 3]}]}))
            check(lex_cli.main(["lora", "--edit", str(tmp / "twice.json"), "--weights-dir", str(ckpt),
                                "-o", str(tmp / "owner_exports" / "bare.gguf")]) == 2
                  and not (tmp / "owner_exports" / "bare.gguf").exists(),
                  "6: a bare recipe is not exported off the model: nothing there would gate it")
            try:
                default_ok = isinstance(lex.TargetBase.default_checkpoint().record, str)
            except lex.ExportRefused:
                default_ok = True
            check(default_ok, "6: with no base named, the cached checkpoint is opened or refused in words (no other error)")

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
            kit_twin = tmp / "kit_twin"
            shutil.copytree(kit_dir, kit_twin)
            (kit_twin / "RULES.md").write_text((kit_twin / "RULES.md").read_text() + "\nedited\n")
            lt._write_sha256sums(kit_twin)
            check(refuses(lambda: lr.verify_rerun_kit(kit_twin), lr.RerunRefused, "home registration"),
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

            stage_ok = tmp / "intake_ok"
            iv_ok = lr.intake_rerun_bundle(vol_tar, prereg_dir, cal_book, stage_ok, allow_dirty=True)
            check(iv_ok["outcome"] == "accepted" and iv_ok["match_class"] == "same_figures"
                  and (stage_ok / "intake_verdict.json").is_file()
                  and iv_ok["class_diffs"]["target:small"]["items"] == 4 and iv_ok["class_diffs"]["reworded:large"]["items"] == 8
                  and all(v["largest_difference"] == 0.0 for v in iv_ok["class_diffs"].values())
                  and iv_ok["judged_from"].startswith("the returned book") and len(iv_ok["on_the_volunteers_word"]) == 2,
                  f"7: the same card re-running its own kit is accepted with the same figures, item by item ({iv_ok['outcome']}, "
                  f"{iv_ok['match_class']}; {iv_ok['reject_reasons'][:1]})")

            ret_ok = {"v": 1, "volunteer": "nobody", "trial_sha256": reg["trial_sha256"], "verdict": t_res["verdict"],
                      "fresh_verdict": t_res["fresh_verdict"]}

            def intake_of(tag: str, folder: Path) -> Dict[str, Any]:
                _pack(folder, tmp / f"{tag}.tar")
                return lr.intake_rerun_bundle(tmp / f"{tag}.tar", prereg_dir, cal_book, tmp / f"stage_{tag}", allow_dirty=True)

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
                check(refuses(lambda: lr.intake_rerun_bundle(bad_tar, prereg_dir, cal_book, tmp / f"stage_{mode}", allow_dirty=True),
                              lr.RerunRefused), f"7: the unpacking refuses a bundle with {mode}")
            (tmp / "vol.tar.gz").write_bytes(gzip.compress(vol_tar.read_bytes()))
            check(refuses(lambda: lr.intake_rerun_bundle(tmp / "vol.tar.gz", prereg_dir, cal_book, tmp / "stage_gz", allow_dirty=True),
                          lr.RerunRefused, "plain tar"),
                  "7: a compressed bundle is refused: nothing is inflated to learn its size")
            tamp_tar = tmp / "evil_tamper.tar"
            _make_evil_tar(tamp_tar, "tamper_sheet")
            iv_tamp = lr.intake_rerun_bundle(tamp_tar, prereg_dir, cal_book, tmp / "stage_tamper", allow_dirty=True)
            check(iv_tamp["outcome"] == "rejected" and any("TRIAL_SHEET.md" in r for r in iv_tamp["reject_reasons"]),
                  f"7: intake rejects tampered file sha256 ({iv_tamp['reject_reasons'][:1]})")
            check(refuses(lambda: lr.intake_rerun_bundle(vol_tar, prereg_dir, bk_cli, tmp / "stage_wrong_book", allow_dirty=True),
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
                                 "--out", str(stage_cli), "--allow-dirty"]) == 0,
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
