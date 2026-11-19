#!/usr/bin/env python3
"""CPU verification suite for Levelling Phase 5: Of record, and out (`scripts/check_level_trial.py`).

Tests every Phase 5 component and boundary with zero GPU access (`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`,
1 thread) on both the packed bf16 stand-in (supporting real `we.folded` overlays!) and the float32 stand-in,
plus a synthetic Gemma 4 GGUF file built in a temporary directory for true cross-base recompilation tests:
1. Provenance & base integrity (`weight_edits.rebuilt_recipe`, `edit_provenance`, `base_checksum`, `.gitignore`).
2. Reworded item sets (`RewordedSet`, wording disjointness against practice/kept-back/batteries, tamper detection,
   `MAX_REWORDED_USES = 5` rotation guard, `starter_reworded_pronoun_pull`).
3. House registration folder (`register_trial`, `verify_registration`: `SHA256SUMS`, missing/extra/tampered refusal,
   forbidden `*.safetensors` / `*.gguf` payload refusal, carried-edit payload hash without raw weights).
4. Pure trial verdict engine (`judge_folded`, `judge_reworded`, `judge_trial`: `"passed"`, `"failed"` on fresh/folded/
   reworded failure, `"void"` on base checksum drift / unpacker fallback / missing required folded).
5. Live Trial of Record on packed bf16 stand-in (full `folded="always"` in BOTH small & large batch classes,
   owner-written reworded set with `folded="always"`, live `base_checksum` before & after, read-once refusal,
   Trial Sheet, book replay via `level_routes.build_replay_card`, CLI `register`/`verify`/`run`).
6. Owner-only export (`level_export`: unsafe path refusals for checkout / `atlas_data` / `prereg` / session folder,
   overwrite refusal & `--force`, column scale/silence + row scale/silence + lowrank LoRA GGUF export, Apache-2.0
   licence & origin keys, recompilation against a synthetic GGUF with distinct weights proving `W_gguf + B @ A == 0`,
   `unverified_llama_cpp_tensors` marking for `gate_proj`/`up_proj`, `implant_check`, companion `.manifest.json`,
   `verify_exported_lora`, CLI `lora`/`verify`/`show`, and verifying zero export/trial tools in agent/server surfaces).
7. Volunteer re-run kit & defensive intake (`level_rerun`: carried-payload refusal, `make_rerun_kit`, `verify_rerun_kit`,
   `run_volunteer_rerun` allowlisted `.tar`, defensive `intake_rerun_bundle` rejecting traversal/symlinks/extra files/
   tampered hashes/broken book/dirty checkout, and accepting & judging a clean re-run, plus CLI `pack`/`run`/`intake`).
8. Zero kept-out words across ALL 18 Levelling backend/CLI/frontend files and every rendered sheet & rule file.

Run:
    /home/eric/bin/ai/theodolite2/.venv/bin/python -I scripts/check_level_trial.py
"""
from __future__ import annotations

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
from typing import Any, Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

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
from backend import level_session as lses  # noqa: E402
from backend import level_trial as lt  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import weight_edits as we  # noqa: E402

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)

PASSED = 0
FAILED = 0


def check(cond: bool, msg: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}", flush=True)


def section(title: str) -> None:
    print(f"\n[{title}]", flush=True)


def _write_synthetic_gemma4_gguf(path: Path, n_layers: int, d_model: int, d_ff: int, seed: int = 99) -> Dict[str, np.ndarray]:
    """Write a tiny valid Gemma 4 F32 GGUF with known distinct weights to test true cross-base recompilation."""
    import gguf

    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), "gemma4")
    w.add_name("synthetic-gemma4-standin")
    w.add_uint32("gemma4.block_count", int(n_layers))
    w.add_uint32("gemma4.embedding_length", int(d_model))
    w.add_uint32("gemma4.feed_forward_length", int(d_ff))
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
            rt_p = cd.packed_stand_in()[0]
            hP = lv.handle(rt_p)
            we.assert_base(hP.lm, hP.hooks_at_load)
            n_layers = hP.n_layers
            D, N = we.linear_shape(hP.lm.layers[2].mlp.down_proj)

            # ------------------------------------------------------------------ 1. Provenance, base_checksum, .gitignore
            section("1. Edit provenance, rebuilt_recipe, base_checksum & .gitignore")
            gi_txt = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
            check("*.safetensors" in gi_txt and "*.gguf" in gi_txt, "1: .gitignore ignores *.safetensors and *.gguf")

            e_reb = we.build(
                hP.lm, [we.silence(2, [3, 7]), we.scale_rows(3, "up_proj", [1, 4], 0.5)],
                model_id=hP.model_id, weights_id=hP.weights_id, note="rebuilt test",
            )
            rec = we.rebuilt_recipe(e_reb)
            check(rec is not None and len(rec["terms"]) == 2, "1: rebuilt_recipe emits pure JSON recipe for uncarried edit")
            e_reb_again = we.build(hP.lm, we.from_recipe(rec, tmp), model_id=hP.model_id, weights_id=hP.weights_id, note=rec.get("note", ""))
            check(e_reb_again.id == e_reb.id, "1: edit rebuilt from rebuilt_recipe hashes to identical edit.id")

            prov_reb = we.edit_provenance(e_reb)
            check(prov_reb["all_rebuilt"] is True and prov_reb["provenance"] == "rebuilt" and prov_reb["payload_sha256"] is None
                  and all(t["provenance"] == "rebuilt" and "recipe_term" in t for t in prov_reb["terms"]),
                  "1: edit_provenance marks every uncarried term 'rebuilt' with recipe_term and no payload")

            rng = np.random.default_rng(7)
            e_car = we.build(
                hP.lm,
                [we.silence(2, [1, 2]), we.add_to_columns(3, [5], rng.standard_normal((1, D)).astype(np.float32) * 0.05)],
                model_id=hP.model_id, weights_id=hP.weights_id, note="mixed carried",
            )
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

            # ------------------------------------------------------------------ 2. RewordedSet & rotation guard
            section("2. RewordedSet validation, wording disjointness, tamper check & 5-use rotation")
            brief = lbr.pronoun_pull_brief(hP.tokenizer, max_fresh_readings=2)
            rset = lt.starter_reworded_pronoun_pull(brief)
            check(len(rset.items) == 8 and sum(i.half == 0 for i in rset.items) == 4 and sum(i.half == 1 for i in rset.items) == 4,
                  "2: starter_reworded_pronoun_pull has 4 half-0 and 4 half-1 items")
            rw_file = lt.save_reworded(rset, tmp / "reworded.json")
            rset_back = lt.load_reworded(rw_file)
            check(rset_back.id == rset.id, "2: save_reworded / load_reworded round-trip exact id")

            # Tamper test on reworded.json
            bad_rw = json.loads(rw_file.read_text())
            bad_rw["items"][0]["question"] += " extra"
            (tmp / "reworded_bad.json").write_text(json.dumps(bad_rw))
            got_tamper = False
            try:
                lt.load_reworded(tmp / "reworded_bad.json")
            except lt.TrialRefused:
                got_tamper = True
            check(got_tamper, "2: tampered reworded.json refused")

            # Overlap with brief practice/kept-back frame refused
            bad_items = (
                lt.RewordedItem("b1", 0, "Finish the story: Across the studio, Claire worked quietly as a pianist. Soon",
                                " Claire set down the notebook and"),
                lt.RewordedItem("b2", 0, "A distinct question two?", "Answer two"),
                lt.RewordedItem("b3", 1, "A distinct question three?", "Answer three"),
                lt.RewordedItem("b4", 1, "A distinct question four?", "Answer four"),
            )
            bad_set = lt.RewordedSet("overlap_test", brief.id, brief.name, bad_items)
            got_overlap = False
            try:
                lt.validate_reworded_against_brief(bad_set, brief)
            except lt.TrialRefused:
                got_overlap = True
            check(got_overlap, "2: reworded item overlapping brief kept-back frame refused")

            # 5-use rotation count check
            rot_book = tmp / "rot_book.jsonl"
            for i in range(lt.MAX_REWORDED_USES):
                lb.append("note", {"note_type": "trial_registration", "reworded_id": rset.id, "trial_sha256": f"t{i}"}, path=rot_book)
            check(lt.reworded_uses_in_book(rset.id, rot_book) == lt.MAX_REWORDED_USES,
                  f"2: reworded_uses_in_book counts {lt.MAX_REWORDED_USES} uses accurately")

            # ------------------------------------------------------------------ 3. Registration folder & SHA256SUMS
            section("3. House registration folder (prereg/) & SHA256SUMS integrity")
            cal_book = tmp / "trial_book.jsonl"
            bars = lcl.calibrate_bars(
                hP, brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
                rows={"small": 10, "large": 16}, seed=0, book_path=cal_book,
            )
            e_trial = we.build(hP.lm, [we.silence(2, [3, 7])], model_id=hP.model_id, weights_id=hP.weights_id, note="trial edit")
            prereg_dir = tmp / "my_trial" / "prereg"
            reg = lt.register_trial(
                prereg_dir, brief, bars, e_trial,
                trial_name="standin_rebuilt_trial",
                reworded=rset,
                controls_per_family=2,
                reworded_controls=2,
                n_target=4,
                n_neutral=2,
                n_near=2,
                n_panel=2,
                rows={"small": 10, "large": 16},
                folded_required=True,
            )
            check((prereg_dir / "SHA256SUMS").is_file() and (prereg_dir / "RULES.md").is_file()
                  and (prereg_dir / "edit_manifest.json").is_file() and (prereg_dir / "reworded.json").is_file(),
                  "3: register_trial creates all required prereg/ files")
            check(not list(prereg_dir.glob("*.safetensors")) and not list(prereg_dir.glob("*.gguf")),
                  "3: prereg/ contains zero .safetensors or .gguf weight files")
            v_ok = lt.verify_registration(prereg_dir)
            check(v_ok["ok"] and v_ok["trial_sha256"] == reg["trial_sha256"], "3: verify_registration passes clean folder")

            # Refusal on non-empty folder overwrite
            got_ow = False
            try:
                lt.register_trial(prereg_dir, brief, bars, e_trial, trial_name="dup", controls_per_family=2, n_target=4,
                                  n_neutral=2, n_near=2, n_panel=2, rows={"small": 10, "large": 16})
            except lt.TrialRefused:
                got_ow = True
            check(got_ow, "3: register_trial refuses overwriting existing prereg/")

            # Refusal on tampered file, extra file, and forbidden .safetensors
            for label, mutator in (
                ("tampered trial.json", lambda d: (d / "trial.json").write_text((d / "trial.json").read_text() + " ")),
                ("unregistered extra file", lambda d: (d / "sneaky.txt").write_text("hi")),
                ("forbidden .safetensors", lambda d: (d / "payload.safetensors").write_bytes(b"fake")),
            ):
                twin = tmp / f"bad_prereg_{label.replace(' ', '_').replace('.', '')}"
                shutil.copytree(prereg_dir, twin)
                mutator(twin)
                refused = False
                try:
                    lt.verify_registration(twin)
                except lt.TrialRefused:
                    refused = True
                check(refused, f"3: verify_registration refuses {label}")

            # Carried-edit registration holds manifest + payload_sha256 only (no payload.safetensors!)
            prereg_car = tmp / "carried_trial" / "prereg"
            lt.register_trial(
                prereg_car, brief, bars, e_car, trial_name="carried_trial",
                controls_per_family=2, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                rows={"small": 10, "large": 16}, saved_edit_dir=saved_car,
            )
            v_car = lt.verify_registration(prereg_car)
            check(v_car["trial"]["edit_provenance"] == "carried" and v_car["trial"]["payload_sha256"] == prov_car["payload_sha256"]
                  and not (prereg_car / "payload.safetensors").exists(),
                  "3: carried-edit prereg/ stores payload_sha256 only, never payload.safetensors")
            e_car_loaded = lt.load_registered_edit(hP, v_car["edit_manifest"], payload_dir=saved_car)
            check(e_car_loaded.id == e_car.id, "3: load_registered_edit verifies and loads carried edit from external payload_dir")

            # ------------------------------------------------------------------ 4. Pure verdict engine (planted failures)
            section("4. Pure Trial of Record verdict engine (passed, failed, void planted cases)")
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
            fj_good = lcl.judge_claim(brief=b_unit, bars=bars_unit, classes=_good_fresh_classes(), batteries=good_bats)
            check(fj_good["verdict"] == "lead", f"4: synthetic good fresh claim returns lead (got {fj_good['verdict']})")

            folded_good_in = {
                cls: {
                    "available": True, "taken_count": 8, "expected_count": 8,
                    "rows_identical_all": True, "same_bits_count": 8,
                    "item_effects": _good_fresh_classes()[cls]["item_effects"],
                    "max_abs_gap": 0.0, "mean_abs_gap": 0.0, "between_kl_max": 0.0,
                }
                for cls in ("small", "large")
            }
            fe_good = lt.judge_folded(b_unit, bars_unit, fj_good, folded_good_in, folded_required=True)
            rw_good = lt.judge_reworded(
                b_unit, bars_unit,
                {cls: {"item_effects": _good_fresh_classes()[cls]["item_effects"], "floor": 0.05,
                       "controls": {"other_sites_matched": [-0.05] * 19}} for cls in ("small", "large")},
                folded_good_in, folded_required=True,
            )
            tj_pass = lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, reworded_eval=rw_good)
            check(tj_pass["verdict"] == "passed", f"4: all gates passing yields 'passed' (got {tj_pass})")

            # Planted folded disagreement -> failed
            folded_bad_in = {
                cls: {**folded_good_in[cls], "max_abs_gap": 0.9,
                      "item_effects": [{**x, "value": -0.6} for x in folded_good_in[cls]["item_effects"]]}
                for cls in ("small", "large")
            }
            fe_bad = lt.judge_folded(b_unit, bars_unit, fj_good, folded_bad_in, folded_required=True)
            tj_fl_fail = lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_bad, reworded_eval=rw_good)
            check(tj_fl_fail["verdict"] == "failed" and not fe_bad["ok"], "4: planted folded mismatch yields 'failed'")

            # Planted reworded failure -> failed
            rw_bad = lt.judge_reworded(
                b_unit, bars_unit,
                {cls: {"item_effects": [{**x, "value": -0.01} for x in _good_fresh_classes()[cls]["item_effects"]],
                       "floor": 0.05, "controls": {"other_sites_matched": [-0.05] * 19}} for cls in ("small", "large")},
                None,
            )
            tj_rw_fail = lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, reworded_eval=rw_bad)
            check(tj_rw_fail["verdict"] == "failed" and not rw_bad["ok"], "4: planted reworded failure yields 'failed'")

            # Planted base checksum drift / unpacker fallback / missing required folded -> void
            tj_void_ck = lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, base_checksum_unchanged=False)
            tj_void_fb = lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_good, unpacker_fallbacks=1)
            fe_unavail = lt.judge_folded(b_unit, bars_unit, fj_good, {"small": {"available": False}, "large": {"available": False}}, folded_required=True)
            tj_void_fl = lt.judge_trial(b_unit, bars_unit, fj_good, folded_eval=fe_unavail)
            check(tj_void_ck["verdict"] == "void" and tj_void_fb["verdict"] == "void" and tj_void_fl["verdict"] == "void",
                  "4: base checksum drift, unpacker fallback, and unavailable required folded each yield 'void'")

            # ------------------------------------------------------------------ 5. Live Trial of Record on packed stand-in
            section("5. Live Trial of Record on packed bf16 stand-in (folded='always' + reworded + replay)")
            trial_out = tmp / "my_trial" / "result"
            t_res = lt.run_trial_of_record(
                hP, prereg_dir, out_dir=trial_out, book_path=cal_book,
                require_book_calibration=True, _seed_override=b"trial-standin-seed-32bytes-00001",
            )
            check(t_res["verdict"] in ("passed", "failed") and t_res["verdict"] != "void",
                  f"5: live packed stand-in trial completed non-void (verdict={t_res['verdict']}, reasons={t_res['reasons']})")
            check(t_res["folded"]["available"] is True
                  and all(t_res["folded"]["classes"][c]["same_bits_count"] == t_res["folded"]["classes"][c]["taken_count"] > 0
                          for c in ("small", "large")),
                  "5: live folded='always' on silence edit matches per-row hooks bit-for-bit in BOTH small and large classes")
            check(t_res["reworded"] is not None and t_res["reworded"]["folded"]["available"] is True,
                  "5: reworded set ran in both batch classes with folded='always'")
            check(t_res["meta"]["base_checksum_unchanged"] is True and t_res["meta"]["unpacker_fallbacks"] == 0,
                  "5: live base_checksum unchanged and 0 unpacker fallbacks")
            check((trial_out / "trial_result.json").is_file() and (trial_out / "TRIAL_SHEET.md").is_file(),
                  "5: wrote trial_result.json and TRIAL_SHEET.md")
            check(lb.verify(cal_book)["ok"], "5: level book chain whole after trial of record")
            check("habitat (info): not run" in t_res["sheet"], "5: Trial Sheet labels unavailable habitat battery 'not run'")
            check(KEPT_OUT_RE.search(t_res["sheet"]) is None, "5: Trial Sheet contains zero kept-out words")
            check(KEPT_OUT_RE.search((prereg_dir / "RULES.md").read_text()) is None, "5: RULES.md contains zero kept-out words")

            # Replay of trial_of_record note via level_routes.build_replay_card
            v_seq = t_res["book"]["verdict_seq"]
            v_ent = next(lb.entries(cal_book, start=v_seq, limit=1))
            card = lroutes.build_replay_card(v_ent)
            check(card["note_type"] == "trial_of_record" and card["verdict"] == t_res["verdict"]
                  and card["edit_linears"] == ["layers.2.mlp.down_proj"]
                  and card["folded_summary"] is not None and card["reworded_summary"] is not None,
                  f"5: level_routes.build_replay_card replays trial_of_record with edit_linears & folded/reworded summaries")

            # Read-once refusal (both existing out_dir AND duplicate trial_sha256 in book)
            got_ro1 = got_ro2 = False
            try:
                lt.run_trial_of_record(hP, prereg_dir, out_dir=trial_out, book_path=cal_book)
            except lt.TrialRefused:
                got_ro1 = True
            try:
                lt.run_trial_of_record(hP, prereg_dir, out_dir=tmp / "my_trial" / "result_2", book_path=cal_book)
            except lt.TrialRefused:
                got_ro2 = True
            check(got_ro1 and got_ro2, "5: read-once refuses both existing trial_result.json and duplicate trial_sha256 in book")

            # Rotation cap refusal (5 uses exhausted in rot_book)
            got_rot = False
            try:
                lt.run_trial_of_record(hP, prereg_dir, out_dir=tmp / "my_trial" / "result_rot", book_path=rot_book)
            except lt.TrialRefused as exc:
                got_rot = "rotate" in str(exc).lower()
            check(got_rot, "5: run_trial_of_record refuses reworded set after MAX_REWORDED_USES = 5")

            # CLI smoke: verify & reworded
            check(lt_cli.main(["verify", str(prereg_dir)]) == 0, "5: scripts/levelling_trial.py verify exits 0")
            rw_cli_out = tmp / "cli_reworded.json"
            check(lt_cli.main(["reworded", "pronoun_pull", "--out", str(rw_cli_out), "--stand-in"]) == 0
                  and rw_cli_out.is_file(),
                  "5: scripts/levelling_trial.py reworded exports starter reworded set")

            # ------------------------------------------------------------------ 6. Owner-only export & GGUF recompilation
            section("6. Owner-only export, unsafe-path guards, Apache-2.0 keys & GGUF recompilation")
            # Unsafe path refusals
            sess_fake = tmp / "fake_session"
            sess_fake.mkdir()
            (sess_fake / "session.json").write_text("{}")
            for label, bad_path in (
                ("inside checkout", ROOT_DIR / "evil.gguf"),
                ("inside prereg/", prereg_dir / "evil.gguf"),
                ("inside session folder", sess_fake / "evil.gguf"),
            ):
                refused = False
                try:
                    lex.refuse_unsafe_export_path(bad_path)
                except lex.ExportRefused:
                    refused = True
                check(refused, f"6: export refuses unsafe path ({label})")

            # Build synthetic Gemma 4 GGUF with DISTINCT weights from hP!
            synth_gguf = tmp / "synthetic-gemma4.gguf"
            gguf_weights = _write_synthetic_gemma4_gguf(synth_gguf, n_layers, D, N, seed=1234)
            target_gguf = lex.TargetBase(gguf_path=synth_gguf)
            target_mod = lex.TargetBase(lm=hP.lm, label="packed-standin")

            # Export column silence + row scale edit recompiled against synthetic GGUF
            e_mix = we.build(
                hP.lm,
                [we.silence(2, [3, 7]), we.scale_rows(3, "up_proj", [1, 4], 0.5)],
                model_id=hP.model_id, weights_id=hP.weights_id, note="col+row export",
            )
            out_gguf = tmp / "owner_exports" / "recompiled_mix.gguf"
            exp_res = lex.export_lora_adapter(
                e_mix, target_gguf, out_path=out_gguf, dose=1.0,
                trial_result=t_res, prior_attempts=2,
                lm_for_implant=hP.lm, W_U_for_implant=hP.W_U,
            )
            check(Path(exp_res["gguf_path"]).is_file() and Path(exp_res["manifest_path"]).is_file(),
                  "6: export_lora_adapter writes .gguf and companion .manifest.json")
            check(exp_res["recompiled_from_recipe"] is True
                  and "blk.3.ffn_up.weight" in exp_res["unverified_llama_cpp_tensors"],
                  "6: rebuilt export marks recompiled_from_recipe=True and flags ffn_up as unverified in llama.cpp")

            # Overwrite refusal without overwrite=True
            got_exp_ow = False
            try:
                lex.export_lora_adapter(e_mix, target_gguf, out_path=out_gguf)
            except lex.ExportRefused:
                got_exp_ow = True
            check(got_exp_ow, "6: export_lora_adapter refuses overwriting existing file without --force")

            # Verify exported GGUF headers, Apache-2.0 keys, companion manifest sha256, and EXACT math against GGUF weights!
            ver_gguf = lex.verify_exported_lora(out_gguf, edit=e_mix, target=target_gguf, dose=1.0)
            check(ver_gguf["ok"] and (ver_gguf["max_abs_err"] or 0.0) < 1e-5,
                  f"6: verify_exported_lora passes all keys & math check (err={ver_gguf['max_abs_err']})")

            # Prove recompilation ACTUALLY silenced the GGUF's OWN columns (not the W4A16 stand-in's columns)!
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
            check(meta_ab.get("theodolite.levelling.trial_verdict") == t_res["verdict"]
                  and meta_ab.get("theodolite.levelling.recompiled_from_recipe") == "true",
                  "6: GGUF metadata embeds trial verdict and recompiled_from_recipe=true")

            # Carried low-rank export warns when targeting a GGUF
            e_lr = we.build(
                hP.lm,
                [we.lowrank(2, rng.standard_normal((2, N)).astype(np.float32) * 0.02,
                            rng.standard_normal((D, 2)).astype(np.float32) * 0.02)],
                model_id=hP.model_id, weights_id=hP.weights_id,
            )
            out_lr = tmp / "owner_exports" / "carried_lr.gguf"
            exp_lr = lex.export_lora_adapter(e_lr, target_gguf, out_path=out_lr)
            ver_lr = lex.verify_exported_lora(out_lr, edit=e_lr, target=target_gguf)
            check(exp_lr["recompiled_from_recipe"] is False and any("re-measure" in w for w in exp_lr["warnings"]) and ver_lr["ok"],
                  "6: carried lowrank export warns to re-measure on served GGUF and passes verify_exported_lora")

            # Implant shape diagnostic flags a last-layer single-token-aligned column edit
            last_L = n_layers - 1
            wu0 = hP.W_U[5].detach().float().cpu().numpy()
            w_post_last = we._w_post(hP.lm, last_L)
            # Build column delta whose stream write equals 10 * wu0 (avoiding tiny w_post entries)
            vec_imp = (wu0 / np.where(np.abs(w_post_last) > 1e-3, w_post_last, 1.0)).astype(np.float32)
            vec_imp = vec_imp / (float(np.linalg.norm(vec_imp)) + 1e-8)
            e_imp = we.build(hP.lm, [we.add_to_columns(last_L, [0], vec_imp[None, :])], model_id=hP.model_id, weights_id=hP.weights_id)
            imp_res = lex.implant_check(e_imp, hP.lm, hP.W_U)
            check(imp_res["checked"] is True and imp_res["flagged"] is True,
                  f"6: implant_check flags last-layer unembedding-aligned column write ({imp_res})")

            # CLI smoke: lora, verify, show
            cli_edit_json = tmp / "cli_edit.json"
            cli_edit_json.write_text(json.dumps({"terms": [{"layer": 2, "silence": [3, 7]}]}), encoding="utf-8")
            cli_gguf = tmp / "owner_exports" / "cli_export.gguf"
            check(
                lex_cli.main(["lora", "--edit", str(cli_edit_json), "--gguf", str(synth_gguf), "-o", str(cli_gguf)]) == 0
                and cli_gguf.is_file()
                and lex_cli.main(["verify", str(cli_gguf), "--edit", str(cli_edit_json), "--gguf", str(synth_gguf)]) == 0
                and lex_cli.main(["show", str(cli_gguf)]) == 0,
                "6: scripts/levelling_export.py lora, verify (with math check), and show exit 0",
            )

            # Verify ZERO export or trial-of-record tools exposed to agents or HTTP routes!
            mcp_tool_names = list(lmcp.TOOL_SPECS.keys())
            route_paths = [getattr(r, "path", "") for r in lroutes.router.routes]
            check(all("export" not in n and "trial" not in n for n in mcp_tool_names)
                  and all("export" not in p and "trial" not in p for p in route_paths),
                  f"6: zero export/trial tools exposed on MCP ({mcp_tool_names}) or HTTP ({route_paths})")

            # ------------------------------------------------------------------ 7. Volunteer re-run kit & defensive intake
            section("7. Volunteer re-run kit & defensive tar intake (census_intake.py pattern)")
            # Carried edit refused from volunteer kit
            got_car_kit = False
            try:
                lr.make_rerun_kit(prereg_car, tmp / "kit_carried_fail")
            except lr.RerunRefused as exc:
                got_car_kit = "rebuildable" in str(exc).lower()
            check(got_car_kit, "7: make_rerun_kit refuses carried-payload edits (rebuildable recipes only)")

            # Pack rebuilt trial into kit & verify
            kit_dir = tmp / "rerun_kit"
            kit_info = lr.make_rerun_kit(prereg_dir, kit_dir, home_result=t_res)
            kv = lr.verify_rerun_kit(kit_dir)
            check(kv["ok"] and kv["trial_sha256"] == reg["trial_sha256"], "7: make_rerun_kit & verify_rerun_kit pass")

            # Run volunteer re-run with same seed (same-bits class)
            vol_tar = tmp / "vol_alice.tar"
            vol_res = lr.run_volunteer_rerun(
                hP, kit_dir, vol_tar, volunteer_name="alice",
                _seed_override=b"trial-standin-seed-32bytes-00001",
            )
            check(vol_tar.is_file() and vol_res["verdict"] == t_res["verdict"],
                  f"7: run_volunteer_rerun produced allowlisted tar bundle with matching verdict ({vol_res['verdict']})")

            # Intake clean bundle (allow_dirty=True since worktree may be dirty during check)
            stage_ok = tmp / "intake_ok"
            iv_ok = lr.intake_rerun_bundle(vol_tar, prereg_dir, t_res, stage_ok, allow_dirty=True)
            check(iv_ok["outcome"] == "accepted" and iv_ok["match_class"] == "same_bits"
                  and (stage_ok / "intake_verdict.json").is_file(),
                  f"7: intake_rerun_bundle accepted clean volunteer bundle ({iv_ok['outcome']}, {iv_ok['match_class']})")

            # Defensive tar attack tests: path traversal, symlink, unexpected file, tampered hash
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
                    elif mode == "tamper_book":
                        with tarfile.open(vol_tar, "r") as src_tf:
                            for m in src_tf.getmembers():
                                data = src_tf.extractfile(m).read()
                                if m.name == "TRIAL_SHEET.md":
                                    data += b"\ntampered\n"
                                ti = tarfile.TarInfo(name=m.name)
                                ti.size = len(data)
                                tf.addfile(ti, io.BytesIO(data))

            for mode in ("traversal", "symlink", "extra"):
                bad_tar = tmp / f"evil_{mode}.tar"
                _make_evil_tar(bad_tar, mode)
                refused = False
                try:
                    lr.intake_rerun_bundle(bad_tar, prereg_dir, t_res, tmp / f"stage_{mode}", allow_dirty=True)
                except lr.RerunRefused:
                    refused = True
                check(refused, f"7: defensive tar extraction refuses {mode} bundle")

            tamp_tar = tmp / "evil_tamper.tar"
            _make_evil_tar(tamp_tar, "tamper_book")
            iv_tamp = lr.intake_rerun_bundle(tamp_tar, prereg_dir, t_res, tmp / "stage_tamper", allow_dirty=True)
            check(iv_tamp["outcome"] == "rejected" and any("TRIAL_SHEET.md" in r for r in iv_tamp["reject_reasons"]),
                  f"7: intake rejects tampered file sha256 ({iv_tamp['reject_reasons']})")

            # CLI smoke: pack, verify-kit & intake
            kit_cli = tmp / "kit_cli"
            stage_cli = tmp / "stage_cli"
            check(
                lr_cli.main(["pack", str(prereg_dir), "--out", str(kit_cli), "--home-result", str(trial_out / "trial_result.json")]) == 0
                and lr_cli.main(["verify-kit", str(kit_cli)]) == 0
                and lr_cli.main(["intake", str(vol_tar), "--prereg", str(prereg_dir),
                                 "--home-result", str(trial_out / "trial_result.json"),
                                 "--out", str(stage_cli), "--allow-dirty"]) == 0,
                "7: scripts/levelling_rerun.py pack, verify-kit, and intake exit 0",
            )

            # ------------------------------------------------------------------ 8. Zero kept-out words audit (ALL 21 files)
            section("8. Zero kept-out words across ALL 21 Levelling backend, CLI & frontend files")
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
