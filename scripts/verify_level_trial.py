#!/usr/bin/env python3
"""Card check of Levelling Phase 5 on the 31B (`scripts/verify_level_trial.py`): a trial read as a DRY RUN, an export,
and a re-run of the kit by this same card.

What it is not: a trial of record. A trial of record is read first in a process of its own, from a clean checkout
that holds its registration folder committed; this script calibrates before it reads, and registers into a temporary
folder. So it asks for a dry run, and first shows that the trial is refused without that word. No trial of record
has been read on the 31B by this script, whatever its dry run comes to.

On `google/gemma-4-31B-it-qat-w4a16-ct` (compiled unpacker, lens freed, server stopped), in this order:
1. `calibrate_bars` for `pronoun_pull` into a book of this run's own.
2. `register_trial` of the E6 layer-50 'she' ten (`[5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070]`, a
   silence; 39 control edits a family, 16 fresh items, the starter reworded set with 19 control edits).
3. `run_trial_of_record`: refused as it stands (the reasons are recorded), then read with `dry_run=True`. Checked: it
   is not void; the base checksum is unchanged over every stored weight; the folded form is the per-row form to the
   bit on every item in both classes (a silence is exact); the reworded set was read, with its folded form; the sheet
   says dry run and why; judged again from its book it gives its note's verdict; it is read once; the same edit,
   registered a second time before the trial was read, is refused at its read, and a registration asked after the
   trial is refused itself, with no folder and no look; the brief's history counts it as a fresh reading. How many
   matched control edits were as loud as the edit is counted again from the readings' own arms, and what the
   control edits moved the metric by, as read, is recorded beside what the judgement counted. Its verdict is
   recorded, not asked for: this brief's known answer has read more than one way across draws (ROADMAP §viii).
4. The export: the adapter of the trial's edit against the loaded model and, off the model, against this machine's
   cached checkpoint; each shown to be the edit on its base (`verify_exported_lora`), with what its keys say. The
   trial's verdict in them is read from the book and judged again with the registration.
5. The re-run: a kit of the dry run, read by this same card in this same process, and its intake from the two books,
   figure by figure. Equal figures here show that the kit carries the home items and that the intake reads books;
   they are not another card's. The intake is given leave for a changed checkout only when this checkout has a
   tracked file changed (the record says which). A bundle typed by hand is rejected.

Outputs (written at the end, so the checkout is not changed while the trial reads):
- `experiments/results/levelling/verify_trial.json`
- `experiments/results/levelling/verify_trial_book.jsonl`
"""
from __future__ import annotations

import datetime
import gc
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_grad_enabled(False)

import export_lora as el  # noqa: E402
from backend import engine_runtime  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_export as lex  # noqa: E402
from backend import level_rerun as lr  # noqa: E402
from backend import level_routes as lroutes  # noqa: E402
from backend import level_trial as lt  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend.census import write_json_atomic  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402

OUT_DIR = ROOT_DIR / "experiments" / "results" / "levelling"
OUT_JSON = OUT_DIR / "verify_trial.json"
OUT_BOOK = OUT_DIR / "verify_trial_book.jsonl"

E6_L50_SHE10 = [5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070]

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)


def _port_open(port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    if _port_open(8765):
        print("ERROR: Theodolite server answers on :8765; stop it before running verify_level_trial.py.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA unavailable.", file=sys.stderr)
        return 2
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)} ({free_gib:.2f} GiB free)", flush=True)
    if free_gib < 22.0:
        print(f"ERROR: Only {free_gib:.2f} GiB free; need >= 22.0 GiB.", file=sys.stderr)
        return 2

    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail is not None else ""), flush=True)

    t_all = time.perf_counter()
    print("Loading 31B W4A16 (lens freed, compiled unpacker)...", flush=True)
    t_load = time.perf_counter()
    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    gc.collect()
    torch.cuda.empty_cache()
    w4a16_fast.install(rt, "compiled", log=lambda s: print(f"  [unpacker] {s}", flush=True))
    fb0 = int(getattr(engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)
    h = lv.handle(rt)
    load_s = round(time.perf_counter() - t_load, 2)
    print(f"Loaded {h.model_id} in {load_s} s (weights {h.weights_id[:16]}).", flush=True)
    we.assert_base(h.lm, h.hooks_at_load)

    with tempfile.TemporaryDirectory(prefix="verify-level-trial-31b-") as td:
        tmp = Path(td)
        book = tmp / "verify_trial_book.jsonl"
        prev_edits, prev_exp = os.environ.get(we.EDITS_ENV), os.environ.get(lex.EXPORT_ENV)
        os.environ[we.EDITS_ENV] = str(tmp / "saved_edits")
        os.environ[lex.EXPORT_ENV] = str(tmp / "exports")

        # 1. the bars, from a calibration reading in this run's own book
        brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2)
        rset = lt.starter_reworded_pronoun_pull(brief)
        print("\nStep 1: calibrate_bars on pronoun_pull (layer 50, 10 cols, 3 controls)...", flush=True)
        t_cal = time.perf_counter()
        bars = lcl.calibrate_bars(
            h, brief, layer=50, n_cols=10, n_controls=3, n_neutral=4, n_near=4, n_panel=4,
            seed=2026, book_path=book,
        )
        cal_s = round(time.perf_counter() - t_cal, 2)
        cal_note = lcl.calibration_note(bars, book)
        check("bars_calibrated", cal_note is not None, {"seconds": cal_s, "bars_sha256": bars.short()})
        follows = lcl.calibration_follows(cal_note, book, brief, bars) if cal_note is not None else "no note"
        check("calibration_follows_from_its_readings_31b", follows is None,
              {"why_not": follows, "settings": (cal_note or {}).get("settings"), "readings": (cal_note or {}).get("book_seqs")})

        # 2. the registration
        edit = we.build(h.lm, [we.silence(50, E6_L50_SHE10)], model_id=h.model_id, weights_id=h.weights_id,
                        note="E6 layer-50 she ten")
        prereg_dir = tmp / "prereg_she10"
        reg_kw: Dict[str, Any] = dict(reworded=rset, book_path=book, controls_per_family=39, reworded_controls=19,
                                      n_target=16, n_neutral=4, n_near=4, n_panel=4)
        reg = lt.register_trial(prereg_dir, h, brief, bars, edit, trial_name="e6_l50_she10_31b", **reg_kw)
        v_reg = lt.verify_registration(prereg_dir)
        check("registration_verified",
              v_reg["ok"] and v_reg["trial"]["edit_provenance"] == "rebuilt" and v_reg["trial_sha256"] == reg["trial_sha256"],
              {"trial_sha256": reg["trial_sha256"][:16], "files": reg["files"],
               "reworded": reg["trial"]["reworded_origin"], "implant_check": reg["trial"]["implant_check"],
               "priced": reg["trial"]["priced"], "source_files_hashed": len(reg["trial"]["sources"])})
        # the look taken at registration: how many separate draws the matched control sets would count as, with the
        # base alone, on a sizing draw of the brief's frames and on the reworded set
        sz = reg["trial"]["sizing"]
        look = [e for e in lb.entries(book) if e["kind"] == "note" and e["body"].get("note_type") == "trial_look"][-1]
        check("the_look_at_registration_31b",
              all(isinstance(sz[r]["fewest_eligible"], int)
                  and sz[r]["counted_as"] == min(sz[r]["control_edits"], sz[r]["disjoint_sets"])
                  and sz[r]["could_pass"] is (sz[r]["all_drawn"] and sz[r]["counted_as"] >= sz[r]["needs"])
                  for r in ("matched", "reworded"))
              and (reg["warnings"] == []) is all(sz[r]["could_pass"] for r in ("matched", "reworded"))
              and look["seq"] == reg["trial"]["look_seq"] and look["body"]["sizing"] == sz
              and reg["rule_stands"] == lt.RULE_STANDS_SAY + "."
              and lt.RULE_STANDS_SAY in (prereg_dir / "RULES.md").read_text(),
              {"matched": sz["matched"], "reworded": sz["reworded"], "warnings": reg["warnings"],
               "its_note_is_book_line": look["seq"], "the_registration_also_says": reg["rule_stands"]})
        hand = lcl.make_bar_set(brief.id, near_miss_abs_max=2.0, near_miss_ratio_max=1.0)
        got_hand = None
        try:
            lt.register_trial(tmp / "prereg_hand_bars", h, brief, hand, edit, trial_name="hand", **reg_kw)
        except lt.TrialRefused as exc:
            got_hand = str(exc)
        check("bars_made_by_hand_refused", got_hand is not None and "calibration reading" in got_hand, (got_hand or "")[:160])
        # the same edit registered a second time now, while the book allows it: it is read after the first trial,
        # when the read is what refuses it (a registration asked after that trial is refused itself, further down)
        again = tmp / "prereg_again"
        lt.register_trial(again, h, brief, bars, edit, trial_name="e6_l50_she10_again", **reg_kw)

        # 3. the trial: refused as it stands, then read as a dry run
        why_not = None
        n_before = lb.verify(book)["entries"]
        try:
            lt.run_trial_of_record(h, prereg_dir, out_dir=tmp / "trial_refused", book_path=book)
        except lt.TrialRefused as exc:
            why_not = str(exc)
        check("refused_when_not_of_record",
              why_not is not None and "would not be of record" in why_not and lb.verify(book)["entries"] == n_before,
              (why_not or "")[:400])

        print("\nStep 2: the trial, as a dry run (16 fresh items, the reworded set, the folded form, 39 control edits a family)...",
              flush=True)
        t_tr = time.perf_counter()
        t_res = lt.run_trial_of_record(h, prereg_dir, out_dir=tmp / "trial_out", book_path=book, dry_run=True)
        tr_s = round(time.perf_counter() - t_tr, 2)
        print("\n" + t_res["sheet"] + "\n", flush=True)
        meta = t_res["meta"]
        check("trial_not_void", t_res["verdict"] in lt.PASSED + ("failed",),
              {"verdict": t_res["verdict"], "fresh_verdict": t_res["fresh_verdict"], "seconds": tr_s,
               "priced_seconds": reg["trial"]["priced"]["seconds"], "cheaper_knob": t_res["fresh_judged"].get("cheaper_knobs", {}).get("matched_by")})
        check("it_is_a_dry_run_and_says_so",
              t_res["of_record"] is False and t_res["claim_sentence"] is None
              and t_res["sheet"].startswith("=== DRY RUN OF A TRIAL (not of record)") and len(t_res["not_of_record"]) >= 1,
              {"not_of_record": t_res["not_of_record"]})
        check("base_checksum_unchanged_31b",
              meta["base_checksum_unchanged"] is True and meta["base_linears_checked"] >= 400
              and meta["base_tensors_checked"] > meta["base_linears_checked"],
              {"linears_checked": meta["base_linears_checked"], "tensors_checked": meta["base_tensors_checked"],
               "sha256": str(meta["base_checksum_before"])[:16]})
        fl = t_res["folded"]["classes"] if t_res["folded"].get("classes") else {}
        fl_sm, fl_lg = fl.get("small") or {}, fl.get("large") or {}
        check("folded_same_bits_31b",
              t_res["folded"]["available"] is True
              and fl_sm.get("same_bits_count") == fl_sm.get("taken_count") == 16
              and fl_lg.get("same_bits_count") == fl_lg.get("taken_count") == 16,
              {"small_same_bits": f"{fl_sm.get('same_bits_count')}/{fl_sm.get('taken_count')}",
               "large_same_bits": f"{fl_lg.get('same_bits_count')}/{fl_lg.get('taken_count')}",
               "small_mean": (fl_sm.get("frames") or {}).get("mean"), "large_mean": (fl_lg.get("frames") or {}).get("mean"),
               "tolerance": [fl_sm.get("tolerance"), fl_lg.get("tolerance")], "folded_ok": t_res["folded"]["ok"]})
        rw = t_res["reworded"] or {}
        rw_cls = rw.get("classes") or {}
        check("reworded_set_read_31b",
              bool(rw) and (rw.get("folded") or {}).get("available") is True and set(rw_cls) == {"small", "large"},
              {"ok": rw.get("ok"), "small_mean": ((rw_cls.get("small") or {}).get("frames") or {}).get("mean"),
               "large_mean": ((rw_cls.get("large") or {}).get("frames") or {}).get("mean"),
               "controls_small": ((rw_cls.get("small") or {}).get("controls") or {}).get("other_sites_matched")})
        check("sheet_and_rules_no_kept_out_words",
              KEPT_OUT_RE.search(t_res["sheet"]) is None
              and KEPT_OUT_RE.search((prereg_dir / "RULES.md").read_text()) is None,
              "0 kept-out words in the sheet and RULES.md")
        rj = lt.rejudge_trial(book, t_res["trial_sha256"], v_reg)
        check("judged_again_from_its_book", rj["as_noted"] and rj["verdict"] == t_res["verdict"],
              {"verdict": rj["verdict"], "noted": rj["noted_verdict"], "items": len(rj["items"]["target"].get("large", {}))})
        check("seed_in_the_result_not_in_the_book",
              len(t_res["seed_hex"]) == 64 and t_res["seed_hex"] not in book.read_text()
              and json.loads((tmp / "trial_out" / "trial_seed.json").read_text())["seed_hex"] == t_res["seed_hex"],
              "and kept in a file of its own from the draw on")
        check("unpacker_fallbacks_counted_31b", meta["unpacker_fallbacks"] == 0 and "unpacker fallbacks 0" in t_res["sheet"],
              {"unpacker_fallbacks": meta["unpacker_fallbacks"]})
        # the matched control edits as read: how many changed the stream within the registered factor of what the
        # edit changed it, over the answer and at the last position where this brief's metric is read (the rest are
        # counted as moving as much), how far those that count sat from the edit, and how many different stand-ins
        # a neuron had over the sets
        as_read = {}
        keys_dv = ("edit_prompt", "controls_median_prompt", "edit", "controls_median", "controls_min", "controls_max",
                   "edit_end", "controls_median_end", "read_at_end", "compared_at", "as_loud", "as_loud_ratio_prompt",
                   "as_loud_ratio_answer", "as_loud_ratio_end", "not_comparable", "of_controls", "fewest_eligible",
                   "fewest_distinct", "disjoint_sets", "draw")
        for cls_name in ("small", "large"):
            ce = t_res["fresh_judged"]["classes"].get(cls_name) or {}
            dv = (ce.get("delivered") or {}).get("other_sites_matched") or {}
            cs = (ce.get("controls") or {}).get("other_sites_matched") or {}
            as_read[cls_name] = {k: dv.get(k) for k in keys_dv}
            as_read[cls_name].update({"at_least": cs.get("at_least"), "effective_count": cs.get("effective_count"),
                                      "p_fw": cs.get("p_fw")})
            rw_dv = ((rw_cls.get(cls_name) or {}).get("delivered")) or {}
            rw_cs = ((rw_cls.get(cls_name) or {}).get("controls") or {}).get("other_sites_matched") or {}
            as_read[f"reworded_{cls_name}"] = {**{k: rw_dv.get(k) for k in keys_dv},
                                               "at_least": rw_cs.get("at_least"),
                                               "effective_count": rw_cs.get("effective_count"), "p_fw": rw_cs.get("p_fw")}
        places = {"prompt": "delivered_prompt", "answer": "delivered", "end": "delivered_end"}

        def loud_again(body: Dict[str, Any], names: Any, cls_name: str) -> Dict[str, Any]:
            """From a reading's own arms in the book: of its control edits, how many changed the stream at least as
            much as the edit (a mean over the items read) at each place, and at all three."""
            n_ctl = int(body["controls"]["count"])
            e_at: Dict[str, List[float]] = {p: [] for p in places}
            c_at: Dict[str, List[List[float]]] = {p: [[] for _ in range(n_ctl)] for p in places}
            for it in body["items"]:
                if it.get("status") != "ok" or it["item"]["name"] not in names:
                    continue
                arms = {a["key"]: a for a in it["classes"][cls_name]["arms"]}
                for p, key in places.items():
                    e_at[p].append(float(arms["edit:1.0"][key]))
                    for c in range(n_ctl):
                        c_at[p][c].append(float(arms[f"control:{c}"][key]))
            e_mean = {p: float(np.mean(v)) for p, v in e_at.items()}
            c_mean = {p: [float(np.mean(v)) for v in c_at[p]] for p in places}
            return {"as_loud_everywhere": sum(1 for c in range(n_ctl) if all(c_mean[p][c] >= e_mean[p] for p in places)),
                    "as_loud_by_place": {p: sum(1 for c in range(n_ctl) if c_mean[p][c] >= e_mean[p]) for p in places},
                    "median_over_the_edits": {p: round(float(np.median(c_mean[p])) / e_mean[p], 4) for p in places}}

        def body_of(role: str) -> Dict[str, Any]:
            return next(lb.entries(book, start=int(rj["readings"][role]["seq"]), limit=1))["body"]

        again_loud = {}
        for role, which, key in (("matched", "target", ""), ("reworded", "reworded", "reworded_")):
            names_now = {r["name"] for r in rj["drawn"][which]}
            for cls_name in ("small", "large"):
                again_loud[f"{key}{cls_name}"] = loud_again(body_of(role), names_now, cls_name)
        same_loud = {cls_name: loud_again(body_of("same_site"), {r["name"] for r in rj["drawn"]["target"]}, cls_name)
                     for cls_name in ("small", "large")}
        check("matched_controls_as_read_31b",
              all(isinstance(v["as_loud"], int) and v["as_loud"] == again_loud[k]["as_loud_everywhere"]
                  and v["not_comparable"] == v["of_controls"] - again_loud[k]["as_loud_everywhere"]
                  and v["at_least"] >= v["not_comparable"] and v["read_at_end"] is True and v["edit_end"] is not None
                  and v["edit_prompt"] is not None and v["compared_at"] == ["prompt", "answer", "end"]
                  and v["draw"] == "spread" and type(v["disjoint_sets"]) is int
                  and v["effective_count"] == min(v["of_controls"], v["disjoint_sets"])
                  for k, v in as_read.items())
              and as_read["small"]["of_controls"] == as_read["large"]["of_controls"] == 39,
              {"as_judged": as_read, "counted_again_from_the_readings_arms": again_loud,
               "the_same_site_family_which_is_not_asked_to_be_as_loud": same_loud})
        # what the control edits moved the metric by, as read: a family can fail because its control edits were set
        # aside as quieter than the edit, with none that was read moving the metric as much, and the record says which
        moved = {}
        for cls_name in ("small", "large"):
            cin = rj["judged"]["inputs"]["classes"][cls_name]
            e_mean = float(np.mean([e["value"] for e in cin["item_effects"]]))
            for fam in ("other_sites_matched", "same_site"):
                vals = [abs(float(v)) for v in cin["controls_as_read"][fam] if v is not None]
                cs_f = t_res["fresh_judged"]["classes"][cls_name]["controls"][fam]
                moved[f"{cls_name}:{fam}"] = {
                    "edit_mean": round(e_mean, 4), "control_edits_read": len(vals),
                    "largest_control_move": round(max(vals), 4), "median_control_move": round(float(np.median(vals)), 4),
                    "moved_as_much_as_read": sum(1 for v in vals if v >= abs(e_mean)),
                    "as_read_at_least": cs_f.get("as_read_at_least"), "at_least": cs_f.get("at_least"),
                    "set_aside_or_no_reading": cs_f.get("non_finite")}
        check("control_edits_as_read_moved_the_metric_31b",
              all(v["control_edits_read"] == 39 and v["at_least"] == v["as_read_at_least"] + v["set_aside_or_no_reading"]
                  and (v["as_read_at_least"] <= v["moved_as_much_as_read"]) for v in moved.values())
              and all(moved[f"{c}:same_site"]["as_read_at_least"] == moved[f"{c}:same_site"]["moved_as_much_as_read"]
                      for c in ("small", "large")), moved)
        # how many of the items each verdict rests on the kernel read under its item gate (the base gives their forced
        # answers under -6 nats a piece): said on the sheet, and recorded here
        gates = {"fresh": (t_res["fresh_judged"]["classes"].get("large") or {}).get("item_gate"),
                 "reworded": (rw_cls.get("large") or {}).get("item_gate"),
                 "batteries": {b: (v.get("items") or {}) for b, v in (t_res["fresh_judged"].get("batteries") or {}).items()
                               if isinstance(v, dict) and isinstance(v.get("items"), dict)}}
        check("items_under_the_item_gate_said_31b",
              isinstance(gates["fresh"], dict) and gates["fresh"]["items"] == 16 and isinstance(gates["reworded"], dict)
              and f"item gate: {gates['fresh']['under_bar']} of the 16 fresh items that read are under the kernel's bar"
                  in t_res["sheet"]
              and "reworded items that read are under the kernel's bar" in t_res["sheet"]
              and all(set(b) == {"read", "given", "under_item_gate"} and b["under_item_gate"] <= b["read"] <= b["given"]
                      for b in gates["batteries"].values()) and len(gates["batteries"]) == 3, gates)
        knobs = t_res["fresh_judged"].get("cheaper_knobs") or {}
        check("prompt_arms_beside_the_edit_31b", knobs.get("prompt_arms_state") == lcl.PROMPT_ARMS_RUN and bool(knobs.get("prompt_arms")),
              {"matched_by": knobs.get("matched_by"), "matched_arm": knobs.get("matched_arm"),
               "arms": [{k: a.get(k) for k in ("index", "directed_effect", "near_miss_abs")} for a in knobs.get("prompt_arms") or []],
               "edit_near_miss_abs": (t_res["fresh_judged"].get("batteries", {}).get("near_misses") or {}).get("abs_effect")})
        check("every_figure_of_the_judgement_31b",
              len(rj["figures"]) > 200 and all(v is not None for k, v in rj["figures"].items() if ":control:" in k),
              {"figures": len(rj["figures"]), "with_no_figure": sum(1 for v in rj["figures"].values() if v is None),
               "controls_small": sum(1 for k in rj["figures"] if k.startswith("target:small:control:")),
               "says": "each control edit's mean is its mean as read, whether or not the judgement set it aside"})

        def refused(fn: Any) -> str:
            try:
                fn()
            except (lt.TrialRefused, lcl.ClaimRefused) as exc:
                return str(exc)
            return ""

        ro = refused(lambda: lt.run_trial_of_record(h, prereg_dir, out_dir=tmp / "trial_out_dup", book_path=book, dry_run=True))
        check("trial_read_once_refused", "read once" in ro.lower(), ro[:120])
        ro2 = refused(lambda: lt.run_trial_of_record(h, again, out_dir=tmp / "trial_out_again", book_path=book, dry_run=True))
        check("same_edit_registered_again_refused", "already had a trial" in ro2, ro2[:160])
        n_late = lb.verify(book)["entries"]
        ro3 = refused(lambda: lt.register_trial(tmp / "prereg_late", h, brief, bars, edit, trial_name="e6_l50_she10_late",
                                                **reg_kw))
        check("registered_after_its_trial_refused_at_registration",
              "already had a trial" in ro3 and not (tmp / "prereg_late").exists() and lb.verify(book)["entries"] == n_late,
              {"says": ro3[:160], "folder_written": (tmp / "prereg_late").exists(),
               "lines_added_to_the_book": lb.verify(book)["entries"] - n_late})
        hist = lcl.brief_history(brief.id, book, name=brief.name)
        check("trial_counted_as_a_fresh_reading",
              hist["fresh_count"] == 1 and len(hist["trials"]) == 1 and meta["reading_index"] == 1,
              {"fresh_count": hist["fresh_count"], "of": brief.max_fresh_readings, "trials": lcl.prior_trials(hist)})

        v_ent = next(lb.entries(book, start=t_res["book"]["verdict_seq"], limit=1))
        card = lroutes.build_replay_card(v_ent)
        check("trial_replay_card_says_dry_run",
              card["note_type"] == lt.NOTE_DRY_RUN and card["status"] == f"dry run: {t_res['verdict']}"
              and card["edit_linears"] == ["layers.50.mlp.down_proj"],
              {"seq": card["seq"], "status": card["status"], "edit_linears": card["edit_linears"]})

        # 4. the export: against the loaded model, and off the model against the cached checkpoint
        print("\nStep 3: the export, against the loaded model and against the cached checkpoint...", flush=True)
        t_exp = time.perf_counter()
        target_31b = lex.TargetBase(lm=h.lm, label=f"{h.model_id} ({h.weights_id[:16]})", weights_id=h.weights_id)
        exp_out = lex.export_lora_adapter(
            edit, target_31b, out_path=tmp / "exports" / "e6_l50_she10_31b.gguf", dose=1.0, trial_result=t_res,
            registration=v_reg, lm_for_implant=h.lm, W_U_for_implant=h.W_U,
        )
        ver_exp = lex.verify_exported_lora(Path(exp_out["gguf_path"]), edit=edit, target=target_31b, dose=1.0)
        keys = el.read_lora(Path(exp_out["gguf_path"]))[1]
        said = {k.split("levelling.")[1]: keys[k] for k in sorted(keys) if k.startswith("theodolite.levelling.")
                and any(s in k for s in ("trial_verdict", "trial_fresh", "trial_reasons", "trial_kind", "trial_brief", "trial_judged",
                                         "applies_to", "implant", "compiled_against_is", ".book", ".gate", ".serving"))}
        exp_s = round(time.perf_counter() - t_exp, 2)
        check("export_is_the_edit_on_the_model",
              ver_exp["ok"] and (ver_exp["max_abs_err"] or 0.0) < 1e-4 and exp_out["recompiled_from_recipe"] is True
              and exp_out["compiled_against_is_edit_base"] is True,
              {"seconds": exp_s, "gguf_bytes": exp_out["gguf_bytes"], "max_abs_err": ver_exp["max_abs_err"]})
        check("export_keys_say_what_was_checked",
              said.get("trial_verdict") == t_res["verdict"] and said.get("trial_kind") == "a dry run"
              and said.get("trial_fresh_verdict") == t_res["fresh_verdict"] and said.get("trial_brief") == brief.name
              and said.get("trial_reasons") == (" | ".join(t_res["reasons"])[:1200] if t_res["reasons"]
                                                else "none recorded: every test was met")
              and str(said.get("trial_judged_again", "")).startswith("yes")
              and str(said.get("trial_applies_to_compiled_base", "")).startswith("yes")
              and str(said.get("compiled_against_is_edit_base", "")).startswith("yes: the same weights id")
              and str(said.get("gate", "")).startswith("built and gated")
              and "llama.cpp does not serve" in str(said.get("serving", ""))
              and (said.get("implant_flagged") in ("true", "false", "nothing in reach of the check")
                   or str(said.get("implant_flagged", "")).startswith("false for the part in reach of the check")), said)
        past_dose = ""
        try:
            lex.export_lora_adapter(edit, target_31b, out_path=tmp / "exports" / "dose_50.gguf", dose=50.0)
        except lex.ExportRefused as exc:
            past_dose = str(exc)
        check("export_refused_at_a_dose_the_kernel_would_not_read", "not exported at this dose" in past_dose
              and not (tmp / "exports" / "dose_50.gguf").exists(), past_dose[:200])
        said_more = {**t_res, "verdict": "passed", "of_record": True}
        out_more = lex.export_lora_adapter(edit, target_31b, out_path=tmp / "exports" / "said_more.gguf", trial_result=said_more,
                                           registration=v_reg)
        keys_more = el.read_lora(Path(out_more["gguf_path"]))[1]
        check("a_result_file_changed_by_hand_is_not_taken_at_its_word",
              keys_more["theodolite.levelling.trial_verdict"] == t_res["verdict"]
              and keys_more["theodolite.levelling.trial_kind"] == "a dry run",
              {"file_said": "passed, of record", "adapter_says": [keys_more["theodolite.levelling.trial_verdict"],
                                                                  keys_more["theodolite.levelling.trial_kind"]]})
        wrong = ""
        try:
            lex.export_lora_adapter(
                we.build(h.lm, [we.silence(50, E6_L50_SHE10[:5])], model_id=h.model_id, weights_id=h.weights_id), target_31b,
                out_path=tmp / "exports" / "another_edit.gguf", trial_result=t_res)
        except lex.ExportRefused as exc:
            wrong = str(exc)
        check("another_edit_does_not_take_this_trials_verdict", "not of the edit exported" in wrong, wrong[:140])
        off = {"ran": False}
        try:
            t_off = time.perf_counter()
            target_ck = lex.TargetBase.default_checkpoint()
            edit_off = lex.edit_from_registration(v_reg["edit_manifest"])
            out_off = lex.export_lora_adapter(edit_off, target_ck, out_path=tmp / "exports" / "e6_l50_she10_checkpoint.gguf",
                                              trial_result=t_res, registration=v_reg)
            ver_off = lex.verify_exported_lora(Path(out_off["gguf_path"]), edit=edit_off, target=target_ck)
            t_a, t_b = el.read_lora(Path(out_off["gguf_path"]))[0], el.read_lora(Path(exp_out["gguf_path"]))[0]
            same_file = set(t_a) == set(t_b) and all(
                np.array_equal(t_a[k][0], t_b[k][0]) and np.array_equal(t_a[k][1], t_b[k][1]) for k in t_a)
            keys_off = el.read_lora(Path(out_off["gguf_path"]))[1]
            off = {"ran": True, "ok": ver_off["ok"], "max_abs_err": ver_off["max_abs_err"], "max_rel_err": ver_off["max_rel_err"],
                   "edit_id_kept": edit_off.id == edit.id,
                   "base_is_edit_base": out_off["compiled_against_is_edit_base"],
                   "whole_base": out_off["manifest"]["compiled_against"]["whole_base_is_the_edits"],
                   "compiled_against_words": keys_off["theodolite.levelling.compiled_against_is_edit_base"],
                   "gate": keys_off["theodolite.levelling.gate"],
                   "implant": out_off["implant_check"].get("checked_at"),
                   "same_numbers_as_the_model_export": same_file, "seconds": round(time.perf_counter() - t_off, 2)}
        except lex.ExportRefused as exc:
            off = {"ran": False, "why": str(exc)[:200]}
        check("export_off_the_model_from_the_cached_checkpoint",
              off.get("ran") and off.get("ok") and off.get("edit_id_kept") and off.get("base_is_edit_base") is True
              and off.get("whole_base") is True and str(off.get("gate", "")).startswith("rebuilt off the model"), off)

        # 5. a kit of the dry run, read by this same card, and its intake from the two books
        print("\nStep 4: the kit, read by this same card, and the intake from the two books...", flush=True)
        t_rr = time.perf_counter()
        kit_dir = tmp / "kit_31b"
        lr.make_rerun_kit(prereg_dir, kit_dir, t_res)
        vol_tar = tmp / "vol_31b.tar"
        vol = lr.run_volunteer_rerun(h, kit_dir, vol_tar, volunteer_name="the_home_card_itself", work_dir=tmp / "vol_work")
        # leave for a changed checkout is given only when this checkout has a tracked file changed, and the record says
        tracked_now = lt.checkout_changes(ROOT_DIR, tracked_only=True)
        leave = (bool(tracked_now) or tracked_now is None or bool(lt.hidden_from_status(ROOT_DIR))
                 or lt.unlisted_code(ROOT_DIR) != [])
        intake_res = lr.intake_rerun_bundle(vol_tar, prereg_dir, book, tmp / "intake_stage", allow_dirty=leave)
        rr_s = round(time.perf_counter() - t_rr, 2)
        check("same_card_rerun_reads_the_home_items",
              intake_res["outcome"] == "accepted" and intake_res["match_class"] == "same_figures"
              and intake_res["class_diffs"]["target:large"]["items"] == 16
              and intake_res["class_diffs"]["reworded:large"]["items"] == 8
              and intake_res["class_diffs"]["target:large:controls"]["items"] == 78
              and intake_res["figures_compared"] + intake_res["figures_neither_trial_has"] == len(rj["figures"])
              and intake_res["control_edits_set_aside_differently"] == {},
              {"seconds": rr_s, "outcome": intake_res["outcome"], "match_class": intake_res["match_class"],
               "figures_compared": intake_res["figures_compared"],
               "figures_neither_trial_has": intake_res["figures_neither_trial_has"],
               "control_edits_set_aside_differently": intake_res["control_edits_set_aside_differently"],
               "leave_given_for_a_changed_checkout": leave,
               "with_leave_for": intake_res["with_leave_for"],
               "verdict": vol["verdict"], "class_diffs": intake_res["class_diffs"],
               "what_this_is": "this card re-running its own kit in this process: not another card",
               "reject_reasons": intake_res["reject_reasons"][:3], "disagree_reasons": intake_res["disagree_reasons"][:3]})
        typed = tmp / "typed"
        typed.mkdir()
        lb.append("note", {"note_type": "hello", "text": "no reading was taken"}, path=typed / "volunteer_book.jsonl")
        (typed / "return.json").write_text(json.dumps({"v": 1, "volunteer": "nobody", "verdict": t_res["verdict"]}))
        (typed / "TRIAL_SHEET.md").write_text("no sheet\n")
        (typed / "manifest.json").write_text(json.dumps({"v": 1, "files": {
            n: {"sha256": hashlib.sha256((typed / n).read_bytes()).hexdigest()}
            for n in ("return.json", "volunteer_book.jsonl", "TRIAL_SHEET.md")}}))
        with tarfile.open(tmp / "typed.tar", "w") as tf:
            for n in lr.PACKED_FILES:
                tf.add(typed / n, arcname=n)
        iv_typed = lr.intake_rerun_bundle(tmp / "typed.tar", prereg_dir, book, tmp / "intake_typed")
        check("bundle_typed_by_hand_rejected", iv_typed["outcome"] == "rejected", iv_typed["reject_reasons"][:1])

        # 6. the base, the unpacker, the book
        base_ok = True
        try:
            we.assert_base(h.lm, h.hooks_at_load)
        except Exception as exc:  # noqa: BLE001
            base_ok = False
            print(f"  assert_base: {exc}", file=sys.stderr)
        check("assert_base_final", base_ok, "the model is the base again")
        fb1 = int(getattr(engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)
        fb = fb1 - fb0
        check("zero_unpacker_fallbacks", fb == 0, {"fallback_count": fb})
        bv = lb.verify(book)
        check("ledger_whole", bv["ok"], {"entries": bv["entries"], "head": bv["head"][:16]})
        peaks = [float(e["body"]["peak_gib"]) for e in lb.entries(book)
                 if e["kind"] == "paired" and isinstance(e["body"].get("peak_gib"), (int, float))]
        peak_gib = round(max(peaks), 3) if peaks else None
        total_s = round(time.perf_counter() - t_all, 2)
        check("peak_memory_recorded", peak_gib is not None and peak_gib > 0.0,
              {"peak_gib": peak_gib, "of": "the largest peak any reading in the book recorded", "total_seconds": total_s})

        n_ok = sum(1 for c in checks if c["ok"])
        n_fail = len(checks) - n_ok
        report = {
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": "measured on the 31B card",
            "what_this_is": ("a DRY RUN of a trial, an export, and a re-run of the kit by this same card in this same "
                             "process. No trial of record has been read by this script, and no other card has run a kit."),
            "gpu": torch.cuda.get_device_name(0),
            "model_id": h.model_id,
            "weights_id": h.weights_id,
            "load_seconds": load_s,
            "total_seconds": total_s,
            "peak_gib": peak_gib,
            "fallback_count": fb,
            "passed": n_ok,
            "failed": n_fail,
            "checks": checks,
            "registration_look": reg["trial"]["sizing"],
            "trial_e6_she10": {
                "seconds": tr_s,
                "of_record": t_res["of_record"],
                "not_of_record": t_res["not_of_record"],
                "verdict": t_res["verdict"],
                "fresh_verdict": t_res["fresh_verdict"],
                "reasons": t_res["reasons"],
                "sheet": t_res["sheet"],
                "folded": t_res["folded"],
                "reworded": t_res["reworded"],
                "meta": meta,
                "book": {k: v for k, v in t_res["book"].items() if k != "path"},
            },
            "export_check": {"seconds": exp_s, "max_abs_err": ver_exp["max_abs_err"], "gguf_bytes": exp_out["gguf_bytes"],
                             "implant_check": exp_out["implant_check"], "keys": said, "off_the_model": off},
            "rerun_intake": {"seconds": rr_s, "outcome": intake_res["outcome"], "match_class": intake_res["match_class"],
                             "figures_compared": intake_res["figures_compared"],
                             "figures_neither_trial_has": intake_res["figures_neither_trial_has"],
                             "leave_given_for_a_changed_checkout": leave,
                             "class_diffs": intake_res["class_diffs"],
                             "what_this_is": "this card re-running its own kit in this process: not another card"},
            "ledger": bv,
        }
        # written last, so the checkout is as it was while the trial read
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(book, OUT_BOOK)
        write_json_atomic(OUT_JSON, report)
        for k, prev in ((we.EDITS_ENV, prev_edits), (lex.EXPORT_ENV, prev_exp)):
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev

    print(f"\nPhase 5 card check complete in {total_s} s (peak {peak_gib} GiB): "
          f"{n_ok}/{len(checks)} passed, {n_fail} failed [measured on the 31B card; a dry run, not a trial of record].",
          flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
