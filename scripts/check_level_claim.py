#!/usr/bin/env python3
"""CPU verification suite for Levelling Phase 3: The Claim Gate (`scripts/check_level_claim.py`).

Tests all Phase 3 contracts on CPU stand-ins with zero GPU access:
1. Versioned briefs, canonical hashing, save/load tamper detection, policy refusal, and FrameGenerator disjointness /
   half balancing / anti-leakage refusal.
2. Statistical primitives: `house_bar` (sqrt(F*M), M < 4F drop), `frame_statistics` (frame-level SE/CI95/sign-flip/halves),
   `control_standing` (p_emp, p_fw, C = 20m - 1 boundary), `written_strata_summary` (Row 21), and `price_written_reading`.
3. Pure `judge_claim` planted-failure suite: true lead, overfit, floor-too-loud, half-only artifact, reversed-dose
   failure, control-beaten null (each family & batch class), harm on each gating battery, cheaper-knob prompt match,
   void on missing class/frames, and `free_pick` vs `declared_rule` wording.
4. 200 planted nulls calibration test (`arbiter.md` Row 24) — both realistic joint-gate and worst-case exchangeable nulls.
5. Live CPU stand-in end-to-end: `calibrate_bars` (hash match + ledger whole), `record_practice_draw`, pre-registration
   leakage refusal, `run_fresh_claim` #1 & #2 (ordering, read-once refusal, m-budget cap, prior-verdicts history on sheet,
   redacted `driver_record`), and live `written_loop` claim with chunked controls + `written_strata` + `settings_audit`.
6. Phase 2 `LevelSession` + MCP (`brief_info`, `draw_practice`, `claim`, blinded exclusion) and CLI (`brief`,
   `calibrate`, `claim --stand-in`) integration.
7. Zero kept-out words audit across all 10 Levelling source files plus every rendered Claim Sheet and claim sentence.

Run:
    /home/eric/bin/ai/theodolite2/.venv/bin/python -I scripts/check_level_claim.py
"""
from __future__ import annotations

import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

torch.set_num_threads(1)

from atlas_stand_in import stand_in, stand_in_tokenizer, use_stand_in_atlas  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_mcp as lmcp  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import weight_edits as we  # noqa: E402

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)

PASSED = 0
FAILED = 0
SHEETS_COLLECTED: List[str] = []


def check(cond: bool, msg: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}", flush=True)


def check_raises(exc_type: Any, fn: Any, msg: str, match: Optional[str] = None) -> None:
    global PASSED, FAILED
    try:
        fn()
    except exc_type as exc:
        if match is not None and match.lower() not in str(exc).lower():
            FAILED += 1
            print(f"  [FAIL] {msg}: raised {exc!r} which did not match {match!r}", flush=True)
        else:
            PASSED += 1
        return
    except Exception as exc:
        FAILED += 1
        print(f"  [FAIL] {msg}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}", flush=True)
        return
    FAILED += 1
    print(f"  [FAIL] {msg}: did not raise {exc_type.__name__}", flush=True)


def _synth_class(
    *,
    h0_vals: List[float],
    h1_vals: List[float],
    floor: Optional[float] = 0.01,
    reversed_mean: Optional[float] = 0.05,
    osm_ctls: Optional[List[float]] = None,
    ss_ctls: Optional[List[float]] = None,
    sham_ctls: Optional[List[float]] = None,
) -> Dict[str, Any]:
    items = []
    for i, v in enumerate(h0_vals):
        items.append({"frame_id": f"k0_{i}", "half": 0, "value": float(v)})
    for i, v in enumerate(h1_vals):
        items.append({"frame_id": f"k1_{i}", "half": 1, "value": float(v)})
    ctls: Dict[str, Any] = {
        "other_sites_matched": list(osm_ctls) if osm_ctls is not None else [0.05] * 39,
        "same_site": list(ss_ctls) if ss_ctls is not None else [0.05] * 39,
    }
    if sham_ctls is not None:
        ctls["sham_target"] = list(sham_ctls)
    return {"item_effects": items, "floor": floor, "reversed_mean": reversed_mean, "controls": ctls}


def _clean_batteries() -> Dict[str, Dict[str, Any]]:
    return {
        "neutral_text": {"mean_kl": 0.01, "control_kl_max": 0.02},
        "near_misses": {"mean_effect": -0.05, "abs_effect": 0.05},
        "labelled_panel": {"own_drop": 0.01, "mean_kl": 0.01},
        "habitat": {"available": False},
    }


def _meta_stub(brief: lbr.Brief, bars: lcl.BarSet, *, reading_index: int = 1) -> Dict[str, Any]:
    return {
        "date": "2026-11-18T23:55:00Z",
        "model_id": "stand-in@0000",
        "weights_id": "a" * 64,
        "unpacker": "plain",
        "edit_id": "e" * 64,
        "edit_linears": ["layers.2.mlp.down_proj"],
        "main_dose": 1.0,
        "brief_id": brief.id,
        "brief_name": brief.name,
        "bars_sha256": bars.id,
        "reading_index": reading_index,
        "max_fresh_readings": brief.max_fresh_readings,
        "prior_verdicts": [],
        "prompt_named_site": False,
        "chat_settings": dict(brief.chat_settings),
    }


def run_section_1(tok: Any, tmp: Path) -> None:
    print("Section 1: Briefs, hashing, tamper detection, policy refusal, FrameGenerator anti-leakage", flush=True)
    starters = lbr.starter_briefs(tok)
    check(
        set(starters.keys()) == {"pronoun_pull", "spider_legs_bleed", "canary_null", "enumeration_loop"},
        f"all 4 starter briefs present: {sorted(starters.keys())}",
    )
    for name, b in starters.items():
        check(len(b.id) == 64, f"{name} canonical brief_id is 64-char sha256")
        check(b.default_controls == 20 * b.max_fresh_readings - 1, f"{name} default_controls == 20m - 1 (39)")
        check(len(b.generator.frames_for("kept_back", half=0)) >= 2, f"{name} >= 2 kept-back frames in half 0")
        check(len(b.generator.frames_for("kept_back", half=1)) >= 2, f"{name} >= 2 kept-back frames in half 1")
        check(len(b.generator.frames_for("practice")) >= 2, f"{name} >= 2 practice frames")

    b5 = lbr.pronoun_pull_brief(tok, max_fresh_readings=5)
    check(b5.default_controls == 99, "m=5 defaults to 99 controls per family")

    # Save/load + tamper detection
    bp = tmp / "pronoun.brief.json"
    saved_id = lbr.save_brief(starters["pronoun_pull"], bp)
    reloaded = lbr.load_brief(bp)
    check(reloaded.id == saved_id == starters["pronoun_pull"].id, "save_brief/load_brief round-trips canonical id")
    tampered = json.loads(bp.read_text(encoding="utf-8"))
    tampered["least_effect"] = 0.01
    tp = tmp / "tampered.brief.json"
    tp.write_text(json.dumps(tampered), encoding="utf-8")
    check_raises(lbr.BriefError, lambda: lbr.load_brief(tp), "load_brief refuses tampered brief", match="hashes to")

    # Policy refusals (arbiter.md Decision 5 / Row 37)
    base_b = starters["canary_null"]
    for bad_txt in ("bypass safeguard on prompts", "remove refusal completely", "implant trigger token", "jailbreak mode"):
        check_raises(
            lbr.BriefError,
            lambda bt=bad_txt: lbr.Brief(
                name="bad", behavior=bt, metric="odds", direction=1, least_effect=0.5,
                readouts=base_b.readouts, generator=base_b.generator, near_miss_generator=base_b.near_miss_generator,
            ),
            f"policy refuses forbidden brief text {bad_txt!r}",
            match="policy refusal",
        )

    # Generator disjointness & half requirements
    check_raises(
        lbr.BriefError,
        lambda: lbr.FrameGenerator(
            "leak_gen",
            [lbr.Frame("f1", "practice", 0, "Say {x}", "ok"), lbr.Frame("f2", "kept_back", 0, "Say {x}", "ok")],
            {"x": ["a", "b"]},
        ),
        "FrameGenerator refuses overlapping template between practice and kept_back",
        match="shares a frame template",
    )
    check_raises(
        lbr.BriefError,
        lambda: lbr.FrameGenerator(
            "missing_slot", [lbr.Frame("f1", "practice", 0, "Say {missing}", "ok")], {"x": ["a"]},
        ),
        "FrameGenerator refuses missing slot",
        match="missing from slots",
    )

    # Balanced kept-back draws & practice collision refusal
    b_pp = starters["pronoun_pull"]
    kb_drawn = b_pp.generator.draw("kept_back", 8, seed_bytes=b"0123456789abcdef", tokenizer=tok)
    check(len(kb_drawn) == 8, "kept_back draw returns requested 8 items")
    check(sum(1 for d in kb_drawn if d.half == 0) == 4, "kept_back draw balances half 0 (4/8)")
    check(sum(1 for d in kb_drawn if d.half == 1) == 4, "kept_back draw balances half 1 (4/8)")
    check(len({d.item.id for d in kb_drawn}) == 8, "kept_back draw produces 8 distinct items")

    # Refusal on forbidden item id and forbidden frame id
    check_raises(
        lbr.BriefError,
        lambda: b_pp.generator.draw(
            "kept_back", 8, seed_bytes=b"0123456789abcdef", tokenizer=tok,
            forbid_item_ids={kb_drawn[0].item.id},
        ),
        "FrameGenerator refuses fresh item colliding with forbid_item_ids",
        match="leakage refused",
    )
    check_raises(
        lbr.BriefError,
        lambda: b_pp.generator.draw(
            "kept_back", 8, seed_bytes=b"0123456789abcdef", tokenizer=tok,
            forbid_frame_ids={kb_drawn[0].frame_id},
        ),
        "FrameGenerator refuses kept_back frame colliding with forbid_frame_ids",
        match="leakage refused",
    )

    # Battery generators
    neu = lbr.neutral_chat_generator().draw("kept_back", 4, seed_bytes=b"neutral_seed_1234", tokenizer=tok)
    pan = lbr.draw_labelled_sanity_panel(4, seed_bytes=b"panel_seed_123456", tokenizer=tok)
    check(len(neu) == 4 and all(isinstance(d.item, lv.Item) for d in neu), "neutral_chat_generator draws valid Items")
    check(len(pan) == 4 and all(isinstance(p["item"], lv.Item) for p in pan), "draw_labelled_sanity_panel draws valid Items")

    # A written brief names the loop rule its loops are counted by; one that names none keeps the instrument's own
    enum = starters["enumeration_loop"]
    check(lbr.loop_rule_of(enum) == "either" and lbr.loop_rule_of(starters["pronoun_pull"]) == "instrument"
          and "has not been seen on this checkpoint" in enum.behavior,
          "the enumeration brief counts loops by either rule and says its fault has not been seen on this checkpoint")
    check_raises(
        lbr.BriefError,
        lambda: lbr.Brief.from_dict({**enum.to_dict(), "written_config": {**enum.written_config, "loop_rule": "a person"}}),
        "a brief is refused a loop rule the kernel does not have",
        match="loop_rule must be one of",
    )

    # Kept-back frames at the door (KeptBack): every rendering of a kept-back frame is found, no practice one is
    guard = lbr.KeptBack(starters.values())
    kb_missed: List[str] = []
    prac_caught: List[str] = []
    for b in starters.values():
        for gen in (b.generator, b.near_miss_generator):
            n_kb = len(gen.frames_for("kept_back"))
            for d in gen.draw("kept_back", 2 * n_kb, seed_bytes=b"kept-back-check", tokenizer=tok):
                if guard.hit(d.item.question) is None:
                    kb_missed.append(f"{b.name}/{d.frame_id}")
            if gen.frames_for("practice"):
                for d in gen.draw("practice", 12, seed_bytes=b"practice-check", tokenizer=tok):
                    if guard.hit(d.item.question, d.item.system, d.item.answer) is not None:
                        prac_caught.append(f"{b.name}/{d.frame_id}")
    check(len(guard) >= 24 and not kb_missed,
          f"every rendering of every starter brief's kept-back and near-miss frames is found (missed: {kb_missed})")
    check(not prac_caught, f"no rendering of a practice frame is taken for a kept-back one (caught: {prac_caught})")
    kq = starters["pronoun_pull"].generator.draw("kept_back", 2, seed_bytes=b"wrapped-check", tokenizer=tok)[0].item.question
    got_plain = guard.hit(kq)
    check(got_plain is not None and got_plain["brief"] == "pronoun_pull"
          and guard.hit("Please do this.   " + kq.upper() + "\n\nThank you.") == got_plain
          and guard.hit(None, "An ordinary question about rivers?", kq) == got_plain,
          "a kept-back frame is found inside a longer text, in any of an item's texts, whatever its case and spacing")
    check(guard.hit("An ordinary question about rivers?", None, " she") is None,
          "an ordinary item is not taken for a kept-back frame")
    f_a = lbr.Frame("a", "kept_back", 0, "State the  leg count of a typical {kind} living near {place}.", "x")
    f_b = lbr.Frame("b", "practice", 1, "state the leg count of a typical {animal} living near {where}.", "y")
    check(f_a.template_id == f_b.template_id and f_a.id != f_b.id
          and f_a.template_id in lbr.published_template_ids(tok),
          "a frame's template_id is one for a template under other ids, slot names, case and spacing")
    f_short = lbr.Frame("s", "kept_back", 0, "{a} or {b}?", "x")
    long_around = ("I went out early and bought bread and milk and then, would you believe it, tea or coffee was all "
                   "they had left in the whole shop that morning, so I took both of them home with me.")
    check(f_short.rendered_in("Tea or coffee?") and not f_short.rendered_in(long_around),
          "a template with few fixed words must be the whole text, each slot a few words, to count as rendered")

    # What the guard goes by: words. Punctuation, case, spacing and characters that print nothing do not hide a
    # frame, nor does moving part of it into another field; a near wording of a frame with long fixed runs is found.
    spider_q = "State the leg count of a typical wolf spider living near a shed."
    dodges = {
        "its comma and full stop dropped": spider_q.replace(".", ""),
        "a zero-width space inside a word": spider_q[:8] + "\u200b" + spider_q[8:],
        "a soft hyphen and a NUL": spider_q[:14] + "\u00ad" + spider_q[14:] + "\x00",
        "fullwidth letters": spider_q.replace("State", "\uff33\uff54\uff41\uff54\uff45"),
        "other case and spacing": "  STATE the   leg COUNT of a typical wolf spider\nliving near a shed  ",
    }
    for how, text in dodges.items():
        got = guard.hit(text)
        check(got is not None and got["brief"] == "spider_legs_bleed" and got["how"] == "wording",
              f"a kept-back frame is found with {how} ({got})")
    got_split = guard.hit("State the leg count of", "a typical wolf spider living near a shed.", "It typically walks on")
    check(got_split is not None and got_split["how"] == "wording",
          "a kept-back frame split between the system turn and the question is found in the two run together")
    for how, text in {
        "one word changed": "State the leg number of a typical wolf spider living near a shed.",
        "its opening words dropped": "the leg count of a typical wolf spider living near a shed, please",
    }.items():
        got = guard.hit(text)
        check(got is not None and got["how"] == "near wording", f"a kept-back frame with {how} is found as a near wording ({got})")
    check(guard.hit("How many legs would you say a wolf spider near a shed walks on?") is None
          and guard.hit(spider_q.replace("a", "\u0430")) is None,
          "what the guard does not find, and says it does not: a question put in other words, and look-alike letters")
    check(guard.hit("Why do hikers pack extra water when crossing ridges?") is None
          and lbr.KeptBack(starters.values(), batteries=True).hit("why do hikers pack extra water when crossing ridges")
          == {"brief": lbr.BATTERIES, "frame": "neu_05", "how": "wording"}
          and lbr.KeptBack(starters.values(), batteries=True).hit("Compute 12 + 30. Reply with the number.") is not None
          and lbr.KeptBack(starters.values(), batteries=True).hit("What is the capital of France") is not None,
          "with batteries, the guard finds the retention batteries' own questions too")
    slow = lbr.Frame("slow", "kept_back", 0, "{a} {b} {c} every single morning before work.", "x")
    long_text = " ".join(f"word{i}" for i in range(4000))
    t_guard = time.perf_counter()
    slow_hit = slow.rendered_in(long_text) or slow.near_in(long_text)
    check(not slow_hit and time.perf_counter() - t_guard < 1.0,
          f"a template opening with slots is looked for in a 4,000-word text in well under a second "
          f"({time.perf_counter() - t_guard:.3f} s)")
    f_braces = lbr.Frame("j", "kept_back", 0, 'Fill in this record: {{"city": "{city}", "country": ""}} and say why.', "x")
    check(f_braces.rendered_in(f_braces.question_tpl.format(city="Oslo")),
          "a template with literal braces finds its own rendering")
    common = lbr.Frame("c", "kept_back", 0, "the {x} and a {y} in the {z}", "x")
    check(not common.rendered_in("The committee met in March. " + "Nothing was decided that week. " * 20
                                 + "Then came rain and a flood, and nobody stayed in the valley.", slot_words=6),
          "a template of common words is not found across a long text: a slot stands for a few words, not a paragraph")

    # One question template is one frame everywhere: in a generator's two splits and between a brief's practice
    # frames and its near-miss frames, whatever the answer after it and whatever its slots are called
    check_raises(
        lbr.BriefError,
        lambda: lbr.FrameGenerator("g", [
            lbr.Frame("p1", "practice", 0, "Name the capital of {city}.", "It is"),
            lbr.Frame("k1", "kept_back", 0, "Name the capital of {town}.", "The capital is"),
            lbr.Frame("k2", "kept_back", 1, "Which river runs through {town}?", "It is"),
        ], {"city": ["Oslo"], "town": ["Bergen"]}),
        "a generator is refused a question template in both splits, under other slot names and another answer",
        match="shares a frame template",
    )
    pp = starters["pronoun_pull"]
    prac0 = pp.generator.frames_for("practice")[0].to_dict()  # the same question, its {name} slot called {mname}
    nm_clash = lbr.FrameGenerator.from_dict({**pp.near_miss_generator.to_dict(), "frames": [
        *pp.near_miss_generator.to_dict()["frames"],
        {**prac0, "id": "nm_k99", "split": "kept_back",
         "question_tpl": prac0["question_tpl"].replace("{name}", "{mname}"),
         "answer_tpl": prac0["answer_tpl"].replace("{name}", "{mname}")}]})
    check_raises(
        lbr.BriefError,
        lambda: lbr.Brief.from_dict({**pp.to_dict(), "near_miss_generator": nm_clash.to_dict()}),
        "a brief is refused a practice frame whose question is one of its near-miss frames'",
        match="practice gates would refuse the brief's own practice items",
    )
    check_raises(lbr.BriefError, lambda: lbr.Brief.from_dict({**pp.to_dict(), "name": "pronoun_pull "}),
                 "a brief is refused a name with a space at its end", match="no space before or after")


def run_section_2() -> None:
    print("Section 2: Statistics (house_bar, frame_statistics, control_standing, written strata, pricing)", flush=True)
    # the strata count loops by the reading's own rule, so they and the verdict speak of the same loops
    by_rule_res = {"summary": {
        "base": {"n": 8, "ended": 6, "cut_off": 2, "looped": 0, "looped_published": 3, "looped_either": 3,
                 "loop_rules_differ": 3},
        "arms": {"edit:1.0": {"n": 8, "ended": 7, "cut_off": 1, "looped": 0, "looped_published": 1,
                              "looped_either": 1, "loop_rules_differ": 1}}}}
    st_either = lcl.written_strata_summary(by_rule_res, rule="either")
    st_own = lcl.written_strata_summary(by_rule_res, rule="instrument")
    check(st_either["loop_rule"] == "either" and st_either["strata"]["looped"]["base_count"] == 3
          and st_either["strata"]["looped"]["edit_count"] == 1 and st_own["strata"]["looped"]["base_count"] == 0,
          "the looped stratum is counted by the reading's loop rule, not always by the instrument's")
    check(lcl._written_rows(5, 3) == (8, 1) and lcl._written_rows(12, 3) == (12, 5) and lcl._written_rows(99, 2)[0] == lv.WRITTEN_ROWS_MAX,
          "a written claim's batch has room for the base, the untouched row, the nulls, each dose and a control, whatever rows were asked")
    sig = inspect.signature(lcl.run_fresh_claim).parameters
    check(all(sig[k].default == v for k, v in lcl.DESIGN_DEFAULTS.items())
          and all(inspect.signature(lcl.estimate_claim).parameters[k].default == v for k, v in lcl.DESIGN_DEFAULTS.items()),
          "DESIGN_DEFAULTS, which a session shows on its card, are run_fresh_claim's and estimate_claim's own defaults")
    # which book is the default one: $XDG_DATA_HOME places it, $THEODOLITE_LEVEL_BOOK moves the book but not the default
    kept_env = {k: os.environ.get(k) for k in ("XDG_DATA_HOME", lb.BOOK_ENV)}
    try:
        os.environ["XDG_DATA_HOME"] = "/nowhere/xdg"
        os.environ[lb.BOOK_ENV] = "/nowhere/elsewhere.jsonl"
        check(lb.default_book_path() == Path("/nowhere/xdg/theodolite/level_book.jsonl")
              and lb.book_path() == Path("/nowhere/elsewhere.jsonl"),
              "default_book_path is where the book lives when nothing moves it; book_path still follows the environment")
        os.environ.pop(lb.BOOK_ENV)
        check(lb.book_path() == lb.default_book_path(), "with nothing moving it, the book is the default book")
    finally:
        for k, v in kept_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # house_bar (arbiter.md Row 4)
    hb1 = lcl.house_bar(0.01, 0.5)
    check(not hb1["floor_too_loud"] and math.isclose(hb1["bar"], math.sqrt(0.005), rel_tol=1e-9), "house_bar = sqrt(F*M)")
    hb2 = lcl.house_bar(0.15, 0.5)  # M < 4F (0.5 < 0.6)
    check(hb2["floor_too_loud"] and hb2["bar"] is None, "house_bar drops bar when M < 4*F")
    hb3 = lcl.house_bar(0.0, 0.5)
    check(not hb3["floor_too_loud"] and hb3["bar"] == 0.5, "house_bar returns M when F == 0")
    hb4 = lcl.house_bar(None, 0.25)
    check(not hb4["floor_too_loud"] and hb4["bar"] == 0.25, "house_bar returns M when F is None")

    # frame_statistics: collapses items inside each frame before CI / sign-flip
    rows = [
        {"frame_id": "f0", "half": 0, "value": -1.0},
        {"frame_id": "f0", "half": 0, "value": -1.4},  # frame mean -1.2
        {"frame_id": "f1", "half": 0, "value": -0.8},
        {"frame_id": "f2", "half": 1, "value": -1.1},
        {"frame_id": "f3", "half": 1, "value": -0.9},
    ]
    fs = lcl.frame_statistics(rows, direction=-1)
    check(fs["n_items"] == 5 and fs["n_frames"] == 4, "frame_statistics aggregates 5 items into 4 frames")
    check(math.isclose(fs["mean"], -1.0, rel_tol=1e-9), f"frame mean == -1.0 (got {fs['mean']})")
    check(math.isclose(fs["directed_mean"], 1.0, rel_tol=1e-9), "directed_mean flips sign for direction=-1")
    check(fs["half_0"]["n_frames"] == 2 and fs["half_1"]["n_frames"] == 2, "half counts are 2 and 2")
    check(fs["sign_flip"]["exact"] and math.isclose(fs["sign_flip"]["one_sided"], 1.0 / 16.0), "exact 2^4 sign-flip p")
    check(fs["ci95"][0] < fs["mean"] < fs["ci95"][1], "95% frame CI brackets frame mean")

    # control_standing (Row 18 & Row 23: tops_every_control never gates alone!)
    cs_pass = lcl.control_standing(-1.0, [0.2] * 39, max_fresh_readings=2, alpha=0.05)
    check(cs_pass["at_least"] == 0 and math.isclose(cs_pass["p_fw"], 0.05) and cs_pass["passes_alpha"],
          "m=2, C=39, at_least=0 -> p_fw=0.05 passes")
    cs_fail1 = lcl.control_standing(-1.0, [1.5] + [0.2] * 38, max_fresh_readings=2, alpha=0.05)
    check(cs_fail1["at_least"] == 1 and math.isclose(cs_fail1["p_fw"], 0.10) and not cs_fail1["passes_alpha"],
          "m=2, C=39, at_least=1 -> p_fw=0.10 fails")
    cs_short = lcl.control_standing(-1.0, [0.2] * 38, max_fresh_readings=2, alpha=0.05)
    check(cs_short["tops_every_control"] and not cs_short["passes_alpha"],
          "m=2, C=38, tops_every_control=True STILL fails p_fw=2/39 > 0.05 (tops_every never gates alone)")
    # a control edit with no finite reading is counted against the edit, never as one it topped
    cs_none = lcl.control_standing(-1.0, [None] * 39, max_fresh_readings=2, alpha=0.05)
    check(cs_none["non_finite"] == 39 and cs_none["at_least"] == 39 and not cs_none["passes_alpha"]
          and not cs_none["tops_every_control"],
          "m=2, C=39, every control reading missing -> at_least=39 and the family fails")
    cs_nan = lcl.control_standing(-1.0, [float("nan")] + [0.2] * 38, max_fresh_readings=2, alpha=0.05)
    check(cs_nan["non_finite"] == 1 and cs_nan["at_least"] == 1 and math.isclose(cs_nan["p_fw"], 0.10)
          and not cs_nan["passes_alpha"],
          "m=2, C=39, one NaN control -> counted as moving as much, p_fw=0.10 fails")

    # written_strata_summary (Row 21) & price_written_reading (§8)
    mock_w = {
        "plan": {"max_new": 512},
        "items": [{"seconds_a_step": 0.09}, {"seconds_a_step": 0.11}],
        "runtime": {"peak_gib": 20.5},
        "summary": {
            "base": {"n": 30, "ended": 28, "cut_off": 2, "looped": 12, "mean_pieces": 200.0},
            "arms": {"edit:1.0": {"ended": 30, "cut_off": 0, "looped": 3}},
            "controls_range": {"count": 5, "looped_min": 10, "looped_max": 14},
        },
    }
    ws = lcl.written_strata_summary(mock_w, main_dose=1.0)
    check(ws["cap_adequate_90pct"], "28/30 base finished -> cap_adequate_90pct=True")
    check(ws["strata"]["ended"]["readable"] and ws["strata"]["ended"]["note"] == "ok", "ended count 28 >= 20 is readable")
    check(not ws["strata"]["looped"]["readable"] and ws["strata"]["looped"]["note"] == "too few to read",
          "looped count 12 < 20 prints 'too few to read'")
    check(ws["strata"]["looped"]["least_gain_detectable"] is not None, "stratum reports least_gain_detectable")

    priced = lcl.price_written_reading(mock_w, n_items=8, n_seeds=2, n_controls=39, doses_count=2, rows_per_batch=12)
    check(priced["controls_per_chunk"] == 6 and priced["chunks_per_family"] == 7,
          f"pricing chunk math (6/chunk -> 7 chunks for C=39): {priced}")
    check(priced["estimated_gpu_seconds"] is not None and priced["estimated_gpu_seconds"] > 0, "pricing estimates GPU-seconds")


def run_section_3(tok: Any, tmp: Path) -> None:
    print("Section 3: Pure judge_claim planted-failure suite & BarSet tamper checks", flush=True)
    brief = lbr.pronoun_pull_brief(tok, max_fresh_readings=2)
    bars = lcl.make_bar_set(brief.id, neutral_kl_max=0.05, near_miss_abs_max=0.25, near_miss_ratio_max=0.35,
                            panel_own_drop_max=0.10, panel_kl_max=0.05)

    # BarSet save/load + tamper
    bar_p = tmp / "bars.json"
    bid = lcl.save_bars(bars, bar_p)
    check(lcl.load_bars(bar_p).id == bid, "save_bars/load_bars preserves bars_sha256")
    raw_b = json.loads(bar_p.read_text(encoding="utf-8"))
    raw_b["neutral_kl_max"] = 9.99
    tb = tmp / "bars_tampered.json"
    tb.write_text(json.dumps(raw_b), encoding="utf-8")
    check_raises(lcl.ClaimRefused, lambda: lcl.load_bars(tb), "load_bars refuses tampered BarSet", match="hashes to")
    raw_nokey = {k: v for k, v in json.loads(bar_p.read_text(encoding="utf-8")).items() if k != "bars_sha256"}
    raw_nokey["near_miss_abs_max"] = 5.0
    nk = tmp / "bars_nokey.json"
    nk.write_text(json.dumps(raw_nokey), encoding="utf-8")
    check_raises(lcl.ClaimRefused, lambda: lcl.load_bars(nk),
                 "load_bars refuses a bar set changed by hand with its bars_sha256 taken out",
                 match="carries no bars_sha256")

    good_cls = _synth_class(h0_vals=[-1.2, -1.0], h1_vals=[-1.1, -0.9], floor=0.01, reversed_mean=0.8)
    good_classes = {"small": good_cls, "large": good_cls}
    meta = _meta_stub(brief, bars)

    # 1. True lead (free_pick)
    j_lead = lcl.judge_claim(brief=brief, bars=bars, classes=good_classes, batteries=_clean_batteries(),
                             selection_mode="free_pick", practice_count=7, practice_best=-1.3)
    s_lead = lcl.claim_sentence(brief, j_lead, meta)
    sh_lead = lcl.render_sheet(brief, bars, j_lead, meta, s_lead)
    SHEETS_COLLECTED.extend([sh_lead, s_lead or ""])
    check(j_lead["verdict"] == "lead", f"planted true signal -> lead (got {j_lead['verdict']})")
    check(s_lead is not None and "selection not matched, 7 practice readings in this book" in s_lead,
          "free_pick claim_sentence carries 'selection not matched, 7 practice readings in this book'")
    check("This is a lead, not a finding of record." in (s_lead or ""), "claim_sentence is bounded as a lead")

    # 2. True lead (declared_rule with sham_target)
    sham_cls = _synth_class(h0_vals=[-1.2, -1.0], h1_vals=[-1.1, -0.9], sham_ctls=[0.08] * 39)
    j_decl = lcl.judge_claim(brief=brief, bars=bars, classes={"small": sham_cls, "large": sham_cls},
                             batteries=_clean_batteries(), selection_mode="declared_rule")
    s_decl = lcl.claim_sentence(brief, j_decl, meta)
    sh_decl = lcl.render_sheet(brief, bars, j_decl, meta, s_decl)
    SHEETS_COLLECTED.extend([sh_decl, s_decl or ""])
    check(j_decl["verdict"] == "lead" and "sham-target" in (s_decl or ""), "declared_rule with passing sham -> lead")
    check("sham-target control edits, supplied with the claim and checked for shape and for moving the stream as "
          "the edit does, moved as much" in (s_decl or ""),
          "the sentence of a declared-rule lead says its sham edits were supplied with the claim, and what was checked of them")
    meta_d = {**meta, "design": {"n_target": 8, "n_neutral": 4, "n_near": 4, "n_panel": 4,
                                 "rows": {"small": 12, "large": 32}, "controls_per_family": 39,
                                 "match_tolerance": 4.0},
              "shams": {"count": 39, "sha256": "f" * 64},
              "kept_back": {"frames": 8, "published": 8}}
    s_tol = lcl.claim_sentence(brief, j_decl, meta_d)
    sh_tol = lcl.render_sheet(brief, bars, j_decl, meta_d, s_tol)
    SHEETS_COLLECTED.extend([sh_tol, s_tol or ""])
    check("matched other-site (matched on what a base pre-pass expects each to deliver, within a factor of 4; p_fw" in (s_tol or "")
          and "on delivered" not in (s_tol or ""),
          "the sentence says what the matched controls were matched on (a base pre-pass's expectation) and within what factor")
    check("other sites matched on what a base pre-pass expects each to deliver, within a factor of 4" in sh_tol
          and "sham-target controls: 39 supplied with the claim" in sh_tol,
          "the sheet prints the registered design and that the shams were supplied, not made by the harness")
    check("kept-back frames, whose templates are published in the instrument's source, drawn from" in (s_tol or "")
          and "8 of the brief's 8 templates are published in the instrument's source" in sh_tol,
          "the sentence and the sheet of a starter brief's lead say its kept-back templates are published")
    meta_side = {**meta, "book": {"name": "check_book.jsonl", "default": False},
                 "book_fresh_readings": {brief.name: 1, "another_brief": 2}}
    s_side = lcl.claim_sentence(brief, j_decl, meta_side)
    sh_side = lcl.render_sheet(brief, bars, j_decl, meta_side, s_side)
    meta_home = {**meta, "book": {"name": "level_book.jsonl", "default": True}}
    s_home = lcl.claim_sentence(brief, j_decl, meta_home)
    sh_home = lcl.render_sheet(brief, bars, j_decl, meta_home, s_home)
    SHEETS_COLLECTED.extend([sh_side, s_side or "", sh_home, s_home or ""])
    check(f'on brief "{brief.name}", taken in a side book and not the default level book)' in (s_side or "")
          and "book: check_book.jsonl (a side book, not the default level book" in sh_side,
          "the sentence and the sheet of a reading taken in a side book say so")
    check("side book" not in (s_home or "")
          and "book: level_book.jsonl (the default level book of the environment this ran in)" in sh_home,
          "a reading taken in the default level book is not called a side-book reading")
    check("fresh readings already taken on other briefs in this book: another_brief 2" in sh_side,
          "a sheet lists the fresh readings other briefs have taken in its book")
    meta_p = {**meta, "kept_back": {"frames": 8, "published": 0}}
    s_priv = lcl.claim_sentence(brief, j_decl, meta_p)
    sh_priv = lcl.render_sheet(brief, bars, j_decl, meta_p, s_priv)
    SHEETS_COLLECTED.extend([sh_priv, s_priv or ""])
    check("published" not in (s_priv or "") and "None of the brief's 8 templates is a published starter template" in sh_priv
          and "a question put in other words is not found by either" in sh_priv
          and "retention batteries: their questions are few, fixed and in the instrument's source" in sh_priv,
          "a brief with no published template is not called published, and the sheet says what the harness cannot know")
    meta_rule = {**meta_d, "declared_rule": "the ten neurons that write she hardest at layer 51",
                 "bars_calibration_seq": 3, "bars_calibrations_in_book": 2}
    sh_rule = lcl.render_sheet(brief, bars, j_decl, meta_rule, s_tol)
    SHEETS_COLLECTED.append(sh_rule)
    check('declared rule, in the words given with the claim: "the ten neurons that write she hardest at layer 51"' in sh_rule
          and "(one of 2 calibrations of this brief in the book)" in sh_rule
          and "it cannot check that they follow the declared rule" in sh_rule,
          "the sheet prints the declared rule as the claimant's words, and how many calibrations the book holds")

    # 2b. declared_rule missing sham_target -> fails controls
    j_nosh = lcl.judge_claim(brief=brief, bars=bars, classes=good_classes, batteries=_clean_batteries(),
                             selection_mode="declared_rule")
    check(j_nosh["verdict"] == "no better than control edits", "declared_rule missing sham_target fails controls")

    # 3. Planted overfit (huge practice_best, near-zero fresh)
    null_cls = _synth_class(h0_vals=[-0.02, 0.01], h1_vals=[0.02, -0.01], floor=0.01)
    j_ovf = lcl.judge_claim(brief=brief, bars=bars, classes={"small": null_cls, "large": null_cls},
                            batteries=_clean_batteries(), practice_count=12, practice_best=-3.5)
    check(j_ovf["verdict"] == "kept the base", f"planted overfit -> kept the base (got {j_ovf['verdict']})")
    check(lcl.claim_sentence(brief, j_ovf, meta) is None, "non-lead emits no claim_sentence")

    # 4. Planted floor-too-loud (M < 4F)
    loud_cls = _synth_class(h0_vals=[-2.0, -2.0], h1_vals=[-2.0, -2.0], floor=0.20)  # M=0.75 < 4*0.20
    j_loud = lcl.judge_claim(brief=brief, bars=bars, classes={"small": loud_cls, "large": loud_cls},
                             batteries=_clean_batteries())
    check(j_loud["verdict"] == "kept the base", f"floor_too_loud -> kept the base (got {j_loud['verdict']})")

    # 5. Planted half-only artifact (half 0 strong, half 1 weak/opposite)
    h_cls = _synth_class(h0_vals=[-2.5, -2.3], h1_vals=[0.1, -0.01], floor=0.01)
    j_half = lcl.judge_claim(brief=brief, bars=bars, classes={"small": h_cls, "large": h_cls},
                             batteries=_clean_batteries())
    check(j_half["verdict"] == "did not replicate", f"half-only artifact -> did not replicate (got {j_half['verdict']})")

    # 6. Planted reversed-dose same-direction failure
    rev_cls = _synth_class(h0_vals=[-1.2, -1.0], h1_vals=[-1.1, -0.9], floor=0.01, reversed_mean=-0.9)
    j_rev = lcl.judge_claim(brief=brief, bars=bars, classes={"small": rev_cls, "large": rev_cls},
                            batteries=_clean_batteries())
    check(j_rev["verdict"] == "did not replicate", f"reversed-dose same-dir -> did not replicate (got {j_rev['verdict']})")

    # 7. Planted control-beaten null (in small class only!)
    bad_ctl_cls = _synth_class(h0_vals=[-1.2, -1.0], h1_vals=[-1.1, -0.9], osm_ctls=[1.5] + [0.05] * 38)
    j_ctl = lcl.judge_claim(brief=brief, bars=bars, classes={"small": bad_ctl_cls, "large": good_cls},
                            batteries=_clean_batteries())
    check(j_ctl["verdict"] == "no better than control edits",
          f"control beaten in small class -> no better than control edits (got {j_ctl['verdict']})")

    # 8. Planted harm on each gating battery
    for tag, mod in (
        ("neutral_kl", {"neutral_text": {"mean_kl": 0.12}}),
        ("near_miss_abs", {"near_misses": {"mean_effect": -0.40, "abs_effect": 0.40}}),
        ("near_miss_ratio", {"near_misses": {"mean_effect": -0.45, "abs_effect": 0.45}}),
        ("panel_own_drop", {"labelled_panel": {"own_drop": 0.25, "mean_kl": 0.01}}),
        ("panel_kl", {"labelled_panel": {"own_drop": 0.01, "mean_kl": 0.09}}),
    ):
        bats = _clean_batteries()
        bats.update(mod)
        jh = lcl.judge_claim(brief=brief, bars=bars, classes=good_classes, batteries=bats)
        check(jh["verdict"] == "works but harms", f"planted harm ({tag}) -> works but harms (got {jh['verdict']})")

    # 9. Planted cheaper-knob prompt match
    knobs = {"prompt_arms": [{"index": 0, "prompt_sha256": "abc123456789", "directed_effect": 1.15, "near_miss_abs": 0.04}]}
    j_knob = lcl.judge_claim(brief=brief, bars=bars, classes=good_classes, batteries=_clean_batteries(),
                             cheaper_knobs=knobs)
    s_knob = lcl.claim_sentence(brief, j_knob, meta)
    sh_knob = lcl.render_sheet(brief, bars, j_knob, meta, s_knob)
    SHEETS_COLLECTED.extend([sh_knob, s_knob or ""])
    check(j_knob["verdict"] == "lead, matched by a cheaper knob",
          f"cheaper-knob prompt match -> lead, matched by a cheaper knob (got {j_knob['verdict']})")
    check(s_knob is not None and "matched by prompt arm 0" in s_knob, "claim_sentence names matching prompt arm")

    # 9b. A control with no finite reading in a required family is no lead, and the gate notes say why
    nan_cls = _synth_class(h0_vals=[-1.2, -1.0], h1_vals=[-1.1, -0.9], floor=0.01, reversed_mean=0.8,
                           ss_ctls=[float("nan")] + [0.05] * 38)
    j_nan = lcl.judge_claim(brief=brief, bars=bars, classes={"small": nan_cls, "large": nan_cls},
                            batteries=_clean_batteries())
    check(j_nan["verdict"] == "no better than control edits"
          and any("gave no finite reading" in r for r in j_nan["reasons"]),
          f"one NaN same-site control -> no better than control edits, said in the gate notes (got {j_nan['verdict']})")

    # 9c. The sentence never says "none matched" of prompt arms that were not run, and the sheet says why not
    arm_short = {"index": 0, "prompt_sha256": "abc123456789", "directed_effect": 0.1, "near_miss_abs": 0.04}
    for state, want in (
        (lcl.PROMPT_ARMS_OFF, "no author prompt arm was compared with it (switched off for this reading)"),
        (lcl.PROMPT_ARMS_NONE, "no author prompt arm was compared with it (none in the brief)"),
        (None, "no author prompt arm was compared with it"),
    ):
        ck_off: Dict[str, Any] = {"prompt_arms": []}
        if state is not None:
            ck_off["prompt_arms_state"] = state
        j_off = lcl.judge_claim(brief=brief, bars=bars, classes=good_classes, batteries=_clean_batteries(),
                                cheaper_knobs=ck_off)
        s_off = lcl.claim_sentence(brief, j_off, meta)
        sh_off = lcl.render_sheet(brief, bars, j_off, meta, s_off)
        SHEETS_COLLECTED.extend([sh_off, s_off or ""])
        check(j_off["verdict"] == "lead" and s_off is not None and want in s_off
              and "no author prompt arm matched it" not in s_off,
              f"prompt arms not run ({state}) -> the sentence says none was compared, not that none matched")
        check(f"author prompt arms: {state or 'not compared'}" in sh_off,
              f"prompt arms not run ({state}) -> the sheet says so under cheaper knobs")
    j_ran = lcl.judge_claim(brief=brief, bars=bars, classes=good_classes, batteries=_clean_batteries(),
                            cheaper_knobs={"prompt_arms": [arm_short], "prompt_arms_state": lcl.PROMPT_ARMS_RUN})
    s_ran = lcl.claim_sentence(brief, j_ran, meta)
    SHEETS_COLLECTED.extend([lcl.render_sheet(brief, bars, j_ran, meta, s_ran), s_ran or ""])
    check(j_ran["verdict"] == "lead" and s_ran is not None and "no author prompt arm matched it" in s_ran,
          "a prompt arm that ran and fell short is reported as not matching")

    # 9d. A written reading's loop rates are counted by the brief's rule, and a written lead says a rule counted them
    wb = lbr.enumeration_loop_brief(tok)
    w_drawn = wb.generator.draw("kept_back", 4, seed_bytes=b"written-rule-check", tokenizer=tok)

    def w_row(own: bool, pub: bool) -> Dict[str, Any]:
        return {"loop": {"looped": own}, "loop_published": {"looped": pub}}

    w_result = {"plan": {"controls": {"count": 1}}, "items": [
        {"item": {"name": d.item.name}, "samples": [{
            "base": w_row(False, True),  # a list collapse: the published rule's alone
            "arms": [{"key": "edit:1.0", **w_row(False, False)}, {"key": "control:0", **w_row(False, True)}],
        }]} for d in w_drawn]}
    by_rule = {r: lcl._extract_written_effects(w_result, w_drawn, 1.0, r) for r in ("instrument", "published", "either")}
    check([e["value"] for e in by_rule["instrument"][0]] == [0.0] * 4
          and [e["value"] for e in by_rule["published"][0]] == [-1.0] * 4
          and [e["value"] for e in by_rule["either"][0]] == [-1.0] * 4
          and by_rule["published"][2] == [0.0] and by_rule["instrument"][2] == [0.0],
          "a loop only the published rule finds moves a written reading counted by it or by either, not by the instrument's")
    wbars_pure = lcl.make_bar_set(wb.id)
    w_cls = _synth_class(h0_vals=[-0.5, -0.5], h1_vals=[-0.5, -0.5], floor=None, reversed_mean=0.0)
    j_w = lcl.judge_claim(brief=wb, bars=wbars_pure, classes={"written": w_cls}, batteries=_clean_batteries())
    meta_w = {**_meta_stub(wb, wbars_pure), "design": {"match_tolerance": 2.0, "loop_rule": "either"},
              "written_strata": {"total_generations": 16, "loop_rule": "either", "cap_note": "ok", "strata": {},
                                 "loop_rules": {"base": {"instrument": 1, "published": 5, "either": 5, "rules_differ": 4},
                                                "edit": {"instrument": 0, "published": 1, "either": 1, "rules_differ": 1}}}}
    s_w = lcl.claim_sentence(wb, j_w, meta_w)
    sh_w = lcl.render_sheet(wb, wbars_pure, j_w, meta_w, s_w)
    SHEETS_COLLECTED.extend([sh_w, s_w or ""])
    check(j_w["verdict"] == "lead" and s_w is not None
          and 'Loops were counted by rule ("either"' in s_w and "no person's reading of this checkpoint's answers has checked" in s_w,
          f"the sentence of a written-loop lead says a rule, not a person, counted its loops (verdict {j_w['verdict']})")
    check('loops, counted by the rule "either" for this reading' in sh_w
          and "the instrument's rule calls 1 of the base's and 0 of the edit's looped, the published rule" in sh_w
          and "Neither rule is checked against a person's reading" in sh_w,
          "the sheet of a written reading prints what each loop rule counted, and that neither is checked by a person")
    check("Loops were counted" not in (s_lead or ""), "a paired lead's sentence says nothing of loop rules")

    # 10. Void on missing class or too few kept-back frames
    j_void1 = lcl.judge_claim(brief=brief, bars=bars, classes={"large": good_cls}, batteries=_clean_batteries())
    check(j_void1["verdict"] == "void", "missing small batch class -> void")
    few_cls = _synth_class(h0_vals=[-1.2], h1_vals=[-1.1, -0.9])
    j_void2 = lcl.judge_claim(brief=brief, bars=bars, classes={"small": few_cls, "large": few_cls},
                              batteries=_clean_batteries())
    check(j_void2["verdict"] == "void", "only 1 kept-back frame in half 0 -> void")


def run_section_4(tok: Any) -> None:
    print("Section 4: 200 planted nulls false-positive calibration (arbiter.md Row 24)", flush=True)
    brief = lbr.pronoun_pull_brief(tok, max_fresh_readings=2)
    bars = lcl.make_bar_set(brief.id)
    rng = np.random.default_rng(20261118)

    # Test A: 200 exchangeable nulls under full joint gate (both classes, 2 independent control families, half replication)
    fp_joint = 0
    for _ in range(200):
        # Exchangeable draw: edit and all 39 controls per family share the exact same N(0, 0.4^2) frame distribution
        edit_frames = rng.normal(0.0, 0.4, size=4)
        osm = rng.normal(0.0, 0.4, size=(39, 4)).mean(axis=1).tolist()
        ss = rng.normal(0.0, 0.4, size=(39, 4)).mean(axis=1).tolist()
        cls_d = _synth_class(
            h0_vals=edit_frames[:2].tolist(), h1_vals=edit_frames[2:].tolist(),
            floor=0.01, reversed_mean=0.0, osm_ctls=osm, ss_ctls=ss,
        )
        res = lcl.judge_claim(brief=brief, bars=bars, classes={"small": cls_d, "large": cls_d},
                              batteries=_clean_batteries())
        if res["verdict"] in ("lead", "lead, matched by a cheaper knob"):
            fp_joint += 1
    check(fp_joint <= 8, f"200 exchangeable joint-gate nulls: {fp_joint}/200 passed (band <= 8)")
    print(f"    200 planted nulls (joint gate, m=2, C=39): {fp_joint}/200 false leads", flush=True)

    # Test B: 200 worst-case exchangeable single-family nulls (tiny M=0.001, identical control pools across families,
    # half replication bypassed by setting identical frame values) -> theoretical per-reading FP <= 1/40 = 2.5% (5/200)
    brief_tiny = lbr.Brief(
        name="tiny_null_cal", behavior="Calibration test", metric="odds", direction=-1, least_effect=0.001,
        readouts=brief.readouts, generator=brief.generator, near_miss_generator=brief.near_miss_generator,
        max_fresh_readings=2,
    )
    bars_tiny = lcl.make_bar_set(brief_tiny.id)
    fp_worst = 0
    for _ in range(200):
        # 1 edit + 39 controls all iid N(0, 1); exact exchangeability on magnitude, single-sided sign (-1) -> 1/80 = 1.25%
        draws = rng.normal(0.0, 1.0, size=40)
        ev = float(draws[0])
        ctls = draws[1:].tolist()
        cls_d = _synth_class(h0_vals=[ev, ev], h1_vals=[ev, ev], floor=0.0, reversed_mean=0.0,
                             osm_ctls=ctls, ss_ctls=ctls)
        res = lcl.judge_claim(brief=brief_tiny, bars=bars_tiny, classes={"small": cls_d, "large": cls_d},
                              batteries=_clean_batteries())
        if res["verdict"] in ("lead", "lead, matched by a cheaper knob"):
            fp_worst += 1
    check(fp_worst <= 10, f"200 worst-case exchangeable nulls: {fp_worst}/200 passed (band <= 10, theory ~2.5)")
    print(f"    200 planted nulls (worst-case exchangeable, m=2, C=39): {fp_worst}/200 false leads", flush=True)


def run_section_5(h: lv.Handle, tmp: Path) -> None:
    print("Section 5: Live CPU stand-in end-to-end (calibrate, practice lock, paired & written claims)", flush=True)
    bk = tmp / "claim_book.jsonl"
    brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=5)

    # 1. Calibrate bars on stand-in
    bars = lcl.calibrate_bars(
        h, brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
        rows={"small": 12, "large": 16}, seed=7, book_path=bk,
    )
    check(len(bars.calibration_book_seqs) == 3, "calibrate_bars links 3 calibration reading seqs")
    hist0 = lcl.brief_history(brief.id, bk)
    check(len(hist0["calibrations"]) == 1, "calibration note written to level book")
    check(hist0["calibrations"][0]["bars_sha256"] == bars.id,
          "logged bars_sha256 matches returned BarSet.id byte-for-byte")
    check(lb.verify(bk)["ok"], "level book whole after calibrate_bars")
    cal0 = hist0["calibrations"][0]
    check(cal0.get("settings", {}).get("mult") == 2.0 and cal0["settings"]["layer"] == 2
          and cal0["settings"]["n_cols"] == 2 and cal0["settings"]["seed"] == 7
          and cal0["settings"]["rows"] == {"small": 12, "large": 16},
          "the calibration note records what the calibration was asked for (probe, counts, rows, seed, mult)")
    found_cal = lcl.calibration_note(bars, bk)
    check(found_cal is not None and found_cal["seq"] == cal0["seq"], "calibration_note finds the bars' own note")
    loose = lcl.make_bar_set(brief.id, near_miss_abs_max=5.0, near_miss_ratio_max=5.0)
    check(lcl.calibration_note(loose, bk) is None, "calibration_note finds nothing for a bar set made by hand")
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(
            h, brief, loose, {"terms": [{"layer": 2, "silence": [0, 1]}]},
            rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
            controls_override=4, match_tolerance=20.0, book_path=bk,
        ),
        "run_fresh_claim refuses bars that no calibration in its book produced",
        match="not from a calibration reading",
    )
    check(len(lcl.brief_history(brief.id, bk)["registrations"]) == 0,
          "the claim refused for its bars registered nothing")

    # 2. Record a practice draw & verify brief_history tracks it
    prac = brief.generator.draw("practice", 4, seed_bytes=b"practice_seed_abc", tokenizer=h.tokenizer)
    lcl.record_practice_draw(brief.id, prac, book_path=bk)
    hist_p = lcl.brief_history(brief.id, bk)
    check(len(hist_p["practice_item_ids"]) == 4, "brief_history records all 4 practice item ids")
    check(set(hist_p["practice_frame_ids"]) == {d.frame_id for d in prac}, "brief_history records practice frame ids")

    # 3. Pre-registration frame-leakage refusal does NOT burn a fresh reading slot
    kb_fid = brief.generator.frames_for("kept_back")[0].id
    edit1 = {"terms": [{"layer": 2, "silence": [0, 1]}]}
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(
            h, brief, bars, edit1, practice_frame_ids=[kb_fid],
            rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
            controls_override=4, match_tolerance=20.0, book_path=bk,
        ),
        "run_fresh_claim refuses kept-back frame leakage before registration",
        match="frame leakage refused",
    )
    check(len(lcl.brief_history(brief.id, bk)["registrations"]) == 0,
          "frame-leakage refusal did not burn a fresh reading slot")

    # 4. Run fresh claim #1
    res1 = lcl.run_fresh_claim(
        h, brief, bars, edit1,
        forecast={"expected_direction": -1, "note": "stand-in dry test 1"},
        practice_count=3, practice_best=-0.4,
        rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
        controls_override=4, match_tolerance=20.0, book_path=bk,
    )
    SHEETS_COLLECTED.append(res1["sheet"])
    check(res1["judged"]["cheaper_knobs"]["prompt_arms_state"] == lcl.PROMPT_ARMS_RUN
          and len(res1["judged"]["cheaper_knobs"]["prompt_arms"]) == len(brief.prompt_arms) > 0
          and "author prompt arms: run" in res1["sheet"],
          "fresh claim #1 compared the brief's author prompt arms and its sheet says they ran")
    reg1 = next(lb.entries(bk, start=res1["book"]["registration_seq"], limit=1))["body"]
    check(reg1.get("bars_calibration_seq") == cal0["seq"]
          and f"bars {bars.short()} from the calibration at book line {cal0['seq']}" in res1["sheet"],
          "fresh claim #1 registers, and prints on its sheet, the calibration its bars came from")
    est1 = lcl.estimate_claim(
        h, brief, edit1, rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
        controls_override=4, match_tolerance=20.0,
    )
    check(est1["forwards"] == res1["cost"]["forwards"] > 0 and est1["seconds"] > 0
          and {r["name"] for r in est1["readings"]} == {"other_sites_matched", "same_site", "neutral", "near_misses",
                                                        "labelled_panel", "prompt_arms"},
          f"a paired claim is priced beforehand at exactly the forwards it then takes "
          f"(priced {est1['forwards']}, took {res1['cost']['forwards']})")
    check(reg1.get("book") == {"name": "claim_book.jsonl", "default": False}
          and "book: claim_book.jsonl (a side book, not the default level book" in res1["sheet"],
          "a reading taken outside the default level book is registered and printed as a side-book reading")
    check(res1["verdict"] in lcl.VERDICTS and res1["verdict"] != "void",
          f"fresh claim #1 completed cleanly (verdict={res1['verdict']}, reasons={res1['judged']['reasons']})")
    reg_s = res1["book"]["registration_seq"]
    verd_s = res1["book"]["verdict_seq"]
    check(reg_s < verd_s and all(reg_s < s < verd_s for s in res1["book"]["all_seqs"][1:-1]),
          "claim_registration precedes all reading seqs, which precede claim_verdict")
    check(lb.verify(bk)["ok"], "level book whole after fresh claim #1")
    we.assert_base(h.lm, h.hooks_at_load)
    check(True, "assert_base clean after fresh claim #1")

    # Anti-leakage: driver_record contains zero kept-back question snippets
    drv_json = json.dumps(res1["driver_record"])
    check("Finish the story:" not in drv_json and "Write the next words:" not in drv_json,
          "driver_record leaks zero raw kept-back fresh item text")

    # 5. Read-once refusal on duplicate edit
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(
            h, brief, bars, edit1,
            rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
            controls_override=4, match_tolerance=20.0, book_path=bk,
        ),
        "run_fresh_claim refuses duplicate (brief_id, edit_id)",
        match="read once",
    )

    # 6. Run fresh claim #2 with a different edit -> sheet prints reading #1 in prior_verdicts!
    edit2 = {"terms": [{"layer": 2, "silence": [2, 3]}]}
    res2 = lcl.run_fresh_claim(
        h, brief, bars, edit2,
        rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
        controls_override=4, match_tolerance=20.0, run_prompt_arms=False, book_path=bk,
    )
    SHEETS_COLLECTED.append(res2["sheet"])
    est2 = lcl.estimate_claim(
        h, brief, edit2, rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
        controls_override=4, match_tolerance=20.0, run_prompt_arms=False,
    )
    check(est2["forwards"] == res2["cost"]["forwards"] and "prompt_arms" not in {r["name"] for r in est2["readings"]},
          f"with its prompt arms switched off a claim is priced without them (priced {est2['forwards']}, "
          f"took {res2['cost']['forwards']})")
    check(res2["judged"]["cheaper_knobs"]["prompt_arms_state"] == lcl.PROMPT_ARMS_OFF
          and res2["judged"]["cheaper_knobs"]["prompt_arms"] == []
          and "author prompt arms: switched off for this reading" in res2["sheet"],
          "a claim run with run_prompt_arms=False records and prints that its prompt arms were switched off")
    check(res2["verdict"] in lcl.VERDICTS and res2["verdict"] != "void",
          f"fresh claim #2 completed cleanly (verdict={res2['verdict']})")
    check("earlier fresh readings on this brief (1):" in res2["sheet"],
          "reading #2 Claim Sheet prints earlier reading #1")
    check(res1["driver_record"]["edit_id"][:12] in res2["sheet"],
          "reading #2 Claim Sheet includes reading #1 edit id")

    # 7. m-budget cap refusal on reading #3
    edit3 = {"terms": [{"layer": 2, "silence": [4, 5]}]}
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(
            h, brief, bars, edit3,
            rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
            controls_override=4, match_tolerance=20.0, book_path=bk,
        ),
        "run_fresh_claim refuses reading m+1 once brief budget m=2 is spent",
        match="already spent all 2",
    )

    # 7b. The budget is the brief's name's: one character changed in a brief does not start a new count
    d_changed = {**brief.to_dict(), "driver_text": brief.driver_text + " "}
    changed = lbr.Brief.from_dict(d_changed)
    by_hash = lcl.brief_history(changed.id, bk)
    by_name = lcl.brief_history(changed.id, bk, name=changed.name)
    check(changed.id != brief.id and changed.name == brief.name
          and len(by_hash["registrations"]) == 0 and len(by_name["registrations"]) == 2
          and by_name["other_ids"] == [brief.id] and len(by_name["verdicts"]) == 2
          and by_name["book_fresh_readings"] == {brief.name: 2},
          "a brief changed by one character has no history by its hash and both fresh readings by its name")
    small_args = dict(rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                      controls_override=4, match_tolerance=20.0, book_path=bk)
    n_lines = lb.verify(bk)["entries"]
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(h, changed, lcl.make_bar_set(changed.id), edit3, **small_args),
        "run_fresh_claim refuses a brief changed since its first fresh reading: no third reading under a new hash",
        match="has changed since its first fresh reading",
    )
    renamed = lbr.Brief.from_dict({**d_changed, "name": "pronoun_pull_b"})
    by_new_name = lcl.brief_history(renamed.id, bk, name=renamed.name)
    check(by_new_name["registrations"] == [] and by_new_name["other_ids"] == []
          and by_new_name["book_fresh_readings"] == {brief.name: 2},
          "a changed brief under a name of its own has no readings of its own, with the book's other readings in view")
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(h, renamed, lcl.make_bar_set(renamed.id), edit3, **small_args),
        "a renamed brief is refused the kept-back frames another brief already read here: a new name is not a new budget",
        match="already read in this book under brief",
    )
    recased = lbr.Brief.from_dict({**brief.to_dict(), "name": brief.name.upper()})
    check(len(lcl.brief_history(recased.id, bk, name=recased.name)["registrations"]) == 2,
          "a brief's name is its budget's whatever its case")
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(h, recased, lcl.make_bar_set(recased.id), edit3, **small_args),
        "the same brief under its name in capitals is the changed brief, not a new one",
        match="has changed since its first fresh reading",
    )

    # 7c. A frame drawn in practice is known by its template: moved into a later brief's kept-back split under a
    # new id and a new brief name, it is still refused
    gen_d = brief.generator.to_dict()
    practised_ids = set(hist_p["practice_frame_ids"])
    moved_id = next(f["id"] for f in gen_d["frames"] if f["split"] == "practice" and f["id"] in practised_ids)
    gen_d["frames"] = [({**f, "id": "pr_k99", "split": "kept_back"} if f["id"] == moved_id else f)
                       for f in gen_d["frames"]]
    moved = lbr.Brief.from_dict({**brief.to_dict(), "name": "pronoun_pull_moved", "generator": gen_d})
    check(len(lcl.brief_history(moved.id, bk, name=moved.name)["practice_frame_ids"]) == 0
          and len(lcl.brief_history(moved.id, bk, name=moved.name)["practice_templates"]) >= 1,
          "the book remembers practised templates whatever brief they were drawn under")
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(h, moved, lcl.make_bar_set(moved.id), edit3, **small_args),
        "run_fresh_claim refuses a kept-back frame whose template was drawn in practice under another brief and id",
        match="frame leakage refused",
    )
    check(lb.verify(bk)["entries"] == n_lines, "the claims refused for a changed brief or a moved frame entered nothing")

    # 8. Live written_loop claim with chunked controls + written_strata + settings_audit
    wbrief_base = lbr.enumeration_loop_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=4)
    wbrief = lbr.Brief.from_dict({
        **wbrief_base.to_dict(),
        "written_config": {"max_new_tokens": 12, "temperature": 1.0, "top_p": 0.95, "seeds": [1],
                           "loop_rule": "either"},
    })
    wbars = lcl.calibrate_bars(
        h, wbrief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
        rows={"small": 12, "large": 16}, seed=7, book_path=bk,
    )
    wres = lcl.run_fresh_claim(
        h, wbrief, wbars, {"terms": [{"layer": 2, "silence": [6, 7]}]},
        rows={"small": 12, "large": 16}, written_rows=9,  # 9 - 2 - 2 - 3 = 2 controls/chunk -> forces 2 chunks!
        n_target=4, n_neutral=2, n_near=2, n_panel=2,
        controls_override=4, match_tolerance=20.0, run_prompt_arms=False, book_path=bk,
    )
    SHEETS_COLLECTED.append(wres["sheet"])
    check(wres["verdict"] in lcl.VERDICTS and wres["verdict"] != "void",
          f"written_loop fresh claim completed cleanly (verdict={wres['verdict']}, reasons={wres['judged']['reasons']})")
    check(wres["meta"]["written_strata"] is not None, "written_loop populates written_strata")
    west = lcl.estimate_claim(
        h, wbrief, {"terms": [{"layer": 2, "silence": [6, 7]}]},
        rows={"small": 12, "large": 16}, written_rows=9, n_target=4, n_neutral=2, n_near=2, n_panel=2,
        controls_override=4, match_tolerance=20.0, run_prompt_arms=False,
    )
    check(west["forwards"] == wres["cost"]["forwards"]
          and sum(1 for r in west["readings"] if r["name"].startswith("settings_audit_")) == 2
          and sum(1 for r in west["readings"] if r["name"].startswith("other_sites_matched_c")) == 2,
          f"a written-loop claim is priced chunk by chunk, with its settings audit, at the forwards it takes "
          f"(priced {west['forwards']}, took {wres['cost']['forwards']})")
    wreg = next(lb.entries(bk, start=wres["book"]["registration_seq"], limit=1))["body"]
    ws_now = wres["meta"]["written_strata"]
    check(wreg["design"].get("loop_rule") == "either" and ws_now.get("loop_rule") == "either"
          and ws_now["loop_rules"]["base"]["published"] is not None
          and 'loops, counted by the rule "either" for this reading' in wres["sheet"],
          "a written-loop claim registers its loop rule and prints what each rule counted")
    sa_cpu = wres["meta"]["settings_audit"]
    check(
        sa_cpu is not None
        and sa_cpu["thinking_off"]["status"] == "ok"
        and sa_cpu["thinking_on"]["status"] == "ok",
        f"compare_settings=True populates ok settings_audit (got {sa_cpu})",
    )
    check("written strata (" in wres["sheet"] and "base settings audit (" in wres["sheet"],
          "written_loop Claim Sheet prints written strata and base settings audit")
    check("(a written reading does not record what they delivered as read)" in wres["sheet"]
          and "is printed with each class" not in wres["sheet"] and "delivered (stream change" not in wres["sheet"],
          "a written claim's sheet does not say that what its control edits delivered is printed: a written reading "
          "records none")
    check(lb.verify(bk)["ok"], "level book whole after written_loop claim")
    we.assert_base(h.lm, h.hooks_at_load)

    # 9. Sham-target controls are built and checked before anything is registered; a good set is registered by hash
    sbk = tmp / "sham_book.jsonl"
    sbars = lcl.calibrate_bars(
        h, brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
        rows={"small": 12, "large": 16}, seed=7, book_path=sbk,
    )
    s_edit = {"terms": [{"layer": 2, "silence": [0, 1]}]}
    small = dict(rows={"small": 12, "large": 16}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                 controls_override=4, match_tolerance=20.0, run_prompt_arms=False, book_path=sbk)
    for bad, why, words in (
        ([{"terms": [{"layer": 2, "silence": [4, 5, 6]}]}], "three columns for the edit's two", "not the claimed edit's shape"),
        ([{"terms": [{"layer": 2, "scale": [4, 5], "factor": 0.5}]}], "a scale for the edit's silencing", "not the claimed edit's shape"),
        ([{"terms": [{"layer": 2, "silence": [0, 1]}]}], "the claimed edit itself", "is the claimed edit itself"),
        ([{"terms": [{"layer": 2, "silence": [4, 5]}]}, {"terms": [{"layer": 2, "silence": [5, 4]}]}],
         "the same sham twice", "is the same edit as sham control 0"),
        ([{"terms": [{"layer": 2, "silence": [4, 5]}]}, {"terms": []}], "a recipe with no terms", "Sham control 1"),
    ):
        check_raises(
            lcl.ClaimRefused,
            lambda bad=bad: lcl.run_fresh_claim(h, brief, sbars, s_edit, selection_mode="declared_rule",
                                                sham_recipes=bad, **small),
            f"a declared-rule claim refuses a sham that is {why}",
            match=words,
        )
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(h, brief, sbars, s_edit, **{**small, "match_tolerance": 0.5}),
        "a claim refuses a matching tolerance under 1",
        match="finite factor of 1 or more",
    )
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(h, brief, sbars, s_edit, **{**small, "n_target": 2}),
        "a claim refuses, before it registers, fewer fresh items than two frames a half need",
        match="n_target",
    )
    for kwargs, why, words in (
        (dict(main_dose=0.0), "a dose of nought", "not registered"),
        (dict(controls_override=10 ** 6), "more control edits than a reading takes", "not registered"),
        (dict(sham_recipes=[{"terms": [{"layer": 2, "silence": [4, 5]}]}]), "sham edits with a free pick",
         "Sham edits are read only for a site chosen by a declared rule"),
        (dict(selection_mode="declared_rule", sham_recipes=[{"terms": [{"layer": 2, "silence": [4, 5]}]}],
              declared_rule="   "), "a declared rule of no words", "declared_rule is the rule"),
    ):
        check_raises(lcl.ClaimRefused, lambda kwargs=kwargs: lcl.run_fresh_claim(h, brief, sbars, s_edit, **{**small, **kwargs}),
                     f"a claim is refused, before it registers, {why}", match=words)
    check_raises(lcl.ClaimRefused, lambda: lcl.run_fresh_claim(h, brief, sbars, ["not", "a recipe"], **small),
                 "a claim is refused an edit that is no recipe, in plain words", match="An edit is a recipe")
    check_raises(lbr.BriefError,
                 lambda: lcl.estimate_claim(h, lbr.canary_null_brief(h.tokenizer), s_edit, n_near=14),
                 "pricing a claim makes its draws: more near misses than the brief can give are refused there, "
                 "before any registration", match="could not draw")
    check(len(lcl.brief_history(brief.id, sbk)["registrations"]) == 0,
          "none of the claims refused for their shams, dose or design registered anything")
    shams = [{"terms": [{"layer": 2, "silence": [a, a + 1]}]} for a in (4, 6, 8, 10)]
    sres = lcl.run_fresh_claim(h, brief, sbars, s_edit, selection_mode="declared_rule", sham_recipes=shams, **small)
    SHEETS_COLLECTED.append(sres["sheet"])
    sest = lcl.estimate_claim(h, brief, s_edit, selection_mode="declared_rule", sham_recipes=shams,
                              **{k: v for k, v in small.items() if k != "book_path"})
    check(sest["forwards"] == sres["cost"]["forwards"] and "sham_target" in {r["name"] for r in sest["readings"]},
          f"a declared-rule claim is priced with its sham reading (priced {sest['forwards']}, took {sres['cost']['forwards']})")
    sreg = next(lb.entries(sbk, start=sres["book"]["registration_seq"], limit=1))["body"]
    check(sres["verdict"] in lcl.VERDICTS and sres["verdict"] != "void"
          and (sreg.get("shams") or {}).get("count") == 4 and len(sreg["shams"]["edit_ids"]) == 4
          and len(sreg["shams"]["sha256"]) == 64 and sreg["design"]["match_tolerance"] == 20.0,
          f"a declared-rule claim registers its sham edits by id and hash before the draw (verdict={sres['verdict']})")
    dl = (sres["judged"]["classes"].get("small") or {}).get("delivered") or {}
    check(all((dl.get(f) or {}).get("controls_read") == 4 for f in ("other_sites_matched", "same_site", "sham_target"))
          and (dl["sham_target"]["edit"] or 0.0) > 0.0
          and "delivered (stream change at the highest edited layer, a mean over the items read) over the prompt: edit " in sres["sheet"]
          and "; over the answer: edit " in sres["sheet"]
          and "; at the last position, where the metric is read: edit " in sres["sheet"]
          and dl["other_sites_matched"]["read_at_end"] is True and dl["other_sites_matched"]["edit_end"] is not None
          and dl["other_sites_matched"]["edit_prompt"] is not None
          and dl["other_sites_matched"]["compared_at"] == ["prompt", "answer", "end"]
          and isinstance(dl["other_sites_matched"]["disjoint_sets"], int)
          and dl["other_sites_matched"]["draw"] == "independent" and "each set drawn on its own" in sres["sheet"]
          and "per_control" not in dl["other_sites_matched"] and "apart" not in dl["sham_target"]
          and "sham-target controls: 4 supplied with the claim" in sres["sheet"],
          "the sheet prints how much the edit and each family's control edits changed the stream, over the answer and "
          "at the last position where this brief's metric is read, and how the matched sets were drawn")
    check(lb.verify(sbk)["ok"], "level book whole after the declared-rule claim")
    we.assert_base(h.lm, h.hooks_at_load)

    # 10. Sham edits that hardly move the stream are no controls: they count against the edit, never for it
    from backend.gguf_weights import dense_cols
    vec_dir = tmp / "sham_vectors"
    vec_dir.mkdir()
    down = h.lm.get_submodule(we.linear_name(2, "down_proj"))
    cols01 = dense_cols(down, torch.tensor([0, 1])).float().cpu().numpy()
    np.save(vec_dir / "big.npy", (-0.5 * cols01).astype(np.float32))
    np.save(vec_dir / "tiny.npy", (1e-6 * cols01).astype(np.float32))
    add_edit = {"terms": [{"layer": 2, "add": [0, 1], "vectors": "big.npy"}]}
    quiet = [{"terms": [{"layer": 2, "add": [a, a + 1], "vectors": "tiny.npy"}]} for a in (4, 6, 8, 10)]
    qres = lcl.run_fresh_claim(h, brief, sbars, add_edit, base_dir=vec_dir, selection_mode="declared_rule",
                               sham_recipes=quiet, declared_rule="two columns chosen by hand, for this check", **small)
    SHEETS_COLLECTED.append(qres["sheet"])
    q_small = (qres["judged"]["classes"].get("small") or {})
    q_sh, q_dv = (q_small.get("controls") or {}).get("sham_target") or {}, (q_small.get("delivered") or {}).get("sham_target") or {}
    check(qres["verdict"] not in ("lead", "lead, matched by a cheaper knob", "void")
          and q_dv.get("not_comparable") == 4 and q_sh.get("at_least") == 4 and q_sh.get("passes_alpha") is False
          and "quieter than the edit at one of those, or with no figure, and counted as moving as much" in qres["sheet"]
          and 'declared rule, in the words given with the claim: "two columns chosen by hand, for this check"' in qres["sheet"],
          f"four sham edits a millionth of the edit's size count against it, and the sheet says so "
          f"(verdict {qres['verdict']}, delivered {q_dv}, standing {q_sh})")
    qreg = next(lb.entries(sbk, start=qres["book"]["registration_seq"], limit=1))["body"]
    check(qreg.get("declared_rule") == "two columns chosen by hand, for this check"
          and qreg.get("priced", {}).get("forwards", 0) > 0 and qreg.get("practice_readings_in_book") == 0
          and qreg.get("bars_calibrations_in_book") == 1,
          "the registration keeps the declared rule, the reading's price and the book's count of practice readings")

    # 11. A fresh draw refused after the registration ends the reading void, with its verdict in the book
    vbk = tmp / "void_book.jsonl"
    vbars = lcl.calibrate_bars(
        h, brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
        rows={"small": 12, "large": 16}, seed=7, book_path=vbk,
    )
    v_seed = bytes([7]) * 32
    will_draw = brief.generator.draw("kept_back", 4, seed_bytes=v_seed + b":target", tokenizer=h.tokenizer,
                                     system_override=(brief.chat_settings or {}).get("system"), name_prefix="fresh_1")
    vres = lcl.run_fresh_claim(h, brief, vbars, s_edit, practice_item_ids=[will_draw[0].item.id], _seed_override=v_seed,
                               **{**small, "book_path": vbk})
    v_hist = lcl.brief_history(brief.id, vbk, name=brief.name)
    check(vres["verdict"] == "void" and "The fresh draw was refused" in " ".join(vres["judged"]["reasons"])
          and len(v_hist["registrations"]) == 1 and len(v_hist["verdicts"]) == 1 and vres["cost"]["forwards"] == 0,
          f"a fresh item that was seen in practice voids the registered reading, with a verdict note and no forward "
          f"({vres['verdict']}: {vres['judged']['reasons']})")

    # 12. Kept-back wording read in practice, by any door, ends a claim on the brief in that book
    pbk = tmp / "practised_book.jsonl"
    kb_seen = brief.generator.draw("kept_back", 2, seed_bytes=b"practised-kept-back", tokenizer=h.tokenizer)[0].item
    env_was = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(pbk)
    try:
        r_own = lv.read(h, {"name": "owner_practice", "kind": "paired", "edit": s_edit,
                            "controls": {"count": 2, "seed": 0}, "rows": {"small": 12, "large": 16}}, {"kb": kb_seen})
    finally:
        if env_was is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = env_was
    p_hist = lcl.brief_history(brief.id, pbk, name=brief.name, guard=lbr.KeptBack([brief]))
    check(r_own["status"] == "ok" and p_hist["practice_readings"] == 1
          and p_hist["kept_back_practised"] == [int(r_own["book"]["seq"])],
          "the book's practice readings are counted, and one that holds a kept-back frame's wording is found")
    check_raises(
        lcl.ClaimRefused,
        lambda: lcl.run_fresh_claim(h, brief, lcl.make_bar_set(brief.id), s_edit, **{**small, "book_path": pbk}),
        "a claim is refused on a brief whose kept-back frames were read in practice in its book, whatever door read them",
        match="were read in practice in this book",
    )


def run_section_6(h: lv.Handle, tmp: Path) -> None:
    print("Section 6: LevelSession + MCP claim gate & CLI subcommands", flush=True)
    brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=4)
    bk = tmp / "mcp_claim_book.jsonl"
    bars = lcl.calibrate_bars(
        h, brief, layer=2, n_cols=2, n_controls=2, n_neutral=2, n_near=2, n_panel=2,
        rows={"small": 12, "large": 16}, seed=7, book_path=bk,
    )

    # A session is offered the claim tool only with bars a calibration in its own book produced
    check_raises(
        lses.SessionRefused,
        lambda: lses.LevelSession(
            h, tmp / "ses_uncal", reach="tools_only", backend="local",
            book_path=bk, brief=brief, bars=lcl.make_bar_set(brief.id, near_miss_abs_max=5.0),
        ),
        "a session is refused at its start with bars that no calibration in its book produced",
        match="not from a calibration reading",
    )
    check_raises(
        lses.SessionRefused,
        lambda: lses.LevelSession(
            h, tmp / "ses_otherbook", reach="tools_only", backend="local",
            book_path=tmp / "other_book.jsonl", brief=brief, bars=bars,
        ),
        "a session is refused at its start with bars calibrated in another book",
        match="not from a calibration reading",
    )

    # Blinded canary session MUST NOT expose claim tool
    ses_blind = lses.LevelSession(
        h, tmp / "ses_blind", reach="tools_only", backend="local",
        book_path=tmp / "blind_book.jsonl", brief=brief, bars=bars, canary=True,
    )
    check("claim" not in ses_blind.allowed_tools(), "blinded canary session hides claim tool")
    ses_blind.close()

    # Bound brief + bars session via MCP dispatch; the stand-in's small design is the owner's, set at the start
    ses_design = {"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "controls_override": 4,
                  "match_tolerance": 20.0}
    check_raises(
        lses.SessionRefused,
        lambda: lses.LevelSession(
            h, tmp / "ses_baddesign", reach="tools_only", backend="local",
            book_path=bk, brief=brief, bars=bars, claim_design={"run_prompt_arms": False},
        ),
        "a session's claim_design takes only the settings named for it",
        match="claim_design takes",
    )
    ses = lses.LevelSession(
        h, tmp / "ses_claim", reach="tools_only", backend="local",
        book_path=bk, brief=brief, bars=bars, claim_design=ses_design,
    )
    check(ses.harness_card().get("claim_design", {}).get("match_tolerance") == 20.0
          and ses.harness_card()["claim_design"]["n_target"] == 4
          and ses.harness_card()["claim_design"]["controls_per_family"] == 4,
          "the harness card shows the driver the design its fresh readings will be taken under")
    ses_plain = lses.LevelSession(h, tmp / "ses_plain", reach="tools_only", backend="local",
                                  book_path=bk, brief=brief, bars=bars)
    plain_design = ses_plain.harness_card()["claim_design"]
    check(plain_design["n_target"] == 8 and plain_design["n_near"] == 4 and plain_design["match_tolerance"] == lv.DEFAULT_MATCH_TOL
          and plain_design["controls_per_family"] == brief.default_controls
          and plain_design["rows"] == {"small": lv.ROWS_SMALL, "large": lv.ROWS_LARGE},
          "with nothing set by its owner, a session's card still shows the whole design: the gate's own")
    ses_plain.close()
    check(set(["brief_info", "draw_practice", "claim"]).issubset(ses.allowed_tools()),
          "bound session exposes brief_info, draw_practice, claim")
    check(not ses.harness_card()["practice_only"], "harness_card sets practice_only=False when claim gate is bound")

    t_list = lmcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, ses)
    tool_names = {t["name"] for t in t_list["result"]["tools"]}
    check({"brief_info", "draw_practice", "claim"}.issubset(tool_names), "MCP tools/list exposes Phase 3 tools")

    r_info = lmcp.dispatch_tool(ses, "brief_info", {})
    check(r_info["brief_id"] == brief.id and r_info["fresh_readings_remaining"] == 2, "MCP brief_info reports remaining m=2")

    r_prac = lmcp.dispatch_tool(ses, "draw_practice", {"n": 2, "seed": 3})
    check(r_prac["count"] == 2 and len(lcl.brief_history(brief.id, bk)["practice_item_ids"]) == 2,
          "MCP draw_practice returns items and records them in ledger")
    check(all(ses._kept_back.hit(v["question"], v["system"], v["answer"]) is None for v in r_prac["items"].values()),
          "the practice items the door hands out pass its own kept-back guard")

    # Kept-back frames are held out at the gates to the model, whoever wrote the item and with a brief bound or not
    kb_item = brief.generator.draw("kept_back", 2, seed_bytes=b"door-kept-back", tokenizer=h.tokenizer)[0].item
    nm_item = brief.near_miss_generator.draw("kept_back", 2, seed_bytes=b"door-near-miss", tokenizer=h.tokenizer)[0].item
    plan_kb = {"name": "kb_try", "kind": "paired", "edit": {"terms": [{"layer": 2, "silence": [0, 1]}]},
               "controls": {"count": 3, "seed": 0}, "rows": {"small": 8, "large": 16}}
    lines_before, fwd_before = lb.verify(bk)["entries"], ses.forwards_used
    for tool, args, what in (
        ("read", {"plan": plan_kb, "items": {"kb": {"question": kb_item.question, "answer": kb_item.answer}}},
         "a paired reading on a kept-back frame"),
        ("read", {"plan": plan_kb, "items": {"nm": {"question": nm_item.question, "answer": nm_item.answer}}},
         "a paired reading on a kept-back near-miss frame"),
        ("write", {"plan": {**plan_kb, "kind": "written", "rows": 8, "max_new": 4},
                   "items": {"kb": {"question": "Before you start: " + kb_item.question.upper(), "answer": "x"}}},
         "a written reading on a kept-back frame inside a longer question"),
        ("capture", {"items_a": [{"question": "Say it.", "system": kb_item.question, "answer": "a"}],
                     "items_b": [{"question": "An ordinary question about rivers?", "answer": "b"}], "layer": 2},
         "a capture with a kept-back frame in the system turn"),
    ):
        check_raises(lses.SessionRefused, lambda tool=tool, args=args: lmcp.dispatch_tool(ses, tool, args),
                     f"the door refuses {what}", match="kept-back frame")
    check(lb.verify(bk)["entries"] == lines_before and ses.forwards_used == fwd_before,
          "the refused practice calls ran nothing and entered nothing in the book")
    transcript = (tmp / "ses_claim" / "transcript.jsonl").read_text(encoding="utf-8")
    check(ses.status()["usage"]["gate_refusals"] == 4 and transcript.count('"why": "kept_back"') == 4
          and "Finish the story" not in transcript and "Write the next words" not in transcript,
          "each refusal for a kept-back frame is counted and written to the transcript, without the item's text")
    check_raises(lses.SessionRefused,
                 lambda: lmcp.dispatch_tool(ses, "read", {"plan": {**plan_kb, "name": "claim_pronoun_pull_1_matched"},
                                                          "items": {"q": {"question": "An ordinary question about rivers?",
                                                                          "answer": "They run to the sea."}}}),
                 "a practice reading may not take a name the harness keeps for its own readings",
                 match="are the harness's own readings")
    check_raises(lses.SessionRefused,
                 lambda: lmcp.dispatch_tool(ses, "read", {"plan": plan_kb, "items": {
                     "q": {"question": "What is the capital of France?", "answer": "Paris"}}}),
                 "the door refuses a question of the retention batteries", match="the gate's retention batteries")
    check_raises(lses.SessionRefused,
                 lambda: lmcp.dispatch_tool(ses, "claim", {
                     "edit": {"terms": [{"layer": 2, "silence": [0, 1]}]}, "selection_mode": "declared_rule",
                     "sham_recipes": [{"terms": [{"layer": 2, "silence": [4, 5]}]}]}),
                 "through the door a claim by a declared rule must state the rule", match="states its rule")
    ses_free = lses.LevelSession(h, tmp / "ses_free", reach="tools_only", backend="local", book_path=bk)
    spider = lbr.starter_briefs(h.tokenizer)["spider_legs_bleed"].generator.draw(
        "kept_back", 2, seed_bytes=b"door-spider", tokenizer=h.tokenizer)[0].item
    check_raises(
        lses.SessionRefused,
        lambda: ses_free.read(plan_kb, {"sp": {"question": spider.question, "answer": spider.answer}}),
        "a session with no brief bound refuses a starter brief's kept-back frame all the same",
        match='kept-back frame of brief "spider_legs_bleed"',
    )
    ses_free.close()

    # The door has no switch for the comparison that can mark a lead "matched by a cheaper knob"
    check("run_prompt_arms" not in lmcp.TOOL_SPECS["claim"]["inputSchema"]["properties"],
          "the MCP claim tool lists no run_prompt_arms argument")
    check_raises(
        lses.SessionRefused,
        lambda: lmcp.dispatch_tool(ses, "claim", {
            "edit": {"terms": [{"layer": 2, "silence": [0, 1]}]}, "run_prompt_arms": False,
        }),
        "MCP claim refuses an argument its schema does not name",
        match="takes no argument named run_prompt_arms",
    )
    check(len(lcl.brief_history(brief.id, bk)["registrations"]) == 0,
          "the refused MCP claim call registered nothing and spent no fresh reading")

    # ... and none for how the reading is taken: the tolerance, the counts and the controls are not the driver's
    claim_props = set(lmcp.TOOL_SPECS["claim"]["inputSchema"]["properties"])
    check(claim_props == {"edit", "main_dose", "forecast", "selection_mode", "sham_recipes", "declared_rule",
                          "practice_best"},
          f"the MCP claim tool takes the edit, its dose, a forecast, the selection, its shams and their rule, nothing else ({sorted(claim_props)})")
    for arg, val in (("match_tolerance", 1e9), ("controls_override", 1), ("n_target", 2)):
        check_raises(
            lses.SessionRefused,
            lambda arg=arg, val=val: lmcp.dispatch_tool(ses, "claim", {
                "edit": {"terms": [{"layer": 2, "silence": [0, 1]}]}, arg: val,
            }),
            f"MCP claim refuses the driver's {arg}",
            match=f"takes no argument named {arg}",
        )
    check(len(lcl.brief_history(brief.id, bk)["registrations"]) == 0,
          "the refused MCP claim calls registered nothing")

    r_claim = lmcp.dispatch_tool(ses, "claim", {
        "edit": {"terms": [{"layer": 2, "silence": [0, 1]}]},
    })
    SHEETS_COLLECTED.append(r_claim["sheet"])
    reg_mcp = next(lb.entries(bk, start=r_claim["book"]["registration_seq"], limit=1))["body"]
    check(reg_mcp.get("design", {}).get("match_tolerance") == 20.0 and reg_mcp["design"]["n_target"] == 4
          and reg_mcp["design"]["controls_per_family"] == 4 and reg_mcp["design"]["prompt_arms"] is True
          and reg_mcp.get("shams") is None,
          "the registration records the reading's whole design before the draw (counts, rows, controls, tolerance)")
    as_read = re.findall(r"; (\d+) of (\d+) at least as loud as the edit over the prompt, over the answer and at the "
                         r"last position", r_claim["sheet"])
    drawn_from = re.findall(r"; (\d+) of the sets share no site with one another \(each neuron had at least (\d+) eligible "
                            r"sites\), each set drawn on its own", r_claim["sheet"])
    check("other sites matched on what a base pre-pass expects each to deliver, within a factor of 20 (what they "
          "delivered as read is printed with each class)" in r_claim["sheet"]
          and len(as_read) == 2 and all(0 <= int(a) <= int(b) == 4 for a, b in as_read) and len(drawn_from) == 2
          and all(1 <= int(d) <= 4 and int(d) <= int(e) for d, e in drawn_from)
          and "on delivered" not in r_claim["sheet"] and "delivery-matched" not in r_claim["sheet"],
          "the sheet prints what the matched controls were matched on and within what factor, how many were at least as "
          f"loud as the edit as read ({as_read}), and how many of the sets share no site, of how many eligible sites "
          f"({drawn_from})")
    ig_claim = re.findall(r"item gate: (\d+) of (\d+) fresh items are under the kernel's bar of -6\.00 nats a piece",
                          r_claim["sheet"])
    check(len(ig_claim) == 1 and 0 <= int(ig_claim[0][0]) <= int(ig_claim[0][1]) == 4,
          f"a claim's sheet says how many of its fresh items the kernel read under its item gate ({ig_claim})")
    use_now = ses.status()["usage"]
    check(r_claim["forwards_charged"] == r_claim["estimated"]["forwards"] == use_now["claim_forwards"] > 0
          and use_now["forwards"] == fwd_before,
          "a claim's forwards are counted beside the practice forwards, as priced, and not against the practice budget")
    check((reg_mcp.get("kept_back") or {}).get("frames") == 8 and reg_mcp["kept_back"]["published"] == 8
          and len(reg_mcp["kept_back"]["template_ids"]) == 8
          and "8 of the brief's 8 templates are published in the instrument's source" in r_claim["sheet"],
          "a starter brief's claim registers, and prints, that its kept-back templates are published")
    check(r_claim["cheaper_knobs"]["prompt_arms_state"] == lcl.PROMPT_ARMS_RUN
          and len(r_claim["cheaper_knobs"]["prompt_arms"]) == len(brief.prompt_arms) > 0,
          "a claim through the door always compares the brief's author prompt arms")
    check(r_claim["verdict"] in lcl.VERDICTS and r_claim["reading_index"] == 1, "MCP claim returns redacted verdict record")
    check("Finish the story:" not in json.dumps(r_claim), "MCP claim leaks zero fresh kept-back text")

    # Verify book_view redacts items of fresh claim readings too!
    bv = lmcp.dispatch_tool(ses, "book_view", {"tail": 15})
    check("Finish the story:" not in json.dumps(bv), "MCP book_view leaks zero fresh kept-back text")
    ses.close()
    check(lb.verify(bk)["ok"], "MCP claim session ledger whole")

    # The session's budget is asked of every call: its first practice reading, and a claim before it is registered
    n_reg = len(lcl.brief_history(brief.id, bk, name=brief.name)["registrations"])
    edit_b = {"terms": [{"layer": 2, "silence": [2, 3]}]}
    plan_b = {"name": "first_call", "kind": "paired", "edit": edit_b, "controls": {"count": 3, "seed": 0},
              "rows": {"small": 8, "large": 16}}
    item_b = {"q": {"question": "An ordinary question about rivers?", "answer": "They run to the sea."}}
    for tag, budget, call, words in (
        ("its first practice reading", lses.SessionBudget(max_gpu_seconds=0.001),
         lambda s: s.read(plan_b, item_b), "GPU-second budget exceeded"),
        ("a claim it cannot afford in seconds", lses.SessionBudget(max_gpu_seconds=0.5),
         lambda s: s.claim(edit_b), "GPU-second budget exceeded"),
        ("a claim that would run past its wall-clock cap", lses.SessionBudget(max_gpu_seconds=1e6, max_wall_seconds=0.5),
         lambda s: s.claim(edit_b), "Wall-clock cap"),
    ):
        s_b = lses.LevelSession(h, tmp / f"ses_budget_{len(tag)}", reach="tools_only", backend="local",
                                book_path=bk, brief=brief, bars=bars, claim_design=ses_design, budget=budget)
        lines_b = lb.verify(bk)["entries"]
        check_raises(lses.SessionRefused, lambda s_b=s_b, call=call: call(s_b),
                     f"a session refuses {tag}, by its price, before anything runs", match=words)
        check(lb.verify(bk)["entries"] == lines_b and s_b.status()["usage"]["gpu_seconds"] == 0.0,
              f"the call refused as {tag} ran nothing and entered nothing")
        s_b.close()
    check(len(lcl.brief_history(brief.id, bk, name=brief.name)["registrations"]) == n_reg,
          "a claim refused for its price is not registered and spends none of the brief's fresh readings")
    s_wall = lses.LevelSession(h, tmp / "ses_wall", reach="tools_only", backend="local", book_path=bk,
                               budget=lses.SessionBudget(max_gpu_seconds=1e6, max_wall_seconds=0.5))
    check(s_wall.estimate(plan_b, item_b)["fits_budget"] is False,
          "the estimate tool says a call does not fit when the gate would refuse it for the wall clock")
    s_wall.close()
    s_bad = lses.LevelSession(h, tmp / "ses_badedit", reach="tools_only", backend="local",
                              book_path=bk, brief=brief, bars=bars, claim_design=ses_design)
    check_raises(lses.SessionRefused, lambda: s_bad.claim({"terms": [{"layer": 999, "silence": [0]}]}),
                 "a claim that cannot be priced is not started", match="cannot be priced")
    s_bad.close()

    # CLI brief / calibrate / claim --stand-in
    cli_dir = tmp / "cli_run"
    cli_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "THEODOLITE_LEVEL_BOOK": str(cli_dir / "cli_book.jsonl"),
           "THEODOLITE_EDITS_DIR": str(cli_dir / "edits")}
    py = sys.executable

    p_list = subprocess.run([py, "-I", "scripts/levelling.py", "brief", "list", "--stand-in"],
                            cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, timeout=30)
    check(p_list.returncode == 0 and "pronoun_pull" in p_list.stdout, f"CLI brief list ok (rc={p_list.returncode})")

    br_file = cli_dir / "exported_brief.json"
    p_exp = subprocess.run([py, "-I", "scripts/levelling.py", "brief", "export", "pronoun_pull",
                            "--stand-in", "--out", str(br_file)],
                           cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, timeout=30)
    check(p_exp.returncode == 0 and br_file.exists(), f"CLI brief export ok (rc={p_exp.returncode})")

    bars_file = cli_dir / "bars.json"
    p_cal = subprocess.run([py, "-I", "scripts/levelling.py", "calibrate", str(br_file),
                            "--layer", "2", "--n-cols", "2", "--n-controls", "2",
                            "--n-neutral", "2", "--n-near", "2", "--n-panel", "2",
                            "--stand-in", "--out", str(bars_file)],
                           cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, timeout=60)
    check(p_cal.returncode == 0 and bars_file.exists(),
          f"CLI calibrate --stand-in ok (rc={p_cal.returncode}, err={p_cal.stderr[:200]})")

    edit_file = cli_dir / "edit.json"
    edit_file.write_text(json.dumps({"terms": [{"layer": 2, "silence": [0, 1]}]}), encoding="utf-8")
    out_file = cli_dir / "claim_out.json"
    p_clm = subprocess.run([py, "-I", "scripts/levelling.py", "claim", str(br_file),
                            "--bars", str(bars_file), "--edit", str(edit_file),
                            "--n-target", "4", "--n-neutral", "2", "--n-near", "2", "--n-panel", "2",
                            "--controls", "4", "--match-tolerance", "20.0", "--no-prompt-arms",
                            "--stand-in", "--out", str(out_file)],
                           cwd=str(ROOT_DIR), env=env, capture_output=True, text=True, timeout=90)
    check(p_clm.returncode == 0 and "CLAIM SHEET" in p_clm.stdout and out_file.exists(),
          f"CLI claim --stand-in ok (rc={p_clm.returncode}, err={p_clm.stderr[:200]})")
    SHEETS_COLLECTED.append(p_clm.stdout)
    check(lb.verify(cli_dir / "cli_book.jsonl")["ok"], "CLI claim ledger whole")

    # The MCP server itself, started as its owner starts it (the same brief, bars and book, the design by its flags)
    # and driven over stdio: the second fresh reading on the brief, after the command line's first
    calls = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "status", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "claim", "arguments": {
            "edit": {"terms": [{"layer": 2, "silence": [2, 3]}]}, "match_tolerance": 50}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "claim", "arguments": {
            "edit": {"terms": [{"layer": 2, "silence": [2, 3]}]}}}},
    ]
    p_mcp = subprocess.run(
        [py, "-I", "backend/level_mcp.py", "--session-dir", str(cli_dir / "mcp_sess"),
         "--book", str(cli_dir / "cli_book.jsonl"), "--reach", "tools_only", "--stand-in", "--no-probe",
         "--brief", str(br_file), "--bars", str(bars_file),
         "--claim-n-target", "4", "--claim-n-neutral", "2", "--claim-n-near", "2", "--claim-n-panel", "2",
         "--claim-controls", "4", "--claim-match-tolerance", "20"],
        cwd=str(ROOT_DIR), env=env, input="".join(json.dumps(c) + "\n" for c in calls),
        capture_output=True, text=True, timeout=180,
    )
    answers = {}
    for ln in p_mcp.stdout.splitlines():
        try:
            msg = json.loads(ln)
            answers[msg["id"]] = msg["result"]
        except (ValueError, KeyError):
            pass

    def told(i: int) -> Any:
        try:
            return json.loads(answers[i]["content"][0]["text"])
        except (KeyError, ValueError, IndexError):
            return {}

    card_design = told(2).get("harness_card", {}).get("claim_design", {})
    check(p_mcp.returncode == 0 and card_design.get("match_tolerance") == 20.0 and card_design.get("n_target") == 4
          and card_design.get("controls_per_family") == 4 and card_design.get("rows") == {"small": 12, "large": 32},
          f"the MCP server's --claim-* flags fix the design, and its harness card shows it (rc={p_mcp.returncode}, "
          f"err={p_mcp.stderr[-200:]!r})")
    check(answers.get(3, {}).get("isError") is True
          and "takes no argument named match_tolerance" in answers[3]["content"][0]["text"],
          "over stdio the server refuses the driver's own matching tolerance")
    r_srv = told(4)
    SHEETS_COLLECTED.append(r_srv.get("sheet", ""))
    check(r_srv.get("verdict") in lcl.VERDICTS and r_srv.get("reading_index") == 2
          and r_srv.get("forwards_charged") == (r_srv.get("estimated") or {}).get("forwards")
          and "other sites matched on what a base pre-pass expects each to deliver, within a factor of 20" in r_srv.get("sheet", "")
          and "earlier fresh readings on this brief (1):" in r_srv.get("sheet", ""),
          f"a claim through the started server is the brief's second reading in the book, after the command line's "
          f"first, under the owner's design (verdict {r_srv.get('verdict')}, reading {r_srv.get('reading_index')})")
    check(lb.verify(cli_dir / "cli_book.jsonl")["ok"], "the book is whole after the started server's claim")


def run_section_7() -> None:
    print("Section 7: Zero kept-out words audit across all 10 Levelling files & all rendered Claim Sheets", flush=True)
    src_files = [
        "backend/level_book.py", "backend/weight_edits.py", "backend/levelling.py",
        "backend/weight_reader.py", "backend/level_sandbox.py", "backend/level_session.py",
        "backend/level_mcp.py", "backend/level_briefs.py", "backend/level_claim.py",
        "scripts/levelling.py",
    ]
    for rel in src_files:
        txt = (ROOT_DIR / rel).read_text(encoding="utf-8")
        hits = [m.group(0) for m in KEPT_OUT_RE.finditer(txt)]
        check(not hits, f"{rel} has zero kept-out words (found: {hits})")
    for i, text in enumerate(SHEETS_COLLECTED):
        hits = [m.group(0) for m in KEPT_OUT_RE.finditer(text)]
        check(not hits, f"rendered Claim Sheet / sentence #{i} has zero kept-out words (found: {hits})")


def main() -> int:
    t0 = time.perf_counter()
    use_stand_in_atlas()
    rt, _vault = stand_in()
    tok = rt.tokenizer
    h = lv.handle(rt)

    with tempfile.TemporaryDirectory(prefix="check-level-claim-") as td:
        tmp = Path(td)
        run_section_1(tok, tmp)
        run_section_2()
        run_section_3(tok, tmp)
        run_section_4(tok)
        run_section_5(h, tmp)
        run_section_6(h, tmp)
        run_section_7()

    elapsed = time.perf_counter() - t0
    print(f"\nPhase 3 CPU checks complete in {elapsed:.1f} s: {PASSED} passed, {FAILED} failed [ran on CPU stand-ins].")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
