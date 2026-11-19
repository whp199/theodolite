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


def run_section_2() -> None:
    print("Section 2: Statistics (house_bar, frame_statistics, control_standing, written strata, pricing)", flush=True)
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
    check(s_lead is not None and "selection not matched, 7 read in practice" in s_lead,
          "free_pick claim_sentence carries 'selection not matched, 7 read in practice'")
    check("This is a lead, not a finding of record." in (s_lead or ""), "claim_sentence is bounded as a lead")

    # 2. True lead (declared_rule with sham_target)
    sham_cls = _synth_class(h0_vals=[-1.2, -1.0], h1_vals=[-1.1, -0.9], sham_ctls=[0.08] * 39)
    j_decl = lcl.judge_claim(brief=brief, bars=bars, classes={"small": sham_cls, "large": sham_cls},
                             batteries=_clean_batteries(), selection_mode="declared_rule")
    s_decl = lcl.claim_sentence(brief, j_decl, meta)
    sh_decl = lcl.render_sheet(brief, bars, j_decl, meta, s_decl)
    SHEETS_COLLECTED.extend([sh_decl, s_decl or ""])
    check(j_decl["verdict"] == "lead" and "sham-target" in (s_decl or ""), "declared_rule with passing sham -> lead")

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

    # 8. Live written_loop claim with chunked controls + written_strata + settings_audit
    wbrief_base = lbr.enumeration_loop_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=4)
    wbrief = lbr.Brief.from_dict({
        **wbrief_base.to_dict(),
        "written_config": {"max_new_tokens": 12, "temperature": 1.0, "top_p": 0.95, "seeds": [1]},
    })
    wbars = lcl.make_bar_set(wbrief.id, neutral_kl_max=0.5, near_miss_abs_max=0.5, panel_own_drop_max=0.5, panel_kl_max=0.5)
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
    sa_cpu = wres["meta"]["settings_audit"]
    check(
        sa_cpu is not None
        and sa_cpu["thinking_off"]["status"] == "ok"
        and sa_cpu["thinking_on"]["status"] == "ok",
        f"compare_settings=True populates ok settings_audit (got {sa_cpu})",
    )
    check("written strata (" in wres["sheet"] and "base settings audit (" in wres["sheet"],
          "written_loop Claim Sheet prints written strata and base settings audit")
    check(lb.verify(bk)["ok"], "level book whole after written_loop claim")
    we.assert_base(h.lm, h.hooks_at_load)


def run_section_6(h: lv.Handle, tmp: Path) -> None:
    print("Section 6: LevelSession + MCP claim gate & CLI subcommands", flush=True)
    brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=4)
    bars = lcl.make_bar_set(brief.id, neutral_kl_max=0.5, near_miss_abs_max=0.5, panel_own_drop_max=0.5, panel_kl_max=0.5)

    # Blinded canary session MUST NOT expose claim tool
    ses_blind = lses.LevelSession(
        h, tmp / "ses_blind", reach="tools_only", backend="local",
        book_path=tmp / "blind_book.jsonl", brief=brief, bars=bars, canary=True,
    )
    check("claim" not in ses_blind.allowed_tools(), "blinded canary session hides claim tool")
    ses_blind.close()

    # Bound brief + bars session via MCP dispatch
    bk = tmp / "mcp_claim_book.jsonl"
    ses = lses.LevelSession(
        h, tmp / "ses_claim", reach="tools_only", backend="local",
        book_path=bk, brief=brief, bars=bars,
    )
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

    r_claim = lmcp.dispatch_tool(ses, "claim", {
        "edit": {"terms": [{"layer": 2, "silence": [0, 1]}]},
        "n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2,
        "controls_override": 4, "match_tolerance": 20.0, "run_prompt_arms": False,
    })
    SHEETS_COLLECTED.append(r_claim["sheet"])
    check(r_claim["verdict"] in lcl.VERDICTS and r_claim["reading_index"] == 1, "MCP claim returns redacted verdict record")
    check("Finish the story:" not in json.dumps(r_claim), "MCP claim leaks zero fresh kept-back text")

    # Verify book_view redacts items of fresh claim readings too!
    bv = lmcp.dispatch_tool(ses, "book_view", {"tail": 15})
    check("Finish the story:" not in json.dumps(bv), "MCP book_view leaks zero fresh kept-back text")
    ses.close()
    check(lb.verify(bk)["ok"], "MCP claim session ledger whole")

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
