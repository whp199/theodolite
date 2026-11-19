#!/usr/bin/env python3
"""Real RTX 5090 31B W4A16 card verification for Levelling Phase 3: The Claim Gate (`scripts/verify_level_claim.py`).

Measures live on `google/gemma-4-31B-it-qat-w4a16-ct` (compiled unpacker, lens freed):
1. `calibrate_bars` on `pronoun_pull_brief` (layer 50, 10 columns, 3 controls) -> hashed `BarSet` recorded in
   `experiments/results/levelling/verify_claim_book.jsonl`.
2. Fresh claim reading #1 of `m = 2` (`C = 39` matched other-site & `C = 39` same-site controls in BOTH small=12 and
   large=32 batch classes, 8 kept-back target items balanced across 4 half-0 & 4 half-1 frames, neutral_text /
   near_misses / labelled_panel gating batteries, and author prompt arm):
   - Candidate: E6 layer-50 top-10 `'she'` neurons `[5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070]`.
3. Fresh claim reading #2 of `m = 2` (`C = 39` per family, both batch classes, full batteries):
   - Candidate: E6 layer-50 7 `'she'`-only neurons `[266, 121, 11503, 11724, 21151, 4165, 2070]` (arbiter.md Row 24
     negative control), verifying reading #1 appears on reading #2's Claim Sheet under earlier readings.
4. Read-once refusal (re-submitting edit #1) and `m = 2` budget refusal (submitting a 3rd edit) against the live book.
5. Real 31B `compare_settings_audit` + `price_written_reading` on `enumeration_loop_brief` (`max_new=64` thinking-off
   vs thinking-on pilot pricing a full `C = 39` written claim).
6. Invariants: 0 unpacker fallbacks, `assert_base` clean throughout, ledger chain whole, zero kept-out words in every
   generated Claim Sheet and claim sentence.

Outputs:
- `experiments/results/levelling/verify_claim.json`
- `experiments/results/levelling/verify_claim_book.jsonl`
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

torch.set_grad_enabled(False)

from backend import engine_runtime  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend.census import write_json_atomic  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402

OUT_DIR = ROOT_DIR / "experiments" / "results" / "levelling"
OUT_JSON = OUT_DIR / "verify_claim.json"
OUT_BOOK = OUT_DIR / "verify_claim_book.jsonl"

E6_L50_SHE10 = [5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070]
E6_L50_SHE_ONLY7 = [266, 121, 11503, 11724, 21151, 4165, 2070]

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
        print("ERROR: Theodolite server answers on :8765; stop it before running verify_level_claim.py.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA unavailable.", file=sys.stderr)
        return 2
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU: {torch.cuda.get_device_name(0)} ({free_gib:.2f} GiB free)", flush=True)
    if free_gib < 22.0:
        print(f"ERROR: Only {free_gib:.2f} GiB free; need >= 22.0 GiB.", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_BOOK.exists():
        OUT_BOOK.unlink()

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
    torch.cuda.reset_peak_memory_stats()

    # Build pronoun_pull starter brief with m=2 -> C=39 controls per family
    brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2)
    check("brief_constructed", brief.default_controls == 39, {"brief_id": brief.short(), "C": brief.default_controls})

    # 1. Real 31B calibration of bars on layer 50 (10 columns, 3 controls)
    print("\nStep 1: calibrate_bars on pronoun_pull (layer 50, 10 cols, 3 controls)...", flush=True)
    t_cal = time.perf_counter()
    bars = lcl.calibrate_bars(
        h, brief, layer=50, n_cols=10, n_controls=3,
        n_neutral=4, n_near=4, n_panel=4, seed=2026, book_path=OUT_BOOK,
    )
    cal_s = round(time.perf_counter() - t_cal, 2)
    hist_cal = lcl.brief_history(brief.id, OUT_BOOK)
    check("calibrate_bars_hash_matches_book",
          len(hist_cal["calibrations"]) == 1 and hist_cal["calibrations"][0]["bars_sha256"] == bars.id,
          {"seconds": cal_s, "bars_sha256": bars.short(), "neutral_kl_max": bars.neutral_kl_max,
           "near_miss_abs_max": bars.near_miss_abs_max, "panel_own_drop_max": bars.panel_own_drop_max})
    we.assert_base(h.lm, h.hooks_at_load)

    # 2. Fresh claim reading #1 of 2: E6 L50 she_10 with FULL C=39 per family!
    print("\nStep 2: Fresh claim #1 (E6 L50 she_10, C=39 per family, both batch classes)...", flush=True)
    edit1 = {"terms": [{"layer": 50, "silence": E6_L50_SHE10}]}
    t_c1 = time.perf_counter()
    res1 = lcl.run_fresh_claim(
        h, brief, bars, edit1,
        forecast={"expected_verdict_note": "E6 L50 she10 known-answer bleed check on pronoun_pull"},
        selection_mode="free_pick", practice_count=1,
        n_target=8, n_neutral=4, n_near=4, n_panel=4,
        book_path=OUT_BOOK,
    )
    c1_s = round(time.perf_counter() - t_c1, 2)
    print("\n" + res1["sheet"] + "\n", flush=True)
    check("claim_1_not_void", res1["verdict"] != "void",
          {"verdict": res1["verdict"], "seconds": c1_s,
           "large_mean": res1["driver_record"]["summary"]["large"]["mean"],
           "small_mean": res1["driver_record"]["summary"]["small"]["mean"],
           "near_miss_abs": res1["driver_record"]["batteries"]["near_misses"]["abs_effect"]})
    check("claim_1_reg_before_verdict",
          res1["book"]["registration_seq"] < res1["book"]["verdict_seq"],
          res1["book"])
    check("claim_1_driver_redacted",
          "Finish the story:" not in json.dumps(res1["driver_record"])
          and "Write the next words:" not in json.dumps(res1["driver_record"]),
          "no raw fresh item text in driver_record")
    check("claim_1_no_kept_out_words",
          KEPT_OUT_RE.search(res1["sheet"]) is None
          and (res1["claim_sentence"] is None or KEPT_OUT_RE.search(res1["claim_sentence"]) is None),
          "0 kept-out words in Claim Sheet #1")
    we.assert_base(h.lm, h.hooks_at_load)

    # 3. Fresh claim reading #2 of 2: E6 L50 she_only_7 (arbiter Row 24 negative control!)
    print("\nStep 3: Fresh claim #2 (E6 L50 she_only_7 negative control, C=39 per family)...", flush=True)
    edit2 = {"terms": [{"layer": 50, "silence": E6_L50_SHE_ONLY7}]}
    t_c2 = time.perf_counter()
    res2 = lcl.run_fresh_claim(
        h, brief, bars, edit2,
        forecast={"expected_verdict_note": "arbiter Row 24: 7 she-only neurons negative control"},
        selection_mode="free_pick", practice_count=1,
        n_target=8, n_neutral=4, n_near=4, n_panel=4,
        book_path=OUT_BOOK,
    )
    c2_s = round(time.perf_counter() - t_c2, 2)
    print("\n" + res2["sheet"] + "\n", flush=True)
    check("claim_2_not_void", res2["verdict"] != "void",
          {"verdict": res2["verdict"], "seconds": c2_s,
           "large_mean": res2["driver_record"]["summary"]["large"]["mean"],
           "small_mean": res2["driver_record"]["summary"]["small"]["mean"]})
    check("claim_2_prints_prior_verdict",
          "earlier fresh readings on this brief (1):" in res2["sheet"]
          and res1["driver_record"]["edit_id"][:12] in res2["sheet"],
          "reading #2 sheet records reading #1")
    check("claim_2_no_kept_out_words",
          KEPT_OUT_RE.search(res2["sheet"]) is None
          and (res2["claim_sentence"] is None or KEPT_OUT_RE.search(res2["claim_sentence"]) is None),
          "0 kept-out words in Claim Sheet #2")
    we.assert_base(h.lm, h.hooks_at_load)

    # 4. Read-once refusal & m-budget refusal against live 31B ledger
    print("\nStep 4: Read-once and m-budget refusals on live book...", flush=True)
    got_ro = False
    try:
        lcl.run_fresh_claim(h, brief, bars, edit1, book_path=OUT_BOOK)
    except lcl.ClaimRefused as exc:
        got_ro = "read once" in str(exc).lower()
    check("read_once_refused", got_ro, "duplicate edit1 refused")

    got_m = False
    try:
        lcl.run_fresh_claim(h, brief, bars, {"terms": [{"layer": 50, "silence": [0, 1, 2]}]}, book_path=OUT_BOOK)
    except lcl.ClaimRefused as exc:
        got_m = "already spent all 2" in str(exc).lower()
    check("m_budget_refused", got_m, "3rd fresh reading refused when m=2")

    # 5. Live 31B compare_settings_audit + written claim pricing on enumeration_loop_brief
    print("\nStep 5: compare_settings_audit & written claim pricing on enumeration_loop (max_new=64)...", flush=True)
    wbrief = lbr.enumeration_loop_brief(h.tokenizer, max_fresh_readings=2)
    w_drawn = wbrief.generator.draw("kept_back", 2, seed_bytes=b"verify_31b_written_audit", tokenizer=h.tokenizer)
    w_edit = we.build(h.lm, [we.silence(36, [0, 1])], model_id=h.model_id, weights_id=h.weights_id)
    prev_bk = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(OUT_BOOK)
    try:
        t_sa = time.perf_counter()
        sa = lcl.compare_settings_audit(h, wbrief, w_edit, w_drawn, max_new_override=64)
        sa_s = round(time.perf_counter() - t_sa, 2)
        # Also pull the thinking_on reading entry from book to price a full C=39 written claim
        th_seq = sa["book_seqs"][-1]
        th_entry = next(lb.entries(OUT_BOOK, start=th_seq, limit=1))
        priced = lcl.price_written_reading(th_entry["body"], n_items=8, n_seeds=2, n_controls=39, doses_count=2, rows_per_batch=12)
    finally:
        if prev_bk is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = prev_bk
    check("settings_audit_ok",
          sa["thinking_off"]["status"] == "ok" and sa["thinking_on"]["status"] == "ok",
          {"seconds": sa_s, "note": sa["note"], "priced_C39_minutes": priced["estimated_minutes"]})

    # 6. Final invariants
    we.assert_base(h.lm, h.hooks_at_load)
    check("assert_base_final", True, "model clean at base")
    fb1 = int(getattr(engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)
    fb = fb1 - fb0
    check("zero_unpacker_fallbacks", fb == 0, {"fallback_count": fb, "before": fb0, "after": fb1})
    bv = lb.verify(OUT_BOOK)
    check("ledger_whole", bv["ok"], {"entries": bv["entries"], "head": bv["head"][:16]})
    peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    total_s = round(time.perf_counter() - t_all, 2)
    check("peak_memory_recorded", peak_gib > 0.0, {"peak_gib": peak_gib, "total_seconds": total_s})

    n_ok = sum(1 for c in checks if c["ok"])
    n_fail = len(checks) - n_ok
    report = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": "measured on the 31B card",
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
        "calibration": {
            "seconds": cal_s,
            "bars": bars.to_dict(),
            "bars_sha256": bars.id,
        },
        "claim_1_she10": {
            "seconds": c1_s,
            "verdict": res1["verdict"],
            "reasons": res1["judged"]["reasons"],
            "claim_sentence": res1["claim_sentence"],
            "sheet": res1["sheet"],
            "summary": res1["driver_record"]["summary"],
            "batteries": res1["driver_record"]["batteries"],
            "cheaper_knobs": res1["driver_record"]["cheaper_knobs"],
            "book": res1["book"],
        },
        "claim_2_she_only7": {
            "seconds": c2_s,
            "verdict": res2["verdict"],
            "reasons": res2["judged"]["reasons"],
            "claim_sentence": res2["claim_sentence"],
            "sheet": res2["sheet"],
            "summary": res2["driver_record"]["summary"],
            "batteries": res2["driver_record"]["batteries"],
            "cheaper_knobs": res2["driver_record"]["cheaper_knobs"],
            "book": res2["book"],
        },
        "written_settings_audit": {"seconds": sa_s, "audit": sa, "priced_full_C39": priced},
        "ledger": bv,
    }
    write_json_atomic(OUT_JSON, report)
    print(f"\nPhase 3 31B card verification complete in {total_s} s (peak {peak_gib:.3f} GiB): "
          f"{n_ok}/{len(checks)} passed, {n_fail} failed [measured on the 31B card].", flush=True)
    print(f"Wrote {OUT_JSON} and {OUT_BOOK}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
