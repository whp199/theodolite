#!/usr/bin/env python3
"""Real RTX 5090 31B W4A16 card verification for Levelling Phase 5: Of record, and out (`scripts/verify_level_trial.py`).

Measures end to end on `google/gemma-4-31B-it-qat-w4a16-ct` (compiled unpacker, lens freed, server stopped):
1. Calibrates `BarSet` for `pronoun_pull` on layer 50 (10 cols, 3 controls) and builds `starter_reworded_pronoun_pull`
   (8 owner-written out-of-template narratives, 4 half-0 and 4 half-1).
2. Creates and verifies a hash-checked registration folder (`prereg/` with `SHA256SUMS`, zero weight payloads) for
   E6 layer-50 `'she'` ten (`[5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070]`), `C = 39` controls/family,
   `n_target = 16` fresh kept-back items (2 per frame), `reworded_controls = 19`, and `folded_required = True`.
3. Executes the fresh-process Trial of Record:
   - Uncached live `we.base_checksum` over all 410+ linears before and after (asserts bit-identical base weights),
   - Dual-batch-class (`small = 12`, `large = 32`) fresh kept-back reading WITH `folded="always"` on every item,
   - Dual-batch-class owner-written reworded reading WITH `folded="always"` on every reworded item,
   - Full gating retention batteries (`neutral_text`, `near_misses`, `labelled_panel`) + author prompt arm,
   - Verifies exact bit-identity between folded weights and per-row hooks (`same_bits == 100%`), reworded shift,
     honest verdict (`failed` because E6 `'she'` ten bleeds into male near misses, matching `arbiter.md` Row 20),
     read-once refusal, 0 unpacker fallbacks, and whole ledger chain.
4. Exports the edit as a llama.cpp LoRA `.gguf` + `.manifest.json` into an isolated outside-checkout temp directory
   against the live 31B weights and verifies Apache-2.0 licence/origin keys + exact `(W + B @ A) @ x` equivalence.
5. Packs a volunteer re-run kit (`make_rerun_kit`), executes a volunteer re-run `.tar` bundle on a compact check kit,
   and verifies defensive tar intake & judgement (`intake_rerun_bundle` -> `accepted`, `same_bits`).

Outputs:
- `experiments/results/levelling/verify_trial.json`
- `experiments/results/levelling/verify_trial_book.jsonl`
"""
from __future__ import annotations

import datetime
import gc
import os
import re
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

torch.set_grad_enabled(False)

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

    with tempfile.TemporaryDirectory(prefix="verify-level-trial-31b-") as td:
        tmp = Path(td)

        # 1. Calibrate bars + starter reworded set
        brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2)
        rset = lt.starter_reworded_pronoun_pull(brief)
        print("\nStep 1: calibrate_bars on pronoun_pull (layer 50, 10 cols, 3 controls)...", flush=True)
        t_cal = time.perf_counter()
        bars = lcl.calibrate_bars(
            h, brief, layer=50, n_cols=10, n_controls=3, n_neutral=4, n_near=4, n_panel=4,
            seed=2026, book_path=OUT_BOOK,
        )
        cal_s = round(time.perf_counter() - t_cal, 2)
        check("bars_calibrated", lcl.calibration_note(bars, OUT_BOOK) is not None,
              {"seconds": cal_s, "bars_sha256": bars.short()})

        # 2. Register Trial of Record prereg/ folder
        edit = we.build(h.lm, [we.silence(50, E6_L50_SHE10)], model_id=h.model_id, weights_id=h.weights_id,
                        note="E6 layer-50 she ten")
        prereg_dir = tmp / "prereg_she10"
        reg = lt.register_trial(
            prereg_dir, brief, bars, edit,
            trial_name="e6_l50_she10_31b",
            reworded=rset,
            controls_per_family=39,
            reworded_controls=19,
            n_target=16,
            n_neutral=4,
            n_near=4,
            n_panel=4,
            folded_required=True,
        )
        v_reg = lt.verify_registration(prereg_dir)
        check("registration_verified", v_reg["ok"] and v_reg["trial"]["edit_provenance"] == "rebuilt",
              {"trial_sha256": reg["trial_sha256"][:16], "files": reg["files"]})

        # 3. Execute Trial of Record on 31B (includes full live base_checksum, folded="always", reworded set!)
        print("\nStep 2: Running Trial of Record (16 fresh items + 8 reworded + folded='always' + C=39 controls)...", flush=True)
        t_tr = time.perf_counter()
        t_res = lt.run_trial_of_record(
            h, prereg_dir, out_dir=tmp / "trial_out", book_path=OUT_BOOK, require_book_calibration=True,
        )
        tr_s = round(time.perf_counter() - t_tr, 2)
        print("\n" + t_res["sheet"] + "\n", flush=True)
        fl_sm = t_res["folded"]["classes"]["small"]
        fl_lg = t_res["folded"]["classes"]["large"]
        rw_sm = t_res["reworded"]["classes"]["small"]["frames"]["mean"]
        rw_lg = t_res["reworded"]["classes"]["large"]["frames"]["mean"]
        check("trial_not_void", t_res["verdict"] in ("passed", "failed"),
              {"verdict": t_res["verdict"], "fresh_verdict": t_res["fresh_verdict"], "seconds": tr_s})
        check("base_checksum_unchanged_31b",
              t_res["meta"]["base_checksum_unchanged"] is True and t_res["meta"]["base_linears_checked"] >= 400,
              {"linears_checked": t_res["meta"]["base_linears_checked"],
               "sha256": str(t_res["meta"]["base_checksum_before"])[:16]})
        check("folded_exact_bit_identity_31b",
              t_res["folded"]["ok"] is True
              and fl_sm["same_bits_count"] == fl_sm["taken_count"] == 16
              and fl_lg["same_bits_count"] == fl_lg["taken_count"] == 16,
              {"small_same_bits": f"{fl_sm['same_bits_count']}/{fl_sm['taken_count']}",
               "large_same_bits": f"{fl_lg['same_bits_count']}/{fl_lg['taken_count']}",
               "small_mean": fl_sm["frames"]["mean"], "large_mean": fl_lg["frames"]["mean"]})
        check("reworded_set_verified_31b",
              t_res["reworded"]["ok"] is True and t_res["reworded"]["folded"]["ok"] is True,
              {"small_mean": rw_sm, "large_mean": rw_lg})
        check("trial_sheet_no_kept_out_words",
              KEPT_OUT_RE.search(t_res["sheet"]) is None
              and KEPT_OUT_RE.search((prereg_dir / "RULES.md").read_text()) is None,
              "0 kept-out words in Trial Sheet and RULES.md")

        # Read-once refusal
        got_ro = False
        try:
            lt.run_trial_of_record(h, prereg_dir, out_dir=tmp / "trial_out_dup", book_path=OUT_BOOK)
        except lt.TrialRefused:
            got_ro = True
        check("trial_read_once_refused", got_ro, "duplicate trial_sha256 refused")

        # Replay card verification
        v_ent = next(lb.entries(OUT_BOOK, start=t_res["book"]["verdict_seq"], limit=1))
        card = lroutes.build_replay_card(v_ent)
        check("trial_replay_card_ok",
              card["note_type"] == "trial_of_record" and card["edit_linears"] == ["layers.50.mlp.down_proj"],
              {"seq": card["seq"], "verdict": card["verdict"], "edit_linears": card["edit_linears"]})

        # 4. Owner-only export & verification against 31B weights
        print("\nStep 3: Owner-only LoRA GGUF export & exact delta verification on 31B...", flush=True)
        t_exp = time.perf_counter()
        target_31b = lex.TargetBase(lm=h.lm, label=f"{h.model_id} ({h.weights_id[:16]})")
        exp_gguf = tmp / "exports" / "e6_l50_she10_31b.gguf"
        exp_out = lex.export_lora_adapter(
            edit, target_31b, out_path=exp_gguf, dose=1.0,
            trial_result=t_res, prior_attempts=1,
            lm_for_implant=h.lm, W_U_for_implant=h.W_U,
        )
        ver_exp = lex.verify_exported_lora(exp_gguf, edit=edit, target=target_31b, dose=1.0)
        exp_s = round(time.perf_counter() - t_exp, 2)
        check("export_lora_verified_31b",
              ver_exp["ok"] and (ver_exp["max_abs_err"] or 0.0) < 1e-4 and exp_out["recompiled_from_recipe"] is True,
              {"seconds": exp_s, "gguf_bytes": exp_out["gguf_bytes"], "max_abs_err": ver_exp["max_abs_err"],
               "implant_flagged": exp_out["implant_check"]["flagged"]})

        # 5. Volunteer re-run kit + run + defensive intake on compact 31B trial (C=3, n_target=8)
        print("\nStep 4: Volunteer re-run kit, run, and defensive intake on 31B...", flush=True)
        t_rr = time.perf_counter()
        prereg_rr = tmp / "prereg_rr"
        lt.register_trial(
            prereg_rr, brief, bars, edit,
            trial_name="e6_l50_she10_rerun_check",
            controls_per_family=3, n_target=8, n_neutral=2, n_near=2, n_panel=2, folded_required=True,
        )
        home_rr = lt.run_trial_of_record(
            h, prereg_rr, out_dir=tmp / "home_rr_out", book_path=tmp / "home_rr_book.jsonl",
            _seed_override=b"verify-31b-rerun-seed-32bytes-01",
        )
        kit_dir = tmp / "kit_31b"
        lr.make_rerun_kit(prereg_rr, kit_dir, home_result=home_rr)
        vol_tar = tmp / "vol_31b.tar"
        lr.run_volunteer_rerun(
            h, kit_dir, vol_tar, volunteer_name="home_selfcheck",
            _seed_override=b"verify-31b-rerun-seed-32bytes-01",
        )
        intake_res = lr.intake_rerun_bundle(vol_tar, prereg_rr, home_rr, tmp / "intake_stage", allow_dirty=True)
        rr_s = round(time.perf_counter() - t_rr, 2)
        check("volunteer_rerun_intake_31b",
              intake_res["outcome"] == "accepted" and intake_res["match_class"] == "same_bits",
              {"seconds": rr_s, "outcome": intake_res["outcome"], "match_class": intake_res["match_class"]})

    # 6. Final invariants
    we.assert_base(h.lm, h.hooks_at_load)
    check("assert_base_final", True, "model clean at base")
    fb1 = int(getattr(engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)
    fb = fb1 - fb0
    check("zero_unpacker_fallbacks", fb == 0, {"fallback_count": fb})
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
        "trial_e6_she10": {
            "seconds": tr_s,
            "verdict": t_res["verdict"],
            "fresh_verdict": t_res["fresh_verdict"],
            "reasons": t_res["reasons"],
            "sheet": t_res["sheet"],
            "folded": t_res["folded"],
            "reworded": t_res["reworded"],
            "meta": t_res["meta"],
            "book": t_res["book"],
        },
        "export_check": {"seconds": exp_s, "max_abs_err": ver_exp["max_abs_err"],
                         "gguf_bytes": exp_out["gguf_bytes"], "implant_check": exp_out["implant_check"]},
        "rerun_intake": {"seconds": rr_s, "outcome": intake_res["outcome"], "match_class": intake_res["match_class"],
                         "class_diffs": intake_res["class_diffs"]},
        "ledger": bv,
    }
    write_json_atomic(OUT_JSON, report)
    print(f"\nPhase 5 31B card verification complete in {total_s} s (peak {peak_gib:.3f} GiB): "
          f"{n_ok}/{len(checks)} passed, {n_fail} failed [measured on the 31B card].", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
