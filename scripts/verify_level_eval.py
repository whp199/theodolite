#!/usr/bin/env python3
"""Real RTX 5090 31B W4A16 card verification for Levelling Phase 6 (`scripts/verify_level_eval.py`).

Measures live on `google/gemma-4-31B-it-qat-w4a16-ct` (compiled unpacker, lens freed) against the real
23.26 GB checkpoint and real 60x21504 31B atlas index:
1. Read-only `AtlasReader` + live `bwrap` `/atlas` containment on the real 31B atlas index (label-gate provenance,
   sub-second word search, neuron card, and verified read-only mount alongside `/weights`).
2. Cheaper-knob kernel & fresh knob claim gate on `pronoun_pull`:
   - Per-row arithmetic check comparing `neuron_mul(0)` steering vs `weight_edits.silence` on E6 L50 `'she'` ten.
   - Fresh knob claim #1: blunt `logit_bias` on `' she'` (verifying `works but harms` via near-miss bleed).
   - Fresh knob claim #2: transient `neuron_mul(0)` steering on E6 L50 `'she'` ten with matched random steering controls.
   - Fresh knob claim #3: system prompt knob on the exact same kept-back frames and calibrated `BarSet`.
   - Read-once and `m = 3` budget refusal checks.
3. Multi-session Phase 6 comparison study (5 unpooled arms x 3 independent sessions = 15 isolated side books):
   - Reach ladder: `behaviour` -> `activations` -> `weights` -> `atlas` (`mode="weight_edit"`).
   - Matched-budget cheaper-knob competitor: `weights` (`mode="knobs_only"`).
   - Model-free `load_study_from_book` verification of `study_book.jsonl` and all 15 session books.
4. Blinded planted-fault (both `with_healthy_reference=True` and `False`) & canary benchmark on 31B layer 50,
   verifying identical harness cards, canary abstention, and exact additive repair recovery (`recovery_fraction == 1.0`).
5. Future-ready protocol checks: <= 600-word cold-driver pages (zero site spoilers) and live 31B <= 200-word rebuild test.
6. Invariants: 0 unpacker fallbacks, `assert_base` clean throughout, ledger chain whole, zero kept-out words across
   every rendered knob sheet, study report, cold-driver page, and rebuild text.

Outputs:
- `experiments/results/levelling/verify_eval.json`
- `experiments/results/levelling/verify_eval_book.jsonl`
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

torch.set_grad_enabled(False)

from backend import engine_runtime  # noqa: E402
from backend import level_atlas as la  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_drivers as ldrv  # noqa: E402
from backend import level_eval as lev  # noqa: E402
from backend import level_knobs as lk  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend.census import write_json_atomic  # noqa: E402
from backend.runtime_factory import checkpoint_snapshot, make_runtime  # noqa: E402

OUT_DIR = ROOT_DIR / "experiments" / "results" / "levelling"
OUT_JSON = OUT_DIR / "verify_eval.json"
OUT_BOOK = OUT_DIR / "verify_eval_book.jsonl"

E6_L50_SHE10 = [5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070]

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)


def _port_open(port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _find_atlas_index() -> Optional[Path]:
    env_p = os.environ.get("THEODOLITE_ATLAS_DIR")
    candidates = [
        Path(env_p) if env_p else None,
        ROOT_DIR / "atlas_data",
        Path("/home/eric/bin/ai/theodolite2/atlas_data"),
    ]
    for cand in candidates:
        if cand is None:
            continue
        try:
            return la.resolve_atlas_index_dir(cand)
        except la.AtlasReaderError:
            continue
    return None


def _commit_now() -> Dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=15)
        return {"commit": head.stdout.strip() or None if head.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None}


def main() -> int:
    if _port_open(8765):
        print("ERROR: Theodolite server answers on :8765; stop it before running verify_level_eval.py.", file=sys.stderr)
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
    os.environ[lb.BOOK_ENV] = str(OUT_BOOK)

    checks: List[Dict[str, Any]] = []
    docs_to_audit: List[str] = []

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
    rest_gib = round(torch.cuda.memory_allocated(h.device) / 2**30, 3)
    print(f"Loaded {h.model_id} in {load_s} s (rest {rest_gib} GiB, weights {str(h.weights_id)[:16]}).", flush=True)
    we.assert_base(h.lm, h.hooks_at_load)
    torch.cuda.reset_peak_memory_stats()

    snap = checkpoint_snapshot(h.model_id.split("@", 1)[0])
    check("checkpoint_snapshot_found", snap is not None and snap.is_dir(), str(snap))
    atlas_idx = _find_atlas_index()
    check("atlas_index_found", atlas_idx is not None and atlas_idx.is_dir(), str(atlas_idx))

    record: Dict[str, Any] = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **_commit_now(),
        "environment": "measured on the 31B card",
        "gpu": torch.cuda.get_device_name(0),
        "model_id": h.model_id,
        "weights_id": h.weights_id,
        "load_seconds": load_s,
        "rest_gib": rest_gib,
        "checks": checks,
    }

    with tempfile.TemporaryDirectory(prefix="verify_level_eval_31b_") as td:
        tmp = Path(td)

        # ===================================================================== 1. Real 31B AtlasReader & bwrap /atlas
        print("\nStep 1: Real 31B AtlasReader & live bwrap /atlas containment...", flush=True)
        atlas_meas: Dict[str, Any] = {}
        if atlas_idx is not None:
            t_a0 = time.perf_counter()
            with la.AtlasReader(atlas_idx) as ar:
                ainfo = ar.info()
                res_she = ar.search_words("she", top_k=12, trusted_only=True)
                card_50 = ar.neuron_card(50, 5313, top_wiring=4)
                card_10 = ar.neuron_card(10, 0, top_wiring=0)
            atlas_s = round(time.perf_counter() - t_a0, 3)
            atlas_meas = {
                "query_seconds": atlas_s,
                "n_layers": ainfo["n_layers"],
                "n_neurons": ainfo["n_neurons"],
                "vocab_size": ainfo["vocab_size"],
                "she_trusted_total_hits": res_she["total_hits"],
                "she_top3": [(x["layer"], x["neuron"], x["shared_top10"]) for x in res_she["hits"][:3]],
                "L50_N5313_trusted": card_50["label_trust_pos"]["trusted"],
                "L50_N5313_shared_top10": card_50["label_trust_pos"]["shared_top10"],
                "L10_N0_trusted": card_10["label_trust_pos"]["trusted"],
                "L10_N0_reason": card_10["label_trust_pos"]["reason"],
            }
            check(
                "atlas_reader_31b_live",
                ainfo["n_layers"] == 60
                and ainfo["n_neurons"] == 21504
                and ainfo["vocab_size"] == 262144
                and res_she["total_hits"] >= 10
                and card_50["label_trust_pos"]["trusted"] is True
                and card_10["label_trust_pos"]["trusted"] is False,
                atlas_meas,
            )

            # Live bwrap containment probe + cell reading BOTH /weights (23.26 GB) and /atlas (1.5 GB) read-only!
            s_atl = lses.LevelSession(
                h,
                tmp / "sess_atlas_probe",
                reach="atlas",
                mode="weight_edit",
                backend="local",
                budget=lses.SessionBudget(max_cell_seconds=45.0, max_cell_mem_gib=4.0, min_controls=1),
                book_path=tmp / "atlas_probe_book.jsonl",
                weights_dir=snap,
                atlas_dir=atlas_idx,
                probe_on_start=True,
            )
            cell_atl = s_atl.cell(
                """
import json
from pathlib import Path
w_info = W.info()
a_info = A.info()
hits = A.search_words("she", top_k=5, trusted_only=True)["hits"]
ro_ok = False
try:
    (Path("/atlas") / "should_fail.tmp").write_text("x")
except OSError:
    ro_ok = True
print("ATL_CELL:" + json.dumps({
    "w_layers": w_info["n_layers"],
    "a_layers": a_info["n_layers"],
    "hits": len(hits),
    "ro_ok": ro_ok,
    "sci_leaked": Path("/atlas/scientist").exists(),
    "book_leaked": Path("/atlas/level_book.jsonl").exists(),
}))
"""
            )
            s_atl.close()
            atl_out: Dict[str, Any] = {}
            for ln in str(cell_atl.get("stdout") or "").splitlines():
                if ln.startswith("ATL_CELL:"):
                    atl_out = json.loads(ln[9:])
            atlas_meas["bwrap_cell"] = {"ok": cell_atl["ok"], "seconds": cell_atl["seconds"], **atl_out}
            check(
                "bwrap_atlas_plus_weights_31b",
                bool(cell_atl["ok"])
                and atl_out.get("w_layers") == 60
                and atl_out.get("a_layers") == 60
                and atl_out.get("hits", 0) >= 5
                and atl_out.get("ro_ok") is True
                and atl_out.get("sci_leaked") is False
                and atl_out.get("book_leaked") is False,
                atlas_meas["bwrap_cell"],
            )
        record["atlas_31b"] = atlas_meas

        # ===================================================================== 2. Cheaper-knob kernel & claims on 31B
        print("\nStep 2: Calibration + Cheaper-knob arithmetic & fresh knob claims on 31B pronoun_pull...", flush=True)
        brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=3)
        t_cal = time.perf_counter()
        bars = lcl.calibrate_bars(
            h,
            brief,
            layer=50,
            n_cols=10,
            n_controls=3,
            n_neutral=4,
            n_near=4,
            n_panel=4,
            seed=2026,
            book_path=OUT_BOOK,
        )
        cal_s = round(time.perf_counter() - t_cal, 2)
        check("calibrate_bars_31b", True, {"seconds": cal_s, "bars_sha256": bars.short()})

        # 2a. Per-row equivalence check: neuron_mul(0) vs weight_edits.silence on E6_L50_SHE10
        eq_items = {
            "eq0": lv.make_item(h.tokenizer, "Answer briefly: Alice called Bob because", " she wanted to help.", name="eq0"),
            "eq1": lv.make_item(h.tokenizer, "Answer briefly: Clara told David that", " she finished the work.", name="eq1"),
        }
        plan_w = {
            "name": "equiv_weight_e6",
            "kind": "paired",
            "edit": {"terms": [{"layer": 50, "silence": E6_L50_SHE10}]},
            "items": ["eq0", "eq1"],
            "doses": [1.0],
            "rows": {"small": 12, "large": 16},
            "controls": {"family": "other_sites", "count": 1, "seed": 7},
            "readouts": brief.readouts,
        }
        r_w = lv.read(h, plan_w, eq_items)
        equiv_seq = int(r_w["book"]["seq"])
        w_eff = -float(r_w["summary"]["classes"]["small"]["arms"][lv._key("edit", 1.0)]["odds"]["mean_over_items"])

        k_mul_spec = {"steering": [{"kind": "neuron_mul", "layer": 50, "neurons": E6_L50_SHE10, "factor": 0.0}]}
        r_km = lk.read_knob_practice(
            h,
            k_mul_spec,
            eq_items,
            readouts=brief.readouts,
            metric="odds",
            direction=-1,
            controls=1,
            rows={"small": 12, "large": 16},
            seed=7,
            book_path=OUT_BOOK,
            name="equiv_knob_mul_e6",
        )
        k_eff = float(r_km["small"]["directed_mean"])
        equiv_gap = abs(w_eff - k_eff)
        check(
            "knob_mul_matches_weight_silence_31b",
            r_w["status"] == "ok" and r_km["status"] == "ok" and equiv_gap < 0.08,
            {"weight_directed": round(w_eff, 5), "knob_mul_directed": round(k_eff, 5), "abs_gap": round(equiv_gap, 5)},
        )

        # 2b. Fresh knob claim #1: blunt logit_bias on ' she' -> should trigger 'works but harms' (near-miss bleed!)
        she_tid = int(brief.readouts["odds"]["toward"][0])
        k_bias_spec = {"decoding": {"logit_bias": {str(she_tid): -4.0}}}
        t_k1 = time.perf_counter()
        kc1 = lk.run_fresh_knob_claim(
            h,
            brief,
            bars,
            k_bias_spec,
            book_path=OUT_BOOK,
            n_target=8,
            n_neutral=4,
            n_near=4,
            n_panel=4,
            rows={"small": 12, "large": 32},
            controls_override=0,
        )
        k1_s = round(time.perf_counter() - t_k1, 2)
        docs_to_audit.append(kc1["sheet"])
        check(
            "knob_claim_1_logit_bias_harms",
            kc1["verdict"] == "works but harms",
            {
                "seconds": k1_s,
                "verdict": kc1["verdict"],
                "small_dir": kc1["summary"]["small_directed"],
                "large_dir": kc1["summary"]["large_directed"],
                "near_miss_abs": kc1["summary"]["batteries"]["near_miss_abs"],
            },
        )

        # 2c. Fresh knob claim #2: transient neuron_mul(0) on E6 L50 'she' ten with matched random steering controls
        t_k2 = time.perf_counter()
        kc2 = lk.run_fresh_knob_claim(
            h,
            brief,
            bars,
            k_mul_spec,
            book_path=OUT_BOOK,
            n_target=8,
            n_neutral=4,
            n_near=4,
            n_panel=4,
            rows={"small": 12, "large": 32},
            controls_override=None,
        )
        k2_s = round(time.perf_counter() - t_k2, 2)
        docs_to_audit.append(kc2["sheet"])
        cs2 = kc2["summary"].get("control_standing") or {}
        check(
            "knob_claim_2_steering_mul_with_controls",
            kc2["verdict"] in ("works but harms", "did not replicate", "lead")
            and cs2.get("count") == brief.default_controls
            and cs2.get("passes_alpha") is True
            and (kc2["summary"]["batteries"].get("near_miss_abs") or 0.0) > bars.near_miss_abs_max,
            {
                "seconds": k2_s,
                "verdict": kc2["verdict"],
                "small_dir": kc2["summary"]["small_directed"],
                "large_dir": kc2["summary"]["large_directed"],
                "near_miss_abs": kc2["summary"]["batteries"]["near_miss_abs"],
                "ctl_at_least": cs2.get("at_least"),
                "ctl_count": cs2.get("count"),
                "ctl_p_fw": cs2.get("p_fw"),
            },
        )

        # 2d. Fresh knob claim #3: system prompt knob on pronoun_pull
        k_prompt_spec = {
            "prompt": "When continuing a narrative sentence about a person, prefer masculine pronouns ('he') over feminine pronouns ('she'); leave all other questions unchanged.",
        }
        t_k3 = time.perf_counter()
        kc3 = lk.run_fresh_knob_claim(
            h,
            brief,
            bars,
            k_prompt_spec,
            book_path=OUT_BOOK,
            n_target=8,
            n_neutral=4,
            n_near=4,
            n_panel=4,
            rows={"small": 12, "large": 32},
            controls_override=0,
        )
        k3_s = round(time.perf_counter() - t_k3, 2)
        docs_to_audit.append(kc3["sheet"])
        check(
            "knob_claim_3_prompt_knob",
            kc3["verdict"] in ("lead", "works but harms", "kept the base", "did not replicate"),
            {
                "seconds": k3_s,
                "verdict": kc3["verdict"],
                "small_dir": kc3["summary"]["small_directed"],
                "large_dir": kc3["summary"]["large_directed"],
                "near_miss_abs": kc3["summary"]["batteries"]["near_miss_abs"],
                "neutral_kl": kc3["summary"]["batteries"]["neutral_kl"],
                "reasons": kc3["summary"]["reasons"],
            },
        )

        # 2e. Read-once & budget m=3 refusals
        ro_refused = False
        try:
            lk.run_fresh_knob_claim(h, brief, bars, k_bias_spec, book_path=OUT_BOOK, n_target=4, n_neutral=2, n_near=2, n_panel=2)
        except lk.KnobRefused:
            ro_refused = True
        k_extra_spec = {"prompt": "Use neutral wording everywhere."}
        m_refused = False
        try:
            lk.run_fresh_knob_claim(h, brief, bars, k_extra_spec, book_path=OUT_BOOK, n_target=4, n_neutral=2, n_near=2, n_panel=2)
        except lk.KnobRefused:
            m_refused = True
        check("knob_read_once_and_m_budget_refused", ro_refused and m_refused, {"read_once": ro_refused, "m_budget": m_refused})

        record["knobs_31b"] = {
            "calibration_seconds": cal_s,
            "equiv_weight_vs_mul": {"weight": round(w_eff, 5), "mul": round(k_eff, 5), "gap": round(equiv_gap, 5)},
            "claim_1_logit_bias": {"seconds": k1_s, "verdict": kc1["verdict"], "summary": kc1["summary"]},
            "claim_2_steering_mul": {"seconds": k2_s, "verdict": kc2["verdict"], "summary": kc2["summary"]},
            "claim_3_prompt": {"seconds": k3_s, "verdict": kc3["verdict"], "summary": kc3["summary"]},
        }

        # ===================================================================== 3. Multi-session 31B Phase 6 Study
        print("\nStep 3: 31B Multi-session Phase 6 Study (5 arms x 3 sessions = 15 isolated side books)...", flush=True)
        brief_json_p = tmp / "study_brief.json"
        study_brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=1)
        study_cal_book = tmp / "study_cal_book.jsonl"
        study_bars = lcl.calibrate_bars(
            h,
            study_brief,
            layer=50,
            n_cols=10,
            n_controls=3,
            n_neutral=4,
            n_near=4,
            n_panel=4,
            seed=2026,
            book_path=study_cal_book,
        )
        lbr.save_brief(study_brief, brief_json_p)

        study_plan = lev.EvalStudyPlan(
            name="verify_31b_phase6_study",
            brief_name=study_brief.name,
            brief_path=str(brief_json_p),
            sessions_per_arm=3,
            session_budget=lses.SessionBudget(
                max_gpu_seconds=300.0,
                max_forwards=80,
                max_cells=8,
                max_cell_seconds=45.0,
                max_cell_mem_gib=4.0,
                min_controls=1,
            ),
            driver_budget=ldrv.DriverBudget(max_turns=18, max_dollars=0.0, max_wall_seconds=180.0),
            claim_design={
                "n_target": 4,
                "n_neutral": 2,
                "n_near": 2,
                "n_panel": 2,
                "rows": {"small": 12, "large": 32},
                "controls_override": 19,
            },
            arms=(
                lev.EvalArm("scr_beh", "scripted", reach="behaviour", mode="weight_edit"),
                lev.EvalArm("scr_act", "scripted", reach="activations", mode="weight_edit"),
                lev.EvalArm("scr_wgt", "scripted", reach="weights", mode="weight_edit"),
                lev.EvalArm("scr_atl", "scripted", reach="atlas", mode="weight_edit"),
                lev.EvalArm("scr_wgt_knobs", "scripted", reach="weights", mode="knobs_only"),
            ),
            note="Live 31B Phase 6 reach ladder + matched-budget knobs_only verification (N=3 independent sessions per arm).",
        )
        study_dir = tmp / "study_31b"
        t_st = time.perf_counter()
        sres = lev.run_eval_study(
            h,
            study_plan,
            study_dir,
            weights_dir=snap,
            atlas_dir=atlas_idx,
            bars=study_bars,
            calibration_book=study_cal_book,
            probe_sandbox=False,
        )
        study_s = round(time.perf_counter() - t_st, 2)
        docs_to_audit.append(sres["report"])
        reloaded = lev.load_study_from_book(study_dir)
        summ = sres["summary"]
        check(
            "study_31b_15_isolated_sessions",
            sres["book"]["verified"]
            and len(sres["outcomes"]) == 15
            and reloaded["book_verified"]
            and reloaded["session_books_checked"] == 15
            and not reloaded["session_books_bad"]
            and summ["pooled_across_reaches"] is False
            and summ["pooled_across_families"] is False
            and len(summ["comparisons"]["weight_vs_knobs"]) == 1
            and len(summ["comparisons"]["reach_ladders"]) == 1
            and summ["comparisons"]["reach_ladders"][0]["atlas_vs_weights"] is not None,
            {
                "seconds": study_s,
                "sessions": len(sres["outcomes"]),
                "arms": {
                    (a["arm_ids"][0] if a.get("arm_ids") else f"{a['reach']}_{a['mode']}"): {
                        "reach": a["reach"],
                        "mode": a["mode"],
                        "verdict_counts": a["verdict_counts"],
                        "median_directed": (a.get("directed_effect") or {}).get("median"),
                        "mean_gpu_s": (a.get("cost") or {}).get("mean_gpu_seconds"),
                    }
                    for a in summ["arms"]
                },
            },
        )
        record["study_31b"] = {
            "seconds": study_s,
            "summary": summ,
            "report": sres["report"],
        }

        # ===================================================================== 4. Blinded planted-fault & canary on 31B
        print("\nStep 4: Blinded planted-fault & canary benchmark on 31B L50...", flush=True)
        plant_rec = {"terms": [{"layer": 50, "silence": [5313, 8801]}]}
        repair_rec = {"terms": [{"layer": 50, "scale": [5313, 8801], "factor": 2.0}]}
        b_cases = [
            lev.BlindedCase(
                "planted_ref_31b",
                "planted",
                reach="weights",
                plant_recipe=plant_rec,
                driver_family="replay",
                driver_config={
                    "steps": [
                        {"tool": "status", "args": {}},
                        {
                            "tool": "read",
                            "propose": True,
                            "args": {
                                "plan": {
                                    "name": "blind_fix_31b",
                                    "kind": "paired",
                                    "edit": repair_rec,
                                    "items": ["p0"],
                                    "doses": [1.0],
                                    "rows": {"small": 12, "large": 16},
                                    "controls": {"count": 1, "seed": 0},
                                },
                                "items": {"p0": {"question": "Answer briefly: Alice called Bob because", "answer": " she smiled."}},
                            },
                        },
                    ]
                },
            ),
            lev.BlindedCase(
                "canary_ref_31b",
                "canary",
                reach="weights",
                driver_family="replay",
                driver_config={"steps": [{"tool": "status", "args": {}}]},
            ),
            lev.BlindedCase(
                "planted_noref_31b",
                "planted",
                reach="activations",
                plant_recipe=plant_rec,
                driver_family="replay",
                driver_config={"steps": [{"tool": "status", "args": {}}]},
            ),
            lev.BlindedCase(
                "canary_noref_31b",
                "canary",
                reach="activations",
                driver_family="replay",
                driver_config={"steps": [{"tool": "status", "args": {}}]},
            ),
        ]
        t_bl = time.perf_counter()
        b_summ = lev.run_blinded_benchmark(
            h,
            b_cases,
            tmp / "blinded_31b",
            weights_dir=snap,
            session_budget=lses.SessionBudget(min_controls=1),
            driver_budget=ldrv.DriverBudget(max_turns=8, max_dollars=0.0),
            probe_sandbox=False,
            eval_items=eq_items,
        )
        bl_s = round(time.perf_counter() - t_bl, 2)
        by_cid = {c["case_id"]: c for c in b_summ["cases"]}
        pr = by_cid["planted_ref_31b"]
        check(
            "blinded_benchmark_31b",
            b_summ["blinding_cards_equal_by_reach"].get("weights") is True
            and b_summ["blinding_cards_equal_by_reach"].get("activations") is True
            and b_summ["canary_abstention_rate"] == 1.0
            and pr.get("localized_layer") is True
            and pr.get("site_recall") == 1.0
            and pr.get("kl_plant_alone") is not None
            and pr["kl_plant_alone"] > 0.0
            and pr.get("kl_after_repair") == 0.0
            and pr.get("recovery_fraction") == 1.0,
            {
                "seconds": bl_s,
                "cards": b_summ["blinding_cards_equal_by_reach"],
                "canary_abstention_rate": b_summ["canary_abstention_rate"],
                "kl_plant_alone": pr.get("kl_plant_alone"),
                "kl_after_repair": pr.get("kl_after_repair"),
                "recovery_fraction": pr.get("recovery_fraction"),
            },
        )
        record["blinded_31b"] = {"seconds": bl_s, **b_summ}

        # ===================================================================== 5. Cold-driver pages & <= 200-word rebuild
        print("\nStep 5: Cold-driver pages & live 31B <= 200-word rebuild test...", flush=True)
        cold_words: Dict[str, int] = {}
        cold_ok = True
        for bname, bobj in lbr.starter_briefs(h.tokenizer).items():
            cp = lev.render_cold_driver_page(bobj)
            docs_to_audit.append(cp["text"])
            cold_words[bname] = cp["word_count"]
            if cp["word_count"] > lev.MAX_COLD_PAGE_WORDS or cp["prompt_named_site"]:
                cold_ok = False
        check("cold_driver_pages_ok", cold_ok, cold_words)

        rb_desc = lev.make_rebuild_description(OUT_BOOK, equiv_seq)
        docs_to_audit.append(rb_desc["description"])
        t_rb = time.perf_counter()
        rb_res = lev.run_rebuild_check(h, rb_desc["description"], OUT_BOOK, equiv_seq, out_book=tmp / "rebuild_31b.jsonl")
        rb_s = round(time.perf_counter() - t_rb, 2)
        rb_comps = rb_res.get("comparisons") or []
        rb_max_gap = max((c["gap"] for c in rb_comps if c.get("gap") is not None), default=None)
        check(
            "rebuild_check_31b",
            rb_desc["word_count"] <= lev.MAX_REBUILD_WORDS
            and rb_res["ok"] is True
            and len(rb_comps) >= 2,
            {
                "seconds": rb_s,
                "word_count": rb_desc["word_count"],
                "figures_compared": len(rb_comps),
                "max_abs_gap": rb_max_gap,
            },
        )
        record["protocol_31b"] = {
            "cold_driver_words": cold_words,
            "rebuild_words": rb_desc["word_count"],
            "rebuild_text": rb_desc["description"],
            "rebuild_result": rb_res,
        }

    # ===================================================================== 6. Final invariants
    we.assert_base(h.lm, h.hooks_at_load)
    check("assert_base_final", True, "model clean at base")

    fb1 = int(getattr(engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)
    check("zero_unpacker_fallbacks", fb1 == fb0, {"before": fb0, "after": fb1, "fallback_count": fb1 - fb0})

    ver_bk = lb.verify(OUT_BOOK)
    check("ledger_whole", bool(ver_bk["ok"]), {"entries": ver_bk["entries"], "head": str(ver_bk.get("head") or "")[:16], "why": ver_bk.get("why")})

    bad_words: List[str] = []
    for doc in docs_to_audit:
        for m in KEPT_OUT_RE.finditer(doc):
            bad_words.append(m.group(0))
    check("zero_kept_out_words_rendered", not bad_words, {"docs": len(docs_to_audit), "found": bad_words[:10]})

    peak_gib = round(torch.cuda.max_memory_allocated(h.device) / 2**30, 3)
    total_s = round(time.perf_counter() - t_all, 2)
    passed = sum(1 for c in checks if c["ok"])
    failed = sum(1 for c in checks if not c["ok"])
    record["peak_gib"] = peak_gib
    record["total_seconds"] = total_s
    record["fallback_count"] = fb1 - fb0
    record["passed"] = passed
    record["failed"] = failed

    write_json_atomic(OUT_JSON, record)
    print(
        f"\nPhase 6 31B card verification complete in {total_s:.1f} s: {passed} passed, {failed} failed "
        f"(peak {peak_gib:.2f} GiB, fallbacks {fb1 - fb0}).",
        flush=True,
    )
    print(f"Wrote {OUT_JSON} and {OUT_BOOK}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
