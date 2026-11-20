#!/usr/bin/env python3
"""CPU verification suite for Levelling Phase 6: Whether agents help (`scripts/check_level_eval.py`).

Tests every Phase 6 module on the CPU stand-in (no GPU, single thread, temporary folders only):
1. Read-only `AtlasReader` (`backend/level_atlas.py`), label-gate provenance, and allowlisted `/atlas` bwrap containment.
2. Cheaper-knob kernel & fresh knob claim gate (`backend/level_knobs.py`): prompts, decoding (sampler, thinking,
   logit_bias), steering (`direction`, `neuron_mul`, `neuron_add`), matched random steering controls, harm batteries,
   written_loop knobs, read-once/m/calibration/leakage refusals.
3. Phase 6 reach ladder (`behaviour`, `activations`, `weights`, `atlas`) & `knobs_only` mode on `LevelSession` and
   `level_mcp.py` (including Unix-socket MCP bridge `level_mcp.py --socket`).
4. Driver runners (`backend/level_drivers.py`): `ScriptedWeightDriver`, `ScriptedKnobDriver`, `ReplayDriver`,
   `OpenAIChatDriver` (mock transport + token/dollar/turn/kill caps), `ClaudeCLIDriver` (folder-reach mock CLI over
   socket bridge + credential stripping + wall-clock SIGKILL), `CodexCLIDriver` containment refusal.
5. Multi-session study orchestration (`backend/level_eval.py`): pre-flight validation, per-session isolated side books,
   unpooled reach ladder & weight-vs-knobs & agent-vs-scripted & `--bare` ablation tables, exact Wilson/Fisher/perm
   statistics, model-free reload from `study_book.jsonl`, and `STOP` file kill switch.
6. Blinded planted-fault (with and without healthy reference) & canary benchmark with identical harness cards.
7. Future-ready protocol checks: cold-driver one-page brief (<= 600 words), <= 200-word rebuild test, and owner CLI
   (`scripts/levelling_eval.py`).
8. Surface isolation & zero kept-out words audit across all Phase 6 files and rendered documents.

Run:
    /home/eric/bin/ai/theodolite2/.venv/bin/python -I scripts/check_level_eval.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_grad_enabled(False)

from scripts.atlas_stand_in import stand_in as make_stand_in, use_stand_in_atlas  # noqa: E402
from backend import level_atlas as la  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_briefs as lbr  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_drivers as ldrv  # noqa: E402
from backend import level_eval as lev  # noqa: E402
from backend import level_knobs as lkn  # noqa: E402
from backend import level_mcp as lmcp  # noqa: E402
from backend import level_sandbox as ls  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend.weight_reader import export_stand_in_safetensors, save_array  # noqa: E402

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)

PASSED = 0
FAILED = 0
DOCS_COLLECTED: List[str] = []


def check(cond: bool, msg: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {msg}", flush=True)
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}", flush=True)


def refuses(fn: Callable[[], Any], exc_type: Any, needle: str = "") -> bool:
    try:
        fn()
    except exc_type as exc:
        return (not needle) or (needle.lower() in str(exc).lower())
    return False


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    t0 = time.perf_counter()
    use_stand_in_atlas()
    rt, _vault = make_stand_in()
    h = lv.handle(rt)
    has_bwrap = shutil.which("bwrap") is not None

    with tempfile.TemporaryDirectory(prefix="levelling_phase6_check_") as tmp_s:
        tmp = Path(tmp_s)
        prev_book = os.environ.get(lb.BOOK_ENV)
        prev_edits = os.environ.get(we.EDITS_ENV)
        os.environ[lb.BOOK_ENV] = str(tmp / "default_book.jsonl")
        os.environ[we.EDITS_ENV] = str(tmp / "edits")
        try:
            weights_dir = export_stand_in_safetensors(h.lm, tmp / "weights", model_id=h.model_id)
            atlas_root = tmp / "atlas_data"
            atlas_idx = la.build_stand_in_atlas_index(h.lm, h.W_U, h.tokenizer, atlas_root / "index", model_id=h.model_id)
            # Plant decoy secret files next to index/ AND inside index/ to test allowlist binding!
            (atlas_root / "scientist").mkdir(parents=True, exist_ok=True)
            (atlas_root / "scientist" / "endpoints.json").write_text('{"secret":"nope"}\n', encoding="utf-8")
            (atlas_idx / "level_book.jsonl").write_text('{"fake":"secret"}\n', encoding="utf-8")

            # ========================================================================= 1. AtlasReader & sandbox /atlas
            section("1. Read-only AtlasReader & allowlisted /atlas bwrap containment")
            check(la.resolve_atlas_index_dir(atlas_root) == atlas_idx.resolve(),
                  "1: resolve_atlas_index_dir finds <atlas_data>/index automatically")
            check(refuses(lambda: la.resolve_atlas_index_dir(tmp / "empty_dir"), la.AtlasReaderError, "does not exist"),
                  "1: resolve_atlas_index_dir refuses missing directory")
            binds = la.collect_atlas_binds(atlas_root)
            bind_dsts = {dst for _s, dst in binds}
            check("/atlas/lens.npz" in bind_dsts and "/atlas/vocab.json" in bind_dsts
                  and all("scientist" not in d and "level_book" not in d for d in bind_dsts),
                  f"1: collect_atlas_binds allowlists index files only ({sorted(bind_dsts)})")

            with la.AtlasReader(atlas_idx) as reader:
                ainfo = reader.info()
                check(ainfo["n_layers"] == h.n_layers and ainfo["has_vocab"] and "lens.npz" in ainfo["files"],
                      f"1: AtlasReader.info reports shape ({ainfo['n_layers']}x{ainfo['n_neurons']}) and files")
                # Pick a real decoded token that appears in lens_pos at the last layer
                last_l = h.n_layers - 1
                sample_tid = int(reader._npz("lens.npz")["lens_pos"][last_l, 0, 0])
                sw = reader.search_words(f"id:{sample_tid}", top_k=10)
                check(sw["total_hits"] >= 1 and any(x["layer"] == last_l and x["neuron"] == 0 for x in sw["hits"]),
                      "1: search_words by id:<tid> locates genuine top-writing neuron")
                # Verify label-gate trust floor: layer 0 is below label_min_layer, last layer is at/above
                c0 = reader.neuron_card(0, 0)
                clast = reader.neuron_card(last_l, 0)
                check(c0["label_trust_pos"]["trusted"] is False and "below_layer" in str(c0["label_trust_pos"]["reason"])
                      and clast["label_trust_pos"]["trusted"] is True,
                      "1: label-gate provenance marks early layers untrusted and late layers trusted")
                w_down = reader.top_wiring(0, 0, direction="down", top_k=4)
                lstats = reader.layer_stats(0, top_k=4)
                check(len(w_down["links"]) == 4 and len(lstats["top_extreme_neurons"]) == 4,
                      "1: top_wiring and layer_stats return bounded structured hits")
                check(refuses(lambda: reader.query("drop_table", {}), la.AtlasReaderError, "Unknown atlas query"),
                      "1: AtlasReader.query refuses unknown op")
                check(refuses(lambda: reader.neuron_card(99, 0), la.AtlasReaderError, "outside"),
                      "1: AtlasReader refuses out-of-range layer")

            if has_bwrap:
                sb_work = tmp / "sb_atlas_work"
                sb_work.mkdir()
                sb_cfg = ls.SandboxConfig(
                    work_dir=sb_work, weights_dir=weights_dir, atlas_dir=atlas_root,
                    timeout_s=15.0, max_mem_gib=1.5, model_id=h.model_id, weights_id=h.weights_id,
                )
                cell_code = f"""
import json
from pathlib import Path
assert A is not None, "A should be mounted"
assert W is not None, "W should be mounted"
info = A.info()
sw = A.search_words("id:{sample_tid}", top_k=5)
leaked_sci = Path("/atlas/scientist").exists()
leaked_bk = Path("/atlas/level_book.jsonl").exists()
ro_blocked = False
try:
    Path("/atlas/pwn.txt").write_text("x")
except OSError:
    ro_blocked = True
print("SB_ATLAS:" + json.dumps({{"layers": info["n_layers"], "hits": len(sw["hits"]), "leaked_sci": leaked_sci, "leaked_bk": leaked_bk, "ro": ro_blocked}}))
"""
                cres = ls.run_cell(cell_code, sb_cfg)
                got_line = [ln for ln in cres["stdout"].splitlines() if ln.startswith("SB_ATLAS:")]
                parsed_sb = json.loads(got_line[0].split("SB_ATLAS:", 1)[1]) if got_line else {}
                check(cres["ok"] and parsed_sb.get("layers") == h.n_layers and parsed_sb.get("hits", 0) >= 1
                      and parsed_sb.get("leaked_sci") is False and parsed_sb.get("leaked_bk") is False
                      and parsed_sb.get("ro") is True,
                      f"1: bwrap sandbox mounts A read-only with allowlisted index only ({parsed_sb}, err={cres['stderr'][:120]})")
            else:
                check(True, "1: bwrap unavailable in environment (skipped live bwrap atlas check)")

            # ========================================================================= 2. Cheaper-knob kernel & gate
            section("2. Cheaper-knob kernel, steering hooks, and fresh knob claim gate")
            brief = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=4)
            cal_book = tmp / "knob_cal_book.jsonl"
            bars = lcl.calibrate_bars(h, brief, layer=h.n_layers - 1, n_cols=4, n_controls=2, rows={"small": 8, "large": 16}, book_path=cal_book)
            we.assert_base(h.lm, h.hooks_at_load)

            # Knob validation refusals
            check(refuses(lambda: lkn.build_knob(h, {}), lkn.KnobRefused, "at least one"),
                  "2: build_knob refuses empty knob")
            check(refuses(lambda: lkn.build_knob(h, {"prompt": "a" * 5000}), lkn.KnobRefused, "exceeds"),
                  "2: build_knob refuses oversized prompt")
            check(refuses(lambda: lkn.build_knob(h, {"decoding": {"sampler": {"min_p": 0.1}}}), lkn.KnobRefused, "min_p"),
                  "2: build_knob refuses min_p (backend/sampler_request.py)")
            check(refuses(lambda: lkn.build_knob(h, {"decoding": {"sampler": {"repetition_penalty": 1.2}}}), lkn.KnobRefused, "repetition_penalty"),
                  "2: build_knob refuses repetition_penalty")
            check(refuses(lambda: lkn.build_knob(h, {"decoding": {"sampler": {"temperature": 0.0, "presence_penalty": 0.5}}}), lkn.KnobRefused, "temperature > 0"),
                  "2: build_knob refuses presence_penalty with temperature=0")
            check(refuses(lambda: lkn.build_knob(h, {"decoding": {"logit_bias": {"0": 99.0}}}), lkn.KnobRefused, "[-10, 10]"),
                  "2: build_knob refuses out-of-range logit_bias")
            check(refuses(lambda: lkn.build_knob(h, {"steering": [{"kind": "direction", "layer": 0, "path": "../escape.npy", "dose": 1.0}]}),
                          lkn.KnobRefused, "relative path"),
                  "2: build_knob refuses path traversal in steering vector")

            # Build genuine direction vector inside work dir, test sha256 mismatch refusal & valid build
            kwork = tmp / "knob_work"
            kwork.mkdir()
            D = h.lm.norm.weight.shape[0]
            rng = np.random.default_rng(7)
            raw_v = rng.standard_normal(D).astype(np.float32)
            qm_last = we.quiet_mask(h.lm, h.n_layers - 1)
            raw_v[qm_last] = 0.0
            vmeta = save_array(kwork / "steer_u.npy", raw_v)
            check(refuses(lambda: lkn.build_knob(
                h, {"steering": [{"kind": "direction", "layer": h.n_layers - 1, "path": "steer_u.npy", "sha256": "0" * 64, "dose": 1.0}]},
                base_dir=kwork,
            ), lkn.KnobRefused, "sha256 mismatch"),
                  "2: build_knob refuses sha256 mismatch on steering vector file")

            knob_dir = lkn.build_knob(
                h,
                {"steering": [{"kind": "direction", "layer": h.n_layers - 1, "path": "steer_u.npy", "sha256": vmeta["sha256"], "dose": 1.5, "scale": 1.0}]},
                base_dir=kwork,
            )
            knob_mul = lkn.build_knob(
                h,
                {"steering": [{"kind": "neuron_mul", "layer": h.n_layers - 1, "neurons": [0, 1, 2], "factor": 0.0}]},
            )
            knob_nadd = lkn.build_knob(
                h,
                {"steering": [{"kind": "neuron_add", "layer": h.n_layers - 1, "neurons": [0, 1], "dose": 1.0, "scale": 1.0}]},
            )
            check(knob_dir.knob_classes == ("steering",) and knob_mul.knob_classes == ("steering",) and knob_nadd.knob_classes == ("steering",),
                  "2: build_knob builds direction, neuron_mul, and neuron_add steering specs")

            # Verify neuron_mul (factor=0, spare_bos=False) matches weight_edits.silence per-row on the same item!
            prac_d = brief.generator.draw("practice", 2, seed_bytes=b"kprac_seed", tokenizer=h.tokenizer)
            prac_map = {d.item.name: d.item for d in prac_d}
            kp_mul = lkn.read_knob_practice(
                h, {"steering": [{"kind": "neuron_mul", "layer": h.n_layers - 1, "neurons": [0, 1, 2], "factor": 0.0}]},
                prac_map, brief=brief, controls=2, rows={"small": 8, "large": 16}, book_path=cal_book,
            )
            with lev._use_book(cal_book):
                w_sil = lv.read(h, {
                    "name": "cmp_sil", "kind": "paired", "edit": {"terms": [{"layer": h.n_layers - 1, "silence": [0, 1, 2]}]},
                    "items": list(prac_map.keys()), "doses": [1.0], "rows": {"small": 8, "large": 16},
                    "readouts": brief.readouts, "controls": {"count": 2, "seed": 0},
                }, prac_map)
            w_sm = w_sil["summary"]["classes"]["small"]["arms"][lv._key("edit", 1.0)]["odds"]["mean_over_items"]
            check(kp_mul["status"] == "ok" and abs(kp_mul["small"]["mean"] - w_sm) < 1e-5,
                  f"2: neuron_mul(0) steering matches weight_edits.silence per-row (knob={kp_mul['small']['mean']:.6f}, weight={w_sm:.6f})")

            # Practice estimate + prompt knob + direction steering knob
            kest = lkn.estimate_knob_reading(h, {"prompt": "Be concise."}, prac_map, rows={"small": 8, "large": 16})
            kp_prompt = lkn.read_knob_practice(
                h, {"prompt": "Use neutral nouns only."}, prac_map, brief=brief,
                controls=2, rows={"small": 8, "large": 16}, book_path=cal_book,
            )
            kp_dir = lkn.read_knob_practice(
                h, {"steering": [{"kind": "direction", "layer": h.n_layers - 1, "path": "steer_u.npy", "sha256": vmeta["sha256"], "dose": 2.0, "scale": 1.0}]},
                prac_map, brief=brief, controls=2, rows={"small": 8, "large": 16}, base_dir=kwork, book_path=cal_book,
            )
            check(kest["forwards"] > 0 and kp_prompt["status"] == "ok" and kp_dir["status"] == "ok"
                  and kp_dir["steering_controls_rank"] is not None,
                  "2: knob_estimate and read_knob_practice succeed for prompt and direction steering knobs")

            # Fresh knob claim: blunt logit_bias moves target AND moves near_misses equally -> works but harms!
            toward_id = int(brief.readouts["odds"]["toward"][0])
            # direction is -1 for pronoun_pull, so negative bias on toward moves directed odds positively!
            blunt_knob = {"decoding": {"logit_bias": {str(toward_id): -3.0}}}
            kcl_book = tmp / "knob_claim_book.jsonl"
            shutil.copy2(cal_book, kcl_book)
            kcl1 = lkn.run_fresh_knob_claim(
                h, brief, bars, blunt_knob, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                rows={"small": 8, "large": 16}, controls_override=2, book_path=kcl_book,
            )
            DOCS_COLLECTED.append(kcl1["sheet"])
            check(kcl1["verdict"] == "works but harms"
                  and (kcl1["summary"]["large_directed"] or 0.0) > 2.5
                  and (kcl1["summary"]["batteries"]["near_miss_abs"] or 0.0) > 2.5,
                  f"2: blunt logit_bias triggers 'works but harms' via near-miss bleed (verdict={kcl1['verdict']}, "
                  f"eff={kcl1['summary']['large_directed']}, nm={kcl1['summary']['batteries']['near_miss_abs']})")
            # Read-once refusal on same knob
            check(refuses(lambda: lkn.run_fresh_knob_claim(
                h, brief, bars, blunt_knob, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                rows={"small": 8, "large": 16}, book_path=kcl_book,
            ), lkn.KnobRefused, "read once"),
                  "2: run_fresh_knob_claim enforces read-once per knob_id")

            # Second fresh knob claim (steering direction with matched random steering controls) -> spends reading #2 of 2
            steer_spec = {"steering": [{"kind": "direction", "layer": h.n_layers - 1, "path": "steer_u.npy", "sha256": vmeta["sha256"], "dose": 1.0, "scale": 1.0}]}
            kcl2 = lkn.run_fresh_knob_claim(
                h, brief, bars, steer_spec, base_dir=kwork, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                rows={"small": 8, "large": 16}, controls_override=3, book_path=kcl_book,
            )
            DOCS_COLLECTED.append(kcl2["sheet"])
            check(kcl2["verdict"] in lkn.KNOB_VERDICTS and kcl2["summary"]["controls_applicable"] is True
                  and kcl2["summary"]["control_standing"] is not None,
                  f"2: steering knob claim runs matched random steering controls (verdict={kcl2['verdict']})")
            # Third fresh knob claim -> refused by m=2 budget!
            check(refuses(lambda: lkn.run_fresh_knob_claim(
                h, brief, bars, {"prompt": "Another prompt."}, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                rows={"small": 8, "large": 16}, book_path=kcl_book,
            ), lkn.KnobRefused, "spent all 2"),
                  "2: run_fresh_knob_claim enforces brief max_fresh_readings budget m")
            # Uncalibrated book refused
            check(refuses(lambda: lkn.run_fresh_knob_claim(
                h, brief, bars, {"prompt": "Hi"}, book_path=tmp / "empty_book.jsonl",
            ), lkn.KnobRefused, "calibration"),
                  "2: run_fresh_knob_claim refuses bars not calibrated in the same book")

            # Written-loop knob claim on stand-in
            lp_brief = lbr.enumeration_loop_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=2)
            lp_brief = lbr.Brief(**{
                **{f: getattr(lp_brief, f) for f in ("name", "behavior", "metric", "direction", "least_effect",
                                                    "readouts", "generator", "near_miss_generator", "allowed_tensors",
                                                    "allowed_layers", "max_fresh_readings", "controls_per_family",
                                                    "prompt_arms", "chat_settings", "compare_settings",
                                                    "prompt_named_site", "driver_text")},
                "written_config": {"max_new_tokens": 12, "temperature": 0.0, "seeds": [1], "loop_rule": "either"},
            })
            lp_book = tmp / "lp_knob_book.jsonl"
            lp_bars = lcl.calibrate_bars(h, lp_brief, layer=h.n_layers - 1, n_cols=4, n_controls=1, rows={"small": 8, "large": 16}, book_path=lp_book)
            lp_kcl = lkn.run_fresh_knob_claim(
                h, lp_brief, lp_bars, {"decoding": {"thinking": False}, "prompt": "Be brief."},
                n_target=4, n_neutral=2, n_near=2, n_panel=2, rows={"small": 8, "large": 16}, book_path=lp_book,
            )
            DOCS_COLLECTED.append(lp_kcl["sheet"])
            check(lp_kcl["verdict"] in lkn.KNOB_VERDICTS and lb.verify(lp_book)["ok"],
                  f"2: written_loop fresh knob claim completes cleanly with intact book (verdict={lp_kcl['verdict']})")
            we.assert_base(h.lm, h.hooks_at_load)

            # ========================================================================= 3. Reach ladder & knobs_only on LevelSession & MCP
            section("3. Phase 6 reach ladder, knobs_only mode, and Unix-socket MCP bridge")
            # Behaviour reach: NO cell, NO capture, NO atlas_query
            sb_beh = lses.LevelSession(
                h, tmp / "s_beh", reach="behaviour", mode="weight_edit", book_path=tmp / "b_beh.jsonl",
                weights_dir=weights_dir, atlas_dir=atlas_root, probe_on_start=False,
            )
            check("cell" not in sb_beh.allowed_tools() and "capture" not in sb_beh.allowed_tools()
                  and "atlas_query" not in sb_beh.allowed_tools()
                  and sb_beh.harness_card()["with_healthy_reference"] is False
                  and sb_beh.harness_card()["atlas_mounted"] is False,
                  "3: behaviour reach hides cell, capture, atlas_query, weights, and atlas")
            check(refuses(lambda: sb_beh.capture([{"question": "Hi", "answer": "Ok"}], [{"question": "Yo", "answer": "No"}], 0),
                          lses.SessionRefused, "not available at reach"),
                  "3: behaviour reach refuses capture")
            check(refuses(lambda: sb_beh.atlas_query("info"), lses.SessionRefused, "not available"),
                  "3: behaviour reach refuses atlas_query")
            sb_beh.close()

            # Activations reach: cell + capture, NO weights, NO atlas even if dirs passed
            sb_act = lses.LevelSession(
                h, tmp / "s_act", reach="activations", mode="weight_edit", book_path=tmp / "b_act.jsonl",
                weights_dir=weights_dir, atlas_dir=atlas_root, probe_on_start=False,
            )
            check("cell" in sb_act.allowed_tools() and "capture" in sb_act.allowed_tools()
                  and "atlas_query" not in sb_act.allowed_tools()
                  and sb_act.harness_card()["with_healthy_reference"] is False
                  and sb_act.harness_card()["atlas_mounted"] is False,
                  "3: activations reach enables cell + capture with zero weights and zero atlas mounted")
            sb_act.close()

            # Weights reach: requires weights_dir, mounts weights, NEVER mounts atlas
            check(refuses(lambda: lses.LevelSession(
                h, tmp / "s_w_bad", reach="weights", book_path=tmp / "b_w_bad.jsonl", weights_dir=None, probe_on_start=False,
            ), lses.SessionRefused, "requires a readable weights_dir"),
                  "3: weights reach refuses missing weights_dir")
            sb_w = lses.LevelSession(
                h, tmp / "s_w", reach="weights", book_path=tmp / "b_w.jsonl",
                weights_dir=weights_dir, atlas_dir=atlas_root, probe_on_start=False,
            )
            check(sb_w.harness_card()["with_healthy_reference"] is True and sb_w.harness_card()["atlas_mounted"] is False
                  and "atlas_query" not in sb_w.allowed_tools(),
                  "3: weights reach mounts weights only and never mounts atlas")
            sb_w.close()

            # Atlas reach: requires BOTH weights_dir and valid atlas_dir
            check(refuses(lambda: lses.LevelSession(
                h, tmp / "s_a_bad", reach="atlas", book_path=tmp / "b_a_bad.jsonl",
                weights_dir=weights_dir, atlas_dir=None, probe_on_start=False,
            ), lses.SessionRefused, "requires a readable atlas_dir"),
                  "3: atlas reach refuses missing atlas_dir")
            sb_a = lses.LevelSession(
                h, tmp / "s_a", reach="atlas", book_path=tmp / "b_a.jsonl",
                weights_dir=weights_dir, atlas_dir=atlas_root, probe_on_start=False,
            )
            check(sb_a.harness_card()["with_healthy_reference"] is True and sb_a.harness_card()["atlas_mounted"] is True
                  and "atlas_query" in sb_a.allowed_tools(),
                  "3: atlas reach mounts weights + atlas and exposes atlas_query")
            aq_r = sb_a.atlas_query("info")
            check(aq_r["n_layers"] == h.n_layers, "3: LevelSession.atlas_query('info') returns valid atlas metadata")
            sb_a.close()

            # Knobs-only mode: hides weight-edit tools, enforces KeptBack on knob_read, refuses blinded plant/canary
            check(refuses(lambda: lses.LevelSession(
                h, tmp / "s_kb_blind", reach="behaviour", mode="knobs_only", book_path=tmp / "b_kbb.jsonl", canary=True, probe_on_start=False,
            ), lses.SessionRefused, "weight_edit"),
                  "3: knobs_only refuses blinded canary/plant sessions")
            bk_kn = tmp / "b_kn_sess.jsonl"
            shutil.copy2(cal_book, bk_kn)
            s_kn = lses.LevelSession(
                h, tmp / "s_kn", reach="activations", mode="knobs_only", bare=True, book_path=bk_kn,
                brief=brief, bars=bars, probe_on_start=False,
                claim_design={"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "rows": {"small": 8, "large": 16}, "controls_override": 2},
            )
            kn_tools = set(s_kn.allowed_tools())
            check({"knob_estimate", "knob_read", "knob_write", "claim_knob"}.issubset(kn_tools)
                  and not ({"estimate", "read", "write", "claim"} & kn_tools)
                  and s_kn.harness_card()["mode"] == "knobs_only" and s_kn.harness_card()["bare"] is True,
                  f"3: knobs_only exposes knob tools and removes all weight-edit tools ({sorted(kn_tools)})")
            check(refuses(lambda: s_kn.read({}, {}), lses.SessionRefused, "knobs_only"),
                  "3: knobs_only refuses weight-edit read()")
            # KeptBack refusal on knob_read!
            kb_drawn = brief.generator.draw("kept_back", 1, seed_bytes=b"leak_test", tokenizer=h.tokenizer)[0].item
            check(refuses(lambda: s_kn.knob_read(
                {"prompt": "Try leak"}, {"leak": {"question": kb_drawn.question, "answer": kb_drawn.answer}},
            ), lses.SessionRefused, "kept-back"),
                  "3: knob_read refuses kept-back frame wording via KeptBack")

            # Unix-socket MCP bridge test (`session_socket_bridge` + subprocess `level_mcp.py --socket`)
            check(refuses(lambda: lmcp.session_socket_bridge(s_kn, s_kn.work_dir / "bad.sock").__enter__(),
                          lses.SessionRefused, "work folder"),
                  "3: session_socket_bridge refuses socket inside work_dir")
            sock_p = s_kn.session_dir / "bridge.sock"
            with lmcp.session_socket_bridge(s_kn, sock_p):
                st_mode = stat.S_IMODE(sock_p.stat().st_mode)
                check(st_mode == 0o600, f"3: bridge socket permissions are 0600 (got {oct(st_mode)})")
                reqs = "\n".join([
                    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
                    json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "status", "arguments": {}}}),
                ]) + "\n"
                proc = subprocess.run(
                    [sys.executable, "-I", str(ROOT_DIR / "backend" / "level_mcp.py"), "--socket", str(sock_p)],
                    input=reqs, text=True, capture_output=True, timeout=15.0,
                )
                lines_out = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
                t_names = [t["name"] for t in lines_out[1]["result"]["tools"]] if len(lines_out) >= 2 else []
                st_body = json.loads(lines_out[2]["result"]["content"][0]["text"]) if len(lines_out) >= 3 else {}
                check(proc.returncode == 0 and "claim_knob" in t_names and "claim" not in t_names
                      and st_body.get("harness_card", {}).get("mode") == "knobs_only",
                      "3: level_mcp.py --socket bridges stdio JSON-RPC to live LevelSession cleanly")
            s_kn.close()

            # ========================================================================= 4. Driver runners & containment
            section("4. Driver runners, prompt scaffolding, caps, kill switch, and CLI containment")
            bk_drv = tmp / "b_drv.jsonl"
            shutil.copy2(cal_book, bk_drv)
            s_std = lses.LevelSession(h, tmp / "s_std", reach="weights", bare=False, book_path=bk_drv, weights_dir=weights_dir, brief=brief, bars=bars, probe_on_start=False)
            s_bare = lses.LevelSession(h, tmp / "s_bare", reach="weights", bare=True, book_path=bk_drv, weights_dir=weights_dir, brief=brief, bars=bars, probe_on_start=False)
            p_std = ldrv.build_driver_prompt(s_std)
            p_bare = ldrv.build_driver_prompt(s_bare)
            DOCS_COLLECTED.extend([p_std["system"], p_std["user"], p_bare["system"], p_bare["user"]])
            check(p_bare["word_count"] < p_std["word_count"] and p_std["prompt_named_site"] is False and p_bare["bare"] is True,
                  f"4: --bare prompt ({p_bare['word_count']} words) strips scaffolding vs standard ({p_std['word_count']} words)")
            s_std.close()
            s_bare.close()

            # OpenAI driver with mock transport: multi-turn loop + dollar/token/kill caps
            bk_oa = tmp / "b_oa.jsonl"
            shutil.copy2(cal_book, bk_oa)
            s_oa = lses.LevelSession(
                h, tmp / "s_oa", reach="behaviour", mode="weight_edit", backend="openai", book_path=bk_oa,
                brief=brief, bars=bars, probe_on_start=False, budget=lses.SessionBudget(min_controls=1),
                claim_design={"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "rows": {"small": 12, "large": 16}, "controls_override": 2, "match_tolerance": 20.0},
            )
            oa_turn = {"i": 0}

            def mock_transport(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                oa_turn["i"] += 1
                if oa_turn["i"] == 1:
                    return {
                        "model": "mock-gpt-test",
                        "usage": {"prompt_tokens": 500, "completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 20}},
                        "choices": [{"message": {
                            "role": "assistant", "content": "",
                            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "status", "arguments": "{}"}}],
                        }}],
                    }
                if oa_turn["i"] == 2:
                    return {
                        "model": "mock-gpt-test",
                        "usage": {"prompt_tokens": 700, "completion_tokens": 120},
                        "choices": [{"message": {
                            "role": "assistant", "content": "",
                            "tool_calls": [{"id": "tc2", "type": "function", "function": {
                                "name": "claim",
                                "arguments": json.dumps({"edit": {"terms": [{"layer": h.n_layers - 1, "silence": [0, 1, 2]}]}}),
                            }}],
                        }}],
                    }
                return {
                    "model": "mock-gpt-test",
                    "usage": {"prompt_tokens": 800, "completion_tokens": 60},
                    "choices": [{"message": {"role": "assistant", "content": "Submitted edit and reviewed sheet."}}],
                }

            oa_out = ldrv.run_openai_driver(
                s_oa, model="mock-gpt",
                driver_budget=ldrv.DriverBudget(max_turns=6, max_tokens=50000, max_dollars=2.0, input_usd_per_mtok=1.0, output_usd_per_mtok=2.0),
                transport_fn=mock_transport,
            )
            check(oa_out["submitted"] is True and oa_out["stop_reason"] == "claimed"
                  and oa_out["harness_card"]["driver_model"] == "mock-gpt-test"
                  and oa_out["usage"]["total_tokens"] == 2280 and oa_out["usage"]["reasoning_tokens"] == 20
                  and oa_out["usage"]["estimated_dollars"] > 0.0 and oa_out["book"]["verified"],
                  f"4: OpenAIChatDriver runs full tool loop, tracks tokens/dollars, records verdict ({oa_out['verdict']})")

            # OpenAI dollar cap cutoff
            s_oa_cap = lses.LevelSession(h, tmp / "s_oa_cap", reach="behaviour", backend="openai", book_path=tmp / "b_oac.jsonl", probe_on_start=False)
            oa_cap_out = ldrv.run_openai_driver(
                s_oa_cap, model="mock-gpt",
                driver_budget=ldrv.DriverBudget(max_turns=10, max_dollars=0.001, input_usd_per_mtok=10.0, output_usd_per_mtok=10.0),
                transport_fn=lambda p, hd, u, t: {
                    "model": "mock-gpt", "usage": {"prompt_tokens": 200, "completion_tokens": 200},
                    "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "status", "arguments": "{}"}}]}}],
                },
            )
            check(oa_cap_out["stop_reason"] == "budget_dollars" and oa_cap_out["usage"]["turns"] == 1,
                  f"4: OpenAIChatDriver enforces max_dollars cap (stop={oa_cap_out['stop_reason']}, turns={oa_cap_out['usage']['turns']})")

            # OpenAI remote URL without max_dollars refused
            s_oa_rem = lses.LevelSession(h, tmp / "s_oa_rem", reach="behaviour", backend="openai", book_path=tmp / "b_oar.jsonl", probe_on_start=False)
            check(refuses(lambda: ldrv.run_openai_driver(
                s_oa_rem, model="x", base_url="https://openrouter.ai/api/v1", driver_budget=ldrv.DriverBudget(max_dollars=None),
            ), ldrv.DriverRefused, "max_dollars"),
                  "4: OpenAIChatDriver refuses remote endpoint without owner-approved max_dollars")
            s_oa_rem.close()

            # ReplayDriver + STOP file KillSwitch
            stop_f = tmp / "STOP_REPLAY"
            stop_f.write_text("halt\n", encoding="utf-8")
            s_rep = lses.LevelSession(h, tmp / "s_rep", reach="behaviour", backend="replay", book_path=tmp / "b_rep.jsonl", probe_on_start=False)
            rep_out = ldrv.run_replay_driver(s_rep, [{"tool": "status", "args": {}}], kill_switch=ldrv.KillSwitch(stop_file=stop_f))
            check(rep_out["stop_reason"] == "killed" and rep_out["usage"]["turns"] == 0,
                  "4: KillSwitch STOP file halts ReplayDriver immediately before first turn")

            # Codex containment refusal (unprobed + /tmp cache)
            s_cdx = lses.LevelSession(h, tmp / "s_cdx", reach="behaviour", backend="local", book_path=tmp / "b_cdx.jsonl", probe_on_start=False)
            check(refuses(lambda: ldrv.run_codex_cli_driver(s_cdx, codex_probed=False), ldrv.DriverRefused, "probed"),
                  "4: CodexCLIDriver strictly refused until codex_probed=True and private compile cache")
            s_cdx.close()

            # ClaudeCLIDriver containment + mock CLI subprocess over Unix-socket MCP bridge + credential stripping!
            s_cl_ref = lses.LevelSession(h, tmp / "s_cl_ref", reach="behaviour", backend="claude", book_path=tmp / "b_clr.jsonl", probe_on_start=False)
            check(refuses(lambda: ldrv.run_claude_cli_driver(s_cl_ref, driver_budget=ldrv.DriverBudget(max_dollars=None)),
                          ldrv.DriverRefused, "max_dollars"),
                  "4: ClaudeCLIDriver refuses launch without owner-approved max_dollars")
            s_cl_ref.close()

            mock_cli = tmp / "mock_claude.py"
            mock_cli.write_text(f"""#!{sys.executable}
import json, os, subprocess, sys
# Verify cwd is work_dir and secret env vars are stripped!
leaked = [k for k in ("HF_TOKEN", "OPENROUTER_API_KEY", "RUNPOD_API_KEY") if k in os.environ]
cfg_path = sys.argv[sys.argv.index("--mcp-config") + 1]
cfg = json.loads(open(cfg_path).read())["mcpServers"]["levelling"]
req = "\\n".join([
    json.dumps({{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{}}}}),
    json.dumps({{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"status","arguments":{{}}}}}}),
]) + "\\n"
p = subprocess.run([cfg["command"], *cfg["args"]], input=req, text=True, capture_output=True, timeout=10)
ok_mcp = (p.returncode == 0 and "harness_card" in p.stdout)
print(json.dumps({{"type":"assistant","message":{{"model":"claude-mock-4","usage":{{"input_tokens":120,"output_tokens":45}}}}}}), flush=True)
print(json.dumps({{"type":"result","total_cost_usd":0.0123,"num_turns":1,"is_error":False,"result":f"ok_mcp={{ok_mcp}} leaked={{leaked}} cwd={{os.getcwd()}}"}}), flush=True)
""", encoding="utf-8")
            os.chmod(mock_cli, 0o755)
            prev_hf = os.environ.get("HF_TOKEN")
            prev_or = os.environ.get("OPENROUTER_API_KEY")
            os.environ["HF_TOKEN"] = "hf_secret_should_not_leak"
            os.environ["OPENROUTER_API_KEY"] = "or_secret_should_not_leak"
            try:
                s_cl = lses.LevelSession(h, tmp / "s_cl", reach="behaviour", backend="claude", book_path=tmp / "b_cl.jsonl", probe_on_start=False)
                cl_out = ldrv.run_claude_cli_driver(
                    s_cl, claude_bin=str(mock_cli), allow_mock_cli=True,
                    driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=15.0),
                )
            finally:
                if prev_hf is None:
                    os.environ.pop("HF_TOKEN", None)
                else:
                    os.environ["HF_TOKEN"] = prev_hf
                if prev_or is None:
                    os.environ.pop("OPENROUTER_API_KEY", None)
                else:
                    os.environ["OPENROUTER_API_KEY"] = prev_or
            check(cl_out["harness_card"]["driver_model"] == "claude-mock-4"
                  and "ok_mcp=True" in cl_out["notes"]
                  and "leaked=[]" in cl_out["notes"]
                  and str(s_cl.work_dir) in cl_out["notes"]
                  and abs(cl_out["usage"]["estimated_dollars"] - 0.0123) < 1e-6,
                  f"4: ClaudeCLIDriver bridges MCP socket, strips secrets, pins cwd=work_dir, logs cost ({cl_out['notes']})")

            # Hanging CLI wall-clock SIGKILL test
            hang_cli = tmp / "hang_claude.py"
            hang_cli.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n", encoding="utf-8")
            os.chmod(hang_cli, 0o755)
            s_hang = lses.LevelSession(h, tmp / "s_hang", reach="behaviour", backend="claude", book_path=tmp / "b_hang.jsonl", probe_on_start=False)
            t_h0 = time.monotonic()
            hang_out = ldrv.run_claude_cli_driver(
                s_hang, claude_bin=str(hang_cli), allow_mock_cli=True,
                driver_budget=ldrv.DriverBudget(max_turns=2, max_dollars=1.0, max_wall_seconds=0.8),
            )
            dt_hang = time.monotonic() - t_h0
            check(hang_out["stop_reason"] == "budget_wall" and dt_hang < 4.0,
                  f"4: ClaudeCLIDriver SIGKILLs hanging subprocess on wall cap ({dt_hang:.2f}s)")

            # ========================================================================= 5. Multi-session study & exact stats
            section("5. Multi-session Phase 6 study, isolation, unpooled aggregation, and exact statistics")
            # Exact statistical primitives
            w0 = lev.wilson_ci(0, 3)
            w3 = lev.wilson_ci(3, 3)
            check(w0[0] == 0.0 and 0.5 < w0[1] < 0.65 and w3[1] == 1.0 and 0.35 < w3[0] < 0.5,
                  f"5: wilson_ci accurate at extremes (0/3={w0}, 3/3={w3})")
            f_sep = lev.fisher_exact_2x2(4, 0, 0, 4)
            f_eq = lev.fisher_exact_2x2(2, 2, 2, 2)
            check(abs((f_sep or 0.0) - (2.0 / 70.0)) < 1e-5 and f_eq == 1.0,
                  f"5: fisher_exact_2x2 matches exact analytical values (sep={f_sep}, eq={f_eq})")
            p_sep = lev.exact_perm_diff_p([2.0, 2.1, 2.2], [0.0, 0.1, 0.2])
            p_small = lev.exact_perm_diff_p([1.0], [2.0])
            check(p_sep["exact"] is True and abs(p_sep["p"] - 0.1) < 1e-5 and p_small["p"] is None,
                  f"5: exact_perm_diff_p exact for N=3+3 (p={p_sep['p']}) and returns None when n<2")

            # Plan pre-flight refusals
            check(refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                name="bad", brief_name="pronoun_pull", arms=(
                    lev.EvalArm("dup", "scripted", reach="behaviour"),
                    lev.EvalArm("dup", "scripted", reach="behaviour"),
                ),
            )), lev.EvalRefused, "Duplicate"),
                  "5: validate_study_plan refuses duplicate arm_id")
            check(refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                name="bad_secret", brief_name="pronoun_pull", arms=(
                    lev.EvalArm("a0", "openai", reach="behaviour", driver_config={"api_key": "sk-oops"}),
                ),
            )), lev.EvalRefused, "forbidden secret"),
                  "5: validate_study_plan refuses raw api_key in driver_config")
            check(refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                name="bad_home", brief_name="pronoun_pull", arms=(
                    lev.EvalArm("a0", "scripted", reach="home"),
                ),
            )), lev.EvalRefused, "home reach"),
                  "5: validate_study_plan refuses home reach")
            check(refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                name="bad_atlas", brief_name="pronoun_pull", arms=(
                    lev.EvalArm("a0", "scripted", reach="atlas"),
                ),
            ), weights_dir=weights_dir, atlas_dir=None), lev.EvalRefused, "atlas_dir"),
                  "5: validate_study_plan refuses atlas reach without atlas_dir")

            # Full multi-session study across all 4 reaches + knobs_only + --bare ablation + second driver family!
            study_cd = {"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "rows": {"small": 12, "large": 16}, "controls_override": 2, "match_tolerance": 20.0}
            study_sb = lses.SessionBudget(max_gpu_seconds=300.0, max_forwards=120, max_cells=10, min_controls=1)
            study_db = ldrv.DriverBudget(max_turns=16, max_dollars=1.0, input_usd_per_mtok=1.0, output_usd_per_mtok=2.0)

            def _make_oa_study_transport() -> Callable[..., Dict[str, Any]]:
                state = {"step": 0}

                def _tr(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    state["step"] += 1
                    if state["step"] % 2 == 1:
                        return {
                            "model": "mock-second-family",
                            "usage": {"prompt_tokens": 300, "completion_tokens": 50},
                            "choices": [{"message": {
                                "role": "assistant", "content": "",
                                "tool_calls": [{"id": "c1", "type": "function", "function": {
                                    "name": "claim",
                                    "arguments": json.dumps({"edit": {"terms": [{"layer": h.n_layers - 1, "silence": [1, 2, 3]}]}}),
                                }}],
                            }}],
                        }
                    return {
                        "model": "mock-second-family",
                        "usage": {"prompt_tokens": 350, "completion_tokens": 30},
                        "choices": [{"message": {"role": "assistant", "content": "Done."}}],
                    }
                return _tr

            brief_json_p = tmp / "study_brief.json"
            lbr.save_brief(brief, brief_json_p)
            study_plan = lev.EvalStudyPlan(
                name="p6_full_standin",
                brief_name=brief.name,
                brief_path=str(brief_json_p),
                sessions_per_arm=3,
                session_budget=study_sb,
                driver_budget=study_db,
                claim_design=study_cd,
                arms=(
                    lev.EvalArm("scr_beh_edit", "scripted", reach="behaviour", mode="weight_edit", driver_config={"submit_even_if_weak": True}),
                    lev.EvalArm("scr_act_edit", "scripted", reach="activations", mode="weight_edit", driver_config={"submit_even_if_weak": True}),
                    lev.EvalArm("scr_wgt_edit", "scripted", reach="weights", mode="weight_edit", driver_config={"submit_even_if_weak": True}),
                    lev.EvalArm("scr_atl_edit", "scripted", reach="atlas", mode="weight_edit", driver_config={"submit_even_if_weak": True}),
                    lev.EvalArm("scr_wgt_edit_bare", "scripted", reach="weights", mode="weight_edit", bare=True, driver_config={"submit_even_if_weak": True}),
                    lev.EvalArm("scr_atl_knobs", "scripted", reach="atlas", mode="knobs_only", driver_config={"submit_even_if_weak": True}),
                    lev.EvalArm("oa_wgt_edit", "openai", reach="weights", mode="weight_edit", driver_model="mock-second-family",
                                driver_config={"transport_fn": _make_oa_study_transport()}),
                ),
            )
            study_dir = tmp / "p6_study"
            sres = lev.run_eval_study(
                h, study_plan, study_dir, weights_dir=weights_dir, atlas_dir=atlas_root,
                bars=bars, calibration_book=cal_book, probe_sandbox=False,
            )
            DOCS_COLLECTED.append(sres["report"])
            summ = sres["summary"]
            check(sres["book"]["verified"] and len(sres["outcomes"]) == 21
                  and summ["pooled_across_reaches"] is False and summ["pooled_across_families"] is False
                  and len(summ["arms"]) == 7,
                  f"5: 7-arm x 3-session (21 sessions) study completed with intact study_book and unpooled arms")
            # Verify per-session book isolation: every session has its own verified level_book.jsonl and all 3 replicates ran!
            session_books = {o["book"]["path"] for o in sres["outcomes"]}
            check(len(session_books) == 21 and all(o["book"]["verified"] for o in sres["outcomes"]),
                  "5: all 21 sessions ran in strictly isolated, individually verified side level books")
            # Verify all four comparison tables populated!
            comps = summ["comparisons"]
            check(len(comps["weight_vs_knobs"]) >= 1
                  and len(comps["reach_ladders"]) >= 1
                  and comps["reach_ladders"][0]["atlas_vs_weights"] is not None
                  and len(comps["agent_vs_scripted"]) >= 1
                  and len(comps["scaffolding_ablations"]) >= 1,
                  "5: study summary populates weight_vs_knobs, reach_ladders (with atlas_vs_weights), agent_vs_scripted, and scaffolding_ablations")
            # Model-free reload from study_book.jsonl!
            reloaded = lev.load_study_from_book(study_dir)
            check(reloaded["book_verified"] and reloaded["session_books_checked"] == 21 and not reloaded["session_books_bad"]
                  and reloaded["summary"]["completed_sessions"] == 21,
                  "5: load_study_from_book verifies study_book and all 21 isolated session books model-free")

            # Study-level STOP file kill switch halts cleanly before running sessions
            halt_dir = tmp / "p6_halt"
            halt_dir.mkdir()
            (halt_dir / "STOP").write_text("stop now\n", encoding="utf-8")
            mini_plan = lev.EvalStudyPlan(
                name="p6_halt_test", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=2,
                session_budget=study_sb, driver_budget=ldrv.DriverBudget(max_dollars=0.0), claim_design=study_cd,
                arms=(lev.EvalArm("scr_beh", "scripted", reach="behaviour", mode="weight_edit"),),
            )
            hres = lev.run_eval_study(h, mini_plan, halt_dir, bars=bars, calibration_book=cal_book, probe_sandbox=False)
            check(hres["halted_reason"] and "kill_switch" in hres["halted_reason"] and len(hres["outcomes"]) == 0
                  and hres["book"]["verified"],
                  f"5: study-level STOP file halts multi-session study cleanly ({hres['halted_reason']})")
            we.assert_base(h.lm, h.hooks_at_load)

            # ========================================================================= 6. Blinded planted & canary eval
            section("6. Blinded planted-fault (healthy-ref & no-healthy-ref) and canary benchmark")
            # Planted silence on layer 0 neurons [2, 3]; exact additive repair is scale by 2.0 (0 + 2 - 1 = 1!)
            plant_rec = {"terms": [{"layer": 0, "silence": [2, 3]}]}
            repair_rec = {"terms": [{"layer": 0, "scale": [2, 3], "factor": 2.0}]}
            b_cases = [
                lev.BlindedCase(
                    "planted_ref", "planted", reach="weights", plant_recipe=plant_rec,
                    driver_family="replay", driver_config={"steps": [
                        {"tool": "status", "args": {}},
                        {"tool": "read", "propose": True, "args": {
                            "plan": {"name": "check_fix", "kind": "paired", "edit": repair_rec, "items": ["p0"], "doses": [1.0],
                                     "rows": {"small": 8, "large": 16}, "controls": {"count": 1, "seed": 0}},
                            "items": {"p0": {"question": "Say hello.", "answer": "Hello there."}},
                        }},
                    ]},
                ),
                lev.BlindedCase(
                    "canary_ref", "canary", reach="weights",
                    driver_family="replay", driver_config={"steps": [{"tool": "status", "args": {}}]},
                ),
                lev.BlindedCase(
                    "planted_noref", "planted", reach="activations", plant_recipe=plant_rec,
                    driver_family="scripted",
                ),
                lev.BlindedCase(
                    "canary_noref", "canary", reach="activations",
                    driver_family="replay", driver_config={"steps": [{"tool": "status", "args": {}}]},
                ),
            ]
            b_summary = lev.run_blinded_benchmark(
                h, b_cases, tmp / "blinded_out", weights_dir=weights_dir,
                session_budget=lses.SessionBudget(min_controls=1),
                driver_budget=ldrv.DriverBudget(max_turns=12, max_dollars=0.0),
                probe_sandbox=False,
            )
            by_cid = {r["case_id"]: r for r in b_summary["cases"]}
            pr = by_cid["planted_ref"]
            check(b_summary["n_cases"] == 4 and b_summary["n_planted"] == 2 and b_summary["n_canary"] == 2
                  and b_summary["blinding_cards_equal_by_reach"].get("weights") is True
                  and b_summary["blinding_cards_equal_by_reach"].get("activations") is True
                  and b_summary["blinding_cards_equal_by_reach_driver"].get("weights:replay") is True
                  and b_summary["canary_abstention_rate"] == 1.0,
                  f"6: blinded benchmark enforces identical cards across planted/canary at both reaches (cards={b_summary['blinding_cards_equal_by_reach']})")
            check(pr.get("localized_layer") is True and pr.get("site_recall") == 1.0 and pr.get("site_jaccard") == 1.0
                  and pr.get("kl_plant_alone") is not None and pr["kl_plant_alone"] > 0.0
                  and pr.get("kl_after_repair") == 0.0 and pr.get("recovery_fraction") == 1.0,
                  f"6: blinded ground-truth scoring verifies exact localization & additive repair recovery (kl_plant={pr.get('kl_plant_alone')}, kl_after={pr.get('kl_after_repair')}, rec={pr.get('recovery_fraction')})")
            we.assert_base(h.lm, h.hooks_at_load)

            # ========================================================================= 7. Cold-driver, Rebuild, & CLI
            section("7. Cold-driver one-page brief, <= 200-word rebuild test, and owner CLI")
            for bname, bobj in lbr.starter_briefs(h.tokenizer).items():
                cp = lev.render_cold_driver_page(bobj)
                DOCS_COLLECTED.append(cp["text"])
                check(cp["word_count"] <= lev.MAX_COLD_PAGE_WORDS and cp["prompt_named_site"] is False
                      and "5313" not in cp["text"] and "layer 50" not in cp["text"].lower(),
                      f"7: cold-driver page for {bname} is <= 600 words ({cp['word_count']}) with no site spoilers")

            # Rebuild test (<= 200 words, reproduces exact reading on CPU stand-in!)
            rb_src_book = tmp / "rebuild_src.jsonl"
            rb_item = lv.make_item(h.tokenizer, "Answer briefly: Alice called Bob.", "She smiled.", name="rb0")
            rb_plan = {
                "name": "orig_for_rebuild", "kind": "paired",
                "edit": {"terms": [{"layer": h.n_layers - 1, "silence": [0, 1, 2]}]},
                "items": ["rb0"], "doses": [1.0], "rows": {"small": 8, "large": 16},
                "readouts": {"kl": True, "own": True, "odds": None},
                "controls": {"family": "other_sites", "count": 2, "seed": 5},
            }
            with lev._use_book(rb_src_book):
                orig_r = lv.read(h, rb_plan, {"rb0": rb_item})
            orig_seq = int(orig_r["book"]["seq"])
            rb_desc = lev.make_rebuild_description(rb_src_book, orig_seq)
            DOCS_COLLECTED.append(rb_desc["description"])
            check(rb_desc["word_count"] <= lev.MAX_REBUILD_WORDS,
                  f"7: rebuild description is <= 200 words ({rb_desc['word_count']} words)")
            rb_chk = lev.run_rebuild_check(
                h, rb_desc["description"], rb_src_book, orig_seq, out_book=tmp / "rebuild_out.jsonl",
            )
            check(rb_chk["ok"] and len(rb_chk["comparisons"]) >= 4 and all(c["gap"] == 0.0 for c in rb_chk["comparisons"]),
                  f"7: rebuild check reproduces reading identically from <= 200-word description ({len(rb_chk['comparisons'])} figures, max gap 0.0)")

            # Owner CLI smoke tests (`scripts/levelling_eval.py`)
            cli_plan_p = tmp / "cli_plan.json"
            p_cli1 = subprocess.run(
                [sys.executable, "-I", str(ROOT_DIR / "scripts" / "levelling_eval.py"), "plan",
                 "--name", "cli_smoke", "--brief", "pronoun_pull", "--sessions", "1",
                 "--driver", "scripted", "--claim-controls", "2", "--claim-n-target", "4",
                 "--max-gpu-seconds", "120", "--out", str(cli_plan_p)],
                text=True, capture_output=True, timeout=30.0,
            )
            p_cli2 = subprocess.run(
                [sys.executable, "-I", str(ROOT_DIR / "scripts" / "levelling_eval.py"), "cold-page", "--brief", "pronoun_pull"],
                text=True, capture_output=True, timeout=30.0,
            )
            p_cli3 = subprocess.run(
                [sys.executable, "-I", str(ROOT_DIR / "scripts" / "levelling_eval.py"), "report", str(study_dir)],
                text=True, capture_output=True, timeout=30.0,
            )
            p_cli4 = subprocess.run(
                [sys.executable, "-I", str(ROOT_DIR / "scripts" / "levelling_eval.py"), "rebuild",
                 "--book", str(rb_src_book), "--seq", str(orig_seq), "--check", "--stand-in",
                 "--out-book", str(tmp / "cli_rb.jsonl")],
                text=True, capture_output=True, timeout=60.0,
            )
            check(p_cli1.returncode == 0 and cli_plan_p.is_file()
                  and p_cli2.returncode == 0 and "Levelling Cold-Driver" in p_cli2.stdout
                  and p_cli3.returncode == 0 and "Per-Arm Results (Unpooled)" in p_cli3.stdout
                  and p_cli4.returncode == 0,
                  f"7: scripts/levelling_eval.py plan, cold-page, report, and rebuild --check --stand-in all exit 0 "
                  f"(rc={p_cli1.returncode},{p_cli2.returncode},{p_cli3.returncode},{p_cli4.returncode})")

            # ========================================================================= 8. Surface isolation & zero kept-out words
            section("8. Agent-surface isolation and zero kept-out words audit")
            surfaces = [
                "backend/level_mcp.py", "backend/level_session.py", "backend/level_routes.py",
                "backend/scientist_tools.py", "backend/scientist_mcp.py", "backend/server.py",
            ]
            check(all(not re.search(r"level_(eval|drivers|trial|export|rerun)", (ROOT_DIR / s).read_text(encoding="utf-8"))
                      for s in surfaces if (ROOT_DIR / s).is_file()),
                  "8: no agent or page surface imports or names level_eval, level_drivers, level_trial, level_export, or level_rerun")

            audit_files = [
                "backend/level_atlas.py", "backend/level_knobs.py", "backend/level_drivers.py",
                "backend/level_eval.py", "backend/level_sandbox.py", "backend/level_session.py",
                "backend/level_mcp.py", "scripts/levelling_eval.py", "docs/SCAFFOLD.md",
            ]
            for rel in audit_files:
                txt = (ROOT_DIR / rel).read_text(encoding="utf-8")
                hits = [m.group(0) for m in KEPT_OUT_RE.finditer(txt)]
                check(not hits, f"8: {rel} has zero kept-out words (found: {hits[:5]})")
            doc_blob = "\n".join(DOCS_COLLECTED)
            doc_hits = [m.group(0) for m in KEPT_OUT_RE.finditer(doc_blob)]
            check(len(DOCS_COLLECTED) >= 8 and not doc_hits,
                  f"8: all {len(DOCS_COLLECTED)} rendered sheets, reports, prompts, cold pages, and rebuild texts have zero kept-out words (found: {doc_hits[:5]})")

        finally:
            if prev_book is None:
                os.environ.pop(lb.BOOK_ENV, None)
            else:
                os.environ[lb.BOOK_ENV] = prev_book
            if prev_edits is None:
                os.environ.pop(we.EDITS_ENV, None)
            else:
                os.environ[we.EDITS_ENV] = prev_edits

    dt = time.perf_counter() - t0
    print(f"\nPhase 6 CPU checks: {PASSED} passed, {FAILED} failed in {dt:.1f} s.", flush=True)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
