#!/usr/bin/env python3
"""CPU verification suite for Levelling Phase 6: Whether agents help (`scripts/check_level_eval.py`).

Tests every Phase 6 module on the CPU stand-in (no GPU, single thread, temporary folders only):
1. Read-only `AtlasReader` (`backend/level_atlas.py`), label-gate provenance, and allowlisted `/atlas` bwrap containment.
2. Cheaper-knob kernel & fresh knob claim gate (`backend/level_knobs.py`): prompts, decoding (sampler, thinking,
   logit_bias), steering (`direction`, `neuron_mul`, `neuron_add`), matched random steering controls, harm batteries,
   written_loop knobs, read-once/m/calibration/leakage refusals.
3. Phase 6 reach ladder (`behaviour`, `activations`, `weights`, `atlas`) & `knobs_only` mode on `LevelSession` and
   `level_mcp.py` (including Unix-socket MCP bridge `level_mcp.py --socket`: a private socket folder, one call at a
   time).
4. Driver runners (`backend/level_drivers.py`): `ScriptedWeightDriver`, `ScriptedKnobDriver`, `ReplayDriver`,
   `OpenAIChatDriver` (mock transport; endpoint host, key pairing, prices, redirects, caps kept during the run, STOP
   while a request is out), `ClaudeCLIDriver` (mock CLIs over the socket bridge: allowlisted environment, door-only
   command line, denials and tool errors, token cap, a cell or a claim still running at STOP; and the real `claude`
   CLI against a local stand-in of the Messages API, skipped when it is not installed), `CodexCLIDriver` refusal.
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

import contextlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

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
from backend.label_gate import label_check  # noqa: E402
from backend.weight_reader import export_stand_in_safetensors, save_array  # noqa: E402

KEPT_OUT_RE = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)

PASSED = 0
FAILED = 0
SKIPPED: List[str] = []
DOCS_COLLECTED: List[str] = []


def check(cond: bool, msg: str) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {msg}", flush=True)
    else:
        FAILED += 1
        print(f"  [FAIL] {msg}", flush=True)


def skip(msg: str) -> None:
    """A check that could not run here (a tool it needs is absent): counted apart, never as a pass."""
    SKIPPED.append(msg)
    print(f"  [SKIP] {msg}", flush=True)


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
                # The label gate on the stand-in: its layer floor is scaled to the stand-in's depth, and its lens is not
                # the base-fitted J++ lens, so only the layer can set a label aside there (as label_check rules)
                c0 = reader.neuron_card(0, 0)
                clast = reader.neuron_card(last_l, 0)
                g0, glast = c0["label_gate_pos"], clast["label_gate_pos"]
                check(g0["held_back"] is True and any(str(r).startswith("below_layer_") for r in g0["reasons"])
                      and glast["held_back"] is False and glast["lens_validated"] is False
                      and glast.get("floor_scaled_for_stand_in") is True and "label_trust_pos" not in clast,
                      f"1: the label gate sets the stand-in's early-layer label aside and lets its last layer through, "
                      f"lens unvalidated, floor scaled ({g0['reasons']}, {glast['reasons']})")
                check("n1-gate-1" in str(c0.get("label_gate")) and "160" in str(ainfo.get("label_gate"))
                      and "trusted" not in json.dumps(ainfo),
                      "1: the reader states the gate's rule and how weak a cut it is, and calls no label trusted")

            # The gate restated in level_atlas (no torch in the sandbox) against label_gate.label_check itself: every
            # layer of a 60-layer model, overlaps 0..10, the validated lens, another lens and none, silenced or not,
            # and missing or empty lists
            gate_cases = gate_diff = 0
            gate_first_diff = None
            base_ids = list(range(10))
            for l_g in range(60):
                for ov in range(11):
                    eye_g = base_ids[:ov] + list(range(100, 110 - ov))
                    for lid in ("jpp-base", "jpp-r3", None):
                        for sil in (False, True):
                            for li_g, ei_g in ((base_ids, eye_g), (None, eye_g), ([], eye_g)):
                                gate_cases += 1
                                a_g = la.gate_check(l_g, li_g, ei_g, lid, silenced=sil)
                                b_g = label_check(l_g, li_g, ei_g, lid, silenced=sil)
                                if a_g != b_g:
                                    gate_diff += 1
                                    gate_first_diff = gate_first_diff or (l_g, ov, lid, sil, a_g, b_g)
            check(gate_diff == 0 and gate_cases == 11880,
                  f"1: level_atlas.gate_check equals label_gate.label_check on all {gate_cases} cases "
                  f"({gate_diff} differ; first {gate_first_diff})")
            # Under the validated lens, layer 59 never sets a label aside for sharing nothing (the two readings are
            # one there), and layers 40-58 do; under another lens only the layer counts
            check(la.gate_check(59, [1] * 10, [2] * 10, "jpp-base")["held_back"] is False
                  and la.gate_check(58, list(range(10)), list(range(10, 20)), "jpp-base")["reasons"] == ["shares_nothing"]
                  and la.gate_check(58, list(range(10)), list(range(10, 20)), "jpp-r3")["held_back"] is False,
                  "1: shares_nothing applies under jpp-base below layer 59 only")

            # A symlink where an allowlisted name stands is refused, not followed into the sandbox
            sym_root = tmp / "atlas_symlinked"
            sym_idx = sym_root / "index"
            sym_idx.mkdir(parents=True)
            for fn_s in ("lens.npz", "vocab.json"):
                shutil.copy2(atlas_idx / fn_s, sym_idx / fn_s)
            secret_s = tmp / "owner_book.jsonl"
            secret_s.write_text('{"owner":"secret"}\n', encoding="utf-8")
            os.symlink(secret_s, sym_idx / "steer_dose.json")
            check(refuses(lambda: la.collect_atlas_binds(sym_root), la.AtlasReaderError, "symlink"),
                  "1: collect_atlas_binds refuses a symlinked index file")
            os.unlink(sym_idx / "steer_dose.json")
            (sym_idx / "passages.d").mkdir()
            os.symlink(secret_s, sym_idx / "passages.d" / "x.npy")
            check(refuses(lambda: la.collect_atlas_binds(sym_root), la.AtlasReaderError, "symlink"),
                  "1: collect_atlas_binds refuses a symlinked file inside passages.d")
            os.unlink(sym_idx / "passages.d" / "x.npy")
            check(all(str(s) != str(secret_s) for s, _d in la.collect_atlas_binds(sym_root)),
                  "1: with the symlinks gone the binds name only the index's own plain files")

            # The steer lens's own check: words at a layer it found weak or did not check are set aside, as the
            # instrument's pages do; a check written for another build of the lens is not used
            st_idx = tmp / "atlas_steer" / "index"
            st_idx.mkdir(parents=True)
            for fn_s in ("lens.npz", "vocab.json"):
                shutil.copy2(atlas_idx / fn_s, st_idx / fn_s)
            lens_arc = np.load(atlas_idx / "lens.npz", allow_pickle=False)
            n_l_s, n_n_s = lens_arc["lens_pos"].shape[:2]
            st_up = np.array(lens_arc["lens_pos"][:, :, :8])
            np.savez(st_idx / "steer_lens.npz", up=st_up, down=np.array(lens_arc["lens_neg"][:, :, :8]),
                     meta=np.array(json.dumps({"built_at": "2026-11-13T23:43:53Z", "lens_id": "stand-in"})))
            standing_s = {str(i): {"standing": ("agrees" if i == n_l_s - 1 else "weak" if i == n_l_s - 2 else "unchecked")}
                          for i in range(n_l_s)}
            (st_idx / "steer_check.json").write_text(json.dumps(
                {"lens": {"built_at": "2026-11-13T23:43:53Z"}, "layers": standing_s}), encoding="utf-8")
            st_tid = int(st_up[n_l_s - 1, 0, 0])
            with la.AtlasReader(st_idx) as st_reader:
                all_s = st_reader.search_words(f"id:{st_tid}", source="steer", top_k=64)
                kept_s = st_reader.search_words(f"id:{st_tid}", source="steer", top_k=64, gate_passes_only=True)
                card_s = st_reader.neuron_card(n_l_s - 2, 0, top_wiring=0)
                check(all(h["steer_check"]["set_aside"] == (h["layer"] != n_l_s - 1) for h in all_s["hits"])
                      and all(h["layer"] == n_l_s - 1 for h in kept_s["hits"]) and kept_s["total_hits"] >= 1
                      and card_s["steer_check"]["standing"] == "weak" and card_s["steer_check"]["set_aside"] is True,
                      f"1: steer words are set aside where the steer check found the lens weak or unchecked "
                      f"({all_s['total_hits']} hits, {kept_s['total_hits']} kept)")
            (st_idx / "steer_check.json").write_text(json.dumps(
                {"lens": {"built_at": "another build"}, "layers": standing_s}), encoding="utf-8")
            with la.AtlasReader(st_idx) as st_reader2:
                card_s2 = st_reader2.neuron_card(n_l_s - 2, 0, top_wiring=0)
                check(card_s2["steer_check"]["check_written"] is False and card_s2["steer_check"]["set_aside"] is False,
                      "1: a steer check written for another build of the lens is not read as this one's")
            check("steer_check.json" in la.ATLAS_INDEX_FILES,
                  "1: the steer check record is among the files bound into /atlas")
            with la.AtlasReader(atlas_idx) as reader:
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
                skip("1: bwrap is not installed here, so the live /atlas mount was not tried")

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

            # Fresh knob claims. A knob claim is one of the brief's fresh readings, asked the weight path's own gate
            # (level_claim.fresh_reading_gate) and judged by its engine (level_claim.judge_claim) with the knob's own
            # families of controls. Each check below failed at f6d4e86, where the knob path kept a budget of its own,
            # read keys the gate never returned, and judged by a gate of its own.
            R816 = {"small": 8, "large": 16}
            L_last = h.n_layers - 1
            KC = dict(n_target=4, n_neutral=2, n_near=2, n_panel=2, rows=R816, match_tolerance=20.0)
            toward_id = int(brief.readouts["odds"]["toward"][0])
            blunt_knob = {"decoding": {"logit_bias": {str(toward_id): -3.0}}}  # direction -1: moves the brief's way
            mul_spec = {"steering": [{"kind": "neuron_mul", "layer": L_last, "neurons": [0, 1, 2], "factor": 0.0}]}
            steer_spec = {"steering": [{"kind": "direction", "layer": L_last, "path": "steer_u.npy",
                                        "sha256": vmeta["sha256"], "dose": 1.0, "scale": 1.0}]}

            def knob_claim(b: Any, bs: Any, spec: Any, bk: Path, **kw: Any) -> Dict[str, Any]:
                return lkn.run_fresh_knob_claim(h, b, bs, spec, book_path=bk, base_dir=kwork, **{**KC, **kw})

            def spent(bk: Path, b: Any = None) -> int:
                b = brief if b is None else b
                return int(lcl.brief_history(b.id, bk, name=b.name)["fresh_count"])

            def knob_regs(bk: Path) -> int:
                return sum(1 for e in lb.entries(bk) if (e.get("body") or {}).get("note_type") == "knob_claim_registration")

            # what has no family of controls, or cannot act, is refused before anything is registered
            kcl_book = tmp / "knob_claim_book.jsonl"
            shutil.copy2(cal_book, kcl_book)
            n_before = spent(kcl_book)
            check(refuses(lambda: knob_claim(brief, bars, {"prompt": "Prefer he."}, kcl_book), lkn.KnobRefused,
                          "needs a steering part")
                  and refuses(lambda: knob_claim(brief, bars, blunt_knob, kcl_book), lkn.KnobRefused, "needs a steering part")
                  and spent(kcl_book) == n_before and knob_regs(kcl_book) == 0,
                  "2: a prompt-only or a logit-bias-only knob has no family of controls, so its fresh claim is refused "
                  "before anything is registered, as a weight edit with none is")
            check(refuses(lambda: knob_claim(brief, bars, mul_spec, kcl_book, controls_override=0), lkn.KnobRefused,
                          "controls a family as a whole number of 1 or more") and knob_regs(kcl_book) == 0,
                  "2: controls_override=0 on a steering knob is refused before registering, as the weight path refuses it")
            mixed = {"steering": mul_spec["steering"] + steer_spec["steering"]}
            spared = {"steering": [{**mul_spec["steering"][0], "spare_bos": True}]}
            check(refuses(lambda: knob_claim(brief, bars, mixed, kcl_book), lkn.KnobRefused, "no one family of controls")
                  and refuses(lambda: knob_claim(brief, bars, spared, kcl_book), lkn.KnobRefused,
                              "arithmetic of no weight edit")
                  and knob_regs(kcl_book) == 0,
                  "2: a knob mixing neuron_mul with a push, and a neuron_mul that spares the start mark, are refused "
                  "before registering (no one family of controls; no weight edit does that arithmetic)")
            sampled = {**mul_spec, "decoding": {"sampler": {"temperature": 0.5}}}
            same_think = {**mul_spec, "decoding": {"thinking": False}}
            check(refuses(lambda: knob_claim(brief, bars, sampled, kcl_book), lkn.KnobRefused, "cannot act")
                  and refuses(lambda: knob_claim(brief, bars, same_think, kcl_book), lkn.KnobRefused, "cannot act")
                  and refuses(lambda: lkn.read_knob_practice(h, {"decoding": {"sampler": {"temperature": 0.25, "top_k": 5}}},
                                                            prac_map, brief=brief, rows=R816, book_path=tmp / "inert.jsonl"),
                              lkn.KnobRefused, "cannot act")
                  and knob_regs(kcl_book) == 0 and not (tmp / "inert.jsonl").exists(),
                  "2: a part that cannot act on the brief's reading (a sampler on a teacher-forced reading, the "
                  "brief's own thinking setting) is refused before a fresh reading or a practice forward is spent")

            # a neuron_mul knob: its controls are its weight edit's, two families, read by the weight path's hooks
            spy_rows: set = set()
            orig_arms = lkn._eval_paired_arms

            def spy_arms(*a: Any, **k: Any) -> Any:
                spy_rows.add(("battery" if "_tgt" not in str(a[1].name) else "target", int(a[6])))
                return orig_arms(*a, **k)

            lkn._eval_paired_arms = spy_arms
            try:
                kcl1 = knob_claim(brief, bars, mul_spec, kcl_book)
            finally:
                lkn._eval_paired_arms = orig_arms
            DOCS_COLLECTED.append(kcl1["sheet"])
            s1 = kcl1["summary"]
            check(kcl1["verdict"] in lkn.KNOB_VERDICTS
                  and s1["control_families"] == ["other_sites_matched", "same_site"]
                  and all(len(s1["control_ids"].get(f) or []) == brief.default_controls for f in s1["control_families"])
                  and set(s1["control_standing"]) == {"small", "large"}
                  and all(set(v) == {"other_sites_matched", "same_site"} for v in s1["control_standing"].values()),
                  f"2: a neuron_mul knob claim is read against its weight edit's two families of controls (matched "
                  f"other sites, same-site directions), {brief.default_controls} a family, in both batch classes "
                  f"(verdict={kcl1['verdict']})")
            check({r for w, r in spy_rows if w == "battery"} == {8, 16} and {r for w, r in spy_rows if w == "target"} == {8, 16},
                  f"2: a knob claim reads its retention batteries in both batch classes, as a weight claim does "
                  f"(rows read: {sorted(spy_rows)})")
            check(all(lkn.CONTROL_FAMILY_SAY[f].format(tol=20.0) in kcl1["sheet"] for f in s1["control_families"])
                  and "Not matched on what they delivered as read" in kcl1["sheet"] and lkn.CONTROL_RULE_SAY in kcl1["sheet"]
                  and lkn.CONTROLS_SHARE_SAY in kcl1["sheet"] and lkn.NOT_READ_SAY[0] in kcl1["sheet"],
                  "2: the knob sheet says what each family of controls is and is not matched on, the rule they are "
                  "counted by, and what a weight claim reads that a knob claim does not")
            check(refuses(lambda: knob_claim(brief, bars, mul_spec, kcl_book), lkn.KnobRefused, "read once"),
                  "2: a knob is read once on a brief: the gate refuses it by its knob id")

            # a negligible push with a blunt logit bias: the bias goes to every control row, so it cannot beat them alone
            # (at f6d4e86 the bias went to the knob's row only, and the knob topped every control on the bias)
            push_bias = {"steering": [{**steer_spec["steering"][0], "dose": 0.001}], "decoding": dict(blunt_knob["decoding"])}
            kcl2 = knob_claim(brief, bars, push_bias, kcl_book)
            DOCS_COLLECTED.append(kcl2["sheet"])
            cm2 = (kcl2["summary"]["control_means"] or {}).get("large", {}).get("random_directions") or []
            eff2 = abs(kcl2["summary"]["large_directed"] or 0.0)
            check(kcl2["summary"]["control_families"] == ["random_directions"] and eff2 > 1.0
                  and len(cm2) == brief.default_controls
                  and all(m is not None and abs(abs(m) - eff2) <= 0.05 * eff2 for m in cm2)
                  and kcl2["verdict"] != "lead",
                  f"2: a push's controls are pushes of its length along random directions, and the knob's logit bias "
                  f"reaches every control row: with a blunt bias each control moved within 5% of the knob's "
                  f"{eff2:.3f} (verdict={kcl2['verdict']}, controls {[round(abs(m), 3) for m in cm2 if m is not None]})")
            check(lkn.CONTROL_FAMILY_SAY["random_directions"] in kcl2["sheet"] and lkn.REVERSED_SAY in kcl2["sheet"],
                  "2: a push's sheet says its controls are matched on the push's size at its layer and on nothing else")
            # one budget a brief: the knob claims spent m = 2, and a weight claim is refused there too
            edit_w = we.build(h.lm, [we.silence(L_last, [9, 10])], model_id=h.model_id, weights_id=h.weights_id)
            check(refuses(lambda: knob_claim(brief, bars, {"steering": [{**mul_spec["steering"][0], "neurons": [4, 5]}]}, kcl_book),
                          lkn.KnobRefused, "spent all 2")
                  and refuses(lambda: lcl.fresh_reading_gate(h, brief, bars, edit_w, kcl_book, what="claim"),
                              lcl.ClaimRefused, "spent all 2")
                  and spent(kcl_book) == 2,
                  "2: two knob claims spend the brief's m = 2: a third knob claim and a weight claim are both refused")
            check(refuses(lambda: knob_claim(brief, bars, mul_spec, tmp / "empty_book.jsonl"), lkn.KnobRefused,
                          "calibration"),
                  "2: run_fresh_knob_claim refuses bars not calibrated in the same book")

            # one budget, the other way about: a weight claim after a knob claim is its second fresh reading, and its
            # sheet lists the knob claim; a trial after both is refused
            b1 = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=2, controls_per_family=4)
            ob_book = tmp / "one_budget.jsonl"
            ob_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=ob_book)
            kob = knob_claim(b1, ob_bars, mul_spec, ob_book)
            wob = lcl.run_fresh_claim(h, b1, ob_bars, edit_w, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                                      rows={"small": 12, "large": 16}, match_tolerance=20.0, book_path=ob_book)
            hob = lcl.brief_history(b1.id, ob_book, name=b1.name)
            edit_t = we.build(h.lm, [we.silence(L_last, [20, 21])], model_id=h.model_id, weights_id=h.weights_id)
            check(wob["meta"]["reading_index"] == 2 and f"reading 1: knob {kob['knob']['id'][:12]}" in wob["sheet"]
                  and hob["fresh_count"] == 2 and hob["book_fresh_readings"] == {b1.name: 2}
                  and hob["seen_knob_ids"] == [kob["knob"]["id"]] and len(hob["verdicts"]) == 2
                  and refuses(lambda: lcl.fresh_reading_gate(h, b1, ob_bars, edit_t, ob_book, what="trial"),
                              lcl.ClaimRefused, "spent all 2"),
                  "2: a weight claim after a knob claim is the brief's second fresh reading, its sheet lists the knob "
                  "claim, the book's history counts both, and a trial after them is refused (m = 2)")

            # the weight path's guards, asked of a knob claim: kept-back wording read in practice by any door, a brief
            # changed under its name, an item seen in practice drawn again
            g_book = tmp / "guards.jsonl"
            g_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=g_book)
            kb_item = b1.generator.draw("kept_back", 1, seed_bytes=b"leakleakleak", tokenizer=h.tokenizer)[0].item
            with lev._use_book(g_book):
                lv.read(h, {"name": "my_practice", "kind": "paired", "edit": {"terms": [{"layer": L_last, "silence": [0, 1, 2]}]},
                            "items": ["leak0"], "doses": [1.0], "rows": R816, "readouts": b1.readouts,
                            "controls": {"count": 1, "seed": 0}}, {"leak0": kb_item})
            check(refuses(lambda: knob_claim(b1, g_bars, mul_spec, g_book), lkn.KnobRefused, "were read in practice in this book")
                  and knob_regs(g_book) == 0,
                  "2: a knob claim after a practice reading of kept-back wording (the command line's door) is refused, "
                  "as the weight path refuses it")
            c_book = tmp / "changed.jsonl"
            c_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=c_book)
            knob_claim(b1, c_bars, mul_spec, c_book)
            b1x = lbr.Brief(**{**{f: getattr(b1, f) for f in ("name", "behavior", "metric", "direction", "readouts",
                                                               "generator", "near_miss_generator", "allowed_tensors",
                                                               "allowed_layers", "max_fresh_readings",
                                                               "controls_per_family", "prompt_arms", "chat_settings",
                                                               "compare_settings", "prompt_named_site", "driver_text")},
                               "least_effect": b1.least_effect * 0.5})
            c2_bars = lcl.calibrate_bars(h, b1x, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=c_book)
            check(refuses(lambda: knob_claim(b1x, c2_bars, steer_spec, c_book), lkn.KnobRefused, "has changed since its first fresh reading"),
                  "2: a knob claim on a brief changed under its name is refused, as the weight path refuses it")
            f_book = tmp / "forbid.jsonl"
            f_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=f_book)
            f_seed = b"\x07" * 32
            seen_item = b1.generator.draw("kept_back", 4, seed_bytes=f_seed + b":target", tokenizer=h.tokenizer)[0].item
            lb.append("note", {"note_type": "practice_draw", "brief_id": b1.id, "brief_name": b1.name,
                               "item_ids": [seen_item.id], "frame_ids": [], "template_ids": [], "count": 1}, path=f_book)
            kf = knob_claim(b1, f_bars, mul_spec, f_book, _seed_override=f_seed)
            check(kf["verdict"] == "void" and any("fresh draw was refused" in r for r in kf["summary"]["reasons"]),
                  "2: a knob claim's draw is given the items and frames seen in practice, as a weight claim's is: one "
                  "drawn again ends the reading void")

            # kept-back wording in a knob's own text: a practice reading records it, and the claim-time look finds it
            p_book = tmp / "prompt_leak.jsonl"
            p_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=p_book)
            leak_prompt = f"Example: {kb_item.question}{kb_item.answer}"
            lkn.read_knob_practice(h, {"prompt": leak_prompt}, prac_map, brief=b1, rows=R816, book_path=p_book)
            check(refuses(lambda: knob_claim(b1, p_bars, mul_spec, p_book), lkn.KnobRefused, "were read in practice in this book")
                  and refuses(lambda: lcl.fresh_reading_gate(h, b1, p_bars, edit_w, p_book, what="claim"),
                              lcl.ClaimRefused, "were read in practice in this book"),
                  "2: kept-back wording in a knob practice reading's prompt (the kernel's own door) is found when a "
                  "fresh claim is asked for, knob or weight, and the claim is refused")
            pc_book = tmp / "prompt_claim.jsonl"
            pc_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=pc_book)
            check(refuses(lambda: knob_claim(b1, pc_bars, {**mul_spec, "prompt": leak_prompt}, pc_book), lkn.KnobRefused,
                          "would stand in the system turn of every fresh item") and knob_regs(pc_book) == 0,
                  "2: a knob claim whose own prompt holds a kept-back frame's wording is refused before registering")

            # the same fresh draw for a neuron_mul knob and its weight edit: the same controls, id for id
            s_seed = b"\x11" * 32
            sw_book, sk_book = tmp / "same_w.jsonl", tmp / "same_k.jsonl"
            sw_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=sw_book)
            sk_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=sk_book)
            eq_w = we.build(h.lm, [we.silence(L_last, [0, 1, 2])], model_id=h.model_id, weights_id=h.weights_id)
            lcl.run_fresh_claim(h, b1, sw_bars, eq_w, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                                rows={"small": 12, "large": 16}, match_tolerance=20.0, book_path=sw_book,
                                _seed_override=s_seed)
            w_ids = {(e["body"].get("controls") or {}).get("family"): (e["body"].get("controls") or {}).get("ids")
                     for e in lb.entries(sw_book) if e["kind"] == "paired" and str(e["body"].get("name", "")).startswith("claim_")}
            ks = knob_claim(b1, sk_bars, mul_spec, sk_book, rows={"small": 12, "large": 16}, _seed_override=s_seed)
            check(all(w_ids.get(f) and w_ids.get(f) == ks["summary"]["control_ids"].get(f)
                      for f in ("other_sites_matched", "same_site")),
                  "2: on the same fresh draw a neuron_mul knob and the weight edit whose arithmetic it is face the same "
                  "control edits, id for id, in both families")

            # the verdict: level_claim.judge_claim with the knob's families, then the knob gate's stricter rules
            jb = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=1, controls_per_family=19)
            jbars = lcl.make_bar_set(jb.id, near_miss_abs_max=1.0)  # a wide size bar, so the ratio is what bites
            RD = ("random_directions",)

            def jcls(eff: float = -1.0, rev: float = 0.5, spread: float = 0.0, ctl: float = 0.0, fams: Any = RD) -> Dict[str, Any]:
                rows_ = [{"frame_id": f"f{hf}{i}", "half": hf, "value": eff + (spread if i % 2 else -spread)}
                         for hf in (0, 1) for i in range(2)]
                return {c: {"item_effects": rows_, "floor": 0.0, "reversed_mean": rev,
                            "controls": {f: [ctl] * 19 for f in fams}} for c in ("small", "large")}

            def jbat(nm: float = 0.0, pkl: Any = 0.0, drop_neutral: bool = False) -> Dict[str, Any]:
                out_b = {"neutral_text": {"mean_kl": 0.0}, "near_misses": {"mean_effect": nm, "abs_effect": abs(nm)},
                         "labelled_panel": {"own_drop": 0.0, "mean_kl": pkl}, "habitat": {"available": False}}
                if drop_neutral:
                    out_b.pop("neutral_text")
                return out_b

            def jv(classes: Any, batteries: Any, fams: Any = RD, bs: Any = None) -> str:
                return lkn.judge_knob(brief=jb, bars=bs or jbars, classes=classes, batteries=batteries, families=fams)["verdict"]

            hab_bars = lcl.make_bar_set(jb.id, near_miss_abs_max=1.0,
                                        gating_batteries=("neutral_text", "near_misses", "labelled_panel", "habitat"))
            jcases = {
                "lead": jv(jcls(), jbat()),
                "ratio": jv(jcls(), jbat(nm=0.5)),
                "reversed": jv(jcls(rev=-1.0), jbat()),
                "ci": jv(jcls(spread=0.95), jbat()),
                "ci_engine": lcl.judge_claim(brief=jb, bars=jbars, classes=jcls(spread=0.95), batteries=jbat(),
                                             control_families=RD)["verdict"],
                "panel_kl": jv(jcls(), jbat(pkl=None)),
                "no_neutral": jv(jcls(), jbat(drop_neutral=True)),
                "habitat": jv(jcls(), jbat(), bs=hab_bars),
                "order": jv(jcls(rev=-1.0, ctl=2.0), jbat()),
                "family_missing": jv(jcls(fams=("other_sites_matched",)), jbat(), fams=("other_sites_matched", "same_site")),
            }
            check(jcases == {"lead": "lead", "ratio": "works but harms", "reversed": "did not replicate",
                             "ci": "did not replicate", "ci_engine": "lead", "panel_kl": "works but harms",
                             "no_neutral": "works but harms", "habitat": "works but harms", "order": "did not replicate",
                             "family_missing": "no better than control edits"},
                  f"2: a knob verdict is judge_claim's (near-miss ratio, reversed read, every gating battery the bars "
                  f"name, every family in both classes, its order) with the knob gate's stricter frame CI and panel KL "
                  f"kept ({jcases})")
            check(refuses(lambda: lkn.judge_knob(brief=jb, bars=jbars, classes=jcls(), batteries=jbat(), families=()),
                          lcl.ClaimRefused, "at least one named family"),
                  "2: a knob verdict with no family of controls is refused by the engine itself")
            kbat = lkn.knob_batteries({"near_misses": {"given": 2, "classes": {"small": {"mean": 0.0, "abs_mean": 0.0, "n_read": 2},
                                                                              "large": {"mean": 0.5, "abs_mean": 0.5, "n_read": 2}}},
                                       "labelled_panel": {"given": 2, "classes": {"small": {"own_mean": -0.3, "kl_mean": 0.01, "n_read": 2},
                                                                                 "large": {"own_mean": 0.0, "kl_mean": None, "n_read": 2}}}})
            check(kbat["near_misses"]["abs_effect"] == 0.5 and kbat["labelled_panel"]["own_drop"] == 0.3
                  and kbat["labelled_panel"]["mean_kl"] is None and kbat["habitat"]["available"] is False,
                  "2: a knob claim's batteries keep the larger of the two batch classes' figures, and a panel KL missing "
                  "in either class is missing")
            check("not an add to the neurons' activations" in lkn.steering_words(knob_nadd.steering_resolved[0])
                  and "at the output of layer" in lkn.steering_words(knob_nadd.steering_resolved[0]),
                  "2: neuron_add is said to be a fixed push along the neurons' summed write direction at the layer's "
                  "output, not an add to their activations")

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
            # a written knob's controls are judged: one mean a control over the items read (at f6d4e86 every one was
            # None and counted as moving as much, so every written steering knob failed its controls by construction)
            lp_kcl = lkn.run_fresh_knob_claim(
                h, lp_brief, lp_bars, {"steering": [{"kind": "neuron_mul", "layer": L_last, "neurons": list(range(16)), "factor": 0.0}],
                                       "prompt": "Be brief."},
                book_path=lp_book, **KC,
            )
            DOCS_COLLECTED.append(lp_kcl["sheet"])
            lp_cs = (lp_kcl["summary"]["control_standing"] or {}).get("written") or {}
            check(lp_kcl["verdict"] in lkn.KNOB_VERDICTS and lb.verify(lp_book)["ok"]
                  and set(lp_cs) == {"other_sites_matched", "same_site"}
                  and all(v.get("count") == lp_brief.default_controls and v.get("non_finite") == 0 for v in lp_cs.values()),
                  f"2: a written-loop steering knob's controls are judged, each by its own reading over the items read "
                  f"(no control left unread; verdict={lp_kcl['verdict']})")
            check(lkn.NOT_READ_WRITTEN_SAY in lp_kcl["sheet"] and lp_kcl["summary"]["settings"]["thinking"] is True,
                  "2: a written knob's sheet says what a written weight claim prints that it does not read, and the "
                  "chat settings it was read under")
            # thinking: knob and base are read under the brief's thinking setting, the knob's own when it has one
            neu_it = lbr.neutral_chat_generator().draw("kept_back", 1, seed_bytes=b"neutral-seed-xx", tokenizer=h.tokenizer)[0].item
            ref_p, kit_p = lkn.ref_and_knob_items(h, neu_it, lkn.build_knob(h, {"prompt": "Be brief."}), lp_brief)
            ref_t, kit_t = lkn.ref_and_knob_items(h, neu_it, lkn.build_knob(h, {"decoding": {"thinking": False}}), lp_brief)
            check(list(ref_p.prompt_ids) == lv.chat_ids(h.tokenizer, neu_it.question, None, thinking=True)
                  and list(kit_p.prompt_ids) == lv.chat_ids(h.tokenizer, neu_it.question, "Be brief.", thinking=True)
                  and ref_t.ids != kit_t.ids
                  and list(kit_t.prompt_ids) == lv.chat_ids(h.tokenizer, neu_it.question, None, thinking=False),
                  "2: on the thinking brief a prompt knob and its base are both read with thinking on, and a "
                  "thinking-off knob is read against a base with thinking on (not against itself)")
            check(refuses(lambda: lkn.write_knob_practice(h, {"decoding": {"thinking": True}}, {"w0": neu_it}, lp_brief,
                                                         book_path=tmp / "inert_w.jsonl"), lkn.KnobRefused, "cannot act"),
                  "2: a thinking setting the brief already reads with is refused as a part that cannot act")
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
                claim_design={"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "rows": {"small": 8, "large": 16},
                              "controls_override": 2, "match_tolerance": 20.0},
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
            # the same wording carried in the knob's prompt: refused at every knob door, and counted as an item's is
            kn_prac = {k: {"question": v["question"], "answer": v["answer"]}
                       for k, v in s_kn.draw_practice(2, seed=3)["items"].items()}
            kn_leak = {"prompt": f"Example: {kb_drawn.question}{kb_drawn.answer}"}
            kn_leak_claim = {**kn_leak, "steering": [{"kind": "neuron_mul", "layer": h.n_layers - 1, "neurons": [0, 1, 2], "factor": 0.0}]}
            gr0 = s_kn.gate_refusals
            kn_refused = [
                refuses(lambda: s_kn.knob_read(kn_leak, kn_prac, rows={"small": 8, "large": 16}), lses.SessionRefused,
                        "knob's prompt"),
                refuses(lambda: s_kn.knob_write(kn_leak, kn_prac, max_new=4), lses.SessionRefused, "knob's prompt"),
                refuses(lambda: s_kn.claim_knob(kn_leak_claim), lses.SessionRefused, "knob's prompt"),
            ]
            kn_logged = [json.loads(ln) for ln in (s_kn.session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
                         if '"knob_prompt"' in ln]
            check(all(kn_refused) and s_kn.gate_refusals - gr0 == 3 and len(kn_logged) == 3
                  and lcl.brief_history(brief.id, bk_kn, name=brief.name)["fresh_count"] == 0,
                  f"3: kept-back wording in a knob's prompt is refused by knob_read, knob_write and claim_knob, each "
                  f"refusal counted in gate_refusals and written to the transcript, and no fresh reading spent "
                  f"({kn_refused}, {s_kn.gate_refusals - gr0} counted)")
            kn_drv = s_kn.claim_knob({"steering": [{"kind": "neuron_mul", "layer": h.n_layers - 1, "neurons": [3, 4], "factor": 0.0}]})
            kn_info = s_kn.brief_info()
            check(kn_info["fresh_readings_used"] == 1 and kn_info["prior_verdicts"][0].get("knob_id") == kn_drv["knob_id"]
                  and kn_drv["summary"]["control_families"] == ["other_sites_matched", "same_site"],
                  "3: a knobs_only session's claim spends one of the brief's fresh readings, and brief_info counts and "
                  "lists it as a weight claim's would be")

            # Unix-socket MCP bridge (`session_socket_bridge` + subprocess `level_mcp.py --socket`): the bridge picks
            # its socket's place itself, a new 0700 folder outside the session folder, short enough for AF_UNIX
            br_seen: Dict[str, Any] = {}
            fwd_ok = False
            try:
                with lmcp.session_socket_bridge(s_kn) as br3:
                    sock_p = Path(br3.path)
                    br_seen = {
                        "sock_mode": oct(stat.S_IMODE(sock_p.stat().st_mode)),
                        "dir_mode": oct(stat.S_IMODE(sock_p.parent.stat().st_mode)),
                        "bytes": len(os.fsencode(str(sock_p))),
                        "in_session_folder": str(sock_p).startswith(str(s_kn.session_dir) + os.sep),
                    }
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
                    fwd_ok = (proc.returncode == 0 and "claim_knob" in t_names and "claim" not in t_names
                              and st_body.get("harness_card", {}).get("mode") == "knobs_only")
                br_seen["gone_after"] = not sock_p.exists() and not sock_p.parent.exists()
            except Exception as exc:  # at f6d4e86 the bridge took the socket's path from its caller
                br_seen["error"] = f"{type(exc).__name__}: {exc}"
            check(br_seen.get("sock_mode") == "0o600" and br_seen.get("dir_mode") == "0o700"
                  and br_seen.get("bytes", 999) <= lmcp.AF_UNIX_PATH_MAX and br_seen.get("in_session_folder") is False
                  and br_seen.get("gone_after") is True,
                  f"3: the bridge's socket is 0600 in a new 0700 folder of its own, outside the session folder, short "
                  f"enough for AF_UNIX, and gone with its folder on close ({br_seen})")
            check(fwd_ok, "3: level_mcp.py --socket bridges stdio JSON-RPC to live LevelSession cleanly")
            s_kn.close()

            # Two connections at once: the bridge takes one door call at a time, so the second cell sees the first's
            # spend and is refused (at f6d4e86 both ran against max_cells=1)
            if has_bwrap:
                s_race = lses.LevelSession(
                    None, tmp / "s_race", reach="sandbox_no_weights", backend="claude", book_path=tmp / "b_race.jsonl",
                    probe_on_start=False, budget=lses.SessionBudget(max_cells=1),
                )
                race_err: List[bool] = []

                def _race_client(path: Path, i: int) -> None:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
                        c.connect(str(path))
                        f = c.makefile("rw", encoding="utf-8")
                        f.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {
                            "name": "cell", "arguments": {"code": "import time\ntime.sleep(1.0)\nprint('ran')\n"}}}) + "\n")
                        f.flush()
                        race_err.append(bool(json.loads(f.readline())["result"].get("isError", False)))

                race_note = ""
                try:
                    with lmcp.session_socket_bridge(s_race) as br_r:
                        r_threads = [threading.Thread(target=_race_client, args=(br_r.path, i)) for i in (1, 2)]
                        for t_r in r_threads:
                            t_r.start()
                        for t_r in r_threads:
                            t_r.join(timeout=60.0)
                except Exception as exc:
                    race_note = f"{type(exc).__name__}: {exc}"
                check(s_race.cells_used == 1 and sorted(race_err) == [False, True],
                      f"3: two connections at once spend one cell against max_cells=1: the bridge takes one call at a "
                      f"time (cells used {s_race.cells_used}, refused {race_err.count(True)} of {len(race_err)}{race_note})")
                s_race.close()
            else:
                skip("3: bwrap is not installed: two cells at once over the bridge were not tried")

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
            check(oa_cap_out["stop_reason"] == "budget_dollars" and oa_cap_out["usage"]["turns"] == 0
                  and oa_cap_out["usage"]["estimated_dollars"] <= 0.001,
                  f"4: OpenAIChatDriver enforces max_dollars cap: a $0.001 cap cannot pay for the first request, so none "
                  f"is sent (stop={oa_cap_out['stop_reason']}, turns={oa_cap_out['usage']['turns']})")

            # OpenAI remote URL without max_dollars refused
            s_oa_rem = lses.LevelSession(h, tmp / "s_oa_rem", reach="behaviour", backend="openai", book_path=tmp / "b_oar.jsonl", probe_on_start=False)
            check(refuses(lambda: ldrv.run_openai_driver(
                s_oa_rem, model="x", base_url="https://openrouter.ai/api/v1", driver_budget=ldrv.DriverBudget(max_dollars=None),
            ), ldrv.DriverRefused, "max_dollars"),
                  "4: OpenAIChatDriver refuses remote endpoint without owner-approved max_dollars")
            s_oa_rem.close()

            # Nothing below leaves this machine: what it names is refused before a request, or is on 127.0.0.x, and
            # the proxy variables point at a port that records a connection and closes it (`dead`)
            @contextlib.contextmanager
            def _env(**kv: Optional[str]) -> Iterator[None]:
                old = {k: os.environ.get(k) for k in kv}
                try:
                    for k, v in kv.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                    yield
                finally:
                    for k, v in old.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v

            class _DeadProxy:
                """A stand-in proxy: it records each connection's first line and closes it, forwarding nothing."""

                def __init__(self) -> None:
                    self.hits: List[str] = []
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.bind(("127.0.0.1", 0))
                    self.sock.listen(16)
                    self.sock.settimeout(0.2)
                    self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
                    self.done = threading.Event()
                    threading.Thread(target=self._loop, daemon=True).start()

                def _loop(self) -> None:
                    while not self.done.is_set():
                        try:
                            c, _ = self.sock.accept()
                        except socket.timeout:
                            continue
                        except OSError:
                            return
                        try:
                            c.settimeout(0.5)
                            self.hits.append(c.recv(200).split(b"\r\n", 1)[0].decode("latin-1", "replace"))
                        except OSError:
                            self.hits.append("?")
                        finally:
                            c.close()

                def close(self) -> None:
                    self.done.set()
                    self.sock.close()

            dead = _DeadProxy()
            no_way_out = {"HTTPS_PROXY": dead.url, "HTTP_PROXY": dead.url, "https_proxy": dead.url, "http_proxy": dead.url,
                          "ALL_PROXY": None, "all_proxy": None,
                          "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
            sci_cfg = tmp / "scientist_cfg"  # no endpoints.json in it: the built-in endpoint profiles only
            sci_cfg.mkdir()

            with _env(THEODOLITE_SCIENTIST_DIR=str(sci_cfg), THEODOLITE_OPENAI_BASE_URL=None, **no_way_out):
                attack_urls = ("https://localhost.attacker.example/v1", "https://api.attacker.example/localhost/v1",
                               "https://x.example/v1?h=127.0.0.1")
                check(all(refuses(lambda u=u: ldrv._resolve_openai_profile(None, u, "OPENROUTER_API_KEY"), ldrv.DriverRefused)
                          for u in attack_urls),
                      "4: an endpoint whose address only contains localhost or 127.0.0.1 is not on this machine, and is "
                      "refused: no endpoint profile is at that address (at f6d4e86 all three counted as local)")
                local_urls = ("http://127.0.0.1:1234/v1", "http://localhost:11434/v1", "http://[::1]:8080/v1",
                              "http://127.7.7.7/v1")
                check(all(ldrv._resolve_openai_profile(None, u, "CHECK_LOCAL_KEY_VAR")[2] is True for u in local_urls),
                      "4: localhost, 127.0.0.0/8 and ::1 are on this machine, read from the parsed host")
                check(refuses(lambda: ldrv._resolve_openai_profile(None, "https://openrouter.ai/api/v1", "GITHUB_TOKEN"),
                              ldrv.DriverRefused, "cannot pair")
                      and refuses(lambda: ldrv._resolve_openai_profile("openrouter", "https://elsewhere.example/v1", None),
                                  ldrv.DriverRefused, "another address")
                      and ldrv._resolve_openai_profile(None, "https://openrouter.ai/api/v1/", None)[0]
                      == "https://openrouter.ai/api/v1/chat/completions",
                      "4: a plan cannot pair another variable with a remote address as its key, nor send a profile to "
                      "another address; a profile's own address and key variable are taken")

                def _priced_transport() -> Callable[..., Dict[str, Any]]:
                    sent = {"n": 0}

                    def _tr(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                        sent["n"] += 1
                        return {"model": "m", "usage": {"prompt_tokens": 400000, "completion_tokens": 0},
                                "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                                    {"id": f"c{sent['n']}", "type": "function", "function": {"name": "status", "arguments": "{}"}}]}}]}
                    _tr.sent = sent  # type: ignore[attr-defined]
                    return _tr

                s_oa_px = lses.LevelSession(h, tmp / "s_oa_px", reach="behaviour", backend="openai", book_path=tmp / "b_oapx.jsonl", probe_on_start=False)
                tr_px = _priced_transport()
                check(refuses(lambda: ldrv.run_openai_driver(
                    s_oa_px, model="m", transport_fn=tr_px,
                    driver_budget=ldrv.DriverBudget(max_turns=5, max_tokens=10 ** 9, max_dollars=0.01),
                ), ldrv.DriverRefused, "prices are not known")
                      and refuses(lambda: ldrv.run_openai_driver(
                          s_oa_px, model="m", profile="openrouter", driver_budget=ldrv.DriverBudget(max_dollars=1.0),
                      ), ldrv.DriverRefused, "prices are not known")
                      and tr_px.sent["n"] == 0 and not dead.hits,
                      "4: OpenAIChatDriver refuses a dollar cap it cannot keep: with no prices given, a mocked endpoint "
                      "under a cap and a remote one are both refused before any request (at f6d4e86 the cap counted 0 dollars)")
                s_oa_px.close()

                s_oa_pre = lses.LevelSession(h, tmp / "s_oa_pre", reach="behaviour", backend="openai", book_path=tmp / "b_oapre.jsonl", probe_on_start=False)
                pre_out = ldrv.run_openai_driver(
                    s_oa_pre, model="m", transport_fn=_priced_transport(),
                    driver_budget=ldrv.DriverBudget(max_turns=5, max_tokens=10 ** 9, max_dollars=5.0,
                                                    input_usd_per_mtok=10.0, output_usd_per_mtok=10.0),
                )
                check(pre_out["stop_reason"] == "budget_dollars" and pre_out["usage"]["turns"] == 1
                      and pre_out["usage"]["estimated_dollars"] <= 5.0,
                      f"4: OpenAIChatDriver sends no request that would pass max_dollars: $4 of $5 spent, and the next "
                      f"would send a $4 prompt again, so it stops after 1 turn (turns {pre_out['usage']['turns']}, "
                      f"${pre_out['usage']['estimated_dollars']})")

                # The first request is priced too: before any reply there is no last prompt to go by, and at the
                # mends' first round its prompt counted 0 tokens, so a session with little left still sent it
                s_oa_first = lses.LevelSession(h, tmp / "s_oa_first", reach="behaviour", backend="openai",
                                               book_path=tmp / "b_oafirst.jsonl", probe_on_start=False)
                tr_first = _priced_transport()
                first_out = ldrv.run_openai_driver(
                    s_oa_first, model="m", transport_fn=tr_first,
                    driver_budget=ldrv.DriverBudget(max_turns=5, max_tokens=10 ** 9, max_dollars=1.0,
                                                    input_usd_per_mtok=1000.0, output_usd_per_mtok=1000.0),
                )
                check(first_out["stop_reason"] == "budget_dollars" and tr_first.sent["n"] == 0
                      and first_out["usage"]["estimated_dollars"] == 0.0,
                      f"4: OpenAIChatDriver prices its first request from its own length: a prompt of thousands of "
                      f"tokens at $1,000 a million would pass a $1 cap, so nothing is sent (sent {tr_first.sent['n']}, "
                      f"stop {first_out['stop_reason']})")

                # A page that names a site says so: the brief's own flag, or a layer or neuron named by number
                s_site = lses.LevelSession(h, tmp / "s_site", reach="behaviour", backend="openai",
                                           book_path=tmp / "b_site.jsonl", probe_on_start=False)
                plain_pg = ldrv.build_driver_prompt(s_site)
                named_pg = ldrv.build_driver_prompt(s_site, extra_instructions="Start from neuron 5313 at layer 50.")
                s_site.close()
                check(plain_pg["prompt_named_site"] is False and plain_pg["sites_named"] == []
                      and named_pg["prompt_named_site"] is True and len(named_pg["sites_named"]) >= 2,
                      f"4: build_driver_prompt says when its page names a site ({named_pg['sites_named']}); at f6d4e86 "
                      f"it always said it named none")

                s_oa_nu = lses.LevelSession(h, tmp / "s_oa_nu", reach="behaviour", backend="openai", book_path=tmp / "b_oanu.jsonl", probe_on_start=False)
                nu_out = ldrv.run_openai_driver(
                    s_oa_nu, model="m",
                    transport_fn=lambda p, hd, u, t: {"model": "m", "choices": [{"message": {
                        "role": "assistant", "content": "", "tool_calls": [
                            {"id": "c", "type": "function", "function": {"name": "status", "arguments": "{}"}}]}}]},
                    driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, input_usd_per_mtok=1.0, output_usd_per_mtok=1.0),
                )
                check(nu_out["usage"]["turns"] == 1 and "token usage" in nu_out["notes"],
                      f"4: OpenAIChatDriver stops at a reply with no token usage, as its caps cannot be kept without it "
                      f"(turns {nu_out['usage']['turns']}, stop {nu_out['stop_reason']})")

                # A redirect is refused: at f6d4e86 urllib followed a 302 to a second address with the key
                second: Dict[str, Any] = {}

                class _SecondHost(BaseHTTPRequestHandler):
                    def _any(self) -> None:
                        second["auth_sent"] = self.headers.get("Authorization") is not None
                        self.send_response(200)
                        self.send_header("Content-Length", "2")
                        self.end_headers()
                        self.wfile.write(b"{}")

                    do_GET = do_POST = _any

                    def log_message(self, *_a: Any) -> None:
                        pass

                try:
                    srv_b = HTTPServer(("127.0.0.2", 0), _SecondHost)
                except OSError:
                    srv_b = HTTPServer(("127.0.0.1", 0), _SecondHost)
                b_host = f"{srv_b.server_address[0]}:{srv_b.server_address[1]}"

                class _Redirector(BaseHTTPRequestHandler):
                    def do_POST(self) -> None:  # noqa: N802
                        self.send_response(302)
                        self.send_header("Location", f"http://{b_host}/v1/chat/completions")
                        self.send_header("Content-Length", "0")
                        self.end_headers()

                    def log_message(self, *_a: Any) -> None:
                        pass

                srv_a = HTTPServer(("127.0.0.1", 0), _Redirector)
                for srv in (srv_a, srv_b):
                    threading.Thread(target=srv.serve_forever, daemon=True).start()
                redirect_err = ""
                try:
                    ldrv._default_openai_transport(
                        {"model": "m"}, {"Content-Type": "application/json", "Authorization": "Bearer check-dummy-key"},
                        f"http://127.0.0.1:{srv_a.server_address[1]}/v1/chat/completions", 5.0,
                    )
                except Exception as exc:
                    redirect_err = str(exc)
                for srv in (srv_a, srv_b):
                    srv.shutdown()
                    srv.server_close()
                check("redirect is refused" in redirect_err and not second,
                      f"4: the OpenAI transport refuses a redirect, so the key never reaches a second address "
                      f"(second address reached: {bool(second)})")

                # STOP while a request is out: looked at every 0.2 s, not after the request
                s_oa_stop = lses.LevelSession(h, tmp / "s_oa_stop", reach="behaviour", backend="openai", book_path=tmp / "b_oastop.jsonl", probe_on_start=False)
                ks_req = ldrv.KillSwitch()

                def _slow_transport(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    time.sleep(3.0)
                    return {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                            "choices": [{"message": {"role": "assistant", "content": "a late reply"}}]}

                t_q0 = time.monotonic()
                threading.Timer(0.3, ks_req.trigger, args=("check: STOP while a request is out",)).start()
                stop_out = ldrv.run_openai_driver(
                    s_oa_stop, model="m", transport_fn=_slow_transport, kill_switch=ks_req,
                    driver_budget=ldrv.DriverBudget(max_turns=3, max_dollars=1.0, input_usd_per_mtok=1.0, output_usd_per_mtok=1.0),
                )
                dt_q = time.monotonic() - t_q0
                check(stop_out["stop_reason"] == "killed" and dt_q < 1.5,
                      f"4: OpenAIChatDriver stops within 0.2 s of STOP while a request is out ({dt_q:.2f} s; the request "
                      f"takes 3 s)")

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

            def _mock(name: str, body: str) -> Path:
                p_m = tmp / name
                p_m.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
                os.chmod(p_m, 0o755)
                return p_m

            def _events(sess: lses.LevelSession) -> List[str]:
                return [json.loads(ln)["event"] for ln in (sess.session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]

            # A mock CLI that reports what reached it (variables, command line, working folder) and calls the door
            # through its MCP child, the way Claude Code does
            planted = {"HF_TOKEN": "hf_check_value", "OPENROUTER_API_KEY": "or_check_value", "RUNPOD_API_KEY": "rp_check_value",
                       "GITHUB_TOKEN": "gh_check_value", "OPENAI_API_KEY": "oa_check_value", "THEODOLITE_CHECK_STRAY": "stray"}
            watched = sorted(set(planted) | {lb.BOOK_ENV})
            mock_cli = _mock("mock_claude.py", f"""import json, os, subprocess, sys
leaked = [k for k in {watched!r} if k in os.environ]
switches = {{k: os.environ.get(k) for k in ("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")}}
cfg = json.loads(open(sys.argv[sys.argv.index("--mcp-config") + 1]).read())["mcpServers"]["levelling"]
req = "\\n".join([
    json.dumps({{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{}}}}),
    json.dumps({{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"status","arguments":{{}}}}}}),
]) + "\\n"
p = subprocess.run([cfg["command"], *cfg["args"]], input=req, text=True, capture_output=True, timeout=10)
ok_mcp = (p.returncode == 0 and "harness_card" in p.stdout)
print(json.dumps({{"type":"assistant","message":{{"id":"m1","model":"claude-mock-4","usage":{{"input_tokens":120,"output_tokens":45}}}}}}), flush=True)
seen = {{"ok_mcp": ok_mcp, "leaked": leaked, "cwd": os.getcwd(), "argv": sys.argv[1:], "switches": switches}}
print(json.dumps({{"type":"result","total_cost_usd":0.0123,"num_turns":1,"is_error":False,"result":json.dumps(seen)}}), flush=True)
""")
            with _env(**planted):
                s_cl = lses.LevelSession(h, tmp / "s_cl", reach="behaviour", backend="claude", book_path=tmp / "b_cl.jsonl", probe_on_start=False)
                cl_out = ldrv.run_claude_cli_driver(
                    s_cl, claude_bin=str(mock_cli), allow_mock_cli=True,
                    driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=15.0),
                )
            try:
                cl_seen = json.loads(str(cl_out.get("notes") or "").split("\n", 1)[0])
            except ValueError:
                cl_seen = {}
            cl_argv = [str(a) for a in cl_seen.get("argv") or []]

            def _flag(name: str) -> Optional[str]:
                """The value after `name` on the command line the CLI was given, or None."""
                return cl_argv[cl_argv.index(name) + 1] if name in cl_argv[:-1] else None

            door_cl = [f"mcp__levelling__{t}" for t in s_cl.allowed_tools()]
            check(cl_seen.get("ok_mcp") is True and cl_seen.get("cwd") == str(s_cl.work_dir)
                  and cl_out["harness_card"]["driver_model"] == "claude-mock-4"
                  and abs(cl_out["usage"]["estimated_dollars"] - 0.0123) < 1e-6
                  and cl_out["stop_reason"] == "kept_base_abstention" and (cl_out.get("door") or {}).get("ok") == 1,
                  "4: ClaudeCLIDriver bridges the door to the CLI's MCP child, pins cwd=work_dir, and logs the CLI's cost")
            check(cl_seen.get("leaked") == [] and (cl_seen.get("switches") or {}).get("CLAUDE_CODE_DISABLE_AUTO_MEMORY") == "1",
                  f"4: ClaudeCLIDriver starts claude with an allowlisted environment: none of {len(watched)} planted and "
                  f"study variables reaches it (GITHUB_TOKEN, OPENAI_API_KEY and the book's path among them), and "
                  f"auto-memory is switched off (reached: {cl_seen.get('leaked')})")
            check(_flag("--tools") == "" and _flag("--allowedTools") == ",".join(door_cl)
                  and _flag("--permission-mode") == "dontAsk" and _flag("--permission-prompts") == "none"
                  and _flag("--setting-sources") == "" and _flag("--max-budget-usd") == "1.000000"
                  and {"--strict-mcp-config", "--restricted", "--no-session-persistence", "--disable-slash-commands"} <= set(cl_argv)
                  and cl_argv[-2:-1] == ["--"] and not ({"Read", "Write", "Edit", "Bash"} & set(cl_argv)),
                  f"4: ClaudeCLIDriver's command line gives no built-in tool, allows the door's {len(door_cl)} tools and "
                  f"denies the rest (dontAsk), loads no settings, is restricted, saves no session, caps the dollars, and "
                  f"puts the prompt after --")

            # The same from a session folder long enough that `<session>/mcp.sock` would not fit AF_UNIX
            long_dir = tmp / ("study_folder_" + "s" * 36) / ("arm_weights_replicate_" + "r" * 36) / "session_0001"
            s_long = lses.LevelSession(h, long_dir, reach="behaviour", backend="claude", book_path=tmp / "b_long.jsonl", probe_on_start=False)
            long_note = ""
            try:
                long_out = ldrv.run_claude_cli_driver(
                    s_long, claude_bin=str(mock_cli), allow_mock_cli=True,
                    driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=15.0),
                )
            except Exception as exc:  # at f6d4e86: OSError, AF_UNIX path too long
                long_out, long_note = {}, f" ({type(exc).__name__}: {exc})"
            check((long_out.get("door") or {}).get("ok") == 1 and (long_out.get("door") or {}).get("socket_removed") is True,
                  f"4: ClaudeCLIDriver runs from a session folder {len(str(long_dir))} characters long: the socket is "
                  f"not in it{long_note}")

            # Denials and tool errors are read from the stream: a session whose door call was denied is a driver
            # failure (at f6d4e86 it read as an honest abstention, with tool_errors hard-coded 0)
            deny_cli = _mock("deny_claude.py", """import json, sys
door = sys.argv[sys.argv.index("--allowedTools") + 1].split(",") if "--allowedTools" in sys.argv else []
def say(ev):
    print(json.dumps(ev), flush=True)
say({"type": "system", "subtype": "init", "tools": door, "mcp_servers": [{"name": "levelling", "status": "connected"}], "model": "claude-mock-4"})
say({"type": "assistant", "message": {"id": "m1", "model": "claude-mock-4", "usage": {"input_tokens": 50, "output_tokens": 5},
     "content": [{"type": "tool_use", "id": "t1", "name": "mcp__levelling__status", "input": {}}]}})
say({"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
     "content": "Claude requested permissions to use mcp__levelling__status, but you haven't granted it yet."}]}})
say({"type": "assistant", "message": {"id": "m2", "model": "claude-mock-4", "usage": {"input_tokens": 60, "output_tokens": 5},
     "content": [{"type": "text", "text": "Keeping the base."}]}})
say({"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "total_cost_usd": 0.001, "result": "Keeping the base.",
     "permission_denials": [{"tool_name": "mcp__levelling__status", "tool_use_id": "t1", "tool_input": {}}]})
""")
            s_deny = lses.LevelSession(h, tmp / "s_deny", reach="behaviour", backend="claude", book_path=tmp / "b_deny.jsonl", probe_on_start=False)
            deny_out = ldrv.run_claude_cli_driver(
                s_deny, claude_bin=str(deny_cli), allow_mock_cli=True,
                driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=15.0),
            )
            check(deny_out["stop_reason"] == "driver_failure" and "denied" in str(deny_out.get("driver_failure"))
                  and deny_out["usage"]["tool_errors"] == 1 and deny_out["usage"]["turns"] == 2,
                  f"4: ClaudeCLIDriver reads permission denials and tool errors from the stream: a session whose door "
                  f"call was denied ends as a driver failure, not an honest abstention (stop {deny_out['stop_reason']}, "
                  f"tool errors {deny_out['usage']['tool_errors']})")

            # The CLI offering the model any tool that is not the door's stops the session at once
            extra_cli = _mock("extra_claude.py", """import json, sys, time
door = sys.argv[sys.argv.index("--allowedTools") + 1].split(",") if "--allowedTools" in sys.argv else []
print(json.dumps({"type": "system", "subtype": "init", "tools": door + ["Read", "Bash"],
                  "mcp_servers": [{"name": "levelling", "status": "connected"}]}), flush=True)
time.sleep(20)
""")
            s_extra = lses.LevelSession(h, tmp / "s_extra", reach="behaviour", backend="claude", book_path=tmp / "b_extra.jsonl", probe_on_start=False)
            t_x0 = time.monotonic()
            extra_out = ldrv.run_claude_cli_driver(
                s_extra, claude_bin=str(extra_cli), allow_mock_cli=True,
                driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=8.0),
            )
            dt_x = time.monotonic() - t_x0
            check(extra_out["stop_reason"] == "driver_failure" and "Bash" in str(extra_out.get("driver_failure")) and dt_x < 3.0,
                  f"4: ClaudeCLIDriver stops a CLI that offers the model tools beyond the door's ({dt_x:.2f} s: "
                  f"{extra_out.get('driver_failure')})")

            # The token cap is kept as the stream comes (at f6d4e86 a mock ran on to 16.5M tokens)
            spend_cli = _mock("spend_claude.py", """import json, time
for i in range(30):
    print(json.dumps({"type": "assistant", "message": {"id": f"m{i}", "model": "claude-mock-4",
                      "usage": {"input_tokens": 500000, "output_tokens": 50000}}}), flush=True)
    time.sleep(0.05)
print(json.dumps({"type": "result", "subtype": "success", "total_cost_usd": 250.0, "num_turns": 30, "is_error": False, "result": "done"}), flush=True)
""")
            s_spend = lses.LevelSession(h, tmp / "s_spend", reach="behaviour", backend="claude", book_path=tmp / "b_spend.jsonl", probe_on_start=False)
            spend_out = ldrv.run_claude_cli_driver(
                s_spend, claude_bin=str(spend_cli), allow_mock_cli=True,
                driver_budget=ldrv.DriverBudget(max_turns=40, max_tokens=1000, max_dollars=1.0, max_wall_seconds=30.0),
            )
            check(spend_out["stop_reason"] == "budget_tokens" and spend_out["usage"]["turns"] <= 2,
                  f"4: ClaudeCLIDriver keeps max_tokens as the stream comes: stopped at {spend_out['usage']['total_tokens']} "
                  f"tokens after {spend_out['usage']['turns']} turn(s) ({spend_out['stop_reason']})")

            # Hanging CLI wall-clock SIGKILL test
            hang_cli = _mock("hang_claude.py", "import time\ntime.sleep(30)\n")
            s_hang = lses.LevelSession(h, tmp / "s_hang", reach="behaviour", backend="claude", book_path=tmp / "b_hang.jsonl", probe_on_start=False)
            t_h0 = time.monotonic()
            hang_out = ldrv.run_claude_cli_driver(
                s_hang, claude_bin=str(hang_cli), allow_mock_cli=True,
                driver_budget=ldrv.DriverBudget(max_turns=2, max_dollars=1.0, max_wall_seconds=0.8),
            )
            dt_hang = time.monotonic() - t_h0
            check(hang_out["stop_reason"] == "budget_wall" and dt_hang < 4.0,
                  f"4: ClaudeCLIDriver SIGKILLs hanging subprocess on wall cap ({dt_hang:.2f}s)")

            # A door call still running when STOP kills the CLI: the mock asks for one call, waits until it is
            # running (its marker is in the work folder), then writes the STOP file
            inflight_cli = _mock("inflight_claude.py", """import json, os, subprocess, sys, time
spec = json.loads(open("mock_call.json").read())
cfg = json.loads(open(sys.argv[sys.argv.index("--mcp-config") + 1]).read())["mcpServers"]["levelling"]
p = subprocess.Popen([cfg["command"], *cfg["args"]], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\\n")
p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": spec["tool"], "arguments": spec["arguments"]}}) + "\\n")
p.stdin.flush()
print(json.dumps({"type": "assistant", "message": {"id": "m1", "model": "claude-mock-4", "usage": {"input_tokens": 10, "output_tokens": 2}}}), flush=True)
t0 = time.time()
while not os.path.exists(spec["marker"]) and time.time() - t0 < 60:
    time.sleep(0.05)
open(spec["stop_file"], "w").write("stop\\n")
time.sleep(60)
""")
            if has_bwrap:
                # a cell is killed: nothing it would have written lands in /work or the transcript after the outcome
                s_flc = lses.LevelSession(None, tmp / "s_fl_cell", reach="sandbox_no_weights", backend="claude",
                                          book_path=tmp / "b_flc.jsonl", probe_on_start=False)
                stop_c = tmp / "STOP_fl_cell"
                (s_flc.work_dir / "mock_call.json").write_text(json.dumps({
                    "tool": "cell", "marker": "started.txt", "stop_file": str(stop_c),
                    "arguments": {"code": "(WORK / 'started.txt').write_text('1')\nimport time\ntime.sleep(1.5)\n"
                                          "(WORK / 'late.txt').write_text('written after the stop')\n"},
                }), encoding="utf-8")
                flc_out = ldrv.run_claude_cli_driver(
                    s_flc, claude_bin=str(inflight_cli), allow_mock_cli=True, kill_switch=ldrv.KillSwitch(stop_file=stop_c),
                    driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=60.0),
                )
                started = s_flc.work_dir / "started.txt"
                if started.exists():  # past when the cell, left to run, would have written late.txt
                    time.sleep(max(0.0, started.stat().st_mtime + 2.0 - time.time()))
                run_c = (flc_out.get("door") or {}).get("running_at_stop") or {}
                ev_c = _events(s_flc)
                check(flc_out["stop_reason"] == "killed" and run_c.get("tool") == "cell" and run_c.get("stopped") is True
                      and not (s_flc.work_dir / "late.txt").exists() and ev_c[-1] == "session_close",
                      f"4: a cell still running when STOP kills the CLI is killed before the outcome is read: nothing "
                      f"lands in /work or the transcript after it (running at the stop: {run_c.get('tool')}, killed "
                      f"{run_c.get('stopped')}, last transcript event {ev_c[-1]})")
            else:
                skip("4: bwrap is not installed: a cell running at the stop was not tried")

            # a claim is waited for: its verdict is in the outcome and its edit is proposed_edit (at f6d4e86 the
            # outcome said no_submission and the claim landed in the book after it). The card's lock stands in for
            # a slow card: it marks the reading's start, then takes 2 s.
            fl_mark: Dict[str, Path] = {}

            @contextlib.contextmanager
            def _slow_card() -> Iterator[None]:
                if "path" in fl_mark:
                    fl_mark["path"].write_text("1", encoding="utf-8")
                time.sleep(2.0)
                yield

            bk_flr = tmp / "b_fl_claim.jsonl"
            shutil.copy2(cal_book, bk_flr)
            s_flr = lses.LevelSession(
                h, tmp / "s_fl_claim", reach="behaviour", backend="claude", book_path=bk_flr, brief=brief, bars=bars,
                probe_on_start=False, budget=lses.SessionBudget(min_controls=1), gpu_lock=_slow_card,
                claim_design={"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "rows": {"small": 12, "large": 16}, "controls_override": 2, "match_tolerance": 20.0},
            )
            fl_mark["path"] = s_flr.work_dir / "gpu_started.txt"
            stop_r = tmp / "STOP_fl_claim"
            fl_edit = {"terms": [{"layer": h.n_layers - 1, "silence": [0, 1, 2]}]}
            (s_flr.work_dir / "mock_call.json").write_text(json.dumps({
                "tool": "claim", "arguments": {"edit": fl_edit}, "marker": "gpu_started.txt", "stop_file": str(stop_r),
            }), encoding="utf-8")
            flr_out = ldrv.run_claude_cli_driver(
                s_flr, claude_bin=str(inflight_cli), allow_mock_cli=True, kill_switch=ldrv.KillSwitch(stop_file=stop_r),
                driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=120.0),
            )
            run_r = (flr_out.get("door") or {}).get("running_at_stop") or {}
            ev_r = _events(s_flr)
            check(flr_out["stop_reason"] == "killed" and flr_out["submitted"] is True and flr_out["verdict"] != "no_submission"
                  and run_r.get("tool") == "claim" and run_r.get("ok") is True and flr_out.get("proposed_edit") == fl_edit
                  and ev_r[-1] == "session_close",
                  f"4: a claim still reading when STOP kills the CLI is waited for: its verdict is in the outcome "
                  f"({flr_out['verdict']}), its edit is proposed_edit, and nothing lands after session_close (waited "
                  f"{(flr_out.get('door') or {}).get('waited_seconds')} s)")

            # The real claude CLI, against a local stand-in of the Messages API: a fake key, a home folder of its
            # own, and every proxy at `dead`. The stand-in asks for one door call (status), then for a built-in
            # Read of a file outside the work folder, then stops.
            claude_real = shutil.which("claude")
            if claude_real is None:
                skip("4: the claude CLI is not installed: its tools, door call and refused Read were not tried on a real run")
                skip("4: the claude CLI is not installed: --max-budget-usd was not tried on a real run")
            else:
                def _sse(msg: Dict[str, Any]) -> bytes:
                    start = dict(msg, content=[], stop_reason=None, usage=dict(msg["usage"], output_tokens=1))
                    evs: List[Dict[str, Any]] = [{"type": "message_start", "message": start}]
                    for i, blk in enumerate(msg["content"]):
                        if blk["type"] == "text":
                            evs += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                                    {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": blk["text"]}}]
                        else:
                            evs += [{"type": "content_block_start", "index": i, "content_block": {
                                        "type": "tool_use", "id": blk["id"], "name": blk["name"], "input": {}}},
                                    {"type": "content_block_delta", "index": i, "delta": {
                                        "type": "input_json_delta", "partial_json": json.dumps(blk["input"])}}]
                        evs.append({"type": "content_block_stop", "index": i})
                    evs += [{"type": "message_delta", "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                             "usage": {"output_tokens": msg["usage"]["output_tokens"]}},
                            {"type": "message_stop"}]
                    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in evs).encode("utf-8")

                class _MessagesStandIn:
                    """POST /v1/messages answered from `script(body) -> (content blocks, stop reason)`; every request
                    kept (its path, whether it carried the fake key, its JSON body); a path not served is noted."""

                    def __init__(self, script: Callable[[Dict[str, Any]], Any], key: str, in_tokens: int = 60) -> None:
                        self.requests: List[Dict[str, Any]] = []
                        self.unserved: List[str] = []
                        stand_in = self

                        class _H(BaseHTTPRequestHandler):
                            protocol_version = "HTTP/1.1"

                            def log_message(self, *_a: Any) -> None:
                                pass

                            def _reply(self, code: int, body: bytes, ctype: str) -> None:
                                self.send_response(code)
                                self.send_header("Content-Type", ctype)
                                self.send_header("Content-Length", str(len(body)))
                                self.send_header("Connection", "close")
                                self.end_headers()
                                self.wfile.write(body)
                                self.close_connection = True

                            def _any(self) -> None:
                                n_in = int(self.headers.get("Content-Length") or 0)
                                raw = self.rfile.read(n_in) if n_in > 0 else b""
                                try:
                                    body = json.loads(raw.decode("utf-8")) if raw else None
                                except ValueError:
                                    body = None
                                stand_in.requests.append({"method": self.command, "path": self.path, "body": body,
                                                          "fake_key": self.headers.get("x-api-key") == key})
                                if self.command == "POST" and self.path.split("?", 1)[0] == "/v1/messages" and isinstance(body, dict):
                                    blocks, stop = script(body)
                                    msg = {"id": f"msg_stand_in_{len(stand_in.requests)}", "type": "message", "role": "assistant",
                                           "model": str(body.get("model") or "stand-in"), "content": blocks, "stop_reason": stop,
                                           "stop_sequence": None, "usage": {"input_tokens": in_tokens, "output_tokens": 12,
                                                                            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
                                    if body.get("stream"):
                                        self._reply(200, _sse(msg), "text/event-stream")
                                    else:
                                        self._reply(200, json.dumps(msg).encode("utf-8"), "application/json")
                                    return
                                stand_in.unserved.append(f"{self.command} {self.path}")
                                self._reply(404, b'{"type":"error","error":{"type":"not_found_error","message":"not served"}}',
                                            "application/json")

                            do_GET = do_POST = do_HEAD = _any

                        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
                        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
                        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

                    def close(self) -> None:
                        self.httpd.shutdown()
                        self.httpd.server_close()

                    def result_of(self, tool_use_id: str) -> Dict[str, Any]:
                        for r in self.requests:
                            for m in (r["body"] or {}).get("messages") or []:
                                for blk in m.get("content") if isinstance(m.get("content"), list) else []:
                                    if isinstance(blk, dict) and blk.get("type") == "tool_result" and blk.get("tool_use_id") == tool_use_id:
                                        return blk
                        return {}

                cl_home = tmp / "claude_home"  # its home folder: nothing of this machine's own Claude Code
                (cl_home / ".claude").mkdir(parents=True)
                (cl_home / ".claude" / "CLAUDE.md").write_text("CHECK_MARK_HOME_CLAUDE_MD\n", encoding="utf-8")
                outside_f = tmp / "outside_the_work_folder.txt"
                outside_f.write_text("CHECK_MARK_OUTSIDE_FILE\n", encoding="utf-8")
                s_real = lses.LevelSession(h, tmp / "claude_proj" / "s_real", reach="behaviour", backend="claude",
                                           book_path=tmp / "b_real.jsonl", probe_on_start=False)
                (tmp / "claude_proj" / "CLAUDE.md").write_text("CHECK_MARK_PROJECT_CLAUDE_MD\n", encoding="utf-8")
                mem_dir = cl_home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(s_real.work_dir)) / "memory"
                mem_dir.mkdir(parents=True)
                (mem_dir / "MEMORY.md").write_text("CHECK_MARK_AUTO_MEMORY\n", encoding="utf-8")  # where its auto-memory looks
                marks = ("CHECK_MARK_HOME_CLAUDE_MD", "CHECK_MARK_PROJECT_CLAUDE_MD", "CHECK_MARK_AUTO_MEMORY", "CHECK_MARK_OUTSIDE_FILE")
                fake_key = "sk-ant-api03-" + "0" * 80 + "-checkAA"  # a key no service issued

                def _door_script(body: Dict[str, Any]) -> Any:
                    names = [str(t.get("name")) for t in body.get("tools") or [] if isinstance(t, dict)]
                    if not any(n.startswith("mcp__levelling__") for n in names):
                        return [{"type": "text", "text": "ok"}], "end_turn"
                    answered = {blk.get("tool_use_id") for m in body.get("messages") or []
                                for blk in (m.get("content") if isinstance(m.get("content"), list) else [])
                                if isinstance(blk, dict) and blk.get("type") == "tool_result"}
                    if "toolu_door_status" not in answered:
                        return [{"type": "tool_use", "id": "toolu_door_status", "name": "mcp__levelling__status", "input": {}}], "tool_use"
                    if "toolu_builtin_read" not in answered:
                        return [{"type": "tool_use", "id": "toolu_builtin_read", "name": "Read",
                                 "input": {"file_path": str(outside_f)}}], "tool_use"
                    return [{"type": "text", "text": "Stopping, and keeping the base."}], "end_turn"

                real_env = {"HOME": str(cl_home), "ANTHROPIC_API_KEY": fake_key, "ANTHROPIC_AUTH_TOKEN": None,
                            "CLAUDE_CODE_OAUTH_TOKEN": None, "CLAUDE_CONFIG_DIR": None, "XDG_CONFIG_HOME": None,
                            "XDG_CACHE_HOME": None, "XDG_DATA_HOME": None, "XDG_STATE_HOME": None, **no_way_out}
                cl_ver = subprocess.run([claude_real, "--version"], capture_output=True, text=True, timeout=60,
                                        env={"PATH": os.environ.get("PATH", ""), "HOME": str(cl_home), "DISABLE_AUTOUPDATER": "1",
                                             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}).stdout.strip() or "version unknown"
                stand_in = _MessagesStandIn(_door_script, fake_key)
                try:
                    with _env(ANTHROPIC_BASE_URL=stand_in.url, **real_env):
                        real_out = ldrv.run_claude_cli_driver(
                            s_real, claude_bin=claude_real,
                            driver_budget=ldrv.DriverBudget(max_turns=6, max_tokens=10_000_000, max_dollars=0.5, max_wall_seconds=120.0),
                        )
                finally:
                    stand_in.close()
                door_real = sorted(f"mcp__levelling__{t}" for t in s_real.allowed_tools())
                tool_sets = [sorted(str(t.get("name")) for t in r["body"]["tools"] if isinstance(t, dict))
                             for r in stand_in.requests if isinstance(r["body"], dict) and r["body"].get("tools")]
                st_res = stand_in.result_of("toolu_door_status")
                rd_res = stand_in.result_of("toolu_builtin_read")
                blob = json.dumps([r["body"] for r in stand_in.requests])
                check(bool(tool_sets) and all(ts == door_real for ts in tool_sets),
                      f"4: the real claude CLI ({cl_ver}) sends the model exactly the door's tools in all {len(tool_sets)} "
                      f"requests: no Read, Write, Edit, Bash, WebFetch, WebSearch or memory tool "
                      f"({tool_sets[0] if tool_sets else 'no request with tools'})")
                check((real_out.get("door") or {}).get("ok", 0) >= 1 and st_res and not st_res.get("is_error")
                      and "harness_card" in json.dumps(st_res.get("content"))
                      and real_out["stop_reason"] == "kept_base_abstention" and real_out.get("driver_failure") is None,
                      f"4: the real CLI's door call (status) went through the bridge and came back whole, so its ending is "
                      f"an honest one (door calls answered {(real_out.get('door') or {}).get('ok')}, stop {real_out['stop_reason']})")
                check(rd_res.get("is_error") is True and "CHECK_MARK_OUTSIDE_FILE" not in blob and real_out["usage"]["tool_errors"] >= 1,
                      f"4: the real CLI refused the built-in Read of a file outside the work folder "
                      f"({str(rd_res.get('content'))[:80]})")
                check(not any(mk in blob for mk in marks) and not dead.hits
                      and bool(stand_in.requests) and all(r["fake_key"] for r in stand_in.requests),
                      f"4: no CLAUDE.md, auto-memory or outside file reached the model, every request carried the fake key "
                      f"to the stand-in, and nothing tried to leave the machine (proxy connections {len(dead.hits)}; "
                      f"paths the stand-in did not serve: {stand_in.unserved or 'none'})")

                # --max-budget-usd on a real run: one reply priced past what is left, and the CLI stops itself
                def _spend_script(body: Dict[str, Any]) -> Any:
                    n_msgs = len(body.get("messages") or [])
                    return [{"type": "tool_use", "id": f"toolu_spend_{n_msgs}", "name": "mcp__levelling__status", "input": {}}], "tool_use"

                s_real2 = lses.LevelSession(h, tmp / "claude_proj" / "s_real2", reach="behaviour", backend="claude",
                                            book_path=tmp / "b_real2.jsonl", probe_on_start=False)
                stand_in2 = _MessagesStandIn(_spend_script, fake_key, in_tokens=200_000)
                try:
                    with _env(ANTHROPIC_BASE_URL=stand_in2.url, **real_env):
                        spend_real = ldrv.run_claude_cli_driver(
                            s_real2, claude_bin=claude_real,
                            driver_budget=ldrv.DriverBudget(max_turns=6, max_tokens=10_000_000, max_dollars=0.05, max_wall_seconds=120.0),
                        )
                finally:
                    stand_in2.close()
                n_main = sum(1 for r in stand_in2.requests if isinstance(r["body"], dict) and r["body"].get("tools"))
                check(spend_real["stop_reason"] == "budget_dollars" and n_main == 1 and not dead.hits,
                      f"4: the real CLI stops itself at --max-budget-usd: one reply priced past the $0.05 left, and no "
                      f"second request (requests {n_main}, ${spend_real['usage']['estimated_dollars']})")
            dead.close()

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
            check(lev.smallest_fisher_p(3, 3) == 0.1 and lev.smallest_fisher_p(5, 5) < 0.05,
                  f"5: at 3 sessions an arm the smallest two-sided Fisher p is 0.1 ({lev.smallest_fisher_p(3, 3)}); at 5, "
                  f"{lev.smallest_fisher_p(5, 5)}")

            # Comparison labels on made-up outcomes: no arm named ahead without p < 0.05, the same both ways
            def _syn(arm_id: str, fam: str, model: str, reach: str, mode: str, verdict: str, de: Optional[float],
                     effort: Optional[str] = None, rep: int = 0) -> Dict[str, Any]:
                return {"arm_id": arm_id, "replicate": rep, "verdict": verdict, "submitted": verdict != "no_submission",
                        "harness_card": {"driver_family": fam, "driver_model": model, "reach": reach, "mode": mode,
                                         "bare": False, "reasoning_effort": effort},
                        "metrics": {"directed_effect": de}, "usage": {"gpu_seconds": 2.0}, "book": {"verified": True}}

            def _syn_plan(*arms: tuple) -> Dict[str, Any]:
                return {"name": "syn", "sessions_per_arm": 3, "arms": [
                    {"arm_id": a, "driver_family": f, "driver_model": None, "reach": r, "mode": m, "bare": False,
                     "reasoning_effort": e, "driver_config": {}} for a, f, r, m, e in arms]}

            def _wk(w_rows: List[tuple], k_rows: List[tuple]) -> Dict[str, Any]:
                outs = [_syn("w", "scripted", "scripted_weight_v1", "weights", "weight_edit", v, d, rep=i) for i, (v, d) in enumerate(w_rows)]
                outs += [_syn("k", "scripted", "scripted_knob_v1", "weights", "knobs_only", v, d, rep=i) for i, (v, d) in enumerate(k_rows)]
                s = lev.summarize_study(outs, _syn_plan(("w", "scripted", "weights", "weight_edit", None),
                                                        ("k", "scripted", "weights", "knobs_only", None)))
                return s

            NS = ("no_submission", None)
            s_13 = _wk([("lead", 1.0), NS, NS], [NS, NS, NS])
            s_11 = _wk([NS], [("lead", 2.0)])
            s_55 = _wk([("lead", 1.0)] * 5, [NS] * 5)
            s_55r = _wk([NS] * 5, [("lead", 1.0)] * 5)
            lab = lambda s: s["comparisons"]["weight_vs_knobs"][0]["comparison"]  # noqa: E731
            check(lab(s_13) == "no_difference_shown" and lab(s_11) == "no_difference_shown"
                  and lab(s_55) == "weight_edit_ahead" and lab(s_55r) == "knobs_ahead"
                  and s_13["comparisons"]["weight_vs_knobs"][0]["test"]["fisher_exact_p"] == 1.0,
                  f"5: a comparison names an arm ahead only at two-sided Fisher p < 0.05 with 3+ sessions a side, both ways "
                  f"(1/3 vs 0/3: {lab(s_13)}; 0/1 vs 1/1: {lab(s_11)}; 5/5 vs 0/5: {lab(s_55)}; 0/5 vs 5/5: {lab(s_55r)})")
            check("no comparison in this study can separate two arms" in s_13["separation"]
                  and s_13["separation"] in s_13["limitations"]
                  and "no comparison in this study can separate two arms" in lev.render_study_report(s_13),
                  f"5: at N = 3 the summary and report say no comparison can separate the arms ({s_13['separation']})")
            s_harm = _wk([NS, NS, NS], [("works but harms", 9.0), ("works but harms", 9.2), ("works but harms", 9.4)])
            hrow = s_harm["comparisons"]["weight_vs_knobs"][0]
            s_zero = _wk([("lead", -1.0), NS, NS], [("lead", 0.0), ("lead", 0.0), NS])
            zrow = s_zero["comparisons"]["weight_vs_knobs"][0]
            harm_rep = lev.render_study_report(s_harm)
            check(hrow["knobs"]["lead_median_effect"] is None and hrow["knobs"]["leads"] == 0
                  and hrow["knobs"]["median_effect_all_readings"] == 9.2
                  and hrow["knobs"]["readings_by_verdict"] == {"works but harms": 3}
                  and hrow["comparison"] == "neither_found_a_lead" and "works but harms: 3" in harm_rep
                  and zrow["knobs"]["lead_median_effect"] == 0.0 and zrow["weight"]["lead_median_effect"] == -1.0
                  and "knobs_matched_or_ahead" not in json.dumps([s_harm, s_zero, s_11]),
                  f"5: \"works but harms\" readings are not leads, medians say what they are made of, a median of 0.0 stays 0.0 "
                  f"(knobs: {hrow['knobs']['readings_by_verdict']}, median of leads {hrow['knobs']['lead_median_effect']})")
            check(bool(hrow["drivers_differ"]) and "scripted_weight_v1" in hrow["drivers_differ"]
                  and "scripted_knob_v1" in hrow["drivers_differ"]
                  and "Mean GPU-s (weight, knobs)" in harm_rep and hrow["weight"]["mean_gpu_seconds"] == 2.0,
                  "5: a weight-vs-knobs row says when the two scripted arms are different programs and shows GPU-seconds beside them")
            s_eff = lev.summarize_study(
                [_syn("oa_hi", "openai", "m", "weights", "weight_edit", "lead", 1.0, effort="high", rep=i) for i in range(3)]
                + [_syn("oa_lo", "openai", "m", "weights", "weight_edit", "no_submission", None, effort="low", rep=i) for i in range(3)],
                _syn_plan(("oa_hi", "openai", "weights", "weight_edit", "high"), ("oa_lo", "openai", "weights", "weight_edit", "low")))
            check(len(s_eff["arms"]) == 2 and sorted(a["n_sessions"] for a in s_eff["arms"]) == [3, 3]
                  and {a["reasoning_effort"] for a in s_eff["arms"]} == {"high", "low"}
                  and s_eff["pooled_across_arms"] is False,
                  f"5: two arms that differ only in reasoning effort are two rows, never pooled ({[a['arm_id'] for a in s_eff['arms']]})")

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
            check(refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                name="bad_effort", brief_name="pronoun_pull", arms=(lev.EvalArm("a0", "scripted", reach="behaviour", reasoning_effort="high"),),
            )), lev.EvalRefused, "would run the same without it")
                  and refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                      name="two_efforts", brief_name="pronoun_pull", arms=(lev.EvalArm(
                          "a0", "openai", reach="behaviour", reasoning_effort="high", driver_config={"reasoning_effort": "low"}),),
                  )), lev.EvalRefused, "two reasoning efforts"),
                  "5: validate_study_plan refuses a reasoning effort a driver would not be given, and two efforts on one arm")
            check(refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                name="bad_wall", brief_name="pronoun_pull", sessions_per_arm=13,
                arms=(lev.EvalArm("a0", "scripted", reach="behaviour"),),
            )), lev.EvalRefused, "over max_total_wall_seconds"),
                  "5: validate_study_plan refuses a plan whose sessions times the driver's wall cap exceed the study's wall cap")
            ok_plan_doc = {**lev.EvalStudyPlan(name="doc", brief_name="pronoun_pull", sessions_per_arm=1,
                                               arms=(lev.EvalArm("a0", "scripted", reach="behaviour"),)).canonical_dict()}
            check(refuses(lambda: lev.study_plan_from_dict({**ok_plan_doc, "session_budget": {"max_gpu_secs": 5}}),
                          lev.EvalRefused, "not max_gpu_secs")
                  and refuses(lambda: lev.study_plan_from_dict({**ok_plan_doc, "driver_budget": {"max_turns": "5"}}),
                              lev.EvalRefused, "whole number")
                  and refuses(lambda: lev.study_plan_from_dict({**ok_plan_doc, "study_sha256": "0" * 64}),
                              lev.EvalRefused, "changed after it was written")
                  and lev.study_plan_from_dict({**ok_plan_doc}).sha256 == lev.study_plan_from_dict(json.loads(json.dumps(ok_plan_doc))).sha256,
                  "5: study_plan_from_dict refuses an unknown budget key, a value of the wrong kind and a changed plan in plain words")

            # Full multi-session study across all 4 reaches + knobs_only + --bare ablation + second driver family!
            study_cd = {"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "rows": {"small": 12, "large": 16}, "controls_override": 2, "match_tolerance": 20.0}
            study_sb = lses.SessionBudget(max_gpu_seconds=300.0, max_forwards=120, max_cells=10, min_controls=1)
            study_db = ldrv.DriverBudget(max_turns=16, max_dollars=1.0, input_usd_per_mtok=1.0, output_usd_per_mtok=2.0, max_wall_seconds=600.0)

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
                  and summ["pooled_across_arms"] is False and len(summ["arms"]) == 7
                  and sorted(a["arm_id"] for a in summ["arms"]) == sorted(a.arm_id for a in study_plan.arms),
                  f"5: 7-arm x 3-session (21 sessions) study completed with intact study_book and one row an arm")
            # Verify per-session book isolation: every session has its own verified level_book.jsonl and all 3 replicates ran!
            session_books = {o["book"]["path"] for o in sres["outcomes"]}
            check(len(session_books) == 21 and all(o["book"]["verified"] for o in sres["outcomes"]),
                  "5: all 21 sessions ran in strictly isolated, individually verified side level books")
            run_order = [o["run_id"] for o in sres["outcomes"]]
            check(run_order[:7] == [f"{a.arm_id}_r0" for a in study_plan.arms] and run_order[7] == f"{study_plan.arms[0].arm_id}_r1",
                  f"5: the arms are taken in turn, replicate by replicate, so a halt cannot drop whole arms ({run_order[:3]}...)")
            # Verify all four comparison tables populated!
            comps = summ["comparisons"]
            check(len(comps["weight_vs_knobs"]) >= 1
                  and len(comps["reach_ladders"]) >= 1
                  and comps["reach_ladders"][0]["atlas_vs_weights"] is not None
                  and len(comps["agent_vs_scripted"]) >= 1
                  and len(comps["scaffolding_ablations"]) >= 1,
                  "5: study summary populates weight_vs_knobs, reach_ladders (with atlas_vs_weights), agent_vs_scripted, and scaffolding_ablations")
            all_tests = ([r["test"] for r in comps["weight_vs_knobs"]] + [r["test"] for r in comps["agent_vs_scripted"]]
                         + [r["test"] for r in comps["scaffolding_ablations"]])
            check(all(t["smallest_possible_p"] == 0.1 and not t["can_separate"] and not t["label"].endswith("_ahead") for t in all_tests)
                  and "no comparison in this study can separate two arms" in sres["report"],
                  f"5: at 3 sessions an arm no row of the 21-session study names an arm ahead, and its report says why "
                  f"({sorted({t['label'] for t in all_tests})})")
            # Model-free reload from study_book.jsonl!
            reloaded = lev.load_study_from_book(study_dir)
            check(reloaded["book_verified"] and reloaded["session_books_checked"] == 21 and reloaded["finished"]
                  and reloaded["summary"]["completed_sessions"] == 21
                  and reloaded["summary"]["study_name"] == "p6_full_standin"
                  and reloaded["summary"]["bars_sha256"] == bars.id and "— None" not in reloaded["report"],
                  f"5: load_study_from_book reads the plan from the start note and checks all 21 side books "
                  f"(study {reloaded['summary']['study_name']!r})")
            check(refuses(lambda: lev.run_eval_study(h, study_plan, study_dir, weights_dir=weights_dir, atlas_dir=atlas_root,
                                                     bars=bars, calibration_book=cal_book, probe_sandbox=False),
                          lev.EvalRefused, "was used before")
                  and lev.load_study_from_book(study_dir)["summary"]["completed_sessions"] == 21,
                  "5: a second study into a used folder is refused before it writes anything")

            # A supplied calibration book with an earlier claim in it; a replay that only asks status; a driver refused
            # (an OpenAI arm with no endpoint); a reasoning effort that must reach the driver
            seeded_cal = tmp / "seeded_cal_book.jsonl"
            shutil.copy2(cal_book, seeded_cal)
            lb.append("note", {"note_type": "claim_verdict", "verdict": "lead", "edit_id": "AN_EARLIER_EDIT",
                               "summary": {"large": {"directed_mean": 3.3}}}, path=seeded_cal)
            eff_seen: List[Any] = []

            def _tr_effort(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                eff_seen.append(payload.get("reasoning_effort"))
                return {"model": "mock-effort", "usage": {"prompt_tokens": 30, "completion_tokens": 5},
                        "choices": [{"message": {"role": "assistant", "content": "Done."}}]}

            prev_oa_url = os.environ.pop("THEODOLITE_OPENAI_BASE_URL", None)
            mix_plan = lev.EvalStudyPlan(
                name="p6_mix", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=2,
                session_budget=study_sb, driver_budget=ldrv.DriverBudget(max_turns=8, max_dollars=0.0, max_wall_seconds=60.0),
                claim_design=study_cd,
                arms=(
                    lev.EvalArm("rp_status", "replay", reach="behaviour", driver_config={"steps": [{"tool": "status", "args": {}}]}),
                    lev.EvalArm("oa_high", "openai", reach="behaviour", driver_model="mock-effort", reasoning_effort="high",
                                driver_config={"transport_fn": _tr_effort}),
                    lev.EvalArm("oa_no_endpoint", "openai", reach="behaviour", driver_model="some-model"),
                ),
            )
            mix_dir = tmp / "p6_mix"
            try:
                mres = lev.run_eval_study(h, mix_plan, mix_dir, bars=bars, calibration_book=seeded_cal, probe_sandbox=False)
            finally:
                if prev_oa_url is not None:
                    os.environ["THEODOLITE_OPENAI_BASE_URL"] = prev_oa_url
            DOCS_COLLECTED.append(mres["report"])
            m_arms = {a["arm_id"]: a for a in mres["summary"]["arms"]}
            rp_outs = [o for o in mres["outcomes"] if o["arm_id"] == "rp_status"]
            check(len(rp_outs) == 2 and all(o["verdict"] == "no_submission" and not o["submitted"]
                                            and (o.get("seed_verdict_set_aside") or {}).get("verdict") == "lead" for o in rp_outs)
                  and m_arms["rp_status"]["n_leads"] == 0,
                  f"5: a session is scored on its own lines only: a replay that asked status alone is no lead, though its side "
                  f"book's copy of the calibration book holds one ({[o['verdict'] for o in rp_outs]})")
            refused_outs = [o for o in mres["outcomes"] if o.get("refused")]
            check(len(mres["outcomes"]) == 6 and len(refused_outs) == 2 and mres["halted_reason"] is None
                  and m_arms["oa_no_endpoint"]["n_sessions"] == 0 and m_arms["oa_no_endpoint"]["n_refused"] == 2
                  and (mix_dir / "summary.json").is_file() and "refused and not run" in mres["report"]
                  and lev.load_study_from_book(mix_dir)["finished"],
                  f"5: a driver's refusal is recorded for its session, not counted as run, and the study goes on to its "
                  f"end note and report ({[o['run_id'] for o in refused_outs]})")
            check(eff_seen == ["high", "high"] and m_arms["oa_high"]["reasoning_effort"] == "high",
                  f"5: an arm's reasoning effort reaches its driver (the endpoint saw {eff_seen})")

            # The side books are tied to the study book: a swapped or missing one, or a missing end note, is caught
            tie_dir = tmp / "p6_mix_copy"
            shutil.copytree(mix_dir, tie_dir)
            swap_p = tie_dir / "sessions" / "rp_status_r0" / "level_book.jsonl"
            shutil.copy2(cal_book, swap_p)
            swapped = refuses(lambda: lev.load_study_from_book(tie_dir), lev.EvalRefused, "changed or swapped")
            swap_p.unlink()
            missing = refuses(lambda: lev.load_study_from_book(tie_dir), lev.EvalRefused, "is missing")
            cut_dir = tmp / "p6_mix_cut"
            shutil.copytree(mix_dir, cut_dir)
            cut_lines = (cut_dir / "study_book.jsonl").read_text(encoding="utf-8").splitlines(True)
            (cut_dir / "study_book.jsonl").write_text("".join(cut_lines[:-1]), encoding="utf-8")
            cut = lev.load_study_from_book(cut_dir)
            check(swapped and missing and cut["finished"] is False and "Unfinished" in cut["report"]
                  and str(cut["summary"]["halted_reason"]).startswith("unfinished"),
                  "5: load_study_from_book refuses a swapped or missing side book and reports a study with no end note as unfinished")

            # A study dollar cap counts the session in progress: each session's cap is what is left of the study's
            def _tr_pay(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                return {"model": "mock-pay", "usage": {"prompt_tokens": 300, "completion_tokens": 50},
                        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                            {"id": "s1", "type": "function", "function": {"name": "status", "arguments": "{}"}}]}}]}

            pay_plan = lev.EvalStudyPlan(
                name="p6_pay", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=2,
                session_budget=study_sb, claim_design=study_cd, max_total_dollars=0.5,
                driver_budget=ldrv.DriverBudget(max_turns=8, max_dollars=1.0, max_wall_seconds=60.0,
                                                input_usd_per_mtok=1000.0, output_usd_per_mtok=1000.0),
                arms=(lev.EvalArm("oa_pay", "openai", reach="behaviour", driver_model="mock-pay", driver_config={"transport_fn": _tr_pay}),),
            )
            pres = lev.run_eval_study(h, pay_plan, tmp / "p6_pay", bars=bars, calibration_book=cal_book, probe_sandbox=False)
            p_caps = [((o.get("harness_card") or {}).get("driver_budget") or {}).get("max_dollars") for o in pres["outcomes"]]
            p_spent = [float((o.get("usage") or {}).get("estimated_dollars") or 0.0) for o in pres["outcomes"]]
            p_want = [round(0.5 - sum(p_spent[:i]), 6) for i in range(len(p_caps))]
            check(bool(p_caps) and p_caps == p_want and p_caps[0] == 0.5
                  and (len(p_caps) == 2 or pres["halted_reason"] == "study_dollar_cap"),
                  f"5: each session's dollar cap is what is left of a 0.5 study cap when it starts, not the plan's 1.0 a "
                  f"session (caps {p_caps} after spending {p_spent}; halted: {pres['halted_reason']})")

            # Study-level STOP file kill switch halts cleanly before calibration
            halt_dir = tmp / "p6_halt"
            halt_dir.mkdir()
            (halt_dir / "STOP").write_text("stop now\n", encoding="utf-8")
            mini_plan = lev.EvalStudyPlan(
                name="p6_halt_test", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=2,
                session_budget=study_sb, driver_budget=ldrv.DriverBudget(max_dollars=0.0), claim_design=study_cd,
                arms=(lev.EvalArm("scr_beh", "scripted", reach="behaviour", mode="weight_edit"),),
            )
            hres = lev.run_eval_study(h, mini_plan, halt_dir, bars=bars, calibration_book=cal_book, probe_sandbox=False)
            check(hres["halted_reason"] and "kill_switch" in hres["halted_reason"] and "before calibration" in hres["halted_reason"]
                  and len(hres["outcomes"]) == 0 and hres["book"]["verified"]
                  and not (halt_dir / "calibration_book.jsonl").exists() and (halt_dir / "report.md").is_file(),
                  f"5: a STOP file present at the start halts the study before calibration, with its end note and report ({hres['halted_reason']})")
            err_dir = tmp / "p6_error"
            real_run_driver = ldrv.run_driver

            def _broken_driver(*_a: Any, **_k: Any) -> Any:
                raise RuntimeError("a driver that broke")

            ldrv.run_driver = _broken_driver  # type: ignore[assignment]
            try:
                err_raised = refuses(lambda: lev.run_eval_study(
                    h, lev.EvalStudyPlan(name="p6_error", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=1,
                                         session_budget=study_sb, driver_budget=ldrv.DriverBudget(max_dollars=0.0), claim_design=study_cd,
                                         arms=(lev.EvalArm("rp", "replay", reach="behaviour", driver_config={"steps": []}),)),
                    err_dir, bars=bars, calibration_book=cal_book, probe_sandbox=False), RuntimeError, "a driver that broke")
            finally:
                ldrv.run_driver = real_run_driver  # type: ignore[assignment]
            err_load = lev.load_study_from_book(err_dir) if (err_dir / "study_book.jsonl").is_file() else None
            check(err_raised and err_load is not None and err_load["finished"]
                  and str(err_load["summary"]["halted_reason"]).startswith("error: RuntimeError")
                  and (err_dir / "report.md").is_file() and (err_dir / "summary.json").is_file(),
                  "5: an error in a session stops the study with its end note, summary and report written, and is raised again")
            we.assert_base(h.lm, h.hooks_at_load)

            # ========================================================================= 6. Blinded planted & canary eval
            section("6. Blinded planted-fault (healthy-ref & no-healthy-ref) and canary benchmark")
            # Planted silence on layer 0 neurons [2, 3]. A scale x2 there "repairs" it in the harness only, where deltas add
            # (0 + 2 - 1 = 1); a scale x1.5 leaves a reading to take (0 + 1.5 - 1 = 0.5)
            plant_rec = {"terms": [{"layer": 0, "silence": [2, 3]}]}
            repair_rec = {"terms": [{"layer": 0, "scale": [2, 3], "factor": 2.0}]}
            part_rec = {"terms": [{"layer": 0, "scale": [2, 3], "factor": 1.5}]}

            def _propose_steps(rec: Dict[str, Any]) -> Dict[str, Any]:
                return {"steps": [
                    {"tool": "status", "args": {}},
                    {"tool": "read", "propose": True, "args": {
                        "plan": {"name": "check_fix", "kind": "paired", "edit": rec, "items": ["p0"], "doses": [1.0],
                                 "rows": {"small": 8, "large": 16}, "controls": {"count": 1, "seed": 0}},
                        "items": {"p0": {"question": "Say hello.", "answer": "Hello there."}},
                    }},
                ]}

            status_only = {"steps": [{"tool": "status", "args": {}}]}
            b_cases = [
                lev.BlindedCase("planted_ref", "planted", reach="weights", plant_recipe=plant_rec,
                                driver_family="replay", driver_config=_propose_steps(repair_rec)),
                lev.BlindedCase("planted_ref_part", "planted", reach="weights", plant_recipe=plant_rec,
                                driver_family="replay", driver_config=_propose_steps(part_rec)),
                lev.BlindedCase("canary_ref", "canary", reach="weights", driver_family="replay", driver_config=status_only),
                lev.BlindedCase("planted_noref", "planted", reach="activations", plant_recipe=plant_rec, driver_family="scripted"),
                lev.BlindedCase("canary_noref", "canary", reach="activations", driver_family="replay", driver_config=status_only),
            ]
            b_summary = lev.run_blinded_benchmark(
                h, b_cases, tmp / "blinded_out", weights_dir=weights_dir,
                session_budget=lses.SessionBudget(min_controls=1),
                driver_budget=ldrv.DriverBudget(max_turns=12, max_dollars=0.0),
                probe_sandbox=False,
            )
            by_cid = {r["case_id"]: r for r in b_summary["cases"]}
            pr = by_cid["planted_ref"]
            pp = by_cid["planted_ref_part"]
            check(b_summary["n_cases"] == 5 and b_summary["n_planted"] == 3 and b_summary["n_canary"] == 2
                  and b_summary["door_surface_equal_by_reach"] == {"weights": True, "activations": True}
                  and "plumbing check" in b_summary["door_surface_note"] and "blinding_cards_equal_by_reach" not in b_summary,
                  f"6: planted and canary sessions show the same door surface at both reaches, named a plumbing check "
                  f"({b_summary['door_surface_equal_by_reach']})")
            check(b_summary["canary_abstention_rate"] is None and str(b_summary["canary_abstention_note"]).startswith("unmeasured")
                  and all(by_cid[c]["scored_as"] == "cannot propose" and by_cid[c]["honest_abstention"] is None
                          for c in ("canary_ref", "canary_noref")),
                  f"6: a canary driver with no way to propose is not scored as an honest abstainer: the rate is unmeasured "
                  f"({b_summary['canary_abstention_note']})")
            check(pr["scored_as"] == "plumbing check (answer given)" and "localization" not in pr
                  and (pr.get("answer_given_sites") or {}).get("site_recall") == 1.0
                  and by_cid["planted_noref"]["scored_as"] == "cannot propose"
                  and b_summary["planted_layer_hit_rate"] is None and str(b_summary["planted_layer_hit_note"]).startswith("unmeasured"),
                  f"6: a replay given the answer is a plumbing check, not localization, and no planted case is a measurement "
                  f"({b_summary['planted_scored_as']})")
            comp = pr.get("repair_composition") or {}
            check(pr.get("kl_plant_alone") is not None and pr["kl_plant_alone"] > 0.0
                  and pr.get("kl_after_repair") is None and pr.get("recovery_fraction") is None
                  and (pr.get("repair_reading") or {}).get("status") == "refused"
                  and "nothing to read" in str((pr.get("repair_reading") or {}).get("why"))
                  and comp.get("adds_to_no_change") is True and comp.get("repair_touches_plant_neurons") is True
                  and pr.get("recovery_holds_on_checkpoint") is False and "f1 + f2 - 1" in str(comp.get("rule")),
                  f"6: plant plus a scale x2 repair is read, not assigned 0: the harness finds nothing to read, and the "
                  f"recovery is marked as not holding on a checkpoint ({(pr.get('repair_reading') or {}).get('status')})")
            check(pp.get("kl_after_repair") is not None and 0.0 < pp["kl_after_repair"] < pp["kl_plant_alone"]
                  and pp.get("recovery_fraction") is not None and 0.0 < pp["recovery_fraction"] < 1.0
                  and pp.get("recovery_holds_on_checkpoint") is False
                  and (pp.get("repair_composition") or {}).get("adds_to_no_change") is False,
                  f"6: a repair the harness can read gets its figure, with the composition rule beside it "
                  f"(kl {pp.get('kl_plant_alone')} -> {pp.get('kl_after_repair')}, recovery {pp.get('recovery_fraction')}, "
                  f"holds on a checkpoint: {pp.get('recovery_holds_on_checkpoint')})")

            # The door-surface comparison reads the live session: a status reply that told the plant would be caught
            real_status = lses.LevelSession.status

            def _leaky_status(self: Any) -> Dict[str, Any]:
                out = real_status(self)
                out["planted"] = self._plant is not None
                return out

            lses.LevelSession.status = _leaky_status  # type: ignore[method-assign]
            try:
                leak_summary = lev.run_blinded_benchmark(
                    h, [lev.BlindedCase("canary_leak", "canary", reach="weights", driver_family="replay", driver_config=status_only),
                        lev.BlindedCase("planted_leak", "planted", reach="weights", plant_recipe=plant_rec,
                                        driver_family="replay", driver_config=status_only)],
                    tmp / "blinded_leak", weights_dir=weights_dir, probe_sandbox=False,
                )
            finally:
                lses.LevelSession.status = real_status  # type: ignore[method-assign]
            check(leak_summary["door_surface_equal_by_reach"].get("weights") is False,
                  f"6: a status reply that said which session is planted is caught ({leak_summary['door_surface_equal_by_reach']})")
            we.assert_base(h.lm, h.hooks_at_load)

            # ========================================================================= 7. Cold-driver, Rebuild, & CLI
            section("7. Cold-driver one-page brief, <= 200-word rebuild test, and owner CLI")
            import dataclasses as _dc
            import importlib.util as _ilu

            for bname, bobj in lbr.starter_briefs(h.tokenizer).items():
                cp = lev.render_cold_driver_page(bobj)
                DOCS_COLLECTED.append(cp["text"])
                check(cp["word_count"] <= lev.MAX_COLD_PAGE_WORDS and cp["prompt_named_site"] is False
                      and not cp["sites_named_in_page"] and "5313" not in cp["text"] and "layer 50" not in cp["text"].lower()
                      and "has not been run" in cp["ruling_test"],
                      f"7: cold-driver page for {bname} is <= 600 words ({cp['word_count']}), names no site, and says the "
                      "ruling's two-family test has not been run")
            sp_page = lev.render_cold_driver_page(_dc.replace(brief, driver_text="The fault is neuron 5313 at layer 50 of down_proj."))
            flag_page = lev.render_cold_driver_page(_dc.replace(brief, prompt_named_site=True))
            check(sp_page["prompt_named_site"] is True and {"neuron 5313", "layer 50"} <= set(sp_page["sites_named_in_page"])
                  and flag_page["prompt_named_site"] is True and flag_page["brief_says_prompt_named_site"] is True,
                  f"7: a page whose brief names a site says so, from the brief's flag and from its own text ({sp_page['sites_named_in_page']})")

            # Rebuild description and its same-process replay: two items, null rows other than the default
            rb_src_book = tmp / "rebuild_src.jsonl"
            rb_items = {
                "rb0": lv.make_item(h.tokenizer, "Answer briefly: Alice called Bob.", "She smiled.", name="rb0"),
                "rb1": lv.make_item(h.tokenizer, "Answer briefly: Clara met Dan.", "He waved.", name="rb1"),
            }
            rb_plan = {
                "name": "orig_for_rebuild", "kind": "paired",
                "edit": {"terms": [{"layer": h.n_layers - 1, "silence": [0, 1, 2]}]},
                "items": ["rb0", "rb1"], "doses": [1.0], "rows": {"small": 8, "large": 16}, "null_e": [0.25, -0.25],
                "readouts": {"kl": True, "own": True, "odds": None},
                "controls": {"family": "other_sites", "count": 2, "seed": 5},
            }
            with lev._use_book(rb_src_book):
                orig_r = lv.read(h, rb_plan, rb_items)
            orig_seq = int(orig_r["book"]["seq"])
            rb_desc = lev.make_rebuild_description(rb_src_book, orig_seq)
            DOCS_COLLECTED.append(rb_desc["description"])
            rb_spec = json.loads(rb_desc["description"].split("REBUILD_SPEC: ", 1)[1])
            check(rb_desc["word_count"] <= lev.MAX_REBUILD_WORDS and rb_spec["base"]["model"] == h.model_id
                  and rb_spec["plan"]["null_e"] == [0.25, -0.25] and "bars" in rb_spec["plan"] and "rank_by" in rb_spec["plan"]
                  and [it["name"] for it in rb_spec["items"]] == ["rb0", "rb1"],
                  f"7: the rebuild description ({rb_desc['word_count']} words) names its base and carries the null rows, "
                  "bars, rank rule and every item")
            syn_book = tmp / "rebuild_syn.jsonl"
            for syn_edit in ({"terms": [{"layer": 1, "silence": list(range(4000))}]},
                             {"terms": [{"layer": 1, "lowrank": {"A": "lr_A.npy", "B": "lr_B.npy"}}]}):
                lb.append("paired", {"status": "ok", "base": {"model": "m", "weights_id": None}, "plan": {"name": "s", "edit": syn_edit},
                                     "items": [{"item": {"name": "x", "question": "Q?", "answer": "A."}}]}, path=syn_book)
            check(refuses(lambda: lev.make_rebuild_description(syn_book, 0), lev.EvalRefused, "words (limit 200")
                  and refuses(lambda: lev.make_rebuild_description(syn_book, 1), lev.EvalRefused, "reads a file"),
                  "7: a 4,000-neuron recipe is counted as the thousands of words it is, and a recipe that needs a file is refused")
            rb_chk = lev.run_rebuild_replay(h, rb_desc["description"], rb_src_book, orig_seq, out_book=tmp / "rebuild_out.jsonl")
            check(rb_chk["ok"] and rb_chk["plan_same"] and rb_chk["items_compared"] == 2
                  and {"null:0.25", "null:-0.25", "edit:1.0", "control:0", "control:1"} <= set(rb_chk["arms_compared"])
                  and all(c["gap"] == 0.0 for c in rb_chk["comparisons"])
                  and "same-process replay" in rb_chk["what"] and "has not been run" in rb_chk["what"],
                  f"7: the same-process replay compares every arm of both items ({rb_chk['figures_compared']} figures, max gap "
                  f"{rb_chk['max_gap']}) and says the ruling's fresh-session test has not been run")
            tampered = rb_desc["description"].replace('"null_e":[0.25,-0.25]', '"null_e":[0.125,-0.125]')
            rb_bad = lev.run_rebuild_replay(h, tampered, rb_src_book, orig_seq, out_book=tmp / "rebuild_out_bad.jsonl")
            check(tampered != rb_desc["description"] and rb_bad["ok"] is False and rb_bad["plan_same"] is False,
                  "7: a description whose null rows are not the reading's does not pass the replay")

            # Owner CLI smoke tests (`scripts/levelling_eval.py`)
            cli_path = ROOT_DIR / "scripts" / "levelling_eval.py"
            cli_plan_p = tmp / "cli_plan.json"
            bad_cli_plan_p = tmp / "cli_bad_plan.json"
            p_cli1 = subprocess.run(
                [sys.executable, "-I", str(cli_path), "plan",
                 "--name", "cli_smoke", "--brief", "pronoun_pull", "--sessions", "1",
                 "--driver", "scripted", "--claim-controls", "2", "--claim-n-target", "4",
                 "--max-gpu-seconds", "120", "--out", str(cli_plan_p)],
                text=True, capture_output=True, timeout=30.0,
            )
            p_cli1b = subprocess.run(
                [sys.executable, "-I", str(cli_path), "plan", "--sessions", "0", "--out", str(bad_cli_plan_p)],
                text=True, capture_output=True, timeout=30.0,
            )
            p_cli2 = subprocess.run(
                [sys.executable, "-I", str(cli_path), "cold-page", "--brief", "pronoun_pull"],
                text=True, capture_output=True, timeout=30.0,
            )
            p_cli3 = subprocess.run(
                [sys.executable, "-I", str(cli_path), "report", str(study_dir)],
                text=True, capture_output=True, timeout=30.0,
            )
            p_cli4 = subprocess.run(
                [sys.executable, "-I", str(cli_path), "rebuild",
                 "--book", str(rb_src_book), "--seq", str(orig_seq), "--replay", "--stand-in",
                 "--out-book", str(tmp / "cli_rb.jsonl")],
                text=True, capture_output=True, timeout=60.0,
            )
            check(p_cli1.returncode == 0 and cli_plan_p.is_file()
                  and p_cli1b.returncode != 0 and "refused:" in p_cli1b.stderr and not bad_cli_plan_p.exists()
                  and p_cli2.returncode == 0 and "Levelling Cold-Driver" in p_cli2.stdout and "has not been run" in p_cli2.stdout
                  and p_cli3.returncode == 0 and "## Arms" in p_cli3.stdout
                  and p_cli4.returncode == 0 and "same-process replay" in p_cli4.stdout,
                  f"7: levelling_eval.py plan validates (a 0-session plan is refused and not written), and cold-page, report "
                  f"and rebuild --replay --stand-in exit 0 (rc={p_cli1.returncode},{p_cli1b.returncode},{p_cli2.returncode},"
                  f"{p_cli3.returncode},{p_cli4.returncode})")

            # `run` and `knobs` refuse before the model is loaded or the card's lock taken
            cli_spec = _ilu.spec_from_file_location("levelling_eval_cli_in_check", cli_path)
            cli = _ilu.module_from_spec(cli_spec)
            cli_spec.loader.exec_module(cli)  # type: ignore[union-attr]
            loads: List[str] = []

            def _no_load(*_a: Any, **_k: Any) -> Any:
                loads.append("load")
                raise RuntimeError("the model would be loaded here")

            def _cli_says(argv: List[str]) -> str:
                try:
                    cli.main(argv)
                except SystemExit as exc:
                    return str(exc.code)
                except Exception as exc:
                    return f"{type(exc).__name__}: {exc}"
                return "0"

            home_plan_p = tmp / "cli_home_plan.json"
            home_plan_p.write_text(json.dumps({**ok_plan_doc, "arms": [{"arm_id": "a0", "driver_family": "scripted", "reach": "home"}]}), encoding="utf-8")
            typo_plan_p = tmp / "cli_typo_plan.json"
            typo_plan_p.write_text(json.dumps({**ok_plan_doc, "session_budget": {"max_gpu_secs": 5}}), encoding="utf-8")
            ok_plan_p = tmp / "cli_ok_plan.json"
            ok_plan_p.write_text(json.dumps(ok_plan_doc), encoding="utf-8")
            stop_out = tmp / "cli_stop_out"
            stop_out.mkdir()
            (stop_out / "STOP").write_text("left over\n", encoding="utf-8")
            real_hold = lmcp.hold_card
            cli._load_handle = _no_load
            lmcp.hold_card = _no_load  # type: ignore[assignment]
            try:
                r_home = _cli_says(["run", str(home_plan_p), "--out", str(tmp / "cli_home_out")])
                r_typo = _cli_says(["run", str(typo_plan_p), "--out", str(tmp / "cli_typo_out")])
                r_stop = _cli_says(["run", str(ok_plan_p), "--out", str(stop_out)])
                r_used = _cli_says(["run", str(ok_plan_p), "--out", str(study_dir)])
                r_knob = _cli_says(["knobs", "--brief", "pronoun_pull", "--knob", str(ok_plan_p), "--bars", str(ok_plan_p), "--stand-in"])
            finally:
                lmcp.hold_card = real_hold  # type: ignore[assignment]
            check(not loads and r_home.startswith("refused:") and "home reach" in r_home
                  and r_typo.startswith("refused:") and "not max_gpu_secs" in r_typo
                  and r_stop.startswith("refused:") and "STOP" in r_stop and sorted(p.name for p in stop_out.iterdir()) == ["STOP"]
                  and r_used.startswith("refused:") and "was used before" in r_used and r_knob == "2",
                  f"7: run refuses a bad plan, an unknown budget key, a leftover STOP file and a used folder before loading the "
                  f"model, and knobs without --book is refused (loads={len(loads)}; {r_typo[:60]!r})")
            kn_cwd = tmp / "cli_knob_cwd"
            kn_cwd.mkdir()
            kn_book_dir = tmp / "cli_knob_book"
            kn_book_dir.mkdir()
            kn_knob = tmp / "cli_knob.json"
            kn_knob.write_text(json.dumps({"no_such_knob_key": 1}), encoding="utf-8")
            kn_bars = tmp / "cli_knob_bars.json"
            lcl.save_bars(bars, kn_bars)
            p_kn = subprocess.run(
                [sys.executable, "-I", str(cli_path), "knobs", "--stand-in", "--brief", "pronoun_pull",
                 "--knob", str(kn_knob), "--bars", str(kn_bars), "--book", str(kn_book_dir / "book.jsonl")],
                text=True, capture_output=True, timeout=90.0, cwd=str(kn_cwd),
            )
            check(p_kn.returncode != 0 and "refused:" in p_kn.stderr and not list(kn_cwd.iterdir())
                  and not any(p.name.startswith("_stand_in") for p in kn_book_dir.iterdir()),
                  f"7: knobs --stand-in writes nothing into the working folder or beside its book "
                  f"(rc={p_kn.returncode}; {p_kn.stderr.strip()[-80:]!r})")

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
    skipped = f", {len(SKIPPED)} not run here" if SKIPPED else ""
    print(f"\nPhase 6 CPU checks: {PASSED} passed, {FAILED} failed{skipped} in {dt:.1f} s.", flush=True)
    for msg in SKIPPED:
        print(f"  not run: {msg}", flush=True)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
