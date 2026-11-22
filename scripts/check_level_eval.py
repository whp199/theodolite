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
6. Blinded planted-fault and canary benchmark: plumbing checks (neutral case ids, the same door surface, the composition rule).
7. The cold-driver page (at most 600 words), the rebuild description and its same-process replay, and the owner's
   command line (`scripts/levelling_eval.py`).
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
            # Layer 1 sits below the stand-in's scaled label-gate floor (so its lens labels are set aside) and its steer
            # check agrees: its steer words must be kept, since the label gate does not read the steer lens
            agree_s = {1, n_l_s - 1}
            standing_s = {str(i): {"standing": ("agrees" if i in agree_s else "weak" if i == n_l_s - 2 else "unchecked")}
                          for i in range(n_l_s)}
            (st_idx / "steer_check.json").write_text(json.dumps(
                {"lens": {"built_at": "2026-11-13T23:43:53Z"}, "layers": standing_s}), encoding="utf-8")
            st_tid = int(st_up[n_l_s - 1, 0, 0])
            st_tid1 = int(st_up[1, 0, 0])
            with la.AtlasReader(st_idx) as st_reader:
                all_s = st_reader.search_words(f"id:{st_tid}", source="steer", top_k=64)
                kept_s = st_reader.search_words(f"id:{st_tid}", source="steer", top_k=64, gate_passes_only=True)
                kept_s1 = st_reader.search_words(f"id:{st_tid1}", source="steer", top_k=64, gate_passes_only=True)
                card_s = st_reader.neuron_card(n_l_s - 2, 0, top_wiring=0)
                low_hit = next((h for h in kept_s1["hits"] if h["layer"] == 1 and h["neuron"] == 0), None)
                check(all(h["steer_check"]["set_aside"] == (h["layer"] not in agree_s) for h in all_s["hits"])
                      and all(h["layer"] in agree_s for h in kept_s["hits"]) and kept_s["total_hits"] >= 1
                      and card_s["steer_check"]["standing"] == "weak" and card_s["steer_check"]["set_aside"] is True,
                      f"1: steer words are set aside where the steer check found the lens weak or unchecked "
                      f"({all_s['total_hits']} hits, {kept_s['total_hits']} kept)")
                check(low_hit is not None and low_hit["label_gate"]["held_back"] is True
                      and low_hit["steer_check"]["standing"] == "agrees" and low_hit["steer_check"]["set_aside"] is False,
                      "1: a steer word at a layer the steer check backs is kept though the label gate sets that face's "
                      "lens label aside: the steer check alone sets steer words aside, as the instrument's pages do")
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
            kv1 = [e["body"] for e in lb.entries(kcl_book) if (e.get("body") or {}).get("note_type") == "knob_claim_verdict"]
            check(kcl1["verdict"] != "void" and len(kv1) == 1 and kv1[0]["verdict"] == kcl1["verdict"]
                  and s1["control_families"] == ["other_sites_matched", "same_site"]
                  and all(len(s1["control_ids"].get(f) or []) == brief.default_controls for f in s1["control_families"])
                  and set(s1["control_standing"]) == {"small", "large"}
                  and all(set(v) == {"other_sites_matched", "same_site"} for v in s1["control_standing"].values()),
                  f"2: a neuron_mul knob claim reads (not void; the book's verdict note says what it returned) against "
                  f"its weight edit's two families of controls (matched other sites, same-site directions), "
                  f"{brief.default_controls} a family, in both batch classes (verdict={kcl1['verdict']})")
            check({r for w, r in spy_rows if w == "battery"} == {8, 16} and {r for w, r in spy_rows if w == "target"} == {8, 16},
                  f"2: a knob claim reads its retention batteries in both batch classes, as a weight claim does "
                  f"(rows read: {sorted(spy_rows)})")
            check(refuses(lambda: knob_claim(brief, bars, mul_spec, kcl_book), lkn.KnobRefused, "read once"),
                  "2: a knob is read once on a brief: the gate refuses it by its knob id")

            # a neuron_mul knob and the weight edit whose arithmetic it is (its twin) are read once between them, and a
            # trial follows the knob's claim only as it follows the edit's own (at d5e6848 the gate was asked with the
            # knob's id alone: after the knob read "kept the base" a claim and a trial of the edit were both accepted,
            # and after the edit's claim the knob was)
            eq_mul = we.build(h.lm, [we.silence(L_last, [0, 1, 2])], model_id=h.model_id, weights_id=h.weights_id)
            tw_b = lbr.pronoun_pull_brief(h.tokenizer, max_fresh_readings=4, controls_per_family=4)
            tw_book = tmp / "twin_knob_first.jsonl"
            tw_bars = lcl.calibrate_bars(h, tw_b, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=tw_book)
            tw_k = knob_claim(tw_b, tw_bars, mul_spec, tw_book, _seed_override=b"\x21" * 32)
            check(tw_k["verdict"] == "kept the base" and tw_k["summary"]["twin_edit_id"] == eq_mul.id
                  and refuses(lambda: lcl.fresh_reading_gate(h, tw_b, tw_bars, eq_mul, tw_book, what="claim"),
                              lcl.ClaimRefused, "by a knob claim of its arithmetic")
                  and refuses(lambda: lcl.fresh_reading_gate(h, tw_b, tw_bars, eq_mul, tw_book, what="trial", main_dose=1.0),
                              lcl.ClaimRefused, 'read "kept the base"')
                  and refuses(lambda: knob_claim(tw_b, tw_bars, {**mul_spec, "prompt": "Answer plainly."}, tw_book),
                              lkn.KnobRefused, "the weight edit whose arithmetic this knob is")
                  and spent(tw_book, tw_b) == 1,
                  f"2: after a neuron_mul knob claim read \"{tw_k['verdict']}\", a claim and a trial of its twin weight "
                  "edit are refused (read once; a trial follows a lead), and so is a knob of the same arithmetic with a "
                  "prompt added")
            tw_book2 = tmp / "twin_edit_first.jsonl"
            tw_bars2 = lcl.calibrate_bars(h, tw_b, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=tw_book2)
            lcl.run_fresh_claim(h, tw_b, tw_bars2, eq_mul, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                                rows={"small": 12, "large": 16}, match_tolerance=20.0, book_path=tw_book2)
            tw_book3 = tmp / "twin_trial_first.jsonl"
            tw_bars3 = lcl.calibrate_bars(h, tw_b, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=tw_book3)
            lb.append("note", {"note_type": "trial_registration", "brief_id": tw_b.id, "brief_name": tw_b.name,
                               "edit_id": eq_mul.id, "reading_index": 1, "kept_back": {"template_ids": []}}, path=tw_book3)
            check(refuses(lambda: knob_claim(tw_b, tw_bars2, mul_spec, tw_book2), lkn.KnobRefused, "by a weight claim")
                  and refuses(lambda: knob_claim(tw_b, tw_bars3, mul_spec, tw_book3), lkn.KnobRefused,
                              "has already had a trial")
                  and knob_regs(tw_book2) == 0 and knob_regs(tw_book3) == 0,
                  "2: after a weight claim of an edit, or a trial of it, a neuron_mul knob that is its arithmetic is "
                  "refused before registering (read once, both ways)")
            # a trial after the twin's lead: it promotes what the knob read, the edit's arithmetic alone at dose 1
            tw_said: Dict[str, Any] = {}
            for tag, kspec in (("alone", mul_spec), ("with a prompt", {**mul_spec, "prompt": "Answer plainly."})):
                kn_t = lkn.build_knob(h, kspec)
                tw_bk = tmp / f"twin_lead_{tag[:5]}.jsonl"
                tw_bs = lcl.calibrate_bars(h, tw_b, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=tw_bk)
                reg_t = lb.append("note", {"note_type": "knob_claim_registration", "brief_id": tw_b.id,
                                           "brief_name": tw_b.name, "knob_id": kn_t.id, "knob": kn_t.to_dict(),
                                           "twin_edit_id": eq_mul.id, "reading_index": 1,
                                           "kept_back": {"template_ids": []}}, path=tw_bk)
                lb.append("note", {"note_type": "knob_claim_verdict", "brief_id": tw_b.id, "brief_name": tw_b.name,
                                   "knob_id": kn_t.id, "twin_edit_id": eq_mul.id, "registration_seq": reg_t["seq"],
                                   "verdict": "lead"}, path=tw_bk)
                for dose in (1.0, 2.0):
                    try:
                        lcl.fresh_reading_gate(h, tw_b, tw_bs, eq_mul, tw_bk, what="trial", main_dose=dose)
                        tw_said[f"{tag} {dose:g}"] = "taken"
                    except lcl.ClaimRefused as exc:
                        tw_said[f"{tag} {dose:g}"] = ("more than" if "more than the edit's arithmetic" in str(exc) else
                                                       "dose" if "at the dose it read it" in str(exc) else str(exc)[:60])
            check(tw_said == {"alone 1": "taken", "alone 2": "dose", "with a prompt 1": "more than",
                              "with a prompt 2": "more than"},
                  f"2: a trial of the edit follows its knob twin's lead when the knob was the edit's arithmetic alone, at "
                  f"dose 1, and not after a lead read with a prompt as well ({tw_said})")

            # an item is kept by the weight path's rule (levelling._stability): every read-out the brief has on, the
            # knob's arm and its reversed read (at d5e6848 the knob's arm alone was asked, on the brief's metric)
            st_seen: Dict[str, Any] = {}
            for tag, arm_n, ro_n in (("kl", "knob", "kl"), ("reversed", "reversed", "odds")):
                def far_large(*a: Any, _arm: str = arm_n, _ro: str = ro_n, **k: Any) -> Any:
                    res = orig_arms(*a, **k)
                    if res.get("scores") and "_tgt" in str(a[1].name) and int(a[6]) == R816["large"]:
                        v = (res["scores"].get(_arm) or {}).get(_ro)
                        if v is not None:
                            res["scores"][_arm][_ro] = v * 50.0 + (1.0 if _ro == "kl" else -5.0)
                    return res

                st_bk = tmp / f"stability_{tag}.jsonl"
                st_bs = lcl.calibrate_bars(h, tw_b, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=st_bk)
                lkn._eval_paired_arms = far_large
                try:
                    st_k = knob_claim(tw_b, st_bs, mul_spec, st_bk)
                finally:
                    lkn._eval_paired_arms = orig_arms
                st_seen[tag] = (st_k["summary"]["unstable_target"], st_k["verdict"])
            check(st_seen == {"kl": (KC["n_target"], "void"), "reversed": (KC["n_target"], "void")},
                  f"2: a target item whose large class reads far from its small class on another read-out (KL) or on "
                  f"the reversed arm is unstable and left out, as a weight reading leaves it out ({st_seen})")

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
            # the reversed read reverses the steering alone: the bias stays, as on every control row (at d5e6848 it was
            # negated, so a push whose reverse moves the metric as far passed the reversed read on the bias alone)
            rv2 = ((kcl2["summary"]["reversed"] or {}).get("large") or {}).get("directed")
            check(rv2 is not None and abs(rv2 - (kcl2["summary"]["large_directed"] or 0.0)) <= 0.05 * eff2
                  and kcl2["verdict"] == "did not replicate"
                  and any("reversed dose moved in the same target direction" in r for r in kcl2["summary"]["reasons"]),
                  f"2: a negligible push with a blunt bias: its reversed read keeps the bias and moves the metric as the "
                  f"knob does ({rv2} beside {kcl2['summary']['large_directed']}), so it does not replicate")
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
            # another brief's kept-back wording: the claim asks the practice gates' own guard (every starter brief's
            # frames), and a knob claim's prompt is text read for every other brief (at d5e6848 the claim looked at the
            # bound brief's frames alone, and knob claim notes were not looked through)
            spider = lbr.starter_briefs(h.tokenizer)["spider_legs_bleed"]
            sp_kb = spider.generator.draw("kept_back", 1, seed_bytes=b"other-brief-kept-back", tokenizer=h.tokenizer)[0].item
            sp_prompt = f"Example: {sp_kb.question}{sp_kb.answer}"
            check(refuses(lambda: knob_claim(b1, pc_bars, {**mul_spec, "prompt": sp_prompt}, pc_book), lkn.KnobRefused,
                          'of brief "spider_legs_bleed"') and knob_regs(pc_book) == 0,
                  "2: a knob claim whose prompt holds another brief's kept-back wording is refused before registering, "
                  "by whichever door it comes in")
            ob2_book = tmp / "other_brief_knob.jsonl"
            ob2_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=ob2_book)
            sp_bars = lcl.calibrate_bars(h, spider, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=ob2_book)
            sp_knob = lkn.build_knob(h, {**mul_spec, "prompt": sp_prompt})
            lb.append("note", {"note_type": "knob_claim_registration", "brief_id": b1.id, "brief_name": b1.name,
                               "knob_id": sp_knob.id, "knob": sp_knob.to_dict(), "reading_index": 1,
                               "design": {"settings": {"system": None, "thinking": False}},
                               "kept_back": {"template_ids": []}}, path=ob2_book)
            check(refuses(lambda: lcl.fresh_reading_gate(h, spider, sp_bars, edit_w, ob2_book, what="claim"),
                          lcl.ClaimRefused, "stand in the prompt of a knob claim of another brief")
                  and lcl.fresh_reading_gate(h, b1, ob2_bars, edit_w, ob2_book, what="claim")["reading_index"] == 2,
                  "2: a book where a knob claim of one brief read another brief's kept-back wording in its prompt refuses "
                  "that other brief's fresh claims, and not the knob's own brief's")

            # the longest fresh and battery items are tried under the knob's text before registering (at d5e6848 one
            # practice item was: a prompt that fit it was registered, and the reading went void on a longer item)
            import itertools as _its
            pre_it = b1.generator.draw("practice", 1, seed_bytes=b"preflight_knob", tokenizer=h.tokenizer)[0].item
            neu_gen = lbr.neutral_chat_generator()
            neu_long: Any = None
            for fr in neu_gen.frames_for("kept_back"):
                phs = sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", fr.question_tpl + fr.answer_tpl)))
                for vals in _its.product(*[neu_gen.slots[p] for p in phs]):
                    sl = dict(zip(phs, vals))
                    cand = lv.make_item(h.tokenizer, fr.question_tpl.format(**sl), fr.answer_tpl.format(**sl),
                                        system=fr.system)
                    neu_long = cand if neu_long is None or len(cand.ids) > len(neu_long.ids) else neu_long
            long_prompt = None
            for n_w in range(1, 900):
                p_try = ("steady " * n_w).strip()
                fits_pre = refuses(lambda: lkn.ref_and_knob_items(h, pre_it, lkn.build_knob(h, {"prompt": p_try}), b1),
                                   lkn.KnobRefused) is False
                fits_neu = refuses(lambda: lv.make_item(h.tokenizer, neu_long.question, neu_long.answer, system=p_try),
                                   lv.Refused) is False
                if fits_pre and not fits_neu:
                    long_prompt = p_try
                    break
            long_book = tmp / "long_prompt.jsonl"
            long_bars = lcl.calibrate_bars(h, b1, layer=L_last, n_cols=4, n_controls=2, rows=R816, book_path=long_book)
            check(long_prompt is not None
                  and refuses(lambda: knob_claim(b1, long_bars, {**mul_spec, "prompt": long_prompt}, long_book),
                              lkn.KnobRefused, "Nothing was registered")
                  and knob_regs(long_book) == 0 and spent(long_book, b1) == 0,
                  f"2: a knob prompt ({len(long_prompt or '')} characters) that fits a practice item but not the longest "
                  f"item the claim can draw is refused before registering, and no fresh reading is spent")

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
            # what neuron_add's words say it is, read: the push a direction knob makes along the neurons' summed write
            # direction (their down_proj columns times the post-feedforward gain, off the quiet mask), at the same length
            from backend.gguf_weights import dense_cols
            lay_m = h.lm.layers[L_last]
            gain_na = lay_m.post_feedforward_layernorm.weight.detach().to("cpu", dtype=torch.float64)

            def write_sum(ns: List[int]) -> List[float]:
                cols_na = dense_cols(lay_m.mlp.down_proj, torch.as_tensor(ns, dtype=torch.long)).T.detach().to(
                    "cpu", dtype=torch.float64)
                s_na = (gain_na[:, None] * cols_na).sum(dim=1).numpy()
                s_na[we.quiet_mask(h.lm, L_last)] = 0.0
                return s_na.tolist()

            na_book = tmp / "neuron_add.jsonl"
            push_of = lambda st: lkn.read_knob_practice(h, {"steering": [st]}, prac_map, brief=brief, controls=0,
                                                        rows=R816, book_path=na_book)
            r_na = push_of({"kind": "neuron_add", "layer": L_last, "neurons": [0, 1], "dose": 1.0, "scale": 40.0})
            r_dir = push_of({"kind": "direction", "layer": L_last, "vector": write_sum([0, 1]), "dose": 1.0, "scale": 40.0})
            r_oth = push_of({"kind": "direction", "layer": L_last, "vector": write_sum([2, 3]), "dose": 1.0, "scale": 40.0})
            na_m, dir_m, oth_m = (r["small"]["mean"] for r in (r_na, r_dir, r_oth))
            check(all(r["status"] == "ok" for r in (r_na, r_dir, r_oth)) and None not in (na_m, dir_m, oth_m)
                  and abs(na_m - dir_m) <= 1e-4 * max(1.0, abs(na_m)) and abs(na_m - oth_m) > 1e-3
                  and abs(r_na["large"]["mean"] - r_dir["large"]["mean"]) <= 1e-4 * max(1.0, abs(na_m)),
                  f"2: a neuron_add knob reads as a fixed push along its own neurons' summed write direction at the "
                  f"layer's output, as its words say, and not as one along other neurons' ({na_m} beside {dir_m}; "
                  f"other neurons' {oth_m})")

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
            check(lp_kcl["verdict"] != "void" and lb.verify(lp_book)["ok"]
                  and set(lp_cs) == {"other_sites_matched", "same_site"}
                  and all(v.get("count") == lp_brief.default_controls and v.get("non_finite") == 0 for v in lp_cs.values()),
                  f"2: a written-loop steering knob's controls are judged, each by its own reading over the items read "
                  f"(no control left unread; verdict={lp_kcl['verdict']})")
            # each statement on a sheet is true of the reading it is printed with (at d5e6848 a written sheet said the
            # matched draw was on what a stand-in delivers over the answer, the rule held "in both batch classes" and
            # the stricter frame CI was the large class's)
            w_sheet, p_sheet = lp_kcl["sheet"], kcl1["sheet"]
            w_words = {
                "over the prompt": "over the prompt" in w_sheet, "no 'over the answer'": "over the answer" not in w_sheet,
                "the written class": "in the written class, the one class a written reading has" in w_sheet,
                "no 'both batch classes' rule": "at most alpha, in both batch classes" not in w_sheet,
                "frame CI (written class)": "frame CI (written class)" in w_sheet,
                "no '(large class)'": "(large class)" not in w_sheet, "no 'small ='": "small = " not in w_sheet,
                "its chunks": f"in chunks of {lp_kcl['summary']['control_chunk']}" in w_sheet,
                "not the edit alone's": "not the stand-ins a written weight claim of the edit alone draws" in w_sheet,
                "lists what a written claim prints": lkn.NOT_READ_WRITTEN_SAY in w_sheet,
                "no item gate listing (a written claim has none)": lkn.NOT_READ_SAY[2] not in w_sheet,
                "no delivered listing (a written claim prints none)": lkn.NOT_READ_SAY[1] not in w_sheet,
            }
            p_words = {
                "over the answer": "expects it to deliver over the answer" in p_sheet,
                "both batch classes": "at most alpha, in both batch classes" in p_sheet,
                "frame CI (large class)": "frame CI (large class)" in p_sheet,
                "no written listing": lkn.NOT_READ_WRITTEN_SAY not in p_sheet,
                "no thinking listing": "thinking setting is not read on them" not in p_sheet,
                "lists the item gate and delivered": lkn.NOT_READ_SAY[1] in p_sheet and lkn.NOT_READ_SAY[2] in p_sheet,
                "lists the prompt arms (the brief has one)": bool(brief.prompt_arms) and lkn.NOT_READ_SAY[0] in p_sheet,
            }
            check(all(w_words.values()) and all(p_words.values()) and lp_kcl["summary"]["settings"]["thinking"] is True,
                  f"2: a written knob sheet says what is true of a written reading (its pre-pass over the prompt, the "
                  f"written class, its chunks), a paired sheet what is true of a paired one, and each lists what a "
                  f"weight claim of its kind prints that it does not read "
                  f"({[k for k, v in {**w_words, **p_words}.items() if not v]} wrong)")

            # the retention batteries are read as the bars were calibrated: thinking off, whatever the brief's setting
            # (at d5e6848 the thinking brief's batteries were read with thinking on, against bars calibrated off)
            bat_seen: List[bool] = []
            orig_arms_b = lkn._eval_paired_arms

            def spy_bat(*a: Any, **k: Any) -> Any:
                ref_b = a[1]
                as_calibrated = lv.chat_ids(h.tokenizer, ref_b.question, ref_b.system, thinking=False)
                bat_seen.append(list(ref_b.prompt_ids) == as_calibrated)
                return orig_arms_b(*a, **k)

            th_book = tmp / "thinking_batteries.jsonl"
            th_bars = lcl.calibrate_bars(h, lp_brief, layer=L_last, n_cols=4, n_controls=1, rows=R816, book_path=th_book)
            lkn._eval_paired_arms = spy_bat
            try:
                th_k = lkn.run_fresh_knob_claim(
                    h, lp_brief, th_bars, {"steering": [{"kind": "neuron_mul", "layer": L_last, "neurons": [1, 2],
                                                         "factor": 0.0}], "decoding": {"thinking": False}},
                    book_path=th_book, **KC)
            finally:
                lkn._eval_paired_arms = orig_arms_b
            check(len(bat_seen) == 2 * (KC["n_neutral"] + KC["n_near"] + KC["n_panel"]) and all(bat_seen)
                  and "thinking setting is not read on them" in th_k["sheet"],
                  f"2: on the thinking brief the retention batteries are read with thinking off, as the bars were "
                  f"calibrated, and a knob's thinking setting is said not to be read on them "
                  f"({sum(bat_seen)} of {len(bat_seen)} battery readings with thinking off)")

            # on the same fresh draw a written neuron_mul knob and its weight edit face the same controls, in the written
            # claim's chunks and seeds (at d5e6848 the knob drew all at seeds +0 and +1, on a pre-pass with thinking on)
            sw_seed = b"\x13" * 32
            w_edit16 = we.build(h.lm, [we.silence(L_last, list(range(16)))], model_id=h.model_id, weights_id=h.weights_id)
            ww_book, wk_book = tmp / "same_written_w.jsonl", tmp / "same_written_k.jsonl"
            ww_bars = lcl.calibrate_bars(h, lp_brief, layer=L_last, n_cols=4, n_controls=1, rows=R816, book_path=ww_book)
            wk_bars = lcl.calibrate_bars(h, lp_brief, layer=L_last, n_cols=4, n_controls=1, rows=R816, book_path=wk_book)
            ww = lcl.run_fresh_claim(h, lp_brief, ww_bars, w_edit16, n_target=4, n_neutral=2, n_near=2, n_panel=2,
                                     rows=R816, match_tolerance=20.0, controls_override=7, book_path=ww_book,
                                     _seed_override=sw_seed)
            ww_ids: Dict[str, List[str]] = {}
            for e_w in lb.entries(ww_book):
                cw = (e_w.get("body") or {}).get("controls") or {}
                if e_w["kind"] == "written" and str(e_w["body"].get("name", "")).startswith("claim_"):
                    ww_ids.setdefault(cw.get("family"), []).extend(cw.get("ids") or [])
            wk = lkn.run_fresh_knob_claim(
                h, lp_brief, wk_bars, {"steering": [{"kind": "neuron_mul", "layer": L_last, "neurons": list(range(16)),
                                                     "factor": 0.0}]},
                book_path=wk_book, controls_override=7, _seed_override=sw_seed, **KC)
            check(ww["verdict"] != "void" and wk["verdict"] != "void"
                  and all(len(ww_ids.get(f) or []) == 7 and ww_ids.get(f) == wk["summary"]["control_ids"].get(f)
                          for f in ("other_sites_matched", "same_site")),
                  f"2: on the same fresh draw a written neuron_mul knob and the weight edit whose arithmetic it is face "
                  f"the same 7 control edits a family, id for id, over two chunks (verdicts {ww['verdict']}, "
                  f"{wk['verdict']})")

            # a sampler is compared by what it does: greedy is greedy whatever else is set (at d5e6848 only a sampler
            # spelled as the brief's own was refused)
            mul_34 = [{"kind": "neuron_mul", "layer": L_last, "neurons": [3, 4], "factor": 0.0}]
            greedy_said = {tag: refuses(lambda: lkn.run_fresh_knob_claim(
                h, lp_brief, lp_bars, {"steering": mul_34, "decoding": {"sampler": smp}}, book_path=lp_book, **KC),
                lkn.KnobRefused, "both are greedy")
                for tag, smp in (("temperature 0, top_k 5", {"temperature": 0.0, "top_k": 5}),
                                 ("temperature 0.7, top_k 1", {"temperature": 0.7, "top_k": 1}))}
            acts = lkn.parts_that_cannot_act(lkn.build_knob(h, {**mul_spec, "decoding": {"sampler": {"temperature": 0.7,
                                                                                                      "top_k": 5}}}),
                                             lp_brief, written=True, vocab_size=int(h.W_U.shape[0]))
            check(all(greedy_said.values()) and acts == [] and spent(lp_book, lp_brief) == 1,
                  f"2: on a greedy written brief a knob sampler that is greedy too (temperature 0, or top_k 1) is refused "
                  f"before registering, and one that samples is not ({greedy_said})")

            # every batch of written arms is laid out as a weight claim's: untouched first and last rows, which must
            # match each other to the bit and write the base row's pieces (at d5e6848 a batch of arms had none)
            orig_write = lkn._write_rows_with_bias
            moved: Dict[str, Any] = {}
            for tag, which in (("the twin parts from the first row", "last"), ("both part from the base row", "untouched")):
                def moved_rows(hh: Any, rows_ids: Any, progs: Any, _w: str = which, **k: Any) -> Any:
                    out_w = orig_write(hh, rows_ids, progs, **k)
                    if len(rows_ids) > 2:
                        for r_w in range(len(out_w)):
                            if (_w == "last" and r_w == len(out_w) - 1) or (_w == "untouched" and progs[r_w] is None):
                                pcs = list(out_w[r_w]["pieces"])
                                out_w[r_w] = {**out_w[r_w], "pieces": pcs[:-1] + [int(pcs[-1]) ^ 1]}
                    return out_w

                mv_book = tmp / f"moved_{which}.jsonl"
                mv_bars = lcl.calibrate_bars(h, lp_brief, layer=L_last, n_cols=4, n_controls=1, rows=R816, book_path=mv_book)
                lkn._write_rows_with_bias = moved_rows
                try:
                    mv = lkn.run_fresh_knob_claim(
                        h, lp_brief, mv_bars, {"steering": [{"kind": "neuron_mul", "layer": L_last, "neurons": [5, 6],
                                                             "factor": 0.0}]}, book_path=mv_book, **KC)
                finally:
                    lkn._write_rows_with_bias = orig_write
                moved[tag] = (mv["verdict"], (mv["summary"]["reasons"] or [""])[0][:60])
            mv_twin, mv_base = moved["the twin parts from the first row"], moved["both part from the base row"]
            check(mv_twin[0] == "void" and "untouched twin row" in mv_twin[1]
                  and mv_base[0] == "void" and "untouched first row" in mv_base[1],
                  f"2: a written knob reading is void when a batch of arms' untouched twin parts from its untouched first "
                  f"row, or when they part alike from the base row ({moved})")
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
            # the session's door: a knob of the same arithmetic is its twin edit read again, and is refused before
            # anything is spent; brief_info names the edit the knob claim read (at d5e6848 it was taken as a new knob)
            twin_34 = we.build(h.lm, [we.silence(h.n_layers - 1, [3, 4])], model_id=h.model_id, weights_id=h.weights_id)
            kn_twin = {"prompt": "Answer in plain words.",
                       "steering": [{"kind": "neuron_mul", "layer": h.n_layers - 1, "neurons": [3, 4], "factor": 0.0}]}
            check(kn_info["prior_verdicts"][0].get("twin_edit_id") == twin_34.id
                  and refuses(lambda: s_kn.claim_knob(kn_twin), lses.SessionRefused,
                              "the weight edit whose arithmetic this knob is")
                  and s_kn.brief_info()["fresh_readings_used"] == 1,
                  "3: a knobs_only session's knob claim names the weight edit whose arithmetic it is, and a second knob "
                  "of that arithmetic (with a prompt added) is refused read once before anything is spent")
            # the claim's look at its prompt is the session's own guard (level_knobs.kept_back_guard), by any door
            check(s_kn._kept_back.template_ids() == lkn.kept_back_guard(h.tokenizer, brief).template_ids()
                  and len(s_kn._kept_back) == len(lkn.kept_back_guard(h.tokenizer, brief)),
                  "3: a session's practice gates and a knob claim ask one guard of every kept-back frame")

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

            # The bridge never sets the process's umask: it is the whole process's, not the binding thread's, so a
            # file another thread made while the socket was bound took it (at d5e6848 it was set to 0o177 around the
            # bind). The 0700 folder alone keeps the socket from other users, and the socket is made 0600 once bound.
            um_calls: List[str] = []
            real_umask = os.umask

            def _rec_umask(mask: int) -> int:
                um_calls.append(oct(mask))
                return real_umask(mask)

            um_modes: Any = None
            os.umask = _rec_umask  # type: ignore[assignment]
            try:
                br_um = lmcp.SocketBridge(s_kn)
                um_modes = (oct(stat.S_IMODE(br_um.path.stat().st_mode)), oct(stat.S_IMODE(br_um.dir.stat().st_mode)))
                br_um.close()
            finally:
                os.umask = real_umask  # type: ignore[assignment]
            check(um_calls == [] and um_modes == ("0o600", "0o700"),
                  f"3: the bridge leaves the process's umask alone (umask calls {um_calls}): its socket is made 0600 "
                  f"in its 0700 folder ({um_modes})")
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
                  and oa_out["usage"].get("provider_total_tokens") == 2280 and oa_out["usage"]["reasoning_tokens"] == 20
                  and 2280 <= oa_out["usage"]["total_tokens"] <= 50000
                  and oa_out["usage"]["estimated_dollars"] > 0.0 and oa_out["book"]["verified"],
                  f"4: OpenAIChatDriver runs full tool loop, tracks tokens/dollars, records verdict ({oa_out['verdict']}; "
                  f"the endpoint's {oa_out['usage'].get('provider_total_tokens')} tokens, counted at the harness's floor "
                  f"as {oa_out['usage']['total_tokens']})")

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
            cap_words = " ".join(str(oa_cap_out.get(k) or "") for k in ("driver_failure", "notes"))
            check(oa_cap_out.get("driver_failure") is None and "none was made" not in cap_words
                  and "dollar cap" in cap_words and (oa_cap_out.get("door") or {}).get("calls") == 0,
                  f"4: a run its own dollar cap stopped before any door call is not a driver failure, and does not say "
                  f"none was made: it says the cap stopped it (at d5e6848 its driver_failure said \"No door call "
                  f"succeeded (none was made)\"; now {oa_cap_out.get('driver_failure')!r}, {oa_cap_out['notes'][:90]!r})")

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
                check(all(ldrv._resolve_openai_profile(None, u, None)[2] is True for u in local_urls),
                      "4: localhost, 127.0.0.0/8 and ::1 are on this machine, read from the parsed host")

                # An endpoint on this machine takes its key only from the variable a profile at its address names:
                # with no profile there it is sent none, and a plan cannot name one (at d5e6848 a plan could send
                # $GITHUB_TOKEN to http://127.0.0.1:9/v1)
                sci_loc = tmp / "scientist_cfg_local"
                sci_loc.mkdir()
                (sci_loc / "endpoints.json").write_text(json.dumps(
                    {"check-local": {"base_url": "http://127.0.0.1:9/v1", "key_env": "CHECK_LOCAL_KEY_VAR"}}), encoding="utf-8")
                with _env(GITHUB_TOKEN="gh_check_value", CHECK_LOCAL_KEY_VAR="local_check_value"):
                    loc_refused = [refuses(lambda: ldrv._resolve_openai_profile(None, "http://127.0.0.1:9/v1", "GITHUB_TOKEN"),
                                           ldrv.DriverRefused, "cannot pair"),
                                   refuses(lambda: ldrv._resolve_openai_profile(None, "http://localhost:59999/v1", "CHECK_LOCAL_KEY_VAR"),
                                           ldrv.DriverRefused, "no endpoint profile is at this address")]
                    loc_none = ldrv._resolve_openai_profile(None, "http://127.0.0.1:9/v1", None)[1]
                    with _env(THEODOLITE_SCIENTIST_DIR=str(sci_loc)):
                        loc_own = ldrv._resolve_openai_profile(None, "http://127.0.0.1:9/v1/", None)[1]
                        loc_named = ldrv._resolve_openai_profile("check-local", None, "CHECK_LOCAL_KEY_VAR")[1]
                        loc_refused.append(refuses(
                            lambda: ldrv._resolve_openai_profile(None, "http://127.0.0.1:9/v1", "GITHUB_TOKEN"),
                            ldrv.DriverRefused, "$CHECK_LOCAL_KEY_VAR"))
                check(all(loc_refused) and loc_none is None and loc_own == "local_check_value" and loc_named == "local_check_value",
                      f"4: an endpoint on this machine takes its key only from the variable a profile at its address "
                      f"names: $GITHUB_TOKEN is refused for http://127.0.0.1:9/v1 with or without a profile there, an "
                      f"address no profile names is sent no key, and a profile's own variable is taken (refused "
                      f"{loc_refused}, key with no profile {loc_none!r})")

                # An endpoint on this machine is never reached through a proxy, whatever the environment says: a
                # process started with the proxy variables set (and none bypassing 127.0.0.1) sends the request, key
                # and all, to the endpoint itself (at d5e6848 its opener, built at import, sent it to the proxy)
                px_rec = _DeadProxy()
                px_target: Dict[str, Any] = {}

                class _PxTarget(BaseHTTPRequestHandler):
                    def do_POST(self) -> None:  # noqa: N802
                        self.rfile.read(int(self.headers.get("Content-Length") or 0))
                        px_target["auth_sent"] = self.headers.get("Authorization") is not None
                        self.send_response(200)
                        self.send_header("Content-Length", "2")
                        self.end_headers()
                        self.wfile.write(b"{}")

                    def log_message(self, *_a: Any) -> None:
                        pass

                srv_px = HTTPServer(("127.0.0.1", 0), _PxTarget)
                threading.Thread(target=srv_px.serve_forever, daemon=True).start()
                px_env = {k: v for k, v in os.environ.items() if k.lower() not in ("no_proxy", "all_proxy")}
                px_env.update({"HTTP_PROXY": px_rec.url, "http_proxy": px_rec.url, "HTTPS_PROXY": px_rec.url,
                               "https_proxy": px_rec.url})
                px_code = ("import json, sys\nsys.path.insert(0, sys.argv[1])\nfrom backend import level_drivers as ldrv\n"
                           "try:\n    r = ldrv._default_openai_transport({'model': 'm'}, {'Content-Type': 'application/json', "
                           "'Authorization': 'Bearer check-dummy-key'}, sys.argv[2], 5.0)\nexcept Exception as exc:\n"
                           "    r = type(exc).__name__ + ': ' + str(exc)\nprint(json.dumps(r))\n")
                try:
                    px_run = subprocess.run(
                        [sys.executable, "-I", "-c", px_code, str(ROOT_DIR),
                         f"http://127.0.0.1:{srv_px.server_address[1]}/v1/chat/completions"],
                        capture_output=True, text=True, timeout=120, env=px_env)
                    px_said = (px_run.stdout.strip().splitlines() or ["(nothing)"])[-1]
                finally:
                    srv_px.shutdown()
                    srv_px.server_close()
                    px_rec.close()
                check(px_target.get("auth_sent") is True and not px_rec.hits,
                      f"4: an endpoint on this machine is reached directly, never through the environment's proxy, so "
                      f"no proxy is handed the request and its key in the clear (proxy saw {px_rec.hits}; endpoint "
                      f"reached: {bool(px_target)}; {px_said[:80]})")
                check(refuses(lambda: ldrv._resolve_openai_profile(None, "https://openrouter.ai/api/v1", "GITHUB_TOKEN"),
                              ldrv.DriverRefused, "cannot pair")
                      and refuses(lambda: ldrv._resolve_openai_profile("openrouter", "https://elsewhere.example/v1", None),
                                  ldrv.DriverRefused, "another address")
                      and ldrv._resolve_openai_profile(None, "https://openrouter.ai/api/v1/", None)[0]
                      == "https://openrouter.ai/api/v1/chat/completions",
                      "4: a plan cannot pair another variable with a remote address as its key, nor send a profile to "
                      "another address; a profile's own address and key variable are taken")

                # A profile in endpoints.json may name the field its endpoint bounds a reply with, one of the two the
                # OpenAI-compatible endpoints take; any other is refused with the entry (at d5e6848 it was dropped)
                sci_cfg2 = tmp / "scientist_cfg_limits"
                sci_cfg2.mkdir()
                (sci_cfg2 / "endpoints.json").write_text(json.dumps({
                    "newer": {"base_url": "https://api.newer.example/v1", "key_env": "NEWER_KEY",
                              "completion_limit_field": "max_completion_tokens"},
                    "oddone": {"base_url": "https://api.odd.example/v1", "key_env": "ODD_KEY",
                               "completion_limit_field": "max_length"},
                }), encoding="utf-8")
                with _env(THEODOLITE_SCIENTIST_DIR=str(sci_cfg2)):
                    from backend import scientist_chat as sc_chk

                    profs_lim = sc_chk.profiles()
                check(profs_lim.get("newer", {}).get("completion_limit_field") == "max_completion_tokens"
                      and "oddone" not in profs_lim,
                      "4: an endpoints.json profile keeps its completion_limit_field when it is max_tokens or "
                      "max_completion_tokens, and an entry naming any other field is refused")

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

                def _oa_reply(usage: Optional[Dict[str, Any]], calls: Optional[List[Any]] = None, text: str = "") -> Dict[str, Any]:
                    msg: Dict[str, Any] = {"role": "assistant", "content": text}
                    if calls:
                        msg["tool_calls"] = [{"id": f"t{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                                             for i, (n, a) in enumerate(calls)]
                    return {"model": "m", "choices": [{"message": msg}], **({"usage": usage} if usage is not None else {})}

                def _oa_session(name: str, **kw: Any) -> lses.LevelSession:
                    return lses.LevelSession(h, tmp / name, reach=kw.pop("reach", "behaviour"), backend="openai",
                                             book_path=tmp / f"b_{name}.jsonl", probe_on_start=False, **kw)

                # A reply's token counts are not taken at their word: each request is counted at no less than its
                # UTF-8 bytes / 4 and each reply at its own, so replies that say 0 cannot pass the token cap (at d5e6848
                # forty of them were sent, and counted 0 tokens and $0)
                zero_floor: List[int] = []

                def _tr_zero(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    zero_floor.append(len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) // 4)
                    return _oa_reply({"prompt_tokens": 0, "completion_tokens": 0}, [("book_view", {})])

                zero_out = ldrv.run_openai_driver(
                    _oa_session("s_oa_zero"), model="m", transport_fn=_tr_zero,
                    driver_budget=ldrv.DriverBudget(max_turns=40, max_tokens=30_000, max_dollars=1.0,
                                                    input_usd_per_mtok=1.0, output_usd_per_mtok=1.0))
                zu = zero_out["usage"]
                check(zero_out["stop_reason"] == "budget_tokens" and 0 < len(zero_floor) < 40
                      and sum(zero_floor) <= zu["total_tokens"] <= 30_000 and zu.get("provider_total_tokens") == 0
                      and zu["estimated_dollars"] >= zu["total_tokens"] / 1e6 - 1e-6,
                      f"4: OpenAIChatDriver counts replies that report 0 tokens at its own floor (each request's bytes / "
                      f"4) and stops at the token cap: {len(zero_floor)} requests, counted {zu['total_tokens']} tokens "
                      f"(the endpoint said {zu.get('provider_total_tokens')}), against max_tokens 30000")

                # The go or no-go prices a prompt at an upper bound, its UTF-8 bytes (a token covers one at least), not
                # its characters / 3, which CJK text or a printed float array beat two or three times: at d5e6848 this
                # request went out, and its one-token-a-byte prompt passed max_tokens
                big_text = "中文字符测试" * 5000 + " ".join(f"{x:.8f}" for x in range(2000))
                s_oa_big = _oa_session("s_oa_big")
                pg_big = ldrv.build_driver_prompt(s_oa_big, extra_instructions=big_text)
                big_cap = len((pg_big["system"] + pg_big["user"]).encode("utf-8")) // 2
                big_sent: List[int] = []

                def _tr_big(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    big_sent.append(len(json.dumps(payload, ensure_ascii=False).encode("utf-8")))
                    return _oa_reply({"prompt_tokens": big_sent[-1], "completion_tokens": 1}, None, "done")

                big_out = ldrv.run_openai_driver(
                    s_oa_big, model="m", transport_fn=_tr_big, extra_instructions=big_text,
                    driver_budget=ldrv.DriverBudget(max_turns=3, max_tokens=big_cap, max_dollars=1.0,
                                                    input_usd_per_mtok=1.0, output_usd_per_mtok=1.0))
                check(not big_sent and big_out["stop_reason"] == "budget_tokens" and big_out["usage"]["total_tokens"] == 0,
                      f"4: OpenAIChatDriver sends no request whose bytes pass what is left of max_tokens ({big_cap}), "
                      f"whatever its characters / 3 say (sent {len(big_sent)}, stop {big_out['stop_reason']})")

                # Each request carries a completion limit sized to what the dollars and tokens left allow (at d5e6848
                # none, and one reply ran to $30 against a $1 cap). An endpoint that keeps it keeps the cap; a reply
                # past it stops the run; a driver config can name the field.
                lim_seen: List[Dict[str, Any]] = []
                lim_margin = getattr(ldrv, "TEMPLATE_MARGIN_TOKENS", 1024)
                lim_most = getattr(ldrv, "MAX_COMPLETION_PER_REPLY", 32768)

                def _tr_keeps(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    rest = {k: v for k, v in payload.items() if k != "max_tokens"}
                    nb = len(json.dumps(rest, ensure_ascii=False).encode("utf-8"))
                    # what $0.10 leaves a first reply at $10 a million, once the prompt's upper bound is paid for
                    lim_seen.append({"sent": payload.get("max_tokens"), "allowed": int((0.1 - (nb + lim_margin) * 1.0 / 1e6) * 1e6 / 10.0)})
                    return _oa_reply({"prompt_tokens": nb, "completion_tokens": payload.get("max_tokens")}, [("status", {})])

                keeps_out = ldrv.run_openai_driver(
                    _oa_session("s_oa_keeps"), model="m", transport_fn=_tr_keeps,
                    driver_budget=ldrv.DriverBudget(max_turns=10, max_tokens=10 ** 7, max_dollars=0.1,
                                                    input_usd_per_mtok=1.0, output_usd_per_mtok=10.0))
                huge_out = ldrv.run_openai_driver(
                    _oa_session("s_oa_huge"), model="m",
                    transport_fn=lambda p, hd, u, t: _oa_reply({"prompt_tokens": 500, "completion_tokens": 3_000_000}, [("status", {})]),
                    driver_budget=ldrv.DriverBudget(max_turns=10, max_tokens=200_000, max_dollars=1.0,
                                                    input_usd_per_mtok=1.0, output_usd_per_mtok=10.0))
                named_seen: List[Dict[str, Any]] = []

                def _tr_named(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    named_seen.append({k: payload.get(k) for k in ("max_tokens", "max_completion_tokens")})
                    return _oa_reply({"prompt_tokens": 10, "completion_tokens": 1}, [("status", {})] if len(named_seen) == 1 else None, "x")

                ldrv.run_driver(_oa_session("s_oa_named"), "openai", driver_model="m",
                                driver_budget=ldrv.DriverBudget(max_turns=3, max_dollars=1.0, input_usd_per_mtok=1.0, output_usd_per_mtok=1.0),
                                config={"transport_fn": _tr_named, "completion_limit_field": "max_completion_tokens"})
                lim_ok = (bool(lim_seen) and lim_seen[0]["sent"] == min(lim_seen[0]["allowed"], lim_most)
                          and all(isinstance(s["sent"], int) and 1 <= s["sent"] <= lim_most for s in lim_seen))
                check(lim_ok and keeps_out["stop_reason"] == "budget_dollars" and keeps_out["usage"]["estimated_dollars"] <= 0.1
                      and huge_out["usage"]["turns"] == 1 and "does not keep the limit" in huge_out["notes"]
                      and bool(named_seen) and all(isinstance(s["max_completion_tokens"], int) and s["max_tokens"] is None for s in named_seen),
                      f"4: OpenAIChatDriver sends each request a completion limit sized to what the dollars and tokens "
                      f"left allow ({[s['sent'] for s in lim_seen]}): an endpoint that keeps it keeps a $0.10 cap "
                      f"(${keeps_out['usage']['estimated_dollars']}), one reply past it stops the run, and a driver "
                      f"config can send it as max_completion_tokens ({named_seen[:1]})")

                # An endpoint that refuses the field fails the run in its own words (the default transport reads
                # the answer's body; at d5e6848 no field was sent)
                nm_fields: List[List[str]] = []

                class _NoMaxTokens(BaseHTTPRequestHandler):
                    def do_POST(self) -> None:  # noqa: N802
                        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                        nm_fields.append(sorted(k for k in ("max_tokens", "max_completion_tokens") if k in body))
                        if "max_tokens" in body:
                            code, out_b = 400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported "
                                                                     "with this model. Use 'max_completion_tokens' instead."}}
                        else:
                            code, out_b = 200, _oa_reply({"prompt_tokens": 10, "completion_tokens": 1}, None, "ok")
                        raw = json.dumps(out_b).encode("utf-8")
                        self.send_response(code)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)

                    def log_message(self, *_a: Any) -> None:
                        pass

                srv_nm = HTTPServer(("127.0.0.1", 0), _NoMaxTokens)
                threading.Thread(target=srv_nm.serve_forever, daemon=True).start()
                nm_url = f"http://127.0.0.1:{srv_nm.server_address[1]}/v1"
                from backend import scientist_chat as sc_mod

                try:
                    nm_out = ldrv.run_openai_driver(
                        _oa_session("s_oa_nomax"), model="m", base_url=nm_url,
                        driver_budget=ldrv.DriverBudget(max_turns=3, max_dollars=1.0, max_wall_seconds=30.0))
                    # an endpoint profile can name the field too (a built-in profile here: endpoints.json keeps only
                    # the keys scientist_chat lets through)
                    sc_mod.PROFILES["check-limit-field"] = {"label": "check", "base_url": nm_url,
                                                            "completion_limit_field": "max_completion_tokens"}
                    nm_fields_before = len(nm_fields)
                    nm_prof = ldrv.run_openai_driver(
                        _oa_session("s_oa_nomax_prof"), model="m", profile="check-limit-field",
                        driver_budget=ldrv.DriverBudget(max_turns=3, max_dollars=1.0, max_wall_seconds=30.0))
                    nm_prof_fields = nm_fields[nm_fields_before:]
                finally:
                    sc_mod.PROFILES.pop("check-limit-field", None)
                    srv_nm.shutdown()
                    srv_nm.server_close()
                check(nm_out["stop_reason"] == "driver_failure" and "HTTP 400" in nm_out["notes"]
                      and "Unsupported parameter" in str(nm_out.get("driver_failure"))
                      and nm_prof_fields == [["max_completion_tokens"]] and "HTTP 400" not in nm_prof["notes"],
                      f"4: an endpoint that refuses the completion limit's field fails the run loudly, in its own words "
                      f"({str(nm_out.get('driver_failure'))[:100]}); one whose profile names max_completion_tokens is "
                      f"sent that ({nm_prof_fields})")

                # The wall clock is looked at between the tool calls of a reply: four captures of 1 s each (the card's
                # lock stands in for a slow card) against a 1 s cap make one (at d5e6848 all four ran)
                @contextlib.contextmanager
                def _card_1s() -> Iterator[None]:
                    time.sleep(1.0)
                    yield

                cap_args = {"items_a": [{"question": "Say hi.", "answer": " Hi there."}],
                            "items_b": [{"question": "Say no.", "answer": " No."}], "layer": 0}
                wall_n = {"i": 0}

                def _tr_wall(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    wall_n["i"] += 1
                    return _oa_reply({"prompt_tokens": 10, "completion_tokens": 1},
                                     [("capture", cap_args)] * 4 if wall_n["i"] == 1 else None, "end")

                t_w0 = time.monotonic()
                wall_out = ldrv.run_openai_driver(
                    _oa_session("s_oa_wall", reach="activations", gpu_lock=_card_1s), model="m", transport_fn=_tr_wall,
                    driver_budget=ldrv.DriverBudget(max_turns=6, max_wall_seconds=1.0, max_dollars=1.0,
                                                    input_usd_per_mtok=1.0, output_usd_per_mtok=1.0))
                dt_w = time.monotonic() - t_w0
                check(wall_out["stop_reason"] == "budget_wall" and wall_out["door"]["calls"] == 1 and dt_w < 2.9
                      and "3 of the 4" in wall_out["notes"],
                      f"4: OpenAIChatDriver keeps the wall cap between the tool calls of one reply: of four 1 s captures "
                      f"against a 1 s cap, {wall_out['door']['calls']} ran ({dt_w:.1f} s)")

                # A dollar cap of 0 is a cap: priced, nothing is sent (at d5e6848 0.0 read as no cap); at prices of 0
                # nothing is paid, and requests go out
                z_sent = {"n": 0}

                def _tr_z(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                    z_sent["n"] += 1
                    return _oa_reply({"prompt_tokens": 10, "completion_tokens": 1}, [("status", {})] if z_sent["n"] < 3 else None, "x")

                z_priced = ldrv.run_openai_driver(
                    _oa_session("s_oa_z0"), model="m", transport_fn=_tr_z,
                    driver_budget=ldrv.DriverBudget(max_turns=5, max_dollars=0.0, input_usd_per_mtok=1.0, output_usd_per_mtok=1.0))
                n_z_priced = z_sent["n"]
                z_sent["n"] = 0
                z_free = ldrv.run_openai_driver(_oa_session("s_oa_z0free"), model="m", transport_fn=_tr_z,
                                                driver_budget=ldrv.DriverBudget(max_turns=5, max_dollars=0.0))
                check(n_z_priced == 0 and z_priced["stop_reason"] == "budget_dollars" and z_priced.get("driver_failure") is None
                      and z_sent["n"] == 3 and z_free["usage"]["estimated_dollars"] == 0.0,
                      f"4: a priced endpoint given max_dollars 0.0 is sent nothing ({n_z_priced} sent, "
                      f"{z_priced['stop_reason']}); an unpriced one is paid nothing and runs ({z_sent['n']} sent)")

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

                # The scripted weight baseline puts forward only an edit the claim gate can take: at the activations
                # reach its one candidate is low-rank, which the gate refuses, so it keeps the base (at d5e6848 it
                # claimed it and ended in an error)
                lr_only = {"terms": [{"layer": 1, "lowrank": {"B": "b.npy", "A": "a.npy"}}]}
                mixed = {"terms": [{"layer": 1, "lowrank": {"B": "b.npy", "A": "a.npy"}}, {"layer": 1, "silence": [3]}]}
                check(ldrv._claimable_recipe(lr_only) is False and ldrv._claimable_recipe(mixed) is True
                      and ldrv._claimable_recipe({"terms": [{"layer": 2, "silence": [0, 1]}]}) is True
                      and ldrv._claimable_recipe({}) is False,
                      "4: the scripted weight baseline counts an edit of low-rank terms alone as one the gate cannot take")

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

            # A claim the gate refused brings the gate's own words into the outcome, whichever driver made it (at
            # d5e6848 no outcome carried them): a replay's, and an OpenAI-compatible driver's
            bad_claim = {"edit": {"terms": [{"layer": 999, "silence": [0]}]}}
            cr_design = {"n_target": 4, "n_neutral": 2, "n_near": 2, "n_panel": 2, "rows": {"small": 12, "large": 16},
                         "controls_override": 2, "match_tolerance": 20.0}

            def _cr_session(name: str, backend: str) -> lses.LevelSession:
                bk = tmp / f"b_{name}.jsonl"
                shutil.copy2(cal_book, bk)
                return lses.LevelSession(h, tmp / name, reach="behaviour", backend=backend, book_path=bk, brief=brief,
                                         bars=bars, probe_on_start=False, budget=lses.SessionBudget(min_controls=1),
                                         claim_design=cr_design)

            cr_rep = ldrv.run_replay_driver(_cr_session("s_cr_rep", "replay"),
                                            [{"tool": "status", "args": {}}, {"tool": "claim", "args": bad_claim}])
            cr_n = {"i": 0}

            def _tr_cr(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                cr_n["i"] += 1
                msg: Dict[str, Any] = {"role": "assistant", "content": "" if cr_n["i"] == 1 else "Keeping the base."}
                if cr_n["i"] == 1:
                    msg["tool_calls"] = [{"id": "s", "type": "function", "function": {"name": "status", "arguments": "{}"}},
                                         {"id": "c", "type": "function", "function": {"name": "claim", "arguments": json.dumps(bad_claim)}}]
                return {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 1}, "choices": [{"message": msg}]}

            cr_oa = ldrv.run_openai_driver(_cr_session("s_cr_oa", "openai"), model="m", transport_fn=_tr_cr,
                                           driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0,
                                                                           input_usd_per_mtok=1.0, output_usd_per_mtok=1.0))
            cr_words = [str(o.get("claim_refused") or "") for o in (cr_rep, cr_oa)]
            check(all("Layer 999 does not exist" in w for w in cr_words) and not cr_rep["submitted"] and not cr_oa["submitted"],
                  f"4: a claim the gate refused brings the gate's own words into the outcome, for a replay and an "
                  f"OpenAI-compatible driver alike ({cr_words[0][:90]!r})")

            # Every driver family's outcome carries driver_failure, stop_reason and door {calls, ok, errors}, true of
            # what ran (at d5e6848 the scripted and replay outcomes had no door): a scripted baseline of each mode
            # stopped by its turn cap after one call, the replays and OpenAI runs above, and the CLI runs below
            scr_w = ldrv.run_scripted_weight_driver(
                lses.LevelSession(h, tmp / "s_scr_w1", reach="behaviour", backend="script", book_path=tmp / "b_scr_w1.jsonl",
                                  probe_on_start=False), driver_budget=ldrv.DriverBudget(max_turns=1, max_dollars=0.0))
            scr_k = ldrv.run_scripted_knob_driver(
                lses.LevelSession(h, tmp / "s_scr_k1", reach="behaviour", mode="knobs_only", backend="script",
                                  book_path=tmp / "b_scr_k1.jsonl", probe_on_start=False),
                driver_budget=ldrv.DriverBudget(max_turns=1, max_dollars=0.0))
            family_outs: Dict[str, Dict[str, Any]] = {"scripted weights": scr_w, "scripted knobs": scr_k, "replay": rep_out,
                                                      "replay with a refused claim": cr_rep, "openai": oa_out,
                                                      "openai at its cap": oa_cap_out}

            def _carries(o: Dict[str, Any]) -> bool:
                d = o.get("door")
                return ("driver_failure" in o and (o["driver_failure"] is None or isinstance(o["driver_failure"], str))
                        and isinstance(o.get("stop_reason"), str) and isinstance(d, dict)
                        and all(isinstance(d.get(k), int) for k in ("calls", "ok", "errors"))
                        and d["ok"] + d["errors"] == d["calls"])

            check(all(_carries(o) for o in family_outs.values())
                  and (scr_w["door"], scr_k["door"]) == ({"calls": 1, "ok": 1, "errors": 0},) * 2
                  and scr_w["stop_reason"] == scr_k["stop_reason"] == "budget_turns" and scr_w["driver_failure"] is None
                  and rep_out["door"]["calls"] == 0 and rep_out["driver_failure"] is None
                  and cr_rep["door"] == {"calls": 2, "ok": 1, "errors": 1},
                  f"4: every driver family's outcome carries driver_failure, stop_reason and door {{calls, ok, errors}}, "
                  f"true of what ran ({', '.join(k for k, o in family_outs.items() if _carries(o))})")

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
            cli_outs = {"door call": cl_out, "denied": deny_out, "other tools": extra_out, "token cap": spend_out,
                        "wall cap": hang_out}
            check(all(_carries(o) for o in cli_outs.values()) and hang_out["driver_failure"] is None
                  and hang_out["door"]["calls"] == 0 and spend_out["driver_failure"] is None,
                  f"4: the Claude CLI driver's outcomes carry driver_failure, stop_reason and door {{calls, ok, errors}} "
                  f"too, and a CLI its own cap stopped before any door call is no driver failure "
                  f"({', '.join(k for k, o in cli_outs.items() if _carries(o))}; wall cap: {hang_out['driver_failure']!r})")

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

            # A link (or a pipe) a cell leaves in /work is removed, unread, as the cell ends: the host never follows
            # it to report another file's size and sha256 as a file written (at d5e6848 and on main since dc241d0 it
            # did), nor does a reading's term file follow it later. The link's target is a scratch file of the check's.
            if has_bwrap:
                import hashlib

                host_only = tmp / "host_only_dir"
                host_only.mkdir()
                host_f = host_only / "host_only.txt"
                host_f.write_text("HOST_ONLY_CHECK_CONTENT\n", encoding="utf-8")
                host_sha = hashlib.sha256(host_f.read_bytes()).hexdigest()
                s_ln = lses.LevelSession(None, tmp / "s_links", reach="sandbox_no_weights", backend="claude",
                                         book_path=tmp / "b_links.jsonl", probe_on_start=False)
                # the cell also hides a link in a folder it takes the read bit off (0300: it can still be gone through)
                ln_res = s_ln.cell(f"import os\nos.symlink({str(host_f)!r}, '/work/f_link.npy')\n"
                                   f"os.symlink({str(host_only)!r}, '/work/d_link')\nos.mkfifo('/work/a_pipe.npy')\n"
                                   f"os.mkdir('/work/hid')\nos.symlink({str(host_f)!r}, '/work/hid/x.npy')\n"
                                   "os.chmod('/work/hid', 0o300)\n"
                                   "open('/work/mine.txt', 'w').write('mine')\nprint('left them')\n")
                ln_written = ln_res.get("files_written") or []
                ln_left = sorted(str(p.relative_to(s_ln.work_dir)) for p in s_ln.work_dir.rglob("*"))
                ln_read = []
                for ln_name in ("f_link.npy", "hid/x.npy"):
                    try:  # a reading's term file named after a link: had it stayed, the refusal would print its hash
                        we.from_recipe({"terms": [{"layer": 0, "remove_direction": [], "direction": ln_name,
                                                   "sha256": "0" * 64}]}, s_ln.work_dir)
                        ln_read.append("read")
                    except Exception as exc:
                        ln_read.append(str(exc))
                s_ln.close()
                check(ln_res.get("ok") is True and [f.get("rel_path") for f in ln_written] == ["mine.txt"]
                      and not any(f.get("sha256") == host_sha for f in ln_written) and ln_left == ["hid", "mine.txt"]
                      and sorted((r.get("rel_path"), r.get("kind")) for r in ln_res.get("removed") or [])
                      == [("a_pipe.npy", "pipe"), ("d_link", "link"), ("f_link.npy", "link"), ("hid/x.npy", "link")]
                      and not any(host_sha[:12] in w or w == "read" for w in ln_read)
                      and host_f.read_text(encoding="utf-8") == "HOST_ONLY_CHECK_CONTENT\n",
                      f"4: a link or pipe a cell leaves in /work, even in a folder it made unlistable, is removed unread "
                      f"as it ends and named as removed: no other file's size or sha256 is reported as one it wrote, "
                      f"and no reading's term file follows it later (files written "
                      f"{[f.get('rel_path') for f in ln_written]}, left {ln_left})")
            else:
                skip("4: bwrap is not installed: links a cell leaves in /work were not tried")

            # a claim is waited for, however long it reads: its verdict is in the outcome and its edit is
            # proposed_edit (at f6d4e86 the outcome said no_submission and the claim landed in the book after it; at
            # d5e6848 the same, once the reading outlived the bridge's drain bound, made 0.5 s here as `_drain_bound`
            # was: the bridge now has no bound and reads no such attribute). The card's lock stands in for a slow
            # card: it marks the reading's start, then takes 2 s.
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
            had_bound = "_drain_bound" in vars(lmcp.SocketBridge)
            old_bound = vars(lmcp.SocketBridge).get("_drain_bound")
            lmcp.SocketBridge._drain_bound = lambda self: 0.5  # type: ignore[attr-defined]
            try:
                flr_out = ldrv.run_claude_cli_driver(
                    s_flr, claude_bin=str(inflight_cli), allow_mock_cli=True, kill_switch=ldrv.KillSwitch(stop_file=stop_r),
                    driver_budget=ldrv.DriverBudget(max_turns=4, max_dollars=1.0, max_wall_seconds=120.0),
                )
            finally:
                if had_bound:
                    lmcp.SocketBridge._drain_bound = old_bound  # type: ignore[attr-defined]
                else:
                    del lmcp.SocketBridge._drain_bound  # type: ignore[attr-defined]
            door_r = flr_out.get("door") or {}
            run_r = door_r.get("running_at_stop") or {}
            ev_r = _events(s_flr)
            check(flr_out["stop_reason"] == "killed" and flr_out["submitted"] is True and flr_out["verdict"] != "no_submission"
                  and run_r.get("tool") == "claim" and run_r.get("ok") is True and flr_out.get("proposed_edit") == fl_edit
                  and ev_r[-1] == "session_close" and flr_out.get("driver_failure") is None,
                  f"4: a claim still reading when STOP kills the CLI is waited for, past any drain bound: its verdict is "
                  f"in the outcome ({flr_out['verdict']}), its edit is proposed_edit, and nothing lands after "
                  f"session_close (waited {door_r.get('waited_seconds')} s)")
            check(isinstance(door_r.get("waited_seconds"), (int, float)) and door_r["waited_seconds"] >= 0.5
                  and "claim call was running at the stop" in str(door_r.get("wait_note"))
                  and "waited" in flr_out["notes"] and "outcome was read" in flr_out["notes"],
                  f"4: the outcome says how long after the CLI ended it was read, and why: {str(door_r.get('wait_note'))[:110]!r}")

            # The real claude CLI, against a local stand-in of the Messages API: a fake key, a home folder of its
            # own, and every proxy at a stand-in that forwards nothing. The stand-in asks for one door call (status),
            # then for a built-in Read of a file outside the work folder, then stops; a second run is priced past
            # --max-budget-usd. Both run in a child process (`real_cli_child`) that, when bwrap is here, has a network
            # namespace of its own whose one interface is loopback: the CLI, the door and the stand-ins share it, and
            # nothing can leave the machine, by a proxy or directly. Without bwrap only the proxy variables are
            # watched, and the check says so.
            real_cli_child = r'''import json, os, re, socket, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root, work, claude_real, out_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, root)
from backend import level_drivers as ldrv
from backend import level_session as lses

seen = {"interfaces": sorted(name for _i, name in socket.if_nameindex()), "errors": []}


class Dead:
    """A stand-in proxy: it records each connection's first line and closes it, forwarding nothing."""

    def __init__(self):
        self.hits = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.sock.settimeout(0.2)
        self.url = "http://127.0.0.1:%d" % self.sock.getsockname()[1]
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
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


def sse(msg):
    start = dict(msg, content=[], stop_reason=None, usage=dict(msg["usage"], output_tokens=1))
    evs = [{"type": "message_start", "message": start}]
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
    return "".join("event: %s\ndata: %s\n\n" % (e["type"], json.dumps(e)) for e in evs).encode("utf-8")


class StandIn:
    """POST /v1/messages answered from `script(body) -> (content blocks, stop reason)`; every request kept (its path,
    whether it carried the fake key, its JSON body); a path not served is noted."""

    def __init__(self, script, key, in_tokens=60):
        self.requests = []
        self.unserved = []
        stand_in = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def reply(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def any(self):
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
                    msg = {"id": "msg_stand_in_%d" % len(stand_in.requests), "type": "message", "role": "assistant",
                           "model": str(body.get("model") or "stand-in"), "content": blocks, "stop_reason": stop,
                           "stop_sequence": None, "usage": {"input_tokens": in_tokens, "output_tokens": 12,
                                                            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
                    if body.get("stream"):
                        self.reply(200, sse(msg), "text/event-stream")
                    else:
                        self.reply(200, json.dumps(msg).encode("utf-8"), "application/json")
                    return
                stand_in.unserved.append("%s %s" % (self.command, self.path))
                self.reply(404, b'{"type":"error","error":{"type":"not_found_error","message":"not served"}}',
                           "application/json")

            do_GET = do_POST = do_HEAD = any

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def result_of(self, tool_use_id):
        for r in self.requests:
            for m in (r["body"] or {}).get("messages") or []:
                for blk in m.get("content") if isinstance(m.get("content"), list) else []:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result" and blk.get("tool_use_id") == tool_use_id:
                        return blk
        return {}


cl_home = work / "claude_home"  # its home folder: nothing of this machine's own Claude Code
(cl_home / ".claude").mkdir(parents=True)
(cl_home / ".claude" / "CLAUDE.md").write_text("CHECK_MARK_HOME_CLAUDE_MD\n", encoding="utf-8")
outside_f = work / "outside_the_work_folder.txt"
outside_f.write_text("CHECK_MARK_OUTSIDE_FILE\n", encoding="utf-8")
marks = ("CHECK_MARK_HOME_CLAUDE_MD", "CHECK_MARK_PROJECT_CLAUDE_MD", "CHECK_MARK_AUTO_MEMORY", "CHECK_MARK_OUTSIDE_FILE")
fake_key = "sk-ant-api03-" + "0" * 80 + "-checkAA"  # a key no service issued
dead = Dead()
for k in ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
          "XDG_DATA_HOME", "XDG_STATE_HOME", "ALL_PROXY", "all_proxy", "ANTHROPIC_BASE_URL"):
    os.environ.pop(k, None)
os.environ.update({"HOME": str(cl_home), "ANTHROPIC_API_KEY": fake_key, "HTTPS_PROXY": dead.url, "HTTP_PROXY": dead.url,
                   "https_proxy": dead.url, "http_proxy": dead.url, "NO_PROXY": "127.0.0.1,localhost",
                   "no_proxy": "127.0.0.1,localhost"})
seen["version"] = subprocess.run([claude_real, "--version"], capture_output=True, text=True, timeout=60, env={
    "PATH": os.environ.get("PATH", ""), "HOME": str(cl_home), "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}).stdout.strip() or "version unknown"


def door_script(body):
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


def spend_script(body):
    n_msgs = len(body.get("messages") or [])
    return [{"type": "tool_use", "id": "toolu_spend_%d" % n_msgs, "name": "mcp__levelling__status", "input": {}}], "tool_use"


def run(name, script, budget, in_tokens=60):
    sess = lses.LevelSession(None, work / "claude_proj" / name, reach="behaviour", backend="claude",
                             book_path=work / ("b_%s.jsonl" % name), probe_on_start=False)
    (work / "claude_proj" / "CLAUDE.md").write_text("CHECK_MARK_PROJECT_CLAUDE_MD\n", encoding="utf-8")
    mem = cl_home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(sess.work_dir)) / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "MEMORY.md").write_text("CHECK_MARK_AUTO_MEMORY\n", encoding="utf-8")  # where its auto-memory looks
    stand_in = StandIn(script, fake_key, in_tokens)
    os.environ["ANTHROPIC_BASE_URL"] = stand_in.url
    try:
        return sess, stand_in, ldrv.run_claude_cli_driver(sess, claude_bin=claude_real, driver_budget=budget)
    finally:
        stand_in.close()


try:
    s1, si1, out1 = run("s_real", door_script, ldrv.DriverBudget(max_turns=6, max_tokens=10_000_000, max_dollars=0.5,
                                                                 max_wall_seconds=120.0))
    blob = json.dumps([r["body"] for r in si1.requests])
    st = si1.result_of("toolu_door_status")
    rd = si1.result_of("toolu_builtin_read")
    seen["door_tools"] = sorted("mcp__levelling__" + t for t in s1.allowed_tools())
    seen["tool_sets"] = [sorted(str(t.get("name")) for t in r["body"]["tools"] if isinstance(t, dict))
                         for r in si1.requests if isinstance(r["body"], dict) and r["body"].get("tools")]
    seen["status_result"] = ({"is_error": bool(st.get("is_error")), "card": "harness_card" in json.dumps(st.get("content"))}
                             if st else None)
    seen["read_result"] = {"is_error": rd.get("is_error") is True, "said": str(rd.get("content"))[:120]} if rd else None
    seen["marks_seen"] = [mk for mk in marks if mk in blob]
    seen["requests"] = len(si1.requests)
    seen["all_fake_key"] = bool(si1.requests) and all(r["fake_key"] for r in si1.requests)
    seen["unserved"] = si1.unserved
    seen["real"] = {"stop_reason": out1["stop_reason"], "driver_failure": out1.get("driver_failure"),
                    "door_ok": (out1.get("door") or {}).get("ok"), "tool_errors": out1["usage"]["tool_errors"]}
except Exception as exc:
    seen["errors"].append("first run: %s: %s" % (type(exc).__name__, exc))
try:
    s2, si2, out2 = run("s_real2", spend_script, ldrv.DriverBudget(max_turns=6, max_tokens=10_000_000, max_dollars=0.05,
                                                                   max_wall_seconds=120.0), in_tokens=200_000)
    seen["spend"] = {"stop_reason": out2["stop_reason"], "dollars": out2["usage"]["estimated_dollars"],
                     "driver_failure": out2.get("driver_failure"),
                     "requests": sum(1 for r in si2.requests if isinstance(r["body"], dict) and r["body"].get("tools"))}
except Exception as exc:
    seen["errors"].append("second run: %s: %s" % (type(exc).__name__, exc))
seen["proxy_hits"] = list(dead.hits)
out_path.write_text(json.dumps(seen), encoding="utf-8")
'''
            claude_real = shutil.which("claude")
            real_what = ("the tools it sends the model", "its door call", "its refusal of a built-in Read",
                         "what reached the model and what left the machine", "--max-budget-usd")
            if claude_real is None:
                for what in real_what:  # each check below that could not run
                    skip(f"4: the claude CLI is not installed: {what} was not tried on a real run")
            else:
                real_dir = tmp / "real_cli"
                real_dir.mkdir()
                (real_dir / "real_cli_child.py").write_text(real_cli_child, encoding="utf-8")
                seen_p = real_dir / "seen.json"
                real_cmd = [sys.executable, "-I", str(real_dir / "real_cli_child.py"), str(ROOT_DIR), str(real_dir),
                            claude_real, str(seen_p)]
                if has_bwrap:  # a network namespace of its own, loopback only, for the child and all it starts
                    real_cmd = [str(shutil.which("bwrap")), "--dev-bind", "/", "/", "--unshare-net", "--die-with-parent",
                                "--", *real_cmd]
                real_run = subprocess.run(real_cmd, capture_output=True, text=True, timeout=900)
                try:
                    seen_r = json.loads(seen_p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    seen_r = {"errors": [f"the child wrote nothing (exit {real_run.returncode}): {real_run.stderr[-300:]}"]}
                real_errs = "; ".join(seen_r.get("errors") or []) or "none"
                tool_sets = seen_r.get("tool_sets") or []
                st_r = seen_r.get("status_result") or {}
                rd_r = seen_r.get("read_result") or {}
                real_r = seen_r.get("real") or {}
                spend_r = seen_r.get("spend") or {}
                hits_r = seen_r.get("proxy_hits")
                check(bool(tool_sets) and all(ts == seen_r.get("door_tools") for ts in tool_sets),
                      f"4: the real claude CLI ({seen_r.get('version')}) sends the model exactly the door's tools in all "
                      f"{len(tool_sets)} requests: no Read, Write, Edit, Bash, WebFetch, WebSearch or memory tool "
                      f"({tool_sets[0] if tool_sets else 'no request with tools'}; errors: {real_errs})")
                check((real_r.get("door_ok") or 0) >= 1 and st_r.get("card") is True and st_r.get("is_error") is False
                      and real_r.get("stop_reason") == "kept_base_abstention" and real_r.get("driver_failure") is None,
                      f"4: the real CLI's door call (status) went through the bridge and came back whole, so its ending is "
                      f"an honest one (door calls answered {real_r.get('door_ok')}, stop {real_r.get('stop_reason')})")
                check(rd_r.get("is_error") is True and "CHECK_MARK_OUTSIDE_FILE" not in (seen_r.get("marks_seen") or [])
                      and (real_r.get("tool_errors") or 0) >= 1,
                      f"4: the real CLI refused the built-in Read of a file outside the work folder ({rd_r.get('said')!r:.80})")
                left_ok = (not seen_r.get("marks_seen") and hits_r == [] and (seen_r.get("requests") or 0) > 0
                           and seen_r.get("all_fake_key") is True)
                if has_bwrap:
                    check(left_ok and seen_r.get("interfaces") == ["lo"],
                          f"4: no CLAUDE.md, auto-memory or outside file reached the model, every request carried the "
                          f"fake key to the stand-in, and nothing could leave the machine: the run's only network was "
                          f"loopback ({seen_r.get('interfaces')}), and no request went to a proxy ({hits_r}; paths the "
                          f"stand-in did not serve: {seen_r.get('unserved') or 'none'})")
                else:
                    check(left_ok,
                          f"4: no CLAUDE.md, auto-memory or outside file reached the model, every request carried the "
                          f"fake key to the stand-in, and no request went to a proxy ({hits_r}): only the proxy "
                          f"variables were watched, as the CLI shared this machine's network")
                    skip("4: bwrap is not installed: the real CLI was not run in a network namespace of its own, so a "
                         "direct connection out of the machine was not watched")
                check(spend_r.get("stop_reason") == "budget_dollars" and spend_r.get("requests") == 1
                      and spend_r.get("driver_failure") is None and hits_r == [],
                      f"4: the real CLI stops itself at --max-budget-usd: one reply priced past the $0.05 left, no second "
                      f"request, and its own cap is no driver failure (requests {spend_r.get('requests')}, "
                      f"${spend_r.get('dollars')}, {spend_r.get('driver_failure')!r})")
            dead.close()

            # 4b. Gemini CLI driver: allowlist, refusals, mock CLI, extra-tool guard, and real `gemini` CLI under bwrap
            os.environ["HF_TOKEN"] = "hf_secret_check"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "aws_secret_check"
            os.environ["OPENROUTER_API_KEY"] = "or_secret_check"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/fake_gcp_sa.json"
            try:
                genv = ldrv.gemini_child_env(tmp / "g_home", tmp / "g_sys.md")
                check(all(k not in genv for k in ("HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "OPENROUTER_API_KEY",
                                                   "GOOGLE_APPLICATION_CREDENTIALS"))
                      and genv["GEMINI_CLI_HOME"] == str(tmp / "g_home")
                      and genv["GEMINI_SYSTEM_MD"] == str(tmp / "g_sys.md")
                      and genv["GEMINI_CLI_TRUST_WORKSPACE"] == "true",
                      "4b: gemini_child_env strips secrets and sets isolated GEMINI_CLI_HOME / GEMINI_SYSTEM_MD")
            finally:
                for k in ("HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "OPENROUTER_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"):
                    os.environ.pop(k, None)

            s_gm_ref = lses.LevelSession(h, tmp / "s_gm_ref", reach="behaviour", backend="gemini",
                                         book_path=tmp / "b_gm_ref.jsonl", probe_on_start=False)
            check(refuses(lambda: ldrv.run_gemini_cli_driver(s_gm_ref, driver_budget=ldrv.DriverBudget(max_dollars=None)),
                          ldrv.DriverRefused),
                  "4b: run_gemini_cli_driver refuses missing max_dollars ceiling")
            os.environ[lses.REACH_ENV] = "home"
            try:
                check(refuses(lambda: ldrv.run_gemini_cli_driver(s_gm_ref, driver_budget=ldrv.DriverBudget(max_dollars=1.0),
                                                                 allow_mock_cli=True), ldrv.DriverRefused),
                      "4b: run_gemini_cli_driver refuses home reach")
            finally:
                os.environ.pop(lses.REACH_ENV, None)
            s_gm_ref.close()

            # Mock Gemini CLI exercising stream-json + live socket bridge + extra-tool guard
            mock_gem = tmp / "mock_gemini.py"
            mock_gem.write_text(f"""#!{sys.executable} -I
import json, os, subprocess, sys
from pathlib import Path
mode = os.environ.get("MOCK_GEM_MODE", "ok")
ghome = Path(os.environ["GEMINI_CLI_HOME"])
cfg = json.loads((ghome / ".gemini" / "settings.json").read_text())
srv = cfg["mcpServers"]["levelling"]
mcp = subprocess.Popen([srv["command"], *srv["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(i, m, p=None):
    mcp.stdin.write(json.dumps({{"jsonrpc": "2.0", "id": i, "method": m, "params": p or {{}}}}) + "\\n")
    mcp.stdin.flush()
    return json.loads(mcp.stdout.readline())
rpc(1, "initialize", {{"protocolVersion": "2025-06-18"}})
print(json.dumps({{"type": "init", "session_id": "g1", "model": "gemini-mock"}}), flush=True)
if mode == "extra":
    print(json.dumps({{"type": "tool_use", "tool_name": "read_file", "tool_id": "t_bad", "parameters": {{}}}}), flush=True)
    import time; time.sleep(5)
else:
    print(json.dumps({{"type": "tool_use", "tool_name": "mcp_levelling_status", "tool_id": "t1", "parameters": {{}}}}), flush=True)
    r = rpc(2, "tools/call", {{"name": "status", "arguments": {{}}}})
    print(json.dumps({{"type": "tool_result", "tool_id": "t1", "status": "success", "output": r["result"]["content"][0]["text"][:80]}}), flush=True)
    print(json.dumps({{"type": "message", "role": "assistant", "content": "Keeping the base."}}), flush=True)
    print(json.dumps({{"type": "result", "status": "success", "stats": {{"input_tokens": 120, "output_tokens": 30, "total_tokens": 150, "tool_calls": 1}}}}), flush=True)
mcp.stdin.close(); mcp.wait()
""", encoding="utf-8")
            mock_gem.chmod(0o755)
            s_gm_ok = lses.LevelSession(h, tmp / "s_gm_ok", reach="behaviour", backend="gemini",
                                        book_path=tmp / "b_gm_ok.jsonl", probe_on_start=False)
            gm_out = ldrv.run_driver(
                s_gm_ok, "gemini", driver_model="gemini-mock",
                driver_budget=ldrv.DriverBudget(max_turns=6, max_tokens=100_000, max_dollars=1.0,
                                                input_usd_per_mtok=1.25, output_usd_per_mtok=5.0),
                config={"gemini_bin": str(mock_gem), "allow_mock_cli": True},
            )
            check(gm_out["harness_card"]["driver_family"] == "gemini" and gm_out["harness_card"]["face"] == "mcp_stdio"
                  and gm_out["stop_reason"] == "kept_base_abstention" and gm_out["driver_failure"] is None
                  and gm_out["door"]["ok"] == 1 and gm_out["usage"]["prompt_tokens"] == 120
                  and gm_out["usage"]["completion_tokens"] == 30 and gm_out["book"]["verified"],
                  f"4b: mock Gemini CLI driver succeeds over socket bridge with exact usage: {gm_out['stop_reason']} {gm_out['door']}")
            mock_gem_extra = tmp / "mock_gemini_extra.py"
            mock_gem_extra.write_text(f"""#!{sys.executable} -I
import json, time
print(json.dumps({{"type": "init", "session_id": "g2", "model": "gemini-mock"}}), flush=True)
print(json.dumps({{"type": "tool_use", "tool_name": "read_file", "tool_id": "t_bad", "parameters": {{}}}}), flush=True)
print(json.dumps({{"type": "tool_result", "tool_id": "t_bad", "status": "success", "output": "leaked"}}), flush=True)
time.sleep(5)
""", encoding="utf-8")
            mock_gem_extra.chmod(0o755)
            s_gm_ex = lses.LevelSession(h, tmp / "s_gm_ex", reach="behaviour", backend="gemini",
                                        book_path=tmp / "b_gm_ex.jsonl", probe_on_start=False)
            gm_ex = ldrv.run_gemini_cli_driver(
                s_gm_ex, gemini_bin=str(mock_gem_extra), allow_mock_cli=True,
                driver_budget=ldrv.DriverBudget(max_turns=6, max_dollars=1.0, max_wall_seconds=30.0),
            )
            check(gm_ex["stop_reason"] == "driver_failure" and "read_file" in (gm_ex.get("driver_failure") or ""),
                  f"4b: Gemini CLI driver kills session if a non-door tool succeeds: {gm_ex['stop_reason']} {gm_ex.get('driver_failure')}")

            # Real Gemini CLI test (door tools only, zero built-ins, loopback-only under bwrap --unshare-net)
            gemini_real = shutil.which("gemini")
            if gemini_real is None:
                skip("4b: the gemini CLI is not installed: real door-only run was skipped")
            else:
                real_gem_child = r'''
import json, os, socket, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
root, work, gem_bin, out_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, str(root))
from backend import level_drivers as ldrv, level_session as lses
seen = {"errors": [], "interfaces": sorted(os.listdir("/sys/class/net")) if os.path.isdir("/sys/class/net") else []}
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    gp = s.getsockname()[1]
reqs = []
class GemStub(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
        reqs.append(b)
        contents = b.get("contents") or []
        n_res = sum(1 for c in contents for p in (c.get("parts") or []) if isinstance(p, dict) and "functionResponse" in p)
        if b.get("tools") and n_res == 0:
            parts = [{"functionCall": {"name": "mcp_levelling_status", "args": {}}}]
        elif b.get("tools") and n_res == 1:
            parts = [{"functionCall": {"name": "read_file", "args": {"file_path": "/etc/hostname"}}}]
        else:
            parts = [{"text": "Stopping and keeping the base."}]
        chunk = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}],
                 "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 25, "totalTokenCount": 115}}
        data = (f"data: {json.dumps(chunk)}\n\n" if "stream" in self.path else json.dumps(chunk)).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream" if "stream" in self.path else "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
srv = ThreadingHTTPServer(("127.0.0.1", gp), GemStub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
fake_home = work / "fake_home"
(fake_home / ".gemini").mkdir(parents=True, exist_ok=True)
(fake_home / ".gemini" / "GEMINI.md").write_text("MARK_HOME_GEMINI_MD_LEAK\n", encoding="utf-8")
os.environ.update({
    "HOME": str(fake_home),
    "GEMINI_API_KEY": "stub-gemini-key",
    "GOOGLE_GEMINI_BASE_URL": f"http://127.0.0.1:{gp}",
    "NO_PROXY": "127.0.0.1,localhost",
})
seen["version"] = subprocess.run([gem_bin, "--version"], capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
try:
    sess = lses.LevelSession(None, work / "gm_proj", reach="behaviour", backend="gemini",
                             book_path=work / "b_gm_real.jsonl", probe_on_start=False)
    out = ldrv.run_gemini_cli_driver(
        sess, model="gemini-2.5-pro", gemini_bin=gem_bin,
        driver_budget=ldrv.DriverBudget(max_turns=6, max_tokens=1_000_000, max_dollars=0.5, max_wall_seconds=120.0),
    )
    door_tools = sorted("mcp_levelling_" + t for t in sess.allowed_tools())
    tool_sets = [sorted(fn["name"] for t in (r.get("tools") or []) for fn in (t.get("functionDeclarations") or []))
                 for r in reqs if r.get("tools")]
    fres = [p["functionResponse"] for c in (reqs[-1].get("contents") or []) for p in (c.get("parts") or [])
            if isinstance(p, dict) and "functionResponse" in p] if reqs else []
    seen["door_tools"] = door_tools
    seen["tool_sets"] = tool_sets
    seen["status_ok"] = bool(fres) and "harness_card" in json.dumps(fres[0])
    seen["read_refused"] = len(fres) >= 2 and "error" in (fres[1].get("response") or {})
    seen["md_leaked"] = "MARK_HOME_GEMINI_MD_LEAK" in json.dumps(reqs)
    seen["out"] = {"stop_reason": out["stop_reason"], "driver_failure": out.get("driver_failure"),
                   "door_ok": (out.get("door") or {}).get("ok"), "book_ok": (out.get("book") or {}).get("verified")}
except Exception as exc:
    seen["errors"].append(f"{type(exc).__name__}: {exc}")
finally:
    srv.shutdown()
    srv.server_close()
out_path.write_text(json.dumps(seen), encoding="utf-8")
'''
                real_gdir = tmp / "real_gemini_cli"
                real_gdir.mkdir()
                (real_gdir / "real_gem_child.py").write_text(real_gem_child, encoding="utf-8")
                gseen_p = real_gdir / "seen.json"
                gcmd = [sys.executable, "-I", str(real_gdir / "real_gem_child.py"), str(ROOT_DIR), str(real_gdir),
                        gemini_real, str(gseen_p)]
                if has_bwrap:
                    gcmd = [str(shutil.which("bwrap")), "--dev-bind", "/", "/", "--unshare-net", "--die-with-parent",
                            "--", *gcmd]
                grun = subprocess.run(gcmd, capture_output=True, text=True, timeout=300)
                try:
                    gseen = json.loads(gseen_p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    gseen = {"errors": [f"child wrote nothing (exit {grun.returncode}): {grun.stderr[-300:]}"]}
                g_sets = gseen.get("tool_sets") or []
                g_out = gseen.get("out") or {}
                check(bool(g_sets) and all(ts == gseen.get("door_tools") for ts in g_sets),
                      f"4b: real gemini CLI ({gseen.get('version')}) sends the model ONLY mcp_levelling_* door tools "
                      f"(0 built-ins; got {g_sets[0] if g_sets else 'none'}; errors={gseen.get('errors')})")
                check(gseen.get("status_ok") is True and gseen.get("read_refused") is True
                      and gseen.get("md_leaked") is False
                      and (g_out.get("door_ok") or 0) >= 1 and g_out.get("stop_reason") == "kept_base_abstention"
                      and g_out.get("driver_failure") is None and g_out.get("book_ok") is True,
                      f"4b: real gemini CLI door call succeeded, built-in read_file refused, no GEMINI.md leaked: {g_out}")

            # 4c. Antigravity CLI driver: allowlist, refusals, mock CLI happy path, and extra-tool kill guard
            os.environ["HF_TOKEN"] = "hf_secret_check"
            os.environ["OPENROUTER_API_KEY"] = "or_secret_check"
            os.environ["ANTHROPIC_API_KEY"] = "ant_secret_check"
            try:
                a_env = ldrv.agy_child_env()
                check("HF_TOKEN" not in a_env and "OPENROUTER_API_KEY" not in a_env
                      and "ANTHROPIC_API_KEY" not in a_env and "THEODOLITE_SCIENTIST_REACH" not in a_env,
                      "4c: agy_child_env strips unrelated API keys and THEODOLITE_SCIENTIST_REACH")
            finally:
                for k in ("HF_TOKEN", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
                    os.environ.pop(k, None)

            sess_a0 = lses.LevelSession(h, tmp / "s_agy_ref", reach="behaviour", backend="agy",
                                        book_path=tmp / "b_agy_ref.jsonl", probe_on_start=False)
            check(refuses(lambda: ldrv.run_agy_cli_driver(sess_a0, driver_budget=ldrv.DriverBudget(max_dollars=None)),
                          ldrv.DriverRefused),
                  "4c: AgyCLIDriver refused without owner-approved max_dollars")

            os.environ["THEODOLITE_SCIENTIST_REACH"] = "home"
            try:
                check(refuses(lambda: ldrv.run_agy_cli_driver(sess_a0, driver_budget=ldrv.DriverBudget(max_dollars=1.0),
                                                              allow_mock_cli=True), ldrv.DriverRefused),
                      "4c: AgyCLIDriver refused when THEODOLITE_SCIENTIST_REACH=home")
            finally:
                os.environ.pop("THEODOLITE_SCIENTIST_REACH", None)
            sess_a0.close()

            mock_agy = tmp / "mock_agy.py"
            mock_agy.write_text(
                f"#!{sys.executable} -I\n"
                "import json, os, subprocess, sys\n"
                "from pathlib import Path\n"
                "cwd = Path.cwd()\n"
                "mode = oct(os.stat(cwd).st_mode & 0o777)\n"
                "mcp_p = cwd / '.agents' / 'mcp_config.json'\n"
                "assert mode == '0o555' and mcp_p.exists(), (mode, mcp_p.exists())\n"
                "cfg = json.loads(mcp_p.read_text(encoding='utf-8'))\n"
                "srv = cfg['mcpServers']['levelling']\n"
                "settings_p = Path(sys.argv[sys.argv.index('--settings') + 1])\n"
                "st = json.loads(settings_p.read_text(encoding='utf-8'))\n"
                "assert 'mcp(levelling/*)' in st['permissions']['allow'] and 'read_file(*)' in st['permissions']['deny'], st\n"
                "p = subprocess.Popen([srv['command'], *srv['args']], stdin=subprocess.PIPE, stdout=subprocess.PIPE,\n"
                "                     text=True, env={**os.environ, **(srv.get('env') or {})})\n"
                "def mrpc(i, m, prm):\n"
                "    p.stdin.write(json.dumps({'jsonrpc':'2.0','id':i,'method':m,'params':prm})+'\\n')\n"
                "    p.stdin.flush()\n"
                "    return json.loads(p.stdout.readline())\n"
                "mrpc(1, 'initialize', {'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'m','version':'1'}})\n"
                "p.stdin.write(json.dumps({'jsonrpc':'2.0','method':'notifications/initialized','params':{}})+'\\n'); p.stdin.flush()\n"
                "mrpc(2, 'tools/call', {'name':'status','arguments':{}})\n"
                "p.stdin.close(); p.wait(timeout=5)\n"
                "print(json.dumps({'event':'init','init':{'cwd':str(cwd),'tools':['call_mcp_tool']}}), flush=True)\n"
                "print(json.dumps({'event':'step_update','step_update':{'step_type':'tool','state':'ACTIVE','tool_name':'call_mcp_tool','tool_info':{'parameters':{'ServerName':'levelling','ToolName':'status'}}}}), flush=True)\n"
                "print(json.dumps({'event':'step_update','step_update':{'step_type':'tool','state':'DONE','tool_name':'call_mcp_tool','tool_info':{'parameters':{'ServerName':'levelling','ToolName':'status'}}}}), flush=True)\n"
                "print(json.dumps({'event':'step_update','step_update':{'step_type':'agent_response','state':'DONE','text_delta':'mock agy abstained','usage':{'input_tokens':110,'output_tokens':22,'thinking_tokens':7}}}), flush=True)\n"
                "print(json.dumps({'event':'result','result':{'status':'SUCCESS','response':'mock agy abstained','num_turns':1,'usage':{'input_tokens':110,'output_tokens':22,'thinking_tokens':7}}}), flush=True)\n",
                encoding="utf-8",
            )
            os.chmod(mock_agy, 0o755)
            sess_a1 = lses.LevelSession(h, tmp / "s_agy_mock", reach="behaviour", backend="agy",
                                        book_path=tmp / "b_agy_mock.jsonl", probe_on_start=False)
            out_a1 = ldrv.run_driver(
                sess_a1, "agy", driver_model="mock-agy",
                driver_budget=ldrv.DriverBudget(max_wall_seconds=30.0, max_tokens=10000, max_dollars=0.5),
                config={"agy_bin": str(mock_agy), "allow_mock_cli": True},
            )
            sess_a1.close()
            check(out_a1["harness_card"]["driver_family"] == "agy" and out_a1["harness_card"]["face"] == "mcp_stdio"
                  and out_a1["stop_reason"] == "kept_base_abstention"
                  and out_a1["driver_failure"] is None and (out_a1["door"].get("ok") or 0) >= 1
                  and out_a1["usage"]["prompt_tokens"] == 110 and out_a1["usage"]["completion_tokens"] == 22
                  and out_a1["usage"]["reasoning_tokens"] == 7 and out_a1["book"]["verified"],
                  f"4c: AgyCLIDriver mock happy path verified: {out_a1['stop_reason']} {out_a1['door']}")

            mock_agy_extra = tmp / "mock_agy_extra.py"
            mock_agy_extra.write_text(
                f"#!{sys.executable} -I\n"
                "import json, time\n"
                "print(json.dumps({'event':'step_update','step_update':{'step_type':'tool','state':'ACTIVE','tool_name':'read_file','tool_info':{'parameters':{'Path':'/etc/hostname'}}}}), flush=True)\n"
                "print(json.dumps({'event':'step_update','step_update':{'step_type':'tool','state':'DONE','tool_name':'read_file','tool_info':{'parameters':{'Path':'/etc/hostname'}}}}), flush=True)\n"
                "time.sleep(5)\n",
                encoding="utf-8",
            )
            os.chmod(mock_agy_extra, 0o755)
            sess_a2 = lses.LevelSession(h, tmp / "s_agy_extra", reach="behaviour", backend="agy",
                                        book_path=tmp / "b_agy_extra.jsonl", probe_on_start=False)
            out_a2 = ldrv.run_driver(
                sess_a2, "agy", driver_model="mock-agy",
                driver_budget=ldrv.DriverBudget(max_wall_seconds=30.0, max_tokens=10000, max_dollars=0.5),
                config={"agy_bin": str(mock_agy_extra), "allow_mock_cli": True},
            )
            sess_a2.close()
            check(out_a2["stop_reason"] == "driver_failure" and out_a2["driver_failure"] is not None
                  and "read_file" in (out_a2["door"].get("other_tools_asked") or []),
                  f"4c: AgyCLIDriver kills & marks driver_failure on non-door tool execution: {out_a2}")

            # ========================================================================= 5. Multi-session study & exact stats
            section("5. Multi-session Phase 6 study, isolation, unpooled aggregation, and exact statistics")
            import math as _m567  # noqa: E402  (sections 5 to 7 use it under this name; the header is shared)
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

            # Comparison labels on made-up outcomes: no arm named ahead without p < 0.05, the same both ways. A made-up
            # session ran to an end (its driver did not fail, a door call succeeded) unless `ended` says otherwise
            def _syn(arm_id: str, fam: str, model: str, reach: str, mode: str, verdict: str, de: Optional[float],
                     effort: Optional[str] = None, rep: int = 0, **ended: Any) -> Dict[str, Any]:
                return {"arm_id": arm_id, "replicate": rep, "run_id": f"{arm_id}_r{rep}", "verdict": verdict,
                        "submitted": verdict != "no_submission",
                        "harness_card": {"driver_family": fam, "driver_model": model, "reach": reach, "mode": mode,
                                         "bare": False, "reasoning_effort": effort},
                        "stop_reason": "claimed" if verdict != "no_submission" else "kept_base_abstention",
                        "driver_failure": None, "door": {"calls": 2, "ok": 2, "errors": 0},
                        "metrics": {"directed_effect": de}, "usage": {"gpu_seconds": 2.0}, "book": {"verified": True},
                        **ended}

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
                  and sorted(a["arm_ids"][0] for a in s_eff["arms"]) == ["oa_hi", "oa_lo"]
                  and all(len(a["arm_ids"]) == 1 for a in s_eff["arms"]),
                  f"5: two arms that differ only in reasoning effort are two rows, never pooled ({[a['arm_id'] for a in s_eff['arms']]})")

            # A session counts in its arm's rates only when it ran to an end. At d5e6848 every session not refused
            # counted, so five sessions that never worked made a no-lead arm and named the other arm ahead
            NOT_RUN = [
                {"driver_failure": "No door call succeeded (1 were made and every one failed): the driver did not work the door.",
                 "stop_reason": "driver_failure", "door": {"calls": 1, "ok": 0, "errors": 1}},
                {"stop_reason": "killed"},
                {"stop_reason": "error", "notes": "endpoint error: boom"},
                {"stop_reason": "still_running"},
                {"door": None},
                {"stop_reason": "budget_dollars", "door": {"calls": 0, "ok": 0, "errors": 0}},
                {"stop_reason": "error", "notes": "status ok", "door": {"calls": 2, "ok": 1, "errors": 1},
                 "claim_refused": 'Brief "pronoun_pull" has already spent all 2 of its fresh readings'},
            ]
            s_fail = lev.summarize_study(
                [_syn("oa_w", "openai", "m", "weights", "weight_edit", "lead", 1.0, rep=i) for i in range(5)]
                + [_syn("scr_w", "scripted", "scripted_weight_v1", "weights", "weight_edit", "no_submission", None, rep=i,
                        **NOT_RUN[i]) for i in range(len(NOT_RUN))],
                {**_syn_plan(("oa_w", "openai", "weights", "weight_edit", None),
                             ("scr_w", "scripted", "weights", "weight_edit", None)), "sessions_per_arm": len(NOT_RUN)})
            f_arms = {a["arm_id"]: a for a in s_fail["arms"]}
            f_row = s_fail["comparisons"]["agent_vs_scripted"][0]
            f_rep = lev.render_study_report(s_fail)
            f_why = [s["reason"] for s in s_fail["not_run_to_an_end"]]
            check(f_arms["scr_w"]["n_sessions"] == 0 and f_arms["scr_w"]["n_not_counted"] == len(NOT_RUN)
                  and f_arms["oa_w"]["n_sessions"] == 5 and f_row["comparison"] == "not_compared"
                  and s_fail["counted_sessions"] == 5 and len(s_fail["not_run_to_an_end"]) == len(NOT_RUN)
                  and "did not work the door" in f_why[0] and "kill switch" in f_why[1] and "endpoint error: boom" in f_why[2]
                  and "still running" in f_why[3] and "records no door calls" in f_why[4] and "no door call succeeded" in f_why[5]
                  and "a claim the door refused: Brief" in f_why[6] and "already spent all 2" in f_rep
                  and "## Sessions that did not run to an end (not counted)" in f_rep and "scr_w_r3" in f_rep,
                  f"5: a session counts only when it ran to an end: a driver failure, STOP, an error (a refused claim's "
                  f"own words among them), a call still running, no door record and no door call that succeeded are each "
                  f"listed apart with their reason, and no arm is named ahead on them ({f_row['comparison']}; "
                  f"{f_arms['scr_w']['n_sessions']} of {len(NOT_RUN)} counted)")
            # A matched lead is a lead (level_claim.LEADS): at d5e6848 five of them against five no-submissions read
            # "neither_found_a_lead"
            s_ml = lev.summarize_study(
                [_syn("oa_w", "openai", "m", "weights", "weight_edit", "lead, matched by a cheaper knob", 2.0, rep=i) for i in range(5)]
                + [_syn("scr_w", "scripted", "scripted_weight_v1", "weights", "weight_edit", "no_submission", None, rep=i) for i in range(5)],
                {**_syn_plan(("oa_w", "openai", "weights", "weight_edit", None),
                             ("scr_w", "scripted", "weights", "weight_edit", None)), "sessions_per_arm": 5})
            ml_arm = next(a for a in s_ml["arms"] if a["arm_id"] == "oa_w")
            ml_row = s_ml["comparisons"]["agent_vs_scripted"][0]
            check(ml_arm["n_leads"] == 5 and ml_arm["n_leads_matched_by_knob"] == 5 and ml_arm["lead_effect"]["n"] == 5
                  and ml_row["comparison"] == "agent_ahead" and list(lcl.LEADS) == ml_arm["lead_verdicts"]
                  and "matched by a cheaper knob" in lev.render_study_report(s_ml),
                  f"5: a lead matched by a cheaper knob counts as a lead in every comparison, and is counted apart too "
                  f"({ml_arm['n_leads']} leads, {ml_row['comparison']})")
            # The separation sentence gives each comparison its own reason: at 2 and 8 sessions a side it is the
            # 3-session minimum, not the smallest p (which is below 0.05 there)
            s_28 = lev.summarize_study(
                [_syn("w", "scripted", "scripted_weight_v1", "weights", "weight_edit", "lead", 1.0, rep=i) for i in range(2)]
                + [_syn("k", "scripted", "scripted_knob_v1", "weights", "knobs_only", "no_submission", None, rep=i) for i in range(8)],
                {**_syn_plan(("w", "scripted", "weights", "weight_edit", None), ("k", "scripted", "weights", "knobs_only", None)),
                 "sessions_per_arm": 8})
            check(lev.smallest_fisher_p(2, 8) < 0.05 and "at least 3 sessions" in s_28["separation"]
                  and "smallest two-sided Fisher p" not in s_28["separation"]
                  and "no comparison in this study can separate two arms" in s_28["separation"],
                  f"5: at 2 and 8 sessions a side the summary gives the 3-session minimum as the reason ({s_28['separation']})")
            nm_blk = (s_harm["arms"][1] if s_harm["arms"][1]["arm_id"] == "k" else s_harm["arms"][0]).get("near_miss_abs") or {}
            check("whatever the verdict" in str(nm_blk.get("of")) and "near_miss_abs_median" not in json.dumps(s_harm)
                  and "whatever the verdict" in str(((s_harm["arms"][0]).get("neutral_kl") or {}).get("of")),
                  "5: the near-miss and neutral-KL medians say they are taken over every counted verdict")

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
                  )), lev.EvalRefused, "in driver_config"),
                  "5: validate_study_plan refuses a reasoning effort a driver would not be given, and two efforts on one arm")
            # Refused before anything is loaded, each of which passed validation at d5e6848: an effort named in
            # driver_config only (on a scripted arm, and on an openai arm whose row said none), a Claude Code effort it
            # does not take, and a claim design with no control edits
            cfg_only = [lev.EvalArm("s_cfg", "scripted", reach="behaviour", driver_config={"reasoning_effort": "high"}),
                        lev.EvalArm("o_cfg", "openai", reach="behaviour", driver_model="m", driver_config={"reasoning_effort": "high"})]
            check(all(refuses(lambda a=a: lev.validate_study_plan(lev.EvalStudyPlan(
                name="cfg_effort", brief_name="pronoun_pull", sessions_per_arm=1, arms=(a,)), check_paths=False),
                lev.EvalRefused, "name it as the arm's reasoning_effort") for a in cfg_only)
                  and refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                      name="claude_effort", brief_name="pronoun_pull", sessions_per_arm=1, driver_budget=ldrv.DriverBudget(max_dollars=1.0),
                      arms=(lev.EvalArm("c0", "claude", reach="behaviour", reasoning_effort="extreme",
                                        driver_config={"allow_mock_cli": True}),)), check_paths=False),
                      lev.EvalRefused, "Claude Code's effort is one of")
                  and refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                      name="no_controls", brief_name="pronoun_pull", sessions_per_arm=1, claim_design={"controls_override": 0},
                      arms=(lev.EvalArm("a0", "scripted", reach="behaviour"),)), check_paths=False),
                      lev.EvalRefused, "controls_override is a whole number of 1 or more")
                  and refuses(lambda: lev.validate_study_plan(lev.EvalStudyPlan(
                      name="bad_tol", brief_name="pronoun_pull", sessions_per_arm=1, claim_design={"match_tolerance": 0.5},
                      arms=(lev.EvalArm("a0", "scripted", reach="behaviour"),)), check_paths=False),
                      lev.EvalRefused, "finite factor of 1 or more"),
                  "5: validate_study_plan refuses an effort named only in driver_config, a Claude Code effort outside "
                  "level_drivers.CLAUDE_EFFORTS, and a claim design with no control edits or a tolerance below 1")
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
            # Every session of it runs to an end and is counted (the scripted weight baseline puts forward only an edit
            # the claim gate can take, so at the activations reach, whose candidate is low-rank, it keeps the base)
            nrte = summ["not_run_to_an_end"]
            check(sres["book"]["verified"] and len(sres["outcomes"]) == 21 and len(summ["arms"]) == 7
                  and sorted(a["arm_id"] for a in summ["arms"]) == sorted(a.arm_id for a in study_plan.arms)
                  and all(a["arm_ids"] == [a["arm_id"]] and a["n_recorded"] == 3 and a["n_sessions"] == 3 for a in summ["arms"])
                  and summ["counted_sessions"] == 21 and not nrte,
                  f"5: 7-arm x 3-session (21 sessions) study completed with intact study_book, one row an arm, and every "
                  f"session run to an end and counted ({summ['counted_sessions']} counted; not: "
                  f"{[(s['run_id'], s['reason'][:70]) for s in nrte][:2]})")
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
            # Beside knob lead counts the report says what a knob claim's controls are matched on, in level_knobs' own
            # statements (at d5e6848 the table set knob leads beside weight leads and said nothing of them)
            wvk_part = sres["report"].split("## Weight edits against cheaper knobs", 1)[-1].split("\n## ", 1)[0]
            check(lkn.CONTROL_RULE_SAY in wvk_part and lkn.CONTROL_FAMILY_SAY["random_directions"] in wvk_part
                  and all(f"- {fam}:" in wvk_part for fam in lkn.CONTROL_FAMILY_SAY),
                  "5: beside the weight-against-knobs table the report gives what each family of knob controls is and is "
                  "not matched on, and the rule they are counted by (level_knobs.CONTROL_FAMILY_SAY, CONTROL_RULE_SAY)")
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
                  and (mix_dir / "summary.json").is_file() and "refused before they ran" in mres["report"]
                  and "refused before it ran" in mres["report"]
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

            # A study book edited by hand, its chain worked out again, does not reload clean: each session's verdict,
            # submitted flag and directed effect are read again from its side book's own lines (at d5e6848 three
            # status-only sessions reloaded as three leads)
            edit_dir = tmp / "p6_mix_edited"
            shutil.copytree(mix_dir, edit_dir)
            ed_rows = [json.loads(ln) for ln in (edit_dir / "study_book.jsonl").read_text(encoding="utf-8").splitlines()]
            ed_prev = lb.GENESIS
            for ed_r in ed_rows:
                ed_b = ed_r.get("body") or {}
                if ed_b.get("note") == "eval_session" and ed_b.get("arm_id") == "rp_status":
                    ed_b["outcome"]["verdict"], ed_b["outcome"]["submitted"] = "lead", True
                    ed_b["outcome"].setdefault("metrics", {})["directed_effect"] = 2.5
                ed_r["prev"] = ed_prev
                ed_r["hash"] = lb.entry_hash(ed_r)
                ed_prev = ed_r["hash"]
            (edit_dir / "study_book.jsonl").write_text(
                "".join(lb.canonical(r).decode("ascii") + "\n" for r in ed_rows), encoding="utf-8")
            check(lb.verify(edit_dir / "study_book.jsonl")["ok"]
                  and refuses(lambda: lev.load_study_from_book(edit_dir), lev.EvalRefused, "its side book's own lines say")
                  and lev.verdict_of_own_lines(mix_dir / "sessions" / "rp_status_r0" / "level_book.jsonl",
                                               lb.verify(seeded_cal)["entries"])["verdict"] == "no_submission",
                  "5: a study book edited by hand (its chain worked again) is refused on reload: a note's verdict, "
                  "submitted flag and directed effect are re-derived from its side book's own lines")

            # A calibration book that leaves the brief no fresh reading is refused before anything is written: every
            # side book would start as a copy of it, and the gate would refuse every session's claim (at d5e6848 the
            # study ran, and every claim was refused)
            spent_book = tmp / "spent_cal_book.jsonl"
            shutil.copy2(study_dir / "sessions" / "scr_beh_edit_r0" / "level_book.jsonl", spent_book)
            for k_i in range(int(lcl.brief_history(brief.id, spent_book, name=brief.name)["fresh_count"]), brief.max_fresh_readings):
                lcl.run_fresh_claim(h, brief, bars, {"terms": [{"layer": h.n_layers - 1, "silence": [20 + 2 * k_i, 21 + 2 * k_i]}]},
                                    base_dir=tmp, book_path=spent_book, run_prompt_arms=True, **study_cd)
            spent_dir = tmp / "p6_spent"
            spent_plan = lev.EvalStudyPlan(
                name="p6_spent", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=1,
                session_budget=study_sb, driver_budget=ldrv.DriverBudget(max_turns=16, max_dollars=0.0), claim_design=study_cd,
                arms=(lev.EvalArm("scr_beh", "scripted", reach="behaviour", driver_config={"submit_even_if_weak": True}),))
            check(int(lcl.brief_history(brief.id, spent_book, name=brief.name)["fresh_count"]) == brief.max_fresh_readings
                  and refuses(lambda: lev.run_eval_study(h, spent_plan, spent_dir, bars=bars, calibration_book=spent_book,
                                                         probe_sandbox=False), lev.EvalRefused, "fresh reading(s)")
                  and not spent_dir.exists(),
                  f"5: a calibration book whose brief has no fresh reading left ({brief.max_fresh_readings} of "
                  f"{brief.max_fresh_readings} spent) is refused before the study writes anything")
            # A claim the door refused is in the outcome in the door's own words, and in the report
            cr_dir = tmp / "p6_claim_refused"
            cr_plan = lev.EvalStudyPlan(
                name="p6_claim_refused", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=1,
                session_budget=study_sb, driver_budget=ldrv.DriverBudget(max_turns=8, max_dollars=0.0, max_wall_seconds=60.0),
                claim_design=study_cd,
                arms=(lev.EvalArm("rp_bad_claim", "replay", reach="behaviour", driver_config={"steps": [
                    {"tool": "status", "args": {}},
                    {"tool": "claim", "args": {"edit": {"terms": [{"layer": h.n_layers + 7, "silence": [1]}]}}}]}),))
            cr_res = lev.run_eval_study(h, cr_plan, cr_dir, bars=bars, calibration_book=cal_book, probe_sandbox=False)
            DOCS_COLLECTED.append(cr_res["report"])
            cr_out = cr_res["outcomes"][0]
            cr_words = str(cr_out.get("claim_refused") or "")
            check(bool(cr_words) and cr_words[:60] in cr_res["report"] and "the door refused" in cr_res["report"]
                  and cr_out.get("verdict") == "no_submission",
                  f"5: a claim the gate refused reaches the outcome and the report in its own words ({cr_words[:90]!r})")
            # A STOP file that appears mid-study: the session it stopped did not run to an end, and is not counted (at
            # d5e6848 it counted as a session that kept the base)
            stop_dir = tmp / "p6_stop_mid"
            stop_dir.mkdir()

            def _tr_stop(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                (stop_dir / "STOP").write_text("now\n", encoding="utf-8")
                return {"model": "mock-stop", "usage": {"prompt_tokens": 600, "completion_tokens": 20},
                        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                            {"id": "s", "type": "function", "function": {"name": "status", "arguments": "{}"}}]}}]}

            sm_plan = lev.EvalStudyPlan(
                name="p6_stop_mid", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=3,
                session_budget=study_sb, claim_design=study_cd,
                driver_budget=ldrv.DriverBudget(max_turns=8, max_dollars=0.0, max_wall_seconds=60.0),
                arms=(lev.EvalArm("oa_stopped", "openai", reach="behaviour", driver_model="mock-stop", driver_config={"transport_fn": _tr_stop}),
                      lev.EvalArm("rp_b", "replay", reach="behaviour", driver_config={"steps": [{"tool": "status", "args": {}}]})))
            sm_res = lev.run_eval_study(h, sm_plan, stop_dir, bars=bars, calibration_book=cal_book, probe_sandbox=False)
            sm_arm = next(a for a in sm_res["summary"]["arms"] if a["arm_id"] == "oa_stopped")
            check(str(sm_res["halted_reason"]).startswith("kill_switch") and len(sm_res["outcomes"]) == 1
                  and sm_res["outcomes"][0]["stop_reason"] == "killed" and sm_arm["n_sessions"] == 0
                  and sm_arm["n_not_counted"] == 1 and "kill switch" in sm_arm["not_counted"][0]["reason"],
                  f"5: a session a STOP file stopped mid-run is listed as not run to an end, not counted as one that kept "
                  f"the base ({sm_arm['n_sessions']} counted; halted: {sm_res['halted_reason']})")

            # A study dollar cap is kept across sessions that really spend. Each session's cap is what is left of the
            # study's (rounded down to a millionth), or the plan's own where smaller; when what is left comes to
            # nothing the study stops. At d5e6848 a remainder of $0.0000004 was handed to the third session as a cap of
            # 0.0, which the OpenAI driver read as none: it ran to its turn cap and spent $0.20 past the study's cap
            def _tr_pay(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                return {"model": "mock-pay", "usage": {"prompt_tokens": 300, "completion_tokens": 50, "cost": 0.025},
                        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                            {"id": "s1", "type": "function", "function": {"name": "status", "arguments": "{}"}}]}}]}

            pay_cap, pay_per = 0.1000004, 0.06
            pay_plan = lev.EvalStudyPlan(
                name="p6_pay", brief_name=brief.name, brief_path=str(brief_json_p), sessions_per_arm=3,
                session_budget=study_sb, claim_design=study_cd, max_total_dollars=pay_cap,
                driver_budget=ldrv.DriverBudget(max_turns=8, max_dollars=pay_per, max_wall_seconds=60.0,
                                                input_usd_per_mtok=1e-6, output_usd_per_mtok=1e-6),
                arms=(lev.EvalArm("oa_pay", "openai", reach="behaviour", driver_model="mock-pay", driver_config={"transport_fn": _tr_pay}),),
            )
            pres = lev.run_eval_study(h, pay_plan, tmp / "p6_pay", bars=bars, calibration_book=cal_book, probe_sandbox=False)
            p_caps = [((o.get("harness_card") or {}).get("driver_budget") or {}).get("max_dollars") for o in pres["outcomes"]]
            p_spent = [float((o.get("usage") or {}).get("estimated_dollars") or 0.0) for o in pres["outcomes"]]
            p_left = [_m567.floor((pay_cap - sum(p_spent[:i])) * 1e6) / 1e6 for i in range(len(p_caps))]
            p_want = [pay_per if pay_per <= left else left for left in p_left]
            check(len(p_caps) >= 2 and p_caps == p_want and all(c is not None and c > 0.0 for c in p_caps)
                  and sum(1 for s in p_spent if s > 0.0) >= 2 and p_caps[1] < pay_per
                  and sum(p_spent) <= pay_cap + 0.025 + 1e-9
                  and pres["halted_reason"] == lev.STUDY_DOLLAR_CAP_REACHED
                  and lev._session_driver_budget(pay_plan, pay_cap - 4e-7) is None
                  and lev._session_driver_budget(pay_plan, 0.0).max_dollars == pay_per,
                  f"5: a {pay_cap} study cap is kept across sessions that spend: each session's cap is what is left (or the "
                  f"plan's {pay_per}), never 0, and the study stops when nothing is left (caps {p_caps}, spent {p_spent}, "
                  f"halted: {pres['halted_reason']})")

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
            # An outcome that says a call was still running when the session was stopped halts the study: the model is
            # not known to be free, so no later session starts (at d5e6848 the study went on to the next session)
            def _still_running_driver(*_a: Any, **_k: Any) -> Dict[str, Any]:
                return {"stop_reason": "still_running", "driver_failure": None, "door": {"calls": 1, "ok": 1, "errors": 0},
                        "verdict": "no_submission", "submitted": False, "metrics": {}, "usage": {}, "all_claims": [],
                        "harness_card": {"driver_family": "replay", "driver_model": "replay_v1", "reach": "behaviour",
                                         "mode": "weight_edit", "bare": False},
                        "book": {"verified": True, "verdict_seq": None}}

            sr_dir = tmp / "p6_still_running"
            ldrv.run_driver = _still_running_driver  # type: ignore[assignment]
            try:
                sr_res = lev.run_eval_study(
                    h, lev.EvalStudyPlan(name="p6_still_running", brief_name=brief.name, brief_path=str(brief_json_p),
                                         sessions_per_arm=2, session_budget=study_sb, claim_design=study_cd,
                                         driver_budget=ldrv.DriverBudget(max_dollars=0.0),
                                         arms=(lev.EvalArm("rp_a", "replay", reach="behaviour", driver_config={"steps": []}),
                                               lev.EvalArm("rp_b", "replay", reach="behaviour", driver_config={"steps": []}))),
                    sr_dir, bars=bars, calibration_book=cal_book, probe_sandbox=False)
            finally:
                ldrv.run_driver = real_run_driver  # type: ignore[assignment]
            check(str(sr_res["halted_reason"]).startswith("still_running") and len(sr_res["outcomes"]) == 1
                  and sr_res["summary"]["counted_sessions"] == 0
                  and "still running" in sr_res["summary"]["not_run_to_an_end"][0]["reason"]
                  and lev.load_study_from_book(sr_dir)["finished"],
                  f"5: an outcome that says a call was still running halts the study before another session starts, and "
                  f"that session is not counted ({sr_res['halted_reason']})")
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
                  and by_cid["planted_noref"]["scored_as"] == "measurement"
                  and by_cid["planted_noref"]["can_propose"] is True
                  and b_summary["planted_layer_hit_rate"] == 0.0
                  and str(b_summary["planted_layer_hit_note"]).startswith("measured on 1 planted case"),
                  f"6: a replay given the answer is a plumbing check, not localization, while a real driver with "
                  f"propose_repair is scored as a measurement ({b_summary['planted_scored_as']})")

            # Live measurement via propose_repair: a mock OpenAI driver submits a checkpoint-faithful `add` repair on a
            # planted case and an explicit abstention on a canary case, with identical door surfaces and zero leakage.
            _mod0 = we._linear_of(h.lm, we.linear_name(0, "down_proj"))
            _w0_cols = we.dense_cols(_mod0, we._long([2, 3])).float().cpu().numpy().astype(np.float32)  # [2, D]
            def _fsha(p: Path) -> str:
                return hashlib.sha256(p.read_bytes()).hexdigest()

            def _make_blind_transport(propose_args: Dict[str, Any]) -> Any:
                step_i = {"n": 0}
                def _tx(_payload: Dict[str, Any], _hdrs: Dict[str, str], _url: str, _timeout: float) -> Dict[str, Any]:
                    step_i["n"] += 1
                    if step_i["n"] == 1:
                        return {
                            "model": "mock-blind",
                            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "status", "arguments": "{}"}}
                            ]}}],
                            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                        }
                    if step_i["n"] == 2:
                        return {
                            "model": "mock-blind",
                            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                                {"id": "c2", "type": "function", "function": {
                                    "name": "propose_repair",
                                    "arguments": json.dumps(propose_args),
                                }}
                            ]}}],
                            "usage": {"prompt_tokens": 60, "completion_tokens": 20},
                        }
                    return {
                        "model": "mock-blind",
                        "choices": [{"message": {"role": "assistant", "content": "Done."}}],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 5},
                    }
                return _tx

            # Use a planted `add` fault so adding its exact negation composes identically on a checkpoint!
            _m_dir = tmp / "blinded_measured"
            _m_dir.mkdir(parents=True, exist_ok=True)
            _delta_vec = np.full((2, _w0_cols.shape[1]), 0.05, dtype=np.float32)
            _plant_add_p = _m_dir / "blind_plant_add.npy"
            np.save(_plant_add_p, _delta_vec)
            _plant_add_rec = {"terms": [{"layer": 0, "add": [2, 3], "vectors": "blind_plant_add.npy",
                                         "sha256": _fsha(_plant_add_p)}]}
            _neg_ref_p = tmp / "neg_ref.npy"
            np.save(_neg_ref_p, np.full((2, _w0_cols.shape[1]), -0.05, dtype=np.float32))
            _neg_ref_sha = _fsha(_neg_ref_p)

            def _make_planted_add_transport() -> Any:
                step_i = {"n": 0}
                def _tx(_payload: Dict[str, Any], _hdrs: Dict[str, str], _url: str, _timeout: float) -> Dict[str, Any]:
                    step_i["n"] += 1
                    if step_i["n"] == 1:
                        code = (
                            "import numpy as np\n"
                            f"arr = np.full((2, {_w0_cols.shape[1]}), -0.05, dtype=np.float32)\n"
                            "info = save_array('neg.npy', arr)\n"
                            "print(info['sha256'])\n"
                        )
                        return {
                            "model": "mock-blind",
                            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "cell", "arguments": json.dumps({"code": code})}}
                            ]}}],
                            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                        }
                    if step_i["n"] == 2:
                        rec = {"terms": [{"layer": 0, "add": [2, 3], "vectors": "neg.npy", "sha256": _neg_ref_sha}]}
                        return {
                            "model": "mock-blind",
                            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
                                {"id": "c2", "type": "function", "function": {
                                    "name": "propose_repair",
                                    "arguments": json.dumps({"edit": rec, "diagnosis": "localized layer 0 cols 2,3", "confidence": 0.9}),
                                }}
                            ]}}],
                            "usage": {"prompt_tokens": 60, "completion_tokens": 30},
                        }
                    return {
                        "model": "mock-blind",
                        "choices": [{"message": {"role": "assistant", "content": "Submitted."}}],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 5},
                    }
                return _tx

            _local_ep = {"base_url": "http://127.0.0.1:1234/v1"}
            m_cases = [
                lev.BlindedCase("diag_alpha", "planted", reach="activations", plant_recipe=_plant_add_rec,
                                driver_family="openai", driver_model="mock-blind",
                                driver_config={**_local_ep, "transport_fn": _make_planted_add_transport()}),
                lev.BlindedCase("diag_beta", "canary", reach="activations",
                                driver_family="openai", driver_model="mock-blind",
                                driver_config={**_local_ep, "transport_fn": _make_blind_transport(
                                    {"edit": None, "diagnosis": "no anomaly found", "confidence": 0.85}
                                )}),
            ]
            m_summ = lev.run_blinded_benchmark(
                h, m_cases, tmp / "blinded_measured", weights_dir=weights_dir,
                session_budget=lses.SessionBudget(min_controls=1),
                driver_budget=ldrv.DriverBudget(max_turns=8, max_dollars=1.0,
                                                input_usd_per_mtok=1.0, output_usd_per_mtok=1.0),
                probe_sandbox=False,
            )
            m_by = {r["case_id"]: r for r in m_summ["cases"]}
            mp_c, mc_c = m_by["diag_alpha"], m_by["diag_beta"]
            check(m_summ["door_surface_equal_by_reach"] == {"activations": True}
                  and m_summ["kind_or_id_words_shown"] == {}
                  and mp_c["scored_as"] == "measurement" and mc_c["scored_as"] == "measurement"
                  and (mp_c.get("localization") or {}).get("layer_hit") is True
                  and (mp_c.get("localization") or {}).get("site_recall") == 1.0
                  and mp_c.get("recovery_fraction") == 1.0 and mp_c.get("recovery_holds_on_checkpoint") is True
                  and mc_c.get("honest_abstention") is True
                  and m_summ["planted_layer_hit_rate"] == 1.0 and m_summ["canary_abstention_rate"] == 1.0,
                  f"6: blinded propose_repair enables real driver measurements with identical door surface, zero "
                  f"leakage, checkpoint-faithful recovery and canary abstention "
                  f"(scored={mp_c.get('scored_as')}/{mc_c.get('scored_as')}, loc={mp_c.get('localization')}, "
                  f"rec={mp_c.get('recovery_fraction')}, ckpt={mp_c.get('recovery_holds_on_checkpoint')}, "
                  f"abst={mc_c.get('honest_abstention')})")
            # Non-blinded session refuses propose_repair; blinded session refuses a second propose_repair call
            s_nb = lses.LevelSession(h, tmp / "nb_sess", reach="behaviour", backend="local",
                                     book_path=tmp / "nb_book.jsonl", probe_on_start=False)
            s_bl2 = lses.LevelSession(h, tmp / "bl2_sess", reach="behaviour", backend="local",
                                      book_path=tmp / "bl2_book.jsonl", probe_on_start=False, canary=True)
            try:
                check("propose_repair" not in s_nb.allowed_tools() and "propose_repair" in s_bl2.allowed_tools(),
                      "6: propose_repair is served on blinded sessions only")
                check(refuses(lambda: s_nb.propose_repair(None), lses.SessionRefused),
                      "6: non-blinded session refuses propose_repair")
                r_first = s_bl2.propose_repair(None, diagnosis="clean", confidence=0.9)
                check(r_first["recorded"] is True and r_first["abstained"] is True and s_bl2.proposal_submitted is True,
                      "6: first propose_repair call succeeds and marks submission")
                check(refuses(lambda: s_bl2.propose_repair(None, diagnosis="again"), lses.SessionRefused),
                      "6: second propose_repair call in the same session is refused")
            finally:
                s_nb.close()
                s_bl2.close()
            comp = pr.get("repair_composition") or {}
            check(pr.get("kl_plant_alone") is not None and pr["kl_plant_alone"] > 0.0
                  and pr.get("kl_after_repair") == 0.0 and pr.get("figures_by") == "arithmetic"
                  and (pr.get("repair_reading") or {}).get("status") == "not read"
                  and lev.NO_CHANGE_SAY in str((pr.get("repair_reading") or {}).get("what"))
                  and comp.get("composite_is_base") is True and comp.get("repair_touches_plant_neurons") is True
                  and pr.get("recovery_holds_on_checkpoint") is False and "f1 + f2 - 1" in str(comp.get("rule")),
                  f"6: plant plus a scale x2 repair comes to the unedited model in the harness: its KL is 0 by "
                  f"arithmetic and not read, no refusal is taken for the figure, and the recovery is marked as not "
                  f"holding on a checkpoint ({(pr.get('repair_reading') or {}).get('status')})")
            check(pp.get("kl_after_repair") is not None and 0.0 < pp["kl_after_repair"] < pp["kl_plant_alone"]
                  and pp.get("recovery_fraction") is not None and 0.0 < pp["recovery_fraction"] < 1.0
                  and pp.get("recovery_holds_on_checkpoint") is False and pp.get("figures_by") == "reading"
                  and (pp.get("repair_composition") or {}).get("composite_is_base") is False,
                  f"6: a repair the harness can read gets its figure, with the composition rule beside it "
                  f"(kl {pp.get('kl_plant_alone')} -> {pp.get('kl_after_repair')}, recovery {pp.get('recovery_fraction')}, "
                  f"holds on a checkpoint: {pp.get('recovery_holds_on_checkpoint')})")

            # Each case runs under a neutral id, as its session's id and its folder's name; no case's kind or id word
            # is in what its driver is shown (at d5e6848 the status said "blind_planted_ref", and the work folder was
            # case_planted_ref/work, which the Claude Code CLI puts in its request)
            b_root = tmp / "blinded_out"
            b_dirs = sorted(p.name for p in b_root.iterdir() if p.is_dir())
            b_nids = [r.get("neutral_id") for r in b_summary["cases"]]
            closed_ids = [json.loads((b_root / str(n) / "session_closed.json").read_text(encoding="utf-8")).get("session_id")
                          if (b_root / str(n) / "session_closed.json").is_file() else None for n in b_nids]
            check(all(isinstance(n, str) and re.fullmatch(r"[0-9a-f]{16}", n) for n in b_nids)
                  and sorted(b_nids) == b_dirs and closed_ids == b_nids
                  and not any(c.case_id in d or c.kind in d for c in b_cases for d in b_dirs)
                  and b_summary["kind_or_id_words_shown"] == {}
                  and all(r["kind_or_id_words_shown"] == [] for r in b_summary["cases"]),
                  f"6: every blinded case runs under a neutral id as its session id and folder, and no kind or case-id word "
                  f"is in its status, harness card, driver prompt or work path ({b_dirs[:2]}...)")
            # The composition table held to weight_edits.stack on the stand-in: for each plant and repair, the harness's
            # composite (the stack's edit on the base columns) against the repair's own ops on the planted columns
            Lc, idx_c = 0, [2, 3]
            mod_d = we._linear_of(h.lm, we.linear_name(Lc, "down_proj"))
            mod_u = we._linear_of(h.lm, we.linear_name(Lc, "up_proj"))
            W_d = {n: we.dense_cols(mod_d, we._long([n])).float().cpu().numpy().astype(np.float64)[0] for n in range(8)}
            W_u = {n: we.dense_rows(mod_u, we._long([n])).float().cpu().numpy().astype(np.float64)[0] for n in range(8)}
            Wc2 = np.stack([W_d[n] for n in idx_c])
            kw_e = dict(model_id=h.model_id, weights_id=h.weights_id)

            def _fold(edit: Any, start: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
                out = {k: v.copy() for k, v in start.items()}
                for t in edit.terms:
                    if t.kind not in ("cols", "rows"):
                        continue
                    for j, n in enumerate(t.idx.tolist()):
                        key = (t.linear, int(n))
                        code = int(t.op[j])
                        if code == 0:
                            out[key] = out[key] * float(t.factor[j])
                        elif code == 1:
                            out[key] = t.vectors[j].astype(np.float64)
                        else:
                            out[key] = out[key] + t.vectors[j].astype(np.float64)
                return out

            name_d, name_u = we.linear_name(Lc, "down_proj"), we.linear_name(Lc, "up_proj")
            base_sl = {**{(name_d, n): W_d[n] for n in range(8)}, **{(name_u, n): W_u[n] for n in range(8)}}
            c_plants = {"silence": [we.silence(Lc, idx_c)], "scale x0.5": [we.scale(Lc, idx_c, 0.5)],
                        "add 0.5W": [we.add_to_columns(Lc, idx_c, (0.5 * Wc2).astype(np.float32))],
                        "set 0.3W": [we.set_columns(Lc, idx_c, (0.3 * Wc2).astype(np.float32))]}
            c_repairs = {"scale x2": [we.scale(Lc, idx_c, 2.0)], "add W": [we.add_to_columns(Lc, idx_c, Wc2.astype(np.float32))],
                         "set W": [we.set_columns(Lc, idx_c, Wc2.astype(np.float32))],
                         "scale x2 elsewhere": [we.scale(Lc, [5, 6], 2.0)],
                         "up_proj rows x2": [we.scale_rows(Lc, "up_proj", idx_c, 2.0)]}
            table_bad: List[str] = []
            n_pairs = n_differ = 0
            for pn, ps in c_plants.items():
                pe = we.build(h.lm, ps, **kw_e)
                planted_sl = _fold(pe, base_sl)
                for rn, rs in c_repairs.items():
                    re_ = we.build(h.lm, rs, **kw_e)
                    st_e = we.stack(h.lm, [pe, re_], plant_first=True, **kw_e)
                    harness_sl, ckpt_sl = _fold(st_e, base_sl), _fold(re_, planted_sl)
                    same_at = {k: float(np.max(np.abs(harness_sl[k] - ckpt_sl[k]))) <= 1e-5 for k in base_sl}
                    cmp_ = lev.repair_composition(pe, re_, st_e)
                    n_pairs += 1
                    n_differ += 0 if all(same_at.values()) else 1
                    per_ok = all(s["same_as_checkpoint"] == same_at[(s["linear"], s["index"])] for s in cmp_["shared"])
                    if not per_ok or cmp_["recovery_holds_on_checkpoint"] is not all(same_at.values()):
                        table_bad.append(f"{pn} + {rn}: table says {cmp_['recovery_holds_on_checkpoint']}, "
                                         f"stack says {all(same_at.values())}")
            lr_out, lr_in = (int(x) for x in we.linear_shape(mod_d))  # down_proj: [hidden, ffn width]
            lr_e = we.build(h.lm, [we.lowrank(Lc, np.full((1, lr_in), 1e-4, np.float32), np.full((lr_out, 1), 1e-4, np.float32))], **kw_e)
            lr_cmp = lev.repair_composition(we.build(h.lm, c_plants["silence"], **kw_e), lr_e)
            check(n_pairs == 20 and not table_bad and n_differ >= 6 and lr_cmp["recovery_holds_on_checkpoint"] is None
                  and "not known" in lr_cmp["why"],
                  f"6: repair_composition's table agrees with weight_edits.stack folded on the stand-in for {n_pairs} plant "
                  f"and repair pairs ({n_differ} compose otherwise on a checkpoint), and says a low-rank repair is not "
                  f"known ({table_bad[:2]})")

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
            check(sorted(leak_summary["kind_or_id_words_shown"]) == ["canary_leak", "planted_leak"]
                  and all("planted" in w for w in leak_summary["kind_or_id_words_shown"].values())
                  and all(r["scored_as"] == "not blinded" for r in leak_summary["cases"]),
                  f"6: a kind word in what a driver is shown is caught case by case, and such a case is scored as not "
                  f"blinded ({leak_summary['kind_or_id_words_shown']})")
            # A canary whose driver failed abstained from nothing: as a study counts sessions, it did not run to an end
            # (at d5e6848 a driver that could propose and failed every door call scored as an honest abstention)
            def _tr_fail(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
                done = any(m.get("role") == "tool" for m in payload.get("messages") or [])
                msg = ({"role": "assistant", "content": "Done."} if done else
                       {"role": "assistant", "content": "", "tool_calls": [
                           {"id": "x1", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}}]})
                return {"model": "mock-fail", "usage": {"prompt_tokens": 40, "completion_tokens": 5}, "choices": [{"message": msg}]}

            real_how = lev._how_driver_proposes
            lev._how_driver_proposes = lambda case, served: (True, "as if it could propose")  # type: ignore[assignment]
            try:
                fail_summary = lev.run_blinded_benchmark(
                    h, [lev.BlindedCase("canary_failed", "canary", reach="activations", driver_family="openai",
                                        driver_model="mock-fail", driver_config={"transport_fn": _tr_fail})],
                    tmp / "blinded_failed", weights_dir=weights_dir, probe_sandbox=False,
                )
            finally:
                lev._how_driver_proposes = real_how  # type: ignore[assignment]
            fc = fail_summary["cases"][0]
            check(fc["scored_as"] == "did not run to an end" and fc["honest_abstention"] is None
                  and fail_summary["canary_abstention_rate"] is None and bool(fc["did_not_run_to_an_end"]),
                  f"6: a canary whose driver failed is not an honest abstention, even where it could have proposed "
                  f"({fc['scored_as']}: {str(fc['did_not_run_to_an_end'])[:80]})")
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
            # The site finder, both ways: each of these names a site and was missed at d5e6848, and each ordinary phrase
            # below was taken for one there; what it does not catch is said on the page (`site_finder`)
            site_misses = ["the 5313th neuron", "L50N5313", "neuron_5313", "neuron id 5313", "neurons {5313, 8801}",
                           "layer=50", "layers.50.mlp.down_proj", "#/atlas/54/15491", "#/neuron/54/15491"]
            site_plain = ["the L2 norm", "L1", "N95", "unit 1 of the course", "column 2 of the table"]
            missed = [s for s in site_misses if not lev.sites_named_in(f"See {s} here.")]
            flagged = [s for s in site_plain if lev.sites_named_in(f"See {s} here.")]
            check(not missed and not flagged and "does not catch" in sp_page["site_finder"]
                  and "fiftieth layer" in sp_page["site_finder"] and lev.sites_named_in("layer 50 and neuron 5313"),
                  f"7: the site finder catches sites named nine more ways and leaves L2 norms, N95, units of a course and "
                  f"columns of a table alone, and the page says what it does not catch (missed {missed}, flagged {flagged})")

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
            # It carries each item's id and the runtime the reading was read under, and says what a fresh process could
            # change (at d5e6848 it carried neither)
            rb_rt = rb_spec.get("runtime") or {}
            check([it.get("id") for it in rb_spec["items"]] == [rb_items["rb0"].id, rb_items["rb1"].id]
                  and {"dtype", "unpacker", "threads", "torch", "numpy"} <= set(rb_rt)
                  and rb_rt["dtype"] == str(h.dtype) and "fresh process may differ" in rb_desc["description"]
                  and rb_desc["characters"] <= lev.MAX_REBUILD_CHARS,
                  f"7: the rebuild description carries every item's id and its reading's runtime ({sorted(rb_rt)}), and says a "
                  f"fresh process may differ in it ({rb_desc['word_count']} words, {rb_desc['characters']} characters)")
            syn_book = tmp / "rebuild_syn.jsonl"
            for syn_edit in ({"terms": [{"layer": 1, "silence": list(range(4000))}]},
                             {"terms": [{"layer": 1, "lowrank": {"A": "lr_A.npy", "B": "lr_B.npy"}}]}):
                lb.append("paired", {"status": "ok", "base": {"model": "m", "weights_id": None}, "plan": {"name": "s", "edit": syn_edit},
                                     "items": [{"item": {"name": "x", "question": "Q?", "answer": "A."}}]}, path=syn_book)
            # a 5,000-character run with no break, and a CJK sentence, each one word at d5e6848
            lb.append("paired", {"status": "ok", "base": {"model": "m", "weights_id": None},
                                 "plan": {"name": "s", "edit": {"terms": [{"layer": 1, "silence": [1]}]}},
                                 "items": [{"item": {"name": "x", "question": "x" * 5000, "answer": "A."}}]}, path=syn_book)
            cjk = "你好世界今天天气很好"
            check(refuses(lambda: lev.make_rebuild_description(syn_book, 0), lev.EvalRefused, "words (limit 200")
                  and refuses(lambda: lev.make_rebuild_description(syn_book, 1), lev.EvalRefused, "reads a file")
                  and refuses(lambda: lev.make_rebuild_description(syn_book, 2), lev.EvalRefused, "characters (limit")
                  and lev.description_words(json.dumps({"q": cjk}, ensure_ascii=True)) == 1 + len(cjk)
                  and refuses(lambda: lev.parse_rebuild_description("REBUILD_SPEC: " + json.dumps({"q": "y" * 5000})),
                              lev.EvalRefused, "characters (limit"),
                  "7: a 4,000-neuron recipe is counted as the thousands of words it is, a recipe that needs a file is "
                  "refused, a 5,000-character run is refused by the character bound, and each escaped CJK character is a word")
            rb_chk = lev.run_rebuild_replay(h, rb_desc["description"], rb_src_book, orig_seq, out_book=tmp / "rebuild_out.jsonl")
            check(rb_chk["ok"] and rb_chk["plan_same"] and rb_chk["items_compared"] == 2
                  and {"null:0.25", "null:-0.25", "edit:1.0", "control:0", "control:1"} <= set(rb_chk["arms_compared"])
                  and all(c["gap"] == 0.0 for c in rb_chk["comparisons"])
                  and rb_chk["items_as_described"] is True and rb_chk["runtime_differs"] == []
                  and "same-process replay" in rb_chk["what"] and "has not been run" in rb_chk["what"],
                  f"7: the same-process replay compares every arm of both items ({rb_chk['figures_compared']} figures, max gap "
                  f"{rb_chk['max_gap']}), on the items and the runtime described, and says the ruling's fresh-session test "
                  f"has not been run")
            tampered = rb_desc["description"].replace('"null_e":[0.25,-0.25]', '"null_e":[0.125,-0.125]')
            rb_bad = lev.run_rebuild_replay(h, tampered, rb_src_book, orig_seq, out_book=tmp / "rebuild_out_bad.jsonl")
            other_id = rb_desc["description"].replace(rb_items["rb1"].id, "0" * 64)
            check(tampered != rb_desc["description"] and rb_bad["ok"] is False and rb_bad["plan_same"] is False
                  and other_id != rb_desc["description"]
                  and refuses(lambda: lev.run_rebuild_replay(h, other_id, rb_src_book, orig_seq, out_book=tmp / "rebuild_out_id.jsonl"),
                              lev.EvalRefused, "would read other tokens"),
                  "7: a description whose null rows are not the reading's does not pass the replay, and one whose item id "
                  "is not what this process builds is refused before anything is read")

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
            # Each of these loaded the model before it was refused at d5e6848 (or was never refused): run with an unknown
            # brief, a missing brief file, a claim design with no control edits, a Claude Code effort it does not take;
            # knobs with a bars file that is not there, an unknown brief, --controls 0, and --stand-in into the default book
            good_knob_p = tmp / "cli_good_knob.json"
            good_knob_p.write_text(json.dumps({"steering": [{"kind": "neuron_mul", "layer": h.n_layers - 1,
                                                             "neurons": [6, 7], "factor": 0.0}]}), encoding="utf-8")
            pre_plans = {
                "unknown_brief": ({**ok_plan_doc, "brief_name": "no_such_brief"}, "Unknown brief_name"),
                "missing_brief_file": ({**ok_plan_doc, "brief_path": str(tmp / "no_such_brief.json")}, "Cannot read brief"),
                "no_controls": ({**ok_plan_doc, "claim_design": {"controls_override": 0}}, "controls_override is a whole number"),
                "claude_effort": ({**ok_plan_doc, "driver_budget": {"max_dollars": 1.0}, "arms": [
                    {"arm_id": "c0", "driver_family": "claude", "reach": "behaviour", "reasoning_effort": "extreme",
                     "driver_config": {"allow_mock_cli": True}}]}, "Claude Code's effort is one of"),
            }
            loads.clear()
            pre_said: Dict[str, str] = {}
            lmcp.hold_card = _no_load  # type: ignore[assignment]
            try:
                for tag, (doc, _w) in pre_plans.items():
                    pp_ = tmp / f"cli_pre_{tag}.json"
                    pp_.write_text(json.dumps(doc), encoding="utf-8")
                    pre_said[tag] = _cli_says(["run", str(pp_), "--out", str(tmp / f"cli_pre_out_{tag}")])
                kn_base = ["knobs", "--knob", str(good_knob_p), "--book", str(tmp / "cli_pre_kb.jsonl")]
                kn_bars0 = tmp / "cli_pre_bars.json"
                lcl.save_bars(bars, kn_bars0)
                pre_said["knobs_no_bars"] = _cli_says(kn_base + ["--brief", str(brief_json_p), "--bars", str(tmp / "no_bars.json")])
                pre_said["knobs_unknown_brief"] = _cli_says(kn_base + ["--brief", "no_such_brief", "--bars", str(kn_bars0)])
                pre_said["knobs_controls_0"] = _cli_says(kn_base + ["--brief", str(brief_json_p), "--bars", str(kn_bars0), "--controls", "0"])
                pre_said["knobs_default_book"] = _cli_says(["knobs", "--stand-in", "--knob", str(good_knob_p), "--brief", str(brief_json_p),
                                                            "--bars", str(kn_bars0), "--book", str(lb.default_book_path())])
            finally:
                lmcp.hold_card = real_hold  # type: ignore[assignment]
            pre_want = {**{t: w for t, (_d, w) in pre_plans.items()}, "knobs_no_bars": "Cannot read bar set",
                        "knobs_unknown_brief": "neither a brief file nor a starter brief", "knobs_controls_0": "--controls",
                        "knobs_default_book": "is the default book"}
            pre_bad = {t: s[:90] for t, s in pre_said.items() if not (s.startswith("refused:") and pre_want[t] in s)}
            p_cc0 = subprocess.run(
                [sys.executable, "-I", str(cli_path), "plan", "--claim-controls", "0", "--out", str(tmp / "cli_cc0_plan.json")],
                text=True, capture_output=True, timeout=30.0,
            )
            check(not loads and not pre_bad and p_cc0.returncode != 0 and "refused:" in p_cc0.stderr
                  and not (tmp / "cli_cc0_plan.json").exists() and not any((tmp / f"cli_pre_out_{t}").exists() for t in pre_plans),
                  f"7: run and knobs refuse an unknown or missing brief, a design with no control edits, an effort Claude "
                  f"Code does not take, a missing bars file, --controls 0 and a stand-in reading into the default book "
                  f"before loading anything, and plan --claim-controls 0 writes no plan (loads={len(loads)}; {pre_bad})")
            # knobs --stand-in reaches its claim: a knob claim is read, entered in its book, and nothing else is written
            # into the working folder or beside the book (at d5e6848 this check was refused before any claim work)
            kn_cwd = tmp / "cli_knob_cwd"
            kn_cwd.mkdir()
            kn_book_dir = tmp / "cli_knob_book"
            kn_book_dir.mkdir()
            kn_book = kn_book_dir / "book.jsonl"
            shutil.copy2(cal_book, kn_book)
            kn_bars = tmp / "cli_knob_bars.json"
            lcl.save_bars(bars, kn_bars)
            kn_before = int(lb.verify(kn_book)["entries"])
            p_kn = subprocess.run(
                [sys.executable, "-I", str(cli_path), "knobs", "--stand-in", "--brief", str(brief_json_p),
                 "--knob", str(good_knob_p), "--bars", str(kn_bars), "--book", str(kn_book), "--controls", "2"],
                text=True, capture_output=True, timeout=300.0, cwd=str(kn_cwd),
            )
            kn_after = lb.verify(kn_book)
            kn_regs = [e for e in lb.entries(kn_book, start=kn_before)
                       if (e.get("body") or {}).get("note_type") in ("knob_claim_registration", "knob_claim_verdict")]
            check(p_kn.returncode == 0 and "Verdict" in p_kn.stdout and kn_after["ok"] and len(kn_regs) == 2
                  and not list(kn_cwd.iterdir())
                  and not any(p.name.startswith("_stand_in") for p in kn_book_dir.iterdir()),
                  f"7: knobs --stand-in reads its knob claim into the book it names, and writes nothing into the working "
                  f"folder or beside its book (rc={p_kn.returncode}; {len(kn_regs)} claim lines; "
                  f"{(p_kn.stderr or p_kn.stdout).strip()[-80:]!r})")

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
