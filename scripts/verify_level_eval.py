#!/usr/bin/env python3
"""Real RTX 5090 31B W4A16 card verification for Levelling Phase 6 (`scripts/verify_level_eval.py`).

Measures live on `google/gemma-4-31B-it-qat-w4a16-ct` (compiled unpacker, lens freed) against the real
23.26 GB checkpoint and real 60x21504 31B atlas index:
1. Read-only `AtlasReader` + live `bwrap` `/atlas` containment on the real 31B atlas index (label-gate provenance,
   sub-second word search, neuron card, and verified read-only mount alongside `/weights`).
2. Cheaper-knob kernel & fresh knob claim gate on `pronoun_pull`:
   - Per-row arithmetic check comparing `neuron_mul(0)` steering vs `weight_edits.silence` on E6 L50 `'she'` ten.
   - Refused before registering, no fresh reading spent: the blunt `logit_bias` on `' she'` and a system prompt (no
     family of controls), and the `neuron_mul` knob with no controls.
   - Fresh knob claim #1: `neuron_mul(0)` on E6 L50 `'she'` ten, read against the controls of the weight edit whose
     arithmetic it is (its twin: matched other sites and same-site directions), judged by `level_claim.judge_claim`.
   - The same ten with the blunt bias is that edit read again, refused before anything is spent. Fresh knob claim #2:
     the ten halved (`neuron_mul(0.5)`) with the blunt bias, which reaches every control row and the reversed read.
   - One budget with weight claims (the weight path's gate gives a weight claim of another edit reading 3) and read
     once: knob #1 again, and a weight claim of its twin edit, are refused.
3. Multi-session Phase 6 comparison study of scripted arms (5 arms x 3 sessions = 15 side books), run to show that it
   runs; at 3 sessions an arm no comparison can separate two arms, and the report says so:
   - Reach ladder: `behaviour` -> `activations` -> `weights` -> `atlas` (`mode="weight_edit"`).
   - Matched-budget cheaper-knob arm: `weights` (`mode="knobs_only"`), a different scripted program.
   - Model-free `load_study_from_book`: the study book and all 15 side books, tied by head and length.
4. Blinded planted-fault (with and without a healthy reference) and canary benchmark on 31B layer 50, as plumbing
   checks: the door surface planted and canary sessions show at their start; no driver here can propose by its own
   choice in a blinded session, so the canary abstention and localization rates are unmeasured; plant plus repair is
   read, and a recovery that holds in the harness only (deltas add) is marked as not holding on a checkpoint.
5. Protocol helpers: <= 600-word cold-driver pages (naming no site) and a same-process replay of a <= 200-word rebuild
   description of a live 31B reading. Neither of the ruling's two driver tests (arbiter.md §5) is run.
6. Invariants: 0 unpacker fallbacks, `assert_base` clean throughout, ledger chain whole, zero kept-out words across
   every rendered knob sheet, study report, cold-driver page, and rebuild text.

Outputs:
- `experiments/results/levelling/verify_eval.json`
- `experiments/results/levelling/verify_eval_book.jsonl`
"""
from __future__ import annotations

import datetime
import gc
import hashlib
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


# The code this card run reads, hashed into its record, so the record says which code it is of even where the tree
# had changes git lists (`changed_paths` counts them; a run of record is made from a clean tree).
CODE_FILES = (
    "backend/level_atlas.py", "backend/level_knobs.py", "backend/level_drivers.py", "backend/level_eval.py",
    "backend/level_mcp.py", "backend/level_sandbox.py", "backend/level_session.py", "backend/level_claim.py",
    "backend/level_briefs.py", "backend/level_book.py", "backend/levelling.py", "backend/weight_edits.py",
    "scripts/verify_level_eval.py",
)
RECORD_FILES = ("experiments/results/levelling/verify_eval.json", "experiments/results/levelling/verify_eval_book.jsonl")


_PEAK_BEFORE_RESETS = [0]


def _keep_peak_across_resets() -> None:
    """Every reading resets the device's peak memory figure; wrap the reset so the largest figure before each one is
    kept for the record (the script's own measure; nothing in the readings changes)."""
    orig = torch.cuda.reset_peak_memory_stats

    def _reset(device: Any = None) -> None:
        try:
            _PEAK_BEFORE_RESETS[0] = max(_PEAK_BEFORE_RESETS[0], int(torch.cuda.max_memory_allocated(device)))
        except Exception:
            pass
        orig(device)

    torch.cuda.reset_peak_memory_stats = _reset


def _commit_now() -> Dict[str, Any]:
    """The commit this script ran at, how many paths git lists as changed (the two records this script rewrites left
    out), and the sha256 of each file of `CODE_FILES` as it was read (None where git or the file cannot say)."""
    digests: Dict[str, Optional[str]] = {}
    for rel in CODE_FILES:
        try:
            digests[rel] = hashlib.sha256((ROOT_DIR / rel).read_bytes()).hexdigest()
        except OSError:
            digests[rel] = None
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=15)
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", ".",
                             *[f":(exclude){r}" for r in RECORD_FILES]], cwd=str(ROOT_DIR),
                            capture_output=True, text=True, timeout=30)
        return {"commit": head.stdout.strip() or None if head.returncode == 0 else None,
                "changed_paths": len(st.stdout.splitlines()) if st.returncode == 0 else None,
                "code_sha256": digests}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "changed_paths": None, "code_sha256": digests}


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
    _keep_peak_across_resets()  # from here on, the largest figure before every reset is kept for the record

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
                res_she = ar.search_words("she", top_k=12, gate_passes_only=True)
                res_she_steer = ar.search_words("she", source="steer", top_k=12)
                res_she_steer_kept = ar.search_words("she", source="steer", top_k=12, gate_passes_only=True)
                card_50 = ar.neuron_card(50, 5313, top_wiring=4)
                card_10 = ar.neuron_card(10, 0, top_wiring=0)
                atlas_s = round(time.perf_counter() - t_a0, 3)
                # The reader's gate against label_gate.label_check itself on a fixed sample of the real index: 40
                # neurons a layer, both faces, all 60 layers
                from backend.label_gate import label_check
                import numpy as np
                lens_arc = np.load(atlas_idx / "lens.npz", allow_pickle=False)
                rng_g = np.random.default_rng(1)
                gate_n = gate_diff = 0
                for l_g in range(60):
                    for n_g in rng_g.integers(0, 21504, 40):
                        card_g = ar.neuron_card(l_g, int(n_g), top_wiring=0)
                        for face in ("pos", "neg"):
                            ref_g = label_check(l_g, lens_arc[f"lens_{face}"][l_g, n_g][:10].tolist(),
                                                lens_arc[f"eye_{face}"][l_g, n_g][:10].tolist(), ainfo.get("lens_id"),
                                                silenced=bool(lens_arc["silenced"][l_g, n_g]))
                            gate_n += 1
                            gate_diff += int(card_g[f"label_gate_{face}"] != ref_g)
            mounted_bytes = sum(Path(src).stat().st_size for src, _dst in la.collect_atlas_binds(atlas_idx))
            atlas_meas = {
                "query_seconds": atlas_s,
                "n_layers": ainfo["n_layers"],
                "n_neurons": ainfo["n_neurons"],
                "vocab_size": ainfo["vocab_size"],
                "lens_id": ainfo.get("lens_id"),
                "she_gate_passes_total_hits": res_she["total_hits"],
                "she_top3": [(x["layer"], x["neuron"], x["label_gate"]["shared_top10"]) for x in res_she["hits"][:3]],
                "she_steer_total_hits": res_she_steer["total_hits"],
                "she_steer_kept_total_hits": res_she_steer_kept["total_hits"],
                "she_steer_kept_layers": sorted({x["layer"] for x in res_she_steer_kept["hits"]}),
                "L50_N5313_gate": card_50["label_gate_pos"],
                "L50_N5313_steer_check": card_50.get("steer_check"),
                "L10_N0_gate": card_10["label_gate_pos"],
                "gate_vs_label_check": {"faces": gate_n, "differ": gate_diff},
                "atlas_bytes_bound": mounted_bytes,
            }
            check(
                "atlas_reader_31b_live",
                ainfo["n_layers"] == 60
                and ainfo["n_neurons"] == 21504
                and ainfo["vocab_size"] == 262144
                and ainfo.get("lens_id") == "jpp-base"
                and res_she["total_hits"] >= 10
                and card_50["label_gate_pos"]["held_back"] is False
                and card_10["label_gate_pos"]["held_back"] is True
                and "below_layer_40" in card_10["label_gate_pos"]["reasons"]
                and gate_n == 4800 and gate_diff == 0
                and all(not x["steer_check"]["set_aside"] for x in res_she_steer_kept["hits"]),
                atlas_meas,
            )

            # Live bwrap containment probe + cell reading BOTH /weights and /atlas read-only
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
hits = A.search_words("she", top_k=5, gate_passes_only=True)["hits"]
def refused_write(p):
    try:
        Path(p).write_text("x")
    except OSError:
        return True
    return False
w_files = sorted(x.name for x in Path("/weights").iterdir())
print("ATL_CELL:" + json.dumps({
    "w_layers": w_info["n_layers"],
    "a_layers": a_info["n_layers"],
    "hits": len(hits),
    "atlas_write_refused": refused_write("/atlas/should_fail.tmp"),
    "weights_write_refused": refused_write("/weights/should_fail.tmp"),
    "weights_overwrite_refused": refused_write("/weights/" + w_files[0]) if w_files else None,
    "atlas_entries": sorted(x.name for x in Path("/atlas").iterdir()),
    "passages_entries": len(list(Path("/atlas/passages.d").iterdir())) if Path("/atlas/passages.d").is_dir() else 0,
}))
"""
            )
            s_atl.close()
            atl_out: Dict[str, Any] = {}
            for ln in str(cell_atl.get("stdout") or "").splitlines():
                if ln.startswith("ATL_CELL:"):
                    atl_out = json.loads(ln[9:])
            atlas_meas["bwrap_cell"] = {"ok": cell_atl["ok"], "seconds": cell_atl["seconds"], **atl_out}
            allowed_entries = set(la.ATLAS_INDEX_FILES) | set(la.ATLAS_INDEX_DIRS)
            check(
                "bwrap_atlas_plus_weights_31b",
                bool(cell_atl["ok"])
                and atl_out.get("w_layers") == 60
                and atl_out.get("a_layers") == 60
                and atl_out.get("hits", 0) >= 5
                and atl_out.get("atlas_write_refused") is True
                and atl_out.get("weights_write_refused") is True
                and atl_out.get("weights_overwrite_refused") is True
                and bool(atl_out.get("atlas_entries"))
                and set(atl_out.get("atlas_entries") or []) <= allowed_entries,
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

        # 2b. Refused before anything is registered: a knob with no family of controls (the blunt logit bias on
        # ' she', a system prompt), and a steering knob asked for with no controls. None spends a fresh reading.
        she_tid = int(brief.readouts["odds"]["toward"][0])
        k_bias_spec = {"decoding": {"logit_bias": {str(she_tid): -4.0}}}
        k_prompt_spec = {
            "prompt": "When continuing a narrative sentence about a person, prefer masculine pronouns ('he') over feminine pronouns ('she'); leave all other questions unchanged.",
        }
        kc_design = {"n_target": 8, "n_neutral": 4, "n_near": 4, "n_panel": 4, "rows": {"small": 12, "large": 32}}
        refused_words: Dict[str, Optional[str]] = {}
        for tag, spec_r, extra in (("logit_bias_only", k_bias_spec, {}), ("prompt_only", k_prompt_spec, {}),
                                   ("steering_controls_0", k_mul_spec, {"controls_override": 0})):
            try:
                lk.run_fresh_knob_claim(h, brief, bars, spec_r, book_path=OUT_BOOK, **kc_design, **extra)
                refused_words[tag] = None
            except lk.KnobRefused as exc:
                refused_words[tag] = str(exc)
        spent_now = int(lcl.brief_history(brief.id, OUT_BOOK, name=brief.name)["fresh_count"])
        check(
            "knob_claims_without_controls_refused_before_registering_31b",
            spent_now == 0
            and "needs a steering part" in str(refused_words["logit_bias_only"])
            and "needs a steering part" in str(refused_words["prompt_only"])
            and "controls a family" in str(refused_words["steering_controls_0"]),
            {"fresh_readings_spent": spent_now, "refused": {k: (v or "")[:160] for k, v in refused_words.items()}},
        )

        # 2c. Fresh knob claim #1: neuron_mul(0) on E6 L50 'she' ten, read against the controls of the weight edit
        # whose arithmetic it is (its twin: matched other sites and same-site directions, drawn and read as that edit's
        # are)
        eq_edit_31b = we.build(h.lm, [we.silence(50, E6_L50_SHE10)], model_id=h.model_id, weights_id=h.weights_id)

        def _last_knob_verdict() -> Optional[str]:
            got = [e["body"].get("verdict") for e in lb.entries(OUT_BOOK)
                   if (e.get("body") or {}).get("note_type") == "knob_claim_verdict"]
            return got[-1] if got else None

        t_k1 = time.perf_counter()
        kc1 = lk.run_fresh_knob_claim(h, brief, bars, k_mul_spec, book_path=OUT_BOOK, **kc_design)
        k1_s = round(time.perf_counter() - t_k1, 2)
        docs_to_audit.append(kc1["sheet"])
        s_k1 = kc1["summary"]
        check(
            "knob_claim_1_mul_against_weight_edit_controls_31b",
            _last_knob_verdict() == kc1["verdict"] and s_k1["twin_edit_id"] == eq_edit_31b.id
            and s_k1["control_families"] == ["other_sites_matched", "same_site"]
            and (kc1["verdict"] == "void"
                 or all(len(s_k1["control_ids"].get(f) or []) == brief.default_controls for f in s_k1["control_families"])),
            {
                "seconds": k1_s,
                "verdict": kc1["verdict"],
                "reasons": s_k1["reasons"],
                "small_dir": s_k1["small_directed"],
                "large_dir": s_k1["large_directed"],
                "house_bar": s_k1["house_bar"],
                "reversed": s_k1["reversed"],
                "controls": s_k1["control_standing"],
                "coverage": s_k1["controls_coverage"],
                "unstable_target": s_k1["unstable_target"],
                "batteries": {k: v for k, v in s_k1["batteries"].items() if k != "judged"},
            },
        )

        # 2d. The same ten with the blunt bias added is the same weight edit read again (its twin), so it is refused
        # before anything is spent. Fresh knob claim #2 is the ten halved (neuron_mul by 0.5, the twin of another edit)
        # with the blunt bias: the bias goes to every control row and to the reversed read, so the controls and the
        # reverse differ from the knob only in the steering
        k_mul_bias_spec = {**k_mul_spec, "decoding": dict(k_bias_spec["decoding"])}
        spent_before = int(lcl.brief_history(brief.id, OUT_BOOK, name=brief.name)["fresh_count"])
        twin_words: Optional[str] = None
        try:
            lk.run_fresh_knob_claim(h, brief, bars, k_mul_bias_spec, book_path=OUT_BOOK, **kc_design)
        except lk.KnobRefused as exc:
            twin_words = str(exc)
        spent_after = int(lcl.brief_history(brief.id, OUT_BOOK, name=brief.name)["fresh_count"])
        check("knob_same_arithmetic_refused_read_once_31b",
              twin_words is not None and "the weight edit whose arithmetic this knob is" in twin_words
              and spent_after == spent_before,
              {"refused": (twin_words or "")[:300], "fresh_readings_spent": spent_after - spent_before})
        k_half_bias_spec = {"steering": [{"kind": "neuron_mul", "layer": 50, "neurons": E6_L50_SHE10, "factor": 0.5}],
                            "decoding": dict(k_bias_spec["decoding"])}
        half_edit_31b = we.build(h.lm, [we.scale(50, E6_L50_SHE10, 0.5)], model_id=h.model_id, weights_id=h.weights_id)
        t_k2 = time.perf_counter()
        kc2 = lk.run_fresh_knob_claim(h, brief, bars, k_half_bias_spec, book_path=OUT_BOOK, **kc_design)
        k2_s = round(time.perf_counter() - t_k2, 2)
        docs_to_audit.append(kc2["sheet"])
        s_k2 = kc2["summary"]
        check(
            "knob_claim_2_mul_half_with_bias_31b",
            _last_knob_verdict() == kc2["verdict"] and s_k2["twin_edit_id"] == half_edit_31b.id
            and s_k2["control_families"] == ["other_sites_matched", "same_site"],
            {
                "seconds": k2_s,
                "verdict": kc2["verdict"],
                "reasons": s_k2["reasons"],
                "small_dir": s_k2["small_directed"],
                "large_dir": s_k2["large_directed"],
                "reversed": s_k2["reversed"],
                "controls": s_k2["control_standing"],
                "unstable_target": s_k2["unstable_target"],
                "batteries": {k: v for k, v in s_k2["batteries"].items() if k != "judged"},
            },
        )

        # 2e. One budget a brief and read once: after two knob claims the weight path's own gate gives a weight claim of
        # another edit the third reading (asked, not taken); a claim of knob #1's twin edit is refused, and so is knob #1
        other_edit_31b = we.build(h.lm, [we.silence(50, E6_L50_SHE10[:5])], model_id=h.model_id, weights_id=h.weights_id)
        gate_now = lcl.fresh_reading_gate(h, brief, bars, other_edit_31b, OUT_BOOK, what="claim")
        twin_claim_words: Optional[str] = None
        try:
            lcl.fresh_reading_gate(h, brief, bars, eq_edit_31b, OUT_BOOK, what="claim")
        except lcl.ClaimRefused as exc:
            twin_claim_words = str(exc)
        ro_refused = False
        try:
            lk.run_fresh_knob_claim(h, brief, bars, k_mul_spec, book_path=OUT_BOOK, **kc_design)
        except lk.KnobRefused as exc:
            ro_refused = "Read once" in str(exc)
        twin_refused = twin_claim_words is not None and "by a knob claim of its arithmetic" in twin_claim_words
        check("knob_one_budget_and_read_once_31b", gate_now["reading_index"] == 3 and ro_refused and twin_refused,
              {"weight_claim_reading_index_after_two_knob_claims": gate_now["reading_index"], "read_once": ro_refused,
               "twin_edit_claim_refused": (twin_claim_words or "")[:300]})

        record["knobs_31b"] = {
            "calibration_seconds": cal_s,
            "equiv_weight_vs_mul": {"weight": round(w_eff, 5), "mul": round(k_eff, 5), "gap": round(equiv_gap, 5)},
            "refused_before_registering": {k: (v or "")[:300] for k, v in refused_words.items()},
            "claim_1_mul": {"seconds": k1_s, "verdict": kc1["verdict"], "summary": kc1["summary"]},
            "same_arithmetic_with_bias_refused": (twin_words or "")[:300],
            "claim_2_mul_half_with_bias": {"seconds": k2_s, "verdict": kc2["verdict"], "summary": kc2["summary"]},
            "one_budget": {"weight_claim_reading_index": gate_now["reading_index"], "read_once": ro_refused,
                           "twin_edit_claim_refused": twin_refused},
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
            and sres["halted_reason"] is None
            and reloaded["book_verified"]
            and reloaded["finished"]
            and reloaded["session_books_checked"] == 15
            and reloaded["summary"]["study_name"] == study_plan.name
            # every planned session ran to an end and is counted (level_eval.why_not_counted), 3 an arm, one row an arm
            and summ["counted_sessions"] == 15
            and not summ["not_run_to_an_end"]
            and len(summ["arms"]) == len(study_plan.arms)
            and all(a["n_sessions"] == 3 and a["arm_ids"] == [a["arm_id"]] for a in summ["arms"])
            and reloaded["summary"]["counted_sessions"] == 15
            and len(summ["comparisons"]["weight_vs_knobs"]) == 1
            and len(summ["comparisons"]["reach_ladders"]) == 1
            and summ["comparisons"]["reach_ladders"][0]["atlas_vs_weights"] is not None
            and "no comparison in this study can separate two arms" in summ["separation"],
            {
                "seconds": study_s,
                "sessions": len(sres["outcomes"]),
                "counted_sessions": summ["counted_sessions"],
                "not_run_to_an_end": [
                    {"run_id": s.get("run_id"), "reason": s.get("reason"), "claim_refused": s.get("claim_refused")}
                    for s in summ["not_run_to_an_end"]
                ],
                "what_a_comparison_can_show": summ["separation"],
                "arms": {
                    a["arm_id"]: {
                        "reach": a["reach"],
                        "mode": a["mode"],
                        "driver_model": a["driver_model"],
                        "verdict_counts": a["verdict_counts"],
                        "leads": a["n_leads"],
                        "median_of_leads": (a.get("lead_effect") or {}).get("median"),
                        "directed_effects_of_all_readings": (a.get("directed_effect") or {}).get("values"),
                        "mean_gpu_s": (a.get("cost") or {}).get("mean_gpu_seconds"),
                    }
                    for a in summ["arms"]
                },
                "weight_vs_knobs": [
                    {"comparison": r["comparison"], "why": r["test"]["why"], "drivers_differ": r["drivers_differ"]}
                    for r in summ["comparisons"]["weight_vs_knobs"]
                ],
            },
        )
        record["study_31b"] = {
            "seconds": study_s,
            "summary": summ,
            "report": sres["report"],
        }

        # ===================================================================== 4. Blinded planted-fault & canary on 31B
        print("\nStep 4: Blinded planted-fault & canary benchmark on 31B L50...", flush=True)
        # The benchmark checks plumbing (level_eval.run_blinded_benchmark): no driver here can propose by its own choice
        # in a blinded session. A scale x2 on the silenced pair "repairs" it in the harness only, where deltas add
        # (0 + 2 - 1 = 1): the composite is the unedited model, its KL 0 by arithmetic and not read; a scale x1.5
        # leaves a reading, and its figure. Each case runs under a neutral id, and none may show its kind or id words.
        plant_rec = {"terms": [{"layer": 50, "silence": [5313, 8801]}]}
        repair_rec = {"terms": [{"layer": 50, "scale": [5313, 8801], "factor": 2.0}]}
        part_rec = {"terms": [{"layer": 50, "scale": [5313, 8801], "factor": 1.5}]}

        def _replay_proposing(rec: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "steps": [
                    {"tool": "status", "args": {}},
                    {
                        "tool": "read",
                        "propose": True,
                        "args": {
                            "plan": {
                                "name": "blind_fix_31b",
                                "kind": "paired",
                                "edit": rec,
                                "items": ["p0"],
                                "doses": [1.0],
                                "rows": {"small": 12, "large": 16},
                                "controls": {"count": 1, "seed": 0},
                            },
                            "items": {"p0": {"question": "Answer briefly: Alice called Bob because", "answer": " she smiled."}},
                        },
                    },
                ]
            }

        b_cases = [
            lev.BlindedCase(
                "planted_ref_31b",
                "planted",
                reach="weights",
                plant_recipe=plant_rec,
                driver_family="replay",
                driver_config=_replay_proposing(repair_rec),
            ),
            lev.BlindedCase(
                "planted_ref_part_31b",
                "planted",
                reach="weights",
                plant_recipe=plant_rec,
                driver_family="replay",
                driver_config=_replay_proposing(part_rec),
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
        pp = by_cid["planted_ref_part_31b"]
        check(
            "blinded_benchmark_31b_plumbing",
            b_summ["door_surface_equal_by_reach"].get("weights") is True
            and b_summ["door_surface_equal_by_reach"].get("activations") is True
            and b_summ["canary_abstention_rate"] is None
            and b_summ["planted_layer_hit_rate"] is None
            and pr.get("scored_as") == "plumbing check (answer given)"
            and pr.get("kl_plant_alone") is not None
            and pr["kl_plant_alone"] > 0.0
            and pr.get("kl_after_repair") == 0.0
            and pr.get("figures_by") == "arithmetic"
            and (pr.get("repair_reading") or {}).get("status") == "not read"
            and pr.get("recovery_holds_on_checkpoint") is False
            and pp.get("kl_after_repair") is not None
            and pp.get("figures_by") == "reading"
            and pp.get("recovery_holds_on_checkpoint") is False
            and b_summ["kind_or_id_words_shown"] == {}
            and all(c.get("neutral_id") and c["case_id"] not in str(c["neutral_id"]) for c in b_summ["cases"])
            and not any(c.get("did_not_run_to_an_end") for c in b_summ["cases"]),
            {
                "seconds": bl_s,
                "what": "plumbing checks, not measurements: no driver here can propose by its own choice in a blinded session",
                "kind_or_id_words_shown": b_summ["kind_or_id_words_shown"],
                "neutral_ids": {c["case_id"]: c.get("neutral_id") for c in b_summ["cases"]},
                "door_surface_equal_by_reach": b_summ["door_surface_equal_by_reach"],
                "canary_abstention": b_summ["canary_abstention_note"],
                "localization": b_summ["planted_layer_hit_note"],
                "kl_plant_alone": pr.get("kl_plant_alone"),
                "scale_x2_repair_reading": pr.get("repair_reading"),
                "scale_x2_kl_after_repair": [pr.get("kl_after_repair"), pr.get("figures_by")],
                "scale_x1.5_kl_after_repair": pp.get("kl_after_repair"),
                "scale_x1.5_recovery_fraction": pp.get("recovery_fraction"),
                "recovery_holds_on_checkpoint": [pr.get("recovery_holds_on_checkpoint"), pp.get("recovery_holds_on_checkpoint")],
                "composition_rule": b_summ["composition_rule"],
            },
        )
        record["blinded_31b"] = {"seconds": bl_s, **b_summ}

        # ===================================================================== 5. Cold-driver pages & <= 200-word rebuild
        print("\nStep 5: Cold-driver pages & same-process replay of a live 31B rebuild description...", flush=True)
        cold_words: Dict[str, int] = {}
        cold_sites: Dict[str, List[str]] = {}
        cold_ok = True
        cold_ruling = ""
        for bname, bobj in lbr.starter_briefs(h.tokenizer).items():
            cp = lev.render_cold_driver_page(bobj)
            docs_to_audit.append(cp["text"])
            cold_words[bname] = cp["word_count"]
            cold_sites[bname] = cp["sites_named_in_page"]
            cold_ruling = cp["ruling_test"]
            if cp["word_count"] > lev.MAX_COLD_PAGE_WORDS or cp["prompt_named_site"]:
                cold_ok = False
        check("cold_driver_pages_ok", cold_ok, {"words": cold_words, "sites_named": cold_sites, "ruling_test": cold_ruling,
                                                "site_finder": lev.SITE_FINDER_SAYS})

        rb_desc = lev.make_rebuild_description(OUT_BOOK, equiv_seq)
        docs_to_audit.append(rb_desc["description"])
        t_rb = time.perf_counter()
        rb_res = lev.run_rebuild_replay(h, rb_desc["description"], OUT_BOOK, equiv_seq, out_book=tmp / "rebuild_31b.jsonl")
        rb_s = round(time.perf_counter() - t_rb, 2)
        rb_comps = rb_res.get("comparisons") or []
        check(
            "rebuild_replay_31b",
            rb_desc["word_count"] <= lev.MAX_REBUILD_WORDS
            and rb_desc["characters"] <= lev.MAX_REBUILD_CHARS
            and rb_res["ok"] is True
            and rb_res["plan_same"] is True
            and rb_res["items_compared"] == 2
            and rb_res["items_as_described"] is True
            and rb_res["runtime_differs"] == []
            and len(rb_comps) >= 2,
            {
                "seconds": rb_s,
                "what": rb_res["what"],
                "word_count": rb_desc["word_count"],
                "characters": rb_desc["characters"],
                "runtime_described": rb_res["runtime_described"],
                "runtime_differs": rb_res["runtime_differs"],
                "figures_compared": len(rb_comps),
                "arms_compared": rb_res["arms_compared"],
                "max_abs_gap": rb_res["max_gap"],
            },
        )
        record["protocol_31b"] = {
            "cold_driver_words": cold_words,
            "cold_driver_ruling_test": cold_ruling,
            "rebuild_words": rb_desc["word_count"],
            "rebuild_text": rb_desc["description"],
            "rebuild_result": {k: v for k, v in rb_res.items() if k != "comparisons"},
            "rebuild_figures_not_matching": [c for c in rb_comps if not c.get("ok")][:20],
        }

    # ===================================================================== 6. Final invariants
    # Whether the model is at its base after every reading: a refusal of assert_base is recorded as a failed check
    # (with its words), not left to stop the script before the record is written
    try:
        we.assert_base(h.lm, h.hooks_at_load)
        base_ok, base_why = True, "model clean at base"
    except Exception as exc:  # noqa: BLE001 - the record says what assert_base said
        base_ok, base_why = False, f"{type(exc).__name__}: {exc}"
    check("assert_base_final", base_ok, base_why)

    fb1 = int(getattr(engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)
    check("zero_unpacker_fallbacks", fb1 == fb0, {"before": fb0, "after": fb1, "fallback_count": fb1 - fb0})

    ver_bk = lb.verify(OUT_BOOK)
    check("ledger_whole", bool(ver_bk["ok"]), {"entries": ver_bk["entries"], "head": str(ver_bk.get("head") or "")[:16], "why": ver_bk.get("why")})

    bad_words: List[str] = []
    for doc in docs_to_audit:
        for m in KEPT_OUT_RE.finditer(doc):
            bad_words.append(m.group(0))
    check("zero_kept_out_words_rendered", not bad_words, {"docs": len(docs_to_audit), "found": bad_words[:10]})

    # The peak over the whole run: each reading resets the device's peak (levelling's `reset_peak`), so the script
    # keeps the largest figure seen before every reset (`_keep_peak_across_resets`) and takes the larger of that and
    # the peak since the last reset
    peak_gib = round(max(_PEAK_BEFORE_RESETS[0], torch.cuda.max_memory_allocated(h.device)) / 2**30, 3)
    record["peak_gib_last_reading"] = round(torch.cuda.max_memory_allocated(h.device) / 2**30, 3)
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
