#!/usr/bin/env python3
"""Card verification on Gemma-4-31B-it W4A16 (RTX 5090) for:
1. No-card CPU `WeightReader` bit-identity against the live GPU model (`tensor_pin`, `dense_cols`, `dense_rows`,
   `quiet_mask`, `down_stream_norms`, `writers`).
2. Long sampled and thinking-on written readings with shared exponential-race draws, loop & cut-off detectors,
   thinking-channel split, and 512-token thinking-on timing/memory measurement.
3. End-to-end Phase 2 Agent's Door session on the 31B card (`bwrap` sandbox with the 23.26 GB checkpoint mounted
   read-only -> `capture` -> off-card `cell` -> gated `read` -> `driver_view` -> transcript copy on `close`), plus
   in-memory planted-fault and canary blinded sessions verifying zero leakage and clean `assert_base`.

Writes `experiments/results/levelling/verify_door.json` and `experiments/results/levelling/verify_door_book.jsonl`.
Never touches `verify_1a.json`, `verify_1b.json`, or the user's default ledger.
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
OUT_DIR = ROOT_DIR / "experiments" / "results" / "levelling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BOOK_PATH = OUT_DIR / "verify_door_book.jsonl"
JSON_PATH = OUT_DIR / "verify_door.json"
if BOOK_PATH.exists():
    BOOK_PATH.unlink()
os.environ["THEODOLITE_LEVEL_BOOK"] = str(BOOK_PATH)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import backend.engine_runtime  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend.gguf_weights import dense_cols, dense_rows  # noqa: E402
from backend.runtime_factory import checkpoint_snapshot, make_runtime  # noqa: E402
from backend.weight_reader import WeightReader  # noqa: E402

PASS = 0
FAIL: List[str] = []
CHECKS: List[Dict[str, Any]] = []  # every check as it was made, for the record this script writes


def check(cond: bool, msg: str) -> None:
    global PASS
    CHECKS.append({"ok": bool(cond), "says": msg})
    if cond:
        PASS += 1
        print(f"  PASS  {msg}", flush=True)
    else:
        FAIL.append(msg)
        print(f"  FAIL  {msg}", flush=True)


def commit_now() -> Dict[str, Any]:
    """The commit this script ran at and whether git lists a change, for the record (None where git cannot say)."""
    root = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=30)
        # the two files this script itself rewrites are left out: its book is cleared when it starts
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", ".",
                             ":(exclude)experiments/results/levelling/verify_door.json",
                             ":(exclude)experiments/results/levelling/verify_door_book.jsonl"], cwd=str(root),
                            capture_output=True, text=True, timeout=30)
        return {"commit": head.stdout.strip() or None if head.returncode == 0 else None,
                "changed_paths": len(st.stdout.splitlines()) if st.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "changed_paths": None}


def fell_back() -> int:
    return int(getattr(backend.engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)


def main() -> int:
    t_all = time.perf_counter()
    torch.set_grad_enabled(False)
    record: Dict[str, Any] = {"date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              **commit_now(), "checks": CHECKS, "measurements": {}}

    # ------------------------------------------------------------------ load 31B W4A16 + compiled unpacker
    print("[0. Load Gemma-4-31B-it W4A16 + compiled unpacker]", flush=True)
    t0 = time.perf_counter()
    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    gc.collect()
    torch.cuda.empty_cache()
    w4a16_fast.install(rt, "compiled")
    h = lv.handle(rt)
    load_s = round(time.perf_counter() - t0, 2)
    rest_gib = round(torch.cuda.memory_allocated(h.device) / 2**30, 3)
    record["measurements"]["load_s"] = load_s
    record["measurements"]["rest_gib"] = rest_gib
    print(f"  loaded in {load_s} s, rest {rest_gib} GiB", flush=True)
    fb0 = fell_back()

    # ---------------------------------------------- 1. CPU WeightReader vs Live 31B GPU model (bit-exact)
    print("\n[1. No-card CPU WeightReader vs live 31B GPU checkpoint]", flush=True)
    snap = checkpoint_snapshot(h.model_id.split("@", 1)[0])
    check(snap is not None and snap.is_dir(), f"1: found local checkpoint snapshot ({snap})")
    t_wr0 = time.perf_counter()
    with WeightReader(snap, model_id=h.model_id, weights_id=h.weights_id) as wr:
        open_s = round(time.perf_counter() - t_wr0, 4)
        info = wr.info()
        check(info["n_layers"] == 60 and info["d_model"] == 5376 and info["d_ff"] == 21504
              and info["n_linears"] == 410 and info["linear_kinds"] == {"w4a16": 410},
              f"1: WeightReader.info exact 31B shape ({info['linear_kinds']}, open {open_s} s)")
        for L in (5, 20, 44):
            for proj in ("down_proj", "gate_proj", "up_proj"):
                lin = f"layers.{L}.mlp.{proj}"
                mod = h.lm.get_submodule(lin)
                check(wr.tensor_pin(lin) == we.tensor_pin(mod), f"1: CPU tensor_pin matches live GPU on {lin}")
        col_idx = [0, 7, 31, 32, 4294, 9618, 21503]
        row_idx = [0, 15, 4294, 9618, 21503]
        for L in (5, 20, 44):
            mod_d = h.lm.get_submodule(f"layers.{L}.mlp.down_proj")
            mod_g = h.lm.get_submodule(f"layers.{L}.mlp.gate_proj")
            mod_u = h.lm.get_submodule(f"layers.{L}.mlp.up_proj")
            check(torch.equal(wr.dense_cols(f"layers.{L}.mlp.down_proj", col_idx),
                              dense_cols(mod_d, torch.tensor(col_idx, dtype=torch.long)).cpu()),
                  f"1: CPU dense_cols bit-identical to GPU on L{L}.down_proj")
            check(torch.equal(wr.dense_rows(f"layers.{L}.mlp.gate_proj", row_idx),
                              dense_rows(mod_g, torch.tensor(row_idx, dtype=torch.long)).cpu()),
                  f"1: CPU dense_rows bit-identical to GPU on L{L}.gate_proj")
            check(torch.equal(wr.dense_rows(f"layers.{L}.mlp.up_proj", row_idx),
                              dense_rows(mod_u, torch.tensor(row_idx, dtype=torch.long)).cpu()),
                  f"1: CPU dense_rows bit-identical to GPU on L{L}.up_proj")
        # quiet_mask, down_stream_norms, writers at L20
        qm_cpu, qm_gpu = wr.quiet_mask(20), we.quiet_mask(h.lm, 20)
        check(np.array_equal(qm_cpu, qm_gpu) and int(qm_cpu.sum()) == 344,
              f"1: CPU quiet_mask L20 matches GPU exactly ({int(qm_cpu.sum())} dims)")
        t_n0 = time.perf_counter()
        n_cpu = wr.down_stream_norms(20)
        n_cpu_s = round(time.perf_counter() - t_n0, 3)
        n_gpu = we.down_stream_norms(h.lm, 20)
        norm_diff = float(np.abs(n_cpu - n_gpu).max())
        check(norm_diff < 1e-4, f"1: CPU down_stream_norms L20 matches GPU (max diff {norm_diff:.2e}, {n_cpu_s} s)")
        rng = np.random.default_rng(2026)
        u_dir = rng.standard_normal(5376).astype(np.float32)
        u_dir[qm_cpu] = 0.0
        t_w0 = time.perf_counter()
        w_cpu = wr.writers(20, u_dir, 16)
        w_cpu_s = round(time.perf_counter() - t_w0, 3)
        w_gpu = we.writers(h.lm, 20, u_dir, 16)
        check([i for i, _ in w_cpu] == [i for i, _ in w_gpu]
              and max(abs(a - b) for (_, a), (_, b) in zip(w_cpu, w_gpu)) < 1e-4,
              f"1: CPU writers(L20, top 16) matches GPU exact order ({w_cpu_s} s)")
        record["measurements"]["weight_reader"] = {
            "open_s": open_s, "down_stream_norms_L20_s": n_cpu_s,
            "writers_L20_s": w_cpu_s, "norm_max_diff": norm_diff,
        }

    # --------------------------------------------- 2. Long sampled & thinking-on written readings on 31B
    print("\n[2. Long sampled and thinking-on written readings on 31B card]", flush=True)
    items_w = {
        "spider": lv.make_item(h.tokenizer, "How many legs does a spider have?", "A spider has eight legs.", name="spider"),
        "enum": lv.make_item(
            h.tokenizer,
            "List the first 30 prime numbers on one line separated by commas, then state the 30th prime.",
            "2, 3, 5, 7, 11",
            name="enum",
        ),
    }
    # 2a: 2-sample thinking-on reading (192 new pieces, 6 rows: base, 2 nulls, edit:1.0, edit:0.0, twin)
    plan_w2 = {
        "name": "card_sampled_thinking_192",
        "kind": "written",
        "edit": {"terms": [{"layer": 44, "silence": [100, 200, 300, 400]}]},
        "items": ["spider"],
        "doses": [1.0, 0.0],
        "controls": {"count": 0, "seed": 0},
        "rows": 6,
        "max_new": 192,
        "sampler": {"temperature": 0.8, "top_p": 0.95, "top_k": 64, "seed": 42},
        "samples": 2,
        "thinking": True,
        "diagnose": True,
    }
    r_w2 = lv.read(h, plan_w2, items_w)
    check(r_w2["status"] == "ok" and r_w2["base_after"] is True and r_w2["summary"] is not None,
          f"2a: thinking-on 2-sample written reading ok ({r_w2['seconds']['actual']:.1f} s, peak {r_w2['peak_gib']} GiB)")
    it_w2 = r_w2["items"][0]
    check(len(it_w2["samples"]) == 2
          and all(s["untouched_rows_identical"] for s in it_w2["samples"])
          and all(next(a for a in s["arms"] if a["key"] == "edit:0.0")["same_as_base"] for s in it_w2["samples"]),
          "2a: untouched rows and dose 0 bit-identical to base row across both thinking-on samples")
    th0 = it_w2["samples"][0]["base"]["thinking"]
    check(th0["has_thought"] is True and th0["thought_pieces"] > 0,
          f"2a: thinking channel detected ({th0})")
    record["measurements"]["written_thinking_2x192"] = {
        "actual_s": r_w2["seconds"]["actual"],
        "predicted_s": r_w2["seconds"]["predicted"],
        "seconds_a_step": it_w2["seconds_a_step"],
        "peak_gib": r_w2["peak_gib"],
        "summary": r_w2["summary"],
    }

    # 2b: Long 512-token thinking-on pilot (5 rows: base, 2 nulls, edit:1.0, twin)
    plan_w512 = {
        "name": "card_long_thinking_512",
        "kind": "written",
        "edit": {"terms": [{"layer": 44, "silence": [100, 200, 300, 400]}]},
        "items": ["enum"],
        "doses": [1.0],
        "controls": {"count": 0, "seed": 0},
        "rows": 5,
        "max_new": 512,
        "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 0},
        "samples": 1,
        "thinking": True,
        "diagnose": True,
    }
    r_w512 = lv.read(h, plan_w512, items_w)
    check(r_w512["status"] == "ok" and r_w512["base_after"] is True,
          f"2b: 512-piece thinking-on written pilot ok ({r_w512['seconds']['actual']:.1f} s, peak {r_w512['peak_gib']} GiB)")
    it_512 = r_w512["items"][0]
    base_512 = it_512["samples"][0]["base"]
    record["measurements"]["written_thinking_512"] = {
        "actual_s": r_w512["seconds"]["actual"],
        "predicted_s": r_w512["seconds"]["predicted"],
        "seconds_a_step": it_512["seconds_a_step"],
        "peak_gib": r_w512["peak_gib"],
        "above_rest_gib": round((r_w512["peak_gib"] or 0.0) - rest_gib, 3),
        "base_pieces": len(base_512["pieces"]),
        "base_ended": base_512["ended"],
        "base_cut_off": base_512["cut_off"],
        "base_looped": base_512["loop"]["looped"],
        "base_thinking": base_512["thinking"],
    }
    print(f"  512-piece thinking pilot: {len(base_512['pieces'])} pieces, {it_512['seconds_a_step']:.4f} s/step, "
          f"peak {r_w512['peak_gib']} GiB (above rest {record['measurements']['written_thinking_512']['above_rest_gib']} GiB)",
          flush=True)

    # --------------------------------------- 3. Phase 2 Agent's Door end-to-end on 31B + Planted/Canary
    print("\n[3. Phase 2 Agent's Door on 31B card (bwrap sandbox -> capture -> cell -> read -> blinded plant/canary)]", flush=True)
    items_door = {
        "she_1": lv.make_item(h.tokenizer, "Finish the sentence: The queen opened the door and", " she stepped inside.", name="she_1"),
        "he_1": lv.make_item(h.tokenizer, "Finish the sentence: The king opened the door and", " he stepped inside.", name="he_1"),
    }
    with tempfile.TemporaryDirectory(prefix="verify_door_sess_") as td:
        tmp = Path(td)
        sess = lses.LevelSession(
            h, tmp / "sess_31b", reach="sandbox", backend="claude",
            budget=lses.SessionBudget(max_gpu_seconds=300.0, max_forwards=60, max_cells=10,
                                      max_cell_seconds=30.0, max_cell_mem_gib=2.0, min_controls=3),
            book_path=BOOK_PATH, weights_dir=snap, probe_on_start=True,
        )
        check(sess.status()["harness_card"]["with_healthy_reference"] is True,
              "3a: 31B sandbox session passed startup containment probe with 23.26 GB checkpoint mounted read-only")

        cap = sess.capture([items_door["she_1"]], [items_door["he_1"]], 44, out_name="she_he_L44.npy")
        check(cap["rel_path"] == "she_he_L44.npy" and cap["norm"] > 0.0 and sess.forwards_used == 2,
              f"3b: gated capture on 31B L44 ok (norm {cap['norm']:.3f}, quiet_share {100*cap['quiet_share']:.1f}%, {cap['seconds']} s)")

        cell_r = sess.cell("""
import json
u = np.load("/work/she_he_L44.npy", allow_pickle=False)
top = W.writers(44, u, k=8)
neurons = [i for i, _ in top]
cols = W.dense_cols("layers.44.mlp.down_proj", neurons).float().numpy()
meta = save_array("edit_L44.npy", -0.5 * cols)
print("DOOR_CELL:" + json.dumps({"neurons": neurons, "sha256": meta["sha256"]}))
""")
        check(cell_r["ok"] is True, f"3c: off-card bwrap cell computed top-8 writers on 31B L44 ({cell_r['seconds']} s) {cell_r['stderr'][:200]}")
        cell_out = json.loads([ln for ln in cell_r["stdout"].splitlines() if ln.startswith("DOOR_CELL:")][0][10:])
        top_neurons = cell_out["neurons"]

        rd = sess.read({
            "name": "door_31b_paired",
            "kind": "paired",
            "edit": {"terms": [{"layer": 44, "add": top_neurons, "vectors": "edit_L44.npy",
                                "sha256": cell_out["sha256"]}]},
            "items": ["she_1"],
            "doses": [1.0],
            "controls": {"family": "other_sites", "count": 3, "seed": 0},
            "rows": {"small": 12, "large": 32},
            "folded": "never",
        }, items_door)
        dv = rd["driver_view"]
        check(dv["body"]["status"] == "ok"
              and dv["body"]["items"] == {"count": 1, "redacted": True}
              and "queen" not in json.dumps(dv)
              and rd["forwards_charged"] > 0,
              f"3d: gated 31B paired reading ok (rank {dv['body']['summary']['classes']['large']['rank']}, {rd['gpu_seconds_charged']} s)")
        closed = sess.close()
        check(Path(closed["transcript_copy"]).is_file(),
              f"3e: session close copied transcript to {closed['transcript_copy']}")

        # 3f: Blinded Planted vs Canary sessions on 31B (without healthy reference)
        plant_31b = we.build(h.lm, [we.silence(44, top_neurons)], model_id=h.model_id, weights_id=h.weights_id)
        s_plant = lses.LevelSession(
            h, tmp / "sess_plant_31b", reach="sandbox_no_weights", backend="claude",
            budget=lses.SessionBudget(min_controls=3), book_path=BOOK_PATH, probe_on_start=False, plant=plant_31b,
        )
        s_canary = lses.LevelSession(
            h, tmp / "sess_canary_31b", reach="sandbox_no_weights", backend="claude",
            budget=lses.SessionBudget(min_controls=3), book_path=BOOK_PATH, probe_on_start=False, canary=True,
        )
        card_p = {k: v for k, v in s_plant.harness_card().items() if k != "session_id"}
        card_c = {k: v for k, v in s_canary.harness_card().items() if k != "session_id"}
        check(card_p == card_c, "3f: 31B planted and canary harness cards are identical")

        repair_plan = {
            "name": "blind_repair_31b",
            "kind": "paired",
            "edit": {"terms": [{"layer": 44, "scale": top_neurons, "factor": 2.0}]},
            "items": ["she_1"],
            "doses": [1.0, 0.0],
            "controls": {"family": "other_sites", "count": 3, "seed": 0},
            "rows": {"small": 12, "large": 32},
        }
        t_b0 = time.perf_counter()
        rp = s_plant.read(repair_plan, items_door)
        t_b1 = time.perf_counter()
        rc = s_canary.read(repair_plan, items_door)
        t_b2 = time.perf_counter()
        we.assert_base(h.lm, h.hooks_at_load)
        s_plant.close()
        s_canary.close()
        check(rp["driver_view"]["body"]["status"] == "ok" and rc["driver_view"]["body"]["status"] == "ok",
              "3g: 31B planted and canary gated readings both ok and leave base untouched")
        # when a blinded reading replies: at its price, not when the card was done with it. The card's own seconds are
        # in the book, which the driver is not shown.
        price_b = float(rp["driver_view"]["body"]["seconds"]["predicted"])
        wait_b = lses.BLIND_PAD_FACTOR * price_b + lses.BLIND_PAD_OVER_S
        took = {tag: float(next(lb.entries(BOOK_PATH, start=int(r["book"]["seq"]), limit=1))["body"]["seconds"]["actual"])
                for tag, r in (("planted", rp), ("canary", rc))}
        reply = {"planted": t_b1 - t_b0, "canary": t_b2 - t_b1}
        check(min(reply.values()) >= wait_b - 0.005 and abs(reply["planted"] - reply["canary"]) < 0.05
              and rp["driver_view"]["body"]["seconds"]["actual"] is None,
              f"3i: the planted and the canary reading reply {reply['planted']:.3f} s and {reply['canary']:.3f} s after they "
              f"were asked (their price asks {wait_b:.3f} s), where the card took {took['planted']:.4f} s and "
              f"{took['canary']:.4f} s")
        disk_chk = (
            (tmp / "sess_plant_31b" / "session.json").read_text()
            + (tmp / "sess_plant_31b" / "transcript.jsonl").read_text()
            + json.dumps(rp)
        ).lower()
        check(plant_31b.id[:12] not in disk_chk and "plant" not in disk_chk and "canary" not in disk_chk,
              "3h: zero leakage of 31B plant/canary state to disk or driver_view")
        record["measurements"]["door_31b"] = {
            "capture_s": cap["seconds"],
            "cell_s": cell_r["seconds"],
            "paired_read_s": rd["gpu_seconds_charged"],
            "paired_rank_large": dv["body"]["summary"]["classes"]["large"]["rank"],
            "blinded_reply_s": {"planted": round(reply["planted"], 4), "canary": round(reply["canary"], 4),
                                "asked_of_both": round(wait_b, 4), "card_took": {k: round(v, 4) for k, v in took.items()}},
            "planted_repair_kl_large": rp["driver_view"]["body"]["summary"]["classes"]["large"]["arms"]["edit:1.0"]["kl"]["mean_over_items"],
            "canary_repair_kl_large": rc["driver_view"]["body"]["summary"]["classes"]["large"]["arms"]["edit:1.0"]["kl"]["mean_over_items"],
        }

    # Final global invariants
    we.assert_base(h.lm, h.hooks_at_load)
    fb1 = fell_back()
    check(fb1 == fb0, f"4: zero compiled unpacker fallbacks across entire 31B run ({fb0} -> {fb1})")
    bv = lb.verify(BOOK_PATH)
    check(bv["ok"] is True and bv["entries"] == 5, f"4: verification ledger whole ({bv})")
    peak_all = round(torch.cuda.max_memory_allocated(h.device) / 2**30, 3)
    total_s = round(time.perf_counter() - t_all, 1)
    record["measurements"]["total_s"] = total_s
    record["measurements"]["peak_gib_all"] = peak_all
    record["measurements"]["fell_back_delta"] = fb1 - fb0
    record["passed"] = PASS
    record["failed"] = len(FAIL)
    record["failures"] = FAIL
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nverify_door: {PASS} passed, {len(FAIL)} failed ({total_s} s, peak {peak_all} GiB, fell_back {fb1 - fb0})",
          flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
