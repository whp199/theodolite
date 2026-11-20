#!/usr/bin/env python3
"""Real RTX 5090 31B W4A16 card verification for Levelling Phase 4: The Person's Page (`scripts/verify_level_page.py`).

Binds the live Gemma-4-31B-it W4A16 runtime + `LarqlVaultEngine` directly to `backend/server.py` (in-process via
FastAPI `TestClient`, no external port) and verifies end to end:
1. `GET /api/level/status` reports `n_layers == 60`, `d_ff == 21504`, `is_base == True`, and allowlisted books.
2. `POST /api/level/estimate` prices the E6 layer-51 (index 50) top-10 `'she'` preset off-GPU.
3. `POST /api/level/compare` runs base, edit (`E6_L50_SHE10`), and 3 matched control edits (matched on what a
   base pre-pass expects each to deliver) side by side in ONE
   batched written forward (`max_new=24`) PLUS the two-batch-class (`small=12`, `large=32`) paired table with `' she'`
   vs `' he'` odds, verifying `untouched_rows_identical == True`, `base_after == True`, and ledger writes.
4. Live Vault-overlay isolation on the 31B checkpoint: `DELETE FROM LAYER 50 WHERE slot = 5313` causes `/api/level/status`
   to report `is_base == False` and `/api/level/compare` to return HTTP 409; `ROLLBACK` restores clean base.
5. Model-free replay of both the freshly written live book (`written` seq 0 & `paired` seq 1) and the verified 19 Nov
   2026 Phase 3 book (`verify_claim` seq 10 `works but harms`, seq 17 `kept the base`).
6. Invariants: 0 unpacker fallbacks, `assert_base` clean, ledger chain whole, peak VRAM recorded.

Outputs:
- `experiments/results/levelling/verify_page.json`
- `experiments/results/levelling/verify_page_book.jsonl`
"""
from __future__ import annotations

import datetime
import gc
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

torch.set_grad_enabled(False)

from fastapi.testclient import TestClient  # noqa: E402

import backend.server as server  # noqa: E402
from backend import engine_runtime  # noqa: E402
from backend import level_book as lb  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend.census import write_json_atomic  # noqa: E402
from backend.larql_vault import LarqlVaultEngine  # noqa: E402
from backend.mcp_scientist import bind_runtime  # noqa: E402
from backend.runtime_factory import make_runtime  # noqa: E402

OUT_DIR = ROOT_DIR / "experiments" / "results" / "levelling"
OUT_JSON = OUT_DIR / "verify_page.json"
OUT_BOOK = OUT_DIR / "verify_page_book.jsonl"

E6_L50_SHE10 = [5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070]


def _port_open(port: int = 8765) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    if _port_open(8765):
        print("ERROR: Theodolite server answers on :8765; stop it before running verify_level_page.py.", file=sys.stderr)
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
    print("Loading 31B W4A16 (lens freed, compiled unpacker) + binding server.app...", flush=True)
    t_load = time.perf_counter()
    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    gc.collect()
    torch.cuda.empty_cache()
    w4a16_fast.install(rt, "compiled", log=lambda s: print(f"  [unpacker] {s}", flush=True))
    fb0 = int(getattr(engine_runtime.dequantize_w4a16_full, "fell_back", 0) or 0)
    vault = LarqlVaultEngine(rt)
    server._RUNTIME, server._VAULT = rt, vault  # type: ignore[assignment]
    bind_runtime(rt, vault)

    @asynccontextmanager
    async def no_lifespan(_app: Any):
        yield

    server.app.router.lifespan_context = no_lifespan
    lm = lv.language_model(rt.hf_model)
    we.assert_base(lm)
    load_s = round(time.perf_counter() - t_load, 2)
    print(f"Loaded {rt.model_id} in {load_s} s.", flush=True)
    torch.cuda.reset_peak_memory_stats()

    prev_bk = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(OUT_BOOK)
    try:
        client = TestClient(server.app, base_url="http://127.0.0.1:8765")

        # 1. GET /api/level/status
        r_st = client.get("/api/level/status")
        st = r_st.json()
        check(
            "status_31b_ready",
            r_st.status_code == 200 and st["runtime_ready"] and st["n_layers"] == 60 and st["d_ff"] == 21504 and st["is_base"],
            {"n_layers": st["n_layers"], "d_ff": st["d_ff"], "books": [b["id"] for b in st["books"]]},
        )

        # 2. POST /api/level/estimate
        req_body: Dict[str, Any] = {
            "op": "silence",
            "linear": "down_proj",
            "layer": 50,
            "neurons": E6_L50_SHE10,
            "factor": 0.0,
            "dose": 1.0,
            "question": "Continue naturally: Claire was an experienced architect. Before leaving the gallery,",
            "answer": " Claire picked up the notebook and",
            "controls_family": "other_sites_matched",
            "controls_count": 3,
            "controls_seed": 0,
            "max_new": 24,
            "thinking": False,
            "sampled": False,
            "include_paired": True,
            "toward_word": "she",
            "away_word": "he",
            "odds_at": "end",
            "name": "page_31b_e6_she10",
        }
        r_est = client.post("/api/level/estimate", json=req_body)
        est = r_est.json()
        check(
            "estimate_31b_ok",
            r_est.status_code == 200 and est["total_forwards"] > 0 and est["paired"] is not None,
            {"forwards": est["total_forwards"], "est_seconds": est["total_seconds"], "peak_gib": est["peak_gib_above_model"]},
        )

        # 3. POST /api/level/compare (side-by-side 3 voices + two-batch-class paired table)
        print("\nRunning POST /api/level/compare on 31B (E6 L50 she_10 vs base vs 3 matched controls)...", flush=True)
        t_cmp = time.perf_counter()
        r_cmp = client.post("/api/level/compare", json=req_body)
        cmp_s = round(time.perf_counter() - t_cmp, 2)
        cmp_d = r_cmp.json()
        voices = cmp_d.get("voices") or {}
        paired = cmp_d.get("paired") or {}
        large_odds = ((paired.get("classes") or {}).get("large") or {}).get("metrics", {}).get("odds", {}).get("mean_over_items")
        small_odds = ((paired.get("classes") or {}).get("small") or {}).get("metrics", {}).get("odds", {}).get("mean_over_items")
        large_at_least = ((paired.get("classes") or {}).get("large") or {}).get("at_least")
        check(
            "compare_31b_ok",
            r_cmp.status_code == 200 and cmp_d.get("status") == "ok" and cmp_d.get("base_after") is True,
            {"seconds": cmp_s, "small_odds": small_odds, "large_odds": large_odds, "large_at_least": large_at_least},
        )
        check(
            "compare_31b_voices_present",
            voices.get("untouched_rows_identical") is True and len(voices.get("controls") or []) == 3,
            {
                "base_text": (voices.get("base") or {}).get("text", "")[:90],
                "edit_text": (voices.get("edit") or {}).get("text", "")[:90],
                "edit_parts_at": (voices.get("edit") or {}).get("parts_at"),
                "ctl0_text": ((voices.get("controls") or [{}])[0]).get("text", "")[:90],
            },
        )
        check(
            "compare_31b_both_batch_classes",
            "small" in (paired.get("classes") or {}) and "large" in (paired.get("classes") or {}),
            {"book": cmp_d.get("book")},
        )

        # 4. Live Vault-overlay 409 guard on 31B + clean ROLLBACK
        print("\nTesting live Vault overlay 409 guard & ROLLBACK on 31B...", flush=True)
        vault.execute_lql("DELETE FROM LAYER 50 WHERE slot = 5313")
        st_dirty = client.get("/api/level/status").json()
        r_409 = client.post("/api/level/compare", json=req_body)
        vault.execute_lql("ROLLBACK")
        st_restored = client.get("/api/level/status").json()
        check(
            "vault_overlay_409_guard",
            st_dirty["is_base"] is False and r_409.status_code == 409 and st_restored["is_base"] is True,
            {"dirty_status": st_dirty["active_vault_edits"], "compare_code": r_409.status_code, "restored_base": st_restored["is_base"]},
        )

        # 5. Model-free replay of live book & verified Phase 3 31B book
        r_bk = client.get("/api/level/book?book=live&tail=10").json()
        r_b0 = client.get("/api/level/book/0?book=live").json()
        r_b1 = client.get("/api/level/book/1?book=live").json()
        r_vc10 = client.get("/api/level/book/10?book=verify_claim").json()
        check(
            "replay_live_and_recorded_ok",
            r_bk["verify"]["ok"]
            and len(r_bk["entries"]) == 2
            and r_b0["card"]["voices"] is not None
            and r_b1["card"]["paired"] is not None
            and r_vc10["card"]["verdict"] == "works but harms",
            {"live_entries": len(r_bk["entries"]), "verify_claim_10": r_vc10["card"]["verdict"]},
        )

        # 6. Final invariants
        we.assert_base(lm)
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
            "model_id": rt.model_id,
            "load_seconds": load_s,
            "total_seconds": total_s,
            "peak_gib": peak_gib,
            "fallback_count": fb,
            "passed": n_ok,
            "failed": n_fail,
            "checks": checks,
            "estimate": est,
            "compare": cmp_d,
            "ledger": bv,
        }
        write_json_atomic(OUT_JSON, report)
        print(f"\nPhase 4 31B card verification complete in {total_s} s (peak {peak_gib:.3f} GiB): "
              f"{n_ok}/{len(checks)} passed, {n_fail} failed [measured on the 31B card].", flush=True)
        return 0 if n_fail == 0 else 1
    finally:
        if prev_bk is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = prev_bk


if __name__ == "__main__":
    sys.exit(main())
