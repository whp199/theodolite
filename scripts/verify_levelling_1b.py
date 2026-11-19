#!/usr/bin/env python3
"""The card's check of slice 1b of the weight-edit kernel (backend/weight_edits.py, backend/levelling.py,
backend/level_book.py): what only the RTX 5090 and the real Gemma-4-31B W4A16 checkpoint can show for the features
added in 1b. The CPU check (scripts/check_levelling.py) tests all of 1b on both CPU stand-ins; this script checks
the real 31B bf16 + W4A16 GPU kernels, quiet-mask counts (including V6_quiet_dims_pers181.npy), folded row overlays
(_memit_row_overlays), masked direction removal, the three new control families, delivered, stack, and diagnostics.

Stop the :8765 server and any census first: this loads the model itself (~18 GiB at rest after freeing the lens;
22 GiB free required before load).

Its readings go through `levelling.read`, each named "verify_1b: ...", into a ledger of its own beside its JSON
(experiments/results/levelling/verify_1b_book.jsonl), never touching the owner's real ledger. It writes
experiments/results/levelling/verify_1b.json after every section.

Sections:
 1. startup: 60 layers, 21,504 FFN width, 5,376 hidden size, quiet_mask counts at layers 5, 20, 44, 50;
 2. row edits (gate_proj and up_proj): dose 0 by bits, silence_rows / scale_rows(2, -0.5) vs folded _memit_row_overlays,
    recomputed set_rows / add_to_rows / scale_rows(1.5) vs folded _memit_row_overlays, and decode steps under row edits;
 3. capture_direction, writers, and masked remove_direction: quiet_share, exact zero delta on quiet_mask, stream-side
    orthogonality, folded twin gap, and paired reading against same_site controls;
 4. control families (other_sites_matched, same_site, sham_target) and delivered metric;
 5. stack and edit_diagnostics (dtype_changed_share, write_ceilings, quiet_mask_energy_share);
 6. teardown: assert_base, unpacker fell_back unchanged, book chain whole, peak GiB recorded.

Run: .venv/bin/python -I scripts/verify_levelling_1b.py   (exits 0 when every check holds, 2 when it could not start)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gc
import json
import math
import socket
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT_DIR = ROOT_DIR / "experiments" / "results" / "levelling"
OUT_FILE = OUT_DIR / "verify_1b.json"
BOOK = OUT_DIR / "verify_1b_book.jsonl"
os.environ["THEODOLITE_LEVEL_BOOK"] = str(BOOK)

import backend.engine_runtime as er  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import backend.atlas_ops as ao  # noqa: E402
import backend.level_book as lb  # noqa: E402
import backend.levelling as lv  # noqa: E402
import backend.weight_edits as we  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import gguf_weights as gw  # noqa: E402
from backend import w4a16_fast  # noqa: E402

MIN_FREE_GIB = 22.0
PORT = 8765

QUESTIONS = [
    ("moon", "In two or three sentences, why does the moon show phases?"),
    ("stock", "Explain briefly why a stock should not be salted while it simmers."),
    ("virus", "What is the difference between a virus and a bacterium? Answer in a short paragraph."),
    ("knife", "Give three practical tips for sharpening a kitchen knife."),
]

failures: List[str] = []
passed = 0
OUT: Dict[str, Any] = {"v": 1, "what": "scripts/verify_levelling_1b.py: the card's check of 1b",
                       "sections": {}, "readings": []}
NOW: Dict[str, Any] = {"section": "start"}
S: Dict[str, Any] = {}


def sec() -> Dict[str, Any]:
    return OUT["sections"].setdefault(NOW["section"], {"lines": [], "figures": {}})


def check(ok: bool, what: str) -> None:
    global passed
    sec()["lines"].append({"ok": bool(ok), "what": what})
    if ok:
        passed += 1
        print(f"  ok    {what}", flush=True)
    else:
        failures.append(f"§{NOW['section']}: {what}")
        print(f"  FAIL  {what}", flush=True)


def record(key: str, value: Any) -> None:
    sec()["figures"][key] = value


def save_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cz.write_json_atomic(OUT_FILE, OUT, indent=1)


def server_answers(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def run_section(tag: str, title: str, fn: Callable[[], None]) -> None:
    NOW["section"] = tag
    print(f"\n[{tag}] {title}", flush=True)
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        check(False, f"unexpected {type(exc).__name__}: {exc}")
        traceback.print_exc()
    sec()["seconds"] = round(time.perf_counter() - t0, 2)
    save_out()


def _own_items(h: Any) -> Dict[str, lv.Item]:
    """Generate the base model's own short greedy answers to QUESTIONS so items sit cleanly above the item gate."""
    stop = [int(t) for t in h.rt.end_of_turn_ids()]
    items: Dict[str, lv.Item] = {}
    for name, q in QUESTIONS:
        prompt = lv.chat_ids(h.tokenizer, q)
        pieces: List[int] = []
        for st in ao.write_rows(h.lm, h.W_U, [prompt], None, max_new=36, stop_ids=stop,
                                bos_id=h.tokenizer.bos_token_id, softcap=h.softcap):
            if st.end:
                break
            pieces.append(int(st.token_id))
        ans = h.tokenizer.decode(pieces).strip() or "Yes."
        items[name] = lv.make_item(h.tokenizer, q, ans, name=name)
    return items


def sec1_startup() -> None:
    h = S["h"]
    check(h.n_layers == 60 and h.d_ff == 21504 and h.d_model == 5376,
          f"60 layers, 21,504 FFN width, 5,376 hidden size (got {h.n_layers}, {h.d_ff}, {h.d_model})")
    check(we.overlays_in_force(h.lm) == [], "no overlays in force at startup")
    qm_counts = {str(L): int(we.quiet_mask(h.lm, L).sum()) for L in (5, 20, 44, 50)}
    record("quiet_mask_counts", qm_counts)
    check(all(c >= 181 for c in qm_counts.values()),
          f"quiet_mask includes at least the 181 persistent quiet dims at layers 5, 20, 44, 50 ({qm_counts})")


def sec2_row_edits() -> None:
    h, items = S["h"], S["items"]
    it = items["moon"]
    ids = torch.tensor([list(it.ids)], dtype=torch.long, device=h.device)
    # 1. Exact power-of-two / silencing row edits vs folded _memit_row_overlays at 12 and 32 rows
    row_gaps: Dict[str, Any] = {}
    for layer in (5, 20):
        for which in ("gate_proj", "up_proj"):
            for tag, spec in (("sil", we.silence_rows(layer, which, [4294])),
                              ("x2", we.scale_rows(layer, which, [4294], 2.0)),
                              ("xm05", we.scale_rows(layer, which, [4294], -0.5))):
                ed = we.build(h.lm, spec, model_id=h.model_id, weights_id=h.weights_id)
                for R in (12, 32):
                    x = ids.expand(R, -1)
                    with we.per_row(h.lm, [None, we.Arm(ed, 1.0), we.Arm(ed, 0.0)] + [None] * (R - 3)):
                        Hpr = h.lm(input_ids=x, use_cache=False).last_hidden_state
                    with we.folded(h.lm, ed, 1.0, hooks_at_load=h.hooks_at_load):
                        Hfo = h.lm(input_ids=x, use_cache=False).last_hidden_state
                    same_d0 = bool(torch.equal(Hpr[2], Hpr[0]))
                    same_un = bool(torch.equal(Hpr[-1], Hpr[0]))
                    same_fo = bool(torch.equal(Hpr[1], Hfo[0]))
                    max_d = float((Hpr[1].float() - Hfo[0].float()).abs().max())
                    row_gaps[f"L{layer}_{which}_{tag}_R{R}"] = {"same_bits": same_fo, "max_diff": max_d}
                    check(same_d0 and same_un,
                          f"L{layer} {which} {tag} R={R}: dose 0 and untouched equal base by bits")
                    check(same_fo,
                          f"L{layer} {which} {tag} R={R}: per_row equals folded _memit_row_overlays by bits (max_diff={max_d:.3e})")
    record("exact_row_edits", row_gaps)
    # 2. Recomputed row edits (scale 1.5, set_rows, add_to_rows) through lv.read with folded="always"
    rng = np.random.default_rng(11)
    recomp_gaps: Dict[str, Any] = {}
    for which in ("gate_proj", "up_proj"):
        mod = h.lm.get_submodule(we.linear_name(20, which))
        base_r = gw.dense_rows(mod, torch.tensor([4294, 5316], dtype=torch.long)).float().cpu().numpy()
        v_set = (base_r * 0.5 + rng.standard_normal(base_r.shape).astype(np.float32) * 1e-3).astype(np.float32)
        v_add = (rng.standard_normal(base_r.shape).astype(np.float32) * 1e-3).astype(np.float32)
        for tag, spec in (("scale1.5", we.scale_rows(20, which, [4294, 5316], 1.5)),
                          ("set", we.set_rows(20, which, [4294, 5316], v_set)),
                          ("add", we.add_to_rows(20, which, [4294, 5316], v_add))):
            ed = we.build(h.lm, spec, model_id=h.model_id, weights_id=h.weights_id)
            r = lv.read(h, {"name": f"verify_1b: row {which} {tag}", "kind": "paired",
                            "items": ["moon"], "doses": [1.0, 0.0], "controls": {"count": 0, "seed": 0},
                            "folded": "always"}, items, edit=ed)
            OUT["readings"].append({"name": r["name"], "status": r["status"], "seq": r["book"]["seq"]})
            fo_s = r["items"][0]["classes"]["small"]["folded"]
            fo_l = r["items"][0]["classes"]["large"]["folded"]
            recomp_gaps[f"{which}_{tag}"] = {
                "small_kl_gap": (fo_s.get("gap") or {}).get("kl"),
                "small_between_kl": fo_s.get("between_kl_mean"),
                "large_kl_gap": (fo_l.get("gap") or {}).get("kl"),
                "large_between_kl": fo_l.get("between_kl_mean"),
            }
            check(r["status"] == "ok" and fo_s["taken"] is True and fo_l["taken"] is True,
                  f"L20 {which} {tag}: paired reading + folded _memit_row_overlays twin ok")
    record("recomp_row_edits", recomp_gaps)
    # 3. Written decode step under a recomputed row edit
    mod_g = h.lm.get_submodule(we.linear_name(20, "gate_proj"))
    base_g = gw.dense_rows(mod_g, torch.tensor([4294], dtype=torch.long)).float().cpu().numpy()
    ed_w = we.build(h.lm, we.set_rows(20, "gate_proj", [4294], base_g * 0.5),
                    model_id=h.model_id, weights_id=h.weights_id)
    rw = lv.read(h, {"name": "verify_1b: written row edit", "kind": "written", "items": ["moon"],
                     "doses": [1.0, 0.0], "controls": {"count": 2, "seed": 0}, "rows": 8, "max_new": 24},
                 items, edit=ed_w)
    OUT["readings"].append({"name": rw["name"], "status": rw["status"], "seq": rw["book"]["seq"]})
    check(rw["status"] == "ok" and rw["base_after"] is True,
          f"written reading under recomputed row edit ok ({rw['status']}: {rw['why']})")


def sec3_direction_removal() -> None:
    h, items = S["h"], S["items"]
    u = lv.capture_direction(h, [items["moon"], items["stock"]], [items["virus"], items["knife"]], 20,
                             where="resid_post", at="span_mean")
    qm = we.quiet_mask(h.lm, 20)
    u64 = u.astype(np.float64)
    q_share = float(np.dot(u64[qm], u64[qm]) / max(float(np.dot(u64, u64)), 1e-30))
    record("captured_dir_L20_quiet_share", q_share)
    # Project u off quiet_mask if raw contrast exceeded MAX_QUIET_SHARE (still tests full pipeline)
    u_use = u.copy()
    if q_share > we.MAX_QUIET_SHARE:
        u_use[qm] = 0.0
    top_w = we.writers(h.lm, 20, u_use, 16)
    neurons = [n for n, _s in top_w]
    ed = we.build(h.lm, we.remove_direction(20, neurons, u_use), model_id=h.model_id, weights_id=h.weights_id)
    t = ed.terms[0]
    dW = t.vectors.astype(np.float64)
    wp = h.lm.layers[20].post_feedforward_layernorm.weight.detach().double().cpu().numpy()
    Wc = gw.dense_cols(h.lm.get_submodule(t.linear), torch.tensor(t.idx, dtype=torch.long)).double().cpu().numpy()
    u_m = u_use.astype(np.float64)
    u_m[qm] = 0.0
    u_m /= np.linalg.norm(u_m)
    orth = float(np.abs(((Wc + dW) * wp[None, :]) @ u_m).max())
    check(bool((dW[:, qm] == 0.0).all()) and orth < 1e-4,
          f"L20 remove_direction: zero delta on quiet_mask and stream orthogonality max={orth:.2e}")
    r = lv.read(h, {"name": "verify_1b: remove_direction + same_site", "kind": "paired",
                    "items": ["moon", "virus"], "doses": [1.0, 0.0],
                    "controls": {"family": "same_site", "count": 8, "seed": 0}, "folded": "always"},
                items, edit=ed)
    OUT["readings"].append({"name": r["name"], "status": r["status"], "seq": r["book"]["seq"]})
    record("remove_direction_report", ed.report.get("direction_removal"))
    record("remove_direction_diag", we.edit_diagnostics(h.lm, ed))
    check(r["status"] == "ok" and r["base_after"] is True
          and r["items"][0]["classes"]["small"]["folded"]["taken"] is True,
          f"L20 remove_direction paired reading + same_site controls + folded twin ok ({r['status']}: {r['why']})")


def sec4_controls_and_delivered() -> None:
    h, items = S["h"], S["items"]
    ed = we.build(h.lm, we.silence(20, [4294, 5316]), model_id=h.model_id, weights_id=h.weights_id)
    r_mat = lv.read(h, {"name": "verify_1b: other_sites_matched", "kind": "paired",
                        "items": ["moon", "stock"], "doses": [1.0, 0.0],
                        "controls": {"family": "other_sites_matched", "count": 10, "seed": 0, "tolerance": 2.0}},
                    items, edit=ed)
    OUT["readings"].append({"name": r_mat["name"], "status": r_mat["status"], "seq": r_mat["book"]["seq"]})
    record("matched_controls", {"status": r_mat["status"], "coverage": (r_mat.get("controls") or {}).get("coverage"),
                                "seconds": r_mat.get("seconds")})
    check(r_mat["status"] == "ok" and r_mat["controls"]["matched"] is True,
          f"L20 other_sites_matched ok ({r_mat['status']}: {r_mat['why']})")
    arm1 = next(a for a in r_mat["items"][0]["classes"]["small"]["arms"] if a["key"] == "edit:1.0")
    arm0 = next(a for a in r_mat["items"][0]["classes"]["small"]["arms"] if a["key"] == "edit:0.0")
    record("delivered_sample", {"dose1": arm1.get("delivered"), "dose0": arm0.get("delivered")})
    check(arm0.get("delivered") == 0.0 and (arm1.get("delivered") or 0.0) > 0.0,
          f"delivered is 0.0 at dose 0 and >0 at dose 1 ({arm1.get('delivered')})")
    shams = [{"terms": [{"layer": 18, "silence": [100, 200]}]},
             {"terms": [{"layer": 20, "silence": [300, 400]}]},
             {"terms": [{"layer": 22, "silence": [500, 600]}]}]
    # rows.small=8 has 2 arm slots (8 - 2 - 4 null), so 1 dose + 3 shams = 4 arms -> 2 forwards, exercising restart from L18
    r_sh = lv.read(h, {"name": "verify_1b: sham_target", "kind": "paired", "items": ["moon"],
                       "rows": {"small": 8, "large": 16},
                       "controls": {"family": "sham_target", "shams": shams}}, items, edit=ed)
    OUT["readings"].append({"name": r_sh["name"], "status": r_sh["status"], "seq": r_sh["book"]["seq"],
                            "peak_gib": r_sh.get("peak_gib")})
    check(r_sh["status"] == "ok" and r_sh["controls"]["family"] == "sham_target"
          and r_sh["items"][0]["classes"]["small"]["restart_from"] == 18,
          f"sham_target ok and lowered restart_from to 18 ({r_sh['status']}, restart_from={r_sh['items'][0]['classes']['small']['restart_from']})")
    # Coverage failure void on real 31B with impossibly tight tolerance
    r_void = lv.read(h, {"name": "verify_1b: matched coverage void", "kind": "paired", "items": ["moon"],
                         "controls": {"family": "other_sites_matched", "count": 40, "seed": 0, "tolerance": 1.000001}},
                     items, edit=ed)
    OUT["readings"].append({"name": r_void["name"], "status": r_void["status"], "seq": r_void["book"]["seq"]})
    cov_v = (r_void.get("controls") or {}).get("coverage") or {}
    record("matched_void_coverage", cov_v)
    check(r_void["status"] == "void" and (r_void.get("void") or {}).get("code") == "insufficient_matched_controls"
          and cov_v.get("sufficient") is False and r_void["base_after"] is True,
          f"other_sites_matched voids cleanly on tight tolerance ({r_void['status']}, {r_void.get('void')})")


def sec5_stack_and_diagnostics() -> None:
    h, items = S["h"], S["items"]
    e1 = we.build(h.lm, we.silence(20, [4294]), model_id=h.model_id, weights_id=h.weights_id)
    e2 = we.build(h.lm, we.silence_rows(20, "gate_proj", [5316]), model_id=h.model_id, weights_id=h.weights_id)
    st = we.stack(h.lm, [e1, e2], model_id=h.model_id, weights_id=h.weights_id, note="col+row stack")
    diag = we.edit_diagnostics(h.lm, st, dose=1.0)
    record("stack_diagnostics", diag)
    r = lv.read(h, {"name": "verify_1b: stacked col+row", "kind": "paired", "items": ["moon"],
                    "doses": [1.0, 0.0], "controls": {"count": 0, "seed": 0}, "folded": "always"},
                items, edit=st)
    OUT["readings"].append({"name": r["name"], "status": r["status"], "seq": r["book"]["seq"]})
    check(r["status"] == "ok" and r["base_after"] is True
          and r["items"][0]["classes"]["small"]["folded"]["taken"] is True,
          f"stacked col+row edit paired reading + folded twin ok ({r['status']}: {r['why']})")


def sec6_teardown() -> None:
    h = S["h"]
    ok_base = True
    try:
        we.assert_base(h.lm, h.hooks_at_load)
    except we.NotBase as exc:
        ok_base = False
        record("not_base", str(exc))
    fb = getattr(er.dequantize_w4a16_full, "fell_back", None)
    v = lb.verify(BOOK)
    record("max_allocated_gib", round(torch.cuda.max_memory_allocated(h.device) / 2**30, 3))
    check(ok_base, "model is the base after all 1b readings")
    check(fb == S["fell_back_start"], f"compiled unpacker fell_back unchanged ({fb} == {S['fell_back_start']})")
    check(v["ok"] is True, f"verify_1b ledger chain is whole ({v})")


def main() -> int:
    torch.set_grad_enabled(False)
    if server_answers(PORT):
        print(f"Something answers on :{PORT}: stop the Theodolite server first.", flush=True)
        return 2
    if not torch.cuda.is_available():
        print("No CUDA device answers in this process.", flush=True)
        return 2
    free, _total = torch.cuda.mem_get_info()
    if free < MIN_FREE_GIB * 2**30:
        print(f"Only {free / 2**30:.1f} GiB free ({MIN_FREE_GIB:g} required): stop server/census first.", flush=True)
        return 2
    from backend.runtime_factory import make_runtime

    t0 = time.perf_counter()
    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    gc.collect()
    torch.cuda.empty_cache()
    w4a16_fast.install(rt, "compiled", log=lambda m: print(m, flush=True))
    h = lv.handle(rt)
    S["h"] = h
    S["fell_back_start"] = getattr(er.dequantize_w4a16_full, "fell_back", None)
    S["items"] = _own_items(h)
    OUT["runtime"] = h.runtime
    OUT["load_s"] = round(time.perf_counter() - t0, 2)

    run_section("1", "startup and quiet_mask", sec1_startup)
    run_section("2", "row edits (gate_proj, up_proj)", sec2_row_edits)
    run_section("3", "capture_direction, writers, remove_direction", sec3_direction_removal)
    run_section("4", "control families and delivered", sec4_controls_and_delivered)
    run_section("5", "stack and edit_diagnostics", sec5_stack_and_diagnostics)
    run_section("6", "teardown", sec6_teardown)

    OUT["total_s"] = round(time.perf_counter() - t0, 1)
    OUT["passed"] = passed
    OUT["failed"] = len(failures)
    save_out()
    if failures:
        print(f"\nverify_1b: {len(failures)} of {passed + len(failures)} checks FAILED ({OUT['total_s']} s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nverify_1b: all {passed} checks hold ({OUT['total_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
