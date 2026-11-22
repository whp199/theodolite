#!/usr/bin/env python3
"""CPU and containment checks for Levelling Phase 1 completion (no-card CPU weights reader, long sampled/thinking-on
written readings with loop & cut-off detectors) and Phase 2 (the agent's door: bwrap off-card sandbox, session gates,
budgets, planted/canary blinding, transcript copy, and stdio MCP server).

Runs on the CPU with GPUs hidden (`CUDA_VISIBLE_DEVICES=""`) at one thread using the house stand-ins
(`scripts/atlas_stand_in.py` and a packed bf16 stand-in). Exits 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from backend import level_book as lb  # noqa: E402
from backend import level_claim as lcl  # noqa: E402
from backend import level_mcp as lmcp  # noqa: E402
from backend import level_sandbox as ls  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import levelling as lv  # noqa: E402
from backend import weight_edits as we  # noqa: E402
from backend import weight_reader as wr_mod  # noqa: E402
from backend.causal_check import language_model  # noqa: E402
from backend.gguf_weights import dense, dense_cols, dense_rows  # noqa: E402

PASS = 0
FAIL: List[str] = []
KEPT_OUT_PATTERN = re.compile(
    r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b",
    re.I,
)


def check(cond: bool, msg: str) -> None:
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(msg)
        print(f"  FAIL  {msg}", flush=True)


def section(title: str) -> None:
    print(f"\n[{title}]", flush=True)


def _build_packed_stand_in() -> Tuple[Any, lv.Handle]:
    """Build the 6-layer packed bf16 stand-in using `scripts/check_dequant.py`."""
    import check_dequant as cd

    rt = cd.packed_stand_in()[0]
    return rt, lv.handle(rt)


def _build_float_stand_in() -> Tuple[Any, lv.Handle]:
    import atlas_stand_in

    atlas_stand_in.use_stand_in_atlas()
    rt, _vault = atlas_stand_in.stand_in()
    return rt, lv.handle(rt)


def _make_items(tok: Any) -> Dict[str, lv.Item]:
    pairs = [
        ("alpha", "What colour is an orange?", "Orange is bright and warm."),
        ("beta", "How many legs does a spider have?", "A spider has eight legs."),
        ("gamma", "Name a river in Egypt.", "The Nile flows north through Egypt."),
    ]
    return {nm: lv.make_item(tok, q, a, name=nm) for nm, q, a in pairs}


# --------------------------------------------------------------------------------------------- 1. weight_reader


def check_weight_reader(Pb: Any, hP: lv.Handle, hF: lv.Handle, tmp: Path) -> Path:
    section("1. No-card CPU weights reader (packed bf16 + dense float32 + sharded)")
    p_dir = wr_mod.export_stand_in_safetensors(hP.lm, tmp / "weights_packed", model_id=hP.model_id)
    f_dir = wr_mod.export_stand_in_safetensors(hF.lm, tmp / "weights_dense", model_id=hF.model_id)

    with wr_mod.WeightReader(p_dir, model_id=hP.model_id) as rP:
        info = rP.info()
        check(info["n_layers"] == hP.n_layers and info["d_model"] == hP.d_model and info["d_ff"] == hP.d_ff,
              f"1: packed WeightReader.info dimensions match ({info})")
        check(info["linear_kinds"].get("w4a16", 0) > 0, f"1: packed WeightReader detected w4a16 linears ({info['linear_kinds']})")
        for L in (0, 2, 5):
            for proj in ("down_proj", "gate_proj", "up_proj"):
                lin = f"layers.{L}.mlp.{proj}"
                mod = hP.lm.get_submodule(lin)
                check(rP.tensor_pin(lin) == we.tensor_pin(mod), f"1: packed tensor_pin matches live model on {lin}")
                check(rP.linear_shape(lin) == we.linear_shape(mod), f"1: packed linear_shape matches on {lin}")
        # Bit-exact dense, dense_cols, dense_rows
        mod_d = hP.lm.get_submodule("layers.3.mlp.down_proj")
        mod_g = hP.lm.get_submodule("layers.3.mlp.gate_proj")
        check(torch.equal(rP.dense("layers.3.mlp.down_proj"), dense(mod_d).cpu()),
              "1: packed dense() bit-identical to engine_runtime unpacker")
        idx_c = [0, 7, 15, 31, 32, 63, 7, 127]
        check(torch.equal(rP.dense_cols("layers.3.mlp.down_proj", idx_c),
                          dense_cols(mod_d, torch.tensor(idx_c, dtype=torch.long)).cpu()),
              "1: packed dense_cols() bit-identical (including duplicates & group boundaries)")
        idx_r = [0, 5, 19, 5, 120]
        check(torch.equal(rP.dense_rows("layers.3.mlp.gate_proj", idx_r),
                          dense_rows(mod_g, torch.tensor(idx_r, dtype=torch.long)).cpu()),
              "1: packed dense_rows() bit-identical (including duplicates)")
        check(rP.dense_cols("layers.3.mlp.down_proj", []).shape == (0, hP.d_model),
              "1: empty dense_cols returns [0, out]")
        check(rP.dense_rows("layers.3.mlp.gate_proj", []).shape == (0, hP.d_model),
              "1: empty dense_rows returns [0, in]")

        # w_post, quiet_mask, down_stream_norms, writers match weight_edits
        qm_live = we.quiet_mask(hP.lm, 3)
        qm_disk = rP.quiet_mask(3)
        check(np.array_equal(qm_live, qm_disk), "1: quiet_mask matches weight_edits.quiet_mask exactly")
        n_live = we.down_stream_norms(hP.lm, 3)
        n_disk = rP.down_stream_norms(3)
        check(np.allclose(n_live, n_disk, rtol=1e-6, atol=1e-6),
              f"1: down_stream_norms matches weight_edits (max diff {float(np.abs(n_live - n_disk).max()):.2e})")
        rng = np.random.default_rng(123)
        u = rng.standard_normal(hP.d_model).astype(np.float32)
        u[qm_disk] = 0.0
        w_live = we.writers(hP.lm, 3, u, 10)
        w_disk = rP.writers(3, u, 10)
        check([i for i, _ in w_live] == [i for i, _ in w_disk]
              and all(abs(a - b) < 1e-5 for (_, a), (_, b) in zip(w_live, w_disk)),
              "1: writers(dot) matches weight_edits.writers in exact order and values")
        wc_live = we.writers(hP.lm, 3, u, 10, cosine=True)
        wc_disk = rP.writers(3, u, 10, cosine=True)
        check([i for i, _ in wc_live] == [i for i, _ in wc_disk]
              and all(abs(a - b) < 1e-5 for (_, a), (_, b) in zip(wc_live, wc_disk)),
              "1: writers(cosine) matches weight_edits.writers")
        # Out-of-range refusal
        refused_oob = False
        try:
            rP.dense_cols("layers.3.mlp.down_proj", [999999])
        except wr_mod.WeightReadError:
            refused_oob = True
        check(refused_oob, "1: dense_cols out-of-range raises WeightReadError")

    # Dense float32 stand-in + single-file open
    with wr_mod.WeightReader(f_dir / "model.safetensors", model_id=hF.model_id) as rF:
        mod_fd = hF.lm.get_submodule("layers.2.mlp.down_proj")
        check(rF.tensor_pin("layers.2.mlp.down_proj") == we.tensor_pin(mod_fd),
              "1: dense single-file WeightReader tensor_pin matches live model")
        check(torch.equal(rF.dense_cols("layers.2.mlp.down_proj", [1, 4, 9]),
                          dense_cols(mod_fd, torch.tensor([1, 4, 9], dtype=torch.long)).cpu()),
              "1: dense WeightReader dense_cols bit-identical")

    # Sharded safetensors (model.safetensors.index.json)
    shard_dir = tmp / "weights_sharded"
    shard_dir.mkdir(parents=True, exist_ok=True)
    from safetensors import safe_open
    from safetensors.torch import save_file as save_st

    t_a, t_b, wmap = {}, {}, {}
    with safe_open(str(p_dir / "model.safetensors"), framework="pt", device="cpu") as sf:
        for i, k in enumerate(sorted(sf.keys())):
            if i % 2 == 0:
                t_a[k] = sf.get_tensor(k)
                wmap[k] = "shard-00001.safetensors"
            else:
                t_b[k] = sf.get_tensor(k)
                wmap[k] = "shard-00002.safetensors"
    save_st(t_a, str(shard_dir / "shard-00001.safetensors"))
    save_st(t_b, str(shard_dir / "shard-00002.safetensors"))
    (shard_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wmap}), encoding="utf-8"
    )
    with wr_mod.WeightReader(shard_dir) as rS:
        check(rS.info()["shards"] == 2
              and rS.tensor_pin("layers.3.mlp.down_proj") == we.tensor_pin(hP.lm.get_submodule("layers.3.mlp.down_proj")),
              "1: sharded WeightReader loads index.json and matches tensor_pin")

    # save_array validation
    meta = wr_mod.save_array(tmp / "arr_ok.npy", np.array([1.5, -2.0], dtype=np.float64))
    check(meta["dtype"] == "float32" and meta["shape"] == [2] and len(meta["sha256"]) == 64,
          "1: save_array writes float32 npy with sha256")
    bad_nan = False
    try:
        wr_mod.save_array(tmp / "arr_bad.npy", np.array([1.0, float("nan")], dtype=np.float32))
    except wr_mod.WeightReadError:
        bad_nan = True
    check(bad_nan, "1: save_array refuses NaN/Inf")
    return p_dir


# ------------------------------------------------------------------------------- 2. long sampled written readings


def check_written_extended(Pb: Any, hP: lv.Handle, items: Dict[str, lv.Item], tmp: Path) -> None:
    section("2. Long sampled / thinking-on written readings & loop/cut-off detectors")
    # Pure detector checks
    d_tail = lv.detect_loop([10, 11, 12] + [1, 2, 3, 4, 5, 6] * 5)
    check(d_tail["looped"] is True and d_tail["kind"] == "periodic" and d_tail["period"] == 6 and d_tail["repeats"] == 5,
          f"2: detect_loop catches periodic tail loop ({d_tail})")
    d_mid = lv.detect_loop([99] * 5 + [7, 8, 9, 10] * 7 + [50, 51, 52])
    check(d_mid["looped"] is True and d_mid["kind"] == "periodic" and d_mid["period"] == 4,
          f"2: detect_loop catches interior periodic loop ({d_mid})")
    enum_seq: List[int] = []
    for k in range(5):
        enum_seq.extend([1000 + k] + list(range(1, 18)))  # 17 shared tokens separated by incrementing counter
    d_ng = lv.detect_loop(enum_seq)
    check(d_ng["looped"] is True and d_ng["kind"] == "ngram" and d_ng["repeats"] >= 4,
          f"2: detect_loop catches incrementing enumeration n-gram loop ({d_ng})")
    d_clean = lv.detect_loop(list(range(200)))
    check(d_clean["looped"] is False and d_clean["kind"] is None, f"2: detect_loop clean on diverse sequence ({d_clean})")
    d_short = lv.detect_loop([1, 2, 3] * 3)
    check(d_short["looped"] is False, "2: detect_loop ignores short <24-piece repeat")

    # The two loop rules on labelled shapes (pieces are made-up ids, the text is what a reader would see). Each line:
    # the shape, its pieces, its text, whether it closed with an end mark, and what each rule must say:
    # (the instrument's, the published rule's tight loop, its soft loop). Where the two part, the label says who errs.
    def numbered(entries: List[str], first: int = 1, gap: str = "\n") -> str:
        return gap.join(f"{first + i}. {e}" for i, e in enumerate(entries))

    def ids_of_list(n_lines: int, body: List[int]) -> List[int]:
        out: List[int] = []
        for i in range(n_lines):
            out.extend([2000 + i] + body)  # a counter piece that rises, then the entry's pieces
        return out

    wire = ["Season 1 Episode 1: The Target", "Season 1 Episode 2: The Detail", "Season 1 Episode 3: The Buys"]
    collapsed = wire + ["Season 3 Episode 9: Slapstick"] * 12
    chorus_ids = list(range(700, 720))
    song_ids: List[int] = []
    for v in range(4):
        song_ids.extend(list(range(800 + 20 * v, 812 + 20 * v)) + chorus_ids)
    song_text = "".join(f"Verse {v}: the river bends by mill number {v}.\nAnd we will walk the long road home tonight.\n"
                        for v in range(4))
    shapes = [
        ("a sentence repeated to the end (a tight loop)",
         list(range(1, 8)) * 30, "The answer is forty-two. " * 30, False, (True, True, False)),
        ("five pieces repeated three times at the end (the published rule's least loop; under the instrument's 24 pieces)",
         list(range(300, 320)) + [1, 2, 3, 4, 5] * 3, "x", False, (False, True, False)),
        ("a repeat inside an answer that then goes on and ends (the instrument finds it; the published rule reads tails)",
         [99] * 5 + [7, 8, 9, 10] * 7 + [50, 51, 52], "x", False, (True, False, False)),
        ("sixty pieces repeated three times (over the published rule's 40: its own stated miss)",
         list(range(100, 160)) * 3, "x", False, (True, False, False)),
        ("a numbered list collapsing to one answer under rising numbers (the published fault's soft loop; the "
         "instrument's rule MISSES it)",
         ids_of_list(15, [500, 501, 502, 503, 504]), numbered(collapsed), False, (False, False, True)),
        ("the same list with a blank line between entries",
         ids_of_list(15, [500, 501, 502, 503, 504]), numbered(collapsed, gap="\n\n"), False, (False, False, True)),
        ("a song whose chorus comes four times (NOT a loop; the instrument's n-gram clause calls it one)",
         song_ids, song_text, True, (True, False, False)),
        ("a run of one piece, as a 400-character rule of '=' is (both rules call it a loop; the paper's readers "
         "called theirs another kind of failure)",
         [9] * 4 + [61] * 30, "Here is the table.\n" + "=" * 400, False, (True, True, False)),
        ("an honest numbered list of forty different entries",
         [3000 + i for i in range(200)], numbered([f"Entry {i} of the list" for i in range(40)]),
         True, (False, False, False)),
        ("list lines with nothing after their markers (passed over: the published detector's own false positives)",
         [4000 + i for i in range(40)], "- \n- \n* \n3.\n- \n(5)\n", True, (False, False, False)),
        ("four entries alike but for their markers, case and spacing",
         [4100 + i for i in range(30)], "- Mercury\n2) mercury\n(3)   MERCURY\n* Mercury", True, (False, False, True)),
        ("three entries alike (one short of a soft loop)",
         [4200 + i for i in range(30)], "1. Mercury\n2. Mercury\n3. Mercury\n4. Venus", True, (False, False, False)),
        ("four doses that differ only before the point (a decimal's whole part is no list marker)",
         [4300 + i for i in range(30)], "0.5 mg\n1.5 mg\n2.5 mg\n3.5 mg", True, (False, False, False)),
        ("four meetings at different hours (an hour is no list marker)",
         [4400 + i for i in range(30)], "8:30 - Meeting\n9:30 - Meeting\n10:30 - Meeting\n11:30 - Meeting", True,
         (False, False, False)),
        ("a collapse written with other markers (\"Item 4: X\"): NEITHER rule finds it, and the sheet says such a "
         "form counts as no loop",
         ids_of_list(12, [600, 601, 602]), "\n".join(f"Item {i}: Slapstick" for i in range(4, 16)), False,
         (False, False, False)),
    ]
    for label, pcs, text, ended, (want_own, want_tight, want_soft) in shapes:
        own = lv.detect_loop(pcs)
        pub = lv.published_loop(pcs, text, ended=ended)
        check(bool(own["looped"]) is want_own and pub["tight"] is want_tight and pub["soft"] is want_soft
              and pub["looped"] is (want_tight or want_soft) and pub["rule"] == lv.LOOP_RULE_PUBLISHED,
              f"2: loop rules on {label}: instrument {own['looped']} (want {want_own}), published tight "
              f"{pub['tight']} (want {want_tight}), soft {pub['soft']} (want {want_soft})")
    tail_then_end = list(range(1, 8)) * 5 + [999]
    check(lv.published_loop(tail_then_end, "x", ended=True)["tight"] is True
          and lv.published_loop(tail_then_end, "x", ended=False)["tight"] is False,
          "2: published_loop leaves a row's end mark out of the tail it reads")
    row_both = {"loop": {"looped": False}, "loop_published": {"looped": True}}
    row_old = {"loop": {"looped": True}}
    check(lv.loop_flag(row_both, "instrument") is False and lv.loop_flag(row_both, "published") is True
          and lv.loop_flag(row_both, "either") is True and lv.loop_flag(row_old, "published") is False
          and lv.loop_flag(row_old, "either") is True,
          "2: loop_flag counts a row by the instrument's rule, the published one, or either")
    got_rule = False
    try:
        lv.loop_flag(row_both, "a person")
    except lv.Refused:
        got_rule = True
    check(got_rule, "2: loop_flag refuses a rule it does not have")

    class _FakeTok:
        unk_token_id = 0
        def convert_tokens_to_ids(self, tok: str) -> int:
            return {"<|channel>": 90, "<channel|>": 91}.get(tok, 0)

    sp1 = lv.split_thinking([90, 5, 6, 7, 91, 12, 13], _FakeTok())
    check(sp1 == {"has_thought": True, "thought_closed": True, "thought_pieces": 5, "answer_pieces": 2},
          f"2: split_thinking closed thought ({sp1})")
    sp2 = lv.split_thinking([90, 5, 6, 7, 8], _FakeTok())
    check(sp2 == {"has_thought": True, "thought_closed": False, "thought_pieces": 5, "answer_pieces": 0},
          f"2: split_thinking unclosed thought ({sp2})")
    check(lv.written_stratum(False, True, True) == "looped"
          and lv.written_stratum(False, False, True) == "cut_off"
          and lv.written_stratum(True, False, False) == "ended",
          "2: written_stratum priority (looped > cut_off > ended)")

    # Normalise & estimate checks for extended written plans
    sil = {"terms": [{"layer": 3, "silence": [4, 9]}]}
    base_wr = {"name": "w_ext", "kind": "written", "edit": sil, "items": ["alpha"], "rows": 8, "max_new": 20,
               "controls": {"count": 2, "seed": 0}, "sampler": {"temperature": 0.8, "top_p": 0.9, "seed": 5},
               "samples": 3}
    norm = lv.normalise(base_wr, n_layers=6, dtype=torch.bfloat16)
    check(norm["samples"] == 3 and norm["sampler"]["temperature"] == 0.8 and norm["sampler"]["seed"] == 5
          and norm["diagnose"] is True and norm["thinking"] is False,
          f"2: normalise fills extended written fields ({norm})")
    est1 = lv.estimate({**norm, "samples": 1}, [(40, 20)], n_layers=6, facts=we.recipe_facts(sil, tmp))
    est3 = lv.estimate(norm, [(40, 20)], n_layers=6, facts=we.recipe_facts(sil, tmp))
    check(est3["forwards"] == 3 * est1["forwards"] and abs(est3["seconds"] - 3 * est1["seconds"]) < 1e-6,
          "2: estimate scales forwards and writing time linearly with samples")
    # Refusals: samples > 1 at temp 0; presence_penalty at temp 0; min_p refused; max_new > MAX_NEW_LONG
    for label, bad in [
        ("greedy multi-sample", {**base_wr, "sampler": {"temperature": 0.0}, "samples": 2}),
        ("greedy presence_penalty", {**base_wr, "sampler": {"temperature": 0.0, "presence_penalty": 0.5}, "samples": 1}),
        ("min_p not offered", {**base_wr, "sampler": {"temperature": 0.8, "min_p": 0.1}}),
        ("max_new > MAX_NEW_LONG", {**base_wr, "max_new": lv.MAX_NEW_LONG + 1}),
    ]:
        got_ref = False
        try:
            lv.normalise(bad, n_layers=6, dtype=torch.bfloat16)
        except lv.Refused:
            got_ref = True
        check(got_ref, f"2: normalise refuses {label}")

    # Live sampled written reading on packed bf16 stand-in
    book = tmp / "written_ext_book.jsonl"
    os.environ[lb.BOOK_ENV] = str(book)
    r = lv.read(hP, {**base_wr, "samples": 2, "max_new": 12, "doses": [1.0, 0.0]}, items)
    check(r["status"] == "ok" and r["base_after"] is True and r["summary"] is not None,
          f"2: sampled written reading ok with summary ({r['status']}: {r['why']})")
    sm = r["summary"]
    rows_seen = [s["base"] for it in r["items"] for s in it["samples"]] + [a for it in r["items"] for s in it["samples"] for a in s["arms"]]
    check(all(row.get("loop_published", {}).get("rule") == lv.LOOP_RULE_PUBLISHED and "soft" in row["loop_published"]
              for row in rows_seen)
          and sm["loop_rules"]["looped_published"] == lv.LOOP_RULE_PUBLISHED
          and {"looped_published", "looped_either", "loop_rules_differ"} <= set(sm["base"])
          and {"looped_published", "looped_either", "loop_rules_differ"} <= set(sm["arms"]["edit:1.0"])
          and any("by the published rule" in ln and "Neither is checked against a person's reading" in ln
                  for ln in lv.lines(r)),
          "2: every diagnosed row is read under both loop rules, and the summary and its printed lines count both")
    check(sm["total_generations"] == 2 and sm["samples_per_item"] == 2 and sm["base"]["n"] == 2
          and "edit:1.0" in sm["arms"] and "edit:0.0" in sm["arms"] and sm["controls_range"] is not None,
          f"2: written summary aggregates 2 samples and controls_range ({sm})")
    it0 = r["items"][0]
    check(len(it0["samples"]) == 2
          and all(s["untouched_rows_identical"] for s in it0["samples"])
          and all(next(a for a in s["arms"] if a["key"] == "edit:0.0")["same_as_base"] for s in it0["samples"]),
          "2: sampled written reading enforces untouched-row and dose-0 bit identity on every sample")
    # Different seeds produce distinct base generations (sanity check sampling actually ran)
    p0 = it0["samples"][0]["base"]["pieces"]
    p1 = it0["samples"][1]["base"]["pieces"]
    check(p0 != p1, "2: sample 0 and sample 1 draw distinct sampled trajectories")
    # Lines render cleanly with zero kept-out words
    rendered = "\n".join(lv.lines(r))
    check("sampled" in rendered and "written summary" in rendered and not KEPT_OUT_PATTERN.search(rendered),
          "2: lv.lines renders sampled written summary cleanly with zero kept-out words")
    # driver_view keeps written summary while redacting raw items
    ent = next(lb.entries(book, start=r["book"]["seq"], limit=1))
    dv = lb.driver_view(ent)
    check(dv["body"]["items"] == {"count": 1, "redacted": True}
          and dv["body"]["summary"]["total_generations"] == 2,
          "2: driver_view keeps written summary while redacting per-item text and tokens")
    # Thinking-on refusal on raw id-only item
    raw_items = {"raw": lv.item_from_ids([1, 2, 3], [4, 5], name="raw")}
    r_th_ref = lv.read(hP, {**base_wr, "items": ["raw"], "thinking": True, "samples": 1, "max_new": 6}, raw_items)
    check(r_th_ref["status"] == "refused" and "chat items" in (r_th_ref["why"] or "").lower(),
          f"2: thinking=True refuses raw id-only item ({r_th_ref['why']})")


# -------------------------------------------------------------------------- 3. stack(doses) & in-memory _plant


def check_plant_and_stack(hP: lv.Handle, items: Dict[str, lv.Item], tmp: Path) -> None:
    section("3. stack(doses=...) & in-memory _plant composition + exact repair")
    plant = we.build(hP.lm, [we.silence(3, [4, 9, 15, 22])], model_id=hP.model_id, weights_id=hP.weights_id)
    # Exact additive inverse of silence (scale by 0 -> delta = -W) is scale by 2.0 (delta = +W)
    repair = we.build(hP.lm, [we.scale(3, [4, 9, 15, 22], 2.0)], model_id=hP.model_id, weights_id=hP.weights_id)
    stacked = we.stack(hP.lm, [plant, repair], model_id=hP.model_id, weights_id=hP.weights_id,
                       doses=[1.0, 1.0], caps=False)
    check(all(float(f) == 1.0 for f in stacked.terms[0].factor),
          "3: stack([silence(0), scale(2)]) sums to exact factor 1.0 (healthy restoration)")
    # Partial dose stack check
    half = we.stack(hP.lm, [plant, repair], model_id=hP.model_id, weights_id=hP.weights_id,
                    doses=[1.0, 0.5], caps=False)
    check(all(abs(float(f) - 0.5) < 1e-6 for f in half.terms[0].factor),
          "3: stack([silence(0), scale(2)], doses=[1.0, 0.5]) sums to factor 0.5")

    book = tmp / "plant_book.jsonl"
    os.environ[lb.BOOK_ENV] = str(book)
    # Measure KL(healthy || planted) directly on healthy model
    plan_ref = {"name": "ref_plant", "kind": "paired", "edit": {"terms": [{"layer": 3, "silence": [4, 9, 15, 22]}]},
                "items": ["alpha"], "doses": [1.0], "controls": {"count": 0, "seed": 0},
                "rows": {"small": 8, "large": 16}, "folded": "never"}
    r_ref = lv.read(hP, plan_ref, items)
    # Now run repair edit under _plant=plant!
    plan_rep = {"name": "try_repair", "kind": "paired",
                "edit": {"terms": [{"layer": 3, "scale": [4, 9, 15, 22], "factor": 2.0}]},
                "items": ["alpha"], "doses": [1.0, 0.0], "controls": {"count": 2, "seed": 0},
                "rows": {"small": 8, "large": 16}, "folded": "never"}
    r_rep = lv.read(hP, plan_rep, items, _plant=plant)
    we.assert_base(hP.lm, hP.hooks_at_load)
    check(r_rep["status"] == "ok" and r_rep["base_after"] is True,
          f"3: paired reading under in-memory _plant ok and leaves base untouched ({r_rep['status']}: {r_rep['why']})")
    arm1 = next(a for a in r_rep["items"][0]["classes"]["small"]["arms"] if a["key"] == "edit:1.0")
    arm0 = next(a for a in r_rep["items"][0]["classes"]["small"]["arms"] if a["key"] == "edit:0.0")
    ref_kl = next(a for a in r_ref["items"][0]["classes"]["small"]["arms"] if a["key"] == "edit:1.0")["metrics"]["kl"]["mean"]
    check(arm0["same_as_base"] is True and arm1["metrics"]["kl"]["mean"] > 0.0 and ref_kl > 0.0,
          f"3: under _plant, dose 0 matches planted base by bits and exact repair moves KL ({arm1['metrics']['kl']['mean']:.5f} vs ref {ref_kl:.5f})")
    # Verify capture_direction under _plant differs from healthy capture and leaves base untouched
    d_healthy = lv.capture_direction(hP, [items["alpha"]], [items["beta"]], 3)
    d_planted = lv.capture_direction(hP, [items["alpha"]], [items["beta"]], 3, _plant=plant)
    we.assert_base(hP.lm, hP.hooks_at_load)
    check(float(np.linalg.norm(d_healthy - d_planted)) > 0.0,
          "3: capture_direction under _plant sees planted activations and leaves model base untouched")


# ------------------------------------------------------------------------------- 4. bwrap off-card sandbox


def check_sandbox(p_dir: Path, tmp: Path) -> None:
    section("4. Off-card bwrap sandbox (containment probe + weights workflow)")
    probe = ls.probe_sandbox(weights_dir=p_dir, max_mem_gib=1.0)
    for c in probe["checks"]:
        check(bool(c["ok"]), f"4: sandbox probe [{c['name']}] ({c['detail']})")
    # End-to-end cell using W to compute a direction vector and save_array to /work
    work = tmp / "sb_work"
    cfg = ls.SandboxConfig(work_dir=work, weights_dir=p_dir, timeout_s=20.0, max_mem_gib=1.5)
    cell_code = """
import json
info = W.info()
qm = W.quiet_mask(3)
norms = W.down_stream_norms(3)
top2 = np.argsort(-norms)[:2].tolist()
cols = W.dense_cols("layers.3.mlp.down_proj", top2).float().numpy()
meta = save_array("edit_vecs.npy", -0.25 * cols)
print("CELL_OUT:" + json.dumps({"n_layers": info["n_layers"], "top2": top2, "meta": meta}))
"""
    r = ls.run_cell(cell_code, cfg, cell_log_dir=tmp / "sb_logs" / "0001")
    check(r["ok"] is True and any(f["rel_path"] == "edit_vecs.npy" for f in r["files_written"]),
          f"4: sandbox cell reads W and saves hashed float32 array ({r['stderr'][:200]})")
    # Without weights mounted (sandbox_no_weights reach)
    cfg_now = ls.SandboxConfig(work_dir=work, weights_dir=None, timeout_s=15.0, max_mem_gib=1.0)
    r_now = ls.run_cell("print('W_IS_NONE:', W is None)", cfg_now)
    check(r_now["ok"] and "W_IS_NONE: True" in r_now["stdout"],
          "4: sandbox with weights_dir=None mounts no /weights and sets W=None")


# ------------------------------------------------------------------------- 5. Phase 2 session & blinding


def check_session(hP: lv.Handle, p_dir: Path, items: Dict[str, lv.Item], tmp: Path) -> None:
    section("5. Phase 2 Agent's Door session (reaches, budgets, gates, blinding, transcript copy)")
    book = tmp / "door_book.jsonl"
    # Policy refusals
    for label, kwargs in [
        ("reach=home refused", {"reach": "home", "backend": "local"}),
        ("codex unprobed refused", {"reach": "tools_only", "backend": "codex", "codex_probed": False}),
    ]:
        got = False
        try:
            lses.LevelSession(hP, tmp / "bad_sess", book_path=book, probe_on_start=False, **kwargs)
        except lses.SessionRefused:
            got = True
        check(got, f"5: {label}")
    # Env REACH=home refusal
    prev_reach = os.environ.get(lses.REACH_ENV)
    os.environ[lses.REACH_ENV] = "home"
    got_env = False
    try:
        lses.LevelSession(hP, tmp / "bad_env", reach="tools_only", book_path=book, probe_on_start=False)
    except lses.SessionRefused:
        got_env = True
    finally:
        if prev_reach is None:
            os.environ.pop(lses.REACH_ENV, None)
        else:
            os.environ[lses.REACH_ENV] = prev_reach
    check(got_env, f"5: ${lses.REACH_ENV}=home refuses session creation")
    # Book inside work_dir refused
    got_bk = False
    try:
        lses.LevelSession(hP, tmp / "bad_bk", reach="tools_only",
                          book_path=tmp / "bad_bk" / "work" / "book.jsonl", probe_on_start=False)
    except lses.SessionRefused:
        got_bk = True
    check(got_bk, "5: book_path inside work_dir refused")

    # Full sandbox session workflow: cell -> capture -> estimate -> read -> write -> book_view -> close
    sess_dir = tmp / "sess_main"
    budget = lses.SessionBudget(max_gpu_seconds=300.0, max_forwards=60, max_cells=5,
                                max_cell_seconds=20.0, max_wall_seconds=600.0, max_cell_mem_gib=1.5,
                                min_controls=2, min_written_controls=1)
    sess = lses.LevelSession(
        hP, sess_dir, reach="sandbox", backend="claude", budget=budget,
        book_path=book, weights_dir=p_dir, probe_on_start=True,
    )
    st = sess.status()
    check(st["harness_card"]["practice_only"] is True
          and st["harness_card"]["with_healthy_reference"] is True
          and "cell" in st["harness_card"]["allowed_tools"]
          and st["remaining"]["forwards"] == 60,
          f"5: status returns harness card and budgets ({st['harness_card']})")

    # Capture direction -> saved in work/
    cap = sess.capture([items["alpha"]], [items["beta"]], 3, out_name="contrast.npy")
    check(cap["rel_path"] == "contrast.npy" and len(cap["sha256"]) == 64
          and (sess.work_dir / "contrast.npy").is_file()
          and sess.forwards_used == 2,
          f"5: capture writes hashed direction to work/ and charges 2 forwards ({cap})")

    # Off-card cell reads W and work/contrast.npy, builds add vectors, writes add_vecs.npy
    cell_res = sess.cell("""
import json
u = np.load("/work/contrast.npy", allow_pickle=False)
top = W.writers(3, u, k=2)
neurons = [i for i, _ in top]
cols = W.dense_cols("layers.3.mlp.down_proj", neurons).float().numpy()
meta = save_array("add_vecs.npy", -0.2 * cols)
print("OUT:" + json.dumps({"neurons": neurons, "sha256": meta["sha256"]}))
""")
    check(cell_res["ok"] is True and sess.cells_used == 1, f"5: cell executes off-card and charges 1 cell ({cell_res['stderr'][:200]})")
    cell_json = json.loads([ln for ln in cell_res["stdout"].splitlines() if ln.startswith("OUT:")][0][4:])

    # Gate refuses controls.count < min_controls
    got_min = False
    try:
        sess.read({"name": "no_ctl", "kind": "paired",
                   "edit": {"terms": [{"layer": 3, "silence": cell_json["neurons"]}]},
                   "controls": {"count": 0, "seed": 0}, "rows": {"small": 8, "large": 16}}, items)
    except lses.SessionRefused:
        got_min = True
    check(got_min, "5: practice gate refuses controls.count below min_controls")

    # Gated paired reading using hashed .npy built in the sandbox cell!
    plan_p = {
        "name": "door_paired",
        "kind": "paired",
        "edit": {"terms": [{"layer": 3, "add": cell_json["neurons"], "vectors": "add_vecs.npy",
                            "sha256": cell_json["sha256"]}]},
        "items": ["alpha"],
        "doses": [1.0],
        "controls": {"family": "other_sites", "count": 2, "seed": 0},
        "rows": {"small": 8, "large": 16},
        "folded": "never",
    }
    est_r = sess.estimate(plan_p, items)
    check(est_r["fits_budget"] is True and est_r["estimate"]["forwards"] > 0, "5: estimate checks budget fit")
    rd = sess.read(plan_p, items)
    dv = rd["driver_view"]
    check(dv["body"]["status"] == "ok" and dv["body"]["items"] == {"count": 1, "redacted": True}
          and "alpha" not in json.dumps(dv) and "Orange" not in json.dumps(dv)
          and rd["forwards_charged"] > 0 and sess.gpu_seconds_used > 0,
          "5: gated read returns driver_view (no raw item text), logs book, charges forwards & GPU-seconds")

    # Gated sampled written reading
    plan_w = {
        "name": "door_written",
        "kind": "written",
        "edit": {"terms": [{"layer": 3, "silence": cell_json["neurons"]}]},
        "items": ["alpha"],
        "rows": 7,
        "max_new": 10,
        "controls": {"count": 1, "seed": 0},
        "sampler": {"temperature": 0.8, "top_p": 0.9, "seed": 3},
        "samples": 2,
    }
    wr_res = sess.write(plan_w, items)
    check(wr_res["driver_view"]["body"]["status"] == "ok"
          and wr_res["driver_view"]["body"]["items"] == {"count": 1, "redacted": True}
          and wr_res["driver_view"]["body"]["summary"]["total_generations"] == 2,
          "5: gated write returns written summary in driver_view with raw items redacted")

    # book_view redacted
    bv = sess.book_view(tail=5)
    check(bv["redacted"] is True and len(bv["entries"]) == 2
          and all(e["body"]["items"]["redacted"] is True for e in bv["entries"]),
          "5: book_view returns redacted driver_view entries only")
    # a trial registered under trial-4, a rule the owner never adopted, is shown to a session that is not blinded
    # with its rule and the sentence said of it, its verdict line too (which names its registration's line alone)
    t4_book = tmp / "trial4_door_book.jsonl"
    t4_common = {"brief_name": "occ", "brief_id": "b" * 64, "edit_id": "e" * 64}
    t4_seqs: Dict[str, int] = {}
    for rule in ("trial-4", "trial-3"):
        r_ent = lb.append("note", {**t4_common, "note_type": "trial_registration", "trial_name": rule,
                                   "of_record": True, "rule_preset": rule, "main_dose": 1.0}, path=t4_book)
        t4_seqs[rule] = int(r_ent["seq"])
        lb.append("note", {**t4_common, "note_type": "trial_of_record", "trial_name": rule, "verdict": "passed",
                           "registration_seq": int(r_ent["seq"]), "of_record": True}, path=t4_book)
    s_t4 = lses.LevelSession(hP, tmp / "sess_trial4", reach="tools_only", backend="openai",
                             budget=lses.SessionBudget(max_forwards=1, max_cells=0), book_path=t4_book,
                             probe_on_start=False)
    bv_t4 = {e["seq"]: e for e in s_t4.book_view(tail=10)["entries"]}
    bv_one = s_t4.book_view(seq=t4_seqs["trial-4"] + 1)["entries"][0]
    s_t4.close()
    note4 = lcl.not_adopted_note("trial-4")
    check(note4 is not None
          and bv_t4[t4_seqs["trial-4"]].get("rule_note") == note4
          and bv_t4[t4_seqs["trial-4"]]["rule_preset"] == "trial-4"
          and bv_t4[t4_seqs["trial-4"] + 1].get("rule_note") == note4 and bv_one.get("rule_note") == note4
          and "rule_note" not in bv_t4[t4_seqs["trial-3"]] and "rule_note" not in bv_t4[t4_seqs["trial-3"] + 1],
          f"5: book_view names a trial-4 trial's rule on its registration and its verdict line ({note4}); a trial-3 "
          "trial's lines carry no such note")

    # Close copies transcript into <book_dir>/transcripts/<session_id>.jsonl and blocks further calls
    closed = sess.close()
    t_copy = Path(closed["transcript_copy"])
    check(t_copy.is_file() and t_copy.parent == book.parent / "transcripts"
          and len(t_copy.read_text(encoding="utf-8").splitlines()) >= 4,
          f"5: close copies transcript to book's transcripts/ folder ({t_copy})")
    got_closed = False
    try:
        sess.book_view()
    except lses.SessionRefused:
        got_closed = True
    check(got_closed, "5: closed session refuses further calls")

    # Budget exhaustion check (max_forwards=1 refuses capture of 2 items)
    sess_tiny = lses.LevelSession(
        hP, tmp / "sess_tiny", reach="tools_only", backend="openai",
        budget=lses.SessionBudget(max_forwards=1, max_cells=0),
        book_path=book, probe_on_start=False,
    )
    check("cell" not in sess_tiny.allowed_tools(), "5: tools_only reach omits cell tool")
    got_fwd_cap = False
    try:
        sess_tiny.capture([items["alpha"]], [items["beta"]], 3)
    except lses.SessionRefused:
        got_fwd_cap = True
    check(got_fwd_cap, "5: forward budget cap refuses over-budget capture")
    sess_tiny.close()

    # Planted vs Canary blinding check (arbiter.md row 33 + review leftover: predicted seconds & lower="whole")
    plant = we.build(hP.lm, [we.silence(0, [4, 9])], model_id=hP.model_id, weights_id=hP.weights_id)
    s_plant = lses.LevelSession(
        hP, tmp / "sess_planted", reach="sandbox_no_weights", backend="claude",
        budget=lses.SessionBudget(min_controls=1), book_path=book, probe_on_start=False, plant=plant,
    )
    s_canary = lses.LevelSession(
        hP, tmp / "sess_canary", reach="sandbox_no_weights", backend="claude",
        budget=lses.SessionBudget(min_controls=1), book_path=book, probe_on_start=False, canary=True,
    )
    c_p = {k: v for k, v in s_plant.harness_card().items() if k != "session_id"}
    c_c = {k: v for k, v in s_canary.harness_card().items() if k != "session_id"}
    check(c_p == c_c and c_p["with_healthy_reference"] is False,
          "5: planted and canary sessions present identical blinded harness cards (sandbox_no_weights)")
    # Edit at layer 4 while hidden plant is at layer 0, with multi-forward controls so restart-layer leak would show
    blind_plan = {
        "name": "blind_read", "kind": "paired", "edit": {"terms": [{"layer": 4, "scale": [4, 9], "factor": 2.0}]},
        "items": ["alpha"], "controls": {"count": 6, "seed": 0}, "rows": {"small": 8, "large": 16},
    }
    ep_blind = s_plant.estimate(blind_plan, items)
    ec_blind = s_canary.estimate(blind_plan, items)
    t_p0 = time.perf_counter()
    rp_blind = s_plant.read(blind_plan, items)
    t_p1 = time.perf_counter()
    rc_blind = s_canary.read(blind_plan, items)
    t_c1 = time.perf_counter()
    bv_plant = s_plant.book_view(tail=4)["entries"]
    t_k0 = time.perf_counter()
    cap_p = s_plant.capture([items["alpha"]], [items["beta"]], 2)
    t_k1 = time.perf_counter()
    cap_c = s_canary.capture([items["alpha"]], [items["beta"]], 2)
    # a reading the kernel refuses (a layer the model lacks): priced at nothing, and its reply waits all the same
    t_r0 = time.perf_counter()
    rr_blind = s_plant.read({**blind_plan, "name": "blind_refused",
                             "edit": {"terms": [{"layer": 99, "scale": [4], "factor": 2.0}]}}, items)
    t_r1 = time.perf_counter()
    we.assert_base(hP.lm, hP.hooks_at_load)
    s_plant.close()
    s_canary.close()
    # where a planted reading scales its null rows: the layer the null program names (the edit's), plant or no plant
    mod_names = {id(m): n for n, m in hP.lm.named_modules()}
    null_on: List[str] = []
    hook_was = torch.nn.Module.register_forward_hook

    def _spy(self: Any, hook: Any, *a: Any, **k: Any) -> Any:
        if getattr(hook, "__name__", "") == "_null_post":
            null_on.append(mod_names.get(id(self), "?"))
        return hook_was(self, hook, *a, **k)

    torch.nn.Module.register_forward_hook = _spy
    try:
        r_low = lv.read(hP, {**blind_plan, "name": "null_place", "controls": {"count": 2, "seed": 0}, "folded": "never",
                             "lower": "whole"}, items, _plant=plant)
    finally:
        torch.nn.Module.register_forward_hook = hook_was
    check(r_low["status"] == "ok" and null_on and set(null_on) == {"layers.4.mlp.down_proj"},
          f"5: with the plant at layer 0 and the edit at layer 4, null rows are scaled at the edit's layer ({sorted(set(null_on))})")
    # Neither session.json, transcript.jsonl, nor driver_view leaks plant/canary secrets
    disk_dump = (
        (tmp / "sess_planted" / "session.json").read_text()
        + (tmp / "sess_planted" / "transcript.jsonl").read_text()
        + json.dumps(rp_blind)
        + (tmp / "sess_canary" / "session.json").read_text()
    ).lower()
    check(plant.id[:12] not in disk_dump and "canary" not in disk_dump and "plant" not in disk_dump,
          "5: planted/canary state stays strictly in memory (zero leakage to session.json, transcript, or driver_view)")
    check(rp_blind["driver_view"]["body"]["plan"]["folded"] == "never"
          and rc_blind["driver_view"]["body"]["plan"]["folded"] == "never"
          and rp_blind["driver_view"]["body"]["plan"]["lower"] == "whole"
          and rc_blind["driver_view"]["body"]["plan"]["lower"] == "whole",
          "5: both planted and canary sessions enforce folded=never and lower=whole (no twin or restart side-channel)")
    p_pred = rp_blind["driver_view"]["body"]["seconds"]["predicted"]
    c_pred = rc_blind["driver_view"]["body"]["seconds"]["predicted"]
    p_parts = rp_blind["driver_view"]["body"]["seconds"]["parts"]
    c_parts = rc_blind["driver_view"]["body"]["seconds"]["parts"]
    check(ep_blind["estimate"]["seconds"] == ec_blind["estimate"]["seconds"] == p_pred == c_pred
          and ep_blind["estimate"]["parts"] == ec_blind["estimate"]["parts"] == p_parts == c_parts,
          f"5: blinded estimate and reading predicted seconds/parts match identically across plant & canary ({p_pred} vs {c_pred})")
    dv_p, dv_c = rp_blind["driver_view"]["body"], rc_blind["driver_view"]["body"]
    check(all(dv["seconds"]["actual"] is None and dv["seconds"]["ratio"] is None and dv.get("peak_gib") is None
              for dv in (dv_p, dv_c)),
          "5: a blinded session is shown nothing measured of a reading's cost (seconds, ratio, peak)")
    check(rp_blind["gpu_seconds_charged"] == rc_blind["gpu_seconds_charged"] == round(float(p_pred), 3)
          and {k: v for k, v in rp_blind["remaining"].items() if "wall" not in k}
          == {k: v for k, v in rc_blind["remaining"].items() if "wall" not in k}
          and cap_p["seconds"] == cap_c["seconds"],
          f"5: planted and canary sessions are charged the same price for the same calls ({rp_blind['gpu_seconds_charged']} s a reading)")
    own_seq = rp_blind["book"]["seq"]
    check(any(e.get("seq") == own_seq and "body" in e and e["body"]["seconds"]["actual"] is None for e in bv_plant)
          and any(e.get("redacted") == "another session's line" for e in bv_plant)
          and all("body" not in e for e in bv_plant if e.get("seq") != own_seq),
          "5: a blinded session's book view shows its own lines, without measured costs, and no other session's")
    check(next(lb.entries(book, start=own_seq, limit=1))["body"]["seconds"]["actual"] is not None,
          "5: the book itself keeps the measured seconds of a blinded session's reading")
    # what `_unmeasured` takes out, on a row that holds every one of them (a reading on the CPU has no peak to hide)
    shown = lses.LevelSession._unmeasured({"seq": 1, "body": {"seconds": {"actual": 1.5, "ratio": 0.9, "predicted": 1.7,
                                                                           "parts": {"whole": 1.7}}, "peak_gib": 3.25}})
    check(shown["body"]["seconds"]["actual"] is None and shown["body"]["seconds"]["ratio"] is None
          and shown["body"]["peak_gib"] is None and shown["body"]["seconds"]["predicted"] == 1.7
          and shown["body"]["seconds"]["parts"] == {"whole": 1.7}
          and shown["body"]["seconds"]["measured"] == "not shown in a blinded session",
          "5: the measured seconds, their ratio and the peak of memory are taken out of a blinded view; the price stays")
    # when a blinded call replies: no sooner than its price says, plant or no plant, and a refused one after the same
    # short wait whatever the plant cost it
    wait_read = lses.BLIND_PAD_FACTOR * float(p_pred) + lses.BLIND_PAD_OVER_S
    wait_cap = lses.BLIND_PAD_FACTOR * float(cap_p["seconds"]) + lses.BLIND_PAD_OVER_S
    check(t_p1 - t_p0 >= wait_read - 0.005 and t_c1 - t_p1 >= wait_read - 0.005 and t_k1 - t_k0 >= wait_cap - 0.005
          and rr_blind["driver_view"]["body"]["status"] == "refused" and rr_blind["gpu_seconds_charged"] == 0.0
          and t_r1 - t_r0 >= lses.BLIND_PAD_OVER_S - 0.005,
          f"5: a blinded call's reply waits for its price ({wait_read:.2f} s a reading here; the planted one came after "
          f"{t_p1 - t_p0:.2f} s, the canary's after {t_c1 - t_p1:.2f} s), and a refused one for the wait over")
    s_open = lses.LevelSession(hP, tmp / "sess_open", reach="tools_only", backend="claude",
                               budget=lses.SessionBudget(min_controls=1), book_path=book, probe_on_start=False)
    t_o0 = time.perf_counter()
    s_open._wait_out(t_o0, 30.0)
    check(time.perf_counter() - t_o0 < 0.5, "5: a session that is not blinded waits for nothing")
    s_open.close()

    # a plant changes nothing about what is refused. An edit of 16 terms that leaves the plant's linear alone is 17
    # with it; an edit that sets the very column a plant set meets a set there. Both read with the plant under them
    # as they read alone, where a caller's own stack of two such edits is still refused.
    kw = dict(model_id=hP.model_id, weights_id=hP.weights_id)
    wide = we.build(hP.lm, [we.silence(L, [1]) for L in range(1, 6)]
                    + [we.silence_rows(L, w, [1]) for L in range(6) for w in ("gate_proj", "up_proj")][:11], **kw)
    d_out = we.linear_shape(hP.lm.layers[0].mlp.down_proj)[0]
    plant_set = we.build(hP.lm, [we.set_columns(0, [4], np.full((1, d_out), 0.01, dtype=np.float32))], **kw)
    v_mine = np.full((1, d_out), -0.02, dtype=np.float32)
    same_col = we.build(hP.lm, [we.set_columns(0, [4], v_mine)], **kw)

    def rd(e: Any, pl: Any, name: str) -> Dict[str, Any]:
        return lv.read(hP, {"name": name, "kind": "paired", "edit": {"built": e.id}, "items": ["alpha"],
                            "controls": {"count": 2, "seed": 0}, "rows": {"small": 8, "large": 16}, "folded": "never",
                            "lower": "whole"}, items, edit=e, _plant=pl)

    def stack_refused(pair: Any, says: str) -> bool:
        try:
            we.stack(hP.lm, pair, **kw)
        except we.Refused as exc:
            return says in str(exc)
        return False

    rw_p, rw_c = rd(wide, plant, "wide_under_a_plant"), rd(wide, None, "wide_alone")
    rs_p, rs_c = rd(same_col, plant_set, "set_under_a_plant"), rd(same_col, None, "set_alone")
    check(len(wide.terms) == we.MAX_TERMS and "layers.0.mlp.down_proj" not in wide.linears()
          and rw_p["status"] == rw_c["status"] == "ok" and stack_refused([plant, wide], "at most 16"),
          f"5: an edit of {we.MAX_TERMS} terms reads with a plant under it as it reads alone ({rw_p['status']}: "
          f"{rw_p.get('why')}); what a planted session refuses does not say where its plant is")
    whole = we.stack(hP.lm, [plant_set, same_col], caps=False, plant_first=True, **kw)
    check(rs_p["status"] == rs_c["status"] == "ok" and stack_refused([plant_set, same_col], "both set")
          and len(whole.terms) == 1 and int(whole.terms[0].op[0]) == 1
          and np.array_equal(whole.terms[0].vectors[0], v_mine[0]),
          f"5: an edit that sets the column a plant set reads as it reads alone ({rs_p['status']}: {rs_p.get('why')}), "
          "and its own set is the one that stands")
    # the cap on a factor: a plant's own factor adds to the driver's, and a dose stretches a factor, and neither
    # may make a planted reading refuse what a canary reads (one such call told the two apart, and gave the plant's factor)
    plant_f8 = we.build(hP.lm, [we.scale(2, [7], 8.0)], **kw)
    big = we.build(hP.lm, [we.scale(4, [4], we.MAX_FACTOR)], **kw)
    on_plant = we.build(hP.lm, [we.scale(2, [7], 60.0)], **kw)

    def rd_dose(e: Any, pl: Any, name: str, dose: float) -> str:
        try:
            r = lv.read(hP, {"name": name, "kind": "paired", "edit": {"built": e.id}, "items": ["alpha"], "doses": [dose, 0.0],
                             "main_dose": dose, "controls": {"count": 2, "seed": 0}, "rows": {"small": 8, "large": 16},
                             "folded": "never", "lower": "whole"}, items, edit=e, _plant=pl)
            return str(r["status"])
        except Exception as exc:  # noqa: BLE001
            return f"raised {type(exc).__name__}: {str(exc)[:80]}"

    # a control edit is gated a second time only under a plant: the tests on an edit's size must not be asked of it
    # then (a same-site control of 8 silenced columns is 8 added columns; with the cap on those set to 4 for the check)
    wide_cols = we.build(hP.lm, [we.silence(4, list(range(8)))], **kw)
    cap_was = we.MAX_VECTOR_COLUMNS
    we.MAX_VECTOR_COLUMNS = 4
    try:
        got_ctl = []
        for i, pl in enumerate((plant, None)):
            try:
                r = lv.read(hP, {"name": f"same_site_under_{i}", "kind": "paired", "edit": {"built": wide_cols.id}, "items": ["alpha"],
                                 "controls": {"family": "same_site", "count": 2, "seed": 0}, "rows": {"small": 8, "large": 16},
                                 "folded": "never", "lower": "whole"}, items, edit=wide_cols, _plant=pl)
                got_ctl.append(str(r["status"]))
            except Exception as exc:  # noqa: BLE001
                got_ctl.append(f"raised {type(exc).__name__}: {str(exc)[:80]}")
    finally:
        we.MAX_VECTOR_COLUMNS = cap_was
    check(got_ctl[0] == got_ctl[1] == "ok",
          f"5: a same-site control edit past a cap on an edit's size reads under a plant as it reads without ({got_ctl})")
    got_big = [rd_dose(big, pl, f"factor_cap_{i}", 1.01) for i, pl in enumerate((plant, None))]
    got_on = [rd_dose(on_plant, pl, f"on_the_plant_{i}", 1.0) for i, pl in enumerate((plant_f8, None))]
    check(got_big[0] == got_big[1] and not got_big[0].startswith("raised") and got_on[0] == got_on[1]
          and not got_on[0].startswith("raised") and stack_refused([plant_f8, on_plant], "at most 64"),
          f"5: a scaling at the cap read at dose 1.01, and a scaling of a neuron the plant scaled, read with a plant as "
          f"without ({got_big}; {got_on})")
    we.assert_base(hP.lm, hP.hooks_at_load)


# ------------------------------------------------------------------------------- 6. MCP stdio protocol


def check_mcp(hP: lv.Handle, p_dir: Path, items: Dict[str, lv.Item], tmp: Path) -> None:
    section("6. Phase 2 stdio MCP server protocol (backend/level_mcp.py)")
    book = tmp / "mcp_book.jsonl"
    sess = lses.LevelSession(
        hP, tmp / "sess_mcp", reach="tools_only", backend="claude",
        budget=lses.SessionBudget(min_controls=1), book_path=book, probe_on_start=False,
    )
    try:
        init = lmcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, sess)
        check(init is not None and init["result"]["serverInfo"]["name"] == "theodolite-levelling",
              "6: MCP initialize returns serverInfo")
        tlist = lmcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, sess)
        names = [t["name"] for t in tlist["result"]["tools"]]
        check("cell" not in names and set(names) == {"status", "estimate", "book_view", "capture", "read", "write"},
              f"6: MCP tools/list respects tools_only filtered tool list ({names})")
        call_st = lmcp.handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                       "params": {"name": "status", "arguments": {}}}, sess)
        st_obj = json.loads(call_st["result"]["content"][0]["text"])
        check(not call_st["result"].get("isError") and st_obj["session_id"] == sess.session_id,
              "6: MCP tools/call status succeeds")
        call_bad = lmcp.handle_request({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                        "params": {"name": "cell", "arguments": {"code": "print(1)"}}}, sess)
        check(call_bad["result"].get("isError") is True,
              "6: MCP tools/call refuses disallowed cell tool in tools_only reach")
        unk = lmcp.handle_request({"jsonrpc": "2.0", "id": 5, "method": "bogus/method"}, sess)
        check(unk["error"]["code"] == -32601, "6: MCP unknown method returns -32601")
        check(lmcp.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}, sess) is None,
              "6: MCP notification returns None")
    finally:
        sess.close()

    # propose_repair at the door: `edit` is required (null to abstain), each argument is type-checked, and a call the
    # door refuses spends nothing (at 55193dd an empty call was recorded as an explicit abstention and used up the
    # session's one submission, and a null diagnosis was written as the string "None")
    bbook = tmp / "mcp_blind_book.jsonl"
    sb = lses.LevelSession(
        hP, tmp / "sess_mcp_blind", reach="tools_only", backend="claude",
        budget=lses.SessionBudget(min_controls=1), book_path=bbook, probe_on_start=False, canary=True,
    )
    try:
        specs = {t["name"]: t for t in lmcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, sb)["result"]["tools"]}
        pr_schema = (specs.get("propose_repair") or {}).get("inputSchema") or {}
        check(pr_schema.get("required") == ["edit"]
              and pr_schema["properties"]["edit"]["type"] == ["object", "null"]
              and pr_schema["properties"]["diagnosis"]["type"] == ["string", "null"]
              and "default" not in pr_schema["properties"]["diagnosis"],
              f"6: a blinded session's propose_repair requires edit (null allowed) and takes a string or null diagnosis "
              f"({pr_schema.get('required')})")

        def _pr(args: Any, i: int) -> Tuple[bool, str]:
            got = lmcp.handle_request({"jsonrpc": "2.0", "id": 10 + i, "method": "tools/call",
                                       "params": {"name": "propose_repair", "arguments": args}}, sb)
            return bool(got["result"].get("isError")), str(got["result"]["content"][0]["text"])

        refused_calls = [
            ({}, "needs the argument edit"),
            ({"diagnosis": "nothing wrong"}, "needs the argument edit"),
            ({"edit": "layer 3, neuron 4"}, "edit must be an edit recipe object"),
            ({"edit": [1, 2]}, "edit must be an edit recipe object"),
            ({"edit": None, "diagnosis": 7}, "diagnosis must be a string"),
            ({"edit": None, "confidence": "high"}, "confidence must be a number"),
            ({"edit": None, "confidence": True}, "confidence must be a number"),
            ({"edit": {"terms": [{"layer": 99, "silence": [1]}]}}, "refused by the edit gate"),
            ([1, 2], "takes its arguments as a JSON object"),
        ]
        bad_refusals: List[str] = []
        lines_before_refusals = lb.verify(bbook)["entries"] if bbook.exists() else 0
        for i, (args, needle) in enumerate(refused_calls):
            is_err, text = _pr(args, i)
            if not (is_err and needle in text):
                bad_refusals.append(f"{args}: {text[:120]}")
        lines_after_refusals = lb.verify(bbook)["entries"] if bbook.exists() else 0
        check(not bad_refusals and sb.proposal_submitted is False and sb.proposal is None
              and lines_after_refusals == lines_before_refusals,
              f"6: propose_repair refuses, in plain words, a call without edit, a wrong type, or an edit the gate "
              f"refuses, and none of them spends the submission ({bad_refusals[:2]})")
        # every refused propose_repair is kept in the door's words and logged in the transcript, so a session with no
        # proposal recorded says whether one was tried and why the door would not take it
        kept_refusals = sb.proposal_refusals
        tr_ref = [json.loads(line) for line in (tmp / "sess_mcp_blind" / "transcript.jsonl").read_text().splitlines()
                  if line.strip() and json.loads(line).get("event") == "propose_repair_refused"]
        check(kept_refusals["count"] == len(refused_calls) and len(kept_refusals["kept"]) == len(refused_calls)
              and all(needle in str(k.get("why")) for k, (_a, needle) in zip(kept_refusals["kept"], refused_calls))
              and len(tr_ref) == len(refused_calls),
              f"6: each refused propose_repair is kept in the door's own words and logged ({kept_refusals['count']} "
              f"kept, {len(tr_ref)} logged)")
        is_err_ok, text_ok = _pr({"edit": None, "diagnosis": None}, 50)
        notes = [e.get("body") or {} for e in lb.entries(bbook)
                 if (e.get("body") or {}).get("note_type") == "blinded_proposal"]
        tr_rows = [json.loads(line) for line in (tmp / "sess_mcp_blind" / "transcript.jsonl").read_text().splitlines()
                   if line.strip()]
        tr_pr = [r for r in tr_rows if r.get("event") == "propose_repair"]
        check(not is_err_ok and json.loads(text_ok).get("abstained") is True and len(notes) == 1
              and notes[0].get("abstained") is True and notes[0].get("diagnosis") is None
              and (sb.proposal or {}).get("diagnosis") is None and len(tr_pr) == 1
              and (tr_pr[0].get("payload") or tr_pr[0]).get("diagnosis") is None,
              f"6: edit null with a null diagnosis is the explicit abstention, its diagnosis kept as null in the book, "
              f"the transcript and the record ({[n.get('diagnosis') for n in notes]})")
        is_err_2, text_2 = _pr({"edit": None}, 51)
        check(is_err_2 and "already been submitted" in text_2 and len(
              [e for e in lb.entries(bbook) if (e.get("body") or {}).get("note_type") == "blinded_proposal"]) == 1
              and sb.proposal_refusals["count"] == len(refused_calls) + 1
              and "already been submitted" in str(sb.proposal_refusals["kept"][-1].get("why")),
              "6: a second propose_repair after the one the door took is refused, and kept as refused")
        is_err_rd, text_rd = (lambda g: (bool(g["result"].get("isError")), str(g["result"]["content"][0]["text"])))(
            lmcp.handle_request({"jsonrpc": "2.0", "id": 60, "method": "tools/call",
                                 "params": {"name": "read", "arguments": {}}}, sb))
        check(is_err_rd and "needs the arguments plan, items" in text_rd,
              f"6: any tool's required arguments are asked for in plain words before anything runs ({text_rd[:90]})")
    finally:
        sb.close()

    # The card: the server takes its lock and refuses to load the model beside another holder
    import socket
    import subprocess

    gib = 2 ** 30
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    busy_port = busy.getsockname()[1]
    spare = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    spare.bind(("127.0.0.1", 0))
    free_port = spare.getsockname()[1]
    spare.close()
    pod_was = os.environ.get(lmcp.POD_ROOT_ENV)
    try:
        why_port = lmcp.card_refusal(busy_port, cuda=lambda: (True, 30 * gib))
        check(why_port is not None and f"Something answers on :{busy_port}" in why_port,
              f"6: the card is refused while something answers on the page server's port ({why_port})")
        why_none = lmcp.card_refusal(free_port, cuda=lambda: (False, 0))
        check(why_none is not None and "No CUDA device" in why_none, "6: the card is refused when no CUDA device answers")
        why_full = lmcp.card_refusal(free_port, cuda=lambda: (True, 10 * gib))
        check(why_full is not None and "Only 10.0 GiB of the card is free (22 are asked" in why_full,
              f"6: the card is refused when a census or anything else has taken its memory ({why_full})")
        check(lmcp.card_refusal(free_port, cuda=lambda: (True, 30 * gib)) is None,
              "6: a free card with no server on the port is not refused")

        lock_p = tmp / "card" / "gpu.lock"
        first = lmcp.hold_card(lock_p)
        second_refused = False
        try:
            lmcp.hold_card(lock_p)
        except SystemExit as exc:
            second_refused = "one at a time" in str(exc)
        check(second_refused, "6: a second holder of the card's lock is refused in plain words, without waiting")
        first.close()
        again = lmcp.hold_card(lock_p)
        again.close()
        check(True, "6: the card's lock is free again once its holder ends")
        (tmp / "afile").write_text("not a folder", encoding="utf-8")
        plain_refusal = False
        try:
            lmcp.hold_card(tmp / "afile" / "gpu.lock")
        except SystemExit as exc:
            plain_refusal = "cannot be opened" in str(exc)
        check(plain_refusal, "6: a lock file that cannot be opened is a plain refusal, not a traceback")

        os.environ[lmcp.POD_ROOT_ENV] = str(tmp / "pod")
        on_pod = lmcp.card_lock_path()
        os.environ.pop(lmcp.POD_ROOT_ENV)
        at_home = lmcp.card_lock_path()
        check(on_pod == tmp / "pod" / "logs" / "gpu.lock" and at_home == lb.default_book_path().parent / "gpu.lock",
              "6: the card's lock is the pod's own logs/gpu.lock on a pod, and beside the level book at home")
        # what card_lock_path says of the page server is what backend/server.py does (at 55193dd the server took this
        # lock and the docstring still said it did not)
        srv_src = (ROOT_DIR / "backend" / "server.py").read_text(encoding="utf-8")
        lock_doc = " ".join(str(lmcp.card_lock_path.__doc__).split())
        check("from backend.level_mcp import hold_card" in srv_src and "card_lock = hold_card()" in srv_src
              and "The page server takes this same lock" in lock_doc and "THEODOLITE_SKIP_CARD_LOCK" in lock_doc
              and "THEODOLITE_SKIP_CARD_LOCK" in srv_src and "does not" not in lock_doc,
              "6: card_lock_path says the page server takes the same lock, as backend/server.py does")

        # the real start, with no stand-in: refused at the port before anything is loaded, the lock taken and let go
        env = {**os.environ, lmcp.POD_ROOT_ENV: str(tmp / "pod"), "HF_HUB_OFFLINE": "1"}
        p_srv = subprocess.run(
            [sys.executable, "-I", str(ROOT_DIR / "backend" / "level_mcp.py"), "--session-dir", str(tmp / "sess_card"),
             "--book", str(tmp / "card_book.jsonl"), "--reach", "tools_only", "--no-probe", "--port", str(busy_port)],
            cwd=str(ROOT_DIR), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        )
        check(p_srv.returncode != 0 and f"Something answers on :{busy_port}" in p_srv.stderr
              and (tmp / "pod" / "logs" / "gpu.lock").exists() and not (tmp / "sess_card").exists(),
              f"6: the server started with no stand-in stops at the busy port before loading anything "
              f"(rc={p_srv.returncode}, err={p_srv.stderr[-200:]!r})")
        after = lmcp.hold_card(tmp / "pod" / "logs" / "gpu.lock")
        after.close()
        check(True, "6: the refused server let the card's lock go")
    finally:
        busy.close()
        if pod_was is None:
            os.environ.pop(lmcp.POD_ROOT_ENV, None)
        else:
            os.environ[lmcp.POD_ROOT_ENV] = pod_was


# ----------------------------------------------------------------------- 7. CLI & kept-out words audit


def check_cli_and_words(p_dir: Path, tmp: Path) -> None:
    section("7. CLI (weights + sandbox-probe) & kept-out words audit")
    r_info = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "levelling.py"), "weights", "info", "--path", str(p_dir)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    check(r_info.returncode == 0 and json.loads(r_info.stdout)["n_layers"] == 6,
          f"7: CLI weights info succeeds on stand-in safetensors (exit {r_info.returncode})")
    r_pin = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "levelling.py"), "weights", "pin",
         "--path", str(p_dir), "--linear", "layers.3.mlp.down_proj"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    check(r_pin.returncode == 0 and len(json.loads(r_pin.stdout)["pin"]) == 64,
          f"7: CLI weights pin returns 64-hex pin (exit {r_pin.returncode})")
    r_probe = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "levelling.py"), "sandbox-probe",
         "--weights-dir", str(p_dir), "--max-mem-gib", "1.0"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    check(r_probe.returncode == 0 and json.loads(r_probe.stdout)["ok"] is True,
          f"7: CLI sandbox-probe exits 0 with ok=True (exit {r_probe.returncode}: {r_probe.stderr[-200:]})")

    source_files = [
        "backend/weight_edits.py", "backend/levelling.py", "backend/level_book.py",
        "backend/weight_reader.py", "backend/level_sandbox.py", "backend/level_session.py",
        "backend/level_mcp.py", "scripts/levelling.py",
    ]
    for rel in source_files:
        txt = (ROOT_DIR / rel).read_text(encoding="utf-8")
        hits = KEPT_OUT_PATTERN.findall(txt)
        check(not hits, f"7: zero kept-out words in {rel} (found {hits[:5]})")


def main() -> int:
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory(prefix="check_level_door_") as td:
        tmp = Path(td)
        Pb, hP = _build_packed_stand_in()
        _rtF, hF = _build_float_stand_in()
        items = _make_items(hP.tokenizer)
        p_dir = check_weight_reader(Pb, hP, hF, tmp)
        check_written_extended(Pb, hP, items, tmp)
        check_plant_and_stack(hP, items, tmp)
        check_sandbox(p_dir, tmp)
        check_session(hP, p_dir, items, tmp)
        check_mcp(hP, p_dir, items, tmp)
        check_cli_and_words(p_dir, tmp)
    dt = time.perf_counter() - t0
    if FAIL:
        print(f"\nlevel_door check: {len(FAIL)} FAILED of {PASS + len(FAIL)} ({dt:.1f} s)")
        return 1
    print(f"\nlevel_door check: all {PASS} checks passed ({dt:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
