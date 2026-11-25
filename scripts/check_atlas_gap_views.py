#!/usr/bin/env python3
"""CPU-only FastAPI check for hardened supplementary atlas routes, publisher, and contracts.

Exercises exact production folder layout using ONLY ``THEODOLITE_ATLAS_DIR`` (no override of
``THEODOLITE_CENSUS_DIR``):
  - ``atlas/census/runs/`` holds a regular census run
  - ``atlas/census/{forced_odd,input_census}/`` are populated exclusively via ``publish_gap_census.publish_regimes``

Verifies: empty state, destination guards, dry-run verify-only refusal, publish + MANIFEST.json,
overwrite refusal vs ``--force`` republish cache invalidation, ``validate_input_census_npz``,
removal of ``fdr_rescue`` / ``/census/regime/search`` / ``circuit_mediated``,
clean optional ``/hover`` ``regime`` summary (exploratory, no ``fdr_rescue``, model-checked, layer-aware stale),
``L-1`` last-layer ``own_lens_match is None``, causal offsets ``(-1, 0)`` vs separate ``next_pieces``,
``ablation_sampled`` naming, malformed JSON & corrupt NPZ resilience (HTTP 200 + refused map),
model & weights gating, and layer-aware vault-edit staleness.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch

TMP = Path(tempfile.mkdtemp(prefix="atlas-gap-views-"))
os.environ["THEODOLITE_ATLAS_DIR"] = str(TMP / "atlas")
os.environ["THEODOLITE_FIELD_NOTES"] = str(TMP / "notes" / "field_notes.json")

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi.testclient import TestClient  # noqa: E402

import backend.atlas as atlas_routes  # noqa: E402
from atlas_stand_in import bind_server, stand_in  # noqa: E402
from backend import atlas_gap_views as agv  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend.causal_check import language_model  # noqa: E402
from backend.larql_vault import PatchOperationRecord  # noqa: E402
from backend.runtime_factory import weights_card  # noqa: E402
from scripts import publish_gap_census as pgc  # noqa: E402

BASE = "http://127.0.0.1:8765"
failures: List[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def _plant_regular_census_run(runs_dir: Path, model_id: str) -> str:
    runs_dir.mkdir(parents=True, exist_ok=True)
    run = runs_dir / "L0_main_rel.jsonl"
    row = {"kind": "baseline", "neuron": -1, "prompt": "who", "dose": 0.0,
           "text": "plain text", "token_ids": [1, 2], "logprob": [-0.1, -0.1], "finish": "stop"}
    run.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cz.meta_path_of(run).write_text(
        json.dumps({
            "layer": 0,
            "model": model_id,
            "design_name": "main_rel",
            "prompts": cz.PROMPTS,
            "version": cz.CENSUS_VERSION,
            "doses_from": cz.DOSES_FROM["main_rel"],
            "p50_source": "stand-in",
        }),
        encoding="utf-8",
    )
    return run.name


def _write_fo_source(
    fo_whole: Path,
    layer: int,
    N: int,
    *,
    model_id: str,
    weights: Dict[str, Any],
    t_pos1: int,
    t_pos2: int,
    t_neg1: int,
    t_neg2: int,
    t_fail: int,
    score_boost: float = 0.0,
) -> None:
    fo_whole.mkdir(parents=True, exist_ok=True)
    T_col = np.zeros(N, dtype=np.float32)
    T_iso = np.zeros(N, dtype=np.float32)
    r_ab_res = np.zeros(N, dtype=np.float32)
    r_ba_res = np.zeros(N, dtype=np.float32)
    named05 = np.zeros(N, dtype=bool)
    named10 = np.zeros(N, dtype=bool)
    is_direct = np.zeros(N, dtype=bool)
    q_direct = np.ones(N, dtype=np.float32)
    kl_mean = np.full(N, 0.04, dtype=np.float32)
    alpha_abs = np.full(N, 0.02, dtype=np.float32)

    topA_p = np.zeros((N, 5), dtype=np.int32)
    topB_p = np.zeros((N, 5), dtype=np.int32)
    topA_n = np.zeros((N, 5), dtype=np.int32)
    topB_n = np.zeros((N, 5), dtype=np.int32)
    posA_onB = np.zeros((N, 5), dtype=np.float32)
    posB_onA = np.zeros((N, 5), dtype=np.float32)
    negA_onB = np.zeros((N, 5), dtype=np.float32)
    negB_onA = np.zeros((N, 5), dtype=np.float32)

    n_pass_dir, n_pass_other, n_fail = 2, 5, 7
    T_col[n_pass_dir], T_iso[n_pass_dir] = 6.5 + score_boost, 5.8 + score_boost
    r_ab_res[n_pass_dir], r_ba_res[n_pass_dir] = 2.9, 3.1
    named05[n_pass_dir] = named10[n_pass_dir] = is_direct[n_pass_dir] = True
    q_direct[n_pass_dir] = 0.012
    topA_p[n_pass_dir, 0] = t_pos1
    posA_onB[n_pass_dir, 0] = 4.25 + score_boost
    topA_n[n_pass_dir, 0] = t_neg1
    negA_onB[n_pass_dir, 0] = -3.75

    T_col[n_pass_other], T_iso[n_pass_other] = 4.1, 4.9
    r_ab_res[n_pass_other], r_ba_res[n_pass_other] = 2.1, 2.3
    named05[n_pass_other] = named10[n_pass_other] = True
    is_direct[n_pass_other] = False
    q_direct[n_pass_other] = 0.42
    topB_p[n_pass_other, 0] = t_pos2
    posB_onA[n_pass_other, 0] = 3.1
    topB_n[n_pass_other, 0] = t_neg2
    negB_onA[n_pass_other, 0] = -2.4

    T_col[n_fail], T_iso[n_fail] = 3.0, 1.1
    named05[n_fail] = False
    topA_p[n_fail, 0] = t_fail
    posA_onB[n_fail, 0] = 1.9

    np.savez(
        fo_whole / f"L{layer}_forced_odd.npz",
        T_col=T_col, T_iso=T_iso, r_ab_res_col=r_ab_res, r_ba_res_col=r_ba_res,
        named_dual05=named05, named_dual10=named10,
        named_col05=named05 | (np.arange(N) == n_fail), named_iso05=named05,
        is_direct=is_direct, q_direct=q_direct, kl_mean=kl_mean, alpha_abs=alpha_abs,
        topA_p=topA_p, topB_p=topB_p, topA_n=topA_n, topB_n=topB_n,
        posA_onB=posA_onB, posB_onA=posB_onA, negA_onB=negA_onB, negB_onA=negB_onA,
    )
    (fo_whole / f"L{layer}_forced_odd.meta.json").write_text(
        json.dumps({
            "layer": layer,
            "d_ff": N,
            "model": model_id,
            "weights": weights,
            "weights_id": gp.weights_id_from_card(weights),
            "readout_mode": "proj/LEX_L2_cent",
            "dose_stat": "kl_p90",
            "target_kl": 0.25,
            "dose_ceil": 1.15,
            "n_cols": 128,
            "n_iso": 64,
            "dual05_count": 2,
            "dual05_pct": round(200.0 / N, 2),
            "dual10_count": 2,
            "vs_columns": {"auc_T": 0.85, "fdp05": {"count": 3}},
            "vs_isotropic": {"auc_T": 0.88, "fdp05": {"count": 2}},
            "null_splits_columns": {"n": 20, "fdp05_false_mean": 0.0, "fdp05_false_max": 0},
            "null_splits_isotropic": {"n": 20, "fdp05_false_mean": 0.0, "fdp05_false_max": 0},
            "half_c": {"evaluated": 2, "vs_columns": {"frac_p05_vs_ctl": 1.0}, "vs_isotropic": {"frac_p05_vs_ctl": 1.0}},
            "weight_word_diag": {"available": True, "direct_among_dual05": 1},
        }),
        encoding="utf-8",
    )


def _stage_sources(staging: Path, L: int, N: int, vocab: List[str], model_id: str, weights: Dict[str, Any]) -> Dict[str, Any]:
    good_ids = [i for i, s in enumerate(vocab) if len(s.strip()) >= 3 and s.strip().isalpha()]
    if len(good_ids) < 10:
        good_ids = list(range(10, min(len(vocab), 40)))
    t_pos1, t_pos2, t_neg1, t_neg2, t_trig, t_next, t_fail = good_ids[:7]

    fo_last = L - 1
    fo_mid = max(0, L - 2)
    fo_src = staging / "census_forced_odd"
    prereg_dir = fo_src / "prereg"
    prereg_dir.mkdir(parents=True, exist_ok=True)
    rules_p = prereg_dir / "RULES.md"
    rules_p.write_text("# exploratory trial rules\n", encoding="utf-8")
    (prereg_dir / "SHA256SUMS").write_text(f"{gp.file_sha256(rules_p)}  RULES.md\n", encoding="utf-8")
    (prereg_dir / "AMENDMENTS.txt").write_text("None\n", encoding="utf-8")

    fo_whole = fo_src / "whole"
    _write_fo_source(fo_whole, fo_last, N, model_id=model_id, weights=weights,
                     t_pos1=t_pos1, t_pos2=t_pos2, t_neg1=t_neg1, t_neg2=t_neg2, t_fail=t_fail)
    if fo_mid != fo_last:
        _write_fo_source(fo_whole, fo_mid, N, model_id=model_id, weights=weights,
                         t_pos1=t_pos1, t_pos2=t_pos2, t_neg1=t_neg1, t_neg2=t_neg2, t_fail=t_fail)

    ic_src = staging / "input_census_L0_L29"
    ic_src.mkdir(parents=True, exist_ok=True)
    ic_n = 4
    for layer in (0, 1):
        elig = np.ones((2, N), dtype=bool)
        ic_named05 = np.zeros((2, N), dtype=bool)
        ic_named10 = np.zeros((2, N), dtype=bool)
        ic_score = np.zeros((2, N), dtype=np.float32)
        top_token_id = np.full((2, N, 5), -1, dtype=np.int32)
        top_offset = np.zeros((2, N, 5), dtype=np.int8)
        top_cnt_e = np.zeros((2, N, 5), dtype=np.uint8)
        top_cnt_o = np.zeros((2, N, 5), dtype=np.uint8)
        top_lift = np.zeros((2, N, 5), dtype=np.float32)
        next_token_id = np.full((2, N, 3), -1, dtype=np.int32)
        next_cnt_e = np.zeros((2, N, 3), dtype=np.uint8)
        next_cnt_o = np.zeros((2, N, 3), dtype=np.uint8)
        next_lift = np.zeros((2, N, 3), dtype=np.float32)
        bigram_rep = np.zeros((2, N), dtype=bool)
        smp_hits = np.zeros((2, N), dtype=np.uint8)
        smp_total = np.zeros((2, N), dtype=np.uint8)

        ic_named05[0, ic_n] = True
        ic_named10[0, ic_n] = True
        ic_score[0, ic_n] = 2.75 + layer
        top_token_id[0, ic_n, 0] = t_trig
        top_offset[0, ic_n, 0] = -1
        top_cnt_e[0, ic_n, 0] = 6
        top_cnt_o[0, ic_n, 0] = 5
        top_lift[0, ic_n, 0] = 18.4
        next_token_id[0, ic_n, 0] = t_next
        next_cnt_e[0, ic_n, 0] = 4
        next_cnt_o[0, ic_n, 0] = 3
        next_lift[0, ic_n, 0] = 9.2
        bigram_rep[0, ic_n] = True
        smp_hits[0, ic_n] = 3
        smp_total[0, ic_n] = 5

        ic_named05[1, ic_n] = False
        ic_score[1, ic_n] = 0.21
        top_token_id[1, ic_n, 0] = t_fail
        top_cnt_e[1, ic_n, 0] = 1
        top_cnt_o[1, ic_n, 0] = 1

        np.savez(
            ic_src / f"L{layer:02d}_input_census.npz",
            eligible=elig, named_fdp05=ic_named05, named_fdp10=ic_named10, score=ic_score,
            top_token_id=top_token_id, top_offset=top_offset, top_cnt_e=top_cnt_e,
            top_cnt_o=top_cnt_o, top_lift=top_lift,
            next_token_id=next_token_id, next_cnt_e=next_cnt_e, next_cnt_o=next_cnt_o, next_lift=next_lift,
            bigram_rep=bigram_rep, smp_hits=smp_hits, smp_total=smp_total,
        )

    (ic_src / "summary.json").write_text(
        json.dumps({
            "model": model_id,
            "weights": weights,
            "weights_id": gp.weights_id_from_card(weights),
            "split_kind": "document_disjoint",
            "layers": [
                {"layer": l, "total_faces": 2 * N, "eligible_faces": 2 * N, "raw_2plus2_count": 1,
                 "named_fdp05_count": 1, "smp_named_rate": 0.6, "smp_freq_rate": 0.05, "smp_shuf_rate": 0.02}
                for l in (0, 1)
            ],
        }),
        encoding="utf-8",
    )
    (ic_src / "ablation_check.json").write_text(
        json.dumps({
            "model": model_id,
            "weights": weights,
            "control_kind": "random_same_layer_neuron",
            "layers": [{
                "layer": 0,
                "neurons": 8,
                "passages": 32,
                "kl_pk": {"auc_active_vs_control": 0.68, "wilcoxon_p": 1e-4, "frac_neurons_max4_gt_0.005": 0.25},
                "kl_span": {"auc_active_vs_control": 0.66, "wilcoxon_p": 2e-4, "frac_neurons_max4_gt_0.005": 0.25},
            }],
        }),
        encoding="utf-8",
    )

    return {
        "fo_last": fo_last,
        "fo_mid": fo_mid,
        "ic_n": ic_n,
        "t_pos1": t_pos1,
        "t_pos2": t_pos2,
        "t_neg1": t_neg1,
        "t_neg2": t_neg2,
        "t_trig": t_trig,
        "t_next": t_next,
        "t_fail": t_fail,
        "t_pos1_str": vocab[t_pos1],
        "t_trig_str": vocab[t_trig],
        "t_next_str": vocab[t_next],
        "t_fail_str": vocab[t_fail],
    }


def main() -> int:
    try:
        torch.set_grad_enabled(False)
        rt, vault = stand_in()
        app = bind_server(rt, vault)
        c = TestClient(app, base_url=BASE)
        w = ao.Weights(language_model(rt.hf_model), rt)
        L, N = w.n_layers, w.d_ff
        vocab = [rt.tokenizer.decode([i]) for i in range(int(rt.lm_head_weight.shape[0]))]
        model_id = str(rt.model_id)
        w_card = weights_card(rt)

        # 1. Verify production folder convention + regular census run beside supplementary_dir()
        check(cz.census_dir().resolve() == (TMP / "atlas" / "census" / "runs").resolve(),
              f"cz.census_dir() uses atlas/census/runs: {cz.census_dir()}")
        check(agv.supplementary_dir().resolve() == (TMP / "atlas" / "census").resolve(),
              f"agv.supplementary_dir() uses atlas/census: {agv.supplementary_dir()}")
        run_name = _plant_regular_census_run(cz.census_dir(), model_id)

        arch0 = c.get("/api/atlas/arch").json()
        check(list(arch0)[-2:] == ["label_gate", "lens"], f"empty /arch ordering preserved: {list(arch0)[-4:]}")
        check(arch0["census"] == {"0": run_name}, f"regular census visible in /arch: {arch0['census']}")
        check(arch0.get("supplementary_census") == {"forced_odd": [], "input_census": []},
              f"empty supplementary_census has only forced_odd & input_census: {arch0.get('supplementary_census')}")
        hv0 = c.get("/api/atlas/hover", params={"layer": 0, "n": 0}).json()
        check("regime" not in hv0, "empty atlas /hover omits regime key")
        check(c.get("/api/atlas/census/regime/search", params={"regime": "forced_odd", "q": "a"}).status_code == 404,
              "/census/regime/search removed (404)")

        # 2. Stage source artifacts and exercise publish_gap_census.py guards & verification
        staging = TMP / "staging"
        fx = _stage_sources(staging, L, N, vocab, model_id, w_card)

        # validate_input_census_npz check: clean passes, non-causal +1 in top_offset fails
        pgc.validate_input_census_npz(staging / "input_census_L0_L29" / "L00_input_census.npz")
        bad_npz = TMP / "bad_offset.npz"
        with np.load(staging / "input_census_L0_L29" / "L00_input_census.npz") as z:
            d = {k: np.array(z[k], copy=True) for k in z.files}
        d["top_offset"][0, 0, 0] = 1
        np.savez(bad_npz, **d)
        try:
            pgc.validate_input_census_npz(bad_npz)
            check(False, "validate_input_census_npz should reject +1 in top_offset")
        except RuntimeError:
            check(True, "validate_input_census_npz rejects +1 in top_offset")

        bad_str_npz = TMP / "bad_str.npz"
        d_str = dict(d)
        d_str["top_offset"] = np.zeros_like(d["top_offset"])
        d_str["score"] = np.full(d["score"].shape, "text", dtype="<U8")
        np.savez(bad_str_npz, **d_str)
        try:
            pgc.validate_input_census_npz(bad_str_npz)
            check(False, "validate_input_census_npz should reject string dtype")
        except RuntimeError:
            check(True, "validate_input_census_npz rejects string dtype")

        # Destination path depth guards
        for forbidden in (TMP / "atlas" / "census" / "runs", TMP / "atlas" / "census" / "runs" / "sub", TMP / "atlas" / "index" / "nested"):
            try:
                pgc.publish_regimes(staging, forbidden, publish=True, strict_full=False)
                check(False, f"publisher should refuse {forbidden}")
            except RuntimeError:
                check(True, f"publisher refuses {forbidden}")

        # Verify-only fails on empty destination
        try:
            pgc.publish_regimes(staging, None, publish=False, strict_full=False)
            check(False, "default verify-only should fail when destination is empty")
        except FileNotFoundError:
            check(True, "default verify-only refuses empty destination")

        # Publish into default supplementary_dir() (dst_root=None), then verify-only passes
        pub_res = pgc.publish_regimes(staging, None, publish=True, force=False, strict_full=False)
        check(pub_res["mode"] == "published" and (agv.supplementary_dir() / "MANIFEST.json").is_file(),
              f"published with MANIFEST.json into supplementary_dir(): {pub_res}")
        ver_res = pgc.publish_regimes(staging, None, publish=False, strict_full=False)
        check(ver_res["mode"] == "verify_only", "post-publish verify-only passes")

        # Verify /arch returns BOTH census run and published supplementary regimes
        arch1 = c.get("/api/atlas/arch").json()
        check(arch1["census"] == {"0": run_name}, "regular census still present alongside supplementary_census")
        check(list(arch1)[-2:] == ["label_gate", "lens"], "/arch still ends with label_gate, lens")
        check(arch1["supplementary_census"] == {
            "forced_odd": sorted({fx["fo_mid"], fx["fo_last"]}),
            "input_census": [0, 1],
        }, f"production supplementary_census populated via publisher: {arch1['supplementary_census']}")
        check("fdr_rescue" not in arch1["supplementary_census"], "fdr_rescue omitted from /arch")

        # 3. Exploratory framing, own_lens_match (None on last layer), +1 separation, ablation_sampled
        rl1 = c.get("/api/atlas/census/regime/layers").json()
        check(rl1["exploratory"] is True and rl1["not_taken"] is True and "fdr_rescue" not in rl1
              and "fdr_rescue" not in rl1["labels"], f"/regime/layers exploratory & no fdr_rescue: {rl1.keys()}")
        check("exploratory" in rl1["labels"]["input_census"].lower() and "fdr" not in rl1["labels"]["input_census"].lower(),
              "input_census label is plain-English exploratory without FDR <= 5%")

        rn_last = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fo_last"], "n": 2}).json()
        fo_last_rec = rn_last["forced_odd"]
        check(fo_last_rec["named_dual05"] is True and "tag" not in fo_last_rec
              and fo_last_rec["own_lens_match"] is None
              and "last layer" in fo_last_rec["own_lens_note"]
              and fo_last_rec["pos"][0]["tok"] == fx["t_pos1_str"],
              f"last layer omits circuit tag and sets own_lens_match=None: {fo_last_rec}")

        if fx["fo_mid"] != fx["fo_last"]:
            rn_mid_t = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fo_mid"], "n": 2}).json()["forced_odd"]
            rn_mid_f = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fo_mid"], "n": 5}).json()["forced_odd"]
            check(rn_mid_t["own_lens_match"] is True and rn_mid_f["own_lens_match"] is False
                  and "circuit_mediated" not in json.dumps(rn_mid_f),
                  f"interior layer own_lens_match True/False without circuit_mediated: {rn_mid_t}, {rn_mid_f}")

        rn_fail = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fo_last"], "n": 7}).json()["forced_odd"]
        check(rn_fail["named_dual05"] is False and rn_fail["pos"] == [] and rn_fail["neg"] == [],
              "non-named forced_odd neuron hides pieces")

        fo_units = c.get("/api/atlas/census/regime/units", params={"layer": fx["fo_last"], "regime": "forced_odd"}).json()
        check([r["n"] for r in fo_units["rows"]] == [2, 5] and "circuit_mediated" not in json.dumps(fo_units),
              f"/regime/units forced_odd rows: {fo_units['rows']}")
        check(c.get("/api/atlas/census/regime/units", params={"layer": fx["fo_last"], "regime": "forced_odd", "q": "n2"}).json()["matched"] == 1
              and c.get("/api/atlas/census/regime/units", params={"layer": fx["fo_last"], "regime": "forced_odd", "q": "2"}).json()["matched"] == 1,
              "exact id filter matches n2 and 2")

        rn_ic = c.get("/api/atlas/census/regime/neuron", params={"layer": 0, "n": fx["ic_n"]}).json()
        ic = rn_ic["input_census"]
        check(ic["no_passages_stored"] is True and ic["exploratory"] is True
              and ic["faces"]["+"]["named"] is True
              and ic["faces"]["+"]["triggers"][0]["tok"] == fx["t_trig_str"]
              and ic["faces"]["+"]["triggers"][0]["offset"] == -1
              and ic["faces"]["+"]["next_pieces"][0]["tok"] == fx["t_next_str"]
              and ic["faces"]["+"]["next_pieces"][0]["offset"] == 1
              and ic["faces"]["-"]["triggers"] == [] and ic["faces"]["-"]["next_pieces"] == []
              and ic["meta"]["ablation_sampled"] is True
              and "ablation_verified" not in ic["meta"]
              and "PASSED" not in json.dumps(ic),
              f"input_census separates +1 next_pieces and uses ablation_sampled: {ic}")
        hv_fo = c.get("/api/atlas/hover", params={"layer": fx["fo_last"], "n": 2}).json()
        hv_reg_fo = hv_fo.get("regime", {}).get("forced_odd") or {}
        check(
            hv_reg_fo.get("exploratory") is True
            and hv_reg_fo.get("is_last_layer") is True
            and hv_reg_fo.get("own_lens_match") is None
            and fx["t_pos1_str"] in (hv_reg_fo.get("pos") or [])
            and "fdr_rescue" not in hv_fo.get("regime", {})
            and "circuit_mediated" not in json.dumps(hv_fo),
            f"/hover exposes exploratory forced_odd without fdr_rescue/circuit: {hv_fo.get('regime')}",
        )
        if fx["fo_mid"] != fx["fo_last"]:
            hv_mid = c.get("/api/atlas/hover", params={"layer": fx["fo_mid"], "n": 2}).json().get("regime", {}).get("forced_odd", {})
            check(hv_mid.get("own_lens_match") is True and hv_mid.get("is_last_layer") is False, f"/hover mid own_lens_match: {hv_mid}")
        check("regime" not in c.get("/api/atlas/hover", params={"layer": fx["fo_last"], "n": 7}).json(),
              "/hover omits regime for un-named neurons")
        hv_ic = c.get("/api/atlas/hover", params={"layer": 0, "n": fx["ic_n"]}).json()
        hv_ic_reg = hv_ic.get("regime", {}).get("input_census") or {}
        hv_ic_plus = (hv_ic_reg.get("faces") or {}).get("+") or []
        check(
            hv_ic_reg.get("exploratory") is True
            and len(hv_ic_plus) >= 1
            and hv_ic_plus[0].get("offset") in (-1, 0)
            and all(x.get("offset") in (-1, 0) for x in hv_ic_plus)
            and "next_pieces" not in json.dumps(hv_ic_reg),
            f"/hover exposes exploratory causal-only input_census: {hv_ic_reg}",
        )

        # 4. Overwrite refusal vs --force republish + bounded cache invalidation
        _write_fo_source(staging / "census_forced_odd" / "whole", fx["fo_last"], N, model_id=model_id, weights=w_card,
                         t_pos1=fx["t_pos1"], t_pos2=fx["t_pos2"], t_neg1=fx["t_neg1"], t_neg2=fx["t_neg2"],
                         t_fail=fx["t_fail"], score_boost=3.25)
        try:
            pgc.publish_regimes(staging, None, publish=True, force=False, strict_full=False)
            check(False, "overwrite without --force must raise RuntimeError")
        except RuntimeError:
            check(True, "publisher refuses differing overwrite without --force")

        cache_len_before = len(agv._CACHE)
        pgc.publish_regimes(staging, None, publish=True, force=True, strict_full=False)
        repub_n = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fo_last"], "n": 2}).json()["forced_odd"]
        check(round(float(repub_n["T_col"]), 2) == 9.75 and len(agv._CACHE) <= cache_len_before + 2,
              f"--force republish invalidates cache in place: T_col={repub_n['T_col']}, cache={len(agv._CACHE)}")

        # 5. Vault-edit regime-specific & layer-aware staleness
        vault.active_patches.append(PatchOperationRecord(
            patch_id=1, op_type="DELETE", layer_idx=0, neuron_idx=0, polarity=1.0,
            target_concept="", strength=0.0, statement="test", timestamp=time.time(), orig_norm=1.0, new_norm=0.0,
        ))
        try:
            u_fo_st = c.get("/api/atlas/census/regime/units", params={"layer": fx["fo_last"], "regime": "forced_odd"}).json()
            u_ic0_st = c.get("/api/atlas/census/regime/units", params={"layer": 0, "regime": "input_census"}).json()
            u_ic1_st = c.get("/api/atlas/census/regime/units", params={"layer": 1, "regime": "input_census"}).json()
            check(u_fo_st.get("stale") == agv.FORCED_ODD_STALE, "forced_odd gets FORCED_ODD_STALE on patch")
            check("stale" not in u_ic0_st, f"input_census at L0 is NOT stale for patch at L0 (own=False): {u_ic0_st.get('stale')}")
            check(u_ic1_st.get("stale") == agv.INPUT_CENSUS_STALE, "input_census at L1 gets INPUT_CENSUS_STALE for earlier patch at L0")
            hv_fo_st = c.get("/api/atlas/hover", params={"layer": fx["fo_last"], "n": 2}).json().get("regime", {})
            hv_ic0_st = c.get("/api/atlas/hover", params={"layer": 0, "n": fx["ic_n"]}).json().get("regime", {})
            check(hv_fo_st.get("stale") == agv.FORCED_ODD_STALE, "/hover forced_odd gets FORCED_ODD_STALE on patch")
            check("stale" not in hv_ic0_st, f"/hover L0 input_census NOT stale on L0 patch: {hv_ic0_st}")
        finally:
            vault.active_patches.clear()
        check("stale" not in c.get("/api/atlas/census/regime/layers").json(), "stale clears after vault edit removed")

        # 6. Malformed file resilience & model/weights mismatch gating (HTTP 200 + refused map)
        fo_dir = agv.supplementary_dir() / "forced_odd"
        (fo_dir / f"L{fx['fo_last']}_forced_odd.meta.json").write_text("{bad json", encoding="utf-8")
        if fx["fo_mid"] != fx["fo_last"]:
            (fo_dir / f"L{fx['fo_mid']}_forced_odd.meta.json").write_text(
                json.dumps({"layer": fx["fo_mid"], "d_ff": N, "model": "wrong/model-id"}), encoding="utf-8"
            )
        bad_ic_path = agv.supplementary_dir() / "input_census" / "L01_input_census.npz"
        np.savez(bad_ic_path, score=np.zeros((2, N), dtype=np.float32))  # missing required keys

        arch_deg = c.get("/api/atlas/arch")
        check(arch_deg.status_code == 200, f"/arch returns 200 despite malformed/mismatched files: {arch_deg.status_code}")
        deg_json = arch_deg.json()
        ref = deg_json.get("supplementary_refused") or {}
        check(f"forced_odd:L{fx['fo_last']}" in ref and "input_census:L01" in ref,
              f"supplementary_refused records bad JSON & incomplete NPZ: {ref}")
        if fx["fo_mid"] != fx["fo_last"]:
            check("wrong/model-id" in ref.get(f"forced_odd:L{fx['fo_mid']}", ""),
                  f"supplementary_refused records wrong model id: {ref}")
        check(deg_json["supplementary_census"]["input_census"] == [0],
              f"valid L00 remains served while bad L01 is refused: {deg_json['supplementary_census']}")

        # Restore & verify weights mismatch refusal
        pgc.publish_regimes(staging, None, publish=True, force=True, strict_full=False)
        meta_last_p = fo_dir / f"L{fx['fo_last']}_forced_odd.meta.json"
        meta_obj = json.loads(meta_last_p.read_text(encoding="utf-8"))
        meta_obj["weights"] = {"format": "gguf", "id": "other-gguf", "sha256_header": "deadbeef"}
        meta_last_p.write_text(json.dumps(meta_obj), encoding="utf-8")
        rl_w = c.get("/api/atlas/census/regime/layers").json()
        check(f"forced_odd:L{fx['fo_last']}" in (rl_w.get("refused") or {}),
              f"weights mismatch refused cleanly: {rl_w.get('refused')}")
        check("regime" not in c.get("/api/atlas/hover", params={"layer": fx["fo_last"], "n": 2}).json(),
              "/hover omits regime cleanly on weights mismatch")

        # Restore clean published state and test 400 validation
        pgc.publish_regimes(staging, None, publish=True, force=True, strict_full=False)
        check(c.get("/api/atlas/census/regime/units", params={"layer": 0, "regime": "fdr_rescue"}).status_code == 400,
              "removed fdr_rescue -> 400")
        check(c.get("/api/atlas/census/regime/units", params={"layer": L, "regime": "forced_odd"}).status_code == 400,
              "invalid layer -> 400")
        check(c.get("/api/atlas/census/regime/units", params={"layer": 0, "regime": "input_census", "face": "x"}).status_code == 400,
              "invalid face -> 400")
        check(c.get("/api/atlas/census/regime/neuron", params={"layer": 0, "n": N}).status_code == 400,
              "invalid neuron -> 400")

        print(f"atlas gap-views check: {passed} passed, {len(failures)} failed")
        return 1 if failures else 0
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
