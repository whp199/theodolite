#!/usr/bin/env python3
"""CPU-only FastAPI check for supplementary atlas routes (`backend/atlas_gap_views.py` + `backend/atlas.py`).

Uses the tiny stand-in Gemma 4 model from `scripts/atlas_stand_in.py` with an isolated temp `THEODOLITE_CENSUS_DIR`
so all empty-state, dual-null / FDR gating, zero-raw-text, hover, search, error-handling, and vault-edit staleness
contracts are verified without GPU hardware or mutating production data.
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
os.environ["THEODOLITE_CENSUS_DIR"] = str(TMP / "census")

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi.testclient import TestClient  # noqa: E402

import backend.atlas as atlas_routes  # noqa: E402
from atlas_stand_in import bind_server, stand_in  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend.causal_check import language_model  # noqa: E402
from backend.larql_vault import PatchOperationRecord  # noqa: E402

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


def _write_fixtures(cdir: Path, L: int, N: int, vocab: List[str]) -> Dict[str, Any]:
    # Pick non-trivial alphanumeric tokens so queries match cleanly across tokenizers.
    good_ids = [i for i, s in enumerate(vocab) if len(s.strip()) >= 3 and s.strip().isalpha()]
    if len(good_ids) < 10:
        good_ids = list(range(10, min(len(vocab), 40)))
    t_pos1, t_pos2, t_neg1, t_neg2, t_trig, t_fail = good_ids[:6]

    # 1. forced_odd on last stand-in layer (L - 1)
    fo_layer = L - 1
    fo_dir = cdir / "forced_odd"
    fo_dir.mkdir(parents=True, exist_ok=True)
    n_pass_dir = 2
    n_pass_cir = 5
    n_fail = 7

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

    # n_pass_dir: passes dual05, direct_unembed
    T_col[n_pass_dir], T_iso[n_pass_dir] = 6.5, 5.8
    r_ab_res[n_pass_dir], r_ba_res[n_pass_dir] = 2.9, 3.1
    named05[n_pass_dir] = named10[n_pass_dir] = is_direct[n_pass_dir] = True
    q_direct[n_pass_dir] = 0.012
    topA_p[n_pass_dir, 0] = t_pos1
    posA_onB[n_pass_dir, 0] = 4.25
    topA_n[n_pass_dir, 0] = t_neg1
    negA_onB[n_pass_dir, 0] = -3.75

    # n_pass_cir: passes dual05, circuit_mediated
    T_col[n_pass_cir], T_iso[n_pass_cir] = 4.1, 4.9
    r_ab_res[n_pass_cir], r_ba_res[n_pass_cir] = 2.1, 2.3
    named05[n_pass_cir] = named10[n_pass_cir] = True
    is_direct[n_pass_cir] = False
    q_direct[n_pass_cir] = 0.42
    topB_p[n_pass_cir, 0] = t_pos2
    posB_onA[n_pass_cir, 0] = 3.1
    topB_n[n_pass_cir, 0] = t_neg2
    negB_onA[n_pass_cir, 0] = -2.4

    # n_fail: MUST NOT expose pieces even if raw arrays hold non-zero values
    T_col[n_fail], T_iso[n_fail] = 3.0, 1.1
    named05[n_fail] = False
    topA_p[n_fail, 0] = t_fail
    posA_onB[n_fail, 0] = 1.9

    np.savez(
        fo_dir / f"L{fo_layer}_forced_odd.npz",
        T_col=T_col,
        T_iso=T_iso,
        r_ab_res_col=r_ab_res,
        r_ba_res_col=r_ba_res,
        named_dual05=named05,
        named_dual10=named10,
        named_col05=named05 | (np.arange(N) == n_fail),
        named_iso05=named05,
        is_direct=is_direct,
        q_direct=q_direct,
        kl_mean=kl_mean,
        alpha_abs=alpha_abs,
        topA_p=topA_p,
        topB_p=topB_p,
        topA_n=topA_n,
        topB_n=topB_n,
        posA_onB=posA_onB,
        posB_onA=posB_onA,
        negA_onB=negA_onB,
        negB_onA=negB_onA,
    )
    (fo_dir / f"L{fo_layer}_forced_odd.meta.json").write_text(
        json.dumps({
            "layer": fo_layer,
            "d_ff": N,
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
            "half_c": {"evaluated": 2, "vs_columns": {"frac_p05_vs_ctl": 1.0, "half_c_auc": 1.0}, "vs_isotropic": {"frac_p05_vs_ctl": 1.0, "half_c_auc": 1.0}},
            "weight_word_diag": {"direct_among_dual05": 1},
        }),
        encoding="utf-8",
    )

    # 2. input_census on layer 0 (and ablation check on layer 0)
    ic_dir = cdir / "input_census"
    ic_dir.mkdir(parents=True, exist_ok=True)
    elig = np.ones((2, N), dtype=bool)
    ic_named05 = np.zeros((2, N), dtype=bool)
    ic_named10 = np.zeros((2, N), dtype=bool)
    ic_score = np.zeros((2, N), dtype=np.float32)
    top_token_id = np.full((2, N, 5), -1, dtype=np.int32)
    top_offset = np.zeros((2, N, 5), dtype=np.int8)
    top_cnt_e = np.zeros((2, N, 5), dtype=np.uint8)
    top_cnt_o = np.zeros((2, N, 5), dtype=np.uint8)
    top_lift = np.zeros((2, N, 5), dtype=np.float32)
    bigram_rep = np.zeros((2, N), dtype=bool)
    smp_hits = np.zeros((2, N), dtype=np.uint8)
    smp_total = np.zeros((2, N), dtype=np.uint8)

    ic_n = 4
    ic_named05[0, ic_n] = True
    ic_named10[0, ic_n] = True
    ic_score[0, ic_n] = 2.75
    top_token_id[0, ic_n, 0] = t_trig
    top_offset[0, ic_n, 0] = -1
    top_cnt_e[0, ic_n, 0] = 6
    top_cnt_o[0, ic_n, 0] = 5
    top_lift[0, ic_n, 0] = 18.4
    bigram_rep[0, ic_n] = True
    smp_hits[0, ic_n] = 3
    smp_total[0, ic_n] = 5

    # Non-passing '-' face of same neuron must keep triggers hidden even if populated
    ic_named05[1, ic_n] = False
    ic_score[1, ic_n] = 0.21
    top_token_id[1, ic_n, 0] = t_fail
    top_cnt_e[1, ic_n, 0] = 1
    top_cnt_o[1, ic_n, 0] = 1
    elig[1, 9] = False

    np.savez(
        ic_dir / "L00_input_census.npz",
        eligible=elig,
        named_fdp05=ic_named05,
        named_fdp10=ic_named10,
        score=ic_score,
        top_token_id=top_token_id,
        top_offset=top_offset,
        top_cnt_e=top_cnt_e,
        top_cnt_o=top_cnt_o,
        top_lift=top_lift,
        bigram_rep=bigram_rep,
        smp_hits=smp_hits,
        smp_total=smp_total,
    )
    (ic_dir / "ablation_check.json").write_text(
        json.dumps({
            "quiet_thresh": 0.05,
            "min_extreme": 0.1,
            "layers": [{
                "layer": 0,
                "neurons": 8,
                "passages": 32,
                "quiet_max_norm_act_max": 0.041,
                "all_quiet_under_thresh": True,
                "kl_pk": {"auc_active_vs_quiet": 0.72, "wilcoxon_p": 1e-6, "frac_neurons_max4_gt_0.005": 0.25},
                "kl_span": {"auc_active_vs_quiet": 0.70, "wilcoxon_p": 2e-6, "frac_neurons_max4_gt_0.005": 0.25},
            }],
        }),
        encoding="utf-8",
    )

    # 3. fdr_rescue on layer 3
    fdr_layer = min(3, L - 1)
    fdr_dir = cdir / "fdr_rescue"
    fdr_dir.mkdir(parents=True, exist_ok=True)
    fdr_n = 11
    (fdr_dir / f"L{fdr_layer}_whole_named.json").write_text(
        json.dumps({
            "tag": f"L{fdr_layer}_whole",
            "layer": fdr_layer,
            "design": "match_ceil",
            "count": 1,
            "neurons": [{
                "neuron": fdr_n,
                "T_dec": 14.5,
                "rep_AB_res": 16.2,
                "rep_BA_res": 15.8,
                "covered": True,
                "rep_words_AB": [["justice", 7.4], ["court", 5.1]],
                "rep_words_BA": [["justice", 7.2], ["trial", 4.9]],
            }],
        }),
        encoding="utf-8",
    )
    (fdr_dir / "summary.json").write_text(
        json.dumps({
            "runs": [
                {
                    "tag": f"L{fdr_layer}_whole",
                    "layer": fdr_layer,
                    "design": "match_ceil",
                    "neurons": N,
                    "controls": 64,
                    "readlib_named": 0,
                    "fdp05": {"count": 1, "threshold": 5.0, "fdp": 0.0},
                    "fdp10": {"count": 2, "threshold": 4.2, "fdp": 0.08},
                    "n3_covered": {"covered": N, "neurons": N, "share_covered": 1.0, "n1_named_inside_covered": 1},
                    "null_splits": {"n": 20, "fdp05_false_mean": 0.0, "fdp05_false_max": 0},
                },
                {
                    "tag": "L99_trial",
                    "layer": 99,
                    "design": "match_ceil",
                    "neurons": 100,
                    "controls": 64,
                    "readlib_named": 0,
                    "fdp05": {"count": 1},
                },
            ],
        }),
        encoding="utf-8",
    )

    return {
        "fo_layer": fo_layer,
        "n_pass_dir": n_pass_dir,
        "n_pass_cir": n_pass_cir,
        "n_fail": n_fail,
        "t_pos1_str": vocab[t_pos1],
        "t_neg2_str": vocab[t_neg2],
        "t_fail_str": vocab[t_fail],
        "ic_n": ic_n,
        "t_trig_str": vocab[t_trig],
        "fdr_layer": fdr_layer,
        "fdr_n": fdr_n,
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

        # 1. Empty state preserves existing /arch and /hover shapes strictly
        arch0 = c.get("/api/atlas/arch").json()
        check(list(arch0)[-2:] == ["label_gate", "lens"], f"empty /arch ordering preserved: {list(arch0)[-4:]}")
        check(arch0["census"] == {}, "empty /arch census empty")
        check(arch0.get("supplementary_census") == {"forced_odd": [], "input_census": [], "fdr_rescue": []},
              f"empty supplementary_census: {arch0.get('supplementary_census')}")
        hv0 = c.get("/api/atlas/hover", params={"layer": 0, "n": 0}).json()
        check("regime" not in hv0, "empty /hover omits regime key")
        rl0 = c.get("/api/atlas/census/regime/layers").json()
        check(rl0["forced_odd"] == [] and rl0["input_census"] == [] and rl0["fdr_rescue"] == [], f"empty regime layers: {rl0}")

        # 2. Populate synthetic fixtures
        fx = _write_fixtures(TMP / "census", L, N, vocab)
        arch1 = c.get("/api/atlas/arch").json()
        check(list(arch1)[-2:] == ["label_gate", "lens"], "/arch still ends with label_gate, lens")
        check(arch1["census"] == {}, "decoding census remains untouched")
        check(arch1["supplementary_census"] == {
            "forced_odd": [fx["fo_layer"]],
            "input_census": [0],
            "fdr_rescue": [fx["fdr_layer"]],
        }, f"supplementary_census populated: {arch1['supplementary_census']}")

        # 3. forced_odd dual-null gating + tag verification
        rn_dir = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fo_layer"], "n": fx["n_pass_dir"]}).json()
        fo_dir = rn_dir["forced_odd"]
        check(fo_dir["named_dual05"] is True and fo_dir["tag"] == "direct_unembed" and fo_dir["rank"] == 1
              and fo_dir["pos"][0]["tok"] == fx["t_pos1_str"] and fo_dir["neg"][0]["score"] == 3.75,
              f"forced_odd direct passing neuron: {fo_dir}")
        rn_fail = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fo_layer"], "n": fx["n_fail"]}).json()
        fo_fail = rn_fail["forced_odd"]
        check(fo_fail["named_dual05"] is False and fo_fail["pos"] == [] and fo_fail["neg"] == []
              and "dual-null FDR" in fo_fail.get("why_no_words", ""),
              f"forced_odd failing neuron hides pieces: {fo_fail}")
        fo_units = c.get("/api/atlas/census/regime/units", params={"layer": fx["fo_layer"], "regime": "forced_odd"}).json()
        check([r["n"] for r in fo_units["rows"]] == [fx["n_pass_dir"], fx["n_pass_cir"]],
              f"forced_odd /units excludes non-dual05 neuron: {fo_units['rows']}")
        fo_search_fail = c.get("/api/atlas/census/regime/search", params={"regime": "forced_odd", "q": fx["t_fail_str"].strip()}).json()
        check(fo_search_fail["total"] == 0, "forced_odd search ignores non-dual05 neuron's suppressed token")
        hv_fo_pass = c.get("/api/atlas/hover", params={"layer": fx["fo_layer"], "n": fx["n_pass_dir"]}).json()
        hv_fo_fail = c.get("/api/atlas/hover", params={"layer": fx["fo_layer"], "n": fx["n_fail"]}).json()
        check(hv_fo_pass["regime"]["forced_odd"]["tag"] == "direct_unembed" and "forced_odd" not in (hv_fo_fail.get("regime") or {}),
              "forced_odd hover strictly gated by dual05")

        # 4. input_census split-half FDR gating + ablation metadata + zero raw text
        rn_ic = c.get("/api/atlas/census/regime/neuron", params={"layer": 0, "n": fx["ic_n"]}).json()
        ic = rn_ic["input_census"]
        check(ic["no_raw_text"] is True
              and ic["faces"]["+"]["named_fdp05"] is True
              and ic["faces"]["+"]["triggers"][0]["tok"] == fx["t_trig_str"]
              and ic["faces"]["+"]["triggers"][0]["offset_label"] == "prev"
              and ic["faces"]["+"]["smp_hits"] == 3 and ic["faces"]["+"]["smp_total"] == 5
              and ic["faces"]["-"]["named_fdp05"] is False
              and ic["faces"]["-"]["triggers"] == []
              and ic["meta"]["ablation_verified"] is True,
              f"input_census per-face gating & ablation: {ic}")
        ic_units_neg = c.get("/api/atlas/census/regime/units", params={"layer": 0, "regime": "input_census", "face": "-"}).json()
        check(ic_units_neg["matched"] == 0, "input_census face='-' filter excludes passing '+' face")
        hv_ic = c.get("/api/atlas/hover", params={"layer": 0, "n": fx["ic_n"]}).json()
        check(set(hv_ic["regime"]["input_census"]["faces"].keys()) == {"+"}, f"input_census hover omits failing '-' face: {hv_ic['regime']}")

        # 5. fdr_rescue summary, units, neuron, and trial exclusion
        rl = c.get("/api/atlas/census/regime/layers").json()
        check(len(rl["fdr_rescue"]) == 1 and rl["fdr_rescue"][0]["layer"] == fx["fdr_layer"],
              f"fdr_rescue layers exclude _trial entries: {rl['fdr_rescue']}")
        fdr_search = c.get("/api/atlas/census/regime/search", params={"regime": "fdr_rescue", "q": "just"}).json()
        check(fdr_search["total"] == 1 and fdr_search["results"][0]["n"] == fx["fdr_n"],
              f"fdr_rescue search finds rescued neuron: {fdr_search}")
        rn_fdr = c.get("/api/atlas/census/regime/neuron", params={"layer": fx["fdr_layer"], "n": fx["fdr_n"]}).json()
        check(rn_fdr["fdr_rescue"]["named_fdp05"] is True and rn_fdr["fdr_rescue"]["covered"] is True,
              f"fdr_rescue neuron detail: {rn_fdr['fdr_rescue']}")

        # 6. Vault-edit staleness contract + 400 validation
        CS = atlas_routes.CENSUS_STALE
        vault.active_patches.append(PatchOperationRecord(
            patch_id=1, op_type="DELETE", layer_idx=0, neuron_idx=0, polarity=1.0,
            target_concept="", strength=0.0, statement="test stale", timestamp=time.time(), orig_norm=1.0, new_norm=0.0,
        ))
        try:
            check(c.get("/api/atlas/census/regime/layers").json().get("stale") == CS, "stale on /regime/layers")
            check(c.get("/api/atlas/census/regime/units", params={"layer": fx["fo_layer"], "regime": "forced_odd"}).json().get("stale") == CS,
                  "stale on /regime/units")
            check(c.get("/api/atlas/census/regime/neuron", params={"layer": 0, "n": fx["ic_n"]}).json().get("stale") == CS,
                  "stale on /regime/neuron")
            check(c.get("/api/atlas/census/regime/search", params={"regime": "fdr_rescue", "q": "court"}).json().get("stale") == CS,
                  "stale on /regime/search")
            check(c.get("/api/atlas/hover", params={"layer": fx["fo_layer"], "n": fx["n_pass_dir"]}).json()["regime"].get("stale") == CS,
                  "stale on /hover.regime")
        finally:
            vault.active_patches.clear()
        check("stale" not in c.get("/api/atlas/census/regime/layers").json(), "stale clears once vault edit removed")

        check(c.get("/api/atlas/census/regime/units", params={"layer": 0, "regime": "bogus"}).status_code == 400, "invalid regime -> 400")
        check(c.get("/api/atlas/census/regime/units", params={"layer": L, "regime": "forced_odd"}).status_code == 400, "invalid layer -> 400")
        check(c.get("/api/atlas/census/regime/units", params={"layer": 0, "regime": "input_census", "face": "x"}).status_code == 400, "invalid face -> 400")
        check(c.get("/api/atlas/census/regime/search", params={"regime": "forced_odd", "q": "  "}).status_code == 400, "empty query -> 400")
        check(c.get("/api/atlas/census/regime/neuron", params={"layer": 0, "n": N}).status_code == 400, "invalid neuron -> 400")

        print(f"atlas gap-views check: {passed} passed, {len(failures)} failed")
        return 1 if failures else 0
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
