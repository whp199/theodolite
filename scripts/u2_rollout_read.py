#!/usr/bin/env python3
"""Confirmatory Path 2 (P2): Dual-control U2 rollout reader (Track 2A T_dec + Track 2B weight_words).

Reads paired L<L>-match_things-u2-cols.jsonl (neurons + columns controls, seed 21) and
L<L>-match_things-u2-iso.jsonl (isotropic controls, seed 22) in isolated trial_dir,
verifies prereg SHA256SUMS & CODE_SHA256SUMS, checks 12/12 baseline bit-equality,
evaluates Track 2A (behavioral U2-T_dec + C4 holdout + cross-pool) and Track 2B
(corrected own-word weight-word verification + shuffled null + cross-pool), and
outputs L<L>_verdict.json with pass_track_2a, pass_track_2b, pass_either.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import census_fdr as cf  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend.gap_stats import mannwhitney_greater  # noqa: E402
from backend.label_gate import label_top_ids  # noqa: E402
from backend.theodolite_math import gemma4_rms_norm  # noqa: E402

HALVES = (
    ("bicycle", "walk", "chair", "kettle", "sleep", "lamp"),
    ("clock", "rice", "vase", "lock", "money", "blanket"),
)


def verify_prereg_integrity(prereg_dir: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    st = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if st.stdout.strip():
        raise RuntimeError(f"Refusing confirmatory read on dirty git tree:\n{st.stdout}")
    head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    for fname in ("SHA256SUMS", "CODE_SHA256SUMS"):
        fpath = prereg_dir / fname
        if not fpath.is_file():
            raise RuntimeError(f"Missing {fpath}")
        cwd = prereg_dir if fname == "SHA256SUMS" else repo_root
        for line in fpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            rel = rel.lstrip("*").strip()
            target = cwd / rel
            if not target.is_file():
                raise RuntimeError(f"Missing hashed file {target}")
            got = hashlib.sha256(target.read_bytes()).hexdigest()
            if got != digest:
                raise RuntimeError(f"SHA256 mismatch for {rel}: expected {digest}, got {got}")
    return {
        "git_head": head,
        "repo_commit": (prereg_dir / "repo_commit").read_text(encoding="utf-8").strip() if (prereg_dir / "repo_commit").is_file() else None,
        "prereg_at": (prereg_dir / "prereg_at").read_text(encoding="utf-8").strip() if (prereg_dir / "prereg_at").is_file() else None,
        "prereg_sha256": hashlib.sha256((prereg_dir / "SHA256SUMS").read_bytes()).hexdigest(),
    }


def _norm_word(s: str) -> str:
    return re.sub(r"[^\w]", "", str(s).replace("▁", " ").strip().lower())


def _bh(p: Sequence[float]) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    n = len(p_arr)
    if n == 0:
        return p_arr
    o = np.argsort(p_arr)
    q = np.minimum.accumulate((p_arr[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[o] = np.minimum(q, 1.0)
    return out


def _rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    rx = np.argsort(np.argsort(np.asarray(x)))
    ry = np.argsort(np.argsort(np.asarray(y)))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _write_cross_pool_run(
    *,
    fake_neu_path: Path,
    fake_neu_collapse: Path,
    ctl_path: Path,
    ctl_collapse: Path,
    out_jsonl: Path,
) -> None:
    """Build a synthetic StreamedRun file setting one control pool as 'neuron' vs the other as 'random'."""
    m_fn = json.loads(cz.meta_path_of(fake_neu_path).read_text(encoding="utf-8"))
    m_ct = json.loads(cz.meta_path_of(ctl_path).read_text(encoding="utf-8"))
    c_words = set()
    for cp in (fake_neu_collapse, ctl_collapse):
        if cp.is_file():
            c_words.update(str(w).lower() for w in (json.loads(cp.read_text(encoding="utf-8")).get("words") or ()))

    new_ids: List[int] = []
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for r in cz.read_rows(fake_neu_path):
            if r["kind"] == "baseline":
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            elif r["kind"] == "random":
                rr = dict(r)
                rr["kind"] = "neuron"
                if int(rr["neuron"]) not in new_ids:
                    new_ids.append(int(rr["neuron"]))
                fout.write(json.dumps(rr, ensure_ascii=False) + "\n")
        for r in cz.read_rows(ctl_path):
            if r["kind"] == "random":
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta_out = dict(m_ct)
    meta_out["neuron_ids"] = cz.neuron_ranges(sorted(new_ids)) if new_ids else ""
    cz.meta_path_of(out_jsonl).write_text(json.dumps(meta_out, indent=1) + "\n", encoding="utf-8")
    cz.collapse_path_of(out_jsonl).write_text(json.dumps({"words": sorted(c_words)}, indent=1) + "\n", encoding="utf-8")


def _build_subsets(prompts: Sequence[str], doses: Sequence[float]) -> Dict[str, Any]:
    half = len(prompts) // 2
    pA = set(prompts[:half])
    pB = set(prompts[half:])
    p4A = set(prompts[:4])
    p4B = set(prompts[4:8])
    p4C = set(prompts[8:12])
    mags = sorted({abs(float(d)) for d in doses})
    lo_d, hi_d = mags[0], mags[-1]
    return {
        "all": lambda p, d: True,
        "A": lambda p, d: p in pA,
        "B": lambda p, d: p in pB,
        "A4": lambda p, d: p in p4A,
        "B4": lambda p, d: p in p4B,
        "C4": lambda p, d: p in p4C,
        "AB8": lambda p, d: (p in p4A) or (p in p4B),
        "lower": lambda p, d: abs(float(d)) == lo_d,
        "upper": lambda p, d: abs(float(d)) == hi_d,
    }


def compute_weight_word_metrics(
    *,
    rows_neu_and_col: Sequence[Tuple[str, int, str, float, List[int]]],
    rows_iso: Sequence[Tuple[str, int, str, float, List[int]]],
    VN: Sequence[str],
    neu_eye_pos: Dict[int, List[int]],
    neu_eye_neg: Dict[int, List[int]],
    col_eye_pos: List[List[int]],
    col_eye_neg: List[List[int]],
    iso_eye_pos: List[List[int]],
    iso_eye_neg: List[List[int]],
) -> Dict[str, Any]:
    """Compute Track 2B corrected own-word scores, shuffled null, and iso<->col cross-pool rates."""
    neurons = sorted({n for k, n, *_ in rows_neu_and_col if k == "neuron"})
    cols = sorted({n for k, n, *_ in rows_neu_and_col if k == "random"})
    isos = sorted({n for k, n, *_ in rows_iso if k == "random"})
    nn, C_col, C_iso = len(neurons), len(cols), len(isos)

    W_neu = {s: [set(_norm_word(VN[t]) for t in (neu_eye_pos if s == 1 else neu_eye_neg)[n][:10] if _norm_word(VN[t])) for n in neurons] for s in (+1, -1)}
    W_col = {s: [set(_norm_word(VN[t]) for t in (col_eye_pos if s == 1 else col_eye_neg)[c][:10] if _norm_word(VN[t])) for c in range(C_col)] for s in (+1, -1)}
    W_iso = {s: [set(_norm_word(VN[t]) for t in (iso_eye_pos if s == 1 else iso_eye_neg)[c][:10] if _norm_word(VN[t])) for c in range(C_iso)] for s in (+1, -1)}

    all_words = sorted(set().union(*W_neu[1], *W_neu[-1], *W_col[1], *W_col[-1], *W_iso[1], *W_iso[-1]))
    vi = {wd: i for i, wd in enumerate(all_words)}
    Vn = len(all_words)

    def _mat(wlist: Sequence[set]) -> np.ndarray:
        m = np.zeros((Vn, len(wlist)), dtype=np.float32)
        for j, ws in enumerate(wlist):
            for wd in ws:
                m[vi[wd], j] = 1.0
        return m

    M_neu = {s: _mat(W_neu[s]) for s in (+1, -1)}
    M_col = {s: _mat(W_col[s]) for s in (+1, -1)}
    M_iso = {s: _mat(W_iso[s]) for s in (+1, -1)}

    mags = sorted({abs(d) for *_, d, _ in rows_neu_and_col})
    subsets = {
        "whole": lambda q, d: True,
        "half A": lambda q, d: q in HALVES[0],
        "half B": lambda q, d: q in HALVES[1],
        "lower": lambda q, d: abs(d) == mags[0],
        "upper": lambda q, d: abs(d) == mags[-1],
    }

    units = [("neuron", n) for n in neurons] + [("col", c) for c in cols] + [("iso", c) for c in isos]
    ui = {u: i for i, u in enumerate(units)}
    acc = {(nm, s): np.zeros((len(units), Vn), dtype=np.float32) for nm in subsets for s in (+1, -1)}
    cnt = {(nm, s): np.zeros(len(units), dtype=np.float32) for nm in subsets for s in (+1, -1)}

    for kind, n, q, d, ids in rows_neu_and_col:
        s = 1 if d > 0 else -1
        u = ui[("neuron" if kind == "neuron" else "col", n)]
        hits = [vi[w0] for w0 in (_norm_word(VN[t]) for t in ids) if w0 in vi]
        vec = np.bincount(hits, minlength=Vn).astype(np.float32) / max(len(ids), 1) if hits else None
        for nm, pred in subsets.items():
            if pred(q, d):
                cnt[(nm, s)][u] += 1.0
                if vec is not None:
                    acc[(nm, s)][u] += vec

    for _kind, n, q, d, ids in rows_iso:
        s = 1 if d > 0 else -1
        u = ui[("iso", n)]
        hits = [vi[w0] for w0 in (_norm_word(VN[t]) for t in ids) if w0 in vi]
        vec = np.bincount(hits, minlength=Vn).astype(np.float32) / max(len(ids), 1) if hits else None
        for nm, pred in subsets.items():
            if pred(q, d):
                cnt[(nm, s)][u] += 1.0
                if vec is not None:
                    acc[(nm, s)][u] += vec

    sub_out: Dict[str, Any] = {}
    neu_scores_by_sub: Dict[str, np.ndarray] = {}
    q_col_whole: Optional[np.ndarray] = None
    q_iso_whole: Optional[np.ndarray] = None
    perm = np.random.default_rng(20261126).permutation(nn)
    while np.any(perm == np.arange(nn)):
        perm = np.random.default_rng(20261127).permutation(nn)

    for nm in subsets:
        F_mat = {s: (acc[(nm, s)] / np.maximum(cnt[(nm, s)], 1.0)[:, None]).astype(np.float32) for s in (+1, -1)}
        fn = {s: F_mat[s][:nn] for s in (+1, -1)}
        fc = {s: F_mat[s][nn : nn + C_col] for s in (+1, -1)}
        fi = {s: F_mat[s][nn + C_col :] for s in (+1, -1)}

        # Own-word diagonal scores
        own_n = np.diag(0.5 * (fn[1] @ M_neu[1] + fn[-1] @ M_neu[-1]))
        shuf_n = 0.5 * ((fn[1] * M_neu[1][:, perm].T).sum(axis=1) + (fn[-1] * M_neu[-1][:, perm].T).sum(axis=1))
        own_c = np.diag(0.5 * (fc[1] @ M_col[1] + fc[-1] @ M_col[-1]))
        own_i = np.diag(0.5 * (fi[1] @ M_iso[1] + fi[-1] @ M_iso[-1]))

        neu_scores_by_sub[nm] = own_n

        def _vs_pool(obs: np.ndarray, null_vals: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, int]:
            auc_v, mw_p = mannwhitney_greater(obs, null_vals)
            s_null = np.sort(null_vals)
            C_len = len(s_null)
            ge = C_len - np.searchsorted(s_null, obs, side="left")
            ps = (1.0 + ge) / (1.0 + C_len)
            qs = _bh(ps)
            return float(auc_v), ps, qs, int(np.sum(obs > float(s_null[-1])))

        auc_col, p_col, q_col, max_col = _vs_pool(own_n, own_c)
        auc_iso, p_iso, q_iso, max_iso = _vs_pool(own_n, own_i)
        auc_shuf_col, _, _, _ = _vs_pool(shuf_n, own_c)

        res_sub: Dict[str, Any] = {
            "auc_vs_col": round(auc_col, 4),
            "q05_vs_col": int(np.sum(q_col <= 0.05)),
            "above_max_col": max_col,
            "auc_vs_iso": round(auc_iso, 4),
            "q05_vs_iso": int(np.sum(q_iso <= 0.05)),
            "above_max_iso": max_iso,
            "dual_q05": int(np.sum((q_col <= 0.05) & (q_iso <= 0.05))),
            "shuf_auc_vs_col": round(auc_shuf_col, 4),
            "mean_own_n": round(float(own_n.mean()), 5),
            "mean_own_col": round(float(own_c.mean()), 5),
            "mean_own_iso": round(float(own_i.mean()), 5),
        }
        if nm == "whole":
            q_col_whole, q_iso_whole = q_col, q_iso
            _, _, q_i2c, _ = _vs_pool(own_i, own_c)
            _, _, q_c2i, _ = _vs_pool(own_c, own_i)
            res_sub["cross_pool"] = {
                "iso_false_q05_vs_col": int(np.sum(q_i2c <= 0.05)),
                "iso_false_frac": round(float(np.mean(q_i2c <= 0.05)), 4),
                "col_false_q05_vs_iso": int(np.sum(q_c2i <= 0.05)),
                "col_false_frac": round(float(np.mean(q_c2i <= 0.05)), 4),
            }
        sub_out[nm] = res_sub

    rho_ab = _rank_corr(neu_scores_by_sub["half A"], neu_scores_by_sub["half B"])
    w = sub_out["whole"]
    ha = sub_out["half A"]
    hb = sub_out["half B"]
    g_ww_lead = (
        w["auc_vs_col"] >= 0.60
        and w["auc_vs_iso"] >= 0.60
        and ha["auc_vs_col"] >= 0.60
        and ha["auc_vs_iso"] >= 0.60
        and hb["auc_vs_col"] >= 0.60
        and hb["auc_vs_iso"] >= 0.60
        and 0.47 <= w["shuf_auc_vs_col"] <= 0.53
        and rho_ab >= 0.50
    )
    min_dual_ww = max(1, int(np.ceil(0.10 * nn)))  # 103 for 1,022
    xp = w["cross_pool"]
    g_ww_yield = (
        w["dual_q05"] >= min_dual_ww
        and xp["iso_false_frac"] <= 0.02
        and xp["col_false_frac"] <= 0.02
    )
    return {
        "neurons": nn,
        "c_col": C_col,
        "c_iso": C_iso,
        "rho_AB_neurons": round(rho_ab, 4),
        "subsets": sub_out,
        "gates": {
            "G_WW_Lead": bool(g_ww_lead),
            "G_WW_Yield": bool(g_ww_yield),
        },
        "pass": bool(g_ww_lead and g_ww_yield),
        "_dual_named_neurons": [
            int(neurons[i])
            for i in range(nn)
            if q_col_whole is not None and q_iso_whole is not None and q_col_whole[i] <= 0.05 and q_iso_whole[i] <= 0.05
        ],
    }


def evaluate_track_2a(
    *,
    res_col: Dict[str, Any],
    res_iso: Dict[str, Any],
    c4_col: Dict[str, Any],
    c4_iso: Dict[str, Any],
    iso_vs_col_res: Dict[str, Any],
    col_vs_iso_res: Dict[str, Any],
    baseline_match_ok: bool,
    complete_ok: bool,
) -> Dict[str, Any]:
    """Evaluate Pre-Registered Track 2A (Behavioral U2-T_dec) gates."""
    n3_c = res_col["n3_covered"]
    n3_i = res_iso["n3_covered"]
    n_neu = int(n3_c["neurons"])

    c_cov_names = {r["neuron"] for r in res_col["_named_records"] if r["covered"]}
    i_cov_names = {r["neuron"] for r in res_iso["_named_records"] if r["covered"]}
    dual_n3 = sorted(c_cov_names & i_cov_names)
    min_yield = max(1, int(np.ceil(0.08 * n_neu)))  # 82 for 1,022

    bc = n3_c.get("bootstrap_fdp05") or {"p10": 0, "frac_zero": 1.0}
    bi = n3_i.get("bootstrap_fdp05") or {"p10": 0, "frac_zero": 1.0}
    g_yield = (
        n3_c["share_covered"] >= 0.50
        and n3_i["share_covered"] >= 0.50
        and len(dual_n3) >= min_yield
        and bc["p10"] >= 50
        and bc["frac_zero"] == 0.0
        and bi["p10"] >= 50
        and bi["frac_zero"] == 0.0
    )

    n_iso_pool = int(iso_vs_col_res["n3_covered"]["neurons"])
    n_col_pool = int(col_vs_iso_res["n3_covered"]["neurons"])
    i2c_cnt = int(iso_vs_col_res["fdp05_count"] if "fdp05_count" in iso_vs_col_res else iso_vs_col_res["fdp05"]["count"])
    c2i_cnt = int(col_vs_iso_res["fdp05_count"] if "fdp05_count" in col_vs_iso_res else col_vs_iso_res["fdp05"]["count"])
    i2c_frac = i2c_cnt / max(n_iso_pool, 1)
    c2i_frac = c2i_cnt / max(n_col_pool, 1)
    g_cross = (i2c_frac <= 0.02) and (c2i_frac <= 0.02)

    g_c4 = (
        int(c4_col.get("fit_named") or 0) >= 40
        and float(c4_col.get("frac_p05_on_C4") or 0.0) >= 0.80
        and float(c4_col.get("c4_auc_vs_paired_ctl") or 0.0) >= 0.85
    )

    sym_c = int((res_col.get("code_symmetry_check") or {}).get("fdp05_false_max", 99))
    sym_i = int((res_iso.get("code_symmetry_check") or {}).get("fdp05_false_max", 99))
    g_sanity = bool(baseline_match_ok and complete_ok and sym_c <= 1 and sym_i <= 1)

    return {
        "pass": bool(g_yield and g_cross and g_c4 and g_sanity),
        "gates": {
            "G_Yield": g_yield,
            "G_CrossPool": g_cross,
            "G_C4": g_c4,
            "G_Sanity": g_sanity,
        },
        "dual05_n3_count": len(dual_n3),
        "dual05_n3_min_bar": min_yield,
        "vs_col": {k: v for k, v in res_col.items() if k != "_named_records"},
        "vs_iso": {k: v for k, v in res_iso.items() if k != "_named_records"},
        "c4_vs_col": c4_col,
        "c4_vs_iso": c4_iso,
        "cross_pool": {
            "iso_false_vs_col_count": i2c_cnt,
            "iso_false_vs_col_frac": round(i2c_frac, 4),
            "col_false_vs_iso_count": c2i_cnt,
            "col_false_vs_iso_frac": round(c2i_frac, 4),
        },
        "top20_named_examples": [
            r for r in res_col["_named_records"] if r["neuron"] in set(dual_n3)
        ][:20],
    }


def _compute_gpu_control_eyes(layer: int, n_col: int, seed_col: int, n_iso: int, seed_iso: int) -> Tuple[List[List[int]], List[List[int]], int]:
    """Compute exact control eye_10 IDs on GPU (or CPU), returning (col_eyes, iso_eyes, overlap_20)."""
    import torch
    from backend import atlas_ops as ao
    from backend.causal_check import language_model, logits_fp32, model_device
    from backend.runtime_factory import make_runtime

    data_dir = ai.atlas_dir()
    lens_npz = np.load(data_dir / "index/lens.npz")
    eye_pos_all, eye_neg_all = lens_npz["eye_pos"], lens_npz["eye_neg"]

    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    lm = language_model(rt.hf_model)
    w_ops = ao.Weights(lm, rt)
    dev = model_device(lm)
    fnw, W_U = rt.final_norm_weight, rt.lm_head_weight

    # Verify 20-neuron lens overlap sanity
    test_ns = list(range(0, 2000, 100))[:20]
    wp = w_ops.norm(layer, "post_feedforward").to(dev)
    vecs_s = (w_ops.down_cols(layer, test_ns) * wp).to(W_U.dtype)
    logits_s = logits_fp32(gemma4_rms_norm(vecs_s, fnw), W_U)
    gp_ids = label_top_ids(vecs_s, fnw, W_U, logits_s, k=10)
    ov_p = sum(len(set(gp_ids[i]) & set(int(x) for x in eye_pos_all[layer, test_ns[i], :10])) for i in range(20))
    del vecs_s, logits_s

    w_post_np = w_ops.norm(layer, "post_feedforward").cpu().numpy()
    writes = np.concatenate(
        [w_ops.write_dirs(layer, list(range(a, min(a + 4096, w_ops.d_ff)))).cpu().numpy() for a in range(0, w_ops.d_ff, 4096)]
    )
    col_dirs = cz.control_dirs(seed_col, n_col, w_post_np, "columns", writes)
    iso_dirs = cz.control_dirs(seed_iso, n_iso, w_post_np, "isotropic", None)
    del writes

    def _eye_pair(dirs_np: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
        ep, en = [], []
        for s in range(0, len(dirs_np), 128):
            cv = torch.from_numpy(dirs_np[s : s + 128]).to(device=dev, dtype=W_U.dtype)
            hn = gemma4_rms_norm(cv, fnw)
            cl = logits_fp32(hn, W_U)
            ep.extend(label_top_ids(cv, fnw, W_U, cl, k=10))
            en.extend(label_top_ids(-cv, fnw, W_U, -cl, k=10))
            del cv, hn, cl
        return ep, en

    col_ep, col_en = _eye_pair(col_dirs)
    iso_ep, iso_en = _eye_pair(iso_dirs)
    del rt, lm, w_ops
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (col_ep, col_en), (iso_ep, iso_en), int(ov_p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirmatory Path 2: Dual-control U2 rollout reader")
    ap.add_argument("--prereg-dir", default="experiments/results/u2_rollout_trial/prereg")
    ap.add_argument("--trial-dir", default="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/u2_rollout")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out-dir", default="experiments/results/u2_rollout_trial")
    args = ap.parse_args()

    t0 = time.time()
    layer = int(args.layer)
    prereg_dir = Path(args.prereg_dir) if Path(args.prereg_dir).is_absolute() else REPO_ROOT / args.prereg_dir
    trial_dir = Path(args.trial_dir)
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prereg_info = verify_prereg_integrity(prereg_dir, REPO_ROOT)

    cols_run = trial_dir / f"L{layer}-match_things-u2-cols.jsonl"
    iso_run = trial_dir / f"L{layer}-match_things-u2-iso.jsonl"
    joined_iso = trial_dir / f"L{layer}-match_things-u2-vs-iso.jsonl"
    if not cols_run.is_file() or not iso_run.is_file():
        raise RuntimeError(f"Missing trial jsonl files at L{layer}: {cols_run}, {iso_run}")

    # Ensure units + collapse.json exist for cols_run
    if not cz.collapse_path_of(cols_run).is_file() or not cz.units_path_of(cols_run).is_file():
        cz.write_units(cols_run)
    # Build joined run (neurons from cols_run + isotropic controls from iso_run), verifying 12/12 exact baselines
    join_info = cz.join_controls(cols_run, iso_run, joined_iso)
    cz.write_units(joined_iso)

    # Also write iso_run collapse if needed via quick units write on joined; compute iso_run collapse via a dummy join if absent
    m_c = json.loads(cz.meta_path_of(cols_run).read_text(encoding="utf-8"))
    m_i = json.loads(cz.meta_path_of(iso_run).read_text(encoding="utf-8"))
    subsets = _build_subsets(
        [q for q in cz.DESIGN_QUESTIONS["match_things"]],
        [float(d) for _, d in m_c["design"]],
    )

    run_vs_col = cf.StreamedRun(cols_run, subsets, max_new=48, load_collapse=True)
    run_vs_iso = cf.StreamedRun(joined_iso, subsets, max_new=48, load_collapse=True)

    # Verify baseline match + completeness (join_controls already verifies both runs hit jobs["n"])
    n_expected_joined = len(run_vs_iso.neurons) * 48 + len(run_vs_iso.controls) * 48
    base_ok = (len(run_vs_col.baseline_words) > 0) and (set(run_vs_col.baseline_words) == set(run_vs_iso.baseline_words))
    complete_ok = (
        len(run_vs_col.neurons) > 0
        and len(run_vs_col.controls) == int(m_c["random_controls"])
        and len(run_vs_iso.controls) == int(m_i["random_controls"])
        and int(join_info["rows"]) == n_expected_joined + len(cz.DESIGN_QUESTIONS["match_things"])
    )

    res_col = cf.eval_pair(run_vs_col, "A", "B", "all", strip_thought=True, strip_collapse=True, alpha=0.05, run_null_splits=50, n_boot=200, seed=layer)
    res_iso = cf.eval_pair(run_vs_iso, "A", "B", "all", strip_thought=True, strip_collapse=True, alpha=0.05, run_null_splits=50, n_boot=200, seed=layer + 100)
    c4_col = cf.out_of_sample_3way(run_vs_col, strip_thought=True, strip_collapse=True, alpha=0.05)
    c4_iso = cf.out_of_sample_3way(run_vs_iso, strip_thought=True, strip_collapse=True, alpha=0.05)

    # Bidirectional cross-pool T_dec exchangeability in temp dir
    with tempfile.TemporaryDirectory() as tmpd:
        td = Path(tmpd)
        p_i2c = td / f"L{layer}_i2c.jsonl"
        p_c2i = td / f"L{layer}_c2i.jsonl"
        _write_cross_pool_run(
            fake_neu_path=iso_run,
            fake_neu_collapse=cz.collapse_path_of(joined_iso),
            ctl_path=cols_run,
            ctl_collapse=cz.collapse_path_of(cols_run),
            out_jsonl=p_i2c,
        )
        _write_cross_pool_run(
            fake_neu_path=cols_run,
            fake_neu_collapse=cz.collapse_path_of(cols_run),
            ctl_path=iso_run,
            ctl_collapse=cz.collapse_path_of(joined_iso),
            out_jsonl=p_c2i,
        )
        r_i2c = cf.StreamedRun(p_i2c, subsets, max_new=48, load_collapse=True)
        r_c2i = cf.StreamedRun(p_c2i, subsets, max_new=48, load_collapse=True)
        res_i2c = cf.eval_pair(r_i2c, "A", "B", "all", strip_thought=True, strip_collapse=True, alpha=0.05, run_null_splits=0, n_boot=0, seed=layer)
        res_c2i = cf.eval_pair(r_c2i, "A", "B", "all", strip_thought=True, strip_collapse=True, alpha=0.05, run_null_splits=0, n_boot=0, seed=layer)

    track_2a = evaluate_track_2a(
        res_col=res_col,
        res_iso=res_iso,
        c4_col=c4_col,
        c4_iso=c4_iso,
        iso_vs_col_res=res_i2c,
        col_vs_iso_res=res_c2i,
        baseline_match_ok=base_ok,
        complete_ok=complete_ok,
    )

    # Track 2B: Corrected Weight-Word Verification
    data_dir = ai.atlas_dir()
    vocab_list = json.loads((data_dir / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    VN = [_norm_word(t) for t in vocab_list]
    lens_npz = np.load(data_dir / "index/lens.npz")
    eye_pos_all, eye_neg_all = lens_npz["eye_pos"], lens_npz["eye_neg"]

    (col_ep, col_en), (iso_ep, iso_en), ov_20 = _compute_gpu_control_eyes(
        layer, int(m_c["random_controls"]), int(m_c["seed"]), int(m_i["random_controls"]), int(m_i["seed"])
    )
    rows_c = [
        (r["kind"], int(r["neuron"]), r["prompt"], float(r["dose"]), r.get("token_ids") or [])
        for r in cz.read_rows(cols_run)
        if r["kind"] != "baseline"
    ]
    rows_i = [
        (r["kind"], int(r["neuron"]), r["prompt"], float(r["dose"]), r.get("token_ids") or [])
        for r in cz.read_rows(iso_run)
        if r["kind"] != "baseline"
    ]
    neu_list = sorted({n for k, n, *_ in rows_c if k == "neuron"})
    neu_ep = {n: [int(x) for x in eye_pos_all[layer, n, :10]] for n in neu_list}
    neu_en = {n: [int(x) for x in eye_neg_all[layer, n, :10]] for n in neu_list}

    track_2b = compute_weight_word_metrics(
        rows_neu_and_col=rows_c,
        rows_iso=rows_i,
        VN=VN,
        neu_eye_pos=neu_ep,
        neu_eye_neg=neu_en,
        col_eye_pos=col_ep,
        col_eye_neg=col_en,
        iso_eye_pos=iso_ep,
        iso_eye_neg=iso_en,
    )
    track_2b["lens_overlap_pos_20"] = ov_20
    ww_named = [int(n) for n in track_2b.pop("_dual_named_neurons")]
    track_2b["dual_q05_top20"] = ww_named[:20]

    verdict = {
        "layer": layer,
        "git_commit": prereg_info["git_head"],
        "git_dirty": False,
        "code_sha256": gp.build_provenance(REPO_ROOT, [__file__, REPO_ROOT / "backend/census_fdr.py", prereg_dir / "SHA256SUMS"], started_at=gp.utc_iso(t0), finished_at=gp.utc_iso(time.time()))["code_sha256"],
        "prereg": prereg_info,
        "elapsed_s": round(time.time() - t0, 2),
        "join_rows": int(join_info["rows"]),
        "pass_track_2a": bool(track_2a["pass"]),
        "pass_track_2b": bool(track_2b["pass"]),
        "pass_either": bool(track_2a["pass"] or track_2b["pass"]),
        "pass": bool(track_2a["pass"] or track_2b["pass"]),
        "track_2a_u2_tdec": track_2a,
        "track_2b_u2_weight_words": track_2b,
    }
    v_path = out_dir / f"L{layer}_verdict.json"
    v_path.write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    (trial_dir / f"L{layer}_verdict.json").write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    print(
        f"L{layer} U2 VERDICT ({verdict['elapsed_s']}s): pass_2a={verdict['pass_track_2a']} "
        f"({track_2a['dual05_n3_count']}/1022, gates={track_2a['gates']}) | "
        f"pass_2b={verdict['pass_track_2b']} (ww_dual={track_2b['subsets']['whole']['dual_q05']}/1022, gates={track_2b['gates']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
