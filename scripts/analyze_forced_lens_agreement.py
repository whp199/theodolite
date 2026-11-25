#!/usr/bin/env python3
"""Compute CPU weight-lens agreement and unconditional decoding overlap for L55-L59 forced_odd."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import forced_odd as fo  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_LEX_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _extract_fo_ids(arrs: Dict[str, np.ndarray], n: int, k: int = 5) -> Tuple[List[int], List[int]]:
    """Reproduce extract_replicated_pieces positive and negative piece IDs in descending score order."""
    rep_pos: List[Tuple[float, int]] = []
    rep_neg: List[Tuple[float, int]] = []
    seen_p, seen_n = set(), set()
    width = int(arrs["topA_p"].shape[1])
    for j in range(width):
        pid = int(arrs["topA_p"][n, j])
        scp = float(arrs["posA_onB"][n, j])
        if scp > 0.0 and pid not in seen_p:
            seen_p.add(pid)
            rep_pos.append((scp, pid))

        pid2 = int(arrs["topB_p"][n, j])
        scp2 = float(arrs["posB_onA"][n, j])
        if scp2 > 0.0 and pid2 not in seen_p:
            seen_p.add(pid2)
            rep_pos.append((scp2, pid2))

        nid = int(arrs["topA_n"][n, j])
        scn = float(arrs["negA_onB"][n, j])
        if scn < 0.0 and nid not in seen_n:
            seen_n.add(nid)
            rep_neg.append((-scn, nid))

        nid2 = int(arrs["topB_n"][n, j])
        scn2 = float(arrs["negB_onA"][n, j])
        if scn2 < 0.0 and nid2 not in seen_n:
            seen_n.add(nid2)
            rep_neg.append((-scn2, nid2))

    rep_pos.sort(key=lambda x: -x[0])
    rep_neg.sort(key=lambda x: -x[0])
    return [pid for _, pid in rep_pos[:k]], [nid for _, nid in rep_neg[:k]]


def _group_metrics(
    indices: Sequence[int],
    fo_pos_all: List[List[int]],
    fo_neg_all: List[List[int]],
    eye_pos: np.ndarray,
    eye_neg: np.ndarray,
    perm: np.ndarray,
    eye_k: int = 12,
) -> Dict[str, Any]:
    if not indices:
        return {"count": 0}
    pos_ge1 = 0
    pos_ge3 = 0
    both_ge1 = 0
    both_ge3 = 0
    top1_in_eye = 0
    flip_pos_in_neg = 0
    shuf_pos_ge1 = 0
    shuf_both_ge1 = 0
    shares: List[float] = []

    for n in indices:
        fp = fo_pos_all[n]
        fn = fo_neg_all[n]
        ep = set(int(x) for x in eye_pos[n, :eye_k])
        en = set(int(x) for x in eye_neg[n, :eye_k])
        m_pos = sum(1 for pid in fp if pid in ep)
        m_neg = sum(1 for nid in fn if nid in en)
        m_both = m_pos + m_neg
        if m_pos >= 1:
            pos_ge1 += 1
        if m_pos >= 3:
            pos_ge3 += 1
        if m_both >= 1:
            both_ge1 += 1
        if m_both >= 3:
            both_ge3 += 1
        if fp and fp[0] in ep:
            top1_in_eye += 1
        if any(pid in en for pid in fp):
            flip_pos_in_neg += 1
        if fp:
            shares.append(m_pos / len(fp))

        pn = int(perm[n])
        sp_e = set(int(x) for x in eye_pos[pn, :eye_k])
        sn_e = set(int(x) for x in eye_neg[pn, :eye_k])
        sm_pos = sum(1 for pid in fp if pid in sp_e)
        sm_both = sm_pos + sum(1 for nid in fn if nid in sn_e)
        if sm_pos >= 1:
            shuf_pos_ge1 += 1
        if sm_both >= 1:
            shuf_both_ge1 += 1

    denom = len(indices)
    return {
        "count": denom,
        "pos_ge1_rate": round(pos_ge1 / denom, 4),
        "pos_ge3_rate": round(pos_ge3 / denom, 4),
        "pos_top1_in_eye_rate": round(top1_in_eye / denom, 4),
        "pos_mean_share_in_eye": round(float(np.mean(shares)) if shares else 0.0, 4),
        "pos_sign_flip_rate": round(flip_pos_in_neg / denom, 4),
        "both_ge1_rate": round(both_ge1 / denom, 4),
        "both_ge3_rate": round(both_ge3 / denom, 4),
        "shuf_pos_ge1_rate": round(shuf_pos_ge1 / denom, 4),
        "shuf_both_ge1_rate": round(shuf_both_ge1 / denom, 4),
    }


def _find_cjk_examples(
    layer: int,
    named_idx: Sequence[int],
    arrs: Dict[str, np.ndarray],
    fo_pos_all: List[List[int]],
    eye_pos: np.ndarray,
    vocab: Sequence[str],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Find named neurons whose eye_pos contains CJK pieces while forced top words are ASCII cognates."""
    t_min = np.minimum(arrs["T_col"], arrs["T_iso"])
    ranked = sorted((int(n) for n in named_idx), key=lambda n: -float(t_min[n]))
    out: List[Dict[str, Any]] = []
    for n in ranked:
        eye_toks = [vocab[int(t)] for t in eye_pos[n, :8]]
        fo_toks = [vocab[int(t)] for t in fo_pos_all[n][:5]]
        has_cjk_eye = sum(1 for s in eye_toks[:5] if _CJK_RE.search(s)) >= 1
        has_eng_fo = sum(1 for s in fo_toks if _ASCII_WORD_RE.search(s)) >= 2
        if has_cjk_eye and has_eng_fo and len(set(fo_pos_all[n]) & set(int(x) for x in eye_pos[n, :12])) >= 1:
            out.append({
                "layer": layer,
                "neuron": n,
                "T_min": round(float(t_min[n]), 3),
                "is_direct": bool(arrs["is_direct"][n]) if "is_direct" in arrs else None,
                "eye_pos_top8": eye_toks,
                "forced_pos_top5": fo_toks,
            })
            if len(out) >= limit:
                break
    return out


def main() -> None:
    t0 = time.time()
    started_iso = gp.utc_iso(t0)
    data_dir = ai.atlas_dir()
    fo_dir = REPO_ROOT / "experiments/results/census_forced_odd"
    whole_dir = fo_dir / "whole"

    vocab = json.loads((data_dir / "index/vocab.json").read_text(encoding="utf-8"))["vocab"]
    lex_bool = np.array([bool(_LEX_RE.fullmatch(s.replace("▁", "").strip())) and not s.startswith("<") for s in vocab], dtype=bool)
    lz = np.load(data_dir / "index/lens.npz", allow_pickle=False)
    eye_pos_all = np.asarray(lz["eye_pos"])
    eye_neg_all = np.asarray(lz["eye_neg"])
    lens_meta = ai.meta_of(lz)
    p_meta = {}
    pm_path = data_dir / "index/passages.d/meta.json"
    if pm_path.is_file():
        p_meta = json.loads(pm_path.read_text(encoding="utf-8"))

    model_id = lens_meta.get("model") or p_meta.get("model")
    weights = lens_meta.get("weights") or p_meta.get("weights")

    layers_out: List[Dict[str, Any]] = []
    cjk_examples: Dict[str, List[Dict[str, Any]]] = {}

    for layer in range(55, 60):
        npz_path = whole_dir / f"L{layer}_forced_odd.npz"
        with np.load(npz_path, allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
        N_ff = int(arrs["named_dual05"].shape[0])
        named_mask = np.asarray(arrs["named_dual05"], dtype=bool)
        named_idx = np.flatnonzero(named_mask).tolist()
        unnamed_idx = np.flatnonzero(~named_mask).tolist()

        fo_pos_all: List[List[int]] = []
        fo_neg_all: List[List[int]] = []
        for n in range(N_ff):
            fp, fn = _extract_fo_ids(arrs, n, k=5)
            fo_pos_all.append(fp)
            fo_neg_all.append(fn)

        perm = np.random.default_rng(20261125 + layer).permutation(N_ff)
        ep_l = eye_pos_all[layer]
        en_l = eye_neg_all[layer]

        named_stats = _group_metrics(named_idx, fo_pos_all, fo_neg_all, ep_l, en_l, perm)
        unnamed_stats = _group_metrics(unnamed_idx, fo_pos_all, fo_neg_all, ep_l, en_l, perm)

        # LEX share vs direct-tag correlation & OLS residual magnitude check
        eye_lex_share = lex_bool[ep_l[named_idx, :10]].mean(axis=1)
        sw = arrs["sw_neu"][named_idx]
        corr_lex_sw = float(np.corrcoef(eye_lex_share, sw)[0, 1]) if len(named_idx) > 1 else 0.0
        fitted_term = arrs["r_ab_col"] - arrs["r_ab_res_col"]

        cjk_examples[str(layer)] = _find_cjk_examples(layer, named_idx, arrs, fo_pos_all, ep_l, vocab, limit=6)

        meta_p = whole_dir / f"L{layer}_forced_odd.meta.json"
        whole_meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        kl_n = whole_meta.get("kl_neurons") or {}

        layers_out.append({
            "layer": layer,
            "d_ff": N_ff,
            "named_dual05": len(named_idx),
            "kl_neurons": kl_n,
            "named": named_stats,
            "unnamed": unnamed_stats,
            "corr_eye_lex_share_vs_sw": round(corr_lex_sw, 4),
            "ols_fitted_term_span_cols": round(float(np.max(fitted_term) - np.min(fitted_term)), 4),
            "ols_fitted_term_p10_p90": [round(float(np.percentile(fitted_term, 10)), 4), round(float(np.percentile(fitted_term, 90)), 4)],
        })

        # Backfill missing provenance & lens agreement into whole meta and trial verdict without touching data arrays
        if meta_p.is_file():
            whole_meta["model"] = model_id
            whole_meta["weights"] = weights
            whole_meta["weights_id"] = gp.weights_id_from_card(weights)
            whole_meta["design_name"] = "forced_odd"
            whole_meta["neuron_count"] = N_ff
            whole_meta["neuron_ids"] = "all"
            whole_meta["weight_lens_agreement"] = {
                "named": named_stats,
                "unnamed_both_ge1_rate": unnamed_stats["both_ge1_rate"],
            }
            whole_meta["legacy_git_commit_note"] = (
                "Originally run locally atop base 16570312; code & registration committed at 5452734; see prereg/AMENDMENTS.txt."
            )
            meta_p.write_text(json.dumps(whole_meta, indent=1) + "\n", encoding="utf-8")

        verdict_p = fo_dir / "trials" / f"L{layer}_verdict.json"
        nfile_p = fo_dir / "prereg" / f"neurons_L{layer}.txt"
        if verdict_p.is_file() and nfile_p.is_file():
            vdoc = json.loads(verdict_p.read_text(encoding="utf-8"))
            nids = [int(tok.strip()) for tok in nfile_p.read_text(encoding="utf-8").replace("\n", ",").split(",") if tok.strip()]
            vdoc["model"] = model_id
            vdoc["weights"] = weights
            vdoc["weights_id"] = gp.weights_id_from_card(weights)
            vdoc["design_name"] = "forced_odd"
            vdoc["neuron_count"] = len(nids)
            vdoc["neuron_ids"] = ",".join(str(x) for x in nids)
            vdoc["exploratory"] = True
            vdoc["legacy_git_commit_note"] = (
                "Registration hashed locally 2026-11-24T23:52:42Z (committed 5452734); F2.2 and asymmetric F3 Half-C are exploratory/vacuous."
            )
            verdict_p.write_text(json.dumps(vdoc, indent=1) + "\n", encoding="utf-8")

    # Calibration unconditional vs conditional decoding-census overlap at L37, L45, L54
    calib_overlap: Dict[str, Any] = {}
    for cl in (37, 45, 54):
        cp = fo_dir / "calibration_p90" / f"calib_L{cl}.json"
        if cp.is_file():
            cd = json.loads(cp.read_text(encoding="utf-8"))
            ov = cd["combos"]["proj/LEX_L2_cent"].get("decoding_overlap") or {}
            calib_overlap[f"L{cl}"] = ov

    payload = {
        **gp.build_provenance(
            REPO_ROOT,
            [Path(__file__), REPO_ROOT / "backend" / "forced_odd.py"],
            started_at=started_iso,
            finished_at=gp.utc_iso(),
            model_id=model_id,
            weights=weights,
            neuron_ids={"range": [0, 21504], "layers": [55, 56, 57, 58, 59]},
        ),
        "elapsed_s": round(time.time() - t0, 2),
        "layers": layers_out,
        "cjk_cognate_examples": cjk_examples,
        "calibration_decoding_overlap": calib_overlap,
    }
    (whole_dir / "lens_agreement.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    lines = [
        "forced_odd vs Weight-Lens Agreement (eye_pos/eye_neg top-12 in lens.npz, whole layers L55-L59)",
        "-" * 116,
        f"{'L':>3} {'Dual05':>7} {'Pos>=1':>7} {'Pos>=3':>7} {'Both>=1':>8} {'Both>=3':>8} "
        f"{'Top1':>6} {'Shuf+>=1':>8} {'UnnBoth>=1':>11} {'corr(LEX,sw)':>12}",
    ]
    for r in layers_out:
        nm, un = r["named"], r["unnamed"]
        lines.append(
            f"{r['layer']:3d} {r['named_dual05']:7d} {nm['pos_ge1_rate']:7.3f} {nm['pos_ge3_rate']:7.3f} "
            f"{nm['both_ge1_rate']:8.3f} {nm['both_ge3_rate']:8.3f} {nm['pos_top1_in_eye_rate']:6.3f} "
            f"{nm['shuf_pos_ge1_rate']:8.3f} {un['both_ge1_rate']:11.3f} {r['corr_eye_lex_share_vs_sw']:12.3f}"
        )
    lines.append("\nWorked CJK eye_pos -> English cognate forced-shift examples:")
    for layer_s, exs in cjk_examples.items():
        for ex in exs[:3]:
            lines.append(
                f"  L{layer_s} n={ex['neuron']:5d} (T_min={ex['T_min']}): "
                f"eye_pos={ex['eye_pos_top8'][:5]} -> forced={ex['forced_pos_top5']}"
            )
    if calib_overlap:
        lines.append("\nStage-1 Calibration decoding-census overlap (proj/LEX_L2_cent):")
        for lk, ov in calib_overlap.items():
            q = ov.get("q05") or {}
            a = ov.get("all") or {}
            lines.append(
                f"  {lk}: checked={ov.get('checked')} | vs q05 census-named: {q.get('matched')}/{q.get('checked_in_ref')} "
                f"(cond={q.get('frac_of_checked')}, uncond={q.get('frac_of_cand')}) | "
                f"vs all shown_words: {a.get('matched')}/{a.get('checked_in_ref')} (uncond={a.get('frac_of_cand')})"
            )
    txt = "\n".join(lines) + "\n"
    (whole_dir / "lens_agreement.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
