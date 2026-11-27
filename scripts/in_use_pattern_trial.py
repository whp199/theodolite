#!/usr/bin/env python3
"""Confirmatory Path 3 (P3): Local Gemma-4-31B-it in_use_pattern trial at L2, L14, L20.

Evaluates pre-registered frozen fresh doc_4_8 faces (128 at L2, 128 at L14, 64 boundary
probe at L20) completely locally on Gemma-4-31B-it:
  - Fit fold: <=8 unique doc_h0 events (preferring pure_h0).
  - Test fold: <=10 unique doc_h1 events (preferring pure_h1), paired with live GPU
    quiet-verified (<0.25x) length-symmetric near misses on doc_h1 outside own_docs.
  - K=3 deranged same-layer other_pattern nulls in single-prefill blind_near scoring.
  - Physical counterfactual behind synthesis behind an UNSEEN doc_h1 passage (tested via
    _StopAfterLayer(l) on own_dropped, 3 FIRE, 3 QUIET, and 3 other-face FIRE endings).

Verifies clean git tree, SHA256SUMS, and CODE_SHA256SUMS at startup.
Never stores raw corpus snippets (kept_tail, fit_snippets, a_txt, b_txt) on disk.
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_ALLOC_CONF_NAMES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
if not any(os.environ.get(k) for k in _ALLOC_CONF_NAMES):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas as at_mod  # noqa: E402
from backend import atlas_index as ai  # noqa: E402
from backend import atlas_ops as ao  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import pattern_tests as pt  # noqa: E402
from backend import w4a16_fast  # noqa: E402
from backend.causal_check import language_model, logits_fp32, model_device, pad_right  # noqa: E402
from backend.gap_stats import wilcoxon_greater  # noqa: E402
from backend.runtime_factory import make_runtime, weights_card  # noqa: E402
from scripts.in_use_common import collect_face_events, load_passages_and_doc_halves  # noqa: E402
from scripts.in_use_context_cpu_probe import _special_token_ids  # noqa: E402
from scripts.in_use_pattern_pilot import (  # noqa: E402
    AUTHOR_SYSTEM,
    BEHIND_SYSTEM,
    SINGLE_PAIR_SYSTEM,
    _StopAfterLayer,
    _author_prompt,
    _behind_prompt,
    _generate_batch,
    _parse_behind_endings,
    _read_multi_neurons_to_layer,
    _score_ab_prefills,
    _single_pair_prompt,
    _snippet_span_text,
    _snippet_text,
    _window_ids,
)


def verify_prereg_integrity(prereg_dir: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    st = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    if st.stdout.strip():
        raise RuntimeError(f"Refusing confirmatory run on dirty git tree:\n{st.stdout}")
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


def build_doc_h1_token_index(
    windows: np.ndarray,
    half_id: np.ndarray,
    doc_id: np.ndarray,
    special_ids: Set[int],
    *,
    min_at: int = 8,
    min_after: int = 8,
) -> Dict[str, Any]:
    """Index all valid non-special tokens in half_id==1 for doc_4_8 quiet near-miss lookup."""
    W_tot, seq_len = windows.shape
    mask = np.zeros_like(windows, dtype=bool)
    mask[:, min_at : seq_len - min_after] = True
    mask &= (half_id == 1) & (windows >= 0)
    if special_ids:
        mask &= ~np.isin(windows, list(special_ids))
    ws, ats = np.nonzero(mask)
    toks = windows[ws, ats].astype(np.int64)
    docs = doc_id[ws, ats].astype(np.int32)
    order = np.argsort(toks, kind="mergesort")
    toks_s = toks[order]
    ws_s = ws[order].astype(np.int32)
    ats_s = ats[order].astype(np.int32)
    docs_s = docs[order]
    uniq, starts, counts = np.unique(toks_s, return_index=True, return_counts=True)
    bounds = {int(t): (int(s), int(s + c)) for t, s, c in zip(uniq, starts, counts)}
    h1_wins = np.unique(ws)
    return {"bounds": bounds, "w": ws_s, "a": ats_s, "d": docs_s, "h1_wins": h1_wins}


def k3_derangement_offsets(N: int) -> Tuple[int, int, int]:
    """Return 3 distinct nonzero cyclic offsets in [1..N-1] for K=3 shuffled other_pattern nulls."""
    assert N >= 4, f"Need at least 4 faces per layer for K=3 deranged nulls, got {N}"
    cands = [1, max(2, N // 4), max(3, N // 2)]
    offs: List[int] = []
    for c in cands:
        x = c % N
        while x == 0 or x in offs:
            x = (x + 1) % N
        offs.append(x)
    assert len(set(offs)) == 3 and all(0 < o < N for o in offs)
    return offs[0], offs[1], offs[2]


def _load_frozen_faces(path: Path) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ns, fc = line.split(",")
        assert fc in ("pos", "neg")
        out.append((int(ns), fc))
    return out


def _select_doc_events(ev: Dict[str, Any], *, n_fit: int = 8, n_test: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Prefer pure_h0/pure_h1, top up from doc_h0/doc_h1, and pick an UNSEEN doc_h1 behind passage."""
    def _merge(primary: Sequence[Dict[str, Any]], secondary: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        seen_w = {int(e["win"]) for e in primary}
        res = list(primary)
        for e in secondary:
            if int(e["win"]) not in seen_w:
                seen_w.add(int(e["win"]))
                res.append(e)
            if len(res) >= limit:
                break
        return res[:limit]

    fit_evs = _merge(ev["pure_h0"], ev["doc_h0"], n_fit)
    test_pool = _merge(ev["pure_h1"], ev["doc_h1"], max(n_test + 2, len(ev["doc_h1"])))
    # Choose test events with at >= 20 so behind cut is valid if needed, and reserve a test event for behind
    test_evs = test_pool[:n_test]
    fit_docs = {int(e["doc_id"]) for e in fit_evs}
    test_docs = {int(e["doc_id"]) for e in test_evs}
    assert not (fit_docs & test_docs), f"Document overlap in fit/test events: {fit_docs & test_docs}"

    behind_ev = None
    # Prefer an unseen doc_h1 event beyond blind_near events if available
    for e in test_pool[n_test:]:
        if int(e["at"]) + 1 - pt.DROP >= 16:
            behind_ev = e
            break
    if behind_ev is None:
        for e in test_pool:
            if int(e["at"]) + 1 - pt.DROP >= 16:
                behind_ev = e
                break
    assert behind_ev is not None and int(behind_ev["doc_id"]) not in fit_docs, "Behind passage leaked into fit docs!"
    return fit_evs, test_evs, behind_ev


def _bh_pvals(pvals: Sequence[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    arr = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(arr)
    q = np.minimum.accumulate((arr[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(q, 1.0)
    return [float(x) for x in out]


def evaluate_p3_layer(
    *,
    layer: int,
    face_keys: Sequence[str],
    items_b: Sequence[Dict[str, Any]],
    own_scores: Sequence[Dict[str, float]],
    other_scores_k: Sequence[Sequence[Dict[str, float]]],  # K=3 lists of length len(items_b)
    len_ratios: Sequence[float],
    behind_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate Pre-Registered Path 3 gates G1-G4 for one layer."""
    K = len(other_scores_k)
    by_face: Dict[str, Dict[str, Any]] = {k: {"truth": [], "own": [], "others": [[] for _ in range(K)], "match": []} for k in face_keys}
    masses: List[float] = []
    own_tot = 0
    oth_tots = [0] * K
    tot = len(items_b)
    for idx_i, it in enumerate(items_b):
        fk = it["face_key"]
        tr = it["truth"]
        so = own_scores[idx_i]
        masses.append(float(so["p_ab"]))
        by_face[fk]["truth"].append(tr)
        by_face[fk]["own"].append(so["pick"])
        by_face[fk]["match"].append(it["match"])
        own_tot += int(so["pick"] == tr)
        for ki in range(K):
            sx = other_scores_k[ki][idx_i]
            by_face[fk]["others"][ki].append(sx["pick"])
            oth_tots[ki] += int(sx["pick"] == tr)
            masses.append(float(sx["p_ab"]))

    faces_ge8 = 0
    own_face_accs: List[float] = []
    mean_oth_face_accs: List[float] = []
    own_pvals_ge8: List[float] = []
    n_p05_own_ge8 = 0
    n_p05_oth_k = [0] * K
    per_face_bn: Dict[str, Any] = {}
    for fk in face_keys:
        fr = by_face[fk]
        npairs = len(fr["truth"])
        if npairs == 0:
            continue
        r_own = pt.blind_result(fr["own"], fr["truth"], fr["match"])
        o_rights = []
        o_ps = []
        for ki in range(K):
            ro = pt.blind_result(fr["others"][ki], fr["truth"], fr["match"])
            o_rights.append(ro["right"])
            o_ps.append(ro["p"])
            if npairs >= 8 and ro["p"] < 0.05:
                n_p05_oth_k[ki] += 1
        own_face_accs.append(r_own["right"] / npairs)
        mean_oth_face_accs.append(float(np.mean(o_rights)) / npairs)
        if npairs >= 8:
            faces_ge8 += 1
            own_pvals_ge8.append(float(r_own["p"]))
            if r_own["p"] < 0.05:
                n_p05_own_ge8 += 1
        per_face_bn[fk] = {
            "pairs": npairs,
            "own_right": r_own["right"],
            "own_p": r_own["p"],
            "other_rights": o_rights,
            "other_ps": o_ps,
        }

    qvals_ge8 = _bh_pvals(own_pvals_ge8)
    n_q10_ge8 = sum(1 for q in qvals_ge8 if q <= 0.10)
    own_share = own_tot / max(tot, 1)
    oth_shares = [c / max(tot, 1) for c in oth_tots]
    mean_oth_share = float(np.mean(oth_shares))
    own_binom_p = pt.binom_tail(own_tot, tot)
    wil_p = float(wilcoxon_greater(np.asarray(own_face_accs), np.asarray(mean_oth_face_accs))) if own_face_accs else 1.0
    med_mass = float(np.median(masses)) if masses else 0.0
    mean_oth_p05_rate = float(np.mean([c / max(faces_ge8, 1) for c in n_p05_oth_k])) if faces_ge8 else 1.0
    frac_ge8 = faces_ge8 / max(len(face_keys), 1)
    frac_p05_ge8 = n_p05_own_ge8 / max(faces_ge8, 1)
    frac_q10_ge8 = n_q10_ge8 / max(faces_ge8, 1)

    g1_pass = (
        own_share >= 0.72
        and (own_share - mean_oth_share) >= 0.18
        and own_binom_p <= 1e-6
        and 0.45 <= mean_oth_share <= 0.55
        and mean_oth_p05_rate <= 0.08
        and wil_p <= 1e-6
        and med_mass >= 0.90
    )
    g2_pass = frac_ge8 >= 0.90 and frac_p05_ge8 >= 0.30 and frac_q10_ge8 >= 0.20

    # G3: Unseen-doc behind causality + other_pattern fire rate
    n_att = len(behind_records)
    valid_beh = [b for b in behind_records if b["own_holds_gt_0.5"]]
    n_valid = len(valid_beh)
    own_holds_rate = n_valid / max(n_att, 1)
    w_tot = sum(b["result"]["fire"] for b in valid_beh)
    w_fired = sum(b["result"]["fired"] for b in valid_beh)
    q_tot = sum(b["result"]["quiet"] for b in valid_beh)
    q_fired = sum(b["result"]["quiet_fired"] for b in valid_beh)
    o_tot = sum(b["other_fire_total"] for b in valid_beh)
    o_fired = sum(b["other_fire_fired"] for b in valid_beh)
    clean_cnt = sum(1 for b in valid_beh if b["result"]["fired"] >= 2 and b["result"]["quiet_fired"] == 0)
    aparts = [float(b["result"]["apart"]) for b in valid_beh if b["result"].get("apart") is not None]

    w_rate = w_fired / max(w_tot, 1)
    q_rate = q_fired / max(q_tot, 1)
    o_rate = o_fired / max(o_tot, 1)
    clean_rate = clean_cnt / max(n_valid, 1)
    mean_apart = float(np.mean(aparts)) if aparts else 0.0

    g3_pass = (
        own_holds_rate >= 0.90
        and w_rate >= 0.30
        and (w_rate - q_rate) >= 0.20
        and q_rate <= 0.10
        and o_rate <= 0.10
        and clean_rate >= 0.25
        and mean_apart >= 0.70
    )

    return {
        "layer": int(layer),
        "n_faces": len(face_keys),
        "pass": bool(g1_pass and g2_pass and g3_pass),
        "gates": {
            "G1_blind_near_pooled_and_null": g1_pass,
            "G2_blind_near_individual_yield": g2_pass,
            "G3_unseen_behind_causality": g3_pass,
        },
        "blind_near": {
            "pairs": tot,
            "own_right": own_tot,
            "own_share": round(own_share, 4),
            "own_binom_p": round(float(own_binom_p), 8),
            "other_shares_k3": [round(s, 4) for s in oth_shares],
            "mean_other_share": round(mean_oth_share, 4),
            "own_minus_mean_other": round(own_share - mean_oth_share, 4),
            "wilcoxon_face_p": float(f"{wil_p:.3e}"),
            "median_p_ab_mass": round(med_mass, 4),
            "faces_ge8_pairs": faces_ge8,
            "frac_faces_ge8_pairs": round(frac_ge8, 4),
            "faces_p05_own_ge8": n_p05_own_ge8,
            "frac_p05_own_ge8": round(frac_p05_ge8, 4),
            "faces_q10_own_ge8": n_q10_ge8,
            "frac_q10_own_ge8": round(frac_q10_ge8, 4),
            "other_p05_counts_ge8": n_p05_oth_k,
            "mean_other_p05_rate_ge8": round(mean_oth_p05_rate, 4),
            "len_ratio_median": round(float(np.median(len_ratios)) if len_ratios else 1.0, 4),
            "len_ratio_iqr": [round(float(np.percentile(len_ratios, 25)), 4), round(float(np.percentile(len_ratios, 75)), 4)] if len_ratios else [1.0, 1.0],
        },
        "behind_unseen": {
            "faces_attempted": n_att,
            "faces_own_holds_gt_0.5": n_valid,
            "own_holds_rate": round(own_holds_rate, 4),
            "written_texts": w_tot,
            "written_fired": w_fired,
            "written_fire_rate": round(w_rate, 4),
            "near_misses": q_tot,
            "near_misses_fired": q_fired,
            "near_miss_false_rate": round(q_rate, 4),
            "other_face_texts": o_tot,
            "other_face_fired": o_fired,
            "other_pattern_fire_rate": round(o_rate, 4),
            "faces_clean_2of3_and_0quiet": clean_cnt,
            "clean_2of3_and_0quiet_rate": round(clean_rate, 4),
            "mean_apart": round(mean_apart, 4),
        },
        "_per_face_bn": per_face_bn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Confirmatory Path 3: Local Gemma-4-31B in_use_pattern trial")
    ap.add_argument("--prereg-dir", default="experiments/results/in_use_pattern_trial/prereg")
    ap.add_argument("--layers", default="2,14,20")
    ap.add_argument("--out-dir", default="/home/eric/bin/ai/theodolite2/atlas_data/census/trials/in_use_pattern")
    ap.add_argument("--results-dir", default="experiments/results/in_use_pattern_trial")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--prefill-batch", type=int, default=16)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    prereg_dir = Path(args.prereg_dir) if Path(args.prereg_dir).is_absolute() else REPO_ROOT / args.prereg_dir
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else REPO_ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    prereg_info = verify_prereg_integrity(prereg_dir, REPO_ROOT)
    print(f"Pre-registration verified: HEAD={prereg_info['git_head'][:12]}", flush=True)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()
    loaded = load_passages_and_doc_halves(data_dir)
    P, act_scale, vocab, windows = loaded["P"], loaded["act_scale"], loaded["vocab"], loaded["windows"]
    doc_id, half_id = loaded["doc_id"], loaded["half_id"]
    pure_h0_mask, pure_h1_mask = loaded["pure_h0_mask"], loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]

    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        rt.jpp_lenses.clear()
    if getattr(rt, "eager_model", None) is not None and hasattr(rt.eager_model, "_lens_jacobians"):
        rt.eager_model._lens_jacobians.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    tok = rt.tokenizer
    marks = at_mod.passage_marks(tok)
    special_ids = _special_token_ids(bos_id, eos_id)
    h1_idx = build_doc_h1_token_index(windows, half_id, doc_id, special_ids, min_at=8, min_after=8)

    ab_ids = {
        "a": [int(tok.encode(s, add_special_tokens=False)[0]) for s in ("a", " a", "A", " A")],
        "b": [int(tok.encode(s, add_special_tokens=False)[0]) for s in ("b", " b", "B", " B")],
    }
    best_apref = ""
    rng_np = np.random.default_rng(2026112603)
    h1_pool = h1_idx["h1_wins"].tolist()

    layer_verdicts: Dict[str, Any] = {}
    for l in layers:
        ckpt_path = results_dir / f"L{l}_verdict.json"
        if ckpt_path.is_file():
            existing = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if existing.get("prereg", {}).get("prereg_sha256") == prereg_info["prereg_sha256"]:
                print(f"L{l}: already completed and matches prereg_sha256, loading checkpoint", flush=True)
                layer_verdicts[str(l)] = existing
                continue

        tl0 = time.time()
        frozen = _load_frozen_faces(prereg_dir / f"faces_L{l}.txt")
        faces_data: List[Dict[str, Any]] = []
        for nid, fc in frozen:
            ev = collect_face_events(
                P, windows, act_scale, pure_h0_mask, pure_h1_mask, doc_id, half_id, l, nid, fc, eos_id=eos_id, bos_id=bos_id
            )
            fit_evs, test_evs, behind_ev = _select_doc_events(ev, n_fit=8, n_test=10)
            own_docs = {int(e["doc_id"]) for e in fit_evs + test_evs} | {int(behind_ev["doc_id"])}
            faces_data.append(
                {
                    "key": f"L{l}-n{nid}{'p' if fc == 'pos' else 'm'}",
                    "layer": int(l),
                    "n": int(nid),
                    "face": fc,
                    "extreme": float(ev["extreme"]),
                    "own_windows": ev["own_windows"],
                    "own_docs": own_docs,
                    "fit_evs": fit_evs,
                    "test_evs": test_evs,
                    "behind_ev": behind_ev,
                }
            )
        N_faces = len(faces_data)
        face_keys = [fd["key"] for fd in faces_data]

        # 1. Author patterns locally from doc_h0 fit events ONLY
        auth_prompts: List[List[int]] = []
        for fd in faces_data:
            snips, reads, follows = [], [], []
            for ev in fd["fit_evs"]:
                ids = _window_ids(windows, ev["win"])
                txt, _, _ = _snippet_text(tok, marks, ids, ev["at"], before=60, after=8)
                snips.append((ev["rel"], txt))
                reads.append(vocab[ids[ev["at"]]])
                if ev["at"] + 1 < len(ids):
                    follows.append(vocab[ids[ev["at"] + 1]])
            pmt = _author_prompt(fd["key"], snips, reads, follows)
            auth_prompts.append(rt.tokenize_prompt(rt.compose_prompt(pmt, system=AUTHOR_SYSTEM, chat=True, thinking=False)))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_auth = time.time()
        pats_txt = _generate_batch(rt, lm, auth_prompts, batch_size=int(args.batch_size), max_new=96)
        gemma_patterns = {fd["key"]: ptxt for fd, ptxt in zip(faces_data, pats_txt)}
        print(f"L{l}: authored {N_faces} patterns in {time.time() - t_auth:.1f}s", flush=True)

        # 2. Gather candidate symmetric quiet near misses in doc_h1 outside own_docs & own_windows
        pair_slots: List[Dict[str, Any]] = []
        bounds = h1_idx["bounds"]
        for f_idx, fd in enumerate(faces_data):
            for ev_i, ev in enumerate(fd["test_evs"]):
                ids = _window_ids(windows, ev["win"])
                at = int(ev["at"])
                fire_txt, s_f, e_f = _snippet_text(tok, marks, ids, at, before=60, after=8)
                before_f, after_f = at - s_f, e_f - at - 1
                tok0 = int(ids[at])
                prev = int(ids[at - 1]) if at >= 1 else None
                nxt = int(ids[at + 1]) if at + 1 < len(ids) else None
                cand_q: List[Tuple[int, int, str, int, int, List[int], str, float]] = []
                seen_q: Set[Tuple[int, int]] = set()
                if tok0 in bounds:
                    lo, hi = bounds[tok0]
                    ws_all = h1_idx["w"][lo:hi]
                    ats_all = h1_idx["a"][lo:hi]
                    docs_all = h1_idx["d"][lo:hi]
                    keep = ~np.isin(docs_all, list(fd["own_docs"]))
                    if fd["own_windows"]:
                        keep &= ~np.isin(ws_all, list(fd["own_windows"]))
                    ws_all, ats_all = ws_all[keep], ats_all[keep]
                    if len(ws_all) > 4096:
                        sub_i = rng_np.choice(len(ws_all), size=4096, replace=False)
                        ws_all, ats_all = ws_all[sub_i], ats_all[sub_i]
                    if len(ws_all) > 0:
                        p_prev = windows[ws_all, ats_all - 1]
                        p_nxt = windows[ws_all, ats_all + 1]
                        for lv in pt.NEAR_LEVELS:
                            if lv == "phrase":
                                if prev is None or nxt is None:
                                    continue
                                m = (p_prev == prev) & (p_nxt == nxt)
                            elif lv == "before":
                                if prev is None:
                                    continue
                                m = p_prev == prev
                            elif lv == "after":
                                if nxt is None:
                                    continue
                                m = p_nxt == nxt
                            else:
                                m = np.ones(len(ws_all), dtype=bool)
                            hits = np.flatnonzero(m)
                            if len(hits) == 0:
                                continue
                            rng_np.shuffle(hits)
                            # Collect symmetric-length candidates first, then relax
                            near_hits: List[Tuple[float, int, int, int, int, List[int], str]] = []
                            for h_j in hits[:48]:
                                wq, aq = int(ws_all[h_j]), int(ats_all[h_j])
                                if (wq, aq) in seen_q:
                                    continue
                                qids = _window_ids(windows, wq)
                                sq, eq, _, _ = at_mod.cut_passage(qids, aq, before_f, after_f, marks)
                                sq, eq = at_mod.whole_characters(tok, qids, sq, eq, marks)
                                if abs((aq - sq) - before_f) > 1 or abs((eq - aq - 1) - after_f) > 1:
                                    continue
                                q_txt = _snippet_span_text(tok, qids[:eq], sq, eq, aq)
                                ratio = len(q_txt) / max(len(fire_txt), 1)
                                seen_q.add((wq, aq))
                                near_hits.append((abs(ratio - 1.0), wq, aq, sq, eq, qids[:eq], q_txt))
                            near_hits.sort(key=lambda x: x[0])
                            for _, wq, aq, sq, eq, qseq, q_txt in near_hits[: pt.NEAR_TRIES]:
                                ratio = len(q_txt) / max(len(fire_txt), 1)
                                cand_q.append((wq, aq, lv, sq, eq, qseq, q_txt, ratio))
                # Fallback doc_h1 anywhere
                for _ in range(12):
                    wq = int(h1_pool[int(rng_np.integers(0, len(h1_pool)))])
                    if wq in fd["own_windows"]:
                        continue
                    qids = _window_ids(windows, wq)
                    if len(qids) <= max(before_f + 4, 16) + after_f:
                        continue
                    aq = int(rng_np.integers(max(8, before_f + 1), len(qids) - max(1, after_f)))
                    if int(half_id[wq, aq]) != 1 or int(doc_id[wq, aq]) in fd["own_docs"] or int(qids[aq]) in marks or (wq, aq) in seen_q:
                        continue
                    sq, eq, _, _ = at_mod.cut_passage(qids, aq, before_f, after_f, marks)
                    sq, eq = at_mod.whole_characters(tok, qids, sq, eq, marks)
                    if abs((aq - sq) - before_f) > 1 or abs((eq - aq - 1) - after_f) > 1:
                        continue
                    seen_q.add((wq, aq))
                    q_txt = _snippet_span_text(tok, qids[:eq], sq, eq, aq)
                    ratio = len(q_txt) / max(len(fire_txt), 1)
                    cand_q.append((wq, aq, pt.ANYWHERE, sq, eq, qids[:eq], q_txt, ratio))

                pair_slots.append(
                    {
                        "f_idx": f_idx,
                        "fd": fd,
                        "fire_txt": fire_txt,
                        "cands": cand_q,
                        "quiet": None,
                        "first_is_fire": bool(rng_np.random() < 0.5),
                    }
                )

        # Round-batched live GPU quiet verification up to layer l
        t_q = time.time()
        max_rounds = max((len(ps["cands"]) for ps in pair_slots), default=0)
        for rnd in range(max_rounds):
            todo = [ps for ps in pair_slots if ps["quiet"] is None and rnd < len(ps["cands"])]
            if not todo:
                break
            seqs_r = [ps["cands"][rnd][5] for ps in todo]
            tgts_r = [(l, ps["fd"]["n"]) for ps in todo]
            acts_r = _read_multi_neurons_to_layer(lm, seqs_r, tgts_r, batch_size=16)
            for ps, act_v in zip(todo, acts_r):
                wq, aq, lv, sq, eq, _, q_txt, ratio = ps["cands"][rnd]
                fd = ps["fd"]
                sgn = 1.0 if fd["face"] == "pos" else -1.0
                rel_span = float(np.max((sgn * act_v[sq:eq]) / max(fd["extreme"], 1e-6)))
                if rel_span < pt.QUIET_BELOW:
                    ps["quiet"] = {"win": wq, "at": aq, "match": lv, "rel": round(rel_span, 3), "txt": q_txt, "len_ratio": ratio}
        n_verified = sum(1 for ps in pair_slots if ps["quiet"] is not None)
        print(f"L{l}: GPU verified {n_verified}/{len(pair_slots)} quiet pairs in {time.time() - t_q:.1f}s", flush=True)

        # 3. Single-prefill blind_near scoring: own + K=3 deranged other_pattern nulls
        o1, o2, o3 = k3_derangement_offsets(N_faces)
        offsets_k = (o1, o2, o3)
        items_b: List[Dict[str, Any]] = []
        len_ratios: List[float] = []
        for ps in pair_slots:
            if ps["quiet"] is None:
                continue
            fi = ps["f_idx"]
            tr = "a" if ps["first_is_fire"] else "b"
            a_txt, b_txt = (ps["fire_txt"], ps["quiet"]["txt"]) if tr == "a" else (ps["quiet"]["txt"], ps["fire_txt"])
            len_ratios.append(float(ps["quiet"]["len_ratio"]))
            items_b.append(
                {
                    "face_key": face_keys[fi],
                    "other_keys": [face_keys[(fi + off) % N_faces] for off in offsets_k],
                    "truth": tr,
                    "match": ps["quiet"]["match"],
                    "a_txt": a_txt,
                    "b_txt": b_txt,
                }
            )

        def _enc_prefills(pat_key_fn: Any) -> List[List[int]]:
            return [
                rt.tokenize_prompt(
                    rt.compose_prompt(_single_pair_prompt(gemma_patterns[pat_key_fn(it)], it["a_txt"], it["b_txt"]), system=SINGLE_PAIR_SYSTEM, chat=True, thinking=False, assistant_prefix=best_apref)
                )
                for it in items_b
            ]

        t_judge = time.time()
        own_sc = _score_ab_prefills(rt, lm, _enc_prefills(lambda it: it["face_key"]), ab_ids, batch_size=int(args.prefill_batch))
        other_sc_k = [
            _score_ab_prefills(rt, lm, _enc_prefills(lambda it, ki_=ki: it["other_keys"][ki_]), ab_ids, batch_size=int(args.prefill_batch))
            for ki in range(3)
        ]
        print(f"L{l}: scored {len(items_b)} blind_near pairs x4 patterns in {time.time() - t_judge:.1f}s", flush=True)

        # 4. Physical counterfactual behind synthesis on UNSEEN doc_h1 passages (+ other_pattern control)
        t_beh = time.time()
        beh_prep: List[Dict[str, Any]] = []
        for fd in faces_data:
            be = fd["behind_ev"]
            ids = _window_ids(windows, be["win"])
            at0 = int(be["at"])
            kept = ids[: at0 + 1 - pt.DROP]
            dropped = ids[at0 + 1 - pt.DROP : at0 + 1]
            kept_tail_str = "".join(t for t, _ in ao.token_units(tok, kept[-60:]))
            first_drop = tok.decode([int(dropped[0])])
            if first_drop.startswith("\n"):
                hint = "Notice the original continuation begins with a newline character."
            elif first_drop.startswith(" "):
                hint = "Notice the original continuation begins with a leading space."
            else:
                hint = "Notice the original continuation begins directly with no leading space."
            pmt = _behind_prompt(gemma_patterns[fd["key"]], kept_tail_str, hint)
            pids = rt.tokenize_prompt(rt.compose_prompt(pmt, system=BEHIND_SYSTEM, chat=True, thinking=False))
            beh_prep.append({"fd": fd, "kept": kept, "dropped": dropped, "pids": pids})

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        beh_outs = _generate_batch(rt, lm, [bp["pids"] for bp in beh_prep], batch_size=int(args.batch_size), max_new=160)
        parsed_fq = [_parse_behind_endings(txt) for txt in beh_outs]

        eval_seqs: List[List[int]] = []
        eval_tgts: List[Tuple[int, int]] = []
        eval_meta: List[Tuple[int, str, int, str]] = []
        for idx_bp, (bp, (f_strs, q_strs)) in enumerate(zip(beh_prep, parsed_fq)):
            fd = bp["fd"]
            kept = bp["kept"]
            eval_seqs.append(list(kept) + list(bp["dropped"]))
            eval_tgts.append((l, fd["n"]))
            eval_meta.append((idx_bp, "own", len(kept), "<own_held_out>"))
            # Own fire & own quiet
            for kind, lst in (("fire", f_strs), ("quiet", q_strs)):
                for es in lst:
                    e_ids = tok.encode(es, add_special_tokens=False)
                    if not e_ids or len(kept) + len(e_ids) > 1024:
                        continue
                    eval_seqs.append(list(kept) + [int(t) for t in e_ids])
                    eval_tgts.append((l, fd["n"]))
                    eval_meta.append((idx_bp, kind, len(kept), es))
            # Other face's FIRE endings appended behind this face's unseen passage (control)
            oth_idx = (idx_bp + 1) % len(beh_prep)
            oth_f_strs = parsed_fq[oth_idx][0]
            for es in oth_f_strs:
                e_ids = tok.encode(es, add_special_tokens=False)
                if not e_ids or len(kept) + len(e_ids) > 1024:
                    continue
                eval_seqs.append(list(kept) + [int(t) for t in e_ids])
                eval_tgts.append((l, fd["n"]))
                eval_meta.append((idx_bp, "other_fire", len(kept), es))

        acts_beh = _read_multi_neurons_to_layer(lm, eval_seqs, eval_tgts, batch_size=16) if eval_seqs else []
        per_bp: Dict[int, Dict[str, Any]] = {i: {"own_rel": None, "fire": [], "quiet": [], "other_fire": []} for i in range(len(beh_prep))}
        for (idx_bp, kind, k_len, es), act_v in zip(eval_meta, acts_beh):
            fd = beh_prep[idx_bp]["fd"]
            sgn = 1.0 if fd["face"] == "pos" else -1.0
            rel_end = float(np.max((sgn * act_v[k_len:]) / max(fd["extreme"], 1e-6)))
            if kind == "own":
                per_bp[idx_bp]["own_rel"] = round(rel_end, 3)
            else:
                per_bp[idx_bp][kind].append({"text": es[:160], "rel": round(rel_end, 3)})

        behind_records: List[Dict[str, Any]] = []
        behind_details: Dict[str, Any] = {}
        for idx_bp, bp in enumerate(beh_prep):
            fd = bp["fd"]
            er = per_bp[idx_bp]
            wr = pt.written_result(er["fire"], er["quiet"])
            own_ok = (er["own_rel"] or 0.0) >= pt.FIRES_AT
            of_fired = sum(1 for r in er["other_fire"] if float(r["rel"]) >= pt.FIRES_AT)
            rec = {
                "key": fd["key"],
                "layer": int(l),
                "own_ending_rel": er["own_rel"],
                "own_holds_gt_0.5": own_ok,
                "result": wr,
                "other_fire_total": len(er["other_fire"]),
                "other_fire_fired": of_fired,
                "pattern": gemma_patterns[fd["key"]],
                "synthetic_fire": er["fire"],
                "synthetic_quiet": er["quiet"],
            }
            behind_records.append(rec)
            behind_details[fd["key"]] = rec

        print(f"L{l}: behind counterfactuals finished in {time.time() - t_beh:.1f}s", flush=True)

        v_layer = evaluate_p3_layer(
            layer=l,
            face_keys=face_keys,
            items_b=items_b,
            own_scores=own_sc,
            other_scores_k=other_sc_k,
            len_ratios=len_ratios,
            behind_records=behind_records,
        )
        v_clean = {k: v for k, v in v_layer.items() if not k.startswith("_")}
        v_clean["prereg"] = prereg_info
        v_clean["elapsed_s"] = round(time.time() - tl0, 2)
        v_clean["git_dirty"] = False
        v_clean["code_sha256"] = gp.build_provenance(REPO_ROOT, [__file__, REPO_ROOT / "scripts/in_use_common.py", prereg_dir / "SHA256SUMS"], started_at=gp.utc_iso(tl0), finished_at=gp.utc_iso(time.time()))["code_sha256"]

        # Full patterns/behind details go to isolated out_dir; compact verdict (without raw corpus text!) goes to results_dir
        (out_dir / f"L{l}_details.json").write_text(
            json.dumps({"blind_near_per_face": v_layer["_per_face_bn"], "behind_details": behind_details}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        v_clean["patterns_sample_20"] = {k: gemma_patterns[k] for k in face_keys[:20]}
        ckpt_path.write_text(json.dumps(v_clean, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (out_dir / f"L{l}_verdict.json").write_text(json.dumps(v_clean, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        layer_verdicts[str(l)] = v_clean

        print(
            f"L{l} P3 VERDICT ({v_clean['elapsed_s']}s): pass={v_clean['pass']} gates={v_clean['gates']} "
            f"bn={v_clean['blind_near']['own_share']} vs {v_clean['blind_near']['mean_other_share']} "
            f"p05={v_clean['blind_near']['frac_p05_own_ge8']} behind_clean={v_clean['behind_unseen']['clean_2of3_and_0quiet_rate']}",
            flush=True,
        )

    t1 = time.time()
    prov = gp.build_provenance(
        REPO_ROOT,
        [__file__, REPO_ROOT / "scripts/in_use_common.py", REPO_ROOT / "backend/pattern_tests.py", prereg_dir / "SHA256SUMS"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(t1),
        model_id=rt.model_id,
        weights=weights_card(rt),
        neuron_ids={str(l): [f"n{n}{'p' if f=='pos' else 'm'}" for n, f in _load_frozen_faces(prereg_dir / f"faces_L{l}.txt")] for l in layers},
    )
    summary = {
        **prov,
        "prereg": prereg_info,
        "elapsed_s": round(t1 - t0, 2),
        "layers": {
            lk: {
                "layer": lv["layer"],
                "n_faces": lv["n_faces"],
                "pass": lv["pass"],
                "gates": lv["gates"],
                "elapsed_s": lv["elapsed_s"],
                "blind_near": lv["blind_near"],
                "behind_unseen": lv["behind_unseen"],
            }
            for lk, lv in layer_verdicts.items()
        },
    }
    (results_dir / "verdict.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "verdict.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {results_dir / 'verdict.json'} in {t1 - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
