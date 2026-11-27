#!/usr/bin/env python3
"""Probe 2 (CPU-only): passages.d <eos> document-half coverage audit & Context-PMI near-miss baseline.

1. Across L=[2,8,14,20,25,29,55,58,59], measures how many non-dead neurons / faces have >=4
   pure-half0 & >=4 pure-half1 events (`pure_4_4`), >=6 & >=6 (`pure_6_6`), and >=4 & >=4
   single-half-at-peak (`doc_4_4`) events.
2. Across L=[2,8,14,20,25,29] (N=256 random pure_4_4 faces/layer), tests whether top-8 context
   tokens in [-32..+4] excluding {-1, 0, +1} ranked by PMI on half-0 fit docs discriminate
   half-1 firing passages from corpus near misses better than a shuffled same-layer face.
   Note: CPU near misses are unverified for live activation quietness (< 0.25x), serving strictly
   as a zero-GPU upper-bound screen; Probe 3 verifies quietness live on GPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402
from backend import pattern_tests as pt  # noqa: E402
from scripts.in_use_common import (  # noqa: E402
    collect_face_events,
    layer_face_counts_vectorized,
    load_passages_and_doc_halves,
)

AUDIT_LAYERS: Tuple[int, ...] = (2, 8, 14, 20, 25, 29, 55, 58, 59)
PMI_LAYERS: Tuple[int, ...] = (2, 8, 14, 20, 25, 29)


def _special_token_ids(bos_id: int, eos_id: int) -> Set[int]:
    """Load exact tokenizer special IDs locally when cached, falling back to Gemma-4 base set."""
    bad = {0, 1, 2, 3, 4, int(bos_id), int(eos_id), 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 106}
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        snap = snapshot_download("google/gemma-4-31B-it-qat-w4a16-ct", local_files_only=True)
        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        bad.update(int(x) for x in tok.all_special_ids)
    except Exception:
        pass
    return bad


def _build_pure_h1_token_index(
    windows: np.ndarray,
    pure_h1_mask: np.ndarray,
    special_ids: Set[int],
    min_at: int = 8,
    min_after: int = 2,
) -> Dict[str, Any]:
    """Build sorted 1D numpy arrays of valid positions in pure-half1 windows for O(log N) near-miss lookup."""
    h1_wins = np.flatnonzero(pure_h1_mask)
    seq_len = int(windows.shape[1])
    sub = windows[h1_wins]
    valid_pos = (sub >= 0) & (~np.isin(sub, list(special_ids)))
    row_len = np.sum(sub >= 0, axis=1)
    pos_grid = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], sub.shape)
    valid_pos &= (pos_grid >= int(min_at)) & (pos_grid < (row_len[:, None] - int(min_after)))

    wi_rel, ai_arr = np.nonzero(valid_pos)
    w_arr = h1_wins[wi_rel].astype(np.int32)
    a_arr = ai_arr.astype(np.int32)
    t_arr = windows[w_arr, a_arr].astype(np.int32)

    order = np.argsort(t_arr, kind="mergesort")
    t_s = t_arr[order]
    w_s = w_arr[order]
    a_s = a_arr[order]
    uniq_t, starts, counts = np.unique(t_s, return_index=True, return_counts=True)
    bounds = {int(t): (int(st), int(st + cnt)) for t, st, cnt in zip(uniq_t, starts, counts)}
    return {"h1_wins": h1_wins, "row_len": row_len, "t": t_s, "w": w_s, "a": a_s, "bounds": bounds}


def _find_unverified_near_miss(
    windows: np.ndarray,
    h1_idx: Dict[str, Any],
    own_windows: Set[int],
    w_f: int,
    at_f: int,
    rng: np.random.Generator,
    max_scan: int = 2048,
) -> Tuple[int, int, str]:
    """Find a deterministic pure-half1 corpus near miss for (w_f, at_f) without GPU activation check."""
    tok0 = int(windows[w_f, at_f])
    prev = int(windows[w_f, at_f - 1]) if at_f >= 1 and int(windows[w_f, at_f - 1]) >= 0 else None
    nxt = int(windows[w_f, at_f + 1]) if at_f + 1 < windows.shape[1] and int(windows[w_f, at_f + 1]) >= 0 else None

    bounds = h1_idx["bounds"]
    if tok0 in bounds:
        lo, hi = bounds[tok0]
        ws = h1_idx["w"][lo:hi]
        ats = h1_idx["a"][lo:hi]
        if own_windows:
            keep = ~np.isin(ws, list(own_windows))
            ws, ats = ws[keep], ats[keep]
        if len(ws) > max_scan:
            pick = rng.choice(len(ws), size=max_scan, replace=False)
            ws, ats = ws[pick], ats[pick]
        if len(ws) > 0:
            p_prev = windows[ws, ats - 1]
            p_nxt = windows[ws, ats + 1]
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
                    m = np.ones(len(ws), dtype=bool)
                hits = np.flatnonzero(m)
                if len(hits) > 0:
                    j = int(hits[int(rng.integers(0, len(hits)))])
                    return int(ws[j]), int(ats[j]), lv

    pool = [int(w) for w in h1_idx["h1_wins"].tolist() if int(w) not in own_windows]
    if not pool:
        pool = [int(w) for w in h1_idx["h1_wins"].tolist()]
    for _ in range(64):
        w = pool[int(rng.integers(0, len(pool)))]
        valid_len = int(np.sum(windows[w] >= 0))
        if valid_len > 16:
            at = int(rng.integers(8, valid_len - 2))
            if int(windows[w, at]) >= 0:
                return w, at, pt.ANYWHERE
    w = pool[0]
    return w, 8, pt.ANYWHERE


def _context_token_set(
    windows: np.ndarray,
    doc_id: np.ndarray,
    w: int,
    at: int,
    special_ids: Set[int],
    back: int = 32,
    fwd: int = 4,
) -> Set[int]:
    """Collect unique tokens in [-back..+fwd] excluding {-1, 0, +1}, bounded to the same doc_id."""
    seq_len = int(windows.shape[1])
    d_target = int(doc_id[w, at])
    lo = max(1, int(at) - int(back))
    hi = min(seq_len, int(at) + int(fwd) + 1)
    out: Set[int] = set()
    for p in range(lo, hi):
        if abs(p - int(at)) <= 1:
            continue
        if int(doc_id[w, p]) != d_target:
            continue
        t = int(windows[w, p])
        if t < 0 or t in special_ids:
            continue
        out.add(t)
    return out


def _score_pmi(ctx_set: Set[int], pmi_map: Dict[int, float]) -> float:
    return float(sum(pmi_map.get(t, 0.0) for t in ctx_set))


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe 2 CPU audit: pure doc halves + Context-PMI near-miss baseline")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-faces", type=int, default=256)
    ap.add_argument("--out-dir", default="experiments/results/in_use_kill_pilots")
    args = ap.parse_args()

    t0 = time.time()
    t0_iso = gp.utc_iso(t0)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ai.atlas_dir()
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_passages_and_doc_halves(data_dir)
    P = loaded["P"]
    act_scale = loaded["act_scale"]
    windows = loaded["windows"]
    unigram = loaded["unigram"]
    doc_id = loaded["doc_id"]
    half_id = loaded["half_id"]
    pure_h0_mask = loaded["pure_h0_mask"]
    pure_h1_mask = loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]

    special_ids = _special_token_ids(bos_id, eos_id)
    unigram_f = np.asarray(unigram, dtype=np.float64)
    tot_tok = max(float(unigram_f.sum()), 1.0)
    bg_freq = unigram_f / tot_tok
    V = int(len(bg_freq))

    print(
        f"passages.d: W={windows.shape[0]} n_docs={loaded['n_docs']} pure_h0={int(pure_h0_mask.sum())} pure_h1={int(pure_h1_mask.sum())}",
        flush=True,
    )

    coverage: List[Dict[str, Any]] = []
    counts_by_layer: Dict[int, Dict[str, np.ndarray]] = {}
    for l in AUDIT_LAYERS:
        cnt = layer_face_counts_vectorized(
            P, windows, act_scale, half_id, pure_h0_mask, pure_h1_mask, l, eos_id=eos_id, bos_id=bos_id
        )
        counts_by_layer[l] = cnt
        nd = max(int(cnt["nondead_mask"].sum()), 1)
        row = {
            "layer": int(l),
            "nondead_neurons": int(cnt["nondead_mask"].sum()),
            "neu_pure44": int(cnt["neu_pure44"].sum()),
            "frac_neu_pure44": round(float(cnt["neu_pure44"].sum() / nd), 4),
            "face_pure44": int(cnt["face_pure44_count"]),
            "neu_pure66": int(cnt["neu_pure66"].sum()),
            "frac_neu_pure66": round(float(cnt["neu_pure66"].sum() / nd), 4),
            "face_pure66": int(cnt["pos_pure66"].sum() + cnt["neg_pure66"].sum()),
            "neu_doc44": int(cnt["neu_doc44"].sum()),
            "frac_neu_doc44": round(float(cnt["neu_doc44"].sum() / nd), 4),
            "face_doc44": int(cnt["pos_doc44"].sum() + cnt["neg_doc44"].sum()),
        }
        coverage.append(row)
        print(
            f"  audit L{l:2d}: nondead={row['nondead_neurons']} pure44={row['neu_pure44']} ({row['frac_neu_pure44']:.1%}) "
            f"pure66={row['neu_pure66']} ({row['frac_neu_pure66']:.1%}) doc44={row['neu_doc44']} ({row['frac_neu_doc44']:.1%})",
            flush=True,
        )

    h1_idx = _build_pure_h1_token_index(windows, pure_h1_mask, special_ids)
    pmi_layers: List[Dict[str, Any]] = []
    sampled_faces_meta: Dict[str, List[str]] = {}

    for l in PMI_LAYERS:
        cnt = counts_by_layer[l]
        cands: List[Tuple[int, str]] = []
        N_ff = int(cnt["pos_pure44"].shape[0])
        for n in range(N_ff):
            if bool(cnt["pos_pure44"][n]):
                cands.append((n, "pos"))
            if bool(cnt["neg_pure44"][n]):
                cands.append((n, "neg"))

        rng = np.random.default_rng(2026112502 + l)
        pick_n = min(int(args.n_faces), len(cands))
        pick_idx = sorted(int(i) for i in rng.choice(len(cands), size=pick_n, replace=False))
        faces = [cands[i] for i in pick_idx]
        sampled_faces_meta[str(l)] = [f"n{n}{'p' if fc == 'pos' else 'm'}" for n, fc in faces]

        face_rows: List[Dict[str, Any]] = []
        for n, fc in faces:
            ev = collect_face_events(
                P, windows, act_scale, pure_h0_mask, pure_h1_mask, doc_id, half_id, l, n, fc, eos_id=eos_id, bos_id=bos_id
            )
            fit_evs = ev["pure_h0"][:16]
            test_evs = ev["pure_h1"][:8]
            n_fit = len(fit_evs)
            docs_tok: Dict[int, Set[int]] = collections.defaultdict(set)
            cnt_tok: Dict[int, int] = collections.Counter()
            for e in fit_evs:
                cset = _context_token_set(windows, doc_id, e["win"], e["at"], special_ids)
                for t in cset:
                    docs_tok[t].add(int(e["doc_id"]))
                    cnt_tok[t] += 1
            scored_toks: List[Tuple[float, int]] = []
            for t, c in cnt_tok.items():
                if len(docs_tok[t]) < 2:
                    continue
                rate = c / max(n_fit, 1)
                bg = max(float(bg_freq[t]) if t < V else 1e-6, 1e-6)
                scored_toks.append((rate * math.log1p(rate / bg), t))
            scored_toks.sort(reverse=True)
            pmi_map = {t: float(w_sc) for w_sc, t in scored_toks[:8]}

            pairs = []
            for e in test_evs:
                w_q, at_q, match_lv = _find_unverified_near_miss(windows, h1_idx, ev["own_windows"], e["win"], e["at"], rng)
                f_ctx = _context_token_set(windows, doc_id, e["win"], e["at"], special_ids)
                q_ctx = _context_token_set(windows, doc_id, w_q, at_q, special_ids)
                pairs.append((f_ctx, q_ctx, match_lv))

            face_rows.append({"pmi": pmi_map, "pairs": pairs, "n_pmi": len(pmi_map)})

        # Score own vs cyclic shifted other_face within same layer
        by_match: Dict[str, Dict[str, float]] = {}
        own_strict = 0
        own_ties = 0
        other_strict = 0
        other_ties = 0
        tot_pairs = 0
        face_accs: List[float] = []
        p_lt_001 = 0
        p_lt_005 = 0
        faces_with_pmi = 0

        M = len(face_rows)
        for i, fr in enumerate(face_rows):
            own_map = fr["pmi"]
            other_map = face_rows[(i + 1) % M]["pmi"]
            if fr["n_pmi"] > 0:
                faces_with_pmi += 1
            f_strict = 0
            f_tie = 0
            f_pairs = len(fr["pairs"])
            for f_ctx, q_ctx, lv in fr["pairs"]:
                sf_own = _score_pmi(f_ctx, own_map)
                sq_own = _score_pmi(q_ctx, own_map)
                sf_oth = _score_pmi(f_ctx, other_map)
                sq_oth = _score_pmi(q_ctx, other_map)

                tot_pairs += 1
                row_m = by_match.setdefault(lv, {"pairs": 0, "own_strict": 0, "own_tie": 0, "other_strict": 0, "other_tie": 0})
                row_m["pairs"] += 1

                if sf_own > sq_own:
                    own_strict += 1
                    f_strict += 1
                    row_m["own_strict"] += 1
                elif sf_own == sq_own:
                    own_ties += 1
                    f_tie += 1
                    row_m["own_tie"] += 1

                if sf_oth > sq_oth:
                    other_strict += 1
                    row_m["other_strict"] += 1
                elif sf_oth == sq_oth:
                    other_ties += 1
                    row_m["other_tie"] += 1

            face_accs.append((f_strict + 0.5 * f_tie) / max(f_pairs, 1))
            pval = pt.binom_tail(f_strict, f_pairs)
            if pval < 0.01:
                p_lt_001 += 1
            if pval < 0.05:
                p_lt_005 += 1

        by_match_out = {}
        for lv in (*pt.NEAR_LEVELS, pt.ANYWHERE):
            if lv not in by_match:
                continue
            r = by_match[lv]
            npairs = max(int(r["pairs"]), 1)
            by_match_out[lv] = {
                "pairs": int(r["pairs"]),
                "own_half_tie_share": round((r["own_strict"] + 0.5 * r["own_tie"]) / npairs, 4),
                "other_half_tie_share": round((r["other_strict"] + 0.5 * r["other_tie"]) / npairs, 4),
                "tie_rate": round(r["own_tie"] / npairs, 4),
            }

        summary_l = {
            "layer": int(l),
            "faces_tested": M,
            "faces_with_nonempty_pmi": faces_with_pmi,
            "pairs": tot_pairs,
            "control_status": "unverified_quiet_corpus_near_miss",
            "own_strict_share": round(own_strict / max(tot_pairs, 1), 4),
            "own_half_tie_share": round((own_strict + 0.5 * own_ties) / max(tot_pairs, 1), 4),
            "other_strict_share": round(other_strict / max(tot_pairs, 1), 4),
            "other_half_tie_share": round((other_strict + 0.5 * other_ties) / max(tot_pairs, 1), 4),
            "own_tie_rate": round(own_ties / max(tot_pairs, 1), 4),
            "other_tie_rate": round(other_ties / max(tot_pairs, 1), 4),
            "median_face_half_tie_acc": round(float(np.median(face_accs)) if face_accs else 0.0, 4),
            "faces_strict_p_lt_001": p_lt_001,
            "frac_faces_p_lt_001": round(p_lt_001 / max(M, 1), 4),
            "faces_strict_p_lt_005": p_lt_005,
            "by_match": by_match_out,
        }
        pmi_layers.append(summary_l)
        print(
            f"  PMI L{l:2d}: own_half_tie={summary_l['own_half_tie_share']:.3f} (strict={summary_l['own_strict_share']:.3f}, "
            f"tie={summary_l['own_tie_rate']:.3f}) vs other={summary_l['other_half_tie_share']:.3f}; "
            f"p<0.01={p_lt_001}/{M} ({summary_l['frac_faces_p_lt_001']:.1%})",
            flush=True,
        )

    t1 = time.time()
    prov = gp.build_provenance(
        REPO_ROOT,
        [__file__, REPO_ROOT / "scripts/in_use_common.py", REPO_ROOT / "backend/input_census.py"],
        started_at=t0_iso,
        finished_at=gp.utc_iso(t1),
        model_id="cpu-passages-audit",
        neuron_ids=sampled_faces_meta,
    )
    out = {
        **prov,
        "elapsed_s": round(t1 - t0, 2),
        "passages_summary": {
            "windows": int(windows.shape[0]),
            "seq": int(windows.shape[1]),
            "n_docs": int(loaded["n_docs"]),
            "pure_h0_windows": int(pure_h0_mask.sum()),
            "pure_h1_windows": int(pure_h1_mask.sum()),
        },
        "coverage_audit": coverage,
        "context_pmi_near_miss": pmi_layers,
    }
    out_path = out_dir / "probe2_cpu_context.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} in {t1 - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
