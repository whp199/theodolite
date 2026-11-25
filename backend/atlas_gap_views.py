"""Read-only readers for Stage 1 & Stage 2 supplementary census products.

Covers three distinct scientific regimes under ``census_dir()`` without touching or conflating
existing ``census/runs/`` rollouts or ``backend/census.py`` semantics, and without importing or
modifying pre-registered GPU modules (``backend/forced_odd.py``):

- ``forced_odd`` (L55-L59): causal forced next-piece shift (FDR <= 5%, dual-null)
- ``input_census`` (L0-L29): input-side firing census (when it fires, FDR <= 5%; zero raw text)
- ``fdr_rescue`` (L30-L54): standalone empirical-FDR decoding rescue (T_dec, no_thought, FDP <= 5%)
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REGIME_LABELS: Dict[str, str] = {
    "forced_odd": "forced next-piece shift (FDR ≤ 5%, dual-null)",
    "input_census": "input-side firing census (when it fires, FDR ≤ 5%)",
    "fdr_rescue": "empirical-FDR decoding rescue (T_dec, no_thought, FDP ≤ 5%)",
}

OFFSET_LABELS: Dict[int, str] = {-1: "prev", 0: "peak", 1: "next"}

_LOCK = threading.RLock()
_CACHE: Dict[Tuple[Any, ...], Any] = {}

_FO_RE = re.compile(r"L(\d+)_forced_odd\.npz$")
_IC_RE = re.compile(r"L(\d+)_input_census\.npz$")
_FDR_RE = re.compile(r"L(\d+)_whole_named\.json$")


def _stamp(path: Path) -> Tuple[str, int, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size), int(getattr(st, "st_ino", 0)))


def _cached_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    key = ("json",) + _stamp(path)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    data = json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        _CACHE[key] = data
    return data


def _tok_str(vocab: Sequence[str], tid: int) -> str:
    if 0 <= tid < len(vocab):
        return vocab[tid]
    return f"<{tid}>"


def _note_name(notes: Dict[str, Any], layer: int, n: int) -> str:
    return str((notes.get(f"{int(layer)}:{int(n)}") or {}).get("name") or "")


def _round_opt(val: Any, ndigits: int = 3) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val), ndigits)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------------- forced_odd (L55-L59)


def _fo_path(root: Path, layer: int) -> Path:
    return root / "forced_odd" / f"L{int(layer)}_forced_odd.npz"


def _fo_meta_path(root: Path, layer: int) -> Path:
    return root / "forced_odd" / f"L{int(layer)}_forced_odd.meta.json"


def _load_forced_odd(root: Path, layer: int) -> Optional[Dict[str, Any]]:
    path = _fo_path(root, layer)
    if not path.is_file():
        return None
    meta_path = _fo_meta_path(root, layer)
    m_stamp = _stamp(meta_path) if meta_path.is_file() else None
    key = ("fo",) + _stamp(path) + (m_stamp,)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = _cached_json(meta_path) if meta_path.is_file() else None
    t_col = np.asarray(arrays["T_col"], dtype=np.float32)
    t_iso = np.asarray(arrays["T_iso"], dtype=np.float32)
    t_min = np.minimum(t_col, t_iso)
    named = np.asarray(arrays["named_dual05"], dtype=bool)
    named_idx = np.flatnonzero(named)
    # Sort descending by T_min, tie-breaking by T_col desc, then neuron id asc.
    order = sorted(
        (int(i) for i in named_idx),
        key=lambda i: (-float(t_min[i]), -float(t_col[i]), i),
    )
    rank_of = {ni: r for r, ni in enumerate(order, 1)}
    entry = {
        "layer": int(layer),
        "arrays": arrays,
        "meta": meta,
        "T_min": t_min,
        "named_order": order,
        "rank_of": rank_of,
    }
    with _LOCK:
        _CACHE[key] = entry
    return entry


def _extract_forced_odd_pieces(arrays: Dict[str, np.ndarray], n: int, vocab: Sequence[str], k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Match backend/forced_odd.py:extract_replicated_pieces exact A->B and B->A replication logic."""
    rep_pos: List[Tuple[str, float, int]] = []
    rep_neg: List[Tuple[str, float, int]] = []
    seen_p = set()
    seen_n = set()
    top_a_p = arrays["topA_p"]
    width = int(top_a_p.shape[1])
    for j in range(width):
        pid = int(top_a_p[n, j])
        sc_p = float(arrays["posA_onB"][n, j])
        if sc_p > 0.0 and pid not in seen_p:
            seen_p.add(pid)
            rep_pos.append((_tok_str(vocab, pid), round(sc_p, 2), pid))

        pid2 = int(arrays["topB_p"][n, j])
        sc_p2 = float(arrays["posB_onA"][n, j])
        if sc_p2 > 0.0 and pid2 not in seen_p:
            seen_p.add(pid2)
            rep_pos.append((_tok_str(vocab, pid2), round(sc_p2, 2), pid2))

        nid = int(arrays["topA_n"][n, j])
        sc_n = float(arrays["negA_onB"][n, j])
        if sc_n < 0.0 and nid not in seen_n:
            seen_n.add(nid)
            rep_neg.append((_tok_str(vocab, nid), round(-sc_n, 2), nid))

        nid2 = int(arrays["topB_n"][n, j])
        sc_n2 = float(arrays["negB_onA"][n, j])
        if sc_n2 < 0.0 and nid2 not in seen_n:
            seen_n.add(nid2)
            rep_neg.append((_tok_str(vocab, nid2), round(-sc_n2, 2), nid2))

    rep_pos.sort(key=lambda x: -x[1])
    rep_neg.sort(key=lambda x: -x[1])
    return {
        "pos": [{"tok": t, "score": s, "id": tid} for t, s, tid in rep_pos[:k]],
        "neg": [{"tok": t, "score": s, "id": tid} for t, s, tid in rep_neg[:k]],
    }


def _forced_odd_meta_card(fo: Dict[str, Any]) -> Dict[str, Any]:
    arrays = fo["arrays"]
    meta = fo["meta"]
    d_ff = int(arrays["named_dual05"].shape[0])
    dual05_count = int(np.asarray(arrays["named_dual05"], dtype=bool).sum())
    dual10_count = int(np.asarray(arrays["named_dual10"], dtype=bool).sum()) if "named_dual10" in arrays else None
    is_direct = np.asarray(arrays["is_direct"], dtype=bool) if "is_direct" in arrays else None
    direct_among = int(np.sum(is_direct & np.asarray(arrays["named_dual05"], dtype=bool))) if is_direct is not None else 0

    if meta:
        vs_col = meta.get("vs_columns") or {}
        vs_iso = meta.get("vs_isotropic") or {}
        ww = meta.get("weight_word_diag") or {}
        return {
            "layer": fo["layer"],
            "d_ff": int(meta.get("d_ff", d_ff)),
            "readout_mode": meta.get("readout_mode"),
            "dose_ceil": meta.get("dose_ceil"),
            "dose_stat": meta.get("dose_stat"),
            "target_kl": meta.get("target_kl"),
            "n_cols": meta.get("n_cols"),
            "n_iso": meta.get("n_iso"),
            "dual05_count": int(meta.get("dual05_count", dual05_count)),
            "dual05_pct": meta.get("dual05_pct", round(100.0 * dual05_count / max(d_ff, 1), 2)),
            "dual10_count": meta.get("dual10_count", dual10_count),
            "direct_among_dual05": ww.get("direct_among_dual05", direct_among),
            "vs_columns": {
                "auc_T": vs_col.get("auc_T"),
                "fdp05": vs_col.get("fdp05"),
                "null_splits_columns": meta.get("null_splits_columns"),
            },
            "vs_isotropic": {
                "auc_T": vs_iso.get("auc_T"),
                "fdp05": vs_iso.get("fdp05"),
                "null_splits_isotropic": meta.get("null_splits_isotropic"),
            },
            "half_c": meta.get("half_c"),
        }

    return {
        "layer": fo["layer"],
        "d_ff": d_ff,
        "readout_mode": None,
        "dose_ceil": None,
        "dose_stat": None,
        "target_kl": None,
        "n_cols": int(arrays["T_ctl_col"].shape[0]) if "T_ctl_col" in arrays else None,
        "n_iso": int(arrays["T_ctl_iso"].shape[0]) if "T_ctl_iso" in arrays else None,
        "dual05_count": dual05_count,
        "dual05_pct": round(100.0 * dual05_count / max(d_ff, 1), 2),
        "dual10_count": dual10_count,
        "direct_among_dual05": direct_among,
        "vs_columns": None,
        "vs_isotropic": None,
        "half_c": None,
    }


def _forced_odd_neuron(root: Path, layer: int, n: int, vocab: Sequence[str]) -> Optional[Dict[str, Any]]:
    fo = _load_forced_odd(root, layer)
    if fo is None:
        return None
    arrays = fo["arrays"]
    if not (0 <= n < int(arrays["T_col"].shape[0])):
        return None
    named05 = bool(arrays["named_dual05"][n])
    named10 = bool(arrays["named_dual10"][n]) if "named_dual10" in arrays else named05
    named_col05 = bool(arrays["named_col05"][n]) if "named_col05" in arrays else named05
    named_iso05 = bool(arrays["named_iso05"][n]) if "named_iso05" in arrays else named05

    out: Dict[str, Any] = {
        "regime": "forced_odd",
        "label": REGIME_LABELS["forced_odd"],
        "named_dual05": named05,
        "named_dual10": named10,
        "named_col05": named_col05,
        "named_iso05": named_iso05,
        "T_col": round(float(arrays["T_col"][n]), 3),
        "T_iso": round(float(arrays["T_iso"][n]), 3),
        "T_min": round(float(fo["T_min"][n]), 3),
        "r_ab_res_col": _round_opt(arrays["r_ab_res_col"][n], 3) if "r_ab_res_col" in arrays else None,
        "r_ba_res_col": _round_opt(arrays["r_ba_res_col"][n], 3) if "r_ba_res_col" in arrays else None,
        "kl_mean": _round_opt(arrays["kl_mean"][n], 4) if "kl_mean" in arrays else None,
        "alpha_abs": _round_opt(arrays["alpha_abs"][n], 4) if "alpha_abs" in arrays else None,
        "meta": _forced_odd_meta_card(fo),
    }
    if named05:
        is_direct = bool(arrays["is_direct"][n]) if "is_direct" in arrays else False
        q_direct = _round_opt(arrays["q_direct"][n], 4) if "q_direct" in arrays else None
        pieces = _extract_forced_odd_pieces(arrays, n, vocab, k=5)
        out.update({
            "rank": fo["rank_of"].get(n),
            "n_named": len(fo["named_order"]),
            "tag": "direct_unembed" if is_direct else "circuit_mediated",
            "q_direct": q_direct,
            "pos": pieces["pos"],
            "neg": pieces["neg"],
        })
    else:
        out.update({
            "rank": None,
            "n_named": len(fo["named_order"]),
            "tag": None,
            "q_direct": None,
            "pos": [],
            "neg": [],
            "why_no_words": "Did not clear dual-null FDR ≤ 5% against both column and isotropic controls; no shift pieces shown.",
        })
    return out


# ------------------------------------------------------------------------------------------ input_census (L0-L29)


def _ic_path(root: Path, layer: int) -> Path:
    return root / "input_census" / f"L{int(layer):02d}_input_census.npz"


def _load_input_census(root: Path, layer: int) -> Optional[Dict[str, Any]]:
    path = _ic_path(root, layer)
    if not path.is_file():
        return None
    key = ("ic",) + _stamp(path)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    named05 = np.asarray(arrays["named_fdp05"], dtype=bool)
    score = np.asarray(arrays["score"], dtype=np.float32)
    # Rank passing faces across both faces sorted by descending score, then n, then face index.
    coords = np.argwhere(named05)
    order = sorted(
        ((int(si), int(ni)) for si, ni in coords),
        key=lambda pair: (-float(score[pair[0], pair[1]]), pair[1], pair[0]),
    )
    rank_of = {pair: r for r, pair in enumerate(order, 1)}
    entry = {
        "layer": int(layer),
        "arrays": arrays,
        "named_order": order,
        "rank_of": rank_of,
    }
    with _LOCK:
        _CACHE[key] = entry
    return entry


def _ablation_by_layer(root: Path) -> Tuple[Optional[Dict[int, Dict[str, Any]]], List[int]]:
    data = _cached_json(root / "input_census" / "ablation_check.json")
    if not isinstance(data, dict):
        return None, []
    out: Dict[int, Dict[str, Any]] = {}
    for row in data.get("layers") or []:
        if not isinstance(row, dict) or "layer" not in row:
            continue
        layer = int(row["layer"])
        pk = row.get("kl_pk") or {}
        sp = row.get("kl_span") or {}
        out[layer] = {
            "layer": layer,
            "neurons": row.get("neurons"),
            "passages": row.get("passages"),
            "quiet_thresh": data.get("quiet_thresh"),
            "quiet_max_norm_act_max": row.get("quiet_max_norm_act_max"),
            "all_quiet_under_thresh": row.get("all_quiet_under_thresh"),
            "kl_pk": {
                "auc_active_vs_quiet": pk.get("auc_active_vs_quiet"),
                "wilcoxon_p": pk.get("wilcoxon_p"),
                "frac_neurons_max4_gt_0.005": pk.get("frac_neurons_max4_gt_0.005"),
            },
            "kl_span": {
                "auc_active_vs_quiet": sp.get("auc_active_vs_quiet"),
                "wilcoxon_p": sp.get("wilcoxon_p"),
                "frac_neurons_max4_gt_0.005": sp.get("frac_neurons_max4_gt_0.005"),
            },
        }
    return out, sorted(out)


def _ic_summary_by_layer(root: Path) -> Dict[int, Dict[str, Any]]:
    data = _cached_json(root / "input_census" / "summary.json")
    out: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for row in data.get("layers") or []:
            if isinstance(row, dict) and "layer" in row:
                out[int(row["layer"])] = row
    return out


def _ic_layer_meta(root: Path, ic: Dict[str, Any], ab_map: Optional[Dict[int, Dict[str, Any]]], ab_layers: List[int]) -> Dict[str, Any]:
    layer = ic["layer"]
    arrays = ic["arrays"]
    sm_map = _ic_summary_by_layer(root)
    row = sm_map.get(layer)
    ab_row = ab_map.get(layer) if ab_map else None
    n_ff = int(arrays["named_fdp05"].shape[1])
    total_faces = 2 * n_ff
    elig = int(np.asarray(arrays["eligible"], dtype=bool).sum())
    named05 = int(np.asarray(arrays["named_fdp05"], dtype=bool).sum())
    named10 = int(np.asarray(arrays["named_fdp10"], dtype=bool).sum()) if "named_fdp10" in arrays else named05
    bg_cnt = int(np.asarray(arrays["bigram_rep"], dtype=bool).sum()) if "bigram_rep" in arrays else 0

    if row is not None:
        return {
            "layer": layer,
            "total_faces": row.get("total_faces", total_faces),
            "eligible_faces": row.get("eligible_faces", elig),
            "named_fdp05_count": row.get("named_fdp05_count", named05),
            "named_fdp05_pct_elig": row.get("named_fdp05_pct_elig"),
            "named_fdp05_pct_total": row.get("named_fdp05_pct_total"),
            "named_fdp10_count": row.get("named_fdp10_count", named10),
            "threshold_05": row.get("threshold_05"),
            "bigram_rep_pct_elig": row.get("bigram_rep_pct_elig"),
            "null_split_false_mean": row.get("null_split_false_mean"),
            "null_split_false_max": row.get("null_split_false_max"),
            "smp_named_hits": row.get("smp_named_hits"),
            "smp_named_tot": row.get("smp_named_tot"),
            "smp_named_rate": row.get("smp_named_rate"),
            "smp_shuf_rate": row.get("smp_shuf_rate"),
            "ablation_verified": ab_row is not None,
            "ablation_check": ab_row,
            "ablation_checked_layers": ab_layers,
        }

    return {
        "layer": layer,
        "total_faces": total_faces,
        "eligible_faces": elig,
        "named_fdp05_count": named05,
        "named_fdp05_pct_elig": round(100.0 * named05 / max(elig, 1), 2),
        "named_fdp05_pct_total": round(100.0 * named05 / max(total_faces, 1), 2),
        "named_fdp10_count": named10,
        "threshold_05": None,
        "bigram_rep_pct_elig": round(100.0 * bg_cnt / max(elig, 1), 2),
        "null_split_false_mean": None,
        "null_split_false_max": None,
        "smp_named_hits": None,
        "smp_named_tot": None,
        "smp_named_rate": None,
        "smp_shuf_rate": None,
        "ablation_verified": ab_row is not None,
        "ablation_check": ab_row,
        "ablation_checked_layers": ab_layers,
    }


def _extract_input_triggers(arrays: Dict[str, np.ndarray], s_idx: int, n: int, vocab: Sequence[str]) -> List[Dict[str, Any]]:
    tids = arrays["top_token_id"][s_idx, n]
    offs = arrays["top_offset"][s_idx, n]
    cnt_e = arrays["top_cnt_e"][s_idx, n]
    cnt_o = arrays["top_cnt_o"][s_idx, n]
    lifts = arrays["top_lift"][s_idx, n]
    out: List[Dict[str, Any]] = []
    for k_i in range(int(tids.shape[0])):
        tid = int(tids[k_i])
        ce = int(cnt_e[k_i])
        co = int(cnt_o[k_i])
        if tid < 0 or (ce == 0 and co == 0):
            continue
        off = int(offs[k_i])
        out.append({
            "tok": _tok_str(vocab, tid),
            "id": tid,
            "offset": off,
            "offset_label": OFFSET_LABELS.get(off, f"{off:+d}"),
            "cnt_e": ce,
            "cnt_o": co,
            "lift": round(float(lifts[k_i]), 2),
        })
    return out


def _input_census_neuron(root: Path, layer: int, n: int, vocab: Sequence[str]) -> Optional[Dict[str, Any]]:
    ic = _load_input_census(root, layer)
    if ic is None:
        return None
    arrays = ic["arrays"]
    if not (0 <= n < int(arrays["named_fdp05"].shape[1])):
        return None
    ab_map, ab_layers = _ablation_by_layer(root)
    faces: Dict[str, Any] = {}
    for s_idx, sg in ((0, "+"), (1, "-")):
        elig = bool(arrays["eligible"][s_idx, n])
        named05 = bool(arrays["named_fdp05"][s_idx, n])
        named10 = bool(arrays["named_fdp10"][s_idx, n]) if "named_fdp10" in arrays else named05
        sc = float(arrays["score"][s_idx, n])
        bg = bool(arrays["bigram_rep"][s_idx, n]) if "bigram_rep" in arrays else False
        sh = int(arrays["smp_hits"][s_idx, n]) if "smp_hits" in arrays else 0
        st = int(arrays["smp_total"][s_idx, n]) if "smp_total" in arrays else 0
        faces[sg] = {
            "eligible": elig,
            "named_fdp05": named05,
            "named_fdp10": named10,
            "rank": ic["rank_of"].get((s_idx, n)) if named05 else None,
            "score": round(sc, 3),
            "bigram_rep": bg,
            "smp_hits": sh,
            "smp_total": st,
            "triggers": _extract_input_triggers(arrays, s_idx, n, vocab) if named05 else [],
        }
    return {
        "regime": "input_census",
        "label": REGIME_LABELS["input_census"],
        "no_raw_text": True,
        "faces": faces,
        "n_named_faces": len(ic["named_order"]),
        "meta": _ic_layer_meta(root, ic, ab_map, ab_layers),
    }


# ------------------------------------------------------------------------------------------- fdr_rescue (L30-L54)


def _fdr_path(root: Path, layer: int) -> Path:
    return root / "fdr_rescue" / f"L{int(layer)}_whole_named.json"


def _fdr_summary_map(root: Path) -> Dict[int, Dict[str, Any]]:
    data = _cached_json(root / "fdr_rescue" / "summary.json")
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return out
    for row in data.get("runs") or []:
        if not isinstance(row, dict) or "layer" not in row:
            continue
        tag = str(row.get("tag") or "")
        if tag.endswith("_trial"):
            continue
        layer = int(row["layer"])
        out[layer] = {
            "layer": layer,
            "tag": tag,
            "design": row.get("design"),
            "neurons": row.get("neurons"),
            "controls": row.get("controls"),
            "readlib_named": row.get("readlib_named"),
            "fdp05": row.get("fdp05"),
            "fdp10": row.get("fdp10"),
            "n3_covered": row.get("n3_covered"),
            "null_splits": row.get("null_splits"),
            "oos_3way": row.get("oos_3way"),
            "has_named_file": _fdr_path(root, layer).is_file(),
        }
    return out


def _load_fdr_named(root: Path, layer: int) -> Optional[Dict[str, Any]]:
    path = _fdr_path(root, layer)
    if not path.is_file():
        return None
    key = ("fdr",) + _stamp(path)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("neurons") or []
    by_n: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    for rank, rec in enumerate(rows, 1):
        by_n[int(rec["neuron"])] = (rank, rec)
    entry = {
        "layer": int(layer),
        "tag": raw.get("tag"),
        "design": raw.get("design"),
        "count": int(raw.get("count", len(rows))),
        "rows": rows,
        "by_n": by_n,
    }
    with _LOCK:
        _CACHE[key] = entry
    return entry


def _fdr_meta_card(root: Path, layer: int, named: Optional[Dict[str, Any]], sm_map: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    sm = sm_map.get(layer)
    if sm is not None:
        return sm
    if named is None:
        return None
    return {
        "layer": layer,
        "tag": named.get("tag"),
        "design": named.get("design"),
        "neurons": None,
        "controls": None,
        "readlib_named": None,
        "fdp05": {"count": named["count"], "threshold": None, "fdp": None},
        "fdp10": None,
        "n3_covered": None,
        "null_splits": None,
        "oos_3way": None,
        "has_named_file": True,
    }


def _fdr_rescue_neuron(root: Path, layer: int, n: int) -> Optional[Dict[str, Any]]:
    sm_map = _fdr_summary_map(root)
    named = _load_fdr_named(root, layer)
    meta = _fdr_meta_card(root, layer, named, sm_map)
    if meta is None:
        return None
    hit = named["by_n"].get(int(n)) if named is not None else None
    if hit is None:
        return {
            "regime": "fdr_rescue",
            "label": REGIME_LABELS["fdr_rescue"],
            "named_fdp05": False,
            "n_named": named["count"] if named is not None else 0,
            "meta": meta,
        }
    rank, rec = hit
    return {
        "regime": "fdr_rescue",
        "label": REGIME_LABELS["fdr_rescue"],
        "named_fdp05": True,
        "rank": rank,
        "n_named": named["count"] if named is not None else 0,
        "T_dec": rec.get("T_dec"),
        "rep_AB_res": rec.get("rep_AB_res"),
        "rep_BA_res": rec.get("rep_BA_res"),
        "covered": rec.get("covered"),
        "rep_words_AB": rec.get("rep_words_AB") or [],
        "rep_words_BA": rec.get("rep_words_BA") or [],
        "meta": meta,
    }


# ----------------------------------------------------------------------------------------------------- public API


def available_regimes(root: Path) -> Dict[str, List[int]]:
    """Return sorted zero-indexed layer lists currently available in each supplementary regime."""
    fo_dir = root / "forced_odd"
    ic_dir = root / "input_census"
    fdr_dir = root / "fdr_rescue"

    fo_layers: List[int] = []
    if fo_dir.is_dir():
        for p in fo_dir.iterdir():
            m = _FO_RE.fullmatch(p.name)
            if m and p.is_file():
                fo_layers.append(int(m.group(1)))

    ic_layers: List[int] = []
    if ic_dir.is_dir():
        for p in ic_dir.iterdir():
            m = _IC_RE.fullmatch(p.name)
            if m and p.is_file():
                ic_layers.append(int(m.group(1)))

    fdr_layers: List[int] = []
    if fdr_dir.is_dir():
        for p in fdr_dir.iterdir():
            m = _FDR_RE.fullmatch(p.name)
            if m and p.is_file():
                layer = int(m.group(1))
                named = _load_fdr_named(root, layer)
                if named is not None and named["count"] > 0:
                    fdr_layers.append(layer)

    return {
        "forced_odd": sorted(fo_layers),
        "input_census": sorted(ic_layers),
        "fdr_rescue": sorted(fdr_layers),
    }


def layer_summaries(root: Path) -> Dict[str, Any]:
    """Per-layer summaries and provenance cards across all three supplementary regimes."""
    avail = available_regimes(root)
    fo_rows = []
    for layer in avail["forced_odd"]:
        fo = _load_forced_odd(root, layer)
        if fo is not None:
            fo_rows.append(_forced_odd_meta_card(fo))

    ab_map, ab_layers = _ablation_by_layer(root)
    ic_rows = []
    for layer in avail["input_census"]:
        ic = _load_input_census(root, layer)
        if ic is not None:
            ic_rows.append(_ic_layer_meta(root, ic, ab_map, ab_layers))

    sm_map = _fdr_summary_map(root)
    fdr_layers = sorted(set(sm_map.keys()) | set(avail["fdr_rescue"]))
    fdr_rows = []
    for layer in fdr_layers:
        named = _load_fdr_named(root, layer) if layer in avail["fdr_rescue"] else None
        card = _fdr_meta_card(root, layer, named, sm_map)
        if card is not None:
            fdr_rows.append(card)

    ab_summary: Optional[Dict[str, Any]] = None
    if ab_map:
        ab_raw = _cached_json(root / "input_census" / "ablation_check.json") or {}
        ab_summary = {
            "quiet_thresh": ab_raw.get("quiet_thresh"),
            "min_extreme": ab_raw.get("min_extreme"),
            "layers": [ab_map[l] for l in ab_layers],
        }

    return {
        "labels": dict(REGIME_LABELS),
        "forced_odd": fo_rows,
        "input_census": ic_rows,
        "input_ablation_check": ab_summary,
        "fdr_rescue": fdr_rows,
    }


def neuron_view(root: Path, layer: int, n: int, vocab: Sequence[str]) -> Dict[str, Any]:
    """Full read-only supplementary census record for `(layer, n)` across any matching regimes."""
    return {
        "layer": int(layer),
        "n": int(n),
        "forced_odd": _forced_odd_neuron(root, layer, n, vocab),
        "input_census": _input_census_neuron(root, layer, n, vocab),
        "fdr_rescue": _fdr_rescue_neuron(root, layer, n),
    }


def hover_summary(root: Path, layer: int, n: int, vocab: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Compact hover summary returned ONLY when the neuron/face cleared FDR <= 5%."""
    out: Dict[str, Any] = {}

    fo = _load_forced_odd(root, layer)
    if fo is not None and 0 <= n < int(fo["arrays"]["named_dual05"].shape[0]) and bool(fo["arrays"]["named_dual05"][n]):
        arrays = fo["arrays"]
        pieces = _extract_forced_odd_pieces(arrays, n, vocab, k=3)
        is_direct = bool(arrays["is_direct"][n]) if "is_direct" in arrays else False
        out["forced_odd"] = {
            "label": REGIME_LABELS["forced_odd"],
            "tag": "direct_unembed" if is_direct else "circuit_mediated",
            "T_min": round(float(fo["T_min"][n]), 2),
            "pos": [p["tok"] for p in pieces["pos"]],
            "neg": [p["tok"] for p in pieces["neg"]],
        }

    ic = _load_input_census(root, layer)
    if ic is not None and 0 <= n < int(ic["arrays"]["named_fdp05"].shape[1]):
        arrays = ic["arrays"]
        ic_faces: Dict[str, List[Dict[str, Any]]] = {}
        for s_idx, sg in ((0, "+"), (1, "-")):
            if bool(arrays["named_fdp05"][s_idx, n]):
                ic_faces[sg] = [
                    {"tok": t["tok"], "offset_label": t["offset_label"], "lift": t["lift"]}
                    for t in _extract_input_triggers(arrays, s_idx, n, vocab)[:2]
                ]
        if ic_faces:
            out["input_census"] = {
                "label": REGIME_LABELS["input_census"],
                "faces": ic_faces,
            }

    fdr = _load_fdr_named(root, layer)
    if fdr is not None and int(n) in fdr["by_n"]:
        _, rec = fdr["by_n"][int(n)]
        out["fdr_rescue"] = {
            "label": REGIME_LABELS["fdr_rescue"],
            "T_dec": rec.get("T_dec"),
            "covered": rec.get("covered"),
            "words": [w for w, _ in (rec.get("rep_words_AB") or [])[:4]],
        }

    return out or None


def _matches_query(ql: str, nid: int, name: str, tokens: Sequence[str], extra: Sequence[str] = ()) -> bool:
    if not ql:
        return True
    if ql == str(nid) or ql in f"n{nid}" or ql in name.lower():
        return True
    for tok in tokens:
        if ql in tok.lower():
            return True
    for ex in extra:
        if ql in ex.lower():
            return True
    return False


def list_units(
    root: Path,
    layer: int,
    regime: str,
    vocab: Sequence[str],
    notes: Dict[str, Any],
    q: str = "",
    limit: int = 600,
    face: str = "both",
) -> Dict[str, Any]:
    """Ranked read-only listing of passing units for one `(layer, regime)`."""
    if regime not in REGIME_LABELS:
        raise ValueError(f"Unknown supplementary regime: {regime!r}")
    ql = q.strip().lower()
    cap = max(1, min(int(limit), 2000))

    if regime == "forced_odd":
        fo = _load_forced_odd(root, layer)
        if fo is None:
            return {"regime": regime, "label": REGIME_LABELS[regime], "layer": int(layer), "available": False, "meta": None, "total": 0, "matched": 0, "rows": []}
        arrays = fo["arrays"]
        t_col = arrays["T_col"]
        t_iso = arrays["T_iso"]
        t_min = fo["T_min"]
        is_dir = arrays["is_direct"] if "is_direct" in arrays else None
        q_dir = arrays["q_direct"] if "q_direct" in arrays else None
        rows: List[Dict[str, Any]] = []
        matched = 0
        for rank, n in enumerate(fo["named_order"], 1):
            tag = "direct_unembed" if (is_dir is not None and bool(is_dir[n])) else "circuit_mediated"
            name = _note_name(notes, layer, n)
            pieces = _extract_forced_odd_pieces(arrays, n, vocab, k=5)
            toks = [p["tok"] for p in pieces["pos"]] + [p["tok"] for p in pieces["neg"]]
            if not _matches_query(ql, n, name, toks, extra=(tag,)):
                continue
            matched += 1
            if len(rows) < cap:
                rows.append({
                    "n": n,
                    "rank": rank,
                    "T_min": round(float(t_min[n]), 3),
                    "T_col": round(float(t_col[n]), 3),
                    "T_iso": round(float(t_iso[n]), 3),
                    "tag": tag,
                    "q_direct": _round_opt(q_dir[n], 4) if q_dir is not None else None,
                    "pos": [[p["tok"], p["score"]] for p in pieces["pos"]],
                    "neg": [[p["tok"], p["score"]] for p in pieces["neg"]],
                    "name": name,
                })
        return {
            "regime": regime,
            "label": REGIME_LABELS[regime],
            "layer": int(layer),
            "available": True,
            "meta": _forced_odd_meta_card(fo),
            "total": len(fo["named_order"]),
            "matched": matched,
            "rows": rows,
        }

    if regime == "input_census":
        ic = _load_input_census(root, layer)
        if ic is None:
            return {"regime": regime, "label": REGIME_LABELS[regime], "layer": int(layer), "available": False, "meta": None, "total": 0, "matched": 0, "rows": []}
        arrays = ic["arrays"]
        ab_map, ab_layers = _ablation_by_layer(root)
        want_s = None if face == "both" else (0 if face == "+" else 1)
        rows = []
        matched = 0
        total_face = 0
        for s_idx, n in ic["named_order"]:
            if want_s is not None and s_idx != want_s:
                continue
            total_face += 1
            fg = "+" if s_idx == 0 else "-"
            name = _note_name(notes, layer, n)
            triggers = _extract_input_triggers(arrays, s_idx, n, vocab)
            toks = [t["tok"] for t in triggers]
            if not _matches_query(ql, n, name, toks):
                continue
            matched += 1
            if len(rows) < cap:
                rows.append({
                    "n": n,
                    "face": fg,
                    "rank": ic["rank_of"][(s_idx, n)],
                    "score": round(float(arrays["score"][s_idx, n]), 3),
                    "bigram_rep": bool(arrays["bigram_rep"][s_idx, n]) if "bigram_rep" in arrays else False,
                    "smp_hits": int(arrays["smp_hits"][s_idx, n]) if "smp_hits" in arrays else 0,
                    "smp_total": int(arrays["smp_total"][s_idx, n]) if "smp_total" in arrays else 0,
                    "triggers": triggers,
                    "name": name,
                })
        return {
            "regime": regime,
            "label": REGIME_LABELS[regime],
            "layer": int(layer),
            "available": True,
            "no_raw_text": True,
            "meta": _ic_layer_meta(root, ic, ab_map, ab_layers),
            "total": total_face,
            "matched": matched,
            "rows": rows,
        }

    # fdr_rescue
    fdr = _load_fdr_named(root, layer)
    sm_map = _fdr_summary_map(root)
    if fdr is None:
        return {
            "regime": regime,
            "label": REGIME_LABELS[regime],
            "layer": int(layer),
            "available": int(layer) in sm_map,
            "meta": sm_map.get(int(layer)),
            "total": 0,
            "matched": 0,
            "rows": [],
        }
    rows = []
    matched = 0
    for rank, rec in enumerate(fdr["rows"], 1):
        n = int(rec["neuron"])
        name = _note_name(notes, layer, n)
        w_ab = rec.get("rep_words_AB") or []
        w_ba = rec.get("rep_words_BA") or []
        toks = [w for w, _ in w_ab] + [w for w, _ in w_ba]
        if not _matches_query(ql, n, name, toks):
            continue
        matched += 1
        if len(rows) < cap:
            rows.append({
                "n": n,
                "rank": rank,
                "T_dec": rec.get("T_dec"),
                "rep_AB_res": rec.get("rep_AB_res"),
                "rep_BA_res": rec.get("rep_BA_res"),
                "covered": rec.get("covered"),
                "words": w_ab[:5],
                "name": name,
            })
    return {
        "regime": regime,
        "label": REGIME_LABELS[regime],
        "layer": int(layer),
        "available": True,
        "meta": _fdr_meta_card(root, int(layer), fdr, sm_map),
        "total": fdr["count"],
        "matched": matched,
        "rows": rows,
    }


def search_regime(
    root: Path,
    regime: str,
    q: str,
    vocab: Sequence[str],
    notes: Dict[str, Any],
    limit: int = 200,
) -> Dict[str, Any]:
    """Read-only cross-layer search within one supplementary regime."""
    if regime not in REGIME_LABELS:
        raise ValueError(f"Unknown supplementary regime: {regime!r}")
    layers = available_regimes(root)[regime]
    cap = max(1, min(int(limit), 500))
    all_rows: List[Dict[str, Any]] = []
    by_layer: List[Dict[str, Any]] = []
    total = 0

    for layer in layers:
        res = list_units(root, layer, regime, vocab, notes, q=q, limit=cap)
        m_cnt = int(res["matched"])
        total += m_cnt
        by_layer.append({"layer": layer, "matched": m_cnt, "total": int(res["total"])})
        for r in res["rows"]:
            all_rows.append({"layer": layer, **r})

    score_key = "T_min" if regime == "forced_odd" else ("score" if regime == "input_census" else "T_dec")
    all_rows.sort(key=lambda r: (-float(r.get(score_key) or 0.0), int(r["layer"]), int(r["n"])))
    return {
        "regime": regime,
        "label": REGIME_LABELS[regime],
        "q": q,
        "total": total,
        "by_layer": by_layer,
        "results": all_rows[:cap],
    }
