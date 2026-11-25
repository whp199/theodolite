"""Read-only readers for supplementary census products under ``supplementary_dir()``.

Covers two exploratory supplementary regimes under ``atlas_dir() / "census"`` without touching
or conflating registered ``census/runs/`` rollouts or ``backend/census.py`` semantics, and without
importing GPU modules (``backend/forced_odd.py``):

- ``forced_odd`` (L55-L59): exploratory push read on set text (pieces shifted most above random controls)
- ``input_census`` (L0-L29): exploratory input-firing table on stored passages (offsets -1 and 0 only)
"""

from __future__ import annotations

import json
import re
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend import atlas_index as ai

REGIME_LABELS: Dict[str, str] = {
    "forced_odd": (
        "exploratory push read on set text: word-pieces whose log-odds shift most above "
        "two families of random directions (not an owner-calibrated census)"
    ),
    "input_census": (
        "exploratory input-firing table: word-pieces seen repeatedly before or at a face's "
        "strongest stored firings (not causal, not a census)"
    ),
}

# Input-firing triggers strictly include only causal offsets (-1, 0). Offset +1 is next-piece only.
OFFSET_LABELS: Dict[int, str] = {-1: "prev", 0: "peak"}

FORCED_ODD_STALE = (
    "read on set text from the model before the vault edits in force: "
    "base activations, pushed write directions, and later blocks may have changed"
)
INPUT_CENSUS_STALE = (
    "read from stored passage activations recorded on the model before the earlier-layer "
    "vault edits in force"
)

_LOCK = threading.RLock()
# Bounded per-(kind, resolved_path) cache storing (stamp, value) so republishing replaces old entries in place.
_CACHE: Dict[Tuple[str, str], Tuple[Tuple[Any, ...], Any]] = {}

_FO_RE = re.compile(r"L(\d+)_forced_odd\.npz$")
_IC_RE = re.compile(r"L(\d+)_input_census\.npz$")

_REQUIRED_FO_KEYS = (
    "T_col",
    "T_iso",
    "named_dual05",
    "topA_p",
    "topB_p",
    "topA_n",
    "topB_n",
    "posA_onB",
    "posB_onA",
    "negA_onB",
    "negB_onA",
)
_REQUIRED_IC_KEYS = (
    "eligible",
    "named_fdp05",
    "score",
    "top_token_id",
    "top_offset",
    "top_cnt_e",
    "top_cnt_o",
    "top_lift",
)


def supplementary_dir() -> Path:
    """Canonical root for supplementary gap regimes: ``atlas_dir() / 'census'``."""
    return ai.atlas_dir() / "census"


def regime_stale(
    edited_units: Iterable[Any],
    *,
    regime: Optional[str] = None,
    layer: Optional[int] = None,
    has_forced_odd: bool = True,
    has_input_census: bool = True,
) -> Dict[str, str]:
    """Return regime-appropriate stale notice while vault edits are in force, else ``{}``."""
    edited = sorted({int(x[0] if isinstance(x, (tuple, list)) else x) for x in edited_units})
    if not edited:
        return {}
    if regime == "input_census":
        if layer is not None and not any(el < int(layer) for el in edited):
            return {}
        return {"stale": INPUT_CENSUS_STALE}
    if regime == "forced_odd":
        return {"stale": FORCED_ODD_STALE}
    if layer is not None and not has_forced_odd and has_input_census:
        if not any(el < int(layer) for el in edited):
            return {}
        return {"stale": INPUT_CENSUS_STALE}
    return {"stale": FORCED_ODD_STALE}


def _stamp(path: Path) -> Tuple[str, int, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size), int(getattr(st, "st_ino", 0)))


def _record_refused(refused: Optional[Dict[str, str]], key: str, reason: str) -> None:
    if refused is not None and key not in refused:
        refused[key] = reason


def _cached_json(path: Path, refused: Optional[Dict[str, str]] = None, refuse_key: Optional[str] = None) -> Optional[Any]:
    if not path.is_file():
        return None
    resolved = str(path.resolve())
    cache_key = ("json", resolved)
    try:
        st = _stamp(path)
    except OSError as exc:
        _record_refused(refused, refuse_key or path.name, f"unreadable file: {exc}")
        return None
    with _LOCK:
        hit = _CACHE.get(cache_key)
        if hit is not None and hit[0] == st:
            return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        _record_refused(refused, refuse_key or path.name, f"malformed JSON: {exc}")
        return None
    with _LOCK:
        _CACHE[cache_key] = (st, data)
    return data


def _model_weights_error(
    meta: Optional[Dict[str, Any]],
    model: Optional[str],
    weights: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return refusal message when ``meta`` does not match ``model`` / ``weights``, else ``None``."""
    if model is not None:
        if not isinstance(meta, dict) or not meta.get("model"):
            return "missing model provenance in metadata"
        if str(meta.get("model")) != str(model):
            return f"built for model {meta.get('model')!r}, not {model!r}"
    if weights is not None and isinstance(meta, dict) and meta.get("weights") is not None:
        diff = ai.weights_differ(meta.get("weights"), weights)
        if diff is not None:
            return f"built from other weights ({diff})"
    return None


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


def _load_forced_odd(
    root: Path,
    layer: int,
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    path = _fo_path(root, layer)
    if not path.is_file():
        return None
    rkey = f"forced_odd:L{int(layer)}"
    meta_path = _fo_meta_path(root, layer)
    meta: Optional[Dict[str, Any]] = None
    if meta_path.is_file():
        raw_meta = _cached_json(meta_path, refused=refused, refuse_key=rkey)
        if raw_meta is None:
            return None
        if not isinstance(raw_meta, dict):
            _record_refused(refused, rkey, "meta JSON must be an object")
            return None
        meta = raw_meta
    elif model is not None:
        _record_refused(refused, rkey, "missing metadata for model verification")
        return None

    err = _model_weights_error(meta, model, weights)
    if err is not None:
        _record_refused(refused, rkey, err)
        return None

    try:
        st = (_stamp(path), _stamp(meta_path) if meta_path.is_file() else None)
    except OSError as exc:
        _record_refused(refused, rkey, f"unreadable file: {exc}")
        return None

    cache_key = ("fo", str(path.resolve()))
    with _LOCK:
        hit = _CACHE.get(cache_key)
        if hit is not None and hit[0] == st:
            return hit[1]

    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            missing = [k for k in _REQUIRED_FO_KEYS if k not in files]
            if missing:
                raise KeyError(f"missing required arrays: {', '.join(missing)}")
            arrays = {k: z[k] for k in z.files}
        t_col = np.asarray(arrays["T_col"], dtype=np.float32)
        t_iso = np.asarray(arrays["T_iso"], dtype=np.float32)
        named = np.asarray(arrays["named_dual05"], dtype=bool)
        if t_col.ndim != 1 or t_iso.shape != t_col.shape or named.shape != t_col.shape:
            raise ValueError("unexpected 1D array shapes in forced_odd npz")
        n_units = int(t_col.shape[0])
        if int(arrays["topA_p"].ndim) != 2 or int(arrays["topA_p"].shape[0]) != n_units:
            raise ValueError("unexpected piece matrix shape in forced_odd npz")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        _record_refused(refused, rkey, f"invalid forced_odd archive: {exc}")
        return None

    t_min = np.minimum(t_col, t_iso)
    named_idx = np.flatnonzero(named)
    order = sorted(
        (int(i) for i in named_idx),
        key=lambda i: (-float(t_min[i]), -float(t_col[i]), i),
    )
    rank_of = np.zeros(n_units, dtype=np.int32)
    for r, ni in enumerate(order, 1):
        rank_of[ni] = r
    entry = {
        "layer": int(layer),
        "arrays": arrays,
        "meta": meta,
        "T_min": t_min,
        "named_order": order,
        "rank_of": rank_of,
    }
    with _LOCK:
        _CACHE[cache_key] = (st, entry)
    return entry


def _extract_forced_odd_pieces(arrays: Dict[str, np.ndarray], n: int, vocab: Sequence[str], k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Match backend/forced_odd.py:extract_replicated_pieces exact A->B and B->A replication logic."""
    rep_pos: List[Tuple[str, float, int]] = []
    rep_neg: List[Tuple[str, float, int]] = []
    seen_p = set()
    seen_n = set()
    width = int(arrays["topA_p"].shape[1])
    for j in range(width):
        pid = int(arrays["topA_p"][n, j])
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


def _own_lens_info(
    layer: int,
    n: int,
    arrays: Dict[str, np.ndarray],
    meta: Optional[Dict[str, Any]],
    n_layers: Optional[int],
) -> Tuple[Optional[bool], Optional[float], str]:
    """Return ``(own_lens_match, q_own_lens, own_lens_note)``; never infers circuit mediation."""
    if int(layer) == 59 or (n_layers is not None and int(layer) == int(n_layers) - 1):
        return None, None, "not applicable at last layer (only final norm and unembedding follow)"
    ww = (meta or {}).get("weight_word_diag") if isinstance(meta, dict) else None
    if ww is None or (isinstance(ww, dict) and ww.get("available") is False) or "is_direct" not in arrays:
        return None, None, "own weight-lens comparison unavailable"
    matched = bool(arrays["is_direct"][n])
    q_val = _round_opt(arrays["q_direct"][n], 4) if "q_direct" in arrays else None
    note = (
        "forced pieces agree with own weight-lens pieces (BH q <= 0.05)"
        if matched
        else "forced pieces did not meet own-lens match bar (q > 0.05; does not show a circuit)"
    )
    return matched, q_val, note


def _forced_odd_meta_card(fo: Dict[str, Any]) -> Dict[str, Any]:
    arrays = fo["arrays"]
    meta = fo["meta"]
    d_ff = int(arrays["named_dual05"].shape[0])
    dual05_count = int(np.asarray(arrays["named_dual05"], dtype=bool).sum())
    dual10_count = int(np.asarray(arrays["named_dual10"], dtype=bool).sum()) if "named_dual10" in arrays else None

    if meta:
        vs_col = meta.get("vs_columns") or {}
        vs_iso = meta.get("vs_isotropic") or {}
        return {
            "layer": fo["layer"],
            "d_ff": int(meta.get("d_ff", d_ff)),
            "model": meta.get("model"),
            "weights_id": meta.get("weights_id"),
            "exploratory": True,
            "not_taken": True,
            "doses_uncalibrated": True,
            "doses_note": "Doses follow this exploratory ladder (0.25 nats at 90th percentile), not owner calibration.",
            "piece_filter": "Restricted to 3+-letter non-special pieces (172,064 of 262,144), weighted equally without base-probability weighting.",
            "readout_mode": meta.get("readout_mode"),
            "dose_ceil": meta.get("dose_ceil"),
            "dose_stat": meta.get("dose_stat"),
            "target_kl": meta.get("target_kl"),
            "n_cols": meta.get("n_cols"),
            "n_iso": meta.get("n_iso"),
            "dual05_count": int(meta.get("dual05_count", dual05_count)),
            "dual05_pct": meta.get("dual05_pct", round(100.0 * dual05_count / max(d_ff, 1), 2)),
            "dual10_count": meta.get("dual10_count", dual10_count),
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
            "weight_lens_agreement": meta.get("weight_lens_agreement"),
        }

    return {
        "layer": fo["layer"],
        "d_ff": d_ff,
        "model": None,
        "weights_id": None,
        "exploratory": True,
        "not_taken": True,
        "doses_uncalibrated": True,
        "doses_note": "Doses follow this exploratory ladder (0.25 nats at 90th percentile), not owner calibration.",
        "piece_filter": "Restricted to 3+-letter non-special pieces (172,064 of 262,144), weighted equally without base-probability weighting.",
        "readout_mode": None,
        "dose_ceil": None,
        "dose_stat": None,
        "target_kl": None,
        "n_cols": int(arrays["T_ctl_col"].shape[0]) if "T_ctl_col" in arrays else None,
        "n_iso": int(arrays["T_ctl_iso"].shape[0]) if "T_ctl_iso" in arrays else None,
        "dual05_count": dual05_count,
        "dual05_pct": round(100.0 * dual05_count / max(d_ff, 1), 2),
        "dual10_count": dual10_count,
        "vs_columns": None,
        "vs_isotropic": None,
        "half_c": None,
        "weight_lens_agreement": None,
    }


def _forced_odd_neuron(
    root: Path,
    layer: int,
    n: int,
    vocab: Sequence[str],
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    n_layers: Optional[int] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    fo = _load_forced_odd(root, layer, model=model, weights=weights, refused=refused)
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
        "exploratory": True,
        "not_taken": True,
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
        own_match, q_own, own_note = _own_lens_info(layer, n, arrays, fo["meta"], n_layers)
        pieces = _extract_forced_odd_pieces(arrays, n, vocab, k=5)
        rk = int(fo["rank_of"][n])
        out.update({
            "rank": rk if rk > 0 else None,
            "n_named": len(fo["named_order"]),
            "own_lens_match": own_match,
            "q_own_lens": q_own,
            "own_lens_note": own_note,
            "pos": pieces["pos"],
            "neg": pieces["neg"],
        })
    else:
        out.update({
            "rank": None,
            "n_named": len(fo["named_order"]),
            "own_lens_match": None,
            "q_own_lens": None,
            "own_lens_note": None,
            "pos": [],
            "neg": [],
            "why_no_words": (
                "Did not exceed the 5% dual-control cut on 3+-letter pieces; no shift pieces shown."
            ),
        })
    return out


# ------------------------------------------------------------------------------------------ input_census (L0-L29)


def _ic_path(root: Path, layer: int) -> Path:
    return root / "input_census" / f"L{int(layer):02d}_input_census.npz"


def _ic_summary_raw(
    root: Path,
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    path = root / "input_census" / "summary.json"
    if not path.is_file():
        if model is not None:
            _record_refused(refused, "input_census", "missing summary.json for model verification")
        return None
    raw = _cached_json(path, refused=refused, refuse_key="input_census")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _record_refused(refused, "input_census", "input_census summary.json must be an object")
        return None
    err = _model_weights_error(raw, model, weights)
    if err is not None:
        _record_refused(refused, "input_census", err)
        return None
    return raw


def _load_input_census(
    root: Path,
    layer: int,
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    path = _ic_path(root, layer)
    if not path.is_file():
        return None
    rkey = f"input_census:L{int(layer):02d}"
    summary = _ic_summary_raw(root, model=model, weights=weights, refused=refused)
    if summary is None and model is not None:
        _record_refused(refused, rkey, refused.get("input_census", "missing or refused summary metadata") if refused else "model check failed")
        return None

    try:
        st = _stamp(path)
    except OSError as exc:
        _record_refused(refused, rkey, f"unreadable file: {exc}")
        return None
    cache_key = ("ic", str(path.resolve()))
    with _LOCK:
        hit = _CACHE.get(cache_key)
        if hit is not None and hit[0] == st:
            return hit[1]

    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            missing = [k for k in _REQUIRED_IC_KEYS if k not in files]
            if missing:
                raise KeyError(f"missing required arrays: {', '.join(missing)}")
            arrays = {k: z[k] for k in z.files}
        named05 = np.asarray(arrays["named_fdp05"], dtype=bool)
        elig = np.asarray(arrays["eligible"], dtype=bool)
        score = np.asarray(arrays["score"], dtype=np.float32)
        tids = np.asarray(arrays["top_token_id"])
        if named05.ndim != 2 or named05.shape[0] != 2 or elig.shape != named05.shape or score.shape != named05.shape:
            raise ValueError("unexpected 2D array shapes in input_census npz")
        if tids.ndim != 3 or tids.shape[:2] != named05.shape:
            raise ValueError("unexpected top_token_id shape in input_census npz")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        _record_refused(refused, rkey, f"invalid input_census archive: {exc}")
        return None

    coords = np.argwhere(named05)
    order = sorted(
        ((int(si), int(ni)) for si, ni in coords),
        key=lambda pair: (-float(score[pair[0], pair[1]]), pair[1], pair[0]),
    )
    rank_of = np.zeros(named05.shape, dtype=np.int32)
    for r, (si, ni) in enumerate(order, 1):
        rank_of[si, ni] = r
    entry = {
        "layer": int(layer),
        "arrays": arrays,
        "named_order": order,
        "rank_of": rank_of,
    }
    with _LOCK:
        _CACHE[cache_key] = (st, entry)
    return entry


def _ablation_by_layer(
    root: Path,
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[int, Dict[str, Any]]], List[int]]:
    data = _cached_json(root / "input_census" / "ablation_check.json", refused=refused, refuse_key="input_census:ablation")
    if not isinstance(data, dict):
        return None, []
    err = _model_weights_error(data, model, weights)
    if err is not None:
        _record_refused(refused, "input_census:ablation", err)
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
            "control_kind": data.get("control_kind") or row.get("control_kind"),
            "kl_pk": {
                "auc_active_vs_control": pk.get("auc_active_vs_control", pk.get("auc_active_vs_quiet")),
                "wilcoxon_p": pk.get("wilcoxon_p"),
                "frac_neurons_max4_gt_0.005": pk.get("frac_neurons_max4_gt_0.005"),
            },
            "kl_span": {
                "auc_active_vs_control": sp.get("auc_active_vs_control", sp.get("auc_active_vs_quiet")),
                "wilcoxon_p": sp.get("wilcoxon_p"),
                "frac_neurons_max4_gt_0.005": sp.get("frac_neurons_max4_gt_0.005"),
            },
        }
    return out, sorted(out)


def _ic_summary_by_layer(root: Path) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    data = _cached_json(root / "input_census" / "summary.json")
    out: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for row in data.get("layers") or []:
            if isinstance(row, dict) and "layer" in row:
                out[int(row["layer"])] = row
        return data, out
    return None, out


def _ic_layer_meta(root: Path, ic: Dict[str, Any], ab_map: Optional[Dict[int, Dict[str, Any]]], ab_layers: List[int]) -> Dict[str, Any]:
    layer = ic["layer"]
    arrays = ic["arrays"]
    sm_top, sm_map = _ic_summary_by_layer(root)
    row = sm_map.get(layer)
    ab_row = ab_map.get(layer) if ab_map else None
    n_ff = int(arrays["named_fdp05"].shape[1])
    total_faces = 2 * n_ff
    elig = int(np.asarray(arrays["eligible"], dtype=bool).sum())
    named05 = int(np.asarray(arrays["named_fdp05"], dtype=bool).sum())
    named10 = int(np.asarray(arrays["named_fdp10"], dtype=bool).sum()) if "named_fdp10" in arrays else named05
    bg_cnt = int(np.asarray(arrays["bigram_rep"], dtype=bool).sum()) if "bigram_rep" in arrays else 0

    base = {
        "layer": layer,
        "model": (sm_top or {}).get("model"),
        "weights_id": (sm_top or {}).get("weights_id"),
        "exploratory": True,
        "not_taken": True,
        "split_kind": (sm_top or {}).get("split_kind", "document_disjoint"),
        "triggers_note": "Strictly offsets -1 (previous) and 0 (peak); any +1 pieces are shown separately as next pieces.",
        "no_passages_stored": True,
        "total_faces": (row or {}).get("total_faces", total_faces),
        "eligible_faces": (row or {}).get("eligible_faces", elig),
        "named_2plus2_count": (row or {}).get("raw_2plus2_count", named05),
        "named_fdp05_count": (row or {}).get("named_fdp05_count", named05),
        "named_fdp05_pct_elig": (row or {}).get("named_fdp05_pct_elig", round(100.0 * named05 / max(elig, 1), 2)),
        "named_fdp05_pct_total": (row or {}).get("named_fdp05_pct_total", round(100.0 * named05 / max(total_faces, 1), 2)),
        "named_fdp10_count": (row or {}).get("named_fdp10_count", named10),
        "threshold_05": (row or {}).get("threshold_05"),
        "bigram_rep_pct_elig": (row or {}).get("bigram_rep_pct_elig", round(100.0 * bg_cnt / max(elig, 1), 2)),
        "other_face_null_rate": (row or {}).get("other_face_null_rate"),
        "smp_named_hits": (row or {}).get("smp_named_hits"),
        "smp_named_tot": (row or {}).get("smp_named_tot"),
        "smp_named_rate": (row or {}).get("smp_named_rate"),
        "smp_freq_rate": (row or {}).get("smp_freq_rate"),
        "smp_shuf_rate": (row or {}).get("smp_shuf_rate"),
        "ablation_sampled": ab_row is not None,
        "ablation_sample_neurons": ab_row.get("neurons") if ab_row else None,
        "ablation_control_kind": ab_row.get("control_kind") if ab_row else None,
        "ablation_check": ab_row,
        "ablation_sampled_layers": ab_layers,
    }
    return base


def _extract_input_pieces(
    arrays: Dict[str, np.ndarray],
    s_idx: int,
    n: int,
    vocab: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(triggers[-1, 0], next_pieces[+1])``, keeping +1 strictly out of triggers."""
    tids = arrays["top_token_id"][s_idx, n]
    offs = arrays["top_offset"][s_idx, n]
    cnt_e = arrays["top_cnt_e"][s_idx, n]
    cnt_o = arrays["top_cnt_o"][s_idx, n]
    lifts = arrays["top_lift"][s_idx, n]
    triggers: List[Dict[str, Any]] = []
    next_pieces: List[Dict[str, Any]] = []
    for k_i in range(int(tids.shape[0])):
        tid = int(tids[k_i])
        ce = int(cnt_e[k_i])
        co = int(cnt_o[k_i])
        if tid < 0 or (ce == 0 and co == 0):
            continue
        off = int(offs[k_i])
        rec = {
            "tok": _tok_str(vocab, tid),
            "id": tid,
            "offset": off,
            "offset_label": OFFSET_LABELS.get(off, "next" if off == 1 else f"{off:+d}"),
            "cnt_e": ce,
            "cnt_o": co,
            "lift": round(float(lifts[k_i]), 2),
        }
        if off in (-1, 0):
            triggers.append(rec)
        elif off == 1:
            next_pieces.append(rec)

    if "next_token_id" in arrays:
        ntids = arrays["next_token_id"][s_idx, n]
        nce = arrays["next_cnt_e"][s_idx, n]
        nco = arrays["next_cnt_o"][s_idx, n]
        nlift = arrays["next_lift"][s_idx, n]
        for k_i in range(int(ntids.shape[0])):
            tid = int(ntids[k_i])
            ce = int(nce[k_i])
            co = int(nco[k_i])
            if tid < 0 or (ce == 0 and co == 0):
                continue
            next_pieces.append({
                "tok": _tok_str(vocab, tid),
                "id": tid,
                "offset": 1,
                "offset_label": "next",
                "cnt_e": ce,
                "cnt_o": co,
                "lift": round(float(nlift[k_i]), 2),
            })
    return triggers, next_pieces


def _input_census_neuron(
    root: Path,
    layer: int,
    n: int,
    vocab: Sequence[str],
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    ic = _load_input_census(root, layer, model=model, weights=weights, refused=refused)
    if ic is None:
        return None
    arrays = ic["arrays"]
    if not (0 <= n < int(arrays["named_fdp05"].shape[1])):
        return None
    ab_map, ab_layers = _ablation_by_layer(root, model=model, weights=weights, refused=refused)
    faces: Dict[str, Any] = {}
    for s_idx, sg in ((0, "+"), (1, "-")):
        elig = bool(arrays["eligible"][s_idx, n])
        named05 = bool(arrays["named_fdp05"][s_idx, n])
        named10 = bool(arrays["named_fdp10"][s_idx, n]) if "named_fdp10" in arrays else named05
        sc = float(arrays["score"][s_idx, n])
        bg = bool(arrays["bigram_rep"][s_idx, n]) if "bigram_rep" in arrays else False
        sh = int(arrays["smp_hits"][s_idx, n]) if "smp_hits" in arrays else 0
        st = int(arrays["smp_total"][s_idx, n]) if "smp_total" in arrays else 0
        if named05:
            trigs, nexts = _extract_input_pieces(arrays, s_idx, n, vocab)
            rk = int(ic["rank_of"][s_idx, n])
        else:
            trigs, nexts, rk = [], [], None
        faces[sg] = {
            "eligible": elig,
            "named": named05,
            "named_2plus2": named05,
            "named_fdp10": named10,
            "rank": rk if (rk and rk > 0) else None,
            "score": round(sc, 3),
            "bigram_rep": bg,
            "smp_hits": sh,
            "smp_total": st,
            "triggers": trigs,
            "next_pieces": nexts,
        }
    return {
        "regime": "input_census",
        "label": REGIME_LABELS["input_census"],
        "exploratory": True,
        "not_taken": True,
        "no_passages_stored": True,
        "faces": faces,
        "n_named_faces": len(ic["named_order"]),
        "meta": _ic_layer_meta(root, ic, ab_map, ab_layers),
    }


# ----------------------------------------------------------------------------------------------------- public API


def available_regimes(
    root: Path,
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Dict[str, List[int]]:
    """Return sorted zero-indexed layer lists available and model-verified per supplementary regime."""
    fo_dir = root / "forced_odd"
    ic_dir = root / "input_census"

    fo_layers: List[int] = []
    if fo_dir.is_dir():
        for p in sorted(fo_dir.iterdir()):
            m = _FO_RE.fullmatch(p.name)
            if m and p.is_file():
                layer = int(m.group(1))
                if _load_forced_odd(root, layer, model=model, weights=weights, refused=refused) is not None:
                    fo_layers.append(layer)

    ic_layers: List[int] = []
    if ic_dir.is_dir():
        for p in sorted(ic_dir.iterdir()):
            m = _IC_RE.fullmatch(p.name)
            if m and p.is_file():
                layer = int(m.group(1))
                if _load_input_census(root, layer, model=model, weights=weights, refused=refused) is not None:
                    ic_layers.append(layer)

    return {
        "forced_odd": sorted(fo_layers),
        "input_census": sorted(ic_layers),
    }


def layer_summaries(
    root: Path,
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Per-layer summaries and provenance cards across exploratory supplementary regimes."""
    ref_map: Dict[str, str] = refused if refused is not None else {}
    avail = available_regimes(root, model=model, weights=weights, refused=ref_map)
    fo_rows = []
    for layer in avail["forced_odd"]:
        fo = _load_forced_odd(root, layer, model=model, weights=weights, refused=ref_map)
        if fo is not None:
            fo_rows.append(_forced_odd_meta_card(fo))

    ab_map, ab_layers = _ablation_by_layer(root, model=model, weights=weights, refused=ref_map)
    ic_rows = []
    for layer in avail["input_census"]:
        ic = _load_input_census(root, layer, model=model, weights=weights, refused=ref_map)
        if ic is not None:
            ic_rows.append(_ic_layer_meta(root, ic, ab_map, ab_layers))

    ab_summary: Optional[Dict[str, Any]] = None
    if ab_map:
        ab_raw = _cached_json(root / "input_census" / "ablation_check.json") or {}
        ab_summary = {
            "control_kind": ab_raw.get("control_kind"),
            "note": "Sampling check (64 neurons at sampled layers only); not a pass/fail gate.",
            "layers": [ab_map[l] for l in ab_layers],
        }

    return {
        "exploratory": True,
        "not_taken": True,
        "labels": dict(REGIME_LABELS),
        "forced_odd": fo_rows,
        "input_census": ic_rows,
        "input_ablation_check": ab_summary,
        "refused": ref_map,
    }


def neuron_view(
    root: Path,
    layer: int,
    n: int,
    vocab: Sequence[str],
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    n_layers: Optional[int] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Full read-only supplementary record for ``(layer, n)`` across matching regimes."""
    ref_map: Dict[str, str] = refused if refused is not None else {}
    return {
        "layer": int(layer),
        "n": int(n),
        "exploratory": True,
        "not_taken": True,
        "forced_odd": _forced_odd_neuron(
            root, layer, n, vocab, model=model, weights=weights, n_layers=n_layers, refused=ref_map
        ),
        "input_census": _input_census_neuron(root, layer, n, vocab, model=model, weights=weights, refused=ref_map),
        "refused": ref_map,
    }


def _matches_query(ql: str, nid: int, name: str, tokens: Sequence[str]) -> bool:
    if not ql:
        return True
    if ql == str(nid) or ql == f"n{nid}" or ql in name.lower():
        return True
    return any(ql in tok.lower() for tok in tokens)


def list_units(
    root: Path,
    layer: int,
    regime: str,
    vocab: Sequence[str],
    notes: Dict[str, Any],
    q: str = "",
    limit: int = 600,
    face: str = "both",
    model: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    n_layers: Optional[int] = None,
    refused: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Ranked read-only listing of units for one ``(layer, regime)``."""
    if regime not in REGIME_LABELS:
        raise ValueError(f"Unknown supplementary regime: {regime!r}")
    ref_map: Dict[str, str] = refused if refused is not None else {}
    ql = q.strip().lower()
    cap = max(1, min(int(limit), 2000))

    if regime == "forced_odd":
        fo = _load_forced_odd(root, layer, model=model, weights=weights, refused=ref_map)
        if fo is None:
            return {
                "regime": regime,
                "label": REGIME_LABELS[regime],
                "exploratory": True,
                "not_taken": True,
                "layer": int(layer),
                "available": False,
                "meta": None,
                "refused": ref_map,
                "total": 0,
                "matched": 0,
                "rows": [],
            }
        arrays = fo["arrays"]
        t_col = arrays["T_col"]
        t_iso = arrays["T_iso"]
        t_min = fo["T_min"]
        rows: List[Dict[str, Any]] = []
        matched = 0
        for rank, n in enumerate(fo["named_order"], 1):
            name = _note_name(notes, layer, n)
            if not ql and len(rows) >= cap:
                matched = len(fo["named_order"])
                break
            pieces = _extract_forced_odd_pieces(arrays, n, vocab, k=5)
            toks = [p["tok"] for p in pieces["pos"]] + [p["tok"] for p in pieces["neg"]]
            if not _matches_query(ql, n, name, toks):
                continue
            matched += 1
            if len(rows) < cap:
                own_match, q_own, own_note = _own_lens_info(layer, n, arrays, fo["meta"], n_layers)
                rows.append({
                    "n": n,
                    "rank": rank,
                    "T_min": round(float(t_min[n]), 3),
                    "T_col": round(float(t_col[n]), 3),
                    "T_iso": round(float(t_iso[n]), 3),
                    "own_lens_match": own_match,
                    "q_own_lens": q_own,
                    "own_lens_note": own_note,
                    "pos": [[p["tok"], p["score"]] for p in pieces["pos"]],
                    "neg": [[p["tok"], p["score"]] for p in pieces["neg"]],
                    "name": name,
                })
        return {
            "regime": regime,
            "label": REGIME_LABELS[regime],
            "exploratory": True,
            "not_taken": True,
            "layer": int(layer),
            "available": True,
            "meta": _forced_odd_meta_card(fo),
            "refused": ref_map,
            "total": len(fo["named_order"]),
            "matched": matched,
            "rows": rows,
        }

    # input_census
    ic = _load_input_census(root, layer, model=model, weights=weights, refused=ref_map)
    if ic is None:
        return {
            "regime": regime,
            "label": REGIME_LABELS[regime],
            "exploratory": True,
            "not_taken": True,
            "layer": int(layer),
            "available": False,
            "meta": None,
            "refused": ref_map,
            "total": 0,
            "matched": 0,
            "rows": [],
        }
    arrays = ic["arrays"]
    ab_map, ab_layers = _ablation_by_layer(root, model=model, weights=weights, refused=ref_map)
    want_s = None if face == "both" else (0 if face == "+" else 1)
    rows = []
    matched = 0
    total_face = 0
    for s_idx, n in ic["named_order"]:
        if want_s is not None and s_idx != want_s:
            continue
        total_face += 1
        if not ql and len(rows) >= cap:
            continue
        fg = "+" if s_idx == 0 else "-"
        name = _note_name(notes, layer, n)
        triggers, next_pieces = _extract_input_pieces(arrays, s_idx, n, vocab)
        toks = [t["tok"] for t in triggers]
        if not _matches_query(ql, n, name, toks):
            continue
        matched += 1
        if len(rows) < cap:
            rows.append({
                "n": n,
                "face": fg,
                "rank": int(ic["rank_of"][s_idx, n]),
                "score": round(float(arrays["score"][s_idx, n]), 3),
                "bigram_rep": bool(arrays["bigram_rep"][s_idx, n]) if "bigram_rep" in arrays else False,
                "smp_hits": int(arrays["smp_hits"][s_idx, n]) if "smp_hits" in arrays else 0,
                "smp_total": int(arrays["smp_total"][s_idx, n]) if "smp_total" in arrays else 0,
                "triggers": triggers,
                "next_pieces": next_pieces,
                "name": name,
            })
    if not ql:
        matched = total_face
    return {
        "regime": regime,
        "label": REGIME_LABELS[regime],
        "exploratory": True,
        "not_taken": True,
        "layer": int(layer),
        "available": True,
        "no_passages_stored": True,
        "meta": _ic_layer_meta(root, ic, ab_map, ab_layers),
        "refused": ref_map,
        "total": total_face,
        "matched": matched,
        "rows": rows,
    }
