"""Levelling Phase 5: Owner-Only Export ("and out", `backend/level_export.py`).

Exports a Levelling `Edit` (columns on `down_proj`, rows on `gate_proj` / `up_proj`, or low-rank terms on
`down_proj`) as a llama.cpp LoRA adapter GGUF with Apache-2.0 licence & origin metadata, a companion provenance
manifest marking every term `"rebuilt"` or `"carried"`, and exact mathematical + optional live `llama-server`
re-measurement verification (`verify/arbiter.md` §4 row 32, §8, §9 decision 5).

Safety & containment invariants:
- Owner-only command line / library: NEVER imported by or exposed through `level_mcp.py`, `level_session.py`,
  `level_routes.py`, or `scientist_tools.py`.
- Exports go ONLY to an owner-controlled directory (`$THEODOLITE_EXPORT_DIR`, default
  `${XDG_DATA_HOME:-~/.local/share}/theodolite/exports`), strictly refusing the repository checkout, `atlas_data/`,
  any registration folder (`prereg/` or folder with `SHA256SUMS` + `trial.json`), and any agent session folder.
- Never overwrites an existing adapter or manifest unless `overwrite=True` (`--force` on the CLI).
- Recompiles `"rebuilt"` terms (column/row `silence` and `scale`) directly against the weights of the served base
  (a `--gguf` file when exporting for llama.cpp, or the W4A16 checkpoint), and states clearly when `"carried"` terms
  (set/add/lowrank) come from a different quantised base and must be re-measured on the served GGUF.
- Marks `ffn_gate` and `ffn_up` tensors as unverified in llama.cpp's Gemma 4 graph (only `ffn_down` and `attn_output`
  have been verified there so far).

No kept-out words appear in code, keys, manifests, or messages. Layers count from 0.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from backend import level_book as lb
from backend import weight_edits as we
from backend.census import write_json_atomic

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EXPORT_VERSION = 1
EXPORT_ENV = "THEODOLITE_EXPORT_DIR"
LARGE_RANK_WARN = 256
IMPLANT_TOP1_SHARE_BAR = 0.25  # top-1 unembedding share above which a last-block write is flagged

GGUF_TENSOR_MAP = {
    "down_proj": "blk.{}.ffn_down.weight",
    "gate_proj": "blk.{}.ffn_gate.weight",
    "up_proj": "blk.{}.ffn_up.weight",
}
VERIFIED_LLAMA_CPP_WHICH = ("down_proj",)  # ffn_down (and attn_output) read/verified in llama.cpp gemma4.cpp

PROVENANCE_NOTICE = (
    "Weights modified from Gemma 4 (Apache License 2.0) by Theodolite Levelling. "
    "Provenance is not assurance: any claim applies only to the base checkpoint measured, "
    "and an adapter served over another quantised GGUF must be re-measured on that object."
)


class ExportRefused(ValueError):
    """Plain-word refusal of an unsafe export path, overwrite, missing base, or invalid adapter."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def exports_dir() -> Path:
    """Owner-controlled export folder outside the checkout and outside tool-writable folders:
    `$THEODOLITE_EXPORT_DIR`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/exports`."""
    given = os.environ.get(EXPORT_ENV)
    if given:
        return Path(given).expanduser()
    data = os.environ.get("XDG_DATA_HOME")
    return (Path(data).expanduser() if data else Path.home() / ".local" / "share") / "theodolite" / "exports"


def refuse_unsafe_export_path(path: Path) -> Path:
    """Refuse writing an exported adapter or manifest inside the checkout, `atlas_data/`, a registration folder,
    or an agent session folder (`arbiter.md` row 32)."""
    resolved = Path(path).expanduser().resolve()
    root_res = ROOT_DIR.resolve()
    if resolved == root_res or root_res in resolved.parents:
        raise ExportRefused(
            f"Exports go to a folder outside the checkout ({root_res}); {resolved} is inside it. "
            f"Point ${EXPORT_ENV} or -o to an outside directory."
        )
    try:
        from backend.atlas_index import atlas_dir

        ad = atlas_dir().resolve()
        if resolved == ad or ad in resolved.parents:
            raise ExportRefused(
                f"Exports never go into atlas_data ({ad}), which agent tools read and write; {resolved} is inside it."
            )
    except ExportRefused:
        raise
    except Exception:
        pass

    for anc in (resolved, *resolved.parents):
        if anc.name == "prereg" or ((anc / "SHA256SUMS").is_file() and (anc / "trial.json").is_file()):
            raise ExportRefused(
                f"Exports never go into a registration folder ({anc}): registration folders hold manifests and hashes only."
            )
        if (anc / "session.json").is_file() or (anc / "transcript.jsonl").is_file():
            raise ExportRefused(
                f"Exports never go into an agent session folder ({anc})."
            )
    return resolved


# --------------------------------------------------------------------------------------------- target weight base


class TargetBase:
    """Unified CPU weight reader for recompiling LoRA adapters against a served GGUF, a safetensors checkpoint,
    or a live `nn.Module` text model."""

    def __init__(
        self,
        *,
        gguf_path: Optional[Path] = None,
        weights_dir: Optional[Path] = None,
        lm: Optional[nn.Module] = None,
        label: Optional[str] = None,
    ) -> None:
        chosen = sum(x is not None for x in (gguf_path, weights_dir, lm))
        if chosen != 1:
            raise ExportRefused("TargetBase requires exactly one of gguf_path, weights_dir, or lm.")
        self.kind: str
        self.record: str
        self.gguf_path: Optional[Path] = None
        self.gguf_sha256: Optional[str] = None
        self._gguf_reader: Any = None
        self._gguf_tensors: Dict[str, Any] = {}
        self._weight_reader: Any = None
        self._lm: Optional[nn.Module] = None

        if gguf_path is not None:
            from backend import gguf_model as gm

            p = Path(gguf_path).expanduser().resolve()
            if not p.is_file():
                raise ExportRefused(f"GGUF file not found: {p}")
            self.kind = "gguf"
            self.gguf_path = p
            self.gguf_sha256 = _sha_file(p)
            self._gguf_reader = gm.open_gguf(p)
            self._gguf_tensors = {t.name: t for t in self._gguf_reader.tensors}
            self.record = label or f"gguf {p.name} (sha256:{self.gguf_sha256[:16]})"
        elif weights_dir is not None:
            from backend.weight_reader import WeightReader

            p = Path(weights_dir).expanduser().resolve()
            self.kind = "checkpoint"
            self._weight_reader = WeightReader(p)
            self.record = label or f"checkpoint {p.name}"
        else:
            self.kind = "module"
            self._lm = lm
            self.record = label or "loaded module"

    def shape(self, layer: int, which: str) -> Tuple[int, int]:
        """Return `(out_features, in_features)` of `layers.<layer>.mlp.<which>` on this target base."""
        if self.kind == "gguf":
            from backend import gguf_model as gm

            tname = GGUF_TENSOR_MAP[which].format(int(layer))
            if tname not in self._gguf_tensors:
                raise ExportRefused(f"Target GGUF lacks tensor {tname}.")
            out_f, in_f = gm.logical_shape(self._gguf_tensors[tname])
            return int(out_f), int(in_f)
        if self.kind == "checkpoint":
            lname = we.linear_name(int(layer), which)
            return self._weight_reader.linear_shape(lname)
        assert self._lm is not None
        mod = self._lm.get_submodule(we.linear_name(int(layer), which))
        return we.linear_shape(mod)

    def _gguf_slice(self, t: Any, idx: Sequence[int], *, axis: str) -> np.ndarray:
        from backend import gguf_model as gm
        from backend.gguf_weights import GGML_TYPES, GGUFLinear, type_code

        out_f, in_f = gm.logical_shape(t)
        ix = torch.tensor([int(i) for i in idx], dtype=torch.long)
        bs = GGML_TYPES[type_code(t.tensor_type)][1] if type_code(t.tensor_type) in GGML_TYPES else 1
        if bs > 1:
            lin = GGUFLinear(
                gm._blocks(t, torch.device("cpu")), t.tensor_type, int(in_f), int(out_f),
                dtype=torch.float32, gguf_name=t.name,
            )
            got = lin.unpacked_cols(ix) if axis == "cols" else lin.unpacked_rows(ix)
            return got.float().cpu().numpy()
        dense = gm._dense(t, torch.device("cpu"), torch.float32)  # [out, in]
        if axis == "cols":
            return dense[:, ix].T.contiguous().cpu().numpy()
        return dense.index_select(0, ix).contiguous().cpu().numpy()

    def dense_cols(self, layer: int, which: str, idx: Sequence[int]) -> np.ndarray:
        """Return `[K, out]` float32 columns `W[:, idx]` from this target base."""
        if self.kind == "gguf":
            tname = GGUF_TENSOR_MAP[which].format(int(layer))
            if tname not in self._gguf_tensors:
                raise ExportRefused(f"Target GGUF lacks tensor {tname}.")
            return self._gguf_slice(self._gguf_tensors[tname], idx, axis="cols")
        if self.kind == "checkpoint":
            lname = we.linear_name(int(layer), which)
            return self._weight_reader.dense_cols(lname, idx).float().cpu().numpy()
        assert self._lm is not None
        mod = self._lm.get_submodule(we.linear_name(int(layer), which))
        return we.dense_cols(mod, torch.tensor([int(i) for i in idx], dtype=torch.long)).float().cpu().numpy()

    def dense_rows(self, layer: int, which: str, idx: Sequence[int]) -> np.ndarray:
        """Return `[K, in]` float32 rows `W[idx, :]` from this target base."""
        if self.kind == "gguf":
            tname = GGUF_TENSOR_MAP[which].format(int(layer))
            if tname not in self._gguf_tensors:
                raise ExportRefused(f"Target GGUF lacks tensor {tname}.")
            return self._gguf_slice(self._gguf_tensors[tname], idx, axis="rows")
        if self.kind == "checkpoint":
            lname = we.linear_name(int(layer), which)
            return self._weight_reader.dense_rows(lname, idx).float().cpu().numpy()
        assert self._lm is not None
        mod = self._lm.get_submodule(we.linear_name(int(layer), which))
        return we.dense_rows(mod, torch.tensor([int(i) for i in idx], dtype=torch.long)).float().cpu().numpy()


# --------------------------------------------------------------------------------------------- implant shape flag


def implant_check(
    edit: we.Edit,
    lm: Optional[nn.Module] = None,
    W_U: Optional[torch.Tensor] = None,
    *,
    dose: float = 1.0,
) -> Dict[str, Any]:
    """Check whether `edit` has the shape of a last-block single-word-piece implant (`synthesis.md` §viii):
    a column edit in the last 10% of layers whose write delta concentrates heavily onto one unembedding row.
    Purely a transparent diagnostic recorded in export metadata."""
    if lm is None or W_U is None:
        return {"checked": False, "flagged": False, "note": "unembedding matrix not supplied"}
    n_layers = len(lm.layers)
    late_cut = max(0, int(math.floor(0.9 * n_layers)))
    wu = W_U.detach().float().cpu()  # [V, D]
    wu_norms = torch.linalg.norm(wu, dim=1).clamp_min(1e-8)
    hits: List[Dict[str, Any]] = []
    d = float(dose)

    for t in edit.terms:
        if t.kind != "cols" or t.layer < late_cut:
            continue
        mod = lm.get_submodule(t.linear)
        with torch.no_grad():
            Wc = we.dense_cols(mod, torch.from_numpy(np.array(t.idx, dtype=np.int64))).float().cpu().numpy()
        g = we._asked(t, d)
        delta = (g[:, None] - 1.0) * Wc  # [K, D]
        if t.vectors is not None:
            m = t.op != 0
            delta[m] = delta[m] + d * t.vectors[m]
        w_post = we._w_post(lm, t.layer)
        dS = torch.from_numpy((delta * w_post[None, :]).astype(np.float32))  # [K, D]
        for j in range(dS.shape[0]):
            v = dS[j]
            vn = float(torch.linalg.norm(v))
            if vn <= 1e-8:
                continue
            dots = (wu @ v) / (wu_norms * vn)
            cos2 = dots * dots
            total = float(cos2.sum())
            top_idx = int(torch.argmax(dots.abs()))
            top_cos = float(dots[top_idx])
            share = float(cos2[top_idx] / total) if total > 0.0 else 0.0
            if share >= IMPLANT_TOP1_SHARE_BAR or abs(top_cos) >= 0.85:
                hits.append({
                    "layer": int(t.layer),
                    "page_layer": int(t.layer) + 1,
                    "neuron": int(t.idx[j]),
                    "top_piece_id": top_idx,
                    "top_cosine": round(top_cos, 4),
                    "top1_energy_share": round(share, 4),
                })
    return {
        "checked": True,
        "flagged": bool(hits),
        "late_layer_cutoff": late_cut,
        "hits": hits[:16],
    }


# --------------------------------------------------------------------------------------------- LoRA tensor compilation


def build_lora_tensors(
    edit: we.Edit,
    target: TargetBase,
    *,
    dose: float = 1.0,
) -> Dict[str, Any]:
    """Compile `edit` at `dose` into llama.cpp LoRA `(A [r, n_in], B [n_out, r])` float32 pairs against `target`.

    - `"rebuilt"` terms (`cols`/`rows` with `op == 0`, scale/silence) read their columns/rows directly from `target`,
      so exporting against `--gguf FILE` recompiles the silence/scale against that GGUF's own weights.
    - `"carried"` terms (`set`/`add`/`lowrank`) carry their deltas into `target` and record explicit warnings whenever
      `target` is a GGUF (different quantised object than the W4A16 checkpoint).
    - `gate_proj` and `up_proj` row edits are recorded in `unverified_tensors` until verified in llama.cpp's Gemma 4 graph.
    """
    d = float(dose)
    if not math.isfinite(d) or d == 0.0:
        raise ExportRefused(f"Export dose must be finite and non-zero; got {dose!r}.")

    # Group contributions per GGUF tensor name so multiple terms on one tensor concatenate cleanly
    parts_a: Dict[str, List[np.ndarray]] = {}
    parts_b: Dict[str, List[np.ndarray]] = {}
    warnings: List[str] = []
    unverified: List[str] = []
    recompiled_all = True

    for t in edit.terms:
        tname = GGUF_TENSOR_MAP[t.which].format(int(t.layer))
        if t.which not in VERIFIED_LLAMA_CPP_WHICH and tname not in unverified:
            unverified.append(tname)
        n_out, n_in = target.shape(t.layer, t.which)

        if t.kind == "cols":
            idx = [int(n) for n in t.idx]
            if max(idx) >= n_in:
                raise ExportRefused(f"{tname} has {n_in} inputs on {target.record}; edit touches column {max(idx)}.")
            W_cols = target.dense_cols(t.layer, t.which, idx)  # [K, n_out]
            g = we._asked(t, d)  # [K]
            delta = (g[:, None] - 1.0) * W_cols  # [K, n_out]
            if t.vectors is not None:
                recompiled_all = False
                if t.vectors.shape[1] != n_out:
                    raise ExportRefused(f"{tname} has out={n_out} on {target.record}, edit vectors have {t.vectors.shape[1]}.")
                mask_vec = t.op != 0
                delta[mask_vec] = delta[mask_vec] + d * t.vectors[mask_vec]
                if target.kind == "gguf":
                    warnings.append(
                        f"{tname}: carried column vectors (set/add) computed on {edit.base.get('model')} "
                        f"are applied onto {target.record}; re-measure on that GGUF."
                    )
            keep = np.max(np.abs(delta), axis=1) > 0.0
            if not keep.any():
                continue
            kept_idx = [idx[i] for i in range(len(idx)) if keep[i]]
            B_part = np.ascontiguousarray(delta[keep].T, dtype=np.float32)  # [n_out, r]
            A_part = np.zeros((len(kept_idx), n_in), dtype=np.float32)
            A_part[np.arange(len(kept_idx)), kept_idx] = 1.0
            parts_a.setdefault(tname, []).append(A_part)
            parts_b.setdefault(tname, []).append(B_part)

        elif t.kind == "rows":
            idx = [int(n) for n in t.idx]
            if max(idx) >= n_out:
                raise ExportRefused(f"{tname} has {n_out} outputs on {target.record}; edit touches row {max(idx)}.")
            W_rows = target.dense_rows(t.layer, t.which, idx)  # [K, n_in]
            g = we._asked(t, d)
            delta = (g[:, None] - 1.0) * W_rows  # [K, n_in]
            if t.vectors is not None:
                recompiled_all = False
                if t.vectors.shape[1] != n_in:
                    raise ExportRefused(f"{tname} has in={n_in} on {target.record}, edit vectors have {t.vectors.shape[1]}.")
                mask_vec = t.op != 0
                delta[mask_vec] = delta[mask_vec] + d * t.vectors[mask_vec]
                if target.kind == "gguf":
                    warnings.append(
                        f"{tname}: carried row vectors (set/add) computed on {edit.base.get('model')} "
                        f"are applied onto {target.record}; re-measure on that GGUF."
                    )
            keep = np.max(np.abs(delta), axis=1) > 0.0
            if not keep.any():
                continue
            kept_idx = [idx[i] for i in range(len(idx)) if keep[i]]
            # For row n with delta row dW_n [1, n_in]: B[:, k] = e_n, A[k, :] = dW_n -> (B @ A)[n, :] = dW_n
            A_part = np.ascontiguousarray(delta[keep], dtype=np.float32)  # [r, n_in]
            B_part = np.zeros((n_out, len(kept_idx)), dtype=np.float32)
            B_part[kept_idx, np.arange(len(kept_idx))] = 1.0
            parts_a.setdefault(tname, []).append(A_part)
            parts_b.setdefault(tname, []).append(B_part)

        else:  # lowrank on down_proj
            recompiled_all = False
            if t.A.shape[1] != n_in or t.B.shape[0] != n_out:
                raise ExportRefused(
                    f"{tname} is [{n_out}, {n_in}] on {target.record}, low-rank edit has A={list(t.A.shape)}, B={list(t.B.shape)}."
                )
            A_part = np.ascontiguousarray(t.A, dtype=np.float32)
            B_part = np.ascontiguousarray(d * t.B, dtype=np.float32)
            parts_a.setdefault(tname, []).append(A_part)
            parts_b.setdefault(tname, []).append(B_part)
            if target.kind == "gguf":
                warnings.append(
                    f"{tname}: carried low-rank term (rank {t.A.shape[0]}) computed on {edit.base.get('model')} "
                    f"is applied onto {target.record}; re-measure on that GGUF."
                )

    tensors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ranks: Dict[str, int] = {}
    for tname in sorted(parts_a):
        A = np.ascontiguousarray(np.concatenate(parts_a[tname], axis=0), dtype=np.float32)
        B = np.ascontiguousarray(np.concatenate(parts_b[tname], axis=1), dtype=np.float32)
        tensors[tname] = (A, B)
        r = int(A.shape[0])
        ranks[tname] = r
        if r > LARGE_RANK_WARN:
            mb = (A.nbytes + B.nbytes) / (2**20)
            warnings.append(f"{tname}: rank {r} exceeds {LARGE_RANK_WARN} (about {mb:.1f} MiB in float32).")

    if not tensors:
        raise ExportRefused("Nothing to export: the edit produces zero delta at this dose.")
    if unverified:
        warnings.append(
            f"Unverified llama.cpp Gemma 4 adapter tensor(s): {', '.join(unverified)} "
            "(only blk.*.ffn_down.weight and blk.*.attn_output.weight have been verified in llama.cpp so far)."
        )

    return {
        "tensors": tensors,
        "ranks": ranks,
        "recompiled_all": recompiled_all,
        "unverified_tensors": sorted(unverified),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------------------------- GGUF + manifest writer


def export_lora_adapter(
    edit: we.Edit,
    target: TargetBase,
    *,
    out_path: Optional[Path] = None,
    dose: float = 1.0,
    overwrite: bool = False,
    book_seq: Optional[int] = None,
    book_hash: Optional[str] = None,
    trial_result: Optional[Mapping[str, Any]] = None,
    prior_attempts: Optional[int] = None,
    saved_edit_dir: Optional[Path] = None,
    lm_for_implant: Optional[nn.Module] = None,
    W_U_for_implant: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Write a llama.cpp LoRA `.gguf` adapter and companion `.manifest.json` into an owner-controlled export folder.

    Carries Apache-2.0 licence & origin keys (`general.license`, `general.license.link`, `general.base_model.*`,
    `theodolite.derived_from`) via `scripts/export_lora.write_lora`, plus full Levelling provenance keys.
    Refuses unsafe export paths and refuses overwriting unless `overwrite=True`.
    """
    import export_control_vector as ecv
    import export_lora as el

    if out_path is None:
        base_dir = exports_dir()
        fname = f"theodolite-levelling-{edit.short()}-d{float(dose):g}-derived-from-gemma.gguf"
        dst = refuse_unsafe_export_path(base_dir / fname)
    else:
        dst = refuse_unsafe_export_path(Path(out_path))
        if dst.suffix.lower() != ".gguf":
            dst = dst.with_suffix(".gguf")

    manifest_path = dst.with_name(dst.stem + ".manifest.json")
    if not overwrite:
        if dst.exists():
            raise ExportRefused(f"Export file already exists: {dst} (pass --force / overwrite=True to replace).")
        if manifest_path.exists():
            raise ExportRefused(f"Export manifest already exists: {manifest_path} (pass --force / overwrite=True to replace).")

    compiled = build_lora_tensors(edit, target, dose=dose)
    prov = we.edit_provenance(edit, saved_dir=saved_edit_dir)
    imp = implant_check(edit, lm_for_implant, W_U_for_implant, dose=dose)

    trial_verdict = trial_result.get("verdict") if isinstance(trial_result, Mapping) else None
    trial_sha = trial_result.get("trial_sha256") if isinstance(trial_result, Mapping) else None
    if book_seq is None and isinstance(trial_result, Mapping):
        book_seq = (trial_result.get("book") or {}).get("verdict_seq")
    if book_hash is None and isinstance(trial_result, Mapping):
        book_hash = (trial_result.get("book") or {}).get("verdict_hash")

    adapter_name = f"theodolite-levelling-{edit.short()}-derived-from-gemma"
    record: Dict[str, Any] = {
        "name": adapter_name,
        "description": (
            f"Modified from Gemma 4 (Apache-2.0) by Theodolite Levelling: edit {edit.short()} at dose {float(dose):g} "
            f"({prov['provenance']}), compiled against {target.record}."
        ),
        "model": str(edit.base.get("model") or el.MODEL_ID),
        "weights": target.record,
        "entries": [f"{t.linear} ({t.kind})" for t in edit.terms],
        "start_mark": el.START_MARK,
        "scale": el.SCALE,
        "llama_cpp": el.LLAMA_CPP_READ,
        "caveat": PROVENANCE_NOTICE,
        "made_by": f"backend/level_export.py at {ecv.code_commit()}",
        "made_at": _utc_now(),
        "levelling.version": str(EXPORT_VERSION),
        "levelling.edit_id": edit.id,
        "levelling.dose": f"{float(dose):g}",
        "levelling.base_weights_id": str(edit.base.get("weights_id") or "none"),
        "levelling.served_base": target.record,
        "levelling.served_gguf_sha256": str(target.gguf_sha256 or "none"),
        "levelling.provenance": prov["provenance"],
        "levelling.recompiled_from_recipe": "true" if compiled["recompiled_all"] else "false",
        "levelling.unverified_tensors": json.dumps(compiled["unverified_tensors"]),
        "levelling.book_seq": str(book_seq) if book_seq is not None else "none",
        "levelling.book_hash": str(book_hash or "none"),
        "levelling.trial_sha256": str(trial_sha or "none"),
        "levelling.trial_verdict": str(trial_verdict or "unverified_practice_or_claim"),
        "levelling.prior_attempts": str(prior_attempts) if prior_attempts is not None else "unknown",
        "levelling.implant_flagged": "true" if imp.get("flagged") else "false",
        "levelling.warnings": json.dumps(compiled["warnings"]),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    el.write_lora(dst, compiled["tensors"], record)
    gguf_sha = _sha_file(dst)
    gguf_bytes = dst.stat().st_size

    companion = {
        "v": EXPORT_VERSION,
        "exported_at": _utc_now(),
        "adapter_name": adapter_name,
        "gguf_file": dst.name,
        "gguf_sha256": gguf_sha,
        "gguf_bytes": gguf_bytes,
        "dose": float(dose),
        "served_base": {
            "kind": target.kind,
            "record": target.record,
            "gguf_sha256": target.gguf_sha256,
        },
        "recompiled_from_recipe": bool(compiled["recompiled_all"]),
        "ranks": compiled["ranks"],
        "unverified_llama_cpp_tensors": compiled["unverified_tensors"],
        "warnings": compiled["warnings"],
        "implant_check": imp,
        "edit_provenance": prov,
        "ledger": {
            "book_seq": book_seq,
            "book_hash": book_hash,
            "trial_sha256": trial_sha,
            "trial_verdict": trial_verdict,
            "prior_attempts": prior_attempts,
        },
        "license": {
            "name": "Apache-2.0",
            "url": "https://ai.google.dev/gemma/docs/gemma_4_license",
            "derived_from": "Gemma 4 (modified weights; derived from Gemma, not Gemma itself)",
            "notice": PROVENANCE_NOTICE,
        },
        "llama_cpp_usage": {
            "start": f"llama-server -m <MODEL.gguf> --lora {dst} --lora-init-without-apply",
            "request_example": '{"prompt": "...", "lora": [{"id": 0, "scale": 1}]}',
        },
    }
    write_json_atomic(manifest_path, companion, indent=2)
    return {
        "gguf_path": str(dst),
        "manifest_path": str(manifest_path),
        "gguf_sha256": gguf_sha,
        "gguf_bytes": gguf_bytes,
        "ranks": compiled["ranks"],
        "recompiled_from_recipe": bool(compiled["recompiled_all"]),
        "unverified_llama_cpp_tensors": compiled["unverified_tensors"],
        "warnings": compiled["warnings"],
        "implant_check": imp,
        "manifest": companion,
    }


# --------------------------------------------------------------------------------------------- verification & re-measure


REQUIRED_GGUF_KEYS = (
    "general.type",
    "general.architecture",
    "adapter.type",
    "general.license",
    "general.license.link",
    "general.base_model.count",
    "theodolite.derived_from",
    "theodolite.levelling.edit_id",
)


def verify_exported_lora(
    gguf_path: Path,
    *,
    edit: Optional[we.Edit] = None,
    target: Optional[TargetBase] = None,
    dose: Optional[float] = None,
    tol: float = 1e-4,
) -> Dict[str, Any]:
    """Verify an exported LoRA `.gguf`:
    1. All required llama.cpp adapter keys, Apache-2.0 licence & origin keys, and Levelling provenance keys.
    2. Companion `.manifest.json` (when present) matches the `.gguf` sha256 byte for byte.
    3. When `edit` and `target` are supplied, checks mathematical equivalence of `(W + B @ A) @ x` against the
       edit's exact arithmetic on `target` (proving recompilation matches the served base)."""
    import export_lora as el

    p = Path(gguf_path).expanduser().resolve()
    if not p.is_file():
        raise ExportRefused(f"Exported GGUF not found: {p}")
    tensors, meta = el.read_lora(p)
    checks: List[Dict[str, Any]] = []

    def _rec(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    missing = [k for k in REQUIRED_GGUF_KEYS if k not in meta]
    _rec("required_keys_present", not missing, {"missing": missing})
    _rec(
        "adapter_headers_valid",
        meta.get("general.architecture") == "gemma4"
        and meta.get("adapter.type") == "lora"
        and "adapter.lora.alpha" not in meta,
        {"arch": meta.get("general.architecture"), "adapter_type": meta.get("adapter.type")},
    )
    _rec(
        "license_and_origin_present",
        meta.get("general.license") == "apache-2.0"
        and "gemma" in str(meta.get("theodolite.derived_from") or "").lower()
        and "gemma" in str(meta.get("general.name") or "").lower(),
        {"license": meta.get("general.license"), "name": meta.get("general.name")},
    )

    man_path = p.with_name(p.stem + ".manifest.json")
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        got_sha = _sha_file(p)
        _rec(
            "companion_manifest_sha256",
            man.get("gguf_sha256") == got_sha and man.get("edit_provenance", {}).get("edit_id") == meta.get("theodolite.levelling.edit_id"),
            {"got": got_sha[:16], "want": str(man.get("gguf_sha256") or "")[:16]},
        )

    max_err: Optional[float] = None
    if edit is not None and target is not None:
        d = float(dose if dose is not None else float(meta.get("theodolite.levelling.dose", "1.0")))
        expected = build_lora_tensors(edit, target, dose=d)["tensors"]
        _rec("tensor_names_match", set(tensors) == set(expected), {"got": sorted(tensors), "want": sorted(expected)})
        worst = 0.0
        rng = np.random.default_rng(2026)
        for tname in sorted(set(tensors) & set(expected)):
            Ag, Bg = tensors[tname]
            Aw, Bw = expected[tname]
            x = rng.standard_normal((4, Aw.shape[1])).astype(np.float32)
            yg = (x @ Ag.T) @ Bg.T
            yw = (x @ Aw.T) @ Bw.T
            err = float(np.max(np.abs(yg - yw)))
            worst = max(worst, err)
        max_err = worst
        _rec("delta_action_matches_target", worst <= tol, {"max_abs_err": worst, "tol": tol})

    ok_all = all(c["ok"] for c in checks)
    return {
        "ok": ok_all,
        "gguf_path": str(p),
        "tensors": {k: {"A_shape": list(v[0].shape), "B_shape": list(v[1].shape)} for k, v in tensors.items()},
        "max_abs_err": max_err,
        "checks": checks,
        "meta_summary": {k: v for k, v in meta.items() if k.startswith(("general.", "adapter.", "theodolite."))},
    }


def remeasure_with_llama_server(
    llama_server_bin: Path,
    model_gguf: Path,
    adapter_gguf: Path,
    prompt: str,
    *,
    scale: float = 1.0,
    n_predict: int = 16,
    port: int = 18089,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Optional live re-measurement through `llama-server --lora-init-without-apply`: compares scale 0 (base)
    against `scale` on `prompt` over `model_gguf` and returns both outputs + adapter table from `/lora-adapters`."""
    exe = Path(llama_server_bin)
    if not exe.is_file():
        raise ExportRefused(f"llama-server binary not found: {exe}")
    cmd = [
        str(exe), "-m", str(model_gguf), "--lora", str(adapter_gguf), "--lora-init-without-apply",
        "--host", "127.0.0.1", "--port", str(int(port)), "-ngl", "0", "-c", "512",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base_url = f"http://127.0.0.1:{int(port)}"
    try:
        deadline = time.monotonic() + float(timeout_s)
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")[-600:]
                raise ExportRefused(f"llama-server exited early ({proc.returncode}): {err}")
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.25)
        if not ready:
            raise ExportRefused(f"llama-server did not become ready within {timeout_s:g} s.")
        with urllib.request.urlopen(f"{base_url}/lora-adapters", timeout=10.0) as r:
            adapters = json.loads(r.read().decode("utf-8"))

        def _complete(sc: float) -> Dict[str, Any]:
            body = json.dumps({
                "prompt": prompt,
                "n_predict": int(n_predict),
                "temperature": 0.0,
                "lora": [{"id": 0, "scale": float(sc)}],
            }).encode("utf-8")
            req = urllib.request.Request(f"{base_url}/completion", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30.0) as r:
                return json.loads(r.read().decode("utf-8"))

        out_base = _complete(0.0)
        out_edit = _complete(float(scale))
        return {
            "ok": True,
            "adapters": adapters,
            "base_content": out_base.get("content", ""),
            "edited_content": out_edit.get("content", ""),
            "changed": out_base.get("content", "") != out_edit.get("content", ""),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
