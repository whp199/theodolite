"""No-card CPU weights reader (`backend/weight_reader.py`): read W4A16 or dense safetensors checkpoints straight from
disk on the CPU without loading the 31B onto the GPU or initialising CUDA.

Designed for two roles:
1. Standalone offline weight inspection (`scripts/levelling.py weights ...`), unpacking individual `down_proj` columns,
   `gate_proj` / `up_proj` rows, norm weights, quiet masks, stream-side column lengths, top writers along a direction,
   and `tensor_pin` hashes in tens of megabytes of RAM.
2. The single plain read-only weights handle (`W`) exposed inside the off-card `bwrap` sandbox (`backend/level_sandbox.py`),
   where the checkpoint files are mounted read-only under `/weights`.

Arithmetic matches `backend/engine_runtime.py` (`dequantize_w4a16_full`, `_rows`, `_cols`) and `backend/weight_edits.py`
(`tensor_pin`, `quiet_mask`, `down_stream_norms`, `writers`) bit for bit on W4A16 and dense checkpoints.

Layers count from 0 in code and files; human-facing strings give both ("layer 20 (page layer 21)").
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_torch

QUIET_WPOST_RATIO = 0.05
MAX_QUIET_SHARE = 0.25
DEFAULT_CHUNK_WORD_COLS = 256  # 256 int32 word-cols = 2,048 neurons (~22 MiB bf16 per chunk)
QUIET_DIMS_PATH = Path(__file__).resolve().parent.parent / "experiments" / "results" / "V6_quiet_dims_pers181.npy"
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_NIBBLE_SHIFTS = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32)
_PERS181: Optional[np.ndarray] = None


class WeightReadError(ValueError):
    """A weight lookup or checkpoint layout error, in plain words."""


def _raw_bytes(t: torch.Tensor) -> memoryview:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    a = np.ascontiguousarray(t.numpy())
    return memoryview(a.tobytes() if a.size == 0 else a).cast("B")


def _dequant_w4a16_full_cpu(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Bit-exact CPU W4A16 group-32 symmetric unpacker matching `engine_runtime.dequantize_w4a16_full`."""
    rows = int(packed.shape[0])
    cols = int(packed.shape[1]) * 8
    w_q = (((packed.unsqueeze(-1) >> _NIBBLE_SHIFTS) & 0xF) - 8).to(dtype=scale.dtype)
    return (w_q.view(rows, -1, 32) * scale.unsqueeze(-1)).view(rows, cols)


def _dequant_w4a16_cols_from_tensors(
    packed: torch.Tensor, scale: torch.Tensor, col_indices: torch.Tensor
) -> torch.Tensor:
    """Bit-exact CPU W4A16 column-slice unpacker matching `engine_runtime.dequantize_w4a16_cols`."""
    idx = col_indices.to(dtype=torch.long)
    word_col = idx // 8
    bit_shift = ((idx % 8) * 4).to(dtype=torch.int32)
    scale_col = idx // 32
    words = packed.index_select(1, word_col)
    scales = scale.index_select(1, scale_col)
    w_q = (((words >> bit_shift.unsqueeze(0)) & 0xF) - 8).to(dtype=scale.dtype)
    return (w_q * scales).T.contiguous()


def _pers181_mask(quiet_dims_path: Optional[Path] = None) -> Optional[np.ndarray]:
    """Read-only bool [5376] mask of the 31B's 181 persistent quiet dimensions when available on disk."""
    global _PERS181
    p = Path(quiet_dims_path) if quiet_dims_path is not None else QUIET_DIMS_PATH
    if p == QUIET_DIMS_PATH and _PERS181 is not None:
        return _PERS181
    if p.is_file():
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=bool).reshape(-1)
        arr.setflags(write=False)
        if p == QUIET_DIMS_PATH:
            _PERS181 = arr
        return arr
    return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def save_array(path: Union[str, Path], arr: Any) -> Dict[str, Any]:
    """Write `arr` as a C-contiguous float32 `.npy` file (no pickle, finite check) and return recipe metadata
    (`path`, `shape`, `dtype`, `sha256`)."""
    p = Path(path)
    if isinstance(arr, torch.Tensor):
        a = arr.detach().cpu().float().numpy()
    else:
        a = np.asarray(arr, dtype=np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise WeightReadError("Cannot save an array containing NaN or infinity.")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, a, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return {
        "path": str(p),
        "shape": [int(s) for s in a.shape],
        "dtype": "float32",
        "sha256": _file_sha256(p),
    }


def _safetensors_header(fp: Path) -> Dict[str, Any]:
    """Read a safetensors file's JSON header directly (8-byte LE length + UTF-8 JSON) without mmapping the weights."""
    with open(fp, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise WeightReadError(f"{fp} is truncated (no safetensors header length).")
        n = int.from_bytes(raw_len, "little")
        if n <= 0 or n > 100 * (2**20):
            raise WeightReadError(f"{fp} has an invalid safetensors header length ({n}).")
        header = json.loads(f.read(n).decode("utf-8"))
    if not isinstance(header, dict):
        raise WeightReadError(f"{fp} safetensors header is not a JSON object.")
    return header


def _discover_shards(root: Path) -> Tuple[Dict[str, Path], List[Path], Dict[str, Dict[str, Any]]]:
    """Map every safetensors key in `root` (a folder or a single `.safetensors` file) to its file path and header
    metadata (`shape`, `dtype`) without mmapping any tensor bytes."""
    root = Path(root)
    if root.is_file() and root.suffix == ".safetensors":
        hdr = _safetensors_header(root)
        meta = {k: v for k, v in hdr.items() if k != "__metadata__" and isinstance(v, dict)}
        return {k: root for k in meta}, [root], meta
    if not root.is_dir():
        raise WeightReadError(f"Weight path {root} does not exist or is not a folder/safetensors file.")
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        wmap = idx.get("weight_map")
        if not isinstance(wmap, dict) or not wmap:
            raise WeightReadError(f"{index_path} has no weight_map.")
        key_to_file: Dict[str, Path] = {}
        files_seen: Dict[str, Path] = {}
        meta_all: Dict[str, Dict[str, Any]] = {}
        for k, rel in wmap.items():
            fp = (root / rel).resolve()
            if not fp.is_file():
                raise WeightReadError(f"Shard {rel!r} named in {index_path.name} is missing under {root}.")
            key_to_file[k] = fp
            files_seen[str(fp)] = fp
        for fp in sorted(files_seen.values()):
            hdr = _safetensors_header(fp)
            for k, v in hdr.items():
                if k != "__metadata__" and isinstance(v, dict):
                    meta_all[k] = v
        return key_to_file, sorted(files_seen.values()), meta_all
    st_files = sorted(p for p in root.glob("*.safetensors") if p.is_file())
    if not st_files:
        raise WeightReadError(f"No .safetensors files found in {root}.")
    key_to_file = {}
    meta_all = {}
    for fp in st_files:
        hdr = _safetensors_header(fp)
        for k, v in hdr.items():
            if k != "__metadata__" and isinstance(v, dict):
                key_to_file[k] = fp
                meta_all[k] = v
    return key_to_file, st_files, meta_all


def _detect_prefix(keys: Mapping[str, Path]) -> str:
    """Detect the language-model prefix inside a checkpoint's safetensors keys."""
    for candidate in ("model.language_model.", "language_model.", "model.", ""):
        if (
            f"{candidate}layers.0.mlp.down_proj.weight_packed" in keys
            or f"{candidate}layers.0.mlp.down_proj.weight" in keys
        ):
            return candidate
    raise WeightReadError(
        "Could not locate layers.0.mlp.down_proj (packed or dense) under any recognised text-model prefix."
    )


class WeightReader:
    """Read-only lazy CPU reader for W4A16 or dense safetensors checkpoints (no GPU, no model load).

    All linear paths follow `weight_edits` conventions relative to the text model: e.g.
    `"layers.20.mlp.down_proj"`, `"layers.20.mlp.gate_proj"`, `"layers.20.mlp.up_proj"`,
    `"layers.20.self_attn.q_proj"`, etc.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        model_id: Optional[str] = None,
        weights_id: Optional[str] = None,
        quiet_dims_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(path)
        self._key_to_file, self._files, self._meta = _discover_shards(self.root)
        self.prefix = _detect_prefix(self._key_to_file)
        self._handles: Dict[Path, Any] = {}
        self._pins_cache: Dict[str, str] = {}
        self._stream_norms_cache: Dict[int, np.ndarray] = {}
        self._w_post_cache: Dict[int, np.ndarray] = {}
        self._quiet_dims_path = quiet_dims_path

        cfg_path = (self.root if self.root.is_dir() else self.root.parent) / "config.json"
        self.config: Dict[str, Any] = {}
        if cfg_path.is_file():
            try:
                self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                self.config = {}

        self.model_id = model_id or str(self.config.get("_name_or_path") or self.config.get("model_id") or self.root.name)
        if weights_id is None and self.root.is_dir():
            try:
                from backend.runtime_factory import checkpoint_files_id

                weights_id = checkpoint_files_id(self.root)
            except Exception:
                weights_id = None
        self.weights_id = weights_id

        # Discover layers and 2D linears from header metadata alone (zero mmap, zero tensor data read!)
        layers_seen = set()
        linears: Dict[str, str] = {}
        p_len = len(self.prefix)
        for k in self._key_to_file:
            if not k.startswith(self.prefix):
                continue
            rel = k[p_len:]
            m = _LAYER_RE.match(rel)
            if m:
                layers_seen.add(int(m.group(1)))
            if rel.endswith(".weight_packed"):
                lin = rel[: -len(".weight_packed")]
                if f"{self.prefix}{lin}.weight_scale" in self._key_to_file:
                    linears[lin] = "w4a16"
            elif rel.endswith(".weight") and (".mlp." in rel or ".self_attn." in rel):
                lin = rel[: -len(".weight")]
                shape = (self._meta.get(k) or {}).get("shape")
                if lin not in linears and isinstance(shape, list) and len(shape) == 2:
                    linears[lin] = "dense"
        self.n_layers = (max(layers_seen) + 1) if layers_seen else 0
        self._linears = linears
        if self.n_layers == 0:
            raise WeightReadError(f"No transformer layers found under {self.root}.")

        d_out, d_in = self.linear_shape("layers.0.mlp.down_proj")
        self.d_model = int(d_out)
        self.d_ff = int(d_in)
        emb_key = f"{self.prefix}embed_tokens.weight"
        emb_sh = (self._meta.get(emb_key) or {}).get("shape")
        self.vocab_size = int(emb_sh[0]) if isinstance(emb_sh, list) and emb_sh else None

    @classmethod
    def from_default(cls, repo_id: str = "google/gemma-4-31B-it-qat-w4a16-ct") -> "WeightReader":
        """Open the locally cached default W4A16 checkpoint snapshot without loading the model."""
        from backend.runtime_factory import checkpoint_files_id, checkpoint_snapshot

        snap = checkpoint_snapshot(repo_id)
        if snap is None:
            raise WeightReadError(f"Checkpoint {repo_id!r} is not present in the local Hugging Face cache.")
        return cls(snap, model_id=repo_id, weights_id=checkpoint_files_id(snap))

    def close(self) -> None:
        for h in self._handles.values():
            try:
                close_fn = getattr(h, "__exit__", None)
                if callable(close_fn):
                    close_fn(None, None, None)
            except Exception:
                pass
        self._handles.clear()

    def __enter__(self) -> "WeightReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _open(self, fp: Path) -> Any:
        h = self._handles.get(fp)
        if h is None:
            h = safe_open(str(fp), framework="pt", device="cpu")
            self._handles[fp] = h
        return h

    def _full_key(self, rel: str) -> str:
        k = f"{self.prefix}{rel}"
        if k not in self._key_to_file:
            raise WeightReadError(f"Tensor {rel!r} (key {k!r}) is not present in {self.root}.")
        return k

    def _slice(self, full_key: str) -> Any:
        fp = self._key_to_file.get(full_key)
        if fp is None:
            raise WeightReadError(f"Tensor key {full_key!r} is not present in {self.root}.")
        return self._open(fp).get_slice(full_key)

    def _tensor(self, full_key: str) -> torch.Tensor:
        fp = self._key_to_file.get(full_key)
        if fp is None:
            raise WeightReadError(f"Tensor key {full_key!r} is not present in {self.root}.")
        return self._open(fp).get_tensor(full_key)

    def _check_layer(self, layer: int) -> int:
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
            raise WeightReadError(f"Layer must be an integer; got {layer!r}.")
        L = int(layer)
        if not 0 <= L < self.n_layers:
            raise WeightReadError(f"Layer {L} does not exist: layers count from 0 to {self.n_layers - 1} here "
                                  f"(page layers 1 to {self.n_layers}).")
        return L

    def _linear_kind(self, linear: str) -> str:
        kind = self._linears.get(linear)
        if kind is None:
            raise WeightReadError(f"Linear {linear!r} is not present in {self.root}.")
        return kind

    def has_linear(self, linear: str) -> bool:
        return linear in self._linears

    def linear_names(self, *, layer: Optional[int] = None) -> List[str]:
        if layer is None:
            return sorted(self._linears)
        L = self._check_layer(layer)
        pref = f"layers.{L}."
        return sorted(k for k in self._linears if k.startswith(pref))

    def linear_kind(self, linear: str) -> str:
        """'w4a16' or 'dense'."""
        return self._linear_kind(linear)

    def linear_shape(self, linear: str) -> Tuple[int, int]:
        """Return `(out_features, in_features)` from header metadata without reading or mmapping tensor data."""
        kind = self._linear_kind(linear)
        key = self._full_key(f"{linear}.weight_packed" if kind == "w4a16" else f"{linear}.weight")
        meta = self._meta.get(key)
        sh = meta["shape"] if isinstance(meta, dict) and "shape" in meta else self._slice(key).get_shape()
        return (int(sh[0]), int(sh[1]) * 8) if kind == "w4a16" else (int(sh[0]), int(sh[1]))

    def info(self) -> Dict[str, Any]:
        """Metadata summary of this checkpoint (zero tensor data read)."""
        v_layers = [L for L in range(self.n_layers) if f"layers.{L}.self_attn.v_proj" in self._linears]
        no_v_layers = [L for L in range(self.n_layers) if L not in v_layers]
        kinds: Dict[str, int] = {}
        for k in self._linears.values():
            kinds[k] = kinds.get(k, 0) + 1
        return {
            "model_id": self.model_id,
            "weights_id": self.weights_id,
            "root": str(self.root),
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "vocab_size": self.vocab_size,
            "n_linears": len(self._linears),
            "linear_kinds": kinds,
            "no_v_proj_layers": no_v_layers,
            "shards": len(self._files),
        }

    def tensor_pin(self, linear: str) -> str:
        """Exact sha256 tensor pin matching `backend/weight_edits.py:tensor_pin`."""
        got = self._pins_cache.get(linear)
        if got is not None:
            return got
        kind = self._linear_kind(linear)
        h = hashlib.sha256()
        if kind == "w4a16":
            p = self._tensor(self._full_key(f"{linear}.weight_packed"))
            s = self._tensor(self._full_key(f"{linear}.weight_scale"))
            p0, p1, s0, s1 = (int(n) for n in (*p.shape, *s.shape))
            h.update(f"w4a16:{p0}x{p1}:{s0}x{s1}:{s.dtype}:".encode("ascii"))
            h.update(_raw_bytes(p.to(torch.int32)))
            h.update(_raw_bytes(s))
        else:
            w = self._tensor(self._full_key(f"{linear}.weight"))
            h.update(f"dense:{int(w.shape[0])}x{int(w.shape[1])}:{w.dtype}:".encode("ascii"))
            h.update(_raw_bytes(w))
        pin = h.hexdigest()
        self._pins_cache[linear] = pin
        return pin

    @torch.no_grad()
    def dense(self, linear: str) -> torch.Tensor:
        """Unpack one linear whole on CPU (`[out, in]` in its own scale/weight dtype). Not cached."""
        kind = self._linear_kind(linear)
        if kind == "w4a16":
            p = self._tensor(self._full_key(f"{linear}.weight_packed"))
            s = self._tensor(self._full_key(f"{linear}.weight_scale"))
            return _dequant_w4a16_full_cpu(p, s)
        return self._tensor(self._full_key(f"{linear}.weight")).clone()

    @torch.no_grad()
    def dense_rows(self, linear: str, row_indices: Sequence[int]) -> torch.Tensor:
        """Unpack only the requested rows (`[K, in]`) directly from disk slices, bit-identical to
        `engine_runtime.dequantize_w4a16_rows`."""
        idx = np.asarray(list(row_indices), dtype=np.int64).reshape(-1)
        out_f, in_f = self.linear_shape(linear)
        if idx.size == 0:
            kind = self._linear_kind(linear)
            dt = self._slice(self._full_key(f"{linear}.weight_scale" if kind == "w4a16" else f"{linear}.weight")).get_dtype()
            torch_dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}.get(dt, torch.float32)
            return torch.empty((0, in_f), dtype=torch_dt)
        if int(idx.min()) < 0 or int(idx.max()) >= out_f:
            raise WeightReadError(f"Row index out of range for {linear} (0 to {out_f - 1}).")
        kind = self._linear_kind(linear)
        if kind == "w4a16":
            p_sl = self._slice(self._full_key(f"{linear}.weight_packed"))
            s_sl = self._slice(self._full_key(f"{linear}.weight_scale"))
            # Read unique rows once from mmap
            uniq, inv = np.unique(idx, return_inverse=True)
            p_rows = torch.stack([p_sl[int(r), :] for r in uniq], dim=0)
            s_rows = torch.stack([s_sl[int(r), :] for r in uniq], dim=0)
            deq = _dequant_w4a16_full_cpu(p_rows, s_rows)
            return deq.index_select(0, torch.from_numpy(inv.astype(np.int64))).contiguous()
        w_sl = self._slice(self._full_key(f"{linear}.weight"))
        uniq, inv = np.unique(idx, return_inverse=True)
        rows_t = torch.stack([w_sl[int(r), :] for r in uniq], dim=0)
        return rows_t.index_select(0, torch.from_numpy(inv.astype(np.int64))).contiguous()

    @torch.no_grad()
    def dense_cols(self, linear: str, col_indices: Sequence[int]) -> torch.Tensor:
        """Unpack only the requested columns (`[K, out]`) directly from disk slices, bit-identical to
        `engine_runtime.dequantize_w4a16_cols`."""
        idx = np.asarray(list(col_indices), dtype=np.int64).reshape(-1)
        out_f, in_f = self.linear_shape(linear)
        if idx.size == 0:
            kind = self._linear_kind(linear)
            dt = self._slice(self._full_key(f"{linear}.weight_scale" if kind == "w4a16" else f"{linear}.weight")).get_dtype()
            torch_dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}.get(dt, torch.float32)
            return torch.empty((0, out_f), dtype=torch_dt)
        if int(idx.min()) < 0 or int(idx.max()) >= in_f:
            raise WeightReadError(f"Column index out of range for {linear} (0 to {in_f - 1}).")
        kind = self._linear_kind(linear)
        if kind == "w4a16":
            p_sl = self._slice(self._full_key(f"{linear}.weight_packed"))
            s_sl = self._slice(self._full_key(f"{linear}.weight_scale"))
            word_cols = idx // 8
            scale_cols = idx // 32
            bit_shift = torch.from_numpy(((idx % 8) * 4).astype(np.int32))
            u_wc, inv_wc = np.unique(word_cols, return_inverse=True)
            u_sc, inv_sc = np.unique(scale_cols, return_inverse=True)
            # If contiguous range is small enough relative to unique columns, slab-read; else slice per unique word
            if u_wc.size > 32 and int(u_wc[-1] - u_wc[0] + 1) <= 2 * u_wc.size:
                base_w = int(u_wc[0])
                slab_p = p_sl[:, base_w : int(u_wc[-1]) + 1]
                words = slab_p.index_select(1, torch.from_numpy((word_cols - base_w).astype(np.int64)))
            else:
                u_p = torch.stack([p_sl[:, int(wc)] for wc in u_wc], dim=1)
                words = u_p.index_select(1, torch.from_numpy(inv_wc.astype(np.int64)))
            if u_sc.size > 16 and int(u_sc[-1] - u_sc[0] + 1) <= 2 * u_sc.size:
                base_s = int(u_sc[0])
                slab_s = s_sl[:, base_s : int(u_sc[-1]) + 1]
                scales = slab_s.index_select(1, torch.from_numpy((scale_cols - base_s).astype(np.int64)))
            else:
                u_s = torch.stack([s_sl[:, int(sc)] for sc in u_sc], dim=1)
                scales = u_s.index_select(1, torch.from_numpy(inv_sc.astype(np.int64)))
            w_q = (((words >> bit_shift.unsqueeze(0)) & 0xF) - 8).to(dtype=scales.dtype)
            return (w_q * scales).T.contiguous()
        w_sl = self._slice(self._full_key(f"{linear}.weight"))
        uniq, inv = np.unique(idx, return_inverse=True)
        u_cols = torch.stack([w_sl[:, int(c)] for c in uniq], dim=1)
        return u_cols.index_select(1, torch.from_numpy(inv.astype(np.int64))).T.contiguous()

    def w_post(self, layer: int) -> np.ndarray:
        """Read-only float32 `[D]` `post_feedforward_layernorm.weight` of `layer` (ones if absent)."""
        L = self._check_layer(layer)
        got = self._w_post_cache.get(L)
        if got is not None:
            return got
        rel = f"layers.{L}.post_feedforward_layernorm.weight"
        full = f"{self.prefix}{rel}"
        if full in self._key_to_file:
            w = self._tensor(full).float().numpy()
        else:
            w = np.ones((self.d_model,), dtype=np.float32)
        arr = np.ascontiguousarray(w, dtype=np.float32)
        arr.setflags(write=False)
        self._w_post_cache[L] = arr
        return arr

    def norm_weight(self, layer: Optional[int], which: str = "post_feedforward_layernorm") -> np.ndarray:
        """Read-only float32 norm weight: pass `layer=None, which='norm'` for final norm, else layer norm name."""
        if layer is None and which == "norm":
            rel = "norm.weight"
        else:
            L = self._check_layer(int(layer))
            rel = f"layers.{L}.{which}.weight"
        arr = np.ascontiguousarray(self._tensor(self._full_key(rel)).float().numpy(), dtype=np.float32)
        arr.setflags(write=False)
        return arr

    def layer_scalar(self, layer: int) -> Optional[float]:
        """Float32 `layer_scalar` of `layer`, or None if the checkpoint has none."""
        L = self._check_layer(layer)
        full = f"{self.prefix}layers.{L}.layer_scalar"
        if full not in self._key_to_file:
            return None
        return float(self._tensor(full).float().reshape(-1)[0].item())

    @torch.no_grad()
    def embed_rows(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Read only the requested rows of `embed_tokens.weight` (`[K, D]`) from disk without loading all 262k rows."""
        idx = np.asarray(list(token_ids), dtype=np.int64).reshape(-1)
        sl = self._slice(self._full_key("embed_tokens.weight"))
        v_size, d = int(sl.get_shape()[0]), int(sl.get_shape()[1])
        if idx.size == 0:
            return torch.empty((0, d), dtype=torch.bfloat16)
        if int(idx.min()) < 0 or int(idx.max()) >= v_size:
            raise WeightReadError(f"Token id out of range (0 to {v_size - 1}).")
        uniq, inv = np.unique(idx, return_inverse=True)
        u_rows = torch.stack([sl[int(r), :] for r in uniq], dim=0)
        return u_rows.index_select(0, torch.from_numpy(inv.astype(np.int64))).contiguous()

    def quiet_mask(
        self,
        layer: int,
        *,
        min_ratio: float = QUIET_WPOST_RATIO,
        extra_mask: Any = None,
        use_pers181: bool = True,
    ) -> np.ndarray:
        """Read-only bool mask `[D]` matching `weight_edits.quiet_mask` exactly."""
        L = self._check_layer(layer)
        ratio = float(min_ratio)
        if not (math.isfinite(ratio) and ratio >= 0.0):
            raise WeightReadError(f"min_ratio must be finite and >= 0; got {min_ratio!r}.")
        wp = self.w_post(L)
        abs_w = np.abs(wp)
        med = float(np.median(abs_w))
        mask = (abs_w < ratio * med) | (abs_w == 0.0) if med > 0.0 else np.ones_like(abs_w, dtype=bool)
        if use_pers181:
            p181 = _pers181_mask(self._quiet_dims_path)
            if p181 is not None and p181.shape == mask.shape:
                mask = mask | p181
        if extra_mask is not None:
            extra = extra_mask.detach().cpu().numpy() if isinstance(extra_mask, torch.Tensor) else np.asarray(extra_mask)
            if extra.dtype != bool or extra.ndim != 1 or extra.shape != mask.shape:
                raise WeightReadError(f"extra_mask must be 1D bool of length {mask.shape[0]}.")
            mask = mask | extra
        mask = np.ascontiguousarray(mask, dtype=bool)
        mask.setflags(write=False)
        return mask

    @torch.no_grad()
    def down_stream_norms(self, layer: int, *, chunk_word_cols: int = DEFAULT_CHUNK_WORD_COLS) -> np.ndarray:
        """Chunked stream-side column lengths `||w_post * W_down[:, n]||_2` (`float64 [N]`, read-only), matching
        `weight_edits.down_stream_norms` in bounded RAM."""
        L = self._check_layer(layer)
        got = self._stream_norms_cache.get(L)
        if got is not None:
            return got
        lin = f"layers.{L}.mlp.down_proj"
        kind = self._linear_kind(lin)
        _out_f, N = self.linear_shape(lin)
        wp = torch.from_numpy(self.w_post(L).astype(np.float32))[:, None]
        out = np.empty(N, dtype=np.float64)
        if kind == "w4a16":
            p_sl = self._slice(self._full_key(f"{lin}.weight_packed"))
            s_sl = self._slice(self._full_key(f"{lin}.weight_scale"))
            # Ensure chunk aligns to scale groups (4 word cols = 32 elements = 1 scale col)
            step_w = max(4, (int(chunk_word_cols) // 4) * 4)
            n_words = N // 8
            for w0 in range(0, n_words, step_w):
                w1 = min(n_words, w0 + step_w)
                p_chunk = p_sl[:, w0:w1]
                s_chunk = s_sl[:, w0 // 4 : (w1 + 3) // 4]
                w_chunk = _dequant_w4a16_full_cpu(p_chunk, s_chunk).float() * wp
                out[w0 * 8 : w1 * 8] = w_chunk.pow(2).sum(dim=0, dtype=torch.float64).sqrt().numpy()
        else:
            w_sl = self._slice(self._full_key(f"{lin}.weight"))
            step_c = max(64, int(chunk_word_cols) * 8)
            for c0 in range(0, N, step_c):
                c1 = min(N, c0 + step_c)
                w_chunk = w_sl[:, c0:c1].float() * wp
                out[c0:c1] = w_chunk.pow(2).sum(dim=0, dtype=torch.float64).sqrt().numpy()
        out.setflags(write=False)
        self._stream_norms_cache[L] = out
        return out

    @torch.no_grad()
    def write_vectors_stream(self, layer: int, neurons: Sequence[int]) -> np.ndarray:
        """Stream-side write vectors `w_post * W_down[:, n]` (`float32 [K, D]`)."""
        L = self._check_layer(layer)
        cols = self.dense_cols(f"layers.{L}.mlp.down_proj", neurons).float().numpy()
        return np.ascontiguousarray(cols * self.w_post(L)[None, :], dtype=np.float32)

    @torch.no_grad()
    def writers(
        self,
        layer: int,
        direction: Any,
        k: int = 32,
        *,
        quiet_mask: Any = None,
        max_quiet_share: float = MAX_QUIET_SHARE,
        cosine: bool = False,
        chunk_word_cols: int = DEFAULT_CHUNK_WORD_COLS,
    ) -> List[Tuple[int, float]]:
        """Top-`k` neurons of `layer` ranked by `|(w_post * W_down[:, n]) . u_m|` (or cosine when `cosine=True`),
        matching `weight_edits.writers` in bounded RAM without unpacking the whole matrix at once."""
        L = self._check_layer(layer)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise WeightReadError(f"writers takes k >= 1; got {k!r}.")
        k = int(k)
        mq = float(max_quiet_share)
        if not (math.isfinite(mq) and 0.0 <= mq <= 1.0):
            raise WeightReadError(f"max_quiet_share must lie in [0, 1]; got {max_quiet_share!r}.")
        wp = self.w_post(L)
        D = int(wp.shape[0])
        u = direction.detach().cpu().float().numpy() if isinstance(direction, torch.Tensor) else np.asarray(direction, dtype=np.float32)
        if u.ndim != 1 or int(u.shape[0]) != D:
            raise WeightReadError(f"Direction at layer {L} (page layer {L + 1}) must be 1D of length {D}; got {list(u.shape)}.")
        if not np.isfinite(u).all():
            raise WeightReadError(f"Direction at layer {L} (page layer {L + 1}) holds a non-finite number.")
        u64 = u.astype(np.float64)
        u_norm2 = float(np.dot(u64, u64))
        if not u_norm2 > 0.0:
            raise WeightReadError(f"Direction at layer {L} (page layer {L + 1}) must be non-zero.")
        mask = self.quiet_mask(L, extra_mask=quiet_mask)
        q_share = float(np.dot(u64[mask], u64[mask]) / u_norm2)
        if q_share > mq or int((~mask).sum()) == 0:
            raise WeightReadError(
                f"{100.0 * q_share:.1f}% of the direction's squared length at layer {L} (page layer {L + 1}) "
                f"sits on the quiet mask (cap {100.0 * mq:.1f}%)."
            )
        u_m = u64.copy()
        u_m[mask] = 0.0
        u_m = u_m / float(np.linalg.norm(u_m))
        u_eff = torch.from_numpy((u_m * wp).astype(np.float32))

        lin = f"layers.{L}.mlp.down_proj"
        kind = self._linear_kind(lin)
        _out_f, N = self.linear_shape(lin)
        scores = np.empty(N, dtype=np.float64)
        if kind == "w4a16":
            p_sl = self._slice(self._full_key(f"{lin}.weight_packed"))
            s_sl = self._slice(self._full_key(f"{lin}.weight_scale"))
            step_w = max(4, (int(chunk_word_cols) // 4) * 4)
            n_words = N // 8
            for w0 in range(0, n_words, step_w):
                w1 = min(n_words, w0 + step_w)
                w_chunk = _dequant_w4a16_full_cpu(
                    p_sl[:, w0:w1], s_sl[:, w0 // 4 : (w1 + 3) // 4]
                ).float()
                scores[w0 * 8 : w1 * 8] = torch.matmul(u_eff, w_chunk).double().numpy()
        else:
            w_sl = self._slice(self._full_key(f"{lin}.weight"))
            step_c = max(64, int(chunk_word_cols) * 8)
            for c0 in range(0, N, step_c):
                c1 = min(N, c0 + step_c)
                scores[c0:c1] = torch.matmul(u_eff, w_sl[:, c0:c1].float()).double().numpy()
        if cosine:
            norms = self.down_stream_norms(L, chunk_word_cols=chunk_word_cols)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(norms > 0.0, scores / norms, 0.0)
        order = sorted(range(N), key=lambda i: (-abs(float(scores[i])), i))
        return [(int(i), float(scores[i])) for i in order[: min(k, N)]]


@torch.no_grad()
def export_stand_in_safetensors(lm: Any, out_dir: Union[str, Path], *, model_id: str = "stand-in") -> Path:
    """Write a stand-in text model's weights (packed W4A16 or dense) to `<out_dir>/model.safetensors` and
    `config.json`, so `WeightReader` and the off-card sandbox can read stand-ins via the exact same disk path as
    the 31B checkpoint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors: Dict[str, torch.Tensor] = {}
    for name, mod in lm.named_modules():
        if hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale"):
            tensors[f"{name}.weight_packed"] = mod.weight_packed.detach().cpu().contiguous()
            tensors[f"{name}.weight_scale"] = mod.weight_scale.detach().cpu().contiguous()
        elif isinstance(getattr(mod, "weight", None), torch.Tensor):
            w = mod.weight.detach().cpu().contiguous()
            if name:
                tensors[f"{name}.weight"] = w
        if hasattr(mod, "layer_scalar") and isinstance(mod.layer_scalar, torch.Tensor):
            tensors[f"{name}.layer_scalar"] = mod.layer_scalar.detach().cpu().contiguous()
    if not tensors:
        raise WeightReadError("No tensors found on the supplied model.")
    save_safetensors_torch(tensors, str(out / "model.safetensors"))
    cfg = {"model_id": str(model_id), "n_layers": len(getattr(lm, "layers", []))}
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return out
