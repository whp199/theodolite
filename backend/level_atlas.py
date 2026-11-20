"""Read-only atlas index reader for Levelling Phase 6 (`backend/level_atlas.py`).

Provides bounded, read-only access to the Theodolite atlas index (`lens.npz`, `act_scale.npz`, `wiring.npz`,
`steer_lens.npz`, `vocab.json`, `layout_writes.npz`) for the `"atlas"` reach — both inside the off-card `bwrap`
sandbox (injected as `A`) and via `LevelSession.atlas_query`.

Design & containment:
- Standard library + NumPy only: imports nothing from the live server, runtime, or PyTorch, so it can be mounted
  read-only into `/opt/app/backend/level_atlas.py` beside `weight_reader.py`.
- Allowlisted index files ONLY (`ATLAS_INDEX_FILES` and `ATLAS_INDEX_DIRS`): never opens `scientist/`, `uploads/`,
  `logs/`, `endpoints.json`, or any level book.
- Enforces the instrument's label-gate provenance on every neuron label: below code layer 40 (page layer 41) on a
  60-layer model, or where lens and naked-eye top-10 word-pieces share zero entries, labels are marked
  `label_trusted: False` with plain reasons (still returning the raw words as leads to check causally).

Layers count from 0 in code and files; human-facing strings give both ("layer 50 (page layer 51)").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Exact allowlist of read-only index artifacts that may be bound into `/atlas` inside the bwrap sandbox.
ATLAS_INDEX_FILES = (
    "lens.npz",
    "act_scale.npz",
    "wiring.npz",
    "steer_lens.npz",
    "steer_dose.json",
    "layout_writes.npz",
    "wiring_check.npz",
    "vocab.json",
    "head_profiles.json",
)
ATLAS_INDEX_DIRS = ("passages.d",)
# Folders/files that must never be mistaken for an index folder or exposed inside `/atlas`.
FORBIDDEN_ATLAS_ENTRIES = ("scientist", "uploads", "logs", "endpoints.json", "level_book.jsonl")

LABEL_MIN_LAYER_31B = 40  # counting from 0 (page layer 41) on 60-layer Gemma 4 31B
EMPTY_WIRING_Z = -1e4
MAX_QUERY_RESULTS = 64


class AtlasReaderError(ValueError):
    """Raised when an atlas index path is missing, malformed, or out of bounds."""


def resolve_atlas_index_dir(path: Path) -> Path:
    """Resolve an atlas index folder. Accepts either `<atlas_dir>/index` directly or `<atlas_dir>` when it has an
    `index/` subdirectory holding at least one allowlisted index file. Raises `AtlasReaderError` otherwise."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise AtlasReaderError(f"Atlas path {p} does not exist or is not a directory.")
    if any((p / fn).is_file() for fn in ATLAS_INDEX_FILES):
        return p
    sub = p / "index"
    if sub.is_dir() and any((sub / fn).is_file() for fn in ATLAS_INDEX_FILES):
        return sub
    raise AtlasReaderError(
        f"Atlas directory {p} contains no recognized index files ({', '.join(ATLAS_INDEX_FILES[:4])}, ...)."
    )


def collect_atlas_binds(path: Path) -> List[Tuple[Path, str]]:
    """Return `(resolved_host_path, container_dst)` pairs for ONLY allowlisted read-only atlas index files and
    `passages.d/` inside `path`. Never binds `scientist/`, `uploads/`, `logs/`, or level books."""
    idx_dir = resolve_atlas_index_dir(path)
    binds: List[Tuple[Path, str]] = []
    for fn in ATLAS_INDEX_FILES:
        fp = idx_dir / fn
        if fp.is_file():
            binds.append((fp.resolve(), f"/atlas/{fn}"))
    for dn in ATLAS_INDEX_DIRS:
        dp = idx_dir / dn
        if dp.is_dir():
            # Bind only plain .npy and meta.json files inside passages.d/
            for child in sorted(dp.iterdir()):
                if child.is_file() and (child.suffix == ".npy" or child.name == "meta.json"):
                    binds.append((child.resolve(), f"/atlas/{dn}/{child.name}"))
    if not binds:
        raise AtlasReaderError(f"Atlas index directory {idx_dir} has no readable index files.")
    return binds


def _decode_piece(vocab: Optional[Sequence[str]], tid: int) -> str:
    if vocab is not None and 0 <= tid < len(vocab):
        return str(vocab[tid])
    return f"<id:{tid}>"


def _clean_json_meta(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            raw = raw.item()
        else:
            return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


class AtlasReader:
    """Bounded read-only reader over Theodolite's atlas index files (CPU, zero GPU, zero server imports)."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = resolve_atlas_index_dir(index_dir)
        self._cache: Dict[str, Any] = {}
        self._arr_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._vocab: Optional[List[str]] = None
        vp = self.index_dir / "vocab.json"
        if vp.is_file():
            try:
                raw_v = json.loads(vp.read_text(encoding="utf-8"))
                if isinstance(raw_v, list):
                    self._vocab = [str(x) for x in raw_v]
                elif isinstance(raw_v, dict) and isinstance(raw_v.get("vocab"), list):
                    self._vocab = [str(x) for x in raw_v["vocab"]]
            except (OSError, ValueError):
                self._vocab = None

    def close(self) -> None:
        self._arr_cache.clear()
        for obj in self._cache.values():
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        self._cache.clear()

    def _arr(self, name: str, key: str) -> Optional[np.ndarray]:
        ck = (name, key)
        if ck in self._arr_cache:
            return self._arr_cache[ck]
        arc = self._npz(name)
        if arc is None or key not in arc:
            return None
        val = arc[key]
        self._arr_cache[ck] = val
        return val

    def __enter__(self) -> "AtlasReader":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _npz(self, name: str) -> Optional[Any]:
        if name in self._cache:
            return self._cache[name]
        fp = self.index_dir / name
        if not fp.is_file():
            self._cache[name] = None
            return None
        arc = np.load(fp, allow_pickle=False)
        self._cache[name] = arc
        return arc

    def _shape(self) -> Tuple[int, int]:
        for fname, key in (
            ("lens.npz", "lens_pos"),
            ("act_scale.npz", "pos"),
            ("wiring.npz", "down_z"),
            ("steer_lens.npz", "up"),
        ):
            arc = self._npz(fname)
            if arc is not None and key in arc:
                arr = arc[key]
                return int(arr.shape[0]), int(arr.shape[1])
        raise AtlasReaderError(f"Could not determine (n_layers, n_neurons) from {self.index_dir}.")

    def _label_min_layer(self, n_layers: int) -> int:
        # On 60-layer Gemma 4 31B, code layer 40 (page layer 41) is the study's measured gate floor.
        # On smaller stand-in models (e.g. 6 layers), scale proportionally so checks can exercise both sides.
        return LABEL_MIN_LAYER_31B if n_layers >= 60 else max(0, (n_layers * 2) // 3)

    def _label_trust(
        self,
        layer: int,
        n_layers: int,
        lens_ids: Optional[Sequence[int]],
        eye_ids: Optional[Sequence[int]],
        silenced: bool = False,
    ) -> Dict[str, Any]:
        if silenced:
            return {"trusted": False, "shared_top10": None, "reason": "writes_nothing"}
        min_l = self._label_min_layer(n_layers)
        shared = None
        if lens_ids is not None and eye_ids is not None:
            shared = len(set(int(x) for x in lens_ids[:10]) & set(int(x) for x in eye_ids[:10]))
        if layer < min_l:
            return {
                "trusted": False,
                "shared_top10": shared,
                "reason": f"below_layer_{min_l}",
            }
        if shared is not None and shared <= 0:
            return {"trusted": False, "shared_top10": 0, "reason": "shares_nothing"}
        return {"trusted": True, "shared_top10": shared, "reason": None}

    def info(self) -> Dict[str, Any]:
        """Summary of available index files, shape, built layers, and label-gate floor."""
        n_layers, n_neurons = self._shape()
        present = [fn for fn in ATLAS_INDEX_FILES if (self.index_dir / fn).is_file()]
        lens = self._npz("lens.npz")
        built_layers: Optional[List[int]] = None
        lens_meta = None
        if lens is not None:
            if "built" in lens:
                built_layers = [int(i) for i, ok in enumerate(lens["built"].tolist()) if ok]
            if "meta" in lens:
                lens_meta = _clean_json_meta(lens["meta"])
        return {
            "n_layers": n_layers,
            "n_neurons": n_neurons,
            "files": present,
            "has_vocab": self._vocab is not None,
            "vocab_size": len(self._vocab) if self._vocab is not None else 0,
            "built_layers_count": len(built_layers) if built_layers is not None else None,
            "label_min_layer": self._label_min_layer(n_layers),
            "lens_meta": lens_meta,
            "caution": (
                "Atlas labels are leads to check causally, not settled findings; labels below label_min_layer "
                "or with zero lens/eye top-10 overlap are marked trusted=False."
            ),
        }

    def search_words(
        self,
        query: str,
        *,
        layer: Optional[int] = None,
        layers: Optional[Sequence[int]] = None,
        face: str = "pos",
        source: str = "lens",
        top_k: int = 20,
        trusted_only: bool = False,
        substring: bool = False,
    ) -> Dict[str, Any]:
        """Search neurons whose top words match `query` (exact word token by default, `substring=True` for substring,
        or exact token id `id:123`). Falls back to substring only if no exact token exists."""
        q_raw = str(query or "")
        if not q_raw.strip():
            raise AtlasReaderError("search_words requires a non-empty query string.")
        if face not in ("pos", "neg"):
            raise AtlasReaderError(f'face must be "pos" or "neg"; got {face!r}.')
        if source not in ("lens", "eye", "steer"):
            raise AtlasReaderError(f'source must be "lens", "eye", or "steer"; got {source!r}.')
        k_lim = max(1, min(int(top_k), MAX_QUERY_RESULTS))
        n_layers, n_neurons = self._shape()

        if source == "steer":
            arr = self._arr("steer_lens.npz", "up" if face == "pos" else "down")
            if arr is None:
                raise AtlasReaderError("steer_lens.npz is not present in this atlas index.")
            lens_name = "lens.npz"
        else:
            arr = self._arr("lens.npz", f"{source}_{face}")
            if arr is None:
                raise AtlasReaderError("lens.npz is not present in this atlas index.")
            lens_name = "lens.npz"

        # Determine matching token ids from vocab or explicit id:<int>
        match_ids: set[int] = set()
        q_strip = q_raw.strip()
        if q_strip.startswith("id:") and q_strip[3:].isdigit():
            match_ids.add(int(q_strip[3:]))
        elif self._vocab is not None:
            q_low = q_raw.lower()
            q_low_strip = q_strip.lower()
            if not substring:
                for tid, piece in enumerate(self._vocab):
                    if piece.lstrip(" \u2581").strip().lower() == q_low_strip:
                        match_ids.add(tid)
            if substring or not match_ids:
                for tid, piece in enumerate(self._vocab):
                    p_low = piece.lower()
                    if q_low in p_low or (q_low_strip and q_low_strip == p_low.lstrip(" \u2581").strip()):
                        match_ids.add(tid)
        else:
            raise AtlasReaderError("vocab.json is absent; query by token id as 'id:<int>'.")

        if layer is not None and layers is not None:
            raise AtlasReaderError("Pass either layer or layers, not both.")
        if layer is not None:
            l_val = int(layer)
            if not 0 <= l_val < n_layers:
                raise AtlasReaderError(f"Layer {l_val} is outside 0..{n_layers - 1}.")
            scan_layers = [l_val]
        elif layers is not None:
            scan_layers = []
            for l_raw in layers:
                lv_i = int(l_raw)
                if not 0 <= lv_i < n_layers:
                    raise AtlasReaderError(f"Layer {lv_i} is outside 0..{n_layers - 1}.")
                scan_layers.append(lv_i)
        else:
            scan_layers = list(range(n_layers))

        if not match_ids:
            return {
                "query": q_raw,
                "source": source,
                "face": face,
                "matched_token_ids": 0,
                "hits": [],
                "note": "No vocabulary pieces matched the query.",
            }

        built = self._arr(lens_name, "built")
        silenced_arr = self._arr(lens_name, "silenced")
        lens_other_arr = self._arr(lens_name, f"lens_{face}")
        eye_other_arr = self._arr(lens_name, f"eye_{face}")
        has_both = lens_other_arr is not None and eye_other_arr is not None

        # Rank hits by best rank (0 is strongest), then later layers first when tied.
        match_np = np.fromiter(sorted(match_ids), dtype=np.int64)
        hits: List[Tuple[int, int, int, Dict[str, Any]]] = []
        for l_i in scan_layers:
            if built is not None and not bool(built[l_i]):
                continue
            mat = arr[l_i]  # [N, K]
            mask = np.isin(mat, match_np)
            nz_neurons = np.flatnonzero(mask.any(axis=1))
            for n_i_np in nz_neurons:
                n_i = int(n_i_np)
                ranks = np.flatnonzero(mask[n_i])
                best_rank = int(ranks[0])
                is_sil = bool(silenced_arr[l_i, n_i]) if silenced_arr is not None else False
                l_ids = lens_other_arr[l_i, n_i].tolist() if has_both else None
                e_ids = eye_other_arr[l_i, n_i].tolist() if has_both else None
                trust = self._label_trust(l_i, n_layers, l_ids, e_ids, silenced=is_sil)
                if trusted_only and not trust["trusted"]:
                    continue
                row_ids = [int(x) for x in mat[n_i].tolist()]
                words = [_decode_piece(self._vocab, x) for x in row_ids[:10]]
                matched_words = [_decode_piece(self._vocab, row_ids[int(r)]) for r in ranks[:4]]
                rec = {
                    "layer": l_i,
                    "page_layer": l_i + 1,
                    "neuron": n_i,
                    "best_rank": best_rank,
                    "matched_words": matched_words,
                    "top_words": words,
                    "label_trusted": trust["trusted"],
                    "shared_top10": trust["shared_top10"],
                    "trust_reason": trust["reason"],
                }
                hits.append((0 if trust["trusted"] else 1, best_rank, -l_i, rec))

        hits.sort(key=lambda t: (t[0], t[1], t[2], t[3]["neuron"]))
        out_hits = [t[3] for t in hits[:k_lim]]
        return {
            "query": q_raw,
            "source": source,
            "face": face,
            "matched_token_ids": len(match_ids),
            "total_hits": len(hits),
            "hits": out_hits,
        }

    def neuron_card(self, layer: int, neuron: int, *, top_wiring: int = 8) -> Dict[str, Any]:
        """Bounded read-only card for one neuron: lens/eye/steer words, activation extremes, and top wiring."""
        n_layers, n_neurons = self._shape()
        l_i, n_i = int(layer), int(neuron)
        if not 0 <= l_i < n_layers:
            raise AtlasReaderError(f"Layer {l_i} is outside 0..{n_layers - 1}.")
        if not 0 <= n_i < n_neurons:
            raise AtlasReaderError(f"Neuron {n_i} is outside 0..{n_neurons - 1}.")
        card: Dict[str, Any] = {
            "layer": l_i,
            "page_layer": l_i + 1,
            "neuron": n_i,
        }
        if self._npz("lens.npz") is not None:
            sil_a = self._arr("lens.npz", "silenced")
            is_sil = bool(sil_a[l_i, n_i]) if sil_a is not None else False
            lp_a = self._arr("lens.npz", "lens_pos")
            ln_a = self._arr("lens.npz", "lens_neg")
            ep_a = self._arr("lens.npz", "eye_pos")
            en_a = self._arr("lens.npz", "eye_neg")
            if lp_a is not None and ln_a is not None and ep_a is not None and en_a is not None:
                lp = [int(x) for x in lp_a[l_i, n_i].tolist()]
                ln = [int(x) for x in ln_a[l_i, n_i].tolist()]
                ep = [int(x) for x in ep_a[l_i, n_i].tolist()]
                en = [int(x) for x in en_a[l_i, n_i].tolist()]
                card["silenced"] = is_sil
                card["lens_pos"] = [_decode_piece(self._vocab, x) for x in lp[:10]]
                card["lens_neg"] = [_decode_piece(self._vocab, x) for x in ln[:10]]
                card["eye_pos"] = [_decode_piece(self._vocab, x) for x in ep[:10]]
                card["eye_neg"] = [_decode_piece(self._vocab, x) for x in en[:10]]
                card["label_trust_pos"] = self._label_trust(l_i, n_layers, lp, ep, silenced=is_sil)
                card["label_trust_neg"] = self._label_trust(l_i, n_layers, ln, en, silenced=is_sil)
        act_pos = self._arr("act_scale.npz", "pos")
        act_neg = self._arr("act_scale.npz", "neg")
        if act_pos is not None and act_neg is not None:
            card["act_extremes"] = {
                "pos": float(act_pos[l_i, n_i]),
                "neg": float(act_neg[l_i, n_i]),
            }
        st_up = self._arr("steer_lens.npz", "up")
        st_dn = self._arr("steer_lens.npz", "down")
        if st_up is not None and st_dn is not None:
            card["steer_up"] = [_decode_piece(self._vocab, int(x)) for x in st_up[l_i, n_i][:8].tolist()]
            card["steer_down"] = [_decode_piece(self._vocab, int(x)) for x in st_dn[l_i, n_i][:8].tolist()]
            st_uk = self._arr("steer_lens.npz", "up_kl")
            st_dk = self._arr("steer_lens.npz", "down_kl")
            if st_uk is not None and st_dk is not None:
                card["steer_kl"] = {
                    "up": float(st_uk[l_i, n_i]),
                    "down": float(st_dk[l_i, n_i]),
                }
        if top_wiring > 0 and self._npz("wiring.npz") is not None:
            card["wiring_down"] = self.top_wiring(l_i, n_i, direction="down", top_k=top_wiring)["links"]
            card["wiring_up"] = self.top_wiring(l_i, n_i, direction="up", top_k=top_wiring)["links"]
        return card

    def top_wiring(
        self,
        layer: int,
        neuron: int,
        *,
        direction: str = "down",
        top_k: int = 12,
    ) -> Dict[str, Any]:
        """Strongest downstream (`down`) or upstream (`up`) weight-wired neurons from `wiring.npz`."""
        if direction not in ("down", "up"):
            raise AtlasReaderError(f'direction must be "down" or "up"; got {direction!r}.')
        n_layers, n_neurons = self._shape()
        l_i, n_i = int(layer), int(neuron)
        if not 0 <= l_i < n_layers:
            raise AtlasReaderError(f"Layer {l_i} is outside 0..{n_layers - 1}.")
        if not 0 <= n_i < n_neurons:
            raise AtlasReaderError(f"Neuron {n_i} is outside 0..{n_neurons - 1}.")
        zs_a = self._arr("wiring.npz", f"{direction}_z")
        ls_a = self._arr("wiring.npz", f"{direction}_layer")
        ns_a = self._arr("wiring.npz", f"{direction}_n")
        if zs_a is None or ls_a is None or ns_a is None:
            raise AtlasReaderError("wiring.npz is not present or missing wiring arrays in this atlas index.")
        k_lim = max(1, min(int(top_k), MAX_QUERY_RESULTS))
        zs = zs_a[l_i, n_i]
        ls_ = ls_a[l_i, n_i]
        ns_ = ns_a[l_i, n_i]
        sig_a = self._arr("wiring.npz", f"{direction}_sign")
        lw_a = self._arr("wiring.npz", f"{direction}_lw")
        signs = sig_a[l_i, n_i] if sig_a is not None else None
        lws = lw_a[l_i, n_i] if lw_a is not None else None
        links: List[Dict[str, Any]] = []
        for idx in range(min(len(zs), k_lim)):
            z_val = float(zs[idx])
            if z_val <= EMPTY_WIRING_Z:
                continue
            tgt_l = int(ls_[idx])
            tgt_n = int(ns_[idx])
            item: Dict[str, Any] = {
                "layer": tgt_l,
                "page_layer": tgt_l + 1,
                "neuron": tgt_n,
                "z": round(z_val, 4),
            }
            if signs is not None:
                item["sign"] = int(signs[idx])
            if lws is not None:
                item["lw"] = round(float(lws[idx]), 4)
            links.append(item)
        return {"layer": l_i, "page_layer": l_i + 1, "neuron": n_i, "direction": direction, "links": links}

    def layer_stats(self, layer: int, *, top_k: int = 10) -> Dict[str, Any]:
        """Layer-level residual/norm quantiles and hardest-firing neurons from `act_scale.npz`."""
        n_layers, n_neurons = self._shape()
        l_i = int(layer)
        if not 0 <= l_i < n_layers:
            raise AtlasReaderError(f"Layer {l_i} is outside 0..{n_layers - 1}.")
        pos_a = self._arr("act_scale.npz", "pos")
        neg_a = self._arr("act_scale.npz", "neg")
        if pos_a is None or neg_a is None:
            raise AtlasReaderError("act_scale.npz is not present in this atlas index.")
        k_lim = max(1, min(int(top_k), MAX_QUERY_RESULTS))
        pos_l = np.asarray(pos_a[l_i], dtype=np.float32)
        neg_l = np.asarray(neg_a[l_i], dtype=np.float32)
        order = np.argsort(-np.maximum(np.abs(pos_l), np.abs(neg_l)))[:k_lim]
        out: Dict[str, Any] = {
            "layer": l_i,
            "page_layer": l_i + 1,
            "n_neurons": n_neurons,
            "top_extreme_neurons": [
                {"neuron": int(n), "pos": round(float(pos_l[n]), 4), "neg": round(float(neg_l[n]), 4)}
                for n in order
            ],
        }
        for q_key in ("resid_q", "mlp_rms_q", "mid_rms_q"):
            qa = self._arr("act_scale.npz", q_key)
            if qa is not None:
                out[q_key] = [round(float(x), 4) for x in qa[l_i].tolist()]
        return out

    def query(self, op: str, args: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Typed dispatcher used by `LevelSession.atlas_query`."""
        a = dict(args or {})
        if op == "info":
            return self.info()
        if op == "search_words":
            return self.search_words(
                a.get("query", ""),
                layer=a.get("layer"),
                layers=a.get("layers"),
                face=str(a.get("face", "pos")),
                source=str(a.get("source", "lens")),
                top_k=int(a.get("top_k", 20)),
                trusted_only=bool(a.get("trusted_only", False)),
                substring=bool(a.get("substring", False)),
            )
        if op == "neuron_card":
            if "layer" not in a or "neuron" not in a:
                raise AtlasReaderError("neuron_card requires integer 'layer' and 'neuron'.")
            return self.neuron_card(int(a["layer"]), int(a["neuron"]), top_wiring=int(a.get("top_wiring", 8)))
        if op == "top_wiring":
            if "layer" not in a or "neuron" not in a:
                raise AtlasReaderError("top_wiring requires integer 'layer' and 'neuron'.")
            return self.top_wiring(
                int(a["layer"]),
                int(a["neuron"]),
                direction=str(a.get("direction", "down")),
                top_k=int(a.get("top_k", 12)),
            )
        if op == "layer_stats":
            if "layer" not in a:
                raise AtlasReaderError("layer_stats requires integer 'layer'.")
            return self.layer_stats(int(a["layer"]), top_k=int(a.get("top_k", 10)))
        raise AtlasReaderError(
            f'Unknown atlas query op {op!r}; choose one of info, search_words, neuron_card, top_wiring, layer_stats.'
        )


def build_stand_in_atlas_index(
    lm: Any,
    W_U: Any,
    tokenizer: Any,
    out_dir: Path,
    *,
    model_id: str = "stand-in",
    k_lens: int = 12,
    k_wiring: int = 8,
) -> Path:
    """Build a compact, genuine read-only atlas index (`lens.npz`, `act_scale.npz`, `wiring.npz`, `vocab.json`)
    directly from a stand-in model's weights on CPU for tests and stand-in Phase 6 studies.

    Computes real logit-lens projections (`w_post * W_down[:, n]` dotted with `W_U`, softcapped at 30) so
    `AtlasReader.search_words` finds genuine word-writing neurons of the stand-in model!"""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_layers = len(lm.layers)
    down0 = lm.layers[0].mlp.down_proj
    n_neurons = int(getattr(down0, "in_features", down0.weight.shape[1]))
    vocab_n = int(W_U.shape[0])
    wu = W_U.detach().to("cpu", dtype=torch.float32)  # [V, H]
    k_l = min(int(k_lens), vocab_n)
    k_w = min(int(k_wiring), n_neurons)

    lens_pos = np.zeros((n_layers, n_neurons, k_l), dtype=np.int32)
    lens_neg = np.zeros((n_layers, n_neurons, k_l), dtype=np.int32)
    eye_pos = np.zeros((n_layers, n_neurons, k_l), dtype=np.int32)
    eye_neg = np.zeros((n_layers, n_neurons, k_l), dtype=np.int32)
    silenced = np.zeros((n_layers, n_neurons), dtype=bool)
    built = np.ones((n_layers,), dtype=bool)

    pos_ext = np.zeros((n_layers, n_neurons), dtype=np.float32)
    neg_ext = np.zeros((n_layers, n_neurons), dtype=np.float32)
    resid_q = np.ones((n_layers, 3), dtype=np.float32)
    mlp_rms_q = np.ones((n_layers, 3), dtype=np.float32)
    mid_rms_q = np.ones((n_layers, 3), dtype=np.float32)

    down_layer = np.full((n_layers, n_neurons, k_w), -1, dtype=np.int8)
    down_n = np.zeros((n_layers, n_neurons, k_w), dtype=np.int16)
    down_z = np.full((n_layers, n_neurons, k_w), -2e4, dtype=np.float16)
    down_lw = np.zeros((n_layers, n_neurons, k_w), dtype=np.float16)
    down_sign = np.ones((n_layers, n_neurons, k_w), dtype=np.int8)
    up_layer = np.full((n_layers, n_neurons, k_w), -1, dtype=np.int8)
    up_n = np.zeros((n_layers, n_neurons, k_w), dtype=np.int16)
    up_z = np.full((n_layers, n_neurons, k_w), -2e4, dtype=np.float16)
    up_lw = np.zeros((n_layers, n_neurons, k_w), dtype=np.float16)
    up_sign = np.ones((n_layers, n_neurons, k_w), dtype=np.int8)

    stream_cols: List[torch.Tensor] = []
    with torch.no_grad():
        for l_i in range(n_layers):
            layer_m = lm.layers[l_i]
            w_down = layer_m.mlp.down_proj.weight.detach().to("cpu", dtype=torch.float32)  # [H, N]
            w_post = layer_m.post_feedforward_layernorm.weight.detach().to("cpu", dtype=torch.float32)  # [H]
            eff = w_post[:, None] * w_down  # [H, N]
            norms = eff.norm(dim=0)  # [N]
            silenced[l_i] = (norms <= 1e-12).numpy()
            unit = eff / norms.clamp_min(1e-12)[None, :]
            stream_cols.append(unit)
            logits = torch.tanh((unit.T @ wu.T) / 30.0) * 30.0  # [N, V]
            top_p = torch.topk(logits, k_l, dim=-1).indices.numpy().astype(np.int32)
            top_n = torch.topk(-logits, k_l, dim=-1).indices.numpy().astype(np.int32)
            eye_pos[l_i] = top_p
            eye_neg[l_i] = top_n
            lens_pos[l_i] = top_p
            lens_neg[l_i] = top_n
            pos_ext[l_i] = norms.numpy()
            neg_ext[l_i] = -0.5 * norms.numpy()

        # Simple genuine adjacent-layer wiring (down_proj -> next layer's up_proj)
        for l_i in range(n_layers - 1):
            w_up_next = lm.layers[l_i + 1].mlp.up_proj.weight.detach().to("cpu", dtype=torch.float32)  # [N, H]
            prod = w_up_next @ stream_cols[l_i]  # [N_next, N_cur]
            abs_p = prod.abs()
            vals, idxs = torch.topk(abs_p, k_w, dim=0)  # [k_w, N_cur]
            for n_i in range(n_neurons):
                top_v = vals[:, n_i]
                top_id = idxs[:, n_i]
                vmax = float(top_v[0].item()) if float(top_v[0].item()) > 1e-12 else 1.0
                for j in range(k_w):
                    tgt = int(top_id[j].item())
                    raw = float(prod[tgt, n_i].item())
                    down_layer[l_i, n_i, j] = l_i + 1
                    down_n[l_i, n_i, j] = tgt
                    down_z[l_i, n_i, j] = np.float16(min(100.0, abs(raw)))
                    down_lw[l_i, n_i, j] = np.float16(np.log10(max(abs(raw) / vmax, 1e-6)))
                    down_sign[l_i, n_i, j] = 1 if raw >= 0.0 else -1

    vocab = [tokenizer.decode([tid]) for tid in range(vocab_n)]
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=True), encoding="utf-8")
    meta = json.dumps({"model": model_id, "lens_id": "stand-in-eye", "k": k_l})
    np.savez(
        out / "lens.npz",
        lens_pos=lens_pos,
        lens_neg=lens_neg,
        eye_pos=eye_pos,
        eye_neg=eye_neg,
        silenced=silenced,
        built=built,
        K=np.int32(k_l),
        lens_id=np.array("stand-in-eye"),
        meta=np.array(meta),
    )
    np.savez(
        out / "act_scale.npz",
        pos=pos_ext,
        neg=neg_ext,
        resid_q=resid_q,
        mlp_rms_q=mlp_rms_q,
        mid_rms_q=mid_rms_q,
        meta=np.array(meta),
    )
    np.savez(
        out / "wiring.npz",
        down_layer=down_layer,
        down_n=down_n,
        down_z=down_z,
        down_lw=down_lw,
        down_sign=down_sign,
        up_layer=up_layer,
        up_n=up_n,
        up_z=up_z,
        up_lw=up_lw,
        up_sign=up_sign,
        K=np.int32(k_w),
        meta=np.array(meta),
    )
    return out
