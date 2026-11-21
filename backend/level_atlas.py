"""Read-only atlas index reader for Levelling Phase 6 (`backend/level_atlas.py`).

Provides bounded, read-only access to the Theodolite atlas index (`lens.npz`, `act_scale.npz`, `wiring.npz`,
`steer_lens.npz`, `vocab.json`, `layout_writes.npz`) for the `"atlas"` reach — both inside the off-card `bwrap`
sandbox (injected as `A`) and via `LevelSession.atlas_query`.

Design & containment:
- Standard library + NumPy only: imports nothing from the live server, runtime, or PyTorch, so it can be mounted
  read-only into `/opt/app/backend/level_atlas.py` beside `weight_reader.py`.
- Allowlisted index files ONLY (`ATLAS_INDEX_FILES` and `ATLAS_INDEX_DIRS`): never opens `scientist/`, `uploads/`,
  `logs/`, `endpoints.json`, or any level book.
- Marks every neuron label with the instrument's label gate (`backend/label_gate.py`, rule "n1-gate-1"), restated
  below because this file imports nothing but the standard library and NumPy: `held_back` below code layer 40 (page
  layer 41), and, under the base-fitted J++ lens only, where the lens and naked-eye readings share none of their
  first ten word-pieces. The gate only sets labels aside; a label it lets through is not thereby sound (as a cut, on
  N1's sample, it kept 209 labels of which 160 were sound). `scripts/check_level_eval.py` section 1 holds this
  restatement to `label_gate.label_check` case by case. The words are always returned, as leads to test.
- Steer-lens words are set aside at layers where the steer lens's own check (`steer_check.json`, written by
  `backend/steer_check.py` for the same build of the lens) found it a weak guide or did not check it, as the
  instrument's pages do (`backend/atlas_index.py` `steer_weak`). The label gate reads the lens and the naked eye, not
  the steer lens, so it says nothing of a steer reading.

Layers count from 0 in code and files; human-facing strings give both ("layer 50 (page layer 51)").
"""
from __future__ import annotations

import json
import os
import stat
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
    "steer_check.json",
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

# The label gate's rule, restated from backend/label_gate.py (rule "n1-gate-1"; counting layers from 0). Section 1 of
# scripts/check_level_eval.py compares `gate_check` with `label_gate.label_check` on every layer, overlap, lens and
# silenced case, so the two cannot part unseen.
GATE_RULE = "n1-gate-1"
GATE_MIN_LAYER = 40
GATE_BAND = (40, 57)  # where the study measured the gate
GATE_SAMPLED = (10, 20, 30, 40, 45, 50, 55, 57)
GATE_ONE_READING_LAYER = 59  # J++[59] is the identity: the lens and the naked eye are one reading there
GATE_TOPK = 10
GATE_CUT = 0  # a label is set aside when its two readings share at most this many of their first ten
GATE_VALIDATED_LENS_ID = "jpp-base"  # the only lens the agreement half of the gate was validated for
GATE_SAYS = (
    "The label gate (rule n1-gate-1) marks a label held_back below layer 40 (counting from 0), and, under the "
    "base-fitted J++ lens (jpp-base) below layer 59, where its lens and naked-eye readings share none of their first "
    "ten word-pieces; under any other lens only the layer counts (lens_validated false). It only sets labels aside: "
    "as a cut it did much less well than as a graded score (on N1's sample it kept 209 labels, of which 160 were "
    "sound; AUC 0.64), so a label it lets through is a lead to test, not a finding."
)
STEER_SET_ASIDE = ("weak", "unchecked")  # backend/atlas_index.py steer_weak: a layer the steer check did not back
STEER_SAYS = (
    "Steer-lens words are set aside at layers where the steer lens's own check (steer_check.json) found it a weak "
    "guide beside the layer's census or did not check it (on the 31B it agrees at layers 39 to 52); the label gate "
    "reads the lens and the naked eye, not the steer lens, and says nothing of a steer reading."
)


def gate_band(layer_idx: int) -> str:
    """Where a layer (counting from 0) stands against the study's sample (`label_gate.band_of`)."""
    l = int(layer_idx)
    if l < GATE_SAMPLED[0]:
        return "below_sampled"
    if l <= 30:
        return "chance_where_sampled"
    if l < GATE_BAND[0]:
        return "not_sampled"
    if l <= GATE_BAND[1]:
        return "band"
    if l < GATE_ONE_READING_LAYER:
        return "above_band"
    return "one_reading"


def gate_check(
    layer_idx: int,
    lens_ids: Optional[Sequence[int]],
    eye_ids: Optional[Sequence[int]],
    lens_id: Optional[str],
    silenced: bool = False,
) -> Dict[str, Any]:
    """`label_gate.label_check` restated for one face of a 60-layer model's index, which reads the shipped weights
    (so `edited` is always false): `shared_top10`, `held_back`, `reasons`, `band`, `sampled_layer`,
    `in_measured_band`, `lens_validated`, `edited`."""
    l = int(layer_idx)
    lens_l = [int(i) for i in list(lens_ids)] if lens_ids is not None else []
    eye_l = [int(i) for i in list(eye_ids)] if eye_ids is not None else []
    shared: Optional[int] = None
    if lens_l and eye_l and not silenced:
        shared = len(set(lens_l[:GATE_TOPK]) & set(eye_l[:GATE_TOPK]))
    base = lens_id == GATE_VALIDATED_LENS_ID
    reasons: List[str] = []
    if l < GATE_MIN_LAYER:
        reasons.append("below_layer_40")
    if shared is not None and shared <= GATE_CUT and l < GATE_ONE_READING_LAYER and base:
        reasons.append("shares_nothing")
    if silenced:
        reasons.append("writes_nothing")
    return {
        "shared_top10": shared,
        "held_back": bool(reasons),
        "reasons": reasons,
        "band": gate_band(l),
        "sampled_layer": l in GATE_SAMPLED,
        "in_measured_band": GATE_BAND[0] <= l <= GATE_BAND[1],
        "lens_validated": bool(base),
        "edited": False,
    }


def _plain_entry(p: Path, *, want_dir: bool = False) -> bool:
    """Whether `p` is itself a plain file (or a plain folder), not a symlink: the sandbox binds what it names."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) if want_dir else stat.S_ISREG(st.st_mode)


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
    """Return `(host_path, container_dst)` pairs for ONLY allowlisted read-only atlas index files and `passages.d/`
    inside `path`. Never binds `scientist/`, `uploads/`, `logs/`, or level books. A symlink where an allowlisted name
    stands is refused, not followed: it could name any file the owner can read."""
    idx_dir = resolve_atlas_index_dir(path)
    binds: List[Tuple[Path, str]] = []
    for fn in ATLAS_INDEX_FILES:
        fp = idx_dir / fn
        if fp.is_symlink():
            raise AtlasReaderError(f"Atlas index entry {fp} is a symlink; the sandbox binds only plain files.")
        if _plain_entry(fp):
            binds.append((fp, f"/atlas/{fn}"))
    for dn in ATLAS_INDEX_DIRS:
        dp = idx_dir / dn
        if dp.is_symlink():
            raise AtlasReaderError(f"Atlas index folder {dp} is a symlink; the sandbox binds only plain folders.")
        if _plain_entry(dp, want_dir=True):
            # Bind only plain .npy and meta.json files inside passages.d/
            for child in sorted(dp.iterdir()):
                if not (child.suffix == ".npy" or child.name == "meta.json"):
                    continue
                if child.is_symlink():
                    raise AtlasReaderError(f"Atlas index entry {child} is a symlink; the sandbox binds only plain files.")
                if _plain_entry(child):
                    binds.append((child, f"/atlas/{dn}/{child.name}"))
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
        # The gate's layer floor is the 31B's (code layer 40, page layer 41). A stand-in model of another depth (the
        # CPU checks use 6 layers) gets the floor scaled to two thirds of its depth, so checks can exercise both
        # sides; such a reading says so (`floor_scaled_for_stand_in`) and has no band.
        return LABEL_MIN_LAYER_31B if n_layers == 60 else max(0, (n_layers * 2) // 3)

    def _lens_id(self) -> Optional[str]:
        """The id of the lens the index's lens labels were read through (lens.npz `meta`, else `lens_id`)."""
        lens = self._npz("lens.npz")
        if lens is None:
            return None
        if "meta" in lens:
            meta = _clean_json_meta(lens["meta"])
            if isinstance(meta, dict) and meta.get("lens_id") is not None:
                return str(meta.get("lens_id"))
        if "lens_id" in lens:
            raw = _clean_json_meta(lens["lens_id"])
            return None if raw is None else str(raw)
        return None

    def _label_gate(
        self,
        layer: int,
        n_layers: int,
        lens_ids: Optional[Sequence[int]],
        eye_ids: Optional[Sequence[int]],
        silenced: bool = False,
    ) -> Dict[str, Any]:
        """The label gate's check of one face (`gate_check`); on a stand-in of another depth, the same rule with the
        layer floor scaled (no band: the study's bands are the 31B's)."""
        lens_id = self._lens_id()
        if n_layers == 60:
            return gate_check(layer, lens_ids, eye_ids, lens_id, silenced=silenced)
        floor = self._label_min_layer(n_layers)
        g = gate_check(GATE_MIN_LAYER if layer >= floor else 0, lens_ids, eye_ids, lens_id, silenced=silenced)
        g["reasons"] = [f"below_layer_{floor}" if r == "below_layer_40" else r for r in g["reasons"]]
        g.update({"band": None, "sampled_layer": False, "in_measured_band": False, "floor_scaled_for_stand_in": True})
        return g

    def _steer_check(self, layer: int) -> Dict[str, Any]:
        """What the steer lens's own check found at `layer` (`steer_check.json`), as `backend/atlas_index.py` reads
        it: `standing` ("agrees", "weak", "unchecked"), or None when no check was written for this build of the lens
        (then nothing is known of the lens there). `set_aside`: the words are left out at this layer."""
        if "steer_check" not in self._cache:
            rec = None
            p = self.index_dir / "steer_check.json"
            if _plain_entry(p):
                try:
                    rec = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    rec = None
            st = self._npz("steer_lens.npz")
            meta = _clean_json_meta(st["meta"]) if st is not None and "meta" in st else None
            built = meta.get("built_at") if isinstance(meta, dict) else None
            if not isinstance(rec, dict) or (rec.get("lens") or {}).get("built_at") != built or built is None:
                rec = None
            self._cache["steer_check"] = rec
        rec = self._cache["steer_check"]
        row = ((rec or {}).get("layers") or {}).get(str(int(layer))) if rec else None
        standing = row.get("standing") if isinstance(row, dict) else None
        return {
            "standing": standing,
            "set_aside": standing in STEER_SET_ASIDE,
            "check_written": rec is not None,
        }

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
            "lens_id": self._lens_id(),
            "lens_meta": lens_meta,
            "label_gate": GATE_SAYS,
            "steer_words": STEER_SAYS,
            "caution": "Atlas words are leads to test by a reading, not findings.",
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
        gate_passes_only: bool = False,
        substring: bool = False,
        trusted_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Search neurons whose top words match `query` (exact word token by default, `substring=True` for substring,
        or exact token id `id:123`). Falls back to substring only if no exact token exists.

        Each hit carries the label gate's check of that face (`label_gate`, of its lens and naked-eye readings), and a
        steer-source hit also what the steer lens's own check found at its layer (`steer_check`). Hits the gate lets
        through come first. `gate_passes_only` leaves out faces the gate marks held_back and, for the steer source,
        layers where the steer check sets the steer words aside. `trusted_only` is an older name for it."""
        if trusted_only is not None:
            gate_passes_only = bool(gate_passes_only or trusted_only)
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
                gate = self._label_gate(l_i, n_layers, l_ids, e_ids, silenced=is_sil)
                steer = self._steer_check(l_i) if source == "steer" else None
                set_aside = bool(gate["held_back"]) or bool(steer and steer["set_aside"])
                if gate_passes_only and set_aside:
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
                    "label_gate": gate,
                }
                if steer is not None:
                    rec["steer_check"] = steer
                hits.append((1 if set_aside else 0, best_rank, -l_i, rec))

        hits.sort(key=lambda t: (t[0], t[1], t[2], t[3]["neuron"]))
        out_hits = [t[3] for t in hits[:k_lim]]
        out = {
            "query": q_raw,
            "source": source,
            "face": face,
            "matched_token_ids": len(match_ids),
            "total_hits": len(hits),
            "hits": out_hits,
            "label_gate": GATE_SAYS,
        }
        if source == "steer":
            out["steer_words"] = STEER_SAYS
        return out

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
                card["label_gate_pos"] = self._label_gate(l_i, n_layers, lp, ep, silenced=is_sil)
                card["label_gate_neg"] = self._label_gate(l_i, n_layers, ln, en, silenced=is_sil)
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
            # The steer words are shown with what the steer lens's own check found at this layer: set aside where it
            # found the lens weak or did not check it (the label gate above reads the lens and the eye, not these).
            card["steer_check"] = self._steer_check(l_i)
            card["steer_words"] = STEER_SAYS
        if "label_gate_pos" in card:
            card["label_gate"] = GATE_SAYS
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
                gate_passes_only=bool(a.get("gate_passes_only", False) or a.get("trusted_only", False)),
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
