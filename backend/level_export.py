"""Levelling Phase 5: the owner's export of an edit (`backend/level_export.py`).

Writes an edit (column terms on `down_proj`, row terms on `gate_proj` and `up_proj`, low-rank terms on `down_proj`)
as a llama.cpp LoRA adapter GGUF with a manifest beside it (`verify/arbiter.md` §4 row 32, §8, §9 decision 5).

What holds, and where:
- It is the owner's command and nothing an agent can reach: `level_mcp.py`, `level_session.py`, `level_routes.py` and
  `scientist_tools.py` do not import it (`scripts/check_level_trial.py` looks).
- An export goes to `$THEODOLITE_EXPORT_DIR`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/exports`, or to a path
  given. Refused: any path inside a git checkout (this one or another), `atlas_data/`, a registration folder and an
  agent session's folder. Nothing stops the owner naming some other folder a tool can write to.
- An adapter and its manifest are not written over unless asked (`overwrite=True`, `--force`).
- A term that only scales or silences is compiled from the weights of the base the adapter will be served over (a
  GGUF, a checkpoint folder, the loaded model): that is what "recompiled" means here. A term that sets or adds
  vectors, and a low-rank one, carries numbers worked on the edit's own base; over another base the file says so.
- `ffn_gate` and `ffn_up` adapters are marked unverified: only `ffn_down` (and `attn_output`) have been read back
  through llama.cpp's Gemma 4 graph.

What the adapter's keys say is what was checked, and no more:
- a trial's verdict goes into the file only for a trial of THIS edit at THIS dose, with whether it was of record, a
  dry run or a re-run, and whether the base it read is the base the adapter is compiled against;
- the implant label is "true", "false", "not checked" or "nothing in reach of the check", with where it was checked
  (at export, with the model at hand; or at the trial's registration);
- where the file's numbers came from is said term by term.

`verify_exported_lora` reads an adapter back and sets its numbers beside the edit's own arithmetic on the target,
worked without the compiler that made the file. It shows the file is the edit; it does not show what the edit does.
`llama_server_smoke` loads an adapter in a local `llama-server` and writes one prompt with it off and on: a check
that llama.cpp takes the file, not a measurement.

Layers count from 0.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from backend import level_book as lb
from backend import weight_edits as we
from backend.census import write_json_atomic

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"

EXPORT_VERSION = 1
EXPORT_ENV = "THEODOLITE_EXPORT_DIR"
LARGE_RANK_WARN = 256

GGUF_TENSOR_MAP = {
    "down_proj": "blk.{}.ffn_down.weight",
    "gate_proj": "blk.{}.ffn_gate.weight",
    "up_proj": "blk.{}.ffn_up.weight",
}
VERIFIED_LLAMA_CPP_WHICH = ("down_proj",)  # ffn_down (and attn_output) read back through llama.cpp's gemma4.cpp

PROVENANCE_NOTICE = (
    "Weights modified from Gemma 4 (Apache License 2.0) by Theodolite Levelling. "
    "Provenance is not assurance: any claim applies only to the base checkpoint measured, "
    "and an adapter served over another quantised GGUF must be re-measured on that object."
)


class ExportRefused(ValueError):
    """Plain-word refusal of an export that would go somewhere it should not, or say something it cannot."""


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


def _writers() -> Tuple[Any, Any]:
    """`scripts/export_control_vector.py` and `scripts/export_lora.py`, whose writer and keys an adapter shares. The
    scripts folder goes at the END of the path, and only when an export is asked for: importing this module moves
    nothing."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.append(str(SCRIPTS_DIR))
    import export_control_vector as ecv
    import export_lora as el

    return ecv, el


def exports_dir() -> Path:
    """Where an export goes when no path is given: `$THEODOLITE_EXPORT_DIR`, else
    `${XDG_DATA_HOME:-~/.local/share}/theodolite/exports`."""
    given = os.environ.get(EXPORT_ENV)
    if given:
        return Path(given).expanduser()
    data = os.environ.get("XDG_DATA_HOME")
    return (Path(data).expanduser() if data else Path.home() / ".local" / "share") / "theodolite" / "exports"


def refuse_unsafe_export_path(path: Path) -> Path:
    """Refuse an adapter or manifest path inside a git checkout (this one or any other: a folder with `.git` above
    it), `atlas_data/`, a registration folder, or an agent session's folder (`arbiter.md` row 32). Returns the path
    resolved. It cannot know every folder a tool can write to: it refuses the ones the instrument has."""
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
        if (anc / ".git").exists():
            raise ExportRefused(
                f"Exports never go into a git checkout ({anc} is one): a file of numbers taken from the model's "
                "weights is kept out of every repository."
            )
        if anc.name == "prereg" or ((anc / "SHA256SUMS").is_file() and (anc / "trial.json").is_file()):
            raise ExportRefused(
                f"Exports never go into a registration folder ({anc}): registration folders hold manifests and hashes only."
            )
        if (anc / "session.json").is_file() or (anc / "transcript.jsonl").is_file():
            raise ExportRefused(
                f"Exports never go into an agent session folder ({anc})."
            )
    return resolved


# --------------------------------------------------------------------------------------------- the base served over


class TargetBase:
    """The weights an adapter is compiled against, read on the CPU: a GGUF file, a checkpoint folder (or this
    machine's cached W4A16 checkpoint, `TargetBase.default_checkpoint()`), or a loaded model."""

    def __init__(
        self,
        *,
        gguf_path: Optional[Path] = None,
        weights_dir: Optional[Path] = None,
        lm: Optional[nn.Module] = None,
        reader: Any = None,
        label: Optional[str] = None,
    ) -> None:
        chosen = sum(x is not None for x in (gguf_path, weights_dir, lm, reader))
        if chosen != 1:
            raise ExportRefused("TargetBase requires exactly one of gguf_path, weights_dir, reader, or lm.")
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
            arch = gm.read_metadata(self._gguf_reader).get("general.architecture")
            if arch != "gemma4":
                raise ExportRefused(
                    f"{p.name} is a GGUF of architecture {arch!r}: an adapter of a Gemma 4 edit is compiled against "
                    "a Gemma 4 file."
                )
            self._gguf_tensors = {t.name: t for t in self._gguf_reader.tensors}
            self.record = label or f"gguf {p.name} (sha256:{self.gguf_sha256[:16]})"
        elif weights_dir is not None or reader is not None:
            from backend.weight_reader import WeightReader, WeightReadError

            self.kind = "checkpoint"
            if reader is not None:
                self._weight_reader = reader
            else:
                p = Path(weights_dir).expanduser().resolve()
                try:
                    self._weight_reader = WeightReader(p)
                except (WeightReadError, OSError, ValueError) as exc:
                    raise ExportRefused(f"{p} cannot be read as a checkpoint folder: {exc}") from None
            wid = getattr(self._weight_reader, "weights_id", None)
            self.record = label or (f"checkpoint {Path(self._weight_reader.root).name}"
                                    + (f" (weights {str(wid)[:16]})" if wid else ""))
        else:
            self.kind = "module"
            self._lm = lm
            self.record = label or "loaded module"

    @classmethod
    def default_checkpoint(cls) -> "TargetBase":
        """This machine's cached W4A16 checkpoint, read from disk with no model loaded."""
        from backend.weight_reader import WeightReader, WeightReadError

        try:
            return cls(reader=WeightReader.from_default())
        except (WeightReadError, OSError, ValueError) as exc:
            raise ExportRefused(f"The W4A16 checkpoint is not in this machine's cache ({exc}): give --gguf FILE or "
                                "--weights-dir FOLDER.") from None

    def pin(self, layer: int, which: str) -> Optional[str]:
        """The pin an edit names this linear's weights by (`weight_edits.tensor_pin`), where this base has one: a
        checkpoint and a loaded model do, a GGUF is another object and has none."""
        name = we.linear_name(int(layer), which)
        try:
            if self.kind == "checkpoint":
                return str(self._weight_reader.tensor_pin(name))
            if self.kind == "module":
                return str(we.tensor_pin(self._lm.get_submodule(name)))
        except (we.Refused, ValueError, AttributeError, KeyError):
            return None
        return None

    def is_base_of(self, edit: we.Edit) -> Optional[bool]:
        """Whether every linear the edit touches has, on this base, the weights the edit was built on (by pin). None
        for a GGUF: its numbers are another quantisation's, and no pin says how they relate."""
        if self.kind == "gguf":
            return None
        pins = (edit.base or {}).get("tensors") or {}
        for t in edit.terms:
            if not pins.get(t.linear) or self.pin(t.layer, t.which) != pins.get(t.linear):
                return False
        return True

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
        try:
            code = type_code(t.tensor_type)
        except (ValueError, KeyError):
            code = None  # a type this module's own unpacker does not know: read whole, through gguf-py
        bs = GGML_TYPES[code][1] if code in GGML_TYPES else 1
        try:
            if bs > 1:
                lin = GGUFLinear(
                    gm._blocks(t, torch.device("cpu")), t.tensor_type, int(in_f), int(out_f),
                    dtype=torch.float32, gguf_name=t.name,
                )
                got = lin.unpacked_cols(ix) if axis == "cols" else lin.unpacked_rows(ix)
                return got.float().cpu().numpy()
            dense = gm._dense(t, torch.device("cpu"), torch.float32)  # [out, in]
        except (ValueError, KeyError, RuntimeError) as exc:
            raise ExportRefused(f"{t.name} of the target GGUF cannot be read ({exc}).") from None
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


# --------------------------------------------------------------------------------------------- the implant label


def implant_check(
    edit: we.Edit,
    lm: Optional[nn.Module] = None,
    W_U: Optional[torch.Tensor] = None,
    *,
    dose: float = 1.0,
) -> Dict[str, Any]:
    """The kernel's implant-shape label (`weight_edits.implant_shape`) when the model and its unembedding are at
    hand; otherwise `{"checked": False}`, which an adapter writes as "not checked" and never as "false"."""
    if lm is None or W_U is None:
        return {"checked": False, "flagged": False, "note": "the model and its unembedding were not at hand"}
    return we.implant_shape(edit, lm, W_U, dose=dose)


def implant_words(imp: Mapping[str, Any]) -> str:
    """The implant label as an adapter's key says it: "true", "false", "not checked" (the model was not at hand), or
    "nothing in reach of the check" (it ran and the edit writes nothing to the stream in the last tenth of the
    layers, which is all it looks at: not a finding that the edit is clear)."""
    if not imp.get("checked"):
        return "not checked"
    if imp.get("flagged"):
        return "true"
    return "false" if int(imp.get("examined") or 0) > 0 else "nothing in reach of the check"


# --------------------------------------------------------------------------------------------- an edit, off the model


def _kept_to_id(terms: List[we.Term], base: Mapping[str, Any], want_id: str, what: str, note: str = "") -> we.Edit:
    terms = sorted(terms, key=lambda t: (t.linear, t.kind))
    got = we._edit_id(dict(base), terms)
    if got != want_id:
        raise ExportRefused(
            f"{what} does not rebuild the edit it names: its terms hash to {got[:12]}, and it says {str(want_id)[:12]}. "
            "An edit is exported as the kernel built and hashed it, or not at all."
        )
    return we.Edit(want_id, tuple(terms), dict(base), note, {})


def edit_from_saved(folder: Path) -> we.Edit:
    """A saved edit (`weight_edits.save`: `manifest.json` and, for carried arrays, `payload.safetensors`) rebuilt
    with no model loaded, and kept to its id: the id hashes the base's pins and every term's arrays, so a payload or
    a manifest changed since the kernel saved it is refused."""
    folder = Path(folder)
    try:
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExportRefused(f"{folder} is not a saved edit ({exc}).") from None
    if not isinstance(man, dict) or man.get("format") != we.FORMAT or man.get("version") != we.VERSION:
        raise ExportRefused(f"{folder / 'manifest.json'} is not a saved edit of this kernel's format.")
    payload: Dict[str, np.ndarray] = {}
    if (folder / "payload.safetensors").is_file():
        from safetensors.numpy import load_file

        payload = load_file(str(folder / "payload.safetensors"))
    terms: List[we.Term] = []
    try:
        for t in man["terms"]:
            layer, which = we._parse(t["linear"])
            if t["kind"] in ("cols", "rows"):
                idx_key = "columns" if t["kind"] == "cols" else "rows"
                idx = we._ro([int(n) for n in t[idx_key]], np.int64)
                op = we._ro([we.OPS.index(o) for o in t["ops"]], np.uint8)
                fac = we._ro([float(f) for f in t["factors"]], np.float32)
                vec = we._ro(payload[t["vectors"]], np.float32) if t.get("vectors") else None
                terms.append(we.Term(t["linear"], t["kind"], layer, which, idx=idx, op=op, factor=fac, vectors=vec))
            else:
                terms.append(we.Term(t["linear"], "lowrank", layer, which, A=we._ro(payload[t["A"]], np.float32),
                                     B=we._ro(payload[t["B"]], np.float32)))
    except (KeyError, TypeError, ValueError, we.Refused) as exc:
        raise ExportRefused(f"{folder} cannot be read as a saved edit ({type(exc).__name__}: {exc}).") from None
    return _kept_to_id(terms, man.get("base") or {}, str(man.get("id")), f"The saved edit in {folder}",
                       str(man.get("note") or ""))


def edit_from_registration(edit_manifest: Mapping[str, Any], *, payload_dir: Optional[Path] = None) -> we.Edit:
    """A registered edit (`edit_manifest.json` of a trial's folder) rebuilt with no model loaded, and kept to the
    id the registration names. Every term rebuilt: from the manifest's recipe (scales and silences alone, which is
    all a rebuilt recipe holds). A carried term: from the saved edit in `payload_dir` (or the saved edits folder, by
    id), which must be the registered edit by id and, where the registration recorded it, by its payload's sha256."""
    want_id = str(edit_manifest.get("edit_id"))
    base = edit_manifest.get("base") or {}
    if not edit_manifest.get("all_rebuilt"):
        try:
            saved = we._find_saved(Path(payload_dir) if payload_dir is not None else want_id)
        except (we.Refused, OSError, ValueError) as exc:
            raise ExportRefused(f"The registered edit carries arrays, and its saved edit was not found ({exc}): give "
                                "--payload FOLDER.") from None
        got = edit_from_saved(saved)
        if got.id != want_id:
            raise ExportRefused(f"The saved edit in {saved} is {got.short()}, not the registered edit {want_id[:12]}.")
        want_p, pfile = edit_manifest.get("payload_sha256"), saved / "payload.safetensors"
        if want_p and pfile.is_file() and _sha_file(pfile) != want_p \
                and we.edit_provenance(got)["payload_sha256"] != want_p:
            raise ExportRefused(f"Payload {pfile} is not the one registered (payload_sha256 {str(want_p)[:12]}).")
        return got
    recipe = edit_manifest.get("rebuilt_recipe")
    if not isinstance(recipe, Mapping):
        raise ExportRefused("The registration's edit manifest says every term is rebuilt and holds no recipe.")
    try:
        specs = we.from_recipe(dict(recipe), Path("."))
    except (we.Refused, ValueError) as exc:
        raise ExportRefused(f"The registered recipe cannot be read: {exc}") from None
    if isinstance(specs, (str, Path)):
        raise ExportRefused("A registered recipe is terms, not a pointer to a saved edit.")
    terms: List[we.Term] = []
    for s in specs:
        if s.kind not in ("cols", "rows") or s.op != "scale":
            raise ExportRefused('A rebuilt recipe holds scales and silences alone; this one holds a '
                                f'"{s.op or s.kind}" term.')
        neurons = [int(n) for n in s.neurons]
        f = s.factor
        fac = [float(f)] * len(neurons) if not isinstance(f, (list, tuple)) else [float(x) for x in f]
        if len(fac) != len(neurons) or len(set(neurons)) != len(neurons):
            raise ExportRefused("A recipe term gives one factor a neuron, and names no neuron twice.")
        order = np.argsort(np.asarray(neurons, dtype=np.int64), kind="stable")
        terms.append(we.Term(we.linear_name(int(s.layer), s.which), s.kind, int(s.layer), s.which,
                             idx=we._ro(np.asarray(neurons, dtype=np.int64)[order], np.int64),
                             op=we._ro(np.zeros(len(neurons), dtype=np.uint8), np.uint8),
                             factor=we._ro(np.asarray(fac, dtype=np.float32)[order] + np.float32(0.0), np.float32)))
    return _kept_to_id(terms, base, want_id, "The registration's recipe", str(edit_manifest.get("note") or ""))


# --------------------------------------------------------------------------------------------- compiling an adapter


def term_change(t: we.Term, target: TargetBase, dose: float) -> np.ndarray:
    """What one column or row term adds to the target's weights at `dose`, from the term's own arithmetic on the
    target's own numbers (float64 [K, width]): a scaled entry `d (f - 1)` times the target's slice, a set entry
    `d (v - slice)`, an added entry `d v`. It is `weight_edits._term_delta_f64` with the slices read from the target,
    and takes nothing from `build_lora_tensors`."""
    idx = [int(n) for n in t.idx]
    W = (target.dense_rows(t.layer, t.which, idx) if t.kind == "rows"
         else target.dense_cols(t.layer, t.which, idx)).astype(np.float64)
    dW = np.zeros_like(W)
    is_scale = t.op == 0
    if is_scale.any():
        dW[is_scale] = (t.factor[is_scale].astype(np.float64) - 1.0)[:, None] * W[is_scale]
    if t.vectors is not None:
        V = t.vectors.astype(np.float64)
        if V.shape != W.shape:
            raise ExportRefused(f"{t.linear}: the edit's vectors are {list(V.shape)} and the target's slices {list(W.shape)}.")
        is_set, is_add = t.op == 1, t.op == 2
        if is_set.any():
            dW[is_set] = V[is_set] - W[is_set]
        if is_add.any():
            dW[is_add] = V[is_add]
    return float(dose) * dW


def build_lora_tensors(
    edit: we.Edit,
    target: TargetBase,
    *,
    dose: float = 1.0,
) -> Dict[str, Any]:
    """Compile `edit` at `dose` into llama.cpp LoRA pairs `(A [r, n_in], B [n_out, r])`, float32, against `target`.

    A column or row term that only scales reads its slices from `target`, so over a GGUF the adapter scales that
    file's own numbers. A term with vectors (set, add) and a low-rank term carry numbers worked on the edit's own
    base; whenever `target` is not that base (a GGUF, or a checkpoint whose pins differ) a warning says so. `gate_proj`
    and `up_proj` adapters are listed as unverified in llama.cpp."""
    d = float(dose)
    if not math.isfinite(d) or d == 0.0:
        raise ExportRefused(f"Export dose must be finite and non-zero; got {dose!r}.")
    same_base = target.is_base_of(edit)

    parts_a: Dict[str, List[np.ndarray]] = {}
    parts_b: Dict[str, List[np.ndarray]] = {}
    warnings: List[str] = []
    unverified: List[str] = []
    made_of: List[str] = []
    recompiled_all = True
    seen: Dict[Tuple[str, str], set] = {}

    def carried_note(tname: str, what: str) -> None:
        if same_base is not True:
            warnings.append(
                f"{tname}: {what} worked on {edit.base.get('model')} "
                f"{'are' if what.endswith('s') else 'is'} applied onto {target.record}"
                + (", another quantisation" if target.kind == "gguf" else ", whose weights are not the edit's base")
                + "; re-measure there."
            )

    for t in edit.terms:
        if t.which not in GGUF_TENSOR_MAP:
            raise ExportRefused(f"{t.linear} has no llama.cpp adapter tensor this exporter writes.")
        tname = GGUF_TENSOR_MAP[t.which].format(int(t.layer))
        if t.which not in VERIFIED_LLAMA_CPP_WHICH and tname not in unverified:
            unverified.append(tname)
        n_out, n_in = target.shape(t.layer, t.which)

        if t.kind in ("cols", "rows"):
            width = n_in if t.kind == "cols" else n_out
            idx = [int(n) for n in t.idx]
            if min(idx) < 0 or max(idx) >= width:
                raise ExportRefused(f"{tname} has {width} {'inputs' if t.kind == 'cols' else 'outputs'} on "
                                    f"{target.record}; the edit names {max(idx)}.")
            was = seen.setdefault((tname, t.kind), set())
            if len(set(idx)) != len(idx) or was & set(idx):
                raise ExportRefused(f"{tname}: a {'column' if t.kind == 'cols' else 'row'} is named twice in this "
                                    "edit: one change a column, as the kernel builds an edit.")
            was |= set(idx)
            delta = term_change(t, target, d).astype(np.float32)
            if t.vectors is not None:
                recompiled_all = False
                carried_note(tname, f"carried {'column' if t.kind == 'cols' else 'row'} vectors")
            keep = np.max(np.abs(delta), axis=1) > 0.0
            if not keep.any():
                continue
            kept_idx = [idx[i] for i in range(len(idx)) if keep[i]]
            if t.kind == "cols":
                # column n of the change: B[:, k] is the change, A[k, :] picks input n
                B_part = np.ascontiguousarray(delta[keep].T, dtype=np.float32)  # [n_out, r]
                A_part = np.zeros((len(kept_idx), n_in), dtype=np.float32)
                A_part[np.arange(len(kept_idx)), kept_idx] = 1.0
                made_of.append(f"{tname}: lora_b holds the change to {len(kept_idx)} column(s), lora_a ones and "
                               "noughts that pick the inputs")
            else:
                # row n of the change: A[k, :] is the change, B[:, k] picks output n
                A_part = np.ascontiguousarray(delta[keep], dtype=np.float32)  # [r, n_in]
                B_part = np.zeros((n_out, len(kept_idx)), dtype=np.float32)
                B_part[kept_idx, np.arange(len(kept_idx))] = 1.0
                made_of.append(f"{tname}: lora_a holds the change to {len(kept_idx)} row(s), lora_b ones and "
                               "noughts that pick the outputs")
            parts_a.setdefault(tname, []).append(A_part)
            parts_b.setdefault(tname, []).append(B_part)
        else:  # low-rank
            recompiled_all = False
            if t.A.shape[1] != n_in or t.B.shape[0] != n_out:
                raise ExportRefused(
                    f"{tname} is [{n_out}, {n_in}] on {target.record}, low-rank edit has A={list(t.A.shape)}, B={list(t.B.shape)}."
                )
            parts_a.setdefault(tname, []).append(np.ascontiguousarray(t.A, dtype=np.float32))
            parts_b.setdefault(tname, []).append(np.ascontiguousarray(d * t.B, dtype=np.float32))
            made_of.append(f"{tname}: lora_a and lora_b are the edit's own low-rank pair (rank {t.A.shape[0]}), "
                           "lora_b times the dose")
            carried_note(tname, f"a carried low-rank term (rank {t.A.shape[0]})")

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
        "served_base_is_edit_base": same_base,
        "unverified_tensors": sorted(unverified),
        "made_of": made_of,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------------------------- what a trial lends


def trial_standing(edit: we.Edit, dose: float, target: TargetBase, same_base: Optional[bool],
                   trial_result: Optional[Mapping[str, Any]], book_path: Optional[Path]) -> Dict[str, Any]:
    """What an adapter may say of a trial: nothing when none is given; otherwise the trial must be of this edit at
    this dose (refused if not), and the words say what kind it was (of record, a dry run, a volunteer's re-run),
    whether the book it names holds its verdict line, and whether the base it read is the base served."""
    if trial_result is None:
        return {"verdict": "none given", "kind": "none", "trial_sha256": "none", "book_seq": None, "book_hash": None,
                "book": "no trial given", "applies_to_served_base": "no trial given", "book_before_it": "not given"}
    if not isinstance(trial_result, Mapping):
        raise ExportRefused("A trial result is the trial's own trial_result.json.")
    cfg = trial_result.get("trial_config") if isinstance(trial_result.get("trial_config"), Mapping) else {}
    if cfg.get("edit_id") != edit.id:
        raise ExportRefused(
            f"The trial result given is of edit {str(cfg.get('edit_id'))[:12]}, not of the edit exported "
            f"({edit.short()}): an adapter carries the verdict of its own edit's trial or of none."
        )
    try:
        t_dose = float(cfg.get("main_dose"))
    except (TypeError, ValueError):
        t_dose = float("nan")
    if not math.isfinite(t_dose) or t_dose != float(dose):
        raise ExportRefused(
            f"The trial read the edit at dose {cfg.get('main_dose')!r}; the adapter is asked at dose {float(dose):g}. "
            "A trial's verdict goes with the dose it read."
        )
    verdict = str(trial_result.get("verdict"))
    if trial_result.get("rerun_of"):
        kind = "a volunteer's re-run"
    elif trial_result.get("of_record"):
        kind = "of record"
    else:
        kind = "a dry run"
    bk = trial_result.get("book") if isinstance(trial_result.get("book"), Mapping) else {}
    seq, vhash = bk.get("verdict_seq"), bk.get("verdict_hash")
    book = Path(book_path) if book_path is not None else (Path(str(bk.get("path"))) if bk.get("path") else None)
    if book is None or not book.is_file() or not isinstance(seq, int):
        book_words = "not looked up: the trial's book was not at hand"
    else:
        ent = next(lb.entries(book, start=int(seq), limit=1), None)
        if ent is None or ent.get("hash") != vhash:
            raise ExportRefused(
                f"The trial result names line {seq} of {book.name} as its verdict, and that line is not it: the "
                "result and the book do not belong together."
            )
        book_words = f"line {seq} of {book.name} is the trial's verdict"
    meta = trial_result.get("meta") if isinstance(trial_result.get("meta"), Mapping) else {}
    read_on = str(meta.get("weights_id") or "unknown")
    if same_base is True and target.kind != "gguf":
        applies = "yes: the adapter is compiled against the base the trial read"
    else:
        applies = (f"no: the trial read {read_on[:16]}, and the adapter is compiled against {target.record}"
                   + (", another quantisation" if target.kind == "gguf" else ""))
    fj = trial_result.get("fresh_judged") if isinstance(trial_result.get("fresh_judged"), Mapping) else {}
    before = (f"practice readings {fj.get('practice_count', 'unknown')}, earlier fresh readings "
              f"{len(meta.get('prior_verdicts') or [])}, earlier trials {len(meta.get('prior_trials') or [])}, as the "
              "trial's book had them")
    return {"verdict": verdict, "kind": kind, "trial_sha256": str(trial_result.get("trial_sha256") or "none"),
            "book_seq": seq, "book_hash": vhash, "book": book_words, "applies_to_served_base": applies,
            "book_before_it": before}


# --------------------------------------------------------------------------------------------- writing an adapter


def export_lora_adapter(
    edit: we.Edit,
    target: TargetBase,
    *,
    out_path: Optional[Path] = None,
    dose: float = 1.0,
    overwrite: bool = False,
    trial_result: Optional[Mapping[str, Any]] = None,
    book_path: Optional[Path] = None,
    saved_edit_dir: Optional[Path] = None,
    lm_for_implant: Optional[nn.Module] = None,
    W_U_for_implant: Optional[torch.Tensor] = None,
    implant_at_registration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a llama.cpp LoRA `.gguf` adapter and its `.manifest.json`.

    The licence and origin keys are the house's (`scripts/export_lora.write_lora`: `general.license`,
    `general.license.link`, `general.base_model.*`, `theodolite.derived_from`), with `theodolite.levelling.*` beside
    them. `trial_result`: the `trial_result.json` of a trial of this edit at this dose (refused otherwise,
    `trial_standing`). The implant label is worked here when `lm_for_implant` and `W_U_for_implant` are given, else
    taken from `implant_at_registration` (what the trial's registration recorded), else "not checked"."""
    ecv, el = _writers()

    if out_path is None:
        base_dir = exports_dir()
        fname = f"theodolite-levelling-{edit.short()}-d{float(dose):g}-derived-from-gemma.gguf"
        dst = refuse_unsafe_export_path(base_dir / fname)
    else:
        dst = Path(out_path).expanduser()
        if dst.is_dir():
            raise ExportRefused(f"{dst} is a folder: name the adapter file to write (FOLDER/NAME.gguf).")
        if dst.suffix.lower() != ".gguf":
            dst = dst.with_suffix(".gguf")
        dst = refuse_unsafe_export_path(dst)

    manifest_path = dst.with_name(dst.stem + ".manifest.json")
    if not overwrite:
        if dst.exists():
            raise ExportRefused(f"Export file already exists: {dst} (pass --force / overwrite=True to replace).")
        if manifest_path.exists():
            raise ExportRefused(f"Export manifest already exists: {manifest_path} (pass --force / overwrite=True to replace).")

    compiled = build_lora_tensors(edit, target, dose=dose)
    prov = we.edit_provenance(edit, saved_dir=saved_edit_dir)
    if lm_for_implant is not None and W_U_for_implant is not None:
        imp = {**implant_check(edit, lm_for_implant, W_U_for_implant, dose=dose), "checked_at": "export"}
    elif isinstance(implant_at_registration, Mapping) and implant_at_registration.get("checked"):
        imp = {**dict(implant_at_registration), "checked_at": "the trial's registration"}
    else:
        imp = {**implant_check(edit), "checked_at": "never"}
    trial = trial_standing(edit, dose, target, compiled["served_base_is_edit_base"], trial_result, book_path)
    same = compiled["served_base_is_edit_base"]
    same_words = {True: "yes", False: "no", None: "another object (a GGUF)"}[same]

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
        "levelling.served_base_is_edit_base": same_words,
        "levelling.provenance": prov["provenance"],
        "levelling.recompiled_from_recipe": "true" if compiled["recompiled_all"] else "false",
        "levelling.unverified_tensors": json.dumps(compiled["unverified_tensors"]),
        "levelling.trial_verdict": trial["verdict"],
        "levelling.trial_kind": trial["kind"],
        "levelling.trial_sha256": trial["trial_sha256"],
        "levelling.trial_applies_to_served_base": trial["applies_to_served_base"],
        "levelling.book_seq": str(trial["book_seq"]) if trial["book_seq"] is not None else "none",
        "levelling.book_hash": str(trial["book_hash"] or "none"),
        "levelling.book": trial["book"],
        "levelling.book_before_it": trial["book_before_it"],
        "levelling.implant_flagged": implant_words(imp),
        "levelling.implant_checked_at": str(imp["checked_at"]),
        "levelling.warnings": json.dumps(compiled["warnings"]),
    }
    derived = (f"this file's tensors are derived from the weights of {ecv.ORIGIN} by a Theodolite Levelling edit: "
               + "; ".join(compiled["made_of"]) + ". The weights read are named under theodolite.weights")

    dst.parent.mkdir(parents=True, exist_ok=True)
    el.write_lora(dst, compiled["tensors"], record, derived=derived)
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
            "is_edit_base": same,
        },
        "recompiled_from_recipe": bool(compiled["recompiled_all"]),
        "ranks": compiled["ranks"],
        "made_of": compiled["made_of"],
        "unverified_llama_cpp_tensors": compiled["unverified_tensors"],
        "warnings": compiled["warnings"],
        "implant_check": imp,
        "edit_provenance": prov,
        "trial": trial,
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
        "served_base_is_edit_base": same,
        "unverified_llama_cpp_tensors": compiled["unverified_tensors"],
        "warnings": compiled["warnings"],
        "implant_check": imp,
        "trial": trial,
        "manifest": companion,
    }


# --------------------------------------------------------------------------------------------- reading one back


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
    """Read an adapter back and check it.

    Always: the keys llama.cpp and the house ask of an adapter are there (type, architecture, licence, origin,
    `theodolite.levelling.edit_id`), and the manifest beside it, when there is one, names this file's sha256 and
    this edit.

    With `edit` and `target`: the file IS the edit on that target. Its `lora_b @ lora_a`, multiplied out, is set
    beside the edit's own arithmetic on the target's numbers (`term_change`, and a low-rank term's own pair), which
    does not go through the compiler that wrote the file: every touched column and row, the columns and rows the
    edit does not touch (which must be nought), and what the whole does to a few random vectors. `max_abs_err` is
    the largest difference found. This shows the file is the edit; what the edit does is a reading's to show."""
    _ecv, el = _writers()

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
        _rec("file_names_this_edit", meta.get("theodolite.levelling.edit_id") == edit.id,
             {"file": str(meta.get("theodolite.levelling.edit_id"))[:12], "edit": edit.short()})
        d = float(dose if dose is not None else float(meta.get("theodolite.levelling.dose", "1.0")))
        by_tensor: Dict[str, List[we.Term]] = {}
        for t in edit.terms:
            by_tensor.setdefault(GGUF_TENSOR_MAP[t.which].format(int(t.layer)), []).append(t)
        _rec("tensor_names_match", set(tensors) <= set(by_tensor),
             {"got": sorted(tensors), "edit touches": sorted(by_tensor)})
        worst = 0.0
        rng = np.random.default_rng(2026)
        for tname, terms in sorted(by_tensor.items()):
            n_out, n_in = target.shape(terms[0].layer, terms[0].which)
            A, B = tensors.get(tname, (np.zeros((0, n_in), np.float32), np.zeros((n_out, 0), np.float32)))
            A64, B64 = A.astype(np.float64), B.astype(np.float64)
            changes = [(t, None if t.kind == "lowrank" else term_change(t, target, d)) for t in terms]
            cols_touched = sorted({int(i) for t in terms if t.kind == "cols" for i in t.idx})
            rows_touched = sorted({int(i) for t in terms if t.kind == "rows" for i in t.idx})
            lowrank = any(t.kind == "lowrank" for t in terms)

            def want_of(x: np.ndarray) -> np.ndarray:
                """What the edit's change does to inputs `x` ([m, n_in]), term by term, from the terms themselves."""
                out = np.zeros((x.shape[0], n_out))
                for t, delta in changes:
                    ix = [int(i) for i in t.idx] if t.kind != "lowrank" else []
                    if t.kind == "cols":
                        out += x[:, ix] @ delta  # delta [K, n_out]
                    elif t.kind == "rows":
                        out[:, ix] += x @ delta.T  # delta [K, n_in]
                    else:
                        out += (x @ t.A.astype(np.float64).T) @ (d * t.B.astype(np.float64)).T
                return out

            def gap(x: np.ndarray) -> float:
                return float(np.max(np.abs((x @ A64.T) @ B64.T - want_of(x)))) if x.size else 0.0

            worst = max(worst, gap(rng.standard_normal((6, n_in))))
            # every touched input on its own: column n of the file's change is the edit's change to column n
            for at in range(0, len(cols_touched), 64):
                part = cols_touched[at:at + 64]
                one_hot = np.zeros((len(part), n_in))
                one_hot[np.arange(len(part)), part] = 1.0
                worst = max(worst, gap(one_hot))
            # every touched output on its own: row n of the file's change is the edit's change to row n
            for t, delta in changes:
                if t.kind == "rows":
                    for k, n in enumerate(int(i) for i in t.idx):
                        worst = max(worst, float(np.max(np.abs(B64[n, :] @ A64 - delta[k]))))
            if not lowrank and A64.size and B64.size:
                # nothing the edit does not touch is changed: with column terms alone lora_a is nought off the touched
                # inputs; with row terms alone lora_b is nought off the touched outputs
                if cols_touched and not rows_touched:
                    off = np.ones(n_in, dtype=bool)
                    off[cols_touched] = False
                    worst = max(worst, float(np.max(np.abs(A64[:, off]))) if off.any() else 0.0)
                if rows_touched and not cols_touched:
                    off = np.ones(n_out, dtype=bool)
                    off[rows_touched] = False
                    worst = max(worst, float(np.max(np.abs(B64[off, :]))) if off.any() else 0.0)
        max_err = worst
        _rec("file_is_the_edit_on_the_target", worst <= tol, {"max_abs_err": worst, "tol": tol})

    ok_all = all(c["ok"] for c in checks)
    return {
        "ok": ok_all,
        "gguf_path": str(p),
        "tensors": {k: {"A_shape": list(v[0].shape), "B_shape": list(v[1].shape)} for k, v in tensors.items()},
        "max_abs_err": max_err,
        "checks": checks,
        "meta_summary": {k: v for k, v in meta.items() if k.startswith(("general.", "adapter.", "theodolite."))},
    }


def llama_server_smoke(
    llama_server_bin: Path,
    model_gguf: Path,
    adapter_gguf: Path,
    prompt: str,
    *,
    scale: float = 1.0,
    n_predict: int = 16,
    port: int = 18089,
    timeout_s: float = 600.0,
    request_timeout_s: float = 300.0,
) -> Dict[str, Any]:
    """A smoke test that llama.cpp takes an adapter: start a local `llama-server --lora-init-without-apply` over
    `model_gguf`, write `prompt` greedily with the adapter at scale 0 and at `scale`, and say whether the two texts
    differ. It is not a measurement of anything a brief names: one prompt, a few pieces, no control. Refused when
    something already answers on the port (it would be another server's text). The server's own output goes to a
    file, so that it cannot stall on an unread pipe."""
    exe = Path(llama_server_bin)
    if not exe.is_file():
        raise ExportRefused(f"llama-server binary not found: {exe}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", int(port))) == 0:
            raise ExportRefused(f"Something already answers on :{int(port)}: its text would be taken for this server's.")
    cmd = [
        str(exe), "-m", str(model_gguf), "--lora", str(adapter_gguf), "--lora-init-without-apply",
        "--host", "127.0.0.1", "--port", str(int(port)), "-ngl", "0", "-c", "512",
    ]
    log = tempfile.NamedTemporaryFile(prefix="llama-server-", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{int(port)}"
    try:
        deadline = time.monotonic() + float(timeout_s)
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = Path(log.name).read_text(encoding="utf-8", errors="replace")[-600:]
                raise ExportRefused(f"llama-server exited early ({proc.returncode}): {tail}")
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.25)
        if not ready:
            raise ExportRefused(f"llama-server did not become ready within {timeout_s:g} s (its output: {log.name}).")
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
            with urllib.request.urlopen(req, timeout=float(request_timeout_s)) as r:
                return json.loads(r.read().decode("utf-8"))

        out_base = _complete(0.0)
        out_edit = _complete(float(scale))
        return {
            "ok": True,
            "what": "a smoke test that llama.cpp takes the adapter: one prompt, not a measurement",
            "adapters": adapters,
            "base_content": out_base.get("content", ""),
            "edited_content": out_edit.get("content", ""),
            "changed": out_base.get("content", "") != out_edit.get("content", ""),
            "server_log": log.name,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
