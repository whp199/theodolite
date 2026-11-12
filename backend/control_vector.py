"""A push found in the atlas, written from the running server as a llama.cpp control vector
(`backend/control_vector.py`): what POST /api/atlas/control_vector does, behind the push tab's "control vector" link and
the scientist's `export_control_vector` tool.

The file is scripts/export_control_vector.py's, by its own code (`pushes`, `write_control_vector`, the checks of
scripts/check_control_vector.py and the mapping of ARCHITECTURE.md §6): each entry `dose x p50_l x u`, summed over a
layer's entries, float32 under `direction.<l>` (l counted from 0), which llama.cpp adds to the stream leaving layer l,
after the layer scalar, where the instrument adds its push. What differs from the script is only where the numbers are
read: the script reads the checkpoint's files and the stats index from disk; this reads the model the server holds
(`atlas_ops.Weights`: the vault's edits in force are in the direction, and the file says so) and the push's unit as
the atlas's own pushes take it (`Atlas.scales`: the stats index, or until it is built the study's texts, said in the
file), so the file is the push the person or the scientist just ran. Only "add" pushes of single neurons go into a
control vector (a mul, a head's scale and a silenced block are scripts/export_lora.py's), and none at the first layer,
which llama.cpp's loader refuses.

Files are written into `<atlas folder>/exports/` (`exports_dir`) under a name of letters, digits, `.`, `_` and `-`
ending in .gguf, a file of that name written over; GET /api/atlas/exports/<name> hands one to the page as a download.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ROOT_DIR / "scripts" / "export_control_vector.py"
NAME_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
# what the file says of itself beside the script's own notes on the push, the start mark and SCALE
CAVEAT = ("the doses are the atlas's add doses as they were pushed there: a share of the stream's median length leaving "
          "each layer, the start mark left unpushed unless the push was run with spare_bos false. llama.cpp pushes the "
          "start mark too, and adds float32 where the instrument adds bfloat16, into a GGUF's own quantisation of the "
          "weights: on Gemma-4-31B two neurons' pushes carried over at SCALE 1 and broke the text at 2 where the "
          "instrument's held (ARCHITECTURE.md §6). A push this size is usually more than the neuron can write by itself, "
          "so the file carries the direction's effect, as tested, not the neuron's own")


def exports_dir() -> Path:
    from backend.atlas_index import atlas_dir

    return atlas_dir() / "exports"


def script() -> Any:
    """scripts/export_control_vector.py as a module, loaded once under the name the other scripts import it by."""
    name = "export_control_vector"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, SCRIPT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return mod


class LiveColumns:
    """The down columns and post-feed-forward norms of the model the server holds, as the script's `pushes` asks for
    them and as the atlas's own pushes read them (`atlas_ops.Weights.down_cols`, `.norm`: the vault's edits included)."""

    def __init__(self, weights: Any, card: Dict[str, Any]) -> None:
        self.w = weights
        self.n_layers = int(weights.n_layers)
        self.card = card

    def width(self, layer: int) -> int:
        from backend.atlas_ops import ffn_width

        return int(ffn_width(self.w.lm, int(layer)))

    def columns(self, layer: int, neurons: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        cols = self.w.down_cols(int(layer), [int(n) for n in neurons]).float().cpu()
        return cols, self.w.norm(int(layer), "post_feedforward").float().cpu()

    def record(self) -> str:
        return "the model the server held: " + json.dumps(self.card, ensure_ascii=False, sort_keys=True, default=str)


def file_name(name: Optional[str], entries: Sequence[Any]) -> str:
    """The file's name: the one asked for (letters, digits, `.`, `_`, `-`; .gguf added when it is not its ending), else
    one that says what is in it, in the pages' count: theodolite-page36-n1095-dose-0.35.gguf for one neuron (dose0.35
    the other way: no sign but a minus, so every name is one GET /exports takes), and for a set its size and a short
    hash of its entries."""
    if name is not None and str(name).strip():
        stem = str(name).strip()
        stem = stem[:-5] if stem.lower().endswith(".gguf") else stem
        if not NAME_OK.fullmatch(stem) or ".." in stem:
            raise ValueError("A control vector's file name is letters, digits, '.', '_' and '-' (at most 80, beginning with "
                             f"a letter or a digit), not {str(name)!r}.")
        return stem + ".gguf"
    if len(entries) == 1:
        e = entries[0]
        return f"theodolite-page{e.layer + 1}-n{e.neuron}-dose{e.dose:g}.gguf"
    key = hashlib.sha256(json.dumps([[e.layer, e.neuron, e.dose] for e in entries]).encode()).hexdigest()[:8]
    return f"theodolite-set-of-{len(entries)}-{key}.gguf"


def entries_of(pushes: Sequence[Any]) -> List[Any]:
    """The script's entries from checked pushes (`atlas_ops.Push`, layers from 0): each an add of one neuron."""
    ecv = script()
    out = []
    for p in pushes:
        if p.neuron is None or p.mode != "add":
            raise ValueError('A control vector carries only "add" pushes of single neurons: llama.cpp adds one vector a '
                             f"layer to the stream. A {p.mode} push, a head's scale and a silenced block go out as a LoRA "
                             "adapter or a copy of the model (scripts/export_lora.py).")
        out.append(ecv.Entry(int(p.layer), int(p.neuron), float(p.dose)))
    if not out:
        raise ValueError("Name the neurons to export, each with its dose.")
    return out


def build(weights: Any, scales: Dict[str, Any], model: str, card: Dict[str, Any], edits: Sequence[Tuple[int, int]],
          pushes: Sequence[Any], name: Optional[str] = None, folder: Optional[Path] = None) -> Dict[str, Any]:
    """Write the control vector of `pushes` (checked `atlas_ops.Push`es) and say what was written: the file, each
    entry's layer, neuron, dose and p50, each direction's length, and how llama.cpp loads it. `scales` is
    `Atlas.scales()` (its p50 and where that was read), `card` the weights' card, `edits` the neurons a vault edit is in
    force on. A ValueError says in plain words what cannot be exported."""
    ecv = script()
    entries = entries_of(pushes)
    source = LiveColumns(weights, card)
    ecv.check_entries(entries, source.n_layers, source.width)
    p50 = [float(x) for x in scales["p50"]]
    if len(p50) != source.n_layers:
        raise ValueError(f"The push's unit is kept for {len(p50)} layers, the model has {source.n_layers}.")
    edited = sorted({(e.layer, e.neuron) for e in entries} & {(int(l), int(n)) for l, n in edits})
    for layer in sorted({e.layer for e in entries}):
        ns = sorted({e.neuron for e in entries if e.layer == layer})
        cols, _w_post = source.columns(layer, ns)
        silent = [n for n, col in zip(ns, cols) if not bool(col.any())]
        if silent:
            raise ValueError(f"Layer {layer} (page {layer + 1}) neuron {silent[0]} writes nothing: its down column is zero"
                             + (" under the vault edit in force on it" if (layer, silent[0]) in edited else "")
                             + ", so its push adds nothing and there is no direction to export.")
    directions = ecv.pushes(entries, source, p50)
    src = scales.get("source") or {}
    out = (folder or exports_dir()) / file_name(name, entries)
    assert NAME_OK.fullmatch(out.name[:-5]), out.name  # whatever is written, GET /exports can hand over
    out.parent.mkdir(parents=True, exist_ok=True)
    one = len(entries) == 1
    record: Dict[str, Any] = {
        "name": (f"theodolite page {entries[0].layer + 1} n{entries[0].neuron} dose {entries[0].dose:+g}" if one
                 else f"theodolite set of {len(entries)} neurons at layers {', '.join(str(l) for l in sorted(directions))}"),
        "description": f"The atlas's add push of {ecv.describe(entries)}, from Theodolite: llama-server "
                       "--control-vector-scaled FILE:1 pushes it at the doses written here.",
        "model": model,
        "weights": source.record(),
        "layers": [int(e.layer) for e in entries],
        "pages": [int(e.layer) + 1 for e in entries],
        "neurons": [int(e.neuron) for e in entries],
        "doses": [float(e.dose) for e in entries],
        "p50": [float(p50[e.layer]) for e in entries],
        "stats_index": "the push's unit as the atlas took it: " + json.dumps(src, ensure_ascii=False, sort_keys=True, default=str)
                       + "; p50 = the median length of the stream leaving each layer",
        "vault_edits_in_force": ", ".join(f"layer {l} neuron {n}" for l, n in edited) or "none on these neurons",
        "push": ecv.PUSH,
        "start_mark": ecv.START_MARK,
        "scale": ecv.SCALE,
        "llama_cpp": ecv.LLAMA_CPP_READ,
        "caveat": CAVEAT,
        "made_by": f"backend/control_vector.py (the atlas's server) at {ecv.code_commit()}",
        "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ecv.write_control_vector(out, directions, record)
    lengths = {l: float(np.linalg.norm(v.astype(np.float64))) for l, v in directions.items()}
    return {
        "file": out.name, "path": str(out), "bytes": out.stat().st_size, "model": model,
        "entries": [{"layer": e.layer, "n": e.neuron, "dose": e.dose, "p50": round(p50[e.layer], 4)} for e in entries],
        "directions": [{"layer": l, "tensor": f"direction.{l}", "length": round(lengths[l], 4),
                        "of_p50": round(lengths[l] / p50[l], 4) if p50[l] else None} for l in sorted(directions)],
        "unit_source": src,
        "vault_edits_in_force": [{"layer": l, "n": n} for l, n in edited],
        "load_with": f"llama-server -m <a Gemma-4-31B-it GGUF> --control-vector-scaled {out}:1",
        "says": ("llama.cpp adds each direction to the stream leaving its layer, after the layer scalar, where the instrument "
                 "adds its push (read in llama.cpp's source and checked through llama.cpp itself: ARCHITECTURE.md §6), so "
                 "SCALE 1 is these doses, -1 the opposite push and 0.5 half of each. It pushes the start mark too, which "
                 "the instrument spares unless asked: steer with spare_bos false to see the push as llama.cpp gives it, and "
                 "stay near SCALE 1."),
    }
