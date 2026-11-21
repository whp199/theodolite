"""The edit as a value (`backend/weight_edits.py`): one change to the base weights, named by its hash, passed through
a gate, saved and loaded, and run as the arithmetic of one row of a batch beside the unedited model.

What it is. An edit is a list of terms. A term names one feed-forward linear of the text model and stands for an
additive difference Delta from the shipped tensor W; at dose d the weight is W + d * Delta (dose 1 the edit, 0 the
base, -1 its mirror). Four forms are built (slices 1a and 1b): a column term on a `down_proj` (what a neuron writes:
a column scaled, silenced, set to a given vector or added to), masked direction removal on `down_proj` columns
(projected off the layer's quiet mask before dividing by `post_feedforward_layernorm`'s weight, lowered to column
adds), a row term on a `gate_proj` or `up_proj` (what a neuron reads: scaled, silenced, set or added to, with the
edited output coordinates recomputed rather than added to), and a low-rank term B @ A on a `down_proj`.
An edit is never left in the model. `per_row` runs it as one row's own arithmetic for the length of a block, with
other rows of the same forward untouched; `folded` lays the same edit into the overlay dicts every packed linear
carries (`_memit_col_overlays` and `_memit_row_overlays`), for every row, for the length of a block; both take off
what they put on, and `assert_base` says whether the model is the base again. `backend/levelling.py` reads an edit
against the base row of the same forward and `backend/level_book.py` keeps what was read; this file calls neither,
and takes the text model, never a runtime.

Why an edit rides a row, and in which form. What follows was measured on 14 November 2026 on the RTX 5090
(Gemma-4-31B-it, the W4A16 checkpoint, compiled unpacker, lens freed, server stopped; one run each; scripts and JSON
in experiments/results/weight_space_exploration/pilot/). The pilots' raw text was text with no start mark, and
pilot 6's and pilot 4's figures below are of one such text (a recipe of 150 pieces). The card check read the same
comparisons on chat-formatted items on 16 November 2026 (after the list).
- MEASURED (pilots 1 and 6): rows of one forward do not leak. An untouched row far from edited rows equals row 0 by
  bits, a hooked forward's row 0 equals an unhooked forward's of the same shape, and one edit in rows 1, 9, 20 and
  31 of 32 reads the same.
- MEASURED (pilot 6, the raw recipe with no start mark): silencing a neuron on the INPUT side of its down_proj (the
  row's activation set to nought before the product) reads as the folded weight does to five decimals of a mean KL
  at the same batch size, 7 of 7 cases at 2 and at 32 rows. A factor of 0 only: no pilot ran another input-side
  factor.
- MEASURED (pilots 4 and 6, the same text): an OUTPUT-side delta for the same silencing is another arithmetic
  (0.0227 where the other two read 0.7258). So a column edit rides its row on the input side, and only new content
  is added after the product.
- MEASURED (pilot 3): `atlas_ops.write_rows` under an output-side hook registered round it wrote the same 48 pieces
  with the same log-probabilities in its untouched rows, at the same 0.095 s a step (8 rows, no row stopping).
- MEASURED (pilot 1): a column overlay costs about 0.3 ms a cloned matrix, to 64 overlay columns on one linear.
- RAN ON THE CPU (14 November 2026; arithmetic, not the card's): 1 + 0.001 and 1 - 0.001 are exactly 1 in bf16 (the
  step at 1 is 1/128 above and 1/256 below), so a null row with such a factor is the base row again; a power of two
  is an exact scaling the norm may undo exactly. A null row's factor is a value of the model's dtype other than 1
  and not a power of two.
- RAN ON THE CPU (the packed bf16 stand-in, 14 November 2026): input-side factors of 0, 2 and -0.5 equal the folded
  overlay by bits at 12 and at 32 rows; 1.5 and -3 do not (the factor rounds once to the dtype and each product
  rounds once, where the folded weight rounds each entry of the column): there the two forms differ by a mean KL of
  1.3e-6 to 1.5e-6 nats, where that stand-in's null rows read 2.9e-7 to 2.7e-6 (this file's builder's own run: a
  scale by 1.5 and by -3, a set and an add, a set at dose 0.5 and an add at dose -1; other columns and items give
  other figures of that size, and the spec and `levelling` each quote their own run's).
- RAN ON THE CPU (both stand-ins, one thread, 14 November 2026): a row at dose 0 of each kind (scale, set, add,
  low-rank) is the base row by bits, in a whole forward and in every step of `write_rows`; rows beside edited rows
  write what they write with no hook; a row that stops early stays in the batch, so the hooks' row count stands.
  None of this is the card's arithmetic.

The per-row forms (`per_row`), one program a row, acting at every position, the start mark included (it is a
weight edit, not a push), and never indexing positions, so a decode step with a cache runs the same code:
- a column scaled by f, at dose d: the row's activation of that neuron times g = 1 + d (f - 1), before the product
  (the arithmetic of the atlas's `mul` push, for every listed column at once); g is worked in Python floats and
  cast once to the model's dtype. Exact where g is 0 or a power of two; any other g rounds, and `exact_at` says so;
- a column set to v: the old column's share that stays, g = 1 - d, on the input side, and the new content
  a (d v) added after the product from the row's own activation a; a column added to: g = 1 and the same add. The
  content is rounded to the model's dtype as the folded form rounds the column it writes, and the sum is made in
  float32 and rounded once (the arithmetic of `int4_cpu.with_overlay_terms`);
- a low-rank term: (x A^T) d B^T added after the product, into the same float32 sum;
- a null row (`Null`): the layer's feed-forward output times 1 + e, in the model's dtype, which the norm after the
  block undoes but for rounding. It measures the rounding a row can show with no edit in it.

Read on the card on 16 November 2026, where no pilot had (scripts/verify_levelling.py, one run; its record is
experiments/results/levelling/verify_1a.json): an input-side factor of 2 equals the folded overlay by bits, as a
silencing does (48 comparisons of 48: 1, 64 and 1,000 columns at layers 5 and 20, at 12 and 32 rows, on a chat-formatted
item and a passage given as an answer); a scale by 1.5 or -3, a set and the float32 output-side add (the pilots' hooks
multiplied and summed in bf16) stand off the folded form by 0.00002 to 0.009 nats of mean KL; a rank-1 low-rank term on
a down_proj that equals a one-column add reads as that add to 0.00004 and 0.00017 nats; a row at dose 0 of a scale, a
set and a low-rank term is the base row by bits at 12 and 32 rows; under the pre-hook in a decode step, two rows
stopping, untouched rows write the base's pieces with its log-probabilities exactly. A low-rank term has no folded
form in 1a, so nothing confirms it as it would ship; every report says so.

Provisional: every cap below (`MAX_TERMS` to `MAX_DOSE`) is a guard against a slip, not a finding.

What the gate refuses, each in plain words (`Refused`, section "the gate"): a low-rank term on `gate_proj` or
`up_proj` (it would be the added form of a change to what neurons read, not the recomputed form of a row edit);
a direction `u` for `remove_direction` whose energy in the layer's quiet mask exceeds `MAX_QUIET_SHARE`; a uniform
positive scale of every column of a `down_proj` or every row of an `up_proj` (undone by the post-feed-forward norm);
attention linears (until their facts are in the gate and a card check has passed); the embedding, the norms and the
layer scalars; a GGUF's layer and an int4mm linear (an edit is pinned to the W4A16 checkpoint's packed tensors or a
stand-in's plain ones); and whatever passes a cap. An edit is pinned to its base: the model's id, the weights card's
id and a sha256 of each tensor it touches, all in its own hash, and `load` refuses an edit made on another base.
Row edits and masked direction removal (slice 1b) were card-checked on 18 November 2026 (`scripts/verify_levelling_1b.py`,
`experiments/results/levelling/verify_1b.json`, 68 of 68 checks passed in 86.2 s, peak 18.61 GiB, 0 unpacker fallbacks):
24 of 24 exact row edits (silencing and scales by 2 and -0.5 on `gate_proj` and `up_proj`, layers 5 and 20, 12 and 32 rows)
equalled folded `_memit_row_overlays` by bits; 11 of 12 recomputed row-edit class readings (scale 1.5, set, add on layer 20)
equalled folded by bits and the 12th (`gate_proj` set at 32 rows) stood off by 0.00086 nats of KL; masked direction removal
at layer 20 gave zero delta on all 344 masked dims and stream orthogonality to 7.21e-10 (0.000095 to 0.00043 nats of KL off
its folded twin).

Saved edits live outside the checkout (`edits_dir`): a payload can carry pieces of Gemma's columns and is never
committed. A term that only scales columns carries no array of Gemma's: its manifest alone rebuilds it.

Nothing reaches the model but through the gate. A saved edit passes it again when it is loaded and an `Edit` handed
to a reading as an object passes it again there (`gate_again`), its id worked from what it holds. A control edit is
let past one cap only (a column's change over its own length): the mark that asks for this sits in a report, which
is not hashed. A recipe's array files are named from the plan file's folder and lie inside it. A -0.0 in a factor or
a vector is kept as 0.0, so that one arithmetic has one id.

Layers count from 0 here and in every file; a message a person reads says both ("layer 20 (page layer 21)").
"By bits" means `torch.equal` on final-normed states, as in the pilots (it reads -0 as 0).
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.census import write_json_atomic
from backend.gguf_weights import dense, dense_cols, dense_rows, packed_kind

FORMAT = "theodolite-edit"
VERSION = 1
OPS = ("scale", "set", "add")  # a column's or row's op, by its code in Term.op (0, 1, 2)
WHICH = ("down_proj", "gate_proj", "up_proj")

# Guards against slips, all provisional: none is a finding about the model.
MAX_TERMS = 16
MAX_COLUMNS = 4096  # columns (or rows) a term
MAX_VECTOR_COLUMNS = 1024  # set or added columns (or rows) a term
MAX_RANK = 64  # a per-row low-rank add was timed to rank 64 (pilot 1, 14 November 2026: terms of zeros, in the
#                pilots' own bf16 hook); the float32 form built here was timed on the card on 16 November 2026: a
#                reading with a rank-64 term 1.56 s, against 1.62 s for the one-column add (verify_1a.json, 5e)
MAX_FACTOR = 64.0
MAX_REL_CHANGE = 64.0  # a column's or row's change over its own length; a low-rank term's bound; what a dose carries
MAX_PAYLOAD_BYTES = 256 * 2**20
MAX_DOSE = 4.0
QUIET_WPOST_RATIO = 0.05  # entries of |w_post| below 5% of median(|w_post|) are masked (bounds division to at most 20x)
MAX_QUIET_SHARE = 0.25  # a direction with more than 25% of its energy in the layer's quiet mask is refused
EDITS_ENV = "THEODOLITE_EDITS_DIR"

CONTROL_CAPS = "not applied to a control edit"
FORM_LOWRANK = "an output-side add; no folded form in 1a"
FORM_ROWS = "recomputed edited coordinates on {which}; folded form through _memit_row_overlays (card-checked 18 November 2026)"
_OVERLAYS = ("_memit_col_overlays", "_memit_row_overlays")  # read from engine_runtime.py, which the kernel may not edit
_NAME = re.compile(r"^layers\.(-?\d+)\.mlp\.(down_proj|gate_proj|up_proj)$")
_ROOT = Path(__file__).resolve().parent.parent
QUIET_DIMS_PATH = _ROOT / "experiments" / "results" / "V6_quiet_dims_pers181.npy"
_PERS181: Optional[np.ndarray] = None


class Refused(ValueError):
    """The gate, in plain words: what was asked is not built, not measured, or past a cap."""


class NotBase(RuntimeError):
    """The model is not the base: a hook or an overlay is in force that was not there at load."""


# ------------------------------------------------------------------------------------------------ the gate's words
# Section 2.2 of the build spec, in its order, which is the order of precedence: an input that breaks two rules is
# told the earlier one. The one sentence not in that table is the low-rank term on gate_proj or up_proj.

_SAY_ATTENTION = ("Edits to attention linears ({name}) are refused until their facts are in the gate and a card check "
                  "has passed: q, k and v each pass a per-head norm, and on a full-attention layer the values are the "
                  "keys. They wait for 1b.")
_SAY_NOT_FFN = ("Only the feed-forward linears take edits in 1a: layers.<L>.mlp.down_proj for a column edit, and "
                'down_proj for a low-rank term. "{name}" is not one. The embedding, the norms and the layer scalars '
                "are not built.")
_SAY_LAYER = "Layer {L} does not exist: layers count from 0 to {last} here (page layers 1 to {n})."
_SAY_COLS_WHICH = "A column edit changes what a neuron writes, so it names a down_proj; {name} is not one."
_SAY_ROWS_WHICH = "A row edit changes what a neuron reads, so it names gate_proj or up_proj; {name} is not one."
_SAY_LOWRANK_WHICH = ("A low-rank term on {name} is not built in 1a: on {which} it would be the added form of a change "
                      "to what neurons read, not the recomputed form asked of a row edit, and that waits for 1b. In "
                      "1a a low-rank term names a down_proj.")
_SAY_NEURON = "Neuron {n} does not exist: layer {L} (page layer {L1}) has {N} (0 to {last})."
_SAY_TWICE = "Neuron {n} of layer {L} (page layer {L1}) is named twice in this edit: one change a column."
_SAY_TWICE_ROW = "Neuron {n} of {name} (page layer {L1}) is named twice in this edit: one change a row."
_SAY_NO_COLUMN = "A term that names no neuron changes nothing."
_SAY_NOT_FINITE = "An edit holds a number that is not finite ({where})."
_SAY_FACTOR = "A column is scaled by at most {cap:g} either way here, not {f:g}."
_SAY_VECTOR = "A column of layer {L}'s down_proj is {out} numbers long; the vector given for neuron {n} has {k}."
_SAY_ROW_VECTOR = "A row of layer {L}'s {which} is {n_in} numbers long; the vector given for neuron {n} has {k}."
_SAY_REL = ("Neuron {n} of layer {L} (page layer {L1}) would change by {x:.0f} times its own length; 1a takes at most "
            "{cap:g}.")
_SAY_UNIFORM = ("Scaling every column of layer {L}'s down_proj by {f:g} changes nothing: the norm after the block "
                "undoes a uniform positive scale.")
_SAY_UNIFORM_UP = ("Scaling every row of layer {L}'s up_proj by {f:g} changes nothing: the norm after the block "
                   "undoes a uniform positive scale.")
_SAY_QUIET_SHARE = ("Direction u puts {pct:.1f}% of its energy in layer {L} (page layer {L1})'s quiet mask (at most "
                    "{cap:.0f}% is taken): dividing by post_feedforward_layernorm's weight there would amplify the "
                    "raw column.")
_SAY_CAP = "An edit takes at most {cap} {what} in 1a, not {k}."
_SAY_LOWRANK_SHAPE = ("A low-rank term on {name} needs A [rank, {n_in}] and B [{n_out}, rank]; it was given A {a} and "
                      "B {b}.")
_SAY_RANK = "A low-rank term has rank at most {cap} here, not {r}."
_SAY_BOUND = ("A low-rank term on {name} could change it by up to {x:.0f} times its own length (the lengths of B and A "
              "over the weight's); 1a takes at most {cap:g}.")
_SAY_BYTES = "This edit carries {mb:.0f} MiB of arrays; 1a takes at most {cap_mb:.0f}."
_SAY_PIN = ("An edit is pinned to the W4A16 checkpoint's packed tensors, or to a stand-in's plain ones. {name} is "
            "{kind}: another object, not taken in 1a.")
_SAY_CONTROL = "Layer {L} (page layer {L1}) has {m} neurons outside this edit: too few for a control edit of {K}."
_SAY_OTHER_BASE = ("This edit was made on {a} (weights {ia}); the loaded model is {b} (weights {ib}). An edit is a "
                   "difference from one base and is not carried to another.")
_SAY_OTHER_PIN = ("This edit was made against other weights: {name} then hashed to {a12}, the loaded model's hashes to "
                  "{b12}.")
_SAY_ARRAY = "The saved edit {id12}: its {array} no longer hashes to what its manifest says."
_SAY_ID = "The saved edit says it is {id12} and hashes to {other12}."
_SAY_FORMAT = "{path} is not a saved edit of format theodolite-edit, version 1."
# 2.4 (the recipe), 2.6 and 4.1 (the plan's words that this file is asked for too), 4.6 (folding)
_SAY_KEY = "An edit term has a key 1a does not know: {key}."
_SAY_FILE_HASH = "The file {path} hashes to {a12}, not the {b12} the plan names."
_SAY_FILE_SHAPE = "The file {path} is {dtype} {shape}; this term takes float32 {want}."
_SAY_NO_FOLD = "A low-rank term has no folded form yet, so this edit cannot be read folded."
_SAY_PLAIN = "{name} is a plain linear whose forward reads no overlay: the folded form is the packed checkpoint's."
_SAY_NULL_RANGE = ("A null row scales the layer's feed-forward output by 1 + e, with e between -0.5 and 2 and not 0; "
                   "{e:g} was asked.")
_SAY_NULL_ONE = ("In {dtype} 1 + e is exactly 1 for e = {e:g}: that null row would be the base row again. In bf16 the "
                 "nearest values are 1 + 1/128 and 1 - 1/256.")
_SAY_NULL_POWER = ("For e = {e:g} the null row's factor is {s:g}, a power of two: an exact scaling, which the norm may "
                   "undo exactly, so it measures no rounding. Factors that serve: 1 + 1/128, 1 - 1/256, 0.75, 1.5, 3.")
_SAY_DOSE_GONE = ("At dose {d:g} every factor of this edit is exactly 1 in {dtype}: its row would be the base row "
                  "again. Doses are dyadic here (1, 0.5, 0.25, -1); for a smaller effect take fewer columns, not a "
                  "smaller dose.")
_SAY_DOSE_NOTHING = ("This edit scales every column it names by exactly 1 and adds nothing: at any dose its row is the "
                     "base row again, so there is nothing to read.")
_SAY_DOSE_SIZE = ("At dose {d:g} this edit would change {name} by {x:.0f} times its own length; 1a takes at most "
                  "{cap:g}.")


# ------------------------------------------------------------------------------------------------ hashes and names


def canonical(obj: Any) -> bytes:
    """The one way an object is written before it is hashed: sorted keys, no spaces, ASCII, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def array_sha256(a: Any, tag: str) -> str:
    """sha256 of an array under one of three tags, "i8" (little-endian int64), "f4" (little-endian float32) or "u1"
    (uint8): the tag, the shape, then the bytes in C order."""
    a = np.ascontiguousarray(a, dtype={"i8": "<i8", "f4": "<f4", "u1": "u1"}[tag])
    h = hashlib.sha256()
    h.update((tag + ":" + ",".join(str(int(s)) for s in a.shape) + ":").encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def _sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def linear_name(layer: int, which: str = "down_proj") -> str:
    """The dotted path of a feed-forward linear inside the text model."""
    return f"layers.{layer}.mlp.{which}"


def linear_shape(mod: nn.Module) -> Tuple[int, int]:
    """(out, in) of a linear, packed or plain."""
    packed = getattr(mod, "weight_packed", None)
    if packed is not None and getattr(mod, "weight_scale", None) is not None:
        return int(packed.shape[0]), 8 * int(packed.shape[1])
    w = getattr(mod, "weight", None)
    if w is not None:
        return int(w.shape[0]), int(w.shape[1])
    return int(mod.out_features), int(mod.in_features)


def _dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    got = getattr(torch, str(dtype).replace("torch.", ""), None)
    if not isinstance(got, torch.dtype):
        raise TypeError(f"{dtype!r} is not a torch dtype.")
    return got


def _dtype_name(dtype: Any) -> str:
    return str(_dtype(dtype)).replace("torch.", "")


def _raw(t: torch.Tensor) -> memoryview:
    """A tensor's stored bytes, on the CPU, in C order (bf16 viewed as int16, which numpy can hold)."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    a = np.ascontiguousarray(t.numpy())
    return memoryview(a.tobytes() if a.size == 0 else a).cast("B")


# A packed linear's pin and its weight's length are kept for the life of the process (a down_proj's packed words are
# 58 MB, and nothing in the process writes packed words). A plain linear's are worked again at every call: its weight
# can be written in place, and only the stand-ins have plain linears.
_PINS: "weakref.WeakKeyDictionary[nn.Module, str]" = weakref.WeakKeyDictionary()
_LENGTHS: "weakref.WeakKeyDictionary[nn.Module, float]" = weakref.WeakKeyDictionary()
_STREAM_NORMS: "weakref.WeakKeyDictionary[nn.Module, np.ndarray]" = weakref.WeakKeyDictionary()


def _pin(mod: nn.Module, name: str) -> str:
    kind = packed_kind(mod)
    if kind.startswith("gguf"):
        raise Refused(_SAY_PIN.format(name=name, kind=kind))
    if getattr(mod, "int4mm", False):  # before the packed test: such a linear keeps weight_packed and weight_scale
        raise Refused(_SAY_PIN.format(name=name, kind="w4a16 int4mm"))
    h = hashlib.sha256()
    if kind == "w4a16":
        got = _PINS.get(mod)
        if got is not None:
            return got
        p, s = mod.weight_packed, mod.weight_scale
        p0, p1, s0, s1 = (int(n) for n in (*p.shape, *s.shape))
        h.update(f"w4a16:{p0}x{p1}:{s0}x{s1}:{s.dtype}:".encode("ascii"))
        h.update(_raw(p.to(torch.int32)))
        h.update(_raw(s))
        _PINS[mod] = h.hexdigest()
        return _PINS[mod]
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        raise Refused(_SAY_PIN.format(name=name, kind="a module with no weight of its own"))
    h.update(f"dense:{int(w.shape[0])}x{int(w.shape[1])}:{w.dtype}:".encode("ascii"))
    h.update(_raw(w))
    return h.hexdigest()


def tensor_pin(mod: nn.Module) -> str:
    """The sha256 an edit pins a linear by: of the packed words and scales as stored, or of a plain weight's own
    bytes. `Refused` for a GGUF's layer and an int4mm linear."""
    return _pin(mod, "This linear")


def _linear_of(lm: nn.Module, name: str) -> Optional[nn.Module]:
    try:
        return lm.get_submodule(name)
    except AttributeError:
        return None


def _parse(name: str) -> Tuple[int, str]:
    m = _NAME.match(name)
    if m is None:
        raise Refused(_SAY_NOT_FFN.format(name=name))
    return int(m.group(1)), m.group(2)


def _exact(g: float) -> bool:
    """A factor the model's arithmetic applies with no rounding: 0, or a power of two with either sign."""
    return g == 0 or math.frexp(abs(g))[0] == 0.5


def _applied(g: float, dtype: torch.dtype) -> float:
    return float(torch.tensor(float(g), dtype=dtype))


def _ro(a: Any, dtype: Any) -> np.ndarray:
    """A C-contiguous copy of its own, set read-only."""
    with np.errstate(all="ignore"):
        out = np.array(a, dtype=dtype, order="C", copy=True)
    out.setflags(write=False)
    return out


def _long(idx: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(idx, dtype=np.int64))


def _clean(x: Any) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------------------------------------------ the edit


@dataclass(frozen=True, eq=False)
class Term:
    """One linear's share of an edit. `kind` "cols" (down_proj) or "rows" (gate_proj, up_proj): indices `idx`
    (strictly ascending), each with an op (0 scale, 1 set, 2 add), a factor (that of a scaled entry; 1.0 for a set or
    added one) and, where any is set or added, a row of `vectors` ([K, out] for cols, [K, in] for rows; zeros for
    scaled entries). `kind` "lowrank": Delta = B @ A, A [r, in], B [out, r]. Arrays are numpy, C-contiguous and
    read-only; a term is compared by identity."""

    linear: str
    kind: str
    layer: int
    which: str
    idx: Optional[np.ndarray] = field(default=None, repr=False)
    op: Optional[np.ndarray] = field(default=None, repr=False)
    factor: Optional[np.ndarray] = field(default=None, repr=False)
    vectors: Optional[np.ndarray] = field(default=None, repr=False)
    A: Optional[np.ndarray] = field(default=None, repr=False)
    B: Optional[np.ndarray] = field(default=None, repr=False)


def _term_core(t: Term) -> Dict[str, Any]:
    if t.kind == "cols":
        return {"linear": t.linear, "kind": "cols", "count": int(t.idx.shape[0]),
                "columns": array_sha256(t.idx, "i8"), "ops": array_sha256(t.op, "u1"),
                "factors": array_sha256(t.factor, "f4"),
                "vectors": array_sha256(t.vectors, "f4") if t.vectors is not None else None}
    if t.kind == "rows":
        return {"linear": t.linear, "kind": "rows", "count": int(t.idx.shape[0]),
                "rows": array_sha256(t.idx, "i8"), "ops": array_sha256(t.op, "u1"),
                "factors": array_sha256(t.factor, "f4"),
                "vectors": array_sha256(t.vectors, "f4") if t.vectors is not None else None}
    return {"linear": t.linear, "kind": "lowrank", "rank": int(t.A.shape[0]),
            "A": array_sha256(t.A, "f4"), "B": array_sha256(t.B, "f4")}


def _edit_id(base: Dict[str, Any], terms: Sequence[Term]) -> str:
    return _sha({"format": FORMAT, "version": VERSION, "base": base, "terms": [_term_core(t) for t in terms]})


@dataclass(frozen=True, eq=False)
class Edit:
    """An edit: its `id` (the sha256 of its base and its terms' arrays), its terms sorted by (linear, kind), the base
    it is a difference from ({"model", "weights_id", "tensors": {linear: pin}}), a note and a report (neither
    hashed). The report is {"terms": [one entry a term, in the terms' order]}, and on a control edit also
    {"caps": "not applied to a control edit"}. Two edits are the same edit when their ids are equal; `==` is
    identity."""

    id: str
    terms: Tuple[Term, ...]
    base: Dict[str, Any] = field(repr=False)
    note: str = ""
    report: Dict[str, Any] = field(default_factory=dict, repr=False)

    def layers(self) -> List[int]:
        return sorted({int(t.layer) for t in self.terms})

    def lowest_layer(self) -> int:
        return min(int(t.layer) for t in self.terms)

    def has_lowrank(self) -> bool:
        return any(t.kind == "lowrank" for t in self.terms)

    def has_rows(self) -> bool:
        return any(t.kind == "rows" for t in self.terms)

    def has_vectors(self) -> bool:
        """Whether any column or row is set or added to."""
        return any(t.kind in ("cols", "rows") and t.vectors is not None for t in self.terms)

    def linears(self) -> List[str]:
        return sorted({t.linear for t in self.terms})

    def short(self) -> str:
        return self.id[:12]

    def manifest(self) -> Dict[str, Any]:
        """What `save` writes as manifest.json, without its time. A term that only scales columns or rows is whole in
        it ("carried": false); a term with vectors, or a low-rank one, names its arrays in the payload."""
        terms: List[Dict[str, Any]] = []
        for i, t in enumerate(self.terms):
            core = _term_core(t)
            if t.kind in ("cols", "rows"):
                idx_key = "columns" if t.kind == "cols" else "rows"
                terms.append({
                    "linear": t.linear, "kind": t.kind, "count": int(t.idx.shape[0]),
                    idx_key: [int(n) for n in t.idx], "ops": [OPS[int(o)] for o in t.op],
                    "factors": [float(f) for f in t.factor],
                    "vectors": f"t{i}.vectors" if t.vectors is not None else None,
                    "sha256": {k: core[k] for k in (idx_key, "ops", "factors", "vectors")},
                    "carried": t.vectors is not None})
            else:
                terms.append({"linear": t.linear, "kind": "lowrank", "rank": int(t.A.shape[0]), "A": f"t{i}.A",
                              "B": f"t{i}.B", "sha256": {"A": core["A"], "B": core["B"]}, "carried": True})
        return {"format": FORMAT, "version": VERSION, "id": self.id, "base": self.base, "terms": terms,
                "note": self.note, "report": self.report}


def _payload(edit: Edit) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for i, t in enumerate(edit.terms):
        if t.kind in ("cols", "rows") and t.vectors is not None:
            out[f"t{i}.vectors"] = t.vectors
        elif t.kind == "lowrank":
            out[f"t{i}.A"], out[f"t{i}.B"] = t.A, t.B
    return out


# ------------------------------------------------------------------------------------------------- recipe terms


@dataclass(frozen=True, eq=False)
class TermSpec:
    """A recipe term before it meets a model: the builder's arguments, as given. `kind` is "cols" or "rows" (with `op`
    one of OPS), "lowrank", or "remove_direction" (lowered in the gate to a masked "cols" add term)."""

    kind: str
    layer: Any
    which: str = "down_proj"
    neurons: Tuple[Any, ...] = ()
    op: Optional[str] = None
    factor: Any = None
    vectors: Any = field(default=None, repr=False)
    A: Any = field(default=None, repr=False)
    B: Any = field(default=None, repr=False)
    direction: Any = field(default=None, repr=False)
    quiet_mask: Any = field(default=None, repr=False)
    max_quiet_share: Optional[float] = None


def _tuple(neurons: Any) -> Tuple[Any, ...]:
    if isinstance(neurons, torch.Tensor):
        neurons = neurons.detach().cpu().tolist()
    if isinstance(neurons, (str, bytes)) or not hasattr(neurons, "__iter__"):
        return (neurons,)
    return tuple(neurons)


def silence(layer: int, neurons: Sequence[int]) -> TermSpec:
    """The neurons' down_proj columns scaled by 0: what they write, gone."""
    return TermSpec("cols", layer, "down_proj", _tuple(neurons), "scale", 0.0)


def scale(layer: int, neurons: Sequence[int], factor: Union[float, Sequence[float]]) -> TermSpec:
    """The neurons' down_proj columns scaled: by one factor, or by one a neuron in the order listed."""
    return TermSpec("cols", layer, "down_proj", _tuple(neurons), "scale", factor)


def set_columns(layer: int, neurons: Sequence[int], vectors: Any) -> TermSpec:
    """The neurons' down_proj columns set to `vectors` [K, out], row j for neurons[j]."""
    return TermSpec("cols", layer, "down_proj", _tuple(neurons), "set", vectors=vectors)


def add_to_columns(layer: int, neurons: Sequence[int], vectors: Any) -> TermSpec:
    """`vectors` [K, out] added to the neurons' down_proj columns, row j to neurons[j]."""
    return TermSpec("cols", layer, "down_proj", _tuple(neurons), "add", vectors=vectors)


def silence_rows(layer: int, which: str, neurons: Sequence[int]) -> TermSpec:
    """The neurons' rows on gate_proj or up_proj scaled by 0: what they read, gone."""
    return TermSpec("rows", layer, which, _tuple(neurons), "scale", 0.0)


def scale_rows(layer: int, which: str, neurons: Sequence[int], factor: Union[float, Sequence[float]]) -> TermSpec:
    """The neurons' rows on gate_proj or up_proj scaled: by one factor, or by one a neuron in the order listed."""
    return TermSpec("rows", layer, which, _tuple(neurons), "scale", factor)


def set_rows(layer: int, which: str, neurons: Sequence[int], vectors: Any) -> TermSpec:
    """The neurons' rows on gate_proj or up_proj set to `vectors` [K, in], row j for neurons[j]."""
    return TermSpec("rows", layer, which, _tuple(neurons), "set", vectors=vectors)


def add_to_rows(layer: int, which: str, neurons: Sequence[int], vectors: Any) -> TermSpec:
    """`vectors` [K, in] added to the neurons' rows on gate_proj or up_proj, row j to neurons[j]."""
    return TermSpec("rows", layer, which, _tuple(neurons), "add", vectors=vectors)


def rows(layer: int, which: str, neurons: Sequence[int], vectors: Any, *, op: str = "set") -> TermSpec:
    """A row edit on gate_proj or up_proj (what a neuron reads): `op` is "set" (default) or "add"."""
    return TermSpec("rows", layer, which, _tuple(neurons), op, vectors=vectors)


def remove_direction(
    layer: int,
    neurons: Sequence[int],
    direction: Any,
    *,
    quiet_mask: Any = None,
    max_quiet_share: float = MAX_QUIET_SHARE,
) -> TermSpec:
    """Remove stream-space direction `direction` [D] from the listed neurons' down_proj columns, projected off the
    layer's quiet mask before dividing by `post_feedforward_layernorm`'s weight. Refused when `direction` puts more
    than `max_quiet_share` of its energy inside the quiet mask; lowered in `build` to a `cols` add term."""
    return TermSpec("remove_direction", layer, "down_proj", _tuple(neurons), direction=direction,
                    quiet_mask=quiet_mask, max_quiet_share=float(max_quiet_share))


def lowrank(layer: int, A: Any, B: Any, which: str = "down_proj") -> TermSpec:
    """B @ A added to a linear's weight: A [rank, in], B [out, rank]. The gate takes it on a down_proj only."""
    return TermSpec("lowrank", layer, which, A=A, B=B)


_COL_OPS = ("silence", "scale", "set", "add")
_ROW_OPS = ("silence_rows", "scale_rows", "set_rows", "add_rows")
_RECIPE_OPS = _COL_OPS + _ROW_OPS + ("remove_direction", "lowrank")
_TERM_KEYS = ("layer",) + _RECIPE_OPS + ("factor", "factors", "vectors", "direction", "max_quiet_share", "which", "sha256")


def _recipe_shape(recipe: Any) -> Tuple[str, Any]:
    """("terms", the list) or ("saved", the id or folder), the recipe's own keys checked."""
    if not isinstance(recipe, dict):
        raise Refused('An edit is given as {"terms": [...]} or as {"saved": "<id or folder>"}; '
                      f"{type(recipe).__name__} is neither.")
    for key in recipe:
        if key not in ("terms", "saved", "note"):
            raise Refused(f"An edit has a key 1a does not know: {key}.")
    if ("terms" in recipe) == ("saved" in recipe):
        raise Refused('An edit is given as "terms" (a list of terms) or as "saved" (the id or folder of a saved '
                      "edit): one of the two.")
    if "saved" in recipe:
        if not isinstance(recipe["saved"], (str, os.PathLike)) or not str(recipe["saved"]):
            raise Refused('"saved" names a saved edit by its id or its folder.')
        return "saved", recipe["saved"]
    if not isinstance(recipe["terms"], (list, tuple)):
        raise Refused('"terms" is a list of terms, each with "layer" and one of '
                      + ", ".join(_RECIPE_OPS) + ".")
    return "terms", list(recipe["terms"])


def _read_npy(path: Path, want_hash: Any, want: str, k: Optional[int], *, ndim: int = 2) -> np.ndarray:
    """A recipe's array: the file's sha256 checked first when the plan names one, then read with no pickle, then its
    dtype and shape checked before anything is done with it (`k`: how many rows it must have, when that is known)."""
    try:
        size = path.stat().st_size
        if size > MAX_PAYLOAD_BYTES + 4096:  # one file alone is over the cap on an edit's arrays: not read at all
            raise Refused(_SAY_BYTES.format(mb=size / 2**20, cap_mb=MAX_PAYLOAD_BYTES / 2**20))
        data = path.read_bytes()
    except (OSError, ValueError):  # ValueError: a name the system will not take at all (a NUL in it)
        raise Refused(f"The file {path!s:.300} cannot be read: a term's arrays are .npy files, their paths taken "
                      "from the plan file's folder.") from None
    if want_hash is not None:
        got = hashlib.sha256(data).hexdigest()
        if got != str(want_hash).strip().lower():
            raise Refused(_SAY_FILE_HASH.format(path=path, a12=got[:12], b12=str(want_hash).strip().lower()[:12]))
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:  # a pickle, an object array, or no .npy file at all
        raise Refused(f"The file {path} is not a plain .npy array ({str(exc).strip() or type(exc).__name__}): a "
                      "term's file is one float32 array saved with numpy.save, with no pickle in it.") from None
    if not isinstance(arr, np.ndarray):
        raise Refused(f"The file {path} holds several arrays (.npz); a term's file is one float32 array saved with "
                      "numpy.save.")
    if arr.dtype != np.float32 or arr.ndim != ndim or (k is not None and arr.shape[0] != k):
        raise Refused(_SAY_FILE_SHAPE.format(path=path, dtype=arr.dtype, shape=list(arr.shape), want=want))
    return arr


def _inside(base_dir: Path, name: str) -> Path:
    """A term's file, named from the plan file's folder. A name that is absolute, or that climbs out of that folder,
    is refused: a plan names its own arrays, not any file the process can read (a wrong sha256 would otherwise print
    the first 12 of another file's hash). Links are not followed: one put inside the folder is the plan's own."""
    base = os.path.normpath(os.path.abspath(str(base_dir)))
    full = os.path.normpath(os.path.join(base, name))
    if os.path.isabs(name) or "\x00" in name or not (full + os.sep).startswith(base + os.sep) or full == base:
        raise Refused(f"A term's file is named from the plan file's folder and lies inside it; {name!s:.300} does not.")
    return Path(base_dir) / name


def _number(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise Refused(f"{what} is a number; {x!r} is not one.")
    try:
        return float(x)
    except OverflowError:  # a whole number past what a float holds
        raise Refused(f"{what} is a number a float can hold; one of {len(str(x))} digits is not.") from None


def _recipe_term(t: Any, base_dir: Path, read: bool, paths: Optional[List[Path]] = None) -> TermSpec:
    """One term of a recipe as a TermSpec. With `read` false its files are not opened (the facts and preflight need
    none); when `paths` is given, every validated file path is appended to it."""
    if not isinstance(t, dict):
        raise Refused('An edit term is a set of keys, "layer" and one of ' + ", ".join(_RECIPE_OPS) + f"; {t!r} is not one.")
    if "layer" not in t:
        raise Refused('An edit term needs "layer": the layer it changes, counted from 0.')
    layer = t["layer"]
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise Refused(f"A layer is named by a whole number counted from 0; {layer!r} is not one.")
    for key in t:
        if key not in _TERM_KEYS:
            raise Refused(_SAY_KEY.format(key=key))
    named = [k for k in _RECIPE_OPS if k in t]
    if len(named) != 1:
        raise Refused("An edit term names exactly one of " + ", ".join(_RECIPE_OPS) + "; this one names "
                      + (" and ".join(named) if named else "none") + ".")
    op = named[0]
    is_row = op in _ROW_OPS
    if is_row and "which" not in t:
        raise Refused(f'A "{op}" term needs "which": "gate_proj" or "up_proj".')
    which = t.get("which", "down_proj")
    if not isinstance(which, str):
        raise Refused(f'"which" names a linear of the layer (down_proj, gate_proj or up_proj); {which!r} is not a name.')
    takes = {
        "silence": (), "scale": ("factor", "factors"), "set": ("vectors", "sha256"), "add": ("vectors", "sha256"),
        "silence_rows": (), "scale_rows": ("factor", "factors"), "set_rows": ("vectors", "sha256"),
        "add_rows": ("vectors", "sha256"),
        "remove_direction": ("direction", "sha256", "max_quiet_share"),
        "lowrank": ("sha256",),
    }[op]
    for key in ("factor", "factors", "vectors", "direction", "max_quiet_share", "sha256"):
        if key in t and key not in takes:
            raise Refused(f'An edit term with "{op}" does not take "{key}".')
    if op == "lowrank":
        files = t["lowrank"]
        named_ab = isinstance(files, dict) and set(files) == {"A", "B"}
        if not named_ab or not all(isinstance(v, str) for v in files.values()):
            raise Refused('"lowrank" names two files, {"A": "<a.npy>", "B": "<b.npy>"}: A [rank, in] and B [out, '
                          "rank], float32.")
        hashes = t.get("sha256") or {}
        if not isinstance(hashes, dict) or not set(hashes) <= {"A", "B"}:
            raise Refused('The "sha256" of a low-rank term is {"A": "<hex>", "B": "<hex>"}.')
        path_a, path_b = _inside(base_dir, files["A"]), _inside(base_dir, files["B"])  # also when no file is opened
        if paths is not None:
            paths.extend((path_a, path_b))
        if not read:
            return TermSpec("lowrank", layer, which)
        A = _read_npy(path_a, hashes.get("A"), "[rank, in]", None)
        B = _read_npy(path_b, hashes.get("B"), "[out, rank]", None)
        return TermSpec("lowrank", layer, which, A=A, B=B)
    neurons = t[op]
    if not isinstance(neurons, (list, tuple)):
        raise Refused(f'"{op}" lists the neurons it changes, as whole numbers; {neurons!r} is not a list.')
    neurons = tuple(neurons)
    if op == "remove_direction":
        if not isinstance(t.get("direction"), str):
            raise Refused('A "remove_direction" term names its direction file: "direction": "<file.npy>", float32 '
                          "[out].")
        if t.get("sha256") is not None and not isinstance(t["sha256"], str):
            raise Refused('The "sha256" of a "remove_direction" term is the hex sha256 of its direction file.')
        mq = MAX_QUIET_SHARE if "max_quiet_share" not in t else _number(t["max_quiet_share"], "max_quiet_share")
        path = _inside(base_dir, t["direction"])
        if paths is not None:
            paths.append(path)
        if not read:
            return TermSpec("remove_direction", layer, which, neurons, max_quiet_share=mq)
        direction = _read_npy(path, t.get("sha256"), "[out]", None, ndim=1)
        return TermSpec("remove_direction", layer, which, neurons, direction=direction, max_quiet_share=mq)
    kind = "rows" if is_row else "cols"
    base_op = op[:-5] if is_row else op
    if base_op == "silence":
        return TermSpec(kind, layer, which, neurons, "scale", 0.0)
    if base_op == "scale":
        if ("factor" in t) == ("factors" in t):
            raise Refused(f'A "{op}" term gives "factor" (one number for every neuron) or "factors" (one a neuron): '
                          "one of the two.")
        if "factor" in t:
            return TermSpec(kind, layer, which, neurons, "scale", _number(t["factor"], "A factor"))
        if not isinstance(t["factors"], (list, tuple)):
            raise Refused('"factors" is a list, one number a neuron in the order listed.')
        return TermSpec(kind, layer, which, neurons, "scale", [_number(f, "A factor") for f in t["factors"]])
    if not isinstance(t.get("vectors"), str):
        width_name = "in" if is_row else "out"
        raise Refused(f'A "{op}" term names its vectors\' file: "vectors": "<file.npy>", float32 [neurons, '
                      f"{width_name}], row j for the j-th neuron as listed.")
    if t.get("sha256") is not None and not isinstance(t["sha256"], str):
        raise Refused(f'The "sha256" of a "{op}" term is the hex sha256 of its vectors\' file.')
    path = _inside(base_dir, t["vectors"])  # also when no file is opened: the estimate says what the reading will
    if paths is not None:
        paths.append(path)
    if not read:
        return TermSpec(kind, layer, which, neurons, base_op)
    width_name = "in" if is_row else "out"
    vectors = _read_npy(path, t.get("sha256"), f"[{len(neurons)}, {width_name}]", len(neurons))
    return TermSpec(kind, layer, which, neurons, base_op, vectors=vectors)


def from_recipe(recipe: dict, base_dir: Path) -> Union[List[TermSpec], str]:
    """A plan's edit as recipe terms, its files read (paths taken from `base_dir`, the plan file's folder) and their
    sha256 checked where the plan names one; or, for {"saved": x}, the folder or id x to hand to `load`. Cumulative
    file sizes are checked against `MAX_PAYLOAD_BYTES` before any array is opened. `Refused` for a key it does not
    know and for a file that is not what the term takes; what the gate refuses, `build` says."""
    kind, body = _recipe_shape(recipe)
    if kind == "saved":
        near = Path(base_dir) / str(body)
        return str(near) if near.is_dir() else str(body)
    root = Path(base_dir)
    paths: List[Path] = []
    for t in body:
        _recipe_term(t, root, False, paths)
    if paths:
        total = 0
        for p in paths:
            try:
                total += p.stat().st_size
            except (OSError, ValueError):
                pass
        if total > MAX_PAYLOAD_BYTES + 4096 * len(paths):
            raise Refused(_SAY_BYTES.format(mb=total / 2**20, cap_mb=MAX_PAYLOAD_BYTES / 2**20))
    return [_recipe_term(t, root, True) for t in body]


def _facts(layers: Sequence[int], linears: Sequence[str], has_vectors: bool, has_lowrank: bool,
           factors: Sequence[float]) -> Dict[str, Any]:
    if not layers:
        raise Refused("An edit with no term changes nothing.")
    with np.errstate(all="ignore"):
        f32 = sorted({float(np.float32(f) + np.float32(0.0)) for f in factors})
    return {"layers": sorted({int(x) for x in layers}), "lowest_layer": int(min(layers)),
            "linears": len(set(linears)), "has_vectors": bool(has_vectors), "has_lowrank": bool(has_lowrank),
            "factors": f32}


def _spec_name(s: TermSpec) -> str:
    """The dotted path a recipe term names. `which` is one of the three feed-forward linears; anything else is kept
    as the name it most plainly means, for the gate to refuse by name."""
    if isinstance(s.layer, bool) or not isinstance(s.layer, (int, np.integer)):
        raise Refused(f"A layer is named by a whole number counted from 0; {s.layer!r} is not one.")
    w = str(s.which)
    if w in WHICH:
        return linear_name(int(s.layer), w)
    if w.startswith("layers."):
        return w
    if w in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"layers.{int(s.layer)}.self_attn.{w}"
    if "." in w:
        return f"layers.{int(s.layer)}.{w}"
    return linear_name(int(s.layer), w)


def recipe_facts(recipe: dict, base_dir: Path) -> dict:
    """What a plan can know of its edit with no model: {"layers", "lowest_layer", "linears", "has_vectors",
    "has_lowrank", "factors": the distinct scale factors}. No file of a term is opened; a saved edit is read from its
    manifest alone."""
    kind, body = _recipe_shape(recipe)
    if kind == "saved":
        near = Path(base_dir) / str(body)
        folder = _find_saved(near if near.is_dir() else body)
        terms = _read_manifest(folder)["terms"]
        try:
            factors = [f for t in terms if t["kind"] in ("cols", "rows")
                       for f, o in zip(t["factors"], t["ops"]) if o == "scale"]
            return _facts([_parse(t["linear"])[0] for t in terms], [t["linear"] for t in terms],
                          any(t["kind"] in ("cols", "rows") and t.get("vectors") for t in terms),
                          any(t["kind"] == "lowrank" for t in terms), factors)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, Refused):
                raise
            raise Refused(_SAY_FORMAT.format(path=folder)) from None
    specs = [_recipe_term(t, Path(base_dir), False) for t in body]
    factors: List[float] = []
    for s in specs:
        if s.kind in ("cols", "rows") and s.op == "scale":
            factors += [float(f) for f in s.factor] if isinstance(s.factor, (list, tuple)) else [float(s.factor)]
    names = [_spec_name(s) for s in specs]
    has_vec = any((s.kind in ("cols", "rows") and s.op in ("set", "add")) or s.kind == "remove_direction"
                  for s in specs)
    return _facts([s.layer for s in specs], names, has_vec,
                  any(s.kind == "lowrank" for s in specs), factors)


def edit_facts(edit: Edit) -> dict:
    """The same keys as `recipe_facts`, from a built edit."""
    factors = [float(f) for t in edit.terms if t.kind in ("cols", "rows")
               for f, o in zip(t.factor, t.op) if int(o) == 0]
    return _facts(edit.layers(), edit.linears(), edit.has_vectors(), edit.has_lowrank(), factors)


def twin_wanted(facts: dict, dose: float) -> bool:
    """Whether a folded twin has something to show at this dose: the edit has no low-rank term (which has no folded
    form), and either sets or adds to a column or row, or scales one by a factor that at this dose is neither 0 nor a
    power of two. A silencing or a scale by a power of two is exact and takes none."""
    if facts["has_lowrank"]:
        return False
    if facts["has_vectors"]:
        return True
    return any(not _exact(1.0 + float(dose) * (float(f) - 1.0)) for f in facts["factors"])


# -------------------------------------------------------------------------------------- quiet mask and writers


def _w_post(lm: nn.Module, layer: int) -> np.ndarray:
    """`post_feedforward_layernorm`'s weight of `layer` as float64 [D]."""
    n_layers = len(lm.layers)
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise Refused(f"A layer is named by a whole number counted from 0; {layer!r} is not one.")
    L = int(layer)
    if not 0 <= L < n_layers:
        raise Refused(_SAY_LAYER.format(L=L, last=n_layers - 1, n=n_layers))
    norm = getattr(lm.layers[L], "post_feedforward_layernorm", None)
    weight = getattr(norm, "weight", None)
    if weight is None:
        raise Refused(f"Layer {L} (page layer {L + 1}) has no post_feedforward_layernorm weight.")
    return weight.detach().to(torch.float64).cpu().numpy()


def _pers181_mask() -> Optional[np.ndarray]:
    """The 181 persistent quiet dimensions of the 31B checkpoint, when the file is present (read-only bool [5376])."""
    global _PERS181
    if _PERS181 is None and QUIET_DIMS_PATH.is_file():
        arr = np.asarray(np.load(QUIET_DIMS_PATH, allow_pickle=False), dtype=bool).reshape(-1)
        arr.setflags(write=False)
        _PERS181 = arr
    return _PERS181


def quiet_mask(
    lm: nn.Module,
    layer: int,
    *,
    min_ratio: float = QUIET_WPOST_RATIO,
    extra_mask: Any = None,
    use_pers181: bool = True,
) -> np.ndarray:
    """Read-only bool mask [D] of `layer`'s quiet dimensions: True where `|w_post|` is zero or below `min_ratio` times
    `median(|w_post|)` (bounding `1 / w_post` to at most `1 / min_ratio`), unioned with the 31B's stored 181 persistent
    quiet dimensions when `D == 5376` and `use_pers181` is true, and unioned with `extra_mask` when given."""
    ratio = _number(min_ratio, "min_ratio")
    if not (math.isfinite(ratio) and ratio >= 0.0):
        raise Refused(f"min_ratio is a finite number from 0 up; {min_ratio!r} is not one.")
    w_post = _w_post(lm, layer)
    abs_w = np.abs(w_post)
    med = float(np.median(abs_w))
    mask = (abs_w < ratio * med) | (abs_w == 0.0) if med > 0.0 else np.ones_like(abs_w, dtype=bool)
    if use_pers181:
        p181 = _pers181_mask()
        if p181 is not None and p181.shape == mask.shape:
            mask = mask | p181
    if extra_mask is not None:
        if isinstance(extra_mask, torch.Tensor):
            extra = extra_mask.detach().cpu().numpy()
        else:
            extra = np.asarray(extra_mask)
        if extra.dtype != bool or extra.ndim != 1 or extra.shape != mask.shape:
            raise Refused(f"quiet_mask must be a 1D bool array of length {mask.shape[0]}; "
                          f"got {extra.dtype} {list(extra.shape)}.")
        mask = mask | extra
    mask = np.ascontiguousarray(mask, dtype=bool)
    mask.setflags(write=False)
    return mask


def down_stream_norms(lm: nn.Module, layer: int) -> np.ndarray:
    """Stream-side column lengths `||w_post * W_down[:, n]||_2` for every neuron n of `layer` (float64 [N], read-only),
    chunked to avoid full float32 copies and cached for packed linears."""
    w_post = _w_post(lm, layer)
    name = linear_name(int(layer), "down_proj")
    mod = _linear_of(lm, name)
    if mod is None:
        raise Refused(_SAY_NOT_FFN.format(name=name))
    packed = packed_kind(mod) == "w4a16"
    if packed and mod in _STREAM_NORMS:
        return _STREAM_NORMS[mod]
    with torch.no_grad():
        W = dense(mod)  # [D, N]
        wp = torch.from_numpy(w_post.astype(np.float32)).to(W.device)[:, None]
        N = int(W.shape[1])
        out = np.empty(N, dtype=np.float64)
        for c0 in range(0, N, 2048):
            chunk = W[:, c0 : c0 + 2048].float() * wp
            out[c0 : c0 + 2048] = chunk.pow(2).sum(dim=0, dtype=torch.float64).sqrt().cpu().numpy()
    out.setflags(write=False)
    if packed:
        _STREAM_NORMS[mod] = out
    return out


def writers(
    lm: nn.Module,
    layer: int,
    direction: Any,
    k: int,
    *,
    quiet_mask: Any = None,
    max_quiet_share: float = MAX_QUIET_SHARE,
    cosine: bool = False,
) -> List[Tuple[int, float]]:
    """Top-`k` neurons of `layer` ranked by `|(w_post * W_down[:, n]) . u_m|` (or cosine when `cosine=True`), where
    `u_m` is `direction` projected off the layer's quiet mask and normalized. Returns `[(neuron, signed_score), ...]`
    sorted by `abs(score)` descending, ties broken by ascending neuron index."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise Refused(f"writers takes k >= 1; {k!r} was given.")
    k = int(k)
    mq = _number(max_quiet_share, "max_quiet_share")
    if not (math.isfinite(mq) and 0.0 <= mq <= 1.0):
        raise Refused(f"max_quiet_share lies between 0 and 1; {max_quiet_share!r} does not.")
    w_post = _w_post(lm, layer)
    L = int(layer)
    D = int(w_post.shape[0])
    u = _f32(direction, f"The direction given for layer {L} (page layer {L + 1})")
    if u.ndim != 1 or int(u.shape[0]) != D:
        raise Refused(f"A direction at layer {L} (page layer {L + 1}) is a 1D vector of length {D}; "
                      f"{list(u.shape)} was given.")
    if not np.isfinite(u).all():
        raise Refused(_SAY_NOT_FINITE.format(where=f"the direction given for layer {L} (page layer {L + 1})"))
    u64 = u.astype(np.float64)
    u_norm2 = float(np.dot(u64, u64))
    if not u_norm2 > 0.0:
        raise Refused(f"A direction at layer {L} (page layer {L + 1}) must be non-zero.")
    mask = globals()["quiet_mask"](lm, L, extra_mask=quiet_mask)
    q_share = float(np.dot(u64[mask], u64[mask]) / u_norm2)
    if q_share > mq or int((~mask).sum()) == 0:
        raise Refused(_SAY_QUIET_SHARE.format(pct=100.0 * q_share, L=L, L1=L + 1, cap=100.0 * mq))
    u_m = u64.copy()
    u_m[mask] = 0.0
    u_m = u_m / float(np.linalg.norm(u_m))
    name = linear_name(L, "down_proj")
    mod = _linear_of(lm, name)
    if mod is None:
        raise Refused(_SAY_NOT_FFN.format(name=name))
    with torch.no_grad():
        W = dense(mod)  # [D, N]
        N = int(W.shape[1])
        u_eff = torch.from_numpy((u_m * w_post).astype(np.float32)).to(W.device)
        scores = np.empty(N, dtype=np.float64)
        wp_t = torch.from_numpy(w_post.astype(np.float32)).to(W.device)[:, None] if cosine else None
        for c0 in range(0, N, 2048):
            chunk = W[:, c0 : c0 + 2048].float()
            dots = (u_eff @ chunk).double()
            if cosine:
                norms = (chunk * wp_t).pow(2).sum(dim=0, dtype=torch.float64).sqrt().clamp(min=1e-30)
                dots = dots / norms
            scores[c0 : c0 + 2048] = dots.cpu().numpy()
    order = sorted(range(N), key=lambda n: (-abs(float(scores[n])), n))
    return [(int(n), float(scores[n])) for n in order[: min(k, N)]]


# ------------------------------------------------------------------------------------------------------ the gate


def _f32(a: Any, what: str) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().to(torch.float32).cpu().numpy()
    try:
        with np.errstate(all="ignore"):
            return np.asarray(a, dtype=np.float32)
    except (TypeError, ValueError):
        raise Refused(f"{what} is not an array of numbers.") from None


def _rel_changes(mod: nn.Module, idx: np.ndarray, op: np.ndarray, factor: np.ndarray,
                 vectors: Optional[np.ndarray], *, kind: str = "cols") -> np.ndarray:
    """Each column's or row's change over its own length, float32 [K]: |f - 1| for a scaled entry; for a set or added
    one the length of its difference over the base slice's, read from the model. A base slice of no length takes the
    median length of the term's other base slices, and 1.0 when that is nought too."""
    with np.errstate(all="ignore"):
        rel = np.abs(factor.astype(np.float32) - np.float32(1.0))
    if vectors is None:
        return rel
    with torch.no_grad():
        Ws = (dense_rows(mod, _long(idx)) if kind == "rows" else dense_cols(mod, _long(idx))).float()
        V = torch.from_numpy(np.array(vectors)).to(Ws.device)
        is_set = torch.from_numpy(np.array(op == 1)).to(Ws.device)
        lengths = Ws.norm(dim=1).cpu().numpy()
        moved = torch.where(is_set[:, None], V - Ws, V).norm(dim=1).cpu().numpy()
    for j in np.nonzero(op != 0)[0]:
        own = float(lengths[j])
        if not own > 0:
            others = np.delete(lengths, j)
            own = float(np.median(others)) if others.size else 0.0
            own = own if own > 0 else 1.0
        with np.errstate(all="ignore"):
            rel[j] = np.float32(float(moved[j]) / own)
    return rel


def _weight_length(mod: nn.Module) -> float:
    """||W||_F of a linear's own weight, summed from float32 squares a few rows at a time."""
    packed = packed_kind(mod) == "w4a16"
    if packed and mod in _LENGTHS:
        return _LENGTHS[mod]
    with torch.no_grad():
        w = dense(mod)
        total = 0.0
        for r0 in range(0, int(w.shape[0]), 512):
            total += float(w[r0 : r0 + 512].float().pow(2).sum(dtype=torch.float64))
    length = math.sqrt(total)
    if packed:
        _LENGTHS[mod] = length
    return length


def _bound(mod: nn.Module, A: np.ndarray, B: np.ndarray) -> float:
    own = _weight_length(mod)
    reach = float(np.linalg.norm(B.astype(np.float64))) * float(np.linalg.norm(A.astype(np.float64)))
    return reach / own if own > 0 else reach


def _entry(mod: nn.Module, t: Term, rel: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """One term's entry in an edit's report, by arithmetic alone."""
    if t.kind in ("cols", "rows"):
        if rel is None:
            rel = _rel_changes(mod, t.idx, t.op, t.factor, t.vectors, kind=t.kind)
        out: Dict[str, Any] = {
            "linear": t.linear, "kind": t.kind, "layer": int(t.layer), "page_layer": int(t.layer) + 1,
            "count": int(t.idx.shape[0]), "ops": {name: int((t.op == code).sum()) for code, name in enumerate(OPS)},
            "rel_change": {"max": _clean(rel.max()), "median": _clean(np.median(rel))},
            "carried": t.vectors is not None,
        }
        if t.kind == "rows":
            out["form"] = FORM_ROWS.format(which=t.which)
        return out
    return {"linear": t.linear, "kind": "lowrank", "layer": int(t.layer), "page_layer": int(t.layer) + 1,
            "rank": int(t.A.shape[0]), "rel_change": {"bound": _clean(_bound(mod, t.A, t.B))}, "carried": True,
            "form": FORM_LOWRANK}


def _neurons(s: TermSpec) -> List[int]:
    out = []
    for n in s.neurons:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise Refused(f"A neuron is named by a whole number; {n!r} is not one.")
        out.append(int(n))
    return out


def _assemble(lm: nn.Module, specs: Sequence[TermSpec], *, model_id: str, weights_id: Optional[str], note: str,
              caps: bool = True, sized: bool = True) -> Edit:
    """The gate. The tests run in the order of the table of refusals, each over every term before the next, so that
    an input that breaks two rules is told the earlier one. Nothing is installed and nothing is written here.
    `caps` false is a control edit met again (loaded, or handed to a reading): it is let past ONE cap, a column's or
    row's change over its own length, which a set carried to a short slice can pass where its edit did not. Its counts,
    its rank, its bound and its bytes are its edit's by construction, so those caps stand for it too: the mark that
    asks for this sits in a saved edit's report, which is not hashed, and must not be a way round the gate."""
    specs = [specs] if isinstance(specs, TermSpec) else list(specs)
    for s in specs:
        if not isinstance(s, TermSpec):
            raise Refused("An edit is built from recipe terms (weight_edits.silence, scale, set_columns, "
                          "add_to_columns, silence_rows, scale_rows, set_rows, add_to_rows, rows, remove_direction, "
                          f"lowrank, or from_recipe); {type(s).__name__} is not one.")
    for s in specs:
        if s.kind not in ("cols", "rows", "lowrank", "remove_direction") or (
            s.kind in ("cols", "rows") and s.op not in OPS
        ):
            raise Refused(f"A term is a column term, a row term (scale, set or add), remove_direction or a low-rank "
                          f"term; {s.kind!r} with op {s.op!r} is none of them.")
    if not specs:
        raise Refused("An edit with no term changes nothing.")
    raw_names = [_spec_name(s) for s in specs]
    for name in raw_names:
        if "self_attn" in name.split("."):
            raise Refused(_SAY_ATTENTION.format(name=name))
    places = [_parse(name) for name in raw_names]  # refuses a name that is no feed-forward linear
    n_layers = len(lm.layers)
    for L, _which in places:
        if not 0 <= L < n_layers:
            raise Refused(_SAY_LAYER.format(L=L, last=n_layers - 1, n=n_layers))
    names = [linear_name(L, which) for L, which in places]
    for s, name, (_L, which) in zip(specs, names, places):
        if s.kind in ("cols", "remove_direction") and which != "down_proj":
            raise Refused(_SAY_COLS_WHICH.format(name=name))
    for s, name, (_L, which) in zip(specs, names, places):
        if s.kind == "rows" and which not in ("gate_proj", "up_proj"):
            raise Refused(_SAY_ROWS_WHICH.format(name=name))
    for s, name, (_L, which) in zip(specs, names, places):
        if s.kind == "lowrank" and which != "down_proj":
            raise Refused(_SAY_LOWRANK_WHICH.format(name=name, which=which))
    mods: Dict[str, nn.Module] = {}
    for name in names:
        mod = _linear_of(lm, name)
        if mod is None:
            raise Refused(_SAY_NOT_FFN.format(name=name))
        mods[name] = mod
    shape = {name: linear_shape(mod) for name, mod in mods.items()}

    # the columns and rows named: in range, once, and some
    col = [(s, name, L, _neurons(s)) for s, name, (L, _w) in zip(specs, names, places) if s.kind == "cols"]
    drem = [(s, name, L, _neurons(s)) for s, name, (L, _w) in zip(specs, names, places) if s.kind == "remove_direction"]
    row = [(s, name, L, w, _neurons(s)) for s, name, (L, w) in zip(specs, names, places) if s.kind == "rows"]
    low = [(s, name, L) for s, name, (L, _w) in zip(specs, names, places) if s.kind == "lowrank"]
    for _s, name, L, neurons in col + drem:
        N = shape[name][1]
        for n in neurons:
            if not 0 <= n < N:
                raise Refused(_SAY_NEURON.format(n=n, L=L, L1=L + 1, N=N, last=N - 1))
    for _s, name, L, _w, neurons in row:
        N = shape[name][0]
        for n in neurons:
            if not 0 <= n < N:
                raise Refused(_SAY_NEURON.format(n=n, L=L, L1=L + 1, N=N, last=N - 1))
    seen: Dict[str, set] = {}
    for _s, name, L, neurons in col + drem:
        for n in neurons:
            if n in seen.setdefault(name, set()):
                raise Refused(_SAY_TWICE.format(n=n, L=L, L1=L + 1))
            seen[name].add(n)
    seen_rows: Dict[str, set] = {}
    for _s, name, L, _w, neurons in row:
        for n in neurons:
            if n in seen_rows.setdefault(name, set()):
                raise Refused(_SAY_TWICE_ROW.format(n=n, name=name, L1=L + 1))
            seen_rows[name].add(n)
    for _s, _name, _L, neurons in col + drem:
        if not neurons:
            raise Refused(_SAY_NO_COLUMN)
    for _s, _name, _L, _w, neurons in row:
        if not neurons:
            raise Refused(_SAY_NO_COLUMN)

    # the numbers given: as many as the neurons, finite, within the factor's cap, of a slice's length
    def _parse_slice_vals(s: TermSpec, L: int, neurons: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        K = len(neurons)
        if s.op == "scale":
            f = s.factor
            if isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            if isinstance(f, np.ndarray) and f.ndim == 0:
                f = f.item()
            if isinstance(f, (list, tuple, np.ndarray)):
                f = _f32(f, f"The factors given for layer {L} (page layer {L + 1})").reshape(-1)
            else:
                f = _f32([_number(f, "A factor")] * K, "A factor")
            if f.shape[0] != K:
                raise Refused(f"{f.shape[0]} factor{'' if f.shape[0] == 1 else 's'} were given for {K} "
                              f"neuron{'' if K == 1 else 's'} of layer {L} (page layer {L + 1}): one a neuron in the "
                              "order listed, or one number for all.")
            return (f, None)
        v = _f32(s.vectors, f"The vectors given for layer {L} (page layer {L + 1})")
        if v.ndim == 1 and K == 1:
            v = v.reshape(1, -1)
        if v.ndim != 2 or v.shape[0] != K:
            width_name = "in" if s.kind == "rows" else "out"
            raise Refused(f"The vectors of a set or add term are one array [neurons, {width_name}], row j for the "
                          f"j-th neuron as listed; {list(v.shape)} was given for {K} neuron"
                          f"{'' if K == 1 else 's'} of layer {L} (page layer {L + 1}).")
        return (None, v)

    given: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = [_parse_slice_vals(s, L, neurons)
                                                                     for s, _name, L, neurons in col]
    given_rows: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = [_parse_slice_vals(s, L, neurons)
                                                                          for s, _name, L, _w, neurons in row]
    low_ab = [(_f32(s.A, f"A of the low-rank term on {name}"), _f32(s.B, f"B of the low-rank term on {name}"))
              for s, name, _L in low]

    for (_s, _name, L, neurons), (f, v) in zip(col, given):
        if f is not None and not np.isfinite(f).all():
            n = neurons[int(np.nonzero(~np.isfinite(f))[0][0])]
            raise Refused(_SAY_NOT_FINITE.format(where=f"the factor of neuron {n}, layer {L} (page layer {L + 1})"))
        if v is not None and not np.isfinite(v).all():
            n = neurons[int(np.nonzero(~np.isfinite(v).all(axis=1))[0][0])]
            raise Refused(_SAY_NOT_FINITE.format(
                where=f"the vector given for neuron {n}, layer {L} (page layer {L + 1})"))
    for (_s, name, L, _w, neurons), (f, v) in zip(row, given_rows):
        if f is not None and not np.isfinite(f).all():
            n = neurons[int(np.nonzero(~np.isfinite(f))[0][0])]
            raise Refused(_SAY_NOT_FINITE.format(where=f"the factor of neuron {n} on {name} (page layer {L + 1})"))
        if v is not None and not np.isfinite(v).all():
            n = neurons[int(np.nonzero(~np.isfinite(v).all(axis=1))[0][0])]
            raise Refused(_SAY_NOT_FINITE.format(where=f"the vector given for neuron {n} on {name} (page layer {L + 1})"))
    for (_s, name, _L), (A, B) in zip(low, low_ab):
        for letter, arr in (("A", A), ("B", B)):
            if not np.isfinite(arr).all():
                raise Refused(_SAY_NOT_FINITE.format(where=f"{letter} of the low-rank term on {name}"))
    # normalize -0.0 -> 0.0 before sorting low-rank parts so -0.0 never changes merge order or edit id
    low_ab = [(A + np.float32(0.0), B + np.float32(0.0)) for A, B in low_ab]

    # the cap on a factor is one of the tests a whole made of a plant and an edit is not asked again (`sized`): a
    # plant's 8 and an edit's 60 add to 67, and an edit's 64 at dose 1.01 is 64.63, where each part met the cap alone
    for (_s, _name, _L, _neurons_), (f, _v) in zip(col, given) if sized else ():
        if f is not None and (np.abs(f) > MAX_FACTOR).any():
            raise Refused(_SAY_FACTOR.format(cap=MAX_FACTOR, f=float(f[np.abs(f) > MAX_FACTOR][0])))
    for (_s, _name, _L, _w, _neurons_), (f, _v) in zip(row, given_rows) if sized else ():
        if f is not None and (np.abs(f) > MAX_FACTOR).any():
            raise Refused(_SAY_FACTOR.format(cap=MAX_FACTOR, f=float(f[np.abs(f) > MAX_FACTOR][0])))
    for (_s, name, L, neurons), (_f, v) in zip(col, given):
        if v is not None and v.shape[1] != shape[name][0]:
            raise Refused(_SAY_VECTOR.format(L=L, out=shape[name][0], n=neurons[0], k=int(v.shape[1])))
    for (_s, name, L, w, neurons), (_f, v) in zip(row, given_rows):
        if v is not None and v.shape[1] != shape[name][1]:
            raise Refused(_SAY_ROW_VECTOR.format(L=L, which=w, n_in=shape[name][1], n=neurons[0], k=int(v.shape[1])))

    # masked direction removal: validate, project off quiet_mask, divide by w_post only on ~mask, lower to cols add
    dir_reports: List[Dict[str, Any]] = []
    for s, name, L, neurons in drem:
        u = _f32(s.direction, f"The direction given for layer {L} (page layer {L + 1})")
        if u.ndim != 1 or int(u.shape[0]) != shape[name][0]:
            raise Refused(_SAY_VECTOR.format(L=L, out=shape[name][0], n=neurons[0],
                                             k=int(u.shape[0]) if u.ndim == 1 else int(u.size)))
        if not np.isfinite(u).all():
            raise Refused(_SAY_NOT_FINITE.format(where=f"the direction given for layer {L} (page layer {L + 1})"))
        mq = MAX_QUIET_SHARE if s.max_quiet_share is None else _number(s.max_quiet_share, "max_quiet_share")
        if not (math.isfinite(mq) and 0.0 <= mq <= 1.0):
            raise Refused(f"max_quiet_share lies between 0 and 1; {s.max_quiet_share!r} does not.")
        u64 = u.astype(np.float64)
        u_norm2 = float(np.dot(u64, u64))
        if not u_norm2 > 0.0:
            raise Refused(f"A direction for remove_direction at layer {L} (page layer {L + 1}) must be non-zero.")
        mask = quiet_mask(lm, L, extra_mask=s.quiet_mask)
        q_share = float(np.dot(u64[mask], u64[mask]) / u_norm2)
        if q_share > mq or int((~mask).sum()) == 0:
            raise Refused(_SAY_QUIET_SHARE.format(pct=100.0 * q_share, L=L, L1=L + 1, cap=100.0 * mq))
        u_m = u64.copy()
        u_m[mask] = 0.0
        u_m = u_m / float(np.linalg.norm(u_m))
        w_post = _w_post(lm, L)
        with torch.no_grad():
            Wc = dense_cols(mods[name], _long(neurons)).float().cpu().numpy().astype(np.float64)
        S = Wc * w_post[None, :]
        proj = S @ u_m
        dS = -proj[:, None] * u_m[None, :]
        dW = np.zeros_like(dS, dtype=np.float64)
        keep = ~mask
        dW[:, keep] = dS[:, keep] / w_post[None, keep]
        dW32 = dW.astype(np.float32) + np.float32(0.0)
        if not np.isfinite(dW32).all():
            raise Refused(_SAY_NOT_FINITE.format(where=f"the column delta from remove_direction on {name}"))
        raw_norms = np.linalg.norm(dW32.astype(np.float64), axis=1)
        stream_norms = np.linalg.norm(dS, axis=1)
        base_stream_norms = np.linalg.norm(S, axis=1)
        s_rel = np.where(base_stream_norms > 0, stream_norms / np.where(base_stream_norms > 0, base_stream_norms, 1.0), 0.0)
        dir_reports.append({
            "linear": name, "layer": L, "page_layer": L + 1, "neurons": len(neurons),
            "quiet_share": _clean(q_share), "max_quiet_share": _clean(mq),
            "masked_dims": int(mask.sum()), "total_dims": int(mask.size),
            "raw_norm": {"max": _clean(raw_norms.max()), "median": _clean(np.median(raw_norms)),
                         "total": _clean(np.linalg.norm(dW32.astype(np.float64)))},
            "stream_norm": {"max": _clean(stream_norms.max()), "median": _clean(np.median(stream_norms)),
                            "total": _clean(np.linalg.norm(dS))},
            "stream_rel_change": {"max": _clean(s_rel.max()), "median": _clean(np.median(s_rel))},
        })
        col.append((TermSpec("cols", L, "down_proj", tuple(neurons), "add"), name, L, neurons))
        given.append((None, dW32))

    # one column term a linear, its columns ascending
    merged: Dict[str, Dict[str, Any]] = {}
    for (s, name, L, neurons), (f, v) in zip(col, given):
        m = merged.setdefault(name, {"layer": L, "idx": [], "op": [], "factor": [], "vectors": []})
        code = OPS.index(s.op)
        m["idx"] += neurons
        m["op"] += [code] * len(neurons)
        m["factor"] += [float(x) for x in f] if f is not None else [1.0] * len(neurons)
        m["vectors"] += [None] * len(neurons) if v is None else list(v)
    cols: Dict[str, Dict[str, Any]] = {}
    for name in sorted(merged):
        m = merged[name]
        order = np.argsort(np.asarray(m["idx"], dtype=np.int64), kind="stable")
        idx = np.asarray(m["idx"], dtype=np.int64)[order]
        op = np.asarray(m["op"], dtype=np.uint8)[order]
        factor = np.asarray(m["factor"], dtype=np.float32)[order] + np.float32(0.0)  # -0.0 is 0.0: one edit, one id
        vectors = None
        if any(v is not None for v in m["vectors"]):
            vectors = np.zeros((len(order), shape[name][0]), dtype=np.float32)
            for j, k in enumerate(order):
                if m["vectors"][int(k)] is not None:
                    vectors[j] = m["vectors"][int(k)]
            vectors += np.float32(0.0)  # -0.0 is 0.0
        cols[name] = {"layer": m["layer"], "idx": idx, "op": op, "factor": factor, "vectors": vectors,
                      "rel": _rel_changes(mods[name], idx, op, factor, vectors, kind="cols")}

    # one row term a linear, its rows ascending
    merged_rows: Dict[str, Dict[str, Any]] = {}
    for (s, name, L, w, neurons), (f, v) in zip(row, given_rows):
        m = merged_rows.setdefault(name, {"layer": L, "which": w, "idx": [], "op": [], "factor": [], "vectors": []})
        code = OPS.index(s.op)
        m["idx"] += neurons
        m["op"] += [code] * len(neurons)
        m["factor"] += [float(x) for x in f] if f is not None else [1.0] * len(neurons)
        m["vectors"] += [None] * len(neurons) if v is None else list(v)
    rows_m: Dict[str, Dict[str, Any]] = {}
    for name in sorted(merged_rows):
        m = merged_rows[name]
        order = np.argsort(np.asarray(m["idx"], dtype=np.int64), kind="stable")
        idx = np.asarray(m["idx"], dtype=np.int64)[order]
        op = np.asarray(m["op"], dtype=np.uint8)[order]
        factor = np.asarray(m["factor"], dtype=np.float32)[order] + np.float32(0.0)
        vectors = None
        if any(v is not None for v in m["vectors"]):
            vectors = np.zeros((len(order), shape[name][1]), dtype=np.float32)
            for j, k in enumerate(order):
                if m["vectors"][int(k)] is not None:
                    vectors[j] = m["vectors"][int(k)]
            vectors += np.float32(0.0)
        rows_m[name] = {"layer": m["layer"], "which": m["which"], "idx": idx, "op": op, "factor": factor,
                        "vectors": vectors, "rel": _rel_changes(mods[name], idx, op, factor, vectors, kind="rows")}

    if caps:
        for name, c in cols.items():
            over = np.nonzero(~(c["rel"] <= MAX_REL_CHANGE))[0]
            if over.size:
                j = int(over[0])
                raise Refused(_SAY_REL.format(n=int(c["idx"][j]), L=c["layer"], L1=c["layer"] + 1,
                                              x=float(c["rel"][j]), cap=MAX_REL_CHANGE))
        for name, c in rows_m.items():
            over = np.nonzero(~(c["rel"] <= MAX_REL_CHANGE))[0]
            if over.size:
                j = int(over[0])
                raise Refused(_SAY_REL.format(n=int(c["idx"][j]), L=c["layer"], L1=c["layer"] + 1,
                                              x=float(c["rel"][j]), cap=MAX_REL_CHANGE))
    # `sized` false is a whole the harness made of a plant and an edit (`stack(plant_first=True)`), each of which
    # met these tests alone: the whole is not asked them again, or what a planted session refuses would say where
    # its plant is (an edit of 16 terms that leaves the plant's linear alone would be 17 with it)
    for name, c in cols.items() if sized else ():
        f0 = float(c["factor"][0])
        every, alike = len(c["idx"]) == shape[name][1], bool((c["factor"] == c["factor"][0]).all())
        if every and alike and f0 > 0 and not c["op"].any():
            raise Refused(_SAY_UNIFORM.format(L=c["layer"], f=f0))
    for name, c in rows_m.items() if sized else ():
        if c["which"] == "up_proj":
            f0 = float(c["factor"][0])
            every, alike = len(c["idx"]) == shape[name][0], bool((c["factor"] == c["factor"][0]).all())
            if every and alike and f0 > 0 and not c["op"].any():
                raise Refused(_SAY_UNIFORM_UP.format(L=c["layer"], f=f0))
    low_names = sorted({name for _s, name, _L in low})
    n_terms = len(cols) + len(rows_m) + len(low_names)
    if sized and n_terms > MAX_TERMS:
        raise Refused(_SAY_CAP.format(cap=MAX_TERMS, what="terms", k=n_terms))
    for c in cols.values() if sized else ():
        if len(c["idx"]) > MAX_COLUMNS:
            raise Refused(_SAY_CAP.format(cap=MAX_COLUMNS, what="columns a term", k=len(c["idx"])))
    for c in rows_m.values() if sized else ():
        if len(c["idx"]) > MAX_COLUMNS:
            raise Refused(_SAY_CAP.format(cap=MAX_COLUMNS, what="rows a term", k=len(c["idx"])))
    for c in cols.values() if sized else ():
        if int((c["op"] != 0).sum()) > MAX_VECTOR_COLUMNS:
            raise Refused(_SAY_CAP.format(cap=MAX_VECTOR_COLUMNS, what="set or added columns a term",
                                          k=int((c["op"] != 0).sum())))
    for c in rows_m.values() if sized else ():
        if int((c["op"] != 0).sum()) > MAX_VECTOR_COLUMNS:
            raise Refused(_SAY_CAP.format(cap=MAX_VECTOR_COLUMNS, what="set or added rows a term",
                                          k=int((c["op"] != 0).sum())))

    # one low-rank term a linear: several are laid side by side, in an order their own hashes fix
    for (_s, name, _L), (A, B) in zip(low, low_ab):
        n_out, n_in = shape[name]
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] < 1 or A.shape[1] != n_in or tuple(B.shape) != (n_out, A.shape[0]):
            raise Refused(_SAY_LOWRANK_SHAPE.format(name=name, n_in=n_in, n_out=n_out, a=list(A.shape),
                                                    b=list(B.shape)))
    lows: Dict[str, Dict[str, Any]] = {}
    for name in low_names:
        parts = sorted(((array_sha256(A, "f4"), array_sha256(B, "f4"), i) for i, ((_s, nm, _L), (A, B))
                        in enumerate(zip(low, low_ab)) if nm == name))
        A = np.concatenate([low_ab[i][0] for _a, _b, i in parts], axis=0) + np.float32(0.0)
        B = np.concatenate([low_ab[i][1] for _a, _b, i in parts], axis=1) + np.float32(0.0)
        lows[name] = {"layer": _parse(name)[0], "A": A, "B": B}
    for name, c in lows.items() if sized else ():
        if c["A"].shape[0] > MAX_RANK:
            raise Refused(_SAY_RANK.format(cap=MAX_RANK, r=int(c["A"].shape[0])))
    for name, c in lows.items() if sized else ():
        x = _bound(mods[name], c["A"], c["B"])
        if not x <= MAX_REL_CHANGE:
            raise Refused(_SAY_BOUND.format(name=name, x=x, cap=MAX_REL_CHANGE))
    carried = sum(c["vectors"].nbytes for c in cols.values() if c["vectors"] is not None)
    carried += sum(c["vectors"].nbytes for c in rows_m.values() if c["vectors"] is not None)
    carried += sum(c["A"].nbytes + c["B"].nbytes for c in lows.values())
    if sized and carried > MAX_PAYLOAD_BYTES:
        raise Refused(_SAY_BYTES.format(mb=carried / 2**20, cap_mb=MAX_PAYLOAD_BYTES / 2**20))
    touched = sorted(set(cols) | set(rows_m) | set(lows))
    pins = {name: _pin(mods[name], name) for name in touched}

    terms: List[Term] = []
    entries: List[Dict[str, Any]] = []
    for name in touched:  # sorted by (linear, kind): "cols" before "lowrank" before "rows"
        if name in cols:
            c = cols[name]
            t = Term(name, "cols", c["layer"], "down_proj", idx=_ro(c["idx"], np.int64), op=_ro(c["op"], np.uint8),
                     factor=_ro(c["factor"], np.float32),
                     vectors=None if c["vectors"] is None else _ro(c["vectors"], np.float32))
            terms.append(t)
            entries.append(_entry(mods[name], t, c["rel"]))
        if name in lows:
            c = lows[name]
            t = Term(name, "lowrank", c["layer"], "down_proj", A=_ro(c["A"], np.float32), B=_ro(c["B"], np.float32))
            terms.append(t)
            entries.append(_entry(mods[name], t))
        if name in rows_m:
            c = rows_m[name]
            t = Term(name, "rows", c["layer"], c["which"], idx=_ro(c["idx"], np.int64), op=_ro(c["op"], np.uint8),
                     factor=_ro(c["factor"], np.float32),
                     vectors=None if c["vectors"] is None else _ro(c["vectors"], np.float32))
            terms.append(t)
            entries.append(_entry(mods[name], t, c["rel"]))
    base = {"model": model_id, "weights_id": weights_id, "tensors": pins}
    report: Dict[str, Any] = {"terms": entries}
    if dir_reports:
        report["direction_removal"] = dir_reports
    if not caps:
        report["caps"] = CONTROL_CAPS
    return Edit(_edit_id(base, terms), tuple(terms), base, str(note or ""), report)


def build(lm: nn.Module, specs: Sequence[TermSpec], *, model_id: str, weights_id: Optional[str],
          note: str = "") -> Edit:
    """An edit from recipe terms, against the text model `lm`: the gate, then the arrays (several terms on one linear
    merged, indices ascending), the pins of the tensors it touches, its id and its report. `Refused`, in the words of
    the gate, for whatever is not built or a cap forbids. Nothing is installed in the model and nothing is written."""
    return _assemble(lm, specs, model_id=model_id, weights_id=weights_id, note=note)


# ------------------------------------------------------------------------------------------------- save and load


def edits_dir() -> Path:
    """Where saved edits live: $THEODOLITE_EDITS_DIR, else ${XDG_DATA_HOME:-~/.local/share}/theodolite/edits, beside
    the book and outside the checkout. Read from the environment at every call."""
    given = os.environ.get(EDITS_ENV)
    if given:
        return Path(given).expanduser()
    data = os.environ.get("XDG_DATA_HOME")
    return (Path(data).expanduser() if data else Path.home() / ".local" / "share") / "theodolite" / "edits"


def save(edit: Edit, folder: Optional[Path] = None) -> Path:
    """Write an edit to `folder` (default `edits_dir() / edit.id`): manifest.json, and payload.safetensors when some
    term carries arrays. Returns the folder. A new folder is staged in a sibling temp directory and renamed in place so
    a crash mid-write leaves no half-written edit. A folder inside the checkout is refused: a payload can carry pieces
    of Gemma's columns and is never committed."""
    folder = Path(folder).expanduser() if folder is not None else edits_dir() / edit.id
    if _ROOT == folder.resolve() or _ROOT in folder.resolve().parents:
        raise Refused(f"Saved edits are kept outside the checkout ({_ROOT}): a payload can carry pieces of Gemma's "
                      f"columns and is never committed. {folder} is inside it; point ${EDITS_ENV} elsewhere.")
    manifest = edit.manifest()
    manifest["saved_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    canonical(manifest)  # JSON-clean, or a ValueError here and nothing written
    payload = _payload(edit)
    if folder.exists():
        if payload:
            from safetensors.numpy import save_file

            tmp = folder / "payload.safetensors.tmp"
            save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in payload.items()}, str(tmp))
            os.replace(tmp, folder / "payload.safetensors")
        write_json_atomic(folder / "manifest.json", manifest, indent=1)
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{folder.name}.", dir=str(folder.parent)))
    try:
        if payload:
            from safetensors.numpy import save_file

            save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in payload.items()},
                      str(stage / "payload.safetensors"))
        write_json_atomic(stage / "manifest.json", manifest, indent=1)
        try:
            os.rename(stage, folder)
        except OSError:
            if not folder.is_dir():
                raise
            if payload:
                os.replace(stage / "payload.safetensors", folder / "payload.safetensors")
            os.replace(stage / "manifest.json", folder / "manifest.json")
            shutil.rmtree(stage, ignore_errors=True)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return folder


def _find_saved(where: Union[str, Path]) -> Path:
    """The folder of a saved edit, given as a folder, or as an id or its first 12 or more hex under `edits_dir()`."""
    if Path(where).is_dir():
        return Path(where)
    name = str(where)
    if re.fullmatch(r"[0-9a-f]{12,64}", name):
        root = edits_dir()
        hits = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith(name)) if root.is_dir() else []
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise Refused(f"No saved edit begins {name} under {root}.")
        raise Refused(f"{len(hits)} saved edits begin {name} under {root}: give more of the id.")
    raise Refused(_SAY_FORMAT.format(path=where))


def _read_manifest(folder: Path) -> Dict[str, Any]:
    """A saved edit's manifest, its shape checked far enough to be read; its format and version are the caller's to
    compare (`load` says an edit of another base before it says a manifest of another format)."""
    try:
        m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        ok = (isinstance(m, dict) and isinstance(m["id"], str) and isinstance(m["base"], dict)
              and isinstance(m["base"]["tensors"], dict) and isinstance(m["terms"], list)
              and all(isinstance(t, dict) and t["kind"] in ("cols", "rows", "lowrank") and isinstance(t["linear"], str)
                      and isinstance(t["sha256"], dict) for t in m["terms"]))
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise Refused(_SAY_FORMAT.format(path=folder))
    return m


def _specs_of(terms: Sequence[Term]) -> List[TermSpec]:
    """An edit's terms as the recipe terms that build them again, so that an edit met a second time (loaded, or
    handed to a reading as an object) passes the gate as a built one does. `ValueError` for a term that is not laid
    out as a term is."""
    specs: List[TermSpec] = []
    for t in terms:
        if t.kind in ("cols", "rows"):
            idx, op, factor = np.asarray(t.idx), np.asarray(t.op), np.asarray(t.factor)
            if not (idx.ndim == op.ndim == factor.ndim == 1 and len(idx) == len(op) == len(factor)):
                raise ValueError("a slice term's lists are of one length")
            for code, opname in enumerate(OPS):
                at = np.nonzero(op == code)[0]
                if at.size and code == 0:
                    specs.append(TermSpec(t.kind, t.layer, t.which, tuple(int(n) for n in idx[at]), "scale",
                                          [float(f) for f in factor[at]]))
                elif at.size:
                    if t.vectors is None or np.asarray(t.vectors).ndim != 2:
                        raise ValueError("a set or added slice with no vector")
                    specs.append(TermSpec(t.kind, t.layer, t.which, tuple(int(n) for n in idx[at]), opname,
                                          vectors=np.asarray(t.vectors)[at]))
            if int(op.max(initial=0)) >= len(OPS):
                raise ValueError("a slice's op is 0, 1 or 2")
        elif t.kind == "lowrank":
            specs.append(TermSpec("lowrank", t.layer, t.which, A=t.A, B=t.B))
        else:
            specs.append(TermSpec(str(t.kind), t.layer, t.which))
    return specs


def _carry_info_report(dest: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Carry purely informational report fields (never caps) across `gate_again` / `load`."""
    dr = src.get("direction_removal")
    if isinstance(dr, list) and all(isinstance(x, dict) for x in dr):
        dest["direction_removal"] = dr
    parents = src.get("parents")
    if isinstance(parents, list) and all(isinstance(x, str) for x in parents):
        dest["parents"] = list(parents)


def static_caps(edit: Edit) -> Optional[str]:
    """The gate's refusals that need no weights, asked of an edit put together with no model at hand (an export off
    the model): None when all hold, else why not, in the gate's words. The caps on an edit's size (terms, columns or
    rows a term, set or added ones, rank, bytes carried); factors, vectors and low-rank pairs that are finite, and
    factors within the cap on one; a kind of term this kernel builds, on a linear that takes it; no neuron named
    twice. What cannot be asked without the weights, and so is not: how far a column, a row or a low-rank term moves
    its linear (`MAX_REL_CHANGE`), whether a scaling of every column is the no-op the norm undoes, and whether a
    neuron exists. (How far a set or added vector moves its column, and a low-rank term's bound, need the weights:
    not asked here.)"""
    try:
        low_names = {t.linear for t in edit.terms if t.kind == "lowrank"}
        n_terms = sum(1 for t in edit.terms if t.kind in ("cols", "rows")) + len(low_names)
        if n_terms > MAX_TERMS:
            return _SAY_CAP.format(cap=MAX_TERMS, what="terms", k=n_terms)
        for t in edit.terms:  # a low-rank term that is not a pair has no rank to ask
            if t.kind == "lowrank" and (np.ndim(t.A) != 2 or np.ndim(t.B) != 2
                                        or np.shape(t.A)[0] != np.shape(t.B)[1]):
                return f"The low-rank term on {t.linear} is not a pair A [rank, in], B [out, rank]."
        for name in sorted(low_names):  # the gate joins the low-rank terms of one linear before it asks their rank
            rank = sum(int(np.asarray(t.A).shape[0]) for t in edit.terms if t.kind == "lowrank" and t.linear == name)
            if rank > MAX_RANK:
                return _SAY_RANK.format(cap=MAX_RANK, r=rank)
        seen: Dict[Tuple[str, str], set] = {}
        carried = 0
        for t in edit.terms:
            if t.kind in ("cols", "rows"):
                what = "columns" if t.kind == "cols" else "rows"
                if t.kind == "cols" and t.which != "down_proj":
                    return _SAY_COLS_WHICH.format(name=t.linear)
                if t.kind == "rows" and t.which not in ("gate_proj", "up_proj"):
                    return _SAY_ROWS_WHICH.format(name=t.linear)
                idx, op, f = np.asarray(t.idx), np.asarray(t.op), np.asarray(t.factor, dtype=np.float64)
                if idx.ndim != 1 or idx.size == 0 or not (len(idx) == len(op) == len(f)):
                    return _SAY_NO_COLUMN
                if len(idx) > MAX_COLUMNS:
                    return _SAY_CAP.format(cap=MAX_COLUMNS, what=f"{what} a term", k=len(idx))
                if int((op != 0).sum()) > MAX_VECTOR_COLUMNS:
                    return _SAY_CAP.format(cap=MAX_VECTOR_COLUMNS, what=f"set or added {what} a term",
                                           k=int((op != 0).sum()))
                if int(op.max(initial=0)) >= len(OPS):
                    return f"A term on {t.linear} holds an operation this kernel does not know."
                if not np.isfinite(f).all():
                    return _SAY_NOT_FINITE.format(where=f"a factor on {t.linear}")
                if (np.abs(f) > MAX_FACTOR).any():
                    return _SAY_FACTOR.format(cap=MAX_FACTOR, f=float(f[np.abs(f) > MAX_FACTOR][0]))
                # a scaled column or row changes by |f - 1| times its own length, whatever the weights are
                far = (op == 0) & (np.abs(f - 1.0) > MAX_REL_CHANGE)
                if far.any():
                    return (f"A neuron of {t.linear} would change by {float(np.abs(f - 1.0)[far][0]):.0f} times its own "
                            f"length; an edit takes at most {MAX_REL_CHANGE:g}.")
                if (op != 0).any() and t.vectors is None:
                    return f"A term on {t.linear} sets or adds {what} and carries no vectors."
                if t.vectors is not None:
                    if not np.isfinite(np.asarray(t.vectors)).all():
                        return _SAY_NOT_FINITE.format(where=f"a vector on {t.linear}")
                    carried += int(np.asarray(t.vectors).nbytes)
                named = {int(n) for n in idx.tolist()}
                was = seen.setdefault((t.linear, t.kind), set())
                if len(named) != len(idx) or was & named:
                    return f"A neuron of {t.linear} is named twice in this edit: one change a {what[:-1]}."
                was |= named
            elif t.kind == "lowrank":
                if t.which != "down_proj":
                    return f"A low-rank term on {t.linear} is not one this kernel builds: it takes a down_proj."
                A, B = np.asarray(t.A), np.asarray(t.B)
                if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
                    return f"The low-rank term on {t.linear} is not a pair A [rank, in], B [out, rank]."
                if A.shape[0] > MAX_RANK:
                    return _SAY_RANK.format(cap=MAX_RANK, r=int(A.shape[0]))
                if not (np.isfinite(A).all() and np.isfinite(B).all()):
                    return _SAY_NOT_FINITE.format(where=f"the low-rank term on {t.linear}")
                carried += int(A.nbytes + B.nbytes)
            else:
                return f"A term of kind {t.kind!r} is not one this kernel builds."
        if carried > MAX_PAYLOAD_BYTES:
            return _SAY_BYTES.format(mb=carried / 2**20, cap_mb=MAX_PAYLOAD_BYTES / 2**20)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        return f"The edit is not laid out as an edit's terms are ({type(exc).__name__})."
    return None


def gate_again(lm: nn.Module, edit: Edit, *, model_id: str, weights_id: Optional[str], sized: bool = True) -> Edit:
    """An edit that was not built here and now (one handed to a reading as an object), passed through the gate as a
    loaded one is: its terms are taken as recipe terms and built again against `lm`, and the id that comes out must be
    the id the object carries. So the ledger never holds an id that is not the hash of what ran, and an object made by
    hand cannot carry what the gate refuses. A control edit (its report says the caps were not applied) is let past
    the one cap `_assemble` names. Returns the edit as built again, with a report worked from the model. `Refused` in
    the gate's words, or `load`'s for an id that is not the edit's own."""
    if not isinstance(edit, Edit):
        raise Refused(f"An edit handed to a reading is a weight_edits.Edit; {type(edit).__name__} is not one.")
    report = edit.report if isinstance(edit.report, dict) else {}
    try:
        specs = _specs_of(edit.terms)
    except (ValueError, TypeError, AttributeError, IndexError):
        raise Refused(f"The edit {str(edit.id)[:12]} is not laid out as an edit's terms are: build it with "
                      "weight_edits.build.") from None
    # `sized` false: a part of a whole made under a plant (`stack(plant_first=True)`). The kernel built it in this
    # reading; a control edit is gated a second time only when there is a plant, so asking it the tests on an edit's
    # size again would refuse under a plant what reads without one.
    again = _assemble(lm, specs, model_id=model_id, weights_id=weights_id, note=str(edit.note or ""),
                      caps=report.get("caps") != CONTROL_CAPS, sized=sized)
    if again.id != edit.id:
        raise Refused(f"The edit handed in says it is {str(edit.id)[:12]} and hashes to {again.id[:12]}.")
    _carry_info_report(again.report, report)
    return again


def load(where: Union[str, Path], lm: nn.Module, *, model_id: str, weights_id: Optional[str]) -> Edit:
    """A saved edit, read against the loaded model. `where` is a folder, or an id (or its first 12 or more hex) under
    `edits_dir()`. Every array is rebuilt and its sha256 compared, the id is worked again, and the base is compared
    with the model (its id, its weights' id, the pin of each linear the edit touches); then the edit goes through the
    gate as a built one does. `Refused`, in the gate's words, on any difference: an edit is a difference from one base
    and is not carried to another."""
    folder = _find_saved(where)
    m = _read_manifest(folder)
    id12 = m["id"][:12]
    try:
        base = m["base"]
        if base.get("model") != model_id or base.get("weights_id") != weights_id:
            raise Refused(_SAY_OTHER_BASE.format(a=base.get("model"), ia=base.get("weights_id"), b=model_id,
                                                 ib=weights_id))
        for name in sorted(base["tensors"]):
            then = str(base["tensors"][name])
            mod = _linear_of(lm, name)
            if mod is None:
                raise Refused(f"This edit was made against other weights: {name} then hashed to {then[:12]}, and the "
                              "loaded model has no such linear.")
            now = _pin(mod, name)
            if now != then:
                raise Refused(_SAY_OTHER_PIN.format(name=name, a12=then[:12], b12=now[:12]))
        payload: Dict[str, np.ndarray] = {}
        carried = [t for t in m["terms"] if t["kind"] == "lowrank" or t.get("vectors")]
        if carried:
            first = carried[0]
            what = f'{"A" if first["kind"] == "lowrank" else "vectors"} array ({first["linear"]})'
            try:
                from safetensors.numpy import load_file

                payload = load_file(str(folder / "payload.safetensors"))
            except Exception as exc:  # absent, cut short, or its header changed: it cannot be hashed at all
                raise Refused(_SAY_ARRAY.format(id12=id12, array=what)[:-1]
                              + f" (payload.safetensors cannot be read: {type(exc).__name__}).") from None
        terms: List[Term] = []
        for t in m["terms"]:
            name, sums = t["linear"], t["sha256"]
            layer, which = _parse(name)

            def same(key: str, arr: np.ndarray, tag: str) -> None:
                if array_sha256(arr, tag) != sums.get(key):
                    raise Refused(_SAY_ARRAY.format(id12=id12, array=f"{key} array ({name})"))

            if t["kind"] in ("cols", "rows"):
                idx_key = "columns" if t["kind"] == "cols" else "rows"
                idx = _ro([int(n) for n in t[idx_key]], np.int64)
                op = _ro([OPS.index(o) for o in t["ops"]], np.uint8)
                factor = _ro([float(f) for f in t["factors"]], np.float32)
                same(idx_key, idx, "i8")
                same("ops", op, "u1")
                same("factors", factor, "f4")
                vectors = None
                if t.get("vectors"):
                    if t["vectors"] not in payload:
                        raise Refused(_SAY_ARRAY.format(id12=id12, array=f"vectors array ({name})"))
                    vectors = _ro(payload[t["vectors"]], np.float32)
                    same("vectors", vectors, "f4")
                elif sums.get("vectors") is not None:
                    raise Refused(_SAY_ARRAY.format(id12=id12, array=f"vectors array ({name})"))
                if not (idx.ndim == op.ndim == factor.ndim == 1 and len(idx) == len(op) == len(factor)):
                    raise ValueError("a slice term's lists are of one length")
                terms.append(Term(name, t["kind"], layer, which, idx=idx, op=op, factor=factor, vectors=vectors))
            else:
                for key in ("A", "B"):
                    if t.get(key) not in payload:
                        raise Refused(_SAY_ARRAY.format(id12=id12, array=f"{key} array ({name})"))
                A, B = _ro(payload[t["A"]], np.float32), _ro(payload[t["B"]], np.float32)
                same("A", A, "f4")
                same("B", B, "f4")
                terms.append(Term(name, "lowrank", layer, which, A=A, B=B))
        specs = _specs_of(terms)
        said = _edit_id(base, terms)
        if said != m["id"]:
            raise Refused(_SAY_ID.format(id12=id12, other12=said[:12]))
    except Refused:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        raise Refused(_SAY_FORMAT.format(path=folder)) from None
    if m.get("format") != FORMAT or m.get("version") != VERSION:
        raise Refused(_SAY_FORMAT.format(path=folder))
    report = m.get("report") if isinstance(m.get("report"), dict) else {}
    edit = _assemble(lm, specs, model_id=model_id, weights_id=weights_id, note=str(m.get("note") or ""),
                     caps=report.get("caps") != CONTROL_CAPS)
    if edit.id != m["id"]:  # arrays that hash as the manifest says but are not laid out as `build` lays them
        raise Refused(_SAY_ID.format(id12=id12, other12=edit.id[:12]))
    _carry_info_report(edit.report, report)
    return edit


# ------------------------------------------------------------------------------------- what a dose does to an edit


def _asked(t: Term, dose: float) -> np.ndarray:
    """A column or row term's scaling factor g for each entry at this dose, in Python's floats (float64): a scaled
    entry 1 + d (f - 1); a set entry 1 - d, the old slice's share that stays; an added entry 1."""
    d = float(dose)
    g = 1.0 + d * (t.factor.astype(np.float64) - 1.0)
    g[t.op == 1] = 1.0 - d
    g[t.op == 2] = 1.0
    return g


def exact_at(edit: Edit, dose: float, dtype: Any) -> dict:
    """What the cast to the model's dtype does to an edit's scaling factors at this dose. `factors`: the distinct
    pairs [g asked, g as applied], ascending, the first 8. `input_side`: "none" (no column/row term, or every g is 1),
    "gone" (some g is not 1 and every one is applied as 1: the row is the base row), "exact" (every g is 0 or a
    power of two), "rounds" (the factor rounds to the dtype and each product rounds once, where the folded weight
    rounds each entry of the column: the two differ by rounding). `output_add`: the row carries vectors or a low-rank
    term. `folded_form`: false with a low-rank term."""
    dt = _dtype(dtype)
    sl = [t for t in edit.terms if t.kind in ("cols", "rows")]
    asked = sorted({float(g) for t in sl for g in np.unique(_asked(t, dose))})
    pairs = [[g, _applied(g, dt)] for g in asked]
    if not sl or all(g == 1.0 for g in asked):
        side = "none"
    elif all(a == 1.0 for _g, a in pairs):
        side = "gone"
    elif all(_exact(g) for g in asked):
        side = "exact"
    else:
        side = "rounds"
    return {"input_side": side, "output_add": edit.has_vectors() or edit.has_lowrank(),
            "folded_form": not edit.has_lowrank(), "factors": pairs[:8]}


def dose_refusal(edit: Edit, dose: float, dtype: Any) -> Optional[str]:
    """None when the dose can be read, else the sentence: an edit that is the base at every dose (every factor asked
    is 1 and nothing is added after the product), a dose the rounding eats whole (every factor exactly 1 in the dtype
    and nothing added), or one that carries the edit past the size cap. None at dose 0, which is the base by
    construction."""
    d = float(dose)
    if d == 0:
        return None
    ex = exact_at(edit, d, dtype)
    if ex["input_side"] == "none" and not ex["output_add"]:  # a scale by 1, or by a number that is 1 in float32
        return _SAY_DOSE_NOTHING
    if ex["input_side"] == "gone" and not ex["output_add"]:
        return _SAY_DOSE_GONE.format(d=d, dtype=_dtype_name(dtype))
    worst, name = 0.0, ""
    for entry in edit.report.get("terms", []):
        rel = entry.get("rel_change", {})
        x = rel.get("bound") if entry.get("kind") == "lowrank" else rel.get("max")
        x = float("inf") if x is None else float(x)
        if x > worst:
            worst, name = x, str(entry.get("linear"))
    if abs(d) * worst > MAX_REL_CHANGE:
        return _SAY_DOSE_SIZE.format(d=d, name=name, x=abs(d) * worst, cap=MAX_REL_CHANGE)
    return None


def null_factor(e: float, dtype: Any) -> float:
    """The factor of a null row as applied: 1 + e in the model's dtype."""
    return float(torch.tensor(1.0 + float(e), dtype=_dtype(dtype)))


def can_fold(lm: nn.Module, edit: Edit) -> Optional[str]:
    """None when `folded` would run this edit, else the sentence it would refuse with: a low-rank term (no folded
    form), or a touched linear whose forward reads no overlay (a plain linear not given
    `gguf_weights.install_overlay_forward`: the float32 stand-in's)."""
    if edit.has_lowrank():
        return _SAY_NO_FOLD
    for t in edit.terms:
        mod = _linear_of(lm, t.linear)
        if mod is None:
            return f"The edit {edit.short()} names {t.linear}, which this model does not have."
        packed = getattr(mod, "weight_packed", None) is not None and getattr(mod, "weight_scale", None) is not None
        attr = _OVERLAYS[1] if t.kind == "rows" else _OVERLAYS[0]
        reads = isinstance(getattr(mod, attr, None), dict) and (packed or "forward" in vars(mod))
        if not reads:
            return _SAY_PLAIN.format(name=t.linear)
    return None


def control_edits(edit: Edit, lm: nn.Module, seed: int, count: int) -> List[Edit]:
    """`count` control edits of one family: other sites of the same layer and size. Control c draws, for term t (its
    place in the edit's terms) from numpy's default_rng([seed, c, t]): for a column or row term as many other neurons
    of the same linear, ascending, the j-th taking the op, factor and vector of the edit's j-th; for a low-rank term
    the same term with its neuron axis permuted (A's columns on a down_proj). Each is an edit on the same base with
    its own id. The size caps are not applied (a set carried to a short column can pass the cap where the edit did
    not), and its report says so. They are NOT matched on how hard the sites fire or on what the edit delivers.
    `Refused` when a layer has too few other neurons."""
    seed, count = int(seed), int(count)
    if seed < 0 or count < 0:
        raise Refused(f"A control edit's seed and count are whole numbers from 0 up; {seed} and {count} were given.")
    mods: Dict[str, nn.Module] = {}
    for t in edit.terms:
        mod = _linear_of(lm, t.linear)
        if mod is None:
            raise Refused(f"The edit {edit.short()} names {t.linear}, which this model does not have.")
        mods[t.linear] = mod
        if t.kind in ("cols", "rows"):
            N = linear_shape(mod)[0] if t.kind == "rows" else linear_shape(mod)[1]
            K = int(t.idx.shape[0])
            if N - K < K:
                raise Refused(_SAY_CONTROL.format(L=t.layer, L1=t.layer + 1, m=N - K, K=K))
    out: List[Edit] = []
    for c in range(count):
        terms: List[Term] = []
        for i, t in enumerate(edit.terms):
            rng = np.random.default_rng([seed, c, i])
            if t.kind in ("cols", "rows"):
                N = linear_shape(mods[t.linear])[0] if t.kind == "rows" else linear_shape(mods[t.linear])[1]
                pool = np.setdiff1d(np.arange(N), t.idx)
                chosen = np.sort(rng.choice(pool, size=int(t.idx.shape[0]), replace=False))
                terms.append(Term(t.linear, t.kind, t.layer, t.which, idx=_ro(chosen, np.int64), op=t.op,
                                  factor=t.factor, vectors=t.vectors))
            else:
                # the neuron axis: A's columns on a down_proj (B's rows on the other two, which the gate does not take)
                down = t.which == "down_proj"
                N = linear_shape(mods[t.linear])[1]
                perm = rng.permutation(N if down else linear_shape(mods[t.linear])[0])
                A, B = (_ro(t.A[:, perm], np.float32), t.B) if down else (t.A, _ro(t.B[perm, :], np.float32))
                terms.append(Term(t.linear, "lowrank", t.layer, t.which, A=A, B=B))
        base = {"model": edit.base.get("model"), "weights_id": edit.base.get("weights_id"),
                "tensors": dict(edit.base.get("tensors", {}))}
        report = {"terms": [_entry(mods[t.linear], t) for t in terms], "caps": CONTROL_CAPS}
        note = f"control {c} of {edit.short()}, seed {seed}"
        out.append(Edit(_edit_id(base, terms), tuple(terms), base, note, report))
    return out


def _term_delta_f64(mod: nn.Module, t: Term) -> np.ndarray:
    """The dose-1.0 weight delta of a `cols` or `rows` term as float64 [K, width]."""
    with torch.no_grad():
        Ws = (dense_rows(mod, _long(t.idx)) if t.kind == "rows" else dense_cols(mod, _long(t.idx)))
        Ws64 = Ws.float().cpu().numpy().astype(np.float64)
    dW = np.zeros_like(Ws64, dtype=np.float64)
    is_scale = t.op == 0
    if is_scale.any():
        dW[is_scale] = (t.factor[is_scale].astype(np.float64) - 1.0)[:, None] * Ws64[is_scale]
    if t.vectors is not None:
        V64 = t.vectors.astype(np.float64)
        is_set = t.op == 1
        is_add = t.op == 2
        if is_set.any():
            dW[is_set] = V64[is_set] - Ws64[is_set]
        if is_add.any():
            dW[is_add] = V64[is_add]
    return dW


def _random_unit_masked(rng: np.random.Generator, count: int, dim: int, keep: np.ndarray) -> np.ndarray:
    """`count` random unit vectors of length `dim` supported strictly on `keep` (float64 [count, dim])."""
    m = int(keep.sum())
    if m < 1:
        raise Refused("Cannot draw a random direction: every dimension is inside the quiet mask.")
    z = rng.standard_normal((count, m))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    # resample any zero draw (measure zero)
    while (norms <= 0.0).any():
        bad = (norms[:, 0] <= 0.0)
        z[bad] = rng.standard_normal((int(bad.sum()), m))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / norms
    out = np.zeros((count, dim), dtype=np.float64)
    out[:, keep] = z
    return out


def _random_orthonormal_masked(rng: np.random.Generator, r: int, dim: int, keep: np.ndarray) -> np.ndarray:
    """`r` Haar-random orthonormal vectors of length `dim` supported strictly on `keep` (float64 [r, dim]).
    Requires `1 <= r <= int(keep.sum())`. Uses QR decomposition with sign-normalized diagonal of R so the resulting
    basis is uniformly distributed on the Stiefel manifold V_r(R^m)."""
    m = int(keep.sum())
    if r < 1 or r > m:
        raise Refused(f"Cannot draw {r} orthonormal masked directions with {m} unmasked dimensions.")
    z = rng.standard_normal((m, r))
    q, rr = np.linalg.qr(z, mode="reduced")
    signs = np.where(np.diag(rr) >= 0.0, 1.0, -1.0)
    q = q * signs[None, :]  # [m, r], columns orthonormal
    out = np.zeros((r, dim), dtype=np.float64)
    out[:, keep] = q.T
    return out


_GRAM_SAME_SITE_MAX_K = 256  # exact Gram-preserving Stiefel rotation for multi-column down_proj same-site controls


def same_site_controls(
    edit: Edit,
    lm: nn.Module,
    seed: int,
    count: int,
    *,
    extra_masks: Optional[Mapping[int, Any]] = None,
) -> List[Edit]:
    """`count` same-site random-direction control edits: the exact same layers, linears and neurons as `edit`, with
    random directions projected off each layer's quiet mask (for `down_proj` terms) and matched in stream-side delta
    length per column (or per-row delta length on `gate_proj`/`up_proj`). When a multi-column `down_proj` edit is
    rank-1 in stream space (such as `remove_direction`), each control keeps that rank-1 coherence across its columns
    using one shared masked random direction; for multi-column edits up to `_GRAM_SAME_SITE_MAX_K` columns (within the
    unmasked subspace dimension), a Haar-random orthonormal basis `R` replaces the edit's right singular vectors so
    every column norm and every pairwise inner product (`dS_ctrl @ dS_ctrl^T == dS @ dS^T`) match in stream space,
    preserving delivered disturbance size across coherent multi-neuron edits. Every control's report carries
    `CONTROL_CAPS` and `"family": "same_site"`."""
    seed, count = int(seed), int(count)
    if seed < 0 or count < 0:
        raise Refused(f"A control edit's seed and count are whole numbers from 0 up; {seed} and {count} were given.")
    mods: Dict[str, nn.Module] = {}
    prep: List[Dict[str, Any]] = []
    for t in edit.terms:
        mod = _linear_of(lm, t.linear)
        if mod is None:
            raise Refused(f"The edit {edit.short()} names {t.linear}, which this model does not have.")
        mods[t.linear] = mod
        if t.kind == "cols":
            w_post = _w_post(lm, t.layer)
            extra = extra_masks.get(t.layer) if extra_masks is not None else None
            mask = quiet_mask(lm, t.layer, extra_mask=extra)
            keep = ~mask
            n_keep = int(keep.sum())
            if n_keep < 2:
                raise Refused(f"Layer {t.layer} (page layer {t.layer + 1}) leaves too few dimensions outside its "
                              "quiet mask for same-site random-direction controls.")
            dW = _term_delta_f64(mod, t)  # [K, D]
            dS = dW * w_post[None, :]
            norms = np.linalg.norm(dS, axis=1)  # [K]
            rank1 = False
            signs = np.ones(len(norms), dtype=np.float64)
            gram_us: Optional[np.ndarray] = None
            if len(norms) > 1 and float((norms ** 2).sum()) > 0.0:
                u_s, sv, vt = np.linalg.svd(dS, full_matrices=False)
                energy = float((sv ** 2).sum())
                if energy > 0.0 and float(sv[0] ** 2) / energy >= 1.0 - 1e-5:
                    rank1 = True
                    proj = dS @ vt[0]
                    signs = np.where(proj >= 0.0, 1.0, -1.0)
                elif len(norms) <= min(_GRAM_SAME_SITE_MAX_K, n_keep):
                    # Exact Gram-preserving factor U @ diag(S) ([K, r]) for Haar-random Stiefel rotation on `keep`
                    r_eff = int(sv.shape[0])
                    if r_eff <= n_keep:
                        gram_us = u_s * sv[None, :]
            prep.append({"kind": "cols", "w_post": w_post, "keep": keep, "norms": norms, "rank1": rank1,
                         "signs": signs, "gram_us": gram_us, "dim": int(w_post.shape[0])})
        elif t.kind == "rows":
            dW = _term_delta_f64(mod, t)
            norms = np.linalg.norm(dW, axis=1)
            prep.append({"kind": "rows", "norms": norms, "dim": int(dW.shape[1])})
        else:
            w_post = _w_post(lm, t.layer)
            extra = extra_masks.get(t.layer) if extra_masks is not None else None
            mask = quiet_mask(lm, t.layer, extra_mask=extra)
            keep = ~mask
            if int(keep.sum()) < 2:
                raise Refused(f"Layer {t.layer} (page layer {t.layer + 1}) leaves too few dimensions outside its "
                              "quiet mask for same-site random-direction controls.")
            B_s = t.B.astype(np.float64) * w_post[:, None]  # [D, r]
            col_norms = np.linalg.norm(B_s, axis=0)  # [r]
            prep.append({"kind": "lowrank", "w_post": w_post, "keep": keep, "col_norms": col_norms,
                         "dim": int(w_post.shape[0])})
    out: List[Edit] = []
    for c in range(count):
        terms: List[Term] = []
        for i, (t, info) in enumerate(zip(edit.terms, prep)):
            rng = np.random.default_rng([seed, c, i, 17])  # distinct stream from other_sites
            if t.kind == "cols":
                K = int(t.idx.shape[0])
                keep, w_post, norms = info["keep"], info["w_post"], info["norms"]
                if info["rank1"]:
                    r_dir = _random_unit_masked(rng, 1, info["dim"], keep)[0]
                    dS_ctrl = (info["signs"] * norms)[:, None] * r_dir[None, :]
                elif info["gram_us"] is not None:
                    r_basis = _random_orthonormal_masked(rng, int(info["gram_us"].shape[1]), info["dim"], keep)
                    dS_ctrl = info["gram_us"] @ r_basis  # [K, D], zero on ~keep, exact Gram matrix match
                else:
                    dirs = _random_unit_masked(rng, K, info["dim"], keep)
                    dS_ctrl = norms[:, None] * dirs
                dW_ctrl = np.zeros_like(dS_ctrl, dtype=np.float64)
                dW_ctrl[:, keep] = dS_ctrl[:, keep] / w_post[None, keep]
                vec = dW_ctrl.astype(np.float32) + np.float32(0.0)
                terms.append(Term(t.linear, "cols", t.layer, t.which, idx=t.idx,
                                  op=_ro(np.full(K, 2, dtype=np.uint8), np.uint8),
                                  factor=_ro(np.ones(K, dtype=np.float32), np.float32),
                                  vectors=_ro(vec, np.float32)))
            elif t.kind == "rows":
                K = int(t.idx.shape[0])
                dirs = _random_unit_masked(rng, K, info["dim"], np.ones(info["dim"], dtype=bool))
                vec = (info["norms"][:, None] * dirs).astype(np.float32) + np.float32(0.0)
                terms.append(Term(t.linear, "rows", t.layer, t.which, idx=t.idx,
                                  op=_ro(np.full(K, 2, dtype=np.uint8), np.uint8),
                                  factor=_ro(np.ones(K, dtype=np.float32), np.float32),
                                  vectors=_ro(vec, np.float32)))
            else:
                r = int(t.A.shape[0])
                keep, w_post, col_norms = info["keep"], info["w_post"], info["col_norms"]
                dirs = _random_unit_masked(rng, r, info["dim"], keep)  # [r, D]
                B_s_ctrl = (col_norms[:, None] * dirs).T  # [D, r]
                B_ctrl = np.zeros_like(B_s_ctrl, dtype=np.float64)
                B_ctrl[keep, :] = B_s_ctrl[keep, :] / w_post[keep, None]
                B32 = B_ctrl.astype(np.float32) + np.float32(0.0)
                terms.append(Term(t.linear, "lowrank", t.layer, t.which, A=t.A, B=_ro(B32, np.float32)))
        base = {"model": edit.base.get("model"), "weights_id": edit.base.get("weights_id"),
                "tensors": dict(edit.base.get("tensors", {}))}
        report = {"terms": [_entry(mods[t.linear], t) for t in terms], "caps": CONTROL_CAPS, "family": "same_site"}
        note = f"same-site control {c} of {edit.short()}, seed {seed}"
        out.append(Edit(_edit_id(base, terms), tuple(terms), base, note, report))
    return out


def down_stream_norms_under(lm: nn.Module, layer: int, plant: Edit) -> np.ndarray:
    """`down_stream_norms` as the model has them with `plant` in its weights at dose 1: a column that a column term
    of the plant on this layer's down projection scales, sets or adds to is measured as the plant leaves it, every
    other column as the base has it (float64 [N], read-only). A low-rank term of the plant moves every column a
    little and is not counted. For the site scores of a reading taken under a plant: the per-row form of a set
    column multiplies the neuron's activation by nought and adds the new column after, so a score taken from the
    activation as the projection receives it would call the neuron silent when it writes the new column."""
    base = down_stream_norms(lm, layer)
    name = linear_name(int(layer), "down_proj")
    terms = [t for t in plant.terms if t.kind == "cols" and t.linear == name]
    mod = _linear_of(lm, name)
    if not terms or mod is None:
        return base
    out = np.array(base, dtype=np.float64)
    w_post = _w_post(lm, layer).astype(np.float64)[:, None]
    with torch.no_grad():
        for t in terms:
            idx = np.asarray(t.idx, dtype=np.int64)
            cols = dense_cols(mod, _long(idx)).double().cpu().numpy()  # [K, D]: the plant's own columns, no others
            new = cols * _asked(t, 1.0)[:, None]
            if t.vectors is not None:
                carried = np.asarray(t.op) != 0
                new[carried] += np.asarray(t.vectors, dtype=np.float64)[carried]
            out[idx] = np.sqrt(((new * w_post.T) ** 2).sum(axis=1))
    out.setflags(write=False)
    return out


def _draw_spread_sets(
    pools: Sequence[np.ndarray],
    groups: Sequence[int],
    count: int,
    rng: np.random.Generator,
    offsets: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], List[bool]]:
    """Up to `count` assignments of one site to each position, drawn to share as few sites as the pools allow.

    `pools[p]`: the sites eligible at position p (whole numbers from 0 up, each once), to each of which
    `offsets[p]` is added to give the site's number; `groups[p]`: the term the position belongs to. Within an
    assignment no site stands twice in one term (a term's indices are distinct); a site may stand at positions of
    two terms of one assignment, as an edit's own neuron does when it is changed in two projections. What a site
    is, is the caller's to say by its offsets: `matched_control_edits` numbers a neuron of a layer the same whatever
    projection it is changed in and makes ONE draw over all the terms, so that two control edits do not share a
    neuron through different projections.

    While a position has a site that no kept assignment holds it takes one of those, at random, and of those first
    the ones no position with fewer eligible sites could take: a neuron with thousands of eligible sites does not
    use up the few another neuron has. An assignment made wholly of such sites shares nothing with any kept before
    it, so those are pairwise disjoint by how they were made, and `clean` says which they are. A position with no
    such site left takes its sites in turn, those it has not stood in itself first and then each once before any
    twice, so a neuron's stand-ins repeat as little as its eligible sites allow; the assignment then shares a site
    with an earlier one. After three attempts in a row that gave nothing new, every such position draws from the
    whole of its pool, so the draw does not lock while assignments remain. No two assignments are the same, and
    their order is shuffled at the end (with `clean` alongside), so that no part of the list is the part drawn
    first.

    Returns (assignments, clean): each assignment an int64 array [P] of site numbers (offsets added) in the
    positions' order. A site found taken is set aside from its position for good, so a draw that runs out of sites
    does not look through every pool at every pick."""
    P, count = len(pools), int(count)
    if P == 0 or count <= 0 or any(len(p) == 0 for p in pools):
        return [], []
    off = [0] * P if offsets is None else [int(x) for x in offsets]
    sizes = [int(len(p)) for p in pools]
    lo = min(int(np.min(p)) + off[p_i] for p_i, p in enumerate(pools))
    span = max(int(np.max(p)) + off[p_i] for p_i, p in enumerate(pools)) - lo + 1
    present = np.zeros(span, dtype=bool)
    for p in range(P):
        present[np.asarray(pools[p], dtype=np.int64).reshape(-1) + (off[p] - lo)] = True
    sites = np.nonzero(present)[0].astype(np.int64) + lo  # every site a position could take, in order
    rank = np.cumsum(present, dtype=np.int64) - 1  # a site's place among them
    del present

    def ids_of(p: int) -> np.ndarray:
        return rank[np.asarray(pools[p], dtype=np.int64).reshape(-1) + (off[p] - lo)]

    narrowest = np.full(int(sites.size), np.iinfo(np.int64).max, dtype=np.int64)
    for p in range(P):  # the fewest eligible sites of any position that could take the site
        mine = ids_of(p)
        narrowest[mine] = np.minimum(narrowest[mine], sizes[p])
    order: List[np.ndarray] = []  # a position's pool: the sites no narrower position could take, then the others
    live: List[List[List[int]]] = []  # its classes of sites, each [first, end of those not yet found taken]
    for p in range(P):
        o = ids_of(p)[rng.permutation(sizes[p])]
        o = o[np.argsort(-narrowest[o], kind="stable")]  # contested by an ever narrower position: ever later
        cuts = [0] + [int(i) + 1 for i in np.nonzero(np.diff(narrowest[o]))[0]] + [sizes[p]]
        order.append(o.astype(np.int32))
        live.append([[a, b] for a, b in zip(cuts[:-1], cuts[1:])])
    del rank
    taken = np.zeros(int(sites.size), dtype=bool)  # the sites of every assignment kept
    first = [0] * P  # the first class of a position that may still hold a site no assignment has
    turn = [0] * P  # where a position stands in its turn through its pool, once it has none
    in_turn = [False] * P
    mine: List[set] = [set() for _ in range(P)]  # the sites a position has stood in, in the assignments kept

    def a_free_site(p: int, used: set) -> int:
        o = order[p]
        ci = first[p]
        while ci < len(live[p]):
            a, hi = live[p][ci]
            found_taken = in_term = 0
            pick = -1
            while hi > a and in_term < 8:
                r = a + int(rng.integers(hi - a))
                s = int(o[r])
                if taken[s]:
                    hi -= 1
                    o[r], o[hi] = o[hi], o[r]  # set aside for good: a kept assignment's site is never free again
                    found_taken += 1
                    if found_taken >= 8:  # most of what is left is taken: set all of it aside in one pass
                        part = o[a:hi]
                        keep = ~taken[part]
                        o[a:hi] = np.concatenate([part[keep], part[~keep]])
                        hi = a + int(keep.sum())
                        found_taken = 0
                    continue
                if s not in used:
                    pick = s
                    break
                in_term += 1
            if pick < 0 and hi > a:  # eight in a row that this assignment's term holds: look at the class whole
                part = o[a:hi]
                keep = ~taken[part]
                o[a:hi] = np.concatenate([part[keep], part[~keep]])
                hi = a + int(keep.sum())
                cand = o[a:hi]
                if used:
                    cand = cand[~np.isin(cand, np.fromiter(used, dtype=np.int64, count=len(used)))]
                if cand.size:
                    pick = int(cand[int(rng.integers(cand.size))])
            live[p][ci][1] = hi
            if hi <= a and ci == first[p]:
                first[p] = ci + 1
            if pick >= 0:
                return pick
            ci += 1
        return -1

    def a_site_in_turn(p: int, used: set, anywhere: bool) -> int:
        o, E = order[p], sizes[p]
        if anywhere:
            for _ in range(16):
                s = int(o[int(rng.integers(E))])
                if s not in used:
                    return s
            cand = o[~np.isin(o, np.fromiter(used, dtype=np.int64, count=len(used)))] if used else o
            return int(cand[int(rng.integers(cand.size))]) if cand.size else -1
        if not in_turn[p] and first[p] >= len(live[p]):  # its first turn: the sites it has not stood in come first
            in_turn[p] = True
            if mine[p]:
                stood = np.isin(o, np.fromiter(mine[p], dtype=np.int64, count=len(mine[p])))
                o[:] = np.concatenate([o[~stood], o[stood]])
            turn[p] = 0
        for step in range(E):
            s = int(o[(turn[p] + step) % E])
            if s not in used:
                turn[p] = (turn[p] + step + 1) % E
                return s
        return -1

    pos_order = sorted(range(P), key=lambda p: (sizes[p], p))
    out: List[np.ndarray] = []
    clean: List[bool] = []
    seen_sets: set = set()
    missed = 0  # attempts in a row that gave no new assignment
    for _attempt in range(max(count * 32, 256)):
        if len(out) >= count:
            break
        chosen = np.empty(P, dtype=np.int64)
        used: Dict[int, set] = {}
        all_free, ok = True, True
        for p in pos_order:
            u = used.setdefault(int(groups[p]), set())
            s = a_free_site(p, u)
            if s < 0:
                all_free = False
                s = a_site_in_turn(p, u, missed >= 3)
                if s < 0:
                    ok = False
                    break
            chosen[p] = s
            u.add(s)
        key = tuple((g, tuple(sorted(used[g]))) for g in sorted(used)) if ok else None
        if not ok or key in seen_sets:
            missed += 1
            continue
        missed = 0
        seen_sets.add(key)
        taken[chosen] = True
        for p in range(P):
            mine[p].add(int(chosen[p]))
        out.append(sites[chosen])
        clean.append(all_free)
    shuffled = [int(i) for i in rng.permutation(len(out))]
    return [out[i] for i in shuffled], [clean[i] for i in shuffled]


def _draw_spread_indices(
    elig: List[np.ndarray],
    count: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """`_draw_matched_indices` for two or more positions of ONE term, with the assignments sharing as few sites as
    the pools allow (`_draw_spread_sets`, which `matched_control_edits` calls itself over all of an edit's terms
    together)."""
    return _draw_spread_sets(elig, [0] * len(elig), count, rng)[0]


def _draw_matched_indices(
    elig: List[np.ndarray],
    count: int,
    rng: np.random.Generator,
    spread: bool = False,
) -> List[np.ndarray]:
    """Draw up to `count` distinct matched assignments `[p_0, ..., p_{K-1}]` where `p_j in elig[j]` and all `p_j` in
    one assignment are distinct. Two assignments are distinct when their sorted sets differ. Each assignment is drawn
    on its own, so with two or more positions the assignments share sites (one position never does: its sites are
    drawn without repeats); `spread` draws them so that they share as few as the pools allow
    (`_draw_spread_indices`)."""
    K = len(elig)
    if K == 0 or count == 0:
        return []
    if any(len(e) == 0 for e in elig):
        return []
    if K == 1:
        pool = np.asarray(elig[0], dtype=np.int64)
        if pool.size < count:
            return [np.array([int(x)], dtype=np.int64) for x in rng.permutation(pool)]
        picked = rng.choice(pool, size=count, replace=False)
        return [np.array([int(x)], dtype=np.int64) for x in picked]
    if spread:
        return _draw_spread_indices(elig, count, rng)
    # Order positions from narrowest eligible pool to widest to minimize collisions during assignment
    pos_order = sorted(range(K), key=lambda j: (len(elig[j]), j))
    out: List[np.ndarray] = []
    seen_sets: set = set()
    budget = max(count * 32, 256)
    for _attempt in range(budget):
        if len(out) >= count:
            break
        chosen = np.empty(K, dtype=np.int64)
        used: set = set()
        ok = True
        for j in pos_order:
            cands = elig[j]
            if len(used) >= len(cands):
                # fast check: filter remaining
                avail = [int(x) for x in cands if int(x) not in used]
                if not avail:
                    ok = False
                    break
                pick = avail[int(rng.integers(len(avail)))]
            else:
                pick = -1
                for _ in range(16):
                    cand = int(cands[int(rng.integers(len(cands)))])
                    if cand not in used:
                        pick = cand
                        break
                if pick < 0:
                    avail = [int(x) for x in cands if int(x) not in used]
                    if not avail:
                        ok = False
                        break
                    pick = avail[int(rng.integers(len(avail)))]
            chosen[j] = pick
            used.add(pick)
        if not ok:
            continue
        key = tuple(sorted(int(x) for x in chosen))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        out.append(chosen)
    return out


def matched_control_edits(
    edit: Edit,
    lm: nn.Module,
    scores_by_linear: Mapping[str, np.ndarray],
    seed: int,
    count: int,
    *,
    tolerance: float = 2.0,
    spread: bool = False,
) -> Tuple[Optional[List[Edit]], Dict[str, Any]]:
    """`count` control edits on other sites matched within factor `tolerance` on per-neuron expected delivered score
    `scores_by_linear[t.linear]`. Returns `(edits, coverage)` when all `count` distinct matched control edits can be
    formed, or `(None, coverage)` when coverage fails (caller voids the reading and records `coverage`).

    `spread`: draw the control sets so that they share as few sites as the pools allow: one draw over all the
    column and row terms together (`_draw_spread_sets`; an edit of one neuron is drawn as it always was). Without it
    each term's sets are drawn on their own, and sets that differ in one stand-in are "distinct". The coverage
    says, a term, the fewest sites that were eligible for one neuron (`min_eligible`) and the fewest different
    stand-ins one neuron had over all the sets (`min_distinct`); and, for the control edits as wholes, how many
    share no site with one another (`disjoint_sets`). A site is a neuron of a layer, whatever projection the control
    edit changes it in: two control edits that change one neuron, one in its down projection's column and one in
    its up projection's row, share it. The count is of a family that was found, not of the largest there is: the
    larger of (a) the control edits taken one after another as they stand, each counted if it shares no site with
    those counted before it, and (b) under `spread`, those the draw made wholly of sites that no earlier set had,
    which are pairwise disjoint by how they were made and are checked to be before they are counted. Only control
    edits that share no site are separate draws: a site that stands in for one neuron in one set and for another
    neuron in another ties the two sets together, which a count of each neuron's own stand-ins does not see."""
    seed, count = int(seed), int(count)
    tol = _number(tolerance, "tolerance")
    if seed < 0 or count < 0:
        raise Refused(f"A control edit's seed and count are whole numbers from 0 up; {seed} and {count} were given.")
    if not (math.isfinite(tol) and tol >= 1.0):
        raise Refused(f"Matched control tolerance must be a finite number >= 1.0; {tolerance!r} was given.")
    if not any(t.kind in ("cols", "rows") for t in edit.terms):
        raise Refused('Control family "other_sites_matched" requires at least one column or row term; use '
                      '"same_site" or "other_sites" for a low-rank-only edit.')
    mods: Dict[str, nn.Module] = {}
    term_draws: List[Optional[List[np.ndarray]]] = []
    term_cov: List[Dict[str, Any]] = []
    eligible: Dict[int, List[np.ndarray]] = {}
    n_sites: Dict[int, int] = {}
    for i, t in enumerate(edit.terms):
        mod = _linear_of(lm, t.linear)
        if mod is None:
            raise Refused(f"The edit {edit.short()} names {t.linear}, which this model does not have.")
        mods[t.linear] = mod
        if t.kind not in ("cols", "rows"):
            term_draws.append(None)
            continue
        N = linear_shape(mod)[0] if t.kind == "rows" else linear_shape(mod)[1]
        K = int(t.idx.shape[0])
        if t.linear not in scores_by_linear:
            raise Refused(f"Matched controls need base pre-pass scores for {t.linear}.")
        scores = np.asarray(scores_by_linear[t.linear], dtype=np.float64).reshape(-1)
        if scores.shape[0] != N or not np.isfinite(scores).all():
            raise Refused(f"Matched control scores for {t.linear} must be finite of length {N}.")
        pool = np.setdiff1d(np.arange(N, dtype=np.int64), t.idx)
        pool_scores = scores[pool]
        edit_scores = scores[t.idx]
        elig: List[np.ndarray] = []
        for s_val in edit_scores:
            if s_val <= 0.0:
                ok_mask = pool_scores <= 1e-12
            else:
                ok_mask = (pool_scores >= s_val / tol) & (pool_scores <= s_val * tol)
            elig.append(pool[ok_mask])
        eligible[i] = elig
        n_sites[i] = int(N)
        union_count = int(len(set().union(*(e.tolist() for e in elig)))) if elig else 0
        term_cov.append({
            "linear": t.linear, "count": K,
            "min_eligible": int(min(len(e) for e in elig)) if elig else 0,
            "min_distinct": 0,
            "median_eligible": _clean(np.median([len(e) for e in elig])) if elig else 0.0,
            "union_eligible": union_count,
            "drawn": 0,
            "edit_scores": {"min": _clean(edit_scores.min()), "median": _clean(np.median(edit_scores)),
                            "max": _clean(edit_scores.max())},
        })
        term_draws.append([])
    # the draws. Spread: one over all the terms, a site being a neuron of a layer whatever projection changes it
    made_clean: Optional[List[bool]] = None
    if count > 0 and spread and sum(len(e) for e in eligible.values()) >= 2:
        layers = sorted({int(edit.terms[i].layer) for i in eligible})
        stride = max(n_sites.values())  # a layer's neurons are numbered after those of the layers before it
        base = {i: layers.index(int(edit.terms[i].layer)) * stride for i in eligible}
        sets, made_clean = _draw_spread_sets(
            [e for elig in eligible.values() for e in elig], [i for i, elig in eligible.items() for _e in elig],
            count, np.random.default_rng([seed, 31]), offsets=[base[i] for i, elig in eligible.items() for _e in elig])
        at = 0
        for i, elig in eligible.items():
            term_draws[i] = [s[at:at + len(elig)] - base[i] for s in sets]
            at += len(elig)
    elif count > 0:
        for i, elig in eligible.items():
            term_draws[i] = _draw_matched_indices(elig, count, np.random.default_rng([seed, i, 29]), spread=spread)
    sufficient = True
    for cov_t, i in zip(term_cov, eligible):
        draws = term_draws[i]
        cov_t["min_distinct"] = (int(min(len({int(d[j]) for d in draws}) for j in range(len(eligible[i]))))
                                 if draws else 0)
        cov_t["drawn"] = len(draws)
        if len(draws) < count:
            sufficient = False
    found = min((len(d) for d in term_draws if d is not None), default=count) if count > 0 else 0
    coverage: Dict[str, Any] = {
        "tolerance": _clean(tol), "requested": count, "found": int(found),
        "sufficient": bool(sufficient and found == count), "terms": term_cov,
        "draw": "spread" if spread else "independent",
    }

    def sites_of(c: int) -> set:
        """The neurons control edit `c` changes, each a (layer, index) whatever projection it is changed in."""
        return {(int(edit.terms[i].layer), int(x)) for i, d in enumerate(term_draws) if d is not None for x in d[c]}

    seen_sites: set = set()
    disjoint = 0
    for c in range(int(found)):
        sites = sites_of(c)
        if not (sites & seen_sites):
            disjoint += 1
            seen_sites |= sites
    if made_clean is not None:
        fam = [sites_of(c) for c in range(int(found)) if made_clean[c]]
        if len(set().union(*fam)) == sum(len(s) for s in fam):  # pairwise disjoint, checked and not taken on trust
            disjoint = max(disjoint, len(fam))
    coverage["disjoint_sets"] = disjoint
    if not coverage["sufficient"]:
        return None, coverage
    out: List[Edit] = []
    for c in range(count):
        terms: List[Term] = []
        for i, t in enumerate(edit.terms):
            if t.kind in ("cols", "rows"):
                assigned = term_draws[i][c]
                order = np.argsort(assigned, kind="stable")
                chosen = assigned[order]
                op = t.op[order]
                factor = t.factor[order]
                vectors = None if t.vectors is None else _ro(t.vectors[order], np.float32)
                terms.append(Term(t.linear, t.kind, t.layer, t.which, idx=_ro(chosen, np.int64),
                                  op=_ro(op, np.uint8), factor=_ro(factor, np.float32), vectors=vectors))
            else:
                rng = np.random.default_rng([seed, c, i, 29])
                down = t.which == "down_proj"
                N = linear_shape(mods[t.linear])[1]
                perm = rng.permutation(N if down else linear_shape(mods[t.linear])[0])
                A, B = (_ro(t.A[:, perm], np.float32), t.B) if down else (t.A, _ro(t.B[perm, :], np.float32))
                terms.append(Term(t.linear, "lowrank", t.layer, t.which, A=A, B=B))
        base = {"model": edit.base.get("model"), "weights_id": edit.base.get("weights_id"),
                "tensors": dict(edit.base.get("tensors", {}))}
        report = {"terms": [_entry(mods[t.linear], t) for t in terms], "caps": CONTROL_CAPS,
                  "family": "other_sites_matched", "tolerance": _clean(tol)}
        note = f"matched control {c} of {edit.short()}, seed {seed}, tol {tol:g}"
        out.append(Edit(_edit_id(base, terms), tuple(terms), base, note, report))
    return out, coverage


# ------------------------------------------------------------------------------------ composition and diagnostics


def stack(
    lm: nn.Module,
    edits: Sequence[Edit],
    *,
    model_id: str,
    weights_id: Optional[str],
    note: str = "",
    doses: Optional[Sequence[float]] = None,
    caps: bool = True,
    plant_first: bool = False,
) -> Edit:
    """Compose a sequence of edits on the same base by adding their weight deltas (`W + sum_k d_k * Delta_k`). Two
    scalings on the same column or row add as `f1 + f2 - 1` (the additive delta rule); two `add` vectors sum;
    `scale` + `add` lowers to an `add`; `set` + `add` or `set` + `scale` stays a `set`; two `set`s on the same index
    are refused. Low-rank terms on the same linear concatenate along rank. The composed edit passes through the gate
    and carries its parents' ids in `report["parents"]`.

    `plant_first`: the first edit is a plant lying under the rest (a blinded session's, `levelling._Sitting`), and the
    whole must be refused nothing that the rest alone is not, or a refusal would say where the plant is. So the whole
    is not asked the caps on one edit's size again (each part met them alone), and where both set one column or row the
    later edit's set stands. The deltas still add: an edit that scales a neuron the plant scaled gives `f1 + f2 - 1`,
    not `f1 * f2`, so on the plant's own neurons the whole is not yet the edit on a model with those weights."""
    edits = list(edits)
    if not edits:
        raise Refused("stack requires at least one edit.")
    if doses is None:
        dose_list = [1.0] * len(edits)
    else:
        dose_list = [_number(d, "dose") for d in doses]
        if len(dose_list) != len(edits):
            raise Refused(f"stack received {len(dose_list)} doses for {len(edits)} edits.")
        for d in dose_list:
            if not math.isfinite(d) or abs(d) > MAX_DOSE:
                raise Refused(f"stack dose must be finite and within +/-{MAX_DOSE:g}; got {d!r}.")
    checked: List[Edit] = []
    for e in edits:
        if not isinstance(e, Edit):
            raise Refused(f"stack takes weight_edits.Edit values; {type(e).__name__} is not one.")
        if not isinstance(e.base, dict) or e.base.get("model") != model_id or e.base.get("weights_id") != weights_id:
            a = e.base.get("model") if isinstance(e.base, dict) else None
            ia = e.base.get("weights_id") if isinstance(e.base, dict) else None
            raise Refused(_SAY_OTHER_BASE.format(a=a, ia=ia, b=model_id, ib=weights_id))
        checked.append(gate_again(lm, e, model_id=model_id, weights_id=weights_id, sized=not plant_first))
    # per (linear, kind) -> idx -> (op_code, factor, vector_or_None)
    slices: Dict[Tuple[str, str, int, str], Dict[int, Tuple[int, float, Optional[np.ndarray]]]] = {}
    low_specs: List[TermSpec] = []
    base_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

    def _base_slice(name: str, kind: str, idx: int) -> np.ndarray:
        key = (name, kind, idx)
        if key not in base_cache:
            mod = _linear_of(lm, name)
            with torch.no_grad():
                Ws = dense_rows(mod, _long([idx])) if kind == "rows" else dense_cols(mod, _long([idx]))
                base_cache[key] = Ws[0].float().cpu().numpy().astype(np.float64)
        return base_cache[key]

    for e, dose_k in zip(checked, dose_list):
        if dose_k == 0.0:
            continue
        for t in e.terms:
            if t.kind == "lowrank":
                B_scaled = (t.B.astype(np.float64) * dose_k).astype(np.float32) if dose_k != 1.0 else t.B
                low_specs.append(TermSpec("lowrank", t.layer, t.which, A=t.A, B=B_scaled))
                continue
            bucket = slices.setdefault((t.linear, t.kind, int(t.layer), t.which), {})
            for j, n_raw in enumerate(t.idx):
                n = int(n_raw)
                raw_op = int(t.op[j])
                raw_f = float(t.factor[j])
                raw_v = None if t.vectors is None or raw_op == 0 else t.vectors[j].astype(np.float64)
                if raw_op == 0:
                    op2, f2, v2 = 0, 1.0 + dose_k * (raw_f - 1.0), None
                elif raw_op == 2:
                    op2, f2, v2 = 2, 1.0, dose_k * raw_v
                else:  # raw_op == 1 (set)
                    if dose_k == 1.0:
                        op2, f2, v2 = 1, 1.0, raw_v
                    else:
                        W = _base_slice(t.linear, t.kind, n)
                        op2, f2, v2 = 2, 1.0, dose_k * (raw_v - W)
                if n not in bucket:
                    bucket[n] = (op2, f2, v2)
                    continue
                op1, f1, v1 = bucket[n]
                if op1 == 1 and op2 == 1:
                    if plant_first:  # the later edit replaces what the plant set: no refusal names the plant's neuron
                        bucket[n] = (1, 1.0, v2)
                        continue
                    raise Refused(f"Cannot stack two edits that both set neuron {n} of {t.linear} "
                                  f"(page layer {t.layer + 1}).")
                if op1 == 0 and op2 == 0:
                    bucket[n] = (0, f1 + f2 - 1.0, None)
                elif op1 == 2 and op2 == 2:
                    bucket[n] = (2, 1.0, v1 + v2)
                elif op1 == 0 and op2 == 2:
                    W = _base_slice(t.linear, t.kind, n)
                    bucket[n] = (2, 1.0, (f1 - 1.0) * W + v2)
                elif op1 == 2 and op2 == 0:
                    W = _base_slice(t.linear, t.kind, n)
                    bucket[n] = (2, 1.0, v1 + (f2 - 1.0) * W)
                elif op1 == 1 and op2 == 2:
                    bucket[n] = (1, 1.0, v1 + v2)
                elif op1 == 2 and op2 == 1:
                    bucket[n] = (1, 1.0, v2 + v1)
                elif op1 == 1 and op2 == 0:
                    W = _base_slice(t.linear, t.kind, n)
                    bucket[n] = (1, 1.0, v1 + (f2 - 1.0) * W)
                elif op1 == 0 and op2 == 1:
                    W = _base_slice(t.linear, t.kind, n)
                    bucket[n] = (1, 1.0, v2 + (f1 - 1.0) * W)

    specs: List[TermSpec] = []
    for (name, kind, layer, which), bucket in sorted(slices.items()):
        by_op: Dict[int, List[Tuple[int, float, Optional[np.ndarray]]]] = {0: [], 1: [], 2: []}
        for n in sorted(bucket):
            op_code, f_val, v_val = bucket[n]
            by_op[op_code].append((n, f_val, v_val))
        if by_op[0]:
            ns = tuple(n for n, _f, _v in by_op[0])
            fs = [float(f) for _n, f, _v in by_op[0]]
            specs.append(TermSpec(kind, layer, which, ns, "scale", fs))
        for code, opname in ((1, "set"), (2, "add")):
            if by_op[code]:
                ns = tuple(n for n, _f, _v in by_op[code])
                vs = np.stack([v.astype(np.float32) for _n, _f, v in by_op[code]], axis=0)
                specs.append(TermSpec(kind, layer, which, ns, opname, vectors=vs))
    specs.extend(low_specs)
    out = _assemble(lm, specs, model_id=model_id, weights_id=weights_id, note=note, caps=caps, sized=not plant_first)
    out.report["parents"] = [e.id for e in checked]
    return out


def edit_diagnostics(lm: nn.Module, edit: Edit, *, dose: float = 1.0) -> Dict[str, Any]:
    """Structural diagnostics for `edit` at `dose` against `lm`: per-term raw and stream-side relative change,
    quiet-mask energy share, write ceilings (for `down_proj` columns), and the share of weight entries that actually
    change after rounding to the model's compute dtype."""
    d = _number(dose, "dose")
    if not math.isfinite(d):
        raise Refused("A dose is a finite number.")
    dt = lm.norm.weight.dtype
    entries: List[Dict[str, Any]] = []
    for t in edit.terms:
        mod = _linear_of(lm, t.linear)
        if mod is None:
            raise Refused(f"The edit {edit.short()} names {t.linear}, which this model does not have.")
        if t.kind in ("cols", "rows"):
            with torch.no_grad():
                W_orig = dense_rows(mod, _long(t.idx)) if t.kind == "rows" else dense_cols(mod, _long(t.idx))
                W_f32 = W_orig.float()
                g = _asked(t, d)
                W_new_f32 = torch.from_numpy(g.astype(np.float32)).to(W_f32.device)[:, None] * W_f32
                if t.vectors is not None:
                    at = torch.from_numpy(np.nonzero(t.op != 0)[0]).to(W_f32.device)
                    V = torch.from_numpy(t.vectors[t.op != 0].copy()).to(W_f32.device)
                    W_new_f32[at] = W_new_f32[at] + d * V
                W_new_dt = W_new_f32.to(dt)
                W_new_dt[torch.from_numpy(np.array((g == 0) & (t.op == 0))).to(W_f32.device)] = 0
                W_base_dt = W_orig.to(dt)
                changed_share = float((W_new_dt != W_base_dt).float().mean().item())
                dW = (W_new_f32 - W_f32).cpu().numpy().astype(np.float64)
                W64 = W_f32.cpu().numpy().astype(np.float64)
            base_norms = np.linalg.norm(W64, axis=1)
            d_norms = np.linalg.norm(dW, axis=1)
            raw_rel = np.where(base_norms > 0, d_norms / np.where(base_norms > 0, base_norms, 1.0), 0.0)
            row_info: Dict[str, Any] = {
                "linear": t.linear, "kind": t.kind, "layer": int(t.layer), "page_layer": int(t.layer) + 1,
                "count": int(t.idx.shape[0]),
                "raw_rel_change": {"max": _clean(raw_rel.max()), "median": _clean(np.median(raw_rel))},
                "dtype_changed_share": _clean(changed_share),
            }
            if t.kind == "cols":
                w_post = _w_post(lm, t.layer)
                mask = quiet_mask(lm, t.layer)
                S_base = W64 * w_post[None, :]
                dS = dW * w_post[None, :]
                sb_norms = np.linalg.norm(S_base, axis=1)
                ds_norms = np.linalg.norm(dS, axis=1)
                s_rel = np.where(sb_norms > 0, ds_norms / np.where(sb_norms > 0, sb_norms, 1.0), 0.0)
                raw_e = float((dW ** 2).sum())
                stream_e = float((dS ** 2).sum())
                raw_q = float((dW[:, mask] ** 2).sum()) / raw_e if raw_e > 0.0 else 0.0
                stream_q = float((dS[:, mask] ** 2).sum()) / stream_e if stream_e > 0.0 else 0.0
                W_new_64 = W_new_f32.cpu().numpy().astype(np.float64)
                wn_norms = np.linalg.norm(W_new_64, axis=1, keepdims=True)
                unit = np.where(wn_norms > 0, W_new_64 / np.where(wn_norms > 0, wn_norms, 1.0), 0.0)
                scalar_t = getattr(lm.layers[int(t.layer)], "layer_scalar", None)
                s_l = float(scalar_t.detach().float().item()) if scalar_t is not None else 1.0
                ceilings = s_l * math.sqrt(float(W64.shape[1])) * np.linalg.norm(unit * w_post[None, :], axis=1)
                row_info["stream_rel_change"] = {"max": _clean(s_rel.max()), "median": _clean(np.median(s_rel))}
                row_info["quiet_mask_energy_share"] = {"raw": _clean(raw_q), "stream": _clean(stream_q)}
                row_info["write_ceilings"] = {"min": _clean(ceilings.min()), "median": _clean(np.median(ceilings)),
                                              "max": _clean(ceilings.max())}
            entries.append(row_info)
        else:
            B64 = t.B.astype(np.float64)
            mask = quiet_mask(lm, t.layer)
            b_e = float((B64 ** 2).sum())
            b_q = float((B64[mask, :] ** 2).sum()) / b_e if b_e > 0.0 else 0.0
            entries.append({
                "linear": t.linear, "kind": "lowrank", "layer": int(t.layer), "page_layer": int(t.layer) + 1,
                "rank": int(t.A.shape[0]),
                "raw_rel_bound": _clean(abs(d) * _bound(mod, t.A, t.B)),
                "quiet_mask_energy_share": {"raw": _clean(b_q)},
            })
    return {"edit_id": edit.id, "dose": _clean(d), "dtype": _dtype_name(dt), "terms": entries}


# --------------------------------------------------------------------------------------------- the per-row forms


@dataclass(frozen=True, eq=False)
class Arm:
    """A row's program: this edit at this dose."""

    edit: Edit
    dose: float = 1.0


@dataclass(frozen=True)
class Null:
    """A row's program: the feed-forward output of `layer` scaled by 1 + e, which the norm undoes but for rounding."""

    layer: int
    e: float


class _Site:
    """One linear's hooks for one `per_row` block: which rows scale which activations before the product, which add
    what after it, which recompute or exactly scale edited output rows, and which are null rows. Tensors are made on
    the first forward, from the tensor the hook is handed (its dtype and device), and kept for the block."""

    def __init__(self, name: str, rows: int, width: int, mod: Optional[nn.Module] = None) -> None:
        self.name, self.rows, self.width, self.mod = name, rows, width, mod
        self.factors: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (row, columns, g asked float64)
        self.adds: Dict[int, Tuple[Term, float]] = {}  # row -> (the column term with vectors, dose)
        self.lows: Dict[int, Tuple[Term, float]] = {}  # row -> (the low-rank term, dose)
        self.row_scales: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # row -> (row idx, exact g float64)
        self.row_recomp: Dict[int, Tuple[Term, np.ndarray, float]] = {}  # row -> (rows term, sel positions, dose)
        self.nulls: List[Tuple[int, float]] = []  # (row, e)
        self.kept: Dict[int, Dict[str, torch.Tensor]] = {}
        self._made: Dict[Any, Any] = {}

    def _as_applied(self, gs: np.ndarray, dtype: torch.dtype) -> np.ndarray:
        """Each g as the dtype applies it (float64 holding a value of the dtype exactly), worked once a distinct g."""
        memo = self._made.setdefault(("applied", dtype), {})
        u, inv = np.unique(gs, return_inverse=True)
        for g in u.tolist():
            if g not in memo:
                memo[g] = _applied(g, dtype)
        return np.array([memo[g] for g in u.tolist()], dtype=np.float64)[inv.reshape(-1)]

    def scales(self, dtype: torch.dtype) -> bool:
        """Whether any input-side factor differs from 1 as the dtype applies it."""
        return any(bool((self._as_applied(gs, dtype) != 1.0).any()) for _r, _idx, gs in self.factors)

    def adds_after(self) -> bool:
        return bool(self.adds or self.lows or self.row_scales or self.row_recomp or self.nulls)

    def _check(self, t: torch.Tensor) -> None:
        if t.dim() != 3:
            raise RuntimeError(f"hooks made for {self.rows} rows of [rows, pieces, width] met a tensor of shape "
                               f"{list(t.shape)} on {self.name}")
        if int(t.shape[0]) != self.rows:
            raise RuntimeError(f"hooks made for {self.rows} rows met {int(t.shape[0])}")

    def _once(self, key: Any, make: Any) -> Any:
        if key not in self._made:
            self._made[key] = make()
        return self._made[key]

    def _factor_matrix(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Fm [R, N] in x's dtype: 1 everywhere but each row's own columns, where it is g as the dtype applies it;
        None when no entry differs from 1."""
        def make() -> Optional[torch.Tensor]:
            if not self.scales(x.dtype):
                return None
            Fm = torch.ones(self.rows, self.width, dtype=x.dtype, device=x.device)
            for r, idx, gs in self.factors:
                applied = torch.from_numpy(self._as_applied(gs, x.dtype)).to(device=x.device, dtype=x.dtype)  # exact
                Fm[r, torch.from_numpy(np.array(idx)).to(x.device)] = applied
            return Fm

        return self._once(("Fm", x.dtype, x.device), make)

    def _recomp_weights(self, t: Term, sel: np.ndarray, d: float, dtype: torch.dtype,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Cached `(idx_t, W_new, b_sub)` for a recomputed row edit: identical float32->dtype formula to `folded`."""
        def make() -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            idx_np = t.idx[sel]
            op_sel = t.op[sel]
            g_sel = _asked(t, d)[sel]
            with torch.no_grad():
                Wr = dense_rows(self.mod, _long(idx_np)).float()  # [K', in]
                g_t = torch.from_numpy(g_sel.astype(np.float32)).to(Wr.device)
                new = g_t[:, None] * Wr
                new[g_t == 0] = 0.0
                if t.vectors is not None:
                    at_local = np.nonzero(op_sel != 0)[0]
                    if at_local.size:
                        at_t = torch.from_numpy(at_local).to(Wr.device)
                        V = torch.from_numpy(t.vectors[sel[at_local]].copy()).to(Wr.device)
                        new[at_t] = new[at_t] + d * V
                W_new = new.to(device=device, dtype=dtype)
                W_new[torch.from_numpy(np.array((g_sel == 0) & (op_sel == 0))).to(device)] = 0
                idx_t = torch.from_numpy(idx_np.copy()).to(device)
                bias = getattr(self.mod, "bias", None)
                b_sub = None if bias is None else bias.detach().index_select(0, idx_t.to(bias.device)).to(
                    device=device, dtype=dtype)
            return idx_t, W_new, b_sub

        return self._once(("row_W", id(t), sel.tobytes(), d, dtype, device), make)

    def pre(self, _mod: nn.Module, args: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        x = args[0]
        self._check(x)
        self.kept = {}
        for r, (t, _d) in self.adds.items():  # from the ORIGINAL input: the forward hook sees this hook's output
            cols = self._once(("cols", id(t), x.device),
                              lambda t=t: torch.from_numpy(t.idx[t.op != 0].copy()).to(x.device))
            self.kept.setdefault(r, {})["a"] = x[r][:, cols]  # [T, K'], the model's dtype
        for r, (t, _d) in self.lows.items():
            At = self._once(("At", id(t), x.device), lambda t=t: torch.from_numpy(np.array(t.A.T)).to(x.device))
            self.kept.setdefault(r, {})["z"] = x[r].float() @ At  # [T, rank] float32
        for r in self.row_recomp:
            self.kept.setdefault(r, {})["rx"] = x[r]
        Fm = self._factor_matrix(x)
        if Fm is None:
            return None
        return (x * Fm[:, None, :],) + tuple(args[1:])

    def post(self, _mod: nn.Module, _args: Tuple[Any, ...], out: torch.Tensor) -> torch.Tensor:
        self._check(out)
        kept, self.kept = self.kept, {}
        out2 = out.clone()
        for r in sorted(set(self.adds) | set(self.lows)):
            acc = out[r].float()
            if r in self.adds:
                t, d = self.adds[r]
                # the content at this dose, rounded to the model's dtype as the folded form rounds the column it writes
                # (a control edit shares its edit's vectors and ops, so at one dose they share this tensor)
                C = self._once(("C", id(t.vectors), id(t.op), d, out.dtype, out.device), lambda t=t, d=d: (
                    (torch.from_numpy(t.vectors[t.op != 0].copy()).to(out.device) * d).to(out.dtype).float()))
                acc = acc + kept[r]["a"].float() @ C
            if r in self.lows:
                t, d = self.lows[r]
                Bt = self._once(("Bt", id(t.B), out.device),
                                lambda t=t: torch.from_numpy(np.array(t.B.T)).to(out.device))
                acc = acc + (kept[r]["z"] * d) @ Bt
            out2[r] = acc.to(out.dtype)  # one sum in float32, rounded once
        for r, (idx_np, gs) in self.row_scales.items():
            idx_z, idx_nz, g_nz = self._once(
                ("row_sc", r, out.dtype, out.device),
                lambda idx_np=idx_np, gs=gs: (
                    torch.from_numpy(idx_np[gs == 0].copy()).to(out.device) if (gs == 0).any() else None,
                    torch.from_numpy(idx_np[gs != 0].copy()).to(out.device) if (gs != 0).any() else None,
                    torch.from_numpy(self._as_applied(gs[gs != 0], out.dtype)).to(
                        device=out.device, dtype=out.dtype) if (gs != 0).any() else None,
                ),
            )
            if idx_z is not None:
                out2[r][:, idx_z] = 0
            if idx_nz is not None:
                out2[r][:, idx_nz] = out[r][:, idx_nz] * g_nz
        for r, (t, sel, d) in self.row_recomp.items():
            idx_t, W_new, b_sub = self._recomp_weights(t, sel, d, out.dtype, out.device)
            out2[r][:, idx_t] = F.linear(kept[r]["rx"].to(out.dtype), W_new, b_sub)
        for r, e in self.nulls:
            s = self._once(("null_s", e, out.dtype, out.device),
                           lambda e=e: torch.tensor(null_factor(e, out.dtype), dtype=out.dtype, device=out.device))
            out2[r] = out[r] * s
        return out2


def _compile(lm: nn.Module, programs: Sequence[Union[Arm, Null, None]]) -> Dict[str, _Site]:
    """The programs as hooks to install, one site a linear touched. `Refused` for a program that cannot be run;
    nothing is installed here."""
    dt = lm.norm.weight.dtype
    n_layers = len(lm.layers)
    sites: Dict[str, _Site] = {}

    def site(name: str, mod: nn.Module) -> _Site:
        if name not in sites:
            sites[name] = _Site(name, len(programs), linear_shape(mod)[1], mod=mod)
        return sites[name]

    for r, p in enumerate(programs):
        if p is None:
            continue
        if isinstance(p, Null):
            whole = isinstance(p.layer, (int, np.integer)) and not isinstance(p.layer, bool)
            if not whole or not 0 <= int(p.layer) < n_layers:
                raise Refused(_SAY_LAYER.format(L=p.layer, last=n_layers - 1, n=n_layers))
            e = float(p.e)
            s = null_factor(e, dt) if math.isfinite(e) else float("nan")
            if not (math.isfinite(s) and s > 0):
                raise Refused(_SAY_NULL_RANGE.format(e=e))
            if s == 1.0:
                raise Refused(_SAY_NULL_ONE.format(dtype=_dtype_name(dt), e=e))
            if _exact(s):
                raise Refused(_SAY_NULL_POWER.format(e=e, s=s))
            name = linear_name(int(p.layer), "down_proj")
            site(name, lm.get_submodule(name)).nulls.append((r, e))
            continue
        if not isinstance(p, Arm) or not isinstance(p.edit, Edit):
            raise Refused(f"A row's program is None (untouched), an Arm (an edit at a dose) or a Null; {p!r} is none "
                          "of them.")
        d = float(p.dose)
        if not math.isfinite(d):
            raise Refused("A dose is a finite number: it multiplies the edit's difference from the base.")
        for t in p.edit.terms:
            mod = _linear_of(lm, t.linear)
            if mod is None:
                raise Refused(f"The edit {p.edit.short()} names {t.linear}, which this model does not have.")
            n_out, n_in = linear_shape(mod)
            if t.kind == "cols":
                fits = int(t.idx[-1]) < n_in and (t.vectors is None or t.vectors.shape[1] == n_out)
            elif t.kind == "rows":
                fits = int(t.idx[-1]) < n_out and (t.vectors is None or t.vectors.shape[1] == n_in)
            else:
                fits = t.A.shape[1] == n_in and t.B.shape[0] == n_out
            if not fits:
                raise Refused(f"The edit {p.edit.short()} was made for another shape of {t.linear}: this model's is "
                              f"[{n_out}, {n_in}].")
            st = site(t.linear, mod)
            if t.kind == "cols":
                g = _asked(t, d)
                if (g != 1.0).any():
                    st.factors.append((r, t.idx, g))
                if t.vectors is not None:
                    st.adds[r] = (t, d)
            elif t.kind == "rows":
                if d == 0.0:
                    continue  # dose 0 leaves edited rows untouched: exact base by bits
                g = _asked(t, d)
                has_bias = getattr(mod, "bias", None) is not None
                sc_pos: List[int] = []
                rc_pos: List[int] = []
                for j in range(len(t.idx)):
                    if int(t.op[j]) == 0:
                        gj = float(g[j])
                        if gj == 1.0:
                            continue
                        if not has_bias and _exact(gj):
                            sc_pos.append(j)
                        else:
                            rc_pos.append(j)
                    else:
                        rc_pos.append(j)
                if sc_pos:
                    sp = np.asarray(sc_pos, dtype=np.int64)
                    st.row_scales[r] = (t.idx[sp], g[sp])
                if rc_pos:
                    st.row_recomp[r] = (t, np.asarray(rc_pos, dtype=np.int64), d)
            else:
                st.lows[r] = (t, d)
    return sites


@contextmanager
def per_row(lm: nn.Module, programs: Sequence[Union[Arm, Null, None]]) -> Iterator[None]:
    """For the length of the block, row r of every forward of `lm` runs `programs[r]`: None (untouched), `Arm(edit,
    dose)` or `Null(layer, e)`. At most one forward pre-hook and one forward hook go on each linear some program
    touches, and every one comes off when the block ends, however it ends. An untouched row's activations are at most
    multiplied by 1, which is exact, and its output copied: no other arithmetic touches it, and a linear no program
    names gets no hook. A row at dose 0 leaves the bits alone (on both stand-ins it is the base row by bits; on the
    card too, for a scale, a set and a low-rank term at 12 and 32 rows, 16 November 2026). The hooks act at every
    position and never index one, so a decode step with a cache runs the same code. `Refused`, before anything is
    installed, for a program that cannot be run: an edit that names a linear this model lacks or of another shape, or
    a null row whose factor is 1 or a power of two in the model's dtype. A forward whose row count is not
    `len(programs)` raises `RuntimeError` from inside the hook. A dose is not judged here: `dose_refusal` is the
    caller's to ask."""
    programs = list(programs)
    sites = _compile(lm, programs)
    dt = lm.norm.weight.dtype
    handles: List[Any] = []
    try:
        for name, st in sites.items():
            mod = lm.get_submodule(name)
            if st.scales(dt) or st.adds or st.lows or st.row_recomp:
                handles.append(mod.register_forward_pre_hook(st.pre))
            if st.adds_after():
                handles.append(mod.register_forward_hook(st.post))
        yield
    finally:
        for h in handles:
            h.remove()
        for st in sites.values():
            st.kept, st._made = {}, {}


# ------------------------------------------------------------------------- the folded form and the base assertion


def hook_census(lm: nn.Module) -> Dict[str, List[int]]:
    """Module path -> the sorted ids of its forward pre-hooks and forward hooks, for every module under `lm` that has
    any (the text model itself is ""). The ids are the handles', which no two hooks of a process share, so a hook
    swapped for another shows as well as one added. Hooks torch keeps for every module at once are listed under
    "(every module)": they act on this model too."""
    out: Dict[str, List[int]] = {}
    for name, m in lm.named_modules():
        ids = sorted(list(m._forward_pre_hooks) + list(m._forward_hooks))
        if ids:
            out[name] = [int(i) for i in ids]
    everywhere = nn.modules.module
    ids = sorted(list(getattr(everywhere, "_global_forward_pre_hooks", {}))
                 + list(getattr(everywhere, "_global_forward_hooks", {})))
    if ids:
        out["(every module)"] = [int(i) for i in ids]
    return out


def _overlay_dicts(m: nn.Module) -> List[Dict[int, torch.Tensor]]:
    return [d for d in (getattr(m, attr, None) for attr in _OVERLAYS) if isinstance(d, dict)]


def overlays_in_force(lm: nn.Module) -> List[Tuple[str, int]]:
    """(module path, entries) for every module under `lm` whose overlay dicts are not empty: the vault's edits, or a
    folded edit still in place. Its column and row overlays are counted together."""
    out: List[Tuple[str, int]] = []
    for name, m in lm.named_modules():
        k = sum(len(d) for d in _overlay_dicts(m))
        if k:
            out.append((name, k))
    return out


def assert_base(lm: nn.Module, hooks_at_load: Optional[Dict[str, List[int]]] = None) -> None:
    """The model is the base: its hooks are those of `hooks_at_load` (a census taken at load; none when None), id for
    id, and every overlay dict on every module is empty. Otherwise `NotBase`, naming up to three of what it found."""
    then = {str(k): sorted(int(i) for i in v) for k, v in (hooks_at_load or {}).items() if v}
    now = hook_census(lm)
    found: List[str] = []
    for name in sorted(set(then) | set(now)):
        where = f"on {name}" if name else "on the text model itself"
        added = [i for i in now.get(name, []) if i not in then.get(name, [])]
        gone = [i for i in then.get(name, []) if i not in now.get(name, [])]
        if added:
            many = len(added) != 1
            found.append(f"{len(added)} hook{'s' if many else ''} that {'were' if many else 'was'} not there at load "
                         f"{where}")
        if gone:
            found.append(f"{len(gone)} hook{'s' if len(gone) != 1 else ''} of the load gone from "
                         f"{name or 'the text model itself'}")
    for name, k in overlays_in_force(lm):
        found.append(f"{k} overlay{'s' if k != 1 else ''} in force on {name}")
    if found:
        more = f", and {len(found) - 3} more" if len(found) > 3 else ""
        raise NotBase(f"The model is not the base: {'; '.join(found[:3])}{more}. A reading is taken on the base model "
                      "only.")


@contextmanager
def folded(lm: nn.Module, edit: Edit, dose: float = 1.0, *,
           hooks_at_load: Optional[Dict[str, List[int]]] = None) -> Iterator[None]:
    """For the length of the block the edit is written into the overlay dicts every packed linear carries
    (`_memit_col_overlays` for column terms, `_memit_row_overlays` for row terms), at this dose, for every row alike:
    the form the vault's edits take, and the one an edit would ship in. It is entered outside any `per_row` block.
    `NotBase` when an overlay is already in force (the vault's edit: it is left as it was); `Refused` with `can_fold`'s
    sentence when the edit cannot be folded. Each new column or row is worked in float32 from the base slice and cast
    once to the model's dtype; where a scale's factor at this dose is exactly 0 the slice is zeros. When the block
    ends, however it ends, every overlay dict of every module is put back as it was found and `assert_base` runs."""
    inforce = overlays_in_force(lm)
    if inforce:
        name, k = inforce[0]
        more = f" and on {len(inforce) - 1} more" if len(inforce) > 1 else ""
        raise NotBase(f"The model is not the base: {k} overlay{'s' if k != 1 else ''} in force on {name}{more} (a "
                      "vault edit, or one left behind). A folded edit is written over the base model only.")
    why = can_fold(lm, edit)
    if why:
        raise Refused(why)
    d = float(dose)
    if not math.isfinite(d):
        raise Refused("A dose is a finite number: it multiplies the edit's difference from the base.")
    dt = lm.norm.weight.dtype
    was = [(od, dict(od)) for m in lm.modules() for od in _overlay_dicts(m)]
    try:
        with torch.no_grad():
            for t in edit.terms:
                mod = lm.get_submodule(t.linear)
                Ws = (dense_rows(mod, _long(t.idx)) if t.kind == "rows" else dense_cols(mod, _long(t.idx))).float()
                g = _asked(t, d)
                g_t = torch.from_numpy(g.astype(np.float32)).to(Ws.device)
                new = g_t[:, None] * Ws
                new[g_t == 0] = 0.0
                if t.vectors is not None:
                    at = torch.from_numpy(np.nonzero(t.op != 0)[0]).to(Ws.device)
                    V = torch.from_numpy(t.vectors[t.op != 0].copy()).to(Ws.device)
                    new[at] = new[at] + d * V
                new = new.to(dt)
                new[torch.from_numpy(np.array((g == 0) & (t.op == 0))).to(Ws.device)] = 0  # zeros, as the pilots wrote
                attr = _OVERLAYS[1] if t.kind == "rows" else _OVERLAYS[0]
                for j, n in enumerate(t.idx):
                    getattr(mod, attr)[int(n)] = new[j]
        yield
    finally:
        for od, before in was:
            od.clear()
            od.update(before)
        assert_base(lm, hooks_at_load)


# ------------------------------------------------------------------------ Phase 5 provenance & base integrity


def rebuilt_recipe(edit: Edit) -> Optional[Dict[str, Any]]:
    """Return a pure-JSON `{"terms": [...]}` recipe when every term of `edit` can be rebuilt from the public base
    weights with no carried payload arrays (column or row `silence` / `scale`), else `None`."""
    terms_out: List[Dict[str, Any]] = []
    for t in edit.terms:
        if t.kind not in ("cols", "rows") or t.vectors is not None:
            return None
        if int(t.op.max(initial=0)) != 0:
            return None
        idx = [int(n) for n in t.idx]
        fac = [float(f) for f in t.factor]
        all_zero = all(f == 0.0 for f in fac)
        all_same = all(f == fac[0] for f in fac)
        if t.kind == "cols":
            if all_zero:
                terms_out.append({"layer": int(t.layer), "silence": idx})
            elif all_same:
                terms_out.append({"layer": int(t.layer), "scale": idx, "factor": float(fac[0])})
            else:
                terms_out.append({"layer": int(t.layer), "scale": idx, "factors": fac})
        else:
            if all_zero:
                terms_out.append({"layer": int(t.layer), "which": t.which, "silence_rows": idx})
            elif all_same:
                terms_out.append({"layer": int(t.layer), "which": t.which, "scale_rows": idx, "factor": float(fac[0])})
            else:
                terms_out.append({"layer": int(t.layer), "which": t.which, "scale_rows": idx, "factors": fac})
    out: Dict[str, Any] = {"terms": terms_out}
    if edit.note:
        out["note"] = edit.note
    return out


def edit_provenance(edit: Edit, *, saved_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Phase 5 provenance manifest: marks each term `"rebuilt"` (reconstructible from the public base with its pure-JSON
    recipe) or `"carried"` (requires external array payload, naming sha256 hashes only and never embedding weights)."""
    man = edit.manifest()
    terms_prov: List[Dict[str, Any]] = []
    any_carried = False
    for t, mt in zip(edit.terms, man["terms"]):
        is_carried = bool(mt.get("carried"))
        if is_carried:
            any_carried = True
            entry: Dict[str, Any] = {
                "linear": t.linear,
                "layer": int(t.layer),
                "which": t.which,
                "kind": t.kind,
                "provenance": "carried",
                "sha256": dict(mt.get("sha256") or {}),
            }
            if t.kind in ("cols", "rows"):
                idx_key = "columns" if t.kind == "cols" else "rows"
                entry[idx_key] = list(mt[idx_key])
                entry["ops"] = list(mt["ops"])
                entry["factors"] = list(mt["factors"])
                entry["payload_key"] = mt.get("vectors")
            else:
                entry["rank"] = int(mt["rank"])
                entry["payload_keys"] = {"A": mt.get("A"), "B": mt.get("B")}
            terms_prov.append(entry)
        else:
            idx = [int(n) for n in t.idx]
            fac = [float(f) for f in t.factor]
            all_zero = all(f == 0.0 for f in fac)
            all_same = all(f == fac[0] for f in fac)
            if t.kind == "cols":
                rec: Dict[str, Any] = (
                    {"layer": int(t.layer), "silence": idx}
                    if all_zero else
                    ({"layer": int(t.layer), "scale": idx, "factor": float(fac[0])}
                     if all_same else {"layer": int(t.layer), "scale": idx, "factors": fac})
                )
            else:
                rec = (
                    {"layer": int(t.layer), "which": t.which, "silence_rows": idx}
                    if all_zero else
                    ({"layer": int(t.layer), "which": t.which, "scale_rows": idx, "factor": float(fac[0])}
                     if all_same else {"layer": int(t.layer), "which": t.which, "scale_rows": idx, "factors": fac})
                )
            terms_prov.append({
                "linear": t.linear,
                "layer": int(t.layer),
                "which": t.which,
                "kind": t.kind,
                "provenance": "rebuilt",
                "recipe_term": rec,
                "sha256": dict(mt.get("sha256") or {}),
            })

    payload_sha256: Optional[str] = None
    if any_carried:
        if saved_dir is not None and (Path(saved_dir) / "payload.safetensors").is_file():
            payload_sha256 = hashlib.sha256((Path(saved_dir) / "payload.safetensors").read_bytes()).hexdigest()
        else:
            # Deterministic hash of the carried array hashes so a registration folder pins payload identity even before saving
            carried_hashes = [
                {"linear": tp["linear"], "kind": tp["kind"], "sha256": tp["sha256"]}
                for tp in terms_prov if tp["provenance"] == "carried"
            ]
            payload_sha256 = _sha({"carried_arrays": carried_hashes})

    return {
        "format": FORMAT,
        "version": VERSION,
        "edit_id": edit.id,
        "base": copy.deepcopy(edit.base),
        "linears": edit.linears(),
        "layers": edit.layers(),
        "all_rebuilt": not any_carried,
        "provenance": "rebuilt" if not any_carried else "carried",
        "rebuilt_recipe": rebuilt_recipe(edit),
        "payload_sha256": payload_sha256,
        "note": edit.note,
        "terms": terms_prov,
    }


IMPLANT_COSINE = 0.85  # a write this close to one unembedding row is called the shape of a single-piece implant
IMPLANT_CHUNK = 16384  # rows of the unembedding read at a time by the implant label: the 31B's table is never copied whole
# where this house's linears keep weights that `Module.state_dict` does not hold (`base_checksum` hashes these too)
_WEIGHT_ATTRS = ("weight_packed", "weight_scale", "weight_blocks", "int4_packed", "int4_scales_zeros")


def implant_shape(edit: Edit, lm: nn.Module, W_U: torch.Tensor, *, dose: float = 1.0) -> Dict[str, Any]:
    """Whether `edit` has the shape of a last-block, single-word-piece implant (`design/synthesis.md` §viii: "a
    last-block neuron whose write is dominated by one word-piece's unembedding"): a label, not a judgement of what
    the edit is for.

    Looked at: every direction the edit writes into the stream in the last tenth of the layers, as the stream takes
    it (times the layer's `post_feedforward_layernorm` weight): the change to each edited `down_proj` column, and
    each column of a low-rank term's `B` on `down_proj`. A direction whose cosine with one unembedding row is
    `IMPLANT_COSINE` or more in size is a hit. Not looked at, and named in `not_examined`: row terms on `gate_proj`
    and `up_proj` (they change what a neuron reads, and write nothing to the stream themselves) and any term below
    the last tenth. `examined` counts the directions looked at: `flagged` False with `examined` 0 says nothing was
    in reach, not that the edit is clear."""
    n_layers = len(lm.layers)
    late_cut = max(0, int(math.floor(0.9 * n_layers)))
    d = float(dose)
    hits: List[Dict[str, Any]] = []
    not_examined: List[str] = []

    cand: List[Tuple[int, str, torch.Tensor]] = []  # (layer, what, the direction it writes into the stream)
    for t in edit.terms:
        if t.which != "down_proj":
            not_examined.append(f"{t.linear} ({t.kind}): a row term changes what a neuron reads and writes nothing "
                                "to the stream itself")
            continue
        if t.layer < late_cut:
            not_examined.append(f"{t.linear} ({t.kind}): below the last tenth of the layers")
            continue
        w_post = _w_post(lm, t.layer)
        if t.kind == "cols":
            mod = lm.get_submodule(t.linear)
            with torch.no_grad():
                Wc = dense_cols(mod, torch.from_numpy(np.array(t.idx, dtype=np.int64))).float().cpu().numpy()
            delta = (_asked(t, d)[:, None] - 1.0) * Wc  # [K, D]
            if t.vectors is not None:
                m = t.op != 0
                delta[m] = delta[m] + d * t.vectors[m]
            names = [f"neuron {int(n)}" for n in t.idx]
        else:  # low-rank: each column of B is a direction written
            delta = (d * t.B).T.astype(np.float64)  # [r, D]
            names = [f"low-rank direction {k}" for k in range(delta.shape[0])]
        dS = torch.from_numpy((delta * w_post[None, :]).astype(np.float32))
        for j in range(dS.shape[0]):
            if float(torch.linalg.norm(dS[j])) > 1e-8:
                cand.append((int(t.layer), names[j], dS[j]))
    examined = len(cand)
    if cand:
        # every direction against every unembedding row, the table read once, `IMPLANT_CHUNK` rows at a time on the
        # CPU (the 31B's is 262,144 by 5,376: it is never copied whole, and not once a direction)
        V = torch.stack([v for _l, _n, v in cand]).float().cpu()
        V = V / torch.linalg.norm(V, dim=1, keepdim=True)
        K = int(V.shape[0])
        best_abs, best_cos = torch.full((K, 2), -1.0), torch.zeros((K, 2))
        best_idx = torch.zeros((K, 2), dtype=torch.long)
        step = max(1, int(IMPLANT_CHUNK))
        for at in range(0, int(W_U.shape[0]), step):
            part = W_U[at:at + step].detach().float().cpu()
            part = part / torch.linalg.norm(part, dim=1, keepdim=True).clamp_min(1e-8)
            dots = V @ part.T  # [directions, rows of this part]
            top = torch.topk(dots.abs(), k=min(2, int(dots.shape[1])), dim=1)
            all_abs = torch.cat([best_abs, top.values], dim=1)
            all_idx = torch.cat([best_idx, top.indices + at], dim=1)
            all_cos = torch.cat([best_cos, torch.gather(dots, 1, top.indices)], dim=1)
            keep = torch.topk(all_abs, k=2, dim=1).indices
            best_abs, best_idx, best_cos = (torch.gather(x, 1, keep) for x in (all_abs, all_idx, all_cos))
        for j, (layer, what, _v) in enumerate(cand):
            top_cos = float(best_cos[j, 0])
            if abs(top_cos) >= IMPLANT_COSINE:
                nxt = float(best_abs[j, 1])
                hits.append({"layer": layer, "page_layer": layer + 1, "what": what,
                             "top_piece_id": int(best_idx[j, 0]), "top_cosine": round(top_cos, 4),
                             "next_cosine": round(nxt, 4) if nxt >= 0.0 else None})
    return {"checked": True, "flagged": bool(hits), "late_layer_cutoff": late_cut, "cosine_bar": IMPLANT_COSINE,
            "examined": examined, "not_examined": not_examined, "hits": hits[:16]}


def base_checksum(
    lm: nn.Module,
    *,
    hooks_at_load: Optional[Dict[str, List[int]]] = None,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """A sha256, worked afresh at every call, over every weight the model under `lm` holds: each module's own
    parameters and the buffers it would save (a norm's weight, an embedding's table, a plain linear's weight), and a
    packed linear's words and scales as stored, whichever way it keeps them. Beside it, `assert_base`'s overlay and
    hook checks. A trial takes it before its first reading and after its last: equal means no stored number of the
    model changed in between.

    `linears_checked` counts the linears among what was hashed (packed, or plain with a weight of two dimensions),
    `tensors_checked` every tensor. A buffer a module does not save is left out (a cache it may fill as it runs is
    not the model), but for the weights this house's own linears keep in such buffers or in plain attributes
    (`_WEIGHT_ATTRS`: a W4A16 linear's words and scales, a GGUF linear's blocks, an int4 linear's packed words). With
    `names`, only the modules so named."""
    assert_base(lm, hooks_at_load)
    h_all = hashlib.sha256()
    linears = tensors = 0
    wanted = set(names) if names is not None else None
    for mod_name, mod in sorted(lm.named_modules(), key=lambda kv: kv[0]):
        if wanted is not None and mod_name not in wanted:
            continue
        store: Dict[str, torch.Tensor] = {}
        for key, t in mod._parameters.items():
            if isinstance(t, torch.Tensor):
                store[key] = t
        unsaved = getattr(mod, "_non_persistent_buffers_set", set())
        for key, t in mod._buffers.items():
            if isinstance(t, torch.Tensor) and key not in unsaved:
                store[key] = t
        for key in _WEIGHT_ATTRS:  # a packed linear's own, however the module keeps them
            t = getattr(mod, key, None)
            if isinstance(t, torch.Tensor):
                store[key] = t
        if not store:
            continue
        packed = ("weight_packed" in store and "weight_scale" in store) or "weight_blocks" in store \
            or "int4_packed" in store
        w = store.get("weight")
        if packed or (isinstance(mod, nn.Linear) and isinstance(w, torch.Tensor) and w.dim() == 2):
            linears += 1
        hm = hashlib.sha256()
        for key in sorted(store):
            t = store[key]
            shape = "x".join(str(int(n)) for n in t.shape)
            hm.update(f"{key}:{shape}:{t.dtype}:".encode("ascii"))
            hm.update(_raw(t.to(torch.int32) if key == "weight_packed" else t))
            tensors += 1
        h_all.update(f"{mod_name}:{hm.hexdigest()}\n".encode("ascii"))
    return {
        "sha256": h_all.hexdigest(),
        "linears_checked": linears,
        "tensors_checked": tensors,
        "overlays_in_force": len(overlays_in_force(lm)),
        "hooks_now": sum(len(v) for v in hook_census(lm).values()),
    }
