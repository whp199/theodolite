#!/usr/bin/env python3
"""A CPU-only check of 1a, the first slice of the weight-edit kernel: the edit as a hashed value
(backend/weight_edits.py), the paired and the written reading (backend/levelling.py), the ledger
(backend/level_book.py) and the command line (scripts/levelling.py). It loads neither the real model nor the lens,
and hides the GPU.

It was written from the build spec (experiments/results/weight_space_exploration/BUILD_SPEC_1a.md, section 6.1), not
from the three modules, and calls only the names section 3 of the spec makes public: its worth is that it holds the
modules to the spec and not to themselves. Every number it compares against is the spec's, and the comment beside it
says where the spec has it. Where the spec fixes a refusal's sentence but not how one of its blanks is written (a
dtype's name, a path, a list of ids), the sentence round the blank is compared and the blank is left free.

Two stand-ins. F is the float32 six-layer stand-in of scripts/atlas_stand_in.py, for plumbing. P is the same model
in bf16 with every block linear packed to 4 bits and patched as the checkpoint is (scripts/check_dequant.py's
packed_stand_in), for everything claimed by bits. Both are random over about 600 word-pieces, so a typed answer reads
about -6.4 nats a piece and every plan here gives its own item bar, -8.0 (spec 6.1; measured on the CPU, 14 November
2026). The whole check runs at ONE thread: above one thread the rows of one written batch on the float32 stand-in do
not write the same log-probabilities (up to 4.8e-7 apart at 12 threads with no hook at all; measured on this machine,
14 November 2026, spec section 1). The ledger and the saved edits go to a temporary folder, never to the real ones.

It checks, section by section (the spec's own numbering):
 1. the hashes: the array hash, the edit id, the item id and the ledger's two entry hashes are the spec's worked
    values; both modules' canonical JSON is the same bytes; an edit's id does not depend on the order of its terms
    or neurons or on its note, and does depend on a factor, the base tensor and the model id; two equal edits are
    not `==`, their ids are;
 2. the gate (F): every refusal of the spec's table 2.2 in its words, the earlier row's words when two rules are
    broken, nothing installed or written by a refusal; a 4-bit linear marked int4mm; a recipe file that is a pickle
    or of another dtype or shape;
 3. save and load (F and P): the same id and arrays back; a scale-only edit carries no payload; a byte changed in
    the payload, a factor changed in the manifest and a model of other weights are each refused in their words;
 4. the per-row hooks by bits (P): untouched rows and dose-0 rows against an unhooked forward; a silenced set and
    sets scaled by 2 and by -0.5 against the folded weight at 12 and at 32 rows; one edit in two rows; a restart
    from the kept stream under hooks at layers 0, 2 and 5; the folded context leaves the model the base; the base
    assertion knows hooks, not counts;
 5. the per-row hooks to rounding (F, with the overlay forward installed where a folded form is wanted, and P):
    scales by 1.5 and -3, a set, an add and a low-rank term against the folded form and against a copy of the model
    with the weight edited; doses; what the cast to the model's dtype does; null rows; a forward of another row
    count; what can be folded;
 6. a decode step (F and P): writing under the hooks, row for row against an unhooked write; sequence length 1 after
    the first step; a row that stops early; an edited row's pieces read again teacher-forced;
 7. control edits (F): seeded, of the same layer and size, none of the edit's own sites, drawn as the spec says;
 8. items and plans (F): the chat ids, the item's parts, every refusal of the spec's table 2.6 in its words, the
    defaults, the layout of rows;
 9. the read-outs against brute force (F): KL, the text's own log-probability and the log-odds of named pieces from
    a whole forward's full log-softmax; the floor; `stats`, `agree` and `rank_among` on hand-made numbers;
10. one reading end to end (F, then P): the result key for key against the spec's 2.7; refusals as results; planted
    faults (a hook of the check's own) void the reading or mark the item unstable;
11. the written reading (P);
12. the ledger: the chain, a changed byte, a torn last line, two processes appending at once;
13. the cost estimate against the spec's arithmetic, worked by hand;
14. the command line and its exit codes;
15. the words the spec keeps out of the four source files and out of what was printed.
What it cannot show is the card's arithmetic: the two batch classes, bf16 on the GPU kernels, the cost. That is
scripts/verify_levelling.py's, on the card.

After the build's three reviewers (14 November 2026) it also holds lines for the breaks they planted in scratch copies
of the modules and that no line then caught: a set and an add at doses other than 1 (5a); which arithmetic the
output-side add runs, by bits (5a, P); the output-side hook in a decode step (6); the read-outs under an unembedding
200 times as sharp, with a strong edit and a main dose of 0.5, compared to 1 part in 10,000 (9, F and P: the first
comparison's 1e-5 is the size of the figures it compares); the twin and the control edits at a main dose that is not
1; which forwards run the layers below the edit; each of the four rules that void a reading by a fault of its own;
the gate at the large row count and on the mean; the sign test; the larger of two floors (10). Each of those lines was
shown to fail against the break it is there for (the mender's notes of that day list which). And lines for what the
mends changed: an edit met again passes the gate again, the control mark lifts one cap only, a recipe's files lie
inside the plan's folder, what a void or a refused item prints, a folded twin that is not finite.

After the card check (16 November 2026) the bars were set from it by the spec's own rules, and the lines that asserted
what changed assert the new values: the cost figures are the card's refit (8, 13), and the item gate is a printed
caution, not a refusal (10): an item under the bar is read in both classes, its gate's caution given in words, and the
summary counts it apart. No line was loosened.

A section that stops at an exception is one failing line saying where; the sections after it still run.

Run: .venv/bin/python -I scripts/check_levelling.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # nothing here needs the GPU, and nothing may be fetched: the stand-ins bring their own tokenizer
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import copy
import dataclasses
import hashlib
import importlib
import itertools
import json
import math
import pickle
import re
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn.functional as F

# Before anything else (spec 6.1): one thread, and the ledger and the saved edits in a temporary folder.
torch.set_num_threads(1)
BOOK_VAR, EDITS_VAR = "THEODOLITE_LEVEL_BOOK", "THEODOLITE_EDITS_DIR"
TMP = Path(tempfile.mkdtemp(prefix="check-levelling-"))
os.environ[BOOK_VAR] = str(TMP / "book" / "level_book.jsonl")
os.environ[EDITS_VAR] = str(TMP / "edits")

import atlas_stand_in as asi
import backend.atlas_ops as ao
import backend.causal_check as cc
from backend import census as cz
from backend import gguf_weights as gw

MISSING: Dict[str, str] = {}


def _load(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - a module that does not import is a finding, said in main()
        MISSING[name.rsplit(".", 1)[-1]] = f"{type(exc).__name__}: {exc}"
        return None


we = _load("backend.weight_edits")
lb = _load("backend.level_book")
lv = _load("backend.levelling")
CLI = ROOT_DIR / "scripts" / "levelling.py"
SOURCES = [ROOT_DIR / "backend" / "weight_edits.py", ROOT_DIR / "backend" / "levelling.py",
           ROOT_DIR / "backend" / "level_book.py", CLI]  # "the four new source files" of 6.1 §15

failures: List[str] = []
passed = 0
TALLY: List[Tuple[str, int, int]] = []  # (section, checks that hold, checks that fail)
PRINTED: List[str] = []  # every line lv.lines gave, and what the command line printed: read again in section 15

# The stand-ins' sizes (scripts/atlas_stand_in.py)
NL, D, N = asi.N_LAYERS, asi.HIDDEN, asi.FFN  # 6 layers, 64 wide, 128 neurons a layer
GATE = -8.0  # the bar every stand-in plan gives (spec 6.1; 5.1)
W = "\x00"  # in a wanted sentence: a blank the spec does not fix the writing of

# ---- the spec's sentences, word for word (2.2, 2.4, 2.5, 2.6, 4.1, 4.6, 4.9, 4.10)
S_ROWS = ("Row edits on gate_proj and up_proj (what a neuron reads) are not built: their per-row form, the edited "
          "coordinate recomputed, has not been measured. They wait for 1b.")
S_DIRECTION = ("Direction removal is not built: it must go through the post-feed-forward norm's weight and the layer's "
               "quiet mask, and that form is 1b's.")
S_ATTENTION = (f"Edits to attention linears ({W}) are refused until their facts are in the gate and a card check has "
               "passed: q, k and v each pass a per-head norm, and on a full-attention layer the values are the keys. "
               "They wait for 1b.")
# decision 1 of the coordinating session takes low-rank terms off gate_proj and up_proj: this sentence as the spec's
# "Found in the build" amends it (2.2's table said "down_proj, gate_proj or up_proj for a low-rank term"), and the
# refusal that decision adds, in the words fixed there
S_NOT_FFN = ("Only the feed-forward linears take edits in 1a: layers.<L>.mlp.down_proj for a column edit, and down_proj "
             f"for a low-rank term. \"{W}\" is not one. The embedding, the norms and the layer scalars are not built.")


def s_lowrank_which(layer: int, which: str) -> str:
    return (f"A low-rank term on layers.{layer}.mlp.{which} is not built in 1a: on {which} it would be the added form of a "
            "change to what neurons read, not the recomputed form asked of a row edit, and that waits for 1b. In 1a a "
            "low-rank term names a down_proj.")
S_NO_COLUMN = "A term that names no neuron changes nothing."
S_NOT_FINITE = f"An edit holds a number that is not finite ({W})."
S_LOWRANK_FOLD = "A low-rank term has no folded form yet, so this edit cannot be read folded."
# 4.7 step 3's sentence as the spec's "Found in the build" amends it, corrected from the card check of 16 November 2026
# (verify_1a.json, sections 5 and 5a: a silencing and a scale by 2 read as their folded forms by bits, 48 of 48)
S_EXACT = ("a silencing or a scale by a power of two is exact in arithmetic on the input side; on the card a silencing "
           "and a scale by 2 read as their folded forms by bits, 48 comparisons of 48 (16 November 2026)")
S_FLOOR = ("The floor is the largest of {n0} null rows, not a bound: on 14 November 2026 a single neuron silenced at "
           "random read above its forward's floor in 26 of 72 readings, up to 2.25 times it.")
S_NOTHING = ("This edit scales every column it names by exactly 1 and adds nothing: at any dose its row is the base row "
             "again, so there is nothing to read.")
S_NO_ANSWER = ("An item needs an answer of at least one word-piece: its pieces are what is scored and what the item is "
               "gated on.")
S_DOSE = ("A dose is a finite number between -4 and 4: it multiplies the edit's difference from the base (1 the edit, "
          "0 the base, -1 its mirror).")
S_DOSE_0 = "A reading needs a dose other than 0: at 0 the edit is the base."
S_NULLS = ("A reading takes at least two null rows, one with e above 0 and one below: one null row is one draw of the "
           "rounding.")
S_NO_READOUT = "A reading reads something: turn on kl, own or odds."
S_CONTROLS = ("Control edits are other sites of the same layer and size, drawn from seed {seed}. They are NOT matched "
              "on how hard the sites fire on these items or on what the edit delivers, and the edit's own sites were "
              "chosen where theirs were drawn, so a rank against them says only that the edit differs from an "
              "unmatched random edit of its size.")
S_ROWS_SAY = ("Every item was read at {small} rows and at {large} rows. Agreement at these two row counts does not "
              "show that another row count would agree: on 14 November 2026 batches of 2, 3 and 4 rows each read "
              "differently.")
S_PAIRED = ("These figures are read along the answer given, piece by piece. They do not say what the edited model "
            "would write.")


def s_rows_range(small: int, large: int) -> str:
    # "with no start mark": the pilots' raw text had none (spec section 1; the amended words are in its section 12)
    return ("The small batch is 5 to 12 rows and the large one 16 to 64 (the ruling's two classes: at 150 pieces of raw "
            "text with no start mark this card ran one arithmetic up to 12 rows and another from 16, 14 November 2026); "
            f"{small} and {large} were asked.")


def s_null_one(e: float) -> str:
    return (f"In {W} 1 + e is exactly 1 for e = {e:g}: that null row would be the base row again. In bf16 the nearest "
            "values are 1 + 1/128 and 1 - 1/256.")


def s_null_two(e: float, s: float) -> str:
    return (f"For e = {e:g} the null row's factor is {s:g}, a power of two: an exact scaling, which the norm may undo "
            "exactly, so it measures no rounding. Factors that serve: 1 + 1/128, 1 - 1/256, 0.75, 1.5, 3.")


# the item gate as a caution (the spec's open point 4, set from the card check of 16 November 2026: the base's
# log-probability did not separate the items whose two row counts disagreed); the words are the spec's section 12's, as
# its later paragraph corrects them: the 3 and the 48 are counted on the EDIT's arms (in sections 9 and 9a alike)
S_GATE_SAY = ("The bar is a caution, not a refusal: on 16 November 2026 the card read 51 items with it set aside, and the "
              "base's log-probability did not separate the 3 on which the edit's two row counts disagreed (-3.09, -4.13 "
              "and -9.64 nats a piece) from the 48 on which they agreed (the lowest -12.13). The item is read, and its two "
              "row counts carry the weight.")


def s_caution(small: float, large: float, rows_small: int, rows_large: int, bar: float) -> str:
    under = [rows for mean, rows in ((small, rows_small), (large, rows_large)) if mean < bar]
    where = "both row counts" if len(under) == 2 else f"{under[0]} rows"
    return (f"The base model gives this item's answer {small:.2f} at {rows_small} rows and {large:.2f} at {rows_large}, "
            f"in nats a piece on average: under the item gate's bar of {bar:.2f} at {where}. " + S_GATE_SAY)


def s_one_row(m: int, layer: int) -> str:
    # corrected from the card check of 16 November 2026 (verify_1a.json, section 6b: on two chat-formatted items at
    # layers 40 and 44 the one-row capture stood off the same-shape reading by 10.2 to 35.3 times the floor in 7 of 8;
    # the eighth, own 4 at layer 44 and 32 rows, sat at 0.14 of it, so "shown to agree", not "agreed", only on raw text)
    return ("A one-row capture replayed at many rows has been shown to agree with the same-shape reading only at layer "
            "44 (page layer 45), on raw text with no start mark, on 14 November 2026; on two chat-formatted items on "
            "16 November 2026 it stood off that reading at layers 40 and 44 by 10 to 35 times the floor in 7 of 8 "
            f"readings. It is taken from layer {m} (page layer {m + 1}) up, as a cheaper reading that is not the "
            f"same-shape one. This edit starts at layer {layer} (page layer {layer + 1}).")


# ------------------------------------------------------------------------------------------------- the check's own


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


def _where(exc: BaseException) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    here = [f for f in tb if Path(f.filename).name == Path(__file__).name]
    last = tb[-1] if tb else None
    said = f"line {here[-1].lineno} of this check" if here else "outside this check"
    if last is not None and Path(last.filename).name != Path(__file__).name:
        said += f", raised at {Path(last.filename).name}:{last.lineno}"
    return said


@contextmanager
def part(what: str) -> Iterator[None]:
    """One part of a section: an exception inside it is one failing line saying where, and the next part still runs."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        check(False, f"{what}: stopped at {type(exc).__name__}: {str(exc)[:300]} ({_where(exc)})")


def run_section(number: int, title: str, needs: Sequence[str], fn: Callable[..., None], *args: Any) -> None:
    before, good_before = len(failures), passed
    lacking = [m for m in needs if m in MISSING]
    if lacking or any(a is None for a in args):
        check(False, f"section {number} ({title}) was not run: "
              + (f"{', '.join(lacking)} did not import" if lacking else "a stand-in it needs could not be made"))
    else:
        with part(f"section {number} ({title})"):
            fn(*args)
    section(f"{number}. {title}", before)
    TALLY.append((f"{number}. {title}", passed - good_before, len(failures) - before))


def words_of(fn: Callable[[], Any], kind: Any = ValueError) -> Optional[str]:
    """What `fn` raises as `kind`, as a string; None when it raises nothing; another exception is said as such."""
    try:
        fn()
    except kind as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__} in place of {getattr(kind, '__name__', kind)}: {str(exc)[:200]}>"
    return None


def like(got: Optional[str], want: str) -> bool:
    """`got` is the sentence `want`, each W in `want` standing for a blank the spec leaves free."""
    if got is None:
        return False
    if W not in want:
        return got == want
    return re.fullmatch("".join(".+?" if c == W else re.escape(c) for c in want), got, re.S) is not None


def says(got: Optional[str], want: str, what: str) -> None:
    ok = like(got, want)
    check(ok, what if ok else f"{what}: wanted {want.replace(W, '...')!r}, got {got!r}")


def close(a: Any, b: Any, atol: float = 1e-5) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def near(a: Any, b: Any, rel: float = 1e-4, floor: float = 1e-7) -> bool:
    """Two figures within `rel` of the larger, or within `floor` of each other: a tolerance that follows the figure,
    for read-outs that run from 1e-5 nats to several (an absolute 1e-5 is the whole of a null row's figure)."""
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), floor)


def sha_arr(a: Any, tag: str) -> str:
    """The spec's array hash (2.1), written out again here."""
    a = np.ascontiguousarray(a, dtype={"i8": "<i8", "f4": "<f4", "u1": "u1"}[tag])
    h = hashlib.sha256()
    h.update((tag + ":" + ",".join(str(int(s)) for s in a.shape) + ":").encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def canon(obj: Any) -> bytes:
    """The spec's canonical JSON (2.1), written out again here."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(canon(obj)).hexdigest()


def _raw(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    return (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()


def own_pin(mod: Any) -> str:
    """The spec's pin of a linear (2.2), written out again here: packed words and scales, or the plain weight."""
    h = hashlib.sha256()
    if hasattr(mod, "weight_packed") and hasattr(mod, "weight_scale"):
        p_, s_ = mod.weight_packed, mod.weight_scale
        h.update(f"w4a16:{p_.shape[0]}x{p_.shape[1]}:{s_.shape[0]}x{s_.shape[1]}:{s_.dtype}:".encode("ascii"))
        h.update(_raw(p_))
        h.update(_raw(s_))
    else:
        w = mod.weight
        h.update(f"dense:{w.shape[0]}x{w.shape[1]}:{w.dtype}:".encode("ascii"))
        h.update(_raw(w))
    return h.hexdigest()


def own_edit_id(edit: Any) -> str:
    """The spec's identity of an edit (2.2), from the edit's own public fields."""
    terms = []
    for t in edit.terms:
        if t.kind == "cols":
            terms.append({"linear": t.linear, "kind": "cols", "count": int(len(t.idx)), "columns": sha_arr(t.idx, "i8"),
                          "ops": sha_arr(t.op, "u1"), "factors": sha_arr(t.factor, "f4"),
                          "vectors": None if t.vectors is None else sha_arr(t.vectors, "f4")})
        else:
            terms.append({"linear": t.linear, "kind": "lowrank", "rank": int(t.A.shape[0]), "A": sha_arr(t.A, "f4"),
                          "B": sha_arr(t.B, "f4")})
    return sha_obj({"format": "theodolite-edit", "version": 1, "base": edit.base, "terms": terms})


def own_agree(a: Any, b: Any, floor: float, signed: bool, ratio: float = 3.0) -> Tuple[Optional[bool], Optional[float]]:
    """The spec's rule of agreement between two row counts (4.9), written out again here."""
    if a is None and b is None:
        return None, None
    if a is None or b is None:
        return False, None
    lo, hi = max(min(abs(a), abs(b)), floor), max(abs(a), abs(b))
    r = 1.0 if hi == 0 else (math.inf if lo == 0 else hi / lo)
    return (r <= ratio and (not signed or a * b >= 0 or hi <= ratio * floor)), r


def own_layout(doses: Sequence[float], n_controls: int, null_e: Sequence[float], rows: int) -> List[List[Optional[str]]]:
    """The spec's rows of a forward (2.6), written out again here, for a paired reading's one class."""
    n0, slots = len(null_e), rows - 2 - len(null_e)
    arms = [f"edit:{float(d)!r}" for d in doses] + [f"control:{c}" for c in range(n_controls)]
    out = []
    for f in range(max(1, math.ceil(len(arms) / slots))):
        mine = arms[f * slots:(f + 1) * slots]
        out.append([None] + [f"null:{float(e)!r}" for e in null_e] + mine + [None] * (slots - len(mine)) + [None])
    return out


def json_clean(obj: Any, path: str = "") -> Optional[str]:
    """None when `obj` holds only what the spec's 2.1 lets a hashed or written value hold, else where it does not."""
    if obj is None or type(obj) in (str, int, bool):
        return None
    if type(obj) is float:
        return None if math.isfinite(obj) else f"{path}: {obj!r} is not finite"
    if type(obj) is list:
        return next((bad for i, v in enumerate(obj) if (bad := json_clean(v, f"{path}[{i}]"))), None)
    if type(obj) is dict:
        if any(type(k) is not str for k in obj):
            return f"{path}: a key that is not a string"
        return next((bad for k, v in obj.items() if (bad := json_clean(v, f"{path}.{k}"))), None)
    return f"{path}: a {type(obj).__name__}"


def find_key(obj: Any, key: str) -> List[Any]:
    """Every value stored under `key` anywhere in a nested dict or list (the edit's report is a dict whose inner
    shape, a list of entries or a dict of them, the spec does not fix)."""
    out: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out += find_key(v, key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += find_key(v, key)
    return out


class Box:
    """One stand-in and what the checks read of it."""

    def __init__(self, rt: Any, label: str) -> None:
        self.rt, self.label = rt, label
        self.hf = rt.hf_model
        self.lm = cc.language_model(self.hf)
        self.tok = rt.tokenizer
        self.W_U = rt.lm_head_weight
        self.model_id = rt.model_id
        self.softcap = ao.softcap_of(self.hf)
        self.dtype = self.lm.norm.weight.dtype

    def mod(self, layer: int, which: str = "down_proj") -> Any:
        return getattr(self.lm.layers[layer].mlp, which)

    def build(self, *specs: Any, note: str = "") -> Any:
        return we.build(self.lm, list(specs), model_id=self.model_id, weights_id=None, note=note)

    def whole(self, ids: Sequence[int], rows: int) -> torch.Tensor:
        """A whole forward as the spec's section 4 runs it: the ids repeated, no mask, no position ids, no cache."""
        x = torch.tensor([[int(t) for t in ids]] * rows, dtype=torch.long)
        return self.lm(input_ids=x, use_cache=False).last_hidden_state

    def write(self, prompt: Sequence[int], rows: int, max_new: int, stop_ids: Sequence[int]) -> Dict[int, List[Tuple[int, float, bool]]]:
        """Each row's written (piece, log-probability, ended) by atlas_ops.write_rows, with whatever hooks are on."""
        out: Dict[int, List[Tuple[int, float, bool]]] = {r: [] for r in range(rows)}
        for st in ao.write_rows(self.lm, self.W_U, [[int(t) for t in prompt]] * rows, None, max_new=max_new,
                                stop_ids=list(stop_ids), bos_id=self.tok.bos_token_id, softcap=self.softcap):
            out[st.row].append((int(st.token_id), float(st.logprob), bool(st.end)))
        return out


def fresh_f(seed: int = 0) -> Box:
    tok = asi.stand_in_tokenizer()
    return Box(asi.stand_in_runtime(asi.stand_in_hf(tok, torch.float32, seed=seed), tok), "F")


def vec(seed: int, *shape: int, size: float = 0.02) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(shape) * size).astype(np.float32)


def programs_of(keys: Sequence[Optional[str]], edit: Any, controls: Sequence[Any], main_dose: float, layer: int) -> List[Any]:
    """A row's key as its program (spec 4.2)."""
    out: List[Any] = []
    for k in keys:
        if k is None:
            out.append(None)
        elif k.startswith("null:"):
            out.append(we.Null(layer, float(k[5:])))
        elif k.startswith("edit:"):
            out.append(we.Arm(edit, float(k[5:])))
        else:
            out.append(we.Arm(controls[int(k[8:])], main_dose))
    return out


def span_of(item: Any) -> Tuple[List[int], int, torch.Tensor]:
    """(the span positions, the end position, the pieces the span predicts): spec 2.5."""
    n_p, T = len(item.prompt_ids), len(item.prompt_ids) + len(item.answer_ids)
    return list(range(n_p - 1, T - 1)), T - 1, torch.tensor([int(t) for t in item.answer_ids], dtype=torch.long)


def tables(box: Box, row: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    """The full log-softmax of fp32 softcapped logits at `positions` of one row's final states."""
    return F.log_softmax(cc.logits_fp32(row[list(positions)], box.W_U), dim=-1)


def brute(box: Box, H: torch.Tensor, r: int, item: Any, odds: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    """Row r against row 0 of one forward, from the full table alone: KL(base || row), the text's own log-probability
    (the row's minus the base's) and the base's own; with `odds`, the log-odds of the named pieces read off the same
    full table (the row's minus the base's) and the base's."""
    P, end, y = span_of(item)
    L0, Lr = tables(box, H[0], P), tables(box, H[r], P)
    out = {"kl": (L0.exp() * (L0 - Lr)).sum(-1), "own": (Lr.gather(1, y[:, None]) - L0.gather(1, y[:, None]))[:, 0],
           "base_own": L0.gather(1, y[:, None])[:, 0]}
    if odds:
        Q = P if odds.get("at", "end") == "span" else [end]

        def lo(row: int) -> torch.Tensor:
            L = tables(box, H[row], Q)
            return torch.logsumexp(L[:, list(odds["toward"])], -1) - torch.logsumexp(L[:, list(odds["away"])], -1)

        out["base_odds"] = lo(0)
        out["odds"] = lo(r) - out["base_odds"]
    return out


def own_stats(v: torch.Tensor) -> Dict[str, Any]:
    """The spec's <stats> (2.7): torch's median, the largest absolute value, the count above a nat."""
    v = v.detach().float().flatten()
    return {"mean": float(v.mean()), "median": float(v.median()), "max_abs": float(v.abs().max()),
            "above_1": int((v.abs() > 1).sum()), "n": int(v.numel())}


def same_stats(got: Any, want: Dict[str, Any], atol: float = 1e-5) -> bool:
    return (isinstance(got, dict) and set(got) == set(want) and got["n"] == want["n"] and got["above_1"] == want["above_1"]
            and all(close(got[k], want[k], atol) for k in ("mean", "median", "max_abs")))


def near_stats(got: Any, want: Dict[str, Any], rel: float = 1e-4, floor: float = 1e-7) -> bool:
    """`same_stats` with a tolerance that follows the figure (`near`); the count above a nat is left to `same_stats`,
    since a position within rounding of a nat may fall either side."""
    return (isinstance(got, dict) and set(got) == set(want) and got["n"] == want["n"]
            and all(near(got[k], want[k], rel, floor) for k in ("mean", "median", "max_abs")))


QA = [("orange", "Describe the color orange.", " Orange is warm, like a sunset over the sea.", None),
      ("who", "Who are you?", " I am a large language model, trained to help.", None),
      ("fox", "Write whatever you like.", " The quick brown fox jumps over the lazy dog.", "Be brief.")]


def plan_file(folder: Path, name: str, readings: List[Dict[str, Any]], items: Sequence[int] = (0, 1, 2)) -> Path:
    """A plan file in the spec's form (5.2), over the check's own questions."""
    folder.mkdir(parents=True, exist_ok=True)
    plan = {"v": 1, "name": name, "readings": readings,
            "items": {QA[i][0]: {"question": QA[i][1], "answer": QA[i][2], "system": QA[i][3]} for i in items}}
    path = folder / f"{name}.json"
    path.write_text(json.dumps(plan, indent=1))
    return path


def items_of(box: Box, which: Sequence[int] = (0, 1, 2)) -> Dict[str, Any]:
    """The check's own questions as items, by name."""
    return {QA[i][0]: lv.make_item(box.tok, QA[i][1], QA[i][2], system=QA[i][3], name=QA[i][0]) for i in which}


def printed(r: Dict[str, Any]) -> None:
    """Keep what lv.lines gives of a result, for section 15."""
    try:
        PRINTED.extend(lv.lines(r))
    except Exception as exc:  # noqa: BLE001
        check(False, f"lines() of {r.get('name')!r} ({r.get('status')}) stopped at {type(exc).__name__}: {str(exc)[:200]} ({_where(exc)})")


def read_all(h: Any, path: Path, box: Box, **kw: Any) -> List[Dict[str, Any]]:
    """Every reading of a plan file through lv.read, its printed lines kept for section 15."""
    plan = lv.load_plan(path)
    items = lv.plan_items(plan, box.tok)
    out = []
    for reading in plan["readings"]:
        r = lv.read(h, reading, items, base_dir=path.parent, **kw)
        printed(r)
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ 1. the hashes


def check_hashes(Fb: Box) -> None:
    # 2.1's worked values
    check(we.array_sha256([5, 17], "i8") == "5b79bedb09ffd0204f01d88edb5a5b9dfe529e0087e7b95060532599efae6e97"
          and we.array_sha256([0, 0], "u1") == "501e17c6fd6079926903d7736534036cb5d6bfa666f66c75b607bc37a1c9fcdc"
          and we.array_sha256([0.0, 0.0], "f4") == "9b2769ca9a7eaa039d65fc6da4a9e3d4861b1d6eca3ee535f0bf1f74aacd031a",
          "array_sha256: the spec's three worked values (2.1)")
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    check(we.array_sha256(a, "i8") == sha_arr(a, "i8") != we.array_sha256(a.reshape(4, 3), "i8"),
          "array_sha256: an array of another dtype is cast, and the shape is hashed with the bytes")
    obj = {"b": [1, 2.5, None, True, "caf\u00e9"], "a": {"z": 0.1, "y": "x"}, "n": -3}
    check(we.canonical(obj) == canon(obj) == lb.canonical(obj), "we.canonical and lb.canonical give the spec's bytes")
    check(words_of(lambda: we.canonical({"x": float("nan")})) is not None
          and words_of(lambda: lb.canonical({"x": float("inf")})) is not None, "canonical JSON refuses a number that is not finite")
    # 2.2's worked value, through the module's own canonical and array hash
    base = {"model": "stand-in", "weights_id": None, "tensors": {"layers.3.mlp.down_proj": "ab" * 32}}
    core = {"format": "theodolite-edit", "version": 1, "base": base, "terms": [{
        "linear": "layers.3.mlp.down_proj", "kind": "cols", "count": 2, "columns": we.array_sha256([5, 17], "i8"),
        "ops": we.array_sha256([0, 0], "u1"), "factors": we.array_sha256([0.0, 0.0], "f4"), "vectors": None}]}
    check(hashlib.sha256(we.canonical(core)).hexdigest() == "f31e6cce0458eed20f50819b0086ae577834bc53ae4e0a063dc9885c25c744cb",
          "the edit id: the spec's worked value (2.2)")
    check(we.FORMAT == "theodolite-edit" and we.VERSION == 1 and we.linear_name(3) == "layers.3.mlp.down_proj"
          and we.linear_name(3, "up_proj") == "layers.3.mlp.up_proj" and tuple(we.linear_shape(Fb.mod(3))) == (D, N)
          and tuple(we.linear_shape(Fb.mod(3, "gate_proj"))) == (N, D), "the format's name, a linear's name and its shape (out, in)")
    with part("1: a built edit"):
        e = Fb.build(we.silence(3, [5, 17]))
        t = e.terms[0]
        check(e.id == own_edit_id(e) and len(e.id) == 64 and e.short() == e.id[:12], "a built edit's id is the spec's hash of its core")
        check(we.tensor_pin(Fb.mod(3)) == own_pin(Fb.mod(3)), "tensor_pin of a plain linear: the spec's prefix and the weight's raw bytes")
        check(e.base == {"model": Fb.model_id, "weights_id": None, "tensors": {"layers.3.mlp.down_proj": own_pin(Fb.mod(3))}},
              "the edit's base: the model id, the weights id and one pin for each linear it touches")
        check((t.linear, t.kind, t.layer, t.which) == ("layers.3.mlp.down_proj", "cols", 3, "down_proj")
              and t.idx.dtype == np.int64 and t.op.dtype == np.uint8 and t.factor.dtype == np.float32
              and t.idx.tolist() == [5, 17] and t.op.tolist() == [0, 0] and t.factor.tolist() == [0.0, 0.0] and t.vectors is None,
              "a silencing's term: columns ascending, op 0, factor 0, no vectors, in the spec's dtypes")
        check(all(not arr.flags.writeable and arr.flags.c_contiguous for arr in (t.idx, t.op, t.factor)),
              "a term's arrays are C-contiguous and read-only")
        same = Fb.build(we.silence(3, [5, 17]))
        raised = words_of(lambda: bool(e == same), Exception)
        check(raised is None and e != same and e == e and same.id == e.id and isinstance(hash(e), int),
              "two equal edits are not == (identity) and their ids are")
        check(Fb.build(we.silence(3, [17, 5])).id == e.id, "neurons listed in another order: the same id")
        check(Fb.build(we.silence(3, [5]), we.scale(3, [17], 0.0)).id == e.id,
              "two recipe terms on one linear are merged: the same id as one term")
        check(Fb.build(we.silence(3, [5, 17]), note="another note").id == e.id, "the note does not change the id")
        check(Fb.build(we.scale(3, [5, 17], 0.5)).id != e.id, "another factor: another id")
        v0 = vec(5, 1, D)
        v0[0, 3] = 0.0
        vm = v0.copy()
        vm[0, 3] = -0.0
        check(Fb.build(we.scale(3, [5, 17], -0.0)).id == e.id and Fb.build(we.scale(3, [5, 17], [0.0, -0.0])).terms[0].factor.tobytes() == t.factor.tobytes(),
              "a scale by -0.0 is the silencing: one arithmetic, one id (the factor is kept as 0.0)")
        check(np.signbit(vm[0, 3]) and Fb.build(we.add_to_columns(3, [5], vm)).id == Fb.build(we.add_to_columns(3, [5], v0)).id
              and Fb.build(we.lowrank(3, -0.0 * np.ones((1, N), dtype=np.float32), vec(6, D, 1))).id == Fb.build(we.lowrank(3, np.zeros((1, N), dtype=np.float32), vec(6, D, 1))).id,
              "a -0.0 in an added vector, or in a low-rank term's A, is 0.0: the same id")
        check(we.build(Fb.lm, [we.silence(3, [5, 17])], model_id="another model", weights_id=None).id != e.id
              and we.build(Fb.lm, [we.silence(3, [5, 17])], model_id=Fb.model_id, weights_id="ab" * 32).id != e.id,
              "another model id, or another weights id: another id")
    with part("1: terms in another order, another base tensor"):
        two = Fb.build(we.scale(2, [4], 1.5), we.silence(3, [17, 5]))
        check(Fb.build(we.silence(3, [5, 17]), we.scale(2, [4], 1.5)).id == two.id, "recipe terms in another order: the same id")
        check([t.linear for t in two.terms] == ["layers.2.mlp.down_proj", "layers.3.mlp.down_proj"]
              and two.layers() == [2, 3] and two.lowest_layer() == 2 and not two.has_lowrank() and not two.has_vectors()
              and two.linears() == ["layers.2.mlp.down_proj", "layers.3.mlp.down_proj"],
              "an edit's terms are sorted by linear; layers(), lowest_layer(), linears(), has_lowrank(), has_vectors()")
        other = fresh_f(seed=1)
        check(other.build(we.silence(3, [5, 17])).id != Fb.build(we.silence(3, [5, 17])).id,
              "the same recipe on a model of other weights (another base tensor): another id")
    # 2.5's and 2.8's worked values
    check(lv.item_from_ids([2, 105, 9], [7, 8]).id == "74ccaa4e58278a91b6f11caefa1895e5fb8906f51d5ec3ee84f8f90b00330210",
          "the item id: the spec's worked value (2.5)")
    e0 = {"v": 1, "seq": 0, "at": "2026-11-14T22:00:00Z", "kind": "note", "prev": "0" * 64, "body": {"says": "first line"}}
    h0 = "81318d96758c7187ba3ed1ad33e101f36e8d983aaae04190ebb680ae6d07317a"
    e1 = {"v": 1, "seq": 1, "at": "2026-11-14T22:00:01Z", "kind": "note", "prev": h0, "body": {"x": 0.1, "n": None}}
    check(lb.entry_hash(e0) == h0 and lb.entry_hash(e1) == "75969610dd2dc42dd9d49f7ca703b72dbaf010f705085e4e158fe82e9a1ba5c1",
          "the ledger's entry hash: the spec's two worked values (2.8)")
    check(lb.entry_hash({**e0, "hash": "f" * 64}) == h0, "entry_hash leaves out a \"hash\" key it is given")


# -------------------------------------------------------------------------------------------------- 2. the gate


def check_gate(Fb: Box, Pb: Box) -> None:
    lm = Fb.lm
    files = TMP / "gate_files"
    files.mkdir(parents=True, exist_ok=True)
    census0, saved0 = we.hook_census(lm), sorted(p.name for p in Path(os.environ[EDITS_VAR]).glob("*"))

    def gate(make: Callable[[], Any]) -> Optional[str]:
        """What the gate says of the term or terms `make` gives (made inside, so a builder that refuses early counts)."""
        def go() -> None:
            specs = make()
            Fb.build(*(specs if isinstance(specs, (list, tuple)) else [specs]))
        return words_of(go, we.Refused)

    def recipe(term: Dict[str, Any]) -> Optional[str]:
        def go() -> None:
            specs = we.from_recipe({"terms": [term]}, files)
            Fb.build(*specs)
        return words_of(go, we.Refused)

    col5 = gw.dense_cols(Fb.mod(3), torch.tensor([5]))[0].float().numpy()
    A2, B2 = vec(1, 2, N), vec(2, D, 2, size=0.5)
    row_vec = np.zeros((1, D), dtype=np.float32)
    check(issubclass(we.Refused, ValueError) and issubclass(we.NotBase, RuntimeError), "Refused is a ValueError and NotBase a RuntimeError")
    # the caps the spec ships with (2.2) [provisional guards]
    check((we.MAX_TERMS, we.MAX_COLUMNS, we.MAX_VECTOR_COLUMNS, we.MAX_RANK, we.MAX_FACTOR, we.MAX_REL_CHANGE,
           we.MAX_PAYLOAD_BYTES, we.MAX_DOSE, we.EDITS_ENV) == (16, 4096, 1024, 64, 64.0, 64.0, 256 * 2**20, 4.0, EDITS_VAR),
          "the caps of 2.2 as the spec ships them")
    with part("2: the table's rows, in order"):
        check(gate(lambda: we.rows(3, "gate_proj", [1], row_vec)) is None, "a row edit on gate_proj is accepted in 1b")
        says(gate(lambda: we.rows(3, "down_proj", [1], row_vec)),
             "A row edit changes what a neuron reads, so it names gate_proj or up_proj; layers.3.mlp.down_proj is not one.",
             "a row edit on down_proj is refused")
        check(recipe({"layer": 3, "remove_direction": [1]}) is not None, "a recipe term with \"remove_direction\" missing its direction file is refused")
        got = gate(lambda: we.lowrank(3, A2, B2, which="self_attn.q_proj"))
        says(got, S_ATTENTION, "an attention linear is refused as 1b's")
        check(got is not None and "q_proj" in got, "the attention refusal names the linear")
        got = gate(lambda: we.lowrank(3, A2, B2, which="pre_feedforward_layernorm"))
        says(got, S_NOT_FFN, "a name that is no feed-forward linear is refused")
        says(gate(lambda: we.silence(NL, [1])), f"Layer {NL} does not exist: layers count from 0 to {NL - 1} here (page layers 1 to {NL}).",
             "a layer past the last is refused")
        says(gate(lambda: we.silence(-1, [1])), f"Layer -1 does not exist: layers count from 0 to {NL - 1} here (page layers 1 to {NL}).",
             "a layer below 0 is refused")
        # the spec names no public way to a column term on gate_proj but a recipe's "which"; that is the way taken here
        says(recipe({"layer": 3, "silence": [1], "which": "gate_proj"}),
             "A column edit changes what a neuron writes, so it names a down_proj; layers.3.mlp.gate_proj is not one.",
             "a column term on gate_proj is refused (through a recipe's \"which\")")
        says(gate(lambda: we.silence(3, [N])), f"Neuron {N} does not exist: layer 3 (page layer 4) has {N} (0 to {N - 1}).",
             "a neuron past the last is refused")
        says(gate(lambda: we.silence(3, [5, 5])), "Neuron 5 of layer 3 (page layer 4) is named twice in this edit: one change a column.",
             "a neuron named twice in one term is refused")
        says(gate(lambda: [we.silence(3, [5]), we.scale(3, [5], 2.0)]),
             "Neuron 5 of layer 3 (page layer 4) is named twice in this edit: one change a column.",
             "a neuron named in two terms of one linear is refused")
        says(gate(lambda: we.silence(3, [])), S_NO_COLUMN, "a term with no neuron is refused")
        says(gate(lambda: we.scale(3, [5], float("nan"))), S_NOT_FINITE, "a factor that is not a number is refused")
        says(gate(lambda: we.set_columns(3, [5], np.full((1, D), np.inf, dtype=np.float32))), S_NOT_FINITE, "a vector holding infinity is refused")
        says(gate(lambda: we.scale(3, [5], 65.0)), "A column is scaled by at most 64 either way here, not 65.", "a factor over the cap is refused")
        says(gate(lambda: we.scale(3, [5], -100.0)), "A column is scaled by at most 64 either way here, not -100.", "a factor under the cap's mirror is refused")
        says(gate(lambda: we.set_columns(3, [5], np.zeros((1, D - 1), dtype=np.float32))),
             f"A column of layer 3's down_proj is {D} numbers long; the vector given for neuron 5 has {D - 1}.",
             "a vector of the wrong length is refused")
        says(gate(lambda: we.add_to_columns(3, [5], (100.0 * col5)[None, :])),
             "Neuron 5 of layer 3 (page layer 4) would change by 100 times its own length; 1a takes at most 64.",
             "a column changed by 100 times its own length is refused")
        says(gate(lambda: we.scale(3, list(range(N)), 2.0)),
             "Scaling every column of layer 3's down_proj by 2 changes nothing: the norm after the block undoes a uniform positive scale.",
             "every column of a down_proj scaled by one positive factor is refused")
        check(gate(lambda: we.scale(3, list(range(N)), -2.0)) is None and gate(lambda: we.scale(3, list(range(N - 1)), 2.0)) is None,
              "a uniform negative scale, and a positive scale of all columns but one, are not that refusal")
        says(gate(lambda: we.lowrank(3, vec(1, 2, N - 1), B2)),
             f"A low-rank term on layers.3.mlp.down_proj needs A [rank, {N}] and B [{D}, rank]; it was given A {W} and B {W}.",
             "a low-rank term whose A is of the wrong width is refused")
        says(gate(lambda: we.lowrank(3, A2, vec(2, D, 3))),
             f"A low-rank term on layers.3.mlp.down_proj needs A [rank, {N}] and B [{D}, rank]; it was given A {W} and B {W}.",
             "a low-rank term whose A and B disagree on the rank is refused")
        says(gate(lambda: we.lowrank(3, vec(1, 65, N), vec(2, D, 65))), "A low-rank term has rank at most 64 here, not 65.",
             "a low-rank term of rank 65 is refused")
        big_a, big_b = vec(3, 1, N, size=1.0), vec(4, D, 1, size=1.0)
        w_len = float(gw.dense(Fb.mod(3)).float().norm())
        big_b = big_b * np.float32(200.3 * w_len / (float(np.linalg.norm(big_a)) * float(np.linalg.norm(big_b))))
        says(gate(lambda: we.lowrank(3, big_a, big_b)),
             "A low-rank term on layers.3.mlp.down_proj could change it by up to 200 times its own length (the lengths of "
             "B and A over the weight's); 1a takes at most 64.", "a low-rank term whose bound is 200 is refused")
    with part("2: low-rank terms off gate_proj and up_proj (decision 1 of the coordinating session)"):
        for which in ("gate_proj", "up_proj"):
            says(gate(lambda: we.lowrank(3, vec(5, 2, D), vec(6, N, 2), which=which)), s_lowrank_which(3, which),
                 f"a low-rank term on {which} is refused in the words the spec's \"Found in the build\" fixes")
            # its place in the order: after a layer out of range and a column term on gate_proj, before the low-rank shapes
            says(gate(lambda: we.lowrank(NL + 1, vec(5, 2, D), vec(6, N, 2), which=which)),
                 f"Layer {NL + 1} does not exist: layers count from 0 to {NL - 1} here (page layers 1 to {NL}).",
                 f"a low-rank term on {which} of a layer that does not exist: the layer's words")
            says(gate(lambda: we.lowrank(3, vec(5, 2, D + 1), vec(6, N, 3), which=which)), s_lowrank_which(3, which),
                 f"a low-rank term on {which} whose shapes are wrong too: the low-rank refusal's words, not the shapes'")
    with part("2: the caps (the module's constants lowered for the length of a call)"):
        kept = {k: getattr(we, k) for k in ("MAX_TERMS", "MAX_COLUMNS", "MAX_VECTOR_COLUMNS", "MAX_PAYLOAD_BYTES")}
        try:
            we.MAX_TERMS = 1
            says(gate(lambda: [we.silence(2, [1]), we.silence(3, [1])]), "An edit takes at most 1 terms in 1a, not 2.", "more terms than the cap")
            we.MAX_TERMS = kept["MAX_TERMS"]
            we.MAX_COLUMNS = 2
            says(gate(lambda: we.silence(3, [1, 2, 3])), "An edit takes at most 2 columns a term in 1a, not 3.", "more columns than the cap")
            we.MAX_COLUMNS = kept["MAX_COLUMNS"]
            we.MAX_VECTOR_COLUMNS = 1
            says(gate(lambda: we.set_columns(3, [1, 2], vec(7, 2, D))), "An edit takes at most 1 set or added columns a term in 1a, not 2.",
                 "more set or added columns than the cap")
            we.MAX_VECTOR_COLUMNS = kept["MAX_VECTOR_COLUMNS"]
            we.MAX_PAYLOAD_BYTES = 1024
            got = gate(lambda: we.set_columns(3, list(range(8)), vec(8, 8, D)))
            check(got is not None and re.fullmatch(r"This edit carries \d+ MiB of arrays; 1a takes at most \d+\.", got) is not None,
                  f"more bytes of arrays than the cap: got {got!r}")
        finally:
            for k, v in kept.items():
                setattr(we, k, v)
    with part("2: two rules broken, the earlier row's words"):
        says(gate(lambda: we.rows(3, "self_attn.q_proj", [1], row_vec)), S_ATTENTION, "a row edit on an attention linear: the attention refusal's words")
        says(gate(lambda: we.silence(NL + 3, [5, 5])), f"Layer {NL + 3} does not exist: layers count from 0 to {NL - 1} here (page layers 1 to {NL}).",
             "a layer that does not exist and a neuron named twice: the layer's words")
        says(gate(lambda: we.scale(3, [N + 372], 100.0)), f"Neuron {N + 372} does not exist: layer 3 (page layer 4) has {N} (0 to {N - 1}).",
             "a neuron that does not exist and a factor over the cap: the neuron's words")
        says(gate(lambda: we.scale(3, [5, 5], 100.0)), "Neuron 5 of layer 3 (page layer 4) is named twice in this edit: one change a column.",
             "a neuron named twice and a factor over the cap: the neuron's words")
        says(gate(lambda: we.scale(3, [5, 6], [float("inf"), 100.0])), S_NOT_FINITE, "a factor that is not finite beside one over the cap: not finite")
        says(gate(lambda: we.lowrank(3, vec(1, 65, N - 1), vec(2, D, 65))),
             f"A low-rank term on layers.3.mlp.down_proj needs A [rank, {N}] and B [{D}, rank]; it was given A {W} and B {W}.",
             "a low-rank term of the wrong width and over the rank cap: the shapes' words")
    with part("2: a linear that cannot be pinned"):
        mod = Pb.mod(1)  # not pinned before this: the spec caches pins by module
        mod.int4mm = True
        try:
            says(words_of(lambda: Pb.build(we.silence(1, [3])), we.Refused),
                 "An edit is pinned to the W4A16 checkpoint's packed tensors, or to a stand-in's plain ones. "
                 "layers.1.mlp.down_proj is w4a16 int4mm: another object, not taken in 1a.",
                 "a packed linear of P marked int4mm is refused as w4a16 int4mm")
            check(words_of(lambda: we.tensor_pin(mod), we.Refused) is not None, "tensor_pin refuses the int4mm linear itself")
        finally:
            del mod.int4mm
        blocks = torch.zeros(4, 8 * 4, dtype=torch.uint8)
        got = words_of(lambda: we.tensor_pin(gw.GGUFLinear(blocks, "F32", 8, 4)), we.Refused)
        check(got is not None and "gguf:F32" in got, f"tensor_pin refuses a GGUF's layer, naming what it is: got {got!r}")
    with part("2: a recipe's files"):
        np.save(files / "ok.npy", vec(9, 1, D))
        check(recipe({"layer": 3, "set": [5], "vectors": "ok.npy"}) is None, "a float32 [K, out] file relative to the plan's folder is taken")
        marker = files / "the_pickle_ran"

        class Planted:
            def __reduce__(self) -> Any:
                return (Path.touch, (marker,))

        (files / "pickle.npy").write_bytes(pickle.dumps(Planted()))
        np.save(files / "object.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
        for name in ("pickle.npy", "object.npy"):
            got = words_of(lambda: we.from_recipe({"terms": [{"layer": 3, "set": [5], "vectors": name}]}, files), ValueError)
            check(got is not None and not got.startswith("<") and not marker.exists(), f"a recipe file that is a pickle ({name}) is refused before it is used: {got!r}")
        np.save(files / "f64.npy", np.zeros((1, D), dtype=np.float64))
        np.save(files / "two_rows.npy", np.zeros((2, D), dtype=np.float32))
        says(recipe({"layer": 3, "set": [5], "vectors": "f64.npy"}), f"The file {W}f64.npy is float64 {W}; this term takes float32 {W}.",
             "a recipe file of another dtype is refused")
        says(recipe({"layer": 3, "add": [5], "vectors": "two_rows.npy"}), f"The file {W}two_rows.npy is float32 {W}; this term takes float32 {W}.",
             "a recipe file of another shape is refused")
        true = cz.file_sha256(files / "ok.npy")
        says(recipe({"layer": 3, "set": [5], "vectors": "ok.npy", "sha256": "0" * 64}),
             f"The file {W}ok.npy hashes to {true[:12]}, not the 000000000000 the plan names.", "a recipe file whose sha256 is not the plan's is refused")
        check(recipe({"layer": 3, "set": [5], "vectors": "ok.npy", "sha256": true}) is None, "a recipe file whose sha256 is the plan's is taken")
        says(recipe({"layer": 3, "silence": [5], "bogus": 1}), "An edit term has a key 1a does not know: bogus.", "a recipe key 1a does not know is refused")
        # a term's file is named from the plan's folder and lies inside it: not any file the process can read
        outside = TMP / "outside_the_plan"
        outside.mkdir(parents=True, exist_ok=True)
        np.save(outside / "vec.npy", vec(9, 1, D))
        (outside / "not_an_array.txt").write_text("a file of someone else's")
        for label, name in (("climbs out of the plan's folder", "../outside_the_plan/vec.npy"), ("is absolute", str(outside / "vec.npy")),
                            ("climbs out and back by another way", "sub/../../outside_the_plan/vec.npy")):
            says(recipe({"layer": 3, "set": [5], "vectors": name}), f"A term's file is named from the plan file's folder and lies inside it; {name} does not.",
                 f"a recipe file whose path {label} is refused")
        got = recipe({"layer": 3, "set": [5], "vectors": "../outside_the_plan/not_an_array.txt", "sha256": "0" * 64})
        check(got is not None and cz.file_sha256(outside / "not_an_array.txt")[:12] not in got, f"and a wrong sha256 does not print the hash of a file outside the plan's folder ({got!r})")
        got = recipe({"layer": 3, "lowrank": {"A": str(outside / "vec.npy"), "B": "ok.npy"}})
        check(got is not None and "lies inside it" in got, f"a low-rank term's files are held to the plan's folder too ({got!r})")
        got = words_of(lambda: we.recipe_facts({"terms": [{"layer": 3, "set": [5], "vectors": "../outside_the_plan/vec.npy"}]}, files), we.Refused)
        check(got is not None and "lies inside it" in got, "and recipe_facts, which opens no file, says so too: the estimate refuses what the reading will")
        (files / "sub").mkdir(exist_ok=True)
        np.save(files / "sub" / "deeper.npy", vec(9, 1, D))
        check(recipe({"layer": 3, "set": [5], "vectors": "sub/deeper.npy"}) is None and recipe({"layer": 3, "set": [5], "vectors": "sub/../ok.npy"}) is None,
              "a file in a folder under the plan's is taken")
        got = recipe({"layer": 3, "scale": [5], "factor": 10 ** 400})
        check(got is not None and not got.startswith("<") and "401 digits" in got, f"a factor past what a float holds is refused in words, not raised as an overflow ({got!r})")
        got = recipe({"layer": 3, "set": [5], "vectors": "a\x00b.npy"})
        check(got is not None and not got.startswith("<"), f"a file name the system will not take (a NUL in it) is refused in words ({got!r})")
        specs = we.from_recipe({"terms": [{"layer": 3, "scale": [5, 17], "factors": [0.5, 2.0]}, {"layer": 2, "silence": [4]}], "note": "n"}, files)
        e = Fb.build(*specs)
        check(e.id == Fb.build(we.silence(2, [4]), we.scale(3, [17, 5], [2.0, 0.5])).id, "a recipe's terms build the edit the builders build")
        check(we.from_recipe({"saved": "abc123"}, files) == "abc123", "a recipe that names a saved edit gives back the name")
    check(we.hook_census(lm) == census0 and we.overlays_in_force(lm) == [] and words_of(lambda: we.assert_base(lm), we.NotBase) is None
          and sorted(p.name for p in Path(os.environ[EDITS_VAR]).glob("*")) == saved0,
          "after every refusal: no hook, no overlay, nothing saved")


# --------------------------------------------------------------------------------------------- 3. save and load


def check_save_load(Fb: Box, Pb: Box) -> None:
    edits = Path(os.environ[EDITS_VAR])
    check(we.edits_dir() == edits and ROOT_DIR not in edits.parents, "edits_dir() is the folder the variable names, outside the repository")
    with part("3: where saved edits go with the variable unset"):
        kept = {k: os.environ.pop(k, None) for k in (EDITS_VAR, "XDG_DATA_HOME")}
        try:
            home_default = we.edits_dir()
            os.environ["XDG_DATA_HOME"] = str(TMP / "xdg")
            xdg_default = we.edits_dir()
        finally:
            for k, v in kept.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        check(home_default == Path.home() / ".local" / "share" / "theodolite" / "edits" and xdg_default == TMP / "xdg" / "theodolite" / "edits"
              and ROOT_DIR not in home_default.parents, "the default folder is beside the ledger under the data home, outside the repository")

    def same_arrays(a: Any, b: Any) -> bool:
        if len(a.terms) != len(b.terms):
            return False
        for s, t in zip(a.terms, b.terms):
            if (s.linear, s.kind) != (t.linear, t.kind):
                return False
            for name in ("idx", "op", "factor", "vectors", "A", "B"):
                x, y = getattr(s, name, None), getattr(t, name, None)
                if (x is None) != (y is None) or (x is not None and not (x.dtype == y.dtype and np.array_equal(x, y))):
                    return False
        return True

    def load(box: Box, where: Any, **kw: Any) -> Any:
        return we.load(where, box.lm, model_id=kw.get("model_id", box.model_id), weights_id=kw.get("weights_id"))

    for box in (Fb, Pb):
        with part(f"3 ({box.label}): a carried edit saved and loaded"):
            e = box.build(we.set_columns(2, [40, 3], vec(11, 2, D)), we.scale(2, [77], 1.5), we.lowrank(3, vec(12, 2, N), vec(13, D, 2, size=0.5)),
                          note="saved by the check")
            folder = we.save(e)
            man = json.loads((folder / "manifest.json").read_text())
            check(folder == edits / e.id and ROOT_DIR not in folder.parents and sorted(p.name for p in folder.iterdir()) == ["manifest.json", "payload.safetensors"],
                  f"{box.label}: save writes manifest.json and payload.safetensors under edits_dir() / id")
            check(set(man) == {"format", "version", "id", "base", "terms", "note", "report", "saved_at"} and man["format"] == "theodolite-edit"
                  and man["version"] == 1 and man["id"] == e.id and man["note"] == "saved by the check"
                  and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", str(man["saved_at"])) is not None, f"{box.label}: the manifest's keys (2.3)")
            check({k: v for k, v in man.items() if k != "saved_at"} == json.loads(json.dumps(e.manifest())), f"{box.label}: Edit.manifest() is the saved manifest without saved_at")
            cols, low = man["terms"][0], man["terms"][1]
            check(set(cols) == {"linear", "kind", "count", "columns", "ops", "factors", "vectors", "sha256", "carried"} and cols["kind"] == "cols"
                  and cols["columns"] == [3, 40, 77] and cols["ops"] == ["set", "set", "scale"] and cols["factors"] == [1.0, 1.0, 1.5]
                  and cols["vectors"] == "t0.vectors" and cols["carried"] is True and set(cols["sha256"]) == {"columns", "ops", "factors", "vectors"},
                  f"{box.label}: a column term in the manifest: columns ascending, ops by name, the vectors' key, carried")
            check(set(low) == {"linear", "kind", "rank", "A", "B", "sha256", "carried"} and (low["A"], low["B"], low["rank"], low["carried"]) == ("t1.A", "t1.B", 2, True),
                  f"{box.label}: a low-rank term in the manifest")
            from safetensors.numpy import load_file

            payload = load_file(str(folder / "payload.safetensors"))
            check(set(payload) == {"t0.vectors", "t1.A", "t1.B"} and all(v.dtype == np.float32 for v in payload.values()),
                  f"{box.label}: the payload holds the keys the manifest names, float32")
            t0 = e.terms[0]
            check(np.array_equal(t0.vectors[0], vec(11, 2, D)[1]) and np.array_equal(t0.vectors[1], vec(11, 2, D)[0]) and not t0.vectors[2].any(),
                  f"{box.label}: a set column keeps its own vector when the neurons are sorted, and a scaled column's row is zeros")
            back = load(box, folder)
            check(back.id == e.id and same_arrays(back, e) and back.note == e.note, f"{box.label}: a saved edit loads to the same id and the same arrays")
            check(load(box, e.id).id == e.id and load(box, e.id[:12]).id == e.id, f"{box.label}: it loads by its id and by the id's first 12")
            check(e.base["tensors"] == {name: own_pin(box.mod(int(name.split(".")[1]))) for name in e.linears()},
                  f"{box.label}: the pins are the spec's hash of each touched linear" + (" (packed words and scales)" if box is Pb else ""))
            says(words_of(lambda: load(box, folder, model_id="another model"), we.Refused),
                 f"This edit was made on {box.model_id} (weights None); the loaded model is another model (weights None). "
                 "An edit is a difference from one base and is not carried to another.", f"{box.label}: loading on another model id is refused")
            says(words_of(lambda: load(box, folder, weights_id="cd" * 32), we.Refused),
                 f"This edit was made on {box.model_id} (weights None); the loaded model is {box.model_id} (weights {'cd' * 32}). "
                 "An edit is a difference from one base and is not carried to another.", f"{box.label}: loading on other weights' id is refused")
            # a byte changed in the payload: the last byte of the file is the last array's
            twin = TMP / f"tampered_{box.label}" / "payload"
            shutil.copytree(folder, twin)
            raw = bytearray((twin / "payload.safetensors").read_bytes())
            raw[-1] ^= 0x01
            (twin / "payload.safetensors").write_bytes(bytes(raw))
            says(words_of(lambda: load(box, twin), we.Refused), f"The saved edit {e.id[:12]}: its {W} no longer hashes to what its manifest says.",
                 f"{box.label}: a byte changed in the payload is refused")
            # a factor changed in the manifest
            twin = TMP / f"tampered_{box.label}" / "factor"
            shutil.copytree(folder, twin)
            man2 = json.loads((twin / "manifest.json").read_text())
            man2["terms"][0]["factors"][2] = 1.25
            (twin / "manifest.json").write_text(json.dumps(man2))
            says(words_of(lambda: load(box, twin), we.Refused), f"The saved edit {e.id[:12]}: its {W} no longer hashes to what its manifest says.",
                 f"{box.label}: a factor changed in the manifest is refused")
            # the id changed in the manifest
            twin = TMP / f"tampered_{box.label}" / "id"
            shutil.copytree(folder, twin)
            man2 = json.loads((twin / "manifest.json").read_text())
            man2["id"] = "0" * 64
            (twin / "manifest.json").write_text(json.dumps(man2))
            says(words_of(lambda: load(box, twin), we.Refused), f"The saved edit says it is 000000000000 and hashes to {e.id[:12]}.",
                 f"{box.label}: an id that is not the manifest's own hash is refused")
            twin = TMP / f"tampered_{box.label}" / "format"
            shutil.copytree(folder, twin)
            (twin / "manifest.json").write_text(json.dumps({**man, "version": 2}))
            says(words_of(lambda: load(box, twin), we.Refused), f"{W} is not a saved edit of format theodolite-edit, version 1.",
                 f"{box.label}: a manifest of another version is refused")
            other = TMP / f"tampered_{box.label}" / "elsewhere"
            check(we.save(e, other) == other and load(box, other).id == e.id, f"{box.label}: save takes a folder of the caller's")
        with part(f"3 ({box.label}): a scale-only edit"):
            s = box.build(we.scale(2, [3, 40], [0.0, 2.0]))
            folder = we.save(s)
            man = json.loads((folder / "manifest.json").read_text())
            check(not (folder / "payload.safetensors").exists() and man["terms"][0]["carried"] is False and man["terms"][0]["vectors"] is None,
                  f"{box.label}: a scale-only edit writes no payload and is marked not carried")
            check(load(box, folder).id == s.id and same_arrays(load(box, folder), s), f"{box.label}: it is rebuilt from the manifest alone")
    with part("3 (F): a stand-in of other weights"):
        e = Fb.build(we.scale(2, [3, 40], [0.0, 2.0]))
        folder = we.save(e)
        tok = asi.stand_in_tokenizer()
        other = Box(asi.stand_in_runtime(asi.stand_in_hf(tok, torch.float32, seed=1), tok), "F of other weights")
        name = "layers.2.mlp.down_proj"
        says(words_of(lambda: we.load(folder, other.lm, model_id=other.model_id, weights_id=None), we.Refused),
             f"This edit was made against other weights: {name} then hashed to {e.base['tensors'][name][:12]}, the loaded model's hashes to "
             f"{own_pin(other.mod(2))[:12]}.", "an edit loaded on a stand-in of other weights is refused, naming the tensor and both hashes")
    with part("3 (F): the control mark in a saved report lifts no cap but a column's own change"):
        # the report is not hashed, so the mark can be written into any saved manifest: it must not switch the caps off
        five = Fb.build(we.silence(2, [1, 2, 3, 4, 5]))
        two_rank = Fb.build(we.lowrank(3, vec(14, 2, N), vec(15, D, 2, size=0.5)))
        marked = {}
        for label, e in (("columns", five), ("rank", two_rank)):
            folder = we.save(e, TMP / "marked" / label)
            man = json.loads((folder / "manifest.json").read_text())
            man["report"] = {**(man.get("report") or {}), "caps": "not applied to a control edit"}
            (folder / "manifest.json").write_text(json.dumps(man))
            marked[label] = folder
        check(we.load(marked["columns"], Fb.lm, model_id=Fb.model_id, weights_id=None).id == five.id, "3: (with the caps as shipped the marked edit loads: it is within them)")
        kept = {k: getattr(we, k) for k in ("MAX_COLUMNS", "MAX_RANK")}
        try:
            we.MAX_COLUMNS, we.MAX_RANK = 2, 1
            says(words_of(lambda: we.load(marked["columns"], Fb.lm, model_id=Fb.model_id, weights_id=None), we.Refused),
                 "An edit takes at most 2 columns a term in 1a, not 5.", "3: a saved edit over the column cap is refused though its report carries the control mark")
            says(words_of(lambda: we.load(marked["rank"], Fb.lm, model_id=Fb.model_id, weights_id=None), we.Refused),
                 "A low-rank term has rank at most 1 here, not 2.", "3: and one over the rank cap likewise")
            hand = we.Edit(five.id, five.terms, five.base, "", {"caps": "not applied to a control edit"})
            says(words_of(lambda: we.gate_again(Fb.lm, hand, model_id=Fb.model_id, weights_id=None), we.Refused),
                 "An edit takes at most 2 columns a term in 1a, not 5.", "3: an Edit made by hand with the mark in its report meets the same cap at the gate")
        finally:
            for k, v in kept.items():
                setattr(we, k, v)
    with part("3 (F): an edit met again passes the gate again"):
        e = Fb.build(we.set_columns(2, [40, 3], vec(11, 2, D)), we.scale(2, [77], 1.5))
        again = we.gate_again(Fb.lm, e, model_id=Fb.model_id, weights_id=None)
        check(again.id == e.id and same_arrays(again, e) and again is not e, "3: gate_again gives the same edit back, built again from its terms")
        false_id = we.Edit("f" * 64, e.terms, e.base, "", e.report)
        says(words_of(lambda: we.gate_again(Fb.lm, false_id, model_id=Fb.model_id, weights_id=None), we.Refused),
             f"The edit handed in says it is ffffffffffff and hashes to {e.id[:12]}.", "3: an Edit whose id is not the hash of what it holds is refused")
        ctl = we.control_edits(e, Fb.lm, 3, 1)[0]
        check(we.gate_again(Fb.lm, ctl, model_id=Fb.model_id, weights_id=None).id == ctl.id, "3: a control edit passes the gate again under its own id")
        got = words_of(lambda: we.gate_again(Fb.lm, "not an edit", model_id=Fb.model_id, weights_id=None), we.Refused)
        check(got is not None and not got.startswith("<"), f"3: what is no Edit is refused in words ({got!r})")
    with part("3: save refuses a folder inside the checkout"):
        inside = ROOT_DIR / "levelling_check_must_not_be_written"
        try:
            got = words_of(lambda: we.save(Fb.build(we.silence(2, [3])), inside), we.Refused)
            check(got is not None and not got.startswith("<") and "outside the checkout" in got and not inside.exists(),
                  f"3: save into a folder under the repository is refused in words and makes nothing ({got!r})")
            kept_env = os.environ[EDITS_VAR]
            os.environ[EDITS_VAR] = str(inside)
            try:
                got = words_of(lambda: we.save(Fb.build(we.silence(2, [3]))), we.Refused)
            finally:
                os.environ[EDITS_VAR] = kept_env
            check(got is not None and not got.startswith("<") and not inside.exists(), "3: and so is the default folder when the variable points into the repository")
        finally:
            if inside.exists():  # only should the guard have failed: the check takes back what it caused
                shutil.rmtree(inside, ignore_errors=True)
    check(not any(ROOT_DIR in p.resolve().parents for p in edits.rglob("*")), "nothing was saved under the repository")


# -------------------------------------------------------------------------------- 4. the per-row hooks, by bits


def check_bits(Pb: Box) -> None:
    lm, tok = Pb.lm, Pb.tok
    item = lv.make_item(tok, QA[0][1], QA[0][2])
    ids, T = list(item.ids), len(item.ids)
    cols = [3, 40, 77, 100]
    sil = Pb.build(we.silence(2, cols))
    by2 = Pb.build(we.scale(2, cols, 2.0))
    by_half = Pb.build(we.scale(2, cols, -0.5))
    set_e = Pb.build(we.set_columns(2, cols, vec(21, 4, D)))
    add_e = Pb.build(we.add_to_columns(2, cols, vec(22, 4, D)))
    low_e = Pb.build(we.lowrank(2, vec(23, 2, N), vec(24, D, 2, size=0.5)))
    census0 = we.hook_census(lm)
    check(census0 == {}, "P carries no hook at rest")
    for R in (12, 32):
        with part(f"4a ({R} rows)"):
            bare = Pb.whole(ids, R)
            with we.per_row(lm, [None] * R):
                h = Pb.whole(ids, R)
            check(torch.equal(h, bare), f"4a: under per_row with every program None, {R} rows equal an unhooked forward by bits")
            progs = [None, we.Arm(by_half, 0.0), we.Arm(set_e, 0.0), we.Arm(add_e, 0.0), we.Arm(low_e, 0.0), we.Arm(sil, 0.0)] + [None] * (R - 6)
            with we.per_row(lm, progs):
                h = Pb.whole(ids, R)
            check(torch.equal(h, bare), f"4a: arms at dose 0 (a scale, a set, an add, a low-rank term) and a twin, {R} rows: every row the unhooked forward's")
            check(we.hook_census(lm) == census0, f"4a: per_row leaves no hook ({R} rows)")
        with part(f"4b ({R} rows)"):
            for label, e in (("silenced", sil), ("scaled by 2", by2), ("scaled by -0.5", by_half)):
                with we.per_row(lm, [None, we.Arm(e, 1.0)] + [None] * (R - 2)):
                    hp = Pb.whole(ids, R)
                with we.folded(lm, e, 1.0):
                    hf = Pb.whole(ids, R)
                check(torch.equal(hp[1], hf[0]) and not torch.equal(hp[1], hp[0]),
                      f"4b: a set {label}: the arm's row equals row 0 of the folded forward by bits, and is not the base row ({R} rows)")
                check(torch.equal(hp[0], bare[0]) and torch.equal(hf[0], hf[R - 1]), f"4b: {label}: the per-row forward's base row is the unhooked one; the folded rows are alike ({R} rows)")
        with part(f"4c ({R} rows)"):
            progs = [None, we.Arm(sil, 1.0), None, we.Arm(set_e, 1.0), we.Arm(sil, 1.0), we.Null(2, 0.125)] + [None] * (R - 7) + [we.Arm(set_e, 1.0)]
            with we.per_row(lm, progs):
                h = Pb.whole(ids, R)
            check(torch.equal(h[1], h[4]) and torch.equal(h[3], h[R - 1]), f"4c: the same edit in two rows gives the same bits ({R} rows)")
            check(all(torch.equal(h[r], bare[r]) for r in range(R) if progs[r] is None), f"4c: the untouched rows beside them are the base's ({R} rows)")
            check(not torch.equal(h[1], h[0]) and not torch.equal(h[3], h[0]) and not torch.equal(h[5], h[0]), f"4c: the edited rows and the null row are not the base row ({R} rows)")
    for L0 in (0, 2, 5):
        with part(f"4d (layer {L0})"):
            R = 12
            e = Pb.build(we.silence(L0, cols[:2]), we.set_columns(min(L0 + 1, NL - 1), [9], vec(25, 1, D)))
            progs = [None, we.Null(L0, 0.125), we.Null(L0, -0.125), we.Arm(e, 1.0), we.Arm(e, -1.0), we.Arm(e, 0.5)] + [None] * (R - 6)
            kept: Dict[str, torch.Tensor] = {}

            def keep(_m: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
                kept["stream"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

            pos = torch.arange(T).unsqueeze(0).expand(R, T)
            with we.per_row(lm, progs):
                hk = lm.layers[L0].register_forward_pre_hook(keep, with_kwargs=True)
                try:
                    whole = Pb.whole(ids, R)
                finally:
                    hk.remove()
                again = ao.forward_from_layer(lm, L0, kept["stream"], None, pos)
            check(tuple(kept["stream"].shape) == (R, T, D) and torch.equal(again, whole),
                  f"4d: a restart from the kept stream under hooks equals the whole forward under the same hooks, layer {L0}")
            check(not torch.equal(whole[3], whole[0]) and torch.equal(whole[R - 1], whole[0]), f"4d: the arm moved and the twin did not (layer {L0})")
    with part("4e: the folded context"):
        R = 12
        bare = Pb.whole(ids, R)
        dicts = {n: (dict(m._memit_col_overlays), dict(m._memit_row_overlays)) for n, m in lm.named_modules() if hasattr(m, "_memit_col_overlays")}
        check(len(dicts) > 0 and we.overlays_in_force(lm) == [], "4e: the packed linears carry overlay dicts, all empty at rest")
        with we.folded(lm, set_e, 1.0):
            inside = we.overlays_in_force(lm)
            check(inside == [("layers.2.mlp.down_proj", len(cols))], f"4e: inside folded the overlays in force are the edit's columns ({inside})")
            check(words_of(lambda: we.assert_base(lm), we.NotBase) is not None, "4e: inside folded the model is not the base")
        check({n: (dict(m._memit_col_overlays), dict(m._memit_row_overlays)) for n, m in lm.named_modules() if hasattr(m, "_memit_col_overlays")} == dicts,
              "4e: after folded every overlay dict is as it was")
        def writes_elsewhere() -> None:
            with we.folded(lm, set_e, 1.0):  # the body writes overlays of its own on linears the edit does not touch
                Pb.mod(5)._memit_col_overlays[9] = torch.zeros(D, dtype=Pb.dtype)
                Pb.mod(3, "gate_proj")._memit_row_overlays[4] = torch.zeros(D, dtype=Pb.dtype)
                del Pb.mod(2)._memit_col_overlays[cols[0]]

        raised = words_of(writes_elsewhere, Exception)
        as_was = {n: (dict(m._memit_col_overlays), dict(m._memit_row_overlays)) for n, m in lm.named_modules() if hasattr(m, "_memit_col_overlays")} == dicts
        Pb.mod(5)._memit_col_overlays.pop(9, None)  # the check's own, should folded have left them
        Pb.mod(3, "gate_proj")._memit_row_overlays.pop(4, None)
        check(raised is None and as_was, "4e: folded puts back the dicts of EVERY module: overlays the block wrote on other linears, a row overlay among them, are gone"
              + ("" if raised is None else f" ({raised[:160]})"))
        check(torch.equal(Pb.whole(ids, R), bare), "4e: after folded the next forward is the base's by bits")
        check(words_of(lambda: we.assert_base(lm), we.NotBase) is None and words_of(lambda: we.assert_base(lm, census0), we.NotBase) is None,
              "4e: after folded assert_base passes")
    with part("4e: a vault overlay already in force"):
        mod = Pb.mod(4)
        theirs = torch.zeros(D, dtype=Pb.dtype)
        mod._memit_col_overlays[7] = theirs
        try:
            ran: List[int] = []

            def enter() -> None:
                with we.folded(lm, sil, 1.0):
                    ran.append(1)
            got = words_of(enter, we.NotBase)
            check(got is not None and not got.startswith("<"), f"4e: folded raises NotBase when a vault overlay is in force: {got!r}")
            check(not ran, "4e: and raises it on the way in: the block never runs over the vault's edit")
            check(mod._memit_col_overlays.get(7) is theirs and we.overlays_in_force(lm) == [("layers.4.mlp.down_proj", 1)]
                  and not Pb.mod(2)._memit_col_overlays, "4e: and leaves the vault's overlay there, with nothing of its own")
            check(words_of(lambda: we.assert_base(lm), we.NotBase) is not None, "4e: assert_base fails while the vault's overlay is in force")
            check(words_of(lambda: lv.handle(Pb.rt), we.NotBase) is not None, "4e: handle() raises NotBase while an overlay is in force")
        finally:
            mod._memit_col_overlays.pop(7, None)
    with part("4e: an exception inside either context"):
        def inside_per_row() -> None:
            with we.per_row(lm, [None, we.Arm(set_e, 1.0), we.Null(2, 0.125)]):
                raise KeyError("planted by the check")

        def inside_folded() -> None:
            with we.folded(lm, set_e, 1.0):
                raise KeyError("planted by the check")

        check(words_of(inside_per_row, KeyError) is not None and we.hook_census(lm) == census0, "4e: an exception inside per_row leaves no hook")
        check(words_of(inside_folded, KeyError) is not None and we.overlays_in_force(lm) == [] and we.hook_census(lm) == census0,
              "4e: an exception inside folded leaves no overlay and no hook")
        check(words_of(lambda: we.assert_base(lm), we.NotBase) is None, "4e: and the model is the base after both")
    with part("4e: the census knows hooks, not counts"):
        mod = Pb.mod(2)
        first = mod.register_forward_hook(lambda m, a, o: None)
        try:
            census = we.hook_census(lm)
            check(census == {"layers.2.mlp.down_proj": [first.id]}, f"4e: hook_census names the module and the hook's id ({census})")
            check(words_of(lambda: we.assert_base(lm, census), we.NotBase) is None, "4e: assert_base passes against a census taken with the check's own hook in place")
            check(words_of(lambda: we.assert_base(lm), we.NotBase) is not None, "4e: assert_base with no census fails on that hook")
            first.remove()
            second = mod.register_forward_pre_hook(lambda m, a: None)
            try:
                check(words_of(lambda: we.assert_base(lm, census), we.NotBase) is not None,
                      "4e: assert_base fails when one hook was taken off and another put on the same module")
            finally:
                second.remove()
        finally:
            first.remove()
        check(we.hook_census(lm) == census0 and words_of(lambda: we.assert_base(lm), we.NotBase) is None, "4e: with the check's hooks gone the model is the base")


# ----------------------------------------------------------------------------- 5. the per-row hooks, to rounding


def edited_copy(box: Box, changes: Sequence[Tuple[int, str, Callable[[torch.Tensor], None]]]) -> Box:
    """A copy of a float32 stand-in with its weights written into: (layer, which, a function that edits W [out, in])."""
    twin = Box(asi.stand_in_runtime(copy.deepcopy(box.hf), box.tok), "a copy")
    for layer, which, change in changes:
        change(twin.mod(layer, which).weight.data)
    return twin


def check_rounding(Fb: Box, Pb: Box) -> None:
    item = lv.make_item(Fb.tok, QA[0][1], QA[0][2])
    ids, R = list(item.ids), 8
    P, _end, _y = span_of(item)
    cols = [3, 40, 77, 100]
    v_set, v_add = vec(31, 4, D), vec(32, 4, D)
    # each form with the weight it stands for at dose d written into a copy: W + d * Delta (spec 2.2)
    tv_set, tv_add = torch.tensor(v_set).T, torch.tensor(v_add).T
    cases = [("a scale by 1.5", lambda: we.scale(2, cols, 1.5), lambda Wt, d: Wt.__setitem__((slice(None), cols), Wt[:, cols] * (1 + d * 0.5))),
             ("a scale by -3", lambda: we.scale(2, cols, -3.0), lambda Wt, d: Wt.__setitem__((slice(None), cols), Wt[:, cols] * (1 + d * -4.0))),
             ("a set", lambda: we.set_columns(2, cols, v_set), lambda Wt, d: Wt.__setitem__((slice(None), cols), (1 - d) * Wt[:, cols] + d * tv_set)),
             ("an add", lambda: we.add_to_columns(2, cols, v_add), lambda Wt, d: Wt.__setitem__((slice(None), cols), Wt[:, cols] + d * tv_add))]
    # F with the overlay forward on the touched linear, so that a folded form exists there (spec 4.6, 6.1 §5)
    F5 = fresh_f()
    gw.install_overlay_forward(F5.mod(2))
    check(torch.equal(F5.whole(ids, R), Fb.whole(ids, R)), "5: a second float32 stand-in, its layer-2 down_proj given the overlay forward, reads as the first")
    for label, spec, change in cases:
        with part(f"5a (F): {label}"):
            e = F5.build(spec())
            # dose 1 is the spec's line (6.1 §5a); 0.5 and -1 are there so that a dose applied twice, to the content
            # or to the share of the old column that stays, cannot pass: at dose 1 d * d is d
            for dose in (1.0, 0.5, -1.0):
                with we.per_row(F5.lm, [None, we.Arm(e, dose)] + [None] * (R - 2)):
                    hp = F5.whole(ids, R)
                with we.folded(F5.lm, e, dose):
                    hf = F5.whole(ids, R)
                hc = edited_copy(Fb, [(2, "down_proj", lambda Wt, d=dose: change(Wt, d))]).whole(ids, R)
                # atol 1e-5 on final states: the spec's (6.1 §5a)
                check(torch.allclose(hp[1], hf[0], atol=1e-5) and torch.allclose(hp[1], hc[0], atol=1e-5) and torch.allclose(hf[0], hc[0], atol=1e-5)
                      and not torch.allclose(hp[1], hp[0], atol=1e-3),
                      f"5a (F): {label} at dose {dose:g}: the arm's row is the folded form's and the edited copy's, to 1e-5 (largest differences "
                      f"{float((hp[1] - hf[0]).abs().max()):.1e} and {float((hp[1] - hc[0]).abs().max()):.1e})")
        with part(f"5a (P): {label}"):
            e = Pb.build(spec())
            for rows in (12, 32):
                with we.per_row(Pb.lm, [None, we.Arm(e, 1.0)] + [None] * (rows - 2)):
                    hp = Pb.whole(ids, rows)
                with we.folded(Pb.lm, e, 1.0):
                    hf = Pb.whole(ids, rows)
                lf, lp = tables(Pb, hf[0], P), tables(Pb, hp[1], P)
                between = float((lf.exp() * (lf - lp)).sum(-1).mean())
                # under 1e-4 nats: the spec's (6.1 §5a; it read 1.2e-6 to 1.5e-6 on 14 November 2026)
                check(between < 1e-4 and not torch.equal(hp[1], hp[0]), f"5a (P): {label}, {rows} rows: mean KL between the folded row and the per-row row {between:.2e} nats, under 1e-4")
                print(f"        (P) {label}, {rows} rows: largest difference of final states {float((hf[0].float() - hp[1].float()).abs().max()):.3f} "
                      f"where states reach {float(hf[0].float().abs().max()):.2f} (no check: no bound follows from bf16's step)")
    with part("5a (P): the output-side add is one float32 sum rounded once"):
        # 4.1: acc = out[r].float() + a.float() @ C, out2[r] = acc.to(dtype), with C = ((v * d).to(dtype)).float() and a
        # the row's ORIGINAL activation of the set columns. The pilots' hooks multiplied and summed in bf16: another
        # arithmetic (open point 6). The check works the right-hand side itself from what the linear was handed.
        mod, rows, r = Pb.mod(2), 12, 3
        for label, e, v, dose in (("a set", Pb.build(we.set_columns(2, cols, v_set)), v_set, 1.0), ("an add at dose 0.5", Pb.build(we.add_to_columns(2, cols, v_add)), v_add, 0.5)):
            got: Dict[str, torch.Tensor] = {}
            first = [mod.register_forward_pre_hook(lambda m, a: got.__setitem__("x", a[0].detach().clone())),      # before per_row's: the original input
                     mod.register_forward_hook(lambda m, a, o: got.__setitem__("raw", o.detach().clone()))]         # before per_row's: the product as the linear made it
            try:
                with we.per_row(Pb.lm, [None] * r + [we.Arm(e, dose)] + [None] * (rows - r - 1)):
                    last = mod.register_forward_hook(lambda m, a, o: got.__setitem__("out", o.detach().clone()))    # after per_row's: what the row leaves with
                    try:
                        Pb.whole(ids, rows)
                    finally:
                        last.remove()
            finally:
                for hk in first:
                    hk.remove()
            C = (torch.tensor(v) * dose).to(Pb.dtype).float()
            want = got["raw"].clone()
            want[r] = (got["raw"][r].float() + got["x"][r][:, cols].float() @ C).to(Pb.dtype)
            pilots = got["raw"].clone()
            pilots[r] = got["raw"][r] + got["x"][r][:, cols] @ C.to(Pb.dtype)
            check(torch.equal(got["out"], want) and not torch.equal(got["out"][r], got["raw"][r]),
                  f"5a (P): {label}: the edited row leaves the linear as the float32 sum of its product and its new content, rounded once, by bits; the other rows as they came")
            check(not torch.equal(want[r], pilots[r]), f"5a (P): {label}: (and that is not what a product and a sum in bf16 give: the line above tells the two arithmetics apart)")
        check(we.hook_census(Pb.lm) == {}, "5a (P): and no hook is left")
    with part("5b (F): a low-rank term on down_proj"):
        A, B = vec(33, 2, N), vec(34, D, 2, size=0.5)
        e = Fb.build(we.lowrank(2, A, B))
        for dose in (1.0, -0.5):
            with we.per_row(Fb.lm, [None, we.Arm(e, dose)] + [None] * (R - 2)):
                hp = Fb.whole(ids, R)
            hc = edited_copy(Fb, [(2, "down_proj", lambda Wt: Wt.add_(dose * torch.tensor(B) @ torch.tensor(A)))]).whole(ids, R)
            check(torch.allclose(hp[1], hc[0], atol=1e-5) and not torch.allclose(hp[1], hp[0], atol=1e-3),
                  f"5b (F): a low-rank term at dose {dose:g} equals a copy with W + d B A written in, to 1e-5 ({float((hp[1] - hc[0]).abs().max()):.1e})")
        rep = find_key(e.report, "form")
        check(rep == ["an output-side add; no folded form in 1a"], f"5b: a low-rank term's report says its form ({rep})")
        bound = find_key(e.report, "bound")
        want = float(np.linalg.norm(B)) * float(np.linalg.norm(A)) / float(Fb.mod(2).weight.float().norm())
        check(len(bound) == 1 and close(bound[0], want, 1e-4), f"5b: its rel_change is the bound ||B|| ||A|| / ||W|| ({bound} against {want:.5f})")
    with part("5c: doses"):
        sil_f, sil_p = Fb.build(we.silence(2, cols)), Pb.build(we.silence(2, cols))
        with we.per_row(Fb.lm, [None, we.Arm(sil_f, -1.0)] + [None] * (R - 2)):
            hp = Fb.whole(ids, R)
        hc = edited_copy(Fb, [(2, "down_proj", lambda Wt: Wt.__setitem__((slice(None), cols), Wt[:, cols] * 2.0))]).whole(ids, R)
        check(torch.allclose(hp[1], hc[0], atol=1e-5) and not torch.allclose(hp[1], hp[0], atol=1e-3), "5c (F): dose -1 of a silencing doubles the column")
        with we.per_row(Pb.lm, [None, we.Arm(sil_p, -1.0)] + [None] * (R - 2)):
            hp = Pb.whole(ids, R)
        with we.folded(Pb.lm, sil_p, -1.0):
            hf = Pb.whole(ids, R)
        with we.folded(Pb.lm, Pb.build(we.scale(2, cols, 2.0)), 1.0):
            h2 = Pb.whole(ids, R)
        check(torch.equal(hp[1], hf[0]) and torch.equal(hf[0], h2[0]), "5c (P): dose -1 of a silencing is the folded form at dose -1 and a scale by 2, by bits")
        bf, f32 = torch.bfloat16, torch.float32
        x = we.exact_at(sil_p, 1.0, bf)
        check(set(x) == {"input_side", "output_add", "folded_form", "factors"} and x["input_side"] == "exact" and x["output_add"] is False
              and x["folded_form"] is True and [list(map(float, f)) for f in x["factors"]] == [[0.0, 0.0]], f"5c: exact_at reads \"exact\" for a silencing ({x})")
        x = we.exact_at(Pb.build(we.scale(2, cols, 1.5)), 1.0, bf)
        check(x["input_side"] == "rounds" and [list(map(float, f)) for f in x["factors"]] == [[1.5, 1.5]], f"5c: exact_at reads \"rounds\" for a scale by 1.5 ({x})")
        x = we.exact_at(sil_p, 0.001, bf)
        check(x["input_side"] == "gone" and len(x["factors"]) == 1 and close(x["factors"][0][0], 0.999, 1e-12) and float(x["factors"][0][1]) == 1.0,
              f"5c: a silencing at dose 0.001 is \"gone\" in bf16: 0.999 asked, 1 applied ({x})")
        says(we.dose_refusal(sil_p, 0.001, bf),
             f"At dose 0.001 every factor of this edit is exactly 1 in {W}: its row would be the base row again. Doses are dyadic here "
             "(1, 0.5, 0.25, -1); for a smaller effect take fewer columns, not a smaller dose.", "5c: where dose_refusal gives its sentence")
        check(we.exact_at(sil_p, 0.001, f32)["input_side"] == "rounds" and we.dose_refusal(sil_p, 0.001, f32) is None, "5c: and \"rounds\" in float32, with no refusal")
        check(we.exact_at(sil_p, -1.0, bf)["input_side"] == "exact" and we.exact_at(sil_p, 0.5, bf)["input_side"] == "exact"
              and we.exact_at(sil_p, 0.0, bf)["input_side"] == "none", "5c: a silencing at doses -1 and 0.5 is exact (factors 2 and 0.5), and at dose 0 there is no input side")
        x = we.exact_at(Pb.build(we.set_columns(2, cols, v_set)), 1.0, bf)
        check(x["output_add"] is True and x["folded_form"] is True and x["input_side"] == "exact", f"5c: a set uses the output-side add and has a folded form ({x})")
        x = we.exact_at(Pb.build(we.lowrank(2, vec(33, 2, N), vec(34, D, 2, size=0.5))), 1.0, bf)
        check(x["output_add"] is True and x["folded_form"] is False and x["input_side"] == "none", f"5c: a low-rank term has no input side and no folded form ({x})")
        col = gw.dense_cols(Fb.mod(3), torch.tensor([9]))[0].float().numpy()
        e20 = Fb.build(we.add_to_columns(3, [9], (20.0 * col)[None, :]))
        rel = find_key(e20.report, "max")
        check(len(rel) == 1 and close(rel[0], 20.0, 1e-3), f"5c: a column added to by 20 times itself reports rel_change 20 ({rel})")
        says(we.dose_refusal(e20, 4.0, f32), "At dose 4 this edit would change layers.3.mlp.down_proj by 80 times its own length; 1a takes at most 64.",
             "5c: dose_refusal gives the size sentence at dose 4 for a column whose rel_change is 20")
        check(we.dose_refusal(e20, 2.0, f32) is None and we.dose_refusal(e20, 0.0, f32) is None and we.dose_refusal(sil_p, 0.0, bf) is None,
              "5c: and nothing at dose 2, or at dose 0")
        ones = Fb.build(we.scale(2, cols, 1.0), we.scale(3, [9], 1.000000001))  # the second is 1 in the float32 a factor is kept in
        says(we.dose_refusal(ones, 1.0, f32), S_NOTHING, "5c: an edit whose every factor is exactly 1 and that adds nothing is refused at any dose: it is the base")
        check(we.dose_refusal(ones, -2.0, bf) == S_NOTHING and we.dose_refusal(ones, 0.0, f32) is None
              and we.dose_refusal(Fb.build(we.scale(2, cols, 1.0), we.add_to_columns(3, [9], vec(35, 1, D))), 1.0, f32) is None,
              "5c: at another dose too; not at dose 0, and not when something is added after the product")
        facts = we.edit_facts(Fb.build(we.scale(2, [4], 1.5), we.silence(3, [17, 5])))
        check(set(facts) == {"layers", "lowest_layer", "linears", "has_vectors", "has_lowrank", "factors"} and facts["layers"] == [2, 3]
              and facts["lowest_layer"] == 2 and facts["linears"] == 2 and facts["has_vectors"] is False and facts["has_lowrank"] is False
              and sorted(float(f) for f in facts["factors"]) == [0.0, 1.5], f"5c: edit_facts ({facts})")
        rf = we.recipe_facts({"terms": [{"layer": 3, "silence": [17, 5]}, {"layer": 2, "scale": [4], "factor": 1.5}]}, TMP)
        check({k: (sorted(v) if isinstance(v, list) else v) for k, v in rf.items()} == {k: (sorted(v) if isinstance(v, list) else v) for k, v in facts.items()},
              f"5c: recipe_facts gives the same facts with no model ({rf})")
        f_sil, f_15 = we.edit_facts(sil_f), we.edit_facts(Fb.build(we.scale(2, cols, 1.5)))
        f_set, f_low = we.edit_facts(Fb.build(we.set_columns(2, cols, v_set))), we.edit_facts(Fb.build(we.lowrank(2, vec(33, 2, N), vec(34, D, 2, size=0.5))))
        check(not we.twin_wanted(f_sil, 1.0) and not we.twin_wanted(f_sil, -1.0) and not we.twin_wanted(f_sil, 0.5) and we.twin_wanted(f_sil, 0.3)
              and we.twin_wanted(f_15, 1.0) and not we.twin_wanted(f_15, 2.0) and we.twin_wanted(f_set, 1.0) and not we.twin_wanted(f_low, 1.0),
              "5c: twin_wanted: not for a factor of 0 or a power of two at the dose, yes for 1.5 and for a set, never for a low-rank term")
    for box in (Pb, Fb):
        with part(f"5d ({box.label}): null rows"):
            seen: Dict[str, torch.Tensor] = {}
            hk = box.lm.layers[2].mlp.register_forward_hook(lambda m, a, o: seen.__setitem__("out", o.detach().clone()))
            try:
                box.whole(ids, R)
                bare = seen["out"]
                with we.per_row(box.lm, [None, we.Null(2, 0.125), None, we.Null(2, -0.125)] + [None] * (R - 4)):
                    h = box.whole(ids, R)
                nulled = seen["out"]
            finally:
                hk.remove()
            s_up, s_dn = we.null_factor(0.125, box.dtype), we.null_factor(-0.125, box.dtype)
            want = bare.clone()
            want[1] = bare[1] * torch.tensor(s_up, dtype=box.dtype)
            want[3] = bare[3] * torch.tensor(s_dn, dtype=box.dtype)
            check((s_up, s_dn) == (1.125, 0.875) and torch.equal(nulled, want),
                  f"5d ({box.label}): a null row's factor multiplies its own row's feed-forward output, in the model's dtype, and no other row")
            check(all(torch.equal(h[r], h[0]) for r in (2, 4, R - 1)), f"5d ({box.label}): the rows beside the null rows end as the base row")
    with part("5d: null factors that are refused"):
        bf = torch.bfloat16
        check(we.null_factor(0.001, bf) == 1.0 and we.null_factor(0.03125, bf) == 1.03125 and we.null_factor(0.005, bf) == we.null_factor(0.01, bf) == 1.0078125
              and we.null_factor(0.001, torch.float32) == float(torch.tensor(1.001, dtype=torch.float32)),
              "5d: null_factor is 1 + e as the dtype holds it (1.001 is 1 in bf16; 1.005 and 1.01 are both 1 + 1/128)")

        def null_row(e: float) -> Callable[[], None]:
            def go() -> None:
                with we.per_row(Pb.lm, [None, we.Null(2, e)]):
                    Pb.whole(ids, 2)
            return go

        says(words_of(null_row(0.001), we.Refused), s_null_one(0.001), "5d: a Null with e = 0.001, a factor of 1 in bf16, is refused")
        says(words_of(null_row(1.0), we.Refused), s_null_two(1.0, 2.0), "5d: a Null with e = 1, a power of two, is refused")
        check(we.hook_census(Pb.lm) == {}, "5d: and no hook is left")
    with part("5e: a forward of another row count"):
        e = Pb.build(we.silence(2, cols))

        def wrong_rows() -> None:
            with we.per_row(Pb.lm, [None, we.Arm(e, 1.0), None, None]):
                Pb.whole(ids, 5)

        says(words_of(wrong_rows, RuntimeError), "hooks made for 4 rows met 5", "5e: a forward of 5 rows under hooks made for 4 raises from the hook")

        def flat() -> None:
            with we.per_row(Pb.lm, [None, we.Arm(e, 1.0), None, None]):
                Pb.mod(2)(torch.zeros(4, N, dtype=Pb.dtype))  # four rows, as the hooks were made for, but no pieces

        got = words_of(flat, RuntimeError)
        check(got is not None and not got.startswith("<") and f"[4, {N}]" in got and got != "hooks made for 4 rows met 4",
              f"5e: a tensor that is not [rows, pieces, width] raises from the hook and is said by its shape, not as a row count that was met ({got!r})")
        check(we.hook_census(Pb.lm) == {}, "5e: and no hook is left")
    with part("5f: what can be folded"):
        plain = "layers.2.mlp.down_proj is a plain linear whose forward reads no overlay: the folded form is the packed checkpoint's."
        e_p, e_f = Pb.build(we.scale(2, cols, 1.5)), Fb.build(we.scale(2, cols, 1.5))
        low_p = Pb.build(we.lowrank(2, vec(33, 2, N), vec(34, D, 2, size=0.5)))
        check(we.can_fold(Pb.lm, e_p) is None, "5f: can_fold is None on P")
        says(we.can_fold(Fb.lm, e_f), plain, "5f: can_fold gives the plain-linear sentence on F")
        says(we.can_fold(Pb.lm, low_p), S_LOWRANK_FOLD, "5f: can_fold gives the low-rank sentence for an edit with a low-rank term")
        check(we.can_fold(F5.lm, F5.build(we.scale(2, cols, 1.5))) is None, "5f: can_fold is None on a plain linear given the overlay forward")

        def fold(box: Box, e: Any) -> Callable[[], None]:
            def go() -> None:
                with we.folded(box.lm, e, 1.0):
                    pass
            return go

        says(words_of(fold(Fb, e_f), we.Refused), plain, "5f: folded raises the plain-linear sentence on F")
        says(words_of(fold(Pb, low_p), we.Refused), S_LOWRANK_FOLD, "5f: folded raises the low-rank sentence")
        check(we.overlays_in_force(Pb.lm) == [] and we.hook_census(Pb.lm) == {} and we.hook_census(Fb.lm) == {}, "5f: and nothing is left on either model")


# ------------------------------------------------------------------------------------------- 6. a decode step


def check_decode(Fb: Box, Pb: Box) -> None:
    for box in (Fb, Pb):
        with part(f"6 ({box.label})"):
            lm, lab = box.lm, box.label
            prompt = list(lv.make_item(box.tok, QA[0][1], QA[0][2]).prompt_ids)
            R, steps = 8, 10
            strong = box.build(we.silence(1, list(range(96))))
            set_e = box.build(we.set_columns(2, [3, 40], vec(41, 2, D)))
            bare = box.write(prompt, R, steps, [])
            with we.per_row(lm, [None, we.Arm(strong, 0.0), we.Arm(set_e, 0.0)] + [None] * (R - 3)):
                quiet = box.write(prompt, R, steps, [])
            check(quiet == bare and all(len(bare[r]) == steps for r in range(R)),
                  f"6 ({lab}): under per_row with untouched rows and dose-0 arms, every row writes the pieces and log-probabilities of the same row of an unhooked write, exactly")
            progs = [None, we.Null(1, 0.125), we.Arm(strong, 1.0), we.Arm(strong, 1.0), we.Arm(set_e, 1.0), None, None, None]
            seen: List[int] = []
            probe = box.mod(1).register_forward_pre_hook(lambda m, a: seen.append(int(a[0].shape[1])))
            try:
                with we.per_row(lm, progs):
                    hooked = box.write(prompt, R, steps, [])
            finally:
                probe.remove()
            check(all(hooked[r] == bare[r] for r in (0, 5, 6, 7)), f"6 ({lab}): beside edited rows, every untouched row writes as the same row of the unhooked write, exactly")
            if box is Pb:
                check(all(hooked[r] == hooked[0] for r in (5, 6, 7)), "6 (P): every untouched row also equals row 0")
                check(hooked[2] == hooked[3], "6 (P): two rows under one edit write the same pieces with the same log-probabilities")
            check([t[0] for t in hooked[2]] != [t[0] for t in hooked[0]], f"6 ({lab}): the edited row writes other pieces than the base row")
            check(len(seen) >= steps and seen[0] == len(prompt) and all(s == 1 for s in seen[1:]),
                  f"6 ({lab}): the hooks see the prompt's length once, then sequence length 1 on every step ({seen[:4]}...)")
            check(we.hook_census(lm) == {}, f"6 ({lab}): no hook is left after the write")
            # a row that stops early: stop at a piece the edited rows write and no other row ever does
            others = {t[0] for r in (0, 1, 4, 5, 6, 7) for t in hooked[r]}
            mine = [t[0] for t in hooked[2]]
            j = next((i for i, t in enumerate(mine) if i < steps - 2 and t not in others and t not in mine[:i]), None)
            check(j is not None, f"6 ({lab}): the edited row writes, before its last pieces, a piece no other row writes (the check's own need)")
            if j is not None:
                with we.per_row(lm, progs):
                    stopped = box.write(prompt, R, steps, [mine[j]])
                bare_stopped = box.write(prompt, R, steps, [mine[j]])
                check(len(stopped[2]) == j + 1 and stopped[2][-1][2] and stopped[2] == hooked[2][:j] + [(mine[j], hooked[2][j][1], True)],
                      f"6 ({lab}): the edited row stops at its piece {j}, the pieces before it unchanged")
                check(all(len(stopped[r]) == steps and not any(t[2] for t in stopped[r]) for r in (0, 1, 4, 5, 6, 7)) and stopped[0] == bare_stopped[0]
                      and all(stopped[r] == hooked[r] for r in (0, 1, 4, 5, 6, 7)),
                      f"6 ({lab}): a batch in which one row stops early goes on for the others, which write as before")
            if box is Fb:
                ids = prompt + mine
                with we.per_row(lm, progs):
                    h = box.whole(ids, R)
                greedy = F.linear(h[2].to(box.W_U.dtype), box.W_U).argmax(-1)[len(prompt) - 1:len(prompt) - 1 + len(mine)].tolist()
                check(greedy == mine, "6 (F): the edited row's written pieces, read again teacher-forced under the same program in one whole forward, are the greedy choice at every step")
                # the same for the OUTPUT-side hook, which the silenced row above never passes: a set, a low-rank term
                # and an add, each written with a cache and then read again teacher-forced, piece and log-probability
                e_set = box.build(we.set_columns(2, list(range(40)), vec(42, 40, D, size=0.6)))
                e_low = box.build(we.lowrank(1, vec(44, 3, N, size=0.5), vec(45, D, 3, size=0.5)))
                e_add = box.build(we.add_to_columns(2, list(range(40)), vec(43, 40, D, size=0.6)))
                progs2 = [None, we.Arm(e_set, 1.0), we.Arm(e_low, 0.5), we.Arm(e_add, -1.0)] + [None] * (R - 4)
                with we.per_row(lm, progs2):
                    wrote = box.write(prompt, R, steps, [])
                for row, what in ((1, "a set"), (2, "a low-rank term at dose 0.5"), (3, "an add at dose -1")):
                    mine = [t[0] for t in wrote[row]]
                    with we.per_row(lm, progs2):
                        h = box.whole(prompt + mine, R)
                    z = F.linear(h[row].to(box.W_U.dtype), box.W_U)  # the writer's own logits: the model's dtype, softcapped
                    lsm = F.log_softmax((torch.tanh(z / box.softcap) * box.softcap).float(), dim=-1)[len(prompt) - 1:len(prompt) - 1 + len(mine)]
                    gap = float((lsm.gather(1, torch.tensor(mine)[:, None])[:, 0] - torch.tensor([t[1] for t in wrote[row]])).abs().max())
                    check(lsm.argmax(-1).tolist() == mine and gap < 1e-4 and mine != [t[0] for t in wrote[0]],
                          f"6 (F): {what}: the row parts from the base row, and every piece it wrote with a cache is the teacher-forced greedy piece under the same "
                          f"program, at the same log-probability (largest difference {gap:.1e})")


# -------------------------------------------------------------------------------------------- 7. control edits


def check_controls(Fb: Box) -> None:
    lm = Fb.lm
    A, B = vec(54, 2, N), vec(55, D, 2, size=0.5)
    e = Fb.build(we.set_columns(2, [5, 17, 90], vec(51, 3, D)), we.scale(2, [30], 1.5), we.lowrank(3, A, B), we.add_to_columns(4, [8], vec(52, 1, D)))
    with part("7: seeded, other sites, the same size"):
        controls = we.control_edits(e, lm, 7, 4)
        ids = [c.id for c in controls]
        check(len(controls) == 4 and len(set(ids)) == 4 and e.id not in ids and [c.id for c in we.control_edits(e, lm, 7, 4)] == ids,
              "7: the same seed gives the same ids, all different and none the edit's")
        check([c.id for c in we.control_edits(e, lm, 8, 4)] != ids and [c.id for c in we.control_edits(e, lm, 7, 2)] == ids[:2] and we.control_edits(e, lm, 7, 0) == [],
              "7: another seed gives other ids; a control's draw does not depend on how many are asked; none asked, none given")
        check(all(c.id == own_edit_id(c) and c.base == e.base for c in controls), "7: each control is an edit on the same base whose id is its own hash")
        check([c.note for c in controls] == [f"control {i} of {e.short()}, seed 7" for i in range(4)], f"7: each control's note says which it is ({controls[0].note!r})")
        check(all(set(find_key(c.report, "caps")) == {"not applied to a control edit"} for c in controls), "7: each control's report says the caps were not applied")
        for i, ctl in enumerate(controls):
            ok, why = len(ctl.terms) == len(e.terms), ""
            for t, (mine, theirs) in enumerate(zip(e.terms, ctl.terms)):
                rng = np.random.default_rng([7, i, t])  # the spec's seeds (4.9): seed, control, the term's place
                if mine.kind == "cols":
                    pool = np.setdiff1d(np.arange(N), mine.idx)
                    chosen = np.sort(rng.choice(pool, size=len(mine.idx), replace=False))
                    good = (theirs.linear == mine.linear and theirs.kind == "cols" and np.array_equal(theirs.idx, chosen)
                            and not set(theirs.idx.tolist()) & set(mine.idx.tolist()) and np.array_equal(theirs.op, mine.op)
                            and np.array_equal(theirs.factor, mine.factor)
                            and ((mine.vectors is None and theirs.vectors is None) or np.array_equal(theirs.vectors, mine.vectors)))
                else:
                    perm = rng.permutation(N)
                    good = theirs.linear == mine.linear and theirs.kind == "lowrank" and np.array_equal(theirs.A, mine.A[:, perm]) and np.array_equal(theirs.B, mine.B)
                if not good:
                    ok, why = False, f" (term {t}, {mine.kind})"
            check(ok, f"7: control {i}: as many columns in the same layer, none the edit's own, drawn as the spec draws them, ops, factors and vectors "
                  f"carried over in order; the low-rank term's neuron axis permuted{why}")
    with part("7: too few other sites"):
        big = Fb.build(we.silence(3, list(range(100))))
        says(words_of(lambda: we.control_edits(big, lm, 0, 1), we.Refused),
             "Layer 3 (page layer 4) has 28 neurons outside this edit: too few for a control edit of 100.", "7: too few other sites is refused")
    with part("7: a set carried to a column made nearly nought"):
        F7 = fresh_f()
        chosen = np.sort(np.random.default_rng([0, 0, 0]).choice(np.setdiff1d(np.arange(N), np.array([5, 17])), size=2, replace=False))
        F7.mod(2).weight.data[:, int(chosen[0])] *= 1e-4  # before anything is pinned on this copy
        v = vec(53, 2, D)
        e7 = F7.build(we.set_columns(2, [5, 17], v))
        ctl = we.control_edits(e7, F7.lm, 0, 1)[0]
        rel = find_key(ctl.report, "max")
        check(ctl.terms[0].idx.tolist() == chosen.tolist() and len(rel) == 1 and rel[0] > we.MAX_REL_CHANGE
              and set(find_key(ctl.report, "caps")) == {"not applied to a control edit"},
              f"7: it builds as a control though it is over MAX_REL_CHANGE (rel_change {rel}), and its report says the caps were not applied")
        says(words_of(lambda: F7.build(we.set_columns(2, chosen.tolist(), v)), we.Refused),
             f"Neuron {int(chosen[0])} of layer 2 (page layer 3) would change by {W} times its own length; 1a takes at most 64.",
             "7: the same columns asked as an edit are refused by the gate")


# ------------------------------------------------------------------------------------------ 8. items and plans


def check_items_plans(Fb: Box) -> None:
    rt, tok = Fb.rt, Fb.tok
    bos = tok.bos_token_id
    with part("8: items"):
        for _name, q, a, system in QA:
            want = rt.tokenize_prompt(rt.compose_prompt(q, chat=True, system=system))
            check(lv.chat_ids(tok, q, system) == want and want[0] == bos and want.count(bos) == 1,
                  f"8: chat_ids equals the runtime's own prompt ids, the start mark first and once ({'with' if system else 'without'} a system turn)")
        check(lv.chat_ids(tok, QA[0][1]) == rt.tokenize_prompt(rt.compose_prompt(QA[0][1], chat=True)) != lv.chat_ids(tok, QA[0][1], "Be brief."),
              "8: chat_ids with no system turn, and another prompt with one")
        _n, q, a, _s = QA[0]
        it = lv.make_item(tok, q, a, name="orange")
        a_ids = [int(t) for t in tok(a, add_special_tokens=False)["input_ids"]]
        check(list(it.prompt_ids) == lv.chat_ids(tok, q) and list(it.answer_ids) == a_ids and list(it.ids) == lv.chat_ids(tok, q) + a_ids
              and isinstance(it.prompt_ids, tuple) and isinstance(it.answer_ids, tuple) and (it.name, it.question, it.answer, it.system) == ("orange", q, a, None),
              "8: an item is the chat prompt's ids and the answer's, tokenised apart and appended, with its texts")
        check(it.id == sha_obj({"v": 1, "prompt_ids": list(it.prompt_ids), "answer_ids": a_ids}), "8: an item's id is the spec's hash of its ids")
        P, end, y = span_of(it)
        check(P[0] == len(it.prompt_ids) - 1 and P[-1] == len(it.ids) - 2 and len(P) == len(a_ids) and end == len(it.ids) - 1
              and [it.ids[p + 1] for p in P] == a_ids, "8: the span positions predict the answer's pieces, and the end position is the last (read against in section 9)")
        check(words_of(lambda: setattr(it, "name", "x"), Exception) is not None, "8: an item is frozen")
        says(words_of(lambda: lv.make_item(tok, q, ""), lv.Refused), S_NO_ANSWER, "8: an item with no answer is refused")
        says(words_of(lambda: lv.make_item(tok, "say <bos> twice", a), lv.Refused), "The chat template gave 2 start marks; an item has exactly one, first.",
             "8: a question that brings a second start mark is refused")
        for label, text in (("a start mark", " Orange <bos> is warm."), ("a turn mark", " Orange is warm.<turn|>")):
            got = words_of(lambda: lv.make_item(tok, q, text), lv.Refused)
            check(got is not None and not got.startswith("<") and "plain text" in got, f"8: an answer that holds {label} is refused: it would be scored as text ({got!r})")
        long_a = " the" * 600
        T = len(lv.chat_ids(tok, q)) + len(tok(long_a, add_special_tokens=False)["input_ids"])
        says(words_of(lambda: lv.make_item(tok, q, long_a), lv.Refused), f"This item is {T} word-pieces; 1a reads at most 512.", "8: an item over 512 pieces is refused")
        raw = lv.item_from_ids([7], [8, 9, 10], name="raw")
        check((list(raw.prompt_ids), list(raw.answer_ids), list(raw.ids), raw.question, raw.answer, raw.system, raw.name) == ([7], [8, 9, 10], [7, 8, 9, 10], None, None, None, "raw"),
              "8: item_from_ids takes ids with no start mark, its texts None")
        check(words_of(lambda: lv.item_from_ids([], [8]), lv.Refused) is not None and words_of(lambda: lv.item_from_ids([7], []), lv.Refused) is not None
              and words_of(lambda: lv.item_from_ids([7], [8] * 512), lv.Refused) is not None and len(lv.item_from_ids([7], [8] * 511).ids) == 512,
              "8: item_from_ids asks for a piece each side and 512 in all")
    with part("8: the module's constants"):
        check((lv.VERSION, lv.ITEM_GATE_BAR, lv.STABLE_RATIO, lv.ONE_ROW_MIN_LAYER, lv.ROWS_SMALL, lv.ROWS_LARGE) == (1, -6.0, 3.0, 44, 12, 32)
              and (lv.SMALL_MIN, lv.SMALL_MAX, lv.LARGE_MIN, lv.LARGE_MAX, lv.WRITTEN_ROWS_MIN, lv.WRITTEN_ROWS_MAX) == (5, 12, 16, 64, 5, 32)
              and tuple(lv.NULL_E) == (0.03125, -0.03125, 0.125, -0.125) and tuple(lv.NULL_E_WRITTEN) == (0.125, -0.125)
              and (lv.MAX_PIECES, lv.MAX_ROW_POSITIONS, lv.MAX_CONTROLS, lv.MAX_DOSES, lv.MAX_ITEMS, lv.MAX_WRITTEN_GIB, lv.READ_CHUNK) == (512, 12288, 199, 8, 256, 6.0, 128),
              "8: the bars and caps of 3.3 as the spec ships them")
        # forward_fixed_s, row_position_s, readout_row_position_s and mib_row_position: the card check's refit of 16 November 2026
        # (verify_1a.json, proposed_bars.cost: 0.13502, 0.000376947, 0.0000880039 and 0.19113, to three figures); the rest as 3.3 has them
        check(lv.COST == {"forward_fixed_s": 0.135, "row_position_s": 0.000377, "readout_row_position_s": 0.000088, "decode_step_s": 0.095,
                          "overlay_matrix_s": 0.0003, "load_s": 9.5, "unpacker_s": 1.83, "mib_row_position": 0.191, "mb_row_piece_written": 0.8},
              "8: COST holds the card check's refit (16 November 2026) and the measured figures of 3.3")
        check(lv.CONTROLS_SAY.format(seed=3) == S_CONTROLS.format(seed=3) and lv.ROWS_SAY.format(small=12, large=32) == S_ROWS_SAY.format(small=12, large=32)
              and lv.PAIRED_SAY == S_PAIRED, "8: the three sentences of 4.9, word for word")
        check(issubclass(lv.Refused, ValueError), "8: levelling.Refused is a ValueError")
    base = {"name": "r", "kind": "paired", "edit": {"terms": [{"layer": 3, "silence": [5]}]}}
    facts = we.recipe_facts(base["edit"], TMP)

    def norm(over: Dict[str, Any], dtype: Any = torch.bfloat16, drop: Sequence[str] = (), **kw: Any) -> Dict[str, Any]:
        reading = {k: v for k, v in {**base, **over}.items() if k not in drop}
        return lv.normalise(reading, n_layers=NL, tokenizer=tok, facts=kw.get("facts", facts), dtype=dtype)

    def refused(over: Dict[str, Any], **kw: Any) -> Optional[str]:
        return words_of(lambda: norm(over, **kw), lv.Refused)

    with part("8: the defaults"):
        given = copy.deepcopy(base)
        plan = lv.normalise(given, n_layers=NL, tokenizer=tok, facts=facts)
        check(given == base, "8: normalise leaves the plan it was given as it was")
        check(plan["doses"] == [1.0] and plan["main_dose"] == 1.0 and plan["items"] == "all" and plan["controls"]["count"] == 19 and plan["controls"]["seed"] == 0
              and plan["null_e"] == [0.03125, -0.03125, 0.125, -0.125] and plan["rows"] == {"small": 12, "large": 32} and plan["folded"] == "auto"
              and plan["lower"] == "same_shape" and plan["bars"] == {"item_gate": -6.0, "stable_ratio": 3.0, "one_row_min_layer": 44}
              and plan["name"] == "r" and plan["kind"] == "paired" and plan["edit"] == base["edit"], f"8: a paired reading's defaults (2.6): {plan}")
        check(plan["readouts"] == {"kl": True, "own": True, "odds": None} and plan["rank_by"] == "kl", "8: a plan that names no pieces reads kl and own, ranked by kl")
        check(json_clean(plan) is None, f"8: the normalised plan is JSON-clean ({json_clean(plan)})")
        the, a_, sea = (tok(w, add_special_tokens=False)["input_ids"] for w in (" the", " a", " sea"))
        check(len(the) == len(a_) == len(sea) == 1, "8: (\" the\", \" a\" and \" sea\" are one piece each on the stand-in's tokenizer)")
        named = norm({"readouts": {"odds": {"toward": [" the", sea[0]], "away": [" a"]}}})
        check(named["readouts"] == {"kl": False, "own": False, "odds": {"toward": [the[0], sea[0]], "away": [a_[0]], "at": "end"}} and named["rank_by"] == "odds",
              f"8: a plan that names pieces reads odds alone, at the end, the words resolved to ids ({named['readouts']})")
        both = norm({"readouts": {"kl": True, "odds": {"toward": [the[0]], "away": [a_[0]], "at": "span"}}})
        check(both["readouts"]["kl"] is True and both["readouts"]["own"] is False and both["readouts"]["odds"]["at"] == "span" and both["rank_by"] == "kl",
              "8: kl asked beside odds is on, and the rank is by kl")
        check(norm({"readouts": {"kl": False, "own": True}})["rank_by"] == "own", "8: with kl off the rank is by own")
        check(norm({"doses": [0.0, 0.5, 2.0]})["main_dose"] == 0.5 and norm({"doses": [2.0, 1.0]})["main_dose"] == 1.0
              and norm({"doses": [2.0, 0.5], "main_dose": 0.5})["main_dose"] == 0.5, "8: main_dose is 1 when listed, else the first dose that is not 0, or the plan's")
        over = norm({"bars": {"item_gate": GATE}})
        check(over["bars"] == {"item_gate": GATE, "stable_ratio": 3.0, "one_row_min_layer": 44}, "8: a bar the plan gives stands beside the module's other two")
        kept = lv.ITEM_GATE_BAR
        try:
            lv.ITEM_GATE_BAR = -7.5
            check(norm({})["bars"]["item_gate"] == -7.5, "8: the module's bars are read when normalise is called")
        finally:
            lv.ITEM_GATE_BAR = kept
        wr = norm({"kind": "written"})
        check(wr["doses"] == [1.0] and wr["main_dose"] == 1.0 and wr["controls"]["count"] == 3 and wr["controls"]["seed"] == 0 and wr["null_e"] == [0.125, -0.125]
              and wr["rows"] == 8 and wr["max_new"] == 48 and "readouts" not in wr and "bars" not in wr, f"8: a written reading's defaults (2.6): {wr}")
        check(norm({"null_e": [0.001, -0.001]}, dtype=torch.float32)["null_e"] == [0.001, -0.001], "8: in float32 a null row of e = 0.001 is taken")
    with part("8: every refusal of 2.6"):
        says(refused({"bogus": 1}), "The plan has a key 1a does not know: bogus.", "8: a key 1a does not know")
        says(refused({"kind": "written", "readouts": {"kl": True}}), "The plan has a key 1a does not know: readouts.", "8: a written reading takes no other key")
        says(refused({"rows": {"small": 4, "large": 32}}), s_rows_range(4, 32), "8: a small batch under 5 rows")
        says(refused({"rows": {"small": 12, "large": 65}}), s_rows_range(12, 65), "8: a large batch over 64 rows")
        says(refused({"rows": {"small": 13, "large": 32}}), s_rows_range(13, 32), "8: a small batch over 12 rows")
        says(refused({"kind": "written", "rows": 33}), "A written reading is one batch of 5 to 32 rows; 33 were asked.", "8: a written batch over 32 rows")
        says(refused({"rows": {"small": 5, "large": 32}, "null_e": [0.125, -0.125], "doses": [1.0, 0.5]}),
             "A batch of 5 rows holds the base row, its twin, 2 null rows and 1 more: too few for 2 doses. Ask fewer doses or null rows, or more rows.",
             "8: doses that do not fit forward 0 of the small class")
        says(refused({"kind": "written", "rows": 5}),
             "A written reading is one batch: 5 rows hold the base row, its twin, 2 null rows and 1 more, too few for 1 doses and 3 control edits. "
             "Ask fewer, or more rows.", "8: a written reading whose arms do not fit its one batch")
        says(refused({"doses": [5.0]}), S_DOSE, "8: a dose over 4")
        says(refused({"doses": [1.0, float("inf")]}), S_DOSE, "8: a dose that is not finite")
        says(refused({"doses": [10 ** 400]}), S_DOSE, "8: a dose past what a float holds is refused in the same words, not raised as an overflow")
        says(refused({"doses": [0.0]}), S_DOSE_0, "8: no dose but 0")
        says(refused({"null_e": [0.125]}), S_NULLS, "8: one null row")
        says(refused({"null_e": [0.125, 0.25]}), S_NULLS, "8: null rows all on one side")
        says(refused({"null_e": [0.125, -0.75]}), "A null row scales the layer's feed-forward output by 1 + e, with e between -0.5 and 2 and not 0; -0.75 was asked.",
             "8: an e under -0.5")
        says(refused({"null_e": [0.125, -0.125, 0.0]}), "A null row scales the layer's feed-forward output by 1 + e, with e between -0.5 and 2 and not 0; 0 was asked.",
             "8: an e of 0")
        says(refused({"null_e": [0.125, -0.125, 0.001]}), s_null_one(0.001), "8: a null factor of 1 in bf16 (e = 0.001)")
        says(refused({"null_e": [1.0, -0.125]}), s_null_two(1.0, 2.0), "8: a null factor that is a power of two (e = 1)")
        says(refused({"null_e": [0.125, -0.5]}), s_null_two(-0.5, 0.5), "8: a null factor that is a power of two (e = -0.5)")
        says(refused({"null_e": [0.005, 0.01, -0.125]}), f"e = 0.005 and e = 0.01 are one factor in {W} (1.00781): the two null rows would be the same row.",
             "8: e = 0.005 and 0.01 are one factor in bf16")
        says(refused({"readouts": {"kl": False, "own": False, "odds": None}}), S_NO_READOUT, "8: no read-out on")
        says(refused({"rank_by": "odds"}), "rank_by names a read-out that is on; odds is not.", "8: rank_by names a read-out that is off")
        word = "two words here"
        k = len(tok(word, add_special_tokens=False)["input_ids"])
        says(refused({"readouts": {"odds": {"toward": [word], "away": [5]}}}),
             f"{word!r} is {k} word-pieces ({W}); a named piece is one. Give its id, or a word that is one piece.", "8: a named piece that is several pieces")
        says(refused({"lower": "one_row"}), s_one_row(44, 3), "8: a one-row capture below the bar's layer")
        check(norm({"lower": "one_row", "bars": {"one_row_min_layer": 3}})["lower"] == "one_row", "8: and taken when the plan lowers that bar to the edit's layer")
        np.save(TMP / "lr_a.npy", vec(61, 2, N))
        np.save(TMP / "lr_b.npy", vec(62, D, 2, size=0.5))
        low = {"terms": [{"layer": 3, "lowrank": {"A": "lr_a.npy", "B": "lr_b.npy"}}]}
        low_facts = we.recipe_facts(low, TMP)
        check(low_facts["has_lowrank"] is True and low_facts["lowest_layer"] == 3, f"8: recipe_facts of a low-rank recipe ({low_facts})")
        says(refused({"edit": low, "folded": "always"}, facts=low_facts), S_LOWRANK_FOLD, "8: folded \"always\" with a low-rank term")
        got = refused({"kind": "sampled"})
        check(got is not None and not got.startswith("<") and "sampled" in got, f"8: a kind that is neither paired nor written is refused, naming it ({got!r})")
        check(refused({"controls": {"count": 200, "seed": 0}}) is not None and refused({"doses": [1.0, 1.0]}) is not None
              and refused({"doses": [0.1 * i for i in range(1, 10)]}) is not None and refused({"folded": "sometimes"}) is not None
              and refused({}, drop=("name",)) is not None and refused({}, drop=("edit",)) is not None,
              "8: 200 control edits, a dose listed twice, nine doses, an unknown folded, and a plan with no name or no edit are refused")
    with part("8: the plan file"):
        good = plan_file(TMP / "plans8", "good", [base, {**base, "name": "s", "items": ["orange"]}])
        plan = lv.load_plan(good)
        items = lv.plan_items(plan, tok)
        check(list(items) == [n for n, *_ in QA] and all(items[n].name == n and items[n].question == q and items[n].answer == a and items[n].system == s
                                                         and list(items[n].prompt_ids) == lv.chat_ids(tok, q, s) for n, q, a, s in QA),
              "8: plan_items gives the plan file's items, in its order, by name")
        check([r["name"] for r in plan["readings"]] == ["r", "s"] and plan["v"] == 1, "8: load_plan gives the plan file's readings")
        twice = plan_file(TMP / "plans8", "twice", [base, base])
        got = words_of(lambda: lv.load_plan(twice), lv.Refused)
        check(got is not None and not got.startswith("<") and "r" in got, f"8: a reading name used twice is refused by load_plan ({got!r})")
        unknown = plan_file(TMP / "plans8", "unknown", [{**base, "items": ["orange", "nobody"]}])
        got = words_of(lambda: lv.load_plan(unknown), lv.Refused)
        check(got is not None and not got.startswith("<") and "nobody" in got, f"8: an unknown item name is refused by load_plan ({got!r})")
        (TMP / "plans8" / "deep.json").write_text("[" * 100000 + "]" * 100000)
        got = words_of(lambda: lv.load_plan(TMP / "plans8" / "deep.json"), lv.Refused)
        check(got is not None and not got.startswith("<"), f"8: a plan file nested too deep to parse is refused in words ({str(got)[:120]!r})")
        (TMP / "plans8" / "broken.json").write_text("{not json")
        check(words_of(lambda: lv.load_plan(TMP / "plans8" / "broken.json"), (ValueError, OSError)) is not None
              and words_of(lambda: lv.load_plan(TMP / "plans8" / "absent.json"), (ValueError, OSError)) is not None, "8: a plan file that is not JSON, or is not there, raises")
    with part("8: the layout of rows"):
        ne = list(lv.NULL_E)
        small, large = lv.layout([1.0], 19, ne, 12), lv.layout([1.0], 19, ne, 32)
        check(len(small) == 4 and len(large) == 1 and all(len(f) == 12 for f in small) and len(large[0]) == 32,
              "8: layout([1.0], 19, NULL_E, 12) is 4 forwards of 12 rows, and at 32 rows 1")
        check(all(f[0] is None and f[-1] is None for f in small + large), "8: row 0 and the last row are None in each forward")
        check(all(f[1:5] == ["null:0.03125", "null:-0.03125", "null:0.125", "null:-0.125"] for f in small + large), "8: the null rows' keys are the same in each, null:0.03125 first")
        check(small[0][5] == "edit:1.0" and large[0][5] == "edit:1.0" and not any("edit:1.0" in f for f in small[1:]), "8: edit:1.0 sits in forward 0, after the null rows")
        check(small == own_layout([1.0], 19, ne, 12) and large == own_layout([1.0], 19, ne, 32) and small[3][5:] == ["control:17", "control:18", None, None, None, None, None],
              "8: both are the spec's layout row for row; rows left over in the last forward are untouched")
        check(lv.layout([1.0, -1.0, 0.0], 60, ne, 12) == own_layout([1.0, -1.0, 0.0], 60, ne, 12) and lv.layout([0.5], 0, [0.125, -0.125], 5) == [[None, "null:0.125", "null:-0.125", "edit:0.5", None]],
              "8: other shapes too: doses written as repr(float), the least batch")
        check(lv.layout([1.0], 3, [0.125, -0.125], 8, one_forward=True) == [[None, "null:0.125", "null:-0.125", "edit:1.0", "control:0", "control:1", "control:2", None]],
              "8: a written reading is one batch laid out the same way")
        check(lv.layout([1.0], 2, [0.125, -0.125], 8, one_forward=True)[0] == [None, "null:0.125", "null:-0.125", "edit:1.0", "control:0", "control:1", None, None],
              "8: and padded with untouched rows before the twin")
        says(words_of(lambda: lv.layout([1.0], 5, [0.125, -0.125], 8, one_forward=True), lv.Refused),
             "A written reading is one batch: 8 rows hold the base row, its twin, 2 null rows and 4 more, too few for 1 doses and 5 control edits. Ask fewer, or more rows.",
             "8: layout(..., one_forward=True) refuses arms that do not fit")
        says(words_of(lambda: lv.layout([1.0, 0.5, 0.25], 0, ne, 8), lv.Refused),
             "A batch of 8 rows holds the base row, its twin, 4 null rows and 2 more: too few for 3 doses. Ask fewer doses or null rows, or more rows.",
             "8: layout refuses doses that do not fit forward 0")


# ----------------------------------------------------------------------------- 9. the read-outs, by brute force


def check_readouts(Fb: Box, hF: Any, Pb: Box, hP: Any) -> None:
    lm, tok = Fb.lm, Fb.tok
    one = lambda w: int(tok(w, add_special_tokens=False)["input_ids"][0])  # noqa: E731
    odds = {"toward": [one(" the"), one(" sea")], "away": [one(" a")], "at": "span"}
    recipe = {"terms": [{"layer": 2, "silence": [3, 40, 77]}, {"layer": 3, "scale": [9], "factor": 1.5}]}
    reading = {"name": "read-outs over the span", "kind": "paired", "edit": recipe, "doses": [1.0, -1.0], "controls": {"count": 3, "seed": 4},
               "rows": {"small": 8, "large": 16}, "readouts": {"kl": True, "own": True, "odds": odds}, "bars": {"item_gate": GATE}}
    at_end = {**reading, "name": "odds at the end", "readouts": {"odds": {**odds, "at": "end"}}}
    path = plan_file(TMP / "plans9", "readouts", [reading, at_end], items=(0, 2))
    items = lv.plan_items(lv.load_plan(path), tok)
    r, r_end = read_all(hF, path, Fb)
    check(r["status"] == "ok" and r_end["status"] == "ok" and all(e["status"] == "ok" for e in r["items"] + r_end["items"]),
          f"9: both readings and their items are ok ({r['status']}: {r['why']}; {r_end['status']}: {r_end['why']})")
    edit = Fb.build(*we.from_recipe(recipe, path.parent))
    controls = we.control_edits(edit, lm, 4, 3)
    check(r["edit"]["id"] == edit.id and r["controls"]["ids"] == [c.id for c in controls], "9: the result names the edit and the control edits the check builds from the same recipe and seed")
    for res, o, metrics in ((r, odds, ("kl", "own", "odds")), (r_end, {**odds, "at": "end"}, ("odds",))):
        for entry in res["items"]:
            item = items[entry["item"]["name"]]
            S = len(item.answer_ids)
            for cls, R in (("small", 8), ("large", 16)):
                with part(f"9: {res['name']}, {entry['item']['name']}, {R} rows"):
                    block = entry["classes"][cls]
                    keys = lv.layout([1.0, -1.0], 3, res["plan"]["null_e"], R)
                    H = []
                    for f_keys in keys:
                        with we.per_row(lm, programs_of(f_keys, edit, controls, 1.0, 2)):
                            H.append(Fb.whole(item.ids, R))
                    bad = []
                    for arm in block["arms"]:
                        b = brute(Fb, H[arm["forward"]], arm["row"], item, o)
                        if keys[arm["forward"]][arm["row"]] != arm["key"] or set(arm["metrics"]) != set(metrics) \
                                or not all(same_stats(arm["metrics"][m], own_stats(b[m])) for m in metrics):
                            bad.append(arm["key"])
                    n_arms = 4 + 2 + 3
                    # atol 1e-5: the spec's (6.1 §9)
                    check(not bad and len(block["arms"]) == n_arms and block["rows"] == R and block["forwards"] == len(keys),
                          f"9: {res['name']}, {entry['item']['name']}, {R} rows: {'/'.join(metrics)} of all {n_arms} arms equal the full log-softmax's, to 1e-5"
                          + (f" (not: {bad})" if bad else ""))
                    want_n = S if o["at"] == "span" else 1
                    check(all(st["n"] == (S if m != "odds" else want_n) for arm in block["arms"] for m, st in arm["metrics"].items()),
                          f"9: kl and own are read over the span's {S} positions, odds at " + ("the span" if o["at"] == "span" else "the end position alone"))
                    b0 = brute(Fb, H[0], 0, item, o)
                    own0 = b0["base_own"]
                    want = {"mean": float(own0.mean()), "median": float(own0.median()), "worst": float(own0.min())}
                    check(all(close(entry["gate"][cls][k], want[k]) for k in want) and all(close(block["base"]["own"][k], want[k]) for k in want),
                          f"9: the base row's log-probability of the answer's pieces (the gate's figure): mean {want['mean']:.3f}, {R} rows")
                    check(block["base"]["odds"] is not None and close(block["base"]["odds"]["mean"], float(b0["base_odds"].mean()))
                          and close(block["base"]["odds"]["median"], float(b0["base_odds"].median())), "9: the base row's own log-odds of the named pieces")
                    nulls = [a for a in block["arms"] if a["what"] == "null"]
                    ok = len(nulls) == 4
                    for m in metrics:
                        top = max(nulls, key=lambda a: abs(a["metrics"][m]["mean"]))
                        fl = block["floor"][m]
                        ok = ok and set(fl) == FLOOR and close(fl["mean_abs"], abs(top["metrics"][m]["mean"]), 1e-9) and fl["from"] == top["key"] and fl["of"] == 4 \
                            and close(fl["median_abs"], max(abs(a["metrics"][m]["median"]) for a in nulls), 1e-9) and fl["above_1"] == max(a["metrics"][m]["above_1"] for a in nulls)
                    check(ok, f"9: the floor of each read-out is the largest null row, names it and says of how many ({R} rows)")
                    m = res["plan"]["rank_by"]
                    mine = abs(next(a for a in block["arms"] if a["key"] == "edit:1.0")["metrics"][m]["mean"])
                    theirs = [abs(a["metrics"][m]["mean"]) for a in block["arms"] if a["what"] == "control"]
                    at_least, floor = sum(1 for c in theirs if c >= mine), block["floor"][m]["mean_abs"]
                    rk = block["rank"]
                    check(set(rk) == RANK and (rk["metric"], rk["dose"], rk["at_least"], rk["non_finite"], rk["of"], rk["rank"]) == (m, 1.0, at_least, 0, 4, at_least + 1)
                          and close(rk["value"], mine, 1e-9) and close(rk["over_floor"], mine / floor if floor else None, 1e-6)
                          and rk["controls_over_floor"] == sum(1 for c in theirs if c > floor), f"9: the rank by {m} at the main dose, against the control edits and the floor ({rk})")
                    check(lv.rank_among(mine, theirs) == {"at_least": at_least, "non_finite": 0, "of": 4, "rank": at_least + 1}, "9: rank_among gives the same count")
            with part(f"9: stability, {res['name']}, {entry['item']['name']}"):
                # since 16 November 2026 the rule is read on the edit's arms alone and the control edits' ratios are kept
                # beside them, not judged (the spec's open point 5, its second clause; the card check showed control arms
                # at the floor tripping it)
                st = entry["stability"]
                ok, disagree = set(st) == {"ratio_bar", "base_rows_same_bits", "metrics", "disagree"} and st["ratio_bar"] == 3.0 and set(st["metrics"]) == set(metrics), []
                for m in metrics:
                    floor = max(entry["classes"][c]["floor"][m]["mean_abs"] for c in ("small", "large"))
                    arms = {a["key"]: a for a in entry["classes"]["small"]["arms"] if a["what"] == "edit"}
                    ctls = {a["key"]: a for a in entry["classes"]["small"]["arms"] if a["what"] == "control"}
                    big = {a["key"]: a for a in entry["classes"]["large"]["arms"] if a["what"] != "null"}
                    ok = ok and set(st["metrics"][m]) == {"arms", "disagree", "controls"} and set(st["metrics"][m]["arms"]) == set(arms) \
                        and set(st["metrics"][m]["controls"]) == set(ctls) and len(ctls) == 3 and set(arms) | set(ctls) == set(big)
                    for key in list(arms) + list(ctls):
                        a, b = (arms.get(key) or ctls[key])["metrics"][m]["mean"], big[key]["metrics"][m]["mean"]
                        agrees, ratio = own_agree(a, b, floor, m != "kl")
                        there = lv.agree(a, b, floor, m != "kl")
                        if key in ctls:  # its two means and the rule's ratio, and no word on whether it agrees
                            got = st["metrics"][m]["controls"].get(key, {})
                            ok = ok and set(got) == {"small", "large", "ratio"} and got["small"] == a and got["large"] == b \
                                and close(got["ratio"], ratio if math.isfinite(ratio) else None, 1e-9) and close(there[1], ratio, 1e-9)
                            continue
                        got = st["metrics"][m]["arms"].get(key, {})
                        ok = ok and set(got) == {"small", "large", "ratio", "agree"} and got["small"] == a and got["large"] == b and got["agree"] == agrees \
                            and close(got["ratio"], ratio if math.isfinite(ratio) else None, 1e-9) and there[0] == agrees and close(there[1], ratio, 1e-9)
                        if not agrees:
                            disagree.append([m, key])
                    ok = ok and sorted(st["metrics"][m]["disagree"]) == sorted(k for mm, k in disagree if mm == m)
                check(ok and sorted(map(list, st["disagree"])) == sorted(disagree) and (entry["status"] == "unstable") == bool(disagree),
                      f"9: every arm of the edit is judged by the spec's rule, and every control edit's ratio is kept beside them by the same rule and not judged, "
                      f"on {', '.join(metrics)} ({len(disagree)} disagree)")
    for box, h in ((Fb, hF), (Pb, hP)):
        with part(f"9 ({box.label}): a strong edit under a sharp unembedding, main dose 0.5"):
            # The reading above cannot tell a wrong read-out from a right one: its edit reads 1e-4 nats and its null rows
            # 1e-7, the size of the tolerance, and the stand-in's logits are too small for the softcap to bend (shown by
            # the build's reviewers, 14 November 2026: KL the wrong way round, no softcap and logits in the model's dtype
            # all passed). Here the handle's unembedding is 200 times the stand-in's and 48 neurons of layer 1 are
            # silenced, so the edit reads over a nat, the softcap bends, the two directions of KL part by a tenth, and
            # the comparison follows the figure (1 part in 10,000). The main dose is 0.5, so control edits at dose 1
            # would show too. On P the model's dtype is bf16: logits made in it are not these.
            sharp_h = dataclasses.replace(h, W_U=h.W_U.float() * 200.0)
            sharp = copy.copy(box)
            sharp.W_U = sharp_h.W_U
            strong = {"terms": [{"layer": 1, "silence": list(range(0, 96, 2))}]}
            reading = {"name": f"{box.label}: a strong edit, a sharp unembedding", "kind": "paired", "edit": strong, "doses": [1.0, 0.5], "main_dose": 0.5,
                       "controls": {"count": 3, "seed": 4}, "rows": {"small": 8, "large": 16}, "readouts": {"kl": True, "own": True, "odds": odds},
                       "bars": {"item_gate": -1e9}}
            its = items_of(box, (0, 2))
            rs = lv.read(sharp_h, reading, its)
            printed(rs)
            edit = box.build(*we.from_recipe(strong, TMP))
            controls = we.control_edits(edit, box.lm, 4, 3)
            check(rs["status"] == "ok" and all(e["status"] == "ok" for e in rs["items"]) and rs["plan"]["main_dose"] == 0.5 and rs["controls"]["ids"] == [c.id for c in controls],
                  f"9 ({box.label}): the reading and its items are ok ({rs['status']}: {rs['why']})")
            bad, sizes, softcap_bends, directions = [], [], 0.0, 0.0
            for entry in rs["items"]:
                item = its[entry["item"]["name"]]
                for cls, R in (("small", 8), ("large", 16)):
                    block = entry["classes"][cls]
                    keys = lv.layout([1.0, 0.5], 3, rs["plan"]["null_e"], R)
                    H = []
                    for f_keys in keys:
                        with we.per_row(box.lm, programs_of(f_keys, edit, controls, 0.5, 1)):  # control edits at the MAIN dose
                            H.append(box.whole(item.ids, R))
                    for arm in block["arms"]:
                        b = brute(sharp, H[arm["forward"]], arm["row"], item, odds)
                        b["kl"] = b["kl"].clamp_min(0.0)  # 2.7: a position's KL is not below 0
                        ok = keys[arm["forward"]][arm["row"]] == arm["key"] and set(arm["metrics"]) == {"kl", "own", "odds"} \
                            and near_stats(arm["metrics"]["kl"], own_stats(b["kl"])) and near_stats(arm["metrics"]["own"], own_stats(b["own"])) \
                            and near_stats(arm["metrics"]["odds"], own_stats(b["odds"]), rel=1e-3, floor=1e-4)  # named pieces: to rounding, not by bits (4.8)
                        if not ok:
                            bad.append((entry["item"]["name"], R, arm["key"]))
                        if arm["key"] == "edit:1.0":
                            sizes.append(arm["metrics"]["kl"]["mean"])
                            P_, _end, _y = span_of(item)
                            l0, l1 = tables(sharp, H[0][0], P_), tables(sharp, H[0][arm["row"]], P_)
                            directions = max(directions, abs(float((l1.exp() * (l1 - l0)).sum(-1).mean()) - arm["metrics"]["kl"]["mean"]))
                            raw = F.log_softmax(H[0][0][P_].float() @ sharp.W_U.float().T, dim=-1)
                            softcap_bends = max(softcap_bends, float((raw - l0).abs().max()))
                    b0 = brute(sharp, H[0], 0, item, odds)["base_own"]
                    if not (near(entry["gate"][cls]["mean"], float(b0.mean())) and near(entry["gate"][cls]["worst"], float(b0.min()))):
                        bad.append((entry["item"]["name"], R, "the gate's figure"))
            check(not bad, f"9 ({box.label}): kl, own and odds of every null row, dose and control edit, and the gate's figure, are the full fp32 softcapped "
                  f"log-softmax's, to 1 part in 10,000 of each figure" + (f" (not: {bad[:6]})" if bad else ""))
            check(min(sizes) > 0.5 and softcap_bends > 1.0 and directions > 0.01,
                  f"9 ({box.label}): (the check's own need: the edit reads over half a nat ({min(sizes):.2f}), the softcap moves a log-probability by over a nat "
                  f"({softcap_bends:.1f}), and KL the other way round would read {directions:.2f} nats off)")
    with part("9: stats, agree and rank_among on hand-made numbers"):
        got = lv.stats(torch.tensor([0.5, -2.0, 1.5, 0.25]))
        check(got == {"mean": 0.0625, "median": 0.25, "max_abs": 2.0, "above_1": 2, "n": 4}, f"9: stats of a known vector: torch's median (the lower middle), the largest absolute value, the count above a nat ({got})")
        check(json_clean(lv.stats(torch.tensor([1.0], dtype=torch.float64))) is None and lv.stats(torch.tensor([1.0, -1.0]))["above_1"] == 0, "9: stats gives plain numbers; exactly a nat is not above a nat")
        check(lv.agree(0.0001, 0.0004, 0.001, False) == (True, 0.4) or (lv.agree(0.0001, 0.0004, 0.001, False)[0] is True and close(lv.agree(0.0001, 0.0004, 0.001, False)[1], 0.4, 1e-12)),
              "9: agree: a pair at the floor agrees (0.0001 and 0.0004 over a floor of 0.001)")
        ok, ratio = lv.agree(0.0085, 0.7258, 0.0002, False)
        check(ok is False and close(ratio, 0.7258 / 0.0085, 1e-9), f"9: agree: 0.0085 against 0.7258 does not (ratio {ratio})")
        check(lv.agree(0.5, -0.4, 0.001, True)[0] is False and lv.agree(0.5, -0.4, 0.001, False)[0] is True and lv.agree(0.002, -0.001, 0.001, True)[0] is True,
              "9: agree: opposite signs above the floor do not agree on a signed read-out; they do unsigned, and at the floor")
        check(lv.agree(None, 0.3, 0.001, False) == (False, None) and lv.agree(0.3, None, 0.001, True) == (False, None) and lv.agree(None, None, 0.001, False) == (None, None),
              "9: agree: None against a number does not agree; None against None is not compared")
        check(lv.agree(0.0, 0.0, 0.0, False) == (True, 1.0) and lv.agree(0.0, 0.5, 0.0, False) == (False, math.inf)
              and lv.agree(1.0, 2.5, 0.0, False)[0] is True and lv.agree(1.0, 2.5, 0.0, False, 2.0)[0] is False and lv.agree(1.0, 3.0, 0.0, False)[0] is True,
              "9: agree: both nought is a ratio of 1; a floor of nought under one nought is infinity; the ratio bar is 3 unless given, and 3 itself agrees")
        got = lv.rank_among(0.5, [0.1, 0.7, None, 0.5])
        check(got == {"at_least": 3, "non_finite": 1, "of": 5, "rank": 4}, f"9: rank_among: a control that is None counts as at least as large and is counted in non_finite ({got})")
        got = lv.rank_among(None, [0.1, 0.2])
        check(got["rank"] is None and got["at_least"] is None and got["of"] == 3, f"9: rank_among: a value of None has no rank ({got})")
        check(lv.rank_among(0.5, []) == {"at_least": 0, "non_finite": 0, "of": 1, "rank": 1}, "9: rank_among: with no control the rank is 1 of 1")


# --------------------------------------------------------------------------------- 10. one reading, end to end

TOP = {"v", "kind", "name", "status", "why", "void", "base_after", "edit", "base", "runtime", "plan", "plan_sha256", "bars_overridden", "rows",
       "controls", "items", "summary", "seconds", "peak_gib", "book"}
ITEM_BLOCK = {"id", "name", "question", "answer", "system", "prompt_ids", "answer_ids", "pieces", "prompt_pieces", "scored"}
CLASS = {"rows", "forwards", "lower", "restart_from", "untouched_rows_identical", "repeats_identical", "base", "arms", "floor", "rank", "folded", "peak_gib"}
ARM_ANY = {"key", "what", "e", "dose", "factor", "edit_id", "forward", "row", "non_finite", "same_as_base", "delivered",
           "delivered_end", "delivered_prompt", "metrics"}
ARM_MUST = {"key", "what", "forward", "row", "non_finite", "same_as_base", "metrics"}
STATS = {"mean", "median", "max_abs", "above_1", "n"}
FOLDED = {"taken", "why", "dose", "base_from", "restart", "rows_identical", "metrics", "per_row", "gap", "between_kl_mean", "same_bits", "seconds"}
RANK = {"metric", "dose", "value", "at_least", "non_finite", "of", "rank", "over_floor", "controls_over_floor"}
FLOOR = {"mean_abs", "median_abs", "above_1", "from", "of"}
KEPT: Dict[str, Any] = {}  # results kept for section 13


def book_entry(seq: int) -> Dict[str, Any]:
    return next(iter(lb.entries(start=seq, limit=1)))


def top_shape(r: Dict[str, Any], box: Box, what: str) -> None:
    """The top level of any result against the spec's 2.7, and its line in the ledger."""
    check(set(r) == TOP, f"{what}: the result's top-level keys are 2.7's and \"book\"" + ("" if set(r) == TOP else f" (missing {sorted(TOP - set(r))}, more {sorted(set(r) - TOP)})"))
    bad = json_clean(r)
    check(bad is None and words_of(lambda: lb.canonical(r)) is None, f"{what}: the result is JSON-clean" + ("" if bad is None else f" ({bad})"))
    check(r["v"] == 1 and r["plan_sha256"] == sha_obj(r["plan"]) and r["base"] == {"model": box.model_id, "weights_id": None},
          f"{what}: v, the plan's hash and the base")
    rt = r["runtime"]
    code = {name: cz.file_sha256(ROOT_DIR / "backend" / f"{name}.py") for name in ("weight_edits", "levelling", "level_book")}
    check({"unpacker", "hooks_at_load", "dtype", "threads", "code"} <= set(rt) and set(rt["unpacker"]) == {"mode", "fell_back_before", "fell_back_after"}
          and rt["code"] == code and rt["threads"] == 1 and isinstance(rt["hooks_at_load"], int) and isinstance(rt["dtype"], str),
          f"{what}: the runtime record with the unpacker, the hooks at load, the dtype, the threads and the three modules' sha256")
    check(rt.get("numpy") == np.__version__ and rt.get("start_mark") == box.tok.bos_token_id,
          f"{what}: the runtime record names numpy's version (control edits are drawn with its generator) and the tokenizer's start mark")
    sec = r["seconds"]
    check(set(sec) == {"predicted", "actual", "ratio", "parts"} and isinstance(sec["parts"], dict)
          and (sec["predicted"] in (None, 0) or close(sec["ratio"], sec["actual"] / sec["predicted"], 1e-6 * max(1.0, abs(sec["ratio"] or 0)))),
          f"{what}: seconds predicted, actual and their ratio ({ {k: sec.get(k) for k in ('predicted', 'actual', 'ratio')} })")
    ctl = r["controls"]
    check(ctl is None or (set(ctl) == {"family", "matched", "seed", "count", "ids", "says"} and ctl["family"] == "other_sites" and ctl["matched"] is False
                          and len(ctl["ids"]) == ctl["count"] and ctl["says"] == S_CONTROLS.format(seed=ctl["seed"])) or r["status"] == "refused",
          f"{what}: the controls block, with the spec's sentence word for word")
    check(r["peak_gib"] is None and isinstance(r["bars_overridden"], bool) and isinstance(r["base_after"], bool), f"{what}: no peak on the CPU; bars_overridden and base_after are bools")
    bk = r["book"]
    entry = book_entry(bk["seq"]) if set(bk) == {"seq", "hash"} else {}
    body = json.loads(json.dumps({k: v for k, v in r.items() if k != "book"}))
    check(entry.get("hash") == bk.get("hash") and entry.get("body") == body and entry.get("kind") == (r["kind"] if r["kind"] in ("paired", "written") else "note"),
          f"{what}: its line in the ledger holds the same result, whole, under its kind")


def paired_shape(r: Dict[str, Any], box: Box, what: str, n_items: int) -> None:
    """A paired result that was read, key for key against the spec's 2.7."""
    plan = r["plan"]
    small, large = plan["rows"]["small"], plan["rows"]["large"]
    check(r["kind"] == "paired" and r["void"] is None and r["why"] is None and r["rows"] == {"small": small, "large": large, "says": S_ROWS_SAY.format(small=small, large=large)},
          f"{what}: kind, no void, and the rows block with the spec's sentence word for word")
    ed = r["edit"]
    check(set(ed) == {"id", "terms", "note", "exact", "saved"} and set(ed["exact"]) == {repr(float(d)) for d in plan["doses"]}
          and all(set(x) == {"input_side", "output_add", "folded_form", "factors"} for x in ed["exact"].values()),
          f"{what}: the edit block: id, report, note, what each dose's cast does, where it was saved")
    check(len(r["items"]) == n_items, f"{what}: one entry an item ({len(r['items'])} of {n_items})")
    n0, C = len(plan["null_e"]), plan["controls"]["count"]
    for e in r["items"]:
        name = e["item"].get("name")
        ok = set(e) == {"item", "status", "why", "gate", "classes", "stability"} and set(e["item"]) == ITEM_BLOCK
        it = e["item"]
        q = next((x for x in QA if x[0] == name), None)
        ok = ok and it["pieces"] == len(it["prompt_ids"]) + len(it["answer_ids"]) and it["prompt_pieces"] == len(it["prompt_ids"]) and it["scored"] == len(it["answer_ids"]) \
            and it["id"] == sha_obj({"v": 1, "prompt_ids": it["prompt_ids"], "answer_ids": it["answer_ids"]}) \
            and (q is None or ((it["question"], it["answer"], it["system"]) == q[1:] and it["prompt_ids"] == lv.chat_ids(box.tok, q[1], q[3])))
        check(ok, f"{what}, {name}: the entry's keys, and the item block with its question, answer and ids")
        g = e["gate"]
        check(set(g) == {"bar", "small", "large", "passed", "caution"} and g["bar"] == plan["bars"]["item_gate"] and all(set(g[c]) == {"mean", "median", "worst"} for c in ("small", "large"))
              and g["passed"] == (g["small"]["mean"] >= g["bar"] and g["large"]["mean"] >= g["bar"])
              and g["caution"] == (None if g["passed"] else s_caution(g["small"]["mean"], g["large"]["mean"], small, large, g["bar"])),
              f"{what}, {name}: the gate's figure for both classes beside the bar, and its caution in words where either is under it")
        # the gate refuses no item since 16 November 2026 (the spec's open point 4, set from the card check): an item is read
        check(e["status"] in ("ok", "unstable"), f"{what}, {name}: the item was read, ok or unstable, whatever its gate's figure ({e['status']})")
        if e["status"] == "refused":
            continue
        for cls, R in (("small", small), ("large", large)):
            c = e["classes"][cls]
            keys = lv.layout(plan["doses"], C, plan["null_e"], R)
            ok = set(c) == CLASS and c["rows"] == R and c["forwards"] == len(keys) and c["peak_gib"] is None and c["lower"] == plan["lower"]
            check(ok, f"{what}, {name}, {cls}: the class block's keys, the planned rows ({R}) and forwards ({len(keys)})"
                  + ("" if ok else f" (got rows {c.get('rows')}, forwards {c.get('forwards')}, keys missing {sorted(CLASS - set(c))}, more {sorted(set(c) - CLASS)})"))
            arms = c["arms"]
            want_keys = [k for k in keys[0] if k] + [k for f in keys[1:] for k in f if k and not k.startswith("null:")]
            ok = [a["key"] for a in arms] == want_keys and all(ARM_MUST <= set(a) <= ARM_ANY for a in arms) \
                and all(keys[a["forward"]][a["row"]] == a["key"] and 1 <= a["row"] <= R - 2 for a in arms) \
                and all(("e" in a and "factor" in a and a["factor"] == we.null_factor(a["e"], box.dtype)) if a["what"] == "null" else ("dose" in a and "edit_id" in a) for a in arms) \
                and all(a["what"] == a["key"].split(":")[0] for a in arms)
            check(ok, f"{what}, {name}, {cls}: every null row, dose and control edit is an arm, in the layout's forward and row, each forward with its own base row")
            check(all(a["metrics"] is not None and all(set(st) == STATS for st in a["metrics"].values()) and a["non_finite"] is False and isinstance(a["same_as_base"], bool) for a in arms)
                  and all(set(fl) == FLOOR for fl in c["floor"].values()) and set(c["base"]) == {"own", "odds"} and set(c["base"]["own"]) == {"mean", "median", "worst"},
                  f"{what}, {name}, {cls}: each arm's read-outs are <stats>; the floor and the base row's own figures")
            check(c["untouched_rows_identical"] is True and c["repeats_identical"] == (True if len(keys) > 1 else None),
                  f"{what}, {name}, {cls}: untouched_rows_identical is true; repeats_identical is true with later forwards, None with one ({c['repeats_identical']})")
            check((c["rank"] is None) == (C == 0) and (c["rank"] is None or set(c["rank"]) == RANK), f"{what}, {name}, {cls}: the rank block (None with no control edit)")
            fo = c["folded"]
            check(isinstance(fo, dict) and isinstance(fo.get("taken"), bool) and ((set(fo) == FOLDED) if fo["taken"] else (isinstance(fo.get("why"), str) and set(fo) <= FOLDED
                  and all(fo[k] is None for k in fo if k not in ("taken", "why")))), f"{what}, {name}, {cls}: the folded block ({ {k: fo.get(k) for k in ('taken', 'why')} })")
        st = e["stability"]
        on = [m for m in ("kl", "own", "odds") if (plan["readouts"][m] if m != "odds" else plan["readouts"]["odds"] is not None)]
        check(set(st) == {"ratio_bar", "base_rows_same_bits", "metrics", "disagree"} and set(st["metrics"]) == set(on) and isinstance(st["base_rows_same_bits"], bool)
              and all(set(v) == {"arms", "disagree", "controls"} for v in st["metrics"].values()), f"{what}, {name}: stability holds every read-out that is on ({', '.join(on)})")
        # the rule on the edit's arms alone (16 November 2026, the spec's open point 5): the edit's arms in "arms", judged;
        # every control edit in "controls", its two means and ratio and no verdict; only an arm of the edit in "disagree"
        doses = {f"edit:{float(d)!r}" for d in plan["doses"]}
        check(all(set(v["arms"]) == doses and set(v["controls"]) == {f"control:{i}" for i in range(C)}
                  and all(set(a) == {"small", "large", "ratio"} for a in v["controls"].values()) and set(v["disagree"]) <= doses for v in st["metrics"].values())
              and all(str(k).startswith("edit:") for _m, k in st["disagree"]) and (e["status"] == "unstable") == bool(st["disagree"]),
              f"{what}, {name}: stability judges the edit's arms alone and keeps each control edit's ratio beside them, unjudged ({e['status']})")


def summary_shape(r: Dict[str, Any], what: str) -> None:
    """The summary against the items it sums (spec 4.9)."""
    s, plan = r["summary"], r["plan"]
    kept = [e for e in r["items"] if e["status"] == "ok"]
    same = [e for e in r["items"] if e["stability"] and e["stability"]["base_rows_same_bits"]]
    counts = {"asked": len(r["items"]), "kept": len(kept), "refused": sum(e["status"] == "refused" for e in r["items"]), "unstable": sum(e["status"] == "unstable" for e in r["items"]),
              "under_bar": sum(bool((e["gate"] or {}).get("caution")) for e in r["items"])}  # read with the gate's caution (16 November 2026)
    # 4.9: an item whose base rows are the same bits "keeps the status its arms give it ... and is counted apart": whatever its status
    check(set(s) == {"items", "classes"} and set(s["items"]) == set(counts) | {"one_arithmetic"} and all(s["items"][k] == v for k, v in counts.items())
          and s["items"]["one_arithmetic"] == len(same),
          f"{what}: the summary counts the items asked, kept, refused, unstable, read in one arithmetic and read under the gate's bar ({s.get('items')})")
    if not kept:
        check(s["classes"] is None or all(s["classes"][c] is None for c in ("small", "large")), f"{what}: with no item kept the summary's classes are None")
        return

    def middles(xs: List[float]) -> List[float]:
        xs = sorted(xs)
        return [xs[(len(xs) - 1) // 2], (xs[(len(xs) - 1) // 2] + xs[len(xs) // 2]) / 2]

    for cls in ("small", "large"):
        sc = s["classes"][cls]
        ok = set(sc) == {"rows", "arms", "floor", "rank"} and sc["rows"] == plan["rows"][cls]
        per = [{a["key"]: a for a in e["classes"][cls]["arms"]} for e in kept]
        wanted = [k for k in per[0] if not k.startswith("null:")]
        ok = ok and set(wanted) <= set(sc["arms"])
        for key in (k for k in sc["arms"] if k in per[0]):
            for m, got in sc["arms"][key].items():
                means, meds = [p[key]["metrics"][m]["mean"] for p in per], [p[key]["metrics"][m]["median"] for p in per]
                ok = ok and set(got) == {"mean_over_items", "median_over_items", "median_of_medians", "above_1_total"} \
                    and close(got["mean_over_items"], sum(means) / len(means), 1e-9) and any(close(got["median_over_items"], x, 1e-9) for x in middles(means)) \
                    and any(close(got["median_of_medians"], x, 1e-9) for x in middles(meds)) and got["above_1_total"] == sum(p[key]["metrics"][m]["above_1"] for p in per)
        for m, got in sc["floor"].items():
            got = got["mean_abs"] if isinstance(got, dict) else got
            ok = ok and close(got, max(e["classes"][cls]["floor"][m]["mean_abs"] for e in kept), 1e-9)
        if plan["controls"]["count"]:
            m, d = plan["rank_by"], repr(float(plan["main_dose"]))
            mine = abs(sc["arms"][f"edit:{d}"][m]["mean_over_items"])
            theirs = [abs(sc["arms"][f"control:{i}"][m]["mean_over_items"]) for i in range(plan["controls"]["count"])]
            at_least = sum(1 for c in theirs if c >= mine)
            rk = sc["rank"]
            ok = ok and set(rk) == {"metric", "dose", "value", "at_least", "non_finite", "of", "rank"} and close(rk["value"], mine, 1e-9) \
                and (rk["metric"], rk["dose"], rk["at_least"], rk["non_finite"], rk["of"], rk["rank"]) == (m, plan["main_dose"], at_least, 0, len(theirs) + 1, at_least + 1)
        check(ok, f"{what}, {cls}: the summary over the {len(kept)} kept items: each arm's mean and medians over items, the largest floor, the rank")


def figures(r: Dict[str, Any]) -> Dict[Tuple[Any, ...], Any]:
    """Every figure a reading read, by where it sits (to set two readings of one plan side by side)."""
    out: Dict[Tuple[Any, ...], Any] = {}
    for e in r["items"]:
        name = e["item"]["name"]
        for cls in ("small", "large"):
            out[(name, cls, "gate")] = e["gate"][cls]["mean"]
            c = (e["classes"] or {}).get(cls)
            for a in (c or {}).get("arms", []):
                for m, st in (a["metrics"] or {}).items():
                    for k, v in st.items():
                        out[(name, cls, a["key"], m, k)] = v
            for m, fl in (c or {}).get("floor", {}).items():
                out[(name, cls, "floor", m)] = fl["mean_abs"]
    return out


def lines_say(r: Dict[str, Any], what: str) -> None:
    """What the printed lines must give of a paired reading that was read (5.3), where the spec fixes the words."""
    text = " ".join(" ".join(lv.lines(r)).split())
    plan = r["plan"]
    check(isinstance(lv.lines(r), list) and all(isinstance(x, str) for x in lv.lines(r)), f"{what}: lines() gives a list of strings")
    check(r["name"] in text and "paired" in text and r["edit"]["id"][:12] in text, f"{what}: the lines give the reading's name, its kind and the edit's first 12")
    L = min(find_key(r["edit"]["terms"], "layer") or [0])
    check(f"layer {L} (page {L + 1})" in text or f"layer {L} (page layer {L + 1})" in text, f"{what}: the lines name the layer both ways (layer {L}, page {L + 1})")
    for label, sentence in (("CONTROLS_SAY", S_CONTROLS.format(seed=plan["controls"]["seed"])), ("ROWS_SAY", S_ROWS_SAY.format(**plan["rows"])), ("PAIRED_SAY", S_PAIRED)):
        n = text.count(" ".join(sentence.split()))
        check(n == 1 or (label == "CONTROLS_SAY" and plan["controls"]["count"] == 0 and n <= 1), f"{what}: the lines carry {label} once ({n} times)")
    check(all(e["item"]["name"] in text for e in r["items"]) and f"{plan['rows']['small']} rows" in text and f"{plan['rows']['large']} rows" in text,
          f"{what}: the lines name each item and the two row counts by their numbers")
    kept = [e for e in r["items"] if e["status"] == "ok"]
    arm_lines = [x for x in lv.lines(r) if "the edit at dose" in x and "mean" in x]
    check(len(arm_lines) >= 2 * len(kept) * len(plan["doses"]) and all("median" in x and "above a nat" in x for x in arm_lines),
          f"{what}: every line of the edit's figures carries the median and the count above a nat beside the mean ({len(arm_lines)} lines)")
    check(text.count(" ".join(S_FLOOR.format(n0=len(plan["null_e"])).split())) == 1 and all(e["item"]["id"][:12] in text for e in r["items"])
          and f"{len(kept)} of {len(r['items'])} item" in lv.lines(r)[0],
          f"{what}: the lines say once that the floor is no bound, name each item by its id's first 12 too, and say on the first line how many items were kept")
    if plan["controls"]["count"]:
        check("unmatched control edits read as much" in text, f"{what}: a rank line carries the word unmatched")
        # the spec's open point 5, its second clause (16 November 2026): the control edits' ratios printed, not judged
        on = [m for m in ("kl", "own", "odds") if plan["readouts"][m]]
        want = sum(len(on) for e in r["items"] if e.get("stability"))
        n = text.count("the control edits' ratios on ")
        firsts = [" ".join(x.split()) for x in lv.lines(r) if "the control edits' ratios on " in x]
        # two decimals each (three figures under a thousandth: a pair under the floor reads its larger over the floor),
        # in the order of the control edits
        words = {"kl": "KL", "own": "the text's own log-probability", "odds": "log-odds of the named pieces"}
        shown = [(m, ", ".join("not finite" if c["ratio"] is None else f"{c['ratio']:.3g}" if 0 < c["ratio"] < 1e-3 else f"{c['ratio']:.2f}"
                               for _k, c in sorted(e["stability"]["metrics"][m]["controls"].items(), key=lambda kc: int(kc[0].split(":")[1]))))
                 for e in r["items"] if e.get("stability") for m in on]
        check(n == want and all("printed and not judged (the rule reads the edit's arms alone), from control edit 0: " in x for x in firsts)
              and all(any(f"ratios on {words[m]} " in x and f"from control edit 0: {s};" in x for x in firsts) for m, s in shown),
              f"{what}: every item's lines print the control edits' ratios on each read-out, from control edit 0, as not judged ({n} of {want})")


class Planted:
    """A forward hook of the check's own that does nothing until a mode is set. It is hung on one layer's feed-forward
    block, or on the text model's final norm (whose output is the final-normed states every rule and read-out reads),
    BEFORE the handle that reads under it is made, so that the handle's census of hooks at load holds it and the base
    assertion passes. Modes, each acting on the rows named by `row` (one row or several):
      "move"     writes 3.0 into every second dimension (not a scaling: the norm after the block would undo one);
      "all"      the same, into every row alike;
      "nan"      writes a number that is not finite; "nan_all" into every row;
      "copy"     makes the rows the same bits as row 0 (on the final norm);
      "reflect"  puts each row on the other side of row 0 (2 * row 0 - row; on the final norm);
      "sharpen"  multiplies every row alike by `by` (on the final norm: the stand-in's logits come out that many times
                 as sharp, so the log-probability of the text's own pieces falls and spreads, with no row parting);
      "raise"    raises.
    `rows` and `pieces`: only in forwards of that many rows and pieces. `only`: only in the n-th such forward, or the
    n-th and m-th (counted from 0 since the mode was set). `seen` keeps the (rows, pieces) of every forward since."""

    def __init__(self) -> None:
        self.mode: Optional[str] = None
        self.row: Any = -1
        self.rows, self.pieces, self.only, self.fired, self.calls, self.by = None, None, None, 0, 0, 4.0
        self.seen: List[Tuple[int, int]] = []

    def set(self, mode: Optional[str], row: Any = -1, rows: Optional[int] = None, pieces: Optional[int] = None, only: Any = None, by: float = 4.0) -> None:
        self.mode, self.row, self.rows, self.pieces, self.fired, self.calls, self.by = mode, row, rows, pieces, 0, 0, float(by)
        self.only = None if only is None else ({int(only)} if isinstance(only, int) else {int(x) for x in only})
        self.seen = []

    def __call__(self, _mod: Any, _args: Any, out: torch.Tensor) -> Optional[torch.Tensor]:
        self.seen.append((int(out.shape[0]), int(out.shape[1])))
        if self.mode is None or self.mode == "look":
            return None
        if self.mode == "raise":
            self.fired += 1
            raise RuntimeError("planted by the check")
        if (self.rows is not None and out.shape[0] != self.rows) or (self.pieces is not None and out.shape[1] != self.pieces):
            return None
        self.calls += 1
        if self.only is not None and self.calls - 1 not in self.only:
            return None
        self.fired += 1
        o = out.clone()
        rows = list(self.row) if isinstance(self.row, (list, tuple)) else [self.row]
        if self.mode == "move":
            o[rows, :, ::2] = 3.0
        elif self.mode == "all":
            o[:, :, ::2] = 3.0
        elif self.mode == "nan":
            o[rows] = float("nan")
        elif self.mode == "nan_all":
            o[:] = float("nan")
        elif self.mode == "copy":
            o[rows] = o[0].clone()
        elif self.mode == "sharpen":
            o *= self.by
        elif self.mode == "reflect":
            o[rows] = 2 * o[0] - o[rows]
        else:
            raise ValueError(f"the check's planted hook has no mode {self.mode!r}")
        return o


def check_end_to_end(Fb: Box, Pb: Box, hF: Any, hP: Any) -> None:
    plans = TMP / "plans10"
    ctl = {"count": 5, "seed": 0}
    rows = {"small": 8, "large": 16}  # 8 rows: 2 arm slots, so 6 arms are 3 forwards; 16 rows: 10 slots, 1 forward
    bars = {"item_gate": GATE}
    sil = {"terms": [{"layer": 2, "silence": [3, 40]}], "note": "the check's silencing"}
    by15 = {"terms": [{"layer": 2, "scale": [3, 40], "factor": 1.5}]}
    paired = {"kind": "paired", "controls": ctl, "rows": rows, "bars": bars}
    with part("10 (F): a scale by 1.5, same_shape and whole"):
        path = plan_file(plans, "f", [{**paired, "name": "F scale", "edit": by15}, {**paired, "name": "F scale, whole", "edit": by15, "lower": "whole"}])
        r, r_whole = read_all(hF, path, Fb)
        KEPT["f"] = (r, by15, path.parent)
        check(r["status"] == "ok" and r["bars_overridden"] is True and r["base_after"] is True and [e["status"] for e in r["items"]] == ["ok"] * 3,
              f"10 (F): the reading and its three items are ok, bars_overridden and base_after true ({r['status']}: {r['why']}; {[e['status'] for e in r['items']]})")
        top_shape(r, Fb, "10 (F)")
        paired_shape(r, Fb, "10 (F)", 3)
        summary_shape(r, "10 (F)")
        lines_say(r, "10 (F)")
        why = we.can_fold(Fb.lm, Fb.build(*we.from_recipe(by15, plans)))
        check(all(e["classes"][c]["folded"]["taken"] is False and e["classes"][c]["folded"]["why"] == why for e in r["items"] for c in ("small", "large")),
              "10 (F): no twin is taken and folded.why is can_fold's sentence")
        check(all(e["classes"]["small"]["restart_from"] == 2 and e["classes"]["large"]["restart_from"] is None for e in r["items"])
              and all(e["classes"][c]["restart_from"] is None for e in r_whole["items"] for c in ("small", "large")),
              "10 (F): the class with later forwards restarts from the edit's layer; a class of one forward, and every class under lower \"whole\", does not")
        a, b = figures(r), figures(r_whole)
        check(r_whole["status"] == "ok" and set(a) == set(b) and all(close(a[k], b[k]) for k in a), "10 (F): lower \"whole\" gives the same figures as \"same_shape\", to 1e-5")
        for e in r["items"]:
            it = lv.plan_items(lv.load_plan(path), Fb.tok)[e["item"]["name"]]
            P, end, _y = span_of(it)
            mine = torch.equal(Fb.whole(it.ids, 8)[0][P + [end]], Fb.whole(it.ids, 16)[0][P + [end]])
            check(e["stability"]["base_rows_same_bits"] == mine, f"10 (F), {it.name}: base_rows_same_bits is the check's own comparison of two unhooked forwards at 8 and 16 rows ({mine})")
        check(r["edit"]["saved"] is None, "10 (F): nothing was saved unasked")
    with part("10 (F): an edit built in Python, saved on request"):
        items = items_of(Fb)
        e = Fb.build(we.silence(2, [3, 40]), note="built by the check")
        r = lv.read(hF, {**paired, "name": "F built", "edit": sil, "items": ["orange"]}, items, edit=e, save_edit=True)
        printed(r)
        check(r["status"] == "ok" and r["edit"]["id"] == e.id and r["plan"]["edit"] == {"built": e.id}, f"10 (F): an Edit handed to read is read in place of the recipe, the plan recording it ({r['status']}: {r['why']})")
        saved = Path(r["edit"]["saved"]) if r["edit"]["saved"] else None
        check(saved is not None and (saved / "manifest.json").exists() and json.loads((saved / "manifest.json").read_text())["id"] == e.id and ROOT_DIR not in saved.parents,
              "10 (F): save_edit saves the edit and names its folder in the result")
        check(all(c["folded"]["taken"] is False and c["folded"]["why"] == S_EXACT for c in r["items"][0]["classes"].values()), "10 (F): a silencing takes no twin: exact on the input side")
    with part("10 (F): refusals are results, in the ledger"):
        items = items_of(Fb)
        n_before, _h = lb.head()
        r = lv.read(hF, {**paired, "name": "a refused edit", "edit": {"terms": [{"layer": 2, "silence": [500]}]}}, items)
        printed(r)
        check(r["status"] == "refused" and r["why"] == f"Neuron 500 does not exist: layer 2 (page layer 3) has {N} (0 to {N - 1})." and r["edit"] is None
              and r["items"] == [] and r["base_after"] is True, f"10: a refused edit is a result with the gate's words ({r['status']}: {r['why']!r})")
        top_shape(r, Fb, "10: a refused edit")
        given = {**paired, "name": "a refused plan", "edit": sil, "rows": {"small": 4, "large": 16}}
        r = lv.read(hF, copy.deepcopy(given), items)
        printed(r)
        check(r["status"] == "refused" and r["why"] == s_rows_range(4, 16) and r["plan"] == given, f"10: a refused plan is a result with normalise's words, its plan as given ({r['status']}: {r['why']!r})")
        top_shape(r, Fb, "10: a refused plan")
        r = lv.read(hF, {"name": "neither kind", "kind": "sampled", "edit": sil}, items)
        check(r["status"] == "refused" and r["kind"] == "sampled" and book_entry(r["book"]["seq"])["kind"] == "note",
              f"10: a reading whose kind is neither is refused and entered as a note ({r['status']}, kind {r['kind']!r})")
        check(lb.head()[0] == n_before + 3 and lb.verify()["ok"] is True, "10: each of the three is in the ledger, and the chain is whole")
    with part("10 (F): every item under the bar"):
        path = plan_file(plans, "f_gate", [{**paired, "name": "the bar at -0.5", "edit": sil, "bars": {"item_gate": -0.5}}])
        (r,) = read_all(hF, path, Fb)
        # since 16 November 2026 the gate is a printed caution (the spec's open point 4, set from the card check): every
        # item is read at both row counts all the same, with its caution in words, and none is refused
        ok = r["status"] == "ok" and len(r["items"]) == 3
        for e in r["items"]:
            g = e["gate"]
            ok = ok and e["status"] in ("ok", "unstable") and g["passed"] is False and g["bar"] == -0.5 and all(g[c]["mean"] < -0.5 for c in ("small", "large")) \
                and all(e["classes"][c] is not None for c in ("small", "large")) and e["stability"] is not None \
                and g["caution"] == s_caution(g["small"]["mean"], g["large"]["mean"], 8, 16, -0.5) and "both row counts" in g["caution"]
        check(ok, f"10: with the bar at -0.5 every item is read at both row counts all the same, with both classes' figures and the gate's caution in its words "
              f"({r['status']}; {(r['items'][0]['gate'] or {}).get('caution') if r['items'] else None!r})")
        n_ok = sum(e["status"] == "ok" for e in r["items"])
        check(r["summary"]["items"]["refused"] == 0 and r["summary"]["items"]["under_bar"] == 3 and r["summary"]["items"]["kept"] == n_ok,
              f"10: and the summary counts them under the bar, none refused, the ok ones kept ({r['summary']['items']})")
        text = " ".join(" ".join(lv.lines(r)).split())
        check(all(" ".join(str((e["gate"] or {}).get("caution")).split()) in text for e in r["items"]) and "3 under the item gate's bar and read with its caution" in text
              and "before the gate refused" not in text, "10: and the lines print each item's caution, and the count of them beside the summary")
        paired_shape(r, Fb, "10: every item under the bar", 3)
        summary_shape(r, "10: every item under the bar")
        top_shape(r, Fb, "10: every item under the bar")
    with part("10 (F): the ledger broken before the model is touched"):
        items = items_of(Fb)
        torn = TMP / "torn_for_read.jsonl"
        torn.write_bytes(b'{"v":1,"seq":0')
        kept = os.environ[BOOK_VAR]
        os.environ[BOOK_VAR] = str(torn)
        try:
            got = words_of(lambda: lv.read(hF, {**paired, "name": "never read", "edit": sil}, items), lb.BookBroken)
        finally:
            os.environ[BOOK_VAR] = kept
        check(got is not None and not got.startswith("<") and torn.read_bytes() == b'{"v":1,"seq":0', f"10: read raises BookBroken when the ledger cannot take an entry, and writes nothing ({got!r})")
    with part("10 (P): a set takes the twin, a silencing does not"):
        np.save(plans / "set.npy", vec(71, 2, D))
        set_e = {"terms": [{"layer": 2, "set": [3, 40], "vectors": "set.npy"}]}
        path = plan_file(plans, "p", [{**paired, "name": "P set", "edit": set_e}, {**paired, "name": "P silence", "edit": sil},
                                      {**paired, "name": "P silence, whole", "edit": sil, "lower": "whole"}], items=(0, 1))
        r, r_sil, r_whole = read_all(hP, path, Pb)
        KEPT["p"] = (r, set_e, path.parent)
        check(r["status"] == "ok" and r_sil["status"] == "ok" and all(e["status"] == "ok" for e in r["items"] + r_sil["items"]),
              f"10 (P): both readings and their items are ok ({r['status']}: {r['why']}; {r_sil['status']}: {r_sil['why']})")
        top_shape(r, Pb, "10 (P)")
        paired_shape(r, Pb, "10 (P)", 2)
        summary_shape(r, "10 (P)")
        lines_say(r, "10 (P)")
        paired_shape(r_sil, Pb, "10 (P), a silencing", 2)
        edit = Pb.build(*we.from_recipe(set_e, plans))
        for e in r["items"]:
            sm, lg = e["classes"]["small"]["folded"], e["classes"]["large"]["folded"]
            check(sm["taken"] is True and lg["taken"] is True and sm["restart"] is True and lg["restart"] is False,
                  f"10 (P), {e['item']['name']}: the twin is taken for a set, restarting where the class has later forwards (8 rows) and whole where it has one (16 rows)")
            ok = True
            for cls, fo in (("small", sm), ("large", lg)):
                arm = next(a for a in e["classes"][cls]["arms"] if a["key"] == "edit:1.0")
                ok = ok and fo["dose"] == 1.0 and fo["base_from"] == "forward 0 of the same shape" and fo["rows_identical"] is True and fo["per_row"] == arm["metrics"] \
                    and isinstance(fo["same_bits"], bool) and fo["seconds"] >= 0 and set(fo["metrics"]) == set(arm["metrics"]) == set(fo["gap"])
                for m in fo["metrics"]:
                    a, b = fo["metrics"][m]["mean"], arm["metrics"][m]["mean"]
                    ok = ok and set(fo["gap"][m]) == {"abs", "rel"} and close(fo["gap"][m]["abs"], abs(a - b), 1e-9) \
                        and close(fo["gap"][m]["rel"], abs(a - b) / max(abs(a), abs(b)) if max(abs(a), abs(b)) else None, 1e-9)
                ok = ok and arm["same_as_base"] is False and e["classes"][cls]["folded"]["between_kl_mean"] is not None and fo["between_kl_mean"] < 1e-4
            check(ok, f"10 (P), {e['item']['name']}: the twin's block: its base row from forward 0, its figures beside the per-row arm's, their gap, the KL between the two rows under 1e-4")
            it = lv.plan_items(lv.load_plan(path), Pb.tok)[e["item"]["name"]]
            with we.folded(Pb.lm, edit, 1.0):
                hf = Pb.whole(it.ids, 16)
            P, _end, _y = span_of(it)
            l0, l1 = tables(Pb, Pb.whole(it.ids, 16)[0], P), tables(Pb, hf[0], P)
            check(close(lg["metrics"]["kl"]["mean"], float((l0.exp() * (l0 - l1)).sum(-1).mean())),
                  f"10 (P), {e['item']['name']}: the twin's KL at 16 rows is the check's own folded forward read against an unhooked base row")
            mine = torch.equal(Pb.whole(it.ids, 8)[0][P + [_end]], Pb.whole(it.ids, 16)[0][P + [_end]])
            check(e["stability"]["base_rows_same_bits"] == mine, f"10 (P), {it.name}: base_rows_same_bits is the check's own comparison ({mine})")
        check(all(e["classes"][c]["folded"]["taken"] is False and e["classes"][c]["folded"]["why"] == S_EXACT for e in r_sil["items"] for c in ("small", "large")),
              "10 (P): no twin for a silencing: exact on the input side")
        check(r_whole["status"] == "ok" and figures(r_sil) == figures(r_whole) and len(figures(r_sil)) > 0, "10 (P): lower \"whole\" gives the same figures as \"same_shape\", exactly")
        used = set(find_key(r["edit"]["exact"], "output_add"))
        check(used == {True} and set(find_key(r_sil["edit"]["exact"], "output_add")) == {False}, "10 (P): the result marks the set as using the output-side add, the silencing not")
    with part("10 (P): a dose the rounding eats, and the one-row capture"):
        items = items_of(Pb, (0, 1))
        r = lv.read(hP, {**paired, "name": "dose 0.001", "edit": sil, "doses": [0.001]}, items)
        printed(r)
        check(r["status"] == "refused" and r["why"] == we.dose_refusal(Pb.build(*we.from_recipe(sil, plans)), 0.001, Pb.dtype) and r["why"] is not None
              and book_entry(r["book"]["seq"])["body"]["status"] == "refused", f"10 (P): a silencing at dose 0.001 is refused with dose_refusal's sentence, and entered ({r['status']}: {r['why']!r})")
        r = lv.read(hP, {**paired, "name": "one row, refused", "edit": sil, "lower": "one_row"}, items)
        printed(r)
        check(r["status"] == "refused" and r["why"] == s_one_row(44, 2), f"10 (P): lower \"one_row\" is refused below bars.one_row_min_layer ({r['status']}: {r['why']!r})")
        r = lv.read(hP, {**paired, "name": "one row, the bar at 2", "edit": sil, "lower": "one_row", "bars": {"item_gate": GATE, "one_row_min_layer": 2}}, items)
        printed(r)
        check(r["status"] == "ok" and all(e["classes"][c]["lower"] == "one_row" and e["classes"][c]["restart_from"] == 2 for e in r["items"] for c in ("small", "large")),
              f"10 (P): with that bar given as 2 the reading is taken and marks both classes one_row ({r['status']}: {r['why']})")
        check("one row" in " ".join(" ".join(lv.lines(r)).split()), "10 (P): and its lines say the layers below the edit ran at one row")
        r0 = lv.read(hP, {**paired, "name": "a zero row", "edit": sil, "doses": [1.0, 0.0]}, items)
        printed(r0)
        zero = [a for e in r0["items"] for c in e["classes"].values() for a in c["arms"] if a["key"] == "edit:0.0"]
        check(r0["status"] == "ok" and len(zero) == 4 and all(a["same_as_base"] is True for a in zero), f"10 (P): a listed dose of 0 is carried and reads as the base row by bits ({r0['status']}: {r0['why']})")
    with part("10 (P): planted faults"):
        planted = Planted()
        hook = Pb.lm.layers[4].mlp.register_forward_hook(planted)
        try:
            h2 = lv.handle(Pb.rt)
            check(h2.hooks_at_load == {"layers.4.mlp": [hook.id]}, f"10: a handle made with the check's hook in place holds it in its census ({h2.hooks_at_load})")
            items = items_of(Pb, (0, 2))
            lens = {n: len(i.ids) for n, i in items.items()}
            check(lens["orange"] != lens["fox"], "10: (the two items are of different lengths, which the planted hook tells them apart by)")
            base_reading = {**paired, "name": "planted", "edit": sil}
            r = lv.read(h2, {**base_reading, "name": "planted: nothing moved"}, items)
            check(r["status"] == "ok" and r["base_after"] is True and r["runtime"]["hooks_at_load"] == 1, f"10: with the hook quiet the reading is ok and the model the base after ({r['status']}: {r['why']})")
            planted.set("move", row=-1)
            r = lv.read(h2, {**base_reading, "name": "planted: an untouched row moved"}, items)
            printed(r)
            v = r["void"] or {}
            check(r["status"] == "void" and set(v) == {"code", "item", "class", "forward"} and v["code"] == "untouched_rows_differ" and v["class"] == "small" and v["forward"] == 0
                  and isinstance(r["why"], str) and r["base_after"] is True and planted.fired >= 1,
                  f"10: a hook that moves one untouched row makes the reading void, untouched_rows_differ, at the first forward ({r['status']}, {v})")
            check(book_entry(r["book"]["seq"])["body"]["status"] == "void", "10: and the void reading is in the ledger")
            zero_row = lv.layout([1.0, 0.0], 5, list(lv.NULL_E), 8)[0].index("edit:0.0")
            planted.set("move", row=zero_row)
            r = lv.read(h2, {**base_reading, "name": "planted: the dose-0 arm moved", "doses": [1.0, 0.0]}, items)
            printed(r)
            check(r["status"] == "void" and (r["void"] or {}).get("code") == "zero_dose_differs" and r["base_after"] is True,
                  f"10: a hook that moves a dose-0 arm makes the reading void, zero_dose_differs ({r['status']}, {r['void']})")
            planted.set("raise")
            n_before, _h = lb.head()
            got = words_of(lambda: lv.read(h2, {**base_reading, "name": "planted: a hook that raises"}, items), RuntimeError)
            planted.set(None)
            last = book_entry(lb.head()[0] - 1)["body"]
            check(got == "planted by the check" and lb.head()[0] == n_before + 1 and last["name"] == "planted: a hook that raises" and last["status"] == "void"
                  and last["void"]["code"] == "error" and "RuntimeError" in last["why"] and "planted by the check" in last["why"] and last["base_after"] is True,
                  f"10: a hook that raises inside a forward is entered as void, error, with base_after true, and raised again out of read ({got!r}; {last.get('status')}, {last.get('void')})")
            check(words_of(lambda: we.assert_base(Pb.lm, h2.hooks_at_load), we.NotBase) is None, "10: and the model is the base after it")
            edit_row = lv.layout([1.0], 5, list(lv.NULL_E), 16)[0].index("edit:1.0")
            planted.set("move", row=edit_row, rows=16, pieces=lens["fox"])
            r = lv.read(h2, {**base_reading, "name": "planted: the large batch alone"}, items)
            printed(r)
            planted.set(None)
            by = {e["item"]["name"]: e for e in r["items"]}
            check(r["status"] == "ok" and by["fox"]["status"] == "unstable" and by["orange"]["status"] == "ok" and isinstance(by["fox"]["why"], str)
                  and ["kl", "edit:1.0"] in [list(x) for x in by["fox"]["stability"]["disagree"]] and by["fox"]["stability"]["metrics"]["kl"]["arms"]["edit:1.0"]["agree"] is False,
                  f"10: a hook that acts only in the batch of 16 rows makes that item unstable ({r['status']}; {[(n, e['status']) for n, e in by.items()]})")
            s = r["summary"]
            kept_mean = by["orange"]["classes"]["small"]["arms"][4]["metrics"]["kl"]["mean"]
            check(s["items"]["asked"] == 2 and s["items"]["kept"] == 1 and s["items"]["unstable"] == 1 and s["items"]["refused"] == 0
                  and by["orange"]["classes"]["small"]["arms"][4]["key"] == "edit:1.0" and close(s["classes"]["small"]["arms"]["edit:1.0"]["kl"]["mean_over_items"], kept_mean, 1e-12),
                  f"10: and leaves it out of the summary, which holds the other item alone ({s['items']})")
            summary_shape(r, "10: one item unstable")
            # the spec's open point 5, its second clause (16 November 2026): a control edit's row that moves with the row
            # count is printed beside the edit's arms and not judged; the item reads as the edit's arms do
            ctl_row = lv.layout([1.0], 5, list(lv.NULL_E), 16)[0].index("control:2")
            planted.set("move", row=ctl_row, rows=16, pieces=lens["fox"])
            r = lv.read(h2, {**base_reading, "name": "planted: a control edit's row moved, the large batch alone"}, items)
            printed(r)
            fired = planted.fired
            planted.set(None)
            by = {e["item"]["name"]: e for e in r["items"]}
            st = by["fox"]["stability"] or {}
            c2 = ((st.get("metrics") or {}).get("kl", {}).get("controls") or {}).get("control:2") or {}
            fl = max(by["fox"]["classes"][c]["floor"]["kl"]["mean_abs"] for c in ("small", "large"))
            check(fired >= 1 and c2.get("ratio") is not None and c2["ratio"] > 3.0 and own_agree(c2["small"], c2["large"], fl, False)[0] is False,
                  f"10: (the check's own need: the moved control edit's two readings disagree by the spec's rule as it stood before: {c2}, floor {fl:.2e})")
            check(r["status"] == "ok" and by["fox"]["status"] == "ok" and by["fox"]["why"] is None and st.get("disagree") == [] and "agree" not in c2
                  and "control:2" not in st["metrics"]["kl"]["arms"] and by["orange"]["status"] == "ok" and r["summary"]["items"]["kept"] == 2 and r["summary"]["items"]["unstable"] == 0,
                  f"10: a hook that moves a control edit's row in the batch of 16 rows alone leaves the item ok and in the summary: the rule reads the edit's arms alone "
                  f"({r['status']}; {[(n, e['status']) for n, e in by.items()]}; {st.get('disagree')})")
            said = [" ".join(x.split()) for x in lv.lines(r)]
            at = next((i for i, x in enumerate(said) if x.startswith("item fox")), None)
            ctl_line = next((x for x in said[at:] if x.startswith("the control edits' ratios on KL")), "") if at is not None else ""
            check(at is not None and f"{c2.get('ratio') or 0:.2f}" in ctl_line and ctl_line.endswith("; 1 not finite or over the bar (control edit 2)")
                  and any(x.startswith("8 and 16 rows agree on KL for the edit's arms") for x in said[at:]),
                  f"10: and fox's lines say its edit's arms agree on KL and print control edit 2's ratio, the one over the bar, not judged ({ctl_line!r})")
            summary_shape(r, "10: a control edit's row moved")
            paired_shape(r, Pb, "10: a control edit's row moved", 2)
        finally:
            planted.set(None)
            hook.remove()
        check(words_of(lambda: we.assert_base(Pb.lm, hP.hooks_at_load), we.NotBase) is None, "10: with the check's hook gone the model is the base against the first handle's census")
    with part("10: the refusals 2.6 leaves to read itself, as results"):
        # 2.6's last list: they need the items or the loaded model, so normalise cannot make them (6.1 §8: every refusal of 2.6)
        items = items_of(Pb, (0, 1))
        refusals: List[Dict[str, Any]] = []
        mod = Pb.mod(4)
        mod._memit_col_overlays[7] = torch.zeros(D, dtype=Pb.dtype)  # as a vault edit would leave it
        try:
            r = lv.read(hP, {**paired, "name": "a vault edit in force", "edit": sil}, items)
        finally:
            mod._memit_col_overlays.pop(7, None)
        printed(r)
        refusals.append(r)
        check(r["status"] == "refused" and r["why"] == "A vault edit is in force on layers.4.mlp.down_proj (1 overlays): a reading is taken on the base model only. "
              "ROLLBACK the vault, or start a fresh process." and r["items"] == [] and r["base_after"] is False,
              f"10 (P): with a vault overlay in force a reading is refused in 2.6's words, and base_after is false ({r['status']}: {r['why']!r})")
        hook = Pb.lm.layers[3].mlp.register_forward_hook(lambda m, a, o: None)
        try:
            r = lv.read(hP, {**paired, "name": "a hook that was not there at load", "edit": sil}, items)
        finally:
            hook.remove()
        printed(r)
        refusals.append(r)
        check(r["status"] == "refused" and r["why"] == "The model carries hooks that were not there at load (on layers.3.mlp): a reading is taken on the base model only."
              and r["items"] == [] and r["base_after"] is False, f"10 (P): with a hook that was not there at load a reading is refused in 2.6's words ({r['status']}: {r['why']!r})")
        r = lv.read(hP, {**paired, "name": "after the overlay and the hook are gone", "edit": sil}, items)
        check(r["status"] == "ok" and r["base_after"] is True, f"10 (P): with both gone the next reading is ok ({r['status']}: {r['why']})")
        items_f = items_of(Fb)
        r = lv.read(hF, {**paired, "name": "folded always on a plain linear", "edit": by15, "folded": "always"}, items_f)
        printed(r)
        refusals.append(r)
        check(r["status"] == "refused" and r["why"] is not None and r["why"] == we.can_fold(Fb.lm, Fb.build(*we.from_recipe(by15, plans))),
              f"10 (F): under folded \"always\" a reading is refused with can_fold's sentence ({r['status']}: {r['why']!r})")
        long_item = {"long": lv.item_from_ids([2] + [300] * 99, list(range(301, 401)), name="long")}  # 200 pieces: 64 rows of it are 12,800 row-positions
        r = lv.read(hF, {**paired, "name": "too many row-positions", "edit": sil, "rows": {"small": 8, "large": 64}}, long_item)
        printed(r)
        refusals.append(r)
        check(r["status"] == "refused" and re.fullmatch(r"64 rows of 200 word-pieces is 12,?800 row-positions; 1a runs at most 12,288 in a forward \(the pilots measured to 11,264\)\.",
                                                        str(r["why"])) is not None, f"10 (F): a forward over 12,288 row-positions is refused as a result, in 2.6's words ({r['status']}: {r['why']!r})")
        n_prompt = len(items_f["orange"].prompt_ids)
        r = lv.read(hF, {"name": "too much written", "kind": "written", "edit": sil, "rows": 32, "max_new": 256, "controls": {"count": 0, "seed": 0}, "items": ["orange"]}, items_f)
        printed(r)
        refusals.append(r)
        gib = 0.8e6 * 32 * (n_prompt + 256) / 2**30
        check(r["status"] == "refused" and r["why"] == f"Writing 32 rows to {n_prompt + 256} word-pieces of context keeps about {gib:.1f} GiB above the model (0.8 MB a row a piece, "
              "measured 28 October 2026); 1a takes at most 6. Ask fewer rows or fewer new pieces.", f"10 (F): a written reading whose cache would not fit is refused as a result, in 2.6's words ({r['status']}: {r['why']!r})")
        check(len(refusals) == 5 and all(book_entry(x["book"]["seq"])["body"]["status"] == "refused" for x in refusals) and lb.verify()["ok"] is True,
              "10: and each of the five is in the ledger, the chain whole")
    with part("10 (P): the twin and the control edits at a main dose that is not 1"):
        # every other reading here has main dose 1, where a twin folded at dose 1 or control edits run at dose 1 would pass
        set_e = {"terms": [{"layer": 2, "set": [3, 40], "vectors": "set.npy"}]}
        items = items_of(Pb, (0, 1))
        r = lv.read(hP, {**paired, "name": "P set, main dose 0.5", "edit": set_e, "doses": [1.0, 0.5], "main_dose": 0.5}, items, base_dir=plans)
        printed(r)
        edit = Pb.build(*we.from_recipe(set_e, plans))
        controls = we.control_edits(edit, Pb.lm, 0, 5)
        check(r["status"] == "ok" and all(e["status"] == "ok" for e in r["items"]) and r["plan"]["main_dose"] == 0.5, f"10 (P): the reading is ok ({r['status']}: {r['why']})")
        for e in r["items"]:
            it = items[e["item"]["name"]]
            P, _end, _y = span_of(it)
            ok, apart, said = True, True, []
            for cls, R in (("small", 8), ("large", 16)):
                block = e["classes"][cls]
                fo = block["folded"]
                arm = next(a for a in block["arms"] if a["key"] == "edit:0.5")
                base = Pb.whole(it.ids, R)[0]
                with we.folded(Pb.lm, edit, 0.5):
                    hf = Pb.whole(it.ids, R)
                l0, l1 = tables(Pb, base, P), tables(Pb, hf[0], P)
                mine = float((l0.exp() * (l0 - l1)).sum(-1).clamp_min(0.0).mean())
                # 1e-9: the two are one arithmetic on one forward (they read the same bits on 14 November 2026); the gap
                # between the folded form and the per-row form is 1e-7 to 2e-6 here, and between doses 0.5 and 1, 3e-5
                ok = ok and fo["taken"] is True and fo["dose"] == 0.5 and fo["per_row"] == arm["metrics"] and close(fo["metrics"]["kl"]["mean"], mine, 1e-9)
                apart = apart and (fo["same_bits"] is True or fo["metrics"]["kl"]["mean"] != arm["metrics"]["kl"]["mean"])
                said.append(f"{fo['metrics']['kl']['mean']:.3e} folded, {mine:.3e} the check's, {arm['metrics']['kl']['mean']:.3e} per row")
                # a control arm against the check's own forward with that control edit at the main dose
                keys = lv.layout([1.0, 0.5], 5, r["plan"]["null_e"], R)
                c0 = next(a for a in block["arms"] if a["key"] == "control:0")
                if c0["forward"] == 0:
                    with we.per_row(Pb.lm, programs_of(keys[0], edit, controls, 0.5, 2)):
                        b = brute(Pb, Pb.whole(it.ids, R), c0["row"], it)
                    ok = ok and c0["dose"] == 0.5 and near(c0["metrics"]["kl"]["mean"], float(b["kl"].clamp_min(0.0).mean()), 1e-4, 1e-9)
            check(ok, f"10 (P), {it.name}: the twin's KL is the check's own forward folded at the main dose, 0.5, read against an unhooked base row, to 1e-9; "
                  f"a control edit's is the check's own at dose 0.5 ({'; '.join(said)})")
            check(apart, f"10 (P), {it.name}: where the folded row and the per-row row are not the same bits, the twin's figure is not the per-row arm's own")
    with part("10 (P): which forwards run the layers below the edit"):
        # on the CPU one row and many give the same bits, so no figure shows a stream kept from the wrong forward: look
        look = Planted()
        hook = Pb.lm.layers[0].mlp.register_forward_hook(look)  # layer 0, below the edit's layer 2
        try:
            h3 = lv.handle(Pb.rt)
            items = items_of(Pb, (0,))
            T = len(items["orange"].ids)
            for lower, want, what in (("same_shape", [(8, T), (16, T)], "once a class, at the class's own rows, and never at one row"),
                                      ("whole", [(8, T)] * 3 + [(16, T)], "in every forward"), ("one_row", [(1, T)], "once an item, at one row")):
                look.set("look")
                r = lv.read(h3, {**paired, "name": f"P looked at, {lower}", "edit": sil, "lower": lower, "bars": {"item_gate": GATE, "one_row_min_layer": 2}}, items)
                check(r["status"] == "ok" and look.seen == want, f"10 (P): under lower \"{lower}\" the layers below the edit run {what} ({look.seen})")
        finally:
            look.set(None)
            hook.remove()
    with part("10 (P): more planted faults, each rule that voids a reading by a fault of its own"):
        mlp4, fin = Planted(), Planted()
        hooks = [Pb.lm.layers[4].mlp.register_forward_hook(mlp4), Pb.lm.norm.register_forward_hook(fin)]
        try:
            h3 = lv.handle(Pb.rt)
            items = items_of(Pb, (0, 2))
            To, Tf = len(items["orange"].ids), len(items["fox"].ids)
            base_reading = {**paired, "name": "planted", "edit": sil}

            def plant(hook_: Planted, name: str, over: Optional[Dict[str, Any]] = None, **how: Any) -> Tuple[Dict[str, Any], int]:
                """One reading under one planted fault: (the result, how many forwards the fault acted in)."""
                hook_.set(**how)
                try:
                    r_ = lv.read(h3, {**base_reading, "name": f"planted: {name}", **(over or {})}, items, base_dir=plans)
                finally:
                    fired = hook_.fired
                    hook_.set(None)
                printed(r_)
                return r_, fired

            def text_of(r_: Dict[str, Any]) -> str:
                return " ".join(" ".join(lv.lines(r_)).split())

            r, n = plant(fin, "a base row that is not finite", mode="nan", row=0, rows=8, pieces=To, only=0)
            check(r["status"] == "void" and r["void"] == {"code": "non_finite_base", "item": "orange", "class": "small", "forward": 0} and r["base_after"] is True and n == 1,
                  f"10: a base row that is not finite makes the reading void, non_finite_base (rule 1) ({r['status']}, {r['void']})")
            r, n = plant(mlp4, "a null row moved in a later forward", mode="move", row=1, rows=8, pieces=To, only=1)
            check(r["status"] == "void" and r["void"] == {"code": "repeat_differs", "item": "orange", "class": "small", "forward": 1} and r["base_after"] is True and n == 1,
                  f"10: a later forward whose null row is not forward 0's makes the reading void, repeat_differs, at forward 1 (rule 4) ({r['status']}, {r['void']})")
            four = {"controls": {"count": 4, "seed": 0}}  # 5 arms in 2 slots: the third forward of 8 rows has one arm and one row left over
            last = lv.layout([1.0], 4, list(lv.NULL_E), 8)[2]
            check(last[6] is None and last[5] == "control:3", f"10: (with four control edits the last forward of 8 rows leaves row 6 untouched: {last})")
            r, n = plant(mlp4, "a row left over moved in the last forward", four, mode="move", row=6, rows=8, pieces=To, only=2)
            check(r["status"] == "void" and r["void"] == {"code": "untouched_rows_differ", "item": "orange", "class": "small", "forward": 2} and n == 1,
                  f"10: an untouched row that is not the twin, moved in the last forward, makes the reading void, untouched_rows_differ (rule 2) ({r['status']}, {r['void']})")
            # a void in the second item: what was read before it is in the book with status void, and is not printed
            r, n = plant(mlp4, "void in the second item", mode="move", row=-1, rows=8, pieces=Tf, only=0)
            first = r["items"][0] if r["items"] else {}
            text = text_of(r)
            check(r["status"] == "void" and (r["void"] or {}).get("item") == "fox" and r["summary"] is None and first.get("status") == "ok"
                  and first.get("classes", {}).get("small") is not None, f"10: a reading void in its second item has no summary; the first item is in the result as it was read ({r['status']}, {r['void']})")
            check("read before the reading came out void" in text and "mean" not in text.replace("on average", ""),
                  "10: and the lines of a void reading print no figure of the items read before the void")
            # the gate at the LARGE row count: every row of the 16-row forward sharpened alike, so nothing is void and the base row reads lower there
            r, n = plant(fin, "the base row lower at 16 rows, the gate set aside", {"bars": {"item_gate": -1e9}}, mode="sharpen", rows=16, pieces=To)
            g = next(e for e in r["items"] if e["item"]["name"] == "orange")["gate"]
            gs, gl = g["small"]["mean"], g["large"]["mean"]
            check(r["status"] == "ok" and n >= 1 and gl < gs - 0.1, f"10: (the check's own need: with every row of the 16-row forward sharpened the base reads lower there: {gs:.2f} at 8 rows, {gl:.2f} at 16)")
            bar = round((gs + gl) / 2, 2)
            r, n = plant(fin, "under the bar at 16 rows only", {"bars": {"item_gate": bar}}, mode="sharpen", rows=16, pieces=To)
            by = {e["item"]["name"]: e for e in r["items"]}
            o = by["orange"]
            # the gate as a caution (16 November 2026): the item is read at both row counts, its caution naming 16 rows
            check(r["status"] == "ok" and o["status"] in ("ok", "unstable") and o["gate"]["passed"] is False
                  and o["gate"]["caution"] == s_caution(o["gate"]["small"]["mean"], o["gate"]["large"]["mean"], 8, 16, bar) and "at 16 rows." in o["gate"]["caution"]
                  and o["gate"]["small"]["mean"] >= bar > o["gate"]["large"]["mean"] and o["classes"]["small"] is not None and o["classes"]["large"] is not None
                  and o["stability"] is not None and by["fox"]["status"] == "ok" and by["fox"]["gate"]["caution"] is None
                  and r["summary"]["items"]["refused"] == 0 and r["summary"]["items"]["under_bar"] == 1
                  and r["summary"]["items"]["kept"] == 1 + (o["status"] == "ok"),
                  f"10: an item over the bar at 8 rows and under it at 16 is read at both, with the gate's caution in its words for 16 rows ({o['status']}: {o['gate']['caution']!r})")
            text = text_of(r)
            check(" ".join(o["gate"]["caution"].split()) in text and "before the gate refused" not in text and text.count("the edit at dose 1:") == 2 + 2 + 2,
                  "10: and its lines print its caution and both its row counts' figures (the two items' two row counts and the summary's two)")
            # the gate is on the MEAN: a bar between an item's mean and its median (every forward of the item sharpened alike, so the two stand apart)
            r0 = lv.read(h3, {**base_reading, "name": "planted: nothing, for the figures of a quiet reading"}, items)
            r, n = plant(fin, "every forward sharpened, the gate set aside", {"items": ["orange"], "bars": {"item_gate": -1e9}}, mode="sharpen", pieces=To, by=40.0)
            gm, gm16 = r["items"][0]["gate"]["small"], r["items"][0]["gate"]["large"]
            bar = round((gm["mean"] + gm["median"]) / 2, 3)
            check(r0["status"] == "ok" and r["status"] == "ok" and abs(gm["mean"] - gm["median"]) > 0.1 and min(gm["mean"], gm["median"]) < bar < max(gm["mean"], gm["median"])
                  and min(gm16["mean"], gm16["median"]) < bar < max(gm16["mean"], gm16["median"]) and (gm["mean"] < bar) == (gm16["mean"] < bar),
                  f"10: (the check's own need: sharpened, the item's mean and median log-probability stand apart at both row counts: {gm['mean']:.3f} and {gm['median']:.3f})")
            r, n = plant(fin, "every forward sharpened, a bar between the mean and the median", {"items": ["orange"], "bars": {"item_gate": bar}}, mode="sharpen", pieces=To, by=40.0)
            e = r["items"][0]
            check(r["status"] == "ok" and e["status"] != "refused" and (e["gate"]["caution"] is not None) == (gm["mean"] < bar) and e["gate"]["passed"] == (gm["mean"] >= bar),
                  f"10: with the bar between an item's mean and its median the gate's caution goes by the mean ({e['status']}; mean {gm['mean']:.3f}, median {gm['median']:.3f}, bar {bar})")
            # stability's sign test: the edit's row put on the other side of the base row at 16 rows only
            row16 = lv.layout([1.0], 5, list(lv.NULL_E), 16)[0].index("edit:1.0")
            r, n = plant(fin, "the edit's row reflected about the base row at 16 rows", mode="reflect", row=row16, rows=16, pieces=To)
            o = next(e for e in r["items"] if e["item"]["name"] == "orange")
            own = ((o.get("stability") or {}).get("metrics", {}).get("own", {}).get("arms", {}).get("edit:1.0")) or {}
            floor = max(o["classes"][c]["floor"]["own"]["mean_abs"] or 0.0 for c in ("small", "large")) if o["classes"]["large"] else None
            check(own.get("ratio") is not None and own["ratio"] <= 3.0 and own["small"] * own["large"] < 0 and floor is not None and max(abs(own["small"]), abs(own["large"])) > 3.0 * floor,
                  f"10: (the check's own need: the reflected row reads the text's own log-probability with the other sign, within the ratio, clear of the floor: {own}, floor {floor})")
            check(own.get("agree") is False and ["own", "edit:1.0"] in [list(x) for x in o["stability"]["disagree"]] and o["status"] == "unstable",
                  "10: two readings of opposite signs, within the ratio and clear of the floor, disagree on a signed read-out: the item is unstable")
            # a null row that moves with the row count: the floor moves, the arms are set against the LARGER floor (4.9), and the lines say the floor moved
            r, n = plant(mlp4, "a null row and the edit's row moved at 16 rows", mode="move", row=[1, row16], rows=16, pieces=To)
            o = next(e for e in r["items"] if e["item"]["name"] == "orange")
            kl = o["stability"]["metrics"]["kl"]["arms"]["edit:1.0"]
            fs, fl_ = (o["classes"][c]["floor"]["kl"]["mean_abs"] for c in ("small", "large"))
            check(kl["large"] > 3.0 * kl["small"] and fl_ > 3.0 * fs and kl["agree"] is True and kl["ratio"] <= 3.0 and close(kl["ratio"], max(kl["large"], fl_) / fl_, 1e-9),
                  f"10: with a null row moved too, the edit's two readings ({kl['small']:.2e} and {kl['large']:.2e}) are set against the larger of the two floors "
                  f"({fs:.2e} and {fl_:.2e}) and agree by the spec's rule (ratio {kl['ratio']:.2f})")
            check("itself moved between the row counts" in text_of(r), "10: and the lines say that the floor itself moved between the row counts, and that the agreement shows less for it")
            check("itself moved between the row counts" not in text_of(r0), "10: (a reading whose floors stand together says no such thing)")
            # null rows that came out the base row by bits are no draws of the floor
            r, n = plant(fin, "one null row the base row by bits", mode="copy", row=1, rows=8, pieces=To)
            fl = next(e for e in r["items"] if e["item"]["name"] == "orange")["classes"]["small"]["floor"]["kl"]
            check(r["status"] == "ok" and fl["of"] == 3 and fl["from"] != "null:0.03125" and "1 null row came out the base row by bits or not finite and is not counted" in text_of(r),
                  f"10: a null row that came out the base row by bits is not counted in the floor, which is then the largest of 3, and the lines say so ({fl})")
            r, n = plant(fin, "every null row the base row by bits", mode="copy", row=[1, 2, 3, 4], rows=8, pieces=To)
            fl = next(e for e in r["items"] if e["item"]["name"] == "orange")["classes"]["small"]["floor"]["kl"]
            check(r["status"] == "ok" and fl == {"mean_abs": None, "median_abs": None, "above_1": None, "from": None, "of": 0} and "no null row gave a figure" in text_of(r)
                  and "above the floor at all" not in text_of(r).split("item fox")[0].split("16 rows, 1 forward")[0],
                  f"10: with every null row the base row by bits there is no floor: None, of 0, and the lines say no null row gave a figure ({fl})")
            # an arm that is not finite on one item of two: the summary's mean is over the other alone, and the lines say over how many
            row8 = lv.layout([1.0], 5, list(lv.NULL_E), 8)[0].index("edit:1.0")
            check(row8 == row16, "10: (the edit's row is the same row at both row counts)")
            r, n = plant(fin, "the edit's row not finite on one item", mode="nan", row=row8, pieces=Tf, only=(0, 3))
            by = {e["item"]["name"]: e for e in r["items"]}
            arms = [a for c in ("small", "large") for a in by["fox"]["classes"][c]["arms"] if a["key"] == "edit:1.0"]
            check(r["status"] == "ok" and by["fox"]["status"] == "ok" and len(arms) == 2 and all(a["non_finite"] is True and a["metrics"] is None for a in arms)
                  and all(by["fox"]["classes"][c]["rank"]["rank"] is None for c in ("small", "large")) and r["summary"]["items"]["kept"] == 2
                  and close(r["summary"]["classes"]["small"]["arms"]["edit:1.0"]["kl"]["mean_over_items"], by["orange"]["classes"]["small"]["arms"][4]["metrics"]["kl"]["mean"], 1e-12),
                  f"10: an edit's row that is not finite is not a void: the arm has no figure and no rank, and the item is kept ({r['status']}: {r['why']}; fox {by['fox']['status']})")
            check("over 1 of the 2 kept" in text_of(r) and "over 1 of the 2 kept" not in text_of(r0), "10: and the summary's line says its mean is over one of the two items kept")
            # a folded twin that is not finite is an arm that is not finite, not a void (4.5): every row of the twin's forward made NaN
            twin_call = len(lv.layout([1.0], 5, list(lv.NULL_E), 8))  # the forwards of the small class come first; the twin's is the next of 8 rows
            set_e = {"terms": [{"layer": 2, "set": [3, 40], "vectors": "set.npy"}]}
            r, n = plant(fin, "the folded twin not finite", {"edit": set_e}, mode="nan_all", rows=8, pieces=To, only=twin_call)
            o = next(e for e in r["items"] if e["item"]["name"] == "orange")
            fo = (o["classes"]["small"] or {}).get("folded") or {}
            check(r["status"] == "ok" and n == 1 and fo.get("taken") is True and fo.get("metrics") is None and fo.get("gap") is None and fo.get("rows_identical") is None
                  and o["classes"]["large"]["folded"]["metrics"] is not None and r["base_after"] is True,
                  f"10: a folded twin whose rows are not finite is taken with no figure and does not void the reading ({r['status']}, {r['void']}; {r['why']})")
            check("a row is not finite" in text_of(r), "10: and its lines say a row is not finite")
        finally:
            for hook_, pl in zip(hooks, (mlp4, fin)):
                pl.set(None)
                hook_.remove()
        check(words_of(lambda: we.assert_base(Pb.lm, hP.hooks_at_load), we.NotBase) is None, "10: with those hooks gone the model is the base against the first handle's census")
    with part("10: what read is handed, and what it records of it"):
        items = items_of(Pb, (0, 1))
        items_f = items_of(Fb)
        n_before, _h = lb.head()
        e = Pb.build(we.silence(2, [3, 40]))
        false_id = we.Edit("f" * 64, e.terms, e.base, "", e.report)
        r = lv.read(hP, {**paired, "name": "an edit under a false id", "edit": sil}, items, edit=false_id)
        check(r["status"] == "refused" and r["why"] == f"The edit handed in says it is ffffffffffff and hashes to {e.id[:12]}." and r["edit"] is None
              and book_entry(r["book"]["seq"])["body"]["status"] == "refused", f"10: an Edit handed to read under an id that is not its hash is refused, and entered ({r['status']}: {r['why']!r})")
        r = lv.read(hP, {**paired, "name": "an edit that is no Edit", "edit": sil}, items, edit="not an edit")
        check(r["status"] == "refused" and "weight_edits.Edit" in str(r["why"]) and book_entry(r["book"]["seq"])["body"]["name"] == "an edit that is no Edit",
              f"10: what is no Edit, handed to read as one, is a refused result in the ledger, not an exception before it ({r['status']}: {r['why']!r})")
        held = np.full((1, D), 0.123456, dtype=np.float32)
        r = lv.read(hP, {**paired, "name": "a recipe that holds its array", "edit": {"terms": [{"layer": 2, "set": [5], "vectors": held}]}}, items)
        body = json.dumps(book_entry(r["book"]["seq"])["body"])
        check(r["status"] == "refused" and "0.123" not in body and isinstance(r["plan"]["edit"]["terms"][0]["vectors"], str) and f"[1, {D}]" in r["plan"]["edit"]["terms"][0]["vectors"]
              and "names its vectors' file" in str(r["why"]),
              f"10: an array written into a recipe is recorded by what it is, never by its numbers ({r['plan']['edit']['terms'][0]['vectors']!r})")
        r = lv.read(hP, {**paired, "name": "a scale by exactly 1", "edit": {"terms": [{"layer": 2, "scale": [3, 40], "factor": 1.0}]}}, items)
        check(r["status"] == "refused" and r["why"] == S_NOTHING, f"10: an edit that scales by exactly 1 is refused as the base at any dose ({r['status']}: {r['why']!r})")
        r = lv.read(hP, {**paired, "name": "a dose the rounding moves", "edit": sil, "doses": [1.0, 0.003]}, items)
        printed(r)
        pairs = (r["edit"] or {}).get("exact", {}).get("0.003", {}).get("factors")
        check(r["status"] == "ok" and pairs is not None and len(pairs) == 1 and close(pairs[0][0], 0.997, 1e-12) and pairs[0][1] == 0.99609375
              and "0.996094 where 0.997 was asked" in " ".join(lv.lines(r)),
              f"10 (P): a dose the rounding moves is read, and its line says the factor as applied beside the one asked ({pairs})")
        # the null factors are judged in the MODEL's dtype: 1 +- 0.001 is 1 in bf16 and is not in float32
        thin = {"null_e": [0.001, -0.001]}
        r = lv.read(hF, {**paired, "name": "F thin null rows", "edit": sil, "items": ["orange"], **thin}, items_f)
        check(r["status"] == "ok", f"10 (F): null rows of e = +-0.001 are taken on the float32 stand-in, where neither factor is 1 ({r['status']}: {r['why']})")
        r = lv.read(hP, {**paired, "name": "P thin null rows", "edit": sil, "items": ["orange"], **thin}, items)
        check(r["status"] == "refused" and like(r["why"], s_null_one(0.001)), f"10 (P): and refused on the bf16 stand-in, where each is the base row again ({r['status']}: {r['why']!r})")
        # an item that does not begin with the start mark is said so on its line
        it = items["orange"]
        bare = {"no mark": lv.item_from_ids(it.prompt_ids[1:], it.answer_ids, name="no mark"), "orange": it}
        r = lv.read(hP, {**paired, "name": "an item with its start mark dropped", "edit": sil}, bare)
        printed(r)
        lines_ = lv.lines(r)
        check(r["status"] == "ok" and any("item no mark" in x and "NO start mark at its head" in x for x in lines_)
              and any("item orange" in x and "NO start mark" not in x for x in lines_),
              "10: an item that does not begin with the tokenizer's start mark is said so on its line, and one that does is not")
        check(lb.head()[0] == n_before + 8 and lb.verify()["ok"] is True, "10: each of these eight is in the ledger, and the chain is whole")
    with part("10: the handle"):
        check((hF.n_layers, hF.d_model, hF.d_ff, hF.model_id, hF.weights_id, hF.hooks_at_load, hF.dtype, hP.dtype) == (NL, D, N, Fb.model_id, None, {}, torch.float32, torch.bfloat16)
              and hF.lm is Fb.lm and hF.tokenizer is Fb.tok and hF.rt is Fb.rt and hF.softcap == 30.0 and isinstance(hF.runtime, dict) and isinstance(hF.weights, dict),
              "10: handle() holds the runtime's parts as the spec lists them")


# -------------------------------------------------------------------------------------- 11. the written reading

W_ITEM = {"item", "rows", "max_new", "stop_ids", "untouched_rows_identical", "seconds_a_step", "base", "arms"}
W_ARM = {"key", "what", "e", "dose", "factor", "edit_id", "row", "pieces", "text", "ended", "same_as_base", "parts_at", "logprob_gap"}


def check_written(Pb: Box, hP: Any) -> None:
    plans = TMP / "plans11"
    sil = {"terms": [{"layer": 1, "silence": list(range(60))}]}
    reading = {"name": "P written", "kind": "written", "edit": sil, "doses": [1.0, 0.0], "controls": {"count": 2, "seed": 0}, "max_new": 12, "items": ["orange"]}
    path = plan_file(plans, "written", [reading], items=(0, 1))
    items = lv.plan_items(lv.load_plan(path), Pb.tok)
    item = items["orange"]
    R, steps = 8, 12
    edit = Pb.build(*we.from_recipe(sil, plans))
    controls = we.control_edits(edit, Pb.lm, 0, 2)
    keys = lv.layout([1.0, 0.0], 2, list(lv.NULL_E_WRITTEN), R, one_forward=True)[0]
    progs = programs_of(keys, edit, controls, 1.0, 1)

    def own_write() -> Dict[int, List[Tuple[int, float, bool]]]:
        with we.per_row(Pb.lm, progs):
            return Pb.write(item.prompt_ids, R, steps, Pb.rt.end_of_turn_ids())

    def arm_facts(mine: List[Tuple[int, float, bool]], base: List[Tuple[int, float, bool]]) -> Dict[str, Any]:
        """Where a row parts from the base row, and the log-probability gap before that (spec 4.12), by hand."""
        parts = next((i for i in range(max(len(mine), len(base))) if i >= len(mine) or i >= len(base) or mine[i][0] != base[i][0]), None)
        n = parts if parts is not None else len(base)
        gaps = [abs(mine[i][1] - base[i][1]) for i in range(n)]
        return {"parts_at": parts, "same_as_base": mine == base, "n": n, "max_abs": max(gaps) if gaps else None, "mean_abs": sum(gaps) / len(gaps) if gaps else None}

    def compare(entry: Dict[str, Any], wrote: Dict[int, List[Tuple[int, float, bool]]], what: str) -> None:
        base = wrote[0]
        b = entry["base"]
        check(set(b) == {"pieces", "text", "logprobs", "ended"} and b["pieces"] in ([t[0] for t in base], [t[0] for t in base if not t[2]])
              and b["logprobs"][:len(b["pieces"])] == [t[1] for t in base][:len(b["pieces"])] and b["ended"] == base[-1][2] and isinstance(b["text"], str),
              f"{what}: the base row's pieces and log-probabilities are the check's own write's, exactly")
        ok, said = [a["key"] for a in entry["arms"]] == [k for k in keys if k], ""
        for a in entry["arms"]:
            row = keys.index(a["key"]) if a["key"] in keys else -1
            want = arm_facts(wrote[row], base)
            mine = [t[0] for t in wrote[row]]
            gap = a["logprob_gap"]
            good = (W_ARM - {"e", "dose", "factor", "edit_id"}) <= set(a) <= W_ARM and a["row"] == row and a["pieces"] in (mine, [t[0] for t in wrote[row] if not t[2]]) \
                and a["ended"] == wrote[row][-1][2] and a["parts_at"] == want["parts_at"] and a["same_as_base"] == want["same_as_base"] \
                and set(gap) == {"max_abs", "mean_abs", "n"} and gap["n"] == want["n"] \
                and ((want["n"] == 0 and gap["max_abs"] in (None, 0.0) and gap["mean_abs"] in (None, 0.0))
                     or (close(gap["max_abs"], want["max_abs"], 1e-9) and close(gap["mean_abs"], want["mean_abs"], 1e-9)))
            if not good:
                ok, said = False, said + f" {a['key']}: {({k: a.get(k) for k in ('row', 'parts_at', 'same_as_base', 'ended', 'logprob_gap')})} against {want};"
        check(ok, f"{what}: every arm's row, pieces, parts_at, same_as_base and logprob_gap are the check's own, worked from its own write under the same programs" + said)

    with part("11: rows as planned"):
        (r,) = read_all(hP, path, Pb)
        wrote = own_write()
        check(r["status"] == "ok" and r["kind"] == "written" and r["summary"] is None and r["rows"] is None and r["base_after"] is True and len(r["items"]) == 1,
              f"11: the written reading is ok, with no summary and no two-row-count block ({r['status']}: {r['why']})")
        top_shape(r, Pb, "11")
        e = r["items"][0]
        check(set(e) == W_ITEM and set(e["item"]) == ITEM_BLOCK and e["rows"] == R and e["max_new"] == steps and e["stop_ids"] == Pb.rt.end_of_turn_ids()
              and e["untouched_rows_identical"] is True and e["seconds_a_step"] >= 0, f"11: the item's entry: rows as planned, the stop ids, untouched rows identical ({sorted(set(e) ^ W_ITEM)})")
        compare(e, wrote, "11")
        by = {a["key"]: a for a in e["arms"]}
        check(by["edit:0.0"]["same_as_base"] is True and by["edit:0.0"]["parts_at"] is None and by["edit:0.0"]["logprob_gap"]["max_abs"] == 0.0
              and by["edit:0.0"]["logprob_gap"]["n"] == len(wrote[0]) and by["edit:1.0"]["same_as_base"] is False and by["edit:1.0"]["parts_at"] is not None,
              "11: the dose-0 arm writes the base row's pieces with its log-probabilities; the edit's arm parts from it")
        check(wrote[R - 1] == wrote[0] and Pb.write(item.prompt_ids, R, steps, Pb.rt.end_of_turn_ids())[0] == wrote[0],
              "11: (the check's own write: the twin writes as the base row, and the base row as an unhooked write of the same shape)")
        text = " ".join(lv.lines(r)).lower()
        check("greedy" in text and "written" in text, "11: the lines say the reading is greedy")
    with part("11: a row that ends early"):
        wrote = own_write()
        mine = [t[0] for t in wrote[keys.index("edit:1.0")]]
        others = {t[0] for row, k in enumerate(keys) if k != "edit:1.0" for t in wrote[row]}
        j = next((i for i, t in enumerate(mine) if 0 < i < steps - 2 and t not in others and t not in mine[:i]), None)
        check(j is not None, "11: the edit's row writes, after its first piece, a piece no other row writes (the check's own need)")
        if j is not None:
            cfg = Pb.hf.generation_config
            kept = cfg.eos_token_id
            cfg.eos_token_id = list(kept) + [mine[j]]  # the runtime reads its end marks from here at each call
            try:
                r = lv.read(hP, {**reading, "name": "P written, a row ends early"}, items)
                printed(r)
                wrote = own_write()
            finally:
                cfg.eos_token_id = kept
            e = r["items"][0]
            by = {a["key"]: a for a in e["arms"]}
            check(r["status"] == "ok" and mine[j] in e["stop_ids"] and by["edit:1.0"]["ended"] is True and len(by["edit:1.0"]["pieces"]) in (j, j + 1)
                  and e["base"]["ended"] is False and len(e["base"]["pieces"]) == steps and all(not a["ended"] for k, a in by.items() if k != "edit:1.0"),
                  f"11: a row that writes an end mark ends early and the others go on ({r['status']}: {r['why']})")
            compare(e, wrote, "11 (a row ends early)")
    with part("11: a planted difference in an untouched row"):
        planted = Planted()
        hook = Pb.lm.layers[4].mlp.register_forward_hook(planted)
        try:
            h2 = lv.handle(Pb.rt)
            planted.set("move", row=-1)
            r = lv.read(h2, {**reading, "name": "P written, planted"}, items)
            printed(r)
        finally:
            planted.set(None)
            hook.remove()
        check(r["status"] == "void" and (r["void"] or {}).get("code") == "untouched_rows_differ" and r["base_after"] is True,
              f"11: a planted difference in an untouched row voids the written reading, untouched_rows_differ ({r['status']}, {r['void']})")
    with part("11: refusals of a written reading"):
        r = lv.read(hP, {**reading, "name": "P written, too many arms", "rows": 5}, items)
        check(r["status"] == "refused" and r["why"] == "A written reading is one batch: 5 rows hold the base row, its twin, 2 null rows and 1 more, too few for 2 doses and 2 control edits. "
              "Ask fewer, or more rows.", f"11: a written reading whose arms do not fit is refused as a result ({r['status']}: {r['why']!r})")


# -------------------------------------------------------------------------------------------------- 12. the book


def check_book() -> None:
    path = TMP / "book12" / "level_book.jsonl"
    stamp = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
    with part("12: entries chain"):
        check((lb.VERSION, lb.BOOK_ENV, lb.GENESIS, tuple(lb.KINDS)) == (1, BOOK_VAR, "0" * 64, ("paired", "written", "note")) and issubclass(lb.BookBroken, RuntimeError),
              "12: the ledger's constants as the spec ships them")
        check(lb.head(path) == (0, "0" * 64) and not path.exists() and list(lb.entries(path)) == [], "12: with no file the head is (0, 64 zeros), and looking makes none")
        v = lb.verify(path)
        check(v["ok"] is True and v["entries"] == 0 and v["first_bad"] is None, f"12: verify of no file is a whole chain of nothing ({v})")
        e0 = lb.append("note", {"says": "first line"}, path=path)
        e1 = lb.append("paired", {"x": 0.1, "n": None, "list": [1, "two", {"three": True}]}, path=path)
        e2 = lb.append("written", {"says": "caf\u00e9"}, path=path)
        check(all(set(e) == {"v", "seq", "at", "kind", "prev", "body", "hash"} for e in (e0, e1, e2)) and [e["seq"] for e in (e0, e1, e2)] == [0, 1, 2]
              and [e["kind"] for e in (e0, e1, e2)] == ["note", "paired", "written"] and all(stamp.fullmatch(e["at"]) for e in (e0, e1, e2)) and e0["v"] == 1,
              "12: append returns the entry written: seven keys, seq from 0, UTC to the second")
        check(e0["prev"] == "0" * 64 and e1["prev"] == e0["hash"] and e2["prev"] == e1["hash"]
              and all(e["hash"] == lb.entry_hash(e) == sha_obj({k: v for k, v in e.items() if k != "hash"}) for e in (e0, e1, e2)),
              "12: each entry's prev is the hash of the line before, and its hash is the spec's hash of the entry without it")
        check(path.read_bytes() == b"".join(canon(e) + b"\n" for e in (e0, e1, e2)), "12: the file is one canonical line an entry")
        check(oct(path.stat().st_mode & 0o777) == "0o600", f"12: the file is made for its owner alone ({oct(path.stat().st_mode & 0o777)})")
        check(lb.head(path) == (3, e2["hash"]) and list(lb.entries(path)) == [e0, e1, e2] and list(lb.entries(path, start=1, limit=1)) == [e1]
              and list(lb.entries(path, start=2)) == [e2], "12: head and entries read them back, from a line on and so many")
        v = lb.verify(path)
        check(v == {"ok": True, "entries": 3, "head": e2["hash"], "first_bad": None, "why": None}, f"12: verify passes ({v})")
    with part("12: a byte changed in a middle line"):
        lines = path.read_bytes().split(b"\n")
        bad = TMP / "book12" / "changed.jsonl"
        bad.write_bytes(b"\n".join([lines[0], lines[1].replace(b'"two"', b'"twp"'), lines[2], b""]))
        v = lb.verify(bad)
        check(v["ok"] is False and v["first_bad"] == 1 and isinstance(v["why"], str) and v["why"], f"12: a byte changed in a middle line is named by its line ({v})")
        bad.write_bytes(b"\n".join([lines[0], b"not json at all", lines[2], b""]))
        v = lb.verify(bad)
        check(v["ok"] is False and v["first_bad"] == 1, f"12: a line that is not JSON is named too, and verify does not raise ({v})")
        bad.write_bytes(b"\n".join([lines[0], lines[2], b""]))
        v = lb.verify(bad)
        check(v["ok"] is False and v["first_bad"] == 1, f"12: a line taken out is seen at the line after it ({v})")
    with part("12: a line of another version or kind; a book that cannot be read; what the chain does not show"):
        lines = path.read_bytes().split(b"\n")
        odd = TMP / "book12" / "odd.jsonl"
        for label, change in (("a kind that is none of the ledger's", {"kind": "verdict"}), ("another version", {"v": 7})):
            entry = {**json.loads(lines[1]), **change}
            entry["hash"] = sha_obj({k: v for k, v in entry.items() if k != "hash"})  # rewritten with its own hash worked again
            odd.write_bytes(b"\n".join([lines[0], canon(entry), b""]))
            v = lb.verify(odd)
            check(v["ok"] is False and v["first_bad"] == 1 and isinstance(v["why"], str), f"12: a line of {label}, its hash worked again, is named by verify ({v})")
            check(words_of(lambda: lb.append("note", {}, path=odd), lb.BookBroken) is not None, f"12: and append will not write after it ({label})")
        v = lb.verify(TMP / "book12")  # a folder where the book should be
        check(v["ok"] is False and v["first_bad"] == 0 and isinstance(v["why"], str), f"12: verify of a path that cannot be read as a book does not raise: it says so ({v})")
        # a number written past a float's reach reads as an infinity, which no line is written with: such a line is
        # not an entry, wherever it stands, and no reader of the book raises anything but BookBroken for it
        for at_line, label in ((1, "a middle line"), (2, "the last line")):
            raw = [lines[0], lines[1], lines[2]]
            raw[at_line] = raw[at_line].replace(b'"body":{', b'"body":{"a_number":1e999,', 1)
            odd.write_bytes(b"\n".join(raw + [b""]))
            v = words_of(lambda: lb.verify(odd), Exception) or lb.verify(odd)
            check(isinstance(v, dict) and v["ok"] is False and v["first_bad"] == at_line and "not finite" in str(v["why"]),
                  f"12: {label} that holds a number past a float's reach is named by verify, which does not raise ({v})")
            got = words_of(lambda: list(lb.entries(odd)), lb.BookBroken)
            check(got is not None and not got.startswith("<") and f"Line {at_line} of the book" in got,
                  f"12: and reading the book's entries raises BookBroken at it, and nothing else ({label}: {got!r})")
        got_h, got_a = words_of(lambda: lb.head(odd), lb.BookBroken), words_of(lambda: lb.append("note", {}, path=odd), lb.BookBroken)
        check(got_h is not None and not got_h.startswith("<") and got_a is not None and not got_a.startswith("<"),
              f"12: with such a line last, head raises BookBroken and append will not write after it ({got_h!r})")
        cut = TMP / "book12" / "cut.jsonl"
        cut.write_bytes(b"\n".join([lines[0], lines[1], b""]))
        v = lb.verify(cut)
        check(v["ok"] is True and v["entries"] == 2, "12: (what the chain does NOT show, as the ledger's docstring says: a book with lines left off its END verifies; "
              f"the head a run prints is the only record of where the end was: {v['entries']} of 3 lines, ok {v['ok']})")
    with part("12: a torn last line"):
        for label, tail in (("no newline", b'{"v":1,"seq":3'), ("not JSON", b"not json at all\n"),
                            ("a hash that is not its own", canon({**json.loads(path.read_bytes().split(b"\n")[2]), "seq": 3}) + b"\n")):
            torn = TMP / "book12" / "torn.jsonl"
            torn.write_bytes(path.read_bytes() + tail)
            before = torn.read_bytes()
            got = words_of(lambda: lb.append("note", {"says": "after the tear"}, path=torn), lb.BookBroken)
            says(got, f"The book's last line (line {W}) is not a whole entry: {W}. Nothing was written. Look at {torn} before anything more is read.",
                 f"12: a torn last line ({label}) makes append raise BookBroken in the spec's words")
            check(torn.read_bytes() == before, f"12: and nothing is written ({label})")
            check(words_of(lambda: lb.head(torn), lb.BookBroken) is not None, f"12: head raises BookBroken on it too ({label})")
            v = lb.verify(torn)
            check(v["ok"] is False and v["first_bad"] == 3, f"12: and verify names line 3 ({label}: {v})")
    with part("12: what append refuses"):
        before = path.read_bytes()
        check(words_of(lambda: lb.append("verdict", {}, path=path)) is not None and words_of(lambda: lb.append("Note", {}, path=path)) is not None,
              "12: a kind outside KINDS raises ValueError")
        for label, body in (("a number that is not finite", {"x": float("nan")}), ("a numpy scalar", {"x": np.float32(1.0)}), ("a set", {"x": {1, 2}}), ("a tuple", {"x": (1, 2)})):
            got = words_of(lambda: lb.append("note", body, path=path))
            check(got is not None and not got.startswith("<"), f"12: a body holding {label} raises ValueError ({got!r})")
        check(path.read_bytes() == before and lb.verify(path)["ok"] is True, "12: and none of them wrote a line")
    with part("12: the default path"):
        kept = {k: os.environ.pop(k, None) for k in (BOOK_VAR, "XDG_DATA_HOME")}
        try:
            home_default = lb.book_path()
            os.environ["XDG_DATA_HOME"] = str(TMP / "xdg")
            xdg_default = lb.book_path()
            os.environ[BOOK_VAR] = str(TMP / "named.jsonl")
            named = lb.book_path()
        finally:
            for k, v in kept.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        check(home_default == Path.home() / ".local" / "share" / "theodolite" / "level_book.jsonl" and ROOT_DIR not in home_default.resolve().parents,
              f"12: the default path is under the data home, outside the repository ({home_default})")
        check(xdg_default == TMP / "xdg" / "theodolite" / "level_book.jsonl" and named == TMP / "named.jsonl" and lb.book_path() == Path(os.environ[BOOK_VAR]),
              "12: book_path reads the environment at every call")
    with part("12: two processes appending at once"):
        shared = TMP / "book12" / "shared.jsonl"
        lb.append("note", {"who": "the check", "i": 0}, path=shared)
        code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import backend.level_book as lb\n"
                "for i in range(int(sys.argv[3])): lb.append('note', {'who': sys.argv[4], 'i': i}, path=Path(sys.argv[2]))")
        each = 25
        procs = [subprocess.Popen([sys.executable, "-I", "-c", code, str(ROOT_DIR), str(shared), str(each), who], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for who in ("one", "two")]
        outs = [p.communicate(timeout=120) for p in procs]
        check(all(p.returncode == 0 for p in procs), f"12: both writers ended well ({[o[1][-200:] for o in outs]})")
        v = lb.verify(shared)
        got = [(e["body"]["who"], e["body"]["i"]) for e in lb.entries(shared)]
        check(v["ok"] is True and v["entries"] == 1 + 2 * each and all([i for w, i in got if w == who] == list(range(each)) for who in ("one", "two")),
              f"12: two appends from two processes both land, each writer's in its own order, and the chain verifies ({v})")


# ---------------------------------------------------------------------------------------------- 13. the estimate


def whole_s(rows: int, pieces: int) -> float:
    """A whole forward's seconds as the module prices it since 16 November 2026: 0.135 s and 0.377 ms a row-position, the
    card check's refit of its own readings (verify_1a.json, proposed_bars.cost). The spec's 4.11 had 0.08 s and 0.33 ms,
    a bare forward measured 14 November 2026 (fact 9)."""
    return 0.135 + 0.000377 * rows * pieces


def check_estimate() -> None:
    near = lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)  # noqa: E731
    check(near(whole_s(32, 176), 2.258264), "13: whole(32, 176) is 2.258264 s by the refit's two figures (1.93856 s by the spec's)")

    def facts(layer: int, **kw: Any) -> Dict[str, Any]:
        return {"layers": [layer], "lowest_layer": layer, "linears": 1, "has_vectors": False, "has_lowrank": False, "factors": [0.0], **kw}

    def est(reading: Dict[str, Any], pieces: List[Tuple[int, int]], f: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
        plan = lv.normalise(reading, n_layers=60, facts=f)
        return lv.estimate(plan, pieces, n_layers=60, facts=f, **kw)

    edit20 = {"terms": [{"layer": 20, "silence": [4294, 9618]}], "note": "the two that tipped the raw recipe in pilot 7"}
    edit44 = {"terms": [{"layer": 44, "silence": [7]}]}
    with part("13: a whole forward, and a restart"):
        one = est({"name": "a", "kind": "paired", "edit": edit44, "controls": {"count": 0, "seed": 0}, "lower": "whole"}, [(176, 30)], facts(44))
        check(set(one) == {"seconds", "parts", "forwards", "row_positions", "gib_above_model"} and set(one["parts"]) == {"forwards", "readout", "folded", "writing"}
              and isinstance(one["forwards"], int) and isinstance(one["row_positions"], int), f"13: estimate's keys ({sorted(one)})")
        check(near(one["parts"]["forwards"], whole_s(12, 176) + whole_s(32, 176)) and one["forwards"] == 2,
              f"13: one forward a class: whole(12, 176) + whole(32, 176), 2.258264 s of it the large one ({one['parts']['forwards']})")
        check(near(one["seconds"], sum(one["parts"].values())) and one["parts"]["folded"] == 0 and one["parts"]["writing"] == 0, "13: the seconds are the parts' sum; no twin and no writing")
        # 60 control edits and one dose are 61 arms: 11 forwards of 12 rows (6 slots) and 3 of 32 (26 slots)
        many = {"name": "a", "kind": "paired", "edit": edit44, "controls": {"count": 60, "seed": 0}}
        share = 16 / 60
        same = est(many, [(176, 30)], facts(44))
        check(near(same["parts"]["forwards"], whole_s(12, 176) * (1 + 10 * share) + whole_s(32, 176) * (1 + 2 * share)),
              "13: under same_shape a later forward restarts: from layer 44 of 60 it is 16/60 of a whole one")
        check(near(est({**many, "lower": "whole"}, [(176, 30)], facts(44))["parts"]["forwards"], 11 * whole_s(12, 176) + 3 * whole_s(32, 176)), "13: under lower \"whole\" every forward is whole")
        check(near(est({**many, "lower": "one_row"}, [(176, 30)], facts(44))["parts"]["forwards"], (11 * whole_s(12, 176) + 3 * whole_s(32, 176)) * share + whole_s(1, 176)),
              "13: under one_row every forward restarts, and one whole forward at one row is added an item")
        S = 176 - 30
        check(near(same["parts"]["readout"], 0.000088 * S * ((11 + 4 + 1 + 60) + (3 + 4 + 1 + 60))),
              "13: the read-out: each forward's base row, the null rows once, every arm, 0.088 ms a scored row-position (the refit of 16 November 2026)")
        named = est({**many, "readouts": {"odds": {"toward": [5], "away": [6]}}}, [(176, 30)], facts(44))
        # 4.11's read-out line is "for one item ... in one class": with kl and own off it is 0.000088 * S a class (0.00011 in the spec), two classes
        check(near(named["parts"]["readout"], 0.000088 * S * 2),
              f"13: with named pieces alone the read-out is the gate's base row, once a class ({named['parts']['readout']})")
    with part("13: the twin"):
        setf = facts(44, has_vectors=True, factors=[])
        lone = est({"name": "a", "kind": "paired", "edit": edit44, "controls": {"count": 0, "seed": 0}}, [(176, 30)], setf, twin=True)
        check(near(lone["parts"]["folded"], whole_s(12, 176) + whole_s(32, 176) + 2 * 0.0003), "13: the twin is priced whole when its class has one forward, plus 0.3 ms a linear touched")
        many = est({"name": "a", "kind": "paired", "edit": edit44, "controls": {"count": 60, "seed": 0}}, [(176, 30)], setf, twin=True)
        check(near(many["parts"]["folded"], (whole_s(12, 176) + whole_s(32, 176)) * 16 / 60 + 2 * 0.0003), "13: and as a restart when its class has later forwards")
        check(near(many["parts"]["readout"], 0.000088 * (176 - 30) * ((11 + 4 + 1 + 60 + 1) + (3 + 4 + 1 + 60 + 1))), "13: the twin's row is read too")
        auto = est({"name": "a", "kind": "paired", "edit": edit44, "controls": {"count": 0, "seed": 0}}, [(176, 30)], setf)
        never = est({"name": "a", "kind": "paired", "edit": edit44, "controls": {"count": 0, "seed": 0}, "folded": "never"}, [(176, 30)], setf)
        check(near(auto["parts"]["folded"], lone["parts"]["folded"]) and never["parts"]["folded"] == 0 and est({"name": "a", "kind": "paired", "edit": edit44}, [(176, 30)], facts(44))["parts"]["folded"] == 0,
              "13: with twin=None the twin is decided from the plan and the edit's facts: taken for a set, not under folded \"never\", not for a silencing")
    with part("13: the worked plan of 5.2, by hand"):
        first = {"name": "silence 4294 and 9618", "kind": "paired", "edit": edit20, "doses": [1.0, -1.0], "controls": {"count": 19, "seed": 0}}
        pieces = [(82, 26), (64, 30)]  # two items of the check's own sizes: (pieces, prompt pieces)
        got = est(first, pieces, facts(20))
        share = 40 / 60
        # an item: 4 forwards of 12 rows (2 doses and 19 control edits are 21 arms in 6 slots) and 1 of 32 (spec 5.2)
        want_f = sum(whole_s(12, T) * (1 + 3 * share) + whole_s(32, T) for T, _p in pieces)
        want_r = sum(0.000088 * (T - p) * ((4 + 4 + 2 + 19) + (1 + 4 + 2 + 19)) for T, p in pieces)
        check(near(got["parts"]["forwards"], want_f), f"13: the worked plan's forwards part ({got['parts']['forwards']} against {want_f})")
        check(near(got["parts"]["readout"], want_r), f"13: its read-out part ({got['parts']['readout']} against {want_r})")
        check(got["parts"]["folded"] == 0 and got["parts"]["writing"] == 0 and near(got["seconds"], want_f + want_r),
              "13: its twin part is nought (a silencing is exact on the input side, and its mirror is a scale by 2)")
        check(est(first, [(82, 26)], facts(20))["forwards"] == 5 and near(est(first, [(82, 26)], facts(20))["gib_above_model"], 0.191 * 32 * 82 / 1024 + 0.5),
              "13: an item is 5 forwards; memory above the model is the largest forward's 0.191 MiB a row-position (the card's refit) and 0.5 GiB for the read-out")
        second = {"name": "the same, written", "kind": "written", "edit": {"terms": [{"layer": 20, "silence": [4294, 9618]}]}, "items": ["moon"]}
        got = est(second, [(82, 26)], facts(20))
        check(near(got["parts"]["writing"], whole_s(8, 26) + 0.095 * 48) and near(got["seconds"], whole_s(8, 26) + 0.095 * 48) and near(got["gib_above_model"], 0.8e6 * 8 * (26 + 48) / 2**30),
              f"13: the written reading: the prompt's forward at 8 rows and 48 steps of 0.095 s ({got['parts']})")
    with part("13: the two size refusals"):
        written = {"name": "w", "kind": "written", "edit": edit20}
        says(words_of(lambda: est({**written, "rows": 32, "max_new": 256}, [(60, 30)], facts(20)), lv.Refused),
             "Writing 32 rows to 286 word-pieces of context keeps about 6.8 GiB above the model (0.8 MB a row a piece, measured 28 October 2026); 1a takes at most 6. "
             "Ask fewer rows or fewer new pieces.", "13: a written reading of 32 rows to 286 pieces of context is 6.8 GiB and refused")
        small = est({**written, "rows": 8, "max_new": 48}, [(60, 30)], facts(20))
        check(abs(small["gib_above_model"] - 0.46) < 0.005 and near(small["gib_above_model"], 0.8e6 * 8 * 78 / 2**30), f"13: and 8 rows to 78 pieces is 0.46 GiB ({small['gib_above_model']:.4f})")
        got = words_of(lambda: est({"name": "a", "kind": "paired", "edit": edit20}, [(400, 30)], facts(20)), lv.Refused)
        check(got is not None and re.fullmatch(r"32 rows of 400 word-pieces is 12,?800 row-positions; 1a runs at most 12,288 in a forward \(the pilots measured to 11,264\)\.", got) is not None,
              f"13: a forward over 12,288 row-positions is refused ({got!r})")
        check(words_of(lambda: est({"name": "a", "kind": "paired", "edit": edit20}, [(384, 30)], facts(20)), lv.Refused) is None, "13: and 32 rows of 384 pieces, 12,288 row-positions, is not")
    with part("13: read's prediction is estimate's"):
        for label, n_layers in (("f", NL), ("p", NL)):
            if label not in KEPT:
                check(False, f"13: section 10 kept no {label.upper()} result to set beside the estimate")
                continue
            r, recipe, base_dir = KEPT[label]
            taken = {c["folded"]["taken"] for e in r["items"] for c in e["classes"].values()}
            got = lv.estimate(r["plan"], [(e["item"]["pieces"], e["item"]["prompt_pieces"]) for e in r["items"]], n_layers=n_layers, facts=we.recipe_facts(recipe, base_dir),
                              twin=taken == {True})
            check(len(taken) == 1 and near(r["seconds"]["predicted"], got["seconds"]), f"13 ({label.upper()}): read's seconds.predicted equals estimate's for the same twin "
                  f"({r['seconds']['predicted']} against {got['seconds']})")


# ---------------------------------------------------------------------------------------- 14. the command line


def kernel_files_under_repo() -> set:
    """Whatever the kernel writes (a ledger, a saved edit's files, a stand-in run's own folder), found ANYWHERE under
    the repository but its .git, .venv and node_modules. By name, so that what other work in the checkout writes while
    the check runs is not taken for the kernel's. A listing of the repository's top level alone would not see a file
    written deep inside it."""
    found = set()
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "node_modules", "__pycache__")]
        here = Path(root)
        found |= {str(here / d) for d in dirs if d.startswith("levelling-stand-in-")}
        found |= {str(here / f) for f in files if f in ("level_book.jsonl", "payload.safetensors", "payload.safetensors.tmp")
                  or (f == "manifest.json" and re.fullmatch(r"[0-9a-f]{64}", here.name) is not None)}
    return found


def run_cli(args: Sequence[str], env: Dict[str, str], timeout: int = 600) -> Tuple[int, str, str]:
    try:
        r = subprocess.run([sys.executable, "-I", str(CLI), *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(ROOT_DIR))
    except subprocess.TimeoutExpired:
        return -1, "", f"no answer in {timeout} s"
    return r.returncode, r.stdout, r.stderr


def check_cli() -> None:
    folder = TMP / "cli"
    sil = {"terms": [{"layer": 2, "silence": [3, 40]}]}
    readings = [{"name": "cli paired", "kind": "paired", "edit": sil, "controls": {"count": 3, "seed": 0}, "rows": {"small": 8, "large": 16}, "bars": {"item_gate": GATE}},
                {"name": "cli written", "kind": "written", "edit": sil, "items": ["orange"], "max_new": 8, "controls": {"count": 2, "seed": 0}}]
    plan = plan_file(folder, "sitting", readings, items=(0, 1))
    book, edits = folder / "book" / "level_book.jsonl", folder / "edits"
    env = {k: v for k, v in os.environ.items() if k not in ("THEODOLITE_GGUF", "THEODOLITE_DEVICE", "THEODOLITE_UNPACKER")}
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", BOOK_VAR: str(book), EDITS_VAR: str(edits)})
    # a run without --stand-in must stop before any load; should it not, it finds no card and no checkpoint
    empty = folder / "no_checkpoint_here"
    empty.mkdir(parents=True, exist_ok=True)
    card_env = {**env, "HF_HOME": str(empty), "HF_HUB_CACHE": str(empty), "TRANSFORMERS_CACHE": str(empty)}
    top0, deep0 = set(os.listdir(ROOT_DIR)), kernel_files_under_repo()
    with part("14: estimate"):
        code, out, err = run_cli(["estimate", str(plan), "--stand-in"], env)
        check(code == 0 and "cli paired" in out and "cli written" in out, f"14: estimate PLAN --stand-in exits 0 and names each reading (exit {code}: {err[-300:]})")
        PRINTED.extend(out.splitlines())
        check(not book.exists(), "14: estimate writes no ledger")
    with part("14: run --stand-in"):
        out_file = folder / "out.json"
        code, out, err = run_cli(["run", str(plan), "--stand-in", "--out", str(out_file), "--save-edits"], env)
        PRINTED.extend(out.splitlines())
        results = json.loads(out_file.read_text()) if out_file.exists() else []
        check(code == 0 and [r.get("name") for r in results] == ["cli paired", "cli written"] and all(r["status"] == "ok" for r in results),
              f"14: run --stand-in on a paired and a written reading exits 0, both ok, --out holding both results (exit {code}: {(out + err)[-400:]})")
        v = lb.verify(book)
        check(v["ok"] is True and v["entries"] == 2 and [e["kind"] for e in lb.entries(book)] == ["paired", "written"], f"14: both are entered in the ledger the variable names ({v})")
        saved = [r["edit"]["saved"] for r in results if r.get("edit")]
        check(len(saved) == 2 and all(s and Path(s).parent == edits and (Path(s) / "manifest.json").exists() for s in saved), f"14: --save-edits saves each reading's edit under the folder the variable names ({saved})")
        check(all(r["runtime"]["threads"] == 1 for r in results), "14: a stand-in run sets one thread")
    with part("14: run --stand-in with the two variables taken out"):
        bare_env = {k: v for k, v in env.items() if k not in (BOOK_VAR, EDITS_VAR)}
        bare_env["XDG_DATA_HOME"] = str(folder / "data_home")  # where a slip would write, in place of the real ledger
        out_file = folder / "only.json"
        code, out, err = run_cli(["run", str(plan), "--stand-in", "--only", "cli written", "--out", str(out_file)], bare_env)
        PRINTED.extend(out.splitlines())
        first = out.splitlines()[0] if out.splitlines() else ""
        found = re.search(r"(/\S*levelling-stand-in-[^\s/'\"]+)", first)
        own = Path(found.group(1)) if found else None
        results = json.loads(out_file.read_text()) if out_file.exists() else []
        check(code == 0 and own is not None and own.is_dir() and ROOT_DIR not in own.resolve().parents, f"14: its first line names a temporary folder of its own (exit {code}; {first!r})")
        check([r.get("name") for r in results] == ["cli written"] and results[0]["status"] == "ok", f"14: --only reads the reading named and no other ({[r.get('name') for r in results]})")
        check(own is not None and any(lb.verify(p)["entries"] == 1 for p in own.rglob("*.jsonl")), "14: and the run's ledger is in that folder")
        check(not (folder / "data_home").exists() and lb.verify(book)["entries"] == 2 and set(os.listdir(ROOT_DIR)) - top0 == set(),
              f"14: nothing is written to the default ledger's place or under the repository ({sorted(set(os.listdir(ROOT_DIR)) - top0)})")
        if own is not None and own.is_dir() and own.name.startswith("levelling-stand-in-"):
            shutil.rmtree(own, ignore_errors=True)
    with part("14: book"):
        code, out, err = run_cli(["book", "--tail", "5"], env)
        PRINTED.extend(out.splitlines())
        rows_out = [x for x in out.splitlines() if x.strip()]
        check(code == 0 and len(rows_out) >= 2 and any("paired" in x and "cli paired" in x for x in rows_out) and any("written" in x for x in rows_out),
              f"14: book --tail gives one line an entry (exit {code}: {out[-300:]!r})")
        code, out, err = run_cli(["book", "--seq", "0"], env)
        try:
            entry = json.loads(out[out.index("{"):out.rindex("}") + 1])
        except ValueError:
            entry = {}
        check(code == 0 and entry.get("seq") == 0 and entry.get("kind") == "paired" and entry.get("hash") == next(iter(lb.entries(book)))["hash"],
              f"14: book --seq 0 prints that entry's JSON (exit {code})")
        code, out, err = run_cli(["book", "--verify"], env)
        check(code == 0, f"14: book --verify exits 0 with the chain whole (exit {code}: {(out + err)[-200:]!r})")
        lines = book.read_bytes().split(b"\n")
        changed = folder / "changed.jsonl"
        changed.write_bytes(b"\n".join([lines[0].replace(b'"cli paired"', b'"cli pairee"'), lines[1], b""]))
        code, out, err = run_cli(["book", "--verify"], {**env, BOOK_VAR: str(changed)})
        check(code == 1 and (out + err).strip() != "", f"14: book --verify exits 1 on a changed line, and says which (exit {code}: {(out + err)[-200:]!r})")
    with part("14: exit 1"):
        bad = plan_file(folder, "refused", [{**readings[0], "name": "an edit that is refused", "edit": {"terms": [{"layer": 2, "silence": [500]}]}}], items=(0,))
        code, out, err = run_cli(["run", str(bad), "--stand-in"], env)
        PRINTED.extend(out.splitlines())
        check(code == 1 and lb.verify(book)["entries"] == 3, f"14: run with a reading refused exits 1, the refusal entered (exit {code}: {(out + err)[-300:]!r})")
    with part("14: exit 2"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            code, out, err = run_cli(["run", str(plan), "--port", str(port)], card_env, timeout=180)
        check(code == 2 and "Something answers on :" in out + err and "stop the Theodolite server first" in out + err,
              f"14: the port guard: a run without --stand-in while the port answers exits 2 before anything is loaded (exit {code}: {(out + err)[-300:]!r})")
        code, out, err = run_cli(["run", str(plan), "--port", str(port)], card_env, timeout=180)  # the port is closed now
        check(code == 2 and "Something answers on :" not in out + err, f"14: no CUDA: a run without --stand-in on a machine with no card exits 2 (exit {code}: {(out + err)[-300:]!r})")
        code, out, err = run_cli(["run", str(plan), "--port", str(port)], {**card_env, "THEODOLITE_GGUF": str(folder / "some.gguf")}, timeout=180)
        check(code == 2 and "1a reads the W4A16 checkpoint on the card, or the stand-in with --stand-in." in out + err,
              f"14: THEODOLITE_GGUF set: exit 2 in the spec's words (exit {code}: {(out + err)[-300:]!r})")
        code, out, err = run_cli(["run", str(plan), "--port", str(port)], {**card_env, "THEODOLITE_DEVICE": "cpu"}, timeout=180)
        check(code == 2 and "1a reads the W4A16 checkpoint on the card, or the stand-in with --stand-in." in out + err,
              f"14: THEODOLITE_DEVICE naming another device than cuda:0: exit 2 in the same words (exit {code}: {(out + err)[-300:]!r})")
        (folder / "broken.json").write_text("{not json")
        code, out, err = run_cli(["run", str(folder / "broken.json"), "--stand-in"], env)
        code2, out2, err2 = run_cli(["estimate", str(folder / "absent.json"), "--stand-in"], env)
        check(code == 2 and code2 == 2, f"14: a plan that cannot be read: exit 2 from run and from estimate (exits {code} and {code2})")
        twice = plan_file(folder, "twice", [readings[0], readings[0]], items=(0,))
        code, out, err = run_cli(["run", str(twice), "--stand-in"], env)
        check(code == 2, f"14: a plan refused at its top level (a reading name used twice): exit 2 (exit {code})")
        torn = folder / "torn.jsonl"
        torn.write_bytes(book.read_bytes() + b'{"v":1,"seq":')
        code, out, err = run_cli(["run", str(plan), "--stand-in"], {**env, BOOK_VAR: str(torn)})
        check(code == 2 and torn.read_bytes().endswith(b'{"v":1,"seq":'), f"14: the ledger broken: exit 2, nothing written (exit {code}: {(out + err)[-300:]!r})")
        print("        (not shown here: too little of the card free, and the unpacker refused; both need the card)")
    with part("14: --out never replaces the book or the plan"):
        before, n = plan.read_bytes(), lb.verify(book)["entries"]
        code, out, err = run_cli(["run", str(plan), "--stand-in", "--out", str(plan)], env)
        check(code == 2 and plan.read_bytes() == before and "--out names the plan file" in out and lb.verify(book)["entries"] == n,
              f"14: --out naming the plan file: exit 2 before anything is loaded, the plan as it was (exit {code}: {(out + err)[-200:]!r})")
        code, out, err = run_cli(["run", str(plan), "--stand-in", "--out", str(book)], env)
        check(code == 2 and "--out names the book" in out and lb.verify(book)["ok"] is True and lb.verify(book)["entries"] == n,
              f"14: --out naming the book: exit 2, the book as it was (exit {code}: {(out + err)[-200:]!r})")
    with part("14: a slip in one reading does not stop the run"):
        # a factor of 401 digits is past what a float holds: the estimate must not stop on it, and the reading is a refusal like any other
        slip = plan_file(folder, "slip", [{**readings[0], "name": "before the slip"},
                                          {**readings[0], "name": "a factor of 401 digits", "edit": {"terms": [{"layer": 2, "scale": [3], "factor": 10 ** 400}]}},
                                          {**readings[0], "name": "after the slip"}], items=(0,))
        n = lb.verify(book)["entries"]
        code, out, err = run_cli(["run", str(slip), "--stand-in"], env)
        PRINTED.extend(out.splitlines())
        got = [(e["body"]["name"], e["body"]["status"]) for e in lb.entries(book, start=n)]
        check(code == 1 and got == [("before the slip", "ok"), ("a factor of 401 digits", "refused"), ("after the slip", "ok")] and "Traceback" not in err,
              f"14: a reading that holds a number no float can hold is refused and entered, and the readings after it are read (exit {code}; {got}; {err[-200:]!r})")
        check("the model is the base" in out and "the book:" in out, "14: and the run still ends by asserting the base and printing the book's head")
    check(lb.verify(book)["ok"] is True and set(os.listdir(ROOT_DIR)) - top0 == set(), "14: after all of it the command line's ledger is whole and nothing new lies at the repository's top")
    new = sorted(kernel_files_under_repo() - deep0)
    check(not new, "14: and no ledger, saved edit or stand-in folder of the kernel's was written anywhere under the repository, however deep" + (f" ({new[:4]})" if new else ""))


# ----------------------------------------------------------------------------------- 15. slice 1b and bugfixes


def check_1b(Fb: Box, Pb: Box, hF: Any, hP: Any) -> None:
    """Slice 1b features (row edits on gate_proj/up_proj, masked direction removal, quiet_mask, writers,
    capture_direction, delivered, other_sites_matched / same_site / sham_target controls, stack, edit_diagnostics)
    and 1a bugfixes, on both F (float32) and P (packed bf16)."""
    items_f = items_of(Fb, (0, 1))
    items_p = items_of(Pb, (0, 1))
    d1b = TMP / "1b"
    d1b.mkdir(parents=True, exist_ok=True)
    paired_base = {"kind": "paired", "rows": {"small": 8, "large": 16}, "bars": {"item_gate": GATE},
                   "controls": {"count": 4, "seed": 1}, "items": ["orange"]}

    with part("15a: 1a bugfixes"):
        # 1. Cumulative recipe file-size preflight refused unread before reading a second corrupt file
        f1, f2 = d1b / "cum1.npy", d1b / "cum2.npy"
        np.save(f1, vec(101, 200, D))  # ~51 KiB
        f2.write_bytes(b"NOT A VALID NPY PAYLOAD" * 2400)  # ~55 KiB
        kept_cap = we.MAX_PAYLOAD_BYTES
        try:
            # Make each individual file pass (size <= cap + 4096) while their sum exceeds cap + 4096 * 2
            s1, s2 = f1.stat().st_size, f2.stat().st_size
            we.MAX_PAYLOAD_BYTES = max(s1, s2)
            assert s1 <= we.MAX_PAYLOAD_BYTES + 4096 and s2 <= we.MAX_PAYLOAD_BYTES + 4096
            assert s1 + s2 > we.MAX_PAYLOAD_BYTES + 4096 * 2
            got = words_of(lambda: we.from_recipe(
                {"terms": [{"layer": 2, "add": list(range(200)), "vectors": "cum1.npy"},
                           {"layer": 3, "add": [2], "vectors": "cum2.npy"}]}, d1b), we.Refused)
            check(got is not None and "MiB of arrays" in got,
                  f"15a: cumulative recipe file-size preflight refuses unread before opening a bad second file ({got!r})")
        finally:
            we.MAX_PAYLOAD_BYTES = kept_cap
        # 2. Lowrank -0.0 merge-order invariance
        A_p = np.zeros((1, N), dtype=np.float32)
        A_n = np.full((1, N), -0.0, dtype=np.float32)
        A_p[0, 3] = 0.05
        A_n[0, 3] = 0.05
        B_p = np.zeros((D, 1), dtype=np.float32)
        B_n = np.full((D, 1), -0.0, dtype=np.float32)
        B_p[4, 0] = 0.05
        B_n[4, 0] = 0.05
        A_q, B_q = vec(102, 1, N, size=0.05), vec(103, D, 1, size=0.05)
        ids_lr = {
            Fb.build(we.lowrank(2, A_p, B_p), we.lowrank(2, A_q, B_q)).id,
            Fb.build(we.lowrank(2, A_n, B_n), we.lowrank(2, A_q, B_q)).id,
            Fb.build(we.lowrank(2, A_q, B_q), we.lowrank(2, A_n, B_n)).id,
            Fb.build(we.lowrank(2, A_q, B_q), we.lowrank(2, A_p, B_p)).id,
        }
        check(len(ids_lr) == 1, f"15a: lowrank merge order and id are invariant to -0.0 ({len(ids_lr)} distinct ids)")
        # 3. _edit_words when 1.0 not in doses
        r_half_sil = lv.read(hP, {**paired_base, "name": "1b half dose sil", "doses": [0.5],
                                  "controls": {"count": 0, "seed": 0},
                                  "edit": {"terms": [{"layer": 2, "silence": [5]}]}}, items_p)
        printed(r_half_sil)
        r_half_sc = lv.read(hP, {**paired_base, "name": "1b half dose scale", "doses": [0.5],
                                 "controls": {"count": 0, "seed": 0},
                                 "edit": {"terms": [{"layer": 2, "scale": [5], "factor": 3.0}]}}, items_p)
        printed(r_half_sc)
        check(r_half_sil["status"] == "ok" and "silenced" in " ".join(lv.lines(r_half_sil)),
              f"15a: _edit_words identifies silencing when dose 1.0 is absent ({lv.lines(r_half_sil)[1]!r})")
        check(r_half_sc["status"] == "ok" and "(by 3)" in " ".join(lv.lines(r_half_sc)),
              f"15a: _edit_words recovers scale factor when dose 1.0 is absent ({lv.lines(r_half_sc)[1]!r})")
        # 4. Malformed edit.base refused cleanly (no AttributeError)
        e_ok = Pb.build(we.silence(2, [3]))
        bad_base = we.Edit(e_ok.id, e_ok.terms, "not-a-dict", "", e_ok.report)  # type: ignore[arg-type]
        r_bad_base = lv.read(hP, {**paired_base, "name": "1b bad edit base", "edit": {"terms": [{"layer": 2, "silence": [3]}]}},
                             items_p, edit=bad_base)
        check(r_bad_base["status"] == "refused" and "base" in str(r_bad_base["why"]).lower(),
              f"15a: malformed edit.base is refused cleanly ({r_bad_base['status']}: {r_bad_base['why']!r})")
        # 5. level_book.entries(start=-1) raises ValueError immediately
        check(words_of(lambda: lb.entries(start=-1), ValueError) is not None,
              "15a: level_book.entries(start=-1) raises ValueError immediately")
        # 6. level_book.driver_view strips item text, token ids, and per-item rows
        raw_entry = book_entry(r_half_sil["book"]["seq"])
        dv = lb.driver_view(raw_entry)
        dv_json = json.dumps(dv)
        check(dv["seq"] == raw_entry["seq"] and dv["body"]["items"] == {"count": 1, "redacted": True}
              and dv["body"]["summary"] == raw_entry["body"]["summary"]
              and items_p["orange"].answer not in dv_json and "prompt_ids" not in dv_json,
              "15a: driver_view strips item text, token ids and per-item rows while keeping summary and metadata")
        # 7. CLI book --seq K --driver-view
        cli_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                   BOOK_VAR: os.environ[BOOK_VAR], EDITS_VAR: os.environ[EDITS_VAR]}
        code, out, err = run_cli(["book", "--seq", str(r_half_sil["book"]["seq"]), "--driver-view"], cli_env)
        dv_cli = json.loads(out[out.index("{"):out.rindex("}") + 1]) if code == 0 and "{" in out else {}
        check(code == 0 and (dv_cli.get("body") or {}).get("items", {}).get("redacted") is True
              and items_p["orange"].answer not in out,
              f"15a: CLI book --seq --driver-view outputs redacted entry (exit {code}: {err[-200:]})")
        # 8. Signed control arm sign-flip reporting in lv.lines
        r_flip = copy.deepcopy(r_half_sil)
        r_flip["plan"]["controls"]["count"] = 1
        r_flip["items"][0]["stability"]["metrics"]["own"]["controls"] = {
            "control:0": {"small": 0.5, "large": -0.5, "ratio": 1.0}
        }
        check("opposite-sign" in " ".join(lv.lines(r_flip)),
              "15a: lv.lines surfaces opposite-sign signed control arms clear of the floor")

    with part("15b: row edits on gate_proj and up_proj"):
        # Gate rules: gate_proj & up_proj accepted; down_proj refused; missing which in recipe refused;
        # uniform positive scale refused on up_proj and ALLOWED on gate_proj; same neuron on gate_proj + up_proj allowed
        check(words_of(lambda: Fb.build(we.scale_rows(2, "up_proj", list(range(N)), 2.0)), we.Refused) is not None,
              "15b: uniform positive scale of all rows on up_proj is refused")
        check(words_of(lambda: Fb.build(we.scale_rows(2, "gate_proj", list(range(N)), 2.0)), we.Refused) is None,
              "15b: uniform positive scale of all rows on gate_proj is allowed (gelu is non-linear)")
        check(words_of(lambda: Fb.build(we.silence_rows(2, "gate_proj", [4]), we.silence_rows(2, "up_proj", [4])), we.Refused) is None,
              "15b: editing the same neuron row on both gate_proj and up_proj is allowed")
        check(words_of(lambda: Fb.build(we.silence_rows(2, "gate_proj", [4]), we.scale_rows(2, "gate_proj", [4], 2.0)), we.Refused) is not None,
              "15b: naming the same row twice on one linear is refused")
        check(words_of(lambda: we.from_recipe({"terms": [{"layer": 2, "silence_rows": [4]}]}, d1b), we.Refused) is not None,
              "15b: row recipe term missing 'which' is refused")
        # Recipe round-trip & save/load round-trip
        rv = vec(104, 2, D, size=0.1)
        np.save(d1b / "rows_v.npy", rv)
        rec_rows = {"terms": [{"layer": 2, "which": "gate_proj", "silence_rows": [3]},
                              {"layer": 2, "which": "up_proj", "set_rows": [5, 7], "vectors": "rows_v.npy"}]}
        rf = we.recipe_facts(rec_rows, d1b)
        e_rows_p = Pb.build(*we.from_recipe(rec_rows, d1b))
        check(rf == we.edit_facts(e_rows_p) and e_rows_p.has_rows() and e_rows_p.has_vectors() and we.can_fold(Pb.lm, e_rows_p) is None,
              f"15b: recipe_facts == edit_facts for row edit, and P can fold it ({rf})")
        saved_dir = we.save(e_rows_p)
        loaded_rows = we.load(saved_dir, Pb.lm, model_id=Pb.model_id, weights_id=None)
        check(loaded_rows.id == e_rows_p.id and we.gate_again(Pb.lm, e_rows_p, model_id=Pb.model_id, weights_id=None).id == e_rows_p.id,
              "15b: row edit save/load and gate_again preserve exact id")
        # By bits on P: silence_rows, scale_rows(2.0), scale_rows(-0.5) equal folded _memit_row_overlays at 12 and 32 rows;
        # dose 0 equals base row by bits
        ids_p = torch.tensor([list(items_p["orange"].ids)], dtype=torch.long)
        for which in ("gate_proj", "up_proj"):
            for tag, spec in (("sil", we.silence_rows(2, which, [3, 17])),
                              ("x2", we.scale_rows(2, which, [3, 17], 2.0)),
                              ("xm05", we.scale_rows(2, which, [3, 17], -0.5))):
                ed = Pb.build(spec)
                for R in (12, 32):
                    x = ids_p.expand(R, -1)
                    with we.per_row(Pb.lm, [None, we.Arm(ed, 1.0), we.Arm(ed, 0.0)] + [None] * (R - 3)):
                        Hpr = Pb.lm(input_ids=x, use_cache=False).last_hidden_state
                    with we.folded(Pb.lm, ed, 1.0):
                        Hfo = Pb.lm(input_ids=x, use_cache=False).last_hidden_state
                    check(torch.equal(Hpr[1], Hfo[0]) and torch.equal(Hpr[2], Hpr[0]) and torch.equal(Hpr[-1], Hpr[0]),
                          f"15b (P): {which} {tag} at {R} rows equals folded _memit_row_overlays by bits, dose 0 and untouched equal base")
        # Recomputed set_rows / add_to_rows / non-exact scale_rows(1.5): zero cancellation + matches folded & weight copy
        for which in ("gate_proj", "up_proj"):
            e_set = Pb.build(we.set_rows(2, which, [4, 9], vec(105, 2, D, size=0.05)))
            e_add = Pb.build(we.add_to_rows(2, which, [4, 9], vec(106, 2, D, size=0.05)))
            e_sc = Pb.build(we.scale_rows(2, which, [4, 9], 1.5))
            x = ids_p.expand(8, -1)
            for tag, ed in (("set", e_set), ("add", e_add), ("scale1.5", e_sc)):
                with we.per_row(Pb.lm, [None, we.Arm(ed, 1.0), we.Arm(ed, 0.0)] + [None] * 5):
                    Hpr = Pb.lm(input_ids=x, use_cache=False).last_hidden_state
                with we.folded(Pb.lm, ed, 1.0):
                    Hfo = Pb.lm(input_ids=x, use_cache=False).last_hidden_state
                gap = float((Hpr[1].float() - Hfo[0].float()).abs().max())
                check(torch.equal(Hpr[2], Hpr[0]) and torch.equal(Hpr[-1], Hpr[0]) and gap < 0.1,
                      f"15b (P): recomputed {which} {tag} matches folded (max diff {gap:.4g}) and dose 0 is base by bits")
        # Zero catastrophic cancellation test on F: set a row to tiny 1e-6 vector while base row is large (~100)
        mod_g = Fb.lm.layers[2].mlp.gate_proj
        orig_row = mod_g.weight.data[10].clone()
        try:
            mod_g.weight.data[10] = torch.full((D,), 100.0, dtype=mod_g.weight.dtype)
            tiny = np.full((1, D), 1e-6, dtype=np.float32)
            e_tiny = Fb.build(we.set_rows(2, "gate_proj", [10], tiny))
            seen_g: Dict[str, torch.Tensor] = {}
            hk_x = mod_g.register_forward_pre_hook(lambda _m, a: seen_g.__setitem__("x", a[0].detach().clone()))
            try:
                x_f = torch.tensor([list(items_f["orange"].ids)] * 4, dtype=torch.long)
                with we.per_row(Fb.lm, [None, we.Arm(e_tiny, 1.0), we.Arm(e_tiny, 0.0), None]):
                    hk_y = mod_g.register_forward_hook(lambda _m, _a, o: seen_g.__setitem__("y", o.detach().clone()))
                    try:
                        Fb.lm(input_ids=x_f, use_cache=False)
                    finally:
                        hk_y.remove()
            finally:
                hk_x.remove()
            want_y = F.linear(seen_g["x"][1], torch.from_numpy(tiny))[:, 0]
            got_y = seen_g["y"][1, :, 10]
            err_cancel = float((got_y - want_y).abs().max())
            check(err_cancel < 1e-10, f"15b (F): recomputed set_rows has zero catastrophic cancellation (err {err_cancel:.2e})")
        finally:
            mod_g.weight.data[10].copy_(orig_row)
        # Decode step with row edits on P
        e_rw = Pb.build(we.set_rows(2, "up_proj", [6, 12], vec(107, 2, D, size=0.1)))
        prompt = list(items_p["orange"].prompt_ids)
        stop = [int(t) for t in Pb.rt.end_of_turn_ids()]
        base_w = list(ao.write_rows(Pb.lm, Pb.rt.lm_head_weight, [prompt] * 4, None, max_new=8, stop_ids=stop,
                                    bos_id=Pb.tok.bos_token_id, softcap=ao.softcap_of(Pb.hf)))
        with we.per_row(Pb.lm, [None, we.Arm(e_rw, 1.0), we.Arm(e_rw, 0.0), None]):
            hook_w = list(ao.write_rows(Pb.lm, Pb.rt.lm_head_weight, [prompt] * 4, None, max_new=8, stop_ids=stop,
                                        bos_id=Pb.tok.bos_token_id, softcap=ao.softcap_of(Pb.hf)))
        b0 = [(s.token_id, s.logprob) for s in base_w if s.row == 0]
        h0 = [(s.token_id, s.logprob) for s in hook_w if s.row == 0]
        h2 = [(s.token_id, s.logprob) for s in hook_w if s.row == 2]
        h3 = [(s.token_id, s.logprob) for s in hook_w if s.row == 3]
        check(b0 == h0 == h2 == h3, "15b (P): decode steps under row edit leave base, dose-0 and untouched rows exact")
        # End-to-end lv.read with row edit and folded twin on P
        r_row = lv.read(hP, {**paired_base, "name": "1b row edit", "edit": rec_rows, "folded": "always"},
                        items_p, base_dir=d1b)
        printed(r_row)
        check(r_row["status"] == "ok" and r_row["base_after"] is True
              and r_row["items"][0]["classes"]["small"]["folded"]["taken"] is True
              and "gate_proj" in " ".join(lv.lines(r_row)) and "up_proj" in " ".join(lv.lines(r_row)),
              f"15b (P): end-to-end paired reading of row edit succeeds with folded twin ({r_row['status']}: {r_row['why']})")

    with part("15c: quiet_mask, writers, capture_direction and masked remove_direction"):
        # quiet_mask detects tiny w_post entries and extra_mask
        ln2 = Fb.lm.layers[2].post_feedforward_layernorm
        orig_wp = ln2.weight.data.clone()
        try:
            med_w = float(orig_wp.abs().median())
            ln2.weight.data[:4] = 0.001 * med_w
            qm = we.quiet_mask(Fb.lm, 2)
            check(qm.dtype == bool and not qm.flags.writeable and list(np.nonzero(qm)[0]) == [0, 1, 2, 3],
                  f"15c: quiet_mask masks the 4 tiny w_post entries ({list(np.nonzero(qm)[0])})")
            extra = np.zeros(D, dtype=bool)
            extra[10] = True
            qm2 = we.quiet_mask(Fb.lm, 2, extra_mask=extra)
            check(list(np.nonzero(qm2)[0]) == [0, 1, 2, 3, 10], "15c: quiet_mask unions extra_mask")
            # Direction with >25% energy in quiet mask is refused by both writers() and remove_direction()
            u_bad = np.zeros(D, dtype=np.float32)
            u_bad[0] = 1.0
            u_bad[5] = 1.0  # 50% energy in quiet dim 0 -> > 25%
            check(words_of(lambda: we.writers(Fb.lm, 2, u_bad, 5), we.Refused) is not None,
                  "15c: writers refuses direction with >25% quiet-mask energy")
            check(words_of(lambda: Fb.build(we.remove_direction(2, [3, 5], u_bad)), we.Refused) is not None,
                  "15c: remove_direction refuses direction with >25% quiet-mask energy")
            # Valid direction with small quiet energy (<25%) is projected off quiet mask
            u_ok = vec(108, D, size=1.0)
            u_ok[:4] = 0.05  # tiny energy in quiet dims
            # Check writers against brute force
            top_w = we.writers(Fb.lm, 2, u_ok, 5)
            wp64 = ln2.weight.detach().double().numpy()
            u_m = u_ok.astype(np.float64).copy()
            u_m[qm] = 0.0
            u_m /= np.linalg.norm(u_m)
            W_d = gw.dense(Fb.mod(2)).double().numpy()
            bf_scores = (u_m * wp64) @ W_d
            bf_top = sorted(range(N), key=lambda n: (-abs(float(bf_scores[n])), n))[:5]
            check([n for n, _s in top_w] == bf_top and all(close(s, bf_scores[n], 1e-5) for n, s in top_w),
                  f"15c: writers matches brute-force stream-side projection ({top_w[:3]})")
            # Check remove_direction exact orthogonality & zero delta on quiet mask
            target_ns = [n for n, _s in top_w[:3]]
            e_dr = Fb.build(we.remove_direction(2, target_ns, u_ok))
            t_dr = e_dr.terms[0]
            Wc = gw.dense_cols(Fb.mod(2), torch.tensor(t_dr.idx, dtype=torch.long)).double().numpy()
            dW = t_dr.vectors.astype(np.float64)
            orth = ((Wc + dW) * wp64[None, :]) @ u_m
            check(bool((dW[:, qm] == 0.0).all()) and float(np.abs(orth).max()) < 1e-5,
                  f"15c: remove_direction leaves quiet dims identically 0 and makes (w_post*(W+dW)).u_m == 0 (max {np.abs(orth).max():.2e})")
            dr_rep = e_dr.report.get("direction_removal")
            check(isinstance(dr_rep, list) and len(dr_rep) == 1 and dr_rep[0]["masked_dims"] == 4
                  and {"quiet_share", "raw_norm", "stream_norm", "stream_rel_change"} <= set(dr_rep[0]),
                  f"15c: remove_direction report records quiet share and raw + stream norms ({dr_rep})")
            # Save/load & gate_again round-trip
            sd = we.save(e_dr)
            ld = we.load(sd, Fb.lm, model_id=Fb.model_id, weights_id=None)
            check(ld.id == e_dr.id and ld.report.get("direction_removal") == dr_rep,
                  "15c: remove_direction edit round-trips through save/load preserving id and direction_removal report")
        finally:
            ln2.weight.data.copy_(orig_wp)
        # capture_direction against manual hook capture
        u_cap = lv.capture_direction(hF, [items_f["orange"]], [items_f["who"]], 2, where="resid_post", at="end")
        box_m: Dict[str, torch.Tensor] = {}
        hk_m = Fb.lm.layers[2].register_forward_hook(lambda _m, _a, o: box_m.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        try:
            Fb.lm(input_ids=torch.tensor([list(items_f["orange"].ids)], dtype=torch.long), use_cache=False)
            ha = box_m["h"][0, -1].double().numpy()
            Fb.lm(input_ids=torch.tensor([list(items_f["who"].ids)], dtype=torch.long), use_cache=False)
            hb = box_m["h"][0, -1].double().numpy()
        finally:
            hk_m.remove()
        check(u_cap.shape == (D,) and u_cap.dtype == np.float32 and float(np.abs(u_cap - (ha - hb)).max()) < 1e-5
              and words_of(lambda: we.assert_base(Fb.lm, hF.hooks_at_load), we.NotBase) is None,
              "15c: capture_direction matches manual hook capture and leaves model base")
        # Recipe remove_direction + end-to-end paired reading on P with folded twin
        u_p = lv.capture_direction(hP, [items_p["orange"]], [items_p["who"]], 2, where="resid_post", at="span_mean")
        np.save(d1b / "dir_u.npy", u_p)
        u_sha = hashlib.sha256((d1b / "dir_u.npy").read_bytes()).hexdigest()
        top_p = [n for n, _s in we.writers(Pb.lm, 2, u_p, 4)]
        rec_dr = {"terms": [{"layer": 2, "remove_direction": top_p, "direction": "dir_u.npy", "sha256": u_sha}]}
        check(words_of(lambda: we.from_recipe({"terms": [{"layer": 2, "remove_direction": [0], "direction": "../escape.npy"}]}, d1b),
                       we.Refused) is not None, "15c: remove_direction recipe refuses path traversal")
        r_dr = lv.read(hP, {**paired_base, "name": "1b remove_direction", "edit": rec_dr, "folded": "always"},
                       items_p, base_dir=d1b)
        printed(r_dr)
        check(r_dr["status"] == "ok" and r_dr["base_after"] is True
              and r_dr["items"][0]["classes"]["small"]["folded"]["taken"] is True
              and "direction removal" in " ".join(lv.lines(r_dr)),
              f"15c (P): end-to-end remove_direction reading succeeds with folded twin ({r_dr['status']}: {r_dr['why']})")

    with part("15d: delivered metric against brute force"):
        e_sil = Pb.build(we.silence(2, [3, 40]))
        r_del = lv.read(hP, {**paired_base, "name": "1b delivered", "doses": [1.0, 0.0],
                             "controls": {"count": 2, "seed": 0},
                             "edit": {"terms": [{"layer": 2, "silence": [3, 40]}]}}, items_p)
        printed(r_del)
        it_o = items_p["orange"]
        lo = len(it_o.prompt_ids) - 1
        T = len(it_o.ids)
        arm1 = next(a for a in r_del["items"][0]["classes"]["small"]["arms"] if a["key"] == "edit:1.0")
        arm0 = next(a for a in r_del["items"][0]["classes"]["small"]["arms"] if a["key"] == "edit:0.0")
        # Brute-force layer-2 output capture at 8 rows
        keys8 = lv.layout([1.0, 0.0], 2, list(lv.NULL_E), 8)[0]
        ctls = we.control_edits(e_sil, Pb.lm, 0, 2)
        progs = [None if k is None else (we.Null(2, float(k.split(":")[1])) if k.startswith("null:")
                                         else we.Arm(e_sil, float(k.split(":")[1])) if k.startswith("edit:")
                                         else we.Arm(ctls[int(k.split(":")[1])], 1.0)) for k in keys8]
        box_d: Dict[str, torch.Tensor] = {}
        hk_d = Pb.lm.layers[2].register_forward_hook(lambda _m, _a, o: box_d.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        try:
            with we.per_row(Pb.lm, progs):
                Pb.lm(input_ids=torch.tensor([list(it_o.ids)] * 8, dtype=torch.long), use_cache=False)
        finally:
            hk_d.remove()
        hs = box_d["h"][:, lo:T - 1, :].float()
        bf_del = float(((hs[arm1["row"]] - hs[0]).norm(dim=-1) / hs[0].norm(dim=-1).clamp_min(1e-12)).mean())
        check(arm0["delivered"] == 0.0 and arm1["delivered"] is not None and arm1["delivered"] > 0.0
              and close(arm1["delivered"], bf_del, 1e-5) and "delivered" in " ".join(lv.lines(r_del)),
              f"15d (P): delivered matches brute-force layer output ({arm1['delivered']} vs {bf_del}, dose 0 is {arm0['delivered']})")
        # the span stops one short of the last position, where a read-out "at the end" is taken: the same figure
        # at that one position is recorded beside it, and is another number
        he = box_d["h"][:, T - 1, :].float()
        bf_end = float((he[arm1["row"]] - he[0]).norm() / he[0].norm().clamp_min(1e-12))
        check(arm0["delivered_end"] == 0.0 and close(arm1["delivered_end"], bf_end, 1e-5) and bf_end > 0.0
              and abs(bf_end - bf_del) > 1e-4 and "at the last position" in " ".join(lv.lines(r_del)),
              f"15d (P): delivered_end is the same figure at the last position alone, which the span leaves out "
              f"({arm1['delivered_end']} vs {bf_end}; over the span {bf_del})")
        hp = box_d["h"][:, :lo, :].float()
        bf_pre = float(((hp[arm1["row"]] - hp[0]).norm(dim=-1) / hp[0].norm(dim=-1).clamp_min(1e-12)).mean())
        check(arm0["delivered_prompt"] == 0.0 and close(arm1["delivered_prompt"], bf_pre, 1e-5) and bf_pre > 0.0 and lo > 1
              and "over the prompt" in " ".join(lv.lines(r_del)),
              f"15d (P): delivered_prompt is the same figure over the prompt's {lo} positions, which the span begins "
              f"after ({arm1['delivered_prompt']} vs {bf_pre})")

    with part("15e: control families (same_site, other_sites_matched, sham_target)"):
        # 1. same_site on a rank-1 remove_direction edit (with quiet mask dims)
        ln2 = Fb.lm.layers[2].post_feedforward_layernorm
        orig_wp = ln2.weight.data.clone()
        try:
            ln2.weight.data[:4] = 0.001 * float(orig_wp.abs().median())
            qm = we.quiet_mask(Fb.lm, 2)
            u_r1 = vec(109, D, size=1.0)
            u_r1[:4] = 0.0
            e_r1 = Fb.build(we.remove_direction(2, [3, 7, 11, 15], u_r1))
            ss = we.same_site_controls(e_r1, Fb.lm, seed=5, count=4)
            wp64 = ln2.weight.detach().double().numpy()
            dS_edit = e_r1.terms[0].vectors.astype(np.float64) * wp64[None, :]
            enorms = np.linalg.norm(dS_edit, axis=1)
            ok_ss = len(ss) == 4 and len({c.id for c in ss}) == 4
            for c in ss:
                tc = c.terms[0]
                dW_c = tc.vectors.astype(np.float64)
                dS_c = dW_c * wp64[None, :]
                cnorms = np.linalg.norm(dS_c, axis=1)
                _u, sv, _vt = np.linalg.svd(dS_c, full_matrices=False)
                r1_share = float(sv[0] ** 2 / (sv ** 2).sum()) if float((sv ** 2).sum()) > 0 else 0.0
                ok_ss = (ok_ss and np.array_equal(tc.idx, e_r1.terms[0].idx) and bool((dW_c[:, qm] == 0.0).all())
                         and bool(np.allclose(cnorms, enorms, rtol=1e-4, atol=1e-6)) and r1_share >= 1.0 - 1e-5)
            check(ok_ss, "15e: same_site controls share exact sites, have zero quiet energy, match stream norms and preserve rank-1 coherence")
            # 1b. a coherent multi-column add that is not rank-1: each control column keeps the edit's stream-side
            # length on a direction drawn on its own, as claim readings have always drawn them (trial-4's draw tied
            # to the edit's inner products came out with that rule: a claim reading's controls are drawn as before)
            u_c = vec(111, D, size=1.0)
            V_c = np.stack([u_c + 0.3 * vec(112 + i, D, size=1.0) for i in range(4)]).astype(np.float32) * 0.05
            V_c[:, :4] = 0.0
            e_co = Fb.build(we.add_to_columns(2, [3, 7, 11, 15], V_c))
            ss_co = we.same_site_controls(e_co, Fb.lm, seed=11, count=3)
            dS_co = e_co.terms[0].vectors.astype(np.float64) * wp64[None, :]

            def off_share(M: np.ndarray) -> float:
                G = M @ M.T
                d = np.sqrt(np.diag(G))
                C = G / np.outer(d, d)
                return float(np.abs(C[~np.eye(len(C), dtype=bool)]).mean())

            _u, sv_co, _vt = np.linalg.svd(dS_co, full_matrices=False)
            ok_co = (len(ss_co) == 3 and float(sv_co[0] ** 2 / (sv_co ** 2).sum()) < 1.0 - 1e-5
                     and [c.id for c in ss_co] == [c.id for c in we.same_site_controls(e_co, Fb.lm, seed=11, count=3)]
                     and off_share(dS_co) > 0.6 and not hasattr(we, "_random_orthonormal_masked"))
            for c in ss_co:
                dW_c = c.terms[0].vectors.astype(np.float64)
                dS_c = dW_c * wp64[None, :]
                ok_co = (ok_co and np.array_equal(c.terms[0].idx, e_co.terms[0].idx) and bool((dW_c[:, qm] == 0.0).all())
                         and bool(np.allclose(np.linalg.norm(dS_c, axis=1), np.linalg.norm(dS_co, axis=1), rtol=1e-4,
                                              atol=1e-6))
                         and off_share(dS_c) < 0.5 * off_share(dS_co))
            check(ok_co, f"15e: a coherent multi-column edit that is not rank-1 gets same-site controls whose columns keep "
                         f"the edit's stream-side lengths on directions drawn each on its own, as claim readings drew them "
                         f"before (the edit's columns {off_share(dS_co):.2f} alike, the controls' far less)")
        finally:
            ln2.weight.data.copy_(orig_wp)
        r_ss = lv.read(hP, {**paired_base, "name": "1b same_site", "controls": {"family": "same_site", "count": 4, "seed": 2},
                            "edit": {"terms": [{"layer": 2, "silence": [3, 40]}]}}, items_p)
        printed(r_ss)
        check(r_ss["status"] == "ok" and r_ss["controls"]["family"] == "same_site" and r_ss["controls"]["matched"] is True
              and len(r_ss["controls"]["ids"]) == 4 and "same-site control edits" in " ".join(lv.lines(r_ss)),
              f"15e (P): end-to-end same_site reading succeeds ({r_ss['status']}: {r_ss['why']})")
        # 2. other_sites_matched: estimate prices pre-pass, reading matches scores, tight tolerance voids
        e_m_rec = {"terms": [{"layer": 2, "silence": [3]}]}
        pl_unm = lv.normalise({**paired_base, "name": "m1", "edit": e_m_rec, "controls": {"count": 4, "seed": 0}},
                              n_layers=NL, tokenizer=Pb.tok, dtype=Pb.dtype)
        pl_mat = lv.normalise({**paired_base, "name": "m2", "edit": e_m_rec,
                               "controls": {"family": "other_sites_matched", "count": 4, "seed": 0, "tolerance": 3.0}},
                              n_layers=NL, tokenizer=Pb.tok, dtype=Pb.dtype)
        pieces_o = [(len(items_p["orange"].ids), len(items_p["orange"].prompt_ids))]
        est_unm = lv.estimate(pl_unm, pieces_o, n_layers=NL, facts=we.recipe_facts(e_m_rec, d1b))
        est_mat = lv.estimate(pl_mat, pieces_o, n_layers=NL, facts=we.recipe_facts(e_m_rec, d1b))
        check(est_mat["forwards"] == est_unm["forwards"] + len(pieces_o)
              and est_mat["row_positions"] == est_unm["row_positions"] + pieces_o[0][0],
              f"15e: estimate prices the 1-row pre-pass for other_sites_matched ({est_mat['forwards']} vs {est_unm['forwards']})")
        r_mat = lv.read(hP, {**paired_base, "name": "1b matched ok", "edit": e_m_rec,
                             "controls": {"family": "other_sites_matched", "count": 4, "seed": 0, "tolerance": 3.0}},
                        items_p)
        printed(r_mat)
        cov_ok = (r_mat.get("controls") or {}).get("coverage") or {}
        check(r_mat["status"] == "ok" and r_mat["controls"]["family"] == "other_sites_matched"
              and r_mat["controls"]["matched"] is True and cov_ok.get("sufficient") is True
              and len(set(r_mat["controls"]["ids"])) == 4 and "matched control edits" in " ".join(lv.lines(r_mat)),
              f"15e (P): other_sites_matched succeeds with sufficient distinct controls ({r_mat['status']}, {cov_ok})")
        r_mat_void = lv.read(hP, {**paired_base, "name": "1b matched void", "edit": e_m_rec,
                                  "controls": {"family": "other_sites_matched", "count": 20, "seed": 0, "tolerance": 1.0001}},
                             items_p)
        cov_bad = (r_mat_void.get("controls") or {}).get("coverage") or {}
        check(r_mat_void["status"] == "void" and (r_mat_void.get("void") or {}).get("code") == "insufficient_matched_controls"
              and cov_bad.get("sufficient") is False and r_mat_void["base_after"] is True,
              f"15e (P): other_sites_matched voids cleanly when coverage fails ({r_mat_void['status']}, {r_mat_void['void']})")
        # Lowrank-only edit refused with other_sites_matched before any forward
        np.save(d1b / "lrA.npy", vec(110, 2, N, size=0.1))
        np.save(d1b / "lrB.npy", vec(111, D, 2, size=0.1))
        r_lr_mat = lv.read(hP, {**paired_base, "name": "1b lr matched refused",
                                "edit": {"terms": [{"layer": 2, "lowrank": {"A": "lrA.npy", "B": "lrB.npy"}}]},
                                "controls": {"family": "other_sites_matched", "count": 2, "seed": 0}},
                           items_p, base_dir=d1b)
        check(r_lr_mat["status"] == "refused", f"15e: lowrank-only edit refused for other_sites_matched ({r_lr_mat['status']})")
        # 3. sham_target (including a sham at a lower layer than the main edit -> L0 lowers to 1)
        shams = [{"terms": [{"layer": 1, "silence": [7]}]},
                 {"terms": [{"layer": 2, "silence": [19]}]},
                 {"terms": [{"layer": 3, "silence": [25]}]}]
        r_sham = lv.read(hP, {**paired_base, "name": "1b sham_target",
                              "edit": {"terms": [{"layer": 3, "silence": [5]}]},
                              "controls": {"family": "sham_target", "shams": shams}}, items_p)
        printed(r_sham)
        check(r_sham["status"] == "ok" and r_sham["controls"]["family"] == "sham_target"
              and r_sham["controls"]["count"] == 3 and len(r_sham["controls"]["ids"]) == 3
              and r_sham["items"][0]["classes"]["small"]["restart_from"] == 1
              and "sham-target control edits" in " ".join(lv.lines(r_sham)),
              f"15e (P): sham_target succeeds and lowers restart_from to lowest sham layer ({r_sham['status']}, "
              f"restart_from={r_sham['items'][0]['classes']['small']['restart_from']})")

    with part("15f: stack and edit_diagnostics"):
        e1 = Fb.build(we.scale(2, [3, 5], [2.0, 0.5]))
        e2 = Fb.build(we.scale(2, [5, 8], [1.5, 3.0]))
        st12 = we.stack(Fb.lm, [e1, e2], model_id=Fb.model_id, weights_id=None, note="stacked")
        st21 = we.stack(Fb.lm, [e2, e1], model_id=Fb.model_id, weights_id=None)
        check(st12.id == st21.id and st12.report.get("parents") == [e1.id, e2.id]
              and list(st12.terms[0].idx) == [3, 5, 8]
              and np.allclose(st12.terms[0].factor, [2.0, 1.0, 3.0], atol=1e-6),
              f"15f: stack adds deltas commutatively (scale f1+f2-1: {st12.terms[0].factor.tolist()})")
        # Overlapping set + set refused; set + add allowed; scale + add allowed
        es1 = Fb.build(we.set_columns(2, [3], vec(112, 1, D, size=0.1)))
        es2 = Fb.build(we.set_columns(2, [3], vec(113, 1, D, size=0.1)))
        ea1 = Fb.build(we.add_to_columns(2, [3], vec(114, 1, D, size=0.1)))
        check(words_of(lambda: we.stack(Fb.lm, [es1, es2], model_id=Fb.model_id, weights_id=None), we.Refused) is not None,
              "15f: stack refuses two edits that both set the same neuron")
        st_sa = we.stack(Fb.lm, [es1, ea1], model_id=Fb.model_id, weights_id=None)
        check(st_sa.terms[0].op[0] == 1 and np.allclose(st_sa.terms[0].vectors[0], es1.terms[0].vectors[0] + ea1.terms[0].vectors[0], atol=1e-6),
              "15f: stack(set, add) sums vectors as a set")
        st_sca = we.stack(Fb.lm, [e1, ea1], model_id=Fb.model_id, weights_id=None)
        Wc3 = gw.dense_cols(Fb.mod(2), torch.tensor([3]))[0].float().numpy()
        check(st_sca.terms[0].op[0] == 2 and np.allclose(st_sca.terms[0].vectors[0], (2.0 - 1.0) * Wc3 + ea1.terms[0].vectors[0], atol=1e-5),
              "15f: stack(scale, add) lowers cleanly to additive delta (f-1)*W + v")
        # Different bases refused
        e_p_base = Pb.build(we.silence(2, [3]))
        check(words_of(lambda: we.stack(Fb.lm, [e1, e_p_base], model_id=Fb.model_id, weights_id=None), we.Refused) is not None,
              "15f: stack refuses edits from different bases")
        # edit_diagnostics
        diag_sil = we.edit_diagnostics(Pb.lm, e_p_base, dose=1.0)
        diag_d0 = we.edit_diagnostics(Pb.lm, e_p_base, dose=0.0)
        t_diag = diag_sil["terms"][0]
        check(diag_d0["terms"][0]["dtype_changed_share"] == 0.0 and t_diag["dtype_changed_share"] > 0.5
              and {"raw_rel_change", "stream_rel_change", "quiet_mask_energy_share", "write_ceilings", "dtype_changed_share"} <= set(t_diag)
              and t_diag["write_ceilings"]["max"] == 0.0,  # silenced column has 0 write ceiling
              f"15f: edit_diagnostics reports changed share, stream rel change, quiet share and write ceilings ({t_diag})")
        diag_row = we.edit_diagnostics(Pb.lm, Pb.build(we.silence_rows(2, "gate_proj", [3])), dose=1.0)
        check(diag_row["terms"][0]["kind"] == "rows" and diag_row["terms"][0]["dtype_changed_share"] > 0.5,
              "15f: edit_diagnostics handles row terms")

    with part("15g: the spread draw, a matched reading's coverage with nothing read, a plant's site scores"):
        # 1. drawn each on its own, control sets share sites; drawn spread, as few as the pools allow. Over many seeds,
        #    not one: a draw that locks at the end of a pass does so on some seeds only.
        def drawn(pools: Sequence[Any], C: int, spread: bool, seed: int = 3) -> List[np.ndarray]:
            return we._draw_matched_indices([np.asarray(list(p)) for p in pools], C, np.random.default_rng([seed, 0, 29]),
                                            spread=spread)

        def disjoint(draws: Sequence[np.ndarray]) -> int:
            seen: set = set()
            n = 0
            for d in draws:
                s = {int(x) for x in d}
                if not (s & seen):
                    n += 1
                    seen |= s
            return n

        def sound(draws: Sequence[np.ndarray], C: int, Kn: int) -> bool:
            return (len(draws) == C and all(len({int(x) for x in d}) == Kn for d in draws)
                    and len({tuple(sorted(int(x) for x in d)) for d in draws}) == C)

        def spread(pools: Sequence[Any], C: int, seed: int = 3, groups: Optional[Sequence[int]] = None,
                   offsets: Optional[Sequence[int]] = None) -> Tuple[List[np.ndarray], List[bool]]:
            pools = [np.asarray(list(p), dtype=np.int64) for p in pools]
            return we._draw_spread_sets(pools, [0] * len(pools) if groups is None else list(groups), C,
                                        np.random.default_rng([seed, 31]), offsets=offsets)

        def clean_family(sets: Sequence[np.ndarray], clean: Sequence[bool]) -> int:
            """How many sets the draw calls clean, or -1 when two of them share a site."""
            fam = [{int(x) for x in s} for s, c in zip(sets, clean) if c]
            return len(fam) if len(set().union(*fam)) == sum(len(s) for s in fam) else -1

        same60, same126 = [range(60)] * 10, [range(126)] * 2
        narrow = [range(49)] + [range(100, 2100)] * 9  # one neuron with few eligible sites, nine with many others
        narrow_in = [range(42)] + [range(3000)] * 9  # its few sites among the nine others' many
        nested = [range(42), range(200)] + [range(3000)] * 8
        tight = {"two neurons, the same 30 sites, 99 sets": ([range(30)] * 2, 99),
                 "three neurons, ten sites each, 39 sets": ([range(10), range(100, 110), range(200, 210)], 39),
                 "two neurons, the same 18 sites, 39 sets": ([range(18)] * 2, 39),
                 "two neurons, the same 20 sites, every one of the 190 sets there are": ([range(20)] * 2, 190),
                 "three neurons, the same 4 sites, all 4 sets": ([range(4)] * 3, 4)}
        short = {name: sum(1 for s in range(120) if not sound(drawn(pools, C, True, s), C, len(pools)))
                 for name, (pools, C) in tight.items()}
        check(all(n == 0 for n in short.values()),
              f"15g: the spread draw gives every set asked for where sets remain, on each of 120 seeds: it does not lock "
              f"when the sites run out ({short})")
        # against every assignment there is, on small pools that overlap as a band of scores makes them overlap
        rng_b = np.random.default_rng(7)
        n_brute = n_short = n_unclean = 0
        for _case in range(160):
            Kb = int(rng_b.integers(2, 5))
            Ub = int(rng_b.integers(Kb, 9))
            pools_b = [sorted(rng_b.choice(Ub, size=int(rng_b.integers(1, Ub + 1)), replace=False).tolist()) for _ in range(Kb)]
            groups_b = [int(g) for g in rng_b.integers(0, 2, size=Kb)] if rng_b.random() < 0.4 else [0] * Kb

            def key_of(combo: Sequence[int]) -> Optional[Tuple[Any, ...]]:
                by: Dict[int, List[int]] = {}
                for p, x in enumerate(combo):
                    by.setdefault(groups_b[p], []).append(int(x))
                if any(len(set(v)) != len(v) for v in by.values()):
                    return None
                return tuple((g, tuple(sorted(v))) for g, v in sorted(by.items()))

            every = {key_of(c) for c in itertools.product(*pools_b)} - {None}
            for C in sorted({1, max(1, len(every) // 2), len(every), len(every) + 3}):
                sets_b, clean_b = spread(pools_b, C, seed=n_brute, groups=groups_b)
                n_brute += 1
                keys = {key_of(s.tolist()) for s in sets_b}
                if len(sets_b) != min(C, len(every)) or None in keys or len(keys) != len(sets_b) or not keys <= every:
                    n_short += 1
                if clean_family(sets_b, clean_b) < 0:
                    n_unclean += 1
        check(n_brute > 400 and n_short == 0 and n_unclean == 0,
              f"15g: on {n_brute} draws from small overlapping pools, in one term or two, the spread draw gives as many "
              "sets as were asked or as there are, each a set there is and none twice, and the sets it calls clean "
              "share no site")
        dj_spread = [disjoint(drawn(same126, 39, True, s)) for s in range(30)]
        dj_each = [disjoint(drawn(same126, 39, False, s)) for s in range(30)]
        cl_narrow = [clean_family(*spread(pools, 39, s)) for pools in (narrow, narrow_in, nested) for s in range(30)]
        cl_shared = [clean_family(*spread(same60, 39, s)) for s in range(10)]
        n_shared = [min(len({int(d[j]) for d in drawn(same60, 39, True, s)}) for j in range(10)) for s in range(10)]
        check(set(dj_spread) == {39} and max(dj_each) < 39 and set(cl_narrow) == {39}
              and set(cl_shared) == {6} and set(n_shared) == {39}
              and all(sound(drawn(same60, 39, sp, s), 39, 10) for sp in (False, True) for s in range(10)),
              f"15g: drawn spread, 39 sets of two from 126 sites share no site (each on its own: {min(dj_each)} to "
              f"{max(dj_each)} of them share none); so do 39 of ten where one neuron has 42 or 49 eligible sites, among "
              "the others' sites or apart from them (a neuron with many sites leaves another's few alone); ten "
              "neurons that share 60 sites give the 6 sets that can share none, on every seed, and each neuron still "
              "has 39 different stand-ins")
        # the count does not fall as more sets are asked for: the sets made of sites no earlier set had are counted as
        # made, wherever the shuffle puts them (taken one after another as they stand, fewer would be found)
        asked_more = {}
        for name, pools, most in (("three neurons, the same 125 sites", [range(125)] * 3, 41),
                                  ("ten neurons, the same 450 sites", [range(450)] * 10, 45)):
            got39 = [clean_family(*spread(pools, 39, s)) for s in range(20)]
            sets99 = [spread(pools, 99, s) for s in range(20)]
            got99 = [clean_family(*x) for x in sets99]
            as_stand = [disjoint(x[0]) for x in sets99]
            asked_more[name] = (set(got39) == {39} and set(got99) == {most} and max(as_stand) < most
                                and all(sound(x[0], 99, len(pools)) for x in sets99), min(as_stand), max(as_stand))
        check(all(v[0] for v in asked_more.values()),
              "15g: asked for 99 sets where 41 or 45 can share no site, the spread draw makes that many of sites no "
              f"earlier set had and says which, and asked for 39 it makes 39 ({asked_more})")
        # two projections of one layer are one draw: a neuron stands in one control edit, whatever projection it is in
        two_proj, apart_draws = [], []
        for s in range(20):
            r2 = np.random.default_rng([s, 5])
            pools2 = [np.sort(r2.choice(2000, size=400, replace=False)) for _ in range(20)]
            sets2, clean2 = spread(pools2, 39, s, groups=[0] * 10 + [1] * 10)
            two_proj.append(clean_family(sets2, clean2))
            a2, b2 = spread(pools2[:10], 39, s)[0], spread(pools2[10:], 39, s + 1000)[0]
            apart_draws.append(disjoint([np.concatenate([x, y]) for x, y in zip(a2, b2)]))
        sets_l, clean_l = spread([range(5)] * 4, 2, 0, groups=[0, 0, 1, 1], offsets=[0, 0, 100, 100])
        check(set(two_proj) == {39} and max(apart_draws) < 39
              and clean_family(sets_l, clean_l) == 2 and all(int(s[2]) >= 100 and int(s[0]) < 100 for s in sets_l),
              f"15g: ten neurons changed in two projections of one layer are drawn together, and 39 control edits then "
              f"share no neuron; the two projections drawn each on its own share neurons between control edits "
              f"({min(apart_draws)} to {max(apart_draws)} of 39 share none); the same numbers in another layer are other sites")
        t_cost = time.time()
        sets_c, clean_c = spread([range(8000)] * 512, 99, 1)
        t_cost = time.time() - t_cost
        check(len(sets_c) == 99 and clean_family(sets_c, clean_c) == 15 and t_cost < 30.0,
              f"15g: 99 sets of 512 neurons that share 8000 sites are drawn in {t_cost:.2f} s (the 15 that can share no "
              "site among them): a draw that runs out of sites does not look through every pool at every pick")
        d_spr, d_again, d_other = drawn(same60, 39, True), drawn(same60, 39, True), drawn(same60, 39, True, seed=4)
        check(all(np.array_equal(a, b) for a, b in zip(d_spr, d_again))
              and not all(np.array_equal(a, b) for a, b in zip(d_spr, d_other)),
              "15g: the spread draw is its seed's")
        d_one = we._draw_matched_indices([np.arange(50)], 20, np.random.default_rng([3, 0, 29]), spread=True)
        d_one_was = we._draw_matched_indices([np.arange(50)], 20, np.random.default_rng([3, 0, 29]))
        check(all(np.array_equal(a, b) for a, b in zip(d_one, d_one_was)) and len({int(d[0]) for d in d_one}) == 20,
              "15g: one neuron's stand-ins were never repeated, and are drawn as they were")


        # 2. a plan asks for it; one that does not is written as it always was
        ctl_m = {"family": "other_sites_matched", "count": 6, "seed": 0, "tolerance": 3.0}
        e_two = {"terms": [{"layer": 2, "silence": [3, 40]}]}
        pl_was = lv.normalise({**paired_base, "name": "g1", "edit": e_two, "controls": ctl_m},
                              n_layers=NL, tokenizer=Pb.tok, dtype=Pb.dtype)
        pl_ind = lv.normalise({**paired_base, "name": "g1", "edit": e_two, "controls": {**ctl_m, "draw": "independent"}},
                              n_layers=NL, tokenizer=Pb.tok, dtype=Pb.dtype)
        pl_spr = lv.normalise({**paired_base, "name": "g1", "edit": e_two, "controls": {**ctl_m, "draw": "spread"}},
                              n_layers=NL, tokenizer=Pb.tok, dtype=Pb.dtype)
        check(pl_was["controls"] == pl_ind["controls"] and "draw" not in pl_was["controls"]
              and pl_spr["controls"] == {**pl_was["controls"], "draw": "spread"}
              and words_of(lambda: lv.normalise({**paired_base, "name": "g1", "edit": e_two,
                                                 "controls": {**ctl_m, "draw": "lucky"}},
                                                n_layers=NL, tokenizer=Pb.tok, dtype=Pb.dtype), lv.Refused) is not None
              and words_of(lambda: lv.normalise({**paired_base, "name": "g1", "edit": e_two,
                                                 "controls": {"count": 4, "seed": 0, "draw": "spread"}},
                                                n_layers=NL, tokenizer=Pb.tok, dtype=Pb.dtype), lv.Refused) is not None,
              "15g: controls.draw is 'independent' (written as a plan always was) or 'spread', for matched other sites alone")
        r_spr = lv.read(hP, {**paired_base, "name": "1b matched spread", "edit": e_two, "controls": {**ctl_m, "draw": "spread"}},
                        items_p)
        r_ind = lv.read(hP, {**paired_base, "name": "1b matched each", "edit": e_two, "controls": ctl_m}, items_p)
        printed(r_spr)
        cov_s, cov_i = r_spr["controls"]["coverage"], r_ind["controls"]["coverage"]
        check(r_spr["status"] == "ok" and r_ind["status"] == "ok" and cov_s["draw"] == "spread" and cov_i["draw"] == "independent"
              and r_spr["controls"].get("draw") == "spread" and "draw" not in r_ind["controls"]
              and cov_s["terms"][0]["min_distinct"] == 6 and cov_s["disjoint_sets"] == 6
              and cov_s["terms"][0]["min_eligible"] >= 12
              and cov_s["terms"][0]["min_eligible"] == cov_i["terms"][0]["min_eligible"]
              and "the spread draw" in " ".join(lv.lines(r_spr)) and "the spread draw" not in " ".join(lv.lines(r_ind)),
              f"15g (P): a reading says how its control sets were drawn and how many of them share no site with one "
              f"another ({cov_s['disjoint_sets']} of 6 spread, {cov_i['disjoint_sets']} each on its own, from "
              f"{cov_s['terms'][0]['min_eligible']} eligible sites a neuron)")
        # the count is the control edits' own: worked again here from the sites of the edits the reading recorded
        made_s, cov_made = we.matched_control_edits(Pb.build(we.silence(2, [3, 40])), Pb.lm,
                                                    lv._score_sites(hP, Pb.build(we.silence(2, [3, 40])),
                                                                    [items_p["orange"]], "paired"), 0, 6,
                                                    tolerance=3.0, spread=True)
        made_i, cov_made_i = we.matched_control_edits(Pb.build(we.silence(2, [3, 40])), Pb.lm,
                                                      lv._score_sites(hP, Pb.build(we.silence(2, [3, 40])),
                                                                      [items_p["orange"]], "paired"), 0, 6, tolerance=3.0)
        # an edit of the same neurons in two projections: the count is by neuron, over both terms
        e_both = Pb.build(we.silence(2, [3, 40]), we.scale_rows(2, "up_proj", [3, 40], 0.5))
        sc_both = lv._score_sites(hP, e_both, [items_p["orange"]], "paired")

        def neurons_of(c: Any) -> set:
            return {(int(t.layer), int(x)) for t in c.terms for x in t.idx}

        def disjoint_by_neuron(made: Sequence[Any]) -> int:
            seen_n: set = set()
            n = 0
            for c in made:
                if not (neurons_of(c) & seen_n):
                    n += 1
                    seen_n |= neurons_of(c)
            return n

        both_s, cov_both_s = we.matched_control_edits(e_both, Pb.lm, sc_both, 0, 6, tolerance=20.0, spread=True)
        both_i = [we.matched_control_edits(e_both, Pb.lm, sc_both, s, 12, tolerance=20.0) for s in range(8)]
        check(both_s is not None and len(e_both.terms) == 2 and cov_both_s["disjoint_sets"] == 6 == disjoint_by_neuron(both_s)
              and len(set().union(*(neurons_of(c) for c in both_s))) == 24
              and all(made is not None and cov["disjoint_sets"] == disjoint_by_neuron(made) for made, cov in both_i)
              and any(cov["disjoint_sets"] < disjoint([np.concatenate([t.idx + 1000 * i for i, t in enumerate(c.terms)])
                                                       for c in made]) for made, cov in both_i),
              "15g (P): for an edit of the same neurons in two projections the count of control edits that share no "
              "site is by neuron over both terms (a count that took each term's sites for its own would find more), "
              f"and drawn spread 6 control edits take 24 different neurons ({[cov['disjoint_sets'] for _m, cov in both_i]} "
              "of 12 drawn each on its own)")
        # asked for more control edits than can share no site: the count is of those made of sites no earlier one
        # had, which is more than taking them one after another as they stand finds, and a family of that size is
        # found again here from the control edits' own sites (those that share a site with the fewest others first)
        many_s, cov_many = we.matched_control_edits(Pb.build(we.silence(2, [3, 40])), Pb.lm,
                                                    lv._score_sites(hP, Pb.build(we.silence(2, [3, 40])),
                                                                    [items_p["orange"]], "paired"), 0, 99,
                                                    tolerance=20.0, spread=True)
        sites_many = [neurons_of(c) for c in many_s or ()]
        left = list(range(len(sites_many)))
        family: List[int] = []
        while left:
            pick = min(left, key=lambda c: (sum(1 for d in left if d != c and sites_many[c] & sites_many[d]), c))
            family.append(pick)
            left = [d for d in left if d != pick and not (sites_many[pick] & sites_many[d])]
        check(many_s is not None and len(many_s) == 99
              and disjoint_by_neuron(many_s) < cov_many["disjoint_sets"] <= len(set().union(*sites_many)) // 2
              and len(family) >= cov_many["disjoint_sets"],
              f"15g (P): asked for 99 control edits of two neurons where {cov_many['disjoint_sets']} can share no site, "
              f"the coverage counts those ({disjoint_by_neuron(many_s)} are found taking them one after another as "
              f"they stand; a family of {len(family)} is found again from the control edits' own sites)")
        check(made_s is not None and cov_made["disjoint_sets"] == disjoint([c.terms[0].idx for c in made_s]) == 6
              and [c.id for c in made_s] == r_spr["controls"]["ids"]
              and made_i is not None and [c.id for c in made_i] == r_ind["controls"]["ids"]
              and cov_i["disjoint_sets"] == cov_made_i["disjoint_sets"] == disjoint([c.terms[0].idx for c in made_i]),
              f"15g (P): the count of sets that share no site is the one the control edits' own sites give, drawn "
              f"spread ({cov_made['disjoint_sets']}) and each on its own ({cov_made_i['disjoint_sets']})")
        # 3. the same coverage with no arm read and nothing entered
        n_book = lb.verify()["entries"]
        asked0 = dict(lv.asked_in_process())
        cov_look = lv.matched_coverage(hP, {**paired_base, "name": "1b look", "edit": e_two,
                                            "controls": {**ctl_m, "draw": "spread"}}, items_p)
        asked1 = lv.asked_in_process()
        cov_s_counts = {**cov_s, "terms": [{k: v for k, v in t.items() if k != "edit_scores"} for t in cov_s["terms"]]}
        check(cov_look == cov_s_counts and all("edit_scores" in t for t in cov_s["terms"])
              and lb.verify()["entries"] == n_book and asked1["readings"] == asked0["readings"]
              and asked1["forwards"] == asked0["forwards"] + 1
              and lv.matched_coverage(hP, {**paired_base, "name": "1b look", "edit": e_two}, items_p) is None
              and words_of(lambda: lv.matched_coverage(hP, {**paired_base, "name": "1b look", "edit": e_two, "doses": [99.0],
                                                            "controls": ctl_m}, items_p), lv.Refused) is not None
              and words_of(lambda: we.assert_base(Pb.lm, hP.hooks_at_load), we.NotBase) is None,
              "15g (P): matched_coverage gives the counts of the coverage a reading would find and not the scores of "
              "the edit's own neurons, with one base forward an item, no reading counted, nothing entered in the "
              "ledger, and the refusals a reading would meet")
        # 4. a reading under a plant scores a neuron as the planted model has it
        v_new = vec(211, 1, D, size=0.03)
        plant_set = Pb.build(we.set_columns(2, [5], v_new))
        plant_sil = Pb.build(we.silence(2, [5]))
        e_on = Pb.build(we.scale(2, [5, 40], 0.5))
        base_n = we.down_stream_norms(Pb.lm, 2)
        set_n = we.down_stream_norms_under(Pb.lm, 2, plant_set)
        w_post = we._w_post(Pb.lm, 2).astype(np.float64)
        want_5 = float(np.sqrt(((v_new[0].astype(np.float64) * w_post) ** 2).sum()))
        other = np.ones(base_n.shape[0], dtype=bool)
        other[5] = False
        sc_base = lv._score_sites(hP, e_on, [items_p["orange"]], "paired")[e_on.terms[0].linear]
        sc_set = lv._score_sites(hP, e_on, [items_p["orange"]], "paired", plant=plant_set)[e_on.terms[0].linear]
        sc_sil = lv._score_sites(hP, e_on, [items_p["orange"]], "paired", plant=plant_sil)[e_on.terms[0].linear]
        check(close(set_n[5], want_5, 1e-6) and np.array_equal(set_n[other], base_n[other])
              and float(we.down_stream_norms_under(Pb.lm, 2, plant_sil)[5]) == 0.0
              and we.down_stream_norms_under(Pb.lm, 3, plant_set) is we.down_stream_norms(Pb.lm, 3)
              and sc_base[5] > 0.0 and sc_set[5] > 0.0 and close(sc_set[5] / sc_base[5], want_5 / base_n[5], 1e-4)
              and np.allclose(sc_set[other], sc_base[other], rtol=1e-6, atol=0.0) and sc_sil[5] == 0.0
              and words_of(lambda: we.assert_base(Pb.lm, hP.hooks_at_load), we.NotBase) is None,
              f"15g (P): under a plant that sets a column, the neuron is scored by its activation times the new column's "
              f"length ({sc_set[5]:.5g}, where the per-row form's nought times the old column would give 0), every other "
              f"neuron as on the base; a silenced one scores 0, which is the planted model's own")
        ctl_p = {"family": "other_sites_matched", "count": 4, "seed": 0, "tolerance": 50.0}
        r_under = lv.read(hP, {**paired_base, "name": "1b under a set plant", "folded": "never", "controls": ctl_p},
                          items_p, edit=e_on, _plant=plant_set)
        check(r_under["status"] == "ok" and r_under["controls"]["coverage"]["sufficient"] is True
              and r_under["base_after"] is True,
              f"15g (P): a matched reading of an edit that touches a planted set column is read, not ended for want of "
              f"stand-ins ({r_under['status']}: {r_under['why']})")
        # 5. a low-rank term that is not a pair is answered in words
        bad_lr = [SimpleNamespace(kind="lowrank", linear="layers.2.mlp.down_proj", which="down_proj", A=None, B=None),
                  SimpleNamespace(kind="lowrank", linear="layers.2.mlp.down_proj", which="down_proj",
                                  A=np.float32(1.0), B=np.zeros((D, 1), dtype=np.float32))]
        said = [we.static_caps(SimpleNamespace(terms=(t,))) for t in bad_lr]
        check(all(isinstance(s, str) and "is not a pair A [rank, in], B [out, rank]" in s for s in said),
              f"15g: static_caps says in words that a low-rank term with no A, or an A that is one number, is not a pair ({said})")


# --------------------------------------------------------------------------------------------- 16. the words

KEPT_OUT = ("proof", "held", "seal", "erratum", "sighting", "plate", "station", "field book", "chance edit")  # spec, section 1
# each word, and what is made of it: its plural and its other endings ("proofs", "sealed", "stations", "errata",
# "field-book"). `held` stays as it is. Wider than the spec's "as a word": a made-up ending would slip by otherwise.
KEPT_OUT_PATTERN = r"\b(proof\w*|held|seal\w*|errat(?:um|a)|sighting\w*|plate\w*|station\w*|field[\s-]+books?|chance[\s-]+edits?)\b"


def check_words() -> None:
    pattern = re.compile(KEPT_OUT_PATTERN, re.I)
    check(all(pattern.search(f"a {w} b") for w in KEPT_OUT) and all(pattern.search(w) for w in ("two stations", "sealed", "the proofs", "errata", "plates", "a field-book", "chance edits"))
          and not pattern.search("withheld a template of the estate"), "15: (the pattern sees each kept-out word and what is made of it, and not a word that only holds one)")
    for path in SOURCES:
        if not path.exists():
            check(False, f"15: {path.relative_to(ROOT_DIR)} is not there to be read")
            continue
        found = [(i + 1, m.group(0)) for i, line in enumerate(path.read_text().splitlines()) for m in pattern.finditer(line)]
        check(not found, f"15: {path.relative_to(ROOT_DIR)} uses none of the words the spec keeps out" + (f" (line, word: {found[:6]})" if found else ""))
    found = sorted({m.group(0).lower() for line in PRINTED for m in pattern.finditer(line)})
    check(len(PRINTED) > 0 and not found, f"15: nothing among the {len(PRINTED)} lines printed above uses them" + (f" ({found})" if found else ""))


def main() -> int:
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)  # every forward of 1a runs under no_grad (spec section 4)
    print(f"the check's ledger and saved edits: {TMP}")
    if not CLI.exists():
        MISSING["scripts/levelling.py"] = "the file is not there"
    for name, why in MISSING.items():
        check(False, f"{name if '/' in name else f'backend/{name}.py'} cannot be used: {why}")
    Fb = Pb = hF = hP = None
    with part("the float32 stand-in"):
        Fb = Box(asi.stand_in()[0], "F")
    with part("the packed bf16 stand-in"):
        import check_dequant as cd

        Pb = Box(cd.packed_stand_in()[0], "P")
    if lv is not None and we is not None and lb is not None:
        with part("handle() on the float32 stand-in"):
            hF = lv.handle(Fb.rt)
        with part("handle() on the packed stand-in"):
            hP = lv.handle(Pb.rt)
    all_three = ("weight_edits", "level_book", "levelling")
    run_section(1, "the hashes", all_three, check_hashes, Fb)
    run_section(2, "the gate", ("weight_edits",), check_gate, Fb, Pb)
    run_section(3, "save and load", ("weight_edits",), check_save_load, Fb, Pb)
    run_section(4, "the per-row hooks, by bits", ("weight_edits", "levelling"), check_bits, Pb)
    run_section(5, "the per-row hooks, to rounding", ("weight_edits", "levelling"), check_rounding, Fb, Pb)
    run_section(6, "a decode step", ("weight_edits", "levelling"), check_decode, Fb, Pb)
    run_section(7, "control edits", ("weight_edits",), check_controls, Fb)
    run_section(8, "items and plans", all_three, check_items_plans, Fb)
    run_section(9, "the read-outs against brute force", all_three, check_readouts, Fb, hF, Pb, hP)
    run_section(10, "one reading, end to end", all_three, check_end_to_end, Fb, Pb, hF, hP)
    run_section(11, "the written reading", all_three, check_written, Pb, hP)
    run_section(12, "the ledger", ("level_book",), check_book)
    run_section(13, "the estimate", all_three, check_estimate)
    run_section(14, "the command line", ("level_book", "scripts/levelling.py"), check_cli)
    run_section(15, "slice 1b and bugfixes", all_three, check_1b, Fb, Pb, hF, hP)
    run_section(16, "the words", (), check_words)
    took = time.perf_counter() - t0
    print("\nby section (checks that hold, checks that fail):")
    for title, good, bad in TALLY:
        print(f"  {title}: {good}, {bad}")
    shutil.rmtree(TMP, ignore_errors=True)
    if failures:
        print(f"\nlevelling check: {len(failures)} of {passed + len(failures)} checks FAILED ({took:.0f} s)")
        return 1
    print(f"\nlevelling check: all {passed} checks hold ({took:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
