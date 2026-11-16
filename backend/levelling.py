"""The paired reading (`backend/levelling.py`): one edit of the base weights read against the untouched model in the
same forward, at two row counts, beside null rows and control edits; and the written reading, the same rows writing.

What it is. An edit (`backend/weight_edits.py`) is never left in the model: it is an argument of a reading. `read`
takes a plan (a dict: the edit as a recipe, the items, doses, how many control edits, the read-outs), runs it, returns
one JSON-clean dict and appends the same dict to the ledger (`backend/level_book.py`), whatever happened: a reading
that was refused or came out void is a result and is entered like any other. It prints numbers, ranks against control
edits, floors, and the two words "void" and "unstable". It gives no word to an edit.

An item is a question and an answer text in the model's own chat format, thinking off (`make_item`). The figures are
read along the answer given, piece by piece, at the positions that predict the answer's own pieces; they do not say
what the edited model would write. The written reading (`kind: "written"`) lets the rows write instead, greedily.

How a paired reading is laid out (`layout`). Every forward of a class has the same number of rows: row 0 is the base
row, untouched; the last row is its twin, untouched; rows 1 to n0 are null rows, each scaling the feed-forward output
of the edit's lowest layer by 1 + e; the rest are arms: the edit at each dose (all in forward 0), then control edits.
Every forward carries its own base row and an arm is read against the base row of the forward it sat in. Each item is
read twice, at `rows.small` rows and at `rows.large`, and an item on which an arm of the edit disagrees with itself
between the two is "unstable" (`agree`). The control edits' arms are set side by side the same way and their ratios
printed beside the edit's, not judged (since 16 November 2026: the spec's open point 5, below).

What voids a reading (the mechanism is then not sound, and what was read so far is written with status "void"): a
base row that is not finite; an untouched row that is not the base row by bits; an arm at dose 0 that is not the base
row by bits; a later forward whose base row or null rows are not forward 0's by bits; running out of memory (a
reading never halves its rows: the row count is part of what it is); a model that is not the base when the reading
ends. "By bits" is `torch.equal` on final-normed states.

The measured facts this rests on. All 14 November 2026 on the RTX 5090, Gemma-4-31B-it W4A16, compiled unpacker, lens
freed, server stopped, one run each, unless said otherwise (experiments/results/weight_space_exploration/pilot).
- Rows of one forward do not leak: an untouched row far from edited rows equals row 0 by bits, and the same edit in
  rows 1, 9, 20 and 31 of 32 reads the same [measured, pilots 1 and 6].
- A restart from layer L on a stream kept at the same batch shape gives the whole forward's bits at layers 20, 40,
  44, 52 and 58, unhooked [measured, pilot 1], and under hooks at layers 5, 20 and 44, at 12 and at 32 rows, where
  `lower: "whole"` gave the same figures exactly [measured, the card check of 16 November 2026, section 6]; every
  reading shows it again by rule (a later forward's base row and null rows must be forward 0's by bits).
- A one-row capture replayed at 32 rows agreed with a same-shape forward at layer 44 and not at layers 5 and 20, on
  raw text with no start mark [measured, pilot 3]. On chat-formatted items it did not agree to the floor at layer 44
  or at layer 40: with each of two items' 64 hardest-firing neurons silenced, the one-row reading stood off the
  same-shape one by 10 to 35 times the item's floor in 7 of 8 readings (on a study chat at layer 44, 0.0039 and 0.0033
  nats of KL against floors of 0.00013 and 0.00011, 1.6% and 1.3% of the reading) and under the floor once [the card
  check of 16 November 2026, section 6b]. `lower: "one_row"` is refused below `ONE_ROW_MIN_LAYER`; at or above it it
  is a cheaper reading that is not the same-shape one, and the result says it ran so.
- On raw text with no start mark one folded edit read 0.0091 nats at 2 rows and 0.7258 at 32; at 150 pieces there
  was one arithmetic up to 12 rows and another from 16. The same passages as chat-formatted answers showed no reading
  differing over fivefold, and at 91 pieces of chat-formatted text six batch sizes gave six different base rows
  [measured, pilots 4 to 7]. Two row counts agreeing does not show a third would.
- Null rows read 0.00014 to 0.0066 nats by text and layer (the 0.0066 on the raw recipe with no start mark): no
  constant floor serves [measured, pilot 5]. The floor here is the largest null row of the item, in the metric's own
  units. It is the largest of a few draws, not a bound: a single neuron silenced at random, a row with next to nothing
  in it, read above its forward's floor in 26 of 72 readings, up to 2.25 times it [measured, pilot 8, the model's own
  answers, layers 19 to 54] (`FLOOR_SAY`).
- The base's mean log-probability of the scored span: its own greedy answers -0.07 to -0.17 nats a piece; a person's
  passage given as the answer -2.46 and -5.51; the same passages raw, with no start mark, -9.15 and -10.09
  [measured, pilot 8]. Over 51 items on the card: the model's own answers -0.07 to -0.17, the study's chats -0.000002
  to -9.40, passages given as answers -2.45 to -8.61, raw text with no start mark -8.12 to -12.13, the same raw texts
  behind a start mark -3.07 to -11.38 [measured, the card check of 16 November 2026, section 8b].
- The model's own bf16 logits differ from fp32 softcapped logits by a mean KL of 0.0013 nats, the size of what is
  read here [measured, pilot 1, on a prose passage with no start mark], so every paired read-out is fp32
  (`causal_check.logits_fp32`). A position's KL is kept from going below nought, so a mean here reads at or above
  the pilots' plain mean where a row is at float32's own noise: on the float32 stand-in a null row read 8.3e-8 where
  the plain mean was 4.1e-8 [ran on the CPU, 14 November 2026; not measured on the card].
- `atlas_ops.write_rows` under a hook registered round it: untouched rows wrote the same 48 pieces with the same
  log-probabilities, with an output-side hook and no row stopping [measured, pilot 3].

The cost figures (`COST`). The three time figures and the memory a row-position were refit on 16 November 2026 from
the card check's own readings (`experiments/results/levelling/verify_1a.json`, `proposed_bars.cost`: a least-squares
fit over 331 paired readings, and over the classes' peaks for the memory; the build spec's 6.2): 0.135 s a forward
and 0.377 ms a row-position (a restart counted as its share of the layers), 0.088 ms a scored row-position of fp32
full-vocabulary read-out, and 0.191 MiB a row-position of memory above the model. They price a reading, its own work
round the forwards included, not a bare forward: a whole forward alone was 0.08 s + 0.33 ms a row-position (within
15% on nine shapes) and the read-out 0.11 ms a scored row-position [measured 14 November 2026, pilot 1], and those
figures put the 163 readings the card check judged 1.02 to 1.44 times their prediction (the 145 predicted under a
second, not judged, 1.15 to 1.87), the worst where a reading carried 19 control edits (21 of 40 such readings more
than a quarter over, none of the other 123). The refit puts all 163 within 0.85 to 1.13 of their prediction, but it
was fit on them: the next card run is its test. The rest stand as measured
before: a decode step 0.095 s hooked and unhooked at 8 rows [pilot 3; the card check read 0.110 s hooked and 0.107 s
unhooked, section 7b]; a column overlay about 0.3 ms a cloned matrix, to 64 columns [pilot 1]; load 9.5 s and
unpacker install 1.83 s [pilot 1, no lens; the card check's load, the lens loaded and then freed, took 16.0 s and its
install with the self-test 3.46 s]; writing with a cache about 0.8 MB a row a word-piece of context [measured 28
October 2026, lens loaded; on the card 8 rows to 74 pieces peaked 0.73 GiB above the model at rest, where that rate
gives 0.44]. The estimate counts each forward's base row, the null rows once, every arm and the twin's row; the code
reads the base row of forward 0 once more for the item gate's figure, and the twin's base and per-row rows again.

The bars, set by hand on 16 November 2026 from the card check (`verify_1a.json`, its `proposed_bars`), each by the
rule the build spec wrote for it before the card ran (BUILD_SPEC_1a.md, section 9; what was found and set is in its
section 12). One card run each, so all of them stay provisional:
- `ITEM_GATE_BAR` stays -6.0 nats a piece and is a printed caution, not a refusal (open point 4: "if the card finds
  that the figure does not separate, the gate stays as a printed caution and the two row counts carry the weight").
  With the gate set aside the card read 51 items: in the four readings of its section 9 (no control edits) 3 read
  "unstable" (-3.09, -4.13 and -9.64 nats a piece) and 48 did not (the lowest -12.13), so the figure does not
  separate; with 19 control edits 3 more read "unstable" from control arms alone under the rule as it then stood, all
  under the bar (-8.61, -8.82 and -10.08), which leaves that so. An item under the bar is read in both classes like any
  other; its `gate.passed` is false and `gate.caution` says, in `GATE_SAY`'s words, why it was read anyway.
- `STABLE_RATIO` stays 3.0 (open point 5: "ship as written"). On items over the bar the largest ratio that agreed was
  2.44 (a read-out that agreed in a reading whose other read-out did not; among readings that agreed on both
  read-outs, 2.07), and 3 readings disagreed (up to 20.7). The point's other clause, "if §9a shows control arms at
  the floor tripping it, read the rule on the edit's arms alone and print the control arms' ratios", is applied as
  written (16 November 2026; the coordinating session decided on 14 November that every open point follows the spec's
  recommendation). Its condition was met: 6 of 969 control arms disagreed, every one on an item under the bar, and
  three of them sat at the floor at one row count (control 14 on prose 10 as an answer, 3.34 on KL; control 15 on the
  raw prose 3 with no start mark, 3.19 on the text's own log-probability, its two readings of opposite signs; control
  9 on the raw recipe with no start mark, 7.57 and 7.40, an item the edit's own arms made unstable too). So an item is
  judged on the edit's arms alone, and each read-out's control-edit ratios are kept (`stability.metrics.<m>.controls`)
  and printed beside them, not judged; the number, the floor and the rest of the rule are as they were. The edit's
  arms disagreed on the same 3 items in section 9 and in 9a, and agreed on the other 48. Read as written, the clause
  sets aside control arms clear of the floor too: on the raw recipe cut with no start mark controls 16 and 2 read
  31.1 and 6.43 times apart on KL, neither at the floor, and that item now reads as its edit's arms do.
- `ONE_ROW_MIN_LAYER` stays 44: open point 14 wrote no rule for moving it, and the card's reading of the one-row
  capture is above.
- `ROWS_SMALL` and `ROWS_LARGE` stay 12 and 32 (open point 3: "set by hand" once the map is read, with no rule for
  other numbers). The map, the base row's bits at 2, 3, 4, 6, 8, 12, 16, 24, 32 and 48 rows: at 35 pieces every row
  count gave its own; at 82 only 12 and 24 were one; at 175, 24, 32 and 48 were one. 12 and 32 never shared bits.
- `COST`: refit, above. `MAX_WRITTEN_GIB` stays 6.0.

Read on the card for the first time on 16 November 2026 (the card check, one run): an input-side factor of 2 is the
folded form by bits, as a silencing is (48 comparisons of 48: 1, 64 and 1,000 columns at layers 5 and 20, at 12 and 32
rows, on two items; sections 5 and 5a); a scale by 1.5 or by -3, a set and the float32 output-side add stand off their
folded twins by 0.00002 to 0.009 nats of KL, 0.04% to 22% of the reading (section 5c); a rank-1 low-rank term read as
the one-column add it equals to 0.00004 and 0.00017 nats (section 5e); under the input-side pre-hook in a decode step,
with two rows stopping, untouched rows and a dose-0 row wrote the base's pieces with its log-probabilities exactly
(section 7); the stability rule on the text's own log-probability (section 9) and on log-odds (six items, section 9a);
control arms' second readings (section 9a); a twin of 1,000 columns, 1.02 to 1.13 times its price at the figures then
shipped (section 10). Not read on the card: a low-rank term that is not a column's add (one of rank 64 was timed, not
compared), and any layer but 5, 20, 40 and 44.

What it refuses, as results with plain words: a plan with a key it does not know or a number out of range
(`normalise`); null rows that would be the base row again or an exact scaling; a forward over `MAX_ROW_POSITIONS`; a
written reading whose cache would not fit; a dose the rounding eats, and an edit that is the base at every dose; an
answer that holds a start mark or another of the tokenizer's special pieces (they would be scored as text); an `Edit`
handed in whose id is not the hash of what it holds; a model that carries a vault edit or a hook that was not there at
load. Any other exception inside a reading, an interrupt included, is entered as void and raised again, so every call
is in the ledger. An item whose base log-probability is under the bar is not refused (since 16 November 2026, above):
it is read with the gate's caution.

What the stability rule does not see [found at the build's review, 14 November 2026, by a planted fault on the CPU]:
an arm's two readings are set against the LARGER of the two row counts' floors, and the two floors are not themselves
set side by side. A null row that moves with the row count therefore lifts the floor and an edit that moved with it
"agrees". The printed lines say when an item's two floors stand apart by more than the ratio bar. The card check
recorded how far they part: by more than the bar in 22 of 204 readings, at most 42.7 times [measured, 16 November
2026, section 9a]. Whether that should make an item "unstable" is still the owner's to decide.

The written reading is greedy, taken at one row count, and is not a two-row-count reading: every untouched row and
an arm at dose 0 must write the base row's pieces with its log-probabilities, exactly, or it is void. Its
log-probabilities are the writer's own, from the model's dtype logits, not the fp32 measure.

A low-rank term (on a down_proj only in 1a: the gate of `weight_edits` refuses the other two) has no folded form, so
no twin is taken for it and a rank is all it gets.

On the float32 stand-in (scripts/atlas_stand_in.py) the same code runs in float32 at one thread; its linears are
plain, so no folded twin is taken there, and its items read about -6.4 nats a piece, so a stand-in plan gives its own
bar.

What was run, and what was not. On the card: the card check of 16 November 2026 (scripts/verify_levelling.py: 369
readings through `read`, into a ledger of its own beside its JSON), 142 of its 145 standing lines passing. The three
that did not: the cost model (above), and two lines asking that no null row has a position above a nat. Every null row
that had one was on raw text: 28 of the 784 null rows read on the raw items (26 of them behind the start mark; 12
distinct rows on three items, since the readings at one layer and row count repeat their null rows), and none of the
2,208 read on the model's own answers (the one with its start mark dropped among them), the study's chats and
passages given as answers.

On the CPU, 14 November 2026 [ran], at one thread, against the repository's `weight_edits` and `level_book`: on the
float32 stand-in, a restart from the kept stream under hooks equals the whole forward under the same hooks by bits at
layers 0, 2 and 5; KL, the text's own
log-probability and log-odds of every arm equal a direct computation from a whole forward's full log-softmax to
1e-5; `lower: "whole"` gives the figures of `"same_shape"`; a one-row capture reads as `"same_shape"` does to 1e-5
(float32 on the CPU: this says nothing of bf16 on the card); planted hooks give each void its code, and one that acts
at the large row count alone makes the item unstable. On a packed bf16 stand-in: `"whole"` and `"same_shape"` give
the same figures exactly; a silencing and a scale by 2 or -0.5 read folded are the per-row row by bits at 12 and at
32 rows; for a scale by 1.5 or -3, a set and an add the mean KL between the per-row row and its folded twin read
1.4e-6 to 1.6e-6 (two or three columns of a 128-neuron layer, one 40-piece item).

Layers count from 0 in code and files; what a person reads says both ("layer 20 (page 21)").
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import backend.engine_runtime
from backend import atlas_ops as ao
from backend import level_book as lb
from backend import weight_edits as we
from backend.causal_check import language_model, logits_fp32, model_device
from backend.census import file_sha256, runtime_record
from backend.cpu_memory import is_oom
from backend.runtime_factory import weights_card
from backend.theodolite_math import FINAL_LOGIT_SOFTCAP
from backend.w4a16_fast import describe as unpacker_describe


class Refused(ValueError):
    """A plan, an item or a reading's size that 1a does not take, in plain words."""


VERSION = 1
ITEM_GATE_BAR = -6.0        # nats a piece [provisional]; a printed caution, not a refusal, since 16 November 2026
STABLE_RATIO = 3.0          # [provisional; kept on 16 November 2026 (the spec's open point 5), and read on the edit's
#                             arms alone since then, as the point's second clause asks]
ONE_ROW_MIN_LAYER = 44      # [provisional: the one layer where the one-row capture was shown, on raw text with no
#                             start mark; on chat-formatted items the card read it off the same-shape reading at layers
#                             40 and 44, 16 November 2026, and no rule was written for moving the bar]
ROWS_SMALL, ROWS_LARGE = 12, 32          # [provisional; unchanged 16 November 2026: open point 3 wrote no rule for
#                                          other numbers (on the card's map, 12 and 32 never shared bits)]
SMALL_MIN, SMALL_MAX, LARGE_MIN, LARGE_MAX = 5, 12, 16, 64
WRITTEN_ROWS_MIN, WRITTEN_ROWS_MAX = 5, 32
NULL_E = (0.03125, -0.03125, 0.125, -0.125)
NULL_E_WRITTEN = (0.125, -0.125)
MAX_PIECES, MAX_ROW_POSITIONS, MAX_CONTROLS, MAX_DOSES, MAX_ITEMS = 512, 12288, 199, 8, 256
MAX_WRITTEN_GIB = 6.0       # [provisional: 8.37 GiB was free after the load, measured 14 November 2026, lens freed]
READ_CHUNK = 128            # positions of full-vocabulary fp32 logits worked at once
COST = {"forward_fixed_s": 0.135, "row_position_s": 0.000377, "readout_row_position_s": 0.000088,
        "decode_step_s": 0.095, "overlay_matrix_s": 0.0003, "load_s": 9.5, "unpacker_s": 1.83,
        "mib_row_position": 0.191, "mb_row_piece_written": 0.8}
# forward_fixed_s, row_position_s, readout_row_position_s and mib_row_position: refit on 16 November 2026 from the card
# check's readings (experiments/results/levelling/verify_1a.json, proposed_bars.cost, to three figures; a reading's
# cost, not a bare forward's); decode_step_s, overlay_matrix_s, load_s and unpacker_s measured 14 November 2026;
# mb_row_piece_written measured 28 October 2026
CONTROLS_SAY = ("Control edits are other sites of the same layer and size, drawn from seed {seed}. They are NOT "
                "matched on how hard the sites fire on these items or on what the edit delivers, and the edit's own "
                "sites were chosen where theirs were drawn, so a rank against them says only that the edit differs "
                "from an unmatched random edit of its size.")
ROWS_SAY = ("Every item was read at {small} rows and at {large} rows. Agreement at these two row counts does not "
            "show that another row count would agree: on 14 November 2026 batches of 2, 3 and 4 rows each read "
            "differently.")
PAIRED_SAY = ("These figures are read along the answer given, piece by piece. They do not say what the edited model "
              "would write.")
FLOOR_SAY = ("The floor is the largest of {n0} null rows, not a bound: on 14 November 2026 a single neuron silenced at "
             "random read above its forward's floor in 26 of 72 readings, up to 2.25 times it.")
# the item gate's caution, after the item's own figures (`_gate_caution`): the spec's open point 4, as the card found
# it. Since 16 November 2026 it names the EDIT's two row counts: its arms disagreed on the same 3 items in sections 9
# and 9a and agreed on the other 48, while 3 of those 48 had read "unstable" in 9a from control arms alone (open
# point 5)
GATE_SAY = ("The bar is a caution, not a refusal: on 16 November 2026 the card read 51 items with it set aside, and "
            "the base's log-probability did not separate the 3 on which the edit's two row counts disagreed (-3.09, "
            "-4.13 and -9.64 nats a piece) from the 48 on which they agreed (the lowest -12.13). The item is read, "
            "and its two row counts carry the weight.")

MAX_NULL_ROWS, MAX_NAMED_PIECES, MAX_NEW_MAX = 8, 16, 256
WRITTEN_ROWS, WRITTEN_MAX_NEW, WRITTEN_CONTROLS = 8, 48, 3
METRICS = ("kl", "own", "odds")
SIGNED = {"kl": False, "own": True, "odds": True}
FOLDED_BASE_FROM = "forward 0 of the same shape"
WHY_NO_TWIN = {"never": "the plan says folded: never",
               "exact": ("a silencing or a scale by a power of two is exact in arithmetic on the input side; on the "
                         "card a silencing and a scale by 2 read as their folded forms by bits, 48 comparisons of 48 "
                         "(16 November 2026)"),
               "lowrank": "the edit holds a low-rank term, which has no folded form in 1a"}
_PAIRED_KEYS = ("name", "kind", "edit", "items", "doses", "main_dose", "controls", "null_e", "rows", "readouts",
                "rank_by", "folded", "lower", "bars")
_WRITTEN_KEYS = ("name", "kind", "edit", "items", "doses", "main_dose", "controls", "null_e", "rows", "max_new")


# ------------------------------------------------------------------------------------------------- small helpers


def _clean(x: Any) -> Any:
    """`x` as plain JSON values by exact type: dict with string keys, list, str, int, float, bool, None; a number
    that is not finite becomes None, a tuple a list, a numpy or torch scalar a Python one."""
    if x is None or type(x) is bool or type(x) is str:
        return x
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        return str(x)
    if isinstance(x, Mapping):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    tolist = getattr(x, "tolist", None)  # a numpy scalar or array, a tensor
    if callable(tolist):
        shape = tuple(getattr(x, "shape", ()) or ())
        if shape:  # an array: said by what it is, never by its numbers (it may hold pieces of the model's columns)
            return f"<{type(x).__name__} {str(getattr(x, 'dtype', '')).replace('torch.', '')} {list(shape)}>"
        return _clean(tolist())
    return str(x)


def _sha(obj: Any) -> str:
    return hashlib.sha256(lb.canonical(obj)).hexdigest()


def _number(x: Any) -> Optional[float]:
    """`x` as a float when it is a finite int or float (a bool is neither), else None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        return float(x) if math.isfinite(x) else None
    except OverflowError:  # a whole number past what a float holds
        return None


def _whole_number(x: Any) -> Optional[int]:
    return int(x) if isinstance(x, int) and not isinstance(x, bool) else None


def _dtype_word(dtype: Any) -> str:
    """A dtype as a refusal names it ("bfloat16", "float32"), as `weight_edits` names it in the same sentences."""
    return str(dtype).replace("torch.", "")


def _exact(g: float) -> bool:
    """0 or a power of two with either sign: a factor the model's dtype applies with no rounding."""
    return g == 0 or math.frexp(abs(g))[0] == 0.5


def _low_median(xs: Sequence[Optional[float]]) -> Optional[float]:
    """The lower middle of the numbers given (torch's median), None when there is none."""
    got = sorted(x for x in xs if x is not None)
    return got[(len(got) - 1) // 2] if got else None


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    got = [x for x in xs if x is not None]
    return sum(got) / len(got) if got else None


def _key(what: str, x: Any) -> str:
    """An arm's key: "null:<e>" and "edit:<dose>" with the number as asked, "control:<c>" with the integer."""
    return f"control:{int(x)}" if what == "control" else f"{what}:{repr(float(x))}"


def whole(rows: int, pieces: int) -> float:
    """The seconds one whole forward of `rows` rows of `pieces` word-pieces adds to a reading (refit on 16 November
    2026 from the card check's readings; a bare forward measured 0.08 s + 0.33 ms a row-position on 14 November)."""
    return COST["forward_fixed_s"] + COST["row_position_s"] * rows * pieces


# ---------------------------------------------------------------------------------------------------------- items


@dataclass(frozen=True)
class Item:
    """A question and an answer as the model is given them: `prompt_ids` (the chat prompt, thinking off) and
    `answer_ids` (the answer, tokenised apart). The three strings are None for an item built from ids."""

    name: Optional[str]
    question: Optional[str]
    answer: Optional[str]
    system: Optional[str]
    prompt_ids: Tuple[int, ...]
    answer_ids: Tuple[int, ...]
    id: str

    @property
    def ids(self) -> Tuple[int, ...]:
        return self.prompt_ids + self.answer_ids


def chat_ids(tokenizer: Any, question: str, system: Optional[str] = None) -> List[int]:
    """The ids of a question in the model's own chat format, thinking off, the model's turn opened."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _item(prompt_ids: Sequence[int], answer_ids: Sequence[int], name: Optional[str], question: Optional[str],
          answer: Optional[str], system: Optional[str]) -> Item:
    p, a = tuple(int(t) for t in prompt_ids), tuple(int(t) for t in answer_ids)
    if not a:
        raise Refused("An item needs an answer of at least one word-piece: its pieces are what is scored and what "
                      "the item is gated on.")
    if not p:
        raise Refused("An item needs at least one word-piece before its answer: the first scored piece is predicted "
                      "from it.")
    if len(p) + len(a) > MAX_PIECES:
        raise Refused(f"This item is {len(p) + len(a)} word-pieces; 1a reads at most {MAX_PIECES}.")
    return Item(name=name, question=question, answer=answer, system=system, prompt_ids=p, answer_ids=a,
                id=_sha({"v": 1, "prompt_ids": list(p), "answer_ids": list(a)}))


def make_item(tokenizer: Any, question: str, answer: str, *, system: Optional[str] = None,
              name: Optional[str] = None) -> Item:
    """An item from a question and an answer text. The prompt is the chat template's (one start mark, first); the
    answer is tokenised apart and appended, as pilots 7 and 8 built theirs, and is plain text: one that holds a start
    mark, a turn mark or another of the tokenizer's special pieces is refused, since those would be scored as text."""
    if not isinstance(question, str) or not isinstance(answer, str) or not (system is None or isinstance(system, str)):
        raise Refused("An item is a question and an answer, each a string, and a system turn that is a string or None.")
    prompt = chat_ids(tokenizer, question, system)
    bos = getattr(tokenizer, "bos_token_id", None)
    k = sum(1 for t in prompt if t == bos) if bos is not None else 0
    if k != 1 or prompt[0] != bos:
        raise Refused(f"The chat template gave {k} start marks; an item has exactly one, first.")
    answer_ids = [int(t) for t in tokenizer(answer, add_special_tokens=False)["input_ids"]]
    special = {int(t) for t in (getattr(tokenizer, "all_special_ids", None) or [])}
    marks = [t for t in answer_ids if t in special]
    if marks:
        raise Refused(f"The answer holds {len(marks)} piece{'' if len(marks) == 1 else 's'} of the chat format or the "
                      f"tokenizer's own marks ({tokenizer.decode(marks[:3])!r}): an answer is plain text, and its "
                      "pieces are what is scored.")
    return _item(prompt, answer_ids, name, question, answer, system)


def item_from_ids(prompt_ids: Sequence[int], answer_ids: Sequence[int], *, name: Optional[str] = None) -> Item:
    """An item from ids alone: at least one piece each side and MAX_PIECES in all. No start mark is asked for, so
    this also builds raw text as the pilots built it (the text's first piece alone as the prompt) and raw text behind
    the start mark (the start mark alone as the prompt)."""
    return _item(prompt_ids, answer_ids, name, None, None, None)


def _item_block(item: Item) -> Dict[str, Any]:
    return {"id": item.id, "name": item.name, "question": item.question, "answer": item.answer, "system": item.system,
            "prompt_ids": list(item.prompt_ids), "answer_ids": list(item.answer_ids),
            "pieces": len(item.prompt_ids) + len(item.answer_ids), "prompt_pieces": len(item.prompt_ids),
            "scored": len(item.answer_ids)}


# ---------------------------------------------------------------------------------------------------------- plans


def load_plan(path: Path) -> dict:
    """The plan file ({"v": 1, "name", "items": {name: {"question", "answer", "system"}}, "readings": [...]}), its
    top level checked: the keys, the items' shape, every reading a dict, no reading's name used twice, no reading
    asking for an item the file lacks. The readings themselves are `normalise`'s to check."""
    path = Path(path)
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused(f"The plan file {path} cannot be read ({exc.strerror or exc}).") from None
    except ValueError as exc:
        raise Refused(f"The plan file {path} is not JSON ({exc}).") from None
    except RecursionError:
        raise Refused(f"The plan file {path} is nested too deep to be a plan.") from None
    if not isinstance(plan, dict):
        raise Refused(f'The plan file {path} is not a JSON object with "v", "name", "items" and "readings".')
    for key in plan:
        if key not in ("v", "name", "items", "readings"):
            raise Refused(f"The plan has a key 1a does not know: {key}.")
    if plan.get("v") != 1 or isinstance(plan.get("v"), bool):
        raise Refused(f'The plan file {path} says "v": {plan.get("v")!r}; 1a reads plan files of "v": 1.')
    if not (plan.get("name") is None or isinstance(plan["name"], str)):
        raise Refused("The plan's name is a string saying what the sitting is for.")
    items = plan.get("items")
    if not isinstance(items, dict) or not items:
        raise Refused('The plan needs "items": at least one, each a name with its question and its answer.')
    for name, it in items.items():
        if not isinstance(it, dict):
            raise Refused(f'The item "{name}" is not an object with a question and an answer.')
        for key in it:
            if key not in ("question", "answer", "system"):
                raise Refused(f'The item "{name}" has a key 1a does not know: {key}.')
        if not isinstance(it.get("question"), str) or not isinstance(it.get("answer"), str):
            raise Refused(f'The item "{name}" needs a question and an answer, each a string.')
        if not (it.get("system") is None or isinstance(it["system"], str)):
            raise Refused(f'The system turn of the item "{name}" is a string or null.')
    readings = plan.get("readings")
    if not isinstance(readings, list) or not readings:
        raise Refused('The plan needs "readings": a list of at least one reading.')
    seen = set()
    for i, r in enumerate(readings):
        if not isinstance(r, dict):
            raise Refused(f"Reading {i} of the plan is not an object.")
        name = r.get("name")
        if isinstance(name, str):
            if name in seen:
                raise Refused(f'The name "{name}" is used by two readings of the plan: each reading has its own.')
            seen.add(name)
        asked = r.get("items", "all")
        if isinstance(asked, list):
            for x in asked:
                if not isinstance(x, str) or x not in items:
                    raise Refused(f'The reading "{name}" asks for the item {x!r}, which the plan file does not have.')
    return plan


def plan_items(plan: dict, tokenizer: Any) -> Dict[str, Item]:
    """The plan file's items, built, in the file's order. A refusal names the item before the item's own words."""
    out: Dict[str, Item] = {}
    for name, it in plan["items"].items():
        try:
            out[name] = make_item(tokenizer, it["question"], it["answer"], system=it.get("system"), name=name)
        except Refused as exc:
            raise Refused(f'Item "{name}": {exc}') from None
    return out


def _unknown_keys(d: Mapping[str, Any], known: Sequence[str]) -> None:
    for key in d:
        if key not in known:
            raise Refused(f"The plan has a key 1a does not know: {key}.")


def _doses(reading: Mapping[str, Any]) -> Tuple[List[float], float]:
    words = ("A dose is a finite number between -4 and 4: it multiplies the edit's difference from the base (1 the "
             "edit, 0 the base, -1 its mirror).")
    given = reading.get("doses", [1.0])
    if not isinstance(given, (list, tuple)):
        raise Refused(words)
    doses = []
    for d in given:
        x = _number(d)
        if x is None or abs(x) > we.MAX_DOSE:
            raise Refused(words)
        doses.append(x + 0.0)  # -0.0 is 0
    if not 1 <= len(doses) <= MAX_DOSES:
        raise Refused(f"A reading takes 1 to {MAX_DOSES} doses; {len(doses)} were asked.")
    for i, d in enumerate(doses):
        if d in doses[:i]:
            raise Refused(f"The dose {d:g} is listed twice: a reading takes each dose once.")
    if all(d == 0 for d in doses):
        raise Refused("A reading needs a dose other than 0: at 0 the edit is the base.")
    if "main_dose" in reading and reading["main_dose"] is not None:
        main = _number(reading["main_dose"])
        if main is None or main == 0 or main not in doses:
            raise Refused(f"main_dose is one of the doses and not 0; {reading['main_dose']!r} is not.")
    else:
        main = 1.0 if 1.0 in doses else next(d for d in doses if d != 0)
    return doses, float(main)


def _controls(reading: Mapping[str, Any], default_count: int) -> Dict[str, int]:
    given = reading.get("controls")
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise Refused('controls is an object: {"count": how many control edits, "seed": the seed they are drawn from}.')
    _unknown_keys(given, ("count", "seed", "family"))
    if given.get("family", "other_sites") != "other_sites":
        raise Refused(f'Control edits are of one family in 1a, "other_sites" (other sites of the same layer and size); '
                      f"{given['family']!r} is not built.")
    count, seed = _whole_number(given.get("count", default_count)), _whole_number(given.get("seed", 0))
    if count is None or not 0 <= count <= MAX_CONTROLS:
        raise Refused(f"A reading takes 0 to {MAX_CONTROLS} control edits; {given.get('count')!r} were asked.")
    if seed is None or seed < 0:
        raise Refused(f"The control edits' seed is a whole number from 0 up; {given.get('seed')!r} was given.")
    return {"count": count, "seed": seed}


def _null_e(reading: Mapping[str, Any], default: Sequence[float], dtype: Any) -> List[float]:
    given = reading.get("null_e", list(default))
    if not isinstance(given, (list, tuple)):
        given = [given]
    for e in given:
        if isinstance(e, bool) or not isinstance(e, (int, float)):
            raise Refused(f"A null row's e is a number; {e!r} is not.")
    es = [float(e) for e in given]
    if len(es) < 2 or not any(e > 0 for e in es) or not any(e < 0 for e in es):
        raise Refused("A reading takes at least two null rows, one with e above 0 and one below: one null row is one "
                      "draw of the rounding.")
    if len(es) > MAX_NULL_ROWS:
        raise Refused(f"A reading takes at most {MAX_NULL_ROWS} null rows; {len(es)} were asked.")
    for e in es:
        if not math.isfinite(e) or not -0.5 <= e <= 2 or e == 0:
            raise Refused("A null row scales the layer's feed-forward output by 1 + e, with e between -0.5 and 2 and "
                          f"not 0; {e:g} was asked.")
    word = _dtype_word(dtype)
    factors = [float(we.null_factor(e, dtype)) for e in es]
    for e, s in zip(es, factors):
        if s == 1:
            raise Refused(f"In {word} 1 + e is exactly 1 for e = {e:g}: that null row would be the base row again. "
                          "In bf16 the nearest values are 1 + 1/128 and 1 - 1/256.")
    for e, s in zip(es, factors):
        if _exact(s):
            raise Refused(f"For e = {e:g} the null row's factor is {s:g}, a power of two: an exact scaling, which the "
                          "norm may undo exactly, so it measures no rounding. Factors that serve: 1 + 1/128, "
                          "1 - 1/256, 0.75, 1.5, 3.")
    for i, s in enumerate(factors):
        for j in range(i):
            if factors[j] == s:
                raise Refused(f"e = {es[j]:g} and e = {es[i]:g} are one factor in {word} ({s:g}): the two null rows "
                              "would be the same row.")
    return es


def _pieces(given: Any, side: str, tokenizer: Any) -> List[int]:
    if not isinstance(given, (list, tuple)) or not 1 <= len(given) <= MAX_NAMED_PIECES:
        raise Refused(f'odds names 1 to {MAX_NAMED_PIECES} pieces each side; "{side}" is {given!r}.')
    out = []
    for w in given:
        if isinstance(w, str):
            if tokenizer is None:
                raise Refused(f"The plan names the piece {w!r} as a word, and no tokenizer was given to look it up: "
                              "give its id.")
            ids = [int(t) for t in tokenizer(w, add_special_tokens=False)["input_ids"]]
            if len(ids) != 1:
                raise Refused(f"{w!r} is {len(ids)} word-pieces ({ids}); a named piece is one. Give its id, or a "
                              "word that is one piece.")
            out.append(ids[0])
        else:
            t = _whole_number(w)
            if t is None or t < 0 or (tokenizer is not None and t >= len(tokenizer)):
                raise Refused(f"A named piece is a word that is one piece, or a piece's id; {w!r} is neither.")
            out.append(t)
    return out


def _readouts(reading: Mapping[str, Any], tokenizer: Any) -> Tuple[Dict[str, Any], str]:
    given = reading.get("readouts")
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise Refused('readouts is an object: {"kl": true or false, "own": true or false, "odds": null or the pieces}.')
    _unknown_keys(given, ("kl", "own", "odds"))
    odds = given.get("odds")
    if odds is not None:
        if not isinstance(odds, dict):
            raise Refused('odds is null, or {"toward": [pieces], "away": [pieces], "at": "end" or "span"}.')
        _unknown_keys(odds, ("toward", "away", "at"))
        at = odds.get("at", "end")
        if at not in ("end", "span"):
            raise Refused(f'odds is read at "end" (the position after the text) or over the "span" (the answer\'s '
                          f"pieces); {at!r} is neither.")
        odds = {"toward": _pieces(odds.get("toward"), "toward", tokenizer),
                "away": _pieces(odds.get("away"), "away", tokenizer), "at": at}
    out = {"kl": given.get("kl", odds is None), "own": given.get("own", odds is None), "odds": odds}
    for m in ("kl", "own"):
        if not isinstance(out[m], bool):
            raise Refused(f"readouts.{m} is true or false; {out[m]!r} is neither.")
    on = [m for m in METRICS if out[m]]
    if not on:
        raise Refused("A reading reads something: turn on kl, own or odds.")
    rank_by = reading.get("rank_by")
    if rank_by is None:
        rank_by = on[0]
    if rank_by not in on:
        raise Refused(f"rank_by names a read-out that is on; {rank_by} is not.")
    return out, rank_by


def _bars(reading: Mapping[str, Any]) -> Dict[str, Any]:
    given = reading.get("bars")
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise Refused('bars is an object with any of "item_gate", "stable_ratio" and "one_row_min_layer".')
    _unknown_keys(given, ("item_gate", "stable_ratio", "one_row_min_layer"))
    gate, ratio = _number(given.get("item_gate", ITEM_GATE_BAR)), _number(given.get("stable_ratio", STABLE_RATIO))
    low = _whole_number(given.get("one_row_min_layer", ONE_ROW_MIN_LAYER))
    if gate is None:
        raise Refused(f"bars.item_gate is a number of nats a piece; {given.get('item_gate')!r} is not.")
    if ratio is None or ratio < 1:
        raise Refused(f"bars.stable_ratio is a ratio of 1 or more; {given.get('stable_ratio')!r} is not.")
    if low is None or low < 0:
        raise Refused(f"bars.one_row_min_layer is a layer, counted from 0; {given.get('one_row_min_layer')!r} is not.")
    return {"item_gate": gate, "stable_ratio": ratio, "one_row_min_layer": low}


def _fact_refusals(plan: Mapping[str, Any], facts: Mapping[str, Any], n_layers: int) -> None:
    """The refusals of a plan that need to know the edit: a layer the model lacks, a one-row capture below its bar, a
    folded reading of an edit with no folded form."""
    for layer in facts.get("layers", []):
        if not 0 <= int(layer) < n_layers:
            raise Refused(f"Layer {int(layer)} does not exist: layers count from 0 to {n_layers - 1} here (page "
                          f"layers 1 to {n_layers}).")
    if plan.get("kind") != "paired":
        return
    m, low = plan["bars"]["one_row_min_layer"], int(facts["lowest_layer"])
    if plan["lower"] == "one_row" and low < m:
        raise Refused("A one-row capture replayed at many rows has been shown to agree with the same-shape reading "
                      "only at layer 44 (page layer 45), on raw text with no start mark, on 14 November 2026; on two "
                      "chat-formatted items on 16 November 2026 it stood off that reading at layers 40 and 44 by 10 to "
                      f"35 times the floor in 7 of 8 readings. It is taken from layer {m} (page layer {m + 1}) up, as "
                      "a cheaper reading that is not the same-shape one. This edit starts at layer "
                      f"{low} (page layer {low + 1}).")
    if plan["folded"] == "always" and facts.get("has_lowrank"):
        raise Refused("A low-rank term has no folded form yet, so this edit cannot be read folded.")


def normalise(reading: dict, *, n_layers: int, tokenizer: Any = None, facts: Optional[dict] = None,
              dtype: Any = torch.bfloat16) -> dict:
    """The plan of one reading with its defaults filled, or `Refused` in plain words. Pure: no model, no file. The
    dict returned is what is run, recorded and hashed, and is taken again unchanged. `tokenizer` resolves named pieces
    to ids (a plan that names pieces as words needs it); `facts` (`we.recipe_facts` or `we.edit_facts`) lets it refuse
    a layer the model lacks, `lower: "one_row"` below its bar and `folded: "always"` for an edit with no folded form;
    `dtype` is the model's, for the null rows' factors. The module's bars are read when this is called."""
    if not isinstance(reading, dict):
        raise Refused("A reading's plan is an object with a name, a kind and an edit.")
    kind = reading.get("kind")
    if kind not in ("paired", "written"):
        raise Refused(f'A reading is "paired" or "written"; its kind was given as {kind!r}.')
    _unknown_keys(reading, _PAIRED_KEYS if kind == "paired" else _WRITTEN_KEYS)
    if not isinstance(reading.get("name"), str) or not reading["name"]:
        raise Refused("A reading needs a name: a string, its own in its plan file.")
    edit = reading.get("edit")
    if not isinstance(edit, dict) or not edit:
        raise Refused('A reading needs an edit: {"terms": [...]} or {"saved": "<id or folder>"}.')
    items = reading.get("items", "all")
    if items != "all":
        if (not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS or len(set(map(str, items))) != len(items)
                or not all(isinstance(x, str) for x in items)):
            raise Refused(f'items is "all", or a list of 1 to {MAX_ITEMS} item names of the plan file, each once.')
        items = list(items)
    doses, main = _doses(reading)
    out: Dict[str, Any] = {"name": reading["name"], "kind": kind, "edit": _clean(edit), "items": items, "doses": doses,
                           "main_dose": main}
    if kind == "written":
        out["controls"] = _controls(reading, WRITTEN_CONTROLS)
        out["null_e"] = _null_e(reading, NULL_E_WRITTEN, dtype)
        rows = _whole_number(reading.get("rows", WRITTEN_ROWS))
        if rows is None or not WRITTEN_ROWS_MIN <= rows <= WRITTEN_ROWS_MAX:
            raise Refused(f"A written reading is one batch of {WRITTEN_ROWS_MIN} to {WRITTEN_ROWS_MAX} rows; "
                          f"{reading.get('rows')} were asked.")
        max_new = _whole_number(reading.get("max_new", WRITTEN_MAX_NEW))
        if max_new is None or not 1 <= max_new <= MAX_NEW_MAX:
            raise Refused(f"A written reading writes 1 to {MAX_NEW_MAX} new word-pieces; {reading.get('max_new')!r} "
                          "were asked.")
        out["rows"], out["max_new"] = rows, max_new
        layout(doses, out["controls"]["count"], out["null_e"], rows, one_forward=True)
    else:
        out["controls"] = _controls(reading, 19)
        out["null_e"] = _null_e(reading, NULL_E, dtype)
        rows = reading.get("rows")
        if rows is None:
            rows = {}
        if not isinstance(rows, dict):
            raise Refused('rows is an object: {"small": 5 to 12, "large": 16 to 64}.')
        _unknown_keys(rows, ("small", "large"))
        small, large = _whole_number(rows.get("small", ROWS_SMALL)), _whole_number(rows.get("large", ROWS_LARGE))
        if small is None or large is None or not SMALL_MIN <= small <= SMALL_MAX or not LARGE_MIN <= large <= LARGE_MAX:
            raise Refused(f"The small batch is {SMALL_MIN} to {SMALL_MAX} rows and the large one {LARGE_MIN} to "
                          f"{LARGE_MAX} (the ruling's two classes: at 150 pieces of raw text with no start mark "
                          "this card ran one arithmetic up to 12 rows and another from 16, 14 November 2026); "
                          f"{rows.get('small', ROWS_SMALL)} and {rows.get('large', ROWS_LARGE)} were asked.")
        out["rows"] = {"small": small, "large": large}
        for r in (small, large):
            layout(doses, out["controls"]["count"], out["null_e"], r)
        out["readouts"], out["rank_by"] = _readouts(reading, tokenizer)
        out["folded"] = reading.get("folded", "auto")
        if out["folded"] not in ("auto", "always", "never"):
            raise Refused(f'folded is "auto", "always" or "never"; {out["folded"]!r} is none of them.')
        out["lower"] = reading.get("lower", "same_shape")
        if out["lower"] not in ("same_shape", "whole", "one_row"):
            raise Refused('lower says how the layers below the edit are run: "same_shape", "whole" or "one_row"; '
                          f'{out["lower"]!r} is none of them.')
        out["bars"] = _bars(reading)
    if facts is not None:
        _fact_refusals(out, facts, int(n_layers))
    return out


def _bars_overridden(plan: Optional[Mapping[str, Any]]) -> bool:
    """Whether the plan runs under a bar other than the module's own, as the module stands now."""
    bars = (plan or {}).get("bars") if isinstance(plan, Mapping) else None
    if not isinstance(bars, Mapping):
        return False
    own = {"item_gate": ITEM_GATE_BAR, "stable_ratio": STABLE_RATIO, "one_row_min_layer": ONE_ROW_MIN_LAYER}
    return any(bars.get(k) != v for k, v in own.items())


def layout(doses: Sequence[float], n_controls: int, null_e: Sequence[float], rows: int, *,
           one_forward: bool = False) -> List[List[Optional[str]]]:
    """The arm key of every row of every forward of one class, None for an untouched row. Row 0 is the base row and
    the last row its twin, both untouched; rows 1 to n0 are the null rows, the same in every forward; the arms (the
    edit at each dose, then control edits 0 on) fill the slots between, forward after forward; rows left over in the
    last forward are untouched. Every dose sits in forward 0. `one_forward` is a written reading: every arm in one
    batch. Refuses, in the plan's words, doses or arms that do not fit."""
    R, n0, nd, C = int(rows), len(null_e), len(doses), int(n_controls)
    s = R - 2 - n0
    if s < 0:
        raise Refused(f"A batch of {R} rows cannot hold the base row, its twin and {n0} null rows. Ask fewer null "
                      "rows, or more rows.")
    if one_forward and nd + C > s:
        raise Refused(f"A written reading is one batch: {R} rows hold the base row, its twin, {n0} null rows and {s} "
                      f"more, too few for {nd} doses and {C} control edits. Ask fewer, or more rows.")
    if nd > s:
        raise Refused(f"A batch of {R} rows holds the base row, its twin, {n0} null rows and {s} more: too few for "
                      f"{nd} doses. Ask fewer doses or null rows, or more rows.")
    nulls = [_key("null", e) for e in null_e]
    arms = [_key("edit", d) for d in doses] + [_key("control", c) for c in range(C)]
    out = []
    for f in range(max(1, math.ceil(len(arms) / s)) if s else 1):
        here = arms[f * s:(f + 1) * s]
        out.append([None] + nulls + here + [None] * (s - len(here)) + [None])
    return out


def estimate(plan: dict, pieces: Sequence[Tuple[int, int]], *, n_layers: int, facts: dict,
             twin: Optional[bool] = None) -> dict:
    """What a reading costs, from the measured figures of `COST` and no model. `plan` is normalised; `pieces` holds
    one (word-pieces, prompt pieces) an item; `facts` is `we.recipe_facts` or `we.edit_facts`. `twin`: whether the
    folded twin is taken; None decides it from the plan's `folded` and `we.twin_wanted` at the main dose. Returns
    {"seconds", "parts": {"forwards", "readout", "folded", "writing"}, "forwards", "row_positions",
    "gib_above_model"}: `forwards` counts every teacher-forced forward (the twin's and a one-row capture's among
    them; a written reading's is its prompt's, one an item), `row_positions` the rows times pieces over all of them
    (a written reading: rows times its context). It assumes every item is read in both classes. Raises `Refused` for
    a forward over MAX_ROW_POSITIONS and for a written reading whose cache would pass MAX_WRITTEN_GIB."""
    parts = {"forwards": 0.0, "readout": 0.0, "folded": 0.0, "writing": 0.0}
    forwards, row_positions, gib = 0, 0, 0.0
    if plan["kind"] == "written":
        R, max_new = int(plan["rows"]), int(plan["max_new"])
        for _T, prompt in pieces:
            n = int(prompt) + max_new
            g = COST["mb_row_piece_written"] * 1e6 * R * n / 2 ** 30
            if g > MAX_WRITTEN_GIB:
                raise Refused(f"Writing {R} rows to {n} word-pieces of context keeps about {g:.1f} GiB above the "
                              "model (0.8 MB a row a piece, measured 28 October 2026); 1a takes at most "
                              f"{MAX_WRITTEN_GIB:g}. Ask fewer rows or fewer new pieces.")
            parts["writing"] += whole(R, int(prompt)) + COST["decode_step_s"] * max_new
            forwards, row_positions, gib = forwards + 1, row_positions + R * n, max(gib, g)
    else:
        if twin is None:
            twin = plan["folded"] == "always" or (plan["folded"] == "auto"
                                                   and bool(we.twin_wanted(facts, plan["main_dose"])))
        share = (int(n_layers) - int(facts["lowest_layer"])) / int(n_layers)
        n0, nd, C, lower = len(plan["null_e"]), len(plan["doses"]), plan["controls"]["count"], plan["lower"]
        full = plan["readouts"]["kl"] or plan["readouts"]["own"]
        for T, prompt in pieces:
            T, S = int(T), int(T) - int(prompt)
            n = plan["rows"]["large"] * T
            if n > MAX_ROW_POSITIONS:
                raise Refused(f"{plan['rows']['large']} rows of {T} word-pieces is {n:,} row-positions; 1a runs at "
                              f"most {MAX_ROW_POSITIONS:,} in a forward (the pilots measured to 11,264).")
            if lower == "one_row":
                parts["forwards"] += whole(1, T)
                forwards, row_positions = forwards + 1, row_positions + T
            for R in (plan["rows"]["small"], plan["rows"]["large"]):
                Fc, w = len(layout(plan["doses"], C, plan["null_e"], R)), whole(R, T)
                parts["forwards"] += {"same_shape": w + (Fc - 1) * w * share, "whole": Fc * w,
                                      "one_row": Fc * w * share}[lower]
                forwards, row_positions = forwards + Fc, row_positions + Fc * R * T
                if twin:
                    restarts = lower == "one_row" or (lower == "same_shape" and Fc >= 2)
                    parts["folded"] += (w * share if restarts else w) + COST["overlay_matrix_s"] * int(facts["linears"])
                    forwards, row_positions = forwards + 1, row_positions + R * T
                parts["readout"] += COST["readout_row_position_s"] * S * (
                    (Fc + n0 + nd + C + (1 if twin else 0)) if full else 1)
            gib = max(gib, COST["mib_row_position"] * n / 1024 + 0.5)
    return {"seconds": sum(parts.values()), "parts": parts, "forwards": forwards, "row_positions": row_positions,
            "gib_above_model": gib}


# ----------------------------------------------------------------------------------------------------- the handle


@dataclass
class Handle:
    """A loaded runtime as a reading needs it, taken once (`handle`): the text model, the unembedding, the model's
    dtype and shape, what the weights are, and the census of hooks at load that `we.assert_base` compares with."""

    rt: Any
    hf: Any
    lm: Any
    tokenizer: Any
    W_U: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    softcap: Optional[float]
    n_layers: int
    d_model: int
    d_ff: int
    model_id: str
    weights: Dict[str, Any]
    weights_id: Optional[str]
    hooks_at_load: Dict[str, List[int]]
    runtime: Dict[str, Any]


def _overlay_words(lm: Any) -> Optional[str]:
    over = we.overlays_in_force(lm)
    if not over:
        return None
    name, k = over[0]
    return (f"A vault edit is in force on {name} ({k} overlays): a reading is taken on the base model only. ROLLBACK "
            "the vault, or start a fresh process.")


def handle(rt: Any) -> Handle:
    """The handle of a loaded runtime. Raises `we.NotBase` when a vault overlay is in force: the census of hooks
    taken here is the base's, and a reading is taken on the base model only."""
    hf = rt.hf_model
    lm = language_model(hf)
    words = _overlay_words(lm)
    if words:
        raise we.NotBase(words)
    device = model_device(lm)
    weights = weights_card(rt)
    return Handle(rt=rt, hf=hf, lm=lm, tokenizer=rt.tokenizer, W_U=rt.lm_head_weight, device=device,
                  dtype=lm.norm.weight.dtype, softcap=ao.softcap_of(hf), n_layers=len(lm.layers),
                  d_model=ao.hidden_size(lm), d_ff=ao.ffn_width(lm, 0), model_id=rt.model_id, weights=weights,
                  weights_id=weights.get("files_sha256"), hooks_at_load=we.hook_census(lm),
                  runtime=runtime_record(device))


# ------------------------------------------------------------------------------- the figures: stats, agree, ranks


def stats(values: torch.Tensor) -> dict:
    """A metric's per-position values as {"mean", "median", "max_abs", "above_1", "n"}: the median is torch's (the
    lower middle of an even count, as the pilots took it), `max_abs` the largest absolute value, `above_1` how many
    positions read over 1 nat either way. A number that is not finite is given as None."""
    v = values.detach().reshape(-1)
    if not v.is_floating_point():
        v = v.float()
    n = int(v.numel())
    if n == 0:
        return {"mean": None, "median": None, "max_abs": None, "above_1": 0, "n": 0}
    got = {"mean": float(v.mean()), "median": float(v.median()), "max_abs": float(v.abs().max())}
    return {**{k: (x if math.isfinite(x) else None) for k, x in got.items()}, "above_1": int((v.abs() > 1).sum()), "n": n}


def agree(a: Optional[float], b: Optional[float], floor: float, signed: bool,
          ratio: float = STABLE_RATIO) -> Tuple[Optional[bool], Optional[float]]:
    """Whether one arm's two readings, `a` at the small row count and `b` at the large, agree: (agree, their ratio).
    The smaller is lifted to the floor before the ratio is taken, so two readings at the floor agree; a `signed`
    read-out also disagrees when the two have opposite signs and the larger stands clear of the floor. None in both
    is not compared (None, None); None in one disagrees. Provisional, as its number is."""
    a = None if a is None or not math.isfinite(a) else float(a)
    b = None if b is None or not math.isfinite(b) else float(b)
    if a is None and b is None:
        return None, None
    if a is None or b is None:
        return False, None
    hi, lo = max(abs(a), abs(b)), max(min(abs(a), abs(b)), float(floor))
    r = 1.0 if hi == 0 else (math.inf if lo == 0 else hi / lo)
    ok = r <= ratio and (not signed or a * b >= 0 or hi <= ratio * floor)
    return bool(ok), r


def rank_among(value: Optional[float], controls: Sequence[Optional[float]]) -> dict:
    """Where `value` stands among the control edits' values: `at_least` counts the controls that read as much or
    more, a control that is not finite (None) among them; `non_finite` says how many those were; `of` is the controls
    and the edit; `rank` is `at_least` + 1 (1: no control read as much). A value of None has no rank."""
    fine = [float(c) for c in controls if c is not None and math.isfinite(c)]
    bad = len(controls) - len(fine)
    if value is None or not math.isfinite(value):
        return {"at_least": None, "non_finite": bad, "of": len(controls) + 1, "rank": None}
    at_least = bad + sum(1 for c in fine if c >= value)
    return {"at_least": at_least, "non_finite": bad, "of": len(controls) + 1, "rank": at_least + 1}


# ------------------------------------------------------------------------------------------- forwards and read-outs


class _Void(Exception):
    """The reading's mechanism was not sound: it stops at once and is written with status "void"."""

    def __init__(self, code: str, why: str) -> None:
        super().__init__(why)
        self.code, self.why = code, why


@contextmanager
def _capture(layer: Any, box: Dict[str, torch.Tensor]) -> Iterator[None]:
    """Keep the stream entering `layer` for the length of the block: [rows, pieces, D] on the model's device."""

    def pre(_mod: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        box["stream"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

    hook = layer.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()


def _whole(h: Handle, ids: Sequence[int], rows: int) -> torch.Tensor:
    """One whole forward of the item's ids repeated to `rows` rows: no attention mask and no position ids, as every
    pilot ran it. Final-normed states [rows, pieces, D] in the model's dtype."""
    x = torch.tensor([list(ids)] * int(rows), dtype=torch.long, device=h.device)
    return h.lm(input_ids=x, use_cache=False).last_hidden_state


def _restart(h: Handle, layer: int, kept: torch.Tensor) -> torch.Tensor:
    """The forward from `layer` on, on a stream kept at this batch shape."""
    R, T = int(kept.shape[0]), int(kept.shape[1])
    pos = torch.arange(T, device=kept.device).unsqueeze(0).expand(R, T)
    return ao.forward_from_layer(h.lm, int(layer), kept, None, pos)


def _logp(h: Handle, states: torch.Tensor) -> torch.Tensor:
    """fp32 log-probabilities over the vocabulary from final-normed states [n, D]: softcapped fp32 logits."""
    return F.log_softmax(logits_fp32(states, h.W_U), dim=-1)


@dataclass
class _Span:
    """Where one item is read: positions `lo` to `T` - 2 predict the answer's pieces `y`; `T` - 1 is the end."""

    T: int
    lo: int
    y: torch.Tensor

    @classmethod
    def of(cls, item: Item, device: Any) -> "_Span":
        ids = item.ids
        lo = len(item.prompt_ids) - 1
        return cls(T=len(ids), lo=lo, y=torch.tensor(list(ids[lo + 1:]), dtype=torch.long, device=device))


def _measure(h: Handle, on: Mapping[str, Any], sp: _Span, base: torch.Tensor,
             rows: Sequence[Optional[torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], List[Optional[Dict[str, torch.Tensor]]]]:
    """The read-outs that are on, a position: each of `rows` ([pieces, D] final states, or None for a row not read)
    against `base`. Returns the base row's own values ({"own": its log-probability of each answer piece, "odds": its
    log-odds}) and, a row, {metric: values}. The full vocabulary is worked `READ_CHUNK` positions at a time and a row
    at a time, reduced on the device: about four [READ_CHUNK, vocabulary] float32 tables exist at the peak."""
    out: List[Optional[Dict[str, List[torch.Tensor]]]] = [None if r is None else {} for r in rows]
    raw: Dict[str, List[torch.Tensor]] = {}
    if on.get("kl") or on.get("own"):
        for a in range(sp.lo, sp.T - 1, READ_CHUNK):
            b = min(a + READ_CHUNK, sp.T - 1)
            y = sp.y[a - sp.lo:b - sp.lo].unsqueeze(1)
            L0 = _logp(h, base[a:b])
            own0 = L0.gather(1, y).squeeze(1)
            raw.setdefault("own", []).append(own0)
            for i, r in enumerate(rows):
                if r is None:
                    continue
                Lr = _logp(h, r[a:b])
                if on.get("own"):
                    out[i].setdefault("own", []).append(Lr.gather(1, y).squeeze(1) - own0)
                if on.get("kl"):
                    Lr.neg_().add_(L0).mul_(L0.exp())  # exp(L0) * (L0 - Lr), in the row's own table
                    out[i].setdefault("kl", []).append(Lr.sum(-1).clamp_min_(0.0))
                del Lr
            del L0
    odds = on.get("odds")
    if odds:
        dev = base.device
        Wt = h.W_U[torch.as_tensor(odds["toward"], dtype=torch.long, device=h.W_U.device)].to(dev).float()
        Wa = h.W_U[torch.as_tensor(odds["away"], dtype=torch.long, device=h.W_U.device)].to(dev).float()
        q = slice(sp.T - 1, sp.T) if odds["at"] == "end" else slice(sp.lo, sp.T - 1)

        def log_odds(row: torch.Tensor) -> torch.Tensor:
            x = row[q].float()
            zt = torch.tanh(x @ Wt.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
            za = torch.tanh(x @ Wa.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
            return torch.logsumexp(zt, -1) - torch.logsumexp(za, -1)

        lo0 = log_odds(base)
        raw["odds"] = [lo0]
        for i, r in enumerate(rows):
            if r is not None:
                out[i]["odds"] = [log_odds(r) - lo0]
    return ({m: torch.cat(v).cpu() for m, v in raw.items()},
            [None if o is None else {m: torch.cat(v).cpu() for m, v in o.items()} for o in out])


def _gate_figures(own: torch.Tensor) -> Dict[str, Optional[float]]:
    s = stats(own)
    worst = float(own.min()) if own.numel() else None
    return {"mean": s["mean"], "median": s["median"], "worst": worst if worst is None or math.isfinite(worst) else None}


@dataclass
class _Sitting:
    """One reading as it runs: the handle, the normalised plan, the edit and its control edits, each arm key's facts
    and program, and where it stands (for a void's words)."""

    h: Handle
    plan: Dict[str, Any]
    edit: Any
    controls: List[Any]
    L0: int
    arms: Dict[str, Dict[str, Any]]
    twin: bool
    twin_why: Optional[str]
    where: Dict[str, Any]
    peaks: List[float]

    def reset_peak(self) -> None:
        if self.h.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.h.device)

    def peak(self) -> Optional[float]:
        if self.h.device.type != "cuda":
            return None
        g = round(torch.cuda.max_memory_allocated(self.h.device) / 2 ** 30, 3)
        self.peaks.append(g)
        return g

    def programs(self, keys: Sequence[Optional[str]]) -> List[Any]:
        return [None if k is None else self.arms[k]["program"] for k in keys]


def _arm_facts(plan: Mapping[str, Any], edit: Any, controls: Sequence[Any], L0: int, dtype: Any) -> Dict[str, Dict[str, Any]]:
    """Each arm key's facts as the result records them, and the program its row runs."""
    out: Dict[str, Dict[str, Any]] = {}
    for e in plan["null_e"]:
        out[_key("null", e)] = {"what": "null", "e": float(e), "factor": float(we.null_factor(e, dtype)),
                                "edit_id": None, "program": we.Null(int(L0), float(e))}
    for d in plan["doses"]:
        out[_key("edit", d)] = {"what": "edit", "dose": float(d), "factor": None, "edit_id": edit.id,
                                "program": we.Arm(edit, float(d))}
    for c, ctl in enumerate(controls):
        out[_key("control", c)] = {"what": "control", "dose": float(plan["main_dose"]), "factor": None,
                                   "edit_id": ctl.id, "program": we.Arm(ctl, float(plan["main_dose"]))}
    return out


def _arm_entry(key: str, facts: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"key": key, "what": facts["what"]}
    out["e" if facts["what"] == "null" else "dose"] = facts["e"] if facts["what"] == "null" else facts["dose"]
    out["factor"], out["edit_id"] = facts["factor"], facts["edit_id"]
    return out


def _check_forward(sit: _Sitting, Hf: torch.Tensor, keys: Sequence[Optional[str]], f: int,
                   first: Optional[torch.Tensor]) -> None:
    """The four rules a forward must keep, in order; the first that fails voids the reading."""
    w = sit.where
    at = f"item {w['item']}, {w['rows']} rows, forward {f}"
    R, row0 = Hf.shape[0], Hf[0]
    if not bool(torch.isfinite(row0).all()):
        raise _Void("non_finite_base", f"The base row holds a number that is not finite ({at}): nothing can be "
                       "read against it.")
    for r in [R - 1] + [r for r in range(1, R - 1) if keys[r] is None]:
        if not torch.equal(Hf[r], row0):
            raise _Void("untouched_rows_differ", f"Row {r}, which nothing touched, is not the base row by bits "
                           f"({at}): the rows of this forward are not independent, so no row of it is read.")
    for r, k in enumerate(keys):
        if k is not None and sit.arms[k]["what"] == "edit" and sit.arms[k]["dose"] == 0 and not torch.equal(Hf[r], row0):
            raise _Void("zero_dose_differs", f"The edit at dose 0 (row {r}) is not the base row by bits ({at}): "
                           "the per-row form changes a row it should leave alone.")
    if first is not None and not torch.equal(Hf[:first.shape[0]], first):
        raise _Void("repeat_differs", f"The base row or a null row of forward {f} is not forward 0's by bits "
                       f"(item {w['item']}, {w['rows']} rows): the forwards of this class are not the same forward, so "
                       "their arms cannot be set side by side.")


@dataclass
class _Begun:
    """A class of one item after its forward 0: the states, the stream kept for later forwards, the gate's figure."""

    cls: str
    R: int
    keys: List[List[Optional[str]]]
    Hf0: torch.Tensor
    kept: Optional[torch.Tensor]
    base_raw: Dict[str, torch.Tensor]
    gate: Dict[str, Optional[float]]


def _begin_class(sit: _Sitting, item: Item, sp: _Span, cls: str, one_row: Dict[str, torch.Tensor]) -> _Begun:
    """Forward 0 of a class under its programs, its four rules, and the gate's figure from its base row."""
    h, plan = sit.h, sit.plan
    R = int(plan["rows"][cls])
    sit.where.update({"class": cls, "rows": R, "forward": 0})
    sit.reset_peak()
    keys = layout(plan["doses"], plan["controls"]["count"], plan["null_e"], R)
    lower, kept = plan["lower"], None
    if lower == "one_row":
        if "stream" not in one_row:  # once an item: the layers below the edit at ONE row, with only the capture hook
            with _capture(h.lm.layers[sit.L0], one_row):
                _whole(h, item.ids, 1)
        kept = one_row["stream"].expand(R, sp.T, one_row["stream"].shape[-1]).contiguous()
        with we.per_row(h.lm, sit.programs(keys[0])):
            Hf0 = _restart(h, sit.L0, kept)
    elif lower == "same_shape" and len(keys) >= 2:
        box: Dict[str, torch.Tensor] = {}
        with we.per_row(h.lm, sit.programs(keys[0])):
            with _capture(h.lm.layers[sit.L0], box):
                Hf0 = _whole(h, item.ids, R)
        kept = box["stream"]
    else:
        with we.per_row(h.lm, sit.programs(keys[0])):
            Hf0 = _whole(h, item.ids, R)
    _check_forward(sit, Hf0, keys[0], 0, None)
    base_raw, _ = _measure(h, {"own": True, "odds": plan["readouts"]["odds"]}, sp, Hf0[0], [])
    return _Begun(cls=cls, R=R, keys=keys, Hf0=Hf0, kept=kept, base_raw=base_raw, gate=_gate_figures(base_raw["own"]))


def _read_rows(sit: _Sitting, sp: _Span, Hf: torch.Tensor, keys: Sequence[Optional[str]], f: int,
               skip_null: bool) -> List[Dict[str, Any]]:
    """Every null row and arm of one forward against that forward's own row 0."""
    on = sit.plan["readouts"]
    rows, entries = [], []
    for r, k in enumerate(keys):
        if k is None or (skip_null and sit.arms[k]["what"] == "null"):
            continue
        entry = _arm_entry(k, sit.arms[k])
        entry.update({"forward": f, "row": r, "non_finite": not bool(torch.isfinite(Hf[r]).all()),
                      "same_as_base": bool(torch.equal(Hf[r], Hf[0])), "metrics": None})
        entries.append(entry)
        rows.append(None if entry["non_finite"] else Hf[r])
    _, got = _measure(sit.h, on, sp, Hf[0], rows)
    for entry, g in zip(entries, got):
        if g is not None:
            entry["metrics"] = {m: stats(g[m]) for m in METRICS if on[m]}
    return entries


def _floor(arms: Sequence[Mapping[str, Any]], on: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """A metric's floor on one item in one class: the largest absolute mean among the null rows, with the null rows'
    largest absolute median and count above a nat; `from` names the row and `of` says how many it is the largest of.
    A null row that came out the base row by bits, or not finite, is no draw of the rounding and is not counted: `of`
    is the null rows that gave a figure, and with none the floor is None (no null row gave a figure)."""
    out: Dict[str, Dict[str, Any]] = {}
    for m in METRICS:
        if not on[m]:
            continue
        got = [(abs(a["metrics"][m]["mean"]), a) for a in arms
               if a["what"] == "null" and not a.get("same_as_base") and a["metrics"]
               and a["metrics"][m]["mean"] is not None]
        if not got:
            out[m] = {"mean_abs": None, "median_abs": None, "above_1": None, "from": None, "of": 0}
            continue
        top = max(got, key=lambda g: g[0])
        medians = [abs(a["metrics"][m]["median"]) for _x, a in got if a["metrics"][m]["median"] is not None]
        out[m] = {"mean_abs": top[0], "median_abs": max(medians) if medians else None,
                  "above_1": max(a["metrics"][m]["above_1"] for _x, a in got), "from": top[1]["key"], "of": len(got)}
    return out


def _abs_mean(arm: Optional[Mapping[str, Any]], metric: str) -> Optional[float]:
    if not arm or not arm.get("metrics") or arm["metrics"].get(metric) is None:
        return None
    x = arm["metrics"][metric]["mean"]
    return None if x is None else abs(x)


def _rank(plan: Mapping[str, Any], arms: Sequence[Mapping[str, Any]], floor: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    C = plan["controls"]["count"]
    if not C:
        return None
    m = plan["rank_by"]
    by_key = {a["key"]: a for a in arms}
    value = _abs_mean(by_key.get(_key("edit", plan["main_dose"])), m)
    ctl = [_abs_mean(by_key.get(_key("control", c)), m) for c in range(C)]
    fl = (floor.get(m) or {}).get("mean_abs")
    return {"metric": m, "dose": plan["main_dose"], "value": value, **rank_among(value, ctl),
            "over_floor": value / fl if value is not None and fl else None,
            "controls_over_floor": sum(1 for c in ctl if c is not None and c > (fl or 0.0))}  # with no floor: above 0


def _sync(h: Handle) -> None:
    if h.device.type == "cuda":
        torch.cuda.synchronize(h.device)


def _twin(sit: _Sitting, item: Item, sp: _Span, st: _Begun, arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The folded twin of one class: the same edit written into the overlays, one forward of the class's rows, its
    row 0 read against FORWARD 0's base row (the folded form has no untouched row). It restarts from the kept stream
    only when the class's later forwards have already shown that restart exact on this item, or under one_row."""
    h, plan, on = sit.h, sit.plan, sit.plan["readouts"]
    none = {k: None for k in ("dose", "base_from", "restart", "rows_identical", "metrics", "per_row", "gap",
                              "between_kl_mean", "same_bits", "seconds")}
    if not sit.twin:
        return {"taken": False, "why": sit.twin_why, **none}
    sit.where["forward"] = "folded"
    main = plan["main_dose"]
    restart = st.kept is not None and (len(st.keys) >= 2 or plan["lower"] == "one_row")
    _sync(h)
    t0 = time.perf_counter()
    with we.folded(h.lm, sit.edit, main, hooks_at_load=h.hooks_at_load):
        Hfold = _restart(h, sit.L0, st.kept) if restart else _whole(h, item.ids, st.R)
    _sync(h)
    seconds = time.perf_counter() - t0
    # a row that is not finite never equals its twin (NaN is not NaN), and an edit can break the model: such a twin
    # has no figures and is not a void, as an arm that is not finite is not (`rows_identical` is then None)
    finite = bool(torch.isfinite(Hfold[0]).all())
    same_rows = bool(torch.equal(Hfold[0], Hfold[st.R - 1])) if finite else None
    if same_rows is False:
        raise _Void("untouched_rows_differ", f"Rows 0 and {st.R - 1} of the folded forward are not the same bits "
                       f"(item {sit.where['item']}, {st.R} rows): every row of a folded forward runs the same weights.")
    key = _key("edit", main)
    arm = next(a for a in arms if a["key"] == key)
    r_arm = st.keys[0].index(key)
    _, got = _measure(h, on, sp, st.Hf0[0], [Hfold[0] if finite else None])
    metrics = None if got[0] is None else {m: stats(got[0][m]) for m in METRICS if on[m]}
    gap, between = None, None
    if metrics is not None and arm["metrics"] is not None:
        gap = {}
        for m, s in metrics.items():
            a, b = s["mean"], arm["metrics"][m]["mean"]
            if a is None or b is None:
                gap[m] = {"abs": None, "rel": None}
                continue
            d, top = abs(a - b), max(abs(a), abs(b))
            gap[m] = {"abs": d, "rel": d / top if top else None}
        if on["kl"]:  # KL(folded row || per-row row) over the span
            _, bt = _measure(h, {"kl": True}, sp, Hfold[0], [st.Hf0[r_arm]])
            between = stats(bt[0]["kl"])["mean"]
    return {"taken": True, "why": None, "dose": main, "base_from": FOLDED_BASE_FROM, "restart": restart,
            "rows_identical": same_rows, "metrics": metrics, "per_row": arm["metrics"], "gap": gap,
            "between_kl_mean": between, "same_bits": bool(torch.equal(Hfold[0], st.Hf0[r_arm])), "seconds": seconds}


def _read_class(sit: _Sitting, item: Item, sp: _Span, st: _Begun) -> Dict[str, Any]:
    """A class of one item, read: forward 0's null rows and arms, the later forwards with their rules and arms, the
    twin, the floor and the rank."""
    h, plan, on = sit.h, sit.plan, sit.plan["readouts"]
    n0 = len(plan["null_e"])
    arms = _read_rows(sit, sp, st.Hf0, st.keys[0], 0, skip_null=False)
    first = st.Hf0[:1 + n0]
    for f in range(1, len(st.keys)):
        sit.where["forward"] = f
        with we.per_row(h.lm, sit.programs(st.keys[f])):
            Hf = _restart(h, sit.L0, st.kept) if st.kept is not None else _whole(h, item.ids, st.R)
        _check_forward(sit, Hf, st.keys[f], f, first)
        arms += _read_rows(sit, sp, Hf, st.keys[f], f, skip_null=True)  # its null rows are forward 0's by bits
        del Hf
    folded = _twin(sit, item, sp, st, arms)
    floor = _floor(arms, on)
    odds0 = st.base_raw.get("odds")
    base = {"own": dict(st.gate), "odds": None}
    if odds0 is not None:
        s = stats(odds0)
        base["odds"] = {"mean": s["mean"], "median": s["median"]}
    return {"rows": st.R, "forwards": len(st.keys), "lower": plan["lower"],
            "restart_from": sit.L0 if st.kept is not None else None, "untouched_rows_identical": True,
            "repeats_identical": True if len(st.keys) > 1 else None, "base": base, "arms": arms, "floor": floor,
            "rank": _rank(plan, arms, floor), "folded": folded, "peak_gib": sit.peak()}


def _stability(plan: Mapping[str, Any], small: Mapping[str, Any], large: Mapping[str, Any], same_bits: bool) -> Dict[str, Any]:
    """The two classes of one item set side by side, for every read-out that is on, against the larger of the two
    floors (`agree`). Since 16 November 2026 the rule is read on the edit's arms alone (the spec's open point 5: the
    card check showed control arms at the floor tripping it): each arm of the edit goes in `arms` with its two means,
    their ratio and whether they agree, and only an arm of the edit can make the item unstable; each control edit's arm
    goes in `controls` with its two means and their ratio, taken as the rule takes it, to be printed, not judged."""
    ratio = plan["bars"]["stable_ratio"]
    big = {a["key"]: a for a in large["arms"]}
    out: Dict[str, Any] = {"ratio_bar": ratio, "base_rows_same_bits": bool(same_bits), "metrics": {}, "disagree": []}
    for m in METRICS:
        if not plan["readouts"][m]:
            continue
        floor = max((small["floor"][m]["mean_abs"] or 0.0), (large["floor"][m]["mean_abs"] or 0.0))
        per: Dict[str, Any] = {"arms": {}, "disagree": [], "controls": {}}
        for a in small["arms"]:
            if a["what"] == "null":
                continue
            b = big.get(a["key"])
            x = a["metrics"][m]["mean"] if a["metrics"] else None
            y = b["metrics"][m]["mean"] if b and b["metrics"] else None
            ok, r = agree(x, y, floor, SIGNED[m], ratio)
            if a["what"] == "control":  # its ratio, beside the edit's; whether it agrees is not asked
                per["controls"][a["key"]] = {"small": x, "large": y, "ratio": r}
                continue
            per["arms"][a["key"]] = {"small": x, "large": y, "ratio": r, "agree": ok}
            if ok is False:
                per["disagree"].append(a["key"])
                out["disagree"].append([m, a["key"]])
        out["metrics"][m] = per
    return out


_METRIC_WORDS = {"kl": "KL", "own": "the text's own log-probability", "odds": "log-odds of the named pieces"}


def _arm_name(key: str) -> str:
    """An arm's key as a person reads it: "edit:1.0" is the edit at dose 1, "control:3" control edit 3."""
    what, _, x = str(key).partition(":")
    if what == "control":
        return f"control edit {x}"
    try:
        number = f"{float(x):g}"
    except ValueError:
        return str(key)
    return f"the edit at dose {number}" if what == "edit" else f"the null row e = {number}"


def _under(figures: Mapping[str, Optional[float]], bar: float) -> bool:
    return figures["mean"] is None or figures["mean"] < bar


def _gate_caution(gate: Mapping[str, Any], R_small: int, R_large: int, bar: float) -> str:
    """The item gate's words for an item under the bar at either row count: its two figures, where it is under, and
    `GATE_SAY`. Since 16 November 2026 they are a caution printed beside an item that is read, not a refusal."""
    def shown(cls: str) -> str:
        mean = gate[cls]["mean"]
        return "a figure that is not finite" if mean is None else f"{mean:.2f}"

    under = [R for cls, R in (("small", R_small), ("large", R_large)) if _under(gate[cls], bar)]
    where = "both row counts" if len(under) == 2 else f"{under[0]} rows"
    return (f"The base model gives this item's answer {shown('small')} at {R_small} rows and {shown('large')} at "
            f"{R_large}, in nats a piece on average: under the item gate's bar of {bar:.2f} at {where}. {GATE_SAY}")


def _paired_item(sit: _Sitting, item: Item, entry: Dict[str, Any]) -> None:
    """One item of a paired reading, both classes: steps a to d of the spec's 4.7, written into `entry` as it goes.
    The item gate is a printed caution since 16 November 2026 (the spec's open point 4, the card having found that the
    base's log-probability does not separate the items whose two row counts disagree): an item under the bar is read
    in both classes like any other, its `gate.passed` is false and `gate.caution` holds the words."""
    h, plan = sit.h, sit.plan
    sp = _Span.of(item, h.device)
    bar, R_small, R_large = plan["bars"]["item_gate"], plan["rows"]["small"], plan["rows"]["large"]
    gate: Dict[str, Any] = {"bar": bar, "small": None, "large": None, "passed": False, "caution": None}
    entry["gate"] = gate
    one_row: Dict[str, torch.Tensor] = {}
    st = _begin_class(sit, item, sp, "small", one_row)
    gate["small"] = st.gate
    small_base = st.Hf0[0, sp.lo:sp.T].clone()  # the base row at the span and end positions, for the large class
    entry["classes"]["small"] = _read_class(sit, item, sp, st)
    del st
    st = _begin_class(sit, item, sp, "large", one_row)
    gate["large"] = st.gate
    same_bits = bool(torch.equal(st.Hf0[0, sp.lo:sp.T], small_base))
    entry["classes"]["large"] = _read_class(sit, item, sp, st)
    del st
    gate["passed"] = not _under(gate["small"], bar) and not _under(gate["large"], bar)
    if not gate["passed"]:
        gate["caution"] = _gate_caution(gate, R_small, R_large, bar)
    stab = _stability(plan, entry["classes"]["small"], entry["classes"]["large"], same_bits)
    entry["stability"] = stab
    if stab["disagree"]:
        m, key = stab["disagree"][0]
        r = stab["metrics"][m]["arms"][key]["ratio"]
        more = f" and {len(stab['disagree']) - 1} more" if len(stab["disagree"]) > 1 else ""
        entry["status"] = "unstable"
        entry["why"] = (f"{R_small} and {R_large} rows disagree on {_METRIC_WORDS[m]} for {_arm_name(key)}{more} "
                        f"(ratio {'not finite' if r is None or not math.isfinite(r) else format(r, '.2f')}, "
                        f"bar {plan['bars']['stable_ratio']:.2f}): the two row counts ran two arithmetics and this "
                        "item's figures moved with them, so it is left out of the summary.")
    else:
        entry["status"] = "ok"


def _summary(plan: Mapping[str, Any], entries: Sequence[Mapping[str, Any]], asked: int) -> Dict[str, Any]:
    """Over the items kept (status ok): each arm's mean and median over items of its per-item mean, the median of its
    per-item medians and the sum of its counts above a nat; the largest floor; the edit's rank among control edits.
    `under_bar` counts the items read with the item gate's caution, whatever their status (an item kept among them is
    in the figures like any other); `refused` stays for the results written before the gate became a caution."""
    kept = [e for e in entries if e["status"] == "ok"]
    out: Dict[str, Any] = {"items": {"asked": asked, "kept": len(kept),
                                     "refused": sum(e["status"] == "refused" for e in entries),
                                     "unstable": sum(e["status"] == "unstable" for e in entries),
                                     "one_arithmetic": sum(bool(e["stability"] and e["stability"]["base_rows_same_bits"])
                                                           for e in entries),
                                     "under_bar": sum(bool((e.get("gate") or {}).get("caution"))
                                                      for e in entries)},
                           "classes": {"small": None, "large": None}}
    on = [m for m in METRICS if plan["readouts"][m]]
    for cls in ("small", "large") if kept else ():
        blocks = [e["classes"][cls] for e in kept]
        arms: Dict[str, Any] = {}
        for key in [a["key"] for a in blocks[0]["arms"]]:
            per = {}
            for m in on:
                got = [a["metrics"][m] for b in blocks for a in b["arms"] if a["key"] == key and a["metrics"]]
                means = [s["mean"] for s in got]
                per[m] = {"mean_over_items": _mean(means), "median_over_items": _low_median(means),
                          "median_of_medians": _low_median([s["median"] for s in got]),
                          "above_1_total": sum(s["above_1"] for s in got)}
            arms[key] = per
        floors = {m: max([b["floor"][m]["mean_abs"] for b in blocks if b["floor"][m]["mean_abs"] is not None],
                         default=None) for m in on}
        rank = None
        C = plan["controls"]["count"]
        if C:
            m = plan["rank_by"]

            def over_items(key: str) -> Optional[float]:
                x = arms.get(key, {}).get(m, {}).get("mean_over_items")
                return None if x is None else abs(x)

            value = over_items(_key("edit", plan["main_dose"]))
            rank = {"metric": m, "dose": plan["main_dose"], "value": value,
                    **rank_among(value, [over_items(_key("control", c)) for c in range(C)])}
        out["classes"][cls] = {"rows": plan["rows"][cls], "arms": arms, "floor": floors, "rank": rank}
    return out


def _written_item(sit: _Sitting, item: Item) -> Dict[str, Any]:
    """One item of a written reading: every row writes greedily from the item's prompt alone, under its program, in
    one batch. Untouched rows and an arm at dose 0 must write the base row's pieces with its log-probabilities,
    exactly. The log-probabilities are the writer's own, from the model's dtype logits: not the fp32 measure."""
    h, plan = sit.h, sit.plan
    R, max_new = int(plan["rows"]), int(plan["max_new"])
    sit.where.update({"class": None, "rows": R, "forward": None})
    sit.reset_peak()
    keys = layout(plan["doses"], plan["controls"]["count"], plan["null_e"], R, one_forward=True)[0]
    stop_ids = [int(t) for t in h.rt.end_of_turn_ids()]
    wrote: List[Dict[str, Any]] = [{"pieces": [], "logprobs": [], "ended": False} for _ in range(R)]
    steps: List[float] = []
    last = R
    with we.per_row(h.lm, sit.programs(keys)):
        for st in ao.write_rows(h.lm, h.W_U, [list(item.prompt_ids)] * R, None, max_new=max_new, stop_ids=stop_ids,
                                bos_id=h.tokenizer.bos_token_id, softcap=h.softcap):
            if st.row <= last:  # the rows of a step come in order, so a row not after the last one begins a step
                steps.append(time.perf_counter())
            last = st.row
            wrote[st.row]["pieces"].append(int(st.token_id))
            wrote[st.row]["logprobs"].append(float(st.logprob))
            wrote[st.row]["ended"] = wrote[st.row]["ended"] or bool(st.end)
    base = wrote[0]
    at = f"item {sit.where['item']}, {R} rows"
    if not all(math.isfinite(x) for x in base["logprobs"]):
        raise _Void("non_finite_base", f"The base row wrote a piece at a log-probability that is not finite ({at}).")

    def same(r: int) -> bool:
        return wrote[r]["pieces"] == base["pieces"] and wrote[r]["logprobs"] == base["logprobs"]

    for r in [R - 1] + [r for r in range(1, R - 1) if keys[r] is None]:
        if not same(r):
            raise _Void("untouched_rows_differ", f"Row {r}, which nothing touched, did not write the base row's "
                           f"pieces with its log-probabilities ({at}): the rows of this batch are not independent.")
    arms = []
    for r, k in enumerate(keys):
        if k is None:
            continue
        facts = sit.arms[k]
        if facts["what"] == "edit" and facts["dose"] == 0 and not same(r):
            raise _Void("zero_dose_differs", f"The edit at dose 0 (row {r}) did not write the base row's pieces "
                           f"with its log-probabilities ({at}).")
        mine = wrote[r]
        n = next((i for i, (a, b) in enumerate(zip(mine["pieces"], base["pieces"])) if a != b), None)
        if n is None and len(mine["pieces"]) != len(base["pieces"]):
            n = min(len(mine["pieces"]), len(base["pieces"]))
        upto = len(base["pieces"]) if n is None else n
        gaps = [abs(a - b) for a, b in zip(mine["logprobs"][:upto], base["logprobs"][:upto])]
        entry = _arm_entry(k, facts)
        entry.update({"row": r, "pieces": mine["pieces"], "text": _text(h.tokenizer, mine), "ended": mine["ended"],
                      "same_as_base": same(r), "parts_at": n,
                      "logprob_gap": {"max_abs": max(gaps) if gaps else None,
                                      "mean_abs": sum(gaps) / len(gaps) if gaps else None, "n": len(gaps)}})
        arms.append(entry)
    sit.peak()
    return {"item": _item_block(item), "rows": R, "max_new": max_new, "stop_ids": stop_ids,
            "untouched_rows_identical": True,
            "seconds_a_step": (steps[-1] - steps[0]) / (len(steps) - 1) if len(steps) > 1 else None,
            "base": {"pieces": base["pieces"], "text": _text(h.tokenizer, base), "logprobs": base["logprobs"],
                     "ended": base["ended"]},
            "arms": arms}


def _text(tokenizer: Any, row: Mapping[str, Any]) -> str:
    """A written row's pieces as text, its end mark left off."""
    pieces = row["pieces"][:-1] if row["ended"] else row["pieces"]
    return tokenizer.decode(pieces)


# ----------------------------------------------------------------------------------------------------- one reading


def _not_base_words(h: Handle) -> Optional[str]:
    """Why a reading cannot begin on this model, in the plan's words; None when it is the base."""
    words = _overlay_words(h.lm)
    if words:
        return words
    now = we.hook_census(h.lm)
    if now != h.hooks_at_load:
        # the census names the text model itself "" and the hooks torch keeps for all modules "(every module)"
        said = {"": "the text model itself", "(every module)": "every module at once"}
        names = sorted(n for n in set(now) | set(h.hooks_at_load) if now.get(n) != h.hooks_at_load.get(n))
        return (f"The model carries hooks that were not there at load (on {', '.join(said.get(n, n) for n in names[:3])}): "
                "a reading is taken on the base model only.")
    try:
        we.assert_base(h.lm, h.hooks_at_load)
    except we.NotBase as exc:
        return str(exc)
    return None


def _same_base(h: Handle, edit: Any) -> None:
    """An edit handed in already built must be of this model's base: the same model, the same weights, the same
    tensors. Raises `Refused` in the words a saved edit is refused with."""
    base = edit.base
    if base.get("model") != h.model_id or base.get("weights_id") != h.weights_id:
        raise Refused(f"This edit was made on {base.get('model')} (weights {base.get('weights_id')}); the loaded "
                      f"model is {h.model_id} (weights {h.weights_id}). An edit is a difference from one base and is "
                      "not carried to another.")
    for name, pin in (base.get("tensors") or {}).items():
        try:
            now = we.tensor_pin(h.lm.get_submodule(name))
        except AttributeError:
            raise Refused(f"This edit names {name}, which the loaded model does not have.") from None
        if now != pin:
            raise Refused(f"This edit was made against other weights: {name} then hashed to {str(pin)[:12]}, the "
                          f"loaded model's hashes to {now[:12]}.")


def _fell_back() -> Optional[int]:
    n = getattr(backend.engine_runtime.dequantize_w4a16_full, "fell_back", None)
    return int(n) if isinstance(n, int) else None


def _code_hashes() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name, path in (("weight_edits", getattr(we, "__file__", None)), ("levelling", __file__),
                       ("level_book", getattr(lb, "__file__", None))):
        try:
            out[name] = file_sha256(Path(path)) if path else None
        except OSError:
            out[name] = None
    return out


class _Reading:
    """One call of `read`, from the plan as given to the result: what it has reached so far is always what the result
    is built from, so that a reading stopped anywhere is still written whole."""

    def __init__(self, h: Handle, reading: Any, items: Mapping[str, Item], base_dir: Path, edit: Any, save_edit: bool):
        self.h, self.items, self.base_dir, self.given_edit, self.save_edit = h, items, base_dir, edit, save_edit
        self.given = dict(reading) if isinstance(reading, dict) else reading
        if edit is not None and isinstance(self.given, dict):
            self.given["edit"] = {"built": getattr(edit, "id", None) if isinstance(edit, we.Edit) else None}
        self.plan: Optional[Dict[str, Any]] = None
        self.kind = self.given.get("kind") if isinstance(self.given, dict) else None
        self.name = self.given.get("name") if isinstance(self.given, dict) else None
        self.status, self.why, self.void = "ok", None, None
        self.edit: Any = None
        self.controls: List[Any] = []
        self.saved: Optional[str] = None
        self.entries: List[Dict[str, Any]] = []
        self.asked = 0
        self.est: Optional[Dict[str, Any]] = None
        self.sit: Optional[_Sitting] = None
        self.peaks: List[float] = []
        self.where: Dict[str, Any] = {"item": None, "class": None, "rows": None, "forward": None}
        self.unpacker = {"mode": None, "fell_back_before": None, "fell_back_after": None}
        self.t_begin = time.perf_counter()
        self.t0: Optional[float] = None

    # -- step 2: everything that can refuse a reading before any forward
    def prepare(self) -> List[Item]:
        h = self.h
        words = _not_base_words(h)
        if words:
            raise Refused(words)
        self.plan = normalise(self.given, n_layers=h.n_layers, tokenizer=h.tokenizer, dtype=h.dtype)
        plan = self.plan
        names = list(self.items) if plan["items"] == "all" else plan["items"]
        for name in names:
            if name not in self.items:
                raise Refused(f'The reading asks for the item "{name}", which was not given.')
        if not 1 <= len(names) <= MAX_ITEMS:
            raise Refused(f"A reading takes 1 to {MAX_ITEMS} items; {len(names)} were given.")
        chosen = [self.items[n] for n in names]
        self.asked = len(chosen)
        if self.given_edit is not None:
            if not isinstance(self.given_edit, we.Edit):
                raise Refused(f"An edit handed to a reading is a weight_edits.Edit; {type(self.given_edit).__name__} "
                              "is not one.")
            _same_base(h, self.given_edit)
            # through the gate again, its id worked from what it holds: the ledger takes no id that is not the hash
            # of what ran, and an object made by hand carries nothing the gate refuses
            self.edit = we.gate_again(h.lm, self.given_edit, model_id=h.model_id, weights_id=h.weights_id)
        else:
            recipe = plan["edit"]
            if "built" in recipe:
                raise Refused(f"The plan names an edit built in Python ({str(recipe['built'])[:12]}) and none was "
                              "handed in: give the edit's recipe, or the id of a saved edit.")
            try:
                # the recipe as it was given: the recorded plan names an array a caller wrote into it by what it is,
                # not by its numbers, and the gate is told what was handed in
                got = we.from_recipe(self.given["edit"], self.base_dir)
                if isinstance(got, (str, Path)):
                    self.edit = we.load(got, h.lm, model_id=h.model_id, weights_id=h.weights_id)
                else:
                    self.edit = we.build(h.lm, got, model_id=h.model_id, weights_id=h.weights_id,
                                         note=str(recipe.get("note") or ""))
            except OSError as exc:
                raise Refused(f"A file the edit names cannot be read ({exc.strerror or exc}).") from None
        edit = self.edit
        facts = we.edit_facts(edit)
        _fact_refusals(plan, facts, h.n_layers)
        for d in plan["doses"]:
            words = we.dose_refusal(edit, d, h.dtype)
            if words:
                raise Refused(words)
        if plan["controls"]["count"]:
            self.controls = list(we.control_edits(edit, h.lm, plan["controls"]["seed"], plan["controls"]["count"]))
        twin, twin_why, cannot = False, None, None
        if plan["kind"] == "paired":
            if plan["folded"] == "never":
                twin_why = WHY_NO_TWIN["never"]
            elif plan["folded"] == "always":
                twin, cannot = True, we.can_fold(h.lm, edit)
            elif facts.get("has_lowrank"):
                twin_why = WHY_NO_TWIN["lowrank"]
            elif not we.twin_wanted(facts, plan["main_dose"]):
                twin_why = WHY_NO_TWIN["exact"]
            else:
                twin_why = we.can_fold(h.lm, edit)
                twin = twin_why is None
        self.est = estimate(plan, [(len(it.ids), len(it.prompt_ids)) for it in chosen], n_layers=h.n_layers,
                            facts=facts, twin=twin)
        if cannot:
            raise Refused(cannot)
        for odds in [plan["readouts"]["odds"]] if plan["kind"] == "paired" and plan["readouts"]["odds"] else []:
            for t in odds["toward"] + odds["away"]:
                if t >= int(h.W_U.shape[0]):
                    raise Refused(f"The piece {t} is not in this model's vocabulary of {int(h.W_U.shape[0])}.")
        if self.save_edit:
            self.saved = str(we.save(edit))
        L0 = int(edit.lowest_layer())
        self.sit = _Sitting(h=h, plan=plan, edit=edit, controls=self.controls, L0=L0,
                            arms=_arm_facts(plan, edit, self.controls, L0, h.dtype), twin=twin, twin_why=twin_why,
                            where=self.where, peaks=self.peaks)
        return chosen

    # -- steps 4 to 6
    def forwards(self, chosen: Sequence[Item]) -> None:
        h, plan, sit = self.h, self.plan, self.sit
        self.unpacker["mode"] = unpacker_describe().get("mode")
        self.unpacker["fell_back_before"] = _fell_back()
        _sync(h)
        self.t0 = time.perf_counter()
        for item in chosen:
            self.where.update({"item": item.name if item.name is not None else item.id[:12], "class": None,
                               "rows": None, "forward": None})
            if plan["kind"] == "written":
                self.entries.append(_written_item(sit, item))
            else:
                entry = {"item": _item_block(item), "status": None, "why": None, "gate": None,
                         "classes": {"small": None, "large": None}, "stability": None}
                self.entries.append(entry)  # an item that was begun is in the result, however the reading ends
                _paired_item(sit, item, entry)

    def stopped(self, code: str, why: str) -> None:
        self.status, self.why = "void", why
        w = self.where
        self.void = {"code": code, "item": w["item"], "class": w["class"], "forward": w["forward"]}
        if self.plan and self.plan.get("kind") == "paired" and self.entries and self.entries[-1]["status"] is None:
            self.entries[-1]["status"], self.entries[-1]["why"] = "void", why

    def out_of_memory(self) -> str:
        w = self.where
        shape = f"{w['rows']} rows" if w["rows"] else "its rows"
        return (f"Memory ran out on item {w['item']} at {shape}. A reading never halves its rows, because the row "
                "count is part of what it is: ask fewer rows, a shorter item or fewer new pieces.")

    # -- step 7
    def result(self) -> Dict[str, Any]:
        h, plan = self.h, self.plan
        self.sit = None
        if h.device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            we.assert_base(h.lm, h.hooks_at_load)
            base_after = True
        except Exception as exc:  # NotBase, or the assertion itself failing: either way the base is not shown
            base_after = False
            if self.status == "ok":
                self.status, self.why = "void", str(exc) or type(exc).__name__
                self.void = {"code": "not_base_after", "item": None, "class": None, "forward": None}
        _sync(h)
        actual = time.perf_counter() - (self.t0 if self.t0 is not None else self.t_begin)
        self.unpacker["fell_back_after"] = _fell_back()
        paired = bool(plan) and plan["kind"] == "paired"
        edit_block = None
        if self.edit is not None:
            doses = plan["doses"] if plan else []
            edit_block = {"id": self.edit.id, "terms": self.edit.report, "note": self.edit.note,
                          "exact": {repr(float(d)): we.exact_at(self.edit, d, h.dtype) for d in doses},
                          "saved": self.saved}
        controls = None
        if plan:
            controls = {"family": "other_sites", "matched": False, "seed": plan["controls"]["seed"],
                        "count": plan["controls"]["count"], "ids": [c.id for c in self.controls],
                        "says": CONTROLS_SAY.format(seed=plan["controls"]["seed"])}
        shown = plan if plan is not None else _clean(self.given)
        predicted = self.est["seconds"] if self.est else None
        out = {
            "v": VERSION, "kind": self.kind, "name": self.name, "status": self.status, "why": self.why,
            "void": self.void, "base_after": base_after, "edit": edit_block,
            "base": {"model": h.model_id, "weights_id": h.weights_id},
            # numpy: control edits are drawn with its Generator, whose stream it does not promise across versions.
            # start_mark: the tokenizer's start mark, so that an entry says of each item whether it began with one.
            "runtime": {**h.runtime, "unpacker": dict(self.unpacker),
                        "hooks_at_load": sum(len(v) for v in h.hooks_at_load.values()), "dtype": str(h.dtype),
                        "threads": torch.get_num_threads(), "code": _code_hashes(), "numpy": np.__version__,
                        "start_mark": getattr(h.tokenizer, "bos_token_id", None)},
            "plan": shown, "plan_sha256": _sha(_clean(shown)), "bars_overridden": _bars_overridden(plan),
            "rows": ({"small": plan["rows"]["small"], "large": plan["rows"]["large"],
                      "says": ROWS_SAY.format(small=plan["rows"]["small"], large=plan["rows"]["large"])}
                     if paired else None),
            "controls": controls, "items": self.entries,
            "summary": _summary(plan, self.entries, self.asked) if paired and self.status == "ok" else None,
            "seconds": {"predicted": predicted, "actual": actual, "ratio": actual / predicted if predicted else None,
                        "parts": dict(self.est["parts"]) if self.est else {}},
            "peak_gib": max(self.peaks) if self.peaks else None,
        }
        return _clean(out)


def read(h: Handle, reading: dict, items: Mapping[str, Item], *, base_dir: Path = Path("."),
         edit: Optional[Any] = None, save_edit: bool = False) -> dict:
    """One reading, paired or written, entered in the ledger whatever happens.

    `reading` is a plan as given (it is normalised here); `items` the items it may name; `base_dir` where a recipe's
    files are looked for. `edit`: an Edit already built in Python, read in place of the plan's recipe; the recorded
    plan then holds "edit": {"built": "<id>"}. `save_edit`: save the edit read (`we.save`) and name its folder in the
    result's edit block. Returns the result with a key "book": {"seq", "hash"} added after the entry is written.

    A refusal or a void is a result, not an exception. `lb.BookBroken` is raised before the model is touched when the
    ledger cannot take an entry. Any other exception inside the reading, a KeyboardInterrupt included, is entered as
    a void ("error" or "interrupted") and then raised again. However the reading ends, `we.assert_base` runs last:
    the result's `base_after` says whether the model is the base again."""
    lb.head()
    run = _Reading(h, reading, items, Path(base_dir), edit, save_edit)
    raised: Optional[BaseException] = None
    try:
        with torch.no_grad():
            try:
                chosen = run.prepare()
            except (Refused, we.Refused, we.NotBase) as exc:
                run.status, run.why, chosen = "refused", str(exc), None
            if chosen is not None:
                try:
                    run.forwards(chosen)
                except _Void as v:
                    run.stopped(v.code, v.why)
    except KeyboardInterrupt as exc:
        run.stopped("interrupted", "The reading was interrupted before it ended.")
        raised = exc
    except BaseException as exc:  # entered, then raised again: every call is in the ledger
        if isinstance(exc, Exception) and is_oom(exc):
            run.stopped("out_of_memory", run.out_of_memory())
        else:
            run.stopped("error", f"{type(exc).__name__}: {exc}")
            raised = exc
    out = run.result()
    del run
    entry = lb.append(out["kind"] if out["kind"] in ("paired", "written") else "note", out)
    out["book"] = {"seq": entry["seq"], "hash": entry["hash"]}
    if raised is not None:
        raise raised
    return out


# ------------------------------------------------------------------------------------------- what a person reads


def _n(x: Any, digits: int = 5) -> str:
    """A figure for a person: `digits` decimals, and three figures for anything under a thousandth (floors on the
    card read 0.00001 to 0.00024, and a floor and an edit beside it must not print alike when they differ by half)."""
    if x is None:
        return "not finite"
    x = float(x)
    return f"{x:.3g}" if x != 0 and abs(x) < 1e-3 else f"{x:.{digits}f}"


def _g(x: Any) -> str:
    return "none" if x is None else f"{float(x):g}"


def _report_entries(report: Any) -> List[Mapping[str, Any]]:
    """The per-term entries of an edit's report, wherever the report keeps them."""
    if isinstance(report, Mapping):
        if isinstance(report.get("terms"), list):
            report = report["terms"]
        elif "linear" in report:
            report = [report]
        else:
            report = [v for v in report.values() if isinstance(v, Mapping) and "linear" in v]
    return [e for e in report or [] if isinstance(e, Mapping)] if isinstance(report, list) else []


def _edit_words(ed: Optional[Mapping[str, Any]]) -> str:
    if not ed:
        return "no edit was built"
    parts = []
    exact = (ed.get("exact") or {}).get("1.0") or {}
    silenced = bool(exact.get("factors")) and all(f[0] == 0 for f in exact["factors"])
    all_scales = all(not (x.get("ops") or {}).get("set") and not (x.get("ops") or {}).get("add")
                     for x in _report_entries(ed.get("terms")) if x.get("kind") != "lowrank")
    for e in _report_entries(ed.get("terms")):
        layer = e.get("layer")
        where = f"layer {layer} (page {e.get('page_layer', None if layer is None else layer + 1)})"
        if e.get("kind") == "lowrank":
            parts.append(f"a low-rank term of rank {e.get('rank')} on {e.get('linear')}, {where}: {e.get('form')}")
            continue
        ops = e.get("ops") or {}
        only_scales = bool(ops.get("scale")) and not ops.get("set") and not ops.get("add")
        if silenced and only_scales:
            done = "silenced"
        else:
            done = ", ".join(f"{ops[k]} {w}" for k, w in (("scale", "scaled"), ("set", "set"), ("add", "added to"))
                             if ops.get(k)) or "changed"
            if only_scales and all_scales and 1 <= len(exact.get("factors") or []) <= 4:
                done += " (by " + ", ".join(_g(f[0]) for f in exact["factors"]) + ")"  # at dose 1 a column's g is its f
        n = e.get("count")
        parts.append(f"{n} column{'' if n == 1 else 's'} of {where} {done}" if done == "silenced"
                     else f"{n} column{'' if n == 1 else 's'} of {where}: {done}")
    note = f' ("{ed["note"]}")' if ed.get("note") else ""
    return f"edit {str(ed.get('id'))[:12]}{note}: " + ("; ".join(parts) if parts else "its terms are in the result")


def _arm_words(arm: Mapping[str, Any], on: Sequence[str], exact: Optional[Mapping[str, Any]] = None) -> str:
    """One arm's figures. `exact` is what the cast to the model's dtype does at this arm's dose (`we.exact_at`): an
    arm whose factors round is said with the factors as applied, and one that uses the output-side add is marked."""
    if arm.get("non_finite"):
        return "this row holds a number that is not finite: no figure"
    if arm.get("same_as_base"):
        return "this row is the base row by bits"
    got = []
    for m in on:
        s = (arm.get("metrics") or {}).get(m)
        if s and s.get("n") == 1:  # one position (log-odds at the end): one figure, not a distribution
            got.append(f"{_METRIC_WORDS[m]} {_n(s['mean'])} at the one position read")
        elif s:
            got.append(f"{_METRIC_WORDS[m]} mean {_n(s['mean'])}, median {_n(s['median'])}, {s['above_1']} "
                       f"position{'' if s['above_1'] == 1 else 's'} above a nat of {s.get('n')}")
    said = "; ".join(got)
    exact = exact or {}
    if exact.get("input_side") == "rounds":
        pairs = [f for f in exact.get("factors") or [] if f[0] != f[1]][:3]
        if pairs:
            said += ("; input-side factor" + ("" if len(pairs) == 1 else "s") + " as applied: "
                     + ", ".join(f"{_g(b)} where {_g(a)} was asked" for a, b in pairs))
    if exact.get("output_add"):
        said += " (output-side add, summed in float32: not yet measured on the card)"
    return said


def _class_lines(result: Mapping[str, Any], cb: Mapping[str, Any], on: Sequence[str]) -> List[str]:
    plan, exact = result["plan"], (result.get("edit") or {}).get("exact") or {}
    R, C, n0 = cb["rows"], plan["controls"]["count"], len(plan["null_e"])
    out = [f"    {R} rows, {cb['forwards']} forward{'' if cb['forwards'] == 1 else 's'}"
           + (f", restarted from layer {cb['restart_from']} (page {cb['restart_from'] + 1})"
              if cb.get("restart_from") is not None else "") + ":"]
    for arm in cb["arms"]:
        if arm["what"] == "edit":
            out.append(f"      the edit at dose {_g(arm['dose'])}: "
                       f"{_arm_words(arm, on, exact.get(repr(float(arm['dose']))))}")
    no_floor = False
    for m in on:
        fl = cb["floor"].get(m) or {}
        if fl.get("mean_abs") is None:
            no_floor = no_floor or m == plan["rank_by"]
            out.append(f"      floor of {_METRIC_WORDS[m]}: no null row gave a figure (each of the {n0} came out the "
                       "base row by bits, or not finite)")
        else:
            e = str(fl["from"]).split(":", 1)[-1]
            left = n0 - int(fl["of"])
            out.append(f"      floor of {_METRIC_WORDS[m]}: {_n(fl['mean_abs'])} (the null row e = {e}, the largest "
                       f"of {fl['of']}; a null row's largest median {_n(fl['median_abs'])}, most positions above a "
                       f"nat {fl['above_1']}"
                       + (f"; {left} null row{'' if left == 1 else 's'} came out the base row by bits or not finite "
                          f"and {'is' if left == 1 else 'are'} not counted" if left else "") + ")")
    rk = cb.get("rank")
    if rk:
        if rk["rank"] is None:
            out.append(f"      the edit's row is not finite at dose {_g(rk['dose'])}: it has no rank")
        else:
            bad = f", {rk['non_finite']} of them not finite" if rk["non_finite"] else ""
            out.append(f"      on {_METRIC_WORDS[rk['metric']]} at dose {_g(rk['dose'])}: {rk['at_least']} of {C} "
                       f"unmatched control edits read as much{bad} (rank {rk['rank']} of {rk['of']})"
                       + ("" if no_floor else f"; {rk['controls_over_floor']} of {C} control edits read above the "
                                              "floor at all")
                       + (f"; the edit reads {_g(round(rk['over_floor'], 2))} times the floor"
                          if rk.get("over_floor") is not None else ""))
    fo = cb["folded"]
    if not fo["taken"]:
        out.append(f"      no twin taken: {fo['why']}")
    else:
        how = "restarted from the kept stream" if fo["restart"] else "a whole forward"
        got = []
        for m in on:
            f, p, gp = (fo["metrics"] or {}).get(m), (fo["per_row"] or {}).get(m), (fo["gap"] or {}).get(m)
            if f and p and gp:
                got.append(f"{_METRIC_WORDS[m]} mean {_n(f['mean'])} folded beside {_n(p['mean'])} per row, a gap of "
                           f"{_n(gp['abs'], 6)}" + (f" ({_g(round(gp['rel'], 6))} of the larger)"
                                                    if gp.get("rel") is not None else ""))
            else:
                got.append(f"no figure of {_METRIC_WORDS[m]}, a row is not finite")
        out.append(f"      folded twin at dose {_g(fo['dose'])} ({how}, read against the base row of "
                   f"{fo['base_from']}): " + "; ".join(got))
        out.append("      the folded row and the per-row row are " + ("the same bits" if fo["same_bits"] else
                   "not the same bits") + (f"; mean KL between them {_n(fo['between_kl_mean'], 7)}"
                                           if fo.get("between_kl_mean") is not None else "")
                   + f"; the twin took {float(fo['seconds']):.2f} s")
    return out


def _item_label(it: Mapping[str, Any], start_mark: Any) -> str:
    """An item as a person reads it: its name, the first 12 of its id, its pieces, and a word when it does not begin
    with the tokenizer's start mark or holds it more than once (the pilots' raw text had none, and read as text the
    runtime would never send)."""
    name = it["name"] if it.get("name") is not None else "unnamed"
    said = f"item {name} [{str(it.get('id'))[:12]}] ({it['pieces']} pieces, {it['scored']} scored"
    if start_mark is not None:
        ids = list(it.get("prompt_ids") or []) + list(it.get("answer_ids") or [])
        k = sum(1 for t in ids if t == start_mark)
        if not ids or ids[0] != start_mark:
            said += "; NO start mark at its head"
        if k > 1:
            said += f"; {k} start marks"
    return said + ")"


def _paired_lines(result: Mapping[str, Any]) -> List[str]:
    plan = result["plan"]
    on = [m for m in METRICS if plan["readouts"][m]]
    small, large = plan["rows"]["small"], plan["rows"]["large"]
    C = plan["controls"]["count"]
    void = result.get("status") == "void"
    start_mark = (result.get("runtime") or {}).get("start_mark")
    out = [f"  rows {small} and {large}; doses {', '.join(_g(d) for d in plan['doses'])} (the control edits, the "
           f"ranks and any twin at dose {_g(plan['main_dose'])}); {len(plan['null_e'])} null rows a forward "
           f"(e = {', '.join(_g(e) for e in plan['null_e'])}); {C} control edit{'' if C == 1 else 's'}",
           f"  read-outs: {', '.join(_METRIC_WORDS[m] for m in on)}"
           + (f"; ranks by {_METRIC_WORDS[plan['rank_by']]}" if C else "; no control edit, so no rank")]
    if plan["readouts"]["odds"]:
        od = plan["readouts"]["odds"]
        out.append(f"  log-odds toward pieces {od['toward']} against {od['away']}, read at "
                   + ("the one position after the answer, not piece by piece" if od["at"] == "end"
                      else "the answer's positions"))
    if plan["lower"] == "one_row":
        out.append("  lower: one_row. The layers below the edit ran at one row at both row counts, so the two row "
                   "counts differ only from the edit's lowest layer up.")
    elif plan["lower"] == "whole":
        out.append("  lower: whole. Every forward ran every layer; nothing was restarted.")
    if C:  # with no control edit there is nothing for the sentence to be about
        out.append("  " + (result.get("controls") or {}).get("says", ""))
    out += ["  " + (result.get("rows") or {}).get("says", ""), "  " + PAIRED_SAY,
            "  " + FLOOR_SAY.format(n0=len(plan["null_e"]))]
    for e in result.get("items") or []:
        it, gate = e["item"], e.get("gate") or {}
        figures = ", ".join(f"{_n((gate.get(c) or {}).get('mean'), 2)} at {plan['rows'][c]} rows"
                            for c in ("small", "large") if gate.get(c))
        out.append(f"  {_item_label(it, start_mark)}: {e['status']}; the base gives "
                   f"{figures or 'no figure'} in nats a piece (bar {_n(gate.get('bar'), 2)})")
        if gate.get("caution"):  # under the bar: read all the same (since 16 November 2026), and said so
            out.append(f"    {gate['caution']}")
        if e.get("why") and e["why"] != result.get("why"):
            out.append(f"    {e['why']}")
        read = [c for c in ("small", "large") if e["classes"].get(c)]
        if void and read:  # what was read before a void is in the book with status void: it is not printed as figures
            out.append("    read before the reading came out void: its figures are in the book, not printed")
            continue
        # a result written before 16 November 2026, when the gate refused an item at the large row count after the
        # small was read: those figures stay unprinted
        if e["status"] == "refused" and read:
            out.append(f"    read at {plan['rows'][read[0]]} rows only, before the gate refused the item: those "
                       "figures are in the book, not printed")
            continue
        for c in read:
            out += _class_lines(result, e["classes"][c], on)
        stab = e.get("stability")
        if stab:
            one = bool(stab["base_rows_same_bits"])
            if one:
                out.append(f"    {small} and {large} rows gave the same base row by bits: one arithmetic on this "
                           "item, so it was read once, not twice")
            for m in on:
                per = stab["metrics"][m]
                ratios = [a["ratio"] for a in per["arms"].values() if a["ratio"] is not None]
                # a ratio the result holds as None beside agree False was not finite (or one side gave no figure)
                if any(a["ratio"] is None and a["agree"] is False for a in per["arms"].values()):
                    top = "largest ratio not finite"
                else:
                    top = f"largest ratio {_n(max(ratios), 2)}" if ratios else "no ratio to take"
                # a result written since 16 November 2026 keeps its control edits apart, in "controls": the rule read
                # the edit's arms alone (before, the control edits were in "arms" too, and were judged)
                ctl = per.get("controls")
                judged = "" if ctl is None else " for the edit's arms"
                if per["disagree"]:
                    out.append(f"    {small} and {large} rows disagree on {_METRIC_WORDS[m]} for "
                               f"{', '.join(_arm_name(k) for k in per['disagree'][:6])}"
                               f"{' and more' if len(per['disagree']) > 6 else ''} ({top}, bar "
                               f"{_n(stab['ratio_bar'], 2)})")
                else:
                    out.append(f"    {small} and {large} rows agree on {_METRIC_WORDS[m]}{judged}"
                               + (" by construction, being one arithmetic" if one else "")
                               + f" ({top}, bar {_n(stab['ratio_bar'], 2)}); this does not show another row count "
                                 "would agree")
                if ctl:
                    seq = sorted(ctl.items(), key=lambda ka: int(str(ka[0]).partition(":")[2]))
                    over = [k for k, a in seq if a["ratio"] is None or a["ratio"] > stab["ratio_bar"]]
                    out.append(f"    the control edits' ratios on {_METRIC_WORDS[m]} between {small} and {large} rows, "
                               "printed and not judged (the rule reads the edit's arms alone), from control edit "
                               f"{seq[0][0].partition(':')[2]}: " + ", ".join(_n(a["ratio"], 2) for _k, a in seq)
                               + (f"; {len(over)} not finite or over the bar ("
                                  f"{', '.join(_arm_name(k) for k in over[:6])}{' and more' if len(over) > 6 else ''})"
                                  if over else "; none over the bar"))
                fa, fb = (((e["classes"][c] or {}).get("floor") or {}).get(m, {}).get("mean_abs") for c in read[:2])
                if fa and fb and max(fa, fb) / min(fa, fb) > stab["ratio_bar"]:
                    apart = _g(round(max(fa, fb) / min(fa, fb), 1))
                    out.append(f"    the floor of {_METRIC_WORDS[m]} itself moved between the row counts: "
                               f"{_n(fa)} at {small} rows and {_n(fb)} at {large}, {apart} times apart. The arms "
                               "were set against the larger of the two, so their agreement here shows less than it "
                               "says: a null row that moves with the row count is itself a sign that the item's "
                               "figures do")
    sm = result.get("summary")
    if sm:
        n = sm["items"]
        kept = [e for e in result.get("items") or [] if e.get("status") == "ok"]
        under = n.get("under_bar") or 0
        out.append(f"  over {n['kept']} item{'' if n['kept'] == 1 else 's'} kept of {n['asked']} ({n['refused']} "
                   f"refused, {n['unstable']} unstable, {n['one_arithmetic']} read in one arithmetic"
                   + (f", {under} under the item gate's bar and read with its caution" if under else "") + "):")
        for c in ("small", "large"):
            cb = (sm.get("classes") or {}).get(c)
            if not cb:
                continue
            for key, per in cb["arms"].items():
                if not key.startswith("edit:"):
                    continue
                gave = sum(1 for e in kept for a in ((e["classes"].get(c) or {}).get("arms") or [])
                           if a["key"] == key and a.get("metrics"))
                short = (f" (over {gave} of the {len(kept)} kept: on the other{'' if len(kept) - gave == 1 else 's'} "
                         "this row was not finite)" if gave < len(kept) else "")
                got = "; ".join(f"{_METRIC_WORDS[m]} mean over items {_n(s['mean_over_items'])}, median over items "
                                f"{_n(s['median_over_items'])}, median of medians {_n(s['median_of_medians'])}, "
                                f"{s['above_1_total']} positions above a nat" for m, s in per.items())
                dose = _g(float(key.split(":", 1)[1]))
                out.append(f"    at {cb['rows']} rows, the edit at dose {dose}{short}: {got}")
            out.append(f"    at {cb['rows']} rows, the largest floor: "
                       + ", ".join(f"{_METRIC_WORDS[m]} {_n(v)}" for m, v in cb["floor"].items()))
            rk = cb.get("rank")
            if rk and rk["rank"] is not None:
                out.append(f"    at {cb['rows']} rows, on {_METRIC_WORDS[rk['metric']]} over items: {rk['at_least']} "
                           f"of {C} unmatched control edits read as much (rank {rk['rank']} of {rk['of']})")
    return out


def _written_lines(result: Mapping[str, Any]) -> List[str]:
    plan = result["plan"]
    out = [f"  a written reading, greedy: every row writes its likeliest piece at each step, up to {plan['max_new']} "
           f"new pieces, in one batch of {plan['rows']} rows. It is taken at one row count and is not a "
           "two-row-count reading. Its log-probabilities are the writer's own, from the model's dtype, not the fp32 "
           "measure of a paired reading. Nothing gates the item: the model writes its own answer."]
    if plan["controls"]["count"]:
        out.append("  " + (result.get("controls") or {}).get("says", ""))
    start_mark = (result.get("runtime") or {}).get("start_mark")
    for e in result.get("items") or []:
        it, base = e["item"], e["base"]
        label = it["name"] if it.get("name") is not None else "unnamed"
        step = f"; {float(e['seconds_a_step']):.3f} s a step" if e.get("seconds_a_step") is not None else ""
        prompt = list(it.get("prompt_ids") or [])
        mark = ("; NO start mark at its head" if start_mark is not None and (not prompt or prompt[0] != start_mark)
                else "")
        out.append(f"  item {label} [{str(it.get('id'))[:12]}] ({it['prompt_pieces']} prompt pieces{mark}){step}")
        out.append(f"    base row, {len(base['pieces'])} pieces, {'ended' if base['ended'] else 'did not end'}: "
                   f"{base['text']!r}")
        for arm in e["arms"]:
            who = (f"null row e = {_g(arm['e'])}" if arm["what"] == "null" else
                   f"the edit at dose {_g(arm['dose'])}" if arm["what"] == "edit" else
                   f"control edit {arm['key'].split(':', 1)[1]}")
            gap = arm["logprob_gap"]
            before = (f"largest log-probability gap before that {_n(gap['max_abs'], 4)} (mean {_n(gap['mean_abs'], 4)}, "
                      f"{gap['n']} piece{'' if gap['n'] == 1 else 's'})" if gap["n"] else "no piece before that to compare")
            if arm["same_as_base"]:
                out.append(f"    {who}: this row wrote the base row's pieces with its log-probabilities, exactly")
            elif arm["parts_at"] is None:
                out.append(f"    {who}: the base row's pieces, every one; {before.replace(' before that', '')}")
            else:
                out.append(f"    {who}: parts from the base row at piece {arm['parts_at']}; {before}; "
                           f"{'ended' if arm['ended'] else 'did not end'}: {arm['text']!r}")
    return out


def lines(result: dict) -> List[str]:
    """What a person reads of a result, in plain words: the reading, the edit, the base model, what was laid out, each
    item's figures at each row count beside its floor and its control edits, whether the two row counts agree, the
    summary and the seconds. The only status words are "ok", "refused", "void" and "unstable"."""
    plan = result.get("plan") if isinstance(result.get("plan"), Mapping) else {}
    out = [f'reading "{result.get("name")}" ({result.get("kind")}): {result.get("status")}']
    counts = (result.get("summary") or {}).get("items")
    if counts:  # a reading is "ok" when its mechanism was sound, whatever its items came to: say how many were kept
        out[0] += f", {counts['kept']} of {counts['asked']} item{'' if counts['asked'] == 1 else 's'} kept"
    if result.get("why"):
        out.append(f"  {result['why']}")
    out.append("  " + _edit_words(result.get("edit")))
    base = result.get("base") or {}
    out.append(f"  base model {base.get('model')} (weights {str(base.get('weights_id'))[:16]})")
    if result.get("bars_overridden"):
        own = {"item_gate": ITEM_GATE_BAR, "stable_ratio": STABLE_RATIO, "one_row_min_layer": ONE_ROW_MIN_LAYER}
        given = plan.get("bars") or {}
        out.append("  bars given by the plan in place of the module's: "
                   + ", ".join(f"{k} {_g(given.get(k))} (the module's {_g(v)})" for k, v in own.items()
                               if given.get(k) != v))
    if result.get("status") != "refused" and plan.get("kind") in ("paired", "written") and result.get("controls"):
        out += _paired_lines(result) if plan["kind"] == "paired" else _written_lines(result)
    if not result.get("base_after", True):
        out.append("  the model was NOT the base when this reading ended: nothing more should be read in this process")
    sec = result.get("seconds") or {}
    if sec.get("actual") is not None:
        out.append(f"  {float(sec['actual']):.1f} s" + (f" (predicted {float(sec['predicted']):.1f} s)"
                                                       if sec.get("predicted") is not None else ""))
    return out
