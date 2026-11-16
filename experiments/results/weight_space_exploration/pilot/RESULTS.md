> **Correction, found later the same day (14 November 2026, about 23:30Z).** Wherever this note says "raw text",
> "prose" or "the recipe" given without the chat template, the text had **no start mark**. The pilots built it with
> `tok(text, add_special_tokens=True)`, and this tokenizer adds no `<bos>` to plain text (checked on the CPU: the
> ids with and without special pieces are the same 13; the chat template's ids do begin with the start mark). So the
> -9.6 nats a piece, the eighty-fold swing between batch sizes and "the instruct model does not model raw text"
> describe text with its start mark missing, and nothing has yet been measured on raw text that has one. The
> pilots' own piece counts show it (the recipe is 90 pieces raw and 25 + 90 as an answer). The timings, the
> bit-equality findings and everything measured on chat-formatted items stand. What a harness should do stands too,
> for a plainer reason: an item must be built as the model is given text (through the chat template, or with the
> start mark put there by hand), and the base's own log-probability catches one that is not.

# Two timing pilots on the home card, 14 November 2026 (19:50Z and 19:54Z)

MEASURED today on the RTX 5090 (32 GB), Gemma-4-31B-it W4A16 checkpoint, compiled unpacker, no lens loaded, server
stopped, an idle ComfyUI process (0.5 GiB) and the desktop on the card (3.4 GiB in all). Read-only: nothing in the
repository was changed. Scripts and raw JSON: `pilot_edit_speed.py/.json`, `pilot2_equivalence.py/.json` in this
folder. One run each; timings are medians of 3 to 5 repeats within the run. Code layer 44 (page 45) was the edited
layer. "Question" is a 30-piece chat prompt; "prose" is a 176-piece paragraph.

## Load and memory
- Cold load of the model (no lens): 9.5 s. Compiled unpacker install with its self-test: 1.83 s.
- Model at rest: 17.97 GiB allocated; 8.37 GiB of the card free after the load.
- Peaks: 19.3 GiB for 64 rows x 30 pieces whole forward; 20.7 GiB for 64 rows x 176 pieces restarted from layer 44
  with the reading; 21.9 GiB for 64 rows x 176 pieces whole forward.

## A whole forward (all positions, no cache) is NOT free per row
| shape | seconds |
|---|---|
| 30 pieces, 1 row | 0.0875 |
| 30 pieces, 2 rows | 0.0886 |
| 30 pieces, 16 rows | 0.209 |
| 30 pieces, 32 rows | 0.354 |
| 30 pieces, 64 rows | 0.670 |
| 176 pieces, 1 row | 0.120 |
| 176 pieces, 8 rows | 0.503 |
| 176 pieces, 16 rows | 0.969 |
| 176 pieces, 32 rows | 1.972 |
So: a fixed cost of about 0.08 s (unpacking all 410 linears) plus about 0.33 ms per row-position through all 60
layers (about 3,000 positions a second). The recorded "a step costs the same from 1 to 64 rows" (28 October 2026)
is true of a DECODE STEP with a cache (one new position a row); it does not carry over to whole-sequence
teacher-forced forwards, where rows cost in proportion to their length.

## A per-row output-side delta costs nothing measurable
Forward hook on a linear adding, for each batch row, its own low-rank term to the linear's output (row 0 left alone).
- one down_proj, 32 rows x 30 pieces: no hook 0.3580 s; rank-1 (one neuron's column changed, a different neuron per
  row) 0.3589 s; rank 8 dense 0.3591 s; rank 64 dense 0.3594 s. At 64 rows: 0.6781 / 0.6784 / 0.6785 / 0.6814 s.
- rank-1 rows on all 60 down_proj at once, 32 rows: 0.3624 s (+1%).
- on attention linears (q, k, v, o of layer 44; q, k, o of full-attention layer 47, which has no v_proj): a zero
  delta gives the unhooked forward's bits exactly, and no measurable cost.
- a zero-delta row is bit-identical to a row of an unhooked forward of the same shape; the same edit in two rows of
  one forward gives identical bits; untouched rows of one forward are identical.

## The existing module-wide overlay (the vault's mechanism) is cheap too
- one column overlay on one down_proj: 0.0876 s against 0.0873 s (1 row); 64 column overlays on it: 0.0879 s.
- one column overlay on EACH of the 60 down_proj (60 clones of a 231 MB weight a forward): 0.106 s against 0.087 s,
  about 0.3 ms a cloned matrix.

## Restarting from the edited layer is exact and proportional
`atlas_ops.forward_from_layer` on a kept stream, same batch shape as the whole forward: the final states are
bit-identical to the whole forward's (torch.equal true at layers 20, 40, 44, 52, 58, both prompts), and the time
is the share of layers run: 32 rows x 30 pieces from layer 44 0.096 s (whole 0.361 s), from 52 0.049 s, from 58
0.013 s; 16 rows x 176 pieces from layer 44 0.263 s (whole 0.982 s). There is no fixed floor, because only the
layers run are unpacked.

## One paired teacher-forced reading: about 0.04 s a candidate edit
1 untouched row + N rows each silencing a different neuron of layer 44, restarted from layer 44 on the prose's kept
stream, reading at all 175 positions the mean log-probability of the text's own next piece and the KL to the
untouched row (fp32 logits over the whole 262,144 vocabulary):
- 15 candidates: 0.59 s; 31: 1.19 s; 63: 2.41 s, so 0.038 s a candidate. The forward is under half of it (1.09 s
  for 64 rows); the rest is the reading.
- The reading alone: 19 ms a row of 175 positions in fp32 (the study's measure), 3.5 ms in the model's own dtype,
  and the two differ by a mean KL of 0.0013 nats (max 0.004), so the cheap reading is too coarse for small effects.

## The two forms of one edit agree closely but not to the bit, and their gap is a floor
The same column change made (a) as a per-row output-side delta and (b) folded into the weight by the module-wide
overlay, read over 175 positions of the prose, batch of 2 in both:
| neuron (layer 44) | factor | effect, mean KL (a) | effect, mean KL (b) | between (a) and (b), mean KL | max | per-position effect correlation |
|---|---|---|---|---|---|---|
| 14802 | 0 | 0.01577 | 0.01551 | 0.00045 | 0.0057 | 0.9930 |
| 14802 | -3 | 0.3878 | 0.3859 | 0.00050 | 0.0040 | 0.9994 |
| 16573 | 0 | 0.00613 | 0.00599 | 0.00044 | 0.0052 | 1.0000 |
| 16573 | -3 | 0.02481 | 0.02518 | 0.00044 | 0.0034 | 1.0000 |
| 13086 | 0 | 0.00246 | 0.00242 | 0.00042 | 0.0026 | 0.9985 |
| 13086 | -3 | 0.01375 | 0.01380 | 0.00045 | 0.0026 | 1.0000 |
The gap between forms is about 0.00045 nats a position whatever the edit's size: a rounding floor of bf16
arithmetic, not a property of the edit. The MEDIAN effect of silencing one of the 63 hardest-firing neurons on this
text was 0.00041 nats, the same size as that floor. So a per-row delta ranks candidates faithfully and measures
effects above about 0.005 nats well; an effect near 0.0005 nats cannot be told from rounding in either form, and
the folded form is what must be confirmed before anything is claimed of an edit as it would ship.

## Batch shape
The untouched row's next-piece odds at batch 2 against batch 1, 16, 32, 64 on the question: never the same bits,
KL 1e-12 to 8e-8 on this (very confident) position. The instrument's recorded worst case elsewhere is 0.014.

## What was not measured
Generation under per-row deltas (the recorded decode step is 0.091 to 0.109 s for 1 to 64 rows, 28 October 2026);
a backward pass under the compiled unpacker; edits to norms, layer scalars or the embedding; low layers; a rank-r
delta folded into the weight; anything with the lens loaded (3.2 GiB less headroom); more than one prompt or run.

---

# Pilots 3 to 7, the same day (20:49Z to 21:00Z). READ THIS: it changes what a reading may be trusted for.

Same card, same set-up (compiled unpacker, no lens, server stopped). Scripts and JSON beside this file:
`pilot3_writing_and_capture`, `pilot4_cancellation`, `pilot5_knife_edge`, `pilot6_which_is_true`, `pilot7_batch_size`.
One run each.

## Writing under per-row edits works (pilot 3, A)
`atlas_ops.write_rows` (with its cache) under hooks registered round it, 8 rows, 48 pieces, two questions: the three
untouched rows wrote the same pieces with the same log-probabilities; a zero-delta row was the same as an untouched
row; the hooked write's untouched row was the same as an unhooked write of the same shape; two rows with the same
edit were the same; a step cost 0.095 s hooked and unhooked alike. A row whose down_proj output was multiplied by 3
(undone by the norm but for rounding) never parted from the base in 48 pieces, with a largest log-probability gap of
0.059 on a written piece.

## The mechanism is sound: no leak between rows, and input-side equals folded (pilot 6)
- The row an edit sits in does not matter: the same silencing in rows 1, 9, 20 and 31 of a batch of 32 gave the same
  figure to every printed digit; an untouched row far from the edited rows equalled row 0 exactly; the hooked batch's
  row 0 equalled an unhooked batch's.
- Silencing a neuron on the INPUT side of its down_proj (the row's activation set to nought before the product)
  gave the folded form's figure to five decimals at the same batch size, in all 7 cases at both sizes (for example
  0.72581 and 0.72581; 0.00908 and 0.0091). The OUTPUT-side delta (subtracting the neuron's term after the product)
  is a different arithmetic and did not: 0.0227 where the other two read 0.7258; 0.7026 where they read 0.5234.
  So for column edits the faithful per-row form is input-side (plus an output-side add for a new column), not a
  pure output-side delta.

## One edit, two readings: batch size alone moved a reading eighty-fold on raw text (pilots 4, 6, 7)
Recipe paragraph given as raw text (no chat template and, as the correction at the top says, NO start mark, though
this line first said it had one), one neuron silenced (input side, row 1), the
mean KL to the untouched row of the same forward over 149 positions:
| neuron | batch 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| layer 20, 4294 | 0.0091 | 0.0127 | 0.0075 | 0.0085 | 0.0085 | 0.0085 | 0.7258 | 0.7258 | 0.7258 | 0.7258 |
| layer 20, 9618 | 0.0053 | 0.0036 | 0.0034 | 0.0052 | 0.0052 | 0.0052 | 0.6019 | 0.6019 | 0.6019 | 0.6019 |
| layer 5, 13133 | 0.0191 | 0.0312 | 0.0186 | 0.0358 | 0.0358 | 0.0358 | 0.6674 | 0.6674 | 0.6674 | 0.6674 |
| layer 20, 2642 | 0.0042 | 0.0055 | 0.0030 | 0.0050 | 0.0050 | 0.0050 | 0.0057 | 0.0057 | 0.0057 | 0.0057 |
There are two arithmetics on this card at this length, one up to 12 rows and one from 16 (identical figures within
each), and for some neurons the edited model parts from the base at position 57 in one and not in the other. The
untouched model itself does not part between batch sizes (mean 0.006 to 0.016, no position above 0.12), and 93 null
rows (the feed-forward output scaled by 1 + e, which the norm undoes) never parted. Of the 30 hardest-firing neurons
at layer 20 on that text, 5 read above 0.1 at batch 32 and none at batch 2 (correlation of the two readings 0.17).

Why that text: it had no start mark, and the instruct model does not model text given so (whether it models raw
text that has its start mark was not measured). Its mean log-probability of the recipe's own next piece
was -9.6 nats, and where the rows part its likeliest next pieces are things like ' uma', '<|channel>', '0', '//'.
The same passages put through the chat template (as the model's answer to a question) behave:
| text, layer | silencing effect, median (batch 32) | max | neurons above 0.1 at batch 32 / at batch 2 | readings differing over 5x | correlation |
|---|---|---|---|---|---|
| recipe raw, 5 | 0.0083 | 0.667 | 2 / 0 | 2 of 30 | 0.23 |
| recipe raw, 20 | 0.0058 | 0.726 | 5 / 0 | 6 of 30 | 0.17 |
| prose raw, 5 | 0.0018 | 0.064 | 0 / 0 | 0 | 0.995 |
| prose raw, 20 | 0.0012 | 0.003 | 0 / 0 | 0 | 0.73 (all at the floor) |
| recipe as chat, 5 | 0.00045 | 0.0013 | 0 / 0 | 0 | 0.81 |
| recipe as chat, 20 | 0.00038 | 0.0058 | 0 / 0 | 0 | 0.998 |
| prose as chat, 5 | 0.00073 | 0.071 | 0 / 0 | 0 | 0.9997 |
| prose as chat, 20 | 0.00055 | 0.0007 | 0 / 0 | 0 | 0.01 (all at the floor) |

What follows for a harness:
1. An item must be in the model's own format and the base must model it: gate every item on the base row's own
   log-probability, and refuse the rest. (The first two pilots used raw text; their timings stand, their KL figures
   are of an off-distribution regime.)
2. A reading is not trusted until it is taken in both arithmetics (a small batch and a large one) and they agree;
   disagreement marks the item unstable rather than the edit strong.
3. Report the median over positions beside the mean, and how many positions exceed a nat: a parted row has dozens.
4. A column edit rides a row on the input side; an output-side delta is for low-rank terms only and is confirmed
   folded.

## The floor depends on the text and the layer (pilots 3 and 5)
Null rows (feed-forward output of one layer scaled by 1 + e, 31 rows), mean KL to the untouched row:
recipe raw 0.0066 / 0.0039 / 0.0006 at layers 5 / 20 / 44 (largest 0.012); prose raw 0.0015 / 0.0009 / 0.0004;
recipe as chat 0.00055 / 0.00046 / 0.00014. The untouched row between batch sizes: recipe raw 0.006 to 0.016;
prose raw 0.002 to 0.003; recipe as chat 0.0006 to 0.001. So a floor is measured on each item, in each reading;
no constant serves. At layer 44 the times-3 floor row read 0.0004 to 0.0006, the size of the delta-against-folded
gap measured there (0.00045).

## A one-row capture replayed at 32 rows (pilot 3, C)
At layer 44 the paired effects agreed with a same-shape whole forward (correlation 0.9998 and 0.9999; largest gap
0.0004 to 0.0005). At layers 5 and 20 they agreed on the prose (0.94 and 0.92) and not on the raw recipe (0.02 and
0.32), which is the two-arithmetics effect above, since a one-row capture and a 32-row forward sit on either side
of it. Not yet repeated on chat-formatted items.

## Single neurons sit at the floor here too
On chat-formatted text the median effect of silencing one of the 30 hardest-firing neurons was 0.0004 to 0.0007
nats at layers 5 and 20, beside null rows of 0.0005. Two to five of thirty single silencings cleared five times the
floor row on the raw texts at layers 5 to 44.

---

# Pilot 8, a first look on chat-formatted items (22:02Z the same day; not registered)

`pilot8_first_look.py` and its JSON. Six questions, each with the model's own greedy answer (56 to 72 pieces) as
the scored span; four layers (code 19, 35, 44, 54: pages 20, 36, 45, 55); sets of columns silenced on the input side
of down_proj; KL to the untouched row of the same forward over the answer's positions. One run.

## The item gate's raw material: the base model's mean log-probability of the scored span
| item | nats a piece (batch 2 / batch 24) | median piece | worst piece |
|---|---|---|---|
| the model's own answers (six) | -0.07 to -0.16 / -0.07 to -0.17 | -0.001 to -0.013 | -0.8 to -1.2 |
| a person's passage given as the answer (recipe) | -2.46 / -2.46 | -0.66 | -19.1 |
| a person's passage given as the answer (survey) | -5.51 / -5.50 | -2.65 | -26.0 |
| the same passages as raw text | -10.1 and -9.2 / -9.9 and -9.1 | -10.0, -9.5 | -23.0, -24.8 |
The three kinds are far apart. A provisional bar has room anywhere between about -1 (below every own answer) and
about -6 (above every raw text); a passage a person wrote sits between, and in pilot 7 such passages were stable.
Two items of each kind is too few to fix a bar: the kernel prints the figure with every item, refuses below a
provisional bar, and its card check measures more items before the bar is settled.

## How large a set must be to be heard (medians over the six items)
Floor: the largest null row (the layer's feed-forward output scaled by 1 + e) on the item, 25-row forward.
| layer (code) | floor | hardest-firing set of 1 | of 10 | of 64 | of 300 | of 1,000 | random set of 64 | of 300 | of 1,000 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.00024 | 0.0075 (35x floor) | 0.024 (100x) | 0.037 (160x) | 0.070 (300x) | 0.110 (440x) | 0.0003 (1.2x) | 0.0007 (2.5x) | 0.0028 (6x) |
| 35 | 0.00022 | 0.0025 (13x) | 0.037 (155x) | 0.060 (315x) | 0.111 (660x) | 0.192 (910x) | 0.0006 (3.1x) | 0.0022 (12x) | 0.0076 (43x) |
| 44 | 0.00008 | 0.0092 (108x) | 0.043 (750x) | 0.131 (2,400x) | 0.133 (2,500x) | 0.154 (3,000x) | 0.0002 (2.4x) | 0.0003 (5x) | 0.0032 (41x) |
| 54 | 0.00001 | 0.0007 (77x) | 0.0031 (270x) | 0.020 (1,660x) | 0.043 (3,360x) | 0.055 (5,260x) | 0.00004 (3.8x) | 0.0004 (36x) | 0.0018 (184x) |
- On the model's own answers the floor is ten to a hundred times lower than on the raw texts of the earlier pilots,
  and the single hardest-firing neuron of an item is heard above it (13 to 108 times; 4 to 6 of 6 items over five
  times the floor). The earlier "single neurons sit at the floor" was true of raw and person-written text.
- Random sets sit at the floor up to 64 columns and rise above it from 300 to 1,000.
- A hardest-firing set against a random set of the same size is not a matched comparison (most random neurons do
  not fire on the item): it shows only that there is something to hear. What it means needs controls matched on
  firing, which the kernel's first slice does not have.
- The median over positions is small (0.0001 to 0.005) where the mean is large: the effect sits at a few positions.

## The two batch arithmetics agree on these items
The hardest-firing sets read again in a forward of 8 rows against the forward of 25: the ratio of the two readings
had a median of 1.01 to 1.10 at every size and layer and a largest value of 1.6 (one neuron, layer 19); for sets of
10 or more the largest was 1.20. Nothing like the eighty-fold swing on raw text.

## A null row can be no row at all
A scale of 1.001 does not exist in bf16 (the next value above 1 is 1 + 2^-7 = 1.0078125), so a null row written as
"times 1.001" is the untouched row, bit for bit, and reads a floor of exactly nought: the 8-row forwards here, whose
two null rows were +-0.001, report a floor of 0. A power of two is no better (an exact scaling, which the norm may
undo exactly). A null row's factor must be a bf16 value other than 1 and not a power of two: 1 + 2^-7, 1 - 2^-8,
1.5, 0.75, 3. Several of pilot 5's null rows (those with e of 0.001 and 0.002) were no rows for the same reason;
its floors came from the others.

---

# The card check of slice 1a (16 November 2026)

MEASURED on the same card, 16 November 2026 01:33 to 01:46Z (UTC; the evening of the 15th on the desk's clock), by
`scripts/verify_levelling.py`: the kernel built from `BUILD_SPEC_1a.md` read on Gemma-4-31B-it W4A16, compiled
unpacker (its self-test passing on all 410 linears), lens loaded and freed, server stopped. One run: 16 s to load and about
777 s after it, 793 s in all; 369 readings (367 paired, 2 written) through `levelling.read` into a ledger of the check's own. Record:
`experiments/results/levelling/verify_1a.json` (its `sections`, `checks` and `proposed_bars`) and
`verify_1a_book.jsonl` beside it (370 entries, the chain whole). The three modules it ran are the ones committed at
`96e4420` (their sha256 in the record's `code` are those files'). Section numbers below are the check's (the spec's
6.2). 51 items: the model's own answers to pilot 8's six questions; the study's twelve chats; pilot 7's recipe and
survey and pilot 8's cuts of them, and the study's twelve prose passages, each given as the answer to a question; the
four passages and the first four prose texts raw with no start mark, as the pilots built them; the same eight behind
the start mark; and the first own answer with the start mark dropped from its prompt.

**142 of 145 standing lines held.** What held (each line in the record by its section):
- The runtime (§1): 60 layers of 21,504, no overlay, no hook at load, the compiled unpacker never fell back. The chat
  ids are the runtime's own on all 21 questions (§2), and the tokenizer adds no start mark to plain text (the recipe
  cut is 90 pieces with special pieces asked for and without): the correction at the top of this note, on the card.
- A row at dose 0 is the base row by bits for a scale, a set and a low-rank term, at 12 and 32 rows (§3); a silencing
  in rows 1, 9, 20 and 31 of 32 gives the same bits (§4).
- Input side against folded (§5, §5a): a silencing and a scale by 2, of 1, 64 and 1,000 hardest-firing columns at
  layers 5 and 20, at 12 and 32 rows, on a model's own answer and the recipe cut given as an answer: the same bits in
  48 of 48. Pilot 6 like for like (§5d, the raw recipe with no start mark, layer 20): neuron 4294 read 0.72581 at 32
  rows (pilot 6: 0.72581) and 0.00850 at 12 (pilot 7: 0.0085), and the item read "unstable", the stability rule's
  one known positive found; neuron 5316 read 0.52337 and 0.0035. The forms that round (§5c, the hardest 64 at layer
  20): a scale by 1.5 stood off its folded twin by 0.00026 to 0.00047 nats of KL (6% to 22% of readings of 0.002 to
  0.004),
  a scale by -3 by 0.0006 to 0.009 (under 1% of 1.4 to 1.6), a set by 0.0010 to 0.0032 (1% to 3%), the float32 add by
  0.00002 to 0.00037 (0.5% to 7%). A rank-1 low-rank term equal to a one-column add read as that add to 0.00004 and
  0.00017 nats (§5e); a reading with a rank-64 term took 1.56 s, with the add 1.62 s.
- A restart from the kept stream under hooks, 60 control edits, at layers 5, 20 and 44: every later forward's base
  and null rows forward 0's by bits, and `lower: "whole"` the same figures exactly (§6).
- Writing under the input-side pre-hook (§7): 8 rows, 48 pieces, on two own answers, not void; untouched and dose-0
  rows wrote the base's pieces with its log-probabilities; two rows stopped in the check's own write and the rest went
  on. A step took 0.110 s hooked and 0.107 s unhooked (within 10%). Null rows in writing never parted from the base in
  48 pieces; their largest log-probability gap was 0.10 (pilot 3: 0.059).
- The gate at -6 (§8): all six own answers and both cuts given as answers passed, the two raw cuts with no start mark
  were refused, at both row counts. On the six own answers no arm disagreed on KL between 12 and 32 rows (largest
  ratios 1.12 to 1.55; §9).
- After everything the model was the base, the unpacker had not fallen back and the ledger verified (§12). The
  largest peak was 19.33 GiB, 1.33 above the model at rest.

**What failed, and why.**
- The cost model (§10): 21 of the 163 readings it was judged on took more than 1.25 times their prediction (median
  1.12, worst 1.44). Every one of the 21 was a reading with 19 control edits (21 of those 40); the other 123 all sat
  within the band (median 1.11). The 145 readings predicted under a second, not judged, took 1.15 to 1.87 times their
  prediction. The model priced
  forwards and read-out from pilot 1's bare forward; what else a reading does is not in it, and the record does not
  say where the time went. A least-squares refit over 331 readings (`proposed_bars.cost`): 0.135 s a forward, 0.377 ms
  a row-position, 0.088 ms a scored row-position of read-out; on the same 163 readings it gives 0.85 to 1.13 (fit on
  them: not a test). Memory refit from the classes' peaks: 0.191 MiB a row-position (the spec had 0.25, inferred).
- Null rows with a position above a nat (§11, two lines: items over the -6 bar, and under it). 28 null rows of the
  784 read on raw items had one, 12 distinct rows (the readings at one layer and row count repeat their null rows) on
  three items: the raw recipe cut behind the start mark (over the bar, -4.82; layers 5 and 20), the raw survey behind
  the start mark (-6.96; layers 5 and 20, at 12 rows only) and the raw recipe with no start mark (-9.64; layer 5).
  Counted by read-out, all 28 had a position above a nat in the text's own log-probability and 17 in KL. None of the
  640 read on the model's own answers, the 608 on the study's chats, the 912 on passages given as answers or the 48
  on the own answer with its start mark dropped had one. On those two raw texts behind the start mark the null rows
  themselves moved with the row count: at layer 20 the recipe cut's floor was 0.0011 nats at 12 rows and 0.046 at 32
  (42.7 times), the survey's 0.025 and 0.0010. Both items still read "ok": an arm is set against the larger of the two
  floors, the blind spot the build's review named. So in this run raw text behind a start mark did not read as the
  chat-formatted items did either (on two of its eight items).

**The item gate does not separate.** With the gate set aside (§9: the hardest neuron and the hardest 64 at layers 5
and 20, silenced, no control edits), 3 of 51 items read "unstable": chat 10 (-3.09 nats a piece), the raw prose 2
behind the start mark (-4.13) and the raw recipe with no start mark (-9.64). The 48 that did not reach down to -12.13
(the raw prose 1 with no start mark). Two of the three are over the -6 bar, and 19 items under it behaved. With 19
control edits (§9a) 3 more items read "unstable" from control arms alone, all under the bar (prose 10 as an answer
-8.61, the raw prose 3 with no start mark -8.82, the raw recipe cut with no start mark -10.08): the figure still does
not separate. By the spec's own rule for this case (open point 4, written before the card ran) the gate is now a
printed caution, not a refusal. The figures by kind (§8b; each range over both row counts' means): own answers -0.07
to -0.17; the study's chats -0.000002 to -9.40; passages given as answers -2.45 to -8.61; raw with no start mark -8.12
to -12.13; raw behind the start mark -3.07 to -11.38; the own answer with the start mark dropped -1.93 and -1.92.

**Stability (§9a).** On items over the bar the largest ratio between the row counts that agreed was 2.44 (the raw
prose 2 behind the start mark, its own log-probability, in a reading whose KL disagreed; among readings that agreed on
both read-outs, 2.07), and 2.4% of ratios were over 2; 3 readings disagreed there (chat 10, 5.2 on KL; the raw prose
2 behind the start mark, 20.7 on KL for one neuron and 6.3 on its own log-probability for 64). Of 969 control arms
read twice (19 control edits, the hardest 64 at layer 20, every item), 6 disagreed, all on items under the bar, and
three of them at the floor at one of the row counts (two by the first count; control 9 on the raw recipe whole with no
start mark is the third, recounted the same day). Log-odds on the six own answers (toward the answer's first piece, away from its second):
floors 0.017 to 0.036, the edit's ratio 0.98 to 1.28. An item's two floors stood more than 3 times apart in 22 of 204
readings (eleven item and layer pairs, each read twice): eight of them 3.0 to 4.1 times apart, on chats, passages and
one raw text, and three 15.7 to 42.7 times, all on the two raw texts behind the start mark above. The spec's own
formula for a new ratio (1.5 times the largest seen over the bar) gives 31.1, from a reading that disagreed; from the
ratios that agreed it gives 3.66, from the readings that agreed on both read-outs 3.10. The bar stays 3.0 (open point
5).

**The map of row counts (§9b).** The base row's final states, by sha256, at 2, 3, 4, 6, 8, 12, 16, 24, 32 and 48
rows: on a study chat of 35 pieces every row count gave its own bits; on an own answer of 82 pieces only 12 and 24 were
one; on the recipe given as an answer, 175 pieces, 24, 32 and 48 were one. Nearly every row count is its own
arithmetic, so "two batch classes" is two of many; 12 and 32 never shared bits. The hardest 64 at layer 20, read at
each row count: 0.035 to 0.052 nats on the chat (1.5 times between the lowest and the highest), 0.129 to 0.141 on the
own answer, 0.0358 to 0.0374 on the recipe.

**The one-row capture (§6b)**, the hardest 64 silenced, a one-row capture against the same-shape reading, on an own
answer and a study chat: at layer 44 the gap was 0.00036 and 0.000004 nats of KL on the own answer (10.8 and 0.14
times its floor) and 0.0039 and 0.0033 on the chat (31 times its floor, 1.6% and 1.3% of the reading); at layer 40,
0.0020 and 0.0009 (35 and 10 times) and 0.0046 and 0.0066 (19 and 26 times). Pilot 3's agreement at layer 44 was on
raw text with no start mark; on these two chat-formatted items the capture was a different reading by more than the
floor in 7 of the 8 class readings, at both layers.

What it set, by the spec's rules written before the run (BUILD_SPEC_1a.md section 12, "The card check"): the gate a
caution at -6.0, the stability ratio 3.0, the one-row bar 44 and the row counts 12 and 32 unchanged, and the cost
figures refit. Not measured: any layer but 5, 20, 40 and 44; a second run; another card; a low-rank term that is not
a column's add. Later the same day, by open point 5's second clause (control arms at the floor tripped the rule), the
stability rule was changed to read the edit's arms alone and print the control arms' ratios beside them, not judged:
replayed on this run's figures it reads 3 items unstable where the old rule read 6. The next card run is its first.
