# Skeptic: the numbers and the mechanism

Attack on "Levelling" (`design/synthesis.md`), 14 November 2026. Read-only: nothing in the repository was changed, no
server started, the model was not loaded, the card was not touched.

**What I did.** Read the synthesis in full, the pilot note and every pilot script and JSON beside it, and the code at
the cited lines. Recomputed every "by arithmetic" time from the pilot JSON. Two things of my own, both on the CPU
with the card hidden (`CUDA_VISIBLE_DEVICES=""`), reading single tensors out of the checkpoint file, not loading the
model:

- the sixty post-feed-forward norm weights and layer scalars (the part of the synthesis's pilot 0 that needs no card);
- a rounding simulation with the real layer 5, 20, 44 and 55 `down_proj` matrices and norm weights and **synthetic**
  activations (script: `experiments/results/weight_space_exploration/verify/numerics_work/floor_sim.py`).

**One fact about timing governs half of what follows.** The synthesis was written before pilots 3 to 7 ran (their
JSON is stamped 20:49Z to 21:00Z; the synthesis cites only the first two). Those five pilots, one run each, contradict
the synthesis's central mechanism on measurement. `pilot/RESULTS.md` now says so itself; the synthesis does not.

Marks: **[measured]** a pilot JSON or a repository record; **[read]** code or a file I opened today; **[my
arithmetic]**; **[simulated]** my CPU simulation, synthetic activations, real weights.


## Verdicts at a glance

| # | Claim attacked | Verdict | Severity |
|---|---|---|---|
| 1 | An edit rides a row as an added term on the linear's output; that is the form of every practice and fresh reading | refuted | serious |
| 2 | A paired difference between rows of one forward measures the edit on the item | refuted on raw text; holds on the two chat-formatted texts tried | serious |
| 3 | A one-row capture replayed at many rows gives the same paired differences | weakened: holds at layer 44; fails at layer 5 on both texts and at layer 20 on the raw one | serious |
| 4 | The times-3 floor row measures the floor, and "five times the floor row" is a bar | weakened | serious |
| 5 | The hook sums in float32 and rounds once | weakened: nobody ran it, and it buys nothing | minor |
| 6 | `remove_direction` takes a stream direction through the norm weight | refuted as worded | serious |
| 7 | A whole forward costs 0.08 s + 0.33 ms a row-position; a restart its share; read-out 0.11 ms | stands, within about 15% | minor |
| 8 | Memory above the model is 0.35 to 0.75 MB a row-position; 5.6 GiB of headroom with the lens | refuted: both peaks are misattributed | minor (it errs safe) |
| 9 | A fresh reading takes 2 to 4 minutes and a trial 25 to 45 | weakened | serious |
| 10 | A dose ladder costs rows, not forwards | weakened | minor |
| 11 | Writing under per-row hooks keeps rows aligned | stands | minor |
| 12 | The base row is never memoised | stands, but it does not buy what is implied | minor |
| 13 | The unpacker's fall-back counter is read on every job | stands; the pilots did not read it | minor |


## 1. The output-side delta is not a faithful per-row form of a column edit. Refuted, serious.

**Claim** (synthesis iv, lines 190-201): a forward hook adds `dose · d` to the linear's output for the rows that
carry a term; this "delta" form serves every practice and fresh reading and differs from the folded weight "by about
0.00045 nats a position whatever the edit's size" (table i).

**Evidence.**

- Pilot 6 [measured, 20:57Z, one run]: silencing a neuron on the **input** side of `down_proj` gave the folded
  form's figure to five decimals in 7 of 7 cases at batch 2 and at batch 32. The output-side delta did not: layer 20
  neuron 4294 read 0.0227 where input-side and folded read 0.7258; neuron 5316 read 0.7026 against 0.5234; layer 5
  neuron 16222 read 0.2065 against 0.0083 (`pilot6_which_is_true.json`).
- Pilot 4 [measured]: over 30 neurons the two per-row forms disagreed by up to 0.635 nats a position at layer 5 and
  0.703 at layer 20 on the raw recipe, with a correlation of 0.02 at layer 5.
- The reason is plain algebra and I read it in the code: the folded form is the unpacked weight cloned, the column
  written, then one `F.linear` (`backend/engine_runtime.py:356-377`). Setting the neuron's input to nought makes the
  same product with the same zero term. An output-side delta instead re-rounds a product that was already rounded.
- The instrument already has the faithful form: the `mul` push multiplies `x[:, :, n]` per row before `down_proj`
  (`backend/atlas_ops.py:369-381`). The synthesis files it under "cheaper knobs" and says its exactness is "not
  assumed".
- "Whatever the edit's size" came from three neurons at layer 44, batch 2, on a raw text the model gave a mean
  log-probability of -7.64 nats a piece (`pilot_edit_speed.json`, P5 `base_mean_logprob`).

**Fix.** For `cols` scale and silence, ride the row on the input side. It is exact for 0 and for powers of two
(the audibility pilot's 0, 2, 4 qualify); any other factor rounds one scalar at bf16's 2^-8, which should be said.
For `rows` edits on `gate_proj` / `up_proj`, recompute the edited output coordinate (`x @ (W_n + Δ_n)`, rounded
once) instead of adding to it. Keep an output-side add only for content that is not a multiple of the column, and
confirm that folded **at the reading's own batch shape** (finding 2).


## 2. The paired difference depends on the batch's size, by up to eighty-fold. Refuted on raw text, serious.

**Claim** (synthesis iii and v): because edit and base share a forward, "the paired differences are exact"; the
measurement's stated weak points do not include the batch shape.

**Evidence.**

- Pilot 7 [measured, 21:00Z]: one silencing (input side, equal to folded), raw recipe, 150 pieces, mean KL to the
  untouched row of the same forward:

  | rows in the batch | 2 | 3 | 4 | 6, 8, 12 | 16, 24, 32, 48 |
  |---|---|---|---|---|---|
  | layer 20, 4294 | 0.0091 | 0.0127 | 0.0075 | 0.0085 | 0.7258 |
  | layer 5, 13133 | 0.0191 | 0.0312 | 0.0186 | 0.0358 | 0.6674 |

- Of the 30 hardest-firing layer-20 neurons on that text, 5 read above 0.1 at batch 32 and none at batch 2;
  correlation between the two readings 0.17 [measured].
- **There are at least six arithmetics by batch size at this length, not two.** Batches of 1, 2, 3 and 4 each give
  their own figures; 6 to 12 agree; 16 to 48 agree (`pilot7_batch_size.json`; `pilot5_knife_edge.json` for batch 1).
  `RESULTS.md` says "two ... identical figures within each"; the JSON does not bear that out below 6 rows.
- Null nudges never tipped a reading (93 null rows, pilot 5), so a floor row cannot see this.
- The synthesis's shapes cross these classes by design: a one-row capture replayed at 45 to 208 rows; a folded twin
  as "two forwards of one shape" beside a delta reading of another shape; a reading halved when memory runs out.
- **What the synthesis reads on raw text**: the habitat and neutral-text batteries (corpus windows are raw text
  begun with the start mark, `backend/atlas_build.py:114-131` [read]); pilot D's "16 stored passages"; the atlas
  product of section ii item 2; the known answer and the generator for the harness's own self-test (the E6 frames are
  raw completions, `experiments/scripts/E6_tool_audit_04_vault.py:66-71` [read]).
- What limits the damage: on two chat-formatted texts no neuron's two readings differed by more than five-fold at
  layers 5 and 20 (pilot 7). That is two texts, one run.

**What would settle it.** A map of where the classes change (by rows, or by rows times pieces: at 30 pieces batch 16
and batch 32 differ, at 150 they agree, so it is not rows alone); the same sweep on 16 or more chat-formatted items;
whether a fresh process and a server with the lens loaded land in the same class.

**Fix.** Items in the model's own format, each gated on the base row's own log-probability. Every reading that
claims anything taken at three batch sizes from different classes, disagreement marking the item unstable. Median
over positions and the count of positions above a nat printed beside the mean. The folded twin run at the reading's
shape. Pilots B and D and the habitat battery re-specified before anything is registered.


## 3. The one-row capture. Weakened, serious for the cost table.

**Claim** (synthesis v; xx.3): a stream captured at one row and replayed to every row gives paired differences that
agree with same-shape forwards within the floor; a practice reading therefore costs "about 5 s from layer 44".

**Evidence** (pilot 3 C [measured, 20:49Z], 30 silencings, 32 rows):

| text, layer | correlation of effects between the two ways | largest gap | floor row |
|---|---|---|---|
| prose, 44 | 0.9998 | 0.0004 | 0.0004 |
| prose, 20 | 0.918 | 0.0008 | 0.0009 |
| prose, 5 | 0.943 | 0.016 | 0.0015 |
| raw recipe, 20 | 0.317 | 0.70 | 0.0034 |
| raw recipe, 5 | 0.022 | 0.20 | 0.0049 |

The base rows of the two ways differ by a mean KL of 0.0017 to 0.011. By the synthesis's own rule in pilot C's row
("disagree by more than the floor: then ... early-layer readings cost whole forwards") the branch is taken at layer
5 on both texts and at layer 20 on the raw recipe; at layer 20 on the prose the gap stays under the floor row, with
every effect there near the floor. Not repeated on chat-formatted items.

**Fix.** Price every reading below about layer 40 at whole forwards until a chat-formatted repeat says otherwise:
the practice reading is about 13 s, not 5, and section ii's "about a second a neuron at mid depth" about doubles.
The time share itself is sound: replays of 32 x 176 from layers 5, 20, 44 took 1.803, 1.312, 0.525 s against
55/60, 40/60, 16/60 of 1.972 s = 1.807, 1.315, 0.526 [measured against my arithmetic].


## 4. The floor row and "five times the floor row". Weakened, serious.

**Claim** (synthesis v, vi; xx.6): the edited linear's output times 3, re-rounded, is undone by the norm, so what it
moves is the floor of the delta path; the bar is the brief's least effect or five times the floor row.

**What stands.** At layer 44 the row read 0.00042 and 0.00058 nats, the size of the 0.00045 gap (pilot 3 E
[measured]). In writing it never parted from the base in 48 pieces on two questions (largest log-probability gap
0.059). The norm is `x · (mean(x²) + 1e-6)^-0.5 · weight` in float32, cast back (`modeling_gemma4.py:205-223`
[read]). [Simulated] with the real layer-44 matrix: times 3 moves what the stream receives by 0.0028 (relative),
and the delta-against-folded gap is 0.0025 to 0.0030 for edits that re-round 40 to 70% of the output entries. With
layer 44's column norms (median 0.49 [read from the checkpoint]) and activations of 3 to 14, the raw output's RMS
is about 0.1, so epsilon is a hundredth of a percent of the mean square and does not show.

**What does not.**

- **Epsilon is unguarded.** [Simulated]: at a raw RMS of 0.01 the row reads 0.0050 instead of 0.0028; at 0.001 it
  reads 0.33, all of it epsilon. Nothing checks the mean square against 1e-6 at each position.
- **It overstates small edits.** A 1% scale moved 2.4% of the output entries and its gap to folded was 0.0006, a
  fifth of the times-3 row's 0.0028 [simulated].
- **It is one draw.** Thirty-one null rows in one forward spread up to 3.1-fold (raw recipe, layer 20: median
  0.0039, largest 0.0123; pilot 5 [measured]).
- **It varies 47-fold by text and layer** (0.00014 to 0.0066, pilot 5), so the 0.00045 in table i is no constant.
- **No floor has been measured in log-odds**, the synthesis's default read-out. A single signed draw averaged over
  items can sit near nought, and five times nought is no bar.
- **It is blind to finding 2.** Null nudges never tipped; real edits did. On an unstable item every real edit,
  chance edits included, clears five times the floor.
- **The `gate_proj` substitute is not a floor.** "The edit at a dose of 0.01" is 1% of the edit; a bar of five times
  it is 5% of the edit's own effect and scales with the edit.

**Fix.** Several null rows a forward with factors near 1 (as pilot 5 did; a factor near 1 also removes the epsilon
artefact), the bar set from their largest, in the metric's own units, log-odds included. Drop the 0.01-dose row.
Decision 1's threshold should not lean on the floor row alone.


## 5. "Summed in float32, rounded once". Weakened, minor.

No pilot ran that hook. Every pilot hook is `out + a * delta.to(out.dtype)`, a bf16 product and a bf16 sum, applied
to every row with zeros for the untouched ones (`pilot_edit_speed.py:206-214`, `pilot2_equivalence.py:121-124`,
`pilot3_writing_and_capture.py:119-127`, `pilot4_cancellation.py:120-128`, `pilot6_which_is_true.py:139-143`
[read]). So the "under 1%" and the 0.00045 are of another arithmetic than the one proposed. [Simulated]: the float32
sum changes the gap to folded by 2 to 20% (0.00253 against 0.00257; 0.00309 against 0.00397 at small outputs). The
floor is the re-rounding of an already rounded product, which no later precision undoes. Moot if finding 1's fix is
taken.


## 6. `remove_direction` through the norm weight. Refuted as worded, serious.

**Claim** (synthesis iv, lines 180-188): each chosen column is changed "so that its write as the stream receives it
loses dose times its component along u"; whether small entries of `w_post` make the raw change large is "not
established; pilot 0 reads the sixty weights".

**Evidence.** I read the sixty weights from the checkpoint file [read, CPU, 14 November 2026]:

- In every layer the largest entry is 1,075 to 1,660 times the smallest. None is negative.
- Layers 30 to 58 each have 200 to 590 of 5,376 entries under 5% of the layer's median. Layer 44: 306 such entries,
  median 1.07, smallest 0.00064. 153 entries are small in all of layers 30 to 58.
- The stream does carry signal there: the embedding's mean square on those entries is 0.00016 against 0.00018
  elsewhere, and the post-attention norm's weight is small on the same entries (0.002 against 0.66 at layer 44).
- For an isotropic unit `u`, `‖u ⊘ w_post‖` is 182 at layer 44 (99.99% of it in those 306 entries), 97 to 233 at
  layers 36 to 55, 511 at layer 5 [my arithmetic on the read weights].

So the literal op needs a raw column change one to five hundred times the write, in entries the block barely
writes; the norm's denominator then rescales the block's whole write. The contingency the synthesis names is
certain, at every layer. Two knock-ons it does not draw:

- the same-site null for a silencing ("an added write of equal stream-side length in a chance direction") needs the
  same division;
- a census-drawn chance direction is built through `w_post` or from write directions (`backend/census.py:1567-1591`
  [read]) and so has almost no mass in the small entries, while a difference-of-means `u` does. Real removals would
  be far louder than their nulls for a reason that has nothing to do with `u`. That biases pilot D, the gate.

**Fix.** Define the op in the block's writable geometry (the least-change form, raw change along `w_post ⊙ u`, or
`u` projected onto entries above a stated threshold), put real and chance directions through the same projection,
and fix the definition before pilot D is registered.


## 7. The cost model. Stands within about 15%, minor.

| Shape | Row-positions | Measured s | 0.08 + 0.33 ms x N | Error |
|---|---|---|---|---|
| 2 x 30 | 60 | 0.0886 | 0.0998 | +12.6% |
| 16 x 30 | 480 | 0.2089 | 0.2384 | +14.1% |
| 64 x 30 | 1,920 | 0.6696 | 0.7136 | +6.6% |
| 16 x 176 | 2,816 | 0.9689 | 1.0093 | +4.2% |
| 32 x 176 | 5,632 | 1.9716 | 1.9386 | -1.7% |
| 64 x 176 from layer 44 | 11,264 | 1.0929 | 1.013 | -7.4% |

- A least-squares fit of the nine whole forwards gives 0.049 s + 0.336 ms [my arithmetic].
- The slope rises with size: 0.303, 0.328, 0.331, 0.356 ms, and 0.364 at 64 x 176. The repository's own recorded
  figure is 0.36 ms a word-piece on windows of 512 (`backend/atlas_build.py:371-374` [read]). At the shapes the
  synthesis uses (2,000 to 10,000 row-positions) the model runs 5 to 10% low.
- Read-out is 0.117 ms a scored row-position once the KL and the own-piece log-probability are in it (0.3288 s /
  2,800, 0.6544 / 5,600, 1.3129 / 11,200), not 0.11.
- Not measured: more than 64 rows; more than 176 pieces; anything with the lens loaded; the capture stopped at the
  edited layer (the pilots ran the whole one-row forward).


## 8. Memory a row-position and the headroom. Refuted, minor because it errs safe.

**Claim** (synthesis v, line 265): "0.35 to 0.75 MB a row-position [arithmetic from the peaks of 21.9 GiB at
64 x 176 and 19.3 GiB at 64 x 30]"; "about 5.6 GiB of headroom" in the server.

**Evidence.**

- **21.9 GiB is the load's peak, not a forward's.** `pilot2_equivalence.py` and `pilot3_writing_and_capture.py`
  never reset the peak counter (lines 242 and 299 [read]) and report the same 21.908 GiB from different work. Model
  and lens rest at 21.2 GiB before the lens is freed (`backend/engine_runtime.py:470`).
- **19.3 GiB is not 64 x 30.** The first pilot resets the peak once a section (`pilot_edit_speed.py:67-72`), and
  that section also ran 32 x 176. Another section that ran 64 x 30 peaked at 18.529 GiB.
- What the JSON does support [measured]: 1.36 GiB above rest for a section reaching 32 x 176 whole; 2.73 GiB for
  64 x 176 from layer 44 with the full reading; about 0.47 GiB at batch 2. That is about 0.25 MiB a row-position all
  in, at 5,600 to 11,300 row-positions. No per-shape peak exists.
- **Writing is a different figure the synthesis omits**: about 0.8 MB a row a piece of context with a cache
  (`scripts/census_run.py:98`; 4.8 to 5.3 GiB above the model at 64 rows, 7.2 to 7.5 at 96 [measured, 28 October
  2026]).
- **5.6 GiB** is the digest's arithmetic at a mid-sized desktop. Like for like with today's measured 8.37 GiB it is
  5.2 GiB with the desktop at 3.4 GiB and 3.2 GiB with it at 5.4; the lens's 3.2 GiB is itself "worked out, not
  measured" (`scripts/census_run.py:100`).

**Fix.** Re-measure with the counter reset per shape, including 89 and 208 rows and the lens loaded (pilot G names
neither row count).


## 9. Does a reading fit in one forward, and how long does it take? Weakened, serious.

**Fit** [my arithmetic at 0.25 MiB a row-position, 1 GiB kept spare; unmeasured above 64 rows]:

| Where | Budget | 45 rows | 89 rows | 208 rows |
|---|---|---|---|---|
| Script, lens freed | 7.4 GiB | up to about 670 pieces | 340 | 145 |
| Server, lens, desktop 3.4 GiB | 4.2 GiB | 380 | 190 | 80 |
| Server, lens, desktop 5.4 GiB | 2.2 GiB | 200 | 100 | 43 |

So 19 + 19, 39 + 39 and 99 + 99 chance edits with the other rows do fit one teacher-forced forward at 48 pieces,
the last only just in a busy server. They do not fit at the lengths the batteries need: stored windows are 512
pieces, and 6 of 21 faces do not fire on their own passage with 60 pieces of context [recorded, 14 November 2026].
A **written** battery at 89 rows needs about 5.6 GB and at 208 rows about 13 GB: neither fits beside the lens.

**Time** [my arithmetic from the synthesis's own model; 0.36 ms in brackets]:

| Reading | Synthesis says | By its own model | What is left out |
|---|---|---|---|
| Known answer, 12 x 38 x 30 | about 2 s; 5.5 whole | 1.96 s; 5.47 s | agrees |
| Practice, 16 x 45 x 48 | about 5 s; 13 whole | 4.8 s; 12.7 s (13.7) | 4 s more with KL; whole below layer 40 (finding 3) |
| Three voices | about 5 s | 4.6 s at 0.095 s a step [measured, pilot 3] | agrees |
| Fresh, 96 x 89 x 48 "plus the batteries" | 2 to 4 min | 2.4 min (2.6) for the item forwards alone | five batteries, unsized. At 32 texts each beside the chance edits: 48 s a teacher-forced one, about 3.8 min a written one. About 10 to 13 min in all |
| Trial, 256 items, 99 + 99 | 25 to 45 min | 14.4 min (15.8) for the item forwards alone | a written gate on 256 items is 20 to 25 min; batteries at 208 rows; about 60 to 75 min in all unless gate and batteries are subsampled |

Two things make these worse than a matter of sizing:

- **The trial's form is stated two ways.** Sections iii, iv and vi say a trial of record is in folded form; the cost
  row says "99 + 99 chance edits, folded twin on a subsample". Folded edits cannot share a forward, and packing
  several texts into one is unbuilt (section xii). Folded nulls at one row each are 199 x 256 forwards: about 20
  minutes from layer 44, 50 from layer 20, 76 whole [my arithmetic].
- **Finding 2's stability check multiplies every line**, by about two if each reading is repeated in a second
  batch class.

**Fix.** Size the batteries and the written gate in the table; say which form the trial's nulls take; add the
stability repeat to the model; drop "under an hour" until a trial has been timed.


## 10. A dose ladder. Weakened, minor.

Rows do carry doses. But bf16 quantises small ones differently in each form. Output side: a 1% change moved 2.4% of
the output entries and its gap to folded was 70% of its effect [simulated]. Input side: exact only at 0 and powers
of two. Folded: every dose is a forward. So "a known edit at shrinking doses says what a fresh reading can detect"
would measure the path's rounding, not the reading's power. **Fix:** ladders at dyadic doses; shrink the set of
columns, not the dose, for a power curve.

The folded twin for many columns is cheap (64 column overlays on one linear add 0.6 ms; 60 cloned matrices 19 ms
[measured]) but belongs at the reading's batch shape; a low-rank term has no folded path, as the synthesis says.


## 11. Writing under per-row hooks. Stands, minor.

Pilot 3 A [measured]: 8 rows, 48 pieces, two questions; untouched rows, a zero row and an unhooked write identical in
pieces and log-probabilities; two rows with one edit identical; 0.095 s a step hooked or not. `write_rows` keeps the
batch whole after a row ends and left-pads (`backend/atlas_ops.py:552-617` [read]), so a row-count check holds, and
the streaming route holds the lock for the whole write (`backend/atlas.py:2167-2191` [read]), so no other forward
can pass through installed hooks. I could not break it. Untried: more than 8 rows, attention edits under a cache,
prompts of unequal length.


## 12. The base row is never memoised. Stands, minor.

With six or more arithmetics by batch size a stored base row belongs to another arithmetic (the untouched row
against batch 1: mean KL 0.0006 to 0.016, pilot 5 [measured]), and the row costs 0.5 to 2% of a forward. But a base
row in the same forward does not make the difference independent of the batch (finding 2); the rule guards less than
its place in the design suggests.


## 13. The fall-back counter. Stands; the pilots did not read it. Minor.

The counter lives on the bound function (`backend/w4a16_fast.py:220-229`); `describe()` returns the install record,
which has no such key (`:320, 563-580` [read]). The first pilot asked `describe()` for it and wrote
`"fell_back": null`. No fall-back happened, by timing (0.0875 s a forward against 0.71 s on the reference). The
backend uses neither `inference_mode` nor `autocast` [read, by grep]. The harness should read
`engine_runtime.dequantize_w4a16_full.fell_back`, as `scripts/check_dequant.py:574` does.


## What the pilots did not measure that the proposal leans on

- The hook as proposed (float32, edited rows only); any `rows` edit on `gate_proj` or `up_proj`; any non-zero
  low-rank or attention term (those timed were all zeros); a set of 64 columns; a direction removal.
- Any floor or effect in log-odds, the default read-out.
- Delta against folded at one batch shape on a chat-formatted item, or at any layer but 44.
- Memory by shape; anything above 64 rows teacher-forced or 8 rows written; anything with the lens loaded or in the
  server.
- Where the batch-size classes change, and whether a fresh process, another card, or a card with less free memory
  lands in the same class as the reading it is meant to confirm.
- Restart exactness below layer 20, or with hooks installed.
- More than two chat-formatted texts for anything.


## What the proposal does not say at all

- That the instruct model gives raw text a mean log-probability of -7.6 to -9.6 nats a piece [measured], and that
  the atlas's passages, the corpus windows, the E6 frames and the pattern-pilot texts are all raw text. No item is
  gated on whether the base models it.
- That input-side scaling equals the folded weight, so the floor of the fast path is avoidable for exactly the
  single-neuron case where it bites.
- Which batch shapes a reading may use, though the shape is now part of the measurement.
- The memory of written batteries, or their size.
- Whether a mean over positions is the right statistic when a parted row has dozens of positions above a nat.


## Overall

The feature survives; the kernel as written does not. Three things must change before phase 0's pilots are
registered: the per-row form (input side for column scaling, a recomputed coordinate for row edits, output-side adds
only where nothing else serves); the reading (items in the model's own format gated on the base, several batch
shapes, medians, floors from several null rows in the metric's own units); and the definition of `remove_direction`
with its nulls. The cost constants hold to about 15%, but the fresh reading and the trial are under-costed by two-
to five-fold once batteries, the written gate and a stability repeat are counted, and the memory line rests on two
misread peaks. Pilot D as written, with raw stored passages, a single floor row and census-drawn nulls against an
undivided `u`, would not decide what the owner's first decision needs it to decide.
