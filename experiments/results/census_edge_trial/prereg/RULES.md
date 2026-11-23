# The edge_things trial at layer 29: what is run, how it is read, what it decides

Fixed on 23 November 2026, before any answer of `edge_things` was written (SHA256SUMS beside this file; `prereg_at`).
The owner's word was "Let's try both of those recommendations" (22 November), to a trial at 29 with its upper dose at
the layer's measured text edge. Layers are counted from 0, as the trials and the board count them: 29 here is 30 on
the pages.

**Why.** `match_things` failed its trial at 29 on 18 November (experiments/results/census_things_walk): lead 0.574
against 0.60, words carried between the question halves 0.528 and 0.520 against 0.57. Its doses, 2.10 and 2.32
ceilings (1.0 and 1.4 nats of the next word-piece, 0.45 and 0.49 of the stream), left the text whole (under 0.01 of
its distinct 4-grams lost), and the lead did not rise with questions (0.57 and 0.58 at three a half), which at 37 had
meant the dose and not the answers. A volunteer's Gemma 4 12B trials (board issue #1, comment 7084903662) found the
lead returned at their layer 24 once the upper dose sat at the layer's text edge (largest push keeping 90% of the
distinct 4-grams) and the lower half a doubling under it; their trial still failed on the words carrying.

**The doses** (`edge_things`, backend/census.py at `repo_commit`): set by the `calib_edge` ladder at 29
(experiments/results/census_edge_trial/pilot: RULE.md hashed 02:31:05Z, ladder 02:31-03:14Z, READ.md), by its rule
and nothing else. The text first broke at 3.99 ceilings (controls pushed up; neurons and controls pushed down), so:
- **upper: ±3.73 ceilings**, the rung under it, about 0.81 of the stream; the ladder's text kept 0.93 of its 4-grams;
- **lower: ±2.64 ceilings**, 3.73 over the square root of two, about 0.57 of the stream.

By the fine dose pilot's curve at 29 (other neurons) these move the next word-piece about 1.8 and 4.4 nats.
A control is pushed by the ceiling of the neuron it is paired with, as in every ceiling design.

**The questions and answers** are `match_things`'s: twelve, read as two halves of six (bicycle, walk, chair, kettle,
sleep, lamp | clock, rice, vase, lock, money, blanket), 48 answers a unit, 48 word-pieces, batches of 64. **Step 0 is
not taken again**, as the walk's RULES.md said for its layers: no push enters an unpushed answer, and the batches are
the same shapes (`census.plan_batches` over job lists of the same lengths), so 30's check stands.

**What is run** (`night.sh`): at layer 29, 1,022 neurons drawn by `draw_neurons.py` (seed 2026112229) from the 20,453
no census run had asked there: the walk's 1,022 and the ladder's 32 left out (`neurons_L29.txt`). 1,024 controls
drawn afresh with seed 13, which no run has used (trials 3 to 11, the ladder 12). 98,220 answers, about 3 hours on the
home RTX 5090, written from a clean checkout of `repo_commit` into `atlas_data/census/trials/edge`, which no page reads
as a layer's census. `repo_commit` differs from `census-kit-7` in nothing that writes an answer but `calib_edge`,
`EDGE_LADDER` and `edge_things`. The trial may be paused between batches for one GPU process at a time and is then
continued row for row by `night.sh`; a stop on a fault is run again once.

**How it is read** (`read.sh`): by `things_read.py`, byte for byte the reader of `match_things`'s trial at 30 and of
its walk, over `trial_read.py` and `readlib.py`, the first trial's of 29 October byte for byte. Read on the walk's
layer-29 answers before this was hashed, it gives that verdict back to the last figure (`reader_test.txt`). All
gates at 48 word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b, every answer at both doses, on the halves of six.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside the
  controls' range; fails if fewer than half are covered.

**The layer passes when N1 and N3 both pass.** Reported and gating nothing: each dose alone, three questions a half,
the cut to 32 word-pieces, text loss, neurons beating every control and q ≤ 0.05.

**What is expected, said beforehand.**
- **It passes about 1 in 4.** If it fails, it fails on the words carrying between the halves (G3) first, as at 29
  before and as the 12B's trial did.
- **N1's lead between 0.56 and 0.66.**
- **The text** loses 0.05 to 0.09 of the neurons' distinct 4-grams at ±3.73 and under 0.02 at ±2.64, as the ladder
  read; most answers still run to the 48-piece cut.
- **The share covered**: 85 to 98 in 100.
- **The controls** that stand out most write other languages, as at 29 before (Portuguese, Indonesian), and stand
  further out than they did there (6.7 to 7.6 standard deviations).
- **Each dose alone**: neither passes G3 alone.

**What it decides.**
- **A pass**: `edge_things` passed N1 and N3 at 29. Nothing opens on that alone: a kit carrying it needs other-card
  bars of its own, and a kit and `DOSES_CALIBRATED` are the owner's word. Whether to walk it on down (28 to 25, each
  with a ladder of its own first) is the owner's word, under a registration of its own.
- **A fail**: no whole layer is taken at 29 with `edge_things`. What to try next is the owner's.

**What a pass would not show**: that these neurons answer the census's other questions as they answer these; what a
neuron does at the doses `match_things` uses at 30 to 54; or where the text edge sits at any layer but 29.
