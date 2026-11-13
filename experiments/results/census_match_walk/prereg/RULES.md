# The walk downward with match_ceil, layers 36 to 30: what is run, how it is read, what it decides

Fixed on 13 November 2026 (the owner's word: "Walk first"), before any answer of any of these layers was
written under any census design (SHA256SUMS beside this file; `prereg_at`).

**Why.** `match_ceil` (each neuron pushed by two counts of its own write ceiling that differ by layer, set where the
next word-piece moves by 1.0 and by 1.4 nats) passed its trial at layer 37 on all three gates, on neurons no trial had
asked (experiments/results/census_match_trials). By that registration it is the design for the layers below 38, a
trial before each. This is that walk, as far as layer 30, where the fine dose pilot's table ends.

**What is run.** At each layer in turn, 36, 35, 34, 33, 32, 31, 30: `match_ceil` on the layer's 1,022 neurons drawn on
12 November (`neurons_L<NN>.txt`; the same bytes as in the `break_ceil` and `soft_ceil` walk registrations, whose
SHA256SUMS they are in) against 1,024 controls (seed 7, as at layer 37), each control pushed by the ceiling of the
neuron it is paired with; six questions, 48 word-pieces, batch 64, the compiled unpacker; 49,110 answers, 75 to 88
minutes a layer. From the clean checkout of the commit in `repo_commit`, which is the commit layer 37's trial ran
from, by `night.sh`, into `atlas_data/census/trials/match_walk`, which no page reads as a layer's census.

**The doses.** As `MATCH_CEIL` holds them at that commit, from the fine dose pilot of 13 November (05:24Z) by the rule
fixed before it (experiments/results/census_scaled_design/MATCH.md). Nothing here changes them.

| layer | 36 | 35 | 34 | 33 | 32 | 31 | 30 | (37) |
|---|---|---|---|---|---|---|---|---|
| lower dose, ceilings (1.0 nats) | 2.27 | 2.27 | 2.62 | 2.31 | 2.21 | 2.58 | 2.42 | (2.41) |
| upper dose, ceilings (1.4 nats) | 2.46 | 2.41 | 2.79 | 2.51 | 2.44 | 2.76 | 2.67 | (2.55) |

**The order, and where it stops** (`walk.sh`). A layer is read as soon as it is written, and the next layer down is
taken only if this one passed all three gates. The walk stops at the first layer that fails, and at layer 30 at the
latest. Layers below a failed one are not trialled here, so nothing is said of them. A layer that fails is not
trialled again under this registration.

**What has been asked before.** No neuron of layers 31 to 36 has been asked anything under any design: the `soft_ceil`
walk that these lists were registered for stopped at layer 37 and took none of them. At layer 30 the ladder of 28
October asked 64 neurons at five doses; a few of them may be among this list's 1,022. No dose here was chosen on any
single neuron's answers.

**How it is read.** By `ceil_read.py` (`read.sh`), on the CPU. The reader and the two files under it are the earlier
registrations' own, byte for byte. The same three gates, all to pass, each at 48 word-pieces, at layer 45's bars
(0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b.
- **N2, the lower dose's answers alone**: G1 to G4, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half are covered.

**What has been seen that bears on it.**
- Layer 37's trial with this design (read at 06:45Z): whole 0.653, 0.678, 0.609 and 0.627, 0.654; the lower dose alone
  0.647, 0.672, 0.599 and 0.600, 0.628; the 857 of 1,022 covered neurons 0.629, 0.664, 0.585 and 0.606, 0.636. The
  upper dose alone led by 0.616, less than the lower.
- Going down from the band the whole lead has fallen a little at each layer that passed: 0.694 at 39, 0.675 at 38,
  0.653 at 37, each with its own design.
- The rule the doses come from is weaker than when it was made: at layer 52 the same movement of the next word-piece
  gave a lead 0.09 under layer 51's.
- The two doses of a layer are 6% to 10% apart, so N2 is close to N1 at every layer: this walk tests whether each
  layer reads at about one size of push.

**What is expected, said beforehand.**
- Layer 36 is about as likely to pass all three as not; if anything a little more likely.
- More likely than not the walk stops at layer 34 or above. That all seven layers pass is unlikely: perhaps one
  chance in twenty.
- Where a layer fails, the figures expected to fall first are the two for words carrying between the question halves,
  in N2 or N3 (0.599, 0.600 and 0.585 at layer 37, against 0.57), before the lead itself.
- N1's whole lead is between 0.61 and 0.68 at every layer reached, and falls a little from layer to layer.
- The upper dose's answers alone (reported) lead by less than the lower dose's at most layers reached, as at 37. If
  they lead by less than 0.60 at a layer that passes, 1.4 nats is too high a target there.
- Seven to nine neurons in ten are covered at every layer reached.
- If a layer fails on N1's lead with both across-the-halves figures clear, the doses and not the layer are suspect,
  and the pilot's measure is not the guide it was taken for.

**What it decides.**
- **A layer passes all three:** it may be offered to volunteers under a kit of its own for this design, and the walk
  goes on to the next layer down. `DOSES_CALIBRATED` is widened to a layer only on its passed trial and the owner's
  word.
- **A layer fails:** no whole layer is taken there with this design, and the walk stops. What failed is said as it is.
  What to try there, or whether to go on below it, is the owner's choice and would be registered anew on neurons this
  walk did not ask.
- **Going on below layer 30,** if all seven pass, wants the fine pilot taken there first, and the owner's word.

**Reported, gating nothing.** The same figures cut to 32 word-pieces; the share covered; what the text loses of its
distinct 4-grams at each dose; how many neurons beat every control; and the upper dose's answers read alone by
`trial_read.py` (`read.sh` writes `L<NN>_upper.json`).

**What a pass would not show.** That 1.0 and 1.4 nats are the best two targets for a layer; that a second, softer dose
reads alone there; that a layer not trialled reads; or anything of layer 52 and above.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
