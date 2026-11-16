# The walk upward with soft_ceil, layers 53 to 59: what is run, how it is read, what it decides

Fixed on 16 November 2026 (the owner's word that day: "continue the census walk in hopes of opening more layers for our
volunteers"), before any answer of any of these layers was written under any census design save the few named below
(SHA256SUMS beside this file; `prereg_at`).

**Why.** `soft_ceil` (each neuron pushed by +-1.6 and +-2.3 of its own write ceiling) passed its trial at layer 52 on
all three gates, on 1,022 neurons no trial had asked (experiments/results/census_soft_trials, `RULES_L52.md`):
the whole reading 0.804, 0.693, 0.680 and 0.669, 0.758; the neurons the controls cover (945 of 1,022) 0.797, 0.684,
0.670 and 0.658, 0.750. Its lower dose alone passed thinly, and thinnest on the new questions: 0.633, 0.657, 0.604
and 0.608, and 0.577 for the new questions alone against a bar of 0.56 (`L52_verdict.txt`). Kit 4 already names
layers 53 to 59 with `soft_ceil` (`KIT_DESIGNS` in volunteers/kit.env), and their home canaries under it are taken, but
they are shut on the board: none has been trialled with these doses. This is that trial, a layer at a time, upward from
52 for as long as each layer passes.

**What is run.** At each layer in turn, 53, 54, 55, 56, 57, 58, 59: `soft_ceil` on the layer's 1,022 neurons
(`neurons_L<NN>.txt`) against 1,024 controls (seed 6, as at layer 52, the trial this walk goes on from), each neuron
pushed by -2.3, -1.6, +1.6 and +2.3 of its own write ceiling and each control by the ceiling of the neuron it is
paired with; six questions (who, orange, free | bicycle, walk, chair), 48 word-pieces, batch 64, the compiled
unpacker; 49,110 answers a layer, as every trial since 12 November wrote. The three `soft_ceil` trials took 81, 87 and
85 minutes (layers 38, 37 and 52), so about an hour and a half a layer with the model's load and the reading. By
`night.sh`, into `atlas_data/census/trials/soft_walk_up`, which no page reads as a layer's census.

**The commit it runs from** (`repo_commit`): 81c1362, tagged `census-kit-4`, the code a volunteer runs, from its own
clean worktree (.claude/worktrees/kit-4), whose HEAD nothing here moves. Checked when this was fixed: the worktree is at
that commit with nothing uncommitted; its `backend/census.py` holds `SOFT_CEIL = (1.6, 2.3)` and `soft_ceil`'s four
doses in each neuron's write ceiling. From b6f7b54, the commit the `soft_ceil` trials at 38, 37 and 52 ran from, to
81c1362 nothing that writes or ranks a `soft_ceil` answer changed: `census.py` gained `match_ceil`, the table of
calibrated layers and the words cards show, and kit 4's loader puts the vision tower on the CPU, with which every home
canary of kit 3 was taken again and was the same (kit.env). Layer 36 was taken whole from this commit on the home card
on 14 November. The census functions the reader takes from the checkout are the same as at the match walk's commit.

**The doses.** As `SOFT_CEIL` holds them at that commit: the same two counts of each neuron's own write ceiling at
every layer, as at 38, 37 and 52. Nothing here changes them. How far each moves the next word-piece, by the steer
lens's dose pilot (experiments/results/atlas_steer_dose/pilot.json: the median over 128 of the layer's neurons of the
KL from the unpushed next word-piece), and what one push of the stream's median length comes to in ceilings:

| layer | (52) | 53 | 54 | 55 | 56 | 57 | 58 | 59 |
|---|---|---|---|---|---|---|---|---|
| +-1.6 ceilings, nats | (0.58) | 0.37 | 0.38 | 0.73 | 0.75 | 1.65 | 3.90 | 12.0 |
| +-2.3 ceilings, nats | (2.39) | 1.53 | 1.99 | 2.99 | 2.80 | 4.75 | 11.0 | 15.6 |
| one stream, in ceilings | (1.06) | 1.63 | 1.77 | 1.71 | 1.48 | 1.21 | 1.25 | 1.66 |

At 53 to 56 the lower dose is 0.90 to 1.08 of the stream, about `break_rel`'s lower dose; the upper is 1.30 to 1.56,
under `break_rel`'s upper.

**The order, and where it stops** (`walk.sh`). Upward only. A layer is read as soon as it is written, and the next
layer up is taken only if this one passed all three gates. The walk stops at the first layer that fails, and at layer
59 at the latest. Layers above a failed one are not trialled here, so nothing is said of them. A trial that stops
before its last answer (a fault of the card, the driver or the process) is begun once more after 30 seconds and goes
on row for row from where it stopped; if it stops a second time the walk stops with no verdict at that layer, and
`walk.sh` run again goes on with the same layer. A layer that has been read is not trialled again under this
registration. Layers 31 and below want a new design and are no part of this.

**What has been asked before.**
- **Layers 53 to 56.** Their lists were drawn and hashed on 12 November 2026 (15:14Z) for `break_rel`'s walk outward
  (experiments/results/census_neighbour_trials/prereg; the same bytes here, the same sha256 as in its SHA256SUMS). That
  walk took 39, 51, 38 and 52 and stopped on both sides at 38 and 52 (its `summary.json`; its log ends "down: stop",
  "up: stop"); no run of any layer from 53 to 56 is in its folder, and no census run of any of them is in
  atlas_data/census/runs. None of their neurons has been asked anything under any design, save the home canaries below.
- **Layers 57 to 59.** No list existed. Drawn on 16 November with the same `draw_neurons.py` (the same bytes as in that
  registration) by the same rule: 1,022 of the layer's 21,504, without replacement, sorted, from a seed fixed in the
  script on 12 November, 20261112 x 100 + the layer (2026111257, 2026111258, 2026111259). No seed is chosen here: it
  is the one that walk's `walk.sh` would have used to draw a layer it reached, so each list is the one that day's rule
  gives. Called as that registration calls it, `.venv/bin/python -I draw_neurons.py 57 > neurons_L57.txt`. Drawn again
  the same way today, the lists for 53 to 56 come out byte for byte as hashed on 12 November (numpy 2.5.4, CPython
  3.12.15), so the script draws as it did.
- **Layer 57 was asked before.** The `break_rel` probe of 29 October (atlas_data/census/trials/L57-break_rel-probe.*,
  and experiments/results/census_break_rel_trial) asked 507 neurons against 512 controls (seed 3) at +-1 and +-2 of the
  stream, 1.21 and 2.41 ceilings there. 28 of its 507 are in the new list (80, 110, 1267, 2124, 2646, 3036, 3579, 3665,
  4033, 4218, 4593, 4827, 4841, 6319, 6377, 6510, 7222, 8387, 9120, 9521, 9558, 11045, 12152, 12509, 12616, 16794,
  17017, 21451), about the 24 that chance gives. The ladders of 28 October (`L57-calib_rel` and `L57-calib_ceil`, the
  same 64 neurons) put 4 more in it (2512, 6024, 6935, 7078; chance gives 3).
- **The home canaries.** At each of 53 to 59 two home canaries wrote a whole run's first batch, kit 2's under
  `break_rel` (12 November) and kit 3's under `soft_ceil` (14 November; taken again with kit 4's code, the same): the
  unpushed answer, neurons 0 to 14 (59 answers: 14 at three of the four doses) and one control at its four doses, all
  on "who", 64 answers, kept to be set bit for bit against another card's, with no figure read off them. Neuron 9 of
  layer 57 and neuron 13 of layer 58 are in the new lists; no other.
- **The steer lens** (built 13 November, all sixty layers) has a reading for every neuron of these layers, the next
  word-piece under one push, and its dose pilot pushed 128 neurons a layer (seed 0) along a ladder of doses. Those are
  next-word-piece readings, not census answers. The table above is the pilot's medians.

The lists are kept as drawn: soft_ceil's doses were set at layer 38 and not on any of these neurons' answers, and the
33 asked before at layer 57 were drawn by the seed, not picked by their answers, and are too few of 1,022 to move any
figure by more than about 0.03 (33 / 1,022) even were each of them to read its opposite.

**How it is read.** By `ceil_read.py` (`read.sh`), on the CPU, at nice 15, with the census functions of the commit run
from. The reader and the two files under it are the earlier registrations' own, byte for byte (the same sha256 in the
`soft_ceil` trials' two folders, the `soft_ceil` walk's and the match walk's). The same three gates, all to pass,
each at 48 word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b.
- **N2, the lower dose's answers alone** (+-1.6 ceilings): G1 to G4, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half are covered.

**What has been seen that bears on it.**
- `soft_ceil` passed at 38 (whole 0.675, the lower dose alone 0.654, 839 of 1,022 covered) and at 52 (above), and
  failed at 37 on words carrying between the question halves, in the lower dose alone (0.583 and 0.542) and among the
  covered neurons (0.602 and 0.568). At 52 the upper dose's answers alone (reported) led by 0.783, and at that dose the
  neurons' text lost 0.60 of its distinct 4-grams, the controls' 0.72.
- `break_rel` failed at 52 on its lower dose alone (0.603, 0.592, 0.546 and 0.577, 0.554): there one push of the
  stream is 1.06 write ceilings, barely one, and moves the next word-piece 0.18 nats. `break_ceil` failed at 52 on the
  whole reading's lead beyond text survival (0.553), because its +-4.6 answers (11.7 nats) read at chance (0.485).
- The `break_rel` probe at 57, read on 29 October by the looser bars written for it, passed at 48 word-pieces by
  0.0008 on its lead beyond text survival and failed at 32; by these three gates it fails all three (354 of 507
  covered). Its +-1 answers alone led by 0.5225 and its +-2 answers by 0.5737 (the review of the first trial,
  census_break_rel_trial/review/dose/q1_dose_split.json). Those pushes, 1.21 and 2.41 ceilings, bracket these at 57
  (1.6 and 2.3), and the first moves the next word-piece 0.60 nats by the pilot.
- Of the 32 doses read alone so far (pilot.json, `read_alone`), the 25 that moved the next word-piece by 0.35 to 2.4
  nats, at layers 31 to 52, all led by 0.60 or more; the two under that span (0.18 and 0.27 nats) led by 0.603 and
  0.633 and failed on words carrying between the halves (at 52 on the new questions alone too); the five over 3 nats
  did not reach 0.60. Nothing has been read between 2.4 and 3.2 nats. Layer 57's +-1 (0.60 nats, 0.5225), not in that
  list, is the one reading inside the span that did not lead.
- G5's read gain (how loudly the next layer reads the controls over the neurons, the band 0.8 to 1.25, in N1 and N2), a
  figure of the weights and the seed, not of any answer: the `soft_ceil` home canaries (seed 7) recorded 1.067, 1.038,
  1.069, 1.069, 1.181, 1.250 and 1.314 at 53 to 59, and 1.191 at 52, where the trial (seed 6) read 1.193. (Added by
  the check of this registration on 16 November, from the canaries' metas, before any answer of the walk existed.)

**What is expected, said beforehand.**
- **Layer 53** is about as likely to pass as not. N1 passes, with a whole lead between 0.72 and 0.82; N3 passes, with
  80% to 95% of the neurons covered. N2 is the open gate: the lower dose moves the next word-piece 0.37 nats, less
  than at 52 (0.58), where it passed with 0.017 to spare, and about as at 38 (0.35), where it passed. If N2 fails, the
  figure for the new questions alone and the two for words carrying between the halves fall first, before the lead.
- **Layer 54**, if reached, is a little less likely to pass than not, for the same reason (0.38 nats).
- **Layers 55 and 56**, if reached: the lower dose is stronger (0.73 and 0.75 nats) and N2 more likely to pass, but
  the upper dose moves the next word-piece 3.0 and 2.8 nats, past anything that has led. The figure expected to fall
  first is the lead beyond text survival, in N1 and N3, as under `break_ceil` at 52. Each is less likely to pass than
  not: about two chances in five.
- **Layer 57**, if reached, fails: one chance in twenty that it passes. The probe's two pushes there bracket these,
  and neither led. N2's lead itself is expected under 0.60.
- **Layers 58 and 59** would fail if reached: even the lower dose moves the next word-piece 3.9 and 12 nats. At 59 G5
  is expected to fail on the read gain alone (about 1.31, over 1.25), whatever the lead; at 58 it sits on the edge.
- So the walk stops at 53 about one time in two and at 54 about one in four; it passes 55 about one time in ten, and
  57 about one in five hundred (the odds above multiplied: 0.5 x 0.45 x 0.4 x 0.4 x 0.05). About one layer is
  expected to pass, more likely none or one than two or more.
- The upper dose's answers alone (reported) lead by 0.72 to 0.82 at 53 and 54, and by less than 0.65 at 55 and 56.
  At 53 and 54 the neurons' text loses less of its distinct 4-grams at the upper dose than at 52 (0.60).
- If layer 53 fails N1 with both doses inside the span, the pilot's span is no guide above layer 52. If 55 or 56
  passes with its upper dose alone leading by 0.65 or more, the span reaches about 3 nats there.

**What it decides.**
- **A layer passes all three:** the walk goes on to the next layer up, and the layer may be opened on the board for
  volunteers under kit 4, which already names it with `soft_ceil`, but only after two things. First, its bars for
  another card class are measured. The kit 4 bars registration (experiments/results/census_other_card_bars/kit4/prereg)
  measured layers 32 to 39, 51 and 52 alone. It says layers 53 to 59 "are not measured, and no run of another card
  class is judged there until they are". Its `measure.sh` names its ten layers, and its R1 pools `soft_ceil`'s 38 and
  52 into one group, whose bars are in use. So a passed layer from 53 is measured under a registration of its own,
  fixed before it runs: the same thirteen variants and the rule of 31 October, decided as R1 to R6 decide, saying
  whether the new layer is pooled with 38 and 52 (which would move the bars in use there) or read alone as R3 reads a
  layer, and writing beside kit4/bars.json, not over it. Second, the owner's word. A whole layer at the home card is
  also the owner's word, and is read by experiments/results/census_scaled_design/read_whole.sh with `PREREG` set to
  this folder's copy beside the runs, as that script says for 53 to 59. `DOSES_CALIBRATED` is widened to a layer only
  on its passed trial and the owner's word.
- **A layer fails:** no whole layer is taken there with this design, the walk stops, and the board may mark it
  stopped. What failed is said as it is. What to try there (`match_ceil` wants the fine dose pilot taken at those
  layers first: `MATCH_CEIL` holds 30 to 38 alone), or whether to try above it, is the owner's choice. Either would be
  registered anew on neurons this walk did not ask.

**Reported, gating nothing.** The same figures cut to 32 word-pieces; the share covered; what the text loses of its
distinct 4-grams at each dose; how many neurons beat every control; the +-2.3 answers read alone by `trial_read.py`
(`read.sh` writes `L<NN>_upper.json`); and each dose's lead set beside the nats in the table above.

**What a pass would not show.** That 1.6 and 2.3 are the best two doses for a layer; that a layer not trialled reads;
that a lead names what a neuron does; or anything of layers 52 and below.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
