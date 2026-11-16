# match_both at layer 30: what is run, how it is read, what it decides

Fixed on 16 November 2026 (the owner's word: "Two doses at 30 first"), before any answer of the design was written at
any layer (SHA256SUMS beside this file; `prereg_at`). Layers are counted from 0, as the trial and the board count
them: 30 here is 31 on the pages.

**Why.** `match_wide` passed its trial at layer 31 and failed at layer 30 on 16 November
(experiments/results/census_wide_trials). At 30 it failed two gates. Its lead was 0.579 against 0.60, and its words
carried between the halves by 0.555 and 0.550 against 0.57. The text was whole: distinct 4-grams fell by 0.002.

That trial was then read again on fewer questions (`qcurve.py` and `qcurve.json` beside this file; exploratory, on
data already read). The lead stopped rising at three questions a half: 0.520, 0.554, 0.584, 0.583, 0.577, 0.579 at one
to six. At layer 31 it levelled off the same way, near 0.63. So more questions would not lift the lead at 30. The
words carrying between the halves did keep rising with questions (0.529 to 0.556).

The walks before give one other lever. At layers 31 to 36, `match_ceil` asked each question at two doses 6% to 10%
apart. Its lead on both doses' answers beat its better dose alone by -0.003 to +0.016, 0.007 on average
(experiments/results/census_match_walk). So this design asks each of match_wide's twelve questions at both of
match_ceil's doses. That is 48 answers a neuron, twice the cost of a layer.

This trial was designed after match_wide's verdict at 30 was read, on the same layer. Its neurons and controls are
drawn afresh. What follows says beforehand that it is more likely to fail than to pass.

**The design, `match_both`** (backend/census.py at `repo_commit`):
- **The doses.** Each neuron is pushed by two counts of its own write ceiling, plus and minus: 2.42 and 2.67 at
  layer 30. These are `match_ceil`'s, the pushes that move the next word-piece by 1.0 and 1.4 nats there
  (`MATCH_BOTH` is `MATCH_CEIL`). A control is pushed by the ceiling of the neuron it is paired with.
- **The questions.** Twelve, read as two halves of six: match_wide's, as its step 0 fixed them. Half A is who, orange,
  free, self, green, interesting. Half B is bicycle, walk, chair, kettle, sleep, lamp. Step 0 here (below) reads them
  again in this design's batches.
- **The answers.** 48 a unit, at 48 word-pieces, in batches of 64, read with every answer counted and each word set
  against the controls alone.

**Step 0, the questions' check, taken before this was hashed.** Twelve unpushed answers come first, then 48 answers a
unit. So each question's unpushed answer is written in another batch shape than under match_wide: 64 for the first
nine questions, 61 for kettle, sleep and lamp (`census.plan_batches` over the trial's job list). The rule is
match_wide's, written in `step0.py` before the check ran (its sha256 and the time are in
`atlas_data/census/trials/both/step0_rule.sha256`). A question whose unpushed answer, in the batch its run writes it
in, parts from itself written alone before 32 word-pieces gives its place to its genre's spare. Three genres still have
a spare (lines, clock, rice). If a question of a genre with none left parts, or a spare parts too, no trial is run
under this registration.

The check was `scripts/verify_atlas_writer.py --layer 30 --max-new 48 --batches 56,60,61,64 --unpacker compiled` on
the twelve questions and the three spares, on the home card. Its record is
`atlas_data/census/trials/both/floor_match_both.json`, its sha256 in `step0.txt`.

All twelve held, so the questions are match_wide's:
- who, orange, free, self, green and interesting held all 48 word-pieces in their batch of 64, and so did bicycle and
  walk. Chair held 46 of 48 there, as in its batch of 63 under match_wide.
- Kettle, sleep and lamp held all 48 in their batch of 61.
- No spare took a place. The spare lines, which no question here needed, parts after 12 word-pieces in batches of 61
  and 64.
- **Reported, gating nothing:** in a trial, 56 of free's and 56 of chair's 8,184 pushed answers are written in a batch
  of 56. There those two questions part from themselves unpushed after 22 and 13 word-pieces. The batches of 60 and
  61 that hold kettle's, sleep's and lamp's other answers held all 48.

**What is run.** At layer 30, `match_both` is run on 1,022 neurons drawn on 16 November (`neurons_L30.txt`, by
`draw_neurons.py`, seed 2026111699). The draw leaves out every neuron a census run had asked at layer 30: the 1,022 of
match_wide's trial there and the 64 of the two ladders of 28 October. They are read against 1,024 controls drawn
afresh (seed 9; the walks before used seeds 7 and 8). That is 98,220 answers, about 2.5 to 3 hours. The trial runs
from the clean checkout of the commit in `repo_commit`, by `night.sh` (`trial.sh` writes and then reads), into
`atlas_data/census/trials/both`, which no page reads as a layer's census. Layer 30 alone is trialled: going below it
needs the fine dose pilot taken there first, and the owner's word.

**How it is read.** By `both_read.py` (`read.sh`), on the CPU, over `trial_read.py` and `readlib.py`. Those two are
the first trial's of 29 October, byte for byte; their sha256 are the ones every registration since has held. Two gates
must both pass, each at 48 word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading.** G1 to G5 and G5b, every answer at both doses, on the halves of six.
- **N3, the neurons the controls cover.** G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range. It fails if fewer than half are covered.

**Why there is no gate on the lower dose alone.** Under match_ceil a third gate, N2, read the lower dose's answers
alone. It was there to show that a lead was not carried by a harder dose that breaks the text. Here the lower dose's
answers alone are match_wide's design at layer 30, which has just failed there. A gate on them would make this trial
a second trial of match_wide. So each dose's answers are read alone as reports, gating nothing.

The text is guarded as in match_wide's registration: by G2 (the lead beyond six covariates of how the text survives),
by N3, and by what the text loses, reported. The two doses are 10% apart, and at 30 the lower one left the text whole.
This is a weaker standard than match_ceil's three gates. The design's words in census.py (`DOSES_FROM`), which
every card of it shows, say so.

**What has been seen that bears on it.**
- match_wide's trial at 30 (1.0 nats alone, 24 answers a neuron, other neurons and controls): whole 0.579, 0.611,
  0.555 and 0.550, 0.584. Covered 974 of 1,022: 0.582, 0.604, 0.557 and 0.551, 0.585. Text: distinct 4-grams down by
  0.001 to 0.002, 94 to 95 in 100 answers run to the cut. On break_rel's six questions alone it led by 0.559; on the
  six added, by 0.594.
- The same trial read on one to six questions a half (above): the lead flat from three; the words carrying rising.
- match_ceil at 31 to 36: both doses' lead against the better dose alone: 31 0.631 against 0.634, 32 0.699 against
  0.690, 33 0.726 against 0.710, 34 0.758 against 0.745, 35 0.664 against 0.667, 36 0.627 against 0.615. The words
  carrying between the halves gained 0.02 to 0.04 from the second dose at each layer.
- The harder dose alone led less than the lower at five of those six layers (at 32 it led more).
- A mock run tested the reader before this was hashed (`mock_read.py`, its output in `mock_read.txt`). It was made of
  match_wide's trial at 30 with every pushed answer written twice, once as if at the upper dose. It reads 0.586 for
  the lead and 0.559 and 0.554 for the words carrying, against 0.579, 0.555 and 0.550 for the same answers once. So
  answers that are only repeated lift those figures by about 0.007 and 0.004. A second dose has to carry more than
  that to count as more than a repeat. At 31 to 36 the second dose's gain on the lead was about that size, and its gain
  on the words carrying was five to ten times it.
- The lead on all answers has fallen at each layer from 34 down: 0.758, 0.726, 0.699, 0.630 (31, match_wide), 0.579
  (30, match_wide).
- Every figure of a trial has a standard error near 0.013 at 1,022 neurons.

**What is expected, said beforehand.**
- Layer 30 fails more likely than not: a pass about 1 in 8.
- If it fails, it fails on the lead (G1) first. N1's lead lies between 0.57 and 0.61, most likely near 0.585.
- The words carrying between the halves lie between 0.56 and 0.60, each.
- The lead beyond text survival (G2) and the new questions alone (G4) pass.
- Reported: the lower dose alone leads between 0.56 and 0.60, as match_wide did at 30. The upper dose alone leads no
  more than the lower more likely than not.
- 90 to 97 neurons in 100 are covered.
- The text loses under 0.005 of its distinct 4-grams at either dose.

**What it decides.**
- **It passes both gates:** layer 30 may be offered to volunteers under a kit that carries match_both. A whole layer
  is 1,130,508 answers, twice match_wide's, about 30 hours on a card like the home 5090. Before that kit, its other
  card class's bars need a bars registration of their own; kit 4 measured none for this design. `DOSES_CALIBRATED` is
  widened to it only on the owner's word.
- **It fails:** no whole layer is taken at 30 with match_both, and it is not trialled again under this registration.
  What failed is said as it is. Kit 5 is then prepared for layer 31 with match_wide alone, as the owner asked, and
  what to try at and below 30 is the owner's choice.

**Reported, gating nothing.** The same figures cut to 32 word-pieces and the length rule; the share covered; what the
text loses of its distinct 4-grams; how many neurons beat every control and how many are named; each dose's answers
alone; and two readings at three questions a half, break_rel's six and the six match_wide added, at both doses
(`both_read.py` writes them beside the verdict).

**What a pass would not show.** That one dose reads alone at layer 30. That `match_both` reads where it was not
trialled. That a neuron's words at 48 answers say the same as at 24. Or anything below layer 30.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in `AMENDMENTS.txt`
beside this file, before a reading is made.
