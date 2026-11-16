# The walk downward with match_wide, layers 31 and 30: what is run, how it is read, what it decides

Fixed on 16 November 2026 (the owner's word: "One dose, same cost"), before any answer of the design was written at
any layer (SHA256SUMS beside this file; `prereg_at`).

**Why.** `match_ceil`'s walk downward passed at layers 36 to 32 and stopped at layer 31
(experiments/results/census_match_walk). It failed one gate of three there. Its lower dose's twelve answers a neuron
did not carry the neuron's words from one half of the six questions to the other (0.553 and 0.548 against 0.57),
while the lead held (0.634). Its two doses were 6% apart and read alike. The trial was later re-read on fewer
questions (experiments/results/census_question_count/NOTE.md), and those figures rose with each question a half
added. Projected to six questions a half, the lower dose alone passed every gate, the thinnest at 0.571 to 0.587.
So this design spends the second dose's answers on six more questions.

**The design, `match_wide`** (backend/census.py at `repo_commit`):
- **The dose.** Each neuron is pushed by one count of its own write ceiling, plus and minus. That count is
  `match_ceil`'s lower one at the layer, the push that moves the next word-piece by 1.0 nats (`MATCH_WIDE`, read from
  `MATCH_CEIL`): 2.58 ceilings at layer 31 and 2.42 at layer 30. A control is pushed by the ceiling of the neuron it
  is paired with.
- **The questions.** Twelve, read as two halves of six, after step 0 (below).
  - Half A: break_rel's first three (who, orange, free) and one more of each of their genres: "Tell me about
    yourself.", "Describe the color green.", "Tell me something interesting."
  - Half B: break_rel's other three (bicycle, walk, chair) and one more of each of theirs: "Explain how a kettle
    works.", "Give me three tips for sleeping well.", "Describe a brass lamp."
- **The answers.** 24 a unit, as before, at 48 word-pieces, in batches of 64, read with every answer counted and
  each word set against the controls alone.

**Step 0, the questions' check, taken before this was hashed.** The questions first chosen were "What can you
do?", "Describe the color blue." and "Tell me something interesting." for half A, and "Explain how a kettle
works.", "Give me three tips for sleeping well." and "Describe a stone bridge." for half B. The second three are
the spares of break_rel's genres fixed on 29 October. Each of the six genres had a spare: "Tell me about
yourself.", "Describe the color green.", "Write a few lines on any subject.", "Explain how a clock works.", "Give me
three tips for cooking rice." and "Describe a brass lamp.". All eighteen were written unpushed on the home card at
15:07Z to 15:17Z: alone, alone again, and in census batches of 60, 63 and 64. That was
`scripts/verify_atlas_writer.py --layer 31 --max-new 48 --batches 60,63,64 --unpacker compiled`; the record is
`atlas_data/census/trials/wide/floor_match_wide.json`, its sha256 in `step0.txt`. The rule was written in census.py
before the check: a question whose unpushed answer parts from itself written alone before 32 word-pieces gives its
place to its genre's spare. `step0.py` applies it to the record, and `step0.txt` is its output.
- **Where the rule is read.** It is read in the batch where the run writes the question's unpushed answer, from the
  design's own plan: 64 for half A, and 63 for half B, because the twelve unpushed answers come first. It reads that
  batch's two rows, the unpushed row and a neuron's push at dose 0. census.py's note first said "a census batch of
  64". The shapes 60 and 63 were added to the check before it ran, once the plan showed them.
- **Settled after results were seen.** Where to read the rule was settled after the first results were on screen. A
  stricter reading, any batch the check wrote, would also have failed orange, free and interesting in the batch of
  60 (after 5, 22 and 4 word-pieces). That batch holds 60 of each half-A question's 4,092 pushed answers in a trial
  and none of its unpushed ones; a whole layer has no batch of 60. Interesting's spare parts too (after 12, in a batch
  of 64), so under that reading no trial would be run. The reading taken is break_rel's practice: its step 0 read
  each question in its run's batch size, and its partial batches went unmeasured.
- **The outcome.** Three questions gave their places to their genres' spares, and each spare held all 48
  word-pieces in the same batch:
  - "What can you do?" parted after 27 word-pieces in its batch of 64, and gave its place to "Tell me about
    yourself."
  - "Describe the color blue." parted after 31, and gave its place to "Describe the color green."
  - "Describe a stone bridge." parted after 4 in its batch of 63, and gave its place to "Describe a brass lamp."
  - The other nine held all 48, or 46 for "Describe a wooden chair." in a batch of 63, as at 64 under break_rel.
- **Reported, gating nothing:** in a trial, 60 of each half-A question's pushed answers are written in a batch of
  60, where orange, free and interesting part from themselves unpushed before 32 word-pieces.

**What is run.** At layer 31, then at layer 30, `match_wide` is run on the layer's 1,022 neurons drawn on 16
November (`neurons_L<NN>.txt`, by `draw_neurons.py`). The draw leaves out every neuron a census run had asked at
that layer: at 31, the 1,022 of `match_ceil`'s trial; at 30, the 64 of the two ladders of 28 October. They are read
against 1,024 controls drawn afresh (seed 8; the walks before used seed 7). That is 49,116 answers, 70 to 90 minutes
a layer. The trial runs from the clean checkout of the commit in `repo_commit`, by `night.sh`, into
`atlas_data/census/trials/wide`, which no page reads as a layer's census.

**The order, and where it stops** (`walk.sh`). Layer 31 is read as soon as it is written, and layer 30 is taken only
if 31 passed both gates. The walk stops at the first layer that fails and at layer 30 at the latest, where
`MATCH_CEIL` ends. A layer that fails is not trialled again under this registration.

**How it is read.** By `wide_read.py` (`read.sh`), on the CPU, over `trial_read.py` and `readlib.py`. Those two are
the first trial's of 29 October, byte for byte; their sha256 are the ones every registration since has held. Two
gates must both pass, each at 48 word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading.** G1 to G5 and G5b, on the halves of six.
- **N3, the neurons the controls cover.** G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range. It fails if fewer than half are covered.

There is no N2. With one dose there are no lower-dose answers to read alone. That gate was there to show that a
lead was not carried by a dose that breaks the text. Here that is left to G2 (the lead beyond six covariates of how
the text survives) and to N3, as in every registration before. Under `match_ceil` the two doses were 6% to 10% apart, and
its registrations said that its N2 was close to its N1.

**What has been seen that bears on it.**
- At layer 31, `match_ceil`'s trial: whole 0.631, 0.634, 0.582 and 0.580, 0.618. The lower dose alone gave 0.634,
  0.632, 0.553 and 0.548, 0.587. The covered neurons, 971 of 1,022, gave 0.631, 0.628, 0.581 and 0.579, 0.616. The
  text was hardly broken: distinct 4-grams fell by under 0.005.
- The re-reading of that trial on fewer questions (NOTE.md), the lower dose alone at 1, 2 and 3 questions a half:
  rep_AB 0.515, 0.535, 0.553; rep_BA 0.525, 0.534, 0.548; half B alone (G4) 0.472, 0.541, 0.587. At six
  questions a half it projects 0.73 to 0.75 for the lead, 0.606 for rep_AB, 0.571 to 0.587 for rep_BA and 0.671 for
  half B alone. That projection assumes the six questions added carry what the six asked did. It was
  made on the data of the trial that failed, and it chose this design. The neurons and controls here are new.
- The same re-reading at layer 37 under `soft_ceil`, a failure that a harder dose later mended, was flat with
  questions (rep_BA 0.568, 0.538, 0.541). At 31 the figures rise.
- Every figure of a trial has a standard error near 0.013 at 1,022 neurons.

**What is expected, said beforehand.**
- Layer 31 is a little more likely to pass both gates than not: about 55 in 100.
- If it fails, it fails on words carrying between the halves (G3), most likely rep_BA, with the lead clear.
- N1's lead lies between 0.66 and 0.76. Its two figures for words carrying between the halves lie between 0.56 and
  0.62, the lesser of them nearest its bar. Half B alone (G4) lies between 0.61 and 0.70.
- Reported: read on break_rel's six questions alone (three a half, as `match_ceil`'s walk read them), this trial
  looks like `match_ceil`'s lower dose alone at 31. Its two figures for words carrying between the halves lie
  between 0.53 and 0.58, under 0.57 more likely than not. If they clear 0.57 by much, the new neurons or controls
  read better than the last trial's, and a pass is not the added questions' doing alone.
- 90 to 97 neurons in 100 are covered.
- If layer 31 passes, layer 30 is about as likely to pass as not.

**What it decides.**
- **A layer passes both:** it may be offered to volunteers under a kit of its own for this design, and the walk goes
  on to layer 30. `DOSES_CALIBRATED` is widened to a layer only on its passed trial and the owner's word. Its other
  card class's bars need a bars registration of their own first: kit 4's measured none for this design.
- **A layer fails:** no whole layer is taken there with this design, and the walk stops. What failed is said as it
  is. What to try there, or whether to go on below it, is the owner's choice, and would be registered anew on
  neurons this walk did not ask.
- **Going on below layer 30,** if both pass, needs the fine dose pilot taken there first, and the owner's word.

**Reported, gating nothing.** The same figures cut to 32 word-pieces and the length rule; the share covered; what
the text loses of its distinct 4-grams; how many neurons beat every control and how many are named; and two
readings at three questions a half, break_rel's six and the six added (`wide_read.py` writes them beside the
verdict).

**What a pass would not show.** That a second dose reads alone at this layer. That `match_wide` reads where it was
not trialled, at 32 to 37 included. That a neuron's words at twelve questions say the same as at six. Or anything
of layer 52 and above.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file, before a reading is made.
