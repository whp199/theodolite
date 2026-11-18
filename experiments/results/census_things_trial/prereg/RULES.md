# match_things at layer 30: what is run, how it is read, what it decides

Fixed on 18 November 2026 (the owner's word of 17 November: "You can go with design whose questions all take words up
the way half B's do"), before any answer of the design was written at any layer (SHA256SUMS beside this file;
`prereg_at`). Layers are counted from 0, as the trial and the board count them: 30 here is 31 on the pages.

**Why.** Two designs have failed their trials at layer 30, both on gate G3, which asks that a neuron's words picked on
one half of its questions come back on the other half. `match_wide` (one dose, twelve questions) failed on G3 and on its
lead: 0.579, and 0.555 and 0.550 for the words carrying. `match_both` (the same twelve questions at two doses) passed
its lead, 0.618, and failed on one figure: words picked on its second half came back on its first by 0.545, against
0.57 (experiments/results/census_wide_trials, census_both_trial).

Both trials were then read again on matched sets of questions (experiments/results/census_word_carry/NOTE.md;
exploratory, on data already read). A neuron's words at 30 carry as well between the halves as within one, so the
halves asking different things is not what stops them. What differs is the questions they come back on. The first
half's questions (who the model is, colours, writing as it likes) take a push's words up less than the second half's
(how a thing works, three tips, an object), whichever half the words were picked on: 0.533 against 0.554 on reads of
three questions and three, under `match_both`, where every first-half question ranked below every second-half question
on reads of two questions and one. So words picked on the second half came back on the first by 0.545, and words
picked on the first came back on the second by 0.573. At layer 31 the two halves took words up alike (0.556 and
0.553).

So this design asks questions of the second half's three genres alone. It was designed after both verdicts at 30 were
read, on the same layer, and its questions were chosen by how the earlier questions read there. Its neurons and
controls are drawn afresh, and half its questions have never been asked by any run, so that part of what it reads was
not chosen by how it reads at 30. What follows says beforehand that it is more likely to fail than to pass.

**The design, `match_things`** (backend/census.py at `repo_commit`):
- **The doses.** `match_both`'s: each neuron pushed by two counts of its own write ceiling, plus and minus, 2.42 and
  2.67 at layer 30. These are `match_ceil`'s, the pushes that move the next word-piece by 1.0 and 1.4 nats there
  (`MATCH_THINGS` is `MATCH_CEIL`). A control is pushed by the ceiling of the neuron it is paired with. One dose alone
  failed the lead at 30 (`match_wide`, 0.579), and nothing read since suggests other questions would lift it that far,
  so both doses are kept.
- **The questions.** Twelve, four in each of three genres, read as two halves of six:
  - Half A is `match_wide`'s half B, the six that took words up at 30: bicycle, walk, chair, kettle, sleep, lamp.
  - Half B is six asked by no run before: clock, rice, vase, lock, money, blanket. Clock and rice are spares fixed on
    16 November that no design took; vase, lock and blanket were written on 17 November before any run asked them,
    and money took suitcase's place at step 0 (below) (census.MORE_PROMPTS).
  - Each half holds two of each genre: how a thing works (bicycle, kettle | clock, lock), three tips (walk, sleep |
    rice, money), an object (chair, lamp | vase, blanket).
- **The answers.** 48 a unit, at 48 word-pieces, in batches of 64, read with every answer counted and each word set
  against the controls alone. A whole layer would be 1,130,508 answers, twice a layer under `match_wide`.

**Step 0, the questions' check, taken before this was hashed.** Twelve unpushed answers come first, then 48 answers a
unit. So each question's unpushed answer is written in a batch of 64 (the first nine) or 61 (the last three)
(`census.plan_batches` over the trial's job list). The rule is the one `match_wide` and `match_both` kept, written in
`step0.py` before the check ran (its sha256 and the time are in `atlas_data/census/trials/things/step0_rule.sha256`).
A question whose unpushed answer, in the batch its run writes it in, parts from itself written alone before 32
word-pieces gives its place to its genre's first spare that holds. Each genre has two spares: windmill and camera,
garden and interview, boot and mug. If a question parts and no spare of its genre holds, no trial is run under this
registration.

The check was `scripts/verify_atlas_writer.py --layer 30 --max-new 48 --batches 56,60,61,64 --unpacker compiled` on
the twelve questions and the six spares, on the home card (record
`atlas_data/census/trials/things/floor_match_things.json`, written 2026-11-18T00:09:25Z, sha256 5d8649f7fda3…; the
rule hashed at 2026-11-17T23:57:45Z, in `step0_rule.sha256` there). It was taken from the main checkout at cec0569
with the design's questions added to census.py, uncommitted; the check takes its questions by key and reads no
design. Its reading is `step0.txt`.

- Eleven held: bicycle, walk, chair, kettle, sleep, lamp, clock, rice and vase in their batch of 64 (chair 46 of 48
  word-pieces, as under `match_both`, the rest all 48), lock and blanket in their batch of 61.
- **suitcase parted** after 28 word-pieces, in its batch of 61 and in every other shape, and both spares of its genre
  parted there too (garden after 3, interview after 16). By that round's rule no trial could be run with those
  questions.

**A second round, for suitcase's place.** No neuron had been read; the check asks only whether a question's unpushed
answer is written the same in a batch as alone, which no push enters. So four more questions of the three-tips genre
were written into census.py, in an order fixed before the check asked them (money, language, winter, tidy), with a
rule of the same test in `step0_round2.py`, hashed at 2026-11-18T00:10:35Z (`step0_round2_rule.sha256`): the first
that holds in suitcase's batch of 61 takes its place; if none holds, no trial is run. The check on the four
(`floor_match_things_round2.json`, written 2026-11-18T00:13:18Z, sha256 c5fb9ac8b1c1…): money held 38 of 48
word-pieces in a batch of 61 and took the place (`step0_round2.txt`). Of the other three, language and winter hold
all 48 there; neither was needed. The docstring of `step0_round2.py` gives the first round's time as "17 November
2026, 18:09Z"; that is the local time, 00:09Z on 18 November, as the record says.

- **Reported, gating nothing:** in a trial, 56 of chair's pushed answers are written in a batch of 56, where chair
  parts from itself unpushed after 13 word-pieces (as under `match_both`). Money parts after 38 in batches of 61 and
  64 and holds all 48 in batches of 56 and 60.

**What is run.** At layer 30, `match_things` is run on 1,022 neurons drawn on 18 November (`neurons_L30.txt`, by
`draw_neurons.py`, seed 2026111730). The draw leaves out every neuron a census run had asked at layer 30: the 1,022
of `match_wide`'s trial, the 1,022 of `match_both`'s and the 64 of the two ladders of 28 October. They are read
against 1,024 controls drawn afresh (seed 10; the trials before used seeds 7, 8 and 9). That is 98,220 answers, about
2.5 to 3 hours. The trial runs from the clean checkout of the commit in `repo_commit`, by `night.sh` (`trial.sh` writes
and then reads), into `atlas_data/census/trials/things`, which no page reads as a layer's census. Layer 30 alone is
trialled: going below it needs the fine dose pilot taken there first, and the owner's word.

**How it is read.** By `things_read.py` (`read.sh`), on the CPU, over `trial_read.py` and `readlib.py`. Those two are
the first trial's of 29 October, byte for byte; their sha256 are the ones every registration since has held.
`things_read.py` is `match_both`'s `both_read.py` with one change: the two readings on three questions a half that it
reports are this design's halves' first three and last three. Two gates must both pass, each at 48 word-pieces, at
layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading.** G1 to G5 and G5b, every answer at both doses, on the halves of six.
- **N3, the neurons the controls cover.** G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range. It fails if fewer than half are covered.

As under `match_both`, each dose's answers are read alone as reports, gating nothing, and the text is guarded by G2 (the
lead beyond six covariates of how the text survives), by N3, and by what the text loses, reported. This is a weaker
standard than `match_ceil`'s three gates, and the design's words in census.py (`DOSES_FROM`) say so.

G4 asks that the new questions alone carry a lead: here that is half B, the six questions no run had asked.

**What has been seen that bears on it.**
- `match_both`'s trial at 30 (other neurons and controls), whole: 0.618, 0.640, 0.573 and 0.545, 0.608. Read on
  break_rel's six questions alone (three a half): 0.597, 0.549 and 0.542; on the six `match_wide` added: 0.616, 0.546
  and 0.551.
- The same trial read on its second half's six questions alone (this design's half A), three a half, both doses
  (exploratory, before this was hashed): bicycle, walk, chair | kettle, sleep, lamp: lead 0.608, words carrying 0.561
  and 0.554, the second three alone 0.599; bicycle, kettle, sleep | walk, chair, lamp: 0.608, 0.557 and 0.553, 0.582.
  That is about 0.01 more carrying than the mixed halves read at three a half. Going from three questions a half to
  six lifted `match_both`'s words carrying by about 0.012 on average (0.024 one way, 0.003 the other) and its lead by
  a little.
- Words carry between half A's questions and half B's as well as within either, at 30 and at every layer read
  (census_word_carry): the halves of this design being of the same genres is not expected to lift them by itself.
- `match_wide`'s trial at 30, on the same six questions as half A here, other neurons: 0.584 for them alone as its
  half B (one dose).
- The lead on all answers has fallen at each layer from 34 down: 0.758, 0.726, 0.699, 0.630 (31, `match_wide`), 0.579
  (30, `match_wide`); at 30 `match_both`'s two doses led by 0.618.
- Every figure of a trial has a standard error near 0.013 at 1,022 neurons.

**What is expected, said beforehand.**
- Layer 30 fails more likely than not: a pass about 1 in 4.
- If it fails, it fails on the words carrying (G3) first, one way or both under 0.57.
- N1's lead lies between 0.59 and 0.64, most likely near 0.615.
- The words carrying lie between 0.55 and 0.59 each, most likely near 0.57.
- The lead beyond text survival (G2) and the new questions alone (G4) pass.
- Reported: neither dose alone passes G1 to G4; the upper dose alone leads no more than the lower, more likely than not.
- 90 to 97 neurons in 100 are covered.
- The text loses under 0.005 of its distinct 4-grams at either dose.

**What it decides.**
- **It passes both gates:** layer 30 may be offered to volunteers under a kit that carries `match_things`. A whole
  layer is 1,130,508 answers, about 30 hours on a card like the home 5090. Before that kit, its other card class's bars
  need a bars registration of their own. `DOSES_CALIBRATED` is widened to it only on the owner's word.
- **It fails:** no whole layer is taken at 30 with `match_things`, and it is not trialled again under this
  registration. What failed is said as it is, and what to try at and below 30 is the owner's choice.

**Reported, gating nothing.** The same figures cut to 32 word-pieces and the length rule; the share covered; what the
text loses of its distinct 4-grams; how many neurons beat every control and how many are named; each dose's answers
alone; and two readings at three questions a half, both doses: bicycle, walk, chair | clock, rice, vase, and kettle,
sleep, lamp | lock, money, blanket.

**What a pass would not show.** That layer 30's neurons answer the census's other questions as they answer these: a
layer taken with this design would be asked none of who, orange or free, which every other layer is asked, so its
answers could not be set beside other layers' question by question but on bicycle, walk and chair. That one dose
reads alone at layer 30. That `match_things` reads where it was not trialled. Or anything below layer 30.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in `AMENDMENTS.txt`
beside this file, before a reading is made.
