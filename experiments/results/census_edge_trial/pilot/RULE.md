# Where the text breaks at layer 29: the ladder, and the rule that turns it into doses

Written on 22 November 2026, before any answer of this ladder existed (its sha256 and time are in the run's log
beside it, `pilot.log`). The owner's word was "Let's try both of those recommendations": a trial at layer 29 with
its upper dose at the layer's measured text edge, as a volunteer's Gemma 4 12B work suggested (issue #1, comment
7084903662, eigentactics' `tach_soft_06`). Layers are counted from 0, as the trials and the board count them: 29 here
is 30 on the pages.

**Why.** `match_things` failed its trial at 29 (18 November, experiments/results/census_things_walk): lead 0.574
against 0.60, words carried 0.528 and 0.520 against 0.57. Its doses there, 2.10 and 2.32 ceilings (0.45 and 0.49 of
the stream; 1.0 and 1.4 nats of the next word-piece), cost the text under 0.01 of its distinct 4-grams. The lead did
not rise with questions (0.57 and 0.58 at three a half), which at 37 meant the dose and not the answers. The old
coarse ladder at 30 (`L30-calib_ceil`, 28 October) kept 0.93 to 0.96 of the 4-grams at 4 ceilings and broke at 8.
So the pushes between 2.32 and the edge have not been tried at 29.

**The ladder** (`calib_edge`, backend/census.py): the twelve questions of `match_things` (bicycle, walk, chair, kettle,
sleep, lamp | clock, rice, vase, lock, money, blanket), sixteen rungs in each unit's write ceiling, 2.32 x 1.07^i for
i = 0..15 (2.32 to 6.40), plus and minus. 32 neurons and 32 isotropic controls drawn with seed 12 (no trial has used
it), 48 word-pieces, batches of 64, greedy: 24,576 pushed answers and the twelve unpushed ones, on the home RTX 5090,
into atlas_data/census/edge_pilot (no page reads it as a census).

```
census_run.py --layers 29 --design calib_edge --neurons random:32 --random-controls 32 --control-kind isotropic \
  --seed 12 --max-new 48 --batch 64 --unpacker compiled --out-dir atlas_data/census/edge_pilot
```

**How it is read.** `census_tools.py curves` on the run, which calls `census.dose_window` unchanged: for each sign
and for neurons and controls apart, the smallest rung at which the answers' mean share of distinct 4-grams falls more
than 0.1 (`BREAKS_BY`) below that of the unpushed answers to the same questions. That is the volunteer's rule (the
largest rung keeping 90% of the 4-grams) in this code's own words.

**The rule.** Let B be the smallest of the four first breaks (neurons up, neurons down, controls up, controls down).
- The upper dose is the rung just under B. The lower dose is the upper over the square root of two, to two decimals,
  as `soft_ceil` and the volunteer's `tach_soft_06` place their pair.
- If B is the first rung (2.32), no design is set: the edge would sit under `match_things`'s own doses, which its
  trial contradicts, and the ladder is looked into before anything else.
- If nothing breaks by 6.40, no design is set: a ladder reaching higher comes first.
- Nothing else moves the doses: not the "moves" column, not how the neurons lead on these 32, not the rungs' nats.

**What it decides.** Only the two doses of a design at layer 29, `edge_things`: `match_things`'s twelve questions and
halves at those doses, 48 answers a unit. Its trial is registered after this is read, on fresh neurons (none of these
32, none of the walk's 1,022 at 29), fresh controls, read by the walk's registered reader with N1 and N3 as gates, as
at 29 before. Whether a pass opens anything is the owner's word.

**What it cannot say.** Whether the neurons lead at those doses; 32 neurons are too few to read a lead. Where the edge
sits at 25 to 28, or anywhere but 29.
