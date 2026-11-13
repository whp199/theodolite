# `soft_ceil`: the census design with both doses inside what reads at layer 38

Designed on 12 November 2026, the evening `break_ceil`'s trial at layer 38 was read (the owner's word: "Yes, design
it"). Built and checked on the stand-in, then trialled at layer 38 that night: **it passed all three gates**
(the last section but one). The trial's registration is in
[`../census_soft_trials/prereg/`](../census_soft_trials/prereg/) and its verdict beside it.

## Why another design

`break_ceil` (±2.3 and ±4.6 of each neuron's write ceiling) failed at layer 38 by 0.001 on its whole reading and
passed on its lower dose alone ([`DESIGN.md`](DESIGN.md), the last section). Its upper dose read at chance there:
half the answers carried nothing and held the whole reading at the bar.

## What is known about a push of each size

A lead is how often a neuron's push is told from a random direction's by the words it brings; 0.5 is chance, and
the bar on it is 0.60. Each row is one pair of doses read alone (`../census_ceil_trials/by_dose.json`, and the
neighbour trials' verdicts).

| layer | push, in ceilings | lead | from |
|---|---|---|---|
| 52 | 1.06 | 0.603 | `break_rel` ±1, registered gate (failed on its other figures) |
| 51 | 1.76 | 0.723 | `break_rel` ±1, registered gate |
| 39 | 2.14 | 0.690 | `break_rel` ±1, registered gate |
| 38 | 2.30 | 0.682 | `break_ceil` ±2.3, registered gate |
| 45 | 2.54 | 0.706 | `break_rel` ±1, the first trial's review |
| 38 | 3.14 | 0.496 | `break_rel` ±1, registered gate |
| 39 | 4.28 | 0.595 | `break_rel` ±2, looked at afterwards |
| 38 | 4.60 | 0.506 | `break_ceil` ±4.6, looked at afterwards |

Every push between 1.76 and 2.54 ceilings leads by 0.68 or more, at the four layers where one was read alone. At
layer 38 nothing above three ceilings leads at all.

Below 38 there is only the ladder of 28 October (`../atlas_calibration/subset_table.json`: 64 neurons and 64
controls a layer, one question set, so each figure is good to about ±0.05):

| layer | ±0.5 | ±1 | ±2 | ±4 | ±8 ceilings |
|---|---|---|---|---|---|
| 40 | 0.526 | 0.619 | 0.622 | 0.658 | 0.487 |
| 30 | 0.540 | 0.506 | 0.604 | 0.416 | 0.648 (text wrecked) |

At layer 30 a push of two ceilings leads and a push of one or of four does not. That is thin, and it agrees with
layer 38: the room is near two ceilings and narrow.

## The design

`break_ceil` with its two doses changed to **±1.6 and ±2.3 of each neuron's own write ceiling**. Nothing else
differs: the six questions and their halves, 48 word-pieces, the controls (a random direction pushed by the ceiling
of the neuron it is paired with), the reading (every answer counted, each word against the controls) and the three
gates.

- **2.3 is kept**, now as the upper dose. It is the push that read at layer 38, so the trial repeats that finding
  on new neurons.
- **1.6 is half a doubling under it** (2.3 / √2). It lies just under the softest push that has passed as a lower
  dose (1.76, layer 51), and above the one ceiling that read at chance at layer 30. A lower dose of 1.15, half of
  2.3, was not taken: 1.06 failed alone at layer 52 and one ceiling read at chance at layer 30.

As shares of the stream this is 0.51 and 0.73 at layer 38, 0.41 and 0.60 at 35 to 37, 0.33 and 0.48 at 30, and
0.15 and 0.21 at 20.

## What it may not do

- **The lower dose may be too soft to carry a lead alone.** That is the gate on the lower dose's answers, and it
  is the open question of the trial.
- **The room may narrow further below 38.** It closed from above between layers 39 and 38 (at 39 a push of 4.3
  ceilings still led by 0.595). If it keeps closing, 2.3 is too hard somewhere below. Each layer has its own trial
  for that reason.
- **Ceilings may be the wrong measure.** Neither a share of the stream nor a count of ceilings puts the room at one
  dose across depth. A third measure comes free tonight: the steer lens's dose pilot gives, at every layer, how far
  a push of a given size moves the next word-piece. A design whose dose at each layer is matched on that would be
  the next thing to try if this one fails. It is a larger change, and it is not built.

## Not taken

- **One dose only, ±2.3, on twelve questions.** It needs six new questions, which changes more than the dose.
- **Three doses.** A third more answers a layer for a dose already known to read at chance or not at all.

## Cost

The trial: 49,110 answers, about 85 minutes on the home card. If it passes, a trial before each layer going down,
and a whole layer (13 hours) for each that passes.

## What the trial at layer 38 said

Read at 00:52Z on 13 November 2026: 1,022 neurons no trial had asked, 1,024 fresh controls, 49,110 answers in 81
minutes. **It passes all three gates.** The bars are layer 45's.

| reading | lead (0.60) | beyond text survival (0.58) | across the halves (0.57, 0.57) | new questions (0.56) | |
|---|---|---|---|---|---|
| whole, all four doses | 0.675 | 0.712 | 0.636, 0.630 | 0.643 | pass |
| the lower dose alone, ±1.6 | 0.654 | 0.671 | 0.594, 0.605 | 0.623 | pass |
| the neurons the controls cover, 839 of 1,022 | 0.628 | 0.673 | 0.589, 0.583 | 0.598 | pass |
| *reported:* the upper dose alone, ±2.3 | 0.655 | 0.706 | 0.647, 0.631 | 0.673 | gates nothing |

- **Against what was said beforehand.** Met: the whole lead (said 0.64 to 0.70), the lower dose's lead, which was
  the open question (said 0.60 to 0.66), and the ±2.3 answers alone on new neurons (said 0.64 to 0.72; 0.682 on
  the first neurons, 0.655 on these). Missed: nine neurons in ten covered or more. It is eight in ten.
- **Why fewer are covered.** At these doses a random direction hardly moves the text: it loses 0.004 of its
  distinct 4-grams at ±2.3 and 0.001 at ±1.6. A neuron moves it more (0.03 and 0.005). So the controls' range is
  narrow and 183 neurons fall outside it, most on the text's likelihood (120) and its distinct 4-grams (107). The
  lead among the covered is lower than the whole lead for that reason.
- **The thinnest figure** is the covered neurons' second replication figure, 0.583 against 0.57. Everything else
  clears its bar by 0.019 or more.
- **Reported.** 110 neurons beat every control, where chance gives one. Cut to 32 word-pieces the whole reading
  passes as well (lead 0.687).

By the registration this is the design for the walk downward: a trial before every whole layer (37, 36, …: 1,022
neurons a layer were drawn in the `break_ceil` registration and none has been asked). The pass does not show that
any layer below 38 reads with these doses, nor that layer 52 does.

## Choices that are the owner's

1. Whether layers that pass are offered to volunteers. A new design is a new kit: its tag and its canaries.
2. Widening `DOSES_CALIBRATED` to a layer, on its passed trial.
3. Whether the same design is trialled above the band, at layer 52. Its `break_ceil` trial failed on the lead
   beyond text survival with its ±2.3 answers alone passing ([`DESIGN.md`](DESIGN.md), the last section).
4. Whether layer 38 is taken whole with this design (13 hours) before the walk goes further down.
