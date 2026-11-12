# `soft_ceil`: the census design with both doses inside what reads at layer 38

Designed on 12 November 2026, the evening `break_ceil`'s trial at layer 38 was read (the owner's word: "Yes, design
it"). Built and checked on the stand-in; nothing of it has been run on the model. Its trial is registered in
[`../census_soft_trials/prereg/`](../census_soft_trials/prereg/).

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

## Choices that are the owner's

1. Whether layers that pass are offered to volunteers. A new design is a new kit: its tag and its canaries.
2. Widening `DOSES_CALIBRATED` to a layer, on its passed trial.
3. Whether the same design is trialled above the band, at layer 52, once that layer's `break_ceil` trial is read.
