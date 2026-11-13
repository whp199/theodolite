# `match_ceil`: the census design whose two doses are set a layer at a time

Designed on 13 November 2026, the night the walk downward with `soft_ceil` stopped at layer 37 (the owner's word:
"Yes, design it"). Built and checked on the stand-in. This note was written, and the rule below fixed, before the
fine dose pilot it rests on was taken and before any answer of the design existed.

## The design

`soft_ceil` with its two fixed counts of ceilings replaced by two counts that differ by layer. At each layer the
two doses are the pushes that move the next word-piece by **1.0 nats and by 1.4 nats**, each neuron pushed by that
count of its own write ceiling. Nothing else differs: the six questions and their halves, 48 word-pieces, the
controls (a random direction pushed by the ceiling of the neuron it is paired with), the reading (every answer
counted, each word against the controls) and the three gates.

## Why a count of ceilings is not enough

`soft_ceil` (±1.6 and ±2.3 ceilings) passed at layer 38 and failed at layer 37 ([`SOFT.md`](SOFT.md)). At 37 its
lower dose alone did not carry between the question halves (0.542 against 0.57), and its upper dose alone passed
every gate. The steer lens's dose pilot says why: the same 1.6 ceilings moves the next word-piece less at 37 than
at 38 (0.27 nats against 0.35, read off its ladder). Across the sixteen doses a trial has read alone, how far the
push moves the next word-piece sorts the leads where neither a share of the stream nor a count of ceilings does
([`../atlas_steer_dose/pilot.json`](../atlas_steer_dose/pilot.json)).

The curve is steep: near these doses a push 7% larger moves the next word-piece about 30% further. So a tenth of a
ceiling matters, and the count that gives a fixed movement differs by layer by more than that.

## Why 1.0 and 1.4 nats

| what a trial has read alone | nats | lead | passes alone |
|---|---|---|---|
| layer 37, 1.6 ceilings | 0.27 | 0.633 | no |
| layer 38, 1.6 ceilings | 0.35 | 0.654 | yes, thinly (0.594, 0.605 across the halves) |
| layer 45, `break_rel`'s lower dose | 0.92 | 0.706 | yes |
| layer 37, 2.3 ceilings | 0.95 | 0.657 | yes (0.604, 0.605) |
| layer 39, `break_rel`'s lower dose | 1.00 | 0.690 | yes |
| layer 38, 2.3 ceilings | 1.34 | 0.655, 0.682 | yes, on two draws of neurons |
| layer 38, 3.14 ceilings | 3.25 | 0.496 | no: chance |

- **The lower dose, 1.0 nats.** It is what `break_rel`'s lower dose makes at layers 39 and 45, where the census is
  calibrated, and it is the push that passed alone at layer 37. A lower target near 0.35 was not taken: read off
  layer 37's own two doses it would fail there, since layer 37 carries less between the halves than layer 38 does
  at the same movement.
- **The upper dose, 1.4 nats.** It is the hardest push that has passed below the band (layer 38, twice). Somewhere
  between 1.4 and 3.3 nats the lead at layer 38 is lost, and nothing in between has been read there, so the upper
  dose is not taken further up.

The two doses are close: about a tenth apart in ceilings. That is what the steep curve leaves. The design is near
to one dose asked twice over, and its gate on the lower dose alone is near to its gate on the whole reading. Both
gates are kept as they are.

## Where the counts come from: the fine dose pilot

The steer lens's pilot has rungs a factor of two apart, and every dose above lies between the same two rungs at
these layers, so each figure in the table above is read off a straight line between two points. A dose set to a
tenth of a ceiling cannot rest on that. `scripts/census_dose_pilot.py` takes the same measurement by the same code
(128 neurons a layer, each pushed alike in the stream's median length, the median of the next word-piece's KL from
the unpushed one) on eighteen rungs 7% apart, from 0.29 to 0.92 of the stream, into a folder of its own.

**The rule, fixed before the pilot is seen.** For each layer the pilot measures, the count of ceilings at which its
ladder meets 1.0 nats and the count at which it meets 1.4 (a line between the two rungs around each, then times
the layer's ceilings in one stream, [`ceilings.json`](ceilings.json)), each rounded to a hundredth of a ceiling.
Those two numbers go into `MATCH_CEIL` in `backend/census.py` as they come out. A layer whose ladder does not hold
both targets gets no doses and is refused until it is measured on a ladder that does.

## What the fine pilot said

Taken on 13 November at 05:14Z to 05:24Z, at layers 30 to 38 and the reference layer 45
([`fine_pilot.json`](fine_pilot.json)). Every ladder rises all the way. This section was written after the pilot;
everything above it was written before.

| layer | ceilings for 1.0 nats | ceilings for 1.4 nats | nats at 1.6 ceilings | nats at 2.3 ceilings |
|---|---|---|---|---|
| 30 | 2.42 | 2.67 | 0.23 | 0.80 |
| 31 | 2.58 | 2.76 | 0.24 | 0.57 |
| 32 | 2.21 | 2.44 | 0.35 | 1.15 |
| 33 | 2.31 | 2.51 | 0.31 | 0.99 |
| 34 | 2.62 | 2.79 | 0.22 | 0.54 |
| 35 | 2.27 | 2.41 | 0.30 | 1.09 |
| 36 | 2.27 | 2.46 | 0.26 | 1.07 |
| 37 | 2.41 | 2.55 | 0.25 | 0.77 |
| 38 | 2.14 | 2.30 | 0.26 | 1.39 |

Those pairs are in `MATCH_CEIL` as they came out. Layer 45 has none: its ladder ends at 0.55 nats, under both
targets.

The finer rungs correct four things said above, which were read off the coarse pilot:

- **Layer 37's 2.3 ceilings is 0.77 nats, not 0.95.** So the push that passed alone at layer 37 is softer than the
  lower target, and both of layer 37's doses here (2.41 and 2.55 ceilings) are harder than any push read at that
  layer. The rule was fixed before the pilot, and the doses stand as they came out.
- **Layer 38's 1.6 ceilings is 0.26 nats, not 0.35, and layer 37's is 0.25.** At that dose the two layers were
  pushed alike by this measure. One passed thinly and the other failed thinly, so the pilot does not say why they
  differ, as the second section above claims it does.
- **Layer 38's 2.3 ceilings is 1.39 nats.** At layer 38 the design's upper dose is `soft_ceil`'s own, 2.30.
- **The curve is steeper than said.** At layer 37, near the doses, one rung (a push 7% larger) moves the next
  word-piece between 1.3 and 1.5 times as far. The two doses are 6% apart.

## What the trial at layer 37 said: it passes

Registered in [`../census_match_trials/prereg/`](../census_match_trials/prereg/) (hashed at 05:28Z on 13 November,
before any answer of the design existed): ±2.41 and ±2.55 ceilings, 1,022 neurons the walk's trial did not ask,
1,024 controls drawn afresh. Read at 06:45Z. 49,110 answers in 75 minutes.

| reading | lead | beyond text survival | across the halves | new questions |
|---|---|---|---|---|
| whole | 0.653 | 0.678 | 0.609, 0.627 | 0.654 |
| the ±2.41 answers alone | 0.647 | 0.672 | 0.599, 0.600 | 0.628 |
| the 857 of 1,022 the controls cover | 0.629 | 0.664 | 0.585, 0.606 | 0.636 |
| the ±2.55 answers alone (reported) | 0.616 | 0.635 | 0.615, 0.596 | 0.618 |
| bars | 0.60 | 0.58 | 0.57, 0.57 | 0.56 |

- **All three gates pass**, and the whole reading cut to 32 word-pieces passes too. The thinnest gating figure is the covered
  neurons' across the halves, 0.585 against 0.57. 30 of the 1,022 neurons beat every control, where one would by
  chance; the walk's trial here had 28.
- **What was said beforehand held, but for the share covered.** About as likely to pass as not: it passes. The
  whole lead (0.64 to 0.70 was said) is 0.653, the lower dose's (0.63 to 0.69) is 0.647. 84 neurons in 100 are
  covered, where between seven and eight in ten was said.
- **The text is hardly broken at these doses.** The neurons' answers lose 0.02 to 0.03 of their distinct 4-grams
  against the unpushed answers, the controls' under 0.01.
- **The harder dose leads by less.** The ±2.55 answers alone lead by 0.616, the ±2.41 answers by 0.647, and the
  walk's ±2.3 answers led by 0.657 on other neurons. So at this layer the lead is already falling between 0.77 and
  1.4 nats, and 1.4 nats is nearer the edge here than at layer 38, where it led by 0.655 and 0.682. For the layers
  below, the upper target is the one to watch.

What the pass does not show:

- **It is close to one dose read twice.** The two doses are 6% apart, so the gate on the lower dose alone is not
  the separate test it is under `soft_ceil`. The trial shows that layer 37 reads at a push of about 2.4 to 2.55
  ceilings on neurons not asked before. It does not show a second, softer dose reading alone.
- **It is this layer's second trial.** Layer 37 failed narrowly with 1.6 and 2.3 ceilings and passes with 2.41 and
  2.55. The second design was made after the first trial was read, and its targets were chosen knowing what that
  trial showed. The neurons, the controls and the registration were new, and the dose rule was fixed before the
  pilot that set the doses; the choice to try again at this layer was not blind.
- **It says nothing of layers 36 and below.** Their doses are in the table from the same pilot; none is trialled.
- **The rule it rests on is weaker than when it was made.** Layer 52's reading showed that the movement of the
  next word-piece does not sort the leads closely from layer to layer, and the fine pilot showed that it does not
  explain why 38 passed and 37 failed at 1.6 ceilings. The rule gave doses that pass here. That is one layer.

**What it decides, by its registration:** `match_ceil` is the design for the layers below 38, a trial before each,
and layer 37 may be offered to volunteers under a kit of its own. `DOSES_CALIBRATED` is not widened, and the walk
does not go on at 36, without the owner's word.

## What the walk downward said

Registered in [`../census_match_walk/prereg/`](../census_match_walk/prereg/) (hashed at 07:25Z on 13 November,
before any answer of layers 30 to 36 existed; the owner's word: "Walk first"). A trial at each layer in turn, the
next layer down only on a pass. Each layer is written here as it is read.

### Layer 36: it passes, one gate by a thousandth

±2.27 and ±2.46 ceilings, 1,022 neurons none of which had been asked anything, 1,024 controls. Read at 08:51Z.
49,110 answers in 68 minutes.

| reading | lead | beyond text survival | across the halves | new questions |
|---|---|---|---|---|
| whole | 0.627 | 0.636 | 0.608, 0.606 | 0.612 |
| the ±2.27 answers alone | 0.615 | 0.622 | 0.586, 0.571 | 0.600 |
| the 980 of 1,022 the controls cover | 0.622 | 0.632 | 0.600, 0.600 | 0.609 |
| the ±2.46 answers alone (reported) | 0.601 | 0.619 | 0.574, 0.584 | 0.595 |
| bars | 0.60 | 0.58 | 0.57, 0.57 | 0.56 |

- **All three gates pass, the second by 0.001.** The lower dose's words carry from one half of the questions to
  the other by 0.5709 against a bar of 0.57. Two draws of neurons at layer 38 gave leads 0.027 apart at one dose,
  and nothing says this figure is steadier than a lead, so another draw here could fail as easily as pass. By its
  registration the layer passes. The margin is no more than that.
- **What was said beforehand held, but for the share covered.** About as likely to pass as not: it passes. The
  whole lead (0.61 to 0.68, falling a little from layer to layer) is 0.627, down from 0.653 at 37. The upper dose
  alone leads by less than the lower, 0.601 against 0.615. The figure nearest its bar is the one said to fall
  first, across the halves in the lower dose alone. 96 neurons in 100 are covered, where seven to nine in ten was
  said.
- **1.4 nats is at its edge here.** It was said beforehand that an upper dose leading alone by under 0.60, at a
  layer that passes, is too high a target there. It leads by 0.601.
- **Fewer neurons stand out one at a time.** 6 of the 1,022 beat every control, where one would by chance; layer
  37 had 30 with this design and layer 38 had 110 with `soft_ceil`. 21 in 100 beat 95 in 100 of the controls (5 by
  chance), against 31 at layer 37 and 35 at layer 38. The gates read the layer as a whole: how often a neuron's
  words beat a control's. They are met here while the neurons that can be told from a control singly grow fewer.
  A whole layer taken at 36 would tell less about each neuron than at 37 or 38.
- **The text is hardly broken.** The neurons' answers lose 0.01 to 0.02 of their distinct 4-grams against the
  unpushed answers, the controls' 0.01 or less.
- **The whole reading cut to 32 word-pieces passes too**, with one across-the-halves figure at 0.5703.

**The whole lead has fallen at every layer going down**, each with its own design: 0.694 at 39, 0.675 at 38, 0.653
at 37, 0.627 at 36, by a little more each time. One more step the size of the last would put layer 35's lead at
the bar. That is said after layer 36 was read and is no part of the registration, which said that the walk more
likely than not stops at layer 34 or above. (It was wrong: layer 35's lead is higher, below.)

**What it decides, by the registration:** layer 36 may be offered to volunteers under a kit of its own for this
design, and the walk goes on to layer 35 (±2.27 and ±2.41 ceilings), which began to be written at 08:51Z.
`DOSES_CALIBRATED` is not widened without the owner's word.

### Layer 35: it passes, with more room than 36

±2.27 and ±2.41 ceilings, 1,022 neurons none of which had been asked anything, 1,024 controls. Read at 10:01Z.
49,110 answers in 69 minutes, in two sessions (the last point below).

| reading | lead | beyond text survival | across the halves | new questions |
|---|---|---|---|---|
| whole | 0.664 | 0.643 | 0.639, 0.623 | 0.657 |
| the ±2.27 answers alone | 0.667 | 0.653 | 0.582, 0.595 | 0.631 |
| the 978 of 1,022 the controls cover | 0.658 | 0.639 | 0.632, 0.615 | 0.653 |
| the ±2.41 answers alone (reported) | 0.645 | 0.641 | 0.607, 0.595 | 0.616 |
| bars | 0.60 | 0.58 | 0.57, 0.57 | 0.56 |

- **All three gates pass.** The thinnest gating figure is again the lower dose's across the halves, 0.582 against
  0.57. The whole reading cut to 32 word-pieces passes too.
- **The lead did not go on falling.** It is 0.664, above layer 36's 0.627 and layer 37's 0.653. It was said
  beforehand that the whole lead falls a little from layer to layer: it rose. The remark made after layer 36, that
  one more step would put this layer at the bar, was wrong.
- **The rest of what was said beforehand.** The whole lead is inside 0.61 to 0.68. The upper dose alone leads by
  less than the lower, 0.645 against 0.667, and not by under 0.60. 96 neurons in 100 are covered, where seven to
  nine in ten was said, as at 36.
- **More of the lead goes with how the text survives.** Beyond text survival (0.643) is under the lead (0.664)
  for the first time below the band; at 36, 37 and 38 it was above. The controls' own masses agree between the
  question halves more than before: a correlation of 0.13, against 0.05 at 36, 0.04 at 37 and 0.01 at 38. So at
  this depth a push in any direction leaves more of a mark of its own on the words. The gate made for that (0.58)
  is met by 0.06. It is the figure to watch further down.
- **Few neurons stand out one at a time, as at 36.** 3 of the 1,022 beat every control (one by chance), and 24 in
  100 beat 95 in 100 of the controls.
- **The card faulted once while the layer was written.** At 09:17:31Z the driver's watchdog stopped the run (Xid 8)
  after 18,240 answers. The walk's script began it again thirty seconds later, as it is registered to, and the
  second session wrote the other 30,870. Every answer of the job list is there. A stop of this kind ends the work
  in flight; that it left the answers already written as they should be is not shown by the reading. The two
  chunks written in the two minutes before the fault (chunks 22 and 23 of 64: 1,536 answers, 1,344 of them from
  before it) are queued to be written again when the walk frees the card, ahead of the steer lens, and compared
  bit for bit. That check is no part of the registration and is written here when it is made.

**What it decides, by the registration:** layer 35 may be offered to volunteers under a kit of its own for this
design, and the walk goes on to layer 34 (±2.62 and ±2.79 ceilings), which began to be written at 10:01Z.
`DOSES_CALIBRATED` is not widened without the owner's word.

### Layer 34: it passes, by more than any layer below the band

±2.62 and ±2.79 ceilings, 1,022 neurons none of which had been asked anything, 1,024 controls. Read at 11:11Z.
49,110 answers in 68 minutes, in one session.

| reading | lead | beyond text survival | across the halves | new questions |
|---|---|---|---|---|
| whole | 0.758 | 0.747 | 0.683, 0.692 | 0.709 |
| the ±2.62 answers alone | 0.745 | 0.740 | 0.640, 0.650 | 0.675 |
| the 977 of 1,022 the controls cover | 0.755 | 0.742 | 0.676, 0.687 | 0.706 |
| the ±2.79 answers alone (reported) | 0.728 | 0.723 | 0.657, 0.653 | 0.659 |
| bars | 0.60 | 0.58 | 0.57, 0.57 | 0.56 |

- **All three gates pass, none of them thinly.** The figure nearest its bar is the lower dose's across the
  halves, 0.640 against 0.57. The whole reading cut to 32 word-pieces passes too (0.727).
- **Two things said beforehand were wrong, both on the low side.** The whole lead was to lie between 0.61 and 0.68
  at every layer reached: it is 0.758, the highest of any trial below the band and above layer 39's 0.694. And
  the walk was more likely than not to stop at layer 34 or above: it has passed 34.
- **What held.** The upper dose alone leads by less than the lower, 0.728 against 0.745. The share covered missed
  as before, 96 in 100.
- **More neurons stand out here than at 35 or 36, though few beat every control.** 42 in 100 beat 95 in 100 of
  the controls (24 at 35, 21 at 36, 31 at 37, 35 at 38). 8 of the 1,022 beat every control.
- **The text is hardly broken.** The neurons' answers lose under 0.01 of their distinct 4-grams, the controls'
  about 0.005. Beyond text survival is a little under the lead, as at 35, and the controls' masses agree between
  the question halves by 0.11.
- **This layer has the largest doses of the walk in ceilings and the largest lead.** The pilot's rule asked for
  more push here because a ceiling moves the next word-piece less at 34. Between this design's two doses the harder
  one leads by less at every layer read, so the larger lead is the layer's and not the push's. From 36 to 34 the
  same two targets in nats gave leads of 0.627, 0.664 and 0.758: the rule gives doses that pass, and does not say
  by how much.

**What it decides, by the registration:** layer 34 may be offered to volunteers under a kit of its own for this
design, and the walk goes on to layer 33 (±2.31 and ±2.51 ceilings), which began to be written at 11:11Z.
`DOSES_CALIBRATED` is not widened without the owner's word.

### Layer 33: it passes, a little under layer 34

±2.31 and ±2.51 ceilings, 1,022 neurons none of which had been asked anything, 1,024 controls. Read at 12:26Z.
49,110 answers in 73 minutes, in one session.

| reading | lead | beyond text survival | across the halves | new questions |
|---|---|---|---|---|
| whole | 0.726 | 0.726 | 0.630, 0.637 | 0.691 |
| the ±2.31 answers alone | 0.710 | 0.706 | 0.601, 0.617 | 0.660 |
| the 982 of 1,022 the controls cover | 0.727 | 0.721 | 0.628, 0.636 | 0.691 |
| the ±2.51 answers alone (reported) | 0.702 | 0.697 | 0.605, 0.591 | 0.649 |
| bars | 0.60 | 0.58 | 0.57, 0.57 | 0.56 |

- **All three gates pass with room.** The gated figure nearest its bar is the lower dose's across the halves,
  0.601 against 0.57. The whole reading cut to 32 word-pieces passes too (0.708). The card did not fault: one
  session, nothing in the run's log or the kernel's.
- **What was said beforehand.** The whole lead was to lie between 0.61 and 0.68: it is 0.726, wrong again on the
  low side. It was to fall a little from layer to layer: it fell here, from 0.758, after rising twice. The upper
  dose alone leads by less than the lower, as said (0.702 against 0.710), and not under 0.60. The share covered
  missed as before, 96 in 100.
- **Neurons stand out about as at 34.** 37 in 100 beat 95 in 100 of the controls (42 at 34). 29 of the 1,022 beat
  every control (8 at 34, 3 at 35, 6 at 36, 30 at 37). That count hangs on the one highest control and swings from
  layer to layer, so it is a rough figure.
- **The text is hardly broken, and the lead does not go with it.** The neurons' answers lose under 0.005 of their
  distinct 4-grams. Beyond text survival equals the lead, 0.726. The controls' masses do not agree between the
  question halves (-0.02); the 0.11 and 0.13 of layers 34 and 35 do not carry on here.
- **Four layers, the same two targets, no slope.** From 36 to 33 the whole lead is 0.627, 0.664, 0.758, 0.726.
  The guess registered beforehand, that the walk would stop by layer 34, read layers 38 to 36 as a fall towards
  the bar. Below 36 there is no such fall so far.

**What it decides, by the registration:** layer 33 may be offered to volunteers under a kit of its own for this
design, and the walk goes on to layer 32 (±2.21 and ±2.44 ceilings), which began to be written at 12:26Z.
`DOSES_CALIBRATED` is not widened without the owner's word.

### Layer 32: it passes, with less room than 33 or 34

±2.21 and ±2.44 ceilings, 1,022 neurons none of which had been asked anything, 1,024 controls. Read at 13:40Z.
49,110 answers in 73 minutes, in one session.

| reading | lead | beyond text survival | across the halves | new questions |
|---|---|---|---|---|
| whole | 0.699 | 0.678 | 0.615, 0.631 | 0.675 |
| the ±2.21 answers alone | 0.671 | 0.634 | 0.589, 0.602 | 0.640 |
| the 942 of 1,022 the controls cover | 0.686 | 0.663 | 0.604, 0.619 | 0.660 |
| the ±2.44 answers alone (reported) | 0.690 | 0.667 | 0.592, 0.601 | 0.631 |
| bars | 0.60 | 0.58 | 0.57, 0.57 | 0.56 |

- **All three gates pass.** The gated figure nearest its bar is the lower dose's across the halves, 0.589 against
  0.57. The whole reading cut to 32 word-pieces passes too (0.677). The card did not fault: one session, nothing
  in the run's log or the kernel's.
- **What was said beforehand.** The whole lead was to lie between 0.61 and 0.68: it is 0.699, above that range
  for the third layer running. It fell a little from layer 33's, as said. 92 neurons in 100 are covered, nearer
  the seven to nine in ten said than at any layer before and still above it.
- **The harder dose leads by more here, the first layer of the walk where it does.** 0.690 for the ±2.44 answers
  alone against 0.671 for the ±2.21. The registration said the upper dose would lead by less at most layers
  reached, which holds at four of five. The remark made at layer 34, that the harder dose leads by less at every
  layer read, stops being true here. ±2.21 is the smallest lower dose of the walk in ceilings, and at this layer
  the 1.0-nat target looks a little low rather than a little high.
- **More of the lower dose's lead goes with how the text survives.** 0.671 falls to 0.634 beyond text survival,
  the widest such gap of the walk, and still 0.054 over its bar. The text itself is hardly broken: the neurons'
  answers lose under 0.005 of their distinct 4-grams. The controls' masses do not agree between the question
  halves (0.03).
- **Neurons standing out.** 32 in 100 beat 95 in 100 of the controls (37 at 33, 42 at 34). 54 of the 1,022 beat
  every control, the most of the walk; that count is a rough one, as said at 33.
- **Five layers: 0.627, 0.664, 0.758, 0.726, 0.699.** The whole lead is highest at 34 and has come down by about
  0.03 a layer since. No guess is drawn from two steps here: the one drawn that way at layer 36 was wrong.

**What it decides, by the registration:** layer 32 may be offered to volunteers under a kit of its own for this
design, and the walk goes on to layer 31 (±2.58 and ±2.76 ceilings), which began to be written at 13:40Z.
`DOSES_CALIBRATED` is not widened without the owner's word.

## What it may not do

- **Layer 37 may fail whatever the dose.** It carried less between the question halves than layer 38 at every
  push read. If it fails with both doses near where it read best, the bars are not met at layer 37, and that is a
  finding about the layer, not about the dose.
- **The measure is the next word-piece, not an answer.** A push matched on one may still differ on the other from
  layer to layer.
- **128 neurons stand for 21,504.** The pilot's median is over a sample; the census pushes each neuron by its own
  ceiling, which varies by a few parts in a hundred within a layer.

## Cost

The fine pilot: ten minutes for layers 30 to 38. A trial: 49,110 answers, 75 to 88 minutes.

## Choices that are the owner's

1. Whether a layer that passes is offered to volunteers, and under which kit. Eight layers outside 39 to 51 now
   pass a trial, under two designs: 32 to 37 with `match_ceil`, 38 and 52 with `soft_ceil`. Layer 36's pass is
   the thinnest of the eight and layer 34's the widest below the band. At 35 and 36 few neurons stand out
   singly, so a whole layer there tells less about each neuron.
2. Widening `DOSES_CALIBRATED` to a layer, on its passed trial.
3. The walk below 32 is under way by its registration, layer 31 next, and stops by itself at the first failure
   and at layer 30. Going on below 30 wants the fine pilot taken there first.
4. Whether one design is to cover every layer outside the band. `match_ceil` has no doses at 52 and above (the
   fine pilot was not taken there) and no trial at 38.
