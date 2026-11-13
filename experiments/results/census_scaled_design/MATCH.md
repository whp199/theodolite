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

## What it may not do

- **Layer 37 may fail whatever the dose.** It carried less between the question halves than layer 38 at every
  push read. If it fails with both doses near where it read best, the bars are not met at layer 37, and that is a
  finding about the layer, not about the dose.
- **The measure is the next word-piece, not an answer.** A push matched on one may still differ on the other from
  layer to layer.
- **128 neurons stand for 21,504.** The pilot's median is over a sample; the census pushes each neuron by its own
  ceiling, which varies by a few parts in a hundred within a layer.

## Cost

The fine pilot: about a quarter of an hour for layers 30 to 38. A trial: 49,110 answers, about 88 minutes.

## Choices that are the owner's

1. Whether a layer that passes is offered to volunteers, and under which kit.
2. Widening `DOSES_CALIBRATED` to a layer, on its passed trial.
3. Whether the walk goes on below a layer that passes.
