# A census design for layers 0 to 38: the push counted in write ceilings

12 November 2026. The owner asked for gentler doses for the layers where `break_rel` stopped. This is the design, what
it rests on, and the one trial that decides whether it is worth building further. Nothing of it has been run. Layers
count from 0.

## What is known

`break_rel` pushes every neuron of a layer by 1 and 2 of the stream's median length there. It passes its trial at
layers 39 and 51 and held on every whole layer from 40 to 50. It failed at layer 38, with its lead at chance (0.497),
and its first probe at layer 57 failed on its +-1 answers (0.52 alone).

A neuron's write ceiling is the most it can write to the stream by itself. `ceilings.py` reads it from the checkpoint
for every layer and sets it against the stream (`ceilings.json`). How many ceilings a push of 1 of the stream is:

| layers | 1 of the stream, in ceilings | `break_rel` |
|---|---|---|
| 39 to 51 | 1.76 to 2.76 | passes at every one |
| 38 | 3.14 | fails: lead 0.497 |
| 35 to 37 | 3.84 to 3.93 | not tried |
| 26 to 34 | 4.19 to 5.03 | not tried (the ladders at 30: weak, 0.58 and 0.59) |
| 2 to 25 | 5.4 to 18.6 | not tried (the ladders at 20: none, 0.52 and 0.50) |
| 0 and 1 | 3.74 and 1.95 | not tried |
| 52 | 1.06 | being trialled; predicted to fail (`PREDICTION.md`) |
| 57 | 1.21 | probe failed on its +-1 answers |

So the design tells a neuron from a random direction where its lower dose is about two ceilings, and at no layer
tried where it is much more or much less. Below layer 39 the stream grows longer and the ceilings smaller, so the
same nominal dose is a harder and harder push: three ceilings at 38, five at 30, eleven at 20.

How the text survives says the same from another side (`breaks.py`, `breaks.json`: every run already taken). In the
band, 1 of the stream sits at the edge of where text breaks (the answers lose 0.04 to 0.10 of their distinct 4-grams
at about 1.0 of the stream, 1.9 to 2.7 ceilings). At layer 38 it is past that edge (0.147 lost). At layer 30 the edge
is at 0.92 to 0.95 of the stream, 4.4 to 4.6 ceilings, and at layer 20 at 0.29 to 0.30, 3.1 to 3.3 ceilings: early
layers stand more ceilings before text breaks, but far fewer streams.

## The design: `break_ceil`

`break_rel`, with its two doses counted in ceilings and not in streams:

- **Doses:** -4.6, -2.3, +2.3 and +4.6 times each neuron's own write ceiling. A control is pushed by the ceiling of
  the neuron it is paired with, as `calib_ceil`'s are. 2.3 is the middle of what the passing band was pushed by
  (1.76 to 2.76); the upper dose is twice the lower, as in `break_rel`.
- **Everything else as `break_rel`:** its six questions, 48 word-pieces, a batch of 64, 2,048 controls for a whole
  layer (1,024 for a trial), every answer counted, each word against the controls alone, the collapse rule read
  from the controls' answers at the upper dose.
- **The gates:** the three the neighbour trials are read by, at layer 45's bars: the whole reading, the lower dose's
  answers alone, and the neurons whose text survives as the controls' does.

What the two doses come to at each layer, in the stream's median length (what `break_rel` calls 1 and 2):

| layer | 1 of the stream, in ceilings | lower dose, of the stream | upper dose, of the stream |
|---:|---:|---:|---:|
| 0 | 3.74 | 0.61 | 1.23 |
| 1 | 1.95 | 1.18 | 2.36 |
| 2 | 18.62 | 0.12 | 0.25 |
| 3 | 12.18 | 0.19 | 0.38 |
| 4 | 13.20 | 0.17 | 0.35 |
| 5 | 15.76 | 0.15 | 0.29 |
| 6 | 13.75 | 0.17 | 0.33 |
| 7 | 9.88 | 0.23 | 0.47 |
| 8 | 8.25 | 0.28 | 0.56 |
| 9 | 13.39 | 0.17 | 0.34 |
| 10 | 12.18 | 0.19 | 0.38 |
| 11 | 13.57 | 0.17 | 0.34 |
| 12 | 14.01 | 0.16 | 0.33 |
| 13 | 16.68 | 0.14 | 0.28 |
| 14 | 17.17 | 0.13 | 0.27 |
| 15 | 11.38 | 0.20 | 0.40 |
| 16 | 17.92 | 0.13 | 0.26 |
| 17 | 17.12 | 0.13 | 0.27 |
| 18 | 15.25 | 0.15 | 0.30 |
| 19 | 11.72 | 0.20 | 0.39 |
| 20 | 10.80 | 0.21 | 0.43 |
| 21 | 11.38 | 0.20 | 0.40 |
| 22 | 8.63 | 0.27 | 0.53 |
| 23 | 6.52 | 0.35 | 0.71 |
| 24 | 5.91 | 0.39 | 0.78 |
| 25 | 5.37 | 0.43 | 0.86 |
| 26 | 4.55 | 0.51 | 1.01 |
| 27 | 4.19 | 0.55 | 1.10 |
| 28 | 4.24 | 0.54 | 1.08 |
| 29 | 4.70 | 0.49 | 0.98 |
| 30 | 4.81 | 0.48 | 0.96 |
| 31 | 4.94 | 0.47 | 0.93 |
| 32 | 5.03 | 0.46 | 0.91 |
| 33 | 4.89 | 0.47 | 0.94 |
| 34 | 4.56 | 0.50 | 1.01 |
| 35 | 3.84 | 0.60 | 1.20 |
| 36 | 3.93 | 0.59 | 1.17 |
| 37 | 3.86 | 0.60 | 1.19 |
| 38 | 3.14 | 0.73 | 1.46 |

At layers 20 and 30 the ladders say what those pushes do to text: the lower dose leaves it whole (nothing lost at
2.4 to 2.7 ceilings at either layer), and the upper dose is at the edge of the break at layer 30 (0.05 to 0.11 lost
at 4.0 to 4.8 ceilings) and past it at 20 (0.49 lost at 5.4). In `break_rel`'s band the upper dose is far past the
break (0.55 to 0.80 lost), so here the upper dose's answers should be the more readable.

## What it may not mend

Two readings of layer 38's failure fit what is known, and only one of them is about the dose.

- **The dose.** At 3.14 ceilings the push is harder than at any layer that passed, and the text is past its break.
  The neurons there are not silent: the lead beyond how the text survives clears its bar on every reading (0.644,
  0.646, 0.636), 44 of the 1,022 are found on both halves of the questions where 2.6 would be by chance, and one in
  ten stands above the controls' 95th percentile. A push of 2.3 ceilings is what layer 39 got (2.14) and passed.
- **The depth.** The fall from 0.694 at layer 39 to 0.497 at layer 38 is a cliff, where the layers from 39 to 51
  show no slope at all across 1.76 to 2.76 ceilings. Layer 38 to 40 is also where the stream's length and the
  ceilings change regime, and it is one layer from where the lens's labels stop being trusted (the label gate:
  page 41 up). A census reads the words a push brings. If a neuron this deep in does not yet write towards words,
  no dose will give it a lead in words, and the census is the wrong reading for these layers.

The design cannot tell these apart on paper. One trial can.

## The trial that decides

Layer 38 again, with `break_ceil`: the same 1,022 neurons as its first trial (they were drawn before anything was
read, and no question-and-dose cell is shared), 1,024 controls, about 75 minutes on the home card, read by the three
gates, fixed and hashed before it runs.

- **It passes:** the dose was the trouble. The walk goes on downward with this design, a trial before every whole
  layer as now, and the layers that pass are offered to volunteers under a new kit (a new design is a new kit: its
  tag, its canaries).
- **It fails with its lead still near chance:** the depth is the trouble. The census stops at layer 39 on this side,
  and what speaks for layers 0 to 38 is the steer lens at each layer's own dose, the passages, and the wiring.

Two other measurements will bear on it without costing anything more: layer 52's verdict tests the reading above
from the other side, and the steer lens's dose pilot gives, for every layer, how far a push of a given size moves
the next word-piece, which is a third way to say how hard a layer is being pushed.

## Costs

The trial: 75 minutes. If it passes, 38 more trials going down (about 48 hours of the home card, fewer if the walk
stops), and a whole layer each for those that pass (13 hours a layer, the volunteers' or ours).

## Choices that are the owner's

1. The number of ceilings. 2.3 is the middle of the passing band. 2.14 is exactly what layer 39 was pushed by, and
   2.54 what layer 45, the first trial's layer, was.
2. Each neuron's own ceiling, or one push for the whole layer (2.3 times the layer's median ceiling). Within a layer
   the ceilings differ little (a tenth between the 10th and 90th percentile at most layers), and the code already
   pushes by each neuron's own; `break_rel` pushed every neuron of a layer alike.
3. When the layer-38 trial runs: before the steer lens's build (it is short, and it decides whether layers 0 to 38
   have a census at all), or after.
