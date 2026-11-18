# match_things at layer 30: it passes both gates

Read on 18 November 2026 at 05:04Z by the registration in `prereg/`. That registration was hashed at 00:20:38Z on 18
November, before any answer of the design existed, and pushed at 02:06:34Z, before the run began at 02:06:46Z. Its files
still match `SHA256SUMS`, and there are no amendments. The run was written from the clean checkout of `ecb94d6`, the
commit in `prereg/repo_commit` (its meta says so). Layers are counted from 0, as the trial and the board count them: 30
here is 31 on the pages. `summary.json` has every figure; `collect.py` copies the verdicts from
`atlas_data/census/trials/things`.

`match_things` asks twelve questions of three genres (how a thing works, three tips, an object) at both of
`match_ceil`'s doses, ±2.42 and ±2.67 ceilings at layer 30: 48 answers a neuron. Half A is bicycle, walk, chair, kettle,
sleep and lamp; half B is clock, rice, vase, lock, money and blanket, which no run had asked before. The trial read
1,022 neurons no census had asked at 30, against 1,024 controls drawn afresh (seed 10).

| reading | lead | beyond text | words carry A→B | words carry B→A | new questions |
|---|---|---|---|---|---|
| whole | 0.618 | 0.637 | 0.582 | 0.587 | 0.625 |
| covered (998 of 1,022) | 0.615 | 0.633 | 0.580 | 0.586 | 0.623 |
| *reported:* the lower dose alone (±2.42) | 0.616 | 0.634 | 0.588 | 0.581 | 0.606 |
| *reported:* the upper dose alone (±2.67) | *0.588* | 0.609 | *0.559* | *0.553* | 0.581 |
| *reported:* the first three of each half | 0.608 | 0.620 | 0.571 | *0.559* | 0.608 |
| *reported:* the last three of each half | 0.621 | 0.638 | *0.560* | *0.569* | 0.604 |
| bars | 0.60 | 0.58 | 0.57 | 0.57 | 0.56 |

The run wrote 98,220 answers in 175 minutes in one session, 9.4 a second (02:06:46Z to 05:01:34Z).

## It passes both gates

Both readings pass every gate, and the sanity checks (G5, G5b) pass on the whole reading. The narrowest margins are on
the words carrying from half A to half B: 0.582 whole and 0.580 covered, 0.012 and 0.010 over the bar. Each is under
one standard error (each figure's is near 0.013). Every other figure clears its bar by 0.015 or more.

**Beside `match_both` at 30** (other neurons and controls, half its questions about the model and colours):
- **The lead** is the same, 0.618 and 0.618.
- **The words carry** both ways now, and about evenly: 0.582 and 0.587, where `match_both` read 0.573 and 0.545. Read
  again (`census_word_carry/NOTE.md`), `match_both`'s questions about the model and colours took a push's words up less
  than its objects and tasks. Without them, the lower of the two figures rose from 0.545 to 0.582.
- **The new questions alone** lead more than the old: 0.625 on half B, 0.601 on half A.

**What the second dose added here: almost nothing.** The lower dose alone, 24 answers a neuron, passes all four figure
gates by itself: 0.616, 0.634, 0.588 and 0.581, 0.606. The upper dose alone fails the lead and the words carrying.
Read together, the two doses lead by 0.002 more than the lower alone, and carry by the same on average (0.5848 against
0.5845). Under `match_both` the second dose had added 0.022 to the lead. The lower dose alone is `match_wide`'s dose.
So at 30 the questions seem to matter, not the second dose. That is a reading of answers already written, gating
nothing. The registration says a pass would not show that one dose reads alone here.

**On three questions a half the words do not reach the bar both ways.** The first three of each half carry 0.571 and
0.559, the last three 0.560 and 0.569. Six a half carry both ways, by 0.017 and 0.023 more than three a half on
average. Before the hash, `match_both` had shown a lift of about 0.012 from three questions a half to six.

**What was said beforehand, and how it came out:**
- A pass about 1 in 4, a failure more likely than not. It passed.
- If it failed, it would fail on the words carrying first. It did not fail.
- The lead between 0.59 and 0.64, most likely near 0.615. It is 0.618.
- The words carrying between 0.55 and 0.59 each, most likely near 0.57. They are 0.582 and 0.587, inside the range
  and above the likeliest value.
- Beyond text survival and the new questions alone would pass. They do, at 0.637 and 0.625.
- Neither dose alone would pass G1 to G4. **Wrong**: the lower dose alone passes all four.
- The upper dose alone would lead no more than the lower, more likely than not. It leads by 0.588, against 0.616.
- 90 to 97 neurons in 100 covered. It is 97.7, just over.
- The text would lose under 0.005 of its distinct 4-grams at either dose. It loses at most 0.003.

**Reported, gating nothing:**
- **The text** barely moves. Neurons lose 0.002 to 0.003 of their distinct 4-grams, controls 0.002. 99.7 to 100 in
  100 answers run to the 48-piece cut (94 to 96 under `match_both`, on other questions).
- **Cut to 32 word-pieces**, it still passes: the lead is 0.604, the words carry 0.571 and 0.577, the new questions
  lead 0.608. So the length rule would take 32 word-pieces, as it would have at 35, 37, 38, 39 and 52. Every kit has
  kept 48.
- **Few neurons stand out singly.** 6 of the 1,022 beat every control (about one would by chance), and none is named
  (q ≤ 0.05). But 78 are strong on both halves (p ≤ 0.05 on each), where 2.6 would be by chance.
- **Controls that stand out.** As at 30, 31 and 35 to 37 before, the controls that stand out most write other
  languages. The strongest stands 8.5 standard deviations over the controls' median and writes Portuguese ("para
  estão aqui que três dicas"); the next two write Spanish and Russian. With the five strongest set aside, 45 neurons
  have q ≤ 0.05. 27 neurons stand more than five standard deviations over the controls' median. These were computed
  after the reading from the run's units file, as `census_L34_whole/read.sh` computes them.
- **The `G5b` lines marked FAIL** in the four report-only readings are expected. That check compares a reading with
  the run's own ranking of every answer at both doses, which a reading of only some answers cannot match.
- **A slip in a docstring.** `things_read.py`'s docstring names the last three of half B as "lock, suitcase, blanket".
  The reader takes the halves from the run's design and read lock, money, blanket, as `RULES.md` says. The docstring
  was hashed as it is and changes nothing read.

## What it decides

By the registration:
- **Layer 30 may be offered to volunteers under a kit that carries `match_things`.** A whole layer is 1,130,508
  answers, twice a `match_wide` layer: about 33 hours at this run's pace on the home 5090.
- **Before that kit, its other card classes' bars need a bars registration of their own.** No accept bars exist for
  `match_things` at any layer, so `census_intake.py` would refuse a PRO 6000 or 3090 run of it today. A 5090 with the
  home card's bits needs none.
- **`DOSES_CALIBRATED` is widened to `match_things` at 30 only on the owner's word.** It is not widened here, and no
  code or kit has changed.

**What this does not show.**
- **That layer 30's neurons answer the census's other questions as they answer these.** A layer taken with this
  design is asked none of who, orange or free, which every other layer is asked. It could be set beside the others
  only on bicycle, walk and chair, and beside 31 (under `match_wide`) also on kettle, sleep and lamp.
- **That one dose reads alone at 30.** The lower dose passes on its own here, but only as a report. A design of that
  dose on these questions (half the answers, about 17 hours a layer) would need a trial of its own on fresh neurons.
- **That `match_things` reads anywhere it was not trialled**, or anything below 30. Below 30 there is no fine dose
  pilot yet.
- **That the pass is more than a near thing.** This is the third design tried at 30. The first two failed there, and
  this one was built from what they showed. Its neurons, its controls and half its questions are new, but three tries
  at one layer give more than one chance to pass. The words carrying clear their bar by under one standard error on
  the covered neurons.
