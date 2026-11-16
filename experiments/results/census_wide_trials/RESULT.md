# The walk downward with match_wide: layer 31 passes, layer 30 fails, the walk stops

Read on 16 November 2026 by the registration in `prereg/` (hashed before any answer of the design existed; its files
still match `SHA256SUMS`, and there are no amendments). Layers are counted from 0, as the trial and the board count
them: 31 and 30 here are 32 and 31 on the pages. `summary.json` has every figure; `collect.py` copies the verdicts
from `atlas_data/census/trials/wide`.

`match_wide` pushes each neuron by one count of its own write ceiling, both ways. The count is `match_ceil`'s lower
one, the push that moves the next word-piece by 1.0 nats. It asks twelve questions, read as two halves of six, at 24
answers a neuron. Each layer was read on 1,022 neurons no census had asked, against 1,024 controls drawn afresh
(seed 8).

| layer | verdict | reading | lead | beyond text | words carry A→B | words carry B→A | new questions |
|---|---|---|---|---|---|---|---|
| 31 | **pass** | whole | 0.630 | 0.645 | 0.583 | 0.581 | 0.608 |
| | | covered (986 of 1,022) | 0.636 | 0.640 | 0.586 | 0.587 | 0.613 |
| 30 | **fail** | whole | *0.579* | 0.611 | *0.555* | *0.550* | 0.584 |
| | | covered (974 of 1,022) | *0.582* | 0.604 | *0.557* | *0.551* | 0.585 |
| bars | | | 0.60 | 0.58 | 0.57 | 0.57 | 0.56 |

## Layer 31: it passes both gates

±2.58 ceilings, 49,116 answers in 79 minutes, 10.4 a second. The run was in two sessions: it was paused for 3 minutes
at 15:45Z, between batches, for the fine dose pilot at 53–59, and resumed row for row (`L31-match_wide-trial.pauses`).
It was read at 16:59Z. Every gate passes on both readings. The thinnest margin is the words carrying from half B to
half A, 0.581 against 0.57, about one standard error (0.013) above the bar.

`match_ceil` failed here on 13 November on its lower dose's twelve answers a neuron: those answers did not carry a
neuron's words between the halves (0.553 and 0.548). The same dose on twelve questions carries them (0.583 and 0.581).

Was that the added questions, or a better draw of neurons and controls? The registration set a check for this. Read
on break_rel's six questions alone, three a half, as `match_ceil`'s walk read them, this trial gives 0.549 and 0.539.
That is where `match_ceil`'s lower dose was, and under 0.57 as said beforehand. So the new neurons and controls read
no better than the last ones, and the pass comes from the added questions. The six added questions, read alone,
give 0.567 and 0.562.

**What was said beforehand, and how it came out:**
- A pass about 55 in 100. It passed.
- The lead between 0.66 and 0.76. It is 0.630, under that range. The lead did not grow with questions: it was 0.634 on
  `match_ceil`'s six. The projection (`census_question_count/NOTE.md`) assumed it would grow, and was wrong on the lead.
- The words carrying between the halves between 0.56 and 0.62. They are 0.583 and 0.581, as said. The projection gave
  0.606 and 0.571–0.587.
- The new questions alone between 0.61 and 0.70. They are 0.608, 0.002 under the range.
- On break_rel's six alone, the words carrying between 0.53 and 0.58 and under 0.57 more likely than not. They are
  0.549 and 0.539.
- 90 to 97 neurons in 100 covered. It is 96.5.

**Reported, gating nothing:**
- The text barely moves. Neurons and controls alike lose 0.002 to 0.003 of their distinct 4-grams. 97 in 100 answers
  run to the 48-piece cut.
- Cut to 32 word-pieces the words carrying fall to 0.567 and 0.564. The length rule keeps 48.
- Few neurons stand out singly. 6 of the 1,022 beat every control, about 1 by chance, and none is named. Read on
  either six questions alone, 31 and 39 are named (38 and 6 above every control).
  - On twelve questions the lead is broad and its top thin; on six it is lower and its top thicker. Why is not read
    here.
  - A whole layer taken with this design at 31 would name few neurons, perhaps none. This bears on whether to gate on
    yield (`census_question_count/NOTE.md`).
- The `G5b` lines marked FAIL in the two three-a-half readings are expected. That check compares a reading with the
  run's own ranking of all twelve questions. A reading on six cannot match it. `match_ceil`'s readings of one dose
  alone show the same.

## Layer 30: it fails both gates, and the walk stops

±2.42 ceilings, 49,116 answers in 82 minutes in one session, 10.0 a second. It was read at 18:23Z.

It fails twice on each reading:
- **The lead.** 0.579 whole and 0.582 covered, against 0.60.
- **Words carrying between the halves.** 0.555 and 0.550, and 0.557 and 0.551, against 0.57.

Beyond text survival (0.611) and the new questions alone (0.584) pass. The text barely moves here too: 0.001 to 0.002
of the distinct 4-grams is lost, and 94 to 95 in 100 answers run to the cut. So the push does not break the text.
Three neurons beat every control, and none is named.

Half A leads by 0.559 alone and half B by 0.584. Read three questions a half, break_rel's six lead by 0.559 and
carry their words by 0.528 and 0.514; the six added lead by 0.594 and carry by 0.547 and 0.547.

The registration said layer 30 was about as likely to pass as not. Which gate would fail first was not said. The
lead fails with the text whole, so this push does not overdrive the layer. Whether it is too weak here, or the
layer's neurons say less on these questions, is not read.

## What it decides

By the registration:
- **Layer 31 (32 on the pages).**
  - It may be offered to volunteers under a kit of its own for `match_wide`. Before that kit, the other card class's
    bars need a bars registration of their own; kit 4 measured none for this design.
  - `DOSES_CALIBRATED` is widened to it only on the owner's word. It has not been widened.
- **Layer 30 (31 on the pages).**
  - No whole layer is taken there with `match_wide`, and this registration does not trial it again.
  - What to try at 30, and whether to go on below it, is the owner's choice. Either would be registered anew on
    neurons this walk did not ask. Below 30 there is no fine dose pilot yet.

With this, nine layers outside 39–51 pass a trial:
- 31 with `match_wide`;
- 32 to 37 with `match_ceil`;
- 38 and 52 with `soft_ceil`.
