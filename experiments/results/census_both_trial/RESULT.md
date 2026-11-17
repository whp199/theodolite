# match_both at layer 30: it fails, on the words carrying from half B to half A

Read on 17 November 2026 at 02:04Z by the registration in `prereg/`. That registration was hashed at 22:34Z on
16 November, before any answer of the design existed. Its files still match `SHA256SUMS`, and there are no amendments.
Layers are counted from 0, as the trial and the board count them: 30 here is 31 on the pages. `summary.json` has every
figure; `collect.py` copies the verdicts from `atlas_data/census/trials/both`.

`match_both` asks `match_wide`'s twelve questions at both of `match_ceil`'s doses: ±2.42 and ±2.67 ceilings at layer
30. That is 48 answers a neuron. The trial read 1,022 neurons no census had asked at 30, against 1,024 controls drawn
afresh (seed 9).

| reading | lead | beyond text | words carry A→B | words carry B→A | new questions |
|---|---|---|---|---|---|
| whole | 0.618 | 0.640 | 0.573 | *0.545* | 0.608 |
| covered (956 of 1,022) | 0.622 | 0.631 | 0.577 | *0.546* | 0.612 |
| *reported:* the lower dose alone (±2.42) | 0.596 | 0.630 | 0.558 | 0.549 | 0.593 |
| *reported:* the upper dose alone (±2.67) | 0.580 | 0.615 | 0.536 | 0.535 | 0.565 |
| *reported:* break_rel's six, three a half | 0.597 | 0.610 | 0.549 | 0.542 | 0.591 |
| *reported:* the six added, three a half | 0.616 | 0.631 | 0.546 | 0.551 | 0.599 |
| bars | 0.60 | 0.58 | 0.57 | 0.57 | 0.56 |

The run wrote 98,220 answers in 167 minutes in one session, 9.8 a second (23:13:52Z to 02:01:06Z).

## It fails both gates on one figure

Both readings fail on the words carrying from half B to half A: 0.545 whole and 0.546 covered, against 0.57. That is
about two standard errors under the bar (each figure's is near 0.013). Every other gate passes on both readings,
including the lead, which `match_wide` missed here.

Half B is bicycle, walk, chair, kettle, sleep and lamp; half A is who, orange, free, self, green and interesting. A
neuron's top words on half B do not come back on half A. They do the other way: half A's words carry to half B by 0.573
whole and 0.577 covered, just over the bar. Half A also leads less on its own (0.571) than half B (0.608).

**What the second dose added.** Read against the lower dose alone (the same answers' half):
- **The lead** rose from 0.596 to 0.618, by 0.022. Over `match_ceil`'s walk at 31 to 36 the second dose added −0.003 to
  +0.016. In the mock run made before the hash, answers written twice rose by 0.007. So here the second dose carried
  more than a repeat.
- **The words carrying** rose from 0.558 to 0.573 A→B and fell from 0.549 to 0.545 B→A. At 31 to 36 the second dose
  added 0.02 to 0.04 to each direction. Here it lifted one direction and not the other.

The lower dose alone is `match_wide`'s design at 30 on other neurons and controls. Its lead here is 0.596. In
`match_wide`'s trial it was 0.579, 0.017 lower (about 1.3 standard errors), so the draw of neurons and controls moves this
figure by that much. Neither trial carries the words between the halves at that dose.

**What was said beforehand, and how it came out:**
- A failure more likely than not, a pass about 1 in 8. It failed.
- If it failed, the lead would fail first, between 0.57 and 0.61 and most likely near 0.585. **Wrong.** The lead passed
  at 0.618, over the range. The failure is on the words carrying.
- The words carrying between 0.56 and 0.60 each. A→B is 0.573, inside; B→A is 0.545, under.
- Beyond text survival and the new questions alone would pass. They do, at 0.640 and 0.608.
- The lower dose alone would lead between 0.56 and 0.60. It leads by 0.596. The upper dose alone would lead no more than
  the lower, more likely than not. It leads by 0.580.
- 90 to 97 neurons in 100 covered. It is 93.5.
- The text would lose under 0.005 of its distinct 4-grams at either dose. The most it loses is 0.0048, for the neurons
  at +2.67. The reader rounds that to 0.005. The fourth place was computed after the reading, by `text_unrounded.py`
  (output in `text_unrounded.txt`). The forecast holds, narrowly.

**Reported, gating nothing:**
- **The text** barely moves. Neurons lose 0.0032 to 0.0048 of their distinct 4-grams, more at the upper dose and on the
  plus side; controls lose 0.0014 to 0.0019. 94 to 96 in 100 answers run to the 48-piece cut.
- **Cut to 32 word-pieces**, the lead falls to 0.599 and the words carrying to 0.569 and 0.532. The length rule keeps 48.
- **Few neurons stand out singly.** One of the 1,022 beats every control, about one by chance, and none is named.
- **Three questions a half.** Neither break_rel's six nor the six added carries its words between the halves (0.549 and
  0.542; 0.546 and 0.551). The six added lead more (0.616 against 0.597), as under `match_wide` at 30.
- **The `G5b` lines marked FAIL** in the four report-only readings are expected. That check compares a reading with the
  run's own ranking of every answer at both doses, which a reading of some answers cannot match. The mock showed the
  same before the hash (`prereg/mock_read.txt`).

## What it decides

By the registration:
- **No whole layer is taken at 30 with `match_both`**, and this registration does not trial it again.
- **Kit 5 is prepared for layer 31 (32 on the pages) with `match_wide` alone**, as the owner asked.
- **What to try at and below 30 is the owner's choice.** Two designs have now failed at 30:
  - `match_wide` failed on the lead and on the words carrying both ways;
  - `match_both` failed on the words carrying from half B to half A alone.

  A new attempt would be registered anew, on neurons neither trial asked: 2,108 of layer 30's 21,504 have been asked,
  counting the ladders' 64. Below 30 there is no fine dose pilot yet.

**What this does not show.** Why half B's words do not come back on half A. Whether another dose or other questions
would carry them. Or whether layer 30's neurons say less on these questions than 31's.
