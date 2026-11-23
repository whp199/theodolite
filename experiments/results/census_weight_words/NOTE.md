# Reading a trial on each neuron's weight-lens words, fixed before its answers are read (exploratory)

22 November 2026, on the owner's "Let's try both of those recommendations", after a volunteer's comment on the board
(issue #1, 7084903662: on Gemma 4 12B, the write direction read through the final norm and the unembedding holds a
neuron's census words for 93% of significant neurons at its layer 37 and 7% at its layer 24; at the 31B's top layers,
they suggested, a weight-only reading "might then be the better instrument"). **Unregistered and read after the
trials it reads were seen; it gates nothing.** `explore.py` is the reader, `explore.txt` its output.

## What it asks

The census picks each neuron's words from its pushed answers and then asks whether they carry from one half of the
questions to the other (G3). At 55 they did not (0.518 and 0.539 against 0.57) though the lead passed (0.685). This
reads the same answers on words fixed before any answer is looked at: each neuron's naked-eye words, the first ten
word-pieces its write direction names up and down when read straight through the final norm and the unembedding
(`index/lens.npz`, `eye_pos` and `eye_neg`, the labels the atlas already shows). A unit's score for a neuron is the
share of its pushed-up answers' word-pieces among that neuron's up words, with its pushed-down answers on the down
words, averaged. Each neuron's own score is set against all 1,024 controls' scores on the same words. Nothing is
picked from the answers, so each half of the questions is a separate test of the same fixed claim, and so is each
dose. The null for "its own words" is each neuron's answers read on another neuron's words (a fixed shuffle).

## What it read (`explore.txt`)

| layer (match_things trial) | lead, every answer (shuffled) | half A | half B | q ≤ 0.05 | census: lead, words carried A→B, B→A, q ≤ 0.05 |
|---|---|---|---|---|---|
| 29 | 0.533 (0.506) | 0.532 | 0.526 | 0 | 0.574, 0.528, 0.520, 0 (failed) |
| 30 | 0.549 (0.506) | 0.538 | 0.543 | 0 | 0.618, 0.582, 0.587, 0 (passed) |
| 53 | 0.746 (0.510) | 0.722 | 0.711 | 464 | 0.751, 0.591, 0.598, 187 (passed) |
| 54 | 0.782 (0.501) | 0.756 | 0.743 | 545 | 0.712, 0.627, 0.620, 265 (passed) |
| 55 | 0.700 (0.512) | 0.677 | 0.676 | 381 | 0.685, 0.518, 0.539, 114 (failed) |
| 29, `edge_things` (23 Nov; ±2.64, ±3.73) | 0.568 (0.495) | 0.558 | 0.562 | 0 | 0.473, 0.488, 0.488, 0 (failed) |

- **At 53 to 55 the fixed words read a clear lead on each half alone**, 0.68 to 0.76 against about 0.51 shuffled, and
  at each dose alone (0.675 to 0.766). At 55, where the census's picked words did not carry, the fixed ones lead on
  both halves by 0.677 and 0.676, a little under 53 and 54. Across neurons, a neuron's standing on half A ranks with
  its standing on half B at rho 0.68 to 0.73 at 53 to 55.
- **At 29 and 30 they read almost nothing** (0.53 to 0.55; no neuron at q ≤ 0.05), whether the census passed there (30)
  or failed (29). On the harder `edge_things` trial at 29 (read the same way after it was written) they read a little more,
  0.568, and 0.575 at its upper dose, while the census's own reading of it fell under chance
  (`census_edge_trial/RESULT.md`); still no neuron at q ≤ 0.05. There the naked eye does not name what a push brings out, as on the 12B's layer 24.
- About half the neurons at 53 to 55 bring out any of their own words at all (0.46 to 0.61), against 0.10 to 0.13 of
  the controls on the same words.

## What it does and does not say

- It says that at 53 to 55 the labels the atlas already shows describe what pushing a neuron does, on both halves of
  the questions, and that the census's miss at 55 is not a miss of those words. Together with the census words' own
  reading (`census-walk-stuck`, 22 November: at 55 the words that come out on one half's questions only are almost
  never weight-lens words), the failure at 55 is in what the census picks, not in the push.
- At the top layers a push along a direction raises the logits of the word-pieces that direction names almost by
  construction. A pass here is a narrower claim than the census's: that a neuron's own unembedding words come out when
  it is pushed, not what else it does, nor anything it does where its words are not what the unembedding names.
- It was read after all five trials were seen, on the same answers. Used to judge a layer, it would need its own
  registration and bars, and fresh answers: a trial at 55 (and 56 to 58) on fresh neurons, read by this reader with
  gates fixed beforehand. Whether a layer read only this way counts as taken is the owner's word.
