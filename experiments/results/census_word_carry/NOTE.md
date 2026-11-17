# Do a neuron's words fail to carry between the halves because the halves ask different things?

17 November 2026. Exploratory: read again from eighteen trials already read and written up, on the CPU, in about two
minutes. Nothing here is a registration and nothing in it decides a layer. Layers count 0–59, as the trials' files
and the commands do.

## The question

Layers 30 and 53 fail their trials on gate G3: a unit's top five words picked on one half of the questions, scored on
the other half (an AUC of neurons over controls), must reach 0.57 both ways. The halves are not alike. Half A asks about
the model and about colours (who, orange, free; at twelve questions also self, green, interesting), half B about
objects and everyday tasks (bicycle, walk, chair; also kettle, sleep, lamp). If a neuron's words came back on another
question of the same half and not on the other half's, its push would bring words that depend on the kind of question,
and no number of questions or second dose would mend G3. That would end the push reading at 31 going down.

## What was done

`carry.py` scores the same statistic on smaller, matched sets of questions:

- **2to1**, every trial: words picked on two questions of a half, scored on one other question, of the same half
  ("within") or of the other ("between"). Every such cell: 6 within and 18 between at six questions, 120 and 180 at
  twelve.
- **3to3**, the twelve-question trials: picked on three, scored on three others. Within: each half split in two, both
  ways (40 cells); between: 40 cells drawn at random (seed 0).

Each on each dose level alone and on the gate's doses together. Scored on one whole half against the other it gives the
registered figures back exactly (30 `match_wide` 0.5555/0.5503, 30 `match_both` 0.5727/0.5449, 53 `soft_ceil`
0.5736/0.5568). Each cell's AUC has a standard error near 0.013. `collect.py` writes `summary.txt` and `summary.json`.

Beside it, for the count of neurons that can be read one at a time: neurons with p ≤ 0.05 on each half's mass alone,
against the same count among the controls (leave-one-out, scaled to 1,022), so chance is read off the controls rather
than taken as 1,022 × 0.05²; and from each run's units file, how far its strongest controls stand out and the q ≤ 0.05
count with the five strongest set aside, as the whole-layer write-ups read it.

## What it says

**The halves are not what stops the words.** On all eighteen trials, passed and failed, words carry as well between
the halves as within one: within minus between is −0.014 to +0.017 over the 55 readings, and −0.005 to
+0.008 at 30. At 30 the carry is faint whichever way it is read: 0.52 to 0.53 picked on two questions and scored on
one, 0.52 to 0.55 on three and three, where 31 under `match_wide` reads 0.530 and 0.555 and the passing `match_ceil`
layers 32–36 read 0.56 to 0.62 on 2to1 with both doses. At 53 the lower dose carries 0.550 within, 0.553 between; the upper dose
0.482 and 0.489, under the controls, both ways (its controls' text breaks more than its neurons').

**What differs at 30 is the questions the words come back on.** Whichever half the words were picked on, half A's
questions take them up less than half B's: at 30 under `match_both`, 0.533 against 0.554 on three-and-three reads, and
every half-A question ranks below every half-B question on 2to1 (who 0.522 the lowest, kettle 0.538 the
highest). That is why B→A fails there and A→B passes (0.545 and 0.573 registered). Under `match_wide` the same holds,
fainter (0.522 against 0.534). At 31 under `match_wide` the halves take words up alike (0.556 and 0.553). "Who are
you?" is the weakest taker at most layers (at 33, 0.554 against 0.59 to 0.61 for the other five).

**The harder dose does not carry more at 30.** Under `match_both` the upper dose alone carries 0.518 to 0.527, the lower
0.519 to 0.533. At 31–36 under `match_ceil` the two doses also carry alike.

**At 30 no neuron is named, for the same reason none is at 31 under `match_wide` or at 34–36.** The count the trials
register (`named`) needs q ≤ 0.05, which one control far out holds back:

| trial | gates | strong on both halves (chance) | q ≤ 0.05 | the same, five strongest controls set aside | strongest controls, in SD of the controls |
|---|---|---|---|---|---|
| 30 `match_wide` | fail | 32 (4.0) | 0 | 65 | 12.3, 4.4, 4.4, 4.0, 3.7 |
| 30 `match_both` | fail | 48 (14.0) | 0 | 0 | 12.2, 8.8, 8.3, 7.4, 7.0 |
| 31 `match_wide` | pass | 82 (7.0) | 0 | 126 | 10.1, 7.0, 5.0, 5.0, 4.9 |
| 35 `match_ceil` | pass | 100 (15.0) | 0 | 25 | 9.6, 7.7, 6.7, 6.4, 6.3 |
| 36 `match_ceil` | pass | 81 (11.0) | 0 | 0 | 10.3, 9.1, 8.7, 8.2, 8.1 |
| 53 `soft_ceil` | fail | 89 (6.0) | 154 | 171 | 3.2, 3.0, 2.8, 2.5, 2.5 |
| 52 `soft_ceil` | pass | 206 (8.0) | 301 | 329 | 3.3, 3.1, 2.9, 2.8, 2.8 |

The far-out control at 30 under `match_wide` writes Spanish (“una que para aquí tienes”), at 31 Turkish and Spanish, as
the strongest controls at 35–37 write other languages. Read against the controls, 30 under `match_wide` has eight
times chance strong on both halves (36 has seven), and with the five strongest controls set aside more neurons at
q ≤ 0.05 than 35 or 36. Under `match_both`, whose neurons and controls are other draws, five controls stand 7 to 12 SD out and the
count stays 0. Above 52 the strongest controls stand near 3 SD and nothing is held back.

## What it changes

1. The fear that layer 30's words depend on the kind of question, and so that the push reading ends at 31, is not
   borne out: they carry as well between unlike questions as between alike ones.
2. G3 fails at 30 on the questions the words are scored on. Half A's questions, about the model and colours, take a
   push's words up less there than half B's. A design whose questions all take words up as half B's do might carry
   about as half B does (0.554 on three-and-three, 0.573 registered A→B), at the bar and not clearly over it. Those
   questions would be picked from 30's own trial, so such a design could only be judged on fresh neurons.
3. A harder push is not indicated by these trials: 1.4 nats carried no more than 1.0 at 30. A ladder past 1.4 remains
   untried.
4. "30 names none" is true of the count the trials register, and equally true of 31 under `match_wide` and of 34–36,
   which passed their gates. Whether a gate on the yield of single neurons would take 30 depends on how it treats
   controls that stand far out, the owner's open question on which per-neuron count the ceiling layers report. Above
   52 the question does not arise: 53 names 117, and its strongest control stands 3.2 SD out.
