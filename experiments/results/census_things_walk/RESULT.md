# The walk with match_things: 53 and 54 pass, 29 and 55 fail

Registered in `prereg/`: hashed at 16:18:23Z on 18 November 2026, committed as 7f558a7, and pushed at 16:23:53Z, before
the walk began at 16:24:00Z. Its files still match `SHA256SUMS`, and there are no amendments. Every run was written
from the clean checkout of 2989b14, the commit in `prereg/repo_commit`; each run's meta records it. Each was read by
the registration's own `read.sh`. Layers are counted from 0, as the trials and the board count them: 53 here is 54 on
the pages. `summary.json` has every figure. `collect.py` copies the verdicts from
`atlas_data/census/trials/things_walk`, and `chain.log` is the walk's own log. `reported.py` computes the two
report-only readings below that the reader does not make, and `reported.txt` is its output.

## What was run

The walk took four layers, in its registered order, and stopped where each arm failed:

| layer | written | sessions | verdict | read at |
|---|---|---|---|---|
| 53 (up) | 16:24Z to 18:50Z, 18 Nov | 1 | **pass** | 18:55Z |
| 29 (down) | 18:55Z to 21:47Z, 18 Nov | 3 | fail: the down arm stops | 21:52Z |
| 54 (up) | 21:52Z to 00:27Z, 18 to 19 Nov | 1 | **pass** | 00:30Z |
| 55 (up) | 00:30Z to 02:53Z, 19 Nov | 2 | fail: the up arm stops | 02:56Z |

Layers 28 to 25 and 56 to 58 were not trialled, and nothing is said of them. Each trial wrote all 98,220 answers, at
9.5 to 11.5 a second.

**Three stops, each continued row for row.** The home RTX 5090 also drives the desktop. Three times the graphics
driver's watchdog judged it locked (kernel Xid 8, "RC watchdog: GPU is probably locked", after 7 seconds) and ended
the run's work on it:
- layer 29 at 19:47:28Z, with 31,488 answers written;
- layer 29 again at 21:24:46Z, with 91,904;
- layer 55 at 02:27:19Z, with 80,768.

The first two came within a minute of a desktop app starting a graphics process on the card. No app start is logged
before the third. After the first and the third, `walk.sh` began the trial once more after 30 seconds, and the run
went on from its last whole batch. At 29's second stop `walk.sh` ended, as it is written to do on a second stop at one
layer, and it was begun again by hand at 21:35:18Z. `RULES.md` ("Pauses") says a trial stopped part way is continued
row for row, and sets no limit on how often. Each continued run found no half-written batch to cut. The final files of
29 and 55 hold 98,220 distinct answers each, none written twice. Each verdict's G5 found every answer of the job list
present.

## The reading

At 48 word-pieces, at layer 45's bars. A layer passes when N1 and N3 both pass. Figures under their bar are in bold.

| layer | reading | lead | beyond text | words carry A→B | B→A | new questions |
|---|---|---|---|---|---|---|
| | bars | 0.60 | 0.58 | 0.57 | 0.57 | 0.56 |
| 53 | N1 whole | 0.751 | 0.745 | 0.591 | 0.598 | 0.751 |
| | N3 covered (995 of 1,022) | 0.748 | 0.741 | 0.588 | 0.593 | 0.748 |
| | N2 the lower dose alone (±2.24) | 0.656 | 0.696 | **0.536** | **0.550** | 0.631 |
| | *reported:* the upper dose alone (±2.42) | 0.745 | 0.666 | **0.545** | **0.561** | 0.707 |
| 29 | N1 whole | **0.574** | 0.599 | **0.528** | **0.520** | 0.592 |
| | N3 covered (990 of 1,022) | **0.570** | 0.602 | **0.522** | **0.516** | 0.589 |
| | *reported:* the lower dose alone (±2.10) | **0.569** | 0.592 | **0.530** | **0.516** | 0.581 |
| | *reported:* the upper dose alone (±2.32) | **0.532** | **0.569** | **0.512** | **0.514** | **0.550** |
| 54 | N1 whole | 0.712 | 0.714 | 0.627 | 0.620 | 0.709 |
| | N3 covered (975 of 1,022) | 0.705 | 0.707 | 0.616 | 0.610 | 0.703 |
| | N2 the lower dose alone (±2.18) | 0.645 | 0.656 | 0.575 | 0.583 | 0.597 |
| | *reported:* the upper dose alone (±2.34) | 0.703 | 0.688 | 0.593 | 0.591 | 0.662 |
| 55 | N1 whole | 0.685 | 0.675 | **0.518** | **0.539** | 0.716 |
| | N3 covered (982 of 1,022) | 0.688 | 0.677 | **0.514** | **0.536** | 0.721 |
| | *reported:* the lower dose alone (±1.95) | **0.577** | 0.625 | **0.494** | **0.528** | **0.541** |
| | *reported:* the upper dose alone (±2.14) | 0.675 | 0.612 | **0.491** | **0.519** | 0.663 |

N2 counts only at a layer that passes N1 and N3, so at 29 and 55 the lower dose alone is a report. The G5 sanity checks
pass at all four layers (read gain 1.03 to 1.07), and so does G5b on each whole reading. G5b is marked FAIL in the
report-only readings, as at 30: it compares a reading with the run's ranking of every answer at both doses, which a
reading of only some answers cannot match.

- **53 passes**, every figure by 0.017 or more. The lower dose alone fails on the words carrying, both ways, so N2
  fails. Two days earlier `soft_ceil` failed here (lead 0.694, words carrying 0.574 and 0.557) on other neurons, other
  controls and six other questions.
- **29 fails** on the lead and on the words carrying. Beyond text and the new questions alone pass. No neuron has
  q ≤ 0.05, and 8 beat every control, where about 1 would by chance.
- **54 passes** with the walk's strongest words carrying (0.627 and 0.620). N2 passes too, narrowly: its words carry
  0.575 and 0.583, and its new questions 0.597. Every figure has a standard error near 0.013, so 0.575 is within one
  standard error of the bar.
- **55 fails** on the words carrying alone. Its lead (0.685), beyond text (0.675) and new questions (0.716) pass. The
  words picked on one half of the questions score near chance on the other at either dose alone (0.491 to 0.528).

## Against what was said beforehand

- "53 passes N1 and N3 about 1 in 3." It passed.
- "29 passes about 1 in 3." It failed.
- "Given a pass at the layer before it, each next layer passes about as likely as not." 54 passed after 53, and 55
  failed after 54.
- "The upward arm reaching 58, or the downward arm reaching 25: each under 1 in 20." Neither did.
- "N1's lead between 0.62 and 0.72 at 53." **Wrong**: 0.751, over the range.
- "Between 0.58 and 0.64 at 29." **Wrong**: 0.574, just under it.
- "The share covered: 90 to 98 neurons in 100 at 53 and 29." Right: 97.4 and 96.9.
- "The text, at 29: it loses under 0.005 of its distinct 4-grams at either dose, as at 30." **Wrong at the minus
  side**: the neurons' text loses 0.006 at −2.10 and 0.008 at −2.32, and 0.004 and 0.005 at the plus side. The
  controls' loses 0.003 at every dose.
- "The text, at 53: both doses lose 0.10 to 0.30 of them." Right: 0.110 and 0.113 at the lower dose, 0.257 and 0.267
  at the upper.
- "N2: at a layer that passes N1 and N3, the lower dose alone passes as likely as not." It failed at 53 and passed at
  54.
- "At 25 to 29 the strongest controls write other languages, as at 30 to 37." Right at 29: Portuguese and Indonesian.
- "At 53 to 58 no control stands more than 5 standard deviations over the controls' median." **Wrong**: three at 53
  (11.3, 5.9, 5.3), three at 54 (7.3, 5.7, 5.2) and two at 55 (7.3, 5.5). See below.
- The predictions about 25 to 28 (that they fail on text survival as often as on the words carrying, and that fewer
  than 90 in 100 are covered) were not tested, since no trial reached them.

## Reported, gating nothing

- **The twelve unpushed answers** are the trial at 30's bit for bit at all four layers: the same word-pieces and the
  same log-probabilities, 12 of 12 each. That was expected, since no push enters them.
- **Cut to 32 word-pieces**, 53, 29 and 55 fail. 54 still passes (lead 0.693; words carrying 0.617 and 0.589), but the
  length rule does not take the shorter length there: the words carrying fall by 0.030. Every kit has kept 48.
- **The share covered**: 97.4, 96.9, 95.4 and 96.1 in 100 at 53, 29, 54 and 55.
- **The text.** At 53 to 55 the neurons' text loses 0.09 to 0.12 of its distinct 4-grams at the lower dose and 0.22 to
  0.27 at the upper. The controls' text loses about as much or more: 0.11 to 0.20 and 0.26 to 0.43. 99.9 to 100 in
  100 answers run to the 48-piece cut there. At 29 the text barely moves (above), and 99.2 to 99.9 in 100 answers run
  to the cut.
- **Neurons that stand out singly**, as the reader counts them:

  | layer | beat every control | q ≤ 0.05 | with the five strongest controls set aside | strong on both halves (p ≤ 0.05 each; 2.6 by chance) |
  |---|---|---|---|---|
  | 53 | 52 | 187 | 198 | 173 |
  | 29 | 8 | 0 | 25 | 54 |
  | 54 | 196 | 265 | 282 | 235 |
  | 55 | 84 | 114 | 118 | 119 |

  About one neuron a layer would beat every control by chance. The q ≤ 0.05 counts from the run's units file
  (`reported.py`) match the reader's at every layer.
- **The controls that stand out most.** These were computed after the reading from each run's units file, as
  `census_L34_whole/read.sh` computes them:
  - At 29 the four strongest write Portuguese ("aqui para estão três como funciona") and Indonesian ("saya memberikan
    untuk adalah"), 6.7 to 7.6 standard deviations over the controls' median. 21 neurons stand more than five over it.
  - At 53 to 55 the strongest write "thought" over and over, run together with other words, and "idea" and
    "depending". The strongest at 53 stands 11.3 standard deviations out. 152, 230 and 101 neurons stand more than
    five over it at 53, 54 and 55.
  - What a random push of this size does to make the model write "thought" is not looked into here.
- **Three questions a half.** At 53 and 55 the words carry under the bar both ways at three questions a half; at 54
  they pass both ways (0.589 to 0.621). At those three layers the leads hold at 0.70 to 0.77. At 29 the leads are 0.57
  and 0.58 (`summary.json`).

## What it decides (as registered)

- **53 and 54 pass N1 and N3.** Each may be offered to volunteers under a kit that carries `match_things` there.
  First, its other card classes' bars need a registration of their own, as kit 6's at 30 had. A kit is the owner's
  word, and so is widening `DOSES_CALIBRATED` to it. Until then the board marks each "no kit carries it yet: not
  taken".
- **54 also passes N2.** So the lower dose alone may be made a design of its own there: 24 answers a neuron, about 17
  hours a layer on a 5090. It would need its own step 0, its own bars and the owner's word before any kit offers it.
  Nothing here builds it.
- **29 and 55 fail.** No whole layer is taken at either with `match_things` under this registration. The downward arm
  stopped at 29, so 25 to 28 were not trialled. The upward arm stopped at 55, so 56 to 58 were not trialled. The
  board marks 29 and 55 "its trial failed (match_things): not taken".
- What to try past either end, or again at 29 and 55, is the owner's choice. Any such trial would be registered anew,
  on neurons this walk did not ask.

**What a pass here would not show** (`RULES.md`): that 53 and 54's neurons answer the census's other questions as they
answer these, since a layer taken with this design is asked none of who, orange or free; or what a neuron does when
it fires. At 53 and 54 the push is longer than the stream's median length (1.23 to 1.48 of it), so a pass shows what
a push that size does.
