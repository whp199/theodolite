# The bars for another card class at layer 31 under kit 5: what is measured, by what rule, and what it decides

Fixed on 17 November 2026, before anything here was measured. The owner's word was "Prepare kit 5 for layer 31"
(16 November). The hashes are in SHA256SUMS beside this file, with `prereg_at` and `repo_commit`. Layers are counted
from 0: 31 here is 32 on the pages.

**Why.** Kit 5 is kit 4 with layer 31 added, taken with `match_wide`. `match_wide` passed its trial there on 16
November (experiments/results/census_wide_trials). A volunteer's run from a card that does not write the home card's
bits is taken if a few chunks, written again at home, stay within set distances of the home card's
(`census_volunteer.ACCEPT_OTHER_CLASS`, laid under `OTHER_CLASS_BARS`). Kit 4 measured those distances for its own
designs and layers (`../../kit4/`), not for `match_wide`, and not at layer 31. Where none were measured the intake
judges nothing (`accept_bars` gives None). So no run of another card class can be taken at 31 until this is measured.

`match_wide` differs from kit 4's designs in two ways that bear on the check.
- **The questions.** It asks twelve, where kit 4's designs ask break_rel's six. A whole layer's plan writes the first
  half's unpushed answers in batches of 64 and the second half's in batches of 63. Its chunk 0 holds 31 and a half
  units' answers, so a batch of 63 is what each of bicycle, walk, chair, kettle, sleep and lamp is left with. Chair's
  unpushed answer then differs from layer 45's in its log-probabilities, and six of the twelve were never written
  there. The intake's unpushed answers are layer 45's six (`volunteers/home_canaries/baselines.json`). A home-bits
  run at 31 would fail on chair, and six answers would go unchecked. Kit 5's code gives a design that asks other
  questions its own unpushed answers, from its trial run on the home card (`census_volunteer.DESIGN_BASELINES`).
  That they are what the kit's own plan writes is checked here (T1).
- **One layer.** A design at one layer has, under kit 4's script, 64 answers to one question (who) to measure on.
  Kit 5's script can write each layer's plan chunk 0 instead (`census_bars.py --chunk0`). Under `match_wide` at batch
  64 that is 768 answers to all twelve questions, at both signs. The reference is then the control's own chunk 0,
  and the control's first batch must be the home canary bit for bit.

**What is run** (`measure.sh`, about 35 minutes on the home RTX 5090 with nothing else on it), at layer 31 with kit
5's settings (`match_wide` at ±2.58 ceilings, batch 64, 48 word-pieces, the compiled unpacker, seed 7, 2,048
controls):
1. **The unpushed answers** (`census_ties.py --layer 31 --reference <the trial run at 31>`). The kit run's chunk 0,
   in the variants of 31 October:
   - the control at batch 64, as the kit writes it;
   - the arithmetic-only variants: batch 1, 2, 4, 8, 16, 24, 32, 40, 48 and 56, and batch 64 without the
     reduced-precision sums.

   Each of the twelve unpushed answers is set against the trial run's
   (`atlas_data/census/trials/wide/L31-match_wide-trial.jsonl`, written on the home card on 16 November; its sha256
   is in `reference.sha256`, and `measure.sh` refuses another file).
   Output in `experiments/results/census_unpushed_ties/kit5/`.
2. **The log-probability bars** (`census_bars.py --layers 31 --chunk0`). The kit run's chunk 0, in the thirteen
   variants of 31 October:
   - the control;
   - batch 16, 32 and 48, and batch 64 without the reduced-precision sums;
   - the push 1, 2, 5 and 10% longer, and the same shorter (the write ceiling scaled, as for kit 4's ceiling designs).

   Output in `../bars.json` and `../bars.txt`.

**The rules.**
- **T1, the unpushed answers.** The control's twelve unpushed answers must equal the trial run's bit for bit. Then
  those twelve are `match_wide`'s own (`DESIGN_BASELINES`, as kit 5's code names them). If any differs, nothing in
  this registration is concluded and no run at 31 is judged. The cause is found first.
- **T2, the near-tie bar.** The rule of 31 October, unchanged (`census_ties.py`). If no arithmetic-only variant parts
  at a piece the home card wrote with probability 0.6 or more, the bar stays 0.6 at 31. If one does, the bar for
  `match_wide` at 31 is the next multiple of 0.05 above the surest such piece. It goes into that design's entry in
  `OTHER_CLASS_BARS` (`baseline_tie`), not into `ACCEPT_OTHER_CLASS`.
- **B0, the instrument.** The control's first batch must be the home canary of layer 31 bit for bit
  (`volunteers/home_canaries/L31-match_wide.canary.jsonl`, taken under kit 5). Every near miss must move something.
  If either fails, nothing is derived.
- **B1, the bars.** The rule of 31 October, unchanged, on layer 31's chunk 0, applied to each of the median and the
  99th percentile of |difference|:
  - the floor F is the largest value among the arithmetic-only variants, at least 0.0001;
  - a near miss of size s is M(s), the smaller of its two signs' values, each sign needing 200 pieces to count;
  - the bar is sqrt(F x M(s)), to two figures, for the smallest s with M(s) at least 4 x F;
  - where no size qualifies, that bar is dropped.

  These are `match_wide`'s bars at 31, whatever they come out as: tighter than `match_ceil`'s at 32 to 37 (0.012 and
  0.25), looser, or dropped. Kit 4's R1, for its ceiling designs, as written. With one layer there is no pooled bar,
  so kit 4's R3 (a layer's own bar) does not arise.
- **B2, first word-pieces.** Kit 4's R4: the bar stays 0.90 if every arithmetic-only variant keeps at least 0.93 of
  first word-pieces. If the lowest keeps fewer, the bar is that lowest share less 0.05.
- **B3, if both log-probability bars are dropped.** Kit 4's R6: the spot check at 31 then rests on first word-pieces
  and the unpushed answers alone. No run of another card class is accepted at 31 on that alone until the owner has
  said so, knowing it.

**What the measurement decides.** `table` in `../bars.json` goes into `census_volunteer.OTHER_CLASS_BARS` as an entry
for `match_wide` at layer 31, copied by hand, nothing rounded or adjusted on the way. With it goes T2's `baseline_tie`
if T2 raised it. `scripts/check_volunteer_kit.py` holds the two to each other, as it does for kit 4's. Until it is
there, kit 5's intake judges no run of another card class at 31. A run of the home card's class is checked bit for bit
whatever this finds.

**What is not changed by this.** Kit 4's bars at 32 to 39, 51 and 52, and the bars of 31 October at 40 to 50. The
six unpushed answers every other design is set against. Kit 4's layers are taken under kit 5 exactly as under kit 4.
A claim taken under kit 4 is checked by kit 4's own file (`census_volunteer.earlier_kit`).

**What had been seen when this was fixed.**
- Kit 4's measurement (`../../kit4/bars.txt`).
  - Under `match_ceil` at 32 to 37 (384 answers a variant): floors 0.0050 and 0.1174; bars 0.012 and 0.25.
  - Each of those layers alone: medians' bars 0.011 to 0.014; 99th percentiles' 0.25 to 0.27 or dropped.
- The tie check of 31 October at layer 45 (`../../../census_unpushed_ties/ties.txt`): 21 of 66 arithmetic-only
  answers parted, the surest at 0.52.
- **The trial run at 31 against layer 45's six unpushed answers.** Who, orange, free, bicycle and walk are the same
  bit for bit. Chair writes the same pieces, but its log-probabilities differ. That was read on 17 November, and is
  why kit 5's code gives `match_wide` answers of its own.
- The stand-in's dry run of `census_bars.py --chunk0` (4 answers a batch; too few to derive anything).
- Kit 5's home canaries, taken before this was hashed. The canary of 31 is plan batch 0 (who's 64 answers). No
  variant of it.
- No answer of any other card under `match_wide`, and no arithmetic-only or near-miss variant of the home card under
  it.

**What this leaves untested.**
- Kit 4's caveats stand.
  - The variants are the home card's own arithmetic, and another card class may move log-probabilities in ways
    they do not span.
  - A near miss is the whole push scaled.
- Chunk 0 holds 32 units' answers: neurons 0 to 28 and three controls, the last of them half asked. Kit 4's
  measurement had fewer: one batch of the first units.
- `match_wide`'s one dose is `match_ceil`'s lower one. A run at 31 with `match_ceil`'s two doses is another job list
  and is refused by the intake's meta check, not by these bars.
