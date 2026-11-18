# The bars for another card class at layer 30 under kit 6: what is measured, by what rule, and what it decides

Fixed on 18 November 2026, before anything here was measured. The owner's word was "Push it and register the bars for
30" (18 November), after `match_things` passed its trial at layer 30. The hashes are in SHA256SUMS beside this file,
with `prereg_at` and `repo_commit`. Layers are counted from 0: 30 here is 31 on the pages. This is kit 5's
registration (`../../kit5/prereg/`) at another layer under another design, its rules unchanged; what differs is said.

**Why.** Kit 6 is kit 5 with layer 30 added, taken with `match_things`. `match_things` passed its trial there on 18
November (experiments/results/census_things_trial). A volunteer's run from a card that does not write the home card's
bits is taken if a few chunks, written again at home, stay within set distances of the home card's
(`census_volunteer.ACCEPT_OTHER_CLASS`, laid under `OTHER_CLASS_BARS`). Kits 4 and 5 measured those distances for their
own designs and layers, not for `match_things`, and not at layer 30. Where none were measured the intake judges nothing
(`accept_bars` gives None). So no run of another card class can be taken at 30 until this is measured. A run of the
home card's class is checked bit for bit whatever this finds.

`match_things` differs from kit 5's `match_wide` in three ways that bear on the check.
- **Its questions.** Twelve of three genres (how a thing works, three tips, an object): bicycle, walk, chair, kettle,
  sleep, lamp | clock, rice, vase, lock, money, blanket. Six were asked by no census run before its trial. A whole
  layer's plan writes the first nine questions' unpushed answers in batches of 64 and the last three's in batches of
  61, as its trial's plan did (`census.plan_batches` over both job lists). Kit 6's code gives the design its own
  unpushed answers, from its trial run on the home card (`census_volunteer.DESIGN_BASELINES`). That they are what the
  kit's own plan writes is checked here (T1).
- **Two doses.** ±2.42 and ±2.67 ceilings at 30, `match_ceil`'s, where `match_wide` has one. A near miss scales the
  push, so both doses move together, as under `match_ceil` in kit 4.
- **Forty-eight answers a unit**, twice `match_wide`'s. The plan's chunk 0 (`census_bars.py --chunk0`) at batch 64 is
  still 768 answers to all twelve questions, at both doses and both signs. It holds the twelve unpushed answers and 16
  units' answers (neurons 0 to 14, the last of them three quarters asked, and one control), where `match_wide`'s held
  32 units. The reference is the control's own chunk 0, and the control's first batch must be the home canary bit for
  bit.

**What is run** (`measure.sh`, about an hour on the home RTX 5090 with nothing else on it, as kit 5's took), at layer
30 with kit 6's settings (`match_things` at ±2.42 and ±2.67 ceilings, batch 64, 48 word-pieces, the compiled unpacker,
seed 7, 2,048 controls):
1. **The unpushed answers** (`census_ties.py --layer 30 --reference <the trial run at 30>`). The kit run's chunk 0, in
   the variants of 31 October:
   - the control at batch 64, as the kit writes it;
   - the arithmetic-only variants: batch 1, 2, 4, 8, 16, 24, 32, 40, 48 and 56, and batch 64 without the
     reduced-precision sums.

   Each of the twelve unpushed answers is set against the trial run's
   (`atlas_data/census/trials/things/L30-match_things-trial.jsonl`, written on the home card on 18 November; its
   sha256 is in `reference.sha256`, and `measure.sh` refuses another file).
   Output in `experiments/results/census_unpushed_ties/kit6/`.
2. **The log-probability bars** (`census_bars.py --layers 30 --chunk0`). The kit run's chunk 0, in the thirteen
   variants of 31 October:
   - the control;
   - batch 16, 32 and 48, and batch 64 without the reduced-precision sums;
   - the push 1, 2, 5 and 10% longer, and the same shorter (the write ceiling scaled, as for kits 4 and 5).

   Output in `../bars.json` and `../bars.txt`.

`census_bars.py` and `census_ties.py` are kit 5's registered files byte for byte (their sha256 are in both
registrations' SHA256SUMS).

**The rules.** Kit 5's, changed only where the layer, the design and its earlier bars are named.
- **T1, the unpushed answers.** The control's twelve unpushed answers must equal the trial run's bit for bit. Then
  those twelve are `match_things`'s own (`DESIGN_BASELINES`, as kit 6's code names them). If any differs, nothing in
  this registration is concluded and no run at 30 is judged. The cause is found first.
- **T2, the near-tie bar.** The rule of 31 October, unchanged (`census_ties.py`). If no arithmetic-only variant parts
  at a piece the home card wrote with probability 0.6 or more, the bar stays 0.6 at 30. If one does, the bar for
  `match_things` at 30 is the next multiple of 0.05 above the surest such piece. It goes into that design's entry in
  `OTHER_CLASS_BARS` (`baseline_tie`), not into `ACCEPT_OTHER_CLASS`.
- **B0, the instrument.** The control's first batch must be the home canary of layer 30 bit for bit
  (`volunteers/home_canaries/L30-match_things.canary.jsonl`, taken under kit 6). Every near miss must move something.
  If either fails, nothing is derived.
- **B1, the bars.** The rule of 31 October, unchanged, on layer 30's chunk 0, applied to each of the median and the
  99th percentile of |difference|:
  - the floor F is the largest value among the arithmetic-only variants, at least 0.0001;
  - a near miss of size s is M(s), the smaller of its two signs' values, each sign needing 200 pieces to count;
  - the bar is sqrt(F x M(s)), to two figures, for the smallest s with M(s) at least 4 x F;
  - where no size qualifies, that bar is dropped.

  These are `match_things`'s bars at 30, whatever they come out as: tighter than `match_wide`'s at 31 (0.013 and
  0.26), looser, or dropped. With one layer there is no pooled bar, so kit 4's R3 (a layer's own bar) does not arise.
- **B2, first word-pieces.** Kit 4's R4: the bar stays 0.90 if every arithmetic-only variant keeps at least 0.93 of
  first word-pieces. If the lowest keeps fewer, the bar is that lowest share less 0.05.
- **B3, if both log-probability bars are dropped.** Kit 4's R6: the spot check at 30 then rests on first word-pieces
  and the unpushed answers alone. No run of another card class is accepted at 30 on that alone until the owner has
  said so, knowing it.

**What the measurement decides.** `table` in `../bars.json` goes into `census_volunteer.OTHER_CLASS_BARS` as an entry
for `match_things` at layer 30, copied by hand, nothing rounded or adjusted on the way. With it goes T2's
`baseline_tie` if T2 raised it. `scripts/check_volunteer_kit.py` holds the two to each other, as it does for kits 4's
and 5's. Until it is there, kit 6's intake judges no run of another card class at 30.

**What is not changed by this.** Kits 4's and 5's bars at 31 to 39, 51 and 52, and the bars of 31 October at 40 to 50.
The six unpushed answers most designs are set against, and `match_wide`'s twelve. Kit 5's layers are taken under kit
6 exactly as under kit 5. A claim taken under kit 5 is checked by kit 5's own file (`census_volunteer.earlier_kit`).
Whether `match_things`'s doses are counted calibrated at 30 (`DOSES_CALIBRATED`), which is the owner's word and
changes no answer, bar or canary.

**What had been seen when this was fixed.**
- Kit 5's measurement at 31 under `match_wide` (`../../kit5/bars.txt`, 768 answers a variant): floors 0.0053 and
  0.1149; bars 0.013 and 0.26; first pieces kept at least 0.9635 by the arithmetic, so 0.90 stood. Its ties: 23 of
  132 arithmetic-only answers parted, the surest at 0.52, so 0.6 stood
  (`../../../census_unpushed_ties/kit5/ties.txt`).
- Kit 4's measurement (`../../kit4/bars.txt`). Under `match_ceil` at 32 to 37 (384 answers a variant): floors 0.0050
  and 0.1174; bars 0.012 and 0.25.
- **The trial run at 30 itself** (18 November), on 1,022 of the layer's neurons and 1,024 controls drawn with another
  seed (10): every one of its answers, its verdict, and its twelve unpushed answers. Its step 0
  (`census_things_trial/prereg/step0.txt` and `step0_round2.txt`): each question's unpushed answer, in the batch its
  run writes it in, holds at least 32 word-pieces with itself written alone (chair 46 and money 38 of 48, the rest all
  48).
- Kit 6's home canaries, taken before this was hashed. The canary of 30 is plan batch 0 (bicycle's 64 answers). No
  variant of it.
- No answer of any other card under `match_things`, and no arithmetic-only or near-miss variant of the home card under
  it.

**What this leaves untested.**
- Kits 4's and 5's caveats stand.
  - The variants are the home card's own arithmetic, and another card class may move log-probabilities in ways
    they do not span.
  - A near miss is the whole push scaled, both doses together.
- Chunk 0 holds 16 units' answers: neurons 0 to 14 (the last three quarters asked) and one control. Kit 5's held 32,
  with three controls.
- A run at 30 with any other design's job list is refused by the intake's meta check, not by these bars.
