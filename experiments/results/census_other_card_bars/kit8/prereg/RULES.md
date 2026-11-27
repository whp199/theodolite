# The bars for another card class at layers 55, 56 and 57 under kit 8: what is measured, by what rule, and what it decides

Fixed on 27 November 2026, before anything here was measured. Layers are counted from 0: 55, 56 and 57 here are 56, 57
and 58 on the pages. This is kit 7's registration (`../../kit7/prereg/`) at three other layers, its rules unchanged; what
differs is said.

**Why.** Kit 8 is kit 7 with layers 55, 56 and 57 taken with `match_things`. The design passed the dual-control `U2`
rollout walk at all three layers (`experiments/results/u2_rollout_trial/RESULT.md`, `scripts/u2_rollout_read.py`,
`1,022` fresh neurons/layer paired against `1,024 columns` seed=21 and `1,024 isotropic` seed=22 controls, clearing both
Track 2A `U2-T_dec` and Track 2B `U2-weight_words` on all three layers with `0 / 1,024` cross-pool false discoveries).
A volunteer's run from a card that does not write the home card's bits is taken if a few chunks, written again at home,
stay within set distances of the home card's (`census_volunteer.ACCEPT_OTHER_CLASS`, laid under `OTHER_CLASS_BARS`).
Kit 6 measured those distances for `match_things` at layer 30 alone, and kit 7 at layers 53 and 54. Where none were
measured the intake judges nothing (`accept_bars` gives None). So no run of another card class can be taken at 55, 56
or 57 until this is measured. A run of the home card's class is checked bit for bit whatever this finds.

`match_things` at 55, 56 and 57 differs from the design at 30 and at 53–54 in what its pushes do, not in what it asks:
- **The same questions and unpushed answers.** Twelve, the same twelve, in the same batch shapes (nine in batches of 64,
  the last three in 61). No push enters an unpushed answer, and the `u2_rollout` trials at 55, 56 and 57 wrote the same
  twelve as the trial at 30 bit for bit (`L55-match_things-u2-cols.jsonl`, `L56-...`, `L57-...`). `DESIGN_BASELINES`
  names the trial run at 30 for them. That they are what the kit's own plan writes at each layer is checked here (T1).
- **The pushes at 55–57.** ±1.95 and ±2.14 ceilings at 55, ±1.96 and ±2.11 at 56, ±1.55 and ±1.68 at 57 (`MATCH_CEIL`).
  Pushed answers go further from the unpushed path than at 30, so the home card's own arithmetic may move their
  log-probabilities differently, and so may a push a little off.
- **Three layers of one design.** Each layer's chunk 0 (`census_bars.py --chunk0`) at batch 64 is 768 answers: the
  twelve unpushed answers and 16 units' answers (neurons 0 to 14, the last three quarters asked, and one control), as at
  30, 53 and 54. The three layers are pooled as one group (2,304 answers a variant), as kit 7 pooled 53–54 and kit 4
  pooled `match_ceil`'s six layers, so kit 4's R3 (a layer's own bar) arises here. The reference is each layer's
  control's own chunk 0, and its first batch must be that layer's home canary bit for bit.

**What is run** (`measure.sh`, on the home RTX 5090 with nothing else on it, after `census_intake.py home-canaries` and
`measure_batches.py --hold-mib`), at layers 55, 56 and 57 with kit 8's settings (`match_things` at each layer's two
counts, batch 64, 48 word-pieces, the compiled unpacker, seed 7, 2,048 controls):
1. **The unpushed answers** (`census_ties.py --layer L --reference <the u2_rollout cols trial run at L>`), at 55, 56 and 57.
   Each kit run's chunk 0, in the variants of 31 October:
   - the control at batch 64, as the kit writes it;
   - the arithmetic-only variants: batch 1, 2, 4, 8, 16, 24, 32, 40, 48 and 56, and batch 64 without the
     reduced-precision sums.

   Each layer's twelve unpushed answers are set against its `u2_rollout` trial run's
   (`atlas_data/census/trials/u2_rollout/L55-match_things-u2-cols.jsonl`, `L56-...`, `L57-...`; their sha256 are in
   `reference.sha256`, and `measure.sh` refuses other files). Output in
   `experiments/results/census_unpushed_ties/kit8/L55/`, `.../L56/`, and `.../L57/`.
2. **The log-probability bars** (`census_bars.py --layers 55,56,57 --chunk0`). Each layer's chunk 0, in the thirteen
   variants of 31 October:
   - the control;
   - batch 16, 32 and 48, and batch 64 without the reduced-precision sums;
   - the push 1, 2, 5 and 10% longer, and the same shorter (the write ceiling scaled, as for kits 4 to 7).

   Output in `../bars.json` and `../bars.txt`.

`census_bars.py` and `census_ties.py` are kits 5's, 6's and 7's registered files byte for byte (their sha256 are in all
four registrations' SHA256SUMS).

**The rules.** Kit 7's, changed only where the layers are named and where three layers make a group.
- **T1, the unpushed answers.** At each layer the control's twelve unpushed answers must equal its `u2_rollout` trial
  run's bit for bit. Then those twelve are `match_things`'s own (`DESIGN_BASELINES`) at that layer. If any differs at
  any layer, nothing in this registration is concluded and no run at 55, 56 or 57 is judged. The cause is found first.
- **T2, the near-tie bar.** The rule of 31 October, unchanged (`census_ties.py`), over all three layers' arithmetic-only
  variants together. If none parts at a piece the home card wrote with probability 0.6 or more, the bar stays 0.6 at
  55, 56 and 57. If one does, the bar for `match_things` at 55, 56 and 57 is the next multiple of 0.05 above the surest
  such piece. It goes into that entry in `OTHER_CLASS_BARS` (`baseline_tie`), not into `ACCEPT_OTHER_CLASS`.
- **B0, the instrument.** Each layer's control's first batch must be that layer's home canary bit for bit
  (`volunteers/home_canaries/L55-match_things.canary.jsonl`, `L56-...`, and `L57-...`, taken under kit 8). Every near
  miss must move something. If either fails, nothing is derived.
- **B1, the bars.** Kit 4's rule, unchanged, on the group's pooled rows (55's, 56's and 57's chunk 0), applied to each of
  the median and the 99th percentile of |difference|:
  - the floor F is the largest value among the arithmetic-only variants, at least 0.0001;
  - a near miss of size s is M(s), the smaller of its two signs' values, each sign needing 200 pooled pieces to count;
  - the bar is sqrt(F x M(s)), to two figures, for the smallest s with M(s) at least 4 x F;
  - where no size qualifies, that bar is dropped.

  These are `match_things`'s bars at 55, 56 and 57, whatever they come out as: tighter than its bars at 30 or 53–54,
  looser, or dropped.
- **B1', a layer of its own** (kit 4's R3). A layer whose own floor (from its 768 answers alone; an arithmetic-only
  variant with fewer than 200 pieces there does not count) is at or above the group's bar for a statistic takes that
  statistic's bar from its own rows by the rule, which may be a dropped bar. Its entry stands before the group's.
- **B2, first word-pieces.** Kit 4's R4, on the group: the bar stays 0.90 if every arithmetic-only variant keeps at
  least 0.93 of first word-pieces. If the lowest keeps fewer, the bar is that lowest share less 0.05.
- **B3, if both log-probability bars are dropped.** Kit 4's R6: the spot check at 55, 56 and 57 then rests on first
  word-pieces and the unpushed answers alone. No run of another card class is accepted there on that alone until the
  owner has said so, knowing it.

**What the measurement decides.** `table` in `../bars.json` goes into `census_volunteer.OTHER_CLASS_BARS` as entries
for `match_things` at 55, 56 and 57 (the group's, and a layer's own where B1' gives one), copied by hand, nothing rounded
or adjusted on the way. With them goes T2's `baseline_tie` if T2 raised it. `scripts/check_volunteer_kit.py` holds the
two to each other, as it does for kits 4 to 7. Until they are there, kit 8's intake judges no run of another card class
at 55, 56 or 57.

**What is not changed by this.** Kits 4's to 7's bars at 30 to 39, 51 to 54, and the bars of 31 October at 40 to 50.
`match_things`'s bars at 30 stand as kit 6 measured them, and at 53–54 as kit 7 measured them. The six unpushed answers
most designs are set against, `match_wide`'s twelve and `match_things`'s twelve. Kit 7's layers are taken under kit 8
exactly as under kit 7. A claim taken under kit 7 is checked by kit 7's own file (`census_volunteer.earlier_kit`). Whether
`match_things`'s doses are counted calibrated at 55, 56 and 57 (`DOSES_CALIBRATED`), which is the owner's word and
changes no answer, bar or canary. Whether kit 8 is cut, which is the owner's word too.

**What had been seen when this was fixed.**
- Kit 7's measurement at 53 and 54 under `match_things` (`../../kit7/bars.txt`, 768 answers a layer pooled): floors
  0.0037 and 0.1141; bars 0.0091 (built to catch a push 5% off) and 0.26 (10% off); first pieces kept at least 0.9961
  by the arithmetic, so 0.90 stood. Its ties: 24 of 132 arithmetic-only answers parted at each layer, the surest at
  0.5194, so 0.6 stood (`../../../census_unpushed_ties/kit7/L53/ties.txt` and `L54/ties.txt`).
- **The `u2_rollout` trial runs at 55, 56 and 57** (26–27 November 2026), on 1,022 fresh neurons of each layer paired
  against 1,024 `columns` controls (`seed=21`) and 1,024 `isotropic` controls (`seed=22`): every layer passed both
  Track 2A (`U2-T_dec`, `4/4` gates) and Track 2B (`U2-weight_words`, `2/2` gates), with `0 / 1,024` cross-pool false
  discoveries, and their twelve unpushed answers match the trial at 30's bit for bit.
- No answer of any other card under `match_things` at 55, 56 or 57, and no arithmetic-only or near-miss variant of the
  home card there.

**What this leaves untested.**
- Kits 4's to 7's caveats stand.
  - The variants are the home card's own arithmetic, and another card class may move log-probabilities in ways
    they do not span.
  - A near miss is the whole push scaled, both doses together.
- Each layer's chunk 0 holds 16 units' answers: neurons 0 to 14 (the last three quarters asked) and one control.
- A run at 55, 56 or 57 with any other design's job list is refused by the intake's meta check, not by these bars.
