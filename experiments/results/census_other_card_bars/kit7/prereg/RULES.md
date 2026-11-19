# The bars for another card class at layers 53 and 54 under kit 7: what is measured, by what rule, and what it decides

Fixed on 19 November 2026, before anything here was measured. The owner's word was "Register the bars for 53 and 54"
(19 November), after `match_things` passed its walk's trials at both. The hashes are in SHA256SUMS beside this file,
with `prereg_at` and `repo_commit`. Layers are counted from 0: 53 and 54 here are 54 and 55 on the pages. This is kit
6's registration (`../../kit6/prereg/`) at two other layers, its rules unchanged; what differs is said.

**Why.** Kit 7 is kit 6 with layers 53 and 54 taken with `match_things`. The design passed its trials there on 18 and
19 November (experiments/results/census_things_walk). A volunteer's run from a card that does not write the home
card's bits is taken if a few chunks, written again at home, stay within set distances of the home card's
(`census_volunteer.ACCEPT_OTHER_CLASS`, laid under `OTHER_CLASS_BARS`). Kit 6 measured those distances for
`match_things` at layer 30 alone. Where none were measured the intake judges nothing (`accept_bars` gives None). So no
run of another card class can be taken at 53 or 54 until this is measured. A run of the home card's class is checked
bit for bit whatever this finds.

`match_things` at 53 and 54 differs from the design at 30 in what its pushes do, not in what it asks:
- **The same questions and unpushed answers.** Twelve, the same twelve, in the same batch shapes (nine in batches of 64,
  the last three in 61). No push enters an unpushed answer, and the walk's trials at 53 and 54 wrote the same twelve as
  the trial at 30 bit for bit (`census_things_walk/reported.txt`). `DESIGN_BASELINES` names the trial run at 30 for
  them. That they are what the kit's own plan writes at each layer is checked here (T1).
- **Larger pushes.** ±2.24 and ±2.42 ceilings at 53, ±2.18 and ±2.34 at 54 (`MATCH_CEIL`), 1.23 to 1.48 of the
  stream's median length, where 30's were about half of it. At 53 and 54 the neurons' text lost 0.11 to 0.27 of its
  distinct 4-grams in the walk; at 30, at most 0.003. Pushed answers there go further from the unpushed path, so the
  home card's own arithmetic may move their log-probabilities more than at 30, and so may a push a little off.
- **Two layers of one design.** Each layer's chunk 0 (`census_bars.py --chunk0`) at batch 64 is 768 answers: the
  twelve unpushed answers and 16 units' answers (neurons 0 to 14, the last three quarters asked, and one control), as at
  30. The two layers are pooled as one group (1,536 answers a variant), as kit 4 pooled `match_ceil`'s six layers, so
  kit 4's R3 (a layer's own bar) arises here. The reference is each layer's control's own chunk 0, and its first batch
  must be that layer's home canary bit for bit.

**What is run** (`measure.sh`, about two and a half hours on the home RTX 5090 with nothing else on it), at layers 53
and 54 with kit 7's settings (`match_things` at each layer's two counts, batch 64, 48 word-pieces, the compiled
unpacker, seed 7, 2,048 controls):
1. **The unpushed answers** (`census_ties.py --layer L --reference <the walk's trial run at L>`), at 53 and then at 54.
   Each kit run's chunk 0, in the variants of 31 October:
   - the control at batch 64, as the kit writes it;
   - the arithmetic-only variants: batch 1, 2, 4, 8, 16, 24, 32, 40, 48 and 56, and batch 64 without the
     reduced-precision sums.

   Each layer's twelve unpushed answers are set against its walk trial run's
   (`atlas_data/census/trials/things_walk/L53-match_things-trial.jsonl` and `L54-...`, written on the home card on 18
   November; their sha256 are in `reference.sha256`, and `measure.sh` refuses other files). Output in
   `experiments/results/census_unpushed_ties/kit7/L53/` and `.../kit7/L54/`.
2. **The log-probability bars** (`census_bars.py --layers 53,54 --chunk0`). Each layer's chunk 0, in the thirteen
   variants of 31 October:
   - the control;
   - batch 16, 32 and 48, and batch 64 without the reduced-precision sums;
   - the push 1, 2, 5 and 10% longer, and the same shorter (the write ceiling scaled, as for kits 4 to 6).

   Output in `../bars.json` and `../bars.txt`.

`census_bars.py` and `census_ties.py` are kits 5's and 6's registered files byte for byte (their sha256 are in all
three registrations' SHA256SUMS).

**The rules.** Kit 6's, changed only where the layers are named and where two layers make a group.
- **T1, the unpushed answers.** At each layer the control's twelve unpushed answers must equal its walk trial run's bit
  for bit. Then those twelve are `match_things`'s own (`DESIGN_BASELINES`) at that layer. If any differs at either
  layer, nothing in this registration is concluded and no run at 53 or 54 is judged. The cause is found first.
- **T2, the near-tie bar.** The rule of 31 October, unchanged (`census_ties.py`), over both layers' arithmetic-only
  variants together. If none parts at a piece the home card wrote with probability 0.6 or more, the bar stays 0.6 at
  53 and 54. If one does, the bar for `match_things` at 53 and 54 is the next multiple of 0.05 above the surest such
  piece. It goes into that entry in `OTHER_CLASS_BARS` (`baseline_tie`), not into `ACCEPT_OTHER_CLASS`.
- **B0, the instrument.** Each layer's control's first batch must be that layer's home canary bit for bit
  (`volunteers/home_canaries/L53-match_things.canary.jsonl` and `L54-...`, taken under kit 7). Every near miss must move
  something. If either fails, nothing is derived.
- **B1, the bars.** Kit 4's rule, unchanged, on the group's pooled rows (53's and 54's chunk 0), applied to each of
  the median and the 99th percentile of |difference|:
  - the floor F is the largest value among the arithmetic-only variants, at least 0.0001;
  - a near miss of size s is M(s), the smaller of its two signs' values, each sign needing 200 pooled pieces to count;
  - the bar is sqrt(F x M(s)), to two figures, for the smallest s with M(s) at least 4 x F;
  - where no size qualifies, that bar is dropped.

  These are `match_things`'s bars at 53 and 54, whatever they come out as: tighter than its bars at 30 (0.013 and
  0.26), looser, or dropped.
- **B1', a layer of its own** (kit 4's R3). A layer whose own floor (from its 768 answers alone; an arithmetic-only
  variant with fewer than 200 pieces there does not count) is at or above the group's bar for a statistic takes that
  statistic's bar from its own rows by the rule, which may be a dropped bar. Its entry stands before the group's.
- **B2, first word-pieces.** Kit 4's R4, on the group: the bar stays 0.90 if every arithmetic-only variant keeps at
  least 0.93 of first word-pieces. If the lowest keeps fewer, the bar is that lowest share less 0.05.
- **B3, if both log-probability bars are dropped.** Kit 4's R6: the spot check at 53 and 54 then rests on first
  word-pieces and the unpushed answers alone. No run of another card class is accepted there on that alone until the
  owner has said so, knowing it.

**What the measurement decides.** `table` in `../bars.json` goes into `census_volunteer.OTHER_CLASS_BARS` as entries
for `match_things` at 53 and 54 (the group's, and a layer's own where B1' gives one), copied by hand, nothing rounded
or adjusted on the way. With them goes T2's `baseline_tie` if T2 raised it. `scripts/check_volunteer_kit.py` holds the
two to each other, as it does for kits 4 to 6. Until they are there, kit 7's intake judges no run of another card class
at 53 or 54.

**What is not changed by this.** Kits 4's to 6's bars at 30 to 39, 51 and 52, and the bars of 31 October at 40 to 50.
`match_things`'s bars at 30 stand as kit 6 measured them. The six unpushed answers most designs are set against,
`match_wide`'s twelve and `match_things`'s twelve. Kit 6's layers are taken under kit 7 exactly as under kit 6. A claim
taken under kit 6 is checked by kit 6's own file (`census_volunteer.earlier_kit`). Whether `match_things`'s doses are
counted calibrated at 53 and 54 (`DOSES_CALIBRATED`), which is the owner's word and changes no answer, bar or canary.
Whether kit 7 is cut, which is the owner's word too.

**What had been seen when this was fixed.**
- Kit 6's measurement at 30 under `match_things` (`../../kit6/bars.txt`, 768 answers a variant): floors 0.0053 and
  0.1178; bars 0.013 (built to catch a push 5% off) and 0.26 (10% off); first pieces kept at least 0.9896 by the
  arithmetic, so 0.90 stood. Its ties: 24 of 132 arithmetic-only answers parted, the surest at 0.52, so 0.6 stood
  (`../../../census_unpushed_ties/kit6/ties.txt`). Chair and money parted in nearly every variant there, at pieces the
  home card wrote with probability 0.32 to 0.50.
- **The walk's trial runs at 53 and 54** (18 November), on 1,022 of each layer's neurons and 1,024 controls drawn with
  another seed (11): every one of their answers, their verdicts (both pass N1 and N3; 54 also N2), and their twelve
  unpushed answers, the trial at 30's bit for bit.
- Kit 7's home canaries, taken before this was hashed (dfb5ef2). Kit 6's seventeen still in the kit, taken again, are
  the same. The canaries of 53 and 54 are plan batch 0 (bicycle's 64 answers). No variant of them.
- Kit 7's batches at 53 and 54, held to 23,900 MiB free on the home card: every question's longest batch was written
  at each, walk the tightest (22.1 GiB allocated without the block held aside, as at 30; 321 MiB to spare at 53).
- No answer of any other card under `match_things` at 53 or 54, and no arithmetic-only or near-miss variant of the
  home card there.

**What this leaves untested.**
- Kits 4's to 6's caveats stand.
  - The variants are the home card's own arithmetic, and another card class may move log-probabilities in ways
    they do not span.
  - A near miss is the whole push scaled, both doses together.
- Each layer's chunk 0 holds 16 units' answers: neurons 0 to 14 (the last three quarters asked) and one control.
- A run at 53 or 54 with any other design's job list is refused by the intake's meta check, not by these bars.
