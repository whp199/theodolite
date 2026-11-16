# Levelling under attack: the statistics and the guards

Written 14 November 2026. Read-only: nothing in the repository was changed, no server started, the model not loaded,
the card not touched. Small CPU sums only (scripts in
`(the session scratchpad)/skeptic_stats_tmp/`).

Marks: **[read]** a file opened today; **[measured, date]** a record in the repository or the pilot folder;
**[my arithmetic]** my own sum; **[inferred]** not checked on the card.

The proposal (`design/synthesis.md`, 14:43) was written before pilots 3 to 7 (14:49 to 15:03). The pilot note's
second half ("READ THIS: it changes what a reading may be trusted for") bears on findings 5, 6 and 12. Line numbers
below are the synthesis's unless a path is given.

## In brief

The rank rule is sound arithmetic and a poor guard. "Top every one of C chance edits" is a valid test of one null
only: that the candidate is a random draw from the chance edits' own distribution. No candidate an agent or a
weights reading picks is such a draw, so the p printed on the sheet is not the chance of what it seems to be. The
house has met this before and wrote it down (`V8_stat_audit_NOTES.md:18`: "the control comparison is uninformative
by construction"). What actually stops a wrong "held" is the bar and the batteries, and neither has a rule: the
five-times-floor bar passed batch-size artefacts in today's own pilot, and the batteries' bars do not exist.
Nothing here stops phase 0. Pilot D's threshold, the rule for *held* and the word *lead* must change before either
is registered.

## Findings

### 1. The p of "tops every chance edit" holds only for a candidate nobody chose. Weakened, serious.

**Claim** (329-330, 742): standing is an empirical p of 1/(C + 1) a family; claim 17 says so "if the edit is
exchangeable with them" and ties the condition to claim 8 (matched disturbance) alone.

**What I checked.**
- The formula is the census's (`backend/census.py:703`) and is right. Requiring both families is a valid
  intersection test at 1/(C + 1). That part stands (finding 11).
- Exchangeability fails by selection, not only by mismatch of size. Three sums [my arithmetic]:
  - *Screening.* Best of K candidates drawn from the chance edits' own distribution, then one fresh reading:
    it tops 2C fresh chance edits with chance K/(K + 2C). K = 63 (one forward, 2.4 s [measured, 14 November
    2026]): 0.45 for *held*. K = 1,000 (one to three minutes by the proposal's cost model): 0.93 for *held*,
    0.83 for *passed*. The sheet would print p at most 0.025. Fresh items do not help: the effect belongs to
    the edit, and items are shared by every row.
  - *A weights reading alone.* Take a layer whose 21,504 columns are pure noise. The 64 top writers along any
    direction u carry a summed write along u 24 standard deviations above a random set of 64. On a read-out
    aligned with u they top any number of chance sets, with no learned structure at all. `writers` (184) is
    this selection, and it costs no practice reading, so the sheet's count of "candidates read on the practice
    items" (333) is nought.
  - *The house's known answer.* Silencing the ten 'she' neurons at layer 50 moved the 12 female contexts by
    -1.799; the ten 'he' neurons moved them by -1.798; ten random, -0.020 [measured, 16 October 2026;
    `experiments/results/E6_tool_audit_04_vault.partial.json`, `causal_delete`]. Under the two-family rule both
    lists earn *held* for the 'she' pull. The rule cannot tell the hypothesis from its neighbour.
- The house found the same fault in its own study: E4's steering "specificity REFUTED" once controls were other
  real directions of equal norm (He 15/16, She 15/16), and a control with nought projection by construction was
  called uninformative (`experiments/results/V8_stat_audit_NOTES.md:17-18`) [read].

**What would settle it.** Planted nulls that are *selected*, not random: run the scripted recipe on sham targets
and count how often it reads *held*.

**Fix.** Add a third family wherever the site came from a declared rule: the same rule on another target (other
word pairs, label-shuffled practice items). For a free pick, print "selection not matched; K read in practice;
chance by selection alone K/(K + 2C)" and no p. Word what the comparison shows as the census does
(`census.py:399-404`): "differs from a random edit; does not say what it does".

### 2. The gate that decides whether to build cannot fail as written. Weakened, serious.

**Claim** (483, 673): go on if at least 5 of 20 column sets top every chance edit of both families and exceed five
times the floor row, at two of four layers.

**What I checked.** The 20 sets are "top writers along a difference-of-means direction". The metric is not named.
On the read-out the direction was built from, finding 1's sum applies: selected sets pass in a noise network.
Under exchangeability the threshold is strict (1 in 9,000 a layer [my arithmetic]), which makes it look like a
hard test; it is hard only for sets nobody chose. Also, "30 to 75 minutes" fits pooled chance edits and the cheap
read-out only. With chance edits matched to each of about 780 candidates a layer it is tens of hours [my
arithmetic, my count of the row list]. And 16 of its 80 items are stored passages, raw text, the regime pilot 7
found unstable (finding 5).

**Fix.** Before hashing: name the metric, and make it one the sets were not built from; add sham-target sets as
the comparison; say pooled or matched, and cost it.

### 3. The family-wise chance is printed, never applied, and no budget is set. Weakened, serious.

**Claim** (332-334): the brief is the family; the bound is printed beside every p; no correction is run because it
"cannot pass".

**What I checked.** True that it cannot pass: with 39 a family the least p is 0.025, so any correction over three
or more readings fails at 0.05. The remedy chosen is to drop the correction, not to draw more. For a candidate
that is exchangeable: one of 10 fresh readings clears with chance 0.22, one of 28 with 0.51; one of 10 practice
readings earns *lead* with 0.40 [my arithmetic]. The number of fresh readings a brief is named as a budget (412)
and given no figure; it is not among the 15 decisions, though the validity design listed it (`validity.md:457`).
Briefs are uncounted, and the agent may propose them.

The census does the opposite: p, then a layer-wide q (`census.py:704-708`), and it says N/(C + 1) land above every
control by chance (`census.py:44-47`). The code digest itself notes "no ledger records tests across calls"
(`digest_code.md:932`). The level book is that ledger and is used only to print k.

**Fix.** The owner fixes m, the fresh readings a brief, before the first. Each reading then needs 20m - 1 chance
edits a family: 39 supports two readings, 99 supports five. No word on practice items.

### 4. Fresh draws from the practised generator guard item noise and nothing else; some starter briefs cannot be drawn fresh at all. Weakened, serious.

**Claim** (307-313): the seal "holds against any reader"; a fresh draw "guards item noise and best-of-search
inflation".

**What I checked.**
- It holds only because there is nothing to seal. Practice and fresh items are draws from one generator the
  driver may sample without limit.
- The harness's own first generator is the E6 frames "with many more names". E6 has 12 frames, one name each
  (`experiments/scripts/E6_tool_audit_04_vault.py:65-71`) [read]. Ninety-six fresh items are the same 12 frames.
  The house has measured that such counts are not independent: "6/6 or 16/16 measure one prompt's mode"
  (`V8_stat_audit_NOTES.md:105`). The interval in the claim sentence (373) has no stated unit.
- Starter briefs are files (300-305): six to ten census questions, 12 contexts, 1,218 contexts. A draw from a
  file after practice on the same file repeats items. The model is deterministic: the census measured that a
  retest on cells its screen had read gave p at most 0.05 a fifth of the time, against one in twenty held out,
  and `take_census` refuses it (`census.py:1640-1648, 2147-2149`) [read]. The proposal has no such refusal. The
  validity design's condition ("honest only when the generator's space is far larger than anything practised
  on", `validity.md:224-225`) was dropped.
- The proposal's own literature: an edit that passed practice prompts failed all 64 targets after rewording
  (91). Rewording is tested only at a trial, yet decision 8 (680) lets a *held* edit on a page.

**Fix.** Hold out whole frames, as the census holds out question halves (gate G3). Refuse a fresh item seen in
practice. Intervals by frame. A file-backed brief takes the file seal's name and limits. No *held* on a page
without held-out frames.

### 5. "Five times the floor row" passed batch-size artefacts today. Refuted as a guard, serious.

**Claim** (233-235, 335-337): the floor row gives the delta path's floor; the bar is the brief's least effect or
five times the floor row.

**What I checked** [measured, 14 November 2026, pilots 3 to 7, one run each].
- Recipe text, layer 20, 32 rows, output-side delta as section iv specifies: floor row 0.00338, bar 0.0169, and 5
  of 30 single silencings cleared it (`pilot3_writing_and_capture.json`, `recipe_layer_20`). Four of the
  loudest read 0.0227, 0.0215, 0.7025 and 0.0689; folded at batch 2 the same four read 0.0091, 0.0033, 0.0058
  and 0.0042, all under the bar (`pilot4_cancellation.json`, `worst_five`).
- The folded form moves too: neuron 4294 reads 0.0091 at 2 rows and 0.7258 at 32, folded and input-side alike
  (`pilot6_which_is_true.json`, `L20_n4294`). There are two arithmetics, up to 12 rows and from 16.
- The floor row cannot see this: 93 null rows never tipped (`pilot5_knife_edge.json`). It measures rounding of a
  null edit; the artefact is a real edit meeting a knife-edge item.
- The floor is not a constant: null rows run 0.00014 to 0.0066 by text and layer, 47-fold.
- On chat-formatted text the trouble was absent: 0 of 120 silencings differed over fivefold between batch sizes
  (`pilot7_batch_size.json`). In this sample it showed on one raw text of two (the recipe, layers 5 and 20) and
  on none of four chat-formatted cells. One run, two texts.
- Raw text is in the design: corpus windows and stored passages are raw text after a start mark
  (`backend/atlas_build.py:115-131`) [read], and they feed two batteries, pilot D and two starter briefs.
- For a signed metric the rule is undefined: the floor row's mean over items is a mean of zero-mean noise and
  shrinks with n. The only floors measured are KL. On one confident position the two forms differed by 1e-7 in
  KL and by 0.16 in one word-piece's log-probability (`pilot_edit_speed.json`, `P3_row_delta_vs_overlay`).
- The house already has a bar rule, registered: floor F from arithmetic-only variants, a near miss M the bar is
  built to catch, bar sqrt(F x M) for the smallest M at least 4F, and the bar dropped where none qualifies
  (`experiments/results/census_other_card_bars/kit4/prereg/RULES.md:53-57`) [read]. Its bars sit at 2.1 and 2.2
  times the floor. "Five" has no second side.

**Fix.** Take the pilot note's four rules into the reader (gate items on the base's own log-probability; read in
both arithmetics and call disagreement an unstable item; median beside mean; input-side for columns). Take the
floor from 19 or more null rows on each item, as a per-item size. Set the bar by the house rule, with the brief's
least effect as M.

### 6. The batteries have no bars, and the natural ones fail every real edit, the known answer included. Weakened, serious.

**Claim** (341-348): five batteries, "each set beside the chance edits' range", each bar "measured and hashed
before first use".

**What I checked.**
- No rule says how a bar is measured, on what, or which batteries gate. Phase 3 allows 3 to 4 hours of card for
  all of it. The house needed a registered measurement of 13 variants for one family of bars.
- "Within the chance edits' range" cannot work for habitat: the passages are where the candidate's own neurons
  fire hardest, where other sites do not fire. Not said: whether each chance edit is read on its own habitat.
- Word-for-word retention is hair-triggered: a push 5% off left a median of 11 to 12 of 48 pieces alike
  (`experiments/results/census_other_card_bars/bars.txt:13-14`) [measured, 31 October 2026].
- An edit exactly as harmless as a chance edit falls above the range of 39 once in 40 a comparison: 0.12 over
  five comparisons, 0.22 over ten [my arithmetic, independence assumed].
- The positive control: the 'she' ten moved the male contexts by +0.904 and the "Gemma as human" prompts by
  -2.278, against -1.799 on target [measured, 16 October 2026, same file]. With male contexts as the near
  misses, any bar under half the target effect reads *works but harms*. The proposal says it must read *held*
  (362-364). The expected verdict of the harness's first self-test is not defined.
- Neutral text and habitat are raw text (finding 5).

**Fix.** Before phase 3, one registered bar rule on the house's pattern, with the planted harm as the near miss.
Say which batteries gate and which are reported. Wrap or gate battery text. Fix the known answer's near-miss set
and expected verdict before it runs.

### 7. The obstacle-or-ability table cannot be read at these sizes, and written gains of the published size cannot be seen. Refuted for the shapes costed, serious.

**Claim** (359-360): rescues and regressions split by five states of the base's answer.

**What I checked.**
- At 48 pieces the base's own answers to all six census questions end "length"
  (`atlas_data/census/runs/L36-match_ceil.jsonl`, baseline rows); 98.7 to 100% of trial answers are cut off
  (`experiments/results/census_break_rel_trial/L45_verdict.json`, `text`) [read]. At the costed shape (276,
  "48 new pieces") the "cut off" stratum is the cap. Answers that finish need hundreds of pieces, uncosted.
- A stratum holding 5% of answers has under 1 item of 16, about 5 of 96, 13 of 256. Three of five rescued has a
  95% interval of 0.15 to 0.95 [my arithmetic].
- "Repeating", "refusing", "wrong" need a classifier; the proposal has no model judge (419).
- A net gain of 3 points on a right-or-wrong metric, the largest held-out gain on the 9B (83-84): a one-sided
  sign test at 0.025 sees it with chance 0.09 at 96 items and 0.36 to 0.49 at 256 [my arithmetic].

**Fix.** Strata at trial only, with a cap under which nine in ten base answers finish; counts under 20 printed as
"too few to read". Print the least gain each tier could have seen.

### 8. The other-site family has no members for the candidates most likely to be heard. Weakened, serious.

**Claim** (317-319, 315): matched on write ceiling and firing; "always both, never switched off".

**What I checked.**
- On the pilot's text the hardest-firing neuron at layer 44 fired 14.25, the next 7.41, the third 3.66
  (`pilot_edit_speed.json`, `neurons.acts8`) [measured, 14 November 2026]. Neuron 14802 gave the largest
  effect, and by that list the next site fires about half as hard. Pilot D's "20 most-active single columns"
  have the same trouble by definition.
- The write ceiling depends on a column's direction only (`backend/atlas_ops.py:192-201`) [read]. Ceiling and
  firing together do not fix the size of a write; the column's own length is missing. `writers` ranks by an
  unnormalised product (184), so it favours long columns [inferred].
- Matching on firing on the brief's items selects neurons tied to the behaviour, so the null may hold true
  effects [inferred; pilot B could read its random tens both ways].
- 39 sets of 64 is 2,496 neurons; with about nine in ten shut at an ordinary position, the sets overlap
  [inferred].
- The house's rule for this is gate N3: read only the neurons the controls cover, fail under half covered
  (`experiments/results/census_neighbour_trials/prereg/RULES.md:25-26`) [read]. The proposal has none.

**Fix.** Match on measured `delivered` an item. Void a reading with fewer than C sites inside a stated tolerance,
and say so.

### 9. The house gates on other statistics and reports "beat every control" as gating nothing. Weakened, serious for the words.

**What I checked.**
- Both registered rule files list "how many neurons beat every control" under "Reported, gating nothing"
  (`census_neighbour_trials/prereg/RULES.md:45-46`; `census_match_trials/prereg/RULES.md:70-71`) [read]. The
  gates are population figures with bars from a calibration, replication across disjoint question halves both
  ways, a covariate residual and coverage (`census_break_rel_trial/prereg/trial_read.py:38-47`).
- The house's own data show why: of 1,022 neurons, 169 beat every one of 1,024 controls and 76 beat every one
  of 2,048, while the counts at p of 0.01 or less held (259, then 268)
  (`experiments/results/census_L45_whole/summary.json`, `the_trials_neurons_against_fresh_controls/ranks`)
  [measured, 29 to 30 October 2026].
- The words. A census *lead* is p against 2,048 controls, a layer-wide q, replication across halves, at a layer
  whose design passed a registered trial. A Levelling *lead* is 19 + 19 on 16 items the driver tuned on, any
  number of times. Both would sit on one instrument's pages. *Passed* rests on 99 + 99.

**Fix.** Gate on a fixed level with enough chance edits, plus replication across frame halves. Report
"tops every one". Keep *lead* for one grade of evidence.

### 10. The harness's negative control tests nothing. Weakened, minor.

"Not for ten random ones" (362-364): a random ten is a draw from the other-site family, so it reads "not held" 39
times in 40 whatever the reader does, and a sound reader fails its own self-test once in 40. The stored -0.020
for the one random ten is three items at steps of an eighth, the read-out's rounding [read, same file,
`random@50`]. **Fix:** 200 planted nulls with a band on the count that hold, as the house bands its null split
(`trial_read.py:47`); and the seven 'she' neurons the 'he' list does not share as the negative that matters.

### 11. What stands.

- The empirical p and the both-families rule, for a candidate that is a random draw (tried: the census's own
  code; the intersection argument).
- "No correction can pass with a handful of nulls": true arithmetic.
- Power on a paired log-odds read-out is ample: the known answer's per-item spread is 0.38, a standard error of
  0.039 at 96 items; topping 39 + 39 normal nulls takes about 2.4 to 3.2 of their standard deviations
  [my arithmetic].
- Quoting a gain against the best chat setting of the base (356-358) is conservative.
- The base row never memoised, differences inside one forward: sound, and pilot 6 confirms no leak between rows.

### 12. *Passed* has undefined pass rules and inherits the house's redesign-after-fail habit. Weakened, serious.

**What I checked.**
- The written gate, the reworded set and the batteries have no size, statistic or bar (295, 384-390). The
  house's registrations fix numeric bars from a calibration and say what is expected beforehand.
- The reworded set gives "at most five readings in its life" (312): five trials against one set.
- The folded form, the form of record, is itself batch-dependent on unstable items (finding 5), and "two
  forwards of one shape" will sit in the small-batch arithmetic while the fresh reading sat in the large.
- The house's record, layers counted from 0: layer 38 passed on its third design after failing two, one by
  0.001; layer 37 passed on its second; layer 36 passed by 0.001 on one gate (`census.py:163-176, 240-251,
  256-265`) [read]. Each was registered and narrated; none was counted as a family. A failed trial followed by a
  new edit and a new trial is the same path.
- The costs do not fit the rule: by the proposal's own model the fresh reading's main panel alone is 2.4 to 3.1
  minutes, so "2 to 4 minutes with its batteries" holds only if the batteries do not carry the chance edits
  [my arithmetic].

**Fix.** Trial readings in both arithmetics. Bars from a calibration reading, hashed. The sheet prints every
earlier trial on the brief and the behaviour.

## The cheapest wrong claim onto a page

1. The agent proposes a brief; the owner keeps it. The brief's author sets the metric, the least effect, the
   near-miss generator and the prompt sentences of the cheaper-knob arm (299-305, 350-352).
2. It picks top writers with the `weights` tool. No practice reading is counted.
3. One fresh reading: it tops 39 + 39 by construction, clears its own least effect, and meets batteries whose
   bars are unset.
4. *Held*. Decision 8 puts it on a page "as a lead, with the sheet". The brief's plain-words name ("the 'she'
   pull") is read as what the neurons do, though the 'he' list does the same.

Cheaper still: the driver's free prose, shown under "the scientist's own words, not checked" (417-419). For the
owner's own enthusiasm: a new brief or a new edit after a narrow fail.

## Not said at all

- The test statistic, and which dose of the dose rows is "the edit".
- A number for fresh readings a brief and for trials a brief.
- Any null matched to how the candidate was chosen.
- Held-out frames; a refusal of repeated items; the least size of a generator.
- A coverage rule for chance edits; when a reading is *void*.
- A rule for deriving any battery's bar; which batteries gate; the order of the seven verdicts.
- Who sets the least effect and the prompt arm when the agent wrote the brief.
- What the interval in the claim sentence is.
- Item gating and the two-arithmetics check from pilots 5 to 7.
- More than one session before a page: three identical searches gave +29.45, +5.00 and +27.30 (86).
