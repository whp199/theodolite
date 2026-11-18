# The walk with match_things, upward from layer 53 and downward from 29: what is run, how it is read, what it decides

Fixed on 18 November 2026, before any answer of `match_things` was written at any of these layers (SHA256SUMS beside
this file; `prereg_at`). The owner's word was "Yes, both ways" (18 November), to registering a `match_things` walk up
from 53 and down from 29. Layers are counted from 0, as the trials and the board count them: 53 here is 54 on the
pages, 29 is 30.

**Why.** `match_things` passed its trial at layer 30 on 18 November (experiments/results/census_things_trial), and kit
6 offers that layer to volunteers. Thirty-seven layers have no design that passed a trial there: 0 to 29 and 53 to 59.
Asked how this kit would do on them, nothing measured so far can say. At layers 30 to 38 the dose rule gave doses that
passed and did not say by how much, and at 53 the span of pushes every earlier dose had passed in was no guide. So
each layer gets a trial of its own, as every layer outside 39 to 51 has.

- **Upward, 53 to 58.** `soft_ceil` failed at 53 on 16 November (experiments/results/census_soft_walk_up). It failed
  on a neuron's words carrying from one half of the questions to the other (0.557 against 0.57), the gate
  `match_things` was built to pass at 30. Its doses there were 0.37 and 1.53 nats; this design's are 1.0 and 1.4. The
  fine dose pilot of 16 November gave 53 to 58 their doses; 59 has none.
- **Downward, 29 to 25.** No design has been trialled below 30. The fine dose pilot was taken at 25 to 29 on 18
  November (below), after the owner asked how the kit would do further on and before this was hashed. It was not
  taken under 25: the coarse pilot puts the lower dose at 22 to 24 at 0.26 to 0.30 of the stream, at or under the
  fine ladder's lowest rung (0.29).

`match_things` was built at layer 30 on what two failed designs showed there. None of these eleven layers had a hand
in it. 53's failure under `soft_ceil` was read again with 30's before it was built (census_word_carry/NOTE.md), and
that reading found what differed at 30, not at 53.

**The design, `match_things`** (backend/census.py at `repo_commit`), as its trial at 30 and kit 6 take it:
- **The doses.** Two counts of each neuron's own write ceiling, plus and minus: the pushes that move the next
  word-piece by 1.0 and by 1.4 nats at that layer (`MATCH_THINGS` is `MATCH_CEIL`). A control is pushed by the ceiling
  of the neuron it is paired with. In ceilings, with what share of the stream's median length that is:

  | layer | 1.0 nats | 1.4 nats | of the stream | from |
  |---|---|---|---|---|
  | 53 | 2.24 | 2.42 | 1.37, 1.48 | fine_pilot_high.json, 16 November |
  | 54 | 2.18 | 2.34 | 1.23, 1.32 | the same |
  | 55 | 1.95 | 2.14 | 1.14, 1.25 | the same |
  | 56 | 1.96 | 2.11 | 1.33, 1.43 | the same |
  | 57 | 1.55 | 1.68 | 1.28, 1.39 | the same |
  | 58 | 1.16 | 1.27 | 0.93, 1.02 | the same |
  | 29 | 2.10 | 2.32 | 0.45, 0.49 | fine_pilot_low.json, 18 November |
  | 28 | 2.04 | 2.16 | 0.48, 0.51 | the same |
  | 27 | 2.03 | 2.13 | 0.48, 0.51 | the same |
  | 26 | 2.05 | 2.12 | 0.45, 0.47 | the same |
  | 25 | 2.15 | 2.20 | 0.40, 0.41 | the same |

- **The questions.** Twelve, four in each of three genres (how a thing works, three tips, an object), read as two halves
  of six: bicycle, walk, chair, kettle, sleep, lamp | clock, rice, vase, lock, money, blanket.
- **The answers.** 48 a unit, at 48 word-pieces, in batches of 64, every answer counted and each word set against the
  controls alone. A whole layer would be 1,130,508 answers.

**The fine pilot at 25 to 29** (`scripts/census_dose_pilot.py measure --layers 25-29`, then `doses --targets 1.0,1.4`;
18 November 2026, about 15:43Z to 15:52Z on the home card; `../../census_scaled_design/fine_pilot_low.json`). It is the
pilot of 13 November on the same rungs (0.29 to 0.92 of the stream, 7% apart), with 128 neurons a layer drawn the same
way (seed 0 plus the layer), the same code and rule; layer 45, its reference, is measured first as always and again
gives no dose on this ladder. Its doses are in `MATCH_CEIL` at `repo_commit`, copied by hand from the pilot's table,
nothing adjusted. What it showed, before this was hashed:
- **29 reads like 30 to 38**: near its lower dose a push 7% larger moves the next word-piece about 1.26 times as far
  (1.26 to 1.55 at 30 to 38), and its two doses are 10% apart (6% to 10% at 30 to 38).
- **25 to 28 sit on a steep step.** Near the lower dose a push 7% larger moves it 1.6 to 2.8 times as far, and the two
  doses are 2% to 6% apart. At 25 the step from 0.41 to 0.44 of the stream takes the next word-piece from 1.2 to 3.1
  nats. So a neuron pushed a little harder than the median moves its text far more, and one pushed a little softer
  barely moves it.
- At 25 the ladder dips once, at its top rung (10.77 to 10.76 nats), far past both targets.

**Step 0 is not taken again.** It asks whether a question's unpushed answer is written the same in the batch its run
writes it in as alone. No push enters an unpushed answer at any layer, and a trial's batches are the same shapes at
every layer (`census.plan_batches` over job lists of the same lengths: checked before this was hashed, the twelve
unpushed answers fall in batches of 64 for the first nine and 61 for the last three at all eleven layers, as at 30). So
30's check stands: each question's unpushed answer holds at least 32 word-pieces with itself written alone (chair 46,
money 38, the rest all 48).

**What is run.** At each layer in turn, in this order: 53, 29, 54, 28, 55, 27, 56, 26, 57, 25, 58 (`walk.sh`).
- **The neurons.** 1,022 drawn for each layer on 18 November by `draw_neurons.py`, seed 20261118 x 100 plus the layer
  (`neurons_L<NN>.txt`). The draw leaves out every neuron a census run had asked at that layer:
  - at 53, the 1,022 of `soft_ceil`'s trial;
  - at 57, the 507 of `break_rel`'s probe and the 64 of the two ladders before it (29 October).

  No census run had asked a neuron at the other nine. The home canaries (plan batch 0 under `break_rel`, the first
  few neurons of a layer) are not left out, as no earlier draw left them out.
- **The controls.** 1,024, drawn afresh with seed 11, which no trial has used (they used 3 to 10).
- **The size and the place.** 98,220 answers a layer, about 2.5 to 3 hours on the home RTX 5090. Up to eleven
  layers, about 32 hours. Written from a clean checkout of the commit in `repo_commit` by `night.sh`, into
  `atlas_data/census/trials/things_walk`, which no page reads as a layer's census.
- **The code.** `repo_commit` differs from kit 6's tag (`census-kit-6`) in nothing that writes an answer but the five
  new rows of `MATCH_CEIL`.
- **Pauses.** The walk may be paused between or inside trials when the home card is needed for one GPU process at a
  time, for instance a volunteer's spot chunks at intake. A trial stopped part way is continued row for row by
  `night.sh`.
- **When it begins.** This registration is pushed before the walk begins, on the owner's word.

**How it is read.** By `things_read.py` (`read.sh`), on the CPU, as soon as a layer is written. It is the reader of
`match_things`'s trial at 30 byte for byte, over `trial_read.py` and `readlib.py`, the first trial's of 29 October
byte for byte. Read again on that trial's answers before this was hashed, it gives the registered verdict back
(`reader_test.txt`). All gates are read at 48 word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56).
- **N1, the whole reading.** G1 to G5 and G5b, every answer at both doses, on the halves of six.
- **N3, the neurons the controls cover.** G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range. It fails if fewer than half are covered.
- **N2, the lower dose alone.** G1 to G4 on the 1.0-nat answers alone: the reading `things_read.py` reports as "the
  lower dose alone" (`reported.each_dose.lower.pass_G1_G4` in the verdict). New here; at 30 it was reported and
  passed. It does not stop or carry the walk, and it counts only at a layer that passes N1 and N3. It is a weaker
  standard than N1 and N3 together: no gate on the covered neurons.

A layer passes when N1 and N3 both pass, as at 30. G4 asks that half B carry a lead by itself (clock, rice, vase,
lock, money, blanket), as at 30.

**What has been seen that bears on it.**
- **`soft_ceil` at 53** (16 November; other neurons, other controls, break_rel's six questions; ±1.6 and ±2.3
  ceilings, 0.37 and 1.53 nats):
  - whole 0.694, 0.631, 0.574 and 0.557, 0.654;
  - the lower dose alone failed every figure (lead 0.551);
  - the upper dose alone led by 0.689, but its words carried at chance between the halves (0.507, 0.511);
  - 123 neurons beat every control and 154 have q ≤ 0.05;
  - the upper dose took 0.17 of the neurons' distinct 4-grams, the controls' 0.28;
  - its strongest control stood 3.2 standard deviations out (census_word_carry/NOTE.md).
- **`soft_ceil` at 52 passed** (whole 0.804), and its whole layer, from a volunteer, has 6,933 neurons with q ≤ 0.05.
- **`break_rel`'s probe at 57 failed** all three readings the neighbour trials were later gated on (29 October), at ±1
  and ±2 of the stream.
- **`match_things` at 30** (18 November):
  - whole 0.618, 0.637, 0.582 and 0.587, 0.625; the covered 0.615, 0.633, 0.580 and 0.586, 0.623;
  - the lower dose alone 0.616, 0.634, 0.588 and 0.581, 0.606;
  - 97.7 neurons in 100 covered;
  - the text lost at most 0.003 of its distinct 4-grams;
  - none named; with the five strongest controls set aside (Portuguese, Spanish, Russian), 45 have q ≤ 0.05.
- **The lead on all answers from 34 down**: 0.758, 0.726, 0.699, then 0.630 at 31 (`match_wide`), and at 30 0.579
  (`match_wide`), 0.618 (`match_both`) and 0.618 (`match_things`).
- **The two pilots, above.** The coarse pilot (the steer lens's, rungs a factor of two apart, every layer) agrees with
  the fine pilots to within about a quarter of a ceiling where both were taken.
- **Every figure of a trial has a standard error near 0.013** at 1,022 neurons.

**What is expected, said beforehand.**
- **53 passes N1 and N3 about 1 in 3.** If it fails, it fails on the words carrying (G3) first.
- **29 passes about 1 in 3.**
- **Further along each arm.** Given a pass at the layer before it, each next layer passes about as likely as not, and
  at 25 to 28 less likely than at 29. The upward arm reaching 58, or the downward arm reaching 25: each under 1 in 20.
- **Where 25 to 28 fail.** If one of them fails, it fails on how the text survives (G2 or N3) as likely as on G3.
- **N1's lead**: between 0.62 and 0.72 at 53, and between 0.58 and 0.64 at 29.
- **The share covered**: 90 to 98 neurons in 100 at 53 and 29. At 25 to 28, under 90 more likely than not.
- **The text, at 29**: it loses under 0.005 of its distinct 4-grams at either dose, as at 30.
- **The text, at 53**: both doses lose 0.10 to 0.30 of them, since both lie near `soft_ceil`'s 2.3 ceilings there.
- **N2**: at a layer that passes N1 and N3, the lower dose alone passes as likely as not.
- **The controls that stand out**: at 25 to 29 the strongest write other languages, as at 30 to 37. At 53 to 58 no
  control stands more than 5 standard deviations over the controls' median.

**What it decides.**
- **A layer passes N1 and N3.** It may be offered to volunteers under a kit that carries `match_things` there. Before
  that kit, its other card classes' bars need a registration of their own, as kit 6's at 30 had. A kit is the owner's
  word, and so is widening `DOSES_CALIBRATED` to it. Its arm goes on to its next layer.
- **A layer passes N1, N3 and N2.** As above, and the lower dose alone may be made a design of its own there: 24
  answers a neuron, about 17 hours a layer on a 5090. It needs its own step 0 (its unpushed answers fall in other
  batches), its own bars and the owner's word before any kit offers it. Nothing here builds it.
- **A layer fails N1 or N3.** No whole layer is taken there with `match_things` under this registration. Its arm
  stops, and the layers past it in that arm are not trialled under this registration. The other arm goes on.
- **The walk ends at 58 and 25 at the latest.** 59 has no doses. Under 25 nothing here measures them. What to try
  past either end is the owner's choice.

**Reported, gating nothing.**
- The same figures cut to 32 word-pieces, and the length rule.
- The share covered.
- What the text loses of its distinct 4-grams.
- How many neurons beat every control and how many are named.
- The upper dose alone.
- The two readings at three questions a half, as at 30.
- Whether each trial's twelve unpushed answers are the trial at 30's bit for bit, as is expected (no push enters
  them). A difference is reported and its cause looked for; it changes no verdict.
- How the strongest controls read (computed after the reading from the run's units file, as at 30).

**What a pass would not show.**
- **That the layer's neurons answer the census's other questions as they answer these.** A layer taken with this
  design is asked none of who, orange or free.
- **What a neuron does when it fires.** A pass says a layer's pushes differ from random directions' on these
  questions. It does not say its words share a theme.
  - At 53 to 58 the push is about as long as the stream itself or longer (0.93 to 1.48 of its median length), so it
    shows what a push that size does.
  - At 25 to 28 two neurons pushed alike in the pilot's sense may move the text far apart, so a rank there may say
    more of how near the step a neuron sits than of its words.
- **That two passing layers are alike.** Each layer is one trial among up to eleven. At about 0.013 a figure, a layer
  near its bars can pass or fail by chance alone.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in `AMENDMENTS.txt`
beside this file, before a reading is made.
