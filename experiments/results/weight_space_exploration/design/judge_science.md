# Judgement: science against the literature

Written 14 November 2026. Lens: would a sceptical interpretability reader, who has read the weight-space-programming
papers of the last two weeks and their critiques, believe what each design reports? I read the two digests in full, the
five designs in full, and opened the repository and the research reports where a claim needed checking. Nothing was run
on the card, nothing in the repository was changed. Layers are counted from 0.

## In brief

| Rank | Design | Score (1-10, this lens only) | One line |
|---|---|---|---|
| 1 | validity (The Proof Room) | 8 | The only design whose pass rule, nulls, batteries and claim sentence would survive a hostile reading. Narrow held-out, and the agent is kept on a short lead. |
| 2 | driver (Traverse) | 7 | The claim boundary is the best engineered (a fresh process that has never run agent code). The loop is unguarded by choice and free code beside the model cannot be held to "no gradients". |
| 3 | future (The Plane Table) | 6.5 | The right things for outsiders to believe a result (re-runs on another card, replay, a flag for a prompt that names the fault). Thin on the measurement itself, and its first agent cannot build a direction from a contrast. |
| 4 | speed (The Proof Press) | 6 | Cheap, well-chosen controls in every call. But its statistics do not add up as written, one of its speed devices breaks the house's own same-shape rule, and it makes a gradient the default first step. |
| 5 | kernel (Levelling) | 5.5 | The soundest first measurement and the best pilots. As a harness for an agent its hold-out and its ledger are conventions, and it leaves out most of what the literature says goes wrong. |

All five converge on the same kernel: an edit as a per-row added term on a linear's output, the unedited row in the
same forward, nothing left in force. Under this lens that convergence is right. It is the paired "both" pass the
original post is praised for, made exact on a card that cannot hold two models, and it removes the batch-shape floor
(KL up to 0.014 between shapes in the study; an unpushed answer in a batch of 64 parting from itself after 22
word-pieces, 28-29 October 2026) from the comparison. The designs differ in what they wrap round it, and that is what
I score.

## What I checked in the repository and the reports

- `experiments/results/E6_tool_audit_04_vault.partial.json`: the ten layer-50 'she' neurons are 5313, 8801, 266, 121,
  11503, 11724, 21151, 4165, 549, 2070; deleting them moved the female contexts by a mean of -1.799 over 12 contexts;
  the random control is -0.020, **one draw of ten neurons**. The report says |delta| <= 0.04 (line 137). Kernel's
  figures are the file's; validity's and driver's are the report's. Both right.
- `experiments/results/vault_example_edit/probe.json` (29 October 2026, reference unpacker): the example edit moves the
  spider question (first-step KL 5.65) and **also turns "A dog has four legs" into "six legs" (KL 6.84)**. The one
  stored before-and-after record of a vault edit already shows bleed onto a near miss.
- `backend/atlas_ops.py:186-190`: a neuron's write direction is `unit(w_post (.) W_down[:, n])`, through the
  post-feed-forward norm's weight. `backend/larql_vault.py:512-521`: the vault adds a stream-space direction to the raw
  column without that weight (the code digest's "surprise").
- `backend/engine_runtime.py:1200-1221`, `backend/census.py:90, 2132`, `backend/atlas.py:331`: Gemma 4's chat template
  has a thinking switch; the instrument asks everything with thinking off.
- Prefill of a census question, compiled unpacker, 28 October 2026 (code digest): 0.10 s at 1 row, 0.26 at 32, 0.45 at
  64, 0.60 at 96. A multi-position forward is not flat in rows; only a decode step is.
- `backend/atlas_build.py:800`: a steer-lens batch is `C + 2 * batch * C` = 516 rows. Transformers'
  `Gemma4TextMLP.forward` is `down(act(gate x) * up x)`, no bias. `/usr/bin/bwrap` and `/usr/bin/systemd-run` exist;
  `free` shows 61 GiB, 51 available. `backend/scientist_chat.py:567-578` carries the confinement flags driver quotes.
  `mcp_scientist.py:719, 831`, `causal_check.py:133`, `census.py:1567`, `atlas_index.py:125` are where the designs say.
- Research reports: IWIBench's 64 turns, 128 calls, 262,144 forward tokens and "an edit passing practice prompts failed
  all 64 terminal targets after rewording" are both supported by the checkers (wsp_reception and rigor_safety).

## The five, one by one

### 1. validity: The Proof Room. Score 8

**Would its results be believed?** Yes, within the sentence it allows. It is the only design that answers every item
of this lens with a mechanism.

- *Development against held-out.* Proof items are drawn from `os.urandom` after the edit and the rule are hashed, so
  they did not exist while the agent searched. Aggregates only. A proof is read once; a changed edit is "the k-th proof
  on this brief" and the family-wise chance is printed. The sheet shows practice-best, proof and trial side by side,
  which is the Hu et al. shrinkage (all 14 tabulated runs, partly built in) put in front of the owner.
- *Nulls.* Two families, 26 draws each at proof and 60 at trial: same site with other content, and other sites
  **matched on write ceiling and on how hard they fire on these items**. No other design matches on firing, and the
  instrument's own wiring check says about nine readers in ten sit with the gate shut, so an unmatched random-site null
  flatters every edit that silences an active neuron.
- *Floors and canaries.* No memo of the base row; whole forwards for proofs; a zero edit that must match by bits; a
  norm-undone rescale as the delta path's floor; canary briefs with no possible fix.
- *Side effects.* Four batteries, of which *habitat* (the passages where the edited neurons fire hardest) is the one
  the literature says general panels miss, and which only this instrument can run.
- *Cheaper knobs.* The activation twin, a matched steering push with the start mark not spared, prompt sentences, a
  decoding bias, all as arms of the same plates, with a verdict of their own ("stands, matched by a cheaper knob").
  "Not compared with a trained adapter" is printed every time.
- *Misreport.* The warden applies the rule and writes the sentence; the agent's conclude tool takes only a ledger
  reference.
- *Trust by layer.* A card on every reply, and the claim says "site chosen by a reading not trusted at this layer".
  "The card limits stories, not effects" is the correct reading of the house's lens caution.
- *What finds edits.* The capture tool keeps a difference of means as a direction handle; the weights tool returns the
  writers along it; the rule requires the reversed sign and a half dose. Every weight-derived answer is typed a lead.
  A scripted recipe goes through the same gates as a no-agent baseline.

**Strongest.** The pass rule and the claim sentence, with pilot 5 as an honest go or no-go gate and bars left
"[not yet measured]" instead of invented.

**Weakest.** Held-out means fresh draws from the same generator the agent practised on without limit. That guards
item noise and best-of-search inflation. It does not guard the failure IWIBench recorded, an edit that passes its own
phrasing and fails every reworded target. The design says so and does not fix it.

**Serious flaws.**
1. No reworded or format-shifted tier (section v, "Three item tiers"; risks, "Generators are narrow").
2. The agent sets the "least effect" it must reach (registration). A low forecast makes rule line 1 empty; the nulls
   then carry everything. The forecast should be scored for calibration, not used as a bar the agent picks.
3. The brief does not record whether it names the suspected fault or site. The original result is disputed for
   exactly that. Future's docket has the flag.
4. Session-to-session variance is one pilot (three sessions). The claim excludes "another search would find it",
   which is honest, but nothing schedules repeats.
5. The decoy audit for weight-reading claims is in an open-ended last phase.
6. No code for the agent until Phase 5. Under this lens that costs discovery, not credibility: a solver for "which
   columns and how much" (a minute against eight hours once the fault was known) has nowhere to run.

### 2. driver: Traverse. Score 7

**Would its results be believed?** Its verdicts, yes. Nothing else it prints, by its own rule, and it prints a great
deal else.

- The verdict is run by a fresh process that has never run a cell, loads the edit as data, and reads items that were
  never mounted where the driver's code runs. That is the literature's mechanical guard done properly.
- 19 same-site nulls, a reversed sign, doses 0.5, 1 and 2, the day's floor, teacher-forced and written side by side,
  cheaper knobs and a no-driver scripted recipe, a near set, gain and retention never multiplied.
- A confirm set gives at most five verdicts in its life and is then rotated; feedback is rounded to standard errors.
  No other design caps a set's lifetime.
- The first verdict is a positive and a negative control: *held* for the ten neurons, *not held* for ten random ones,
  on a few hundred owner-written contexts, because twelve cannot be split.
- The one-page prompt names no suspected fault and is hashed into the ledger. Pilot 2 tests the harness's own bet
  (free cells against typed verbs). Phase 5 measures session variance and a second driver family. `SCAFFOLD.md` and
  `--bare` make the scaffolding's worth measurable.
- It lets the agent do what found the published edit: the recipe is a ten-line cell, and a dose sweep costs rows.

**Strongest.** The verdict worker and the rationed, life-limited confirm set.

**Weakest.** Free code in the process that holds the model. The design admits a gradient cannot be stopped, only
found afterwards in the logged cells (section 3), and "it does not train" (section 7) is therefore a sentence. Every
published harness forbids backward passes; where training was compared it beat editing. A held verdict from this
bench cannot say whether the edit was reasoned or fitted without someone reading the cells.

**Serious flaws.**
1. The gradient level is unenforced (above).
2. The base arm is memoised per (item, batch shape, settings, weights id) in the loop (section 4, device 4), against
   the house's own device and against the 9% batch-and-hardware variation the research cites. It also contradicts its
   own layout of rows as item x (base + arms). Pilot 3 may rescue it; the default should be the other way.
3. The loop's default is three random nulls, and the driver may opt out. The design's own argument is that one draw
   bounds nothing; three bound little.
4. The verdict's random-site control is one draw, not matched on firing.
5. The worked example scores writers as `d @ W_down` with no post-feed-forward norm weight. That is not the
   instrument's write direction and it is the vault's known flaw (see factual errors).
6. Held items are files the owner must write, a few hundred per task. In practice that means few tasks, or packs the
   driver wrote, whose verdicts the design itself limits.
7. Phase 1 reaches the known answer through `mul` pushes "as the exact twin" of a column edit. Pushes spare the start
   mark by default and bit-identity with a DELETE on the GPU is not established (code digest).

### 3. future: The Plane Table. Score 6.5

**Would its results be believed?** By the owner, as far as `confirmed`. By outsiders, more than any other design's,
if increment 5 is reached: the published Qwen edit is believed because five parties re-scored the diff.

- A volunteer's card re-running a proof under the kit's intake pattern, an import path for other people's edits so
  they can be re-measured here, "pull this proof again" by bits, and replay without the card.
- The docket records the driver's prompt word for word **with a flag for whether it names a suspected fault**.
- Evidence level (reasoning, activations, attribution) is declared and never pooled.
- Tiers enforced at the route, forecasts hashed before the result, the harness applies the keep rule, nulls in the
  edit's own representation with reversed sign and a dose sweep, a standing panel, cheaper knobs as arms.
- Source against lowered payload, with `relower`: a way to ask whether a recipe survives a different packing of the
  same weights, which no public work has measured.

**Strongest.** Results that someone else can load, check and re-measure, and a record of what the driver was told.

**Weakest.** The measurement is specified less tightly than the formats. Formats are fixed "because changing them
later breaks every stored result" before any pilot has shown what a result needs to hold.

**Serious flaws.**
1. No route from an activation contrast to an edit direction until the sandboxed kernel in increment 4. The nine
   verbs are datum, read, stats, compile, proof, forecast, verdict, ledger, export; a source may name only `col`,
   `row`, `write_dir`, `unit` and `input`; "anything richer the agent computes itself", and a tools-only agent has
   nowhere to compute. The first agent sessions (increments 2 and 3) can therefore only edit along atlas neurons'
   own columns, which the vault's own note says are rarely heard, and cannot run the difference-of-means recipe that
   produced the one large published result.
2. No known-answer reproduction and no power pilot among P0 to P8. Every other design reproduces the -1.80 result
   first.
3. The number of nulls is "several". Looks at the confirmation tier are limited per claim, not per set, so one set
   can be worn out by many claims.
4. `stands against nulls` is a named verdict on practice items alone. A candidate chosen as the best of many on
   those items is being set against nulls that were not chosen at all.
5. Gain times retention is reported as one number beside the two. A product hides which term moved.
6. No plan for repeated sessions.

### 4. speed: The Proof Press. Score 6

**Would its results be believed?** `passed`, yes: it is the house registration with written answers. `confirmed
(teacher-forced)` and `lead`, not as specified.

- Good and cheap: matched nulls of two kinds, a dose ladder (0.01 as the rounding floor, then 0.25 to 2), the
  flipped sign and the cheaper knobs as rows of the same forward, in every call, unasked. A forecast field hashed in
  every plan. Tiers enforced at the route. The model's process runs typed plans only.
- The snapped (4-bit grid) form rides as one more row beside the float form, so survival on the grid is a column of
  every table. That is the most direct answer any design gives to "re-measure in the form that ships".
- The first-order score carries a per-layer standing checked against real forwards, in the manner of the steer check.
- Pilot 7 sizes the tiers from measured standard errors.

**Strongest.** The standard controls bundle and the snapped row.

**Weakest.** The statistics. An empirical p from C nulls cannot go below 1/(C+1): 1/16 in the standard controls,
1/8 on the confirmation set of the worked example (section 4.4: "edit, 7 nulls, base on 64 held-out prompts"). A
Benjamini-Hochberg correction "over everything screened in that family" (section 5) can then never pass at 0.05, so
either nothing is ever a lead or the correction is not what decides. The design does not say which p is corrected,
and "family" is left to whoever writes the plan.

**Serious flaws.**
1. The above. "Confirmed" at 1 in 8, in seconds, is a result word on weak evidence.
2. "Parting along stored answers" (device 4) compares an edited row with an answer stored from another batch shape
   and a cached decode path. The house's measured floor is that an unpushed answer departs from itself across shapes.
   Parting must be read against the base row of the same forme; the design flags near ties and keeps the comparison.
3. A backward pass is the default first step. None of the published harnesses allows one, so results are not
   comparable with them; the design's own expectation is that the first-order score fails for large edits through
   the saturating norm (pilot 6). It should be a declared level, off by default.
4. Throughput and several drivers at once multiply looks faster than any ledger correction specified here.
5. Much of phase 4 is a register of weight statistics (singular values, code histograms for 410 linears). The
   literature's weight-only readings validate for 25 to 59% of units and the instrument's own link signs are below
   chance (0.336). The design does not type register readings as leads that need a confirming forward.
6. No near-miss set. The vault's own example edit bleeds from spiders to dogs.

### 5. kernel: Levelling. Score 5.5

**Would its results be believed?** As screened leads, yes, and more readily than anyone's, because the surface is so
small. As `held`, only on trust in an attended session.

- The zero-edit row that must close to exactly nought on every forward, or the reading is refused.
- `delivered`: how much of the edit survived the post-feed-forward norm, on every reading.
- P6 asks the question the other four assume away: is a random edit of equal weight length an equal disturbance?
  With the fallback of matching nulls on delivered displacement.
- P7 asks whether one column is ever heard before building tools for single columns.
- At least 99 controls at `confirm`; nothing `held` on next-piece odds alone; a count of looks on every result.
- The "reviewer's question about the core measurement, answered in advance" is what the house's reviews ask for.
- It went to the stored file for the known answer and states a prediction beforehand (5313 alone carries it).

**Strongest.** The pilots, and the plainness about what a reading is.

**Weakest.** For three increments the kept-back half, the level book and the driver's code live in one process run as
the user. The design says a `held` is "honest against mistakes, not against a driver that cheats". The literature's
finding is that rules which are not mechanisms did not hold for agents with free code, attended or not.

**Serious flaws.**
1. The above (section v, "What this leaves open").
2. `confirm` refuses a second call per registered id, but any number of edits may be confirmed against the same
   kept-back half; only looks at the working half are counted.
3. No reversed sign or dose ladder in the standing controls; no prompt, decoding or settings arm (only pushes); no
   near-miss set. A Qwen-style settings artefact would pass.
4. The first set is twelve contexts, six a half.
5. No guard on priors, no plan for session variance, no trust card until the second increment.

## Factual errors

| Design | Claim | What is true | Where checked |
|---|---|---|---|
| driver | The example cell finds writers by `(d @ tv.W.down(50).dense().float()).abs().topk(10)`, with `d` a stream-space difference of means. | The instrument's write direction is `unit(w_post (.) W_down[:, n])`; a stream-space direction must be compared with columns after the post-feed-forward norm's weight. The vault's UPDATE has this same flaw and the code digest lists it as a surprise. | `backend/atlas_ops.py:186-190`; `backend/larql_vault.py:512-521`; code digest, edit_paths surprises |
| driver, validity, kernel | "Rows are nearly free up to memory" / "controls ride almost free" / "rows are close to free", applied to teacher-forced forwards. | True of a decode step (0.091 s at 1 row, 0.109 at 64, 28 October 2026). A multi-position forward grows with rows: prefill 0.10 s at 1 row, 0.45 at 64, 0.60 at 96 (28 October 2026). Controls cost seconds, in proportion. Speed's own cost model gets this right. The three designs' cost tables are not wrong, only the sentence. | code digest, evaluators and runtime_weights numbers |
| driver | The base arm may be memoised because a row's bits depend only on its batch's size. | Found for the census writer in two processes on this card; not established for a hooked teacher-forced path, and the design itself lists it as pilot 3. Stated as a design device before it is known. | code digest, evaluators key facts and not-established |
| driver | `mul` "is already the exact twin of scaling or zeroing a column" (phase 1). | The same column change by the documents, except that pushes spare the start mark by default; whether a `mul 0` push with the start mark not spared and a DELETE are bit-identical on the GPU is not established. | code digest, evaluators key facts and not-established |
| future | "6 of 16 searches rightly ended with the original weights." | The checker struck "rightly": one discarded 9B candidate scored +1.01 held out, and the kept-base count is inconsistent within the paper. | research digest, lineage, takeaways to soften |
| future | Cost table rows "0.85 s" and "3.5 s" marked "Measured". | Measured are 287 s and 1,181 s for a layer of 336 batches (29 October 2026); the per-forward figure is a division and includes the read-out. Kernel marks the same figure as its own arithmetic. | `experiments/results/atlas_steer_timing/summary.json` |
| speed, driver | A provider's content filter refused a turn "that printed raw vocabulary". | The harness's own notice names two possible triggers, raw vocabulary tokens or long reasoning transcripts; the cause is not established. | research digest, wsp_paper corrections |
| validity | The trial preset "doubles the chance edits to 60 a family". | From 26 to 60 is not a doubling. A slip, nothing rests on it. | the design's own section v |
| speed | A family-wide Benjamini-Hochberg correction applied to leads that are ranked by an empirical p from 7 or 15 nulls. | Such a p cannot be below 1/8 or 1/16, so the correction as written cannot pass at 0.05. An internal inconsistency, not a misreading of a source. | the design's sections 4.4 and 5 |

I found no error in validity's or kernel's use of the literature. Both carry the corrections (the +29.69 and -0.35
case as a candidate the editor discarded on a base at 97.70%; the dev-above-held-out pattern as partly built in).

## What a synthesis should take, whoever loses

**From validity.** Items drawn after the edit is frozen. Two null families with dozens of draws, the other-site one
matched on write ceiling and on firing. The zero-edit and norm-undone canaries and canary briefs. The habitat and
own-answers batteries. Cheaper knobs as arms with their own verdict. The warden's fixed sentence, with "site chosen
by a reading not trusted at this layer" and "not compared with a trained adapter". The proof sheet with practice-best
beside proof and trial. Read once, counted always. Planted-failure checks of the reader. No memo of the base row.
Pilot 5 as the gate on building anything past the first phase.

**From driver.** A verdict worker that has never run agent code and loads the edit as data. A lifetime cap on each
confirm set, rotation, and feedback rounded to standard errors. A positive and a negative control as the bench's
first verdict, on a set large enough to split. The scripted no-driver recipe inside every verdict. A prompt that
names no fault, hashed. The free-cells against typed-verbs pilot. Repeated sessions and a second driver family as a
phase. `SCAFFOLD.md` and `--bare`. Gain and retention side by side, never multiplied. The save-time flag for
implant-shaped edits.

**From future.** The docket's flag for a prompt that names the fault. Declared evidence levels, never pooled.
Re-running a proof on a volunteer's card and importing outside edits to re-measure them. Replay, and pulling a proof
again by bits. Source and lowered payload with `relower`. The hook that refuses when the batch's rows do not match
the programs it was given (a reader that halves on out-of-memory would misalign every arm). Tensors named by role
from structure, not by module name.

**From speed.** The snapped form as a row beside the float form. The standard controls run unasked in one call,
with a 0.01 dose as the rounding floor. Base rows checked by bits against the first forward as a standing canary.
The forecast field with the driver's hit rate kept. Predicted against actual seconds on every job, and the unpacker's
fall-back counter read before and after. The stack's drift row. A per-layer standing for any first-order score. The
power pilot.

**From kernel.** The closing row and the refusal. `delivered`. P6 and P7. One text a forward with no padding where
bit-safety matters most. The count of looks. The folded cross-check that snapshots and restores all 410 linears, not
the three the vault's ROLLBACK clears. Refusing to read while any vault overlay is live. The section that answers
the validity reviewer in advance.

## What all five missed

1. **The baseline's settings.** The published 7.9% baseline was a thinking-mode artefact; the unedited model at the
   card's settings scored 44.0 to 49.7%. Gemma 4's template has the same switch and this instrument asks everything
   with thinking off, greedy, to 48 word-pieces. No design makes the chat settings part of an item set's identity, or
   runs the unedited model under the other settings as the first row of a brief. A prompt sentence and a logit bias
   are cheaper knobs; they are not an audit of the baseline. A behaviour should be admitted as a target only after
   the base has been read under each setting, and a gain should be quoted against the best of them.
2. **Obstacle or ability, measured and not only forbidden in wording.** What showed the Qwen gain was loop repair
   was a split: 576 of 579 rescues had been cut off at the cap, and accuracy on problems that already finished did
   not move. No design splits a gain by the state of the base's own answer (cut off, repeating, refusing, off-format,
   or finished and wrong) or reports rescues and regressions per stratum. Validity and driver forbid the words "an
   ability was added"; none computes the table that would show it.
3. **A reworded tier.** On IWIBench an edit that passed its practice prompts failed all 64 terminal targets after
   rewording. Every design's held-out items share the form of the practice items (the same generator, the same file
   split, the same pack). Near misses test that the edit does not spread; nothing tests that it holds when the target
   is phrased differently, in another format, or in a later turn.
4. **A cheaper-knob search with the same budget.** The edit gets hours of an agent's search; the prompt baseline is
   three sentences from the brief's author or a line in a task pack. The fair row is the same driver, the same
   budget, the same gates, allowed only prompts, decoding and steering. The same device run by access level
   (behaviour only; plus activations; plus weights; plus the atlas) is also the only way to learn whether the atlas,
   the project's selling point, helps a driver at all. Two cited studies found no uplift from giving agents
   interpretability tools. Driver and speed count whether atlas cards are used; none compares with and without.
5. **A trained small adapter as a built arm.** Where training was compared it won (8.1 points on average in one
   replication; a LoRA on 947 examples beat a hand edit on both off-target measures). All five exclude training.
   Only validity prints the hole. Until one adapter is fitted on the practice items and put through the same gates,
   no result can say the edit does something a small fit does not.
6. **Planted faults.** A per-row delta makes it trivial for the owner to lay a hidden edit on every row and ask the
   agent to find and repair it. That gives ground truth for discovery on the 31B with no public literature to recite,
   which is what issue #8 on the authors' logs asks for and what no 10B-plus result has. The designs have canary
   briefs for the null side and one known answer that is in the repository's own report. None plants a fault, and
   none injects an edit of known small size to learn what the proof can detect.
7. **Which space a direction lives in.** On Gemma a column of `down_proj` reaches the stream through the
   post-feed-forward RMSNorm (its weight and its denominator) and the layer scalar. "Subtract direction u from these
   columns", the published recipe, is not directional ablation of the stream unless u is carried through that norm.
   The vault already gets this wrong, driver's example repeats it, and no design defines the space of `project_out`
   or schedules, for each kind of edit, the check the lineage report asks for: that the weight edit reproduces the
   activation intervention it was derived from. Kernel's `delivered` and the `col_scale` against `mul` check are the
   nearest things.
8. **A labelled panel of unrelated tasks.** Low KL did not predict benchmark movement in one 35B ablation, and one
   replication lost 4.6 points on a benchmark nobody was watching. The repository has no capability suite. Validity's
   own-answers battery is a good proxy for retention and cannot tell a harmless rewording from a wrong answer. A
   small fixed set of unrelated questions with computable answers is missing from all five.

## Advice to the synthesiser

1. Build the shared kernel first and prove it with kernel's pilots (closing by bits, partial against whole forward,
   delta against folded, the known answer with 28 or more random tens, which null is fair, whether one column is
   heard) before any format, tool or page is fixed. Validity's pilot 5 is the gate on the rest.
2. Take validity's rule, nulls, batteries and sentence as the spine. Take driver's fresh verdict process and
   life-limited sets for anything that is a file, and validity's draw-after-freeze for anything that is a generator.
   Use both: generator draws for item noise, and an owner-written reworded set for generalisation.
3. Keep driver code out of the process that holds the model. A CPU sandbox with handles (validity, speed and future
   all have it) still lets a driver write a solver, and it is the only arrangement under which "no gradients" is a
   fact and results can be set beside the published ones. If attribution is wanted, make it speed's typed sweep at a
   declared level, off by default, with future's rule that levels are never pooled.
4. Give the tools-only agent validity's capture-to-direction handle from the first agent increment, and define
   direction edits in stream space through the post-feed-forward norm, with a per-op check against the activation
   twin.
5. Write the statistics down once: which p (empirical against nulls for standing, a paired test only for direction),
   at least 19 nulls for any named standing and 99 or more of record, both null families, the other-site one matched
   on firing, the brief as the family. Never memoise the base row. Read parting against the base row of the same
   forward.
6. Add the eight missing pieces above. The first two (the base under each setting; rescues and regressions split by
   the base's state) cost a few rows each and answer the two loudest critiques of the original post.
7. Record what the driver was told and whether it named the fault; plan repeated sessions before any gain is
   believed; keep "kept the base" as the expected verdict.
8. Correct the shared sentence about free rows: on a teacher-forced plate, half the rows on controls costs about
   half the seconds. It is still seconds.
