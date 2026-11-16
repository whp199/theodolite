# Levelling: a weight-space environment for Theodolite

One proposal, joined from five designs and three judgements. Written 14 November 2026. Design only: nothing in the
repository was edited, created or deleted; no server was started; the model was not loaded.

What I read in full: both digests, the five designs, the three written judgements, and the pilot note of today
(`experiments/results/weight_space_exploration/pilot/RESULTS.md`, with its two raw JSON files). What I opened in the repository to check a claim
before building on it is listed in section xiv.

Every figure carries one of these marks:

- **[measured, date]** a record in the repository or in today's pilot note;
- **[read]** read from the code or a file by me today, 14 November 2026;
- **[arithmetic]** my own sum from measured figures;
- **[literature]** from the research digest, with its checkers' corrections applied;
- **[not established]** nobody has measured it; a named pilot settles it.

Layers are counted from 0 (a page is the layer plus 1).


## o. In brief

**Is the feature feasible?** Yes, in a plain sense: the one mechanism all five designs share was measured on the
card today and works. A weight edit can ride one row of a batch as an added term on a linear's output; the unedited
model is another row of the same forward; nothing is left in force. A row carrying a zero edit is bit-identical to a
row with no hook, the term costs under 1% of a forward, and one candidate edit read against the base at all 175
positions of a paragraph costs about 0.04 s [measured, 14 November 2026, one run, one text, layer 44, no lens].

**What is the honest ceiling?** On a healthy 31B, most inquiries will end "kept the base". The published successes
removed an obstacle on small or damaged models; on stronger models held-out gains were a few points or nothing;
training beat editing where the two were compared; and on the one text measured today, silencing a single neuron
moved the model by about as much as rounding does. The environment will be good at testing a hypothesis about a set
of neurons or a direction quickly and honestly, and at saying no. It will not add abilities, and it should not be
sold on benchmark gains. Section ii says this at length and says what the feature is worth to the instrument even if
agents find little.

**What to build.** Start from the validity design (ranked first by two of three judges; 22 of 30 points against 20.5
for kernel, 19 for future, 18.5 for speed, 17 for driver [arithmetic from the judges' scores]). Give it the kernel
design's size and pilot ladder, the speed design's cost model, the future design's page and account of the claim,
and the driver design's fresh-process verdict and its first decision. In order:

1. a short list of repairs, and about two hours of pilots on the card, two of them registered;
2. a small kernel run by command line (an edit as a hashed value, one paired reading, a book written by the reading);
3. **the owner's first decision: build further or stop, on the audibility pilot;**
4. a person's surface: base, edit and a chance edit heard side by side on the vault page;
5. fresh-item readings whose rule the harness applies;
6. typed tools for the scientist;
7. trials of record in a fresh process, and export.

The name proposed is **Levelling**: in a level run the instrument measures only differences from a known point, both
readings are taken from one set-up and entered in the level book, and a run that does not close on its start shows
its error.


## i. What stands on measurement and what does not

| Fact | Standing |
|---|---|
| A per-row output-side term leaves untouched rows bit-identical, a zero term adds exactly nothing, and rank 1 to rank 64 terms add under 1% (0.3580 s against 0.3594 s at 32 rows x 30 pieces); the same on attention linears, a full-attention layer included | measured, 14 November 2026, one run |
| A whole teacher-forced forward costs about 0.08 s plus about 0.33 ms a row-position (64 rows x 30 pieces 0.670 s; 32 x 176 1.972 s). Rows are free only in a decode step (0.091 s at 1 row, 0.109 s at 64, 28 October 2026) | measured, 14 November and 28 October 2026 |
| Restarting from the edited layer on a kept stream of the same batch shape gives the whole forward's bits (layers 20, 40, 44, 52, 58, two prompts) at the share of layers run (32 x 30 from layer 44: 0.096 s against 0.361 s) | measured, 14 November 2026 |
| Full-vocabulary fp32 read-out costs 19 ms a row of 175 positions, more than half of a restarted reading; the model's own dtype is 3.5 ms and differs by a mean KL of 0.0013, too coarse | measured, 14 November 2026 |
| The same column change as a per-row delta and as a folded overlay differs by about 0.00045 nats a position (largest 0.0057) whatever the edit's size; the median effect of silencing one of the 63 hardest-firing neurons on that text was 0.00041 nats | measured, 14 November 2026, one text, three neurons |
| The vault's overlay costs about 0.3 ms a cloned matrix | measured, 14 November 2026 |
| Cold load without the lens 9.5 s; compiled unpacker installed and tested in 1.83 s; 17.97 GiB at rest; 8.37 GiB free | measured, 14 November 2026 |
| The ten layer-50 'she' neurons silenced moved 12 female contexts by -1.799 log-odds; ten random neurons, one draw, -0.020; the 'he' list -1.798; the two lists share neurons 549, 5313 and 8801 | read (`experiments/results/E6_tool_audit_04_vault.partial.json`; run 16 October 2026) |
| Generation under per-row deltas; anything with the lens loaded; layers below 20; a backward pass under the compiled unpacker; more than one text; audibility of sets and directions; which chance edit is a fair null | not established (pilots, section x) |

One reading of the table matters more than the rest. **The floor inside one forward is not nought.** A zero edit
adds exactly nothing and so cannot see that a real delta is rounded twice. The gap between the two forms of one edit
is the working floor of the fast path, and it is the size of a typical single-neuron effect. The design below prints
a floor with every result and confirms anything it claims in the folded form.


## ii. The ceiling, and what the feature is worth below it

**What the literature suggests such an environment will find on a strong model** [literature]:

- The one large published result (Qwen3.5-0.8B, MATH-500 from 7.9 to 48.1, a blog post of 2 November 2026) was loop
  repair: 576 of 579 rescued answers had been cut off at the token cap, and the 7.9% baseline was a thinking-mode
  artefact (the unedited model scored 44.0 to 49.7% at the model card's settings in one pre-registered re-run). The
  same procedure on a Gemma 3 270M that does not loop moved 6.8 to at most 7.7.
- On Qwen3.5-9B (arXiv 2611.09382, 6 November 2026) three of seven searches kept an edit, with held-out gains of
  +1.85, +3.03 and +1.54 points; the authors blame a strong base with little headroom. The best gain seen during a
  search exceeded the held-out gain in all 14 tabulated runs (partly built in by best-of-many on small batches), and
  three identical searches gave +29.45, +5.00 and +27.30.
- Once a fault and its layers were known, a solver did the weight-selection step in about a minute; where training
  was compared it beat editing (8.1 points on average in one replication; a small adapter beat a hand edit on both
  off-target measures in another).
- Closed frontier editors beat the unedited model on IWIBench (arXiv 2611.08647); the three open-weights editors did
  not. An edit that passed its practice prompts failed all 64 terminal targets after rewording.
- Giving agents interpretability tools gave no uplift in two studies and 10 to 20 points in a third, with a qualifier.
- No agent-driven weight edit above 9B was found in a preprint, none on this 31B, and no measurement of an edit
  carried across re-packing of a QAT checkpoint. These are absence claims from searches dated to 14 November 2026.
  Community work reaches 27B (one harness runs a 27B in a 4-bit format in 19.6 GB on an RTX 5090) and includes one
  agent's edit of Gemma 4 12B.

**What this instrument's own records add:**

- A heard one-column edit is rare: the most active neurons at pages 41, 46, 51, 56 and 59 were not heard on the
  vault's example question at strengths 3 or 30 [recorded, ROADMAP]. Today's median single-neuron effect sat at the
  rounding floor [measured, 14 November 2026, one text].
- A set of ten neurons moved a pronoun pull by 1.8 log-odds [measured, 16 October 2026, 12 contexts, one random
  draw], and the vault's one example edit bleeds: it turns "A dog has four legs" into "six legs" (first-step KL
  6.84) as well as the spider question it was made for [read, `vault_example_edit/probe.json`, 29 October 2026].

**So, plainly.** Expect it to find: sets of neurons and directions whose removal or scaling shifts a narrow
behaviour (a pronoun pull, a word preference, a format default), with the bleed measured; obstacle-like faults if
any exist under some chat setting; planted faults; and a great many honest nulls. Do not expect: a new ability; a
benchmark jump on a healthy 31B; a single neuron that matters by itself, except rarely; an edit that beats a small
trained adapter; an open-weights driver that does the work; or any statement about the 4-bit grid before the grid is
measured.

**What it is worth to the instrument if agents find little:**

1. The vault's missing half as a library: an edit with a file, a hash, composition, undo by construction, and the
   unedited model in the same forward. These are gaps the code digest lists one by one.
2. A causal reading that needs no dose. Thirty-nine of sixty layers have no calibrated push. Scaling a neuron's own
   write column by 0 or 2, read on its stored passages against matched columns, pushes nothing. If the audibility
   pilot shows such edits are heard, that is an atlas product for the layers the census cannot reach; if it shows
   they are not, that too is a fact about the atlas. (Cost: about a second a neuron at mid depth, so a sample of
   about a thousand neurons a layer in a quarter of an hour, a whole layer in about six hours [arithmetic,
   unmeasured].)
3. A learning aid: base, edit and a chance edit heard side by side, with rubrics saying why each row is there.
4. Published time and cost per test on a 31B on one card, which no source gives [literature].
5. Measurements that are new in kind for this object: the gap between a side delta and a folded weight, and later
   the grid.
6. A short list of repairs to default-visible readings (section xi) that should be made whatever is decided here.


## iii. What the environment is

**The thesis.** An edit is an argument of a measurement, never a state of the server. Because nothing is ever in
force between calls, no index, census, stats unit or page reading is ever taken on an edited model, there is nothing
to roll back, and every row of every reading names its own edit.

**Which process holds the model, by phase.**

| Phase | Process | Why | What it costs |
|---|---|---|---|
| Pilots and phase 1 | A plain command-line script with the server stopped, as census jobs run today. It installs the compiled unpacker and frees the J++ lens with `rt.jpp_lenses.clear()`, as `scripts/census_run.py:135-150` does [read] | No long-lived process to restart on every code change; 8.4 GiB of headroom; no dependence on the lab daemon, which has no cancel, runs jobs on its event loop, loads both lenses and reloads only labkit (`experiments/lab/labd.py:161, 189-215`) [read] | About 11 s of load a call (9.5 s + 1.83 s [measured, 14 November 2026]); the instrument is down for the minutes a sitting lasts. One call may hold many readings |
| Phases 2 to 4 | The instrument's server, under `_GPU_LOCK`, one forward a lock hold (`atlas.on_gpu`, `backend/atlas.py:148`) | The pages stay up; the atlas, the vault page and the scientist's registry are there | The lens stays loaded (sightings need it), so about 5.6 GiB of headroom, not 8.4; every code change is a server restart; shapes are smaller |
| Trial of record | A fresh process that has run nothing else, loads the edit as data, and exits | The house's own registration pattern; the natural home of the folded form | 25 to 45 minutes and the server down [arithmetic] |

The lab daemon is not used and not changed. No keeper, no bench worker, no second agent surface is built.

**Components.**

| Component | What it does | Stands on (existing) | New code |
|---|---|---|---|
| `backend/weight_edits.py` | The edit as a frozen value: terms, builders, hash, save and load, the compile-time gate, the context manager that installs per-row hooks | the difference-term arithmetic of `int4_cpu.with_overlay_terms` (`backend/int4_cpu.py:138-164`); the hook discipline of `atlas_ops.installed`; `gguf_weights.dense_cols` / `dense_rows` as the one reader of base weights; `atlas_ops.Weights.write_dirs` and `write_ceiling` | about 500 lines |
| `backend/levelling.py` | One paired reading: rows, capture, restart, read-out in two classes, chance edits, floor rows, `delivered`, the folded twin, the cost estimate | `atlas_ops.forward_from_layer`; `causal_check.logits_fp32` (chunked by rows); `census.control_dirs`; `mcp_scientist.paired_statistics`; `atlas_ops.write_rows`, wrapped and never edited (kit 4 is live) | about 700 lines |
| `backend/level_book.py` | Append-only, hash-chained JSON lines written by the reading itself | `census.write_json_atomic`, `file_sha256`, `runtime_record` | about 150 lines |
| `scripts/levelling.py` | `estimate`, `run plan.json`, `book` | the census scripts' frame | about 200 lines |
| `scripts/check_levelling.py`, `scripts/verify_levelling.py` | CPU checks and card checks | `scripts/atlas_stand_in.py` for plumbing; the packed bf16 stand-in of `scripts/check_dequant.py:150-154` for anything by bits; the four-part edit check of `scripts/check_gguf.py:825-874` | about 600 lines |
| Phase 2: routes and a tab on the vault page | Three voices and the table under them | the vault's edit form (`EditPanel.tsx`), the push tab's three-voice `Thread`, `streamLines`, `Rubric`, `Apparatus` | routes; one tab |
| Phase 3: briefs and the reader | Briefs, generators, draws after the freeze, the rule, batteries, the sheet, a forward-by-forward job with cancel | the intake's draw from `os.urandom` (`scripts/census_intake.py:332`) [read]; `passage_kept_back`; `census.distinct4`, `first_departure`; `field_notes.propose` / `decide` | the largest piece |
| Phase 4: six tools | Typed tools in the scientist's one registry | `scientist_tools.TOOLS` and `call`; `from_page` / `to_page`; the prompt builder | tools, a one-page method file |
| Phase 5: `scripts/levelling_trial.py` | The trial of record and the export | the registration folders under `experiments/results/*/prereg/`; `export_lora.write_lora` | script and folder generator |

**Relations.** The vault stays as it is; a levelling refuses to read while any overlay dict on any of the 410
linears is non-empty, so a vault edit can never leak into a reading. The atlas and census are read as facts about
the base and labelled so. No pinned file is touched; everything binds from outside. The library takes a runtime and
does not know which process it is in, so the same code serves the script, the server and the trial. The level book
lives outside the scientist's work folder, where no driver's sandbox can write. A logged session can be replayed
with no model, and that replay is the regression test.


## iv. What an edit is

**Representation.** An edit is a list of terms. A term names one linear and holds an additive difference from the
shipped tensor, in one of three kinds: `cols` (vectors added to chosen columns: what a neuron writes), `rows`
(vectors added to chosen rows: what a neuron reads), `lowrank` (B·A, rank at most 64). "Set" is stored as new minus
old, where the new entries are first rounded to bf16 as the reference forward would lay them on. Composing edits is
adding them, the rule llama.cpp applies to adapters served together (`scripts/export_lora.py:50-55`).

**Builders in the first increment**, because one neuron alone looks inaudible: `silence`, `scale` and `set` on a
set of columns; and `remove_direction(layer, neurons, u, dose)`.

**Which space a direction lives in.** A column of `down_proj` reaches the stream through the post-feed-forward
norm's weight: the instrument's write direction is `unit(w_post ⊙ W_down[:, n])` (`backend/atlas_ops.py:186-190`)
[read]. The vault's UPDATE adds a stream-space direction to the raw column without that weight
(`backend/larql_vault.py:513-523`) [read], and one design's worked example repeated the fault. Here `u` is a
direction in the stream, `writers(layer, u, k)` ranks neurons by `u · (w_post ⊙ W_down[:, n])`, and
`remove_direction` changes each chosen column so that its write as the stream receives it loses `dose` times its
component along `u`. The norm's denominator and the layer scalar still stand between this and an activation-level
projection, so the op is admitted only with a measured check against its activation twin. Whether small entries of
`w_post` make the raw change large is not established; pilot 0 reads the sixty weights.

**How it is applied.** A forward hook on the linear adds, for the rows that carry a term only,
`y[row] = (y[row].float() + dose[row] · d).to(y.dtype)`: summed in float32, rounded once. Each row has its own dose,
so a dose ladder costs rows, not forwards. The hook refuses loudly if the batch it sees has a different row count
from the one it was given programs for. It acts at every position, the start mark included; panels mask pads.

**Three forms, never mixed, named on every reading.**

| Form | What it is | Use |
|---|---|---|
| delta | 4-bit base product plus a float side term, per row | every practice and fresh reading |
| folded | the change written into the unpacked bf16 weight for the whole batch, through the overlay dicts; base and edited as two forwards of one shape from one kept stream | the cross-check from the first day; the form of a trial of record. The kernel snapshots and restores the dicts on all 410 linears; the vault's ROLLBACK clears only three projections a layer (`backend/larql_vault.py:556-567`) [read] |
| exported | a LoRA adapter served by llama.cpp over a GGUF | phase 5; another object (the lmstudio QAT Q4_0 file differs from the checkpoint in all 410 linears), re-measured there |

On the 4-bit grid: not built. A `down_proj` column shares each of its 5,376 scale groups with 31 other neurons
(derived from the layout). A low-rank term has no folded path until one is built, so it cannot go past a lead.

**The compile-time gate.** Refused: non-finite numbers; an index outside the loaded model (the vault clamps
instead); the final norm; a positive uniform rescale of a whole block's `down_proj` or of every head (the norm
undoes it); `v_proj` on the ten full-attention layers. **Attention is gated off until its facts are in the gate and
a card check has passed**: q, k and v each pass a per-head norm, so a positive uniform scale of a head's q, k or v
block is undone; and on the ten full-attention layers the values are the `k_proj` output, so a `k_proj` edit there
is a value edit too (`transformers` `modeling_gemma4.py:1216-1238`) [read]. Embedding, norm and layer-scalar edits
are unbuilt; an output-side embedding edit would not be heard in writing, because the writer takes its logits
outside any module (`backend/atlas_ops.py:493-499`) [read].

**Identity and file.** The edit id is the sha256 of a small canonical manifest plus the raw arrays. The manifest
names the base twice: the weights card, and a sha256 of the packed words and scales of each touched tensor (the card
names files, not tensors). Saved as `atlas_data/edits/<id>/` with `manifest.json` (no weights; may be committed
beside a registered result) and `payload.safetensors` (pieces of Gemma's columns; stays local). A reload verifies
both pins and refuses on a difference. This one manifest is versioned and nothing else is frozen until an edit is
known to be heard.

**Reported with every edit, by arithmetic alone:** relative change per tensor, columns and rows touched, each
changed column in write ceilings, the share of entries that would round away in bf16.


## v. One levelling, and what it costs

**The rows of one forward, on one item** (one text a forward, no padding, in the first increments):

- the **base row**, untouched;
- the **zero row**, a term of zeros at the same site; it must equal the base row bit for bit or the reading is
  refused and only the refusal is written;
- the **floor row**: the edited linear's output multiplied by 3 and re-rounded, which the following norm undoes
  (for `gate_proj`, where no norm undoes it, the edit at a dose of 0.01 instead). What it moves is the floor of the
  delta path, in the metric's own units;
- the edit, its reversed sign and its other doses;
- **chance edits**, two families (section vi);
- cheaper knobs as arms: a push is an independent hook and can share a forward with an edit.

**Two classes of read-out, budgeted apart.** A logit difference or a group log-odds needs a few rows of the
unembedding and is nearly free; it is the default. KL, a log-probability and top-piece agreement need all 262,144
logits in fp32 at every scored position: 0.11 ms a row-position [arithmetic from 19 ms a row of 175 positions,
measured 14 November 2026]. `logits_fp32` allocates rows x 262,144 in one piece (`backend/causal_check.py:133-139`)
[read], so the reader chunks by rows and reduces on the card. KL is always KL(base ‖ edited), in nats, non-finite
logits masked, and the record says so.

**Capture and restart.** A practice reading captures the stream entering the lowest edited layer in a one-row
forward and replays it to every row; the paired differences are exact with respect to that stream, and the record
says the lower layers ran at one row. A fresh reading runs its first forward on an item whole and restarts later
forwards on that item from the stream kept at the same shape, which is the case measured exact today. Pilot C
decides whether the one-row capture may serve fresh readings too. Only the one boundary needed for the item in hand
is kept, on the card (32 rows x 48 pieces is 16.5 MB [arithmetic]). If a forward runs out of memory the reading
halves its rows and installs its programs again, with a base row in every forward; it never lets a reader halve a
batch under the hooks.

**The core measurement's weak points, said in advance**, as the house's reviews ask. It is a paired difference in
fp32 odds between rows of one forward. Teacher-forced odds are not behaviour (hence the three voices, and a written
gate at trial). The delta form is not the folded weight (the floor row; the folded twin). A chance edit of equal
length may not be an equal disturbance (pilot D). The base row of a one-row capture is no whole forward's row (pilot
C). Small sets have little power (generators, and a known edit at shrinking doses). And everything measured so far
is one run.

**The cost model**, refitted to today's constants and checked on every job by logging predicted against actual
seconds: a whole forward of N row-positions costs about 0.08 s + 0.33 ms x N; a restart from layer L costs that
times (60 - L)/60; full-vocabulary read-out adds 0.11 ms a scored row-position. Memory above the model is 0.35 to
0.75 MB a row-position [arithmetic from the peaks of 21.9 GiB at 64 x 176 and 19.3 GiB at 64 x 30, measured 14
November 2026]. The unpacker's fall-back counter is read before and after every job; a silent fall-back to the
reference unpacker costs about seven times (`backend/w4a16_fast.py:205-229`) [read].

| Reading | Shape | Seconds | Standing |
|---|---|---|---|
| One base row and 63 single-neuron edits, every-position KL | 64 rows x 176 pieces, from layer 44 | 2.41 (forward 1.09) | measured, 14 November 2026 |
| The known answer | 12 items x 38 rows x about 30 pieces, layer 50, log-odds | about 2 with one-row capture; about 5.5 as whole forwards | arithmetic |
| A practice reading that can earn "lead" | 16 items x 45 rows x 48 pieces | about 5 from layer 44; about 13 whole | arithmetic |
| A fresh reading with its batteries | 96 items x 89 rows x 48 pieces, plus the batteries | 2 to 4 minutes | arithmetic |
| Three voices heard | 3 rows x 48 new pieces | about 5 | arithmetic from the decode step (28 October 2026); generation under per-row deltas is not established |
| A trial of record | fresh process, 256 items, 99 + 99 chance edits, folded twin on a subsample, written gate, reworded set | 25 to 45 minutes | arithmetic |

A driver's own turn took 20 to 70 s in the published logs [literature], so the card is not the slow part of an
agent's loop. Half the rows on controls costs about half the seconds; it is still seconds.

**Who else can run it.** With the lens freed the model rests at 17.97 GiB [measured, 14 November 2026], so a
volunteer's headless 24 GB card has about 5 GiB for a reading: forwards of up to about 6,000 row-positions
[arithmetic; not measured on such a card].


## vi. What stops the agent, and us, fooling ourselves

**Three grades, in plain words.**

| Grade | Items | Who applies the rule | The words it may earn |
|---|---|---|---|
| practice reading (seconds) | practice items the driver may read | nobody; it claims nothing | *lead*, and only with at least 19 chance edits a family; otherwise a rank and no word |
| fresh reading (minutes) | drawn after the edit and the rule are hashed | the harness | *held*; *held, matched by a cheaper knob*; *no better than chance edits*; *works but harms*; *did not hold*; *kept the base*; *void*. A held edit is still a lead |
| trial of record (under an hour) | fresh draws and an owner-written reworded set, in a fresh process, folded form, house registration | the registered reader, once | *passed* or *failed*; only the owner counts it in |

Nothing is called a proof. The house's public text says "a lead, not a proof" (`VOLUNTEERS.md:338`) [read].

**Items and seals.** A **brief** fixes, before any search: the behaviour in plain words; the metric; an item
generator (frames and slot lists) and a near-miss generator; the chat settings; the least effect that would matter;
which tensors may change; the budgets; the gradient level (none); and the text the driver is shown, word for word,
with a flag for whether it names a suspected fault or site. The owner writes a brief or keeps one the agent
proposed. Starter briefs come from data already on disk: the E6 frames and names, the 1,218 study contexts, the
census questions with stored answers, the corpus windows with their kept-back fold, the pattern pilot's texts and
near misses.

Fresh items are drawn from `os.urandom` after the edit id and the rule are hashed into the level book. The driver
gets aggregates only, never an item or a per-item row, with the effect rounded to units of its standard error. That
seal holds against any reader. A seal that is a file does not: **under Codex the sandboxed shell "can read files
elsewhere"** (`backend/scientist_chat.py:103`, `README.md:692-695`) [read], and three of the designs said the
opposite. So the default is fresh draws; a file-held set (the reworded set) is used only at a trial of record, gives
at most five readings in its life and is then rotated, and every sheet names which seal it had. A fresh draw guards
item noise and best-of-search inflation. It does not guard rewording, which is why the trial has the reworded set.

**Chance edits, two families, always both, never switched off.**

- *Another site, the same rule*: the same operation on other neurons of the layer, matched on write ceiling and on
  how hard they fire on these very items (one base pre-pass). About nine readers in ten sit with the gate shut at
  an ordinary position [measured, wiring check], so an unmatched draw flatters every edit of an active neuron.
- *The same site, another content*: for a direction removal, a chance direction removed at the same dose, drawn as
  the census draws its "columns" controls; for a silencing, an added write of equal stream-side length in a chance
  direction (rotating the column would itself silence the neuron, so that is no null).

Whether equal weight length is an equal disturbance is not assumed: the audibility pilot compares `delivered` and
neutral-text KL for chance and real directions, and nulls are matched on `delivered` if length fails.

**The statistics, written once.**

- Standing against chance is an empirical p, (1 + chance edits at least as strong)/(C + 1), a family. To stand, an
  edit must exceed every chance edit of both families: 19 a family for *lead*, 39 for *held*, 99 for *passed*.
- A paired sign-flip test over items is used for direction only.
- The brief is the family. The sheet prints "the k-th fresh reading on this brief" and the family-wise bound
  beside every p, and how many candidates were read on the practice items first. No correction is run over
  empirical p values from a handful of nulls; that cannot pass and was an error in one design.
- The driver's forecast is hashed before the result and scored for calibration. It is not the bar. The bar is the
  brief's least effect or five times the floor row, whichever is larger. (Five is my recommendation; the owner
  fixes it before the audibility pilot.)
- The base row is never memoised. Parting is read against the base row of the same forward, never against a stored
  answer.

**Batteries on every fresh reading and trial**, each set beside the chance edits' range, each with a bar measured
and hashed before first use (none is invented here): neutral text (KL and top-piece agreement on a fresh draw of
kept-back corpus windows); **habitat** (the same on the stored passages where the edited neurons fire hardest,
which only this instrument can run); near misses; the base's own answers (the share kept word for word, which
needs no labels and claims only retention); and a small **labelled panel** of generated questions with computable
answers, because low KL did not predict benchmark movement in one outside ablation and an own-answers count cannot
tell a harmless rewording from a wrong answer. Gain and retention are shown side by side and never multiplied into
one number.

**Cheaper knobs, as arms of the same forwards**: the activation twin (`mul` for a column scale), a matched
steering push with the start mark not spared, the brief author's prompt sentences, a decoding bias, and a scripted
recipe with no driver in it (difference of means on the practice items, top writers, three doses), so that the
agent's own contribution has a baseline. A verdict can cut a claim down without failing it. Two additions answer
the loudest critiques of the original post and cost a few rows each:

- **The base under each chat setting.** The item set's identity includes thinking on or off, the system prompt and
  the decoding. A behaviour is admitted as a target only after the unedited model has been read under each setting,
  and a gain is quoted against the best of them.
- **Obstacle or ability, measured.** For written metrics the sheet splits rescues and regressions by the state of
  the base's own answer: cut off, repeating, refusing or off-format, finished and wrong, finished and right.

**The harness tests itself.** Its first fresh reading is a positive and a negative control: *held* for the ten
layer-50 neurons and not for ten random ones, on a generator built from the E6 frames with many more names, since
twelve contexts cannot be split. Then planted-failure checks of the reader (a planted overfit must read *did not
hold*, a planted harm *works but harms*); canary briefs with no possible fix; and **planted faults**: with per-row deltas the
owner can lay a hidden edit on every row and ask the driver to find and repair it, which gives ground truth for
discovery on the 31B with no public literature to recite. The read tools then serve the planted state. An edit of
known size at a shrinking dose says what a fresh reading can detect.

**What a held edit may say**, filled by the harness and nothing else:

> On [date], [card], [unpacker], in delta form: edit [id] to [tensors] of base [weights id] moved [metric] on [n]
> items drawn from [generator] after the edit was frozen by [effect, interval]; the floor row moved it [f];
> [k of C] other-site and [k of C] same-site chance edits moved it as much (the [k]-th fresh reading on this brief);
> the reversed edit moved it [x]; neutral text, habitat, near misses and the labelled panel [figures against the
> chance edits' ranges]; base under each setting [figures]; rescues and regressions by stratum [table]; a prompt
> reached [..], a decoding change [..], a push [..], with no search over cheaper knobs; not compared with a trained
> adapter. Site chosen by [reading], [trusted | not trusted] at layer [l]. The driver's prompt [named | did not
> name] a suspected fault.

It may not say that the model is better, that an ability was added, what a neuron means, or anything about the
folded, exported or grid form without that form's own measurement.

**Registration.** The hashed entry in the level book before the draw is a light registration and earns only *held*.
*Passed* needs the house's own: a folder generated from the book entry (rules, exact runner and reader, inputs,
commit, time, SHA256SUMS), committed, run from a clean checkout in a fresh process with the server down, read once.
The trial checks that the base is untouched by a checksum of the packed words and scales on the card, empty overlay
dicts and hook lists on all 410 linears, and the unpacker as found. A one-prompt canary would not do: the vault's
own example edit, in force, left "Tell me about yourself" at a first-step KL of exactly 0.0 [read, probe.json, 29
October 2026].


## vii. The agent interface

**From phase 1, attended.** The owner's own Claude Code session writes a plan file and runs
`scripts/levelling.py run plan.json`. That is the trust the lab daemon had for the study's 314 jobs, and nothing
new is exposed.

**From phase 4, typed tools in the scientist's one registry**, so Claude Code, Codex and any OpenAI-compatible
endpoint get them at once: `brief` (what is open, budgets left), `weights` (bounded slices, norms, writers along a
direction, pair products; every reply typed a lead with how to confirm it), `capture` (activations on its own
texts, and the difference of means between two sets kept as a direction handle), `draft_edit`, `level` (a practice
reading; with `fresh` it registers and runs a fresh reading), `book`. The capture-to-direction handle is there from
the first agent increment, because without it a tools-only agent cannot run the recipe behind the one large
published result. Tensors stay server-side; replies are small tables and handles; layers pass through `from_page`
and `to_page`; the trust card rides on every reply.

**No agent code runs in the process that holds the model.** That is the only arrangement under which "no
gradients" is a fact and results can be set beside the published ones. Free code beyond the owner's attended
session waits for a boundary at the operating system (section xii).

**Budgets**, on every reply: GPU-seconds under the lock, fresh readings a brief, and tokens and dollars against a
ceiling the owner approves ahead, as the house budget rule asks for rented cards. No design estimated a session;
public figures for one run range from about $15 to $312 [literature].

**State** is files: the level book, the edits, the briefs, the driver's notes. A session starts from a page
generated from the book with two headings, *measured* and *your earlier notes, not checked*. The driver's
conclusion is filed through a tool that takes only a book reference; its free prose is shown under "the
scientist's own words, not checked". There is no model judge anywhere.

**Thin on purpose, and written down.** No workflow, stages, councils or context machinery. `SCAFFOLD.md` lists
each convenience that is not a trust boundary with the assumption about the driver it encodes, and a bare switch
runs without them. Every result carries a harness card: the driver as served, its effort, tools, prompt hash,
budgets, reach.

**A stronger driver later** uses the same three gates (observe the live model, freeze an edit, ask for a verdict)
harder. What it cannot do without review is invent a new kind of operation; that is the price of the harness
applying the rule.


## viii. Safety and scope

- Nothing an agent does changes the model the server holds. A restart always comes up as the base.
- The harness will not: run agent code beside the model; bake into 4-bit codes; write a model file; push, publish
  or export by a tool; open a behaviour without a kept brief; touch what writes census answers.
- Export is the owner's command. A file that leaves the machine carries the base id, the edit id, the book entry
  with its nulls and discarded attempts, a notice that the weights are modified, the Apache 2.0 text, and a name
  that says derived from Gemma. An edit shaped like the published implant (a last-block neuron whose write is
  dominated by one word-piece's unembedding) is flagged at save.
- No recipe or brief for stripping safeguards or planting a trigger is shipped or kept. A weight-edit backdoor was
  shown on the bf16 release of this model on 11 November 2026 (its headline figure is one best cell on recorded
  trajectories) [literature].
- What the edited model writes, and every item string, is data when shown to the driver. Word-pieces print as ids
  with short ASCII; a provider's filter once refused a turn, and its cause (raw vocabulary or long reasoning
  transcripts) is not established [literature].
- Each phase lands on its own branch, is reviewed with a reader who questions the core measurement, and nothing is
  pushed without the owner.


## ix. Phases

Effort is in focused days of one person and is rough. Following the feasibility judge, the first increment's
figure is believed and later ones are planned at about double their designers' figures. The house's own gates
(review with a validity reader, Lexicon entries, the ROADMAP section, REFERENCES entries) are counted apart.

| Phase | Delivers | Proves | Build | House gates | Card |
|---|---|---|---|---|---|
| **0. Repairs and pilots** | The repairs of section xi; the pilots of section x as scripts under `experiments/results/`, pilots B and D registered | Whether sets and directions are heard; what a floor is; the known answer | 3 to 4 days | 1 to 2 days | about 2 hours |
| **1. The kernel, by command line** (small) | `weight_edits.py` (cols, rows, lowrank; silence, scale, set, remove_direction), `levelling.py`, `level_book.py`, `scripts/levelling.py`, the two check scripts; `direction` and `writers` | A hypothesis about a set of neurons on the 31B tested in seconds with its controls in the same forward; predicted against actual seconds | 6 to 8 days | 2 to 3 days | about 1 hour |
| *The owner's first decision* | | | | | |
| **2. Three voices** | The library in the server; routes; a tab on the vault page where a person picks neurons and an operation and hears base, edit and one chance edit in one batch, with the table and the floor line beneath; a scan of which neurons in all sixty layers write along a direction; rubrics (why a row with no edit, what a chance edit is, why a lead is not a finding); replay of a stored reading without the card, and "take this reading again" | A person with no agent can see and hear what an edit does against chance; the instrument stays up | 3 to 4 weeks | 3 to 4 days | about 1 hour |
| **3. Fresh readings** | Briefs, generators, draws after the freeze, one rule preset, both families at 39, the batteries one at a time each with its measured bar, cheaper-knob arms, the settings audit, the strata table, the sheet (practice-best, fresh and of record side by side), planted-failure checks | A person gets a verdict in minutes that the harness wrote | 4 to 5 weeks | 4 to 5 days | 3 to 4 hours |
| **4. The scientist's tools** | Six tools; edits as proposals; budgets and the dollar ceiling; the method file; the prompt recorded with its flag; a scripted no-agent recipe (difference of means, top writers, three doses) through the same gates; first sessions on a planted fault and a canary brief | Whether a current driver finds a planted fault with typed tools and stays honest on a canary; that a planted cheat (a session that tries to read a held file or write the book) fails; cost per session | about 3 weeks | 3 days | sessions |
| **5. Of record, and out** | The trial script and folder; the folded form; the reworded set; adapter export with its manifest and a llama.cpp re-measure; a volunteer's re-run under the intake pattern | A claim survives a fresh process, the folded form and another card | 3 to 4 weeks | 3 days | hours |
| **6. Whether agents help** | Three or more sessions a brief; a second driver family; the same driver and budget allowed only prompts, decoding and steering; the same run by access level (behaviour, plus activations, plus weights, plus the atlas) | Session variance here; whether the atlas helps a driver at all | open | | dollars |

Phase 1 is useful if the work stops there: it answers what the vault's one heard question cannot, whether a set of
columns does more than matched chance sets. To the end of phase 4 is roughly three to four months of one person's
focused time [my estimate].


## x. Pilots to run first

Retired, because today's note answers them (they stay as lines in `verify_levelling.py`): closing by bits; the
cost of terms; restart exactness at the same shape; the overlay's cost; delta against folded on one text.

| # | Pilot | Cost | It changes the design if |
|---|---|---|---|
| 0 | No card. Re-run `scripts/verify_scientist_cli.py` on the Claude Code in use; read the sixty `w_post` weights for small entries; time the hash of a touched tensor; run the delta hook by bits on the packed bf16 stand-in | minutes | folder reach no longer holds for Claude Code (then file seals are void for every backend); small `w_post` entries (then `remove_direction` refuses those dimensions or takes its least-change form) |
| A | Generation under per-row deltas: two base rows and a zero row written for 48 pieces with the hooks round `write_rows` stay identical to the last piece; seconds a step | about 10 minutes | base rows part (then every written comparison carries twin base rows and a measured floor); a step costs clearly more than 0.091 to 0.109 s |
| B | **The known answer, registered, tolerance fixed first** (I suggest ±0.1 on the mean). The ten 'she' neurons at layer 50 on the 12 contexts, as a delta arm, as a folded overlay, and as a `mul 0` push with the start mark not spared; the three shared neurons each alone and together; the other seven; the 'he' list; 28 random tens. Said beforehand: the three shared neurons carry nearly all of it | about 5 minutes | the delta arm does not reproduce -1.80: stop, the kernel or my reading of the measure is wrong. The gap between the three forms sizes the floor in log-odds |
| C | Mixed-shape capture: paired differences from a one-row capture replayed at 32 rows against same-shape whole forwards, 16 texts, layers 5, 20, 44; restart exactness below layer 20 | about 15 minutes | they disagree by more than the floor (then fresh readings keep the same-shape rule and early-layer readings cost whole forwards) |
| D | **Audibility and the chance edits' spread, registered: the gate.** At layers 20, 36, 45, 55: 200 random and 20 most-active single columns at scales 0, 2, 4; 20 sets of 64 columns (top writers along a difference-of-means direction; census standouts where a census exists) and direction removals at three doses, at half, one and two write ceilings; both chance families at 39; both floor rows; `delivered`; neutral-text KL; a known edit at shrinking doses. On 64 generator items and 16 stored passages | 30 to 75 minutes [arithmetic] | almost nothing at one write ceiling or less clears its chance edits and five times the floor: the owner's stop decision. Chance edits several times louder or quieter than real directions: match on `delivered`. The spread sets item counts and bars |
| E | The floor rows: does the times-3 row read the same size as the delta-against-folded gap, on 16 texts and four layers? | about 10 minutes | it does not (then the printed floor is a folded twin on a subsample, at two forwards an item) |
| F | A backward pass under the compiled unpacker with a delta hook installed: does it run, seconds, GiB | about 10 minutes | nothing in phases 1 to 4; it informs the owner's gradient decision and the trained-adapter arm |
| G | With the J++ lens loaded, as in the server: peak memory and seconds for 64 x 48 and 32 x 176 readings with chunked read-out; time of a checksum over the 15.3 GiB of packed tensors | about 10 minutes | headroom under a heavy desktop is below about 2 GiB (then fresh readings run in a command-line sitting with the lens freed) |
| H | One attended afternoon: a current driver through the phase-1 command line on the known answer, unnamed, and one planted fault. Turns, seconds a turn, failed calls, controls run unasked, dollars | an afternoon; dollars | only budgets and the ceiling. One session an arm cannot decide typed tools against free code: three identical searches gave +29.45, +5.00 and +27.30 [literature] |

About two to two and a half hours of the card in all [arithmetic], between census layers.


## xi. Repairs before the feature

Faults in default-visible readings that the code digest found, none of which a design scheduled:

1. After a vault edit the stats index and the whole census are served with no stale flag (`backend/atlas.py:273-281`) [read].
2. Steer-lens layers 53 to 59 are shown as usable though no census checks them (`backend/atlas_index.py:318-328`) [read].
3. The vault clamps layer 99 to layer 59 instead of refusing (`backend/larql_vault.py:481, 500`) [read].
4. An overlay on an attention linear would survive ROLLBACK (`backend/larql_vault.py:556-567`) [read].
5. The README says 30 tools; the code holds 35 [read, counted by loading the module].
6. R4's sha256 pin on `backend/engine_runtime.py` has not held since 12 November 2026 (f1a0b4c3... against the pinned 69eed1b4...) [read, hashed today], while the documents still call it pinned.

And one gap, useful with no agent at all: exclusion on the home card rests on probing port 8765. One lock file
that the server, the census wrappers and the levelling script all take would mend it.

None of these touches what writes census answers. They are the owner's to approve.


## xii. Unbuilt until needed

| Unbuilt | Built when |
|---|---|
| A sandboxed CPU kernel for driver code (weights mapped read-only, no card, no network) | the owner approves free code for a driver other than the attended session, or phase 4 shows the typed tools bind |
| A separate operating-system account for held items | a Codex-driven or unattended session is wanted with a file seal |
| A single attribution backward as a declared level, never pooled with gradient-free results | the owner allows it; pilot F says it runs |
| A trained small adapter as an arm | the first result the owner wants to state as "an edit does what a small fit does not"; it needs a weight-gradient path that does not exist |
| The grid: a grid-rounded delta as one more row (named as an emulation in delta form, not a reading of the grid) and a bake | a grid pilot shows something survives |
| Embedding, norm and layer-scalar edits; attention edits | a hypothesis needs one, and its gate facts and card check exist |
| Source against lowered payload with `relower` held to a tolerance or one pinned CPU path, never by bits across machines | the unit of an edit is known and a second packing is wanted |
| A task-pack and result-pack format; an object protocol with tensors named by role from structure; the 12B as a second object | an outside harness or a second object is real |
| An importer for other people's published edits, so they can be re-measured here | the first outside edit worth re-measuring |
| A tensor register and precomputed spectra; a least-squares fit as a typed call | a driver is seen recomputing them, or a brief needs a solver |
| Writing from a layer upward on cached lower layers; packing several texts into a forward | predicted-against-actual shows the card, not the driver, binds |
| A lane on the CPU's int4 kernel while a census layer holds the card (0.45 s a word-piece, about 21 GB, another numeric class and, from the GGUF, another object [measured, 2 November 2026]) | the census blocks work for days on end; whether it slows a running census is not established |
| The base model against the instruct one (`ROADMAP.md:133-134`): a base column laid on as a delta, with no agent | the owner approves the download; the base weights are not in the cache [read] |
| Several drivers at once; a keeper process; fenced-block and command-line faces | never, on present evidence |


## xiii. Where the lenses disagreed, and what each settlement costs

1. **Where it runs** (kernel: the lab daemon; validity, speed, future: the server; driver: its own processes).
   Settled: a script first, the server from phase 2, a fresh process for trials. Cost: an 11-second load on every
   command-line call; restarts while developing in the server; less headroom with the lens loaded; free code has no
   home.
2. **Free code or typed tools** (driver against the rest). Settled: typed tools for any agent but the owner's
   attended session. Cost: a tools-only driver cannot write a solver or a custom analysis, and the typed-or-free
   question is not answered by measurement. It is put to the owner as a matter of containment.
3. **Person first or agent first** (validity and the owner judge against the ask as worded). Settled: person first,
   agent in phase 4. Cost: an unattended frontier driver arrives in months, not days. An attended one drives the
   command line from phase 1.
4. **Remember the base row?** (speed and driver: yes, to save a row; validity, kernel, future: no). Settled: never.
   Cost: one row in every forward and the base written again in every writing batch.
5. **Whole forwards or restarts for anything claimed** (validity: whole; the feasibility judge: restarts cost
   nothing). Both are partly right. A same-shape restart is exact, but the lower layers must first run at that
   shape, which is the whole forward when an item needs one forward. Settled as in section v, with pilot C. Cost:
   fresh readings on early layers run at whole-forward prices until pilot C says otherwise.
6. **Formats now or later** (future: fix six standards now; three judges: not before audibility). Settled: one small
   manifest. Cost: results stored in phases 1 and 2 may need one migration, and no outside harness can read a
   result until phase 5.
7. **Registration at two speeds** (speed: a fast one that earns "confirmed"; the house habit). Settled: the light
   one earns *held*, which is still a lead. Cost: a quotable finding takes a trial of record and the owner's time.
8. **Gradients** (speed: a backward as the default first step; the published harnesses forbid one). Settled: none.
   Cost: no first-order pre-filter; screening is by forwards, at about 0.04 s a candidate.
9. **Hold-out as fresh draws or as files in another process** (validity against driver). Settled: both, each named;
   fresh by default. Cost: the owner writes or keeps generators, which sets the pace more than the card does.
10. **How many chance edits** (kernel: at least 99 at confirm; validity 26 and 60; driver 19; speed 7 or 15; the
    science judge: at least 19 for any named standing). Settled: 19, 39, 99 a family. Cost: a "lead" needs about 45
    rows an item, so a practice reading is about 5 to 13 s, not one.
11. **Parting along a stored answer** (the feasibility judge would graft it; the science judge calls it a flaw).
    Settled for the science judge: allowed only as a practice screen with the base row teacher-forced along the
    same answer in the same forward.
12. **The snapped grid row** (science and owner judges would graft it; the feasibility judge calls it a float
    emulation). Settled: unbuilt, and named honestly if built. Cost: no grid figure in the early tables.
13. **A base-row-by-bits canary across forwards** (speed; grafted by the science judge; contradicted by the
    feasibility judge). Settled: by bits only inside one forward, or between forwards of one shape on one item.

**Where a judge was wrong or overstated, on checking.**

- The feasibility judge says `scripts/check_scientist.py:537-538` "asserts the exact tool-name list", so new tools
  trip it. The list it compares against is built from the registry itself (`names = [t["name"] for t in st.TOOLS]`,
  line 484) [read]. A tool added to the registry passes that line. What new tools need is their own check sections,
  routes in the check's stand-in server, and the documents' count put right. The lab daemon's three routes are
  hard-coded (`scripts/check_local_only.py:369`) [read]; this proposal does not touch the daemon.
- The same judge's "pays up to four times for nothing" is settlement 5 above.
- The same judge names `check_control_vector.packed_stand_in` as a packed bf16 stand-in. That one is float32 with
  only its down projections replaced by their unpacking (`scripts/check_control_vector.py:97-101`) [read]. The
  packed bf16 stand-in is in `scripts/check_dequant.py:150-154` [read].


## xiv. Every flaw and error the judges found: checked, then repaired

Checked by me today in the repository: `engine_runtime.py:907-933` (J++ loads unconditionally; `load_jlens` gates
only the second lens); `census_run.py:135-150`; `server.py:122, 1660`; `scientist_chat.py:101-104, 843`;
`README.md:692-695`; the E6 file (lists, effects, the three shared neurons); `vault_example_edit/probe.json`;
`atlas_ops.py:186-201, 493-499`; `larql_vault.py:481, 500, 504-524, 556-567`; `labd.py:150-235`;
`atlas_build.py:800-824`; `modeling_gemma4.py:1074-1076, 1203-1238, 1443-1444`; `atlas_stand_in.py:102-142`;
`causal_check.py:132-139`; `w4a16_fast.py:205-229`; `census_intake.py:326-336`; `atlas_index.py:125-130, 318-328`;
`atlas.py:146-153, 273-294`; `check_scientist.py:484-487, 538`; `check_local_only.py:369`; `VOLUNTEERS.md:51, 338`;
`ROADMAP.md:133-134`; the pilot's two JSON files; the Hugging Face cache; the R4 pin's hash; name collisions by
grep. All held except the three points in section xiii.

| Finding | Repair |
|---|---|
| "Rows are nearly free" applied to teacher-forced forwards (kernel, validity, driver) | The cost model of section v; controls cost seconds in proportion |
| `make_runtime(load_jlens=False)` leaves the lens out (driver) | It does not. The script frees it with `rt.jpp_lenses.clear()`; the server keeps it |
| The steer build's "0.85 s from layer 45, measured" (future) | A division, of forwards that started at layer 46 and include the read-out. Not used |
| Free code beside the model; a canary that "proves" the base untouched; SIGINT to a worker thread; three processes first (driver) | No free code beside the model; the trial's checksum of section vi; no worker built. If a sandboxed worker is ever built its cells run on its main thread |
| The base arm remembered; three nulls a driver may switch off; one unmatched random-site draw (driver) | Never remembered; chance edits not optional; both families, the other-site one matched on firing |
| Writers found as `d @ W_down` with no norm weight (driver) | Directions live in the stream, through `w_post` (section iv) |
| `mul` "the exact twin" of a column edit (driver) | Not assumed; pilot B reads the three forms side by side |
| Base-row canary across formes of different shapes; the panel costed without its read-out; a snapped row called a grid reading (speed) | Settlements 13 and 12; two read-out classes |
| p from 7 or 15 nulls corrected over a family; "confirmed" on 7 nulls (speed) | The statistics of section vi |
| Parting against a stored answer; a backward by default; several drivers; no near misses; register readings not typed as leads (speed) | Settlements 11 and 8; one driver; near misses in every fresh reading; every weight-derived reply typed a lead |
| "No public work has measured this", stated flat (speed) | "Not found in a search dated 14 November 2026"; one repository kept 1.03 of a change in another 4-bit format on a 27B; one Gemma 4 12B edit was checked after Q4_K_M [literature] |
| The filter's cause stated as raw vocabulary (speed, driver) | Two possible triggers; not established. The sanitiser stays as a cheap precaution, listed in `SCAFFOLD.md` |
| Phase 2 the largest bundle; owner-written everything (validity) | Split into phases 2 and 3 and planned at double; starter briefs from data on disk; the agent may propose a brief the owner keeps |
| Controls "ride almost free"; whole forwards for everything claimed (validity) | Corrected; settlement 5 |
| No reworded tier; the agent picks its own least effect; no record of a prompt that names the fault; "doubles to 60" (validity) | The reworded set at trial; the bar from the brief or the floor; the flag; 19, 39, 99 |
| The seal "holds at the folder reach" (validity, driver, future); "no shell" (driver, kernel) | Wrong for Codex. Fresh draws by default; file seals named and limited |
| "Proof", "sighting", "plate"; "errata" (validity, speed, future) | None used. "Errata" is the study's own word (`README.md:308-309`) [read] |
| Six standards fixed first; `relower` by bits; the form gap judged against 0.014 (future, driver) | One manifest; a tolerance; the gap judged against the effect claimed |
| No route from a contrast to a direction; no known answer; "several" nulls; looks capped a claim, not a set; "stands against nulls" on practice items; "rightly" kept the base (future) | The capture handle; pilot B; fixed counts; a brief-wide count and a five-reading life for a file-held set; no standing word past *lead* on practice items; "6 of 16 searches ended with the original weights" without "rightly" |
| The lab daemon as host; capture for all 60 boundaries in memory; mixed shapes (kernel) | A script; one boundary, on the card; pilot C |
| The kept-back half kept by convention; any number of confirms on one half; no reversed sign, dose ladder, settings arm or near misses; twelve contexts (kernel) | No *held* before the fresh-draw mechanism of phase 3; the counts above; all four added; sets from generators |
| Down for weeks with nothing shown; no public claim (kernel) | Sittings of minutes; the page in phase 2; section xvi |
| The first increment edits nothing; the instrument dark for hours; a third agent surface; a pilot that cannot decide (driver) | Phase 1 is the edit kernel; the server from phase 2; the scientist's one registry; pilot H only sizes |
| "No record of a real driver's behaviour at all" (driver) | A real scientist chat of 11 November 2026 is on record for its timing; no transcript-level evaluation exists |


## xv. What the judges say all five missed

| Missed | Dealt with |
|---|---|
| Today's pilots | Section i; the retired list in section x |
| The floor inside a forward is not nought | The floor row, the five-times rule, the folded twin |
| Read-out cost and metric choice | Two classes, section v |
| Attention is not the feed-forward block | In the gate, gated off, section iv |
| Which stand-in | The packed bf16 one for anything by bits |
| The lens | Freed in the script, kept in the server, pilot G |
| Checks that pin lists | Section xiii |
| Building inside a long-lived process | A script first; restarts counted from phase 2 |
| Pads, and the writer's bf16 logits | Pads masked; output-side embedding edits unbuilt |
| The baseline's settings | Part of an item set's identity; the base read under each |
| Obstacle or ability, measured | The strata table |
| A reworded tier | At the trial of record |
| A cheaper-knob search with the same budget; runs by access level | Phase 6. Until then the sheet says "no search over cheaper knobs" |
| A trained adapter as a built arm | Not built, and said on every sheet; trigger in section xii. Why not now: it needs an optimiser loop and a weight-gradient path, another access level, at a cost nobody has measured |
| Planted faults | Phase 3 for power, phase 4 for the driver |
| Which space a direction lives in | Section iv |
| A labelled panel of unrelated tasks | A generated one among the batteries |
| Codex reads the disk | Section vi |
| The vault form and the three-voice thread | Phase 2 |
| Faults left standing | Section xi |
| Nothing written back to the atlas | Section ii, item 2, if pilot D allows |
| The roadmap's base-against-instruct item | Section xii |
| No dollar ceiling | Section vii; decision 7 |
| No lane while the card is busy | Section xii, with the reason |
| Who else can run it | Section v |
| The size of "first" | Section xvi |
| The house's gates in the estimates | Section ix |
| Colliding names | Section xix |


## xvi. The public claim

"First on a 31B" would be true by model and not by kind: a public harness already runs a 27B in a 4-bit format on
the same class of card, and weight edits of this very 31B exist (abliteration builds; a backdoor study on the bf16
release) [literature]. What would be new in kind is four things: a causal atlas as the symbol table, with its
trust bands stated; every number paired in one forward with matched chance edits; time and cost per test published;
and the gap between forms, later the grid, measured.

The wording should follow `VOLUNTEERS.md:51` ("We found none published"): an absence claim from a search, with
its date, and with the 27B analogue named. One such claim in today's research was already found wrong. What would
make it true: a registered known answer reproduced; one recorded session with its level book, seconds and dollars;
a volunteer's re-run; and REFERENCES entries for the post and its follow-ups, which the repository does not yet
cite.


## xvii. The owner's decisions

| # | Decision | Choices | Recommendation |
|---|---|---|---|
| 1 | Build past phase 1? | (a) stop and keep the kernel as a research tool; (b) go on | Decide on pilot D, with the threshold fixed and hashed first. I suggest: go on if, at two or more of the four layers, at least 5 of the 20 column sets at one write ceiling or less beat every chance edit of both families and exceed five times the floor row |
| 2 | May driver code run on the card at all? | (a) typed verbs only; (b) free cells for any driver | (a), for every driver but your own attended session, until a boundary at the operating system exists |
| 3 | The repairs of section xi first? | yes; no; some | Yes, all six and the lock file, before phase 1 merges |
| 4 | Seals | fresh draws only; a separate account; file seals for Claude Code and endpoint drivers alone | Fresh draws now; a file-held reworded set only at trials, named as such; a separate account when a Codex or unattended session wants a file seal |
| 5 | The name and the words | Levelling; Collation; Traverse | Levelling, with level book, chance edit and brief as the only new Lexicon words |
| 6 | Gradients | none; one attribution backward as a declared level | None for now; run pilot F so the choice is informed |
| 7 | Budgets | a dollar ceiling a session and GPU-seconds a brief, approved ahead; log only | Approve ahead, as for rented cards |
| 8 | What may be shown on a page | held leads with their sheet; only passed trials | Held leads as leads, with the sheet; "passed" only of record |
| 9 | What may leave the machine | nothing; manifests with a registered result; adapters too | Manifests only until you write a policy; none exists today |
| 10 | The R4 pin | restore; re-pin at today's hash with a dated note; retire | Re-pin with a note, since the documents still say pinned |
| 11 | Which behaviours a brief may name, and who writes generators | | You write or keep each brief; none on safeguards or triggers |
| 12 | The claim's wording and the dated search on record | | As section xvi |
| 13 | Card time against the census; the lens freed in a sitting | | Sittings between layers; lens freed in the script only |
| 14 | Volunteers re-running a reading | yes; no | Yes, after phase 5 |
| 15 | The vault's UPDATE rebuilt on this kernel, with its direction through `w_post` | later; never | Later, after phase 2 |


## xviii. Risks

1. **It works and finds little.** The likeliest outcome. Decision 1 exists so that this is found out for two hours
   of the card and two weeks of work, not three months.
2. **The floor eats the effects.** If sets and directions also sit near the floor, the fast path ranks and cannot
   measure, and everything worth saying needs the folded form at one edit a forward.
3. **The chance edits are the wrong null.** Too quiet passes everything; too loud fails everything. Pilot D; a
   validity reader on every op.
4. **Three forms invite sliding** from "in delta form" to "the model". Every sheet names its form.
5. **The reader is new code and is now the instrument.** Planted-failure checks; house review.
6. **The owner is the pace.** Briefs, generators, bars and reviews are one person's. Phase 3 is where this bites.
7. **Generators are narrow.** An edit can hold on a family of frames and mean little outside it. The reworded set
   is small and owner-written.
8. **One card.** A census layer takes 13.2 hours [measured, 29 to 30 October 2026]. Sittings are scheduled.
9. **One run each.** Everything measured today is one run on one or two texts at one layer.
10. **Effort.** Every figure after phase 1 is a guess already doubled once.


## xix. Names

**Levelling** (recommended), with the **level book**. The words are free in the repository [read, by grep] and
carry no claim. The plain phrases base row, zero row, floor row, fresh items, delta form and folded form need no
Lexicon entry of their own; **chance edit** (the literature's matched null) and **brief** (its task) do. Four new
words.

Alternatives: **Collation** (two states of a text set side by side to find where they part); **Traverse** (a run
of legs that must close on a known point).

Not to be used: proof, proved, erratum, errata, sighting, plate, station, field book. Each is taken or says what
the thing is not.


## xx. The load-bearing claims, for a sceptic to attack next

| # | Claim | Kind | How to check |
|---|---|---|---|
| 1 | A per-row output-side term leaves untouched rows bit-identical and costs under 1% | hardware | Re-run on 16 texts, layers 5 to 58, with the lens loaded; today's record is one run at layer 44 |
| 2 | A restart from the edited layer is the whole forward by bits at the same shape | hardware | Layers 0 to 19, hooked rows, attention terms; `torch.equal` on final states |
| 3 | A one-row capture replayed at many rows gives paired differences that agree with same-shape forwards within the floor | method | Pilot C |
| 4 | A whole forward costs 0.08 s + 0.33 ms a row-position; a restart the share of layers run; full-vocabulary read-out 0.11 ms a row-position | hardware | Predicted against actual on every job; refit; pilot G for the server's state |
| 5 | Delta and folded forms differ by about 0.00045 nats a position whatever the edit's size | hardware | More texts, layers and batch shapes; today is one text, three neurons, a batch of 2 |
| 6 | The times-3 floor row measures that same floor | method | Pilot E; nobody has run it |
| 7 | Single columns sit at the floor, but sets of 64 and direction removals are heard | hardware | Pilot D; the whole design past phase 1 rests on it |
| 8 | A chance edit matched on weight length, and an other-site one matched on firing, are fair nulls | method | Pilot D: `delivered` and neutral-text KL for chance against real directions |
| 9 | The stored known answer is -1.799 against -0.020 for one random ten, and the two lists share 549, 5313, 8801 | code | Open the E6 JSON; then pilot B with 28 random tens and the three alone |
| 10 | `remove_direction` through `w_post` reproduces the activation-level projection it stands for | method | The per-op twin check on the card; `w_post`'s small entries from pilot 0 |
| 11 | J++ loads whatever `load_jlens` says; only `rt.jpp_lenses.clear()` frees it; the server needs it | code | `engine_runtime.py:907-933`; `census_run.py:135-150` |
| 12 | Under Codex at the default reach the shell reads files elsewhere, so file seals fail and fresh draws hold | code | `scientist_chat.py:103, 843`; `README.md:692-695`; then try to read a held file from a Codex chat |
| 13 | The forward honours overlays on all 410 linears and ROLLBACK clears three a layer | code | `engine_runtime.py:352-377, 429-436`; `larql_vault.py:556-567` |
| 14 | q, k and v pass per-head norms; on full-attention layers the values are the `k_proj` output | code | `modeling_gemma4.py:1216-1238`; a uniform head scale by bits on the stand-in |
| 15 | Writing under per-row deltas keeps rows aligned and base rows identical | hardware | Pilot A; nobody has run it |
| 16 | A tool added to the registry passes the check's list assertion; the daemon's route list is hard-coded | code | `check_scientist.py:484, 538`; `check_local_only.py:369`; add a dummy tool on the stand-in |
| 17 | To top every one of C chance edits in a family has chance 1/(C + 1) if the edit is exchangeable with them | method | Planted nulls through the reader; it fails if claim 8 fails |
| 18 | On strong models held-out gains were a few points or none; successes removed obstacles; training beat editing | literature | Read arXiv 2611.09382 and the replications in full; some were read as abstracts only |
| 19 | No agent-driven weight edit above 9B in a preprint, none on this 31B, no edit measured across re-packing of a QAT checkpoint | literature | A fresh dated search; one such absence claim was already found wrong |
| 20 | Phase 1 is 6 to 8 days and about 1,500 to 2,000 lines | method | Count after the build; the pilots' inline hook is the first evidence |
