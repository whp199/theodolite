# Judgement under one lens: feasibility against the code and the card

Written 14 November 2026. I read both digests and all five designs in full, then opened the repository to check
what each design leans on. Nothing was edited, started or loaded. A figure is marked *measured* (a record, with its
date), *read* (I read the code or file today) or *my arithmetic*.

## What I checked in the repository, and what I found

| Claim the designs lean on | Where I looked | Finding |
|---|---|---|
| A hook on a packed linear fires and can add a per-row term to its output | `backend/engine_runtime.py:352-377, 429-436`; `backend/atlas_ops.py:344-412` | Holds. The forward is bound on the instance, so `nn.Module` hooks still run; the existing `mul` push is already a pre-hook on `mlp.down_proj` and leaves untouched rows at x1 / +0. |
| The difference-term arithmetic exists | `backend/int4_cpu.py:138-164` | Holds, as described (float32 sum, one rounding). |
| `forward_from_layer` restarts at a layer with hooks firing | `backend/atlas_ops.py:1017-1043` | Holds. Under `torch.no_grad`, not `inference_mode`. Takes the stream, a mask and position ids. |
| The steer-lens forward is 516 rows, and ran beside the J++ lens | `backend/atlas_build.py:775-836`; `scripts/atlas_build.py:232` | Holds: 4 + 2 x 64 x 4 rows; the builder never frees the lens. It starts at layer l + 1, so "layer 45" timed 14 layers. |
| The base weights can be read the same way on any holding | `backend/gguf_weights.py:542-576` | Holds for plain, W4A16 and GGUF linears. |
| The lab daemon honours `THEODOLITE_UNPACKER`, runs under `no_grad` | `experiments/lab/labd.py:153-215` | Holds. It also hard-codes `make_runtime(load_jlens=True)`, runs the job on the event loop, and `/reload` reloads only labkit. |
| The compiled unpacker falls back under `inference_mode` | `backend/w4a16_fast.py:205-229` | Holds; counted in `.fell_back`. No `inference_mode` is used anywhere in `backend/` or `experiments/lab/` (grep). |
| R4's pin on `engine_runtime.py` is broken | `experiments/scripts/R4_common.py:101`; `sha256sum` today | Holds: f1a0b4c3... against the pinned 69eed1b4.... `labd.py` and `labkit.py` are not in the pin list. |
| The known answer (ten layer-50 neurons, -1.80) | `experiments/results/E6_tool_audit_04_vault.partial.json` | Holds to the digit: she@50 = [5313, 8801, 266, 121, 11503, 11724, 21151, 4165, 549, 2070], female contexts -1.799, random -0.020, the 'he' list -1.798. |
| The vault's example edit, KL 5.65, reference unpacker, 29 October 2026 | `experiments/results/vault_example_edit/probe.json` | Holds. The same record shows the edit in force leaving another question's first-step odds at KL exactly 0.0. |
| `write_rows` keeps every row to the end | `backend/atlas_ops.py:502-611` | Holds: finished rows stay in the batch, so per-row hooks stay aligned through decode steps. It chooses from bf16 logits computed outside any module (`_model_logits`, 493-499). |
| `make_runtime(load_jlens=False)` leaves the J++ lens out | `backend/engine_runtime.py:908-933`; `scripts/census_run.py:135-150`; `backend/server.py:122` | **Does not hold.** J++ loads unconditionally; `load_jlens` governs the second lens. The census frees J++ with `rt.jpp_lenses.clear()`. |
| Feed-forward block and attention shape | `transformers/models/gemma4/modeling_gemma4.py:1074-1076, 1203-1245, 1429, 1443-1444` | `down(act(gate x) * up x)` holds. Also: q, k and v each pass a per-head norm; on a layer with no `v_proj` the values are the keys; the layer scalar is an in-place multiply; the embedding's scale is inside its forward. |
| The Claude CLI confinement flags | `backend/scientist_chat.py:562-581` | Hold as the driver design quotes them. |
| `/usr/bin/bwrap`, `/usr/bin/systemd-run` exist | `ls` | Hold. |
| The stand-in | `scripts/atlas_stand_in.py:102-142` | Dense `nn.Linear`, float32 by default; its last layer is full-attention with keys as values. |
| Checks that pin lists | `scripts/check_scientist.py:537-538`; `scripts/check_local_only.py:369` | The scientist's exact tool-name list and the daemon's three routes are asserted. |

### A record the designs did not use

Beside the digests sits a pilot note, measured on the card today:
`experiments/results/weight_space_exploration/pilot/RESULTS.md`
(14 November 2026, 19:50Z and 19:54Z; compiled unpacker, no lens loaded, server stopped; one run each, one 30-piece
question and one 176-piece paragraph, edits at layer 44). I checked it against its raw JSON. Every design says nothing
was run and lists these as pilots still to do. They are done, and they decide several of the designs' own rules:

- **A per-row output-side delta is bit-safe and free.** A zero-delta row equals an unhooked row of the same shape by
  bits; untouched rows are identical; rank 1 to rank 64 terms add under 1% (0.3580 s against 0.3594 s at 32 rows);
  attention linears too, including a full-attention layer with no `v_proj`.
- **Restarting from the edited layer is exact**: final states bit-identical to the whole forward at the same batch
  shape (layers 20 to 58), time in proportion to layers run, no fixed floor.
- **The vault's module-wide overlay is cheap**: about 0.3 ms a cloned matrix.
- **A whole teacher-forced forward is not free per row**: about 0.08 s fixed plus about 0.33 ms a row-position
  (64 rows x 30 pieces 0.670 s; 32 x 176 1.97 s). "A step costs the same from 1 to 64 rows" is true of a decode step
  only. Peak memory 21.9 GiB for 64 x 176, which is about 0.35 MB a row-position above the model (my arithmetic).
- **The reading is more than half of an every-position reading**: 19 ms a row of 175 positions for full-vocabulary
  fp32 logits; the model's own dtype is 3.5 ms but differs by mean KL 0.0013, too coarse.
- **The two forms of one edit differ by a floor**: per-row delta against folded overlay, mean KL about 0.00045 nats
  a position whatever the edit's size (max 0.0057); per-position effects correlate 0.993 to 1.000. The median effect
  of silencing one of the 63 hardest-firing neurons on that text was 0.00041 nats: the same size as the floor.
- Cold load without the lens 9.5 s; 17.97 GiB at rest; 8.37 GiB free.

So the mechanism all five designs share works on this card at the speeds they hoped for. What separates them under
my lens is what each builds around it, how much, and in what order.

## Scores and ranking

| Rank | Design | Score | In one line |
|---|---|---|---|
| 1 | kernel (Levelling) | 8 | The smallest thing that could be built next week; its code citations are exact to the neuron list. |
| 2 | speed (Proof Press) | 7 | The only cost model that matches today's measurements; too much built on top of it. |
| 3 | validity (Proof Room) | 6.5 | A small first phase and no collision with the card; the second phase is the largest single bundle of new code in any design. |
| 4 | future (Plane Table) | 6 | No factual error found and an honest list of what is not established; fixes formats before the evidence and bundles too much into each increment. |
| 5 | driver (Traverse) | 5 | Three processes and a sandbox before any new measurement; one wrong fact about the lens; two mechanisms that do not do what is claimed. |

## Design by design

### kernel: 8

**Strongest.** The first increment is truly small: three library modules and two scripts, hosted in a process that
exists, with a known answer whose stored numbers I confirmed (-1.799 against -0.020, the exact ten neurons, the 'he'
list's -1.798). The folded cross-check is in the kernel from the start, and its rule P4 already says what today's
gap means: confirm in folded form; a low-rank term cannot reach `held` until it has a folded path.

**Weakest.** The host. The lab daemon has no cancel and no time limit, runs the job on its event loop, loads both
lenses unconditionally (`labd.py:161`), and `/reload` reloads labkit only, so a changed kernel module means a
restart or an explicit reload. The server is down for every session of the first two increments.

**Checked.** Per-forward costs: "24 texts x 32 rows from layer 45, 1.5 to 2 s" against the pilot's 0.096 s for
32 x 30 from layer 44, which gives 2.3 s (my arithmetic). Holds. The gate P3 (under 10 s) passes with room.

**Flaws.**
- It says rows are close to free and "the limit is memory, not time" for its partial forwards. Measured today, time
  is in proportion to rows and layers. The cost table survives because it was worked from prefill rates.
- Capture "at one row" then replay at 32 rows mixes two batch shapes below and above the edited layer. The design
  names this itself; the pilot's exactness was shown at one shape only.
- "Kept in process memory" for all 60 boundaries: 100 texts of 64 pieces is 4.1 GB (my arithmetic, 60 x 69 MB). It
  must be host memory, not the card.
- Single-neuron edits look inaudible (pilot, one text). The design's own P7 then moves `direction` and `writers`
  into increment one, which grows it a little.
- No driver but an attended Claude Code session for weeks; embedding edits left out.

**Effort.** 5 to 7 days for increment one is believable. The pilots it budgets 1 to 2 days for are mostly done.

### speed: 7

**Strongest.** The cost model. t = layers x (1.5 ms + 7.2 microseconds x row-tokens) was fitted from four records
and is the only design statement that a whole forward costs in proportion to its rows. Today's measurement is about
5.5 microseconds a row-token a layer (my arithmetic from 0.33 ms over 60 layers), so the model is right and about
30% cautious. Its refusal of dense caches stands on the same arithmetic. Its read-out line (0.05 to 0.1 s for 512
positions) matches the measured 19 ms a row of 175.

**Weakest.** Scope. Seven phases; a tensor register over 410 linears; a layer-by-layer backward sweep; a snap
algebra; a CPU kernel sandbox; a queue for several drivers on a one-owner machine. Phases 2 to 5 are given 25 to 32
days; I would double that.

**Checked.** A 9,216 row-token forme beside the lens in the server: 3.2 GiB above the model at 0.35 MB a
row-position, so 24.4 GiB with the lens, 27.8 to 29.8 with the desktop, of 31.36 (my arithmetic). Fits. Raising the
forme to 16,384 would not fit beside the lens under a heavy desktop.

**Flaws.**
- "The base rows' read-out must match the first forme's by bits, or the job is marked" contradicts its own
  scheduler, which cuts formes of different sizes from different start layers. Rows of different batch shapes are
  never the same bits (pilot: batch 2 against 1, 16, 32, 64). The canary needs one fixed forme shape or must go.
- The side-effect panel is costed at the forward alone. Every-position KL over 9,216 positions adds about 1.0 s of
  read-out (my arithmetic from 19 ms a row of 175), and unchunked would need 9.7 GB of logits.
- A "snapped" row is a float emulation of the grid edit in row-delta arithmetic. Calling its reading a reading of
  the snapped standing overstates it.
- `embed_row` on the output side is not heard by `write_rows`, which takes its logits outside any module
  (`atlas_ops.py:493-499`); on the input side the term must carry the embedding's scale
  (`modeling_gemma4.py:1443-1444`).
- Adding tools to the registry is not "no other change": `check_scientist.py:538` asserts the tool list.

### validity: 6.5

**Strongest.** Phase 1 is as small as the kernel's (column operations, one plate, a command line, the known
answer), and nothing in the plan strains the card: a 64 x 48 plate is about 1.1 s and 1.1 GiB (my arithmetic from
today's rates), well inside the 5.6 GiB beside the lens. Pilot 5 is placed where it should be: a go or no-go on
audibility before anything past Phase 1. Its second floor, a norm-undone rescale that re-rounds every output of the
edited linear, is the only device in any design that would see the rounding floor measured today.

**Weakest.** Phase 2. Briefs, item generators, fresh draws, registration, two null families (one matched on firing,
which needs a pre-pass over all 21,504 activations), four batteries, cheaper-knob arms, canaries, a reader, a proof
sheet, routes and a job runner, in 8 to 12 days, with every battery bar "[not yet measured]". I would plan a month.
And each brief needs the owner to write a generator, a near-miss generator and prompt arms before a search starts:
on a one-owner project that, not the card, sets the pace.

**Flaws.**
- "A step costs nearly the same from 1 to 64 rows ... so controls ride almost free" is wrong for the teacher-forced
  plates it describes. Fifty-two chance edits cost fifty-two edit-rows of time. Its totals still hold (a proof's
  forwards are about 105 s by today's rates, my arithmetic) because they were worked from prefill and corpus rates.
- "Proofs and trials run whole forwards" now pays up to four times for nothing: restart from the edited layer was
  bit-identical at the same shape today.
- The norm-undone canary is a full-rank change and is not expressible in its own operation list; it needs an
  output-gain operation. Small.
- The measurement vocabulary grows only by code review. That is the lens's price, stated; it is also a build cost
  each time a hypothesis needs a new reading.

### future: 6

**Strongest.** Care. I found no claim about the code that was wrong. It flags that the 0.85 s and 3.5 s figures
are for prompts of a few word-pieces, lists in section xiv what it rests on that is not established, wraps
`write_rows` instead of touching it because kit 4 is live, and makes the hook refuse when the batch it sees is not
the batch it was given programs for (a reader that halves on out-of-memory would misalign the masks). It is the only
design that names the right mechanism for the lens question (free it, at the price of the sightings pages).

**Weakest.** Order and size. Role names, manifest, normal form, three hashes, nine verbs, ledger entry, docket and
return are "fixed now" before the audibility pilot, which today's one-text evidence suggests will change the unit of
an edit. Increment 1 (format, compiler, gate, hashes, apply kernel, LoRA round trip, two check scripts, a
command-line proof) is given 4 to 6 days; I would plan two weeks. Increment 2 puts routes, proofs, nulls, the panel,
a ledger, six tools and a new page into 2 to 3 weeks; the page alone touches four routing places under a strict
type check.

**Flaws.**
- `relower` "compares by bits" at import and in the conformance suite. Lowering a relative op is float arithmetic on
  unpacked weights; the project's own records show CPU and GPU, and even batch size on the CPU, moving numbers. A
  by-bits relower will raise false alarms on another machine unless lowering is pinned to one deterministic path.
- Its rule P2 judges the gap between `term` and `folded` against 0.014. Today's gap (0.00045 a position) is inside
  that, so by the rule `folded` is dropped; yet the gap is as large as a typical single-neuron effect. The yardstick
  is the wrong one. (The driver's pilot 5 has the same yardstick.)
- No agent until increment 2, no `confirmed` result until increment 3.

### driver: 5

**Strongest.** The verdict path: a fresh process that has never run a cell, loading the edit as data (safetensors
and JSON, nothing executable), scoring held items the bench worker cannot mount. A fresh process with one edit in
force is also the natural home for a folded-form confirmation. And pilot 2 (free cells against five typed verbs, on
the existing daemon, one attended afternoon) is cheap and would settle the design's one bet.

**Weakest.** What must exist before anything new is measured: a keeper, a bench worker, a command line, a ledger
and a brief, then an edit library with five term kinds in two forms, then a verdict worker, a sandbox and three
faces. Phase 1 (3 to 4 days) delivers the lab daemon made safer and no new edit machinery. Phases 1 to 3 are given
15 to 21 days for about 3,000 lines across three cooperating processes; I would plan six to eight weeks.

**Flaws.**
- "The worker ... restores every parameter afterwards, and proves the base is untouched with a canary forward." No
  way is given to restore 15.3 GiB of packed words and scales after free code, and a one-prompt canary does not
  prove anything: the stored record of the vault's own example shows an edit in force leaving another question's
  first-step odds at KL exactly 0.0 (`vault_example_edit/probe.json`, 29 October 2026). A leftover edit can pass.
  What would work: a checksum of the packed tensors on the card, empty overlay dicts and hook lists on all 410
  linears, and the forwards and the module-global unpacker as found.
- "Runs each cell in a worker thread" with "a time limit per cell (SIGINT, then kill and reload)". Python delivers
  a signal to the main thread only. A cell in a worker thread cannot be interrupted that way; cancel is kill and
  reload (about 11 s today without the lens). Run the cell on the worker's main thread; the keeper already answers
  status.
- The sandbox (bubblewrap, the card's devices, and Triton's compile cache for the unpacker, which
  `w4a16_fast.py:14` says is a Triton kernel on a GPU) is unproven, and all of Phase 3 stands on it. The design
  says so and has a fallback.
- Its headline premise, "every forward unpacks all 29.3 billion weights whatever the batch. Rows are nearly free",
  is true of decode steps only. Its cost table survives (64 x 64 is 1.43 s by today's rates, inside its 0.5 to
  1.5 s range's upper end).

## Factual errors

| Design | Claim | What is true | Where checked |
|---|---|---|---|
| driver | The worker "loads without the J++ lens" through `runtime_factory.make_runtime(load_jlens=False)` | That call loads J++; `load_jlens` governs the second lens. The server itself calls it and holds J++. The 17.97 GiB figure needs `rt.jpp_lenses.clear()` after the load, as the census does. | `backend/engine_runtime.py:908-933`; `scripts/census_run.py:135-150`; `backend/server.py:122` |
| driver | "every forward unpacks all 29.3 billion weights whatever the batch. Rows are nearly free up to memory" as the fact that shapes the design | True of a decode step with a cache. A whole teacher-forced forward costs about 0.08 s plus about 0.33 ms a row-position (measured 14 November 2026). | pilot note, `pilot_edit_speed.json` |
| validity | "a step costs nearly the same from 1 to 64 rows ... so controls ride almost free", said of a 64 x 48 teacher-forced plate | The same. Controls cost rows, and rows cost time in proportion. | the same |
| kernel | Rows "close to free"; for the partial forward "the limit is memory, not time" | The same. From layer 44, 32 rows x 30 pieces took 0.096 s, in proportion to rows and layers run. | the same |
| future (small) | "Next-piece from layer 45 ... 0.85 s, measured" | The timed forwards started at layer 46 (`forward_from_layer(lm, l + 1, ...)`), 14 layers; an edit at layer 45 runs 15. A 7% matter; the kernel design words it exactly and the speed design's model counts 15. | `backend/atlas_build.py:821` |

I found no claim about the literature that contradicts the research digest's corrections; all five carry the
corrected wordings (the frozen conversation, the best-of-search caveat, the Lazarus qualifiers, the open-weights
editors).

## Which first increment is truly small

1. **kernel**: yes. About 1,500 lines in a process that exists; most of its pilots are already answered.
2. **validity Phase 1**: yes. Column operations and one plate; narrower than the kernel's.
3. **speed Phase 1**: yes, a little larger (stream cache at all boundaries, plan JSON, packing rows of several
   prompts).
4. **future increment 1**: medium. The same kernel plus a format, a compiler and a round trip.
5. **driver Phase 1**: not small, and it buys no new measurement.

Who underestimates most: driver (process supervision, sandbox), future (formats and a page), speed (everything after
Phase 2), validity (Phase 2). A fair rule for the synthesis: believe the first increment's estimate, double the rest.

## What a synthesis should take from each

- **kernel**: the edit as a hashed value with three kinds; one text a forward with no padding for the first
  increment; the folded cross-check that snapshots and restores all 410 linears; refusing to read while any overlay
  dict is non-empty; `delivered`; tensor classes gated by a verify section; the book written by the reading; the
  known answer with its tolerance fixed first and the "is it 5313 alone?" question in the same call.
- **speed**: the cost model, refitted to today's constants and checked on every job (predicted against actual
  seconds); the `fell_back` check before and after each job; an estimate call; parting read teacher-forced along the
  stored answer, with near ties flagged; a 0.01 dose as a rounding-floor arm.
- **validity**: the norm-undone rescale as the delta path's floor; the audibility pilot as the go or no-go; nulls at
  another site matched on firing; the habitat battery from stored passages; making a set delta as "new entries
  rounded to bf16, minus base"; planted-failure checks of the reader; pilot 0 (re-run `verify_scientist_cli.py`,
  no card).
- **future**: the hook that refuses a batch whose rows do not match its programs; source and lowered payload as two
  things with a `source_id`, added once the unit of an edit is known; the "rests on what is not established" table;
  replay of logged calls with no model as the regression test; wrapping `write_rows`, never editing it.
- **driver**: the verdict in a fresh process with the edit loaded as data; a machine-wide card lock taken by the
  server and the census wrappers; the footer that warns of an unpacker fall-back; out-of-memory halving that keeps
  a base row; per-row doses so a dose sweep costs rows; `SCAFFOLD.md` with a `--bare` switch; the typed-or-free
  driver pilot on the existing daemon.

## What all five missed

1. **Today's pilots.** Each design plans to measure what is already measured (bit-safety and cost of the per-row
   delta, restart exactness, overlay cost). The first card hour should go to what is still open in that note:
   generation under per-row deltas, anything with the lens loaded, low layers, a backward under the compiled
   unpacker, more than one text.
2. **The floor inside one forward is not nought.** Every design leans on "rows of one forward differ by the edit
   alone" and tests it with a zero edit. A zero edit adds exactly nothing and cannot see that a real delta is
   rounded twice. The measured gap between the two forms (0.00045 nats a position) is the working floor, and it is
   the size of a typical single-neuron effect. Only validity's rescale canary would show it. A claim needs an effect
   several times that floor, or a folded-form confirmation.
3. **Reading cost and metric choice.** A logit difference or a group log-odds needs a few rows of the unembedding
   and is nearly free; KL and a log-probability need all 262,144 logits in fp32 at every scored position, which is
   more than half the time of a restarted reading. `logits_fp32` allocates rows x 262,144 in one piece
   (`causal_check.py:133-139`), so an every-position reader must chunk by rows. No design separates the two kinds
   of metric in its budget.
4. **Attention is not the feed-forward block.** q, k and v each pass a per-head norm
   (`modeling_gemma4.py:1218-1239`), so a positive uniform scale of a head's q, k or v block is undone, like the
   block scales the designs do refuse. On the ten full-attention layers the values are the keys (line 1232), so a
   `k_proj` edit there is a value edit too. No gate, null generator or export note knows either. The designs say
   only that `v_proj` is absent.
5. **Which stand-in.** The default stand-in is dense float32. A by-bits check of "float32 sum, rounded once to
   bf16" on it tests no rounding and no packed forward. The conformance checks need the packed bf16 stand-in
   (`check_control_vector.packed_stand_in`, the fixtures of `check_gguf.py`) as well as the card.
6. **The lens.** Freeing J++ is one line (`rt.jpp_lenses.clear()`), worth about 3.2 to 3.4 GiB, and possible only
   in a process that serves no sightings page. One design names the wrong mechanism, one assumes a daemon switch
   that does not exist, one calls it not established.
7. **Existing checks will fail on arrival.** The scientist's tool list and the daemon's route list are asserted
   (`check_scientist.py:538`, `check_local_only.py:369`). Budget the check updates with the tools.
8. **Building inside a long-lived process.** A harness module imported by the server or the daemon is cached;
   every code change is a restart (about 11 to 20 s) or an explicit reload. None of the plans counts it.

## Advice to the synthesiser

Build the kernel design's three modules first, in the lab daemon or a command-line process, with the compiled
unpacker and the lens freed by `rt.jpp_lenses.clear()`. Use the speed design's cost model as the estimator, with
today's constants. Put sets of columns and a direction removed from its writers into the first increment, because
one neuron alone looks inaudible. Keep the folded confirmation for column and row terms from the first day, add
validity's rescale floor, and state the floor on every result. Read logit differences by default and chunk
every-position KL by rows. Move into the server (the pattern validity, speed and future share) only when a
tools-only driver is wanted; build the driver design's sandboxed worker only when the owner approves an unattended
driver that runs code, and take its fresh-process verdict then. Do not freeze formats before the audibility pilot
has been run on more than one text. Write the attention facts into the gate before any attention edit is allowed.
