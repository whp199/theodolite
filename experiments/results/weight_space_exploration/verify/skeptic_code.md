# Skeptic, lens: the code. Findings on "Levelling"

Written 14 November 2026. Read-only: nothing in the repository was changed, no server started, the model was not
loaded and the card was not touched. Two small CPU reads were made beyond grep: the sixty post-feed-forward norm
weights straight from the checkpoint file's header and bytes (60 x 5,376 bf16 values), and the stats index's
`mlp_rms_q` (numpy).

Marks: **[read]** opened by me today at the path and line given; **[measured, pilot N]** from the raw JSON beside
`pilot/RESULTS.md`, one run each; **[arithmetic]** my own sum.

One thing to know first. The proposal was written at 14:43 local with two pilots in hand. Five more pilots
(3 to 7) finished between 14:50 and 15:02 and `RESULTS.md` now carries a second half headed "READ THIS: it changes
what a reading may be trusted for". Findings 1 and 2 rest on those. The proposal could not have known them; they
still decide what phase 1 builds.


## Findings, most serious first

### 1. The per-row mechanism is an output-side term for every kind of edit. REFUTED for column edits. Serious.

**The claim.** Section o: "A weight edit can ride one row of a batch as an added term on a linear's output".
Section iv: one hook, `y[row] = (y[row].float() + dose[row] · d).to(y.dtype)`, for `cols`, `rows` and `lowrank`
alike; this "delta" form is "every practice and fresh reading".

**What I saw.**
- [measured, pilot 6, `pilot6_which_is_true.json`] Seven single-neuron silencings at layers 5 and 20, each as a
  folded column overlay, as an input-side per-row zero of the activation, and as an output-side per-row delta, at
  batches of 2 and 32. Input-side equals folded to every printed digit in all 7 at both sizes (L20 n4294: 0.00908
  and 0.00908; 0.72581 and 0.72581). Output-side does not: 0.02274 where both others read 0.72581 (L20 n4294);
  0.70255 against 0.52337 (L20 n5316); 0.20649 against 0.00829 (L5 n16222).
- [read] The input-side per-row form already exists in the repository: `atlas_ops.installed` -> `push_act`,
  `backend/atlas_ops.py:369-383` (`x[:, :, n] = x[:, :, n] * ff`, one factor a row), reached by `Push(mode="mul")`.
  The proposal lists it only as a "cheaper knob" arm and as an unproven twin ("Not assumed; pilot B reads the
  three forms", section xiv).
- [read] Both pilots' output-side hooks add in bf16 (`out + a * delta.to(out.dtype)`, `pilot2_equivalence.py:124`,
  `pilot4_cancellation.py`). The proposal's own formula (sum in float32, round once) has been measured nowhere.
  It still starts from a `y` the base product has already rounded to bf16, so it cannot recover what the folded
  product would have rounded; whether it closes the gap at layers 5 and 20 is not established.

**What follows.** Two of the four first-increment builders (`silence`, `scale`) should ride the row on the input
side, which is the existing `mul` arithmetic with the start mark not spared, vectorised (the existing hook loops
in Python a neuron at a time and `check_pushes` caps a request at 32, `atlas_ops.py:50, 233`). `set` and
`remove_direction` are an input-side zero plus an output-side add of the new content. A pure output-side delta is
left for low-rank terms, which have no folded path at all. The "three forms" table, the phrase "the gap between the
two forms of one edit is the working floor", pilot E and the folded twin's job all need rewriting: for column
edits there is no gap between forms at one batch size.

**Smallest fix.** In section iv, replace the one hook by a form per kind (input-side scale for columns; set-to-value
for a silenced row's output element; output-side add only for new content and low rank), and say that the
proposal's float32 output-side formula is unmeasured.


### 2. "The floor is about 0.00045 nats whatever the edit's size", and the folded form settles a claim. WEAKENED. Serious.

**The claim.** Section i's table and its "one reading that matters"; the bar "five times the floor row"; the
trial of record "folded form" as the form that may be claimed; a practice reading on a one-row capture replayed at
many rows.

**What I saw.**
- [measured, pilot 6 and 7] The same folded edit reads 0.0091 at a batch of 2 and 0.7258 at a batch of 32 (L20
  n4294, raw recipe text). Across ten batch sizes there are two arithmetics, one up to 12 rows and one from 16,
  identical figures inside each (`pilot7_batch_size.json`, `one_silencing_across_batch_sizes_recipe`). So "folded"
  does not name one reading; a trial of record must name its batch shape too.
- [measured, pilot 5] Null rows (the block's output scaled by 1 + e) read 0.00014 to 0.0088 depending on text and
  layer. No constant floor serves; RESULTS.md says so in words (lines 155-161).
- [measured, pilot 3] A one-row capture replayed at 32 rows agreed with same-shape forwards at layer 44
  (correlation 0.9998, 0.9999) and not at layers 5 and 20 on the raw recipe (0.02 and 0.32): a one-row capture and
  a 32-row forward sit on either side of the two arithmetics.
- [measured, pilot 7] The instability was on raw text, which the instruct model does not model (mean log-probability
  of the recipe's own next piece -9.6 nats). The same passages through the chat template behaved (no reading
  differing over five-fold). RESULTS.md, lines 146-148: "The first two pilots used raw text; their timings stand,
  their KL figures are of an off-distribution regime."
- [read] The proposal leans on raw text in four places. (a) The E6 known answer: `"{N} went to the store because"`
  with a start mark and no chat template (`experiments/scripts/E6_tool_audit_04_vault.py:65-71`,
  `experiments/lab/labkit.py:79-83`). (b) The neutral-text battery, "a fresh draw of kept-back corpus windows".
  (c) The habitat battery, "the stored passages where the edited neurons fire hardest". (d) Value item 2 of
  section ii, a causal reading "on its stored passages" for the layers the census cannot reach, which are layers
  1 to 32 and 54 to 60. The corpus behind (b) to (d) is half raw web text: the stats index's meta names
  "HuggingFaceFW/fineweb-edu:sample-10BT + HuggingFaceH4/ultrachat_200k::train_sft", and only the second is put
  through the chat template (`scripts/atlas_build.py:119-144`).

**What would settle it.** Pilot 7's table repeated on chat-formatted items at layers 5, 20, 36 and 44 with sets of
columns, and on fineweb passages from the index, each at a batch of 2, 12, 16 and 32.

**Smallest fix.** Adopt RESULTS.md's four rules as design (lines 145-153): gate each item on the base row's own
log-probability; take any reading that earns a word in both arithmetics; print the median over positions and the
count of positions over a nat; column edits input-side. Drop the constant floor from section i. Restrict the
one-row capture to layers where it has been shown (44 so far). State the batch shape of a trial of record. Mark
value item 2 and the two corpus batteries "not established on raw text".


### 3. `remove_direction` through `w_post` waits on pilot 0. WEAKENED: pilot 0's condition is already met. Serious.

**The claim.** Section iv: the op changes each column "so that its write as the stream receives it loses `dose`
times its component along `u`"; "Whether small entries of `w_post` make the raw change large is not established;
pilot 0 reads the sixty weights." Claim 10 of section xx. Pilot D (registered) runs direction removals at layers
20, 36, 45 and 55.

**What I saw.**
- [read, checkpoint file, CPU] I read the sixty weights. From layer 13 to layer 58 every layer has entries under 1%
  of its median: 58 at layer 20, 317 at layer 36, 260 at layer 44, 246 at layer 45, 196 at layer 55, of 5,376. The
  smallest is 1/815 of the median at layer 20, 1/1,328 at 36, 1/1,660 at 44, 1/1,625 at 45. At layers 37 to 58 the
  median equals the largest value: nearly every dimension sits at one weight and a few hundred sit a thousand times
  lower.
- [read] Those are the instrument's own quiet carry channel. Of the 181 persistently quiet dimensions
  (`V6_quiet_dims_pers181.npy`), 165 are among layer 36's small entries, 180 among layer 44's and 45's, 175 among
  layer 55's. `README.md:334-337` records the channel ("read, through RMSNorm, with up to 1700x the gain of the
  others ... that is why lens-derived steering vectors break the model"), and the runtime already carries masks for
  it (`backend/engine_runtime.py:1234-1277`: `quiet_dims_mask`, `ensure_quiet_masks`, `layer_quiet_mask`). The
  proposal does not mention the channel or the masks anywhere.
- [arithmetic] The raw column change is the stream-side change divided by `w_post`, entry by entry. For a direction
  with even weight across dimensions, that division lengthens the raw change 72-fold at layer 20, 127-fold at 36,
  191-fold at 45 and 181-fold at 55, with 99% or more of it in the small entries. A real contrast direction is not
  even-weighted, so the true factor is not established; the README says gradient-derived directions put 99 to 100%
  of their energy there at layers 36 to 59.
- [read] The same-site chance edit is "a chance direction ... drawn as the census draws its 'columns' controls".
  Those are mixes of write directions, `unit(g @ writes)` with `writes = w_post ⊙ W_down` (`backend/census.py:1567-1591`),
  so dividing them by `w_post` gives back ordinary raw columns with no such growth. A real direction and its null
  would then differ by construction in how much raw change lands in the quiet channel. That is claim 8 failing
  before it is run.

**Smallest fix.** Write the masked form now, not after pilot 0: project `u` off the layer's quiet mask (the
existing one) before the division, apply the same projection to every chance direction, refuse a `u` whose share
in the mask exceeds a stated bound, and print the raw change's length beside the stream-side one. Fix this before
pilot D is registered, since D's runner freezes the op.


### 4. "Phase 0 is 3 to 4 days", "phase 1 is 6 to 8 days and about 1,500 to 2,000 lines", checks "about 600 lines". WEAKENED. Serious.

**What I saw.**
- [arithmetic] The components table already sums past the claim: 500 + 700 + 150 + 200 + 600 = 2,150 lines against
  claim 20's 1,500 to 2,000.
- [read, counted] The house's own ratio of check lines to module lines, over twelve pairings: 0.56, 0.57, 0.62,
  0.70, 0.83, 1.34, 1.45, 1.64, 2.62, 2.87, 3.79, 3.82; median 1.45. For modules under 800 lines it runs 1.3 to
  3.8. The proposal plans 600 lines of checks for 1,550 of module and command line: 0.39, under every existing
  pairing. At the median that is about 2,250 lines of checks, and phase 1 about 3,800 lines in all.
- [read] The nearest existing thing is `backend/causal_check.py`: one neuron, one position, activation pushes, a base
  row in the same forward, floor rows at ±0.01, halving on out-of-memory with a base row in every forward. It is
  490 lines with 1,284 lines of checks (`scripts/check_causal_check.py` 949, `scripts/verify_causal_check.py` 335).
  `levelling.py` is to do more than that (three edit kinds, two chance families matched on ceiling and firing,
  capture and restart, two read-out classes, a folded twin, a cost model) in 700. The proposal names only
  `logits_fp32` from that module; it does not say it is building a second paired-reading kernel beside the first,
  with a second floor convention.
- [read] Phase 0 holds pilot D. As specified (section x) D needs column scaling, sets of 64 "top writers along a
  difference-of-means direction", direction removal at three doses, both chance families at 39 matched on firing,
  both floor rows, `delivered`, neutral-text KL, and "64 generator items". Those are phase 1's `direction`,
  `writers`, `remove_direction` and `levelling.py`, and a generator from phase 3. D is registered, so its runner is
  frozen: either the kernel is built first (then the owner's stop decision costs phases 0 and 1, not "two hours of
  the card and two weeks of work"), or D is a second implementation of the kernel's hardest parts. Today's seven
  pilot scripts are 1,779 lines and contain no chance edit, no matching and no direction removal.

**Smallest fix.** Split D: a D-small in phase 0 that needs no direction and no generator (single columns and random
sets of 64 at scales 0 and 2, input-side, chat-formatted items, both arithmetics, other-site controls matched on
firing only), and the full D run on the phase-1 kernel at a tagged commit. Plan checks at the house ratio. Say that
the first decision comes after phase 1.


### 5. Planted faults, the unnamed known answer and "aggregates only" under a driver that reads the disk. WEAKENED. Serious for phase 4 and pilot H.

**The claim.** Section vi applies "Codex reads files elsewhere" to held item sets only, and says fresh draws hold
"against any reader". It also offers planted faults as "ground truth for discovery", a known answer given
"unnamed" (pilot H), and "the driver gets aggregates only".

**What I saw.**
- [read] The Codex finding itself stands: `backend/scientist_chat.py:103` and `:843` (`workspace-write` at the
  default reach), `README.md:692-695`.
- [read] The known answer is named in the repository: `experiments/results/E6_tool_audit_REPORT.md:81` and `:137`
  ("L50 n5313 ... The 'he' list at L50 gives an identical effect because it shares the bipolar neuron 5313"),
  `experiments/REPORT.md:349`, and the JSON itself. Pilot H's driver is the owner's attended Claude Code session,
  which reads the whole repository; a Codex chat reads it too.
- [read] A planted edit is an edit, and edits are saved as `atlas_data/edits/<id>/manifest.json`. The scientist's
  work folder is `atlas_data/scientist/work` (`scientist_chat.py:143-155`), so the edits folder and the level book
  are siblings a Codex shell can read. A manifest names the layer and the neurons.
- [read] "Replay of a stored reading without the card" needs per-item rows on disk. If they are in the level book,
  a Codex driver reads unrounded per-item results, and the rounding "to units of its standard error" guards
  nothing. The proposal does not say where per-item rows of a fresh reading are kept.
- [read] At `THEODOLITE_SCIENTIST_REACH=home` Claude Code gets a shell and file tools over the home folder and
  Codex gets `danger-full-access` (`scientist_chat.py:99-107, 567, 843`). Then the book, the edits, the briefs and
  the harness's own code are writable by the driver. "The level book lives ... where no driver's sandbox can write"
  is false at that reach, and the proposal only records the reach on a card.
- [read] "The read tools then serve the planted state" contradicts the thesis two pages earlier ("An edit is an
  argument of a measurement, never a state of the server"). Ten of the scientist's existing tools run the live
  model through routes with no edit argument (`/steer`, `/reads`, `/logprobs`, `/compare`, `/fires`, `/attention`,
  `/attn_drive`, the written and blind tests, `/api/scientist/causal_check`; `backend/scientist_tools.py`). Either
  each is threaded with a standing edit (unsized, and it is server state), or they go on serving the base and a
  driver finds the fault by setting an old tool beside a new one. There is also no edit-aware view of the weights
  for a `weights` tool: the runtime's accessors honour only column overlays on `down_proj` and row overlays on
  `gate_proj`; `slice_w_up` and `get_full_w_up` ignore overlays (`backend/engine_runtime.py:1069-1077, 1111-1119`),
  and that file is one R4 pins.

**Smallest fix.** Keep planted edits and per-item rows out of any file a driver's shell can read (memory only, or
the separate account of section xii moved forward to phase 4). Make every levelling tool refuse at reach `home`.
For the known answer as a driver test, use a planted fault instead, since the real one is published in the
repository. Size "the read tools serve the planted state" or drop planted faults from phase 4's promises.


### 6. Pilot B's stop rule: "the delta arm does not reproduce -1.80 (±0.1): stop". WEAKENED. Minor to serious (it is registered).

**What I saw.** The stored figures stand [read]: 'she' at layer 50 -1.7989, 'he' -1.7981, one random ten -0.0199
on the 12 female contexts; the lists share 549, 5313 and 8801. But the reading that produced them is not the
kernel's:
- the edit was the vault's DELETE, a folded zero column (`E6_tool_audit_04_vault.py:172-181`);
- the 12 contexts ran as one right-padded batch of 12 (`experiments/lab/labkit.py:394-405`), which on today's
  evidence is the small-batch arithmetic, while the proposal reads "38 rows" an item, the large one;
- the logits were the model's own bf16 logits cast to float, not fp32 (`labkit.py:403-405`); the stored
  per-context differences show it, many sitting on multiples of 0.125 (-0.3750, -0.2500, 0.1249, 0.4999), the bf16
  step for logits between 16 and 32. The proposal elsewhere calls that read-out "too coarse";
- the contexts are raw text of about 8 pieces, not "about 30";
- `backend/engine_runtime.py` has changed since (two commits of 12 November 2026).

With four differences at once, a miss says nothing about the kernel. **Fix:** the registered pilot first re-runs
E6's own reading (vault DELETE, batch of 12, the model's logits) on today's code, then adds one difference at a
time; the tolerance is set from the bf16 step, and the stop rule attaches only to the like-for-like arm.


### 7. "One forward a lock hold" in the server, and "nothing an agent does changes the model the server holds". WEAKENED. Minor.

- [read] The house holds the GPU lock across a whole reading so that no vault edit lands inside it
  (`README.md:599-600`; the push route, `backend/atlas.py:2135, 2169-2190`). A levelling of many forwards that
  releases the lock between them lets a vault edit land between two items. The guard "refuses to read while any
  overlay dict ... is non-empty" is stated once, not per lock hold. **Fix:** check the dicts inside every hold and
  void the reading if they change.
- [read] There is a second agent surface, which the proposal's "the scientist's one registry" passes over:
  `backend/mcp_scientist.py` serves five FastMCP tools, also over HTTP and the WebSocket, and one of them,
  `query_larql_vault` (line 1569), executes UPDATE, INSERT and DELETE. An agent on that surface changes the held
  model today. The safety line is true of the new tools only and should say so.


### 8. The times-3 floor row "which the following norm undoes". STANDS (for the feed-forward block).

I tried to break it through the norm's epsilon. [read, stats index] The block's output rms before the norm has a
median of 0.079 to 1.34 across the sixty layers (`mlp_rms_q`), so with eps 1e-6 a threefold scale changes the
normed output by at most 7e-5 of itself (layers 27 to 38), under bf16's rounding. `up_proj` times 3 is undone too
(the product feeds `down_proj`); only `gate_proj` is not, as the proposal says. [measured, pilot 3 and 5] At layer
44 the times-3 row read 0.0004 to 0.0006, the size of the output-side gap there. The row is sound as a null; what
finding 2 changes is that its size varies twenty-fold with text and layer.


### 9. Pilot A's question (writing under per-row hooks round `write_rows`). STANDS, now measured.

[measured, pilot 3] 8 rows, 48 pieces, two questions: untouched rows wrote the same pieces with the same
log-probabilities; a zero row equalled a base row; the hooked write's base row equalled an unhooked write; 0.095 s
a step hooked and unhooked. [read] `write_rows` keeps every row in the batch to the end (`atlas_ops.py:590-617`),
so a hook that refuses on a changed row count never trips. "Wrapped and never edited" is buildable.


### 10. The path:line citations. STAND.

I opened each of these and found what the proposal says is there: `int4_cpu.py:138-164`; `atlas_ops.py:186-201,
344-412, 493-499, 1017-1043`; `larql_vault.py:481, 500, 504-524, 556-567`; `engine_runtime.py:352-377, 429-436,
907-958`; `causal_check.py:132-139`; `census.py:1567-1591` and the five helpers named; `census_run.py:135-150`;
`w4a16_fast.py:205-229`; `labd.py:150-235`; `server.py:122, 1660`; `atlas.py:148, 273-294`;
`atlas_index.py:125-130, 318-328`; `scientist_chat.py:101-104, 843`; `README.md:692-695`;
`check_scientist.py:484-487, 538`; `check_local_only.py:369`; `check_dequant.py:150`; `check_control_vector.py:97-101`;
`check_gguf.py:825-874`; `census_intake.py:332`; `export_lora.py:50-55, 501`; `modeling_gemma4.py:1074-1076,
1216-1238, 1443-1444`; `VOLUNTEERS.md:51, 338`; `ROADMAP.md:133-134`; the E6 JSON; `vault_example_edit/probe.json`;
35 tools by loading the module; the R4 pin (f1a0b4c3... against the pinned 69eed1b4..., changed 12 November 2026).
The proposed file names and the words levelling, level book and chance edit are free by grep. New server routes are
swept by `check_local_only.py:188-205` without a pinned list.

Three small corrections. The "about seven times" is at `w4a16_fast.py:410-415`, not 205-229. The tool count of 30
is also in `ARCHITECTURE.md:43`. And repair 1 is narrower than worded: a neuron's face, wiring and steer words do
carry stale flags after a vault edit (`atlas.py:268-269, 1274, 1402, 1604-1607, 2106-2108`); the stats index and
the census replies do not.


### 11. Smaller things that are less than they are called. WEAKENED. Minor.

- "Cold load without the lens 9.5 s" [measured]. By the code the load includes the J++ lens, which the pilot then
  cleared (`engine_runtime.py:868`; `pilot_edit_speed.py:86-90`). It is the cost a command-line call pays, so the
  11 s stands, but it is a load with the lens, on a warm page cache.
- `delivered` is used four times and never defined in the proposal; its only definition is `kernel.md:249`.
  Nothing of that name exists in the repository.
- "Edits as proposals" on `field_notes.propose / decide`: those take `(kind, layer, index, name, note, chat)` and
  validate a name or a note for one neuron or head (`backend/field_notes.py:496-579`). An edit or a brief needs a
  new store; only the pattern is reused.
- The three-voice `Thread` is a private component of the atlas's push tab (`frontend/src/screens/atlas/PushTab.tsx:226`),
  bound to that tab's state; the vault page is another screen.
- "Tokens and dollars on every reply": only the Claude Code pump passes on a cost, once a turn
  (`scientist_chat.py:657`); nothing reads usage for Codex or an endpoint, and a tool runs in a separate MCP
  process that knows no cost.
- The lock file of section xi has a precedent it does not name: the pod scripts already take `logs/gpu.lock` with
  `flock` (`scripts/check_pod_scripts.py:331-348`). The census wrappers that probe port 8765 sit in registered
  folders under SHA256SUMS and cannot be edited in place.


## What the proposal does not say at all

1. That the faithful per-row form of a column edit already exists as the `mul` push, and that the new hook as
   specified is the less faithful of the two.
2. The quiet carry channel and the runtime's quiet masks, though `remove_direction` divides by the very weights
   that define the channel.
3. `causal_check.run_check` as an existing paired kernel with its own floor rows and halving; which of the two a
   later reader should trust when they differ.
4. Which batch shapes a reading and a trial of record use, now that there are two arithmetics on this card.
5. Whether items are chat-formatted. The known answer, the corpus batteries and the habitat are raw text in part or
   whole.
6. What the level book holds (per-item rows or not) and who can read it.
7. What happens at reach `home`.
8. The second agent surface (`mcp_scientist.py`) and its vault-editing tool.
9. That pilot D needs most of phase 1 and a generator from phase 3.


## Overall

The proposal reads the repository accurately: every citation I opened holds, and the three corrections it makes to
the judges are right. It does not survive as written on four points, none of them fatal, each needing a change
before phase 0 is registered. First, the central hook must change: for column edits the per-row form that equals
the folded weight is input-side, and the repository already has it; the output-side delta is for new content and
low rank. Second, the floor is not a constant and "folded" is not one reading: a claimed reading has to be taken
on chat-formatted items the base models, in both batch arithmetics, and the proposal's raw-text pieces (the E6
contexts, the corpus batteries, the habitat, the atlas product for uncensused layers) are unproven. Third,
`remove_direction` needs its quiet-channel form written now, with the same mask on real and chance directions,
because the weights it divides by are a thousand times smaller on a few hundred dimensions at every layer pilot D
names. Fourth, the sizes are low by the house's own ratios and pilot D cannot sit in phase 0, so the owner's first
decision costs more than the proposal says. The seals, planted faults and the known answer also need a disk a
driver cannot read before phase 4 can prove what it promises.
