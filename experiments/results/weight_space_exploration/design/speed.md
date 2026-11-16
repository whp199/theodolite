# The Proof Press: a weight-space bench for Theodolite, designed speed first

Design only, written 14 November 2026. Nothing here was run on the card, nothing in the repository was changed.
Layers are counted from 0 throughout (the pages show layer + 1).

Every figure carries one of three marks:

- **[measured, date]** a number recorded in the repository, taken from the two digests or read from the record itself;
- **[arithmetic]** my own sum from measured numbers, shown so it can be checked;
- **[to pilot]** not known; a named pilot in section 9 settles it.

The lens of this design is hypotheses per hour on one RTX 5090. Where that conflicts with another priority I choose speed
and say what it costs.


## 0. The short version

1. **The model is never edited.** An edit exists only as a small difference term carried by some rows of a batch. The
   packed 4-bit model stays as shipped in the server's memory; the unedited model, a candidate edit, its dose ladder and
   its random controls are rows of one forward. Nothing is applied, so nothing is rolled back, and the instrument's pages
   keep reading the shipped model while the press works.
2. **A forward starts at the lowest edited layer.** The stream entering every layer is cached once for each prompt set.
   A candidate at layer 45 costs a quarter of a forward; one at layer 55 a twelfth.
3. **Teacher-forced reading is the default and generation the exception.** Even "where do the two answers part" is read
   without generating, by running the edited rows along the unedited model's own stored answer.
4. **One backward pass ranks every neuron and any list of candidate edits before a single candidate forward is spent.**
   The cached streams make that backward cost one layer of memory at a time.
5. **The card is not the bottleneck; the driver model is.** By my arithmetic a fully controlled hypothesis costs 6 to 27
   seconds of the card and 1 to 5 minutes of the driver's turns. So the interface is built to carry families of candidates
   in one call, to run the controls without being asked, and to serve several driver sessions from one queue.
6. **Partial dense caches of the base weights are not worth building on this card.** With the compiled unpacker the
   unpacking floor is about 1.5 ms a layer against about 66 ms a layer of products in a full batch [arithmetic]. The
   headroom buys more by holding rows. Pilot 4 can overturn this.

What it costs: a fast reading is a reading of a float difference term laid on the output of a linear, which is a third
numeric class beside "folded into the bf16 weight" and "written onto the 4-bit grid". Every result says which it is, and
nothing leaves the machine before it is measured again in the form that ships.


## 1. What the environment is

### 1.1 Which process holds the model

The instrument's server (backend/server.py) holds the one copy of the 31B, as it does today. The press is a new router
inside it, bound the way the atlas router is bound (it receives the runtime and `_GPU_LOCK`; the pattern is `on_gpu` at
backend/atlas.py:148-153). I choose this over the lab daemon for three reasons.

- Because the press never mutates the model, there is no model state to isolate, so there is no reason to take the card
  away from the pages. Only one process can hold the model on this card [README.md:485].
- The atlas indexes, the census readers and the scientist's three driver backends are already reachable there.
- The server defaults to the compiled unpacker; the lab daemon defaults to the reference one, about seven times slower a
  step (0.71 s against 0.09 to 0.13 s) [measured, 27-28 October 2026; experiments/lab/labd.py:157].

The server process runs **typed plans only**. It never executes code written by a driver model. Free code runs in a
separate CPU-only kernel process (1.3).

### 1.2 Components

| Component | What it is | Builds on | New code |
|---|---|---|---|
| Row-edit kernel | Edits as per-row terms on any of the 410 packed linears, plus the embedding, the norms and the layer scalars, installed as hooks for one `with` block. | `atlas_ops.Interventions` and `installed` (backend/atlas_ops.py:267-412): per-row hooks, removed in a `finally`. The difference-term arithmetic of `int4_cpu.with_overlay_terms` (backend/int4_cpu.py:138-164). The bit comparators `w4a16_fast.same_bits`, `same_layout`. | `backend/press_deltas.py` |
| Span runner | Runs layers L to M on a cached stream with captures and fp32 read-outs; packs rows into a batch (a "forme"); writes text when asked; runs the layer-by-layer backward. | `atlas_ops.forward_from_layer` (backend/atlas_ops.py:1017-1043), which starts at a layer but cannot stop at one. `causal_check.logits_fp32` (backend/causal_check.py:133). `atlas_ops.write_rows` (backend/atlas_ops.py:503-617). labkit's recompute-in-backward linear `_W4A16LinearFn` (experiments/lab/labkit.py:417-433), copied, not edited. | `backend/press_run.py` |
| Errata | The edit program: typed operations, canonical bytes, hash, algebra (scale, restrict, matched null, snap to the grid), compile to one normal form per linear, export. | `export_lora.Plan`, `plan_of`, `write_lora`, `read_lora` (scripts/export_lora.py:169-297, 501-552). `export_control_vector.add_record`, `code_commit`. The four-part edit check (scripts/check_gguf.py:825-874). | `backend/press_errata.py` |
| Suites and caches | Prompt sets in three tiers, the stream cache, the stored unedited answers. | Seed data already on disk: the study's 36 texts, the 1,218 study contexts, the ten census questions with stored unpushed answers, 3,914 corpus windows with the odd-window kept-back fold, the pattern pilot's 63 texts and 63 near misses. | `backend/press_suites.py` |
| Ledger and budgets | An append-only, hash-chained record written by the server for every plan it runs; seconds of the card spent and left; registrations. | `census.file_sha256`, `write_json_atomic`, `runtime_record` (backend/census.py:1150, 1801-1846). The registration folder pattern (experiments/results/census_match_walk/prereg/). `field_notes.propose` and `decide` for kept edits. | `backend/press_ledger.py` |
| Queue and routes | `/api/press/*`: estimate, submit, status, result, cancel. Jobs run a forme at a time under `_GPU_LOCK` in a worker thread and give the lock up between formes. | `_GPU_LOCK` with `asyncio.to_thread` (backend/server.py:99). The lab daemon's server-side result file (experiments/lab/labd.py:228-233). The sweep's cooperative cancel. | `backend/press.py` |
| Tensor register | Precomputed statistics for all 410 linears and the embedding; live direction scans. | `atlas_ops.Weights`, `dense_weight`, `pair_products` (backend/atlas_ops.py:57, 128-209, 1616). `AtlasIndex` and `save_npz` for the on-disk form and its refusals. | `backend/press_register.py`, `scripts/press_build.py` |
| Tools | Six tools added to the scientist's registry, so Claude Code, Codex and any OpenAI-compatible endpoint get them at once. | `TOOLS` and `call` (backend/scientist_tools.py:191, 534); `from_page` and `to_page`; `backend/scientist_mcp.py`. | a tool group in `backend/scientist_tools.py`; `backend/press_client.py` (standard library only) |
| Kernel | A CPU-only Python process per session for free analysis and for writing plans in loops. | The lab daemon's exec loop and token scheme (experiments/lab/labd.py:98-115, 198-215) as a pattern; Codex's no-network sandbox as a precedent. | `scripts/press_kernel.py` |
| Checks | CPU checks on the six-layer stand-in; card checks. | The check-script idiom (scripts/check_census.py:108-123), `scripts/atlas_stand_in.py`. | `scripts/check_press.py`, `scripts/verify_press.py` |

No pinned file is edited. `backend/engine_runtime.py`, `theodolite_math.py`, `label_gate.py` and `backend/__init__.py`
are left alone; the press binds from outside as `w4a16_fast.py` does. Nothing that writes census answers is touched, so
kit 4 stays kit 4.

### 1.3 How it stands with the things that exist

- **The server.** The press is a router in it. A job holds `_GPU_LOCK` for one forme (at most about 4 s by the cost
  model in 4.1) or one writing batch (5 to 7 s) and releases it, so a page waits seconds, not minutes. Status and
  estimate routes never take the lock.
- **The vault.** Untouched. The press never writes `_memit_col_overlays` or `_memit_row_overlays`. It refuses to run
  while vault edits are in force, because its caches describe the shipped weights. Carrying a kept program into the
  vault as a global edit is a later, owner-only step.
- **The lab daemon.** Not used by the press. It stays the owner's open bench for work outside the plan vocabulary, with
  the model in-process and no containment. One change is recommended whatever else is decided: make it default to the
  compiled unpacker.
- **The scientist.** Gains six tools through the existing registry. The same tools serve an outside agent over the
  standard-library MCP server with `THEODOLITE_URL`.
- **The atlas and census.** Read-only inputs. They describe the shipped model only; the press never reads them as facts
  about an edited one (section 5).


## 2. What an edit is

### 2.1 The representation

An **erratum** is one typed operation on one named tensor. An **errata sheet** (a program) is an ordered list of them.
Every additive operation is stored as a **difference from the shipped tensor**, never as an absolute value, so a program
means the same thing on any quantisation of the same model.

| Operation | Tensors | Stored as | Applied to a row as |
|---|---|---|---|
| `gain` | a neuron (its `down_proj` column, equally its `up_proj` row); a head (its block of `o_proj`'s input) | indices, factors | scaling the linear's input for that row. This is the existing `mul` push and `HeadScale` at every position (backend/atlas_ops.py:366-408). Factor 0 is the vault's DELETE. |
| `col_add` | any linear; meant for `down_proj` (what a neuron writes) | k indices, D [k, out] | y += x[..., idx] @ D |
| `row_add` | any linear; meant for `gate_proj`, `up_proj` (what a neuron reads) | k indices, D [k, in] | y[..., idx] += x @ D^T |
| `low_rank` | any of the 410 linears | A [r, in], B [out, r] | y += (x @ A^T) @ B^T |
| `sparse` | any linear | n triples (out, in, value) | gather, multiply, scatter-add |
| `dense` | any linear, at most four at once | D [out, in], bf16, 0.215 GiB for one feed-forward matrix [arithmetic from the shape] | y[rows] += x[rows] @ D^T |
| `embed_row` | the tied embedding and unembedding | token ids, D [k, 5376], and which side: both, in, out | in: added to the embedding's output where the id matches; out: added to that piece's logit in the press's own fp32 read-out |
| `norm_gain`, `layer_scale` | an RMSNorm weight; a layer scalar | a vector or a number | scaling that module's output for the row |

Why these: they are the forms editors actually chose in the published work (column direction-subtract, column scale or
zero, head scale, rank-one, embedding rows; lineage digest, with the correction that embedding-matrix, multi-layer and
mid-layer edits also worked), and each one is exact as an operation at the linear's input or output, which is what lets
it ride on a row.

The feed-forward block is `down(act(gate x) * up x)` (transformers' `Gemma4TextMLP.forward`, read today), so a neuron's
gain may be placed on its `down_proj` column or on its `up_proj` row with the same result.

The press refuses at registration what cannot be an edit: a uniform scale of a whole block's `down_proj` or of every
head of a layer (the norm after it undoes it; backend/atlas_ops.py:244-264), an index outside the loaded model (the
vault clamps instead; backend/larql_vault.py:481, 500), a `v_proj` edit on the ten full-attention layers, which have no
`v_proj` (5, 11, 17, 23, 29, 35, 41, 47, 53, 59).

### 2.2 How it is applied

Each operation becomes a forward hook or a forward pre-hook on its module for the length of one forme, holding one value
per row. The module's own forward is not replaced and its base product is not recomputed, so the numeric path of the
unedited rows does not change. The additive term is computed in float32 from the bf16 input, added to the product in
float32 for the edited rows only, and rounded once to bf16, which is how `int4_cpu.with_overlay_terms` does it. Unedited
rows are not touched at all.

Two properties must be proved at install, by bits, in the way the unpackers prove theirs (pilot 2): with hooks installed
and no edit, every row equals the plain forward; and a row carrying an edit of zero equals the base row.

Cost of the term against the product it rides on [arithmetic]: one feed-forward product is 1.16e8 multiply-adds a
word-piece; a rank-64 term is 64 x (5,376 + 21,504) = 1.7e6, about 1.5%. A `gain` or a `col_add` on a few dozen
neurons is smaller still.

### 2.3 Composing, undoing, hashing, saving, reloading

- **Composing.** A program is an ordered list. It compiles to one normal form per linear:
  y' = g_out * (W (g_in * x)) + dW x, with gains multiplied in order and additive terms summed, a later gain folded into
  the additive terms it follows. Two programs with the same normal form by bits share an "effective id".
- **Stacks.** A kept program becomes the **parent** of the next round: parent rows carry the stack, candidate rows carry
  the stack plus the candidate, and a row of the shipped model can ride along to show the drift of the whole stack.
- **Undoing.** There is nothing to undo, because nothing was applied. Dropping one erratum is deleting one item of a
  list. This removes the vault's all-or-nothing ROLLBACK from the loop altogether.
- **Algebra, all typed so the ledger knows what each row is:** `scale(p, s)` for dose ladders; `restrict(p, subset)` for
  leave-one-out; `negate(p)`; `null_like(p, seed, kind)` for a random edit of the same sites and the same Frobenius norm,
  with the census's two control kinds (through the post-feed-forward norm, or mixed from the layer's own write
  directions; `census.control_dirs`, backend/census.py:1567); `shift_like(p, seed)` for the same operation on random
  neurons of the same layer; `snap(p)` (2.4).
- **Hashing.** Canonical JSON with sorted keys; arrays named by the sha256 of their little-endian bytes with dtype and
  shape. The program id is the sha256 of that. The base is pinned twice: by the weights card, and by a sha256 of the
  packed words and scales of each tensor the program touches. The second pin is new and is needed because the weights
  card names files and the GGUF fingerprint skips the middle of the file (backend/gguf_model.py:741-764).
- **Saving and reloading.** A program is a folder under the git-ignored data folder: `program.json` plus one array
  file. Reloading is reading it and checking both pins; a program whose pins do not match the loaded model is refused in
  plain words.
- **Export.** Owner-only, by script, never by a tool. Low-rank, column, row and gain forms go out through
  `export_lora.write_lora`, which already takes any tensor name and any rank. The known limits carry over: llama.cpp
  adds adapters where the instrument multiplies gains, acts on the start mark, and whether its Gemma 4 graph applies a
  LoRA to tensors other than `ffn_down` and `attn_output` is not established.

### 2.4 The 4-bit grid

A program has one of three standings and every reading names it.

1. **Float** (the default). Off the grid, as a vault edit is today. This is what the fast loop measures.
2. **Snapped.** `snap(p)` derives the nearest program the int4 layout can hold, and returns with it the share of the
   change kept, the number of entries clamped to the range, and the number of codes changed. The rules follow the
   layout (32 consecutive inputs of one output row share a scale):
   - a neuron's `gain` becomes a multiplication of the 168 scales of that neuron's `up_proj` row, codes untouched; a
     negative factor is a negative scale, zero is zero scales. This is my derivation and is untested (pilot 8);
   - a `row_add` on `gate_proj` or `up_proj` re-encodes that row's 168 blocks, which belong to the neuron alone;
   - a `col_add` on `down_proj` can only move each entry by whole steps of a scale it shares with 31 other neurons, so
     each entry is rounded to the nearest of 16 levels. Small changes vanish, as the literature reports for 4-bit
     requantisation (Zhang et al.: retained forgotten knowledge 21% to 83%).
   The snapped program is itself a difference from the shipped tensor, so it **rides as one more row beside the float
   program in the same forward**. Survival under the grid is a column of the result table at no extra cost. No public
   work has measured this on a QAT checkpoint.
3. **Written.** Baked into a copy of the checkpoint or a GGUF. No writer exists for a packed 31B. This is a late phase,
   owner-only, with a tensor-by-tensor comparison against base plus program afterwards.

### 2.5 How a reading records its edit

Every row of every result carries: program id and effective id, parent id, standing (float, snapped, written), the apply
class `rowdelta-f32-v1`, both base pins, suite id and tier, the forme's shape, the start and stop layers, the unpacker
and its fall-back count, the KL direction, the code commit, the time and the seconds of the card. The model id is not
changed, because the model was not; the reading's class is carried in the record instead.


## 3. How the agent reads

A language model cannot read 29.3 billion weights, and the evidence says it should not try: tool-free reading has been
shown only to about 172,000 weights (research digest). But on this card the whole of the weights passes through the
unpacker in one decode step, 0.091 s [measured, 28 October 2026], and one feed-forward matrix unpacks in 0.24 to 0.46 ms
[measured; ARCHITECTURE.md:1317-1320, date not stated in the digest]. So the right representation is not a store of
weights as text; it is **a set of questions about the weights that each answer in well under a second**.

Six read surfaces, cheapest first.

1. **The tensor register.** One row for each of the 410 linears and the embedding: name, layer, role, shape, layer
   kind; Frobenius norm; the top 64 singular values, stable rank and participation ratio; percentiles of row and
   column norms; the 16-level code histogram, the share of codes at the two ends, percentiles of the group scales. The
   top 16 singular vectors each side and the full norm arrays are stored and named by handle, not printed. Built once
   for the shipped model; about 0.4 GB on disk [arithmetic]. Its build time is not known [to pilot; pilot 4]; I expect
   under an hour of the card. As a table the 410 rows are about 60 KB, small enough to filter and send.
2. **The neuron card.** The atlas's join for one neuron or a list: lens words under the label gate, activation range,
   24 strongest links each way, steer lens with its standing, census rank where a census exists, stored passages. Each
   reply carries the trust that applies at that depth: labels held back below layer 40, steer lens sound at 39-52 only,
   doses calibrated at 32-52 only, link signs below chance at an ordinary position (0.336 from rest) [measured;
   atlas digest]. Census reads cost 0.13 s for a layer's list and 0.01 s for a neuron [measured, 12 November 2026].
3. **Direction scans.** Given a direction in the stream (a word-piece's unembedding row, a difference of means from a
   contrast set, a column, a singular vector), return across all 60 layers the neurons that write along it and the
   neurons and heads that read it, with cosines, norms and write ceilings. Sixty unpackings and sixty products:
   60 x (0.24 to 0.46 ms + 0.147 ms) is under 0.05 s [arithmetic from the measured per-matrix figures]; budget 0.2 s.
   This is the question the Sparks driver answered by hand every run.
4. **The lattice view.** Any slice of at most 4,096 entries as integer codes times group scales, for the rare case
   where actual numbers matter.
5. **Activations on the agent's own texts.** Neuron activations, stream projections, attention patterns at chosen
   places, by the same span runner, at the cost of a forward up to the layer asked.
6. **Attribution tables.** For a metric on a suite: the first-order effect of silencing, doubling or redirecting every
   one of the 1,290,240 neurons and every head, from one backward pass (4.2, device 5). Of the six this is the view of
   the weights that bears most directly on a decision: weight times activation times gradient.

Large arrays never travel through the driver's context. They stay in a server-side store under handles (2 GiB of the
card, 16 GB of host memory; the machine has 61 GB, 51 available [measured today with `free`]). A tool reply is a table
under the 16,000-character limit the endpoint loop already enforces; the kernel fetches arrays by handle.

Word-piece strings shown to the driver are short ASCII excerpts with ids, since one provider filter refused a turn that
printed raw vocabulary in the Sparks logs.


## 4. How a hypothesis is tested, fast

### 4.1 The cost model

Two constants, both from the repository's own records with the compiled unpacker.

- **Floor, f = 1.5 ms a layer.** A decode step at one row is 0.091 s [measured, 28-29 October 2026]; divided by 60
  [arithmetic]. It is mostly the cost of unpacking, paid whatever the batch. By the per-matrix figure (0.24 to 0.46
  ms for 115.6 million weights) unpacking all 29.3 billion is 0.06 to 0.11 s, so at one row the step is nearly all
  unpacking [arithmetic; the split itself was never timed].
- **Product, c = 7.2 microseconds a row-token a layer.** The steer-lens build timed two layers on 29 October 2026; a
  line through them gives 22.35 s per layer for 336 batches, 66.5 ms a batch a layer. A batch is 516 rows
  (4 + 2 x 64 x 4; backend/atlas_build.py:775-836) padded to about 18 word-pieces (I counted the four contexts with
  the tokenizer on the CPU today: 17 and 18 for the two chats under the model's own template, 9 and 7 for the plain
  openings; the repository's `compose_prompt` may differ by a piece or two). 66.5 ms / 9,288 = 7.2 microseconds
  [arithmetic].

Cross-checks [arithmetic on measured records]: the census prefill, (0.5996 - 0.1025) / 95 rows = 5.2 ms a row of about
19 pieces, gives 4.6; the decode step's margin, (0.1263 - 0.0912) / 95, gives 6.2; the passages pass, 2,000,054
pieces in 20 minutes with every activation hooked, gives 7.1. Four records give 4.6 to 7.2. I plan on the slowest.

> **t(N, L to M) = (M - L) x (1.5 ms + 7.2 microseconds x N)** for N row-tokens run from layer L to layer M.

A whole forward of one word-piece is 0.09 s; of 1,024 row-tokens 0.53 s; of 9,216 row-tokens 4.1 s; the same from
layer 45, 1.0 s. All [arithmetic]; pilot 1 measures the grid and replaces the constants.

**The forme.** One forward holds at most about 9,216 row-tokens by default. That shape (516 x 18) ran through the
whole steer-lens build beside the lens [measured as having run, 13 November 2026; its peak memory was not recorded in
what I read]. Pilot 1 measures memory per row-token and may raise the limit to 16,384.

### 4.2 The seven speed devices

**1. Row-scoped edits, so candidates and the unedited model share a forward.** Rows of one forward are bit-identical
where their inputs are; rows of different batch shapes are not (KL up to 0.014; an unpushed answer in a batch of 64
parts from itself after 22 word-pieces) [measured, October 2026]. Putting the base row in the forward is therefore the
right comparison as well as the cheap one. How many candidates share a forme depends only on the prompt set: with a
screening set of 16 prompts of 24 slots (384 row-tokens) a forme holds the base and 23 candidates; with 8 prompts, 47;
with one contrast pair, hundreds, which is what the steer lens already does with 128 pushes. The published support for
additive per-row low-rank terms is the batched-matmul baseline in PEFT and EGGROLL, not FLoRA's faster figure, which is
for multiplicative gains (research digest, correction).

**2. Restart from the lowest edited layer.** The stream entering layer L is the same for every candidate that edits
nothing below L. `forward_from_layer` already does this for the steer lens. The scheduler sorts candidates by lowest
edited layer and cuts them into formes so that few rows are run from lower than they need. Two special cases cost
nothing: an edit to the unembedding side or the final norm needs no forward at all, only a new read-out of cached
final states; and a hypothesis about an internal effect ("does this edit at 31 change that neuron at 34") stops at the
layer read, which the span runner allows and `forward_from_layer` does not.

**3. Memoising the parent.** Kept on host memory and on disk, keyed by base pins, parent program, suite and shape:
- the stream at all 60 boundaries for each suite: 2,048 row-tokens x 60 x 5,376 x 2 bytes = 1.3 GB [arithmetic];
- the parent's greedy answers for each suite at each fixed writing shape;
- for each (suite, metric), the cotangent at all 60 boundaries after the first backward (device 5).
The parent's teacher-forced log-probabilities are **not** memoised. The base rows ride in every forme instead, at a
cost of 1 row in (K + 1), and double as a standing canary: the base rows' read-out must match the first forme's by
bits, or the job is marked.

**4. Teacher-forced by default, and generation-free parting.** The default read-outs are the log-probability of named
target and foil pieces, a logit difference, per-position negative log-likelihood, and KL(base || edited) at scored
positions, all from fp32 logits. The answer-level question the vault's voice asks, "where do the two answers part",
does not need generation either: run the edited rows teacher-forced along the parent's stored answer, and the first
position where the edited row's likeliest piece differs from the stored piece is where a greedy answer would part.
That is one forward for all candidates instead of one generation each. Near ties are flagged, because the no-cache
forward and the cached decode step are different shapes. Only the text after the parting needs writing, and only for a
candidate worth reading.

**5. One backward pass as an attribution tool.** For a metric m on a suite, the first-order effect of any edit dW on a
linear is the sum over positions of g_y . dW x, where x is the linear's input and g_y the gradient at its output. For
a low-rank edit that is (B^T g_y) . (A x): two small products. So one backward gives the predicted effect of silencing,
scaling or redirecting each of the 1,290,240 neurons, and of any list of candidate programs.
The repository has run backward passes through the packed weights with a recompute-in-backward linear: forward with
graph 0.661 s, backward 0.314 s, for a 128-piece prompt on the reference unpacker, with about 2 GiB of graph
[measured, 22 October 2026; whether it crossed all 60 layers is not established]. Four such prompts at once ran out of
memory. The press avoids that limit with the stream cache: it walks layers 59 down to 0, recomputes one layer with a
graph from its cached input, pulls the cotangent through that one layer, harvests, and frees the graph. Memory is one
layer's graph, about 0.27 to 0.54 MB a row-token [arithmetic from the 2 GiB figure, the range for 60 or 30 layers
crossed], so a 9,216 row-token forme needs 2.4 to 4.9 GiB and a 2,048 row-token suite 0.5 to 1.1 GiB. Time is about
three forwards of the same size [to pilot; pilot 5]. After the first sweep the boundary cotangents are cached, and
scoring a new list of candidates at one layer costs one layer of forward and backward, under 0.1 s [arithmetic].
The top of the ranked list goes straight to a real forme. The press keeps, layer by layer, the rank agreement between
what the backward predicted and what the forward measured, and shows it as a standing (agrees, weak, unchecked), as the
steer check does for the steer lens.

**6. Precomputed tensor statistics and live scans** (section 3, surfaces 1 and 3). A scan of all 60 layers fits one
turn because unpacking all the weights costs 0.09 s.

**7. Partial dense caches: worked, and declined.** Holding a layer unpacked costs 0.89 GiB (0.99 for a full-attention
layer) [arithmetic from the shapes] and saves that layer's unpacking, at most about 1.5 ms a forward. In a full forme
the same layer costs about 66 ms of products, so the saving is about 2%. In generation a step is nearly all floor, but
four dense layers (3.6 GiB) save at most 6 of 91 ms, and the same memory holds about 45 more rows of a writing batch
for 0.02 s more a step (4.8 to 5.3 GiB for 64 rows of 64 new pieces; 0.109 s a step at 64 rows against 0.126 s at 96)
[measured, 28-29 October 2026]. So the headroom goes to
rows. The one dense object worth holding is a dense **difference** for an edit that will not factor. The split of a
compiled step between unpacking and products has never been timed; pilot 4 times it, and its rule is stated there.

### 4.3 Latency budget

"Forme" below means 9,216 row-tokens. Times from layer L use t(N, L to 60).

| Operation | Size | Sum | Budget | Standing |
|---|---|---|---|---|
| Census list for a layer; one neuron's answers | | | 0.13 s; 0.01 s | measured, 12 November 2026 |
| Register or atlas index read | numpy on the CPU | | under 0.05 s | arithmetic |
| A column or a row in the lattice view | only the blocks touched | | under 0.01 s | arithmetic |
| Direction scan over 60 layers | 60 unpackings, 60 products | 60 x 0.6 ms | 0.2 s | arithmetic |
| A streaming statistic over all 410 linears | one step's unpacking | | 0.1 to 0.5 s | arithmetic |
| Top-64 SVD of one matrix on demand | 5,376 x 21,504 | | 1 to 5 s | to pilot |
| Suite preparation: parent forward, 60 boundaries | 2,048 row-tokens | 60 x 16.2 ms | 1.0 s, once | arithmetic |
| One forme from layer 0, 30, 45, 55 | 9,216 row-tokens | (60 - L) x 67.9 ms | 4.1, 2.0, 1.0, 0.34 s | arithmetic |
| fp32 read-out of 512 scored positions | 512 x 262,144 logits | 1.4e12 flops | 0.05 to 0.1 s | to pilot |
| Attribution sweep | 2,048 row-tokens | about 3 forwards | 3 s | to pilot; anchor 0.98 s on the reference unpacker, 128 pieces, 22 October 2026 |
| Re-scoring candidates on cached cotangents | one layer | | 0.1 s | arithmetic |
| Parting along stored answers | 16 prompts x 68 slots x 4 rows | 4,352 row-tokens from 45 | 0.5 s | arithmetic |
| Writing, 32 rows x 48 pieces | | 0.26 + 48 x 0.103 | 5.2 s | measured constants, 28-29 October 2026 |
| Writing, 96 rows x 48 pieces | | 0.60 + 48 x 0.126 | 6.7 s | measured constants, 28-29 October 2026 |
| Sealed trial, teacher-forced | 256 items x 24 slots x 10 rows | 61,440 row-tokens | 27 s from 0; 7 s from 45 | arithmetic |
| Sealed trial, written | 256 prompts x 48 pieces, batches of 32 | 8 x 5.2 s | 42 s, parent stored | arithmetic |

For scale, what exists today: one heard vault edit, one edit on one question with no control, 12.6 s [measured; date
and unpacker not stated]; a `/steer` call of 3 to 15 rows of 80 pieces, 8.3 to 9.2 s [measured, 11 November 2026];
the study's lab jobs, median 71 s [measured, 16 October 2026]; the Sparks run, 50 passes in 20.3 hours.

### 4.4 A whole hypothesis

Take "these ten neurons at layer 45, silenced, move this logit difference by at least d and leave unrelated text
alone". The press runs it as five steps, the first four in **one call**.

| Step | Rows | Card time at layer 45 | at 30 | at 0 |
|---|---|---|---|---|
| Screen: 23 candidates and the base on 16 prompts | 1 forme | 1.0 s | 2.0 s | 4.1 s |
| Standard controls for the best: doses 0.01, 0.25, 0.5, 1, 2; the sign flipped; 15 matched nulls; the snapped form | 1 forme | 1.0 s | 2.0 s | 4.1 s |
| Side-effect panel: base and edit on 128 corpus windows of 36 slots | 1 forme | 1.0 s | 2.0 s | 4.1 s |
| Parting along the parent's stored answers | half a forme | 0.5 s | 1.0 s | 1.9 s |
| Confirmation set: edit, 7 nulls, base on 64 held-out prompts | 2 formes | 2.0 s | 4.1 s | 8.1 s |
| **Teacher-forced total** | | **5.5 s** | **11 s** | **22 s** |
| Written answers for a kept lead (the exception) | 32 rows x 48 | 5.2 s | 5.2 s | 5.2 s |

All [arithmetic] except the writing constants. Against that, the driver. The Sparks logs put their driver's turns at 20
to 70 s; this instrument has no measurement of its own (pilot 10). Two to four turns is 40 to 280 s.

So one fully controlled hypothesis is 1 to 5 minutes of wall time, of which the card works 6 to 27 s: **12 to 60 an
hour for one driver, with the card busy between 2% (6 s in 286) and 40% (27 s in 67) of the time** [arithmetic].
Screening alone is bound by the card at about 23 candidates a second at layer 45 and 5.6 a second at layer 0
[arithmetic]; no driver thinks that fast.

Three consequences shape the interface, and they matter more than any further speed-up of the forward:

- one call carries a **family** of candidates, written as a generator ("the top 500 of this attribution table, at
  factors 0 and 2"), and returns a ranked table;
- the controls, the panel and the snapped form run **without being asked**, in the same job, so a hypothesis is one or
  two turns and not five;
- the queue serves **several sessions**. Because no session changes the model, sessions need no isolation from each
  other on the card. At a middling duty of 10 to 20% a driver, five to eight drivers fill it [arithmetic].

### 4.5 What guards the speed itself

The press checks before every job that the unpacker is the compiled one and after it that the fall-back count did not
rise; a forward under `torch.inference_mode` or autocast silently falls back to the reference and costs seven times
as much (backend/w4a16_fast.py:205-229). Every job logs predicted against actual seconds, and the two constants are
refitted from the log.


## 5. What stops the agent fooling itself, and the owner

Speed makes this section more necessary: at 23 candidates a second, chance alone produces leads by the hundred.
The devices below are the ones that cost little time. Where a slower device would be sounder I say so.

**In every forme, at no extra turn.**
- The base rows, compared by bits with the first forme's.
- A dose of 0.01 of the edit as the rounding floor, the causal check's own idea (backend/causal_check.py:53).
- Matched nulls made by `null_like` and `shift_like`: the same sites with a random direction of equal norm, and the same
  operation at random neurons of the same layer. The empirical p is (1 + nulls at least as strong) / (C + 1), and the
  reply states that about N / (C + 1) of N candidates beat every null by chance, as the census does.
- The cheaper interventions as rows of the same forme: the same direction as an activation add, a logit bias, the
  existing push. A prompt sentence as its own row. Several published weight edits were matched by exactly these
  (research digest), and here the comparison costs rows, not turns.

**Three tiers of data for each behaviour.** Practice items the agent sees in full. Confirmation items it never sees:
the press returns the effect in standard errors and a pass or fail against a rule fixed before the call, and the
number of confirmation calls per program and per session is capped. Sealed items the owner opens once for a registered
claim. The confirmation and sealed files live where the kernel's sandbox cannot read them, and only the server reads
them. Today's one hold-out is enforced in a tool and not at the route (backend/atlas.py:1612-1614); the press enforces
at the route.

**The side-effect panel.** KL(base || edited) and agreement of the likeliest piece on corpus windows unrelated to the
behaviour, with non-finite logits masked, plus the stored unpushed answers of the census questions read by parting. It
runs on anything that goes past screening. The practice panel uses even-numbered windows; the sealed panel the odd ones.
The published side-effect checks shared the fault being fixed and could only improve; this one does not.

**The ledger.** Written by the server, not the agent, for every plan: the plan's hash, every program id, the suite and
tier, the shape, the result table, predicted and actual seconds. Append-only and hash-chained. It counts candidates per
hypothesis family, so a lead is corrected over everything screened in that family and not only over one call
(`mcp_scientist.benjamini_hochberg`, backend/mcp_scientist.py:831). Failed and discarded candidates stay in it. The
agent's own notes are a separate file and are marked unverified.

**A forecast in every plan.** One field: which metric, which way, at least how much. It is hashed into the ledger before
the result exists, and the driver's hit rate is kept. It costs no turn.

**Registration at two speeds.** This is where my lens departs from the house habit, and I say so plainly.
- *Fast registration*: a ledger entry that fixes program, suites, metric, threshold, nulls and rule, hashed before the
  confirmation run. The press applies the rule. Seconds.
- *House registration*: the hashed folder committed before the run, run from a clean checkout, read once on the CPU, as
  the census trials are. Required for anything the owner would act on or publish.

**What a result may claim.** Four words, fixed.
- *screened*: seen on practice items only;
- *lead*: beats its nulls on practice items, corrected over its family;
- *confirmed (teacher-forced)*: beats its nulls on the confirmation set under fast registration;
- *passed*: beats its nulls on the sealed set, with written answers scored, under house registration and the owner's word.

"No edit kept" is an ordinary recorded outcome; six of sixteen published searches ended that way.

**What the press will not say.** It does not show a lens label of an edited neuron as evidence; the label gate was never
measured on edited neurons, and the vault's UPDATE is built through the same lens. It does not present atlas or census
facts as describing an edited model. It does not call a first-order score a measurement. It does not treat a float
reading as a reading of the snapped or written form.

**What this costs.** Teacher-forced scoring is the most severe of the flaws that inflated editing success from 38.5%
to 96.8% in one study, so *confirmed* carries "(teacher-forced)" in its name and only *passed* requires written
answers. Fast registration is weaker than the house's. Single sessions vary widely (+29.45, +5.00, +27.30 on one task
in one study), and the press does not repeat a search by itself; it only makes a repeat cheap.


## 6. The agent interface

**One contract: the plan.** A plan is a JSON object that names a suite, a parent, variants (each a list of errata or a
generator), read-outs, the controls wanted, a forecast and a budget. The typed tools, the kernel's client and the HTTP
route all carry the same object. A driver that can write JSON can use the press; a driver that can write Python can
write plans in loops.

```json
{"suite": "she-he/practice", "parent": "base",
 "variants": [
   {"name": "silence10", "errata": [{"op": "gain", "site": "neuron", "layer": 50, "idx": [/* ten */], "factor": 0}]},
   {"generate": "shift_like", "of": "silence10", "count": 15, "seed": 1}],
 "read": [{"metric": "logit_diff", "target": " she", "foil": " he", "at": "last"}],
 "controls": "standard",
 "expect": {"metric": "logit_diff", "sign": "-", "at_least": 1.0},
 "budget_s": 10}
```

The reply is a table, the seconds predicted, spent and left, and the ledger id. That example is the study's one causal
edit test (silencing ten layer-50 'she' neurons moved the female contexts by -1.80 log-odds [measured, the E6 audit])
and is the first thing the press should reproduce.

**Six tools**, added to the existing registry so all three backends get them:
`press_read` (register, neuron cards, scans, lattice slices), `press_estimate` (free: predicted seconds and whether it
fits), `press_run` (a plan; returns the table, or a job id if it runs past a few seconds), `press_job` (status, result,
cancel), `press_attribute` (the backward sweep and re-scoring), `press_confirm` (fast registration and the confirmation
run). Layers go through `from_page` and `to_page` like every other tool.

**Code execution** through a seventh tool, `press_exec`, added in phase 3: Python in the CPU-only kernel with numpy and
the press client loaded, a namespace that lasts the session, and its own work folder. The kernel holds no model and no
card. It reads arrays by handle and submits plans.

**How a current model plugs in today.** Through the scientist chat (Claude Code or Codex over the stdio MCP server, or
any OpenAI-compatible endpoint through `endpoints.json`), or from outside with the same MCP server and
`THEODOLITE_URL`. No new adapter.

**How a stronger one plugs in later.** The same plan. It will write larger families and longer kernel programs and need
fewer of the bundled controls spelled out; none of that changes the contract. Typed-tool evidence comes from mid-2026
models and may not hold for the next generation, so the tools are kept as thin projections of one route and the kernel
is there for a driver that prefers code.

**Budgets** are in seconds of the card, not in passes, shown with every reply. Confirmation calls are counted
separately. A session, a job and a forme each have a limit; a job is cancelled between formes.

**State across sessions** is on disk and outside any conversation: the ledger, the programs, the suites, the handles'
store, the agent's notes file. A session starts fresh from those. In the Sparks logs the driver ran 26 turns with its
conversation frozen and survived on its notebook and work folder (research digest, correction); the files must be
enough by themselves.

**Left thin on purpose.** No workflow engine, no fixed stages, no councils, no context-reset machinery, no per-tool
rendering in the page at first (a new tool already shows in the thread as a collapsed row with no frontend change).
The method goes into one short third prompt file through the existing prompt builder. Reasoning effort is left at the
driver's default; the evidence that more effort helps is thin.


## 7. Safety and scope

**Containment.**
- The process that holds the model runs typed plans only. This is the main line of defence, and it is cheap because
  the plan vocabulary is small.
- The kernel is the only place the driver's code runs: no network, no card devices, the repository read-only, writes
  only in its work folder, memory and time limits, a per-launch token for the press's socket. The driver's default
  reach stays "folder". Rules in a prompt did not hold for agents with free code in the published work, so these are
  enforced by the operating system.
- The confirmation and sealed sets, the ledger and the tools' configuration are not writable or readable from the
  kernel.
- Everything the edited model writes and every dataset string is data when shown to the driver.
- The lab daemon is not exposed to the driver in any phase.

**What happens to edited weights.** There are none in memory. A program is a small file in a git-ignored folder. It
leaves the machine only through an owner-run export script, with a manifest: base pins, the exact program, the ledger
entries including nulls and discarded attempts, a notice that the files are modified, the Apache 2.0 text, and a name
that says "derived from Gemma". The repository redistributes no weights today and this does not change that.

**What the press will not do.** No optimiser and no training: the backward pass ranks, it does not fit. No persistent
global edit by an agent. No export, publication or vault promotion by a tool. No preset suites or recipes for removing
safety behaviour or for trigger-to-payload implants; a weight-edit backdoor has been demonstrated on this model in
bf16 (research digest, with its qualifiers), and the press should not ship the recipe. No model family but Gemma 4 31B.


## 8. A phased path

Effort is in working days of the owner with coding sessions, rough.

| Phase | Delivers | Proves | Effort |
|---|---|---|---|
| 0. Pilots 1 to 3 | Three small card scripts; a filled-in cost table. | Whether the cost model and the same-forward design hold. | 1-2 |
| 1. The kernel and one evaluator, by command line | `press_deltas.py` with `gain`, `col_add`, `row_add`, `low_rank`; `press_run.py` with the stream cache and paired teacher-forced read-outs; `check_press.py` on the stand-in; `verify_press.py` on the card; `scripts/press.py run plan.json`. | That K edits and the base share a forward at the predicted cost, and that the E6 result (-1.80) reproduces with 15 matched nulls in one call. Useful at once to the owner. | 4-5 |
| 2. Programs, ledger, suites, controls, queue | The program format and hash; the algebra; the three tiers; standard controls; the panel; `/api/press/*` under `_GPU_LOCK`; budgets. | That a hypothesis is one call and leaves a full record. | 6-8 |
| 3. The driver | Six tools in the registry; the method file; the kernel and `press_exec`; one hour with a real driver (pilot 10). | That a current model produces controlled hypotheses at the predicted rate. | 5-6 |
| 4. Reading | The tensor register; direction scans; the lattice view; the backward sweep with its standing by layer. | That one backward ranks candidates well enough to save forwards (pilot 6). | 6-8 |
| 5. Writing and the grid | Parting along stored answers; written answers with the parent stored; `snap`; sealed trials with house registration; the owner's export script. | Whether an edit survives the 4-bit grid on this checkpoint, which no public work has measured. | 8-10 |
| 6. Later, each by itself | A read-only ledger page; promotion to the vault; several drivers at once; a written checkpoint; the 12B as a rehearsal object. | | open |

Phase 1 is small on purpose: two modules, two check scripts and a command, no server change, no agent.


## 9. Pilots, in order

Each runs in minutes on the card with the server stopped, and each has a result that changes the design.

1. **The cost grid.** Time the span runner without edits at rows x length in {64, 256, 512} x {16, 32, 64} from
   layers 0, 30, 45, 55, with peak memory. *Changes:* the two constants; the forme limit. If memory a row-token is
   above 0.5 MB the forme stays at 9,216; if the product constant is above 15 microseconds the default suites halve.
2. **The row edit against the folded edit.** With hooks installed and no edit, are all rows the plain forward by
   bits? Is a zero-edit row the base row by bits? How far is a row-edit reading from the same edit as a vault overlay
   in a forward of the same shape, against the batch-shape floor? What does a hook cost? *Changes:* if unedited rows
   are not bit-equal, the same-forward design fails and the press falls back to two forwards of one shape, doubling
   every figure in 4.4. If the gap to the folded form exceeds the floor, every kept lead is re-read folded before it
   is called confirmed.
3. **Restart and repeat.** Is a forme from a cached stream equal to the whole forward of the same shape (checked only
   on the stand-in at 1e-5 so far)? Are two same-shape formes identical across calls and processes? Does a row depend
   on what its neighbours carry? *Changes:* if the cache is not bit-stable across processes it is rebuilt at start
   (about 1 s a suite) and not kept on disk.
4. **The split of a step, and the register's build time.** Time unpacking against products at 1, 32 and 96 rows; time
   a forward with the top k layers held dense; time norms, histograms and a rank-64 SVD for one tensor of each of the
   eight shapes. *Rule:* dense caches are built only if they cut a typical screen by more than 20%; my arithmetic
   says about 2%.
5. **The backward.** Does the recompute-in-backward linear keep the compiled unpacker, or fall back? Does the
   layer-by-layer sweep from cached streams equal the whole-graph gradient on a 128-piece prompt? Time and memory a
   row-token. *Changes:* if graph memory a row-token-layer is above 1 MB the sweep is chunked; if the sweep costs more
   than six forwards, attribution becomes an asked-for tool and not the default first step.
6. **Does the first-order score rank?** For 256 random single-neuron gains at each of layers 5, 20, 35, 45, 55:
   rank agreement of the predicted with the measured change. *Changes:* where it is below 0.5 the standing is "weak"
   and screening there is by forward only. The norm after the feed-forward block saturates large pushes at layers
   20-50 [measured in the study], so I expect it to hold for small edits and fail for large ones.
7. **Power.** The standard error of the paired logit difference under matched nulls for 16, 64 and 256 prompts.
   *Changes:* the sizes of the three tiers. Published held-out gains on a strong model were at most about three
   points, so the confirmation set may need to be several times 64.
8. **The grid.** Is a neuron's gain through its `up_proj` row's scales the same as the float gain, and to how many
   bits? What share of a `col_add` survives snapping, as a function of its size against the group scale? *Changes:*
   whether "snapped" is offered for column edits at all, and what the export script promises.
9. **Stored answers.** Is the parent's greedy answer in a batch of 32 identical across calls, with hooks installed and
   other rows edited? *Changes:* if not, the parent is written again in every writing batch and the writing figures
   in 4.3 double.
10. **A driver for an hour.** A current model, the phase-3 tools, the E6 task and one open task; turns an hour,
    seconds of the card per turn, how often it uses families and the kernel. *Changes:* whether the next effort goes
    into several drivers at once or into the forward.


## 10. Risks, what is left out, and what is the owner's

### Risks

- **The row edit is not the folded edit and not the grid.** Three numeric classes. Mitigated by naming the class in
  every reading, by pilot 2, and by the snapped row riding in the same forme.
- **Speed manufactures false leads.** Mitigated by family-wide correction in the ledger, capped confirmation calls and
  the sealed tier; not removed.
- **Teacher-forced results flatter.** Named in the word "confirmed (teacher-forced)"; written answers required for
  "passed".
- **The card may not be the bottleneck at all.** If pilot 10 shows the driver takes minutes a turn, devices 5 to 7
  matter less than the interface, and phase 4 should shrink.
- **A healthy 31B may offer little to edit.** Published edits removed an obstacle or shifted a default; on stronger
  models most searches kept the original. The press should be judged on how fast and how honestly it tests, not on
  gains.
- **The first-order score may mislead at depth** (pilot 6).
- **Caches are state.** A stale stream cache would silently corrupt every reading; hence the two pins, the base-row
  canary and the refusal while vault edits are in force.
- **Hooks beside other hooks.** Each packed linear's forward is already replaced in turn by several mechanisms; the
  press uses ordinary hooks and refuses to run under labkit's gradient linears except in its own sweep.
- **The plan vocabulary will be too small for something.** Then the owner adds an operation or uses the lab daemon.
- **Shared card.** While a census layer or an intake spot-check holds the card, the press is down like everything else.

### Left out, on purpose

- Gradient descent or any optimiser on an edit.
- Driver code in the process that holds the model.
- The frozen-prefix shortcut for edits. It would be about T times cheaper for screening, but an edit acts at every
  position and the repository's own constraint forbids it.
- Dense caches of base weights (pending pilot 4).
- Rebuilding any atlas index or census for an edited model.
- Key-covariance edits (1.85 GB a layer in float32), SAE dictionaries, decomposition of a matrix on demand.
- Other model families; the 12B as an object, though a trained 12B QAT GGUF is on the disk and has never been built.
- Truncating the last layer to scored positions; batching different sessions' rows into one forme.
- A page beyond a read-only ledger.

### The owner's decisions

1. Whether the press lives in the server process, as designed, or takes the card as the lab daemon does.
2. Whether a single backward pass for ranking is allowed by default. It makes results not comparable with the
   gradient-free published work; the level is recorded per result either way.
3. Whether fast registration may carry the word "confirmed", or whether only house registration earns any word past
   "lead".
4. The pin on `backend/engine_runtime.py`, which has not held since 12 November 2026. The press does not touch the
   file, but the owner should know before anyone relies on the pin.
5. Export policy: whether programs, adapters or written copies are ever published, and under what notice.
6. Which behaviours get suites, and that none targets safety behaviour.
7. The budget of the card per session and per day, and whether several drivers may run at once.
8. Whether to try a rented 96 GB card ($2.49 an hour as read on 28 October 2026), where resident weights would remove
   the unpacking floor and make writing several times faster. The design does not need it.
9. Whether the scientist in the atlas gets the tools, or a separate press agent does.
10. The name.


## Names

- **The Proof Press** (my choice). In printing a proof is a trial impression pulled quickly to see what a correction
  does before anything is printed. The words that come with it fit the mechanism: an *erratum* is one correction laid
  over a fixed printing, an *errata sheet* is a program, a *forme* is everything printed in one impression, which is
  the base and dozens of candidates in one forward. Caution: "proof" must not be read as "proved"; the Lexicon entry
  should say trial impression in its first line.
- **The Plane Table.** The surveyor's board on which a map is drawn in the field, sight by sight, and corrected on the
  spot. It says fast iteration against the ground and carries no risk of "proved".
- **Errata.** The plainest: the feature named for its object, corrections laid over a text that itself never changes.
