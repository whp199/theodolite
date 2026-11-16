# Levelling: a weight-edit kernel for Theodolite (design, kernel first)

Written 14 November 2026. Design only: nothing in the repository was changed, nothing was run on the card.
Layers are counted from 0 throughout (page = layer + 1) unless "page" is said.
Every figure carries its date and is marked *measured* (read from the two digests or from the repository) or
*my arithmetic* (derived here, not run).

## The position in one paragraph

Build one small thing first and prove it on the card before anything else is designed in detail: an **edit as a
value** (a hashed, saved, additive delta on named linears, never written into the model) and **one paired reading**
(base row, edited row and matched random edits as rows of the same forward, started at the lowest edited layer).
No server routes, no scientist tools, no page, no queue, no sandbox, no requantiser until that kernel has
reproduced a result the project already holds (the 16 October 2026 deletion of ten layer-50 neurons, -1.80
log-odds against -0.02 for ten random neurons) in under a few seconds on the 5090. Everything the other priorities
want (a driver surface, sealed test sets, a job runner, another model family) is a caller of this kernel and can
be added without changing it. If the kernel cannot do this, none of the rest is worth building.

Three facts from the code digest make this the right first step:

1. The existing weight edit (the vault's column overlay) applies to every row of a batch, so an edited and an
   unedited row cannot share a forward (backend/engine_runtime.py:356-375). On this card rows of different batch
   shapes differ by up to 0.014 KL (measured in the study, backend/causal_check.py:6-7) and an unpushed answer in a
   batch of 64 parted from itself after 22 word-pieces (measured 29 October 2026). Every honest comparison the
   project has made puts the base row in the same forward. Weight edits have no such path today.
2. A weight edit on a linear is exactly an added term on that linear's product: for a changed column i,
   `x[..., i] * (v - W[:, i])`. The repository already holds that arithmetic for the CPU kernel
   (backend/int4_cpu.py:138-164, summed in float32 and rounded once). As a forward hook it is per row, costs
   nothing beside the unpacking, never clones a 231 MB weight (my arithmetic: 5,376 x 21,504 x 2 bytes), and
   leaves the base product's numeric path untouched.
3. `atlas_ops.forward_from_layer` (backend/atlas_ops.py:1017-1043) already runs only layers L..59 from a stored
   stream with hooks firing. The steer-lens build used it to run 516 rows a forward through the 14 layers above
   layer 45 in about 0.85 s (measured 29 October 2026: 287 s for a layer; the division over 336 forwards is my
   arithmetic).

So the kernel is mostly a joining of parts that exist. The new idea is only this: **stop treating an edit as a
state of the model and treat it as an argument to a measurement.**

---

## i. What the environment is

Three library modules, one check script, one verify script. No new process.

| Component | What it is | Builds on | New code |
|---|---|---|---|
| **Edit** (`backend/weight_edits.py`) | A frozen value: a list of additive terms on named linears, with a content hash, save and load. Constructors for column, row and low-rank forms. A context manager that installs the terms as per-row forward hooks and removes them in `finally`. | The delta-term arithmetic of `int4_cpu.with_overlay_terms`; the hook pattern of `atlas_ops.installed` (backend/atlas_ops.py:344-412); `gguf_weights.dense_cols` / `dense_rows` (backend/gguf_weights.py:553-576) as the one reader of base weights on any holding; `census.control_dirs` (backend/census.py:1567) for matched random directions; `atlas_index.save_npz` for atomic writes. | About 400 lines. |
| **Levelling** (`backend/levelling.py`) | The paired reading: a prompt set, a capture of the base stream at layer boundaries, a forward from the lowest edited layer with base, edit, zero-edit and control rows, fp32 next-piece log-probabilities, a per-item table and a summary. | `atlas_ops.forward_from_layer`; the capture-once, rerun-above shape of `atlas_build.build_steer_layer` (backend/atlas_build.py:775-836); `causal_check.logits_fp32` (backend/causal_check.py:133); `mcp_scientist.paired_statistics` and `benjamini_hochberg` (backend/mcp_scientist.py:719, 831); later `atlas_ops.write_rows` for heard answers. | About 400 lines. |
| **Level book** (`backend/level_book.py`) | An append-only, hash-chained JSON-lines ledger written by the reading itself, not by the caller. | `census.write_json_atomic`, `file_sha256`, `runtime_record` (backend/census.py:1150, 1721, 1809-1846); `export_control_vector.code_commit`. | About 150 lines. |
| **Check** (`scripts/check_weight_edits.py`) | CPU, on the six-layer stand-in, house idiom (failures list, `check()`, exit code). | `scripts/atlas_stand_in.py`; the four-part edit check of scripts/check_gguf.py:825-874 (heard, seen, base untouched, rollback by bits). | About 300 lines. |
| **Verify** (`scripts/verify_weight_edits.py`) | GPU, on the 31B: the pilots of section ix as a standing script. | `w4a16_fast.same_bits`; the stored E6 result. | About 300 lines. |

**Which process holds the model.** In the first two increments, the lab daemon (experiments/lab/labd.py), started
with `THEODOLITE_UNPACKER=compiled`. It already keeps the 31B resident (ready in 13.3 s, 21.24 GB with both lenses,
measured 16 October 2026), executes submitted Python with `rt` bound, answers only loopback with a per-launch
token, and writes each reply to a server-side file. It defaults to the reference unpacker (0.71 s a step against
0.09 to 0.13 s compiled, measured 28 October 2026), so the environment variable is not optional. The kernel is a
plain library that takes `rt`; it does not know which process it is in.

**The server.** Down while the lab daemon is up: they share port 8765 on purpose and only one process may hold
the 31B. I accept that for the first two increments. In the third increment the same two functions are called
from two new routes under `_GPU_LOCK` through `asyncio.to_thread`, as every other GPU route is. Because the
kernel leaves no edit in force between calls, it can live in the server process without a single stale flag: no
index, census, stats unit or page reading is ever taken on an edited model.

**The vault.** Untouched. It stays the one-column, heard-on-one-question edit a person makes on the vault page.
The kernel refuses to read while any `_memit_col_overlays` or `_memit_row_overlays` dict on any of the 410
linears is non-empty, so a vault edit can never leak into a reading unrecorded. The overlay dicts are used by the
kernel for one thing only: the folded cross-check (section ii).

**The scientist.** Untouched until the fourth increment, when the kernel's three verbs become three tools in
`scientist_tools.TOOLS`.

**What is not in the environment.** No job queue, no sandbox, no page, no typed tool surface, no second model, no
gradient path. See section x.

---

## ii. What an edit is

### Representation

An edit is a list of **terms**. A term names one linear by its dotted module path inside the text model (for
example `layers.50.mlp.down_proj`) and holds a delta in one of three kinds:

| Kind | Stored | Meaning | Covers |
|---|---|---|---|
| `cols` | column indices `[k]`, vectors `[k, n_out]` float32 | add vector j to column `idx[j]` | set, zero, scale or negate a neuron's write column; subtract a direction from named columns (the form the Qwen edit took); the vault's UPDATE |
| `rows` | row indices `[k]`, vectors `[k, n_in]` float32 | add vector j to row `idx[j]` | change what a neuron reads (gate_proj, up_proj rows); scale a neuron by its up row |
| `lowrank` | `A [r, n_in]`, `B [n_out, r]` float32, r at most 64 | add `B @ A` | rank-one and small-rank updates on any linear; a head's output block on o_proj |

Every term is an **additive delta against the shipped base weight**. "Set column i to v" is lowered at make time
to `v - W[:, i]`, read once from the base through `gguf_weights.dense_cols`. Scaling by f is `(f - 1) * W[:, i]`.
There is no "replace" at apply time, so there is nothing to order and nothing to undo.

Deliberately not representable in the first increments: scattered sparse entries over thousands of columns,
embedding rows (the embedding is tied to the unembedding, 2.625 GiB, and has no hook form yet), norm weights and
layer scalars. This costs something real: on IWIBench, whose target was a Gemma 3 4B, all 273 of one leading
editor's successful writes went to the tied embedding matrix (research digest, wsp_reception corrections). I
leave it out anyway; the kernel must be proved on the feed-forward linears first.

The representation takes any of the 410 linears. **Use on the card is gated per tensor class by the verify
script**: `down_proj` columns and `gate_proj` / `up_proj` rows in the first increment; an attention term is refused
on the 31B until its verify section has passed there. Whether an overlay or a delta on an attention projection
has ever run on the 31B is not established (code digest, edit_paths).

### How it is applied

`with weight_edits.applied(lm, arms, rows_of):` registers one forward hook on each named linear. For an arm whose
batch rows are R the hook computes, in float32,

- `cols`: `d = x[R][..., idx] @ vectors`
- `rows`: `d[..., idx] = x[R] @ vectors.T`
- `lowrank`: `d = (x[R] @ A.T) @ B.T`

and writes `y[R] = (y[R].float() + d).to(y.dtype)`: summed in float32, rounded once, as the CPU kernel does it.
Rows that belong to no arm are not touched, so the base row's bits are those of a forward with no hook at all.
Hooks are removed in `finally`. It acts at every position, the start mark included: it is a weight edit, not a
push, and has no `spare_bos`.

### Composed, rolled back

`stack(e1, e2)` is the union of the terms. Deltas add. Two terms that "set" the same column or row are refused;
two scalings of one column add as `f1 + f2 - 1`, which is exactly how llama.cpp adds two adapters
(scripts/export_lora.py:50-55), so the kernel's composition and the export's composition are the same rule.
Terms are applied in a canonical order (by tensor name, then by term hash) so the float sum is the same every
time. Each stack records its parents' hashes and, per touched column or row, the delta's length as a share of the
base column's or row's length, so growth under stacking is visible.

There is no rollback because there is nothing to roll back. The model is never left edited. The one exception
is the folded cross-check below, which snapshots and restores exactly the dict entries it sets (the pattern of
`export_lora.vault_kept`, scripts/export_lora.py:383-406), on all 410 linears, not only the three the vault's
ROLLBACK clears.

### Hashed, saved, reloaded

- **Hash**: sha256 over the base weights' id (the weights card's `files_sha256`; the checkpoint's id is
  `e0a93173c31ed2ec`, read from the index metas on 14 November 2026) and, for each term in canonical order, its
  tensor name, kind, shapes, sorted indices and little-endian float32 bytes. Notes and times are not hashed.
- **Saved**: one `.npz` per edit under `atlas_data/edits/<first 12 of hash>.npz` (git-ignored; an edit that
  holds scaled columns of Gemma's weights is a derivative of the weights and the repository redistributes none).
  The file's meta holds the format version, the full weights card, the arithmetic's name, the code commit, who
  made it, parents, and a note of at most 200 words saying what the edit is meant to do, written before it is
  read.
- **Reloaded**: `load(path)` refuses when the base weights' id differs from the loaded runtime's, in plain words.

### Exported (third increment)

`to_lora(edit, path)` hands each term to `export_lora.write_lora` (scripts/export_lora.py:501), which already
takes any tensor name at any rank as `(A, B)` and writes provenance under `theodolite.*`. A `cols` term becomes
one-hot `A` and the vectors as `B`, which is what the exporter writes for a vault column today. No model reload:
the edit already holds the delta, where `export_lora.py --vault` has to load the whole model on the CPU to
recompute it. Two limits carried from the digest: llama.cpp is known to apply an adapter to `ffn_down` and
`attn_output` only (other tensors not established), and a GGUF of the 31B differs from the W4A16 checkpoint in
all 410 linears, so the exported adapter acts on a different object and must be measured there, not assumed.

### Where it stands with the 4-bit grid

Off it, and it says so. An edit is a float32 delta beside the packed 4-bit base. The kernel never touches
`weight_packed` or `weight_scale` and never requantises. Three things follow, stated on every record:

1. What was measured is **base + delta in delta-term arithmetic**. A column folded into the bf16 weight rounds
   differently; pilot P4 measures by how much.
2. A requantised file would be a third object. For a `down_proj` column it cannot be made cleanly at all: each
   of the column's 5,376 entries shares its scale with 31 other neurons, where a `gate_proj` or `up_proj` row owns
   its 168 scale groups (code digest, gguf_export; the reader's derivation).
3. In the third increment a `grid_report(edit)` reads the group scales and states, for `cols` and `rows` terms,
   what share of the delta's entries exceed half a 4-bit step. It is a prediction, marked inferred. No public
   work has measured an edit surviving requantisation of a QAT checkpoint (research digest).

### How a reading under an edit records it

Every line of the level book carries: the base weights card, the edit's hash, the arithmetic
(`delta-term, float32 sum, one rounding`), the unpacker, the card, rows and positions of the forward, the start
layer, the code commit, the prompt set's hash and which half was read. Where a model id is needed it is
`<base model id>+edit:<first 12 of hash>`, following the `+int4mm` precedent. Nothing built for the unedited
model (an index, a census, a stats unit) is ever read as the edited model's, because the edited model exists
only for the length of one forward.

### The folded cross-check

For `cols` and `rows` terms the kernel can also lay the edit on through the existing overlay dicts
(`_memit_col_overlays`, `_memit_row_overlays`), run base and edited as two forwards of one shape, and compare
with the delta-term reading. This is the forward the model itself would run if the edit were written into the
weight. It is slower (two forwards, a weight clone per edited module per forward, cost not measured anywhere)
and pays the batch-shape floor, so it is a check, not the working path.

---

## iii. How the agent reads

**No language model can work with 29.3 billion 4-bit weights, and this design does not pretend otherwise.** Raw
weights as text have been shown readable only up to about 172,000 weights (research digest; one study). What a
model can work with is three things, and the kernel hands it exactly those:

1. **Addresses with facts.** A neuron is `(layer, neuron)`. For all 1,290,240 of them the atlas holds stored
   passages with token ids (unpushed evidence, every layer), the 24 strongest links each way with sizes, lens
   words, a steer-lens entry; for 12 layers a whole census against 2,048 controls. All of it describes the
   unedited checkpoint, which is what an edit is a delta against. Read through `AtlasIndex` (numpy only,
   backend/atlas_index.py:216) and the census's `RowsIndex` / `UnitsTable` directly in the job, no server needed.
2. **Vectors as statistics, never as numbers.** A column is 5,376 numbers; the agent sees its length, its cosine
   with a named direction, its write ceiling, its top projections. Read through `atlas_ops.Weights`
   (`down_cols`, `gate_rows`, `up_rows`, `write_dirs`, `write_ceiling`; backend/atlas_ops.py:128-209) and
   `neuron_wiring` / `pair_products`.
3. **Measured effects.** Activations on texts it writes (`labkit.Lab.forward_resid`, `atlas_ops.read_neurons`)
   and the paired reading.

The agent's handle on a weight is its address plus a delta it builds by function from other addresses and from
captured activations. The tensors stay in the job; only hashes and summaries cross into the agent's context.

**New read code in the first increment: none.** Everything above exists and is importable in a lab daemon job.
The second increment adds two helpers, because they are the first two steps of every edit that has worked in the
literature and the atlas does not hold them for an arbitrary direction:

- `direction(set_a, set_b, layer)`: the mean stream difference between two text sets at a layer (one capture).
- `writers(layer, u, k)`: the k neurons whose write columns lie most along `u` (one layer's `down_proj` unpacked,
  0.24 to 0.46 ms compiled, measured, date not given in the digest, and one product).

**Trust travels with the read.** A `trust(layer)` card (second increment) returns the measured bands so they
cannot be forgotten: lens words only at layers 40 to 57 under the label gate; steer lens agreeing at 39 to 52;
push doses calibrated at 32 to 52; link sizes real, link signs below chance at an ordinary word-piece (0.336,
measured). And one rule from the digest that the kernel enforces by construction: the lens label of an edited
neuron is never evidence that an edit worked. The kernel reads the model's own odds and nothing through a lens.
Its readings are lens-free, so the base-lens fault does not touch them.

---

## iv. How a hypothesis is tested, fast

### The one measurement

`levelling.read(rt, set, arms, metric)`:

1. **The set.** A prompt set is a list of texts as fixed token ids, each with a read position and a metric:
   a log-odds between two groups of token ids (as the study's F/M pronoun measure), a target token's
   log-probability, or KL at chosen positions. The set has a hash and a fixed split into a working half and a
   kept-back half (section v).
2. **Capture, once per set.** One forward per text at one row with a hook on every layer boundary stores the
   base stream entering each layer. About 69 MB per boundary for 100 texts of 64 pieces in bf16 (research
   digest's arithmetic). Kept in process memory for the session, keyed by set hash and weights id.
3. **One forward per text**, from the lowest edited layer L, with the text's cached stream copied to every row:
   row 0 the base, row 1 a **zero edit** (a term of all zeros at the same site), then the candidate edits, then
   the matched random edits. No padding: every row is the same text. Hooks as in section ii.
4. **Read** the final-normed states at the read positions, fp32 softcapped logits through
   `causal_check.logits_fp32`, and from them the metric for every row. KL is always `KL(base || edited)` and the
   record says so (the instrument's three existing KLs do not share a direction).
5. **Return and write** a per-item table and a summary per arm: mean paired difference against the base row,
   its standard error, a paired sign-flip p over texts, the arm's rank among the controls, KL and top-1
   agreement at the read position, and `delivered`: the relative change of the stream leaving the edited layer,
   read by a hook, which says how much of the edit survived the post-feed-forward norm.

The zero-edit row is the run's closing error. Its difference from the base row must be exactly 0.0; if it is
not, the reading is refused and nothing is written but the refusal.

### What it costs

Measured figures used (compiled unpacker, RTX 5090): a decode step 0.091 s at 1 row, 0.103 s at 32, 0.109 s at
64 (28 October 2026); a census question's prefill 0.10 s at 1 row, 0.26 s at 32, 0.45 s at 64 (28 October 2026);
516 short rows through 14 layers in about 0.85 s and through 54 layers in about 3.5 s (29 October 2026, steer
lens; the per-forward division is mine). A step costs nearly the same from 1 to 64 rows because the time is the
unpacking, so rows are close to free up to memory.

| Reading | Shape | Cost | Standing |
|---|---|---|---|
| Capture | 24 texts, 1 row each, all 60 boundaries | about 2.4 s, once per set | my arithmetic: 24 x 0.10 s |
| The known-answer test (section ix, P5) | 12 texts x 32 rows, from layer 50 | about 0.5 s | my arithmetic: 12 x (10/60) x 0.26 s |
| A screen at a late layer | 24 texts x 32 rows, from layer 45 | 1.5 to 2 s | my arithmetic, two ways: (15/60) x 0.26 s a text, and 0.118 ms a row-layer from the steer-lens timing |
| A screen at an early layer | 24 texts x 32 rows, from layer 5 | 6 to 7 s | my arithmetic, same two ways |
| Neutral-text panel | 32 windows x 64 pieces x 4 rows, from layer 45 / layer 5 | about 2 s / about 8 s | weak inference from the reference unpacker's whole-sequence figures (1.1 s at 512 pieces against 0.706 s at one, 18 and 28 October 2026); pilot P3 replaces it |
| Heard answers (second increment) | 8 questions x 4 arms, 48 pieces, whole model | about 5.2 s | my arithmetic: 0.26 + 48 x 0.103 s; consistent with the measured 8.3 s for 3 rows of 80 pieces through the server (11 November 2026) |
| For contrast: the vault's heard edit | 1 question, 2 answers of 12 pieces | 12.6 s | measured, date and unpacker not stated |
| For contrast: a registered census trial | 1,022 neurons, 49,110 answers | 68 to 88 minutes | measured, 13 November 2026 |

None of the kernel's own figures is measured. Pilot P3 measures them before a line of the kernel is committed.

### How many candidates share a forward

Thirty-two rows by default: base, zero edit, and thirty arms split between candidates and controls. The limit is
memory, not time. The lab daemon holds the model and both lenses at 21.2 GiB (measured 18 and 27 October 2026),
the desktop holds 3.4 to 5.4 GiB, so about 5.6 GiB is free; a cached census batch of 32 rows at 64 new pieces
peaks 2.4 to 3.1 GiB above the model (measured 28 October 2026) and an uncached partial forward should need
less (inferred; P3). Candidates may edit different layers: the forward starts at the lowest.

### What is cached, and what is not

Cached: the base stream per set and layer; the base weights' columns and rows an edit was built from; the
controls' random draws by seed. Not cached: base results. The base row rides in every forward for nothing, and
pairing by construction is worth more than a memo that would have to be keyed on batch shape.

### What I give up for simplicity

One text per forward and no packing of texts leaves perhaps a factor of two to five of speed unused. No writing
from layer L upward with cached lower layers (no such function exists, and once an edited row's text departs
from the base the lower layers must run anyway). Speed-first would take both. Kernel-first does not: a reading
that is already one to seven seconds does not need them, and each adds a way for rows to differ that is not the
edit.

---

## v. What stops the agent fooling itself and the owner

The kernel carries the guards that cost nearly nothing and cannot be skipped because they are inside the one
function that measures. Guards that need a second process or a second OS user are left to later, and I say what
that costs.

**In the kernel, from the first increment**

1. **Pairing by construction.** Base, zero edit, edit and controls are rows of one forward on one cached stream.
   The numerical floor between them is nought and the zero-edit row proves it on every forward.
2. **Matched random edits in the same forward, by default.** For each candidate: edits of the same kind, at the
   same site, with the same delta length, in a random direction. Two kinds, as the census has them: `isotropic`
   (a random vector in the raw column space, which the post-feed-forward norm then carries as the census's
   `unit(w_post * r)` control is carried) and `columns` (a random mix of the layer's own write columns). The
   study found a random stream direction of about 1% of the stream's length collapses the model through the
   quiet dimensions (E6 report, October 2026), so which null is fair for a weight delta is not assumed: pilot
   P6 measures it. The result states the candidate's rank among its controls and that, with C controls, about
   one candidate in C + 1 tops them all by chance.
3. **A random-site control**, from the second increment: the same delta on other neurons of the layer.
4. **Counting looks.** The level book counts how many distinct edits have been read against each set's working
   half, and every result carries that count ("the 37th edit read against this set"). Within one call,
   Benjamini-Hochberg over the candidates. Across calls there is no correction, only the count, stated.
5. **The level book is written by the reading.** Every call leaves a line whether or not the caller reports it:
   edit hashes, set hash, the full per-item table, GPU seconds, the caller's label. Hash-chained. A null or a
   refused reading is a line like any other.
6. **Side effects, measured against the same controls.** `delivered` on every reading. From the second
   increment a neutral-text panel: KL and top-1 agreement at every position over 32 stored corpus windows
   (even-numbered for working, odd-numbered kept back, the fold the pattern tests already use), for the edit and
   for its controls, so the statement is "this edit disturbed unrelated text no more than a random edit of its
   size", not "KL was small". Window ids only are recorded; the corpus text stays in `atlas_data`.

**From the second increment**

7. **A kept-back half and one look at it.** `register(edit, set, metric, rule, said_beforehand)` writes a hashed
   record into the level book. `confirm(id)` reads the kept-back half once, with at least 99 controls, applies the
   rule as written (0.599 against 0.60 is a fail, as the house applies it), writes `held` or `not held`, and
   refuses a second call. Matched pairs stay in one half.
8. **Heard, not only teacher-forced.** Nothing is called `held` on next-piece odds alone. The second increment
   writes greedy answers for base, edit and controls in one batch through `atlas_ops.write_rows` with the
   kernel's hooks around it, and reports where each parts from the base **against where the controls part**.
   Within one forward the writer's floor is not the yardstick; the matched random edit is.
9. **The cheaper knob in the same forward.** An arm may be an `atlas_ops` push instead of an edit (the hooks
   compose). A column scaling beside the `mul` push it should equal, or a direction removal beside the steering
   vector it came from, are then rows of one forward. The Qwen headline edit was nearly matched by a settings
   change (research digest); the kernel makes that comparison cost one row.

**What a result is allowed to claim.** Four standings and no others:

- `screened`: read on the working half; a lead.
- `held` / `not held`: registered beforehand, read once on the kept-back half, heard.
- `kept the base`: no edit survived; an ordinary outcome (6 of 16 trajectories in Hu et al.).

And one sentence form, filled from the record: "On *set* (*n* texts), adding *edit hash* to *base id* moved
*metric* by *mean* (SE *s*) against the base row of the same forward; *C* matched random edits moved it between
*a* and *b*; delta-term arithmetic, compiled unpacker, *R* rows; teacher-forced / heard." Never "the neuron
does X", never a benchmark figure, never a statement about a requantised or exported model unless that object
was the one measured.

**What the house rule still requires.** Anything that would be written into README or ROADMAP as a finding
needs the house's own registration: a hashed folder with rules, runner, reader and inputs, committed before the
run. The kernel's `register` is the light form for work inside a session; it does not replace the heavy one.

**What this leaves open, plainly.** In the first three increments the kept-back half is kept back by
convention. A driver with code execution in the lab daemon could read it; the level book would show the read,
nothing would prevent it. Scorer and held-out data under a separate OS user is the mechanical guard the
literature supports, and I leave it out until there is an unattended driver to guard against. Validity-first
would not accept that. The cost of my choice: until then a `held` is honest against mistakes, not against a
driver that cheats.

**The reviewer's question about the core measurement**, answered in advance. The core measurement is a paired
difference in fp32 next-piece log-odds between rows of one forward. Its known weak points are: teacher-forced
odds are not behaviour (item 8); the delta term is not the folded weight (P4); a random edit of equal length may
not be an equal disturbance (P6); small sets have little power (the standard error is on every line, and the
motif test's floor of six pairs is kept); and the base row of a partial forward mixes two batch shapes, one for
the capture below L and one above (P2).

---

## vi. The agent interface

**Three verbs, as plain Python, and nothing else.**

```
make    weight_edits.cols(...) / rows(...) / lowrank(...) / stack(...)   -> an Edit (hash, summary)
read    levelling.read(rt, set, arms, metric)                           -> a table, and a line in the level book
look    level_book.lines(...) / weight_edits.load(...)                  -> what has been read and made so far
```

**Today.** The driver is a frontier model in Claude Code at the owner's terminal, attended, under the owner's
permission system, submitting job files to the lab daemon with `labclient.py`. This is the route by which
agents ran the study's 314 jobs on 16 October 2026 (282 ok, median 71 s a job; measured). The job imports the
kernel and calls the three verbs; reads of the atlas and of weights are ordinary Python against `rt`,
`atlas_ops.Weights` and `AtlasIndex`. Nothing has to be built for the driver before the kernel exists.

**A tools-only driver** (an endpoint model, or the scientist at its default reach, which has no shell) gets the
same three verbs as three tools in the fourth increment, added once to `scientist_tools.TOOLS` so that Claude
Code, Codex and any Chat Completions endpoint receive them through the one dispatcher. Tensors stay
server-side; a tool reply is the summary and the hash.

**A stronger future driver** plugs into the same three verbs. What it will want is more reach (unattended
runs, more tensor classes, an attribution backward), and each of those is a guard to add around the kernel, not
a change inside it. The edit format and the level book carry a version field. The kernel takes its shapes from
the modules it is given and holds no 31B constant, unlike the vault.

**Code execution or typed tools?** Both, in that order. Code first, because the only thing that needs proving
is the kernel, and an attended code driver needs nothing built. Typed tools second, as projections of the same
functions.

**Budgets.** The reading times its own forwards and writes GPU seconds on every line. From the second
increment a session has a cap in GPU seconds in a small file; the reading refuses when it is spent and says
what is left on every reply. Seconds, not a count of passes: on this card one pass may hold 12 rows or 500 at a
cost within a small factor of each other, so a count of passes says little about what was spent.

**State across sessions.** Files only: the level book, the saved edits, the prompt sets, and whatever notes
the driver keeps in its work folder. A new session starts by reading the level book. Notes are the driver's
claims; the level book is what was measured; the two are never merged.

**Deliberately thin.** No loop, no planner, no handoff machinery, no context management, no prompt beyond a
one-page method note (state the hypothesis, read it paired, compare with the controls, register before the
kept-back half). Scaffolding tuned to one driver has gone stale within a model generation elsewhere (research
digest); the driver's own harness does that work.

---

## vii. Safety and scope

**Containment, by increment.**

- Increments one to three: **attended only.** The driver is the owner's own Claude Code session; the lab daemon
  runs submitted Python as the user with no sandbox, loopback only, with a per-launch token, exactly as during
  the study. I add no containment because I add no new exposure: no unattended driver, no remote driver, no
  scientist reach to the daemon.
- Increment four: the scientist gets tools, not code. Its reach rules stand (no shell, files only in its work
  folder, no credentials). A saved edit made by the scientist is a proposal file that nothing loads until a
  person keeps it, as names and notes are.
- **An unattended driver with code and the GPU is out of scope** until a contained runner exists (a separate OS
  user, no network, weights mounted read-only, scorer and kept-back data out of reach). That is a later piece
  of work and the owner's decision.

**What the harness will not do.**

- Write, patch or requantise a checkpoint or a GGUF.
- Leave an edit in force between calls, or build an index, a census or a stats unit on an edited model.
- Edit embeddings, norms or layer scalars (first increments).
- Run while a vault edit is in force.
- Publish, push or upload anything.
- Ship a trigger-to-payload helper. The research digest records a weight-edit backdoor shown on the bf16
  Gemma-4-31B-it by edits of about one feed-forward neuron per payload token (Lazarus, 11 November 2026; best
  cell, recorded trajectories). The kernel's `cols` and `rows` terms are that primitive. They are also the
  vault's primitive and every abliteration tool's, so the kernel adds little uplift; it adds no recipe.

**What happens to edited weights.** They never exist as weights. An edit is a small delta file under the
git-ignored `atlas_data/edits/`, named as derived from Gemma, carrying the base id and its own hash. Export to
a llama.cpp adapter is the owner's command, never a driver's, and the adapter carries a manifest: base id, edit
hash, the level-book lines that read it, a statement that the weights it applies to are modified, and the
Apache 2.0 text. The research digest reads Gemma 4 as under plain Apache 2.0 (model cards and licence pages);
no document of the project states a policy on distributing edited weights or adapters, and that policy is the
owner's to write before the first export leaves the machine.

---

## viii. The phased path

Each increment is useful if the work stops there.

| | Delivers | Proves | Effort (one person) |
|---|---|---|---|
| **0. Pilots** | One lab daemon job file in a scratch folder, about 100 lines, with the hook written inline; a short dated note of results. Nothing committed to `backend/`. | Whether the design's three premises hold on the card (section ix, P1 to P5). | 1 to 2 days, under an hour of card time by my arithmetic. |
| **1. The kernel** | `backend/weight_edits.py` (cols, rows, lowrank, stack, hash, save, load, applied, controls), `backend/levelling.py` (set, capture, read), `backend/level_book.py`, `scripts/check_weight_edits.py` on the stand-in, `scripts/verify_weight_edits.py` on the card. Run from the lab daemon. | That one hypothesis about the 31B is tested in seconds with its controls in the same forward: the 16 October 2026 result reproduced, and the question "is it neuron 5313 alone?" answered in the same call. | 5 to 7 days. |
| **2. Honest enough to keep a result** | Heard answers through `write_rows`; the neutral-text panel; the kept-back half with `register` and `confirm`; the random-site control; push arms in the same forward; `direction` and `writers`; the `trust` card; the GPU-seconds cap; the lab daemon started without the lens for this work. | That a `held` means something: one edit taken from screen to registered confirmation and one expected null taken the same way. | About 1 week. |
| **3. Out of the daemon** | Two server routes under `_GPU_LOCK` (`POST /api/edits/read`, `GET /api/edits/book`); `to_lora` with a check through llama.cpp on the stand-in in the manner of `check_lora_export.py`; `grid_report`; verify sections for attention terms. | That the kernel shares the card with the instrument without a stale flag, and that an edit leaves the machine only with its record. | About 1 week. |
| **4. A driver surface** | Three scientist tools; saved edits as proposals; a plain table of the level book on the vault page; a Lexicon entry, a ROADMAP section, REFERENCES entries. | That a tools-only driver can run make, read, look. | 1 to 2 weeks. |

Not scheduled: a standing edit that a person can chat with inside the instrument, a contained unattended
runner, an attribution backward, a solver for "which columns and how much", embedding and norm edits, a
requantiser, a smaller rehearsal model. Each is a caller of the kernel or a guard around it.

---

## ix. Pilots to run on the card before committing

In this order. Each names the result that would change the design. P1 to P5 fit in one job.

**P1. Does the run close?** With hooks installed and zero terms on `layers.50.mlp.down_proj`: (a) the base
row's final states equal, by bits, the same row in a forward of the same shape with no hook; (b) two base rows
are identical; (c) the zero-edit row is identical to the base row. Repeat at 8 and 32 rows and from a partial
forward. *If (a) fails* the hook perturbs the base path and must be installed otherwise. *If (b) or (c) fails*
the premise "rows of one forward differ by the edit alone" does not hold for this path; every forward then needs
several base rows and a measured floor, and the standings of section v need a floor term.

**P2. Is a partial forward the whole forward?** For 12 texts and one real edit, compare the paired difference
(edit minus base) from `forward_from_layer` on a captured stream with the same difference from a whole forward
of the same rows. `forward_from_layer` has been compared with a whole forward only on the CPU stand-in, at
atol 1e-5 (code digest). *If the two paired differences disagree by more than the difference between two batch
shapes*, partial forwards are dropped: every reading costs a whole forward (up to six times more at late
layers), the early-layer figures of section iv become the only figures, and the capture cache goes.

**P3. Is it seconds?** A timing and peak-memory table for the partial forward with hooks: start layer in
{0, 20, 40, 50, 58}, rows in {2, 8, 32, 64}, pieces in {16, 64, 128, 256}, compiled unpacker, with the
unpacker's `.fell_back` counter read before and after. *Gate:* 24 texts x 32 rows from layer 40 in under 10 s
with no fall-back. *If it fails* find out why before anything else (a fall-back to the reference unpacker costs
about seven times). The table also sets the default rows and whether the lens must be freed.

**P4. Delta term against folded weight.** For the vault's example edit (layer 59, neuron 7830, 'six', strength
3.0; first-step KL 5.65 on the spider question, measured 29 October 2026 on the reference unpacker) and for the
ten-neuron deletion at layer 50: the delta-term row, a forward with the column laid on through the overlay
dict, and a `mul 0` push with the start mark not spared, all at one batch shape. Report KL and metric
differences between the three, beside the difference between two batch shapes of the unedited model. *If
delta-term and folded differ by more than batch shapes do*, the delta term is named a screening arithmetic,
`confirm` runs folded (two forwards of one shape), and low-rank terms, which have no folded path, cannot reach
`held` until one is built.

**P5. The known answer.** Zero the ten layer-50 neurons of the study's list (5313, 8801, 266, 121, 11503,
11724, 21151, 4165, 549, 2070) on its 12 female-name contexts with its measure (log-sum-exp of the female
pronouns' logits minus the male's). Stored result, 16 October 2026: -1.80 mean, against -0.02 for ten random
neurons (experiments/results/E6_tool_audit_04_vault.partial.json). In the same forwards: 5313 alone, the other
nine alone, and 28 random tens. The report says the 'he' list gives the same effect "because it shares the
bipolar neuron 5313", so the prediction, said beforehand, is that 5313 alone carries nearly all of it. The
tolerance for "reproduced" is fixed before the run. *If the ten do not reproduce* there is a fault in the
kernel or in my reading of the measure; stop. *If they do*, the first increment's claim is proved in its
smallest form: a hypothesis about the 31B (is it one neuron?) answered in about half a second of forwards.

**P6. Which random edit is a fair null?** For 64 real neurons at each of layers 20, 36, 45 and 55: a column
delta of a fixed share of the column's length in the neuron's own direction, in an `isotropic` direction and in
a `columns` direction. Compare `delivered` and neutral-text KL. *If isotropic nulls are several times louder*,
the default null is `columns`. *If both differ from real directions by a large factor*, nulls are matched on
`delivered` instead of on weight length, which changes the controls' constructor.

**P7. Is one column ever heard?** The vault's own note says an edit that is heard is rare (the most active
neurons at pages 41, 46, 51, 56 and 59 were not heard at strengths 3 or 30). For 20 census standouts at each of
layers 36, 45 and 50 and 20 random neurons: scale the column by 0, 2 and 4 and read KL on the neuron's own
stored passages and on neutral text. *If single columns are mostly below their controls*, the kernel's default
unit becomes a set of columns and a direction removed from its writers, `writers` and `direction` move into the
first increment, and the default set size grows.

**P8. Does a heard answer close?** (Before the second increment.) Two base rows and a zero-edit row written
for 48 pieces with the hooks installed stay identical to the last piece.

---

## x. Risks, what is left out, and the owner's decisions

### Risks

1. **The kernel works and finds little.** On a healthy model edits have mostly removed obstacles or shifted
   defaults; held-out gains on a strong 9B base were at most about three points, and where the base was near its
   ceiling there was no headroom (research digest, with its corrections). The vault's own note says a heard
   one-column edit is rare. The kernel should be judged on how fast and how honestly it tests a hypothesis.
   `kept the base` will be the commonest line in the level book.
2. **Three arithmetics, three objects.** Delta term on the checkpoint; the same edit folded into bf16; the
   adapter on a GGUF whose 410 linears all differ from the checkpoint's. A claim belongs to the one that was
   measured. P4 sizes the first gap; the third is unmeasured until the third increment.
3. **Teacher-forced odds flatter an edit.** In the model-editing literature realistic free-generation scoring
   cut reported success from 96.8% to 38.5%, teacher forcing being the most severe of several causes (research
   digest). Until the second increment every result is `screened` and no more.
4. **The null is wrong.** The quiet dimensions make a random direction far louder than a neuron's write in
   stream space. If the weight-space controls are unfair in either direction, rank among controls means little.
   P6 is there for this, and it could force a different matching rule.
5. **The instrument is down during sessions** in the first two increments, and the card is already wanted for
   13 to 15 hours at a time by census layers and for volunteer spot-checks. Sessions have to be scheduled.
6. **No containment for code** in the first three increments. Acceptable only because attended.
7. **The kept-back half is not sealed** until an unattended driver forces the issue.
8. **Small sets.** Twelve contexts resolve a 1.8 log-odds effect; they will not resolve the effects a careful
   edit on a healthy model is likely to have. The cost table says larger sets are affordable; someone has to
   write them, and prompt sets written after seeing results are a forking path the level book can count but not
   close.
9. **A convention is already broken underfoot.** `backend/engine_runtime.py` no longer matches R4's sha256 pin
   since two commits of 12 November 2026 (code digest, computed not run), while documents still say it is
   pinned. The kernel does not touch the file, but the rule it obeys ("new behaviour goes beside, not in") rests
   on a pin that should be settled.

### Deliberately left out

A typed tool surface and any page (until the fourth increment); a job queue, cancellation, progress; a sandbox
and unattended runs; sealed scoring under another OS user; gradients and attribution; a solver for choosing
columns; a requantiser and any whole-model export; embedding, norm and layer-scalar edits; attention edits on
the card until verified; a standing edit inside the instrument; rebuilding any index on an edited model; decoy
or shuffled-weight probes of the driver; capability batteries; a second model family or a runtime protocol; a
trained small rehearsal model (a Gemma 4 12B GGUF is on the machine and has never been built or run); packing
texts into shared forwards; multi-agent anything.

What that costs, against the other priorities: no driver other than an attended Claude Code session can use it
for weeks; a result cannot be more than a lead until the second increment and cannot be trusted against a
cheating driver at all; a factor of two to five in speed is left unused; and nothing here makes another model
the object.

### The owner's decisions

1. The name, and whether "weight-space programming" is the term the documents set beside it.
2. Whether to run the pilots at all, and when the card can be spared from the census for them.
3. Whether sessions may take the card from the server (the lab daemon, increments one and two), or whether the
   routes of increment three come first.
4. What may be called `held` in the documents, and whether a kernel `confirm` ever suffices without the house's
   hashed registration.
5. The policy on edited artefacts: whether adapters are exported at all, and whether any is ever published.
6. Whether the vault's UPDATE is later rebuilt on the kernel (per-edit undo, hash and save would come for
   nothing) or left as it is.
7. Gradient access: none in this design. Whether a single attribution backward is later allowed is a choice
   the published work has not settled.
8. Unattended drivers: not until containment exists, and then which models at which reach.
9. The R4 pin on `engine_runtime.py`: restore it, re-pin it, or retire it.

---

## Names

In keeping with the house vocabulary:

1. **Levelling** (recommended). In a level run the instrument measures only differences: a back-sight on the
   known point and a fore-sight on the new one from the same set-up, both entered in the **level book**, and a
   run that closes back on its start shows its error. Base row, edited row, one forward, the level book, the
   zero edit that must close to nought: the kernel is this and nothing more. An edit stays "an edit", the word
   the vault already uses.
2. **Collation.** The manuscript word for setting two states of a text side by side, line against line, to find
   where they part; an edit would be an *emendation*, held in the apparatus and not in the text.
3. **Traverse.** A run of sightings out from a fixed station and closed back upon it; fits the round of
   hypothesis, edit and reading, less exactly the paired measurement.

("Field book" is taken: it is the folio's per-face book in `field_book.json`.)
