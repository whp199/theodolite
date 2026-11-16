# A weight-space environment for Theodolite: the future-proof design

Written 14 November 2026. Design only: nothing was run, loaded, edited or committed. Every figure below carries its date and says whether it was measured (by the project, as the digests record) or is my arithmetic. Layers are counted from 0 unless a page is named.

Lens: future-proof first. Where this conflicts with speed, with the shortest path to a first agent session, or with the vault's present way of holding an edit, I choose the lens and say what it costs.

## o. The position

The environment should be four standards and one thin kernel. The standards are what outlive the driver model, the object and the card. The kernel is the least code that makes them true on the 31B today.

The four standards:

1. **The datum** (the literature's base checkpoint): an object protocol. A model is a table of named tensors with roles, a block formula, a storage description and a content identity. Gemma-4-31B W4A16, a Gemma 4 GGUF, the six-layer stand-in and the Gemma 4 12B on disk are four datums behind one protocol.
2. **Errata** (the literature's edit program, patch, delta or adapter): a portable edit format. A readable source program plus a lowered payload, hash-pinned to a datum, diffable, exportable as a LoRA GGUF and importable back.
3. **The docket and the return** (task pack and result pack): what an agent was asked, under what rules and budgets, and what came back, in a form another harness or benchmark can read.
4. **The ledger and its page**: an append-only record of every forecast, proof and verdict, and a page in the instrument where a person watches, learns from and audits an agent at work, and can replay a past session without the card.

The one decision everything else follows from: **in this environment an edit is an argument of a measurement, never the state of the server.** An erratum rides one row of a batch as an added term on the outputs of the linears it touches. The row beside it carries no erratum and is the unedited model in the same forward. The model the server holds is always the datum. That removes, at a stroke, the staleness problem (every atlas index keeps describing the model that is loaded), the rollback problem (there is nothing to roll back), the identity problem (each row of each reading names its own erratum) and the same-forward problem (the digest's gap: "a weight edit cannot be compared with the unedited model inside one forward").

What this costs, stated once:

- The canonical arithmetic of an edit becomes "base product plus a float32 term, rounded once", which is an adapter's arithmetic, not the vault's "write the column into the unpacked weight". The two round differently. Until a pilot says how differently (x, P2), they are two named instruments.
- The first increment contains no agent. A driver-first design would have an agent editing through the vault in days; this one has an agent in the second increment.
- The scientist's 35 existing tools do not hear an erratum. Only the environment's own proofs do. That is deliberate, and it means a few readers are written again in a multi-row form.
- I refuse the speed designer's cache of base-row results. One row in every forward is spent on the base (3% of a batch of 32). It is the house's own honesty device and I will not trade it.

### Standard now, and unbuilt until needed

Fixed now, versioned, and each with a line in the conformance check, because changing them later breaks every stored result:

- the datum protocol and the role names;
- the errata manifest, the normal form and the three hashes;
- the three realisations' names, and the rule that a reading names its own;
- the per-row state on every reading;
- the nine verbs of the wire;
- the ledger entry, the verdict vocabulary, the docket and the return.

Named in the formats and left unbuilt until the stated need arrives:

| Unbuilt | Built when |
|---|---|
| Appliers for the embedding, norms and layer scalars | a hypothesis needs one, or P4 shows feed-forward columns are too quiet |
| Grid ops and the bake | P7 shows something survives the grid |
| An edited model as server state | a person wants to walk the whole instrument under an erratum |
| The sandboxed kernel | the typed tools have been used in earnest (increment 4) |
| A headless host for unattended runs | the owner approves one |
| A plain-text protocol for models without tool calling | an open-weights driver is actually tried |
| Importers for other people's patch formats | the first outside edit worth re-measuring |
| A builder for another model family | another object is chosen |
| Writing from a layer upward on cached lower layers | P6 shows GPU time, not driver time, binds |
| A separate OS user for kept-back items | a session is to run unattended or at a wider reach |

## i. What the environment is

### The process

The server holds the model, as now. The environment is a library hosted inside the server process, with routes under `/api/table/*` bound to the one GPU lock the way the atlas's are (`atlas_routes.bind(get_app_state, _GPU_LOCK)`, backend/server.py:1660; `on_gpu`, backend/atlas.py:148). It is not a second process and not the lab daemon.

Why the server and not the lab daemon: only one process may hold the 31B on the card; the person must be able to watch; the atlas's readers and the scientist's tools are HTTP proxies onto the server; the server's default unpacker is the compiled one (0.091 s a step at one row against 0.706 s on the reference the daemon defaults to; measured 28 October 2026); the server keeps answering during GPU work because it runs it in a worker thread, where the daemon evaluates job code on its event loop, has no cancel and no time limit, and leaves whatever a job changed in place for the next job (experiments/lab/labd.py:198-219, read by the digest). The daemon's last recorded job is dated 17 October 2026. I take its patterns and none of its process: the loopback-and-token guard (labd.py:98-115), the server-side result file that survives a client timeout (labd.py:228-233), and later labkit's gradient linears as a declared evidence level.

Because the library takes a datum and not "the server", the same code can later be hosted headless in the server's place for an unattended run (the daemon's slot, the same port convention). That host is left unbuilt until an unattended run is approved.

### The components

| Component | What it is | Builds on | New code |
|---|---|---|---|
| `backend/datum.py` | The object protocol and Gemma 4's role table | `gguf_weights.dense`, `dense_cols`, `dense_rows` (general, name no Gemma: gguf_weights.py:542-576); `TheodoliteRuntime.slice_w_*`; `atlas_ops.Weights` (atlas_ops.py:128-209); `runtime_factory.weights_card` (runtime_factory.py:251); interp_engine's ArchSpec and named points | The `Datum` protocol; the role table; per-tensor sha256 of packed bytes; one read path that is the same for every linear |
| `backend/errata.py` | The edit format: source, lowering, identity, lint, store | `export_lora.write_lora` / `read_lora` (scripts/export_lora.py:501-552) as the export container; `export_lora.Plan` as the nearest existing record; `census.write_json_atomic`, `file_sha256` | The manifest, the normal form, the compiler from source to lowered terms, the gate |
| `backend/errata_rows.py` | The apply kernel: per-row terms as forward hooks | The arithmetic of `int4_cpu.with_overlay_terms` (int4_cpu.py:138-164); the hook-in-a-`with`-block discipline of `atlas_ops.installed`; `w4a16_fast.same_bits` for the install-time proof | The hooks, the row masks, the refusal when a batch's rows do not match |
| `backend/proofs.py` | Paired measurements | `atlas_ops.write_rows`, `forward_from_layer` (atlas_ops.py:502, 1017); `causal_check.logits_fp32` and `kl`; `build_steer_layer` as the capture-once template (atlas_build.py:775); `census.control_dirs`, `writer_floor`, `first_departure`; the pure-numpy statistics in `mcp_scientist.py:647-946` | A multi-row, multi-item next-piece reader; a teacher-forced reader over a text; the null generator for errata |
| `backend/table_ledger.py` | The append-only, hash-chained ledger and job records | `census.runtime_record` (census.py:1809), `settings_differ`; the prereg folder pattern under `experiments/results/*/prereg/` | The entry schema, the chain, forecast-before-result |
| `backend/dockets.py` | Task packs, item tiers, the kept-back store, verdicts | The blind test's truth-held-server-side pattern (atlas.py:1939-2004); the passages' odd-window fold (atlas_index.py:125) | The docket and return formats; the tier mechanism at the route, not in the tool |
| `backend/table.py` | The wire: nine verbs over loopback JSON | The atlas router pattern; `local_only` | The routes and the manifest |
| scientist tools | Six to eight tools projected to all three backends | `scientist_tools.TOOLS`, `call`, `openai_tools` (scientist_tools.py:191, 534, 501); `system_prompt`'s file assembly (scientist_chat.py:247) | Tool entries; a third method file; the ledger tap in `call` |
| `#/table` | The person's page | `Page`, `Rubric`, `Apparatus`, `Missing`, `NotYet` (kit.tsx); `streamLines`; `VoiceLine` and the three-voice `Thread` as layouts | The page, three per-tool renderers, replay |
| checks | `scripts/check_errata.py` (CPU, stand-in), `scripts/verify_errata.py` (card) | `scripts/atlas_stand_in.py`; the four-part edit check at scripts/check_gguf.py:825-874; the check idiom | The conformance suite, which is also the standard's test |

Nothing in `backend/engine_runtime.py`, `theodolite_math.py`, `label_gate.py` or `atlas_ops.py` is edited. The first three are pinned by convention; `atlas_ops.write_rows` writes census answers and kit 4 is live, so I wrap it and do not touch it. (The digest found that R4's sha256 pin on `engine_runtime.py` stopped holding on 12 November 2026; that is the owner's to settle and this design does not lean on it either way.)

### The datum protocol

What a model must offer to be the object. Nothing in it names Gemma or a layer count.

    Datum
      id, family, card              content identity; "gemma4"; shape facts and the block formula
      tensors() -> [TensorInfo]     the role table
      read(role, rows, cols, view)  dense numbers or grid codes and scales; one read path for every linear
      tensor_sha256(role)           of the stored bytes
      points() -> [PointInfo]       activation points by role (interp_engine's named points underneath)
      run(tokens, programs_by_row, start_layer, capture) -> readings

    TensorInfo
      role        "L59.mlp.down", "L45.attn.o", "L12.norm.post_mlp", "embed"
      kind        linear | vector | table
      shape       [out, in]
      reads       the space its input lives in (mlp_act, attn_z, resid_normed, ...)
      writes      where its output goes, and through which norm
      storage     packing, group width (32), what a scale belongs to (a row and a group)
      native      its Hugging Face name and its GGUF name
      tied_to     for the embedding: the unembedding
      editable    yes, or no with the reason in plain words

Four datums stand behind it. The 31B W4A16 checkpoint through `TheodoliteRuntime`. Any Gemma 4 GGUF through `GGUFRuntime`. The six-layer random stand-in, on which every check runs and which rehearses plumbing but no hypothesis. And the trained Gemma 4 12B QAT Q4_0 GGUF already on the disk (6,975,878,560 bytes, 48 layers, width 3,840, feed-forward 15,360; header read 14 November 2026), which has never been built or run (x, P5). The 12B matters twice: it is the test that the protocol is not 31B-shaped (the vault, the server and the scientist's tools hard-code 60, 5,376 and 21,504 today; the environment reads every size from the datum), and through the int4 CPU kernel it could be a trained rehearsal object while the card is busy with a census layer.

The role names are the ordinary decoder-only ones (`attn.q`, `attn.k`, `attn.v`, `attn.o`, `mlp.gate`, `mlp.up`, `mlp.down`, the norms, `embed`) plus what a family adds (Gemma's post-norms and layer scalar). Another family needs a builder, a runtime and a role table; the protocol is written so that is all it needs. None is built.

### Relations

- **The vault.** Left as it is in the first increments. The two are kept apart by a guard each way: the environment refuses to pull a proof while the vault has an edit in force, and says so in plain words. Later (increment 6) a vault statement compiles to an erratum of one `col_set`, and the vault gains a file format, single-edit undo and a direct export for nothing.
- **The atlas and census.** Read-only, and always a map of the datum. Every atlas fact the environment hands on carries `describes: "datum"` and its trust card (label gate below page 41; steer lens sound at layers 39-52; doses calibrated at 32-52; link size trustworthy, link sign not).
- **The scientist.** The driver's seat. The chat, its three backends, its stop, rewind and mid-turn messages are kept. The environment adds tools to its registry and a method file to its prompt.
- **The lab daemon.** Not extended and not exposed to the agent.

## ii. What an edit is

### Errata: source and lowered payload

An erratum is a folder of two files.

`errata.json` is the manifest: canonical JSON, small, diffable, holding no weights. It carries the **source**, an ordered list of ops a person can read aloud, and the identities.

`payload.safetensors` holds the **lowered** form: float32 arrays, the exact numbers that are laid on. (safetensors is already a declared dependency; it loads without running code.)

Source is what the agent or person wrote. Lowered is what the source means against one particular datum. Apply uses the lowered payload and never recomputes it. `relower` recompiles the source against a datum and compares by bits; it is run at import and in the conformance suite.

A manifest, abridged (layers from 0, as on the wire):

    {
      "format": "theodolite.errata/1",
      "title": "silence one writer at page 60; take a direction out of three writers at page 46",
      "datum": {"family": "gemma4", "id": "w4a16:e0a93173c31ed2ec",
                "tensors": {"L59.mlp.down": "sha256:...", "L45.mlp.down": "sha256:..."}},
      "source": [
        {"op": "col_scale",   "tensor": "L59.mlp.down", "cols": [7830], "f": 0},
        {"op": "project_out", "tensor": "L45.mlp.down", "cols": [1203, 88, 4410],
         "u": {"unit": {"write_dir": [45, 1203]}}, "dose": 4}
      ],
      "inputs": {},
      "lowered": {"payload": "payload.safetensors", "sha256": "...",
                  "terms": [{"tensor": "L59.mlp.down", "rows": "all", "cols": [7830], "rank": 1}, "..."]},
      "report": {"lowest_layer": 45,
                 "L59.mlp.down": {"relative_change": "...", "cols": 1, "entries": 5376, "grid_exact": true}},
      "parent": null, "source_id": "...", "errata_id": "...",
      "made": {"by": "chat ...", "driver": "...", "code_commit": "...", "at": "..."}
    }

(An illustration of the form. Neuron 7830 of layer 59 is the vault's own example; the layer-45 neuron numbers are made up.)

The per-tensor report holds the same quantities the original authors' released edit manifest holds (changed entries, changed columns and rows, relative change, a hash of the base tensor), so a manifest of theirs can be read in. One in their shape can be written out only for the `folded` realisation, because their `edited_sha256` is of a dense tensor, which here exists only when one tensor is unpacked and the change written into it.

Why both: source ops that are relative to the base (scale this column, project this direction out of these columns) are tiny and readable, but their numbers depend on which encoding of the weights they are read from. The W4A16 checkpoint and the lmstudio QAT Q4_0 GGUF differ in all 410 linears (68-71% of codes equal; write directions agree at cosine 0.9965 at least over 192 neurons, measured 30 October 2026). One source lowered against each gives two payloads with one `source_id`. That is how a recipe outlives a packing.

### The ops (format version 1)

On a linear tensor, every op lowers to one normal form: **a term (R, C, B, A) meaning ΔW[R, C] = B·A**, with R a row set or all, C a column set or all, B of shape [|R|, r], A of shape [r, |C|] (A omitted means a dense block).

| Op | Meaning | Lowers to |
|---|---|---|
| `col_scale(tensor, cols, f)` | multiply columns by f; f = 0 is the vault's DELETE | C = cols, B = (f - 1)·W[:, cols] |
| `row_scale(tensor, rows, f)` | the same for rows | R = rows |
| `col_add`, `col_set`, `row_add`, `row_set` | add to or replace one column or row | rank one on that column or row; `set` stores the difference from the datum |
| `rank_one(tensor, u, v, scale)`, `low_rank(tensor, B, A)` | the adapter form | R, C all |
| `project_out(tensor, u, cols, dose)` | ΔW[:, cols] = -dose·u (uᵀ W[:, cols]) | rank one on cols |
| `block_add(tensor, rows, cols, D)` | any dense change to a sub-block | dense block |
| `sparse_add(tensor, entries)` | a list of (row, col, value) | coordinate list |
| `head_scale(layer, head, f)` | scale one head's block of `attn.o` columns | column block |

Vectors an agent supplies (a direction it computed) sit in the payload's inputs and are named in the source. A small set of symbolic vectors is allowed in source so that common recipes hold no numbers at all: `col(tensor, n)`, `row(tensor, n)`, `write_dir(layer, n)`, `unit(x)`, `input(name)`. Anything richer the agent computes itself and passes as an input.

Format version 1 also names, and the 31B applier refuses in plain words until built: `vec_add` (norm weights, layer scalars), `table_row_add` (the embedding, which is tied to the unembedding, so one op is two terms), and the grid ops `code_set` and `scale_mul` (below). The format is ready for them now because outside work used them: on IWIBench's Gemma 3 4B all 273 of one editor's successful writes went to the tied embedding matrix.

Ops lower in program order against the running state, so "scale, then add" and "add, then scale" differ as a person expects. The lowered terms are all differences from the datum, so laying a program on is a sum.

Tensors are named by role, not by module path: `L59.mlp.down`, `L45.attn.o`, `L12.norm.post_mlp`, `embed`. The role table maps each to its Hugging Face name and its GGUF name. Shi et al. found that finding writers by module name reached 0.066 of a 0.776 effect on a 320B model; roles come from the datum's structure and the table says which roles a family lacks (the ten full-attention layers 5, 11, ..., 59 have no `attn.v`).

### The gate at compile time

Every erratum passes one gate, and its report is written into the manifest:

- Refused: non-finite numbers; an index outside the loaded datum (the vault clamps layer 99 to layer 59; this refuses); a uniform positive scale of a whole `mlp.down` or `attn.o`, which the following norm undoes (atlas_ops.py:244-264 says the same of pushes); anything outside the docket's allowed roles, changed-fraction cap or norm cap.
- Reported, always: relative Frobenius change per tensor; rows, columns and entries changed; the lowest layer touched; for a column op, the size of the change against that neuron's write ceiling (`Weights.write_ceiling`); whether each op is exact on the 4-bit grid.

### How it is laid on

`errata_rows.laid_on(lm, programs_by_row)` is a context manager. For each linear any program touches it registers one forward hook, and removes every hook in a `finally`. The hook adds, for the batch rows whose program has a term there,

    y[row] += ((x[row][..., C] @ Aᵀ) @ Bᵀ) scattered to R

computed in float32 and rounded once into y's dtype, as `int4_cpu.with_overlay_terms` does. Terms on one linear are stacked, so sixty-four rank-one terms cost one extra product of width 64.

Properties, each a line in the conformance suite:

- The base product is untouched: no clone of the unpacked weight (the vault's way clones about 231 MB per touched `mlp.down` per forward, by the digest's arithmetic), no change of numeric path. The digest records what a changed path does: an edit setting a column to its own value changed 13 of 30 greedy answers on the stand-in before the difference-term fix.
- A row with no program is the datum's row. To be proved by bits at install on the card (x, P0), as the unpackers prove theirs.
- A null erratum (a term of zeros) adds exactly nought.
- An erratum acts at every position, the start mark included, as a weight does and as llama.cpp's adapters do. There is no `spare_bos` for errata.
- The hook refuses loudly if the batch it sees does not have the number of rows it was given programs for (a reader that halves a batch on running out of memory would otherwise misalign the masks).
- A push and an erratum can share a row, since they are independent hooks. That gives the check the lineage report asks for: a `col_scale` erratum against the `mul` push it is the same change as, row beside row.

### The 4-bit grid: three named realisations

The checkpoint is 4-bit codes times one bf16 scale per output row per 32 input columns. An erratum's numbers are float32. How the two meet has three answers, and a reading says which it used.

1. **`term`**: the 4-bit base product plus the float32 term. Off the grid by construction. This is the reference meaning of an erratum and the only realisation inside the hypothesis loop. It is an adapter's arithmetic, so export to a LoRA is the identity at the level of the format.
2. **`folded`**: the change written into the unpacked bf16 weight, as the vault's overlays do. Global to a batch. Kept as a named second instrument: for the vault, and as the nearest thing on this card to "what a merged bf16 checkpoint would compute".
3. **`baked`**: the change snapped onto the 4-bit grid. Lossy, and a separate measured step outside the loop. I would bake in one way only: keep every scale, snap the touched entries to the sixteen levels, change no other code, and prove by a tensor-by-tensor diff that nothing else moved. A `mlp.down` column shares each of its 5,376 scales with 31 other neurons, so re-fitting a scale disturbs neighbours; that is refused. The bake reports survival: the share of the asked-for change the grid kept, codes changed, entries clamped. (The one outside figure is a repository's 1.03 on another 4-bit format for a 27B model.)

Some ops are exact on the grid and the gate says so: a `col_scale` or `row_scale` by 0 sets codes to the middle level, which is exactly zero. Negation and scaling a gate or up row through its own 168 scales may be exact; the research check marks both untested (a code of -8 has no negation in range). The grid ops `code_set` and `scale_mul` let a future driver write in the model's native units. None of this is built before P7 says it is worth building.

### Identity

- `datum_id`: the weights card's files id (for the checkpoint, the hash of file names and content-named blobs; `e0a93173c31ed2ec` today), plus a sha256 of the packed bytes of every tensor an erratum touches, computed on first touch and kept in a table. The per-tensor hash is needed because a GGUF's fingerprint skips the middle of the file and three index files carry no card.
- `source_id`: sha256 of the canonical source and its inputs. The same across encodings of the same weights.
- `errata_id`: sha256 of the canonical manifest, which includes the datum's identities and the payload's hash.
- A row's state is `datum` or an `errata_id` (or an ordered list with scales), plus the realisation.

**What a reading taken under an edit records**: per row, the state, the realisation, the datum id, the batch's rows and plan, the unpacker, the card and the code commit. The ledger keeps every manifest by its id, so a state names exact numbers for ever.

If an edited model is ever held as server state (a later, separate mode), its model id takes a suffix in the style of `+int4mm`: `+errata:<first 16 of the state's hash>`. The existing rule that an index built for another model id is refused then keeps every atlas index from being read as the edited model's, with no new code.

### Composing, undoing, saving, reloading, exporting, importing

- **Compose.** Two errata pinned to one datum add. That is also what llama.cpp does with two adapters (the repository's own note: `mul f1` and `mul f2` served together give f1 + f2 - 1, not f1·f2). The tool reports cells both write, because a sum there is neither's intention. `rebase` recompiles one's source on top of the other and gives a new erratum with a parent link.
- **Undo.** An erratum is a value. A new version is a new id with `parent`. There is no state to roll back.
- **Save and reload.** `atlas_data/errata/<errata_id>/`, git-ignored. Reload verifies the payload's hash and the touched tensors' hashes against the loaded datum and refuses on any difference.
- **Export.** Per tensor, the terms are concatenated into one full-shape (A, B) pair and written through `export_lora.write_lora`, with the manifest's own text stored under `theodolite.errata.*`. A 64-column change is rank 64. A 4,096-column block is rank 4,096 and a large file (A alone about 352 MB in float32, my arithmetic); the export warns.
- **Import.** `read_lora` gives arrays and keys. If the file carries a manifest, the erratum is rebuilt and its hashes checked, so the round trip is lossless; otherwise each tensor comes in as a `low_rank` op. Whether the round trip is exact by bits is not established today and is the first line of the CPU check.
- **What export may claim.** The repository has read llama.cpp's Gemma 4 graph only for `ffn_down` and `attn_output`. An export touching other tensors says they are unverified until the three-level export check is extended to them. An erratum lowered against the checkpoint and served over a GGUF is an erratum on a different datum; the export says so and offers `relower` against that GGUF.
- **What may be published.** A manifest holds no weights and is a recipe anyone with the datum can lower. A payload of base-relative ops holds pieces of Gemma's columns and is a derivative under Apache 2.0. The default is that payloads stay in the git-ignored folder and only manifests are committed with a registered result. Anything further is the owner's.

## iii. How the agent reads

A language model cannot read 29.3 billion weights. Raw-weight reading has been shown only to about 172,000 weights, and one audit found a reader leaning on priors. What it can work with is a ladder, each rung the same verbs on any datum, each reply carrying its trust card.

0. **The datum card.** A few kilobytes: the role table (410 linears in eight shapes, the norms, the scalars, the tied embedding), the block formula (post = s·(mid + post_norm(mlp(pre_norm(mid))))), the storage (codes, scales, groups of 32), and the traps stated as facts: a uniform rescale is undone by the norm; one neuron's write has a ceiling; about 180 stream dimensions are read with far higher gain. `GET /api/atlas/arch` is the existing self-description this extends.
1. **The atlas as the symbol table.** For every one of the 60 × 21,504 neurons: lens words (under the gate), usual range, the 24 strongest links each way, stored passages, a steer-lens entry; for 12 layers a whole causal census against 2,048 controls. This exists and no other 31B has it. The 35 tools already serve it. The environment stamps each fact with what it is known to be worth.
2. **Statistics of a tensor, computed once and cached by the tensor's hash.** Row and column norms, the leading singular values and vectors, a histogram of the codes, the projection of a direction onto every write column of every layer (which neurons write along u). These are harness services so that a driver's turn is not spent recomputing them.
3. **Bounded slices** in either view: the dequantised numbers, or the grid (integer codes -8 to 7 and the group's scale). Capped to what a reply holds (an endpoint model reads at most 16,000 characters of a tool reply).
4. **Tensor handles in a kernel** (increment 4). A CPU-only Python process with the packed weights mapped read-only. A tensor is unpacked on demand, by row or column in O(K) (`dequantize_w4a16_rows`, `_cols`; `GGUFLinear.unpacked_rows`, `_cols`), or whole one at a time (one `mlp.down` is 231 MB in bf16; the machine has 61 GiB of memory by `free` on 14 November 2026, so the whole model unpacked, about 62 GB, does not fit there either). Only what the agent's code prints crosses into its context.
5. **Activations, through proofs.** Next-piece odds in fp32, the stream at named points, chosen neurons, attention, on texts the agent writes. Budgeted.
6. **Gradients, as a declared level.** Off unless the docket says `attribution`. labkit's same-position VJP gives input gradients through the packed weights (0.661 s for a forward with a graph and 0.314 s for one backward on a 128-piece prompt on the reference unpacker, measured 22 October 2026). The level is written into the ledger and results are never pooled across levels. Optimisation loops are refused at every level.

Nothing here assumes how good the reader is. A stronger driver lives on rungs 4 and 5 and ignores my statistics. That is the point of making the verbs the standard and the rungs optional.

## iv. How a hypothesis is tested, fast

### The proof

A **proof** (printing: a trial impression pulled and read before anything is printed; the literature's paired forward pass) is one request: an item set, a list of arms, a kind. Arms are errata, pushes (the cheaper knob), nulls the harness draws, and always the datum. Rows of one forward carry different arms on the same item. The reply is a table: per arm, the paired difference from the datum's row on each item, its mean and standard error, counts of items helped and harmed, and where it stands among the nulls.

Three kinds:

- **Next-piece, from a layer.** The item set is run once to the stream entering the lowest layer any arm touches; then only the layers above are run for every arm (`forward_from_layer`, the steer lens's build as the template). Odds from fp32 logits. KL is always KL(datum ‖ arm), in nats, with non-finite logits masked. (The existing evaluators disagree on KL's direction; this one fixes it and names it.)
- **Teacher-forced over a text.** The same, scored at every position of a continuation: the log-probability the arm gives the text, and per-position KL on neutral text for side effects.
- **Greedy writing.** `write_rows` under the hooks, up to 48 word-pieces, one batch shape, the datum's row in the same batch, read against the writer's floor.

### What it costs on this card

| Proof | Rows | Seconds | Basis |
|---|---|---|---|
| Next-piece from layer 45, short prompts | 516 | 0.85 | Measured: the steer lens ran 336 such forwards in 287 s, 29 October 2026 |
| Next-piece from layer 5, short prompts | 516 | 3.5 | Measured: 1,181 s for the same 336, 29 October 2026 |
| Whole forward of a census question | 32 | 0.26 | Measured prefill, compiled, 28-29 October 2026 |
| The same | 96 | 0.60 | Measured, same file |
| Greedy writing, 80 pieces | 15 | 9.2 | Measured through `/steer`, 11 November 2026 |
| Greedy writing, 48 pieces | 32 | 5.2 | My arithmetic: 0.26 + 48 × 0.103 (steps measured 28 October 2026) |
| Practice proof: 64 short items × 8 arms, next-piece from layer 45 | 512 | about 1, plus about 0.5 once for the capture | My arithmetic from the rows above |
| Confirmation proof: 256 items × 8 arms, 48 pieces written, batches of 32 | 2,048 | about 330 | My arithmetic: 64 batches × 5.2 s |
| One heard vault edit, for scale | 2 answers | 12.6 | Recorded; date and unpacker not stated |

The steer lens's four prompts are a few word-pieces each. For 64-piece texts the row counts fall to what memory allows: a batch of 32 rows writing 64 pieces took 2.4-3.1 GiB above the model and 96 rows 7.2-7.5 GiB (measured 28 October 2026), and with the J++ lens loaded there is about 5.6 GiB of headroom beside a desktop (about 8.8 without it). The default proof batch is therefore 32 rows, which is also the shape with the better writer's floor (an unpushed answer holds 54 pieces at 32 rows against 22 at 64; 29 October 2026). What the terms themselves add per forward has never been measured (x, P1); by arithmetic a few dozen rank-one terms are far below the cost of the products they sit beside.

So a practice proof with its nulls is a second or two, a written one under a minute, a registered confirmation about five minutes. A driver's own turn takes longer than the first two (20 to 70 s a turn in the original authors' logs). The card is idle most of a session, the lock is released between calls, and the person's pages stay usable.

### How many candidates share a forward

Next-piece from a layer: about five hundred short rows, so 8 arms on 64 items or 128 arms on 4. Writing: 31 arms and the datum on one item at a batch of 32, or 8 items × 4 arms. A written step costs 0.091 s at one row and 0.109 s at 64 with the compiled unpacker (measured 28 October 2026), because much of the time is the unpacking, so filling a forward is close to free up to memory.

### What is cached, and what is not

Cached: the stream entering a layer for an item set (about 44 MB a layer boundary for 64 items of 64 pieces in bf16, my arithmetic), keyed by the items' hash, the datum, the layer and the batch plan; tensor statistics by tensor hash; lowered errata by id.

Not cached: the datum's row. It is run again in every forward. Rows of one forward are bit-identical and rows of different batch shapes are not (KL up to 0.014 in the study), and that fact is the reason the comparison is worth anything.

Not built: writing from layer L upward on cached keys and values of the lower layers. No function does it today; it would roughly halve a written proof for late-layer edits; it is the speed designer's to argue for.

## v. What stops the agent fooling itself, and the owner

The literature is plain about this: in the replication that tabulated it, the best development gain exceeded the held-out gain in all 14 runs (a best-of-search figure on small batches, so partly built in); three identical searches gave +29.45, +5.00 and +27.30 points; 6 of 16 searches rightly ended with the original weights; agents' own accounts of incomplete runs were misleading four times in five in one benchmark. The guards are therefore mechanisms, and the agent's prose is never evidence.

1. **Three tiers of items.** Practice items the agent may read. Confirmation items it never sees: it names the set and gets back aggregates, a fixed number of times per claim. Sealed items that only the owner can spend, once per registered claim. The kept-back tiers live outside the scientist's work folder and no verb returns them; the hold-out is enforced at the route (the blind test's hold-out today is enforced only in the tool, and the route's default shows everything). The environment refuses to run when the scientist's reach is `home`, because then the agent can read the disk.
2. **Nulls drawn by the harness, in the edit's own representation.** For any arm: several random directions on the same rows and columns at equal Frobenius norm, drawn as the census draws its controls (`unit(w_post ⊙ r)` or mixes of the layer's own write directions, because a plain isotropic direction is read about 100 times louder downstream); the same size at random places; the reversed sign; a dose sweep. The place among nulls is an empirical p of (1 + nulls at least as strong)/(C + 1), and the reply says what it was corrected over.
3. **Noise floors.** The datum's row in the same forward; the writer's floor for the batch shape; the null erratum. A departure inside the floor is reported as no departure.
4. **Side effects, always.** A standing panel no arm can skip: KL(datum ‖ arm) and top-piece agreement at every position of a fixed set of neutral windows (drawn from the stored corpus's even windows; the odd ones stay the kept-back fold), and near-miss items beside the target. Reported next to the target, and as the product gain × retention that IWIBench uses, so the number travels.
5. **Cheaper knobs as arms.** A sentence in the prompt, the activation push the erratum was derived from, a control vector. A weight edit that does nothing a push does not is reported as such. The headline published edit was matched or nearly matched by a settings change, and most of the way by banning one word at decoding.
6. **Forecast before result.** A proof on a kept-back tier needs a forecast (direction, least size, which nulls it will beat), hashed into the ledger before the proof runs. The ledger is a chain: each entry carries the hash of the one before. The harness, not the agent, applies the keep rule.
7. **The ledger keeps everything**: every proof, failed ones and discarded errata included, with rows, GPU seconds, the driver as served and the runtime record. Multiple looks are counted because they are all there.
8. **A fixed vocabulary of verdicts.** `kept the datum` (no arm cleared its nulls: an ordinary, expected outcome); `lead` (practice only); `stands against nulls` (practice, with nulls and side effects); `confirmed` (confirmation tier, forecast met, cheaper knobs reported); `of record` (the house's own registration: a hashed prereg folder, the sealed tier, the owner's word); `regressed`; `not measured`.
9. **What any result may claim** is bounded by what it was measured in: the realisation (`term` says nothing of a baked file), the batch shape, the card class, the items' domain, the evidence level, and the trust bands of any atlas fact it leaned on. A lens label of a neuron the erratum touched is never evidence that the edit worked: the gate was not measured on edited neurons and the vault's own UPDATE is built through the same lens.

Two levels of registration, on purpose. The ledger's forecast is automatic and cheap and covers every proof. The house's prereg folder (rules, runner, reader, inputs, commit, SHA256SUMS) stays the owner's act for a result of record, made from a ledger entry by a script. An agent cannot commit and cannot spend the sealed tier.

## vi. The agent interface

### One wire, three clients

The standard is the wire: nine verbs, JSON over loopback, described by a manifest a driver reads first.

| Verb | Does |
|---|---|
| `datum` | the object card, roles, trust cards, budgets left |
| `read` | a bounded slice of a tensor, dense or grid view |
| `stats` | cached statistics of a tensor; projections of a direction |
| `compile` | source in, erratum id and the gate's report out |
| `proof` | a paired measurement; returns a job id when the estimate passes a minute |
| `forecast` | file a forecast for a kept-back proof |
| `verdict` | ask the harness to apply the keep rule |
| `ledger` | read this session's entries and any job's state |
| `export` | the owner's only; refused to an agent |

Three of these are the ones that must be typed, counted and logged whatever else changes: observing the live model (`proof`), committing an edit (`compile`), asking for a verdict.

Clients:

- **Today: tools.** Each verb is an entry in `scientist_tools.TOOLS`, so Claude Code, Codex and any OpenAI-compatible endpoint get them at once, and a new endpoint model plugs in with one entry in `endpoints.json` and no code. Layer numbers pass through `from_page` and `to_page` like every other tool. This is all a current frontier model needs to form a hypothesis from the atlas, write an erratum, and test it with nulls in seconds.
- **Next: code** (increment 4). A CPU-only kernel under bubblewrap (`/usr/bin/bwrap` is on the machine): no network, no GPU device, the packed weights and the atlas index mounted read-only, one scratch folder. Its `table.*` calls go down a pipe to the server's dispatcher, the same shape as the stdio MCP server, so the sandbox needs no loopback. The kernel can hold weights and cannot run the model except through `proof`. The same verbs, importable. A stronger model that wants to solve for an edit (a least-squares fit, a linear program over columns; once the fault and the layers were known, one outside party's solver did in about a minute the weight-selection step that had taken an agent about eight hours) does it here and hands the result to `compile` as inputs. No rewrite: the wire did not change.
- **Always: the page.** The person uses the same verbs.

Left unbuilt until someone needs it: a plain-text protocol of one fenced block a turn for models without tool calling. It is an adapter of a hundred lines over the same verbs. On IWIBench the three open-weights editors tested scored no better than the unedited model, so this is not where to spend first.

### Budgets

In GPU seconds under the lock and in rows × word-pieces, not in passes. The balance comes back with every reply. Looks at the confirmation tier are counted per claim. A docket may also state turns and calls so that a result can be set beside benchmarks that count those (IWIBench: 64 turns, 128 calls, 262,144 forward tokens). Dollars and tokens are recorded where the backend reports them (Claude Code's per-turn cost is emitted today and dropped).

### State across sessions

Three things on disk and nothing in the conversation: the ledger, the errata store, and the session's notes in the work folder. A fresh session is handed the docket, a digest of the ledger written by the harness (what was tried, what stands, what was discarded), and its own notes marked as unverified notes. The marking matters: one lab's auditing agent grew falsely confident from its own memory files, and the original authors' agent ran 26 turns with its conversation frozen and survived only on its notebook and work folder. Jobs longer than one held request (the MCP call's timeout is 900 s) are submit-and-poll with a job record.

### What is deliberately thin

No required workflow, no stages, no councils, no context-reset machinery, no planner. The loop is the driver's. Each piece of scaffolding that compensates for a weakness of today's drivers is listed in the harness card with the weakness it is for, and is reviewed whenever the driver changes. (One lab removed two such pieces within a model generation each.)

Two things are thick, because drivers skip them: `proof` adds nulls and the side-effect panel without being asked, and one bundled call runs the positive, negative and near-miss cases together.

### The harness card and replay

Every return carries a harness card: the driver as actually served, its effort, the tools it had, what it could and could not see, the budgets, the context policy, the code commit, the unpacker, the card. One study measured harness variance at 7.8 times model variance; a result from here is a fact about model plus harness and says so.

A session can be replayed with no model: a scripted driver sends the logged wire calls again. That is the environment's own regression test, and it is how a person steps through a past session as a lesson.

### The docket and the return

The **docket** (printing: the sheet of instructions that travels with a job; the literature's task) is a folder another harness could run:

- `docket.json`: format and version; the question in plain words; the datum it needs (an exact id, or a family and the roles it must have); the policy (roles that may be edited, the cap on the share of weights changed, norm caps, the evidence level: reasoning only, activations, or attribution); the budgets (GPU seconds, rows × word-pieces, turns and calls if wanted, looks at the confirmation tier); the metric, stated as data and not as code (kind: target log-probability, logit difference, greedy match or KL to the datum; its direction and precision; how it is aggregated); the nulls required; the keep rule; the side-effect panel; the cheaper-knob arms; and the text the driver is shown, word for word, with a flag saying whether it names a suspected fault. That flag is there because the original result's prompt named the fault that became the fix, and the claim of discovery is disputed for it.
- `items/practice.jsonl` in full. For the confirmation and sealed tiers, only their hashes and sizes: the items themselves stay with whoever holds the docket.

The **return** (a surveyor's return is the report filed of a survey; the literature's result) is what comes back:

- `return.json`: the docket's id; the datum's id; the harness card; the verdict from the fixed vocabulary; the final errata by id, or none; for each tier the paired numbers with standard errors and the place among nulls; the side effects; the cheaper knobs; the realisation; what was spent (GPU seconds, rows, tokens and dollars where known); the count of discarded attempts; deviations; and the hash of the ledger segment the session wrote.
- The manifests of the errata it names. Payloads only if the owner chooses.

A docket's budgets and metric are stated so that the published benchmarks' terms can be expressed in them: IWIBench's gain × retention and its turn, call and token caps; a forecast before an edit, as WSPBench asks. I do not claim compatibility with anything: the original authors' benchmark and harness are announced and not released, and the right response is a thin adapter at the edge when they are.

## vii. The person's place: the page

`#/table`, a new page (four places to add it: the route, the href table, the titles and screen in App.tsx, the nav list). Vellum, Garamond, rubrics on by default, apparatus off. Layers shown 1 to 60. Three parts:

- **The ledger**, as a reading text and not as JSON: for each step the agent's hypothesis in its own words, its forecast, the proof, the result with its nulls, the verdict. The chat renders every tool call as a collapsed row of raw JSON in a 580-pixel panel today; this page gives `compile`, `proof` and `verdict` their own renderers.
- **The erratum reader**: the source in plain words ("at page 46, neuron 1,203: subtract four times this direction from its write column"), the gate's report, the neurons it touches with their atlas labels where the gate allows them, a diff between two errata, and what the grid would keep.
- **The proof sheet**: the datum's row beside each arm, the nulls as a strip with the arm marked on it, the floor drawn as a line, the side-effect panel beside the target. The three-voice thread of the push tab is the layout for a written proof.

Rubrics explain each step to a newcomer: why there is a row with no edit, what a null is, why a lead is not a finding. Apparatus shows the formulas, the raw values, the hashes.

Two controls make it an auditor's page. **Replay** steps through any stored session without the card. **Pull this proof again** re-runs one entry live and shows whether it reproduces by bits (same batch shape, same card) or within the floor.

The chat stays in the atlas for now, since its component is tied to atlas state. The page shows the session's thread read-only and links across. In the first agent increment the page is read-only.

## viii. Safety and scope

- **Reach.** The scientist's default reach is unchanged: no shell, files only in its work folder, no credentials. The environment refuses at reach `home`. The kernel, when it comes, is a sandbox by mechanism, not by prompt; outside harnesses say plainly that prompt rules were not a boundary, and one agent bypassed a blocked command through its notebook tool.
- **Nothing the agent does changes the model the server holds.** Errata exist inside a proof's `with` block. The weights are mapped copy-on-write and nothing is ever written back to the checkpoint or the cache.
- **Exports are the owner's.** No agent verb exports. An exported file carries the manifest, the datum's identity, the ledger entry and the verdict, a plain statement that the weights were modified, and the Apache 2.0 notices, and is named as derived from Gemma, not as Gemma. Provenance is not assurance and the file says that too.
- **What the edited model writes is untrusted data** when shown to the agent, like every other model output.
- **The harness will not**: run gradient optimisation; bake into the checkpoint; publish anything; let the agent keep its own proposal, spend the sealed tier or write the ledger; ship a trigger-to-payload recipe. A weight-edit backdoor has been shown on the bf16 release of this very model (11 November 2026; its headline figure is for a gradient-refined variant in one of four settings, on recorded trajectories). Refusal-stripping is already a commodity and this adds little there. If an agent finds an attack-like edit, it goes to the owner and not into the repository.
- **Imports from strangers** are linted before they are loaded (support, norms, last-block write magnitudes) and diffed against their declared datum. Unbuilt until the first such import.
- **Unattended runs** need the owner's approval in GPU hours on the home card, in the spirit of the budget rule for rented cards, and a stop rule. Until then every session has a person able to stop it.

## ix. The phased path

Each increment is useful if the work stops there.

| # | Increment | Delivers | Proves | Rough effort |
|---|---|---|---|---|
| 1 | **Errata and the apply kernel, on the stand-in; pilots on the card.** `errata.py`, `errata_rows.py`, `check_errata.py`, `verify_errata.py`, a command-line proof. No server change, no agent. | The vault's missing pieces as a library: an edit with a file, an identity, undo by construction, an export, and a same-forward comparison. | P8 and P0-P4: base rows keep their bits; what a term costs; how far `term` and `folded` part; whether from-layer equals a whole forward; whether a one-neuron erratum is heard. P5 (the 12B) runs alongside, since it needs none of this code. | 4-6 days; one to two hours of card |
| 2 | **The table in the server.** Routes, proofs with nulls and the side-effect panel, the ledger, practice tier, six scientist tools, the read-only page. | A current frontier model forms and tests hypotheses on the 31B in seconds, and a person can read what it did. | P6: proofs per hour in a real session; whether the driver keeps using the tools. Verdicts stop at `stands against nulls`. | 2-3 weeks |
| 3 | **The guards and the exchange formats.** Kept-back tiers at the route, forecasts, verdicts, cheaper-knob arms, docket and return version 1, the harness card. | The first `confirmed` result, and the first return another harness could read. | The development-to-held-out gap, measured here. | 1-2 weeks |
| 4 | **The reading room and a second object.** The `Datum` protocol written down and tested; the 12B as a datum; the sandboxed kernel; the statistics cache; replay. | A code-executing driver plugs in with no server change; rehearsal on a trained small model, on the CPU while the card runs a census layer. | That the protocol is not 31B-shaped (P5). | 2 weeks |
| 5 | **Portability.** Export and import by bits; an erratum served through llama.cpp on the 31B against a bar fixed beforehand; `relower` against a GGUF; an outside published edit imported and re-measured; a volunteer's card re-running a proof under the kit's intake pattern. | An edit made here is an object other people can load, check and re-measure. | Engine-to-engine agreement for errata beyond one column. | 1-2 weeks |
| 6 | **The grid and the rest of the tensors**, only if P7 says so. Grid ops, the bake with its survival report, appliers for the embedding, norms and scalars, an edited model as server state with the person's keep, the vault compiled into errata, attribution as a level. | The model's native units; the whole tensor table editable. | Whether anything survives the grid. | 2-3 weeks |

## x. Pilots, in order

Each names what would change. P8 and P0 to P5 are run on increment 1's small rig, before anything from increment 2 on is committed to; they need one to two hours of the card between census work. P6 needs increment 2 and decides the shape of increments 3 and 4. P7 decides whether increment 6 exists.

| # | Measurement | If the result is otherwise |
|---|---|---|
| P8 | On the CPU, free: the erratum to LoRA GGUF and back by bits on the stand-in; time to hash the datum's tensors; time for one `mlp.down`'s leading singular vectors. | If the round trip is not exact, the GGUF carries the payload's hash and import verifies instead of trusting. If the SVD takes minutes, it is a queued job, not a call. |
| P0 | On the 31B, compiled unpacker: with the hooks installed and no program, are all rows bit-equal to the same batch without hooks, and does the unpacker's `fell_back` counter stay at zero? Is a row with no program bit-equal beside rows that have one? | If hooks move the bits, "the datum's row" means "the row under empty hooks" and environment readings never mix with atlas readings. If the unpacker falls back, the hook is restructured before anything else. |
| P1 | Seconds added per forward by 1, 64 and 4,096 terms at 32 rows. | Above about a fifth of a step, terms are batched differently or capped per proof. |
| P3 | `forward_from_layer` against a whole forward on the 31B (compared only on the stand-in so far), and two same-shape from-layer runs against each other. | If not equal, from-layer proofs are a pre-filter and never a confirmation. |
| P2 | `term` against `folded` for the vault's example edit and for a 64-column erratum: KL between them, set against the batch-shape floor (0.014). Also: a `col_scale` 0 erratum against a `mul` 0 push with the start mark not spared, by bits. | If the gap is inside the floor, `folded` is dropped from the first increments. If outside, every claim about a merged file needs its own measurement, and the vault's compile into errata waits. |
| P4 | Audibility. For the census's leading neurons at one calibrated layer: `col_scale` at 0, 2 and 4 and `project_out` at three doses, 64 practice items, next-piece. What share clear the 95th percentile of their nulls? (The vault's own note says a heard one-column edit is rare.) | If only a few in a hundred do, the unit of a hypothesis is a many-neuron program along a direction, the single-neuron tools are demoted, and the solver-in-the-kernel moves earlier. |
| P5 | Does the builder build the 12B QAT Q4_0 GGUF on disk and give a sound forward, on the card and through the int4 kernel on the CPU; seconds and GiB? (Its header passes the refusals; nothing has run it. One outside source describes the 12B as a variant whose reserved word-pieces give -inf logits.) | If it fails, rehearsal stays on the stand-in and the protocol's second-object claim waits for a builder fix. |
| P6 | Two hours of a real session with a current driver on typed tools: GPU seconds used against wall time, calls per verb, whether it keeps using proofs. | If it drifts to chat, the bundled proof becomes the default and the method file changes. If GPU time binds, the written-proof shortcut is built after all. |
| P7 | Survival on the grid: 100 single-column changes at three sizes relative to the group scale, snapped with scales kept: share of the change kept, and KL(`term` ‖ `baked`). | If small edits vanish and large ones distort, baking is dropped from the plan and "edits live beside the 4-bit base" becomes the stated limit. |

## xi. Risks, what is left out, and the owner's decisions

### Risks

1. **A healthy 31B gives small effects and many nulls.** Outside results on stronger models were gains of one to three points held out, and edits removed obstacles more than they added abilities. The environment has to be judged on how fast and how honestly it tests a hypothesis. `kept the datum` must read as a result.
2. **Three realisations are three objects.** If `term`, `folded` and `baked` part by more than the floor, careless wording will overclaim. The realisation is in every reading for that reason.
3. **The formats become a private dialect.** The original authors have announced a benchmark and harness and released neither. I keep importers and exporters at the edges and the core small, so a de facto format can be adopted by writing an adapter.
4. **The base row is not the atlas's row.** Batch shape moves numbers. Results from the environment and results from the atlas are comparable only through their own base rows.
5. **The card is shared.** A census layer holds it for 13 hours and an intake's spot check needs it. Sessions will be interrupted; state on disk is the answer, and the 12B on the CPU is the rehearsal.
6. **Today's drivers may under-use it.** Agents call an effective tool a few times and fall back to chat; more reasoning effort did not reliably help.
7. **Injection.** Model output and stored passages reach an agent that can now write errata. Errata change nothing outside a proof, which bounds the harm.
8. **Up-front cost.** This design spends its first increment on a format and a kernel with no agent in sight. If the pilots go badly (P0 or P4), part of that is rework.

### Left out, deliberately

Training and optimisers. Re-quantising a whole model. A builder for any family but Gemma 4 (a Qwen GGUF is on the disk; the protocol is ready and no builder is). Learned dictionaries: community SAEs exist for the 31B at unverified quality, and the atlas is preferred because it is causally tested here and covers every layer. Rebuilding any index for an edited model. Rented cards. Councils of agents. A leaderboard. A language richer than JSON ops. A separate operating-system user for the kept-back tiers (the stronger boundary, at the price of an install step on a one-owner machine).

### The owner's decisions

1. The name.
2. Whether the pin on `backend/engine_runtime.py` stands, given that its hash has not matched since 12 November 2026.
3. Whether attribution gradients are a permitted evidence level, and whether a result reached with them may be called weight-space programming beside the published ones.
4. What may be published: manifests only, or payloads too, and under what notice.
5. Who writes the confirmation and sealed item sets, and on what behaviours. The agent must not.
6. Whether a session may run unattended, and its budget in GPU hours.
7. Whether the J++ lens is freed during a session (about 3.2 GiB more headroom, worked out not measured) at the price of the sightings pages.
8. When, if ever, the vault's state moves under errata.
9. Which driver, at what effort, for the first sessions, and whether a docket's prompt may name a suspected fault (it must be recorded if it does).
10. Whether volunteers may be asked to re-run proofs.

## xii. What would make this the first credible weight-space environment for a 31B

As of 14 November 2026 the searches found no agent-driven weight-editing experiment in a preprint on a model above 9B, none anywhere on the 31B, and no measurement of an edit carried across re-packing of a QAT checkpoint. Community repositories reach 27B (three edits of one model, one of them made and measured in a 4-bit format) and include an agent's edit of Gemma 4 12B. These are absence claims from a search and one such claim in the research was already found wrong. What credibility needs, and what each rests on:

1. The object is pinned and its packing is dealt with openly: three realisations, named, with measured gaps (P2, P7).
2. Every number is a paired difference from a base row in the same forward, set against a measured floor (P0).
3. Nulls, kept-back tiers and forecasts are mechanisms (increments 2 and 3).
4. The symbol table is causal and says where it is weak: the atlas and census.
5. The artefacts can be loaded and re-measured by someone else: a manifest, a docket, a return, a conformance check, and another card's re-run (increment 5).
6. Time and cost are published per proof and per session. No source gives them for a model of 27B or more.
7. Null results are kept and shown.

## xiii. Names

In the house's vocabulary, with the literature's term beside each. Five new words, each needing a Lexicon entry; the rest stays in plain words (base row, noise floor, ledger, session).

- **The Plane Table** (recommended). The surveyor's board on which the map is drawn in the field while the sights are being taken: the place where observing and drawing happen together. With it: the *datum* (base checkpoint), *errata* (edit program), a *proof* (paired test), the *docket* (task pack), the *return* (result pack; a surveyor's return is the report filed of a survey).
- **Errata.** The feature named for its central object: a dated list of corrections pinned to one edition, which changes nothing in the printed book.
- **The Proofing Press.** Where a trial impression is pulled and read before anything is printed: every edit here is a proof and never an edition until the owner passes it.

## xiv. What this design rests on that is not established

| Rests on | Status | Settled by |
|---|---|---|
| Forward hooks on the packed linears leave untouched rows bit-equal and do not trip the compiled unpacker | Not measured; hooks on these modules are already used by pushes | P0 |
| The cost of terms per forward is small | My arithmetic only | P1 |
| `term` and `folded` agree to within the batch-shape floor | Not measured; the documents say only that they round differently | P2 |
| A from-layer run equals a whole forward on the 31B | Checked on the stand-in only (atol 1e-5) | P3 |
| Same-shape forwards repeat by bits through `forward_from_layer` | Inferred from the census determinism run | P3 |
| A single-neuron weight edit is often audible | The vault's note suggests it is rare | P4 |
| The 12B on disk builds and reads soundly | Header only | P5 |
| A current driver will use typed proofs productively | No transcript-level record of any real driver in the scientist exists | P6 |
| Anything survives the 4-bit grid | One outside figure on another format | P7 |
| llama.cpp applies an adapter to Gemma 4 tensors other than `ffn_down` and `attn_output` | Not established in the repository | Extending the export check, increment 5 |
| The short prompts behind the 0.85 s and 3.5 s figures stand for 64-piece items | They do not; those rows are a few word-pieces | P1 times the real shapes |
| The proposals keep route cannot be called by an agent at the default reach | Not established for the existing route | A line in `verify_scientist_cli.py` before any owner-only verb relies on it |
