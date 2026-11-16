# Traverse: a weight-space bench, designed from the driver's seat

Design only, written 14 November 2026 against Theodolite at commit e6ac64f. Nothing here was run on the
card. Every figure is carried from the two digests with its date and is marked **measured** (a record in
the repository) or **arithmetic** (mine, worked from those records). Layers are counted from 0, as in
every script and file; a page number is the layer plus 1. "Traverse" is a working name (section 11).

Lens: driver first. I design from the seat of the model that drives the loop and push that to its end.
Where it conflicts with another priority I choose the driver and say what it costs.

---

## 0. The position

The driver gets **one act: run a cell of Python on the card with the model in scope**, and one
workspace of plain files that is its whole memory. Everything it reads, measures and tries happens in
cells. There is no menu of reading tools, no workflow, no chat runtime of ours.

Three things are not free code. They are typed and narrow because they are where the owner has to be
able to believe something, and that reason does not age with the driver:

1. **Holding the card** (a budget in card time, counted from outside the cell).
2. **Freezing an edit** (a small file of deltas against the pinned base, hashed).
3. **Asking for a verdict** (a registered claim, scored on items the driver never saw, by a fresh
   process that runs no driver code).

The rule that follows: **the loop is free and optimistic; the claim is registered and expensive.
Nothing measured in the loop is a result.** It is a lead, in the house's own word.

Why this shape and not a set of typed tools:

- The repository's own history: Claude Code agents designed and ran the study's 314 GPU jobs by
  submitting Python to the lab daemon (282 ok, median 71.3 s, 16 October 2026, measured). That is a
  working driver loop on this model on this card. The current scientist, with 35 typed tools and 86 KB
  of prompt and schema (39,051 + 34,660 + 12,345 bytes, measured 14 November 2026), has no record of a
  real driver's behaviour at all.
- The literature on agents and tools: agents call an effective tool 5 to 10 times and fall back to
  chat (AuditBench); activation-reading tools gave no uplift over a transcript-only agent (CHIVE);
  helper-API friction made the driver with the soundest plans the least successful at execution
  (GPT-5.4 in HyVE, 0.52). The lesson I take: a wide tool surface is ignored or misused. One entry
  point cannot be ignored, because nothing else makes progress.
- The literature on harnesses ageing: context resets and sprint decomposition were each removed
  within one model generation (Anthropic's harness posts); councils did not beat their strongest
  member on any natural task (wsplab); a shared thin harness beat native ones in 11 of 20 pairings
  (HarnessOpt-Bench). Every component encodes an assumption about what the model cannot do. I build
  only components whose reason is the owner's trust or the card's physics.
- The one event in the weight-space-programming logs that matters most for design: the agent ran 26
  turns with its conversation frozen at turn 14 and survived on its notebook and work folder
  (checker's correction to the paper digest). State on disk is the mechanism that works. Conversation
  is not.

What this costs, plainly:

- Loop numbers are unguarded. A driver can fool itself for an hour before a verdict says no. The
  ledger will hold many optimistic leads, and none may be quoted.
- Free code beside the model cannot be trusted to count its own forwards or keep its own rules. So
  containment must be an operating-system sandbox, and until that exists sessions are attended.
- The instrument's server is down while a traverse holds the card, as it is for a census job. There
  is no page for it until phase 4.
- I do not build a general overlay kernel inside the runtime, a job scheduler, layer-resumed
  generation, a spectral index of the weights or a model-neutral object protocol. Other lenses would.

The bet that could be wrong: that a current frontier driver does better with free cells than with
typed verbs. The evidence for typed tools comes from Claude Haiku 4.5 and mid-2026 models; IWIBench
(arXiv 2611.08647) shows closed editors working with four typed tools and no Python. Pilot 2 in
section 9 settles it for this bench. If it goes the other way, the verbs are Python functions first
and projecting them as tools is an afternoon.

---

## 1. What the environment is

```
 driver: Claude Code | Codex | any chat model          (not ours; its vendor's harness is the loop)
    |   one of three thin faces: MCP (7 tools) | a command line | one fenced block per reply
    v
 KEEPER      new, standard library only, no torch, runs as the owner
    |        owns: the Unix socket, the card lock, the ledger, the budgets, the held items
    |        starts, stops and kills the two workers; always answers status
    +--> BENCH WORKER     holds the 31B on the card; runs the driver's cells; sandboxed
    +--> VERDICT WORKER   fresh process; runs only registered, hashed code; never a cell
```

Only one of the two workers exists at a time, because only one process can hold the model on the
32 GB card.

| Component | What it does | Built on (existing) | New code |
|---|---|---|---|
| **Keeper** | The single door. A JSON protocol of seven calls over a Unix socket (mode 0600) in `$XDG_RUNTIME_DIR/theodolite/`. Takes a machine-wide card lock, refuses while `127.0.0.1:8765` answers, enforces a time limit per cell (SIGINT, then kill and reload), counts card time, writes the hash-chained ledger, keeps held items where no worker or driver can read them. | `labd.py`'s guard idea (`local_scripts_only`, `launch_token`) replaced by socket permissions; `scripts/pod/common.sh hold_card` (flock); `census.write_json_atomic`, `file_sha256`, `runtime_record`; `R3_gpu.gpu_job` / `set_job_cap` as the job frame | `traverse/keeper.py`, about 400 lines |
| **Bench worker** | Loads the runtime once, runs each cell in a worker thread in a fresh namespace with `tv` in scope, clears every hook and restores every parameter afterwards, and proves the base is untouched with a canary forward. | `experiments/lab/labd.py`'s `/exec` loop; `runtime_factory.make_runtime(load_jlens=False)`; `w4a16_fast.install(rt, "compiled")`; `labkit.Lab` by import | `traverse/worker.py`, about 300 lines |
| **`tv`, the library a cell sees** | About twelve names: the runtime itself, lazy base weights, atlas cards, `read`, `write`, `trial`, edit builders, `under`, `keep`/`load`, `save_edit`. | `atlas_ops` (`Weights`, `Interventions`, `resolve`, `installed`, `write_rows`, `forward_from_layer`, `neuron_wiring`, `pair_products`); `causal_check.logits_fp32`, `kl`; `mcp_scientist.paired_statistics`, `benjamini_hochberg`; `census.control_dirs`, `first_departure`, `distinct4`; `AtlasIndex`; census `RowsIndex`/`UnitsTable`; `label_gate.label_check` | `traverse/tv.py`, about 600 lines |
| **Edits** | The edit object, its hash, its file, its three ways of being applied, its report against the 4-bit grid. | `int4_cpu.with_overlay_terms` (the difference-term arithmetic); `gguf_weights.dense`, `dense_cols`, `dense_rows`; `w4a16_fast._make_resident_forward`, `same_bits` (bind from outside, prove by bits); `scripts/check_gguf.py:825-874` (heard, seen, base untouched, undone by bits); `export_lora.write_lora`, `read_lora`, `Plan` | `traverse/edit.py` and `apply.py`, about 750 lines |
| **Verdict worker** | Loads the base fresh, applies one registered edit file in its registered form, scores it on held items with nulls, cheaper knobs and a side-effect panel, applies the registered rule as written, exits. | The registration folder pattern (`experiments/results/census_match_walk/prereg/`: RULES, runner, reader, SHA256SUMS, `prereg_at`, read once); `census_intake`'s draw from `os.urandom` after submission; `census.control_dirs`; `pattern_tests`' kept-back fold | `traverse/verdict.py`, about 400 lines |
| **Faces** | Three thin projections of the keeper's seven calls. | `backend/scientist_mcp.py` (standard-library stdio MCP skeleton); `scientist_chat._claude_cmd`, `child_env`, `reach`, `profiles` (confinement flags and endpoint profiles); `scripts/verify_scientist_cli.py` | `traverse/faces/`, about 150 lines each |
| **Brief** | A page generated at each session start from the ledger and the notes, with measured facts and the driver's own earlier notes kept apart. | nothing; it is new | `traverse/brief.py`, about 150 lines |
| **Checks** | A CPU check on the six-layer stand-in with a replayed driver transcript, and a card check. | `scripts/atlas_stand_in.py`; the `check_*.py` idiom (`check`, `section`, exit code); planted-failure tests as in `check_volunteer_kit.py` section 5 | `scripts/check_traverse.py`, `scripts/verify_traverse.py` |

**Which process holds the model.** The bench worker, a process of its own that takes the card in the
server's place, exactly as the lab daemon and every census or index job do today. I do not put the
bench inside the server. Reasons: driver code must not share a process with the notes, the proposals
and the pages (the keep route has no guard beyond the local-only rule); a runaway cell must be
killable, and the instrument must not be; the worker's sandbox (no network, read-only home) is
incompatible with the server's work; and an edit that never enters the server process can never be
read through the atlas's baseline indexes by mistake. That last point removes a whole class of the
digest's gaps at a stroke (the stats index and the census are served unflagged after a vault edit).

**The server** (`backend/server.py`): untouched in phases 1 to 3. It is down while a traverse holds
the card. One small change is asked of it and of the census wrappers: take the same card lock file,
so that exclusion on the home card stops resting on a port probe. This touches wrappers, not what
writes census answers, so kit 4 is not disturbed. It is the owner's decision.

**The lab daemon**: stays as the study's record. The bench worker is its descendant with the faults
the digest lists repaired: the compiled unpacker by default (labd defaults to the reference, about
seven times slower a step), cells off the event loop so status always answers, a time limit, a cancel,
and model state restored and proved after every cell.

**The vault** (`backend/larql_vault.py`): untouched. A vault statement can be turned into a Traverse
edit (`tv.edit.from_vault("UPDATE ...")` computes the same column and stores it as a delta), which
gives vault edits the things the digest says they lack: a file, a hash, composition, an undo of one
edit, and a comparison with the untouched model inside one forward.

**The scientist** (`backend/scientist_*.py`): untouched, and not the driver's loop. It lives in the
server, its tools are HTTP proxies with replies cut at 16,000 characters, and it has no code execution.
Traverse reuses its tested confinement of a driver CLI and its endpoint profiles, nothing else.

**The atlas and census**: read from their files by the worker, read-only, as a map of the baseline.

**The pinned runtime.** Nothing in `backend/engine_runtime.py`, `theodolite_math.py`, `label_gate.py`
or `backend/__init__.py` is edited; everything is bound from outside, as `w4a16_fast.py` does. The
digest found that R4's sha256 pin on `engine_runtime.py` has not held since 12 November 2026
(f1a0b4c3... against the pinned 69eed1b4...). That should be raised with the owner whatever is decided
here; this design does not depend on it either way.

---

## 2. What an edit is

**An edit is data, never code and never a state of the model.** It is an ordered list of terms. Each
term names one tensor of the base and holds a **delta against the base in float32**:

| Term | Meaning | Covers |
|---|---|---|
| `lowrank(tensor, B, A)` | ΔW = B·A, B [out, r], A [r, in] | a rank-one u·vᵀ; a direction subtracted from chosen columns (B = −dose·u, A = mask ⊙ uᵀW₀, which is rank one: the published Qwen edit) |
| `cols(tensor, idx, Δ)` | chosen columns moved by Δ | a vault column (v − W₀[:, n]); zeroing or scaling a neuron's write |
| `rows(tensor, idx, Δ)` | chosen rows moved by Δ | a neuron's gate or up row; one head's block of q, k or v |
| `entries(tensor, idx, values)` | single weights moved | sparse solver output (the linear-program edit of 16,291 weights) |
| `param(name, idx, Δ)` | a plain bf16 tensor moved | embedding rows, a norm weight, a layer scalar |

`scale(...)`, `zero(...)` and `from_vault(...)` are builders that produce these terms against W₀.
The tensors are any of the 410 quantised linears (attention q, k, v, o and feed-forward gate, up,
down), named by layer and role, plus the plain tensors. The ten full-attention layers (5, 11, 17, 23,
29, 35, 41, 47, 53, 59) have no v_proj and the builder says so. A layer or index out of range is
refused, never clamped as the vault clamps it.

**Composition.** Frozen edits add: `e1 + e2` concatenates terms and the deltas sum; `e * 0.5` scales
every delta. One rule, the same rule llama.cpp applies to adapters served together. Scaling the same
column twice inside one builder multiplies before freezing; after freezing, two scalings add, and the
builder's doc says so.

**Identity.** `edit_id` is the first 16 hex digits of the sha256 of the canonical bytes: the base's
weights card, then for each term the tensor name, kind, index arrays and float32 values. The file is
`edits/<edit_id>.safetensors` with a JSON header: the base weights card, the sha256 of the packed
words and scales of every touched base tensor, per-term sizes (Frobenius norm of Δ, its share of
‖W₀‖, entries, rows and columns touched), the lowest layer touched, parent edits, the run that made
it and the code commit. Loading checks the touched tensors' hashes against the model in memory and
refuses on a difference.

**Three forms of applying it, always named.**

- **side** (the loop's form): a forward hook on the linear adds the delta's product to the output,
  `y[r] += dose[r] · (x[r] Aᵀ) Bᵀ`, for the rows that carry the edit only. This is the arithmetic
  `int4_cpu.with_overlay_terms` already uses for a column, made per row. Rows that carry no edit are
  not written at all, so the base row in the same forward is the untouched model. Each row has its
  own dose, so **a dose sweep costs rows, not forwards**. A weight edit acts at every position,
  start mark included; no keep-mask.
- **folded**: the delta is added into the freshly unpacked bf16 weight of the touched modules before
  the product, by a forward bound from outside and restored afterwards. This is what the vault's
  overlay does today, and what a deployment of "W4A16 base plus a float edit file" computes. It
  applies to the whole batch. `param` terms exist only in this form (an in-place add with a snapshot,
  put back in a `finally`).
- **grid**: the touched blocks' 4-bit codes and scales are rewritten. Not in the loop, ever. Phase 4.

Side and folded round differently (the digest's constraint), so every reading names its form and a
pilot measures how far they part. A verdict is taken in the form the claim names.

**The 4-bit grid.** Edits live off the grid by default, as the vault's do and as QLoRA and Heretic
practice does. What the grid would do to an edit is reported, not assumed. `save_edit` computes, per
term, `grid_keep`: the share of the delta that survives snapping each touched weight to the 16 levels
under its existing scale, leaving every neighbour's code and every shared scale alone. From the
layout (weight = (piece − 8) × scale, one scale per output row per 32 inputs): zeroing any single
weight is exact (piece 8); a gate or up row owns its 168 scales, so scaling that row is a change of
scales alone; a down_proj column shares every scale with 31 other neurons, so scaling it must round
entry by entry; a code of −8 has no negation in range. All of that is derivation and is pilot 8.

**Rollback** does not exist as an operation, because nothing persists. An edit is in force only for
the rows of a call that name it, or inside `with tv.under(e):`. The worker removes every hook and
restores every parameter after each cell, then runs a canary: one fixed prompt at one row, whose
logits must hash to the pinned value of the base. A failed canary kills and reloads the worker
(about 20 s: 13 to 16 s load measured 16 and 27 October 2026, 1.8 to 4.2 s unpacker install).

**A reading under an edit records the edit.** Every record the bench writes carries
`weights: {base: <weights card id>, edit: <edit_id or none>, form: side|folded|grid}` and the model id
gains a suffix `+edit:<id>:<form>`, in the manner of `+int4mm`. No index or census is ever read as a
description of an edited model; atlas cards printed under `tv.under` carry the word "baseline".

**Saved, reloaded, exported.** Saved and reloaded as above. Export is the owner's act, never the
driver's (section 7):

- the edit file itself is the canonical portable form;
- a llama.cpp LoRA through `export_lora.write_lora`, which already takes any tensor name and any
  rank, for `lowrank` and `cols` terms on `ffn_down` and `attn_output` only, because those are the
  two tensors the repository has shown llama.cpp's Gemma 4 graph applies an adapter to; other tensors
  are refused with that reason until checked. A GGUF is a different quantisation from the checkpoint
  in all 410 linears, so an exported adapter is re-measured there with the existing three-level check;
- a grid form, phase 4, with its own tensor hashes (the GGUF fingerprint skips the middle of a file).

**Flags at save, not refusals** (the only refusals are non-finite values, a wrong shape and a wrong
base hash): a whole-tensor positive scalar multiple (the norm undoes it); a touched final norm or
embedding; relative change per tensor; cumulative change per tensor across parents; and a last-block
feed-forward neuron whose write is dominated by one token's unembedding, which is the signature of
the Lazarus implant shown on Gemma-4-31B-it in bf16 (arXiv 2611.15118).

---

## 3. How the driver reads

A language model cannot work with 29.3 billion 4-bit weights as text. No-tools reading has been shown
only to about 172,000 weights, and one audit found a reader leaning on priors (Ahmed, arXiv
2611.08070, which its author says does not refute the original claims). What the driver works with
here is three things, all reached from a cell.

**1. Names and cards (the atlas as the symbol table).** `tv.atlas.neuron(45, 1234)` returns an object
whose printed form is a card of at most about 600 characters, read from files on disk with no forward:
write norm and write ceiling; usual range; the 24 strongest links each way with sizes; census rank and
q where the layer has a census (layers 36 and 40 to 50); steer-lens words where its check says
"agrees" (layers 39 to 52); lens words only where the label gate passes (held back below layer 40);
passage ids; pattern-test results. Each line carries its trust in a word: *sound*, *weak*,
*unchecked*, *held back*. Link signs are printed as "size only: sign from weights was below chance
(0.336, measured)". The card is the driver's first look, and it arrives because the driver touched
the neuron, not because it remembered to call a tool. That is my answer to the tool-to-agent gap:
the precomputed facts ride on the objects the driver already handles.

**2. Its own summaries, computed on the card.** `tv.W.down(45)` is lazy and always the **base**.
`.cols(idx)` and `.rows(idx)` unpack only the blocks needed (`dequantize_w4a16_cols` / `_rows`);
`.dense()` unpacks one matrix, which takes 0.24 to 0.46 ms compiled on the 5090 (measured; date not
stated in the digest) and 231 MB in bf16. So "which neurons in every layer write along direction d"
is 60 unpacks and 60 products, well under a second (arithmetic). I therefore precompute nothing new.
The Qwen agent's recipe (a difference-of-means direction, then the columns that write it) is a
ten-line cell here. Whether a full SVD of a 5,376 × 21,504 matrix fits and how long it takes is
pilot 7; if it is slow, spectra become an index job.

**3. Bounded numbers when it asks.** Any slice it prints, and `tv.W.down(45).codes(rows, cols)` with
`.scales(...)` for the lattice view (integer codes times group scales).

Two mechanical rules shape what reaches the driver's context:

- **The printed tail of a run is capped at 8,000 characters** and a returned `result` at 4,000; the
  rest is on disk in `runs/NNNN/`, and the reply says where. Tensors never travel through context.
  The cap is what makes the driver print summaries; no instruction is needed.
- **Output is sanitised**: word-pieces print as ids with short escaped ASCII. A provider's content
  filter refused a turn that printed raw vocabulary in the published logs.

**Activations.** `tv.read(items, capture=["resid_post@30", "mlp_act@45"], at="last")` returns tensors
on the card by handle, using interp_engine's named points (about 25, plus any module path). And
because the cell has the runtime, the driver can write any hook itself. That freedom is the point:
I do not design a probe language for a model that can write `register_forward_hook`.

**Gradients as attribution.** `tv.lab` carries labkit's recompute-in-backward linears. One 128-piece
prompt cost 0.661 s forward with graph and 0.314 s backward on the reference unpacker at 20.04 GiB
(measured 22 October 2026); four prompts at once ran out of memory under a 24 GiB cap. A session
declares its rule set (`grads: none | attribution`), the ledger records it, and since free code
cannot be stopped from differentiating, the declaration is checked after the fact from the logged
cells, not enforced. Labkit's gradient linear ignores the vault's overlays; whether a side arm
composes with it is pilot 9, and until then a gradient is to be read as the base's.

**What the driver is told about trust, once, in the one-page prompt:** results read from the model's
own odds in fp32 need no lens; lens words are sound only at layers 40 to 57; doses are calibrated only
at 32 to 52; the steer lens agrees with the census only at 39 to 52; a push is larger than a neuron
can write (1.76 to 2.76 write ceilings at layers 39 to 51) so a push is not the neuron firing and not
an edit of its column; weights give a link's size, not its sign.

---

## 4. How a hypothesis is tested, fast

**The fact that shapes everything.** With the compiled unpacker a decode step costs 0.091 s at 1 row,
0.103 at 32, 0.109 at 64 and 0.126 at 96 (measured 28 October 2026), because every forward unpacks
all 29.3 billion weights whatever the batch. Rows are nearly free up to memory. A driver's turn took
20 to 70 s in the published logs. **So on this card the slow part of the loop is the driver, not the
GPU**, and the design aim is fewer turns per hypothesis: each call carries many candidates, their
doses and their controls, and returns one small table.

**Memory.** The worker loads without the J++ lens (lens words come from the index on disk; the lens
loads on request). The model then rests at 17.97 GiB (measured 14 November 2026) on a card where the
desktop held 5.1 GB and left 26.0 (14 November 2026). That is about 8 GiB of headroom (arithmetic). A 64-row batch
of 64 new pieces peaks 4.8 to 5.3 GiB above the model and 96 rows 7.2 to 7.5 GiB (measured 28
October 2026). **Default 64 rows a forward**; 96 when the desktop is light.

**The unit: a paired, teacher-forced read with every arm in one forward.**
`tv.trial(arms, items, metric)` lays out rows as item × (base + arms), pads every forward to one
fixed shape, and reads log-probabilities through `causal_check.logits_fp32` (the fp32 GEMM the study
insists on), in chunks, reduced on the card to: target log-probability or logit difference,
KL(base ‖ arm) per position in nats with non-finite logits masked, and top-1 agreement. One KL
direction, named, everywhere; the instrument today has three. Four things make it fast:

1. **Per-row arms** (side form). Eight arms and a base row for seven items is one forward.
2. **Controls ride along by default.** For each arm with a direction: three same-norm random edits at
   the same site, drawn as `census.control_dirs` draws them (in the layer's write space, because an
   isotropic direction is read about 100 times louder), and the reversed sign. The driver has to opt
   out, not in. Plain Claude Code ran a causal intervention in only 15 of 31 tasks in HyVE; here the
   check is the default and costs rows.
3. **Resume above the lowest edited layer.** `atlas_ops.forward_from_layer` reruns layers L to 59
   from a cached stream. The stream entering layer L for the practice items is captured once per
   (item set, L): 200 items × 64 positions × 5,376 × 2 bytes is 138 MB (arithmetic). An edit at layer
   45 then costs 15/60 of a forward (derivation; pilot 4).
4. **The base arm is remembered** per (item, batch shape, settings, weights id). This is valid only
   if a row's bits depend on its batch's size and not on its companions, which the census determinism
   run found for the census writer on this card (measured) and pilot 3 must confirm for this path.

**Expected costs on the 5090, compiled unpacker.** Measured figures are the repository's; the rest is
my arithmetic and is what pilot 1 replaces.

| Measurement | Shape | Cost | Basis |
|---|---|---|---|
| Scan a direction against all 60 down_proj | 60 unpacks + products | under 1 s | arithmetic from 0.24-0.46 ms unpack (measured) |
| One teacher-forced forward | 64 rows × 64 positions | 0.5 to 1.5 s | arithmetic from prefill 0.45 s at 64 rows (measured 28-29 Oct 2026) and the corpus pass, 2,000,054 pieces in 20 min with every activation hooked (measured 28 Oct 2026) |
| The same from layer 45 | | about a quarter of that | derivation, unmeasured |
| Quick look: 24 items × 8 arms | 3 forwards | about 3 s; under 1 s from layer 45 | arithmetic |
| Full practice trial: 200 items × 8 arms | 25 forwards | 15 to 35 s; 4 to 9 s from layer 45 | arithmetic |
| Greedy writing, 64 answers × 48 pieces | 1 batch | about 5.5 s | 48 × 0.109 s + prefill; agrees with the whole-census rate of 11.88 answers a second (measured 29-30 Oct 2026) |
| Three pushed answers × 80 pieces through the server | | 8.3 s | measured 11 Nov 2026 |
| One attribution pass, 128 pieces | 1 prompt | about 1 s | 0.661 + 0.314 s on the reference (measured 22 Oct 2026) |
| Side-effect panel: 256 windows × 128 positions, edit arm only | | about 20 s | arithmetic from the corpus-pass rate |
| A verdict | fresh process | 5 to 7 minutes | arithmetic, section 5 |
| For comparison: one heard vault edit / one registered census trial | | 12.6 s (undated) / 68 to 86 minutes | measured |

At those costs a driver could run 50 to 150 teacher-forced runs an hour and the card would idle most
of the time (arithmetic). The binding budgets are the driver's turns and the verdict count, not GPU
seconds.

**Writing (free generation)** is `tv.write(items, arms, max_new=48)` on `atlas_ops.write_rows`, with
the same per-row arms and a base row of the same batch size in the same forward. It is the costly
exception in the loop and the rule in the verdict, because teacher forcing is the most severe of the
flaws that inflated edit success from 38.5% to 96.8% in the model-editing literature. Layer-resumed
writing does not exist in the repository and I do not build it.

**What is cached:** the stream entering a layer for the practice items; base-arm results; anything
the driver passes to `tv.keep(name, tensor)`, which goes to `scratch/` as safetensors and survives a
worker restart. Variables do not survive between cells; `tv.S` is an explicit store that does until a
restart. Kernel memory is a convenience and never the truth.

**Friction the worker removes without being asked** (each from a fault the digest records): it warns
in the reply's footer when a cell made the compiled unpacker fall back (`torch.inference_mode` or
autocast makes a step about seven times slower, silently); it halves the batch on out-of-memory with
a base row kept in every forward, as `causal_check` does; it trims tracebacks to the cell's own
frames; a run that died from the worker's own failure is free and is re-run by its code hash.

**What a turn looks like.** A cell (bracketed values stand for numbers not measured):

```python
# say: columns that write along she-minus-he at layer 50 carry the pronoun pull; forecast: zeroing ten moves it by more than 1
it  = tv.items("practice")
d   = tv.load("she_minus_he_L50")                     # kept by an earlier run
top = (d @ tv.W.down(50).dense().float()).abs().topk(10).indices
arms = {"zero10": tv.edit.zero("down", 50, cols=top),
        "half10": tv.edit.scale("down", 50, cols=top, factor=0.5)}
print(tv.trial(arms, items=it, metric="logit_diff"))  # base, both arms, three random and one reversed arm each
```

and its reply:

```
run 0012 ok [s] s | card [m] of 180 min | verdicts 3 left | no edit in force
arm      mean paired diff   se    better/worse   3 random same-site   reversed   panel KL
zero10        [..]         [..]     [..]/[..]      [..] to [..]         [..]       [..]
half10        [..]         [..]     [..]/[..]      [..] to [..]         [..]       [..]
practice items, teacher-forced, side form: a lead, not a result. full output: runs/0012/
```

---

## 5. What stops the driver fooling itself and the owner

The loop is unguarded by choice. The guards sit where a number becomes a claim, and they are
mechanisms, not sentences in a prompt. Rules stated in a prompt did not hold for agents with free
code in any source the digests cite.

**Three tiers of items, split by where they live.** *Practice* items are in the workspace; the driver
may read and fit them freely. *Confirm* and *sealed* items live in the keeper's `held/` folder, which
is not mounted into the bench worker's sandbox and is outside the driver CLI's folder reach. Only the
verdict worker opens them. The driver receives aggregates, never an item.

**A verdict is run by a process that has never run a cell.** The keeper stops the bench worker,
starts a fresh verdict worker, and that process loads the base, loads the edit **as data** (a
safetensors file and JSON; no pickles, no code), and runs the task's registered evaluator. Then it
exits and the bench worker reloads. A re-score after a fresh-process reload is the literature's
advice for any claim headed to a sealed set; here it is the only way a claim is scored at all.

**Registration, in the house form.** `register` takes a short claim file: the edit id and form, the
task, the metric, the decision rule, and the driver's forecast. The keeper adds what had been seen
(runs so far, practice figures printed, earlier claims on this task), then writes a folder shaped
like `census_match_walk/prereg/`: RULES, the exact evaluator, the item-set hashes, the edit file's
hash, `repo_commit`, `prereg_at`, SHA256SUMS. The rule is applied as written; a figure of 0.599
against 0.60 fails. A claim is read once. A second try is a new claim and its number is printed
beside the verdict.

**What the verdict worker always measures** (the driver cannot switch any of it off):

- the paired gain on confirm items, base and edit rows in one forward, teacher-forced and by free
  greedy writing side by side;
- **19 same-norm random edits at the same site** (so the smallest empirical p is 1 in 20), a
  same-size edit at a random site, the reversed sign and doses 0.5, 1 and 2. A random direction on
  the same neurons recovered about a fifth of the published Qwen gain; one random draw bounds nothing;
- **the noise floor of the day**: the base run twice in different batch compositions. Different batch
  shapes differ by KL up to 0.014 (measured, the study), and an unpushed answer in a batch of 64
  parted from itself written alone after 22 word-pieces (measured 28-29 October 2026), so no
  comparison is ever made across shapes and a gain must clear the measured floor;
- **cheaper knobs**: the task pack's prompt-sentence and decoding baselines, the edit's activation
  twin where one exists (`mul` is the exact per-row twin of scaling a column), and a no-driver
  scripted recipe (difference of means, top writers, three doses). A prompt sentence, a thinking
  toggle, a token ban and a steering vector each matched or nearly matched a published weight edit.
  These are reported beside the verdict, and the claim's wording must carry them;
- **the side-effect panel**: mean KL to the base and top-1 agreement on 256 kept-back corpus windows
  (the odd-numbered windows of the passages index, which already serve as a kept-back fold), text
  health from `census.distinct4` and the cut-off share, a *near* set from the task pack (items close
  to the target that must not move), and norm growth per touched tensor. Gain and retention are
  shown side by side, never multiplied into one number.

Cost, by arithmetic from measured rates: about 20 s to load, about 100 s for 300 items × 21
teacher-forced arms (25 s for a layer-45 edit), about 200 s for 300 items × 8 written arms at 11.88
answers a second, about 40 s of panel, 20 s to hand the card back: **5 to 7 minutes**.

**Rationed feedback.** A confirm verdict returns the word, pass or fail per rule line, the gain
rounded to units of its standard error, the nulls' range and the flags. A confirm set gives **at most
five verdicts in its life**, then it is spent and the owner rotates it. A sealed set gives one
reading per claim and only the owner can ask for it. Development figures exceeded held-out figures
in all 14 tabulated runs of Hu et al. (arXiv 2611.09382), partly by construction; fixed query budgets
are the cheapest known answer.

**The ledger.** One hash-chained JSON line per run, saved edit, claim, verdict and session event,
written only by the keeper, outside every sandbox. Each run line has the driver's `say` (its
hypothesis in a sentence, required, one line), the sha256 of the cell and of every workspace file it
imported, the seconds, the edits it named, the weights id, the unpacker, the batch shape, the
commit. The full cell and output are kept. Failed and abandoned attempts stay. The driver reads a
mirror of it; it cannot write it.

**What a result is allowed to claim.** Only a verdict line may be quoted, in these words: *held*,
*not held*, *unmeasured*; and a session may end *kept the base*, which is an ordinary ending (6 of 16
trajectories in Hu et al.). A held claim says exactly: under edit E in form F, metric M on N confirm
items moved by x ± se against the base in the same forward; k of 19 same-norm random edits did as
well; these cheaper knobs moved it by this much; the panel moved by this much. It may not say what a
neuron *does* (that needs the pattern tests), that an ability was added, anything about the grid form
unless scored in the grid form, or anything about a self-written task beyond "on items the driver
wrote". The driver's prose report is generated around a skeleton of ledger figures it cannot alter,
and it is a **proposal** until the owner keeps it, like a name or a note.

**Memory without false confidence.** The brief prints two blocks under two headings: *measured* (from
the ledger: verdicts, saved edits, the last fifteen runs) and *your earlier notes, not checked*. An
auditing agent grew falsely confident from its own memory files after resets; the fix here is the
house habit of marking every figure measured or said.

**A positive and a negative control for the bench itself.** The first verdict the bench ever gives is
a known answer: the study's one causal edit test, zeroing the ten 'she' neurons at layer 50, which
moved the female-context log-odds by −1.80 against at most 0.04 for ten random neurons
(`E6_tool_audit_REPORT.md`, on twelve matched name-to-pronoun contexts). Twelve are too few to split,
so the first task pack is a few hundred contexts written in that form by the owner and split by the
keeper. The bench must return *held* for the ten neurons on it and *not held* for ten random ones.
If it does not, either the bench or the old figure is wrong, and that is settled before anything
else the bench says is believed.

**What I leave to the validity lens:** decoy weights served unannounced, canary tasks with a known
null, a judge from another model family. They are reasonable and unproven as needs (the checker's
word on decoys). The verdict path has no model judge in it at all: every figure is computed.

---

## 6. The agent interface

**The protocol is seven calls on the keeper's socket.** Everything else is a face.

| Call | Does |
|---|---|
| `run {code or path, say, limit_s}` | queues a cell, waits up to 100 s, returns the capped tail, the `result`, and the footer; if still running, returns a handle |
| `wait {run}` / `cancel {run}` | follow or stop a run |
| `status {}` | card, budgets, what is running, whether any edit is in force (always "none" between runs) |
| `register {claim}` / `verdict {claim}` | section 5 |
| `brief {}` | the regenerated first page |

Freezing an edit happens inside a cell (`tv.save_edit`), and the keeper records it.

**Three faces, each about 150 lines, none holding any logic:**

- **MCP, stdio**: seven tools whose schemas come to under 2 KB, against the scientist's 39,051 bytes
  (measured 14 November 2026). This is the face for Claude Code and Codex at the *folder reach*, where
  they have file tools in the workspace and no shell. The server is `scientist_mcp.py`'s skeleton
  pointed at the socket.
- **A command line**: `traverse run cells/0012.py --say "..."`. For a person, and for an attended
  driver that has a shell.
- **One fenced block per reply** (a block fenced as `python`, `note`, `register`, `verdict` or
  `done`): for any chat model with no tool calling, driven by a small loop that reuses the
  scientist's endpoint profiles. This is the protocol the published harnesses proved with one model;
  any model can speak it, and nothing shows any model can do the work. The same loop replays a logged
  transcript with no model, which is how `check_traverse.py` runs every face on the CPU stand-in.

**How a current frontier model plugs in today.** Phase 1: the owner's own Claude Code session, with
a shell, runs `traverse run`. That is the trust the lab daemon had for 314 jobs, and nothing new.
Phase 3: `traverse session start --driver claude` launches the CLI with the scientist's tested
recipe (`_claude_cmd`: `--tools Read Write Edit`, `--disallowedTools Bash`, `--strict-mcp-config`,
`--setting-sources ""`, a prompt file, `child_env` stripping credentials), its folder being the
workspace. No settings file and no CLAUDE.md are loaded, because a CLAUDE.md the scientist wrote once
reached every later chat as instructions. The driver's notes come back to it as *data under a
heading*, never as instructions.

**How a stronger one plugs in without a rewrite.** It gets the same seven calls. What changes is in
its favour and costs us nothing: it writes better cells; its vendor's harness handles its context,
its planning and its sub-agents; it can read the whole raw ledger instead of the brief; it can build
its own helper library in `lib/`, which the worker imports and the ledger hashes. If its harness
speaks something other than MCP, a face is 150 lines. The freedom it will want is already there,
because the interface was never a list of things we thought it needed.

**The workspace is the memory.** One folder per traverse (a line of inquiry), many sessions in it:

```
~/.local/share/theodolite/traverse/
  held/<task>/                 confirm, sealed, near        keeper only
  ledger/<traverse>.jsonl      hash-chained                 keeper only
  work/<traverse>/             the driver's whole world
     BRIEF.md   API.md   METHOD.md   examples/   task/      written by the keeper, read-only
     notes.md   cells/   lib/                               the driver's own
     runs/NNNN/{cell.py, stdout.txt, result.json, meta.json}
     edits/<id>.safetensors + .json     scratch/     claims/     verdicts/
     ledger.view.jsonl   card.json
```

Sessions start fresh and read `BRIEF.md` (at most about 6,000 characters). A fresh session with a
memory file worked for Anthropic's automated researchers; raw history on disk beat summaries in
Meta-Harness, while trace spans were requested 16 times by 7 of 111 cells in HarnessOpt-Bench, which
its authors say does not establish that reading them is unnecessary. So the brief is short,
the raw record is all there, and nothing is forced. Nothing crosses between traverses except what the
owner has kept.

**The fixed prompt is one page**, about 600 words: what the object is; the one way to act; the twelve
names; three physical facts (rows are nearly free to 64; compare only inside one call; output is
capped, so print summaries); the trust lines of section 3; what counts (only verdicts; keeping the
base is a fine ending); where state lives. It teaches no method, orders no workflow and names no
suspected fault. The published headline result is disputed in part because its prompt named the
failure. `METHOD.md` (the scientist's methodology) and six example cells sit in the workspace for a
driver that wants them. The prompt, the task pack and the library commit are hashed into the ledger
at session start, so what the driver was told is part of every result.

**Budgets**, shown in the footer of every reply:

- **card time**: wall-clock the keeper holds the card for this session (default 3 hours). This is the
  unit the owner actually gives up, since the server is down;
- **a limit per cell** (default 120 s, up to 900 s when the call says why);
- **verdicts** (default 3 a session, 5 in a confirm set's life);
- **tokens and dollars** are the driver harness's to cap. The fenced loop records usage because it
  sees it; the other faces record what the CLI reports. I state no dollar figure: the same published
  run has been priced at about $15 and at $312.

Reasoning effort is the driver harness's setting, recorded in `card.json`. On IWIBench medium effort
scored highest and high beat low by a small margin at 12.8 times the reasoning tokens, so I set no
default above medium and claim nothing more.

**The harness card.** `card.json` per session: the driver as actually served, effort, face, prompt
hash, task hash, library commit, budgets, reach, rule set, unpacker, card class, weights id. Harness
variance was measured at 7.8 times model variance on one benchmark (Zhang et al.), while on another
the model mattered 1.8 times more than the harness (HarnessOpt-Bench); either way a result from this
bench is a fact about a driver in this harness and says so.

**What I refuse to build, because the next driver will not need it or the current one's vendor
already ships it:**

1. A chat runtime, turn loop, context reset or hand-off for CLI drivers.
2. A menu of reading tools. The 53 atlas routes' functions are importable in a cell as they are.
3. Ordered stages (observe, hypothesise, validate) or sending back unexplained actions. The `say`
   line is required because the owner reads the ledger, not to discipline the driver.
4. Councils, critics or a second agent in the loop. Councils did not beat their strongest member on
   any natural task in wsplab.
5. Summarisers, retrieval or a hypothesis tracker. `notes.md` is the driver's and free-form.
6. A probe language. Hooks are Python.
7. A solver tool for "which columns and how much". Once the fault and the layers were known, the
   step that took the published agent about eight hours took a linear program about a minute; here
   the driver has scipy and torch on the card and one example cell that fits columns by least squares.
8. Renderers of weights as text, and any new page before phase 4.

**What I keep although it is scaffolding, with its assumption written down.** A file `SCAFFOLD.md`
lists every convenience that is not a trust boundary (the brief, the output cap's size, the default
batch, the `say` line, the default controls in `tv.trial`, the sanitiser) with the assumption about
the driver it encodes and the measurement that would retire it. `--bare` starts a session without
them, so their worth can be measured when the driver changes. Deleting scaffolding on a schedule is
the one piece of harness advice every source agrees on.

---

## 7. Safety and scope

**Containment is by the environment, not by approval.** Users approved about 93% of permission
prompts in Anthropic's data; a monitor was switched off by an environment variable in METR's.

- **The bench worker** runs under bubblewrap inside a systemd scope (`/usr/bin/bwrap` and
  `/usr/bin/systemd-run` are on this machine, checked 14 November 2026): no network namespace; the
  repository, the checkpoint, the atlas and the virtual environment read-only; only `scratch/`,
  `edits/` and a temporary folder writable; the card's devices bound in; credentials stripped as
  `child_env` strips them; a memory and time limit on the scope. It reaches the keeper by a pipe. It
  cannot fetch, install, send, or write to the harness, the evaluator, the ledger, the prompt, or any
  agent configuration. Whether the card and `torch.compile` work inside is pilot 10. If they do not,
  the fallback is a separate OS user, which is also the stronger way to hold the held items.
- **The driver CLI** runs at the scientist's folder reach, verified by `scripts/verify_scientist_cli.py`,
  which today has been run against Claude Code 2.1.312 only. The keeper refuses to start an unattended
  session with a CLI version that has not passed it.
- **Until both exist (phases 1 and 2), sessions are attended**: the owner's own session, at the
  owner's terminal.
- **Untrusted text.** What the edited Gemma writes, dataset strings, census answers and the driver's
  own old notes can all carry instructions. They reach the driver as capped, sanitised data. At the
  folder reach the driver has no shell and no network of its own to act on them, and a cell they
  talk it into writing runs in the sandbox.

**What the bench will not do.**

- It never writes to the checkpoint, the Hugging Face cache, an atlas index or a census run, and
  never requantises inside the loop.
- It never commits, pushes, exports or publishes. `keep`, `unseal` and `export` are commands of the
  owner's terminal and are not in any face.
- It ships no task pack and no example for stripping refusals or planting a trigger, and no
  trigger-to-payload recipe. Free code means a driver could write one; scope is held by the task
  packs the owner supplies, by the save-time flag for implant-shaped edits, by the logged cells, and
  by the driver vendor's own policy. I say plainly that this is weaker than a mechanism.
- It does not train. A gradient loop over an edit is outside the declared rule sets.
- It runs nothing from a third-party file. A checkpoint or adapter from elsewhere is compared with
  its declared base tensor by tensor before it is ever the object (the streaming comparator in
  `scripts/compare_gguf_ct.py` has the pieces).

**What happens to edited weights.** An edit is a file of kilobytes to megabytes bound to the base's
hash. It is never merged into the base. It stays in the git-ignored data folder; only its small
record (claim, verdict, hashes) may be committed under `experiments/results/traverse/<claim>/`, by the
house rule. A kept edit that leaves the machine carries a manifest: base id and hash, edit id, the
claim and its verdict with nulls and discarded attempts, a prominent statement that the weights were
modified, the Apache 2.0 text, and a name that says "derived from Gemma", not "Gemma". Provenance is
not assurance and the manifest says that too. Whether anything leaves the machine is the owner's
decision; no policy exists today.

---

## 8. A phased path

Each phase is useful if the next is never built. Effort is in focused days of the kind this project's
sessions produce, and is rough.

| Phase | Delivers | Proves | Effort |
|---|---|---|---|
| **0. Pilots** | The measurements of section 9, as scripts under `experiments/results/traverse_pilots/`, their decision thresholds hashed before any is run. No product code. | Whether the cost table, the per-row arms and the free-cell bet hold on this card. | 2 to 3 evenings of the card, one attended afternoon |
| **1. A cell runs** (small) | Keeper, bench worker, the command line, the ledger, the brief; `tv` with the runtime, base weights, atlas cards, `read` and `write` using the per-row pushes that exist today (`mul` is already the exact twin of scaling or zeroing a column). Attended only. `check_traverse.py` on the stand-in. | A current driver completes a known-answer inquiry (the 'she' neurons at layer 50) with status, stop and a record; turns, seconds and failed cells are counted. The lab daemon becomes something that can be handed to an agent. | 3 to 4 days |
| **2. Edits and trials** | The edit object and file, side and folded application on any of the 410 linears and the plain tensors, `tv.trial` with default controls, resume from a layer, the base memo, the grid report as numbers, `verify_traverse.py` with the four-part edit check on the card. | The measured speed table; side against folded; the vault's listed gaps (delta form, composition, a file, one-edit scope, edit and base in one forward, attention and gate edits) closed as a library. | 5 to 7 days |
| **3. Claims and containment** | Task packs and held tiers, `register` and the verdict worker with nulls, cheaper knobs, the scripted recipe and the panel; the sandbox; the MCP face and the confined launch; budgets; the report as a proposal. | The positive control returns *held* and its random twin *not held*; a planted cheat (a cell that tries to read held items or write the ledger) fails; an unattended session ends in a verdict the owner can check line by line. | 7 to 10 days |
| **4. Carry-out and a page** | LoRA export of an edit with re-measurement through llama.cpp; the grid form with its own verdicts; a read-only page built from the ledger's files (sessions, edits, verdicts, keep or discard) in the kit's tokens with Lexicon entries; the server opening one kept edit under its own model id; the fenced face. | An edit's effect in the form that ships; the owner can follow and decide without a terminal. | 5 to 7 days |
| **5. Repeats and a second object** | N independent sessions per task with their spread; a second driver family on the same tasks; the trained Gemma 4 12B QAT GGUF on disk as a cheaper object, if `build_gemma4` builds it. | Session-to-session variance (+29.45, +5.00, +27.30 in Hu et al.) on this bench; whether the plain protocol carries across drivers. | open |

The first product increment is phase 1 and it is deliberately the lab daemon made safe to hand over,
plus a ledger. It needs no new edit machinery at all.

---

## 9. Pilots, in order, before committing

Each has its threshold fixed and hashed beforehand, in the house manner. The order is by how much of
this design a result would overturn.

1. **Time and memory of the unit.** Teacher-forced forwards at (rows, positions) of (32, 64), (64, 64),
   (96, 64), (64, 128), (16, 512), with the fp32 reduction, without the lens, compiled unpacker.
   *Changes the design if* a 64 × 64 forward costs more than about 3 s: then the card, not the
   driver, is the slow part, trials shrink, and the speed lens's caches (layer-resumed writing, a
   scheduler) are worth building after all.
2. **The driver pilot.** One attended afternoon with the existing lab daemon (compiled unpacker) and
   a 150-line helper, on the known-answer task and one open task, run twice: once with free cells,
   once with the same driver allowed only five typed verbs. Count cells an hour, the share of cells
   that fail on helper friction, the driver's seconds a turn, whether controls were run unprompted,
   whether the atlas cards were used, whether notes were kept. Then the same with a second driver.
   *Changes the design if* more than about a third of free cells fail on friction, or the typed arm
   reaches the known answer in clearly fewer turns: then the verbs become the default face and cells
   the escape hatch. This is the pilot my whole lens rests on.
3. **Per-row side arms.** A side-delta hook on one down_proj at 64 rows: are the rows without an edit
   bit-identical to a forward with no hook at the same shape (`w4a16_fast.same_bits`); is a base row
   bit-identical across two calls with different companions; what does the hook cost with 1, 8 and 64
   terms. *Changes the design if* base rows are not bit-identical: then the base memo goes, every
   forward carries its own base row and a measured floor replaces the equality.
4. **Resume from a layer.** `forward_from_layer` against a whole forward on the 31B (the existing
   check is on the stand-in only): equal by bits or by how much, and the speed at layers 15, 30, 45.
   *Changes the design if* it is not exact or saves little: then the stream cache is dropped and
   every trial is a whole forward.
5. **Side against folded.** The same edit both ways on 64 items: largest difference in
   log-probability, against the batch-shape floor (KL up to 0.014, measured). And the cost of the
   folded forward for 1, 8 and 64 columns (the digest records that an overlay's cost per forward has
   never been measured). *Changes the design if* they differ by more than the floor: then the loop
   ranks in side form and every candidate worth saving is re-read folded, one forward each.
6. **Effect size and item counts.** Per-item paired differences for the 'she' edit and for a weak
   edit: how many items give three standard errors. *Changes the design if* a small effect needs more
   than about a thousand items: then confirm sets grow, a verdict takes tens of minutes, and the
   verdict budget falls to one a session.
7. **Weights on the card.** Unpack, Gram matrix, eigen-decomposition and a full SVD of one down_proj
   in fp32: seconds and peak GiB. *Changes the design if* the SVD does not fit or takes more than a
   minute: then spectra of the 60 down_proj become a precomputed index.
8. **The grid.** For three edits (zero a column; a rank-one direction subtraction on 64 columns at
   dose 1; scale a gate row by 2): `grid_keep`, and behaviour in side form against a snapped copy on
   64 items. *Changes the design if* typical rank-one edits keep less than half: then the bench ships
   "base plus a float edit file" only and phase 4's grid form is dropped; *if* zeroing and row
   scaling are exact, they are advertised as the grid-exact class.
9. **Attribution.** One backward pass under the compiled unpacker without the lens, with a side arm
   installed. *Changes the design if* it fails: attribution runs on the reference unpacker only, or
   is left out.
10. **The sandbox.** The bench worker under bubblewrap and a systemd scope with the card: does it
    load, does the compiled unpacker install and pass its self-test, what does a step cost.
    *Changes the design if* not: a separate OS user replaces the namespace.

Pilots 1 and 3 to 9 can share one evening through the existing lab daemon. Pilot 2 needs the owner.

---

## 10. Risks, what is left out, and the owner's decisions

**Risks.**

- **Nothing worth a verdict may be found.** The published successes removed an obstacle on small or
  damaged models; held-out gains on a strong 9B base were +1.5 to +3.0 points where any edit was
  kept; above that size the digests found only community work (Qwen3.8-27B three times, one Gemma 4
  12B session of about two hours) and nothing on the 31B. The bench is to be judged on how fast and
  how honestly it tests a hypothesis, and *kept the base* will be its commonest ending.
- **The free-cell bet is wrong for a given driver.** Pilot 2; the fallback is cheap.
- **Leads leak into prose.** Loop figures are optimistic by design. The guard is the vocabulary rule
  and the report skeleton; a careless reader of the ledger can still quote a practice number.
- **Self-written tasks leak.** When the driver writes the items, the keeper's random split after
  submission hides them from later sessions but not from the session that wrote them. Verdicts on
  such packs are labelled and limited.
- **The sandbox with a GPU may be awkward**, and the reach rules must be re-verified at each CLI
  release.
- **Notes harden.** The measured-or-said split reduces it and does not remove it.
- **Session variance.** A single session is unreliable; repeats are phase 5.
- **A driver's vendor may refuse or filter**: a content filter refused printed vocabulary in the
  published logs, the model API was unreachable for 4.7 hours, and 18 of 69 forward requests failed
  on the harness side. Runs are idempotent by code hash and free when the bench fails.
- **One card.** A traverse takes the card from the server and from census work for hours.
- **The lens-free rule narrows the map.** Thirty-nine layers have no calibrated dose and lens words
  are held back below layer 40, so below the middle of the model the driver has passages, weights and
  the model's own odds, and little else to lean on.

**Left out on purpose.** A page before phase 4; any change to the scientist's loop; a general overlay
kernel inside the runtime; a queue beyond first come, first served; layer-resumed writing;
precomputed spectra; decoy weights and canary tasks; a model judge; per-row application of
embedding, norm and scalar edits (they are whole-batch, so K candidates cost K forwards);
requantising in the loop; a model-neutral object protocol (the runtime stays Gemma-4-shaped, and
`tv` reads shapes from the model so that a 12B object is at least not blocked by constants);
training.

**The owner's decisions.**

1. Whether driver code may run on the card at all (free cells), or only typed verbs. This is the
   choice the design turns on.
2. The name.
3. Whether the server and the census wrappers take a shared card lock, and whether the keeper may
   stop and restart the server itself.
4. Whether gradients as attribution are allowed, and whether results under them are to be compared
   with the published work, none of which draws that line.
5. The form that counts: the W4A16 base plus a float edit file (my default), or a pure 4-bit object.
6. The first task packs, who writes held items, and whether a self-written pack can yield a
   quotable verdict.
7. Card time per session, and which driver versions may run unattended.
8. A separate OS user for the held items and the worker.
9. What is out of scope by policy (safeguard removal, triggers), and whether any edit, adapter or
   record is ever published.
10. What to do about the broken R4 pin on `backend/engine_runtime.py`.
11. Whether the instrument's server may open a kept edit (phase 4), and under what staleness rule.

---

## 11. Names

In the house vocabulary of surveying, printing and manuscripts. None is used in the repository today
(checked 14 November 2026; "station" is taken by the frontispiece and "field book" by the notes).

1. **Traverse** (my choice). A surveyor's traverse is a run of legs from a known point: sight,
   measure, move on, and at the end *close* on a known point to find the error gathered on the way.
   A traverse that does not close is not a survey. A line of inquiry is a traverse, each run a leg,
   the verdict its closing, the gain against the nulls its misclosure. The word carries the design's
   one rule: nothing counts until the traverse is closed.
2. **Plane table.** The surveyor's board set up in the field, on which the map is drawn while the
   sights are taken. It names the bench as a place: the object, the instrument and the drawing on one
   table.
3. **Galley.** In printing, the tray of set type from which rough proofs are pulled for correction
   before anything goes to press. It names the loop's standing: everything in galley is provisional.

Whichever is chosen needs a Lexicon entry with a stable anchor for each new term (edit file, claim,
verdict, held items, ledger), the literature's term beside the plain one, and no method name the
thing is not: an edit here is not MEMIT, and a verdict is not a proof.
