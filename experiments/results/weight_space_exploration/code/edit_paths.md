# Every existing way to intervene on or edit the model

Read at commit `e6ac64f` (main, clean) on 14 November 2026. Read only: no file in the repository was changed, no model
was loaded. Paths are relative to `(the repository root)`. "Code" means I read the lines cited; "doc" means a
document says it and I did not check it against code or a run.

Files read in full: `backend/larql_vault.py`, `backend/control_vector.py`, `scripts/export_lora.py`,
`scripts/export_control_vector.py`, `scripts/check_lora_export.py`, `scripts/verify_atlas_writer.py`,
ARCHITECTURE.md §4.6 (595–642) and §6 (1552–1859). Read in the parts that answer the questions:
`backend/atlas_ops.py` (1–720, 1010–1045, 1640–1735, 1918–1972), `scripts/check_control_vector.py` (1–150 and its
outline), and, because the forward that applies an edit lives there, `backend/engine_runtime.py` (284–442, 1027–1119,
1280–1310, 1351–1488, 1666–1750), `backend/gguf_weights.py` (1–60, 372–560), `backend/w4a16_fast.py` (1–330, 500–580),
`backend/int4_cpu.py` (138–164, 260–300), `backend/gguf_runtime.py` (164–253), `backend/causal_check.py` (1–60,
229–338), `backend/atlas.py` (186–300, 338–397), `backend/server.py` (296–386, 655–676, 1444–1466),
`experiments/lab/labkit.py` (267–313, 410–480) and `experiments/lab/labd.py` (1–34).

---

## 1. The vocabulary of interventions and edits today

There are two families and they never meet in one data structure:

- **Activation interventions**, installed as torch hooks for the length of one `with` block and gone when it ends.
  They are per batch row, so a pushed row and an unpushed row can share one forward.
- **Weight edits**, of which there is exactly one kind: a replacement of one column of one layer's `down_proj`, held
  as a dense vector in a dict on the module and laid onto the unpacked weight at every forward. They are for the whole
  batch and last until ROLLBACK or the process ends.

### 1.1 Table

| Name | Mathematics | Data structure | Installed by | Composes | Undone by | Changes |
|---|---|---|---|---|---|---|
| push `add` | stream leaving layer l gains `sign × dose × p50_l × u`, `u = unit(w_post ⊙ W_down[:, n])` | `Push(layer, neuron, "add", dose)` → `Interventions.adds[l]` `[B, D]` float32 | forward hook on `lm.layers[l]` (`atlas_ops.py:384-392`) | adds at one layer sum (`atlas_ops.py:306`, `329-330`) | hook removed in `finally` (`atlas_ops.py:410-412`) | activations |
| push `mul` | `a_n ← f × a_n` at `down_proj`'s input; f = dose for sign +1, 1 for 0, −dose for −1 | `Interventions.muls[l]` list of `(neuron or None, [B] factors)` | forward pre-hook on `mlp.down_proj` (`atlas_ops.py:369-383`) | several muls of one neuron multiply in turn | hook removed | activations |
| push `act` | `a_n ← a_n + sign × dose × act_scale(l, n)` | `Interventions.acts[l]` list of `(neuron, [B] values)` | same pre-hook, after every mul | acts add | hook removed | activations |
| silenced block | `mul` with `neuron=None`, dose 0 only: the whole activation vector × 0, then the norm of 0 is 0 | `Push(layer, None, "mul", 0.0)` | same pre-hook | — | hook removed | activations |
| head scale | head h's slice of `o_proj`'s input (`h·hd … (h+1)·hd`) × f | `HeadScale(layer, head, scale)` → `Interventions.heads[l]` | forward pre-hook on `self_attn.o_proj` (`atlas_ops.py:393-408`) | several scales of one head multiply | hook removed | activations |
| every head silenced | `HeadScale(layer, None, 0.0)`, 0 only | same | same | — | hook removed | activations |
| causal check, `activation` | `mlp_act[l, p, i] += s × k × max|a_i|` at one position p | arguments of `causal_check.run_check` | pre-hook on `down_proj`, prepended (`causal_check.py:307-310, 321`) | one neuron a call | hooks removed in `finally` | activations |
| causal check, `frozen_norm` | post-FFN norm's output at p gains `delta × (w_post ⊙ W_down[:, i]) / rms(y_p)` | same | forward hook on `post_feedforward_layernorm` (`causal_check.py:312-317, 323`) | one neuron a call | hooks removed | activations |
| the sweep's nudge | a vector added at `resid_post` of one or more layers: absolute length `scale` (`AddSpec`) or `strength × ‖h‖` per position (`NormScaledAddSpec`) | interp-engine `SteeringSpec` (`server.py:324-382`, `engine_runtime.py:1351-1373`) | interp-engine's `steer()` context (`engine_runtime.py:1464-1488`) | one op per layer, several layers | context exit | activations |
| lab kit hooks | `ablate_direction` (project a unit direction out of `resid_post[l]`), `steer_add` (add `alpha × dir`, optionally × ‖h‖) | closures on `Lab._hooks` (`experiments/lab/labkit.py:272-313`) | forward hook on the decoder layer | any number | `Lab.clear_hooks()` by hand | activations |
| vault DELETE | `W_down[l][:, n] ← 0` | `down_proj._memit_col_overlays[n] = zeros` + a `PatchOperationRecord` | the linear's own forward lays it on (`engine_runtime.py:356-375`) | latest edit of a neuron replaces the earlier | ROLLBACK (all edits at once) | weights (as an overlay) |
| vault UPDATE | `W_down[l][:, n] ← orig + pol × strength × max(‖orig‖, 0.25) × unit((W_U[t] ⊙ w_final) · J_l)` | same dict, same record | same | same | ROLLBACK | weights (as an overlay) |
| vault INSERT | UPDATE on the neuron whose `gate_proj` row is nearest the subject's pulled-back direction | same | same | same | ROLLBACK | weights (as an overlay) |
| vault ROLLBACK | clears every overlay on `down_proj`, `gate_proj`, `up_proj` of all 60 layers and every record | — | `larql_vault.py:556-567` | — | — | weights |

### 1.2 Pushes and head scales in detail (`backend/atlas_ops.py`)

**Where a push lands in Gemma 4's layer.** The layer is
`mid = pre + post_attn_norm(attn(input_norm(pre)))`, `post = s × (mid + post_ffn_norm(mlp(pre_ffn_norm(mid))))`, every
norm `x / rms(x) × w` (`atlas_ops.py:8-13`). So a change of one activation passes through a norm that squashes a large
one, and "add" was placed after the norm and the scalar for that reason (`atlas_ops.py:15-21`).

**Data structures.** `Push(layer, neuron: Optional[int], mode="add", dose=0.8)` and
`HeadScale(layer, head: Optional[int], scale=0.0)` (`atlas_ops.py:215-227`). `resolve(weights, pushes, heads, signs,
p50, act_scale)` turns them into `Interventions(rows, adds, acts, muls, heads)` with one value per batch row, one row
per entry of `signs` (+1 as asked, 0 left alone, −1 the other way) (`atlas_ops.py:267-315`). The three voices of the
push chat are `THREE_VOICES = (1, 0, -1)` (`atlas_ops.py:49`).

**Exact arithmetic.**
- `add`: `delta = (sg × d × p50(l))[:, None] × u[None, :]`, with `u = Weights.write_dirs(l, [n])[0]`, that is
  `v = down_cols × w_post; v / ‖v‖` (`atlas_ops.py:186-190, 303-306`). The hook adds
  `keep[:, :, None] × delta[:, None, :]`, cast to the stream's dtype (bfloat16 on the checkpoint), to the decoder
  layer's output (`atlas_ops.py:386-390`).
- `mul`: the factor is `d` where the sign is +1, `−d` where it is −1, 1 where it is 0 (`atlas_ops.py:310`). A whole
  block (`neuron=None`) takes only dose 0, because the norm after the block undoes any other uniform scale
  (`atlas_ops.py:244-249`).
- `act`: `sg × d × act_scale(l, n)` added to the activation (`atlas_ops.py:307-308`); the atlas gives
  `act_scale = max(|pos|, |neg|, 1e-6)` from the stats index's extremes (`backend/atlas.py:365-369`).
- head scale: the factor is `scale` wherever the sign is not 0, and it is not negated on the −1 row
  (`atlas_ops.py:312-314`). All heads together take only 0 (`atlas_ops.py:261-264`).

**Order within one layer of one forward.** Head scales act first (at `o_proj`'s input, inside attention), then every
`mul`, then every `act` (the one pre-hook on `down_proj` runs all muls before all acts whatever order they were given
in, `atlas_ops.py:372-380`), then the `add` at the layer's output. A neuron read (`reads`) is registered with
`prepend=True`, so it is taken before any push at its own layer (`atlas_ops.py:352-364`).

**Positions.** `keep_mask(ids, attention, bos_id, spare_bos)` is `[B, T]` bool: every real word-piece but, with
`spare_bos`, each row's first one when it is the start mark (`atlas_ops.py:415-422`). In `write_rows` the mask is set
for the prefill and cleared for the decode steps, where a written piece is never the start mark
(`atlas_ops.py:559-561, 611`). The hooks stay installed for the whole writing loop.

**Composition.** `merge(parts, rows)` joins several `Interventions` over the same rows: adds at one layer sum, the
rest are kept side by side, so sets resolved with signs naming different rows give each row its own pushes
(`atlas_ops.py:318-334`). Limits: `MAX_PUSHES = 32`, `MAX_HEAD_SCALES = 64` (`atlas_ops.py:50-51`); `check_pushes`
refuses what does not exist in plain words (`atlas_ops.py:230-264`).

**Composition with the vault.** `Weights` reads through the runtime's accessors when it has a runtime, so `down_cols`
and `write_dirs` carry the vault's column edits (`atlas_ops.py:128-131, 167-190`). An `add` push of an edited neuron
therefore pushes along the edited column, and a `mul` on an edited neuron multiplies an activation whose write is the
edited column.

**What makes pushes honest.** Rows of one forward are bit-identical while different batch shapes are not (N1A), so
every reader keeps an unpushed row in the same forward: `next_logprobs` insists on signs `(0, 1)`
(`atlas_ops.py:666-690`), `steer_lens` runs 3 × C rows with an unpushed row per context (`atlas_ops.py:1929-1972`),
`pushed_reads` puts an unpushed row first in every forward (`atlas_ops.py:1643-1660`).
`scripts/verify_atlas_writer.py` measures the floor this leaves: with no push at all, how many word-pieces an answer
written in each batch shape shares with the same answer written alone (`scripts/verify_atlas_writer.py:1-37`).

### 1.3 The weight edit in detail

**What is stored.** A dict `{column index: dense vector}` named `_memit_col_overlays` on the `down_proj` module, and a
second dict `_memit_row_overlays` that no code writes outside the checks ("none is written today",
`backend/engine_runtime.py:365-366`). The value is the *new column*, not a delta: `w[:, col] = new_col`
(`engine_runtime.py:360-364`).

**How it enters a forward.** The checkpoint's weights are never unpacked at rest. Every forward of every quantised
linear unpacks the whole weight from the 4-bit words to bfloat16 and then, if the module has overlays, clones it and
writes the columns and then the rows before `F.linear` (`engine_runtime.py:356-375`). The same rule is repeated in four
other forwards, each written out by hand:

| Path | Where | How the overlay is applied |
|---|---|---|
| W4A16 reference / `lut` / `compiled` | `backend/engine_runtime.py:352-377` | clone, columns, then rows |
| W4A16 `resident` | `backend/w4a16_fast.py:236-265` | clone of the kept weight, never the kept weight itself |
| GGUF packed or resident | `backend/gguf_weights.py:375-390, 530-532` | `with_overlays` on a clone |
| plain `nn.Linear` | `backend/gguf_weights.py:393-405` | only if `install_overlay_forward` was called on it |
| int4mm (CPU kernel) | `backend/int4_cpu.py:138-164` | as a *difference*: `x[..., i] × (v − W[:, i])` added to the kernel's product in float32 |

**How it is read back.** `slice_w_down`, `get_full_w_down`, `slice_w_gate`, `get_full_w_gate` and `write_col_norms`
lay the overlays on (`engine_runtime.py:1031-1109, 1280-1310`; `gguf_runtime.py:166-253`). `slice_w_up` and
`get_full_w_up` do not (`engine_runtime.py:1069-1077, 1111-1119`; `gguf_runtime.py:192-195, 221-224`).
`gguf_weights.dense`, `dense_cols` and `dense_rows` read the layer's own weight with no overlays
(`gguf_weights.py:542-560`), and `atlas_ops.Weights.attn_weight` goes through `dense` (`atlas_ops.py:203-209`).

---

## 2. The vault (`backend/larql_vault.py`)

### 2.1 What a `.patch` is

There is no patch file. "`.patch`" is the name of the in-memory pair (an overlay vector on a module, a
`PatchOperationRecord` in `LarqlVaultEngine.active_patches`). I found no code that writes or reads a patch on disk
(checked by grep over `backend/` and `scripts/`). The module says of itself that it is named after MEMIT and LARQL
and implements neither: "one down_proj column per edit, rank one, no covariance solve" (`larql_vault.py:13-15, 78`),
and the `larql` / `vindex` binaries are called only for `--version` (`larql_vault.py:3-4, 219-231`).

### 2.2 The edit record

`PatchOperationRecord` (`larql_vault.py:183-202`): `patch_id` (a counter that starts at 1 in each process), `op_type`
(INSERT / UPDATE / DELETE), `layer_idx`, `neuron_idx`, `polarity`, `target_concept`, `strength`, `statement` (the
statement as typed), `timestamp`, `orig_norm`, `new_norm`, `voice` (the before-and-after hearing, or None),
`edit_kind` (always `"one_neuron_column"`), `replaces_patch` (the id of the edit in force on the same neuron that this
one replaced). It does not keep the vector written, the token id the concept resolved to, the lens's id, or the
weights' identity.

`edits_in_force()` is the latest record per `(layer_idx, neuron_idx)` (`larql_vault.py:233-239`). Replaced records stay
in `active_patches`, and `active_patches_count` counts them all (`larql_vault.py:264, 553`).

### 2.3 How the direction is chosen

Code, `apply_patch_surgery` (`larql_vault.py:463-554`):

1. The layer and the neuron are clamped into range, silently (`larql_vault.py:481, 500`).
2. The original column is read from the layer's own weight, without overlays: `dense_cols(down_proj, idx)`
   (`larql_vault.py:504`). So every edit of a neuron starts from the shipped column, never from an earlier edit.
3. DELETE: the overlay is zeros (`larql_vault.py:507-510`).
4. UPDATE / INSERT: the target word is resolved to one token id; `u_scaled = W_U[tid] × final_norm_weight`;
   `pulled_dir = unit(u_scaled @ J_l)`, J_l the J++ lens of the layer; the overlay is
   `orig_col + pol × strength × max(orig_norm, 0.25) × pulled_dir`, cast to the column's dtype
   (`larql_vault.py:511-524`). Default strength 2.5 (`larql_vault.py:471, 768, 794`).
5. INSERT differs only in choosing the neuron: with a subject, the neuron whose `gate_proj` row has the largest
   |cosine| with the subject's pulled-back direction, its polarity the sign of that cosine
   (`larql_vault.py:487-498`, through `query_nearest_edges` at `270-381`). It edits no gate weight.

The statements are parsed by regular expressions (`larql_vault.py:676-806`): `DESCRIBE` / `SHOW LAYERS` / `STATUS`,
`ROLLBACK`, `SELECT * FROM EDGES|POLES NEAREST '<c>' [WHERE layer = l] [LIMIT k]` (EDGES and POLES are treated alike;
defaults layer 35, limit 10), `DESCRIBE POLE LAYER l SLOT i [POLARITY ±]`, `DELETE FROM LAYER l WHERE slot = i`,
`UPDATE LAYER l SLOT i SET object = '<c>' [STRENGTH a] [POLARITY ±]`,
`INSERT INTO LAYER l (subject, object) VALUES ('<s>', '<o>') [STRENGTH a]`.

### 2.4 The voice

`execute_lql(query, voice)` → `_heard` (`larql_vault.py:569-637`): with a voice (`prompt`, `chat`, `system`,
`assistant_prefix`, `max_new_tokens` 1–64, default 12; `larql_vault.py:81-82, 96-103`) the model writes its greedy
answer just before the edit and again just after (`TheodoliteRuntime.write_voice`, `engine_runtime.py:1666-1750`), with
nothing between, the caller holding the GPU lock around all three. Kept on the record and returned: both answers
(ids, text, per-step probability and rank, the first step's five likeliest), `first_step.kl_before_after` in nats
from float32 log-softmaxes (`larql_vault.py:176-180`), the target word's probability and rank at the first step
(`watch`), `parted_at` (the first word-piece where the answers differ), `edits_in_force_before`, `elapsed_ms`. If the
answer before cannot be written, nothing is edited; if the answer after cannot be written, the edit stays with
`after` null and `error` set (`larql_vault.py:600-613`).

It is one question, one greedy answer: "it shows what the edit did to that answer only" (ARCHITECTURE.md:629-630).
One default heard edit took 12.6 s on the GPU (ARCHITECTURE.md:605-606; `server.py:1452-1457`; the date and the
unpacker are not stated there).

### 2.5 Rollback

`rollback_patches` clears `_memit_col_overlays` and `_memit_row_overlays` on `down_proj`, `gate_proj` and `up_proj` of
layers `range(N_LAYERS)` and empties the record list (`larql_vault.py:556-567`). It is all or nothing: there is no
statement that undoes one edit. `N_LAYERS` is the constant 60 from `theodolite_math` (`larql_vault.py:68-75`,
`theodolite_math.py:24-26`); the checks that run ROLLBACK on the six-layer stand-in overwrite `larql_vault.N_LAYERS`
first (`scripts/check_gguf_fast.py:583-588`).

A second, scoped kind of undo exists outside the vault: `export_lora.vault_kept(vault)` snapshots every layer's
`down_proj` column overlays, the record list and the counter, and puts them back after the block
(`scripts/export_lora.py:383-406`).

### 2.6 Persistence across restarts

None. The records live in a Python list and the overlays in module attributes; the server builds a fresh
`LarqlVaultEngine` at start (`backend/server.py:136`) and "patch ids start again with the server"
(ARCHITECTURE.md:639). The only durable forms of an edit are the statement text (replayable, since each record keeps
`statement`) and an exported LoRA adapter (§3).

### 2.7 Concurrency

The vault has no lock of its own. The server holds its one `_GPU_LOCK` around each statement and runs it in a worker
thread (`server.py:1444-1466`); a sweep holds the same lock for its whole stream, "so no vault edit lands between two
positions of one answer" (ARCHITECTURE.md:491-492). Run alone, the MCP server's tools take turns under one lock of
their own, and a cancelled call keeps the lock until its thread ends (ARCHITECTURE.md:611-616). The lab daemon and the
server use the same port, one GPU process at a time (`experiments/lab/labd.py:22-24`).

Other tools learn of edits through `edits_in_force()`: the atlas keys its scale cache by the tuple of patch ids in
force (`backend/atlas.py:215-219, 273-293`) and marks index readings `stale` for an edited neuron or for edits at
earlier layers (`backend/atlas.py:268-269, 1401, 1602-1607`); the causal check reports `vault_edits`
(ARCHITECTURE.md:563); the server's control-vector export names `vault_edits_in_force`
(`backend/control_vector.py:138, 167`).

### 2.8 Documented weaknesses

- **Not MEMIT, not LARQL** (E6 §3 item 12, `experiments/results/E6_tool_audit_REPORT.md:188`;
  `experiments/REPORT.md:343-346`; README.md:460-463).
- **The direction goes through a lens fitted on the base model.** README.md:383 calls this the largest accuracy fault
  left: the J++ lens agrees with the instruct model's own average gradients at only cos 0.15–0.33 for the pronoun
  direction at layers 31–59, and the vault is named among what still reads through it.
- **NEAREST finds neurons, most of them not selective.** Of 102 neurons from `NEAREST 'she'` and `'he'`, 35 were
  selective at BH q < 0.05, and several were orthographic neighbours (E6 §2.3,
  `experiments/results/E6_tool_audit_REPORT.md:122-127`). "`EDGES NEAREST` is exactly a projected contribution", and a
  projected contribution is not a counterfactual (`experiments/REPORT.md:344-346`).
- **Only DELETE was tested causally by the study.** Deleting the top-10 layer-50 "she" neurons moved female contexts
  by −1.80 log-odds (E6 §2.3, lines 128–135). UPDATE and INSERT are tested by cause and effect "only as far as that one
  answer goes" (ROADMAP.md:184-185).
- **A heard edit is rare.** The first example edit changed one of seven answers, by a space; the most active neurons
  at layers 41, 46, 51, 56 and 59 were not heard on the example question at strengths 3 or 30 (ROADMAP.md:172-181).
- **The voice is one question** (ARCHITECTURE.md:629-630).

### 2.9 Weaknesses I found in the code that no document I read mentions

- **UPDATE writes a stream-space direction into a pre-norm column.** NEAREST multiplies `W_down` by the post-FFN
  norm's weight before comparing it with the pulled-back direction (`larql_vault.py:311-316`); UPDATE adds the same
  pulled-back direction to the raw column without dividing by that weight (`larql_vault.py:516-523`). What the stream
  receives is therefore `w_post ⊙ pulled_dir`, not `pulled_dir`. How far the two differ on the 31B is not established.
- **The overlay is a bfloat16 vector in a 4-bit matrix.** The edited column is cast to the column's dtype
  (`larql_vault.py:520-522`), not put back on the 4-bit grid; it is laid on after unpacking.
- **Indexes out of range are clamped, not refused** (`larql_vault.py:286, 392-393, 481, 500`): a statement naming
  layer 99 edits layer 59. The record does name the neuron actually edited.
- **`active_patches_count` counts records, not edits in force** (`larql_vault.py:264, 553`).
- **On a plain `nn.Linear` the edit is seen and not heard.** The vault creates the dict if it is missing
  (`larql_vault.py:484-485`) and the accessors read it, but `nn.Linear.forward` ignores it unless
  `install_overlay_forward` was called; `scripts/atlas_stand_in.py` does not call it,
  `scripts/check_lora_export.py:595-596` and `backend/gguf_model.py:592` do.

---

## 3. Persistence and export formats

### 3.1 The three formats

| | Control vector | LoRA adapter | Whole-model GGUF copy |
|---|---|---|---|
| Written by | `scripts/export_control_vector.py`; from the server, `backend/control_vector.py` (`POST /api/atlas/control_vector`) | `scripts/export_lora.py` | `scripts/export_lora.py --model-out` |
| Tensors | `direction.<l>`, F32, one per layer, the model's width (`export_control_vector.py:385-398`) | `<model tensor>.lora_a` `[n_in, r]` and `.lora_b` `[r, n_out]`, F32, one pair per model tensor (`export_lora.py:501-529`) | every tensor copied, plus a one-number F32 `blk.<l>.ffn_down.scale` or `attn_output.scale` = 0; or the F32 `post_ffw_norm` / `post_attention_norm` zeroed in place (`export_lora.py:607-654`) |
| Carries | `add` pushes of single neurons, summed per layer | a neuron's `mul`, a vault column edit, one head's scale | a whole feed-forward block or every head of a layer silenced |
| Cannot carry | layer 0 (llama.cpp refuses `direction.0`); mul, act, heads, vault edits as such; sparing the start mark | `act` (refused: no hook in llama.cpp); `add` (that is the control vector); full-rank changes; sparing the start mark | anything but silencing; a copy of a silenced copy is refused (`export_lora.py:584-591`) |
| Mapping | `dose × p50_l × unit(w_post ⊙ W_down[:, n])`, added after `layer_output_scale` | mul f: `A = e_nᵀ`, `B = (f − 1) W_down[:, n]`; vault: `B = f·v − W_down[:, n]`; head: A the identity on the head's inputs, `B = (f − 1) W_O[:, h·hd:(h+1)·hd]` (`export_lora.py:442-479`) | — |
| Rank | — | 1 per neuron; hd per head: 256 on the 31B's sliding layers, 512 on its full ones (ARCHITECTURE.md:1785) | — |
| Scale at load | `--control-vector-scaled FILE:SCALE` multiplies every dose | SCALE per request; no `adapter.lora.alpha` is written; at SCALE s a factor f becomes `1 + s (f − 1)` | none |
| Provenance in the file | `theodolite.*`: model, weights read (revision and sha256, or a GGUF fingerprint), each entry, the stats index's sha256, the code's commit, the time, a caveat | `theodolite.*`: the pushes as given, each entry, the weights read, the code's commit, the time | `theodolite.*` in the file (by scale) or `FILE.theodolite.json` beside it (by norm) |

Both writers are atomic (a temporary file, then a rename) and both have a reader (`read_control_vector`,
`export_control_vector.py:423-435`; `read_lora`, `export_lora.py:532-552`) and a `--show`.

### 3.2 Where the numbers come from

- The scripts read a few tensors on the CPU from the W4A16 checkpoint in the local cache, or from a GGUF with
  `--gguf` (`export_control_vector.py:159-286`; `export_lora.py:303-380`). The vault's edits are **not** in a
  script-written control vector.
- The server route reads the model it holds, so the vault's edits in force **are** in the direction, the file says
  so, and a neuron silenced by an edit is refused (`backend/control_vector.py:69-88, 138-146`).
- `export_lora.py --vault STATEMENT` does not export a running server's edits. It loads the instrument's runtime on
  the CPU ("the whole model, minutes and some 20 GB on the 31B"), runs the statement there, reads the column laid on,
  and leaves that vault as it was (`export_lora.py:86-91, 419-439, 801-827`). There is no route that turns the live
  vault's state into an adapter.
- The 31B has no whole-model writer with edits baked in. `gguf_model.write_gemma4_gguf` reads `.weight` of plain
  linears (`backend/gguf_model.py:974-991`): it is the stand-in's writer. `pack_w4a16` is a helper in a check script
  (`scripts/check_dequant.py:102-111`).

### 3.3 What differs from the instrument by construction

- llama.cpp acts at every position, the start mark included; the instrument's pushes and head scales spare it. A vault
  edit is a weight, everywhere, in both (`export_lora.py:56-57`).
- Adapters served together add. On different neurons or heads that is the instrument's combined push; on the same
  one it is not: mul f1 and mul f2 give `f1 + f2 − 1`, not `f1 × f2`, and a mul-0 file beside a vault-edit file of
  that neuron leaves `v − W` where the instrument leaves 0 (`export_lora.py:50-55`).
- B is made from the columns read. Read from the checkpoint and served on a GGUF of other quantisation, a silenced
  neuron keeps `(W_gguf − W_checkpoint)[:, n]` (`export_lora.py:60-63`).
- A by-scale silenced copy of a layer's attention does not silence an adapter on that layer's heads: llama.cpp scales
  the projection before it adds the adapter's part (`export_lora.py:40-44`).

### 3.4 How exports are verified

Three levels, each with a script.

1. **Arithmetic on a stand-in, CPU.** `scripts/check_control_vector.py` (48 checks in about 15 s,
   ARCHITECTURE.md:1646): every push a census of two neurons adds is an exported vector bit for bit.
   `scripts/check_lora_export.py` (77 checks in about 50 s, ARCHITECTURE.md:1815): each adapter's `W + B A` is the
   instrument's change on random inputs within float32 rounding (the check's bar is 1e-5 of the largest element,
   `check_lora_export.py:210`; 5.0e-7 when written, ARCHITECTURE.md:1818), for mul 0, 2 and −1.5, SCALE −1, 0.5 and
   0, heads on a sliding and a full layer, the vault's UPDATE, and a neuron both edited and multiplied; the checkpoint
   and the same weights as a float32 GGUF give the same adapter bit for bit.
2. **Through llama.cpp itself on the stand-in.** `check_control_vector.LlamaServer`
   (`scripts/check_control_vector.py:319-382`) starts a `llama-server` on the stand-in's float32 GGUF on the CPU and
   reads every word-piece's log-probability at the first one written. Alike: largest difference 4.4e-4 unpushed,
   2.0e-4 for a file at SCALE 2 against the push at dose 2. Apart: 0.18 / 0.13 a layer below / above, 3.9e-2 with the
   start mark spared (ARCHITECTURE.md:1655-1662). For adapters: 3.8e-4 to 6.8e-4 alike, 4.6e-2 to 0.36 apart
   (ARCHITECTURE.md:1826-1834). The script also checks what the server does with what it cannot use
   (`check_lora_export.py:551-578`).
3. **On the 31B, against a bar fixed beforehand.** Control vector, 30 October 2026: over 2,304 word-pieces llama.cpp's
   change in log-probability correlates with the census's at r = 0.986, median ratio 1.11; it carries over at SCALE
   ±1 and breaks at ±2 because llama.cpp also pushes the start mark; 81 s for the instrument's side, 279 s for
   llama.cpp's (ARCHITECTURE.md:1708-1747). LoRA, 2 November 2026, both engines on the CPU: every push that moves the
   first piece by 2 or more passes (r 0.88–0.98, ratio 0.91–1.15); head 0 at × 0 does not (r 0.62)
   (ARCHITECTURE.md:1842-1855). Not measured: a GPU backend's arithmetic for `B (A x)`, other quantised GGUFs,
   `--vault` on the real runtime, a silenced copy of the 31B (ARCHITECTURE.md:1857-1859).

`scripts/verify_atlas_writer.py` is not an export check. It measures the writer's noise floor on the real model
(arithmetic says about 18 minutes with the reference unpacker; measured 2 minutes with `compiled` on 28 October 2026,
COMPUTE.md:25) and fails if the same answer written alone twice is not the same (`verify_atlas_writer.py:214-216`).

---

## 4. The gap between today's edit and a general weight program

### 4.1 What separates them

| A weight program needs | Today |
|---|---|
| Many tensors | One: `down_proj`. The vault writes no other (`larql_vault.py:482-485`). |
| A delta representation | A replacement vector per column index, dense, `Dict[int, Tensor]`. No low-rank `B A`, no sparse element list, no additive delta, no scale. |
| Attention weights | The overlay dicts and the overlay-applying forward are installed on every quantised linear, attention's q, k, v and o among them (`engine_runtime.py:429-436`), but nothing writes them, ROLLBACK does not clear them (`larql_vault.py:559-565`), and `Weights.attn_weight` would not show them (`atlas_ops.py:203-209`). |
| `up_proj` rows | The forward would apply a row overlay; `slice_w_up` and `get_full_w_up` would not show it (`engine_runtime.py:1069-1077, 1111-1119`). |
| Embeddings, the tied lm_head, norms, layer scalars | Plain parameters with no overlay, no record and no rollback. Reading them: `Weights.norm`, `layer_scalars` (`atlas_ops.py:112-118, 142-147`). |
| Composition | The latest edit of a neuron replaces the earlier; every edit starts from the shipped column (`larql_vault.py:504`). Two edits cannot be stacked or scaled. |
| Partial undo | ROLLBACK is all or nothing. |
| Versioning, naming, diffing | A per-process counter. No named program, no parent, no diff between two states, no hash of the state. |
| Persistence | None (§2.6). |
| A/B inside one forward | Pushes have it (per-row values); a weight overlay applies to the whole batch, so the voice uses two separate writes. |
| Provenance of a reading | `weights_card` names the checkpoint's files and not the overlay state (`backend/runtime_factory.py:251-272`); some tools add `vault_edits` themselves. |
| A general direction chooser | One: a single token's unembedding pulled back through the base-fitted J++ lens. |
| Model-agnostic sizes | `N_LAYERS = 60`, `D_FF = 21504` are constants in the vault (`larql_vault.py:68-75`). |

### 4.2 What can serve as the kernel as it is

- **The per-module overlay hook point.** Every quantised linear already consults two dicts at every forward, on all
  five numeric paths, and each path is checked against the others (`scripts/check_gguf.py`, `check_gguf_cpu.py`,
  `check_gguf_fast.py`, `check_dequant.py`). A column or row replacement on any of the 410 linears works in the
  forward today without new forward code.
- **`int4_cpu.with_overlay_terms`** (`backend/int4_cpu.py:138-164`): the one place that already applies an edit as a
  delta term, `x[..., i] (v − W[:, i])`, summed in float32 and rounded once. This is the form a low-rank or sparse
  delta needs, and it leaves the base product untouched.
- **`export_lora.lora_tensors` / `write_lora` / `read_lora`** (`scripts/export_lora.py:442-552`): `write_lora` takes
  `{model tensor name: (A [r, n_in], B [n_out, r])}` for any tensor name and any rank. It is already a general
  low-rank container with provenance, and llama.cpp reads it.
- **`export_lora.Plan` and `plan_of`** (`scripts/export_lora.py:169-297`): one checked structure holding neuron
  factors, vault columns, head factors and silenced blocks by layer: the nearest thing to a weight-program record.
- **`export_lora.vault_kept`** (`scripts/export_lora.py:383-406`): a snapshot-and-restore context manager, the
  pattern for "try this edit, then put everything back".
- **`atlas_ops.Interventions` / `resolve` / `merge` / `installed` / `keep_mask`**: a per-row intervention kernel with
  an unpushed row in the same forward. `write_rows` and `next_logprobs` are the fast readers built on it.
- **`atlas_ops.forward_from_layer`** (`atlas_ops.py:1017-1043`): the forward from layer `start` on, given the stream
  entering it, hooks included. An edit at layer l only needs layers l onward re-run.
- **`LarqlVaultEngine._heard`** (`larql_vault.py:569-637`): the shape of a before / edit / after transaction with a
  refusal rule on each side.
- **`check_control_vector.LlamaServer`** and **`packed_stand_in`** (`scripts/check_control_vector.py:97-134, 319-382`):
  a harness that serves a GGUF through llama.cpp on the CPU and reads first-piece log-probabilities, and a stand-in
  packed as the real checkpoint is.
- **`Atlas.edit_key()`** (`backend/atlas.py:215-216`): caches keyed by the set of edits in force.
- **The lab daemon** (`experiments/lab/labd.py:1-34`, read as far as its header): one GPU process that keeps the model
  loaded and executes submitted Python against `rt`, `lab`, `torch`, with a persistent `STORE`. It is the nearest
  thing in the repository to the paper's "code execution and forward passes".

### 4.3 What needs generalising

- The record: from `(layer, neuron)` to `(tensor, index or factors)`, keeping the vector or the factors, the base
  weights' identity, the lens's id, and a parent.
- `edits_in_force`, `Atlas.edited()` and every `stale` rule, which are keyed by `(layer, neuron)` of `down_proj`.
- ROLLBACK, to walk every module that has overlays and to undo one edit or one named program.
- The accessors, so that every weight reader shows what the forward uses (`slice_w_up`, `get_full_w_up`,
  `attn_weight`, `dense*`).
- `Columns` in the exporter, which reads only `down_proj` and `o_proj` columns (`export_lora.py:303-380`).
- The vault's hard-coded sizes and its silent clamping.
- The voice, from one question to a held-out set with a control.

### 4.4 What would break

- **The clone per forward.** With an overlay the W4A16 forward clones the whole unpacked weight
  (`engine_runtime.py:361-362`). One `down_proj` is 5376 × 21504 bfloat16 values, 231 MB by arithmetic; an edit in
  each of the 410 linears is a clone of each at every forward. The cost in seconds has not been measured.
- **Dense deltas do not fit.** The whole model unpacked is 62 GB in bfloat16 (`engine_runtime.py:411`); the card holds
  32 GB. Any general delta must be sparse or low-rank.
- **`resident` forbids in-place writes.** Its forward checks the kept weight's version counter and raises if it was
  written (`w4a16_fast.py:243-249`).
- **`engine_runtime.py` is pinned by sha256 for R4** (`experiments/scripts/R4_common.py:101`; `w4a16_fast.py:1-3`), so
  the pinned forward cannot be edited. A new delta mechanism has to be bound from outside, as `w4a16_fast` and
  `int4_cpu` bind theirs.
- **Numeric path.** Before the int4mm kernel applied edits as difference terms, the edited layer fell back to the
  reference forward, and on the stand-in an edit that set a column to its own value changed 13 of 30 greedy answers
  (ARCHITECTURE.md:1391-1395). Any new apply path can repeat this.
- **Every index was built on the shipped weights.** The wiring, steer-lens, passages and stats indexes and the census
  are refused for other weights by the weights card (`backend/atlas.py:202-204`), which does not see overlays; the
  per-neuron `stale` flags are the only guard, and they know nothing of attention or norm edits.
- **The lab kit's gradient pass on the checkpoint ignores overlays.** `_W4A16LinearFn` unpacks the packed weight with
  no overlay; its GGUF twin lays them on (`experiments/lab/labkit.py:417-453`).
- **Exports assume one kind.** `vault_columns` reads `down_proj._memit_col_overlays` only (`export_lora.py:435-438`).
- **The start mark.** A weight edit acts on it; pushes spare it. A program mixing both has two position rules.

---

## 5. Where the code already anticipates more general edits

1. `_memit_row_overlays`, installed beside the column dict on every quantised linear, honoured by every forward,
   cleared by ROLLBACK, shown by the gate accessors, and tested (`engine_runtime.py:365-372, 433-434`;
   `larql_vault.py:564-565`; `scripts/check_gguf.py:382-402`). Nothing writes it.
2. The overlay dicts on all 410 quantised linears, not only the MLP's (`engine_runtime.py:429-436`).
3. `PatchOperationRecord.edit_kind = "one_neuron_column"` and `replaces_patch` (`larql_vault.py:200-202`): a field
   that names the kind of edit, with one value.
4. `gguf_weights.install_overlay_forward` for plain linears and `GGUFLinear`'s own overlays
   (`gguf_weights.py:393-405, 456-457`): the vault works on every weight format the runtime loads.
5. `int4_cpu.with_overlay_terms` handles rows after columns, including the cell where they cross
   (`int4_cpu.py:157-163`).
6. `export_lora.write_lora` is general in tensor name and rank; `Plan` already unifies four kinds of change.
7. `HeadScale`, `AttnShape`, `Weights.attn_weight` and `HeadSpaces` (`atlas_ops.py:78-109, 203-209, 1736-1815`): the
   attention weights are already read by role and head.
8. `Atlas.edit_key()` and the `stale` flags: readings already know that the weights can change under them.

---

## Numbers

| What | Value | Where | When |
|---|---|---|---|
| Quantised linears unpacked at every forward | 410 | ARCHITECTURE.md:131; `w4a16_fast.py:5` | — |
| Whole model unpacked to bfloat16 | 62 GB | `engine_runtime.py:411` | — |
| One decode step, 1 row: reference / lut / compiled | 0.706 / 0.275 / 0.091 s | COMPUTE.md:52-57 | 28 Oct 2026 |
| One decode step, 64 rows: reference / lut / compiled | 0.721 / 0.287 / 0.109 s | COMPUTE.md:52-57 | 28 Oct 2026 |
| Three pushed answers of 80 word-pieces, compiled / reference | 8.3 s / 50–54 s | COMPUTE.md:61-64 | 11 Nov 2026 |
| One default heard edit (two 12-piece answers and the edit) | 12.6 s | ARCHITECTURE.md:605-606 | not stated |
| Voice length: default / limit | 12 / 64 word-pieces | `larql_vault.py:81-82` | — |
| UPDATE default strength; floor on the column's length | 2.5; 0.25 | `larql_vault.py:471, 521` | — |
| Pushes / head scales in one reading | 32 / 64 | `atlas_ops.py:50-51` | — |
| Default add dose | 0.8 of p50 | `atlas_ops.py:220, 1925` | — |
| A dose of 1 against a neuron's write ceiling, layers 40–50 | 1.8–2.5 times | ARCHITECTURE.md:1671-1672 | — |
| LoRA rank: a neuron / a head (sliding, full) | 1 / 256, 512 | ARCHITECTURE.md:1783-1785 | — |
| `check_control_vector.py` | 48 checks, about 15 s, CPU | ARCHITECTURE.md:1646 | — |
| `check_lora_export.py` | 77 checks, about 50 s, CPU | ARCHITECTURE.md:1815 | — |
| `W + B A` against the instrument | 5.0e-7 of the largest element | ARCHITECTURE.md:1818 | — |
| Control vector on the 31B | r 0.986, ratio 1.11, 2,304 pieces; 81 s + 279 s | ARCHITECTURE.md:1708-1715 | 30 Oct 2026 |
| LoRA on the 31B (CPU both sides) | r 0.88–0.98, ratio 0.91–1.15 for pushes moving the first piece by ≥ 2 | ARCHITECTURE.md:1842-1855 | 2 Nov 2026 |
| `--vault` export's CPU runtime | minutes, some 20 GB | `export_lora.py:86-88` | — |
| QAT Q4_0 GGUF direction against the checkpoint's | cosine least 0.9965, median 0.9969 (192 neurons) | ARCHITECTURE.md:1606-1613 | 30 Oct 2026 |
| NEAREST 'she' / 'he' neurons selective | 35 of 102 | E6_tool_audit_REPORT.md:124-126 | — |
| DELETE of ten layer-50 'she' neurons, female contexts | −1.80 log-odds | E6_tool_audit_REPORT.md:128-133 | — |
| J++ lens against the instruct model's gradients | cos 0.15–0.33 (pronoun direction, layers 31–59) | README.md:383 | — |
| An edit to a column's own value, before the fix, stand-in | changed 13 of 30 greedy answers | ARCHITECTURE.md:1394-1395 | — |
| One `down_proj` in bfloat16 | 231 MB (5376 × 21504 × 2, my arithmetic) | — | — |

---

## Documents against code

- ARCHITECTURE.md §4.6 matches the code I read in `larql_vault.py` and `server.py` (the request model, the order of
  the three steps, the refusal rules, the reply's keys, the worker thread under the lock).
- ARCHITECTURE.md §6's mapping tables match `export_control_vector.py` and `export_lora.py` (tensor names, the A and B
  formulas, the refusals). The measured tables are the document's; I did not rerun anything.
- The vault's module docstring lists `EDGES` and `POLES` as two searches; the code discards the keyword
  (`larql_vault.py:701`).
- E6 item 12 said INSERT's docstring claimed a gate edit. The docstring now says it does not
  (`larql_vault.py:13-15, 475-478`): fixed in the text, and the code never edited the gate.

## Not established

- What an overlay costs per forward in seconds on the 5090, for one edit or for many.
- Whether llama.cpp's Gemma 4 graph applies an adapter to tensors other than `ffn_down` and `attn_output` (the
  repository read its source only for those two).
- How far `w_post ⊙ pulled_dir` is from `pulled_dir` on the 31B (the UPDATE direction, §2.9).
- Whether a census run or an index build records the vault's edits in force in its meta.
- Whether a row overlay on `gate_proj` or `up_proj`, or any overlay on an attention projection, has ever been run on
  the 31B (the checks use stand-ins).
- What the lab daemon's `/exec` allows beyond its header: I read only `labd.py:1-34` and the lab kit's outline.
- The unpacker and the date of the 12.6 s heard edit.
