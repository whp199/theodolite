# The runtime and how the weights are held

Reader's report for the weight-space harness design, written 14 November 2026. Read only: no file in the repository
was changed, no model was loaded and the GPU was not touched. Paths are relative to `(the repository root)`
unless they start with `/`. "ie:" means the installed library at
`.venv/lib/python3.12/site-packages/interp_engine/` (version 1.14.0), and "hf:" means
`.venv/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py` (transformers 5.20.0).

Three kinds of statement are kept apart below:

- **Code**: I read the lines cited and the statement is what they do.
- **Recorded**: a number written in a result file, a log or a document, with its date where one is given.
- **Computed here**: my own arithmetic from the tensor shapes; nobody measured it.

The tensor inventory in section 1 comes from the JSON header of the checkpoint's `model.safetensors` in the local
Hugging Face cache (snapshot `52f3f65bc7a02d555763bc923bd1d9094898219d`, 23,265,352,448 bytes, 2,009 tensors), which I
parsed with a pure-Python one-liner; no weight was read.

---

## 0. What a designer must not get wrong

1. Every one of the 410 text linears (attention q, k, v, o and feed-forward gate, up, down) is packed 4-bit. The
   embedding, the unembedding (tied to it), every norm weight and every layer scalar are plain bf16.
2. A forward never holds a dense copy of the model. Each linear's dense bf16 weight is rebuilt from the packed words
   inside that linear's own `forward`, used once and dropped. All 410 are rebuilt on every forward and on every
   decode step.
3. The only weight overlay that exists replaces whole columns or whole rows of one linear, from a dict on the module,
   on a clone of the freshly unpacked weight. It applies to every row of the batch alike. Only `down_proj` columns are
   ever written today.
4. `engine_runtime.py` is supposed to be untouched because R4 pins it by sha256. **That pin does not hold today**: the
   file was edited on 12 November 2026 and its hash no longer matches (section 7). The documents still say it does.
5. One process owns the card at a time, and nothing enforces that on the home machine except a shared port number.
6. On the GPU a row's numbers depend on the shape of its batch. The unpacker does not change a bit; the batch does.
7. A backward pass has run on this model on this card (R1, R3), through a custom autograd function that unpacks each
   weight again in the backward. It gives gradients with respect to activations only; no code returns a gradient with
   respect to a weight.

---

## 1. How the weights are held

### 1.1 The tensors (from the checkpoint header)

| Tensor | Count | dtype | Shape | Bytes in file |
|---|---|---|---|---|
| `embed_tokens.weight` | 1 | BF16 | [262144, 5376] | 2.625 GiB |
| `lm_head.weight` (stored apart, tied at load) | 1 | BF16 | [262144, 5376] | 2.625 GiB |
| `layers.N.mlp.gate_proj.weight_packed` / `weight_scale` | 60 | I32 / BF16 | [21504, 672] / [21504, 168] | 3.230 + 0.404 GiB |
| `layers.N.mlp.up_proj.weight_packed` / `weight_scale` | 60 | I32 / BF16 | [21504, 672] / [21504, 168] | 3.230 + 0.404 GiB |
| `layers.N.mlp.down_proj.weight_packed` / `weight_scale` | 60 | I32 / BF16 | [5376, 2688] / [5376, 672] | 3.230 + 0.404 GiB |
| `layers.N.self_attn.q_proj` packed / scale | 60 | I32 / BF16 | [8192, 672] sliding, [16384, 672] full | 1.436 + 0.179 GiB |
| `layers.N.self_attn.k_proj` packed / scale | 60 | I32 / BF16 | [4096, 672] sliding, [2048, 672] full | 0.564 + 0.070 GiB |
| `layers.N.self_attn.v_proj` packed / scale | 50 | I32 / BF16 | [4096, 672] (sliding layers only) | 0.513 + 0.064 GiB |
| `layers.N.self_attn.o_proj` packed / scale | 60 | I32 / BF16 | [5376, 1024] sliding, [5376, 2048] full | 1.436 + 0.179 GiB |
| each linear's `weight_shape` | 410 | I64 | [2] | negligible |
| `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` `.weight` | 60 each | BF16 | [5376] | negligible |
| `self_attn.q_norm.weight`, `k_norm.weight` | 60 each | BF16 | [256] sliding, [512] full | negligible |
| `layers.N.layer_scalar` | 60 | BF16 | [1] | negligible |
| `norm.weight` (final norm) | 1 | BF16 | [5376] | negligible |

- The language model's tensors total 17.97 GiB in the file. The other 357 tensors (vision tower and its projection,
  3.70 GiB with `lm_head`) are dropped or tied at load.
- 410 = 60 x 6 + 50. The ten full-attention layers (0-based 5, 11, 17, 23, 29, 35, 41, 47, 53, 59) have no `v_proj`:
  the config sets `attention_k_eq_v: true`, so their values are the normed keys (hf:1196-1202, hf:1231-1232;
  `backend/atlas_ops.py:107`).
- The linears come in eight dense shapes: [21504, 5376], [5376, 21504], [8192, 5376], [16384, 5376], [4096, 5376],
  [2048, 5376], [5376, 8192], [5376, 16384]. `backend/w4a16_fast.py:184-185` says "some eight shapes", and the
  recorded install record says `"shapes": 8` (`experiments/results/atlas_writer_floor_rtx5090.json`).
- **Computed here**: 29.29 billion packed weights (50 x 478.9 M a sliding layer + 10 x 534.0 M a full layer), which
  matches the documents' "29.3 billion" (`COMPUTE.md:28-29`).

### 1.2 The packed layout

- **Config** (`config.json`, `quantization_config`): format `pack-quantized`, `num_bits: 4`, `type: int`,
  `symmetric: true`, `strategy: group`, `group_size: 32`, targets `Linear`, and an ignore list of 192 names: every
  vision-tower linear, `model.embed_vision.embedding_projection` and `lm_head`.
- **Code** (`backend/engine_runtime.py:296-309`): `weight_packed` is int32 `[out, in // 8]`, eight 4-bit pieces a
  word, piece j of word k being input column 8k + j, read by shifts 0, 4, ..., 28 (`:287-292`). `weight_scale` is bf16
  `[out, in // 32]`: one scale per output row per group of 32 consecutive input columns. The dense weight is
  `(piece - 8) * scale`, so each weight is one of 16 values in `[-8, 7] x scale`, with no zero point.
- **Computed here**: 4.5 bits a weight (4 for the piece, 16/32 for the scale). One feed-forward matrix is 65.0 MB
  packed and 231.2 MB (0.215 GiB) dense in bf16.
- The groups run along the input axis for every linear. So in `down_proj` [5376, 21504] a neuron's column shares each
  of its 5,376 scales with the 31 neighbouring neurons of its group, and in `gate_proj` / `up_proj` [21504, 5376] a
  neuron's row has 168 scales of its own. A designer who wants an edit that stays on the 4-bit grid must respect that.

### 1.3 What is not quantised, and how each is applied

- **Embedding**: `Gemma4TextScaledWordEmbedding`, the looked-up row times `embed_scale = sqrt(5376)`, a
  non-persistent buffer (hf:1433-1444, hf:1565).
- **Unembedding**: tied to the embedding. At rest the card holds one copy: the model without a lens allocates
  18.001 GiB (`experiments/results/R3_00_pilot.json`, `vram.after_load`, 22 October 2026), which is the packed words
  (13.64 GiB) plus scales (1.70 GiB) plus one 2.625 GiB embedding. The runtime keeps `arch.lm_head.weight.detach()`
  as `rt.lm_head_weight` (`backend/engine_runtime.py:858`), which shares storage with it. An in-place edit of an
  embedding row therefore changes the unembedding row too.
- **Norms**: `Gemma4RMSNorm` computes in float32, `x * (mean(x^2) + eps)^-0.5 * weight`, and casts back
  (hf:205-223). The weight is multiplied as it is. `ARCHITECTURE.md:125` says "`(1 + w)` unit-offset scaling"; the
  code multiplies by `w` alone and `backend/atlas_ops.py:11` says "the unit offset is already in w". The code and
  `atlas_ops` agree; the table row in ARCHITECTURE is loosely worded.
- **Layer scalar**: an `nn.Buffer` of one bf16 number per layer, multiplied into the whole stream in place as the
  layer's last act, `hidden_states *= self.layer_scalar` (hf:1356, hf:1429). So
  `post = s * (mid + post_ffn_norm(mlp(pre_ffn_norm(mid))))` (`backend/atlas_ops.py:8-9`).
- **Caches of these that an edit would leave stale**: `rt._final_norm_weight`, `rt._pre_ffn_norm_weights`,
  `rt._post_ffn_norm_weights` are `.detach()` views (`backend/engine_runtime.py:857-865`), so an in-place edit shows
  through them but a replaced Parameter would not. `rt._write_norm_cache` (`:1280-1310`) is computed once per layer
  from the unedited `down_proj` and the post norm, and only column overlays are recomputed. `atlas_ops.Weights`
  reads the scalars once at construction (`backend/atlas_ops.py:140`). Every atlas index on disk (stats, wiring,
  steer lens, passages) was built from the unedited weights.

### 1.4 How a forward gets a dense weight

`patch_compressed_tensors_for_low_vram` (`backend/engine_runtime.py:410-442`) removes compressed-tensors' own
decompress hook (which would unpack the model to about 62 GB), drops the vision and audio towers, and on every module
holding `weight_packed` and `weight_scale` sets two empty dicts (`_memit_col_overlays`, `_memit_row_overlays`) and an
instance-level `forward` (`:431-436`). That forward (`_w4a16_forward`, `:356-375`) does, on every call:

1. `w = dequantize_w4a16_full(self.weight_packed, self.weight_scale)`, looked up as a module global at call time;
2. the overlays, if any (section 2);
3. `F.linear(x.to(w.dtype), w, bias)`, with `bias` None on this model.

The four unpacker modes (`backend/w4a16_fast.py`, installed from outside the runtime file):

| Mode | What it does (code) | Memory traffic a weight (`w4a16_fast.py:5-19`, arithmetic) | Decode step, 1 / 32 / 64 / 96 rows (recorded) | torch's peak above the model at 1 row (recorded) | Extra memory held |
|---|---|---|---|---|---|
| `reference` | five eager ops, three over int32 `[out, in/8, 8]` temporaries (`engine_runtime.py:305-309`) | about 32.6 bytes | 0.706 / 0.713 / 0.721 / 0.743 s | 0.962 GiB | none |
| `lut` | packed bytes through a 256-entry table, then the same multiply (`w4a16_fast.py:153-161`) | about 12.6 bytes | 0.275 / 0.282 / 0.287 / 0.303 s | 0.500 GiB | none |
| `compiled` | the reference's arithmetic through `torch.compile`, one graph per weight shape, the model itself left eager (`:170-230`) | about 4.6 bytes | 0.091 / 0.103 / 0.109 / 0.126 s | 0.285 GiB | none |
| `resident` | every weight unpacked once and kept on the module as `_resident_w`, a new forward reading it (`:236-265`, `:499-547`) | 2 bytes (the product alone) | never timed | never measured | 58.6 GB (54.6 GiB) beside the 4-bit weights (computed; `COMPUTE.md:44` says "about 60 GB") |

- The times and peaks are from `experiments/results/atlas_unpacker_steps_rtx5090.json`, written 2026-10-29T01:48Z
  (the evening of 28 October local time), layer 45, census-shaped batches at `--max-new 64`, the lens freed, beside
  a desktop. They are repeated in `COMPUTE.md:52-57`.
- `resident` cannot be installed on the 5090: `install` refuses when the kept bytes plus the largest unpacking
  transient exceed what is free (`w4a16_fast.py:499-509`). `COMPUTE.md:121` names a 96 GB card for it.
- How `lut` and `compiled` are installed: they rebind `engine_runtime.dequantize_w4a16_full` (`w4a16_fast.py:494-497`).
  The pinned forward, `dequantize_w4a16_rows` and the full-matrix accessors look that name up at call time, so they
  follow; `dequantize_w4a16_cols` has its own arithmetic and is left alone (`w4a16_fast.py:34-41`).
- How `resident` is installed: `mod.forward` is replaced on each packed module, the old one saved as
  `_w4a16_fast_saved_forward` (`:514-520`); `install(..., "reference")` puts every one back (`:332-363`).
- `compiled` only serves the shapes it tested at install. Any other shape (a row slice, another model) goes through
  the reference. A call whose guards fail, "under torch.inference_mode or autocast, say", falls back to the
  reference too, is counted in `.fell_back` and logged once (`:205-230`). A harness that wraps its forwards in
  `torch.inference_mode()` or autocast would silently lose the sevenfold speed-up.
- The server takes `compiled` by default for the checkpoint on a CUDA card since 11 November 2026
  (`w4a16_fast.py:410-418`, `backend/server.py:117-133`). The scripts and the lab daemon default to `reference`
  (`experiments/lab/labd.py:157`, `ARCHITECTURE.md:135`).
- Why the bits agree: the pieces minus 8 are exact in bf16, so the one rounding on any route is the multiply by the
  scale (`w4a16_fast.py:21-32`). `install` tests `compiled` on every linear by bits and strides and on all 16 x
  65,536 piece-by-scale products (`:366-407`); `resident` by each kept weight against a fresh unpacking and each
  forward against the pinned one on a fixed two-row probe (`:521-528`).

---

## 2. Where weight overlays enter a forward

### 2.1 The mechanism, exactly (code)

`backend/engine_runtime.py:356-375`:

```
w = dequantize_w4a16_full(self.weight_packed, self.weight_scale)
col_overlays = getattr(self, "_memit_col_overlays", None)      # Dict[int, Tensor[out]]
if col_overlays:
    w = w.clone()
    for col_idx, new_col in col_overlays.items():
        w[:, col_idx] = new_col.to(device=w.device, dtype=w.dtype)
row_overlays = getattr(self, "_memit_row_overlays", None)      # Dict[int, Tensor[in]]
if row_overlays:
    if not col_overlays:
        w = w.clone()
    for row_idx, new_row in row_overlays.items():
        w[row_idx, :] = new_row.to(device=w.device, dtype=w.dtype)
return F.linear(x.to(dtype=w.dtype), w, bias)
```

- The rules: columns first, then rows; one clone of the whole dense weight when any overlay is in force; one indexed
  assignment (one kernel launch) per entry in a Python loop; the replacement is a whole column or a whole row; the
  value is whatever bf16 vector was stored, not held to the 4-bit grid; the dict is keyed by index so a later edit of
  the same index replaces the earlier one.
- The overlay is global to the forward: every row of a batch sees the same edited weight. There is no per-row
  variant.
- The resident forward repeats these rules "line for line" over the kept weight, clones only when an overlay exists
  and checks the kept weight's version counter on every call, which catches an in-place write but not one through
  `.data` (`backend/w4a16_fast.py:236-265`).
- Every one of the 410 linears has both dicts, attention included (`engine_runtime.py:431-436`). The vault writes
  only `down_proj` columns (`backend/larql_vault.py:483-523`): DELETE stores zeros; UPDATE and INSERT store
  `orig + sign * strength * max(||orig||, 0.25) * dir`, cast to the column's dtype. "none is written today" for rows
  (`engine_runtime.py:365-366`).
- Rollback clears the dicts of `down_proj`, `gate_proj` and `up_proj` at every layer and the record list
  (`larql_vault.py:556-567`). It does not touch attention linears: an overlay put on q, k, v or o by new code would
  survive the vault's rollback.
- Readers that honour overlays: `slice_w_down`, `slice_w_gate`, `get_full_w_down`, `get_full_w_gate`,
  `write_col_norms` (`engine_runtime.py:1032-1109`, `:1280-1310`). Readers that do not: `slice_w_up` and
  `get_full_w_up` (`:1070-1077`, `:1111-1119`), `gguf_weights.dense`, `dense_cols`, `dense_rows`
  (`backend/gguf_weights.py:543-576`, "no overlays") and so `atlas_ops.dense_weight` and `Weights.attn_weight`
  (`backend/atlas_ops.py:57-63`, `:203-209`). A row overlay on `up_proj` would change the forward and not the
  accessor.
- The gradient path of the study ignores overlays: labkit's `_W4A16LinearFn` unpacks the bare weight
  (`experiments/lab/labkit.py:417-432`), and R3's `grad_linears` refuses to run while any overlay is live
  (`experiments/scripts/R3_gpu.py:94-98`).
- A second formulation already exists for the CPU kernel path: `int4_cpu.with_overlay_terms` leaves the weight alone
  and adds the overlay's difference to the product, `x[..., i] (v - W[:, i])` for a column, summed in float32 and
  rounded once (`backend/int4_cpu.py:138-164`). Its sums are not the reference's bits, which is why that path carries
  `+int4mm` in its model id.
- What an overlay costs today (computed here): the clone is one 231 MB copy per overlaid feed-forward matrix per
  forward, on top of the unpacking, plus one launch per entry. Nothing times it.
- Nothing records an overlay in the reading's provenance: `weights_card` names the checkpoint's files only
  (`backend/runtime_factory.py:251-272`), and `GET /api/health` sends that card and the unpacker
  (`backend/server.py:1386-1413`). Whether another route reports the edits in force is the vault reader's to say.

### 2.2 What a general overlay would need (my analysis, on the facts above)

No general overlay exists. The pieces a design has to supply, and what the code implies for each:

| Edit | Cheapest honest form | What the code implies |
|---|---|---|
| Low-rank delta `B A` on any linear | add `(x A^T) B^T` to the linear's output | Needs no dense weight and no clone: two small products. It can be a forward hook on the linear (`ie:hooks.py:188-208` writes a module's output; any dotted module path resolves, `ie:model.py:1158-1166`), so nothing pinned is edited. It can carry a per-row mask, so the base and the variant sit in one batch. Its rounding differs from folding `B A` into the bf16 weight, so it is its own numeric class against an exported, merged weight. |
| Sparse set of entries | scatter into the unpacked weight, or add `x[..., j] * d_ij` at output i | The in-weight form needs a new forward bound from outside (as `resident` binds one) with a vectorised `index_put_` in place of today's per-entry loop. The output form extends `int4_cpu.with_overlay_terms`. |
| Dense block on a linear | write `w[r0:r1, c0:c1]` into the unpacked weight, or add `x[..., c0:c1] D^T` to outputs `r0:r1` | Same choice. A whole-matrix replacement is a per-linear resident weight: 0.215 GiB a feed-forward matrix (section 3). |
| Whole-layer variant | keep dense bf16 copies of that layer's six or seven linears | 0.89 GiB a sliding layer, 0.99 GiB a full layer (computed here). `resident` is all-or-nothing today; a per-linear resident does not exist. |
| Embedding row | in-place write with the original row saved | Tied to the unembedding, so the edit moves both unless the tie is broken (which costs 2.6 GiB). The embedding is scaled by sqrt(5376) after the look-up. |
| Norm weight, layer scalar, q_norm / k_norm | in-place write with the original saved | Small plain tensors; no overlay mechanism, no rollback, and the caches listed in 1.3 go stale. The norm computes in float32 from a bf16 weight. |

Three things any such overlay must settle that today's does not:

- **Who owns `mod.forward`.** Five parties already replace it: the pinned patch, `resident`, labkit's and R3's
  gradient linears (`labkit.py:456-482`), and `int4_cpu`. Each saves and restores what it found. A new overlay
  forward has to compose with them or refuse, as `w4a16_fast.install` refuses `resident` over int4mm linears
  (`w4a16_fast.py:438-440`).
- **Bit parity with no overlay.** An overlay layer that is installed but empty must give the pinned forward's bits.
  `w4a16_fast.same_layout`, `_probe` and `_self_test` are the existing way to prove that at install.
- **The record.** A reading taken under an edit must say so; nothing in the weights card or the model id does.

---

## 3. Memory

| What | Value | Where recorded, when |
|---|---|---|
| The card as torch counts it | 31.36 GiB (32,607 MiB by nvidia-smi) | `R3_00_pilot.json` (OOM text; `cap.total_gib` 31.355 in `R3_gpu_runs.jsonl`), 22 October 2026 |
| The model alone at rest | 18.00 GiB allocated | `R3_00_pilot.json` `vram.after_load`, 22 October 2026 |
| The model and the J++ lens at rest | 21.2 GiB (21.24 in the lab daemon's log) | `engine_runtime.py:470-471` (18 October 2026); `experiments/results/labd.log` (16 October); `scripts/atlas_build.py:22-23` (27 October) |
| The J++ lens on the card | 3.2 GiB (59 matrices of 5376 x 5376 in bf16; worked out) | `scripts/census_run.py:100-101`, `COMPUTE.md:87-88` |
| Peak during the load | 23.02 GiB allocated | `R3_gpu_runs.jsonl`, 22 October 2026, before the vision tower was moved to the CPU on 12 November (about 1.07 GiB less since, by `engine_runtime.py:385`; the new peak itself not recorded) |
| What the desktop holds | 3.4 to 5.4 GiB | `R3_00_pilot.json` (3,406 MiB used before load); `engine_runtime.py:488-490` (4.5 and 5.4 GiB) |
| Worst sweep the server runs, 2,048 word-pieces | torch's peak 25.5 GiB; the process's share of the card 26,857 to 26,904 MiB | `engine_runtime.py:479-490`, 18-19 October 2026 |
| A sweep's parts above the 21.2 GiB rest, at 512 / 1024 / 1536 / 2048 pieces | recapture 0.95 / 1.00 / 1.06 / 1.12 GiB; writing 8 pieces 1.57 / 2.25 / 2.92 / 3.76; kinship 1.21 / 2.00 / 2.91 / 3.93 | `engine_runtime.py:470-478`, 18 October 2026 |
| Census batch at `--max-new 64`, torch's peak above the model | 32 rows 2.4-3.1 GiB; 64 rows 4.8-5.3; 96 rows 7.2-7.5 (the lens freed); 128 and more not tried | `atlas_unpacker_steps_rtx5090.json`, `COMPUTE.md:52-60`, 28 October 2026 |
| A row's cost | about 0.8 MB per word-piece of context | `scripts/census_run.py:97-99`, 27 October 2026 |
| Unpacking transient of the widest linear | reference 12 bytes a weight = 1.39 GB by arithmetic (`w4a16_fast.py:306-315`, `cpu_memory.py:20-21`); measured peaks at one row 0.96 / 0.50 / 0.29 GiB for reference / lut / compiled | as above |
| Resident weights, all 410 | 58.6 GB (54.6 GiB), computed; "about 60 GB", "62 GB" in the documents | `COMPUTE.md:44`, `engine_runtime.py:411` |
| Limits in code | one sweep holds at most 2,048 word-pieces (`MAX_SWEEP_POSITIONS`); sliding window 1,024 | `engine_runtime.py:493`, `:895-905` |
| Other jobs above the model | lens build 1.5 GiB; stats pass at `--seq 512` 3.4 GiB; wiring 4.3 GiB | `scripts/atlas_build.py:24-31`, 27 October 2026 |

**Headroom (computed here).** Beside a desktop holding about 4.5 GiB, with the lens loaded, about 5.6 GiB is free at
rest; without the lens about 8.8 GiB. Headless, about 10 and 13 GiB.

**Could two variants of a layer be held at once?** In memory, yes. A dense bf16 copy costs 0.215 GiB a feed-forward
matrix, 0.65 GiB a layer's three, 0.89 GiB a whole sliding layer and 0.99 GiB a whole full layer. A handful of dense
variants fits beside a small batch; a few dozen would take the headroom a batch needs. In code, no mechanism holds
one: `resident` keeps all or none and is refused on this card, and the overlay dicts hold columns and rows, not
matrices. A low-rank or sparse variant held as its factors costs next to nothing.

**CPU.** A reading on the CPU first asks `cpu_memory.ensure`, which sets an estimate against `MemAvailable` and
refuses in plain words (`backend/cpu_memory.py:159-230`); its estimates lean high by about half (`:21-26`). Nothing
of it touches a GPU reading.

---

## 4. Speed

Everything recorded for the home RTX 5090 unless said otherwise.

| What | Time | Where, when |
|---|---|---|
| Cold start: model and J++ lens | about 16 s; the lab daemon logged "ready in 13.3s" with both lenses | `scripts/atlas_build.py:22-23` (27 October 2026); `experiments/results/labd.log` (16 October) |
| `compiled` install with its self-test on 410 linears | 4.23 s; 1.82 s in a later record; "about 4 s at load" | `atlas_unpacker_steps_rtx5090.json`; `atlas_writer_floor_rtx5090.json`; `COMPUTE.md:59` |
| `lut` install | 1.06 s | `atlas_unpacker_steps_rtx5090.json` |
| Decode step, reference, 1 / 32 / 64 / 96 rows | 0.706 / 0.713 / 0.721 / 0.743 s | same file, 28 October 2026 |
| Decode step, lut | 0.275 / 0.282 / 0.287 / 0.303 s | same |
| Decode step, compiled | 0.091 / 0.103 / 0.109 / 0.126 s | same |
| Prefill of a census question, compiled, 1 / 32 / 64 / 96 rows | 0.10 / 0.26 / 0.45 / 0.60 s (reference 0.89 / 0.89 / 1.08 / 1.27) | same (`prefill_s`) |
| One whole-sequence forward (a sweep's recapture), reference | 1.1 s at 512 pieces, 1.3 s at 1,024, 1.9 s at 2,048 | `engine_runtime.py:490-491`, 18 October 2026 |
| One forward of a 128-piece prompt, no gradient, reference | 0.649 s (0.665 s for the longest, 141 pieces) | `R3_00_pilot.json`, 22 October 2026 |
| Three pushed answers of 80 word-pieces through the server, compiled | 8.3 s (50 to 54 s on reference) | `COMPUTE.md:61-64`, 11 November 2026 |
| A heard vault edit (two answers and the edit) | 12.6 s; the unpacker and date not stated | `backend/server.py:1453-1455` |
| Corpus pass | 1.75 ms a word-piece at `--seq 512` on reference (58 min for 2,000,054 pieces); 20 min with compiled | `scripts/atlas_build.py:26-27`, `COMPUTE.md:23` |
| Steer lens, a partial-stack forward | 20.5 to 21.9 s a layer-forward for a whole layer's neurons, compiled; 10.5 hours for all 60 | `experiments/results/atlas_steer_timing/summary.json` (29 October); `COMPUTE.md:169-171` (13 November) |
| Whole-layer census, `break_rel`, batch 64, `--max-new 48`, compiled | 13.2 hours at 11.88 answers a second | `COMPUTE.md:19`, 29-30 October 2026 |
| A GGUF's packed layers on the card | 0.54 to 0.82 s a word-piece on reference, 0.12 to 0.13 s compiled | `COMPUTE.md:73-77`, 2 November 2026 |
| The checkpoint on the CPU (Ryzen 9 7900X), reference | a prefill of 17 to 20 pieces in 31 to 37 s | `COMPUTE.md:257-260`, 29 October 2026 |
| int4mm on the CPU, from a QAT Q4_0 GGUF | 0.45 s a word-piece (not the reference's bits) | `COMPUTE.md:263-264`, 2 November 2026 |

**Where the time goes.** On the reference, unpacking: "every forward unpacks all 29.3 billion 4-bit weights ... about
32 bytes of memory traffic a weight, where the products themselves read 2", and "the products are under a tenth of
the step until the batch passes about a hundred rows" (`COMPUTE.md:28-31`). That is why the step is flat from 1 to
64 rows. With `compiled` the step grows with the batch, "the products now counting beside the unpacking"
(`COMPUTE.md:59-60`). The split itself "has not been timed" (`w4a16_fast.py:8`). A probe on GGUF blocks found a
21,504 x 5,376 layer unpacking in 0.24 to 0.46 ms compiled against 0.147 ms for its one-row product
(`COMPUTE.md:83-86`, not kept), so unpacking is still the larger part at small batches.

**Computed here, not measured.** About 1.5 ms a layer at one row with `compiled` (0.091 s / 60) and 11.8 ms on the
reference. So one hypothesis tested by a single short forward costs about a tenth of a second, a batch of 32
variants of a short prompt about a quarter of a second, and a forward that starts at layer 45 about a quarter of a
whole one. A resident path would remove the unpacking; how fast it would then be was never timed.

---

## 5. Determinism

**Bit-reproducible (stated by the code or its records):**

- Every unpacker gives the reference's weights bit for bit, tested at install (`w4a16_fast.py:21-32`, `:366-407`).
  "The batch fixes the bits, the unpacker does not" (`COMPUTE.md:89-92`).
- The same answer written alone twice is the same (`atlas_writer_floor_rtx5090.json`, `alone_twice_same: true`;
  the script fails otherwise, `scripts/verify_atlas_writer.py:12-13`).
- Rows of one forward: "rows of one forward are bit-identical" (`ARCHITECTURE.md:544-545`).
- `capture_rows` keeps the same values `EagerModel.capture` returns, checked with `torch.equal` on a stand-in
  (`engine_runtime.py:1410-1416`).
- The allocator's expandable segments move memory and change no reading (`engine_runtime.py:29-41`).

**Not bit-reproducible:**

- **Batch shape.** "On a GPU the rows of different batch shapes are not bit-identical (cuBLAS chooses its kernels by
  the shape)" (`scripts/verify_atlas_writer.py:5-7`). Recorded: KL up to 0.014 between batch shapes
  (`ARCHITECTURE.md:545`). An unpushed greedy answer in a census batch of 64 parts from the same answer written alone
  after 22 word-pieces at worst, and in a batch of 32 after 54 (`atlas_writer_floor_rtx5090.json`, 29 October 2026).
- **Card class.** "Hopper and Blackwell kernels differ from the 5090's, so a greedy answer at a near tie may fall the
  other way" (`COMPUTE.md:213-216`).
- **Device and dtype.** A reading on the CPU or in float32 is its own class and carries a suffix in its model id
  (`backend/runtime_factory.py:102-110`); the int4mm kernel adds `+int4mm` (`:36-45`).
- **The writing forward against the reading forward.** A decode step with a key-value cache and the whole-sequence
  recapture "agree to bf16 rounding", not to the bit (`engine_runtime.py:675-678`).
- **Starting part-way.** `forward_from_layer` is checked against the whole forward with `allclose(atol=1e-5)` on a
  stand-in, not by bits (`scripts/check_atlas_index.py:376-378`).
- **An output-side delta against a weight-side edit** (`int4_cpu.with_overlay_terms`): different sums, by design.
- No determinism switch is set anywhere: I found no `use_deterministic_algorithms`, `cudnn.deterministic`,
  `CUBLAS_WORKSPACE_CONFIG` or TF32 setting in `backend/`, `scripts/` or `experiments/`.

**What the code does to keep comparisons honest:**

- The base row sits in the same forward as the pushed rows (`ARCHITECTURE.md:544-545`).
- The batch size and the batch plan are resume keys of a census run, and each run records its card, versions,
  commit and the stats index's sha256 (`COMPUTE.md:89-92`).
- Every shard writes a canary batch and the merge compares them (`COMPUTE.md:93-95`).
- The writer's floor is measured per card class and shown on the census card
  (`scripts/verify_atlas_writer.py:1-34`): a pushed answer that parts from an unpushed one no sooner than the floor
  "may have parted by the shape alone".
- The model id names the reading class; a run resumes only on the same id and the server reads an index only for
  it (`backend/runtime_factory.py:36-45`).
- Every reading records the unpacker (`w4a16_fast.describe`, `:563-580`) and the weights card
  (`runtime_factory.weights_card`).
- A sweep's writing and its recapture run inside one steering block (`engine_runtime.py:1464-1488`, `:1587-1633`).
- A vault edit is heard with one answer just before and one just after, under one lock with nothing between
  (`backend/server.py:1443-1457`).
- Decoding is greedy by default, and a sampled answer records its seed (`engine_runtime.py:500-517`, `:1561-1565`).

---

## 6. Hooks, partial forwards and autograd

### 6.1 Addresses

The model is transformers' eager Gemma 4 with real `nn.Module` linears, so ordinary forward hooks fire on all of
them; the instance-level `forward` does not bypass `__call__`. interp_engine resolves a named point to a module and a
side (`ie:model.py:1149-1400`):

| Point | Where it is read | Line |
|---|---|---|
| `resid_pre` | the decoder block's input | `ie:model.py:1184-1189` |
| `resid_mid` | the input of `pre_feedforward_layernorm` | `:1203-1230` |
| `resid_post` | the decoder block's output, which is after the layer scalar | `:1181-1183`; hf:1429 |
| `mlp_act` | the input of `down_proj` (the 21,504 activations) | `:1289-1290` |
| `mlp_pre`, `mlp_pre_linear` | the outputs of `gate_proj` and `up_proj` | `:1291-1299` |
| `mlp_in`, `mlp_out`, `mlp_out_post` | the feed-forward block's boundaries and the post norm's output | `:1190-1196`, `:1328-1334` |
| `attn_in`, `attn_out`, `attn_out_post` | the attention block's boundaries and its post norm's output | `:1197-1202`, `:1335-1337` |
| `z` | the input of `o_proj` (per-head output before W_O) | `:1338-1342` |
| `value`, `q_norm_in/out`, `k_norm_in/out` | inside attention | `:1265-1276`, `:1355-1364` |
| `attn_scores`, `attn_probs` | rebuilt by the capture path, no module holds them | `ie:points.py:158-181` |
| `embeddings`, `final_norm`, `lm_head` | global | `ie:model.py:1151-1156` |
| any dotted module path, with no layer | that module's output | `:1158-1166` |

`ARCHITECTURE.md:158-163` lists only four of these (`resid_pre`, `resid_mid`, `mlp_act`, `resid_post`) and describes
`resid_post` as "after post-FFN sandwich RMSNorm + residual add" without the layer scalar; the code reads it after
the scalar.

### 6.2 Capture and patch

- `HookManager.read` observes and `HookManager.write` replaces the tensor at a module's input or output
  (`ie:hooks.py:163-218`). A write may only replace element 0 of an output.
- `rt.capture_rows` is the low-memory capture: one forward from position 0 under `torch.no_grad`, each address moved
  to the CPU as its hook fires, logits for the last row only; it accepts only `resid_pre`, `resid_mid`, `resid_post`,
  `mlp_act`, `mlp_out` (`engine_runtime.py:497`, `:1397-1462`).
- Steering through interp_engine takes `AddSpec`, `ProjectionCapSpec`, `OrthogonalDecompSpec`, `NormScaledAddSpec`,
  `AblateSpec`, `SwapSpec` at a named point with an optional position mask (`ie:steer_specs.py:83-200`). The atlas's
  own pushes are another reader's area (`backend/atlas_ops.py:216-415`).

### 6.3 Running part of the stack

- **From a layer on**: `atlas_ops.forward_from_layer(lm, start, hidden, attention_mask, position_ids)` runs layers
  `start..59` and the final norm with the model's own masks and rotary embeddings, hooks firing as usual, under
  `torch.no_grad` (`backend/atlas_ops.py:1017-1043`). The steer lens is built with it
  (`backend/atlas_build.py:824`).
- **Stopping early**: I found no helper that stops a forward at a layer.
- `interp_engine.steer.forward_from(position)` is a different thing: it declares which *position* a forward starts
  at, for position masks (`ie:steer.py:315-329`). It does not start at a layer.

### 6.4 Autograd

- **By default nothing builds a graph.** The runtime builds `EagerModel` with the default `requires_grad=False`
  (`engine_runtime.py:845-851`), which freezes every parameter (`ie:model.py:731`); `capture(detach=False)` would
  refuse (`ie:model.py:1457-1458`). The unpackers are under `@torch.no_grad()` (`engine_runtime.py:295`), and so are
  the runtime's captures, the lab daemon's `/exec` (`labd.py:212`) and all of `atlas_ops`.
- **A backward can run and has.** The pinned forward's `F.linear` is differentiable in `x`, but it would keep each
  dense weight alive for the backward, about 58.6 GB over a whole forward. The study's answer is a custom autograd
  function that saves only the packed words and scales and unpacks again in the backward:
  `labkit._W4A16LinearFn` and the context manager `Lab.grad_linears` (`experiments/lab/labkit.py:417-482`), and
  R3's `W4A16GradLinear` in the `setup_context` style so that batched (`vmap`) cotangents pass through it
  (`experiments/scripts/R3_gpu.py:63-83`). Both return `None` for the packed words and the scales: they give the
  gradient with respect to the input only.
- **Runs on the 31B on this card** (`experiments/results/R3_gpu_runs.jsonl`, 22 October 2026, the server stopped,
  each job capped at 24 GiB, no lens): a router pass of 2,688 s, eight fit shards of about 1,713 to 1,987 s each, a
  weights job of 633 s and a J-lens job of 367 s, each peaking at 21.8 to 23.0 GiB allocated; 5.54 of 8 approved
  GPU-hours (`COMPUTE.md:222`). R1 before it: `Lab.same_pos_vjp` (`labkit.py:485-519`), and
  `experiments/results/R1_00_vjpcheck.json` records a reference VJP of 1.44 s and 128 batched directions in 2.2 s at
  a peak of 25.0 GB.
- **What a gradient costs** (`experiments/results/R3_00_pilot.json`, 22 October 2026, prompts cut to 128 pieces,
  reference unpacker):

  | | Time | Peak allocated |
  |---|---|---|
  | forward, no graph | 0.649 s | |
  | forward keeping the graph | 0.661 s | 20.04 GiB (2.0 above the 18.0 rest) |
  | one backward pass, one cotangent | 0.314 s | 20.18 GiB |
  | one pass, 8 / 16 / 32 cotangents by vmap | 0.495 / 0.718 / 1.199 s | 20.78 / 21.47 / 22.84 GiB |
  | 64 cotangents, or 4 prompts at once | out of memory under the 24 GiB cap | 23.1 to 23.2 GiB |

  The fit read gradients at source layers 30 to 58 (`experiments/scripts/R3_common.py:163`); whether the timed
  backward crossed all 60 layers or only the upper 30 is not established. The same passes under `compiled` were
  never timed (it was built six days later).
- **Gradient with respect to a weight delta: not built.** For a low-rank delta applied on the output side with
  plain torch ops, autograd would give the factors' gradients for free once the base linears run through
  `grad_linears`. A dense gradient of one feed-forward matrix is 115.6 M numbers, 0.43 GiB in float32 (computed
  here), and could be formed from the linear's captured input and its output's gradient. Neither has been run.
- **bf16**: the forward and the backward ran in bf16 through the linears and the norms compute in float32
  (hf:219-223). interp_engine's own verdict table has no entry for `compressed-tensors`, which it treats as a caveat
  and not a blocker (`ie:facts.py:895-922`, `ie:autograd_support.py:205-221`).

---

## 7. Pins

**R4** (`experiments/scripts/R4_common.py:81-103`, `PINNED_UNCHANGED`, registered 28 October 2026) pins by sha256:

- four runtime files: `backend/__init__.py`, `backend/theodolite_math.py`, `backend/engine_runtime.py`,
  `backend/label_gate.py`;
- twelve study scripts and prompt files: `R1_common.py`, `R2_common.py`, `R2_03_analyze.py`, `R2_prompts.json`,
  `R3_common.py`, `R3_05_eval.py`, `R3_06_analyze.py`, `R3_gpu.py`, `R3_fitlib.py`, `R3_lrp.py`, `R3_prompts.json`,
  `R3_sources.json`;
- five result files: `R2_01_check.json`, `R2_02_eval.json`, `R2_03_analysis.json`, `R3_05_eval.json`,
  `R3_06_analysis.json`.

It also pins the model snapshot's blobs (`:56-66`), torch `2.14.1+cu130` and transformers `5.20.0` (`:67-68`), and
R1's lens file by sha256 and size (`:72-75`). R4_02 and R4_00 "refuse to run while any differs" (`:80`).

**R3** (`experiments/scripts/R3_common.py:236-248`) pins a runtime commit (`d5405ee9...`) and eight files, among them
older hashes of `theodolite_math.py` and `engine_runtime.py`. Main left those long ago; R3 reruns go from the
`it-jpp-refit` worktree (`ARCHITECTURE.md:1530-1531`).

**The state of R4's runtime pin today (I computed the hashes; I did not run any check script):**

| File | sha256 now | R4's pin | Holds |
|---|---|---|---|
| `backend/__init__.py` | `4af5cefc...` | `4af5cefc...` | yes |
| `backend/theodolite_math.py` | `d738dc2f...` | `d738dc2f...` | yes |
| `backend/label_gate.py` | `83a998b7...` | `83a998b7...` | yes |
| `backend/engine_runtime.py` | `f1a0b4c3...` | `69eed1b4...` | **no** |

- The pinned hash is the file at commit `45f1630` (22 October 2026). Two commits of 12 November 2026 changed it:
  `a766058` ("Load the vision tower on the CPU", 34 lines added) and `960b9ab` ("Review fixes: the audio tower in
  the split map"). They added `text_only_device_map` and `settle_text_only_device_map` and changed `load()`
  (`engine_runtime.py:380-407`, `:809-842`). The unpacker and the forward are untouched.
- `scripts/check_dequant.py:725-739` compares these four files with R4's table, so its seventh section would fail on
  `engine_runtime.py` as the code stands. It was last changed on 11 November, before the edit.
- The documents still say the pin holds: `ARCHITECTURE.md:133` ("whose file R4 pins by sha256"),
  `backend/w4a16_fast.py:2-3`, `COMPUTE.md:46-47`, `backend/gguf_runtime.py:4`.
- The stated policy is that a change to the pinned file means "R4 reruns would then go from R4's registered commit"
  (`ARCHITECTURE.md:1530-1531`), and `ROADMAP.md:1609-1611` says mending the file "waits on R4's pin". On 2 November
  the owner had a rename's break of this pin put right (commit `2a3330c`, "R4's pin holds again").
- Whether the owner knows the pin broke on 12 November, and whether it was meant, is not established.

**What it implies for new code.** By convention the four runtime files are not edited: new behaviour is installed
from outside, as `backend/w4a16_fast.py` does (rebinding the module global `dequantize_w4a16_full`, or binding a new
`forward` on each packed module and saving the old one), as `backend/runtime_factory.py` does (a subclass,
`CheckpointRuntime`), and as `backend/gguf_runtime.py` does. A weight-space harness should live in new modules under
`backend/` and scripts beside them, and should raise the broken pin with the owner before relying on either reading
of it.

The files ending `.orig` (`engine_runtime.py.pre_bos.orig`, `engine_runtime.py.pre_module1.orig` and three others)
are old copies; no Python file or document I searched refers to them.

---

## 8. Process model

- **One process owns the card.** "Run one GPU process at a time: this loads the model, as the server and the lab
  daemon do" (`scripts/atlas_build.py:19-20`, `scripts/census_run.py:115-116`). Every entry point builds its own
  runtime through `runtime_factory.make_runtime()` and calls `.load()`: the server (`backend/server.py:122`),
  `census_run.py` (`:253-255`), `atlas_build.py`, the lab daemon (`experiments/lab/labd.py:161`), the standalone
  scientist, and the GPU checks.
- **Nothing arbitrates between processes at home.** The lab daemon and the server share port 8765 by default, so
  one of them fails to bind (`labd.py:22-24`, `:241-250`). The scripts' help says to stop the server first
  (`scripts/verify_atlas_writer.py:33-34`). Only the rented-pod scripts take a file lock (`scripts/pod/common.sh:104-110`).
- **Inside the server**, one `asyncio.Lock`, `_GPU_LOCK` (`backend/server.py:99`), is held around a sweep, a vault
  statement, a scientist tool and every atlas route (`:1019`, `:1463-1604`, `:1660`). Synchronous GPU work runs in a
  worker thread under it so the event loop keeps answering (`:1451-1457`). The scientist's tools take no lock of
  their own when the server bound the runtime (`backend/mcp_scientist.py:85-103`).
- **How the LLM agent reaches the model.** The scientist chat runs a CLI (Claude Code or Codex) whose MCP server,
  `backend/scientist_mcp.py`, is a stdio process that calls the instrument's server over HTTP with a 900 s timeout
  (`:1-33`). It never holds the model itself.
- **The lab daemon is the existing code-execution environment** (`experiments/lab/labd.py`): it keeps the model in
  one process and runs submitted Python against it under one lock, with `rt`, `lab`, `torch`, `np`, `F` and a
  persistent `STORE` dict in scope, top-level `await` allowed, the result returned as JSON and written to a file on
  request, a per-launch token, loopback only, every call logged (`:10-34`, `:198-237`). Its limits: the code runs
  under `torch.no_grad`, synchronously on the event loop (so `/health` waits unless the code awaits), with no
  timeout and no sandbox, and after a call it clears only labkit's hooks (`:219`), so a weight, an overlay or an
  unpacker a job changed stays changed for the next. The study's log holds 314 calls, 282 of them ok, with a median
  of 71 s (`experiments/results/labd_exec_log_2026-10-16.jsonl`).
- **Cold start**: about 13 to 16 s for the model and the J++ lens (section 4), then 2 to 4 s for the compiled
  unpacker's test. `backend/server.py:114` says the load "takes minutes on the 31B"; the two recorded figures say
  seconds. Whether a first load from a cold disk takes minutes is not established.
- **The start-up choices** are environment variables read once: `THEODOLITE_UNPACKER`, `THEODOLITE_DEVICE`,
  `THEODOLITE_GGUF` and its companions (`backend/runtime_factory.py:1-51`, `backend/server.py:117-133`).

---

## 9. Existing pieces a harness could stand on

- `backend/w4a16_fast.py`: the pattern for adding a forward path without touching the runtime file (`install`,
  `_uninstall`, `Refused`), and its bit-level test kit (`same_bits`, `same_layout`, `_probe`, `_self_test`,
  `piece_scale_table`, `table_agrees`).
- `w4a16_fast._make_resident_forward`: a template for a per-linear forward over a kept dense weight, with overlay
  handling and a tamper check.
- `engine_runtime.dequantize_w4a16_full / _rows / _cols`, and the runtime's `slice_w_*` and `get_full_w_*`.
- `gguf_weights.dense`, `dense_cols`, `dense_rows`, `with_overlays`; `atlas_ops.dense_weight`, `Weights`,
  `attn_shape`, `layer_scalars`.
- `int4_cpu.with_overlay_terms`: an overlay as a delta on the product.
- `interp_engine`: `HookManager.read / write`, `EagerModel.resolve_point`, `capture`, `steer`.
- `TheodoliteRuntime.capture_rows`, `write_voice`, `capture_for_sweep`.
- `atlas_ops.forward_from_layer`.
- `labkit._grad_linears` and `_same_pos_vjp`; `R3_gpu.W4A16GradLinear` and `grad_linears` (R3's is pinned by R4, so
  it is to be imported or copied, not edited).
- `experiments/lab/labd.py` and `labclient.py`.
- `runtime_factory.make_runtime`, `weights_card`, `reading_class`.
- `cpu_memory` and `w4a16_fast.free_bytes`.
- `scripts/pod/pod_tool.py steps` (the step timer), `scripts/verify_atlas_writer.py` (the noise floor),
  `scripts/check_dequant.py` and `scripts/atlas_stand_in.py` (a tiny random Gemma 4 with 4-bit linears, for checks
  on the CPU with no card).

## 10. Not established

- Whether `weight_packed` and `weight_scale` are registered as parameters or as buffers by compressed-tensors.
- The load's peak memory since the vision tower moved to the CPU (12 November).
- The split of a `compiled` decode step between unpacking and products.
- How fast a resident forward is, and how fast a partly resident one would be.
- What an overlay costs per forward (the clone and the per-entry loop).
- Whether the timed backward of R3's pilot crossed all 60 layers.
- A backward's time under `compiled`.
- Whether the owner knows R4's pin on `engine_runtime.py` no longer holds.
- Whether any route reports the vault's edits in force beside a reading (the vault reader's area).
- Whether a requantiser from a dense edited weight back to the 4-bit grid exists (the export reader's area; none in
  the files I was given).
