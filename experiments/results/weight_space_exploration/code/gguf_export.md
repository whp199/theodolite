# Reading and writing whole models; how Gemma-specific the stack is

Reader's area: `backend/gguf_model.py`, `backend/gguf_weights.py`, `backend/gguf_runtime.py`, `backend/gguf_fast.py`,
`backend/int4_cpu.py`, the five check and compare scripts, ARCHITECTURE.md section 5 (lines 1152-1551) and COMPUTE.md
section vii (lines 239-277). All paths below are relative to `(the repository root)` unless they start with `/`.

## 0. What was read, what was run

- Read in full: the five backend files, `backend/runtime_factory.py`, ARCHITECTURE.md 1152-1551, COMPUTE.md 36-70 and
  239-277, `experiments/results/gguf_31b_gpu/README.md` and two of its JSON records, `experiments/results/gguf_31b_cpu/README.md`.
- Read in part (docstring, outline, the sections cited): the five scripts, `backend/w4a16_fast.py` (1-135),
  `backend/engine_runtime.py` (284-442, 800-893), `backend/atlas_ops.py` (60-260), `backend/larql_vault.py` (464-567),
  `scripts/export_lora.py` (1-110, 440-655), `scripts/atlas_stand_in.py`.
- Nothing in the repository was edited; no model was loaded; torch was never imported. One thing was run beyond grep:
  gguf-py's `GGUFReader` (numpy only, torch not imported, checked) over the header of a 12B GGUF that sits in the LM Studio
  folder outside the repository, to list its keys and tensor types (section 6).
- Where a statement is my own derivation from a format's layout rather than something the code or a document states,
  it is marked "derived".

## 1. Reading a model from a GGUF

### 1.1 The pipeline (code, checked)

1. `gguf_model.open_gguf` (`backend/gguf_model.py:118-155`) opens the file with `gguf.GGUFReader(path, "c")`, a
   copy-on-write memory map: torch reads blocks in place and nothing is ever written back. A split file is read from any
   shard as a `SplitGGUF` (`:99-107`).
2. `text_config_from_gguf` (`:204-308`) turns the `gemma4.*` keys into `Gemma4TextConfig` keyword arguments. It refuses any
   arch other than `gemma4` (`:214-215`), a mixture-of-experts model (`:216-217`), per-layer embeddings (`:218-219`),
   shared key-value layers (`:220-221`) and a file lacking any of the ten `NEEDED_KEYS` (`:192-194`, `:222-225`).
3. `build_gemma4` (`:485-647`) builds transformers' `Gemma4ForCausalLM` with parameters on the meta device (no random
   weights allocated: `_parameters_on_meta`, `:395-410`), then fills it by gguf-py's own name map for the gemma4 arch
   (`hf_to_gguf_names`, `:465-482`). The seven linears of each block (`LINEAR_NAMES`, `:83-84`) are replaced one by one
   (`:562-596`); embeddings, norms and layer scalars are unpacked to the model's dtype as plain parameters and buffers
   (`:603-619`); `lm_head` is tied to the embedding (`:620-621`). A file that leaves a parameter unfilled or holds a
   tensor the model has no place for is refused (`:627-633`).
4. `GGUFRuntime` (`backend/gguf_runtime.py:60-162`) subclasses the pinned `TheodoliteRuntime` and replaces only `load()` and
   the seven accessors that read the packing (`slice_w_down/gate/up`, `get_full_w_down/gate/up`, `write_col_norms`,
   `:166-253`). Everything else is the pinned runtime's own code.
5. `runtime_factory.make_runtime()` (`backend/runtime_factory.py:154-212`) picks the runtime from environment variables
   (`THEODOLITE_GGUF`, `THEODOLITE_DEVICE`, `THEODOLITE_GGUF_LINEARS`, `THEODOLITE_GGUF_RESIDENT_GB`, `THEODOLITE_DTYPE`,
   `THEODOLITE_GGUF_LENS`, `THEODOLITE_CHECKPOINT_LINEARS`, `THEODOLITE_GGUF_CONFIG`).

### 1.2 Quantisation types (code, `backend/gguf_weights.py`)

| class | types | where |
|---|---|---|
| unpacked in torch, bit-equal to gguf-py 0.19 (15) | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS | `GGML_TYPES`, `:48-64` |
| unpacked once at load by gguf-py into dense layers (11) | IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ1_M, TQ1_0, TQ2_0, MXFP4, NVFP4 | `BY_GGUF_PY`, `:67-71` |
| refused in plain words (8) | Q8_1, Q8_K, I8, I16, I32, I64, F64, Q1_0 | `NOT_YET`, `:73-75` |

Blocks run along `in`: a weight [out, in] is stored as rows of blocks of 32 elements (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
IQ4_NL) or 256 (the K types, IQ4_XS), each block with its own fp16 scale and, for some types, a minimum
(`backend/gguf_weights.py:4-7`). COMPUTE.md:243 says "thirteen GGUF types"; that was the count on 28 October before
IQ4_NL and IQ4_XS were added on 2 November (COMPUTE.md:251), so document and code agree.

### 1.3 How layers are held, and the cost

| holding | what it is | memory (31B) | forward | bits | where |
|---|---|---|---|---|---|
| packed (default) | `GGUFLinear` keeps the raw blocks, unpacks the whole weight at every forward | the file's size at rest; 18.93-21.2 GiB on the card after loading | pays the unpacking each forward | the reference | `gguf_weights.py:416-532`; ARCHITECTURE.md:1487 |
| resident | same layer, unpacked once at load, blocks dropped | 2 bytes a weight in bf16: 59 GB for the linears, about 62 GB whole | a plain product | same bits as packed | `gguf_weights.py:469-474`; `int4_cpu.py:11-12`; ARCHITECTURE.md:1362 |
| partly resident | packed layers made resident in model order while they fit a GB budget | the budget | mixed | same bits as packed | `gguf_model.py:505-512`, `:581-588` |
| dense | a plain `nn.Linear` with an overlay-aware forward | as resident | a plain product | same bits as packed | `gguf_model.py:589-595` |
| int4mm (CPU only) | `Int4GGUFLinear`: torch's `_weight_int4pack_mm_for_cpu` on Q4_0, Q4_1, Q4_K; blocks kept for the accessors | 0.625 bytes a weight: 18 GB for the linears | no unpacking | NOT the reference's bits; model id ends `+int4mm` | `int4_cpu.py:229-253`, `:61-64` |

A full resident or dense 31B (62 GB) does not fit the 32 GB card; packed, or packed with a budget, is the only GPU
holding. How the layers are held is not in the model id because packed, resident and dense give the same forward bit for
bit (`gguf_model.py:79`; checked on the stand-in in float32 and bf16, `scripts/check_gguf.py:439-445`, `:462-464`).

The faster unpacker, `compiled` (`backend/gguf_fast.py`), runs the reference's own arithmetic through `torch.compile`,
one graph per (type, blocks shape, device, dtype), with the model itself left eager so every hook still fires. At install
it compares every packed layer's whole weight with the reference by bits and strides, and every one of the 65,536
patterns of each fp16 field through each graph, and refuses on any difference (`gguf_fast.py:270-315`, `:318-386`).
Only whole-weight unpacking goes through the graphs; the row and column slicers stay on the reference (`:36-42`).

### 1.4 Measured speeds

GPU, the home RTX 5090, 2 November 2026, three Q4 GGUFs of the 31B, packed, bf16
(ARCHITECTURE.md:1479-1491; `experiments/results/gguf_31b_gpu/README.md`):

| | QAT Q4_0 (with the J++ lens) | Q4_K_M | UD-Q4_K_XL |
|---|---|---|---|
| load | 22.3 s | 21.5 s | 20.5 s |
| on the card after loading | 21.2 GiB | 18.93 GiB | 19.18 GiB |
| a written word-piece, reference / compiled | 0.542 / 0.121 s | 0.819 / 0.126 s | 0.758 / 0.121 s |
| a step at a batch of 32, reference / compiled | 0.553 / 0.131 s | 0.736 / 0.136 s | 0.737 / 0.132 s |
| compiled's install with its tests (graphs) | 2.8 s (8) | 8.7 s (10) | 10.4 s (19) |
| peak, reference / compiled | 22.71 / 23.05 GiB | 20.57 / 20.33 GiB | 20.89 / 20.57 GiB |

- One forward of "The capital of France is" on the QAT Q4_0: 0.555 s on the reference, 0.14 s compiled
  (`experiments/results/gguf_31b_gpu/qat_q4_0.json`).
- A whole sighting (60 layers through the lens, 6 written): 5.22 s on the reference, 1.92 s compiled (ARCHITECTURE.md:1503-1504).
- The W4A16 checkpoint itself, same card, 28 October 2026 (COMPUTE.md:55-58): 0.706 / 0.275 / 0.091 s a step at one row
  on reference / lut / compiled, and 0.743 / 0.303 / 0.126 s at 96 rows. Since 11 November the server takes `compiled`
  by default on a CUDA card (COMPUTE.md:61-65).
- One layer of 21,504 x 5,376 unpacks in 0.24-0.46 ms compiled against 1.3-3.6 ms on the reference, where its one-row
  bf16 product takes 0.147 ms (ARCHITECTURE.md:1317-1320): unpacking still costs more than the product.
- The reference's step moved by 6 to 19% between two runs of the same file; the cause was not established
  (ARCHITECTURE.md:1493-1495).

CPU, the 12-core Ryzen 9 7900X with 61 GB, 2 November 2026:

| path | load | speed | memory | where |
|---|---|---|---|---|
| packed, reference (QAT Q4_0) | 15.8 s | one 6-piece forward 37.2 s | about 20 GB | `gguf_31b_gpu/qat_q4_0_cpu.json`; ARCHITECTURE.md:1360 |
| packed, compiled (QAT Q4_0) | install 64.3 s | the same forward 11.9 s, the reference's bits | as packed | same file |
| packed, reference (UD-Q4_K_XL) | 14.9 s | one forward 38.8 s unhindered (116 s and 185 s beside other work) | 5.0 GiB after load, 21.9 GiB at most | `gguf_31b_cpu/ud_packed.json` |
| int4mm (QAT Q4_0) | 46 s | 59-piece prefill 3.5 s, a written piece 0.45 s | about 21 GB plus the file's mapped pages | ARCHITECTURE.md:1460-1461, :1363 |
| int4mm + 8.5 GB resident (Q4_K_M) | 48 s | prefill 3.5 s, a written piece 0.56 s | 26.5 GiB own + 17.6 GiB mapped | ARCHITECTURE.md:1455-1459 |
| int4mm + 16 GB resident (UD-Q4_K_XL) | 57-68 s | a written piece 1.5 s | 40.5-43 GiB | ARCHITECTURE.md:1228-1231 |
| packed W4A16 checkpoint, reference | not stated | prefill of 17-20 pieces 31-37 s; 25 rows 43-51 s (29 October) | not stated | COMPUTE.md:257-260 |

The document's own judgement: the CPU suits weight reads, single forwards and pilots, not a census (some 50 days a layer
through int4mm, inferred, not run: ARCHITECTURE.md:1471-1474).

## 2. Writing weights, and persisting an edited model

### 2.1 Everything that writes weights (code, checked by grep for `GGUFWriter`, `save_file`, `torch.save`)

| writer | what it writes | limits |
|---|---|---|
| `gguf_model.write_gemma4_gguf` (`backend/gguf_model.py:865-999`) | a whole Gemma 4 text model as a GGUF in llama.cpp's layout, optionally split into shards | A fixture writer: "It is not llama.cpp's converter, and its Q4_K is a plain quantiser" (`:54-58`). It reads `layer.mlp.down_proj.weight` and the like (`:970-991`), an attribute `GGUFLinear` does not have, so it cannot re-emit a model loaded packed. `blocks=` hands some tensors' bytes outright (`:873`, `:948-953`). |
| `gguf_model.quantize_rows`, `quantize_q4_k` (`:799-857`) | rows to GGUF blocks | gguf-py's quantisers, and a plain per-sub-block Q4_K; `_FILE_TYPES` lists F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K (`:860-862`). |
| `export_lora.write_lora` (`scripts/export_lora.py:501-529`) | a gemma4 LoRA adapter GGUF, F32: a neuron's mul, a vault column edit (A = e_n^T, B = v - W_down[:, n]), a head's scale (`:442-479`) | One way, for llama.cpp. `read_lora` (`:532-552`) reads the arrays back but nothing applies them to the instrument's runtime (grep: used only by `--show` and `scripts/check_lora_export.py`). |
| `export_lora.write_silenced` (`:607-654`) | a copy of a model GGUF with a whole block silenced | Two patterns worth reusing: "scale" rewrites the file through `GGUFWriter`, copying every key and every tensor's bytes verbatim and adding a tensor (`:626-650`); "norm" makes a reflink or byte copy and overwrites one F32 tensor in place at `t.data_offset` (`:617-625`). |
| `export_control_vector.write_control_vector` (`scripts/export_control_vector.py:368`) | a llama.cpp control vector GGUF for add pushes | not a weight edit |
| `compare_gguf_ct.write_pair` (`scripts/compare_gguf_ct.py:302-365`) | a synthetic W4A16 safetensors folder and a Q4_0 GGUF from the same codes and scales, at the stand-in's shape | self-test fixture only |

Nothing writes the real model in the compressed-tensors W4A16 format. The vault writes nothing to disk: its edits are
in-memory overlays and records (`backend/larql_vault.py:464-567`; grep finds no file write in that module).

### 2.2 What an in-memory edit is today (code)

- Every quantised linear carries two dicts, `_memit_col_overlays` {column index: vector} and `_memit_row_overlays`
  {row index: vector}. The forward lays them on a clone of the unpacked weight, columns then rows, cast to the weight's
  dtype (`backend/gguf_weights.py:375-390`; the pinned W4A16 forward does the same, `backend/engine_runtime.py:356-375`).
- The dicts exist on all seven linears of every block, attention included: `GGUFLinear.__init__` makes both
  (`gguf_weights.py:456-457`), `install_overlay_forward` gives them to dense layers (`:393-405`), and
  `patch_compressed_tensors_for_low_vram` to every W4A16 module (`engine_runtime.py:431-436`).
- The stored blocks are never touched (checked: `scripts/check_gguf.py:839`, `:868`), so rollback is clearing a dict.
- An edit is a bf16 vector laid over a 4-bit base. It is not on the 4-bit grid.
- The vault itself writes only `down_proj` columns (`larql_vault.py:475-478`); row overlays are honoured by the forward
  but "none is written today" (`engine_runtime.py:365-366`).

### 2.3 Options for persisting an edited model

**A. A delta file of our own (base identity + overlays). Does not exist; smallest to build; exact.**
Record the base's model id and weights card, and for each edit the tensor, kind (column or row), index and the vector
in the model's dtype. Reloading is setting the overlay dicts again; since the forward casts the overlay to the weight's
dtype, a bf16 vector stored as bf16 reloads to the same bits. One column of a 31B `down_proj` is 5,376 numbers
(10.75 kB in bf16). Derived, not tested in the repository.

**B. A LoRA adapter GGUF. Exists, one way.**
`export_lora.py` writes rank-1 pairs for column edits. It is the route to llama.cpp, which is a third, independent
implementation. Its docstring lists what differs from the instrument by construction (adapters add rather than compose;
llama.cpp acts at the start mark too; float32 arithmetic on its own quantised product: `scripts/export_lora.py:50-62`).
Loading an adapter back into the instrument would have to be built. Whether `W + B` reproduces the overlay bit for bit
was not established (no test).

**C. A patched GGUF copy. The two copy mechanics exist; applying them to a quantised tensor does not.**
- In place (reflink, then overwrite at `data_offset`): only works if the new bytes are valid blocks of the tensor's own
  type and size, which means requantising the edit (2.4).
- Rewritten through `GGUFWriter`: a GGUF may hold a different type per tensor, and the loader makes any F32, F16 or BF16
  tensor a plain dense layer in every mode (`backend/gguf_model.py:24`, `:589-595`). So an edited `down_proj` could be
  stored exactly as BF16 in an otherwise unchanged copy, at 231 MB for that tensor against 65 MB as Q4_0 (derived from
  5,376 x 21,504 weights), and it would then be dense in memory. Derived from how the loader treats per-tensor types;
  not tested for this purpose.
- **Identity hazard:** the provenance fingerprint hashes the file's size, its header, the first MiB of tensor data and
  the last MiB, and "not ... one in between" (`gguf_model.py:747-748`). A copy patched in place in the middle of the
  data, with the same size, has the same fingerprint, and the index check compares only fingerprint and header hash
  (`backend/atlas_index.py:188-190`). Only the file name inside the model id (`gguf_model.py:1002-1007`) would tell the
  copy apart. A patched copy must change the header (a `theodolite.*` record, as `write_silenced`'s "scale" mode writes)
  or the harness must hash the edited tensors itself.

**D. `write_gemma4_gguf` on an edited model.** Needs plain `.weight` layers, so a dense 31B (62 GB), and its quantisers
are not llama.cpp's. Fit for the stand-in, not for the 31B as written.

### 2.4 What requantising an edited layer does to the edit (derived from the layouts; no code does this, nothing measured)

- A Q4_0 block is 32 consecutive `in` elements of one `out` row: one fp16 scale d and 32 four-bit codes, w = d x (q - 8)
  (`gguf_weights.py:183-188`). W4A16 has the same grouping: a bf16 scale per 32 along `in`
  (`engine_runtime.py:296-309`).
- `down_proj` is [d_model, d_ff] = [5,376, 21,504]. A neuron's column crosses 5,376 blocks, one per row, and each block
  is shared with the 31 other neurons of its group of 32. So a column edit cannot be requantised in isolation:
  - keep each block's scale: the 31 neighbours stay bit-equal, and each edited entry snaps to one of 16 levels
    d x (q - 8), clipped to [-8d, 7d], with error up to |d|/2;
  - refit the block's scale to hold the new value: up to 31 x 5,376 neighbouring weights are re-rounded.
- `gate_proj` and `up_proj` are [d_ff, d_model] = [21,504, 5,376]. A neuron's row is 168 whole blocks that belong to that
  neuron alone, so a row edit requantises without disturbing any other neuron.
- A zero (the vault's DELETE) is exactly representable in Q4_0 and W4A16 (code 8) and in Q6_K (code 32, `:290-302`).
  It is not in general in Q4_1, Q4_K or Q5_K, which subtract or add a minimum (`:191-196`, `:264-287`), nor in IQ4_NL and
  IQ4_XS, whose sixteen levels (-127 ... -10, 1, 13 ... 113) have no zero (`:306`).
- Whether gguf-py's quantiser reproduces an existing block from its own dequantised values was not established; llama.cpp's
  quantiser writes d = max / -8 (`scripts/compare_gguf_ct.py:24-26`), so a stock requantisation of a whole edited tensor
  is likely to change unedited blocks too. Only re-encoding the touched blocks and copying the rest byte for byte avoids that.
- The conclusion for a designer: the exact form of an edited model is "4-bit base + higher-precision delta". A fully
  4-bit edited model is a lossy projection of the edit and should be re-tested as a new object.

## 3. How bit-equality is checked, and what can be reused

### 3.1 The checks that exist

| check | what it shows | where |
|---|---|---|
| unpackers against gguf-py's dequantiser | bit for bit, exhaustively over fp16 scales where cheap, on random blocks otherwise, on the GPU too when there is one | `scripts/check_gguf.py:4-9`, `same_bits` at `:112-118` |
| packed = resident = chunked = dense | same weight, same forward, float32 and bf16, stand-in | `scripts/check_gguf.py:349-359`, `:439-445`, `:462-464` |
| a Q4_0 layer holding a W4A16 module's codes and scales | the pinned forward's output bit for bit, overlays and all | `scripts/check_gguf.py:360-393` |
| a built model against the file | every tensor is gguf-py's dequantised one (`model_tensors_equal`) | `scripts/check_gguf.py:222-239` |
| a vault edit on a packed layer | logits equal a dense twin with the column written in; accessors see it; blocks untouched; with the edit cleared the logits are the clean ones | `scripts/check_gguf.py:825-874` |
| compiled against reference | every layer by bits and strides, every fp16 pattern per graph, then readings byte for byte: logits, three pushed answers, a vault UPDATE and ROLLBACK, a census | `backend/gguf_fast.py:270-315`; `scripts/check_gguf_fast.py:565-643` |
| compiled on the 31B on the card | logits of a forward and a prefill the reference's bits, written pieces the same alone and at a batch of 32, `fell_back` 0 | `experiments/results/gguf_31b_gpu/README.md` |
| W4A16 unpackers (lut, compiled, resident) | the pinned unpacker's bits on all 16 pieces x 65,536 bf16 scales and on a 4-bit stand-in | `scripts/check_dequant.py:1-35` |
| does a GGUF hold the checkpoint? | tensor by tensor, value for value; for Q4_0 against W4A16 also codes and scales | `scripts/compare_gguf_ct.py:170-296` |
| int4mm (not bit-equal) | the kernel's weight within rounding of the file's (`kernel_departure`); an edit that changes no weight adds exactly nought; kernel off gives the reference's logits | `backend/int4_cpu.py:119-135`; `scripts/check_gguf_cpu.py:611-617`, `:639-645` |
| the pins | the R4-pinned files unchanged | `experiments/scripts/R4_common.py:357-359` |

### 3.2 Where bit-equality stops (documents, measured)

- Across devices: CPU against GPU, same file, both on the reference: the same five likeliest pieces in the same order,
  the largest |delta log-prob| among them 0.25, KL 0.0015 (ARCHITECTURE.md:1508-1511).
- Across batch size on an AVX-512 CPU: every answer the same but a log-prob moves by up to 0.0039 on the 7900X, nought
  with `ONEDNN_MAX_CPU_ISA=AVX2` (ARCHITECTURE.md:1444-1445). "A CPU's floor is to be measured on it, as a card's is."
- Across dtype and device classes the model id differs (`@cpu/bfloat16`, `@cpu/float32`, `@cuda/float32`:
  `backend/runtime_factory.py:102-110`) and a census refuses to continue across them (`scripts/check_gguf.py:887-894`).
- Whether logits are bit-equal across batch sizes on the card was not established here; the record compares written
  pieces alone against pieces at a batch of 32, not logits.

### 3.3 Reuse for "the edit was applied exactly as recorded"

1. **Bit comparators.** `w4a16_fast.same_bits` and `same_layout` (`backend/w4a16_fast.py:88-102`) compare bit patterns, so
   a NaN equals the same NaN and -0 differs from +0; `gguf_fast.table_agrees` (`backend/gguf_fast.py:117-129`) reports
   how many numbers differ.
2. **The four-part edit check** of `scripts/check_gguf.py:825-874`, lifted as a harness routine: (a) heard: the logits
   equal a reference built another way; (b) seen: `slice_w_down`, `get_full_w_down`, `write_col_norms` return the
   recorded vector at the recorded place and the old values elsewhere; (c) the base is untouched:
   `torch.equal(mod.weight_blocks, saved)` and `dense(mod)` unchanged; (d) rollback: with the overlay cleared, the logits
   are the clean logits by bits. On the 31B, (b), (c) and (d) cost a slice and two forwards.
3. **The readings bundle** of `scripts/check_gguf_fast.py:565-595` (`readings()`): logits, pushed answers, a vault UPDATE,
   weight slices, ROLLBACK, a small census, compared between two configurations with `same_layout` for tensors and `==`
   for rows. It is a ready conformance suite between two ways of holding or evaluating the same edited model.
4. **The null edit.** Setting a column to its own value must change nothing (`scripts/check_gguf_cpu.py:611-617`). It is
   the control that separates "the edit's effect" from "a change of numeric path"; ARCHITECTURE.md:1395-1396 records
   that before this was fixed such an edit changed 13 of 30 greedy answers on the stand-in.
5. **The streaming comparator.** `compare_gguf_ct.compare_tensor` and its `Checkpoint` reader hold one tensor pair in
   memory at a time and report share equal and largest difference. Pointed at a base file and a persisted edited file it
   would verify that every tensor outside the edit record is equal and the edited ones equal base + delta at the recorded
   places. That pairing would have to be written; the pieces exist.
6. **Identity.** `gguf_model.provenance` and `runtime_factory.weights_card` name the base. The model id does not change
   when an overlay is in force, so a harness must add its own edit identity to every reading (the atlas already keys a
   cache on `vault.edits_in_force()`, `backend/atlas.py:216-219`; grep finds no such reference in `backend/census.py`).

## 4. The CPU runtime as a second evaluator of an edit

- **What is independent and what is shared.** int4mm multiplies through torch's `_weight_int4pack_mm_for_cpu`, with its
  own summation order, on other hardware, and lays an edit on as float32 difference terms added to the kernel's product
  (`int4_cpu.with_overlay_terms`, `backend/int4_cpu.py:138-164`) rather than by clone and replace. It shares the loader,
  the block decoding and the same overlay dicts with the GPU path. So it is a second arithmetic, not a second parser or a
  second edit record.
- **Speed and memory** (section 1.4): 0.45 s a written piece and a 59-piece prefill in 3.5 s from the QAT Q4_0, about
  21 GB of its own; 0.56 s and 26.5 GiB own for the Q4_K_M with 8.5 GB resident. The machine has 61 GB and the GPU model
  lives on the card, so both can run at once (the documents' CPU timings were all taken "beside other sessions' work").
- **The reference on the CPU** is bit-defined but slow: 37-39 s a forward, or 11.9 s with `compiled` after a 64 s install.
- **How far it agrees.** int4mm against the CPU reference on the Q4_K_M: the same five likeliest in the same order, KL
  0.001, largest |delta log-prob| 0.23 (ARCHITECTURE.md:1462-1465); on the UD-Q4_K_XL, KL under 1e-5 and 0.125
  (`experiments/results/gguf_31b_cpu/README.md`). CPU reference against GPU reference: 0.25. On the stand-in int4mm
  gives 30 of 64 canary answers word for word (ARCHITECTURE.md:1446). The canary on the real file "has not been run"
  (ARCHITECTURE.md:1469).
- **So:** it can confirm the direction and rough size of an edit's effect when that effect is well above about 0.25
  nats on a top log-probability, and it cannot confirm small effects. Each side needs its own floor, taken with a null
  edit and an unedited rerun. `int4_cpu.compare_rows` (`:323-363`) is the ready comparison: shared prefix, the split
  step, largest |delta log-prob|, top-5 overlap.
- **The checkpoint on the CPU.** `CheckpointRuntime` with `THEODOLITE_CHECKPOINT_LINEARS=int4mm` holds the W4A16 weights
  exactly, since a W4A16 scale is already bf16 (`backend/int4_cpu.py:21-22`; `runtime_factory.py:113-133`). A measured
  31B run of it was not found in the sections read.
- **Which object.** The lmstudio QAT Q4_0 GGUF is not the checkpoint: all 410 linears differ (11-15% of values equal),
  though the norms and layer scalars are equal (ARCHITECTURE.md:1249-1253). A GGUF evaluator therefore tests the edit on
  a different quantisation of the model unless the base is that same GGUF on both sides.
- **A third evaluator** already has an export path: llama.cpp through a LoRA adapter (`scripts/export_lora.py`), tested
  through llama-server on the stand-in (`scripts/export_lora.py:76`).

## 5. How Gemma-specific the stack is

### 5.1 What assumes Gemma 4, concretely

| assumption | where |
|---|---|
| arch name `gemma4`; any other arch refused | `backend/gguf_model.py:78`, `:214-215` |
| transformers classes `Gemma4TextConfig`, `Gemma4ForCausalLM`; the pinned load calls `Gemma4ForConditionalGeneration.from_pretrained` | `gguf_model.py:499-501`; `backend/engine_runtime.py:816` |
| tensor names by `gguf.MODEL_ARCH.GEMMA4` | `gguf_model.py:469` |
| seven linears per block by name: q, k, v, o, gate, up, down | `gguf_model.py:83-84` |
| GeGLU: `hidden_activation = "gelu_pytorch_tanh"` set by hand because the file does not state it | `gguf_model.py:302` |
| norms: `Gemma4RMSNorm` multiplies by w with no unit offset; four per block plus q_norm and k_norm | `gguf_model.py:32-37`, `:977-990` |
| layer scalars: `layer_scalar` buffers from `blk.N.layer_output_scale.weight` | `gguf_model.py:614-619` |
| attention layout: sliding and full layers, per-layer key-value heads, two head sizes, values equal to keys on full layers (no `v_proj`), proportional rope with a partial rotary factor read from `rope_freqs` | `gguf_model.py:226-301` |
| softcap: `final_logit_softcapping`; `FINAL_LOGIT_SOFTCAP = 30.0`; `unembed_gemma4` | `gguf_model.py:291-292`; `backend/theodolite_math.py:28`, `:85-103` |
| tokenizer: looked for in the two Gemma repos; end pieces `<turn|>` and `<|tool_response>`; Gemma's chat template | `backend/gguf_runtime.py:57`; `gguf_model.py:87`, `:634-638` |
| the 31B's shape as constants: `D_MODEL 5376`, `N_LAYERS 60`, `D_FF 21504`, `VOCAB_SIZE 262144`, the ten full layers | `theodolite_math.py:24-33` |
| those constants in the pinned runtime (norm loop, lens load, sweep, captures) | `engine_runtime.py:860`, `:920-928`, `:1258-1267`, `:1393-1394`, `:2410` |
| those constants in unpinned code: the vault clamps to them and ROLLBACK loops 60 layers; the server; the scientist's MCP tools | `backend/larql_vault.py:286`, `:393`, `:481`, `:500`, `:559`; `backend/server.py:347-376`, `:1395-1397`; `backend/mcp_scientist.py:249-251`, `:290` |
| the J++ lens fitted for gemma-4-31b | `engine_runtime.py:113-115`, `:124` |
| module names in the weights facade: `lm.layers[l].mlp.down_proj`, `{which}_layernorm`, `is_sliding`, `head_dim` | `backend/atlas_ops.py:96-110`, `:142-184` |
| `language_model(hf)` = `hf.model.language_model` or `hf.model` | `backend/causal_check.py:105-108` |
| the checkpoint prefix `model.language_model.` and the LoRA's arch | `scripts/compare_gguf_ct.py:57`; `scripts/export_lora.py:126` |

### 5.2 What is general

- All of `backend/gguf_weights.py`: the unpackers, `GGUFLinear`, the overlays, and `dense`, `dense_cols`, `dense_rows`,
  `packed_kind`, which read any linear the same way whether it is plain, W4A16 or GGUF (`:542-583`). Nothing in it names Gemma.
- In `backend/gguf_model.py`: `open_gguf`, `SplitGGUF`, `read_metadata`, `tensor_table`, `logical_shape`, `provenance`
  (it keeps whatever arch's keys the file has, `:774`), `_parameters_on_meta`, `_dense`, `quantize_rows`.
- `backend/gguf_fast.py` and `backend/int4_cpu.py` work layer by layer; `backend/w4a16_fast.py` works on any
  compressed-tensors W4A16 module.
- Below the instrument, `interp_engine` (1.14.0, a dependency) resolves a model's layers, final norm and lm_head
  programmatically for "the overwhelming majority of models" (`.venv/lib/python3.12/site-packages/interp_engine/arch.py:1-13`,
  `resolve_arch` at `:1126`).

### 5.3 Is there an abstraction boundary?

Partly, at three levels, none of them a model-family-neutral runtime:

1. **Per linear layer: yes, and general.** `gguf_weights.dense / dense_cols / dense_rows / with_overlays`.
2. **Per runtime: a single choice point, not an interface.** `runtime_factory.make_runtime()` returns `TheodoliteRuntime`,
   `CheckpointRuntime` or `GGUFRuntime`, the last two subclasses of the first, chosen from environment variables. There
   is no protocol class. `GGUFRuntime` shows the pattern for a second weights source (replace `load()` and seven
   accessors), but it inherits every 31B constant of the pinned parent.
3. **Per model: a facade with Gemma names inside.** `atlas_ops.Weights(lm, runtime)` (`backend/atlas_ops.py:128-209`) gives
   `w_down`, `w_gate`, `w_up`, `down_cols`, `gate_rows`, `up_rows`, `norm`, `attn_weight`, `write_ceiling`, and takes its
   shape from the model (`n_layers`, `d_model`, `d_ff`), not from the constants. It is the nearest thing to a `Weights`
   class and it is feed-forward-centred.

### 5.4 What another family would take

1. A builder beside `build_gemma4`: config from the family's GGUF keys, gguf-py's name map for its arch, and wrapping
   every block linear rather than the seven names.
2. A runtime that does not subclass the pinned one, or a small protocol written first (model id, weights card, the
   model and tokenizer, a linear by (layer, role), forward and capture) that both the pinned runtime and a new one satisfy.
3. A role map in the weights facade: MLP roles (gate/up/down against a two-matrix MLP), norm names, no layer scalar,
   the activation.
4. An unembed for the family (norm type, softcap or none, a logit scale).
5. The 31B constants replaced by values read from the model in the vault, the server and the scientist's tools.
6. The family's chat template and end-of-turn pieces.
7. A stand-in of the new arch for the CPU checks.
8. The lens is optional: `load_jpp=False` already runs without it (`backend/gguf_runtime.py:10-13`), at the cost of the
   sightings and the vault's UPDATE and INSERT.

The pinned files (`backend/__init__.py`, `theodolite_math.py`, `engine_runtime.py`, `label_gate.py`) cannot be edited
without `R4_common.pinned_differ()` naming them (ARCHITECTURE.md:1529-1531); the GGUF work was all done in new modules for
that reason (`backend/gguf_runtime.py:4-6`).

## 6. Small models as rehearsal objects

| object | shape | what it is good for | status |
|---|---|---|---|
| the stand-in (`scripts/atlas_stand_in.py:44-48`, `:102-142`) | 6 layers, d_model 64, FFN 128, about 600 pieces, 41 linears, random weights, a real `Gemma4ForCausalLM` | the plumbing: apply, read back, roll back, persist, reload, compare; runs on the CPU; serves the whole instrument with `--serve 8765`; writes to GGUF as F32, F16, BF16, Q8_0, Q4_0, Q4_1 | loads and passes every check listed above |
| the wide stand-in (`scripts/check_gguf.py:213-219`) | the same at d_model 256, FFN 512 | Q4_K (rows of 256) with llama.cpp's Q5_0 fallback | loads |
| the 4-bit stand-in (`scripts/check_dequant.py:21-23`) | the stand-in in the W4A16 pack-quantized layout, patched as the real checkpoint is | the checkpoint path's plumbing | loads in that check |
| `gemma-4-12B-it-QAT-Q4_0.gguf` in `~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/` | 6.98 GB; arch `gemma4`; 48 layers; d_model 3,840; FFN 15,360; 16 heads; 667 tensors (328 Q4_0, 338 F32, one Q6_K embedding) | a trained, dense Gemma 4 about 37% of the 31B's linear weights (10.9 billion by the header's shapes against 29.287 billion; my arithmetic) | its header states all ten `NEEDED_KEYS`, `shared_kv_layers` 0, `embedding_length_per_layer_input` 0, no `expert_count`, and holds `rope_freqs`, so it passes every refusal of `text_config_from_gguf` as I read them. It was NOT built or run; no document mentions it. |
| bf16 Gemma 4 12B at `~/bin/ai-models/txt/google/gemma-4-12B` | 23.9 GB safetensors, `Gemma4UnifiedForConditionalGeneration` | used by the study's R3 dry run (`experiments/scripts/R3_common.py:213`) | not loadable by the instrument's runtimes as written (another transformers class, no GGUF) |

Refused by the builder: Gemma 4 E2B and E4B (per-layer embeddings), 26B-A4B (experts), any model with shared key-value
layers, and every non-Gemma-4 GGUF on the machine (`backend/gguf_model.py:43-46`, `:214-221`).

What a 12B run would meet: `build_gemma4` and `GGUFRuntime.load()` are written for any layer count (`gguf_runtime.py:128-132`,
`:233`), and on the six-layer stand-in the accessors, pushes, neuron reads, row captures, a vault DELETE and a census
all work (`scripts/check_gguf.py:774-896`). But a whole sighting is not run there because "the pinned sweep counts 60
layers of 5,376 dims, and reads the J++ lens" (`scripts/check_gguf.py:822`), and the vault's ROLLBACK loops
`range(N_LAYERS)` = 60, which the checks work around by patching the constant (`scripts/check_gguf_fast.py:583-588`).
So on a 12B: forwards, weight reads, overlays, pushes and a census are plausible; sightings, the lens, UPDATE/INSERT and
an unpatched ROLLBACK are not. All of this is inference until someone runs it.

The stand-in is random: it rehearses the harness, not hypotheses about what weights compute.

## 7. Surprises

1. The GGUF fingerprint does not see the middle of the file (2.3 C). A patched copy can share its base's fingerprint.
2. `write_gemma4_gguf` cannot write a model that was loaded packed, and is declared not to be llama.cpp's converter.
3. An overlay on `up_proj` or an attention projection is heard by the forward but not seen by the readers:
   `slice_w_up` and `get_full_w_up` lay no overlays on (`backend/gguf_runtime.py:193-195`, `:222-224`), and
   `atlas_ops.dense_weight` reads the layer's own weight.
4. The QAT Q4_0 GGUF is a different quantisation from the W4A16 checkpoint in all 410 linears. "The Gemma model" is two
   objects depending on the path.
5. The LoRA export cannot be loaded back.
6. `compiled` silently serves the reference for a call its graphs were not tested for, for instance under
   `torch.inference_mode` (`backend/gguf_fast.py:221-246`). It is counted (`fell_back`) but the step is then 4 to 6 times slower.
7. `GGUFLinear.unpacked()` returns the resident weight itself with the note "do not write into it"
   (`backend/gguf_weights.py:487`). Nothing guards it; the W4A16 resident mode does catch an in-place write at the next
   forward (`scripts/check_dequant.py:20`).
8. A dense, trained 12B Gemma 4 GGUF sits on the machine and passes the builder's refusals by its header; nothing in
   the documents mentions trying it.
9. Document against environment: ARCHITECTURE.md:1351 speaks of transformers 5.21; the venv holds transformers 5.20.0
   and pyproject pins `>=5.20.0`.
10. The vault clamps layer and neuron indices to the 31B's constants instead of refusing
    (`backend/larql_vault.py:481`, `:500`). On the 31B an out-of-range request is silently moved to the edge; on a
    smaller model the clamp does not protect, so an index valid for the 31B but not for the model reaches the model.

## 8. Not established

- Whether `build_gemma4` builds the 12B GGUF and gives a correct forward; its speed and memory on the card.
- Whether gguf-py's quantisers reproduce an existing block from its own dequantised values (requantisation round trip).
- Whether a LoRA written by `export_lora.py` reloads to the overlay's exact bits.
- Whether logits are bit-equal across batch sizes on the GPU.
- The W4A16 checkpoint through int4mm on the 31B: speed and memory.
- Load time of the W4A16 checkpoint on the card (not in the files of this area).
- Whether a census or atlas record names the vault edits in force (grep finds none in `backend/census.py`; another reader's area).
- Whether `weight_blocks` can be rewritten in place as a quantised-domain edit; nothing in the repository does it or tests it.
- The cause of the 6-19% run-to-run movement of the reference step.
- A census, an atlas build or the server's pages on a GGUF with `compiled` on the card (ARCHITECTURE.md:1513-1515 lists these as not measured).
