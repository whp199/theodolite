# Build spec for 1a: the first slice of the weight-edit kernel

Written 14 November 2026 by the spec's author and revised the same day after two critics read it (one against
the code, one against the ruling and the pilots). Both were read-only agents: nothing in the repository was
edited but this file, no git command changed the tree, the model was not loaded and the card was not touched; the
critics' probes ran on the CPU, on the stand-ins and the tokenizer. Three implementers build from this file in
parallel, each seeing only it and the repository.

Marks on figures: **[measured, pilot N]** a pilot JSON in `pilot/` (14 November 2026, one run each);
**[read]** code opened today at the line given; **[ran]** a small run on the CPU today (a stand-in or the
tokenizer; never the card); **[inferred]** worked out, not measured; **[provisional]** a number 1a ships with and
the card check re-measures.

Where the binding documents disagree this spec follows the pilots, then the ruling (`verify/arbiter.md`), then
the proposal (`design/synthesis.md`). Section 10 lists each place that happened and each place this spec departs
from the builder's decisions; section 11 lists the critics' objections that were not taken, with the reason.

Layers count from 0 in code, in files and in this spec. Anything a person reads says both: "layer 20 (page 21)".


## 1. What 1a is and is not

**It is** a library and a command line, driven by the owner's own session, that takes one edit of the base
weights as a value, reads it against the unedited model in the same forward, and writes what it read into a
ledger. Six new files and no change to any existing one:

| file | what | owner |
|---|---|---|
| `backend/weight_edits.py` | the edit as a hashed value; its gate; save and load; control edits; per-row hooks; the folded context; the base assertion | implementer A |
| `backend/levelling.py` | items, the plan of a reading, the cost estimate, the paired reading, the written reading, the read-outs | implementer B |
| `backend/level_book.py` | the append-only, hash-chained ledger | implementer C |
| `scripts/levelling.py` | `estimate`, `run`, `book` | implementer C |
| `scripts/check_levelling.py` | CPU checks on the stand-ins | implementer C |
| `scripts/verify_levelling.py` | what only the card can show | implementer C |

**It is not** an agent's tool, a page, a server route, a claim, or a verdict. It prints numbers, ranks against
control edits, floors, and the two words "void" and "unstable". It never leaves the model edited: an edit is an
argument of a reading. It touches no pinned or live file: `backend/engine_runtime.py`, `theodolite_math.py`,
`__init__.py`, `label_gate.py`, the census code and the volunteer kit are read, never written, and
`atlas_ops.write_rows` is called with hooks registered round the call, its body untouched.

**Words.** The code, its messages and its output do not use: proof, held, seal, erratum, sighting, plate, station,
field book, chance edit. Say "control edit". "Levelling" and "level book" appear only in file names, one
environment variable and module docstrings (the name is still the owner's to choose, section 9).

**The measured facts it rests on.** All 14 November 2026, RTX 5090, compiled unpacker, lens freed, server stopped,
unless the row says otherwise.

| # | fact | recorded |
|---|---|---|
| 1 | Rows of one forward do not leak: an untouched row far from edited rows equals row 0 by bits; a hooked forward's row 0 equals an unhooked forward's of the same shape; the same edit in rows 1, 9, 20 and 31 of 32 reads the same | pilots 1 and 6; `pilot/RESULTS.md` "The mechanism is sound" |
| 2 | Silencing a neuron on the INPUT side of its `down_proj` (the row's activation set to nought) reads as the folded weight does to five decimals at the same batch size, 7 of 7 cases at 2 and at 32 rows, also where the reading had parted (0.72581 and 0.72581). A factor of 0 only: no pilot ran another input-side factor | pilot 6; `pilot6_which_is_true.json` |
| 3 | An OUTPUT-side delta for the same edit is another arithmetic (0.0227 where the other two read 0.7258) | pilots 4 and 6 |
| 4 | On raw text with no start mark one folded edit read 0.0091 at 2 rows and 0.7258 at 32: at 150 pieces there is one arithmetic up to 12 rows and one from 16. The same passages as chat-formatted answers showed no reading differing over fivefold (2 of 120 over twofold; read over every position, the question's included). At 91 pieces of chat-formatted text six batch sizes (2, 3, 4, 8, 16 and 32 rows) gave six different base rows | pilots 4, 6, 7; `pilot5_knife_edge.json` (`untouched_row_against_batch_1`) |
| 5 | Null rows (a layer's feed-forward output scaled by 1 + e) read 0.00014 to 0.0066 nats by text and layer: no constant floor serves | pilot 5 |
| 6 | A restart from layer L on a stream kept at the same batch shape gives the whole forward's bits (layers 20, 40, 44, 52, 58; unhooked) at the share of layers run | pilot 1, `P4_restart_from_layer` |
| 7 | A one-row capture replayed at 32 rows agreed with a same-shape forward at layer 44 (0.9998) and not at layers 5 and 20, on raw text with no start mark. Layers 21 to 43 and chat-formatted items were not read | pilot 3, part C |
| 8 | `atlas_ops.write_rows` under a hook registered round it (an OUTPUT-side forward hook; `stop_ids=[]`, so no row stopped): untouched rows wrote the same 48 pieces with the same log-probabilities; a step cost 0.095 s hooked and unhooked (8 rows) | pilot 3, part A |
| 9 | A whole forward costs 0.08 s + 0.33 ms a row-position (within 15% on nine shapes); fp32 full-vocabulary read-out 0.11 ms a scored row-position; the model's own dtype differs from fp32 by a mean KL of 0.0013 nats, too coarse | pilot 1; `verify/skeptic_numerics.md` §7 |
| 10 | A column overlay costs about 0.3 ms a cloned matrix (at most 64 overlay columns on one linear) | pilot 1, `P2_module_overlay_overhead` |
| 11 | Load 9.5 s, unpacker install 1.83 s, 17.97 GiB at rest, 8.37 GiB free | pilot 1 |
| 12 | The base's mean log-probability of the scored span: its own greedy answers -0.07 to -0.17 nats a piece; a person's passage given as the answer -2.46 and -5.51; the same passages raw, with no start mark, -9.15 and -10.09 (pilot 7's raw recipe -9.6) | pilot 8; `RESULTS.md` "Pilot 8" |
| 13 | On six chat-formatted items the hardest-firing sets of 1 to 1,000 columns read at 8 rows and at 25 rows differed by at most 1.60-fold (medians 1.01 to 1.10), over the answer's span | pilot 8 |
| 14 | Memory above the model is about 0.25 MiB a row-position in a teacher-forced forward | inferred by the numerics skeptic from the pilots' peaks; no per-shape peak exists |
| 15 | Writing with a cache costs about 0.8 MB a row a word-piece of context above the model (32 rows at 64 new pieces peaked 3.1 GiB above it); the model and the lens took 21.2 GiB before the lens is freed | measured 28 October 2026, lens loaded; `scripts/census_run.py:96-99` [read] |

What the pilots' record does not say, or says wrongly, and this spec builds on:

- **The pilots' raw text had no start mark**, though `RESULTS.md` ("start mark, no chat template") says it had.
  Every pilot built it with `tok(text, add_special_tokens=True)`, which adds nothing on this tokenizer; the JSON
  agrees (pilot 8's "recipe raw" is 90 pieces where "recipe as an answer" is 25 + 90; pilot 7's raw recipe 150
  where the chat form is 175) [read today, and run on the tokenizer alone by the code critic]. So the -9.6 nats,
  the eighty-fold swing and fact 7 are of text the runtime itself would not send (`tokenize_prompt` adds the
  mark, `engine_runtime.py:1225-1232`). Raw text behind a start mark is unmeasured.
- **A null factor must be a bf16 value other than 1 and not a power of two** (`RESULTS.md`, last paragraph):
  1 ± 0.001 is exactly 1 in bf16 (the step at 1 is 1/128 above and 1/256 below), so such a row is the base row
  again, and a power of two is an exact scaling the norm may undo exactly. Six of the 31 null rows of each of
  pilot 5's forwards and both of pilot 8's small-batch ones were the base row for the first reason [rebuilt on
  the CPU].
- `w4a16_fast.describe()` has no `fell_back` key [read, `backend/w4a16_fast.py:564-581`]; the counter is on
  the bound function, `engine_runtime.dequantize_w4a16_full.fell_back` [read, `:204-230`].
- The compiled unpacker serves only calls under the grad mode it was tested in; under `torch.inference_mode()`
  or autocast it falls back to the reference unpacker, about seven times slower [read, `:204-230`]. Every forward
  of 1a runs under `torch.no_grad()`, as the pilots ran.
- **On the float32 stand-in the rows of one written batch do not write the same log-probabilities above one
  thread**: with no hook at all, at 12 threads rows 1, 3, 5, 7, 9 and 11 of 12 differ from row 0 by up to 4.8e-7
  (the pieces are the same); at one thread all rows are alike; whole forwards are alike at any thread count, and
  the packed bf16 stand-in is alike in both [ran on the CPU, 14 November 2026, MKL 2024.2, this machine]. So
  every CPU run of 1a on a stand-in sets one thread (5.1, 6.1).
- **The stand-ins are random over about 600 pieces**: a typed answer reads about -6.4 nats a piece (ln 600) and
  the stand-in's own greedy answer about -5.9 [ran on the CPU], so the shipped item gate (-6.0) refuses nearly
  every stand-in item. A stand-in plan gives its own bar (6.1).


## 2. The data model

### 2.1 Shared rules (all three modules)

- **Canonical JSON.** `canonical(obj) = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
  allow_nan=False).encode("ascii")`. Every hash below is `hashlib.sha256(canonical(...)).hexdigest()` unless it
  says otherwise. `weight_edits` and `level_book` each define this one-liner (no import between them);
  `levelling` imports `level_book.canonical`.
- **JSON-clean values.** Whatever is hashed or written holds only `dict`, `list`, `str`, `int`, `float`, `bool`
  and `None`. No numpy scalar, no tuple, no NaN and no infinity: a non-finite number is written as `None`.
- **Time** is UTC to the second, `"2026-11-14T22:00:00Z"`.
- **Array hash.** `array_sha256(a, tag)` with `tag` one of `"i8"` (little-endian int64), `"f4"` (little-endian
  float32), `"u1"` (uint8):

      a = np.ascontiguousarray(a, dtype={"i8": "<i8", "f4": "<f4", "u1": "u1"}[tag])
      h = hashlib.sha256()
      h.update((tag + ":" + ",".join(str(int(s)) for s in a.shape) + ":").encode("ascii"))
      h.update(a.tobytes())
      return h.hexdigest()

  Worked values, for the checks (computed on 14 November 2026 with the definitions above; the code critic
  recomputed all seven of this section's worked hashes and they stand):
  `array_sha256([5, 17], "i8")` is `5b79bedb09ffd0204f01d88edb5a5b9dfe529e0087e7b95060532599efae6e97`;
  `array_sha256([0, 0], "u1")` is `501e17c6fd6079926903d7736534036cb5d6bfa666f66c75b607bc37a1c9fcdc`;
  `array_sha256([0.0, 0.0], "f4")` is `9b2769ca9a7eaa039d65fc6da4a9e3d4861b1d6eca3ee535f0bf1f74aacd031a`.
- **A power of two.** A factor g is "exact" when `g == 0 or math.frexp(abs(g))[0] == 0.5` (0, ±1, ±2, ±0.5, ...).

### 2.2 The edit

An **edit** is a list of **terms**. A term names one linear of the text model and holds an additive difference
from the shipped tensor. At **dose** d the weight is `W + d * Delta`: dose 1 is the edit, 0 the base, -1 its
mirror.

A linear is named by its dotted path inside the text model (`causal_check.language_model(hf)`):
`layers.<L>.mlp.down_proj`, `.gate_proj`, `.up_proj`. Its shape is `(out, in)`: read as
`(weight_packed.shape[0], 8 * weight_packed.shape[1])` when packed, else `weight.shape`. On the 31B `down_proj`
is (5376, 21504) and the other two (21504, 5376). The model's dtype is bf16 on the checkpoint and on the packed
stand-in, float32 on the plain stand-in (`lm.norm.weight.dtype`); hooks take it from the tensor they are
handed.

`Term` (`@dataclass(frozen=True, eq=False)`: compared by identity, 3.1; arrays are numpy, C-contiguous, set
read-only):

| field | type | meaning |
|---|---|---|
| `linear` | str | the dotted path |
| `kind` | str | `"cols"`, `"lowrank"`; `"rows"` is a legal value that the gate refuses in 1a |
| `layer`, `which` | int, str | parsed from `linear` (`which` is `down_proj`, `gate_proj` or `up_proj`) |
| `idx` | int64 `[K]` | cols: the columns (neurons), strictly ascending |
| `op` | uint8 `[K]` | cols: 0 scale, 1 set, 2 add, one a column |
| `factor` | float32 `[K]` | cols: the factor f of a scaled column (0 silences); 1.0 for a set or added column |
| `vectors` | float32 `[K, out]` or None | cols: the given vector of a set or added column, zeros in the rows of scaled columns; None when every op is scale |
| `A`, `B` | float32 `[r, in]`, `[out, r]` | lowrank: `Delta = B @ A` |

The difference a cols term stands for, column n of the base W: scale, `(f - 1) * W[:, n]`; set to v,
`v - W[:, n]`; add v, `v`. A scale term carries no array of Gemma's: it is rebuilt from the public base and the
manifest alone. One edit holds at most one cols term and one lowrank term a linear (several recipe terms on one
linear are merged at build; a column named twice is refused). Terms are kept sorted by `(linear, kind)`.

`Edit` (`@dataclass(frozen=True, eq=False)`): `id` (64 hex), `terms` (tuple of `Term`), `base` (dict, below),
`note` (str, not hashed), `report` (dict, not hashed). Methods: `layers() -> List[int]` (ascending, distinct),
`lowest_layer() -> int`, `has_lowrank() -> bool`, `has_vectors() -> bool` (any set or added column),
`linears() -> List[str]`, `short() -> str` (first 12 of `id`), `manifest() -> dict` (2.3).

**How an edit is pinned to its base.** `base = {"model": <rt.model_id>, "weights_id": <weights card's
"files_sha256", None on a stand-in>, "tensors": {<linear>: <pin>}}`, one pin for each linear the edit touches.
The pin is `tensor_pin(mod)`:

- a GGUF's layer (`gguf_weights.GGUFLinear`) and an int4mm linear are refused. The int4mm test,
  `getattr(mod, "int4mm", False)`, comes before the next bullet's: such a linear keeps `weight_packed` and
  `weight_scale` (`backend/int4_cpu.py:276-297` [read]);
- a packed linear (`weight_packed` and `weight_scale`): sha256 over the ASCII prefix
  `"w4a16:{p0}x{p1}:{s0}x{s1}:{scale dtype}:"` (the two shapes; the dtype as `str(s.dtype)`), then the packed
  words' bytes (int32, as stored, C order), then the scales' raw bytes (bf16 viewed as int16; float32 as it is);
- a plain linear: sha256 over `"dense:{out}x{in}:{dtype}:"` then the weight's raw bytes in its own dtype (bf16
  viewed as int16).

Pins are computed on the CPU from `.detach().cpu()` copies and cached by module for the life of the process
(a `down_proj`'s packed words are 58 MB); `load` reads the cache too, since nothing in the process writes packed
words.

**Identity.** `id = sha256(canonical(core))`, where

    core = {"format": "theodolite-edit", "version": 1, "base": base, "terms": [term_core, ...]}
    cols term_core    = {"linear": s, "kind": "cols", "count": K,
                         "columns": array_sha256(idx, "i8"), "ops": array_sha256(op, "u1"),
                         "factors": array_sha256(factor, "f4"),
                         "vectors": array_sha256(vectors, "f4") or None}
    lowrank term_core = {"linear": s, "kind": "lowrank", "rank": r,
                         "A": array_sha256(A, "f4"), "B": array_sha256(B, "f4")}

Worked value: base `{"model": "stand-in", "weights_id": None, "tensors": {"layers.3.mlp.down_proj": "ab" * 32}}`
and one term silencing columns 5 and 17 of `layers.3.mlp.down_proj` (idx `[5, 17]`, op `[0, 0]`, factor
`[0.0, 0.0]`, no vectors) gives id `f31e6cce0458eed20f50819b0086ae577834bc53ae4e0a063dc9885c25c744cb`.

**The report**, by arithmetic alone, one entry a term: `{"linear", "kind", "layer", "page_layer", "count" or
"rank", "ops": {"scale": k, "set": k, "add": k}, "rel_change": {"max": x, "median": y}, "carried": bool}`.
`rel_change` of a column is `||Delta[:, n]|| / ||W[:, n]||` in float32 (`|f - 1|` for a scale); a base column
of zero length takes the median length of the term's other base columns as its denominator, and 1.0 if all are
zero. For a lowrank term `rel_change` is `{"bound": ||B||_F * ||A||_F / ||W||_F}`, and its entry also holds
`"form"`: on a `down_proj`, `an output-side add; no folded form in 1a`; on `gate_proj` or `up_proj`, `an
output-side add on {which}: the added form of a change to what neurons read, not the recomputed form asked of a
row edit; not measured`. Base columns are read with
`gguf_weights.dense_cols(mod, torch.as_tensor(idx, dtype=torch.long))` (it takes a LongTensor and returns
`[K, out]` in the model's dtype on the model's device) and a whole weight with `gguf_weights.dense(mod)`.

**Caps** [provisional guards against slips, not findings]: `MAX_TERMS = 16`; `MAX_COLUMNS = 4096` a term;
`MAX_VECTOR_COLUMNS = 1024` set or added columns a term; `MAX_RANK = 64` (the cost was measured to rank 64);
`MAX_FACTOR = 64.0`; `MAX_REL_CHANGE = 64.0` for a column, for a lowrank term's bound, and for what a dose
carries (`dose_refusal`, 4.1); `MAX_PAYLOAD_BYTES = 256 * 2**20`; `MAX_DOSE = 4.0`.

**The gate: every refusal, in these words.** `build` raises `weight_edits.Refused` (a `ValueError`) with the
message; braces are filled in; `L1` is `L + 1`. The table's order is the order of precedence: an input that
breaks two rules gets the earlier row's words, however the tests are computed.

| when | message |
|---|---|
| kind `rows`, or a recipe term with `"rows"` | `Row edits on gate_proj and up_proj (what a neuron reads) are not built: their per-row form, the edited coordinate recomputed, has not been measured. They wait for 1b.` |
| a recipe term with `"remove_direction"` | `Direction removal is not built: it must go through the post-feed-forward norm's weight and the layer's quiet mask, and that form is 1b's.` |
| a linear under `self_attn` | `Edits to attention linears ({name}) are refused until their facts are in the gate and a card check has passed: q, k and v each pass a per-head norm, and on a full-attention layer the values are the keys. They wait for 1b.` |
| any other name that is not a feed-forward linear | `Only the feed-forward linears take edits in 1a: layers.<L>.mlp.down_proj for a column edit, and down_proj, gate_proj or up_proj for a low-rank term. "{name}" is not one. The embedding, the norms and the layer scalars are not built.` |
| layer out of range | `Layer {L} does not exist: layers count from 0 to {n - 1} here (page layers 1 to {n}).` |
| cols on `gate_proj` or `up_proj` | `A column edit changes what a neuron writes, so it names a down_proj; {name} is not one.` |
| column out of range | `Neuron {n} does not exist: layer {L} (page layer {L1}) has {N} (0 to {N - 1}).` |
| a column named twice | `Neuron {n} of layer {L} (page layer {L1}) is named twice in this edit: one change a column.` |
| a term with no column | `A term that names no neuron changes nothing.` |
| not finite | `An edit holds a number that is not finite ({where}).` |
| `abs(f) > MAX_FACTOR` | `A column is scaled by at most {MAX_FACTOR:g} either way here, not {f:g}.` |
| vector of the wrong length | `A column of layer {L}'s down_proj is {out} numbers long; the vector given for neuron {n} has {k}.` |
| a column's `rel_change > MAX_REL_CHANGE` | `Neuron {n} of layer {L} (page layer {L1}) would change by {x:.0f} times its own length; 1a takes at most {MAX_REL_CHANGE:g}.` |
| every column of a `down_proj` scaled by one positive factor | `Scaling every column of layer {L}'s down_proj by {f:g} changes nothing: the norm after the block undoes a uniform positive scale.` |
| too many terms, columns, vector columns | `An edit takes at most {cap} {what} in 1a, not {k}.` (`what`: `terms`, `columns a term`, `set or added columns a term`) |
| lowrank shapes | `A low-rank term on {name} needs A [rank, {in}] and B [{out}, rank]; it was given A {a} and B {b}.` |
| rank over the cap | `A low-rank term has rank at most {MAX_RANK} here, not {r}.` |
| a lowrank term's bound over `MAX_REL_CHANGE` | `A low-rank term on {name} could change it by up to {x:.0f} times its own length (the lengths of B and A over the weight's); 1a takes at most {MAX_REL_CHANGE:g}.` |
| arrays over the byte cap | `This edit carries {mb:.0f} MiB of arrays; 1a takes at most {cap_mb:.0f}.` |
| a linear that cannot be pinned | `An edit is pinned to the W4A16 checkpoint's packed tensors, or to a stand-in's plain ones. {name} is {kind}: another object, not taken in 1a.` (`kind` from `gguf_weights.packed_kind`; `w4a16 int4mm` for an int4mm linear, which `packed_kind` calls `w4a16`) |
| control edits: too few other sites | `Layer {L} (page layer {L1}) has {m} neurons outside this edit: too few for a control edit of {K}.` |
| load: model or weights differ | `This edit was made on {a} (weights {ia}); the loaded model is {b} (weights {ib}). An edit is a difference from one base and is not carried to another.` |
| load: a tensor's pin differs | `This edit was made against other weights: {name} then hashed to {a12}, the loaded model's hashes to {b12}.` |
| load: an array's hash differs | `The saved edit {id12}: its {array} no longer hashes to what its manifest says.` |
| load: the id differs | `The saved edit says it is {id12} and hashes to {other12}.` |
| load: not this format | `{path} is not a saved edit of format theodolite-edit, version 1.` |

### 2.3 A saved edit

Folder `edits_dir() / <id>`, where `edits_dir()` is `$THEODOLITE_EDITS_DIR`, else
`${XDG_DATA_HOME:-~/.local/share}/theodolite/edits` (beside the book, outside the checkout). Two files:

- `manifest.json` (written by `census.write_json_atomic`, indent 1): `{"format": "theodolite-edit", "version": 1,
  "id", "base", "terms": [...], "note", "report", "saved_at"}`. A cols term:
  `{"linear", "kind": "cols", "count", "columns": [ints], "ops": ["scale" | "set" | "add", ...],
  "factors": [floats], "vectors": "t<i>.vectors" or None, "sha256": {"columns", "ops", "factors", "vectors"},
  "carried": bool}`. A lowrank term: `{"linear", "kind": "lowrank", "rank", "A": "t<i>.A", "B": "t<i>.B",
  "sha256": {"A", "B"}, "carried": true}`. `i` is the term's place in the sorted order. `carried` is true when the
  term's arrays are in the payload, false when the manifest alone rebuilds it. `Edit.manifest()` returns this
  dict without `saved_at`.
- `payload.safetensors` (only when some term is carried; `safetensors.numpy.save_file`): the keys the manifest
  names, float32.

`load` rebuilds every array, compares each `sha256`, recomputes the id, then compares the base with the loaded
model (model id, weights id, the `tensor_pin` of each touched linear), and refuses on any difference with the
words above. Nothing under the checkout is written, and a payload is never committed: it can hold pieces of
Gemma's columns.

### 2.4 The edit as a recipe (what a plan file holds)

    "edit": {"terms": [<term>, ...], "note": "<optional>"}        or        "edit": {"saved": "<id or folder>"}

Each recipe term has `"layer": L` and exactly one of:

| key | value | builds |
|---|---|---|
| `"silence"` | `[n, ...]` | scale by 0 |
| `"scale"` | `[n, ...]` with `"factor": f` or `"factors": [f, ...]` (one a neuron) | scale |
| `"set"` | `[n, ...]` with `"vectors": "<file.npy>"` | set; the file is float32 `[K, out]`, row j for the j-th neuron as listed |
| `"add"` | `[n, ...]` with `"vectors": "<file.npy>"` | add |
| `"lowrank"` | `{"A": "<a.npy>", "B": "<b.npy>"}` with `"which"` (default `"down_proj"`) | lowrank |

Neurons are ints. A file path is taken relative to the plan file's folder. A term may give `"sha256": "<hex>"`
of its `.npy` file (for `lowrank`, `"sha256": {"A": ..., "B": ...}`), checked before the file is read:
`The file {path} hashes to {a12}, not the {b12} the plan names.` A file is read with
`np.load(path, allow_pickle=False)` and its dtype and shape are checked before anything else, as
`engine_runtime.read_study_mean` does: `The file {path} is {dtype} {shape}; this term takes float32 {want}.` Any
other key is refused: `An edit term has a key 1a does not know: {key}.`

### 2.5 The item

An item is a question and an answer text, in the model's own format, thinking off, as the instrument asks
everything (`engine_runtime.py:1193-1223`, `census.py:2048`).

    def chat_ids(tokenizer, question, system=None) -> List[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]

- `prompt_ids = chat_ids(tokenizer, question, system)`. It must begin with the tokenizer's start mark and hold it
  once (`The chat template gave {k} start marks; an item has exactly one, first.`). Both checks assert that it
  equals `rt.tokenize_prompt(rt.compose_prompt(question, chat=True, system=system))`.
- `answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]`, at least one piece
  (`An item needs an answer of at least one word-piece: its pieces are what is scored and what the item is gated
  on.`). The answer is tokenised apart and appended, as pilots 7 and 8 did.
- `ids = prompt_ids + answer_ids`; `T = len(ids)`; `T <= MAX_PIECES = 512`
  (`This item is {T} word-pieces; 1a reads at most 512.`).
- **Span positions** `P = [len(prompt_ids) - 1, ..., T - 2]`: position p predicts `ids[p + 1]`, an answer piece.
  `S = len(answer_ids)` positions. **End position** `T - 1` predicts what follows the text.
- **Identity** `id = sha256(canonical({"v": 1, "prompt_ids": [...], "answer_ids": [...]}))`. Worked value:
  prompt `[2, 105, 9]`, answer `[7, 8]` gives
  `74ccaa4e58278a91b6f11caefa1895e5fb8906f51d5ec3ee84f8f90b00330210`.

`Item` is a frozen dataclass: `name`, `question`, `answer`, `system` (str or None each), `prompt_ids`,
`answer_ids` (tuples of int), `id`; property `ids`. `item_from_ids(prompt_ids, answer_ids, name=None)` builds one
from ids alone and asks for no start mark (at least one piece each side, `MAX_PIECES` in all): the model's own
greedy answer behind a chat prompt; and, for the card check, raw text as the pilots built it (`prompt_ids` the
text's first piece alone, no start mark) and again behind the start mark (`prompt_ids` the start mark alone).

### 2.6 The plan of a reading

A reading's plan is a dict. `levelling.normalise` fills the defaults, refuses what it must, and returns the dict
that is run, recorded and hashed (`plan_sha256 = sha256(canonical(plan))`).

A **paired** reading:

| key | default | rule |
|---|---|---|
| `name` | required | a string, unique in its plan file |
| `kind` | required | `"paired"` |
| `edit` | required | a recipe (2.4) |
| `items` | `"all"` | `"all"` or a list of item names of the plan file; 1 to 256 |
| `doses` | `[1.0]` | 1 to 8 distinct finite numbers, each `abs(d) <= 4`. A dose of 0 may be listed: it is the zero row, and must be the base row by bits (4.5) |
| `main_dose` | 1.0 if listed, else the first dose that is not 0 | one of `doses`, not 0: the dose the control edits take, the folded twin is taken at, and the ranks are read at |
| `controls` | `{"count": 19, "seed": 0}` | count 0 to 199; `family` is always `"other_sites"` in 1a |
| `null_e` | `[0.03125, -0.03125, 0.125, -0.125]` | 2 to 8 numbers, each `-0.5 <= e <= 2`, at least one above 0 and one below; their factors as applied, `we.null_factor(e, dtype)` (4.1), all different, none 1 and none a power of two |
| `rows` | `{"small": 12, "large": 32}` | small 5 to 12 (the least that holds a base row, its twin, two null rows and one arm), large 16 to 64 |
| `readouts` | `odds` alone when the plan names pieces; `kl` and `own` when it names none | `{"kl": bool, "own": bool, "odds": None or {"toward": [...], "away": [...], "at": "end" or "span"}}` (`at` defaults to `"end"`); each list 1 to 16 piece ids, or strings that are one word-piece each as given (a leading space counts); one piece each side is a logit difference. `kl` and `own` default to true when `odds` is None, to false when it is given; at least one read-out on |
| `rank_by` | `"kl"` if on, else `"own"`, else `"odds"` | a read-out that is on |
| `folded` | `"auto"` | `"auto"`, `"always"`, `"never"` |
| `lower` | `"same_shape"` | how the layers below the edit are run: `"same_shape"`, `"whole"`, `"one_row"` (4.4) |
| `bars` | `{"item_gate": -6.0, "stable_ratio": 3.0, "one_row_min_layer": 44}` | the module's constants, read when `normalise` is called; a plan may give any of the three, and is then recorded as doing so (`"bars_overridden": true`) |

A **written** reading: `name`, `kind: "written"`, `edit`, `items`, `doses` (default `[1.0]`), `main_dose`,
`controls` (default `{"count": 3, "seed": 0}`), `null_e` (default `[0.125, -0.125]`, the rules above), `rows` (an
int, default 8, 5 to 32), `max_new` (default 48, 1 to 256). No other key.

**Rows of a forward, fixed for a class.** A paired reading is taken twice on every item, once in each **batch
class**: all its forwards of the small class have exactly `rows.small` rows and all of the large class exactly
`rows.large`. With `n0 = len(null_e)`, `nd = len(doses)`, `C = controls.count` and R the class's rows:

- row 0 is the **base row**, untouched; row R - 1 is its **twin**, untouched;
- rows 1 to n0 are the **null rows**, one for each e, in order, the same in every forward;
- the **arms** are the edit at each dose, in order, then control edits 0 to C - 1; forward f carries arms
  `f * s` to `(f + 1) * s - 1` in rows `n0 + 1` on, where `s = R - 2 - n0` is the arm slots a forward;
- rows left over in the last forward are untouched;
- `F = ceil((nd + C) / s)` forwards. All doses must sit in forward 0: `nd <= s`.

So every forward carries its own base row, its twin and every null row; only the arms differ. A written reading
is one batch laid out the same way, and every arm must fit it: `nd + C <= s`. `levelling.layout` computes both.

Refusals of `normalise` (raising `levelling.Refused`, a `ValueError`); `m` is `bars.one_row_min_layer`, `L1` is
`L + 1`:

| when | message |
|---|---|
| a key it does not know | `The plan has a key 1a does not know: {key}.` |
| `rows` out of range | `The small batch is 5 to 12 rows and the large one 16 to 64 (the ruling's two classes: at 150 pieces of raw text this card ran one arithmetic up to 12 rows and another from 16, 14 November 2026); {small} and {large} were asked.` (a written reading: `A written reading is one batch of 5 to 32 rows; {rows} were asked.`) |
| `nd > s` in a class | `A batch of {R} rows holds the base row, its twin, {n0} null rows and {s} more: too few for {nd} doses. Ask fewer doses or null rows, or more rows.` |
| a written reading whose arms do not fit | `A written reading is one batch: {R} rows hold the base row, its twin, {n0} null rows and {s} more, too few for {nd} doses and {C} control edits. Ask fewer, or more rows.` |
| a dose | `A dose is a finite number between -4 and 4: it multiplies the edit's difference from the base (1 the edit, 0 the base, -1 its mirror).` |
| no dose but 0 | `A reading needs a dose other than 0: at 0 the edit is the base.` |
| fewer than two e, or all on one side | `A reading takes at least two null rows, one with e above 0 and one below: one null row is one draw of the rounding.` |
| an e out of range | `A null row scales the layer's feed-forward output by 1 + e, with e between -0.5 and 2 and not 0; {e:g} was asked.` |
| a null factor of 1 | `In {dtype} 1 + e is exactly 1 for e = {e:g}: that null row would be the base row again. In bf16 the nearest values are 1 + 1/128 and 1 - 1/256.` |
| a null factor that is a power of two | `For e = {e:g} the null row's factor is {s:g}, a power of two: an exact scaling, which the norm may undo exactly, so it measures no rounding. Factors that serve: 1 + 1/128, 1 - 1/256, 0.75, 1.5, 3.` |
| two e with one factor | `e = {a:g} and e = {b:g} are one factor in {dtype} ({s:g}): the two null rows would be the same row.` |
| no read-out on | `A reading reads something: turn on kl, own or odds.` |
| `rank_by` off | `rank_by names a read-out that is on; {x} is not.` |
| a named piece that is not one piece | `{word!r} is {k} word-pieces ({ids}); a named piece is one. Give its id, or a word that is one piece.` |
| `lower: "one_row"` below `m` | `A one-row capture replayed at many rows has been shown only at layer 44 (page layer 45), on raw text, on 14 November 2026; it is taken from layer {m} (page layer {m1}) up. This edit starts at layer {L} (page layer {L1}).` |
| `folded: "always"` with a low-rank term | `A low-rank term has no folded form yet, so this edit cannot be read folded.` |
| a `kind` that is neither; an unknown item name or a reading name used twice (these two by `load_plan`) | plain words naming it |

`dtype` is the model's (`normalise` takes it; bf16 when none is given, as on the checkpoint). Six refusals need
the items or the loaded model and are made by `read` itself, as results (the first two by `estimate` too, which
raises them):

- `{R} rows of {T} word-pieces is {n} row-positions; 1a runs at most 12,288 in a forward (the pilots measured
  to 11,264).` (any item, in the large class);
- `Writing {R} rows to {n} word-pieces of context keeps about {g:.1f} GiB above the model (0.8 MB a row a piece,
  measured 28 October 2026); 1a takes at most {cap:g}. Ask fewer rows or fewer new pieces.` (a written reading;
  4.11);
- `we.dose_refusal(edit, d, dtype)` for each dose (4.1): a dose the rounding eats, or one that carries the edit
  past the size cap;
- under `folded: "always"`, `we.can_fold(lm, edit)` when it gives a sentence (4.6);
- `A vault edit is in force on {name} ({k} overlays): a reading is taken on the base model only. ROLLBACK the
  vault, or start a fresh process.`;
- `The model carries hooks that were not there at load (on {name}): a reading is taken on the base model only.`

### 2.7 The result of a reading

`levelling.read` returns one JSON-clean dict and appends the same dict to the book. Units: every KL and every
log-probability or log-odds is in nats; `seconds` are wall seconds; layers count from 0.

Top level, both kinds:

| key | value |
|---|---|
| `v` | 1 |
| `kind` | `"paired"` or `"written"` (what the plan gave, when it is neither) |
| `name` | the plan's |
| `status` | `"ok"`, `"void"` or `"refused"` |
| `why` | None, or plain words for a void or refused reading |
| `void` | None, or `{"code", "item", "class", "forward"}`; codes: `non_finite_base`, `untouched_rows_differ`, `repeat_differs`, `zero_dose_differs`, `out_of_memory`, `not_base_after`, `interrupted`, `error` (4.5) |
| `base_after` | bool: `we.assert_base` passed when the reading ended, however it ended |
| `edit` | None when the edit was refused, else `{"id", "terms": <edit.report>, "note", "exact": {<repr(float(dose))>: <we.exact_at(edit, dose, dtype)>}, "saved": <folder or None>}` |
| `base` | `{"model", "weights_id"}` |
| `runtime` | `census.runtime_record(device)` plus `"unpacker": {"mode", "fell_back_before", "fell_back_after"}`, `"hooks_at_load": <how many>`, `"dtype": str`, `"threads": torch.get_num_threads()`, `"code": {"weight_edits", "levelling", "level_book": <census.file_sha256 of each module's file>}` (the record's git flag does not see untracked files) |
| `plan`, `plan_sha256` | the normalised plan and its hash (the plan as given when it was refused before it could be normalised) |
| `bars_overridden` | bool |
| `rows` | a paired reading: `{"small", "large", "says": <the sentence of 4.9>}`; else None |
| `controls` | `{"family": "other_sites", "matched": false, "seed", "count", "ids": [edit ids], "says": <the sentence of 4.9>}` |
| `items` | a list, one entry an item that was begun |
| `summary` | 4.9 (None for a written reading) |
| `seconds` | `{"predicted", "actual", "ratio", "parts": {...}}` (`ratio = actual / predicted`) |
| `peak_gib` | the largest of the classes' peaks (a written reading: its own), None on the CPU |

The `item` block, both kinds: `{"id", "name", "question", "answer", "system", "prompt_ids", "answer_ids",
"pieces": T, "prompt_pieces", "scored": S}` (the three strings None for an item built from ids), so that an
entry says what was read without its plan file.

An entry of `items` in a **paired** reading:

| key | value |
|---|---|
| `item` | the item block |
| `status` | `"ok"`, `"refused"` (the item gate) or `"unstable"` |
| `why` | None or plain words |
| `gate` | `{"bar", "small": {"mean", "median", "worst"}, "large": {...}, "passed": bool}`: the base row's log-probability of the answer's own pieces, in both classes (4.10) |
| `classes` | `{"small": <class>, "large": <class>}`; a class not read is None |
| `stability` | None for an item the gate refused, else `{"ratio_bar", "base_rows_same_bits": bool, "metrics": {<metric>: {"arms": {<key>: {"small", "large", "ratio", "agree"}}, "disagree": [<key>, ...]}}, "disagree": [[<metric>, <key>], ...]}` |

A `<class>`:

| key | value |
|---|---|
| `rows`, `forwards` | R and F |
| `lower` | `"same_shape"`, `"whole"` or `"one_row"` |
| `restart_from` | the layer forwards restart from, None when every forward is whole |
| `untouched_rows_identical`, `repeats_identical` | bools (false voids the reading); `repeats_identical` is None with one forward |
| `base` | `{"own": {"mean", "median", "worst"}, "odds": {"mean", "median"} or None}`: the base row of forward 0, raw values |
| `arms` | a list of `{"key", "what": "null" or "edit" or "control", "e" or "dose", "factor" (a null row: as applied), "edit_id", "forward", "row", "non_finite": bool, "same_as_base": bool, "metrics": {<metric>: <stats>}}` |
| `floor` | `{<metric>: {"mean_abs", "median_abs", "above_1", "from": <key>, "of": n0}}` |
| `rank` | `{"metric", "dose", "value", "at_least", "non_finite", "of", "rank", "over_floor", "controls_over_floor"}`; None when there are no controls |
| `folded` | `{"taken": bool, "why", "dose", "base_from": "forward 0 of the same shape", "restart": bool, "rows_identical", "metrics": {<metric>: <stats>}, "per_row": {<metric>: <stats>}, "gap": {<metric>: {"abs", "rel"}}, "between_kl_mean", "same_bits", "seconds"}`; when not taken only `taken` and `why` are filled |
| `peak_gib` | the card's peak allocation while this class of this item was read (the counter reset at its start), None on the CPU |

Arm keys: `"null:<e>"` and `"edit:<dose>"` with the number as asked, written by `repr(float(x))` (`"null:-0.125"`,
`"edit:1.0"`), and `"control:<c>"` with c the integer. Metrics:
`"kl"` (KL(base || row), not below 0), `"own"` (the row's log-probability of the text's own next piece minus the
base row's), `"odds"` (the row's log-odds of the toward pieces against the away pieces minus the base row's).
`<stats>` over a metric's positions: `{"mean", "median", "max_abs", "above_1", "n"}`, with `median` torch's (the
lower middle of an even count, as the pilots took it), `max_abs` the largest absolute value and `above_1` the
count of positions whose absolute value is over 1 nat. The metrics of a row with a non-finite number are None.

An entry of `items` in a **written** reading: `{"item", "rows", "max_new", "stop_ids", "untouched_rows_identical",
"seconds_a_step", "base": {"pieces": [ids], "text", "logprobs": [...], "ended": bool}, "arms": [{"key", "what",
"e" or "dose", "factor", "edit_id", "row", "pieces", "text", "ended", "same_as_base": bool, "parts_at": int or
None, "logprob_gap": {"max_abs", "mean_abs", "n"}}]}`.

### 2.8 The book entry

One line of JSON a call, in `$THEODOLITE_LEVEL_BOOK`, else
`${XDG_DATA_HOME:-~/.local/share}/theodolite/level_book.jsonl`:

    {"v": 1, "seq": <0, 1, 2, ...>, "at": "<UTC>", "kind": "paired" | "written" | "note",
     "prev": <the "hash" of the line before; 64 zeros for line 0>, "body": {...}, "hash": "<64 hex>"}

`hash = sha256(canonical({"v", "seq", "at", "kind", "prev", "body"}))`: the entry without its own `hash`. The line
written is `canonical(entry with hash) + b"\n"`. Worked values: the entry `{"v": 1, "seq": 0,
"at": "2026-11-14T22:00:00Z", "kind": "note", "prev": "0" * 64, "body": {"says": "first line"}}` hashes to
`81318d96758c7187ba3ed1ad33e101f36e8d983aaae04190ebb680ae6d07317a`; the next, `{"v": 1, "seq": 1,
"at": "2026-11-14T22:00:01Z", "kind": "note", "prev": <that hash>, "body": {"x": 0.1, "n": None}}`, to
`75969610dd2dc42dd9d49f7ca703b72dbaf010f705085e4e158fe82e9a1ba5c1`.

The body of a reading's entry is its result (2.7), whole, per-item rows and the items' text included. A refused
or void reading is entered like any other; one refused because its `kind` is neither is entered as a `"note"`.


## 3. The public interface

Private helpers are the owning implementer's to shape. Everything named here is called across a seam and must
have exactly this signature.

### 3.1 `backend/weight_edits.py` (implementer A)

Imports torch, numpy, `backend.gguf_weights` (`dense`, `dense_cols`, `packed_kind`), `backend.census`
(`write_json_atomic`). It imports neither `levelling` nor `level_book`, and takes the text model `lm`, never a
runtime. The model's dtype, wherever one is asked, is `lm.norm.weight.dtype`.

    class Refused(ValueError)          # the gate, in plain words
    class NotBase(RuntimeError)        # the model is not the base: a hook or an overlay is in force
    class Term, class Edit             # 2.2: @dataclass(frozen=True, eq=False)

    # constants: FORMAT, VERSION, OPS, MAX_TERMS, MAX_COLUMNS, MAX_VECTOR_COLUMNS, MAX_RANK, MAX_FACTOR,
    #            MAX_REL_CHANGE, MAX_PAYLOAD_BYTES, MAX_DOSE, EDITS_ENV = "THEODOLITE_EDITS_DIR"

    def canonical(obj) -> bytes
    def array_sha256(a, tag: str) -> str
    def linear_name(layer: int, which: str = "down_proj") -> str            # "layers.{layer}.mlp.{which}"
    def linear_shape(mod) -> Tuple[int, int]                                # (out, in)
    def tensor_pin(mod) -> str                                              # raises Refused when it cannot be pinned

    # recipe terms, before they meet a model (TermSpec: a frozen dataclass, eq=False, of the builder's arguments)
    def silence(layer: int, neurons: Sequence[int]) -> TermSpec
    def scale(layer: int, neurons: Sequence[int], factor: Union[float, Sequence[float]]) -> TermSpec
    def set_columns(layer: int, neurons: Sequence[int], vectors) -> TermSpec        # vectors [K, out], row j for neurons[j]
    def add_to_columns(layer: int, neurons: Sequence[int], vectors) -> TermSpec
    def lowrank(layer: int, A, B, which: str = "down_proj") -> TermSpec
    def rows(layer: int, which: str, neurons: Sequence[int], vectors) -> TermSpec   # representable; build refuses it
    def from_recipe(recipe: dict, base_dir: Path) -> Union[List[TermSpec], str]
        # {"terms": [...]} -> the TermSpecs, files read and their sha256 checked; {"saved": x} -> the folder or id x
    def recipe_facts(recipe: dict, base_dir: Path) -> dict
        # no model: {"layers": [ints], "lowest_layer": int, "linears": int, "has_vectors": bool,
        #            "has_lowrank": bool, "factors": [the distinct scale factors]}; for {"saved": x} read
        #            from its manifest alone
    def edit_facts(edit: Edit) -> dict                                      # the same keys, from a built edit
    def twin_wanted(facts: dict, dose: float) -> bool                       # 4.7; no model, no dtype

    def build(lm, specs: Sequence[TermSpec], *, model_id: str, weights_id: Optional[str], note: str = "") -> Edit
    def save(edit: Edit, folder: Optional[Path] = None) -> Path             # default edits_dir() / edit.id
    def load(where: Union[str, Path], lm, *, model_id: str, weights_id: Optional[str]) -> Edit
        # `where`: a folder, or an id (or its first 12 or more hex) looked up under edits_dir()
    def edits_dir() -> Path
    def exact_at(edit: Edit, dose: float, dtype) -> dict
        # {"input_side": "exact" | "rounds" | "gone" | "none", "output_add": bool, "folded_form": bool,
        #  "factors": [[asked, applied], ...]}                              (4.1)
    def dose_refusal(edit: Edit, dose: float, dtype) -> Optional[str]       # None, or the sentence (4.1)
    def null_factor(e: float, dtype) -> float       # float(torch.tensor(1.0 + e, dtype=dtype)): the factor as applied
    def can_fold(lm, edit: Edit) -> Optional[str]   # None when `folded` would run, else the sentence it would raise (4.6)
    def control_edits(edit: Edit, lm, seed: int, count: int) -> List[Edit]  # 4.9; raises Refused

    @dataclass(frozen=True, eq=False) class Arm:  edit: Edit; dose: float = 1.0
    @dataclass(frozen=True)           class Null: layer: int; e: float

    @contextmanager
    def per_row(lm, programs: Sequence[Union[Arm, Null, None]]) -> Iterator[None]      # 4.1 to 4.3
    @contextmanager
    def folded(lm, edit: Edit, dose: float = 1.0, *,
               hooks_at_load: Optional[Dict[str, List[int]]] = None) -> Iterator[None] # 4.6; raises Refused, NotBase

    def hook_census(lm) -> Dict[str, List[int]]   # module path -> the ids of its forward and forward pre-hooks (4.6)
    def overlays_in_force(lm) -> List[Tuple[str, int]]     # (module path, entries) for every non-empty overlay dict
    def assert_base(lm, hooks_at_load: Optional[Dict[str, List[int]]] = None) -> None  # 4.6; raises NotBase

`Term`, `Edit`, `TermSpec` and `Arm` hold arrays, so they are compared and hashed by identity (`eq=False`; with
the dataclass default `a == b` raises on the arrays and `hash` on an ndarray): two edits are the same edit when
their `id`s are equal.

`build`, `load`, `control_edits`, `dose_refusal`, `can_fold`, `exact_at`, `per_row`, `folded` and `assert_base`
are the calls `levelling` makes. `build` and `load` raise `Refused`; `per_row` raises `Refused` for a program it
cannot run (an `Arm` whose edit names a linear this model lacks or of another shape; a `Null` whose factor is 1
or a power of two in the model's dtype) and `RuntimeError` from inside a hook when a forward's row count is not
`len(programs)`. `per_row` does not judge an `Arm`'s dose: `dose_refusal` is the caller's to ask.

### 3.2 `backend/level_book.py` (implementer C)

Standard library only (`json`, `hashlib`, `os`, `fcntl`, `datetime`, `pathlib`).

    VERSION = 1
    BOOK_ENV = "THEODOLITE_LEVEL_BOOK"
    GENESIS = "0" * 64
    KINDS = ("paired", "written", "note")
    class BookBroken(RuntimeError)

    def book_path() -> Path                         # read from the environment at every call
    def canonical(obj) -> bytes
    def entry_hash(entry: dict) -> str              # of the entry without its "hash" key (a "hash" key given is ignored)
    def head(path: Optional[Path] = None) -> Tuple[int, str]
        # (entries so far, the last entry's hash; (0, GENESIS) when there is no file). Reads the last line only.
        # Raises BookBroken when the last line is not a whole entry whose hash is its own.
    def append(kind: str, body: dict, *, path: Optional[Path] = None) -> dict
        # Writes one line and returns the entry written. Raises BookBroken (as head does), ValueError for a body
        # that is not JSON-clean or a kind that is not one of KINDS.
    def entries(path: Optional[Path] = None, *, start: int = 0, limit: Optional[int] = None) -> Iterator[dict]
    def verify(path: Optional[Path] = None) -> dict
        # {"ok": bool, "entries": int, "head": str, "first_bad": Optional[int], "why": Optional[str]}

`append`, step by step: make the parent folder; open the file for append (`os.O_APPEND | os.O_CREAT`, mode
0o600); take `fcntl.flock(fd, LOCK_EX)`; read the last line as `head` does (seq and hash); build the entry with
`seq = entries so far`, `at` now, `prev` that hash; write `canonical(entry) + b"\n"` in one `os.write` and check
the count it returns (a short write raises `BookBroken("The book took {k} of {n} bytes of line {seq}: its last
line is torn. Look at {path} before anything more is read.")`); `os.fsync`; unlock; close. It never rewrites,
truncates or moves the file. A last line that is torn (no newline, not JSON, or a hash that is not its own) is
not written over: `BookBroken("The book's last line (line {n}) is not a whole entry: {why}. Nothing was written.
Look at {path} before anything more is read.")`.

`verify` walks every line: each parses, has exactly the seven keys, `seq` counts from 0, `prev` is the hash of
the line before, `hash` is its own. It reports the first line that fails and why, and never raises for a bad book.

### 3.3 `backend/levelling.py` (implementer B)

Imports `backend.weight_edits as we`, `backend.level_book as lb`, `backend.atlas_ops as ao`
(`write_rows`, `forward_from_layer`, `ffn_width`, `hidden_size`, `softcap_of`), `backend.causal_check`
(`language_model`, `model_device`, `logits_fp32`), `backend.theodolite_math.FINAL_LOGIT_SOFTCAP`,
`backend.census` (`runtime_record`, `file_sha256`), `backend.runtime_factory.weights_card`,
`backend.cpu_memory.is_oom`, `backend.w4a16_fast.describe` and, to read the fall-back counter only,
`backend.engine_runtime`.

    class Refused(ValueError)

    VERSION = 1
    ITEM_GATE_BAR = -6.0        # nats a piece [provisional]
    STABLE_RATIO = 3.0          # [provisional]
    ONE_ROW_MIN_LAYER = 44      # [provisional: the one layer where the one-row capture was shown]
    ROWS_SMALL, ROWS_LARGE = 12, 32          # [provisional]
    SMALL_MIN, SMALL_MAX, LARGE_MIN, LARGE_MAX = 5, 12, 16, 64
    WRITTEN_ROWS_MIN, WRITTEN_ROWS_MAX = 5, 32
    NULL_E = (0.03125, -0.03125, 0.125, -0.125)
    NULL_E_WRITTEN = (0.125, -0.125)
    MAX_PIECES, MAX_ROW_POSITIONS, MAX_CONTROLS, MAX_DOSES, MAX_ITEMS = 512, 12288, 199, 8, 256
    MAX_WRITTEN_GIB = 6.0       # [provisional: 8.37 GiB was free after the load, fact 11]
    READ_CHUNK = 128            # positions of full-vocabulary fp32 logits worked at once (4.8)
    COST = {"forward_fixed_s": 0.08, "row_position_s": 0.00033, "readout_row_position_s": 0.00011,
            "decode_step_s": 0.095, "overlay_matrix_s": 0.0003, "load_s": 9.5, "unpacker_s": 1.83,
            "mib_row_position": 0.25, "mb_row_piece_written": 0.8}
            # measured 14 November 2026; mib_row_position inferred; mb_row_piece_written measured 28 October 2026
    CONTROLS_SAY, ROWS_SAY, PAIRED_SAY       # the three sentences of 4.9, as format strings

    def chat_ids(tokenizer, question: str, system: Optional[str] = None) -> List[int]
    def make_item(tokenizer, question: str, answer: str, *, system: Optional[str] = None,
                  name: Optional[str] = None) -> Item                       # raises Refused
    def item_from_ids(prompt_ids: Sequence[int], answer_ids: Sequence[int], *, name: Optional[str] = None) -> Item
        # asks only for at least one piece each side and MAX_PIECES in all; no start mark is asked for

    def load_plan(path: Path) -> dict                 # the plan file, its top level checked; raises Refused
    def plan_items(plan: dict, tokenizer) -> Dict[str, Item]                # in the plan file's order
    def normalise(reading: dict, *, n_layers: int, tokenizer=None, facts: Optional[dict] = None,
                  dtype=torch.bfloat16) -> dict
        # pure: defaults filled, 2.6's refusals raised. `tokenizer` resolves named pieces to ids (a plan that names
        # pieces as strings needs it); `facts` (we.recipe_facts) lets it check lower="one_row" and folded="always";
        # `dtype` is the model's, for the null factors. The module's bars are read when it is called.
    def layout(doses: Sequence[float], n_controls: int, null_e: Sequence[float], rows: int, *,
               one_forward: bool = False) -> List[List[Optional[str]]]
        # the arm key (2.7) of every row of every forward of one class, None for an untouched row;
        # `one_forward` for a written reading (every arm in one batch); raises Refused in 2.6's words
    def estimate(plan: dict, pieces: Sequence[Tuple[int, int]], *, n_layers: int, facts: dict,
                 twin: Optional[bool] = None) -> dict
        # `plan` normalised; `pieces` one (T, prompt pieces) an item; no model. `twin`: whether the folded twin is
        # taken; None decides it from plan["folded"] and we.twin_wanted(facts, main_dose). Returns
        # {"seconds": float, "parts": {"forwards", "readout", "folded", "writing"}, "forwards": int,
        #  "row_positions": int, "gib_above_model": float}; raises Refused for 2.6's two size refusals   (4.11)

    @dataclass class Handle: rt, hf, lm, tokenizer, W_U, device, dtype, softcap, n_layers, d_model, d_ff,
                             model_id, weights, weights_id, hooks_at_load, runtime
    def handle(rt) -> Handle
        # hf = rt.hf_model; lm = language_model(hf); tokenizer = rt.tokenizer; W_U = rt.lm_head_weight;
        # device = model_device(lm); dtype = lm.norm.weight.dtype; softcap = ao.softcap_of(hf);
        # n_layers = len(lm.layers); d_model = ao.hidden_size(lm); d_ff = ao.ffn_width(lm, 0);
        # model_id = rt.model_id; weights = weights_card(rt); weights_id = weights.get("files_sha256");
        # hooks_at_load = we.hook_census(lm); runtime = runtime_record(device).
        # Raises we.NotBase when an overlay is in force.

    def read(h: Handle, reading: dict, items: Mapping[str, Item], *, base_dir: Path = Path("."),
             edit: Optional[we.Edit] = None, save_edit: bool = False) -> dict
        # One reading, paired or written. `reading` is a plan as given (read normalises it); `base_dir` is where
        # a recipe's files are looked for. `edit`: an Edit already built in Python, read in place of the plan's
        # recipe; the recorded plan then holds "edit": {"built": "<id>"}. `save_edit`: save the edit read
        # (we.save) and name its folder in the result's edit block ("saved"). Returns the result (2.7) with a key
        # "book": {"seq", "hash"} added after the entry is written. A refusal or a void is a result, entered in
        # the book, not an exception. It raises lb.BookBroken before the model is touched when the book cannot
        # take an entry. Any other exception inside it, a KeyboardInterrupt included, is entered as a void
        # ("error" or "interrupted") and then raised again (4.5).
    def lines(result: dict) -> List[str]              # what a person reads of a result (5.3)

    def agree(a: Optional[float], b: Optional[float], floor: float, signed: bool,
              ratio: float = STABLE_RATIO) -> Tuple[Optional[bool], Optional[float]]            # 4.9
    def rank_among(value: Optional[float], controls: Sequence[Optional[float]]) -> dict
        # {"at_least", "non_finite", "of", "rank"}                                              # 4.9
    def stats(values: torch.Tensor) -> dict                                     # 2.7's <stats>


## 4. The algorithms

Everything runs under `torch.no_grad()`. A whole forward is exactly
`lm(input_ids=x, use_cache=False).last_hidden_state` with `x` the item's ids repeated to `[R, T]`: no attention
mask and no position ids, as every pilot ran it. A restart is
`ao.forward_from_layer(lm, L0, kept, None, pos)` with `pos = torch.arange(T, device=dev).unsqueeze(0).expand(R, T)`.
Both return final-normed states `[R, T, D]` in the model's dtype.

### 4.1 The per-row forms

`per_row(lm, programs)` installs, for the length of the block, at most one forward pre-hook and one forward hook
on each linear that some program touches, and removes every one in `finally`. `R = len(programs)`. Row r runs
program `programs[r]`: `None` (untouched), `Arm(edit, dose)` or `Null(layer, e)`. A row has one program. The
hooks act at every position, the start mark included (it is a weight edit, not a push), and never index
positions, so a decode step with a cache (sequence length 1) runs the same code as a whole forward.

Both hooks first check the tensor they are handed is `[R, T, .]` and raise
`RuntimeError(f"hooks made for {R} rows met {n}")` otherwise.

**Column scale and silence: a factor on the row's activation at `down_proj`'s input.** This is the form measured
for a factor of 0 (fact 2) and the arithmetic of the `mul` push (`atlas_ops.py:369-381` [read]), vectorised. Any
other factor is the same arithmetic and new on the card: on the packed stand-in factors of 0, 2 and -0.5 equal
the folded overlay by bits at 12 and at 32 rows, and 1.5 and -3 do not [ran on the CPU, 14 November 2026]; the
card check shows it (6.2 §5a). For each touched `down_proj`, `per_row` builds once a factor matrix `Fm` `[R, N]`
in the model's dtype, 1 everywhere but, for a row running `Arm(edit, d)` and a column n of the edit:

- scaled by f: `g = 1 + d * (f - 1)`;
- set: `g = 1 - d` (the old column's share that stays);
- added to: `g = 1`.

g is computed in Python floats and cast once to the model's dtype. The pre-hook returns
`(x * Fm[:, None, :],) + args[1:]`. Multiplying by 1 is exact, so every other row and column keeps its bits.
When no factor differs from 1 the pre-hook leaves the input alone.

`exact_at(edit, d, dtype)` says what that cast does. `factors` are the distinct pairs `[g asked, g as applied]`
over the edit's cols terms (ascending by g asked, the first 8), g as applied being
`float(torch.tensor(g, dtype=dtype))`. `input_side` is `"none"` when the edit has no cols term or every g asked
is 1; `"gone"` when some g asked is not 1 and every g as applied is 1 (a silencing at dose 0.001 is 0.999, which
is 1 in bf16: the row is the base row); `"exact"` when every g asked is 0 or a power of two with either sign
(2.1); `"rounds"` otherwise (the factor rounds to the dtype and each product rounds once, where the folded weight
rounds each entry of the column: the two differ by rounding, and the result says so). `output_add` is true when
the edit has a set or added column or a lowrank term; `folded_form` is false when it has a lowrank term.

`dose_refusal(edit, d, dtype)` is None at dose 0, and otherwise the first of these that holds:

- `input_side` is `"gone"` and `output_add` is false: `At dose {d:g} every factor of this edit is exactly 1 in
  {dtype}: its row would be the base row again. Doses are dyadic here (1, 0.5, 0.25, -1); for a smaller effect
  take fewer columns, not a smaller dose.`
- `abs(d)` times the largest `rel_change` in the edit's report (a column's `max`, a lowrank term's `bound`) is
  over `MAX_REL_CHANGE`: `At dose {d:g} this edit would change {name} by {x:.0f} times its own length; 1a takes
  at most {MAX_REL_CHANGE:g}.`

**Column set and add: the new content from the row's own activation, on the output side.** The forward hook's
`args` are the pre-hook's OUTPUT, so the pre-hook must first keep what the forward hook needs from the original
input: for each row with set or added columns, `a = x[r][:, cols]` (`[T, K']`, the model's dtype). The forward
hook then writes, for such a row,

    acc = out[r].float() + a.float() @ C          # C [K', out] float32
    out2[r] = acc.to(out.dtype)

with `C = ((v * d).to(model dtype)).float()`: the content at this dose, rounded to the model's dtype as the
folded form rounds the column it writes, and the sum made in float32 and rounded once. That is the arithmetic of
`int4_cpu.with_overlay_terms` (`backend/int4_cpu.py:138-164` [read]). It is NOT one of the pilots' hooks (theirs
multiplied and summed in bf16) and has not been measured on the card: the result marks every arm that uses it
(`exact_at(...)["output_add"]` true), and the folded twin reads its gap at the reading's own batch shape (4.7).

**Low-rank: an output-side add.** The pre-hook keeps, for each row with a lowrank term on this linear,
`z = x[r].float() @ A.T` (`[T, r]` float32, from the original input). The forward hook adds
`(z * d) @ B.T` into the same float32 `acc` as the column content, so a row with both rounds once. It has no
folded form in 1a (`exact_at(...)["folded_form"]` false), so no twin is taken and a rank is all it gets. On
`gate_proj` and `up_proj` it is the ADDED form of a change to what neurons read, which the ruling struck for row
edits (the edited coordinate recomputed, never added to); 1a builds it as decision B says and the edit's report
says so (2.2, `form`).

**Null rows.** For `Null(layer, e)` the forward hook on that layer's `down_proj` writes `out2[r] = out[r] * s`,
with `s` the tensor of `we.null_factor(e, out.dtype)` in `out.dtype`: the product in the model's dtype, as pilot 5
ran it. The norm after the block undoes the scale but for rounding. `per_row` refuses a `Null` whose factor is 1
or a power of two (2.6's words): the first is the base row again, the second an exact scaling with no rounding
to leave.

The forward hook clones `out` once (`out2 = out.clone()`), writes only the rows above, drops what the pre-hook
kept, and returns `out2`. An untouched row passes through no arithmetic at all. A row at dose 0 may pass through
the arithmetic or be left alone, as the implementer chooses: either way it is the base row by bits (checked, 6.1
§4a, 6.2 §3).

**"By bits"** in this spec, as in the pilots, means `torch.equal` on final-normed states (which reads -0 as 0).

### 4.2 The layout of doses and rows

Fixed in 2.6. `levelling.layout` is the one place it is computed, and `read`, `estimate`, the checks and the card
check all call it. A key becomes a program so: None stays None; `"null:<e>"` is `we.Null(L0, e)`; `"edit:<d>"` is
`we.Arm(edit, d)`; `"control:<c>"` is `we.Arm(controls[c], main_dose)`.

### 4.3 A decode step

`write_rows` first runs the prompt (`[R, T, .]`), then one new position a step with a cache (`[R, 1, .]`), and
keeps every row in the batch after it ends (`atlas_ops.py:590-617` [read]), so R never changes. `per_row`'s
hooks need nothing more. The written reading passes `iv=None`: no push of the atlas's is installed beside them.
Pilot 3 wrote under an output-side hook with no row stopping (fact 8); the input-side pre-hook in a decode step,
and a batch in which rows stop, are new on the card (6.2 §7) and run on both stand-ins [ran on the CPU].

### 4.4 Capture and restart

`L0 = edit.lowest_layer()`: every arm's lowest touched layer (control edits sit in the edit's own layers) and
the layer of the null rows. By `plan["lower"]`:

- `"same_shape"` (default). Forward 0 of a class is a whole forward at R rows under its programs. When the
  class has later forwards (F >= 2), one more pre-hook on `lm.layers[L0]` (`with_kwargs=True`) keeps
  `(args[0] if args else kwargs["hidden_states"]).detach().clone()`, the stream entering layer L0, `[R, T, D]`,
  and forwards 1 on restart from it. The stream is kept on the model's device, for that item and class only,
  and dropped when the class is done. This is the case measured exact (fact 6); with hooks installed and below
  layer 20 it is the card check's to show (6.2 §6), and every reading shows it again: a later forward's base row
  and null rows must be forward 0's by bits (4.5).
- `"whole"`. Every forward is whole; nothing is kept.
- `"one_row"`. One whole forward at ONE row with only the capture hook, once an item; its stream expanded to
  R rows (`.expand(R, T, D).contiguous()`) feeds every forward of both classes, forward 0 included. Refused
  below `bars.one_row_min_layer` (fact 7: shown at layer 44 only). The result marks both classes
  `"lower": "one_row"`: the layers below the edit ran at one row, so the two classes differ only from layer L0
  up, and the printed lines say so (5.3).

The base row is never remembered: every forward carries its own, and an arm is read against the base row of the
forward it sat in.

### 4.5 What voids a reading, and the null rows

After each forward of a class, in this order:

1. row 0 holds only finite numbers; else void, `non_finite_base` (first, because a NaN never equals its twin);
2. every untouched row equals row 0 by bits, the twin first; else void, `untouched_rows_differ`;
3. every arm at dose 0 equals row 0 by bits; else void, `zero_dose_differs` (the proposal's zero row, carried
   when the plan lists a dose of 0);
4. for forward f > 0: its row 0 and its null rows equal forward 0's by bits; else void, `repeat_differs`
   (forward 0's rows 0 to n0 are kept for this, `[1 + n0, T, D]`).

A void stops the whole reading at once: its mechanism is not sound, and what was read so far is written with
`status: "void"`. Running out of memory voids it too (`out_of_memory`, with the shape that did not fit): a reading
never halves its rows, because the row count is part of what it is. A `KeyboardInterrupt` is entered as
`interrupted`, and any other exception as `error` with `why` its type and words (the hooks' own `RuntimeError`,
`NotBase` out of `folded`, a CUDA error that is not out of memory); both are raised again once the entry is
written, so that every call is in the book. However a reading ends, `we.assert_base` runs last and its outcome
is the result's `base_after`; when it fails on a reading that was otherwise sound the reading is void,
`not_base_after`.

An arm row with a non-finite number is not a void (an edit can break the model): its metrics are None and
`non_finite` is true. Every arm also records `same_as_base`: `torch.equal` of its row with row 0 of its own
forward. A dose or an added vector that the rounding eats shows there, and the row's line says so in place of
a figure (5.3).

**The floor** of a metric, on one item in one class, is the largest `abs(mean)` among that class's null rows on
that item (forward 0's; the others are the same rows by bits), in the metric's own units; `median_abs` and
`above_1` are the null rows' largest too, `from` names the row and `of` says how many null rows it is the largest
of. There is no constant floor. Null rows all act at L0, the edit's lowest layer, where the pilots' floors were
largest (fact 5).

### 4.6 The folded context and the base assertion

`hook_census(lm)`: for every module under `lm` that has any, the sorted ids of its forward pre-hooks and forward
hooks (`sorted(list(m._forward_pre_hooks) + list(m._forward_hooks))`: the keys are the handles' ids, which no two
hooks of a process share). `assert_base(lm, hooks_at_load=None)`: `hook_census(lm)` equals `hooks_at_load` (an
empty dict when None), so a hook swapped for another is seen as well as one added; and every dict found as
`_memit_col_overlays` or `_memit_row_overlays` on any module is empty. Otherwise `NotBase`, naming up to three
offenders. `handle()` takes the census at load, so a runtime that carries hooks of its own at rest (none is
registered by `engine_runtime`, `runtime_factory` or `w4a16_fast` [read, by grep]; neither stand-in carries one
[ran]) is told apart from a leak.

`can_fold(lm, edit)` gives the sentence `folded` would refuse with, or None: a lowrank term (2.6's words); a
touched linear whose forward reads no overlay, that is, one that is neither packed nor a linear given
`gguf_weights.install_overlay_forward` (it has no `forward` of its own in `vars(mod)`): `{name} is a plain linear
whose forward reads no overlay: the folded form is the packed checkpoint's.` The float32 stand-in's linears are
plain, so on it no twin is taken unless the check installs that forward.

`folded(lm, edit, dose, hooks_at_load=None)`, the same edit written into the overlay dicts every packed linear
carries (`engine_runtime.py:352-377, 429-436` [read]; the compiled unpacker keeps that forward). It is entered
outside any `per_row` block:

1. `overlays_in_force(lm)` must be empty, else `NotBase` (the vault's edit is in force).
2. `can_fold(lm, edit)` must be None, else `Refused` with its sentence.
3. Snapshot: a shallow copy of both overlay dicts of EVERY module under `lm` that has them (all 410 on the
   31B), not only the touched ones.
4. For each cols term read the base columns
   `Wc = dense_cols(mod, torch.as_tensor(idx, dtype=torch.long)).float()` (`[K, out]`; `dense_cols` takes a
   LongTensor and returns the model's dtype on the model's device) and write, for column j,
   `new = (1 + d * (f - 1)) * Wc[j]` (scale), `(1 - d) * Wc[j] + d * v[j]` (set), `Wc[j] + d * v[j]` (add), in
   float32, cast once to the model's dtype, on the module's device; where the factor at this dose,
   `1 + d * (f - 1)`, is exactly 0, zeros (as the pilots wrote). `mod._memit_col_overlays[int(n)] = new`.
5. `yield`.
6. In `finally`: put every snapshotted dict back in place (`d.clear(); d.update(saved)`), then
   `assert_base(lm, hooks_at_load)`.

### 4.7 One paired reading, step by step

`read(h, reading, items)` for `kind: "paired"`. Steps 2 to 6 run inside one `try`; step 7 runs however they end.

1. `lb.head()`: the book can take an entry, or `BookBroken` is raised and nothing else happens.
2. `we.assert_base(h.lm, h.hooks_at_load)`; `normalise` (with `dtype=h.dtype`); `we.from_recipe` then `we.build`
   (or `we.load`, or the `edit` handed in); `we.dose_refusal(edit, d, h.dtype)` for every dose;
   `we.control_edits(edit, h.lm, seed, count)`; the two size refusals `estimate` raises; under `folded: "always"`,
   `we.can_fold(h.lm, edit)`. Any `Refused`, `NotBase` or sentence here ends the reading as `status: "refused"`
   with the message as `why`, before any forward.
3. The twin is taken when `folded` is `"always"`, or when it is `"auto"`, `we.twin_wanted(we.edit_facts(edit),
   main_dose)` is true and `we.can_fold(h.lm, edit)` is None. `twin_wanted`: the edit has no lowrank term and
   either has a set or added column or some `g = 1 + main_dose * (f - 1)` over its scale factors is neither 0
   nor a power of two. When no twin is taken `folded.why` is one of: `the plan says folded: never`; `a silencing
   or a scale by a power of two is exact on the input side`; `the edit holds a low-rank term, which has no folded
   form in 1a`; or `can_fold`'s sentence.
4. `estimate(..., twin=<step 3's answer>)`; note the unpacker's mode (`w4a16_fast.describe().get("mode")`) and
   its fall-back counter (`getattr(backend.engine_runtime.dequantize_w4a16_full, "fell_back", None)`, read again
   at the end; a counter that moved is recorded, not a void: the reference unpacker gives the same bits, seven
   times slower); start the clock (after `torch.cuda.synchronize()` on a card).
5. For each item in plan order:
   a. **the small class, begun**: reset the card's peak counter (`torch.cuda.reset_peak_memory_stats()`);
      forward 0 (4.4) under `we.per_row(h.lm, programs_0)`; the checks of 4.5; the gate's figure (4.10) from its
      base row; keep that row's final states at the span and end positions (`[S + 1, D]`) for step d;
   b. if the small figure is under the bar, begin the large class the same way for its figure alone, mark the
      item `"refused"` with both classes None, and go on to the next item;
   c. **the small class, read**: every null row and arm of forward 0 against its row 0 (4.8); forwards 1 to
      F - 1, each under its own programs, with 4.5's checks, its arms read against its own row 0 (its null rows
      are forward 0's by bits and are not read again); the twin (e); drop the kept stream; record the peak;
   d. **the large class**: begun as in a; `stability.base_rows_same_bits` is `torch.equal` of its base row's
      span-and-end states with the small class's (true means the two row counts ran one arithmetic on this item,
      and the item was read once, not twice: 4.9). If its figure is under the bar the item is `"refused"`, the
      small class staying as read and the large None; else it is read as in c;
   e. **the twin**, when taken: `with we.folded(h.lm, edit, main_dose, hooks_at_load=h.hooks_at_load):` one
      forward of R rows. It restarts from the kept stream only when the class has later forwards (so that rule 4
      of 4.5 has already shown the restart exact at this shape on this item) or under `lower: "one_row"`;
      otherwise it is a whole forward; `folded.restart` says which. Rows 0 and R - 1 equal by bits
      (`rows_identical`; false voids, `untouched_rows_differ`); its row 0 is read against FORWARD 0's base row.
      This is the one place a base row is taken from another forward: the folded form has no untouched row by
      construction, the two forwards are of one shape, and the result says
      `"base_from": "forward 0 of the same shape"` (pilot 6 read its folded forward the same way).
      `gap[m] = {"abs": abs(folded mean - per-row mean), "rel": abs / max(abs(folded mean), abs(per-row mean))}`
      (`rel` None when both are 0); `between_kl_mean` is the mean over span positions of
      KL(folded row || per-row row) when `kl` is on; `same_bits` is `torch.equal` of the folded row 0 and the
      per-row `edit:<main_dose>` row of forward 0; `seconds` is the twin's own wall time.
6. Stability, ranks and the summary (4.9).
7. Always: `we.assert_base(h.lm, h.hooks_at_load)` for `base_after`; stop the clock; build the result;
   `lb.append`; then raise again whatever step 2 to 6 raised that was not a refusal or a void (4.5).

### 4.8 The read-outs

All logits are fp32 and softcapped, from the final-normed states cast to float32, as
`causal_check.logits_fp32` computes them (`causal_check.py:132-139` [read]): the model's own bf16 logits differ
from these by a mean KL of 0.0013 nats, the size of the effects read here (fact 9).

For one forward with final states `Hf`, span positions P and targets `y[p] = ids[p + 1]`:

- **Full vocabulary** (when `kl` or `own` is on; and always for row 0 of forward 0 of each class, for the gate).
  For a row r: `L_r = log_softmax(logits_fp32(Hf[r, P], h.W_U), -1)`, `[S, V]` float32. Then, a position:
  `kl[p] = sum_v exp(L_0[p, v]) * (L_0[p, v] - L_r[p, v])`; `own[p] = L_r[p, y[p]] - L_0[p, y[p]]`; the base
  row's own values are `L_0[p, y[p]]`. Work through P in chunks of `READ_CHUNK` positions and through the rows
  one at a time. `logits_fp32` is not in place and the KL holds both rows' tables and their product, so about
  four `[READ_CHUNK, V]` float32 tables exist at the peak: 0.5 GiB at 128 positions and V = 262,144 [ran on the
  CPU at 64 positions: 4.1 tables]. Reduce on the device and bring back only the per-position values.
- **Named pieces** (`odds`). With `Wt = h.W_U[toward].float()`, `Wa = h.W_U[away].float()` and Q the end
  position or P: `zt = softcap(Hf[r, Q].float() @ Wt.T)`, `za` likewise,
  `softcap(z) = tanh(z / 30) * 30` (30 is `FINAL_LOGIT_SOFTCAP`);
  `lo_r[q] = logsumexp(zt, -1) - logsumexp(za, -1)`; `odds[q] = lo_r[q] - lo_0[q]`. A few rows of the
  unembedding, no softmax over the vocabulary: the difference of two log-sum-exps of logits is the log-odds of
  the two groups, and with one piece a side it is the logit difference. These agree with the same pieces read
  from the full table to rounding, not by bits. No floor or stability has been measured in log-odds on the card
  (6.2 §9a reads the first).

Each metric's per-position values go through `stats`. On the float32 stand-in the same code runs in float32.

### 4.9 Control edits, ranks, stability, the summary

**Control edits** (`we.control_edits(edit, lm, seed, count)`), one family in 1a: other sites of the same layer
and size. Control c maps each term t (its place in the sorted order) with
`rng = np.random.default_rng([int(seed), int(c), int(t)])`:

- cols: `pool = np.setdiff1d(np.arange(N), term.idx)`; `chosen = np.sort(rng.choice(pool, size=K, replace=False))`;
  column `chosen[j]` takes the op, factor and vector of the edit's j-th column;
- lowrank: `perm = rng.permutation(N)` on the neuron axis (the in-axis of `down_proj`, the out-axis of
  `gate_proj` and `up_proj`): `A' = A[:, perm]` on a `down_proj`, `B' = B[perm, :]` on the other two.

Each control is an `Edit` on the same base with its own id and the note
`control {c} of {edit.short()}, seed {seed}`. A builds it without the size caps (a set carried to a short column
can pass `MAX_REL_CHANGE` where the edit did not), and its report says `"caps": "not applied to a control edit"`.
Every result carries, word for word (`CONTROLS_SAY`),

> Control edits are other sites of the same layer and size, drawn from seed {seed}. They are NOT matched on how
> hard the sites fire on these items or on what the edit delivers, and the edit's own sites were chosen where
> theirs were drawn, so a rank against them says only that the edit differs from an unmatched random edit of its
> size.

and every paired result (`ROWS_SAY`, in `rows.says`; `PAIRED_SAY`, printed by `lines`)

> Every item was read at {small} rows and at {large} rows. Agreement at these two row counts does not show that
> another row count would agree: on 14 November 2026 batches of 2, 3 and 4 rows each read differently.

> These figures are read along the answer given, piece by piece. They do not say what the edited model would
> write.

**Rank**, on one item in one class, for metric `rank_by` at `main_dose`: `value = abs(mean)` of the edit's arm;
`rank_among(value, [abs(mean) of each control, None for a control that is not finite])` returns `at_least`
(controls whose value is `>= value`, a non-finite control among them), `non_finite` (how many of those were not
finite), `of = C + 1` and `rank = at_least + 1`; when `value` is None (the edit's own arm is not finite) `rank`
and `at_least` are None. `over_floor = value / floor.mean_abs` (None when the floor is 0);
`controls_over_floor` is how many control arms read above the floor at all: a rank against rows that are
themselves at the floor shows for what it is.

**Stability** between the two classes, for every read-out that is on and every arm that is an edit or a control:
`agree(a, b, floor, signed, ratio)` with `a`, `b` the two classes' means and `floor` the larger of the two
classes' floors of that metric:

    if a is None and b is None: return None, None        # not finite in either class: not compared
    if a is None or b is None:  return False, None
    lo = max(min(abs(a), abs(b)), floor)
    r = max(abs(a), abs(b)) / lo            (1.0 when both are 0; infinity when lo is 0 and they are not)
    ok = r <= ratio and (not signed or a * b >= 0 or max(abs(a), abs(b)) <= ratio * floor)
    return ok, r

`signed` is false for `kl`, true for `own` and `odds`. An item is **unstable** when any such arm disagrees on
any read-out; its `status` is `"unstable"`, its `stability.disagree` lists the pairs, and it is left out of the
summary and counted. An item whose `base_rows_same_bits` is true keeps the status its arms give it (they agree
by construction) and is counted apart as `one_arithmetic`: it was read once. The rule is provisional and so is
its number, 3.0: over the answer's span pilot 8's largest ratio was 1.60 (sets of 10 or more: 1.20); pilot 7,
read over every position, saw 2 readings of 120 differ over twofold and none over fivefold. The card check
measures the ratios, control arms included, and proposes the number (6.2 §9).

**The summary**: `{"items": {"asked", "kept", "refused", "unstable", "one_arithmetic"}, "classes": {"small":
{...}, "large": {...}}}`, each class `{"rows", "arms": {<key>: {<metric>: {"mean_over_items",
"median_over_items", "median_of_medians", "above_1_total"}}}, "floor": {<metric>: <the largest floor over kept
items>}, "rank": {"metric", "dose", "value", "at_least", "non_finite", "of", "rank"}}`. Over the kept items:
`mean_over_items` and `median_over_items` are of an arm's per-item `mean`, `median_of_medians` is the median of
its per-item `median`, `above_1_total` the sum of its per-item `above_1`; the summary's rank sets the edit's
`abs(mean_over_items)` against each control's. With no item kept the classes are None.

### 4.10 The item gate

From forward 0's base row in each class: `mean`, `median` and `worst` (the least) of `L_0[p, y[p]]` over the
span. The item passes when both classes' `mean >= bars.item_gate`. Otherwise its `status` is `"refused"` and
`why` is `The base model gives this item's answer {mean:.2f} nats a piece on average at {R} rows, under the bar
of {bar:.2f}. On 14 November 2026 one raw text of two read at about that level gave readings that moved
eighty-fold with the batch size, so an item under the bar is not read.` Both classes' figures are taken before
an item is refused (they differ: pilot 8's raw recipe read -10.09 at 2 rows and -9.88 at 24).

The bar, -6.0, is provisional: it sits between the lowest chat-formatted item read (-5.51, a passage given as
the answer; a longer cut of the same passage behaved in pilot 7) and the raw texts with no start mark (-7.64 in
pilot 1, -9.15 to -10.09 in pilot 8) (fact 12). Whether the base's log-probability is the right thing to gate on
is itself unmeasured: of two raw texts under the bar one misbehaved and one (pilot 7's raw prose) did not. The
card check reads the gate's figure and the two row counts on items from -0.1 to -10 nats with the gate set aside
(6.2 §8 and §9), and "the bar does not separate" is one of the things it may find.

### 4.11 The cost estimate

With `whole(R, T) = 0.08 + 0.00033 * R * T` seconds and `share = (n_layers - L0) / n_layers`, for one item of T
pieces and S scored, in one class of R rows and F forwards:

| `lower` | forwards |
|---|---|
| `same_shape` | `whole(R, T) + (F - 1) * whole(R, T) * share` |
| `whole` | `F * whole(R, T)` |
| `one_row` | `F * whole(R, T) * share`, plus `whole(1, T)` once an item |

- the twin: `whole(R, T) * share` when it restarts (F >= 2 under `same_shape`, or `one_row`), else `whole(R, T)`;
  `+ 0.0003 * (linears touched)`. A twin of many columns costs more than this: the pinned forward writes each
  overlay column in a Python loop on every forward (`engine_runtime.py:361-364` [read]) and the 0.3 ms was
  measured to 64 columns. Unmeasured; the card check times it (6.2 §5b);
- read-out, when `kl` or `own` is on: `0.00011 * S * (F + n0 + nd + C + twin)` (each forward's own base row,
  the null rows once, every arm, the twin's row); else `0.00011 * S` (the gate's base row); named pieces are
  counted as free [not measured];
- a written reading, an item: `whole(R, prompt pieces) + 0.095 * max_new`.

`gib_above_model`: a paired reading, the largest forward's `0.25 MiB * R * T` plus 0.5 GiB for the
full-vocabulary read-out [inferred]; a written reading, `0.8e6 * rows * (prompt pieces + max_new) / 2**30` (the
writer's cache, fact 15: 32 rows to 286 pieces is 6.8 GiB). `estimate` raises 2.6's two size refusals: a forward
over `MAX_ROW_POSITIONS`, a written reading over `MAX_WRITTEN_GIB`. `read` puts the estimate in
`seconds.predicted` and the clock's in `seconds.actual`; the command line adds the load (9.5 s + 1.83 s) once a
run. Every figure in `COST` is in the module docstring with its date.

### 4.12 The written reading

`read` for `kind: "written"` goes as 4.7 does but for its steps 3, 5 and 6 (the book first; the base assertion,
the edit and the refusals; the estimate and the clock; and step 7 whole). An item: rows `[base, null rows, the
edit at each dose, control edits 0 to C - 1, twin]`, padded with untouched rows before the twin to exactly `rows`
(`layout(..., one_forward=True)`). Then, with the hooks staying installed until the generator is spent:

    with we.per_row(h.lm, programs):
        for st in ao.write_rows(h.lm, h.W_U, [list(item.prompt_ids)] * R, None, max_new=max_new,
                                stop_ids=h.rt.end_of_turn_ids(), bos_id=h.tokenizer.bos_token_id,
                                softcap=h.softcap):
            ...                         # st.row, st.token_id, st.logprob, st.end

Greedy (no sampler), from the item's prompt alone: the model writes its own answer, so there is no item gate.
A row that writes an end mark yields no more steps and stays in the batch. Every untouched row, and every arm at
dose 0, must write the base row's pieces with the base row's log-probabilities, exactly; else void,
`untouched_rows_differ` or `zero_dose_differs`. (On the float32 stand-in that holds only at one thread, section
1; on the card pilot 3 saw it hold for 48 pieces.) For each null row and arm: its pieces and text;
`same_as_base` (its pieces and log-probabilities are the base's, exactly); `parts_at`, the index of the first
piece that is not the base's (None when none differs); and `logprob_gap` over the pieces before that, the
absolute difference of the row's and the base's log-probability of the same piece (`max_abs`, `mean_abs`, `n`).
These are the writer's own log-probabilities, from the model's dtype logits: not the fp32 measure. The reading is
greedy, taken at one row count, recorded, and is not a two-class reading; its lines say all four (5.3).


## 5. The command line

`scripts/levelling.py`, run as `.venv/bin/python -I scripts/levelling.py <sub-command>`. It puts the repository
root and `scripts/` on `sys.path` as `scripts/census_run.py:129-132` does.

### 5.1 Sub-commands

| sub-command | what it does |
|---|---|
| `estimate PLAN [--stand-in]` | No model. Loads the tokenizer alone (`AutoTokenizer.from_pretrained(DEFAULT_W4A16_MODEL_ID, local_files_only=True)`; the stand-in's with `--stand-in`), builds the items, normalises every reading, prints each one's forwards, row-positions, seconds and GiB, and the run's total with the load. `n_layers` is `theodolite_math.N_LAYERS` (60), or `atlas_stand_in.N_LAYERS` with `--stand-in` |
| `run PLAN [--stand-in] [--only NAME ...] [--out FILE] [--save-edits] [--port 8765]` | One load of the model, every reading of the plan in order (or those named), each estimated, read, entered in the book and printed. `--out` writes all results as one JSON list (`census.write_json_atomic`). `--save-edits` passes `save_edit=True` to `lv.read` |
| `book [--tail N] [--seq K] [--verify]` | No model. `--tail` (default 20): one line an entry (seq, time, kind, status, the edit's first 12, name, seconds). `--seq K`: that entry's JSON, indented. `--verify`: walks the chain |

`run`, in order:

1. refuse when `THEODOLITE_GGUF` is set or `THEODOLITE_DEVICE` names anything but `cuda:0`
   (`1a reads the W4A16 checkpoint on the card, or the stand-in with --stand-in.`);
2. **refuse while port 8765 answers**, by the probe of `scripts/verify_causal_check.py:98-101` (a TCP connect to
   127.0.0.1 with a 0.5 s timeout): `Something answers on :8765: stop the Theodolite server first (this loads the
   model itself).`;
3. refuse without CUDA; then **ask the card what is free**, as `scripts/census_bars.py:278-284` does (a census
   does not answer on :8765): with `free, _ = torch.cuda.mem_get_info()`, under `MIN_FREE_GIB = 22.0` GiB,
   `Only {free:.1f} GiB of the card is free ({MIN_FREE_GIB:g} are asked: the load peaks at 21.2 with the lens,
   before it is freed): stop the :8765 server and any census first.` [provisional; fact 15];
4. `lb.head()`; `rt = make_runtime(load_jlens=False).load()`; free the lens as `scripts/census_run.py:135-150`
   does (`rt.jpp_lenses.clear()`, `gc.collect()`, `torch.cuda.empty_cache()`, since
   `make_runtime(load_jlens=False)` still loads it); `w4a16_fast.install(rt, "compiled", log=print)` (a
   `w4a16_fast.Refused` ends the run); `h = lv.handle(rt)`;
5. the readings. After one whose `base_after` is false, or out of which `lv.read` raised, nothing more is read:
   `The model is not the base after "{name}": nothing more is read in this process.`, or the exception's own
   words; exit 1;
6. `we.assert_base(h.lm, h.hooks_at_load)`; the book's head.

It never sets `TORCHINDUCTOR_CACHE_DIR`: `make_runtime` sees to the compile cache (section 9).

With `--stand-in`: hide the GPU and go offline before torch is imported; `torch.set_num_threads(1)` (section 1:
above one thread a written reading on the float32 stand-in is void by rule); `use_stand_in_atlas()`;
`rt, _vault = atlas_stand_in.stand_in()`; no port probe, no memory question, no unpacker; and, unless the
variables are already set, `THEODOLITE_LEVEL_BOOK` and `THEODOLITE_EDITS_DIR` point into a fresh temporary folder
(`tempfile.mkdtemp(prefix="levelling-stand-in-")`), named in the first line printed, so a dry run never writes
the real book or the checkout. The stand-in's linears are plain, so no twin is taken (4.6), and its items read
about -6.4 nats a piece: a stand-in plan gives `"bars": {"item_gate": -8.0}` or every item is refused by the gate.

### 5.2 The plan file

    {"v": 1, "name": "<what this sitting is for>",
     "items": {"<name>": {"question": "...", "answer": "...", "system": null}, ...},
     "readings": [<a reading's plan, 2.6>, ...]}

A worked example a person could type (saved anywhere; the build adds no folder for plans):

    {
      "v": 1,
      "name": "two layer-20 neurons silenced, on two questions",
      "items": {
        "moon": {"question": "In two or three sentences, why does the moon show phases?",
                 "answer": "The moon shows phases because we see different portions of its sunlit half as it orbits the Earth."},
        "stock": {"question": "Explain briefly why a stock should not be salted while it simmers.",
                  "answer": "You should not salt a stock while it simmers because of evaporation."}
      },
      "readings": [
        {"name": "silence 4294 and 9618",
         "kind": "paired",
         "edit": {"terms": [{"layer": 20, "silence": [4294, 9618]}],
                  "note": "the two that tipped the raw recipe in pilot 7"},
         "doses": [1.0, -1.0],
         "controls": {"count": 19, "seed": 0}},
        {"name": "the same, written",
         "kind": "written",
         "edit": {"terms": [{"layer": 20, "silence": [4294, 9618]}]},
         "items": ["moon"]}
      ]
    }

The first reading takes, an item: 4 forwards of 12 rows (base, 4 null rows, 6 arm slots, twin; 2 doses and 19
control edits are 21 arms) and 1 forward of 32 rows; no twin (a silencing is exact on the input side, and its
mirror is a scale by 2); KL and the text's own log-probability over the answer's pieces, since the plan names no
pieces. The second is one batch of 8 rows: base, 2 null rows, the edit, 3 control edits, twin.

### 5.3 What is printed

`lv.lines(result)` writes it; the command line prints those lines and adds nothing of its own but the estimate
before a reading and the book's line number after. The only status words are "ok", "refused", "void" and
"unstable". The lines must give, in plain words:

- the reading's name and kind; the edit's first 12 and what it is (terms, columns, "layer 20 (page 21)"; a
  lowrank term's `form`); the base model; a bar the plan gave in place of the module's, marked as the plan's;
- a paired reading: the two row counts, doses, null rows a forward, control edits with `CONTROLS_SAY`;
  `lower` when it is not `same_shape` (under `one_row`: "the layers below the edit ran at one row at both row
  counts"); `ROWS_SAY` and `PAIRED_SAY`, once each;
- for each item: its pieces; the base's mean log-probability at each row count beside the bar; and for each row
  count the edit's mean, median and positions above a nat; the floor, which null row it came from and of how
  many ("the largest of 4"); how many control edits read as much, the rank, and how many control edits read
  above the floor at all; the twin's figure beside the per-row one with their gap, or "no twin taken" and why;
- an arm whose row is the base row by bits says so in place of a figure ("this row is the base row by bits");
  an arm that uses the output-side add says so ("output-side add, summed in float32");
- whether the two row counts agree and by what ratio, named by their numbers ("12 and 32 rows agree"), for each
  read-out that is on; an item whose base rows were the same bits: "12 and 32 rows gave the same base row by
  bits: one arithmetic on this item, so it was read once, not twice";
- the summary over kept items with the row counts beside its figures, and how many items were refused, unstable
  and read in one arithmetic; seconds beside predicted;
- a written reading: that it is greedy, at one row count, not a two-row-count reading, and that its
  log-probabilities are the writer's own, from the model's dtype; then each row's text, where it parts from the
  base and the gap before that.

An example of the shape, not of the wording:

    reading "silence 4294 and 9618" (paired): edit 3fa2c1d0e9ab, 2 columns of layer 20 (page 21) silenced
      rows 12 and 32; doses 1, -1; 4 null rows a forward; 19 control edits, other sites of layer 20 (page 21),
      seed 0, NOT matched on firing or on what the edit delivers, the edit's own sites chosen
      read along the answer given: not what the edited model would write
      item moon (82 pieces, 56 scored): the base gives -0.08 nats a piece at 12 rows, -0.08 at 32 (bar -6.00)
        12 rows: KL mean 0.01295, median 0.00013, 0 positions above a nat; floor 0.00016 (null +0.125, the
                 largest of 4); 0 of 19 control edits read as much (rank 1 of 20), 2 of 19 above the floor;
                 no twin taken: a silencing is exact on the input side
        32 rows: KL mean 0.01310, ...
        12 and 32 rows agree on KL (ratio 1.01, bar 3.00) and on the text's own log-probability (1.03);
        this does not show another row count would agree
      over 2 items kept of 2 (0 refused, 0 unstable, 0 read in one arithmetic), at 12 rows: ...
      7.9 s (predicted 7.1 s); book line 41

### 5.4 Exit codes

| code | when |
|---|---|
| 0 | `estimate` done; `run` with every reading `ok` (items refused by the gate or unstable are results, not failures); `book --verify` with the chain whole |
| 1 | `run` with any reading void or refused, or stopped at step 5; `book --verify` with a bad line (it prints the line and why) |
| 2 | the run could not start: the port answers, no CUDA, too little of the card free, a plan that cannot be read or is refused at its top level, `THEODOLITE_GGUF` set, the unpacker refused, the book broken |


## 6. The checks

Both are standalone scripts in the house's form: numbered sections, `check(ok, "what")` lines, `section(title,
before)`, a count at the end, exit 0 when every check holds (read `scripts/check_causal_check.py:88-103, 908-945`).
Each line below is one or more `check` calls. "By bits" is `torch.equal` on final-normed states.

### 6.1 `scripts/check_levelling.py` (CPU; hides the GPU and goes offline before importing torch)

Two stand-ins: **F**, the float32 six-layer stand-in (`atlas_stand_in.stand_in()`), for plumbing; **P**, the
packed bf16 stand-in (`check_dequant.packed_stand_in()`, which returns five things, the runtime first, and takes
no seed; `scripts/check_dequant.py`, about line 152; imported as `scripts/check_gguf_fast.py:762` imports it), for
everything claimed by bits. Before anything else it calls `torch.set_num_threads(1)` (section 1) and points
`THEODOLITE_LEVEL_BOOK` and `THEODOLITE_EDITS_DIR` into a temporary folder. Every plan it runs on a stand-in
gives `"bars": {"item_gate": -8.0}` (the stand-ins read about -6.4 nats a piece), so its results read
`bars_overridden: true` and the checks expect that.

1. **The hashes.** `array_sha256`, the edit id, the item id and the two book hashes equal 2.1, 2.2, 2.5 and
   2.8's worked values. `we.canonical` and `lb.canonical` give the same bytes. Recipe terms in another order, or
   neurons listed in another order, give the same id; another factor, another base tensor or another model id
   gives another; the note does not change it. Two equal edits are not `==` (identity), and their ids are.
2. **The gate** (F). Every row of 2.2's refusal table is raised by `we.build` or `we.load` with its words, and
   nothing is installed or written when it is; an input that breaks two rules gets the earlier row's words.
   `rows`, `remove_direction` and an attention linear are refused as 1b's. A packed linear of P marked
   `int4mm = True` is refused as `w4a16 int4mm`. A recipe file that is a pickle, or of another dtype or shape,
   is refused before it is used.
3. **Save and load** (F and P). A saved edit loads to the same id and the same arrays; a scale-only edit writes
   no payload and is marked not carried; a byte changed in the payload, a factor changed in the manifest, and (F)
   a stand-in of other weights, `stand_in_runtime(stand_in_hf(tok, torch.float32, seed=1), tok)`, are each
   refused with their words. Nothing is written under the repository.
4. **Per-row hooks by bits** (P). (a) Under `per_row` with every program None, and with an `Arm` at dose 0 (a
   scale, a set, an add and a low-rank term) and a twin, every row equals an unhooked forward of the same shape.
   (b) A silenced set and a set scaled by 2 and by -0.5: the arm's row equals row 0 of a forward under
   `we.folded` at the same batch size, by bits, at 12 and at 32 rows. (c) The same edit in two rows gives the
   same bits; untouched rows beside them are the base's. (d) A restart from the kept stream under hooks equals
   the whole forward under the same hooks, at layers 0, 2 and 5. (e) After `folded` every overlay dict is as it
   was, the next forward is the base's by bits, and `assert_base` passes; `folded` raises `NotBase` when a vault
   overlay is already in force and leaves it there; an exception inside either context leaves no hook and no
   overlay; `assert_base` fails when one hook was taken off and another put on the same module (the census
   knows hooks, not counts), and passes against a census taken with a hook of the check's own in place.
5. **Per-row hooks to rounding** (F, with `gguf_weights.install_overlay_forward` on the touched linears where a
   folded form is wanted; and P where said). (a) A scale by 1.5 and by -3, a set and an add. F: each equals the
   folded form, and a model whose weight was edited in a copy, to `allclose(atol=1e-5)` on final states. P: the
   mean KL between the per-row row and the folded row over the span is under 1e-4 nats (it read 1.2e-6 to
   1.5e-6 on 14 November 2026); the largest difference of their final states is printed with no check (it read
   0.05 to 0.07 where states reach 4.2: no bound follows from bf16's step through five more layers).
   (b) F: a lowrank term equals a copy of the model with `W + d * B @ A` written into the weight, on `down_proj`,
   `gate_proj` and `up_proj`. (c) Doses: dose -1 of a silencing doubles the column; `exact_at` reads `"exact"`
   for a silencing, `"rounds"` for a scale by 1.5, and for a silencing at dose 0.001 `"gone"` in bf16 (where
   `dose_refusal` gives its sentence) and `"rounds"` in float32; `dose_refusal` gives the size sentence at dose 4
   for a column whose `rel_change` is 20. (d) A null row's factor is applied to its row alone; a `Null` with
   e = 0.001 (a factor of 1 in bf16) and one with e = 1 (a power of two) are refused. (e) A forward of another
   row count raises from the hook. (f) `can_fold` is None on P, the plain-linear sentence on F, the low-rank
   sentence for an edit with a lowrank term; `folded` raises the same sentences.
6. **A decode step** (F and P). Under `write_rows` with `per_row` round it: every row writes the same pieces
   and log-probabilities as THE SAME ROW of an unhooked write of the same shape (exactly, on both); on P every
   untouched row also equals row 0; the hooks see sequence length 1 on every step after the first; a batch in
   which one row stops early goes on for the others; an edited row's written pieces, re-read teacher-forced
   under the same program in one whole forward, are the greedy choice at every step (F).
7. **Control edits** (F). The same seed gives the same ids; each control's columns are as many, in the same
   layer, none the edit's own; ops, factors and vectors are carried over in order; a lowrank control is the
   edit's with the neuron axis permuted; too few other sites is refused; a set carried to a column the check has
   made nearly nought builds as a control though it is over `MAX_REL_CHANGE`, and its report says the caps were
   not applied.
8. **Items and plans** (F). `chat_ids` equals `rt.tokenize_prompt(rt.compose_prompt(q, chat=True))`, with and
   without a system turn; the span and end positions; `item_from_ids` takes ids with no start mark; every
   refusal of 2.6 by its words (among them a null factor of 1, one that is a power of two, e = 0.005 and 0.01 as
   one factor in bf16, and null rows all on one side); the read-outs' defaults (`odds` alone when pieces are
   named, `kl` and `own` when none is); `layout([1.0], 19, NULL_E, 12)` is 4 forwards and at 32 rows 1, row 0 and
   the last row None in each, the null rows' keys the same in each (`"null:0.03125"` first), `"edit:1.0"` in
   forward 0; `layout(..., one_forward=True)` refuses arms that do not fit.
9. **The read-outs against brute force** (F). KL, `own` and `odds` from `read` equal the same quantities
   computed directly from a whole forward's full log-softmax (`atol=1e-5`); named pieces read from a few rows
   equal the full table's; `stats` on a known vector; the floor is the largest null row and says of how many;
   `agree` and `rank_among` on hand-made numbers (a pair at the floor agrees; 0.0085 against 0.7258 does not;
   opposite signs above the floor do not; None against a number does not, None against None is not compared; a
   control that is None counts as at least as large and is counted in `non_finite`; a value of None has no
   rank).
10. **One reading, end to end** (F, then P). Both classes are taken with the planned rows and forwards; every
    forward has its base row; the gate's figure is there for both classes; `repeats_identical` and
    `untouched_rows_identical` are true; `lower: "whole"` gives the same figures as `"same_shape"` (P: by bits);
    with `kl` and `own` on, `stability.metrics` holds both; `base_rows_same_bits` equals the check's own
    comparison of two unhooked forwards at the two row counts; on P the twin is taken for a set and not for a
    silencing, with `restart` true when the class has later forwards and false when it has one; on F no twin is
    taken and `folded.why` is `can_fold`'s sentence; the result is JSON-clean, matches 2.7 key for key, holds the
    items' questions, answers and ids, and reads `base_after: true`. Each of these is a result, with its words,
    and is in the book: a refused edit; a refused plan; a silencing at dose 0.001 (P); and, in a second plan
    whose bar is -0.5, every item refused by the gate with both classes' figures. Planted failures: a hook the
    check adds to one untouched row makes the reading void, `untouched_rows_differ`; one that moves a dose-0 arm,
    `zero_dose_differs`; one that raises inside a forward is entered as void `error` and raised again out of
    `read`, with `base_after` true; a hook that acts only when the batch has `rows.large` rows makes the item
    unstable and leaves it out of the summary. `lower: "one_row"` is refused below `bars.one_row_min_layer` and,
    with that bar given as 2 in the plan, marks both classes.
11. **The written reading** (P, through `lv.read`). Rows as planned; untouched rows write the base's pieces and
    log-probabilities; a planted difference in an untouched row voids it; `parts_at`, `same_as_base` and
    `logprob_gap` on a hand-checked case; a row that ends early.
12. **The book.** Entries chain; `verify` passes; a byte changed in a middle line is named by its line; a torn
    last line makes `append` raise `BookBroken` and write nothing; a kind outside `KINDS` raises; the default
    path is outside the repository; two appends from two processes both land and the chain verifies.
13. **The estimate.** `whole(32, 176)` is 1.93856 s; a restart from layer 44 of 60 is 16/60 of it; the twin is
    priced whole when its class has one forward; a written reading of 32 rows to 286 pieces of context is 6.8 GiB
    and refused, and 8 rows to 78 pieces is 0.46 GiB; the forwards, read-out and twin parts for the worked plan
    of 5.2 by hand; `read`'s `seconds.predicted` equals `estimate`'s for the same `twin`.
14. **The command line.** `estimate` and `run --stand-in` on a temporary plan with one paired and one written
    reading (the float32 stand-in, at one thread: exit 0, both `ok`); run once with the two variables taken out
    of its environment, its first line names a temporary folder of its own and nothing is written under the
    repository; `book --tail`, `book --seq`, `book --verify`; each exit code of 5.4, the port guard among them
    (a socket the check opens on a free port, passed as `--port` to a `run` without `--stand-in`: exit 2 before
    anything is loaded); `--only`, `--out`, `--save-edits`.
15. **The words.** None of proof, held, seal, erratum, sighting, plate, station, "field book", "chance edit"
    occurs as a word in the four new source files or in anything `lv.lines` printed above.

What it cannot show is the card's arithmetic: the two batch classes, bf16 on the GPU kernels, the cost.

### 6.2 `scripts/verify_levelling.py` (the card; server stopped)

It starts as `run` does (steps 1 to 4 of 5.1: the port guard, the card's free memory, the lens freed, the
compiled unpacker) and writes `experiments/results/levelling/verify_1a.json` after every section
(`census.write_json_atomic`). Its readings go through `lv.read` and so into the book, each named `verify: ...`
(some 400 entries a run; section 9). A sweep is ordinary readings, one edit and one item each. An item's
hardest-firing neurons at a layer are found by the script's own one-row whole forward with a pre-hook on that
layer's `down_proj` (as `pilot8_first_look.py:160-166`): the mean absolute activation over the span positions,
largest first; the hook is removed before any reading. The whole check is about half an hour of the card
[inferred from 4.11]. A line with no tolerance is recorded and has no `check`.

**Items** (50), built once, their texts copied into the script from the files named:

- *own* (6): the six questions of `pilot/pilot8_first_look.py:55-62`, each with the model's own greedy answer
  (written first at one row, 72 pieces at most, the end mark dropped): `item_from_ids(chat_ids(q), answer)`;
- *study chats* (12): the twelve `"chat"` entries of `causal_check.STUDY_TEXTS`, the user turn as the question
  and the begun answer as the answer: short answers a person wrote;
- *passages as answers* (16): pilot 7's recipe and survey passages whole (`pilot/pilot7_batch_size.py:46-61`;
  150 and 129 pieces) and pilot 8's cuts of them (`pilot8_first_look.py:63-75`; 90 and 64 pieces), each as the
  answer to pilot 8's question for it; and the twelve `"prose"` entries of `STUDY_TEXTS`, each as the answer to
  "Write the opening of a short passage.";
- *raw, as the pilots built it* (8): those four passages and the first four `"prose"` entries, tokenised alone
  (`add_special_tokens=False`) and built as `item_from_ids([ids[0]], ids[1:])`: no start mark, every position
  but the last scored, as pilots 6 to 8 read it;
- *raw behind the start mark* (8): the same eight as `item_from_ids([bos], ids)`.

| § | standing line | tolerance | where the number came from |
|---|---|---|---|
| 1 | The model is 60 layers of 21,504; no overlay; `fell_back` is 0; the hooks at load (how many, on what) | exact; recorded | the runtime's shape; `w4a16_fast` |
| 2 | `chat_ids` equals `rt.tokenize_prompt(rt.compose_prompt(q, chat=True))` on every question; whether `tokenizer(text, add_special_tokens=True)` adds a start mark | exact; recorded | `census.py:2048`; section 1 |
| 3 | **A zero-edit row by bits**: readings whose doses are `[1.0, 0.0]` (a scale; a set to a seeded vector; a low-rank term on each of the three linears) are not void and their dose-0 arm reads `same_as_base`; by the script's own forwards, a hooked forward's row 0 equals an unhooked forward's, at 12 and at 32 rows | exact | pilots 1 and 6; the set and the low-rank term are new |
| 4 | **A row's place does not matter**: one silencing in rows 1, 9, 20 and 31 of 32 (the script's own forward under `we.per_row`) gives the same bits | exact | pilot 6 |
| 5 | **Input-side equals folded at the same row count, silenced**: on one *own* item and the recipe cut as an answer, each item's own hardest-firing neuron and hardest-firing sets of 64 and 1,000 at layers 5 and 20, `folded: "always"`, 12 and 32 rows: the per-row and folded mean KLs differ by at most 1e-5 nats and by at most 1 part in 10,000 of the larger | both | pilot 6: seven single neurons of the raw recipe at 2 and 32 rows, equal to every printed digit (five and six) |
| 5a | The same sets **scaled by 2** | 5's | new: no pilot ran an input-side factor other than 0; equal by bits on the packed stand-in |
| 5b | `same_bits` in 5 and 5a; the twin's seconds at 1, 64 and 1,000 columns | recorded | not measured (fact 10 is to 64 columns) |
| 5c | The gap to folded of a scale by 1.5 and by -3, of a set and of an add (the float32 output-side add), on the set of 64 | recorded | not measured; any bar for these forms is set from this |
| 5d | **Like for like with pilot 6**: the raw recipe (pilot 6's text, 150 pieces, no start mark), layer 20, neurons 4294 and 5316 each silenced, 12 and 32 rows, `folded: "always"`, the gate set aside (`bars.item_gate` -30): per-row equals folded at each row count; the 32-row figures beside 0.72581 and 0.52337 and 4294's 12-row figure beside pilot 7's 0.0085; whether the item reads `"unstable"` (expected) | 5's; recorded; recorded | pilots 6 and 7 |
| 5e | A rank-1 low-rank term on a `down_proj` that equals a one-column add, against that add (gap); seconds a reading with and without a rank-64 term | recorded | not measured: the pilots' low-rank terms were zeros |
| 6 | **A same-shape restart by bits, under hooks**: a reading with 60 control edits (so both row counts have later forwards) at layers 5, 20 and 44: `repeats_identical` is true at both row counts, and `lower: "whole"` gives the same figures exactly | exact | pilot 1 (unhooked, layers 20 to 58); hooked and layer 5 are new |
| 6b | The one-row capture against `same_shape`: at layers 44 and 40 (`bars.one_row_min_layer` 40), on two chat-formatted items, the item's hardest-firing set of 64 silenced: the gap between the two readings at each row count, beside the floor | recorded | pilot 3 C: layer 44, raw text; layer 40 and chat-formatted items are new |
| 7 | **Writing under hooks**: written readings of 8 rows and 48 pieces on two questions (doses `[1.0, 0.0]`, two control edits) are not void: untouched rows and the dose-0 row write the base's pieces with its log-probabilities; by the script's own write under `we.per_row`, two rows with one edit are the same and a hooked write's base row equals an unhooked write's | exact | pilot 3 A wrote under an output-side hook with no row stopping; the input-side pre-hook in a decode step and rows that stop are new |
| 7b | A step under hooks costs within 10% of a step without | 10% | pilot 3: 0.095 s both |
| 7c | Where a null row parts in writing and its largest log-probability gap | recorded | pilot 3: never in 48 pieces, 0.059, with a factor of 3 |
| 8 | **The item gate at the shipped bar** (one reading over these ten items, any edit): all six *own* items pass; the two cuts given as answers pass; the same two cuts raw, with no start mark, are refused; at both row counts | categorical | pilot 8: -0.07 to -0.17; -2.46, -5.51; -9.15, -10.09 |
| 8b | The gate's figure, at both row counts, for every other item (from §9's readings): the study chats, the other passages, the other raw texts, and every raw text behind the start mark | recorded | not measured |
| 9 | **The two row counts on every item, the gate set aside** (`bars.item_gate` -30): each item's own hardest-firing neuron and hardest-firing set of 64, silenced, at layers 5 and 20, `kl` and `own` on, `controls.count` 0, at 12 and 32 rows. On the six *own* items no arm disagrees on `kl` under `agree` at 3.0 | ratio 3.0 | pilot 8: largest 1.60, on such items, on KL over the span |
| 9a | By item, for all fifty: the gate's figures, the largest ratio on each read-out, `base_rows_same_bits`, whether any arm disagreed. The set of 64 at layer 20 again with the default 19 control edits: how many control arms disagree. On the six *own* items `odds` too (toward the answer's first piece, away its second, at `"span"`): the floor and the ratio in log-odds | recorded | not measured: `own` between row counts; items a person wrote, over the span (pilot 7 read every position: none over 5, 2 of 120 over 2); a control arm's second reading; any floor in log-odds |
| 9b | **The map**, by the script's own forwards: for a study chat, an *own* item and the recipe whole as an answer (about 30, 90 and 175 pieces), the sha256 of the base row's final states at 2, 3, 4, 6, 8, 12, 16, 24, 32 and 48 rows (which row counts share bits), and the item's hardest-firing set of 64 at layer 20 silenced at each (`we.per_row` with `[None, Arm, None, ...]`; mean KL over the span from `causal_check.logits_fp32`, as pilot 8's `span_lsm` and `klpos`). Entered in the book as one `note` | recorded | the ruling's open measurement 2; pilot 5 saw six base rows at six row counts on 91 pieces |
| 10 | **The cost model**: every paired reading above that was predicted at 1 s or more, those with a twin of more than 64 columns left out, reads `0.75 <= seconds.ratio <= 1.25` | 25% | the model stood within 15% on the pilots' nine shapes; short readings and large twins are recorded |
| 11 | Every default null factor differs from 1 and is no power of two in bf16; no null row has a position above a nat; the floors by item at layers 5, 20 and 44 | exact; exact; recorded | pilot 5; `RESULTS.md`, last paragraph |
| 12 | After everything: `assert_base` passes, `fell_back` has not moved, `lb.verify()` is whole; every class block's `peak_gib` with its rows and pieces | exact; recorded | the ruling, row 8: memory with the peak reset a shape |

**The JSON.** `verify_1a.json` holds the runtime record, every section's lines and figures, and a block

    "proposed_bars": {
      "item_gate": {"items": [{"name", "kind", "small", "large", "misbehaved": bool, "base_rows_same_bits": bool}],
                    "lowest_that_behaved": x, "highest_that_misbehaved": y or None, "separates": bool,
                    "proposed": z or None},
      "stable_ratio": {"largest_above_the_bar": x, "share_over_2": s, "control_arms_disagreeing": [k, n],
                       "proposed": z},
      "rows": {"small": n, "large": m, "share_bits": {<pieces>: [[rows that share bits], ...]}, "why": "..."},
      "one_row": {<layer>: {"gap_small", "gap_large", "floor"}},
      "cost": {"forward_fixed_s": a, "row_position_s": b, "readout_row_position_s": c, "mib_row_position": d}}

An item's figure is the lower of its two classes' means; it `misbehaved` when it read `"unstable"` in any reading
of §9. `separates` is true when every item that misbehaved sits under every item that did not; `proposed` is the
midpoint of x and y when it separates and some item misbehaved, else None: "the base's log-probability does not
separate" and "nothing misbehaved" are both findings. `stable_ratio.proposed` is the larger of 2 and one and a
half times the largest ratio seen, on any read-out, on items above the shipped bar. `cost` is a least-squares
refit, its `mib_row_position` from §12's peaks. **1a's provisional bars are set from this file by hand**: the
coordinating session reads it and changes `ITEM_GATE_BAR`, `STABLE_RATIO`, `ONE_ROW_MIN_LAYER`, `ROWS_SMALL`,
`ROWS_LARGE` and `COST` in `backend/levelling.py` in a reviewed commit, with the date and the file named in the
module docstring; if §5b read `same_bits` true in every case, that may become §5's standing line. Nothing reads
the JSON at run time.


## 7. Who builds what

| implementer | files | rough size |
|---|---|---|
| A | `backend/weight_edits.py` | about 700 lines |
| B | `backend/levelling.py` | about 950 lines |
| C | `backend/level_book.py`, `scripts/levelling.py`, `scripts/check_levelling.py`, `scripts/verify_levelling.py` | about 150, 300, 1,500 and 750 lines |

The checks come to about 1.07 times the code they check. The house's median is 1.45 (`verify/skeptic_code.md`
§4; the ruling, row 6), which would be some 3,000 lines. The sizes are guesses; the contract is section 6, line
by line, and a section that needs more lines gets them.

**The seams.** All three work at once, each against sections 2 and 3 as written.

- B calls A: `we.from_recipe`, `we.recipe_facts`, `we.edit_facts`, `we.twin_wanted`, `we.build`, `we.load`,
  `we.save`, `we.control_edits`, `we.exact_at`, `we.dose_refusal`, `we.null_factor`, `we.can_fold`, `we.Arm`,
  `we.Null`, `we.per_row`, `we.folded`, `we.hook_census`, `we.assert_base`, `we.MAX_DOSE`, the `Edit` fields and
  methods of 2.2, and the exceptions `we.Refused` and `we.NotBase`. A calls nothing of B's or C's.
- B calls C's ledger: `lb.head`, `lb.append`, `lb.canonical`, `lb.BookBroken`. C's ledger calls nothing of A's
  or B's.
- C's command line calls B: `lv.load_plan`, `lv.plan_items`, `lv.normalise`, `lv.estimate`, `lv.handle`,
  `lv.read`, `lv.lines`; and of A only `we.recipe_facts` and `we.assert_base`.
- C's checks call everything in section 3 and nothing private. The card check also runs forwards of its own
  under `we.per_row` (6.2 §7 and §9b).

**Until the others land**, B may write against a ten-line stand-in of `lb.append` and C against the schema of
2.7; each replaces it with the import before handing in. A needs no stand-in.

**Each module's docstring** says what the thing is and why, as `backend/causal_check.py`, `backend/steer_check.py`
and `backend/pattern_tests.py` do: the measured facts it rests on with their date (section 1's), each marked
measured or inferred, raw text said to be "with no start mark"; what is provisional; what is new on the card and
unmeasured (an input-side factor other than 0, the float32 output-side add, a low-rank term that is not zeros,
the pre-hook in a decode step); what it refuses and why; that layers count from 0. Refusals are plain words, a
whole sentence, saying what to do instead where there is something to do.

**Order of joining.** A's file and C's ledger are complete without the others. B joins them. Then the
coordinating session runs `scripts/check_levelling.py`, and `scripts/verify_levelling.py` on the card between
census layers.


## 8. Out of 1a

| left out | why | what brings it in |
|---|---|---|
| Direction removal | It must divide by the post-feed-forward norm's weight, which is a thousand times smaller on a few hundred dimensions at every layer from 13 to 58; the masked form (the direction projected off the layer's quiet mask, every control through the same projection) is written in the ruling and unbuilt | 1b: the masked form, its controls, and a card check against its activation twin |
| Row edits on `gate_proj` and `up_proj` | Their per-row form (the edited coordinate recomputed, never added to) has not been measured | 1b: the form and its folded twin through `_memit_row_overlays` |
| Attention edits | q, k and v pass per-head norms; on full-attention layers the values are the keys; no overlay or per-row term on an attention linear has been read on the 31B with a non-zero edit | its gate facts written and a card check passed |
| Embedding, norm and layer-scalar edits | No hook form; the writer takes its logits outside any module, so an output-side embedding edit would not be heard in writing | a hypothesis that needs one |
| A folded form for low-rank terms | Folding B @ A touches every column of the weight | a dense overlay path, measured |
| Matched control edits (on firing, on what an edit delivers) and sham-target controls | Matching needs a base pre-pass and a definition of `delivered`; sham targets need a declared rule for choosing sites | 1b, before any rank is given a word |
| Write ceilings and the share of entries that round away, in the edit's report | 1a reports instead the factor as applied and whether a row came out the base row by bits (`same_as_base`) | when an edit's size becomes something a gate must judge |
| Long sampled written readings with loop and cut-off detectors; writing in two batch classes | The writer under hooks was measured greedy, 48 pieces, 8 rows; thinking-on answers are uncosted | the ruling's ten-minute pilot of twelve thinking-on answers |
| A weights reader that needs no card | Nothing in 1a reads weights for a driver | the agent's door |
| Gradients | A backward pass under the compiled unpacker is unmeasured; the reach is the owner's decision 4 | pilot F and that decision |
| Anything an agent calls: the cell tool, the gates, the sandbox | 1a is the owner's attended session | the owner's decision 0 and the containment repairs |
| Briefs, fresh draws, frames kept back, any named standing | They are the claim gate; 1a claims nothing | phase 3 |
| Any page, any server route, the lock file shared with the server | 1a runs with the server stopped | the owner's order of phases |
| Any export | The licence repair comes first | phase 5 |
| A third batch class | The ruling overruled it until the map of where classes change exists; every paired result says that two row counts agreeing does not show a third would | the card check's §9b |
| Halving a reading on running out of memory | It changes the batch class | never |


## 9. Open points

Each is something only the card or the owner can settle. None blocks the build: the spec takes the
recommendation.

1. **The names.** "Levelling" and "level book" are the owner's decision 7, not yet made. *Smallest decision*:
   keep or rename before the first commit. *Recommendation*: build under these names; they sit in five file
   names, one environment variable and docstrings, and nothing a page shows.
2. **Where saved edits live.** The proposal said `atlas_data/edits/`; this spec puts them beside the book,
   outside the checkout. *Decision*: one of the two. *Recommendation*: beside the book. A payload can hold
   Gemma's columns; `atlas_data/` is ignored by git but is the parent of the scientist's work folder, and 1a may
   not add an ignore rule.
3. **The row counts of the two classes** (12 and 32). "12 or fewer; 16 or more" is the map at 150 pieces of raw
   text. At 91 pieces of chat-formatted text every row count tried gave its own base row (fact 4), so there 12
   and 32 are two arithmetics of many, and at some length they may be one. *Decision*: the defaults.
   *Recommendation*: 12 and 32 until §9b's map is read, then set by hand; every result already says what two
   row counts do not show.
4. **The item gate's bar** (-6.0) **and whether the base's log-probability is the right gate.** It is set by
   caution, between -5.51 (a passage given as the answer, which behaved) and the raw texts with no start mark,
   of which one misbehaved and one did not. *Decision*: the number, or no gate, after §8 and §9 have read fifty
   items from -0.1 to -10 nats with the gate set aside. *Recommendation*: ship -6.0; if the card finds that the
   figure does not separate, the gate stays as a printed caution and the two row counts carry the weight.
5. **The stability number** (3.0) **and its rule** (any edit or control arm, on any read-out that is on). In use
   an item carries 19 control arms near the floor, each a chance to pass a ratio of 3; no pilot kept a control
   arm's second reading. *Decision*: the number and the rule, after §9a. *Recommendation*: ship as written; if
   §9a shows control arms at the floor tripping it, read the rule on the edit's arms alone and print the control
   arms' ratios.
6. **The output-side add's arithmetic.** This spec sums in float32 and rounds once (the house's
   `with_overlay_terms`); the pilots' hooks multiplied and summed in bf16; the ruling marks the float32 form
   unmeasured. *Decision*: which one 1a builds. *Recommendation*: float32, one rounding where the other has K + 1,
   with §5c reading its gap to folded before anything leans on a set or an add.
7. **Low-rank terms in 1a at all.** They have no folded form, a control family by permutation that nobody has
   measured, an unmeasured non-zero effect, and on `gate_proj` and `up_proj` they are the added form of a row
   change, which the ruling struck for row edits. *Decision*: build as specified; or on `down_proj` only; or
   refuse `lowrank` in 1a. *Recommendation*: build as the builder's decision B says, marked in every report, with
   §3 and §5e reading it on the card.
8. **Whether the card check's readings belong in the real book.** A run adds some 400 entries. *Recommendation*:
   yes, named `verify: ...`: every call is in the book. If the owner would keep the real book for his own
   readings, the script points `THEODOLITE_LEVEL_BOOK` at a book beside `verify_1a.json`: one line.
9. **Hooks at load.** Nobody has loaded the model to see; no file read registers a hook that stays, and the
   stand-ins carry none. `assert_base` compares with a census of hook ids taken at load, so a hook of the
   runtime's own is not taken for a leak. *Decision*: none unless §1 records any.
10. **The compiled unpacker's cache in `/tmp`** (the ruling's first standing fault). An uncommitted change in
    this checkout, another agent's, moves the cache inside `runtime_factory.make_runtime()`
    (`private_compile_cache`). 1a builds its runtime through `make_runtime`, so it takes that repair with no code
    of its own, and must not set `TORCHINDUCTOR_CACHE_DIR` itself. *Decision*: none, unless that change does not
    land; then settle it before any agent's door.
11. **`RESULTS.md` says the pilots' raw text had a start mark; it had none** (section 1). This spec may edit no
    other file. *Decision*: the coordinating session's. *Recommendation*: one dated line there, and "raw text
    with no start mark" wherever the -9.6 nats is quoted. Raw text behind a start mark is unmeasured; §8b reads
    it.
12. **Per-item rows, and now the items' text and ids, in the book.** Right for the owner's session and for
    replay; the ruling keeps them out of any file a driver's shell can read. *Decision*: none in 1a. It must be
    reopened before the agent's door.
13. **The overlay dicts' names** (`_memit_col_overlays`, `_memit_row_overlays`) are read from a pinned file 1a
    may not edit. If the R4 pin's repair ever renames them, `folded`, `assert_base` and `overlays_in_force`
    follow. *Decision*: none now.
14. **The one-row capture's lowest layer.** The builder's decision E says layer 40 and above; the ruling says
    where shown, which is layer 44, on raw text. This spec ships 44 as a bar a plan may lower (recorded as
    overridden). *Decision*: the constant, after §6b has read layers 40 and 44 on chat-formatted items.
    *Recommendation*: 44 until then.
15. **Two guards are inferred, not measured**: 22 GiB of the card free before the load (from the 21.2 GiB of
    model and lens) and 6 GiB above the model for a written reading (0.8 MB a row a piece, measured with the
    lens loaded, against 8.37 GiB free). *Recommendation*: ship both; a written reading's `peak_gib` re-measures
    the second.
16. **The default read-out.** Decision E makes named-piece log-odds the default and the full-vocabulary pair on
    request. A plan that names no pieces has no log-odds to read, so it reads KL and the text's own
    log-probability. *Decision*: whether that fallback is wanted, or such a plan should be refused.
    *Recommendation*: the fallback.
17. **An item whose two row counts ran one arithmetic** (the same base row by bits) was read once. This spec
    flags and counts it and keeps its figures in the summary. *Decision*: whether it should be left out, as an
    unstable item is. *Recommendation*: keep it in until §9b says how common it is.
18. **The float32 stand-in at one thread** writes alike row for row on this machine (MKL 2024.2); elsewhere a
    dry run's written reading may be void. *Decision*: none now: bits are claimed only on the packed stand-in.


## 10. Where this spec departs, and where the documents disagreed

**Following the pilots over the proposal** (the ruling says the same): column edits ride a row on the input
side, not as an output-side term; there is no constant floor and no "five times the floor"; items are in chat
format and gated; every paired reading is taken in two batch classes; the median and the count above a nat stand
beside the mean. **Following the ruling over the proposal**: several null rows in place of the times-3 row; a
reading that runs out of memory is void, not halved; the one-row capture only where it was shown; a cap on what
rides a row (dose times relative change); "control edit"; no verdict.

**Departures from the builder's decisions, each small:**

1. *E, "every forward carries ... the edit at its doses".* Every forward carries the base row, its twin and
   every null row; the edit's doses sit in forward 0 only, and control edits fill the rest. Repeating two doses
   in every forward of a 12-row batch would leave four slots for control edits and five forwards where four
   serve; the null rows, repeated, already show by bits that the forwards are the same.
2. *E, "the base row is never remembered across forwards".* The folded twin reads against forward 0's base row,
   a forward of the same shape, and says so: the folded form has no untouched row. The small class's base row
   is also kept until the large class begins, only to see whether the two are the same bits; no arm is read
   against it.
3. *E, null rows.* A factor must be a value of the model's dtype other than 1 and not a power of two (the
   record's last paragraph); at least two rows, one each side of 1; the default is four, ±1/32 and ±1/8.
4. *E, "a one-row capture ... only at layer 40 and above".* Shipped from layer 44 (open point 14).
5. *E, the default read-outs.* Named pieces are the default when the plan names any; a plan that names none
   reads KL and the text's own log-probability, at 0.11 ms a scored row-position on every arm (open point 16).
6. *E, "any reading is taken in two batch classes".* A written reading is taken at one row count (decision F
   measured 8 rows, greedy) and says so.
7. *C, set and add.* The output-side add is summed in float32 and rounded once (open point 6).
8. *B and E, control edits of a low-rank term* are the term with its neuron axis permuted: a reading of "other
   sites of the same layer and size" for a term that names no site.
9. *G.* A refusal is a result that `read` returns and enters, not an exception, so that every call is in the
   book and the command line can go on to the next reading; any other exception is entered and raised again.
10. *The proposal's zero row* is carried only when the plan lists a dose of 0, and must then be the base row by
    bits or the reading is void. Otherwise the untouched twin stands in every forward, its slot goes to a control
    edit, and the zero row is shown by the checks (6.1 §4a, 6.2 §3). The ruling did not strike it; this is the
    spec's own.
11. *The proposal's `atlas_data/edits/`* becomes a folder beside the book (open point 2); its `.npz` or
    safetensors payload is safetensors, and a scale-only edit has none.
12. *KL's direction* is KL(base || row), as the proposal says; `causal_check.kl` is the other way round and is
    not used.
13. *Batch rows.* Each class has one fixed row count, small 5 to 12 and large 16 to 64; the ruling's "12 or
    fewer" allowed 2.
14. *I, the status words.* An item read in one arithmetic is flagged and counted, not given a fifth word (open
    point 17).


## 11. Objections not taken

Two critics read the first draft, one against the code ("code N") and one against the ruling and the pilots
("ruling XN"). Every objection not listed here was taken as asked, or by one of the fixes its critic offered.

- *Ruling M2, an item read in one arithmetic "is not ok".* A flag, a printed line and a count, not a fifth
  status word (decision I; open point 17).
- *Ruling W1, "a dozen or more passages by a person".* The repository's own texts are used (the pilots' passages,
  the study's prose and chats): a passage an agent writes for the check is not a person's.
- *Ruling M9, "lowrank on down_proj only".* Decision B builds it on all three linears; the other fix was taken
  (the report's mark, card lines §3 and §5e; open point 7).
- *Ruling M3 (i), "per_row refuses an eaten dose".* `read` refuses it first, by `we.dose_refusal`; `per_row`
  carries dose-0 arms and control edits too, and `same_as_base` shows what slips through.
- *Ruling M4, "four null rows is thin".* Two are required and four stay the default (each costs an arm slot in
  every forward); the count is printed and `null_e` takes eight.
- *Ruling m9, "nineteen control edits cost three more forwards".* 19 stays the default (the least for a rank of
  1 in 20); how many read above the floor is printed.
- *Ruling m5, "plan 2,600 lines of checks".* The ratio is stated instead (section 7).
- *Ruling W2 (iv), "make same_bits the standing line".* By hand, once the card has read it (6.2).
- *Code 13, "compare at the edited down_proj's own output".* The other fix was taken (6.1 §5a): a bound there
  needs the bf16 kernel's order of summation, which nothing pins.
- *Code 2, the dry run.* `--stand-in` stays the float32 stand-in at one thread, the census runner's own; bits
  are claimed only on the packed one (open point 18).
- *Code 1, "one dated line in `RESULTS.md`".* This spec may edit no other file (open point 11).


## 12. Found in the build

Written 14 November 2026 by the integrating agent, after the four builders' files were joined on the CPU. The card
was not touched and the 31B was not loaded. Nothing above this section was changed: where it and this section
differ, this section says what the code does.

Three decisions of the coordinating session amend the spec and are in the code: (1) a low-rank term is taken on
`down_proj` only; (2) the card check keeps a ledger of its own beside its JSON and never writes the owner's;
(3) every other open point of section 9 follows its recommendation.

**Where the spec was wrong, or two of its passages disagreed, and what the build does**

1. *Low-rank terms on `gate_proj` and `up_proj`* (2.2's fourth refusal and its `form`; 3.1; 4.1; 4.9; 6.1 §5b;
   6.2 §3; open point 7; section 11, "Ruling M9"). Decision 1 overrules them. The fourth row of 2.2's table now
   reads `Only the feed-forward linears take edits in 1a: layers.<L>.mlp.down_proj for a column edit, and down_proj
   for a low-rank term. "{name}" is not one. The embedding, the norms and the layer scalars are not built.` A new
   refusal sits in the order of precedence just after "cols on `gate_proj` or `up_proj`": `A low-rank term on {name}
   is not built in 1a: on {which} it would be the added form of a change to what neurons read, not the recomputed
   form asked of a row edit, and that waits for 1b. In 1a a low-rank term names a down_proj.` `lowrank(...,
   which=...)` still writes such a term down; `build` refuses it. 6.1 §5b and 6.2 §3 read the low-rank form on
   `down_proj` alone, and the CPU check compares both sentences word for word.
2. *3.2, how `append` opens the file.* `os.O_APPEND | os.O_CREAT` is write-only, and the last line is read from the
   same descriptor under the lock. The file is opened `os.O_RDWR | os.O_APPEND | os.O_CREAT`, mode 0o600.
3. *3.2, `verify`'s five rules, against 6.1 §12.* A byte changed where the JSON's meaning is not (an exponent's `e`
   written `E`) passes all five, so "a byte changed in a middle line is named by its line" would not stand for
   every byte. `verify`, `head` and `append` also ask that a line's bytes are `canonical(entry) + b"\n"`.
4. *2.2, pins "cached by module for the life of the process".* The reason given ("nothing in the process writes
   packed words") covers packed linears only. A plain linear's weight can be written in place, and 6.1 §7 does so:
   a plain linear's pin is worked at each call, a packed one's is cached.
5. *2.2's table against 2.3's prose, the order of `load`'s refusals.* The table says its order is the order of
   precedence (base, pins, arrays, id, format); 2.3 says arrays, id, then base. The table's is taken; "not this
   format" comes first only when the manifest cannot be parsed at all. A loaded edit then passes the gate as a
   built one does (a control edit without the size caps), and its id must come out the same.
6. *2.6 against 4.7 step 2.* `read` normalises before the edit is built, but two of `normalise`'s refusals need
   the edit's facts (`lower: "one_row"` below the bar; `folded: "always"` with a low-rank term). `read` makes them
   after the build, from `we.edit_facts`, in the same words, so an edit the gate refuses is told the gate's words
   first. `normalise` makes them itself when it is handed `facts`, and then also refuses a layer the model lacks.
7. *4.11's read-out count against 4.7's order.* As 4.7 orders it the code reads the base row of forward 0 once
   more a class (the gate's figure is taken before the class is read) and, with a twin, forward 0's base row and
   the per-row arm once more each. The estimate keeps 4.11's formula and so runs a little under (one row in 29 on
   the default small class) [inferred from the code, not timed]; the card check's refit takes it up.
8. *Section 1's kept-out words against 6.2.* The pilots' passages, copied into the card check as 6.2 asks, hold
   three of them. 6.1 §15's "four new source files" is read as the three modules and `scripts/levelling.py`; the
   two check scripts hold those words only in the copied passages and in the list itself.
9. *6.2 §11 against §9.* §11 asks the floors at layer 44, which §9 does not read. The card check adds one reading
   an item there: fifty more, 363 readings and one note a run.
10. *6.2's "into the book" against decision 2* (open point 8). The card check's ledger is
    `experiments/results/levelling/verify_1a_book.jsonl`, pointed to before the ledger is imported. It lies inside
    the checkout and no ignore rule covers it or the JSON beside it; it carries the fifty items' text and ids and
    every per-item row (7 MB in a CPU dry run). Whether it is committed is the coordinating session's to say.

**Where the spec was silent, and the reading the build took** (each is a line or two to change)

- `Edit.report` is `{"terms": [one entry a term, in the terms' order]}`, with `"caps"` beside it on a control
  edit. A result's per-term entries therefore sit at `edit.terms.terms`.
- `bars_overridden` is true when the bars a reading ran under differ from the module's own as they stand. A plan
  as given and the same plan already normalised then read alike; a plan that gives a bar equal to the module's
  reads false.
- An item that was being read when the reading came out void has `status: "void"` in `items` (2.7 lists three
  statuses). A void's `forward` is `"folded"` in the twin's forward; its `class` and `forward` are None in a
  written reading.
- The summary: with no item kept, `classes` is `{"small": None, "large": None}`; `one_arithmetic` counts every
  item whose base rows were the same bits, whatever its status; a median over items is the lower middle, as
  `stats` takes its own; a class's `floor` holds one number a read-out.
- `estimate`: `forwards` counts every teacher-forced forward (the twin's and a one-row capture's among them) and
  `row_positions` their rows times pieces; with `kl` and `own` off the read-out is `0.00011 * S` a class, twice an
  item. It has no model, so it prices a reading whose edit only the gate can refuse (a neuron out of range, a
  low-rank term on `gate_proj`); `run` then refuses that reading in the gate's words.
- `hook_census` names the text model itself `""` and lists the hooks torch keeps for all modules at once under
  `"(every module)"`; a refusal names them in words. `overlays_in_force` gives one pair a module.
- `save` refuses a folder inside the checkout. A recipe's file larger than `MAX_PAYLOAD_BYTES` is refused unread.
  An edit with no term is refused (`An edit with no term changes nothing.`). `MAX_TERMS` counts terms after merging.
- An `Edit` handed to `read` must be of the loaded base (the model's id, the weights' id, each pin) or it is
  refused in `load`'s two sentences.
- The command line: step 1's two refusals are not made under `--stand-in` (their own sentence offers it);
  `estimate` exits 1 when a reading's plan is refused and 0 otherwise; `book --seq` of a line the book has not
  exits 2; the book's line number is printed on a line of its own after `lv.lines`; `THEODOLITE_DEVICE=cuda` is
  read as `cuda:0`, as `runtime_factory` reads it.
- A dtype in a refusal is torch's name without `torch.` (`bfloat16`, `float32`); a count of row-positions is
  written with its comma (`12,800`).
- A written reading's `seconds_a_step` is the mean time between steps and None when one step was written; `text`
  leaves the end mark off and `pieces` keeps it; `logprob_gap` over no pieces is None.

**What the join ran**, all on the CPU with the card hidden (none of it is the card's arithmetic):
`scripts/check_levelling.py`, every line passing in about half a minute; the command line's `estimate`,
`run --stand-in` and `book` as section 5 describes, the book in a temporary folder, with exit codes 0, 1 and 2 as
5.4 gives them; `estimate` on the worked plan of 5.2 with the checkpoint's tokenizer alone (10 forwards, no twin,
as 5.2 says; the real chat template gives one start mark, first); a scratch probe of what 6.1 does not list (the
voids `non_finite_base`, `repeat_differs`, `interrupted`, `out_of_memory` and `not_base_after`; an arm that is not
finite; a saved control edit loaded again). `scripts/verify_levelling.py` was imported with the card hidden, its
calls on the modules bound against their signatures, and its code after the load run on a tall packed stand-in.
It has not run on the card: its guards, the load, the timings and the memory peaks have never executed.

### Found at the build's review

Written 14 November 2026 by the mending agent, after three reviewers attacked the joined build (the arithmetic and
the hooks; the measurement; the words, the record and safety). Everything ran on the CPU with the card hidden; the 31B
was not loaded and `scripts/verify_levelling.py` has still not run. Nothing above this heading was changed: where it and
this part differ, this part says what the code does. The reviewers' notes and the mender's are beside the build's
scratch notes (`review_arithmetic.md`, `review_measurement.md`, `review_words_safety.md`, `mend.md`).

**Sentences of sections 2 and 4 that changed** (the module, the CPU check's copy and this list were changed together)

1. *The pilots' raw text had no start mark, and three sentences cited it without saying so* (open point 11's
   recommendation, which decision 3 takes). 2.6, `rows` out of range: `... (the ruling's two classes: at 150 pieces of
   raw text with no start mark this card ran one arithmetic up to 12 rows and another from 16, 14 November 2026); ...`.
   2.6, `lower: "one_row"` below `m`: `... has been shown only at layer 44 (page layer 45), on raw text with no start
   mark, on 14 November 2026; ...`. 4.10, the gate: `... under the bar of {bar:.2f}. On 14 November 2026 one of two raw
   texts with no start mark, which the base gave -9.6 nats a piece, gave readings that moved eighty-fold with the batch
   size, so an item under the bar is not read.` ("about that level" was printed for items at -6.4.)
2. *4.7 step 3, why no twin is taken.* `a silencing or a scale by a power of two is exact in arithmetic on the input
   side; on the card it was read beside the folded form for a silencing only (14 November 2026)`. The card has read a
   factor of 0 alone (fact 2); the sentence said the rest as a fact. 5.3's mark on an arm that uses the output-side add
   is now `(output-side add, summed in float32: not yet measured on the card)`.
3. *4.1, the hooks' `RuntimeError`.* A wrong row count keeps `hooks made for {R} rows met {n}`. A tensor that is not
   `[rows, pieces, width]` is said by its shape (`hooks made for {R} rows of [rows, pieces, width] met a tensor of shape
   {shape} on {name}`): the old sentence could read "made for 2 rows met 2".

**New refusals** (none is in 2.2's or 2.6's tables)

- `we.dose_refusal`, before its other two: `This edit scales every column it names by exactly 1 and adds nothing: at
  any dose its row is the base row again, so there is nothing to read.` (a scale by 1, or by a number that is 1 in the
  float32 a factor is kept in).
- `lv.make_item`: an answer that holds any of the tokenizer's special pieces (a start mark, a turn mark) is refused:
  `The answer holds {k} piece(s) of the chat format or the tokenizer's own marks ({text!r}): an answer is plain text, and
  its pieces are what is scored.` 2.5 held only the prompt to one start mark. `item_from_ids` still asks nothing.
- 2.4: a term's file whose path is absolute or climbs out of the plan file's folder: `A term's file is named from the
  plan file's folder and lies inside it; {path} does not.` Links are not followed. `{"saved": folder}` is as it was.
- A recipe's number past what a float holds, and a file name the system will not take, are `Refused` in words (they
  raised `OverflowError` and `ValueError` out of the command line); a plan file nested too deep to parse is `Refused`.
- `scripts/levelling.py run`: `--out` naming the book or the plan file is refused before the load, exit 2.

**The gate, met again** (3.1 gains one call; 3.3's `edit=` changes)

- `we.gate_again(lm, edit, *, model_id, weights_id) -> Edit`: an `Edit` that was not built here and now is taken apart
  into recipe terms, built again against `lm`, and refused when the id that comes out is not the id it carries (`The
  edit handed in says it is {id12} and hashes to {other12}.`). `lv.read(..., edit=)` calls it after comparing the base,
  so the book never holds an id that is not the hash of what ran; what is no `Edit` is a refused result, entered.
- A control edit met again (loaded, or handed in) is let past ONE cap, a column's change over its own length. The mark
  that asks for this sits in a saved edit's report, which is not hashed: with it, a hand-written manifest switched every
  size cap off (a rank-65 edit loaded and was read). Its counts, rank, bound and bytes are its edit's by construction.
- `-0.0` in a factor, a vector, `A` or `B` is kept as `0.0`: a scale by `-0.0` had another id than the silencing.

**What a result holds that 2.7 does not list** (additions only; no key of 2.7 was renamed or removed)

- `runtime.numpy` (control edits are drawn with numpy's `Generator`, whose stream numpy does not promise across
  versions) and `runtime.start_mark` (the tokenizer's start mark, so that an entry says of each item whether it began
  with one: the item block is as 2.7 has it, and `lines` works it out).
- A floor's `of` is the null rows that gave a figure: one that came out the base row by bits, or not finite, is not a
  draw. With none, the floor is `{"mean_abs": None, ..., "of": 0}`.
- A folded twin whose rows are not finite is `taken` with `metrics`, `gap` and `rows_identical` None, and is not a void
  (4.5: "an edit can break the model"). It was a void, `untouched_rows_differ`, since a NaN never equals its twin.
- An array a Python caller writes into a recipe is recorded as `<ndarray float32 [1, 5376]>`, never as its numbers
  (the reading was refused, and the book still took the array).
- `FLOOR_SAY`, printed once by `lines` beside the three sentences of 4.9: `The floor is the largest of {n0} null rows,
  not a bound: on 14 November 2026 a single neuron silenced at random read above its forward's floor in 26 of 72
  readings, up to 2.25 times it.` [measured, pilot 8: `pilot8_first_look.json`, the model's own answers, layers 19 to
  54; recounted by the mender]. It is not a key of the result.

**What `lines` prints differently** (5.3): how many items were kept, on the first line; no figure of an item read
before a void, nor of the row count read before the gate refused an item at the other; no `CONTROLS_SAY` and no
"ranks by" with no control edit; "unmatched" in every rank line; an item's id beside its name, and "NO start mark at
its head" where that is so; one figure, not a mean and a median, for log-odds read at one position; three figures for
anything under a thousandth; a scale's factor, and a factor as applied beside the one asked where the cast moves it;
an unstable item's reason in words; "by construction" where the two row counts ran one arithmetic; and, where an
item's two floors stand apart by more than the ratio bar, a line saying the floor itself moved. `book --tail` gains a
column: items kept of asked.

**The ledger** (3.2): `verify`, `head` and `append` also ask that a line's `v` is 1 and its `kind` one of `KINDS`;
`verify` of a path that cannot be read says so and does not raise. The docstring no longer says the chain shows lines
left out: lines left off the END, or the last line rewritten and rehashed, leave a book that verifies (the head a run
prints is the only record of where the end was). A book that exists keeps its mode. 5.1's "a dry run never writes the
real book" is true of the default book only, and the words now say so.

**The CPU check** (6.1): 854 lines where the join left 737, none loosened. Its new lines, and the break each was
shown to catch on a scratch copy, are in the mender's notes. 6.1 §9's `atol=1e-5` stays for the first reading and is
joined by a second, under an unembedding 200 times as sharp, compared to 1 part in 10,000: on the first, the edit reads
1e-4 nats and a null row 1e-7, so KL the wrong way round, no softcap and logits in the model's dtype all passed.
6.1 §15's pattern also sees what is made of a kept-out word ("stations", "sealed", "errata").

**The card check** (6.2), read again against the modules; none of it has run on the card:

- §5's two tolerances are two lines, so that a failure says which: 1e-5 nats is measured (pilot 6, five decimals); 1
  part in 10,000 asks nearly the same bits at the size read here, which no pilot showed.
- §5d's two recorded lines for neuron 4294 are now standing lines: its 32-row figure within 0.001 of 0.72581 (pilots 6
  and 7, two processes), and the item `"unstable"`. It is the stability rule's one known positive, and nothing failed
  if the rule missed it. Neuron 5316 stays recorded (pilot 6 alone read it).
- §11's "no null row above a nat" is two standing lines: items over the shipped bar, and items under it.
- A fifty-first item: the model's first own answer with the start mark dropped from its chat prompt (six more
  readings). Recorded only; it enters `proposed_bars.item_gate` by its kind.
- Recorded besides: null rows by `e` (§11); each reading's ratio of its two floors, and how many stand apart over the
  bar (`proposed_bars.stable_ratio`); beside `separates`, how many items behaved and misbehaved and the words for
  which case it is (it is true when nothing misbehaved); beside `stable_ratio.proposed`, the same from the readings that
  agreed, and how many did not.

**Not changed, and why** (each is the coordinating session's or the owner's)

- *The stability rule* (4.9; open point 5). An arm's two readings are set against the larger of the two row counts'
  floors, and the floors are not set side by side: a null row that moves with the row count hides an edit that moved
  with it (planted on the CPU: a ratio of 153 reads 1.04, item "ok"). The lines now say when the floors part; the rule
  stands, because how far two floors part on the card is unmeasured and a rule on it could mark sound items. §5d's
  new line is where the card would show it.
- *Two sentences of 2.2's table name a layer by the code's count alone* (`A column of layer {L}'s down_proj ...`,
  `Scaling every column of layer {L}'s down_proj ...`), against this spec's own line 18. Left: they are a seam. The
  builder's own sentences beside them now say both counts.
- *6.2 §10's 25% on every reading* rests on the forward's timing alone; loosening it would weaken a standing line.
- *The card check's ledger and JSON inside the checkout with no ignore rule* (item 10 above); *an `edit=` reading
  cannot be rebuilt from the book unless `save_edit` was passed* (3.3's design); *a recipe's files are read before
  their sizes are added up* (one file over the cap is refused unread); *`handle()` runs `git status`, which may
  refresh the index* (`census.runtime_record`, another file's).

### The card check (16 November 2026), and what was set from it

Written 15 November 2026 (the desk's clock; the card ran 16 November 2026 01:33 to 01:46Z) by the agent that set the
bars, from the record alone: `experiments/results/levelling/verify_1a.json` and its ledger `verify_1a_book.jsonl`.
The card was not touched again and no server was started; the only runs were the CPU check and scratch reads of the
record. Nothing above this heading was changed. The full account of the run is the dated section at the end of
`pilot/RESULTS.md`.

**What ran.** `scripts/verify_levelling.py`, one run: a 16 s load, then 369 readings through `lv.read` (367 paired, 2
written) in 793 s, into the check's own ledger (370 entries, the chain whole). The three modules' sha256 in the
record are the files committed at `96e4420`. 142 of 145 standing lines passed. The three that failed:

1. *6.2 §10, the cost model:* 21 of 163 readings outside 0.75 to 1.25 of their prediction (median 1.12, worst 1.44),
   every one of the 21 a reading with 19 control edits (21 of 40 such; the other 123 all inside, median 1.11).
2. and 3. *6.2 §11, no null row with a position above a nat,* on items over the shipped bar (12 null-row readings
   had one) and under it (16). All 28 are on raw text: 12 distinct rows on three items, the raw recipe cut and the
   raw survey behind the start mark and the raw recipe with no start mark. None of the 2,208 null rows read on the
   model's own answers (640), the study's chats (608), passages given as answers (912) or the own answer with its
   start mark dropped (48) had one. Counted on KL alone the rows are 17, as the coordinating session counted them; on
   the text's own log-probability, 28.

**Each bar, by the rule written for it here before the card ran** (section 9; the coordinating session decided on 14
November that every open point follows its recommendation). Where no rule exists the bar was left as it was.

| bar | rule | what the card read | set |
|---|---|---|---|
| `ITEM_GATE_BAR` and the gate | open point 4: "ship -6.0; if the card finds that the figure does not separate, the gate stays as a printed caution and the two row counts carry the weight" | `proposed_bars.item_gate.separates` false: the 3 items that read "unstable" in §9's four readings with no control edits stand at -3.09 (chat 10), -4.13 (raw prose 2 behind the start mark) and -9.64 (the raw recipe with no start mark); the 48 that did not reach down to -12.13; 19 items under -6 behaved. With 19 control edits (§9a) 3 more read "unstable" from control arms alone, all under the bar (-8.61, -8.82, -10.08): still no separation | -6.0, a printed caution, not a refusal (below) |
| `STABLE_RATIO` | open point 5: "ship as written; if §9a shows control arms at the floor tripping it, read the rule on the edit's arms alone and print the control arms' ratios" | over the bar, the largest ratio that agreed 2.44 (a read-out that agreed in a reading whose other read-out disagreed; 2.07 among readings that agreed on both); 3 readings disagreed (5.2, 20.7, 6.3); 6 of 969 control arms disagreed, all on items under the bar, two of them at the floor at one row count | 3.0. The second clause's change is a change of the rule, not of a bar: left to the coordinating session (below) |
| `ONE_ROW_MIN_LAYER` | open point 14: "44 until then", the decision to be taken after §6b; no rule for another value | §6b: the one-row capture stood off the same-shape reading by 10 to 35 times the floor in 7 of 8 class readings at layers 40 and 44 on two chat-formatted items (up to 0.0066 nats of KL, 2.5% of the reading) | 44; the module's words about it corrected |
| `ROWS_SMALL`, `ROWS_LARGE` | open point 3: "12 and 32 until §9b's map is read, then set by hand"; no rule for other numbers | §9b: at 35 pieces all ten row counts gave their own base row; at 82, 12 and 24 were one; at 175, 24, 32 and 48. 12 and 32 never one | 12 and 32 |
| `COST` | 6.2: "`cost` is a least-squares refit, its `mib_row_position` from §12's peaks", the bars "set from this file by hand" | `proposed_bars.cost`: 0.13502 s, 0.000376947 s, 0.0000880039 s, 0.19113 MiB (331 readings) | 0.135, 0.000377, 0.000088 and 0.191, to three figures; `decode_step_s`, `overlay_matrix_s`, `load_s`, `unpacker_s` and `mb_row_piece_written` are not in the refit and stay |

6.2's formula for `stable_ratio.proposed` (the larger of 2 and 1.5 times the largest ratio seen over the bar) gives
31.09, from the raw prose 2 behind the start mark, a reading that disagreed; from the ratios that agreed it gives
3.66 (the record's `proposed_from_those_that_agreed`, which counts a read-out that agreed in a reading whose other
read-out did not), and from the readings that agreed on both read-outs 3.10. None was taken: open point 5 is the rule
the coordinating session adopted, and a bar set from the readings it judges would let through what it is there to
catch.

**The gate as a caution, in the code** (`backend/levelling.py`; 2.7, 4.7 b and d, 4.10 and 5.3 as they now read):

- An item is read in both classes whatever its gate's figure. 4.7 b's early refusal and d's refusal are gone; no
  item's `status` is "refused" any more (2.7's third item status no longer occurs; a reading's own "refused" stands).
- The item's `gate` block gains `"caution"`: None when both classes' mean is at or over the bar (`passed` true), else
  the words `The base model gives this item's answer {small:.2f} at {R_small} rows and {large:.2f} at {R_large}, in
  nats a piece on average: under the item gate's bar of {bar:.2f} at {both row counts | R rows}. The bar is a
  caution, not a refusal: on 16 November 2026 the card read 51 items with it set aside, and the base's
  log-probability did not separate the 3 whose two row counts disagreed (-3.09, -4.13 and -9.64 nats a piece) from
  the 48 that agreed (the lowest -12.13). The item is read, and its two row counts carry the weight.` (the second
  half is `GATE_SAY`). 4.10's refusal sentence is retired with it. An item under the bar that reads "ok" is kept in
  the summary like any other.
- The summary's `items` gains `"under_bar"`, the items read with the caution; `refused` stays (always 0 now) so that
  older results keep their shape.
- `lines` prints the caution under the item's line, and ", N under the item gate's bar and read with its caution" in
  the summary's line when N is not 0. The branch that printed an item refused after its small class was read stays
  for results written before this change.

**The words corrected** where the record contradicts what a docstring said was measured or unmeasured, each dated
16 November 2026: in `levelling`, the one-row capture (no longer "shown at layer 44" for chat-formatted items), the
restart under hooks (now shown at layers 5, 20 and 44), the base's figures by kind, the cost paragraph, the bars
paragraph, what was new on the card and is now read there, the stability rule's floors (22 of 204 readings over the
bar apart, at most 42.7 times) and what has run on the card; the constants' comments. In `weight_edits`, the
paragraph of what was not measured on the card and `per_row`'s dose-0 sentence. In `scripts/levelling.py`, the cost
figures and the gate's words. The CPU check's lines that asserted the old cost figures and the gate's refusal now
assert the new ones (section 8's `COST`, section 13's arithmetic, section 10's gate lines and the result's shape);
none was loosened, and the gate's lines gained what the caution must say. It now has 914 lines (854 before), every one
passing on the CPU on 15 November 2026 (the desk's clock), in 45 s.

**Not changed, and whose it is**

1. *Open point 5's second clause.* Two control arms tripped the rule from the floor (control 14 on prose 10 as an
   answer read 0.00050 nats of KL at 12 rows and 0.0018 at 32, against the larger floor, 0.00054; control 15 on the
   raw prose 3 with no start mark read -0.091 and 0.013 of its own log-probability against 0.0285), both on items
   under the bar. Whether
   that is "control arms at the floor tripping it", and so whether the rule moves to the edit's arms alone, is the
   coordinating session's: it changes what the rule does, which setting a bar may not.
2. *`scripts/verify_levelling.py` §8's third line* asserts that the two raw cuts with no start mark are refused; with
   the gate a caution they are read, so a second card run fails that line until it asserts the caution. Its §11 words
   ("which the gate would not have read") are stale too. That file was not to be touched here.
3. *Printed words in `levelling` that the card made stale and that this pass might not change* (each is a sentence a
   result carries, and the CPU check compares it word for word): `WHY_NO_TWIN["exact"]` says a power-of-two scale was
   read beside its folded form "for a silencing only (14 November 2026)", and §5a read a scale by 2 the same bits 24
   times; the one-row refusal says the capture "has been shown only at layer 44 ... on raw text", true, and silent on
   §6b; the output-side add's mark says "not yet measured on the card", and §5c measured it. In `weight_edits` the
   comment on `MAX_RANK` says the float32 form "is not timed"; §5e timed rank 64.
4. *The stability rule's blind spot, now seen on the card.* On the raw recipe cut and the raw survey behind the start
   mark the null rows moved 15.7 to 42.7 times between the row counts, and both items read "ok". Whether two floors
   apart should make an item unstable is the owner's (section 12's review, "Not changed").
5. *Raw text behind the start mark.* Every null row with a position above a nat was on raw text, 26 of the 28 behind
   the start mark; one such item (-4.82) stands over the bar. What the instrument should do with raw text is the
   owner's; with the gate a caution nothing refuses it.
6. *Figures the card read that no rule sets:* a decode step 0.110 s hooked and 0.107 unhooked (0.095 shipped); the
   load 16.0 s with the lens loaded and freed and the install 3.46 s (9.5 and 1.83 shipped, no lens); a written
   reading of 8 rows to 74 pieces 0.73 GiB above the model where 0.8 MB a row a piece gives 0.44. They are beside the
   shipped figures in the module's docstring.
7. *The two row counts.* The map makes "two batch classes" two of many arithmetics; whether 12 and 32 are the two to
   read, and whether `lower: "one_row"` should be offered at all on chat-formatted items, are the owner's.
8. *The refit is in sample.* It was fit on the readings it would be judged on; the next card run is its test.
