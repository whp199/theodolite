# The atlas and census as documentation of the weights

Reader's report for the weight-space harness design. Read-only pass over `(the repository root)` at commit
`e6ac64f` (14 November 2026, clean tree). Layers count from 0 everywhere below (a page shows layer + 1).

How to read the citations: `path:line` points at code or a document I read. Where a statement comes from a document
only and I did not check it against code or data, it says "doc". Where I read file headers or JSON on disk, it says
"disk". Where I derived something myself, it says "derived".

---

## 0. The short version

1. **A large precomputed map of the clean checkpoint exists.** For all 60 x 21,504 feed-forward neurons there are, on
   disk: lens words (both faces, two readings), activation extremes and stream scales, the 24 strongest weight links
   each way with their two weight products, a steer lens (measured next-token effect of a push), 16 + 6 stored passages
   a face, a layout, and a measured check of the links. For 12 layers (36 and 40-50) there is a whole causal census
   (565,254 written answers a layer, against 2,048 random controls).
2. **The map is of the unedited model only.** Every builder and the census run in their own process on a freshly
   loaded checkpoint and never apply vault edits. The identity an index carries (`weights` card) names the checkpoint's
   files, not the weights in memory, so a vault edit does not change it and no index is refused after an edit.
3. **Staleness is handled by hand-written, per-route flags that know one edit type**: a rank-one edit of one neuron's
   `down_proj` column, as the vault makes. Several products are served unflagged after such an edit (the stats index and
   the push's unit, the census, the steer lens of other neurons, other neurons' upstream link lists, head profiles).
4. **A full rebuild is about 13 hours of the card without the census, and 13.2 hours more per census layer.** So a
   harness cannot rebuild after each edit; it needs to know what an edit invalidates and re-read only that, live.
5. **Trust is uneven by depth.** Lens labels are held back below layer 40; the steer lens is checked sound at 39-52 and
   weak at 0-38; census doses are calibrated only at 32-52 and a whole census exists only at 36 and 40-50; a link's
   existence and size are real at every layer but its served sign is below chance at an ordinary position. Below layer
   32 the only reading that is not known to be weak is the stored passages.
6. **All long jobs are command-line processes.** No HTTP route starts, queues, watches or cancels a build or a census,
   and on the home machine nothing but convention stops two processes loading the model onto the one card.
7. **The stand-in is a 6-layer random Gemma 4 on the CPU.** It rehearses plumbing (routes, indexes, vault, staleness,
   resume) and cannot rehearse a hypothesis about what weights compute. No small real model loads today.

---

## 1. What is on disk

Folder: `atlas_data/` in the checkout (git-ignored), or `$THEODOLITE_ATLAS_DIR` (`backend/atlas_index.py:107-111`).
Sizes from `du`/`ls` on 14 November 2026:

| folder | size | what |
|---|---|---|
| `atlas_data/index` | 3.5 GB | the indexes the server reads (below) |
| `atlas_data/census` | 61 GB | `runs` 32 GB, `volunteer` 22 GB, `trials` 4.0 GB, `nulls` 3.3 GB, `bars` 73 MB, `ties` 21 MB |
| `atlas_data/offline` | 1.9 GB | inputs of offline analyses (wiring check, two scratch steer-lens layers, determinism run) |
| `atlas_data/pilot` | 601 MB | the first index build of 27 October 2026 (lens, act_scale, wiring) |
| `atlas_data/backup-2026-10-27` | 323 MB | a wiring index from before links kept g and u |
| `atlas_data/scientist` | 106 MB | the scientist agent's work folder, logs, system prompt |
| `atlas_data/stand-in` | 1.6 MB | the stand-in's own indexes and census folder |
| `atlas_data/steer_run` | 92 KB | logs and timestamps of the steer-lens build |
| `atlas_data/exports` | 180 KB | three exported control vectors (`.gguf`) and a llama.cpp check |

### 1.1 The index files (`atlas_data/index`)

Every `.npz` holds a JSON string `meta` with `model`, `built_at`, `lens_id`, `stand_in`, and, for files built since
about 28 October 2026, `unpacker` and `weights` (`backend/atlas_build.py:50-56`). The reader class is
`AtlasIndex` (`backend/atlas_index.py:216`); file names are in `FILES` (`backend/atlas_index.py:82-90`). All arrays
are `[L=60, N=21504, ...]` (disk: array headers read from the files).

| file | bytes on disk | built (meta `built_at`, UTC) | carries `weights` id | covers |
|---|---|---|---|---|
| `lens.npz` | 249,019,830 | 2026-10-27 21:28 | **no** | all 60 layers |
| `act_scale.npz` | 30,971,616 | 2026-10-27 22:26 | **no** | all 60 layers |
| `layout_writes.npz` | 2,581,846 | 2026-10-27 22:39 | **no** | all 60 layers |
| `wiring.npz` | 509,595,825 | 2026-10-29 04:50 | `e0a93173c31ed2ec` | all 60 layers (links between every pair i < j) |
| `passages.d/` (24 `.npy` + `meta.json`) | about 690 MB | 2026-10-29 04:46 | `e0a93173c31ed2ec` | all 60 layers |
| `wiring_check.npz` | 510,052,990 | 2026-10-29 09:16 | `e0a93173c31ed2ec` | layers 0-58 (59 writes into no later layer) |
| `steer_lens.npz` | 129,032,742 | 2026-11-13 23:43 | `e0a93173c31ed2ec` | all 60 layers |
| `steer_lens.d/L{0..59}.npz` | 60 x 2.24 MB | 12-13 Nov 2026 | per shard | the per-layer shards the merged file is made from |
| `steer_dose.json` | 31,766 | written 2026-11-13 14:56 | yes | a dose for each of 60 layers |
| `steer_check.json` | 23,163 | written 2026-11-14 17:40 | names the lens build | a standing for each of 60 layers |
| `vocab.json` | 3,090,804 | 2026-10-27 | key = model id + vocabulary size | 262,144 word-pieces decoded alone |
| `head_profiles.json` | **absent** | never written here | (key would be texts + model id + layer count) | — |
| `wiring_check.set-aside-20261029T084541Z.npz`, `wiring_check.d.set-aside-...` | 142 MB | 28-29 Oct | — | the first night's check, older format, kept aside |

**`lens.npz` — each face's words through two readings.**
- Content: `lens_pos`, `lens_neg`, `eye_pos`, `eye_neg` `[60, 21504, 12]` int32 (word-piece ids), `silenced` `[60, 21504]`
  bool, `built` `[60]` bool, `K` (disk).
- How: for each neuron, the write column times the post-feed-forward norm weight, through the J++ lens matrix of the
  layer (`lens`) or not (`eye`, the logit lens), through the unembedding with the softcap, ranked by `label_top_ids`;
  the negative face is the positive face's logits negated (`backend/atlas_build.py:63-108`). It is a pure function
  of weights: one neuron's `W_down` column, `w_post`, `J_l`, the final norm and `W_U`. No forward pass.
- Cost: 144 s for all 60 layers on the 5090 on 27 October 2026 (`ROADMAP.md:328`); about 2.6 s a layer, 1.5 GiB above
  the model (`scripts/atlas_build.py:24-25`). `--layers` adds layers to the file in place (`backend/atlas_build.py:81-89`).
- Read by: `AtlasIndex.face_ids`, `face_check`, `search` (`backend/atlas_index.py:402-424`, `:458-494`); routes
  `/hover`, `/search`, and every "words" field beside a link. `GET /neuron/lens` does **not** read it: it reads live
  through the vault (`backend/atlas.py:1246-1247`).
- Trust: the label gate (`backend/label_gate.py:30-37`) holds a label back below layer 40 and, at 40-58, where the two
  readings share none of their first ten word-pieces. It was validated only for the `jpp-base` lens in layers 40-57
  (as a graded score, AUC 0.84; the cut's own record: of 240 faces, 209 kept of which 160 sound, 31 held back of which
  7 sound; `backend/label_gate.py:42-64`).

**`act_scale.npz` — activation extremes and the stream's scales.**
- Content: `pos`, `neg`, and the four GeGLU sign regimes `pp`, `nn`, `pn`, `np` `[60, 21504]` float32; `resid_q`,
  `mlp_rms_q`, `mid_rms_q` `[60, 3]` (10th, 50th, 90th percentile); `tokens` (disk).
- How: one forward per window over a corpus (`backend/atlas_build.py:136-150`). The corpus on disk is
  `HuggingFaceFW/fineweb-edu:sample-10BT + HuggingFaceH4/ultrachat_200k::train_sft`, 2,000,054 word-pieces in 3,914
  windows of 512 (disk: meta).
- Cost: 58 minutes on 27 October 2026 with the reference unpacker (`ROADMAP.md:329`, `scripts/atlas_build.py:54`);
  3.4 GiB above the model (`scripts/atlas_build.py:26-28`). Not resumable, no per-layer option.
- Read by: `Atlas.scales()` (`backend/atlas.py:273-293`): it gives `p50` (the unit of every "add" push), the extremes
  that scale every score, and the norm scales that weigh links. The census also reads its `p50` from this file and
  records the file's sha256 (`backend/census.py:1845`; disk: L45 meta `p50` 144.42, `act_scale_sha256`).

**`wiring.npz` — the 24 strongest direct-path links each way.**
- Content: for `down` (readers in later layers) and `up` (writers in earlier layers): `layer` int8, `n` int16, `z`
  float16, `lw` float16 (log10 strength over the neuron's strongest), `sign` int8, `gate` int8, `g` float16, `u`
  float16, each `[60, 21504, 24]`; `K` (disk). 30,449,664 live downstream links (disk: `sign_disagreement.links`).
- How: for every pair of layers i < j (1,770 pairs), `G = W_gate_j @ V`, `U = W_up_j @ V`, `P = G * U`, with
  `V = w_pre_j * w_post_i * W_down_i`; top 24 by `|P| f[i,j]^2`, f from the layer scalars and the norms' median scales
  (`backend/atlas_build.py:259-366`). The products are weight-only; the scale f uses the stats index.
- Cost: 224 s on the 5090 on 28 October 2026 (`scripts/atlas_build.py:31-34`); about 4.3 + 0.6 GiB above the model.
  One pass, no per-layer or per-neuron option.
- Read by: `AtlasIndex.links`, `cone`, `walk`, `web_summary`, `active_paths` (`backend/atlas_index.py:498-880`);
  routes `/neuron/wiring`, `/hover`, `/web`, `/web/summary`, `/cone`, `/run/paths`.
- It is a **top-K view**: a pair (A, B) outside a neuron's 24 strongest is not in the file. `/neuron/wiring?k=` up to
  64 falls back to a live read when k > 24 (`backend/atlas_index.py:503`, `backend/atlas.py:1420-1424`). No route gives
  the products of an arbitrary pair; `atlas_ops.pair_products` does, in Python (`backend/atlas_ops.py:1616`).

**`passages.d/` — where each face fires.**
- Content: `windows` `[3914, 512]` int32 (the corpus text itself, never committed), `unigram` `[262144]` int32, and
  per face `{pos,neg}_{win,at,act,g,u}` `[60, 21504, 16]` plus `{pos,neg}_smp_*` `[60, 21504, 6]` (disk: file sizes
  and meta: `K` 16, `sample` 6, `seq` 512, `per_window` 1, `seed` 0, `parts` 4).
- How: the same forward per window as stats; top 16 events a face (at most one a window) with the activation and the
  gate and up pre-activations there, and 6 drawn at random above half the face's extreme
  (`backend/atlas_build.py:161-252`). Parts of 1,024 windows are written as it goes and a stopped run resumes at the
  first missing part; a plan file refuses a resume with other settings or other weights (`:191-203`).
- Cost: 20 minutes on 28 October 2026 with `--unpacker compiled` (`scripts/atlas_build.py:51-55`, `ROADMAP.md:1490-1491`).
- Read by: `AtlasIndex.passages`, `window_ids`, `op_point` (`backend/atlas_index.py:428-454`, `:550-574`); route
  `GET /neuron/passages` (memory-mapped, no GPU; `backend/atlas.py:1534-1638`). Each passage returns `ids` that
  `POST /reads` reads again exactly as the window was read (`backend/atlas.py:1600-1601`, `:2414-2432`).
- Half of the events are kept back from the scientist for blind tests: those in odd-numbered windows
  (`backend/atlas_index.py:125-130`).

**`steer_lens.npz` — the measured next-token effect of a push.**
- Content: `up`, `down` `[60, 21504, 12]` int32 (the twelve word-pieces whose log-probability rose most under +d and
  -d), `up_kl`, `down_kl` `[60, 21504]` float16, `dose` `[60]`, `contexts` (4 strings) (disk).
- How: for each neuron, ±dose x p50 along its write direction at every position but the start mark of four contexts;
  the stream leaving the layer is computed once, then only the later layers run (`forward_from_layer`), in batches of
  64 neurons (`backend/atlas_build.py:776-836`). The dose is per layer, from a pilot that matches each layer's median
  KL to layer 45's at 0.8 of the stream (`:871-921`; disk: doses run 0.19 at layer 20, 0.31 at 32, 0.76 at 39, 0.8
  at 45, 1.28 at 52, 0.28 at 59; reference KL 0.355 nats over 128 neurons).
- Cost: the pilot 38 minutes (13 November 2026, 00:52Z-01:31Z); the build 10.5 hours over three sittings (2,702 s +
  3,395 s + 31,646 s) (`ROADMAP.md:369-381`). One layer alone: layer 45 took 287 s, layer 5 took 1,181 s (29 October
  2026, `ROADMAP.md:346-349`); cost grows with the number of layers after it. Resumable by layer shard
  (`backend/atlas_build.py:984-1055`); a shard built at another dose or on other weights is refused, not replaced
  (`:965-981`, `:1020-1026`).
- Read by: `Atlas.face` (`backend/atlas.py:250-270`), `_lens_reply` (`:1263-1269`), `AtlasIndex.search`,
  `web_summary`, `steer_dose`, `steer_common`, `steer_check`, `steer_weak` (`backend/atlas_index.py:241-327`).
- Its own check (`steer_check.json`, written on the CPU by `scripts/atlas_steer_check.py`): per layer, how many of a
  neuron's twelve word-pieces the whole layer shares, and how often a steer word is one of the neuron's five census
  words. Standing on disk: **unchecked at 0-31** (nearest checked layer is 32, weak), **weak at 32-38**, **agrees at
  39-52**, **unchecked at 53-59** (nearest checked layer is 52, agrees). At layer 0 a layer shares 5.76 of 12
  word-pieces, at 32 it shares 3.4, at 40 it shares 1.0, at 45 it shares 0.16, at 51 none. Match with the census over
  neurons above 95 in 100 controls: 0.07 at 32, 0.185 at 36, 0.33 at 38, 0.62 at 39, 0.60 at 40, 0.86 at 45, 0.84 at
  51, 0.75 at 52; shuffled 0.01-0.05 (disk).

**`wiring_check.npz` — the links measured.**
- Content: per layer `L{l}_*`: 512 writers, 48 readers each (24 linked + 24 null), `g`, `u`, `sign`, `gate`, `f`,
  `strength`, `op`, `reads` `[512, 48, 4, 3, 2]`, `g0`, `u0` `[512, 48, 4, 19]`; the meta holds a summary and a cost
  for each of 59 layers (disk). The server reads only the meta (`backend/atlas_index.py:92`, `:343`).
- How: prefill only; each writer pushed by ±0.25 of its own write ceiling on the census's four old questions; linked
  readers and as many unlinked readers are read (`backend/atlas_build.py:568-725`). Resumable by layer; refuses a run
  against another wiring, passages or stats index (`:616-636`).
- Cost: 59 layers, 68 prefills of 63 rows a layer, 1,840 s in all (disk: summed meta `cost.seconds`), reported as about
  32 minutes (`ROADMAP.md:1473-1474`).
- What it found (disk, medians over 59 layers, linked readers, at the last word-piece): the sign from rest held at
  0.336; the sign in use as the atlas serves it (read where the reader fires hardest) at 0.418; the sign in use scored
  where the reader actually sat at 0.841; Spearman's rho of the change's size against the first-order size at the
  site 0.89, against `|P| f^2` 0.31. Linked readers moved 5.5 times as much as the null's (`ARCHITECTURE.md:790`; 5.92
  at layer 45, disk). About 80% of linked readers sat below gelu's minimum at that position (`ARCHITECTURE.md:799-800`).
- Read by: `links_card` (`backend/atlas.py:631-649`) into `GET /arch` -> `links.measured`.

**`layout_writes.npz`** — `cell` `[60, 21504]` int16, each neuron's place on a 147 x 147 sheet by UMAP of unit write
directions (`backend/atlas_build.py:1080-1114`); 9 minutes (`ROADMAP.md:330`). Cosmetic. Route `GET /layout`.

**`head_profiles.json`** — where each head looks over eight reference texts; built by the server on the first
`GET /head/profile`, one forward a text (`backend/atlas.py:2836-2858`). It is not on disk in `atlas_data/index`, so it
has not been built on the 31B through this folder. Its cache key is the texts, the model id and the layer count; it
names neither weights nor vault edits.

### 1.2 The census (`atlas_data/census/runs`, `backend/census.py`)

- **What it is.** For one layer, every neuron and some random directions are pushed along a write direction on fixed
  questions at fixed doses, greedy, and every answer is kept; a neuron is ranked by the words it brings against the
  controls' mean (`backend/census.py:1-71`, `:610`). It is the instrument's only causal reading with a null.
- **On disk, whole layers (disk: every `*.meta.json` read):** layer 36 (`match_ceil`, started 2026-11-14 01:07Z) and
  layers 40-50 (`break_rel`, 29 October to 10 November 2026). Each has 21,504 neurons and 2,048 controls, 565,254
  answers (disk: L45 rows index), a 2.75-2.78 GB `.jsonl`, a 30 MB `.units.jsonl` ranking (23,552 records), a
  1.0-1.2 MB `.table.npz`, a `.rows.npz`, a `.collapse.json`, and for volunteers' layers a `.provenance.json`.
  Layers 42, 44, 47, 48, 50 were written on an RTX PRO 6000 Blackwell, the rest on the RTX 5090 (disk: meta `runtime.device`).
- **Also on disk:** calibration ladders at layers 20, 30, 40, 45, 50, 57 (64 neurons + 64 controls each, two designs);
  trials under `census/trials` (about 1,022 neurons against 1,024 controls each; folders `neighbours`, `ceil`, `soft`,
  `soft_walk`, `match`, `match_walk` and others); a second null at layer 45 under `census/nulls`; six volunteer bundles.
- **A ranking record** (disk, first line of `L45-break_rel.units.jsonl`): `kind`, `neuron`, `layer`, `z`, `mass`,
  `words` (12, each with count and z), `per_prompt`, `dose` (per dose: `rel_push`, `distinct4`, `capped`, `logprob`),
  `null_pct`, `p`, `q`, `collapse`, `collapse_mass`, `mass_shown`, `shift`, `shift_up`, `shift_down`, `reading`.
- **Cost.** A whole layer is 565,254 answers: 13.2 hours on the home 5090 and 2.7 GB (`ROADMAP.md:711`). A trial is
  about 49,000 answers and 70 minutes (`ROADMAP.md:725`). Layer 36's `.t0`/`.t1` files are 58,022 s apart with one
  2,016 s pause recorded in `.pauses` (disk; which script writes those three files is not established: no file under
  `scripts/` or `backend/` names `.pauses`).
- **Where its doses are counted calibrated** (`backend/census.py:268-272`, `:277`): `break_rel` at layers 39-51,
  `soft_ceil` at 38 and 52, `match_ceil` at 32-37, for answers of 48 word-pieces (`break_rel` also 32). "No push
  strength has passed a trial below layer 32" (`ARCHITECTURE.md:825`). Layer 36's whole reading passed and its lower
  dose alone failed one gate by 0.006 (commit `eb3a2cc`, 14 November 2026); what to call that is the owner's.
- **How the server reads it.** `layer_runs` picks one run a layer by model id, leaving out shards, joined runs,
  controls-only runs, calibration designs and runs of `version` < 2 (`backend/census.py:1011-1044`). It reads by
  seeking: `RowsIndex` and `UnitsTable` beside each run, remade when the file's size or time changes
  (`backend/census.py:1201`, `:1290`; `backend/atlas.py:856-955`). A layer's list opens in 0.13 s and a neuron's
  answers in 0.01 s, after 2.4-3.4 s once a layer for the rows index (`ROADMAP.md:1396-1397`).
- **Chance.** About N / (C + 1) neurons beat every control by chance (`backend/census.py:44-46`): about 10 of a layer
  with 2,048 controls (derived: 21,504 / 2,049). Counts above every control hang on each layer's single largest
  control (95 at layer 41, 4,793 at layer 44: disk, `steer_run/against_census.log`); `q <= 0.05` is the steadier
  count (3,534 to 9,133 a layer, `ROADMAP.md:683`).

### 1.3 Labels written by people and by the scientist

Kept outside the checkout in `~/.local/share/theodolite/` (`backend/field_notes.py:104-134`). On disk on 14 November
2026: `field_notes.json` holds 8 named neurons and no heads; `field_book.json` does not exist; `field_proposals.json`
holds nothing waiting; `field_tests.json` (165 KB) holds pattern-test results for 21 neurons. So human-curated labels
cover a few dozen neurons out of 1,290,240.

---

## 2. The HTTP API (`/api/atlas/*`)

53 routes in `backend/atlas.py` (router at `:114`). Every route answers only requests whose peer is loopback and whose
`Host` and `Origin` name this machine, and refuses a request a browser marks cross-site (`backend/atlas.py:84-111`).
A script on the same machine can call them. No route under `/api/atlas` writes a weight: pushes act on activations and
on the residual stream for one request; weight edits belong to the vault (another reader's area).

"GPU" below means the route takes the server's one `asyncio` lock and runs in a worker thread (`on_gpu`,
`backend/atlas.py:148-165`); it waits behind any other reading, a push chat included. I read the bodies of the routes
marked (code); the others are described from `ARCHITECTURE.md:901-933` and their signatures.

### 2.1 Reads from disk, no GPU (fast)

| route | returns |
|---|---|
| `GET /arch` (code, `:652-694`) | model id, `stand_in`, `n_layers`, `d_model`, `d_ff`, `vocab_size`, each layer's attention shape, `layer_scalars`, `indexes` (per file: built, bytes, its whole meta, or `refused` with the reason), `census` (layer -> run file), `census_refused`, the push card (modes, units, defaults, caps, `scale_source`), limits, `links` (both sign models, whether "use" is readable, the measured check per layer), label-gate and lens cards |
| `GET /hover?layer&n` (code, `:2017-2032`) | both faces' words (lens, eye, check, steer), 16 links each way, name, `links_stale` |
| `GET /search?q&limit` (code, `:2035-2073`) | neurons whose face holds a word-piece containing `q` among its twelve (lens, eye, or steer lens at non-weak layers); `results` and `held` (label gate), with census words |
| `GET /neuron/passages?layer&n&face&k&before&after&fold` (code, `:1534-1638`) | the face's stored events with `act`, `rel`, `regime`, `g`, `u`, `window`, `at`, display units, `ids`; `reads`/`follows` word counts with lift; `stale` |
| `GET /neuron/wiring?layer&n&k&model` (code, `:1406-1436`) | `down` and `up` links with `layer`, `n`, `z`, `w`, `sign`, `gate`, `face`, words, census words, name; `source` "index" or "live"; with `model=use`, `use_sign` per reader face and a `use` card. No GPU only when the index has the neuron and it is not edited |
| `GET /web`, `GET /web/summary`, `GET /cone` (code, `:2076-2109`, `:2993-3038`) | the cone as int16 rows; its pooled words; the cone as a list of neurons with hop, strength, face, words |
| `GET /layout` (code, `:2112-2122`) | `cell` as base64 int16 |
| `GET /census/layers`, `/census/levels`, `/census/search?q` (code for the first two, `:1077-1139`) | every census at once: per layer its run, design, null counts, `q05_neurons`, `doses_calibrated`; one byte a neuron for the map; neurons whose census words match `q`, with the controls matched the same way as the null |
| `GET /census/units?layer`, `/census/card?layer`, `/census/unit?layer&n&kind`, `/census/rollout?layer&i`, `/census/spread?layer` (code for units, card, unit, rollout, `:1010-1065`) | the layer's ranking; the run's card (design, doses, `doses_calibrated`, `read_gain`, null, device, `mixed_runtime`, `writer_floor`, `lead_says`); one unit's 24 answers, the unpushed answers and its statistics; one answer |
| `GET /neuron/tests`, `GET /neuron/written_test/lead`, `GET /notes`, `GET /fieldbook`, `GET /proposals` | kept pattern tests; where a behind-its-passage test cuts; names, notes, field book, proposals |
| `GET /run/view`, `GET /run/paths` | filters and active links over the text last run with `POST /run` |

### 2.2 Live operations, GPU

| route | what it runs, what it returns |
|---|---|
| `GET /neuron/lens?layer&n` (code, `:1311-1319`) | both faces read live from the weights through the vault, edits included; `gate_trigger`; the steer lens from the index (or as last read live); `act_range`; `ceiling` {write, p50, rel}; `edited` |
| `POST /neuron/steer_lens?layer&n` (code, `:1322-1346`) | one neuron's steer lens read live at the layer's dose |
| `GET /neuron/heads`, `GET /head`, `GET /head/profile`, `GET /head/row`, `POST /head/pattern`, `POST /attention`, `POST /attn_drive` | head roles and head wiring from the weights; attention on a text; which heads, blocks and earlier neurons drive one neuron at one position, split exactly |
| `POST /reads` (code, `:2428-2450`) | one neuron's activation at every display unit of a text or of exact `ids` (at most 2,048 positions), optionally with up to 32 pushes and 64 head scales in the same pass |
| `POST /logprobs` (code, `:2453-2487`) | next word-piece log-probabilities; with pushes, base and intervened rows of one forward, their KL, the twelve that rose and fell, and named `targets` |
| `POST /compare` (code, `:2490-2521`) | up to 300 neurons over up to 24 texts, one forward a text: max, min and peak score per neuron and text |
| `POST /fires`, `POST /run` (code for run, `:2586-2601`) | neurons firing hardest on a text; a text run once (at most 256 positions) with every activation kept in one server-wide slot |
| `POST /push` | the three voices (+d, 0, -d) as server-sent events, greedy or sampled |
| `POST /steer` | the three voices as whole texts; or up to 8 prompts under one set of pushes; or up to 12 candidate sets of pushes on one prompt, in one batch |
| `POST /control_vector` (code, `:2376-2399`) | writes an add push as a llama.cpp control vector under `atlas_data/exports/`, recording `vault_edits_in_force` |
| `POST /neuron/written_test`, `POST /neuron/blind_test/start` | a face's pattern tested by written texts, or by pairs with kept-back passages |

Recorded speeds: a `/steer` call of three rows takes 8.3 s with `compiled`; five prompts 9.2 s; six candidates 8.8 s
(11 November 2026, `ARCHITECTURE.md:920`). A decode step costs 0.71 s with the reference unpacker whatever the batch,
0.103 s at 32 rows and 0.126 s at 96 rows with `compiled` (28 October 2026, `ROADMAP.md:443-445`). On the GPU a
batch's shape moves a row's numbers, so a pushed row must be read against the unpushed row of its own call
(`ARCHITECTURE.md:920`).

Caps: 32 pushes and 64 head scales a reading; a whole block or every head of a layer can only be silenced
(`ARCHITECTURE.md:683-690`). Push units: `add` is a share of the stream's median length (p50) along the neuron's
write direction; `mul` scales the activation; `act` adds a multiple of the neuron's usual extreme.

---

## 3. Identity and staleness

### 3.1 What the atlas keys its data by

- **Model id.** Every index and every census meta records `model` (`google/gemma-4-31B-it-qat-w4a16-ct` on disk).
  A reading off the GPU or in float32 has another id, for example `...@cpu/bfloat16`
  (`backend/runtime_factory.py`, module docstring lines 30-43).
- **Weights card.** `weights_card(rt)` gives, for the checkpoint, `repo`, `snapshot`, `files_sha256` and `id` (its
  first 16 hex digits), `linears`, `dtype`, `device`, `runtime` (`backend/runtime_factory.py:251-272`).
  `files_sha256` is the sha256 of the names of `config.json` and `*.safetensors` in the Hugging Face snapshot and of
  the blob names their links point to (`backend/runtime_factory.py:228-248`). It names files in the cache. It is not
  a hash of the tensors in memory. A GGUF is compared by its fingerprint and header sha256.
- **The comparison.** `weights_differ` (`backend/atlas_index.py:173-196`) returns None when either side has no card,
  and for the stand-in always. `AtlasIndex._refusal` refuses a file for another model id, other weights, another
  shape, or missing arrays (`backend/atlas_index.py:349-371`); `GET /arch` shows the reason.
- **On disk today (disk).** `wiring`, `passages`, `steer_lens`, `wiring_check`, `steer_dose.json` carry weights id
  `e0a93173c31ed2ec`. `lens.npz`, `act_scale.npz` and `layout_writes.npz` were built on 27 October 2026 before metas
  named weights, so they are accepted on the model id alone. Of the 12 whole census layers only layer 36's meta has
  `weights`; and `layer_runs` filters on the model id alone in any case (`backend/census.py:1031`). The census
  compares weights only when a run is resumed, merged or joined (`backend/census.py:2195-2196`, `:2378`, `:2449`).
- **Other keys.** `passages` and `wiring_check` plans refuse a resume against changed settings or changed sibling
  indexes (`backend/atlas_build.py:191-203`, `:616-636`). `steer_check.json` is read only for the lens build whose
  `built_at` it names (`backend/atlas_index.py:311`). A census run records the stats index's sha256, the git commit,
  the card and library versions (`backend/census.py:1809-1846`).

### 3.2 What a vault edit does to that identity

Nothing. A vault edit is an in-memory overlay on one neuron's `down_proj` column; the vault reports
`edits_in_force()`, the latest record for each (layer, neuron) (`backend/larql_vault.py:233-239`). The weights card
does not change, so no index is refused. The atlas's own key for "the model as it now stands" is
`Atlas.edit_key()`, the tuple of patch ids in force, and `Atlas.edited()`, the set of edited (layer, neuron) pairs
(`backend/atlas.py:215-219`). There is no content hash of an edited model anywhere in the files I read.

### 3.3 Which products go stale, and what detects it (code read; the "ought" column is derived)

An edit of neuron (l, n)'s write column changes, in principle: that neuron's own words and links; everything read
from activations at layers after l; and every measured push effect that runs through layer l.

| product | what the code does after an edit | what is not flagged |
|---|---|---|
| lens words | `GET /neuron/lens` reads live (`:1247`). Index words of the edited neuron carry `stale` and its label check is marked `edited` (`backend/atlas.py:257-269`) | nothing (weight-only, only that neuron changes) |
| wiring | the edited neuron's own links are read live and cached by edit key (`:1363-1372`); `/hover`, `/web/summary` and `/cone` say `STALE_WEB`, and `/web` sets an `X-Web-Stale` header, when the root neuron is edited (`:1278`, `:2029`, `:2087-2088`, `:2105-2106`, `:3027`) | the `up` lists of later neurons that read the edited neuron come from the index as they were; a cone that passes through an edited neuron is flagged only at the root |
| sign in use | `use.stale` when an edit lies at an earlier layer than a reader (`:1382-1403`) | — |
| passages | `stale` for the edited neuron and for any neuron at a later layer (`:1602-1608`) | — |
| steer lens | the edited neuron's index entry is withheld with `steer_stale` until read live (`:1263-1275`) | every other neuron's steer lens is served from the index, though a push anywhere runs through later layers |
| stats (`p50`, extremes, norm scales) | **nothing**: `scales()` returns the index whenever it is built (`:277-283`). Only the fallback without an index is keyed by the edits in force (`:284-293`) | the push's unit, every score, `act_range`, the write ceiling's `rel`, and the link scale after any edit |
| census | **nothing**: no census route reads `edited()` (no reference between `backend/atlas.py:828` and `:1245`) | census words and ranks of every neuron |
| head profiles | **nothing**: keyed by texts, model id and layer count, held in memory for the server's life (`:2836-2858`) | — |
| wiring check (`links.measured`) | nothing (meta only) | — |
| the last run (`POST /run`) | nothing: one slot, not keyed by edits (`:2586-2594`) | activations kept from before an edit |
| live caches (wiring, head roles, head wiring, steer, captures) | keyed by `edit_key()` so never reused across edits (`:1368`, `:1440`, `:2679`, `:2812`) | — |

`scripts/check_atlas_api.py` section 7 (docstring lines 43-45) checks exactly the flagged column on the stand-in:
wiring read live again, faces say stale, caches not reused, passages of the edited neuron and of later layers stale.

The flags assume the vault's one edit type. An edit of a `gate_proj` or `up_proj` row, of attention weights, of a
norm, of a layer scalar or of the embedding would not appear in `edits_in_force()` and nothing above would fire
(derived from `backend/atlas.py:215-219`; whether the vault can make such edits is another reader's area).

### 3.4 Can an index be built for an edited model?

Not through the builders as they are. `scripts/atlas_build.py` and `scripts/census_run.py` load a runtime with
`make_runtime(...).load()` and never construct a vault or apply patches (`scripts/atlas_build.py:230-232`; the only
vault either script touches is the stand-in's, unused, `:228`). `backend/atlas_build.py:2-4` says a fresh process has
no edit in force. The one path that exists: a GGUF of Gemma 4 can be indexed into another folder
(`THEODOLITE_GGUF=<file>`, `--out`; `scripts/atlas_build.py:85-87`), and a GGUF has its own fingerprint and model id,
so an edit exported as a full GGUF would get its own identity. Whether such an export exists is not established here.

---

## 4. Long jobs

- **Started by** the command line only: `scripts/atlas_build.py <lens|stats|wiring|layout|steer_dose|steer|passages|wiring_check|all>`
  (`scripts/atlas_build.py:173`; `all` is lens, stats, wiring, layout, `:248`) and `scripts/census_run.py`. No
  `/api/atlas` route starts one (no `subprocess`, `take_census` or builder call in `backend/atlas.py`).
- **Each loads the model itself**: 21.2 GiB of the 32 GB card with the J++ lens, about 16 s to load
  (`scripts/atlas_build.py:22-23`). Both scripts say "Run one GPU process at a time" (`scripts/atlas_build.py:19-20`).
  The server holds the card too, so on the home machine it is stopped for a build or a census
  (`scripts/census_intake.py:389`; the owner's working notes allow restarting the `:8765` server for GPU checks).
- **Who holds the card.** Inside the server one `asyncio.Lock` serialises every reading (`backend/server.py:99`,
  `backend/atlas.py:121-124`). Between processes on the home machine there is no lock: only convention. Rented pods
  have a `flock` on `logs/gpu.lock` (`scripts/pod/common.sh:104-110`). The steer-lens build was paused twice by hand
  to give the census the card (`ROADMAP.md:376-380`).
- **Resume.**
  - `lens --layers` adds layers to the file (`backend/atlas_build.py:81-89`).
  - `passages` resumes at the first missing part of 1,024 windows (`:210-216`).
  - `wiring_check` and `steer` resume at the first missing layer shard (`:655-658`, `:1028-1031`).
  - `steer_dose` keeps each measured layer in its JSON (`:906-908`).
  - `stats`, `wiring` and `layout` are single passes with no resume.
  - A census's batches are planned from the whole job list and written whole; the same command continues a stopped
    run row for row, and other settings are refused (`backend/census.py:1906-1927`, `:2004-2008`).
- **Cancel.** A build is stopped by killing it; every index file is written to a temporary name and renamed
  (`backend/atlas_index.py:155-160`), and the passages folder removes its `meta.json` first and writes it last
  (`backend/atlas_build.py:237-248`). A census handles SIGINT: it stops between decode steps, drops the batch under
  way and keeps every whole one (`scripts/census_run.py:313-317`, `backend/census.py:1964-1965`).
- **Sharing results with the server.** The server reloads an index file when its mtime changes
  (`backend/atlas_index.py:329-347`) and looks for census runs again every five seconds (`backend/atlas.py:843-853`),
  so a job's output appears without a restart. `scripts/atlas_build.py` refuses to write over an index built for
  another model or other weights unless given `--replace` (`:249-258`).
- **Volunteers and shards.** A layer can be taken in shards with a canary batch each, merged only whole
  (`backend/census.py:2011-2013`, `:2354`); six of the eleven `break_rel` layers came from volunteers' cards.

---

## 5. The stand-in and other models

- `scripts/atlas_stand_in.py` builds a real `Gemma4ForCausalLM` with **6 layers, hidden 64, feed-forward 128,
  a trained byte-level vocabulary of about 600, random weights**, float32 on the CPU (`:44-48`, `:102-142`), wrapped
  in the instrument's own `TheodoliteRuntime` with a random "lens" a layer (`:168-197`) and a real vault
  (`:200-206`). Its model id is fixed text (`:145`).
- It runs everything: `atlas_build.py <step> --stand-in`, `census_run.py --stand-in`, and
  `atlas_stand_in.py --serve PORT` serves the whole instrument on it (`:15-23`, `:225-243`). Everything it writes
  goes under `atlas_data/stand-in/` (`:148-165`); its whole index set is 1.6 MB on disk.
- The check scripts are built on it: `scripts/check_atlas_index.py` runs every builder into a temporary folder and
  compares each file with the same quantity read live (docstring lines 2-56); `scripts/check_atlas_api.py` drives the
  FastAPI app with the stand-in bound, including a vault edit in force and the GPU lock (docstring lines 2-55).
  How long these checks take is not established: no document I read records it.
- **What it can rehearse:** the mechanics of an edit loop — apply, read live, staleness flags, caches, rollback,
  resume, refusals, the API's shapes. **What it cannot:** any hypothesis about what the weights compute. The weights
  are random, every lens label is held back on six layers, and on it "every other path counts as much as the direct
  one" (`ARCHITECTURE.md:768-770`).
- **Other real models.** `THEODOLITE_GGUF` loads "a GGUF of Gemma 4" with its shape read from the file, and the
  indexes can be built for it. The GGUF builder refuses Gemma 4's mixture-of-experts model (26B-A4B) and the models
  with per-layer embeddings (E2B, E4B) in plain words (`backend/gguf_model.py:43-44`, `:213-219`). So no smaller real
  Gemma 4 loads today. The J++ lens is fitted for the 31B only. `weights_differ` treats any two stand-ins as the same
  weights (`backend/atlas_index.py:191-192`), so a stand-in with another seed would not be told apart.

---

## 6. Using it as a map for weight-space programming

### 6.1 Cheap lookups (milliseconds, no GPU, no lock)

- A neuron's words on both faces, with the gate's verdict: `/hover`.
- Words to neurons: `/search` (lens, eye, steer) and `/census/search` (causal, with its own chance count).
- A neuron's 24 strongest readers and writers with `g`, `u`, `z`, relative strength: `/neuron/wiring`.
- Multi-hop leads: `/cone`, `/web/summary`.
- Where a face fires in real text, with the gate and up pre-activations at each event: `/neuron/passages`.
- What a push of the neuron does to written answers, where a census exists: `/census/unit` (24 answers, p, q).
- Which products exist and for which weights: `/arch`.
- Directly in Python without the server: `AtlasIndex(root, model, shape, weights)` reads every file with numpy.

### 6.2 Cheap live reads (one prefill)

`/neuron/lens` (weights only, but it waits for the lock), `/reads`, `/logprobs` (base and intervened in one forward,
with KL), `/compare` (300 neurons x 24 texts), `/attn_drive`. These are the natural "test one hypothesis" calls.

### 6.3 Expensive

Written text: `/steer` at about 8-9 s for three to fifteen rows. A trial with controls: 70 minutes. A whole census
layer: 13.2 hours. A steer-lens layer: about 5 minutes at layer 45 to about 22 minutes at layers 0-5. Any corpus pass
(stats, passages): 20 to 58 minutes.

### 6.4 What to trust at which layer

| reading | 0-31 | 32-38 | 39 | 40-50 | 51-52 | 53-58 | 59 |
|---|---|---|---|---|---|---|---|
| lens / eye labels | held back | held back | held back | gated; validated band | gated; validated band | gated; validated to 57 | lens and eye are one reading |
| steer lens | unchecked beside a weak layer: treated as weak | weak | agrees | agrees | agrees | unchecked, nearest agrees | unchecked, nearest agrees |
| census, whole layer | none | layer 36 only | none | every layer | none | none | none |
| census doses calibrated | no | `match_ceil` 32-37, `soft_ceil` 38 | `break_rel` | `break_rel` | `break_rel` 51, `soft_ceil` 52 | no | no |
| wiring: link exists, relative size | yes | yes | yes | yes | yes | yes | reads nothing later |
| wiring: served sign | below chance at an ordinary position, every layer | | | | | | |
| passages, activation extremes | yes | yes | yes | yes | yes | yes | yes |
| pattern tests (no push) | work at every layer; run for 21 neurons | | | | | | |

Sources: label gate `backend/label_gate.py:30-37`; steer standing `steer_check.json` (disk); census
`backend/census.py:268-272` and the run metas (disk); wiring `ARCHITECTURE.md:772-800` and the check's meta (disk).

Two further cautions from the documents:
- A steer-lens reading is "two readings of the same push" as the census, never a test of what the neuron does
  unpushed (`ROADMAP.md:412-413`).
- A face's passages need not be reproducible by a short text: of 176 faces tried, 30 need more than 60 word-pieces
  of their own document and 8 need more than 256, all 8 at layer 26 or later (`ARCHITECTURE.md:848-849`).

### 6.5 Facts about the object that bound what an edit can do

- The feed-forward output passes a norm and the layer's `layer_scalar` (0.03 to 0.99) before it joins the stream.
  Turning one neuron's activation up is squashed by that norm; one neuron's most is its **write ceiling**
  (`backend/census.py:19-23`, `ARCHITECTURE.md:694-697`, `:701`).
- The census's and the steer lens's pushes are therefore mostly larger than any single neuron can write: 0.8 of the
  stream is 1 to 10 ceilings at most layers; 1 of the stream is 1.76-2.76 ceilings at 39-51 and 5 to 19 at layers
  2-25 (`ROADMAP.md:740-742`). A census word is evidence of what the direction does when forced, not of what a weight
  edit of that neuron can achieve.
- At an ordinary position about nine readers in ten sit with the GeGLU gate shut (`ARCHITECTURE.md:713-714`), which
  is why weight-only signs fail there.

---

## 7. Where documents and code or data disagree

1. `backend/atlas_build.py:10` says the costs in `scripts/atlas_build.py`'s help are "estimated ... from the
   arithmetic, not measured"; that help says they were "measured in a pilot on one RTX 5090 (27 October 2026)"
   (`scripts/atlas_build.py:22-23`). The script's text is the later one.
2. `scripts/atlas_build.py:49-50` says the steer-dose pilot takes "about half an hour ... (inferred from the build's
   rate, not timed)"; it was timed at 38 minutes on 13 November 2026 (`ROADMAP.md:369-370`).
3. `ARCHITECTURE.md:664-666` says the server reads a file "only for the model and shape it was built on"; the code
   also compares the weights card (`backend/atlas_index.py:354-356`), and three files on disk have no card to compare.
4. `ARCHITECTURE.md:677` lists `head_profiles.json` among the index files; it does not exist in `atlas_data/index`.
5. `ARCHITECTURE.md:882-884` and the owner's working notes summarise the steer lens as "sound from layer 39 up". The check
   on disk covers 39-52 only; 53-59 are `unchecked`, and the code treats them as not weak because their nearest
   checked layer agrees (`backend/atlas_index.py:316-327`).
6. `backend/census.py:36-37` still opens with "on Gemma-4-31B they are not yet calibrated", which is true of
   `main_rel` only; the same docstring and `DOSES_CALIBRATED` (`:268-272`) say where `break_rel`, `soft_ceil` and
   `match_ceil` are calibrated.
7. `ROADMAP.md:700` heads the forty-nine-layer plan "none of it run yet"; the text beneath it and the trials on disk
   show trials at 32-39, 51 and 52 run since.

---

## 8. Not established

- Which script writes a census run's `.t0`, `.t1`, `.pauses` and `.commit` files, and so exactly what their
  difference measures.
- How long the stand-in checks (`check_atlas_api.py`, `check_atlas_index.py`) or a stand-in build take.
- Whether the vault can edit anything but a `down_proj` column, and whether an edited model can be exported whole as
  a GGUF and indexed under its own identity (another reader's area).
- Which unpacker the running server uses, and so the real latency of each live route today.
- The bodies of `/push`, `/steer`, `/fires`, `/run/heads`, `/run/paths`, `/head/row`, `/head/pattern`, `/attention`,
  `/attn_drive`, the pattern-test routes and the notes routes: described from `ARCHITECTURE.md` and their signatures,
  not read line by line.
- The many statistics inside `census.py` (`compute_units`, `dose_window`, the collapse rule): structure and products
  read, the arithmetic not followed.
- How much an edit at one neuron actually moves the stats index, the census or other neurons' steer lenses. Section
  3.3's "not flagged" column says what could change, not how much.
