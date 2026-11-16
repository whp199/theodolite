# Measurements that could judge a weight edit

Reader's report for the weight-space harness design. Read-only pass over the repository at
`(the repository root)` (HEAD `e6ac64f`, 14 November 2026). Nothing was run on the GPU and no file in the
repository was changed. Paths are relative to the repository unless they begin with `/`. Layers count from 0
throughout (the pages count from 1), unless a line says "page layer".

Files read in full: `backend/causal_check.py`, `backend/steer_check.py`, `backend/pattern_tests.py`,
`backend/label_gate.py`, `backend/scientist_methodology.md`. Read in the parts that answer the questions:
`backend/atlas_ops.py` (lines 1-1045, 1420-1735, 1810-1972), `backend/theodolite_math.py` (1-600, 756-1022),
`backend/mcp_scientist.py` (400-1609), `backend/census.py` (1-345, 600-780, 1420-1620), `backend/atlas_build.py`
(355-385, 690-950), `backend/atlas.py` (2414-2603 and the route map), `backend/larql_vault.py` (120-205, 464-637),
`backend/engine_runtime.py` (335-440, 455-493, 1525-1745), `ARCHITECTURE.md` §2, §3, §4.5, §4.6, §4.7 (700-1050),
§6 (1752-1800), `COMPUTE.md` 1-80, `README.md` §v, `ROADMAP.md` 1-235, 660-684, 1280-1362, and the result files
named below.

---

## 0. The five things a designer must not get wrong

1. **A push is a per-row hook; a vault edit is a module-wide overlay.** Pushes are resolved to per-row tensors
   (`atlas_ops.Interventions`, `atlas_ops.py:267-282`) and installed as forward hooks (`installed`,
   `atlas_ops.py:344-412`), so a pushed row and an unpushed row sit in ONE forward. The vault's edit replaces a
   column of the dequantised weight for every row of the forward (`engine_runtime.py:356-375`), so as built an edited
   row and an unedited row CANNOT share a forward. Every "same forward" guarantee below is a guarantee about pushes,
   not about vault edits.
2. **Rows of one forward are bit-identical; rows of different batch shapes are not** (KL up to 0.014, N1A, cited at
   `causal_check.py:6-7`). Every evaluator that compares two conditions was built around that fact. A harness that
   compares an edited forward with an unedited forward must hold the batch shape fixed, or put both in one forward.
3. **Three classes of weight edit already have an exact per-row stand-in.** `mul f` on neuron n is the column
   `W_down[:, n]` scaled by f (a rank-one edit; `f = 0` is the vault's DELETE); a head scale f is `o_proj`'s input
   block of that head scaled by f; the mapping table is written out for the LoRA export at
   `ARCHITECTURE.md:1781-1791`. The only difference is positions: a push spares the start mark unless
   `spare_bos=False`, a vault edit acts everywhere (`ARCHITECTURE.md:1790`). A general column replacement (UPDATE,
   INSERT) has no per-row stand-in.
4. **KL is computed in two directions and from two logit precisions.** `/logprobs` reports KL(base ‖ intervened)
   (`atlas.py:2477`); `steer_lens` and the causal check report KL(pushed ‖ base) (`atlas_ops.py:1967`,
   `causal_check.py:425-427`); the vault's voice reports KL(before ‖ after) (`larql_vault.py:176-180`). The first
   three use an fp32 GEMM on the final-normed state (`causal_check.logits_fp32`, `causal_check.py:132-139`); the
   voice takes the writing loop's own logits cast to float32 (`engine_runtime.py:1727`).
5. **There is no regression suite, no capability check and no multi-token KL evaluator anywhere in the code.** The
   only existing measurement of an edit's effect is the vault's voice: ONE question, greedy, 12 word-pieces
   (`larql_vault.py:569-637`).

---

## 1. Catalogue of measurements

Legend. "Same forward" = can the changed and unchanged conditions be rows of one forward (for pushes). "Edit-aware"
= does the measurement read the model with vault edits in force. All forward-based measurements are edit-aware,
because the overlay sits inside each quantised linear's forward (`engine_runtime.py:356-375`) and `Weights` reads
columns through the runtime's accessors, "the vault's edits included" (`atlas_ops.py:128-131`, `167-172`).

### 1.1 Behavioural: the model's own odds

| measurement | where | inputs | outputs | batches? | same forward? |
|---|---|---|---|---|---|
| `next_logprobs` | `atlas_ops.py:665-690` | one text's ids; optional `Interventions` of exactly two rows (signs `(0, 1)`) | `[V]` float32 log-softmax of the next piece, base and intervened, from fp32 softcapped logits | one text a call; 1 or 2 rows | yes, by construction (`:677-678`) |
| `POST /logprobs` (scientist tool `logprobs`) | `atlas.py:2453-2487` | text, chat, prefix, pushes or head scales, `targets`, `k` | top-k of each row, `targets`' log-probabilities (first piece only for a multi-piece word), `kl` = KL(base ‖ intervened), twelve pieces that gained and lost most probability | no | yes |
| `steer_lens` (one neuron) | `atlas_ops.py:1928-1972` | C contexts, layer, neuron, p50, dose | per sign: mean Δlog-prob over contexts, its k biggest rises, KL(pushed ‖ base) averaged over contexts | 3·C rows in one forward (base, +, −), right-padded | yes |
| `build_steer_layer` (a whole layer) | `atlas_build.py:775-836` | layer, p50, dose, batch of neurons | per neuron: twelve risen pieces each sign, `up_kl`, `down_kl` | 64 neurons × 2 signs × 4 contexts + 4 base rows a forward, layers above only | yes |
| `causal_check.run_check` + `face_effect` | `causal_check.py:226-391`, `470-480` | layer, neuron, list of deltas, write mode, `token_ids` (one text) or `texts_ids` (36 texts) | per pushed slot `[C, V]` log-softmax beside its own forward's base rows; `c` (mean Δlog-prob), KL per context, split-half ρ of even and odd texts | study mode: 36 × (1 + slots) rows, 180 by default; halves the slots on out-of-memory, each forward keeping its own base row | yes; `base_rows_identical` says whether split forwards' base rows matched |
| MCP `causal_spot_check` | `mcp_scientist.py:1201-1557` | as above plus `k`, `write`, `scale_ref`, `floor`, `top_n`, `track_ids` | per face: `kl_mean`, `kl_median`, `kl_by_context`, `kl_over_floor`, `split_half_rho`, rises, falls, tracked ids, `floor_top100_overlap`, `linear` (how straight the push stayed), `label_overlap` (P@10); `antipodal` ρ of c+ and c−; `timing_ms` | as `run_check` | yes |
| the vault's voice | `larql_vault.py:569-637`, `engine_runtime.py:1667-1745` | one question (chat by default), `max_new_tokens` 1-64 (default 12) | both greedy answers, first-step KL(before ‖ after), the target word's probability and rank at step 0 before and after, `parted_at` | no: one question, one row, two separate writes | no: before and after are separate forwards of the same shape |

Notes that matter for an edit:

- `next_logprobs` refuses anything but the two-row layout (`atlas_ops.py:681-682`), so it compares one intervention
  with base on one text. A sweep over texts is a loop of calls.
- The causal check's "given" mode runs the text before the read position once, batch 1, and then only the last
  position for every row over that frozen prefix (`causal_check.py:268-283`, `_frozen_prefix` `:180-205`). That is
  exact for a push at the last position ("nothing before that word-piece can change", `:19-21`). It is NOT exact
  for a weight edit, which acts at every position the neuron fires on.
- The causal check sizes its push by the neuron's largest |activation| over the study's 36 texts, cached per set of
  vault edits in force (`mcp_scientist.py:1175-1191`): the cache key is the tuple of patch ids, and the clean
  model's table is reused after a ROLLBACK. This is the one evaluator that already keys a cache on the edit set.
- The causal check also runs the same push at k = 0.01, "N1A's bf16 floor" (`causal_check.py:53`), and reports the
  push's KL over the floor's (`mcp_scientist.py:1455-1459`). That is an existing noise-floor design a harness
  should copy for edits.

### 1.2 Behavioural: written text

| measurement | where | inputs | outputs | batches? | same forward? |
|---|---|---|---|---|---|
| `write_rows` | `atlas_ops.py:502-617` | several prompts' ids, per-row `Interventions`, `max_new`, stop ids, optional sampler and `samples_of` | a stream of `Step`s: row, piece, its log-probability (float32 softmax), top-k ids and log-probabilities, the sampler's own chance when sampled | yes: left-padded, one KV cache, each row under its own pushes | yes |
| `POST /steer` (scientist `steer`) | `atlas.py:2247-2372`, `ARCHITECTURE.md:920` | prompt(s), pushes; `prompts` (≤ 8) or `candidates` (≤ 12 sets of pushes); `samples` 1-8 | three voices (up, as it is, down) per prompt, or one row per candidate | yes; at most 6,144 word-pieces over the rows past three rows | yes |
| the census (`take_census`, `write_census_batch`) | `census.py:1940-2350` | a layer's neurons and random controls, a design's (question, dose) cells | one JSON row an answer: text, token ids, each piece's log-probability, top-5 ids and log-probabilities, finish, push size | yes: batches of 64 planned from the whole job list (`plan_batches`) | the unpushed answer sits in its question's first batch |
| text-health figures | `census.distinct4` `:558-561`, `first_departure` `:766-772`, `compute_units`' per-dose block `:672-680` | written token ids | share of distinct 4-grams (low = loops), where an answer first parts from the unpushed one, share cut off at the length limit, mean log-probability of the pieces written | CPU, over stored rows | n/a |
| `larql_vault.parted_at` | `larql_vault.py:167-173` | two answers' ids | first index at which they differ | CPU | n/a |
| `census.writer_floor` | `census.py:1434-1519` | a run's meta | after how many word-pieces an UNPUSHED answer written in that batch shape parts from itself written alone | CPU, reads `experiments/results/atlas_writer_floor*.json` | n/a |

Sampling (`_Race`, `atlas_ops.py:464-490`): one Exp(1) vector over the vocabulary per step and per sample, from a
CPU generator seeded `seed + s`, shared by every row of that sample; each row takes argmax(chances / q). Rows part
only where their chances differ, so this is common random numbers between a pushed and an unpushed row. Because
the generator is on the CPU and seeded by `seed + s` alone, two separate calls with the same seed draw the same
noise step by step (the draw "does not depend on the device", `:466`), which is what an edited-versus-unedited
comparison across two forwards would need. It is not interp_engine's draw: "a seed here does not draw what the same
seed draws in a sweep" (`:535-537`). There is no min-p and no repetition penalty (`:538-539`); a presence penalty at
temperature 0 is refused (`:450-454`).

### 1.3 Internal: activations

| measurement | where | inputs | outputs | interventions? | batch |
|---|---|---|---|---|---|
| `read_neuron` | `atlas_ops.py:643-662` | one text, one (layer, neuron), optional one-row `Interventions` | activation at every word-piece | yes (the read is taken before any push at its own layer, `installed` `:352-354`) | 1 row |
| `read_neurons` | `:693-717` | one text, many (layer, neuron) | `{(l, n): [T]}` from one forward | none | 1 row |
| `activation_peaks` | `:720-745` | one text | each of 60 × 21,504 neurons' max and min past the start mark and where, reduced on the device | none | 1 row |
| `activations_at` | `:748-763` | one text, one position | `[L, N]` float32 | none | 1 row |
| `all_activations` | `:766-785` | one text | `[L, T, N]` float16 on the CPU, 2.6 MB a word-piece | none | 1 row; the `/run` route caps it at 256 positions (`atlas.py:60`) |
| `activation_extremes` | `:814-878` | many texts | each neuron's extremes in the four GeGLU sign regimes, the stream's length and both norms' rms at every word-piece | none | one text a forward |
| `passage_events` | `:926-1014` | corpus windows | each face's k strongest events, one a window, with gate and up values | none | one window a forward; about 0.6-0.8 GB on the device |
| `pushed_reads` | `:1642-1733` | questions, writers, each writer's readers, push lengths | `[W, R, Q, 3, 2]`: reader activations (unpushed, +d, −d) × (last piece, mean over the prompt), and each reader's own gate and up values on the unpushed row | add pushes on the writers | up to 64 rows: one unpushed row then ±d for up to 31 writers; one question a forward so nothing is padded |
| `POST /compare` | `atlas.py:2490-2521` | ≤ 300 neurons × ≤ 24 texts | per neuron and text its max, min and peak score over its usual extreme | none | one forward a text |
| `POST /fires` | `atlas.py:2539-2580` | one text | the neurons firing hardest against their usual range | none | 1 row |
| MCP `test_motif_hypothesis` readings | `mcp_scientist.py:1050-1075` | sentences on two sides, up to 512 neurons | each sentence's last and strongest activation per neuron | none | one forward a sentence, whatever the neuron count |

The scores that divide by a neuron's "usual extreme" (`rel`, `peak_score`, `FIRES_AT`) read the stats index
(`atlas_data/index/act_scale.npz`, 2,000,054 word-pieces), which is built on the unedited model. After an edit the
index is not rebuilt; the atlas only marks affected readings `stale` (`atlas.py:214-270`, `1274`, `1402`,
`1604-1607`).

### 1.4 Internal: wiring read from the weights

| measurement | where | reads | output | cost note |
|---|---|---|---|---|
| `neuron_wiring` | `atlas_ops.py:1535-1592` | `W_down` column of the neuron, every other layer's `W_gate`, `W_up` (or `W_down`), norm weights, layer scalars | top-k links each way: `z` (|g·u| z-scored in the layer pair), `w` (strength |P| f² relative to its strongest), `sign`, `gate`, optionally g and u | every other layer's full matrix is dequantised once; the docs say "a few seconds on the GPU" live (`scientist_methodology.md:157`) |
| `pair_products` | `:1616-1639` | named (writer, reader) pairs | g and u for exactly those pairs | cheap: only the named rows and columns |
| `link_scale` | `:1429-1444` | layer scalars, median rms of each block's output and of the stream entering each block | f[i, j], the across-layer scale | CPU |
| `site_coefficients`, `operating_points`, `use_sign` | `:1495-1532` | a reader's gate and up values | first-order coefficients α = gelu′(g₀)u₀, β = gelu(g₀), and the sign in use | CPU |
| `neuron_head_roles`, `head_wiring`, `HeadSpaces` | `:1736-1917` | attention q/k/v/o weights through Gram matrices | heads a neuron meets by role; neurons a head reads and writes; z-scores | one layer's Gram matrices "tens of milliseconds" (`:1740-1741`) |
| `attention_drive` | `:1336-1423` | one captured forward (`capture_heads`) | a neuron's activation at one unit split exactly over heads, earlier blocks and neurons | one forward plus one mat-vec a layer |
| `theodolite_math.direct_path_source_write`, `direct_path_reader_effects` | `theodolite_math.py:415-461` | the clean forward's `mlp_out` and `resid_mid`, exact RMSNorm Jacobians | linearised effect of a writer on a reader's gate and activation | V7 measured ρ = 0.87 against ablation for the gate, 0.84 for the activation (`:448-449`) |
| `theodolite_math.compute_causal_vector_split` | `:148-211` | resid_pre, resid_mid, resid_post, J++ | attention write, feed-forward write and layer-scalar rescale in lens space | per layer, per position |
| `theodolite_math.pole_write_shares` | `:319-356` | a layer's top poles, `mlp_out` | each pole's write size and the share of the feed-forward write the listed poles explain (10-41% for the top 15 at L20-58) | per position |
| `theodolite_math.contrast_of_state` | `:784-807` | a state, paired word ids | log-odds of one word against another, fp32, on the pair rows only | cheap: 2n unembedding rows |
| `theodolite_math.compute_linear_cka_matrix` | `:977-1022` | `[60, n, 5376]` states | 60 × 60 linear CKA | `[60, n, n]` Gram matrices |

`Weights` reads through the runtime, so `neuron_wiring`, `pair_products`, `write_dirs` and `write_ceiling` all see
a vault edit the moment it is made (`atlas_ops.py:128-131`). The wiring INDEX on disk does not; the route reads an
edited neuron's links live and marks the rest stale (`ARCHITECTURE.md:736-737`).

### 1.5 Statistical machinery

| piece | where | what it does | limits stated in the code |
|---|---|---|---|
| `motif_statistics` (unpaired) | `mcp_scientist.py:596-669` | Welch's t; two-sided p by relabelling the sentences, exact over every split when there are ≤ 10,000, else 10,000 random splits under seed 0 with p = (1 + as large)/(1 + draws) | no judgement under 4 sentences a side (`MOTIF_MIN_N`, `:419`) |
| `paired_statistics` | `:672-738` | mean of paired differences; p by flipping signs, exact over 2ⁿ up to 10,000 | no judgement under 6 pairs (`:420`); the pairing is the caller's word |
| `either_statistics` | `:741-828` | pays for choosing between two readings (last piece, strongest piece): the larger |t| recomputed under every relabelling | pays for those two readings only |
| `benjamini_hochberg` | `:831-846` | BH q over the p's given | "corrects only among the p's given: not for tests run and left out, nor for how the sentences were chosen" (`:836-837`) |
| `_correction` | `:883-946` | q per neuron and reading over the neurons of one call (≤ 512) | "not neurons tested in other calls, not sentences tried and changed" (`:897-899`) |
| `pattern_tests.binom_tail`, `blind_result` | `pattern_tests.py:87-91`, `170-196` | chance of ≥ `right` of n pairs by guessing; scores by how near the miss | a test is answered once (`ARCHITECTURE.md:912`) |
| `pattern_tests.written_result` | `:94-107` | how many texts meant to fire did (≥ 0.5 of the face's extreme), how many near misses did, `apart` (rank-sum share) | reads against the stats index's extreme |
| census ranking `compute_units` | `census.py:610-712` | per unit: Poisson z of its words against the controls' mean, `mass`, `null_pct`, empirical p = (1 + controls at least as strong)/(C + 1), BH q over the run's neurons | no p below 1/(C + 1); about N/(C + 1) neurons beat every control by chance (`:623-624`) |
| `chance_above_all` | `census.py:715-718` | N/(C + 1) | — |
| `shift_scores` | `census.py:721-756` | word-shift of a unit against the controls, every word counted | — |
| random controls `control_dirs` | `census.py:1567-1591` | "isotropic": unit(w_post ⊙ r); "columns": a random mix of the layer's own write directions | a plain isotropic direction is read about 100× louder by the next layer than any neuron's write (`census.py:25-30`) |
| `read_gain` | `census.py:1594-1602` | how loudly the next layer's norm reads neurons against controls; "a fair null has a ratio near 1" | — |
| `null_readers` | `atlas_ops.py:1598-1613` | as many unlinked readers at the same layers as a writer's linked ones | tests size, not sign (`ARCHITECTURE.md:762-764`) |
| held-out halves | `census.py:112-117` (`DESIGN_HALVES`), `retest_of` `:1618` | a unit's words picked on three questions, tested on the other three; a retest must use (question, dose) cells its screen did not ask | a retest on shared cells is refused, because in batches of one size a neuron writes the same answer bit for bit (`census.py:47-50`) |
| kept-back passages | `atlas_index.passage_kept_back` (cited `pattern_tests.py:25-27`) | a face's stored events in odd-numbered corpus windows are withheld from the scientist | — |
| the steer lens's own check | `steer_check.py:49-65`, `95-158` | share of neurons above 95 in 100 controls whose steer-lens word-piece matches a census word, beside five shuffles under seed 0 | CPU only; a layer without a census is "unchecked" |
| `causal_check.spearman`, `p_at`, `linear_regime` | `causal_check.py:394-450` | rank correlation over the vocabulary; P@10 of a label among the hundred pieces raised most; closed-form linearity of a write through the post-feed-forward norm | P@10's bar (3 of 10) was set at k = 60 over 36 texts (`mcp_scientist.py:1243`) |
| the label gate `label_check` | `label_gate.py:164-203` | whether a neuron's lens label may be shown: layer ≥ 40 and the lens and naked-eye labels share at least one of ten | validated only for base-fitted J++ labels of random, unedited neurons at layers 40-57; an edited neuron gets `lens_validated: false` (`:201`) |

### 1.6 Cost per call, where a number is recorded

All on the one RTX 5090 (32,607 MiB; the model and lens rest at 21.2 GiB, `engine_runtime.py:470-471`).

| what | number | where recorded, when |
|---|---|---|
| a decode step, `reference` unpacker | 0.706 s (1 row), 0.713 (32), 0.721 (64), 0.743 (96) | `experiments/results/atlas_unpacker_steps_rtx5090.json`, 2026-10-29 01:48Z; `COMPUTE.md:52-57` |
| a decode step, `compiled` unpacker | 0.091 s (1 row), 0.103 (32), 0.109 (64), 0.126 (96) | same |
| a census-question prefill, `compiled` | 0.10 s (1 row), 0.26 (32), 0.45 (64), 0.60 (96) | same file, `prefill_s` |
| torch's peak above the model for a 64-piece census batch | 0.29 GiB (1 row), 2.4 (32), 4.8 (64), 7.2 (96), `compiled` | same file, `peak_gib` |
| three pushed answers of 80 word-pieces (`/steer`) | 8.3 s with `compiled`; 50-54 s on `reference` | `COMPUTE.md:61-64`, `ARCHITECTURE.md:137-138`, 11 November 2026 |
| `/steer` with five prompts (15 rows); with six candidates (13 rows) | 9.2 s; 8.8 s (`compiled`) | `ARCHITECTURE.md:920`, 11 November 2026 |
| one default heard edit (two 12-piece answers and the edit) | 12.6 s | `ARCHITECTURE.md:605-606`; unpacker and date not stated there |
| the server's recapture, one forward | 1.1 s at 512 positions, 1.3 s at 1,024, 1.9 s at 2,048 | `engine_runtime.py:490-491`, measured 2026-10-18 (before `compiled` existed) |
| one layer of the steer lens (21,504 neurons × 2 signs × 4 contexts), layers above only | layer 45 (14 layers above): 287 s; layer 5 (54 above): 1,181 s; 20.5-21.9 s per layer-forward | `experiments/results/atlas_steer_timing/summary.json`, 29 October 2026, `compiled` |
| the whole steer lens, 60 layers; its dose pilot | 10.5 hours; 38 minutes | `ARCHITECTURE.md:820-821`, 13 November 2026 |
| the wiring check | 29-35 s a layer (512 writers, 4 questions, `compiled`), 59 layers in about 31 minutes | `COMPUTE.md:24`, night of 28-29 October 2026 |
| the weight-only wiring index, every pair of layers | about 4 minutes | `COMPUTE.md:24` |
| a corpus pass of 2,000,054 word-pieces with every activation hooked | stats: 58 minutes (`reference`); passages: 20 minutes (`compiled`) | `COMPUTE.md:23`, 28 October 2026 |
| the writer's floor script | 165.2 s (`compiled`) | `experiments/results/atlas_writer_floor_rtx5090.json`, `seconds`, 2026-10-29 |
| a `break_rel` census while writing | 12.4 answers a second at batch 64, 48 pieces (`compiled`); a whole layer 13.2 hours | `COMPUTE.md:19`, 29-30 October 2026 |
| the pattern-test pilot (176 faces at eleven layers, 21 tested) | the card held for 34 minutes | `ARCHITECTURE.md:839`, 14 November 2026 |
| a causal check of every label at page layers 41-59 | about 2 × 10²⁰ multiply-adds, some 17 days on the 5090 (inferred), 3.7 days of one H100 (inferred) | `COMPUTE.md:26`, `ROADMAP.md:120` |
| a prefill on the CPU (12-core desktop, packed checkpoint, `reference`) | 31-37 s | `ARCHITECTURE.md:782-783`, 29 October 2026 |

Not recorded in anything I read: the wall-clock of one `next_logprobs`, `read_neuron`, `read_neurons`,
`activation_peaks`, `neuron_wiring` or `causal_spot_check` call on the 31B. The causal check's reply carries
`timing_ms` (`mcp_scientist.py:1550-1554`), but no figure from it is written down. From the prefill row above, a
single short-prompt forward with `compiled` is on the order of 0.1 s; that is an inference, not a measurement of
those functions.

Why a step costs the same from 1 to 64 rows: every forward unpacks all 29.3 billion 4-bit weights
(`COMPUTE.md:28-32`); with `reference` the products are under a tenth of the step until the batch passes about a
hundred rows. So batching is nearly free up to memory, and a harness should batch hypotheses, not loop.

---

## 2. Partial recomputation

### 2.1 `forward_from_layer` (`atlas_ops.py:1017-1043`)

What the code does:

- Signature `forward_from_layer(lm, start, hidden, attention_mask, position_ids) -> [B, T, D]`. `hidden` is the
  stream ENTERING layer `start` (leaving `start − 1`). It runs layers `start … last` and the final norm, and returns
  final-normed states at every position.
- It rebuilds what `Gemma4TextModel.forward` builds: the causal mask and the sliding-window causal mask
  (`create_causal_mask`, `create_sliding_window_causal_mask`), the rotary embeddings per layer kind, and an empty
  shared key-value dict (`:1033-1037`).
- It runs WITHOUT a cache (`past_key_values=None`, `:1033`, `:1041-1042`). It caches nothing itself. The caller
  holds the cached stream.
- It refuses a model with per-layer inputs or layers that share keys and values (`:1031-1032`); Gemma-4-31B has
  neither.
- "Hooks on the layers run as in a whole forward" (`:1025`): the per-row push hooks of `installed` and any read
  hook work inside it.
- Checked: `scripts/check_atlas_index.py:376-378` compares it with the whole forward "from any layer, on
  right-padded rows" at `atol=1e-5`, on the CPU stand-in. I found no record of that comparison on the 31B itself.

Its one caller is the steer-lens build (`atlas_build.py:775-836`): the four contexts are read to the stream leaving
layer l ONCE with a forward hook (`:793-798`), then for each batch of 64 neurons the pushed copies of that stream
(`h_l + d`, both signs) are concatenated with the unpushed rows and only layers `l + 1 …` run (`:816-824`). The
unpushed rows are in the same forward (`:816`, `:828`).

### 2.2 Could an edit at layer L be judged by rerunning only layers L and above?

Yes, for single-forward readings, and nothing new is needed in `forward_from_layer` itself:

- An edit to any weight of layer L (its `down_proj`, `gate_proj`, `up_proj`, or its attention) leaves the stream
  entering layer L untouched. Capture that stream once per batch of prompts (a forward pre-hook on `lm.layers[L]`,
  or the forward hook on `lm.layers[L − 1]` that `build_steer_layer` uses), then call
  `forward_from_layer(lm, L, hidden, att, pos)` with the edit in force. The steer build starts at `l + 1` because
  its push is added to the stream LEAVING l; an edit needs `start = L`.
- Cost scales with the layers rerun. Measured: 287 s for 14 layers against 1,181 s for 54 layers of the same row
  count, 20.5 against 21.9 s per layer-forward (`experiments/results/atlas_steer_timing/summary.json`). An edit at
  layer 45 costs about a quarter of a whole forward; an edit at layer 5 saves almost nothing.
- Because the function returns states at every position, logits at every position of a fixed text are available
  from one call, so a teacher-forced per-position KL over a held-out text can be computed at the partial cost. No
  such evaluator exists yet (section 4).

What it does not support:

- **Writing.** There is no key-value cache, so it cannot continue a text. `write_rows` runs the whole model with its
  own cache (`atlas_ops.py:562-563`, `:614-615`). A "write from layer L" would need new machinery: the cached
  keys and values of layers below L for the prompt are still valid under an edit at L, but nothing reuses them.
- **Edited and unedited in one forward for a general column edit.** The overlay is module-wide. Today the choices
  are: (a) two forwards of the same batch shape, one with the overlay and one without; (b) for column scalings and
  head scalings, the per-row `mul` and `HeadScale` hooks, which do put both in one forward (section 0, point 3).
- **The sequence-axis shortcut.** The causal check's frozen prefix (`causal_check.py:180-205`) is a second kind
  of partial recomputation: the prefix once at batch 1, then the last position for every row. It is exact only for
  an intervention at the last position.

On choice (a): same-shape forwards on one card are reproducible. A census batch of 64 written in two separate
processes was identical in every field, and a run taken unbroken matched the same run taken in two shards byte for
byte (`experiments/results/census_determinism/summary.json`, 29 October 2026, `compiled`, layer 45). "On the card a
row's numbers depend on its batch's size, not on which rows share it" (`scripts/census_ties.py:23`). So an unedited
pass and an edited pass of identical shape should differ by the edit alone on the home card; that follows from the
census measurement and was not measured for `forward_from_layer`.

### 2.3 Other things that are cached

- The study's `max |a|` table per set of vault edits in force (`mcp_scientist.py:1175-1191`).
- The four steer contexts' stream leaving the layer, for the length of one layer's build (`atlas_build.py:798`).
- The atlas's live wiring and head readings in small caches keyed by neuron (`atlas.py:1371`, `2815`).
- Nothing caches residual streams across calls for reuse by a later evaluator.

---

## 3. Trust limits already measured

A harness should carry each of these as a machine-readable guard, not as prose.

### 3.1 The lens

- The J++ lens was fitted on the base model and is read on the 4-bit instruct model, where it agrees with the
  model's own average gradients for the pronoun direction at cos 0.15-0.33 at page layers 31-59
  (`README.md:383`, `ROADMAP.md:26-28`). No refit passed R2, R3 or R4 (`ROADMAP.md:44-89`).
- Output labels held up at layers 40-57 and were at chance at the sampled layers 10-30; 31-39 and 58 were not
  sampled (`label_gate.py:3-13`, `32`). The gate's agreement count scored AUC 0.84 as a graded score inside 40-57;
  the instrument's yes/no cut scores 0.64 (7 of the 31 labels it holds back were sound, 160 of the 209 it keeps;
  `label_gate.py:42-65`).
- "The gate was never measured on edited neurons" (`label_gate.py:72-73`).
- The vault's UPDATE and INSERT build the new column from a direction pulled back through this same lens
  (`larql_vault.py:511-521`). Reading the edited column's label through the lens afterwards is therefore reading
  the lens's own output back; this is my inference from the code, not a stated finding. The causal check reads the
  effect "from the model's own odds ... so it does not depend on the lens" (`mcp_scientist.py:1238-1239`).
- About 180 stream dimensions are read through the norms with up to 1,700 times the gain of the rest; "a push built
  from a lens breaks the text" (`scientist_methodology.md:124-125`, `census.py:25-30`).

### 3.2 Push doses

- `break_rel` (±1 and ±2 of the stream's median length) is calibrated at layers 39-51; `soft_ceil` at 38 and 52;
  `match_ceil` at 32-37 (`census.py:268-272`). Each layer was counted in on a passed trial of 1,022 fresh neurons
  against 1,024 controls and then the owner's word (`census.py:253-265`).
- Calibration holds only for answers of 32 or 48 word-pieces for `break_rel`, and 48 alone for the two ceiling
  designs (`census.py:277`).
- Trials that failed: `break_rel` at 38 and 52, `soft_ceil` at 37, `match_ceil` at 31 (`census.py:260-264`).
  Layers 53-58 are not calibrated: the probe at 57 passed by 0.0008 and its lead "could not be told from how the
  text survives" (`census.py:257-258`). No push strength has passed a trial below layer 32
  (`pattern_tests.py:4-5`).
- At ±2 of the stream "the text of neurons and controls alike breaks into loops" (`census.py:223-224`). At layer 38
  the lead "is gone by about three ceilings" (`census.py:172-175`).
- The older doses (`main_rel`, the steer default 0.8) were chosen on Qwen3.5-4B (`census.py:36-38`, `210-213`).
- One nominal dose in the stream's unit is a different push at each layer, by about sixteen-fold across the 31B's
  layers (`census.py:2616-2618`, the depth report's docstring).

### 3.3 The steer lens

From `atlas_data/index/steer_check.json` (written 2026-11-14T17:40:55Z, on the lens built 2026-11-13T23:43:53Z):

| layers | standing | share of well-above-control neurons whose steer word matches a census word | shuffled |
|---|---|---|---|
| 0-31 | unchecked (no census) | — | — |
| 32-38 | weak | 0.07, 0.10, 0.17, 0.13, 0.19, 0.20, 0.33 | 0.02-0.04 |
| 39-52 | agrees | 0.60-0.91 | 0.01-0.06 |
| 53-59 | unchecked (no census) | — | — |

- Below layer 40 part of every reading is the layer's own: of a neuron's twelve word-pieces, 2.1 to 7.5 are shared
  by 2 in 100 or more of the layer's neurons at layers 0-39; 1.03 or fewer from layer 40 up (same file, `shared`).
- Its dose is set a layer at a time to move the next word-piece by the reference's median (0.36 nats over 128
  neurons at layer 45); one dose of 0.8 everywhere moved layer 5 by a median 7.5 nats, where "nearly every neuron
  reads alike" (`atlas_build.py:839-844`).
- "Nothing in it sets the neuron against other pushes, as the census's controls do" (`steer_check.py:8-9`).
- It ranks by gain, so a piece moving between one in a billion and one in a hundred million can top the list
  (`scientist_methodology.md:70-73`).
- A neuron that sits near one value everywhere can show a large steer KL while `mul 0` and `mul 2` barely move the
  next word-piece (`scientist_methodology.md:74-76`). For a weight harness this is the central warning: an add push
  along a column is not the same intervention as changing the column.

### 3.4 Saturation through the norm

- Pushes of 60× a neuron's scale on its activation saturate the post-feed-forward norm at layers 20-50; the default
  is k = 10; the frozen-norm write cannot squash, "but a large one still disturbs layers 20-30"
  (`causal_check.py:11-16`, erratum E47).
- A single neuron's write has a ceiling: however hard it fires alone, the norm turns `a · W_down[:, n]` into
  `w_post ⊙ unit(W_down[:, n]) · sqrt(D)`, scaled by the layer scalar (`atlas_ops.py:192-201`). The steer lens's
  0.8 of the stream "is already 1 to 10 times it at most layers" (`:196-197`).
- A uniform scale of a whole feed-forward block, or of every head of a layer, is undone by the norm that follows:
  only silencing (factor 0) is accepted (`atlas_ops.py:244-249`, `261-264`). The consequence for weight editing:
  scaling all of `W_down` or all of `W_O` at a layer by a positive constant changes nothing.
- `linear_regime` gives, in closed form from the layer's base output, how straight a write of size δ along a column
  stays through the norm (`causal_check.py:394-422`); the causal check reports it per test.
- Layer scalars run from 0.03 to 0.99 on the 31B (`theodolite_math.py:162`).

### 3.5 Batch shape, card and precision

- Rows of one forward are bit-identical; different batch shapes gave KL up to 0.014 (`causal_check.py:6-7`).
- The writer's floor on the home card (`experiments/results/atlas_writer_floor_rtx5090.json`, 2026-10-29, layer
  45, `compiled`, 64 word-pieces): an unpushed answer written inside a census batch of 64 parted from the same
  answer written alone after as few as 22 word-pieces (the snake question); in a batch of 32 after 54 (the free
  question). The three-row `/steer` shape and `write_voice` matched the lone answer on all four questions.
- `/steer` with several prompts (11 November 2026): 6 of the 15 rows of five prompts and 6 of the 13 of six
  candidates parted from their own batch of three before 80 word-pieces, two unpushed rows among them
  (`ARCHITECTURE.md:920`).
- Sampled writing: sample s of eight was the write alone with seed + s for only 3 of 8 (`atlas_ops.py:532-535`).
- Same batch size, same card: bit for bit across processes (section 2.2).
- Another card class (RTX PRO 6000 Blackwell against the home 5090): the first word-piece was the same in
  0.975-0.987 of answers, log-probabilities differed by a median 0.0131-0.0146 and 0.155-0.215 at the 99th
  percentile (`ROADMAP.md:674-677`); unpushed answers parted only at near-ties the home card wrote with probability
  0.41-0.50 (`:677-679`).
- The unpackers (`reference`, `lut`, `compiled`, `resident`) give the reference's bits (`ARCHITECTURE.md:131-148`).
- bf16 logits swamp small effects; the study's measure is an fp32 GEMM on the final-normed state
  (`causal_check.py:8`, `132-139`). `write_rows` chooses from the model's own logits in the model's dtype and
  reports a float32 softmax of them (`atlas_ops.py:493-499`, `569-574`).

### 3.6 Wiring read from the weights

Measured on the 31B on the night of 28-29 October 2026 (512 writers a layer, 59 layers, a quarter of each writer's
ceiling; `ARCHITECTURE.md:772-800`):

- The sign from rest agreed with the measured change at 0.336 of links (0.341 for the null); the sign in use as
  served at 0.418 (0.438). Both are below chance.
- About nine readers in ten sat with the gate shut at the one place measured. Scored at each reader's own gate and
  up values, the sign in use agreed at 0.841 (0.853) and size followed the first-order term at ρ 0.89 (0.87),
  against 0.31 for |P| f².
- Linked readers moved 5.5 times as much as the null's: "the links are real by size."
- So weight-only wiring is a lead about strength; its sign is not to be trusted at an ordinary word-piece.

### 3.7 Tests of when a neuron fires

- Six of 21 pilot faces do not fire on their own strongest passage when only the 60 word-pieces before the peak are
  read; of 176 faces, 30 need more than 60 word-pieces and 8 more than 256, all 8 at layer 26 or later
  (`pattern_tests.py:48-58`).
- Blind test against a passage from anywhere: 177 of 196 pairs with the face's own pattern, 86 with another
  face's. Against near misses: 158 of 196 with its own, 106 with another's (`ROADMAP.md:1340-1345`, 14 November
  2026). "A blind test passed against passages drawn anywhere in the corpus does not show that the pattern is
  enough to fire the face" (`pattern_tests.py:57-58`).
- A face's stored passages can be weak: 50 of the pilot's 196 firing passages read under half the face's extreme
  (`ARCHITECTURE.md:866-867`).
- The peak reading of the motif test counts the start mark; where the start mark is the loudest piece it reads
  nothing about the sentence (`mcp_scientist.py:849-852`, `871-877`).

### 3.8 Multiple comparisons

- With 256 controls and 21,504 neurons about 84 land above every control by chance (`census.py:44-47`).
- Benjamini-Hochberg is applied within one census run and within one motif-test call. "Nothing corrects for tests
  run in separate calls or for sentences tried and changed" (`ROADMAP.md:226-227`).
- The legacy motif verdict gave "CONFIRMED" for one sentence a side (d near 10⁴) and is still sent for old callers
  (`mcp_scientist.py:971-973`, `1081-1099`). A driver must read `selectivity`, not `verdict`.

---

## 4. What is missing for judging an edit

### 4.1 Assemblable from existing functions (a driver, no new mechanism)

| evaluator | built from | what the driver adds |
|---|---|---|
| Next-piece KL on unrelated text, edited against unedited | `causal_check.logits_fp32` and `kl`; `pad_right`; held-out corpus windows (section 5) | two passes of one batch shape (overlay on, overlay off); or one pass with `mul` / `HeadScale` for scaling edits |
| The same at partial cost | `forward_from_layer` from layer L on a stream captured once | a hook to capture the stream entering L; chunking of all-position logits (262,144 × 4 bytes a position a row in fp32) |
| Did the answer change, and did the text break | `write_rows`; `census.first_departure`, `distinct4`, the cut-off share, mean log-probability; `census.writer_floor` to say whether a late departure means anything | a prompt set; storage of the unedited answers in the SAME batch shape |
| A null for an edit | `census.control_dirs` (isotropic through `w_post`, or a mix of the layer's columns), `read_gain`, `chance_above_all`, the empirical p of `compute_units` | applying a control direction as a column edit of the same norm on the same neuron or a matched one |
| Paired statistics over prompts | `paired_statistics` (exact sign-flip p), `benjamini_hochberg` | pairing each prompt's edited and unedited reading; a minimum of 6 pairs |
| Does the edited neuron still fire where it did; do its readers move | `read_neurons`, `/compare`, `pushed_reads` with `null_readers` | before-and-after bookkeeping; `rel` scores must say they use pre-edit extremes |
| Wiring before and after | `neuron_wiring`, `pair_products` (both read the edit live) | a diff |
| Is the new column's write squashed by the norm | `linear_regime`, `Weights.write_ceiling` | evaluating it at observed activations for a column change rather than an activation push |
| A held-out split | the census's two halves of three questions; the passages' odd windows | reuse of the idea for edit prompts; nothing enforces it for edits today |
| A seven-question probe of one edit | `experiments/results/vault_example_edit/probe.py` (edit with voice, ROLLBACK, next question) | it talks to the server over HTTP, one question at a time |

### 4.2 Needs new machinery

1. **A per-row weight-edit hook**, so that an arbitrary column (or row) replacement and the untouched model are rows
   of one forward. The arithmetic already exists for the CPU kernel: a column overlay adds
   `x[..., i] · (v − W[:, i])` to the layer's output (`int4_cpu.with_overlay_terms`, `backend/int4_cpu.py:136-163`).
   `Interventions` has no slot for it (`atlas_ops.py:267-279`: adds, acts, muls, heads only).
2. **A multi-token evaluator**: teacher-forced log-likelihood or per-position KL of a fixed text under an edit.
   The server's sweep can read a given answer back teacher-forced (`capture_for_sweep`'s `answer_token_ids`,
   `engine_runtime.py:1541-1543`) and `scripts/verify_control_vector.py:14-18` does it in one forward of three rows,
   but neither is a reusable function over a corpus.
3. **A capability or regression suite.** Nothing in the repository scores a task (no question-answering accuracy,
   no code test, no benchmark). The only proxies are the text-health figures of section 1.2.
4. **Paraphrase generalisation.** No paraphrase sets exist on disk; `/steer`'s `prompts` (≤ 8) is the only
   multi-prompt mechanism and it is for pushes.
5. **Sequential-edit stability.** The vault keeps one overlay per neuron and its ROLLBACK clears every overlay at
   once (`larql_vault.py:556-567`). No evaluator follows drift over a series of edits. Index-derived units (the
   add push's p50, each neuron's extremes, the passages, the steer lens, the wiring index) are built on the
   unedited model and are only flagged stale.
6. **A ledger of every test run**, so that multiple comparisons across calls are paid for. The code is explicit
   that it does not do this (section 3.8). An agent that iterates quickly is exactly the case it leaves open.
7. **Writing from layer L** with the lower layers' cached keys and values (section 2.2).
8. **A noise floor for edits.** The causal check's k = 0.01 floor is for pushes; the analogue for an edit (the same
   pipeline with an edit of negligible size, or an overlay that changes no weight) does not exist.
9. **Edits other than one `down_proj` column.** Row overlays on `gate_proj` and `up_proj` are honoured by the
   forward but "none is written today" (`engine_runtime.py:365-366`); no evaluator is specific to them, and the
   label gate, the causal check and the voice all assume the one-column edit.

---

## 5. Data on disk that could seed evaluators

| data | where | size and shape | caveat |
|---|---|---|---|
| The study's 36 texts (12 chats with begun answers, 12 prose, 12 code) | in code: `causal_check.STUDY_TEXTS`, `causal_check.py:64-101` | 36 short texts, each read at its last piece | the causal check's reference set; `max |a|` is sized on them, so they are not held out from it |
| The census questions | in code: `census.PROMPTS` and `MORE_PROMPTS`, `census.py:93-110` | 4 + 6 chat questions; `break_rel` asks six in two halves of three | their unpushed answers are stored in every run's rows (`kind: "baseline"`) and in `volunteers/home_canaries/` |
| The steer lens's four contexts | in code: `atlas_ops.py:1923-1924` | 2 chat turns, 2 plain openings | the lens was measured on these |
| The corpus windows | `atlas_data/index/passages.d/windows.npy` | `[3914, 512]` int32, 2,000,054 word-pieces of fineweb-edu and ultrachat; `unigram.npy` beside it | "the corpus's own text: keep it in this folder, never commit it" (its `meta.json`); odd-numbered windows are the kept-back fold |
| Each neuron's usual range and the stream's median length | `atlas_data/index/act_scale.npz` | `pos`, `neg`, `pp`, `nn`, `pn`, `np` `[60, 21504]`; `resid_q`, `mlp_rms_q`, `mid_rms_q` `[60, 3]` | built on the unedited model |
| Each face's stored passages | `atlas_data/index/passages.d/{pos,neg}_*.npy` | 16 strongest and 6 sampled events a face | "not validated as a label" (`ARCHITECTURE.md:909`) |
| Whole-layer censuses with random controls | `atlas_data/census/runs/L{36,40-50}-*.jsonl` and their `.units.jsonl`, `.table.npz` | 21,504 neurons and 2,048 controls a `break_rel` layer, 24 answers a unit | calibrated only where section 3.2 says |
| Columns-kind controls for layer 45 | `atlas_data/census/nulls/L45-break_rel-columns*` | 2,048 controls | layer 45 only |
| Calibration ladders | `atlas_data/census/runs/L{20,30,40,45,50,57}-calib_{rel,ceil}.*`; curves in `experiments/results/atlas_calibration/` | 64 neurons + 64 controls a layer | — |
| The steer lens and its check | `atlas_data/index/steer_lens.npz`, `steer_check.json`, `steer_dose.json` | twelve pieces each sign for every neuron; per-layer dose and KL | section 3.3 |
| The measured wiring | `atlas_data/index/wiring_check.npz`, `wiring.npz` | reads under ±d for 512 writers a layer; 24 links each way a neuron | section 3.6 |
| The writer's floor | `experiments/results/atlas_writer_floor_rtx5090.json` | four questions × six batch shapes, 64 pieces | layer 45's settings, one card |
| The pattern-test pilot | `experiments/results/pattern_tests_pilot/` (`patterns.json`, `written.json`, `blind.json`, `blind_near.json`, `behind.json`, `behind_results.json`) | 21 faces' patterns; 63 texts written to fire and 63 near misses; 196 kept-back pairs | one writer wrote the patterns and the texts (`ROADMAP.md:1309-1311`) |
| One edit probed on seven questions | `experiments/results/vault_example_edit/probe.json` | layer 59 neuron 7830, before and after, first-step KL, the watched word | the only stored before-and-after record of a vault edit I found |
| The study's N1B corpus | `experiments/results/N1B_corpus.jsonl` | 1,218 contexts with token ids | generated chats and other sources; provenance in the study's report |
| The study's causal caches | `experiments/results/N1A_cache/causal_k{A,B}_L{10..57}.pt`, `N1A_causal_top2000.pt` | per sampled layer | pushes at k = 60 and 180, the regime E47 warns about |
| E3's 24 chats and E3's average state | `experiments/results/E3_lens_validation_chats.json`, `E3_background_mu.npy` | 24 chats; `[60, 5376]` | — |
| Label-gate ground truth | `experiments/results/N1C_gate_cut.json`, `N1C_labels.npz` | 240 faces at layers 40-57 with causal P@10 | random unedited neurons |
| Saved questions | `~/.local/share/theodolite/saved_prompts.json` (outside the checkout) | 0 questions and 1 system prompt today | the owner's shelf, not a test set |
| Decode-step and memory figures | `experiments/results/atlas_unpacker_steps_rtx5090.json` | twelve rows: three unpackers × four batch sizes | 64-piece census batches |

---

## 6. Where a document and the code differ (each checked on both sides)

1. **The attention / feed-forward split.** `ARCHITECTURE.md:180-181` gives
   `Δattn = (resid_mid − resid_pre) Jᵀ` and `Δffn = (resid_post − resid_mid) Jᵀ`. The code computes
   `s·(mid − pre)`, `post − s·mid` and a separate rescale term `(s − 1)·pre`, with s the layer scalar
   (`theodolite_math.py:180-188`), and its docstring says the older form "folds the rescale ... into the 'FFN
   vault' vector, and reported the attention write 1/s too large" (`:170-172`). The document shows the pre-fix
   formula.
2. **Pattern tests on the 31B.** `ARCHITECTURE.md:868-869` says the near-miss blind test is "not yet run on the
   31B" and `:879-880` that the behind-its-own-passage endings "wait for the card". `ROADMAP.md:1291-1345` records
   both run on the 31B on 14 November 2026 (17:43Z to 17:58Z), and the last commit (`e6ac64f`) says so.
   `pattern_tests.py:48-58` describes the earlier pilot only.
3. **The voice's "float32 distributions".** `ARCHITECTURE.md:623-624` calls the first-step KL one "of the first
   word-piece's float32 distributions". The code takes `F.log_softmax(st.logits.float())` of the writing loop's
   logits (`engine_runtime.py:1727`): a cast to float32, not the fp32 GEMM the causal check uses. The sentence is
   not false, but the two KLs are not the same measurement.
4. **`ARCHITECTURE.md` §4.5 against `causal_check.py` and `causal_spot_check`**: I compared the request table, the
   computation and the reply keys and found no difference.

---

## 7. Not established

- The wall-clock of a single `next_logprobs`, `read_neuron`, `activation_peaks`, `neuron_wiring` or
  `causal_spot_check` call on the 31B (no recorded figure found).
- Peak memory of the causal check's 180-row study forward.
- Whether `forward_from_layer` has been compared with a whole forward on the 31B itself (the stand-in check is at
  `scripts/check_atlas_index.py:376-378`).
- Whether a `mul 0` push with `spare_bos=False` and a vault DELETE give bit-identical rows on the GPU. They are the
  same column change (`ARCHITECTURE.md:1774`, `1783-1784`), but one zeroes the activation and the other the weight.
- The dtype of the logits the voice's writing loop yields (interp_engine was not read).
- Which unpacker the 12.6 s heard edit ran on.
- Whether any evaluator has been run on a series of edits in force together.
