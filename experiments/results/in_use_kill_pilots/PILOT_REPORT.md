# On-Manifold ("In-Use") Census Path-Forward: Three-Probe Kill-Pilot Report

**Branch**: `in_use_kill_pilots` (branched from `origin/main` commit `b923b2f`, PR #9 merged)  
**Model**: `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`)  
**Status**: Exploratory pilot complete (`check_in_use_pilots.py` `5/5` pass). **No protected files (`backend/census.py`, `backend/census_volunteer.py`, `volunteers/*`, `atlas_data/*`) modified; no confirmatory run or layer opening started.**

---

## Executive Summary & Pass / Kill Verdicts

Off-manifold weight pushing stops working below `L30` (`1/6`–`1/19` of stream norm, 89.7% of downstream gates shut on generic prompts) and breaks isotropic-null exchangeability above `L54`. On `passages.d` (`94.5%` of GeGLU gates sit naturally open), **73.3%–96.5%** of all non-dead neurons across `L2..L59` satisfy strict `pure_4_4` (`>= 4` pure document-half 0 and `>= 4` pure document-half 1 events). Across the three authorized kill-pilots, the **layer gap splits cleanly into three distinct mechanistic regimes**:

| Candidate Instrument | Layers Tested | Headline Pilot Evidence | Verdict | Projected 5090 Cost / Yield (if scaled) |
|---|---|---|---|---|
| **Context-PMI N-gram** (CPU baseline) | `L2, L8, L14, L20, L25, L29` (`256` faces/layer) | `own_half_tie = 0.454–0.557` vs `other = 0.412–0.497`; `p < 0.01` on `0.4%–2.3%` (`1–6/256`). | **KILL** | n/a |
| **`L1` Mechanical Output (`dp_sp`) & `L30–L54` Bridge** | `L2, L14, L25, L29` (`N=64` vs `C=64` `columns`) | Norm-matched `columns` cut `KL_sp AUC` to `0.576–0.633`; `M2 dp_sp AUC = 0.453–0.546`; `M4 bridge AUC = 0.453–0.569` (`1–3/64` signed at `p<=0.05`). | **KILL** (as standalone early-layer namer / Stage-1 KL filter) | n/a |
| **`U1-lens` (`in_use_lens` — natural peak `eye_10` ablation)** | `L58, L59` (`N=64` vs `C=64` `columns`) | **`L58`**: `eye_pk_all8 AUC = 0.6343`, **`18/64 (28.1%)`** `p<=0.05`, **`13/64 (20.3%)`** `> max ctl`, `1,000x` `p90` drop.<br>**`L59`**: `eye_pk_all8 AUC = 0.6636`, **`17/64 (26.6%)`** `p<=0.05`, **`9/64 (14.1%)`** `> max ctl`, `59x` `p90` drop.<br>Holds on unseen test-4 (`AUC = 0.6104 / 0.6604`, `16/64 = 25%` `p<=0.05`). | **PASS (`L58–L59`)** | **Naive uncached**: `4.9–6.3 GPU-h/layer` (`~790–1,050` `p<=0.05` neurons/GPU-h).<br>**Est. passage-cached `h_l` optimize**: `~0.5–1.5 GPU-h/layer` (`~3,500+` `p<=0.05` neurons/GPU-h). |
| **`U1-lens` vs `U2` Rollout (`L55–L57`)** | `L55` (`N=64` vs `C=64` `columns`) | Single-token `eye_pk_all8 AUC = 0.5625` (`8/64 = 12.5%` `p<=0.05`); `M2 dp_sp = 0.5776`. Four blocks before unembedding weaken single-step logit drop; existing `match_things` (`148/1022` `C4=0.961`) + `weight_words` (`228/1022` `q<=0.05`) capture multi-step effects. | **KILL single-token `U1-lens` at `L55`; route `L55–L57` to `U2`** | `match_things` baseline: `~33 GPU-h/layer` (already timed at `~9.4 ans/s` on 5090; 1-dose half-cost `~17 GPU-h/layer` untrialled). |
| **`L2` Self-Contained Local `in_use_pattern` (`Gemma-4-31B-it` author + prefill `blind_near` + `behind`)** | `Group A` (`21` faces `L2..L56`) & `Group B` (`24` fresh `pure_4_4` faces `L2, L14, L26`) | **`Arm A1`** (`Claude pat + Gemma prefill`): **`0.7449` vs `0.5102`** (`L2: 0.967, L14: 0.846`).<br>**`Arm A2`** (`Gemma pat + Gemma prefill`): **`0.6786` vs `0.4643`** (`L2: 0.833, L14: 0.846`).<br>**`Arm B` (fresh `L2/L14/L26`)**: **`0.7898` vs `0.5669`** (`L2: 0.800, L14: 0.868, L26: 0.704`).<br>**`behind` counterfactuals**: `24/24` held passages `>=0.5x`; clean `>=2/3 & 0/3 quiet` = **`4/8 (50.0%)` at `L2`**, **`3/8 (37.5%)` at `L14`**, **`1/8 (12.5%)` at `L26`**, `quiet false fire = 2/72 (2.8%)`. | **PASS (`L0–L15` strong, esp. `L2–L14`)** / **Weak at `L26+`** | Observed full local pipeline: `~8.5s/face` (`~425 faces/GPU-h`, `~43 GPU-h` per `18.2k`-neuron `L2` layer; **`~160–210` dual-pass faces/GPU-h**, or **`~690 faces/GPU-h`** prefill-only). |

---

## 1. Provenance, Data Hygiene, & Instrumentation Checks

Every JSON artefact on `in_use_kill_pilots` was recorded on a clean git tree (`"git_dirty": false`), uses strict `<eos>` document-half isolation (`ic.build_doc_halves` — zero shared documents between fit and test folds), stores zero raw corpus snippets (verified by `scripts/check_in_use_pilots.py`), and passes bit-level verification:

| Artefact | Commit | Git Dirty | Runtime | Script / Test Coverage |
|---|---|---|---|---|
| `probe2_cpu_context.json` | `3a2fc00` | `false` | `7.62 s` (CPU) | [`in_use_context_cpu_probe.py`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/scripts/in_use_context_cpu_probe.py) |
| `probe1_in_use_causal.json` | `469d5db` | `false` | `868.46 s` (`14.5 m`, 5090) | [`in_use_causal_pilot.py`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/scripts/in_use_causal_pilot.py) |
| `probe3_local_pattern.json` | `2767839` | `false` | `372.94 s` (`6.2 m`, 5090) | [`in_use_pattern_pilot.py`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/scripts/in_use_pattern_pilot.py) |
| CPU Unit Suite (`5/5 PASS`) | `e9fa15d`+ | — | `~12 s` (CPU) | [`check_in_use_pilots.py`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/scripts/check_in_use_pilots.py) |

**Probe 1 bit-exact causal instrumentation (`L2` batch-0 sanity)**:
- `noop_kl_same_batch = 0.0`, `noop_max_lsm_diff_same_batch = 0.0` (identity reconstruction through `forward_from_layer`).
- `mean_kl_alg_vs_hook_same_batch = 0.0`, `max_kl_alg_vs_hook_same_batch = 0.0` (zeroing neuron `n` in activation space before `F.linear(..., w_down_l)` matches an eager `down_proj` pre-hook bit-for-bit once double `bfloat16` rounding and cross-batch unpacking variance were eliminated in commits `f2db3d8`–`497d876`).
- `achieved_ctl_over_abl_post_norm_median = 1.026584` (`post_ffn_ln`-weighted write norm match against 64-column random `W_down` controls).

---

## 2. Probe 2: Corpus Coverage & `Context-PMI` Kill Baseline (`probe2_cpu_context.json`)

Corpus: `passages.d` contains **`3,914` windows × `512` tokens** spanning **`1,785` documents**, partitioned strictly at `<eos>` boundaries into **`1,516` pure half-0 windows** (`pure_h0`) and **`1,553` pure half-1 windows** (`pure_h1`). All counts require non-special token triggers at `>= 0.5x` face extreme, deduplicated to distinct documents/windows.

| Layer | Non-Dead Neurons | `neu_pure44` (`%`) | `face_pure44` | `neu_pure66` (`%`) | `neu_doc44` (`%`) | `Context-PMI` `own_half_tie` vs `other` | `Context-PMI` `p < 0.01` Faces |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **2** | 21,504 | **18,208 (84.7%)** | 27,873 | 12,563 (58.4%) | 19,679 (91.5%) | `0.5336` vs `0.4880` (`1,712` pairs) | `2 / 256 (0.8%)` |
| **8** | 21,504 | **17,206 (80.0%)** | 25,781 | 11,690 (54.4%) | 18,799 (87.4%) | `0.5301` vs `0.4893` (`1,778` pairs) | `6 / 256 (2.3%)` |
| **14** | 21,502 | **15,764 (73.3%)** | 23,081 | 10,704 (49.8%) | 17,439 (81.1%) | `0.5574` vs `0.4974` (`1,716` pairs) | `2 / 256 (0.8%)` |
| **20** | 21,504 | **17,591 (81.8%)** | 26,163 | 12,100 (56.3%) | 19,107 (88.9%) | `0.5286` vs `0.4951` (`1,748` pairs) | `1 / 256 (0.4%)` |
| **25** | 21,504 | **20,753 (96.5%)** | 35,909 | 17,050 (79.3%) | 21,231 (98.7%) | `0.5424` vs `0.4704` (`1,792` pairs) | `4 / 256 (1.6%)` |
| **29** | 21,504 | **20,388 (94.8%)** | 34,214 | 16,228 (75.5%) | 21,069 (98.0%) | `0.4542` vs `0.4118` (`1,747` pairs) | `1 / 256 (0.4%)` |
| **55** | 21,504 | **17,690 (82.3%)** | 26,462 | 11,241 (52.3%) | 19,499 (90.7%) | — | — |
| **58** | 21,504 | **18,322 (85.2%)** | 27,494 | 12,653 (58.8%) | 19,813 (92.1%) | — | — |
| **59** | 21,504 | **20,080 (93.4%)** | 32,816 | 15,614 (72.6%) | 20,871 (97.1%) | — | — |

**Conclusion**: Corpus sample sufficiency is **not** the bottleneck (`73.3%–96.5%` under strict pure windows; `81.1%–98.7%` under doc-level `doc_4_4`), whereas simple local N-gram co-occurrence (`Context-PMI`) captures almost none of what differentiates real neuron firings from matched-token near misses (**KILLED**: `0.4%–2.3%` at `p < 0.01`, near chance). Even before GPU near-miss filtering (which lowers `Context-PMI` further when ambiguous cases are filtered), positional/syntactic/semantic structure beyond 1-gram bag-of-neighbors is required.

---

## 3. Probe 1: Norm-Matched On-Manifold Causal Ablation (`probe1_in_use_causal.json`)

For each layer (`L=[2, 14, 25, 29, 55, 58, 59]`), `N=64` random `pure_4_4` neurons (`4` fit + `4` test pure-doc passages each = `512` sequences/layer) were silenced over `[pk..min(seq-1, pk+3)]` and paired against `C=64` norm-matched `columns` controls (`64`-column random `W_down` mixtures) on the exact same `512` passages and scales.

| Layer | Time (s) | `M1 KL_sp` AUC (`w_p` / `p<=0.05`) | `M2 dp_sp` Mean AUC (`p<=0.05`) | `M3 eye_pk_all8` AUC (`mw_p` / `p<=0.05` / `>max`) | `M3` `p90` Drop (Neu vs Ctl) | `M4 Bridge` `span_signed` / `span_pos` AUC (`p<=0.05`) |
|---:|---:|---|---|---|---|---|
| **2** | 192.9 | `0.5786` (`0.0043` / `8/64 = 12.5%`) | `0.5032` (`1/64`) | `0.4846` (`0.619` / `6/64` / `1`) | `8e-6` vs `2e-6` | `0.5105` (`0/64`) / `0.4812` (`0/64`) |
| **14** | 156.8 | `0.5764` (`0.00026` / `10/64 = 15.6%`) | `0.5461` (`4/64`) | `0.4587` (`0.790` / `4/64` / `0`) | `0` vs `1e-6` | `0.5466` (`3/64`) / `0.5688` (`9/64`) |
| **25** | 144.8 | `0.6260` (`3.1e-7` / `12/64 = 18.8%`) | `0.4531` (`3/64`) | `0.4849` (`0.617` / `5/64` / `1`) | `7e-6` vs `1e-6` | `0.4526` (`2/64`) / `0.4565` (`2/64`) |
| **29** | 112.7 | `0.6328` (`0.00027` / `3/64`, `kl_pk 7/64`) | `0.5425` (`3/64`) | `0.4951` (`0.539` / `4/64` / `0`) | `4e-6` vs `1.1e-5` | `0.5061` (`7/64`) / `0.4851` (`6/64`, `pk_signed 0.517`) |
| **55** | 78.5 | `0.5010` (`0.067` / `4/64`) | `0.5776` (`w_p=0.039`, `ft=0.619`, `2/64`) | `0.5625` (`0.112` / `8/64 = 12.5%` / `5`) | `2.2e-5` vs `3e-6` | n/a |
| **58** | 62.1 | `0.5620` (`0.00074`, `kl_pk p90=0.045 vs 0.007`) | `0.4871` (`4/64`) | **`0.6343`** (`0.0044` / **`18/64 = 28.1%`** / **`13/64 = 20.3%`**) | **`0.00505` vs `5e-6` (`1,000x`)** | n/a |
| **59** | 71.8 | `0.6021` (`3.6e-5`, `kl_pk AUC=0.606, 7/64`) | `0.5959` (`w_p=0.0028`, `6/64`) | **`0.6636`** (`0.00071` / **`17/64 = 26.6%`** / **`9/64 = 14.1%`**) | **`0.00891` vs `0.00015` (`59x`)** | n/a |

### Key Causal Takeaways
1. **Why Pilot 3a overstated early-layer individual KL**: Against a norm-matched `columns` direction at the exact same token position, random `columns` ablations already perturb predictions nearly as much as individual neurons at `L2` (`KL_sp` median `0.00198` vs `0.00146`, `AUC = 0.5786`) and `L14` (`0.00136` vs `0.00128`, `AUC = 0.5764`). Early neurons show distributed/redundant functional write norms, so **single-site `mul=0` KL should NOT be used as a hard Stage-1 filter for `L0–L14`**, even though `wilcoxon_p < 0.005` proves genuine average specialization (`p90` KL `2.2x–4.0x` over controls across `L2..L29`).
2. **`L1` (Mechanical Output / Bridge) KILLED for `L0–L29`**: At `L2–L29`, held-out probability-drop word carry (`M2 dp_sp`: `AUC = 0.453–0.546`) and downstream `L30–L54` bridge carry (`M4 span_signed`: `AUC = 0.453–0.547`, with `pk_signed_mean = 0.487–0.527`) fail to beat norm-matched `columns` controls (`example` bridge hubs mostly hit shared generic `L30` multilingual openers such as `L30-n20453` rather than feature-specific downstream circuits).
3. **`L58–L59` Break Open via `U1-lens` (`in_use_lens`)**: Just `0–1` blocks before unembedding, pre-registered naked-eye `lens.npz` `eye_10` labels predict large causal probability drops at natural firing peaks (`L58`: `13/64` pass `BH q <= 0.10` and beat **all 64** paired controls, `18/64 = 28.1%` at `p <= 0.05`, `span_all8 = 19/64 = 29.7%`; `L59`: `17/64 = 26.6%` at `p <= 0.05`). Because `eye_10` is fixed from static weights prior to inference, evaluation on the 4 held-out test events alone reproduces the signal with zero selection inflation (`L58 pk_test4 AUC = 0.6104, 16/64 p <= 0.05`; `L59 pk_test4 AUC = 0.6604, 16/64 p <= 0.05`).
4. **`L55–L57` Need Rollout (`U2`) Instead of Single-Token `U1-lens`**: Four blocks upstream at `L55`, single-token `eye_10` drops to `AUC = 0.5625` (`test-4 AUC = 0.5061`), confirming that `L55–L57` neurons act cumulatively across generated rollouts (where `match_things` already reached `148/1,022` `C4` replicated and `weight_words` reached `228/1,022` `q <= 0.05` at `L55`).

---

## 4. Probe 3: Self-Contained Local `Gemma-4-31B-it` Patterns (`probe3_local_pattern.json`)

**Zero external API calls**. `Gemma-4-31B-it` performed (a) pattern authoring from `pure_h0` fit snippets (`84.7 s` for 45 faces = **`1.88 s/face`**), (b) single-prefill `log P(a) - log P(b)` `blind_near` judging (`assistant_prefix = ""`, `median_p_ab_mass = 0.998–0.9997`, `94.0 s` for 706 prefills = **`0.133 s/prefill`**), (c) GPU verification of `157/158 (99.4%)` quiet (`<0.25x`) near misses via `_StopAfterLayer(26)` (`28.7 s`), and (d) counterfactual `behind` minimal-pair generation + forward testing on 24 fresh `pure_4_4` faces (`88.0 s`). Total runtime: **`372.9 s` (`6.2 min`)**.

### 4a. Single-Prefill `blind_near` Comparison Across Arms & Layers

| Arm / Cohort | Author -> Judge | Faces / Pairs | `own_share` (`p_binom`) | `other_share` | `own - other` | Faces `p < 0.05` (`other`) |
|---|---|---:|---:|---:|---:|---:|
| Baseline (`blind_near.json`, ref) | Claude pat -> manual Claude | 21 / 196 | `0.8061` (`158/196`) | `0.5408` (`106/196`) | `+0.2653` | `9/21` (`0/21`) |
| **Arm A1 (`Group A`)** | Claude pat -> **local Gemma prefill** | 21 / 196 | **`0.7449`** (`146/196`, `p<1e-8`) | `0.5102` (`100/196`) | **`+0.2347`** | `7/21` (`0/21`) |
| **Arm A2 (`Group A`)** | **Gemma pat** -> **local Gemma prefill** | 21 / 196 | **`0.6786`** (`133/196`, `3.2e-7`) | `0.4643` (`91/196`) | **`+0.2143`** | `6/21` (`0/21`) |
| **Arm B (Fresh `L2/14/26`)** | **Gemma pat** -> **local Gemma prefill** | 24 / 157 | **`0.7898`** (`124/157`, `p<1e-8`) | `0.5669` (`89/157`) | **`+0.2229`** | `9/24` (`1/24`) |

**Per-layer `blind_near` breakdown**:
- **Arm A1** (`Claude pat -> Gemma judge`): `L2 = 0.9667` (`3/3 p<0.05`), `L14 = 0.8462` (`2/3`), `L26 = 0.6875` (`0/3`), `L31 = 0.5667` (`0/3`), `L36 = 0.5862` (`0/3`), `L45 = 0.8077` (`1/3`), `L56 = 0.7826` (`1/3`).
- **Arm A2** (`Gemma pat -> Gemma judge`): `L2 = 0.8333` (`2/3`), `L14 = 0.8462` (`2/3`), `L26 = 0.5625` (`0/3`), `L31 = 0.5333` (`0/3`), `L36 = 0.5517` (`0/3`), `L45 = 0.7308` (`1/3`), `L56 = 0.7391` (`1/3`).
- **Arm B** (fresh `pure_4_4` Cohort):
  - **`L2`** (`8` faces, `50` pairs): `own = 0.8000` vs `other = 0.5800` (`4/8 p<0.05`, **`5/8 >= 75%`**; note `L2-n17569p` scored `4/4` but `n=4` yields min possible binomial `p=0.0625`).
  - **`L14`** (`8` faces, `53` pairs): **`own = 0.8679`** vs `other = 0.5094` (**`+0.3585`**, `4/8 p<0.05`, **`7/8 >= 75%`**; `L14-n5201m` scored `4/4`, `p=0.0625` vs `other = 0/4`).
  - **`L26`** (`8` faces, `54` pairs): `own = 0.7037` vs `other = 0.6111` (`1/8 p<0.05`, `3/8 >= 75%`).

### 4b. Fresh `Group B` Counterfactual `behind` Verification (`24` Faces)

| Layer | Own Held `>=0.5x` | `FIRE` Hits (`/24`) | `QUIET` False Fires (`/24`) | Faces `>=1 Fire & 0 Quiet` | Faces `>=2/3 Fire & 0 Quiet` | `mean_apart` | Union (`blind p<0.05` \| clean `behind`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **2** | `8/8 (100%)` | **`12/24 (50.0%)`** | **`0/24 (0.0%)`** | **`5/8 (62.5%)`** | **`4/8 (50.0%)`** | `0.847` | **`6/8 (75.0%)`** |
| **14** | `8/8 (100%)` | **`10/24 (41.7%)`** | `1/24 (4.2%)` | `3/8 (37.5%)` | **`3/8 (37.5%)`** | `0.834` | **`5/8 (62.5%)`** |
| **26** | `8/8 (100%)` | `3/24 (12.5%)` | `1/24 (4.2%)` | `1/8 (12.5%)` | `1/8 (12.5%)` | `0.750` | `2/8 (25.0%)` |
| **Total** | **`24/24 (100%)`** | `25/72 (34.7%)` | **`2/72 (2.8%)`** | `9/24 (37.5%)` | **`8/24 (33.3%)`** | `0.810` | `13/24 (54.2%)` |

Representative clean causal counterfactual passes authored completely locally (behind held `pure_h0` contexts):
- **`L2-n6200m`**: `dispose of ...` (`0.865–0.891`) vs `dispose the ...` (`0.075–0.220`) — `3/3` fire, `0/3` quiet.
- **`L2-n6282m`**: `in recent years` (`0.948`) vs `in previous/several/later years` (`0.020–0.295`) — `3/3` fire, `0/3` quiet (+ `blind_near 7/7`, `p=0.0078`).
- **`L14-n5201m`**: `join communities/programs/groups` (`0.638–0.876`) vs `join calls/meetings` (`0.190–0.241`) — `3/3` fire, `0/3` quiet (+ `blind_near 4/4`).
- **`L14-n20120p`**: Verilog bitwise assignment `assign out = a | b;` (`0.620`), `c = a & b;` (`0.553`) vs English prose (`0.076–0.363`) — `2/3` fire, `0/3` quiet (+ `blind_near 7/8`, `p=0.0352`).

### 4c. Critical Probe 3 Limitations Caught Before Any Pre-Registration
1. **Binomial power floor from `<= 8` `pure_h1` pairs**: Requiring `pure_4_4` yielded only `4–7` valid `blind_near` pairs on several faces (`L2-n17569p` `4/4`, `L14-n5201m` `4/4`), making `p < 0.05` algebraically impossible (`0.5^4 = 0.0625`). Any `in_use_pattern` pre-registration must select `pure_4_8` or `doc_h1` fallback so every judged face has `>= 8–10` pairs.
2. **`Arm B` `other_share = 0.5669` vs `Arm A2 = 0.4643`**: Cyclic shifting across only `8` same-layer faces left minor variance (`L2-n11m` scored `other = 5/5`); confirmatory runs must use **`K >= 3` disjoint shuffled `other_pattern` nulls** and exact span length symmetry between fire/quiet snippets.
3. **`L26–L36` Local Authoring Drop**: Local single-shot authoring captures `L2–L14` lexical/syntactic rules cleanly (`50%` and `37.5%` clean `behind`), whereas by `L26` (`12.5%` clean `behind`) many features depend on subtle discourse or multi-piece triggers where single-shot local summaries miss the exact boundary.

---

## 5. Pre-Registrable Census Proposals (For Owner Decision)

| Proposal | Target Layers | Core Design | Estimated 5090 Cost | Expected Validated Yield | Key Risk / Decision Needed |
|---|---|---|---|---|---|
| **P1: `in_use_lens` (`U1-lens`) Confirmatory Trial** | **`L58–L59`** (immediately openable candidate) | Natural peak `eye_10` ablation (`N=1,024` vs `C=1,024` paired `columns`), fit-selected vs strict held-out `test-4` events, with `L58/L59` passage-cached `(resid_mid, act_all)` batching. | **`~0.5–1.5 GPU-h/layer`** (est. w/ passage cache; `~5–6 GPU-h/layer` uncached). | **`~25%–28%` of active neurons** (`p<=0.05`, **`~4,500–5,500` neurons/layer**; ~`3,500+`/GPU-h cached). | **Owner policy**: Does weight-lens causal confirmation (`in_use_lens`) open `L58–L59` as "taken" on parity with push, or under a distinct regime badge? Also shares `lens.npz` dependency rather than discovering novel non-unembedding circuits. |
| **P2: Multi-Block Rollout Trial (`U2`: `match_things` per-neuron FDR or `census_weight_words`)** | **`L55–L57`** (bridge above `L54`) | 3-split cohort-gated `match_things` (`~14.5%` `L55` `C4`-replicated cohort in trial) with matched `columns` controls at `L55` (`N=1,024`), OR fresh-text `weight_words` trial at `L55–L57`. | `~33 GPU-h/layer` (2-dose timed), or `~17 GPU-h/layer` (1-dose `-2.42` only, untrialled as primary). | **`~13%–15%`** behavioral rollout cohort (`~2,500–3,000`/layer) or `~22%` weight-word-verified (`~3,800`/layer). | Needs explicit confirmation of **matched `columns` controls** at `L55–L57` (isotropic controls break down as unembedding approaches) and whether 1-dose `match_things` is tested first. |
| **P3: `in_use_pattern` (`L2` Local Cohort Trial)** | **`L2–L14`** first (`L0–L15`), held apart from `L16–L29` | Strict `>=8`-pair `pure_h1`/`doc_h1` `blind_near` (`K=3` shuffled nulls) + physical `behind` counterfactual synthesis (`3+3`), entirely local on `Gemma-4-31B-it`. Pilot on `N=128–256` faces at `L2` and `L14`, or `N=1,024` layer sample. | **`~2.4 GPU-h`** per `1,024` faces (full `18.2k`-neuron `L2` layer: `~43 GPU-h`; `prefill-only`: `~26 GPU-h`). | At observed `L2/L14` rates (`4/8` `p<=0.05`, `3–4/8` clean `behind`): **`~38%–50%`** (`~160–210` dual-pass faces/GPU-h; `~6,000–9,000` in a full layer). | Two open design choices: (i) whether `L0–L15` accepts **local-Gemma reading** vs **API/stronger teacher authoring + local prefill judging**, and (ii) what strategy (`L16–L29`) replaces single-turn authoring where `L26` clean `behind` drops to `12.5%`. |

**Explicit Stops Observed**: No changes have been pushed (`in_use_kill_pilots` is local only), no census registration has been committed or run, and all protected census/volunteer files remain byte-identical to `origin/main` (`b923b2f`).
