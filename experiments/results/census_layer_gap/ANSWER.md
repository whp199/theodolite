# Closing the Census Layer Gap (`L0–L29` and `L55–L59`): Written Analysis & Registrable Designs

> **Disclosure & Timing**: Written **after** the exploratory runs on this branch (`forced_odd` `L55–L59`, `input_census` `L0–L29`, and the 25-layer `census_fdr_sweep` `L30–L54`) and after the PR #9 review. Every figure below comes from committed CPU artefacts or the review's verified counts. **No further GPU runs have been executed**, and nothing below changes `backend/census.py`, `DOSES_CALIBRATED`, volunteer kits, or live `atlas_data/`. Layers are 0-indexed (`0–59`).

---

## 1. Critique of `BRIEF.md` Sections 3 and 4

### 1a. Layer gate vs per-neuron gate at L55
- **What the brief got right**: `BRIEF.md §3a` correctly identified why `L55` failed its trial gate (`lead = 0.685`, `carry A->B / B->A = 0.518 / 0.539 < 0.57` bar) even though a real minority of neurons replicate cleanly (`"distance / distances / distant"`, `"often / sometimes / frequently"`, `"sad / sadly"`): non-replicating neurons plus empty thinking-channel `"thought"` glue dilute the layer-wide carry AUC to chance.
- **What the brief missed**:
  1. **Null-exchangeability hazard**: Switching from a layer gate to a per-neuron empirical-FDR gate assumes that null neurons and controls are exchangeable in the upper tail. Even after stripping `"thought"` glue and `COLLAPSE_RULE` words at fixed `alpha = 0.05` (`L55_trial: 148 / 1,022`, `C4_p05 = 0.961`), whole-layer runs only have `isotropic` controls (`w_post ⊙ r`). While our `L45` column-mix check (`L45-mixes-as-neurons.jsonl`) scored **`0` false names** under `T_dec` (`T_auc = 0.4818`), no column-control decoding runs exist at `L53–L55`, and at `L59` isotropic controls differ strongly from column controls (`288/1,024 = 28.1%` trial and `596/2,048 = 29.1%` whole isotropic controls clear `FDP <= 0.05` against columns under `forced_odd`).
  2. **Overlap with existing weight-lens labels**: `census_weight_words/NOTE.md` and `census_gap_pilots/weight_words_corrected.txt` show `L55` pushes predominantly amplify direct unembedding labels already on the atlas (`corr = 0.662` on `L55` `forced_odd`, `228/1,022` clearing corrected `q <= 0.05` vs `0` at `L29–L30`). Treating a high-layer per-neuron push yield as a pure behavioral census without accounting for direct weight-lens alignment overstates how much *new* knowledge is added.

### 1b. Greedy path forking at L29 — necessary, not sufficient
- **What the brief got right**: Pushing at `L29` flips token `0` in over half of answers (median first departure = `0` across the first `5,000` rows of the `L29` trial), so 48-token greedy rollouts read chaotic continuation branches. Reading at position `0` (`promoted.py`) doubled the raw `p <= 0.01` tail (`22` vs `2` expected, mostly sentence openers `"To"`, `"Since"`, `"Actually"`, `"First"`, `<|channel>`).
- **What our experiments proved missing from that diagnosis**: **Eliminating greedy forking completely still yields `0` named neurons at `L20–L30`.** In `census_gap_pilots/forced_odd_summary.txt`, fixed-prefix `forced_odd` (zero rollout) named **`0` neurons** at `L20` (`256` neurons, `T_auc = 0.538`), `L25` (`256`, `T_auc = 0.488`), and `L29` (`1,022`, `T_auc = 0.461`), and stage-1 calibration named **`0 / 1,022`** at `L30` (`T_auc = 0.552`). Likewise `census_fdr_read.py` named `0` at `L29_trial` (`T_auc = 0.4968`) and `0` at `L30_whole` (`T_auc = 0.548`). Thus path forking masks weak signal at the `L30–L31` transition, whereas below `L30` the fundamental barriers are small write ceilings (`1/6` to `1/19` of stream norm at `L2–L24`, `1/4.7` at `L29`) and distributed early-layer representations whose final-logit image (`weight_words`: `0.533` old / `0.499` corrected at `L29`) carries no recoverable single-neuron word concentration.

### 1c. Soundness of `BRIEF.md §3a` yield counts (`yield_all.jsonl`)
The `§3a` yield table is a useful exploratory sanity indicator but **is not statistically sound as a selection or reporting metric**:
1. **Unadjusted four-way conjunctive `p <= 0.01` without covariate residualization**: Requiring half-A mass, half-B mass, `A->B`, and `B->A` all at `p <= 0.01` against `1,024` controls applies no multiplicity correction across four correlated statistics, ignores survival covariates (`lnw`, `d4`, `cut`, `lp`, `dep`, `lost`), and applies no collapse-word exclusion.
2. **High-variance `0–4 / 1,024` empirical chance baseline**: Rescaling `0`, `1`, `2`, `3`, or `4` surviving controls to `1,022` produces discrete jumps with high sampling noise and no valid FDR bound.
3. **Post-hoc variant switching**: Adding an `"English words only"` column (`L29: 15 (1.0)` vs `11 (4.0)`) after inspecting the table inflates apparent signal-to-noise without prior registration. Our `T_dec` sweep (`backend/census_fdr.py`) fixes covariate residualization, collapse/glue removal, and step-up `FDP <= 0.05` (`L29_trial: 0`, `L30_whole: 0`, `L53_whole: 3,964`, `L54_whole: 6,128`, `L55_trial: 148`), yet remains post-hoc and unregistered.

---

## 2. Ranking of Directions by Honestly Described Neurons per GPU-Hour

*Benchmarks from `BRIEF.md` (~`30 GPU-h` / whole decoding layer, `3 GPU-h` / trial at ~`10.5 answers/s` on one 5090), our CPU/GPU measurements (`forced_odd` whole: `0.38–0.66 GPU-h/layer`; `input_census` CPU index build: `~30s`), and PR #9 review audits:*

| Rank | Direction | Target Layers | Cost per Whole Layer (`5090`) | Expected Honest Yield / Layer | Honest Yield per GPU-Hour | Primary Failure Modes & Validity Risks |
|---:|---|---|---:|---:|---:|---|
| **1** | **Dir 1 (modified): Per-neuron `T_dec` FDR (`alpha=0.05`, `N3` covered) + Open-Layer Registration** | `L30–L54` (re-read), `L55` (and trial `L56–58`) | **`0 GPU-h`** on existing `L30–L54`; `~30 GPU-h` on new layers (`3 GPU-h` trial) | `~1,500–7,000` (`L31–L54`); `~3,100` (`L55`: `14.5%` trial rate) | **`∞`** (`0 GPU-h` on `L30–L54`); **`~100/GPU-h`** (`L55`); **`0`** (`L0–29`) | Post-hoc on `L30–54`; isotropic-only controls; `L35` boundary fragile (`ctl>=t=0`); `6`-q layers lack `C4` holdout; `0` at `L29–30`; `L56–58` may hit rollout cliff. |
| **2** | **Dir 2 (redesigned): Fresh-Text `forced_odd` + Direct-vs-Carried Split + Outcome Check** | `L55–L59` (`L45/53/54` validation) | `~0.4–0.66 GPU-h` (`~6–8 min` trial) | `~8,000–15,400` raw `Dual05`, **but `79.7–91.7%` `Pos>=1` (`96.0–99.5%` `Both>=1`) overlap lens** -> smaller non-redundant/carried share | **`~1,000–4,000` net / GPU-h** (`L55–58`); **`0` incremental over lens at `L59`**; **`0` at `L0–30`** | Heavy-tail kurtosis inflation; `L59` isotropic pool breaks columns null (`29.1%`); rare-`LEX` bias (`0.008–0.036` nat KL); `L59` = pure unembed lens by construction (`0` below `L30`). |
| **3** | **Dir 4 (upgraded): Doc-Disjoint Causal `(-1,0)` Input Triggers + Corpus Precision Scan** | `L0–L29` (`L0–L17` cleanest) | **`0 GPU-h`** table + `~1.5 GPU-h` total (`30` layers in 1 pass) + `~0.5 GPU-h` ablation check | `~8,600–28,600` raw `2+2` faces/layer (`~15–40%` surviving precision gate est.) | **`~50,000+` input descriptions / GPU-h** *(NOT a push census)* | **Not push causality**: current `rand`-pos null gates nothing; `73–87%` top triggers at `L24–29` are common words (`OthFaceNull` `3–6%`); old quiet-control ablation sat near `bf16` floor (`0.0006–0.0008` nats). |
| **4** | **Dir 3 (untested): Downstream Fixed-Text Readout onto `L30–L54` `N3`-Named Targets** | `L20–L29` | `~0.4–0.5 GPU-h` (`~4 min` trial) | Unknown (`0` tested; plausible `5–15%` at `L25–29`) | Unknown (`~1,000–3,000` if `10%` works; `0` if style-dominated) | Borrowed semantics inherit `L30–54` flaws; pushes along `w_post ⊙ W_down` may drive generic downstream hubs rather than concept circuits. Worth a 10-min kill-pilot. |
| **5** | **Dir 6: Answer-Only Decoding Pushes** | `L25–L30` | `~30 GPU-h` (`~3 GPU-h` trial) | Low (`~1–3%`) | `~10–20 / GPU-h` | Preserves question understanding but still suffers `1/4.7–1/19` write ceiling deficit at `L0–29`. |
| **6** | **Dir 5: Gentler Pushes (`< 1.0` nat) at `L29`** | `L29` | `~30 GPU-h` (`~3 GPU-h` trial) | `~0` (`forced_odd` at `KL90=0.25` gave `0`) | `~0 / GPU-h` | Already falsified downward (`edge_things` worsened, `forced_odd` at `kl_p90=0.25` named `0` at `L20, 25, 29`). |
| **7** | **Dir 7: Multi-Sample Averaging (`k >= 5`)** | `L25–L31` | `~150+ GPU-h` | `~2–10%` | `<= 15 / GPU-h` | Inherits memory-bound decoding latency (`0.10–0.13s/step` x `k` x `1.13M` answers); hardware-prohibitive. |

---

## 3. Registrable Designs & Cheapest Kill-Pilots

### Design A (Primary Output-Side Retrial): Fresh-Text Fixed-Context Push Readout + Direct-vs-Carried Split (`L55–L59`, Anchored at `L45/L53/L54`)

Because every one of the `21,504` neurons per layer at `L55–L59` has already been read once under the `A/B/C` prompt set in the exploratory whole runs, **a clean confirmatory trial at `L55–L59` cannot draw fresh neurons**. Instead, registration must use **brand-new text**:

1. **Pre-registered artifact before any run**:
   - Generate `18` brand-new object/concept chat prompts (`A': 6`, `B': 6`, `C': 6`) with fixed unpushed greedy continuations generated once *before* hashing, disjoint from all prompts in `atlas_data/`.
   - Commit AND push `RULES.md`, prompt file, `backend/forced_odd.py`, `scripts/forced_odd_census.py`, and `SHA256SUMS` to remote **`main` (or registration branch) BEFORE executing the first GPU forward pass**. Scripts must verify `sha256sum -c SHA256SUMS` at startup and refuse on mismatch.
2. **What is pushed & read (`N = 1,022` sampled neurons + `C_cols = 2,048` + `C_iso = 2,048`)**:
   - Same exact antisymmetric `+/-` residual pushes leaving `layer` (`L2`-RMS centered vocabulary odd log-prob shift averaged over `16` answer positions per prompt, calibrated on open layers `L45, L53, L54` ONLY — never `L55` or `L59`!).
   - **Probability-weighted + raw LEX readouts**: Record BOTH normalized `LEX` odd shift AND probability-weighted `Delta p_tok = p(+d) - p(-d)` / base-top-20 entry rate so rare ungenerated tokens cannot masquerade as spoken words.
   - **Direct-vs-Carried decomposition (`L55–L58` ONLY; never `L59`)**: On the exact same pushes at `L55–L58`, also run `forward_from_layer(60, h_final ± d)` (no intermediate layers) to obtain `delta_direct`, compute `delta_carried = delta_total - delta_direct`, and score `delta_carried` identically against `columns` and `isotropic` carried-control nulls. Report `carried_share` continuously; **never assign a `"circuit"` label**.
3. **Statistic, controls, and non-vacuous gates (bars fixed on `L45/L53/L54`)**:
   - **G-Yield (`Dual05-N3`)**: `FDP <= 0.05` simultaneously against BOTH `C_cols = 2,048` and `C_iso = 2,048`, **restricted to `N3`-covered KL range** (`>= 153 / 1,022`, `15%`), plus `p_A` & `p_B` residuals `> 0`.
   - **G-CrossPool (replaces vacuous `Null05` half-splits)**: Bidirectional pool exchangeability **`iso_false_vs_cols <= 0.02`** AND **`cols_false_vs_iso <= 0.02`**. (If either exceeds `2%`, as `L59` `iso->cols = 29%`, reject dual-null FDR validity at that layer.)
   - **G-HalfC' (like-for-like matched selection via `eval_half_c_matched`)**: Both neurons and controls pick their own `A'/B'` top-5 pieces and score them on their own unseen `C'`. Gate on neuron-vs-control **`Half-C' AUC >= 0.65`** AND fraction with `p_matched <= 0.05` **`>= 0.25`** (`5x` nominal control rate).
   - **G-Incremental (Outcome validation at open `L45, L53, L54`)**: On `1,022` neurons at `L45, L53, L54`, forced words must predict **held-out decoding census `q <= 0.05` words** AND **causal write KL on `C'`** strictly better than `lens.npz` (`eye_pos/eye_neg`) alone (`>= +0.05` unconditional recall or rank correlation over `lens.npz`).
4. **Sample size & 5090 cost**:
   - Trial (`1,022` + `4,096` controls across `18` fixed texts, including direct+total pass): `~8–10 GPU-min` per layer (`~0.75 GPU-h` across `L55–L59`, plus `~0.4 GPU-h` for `L45/L53/L54` outcome benchmark). Whole layers (`21,504` + `6,144` controls): `~0.6–0.9 GPU-h/layer`.
5. **Cheapest Kill-Pilot (`~15 GPU-min` total)**:
   - Run `256` neurons + `512` columns + `512` isotropic controls on **`L45, L54, L58, L59`** with **6 brand-new texts**:
     - **Kill Criterion 1**: If `L58` or `L59` still exhibits bidirectional cross-pool contamination `> 0.05`, isotropic/column empirical FDR is structurally invalid near unembedding -> do **not** run full `L59`/`L58` retrial as a "5% FDR census".
     - **Kill Criterion 2**: If `L45` and `L54` forced words add `< +0.03` incremental accuracy over `lens.npz` alone at predicting decoding census words, or if `L58` `carried` residual names `< 5%`, abandon fixed-text retrial as a new census and treat `L55–L59` strictly as a **weight-lens check**.
6. **What a pass shows — and does not show**:
   - *Shows*: These neurons produce context-independent shifts beyond two random families across brand-new prompts (`A'/B'/C'`), hold their own `A'/B'` piece ranking on `C'`, and (at `L55–L58`) how much of that signal passes through downstream layers vs directly through final norm/unembed.
   - *Does not show*: Multi-step conversational concepts during free generation (`KL` here is `~0.01–0.04` nats over improbable vocabulary tails unless verified by `Delta p`).

### Design B (Top Input-Side Design + `L25/L29` Downstream Pilot): Registrable Input Precision & Fair Ablation (`L0–L29`)

1. **Corpus-precision + fair ablation (`L0–L29`)**:
   - **Pre-register & push** rules using a **brand-new `2M`-piece corpus draw** (fresh `FineWeb-Edu` / `UltraChat` shard disjoint from the `1,785` documents already inspected in `passages.d`!).
   - Hold out `50%` of documents as `D_train` (select top-1 causal `(-1, 0)` trigger among faces with `>= 2` docs in each `D_train` split) and `50%` as `D_test`.
   - In **one single streaming GPU forward pass over `D_test` (`~1.5 GPU-h` across all 30 layers simultaneously)**, record at every occurrence of each selected trigger (and a frequency-matched same-bin wrong piece) whether `|act(n)| >= 0.5 * extreme(n)`. Gate each face on **held-out precision >= `0.20` AND lift over frequency-matched control >= `3x`** (plus unseen-doc recall >= `0.25`).
   - Run the corrected `scripts/input_ablation_check.py` (already updated in code: random same-layer non-dead neuron control on identical **full-context** passages, neuron-clustered Wilcoxon `N=64`, reporting `bf16` floor `0.00045` nats). Cost: `~0.5 GPU-h` across `4–8` layers.
   - **Cheapest Kill-Pilot (`~15 GPU-min` on `L2, L14, L29`)**: Run fair full-context ablation first. If median peak KL does not beat `2x bf16` floor (`0.0009` nats) and random active same-layer neurons (`p <= 0.001` clustered), drop causal ablation as a validation bar and retain the reading purely as descriptive firing precision/recall on inputs.
2. **Cheapest Direction-3 Kill-Pilot (`~10 GPU-min` on `L25, L29`)**:
   - Push `256` neurons at `L25` and `L29` (`±` ceiling doses on fixed prompts, forward up to `L54`), score `T_dec` on the `94k` `N3`-covered `L30–L54` decoding neurons' activation changes against `512` column controls. **Kill immediately** if `T_dec AUC <= 0.55` or `FDP05 <= 5 / 256`.

---

## 4. Question 4 (`L55–L58`): Per-Neuron Decoding Gate Alone vs Combining with Weight-Word Reading

**Neither should simply replace or absorb the other; they answer two different questions:**

- **Per-neuron decoding gate alone (`T_dec` `alpha=0.05` + `C4` holdout)**: Gives **behavioral, multi-step semantic evidence** (`L55_trial`: `148 / 1,022 = 14.5%`, `C4_p05 = 0.961` on the `102`-neuron 4+4 subset), clean of near-tautological unembedding projection, and is directly comparable in kind to `L30–L54`. However, yield is modest (`~14.5%` at `L55`), `L56–L58` have **no `match_things` trials yet**, and decoding doses near `L56–L58` push close to the final-norm cliff (`> 0.9` stream norm).
- **Weight-word reading (`census_weight_words`) & fixed-text `forced_odd`**: Reads **`228 / 1,022` (`22.3%`, corrected own-word `q <= 0.05`)** on `L55` answers (`0` at `L29–L30`) and agrees strongly with `eye_pos/eye_neg` (`79.7–86.0%` `Pos>=1`, `96.0–97.1%` `Both>=1` at `L55–L58`). By construction, however, its labels overlap what the atlas *already shows* via the weight lens (`corr(LEX, sw) = 0.66–0.71`; `L54` unconditional `q05` decoding overlap is only `32.6%`, `180/553`, vs `80.0%`, `180/225`, conditioned on both naming).
- **Recommended rule for `L55–L58`**:
  1. Do **not** merge weight-word matches into the decoding census's "taken" count without explicit owner authorization.
  2. If `L55–L58` is tested with `T_dec` under `match_things`, keep `T_dec` (`alpha = 0.05`, `N3`, with a column-control null!) as the **push census**, and separately display **weight-lens agreement** (`census_weight_words` / `forced_odd`) as a **"lens-push verification"** (never calling it an independent census, and at `L59` noting explicitly that forced push and weight lens collapse to one unembedding reading).

---

## 5. Question 5 (`L0–L29`): Can a Push Census Work Below Layer 30, and What Stands In?

**Plain answer**: Every push-side output readout built so far **fails below layer 30 (`0` named across `L20–L30`)**:
- Autoregressive decoding (`T_dec`): `L29_trial` = **`0 / 1,022`** (`T_auc = 0.4968`), `L30_whole` = **`0 / 21,504`** (`T_auc = 0.548`).
- Weight-word reading (`census_weight_words/NOTE.md`, `weight_words_corrected.txt`): **`0` `q <= 0.05`** at `L29` (`lead = 0.499` corrected) and `L30` (`lead = 0.549` old).
- Fixed-context logit readout (`forced_odd` pilots & calibration): **`0` named** at `L20` (`0/256`, `T_auc = 0.538`), `L25` (`0/256`, `T_auc = 0.488`), `L29` (`0/1,022`, `T_auc = 0.461`), and `L30` (`0/1,022`, `T_auc = 0.552`).

While an untested **Direction 3 downstream activation readout** (`L20–L29` -> `L30–L54` neurons) could still reveal intermediate feature wiring at `L20–L29` (testable in a 10-GPU-min pilot), **below roughly layer 20–30 single-neuron pushes are `1/6` to `1/19` of the residual stream norm** and mostly alter style/openers rather than writable downstream concepts. A standard output-word census cannot honestly populate `L0–L29`.

**What should stand in on the page (only after owner sign-off and exploratory labelling)**:
- Replace push-census expectations on `L0–L29` with **exploratory input-side firing descriptions**: our recomputed document-disjoint causal `(-1, 0)` tables (`467,498` faces with `>= 2 + 2` occurrences across disjoint documents; pooled unseen-doc recall `33.39%` vs `0.014%` frequency-matched / `1.20%` shuffled, strongest at `L0–L17`), supplemented by `pattern_tests` (`177/196` vs `86` wrong-pattern in prior pilot) and (if registered and passing) held-out corpus **precision**.
- On every `L0–L29` surface where shown, state plainly in plain words: **"Exploratory reading of what makes these neurons fire on corpus text (not a push census and not registered); pushing neurons individually at L20–L30 described 0 neurons."**

---

## 6. Owner Decisions Needed Before Any GPU Run or Atlas Integration

1. **Open Decision 1 (`L30–L54` `census_fdr_sweep/NOTE.md`)**: Keep the current `.units.jsonl` page rule (`29,265` described), adopt exploratory `T_dec` (`N3` `94,237` / `N1` `115,803` at `alpha = 0.05`) on taken layers (with or without column-control runs at `L34–L37`), or leave `census_fdr_sweep` as a research note?
2. **`L55–L59` (`forced_odd`)**: Accept the exploratory lens-agreement finding (`79.7–91.7%` `Pos>=1` and `96.0–99.5%` `Both>=1` overlap with `eye_pos`/`eye_neg` at `L55–L59`, `0` below `L30`) as a **weight-lens check record only** (never calling `55–59` "taken"), OR authorize the **`~15 GPU-min` Design A fresh-text kill-pilot** before any pushed registration?
3. **`L0–L29` (`input_census_L0_L29/NOTE.md`)**: Keep the document-disjoint causal tables as an exploratory record only, authorize the **`~15 GPU-min` fair full-context ablation + Design B precision kill-pilot**, and/or authorize the **`~10 GPU-min` Direction-3 pilot at `L25/L29`**?
4. **Already-published files in `<atlas_data>/census/{forced_odd,input_census,fdr_rescue}/` (`67` files from prior commit)**: Delete those unreviewed files from live `<atlas_data>/census/` until page integration is authorized in Package 4, or retain them in place? (Left completely untouched by this branch revision per review rules.)
