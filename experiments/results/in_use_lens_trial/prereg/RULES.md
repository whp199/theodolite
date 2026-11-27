# Path 1 (`in_use_lens`) Confirmatory Trial at Layers 59, 58, and 57

Fixed on 26 November 2026 before any confirmatory GPU forward pass ran (`prereg_at`, `SHA256SUMS`, `CODE_SHA256SUMS`). Layers are 0-indexed (`59, 58, 57`; pages `60, 59, 58`).

## Design
- **Cohort (`N = 1,024` per layer)**: Frozen `pure_4_4` neuron+face pairs drawn without replacement (`neurons_L<L>.txt`), strictly disjoint from Probe 1 (`probe1_in_use_causal.json`) and from prior `L57` runs (`L57-break_rel-probe`, `L57-calib_rel`, `L57-calib_ceil`).
- **Evaluation Data (`8,192` events/layer)**: Strict `<eos>` document-half split (`ic.build_doc_halves` on `passages.d`): first `4` unique-window `pure_h0` fit events + first `4` unique-window `pure_h1` test events per neuron (`0` shared documents).
- **Intervention & Exact Window Caching**: At peak position `pk`, silence neuron `n` in activation space before `F.linear(..., w_down_l)` (`x_abl[pk, n] = 0`). Measure `eye_pk` probability drop `sum_{w in eye_10}(p_rec[w] - p_pert[w])` on pre-fixed `lens.npz` `eye_pos`/`eye_neg` top-10 IDs. Prefix states `0..pk` are computed cleanly per window batch (`bw=8`), with causal suffix `l+1..59` (`0` blocks at `L59`, `1` block at `L58`, `2` blocks at `L57`) run only over affected `post_{rec, abl, col, iso}`.
- **Dual Paired Norm-Matched Controls & Shuffled Null (`C_col = 1,024`, `C_iso = 1,024`)**:
  - `columns`: 64-column random `W_down` mixture matched to `||w_post * W_down[:, n]||_2` in activation space, scored on its own `label_top_ids` `eye_10`.
  - `isotropic`: `N(0, I)` direction matched to `||w_post * W_down[:, n]||_2` pre-`post_ffn_ln`, scored on its own `label_top_ids` `eye_10`.
  - Shuffled-label null: fixed derangement `perm[i] != i` assigning neuron `i` the `eye_10` IDs of neuron `perm[i]`.

## Pre-Registered Confirmatory Gates (Independent per Layer)
All five gates must pass simultaneously at a layer:
1. **`G1` (`all8` Effect vs Both Controls)**: `eye_pk_all8 AUC >= 0.58` AND `mw_p <= 1e-4` vs BOTH `columns` and `isotropic`.
2. **`G2` (Held-Out `test4` & Fit-Test Stability)**: `eye_pk_test4 AUC >= 0.57` AND `mw_p <= 1e-3` vs BOTH controls, AND Spearman `rho(fit4, test4) >= 0.30`.
3. **`G3` (Dual Step-Up Empirical FDP <= 0.05 Yield)**: Requiring `score > 0`, step-up `FDP <= 0.05` simultaneously against BOTH `columns` and `isotropic` yields **`Dual05 >= 103 / 1,024 (10.0%)`**, with 200-bootstrap `p10 >= 60` and `frac_zero == 0.0` on both nulls.
4. **`G4` (Bidirectional Cross-Pool Exchangeability)**: Treating `isotropic` controls as pseudo-neurons vs `columns` (and vice versa) gives false `FDP <= 0.05` rate **`<= 0.02`** in BOTH directions (`iso->col <= 0.02` AND `col->iso <= 0.02`).
5. **`G5` (Shuffled-Label Null & Bit-Exact Sanity)**: Shuffled-label `AUC in [0.45, 0.55]` and `fdp05 == 0` vs `columns`; batch-0 eager hook check passes `noop_kl_same_batch < 1e-5`, `max_kl_alg_vs_hook_same_batch < 1e-5`, `max_layer_rec_diff < 0.05`, and `col`/`iso` post-norm median ratios in `[0.90, 1.15]`.

## What a Pass Shows — and Does Not Show
- **Shows**: Silencing naturally firing neurons reduces predicted probability on their static unembedding weight labels (`lens.npz`) significantly more than both `columns` and `isotropic` norm-matched perturbations, generalizes to unseen half-1 documents, and passes cross-pool null exchangeability.
- **Does Not Show**: Multi-step generative semantics beyond weight-lens vocabulary tokens, nor causality below `L57`. Opening any layer onto the volunteer census or main atlas remains an explicit owner decision.
