# Pre-Registration: Controlled Antisymmetric Forced Logit Census (`forced_odd`) at Layers 55–59

Registered on `2026-11-24` (exact UTC timestamp in `prereg_at`, repository commit in `repo_commit`, SHA-256 manifest in `SHA256SUMS`), **strictly before any fresh forward pass or trial readout was executed on layers 55–59**.

Layers are indexed from 0 (`0..59`, matching model indices and trial files). All work is conducted on `google/gemma-4-31B-it-qat-w4a16-ct` (`d_ff = 21,504` per layer) in an isolated worktree without modifying the owner's `atlas_data/` (resolved via `THEODOLITE_ATLAS_DIR`) or `backend/census.py`.

---

## 1. Motivation & Stage-1 Open-Layer Calibration (Zero Peeking at L55–L58)

Standard autoregressive census decoding (`match_things`, 48 generated tokens) failed at L55 and above (`L55` gave `108/1,022` empirical-FDR hits and failed cross-half decoding bars because greedy rollout trajectories compound into the final RMSNorm / unembedding boundary cliff).

Instead of waiting for multi-step rollouts to stay uncorrupted across 48 generated steps, **`forced_odd`** holds fixed 12-chat-prompt + greedy-answer prefixes (`HALVES`: A = `bicycle, walk, chair, kettle, sleep, lamp`, B = `clock, rice, vase, lock, money, blanket`, 16 post-prompt positions each) plus out-of-sample held-out Half C (`bridge, windmill, camera, boot, mug, winter`) and applies antisymmetric `+/-` write-ceiling residual pushes leaving `layer` (sparing BOS).

Calibration and method selection (`cent` vs `proj`, variance scaling vs L2-centering, KL dose targeting, log-KL OLS covariates) were finalized strictly on already-open layers **L30, L37, L45, L54** plus the already-explored **L59_pilot** bugfix check (`rng(11)`), with **zero trial runs executed on fresh L55–L59 neurons** prior to hashing this registration:

| Anchor Layer | Dose (`kl_p90=0.25`) | `KL50/90` (nats) | `rABr` | `rBAr` | `T_auc` | `FDP05` (cols) | `Dual05` | `Null05` (mean) | `HalfC` (`frac_p05`) | `DecMatch(q05)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **L30** (floor) | `0.8858` | `0.17 / 0.24` | `0.550` | `0.547` | `0.552` | `0` | `0` | `0.00` | `-` | `0.055(a)` |
| **L37** (open) | `0.8014` | `0.12 / 0.25` | `0.672` | `0.657` | `0.667` | `193` | `169` | `0.00` | `-` | `0.333(q)` |
| **L45** (open) | `1.225` | `0.13 / 0.26` | `0.776` | `0.768` | `0.770` | `536` | `525` | `0.00` | `1.000` | `0.983(q)` |
| **L54** (open) | `1.421` | `0.06 / 0.33` | `0.830` | `0.821` | `0.826` | `553` | `544` | `0.00` | `1.000` | `0.800(q)` |
| **L59_pilot** (`rng(11)`) | `0.2482` | `0.01 / 0.16` | `0.931` | `0.931` | `0.931` | `813` | `573` | `0.00` | `1.000` | `-` |

Key findings baked into the frozen method:
1. **`kl_p90 = 0.25` nats targeting**: keeps all neurons (including high-leverage tail neurons at L58–L59) in the non-clipping linear logit regime (`KL90 ~ 0.16–0.33`), eliminating the `9–10` nat saturation seen when targeting `kl_p50 = 0.25`.
2. **Vocab-mean centering + radial `z0` temperature projection (`proj/LEX_L2_cent`)**: strips per-position RMSNorm partition shifts (`Δ log Z`) and radial logit-temperature scaling `alpha * z0`, then L2-normalizes centered vectors over `LEX` (non-special `3+`-letter pieces) without inverse-variance weighting (`zscore_var=False`, avoiding zero-variance rare code token blowups).
3. **5-column log-KL OLS residualization**: `[1, log(kl_pos + 1e-4), log(kl_neg + 1e-4), log(odd_l2 + 1e-8), even_l2 / (odd_l2 + 1e-8)]` fit strictly inside controls.
4. **Dual-Null (`columns` AND `isotropic`) requirement**: At L30–L54, `columns` controls are stricter (`iso_vs_cols_auc_T ~ 0.38–0.50`, `IsoF = 0`); at L59 (`0` layers remaining before unembedding), `isotropic` directions align more closely with high-kurtosis unembed directions (`iso_vs_cols_auc_T = 0.70–0.73`). Requiring every named neuron and every layer metric to clear empirical step-up `FDP <= 0.05` against **both** `columns` and `isotropic` controls closes both null failure modes uniformly across L55–L59.

---

## 2. Frozen Trial Configuration (`scripts/forced_odd_census.py trial`)

- **Doses (`kl_p90 = 0.25` ceilings interpolated in log-dose vs log-KL from `../dose_ladder.json`, verified `note: null`)**:
  - **L55**: `1.178`
  - **L56**: `1.211`
  - **L57**: `0.9707`
  - **L58**: `0.5818`
  - **L59**: `0.2482`
- **Fresh disjoint neurons (`N = 1,022` per layer)**: drawn via `draw_neurons.py` (seed `20261124 * 100 + layer`) into `neurons_L55.txt .. neurons_L59.txt`, strictly excluding:
  - Every non-volunteer `atlas_data/census/**/*.meta.json` at that layer (`L55`: `L55-match_things-trial.meta.json`, 1,022 neurons; `L57`: `L57-break_rel-probe.meta.json`, `L57-calib_rel.meta.json`, `L57-calib_ceil.meta.json`, 575 unique neurons; `L56, L58, L59`: none),
  - All `64` dose-ladder neurons from `../dose_ladder.json` at each of `L55..L59`,
  - All `1,022` exploratory `L59_pilot` neurons (`rng(11)`) at `L59`.
- **Fresh controls per layer**: `n_cols = 1,024` `columns` controls (`seed = 5100 + layer`) and `n_iso = 1,024` `isotropic` controls (`seed = 6100 + layer`).
- **Null calibration**: `50` random half-control splits on `columns` (`seed = 500 + layer`) and `50` on `isotropic` (`seed = 600 + layer`), re-fitting mean centering, 5-column OLS residualization, and robust scaling inside every split.
- **Out-of-sample Half C verification**: Uniform random sample (`rng(9000 + layer)`) of up to `300` `Dual05`-passing neurons (or top-50 fallback if `< 20` pass) evaluated on held-out 6 prompts `HALF_C` against `256` `columns` controls and `256` `isotropic` controls (exact leave-one-out centering on Half C).

---

## 3. Pre-Registered Pass / Fail Gates (Applied Independently to Each Layer `55–59`)

A layer **PASSES** if and only if ALL THREE gates (**F1**, **F2**, and **F3**) hold simultaneously under `proj/LEX_L2_cent`:

- **F1 (Dual-Null Empirical-FDR Yield)**:
  At least **`50` of `1,022`** fresh neurons (`>= 4.9%`) individually clear empirical step-up `FDP <= 0.05` against **BOTH** `columns` (`1,024`) **AND** `isotropic` (`1,024`) control pools (`Dual05 >= 50`), with positive replication residuals on both split directions (`r_ab_res > 0` and `r_ba_res > 0`) in both pools.
- **F2 (Cross-Half Layer Separation & Null Calibration)**:
  1. Both residualized cross-half AUCs (`auc_r_AB_res` and `auc_r_BA_res`) are `>= 0.58` against **BOTH** `columns` **AND** `isotropic` controls; AND
  2. Across all `50` half-control null splits on **BOTH** `columns` **AND** `isotropic`, `fdp05_false_mean < 1.0` AND `fdp05_false_max <= 2`.
- **F3 (Preflight Exactness & Half-C Out-of-Sample Replication)**:
  1. Unpushed base answer `min_ans_lp > -0.35` and `forward_from_layer` `max_diff < 1e-2` across Groups A, B, and C; AND
  2. At least `20` neurons passed `Dual05` (`were_dual05_named == True`), and among up to `300` uniformly sampled `Dual05` neurons evaluated on Half C, at least **`80%` (`frac_p05_vs_ctl >= 0.80`)** clear `p <= 0.05` against `256` Half-C `columns` controls AND against `256` Half-C `isotropic` controls.

No gate or bar may be loosened after inspecting any fresh trial output.

---

## 4. Whole-Layer Decision Rule (`scripts/forced_odd_census.py whole`)

- Every layer in `{55, 56, 57, 58, 59}` that **PASSES (F1 & F2 & F3)** proceeds to a full whole-layer streaming `forced_odd` run covering all `21,504` neurons (`42` checkpointed chunks of `512` neurons) against expanded dual controls (`n_cols = 4,096` `seed = 7100 + layer`, `n_iso = 2,048` `seed = 8100 + layer`), `50` null splits on each pool (`seeds 700+L`, `800+L`), and Half-C verification on `300` uniformly sampled `Dual05` whole-layer neurons (`rng(9000 + L)`).
- Any layer that **FAILS** stops at the trial stage; no whole-layer `forced_odd` run is executed for a failing layer.

---

## 5. Display, Diagnostic Tagging & Interpretation Rule

1. **Dual-null gating for displayed words**: Next-piece shifts are displayed **ONLY** for neurons individually clearing `Dual05` (`FDP <= 0.05` vs both `columns` and `isotropic`). Non-passing neurons are never labeled with words.
2. **Honest naming**: Passing `forced_odd` neurons are labeled **`forced next-piece shift (FDR <= 5%, dual-null)`** — explicitly distinct from free-running autoregressive decoding (`L30–L54`) and from input-side corpus firing (`L0–L29`).
3. **Secondary mechanistic tag**: Every neuron additionally carries the corrected own-word unembed diagnostic tag (`W(u_n)` vs control's own naked-eye `W(u_c)`, BH `q <= 0.05` & `sw_neu > 0`), tagging each passing neuron as `[direct_unembed]` or `[circuit_mediated]` without censoring non-direct circuit neurons.

---

## 6. What a Pass Does Not Show

- A pass on `forced_odd` measures a neuron's linear/first-pass antisymmetric effect on fixed chat continuations (`HALVES`, `HALF_C`); it does **not** claim that pushing the same neuron during free-running multi-token decoding avoids entropy collapse or reproduces free-running text, nor does it show what causes the neuron to fire naturally on arbitrary inputs.
- **Amendments**: Once `SHA256SUMS` is generated alongside this file, any post-freeze change to hashed code, neuron lists, doses, or rules invalidates the pre-registration unless explicitly logged before analysis in an `AMENDMENTS.txt` file with reason and timestamp.
