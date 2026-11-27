# Path 2 Extension (`u2_rollout_L58_L59`): Dual-Control `match_things` Walk at Layers 58 and 59

- **Branch / Commits**: `swarm-next-steps-author-643f920b` (`prereg/` frozen at `80d8c51`, `L58` verdict at `9db0f1d`, `L59` dose-unlock amendment `(0.61, 0.64)` at `c22c254`, `L59` verdict + `walk.done` at `2606f8e`)
- **Model**: `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`)
- **Design**: Dual-control `U2` (`match_things`, `48` answers/unit, `max_new = 48`, `batch = 64`), evaluating `N = 1,022` fresh neurons/layer (`0` overlap with Probe 1 and Path 1 `in_use_lens_trial/prereg/neurons_L{58,59}.txt`) against `C_col = 1,024` `columns` (`seed=21`) and `C_iso = 1,024` `isotropic` (`seed=22`) controls (`98,220` joined rows/layer, `12/12` bit-exact baselines, `161.8 s` read time/layer).
- **Walk Verdict**: **2-for-2 PASS (`L58 PASS`, `L59 PASS`, `walk.done = 2026-11-27T21:37:55Z`)**, extending the `u2_rollout` upward walk (`L55–L57`) to a **5-for-5 `L55–L59` pass** and closing `L30–L59`.
  - **`L58`**: `pass_either = True` via **Track 2B `U2-weight_words` (`2/2` gates, `483 / 1,022 = 47.3%`)**; Track 2A `3/4` (`424` all-`N>=1`, `42` `N>=3 < 82`).
  - **`L59`**: `pass_either = True` via **Track 2A `U2-T_dec` (`4/4` gates, `112 / 1,022` dual `N>=3`, `C4 = 0.9421 / 0.9622`)**; Track 2B `G_WW_Yield = True` (`432 / 1,022 = 42.3%`, `rho_AB = 0.9578`), `G_WW_Lead = False` solely on asymmetric `shuf_auc_vs_col = 0.4169 < 0.47`.

---

## `L58` & `L59` Confirmatory Metrics (`L58_verdict.json`, `L59_verdict.json`)

| Track & Gate (Pre-Registered Bar) | `L58` (`(1.16, 1.27)` ceil) | `L59` (`(0.61, 0.64)` ceil) |
|---|---|---|
| **Overall (`pass_either`)** | **PASS** (`2A=False`, `2B=True`) | **PASS** (`2A=True`, `2B=False`) |
| **Track 2A (`U2-T_dec`)** | **FAIL (`3/4`)** (`G_Yield=False`) | **PASS (`4/4`)** |
| `dual05_n3_count` (`>= 82`) | `42 / 1,022` | **`112 / 1,022`** |
| `vs_col` `N>=3` `fdp05` (`p10 >= 50`) & `share_covered` (`>= 0.50`) | `50` (`p10=50`), `0.5117` (`523`) | **`117`** (`p10=107`), `0.7055` (`721`) |
| `vs_iso` `N>=3` `fdp05` (`p10 >= 50`) & `share_covered` (`>= 0.50`) | `224` (`p10=222`), `0.5000` (`511`) | **`273`** (`p10=270`), `0.8865` (`906`) |
| `all-N>=1` `fdp05` (`vs_col` / `vs_iso`, `boot p10`) | **`424`** (`p10=422`) / **`388`** (`p10=386`) | `366` (`p10=337`) / `321` (`p10=316`) |
| `T_dec_auc` (`vs_col` / `vs_iso`) | `0.6781` / `0.8747` | `0.5221` / `0.5445` (`n3: 0.4302 / 0.5486`) |
| `C4` vs `col` (`fit>=40`, `p05>=0.80`, `AUC>=0.85`) | `383` / `0.9426` / `0.9602` | **`328` / `0.9421` / `0.9622`** |
| `C4` vs `iso` (`fit` / `p05` / `AUC`) | `374` / `0.9599` / `0.9570` | `304` / `0.9671` / `0.9788` |
| Track 2A cross-pool (`<= 2%`) & `code_symmetry_check` (`<= 1`) | `0 / 1,024` (`0.0%`), `0 / 0` | `0 / 1,024` (`0.0%`), `0 / 0` |
| **Track 2B (`U2-weight_words`)** | **PASS (`2/2`)** | **FAIL (`G_WW_Lead=False`, `G_WW_Yield=True`)** |
| `whole.dual_q05` (`>= 103`) | **`483 / 1,022 (47.3%)`** | **`432 / 1,022 (42.3%)`** |
| `whole` `q05_vs_col` (`above_max`) / `q05_vs_iso` (`above_max`) | `499` (`417`) / `483` (`375`) | `551` (`3`) / `432` (`23`) |
| `whole` `auc_vs_col` / `auc_vs_iso` (`>= 0.60`) | `0.7743` / `0.7704` | `0.7760` / `0.6470` |
| `half A` / `half B` `auc_vs_col` (`>= 0.60`) | `0.7698` / `0.7480` | `0.7750` / `0.7643` |
| `half A` / `half B` `auc_vs_iso` (`>= 0.60`) | `0.7638` / `0.7445` | `0.6639` / `0.6392` |
| Subsets `dual_q05` (`half A` / `half B` / `lower` / `upper`) | `474` / `448` / `453` / `483` | `427` / `435` / `429` / `435` |
| `rho_AB_neurons` (`>= 0.50`) | `0.8418` | **`0.9578`** |
| `whole.shuf_auc_vs_col` (`0.47–0.53`) | `0.4958` (PASS) | **`0.4169` (FAIL)** |
| `mean_own_n` vs `col` vs `iso` | `0.15042` vs `0.00029` (`518.7x`) vs `0.00054` (`278.6x`) | `0.22424` vs `0.00399` (`56.2x`) vs `0.02252` (`10.0x`) |
| Track 2B cross-pool (`<= 2%`) & `lens_overlap_pos_20` | `0 / 1,024` (`0.0%`), `200` | `0 / 1,024` (`0.0%`), `200` |

---

## Full Upward Trajectory Across `L55 -> L56 -> L57 -> L58 -> L59`

| Metric | `L55` | `L56` | `L57` | `L58` | `L59` |
|---|---:|---:|---:|---:|---:|
| **Walk Verdict (`pass_either`)** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS (`walk.done`)** |
| **Track 2A `pass` (`U2-T_dec`, `N>=3`)** | **PASS** | **PASS** | **PASS** | FAIL (`G_Yield`) | **PASS** |
| Track 2A `dual05_n3_count` (`>= 82`) | `94` | `134` | `82` | `42` | **`112`** |
| Track 2A `all-N>=1` `fdp05` (`vs_col` / `vs_iso`) | `136 / 120` | `220 / 204` | `231 / 184` | `424 / 388` | `366 / 321` |
| Track 2A `N>=3` `share_covered` (`col` / `iso`) | `0.9188 / 0.9726` | `0.8640 / 0.9618` | `0.7916 / 0.9110` | `0.5117 / 0.5000` | `0.7055 / 0.8865` |
| Track 2A `C4` vs `col` (`fit` / `p05` / `AUC`) | `83 / 1.000 / 0.9887` | `200 / 0.9750 / 0.9845` | `191 / 0.9372 / 0.9587` | `383 / 0.9426 / 0.9602` | `328 / 0.9421 / 0.9622` |
| **Track 2B `pass` (`U2-weight_words`)** | **PASS** | **PASS** | **PASS** | **PASS** | FAIL (`G_WW_Lead`) |
| Track 2B `whole.dual_q05` (`/ 1,022`) | `177 (17.3%)` | `307 (30.0%)` | `382 (37.4%)` | **`483 (47.3%)`** | `432 (42.3%)` |
| Track 2B `auc_vs_col` / `auc_vs_iso` | `0.6750 / 0.6856` | `0.6969 / 0.7005` | `0.7291 / 0.7339` | `0.7743 / 0.7704` | `0.7760 / 0.6470` |
| Track 2B `mean_own_n` (`x col` / `x iso`) | `0.02335` (`47.7x` / `77.8x`) | `0.05247` (`194.3x` / `122.0x`) | `0.09664` (`386.6x` / `284.2x`) | `0.15042` (`518.7x` / `278.6x`) | `0.22424` (`56.2x` / `10.0x`) |

---

## Mechanistic Notes (`L58` vs `L59`) & `L59` Dose Amendment

1. **Why `L59` Track 2A rebounds while `G_WW_Lead` trips on `shuf_auc_vs_col = 0.4169`**:
   - At `L59` (`0` blocks before `lm.norm`), the fine-ladder doses `(0.61, 0.64)` ceilings keep multi-word `N>=3` semantic coverage high (`0.7055` vs `col`, `0.8865` vs `iso`), restoring `dual05_n3 = 112 / 1,022` and clearing all `4/4` Track 2A gates (`C4 = 0.9421`, `AUC = 0.9622`, `0 / 1,024` cross-pool).
   - Meanwhile, on Track 2B, neurons reach `mean_own_n = 0.22424` (`dual_q05 = 432 / 1,022`, `rho_AB = 0.9578`), and all six `AUC >= 0.60` bars pass (`0.7760 / 0.6470` whole, `0.7750 / 0.6639` half A, `0.7643 / 0.6392` half B), **but `controls` also begin emitting their own `eye_10` tokens directly into the unembedding** (`mean_own_col` jumps `13.8x` from `0.00029 -> 0.00399`, and `mean_own_iso` jumps `41.7x` from `0.00054 -> 0.02252`). Comparing shuffled neuron labels (`shuf_n` on deranged `eye_10`) against `columns` own `eye_10` (`own_c`) reproduces the same asymmetric-null artifact seen in Path 1 (`shuf_auc_vs_col = 0.4169 < 0.47`).
2. **`L59` Dose Pilot & Pre-Run Amendment (`c22c254`)**:
   - The `18`-rung `fine` ladder dose pilot (`0.29..0.9159` of stream, `128` neurons, `seed = 0`, `rising = True`, `experiments/results/census_scaled_design/fine_pilot_L59.json`) measured `L59` (`p50 = 9.7925`, `ceilings_in_one_stream = 1.655`) at **`(0.61, 0.64)` ceilings** (`0.3680 / 0.3867` of stream).
   - Added `59: (0.61, 0.64)` to `backend/census.py:MATCH_CEIL` under `prereg/AMENDMENTS.txt` before any `L59` rollout ran, with `0` changes to `0..58` or `neurons_L{58,59}.txt`.
3. **Provenance**: Both `L58_verdict.json` and `L59_verdict.json` record `"git_dirty": false`, verified `prereg` hashes (`SHA256SUMS` & `CODE_SHA256SUMS`), and `join_rows = 98,220`; raw `.jsonl` rollouts remain outside git in `/home/eric/bin/ai/theodolite2/atlas_data/census/trials/u2_rollout_L58_L59/`.
