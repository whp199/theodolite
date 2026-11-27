# Path 2 Extension (`u2_rollout_L58_L59`): Dual-Control `match_things` at Layer 58 & Layer 59 Dose Pilot

- **Branch / Commits**: `swarm-next-steps-author-643f920b` (`prereg/` frozen at `80d8c51`, `L58` verdict + `L59` gate recorded at `9db0f1d`)
- **Model**: `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`)
- **Design**: Dual-control `U2` (`match_things`, `48` answers/unit, `max_new = 48`, `batch = 64`), evaluating `N = 1,022` fresh neurons at `L58` (`0` overlap with Probe 1 and Path 1 `in_use_lens_trial/prereg/neurons_L58.txt`) against `C_col = 1,024` `columns` (`seed=21`) and `C_iso = 1,024` `isotropic` (`seed=22`) controls (`98,220` joined rows, `12/12` bit-exact baselines, `161.8 s` read time).
- **Verdict (`L58`)**: **`pass_either = True`** (**Track 2B `U2-weight_words` PASS `2/2`**, Track 2A `U2-T_dec` `3/4` gates — missed `N>=3` yield `G_Yield` as neurons sharpen onto 1–2 weight words); `walk.L58.done = 2026-11-27T16:55:38Z`.
- **Verdict (`L59`)**: Intentionally gated prior to rollout (`walk.up.blocked_no_doses = 59`) without modifying locked `backend/census.py`; fine-ladder dose pilot completed cleanly (`(0.61, 0.64)` ceilings).

---

## `L58` Confirmatory Metrics (`L58_verdict.json`)

| Track & Gate | Result | Headline Measured Numbers |
|---|---|---|
| **Track 2B (`U2-weight_words`)** | **PASS (`2/2`)** | `G_WW_Lead = True`, `G_WW_Yield = True` |
| `whole.dual_q05` (`>= 103`) | **PASS** | **`483 / 1,022 (47.3%)`** (`q05_vs_col = 499`, `above_max_col = 417`; `q05_vs_iso = 483`, `above_max_iso = 375`) |
| `whole` `AUC` (`>= 0.60`) & `rho_AB` (`>= 0.50`) | **PASS** | `auc_vs_col = 0.7743`, `auc_vs_iso = 0.7704`, **`rho_AB_neurons = 0.8418`**, `shuf_auc_vs_col = 0.4958` |
| Subsets `dual_q05` (`half A` / `half B` / `lower` / `upper`) | **PASS** | `474` / `448` / `453` / `483` (`half A auc = 0.7698/0.7638`, `half B auc = 0.7480/0.7445`) |
| Mean own-word emission rate (`neu` vs `col` vs `iso`) | **PASS** | **`0.15042`** vs `0.00029` (**`518.7x` `col`**) vs `0.00054` (**`278.6x` `iso`**) |
| Track 2B cross-pool false rate (`<= 2%`) & `lens_overlap_pos_20` | **PASS** | `0 / 1,024 (0.0%)` both directions; `200 / 200` |
| **Track 2A (`U2-T_dec`)** | **FAIL (`3/4`)** | `G_Yield = False`, `G_CrossPool = True`, `G_C4 = True`, `G_Sanity = True` |
| `all-N>=1` `fdp05` (`vs_col` / `vs_iso`, `boot p10`) | — | **`424`** (`p10 = 422`, `above_max = 382`) / **`388`** (`p10 = 386`, `above_max = 294`), `T_dec_auc = 0.6781 / 0.8747` |
| `N>=3` `share_covered` (`>= 0.50`) | **PASS** | `0.5117` (`523 / 1,022` `vs_col`) / `0.5000` (`511 / 1,022` `vs_iso`) |
| `N>=3` `fdp05` (`vs_col` / `vs_iso`) & `dual05_n3_count` (`>= 82`) | **FAIL** | `50` (`p10 = 50`) / `224` (`p10 = 222`), **`dual05_n3_count = 42 / 1,022`** |
| `C4` held-out (`vs_col`: `fit>=40`, `p05>=0.80`, `AUC>=0.85`) | **PASS** | `383` / `0.9426` / `0.9602` (`vs_iso`: `374` / `0.9599` / `0.9570`) |
| Track 2A cross-pool (`<= 2%`) & `code_symmetry_check` (`<= 1`) | **PASS** | `0 / 1,024 (0.0%)` both directions; `0` / `0` |

---

## Upward Trajectory Across `L55 -> L56 -> L57 -> L58`

| Metric | `L55` | `L56` | `L57` | `L58` |
|---|---:|---:|---:|---:|
| **Track 2B `pass` (`U2-weight_words`)** | **PASS** | **PASS** | **PASS** | **PASS** |
| Track 2B `whole.dual_q05` (`/ 1,022`) | `177 (17.3%)` | `307 (30.0%)` | `382 (37.4%)` | **`483 (47.3%)`** |
| Track 2B `auc_vs_col` / `auc_vs_iso` | `0.6750 / 0.6856` | `0.6969 / 0.7005` | `0.7291 / 0.7339` | **`0.7743 / 0.7704`** |
| Track 2B `mean_own_n` (`x col` / `x iso`) | `0.02335` (`47.7x` / `77.8x`) | `0.05247` (`194.3x` / `122.0x`) | `0.09664` (`386.6x` / `284.2x`) | **`0.15042` (`518.7x` / `278.6x`)** |
| **Track 2A `pass` (`U2-T_dec`, `N>=3`)** | **PASS** | **PASS** | **PASS** | **FAIL (`G_Yield`)** |
| Track 2A `dual05_n3_count` (`>= 82`) | `94` | `134` | `82` | `42` |
| Track 2A `all-N>=1` `fdp05` (`vs_col` / `vs_iso`) | `136 / 120` | `220 / 204` | `231 / 184` | **`424 / 388`** |
| Track 2A `N>=3` `share_covered` (`col` / `iso`) | `0.9188 / 0.9726` | `0.8640 / 0.9618` | `0.7916 / 0.9110` | `0.5117 / 0.5000` |
| Track 2A `C4` vs `col` (`fit` / `p05` / `AUC`) | `83 / 1.000 / 0.9887` | `200 / 0.9750 / 0.9845` | `191 / 0.9372 / 0.9587` | `383 / 0.9426 / 0.9602` |

**Mechanistic interpretation**: Moving from `L55` to `L58` (`1` block before `lm.norm`), pushed neurons increasingly concentrate their rollout emissions onto 1–2 exact target vocabulary tokens rather than `>= 3` distinct semantic words. Consequently, `U2-weight_words` climbs monotonically (`17.3% -> 30.0% -> 37.4% -> 47.3%`, `518.7x` over `columns` at `L58`) and `all-N>=1` `T_dec` nearly doubles (`231 -> 424` `vs_col`), while `N>=3` coverage drops to `50%` (`dual05_n3 = 42`), dropping Track 2A's `N>=3` gate as `U2-weight_words` takes over near the unembedding layer.

---

## `L59` Gate & Fine-Ladder Dose Pilot

- `walk.sh` intentionally stopped cleanly before `L59` (`walk.up.blocked_no_doses = 59`) because `59` is absent from locked `backend/census.py:MATCH_CEIL` (which was held untouched to preserve `u2_rollout_trial/prereg/CODE_SHA256SUMS` and Kit 8 parity).
- The `18`-rung `fine` ladder dose pilot (`0.29..0.9159` of stream, `128` neurons, `seed = 0`, `atlas_data/census/dose_pilot_L59/{steer_dose,doses}.json`) measured `L59` (`p50 = 9.7925`, `ceilings_in_one_stream = 1.655`, `rising = True`):
  - `1.0` nat target: **`0.61` ceilings** (`0.3680` of stream)
  - `1.4` nats target: **`0.64` ceilings** (`0.3867` of stream)
- If a future phase opens `L59` to `match_things` (`(0.61, 0.64)` in `MATCH_CEIL[59]`), `prereg/neurons_L59.txt` (`1,022` fresh neurons, `0` overlap with `in_use_lens_trial` and Probe 1) is already drawn and hashed.
