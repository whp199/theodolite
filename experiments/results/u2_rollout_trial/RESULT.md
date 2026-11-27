# Path 2 (`u2_rollout`) Confirmatory Trial Result

- **Branch / Commit**: `in_use_prereg_trials` (`fcaa21c9a0f765920906cf55fae61ddce5ad4ca3`, `AMENDMENTS.txt` recorded prior to any verdict write)
- **Model**: `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`)
- **Design**: Upward `U2` (`match_things`) rollout walk across `L55 -> L56 -> L57`, evaluating `N = 1,022` fresh neurons/layer paired against `C_col = 1,024` `columns` (`seed=21`) and `C_iso = 1,024` `isotropic` (`seed=22`) controls (`98,220` joined answers/layer). Layer `L57` excluded `1,595` prior/Path-1 neurons (`0` overlap with Path 1 `L57`).
- **Walk Verdict**: **3-for-3 PASS (`L55 PASS`, `L56 PASS`, `L57 PASS`)** on **BOTH** Track 2A (`U2-T_dec`, `4/4` gates) and Track 2B (corrected `U2-weight_words`, `2/2` gates); `walk.done = 2026-11-27T07:31:30Z` (zero stops).

---

## Confirmatory Verdict Table (`L55 -> L56 -> L57`)

| Layer | Walk Outcome | Track 2A (`U2-T_dec`, `4/4` gates) | Track 2B (`U2-weight_words`, `2/2` gates) | Cross-Pool False Rate (2A / 2B) | Read Time (`s`) |
|---:|:---:|:---:|:---:|:---:|---:|
| **L55** | **PASS -> Advance** | **PASS** (`94 / 1,022` dual `N>=3`, `C4 = 1.000`) | **PASS** (`177 / 1,022 = 17.3%` dual `q<=0.05`) | `0 / 1,024` (`0.0%`) / `0 / 1,024` (`0.0%`) | `107.9 s` |
| **L56** | **PASS -> Advance** | **PASS** (`134 / 1,022` dual `N>=3`, `C4 = 0.975`) | **PASS** (**`307 / 1,022 = 30.0%`** dual `q<=0.05`) | `0 / 1,024` (`0.0%`) / `0 / 1,024` (`0.0%`) | `122.7 s` |
| **L57** | **PASS (`walk.done`)** | **PASS** (`82 / 1,022` dual `N>=3`, `C4 = 0.937`) | **PASS** (**`382 / 1,022 = 37.4%`** dual `q<=0.05`) | `0 / 1,024` (`0.0%`) / `0 / 1,024` (`0.0%`) | `124.9 s` |

---

## Track 2A (`U2-T_dec` Behavioral Reading + `C4` Held-Out)

| Metric (Pre-Registered Bar) | `L55` | `L56` | `L57` |
|---|---:|---:|---:|
| `dual05_n3_count` (`>= 82`) | **94** | **134** | **82** |
| `vs_col` `N>=3` `fdp05` (`boot p10 >= 50`, `frac_zero=0`) | `110` (`p10=89`, `0.0`) | `151` (`p10=149`, `0.0`) | `108` (`p10=85`, `0.0`) |
| `vs_col` `all-N>=1` `fdp05` (`boot p10`) | `136` (`p10=114`) | `220` (`p10=213`) | `231` (`p10=219`) |
| `vs_col` `N>=3` `share_covered` (`>= 0.50`) | `0.9188` (`939/1,022`) | `0.8640` (`883/1,022`) | `0.7916` (`809/1,022`) |
| `vs_iso` `N>=3` `fdp05` (`boot p10 >= 50`, `frac_zero=0`) | `113` (`p10=104`, `0.0`) | `195` (`p10=186`, `0.0`) | `148` (`p10=148`, `0.0`) |
| `vs_iso` `all-N>=1` `fdp05` (`boot p10`) | `120` (`p10=111`) | `204` (`p10=197`) | `184` (`p10=184`) |
| `vs_iso` `N>=3` `share_covered` (`>= 0.50`) | `0.9726` (`994/1,022`) | `0.9618` (`983/1,022`) | `0.9110` (`931/1,022`) |
| `C4` vs `col`: `fit_named` (`>=40`) / `frac_p05` (`>=0.80`) / `AUC` (`>=0.85`) | `83` / `1.000` / `0.9887` | `200` / `0.975` / `0.9845` | `191` / `0.9372` / `0.9587` |
| `C4` vs `iso`: `fit_named` / `frac_p05` / `AUC` | `100` / `0.960` / `0.9660` | `180` / `0.9889` / `0.9827` | `188` / `0.9415` / `0.9618` |
| Cross-pool false rate (`iso vs col` / `col vs iso`, `<=2%`) | `0 / 1,024` / `0 / 1,024` | `0 / 1,024` / `0 / 1,024` | `0 / 1,024` / `0 / 1,024` |
| Code-symmetry max false (`col` / `iso`, `<=1`) | `0` / `0` | `0` / `0` | `0` / `0` |

---

## Track 2B (Corrected `U2-weight_words`)

| Metric (Pre-Registered Bar) | `L55` | `L56` | `L57` |
|---|---:|---:|---:|
| `whole.dual_q05` (`>= 103`) | **177 (`17.3%`)** | **307 (`30.0%`)** | **382 (`37.4%`)** |
| `whole.q05_vs_col` (`above_max_col`) | `177` (`109`) | `324` (`240`) | `402` (`325`) |
| `whole.q05_vs_iso` (`above_max_iso`) | `234` (`146`) | `307` (`164`) | `382` (`275`) |
| `whole.auc_vs_col` (`>= 0.60`) / `auc_vs_iso` (`>= 0.60`) | `0.6750` / `0.6856` | `0.6969` / `0.7005` | `0.7291` / `0.7339` |
| `half A` / `half B` `auc_vs_col` (`>= 0.60`) | `0.6621` / `0.6547` | `0.6807` / `0.6793` | `0.7177` / `0.7088` |
| `half A` / `half B` `auc_vs_iso` (`>= 0.60`) | `0.6746` / `0.6571` | `0.6852` / `0.6776` | `0.7233` / `0.7036` |
| `rho_AB_neurons` (`>= 0.50`) | `0.7925` | `0.6383` | `0.8536` |
| `whole.shuf_auc_vs_col` (`0.47–0.53`) | `0.4766` | `0.5056` | `0.4861` |
| `mean_own_n` vs `col` vs `iso` | `0.02335` vs `0.00049` vs `0.00030` | `0.05247` vs `0.00027` vs `0.00043` | **`0.09664`** vs `0.00025` vs `0.00034` |
| Cross-pool false rate (`iso vs col` / `col vs iso`, `<=2%`) | `0 / 1,024` / `0 / 1,024` | `0 / 1,024` / `0 / 1,024` | `0 / 1,024` / `0 / 1,024` |
| `lens_overlap_pos_20` | `200 / 200` | `200 / 200` | `200 / 200` |

---

## Key Takeaways

1. **Full pre-registered clearance (`L55–L57`)**: With dual paired controls (`columns` + `isotropic`), `match_things` rollout reading (`U2-T_dec` and corrected `U2-weight_words`) clears every gate across all three layers with **zero** cross-pool false discoveries (`0 / 1,024` everywhere).
2. **`U2-weight_words` doubles from `L55` to `L57` (`17.3% -> 30.0% -> 37.4%`)**: Closer to unembedding, neurons emit their static `lens.npz` `eye_10` vocabulary during free rollout at `284x–387x` control rates (`0.09664` vs `0.00025–0.00034` at `L57`, `325 / 1,022` above the maximum `columns` control). Notice at `L57`, `N>=3`-covered `T_dec` dips to the exact `82 / 1,022` gate boundary (while `all-N>=1` `fdp05` climbs to `231` and `U2-weight_words` reaches `382 / 1,022`), because near-output neurons shift sharply onto specific target tokens (`weight_words`) rather than broad semantic clusters (`>= 3` distinct answer words).
3. **Provenance**: Startup `verify_prereg_integrity` passed on clean `fcaa21c`; every `L{55,56,57}_verdict.json` records `"git_dirty": false`, exact `code_sha256`, and `prereg` metadata. Raw `.jsonl` files remain strictly outside git in `/home/eric/bin/ai/theodolite2/atlas_data/census/trials/u2_rollout/`.
