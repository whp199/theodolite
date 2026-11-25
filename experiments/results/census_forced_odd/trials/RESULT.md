# Exploratory `forced_odd` Results (Layers 55–59): Registration Hashed Locally, Not Pushed First

> **Exploratory Status**: These runs (`trials/` and `whole/`) are **trials whose registration was hashed locally at `2026-11-24T23:52:42Z` in `prereg/prereg_at`, NOT committed or pushed prior to execution** (first committed together with outputs in `5452734` at `2026-11-25T03:44:58Z`; original metas recorded base commit `16570312` lacking the new code). Furthermore, `L59` and `L55` were piloted beforehand, two of the three stated gates (`F2.2` and `F3`) were vacuous by construction, and all `21,504` neurons per layer at `L55–L59` have now been read. Nothing in this record makes layers `55–59` census-confirmed, and no retrial using fresh neurons is possible at `L55–L59`.

---

## 1. Main Empirical Finding: Agreement with the Existing Weight Lens (`L55–L59`)

At layers `55–59`, `forced_odd` primarily confirms and agrees with the naked-eye weight-lens labels (`lens.npz` `eye_pos` / `eye_neg` top-12) that the atlas already displays. Because the readout centers log-prob differences across the entire vocabulary, averages over `96` fixed chat continuation positions, and RMS-normalizes over all `172,064` `3+`-letter pieces (`LEX`) weighted equally with no base-probability weighting, median KL shifts are small (`0.0075–0.0362` nats across `L55–L59`) and shown top pieces reflect each neuron's direct logit image rather than probable next tokens the model would generate.

Computed on CPU from `whole/L{55..59}_forced_odd.npz` and `index/lens.npz` (`scripts/analyze_forced_lens_agreement.py`, saved in `whole/lens_agreement.json`):

| Layer | `Dual05` Named | Named `Pos>=1` | Named `Pos>=3` | Named `Both>=1` | Named `Both>=3` | Named `Top1 in eye+` | Shuffled `Pos>=1` | Unnamed `Both>=1` | Named `KL50 / KL90` (nats) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **55** | `8,086` | `79.7%` (`0.7972`) | `41.8%` (`0.4184`) | `96.0%` (`0.9598`) | `78.1%` (`0.7806`) | `58.1%` (`0.5814`) | `0.04%` (`0.0004`) | `82.0%` (`0.8195`) | `0.0362 / 0.1537` |
| **56** | `9,997` | `83.5%` (`0.8347`) | `46.8%` (`0.4679`) | `97.1%` (`0.9710`) | `82.2%` (`0.8218`) | `61.2%` (`0.6120`) | `0.03%` (`0.0003`) | `84.0%` (`0.8401`) | `0.0346 / 0.3095` |
| **57** | `10,486` | `85.1%` (`0.8507`) | `44.5%` (`0.4449`) | `96.7%` (`0.9670`) | `80.0%` (`0.8004`) | `60.8%` (`0.6077`) | `0.03%` (`0.0003`) | `84.9%` (`0.8494`) | `0.0238 / 0.3481` |
| **58** | `15,441` | `86.0%` (`0.8599`) | `49.8%` (`0.4984`) | `96.4%` (`0.9640`) | `82.5%` (`0.8249`) | `64.3%` (`0.6430`) | `0.04%` (`0.0004`) | `83.2%` (`0.8316`) | `0.0123 / 0.4461` |
| **59** | `10,845` | `91.7%` (`0.9174`) | `77.0%` (`0.7704`) | `99.5%` (`0.9948`) | `96.7%` (`0.9670`) | `83.4%` (`0.8337`) | `0.21%` (`0.0021`) | `97.8%` (`0.9779`) | `0.0075 / 0.1924` |

Key observations from `whole/lens_agreement.json`:
- **Positive sign agreement with zero flips**: In `79.7–91.7%` of named neurons (`0.03–0.21%` shuffled), at least one of the top-5 shown `+` forced pieces appears in `eye_pos[:12]`, and `Pos>=3` holds in `41.8–77.0%` of cases (`Both>=1` across `+` and `-`: `96.0–99.5%`, `Both>=3`: `78.1–96.7%`). Sign flips (`+` forced piece appearing in `eye_neg[:12]`) were `0.0%` at every layer `55–59`.
- **Even unnamed units share lens pieces**: At `L59`, `97.8%` (`Both>=1`) and `94.3%` (`Both>=3`) of unnamed neurons also share replicated pieces with their lens labels (the readout acts as an unembedding/final-norm tail check at L59).
- **Worked examples where forced top words are English cognates of CJK `eye_pos` pieces**:
  - `L55 n=19328`: `eye_pos=[' 熱', ' Hot', 'Hot', ' hot', 'heat']` -> forced `[' Hot', ' hot', 'Hot', ' hotter', 'heating']`
  - `L55 n=8870`: `eye_pos=[' terbuka', ' abiertos', ' abiertas', '开放', ' откры']` -> forced `[' откры', ' terbuka', ' відкри', ' otev', 'open']` (`is_direct=False` due to non-LEX CJK/short filtering)
  - `L55 n=19957`: `eye_pos=[' 신', '신', ' tín', ' sin', '信号']` -> forced `[' signals', ' сигнала', 'sin', ' Signals', 'signals']`
  - `L56 n=8706`: `eye_pos=['高', 'high', ' yüksek', ' wysok', ' high']` -> forced `[' high', 'high', ' highest', ' High', ' wysok']`
  - `L58 n=5546`: `eye_pos=['计划', '計劃', '計画', ' planificación', ' plans']` -> forced `[' planned', ' rencana', ' Plans', 'Plans', ' planificación']`
  - `L59 n=7510`: `eye_pos=['愛', 'love', ' love', 'Love', '爱']` -> forced `[' любовь', 'Love', 'love', ' love', ' Love']`
  - `L59 n=14407`: `eye_pos=[' jadi', ' deviennent', '變成', '变成', ' Jadi']` -> forced `[' jadi', ' deviennent', ' Jadi', ' become', ' became']`

---

## 2. Methodological & Procedural Disclosures

1. **Registration status & prior peeking (`L55` and `L59`)**:
   - Rules were hashed locally at `2026-11-24T23:52:42Z` (`prereg/SHA256SUMS`), not committed or pushed before running. Commit `5452734` later added both registration and results together, and original trial/whole JSONs recorded `git_commit = 16570312` (which lacked the `forced_odd` code).
   - `census_gap_pilots/forced_odd_summary.txt` had **already** run a `forced_odd` pilot on `L55` (`1,022` neurons vs `1,024` columns, `DoseC = 0.799`, `T_auc = 0.700`, `FDP05 = 290`), making `RULES.md`'s earlier `"Zero Peeking at L55–L58"` phrase inaccurate for `L55`.
   - At `L59`, the method initially failed on the `rng(11)` `1,022`-neuron pilot (`T_auc = 0.472`, `FDP05 = 94` at `DoseC = 0.550`) and was retuned (`kl_p90 = 0.25`, `proj/LEX_L2_cent`) on those same `1,022` `L59` neurons (`T_auc = 0.931`, `FDP05 = 813`, `Dual05 = 573`) before setting the registration rules.
   - The combined `L55–L59` trial hit an unlogged transient desktop GPU timeout on `L59`, after which `L59` was re-run separately and `trials_summary.json` was assembled across runs.
2. **Vacuous `F2.2` and `F3` gates**:
   - **`F2.2` (`Null05 = 0.00 mean, 0 max`)**: Splits the control pool half-against-half, so exchangeable random units require ~20 of 512 half-controls to beat all other 512 controls; it cannot detect a mismatched control distribution and reads `0` by construction.
   - **`F3` (`Half C = 1.000 / 1.000`)**: `backend/forced_odd.py:_eval_half_c_against_ctl` scored each neuron on its OWN A/B chosen pieces while scoring controls on those SAME unchosen pieces. Even at `L29` in the pilot where `FDP05 = 0`, asymmetric Half-C gave `1.000`. It has been replaced in code by like-for-like `eval_half_c_matched` / `score_half_c_matched` (tested in `scripts/check_forced_odd.py`, where relabelled controls score `auc = 0.4512`, `p05 = 0.0469` vs `1.0` under the legacy bug). Only `F1` tested anything, and `F1` reflects heavy-tail concentration above Gaussian/column/isotropic maxima rather than validated next-token causality.
3. **L59 cross-pool null mismatch**:
   - Because isotropic directions align more strongly with kurtosis near unembedding than write columns (`iso_vs_cols_auc_T = 0.6942` trial, `0.7047` whole), **`288 of 1,024` (28.1%)** isotropic controls in the `L59` trial and **`596 of 2,048` (29.1%)** in whole `L59` cleared `FDP <= 0.05` against the `columns` null (`Col05 = 16,577` vs `Dual05 = 10,845` at `L59`). The reverse cross-pool test (columns vs isotropic) was not computed.
4. **Equal weighting over `172,064` `LEX` pieces & near-constant OLS adjustment**:
   - Weighting all `172,064` `3+`-letter pieces equally without base-probability weighting elevates rare/arbitrary scripts even while median KL is ~`0.01–0.04` nats.
   - Across `L55–L58`, the 5-column log-KL OLS adjustment (`r_ab - r_ab_res` against columns) spans only `0.71–0.83` (`p10–p90` width `0.05–0.14`; `L59` `p10–p90` `[4.130, 4.306]`), leaving neuron KL outside control range without an `N3`-style in-range gate.
5. **Removal of `circuit_mediated`**:
   - Earlier outputs labeled `29,018` dual-named neurons across `L55–L59` (`L55: 3,910`, `L56: 4,064`, `L57: 7,531`, `L58: 8,345`, `L59: 5,168`) as `circuit_mediated` merely when BH `q > 0.05` on own-lens `3+`-letter pieces. At `L59` (`forward_from_layer(lm, 60, ...)`), no downstream transformer layer exists, making circuit mediation physically impossible by construction; across all five layers `sw_neu` correlated strongly with the fraction of `eye_pos[:10]` falling in the `3+`-letter `LEX` mask (`corr = 0.662–0.713`). The `circuit_mediated` label is removed (`own_lens_match = None` at `L59`).

---

## 3. Observed Trial & Whole Numbers (Exploratory Record)

### Exploratory Trials (`N = 1,022` neurons/layer, `C_cols = 1,024`, `C_iso = 1,024`)

| Layer | Dose (`kl_p90=0.25`) | `KL50 / KL90` (nats) | `rABr / rBAr` vs Cols | `rABr / rBAr` vs Iso | `Col05` | `Iso05` | `Dual05` | `NullC / NullI` (vacuous) | Legacy Asym `Half C` (vacuous) | `OwnLens` (`Direct / Dual05`) | Elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **55** | `1.178` | `0.0362 / 0.1649` | `0.7420 / 0.7370` | `0.7691 / 0.7613` | `403` | `436` | `403` (`39.4%`) | `0.00 / 0.00` | `1.000 / 1.000` | `215 / 403` | `315.7s` |
| **56** | `1.211` | `0.0345 / 0.3666` | `0.7866 / 0.7808` | `0.7920 / 0.7968` | `466` | `522` | `465` (`45.5%`) | `0.00 / 0.00` | `1.000 / 1.000` | `297 / 465` | `287.4s` |
| **57** | `0.9707` | `0.0238 / 0.3514` | `0.7936 / 0.7935` | `0.7781 / 0.7797` | `514` | `598` | `512` (`50.1%`) | `0.00 / 0.00` | `1.000 / 1.000` | `169 / 512` | `253.8s` |
| **58** | `0.5818` | `0.0124 / 0.4681` | `0.8803 / 0.8802` | `0.8973 / 0.8958` | `720` | `768` | `718` (`70.3%`) | `0.00 / 0.00` | `1.000 / 1.000` | `335 / 718` | `225.3s` |
| **59** | `0.2482` | `0.0076 / 0.2150` | `0.9067 / 0.9069` | `0.8046 / 0.8024` | `764` | `475` | `475` (`46.5%`) | `0.00 / 0.00` | `1.000 / 1.000` | `n/a (last layer)` | `218.3s` |

### Exploratory Whole Layers (`N = 21,504` neurons/layer, `C_cols = 4,096`, `C_iso = 2,048`)

| Layer | `Col05` | `Iso05` | `Dual05` (`%`) | `Dual10` | `Iso false vs Cols` | Elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| **55** | `8,166` | `9,114` | `8,086` (`37.60%`) | `9,127` | `0 / 2,048` | `2393.3s` |
| **56** | `10,011` | `11,566` | `9,997` (`46.49%`) | `10,935` | `0 / 2,048` | `2156.8s` |
| **57** | `10,681` | `11,884` | `10,486` (`48.76%`) | `11,410` | `0 / 2,048` | `1894.4s` |
| **58** | `15,441` | `16,678` | `15,441` (`71.81%`) | `16,170` | `0 / 2,048` | `1699.5s` |
| **59** | `16,577` | `10,865` | `10,845` (`50.43%`) | `12,283` | `596 / 2,048` | `1356.9s` |

---

## 4. Calibration Decoding Overlap (`L37, L45, L54`) & Negative Pilots (`L20–L30`)

- **Unconditional vs conditional decoding-census overlap (`calibration_p90/calib_L{37,45,54}.json`, `proj/LEX_L2_cent`)**:
  - `L37` (`checked = 193` forced-named vs columns): among `51` also named by `q05` decoding census, `17/51` (`33.33%`) matched, which is **`17/193` (`8.81%`)** unconditionally across all forced-named neurons (`29/192` = `15.10%` / `15.03%` against all units carrying any `shown_words` in `units.jsonl`).
  - `L45` (`checked = 536`): `283/288` (`98.26%`) conditional on `q05` decoding census; **`283/536` (`52.80%`)** unconditionally across all forced-named (`401/536` = `74.81%` vs all `shown_words`).
  - `L54` (`checked = 553`): `180/225` (`80.00%`) conditional on `q05` decoding census; **`180/553` (`32.55%`)** unconditionally across all forced-named (`242/550` = `43.76%` of candidates vs all `shown_words`).
- **Negative results below Layer 30 (`BRIEF.md` Direction 2)**:
  - `census_gap_pilots/forced_odd_summary.txt` named **`0` neurons** at **`L20`** (`256` neurons/controls, `T_auc = 0.538`), **`L25`** (`256`, `T_auc = 0.488`), and **`L29`** (`1,022`, `T_auc = 0.461`), matching **`L30`** calibration (`0/1,022`, `T_auc = 0.552`). As built, `forced_odd` fails to extract signal below layer 30.
