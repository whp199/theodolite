# Exploratory Proposal: Per-Neuron Empirical-FDR Decoding Reader Across All 25 Whole Census Layers (`L30–L54`)

> **Status & Scope**: **Post-hoc exploratory proposal — unregistered, gates nothing, and has NO page integration.** This reader (`backend/census_fdr.py`, `scripts/census_fdr_read.py`) was designed and tuned after whole-layer outputs (`L30–L54`) and trials were inspected. Layers `L34–L37` were never shut; what this proposal addresses is **Open Decision 1** — which per-neuron count a taken decoding census layer reports on the page alongside or in place of `mass > max_ctl_mass & mass_shown > max_ctl_mass_shown`. Any adoption onto the atlas requires the owner's explicit ruling (and a pre-registered, hashed, pushed trial if used as a gate).

---

## 1. Method Changes Fixed in This Proposal

1. **Single fixed level (`alpha = 0.05`) everywhere**: Every layer is evaluated solely at `FDP <= 0.05` (`(N / (C + 1)) * (1 + c_ge) / n_ge <= 0.05`, leave-one-out for controls) against isotropic controls (`C = 2,048` on whole runs, `1,024` on trials). Selective `FDP <= 0.10` switching (used previously on `L36` after `FDP05 == 0`) has been removed; **`L30` and `L36` both yield `0` at fixed `alpha = 0.05`**.
2. **Respect `census.COLLAPSE_RULE`**: Loads each run's `<run>.collapse.json` word list and excludes those collapse words from word counts, replicated word lists, AND the `lnw` survival covariate. Previously, up to `22%` of named neurons relied solely on collapse words; across `L30–L54` (`14,513` dropped whose names depended solely on collapse words, vs `18,483` gained once collapse noise was also stripped from controls and `lnw`), the filter materially alters per-layer counts (e.g., `L34: 4,068 -> 3,517`, `L35: 647 -> 301`, `L37: 3,347 -> 2,596`, `L31: 494 -> 1,304`).
3. **Exact thinking-channel glue filter (including `lnw`)**: Strips leading/trailing `"thought"`-glued tokens (e.g. `"bythought"`, `"thoughtdepending"`) while preserving `ENGLISH_THOUGHT_WORDS` (`"thought"`, `"thoughtful"`, `"afterthought"`, etc.) and any word present in baseline answers, removing stripped glue tokens from both word scoring and `lnw`.
4. **Control bootstrap stability & honest symmetry sanity check**: Reports 200-replicate control bootstrap percentiles (`Boot p10–p90`), `controls_above_thresh` (`Ctl>=t`), and renames half-control splits to `code_symmetry_check` (`0` false positives by construction for exchangeable halves; NOT an empirical-FDR guarantee for neuron-vs-control exchangeability).

---

## 2. Full 25-Layer Results (`L30–L54`, Fixed `alpha = 0.05`, `48`-Piece Reads)

| Layer | Design | `Q` | `Reg q05` | `Reg Named` | `Page Desc` | `T_dec AUC` | `N1 FDP05` | `Boot p10–p90` | `Ctl>=t` | `N3 Cov%` | `N3 FDP05` | `Unfilt05` | `ColDrop` | `C4 p05` | `SymMax` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **L30** | `match_things` | 12 | `0` | `0` | `24` | `0.548` | **`0`** | `0–0` | `0` | `96.4%` | `0` | `0` | `0` | `-` | `0` |
| **L31** | `match_wide` | 12 | `1,893` | `1,443` | `50` | `0.470` | **`1,304`** | `671–1,609` | `5` | `97.6%` | `1,251` | `494` | `142` | `0.969` | `0` |
| **L32** | `match_ceil` | 6 | `2,946` | `2,075` | `470` | `0.524` | **`1,052`** | `661–1,501` | `4` | `97.1%` | `910` | `338` | `58` | `-` | `0` |
| **L33** | `match_ceil` | 6 | `5,576` | `3,600` | `423` | `0.537` | **`2,965`** | `2,328–3,374` | `13` | `97.9%` | `2,824` | `3,070` | `896` | `-` | `0` |
| **L34** | `match_ceil` | 6 | `0` | `0` | `95` | `0.598` | **`3,517`** | `3,408–3,989` | `15` | `98.3%` | `3,427` | `4,068` | `1,154` | `-` | `0` |
| **L35** | `match_ceil` | 6 | `0` | `0` | `12` | `0.533` | **`301`** | `301–1,067` | `0` | `96.1%` | `277` | `647` | `377` | `-` | `0` |
| **L36** | `match_ceil` | 6 | `0` | `0` | `10` | `0.541` | **`0`** | `0–1,096` | `0` | `97.3%` | `0` | `0` | `0` | `-` | `0` |
| **L37** | `match_ceil` | 6 | `2,332` | `2,279` | `147` | `0.561` | **`2,596`** | `2,043–2,843` | `11` | `89.3%` | `1,590` | `3,347` | `1,103` | `-` | `0` |
| **L38** | `soft_ceil` | 6 | `7,003` | `5,623` | `111` | `0.586` | **`4,422`** | `3,966–4,733` | `20` | `85.0%` | `2,290` | `5,330` | `1,257` | `-` | `0` |
| **L39** | `break_rel` | 6 | `436` | `434` | `70` | `0.710` | **`2,120`** | `1,689–2,442` | `9` | `94.5%` | `1,954` | `2,811` | `905` | `-` | `0` |
| **L40** | `break_rel` | 6 | `4,659` | `4,336` | `743` | `0.614` | **`5,370`** | `4,746–5,869` | `24` | `86.4%` | `4,543` | `5,347` | `802` | `-` | `0` |
| **L41** | `break_rel` | 6 | `3,534` | `3,293` | `79` | `0.694` | **`3,914`** | `3,456–4,645` | `17` | `91.7%` | `3,416` | `4,260` | `660` | `-` | `0` |
| **L42** | `break_rel` | 6 | `5,477` | `4,648` | `2,087` | `0.665` | **`5,381`** | `5,221–5,682` | `24` | `88.8%` | `3,795` | `5,702` | `780` | `-` | `0` |
| **L43** | `break_rel` | 6 | `8,070` | `6,469` | `1,909` | `0.695` | **`7,736`** | `7,564–7,959` | `35` | `87.8%` | `5,736` | `7,570` | `631` | `-` | `0` |
| **L44** | `break_rel` | 6 | `9,133` | `7,145` | `4,478` | `0.722` | **`8,746`** | `8,473–8,875` | `40` | `86.2%` | `6,288` | `8,887` | `829` | `-` | `0` |
| **L45** | `break_rel` | 6 | `6,308` | `5,123` | `1,023` | `0.667` | **`6,620`** | `6,452–6,865` | `30` | `90.3%` | `5,043` | `6,538` | `727` | `-` | `0` |
| **L46** | `break_rel` | 6 | `6,830` | `5,636` | `2,156` | `0.697` | **`7,495`** | `7,428–7,686` | `34` | `88.3%` | `5,686` | `7,442` | `834` | `-` | `0` |
| **L47** | `break_rel` | 6 | `7,354` | `5,727` | `2,163` | `0.693` | **`7,503`** | `7,425–7,720` | `34` | `83.7%` | `5,225` | `7,201` | `750` | `-` | `0` |
| **L48** | `break_rel` | 6 | `6,554` | `4,885` | `939` | `0.697` | **`7,497`** | `7,280–7,657` | `34` | `93.3%` | `6,682` | `6,380` | `417` | `-` | `0` |
| **L49** | `break_rel` | 6 | `5,899` | `4,616` | `1,116` | `0.712` | **`7,851`** | `7,594–7,994` | `36` | `93.1%` | `7,033` | `7,024` | `520` | `-` | `0` |
| **L50** | `break_rel` | 6 | `5,914` | `4,519` | `375` | `0.686` | **`7,179`** | `7,051–7,237` | `33` | `91.8%` | `6,054` | `5,690` | `394` | `-` | `0` |
| **L51** | `break_rel` | 6 | `7,145` | `5,126` | `1,781` | `0.648` | **`6,533`** | `6,383–6,582` | `30` | `95.1%` | `5,756` | `5,898` | `586` | `-` | `0` |
| **L52** | `soft_ceil` | 6 | `6,933` | `4,814` | `3,639` | `0.580` | **`5,609`** | `5,497–5,740` | `25` | `97.4%` | `5,318` | `5,090` | `263` | `-` | `0` |
| **L53** | `match_things` | 12 | `3,640` | `3,289` | `1,456` | `0.565` | **`3,964`** | `3,705–4,146` | `17` | `96.9%` | `3,691` | `3,386` | `205` | `0.948` | `0` |
| **L54** | `match_things` | 12 | `5,481` | `4,928` | `3,909` | `0.700` | **`6,128`** | `5,804–6,268` | `28` | `94.3%` | `5,448` | `5,313` | `223` | `0.944` | `0` |
| **Total** | `L30–L54` (`25` layers) | — | `113,117` | `90,008` | `29,265` | — | **`115,803`** | — | — | — | **`94,237`** | `111,833` | `14,513` | — | `0` |

*(Notes on columns: `Reg q05` and `Reg Named` are the 48-piece registered verdict counts for comparison against the 48-piece `.jsonl` read here; at `L36–L43` and `L52`, the registered verdict rule selected the shorter `max_new=32` cut in its `take_shorter` check. `Page Desc` counts neurons in `<atlas_data>/census/runs/*.units.jsonl` whose `mass > max(ctl mass)` and `mass_shown > max(ctl mass_shown)`. `N3 Cov%` and `N3 FDP05` restrict to neurons whose 6 covariates lie inside controls' `[min, max]` range.)*

---

## 3. Additional Diagnostic Checks (Not Whole Census Layers)

| Check | `L` | `N / C` | `T_dec AUC` | `N1 FDP05` | `Boot p10–p90` | `Ctl>=t` | `N3 FDP05` | `C4 p05` | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `L45_cols_mix_null` | 45 | `2,048 / 2,048` | `0.4818` | **`0`** | `0–0` (`frac_zero=1.0`) | `0` | `0` (`99.8%` cov) | `-` | Column-mix controls standing as neurons vs isotropic controls (`L45-mixes-as-neurons.jsonl`) |
| `L29_trial` | 29 | `1,022 / 1,024` | `0.4968` | **`0`** | `0–0` (`frac_zero=0.995`) | `0` | `0` (`96.8%` cov) | `-` | `match_things` trial: `0` named under `T_dec` |
| `L55_trial` | 55 | `1,022 / 1,024` | `0.5750` | **`148`** | `103–157` (`frac_zero=0.0`) | `6` | `133` (`94.8%` cov) | `0.961` | `102` fit at 4+4, `96.1%` clear `p<=0.05` vs paired C4 controls |

---

## 4. Key Caveats & Limitations

1. **Isotropic controls only (no layer-wide column-control null)**: All 25 whole-layer runs use isotropic controls (`w_post ⊙ r`). While the `L45` column-mix null (`L45-mixes-as-neurons.jsonl`) gave **`0` false names** (`T_auc = 0.4818`), no column-control decoding runs exist at `L34–L37` or `L53–L54`. Any false-discovery claim is strictly relative to isotropic random pushes.
2. **N3 covariate coverage gap**: At `L37–L47`, `10–16%` of neurons fall outside the controls' `[min, max]` survival range, so `N1 FDP05` extrapolates OLS residualization outside support (`N1: 2,596` vs `N3: 1,590` at `L37`; `4,422` vs `2,290` at `L38`). If used on a page, **only `N3` covered neurons** should avoid extrapolation, or uncovered neurons must be marked explicitly.
3. **Boundary fragility on low-yield layers**: At `L35`, `301` names sit above `0` controls (`34/2049 = 0.0349 <= 0.05`), but `Boot p10–p90 = 301–1,067`, and `L36` has `0` names yet `boot p90 = 1,096` (`frac_zero = 0.705`). Similarly `L31` (`1,304`, `ctl>=t = 5`, `boot 671–1,609`) and `L32` (`1,052`, `ctl>=t = 4`, `boot 661–1,501`) depend strongly on a handful of tail controls.
4. **No fresh-question replication on `6`-question layers (`L32–L52`)**: `out_of_sample_3way` (`A4/B4` fit -> held-out `C4`) can run only on `12`-question layers (`L31: 0.969`, `L53: 0.948`, `L54: 0.944`, testing the `4+4` fit subset of `743`, `3,874`, and `5,260` neurons rather than the full `6+6` sets of `1,304`, `3,964`, and `6,128`). On `L32–L52` (`6` questions total), `3+3` selection has no unseen-question holdout.
5. **Re-read after seeing data**: Because every rule (`T_dec`, exact glue filter, `COLLAPSE_RULE` exclusion) was settled after inspecting whole layers, this reader gates nothing without an owner decision and pre-registered validation.
