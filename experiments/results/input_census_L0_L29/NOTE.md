# Input-Side Token-Trigger Tables (L0–L29): Exploratory, Unregistered, Read on the Existing Passages Index

> **Status & Scope**: Exploratory descriptive reading computed on CPU from the existing `passages.d` index (`2,000,054` word-pieces across `3,914` windows, `1,785` documents). **Unregistered; gates nothing.** This does **not** establish push-side causality or turn layers `0–29` into a census (`forced_odd` push-side pilots at `L20, L25, L29` and `L30` calibration produced `0` replicated descriptions).

---

## 1. Recomputed Method (Fixing Document Leakage & Non-Causal `+1` Offset)

The initial PR run split `passages.d` by even/odd window parity and allowed offset `+1` (one token after the neuron fired). Both flaws have been removed in `backend/input_census.py` and recomputed cleanly on CPU (`scripts/build_input_census.py --layers 0-29`):

1. **Random document-disjoint split**: `windows.npy` separates documents at `<eos>` (`token_id == 1`, `1,784` occurrences yielding `1,785` documents). Each document is assigned deterministically to half `0` or half `1` via `np.random.default_rng(20261124)` so neighboring windows from the same document cannot straddle the split.
2. **Causal offsets only (`-1` and `0`)**: Triggers come strictly from `offset == -1` (piece before peak, excluding `<bos>` at index 0) and `offset == 0` (peak token). Replicated `+1` ("next piece") tokens are stripped from triggers and reported separately (`Next+1%`).
3. **Raw `>= 2 + 2` count before random-corpus FDP**: Across `29` of `30` layers, empirical FDP against random corpus positions (`C_null = 4,096`) selects **every single eligible face with `>= 2` hits in both halves** (`RandFDP05 == Raw2+2`, differing only at `L27`: `8,720 -> 8,468`). Because a random-position null only tests against uniform noise, this table reports raw `>= 2 + 2` directly rather than claiming an `"FDR <= 5%"` guarantee.
4. **Two stronger checks on description specificity**:
   - **Same-layer frequency-matched other-face null (`OthFaceNull`)**: Tests whether a face's rank-1 causal `(token, offset)` trigger replicates `>= 2 + 2` across the halves of another eligible face in the same layer matched by unigram frequency bin.
   - **Unseen-document held-out firing recall (`SmpHit` vs `SmpFreq` / `SmpShuf`)**: Excludes held-out `smp` events sharing any document ID with the face's top-16 events, then compares rank-1 trigger match rate on unseen documents against both a frequency-matched wrong piece (`SmpFreq`, same unigram frequency bin & offset) and a shuffled face trigger (`SmpShuf`).

### Summary Effect of Fixes (`L0–L29` Total Over `1,290,240` Faces)

- Prior window-split `(-1, 0, +1)`: `618,372` faces (`48.6%` of eligible) marked at `FDP05`.
- Recomputed document-disjoint causal-only `(-1, 0)`: **`467,498`** faces with `>= 2 + 2` hits (`38.7%` of `1,207,617` eligible faces; `RandFDP05 = 467,246`, selecting only `252` fewer at `L27`).
- Pooled unseen-document held-out trigger match rate: **`343,231 / 1,027,956` (`33.39%`)**, vs **`148 / 1,027,956` (`0.014%`)** for a frequency-matched wrong token and **`12,320 / 1,027,956` (`1.20%`)** for shuffled triggers. *(Note: `smp` statistics apply only to `raw_2+2` faces with `>= 1` held-out firing on unseen documents; many top-16 specific faces have `smp = 0/0`.)*

---

## 2. Full Recomputed Per-Layer Table (`L0–L29`)

| `L` | `Elig / Total` | `Raw >=2+2` | `Elig%` | `RandFDP05` | `OthFaceNull` | `Top100%` | `Bigram%` | `Next+1%` | `SmpHit` | `SmpFreq` | `SmpShuf` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | `40,719 / 43,008` | `24,658` | `60.6%` | `24,658` | `0.011` | `31.9%` | `23.6%` | `14.9%` | `0.359` | `0.000` | `0.005` |
| **1** | `40,516 / 43,008` | `28,654` | `70.7%` | `28,654` | `0.005` | `24.6%` | `32.6%` | `23.0%` | `0.336` | `0.000` | `0.003` |
| **2** | `40,792 / 43,008` | `26,991` | `66.2%` | `26,991` | `0.020` | `43.5%` | `31.9%` | `17.5%` | `0.423` | `0.000` | `0.011` |
| **3** | `40,757 / 43,008` | `23,770` | `58.3%` | `23,770` | `0.010` | `51.7%` | `22.4%` | `19.6%` | `0.361` | `0.000` | `0.010` |
| **4** | `40,606 / 43,008` | `26,999` | `66.5%` | `26,999` | `0.015` | `40.4%` | `23.1%` | `19.0%` | `0.404` | `0.000` | `0.010` |
| **5** | `40,223 / 43,008` | `21,163` | `52.6%` | `21,163` | `0.015` | `51.6%` | `16.4%` | `16.6%` | `0.306` | `0.000` | `0.013` |
| **6** | `39,382 / 43,008` | `14,784` | `37.5%` | `14,784` | `0.008` | `34.5%` | `11.1%` | `13.1%` | `0.330` | `0.000` | `0.007` |
| **7** | `39,636 / 43,008` | `16,397` | `41.4%` | `16,397` | `0.016` | `62.8%` | `12.0%` | `14.3%` | `0.324` | `0.000` | `0.014` |
| **8** | `39,549 / 43,008` | `17,356` | `43.9%` | `17,356` | `0.010` | `49.1%` | `13.2%` | `14.9%` | `0.355` | `0.000` | `0.008` |
| **9** | `39,820 / 43,008` | `14,670` | `36.8%` | `14,670` | `0.008` | `44.1%` | `10.0%` | `9.6%` | `0.324` | `0.000` | `0.008` |
| **10** | `39,569 / 43,008` | `11,832` | `29.9%` | `11,832` | `0.007` | `37.4%` | `7.6%` | `9.1%` | `0.374` | `0.000` | `0.007` |
| **11** | `40,033 / 43,008` | `22,398` | `56.0%` | `22,398` | `0.003` | `20.0%` | `14.2%` | `16.0%` | `0.494` | `0.000` | `0.003` |
| **12** | `39,634 / 43,008` | `18,658` | `47.1%` | `18,658` | `0.006` | `30.7%` | `13.2%` | `17.8%` | `0.356` | `0.000` | `0.005` |
| **13** | `40,265 / 43,008` | `22,413` | `55.7%` | `22,413` | `0.006` | `41.5%` | `15.5%` | `19.0%` | `0.380` | `0.000` | `0.006` |
| **14** | `40,343 / 43,008` | `19,270` | `47.8%` | `19,270` | `0.013` | `50.4%` | `12.8%` | `18.0%` | `0.338` | `0.000` | `0.009` |
| **15** | `39,875 / 43,008` | `16,437` | `41.2%` | `16,437` | `0.011` | `48.2%` | `11.7%` | `18.2%` | `0.324` | `0.000` | `0.010` |
| **16** | `39,864 / 43,008` | `14,758` | `37.0%` | `14,758` | `0.011` | `48.3%` | `8.7%` | `13.9%` | `0.301` | `0.000` | `0.010` |
| **17** | `39,914 / 43,008` | `16,645` | `41.7%` | `16,645` | `0.007` | `34.8%` | `11.4%` | `17.8%` | `0.347` | `0.000` | `0.006` |
| **18** | `39,457 / 43,008` | `12,243` | `31.0%` | `12,243` | `0.011` | `41.5%` | `7.8%` | `13.6%` | `0.294` | `0.000` | `0.010` |
| **19** | `39,785 / 43,008` | `10,356` | `26.0%` | `10,356` | `0.014` | `51.5%` | `6.6%` | `11.5%` | `0.265` | `0.000` | `0.013` |
| **20** | `39,857 / 43,008` | `9,331` | `23.4%` | `9,331` | `0.018` | `51.9%` | `6.2%` | `12.2%` | `0.260` | `0.000` | `0.011` |
| **21** | `39,757 / 43,008` | `8,587` | `21.6%` | `8,587` | `0.034` | `64.2%` | `5.0%` | `9.9%` | `0.245` | `0.000` | `0.023` |
| **22** | `39,486 / 43,008` | `6,988` | `17.7%` | `6,988` | `0.024` | `56.9%` | `4.0%` | `7.7%` | `0.232` | `0.000` | `0.013` |
| **23** | `40,365 / 43,008` | `8,212` | `20.3%` | `8,212` | `0.024` | `64.1%` | `5.3%` | `9.3%` | `0.255` | `0.000` | `0.017` |
| **24** | `40,827 / 43,008` | `9,254` | `22.7%` | `9,254` | `0.027` | `73.2%` | `5.5%` | `10.4%` | `0.252` | `0.000` | `0.018` |
| **25** | `41,214 / 43,008` | `9,608` | `23.3%` | `9,608` | `0.031` | `78.5%` | `5.8%` | `10.9%` | `0.280` | `0.000` | `0.022` |
| **26** | `41,316 / 43,008` | `8,611` | `20.8%` | `8,611` | `0.040` | `84.6%` | `5.0%` | `9.3%` | `0.278` | `0.000` | `0.025` |
| **27** | `41,447 / 43,008` | `8,720` | `21.0%` | `8,468` | `0.042` | `84.1%` | `5.7%` | `9.2%` | `0.307` | `0.000` | `0.030` |
| **28** | `41,305 / 43,008` | `9,084` | `22.0%` | `9,084` | `0.062` | `87.0%` | `5.0%` | `7.4%` | `0.286` | `0.000` | `0.042` |
| **29** | `41,304 / 43,008` | `8,651` | `20.9%` | `8,651` | `0.058` | `83.9%` | `4.9%` | `6.5%` | `0.287` | `0.000` | `0.036` |

Key structural pattern visible in `Top100%` and `OthFaceNull`:
- At `L0–L17`, `20.0–62.8%` of top triggers lie in the 100 commonest corpus pieces and `OthFaceNull` is `0.3–2.0%` (genuine lexical token detectors such as ` blueberries`, ` lastname @`, `rizzle`).
- By `L24–L29`, **`73.2–87.0%`** of rank-1 triggers are top-100 ubiquitous pieces (`,` , `.` , `the` , `\n`), and **`4.0–6.2%`** of those triggers falsely replicate `>=2+2` on unrelated faces at the same layer.

---

## 3. Corrections to Prior PR #9 Body & Status of Natural-Firing Ablation Check

1. **Corrected prior PR #9 misquotes** (from the superseded initial window-split run and quiet-control ablation):
   - Prior held-out `smp_named_rate`: **`18.8%` (`L22`) to `47.7%` (`L11`)**, pooled **`30.3%`** across `30` layers (`18.8–27.9%` across `L19–L29`), NOT `~32–40%`.
   - Prior `smp_shuf_rate`: **`0.38%` (`L1`, `L11`) to `3.82%` (`L28`)**, NOT `~0.4–1.2%`.
   - Prior quiet-control `ablation_check.json` AUCs: peak AUC **`0.6155–0.7319`** (`L2: 0.6891`, `L14: 0.7319`, `L20: 0.6301`, `L29: 0.6155`), span AUC **`0.6173–0.7187`** (`L2: 0.6519`, `L14: 0.7187`, `L20: 0.6924`, `L29: 0.6173`), NOT `0.6519–0.7319`.
   - Prior `Null05 = 0.00 max 0`: split an exchangeable null 2,048/2,048, which cannot fail by construction and provides no validation of the null.
2. **Superseded quiet-control ablation & pending GPU rerun**:
   - The existing `ablation_check.json` / `ablation_check.txt` tested zeroing active `named_fdp05` neurons vs quiet controls (`|act| < 0.05 * extreme`) truncated to `22` context pieces, had no pass criterion (`"PASSED"` / `"verified"` removed), treated `256` passages from `64` neurons as independent, and measured small median KL shifts near the `bf16` floor (`~0.00045`).
   - `scripts/input_ablation_check.py` has now been updated to pair each neuron with a **random non-dead same-layer control neuron** on the exact same **full-context** passages (`pos 1..min(seq, pk+4)`), reporting neuron-clustered Wilcoxon tests (`N=64` independent neurons), write-norm ratios, and the `bf16` noise floor. It has **not** been executed on GPU and awaits the owner's explicit decision.
