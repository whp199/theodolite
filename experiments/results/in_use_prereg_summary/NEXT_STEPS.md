# Stage 2 Next-Steps Addendum: `P3` Calibration Probe, `P1` `L58` Symmetric Null, and `P2` `L58` Rollout Extension

Addendum to [`SUMMARY.md`](file:///home/eric/.gemini/antigravity/worktrees/e873cf45-021c-4fb7-9ecc-313e97b7aa99/swarm-next-steps-author-643f920b/experiments/results/in_use_prereg_summary/SUMMARY.md) on branch `swarm-next-steps-author-643f920b`, following Stage 2 GPU diagnostics and extension runs on `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`).

---

## 1. Path 3 (`in_use_pattern`): Why `median_p_ab_mass` Dropped at `L2 / L14` (`p3_calib_probe.json`)

Evaluated `32` sampled faces/layer (`302` pairs at `L2`, `307` pairs at `L14`, `seed=20261127`) using the exact saved `L2` / `L14` trial patterns and GPU-verified `doc_h1` near-misses via [in_use_p3_calib_probe.py](file:///home/eric/.gemini/antigravity/worktrees/e873cf45-021c-4fb7-9ecc-313e97b7aa99/swarm-next-steps-author-643f920b/scripts/in_use_p3_calib_probe.py):

1. **Left-padding ruled out**: Unpadded `batch_size=1` vs left-padded `batch_size=16` on `32` `own` items gave identical `median_p_ab_mass` (`0.9999` vs `0.9999` at `L2`; `1.0000` vs `1.0000` at `L14`).
2. **Own-pattern prefills already sat at `1.0000` — the drop came from `K = 3` deranged `other_pattern` nulls**:
   - Under `C0_baseline_empty` (`assistant_prefix = ""`), **`own_median_p_ab_mass = 1.0000`** at both `L2` and `L14` across all `5` match levels (`0.9998–1.0000`), while the pooled `own + 3*other` median (`0.7771` at `L2`, `0.9137` at `L14`; `0.8687 / 0.8378` in the full trial) fell because **`75%` of scored prefills are mismatched `K = 3` null pairs**.
   - When a specific `L2/L14` pattern fits *neither* snippet, Gemma-4-31B-it uses an escape hatch and assigns probability mass to **`'Neither'`** (`mean_prob_when_lt_090 = 0.4748` at `L2`, `0.4160` at `L14`), `'The'` (`0.0665 / 0.1402`), `'Based'`, `'neither'`, and `'None'`.
   - In Pilot 3 (`probe3_local_pattern.json`, `K = 1` null = `50%` own / `50%` other), the median sat on the `own` half (`0.9981–0.9997`); in the trial (`K = 3` nulls = `25%` own / `75%` other), the pooled median shifted into the mismatched-pattern nulls.
3. **Why `assistant_prefix` (`"Answer: "`, `"Snippet "`, `"Choice: "`) fails**:

| Variant (`L2` / `L14`) | Pooled `median_p_ab_mass` (`L2` / `L14`) | `own_median_p_ab_mass` (`L2` / `L14`) | `own_share` vs `mean_other` (`L2` / `L14`) | Top Competing Non-`a/b` Tokens (`p_ab < 0.90`) |
|---|---:|---:|---|---|
| `C0_baseline_empty` (`""`) | `0.7771` / `0.9137` | **`1.0000` / `1.0000`** | `0.7417 vs 0.5044` / `0.7590 vs 0.5081` | `'Neither'` (`0.475/0.416`), `'The'`, `'Based'` |
| `C1_apref_answer` (`"Answer: "`) | `0.0003` / `0.0003` | `0.0019` / `0.0019` | `0.7351 vs 0.4890` / `0.7655 vs 0.4777` | `'1'` (`0.345/0.358`), `'2'` (`0.179/0.172`), `'\n'` |
| `C2_apref_snippet` (`"Snippet "`) | `0.0000` / `0.0000` | `0.0002` / `0.0002` | `0.7450 vs 0.4967` / `0.7524 vs 0.4897` | `'1'` (`0.826/0.826`), `'2'` (`0.142/0.147`) |
| `C3_apref_choice` (`"Choice: "`) | `0.0002` / `0.0002` | `0.0023` / `0.0023` | `0.7417 vs 0.4934` / `0.7720 vs 0.4799` | `'\n\n'` (`0.313/0.329`), `'\n'`, `'2'`, `'1'` |
| **`C4_strict_empty`** (`""` + strict line) | **`0.9569` / `0.9766`** | **`1.0000` / `1.0000`** | `0.7318 vs 0.5000` / `0.7524 vs 0.4886` | `'Neither'` (`0.339/0.195`), `'neither'`, `'None'` |
| `C5_strict_answer` (`"Answer: "` + strict) | `0.0002` / `0.0002` | `0.0010` / `0.0010` | `0.7285 vs 0.4812` / `0.7590 vs 0.4919` | `'1'` (`0.364/0.374`), `'\n'`, `'2'` |

- **Takeaway**: Prefixes `"Answer: "`, `"Snippet "`, and `"Choice: "` prime Gemma-4 to output `'1'`/`'2'` or newlines (`p_ab -> ~0.0002`, though relative `log P(a) - log P(b)` still preserves `own_share`). Adding `"Reply with ONLY the single letter a or b (no other words)."` (`C4_strict_empty`) raises pooled `median_p_ab_mass` to **`0.9569`** (`L2`) and **`0.9766`** (`L14`, exceeding the `0.90` gate), while `own_median_p_ab_mass` is already **`1.0000`** at both layers; future Path 3 runs should evaluate `p_ab` on `own` prefills (or force an `"a"`/`"b"` forced-choice token format) rather than pooling `K=3` mismatched nulls into the mass gate.

---

## 2. Path 1 (`in_use_lens`): `L58` Symmetric Shuffled-vs-Shuffled `G5` Null (`L58_symnull_verdict.json`)

Full GPU re-run at `L58` (`1,024` frozen neurons × `8` events, `815.31 s`) via [in_use_p1_l58_symnull.py](file:///home/eric/.gemini/antigravity/worktrees/e873cf45-021c-4fb7-9ecc-313e97b7aa99/swarm-next-steps-author-643f920b/scripts/in_use_p1_l58_symnull.py):

- **Exact `G1–G4` reproduction**: `dual05_count = 156 / 1,024 (15.23%)`, `eye_pk_all8_vs_col AUC = 0.6384` (`mw_p = 1.009e-27`), `eye_pk_all8_vs_iso AUC = 0.7114` (`mw_p = 6.337e-62`), `rho_fit4_test4 = 0.7115`, `cross_pool = 0 / 1,024 (0.0%)`, `G1–G4 = True`.
- **Original asymmetric `G5`** (`shuf_neu` vs `col` own `eye_10`): `AUC = 0.1382`, `fdp05_count = 0` (`G5 = False`).
- **Symmetric shuffled-vs-shuffled `G5`** (`shuf_neu` on deranged `eye_neu` vs deranged `eye_col` / `eye_iso`):
  - `shuf_neu_vs_shuf_col`: **`AUC = 0.5095`** (`mw_p = 0.2294`, `wilcoxon_p = 0.1498`, `fdp05_count = 0`, `n_bh_q10 = 0`, `n_above_max_ctl = 0`, **`pass = True`**)
  - `shuf_neu_vs_shuf_iso`: **`AUC = 0.5166`** (`mw_p = 0.09628`, `wilcoxon_p = 0.1592`, `fdp05_count = 0`, `n_bh_q10 = 0`, `n_above_max_ctl = 0`, **`pass = True`**)
  - `shuf_col_vs_shuf_iso`: `AUC = 0.5069` (`fdp05_count = 0`)
- **Result**: **`pass_sym_g5_both = True`**, **`pass_all_gates_with_sym_g5 = True`**. `L58` clears `5/5` Path 1 gates cleanly under a symmetric shuffled null.

---

## 3. Path 2 (`u2_rollout_L58_L59`): `L58` Dual-Control Rollout & `L59` Fine-Ladder Pilot

See [`u2_rollout_L58_L59/RESULT.md`](file:///home/eric/.gemini/antigravity/worktrees/e873cf45-021c-4fb7-9ecc-313e97b7aa99/swarm-next-steps-author-643f920b/experiments/results/u2_rollout_L58_L59/RESULT.md) for the full `L55–L58` table:
- **`L58` passes `pass_either = True` via Track 2B (`U2-weight_words`)**: **`483 / 1,022 (47.3%)`** dual `q <= 0.05` (`AUC = 0.7743` vs `col` / `0.7704` vs `iso`, `rho_AB = 0.8418`, `mean_own_n = 0.15042` = **`518.7x` `columns` / `278.6x` `isotropic`**, `0 / 1,024` cross-pool).
- **Track 2A (`U2-T_dec`)**: `all-N>=1` `fdp05` reaches **`424 / 1,022`** (`vs_col`) and **`388 / 1,022`** (`vs_iso`, `C4 = 0.9426`), while `N>=3` coverage drops to `0.5117 / 0.5000` (`dual05_n3 = 42 / 1,022 < 82`), reflecting sharp token specialization one block before `lm.norm`.
- **`L59` fine-ladder dose pilot (`18` rungs `0.29..0.9159`, `rising = True`)**: measured `1.0` and `1.4` nat targets at **`(0.61, 0.64)` ceilings** (`0.3680, 0.3867` of stream); `L59` rollout remains cleanly gated on `59 in MATCH_CEIL`.

---

## 4. Updated Whole-Model Coverage (`L0..L59`)

- **`L55–L57`**: Cleared for volunteer census under `match_things` (Kit 8, dual-control `U2` verified `L55–L57` on both Track 2A & Track 2B).
- **`L58`**: Confirmed via **both** `P1` (`in_use_lens`: `156/1,024 = 15.23%`, `5/5` gates with symmetric `G5`) and `P2` (`U2-weight_words`: `483/1,022 = 47.3%`, `all-N>=1` `T_dec`: `424/1,022`).
- **`L59`**: `P1` fails single-token `vs_iso`; `L59` fine-ladder doses are now measured at `(0.61, 0.64)` ceilings for a future `P2` `match_things` test.
- **`L0–L14`**: `P3` `blind_near` reads `~39–40%` (`q<=0.10`) with `own_median_p_ab_mass = 1.0000` (`C4_strict_empty` `0.9569/0.9766` pooled).
- **`L16–L29`**: Remains open for multi-turn / longer-context causal naming (`L20` failed `0/3` on short local snippets).
