# Path 3 (`in_use_pattern`) Confirmatory Trial Result

- **Branch / Commit**: `in_use_prereg_trials` (`965fa1e959270d1cf9485d1953a1fa2543bb1086`, `AMENDMENTS.txt` recorded pre-verdict after `binom_tail` float64 overflow fix)
- **Model**: `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`, zero external API calls)
- **Sample**: Fresh `doc_4_8` faces — **`128` at `L2`**, **`128` at `L14`**, **`64` boundary probe at `L20`**; `blind_near` judged on `8–10` held-out `doc_h1` pairs/face against `K = 3` deranged `other_pattern` nulls (`3,048` total GPU-verified pairs), plus unseen-`doc_h1` `behind` counterfactual tests (`3` firing + `3` quiet + `3` other-face controls per face).
- **Total GPU Runtime**: `3,653.8 s` (`60.9 min` on RTX 5090: `1,429.6 s` at `L2`, `1,508.1 s` at `L14`, `706.8 s` at `L20`).

---

## Confirmatory Gate Verdicts (`L2, L14, L20`)

Per [`prereg/RULES.md`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_pattern_trial/prereg/RULES.md), a layer passes strictly only if **all three gates (`G1–G3`)** pass with zero post-hoc changes.

| Layer (`Faces`) | Verdict (`G1–G3`) | `G1` `blind_near` Pooled & Null (`>=0.72`, `>=+0.18`, `mass>=0.90`) | `G2` `blind_near` Yield (`>=90% ge8`, `>=30% p<0.05`, `>=20% q<=0.10`) | `G3` Unseen `behind` (`>=90% holds`, `>=30% fire`, `>=25% clean`, `apart>=0.70`) |
|---:|:---:|---|---|---|
| **`L2` (`128`)** | **FAIL (`1/3`)** | **FAIL** (`own=0.7535`, `null=0.5183`, **`+0.2352`**, `p=0.0`, `wil_p=3.86e-16`; missed `mass=0.8687 < 0.90`) | **PASS** (`100%` `>=8` pairs, **`39.06%`** `p<0.05`, **`39.06%`** `BH q<=0.10`) | **FAIL** (`100%` holds, **`35.42%`** fire, `6.01%` quiet, **`0.0%`** other, `apart=0.737`; missed `clean=23.44% = 30/128` vs `25%`) |
| **`L14` (`128`)** | **FAIL (`1/3`)** | **FAIL** (`own=0.7392`, `null=0.4828`, **`+0.2564`**, `p=0.0`, `wil_p=1.01e-17`; missed `mass=0.8378 < 0.90`) | **PASS** (`100%` `>=8` pairs, **`39.84%`** `p<0.05`, **`39.84%`** `BH q<=0.10`) | **FAIL** (`98.44%` holds, `22.40%` fire, `7.75%` quiet, `1.07%` other, `apart=0.716`; `clean=11.90% = 15/126`) |
| **`L20` (`64`)** | **FAIL (`0/3`)** | **FAIL** (`own=0.7023 < 0.72`, `null=0.5110`, `+0.1913`, `wil_p=2.49e-9`, `mass=0.9995`) | **FAIL** (`98.44%` `>=8` pairs, `22.22%` `p<0.05`, `17.46%` `BH q<=0.10`) | **FAIL** (`100%` holds, `15.71%` fire, `5.82%` quiet, `1.57%` other, `apart=0.685`; `clean=1.56% = 1/64`) |

---

## Detailed Per-Layer Breakdown (`L2_verdict.json`, `L14_verdict.json`, `L20_verdict.json`)

| Metric | `L2` (`128` faces) | `L14` (`128` faces) | `L20` (`64` faces) |
|---|---:|---:|---:|
| Runtime (`s`) | `1,429.6 s` | `1,508.1 s` | `706.8 s` |
| Quiet pairs GPU-verified (`< 0.25x`) | `1,209 / 1,209 (100%)` | `1,231 / 1,231 (100%)` | `608 / 611 (99.5%)` |
| `blind_near` `own_right / pairs` (`own_share`) | `911 / 1,209 (0.7535)` | `910 / 1,231 (0.7392)` | `427 / 608 (0.7023)` |
| `other_shares_k3` (`mean_other_share`) | `0.5095, 0.5294, 0.5161` (`0.5183`) | `0.4898, 0.4809, 0.4777` (`0.4828`) | `0.5099, 0.4934, 0.5296` (`0.5110`) |
| `own_minus_mean_other` (`wilcoxon_face_p`) | **`+0.2352`** (`3.855e-16`) | **`+0.2564`** (`1.005e-17`) | `+0.1913` (`2.488e-9`) |
| `median_p_ab_mass` | `0.8687` | `0.8378` | `0.9995` |
| `faces_ge8_pairs` (`frac`) | `128 / 128 (1.0000)` | `128 / 128 (1.0000)` | `63 / 64 (0.9844)` |
| `faces_p05_own_ge8` (`frac`) | **`50 / 128 (39.06%)`** | **`51 / 128 (39.84%)`** | `14 / 63 (22.22%)` |
| `faces_q10_own_ge8` (`frac`) | **`50 / 128 (39.06%)`** | **`51 / 128 (39.84%)`** | `11 / 63 (17.46%)` |
| `other_p05_counts_ge8` (`mean_rate`) | `3, 4, 4` (`2.86%`) | `8, 1, 4` (`3.39%`) | `0, 1, 1` (`1.06%`) |
| `behind`: `faces_own_holds_gt_0.5` | `128 / 128 (100%)` | `126 / 128 (98.44%)` | `64 / 64 (100%)` |
| `behind`: `written_fired / texts` (`rate`) | `136 / 384 (35.42%)` | `84 / 375 (22.40%)` | `30 / 191 (15.71%)` |
| `behind`: `near_misses_fired / misses` (`rate`) | `23 / 383 (6.01%)` | `29 / 374 (7.75%)` | `11 / 189 (5.82%)` |
| `behind`: `other_face_fired / texts` (`rate`) | `0 / 384 (0.00%)` | `4 / 375 (1.07%)` | `3 / 191 (1.57%)` |
| `behind`: `faces_clean_2of3_and_0quiet` (`rate`) | `30 / 128 (23.44%)` | `15 / 126 (11.90%)` | `1 / 64 (1.56%)` |
| `behind`: `mean_apart` | `0.7370` | `0.7163` | `0.6852` |

---

## Mechanistic Diagnosis (No Goalpost Shifts)

1. **`blind_near` discriminability replicates strongly at `L2` and `L14` (`~39–40%` `BH q<=0.10`)**:
   - Eliminating the pilot's `n < 8` binomial power floor (`100%` of `L2` and `L14` faces reached `8–10` pairs) yielded **`50 / 128 (39.06%)`** individually significant faces at `L2` and **`51 / 128 (39.84%)`** at `L14` under both `p < 0.05` and `BH q <= 0.10`, while `K = 3` deranged null patterns scored at chance (`51.83%` and `48.28%`, only `2.86%–3.39%` at `p < 0.05`).
   - All `6` statistical sub-checks of `G1` passed at `L2` and `L14`; the single `G1` miss was the secondary `a`/`b` calibration check (`median_p_ab_mass = 0.8687 / 0.8378 < 0.90`).
2. **Physical `behind` counterfactual controllability drops steeply with depth**:
   - At `L2`, local Gemma-4 counterfactuals held every requirement except the last `2` faces of `clean_2of3_and_0quiet` (`30/128 = 23.44%` vs `25.0%` gate; `written_fire_rate = 35.42%` vs `6.01%` quiet and `0.00%` other-pattern fire, `mean_apart = 0.7370`).
   - By `L14`, even though `blind_near` reading stayed at `39.84%`, short `behind` continuation rewrites dropped to `22.40%` fire (`11.90%` clean).
   - At `L20`, every gate failed (`blind_near own_share = 0.7023`, `17.46%` `q<=0.10`, `behind clean = 1.56%`), confirming that `in_use_pattern` (`L2` reader + local `behind` counterfactuals) degrades between `L14` and `L20` and is **not** a whole-depth substitute for `L20+`.

---

## Provenance Note

- Startup `verify_prereg_integrity` passed on a clean tracked tree (`HEAD=965fa1e95927`) and verified all `SHA256SUMS` and `CODE_SHA256SUMS` digests.
- Each per-layer `L{2,14,20}_verdict.json` records `"git_dirty": false`; top-level `verdict.json` records `"git_dirty": true` solely because `L*_verdict.json` checkpoints were written to `experiments/results/in_use_pattern_trial/` prior to end-of-run `gp.build_provenance(...)`.
