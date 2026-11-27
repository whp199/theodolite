# Confirmatory Path-Forward Master Summary (`in_use_prereg_trials`)

- **Branch**: `in_use_prereg_trials` (branched from `origin/main` after PR #9)
- **Model**: `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`)
- **Pre-Registrations & Pre-Verdict Amendments**:
  - **Path 1 (`in_use_lens`)**: [`prereg/`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_lens_trial/prereg/RULES.md) — `49876e2` -> pre-run bugfix amendment `d68d9d5` ([AMENDMENTS.txt](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_lens_trial/prereg/AMENDMENTS.txt): pass unpacked `w_ops` weights into batch-0 sanity for W4A16 packed linears; 0 events run prior).
  - **Path 3 (`in_use_pattern`)**: [`prereg/`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_pattern_trial/prereg/RULES.md) — `49876e2` -> pre-verdict bugfix amendment `965fa1e` ([AMENDMENTS.txt](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_pattern_trial/prereg/AMENDMENTS.txt): avoid `2.0 ** 1209` float64 overflow in `binom_tail`; 0 verdict files written prior).
  - **Path 2 (`u2_rollout`)**: [`prereg/`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/u2_rollout_trial/prereg/RULES.md) — `49876e2` -> pre-verdict bugfix amendment `fcaa21c` ([AMENDMENTS.txt](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/u2_rollout_trial/prereg/AMENDMENTS.txt): read nested `fdp05["count"]` from `census_fdr.eval_pair`; 0 verdict files written prior).
- **Protected Census Files**: Zero edits to `backend/census.py`, `backend/census_volunteer.py`, `volunteers/*`, or `/home/eric/bin/ai/theodolite2/atlas_data/census/runs/*`.

---

## Executive Verdict Across All 3 Confirmatory Paths

| Path & Instrument | Layers Tested | Strict Prereg Verdict | Headline Verified Numbers | Mechanistic Status & Recommendation |
|---|---|---:|---|---|
| **Path 2 (`u2_rollout`)**<br>Dual-control `match_things` ([RESULT.md](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/u2_rollout_trial/RESULT.md)) | `L55, L56, L57`<br>(`1,022` neurons/layer vs `1,024 col` + `1,024 iso`) | **`L55`: PASS (`2A & 2B`)**<br>**`L56`: PASS (`2A & 2B`)**<br>**`L57`: PASS (`2A & 2B`)** | **`2A dual05_n3`**: `94 / 134 / 82` (`C4 = 1.000 / 0.975 / 0.937`)<br>**`2B ww_dual`**: **`177 (17.3%) -> 307 (30.0%) -> 382 (37.4%)`** (`AUC = 0.675–0.734`, `387x` mean own-word at `L57`, cross-pool **`0/1,024`**) | **CLEARED (`L55–L57`)**. Extend upward `U2` walk / rollout read to **`L58–L59`** (fast at `L58–L59`? Same rollout cost) or open `L55–L57` for full census. |
| **Path 1 (`in_use_lens`)**<br>Window-cached natural ablation ([RESULT.md](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_lens_trial/RESULT.md)) | `L59, L58, L57`<br>(`1,024` `pure_4_4` /layer vs `1,024 col` + `1,024 iso`) | **`L59`: FAIL (`1/5`)**<br>**`L58`: FAIL (`4/5`)**<br>**`L57`: FAIL (`3/5`)** | **`L58`**: **`156 / 1,024 (15.23%)`** dual `FDP<=0.05` (`all8 AUC = 0.6384 / 0.7114`, `rho = 0.7115`, cross-pool `0/1,024`), missed only `G5` (`shuf AUC = 0.1382`, `fdp05 = 0`).<br>**`L57`**: `dual05 = 95/1,024 (9.28%)`.<br>**`L59`**: `all8_vs_col = 0.6540`, `vs_iso = 0.5552`, `dual05 = 0`. | **`L58` works in `8.7 min/layer`** once `G5` null compares shuffled-vs-shuffled (or one-sided `fdp05==0`). **`L59`** fails single-token `vs_iso` (use `P2` rollout at `L59`). |
| **Path 3 (`in_use_pattern`)**<br>Local `Gemma-4-31B` patterns ([RESULT.md](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_pattern_trial/RESULT.md)) | `L2` (`128`), `L14` (`128`), `L20` (`64`) on `doc_4_8` faces | **`L2`: FAIL (`1/3`)**<br>**`L14`: FAIL (`1/3`)**<br>**`L20`: FAIL (`0/3`)** | **`L2`**: `G2` **PASS** (**`50/128 = 39.06%`** `q<=0.10`, `own=0.7535` vs `0.5183`), `mass=0.8687`, `behind clean = 30/128 (23.44%)` (`0.0%` other fire).<br>**`L14`**: `G2` **PASS** (**`51/128 = 39.84%`** `q<=0.10`, `0.7392` vs `0.4828`), `mass=0.8378`, `behind clean = 11.90%`.<br>**`L20`**: `own=0.7023`, `17.46%` `q<=0.10`, `1.56%` clean. | **`blind_near` reads `L2–L14` at `~39–40%`** (`BH q<=0.10`); fix `a`/`b` prompt calibration (`p_ab_mass < 0.90`). Physical short `behind` drops off by `L14`, and **`L20–L29` remain unresolved**. |

---

## Global Picture of Gemma-4 31B (`L0..L59`)

1. **Top End (`L55–L57`, plus `L58`):**
   - `P2` (`u2_rollout`) definitively clears **`L55, L56, L57`** under dual paired controls (`columns` + `isotropic`), proving that closing the upper ceiling above `L54` only needed dual-control rollout reading + the corrected `U2-weight_words` metric (`17.3% -> 37.4%` yield).
   - `P1` (`in_use_lens`) shows why `L58` and `L59` behave differently under single-token natural peak ablation: at `L58` (one block before `lm.norm`), natural peak `eye_10` ablation strongly beats both controls (`156/1,024`, `0/1,024` cross-pool), whereas at `L59` (zero blocks before `lm.norm`), random `isotropic` directions align directly with raw vocabulary tokens and overtake `columns` (`4.9x` `p90`). Testing `P2` (`u2_rollout`) at **`L58` and `L59`** (or combining `P2` + corrected `G5` `P1` at `L58`) is the clear path to finishing the top layers.
2. **Bottom End (`L0–L15`):**
   - `P3` (`in_use_pattern`) proves `100%`-local `Gemma-4-31B-it` reading (`doc_h0` authoring -> single-prefill `blind_near` on `>= 8` `doc_h1` pairs) achieves **`39.1%–39.8%` `BH q <= 0.10`** discovery yield at `L2` and `L14` against `K = 3` deranged nulls (`wilcoxon p <= 3.86e-16`), eliminating the 4-pair binomial floor seen in Pilot 3.
   - Strict gate failure at `L2` and `L14` stemmed from two concrete points: `median_p_ab_mass` dropped slightly (`0.869 / 0.838 < 0.90`), and physical `behind` counterfactuals missed `25%` cleanly at `L2` by `2` faces (`30/128 = 23.44%`, with `35.42%` fire vs `6.01%` quiet and `0.0%` other-pattern fire) before weakening at `L14` (`11.90%`).
3. **Remaining Middle Gap (`L20–L29`):**
   - `L20` failed `0/3` gates cleanly (`own_share = 0.7023`, `17.46%` `BH q <= 0.10`, `1.56%` `behind clean`), confirming `in_use_pattern` on short local snippets does not span `L20–L29`. That band (`L16–L29`) awaits either long-context / multi-turn pattern authoring & judging or intermediate residual circuit tracing.

---

## Verification & Provenance Note

- All pre-registration checksums (`SHA256SUMS` and `CODE_SHA256SUMS` across all 3 `prereg/` directories) pass `sha256sum -c`.
- [verify_trial_outputs.py](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/scripts/verify_trial_outputs.py) passes `7/7` checks (including `binom_tail` large-`n`, `u2` nested `fdp05` schema, all 3 gate logic tests, tamper detection, zero raw corpus keys across all result JSONs, and exact `code_sha256` match against `CODE_SHA256SUMS`).
- Untouched [check_in_use_trials.py](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/scripts/check_in_use_trials.py) was left unedited to keep every locked `CODE_SHA256SUMS` intact; its tests `1–4` pass and only test `5` (`test_recorded_trial_outputs_if_present`) expects `git_dirty: false` + `prereg` on every `*.json` file in `results_dir` (including per-layer `in_use_lens_trial/L*_verdict.json` and end-of-run `verdict.json`, which runs `gp.build_provenance()` after `L*_verdict.json` checkpoints exist).
