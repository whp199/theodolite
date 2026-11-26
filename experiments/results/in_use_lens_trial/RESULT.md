# Path 1 (`in_use_lens`) Confirmatory Trial Result

- **Branch / Commit**: `in_use_prereg_trials` (`d68d9d562fbc8c4450582f23d4550c1bd682f737`, `AMENDMENTS.txt` recorded pre-run)
- **Model**: `google/gemma-4-31B-it-qat-w4a16-ct` (`weights_id = e0a93173c31ed2ec`)
- **Sample**: `N = 1,024` fresh `pure_4_4` neurons × `8` events (`4 pure_h0 + 4 pure_h1`) = `8,192` events/layer, paired against `C_col = 1,024` `columns` + `C_iso = 1,024` `isotropic` norm-matched controls + deranged shuffled-label null.
- **Total GPU Runtime**: `1,608.3 s` (`26.8 min` total on RTX 5090; `~8.3–9.6 min/layer` via window-prefix caching).

---

## Confirmatory Gate Verdicts (`L59, L58, L57`)

Per [`prereg/RULES.md`](file:///home/eric/.gemini/antigravity/worktrees/theodolite2/weight_space_solution_brief/experiments/results/in_use_lens_trial/prereg/RULES.md), a layer passes strictly only if **all five gates (`G1–G5`)** pass with zero post-hoc changes.

| Layer | Verdict (`G1–G5`) | `G1` All-8 Effect | `G2` Test-4 & `rho` | `G3` Dual `FDP<=0.05` (`>=103`) | `G4` Cross-Pool (`<=2%`) | `G5` Shuffled Null + Sanity | Runtime / Windows |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| **L59** | **FAIL (`1/5`)** | **FAIL** (`vs col 0.6540`, `vs iso 0.5552`) | **FAIL** (`vs col 0.6374`, `vs iso 0.5426`, `rho=0.7840`) | **FAIL** (`0 / 1,024 = 0.0%`) | **PASS** (`0 / 1,024 = 0.0%`) | **FAIL** (`shuf AUC=0.1273`, sanity **PASS**) | `496.3 s` (`2,719` win) |
| **L58** | **FAIL (`4/5`)** | **PASS** (`vs col 0.6384`, `vs iso 0.7114`) | **PASS** (`vs col 0.6257`, `vs iso 0.6966`, `rho=0.7115`) | **PASS** (**`156 / 1,024 = 15.23%`**) | **PASS** (`0 / 1,024 = 0.0%`) | **FAIL** (`shuf AUC=0.1382`, sanity **PASS**) | `523.4 s` (`2,767` win) |
| **L57** | **FAIL (`3/5`)** | **PASS** (`vs col 0.5849`, `vs iso 0.6350`) | **PASS** (`vs col 0.5812`, `vs iso 0.6238`, `rho=0.6321`) | **FAIL** (`95 / 1,024 = 9.28%`) | **PASS** (`0 / 1,024 = 0.0%`) | **FAIL** (`shuf AUC=0.1537`, sanity **PASS**) | `574.6 s` (`2,722` win) |

---

## Detailed Per-Layer Metrics (`verdict.json` / `L*_verdict.json`)

| Metric | `L59` | `L58` | `L57` |
|---|---:|---:|---:|
| `eye_pk_all8_vs_col`: `AUC` (`mw_p` / `wilcoxon_p`) | `0.6540` (`7.54e-34` / `2.07e-31`) | `0.6384` (`1.01e-27` / `9.66e-40`) | `0.5849` (`1.41e-11` / `3.43e-22`) |
| `eye_pk_all8_vs_iso`: `AUC` (`mw_p` / `wilcoxon_p`) | `0.5552` (`7.60e-6` / `1.68e-6`) | `0.7114` (`6.34e-62` / `1.01e-72`) | `0.6350` (`1.92e-26` / `2.88e-45`) |
| `eye_pk_test4_vs_col`: `AUC` (`mw_p`) | `0.6374` (`2.39e-27`) | `0.6257` (`3.35e-23`) | `0.5812` (`9.74e-11`) |
| `eye_pk_test4_vs_iso`: `AUC` (`mw_p`) | `0.5426` (`4.24e-4`) | `0.6966` (`7.75e-54`) | `0.6238` (`1.53e-22`) |
| `rho_fit4_test4` (`p`) | `0.7840` (`8.87e-139`) | `0.7115` (`1.27e-114`) | `0.6321` (`6.92e-91`) |
| `p90` `eye_pk_all8` drop: Neuron vs `col` vs `iso` | `9.41e-3` vs `2.62e-4` vs `1.28e-3` | `1.38e-3` vs `7.0e-6` vs `3.0e-6` | `8.7e-5` vs `2.0e-6` vs `1.0e-6` |
| `emp_p <= 0.05` vs `col` / vs `iso` | `191 (18.65%)` / `146 (14.26%)` | **`303 (29.59%)`** / **`382 (37.30%)`** | `219 (21.39%)` / `277 (27.05%)` |
| `fdp05_vs_col`: count (`boot p10 / p50 / frac_zero`) | `0` (`0 / 0 / 0.515`) | **`156`** (`147 / 159 / 0.0`) | `95` (`87 / 98 / 0.0`) |
| `fdp05_vs_iso`: count (`boot p10 / p50 / frac_zero`) | `0` (`0 / 32 / 0.470`) | **`222`** (`175 / 222 / 0.0`) | `138` (`121 / 139 / 0.0`) |
| `n_above_max_ctl` (`vs col` / `vs iso`) | `15` / `13` | `76` / `98` | `77` / `78` |
| `n_bh_q10` (`vs col` / `vs iso`) | `52` / `76` | `235` / `321` | `138` / `221` |
| `dual05_count` (`>=103` required) | `0` (`0.0%`) | **`156` (`15.23%`)** | `95` (`9.28%`) |
| `kl_all8_vs_col` / `kl_all8_vs_iso` `AUC` | `0.5800` / `0.4986` | `0.5584` / `0.5110` | `0.5417` / `0.5089` |
| `shuffled_null_vs_col`: `AUC` / `fdp05_count` | `0.1273` / `0` | `0.1382` / `0` | `0.1537` / `0` |
| Batch-0 sanity (`noop_kl` / `alg_vs_hook` / `lens_20`) | `0.0` / `0.0` / `200+200` | `0.0` / `0.0` / `200+200` | `0.0` / `0.0` / `200+200` |
| Post-norm match ratio (`col` / `iso`) | `1.0019` / `0.9990` | `0.9994` / `0.9974` | `0.9998` / `0.9997` |

---

## Mechanistic Diagnosis (No Goalpost Shifts)

1. **`L58` cleared every substantive biological, out-of-sample, discovery-yield, and control-exchangeability gate (`G1–G4`)**:
   - At `L58` (one block before unembedding), `156 / 1,024 (15.23%)` fresh `pure_4_4` neurons pass dual `FDP <= 0.05` against both `columns` (`156`, `boot p10 = 147`, `0%` zero bootstraps) and `isotropic` (`222`, `boot p10 = 175`), with `0 / 1,024` cross-pool false discoveries (`G4`). Both overall `all8` (`AUC = 0.6384 / 0.7114`) and strictly unseen `test4` (`AUC = 0.6257 / 0.6966`, `rho = 0.7115`) replicate cleanly.
2. **`L57` attenuates just below the 10% yield floor (`G1, G2, G4` pass; `G3` missed by 8 neurons)**:
   - Two blocks before unembedding, `all8_vs_col AUC` drops to `0.5849` and `dual05 = 95 / 1,024 (9.28%)` (`fdp05_vs_iso = 138`), just short of the pre-registered `103 (10%)` gate (`G3`).
3. **`L59` replicates the `columns` separation (`AUC = 0.6540`) but is outperformed by `isotropic` noise at `0` blocks before unembedding**:
   - Against `columns`, `L59` shows strong `p90` drop (`36x`: `0.009412` vs `0.000262`) and `rho = 0.7840`.
   - However, at `L59` (directly normalized by `lm.norm` into logits with no intermediate block), a random isotropic write direction (`v_iso`) aligned to its own top-10 logits causes `4.9x` larger `eye_10` drops than `columns` at `p90` (`0.001279` vs `0.000262`), cutting `all8_vs_iso AUC` to `0.5552` (`kl_all8_vs_iso AUC = 0.4986`). Additionally, discrete `FDP <= 0.05` (`(1 + c)/k <= 0.05`) requires at least `20` true discoveries above the maximum control when `c = 0`; `L59` had `above_max_ctl = 15` (`vs col`) and `13` (`vs iso`), yielding `dual05 = 0` despite `52 / 76` passing `BH q <= 0.10`.
4. **Why `G5` failed on all three layers (`shuffled_null_vs_col AUC = 0.127–0.154`)**:
   - Instrumentation sanity passed bit-for-bit on every layer (`noop_kl = 0.0`, `alg_vs_hook = 0.0`, `200/200` lens overlap, `0.997–1.002` norm match) and shuffled neuron labels produced `0 / 1,024` `FDP <= 0.05` discoveries (`0` `BH q <= 0.10`).
   - The `G5` check tripped because `shuf_all` measures probability change on **wrong** shuffled tokens (`median = 0.0`, often negative when ablation shifts mass away from true tokens toward alternatives) whereas `col_all` measures probability drop on the control's **own** `eye_10` tokens (`median = 0..2e-6`), making `AUC < 0.45` even though the false-discovery count is identically `0`. Under any corrected null (shuffled vs shuffled `eye_10` of `col`, or one-sided `fdp05 == 0 & AUC <= 0.55`), `L58` passes cleanly; per pre-registration rules we retain `G5 = False` as specified.

---

## Provenance Note

- Startup `verify_prereg_integrity` passed on a clean tracked working tree (`HEAD=d68d9d562fbc`) and verified `SHA256SUMS` + `CODE_SHA256SUMS` before running any layer.
- `verdict.json` records `"git_dirty": true` solely because `in_use_lens_trial.py` writes the per-layer `L{59,58,57}_verdict.json` checkpoints into `experiments/results/in_use_lens_trial/` before calling `gp.build_provenance(...)` at the end of `main()`. All `code_sha256` entries match the locked pre-registration digests verbatim.
