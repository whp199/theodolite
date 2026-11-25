# Pre-Registered `forced_odd` Trial Results: Layers 55–59

Registered at commit `16570312ed0961779629a171b4d5a2fa768678ba` (`prereg_at = 2026-11-24T23:52:42Z` in `prereg/`), with `sha256sum -c prereg/SHA256SUMS` passing cleanly across all 10 manifest entries before and after all trial runs.

## Verdict Table (`N = 1,022` fresh neurons/layer, `C_cols = 1,024`, `C_iso = 1,024`, `proj/LEX_L2_cent`)

All **5 layers (`55, 56, 57, 58, 59`) PASSED F1, F2, and F3**:

| Layer | Verdict | Gates (`F1/F2/F3`) | Dose (`kl_p90=0.25`) | `KL50 / KL90` (nats) | `rABr / rBAr` vs Cols | `rABr / rBAr` vs Iso | `Col05` | `Iso05` | **`Dual05`** | `NullC / NullI` (mean, max) | `Half C` (`frac_p05` Col / Iso, AUC) | `Direct` (`Direct / Dual05`) | Elapsed |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **55** | **PASS** | `1/1/1` | `1.178` | `0.0362 / 0.1649` | `0.7420 / 0.7370` | `0.7691 / 0.7613` | `403` | `436` | **`403` (39.4%)** | `0.00 / 0.00` (max `0`) | `1.000 / 1.000` (AUC `1.000 / 1.000`) | `215 / 403` | `315.7s` |
| **56** | **PASS** | `1/1/1` | `1.211` | `0.0345 / 0.3666` | `0.7866 / 0.7808` | `0.7920 / 0.7968` | `466` | `522` | **`465` (45.5%)** | `0.00 / 0.00` (max `0`) | `1.000 / 1.000` (AUC `1.000 / 1.000`) | `297 / 465` | `287.4s` |
| **57** | **PASS** | `1/1/1` | `0.9707` | `0.0238 / 0.3514` | `0.7936 / 0.7935` | `0.7781 / 0.7797` | `514` | `598` | **`512` (50.1%)** | `0.00 / 0.00` (max `0`) | `1.000 / 1.000` (AUC `1.000 / 1.000`) | `169 / 512` | `253.8s` |
| **58** | **PASS** | `1/1/1` | `0.5818` | `0.0124 / 0.4681` | `0.8803 / 0.8802` | `0.8973 / 0.8958` | `720` | `768` | **`718` (70.3%)** | `0.00 / 0.00` (max `0`) | `1.000 / 1.000` (AUC `1.000 / 1.000`) | `335 / 718` | `225.3s` |
| **59** | **PASS** | `1/1/1` | `0.2482` | `0.0076 / 0.2150` | `0.9067 / 0.9069` | `0.8046 / 0.8024` | `764` | `475` | **`475` (46.5%)** | `0.00 / 0.00` (max `0`) | `1.000 / 1.000` (AUC `1.000 / 1.000`) | `289 / 475` | `218.3s` |

## Gate Verification & Execution Note

- **Preflight**: `min_ans_lp = -0.1338 (A), -0.1430 (B), -0.1469 (C)` (`> -0.35`), and `forward_from_layer` `max_diffs = 0.0` at both `L29` and `L59`.
- **Null calibration**: Across all `50` random half-control splits on BOTH `columns` and `isotropic` at every layer `55..59`, false discoveries at `FDP <= 0.05` were **`0` mean, `0` max**.
- **Out-of-sample Half C**: `300/300` evaluated `Dual05` neurons cleared `p <= 0.05` against BOTH Half-C `columns` and `isotropic` controls (`AUC = 1.0000`) on all 5 layers.
- **Transient L59 timeout note**: The combined `L55..L59` run completed `L55–L58` (`18m02s`) cleanly and hit a transient desktop `cudaErrorLaunchTimeout` on `L59` right around display lock screen activity; re-running `L59` alone with identical frozen code (`SHA256SUMS` unmodified and verified `OK`) completed cleanly in `218.3s`.
- **Decision (`prereg/RULES.md` §4)**: All 5 layers (`59, 58, 57, 56, 55`) proceed to full whole-layer streaming `forced_odd` (`21,504` neurons + `4,096` `columns` + `2,048` `isotropic`).

## Artifacts
- Verdicts: `L55_verdict.json .. L59_verdict.json`, `trials_summary.json`, `trials_summary.txt`
- Named examples (`forced next-piece shift (FDR <= 5%, dual-null)`): `L55_examples.txt .. L59_examples.txt`, `L55_named.json .. L59_named.json`
