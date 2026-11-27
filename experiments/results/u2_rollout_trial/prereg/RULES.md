# Path 2 (`u2_rollout`): Dual-Control Rollout Trial at Layers 55, 56, and 57

Fixed on 26 November 2026 before any confirmatory `match_things` rollout answer of this trial existed (`prereg_at`, `SHA256SUMS`, `CODE_SHA256SUMS`).

## Why Two Controls & Two Independent Tracks
Previous exploratory `things_walk` at `L55` (`L55_trial`) used `isotropic` controls only, and by `L59` isotropic controls fail exchangeability against `columns` mixtures (`29–31%`). Because `L55–L57` approach unembedding, **both control families are drawn afresh** at every layer tested:
- **Neurons + `columns` (`L<L>-match_things-u2-cols`)**: `1,022` fresh neurons (disjoint from `things_walk`, Probe 1, `L56` `pattern_tests_pilot`, `L57` probes/calibs, AND Path 1 `neurons_L57.txt`) + `1,024` `columns` controls (`seed = 21`), `match_things` (`48` answers/unit, `max_new = 48`, `batch = 64`).
- **`isotropic` controls (`L<L>-match_things-u2-iso`)**: `--neurons none --random-controls 1024 --control-kind isotropic --seed 22`, verifying 12/12 bit-identical baselines via `census.join_controls`.

## Pre-Registered Confirmatory Gates (Per Layer, Upward Walk `55 -> 56 -> 57`)
Upward walk runs `L55` first; advances to `L56` (then `L57`) ONLY if at least one track passes cleanly (`pass_either = pass_track_2a or pass_track_2b`):

### Track 2A: Behavioral Per-Neuron Rollout (`U2-T_dec`, via `backend/census_fdr.py`)
All four gates required for `pass_track_2a`:
1. **`G-Yield`**: `N3` coverage `>= 0.50` on both controls; `Dual05-N3` (passing `FDP <= 0.05` AND `N3`-covered against BOTH `columns` AND `isotropic`) **`>= 82 / 1,022 (8.0%)`**, with 200-boot `p10 >= 50` and `frac_zero == 0.0` on both nulls.
2. **`G-CrossPool`**: Bidirectional `T_dec` `FDP <= 0.05` false discovery between control pools **`<= 0.02`** (`iso->col <= 0.02` AND `col->iso <= 0.02`).
3. **`G-C4`**: On `out_of_sample_3way` (`A4/B4 -> unseen C4`) against `columns`: `fit_named >= 40`, `frac_p05_on_C4 >= 0.80`, AND `c4_auc_vs_paired_ctl >= 0.85`.
4. **`G-Sanity`**: Exact 12/12 baseline match, complete row counts (`98,220` + `49,164`), and `code_symmetry_check` `fdp05_false_max <= 1` on both controls.

### Track 2B: Corrected Own-Word Weight-Word Verification (`U2-weight_words`)
Evaluated on own unembedded top-10 `eye_10` word-pieces (neurons on `lens.npz`, controls on their OWN `label_top_ids`), reported and gated independently of Track 2A:
1. **`G-WW-Lead`**: `whole`, `half A`, and `half B` `corr_lead_auc >= 0.60` vs BOTH `columns` and `isotropic`; shuffled-label `AUC in [0.47, 0.53]`; `rho_AB_neurons >= 0.50`.
2. **`G-WW-Yield`**: `Dual05` (`BH q <= 0.05` simultaneously vs `columns` and `isotropic`) **`>= 103 / 1,022 (10.0%)`**, with bidirectional `iso <-> col` `q <= 0.05` false discovery rate **`<= 0.02`**.

## What a Pass Shows — and Does Not Show
- **Track 2A pass**: Behavioral per-neuron semantic rollouts replicate across disjoint halves and unseen `C4` questions beyond both `columns` and `isotropic` nulls.
- **Track 2B pass**: Pushing neurons causes them to amplify their static weight-lens vocabulary beyond both control families (weight-lens-coupled behavior, kept distinct from Track 2A).
- **Nothing opens any layer or mutates `backend/census.py` / `MANIFEST.json` without explicit owner approval.**
