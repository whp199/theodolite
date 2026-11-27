# Path 2 (`u2_rollout_L58_L59`): Dual-Control Rollout Extension at Layers 58 and 59

Fixed on 27 November 2026 before any confirmatory `match_things` rollout answer of this extension existed (`prereg_at`, `SHA256SUMS`, `CODE_SHA256SUMS`).

## Extension Scope & Cohort Selection
Extends the cleared `u2_rollout` (`L55 -> L56 -> L57`, 3-for-3 pass on both Track 2A and Track 2B) upward walk to `L58 -> L59` using the exact same `match_things` dual-control protocol:
- **Neurons + `columns` (`L<L>-match_things-u2-cols`)**: `1,022` fresh neurons per layer (strictly excluding `probe1_in_use_causal.json` `L58`/`L59` AND every neuron in Path 1 `experiments/results/in_use_lens_trial/prereg/neurons_L58.txt` and `neurons_L59.txt`, verified `0` overlap) + `1,024` `columns` controls (`seed = 21`), `match_things` (`48` answers/unit, `max_new = 48`, `batch = 64`).
- **`isotropic` controls (`L<L>-match_things-u2-iso`)**: `--neurons none --random-controls 1024 --control-kind isotropic --seed 22`, verifying 12/12 bit-identical baselines via `census.join_controls` (and verified beforehand via `cz.census_jobs` + `cz.plan_batches(..., 64)` to share identical chunk-0 baseline batch shapes).

## Layer 59 Dose Prerequisite (`MATCH_CEIL[59]`)
- **`L58` is ready immediately**: `58: (1.16, 1.27)` ceilings is already populated in `backend/census.py:MATCH_CEIL`.
- **`L59` requires a pre-run fine dose pilot**: `59` is not yet in `MATCH_CEIL` because `fine_pilot_high.json` (`0.53..2.21` of stream) bottomed at `5.85` nats (`> 1.4` nats), whereas `steer_dose.json` places `1.0` and `1.4` nats between `0.2` (`0.078` nats) and `0.4` (`1.699` nats) of the stream. Accordingly, `walk.sh` runs `L58` cleanly and gates `L59` on `59 in cz.MATCH_CEIL` (requiring `scripts/census_dose_pilot.py measure --layers 59 --ladder fine` + `doses --targets 1.0,1.4` and an explicit `MATCH_CEIL[59]` amendment before `L59` executes, keeping `backend/census.py` untouched during Kit 8 and `L58`).

## Pre-Registered Confirmatory Gates (Per Layer, Upward Walk `58 -> 59`)
Upward walk runs `L58` first; advances to `L59` ONLY if `pass_either = pass_track_2a or pass_track_2b` at `L58` (evaluated identically via `scripts/u2_rollout_read.py`):

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
- **Track 2B pass**: Pushing neurons causes them to amplify their static weight-lens vocabulary beyond both control families (kept distinct from Track 2A).
- **Nothing opens any layer or mutates `backend/census.py` / `MANIFEST.json` without explicit owner approval.**
