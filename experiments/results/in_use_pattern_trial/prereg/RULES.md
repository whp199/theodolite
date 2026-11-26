# Path 3 (`in_use_pattern`) Confirmatory Trial at Layers 2, 14, and 20 (Boundary Probe)

Fixed on 26 November 2026 before any confirmatory GPU forward pass ran (`prereg_at`, `SHA256SUMS`, `CODE_SHA256SUMS`).

## Design (Completely Self-Contained Local `Gemma-4-31B-it`)
- **Cohorts**: `128` fresh `doc_4_8` faces at `L2`, `128` at `L14`, and `64` boundary probe at `L20` (`faces_L<L>.txt`), strictly excluding `pattern_tests_pilot`, Probe 1, and Probe 3.
- **Strict `<eos>` Document-Half Isolation**: Fit patterns from up to `8` `doc_h0` events (preferring `pure_h0`); judge `blind_near` on up to `10` `doc_h1` events (preferring `pure_h1`); test `behind` counterfactuals behind an UNSEEN `doc_h1` passage (never from `doc_h0` fit passages).
- **Live GPU Symmetric Near-Miss Verification & `K = 3` Shuffled Nulls**:
  - Quiet near-miss candidates (`rel_span < 0.25`) are verified via `_StopAfterLayer(l)` on `doc_h1` outside each face's own documents and windows, matching left/right span widths (`+-1` token) and sorting by length symmetry.
  - Zero-decode single-prefill `log P(a) - log P(b)` (`assistant_prefix = ""`) scores each pattern vs `K = 3` deranged same-layer `other_pattern` nulls.
- **Physical `behind` Counterfactuals + Cross-Face Null**: Generates `3 FIRE` + `3 QUIET` minimal-pair continuations behind held `doc_h1` contexts (`max_new=160`) and runs `_StopAfterLayer(l)` on `own_dropped`, own `FIRE`, own `QUIET`, and the next face's `FIRE` continuations (`other_pattern_fire_rate`).

## Pre-Registered Confirmatory Gates (Independent per Layer)
1. **`G1` (`blind_near` Pooled & Null Calibration)**: `own_share >= 0.72`, `own_share - mean_other_share >= 0.18`, `own_binom_p <= 1e-6`, `mean_other_share in [0.45, 0.55]`, `mean_other_p05_rate_ge8 <= 0.08`, paired Wilcoxon `own_face_acc > mean_other_face_acc` `p <= 1e-6`, `median_p_ab_mass >= 0.90`.
2. **`G2` (`blind_near` Individual Yield)**: `>= 90%` of faces reach `>= 8` pairs; among `>= 8`-pair faces, `p < 0.05 >= 0.30` AND `BH q <= 0.10 >= 0.20`.
3. **`G3` (Unseen-Doc `behind` Causality)**: `own_holds_gt_0.5 >= 0.90`, `written_fire_rate >= 0.30`, `written_fire_rate - near_miss_false_rate >= 0.20`, `near_miss_false_rate <= 0.10`, `other_pattern_fire_rate <= 0.10`, `clean_2of3_and_0quiet_rate >= 0.25`, `mean_apart >= 0.70`.
4. **`G4` (Provenance & Zero Corpus Leaks)**: Clean git tree, verified `SHA256SUMS` & `CODE_SHA256SUMS`, zero raw corpus text stored.

## What a Pass Shows — and Does Not Show
- **Shows**: Local `Gemma-4-31B-it` produces human-interpretable firing descriptions on unseen documents that both distinguish true firings from length-matched quiet near-misses (`blind_near`) and physically steer counterfactual activations (`behind`), with `L20` acting as the depth boundary check.
- **Does Not Show**: Downstream output-side causal behavior; early-layer push censuses remain non-viable (`L1` killed). Opening a reader-mode layer still requires owner sign-off.
