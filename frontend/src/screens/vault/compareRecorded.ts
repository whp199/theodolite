/**
 * Exact recorded 31B card replay cards (measured on the RTX 5090 Gemma-4-31B-it W4A16 checkpoint, 19 November 2026)
 * used by the Vault's `compare` tab when no live server answers, so a person can inspect genuine 31B readings and
 * Phase 3 Claim Sheets with zero model loaded.
 *
 * Source files:
 * - `experiments/results/levelling/verify_claim.json` & `verify_claim_book.jsonl`
 * - `experiments/results/levelling/verify_door_book.jsonl`
 */
import type { ReplayCard } from "./compare";

export const RECORDED_REPLAY_CARDS: ReplayCard[] = [
  {
    seq: 10,
    at: "2026-11-19T06:08:50Z",
    hash: "b84e19c3a7d102f4",
    kind: "note",
    note_type: "claim_verdict",
    status: "works but harms",
    verdict: "works but harms",
    name: "pronoun_pull (1/2) — E6 layer-51 ‘she’ ten",
    brief_name: "pronoun_pull",
    brief_id: "92c7776f8ed0",
    edit_id: "be4328fdb60d05f8",
    edit_linears: ["layers.50.mlp.down_proj"],
    seconds: 44.75,
    bars_sha256: "80a466b7f664",
    reasons: [
      "near_misses abs effect (0.8009, ratio 0.4862) exceeded bar (abs <= 0.25, ratio <= 0.35).",
    ],
    summary: {
      small: {
        n_frames: 8,
        mean: -1.647,
        directed_mean: 1.647,
        ci95: [-2.668, -0.6261],
        half_0: { n_frames: 4, mean: -1.8218, directed_mean: 1.8218 },
        half_1: { n_frames: 4, mean: -1.4722, directed_mean: 1.4722 },
        house_bar: { floor: 0.112, least_effect: 0.75, bar: 0.2898, floor_too_loud: false },
        controls: {
          other_sites_matched: { count: 39, at_least: 0, p_emp: 0.025, p_fw: 0.05, passes_alpha: true, tops_every_control: true },
          same_site: { count: 39, at_least: 0, p_emp: 0.025, p_fw: 0.05, passes_alpha: true, tops_every_control: true },
        },
      },
      large: {
        n_frames: 8,
        mean: -1.6779,
        directed_mean: 1.6779,
        ci95: [-2.6625, -0.6934],
        half_0: { n_frames: 4, mean: -1.8445, directed_mean: 1.8445 },
        half_1: { n_frames: 4, mean: -1.5114, directed_mean: 1.5114 },
        house_bar: { floor: 0.1657, least_effect: 0.75, bar: 0.3525, floor_too_loud: false },
        controls: {
          other_sites_matched: { count: 39, at_least: 0, p_emp: 0.025, p_fw: 0.05, passes_alpha: true, tops_every_control: true },
          same_site: { count: 39, at_least: 0, p_emp: 0.025, p_fw: 0.05, passes_alpha: true, tops_every_control: true },
        },
      },
    },
    batteries: {
      neutral_text: { available: true, gating: true, passed: true, mean_kl: 0.000031, bar: 0.05 },
      near_misses: { available: true, gating: true, passed: false, mean_effect: 0.7701, abs_effect: 0.8009, ratio_to_target: 0.4862, abs_bar: 0.25, ratio_bar: 0.35 },
      labelled_panel: { available: true, gating: true, passed: true, own_drop: 0.0, mean_kl: 0.00000005, own_drop_bar: 0.1, kl_bar: 0.05 },
    },
    cheaper_knobs: {
      matched_by: null,
      prompt_arms: [{ index: 0, prompt_sha256: "255cdbb968d4", directed_effect: 8.116, near_miss_abs: 1.832 }],
    },
    recorded_tag: "recorded on the 31B card (19 November 2026, verify_claim_book.jsonl)",
  },
  {
    seq: 17,
    at: "2026-11-19T06:09:35Z",
    hash: "6cf3621dbee8c2a3",
    kind: "note",
    note_type: "claim_verdict",
    status: "kept the base",
    verdict: "kept the base",
    name: "pronoun_pull (2/2) — E6 layer-51 ‘she’-only seven",
    brief_name: "pronoun_pull",
    brief_id: "92c7776f8ed0",
    edit_id: "5398ea7874060af8",
    edit_linears: ["layers.50.mlp.down_proj"],
    seconds: 45.1,
    bars_sha256: "80a466b7f664",
    reasons: [
      "small class directed mean (0.0306) is below house bar (0.2763).",
      "large class directed mean (0.0108) is below house bar (0.3096).",
      "large class failed other_sites_matched (18/39, p_fw 0.95) and same_site (20/39, p_fw 1.00).",
    ],
    summary: {
      small: {
        n_frames: 8,
        mean: -0.0306,
        directed_mean: 0.0306,
        ci95: [-0.0939, 0.0327],
        half_0: { n_frames: 4, mean: -0.048, directed_mean: 0.048 },
        half_1: { n_frames: 4, mean: -0.0132, directed_mean: 0.0132 },
        house_bar: { floor: 0.1018, least_effect: 0.75, bar: 0.2763, floor_too_loud: false },
        controls: {
          other_sites_matched: { count: 39, at_least: 0, p_emp: 0.025, p_fw: 0.05, passes_alpha: true, tops_every_control: true },
          same_site: { count: 39, at_least: 3, p_emp: 0.1, p_fw: 0.2, passes_alpha: false, tops_every_control: false },
        },
      },
      large: {
        n_frames: 8,
        mean: -0.0108,
        directed_mean: 0.0108,
        ci95: [-0.053, 0.0314],
        half_0: { n_frames: 4, mean: 0.0209, directed_mean: -0.0209 },
        half_1: { n_frames: 4, mean: -0.0425, directed_mean: 0.0425 },
        house_bar: { floor: 0.1278, least_effect: 0.75, bar: 0.3096, floor_too_loud: false },
        controls: {
          other_sites_matched: { count: 39, at_least: 18, p_emp: 0.475, p_fw: 0.95, passes_alpha: false, tops_every_control: false },
          same_site: { count: 39, at_least: 20, p_emp: 0.525, p_fw: 1.0, passes_alpha: false, tops_every_control: false },
        },
      },
    },
    batteries: {
      neutral_text: { available: true, gating: true, passed: true, mean_kl: 0.000029, bar: 0.05 },
      near_misses: { available: true, gating: true, passed: true, mean_effect: 0.0054, abs_effect: 0.02, ratio_to_target: 1.846, abs_bar: 0.25, ratio_bar: 0.35 },
      labelled_panel: { available: true, gating: true, passed: true, own_drop: 0.0077, mean_kl: 0.00000001, own_drop_bar: 0.1, kl_bar: 0.05 },
    },
    cheaper_knobs: {
      matched_by: null,
      prompt_arms: [{ index: 0, prompt_sha256: "255cdbb968d4", directed_effect: 7.6762, near_miss_abs: 1.9546 }],
    },
    recorded_tag: "recorded on the 31B card (19 November 2026, verify_claim_book.jsonl)",
  },
];
