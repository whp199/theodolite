/**
 * The 31B replay cards the Vault's `compare` tab shows when no live server answers, so a person can inspect the card
 * check's two claim verdicts with no model loaded.
 *
 * MADE BY `scripts/level_recorded_cards.py`; do not edit by hand. Each card is the server's own replay card
 * (`backend/level_routes.py` `build_replay_card`) for the line of
 * `experiments/results/levelling/verify_claim_book.jsonl` it names by number, time and hash, figures unrounded, with
 * the edit's linear (from the same book), the reading's seconds and the edit's label (from `verify_claim.json`) and
 * the tag added. After `scripts/verify_level_claim.py` rewrites the book, run the script with --write;
 * `scripts/check_level_page.py` holds this file to the book.
 */
import type { ReplayCard } from "./compare";

export const RECORDED_REPLAY_CARDS: ReplayCard[] = [
  {
    "seq": 10,
    "at": "2026-11-19T06:09:48Z",
    "hash": "0c28778b9ad8bc10",
    "kind": "note",
    "status": "works but harms",
    "why": null,
    "name": "pronoun_pull (1/2) — E6 layer-51 ‘she’ ten",
    "edit_id": "be4328fdb60d05f8ad95d31326f3ae2b3482add121488206db2998e38b60312a",
    "edit_linears": [
      "layers.50.mlp.down_proj"
    ],
    "seconds": 44.75,
    "base_after": null,
    "note_type": "claim_verdict",
    "brief_name": "pronoun_pull",
    "brief_id": "92c7776f8ed0788eb12354893e8c3fe78dc5b167b8788a8e98a821b6c0bccd3e",
    "verdict": "works but harms",
    "reasons": [
      "near_misses abs effect (0.8008685111999512, ratio 0.4862498339144315) exceeded bar (abs <= 0.25, ratio <= 0.35)."
    ],
    "claim_sentence": null,
    "summary": {
      "large": {
        "ci95": [
          -2.6625279898725225,
          -0.6933515553392694
        ],
        "controls": {
          "other_sites_matched": {
            "at_least": 0,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "same_site": {
            "at_least": 0,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 read in practice",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "directed_mean": 1.677939772605896,
        "half_0": {
          "directed_mean": 1.8444660902023315,
          "mean": -1.8444660902023315,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 1.5114134550094604,
          "mean": -1.5114134550094604,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.35251792780961516,
          "floor": 0.16569185256958008,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 4.526474828839563
        },
        "mean": -1.677939772605896,
        "n_frames": 8
      },
      "small": {
        "ci95": [
          -2.667993781519038,
          -0.6260680580614981
        ],
        "controls": {
          "other_sites_matched": {
            "at_least": 0,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "same_site": {
            "at_least": 0,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 read in practice",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "directed_mean": 1.647030919790268,
        "half_0": {
          "directed_mean": 1.8218348026275635,
          "mean": -1.8218348026275635,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 1.4722270369529724,
          "mean": -1.4722270369529724,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.28977507320005236,
          "floor": 0.11195945739746094,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 6.698853472801922
        },
        "mean": -1.647030919790268,
        "n_frames": 8
      }
    },
    "batteries": {
      "habitat": {
        "available": false,
        "bar": 0.25,
        "gating": false,
        "mean_kl": null,
        "passed": true
      },
      "labelled_panel": {
        "available": true,
        "gating": true,
        "kl_bar": 0.05,
        "mean_kl": 4.906697448348041e-08,
        "own_drop": 0.0,
        "own_drop_bar": 0.1,
        "passed": true
      },
      "near_misses": {
        "abs_bar": 0.25,
        "abs_effect": 0.8008685111999512,
        "available": true,
        "gating": true,
        "mean_effect": 0.7701230049133301,
        "passed": false,
        "ratio_bar": 0.35,
        "ratio_to_target": 0.4862498339144315
      },
      "neutral_text": {
        "available": true,
        "bar": 0.05,
        "control_kl_max": 6.430346093111439e-05,
        "gating": true,
        "mean_kl": 6.926778496563202e-05,
        "passed": true
      }
    },
    "cheaper_knobs": {
      "activation_twin": {
        "is_exact_mul": true,
        "note": "no search over cheaper knobs; not compared with a trained adapter"
      },
      "decoding_bias": {
        "available": true,
        "gating": false,
        "note": "logit offset on named pieces shifts end-position odds at zero extra forwards"
      },
      "matched_by": null,
      "prompt_arms": [
        {
          "directed_effect": 8.115991353988647,
          "index": 0,
          "mean_effect": -8.115991353988647,
          "near_miss_abs": 1.8319659233093262,
          "prompt_sha256": "255cdbb968d4"
        }
      ]
    },
    "bars_sha256": "80a466b7f6649e5bc3c8c8ca2675eb903d9413ead4316346d11fc0762ef18571",
    "book_seqs": [
      4,
      5,
      6,
      7,
      8,
      9
    ],
    "recorded_tag": "recorded on the 31B card (19 November 2026, verify_claim_book.jsonl)"
  },
  {
    "seq": 17,
    "at": "2026-11-19T06:10:34Z",
    "hash": "b78d9dc4a9ab7eea",
    "kind": "note",
    "status": "kept the base",
    "why": null,
    "name": "pronoun_pull (2/2) — E6 layer-51 ‘she’-only seven",
    "edit_id": "5398ea7874060af831d36caa902130519c117e9c17acbeee9189a8a111581bbd",
    "edit_linears": [
      "layers.50.mlp.down_proj"
    ],
    "seconds": 45.1,
    "base_after": null,
    "note_type": "claim_verdict",
    "brief_name": "pronoun_pull",
    "brief_id": "92c7776f8ed0788eb12354893e8c3fe78dc5b167b8788a8e98a821b6c0bccd3e",
    "verdict": "kept the base",
    "reasons": [
      "small class directed mean (0.03061985969543457) is below house bar (0.2762680684735599).",
      "small class failed \"same_site\" controls (at_least=3/39, p_fw=0.2, alpha=0.05).",
      "large class directed mean (0.010810941457748413) is below house bar (0.30963461415684396).",
      "large class failed \"other_sites_matched\" controls (at_least=18/39, p_fw=0.95, alpha=0.05).",
      "large class failed \"same_site\" controls (at_least=20/39, p_fw=1.0, alpha=0.05)."
    ],
    "claim_sentence": null,
    "summary": {
      "large": {
        "ci95": [
          -0.053028025534114114,
          0.03140614261861729
        ],
        "controls": {
          "other_sites_matched": {
            "at_least": 18,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.475,
            "p_fw": 0.95,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "same_site": {
            "at_least": 20,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.525,
            "p_fw": 1.0,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 read in practice",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "directed_mean": 0.010810941457748413,
        "half_0": {
          "directed_mean": -0.020874977111816406,
          "mean": 0.020874977111816406,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 0.04249686002731323,
          "mean": -0.04249686002731323,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.30963461415684396,
          "floor": 0.12783145904541016,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 5.867100364813751
        },
        "mean": -0.010810941457748413,
        "n_frames": 8
      },
      "small": {
        "ci95": [
          -0.09394557161178839,
          0.03270585222091925
        ],
        "controls": {
          "other_sites_matched": {
            "at_least": 0,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "same_site": {
            "at_least": 3,
            "count": 39,
            "non_finite": 0,
            "p_emp": 0.1,
            "p_fw": 0.2,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 read in practice",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "directed_mean": 0.03061985969543457,
        "half_0": {
          "directed_mean": 0.048035621643066406,
          "mean": -0.048035621643066406,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 0.013204097747802734,
          "mean": -0.013204097747802734,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.2762680684735599,
          "floor": 0.10176539421081543,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 7.36989234715991
        },
        "mean": -0.03061985969543457,
        "n_frames": 8
      }
    },
    "batteries": {
      "habitat": {
        "available": false,
        "bar": 0.25,
        "gating": false,
        "mean_kl": null,
        "passed": true
      },
      "labelled_panel": {
        "available": true,
        "gating": true,
        "kl_bar": 0.05,
        "mean_kl": 1.2809641125599569e-08,
        "own_drop": 0.007652769486043098,
        "own_drop_bar": 0.1,
        "passed": true
      },
      "near_misses": {
        "abs_bar": 0.25,
        "abs_effect": 0.01995706558227539,
        "available": true,
        "gating": true,
        "mean_effect": 0.005430698394775391,
        "passed": true,
        "ratio_bar": 0.35,
        "ratio_to_target": 1.8460062576670204
      },
      "neutral_text": {
        "available": true,
        "bar": 0.05,
        "control_kl_max": 3.4018107726296876e-05,
        "gating": true,
        "mean_kl": 2.9289720032465993e-05,
        "passed": true
      }
    },
    "cheaper_knobs": {
      "activation_twin": {
        "is_exact_mul": true,
        "note": "no search over cheaper knobs; not compared with a trained adapter"
      },
      "decoding_bias": {
        "available": true,
        "gating": false,
        "note": "logit offset on named pieces shifts end-position odds at zero extra forwards"
      },
      "matched_by": null,
      "prompt_arms": [
        {
          "directed_effect": 7.676213845610619,
          "index": 0,
          "mean_effect": -7.676213845610619,
          "near_miss_abs": 1.9546449184417725,
          "prompt_sha256": "255cdbb968d4"
        }
      ]
    },
    "bars_sha256": "80a466b7f6649e5bc3c8c8ca2675eb903d9413ead4316346d11fc0762ef18571",
    "book_seqs": [
      11,
      12,
      13,
      14,
      15,
      16
    ],
    "recorded_tag": "recorded on the 31B card (19 November 2026, verify_claim_book.jsonl)"
  }
];
