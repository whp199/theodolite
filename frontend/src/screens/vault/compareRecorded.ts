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
    "at": "2026-11-22T05:05:58Z",
    "hash": "c6e6bbf519c73c04",
    "kind": "note",
    "status": "works but harms",
    "why": null,
    "name": "pronoun_pull (1/2) — E6 layer-51 ‘she’ ten",
    "edit_id": "be4328fdb60d05f8ad95d31326f3ae2b3482add121488206db2998e38b60312a",
    "edit_linears": [
      "layers.50.mlp.down_proj"
    ],
    "seconds": 38.97,
    "base_after": null,
    "note_type": "claim_verdict",
    "brief_name": "pronoun_pull",
    "brief_id": "935d9f723200a38d42a715d1a8b405b3106fb90f1639a7a7bdc3ff7b2f4b676b",
    "verdict": "works but harms",
    "reasons": [
      "near_misses abs effect (0.883765697479248, ratio 0.7033051642132468) exceeded bar (abs <= 0.25, ratio <= 0.35)."
    ],
    "claim_sentence": null,
    "summary": {
      "large": {
        "ci95": [
          -2.3305676276447453,
          -0.18462511947134375
        ],
        "controls": {
          "other_sites_matched": {
            "as_read_at_least": 0,
            "at_least": 0,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "same_site": {
            "as_read_at_least": 0,
            "at_least": 0,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 practice readings in this book",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "delivered": {
          "other_sites_matched": {
            "as_loud": 0,
            "as_loud_ratio_answer": null,
            "as_loud_ratio_end": null,
            "as_loud_ratio_prompt": null,
            "compared_at": [
              "prompt",
              "answer",
              "end"
            ],
            "controls_max": 0.018232166534289718,
            "controls_median": 0.005843355640536174,
            "controls_median_end": 0.0034432969259796664,
            "controls_median_prompt": 0.003674583975225687,
            "controls_min": 0.003746605187188834,
            "controls_read": 39,
            "disjoint_sets": 27,
            "draw": "independent",
            "edit": 0.011591552698519081,
            "edit_end": 0.03358625585678965,
            "edit_prompt": 0.009977004025131464,
            "fewest_distinct": 30,
            "fewest_eligible": 54,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.011587329674512148,
            "controls_median": 0.011469162767753005,
            "controls_median_end": 0.03223939353483729,
            "controls_median_prompt": 0.009729014243930578,
            "controls_min": 0.011398265894968063,
            "controls_read": 39,
            "edit": 0.011591552698519081,
            "edit_end": 0.03358625585678965,
            "edit_prompt": 0.009977004025131464,
            "read_at_end": true
          }
        },
        "directed_mean": 1.2575963735580444,
        "half_0": {
          "directed_mean": 0.684069037437439,
          "mean": -0.684069037437439,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 1.83112370967865,
          "mean": -1.83112370967865,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.3200641692910802,
          "floor": 0.13658809661865234,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 5.490961647221466
        },
        "mean": -1.2575963735580444,
        "n_frames": 8
      },
      "small": {
        "ci95": [
          -2.327157463015967,
          -0.18602100473458227
        ],
        "controls": {
          "other_sites_matched": {
            "as_read_at_least": 0,
            "at_least": 0,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "same_site": {
            "as_read_at_least": 0,
            "at_least": 0,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.025,
            "p_fw": 0.05,
            "passes_alpha": true,
            "tops_every_control": true
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 practice readings in this book",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "delivered": {
          "other_sites_matched": {
            "as_loud": 0,
            "as_loud_ratio_answer": null,
            "as_loud_ratio_end": null,
            "as_loud_ratio_prompt": null,
            "compared_at": [
              "prompt",
              "answer",
              "end"
            ],
            "controls_max": 0.01820593443699181,
            "controls_median": 0.0057803302188403904,
            "controls_median_end": 0.0035677241685334593,
            "controls_median_prompt": 0.0036464431323111057,
            "controls_min": 0.0037539206678047776,
            "controls_read": 39,
            "disjoint_sets": 27,
            "draw": "independent",
            "edit": 0.011481747555080801,
            "edit_end": 0.0337552402052097,
            "edit_prompt": 0.00995479611447081,
            "fewest_distinct": 30,
            "fewest_eligible": 54,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.011548775248229504,
            "controls_median": 0.011439552530646324,
            "controls_median_end": 0.03249347204109654,
            "controls_median_prompt": 0.009720656787976623,
            "controls_min": 0.011387840029783547,
            "controls_read": 39,
            "edit": 0.011481747555080801,
            "edit_end": 0.0337552402052097,
            "edit_prompt": 0.00995479611447081,
            "read_at_end": true
          }
        },
        "directed_mean": 1.2565892338752747,
        "half_0": {
          "directed_mean": 0.6901611089706421,
          "mean": -0.6901611089706421,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 1.8230173587799072,
          "mean": -1.8230173587799072,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.33453751652143165,
          "floor": 0.14922046661376953,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 5.026120189941778
        },
        "mean": -1.2565892338752747,
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
        "items": {
          "given": 4,
          "read": 3,
          "under_item_gate": 1
        },
        "kl_bar": 0.05,
        "mean_kl": 6.047049971869001e-08,
        "own_drop": 0.0,
        "own_drop_bar": 0.1,
        "passed": true
      },
      "near_misses": {
        "abs_bar": 0.25,
        "abs_effect": 0.883765697479248,
        "available": true,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 4
        },
        "mean_effect": 0.883765697479248,
        "passed": false,
        "ratio_bar": 0.35,
        "ratio_to_target": 0.7033051642132468
      },
      "neutral_text": {
        "available": true,
        "bar": 0.05,
        "control_kl_max": 0.00017886740124595235,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 2
        },
        "mean_kl": 6.249064063013066e-05,
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
      "matched_arm": {
        "directed_effect": 8.702642679214478,
        "index": 0,
        "near_miss": "within the registered bar as a share of its own effect",
        "near_miss_abs": 1.1967780590057373,
        "near_miss_share": 0.13751892420725717
      },
      "matched_by": "prompt arm 0",
      "prompt_arms": [
        {
          "directed_effect": 8.702642679214478,
          "index": 0,
          "mean_effect": -8.702642679214478,
          "near_miss_abs": 1.1967780590057373,
          "prompt_sha256": "255cdbb968d4"
        }
      ],
      "prompt_arms_state": "run"
    },
    "bars_sha256": "d79cc9a7ae4f10f18b6aee8023eaba4ba352215b4a260599892b8dca0a14df95",
    "book_seqs": [
      4,
      5,
      6,
      7,
      8,
      9
    ],
    "recorded_tag": "recorded on the 31B card (22 November 2026, verify_claim_book.jsonl)"
  },
  {
    "seq": 17,
    "at": "2026-11-22T05:06:38Z",
    "hash": "1bc5e734cf114f6d",
    "kind": "note",
    "status": "kept the base",
    "why": null,
    "name": "pronoun_pull (2/2) — E6 layer-51 ‘she’-only seven",
    "edit_id": "5398ea7874060af831d36caa902130519c117e9c17acbeee9189a8a111581bbd",
    "edit_linears": [
      "layers.50.mlp.down_proj"
    ],
    "seconds": 39.09,
    "base_after": null,
    "note_type": "claim_verdict",
    "brief_name": "pronoun_pull",
    "brief_id": "935d9f723200a38d42a715d1a8b405b3106fb90f1639a7a7bdc3ff7b2f4b676b",
    "verdict": "kept the base",
    "reasons": [
      "small class directed mean (0.004317820072174072) is below house bar (0.277607537130444).",
      "small class failed \"other_sites_matched\" controls (at_least=31/39, p_fw=1.0, alpha=0.05).",
      "small class failed \"same_site\" controls (at_least=31/39, p_fw=1.0, alpha=0.05).",
      "large class floor too loud (F=0.188, M=0.75, M < 4F).",
      "large class failed \"other_sites_matched\" controls (at_least=38/39, p_fw=1.0, alpha=0.05).",
      "large class failed \"same_site\" controls (at_least=39/39, p_fw=1.0, alpha=0.05)."
    ],
    "claim_sentence": null,
    "summary": {
      "large": {
        "ci95": [
          -0.0578766062861867,
          0.05722262412371112
        ],
        "controls": {
          "other_sites_matched": {
            "as_read_at_least": 38,
            "at_least": 38,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.975,
            "p_fw": 1.0,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "same_site": {
            "as_read_at_least": 39,
            "at_least": 39,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 1.0,
            "p_fw": 1.0,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 practice readings in this book",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "delivered": {
          "other_sites_matched": {
            "as_loud": 0,
            "as_loud_ratio_answer": null,
            "as_loud_ratio_end": null,
            "as_loud_ratio_prompt": null,
            "compared_at": [
              "prompt",
              "answer",
              "end"
            ],
            "controls_max": 0.002393139526247978,
            "controls_median": 0.0019904652726836503,
            "controls_median_end": 0.001931486011017114,
            "controls_median_prompt": 0.001968743294128217,
            "controls_min": 0.0017402016674168408,
            "controls_read": 39,
            "disjoint_sets": 36,
            "draw": "independent",
            "edit": 0.002256751758977771,
            "edit_end": 0.002361294551519677,
            "edit_prompt": 0.0026548300229478627,
            "fewest_distinct": 38,
            "fewest_eligible": 497,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.002235609630588442,
            "controls_median": 0.0022002332407282665,
            "controls_median_end": 0.002336450488655828,
            "controls_median_prompt": 0.0025629582814872265,
            "controls_min": 0.002175214496674016,
            "controls_read": 39,
            "edit": 0.002256751758977771,
            "edit_end": 0.002361294551519677,
            "edit_prompt": 0.0026548300229478627,
            "read_at_end": true
          }
        },
        "directed_mean": 0.00032699108123779297,
        "half_0": {
          "directed_mean": 0.0053827762603759766,
          "mean": -0.0053827762603759766,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": -0.004728794097900391,
          "mean": 0.004728794097900391,
          "n_frames": 4
        },
        "house_bar": {
          "bar": null,
          "floor": 0.18799781799316406,
          "floor_too_loud": true,
          "least_effect": 0.75,
          "ratio_m_over_f": 3.989408004869883
        },
        "mean": -0.00032699108123779297,
        "n_frames": 8
      },
      "small": {
        "ci95": [
          -0.04595867139756672,
          0.03732303125321858
        ],
        "controls": {
          "other_sites_matched": {
            "as_read_at_least": 31,
            "at_least": 31,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.8,
            "p_fw": 1.0,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "same_site": {
            "as_read_at_least": 31,
            "at_least": 31,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.8,
            "p_fw": 1.0,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "sham_target": {
            "count": 0,
            "note": "selection not matched, 1 practice readings in this book",
            "p_emp": null,
            "p_fw": null,
            "passes_alpha": null
          }
        },
        "delivered": {
          "other_sites_matched": {
            "as_loud": 0,
            "as_loud_ratio_answer": null,
            "as_loud_ratio_end": null,
            "as_loud_ratio_prompt": null,
            "compared_at": [
              "prompt",
              "answer",
              "end"
            ],
            "controls_max": 0.0024264236562885344,
            "controls_median": 0.001992487726965919,
            "controls_median_end": 0.0019584969559218735,
            "controls_median_prompt": 0.0019446627993602306,
            "controls_min": 0.0016849732783157378,
            "controls_read": 39,
            "disjoint_sets": 36,
            "draw": "independent",
            "edit": 0.002283267443999648,
            "edit_end": 0.0023702983889961615,
            "edit_prompt": 0.0026891255693044513,
            "fewest_distinct": 38,
            "fewest_eligible": 497,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.002275909806485288,
            "controls_median": 0.002202257892349735,
            "controls_median_end": 0.0023436172486981377,
            "controls_median_prompt": 0.0025491128908470273,
            "controls_min": 0.002191766005125828,
            "controls_read": 39,
            "edit": 0.002283267443999648,
            "edit_end": 0.0023702983889961615,
            "edit_prompt": 0.0026891255693044513,
            "read_at_end": true
          }
        },
        "directed_mean": 0.004317820072174072,
        "half_0": {
          "directed_mean": 0.014072656631469727,
          "mean": -0.014072656631469727,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": -0.005437016487121582,
          "mean": 0.005437016487121582,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.277607537130444,
          "floor": 0.10275459289550781,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 7.298943812299297
        },
        "mean": -0.004317820072174072,
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
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 2
        },
        "kl_bar": 0.05,
        "mean_kl": 4.4691296885446263e-08,
        "own_drop": 0.007652759551996624,
        "own_drop_bar": 0.1,
        "passed": true
      },
      "near_misses": {
        "abs_bar": 0.25,
        "abs_effect": 0.08511090278625488,
        "available": true,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 4
        },
        "mean_effect": 0.0121382474899292,
        "passed": true,
        "ratio_bar": 0.35,
        "ratio_to_target": 260.2850893182647
      },
      "neutral_text": {
        "available": true,
        "bar": 0.05,
        "control_kl_max": 5.577195952355396e-05,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 3
        },
        "mean_kl": 4.680823872149631e-05,
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
      "matched_arm": {
        "directed_effect": 7.8406935930252075,
        "index": 0,
        "near_miss": "within the registered bar as a share of its own effect",
        "near_miss_abs": 1.0193231105804443,
        "near_miss_share": 0.13000420160369444
      },
      "matched_by": "prompt arm 0",
      "prompt_arms": [
        {
          "directed_effect": 7.8406935930252075,
          "index": 0,
          "mean_effect": -7.8406935930252075,
          "near_miss_abs": 1.0193231105804443,
          "prompt_sha256": "255cdbb968d4"
        }
      ],
      "prompt_arms_state": "run"
    },
    "bars_sha256": "d79cc9a7ae4f10f18b6aee8023eaba4ba352215b4a260599892b8dca0a14df95",
    "book_seqs": [
      11,
      12,
      13,
      14,
      15,
      16
    ],
    "recorded_tag": "recorded on the 31B card (22 November 2026, verify_claim_book.jsonl)"
  }
];

/** The book the cards are from: its last run, as the server dates a recorded book, and its head. */
export const RECORDED_BOOK_LAST_RUN = "last run 22 Nov 2026, UTC";
export const RECORDED_BOOK_HEAD = "4e1b8729a3c15d49";
