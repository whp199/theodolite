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
    "at": "2026-11-22T10:18:44Z",
    "hash": "ea80c24ece006b91",
    "kind": "note",
    "status": "works but harms",
    "why": null,
    "name": "pronoun_pull (1/2) — E6 layer-51 ‘she’ ten",
    "edit_id": "be4328fdb60d05f8ad95d31326f3ae2b3482add121488206db2998e38b60312a",
    "edit_linears": [
      "layers.50.mlp.down_proj"
    ],
    "seconds": 38.61,
    "base_after": null,
    "note_type": "claim_verdict",
    "brief_name": "pronoun_pull",
    "brief_id": "935d9f723200a38d42a715d1a8b405b3106fb90f1639a7a7bdc3ff7b2f4b676b",
    "verdict": "works but harms",
    "reasons": [
      "near_misses abs effect (1.0118346214294434, ratio 1.2373741971736372) exceeded bar (abs <= 0.25, ratio <= 0.35)."
    ],
    "claim_sentence": null,
    "summary": {
      "large": {
        "ci95": [
          -2.0591650698309243,
          0.37844849563887717
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
            "controls_max": 0.018185526365414262,
            "controls_median": 0.006868030875921249,
            "controls_median_end": 0.003659663867438212,
            "controls_median_prompt": 0.003479682927718386,
            "controls_min": 0.004119543562410399,
            "controls_read": 39,
            "disjoint_sets": 19,
            "draw": "independent",
            "edit": 0.013793226389680058,
            "edit_end": 0.02630000122007914,
            "edit_prompt": 0.010326140327379107,
            "fewest_distinct": 19,
            "fewest_eligible": 30,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.013726360921282321,
            "controls_median": 0.013687497761566192,
            "controls_median_end": 0.02546828699996695,
            "controls_median_prompt": 0.010120352439116687,
            "controls_min": 0.01363189151743427,
            "controls_read": 39,
            "edit": 0.013793226389680058,
            "edit_end": 0.02630000122007914,
            "edit_prompt": 0.010326140327379107,
            "read_at_end": true
          }
        },
        "directed_mean": 0.8403582870960236,
        "half_0": {
          "directed_mean": 0.3939182758331299,
          "mean": -0.3939182758331299,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 1.2867982983589172,
          "mean": -1.2867982983589172,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.3040909429029966,
          "floor": 0.12329506874084473,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 6.08296834301162
        },
        "mean": -0.8403582870960236,
        "n_frames": 8
      },
      "small": {
        "ci95": [
          -2.0582735291348007,
          0.4228189936504867
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
            "controls_max": 0.018157439306378365,
            "controls_median": 0.006890708056744188,
            "controls_median_end": 0.0034723250864772126,
            "controls_median_prompt": 0.0035029210266657174,
            "controls_min": 0.004160105308983475,
            "controls_read": 39,
            "disjoint_sets": 19,
            "draw": "independent",
            "edit": 0.013824402471072972,
            "edit_end": 0.026464300783118233,
            "edit_prompt": 0.01033506472595036,
            "fewest_distinct": 19,
            "fewest_eligible": 30,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.013776610663626343,
            "controls_median": 0.013719892187509686,
            "controls_median_end": 0.02568600806989707,
            "controls_median_prompt": 0.01011133607244119,
            "controls_min": 0.013668540806975216,
            "controls_read": 39,
            "edit": 0.013824402471072972,
            "edit_end": 0.026464300783118233,
            "edit_prompt": 0.01033506472595036,
            "read_at_end": true
          }
        },
        "directed_mean": 0.817727267742157,
        "half_0": {
          "directed_mean": 0.3512570858001709,
          "mean": -0.3512570858001709,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 1.284197449684143,
          "mean": -1.284197449684143,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.335552616092475,
          "floor": 0.15012741088867188,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 4.995756574768136
        },
        "mean": -0.817727267742157,
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
        "mean_kl": 6.20622628844103e-08,
        "own_drop": 0.0,
        "own_drop_bar": 0.1,
        "passed": true
      },
      "near_misses": {
        "abs_bar": 0.25,
        "abs_effect": 1.0118346214294434,
        "available": true,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 4
        },
        "mean_effect": 1.0118346214294434,
        "passed": false,
        "ratio_bar": 0.35,
        "ratio_to_target": 1.2373741971736372
      },
      "neutral_text": {
        "available": true,
        "bar": 0.05,
        "control_kl_max": 4.0776191212898993e-05,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 3
        },
        "mean_kl": 3.2802249734231737e-05,
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
        "directed_effect": 8.1468985080719,
        "index": 0,
        "near_miss": "within the registered bar as a share of its own effect",
        "near_miss_abs": 1.9119431972503662,
        "near_miss_share": 0.2346835664340269
      },
      "matched_by": "prompt arm 0",
      "prompt_arms": [
        {
          "directed_effect": 8.1468985080719,
          "index": 0,
          "mean_effect": -8.1468985080719,
          "near_miss_abs": 1.9119431972503662,
          "prompt_sha256": "255cdbb968d4"
        }
      ],
      "prompt_arms_state": "run"
    },
    "bars_sha256": "170f517707593622d7a84e3a4eb8d1f5b2250524231a43611258d829b0744d12",
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
    "at": "2026-11-22T10:19:23Z",
    "hash": "e72d551b9833b5e2",
    "kind": "note",
    "status": "kept the base",
    "why": null,
    "name": "pronoun_pull (2/2) — E6 layer-51 ‘she’-only seven",
    "edit_id": "5398ea7874060af831d36caa902130519c117e9c17acbeee9189a8a111581bbd",
    "edit_linears": [
      "layers.50.mlp.down_proj"
    ],
    "seconds": 39.07,
    "base_after": null,
    "note_type": "claim_verdict",
    "brief_name": "pronoun_pull",
    "brief_id": "935d9f723200a38d42a715d1a8b405b3106fb90f1639a7a7bdc3ff7b2f4b676b",
    "verdict": "kept the base",
    "reasons": [
      "small class directed mean (0.012635082006454468) is below house bar (0.3135638249236034).",
      "small class failed \"other_sites_matched\" controls (at_least=10/39, p_fw=0.55, alpha=0.05).",
      "small class failed \"same_site\" controls (at_least=23/39, p_fw=1.0, alpha=0.05).",
      "large class directed mean (0.001885145902633667) is below house bar (0.21093919541419348).",
      "large class failed \"other_sites_matched\" controls (at_least=36/39, p_fw=1.0, alpha=0.05).",
      "large class failed \"same_site\" controls (at_least=37/39, p_fw=1.0, alpha=0.05)."
    ],
    "claim_sentence": null,
    "summary": {
      "large": {
        "ci95": [
          -0.031482581879716476,
          0.027712290074449145
        ],
        "controls": {
          "other_sites_matched": {
            "as_read_at_least": 36,
            "at_least": 36,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.925,
            "p_fw": 1.0,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "same_site": {
            "as_read_at_least": 37,
            "at_least": 37,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.95,
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
            "controls_max": 0.0026263989857397974,
            "controls_median": 0.0020477364014368504,
            "controls_median_end": 0.0018567527658888139,
            "controls_median_prompt": 0.0019526628020685166,
            "controls_min": 0.001664399926085025,
            "controls_read": 39,
            "disjoint_sets": 35,
            "draw": "independent",
            "edit": 0.0023328641545958817,
            "edit_end": 0.0023019198124529794,
            "edit_prompt": 0.002599397615995258,
            "fewest_distinct": 36,
            "fewest_eligible": 748,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.002226215467089787,
            "controls_median": 0.0021957585704512894,
            "controls_median_end": 0.0022906774538569152,
            "controls_median_prompt": 0.00249435257865116,
            "controls_min": 0.002175567511585541,
            "controls_read": 39,
            "edit": 0.0023328641545958817,
            "edit_end": 0.0023019198124529794,
            "edit_prompt": 0.002599397615995258,
            "read_at_end": true
          }
        },
        "directed_mean": 0.001885145902633667,
        "half_0": {
          "directed_mean": 0.021150588989257812,
          "mean": -0.021150588989257812,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": -0.01738029718399048,
          "mean": 0.01738029718399048,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.21093919541419348,
          "floor": 0.059327125549316406,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 12.64177209085502
        },
        "mean": -0.001885145902633667,
        "n_frames": 8
      },
      "small": {
        "ci95": [
          -0.059670764552924756,
          0.03440060054001582
        ],
        "controls": {
          "other_sites_matched": {
            "as_read_at_least": 10,
            "at_least": 10,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.275,
            "p_fw": 0.55,
            "passes_alpha": false,
            "tops_every_control": false
          },
          "same_site": {
            "as_read_at_least": 23,
            "at_least": 23,
            "count": 39,
            "effective_count": 39,
            "non_finite": 0,
            "p_emp": 0.6,
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
            "controls_max": 0.002564726455602795,
            "controls_median": 0.0020804620726266876,
            "controls_median_end": 0.0018635017549968325,
            "controls_median_prompt": 0.0019173656910425052,
            "controls_min": 0.0016491347050759941,
            "controls_read": 39,
            "disjoint_sets": 35,
            "draw": "independent",
            "edit": 0.0022795830736868083,
            "edit_end": 0.0023312990815611556,
            "edit_prompt": 0.0026011753361672163,
            "fewest_distinct": 36,
            "fewest_eligible": 748,
            "of_controls": 39,
            "read_at_end": true
          },
          "same_site": {
            "controls_max": 0.0022822227037977427,
            "controls_median": 0.002203506388468668,
            "controls_median_end": 0.0022799261641921476,
            "controls_median_prompt": 0.002491638093488291,
            "controls_min": 0.0021703827660530806,
            "controls_read": 39,
            "edit": 0.0022795830736868083,
            "edit_end": 0.0023312990815611556,
            "edit_prompt": 0.0026011753361672163,
            "read_at_end": true
          }
        },
        "directed_mean": 0.012635082006454468,
        "half_0": {
          "directed_mean": 0.0037364959716796875,
          "mean": -0.0037364959716796875,
          "n_frames": 4
        },
        "half_1": {
          "directed_mean": 0.021533668041229248,
          "mean": -0.021533668041229248,
          "n_frames": 4
        },
        "house_bar": {
          "bar": 0.3135638249236034,
          "floor": 0.13109636306762695,
          "floor_too_loud": false,
          "least_effect": 0.75,
          "ratio_m_over_f": 5.720982508211211
        },
        "mean": -0.012635082006454468,
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
        "mean_kl": 2.8246645022989014e-08,
        "own_drop": 0.00765275458496717,
        "own_drop_bar": 0.1,
        "passed": true
      },
      "near_misses": {
        "abs_bar": 0.25,
        "abs_effect": 0.02995896339416504,
        "available": true,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 4
        },
        "mean_effect": 0.02995896339416504,
        "passed": true,
        "ratio_bar": 0.35,
        "ratio_to_target": 15.892119200063236
      },
      "neutral_text": {
        "available": true,
        "bar": 0.05,
        "control_kl_max": 4.705491460299527e-05,
        "gating": true,
        "items": {
          "given": 4,
          "read": 4,
          "under_item_gate": 3
        },
        "mean_kl": 3.8006565318937646e-05,
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
        "directed_effect": 7.369401693344116,
        "index": 0,
        "near_miss": "within the registered bar as a share of its own effect",
        "near_miss_abs": 0.9159023761749268,
        "near_miss_share": 0.1242844961215983
      },
      "matched_by": "prompt arm 0",
      "prompt_arms": [
        {
          "directed_effect": 7.369401693344116,
          "index": 0,
          "mean_effect": -7.369401693344116,
          "near_miss_abs": 0.9159023761749268,
          "prompt_sha256": "255cdbb968d4"
        }
      ],
      "prompt_arms_state": "run"
    },
    "bars_sha256": "170f517707593622d7a84e3a4eb8d1f5b2250524231a43611258d829b0744d12",
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
export const RECORDED_BOOK_HEAD = "98992e66303f513f";
