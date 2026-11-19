"""Phase 3 Claim Gate (`backend/level_claim.py`): registered bar calibration, post-freeze fresh draws on kept-back
frames, three control families, gating retention batteries, cheaper-knob baselines, pure statistical verdict engine,
and bounded claim sheets.

Follows the arbiter's rulings (`experiments/results/weight_space_exploration/verify/arbiter.md` §4 rows 4, 17-25, §8):
- Fixed fresh-reading budget `m` per brief (`brief.max_fresh_readings`), read once per `(brief_id, edit_id)`.
- Registration entry hashed into the level book BEFORE `os.urandom(32)` draws fresh kept-back items.
- Kept-back frames split into `half = 0` and `half = 1`; frame-level confidence intervals; both halves must replicate.
- House bar rule: `bar = sqrt(F * M)` with `F` the item-max null floor and `M` the brief's `least_effect`, dropped
  (`floor_too_loud`) whenever `M < 4 * F`.
- Control families (`other_sites_matched`, `same_site`, plus `sham_target` when site came from a declared rule) gated
  by family-wise level `p_fw = min(1, m * (1 + at_least) / (C + 1)) <= alpha` in BOTH batch classes (`small`, `large`).
  "Tops every control" is reported descriptively and never gates alone.
- Gating retention batteries (`neutral_text`, `near_misses`, `labelled_panel`, optional `habitat`) evaluated against
  a hashed `BarSet` from a calibration reading.
- Verdicts (no kept-out words): `"lead"`, `"lead, matched by a cheaper knob"`, `"no better than control edits"`,
  `"works but harms"`, `"did not replicate"`, `"kept the base"`, `"void"`.

Layers count from 0 in code and files; human-facing strings give both ("layer 20 (page layer 21)").
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend import level_book as lb
from backend import level_briefs as lbr
from backend import levelling as lv
from backend import weight_edits as we
from backend.causal_check import logits_fp32
from backend.theodolite_math import FINAL_LOGIT_SOFTCAP

RULE_VERSION = 1
RULE_PRESET = "levelling-claim-v1"
DEFAULT_ALPHA = 0.05
MIN_FRAMES_PER_HALF = 2
HALF_BAR_FRACTION = 0.5
KNOBS_MATCH_FRACTION = 0.95

# whether a reading compared the brief's author prompt arms with the edit (`cheaper_knobs["prompt_arms_state"]`);
# any other value is the reason they were not run, in plain words
PROMPT_ARMS_RUN = "run"
PROMPT_ARMS_NONE = "none in the brief"
PROMPT_ARMS_OFF = "switched off for this reading"

# How a fresh reading is taken when its caller says nothing (`run_fresh_claim`'s own defaults; a session shows them
# on its harness card, `scripts/check_level_claim.py` checks the two agree).
DESIGN_DEFAULTS = {"n_target": 8, "n_neutral": 4, "n_near": 4, "n_panel": 4, "written_rows": 12}

# A reading whose name begins with one of these is the harness's own (a calibration's, a claim's, a trial of record's,
# a settings audit's, a pricing's); a session's practice gates refuse such a name, so in a book every other reading is practice.
RESERVED_NAMES = ("cal_", "claim_", "trial_", "settings_audit_", "sizing_")
_LOOP_COUNT_KEY = {"instrument": "looped", "published": "looped_published", "either": "looped_either"}


def _name_key(name: Any) -> str:
    """A brief's name as its budget goes by it: spaces at its ends and case set aside, so "Pronoun_pull " is not a
    second brief with a budget of its own."""
    return str(name).strip().casefold() if name is not None else ""


def _written_rows(written_rows: int, n_doses: int) -> Tuple[int, int]:
    """(the rows of one written batch of a claim, how many control edits fit in it): the base row, the untouched last
    row, the null rows, a row a dose and at least one control edit. Fewer rows asked than that are raised to it."""
    need = 2 + len(lv.NULL_E_WRITTEN) + int(n_doses) + 1
    rows = min(max(int(written_rows), lv.WRITTEN_ROWS_MIN, need), lv.WRITTEN_ROWS_MAX)
    return rows, max(1, rows - 2 - len(lv.NULL_E_WRITTEN) - int(n_doses))

VERDICTS = (
    "lead",
    "lead, matched by a cheaper knob",
    "no better than control edits",
    "works but harms",
    "did not replicate",
    "kept the base",
    "void",
)

# Two-sided 95% Student-t critical values for df = 1..30 (standard table, zero external deps).
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


class ClaimRefused(ValueError):
    """A claim gate policy, registration, leakage, or budget refusal in plain words."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError, OverflowError):
        return None


# --------------------------------------------------------------------------------------------- registered bars


@dataclass(frozen=True)
class BarSet:
    """Hashed, registered thresholds for a brief's claim gate and retention batteries."""

    brief_id: str
    gating_batteries: Tuple[str, ...] = ("neutral_text", "near_misses", "labelled_panel")
    neutral_kl_max: float = 0.05
    near_miss_abs_max: float = 0.25
    near_miss_ratio_max: float = 0.35
    panel_own_drop_max: float = 0.10
    panel_kl_max: float = 0.05
    habitat_kl_max: float = 0.25
    alpha: float = DEFAULT_ALPHA
    half_bar_fraction: float = HALF_BAR_FRACTION
    min_frames_per_half: int = MIN_FRAMES_PER_HALF
    knobs_match_fraction: float = KNOBS_MATCH_FRACTION
    calibration_book_seqs: Tuple[int, ...] = ()
    calibrated_at: Optional[str] = None
    rule_preset: str = RULE_PRESET
    rule_version: int = RULE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_preset": self.rule_preset,
            "rule_version": int(self.rule_version),
            "brief_id": self.brief_id,
            "gating_batteries": list(self.gating_batteries),
            "neutral_kl_max": float(self.neutral_kl_max),
            "near_miss_abs_max": float(self.near_miss_abs_max),
            "near_miss_ratio_max": float(self.near_miss_ratio_max),
            "panel_own_drop_max": float(self.panel_own_drop_max),
            "panel_kl_max": float(self.panel_kl_max),
            "habitat_kl_max": float(self.habitat_kl_max),
            "alpha": float(self.alpha),
            "half_bar_fraction": float(self.half_bar_fraction),
            "min_frames_per_half": int(self.min_frames_per_half),
            "knobs_match_fraction": float(self.knobs_match_fraction),
            "calibration_book_seqs": [int(s) for s in self.calibration_book_seqs],
            "calibrated_at": self.calibrated_at,
        }

    @property
    def id(self) -> str:
        return hashlib.sha256(lb.canonical(self.to_dict())).hexdigest()

    def short(self) -> str:
        return self.id[:12]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BarSet":
        return cls(
            brief_id=str(raw["brief_id"]),
            gating_batteries=tuple(str(x) for x in raw.get("gating_batteries", ("neutral_text", "near_misses", "labelled_panel"))),
            neutral_kl_max=float(raw["neutral_kl_max"]),
            near_miss_abs_max=float(raw["near_miss_abs_max"]),
            near_miss_ratio_max=float(raw["near_miss_ratio_max"]),
            panel_own_drop_max=float(raw["panel_own_drop_max"]),
            panel_kl_max=float(raw["panel_kl_max"]),
            habitat_kl_max=float(raw.get("habitat_kl_max", 0.25)),
            alpha=float(raw.get("alpha", DEFAULT_ALPHA)),
            half_bar_fraction=float(raw.get("half_bar_fraction", HALF_BAR_FRACTION)),
            min_frames_per_half=int(raw.get("min_frames_per_half", MIN_FRAMES_PER_HALF)),
            knobs_match_fraction=float(raw.get("knobs_match_fraction", KNOBS_MATCH_FRACTION)),
            calibration_book_seqs=tuple(int(s) for s in (raw.get("calibration_book_seqs") or ())),
            calibrated_at=raw.get("calibrated_at"),
            rule_preset=str(raw.get("rule_preset", RULE_PRESET)),
            rule_version=int(raw.get("rule_version", RULE_VERSION)),
        )


def make_bar_set(brief_id: str, **kwargs: Any) -> BarSet:
    """Construct a hashed `BarSet` directly, for the pure verdict engine (`judge_claim`) and its checks.
    `run_fresh_claim` does not take such a set as it stands: a claim's bars are the ones a calibration reading wrote
    into the claim's own book (`calibrate_bars`, `calibration_note`)."""
    return BarSet(brief_id=brief_id, calibrated_at=kwargs.pop("calibrated_at", _utc_now()), **kwargs)


def save_bars(bars: BarSet, path: Path) -> str:
    """Save `bars` as canonical JSON with its embedded `bars_sha256` and return `bars.id`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**bars.to_dict(), "bars_sha256": bars.id}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return bars.id


def load_bars(path: Path) -> BarSet:
    """Load a saved `BarSet` JSON and verify its embedded `bars_sha256`."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClaimRefused(f"Cannot read bar set {p}: {exc}") from None
    if not isinstance(raw, dict):
        raise ClaimRefused(f"Bar set file {p} is not a JSON object.")
    wanted = raw.get("bars_sha256")
    if not isinstance(wanted, str) or not wanted:
        raise ClaimRefused(f"Bar set {p} carries no bars_sha256: save_bars writes one with every set, and a file "
                           "without it cannot be told from one changed by hand.")
    try:
        bars = BarSet.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ClaimRefused(f"Bar set {p} is not a whole bar set: {exc!r}.") from None
    if wanted != bars.id:
        raise ClaimRefused(f"Bar set {p} claims bars_sha256 {wanted[:12]} but hashes to {bars.short()}.")
    return bars


def calibration_note(bars: BarSet, book_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The `brief_calibration` note that recorded `bars` in the book (`{"seq", "at", ...its body}`), or None: the
    newest note of the bars' own brief whose `bars_sha256` is theirs. A claim takes no other bars (arbiter.md Row 25:
    "its bars come from a calibration reading, hashed"), and a session offers its claim tool with no other."""
    found = None
    for c in brief_history(bars.brief_id, book_path)["calibrations"]:
        if c.get("bars_sha256") == bars.id:
            found = c
    return found


# --------------------------------------------------------------------------------------------- statistics


def house_bar(floor: Optional[float], least_effect: float) -> Dict[str, Any]:
    """Arbiter Row 4 house bar rule: `sqrt(F * M)` where `F` is the null-row floor and `M` is `least_effect`,
    dropped (`floor_too_loud`) whenever `M < 4 * F`."""
    M = float(least_effect)
    if floor is None or not math.isfinite(float(floor)) or float(floor) < 0.0:
        return {"floor": None, "least_effect": M, "bar": M, "floor_too_loud": False, "ratio_m_over_f": None}
    F = float(floor)
    if F == 0.0:
        return {"floor": 0.0, "least_effect": M, "bar": M, "floor_too_loud": False, "ratio_m_over_f": None}
    too_loud = M < 4.0 * F
    bar = None if too_loud else math.sqrt(F * M)
    return {"floor": F, "least_effect": M, "bar": bar, "floor_too_loud": too_loud, "ratio_m_over_f": M / F}


def _sign_flip_p(values: Sequence[float], *, seed: int = 0) -> Dict[str, Optional[float]]:
    """Exact (n <= 12) or Monte-Carlo (10,000 draws) sign-flip permutation p-values over `values` (already directed)."""
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    n = int(arr.size)
    if n < 2:
        return {"two_sided": None, "one_sided": None, "n": n, "exact": False}
    obs = float(arr.mean())
    if n <= 12:
        # Enumerate all 2^n sign vectors exactly
        total = 1 << n
        masks = ((np.arange(total, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1) * 2 - 1
        means = (masks @ arr) / float(n)
        ge_two = int(np.sum(np.abs(means) >= abs(obs) - 1e-12))
        ge_one = int(np.sum(means >= obs - 1e-12))
        return {"two_sided": ge_two / float(total), "one_sided": ge_one / float(total), "n": n, "exact": True}
    rng = np.random.default_rng(seed)
    B = 10000
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, n))
    means = (signs @ arr) / float(n)
    ge_two = 1 + int(np.sum(np.abs(means) >= abs(obs) - 1e-12))
    ge_one = 1 + int(np.sum(means >= obs - 1e-12))
    return {"two_sided": ge_two / float(B + 1), "one_sided": ge_one / float(B + 1), "n": n, "exact": False}


def frame_statistics(
    item_effects: Sequence[Mapping[str, Any]],
    *,
    direction: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Aggregate per-item effects by `frame_id` so confidence intervals and half-splits are at the frame level
    (arbiter.md Row 19)."""
    by_frame: Dict[str, Tuple[int, List[float]]] = {}
    for row in item_effects:
        val = _clean_float(row.get("value"))
        if val is None:
            continue
        fid = str(row["frame_id"])
        half = int(row["half"])
        if fid not in by_frame:
            by_frame[fid] = (half, [val])
        else:
            by_frame[fid][1].append(val)

    frame_rows: List[Tuple[str, int, float, float]] = []
    sgn = 1.0 if direction == 0 else float(direction)
    for fid in sorted(by_frame):
        half, vals = by_frame[fid]
        m_raw = float(np.mean(vals))
        frame_rows.append((fid, half, m_raw, sgn * m_raw))

    n_f = len(frame_rows)
    if n_f == 0:
        empty_h = {"n_frames": 0, "mean": None, "directed_mean": None}
        return {
            "n_items": len(item_effects), "n_frames": 0, "mean": None, "directed_mean": None,
            "se": None, "ci95": [None, None], "directed_ci95": [None, None],
            "sign_flip": {"two_sided": None, "one_sided": None, "n": 0, "exact": False},
            "half_0": dict(empty_h), "half_1": dict(empty_h),
        }

    raw_vals = np.array([r[2] for r in frame_rows], dtype=np.float64)
    dir_vals = np.array([r[3] for r in frame_rows], dtype=np.float64)
    mean_raw = float(raw_vals.mean())
    mean_dir = float(dir_vals.mean())
    if n_f >= 2:
        se = float(raw_vals.std(ddof=1) / math.sqrt(n_f))
        tcrit = _T_CRIT_95.get(n_f - 1, 1.96)
        ci_raw = [mean_raw - tcrit * se, mean_raw + tcrit * se]
        ci_dir = [mean_dir - tcrit * se, mean_dir + tcrit * se]
    else:
        se = None
        ci_raw = [None, None]
        ci_dir = [None, None]

    def _half_block(h_idx: int) -> Dict[str, Any]:
        sub = [r for r in frame_rows if r[1] == h_idx]
        if not sub:
            return {"n_frames": 0, "mean": None, "directed_mean": None}
        return {
            "n_frames": len(sub),
            "mean": float(np.mean([r[2] for r in sub])),
            "directed_mean": float(np.mean([r[3] for r in sub])),
        }

    return {
        "n_items": len(item_effects),
        "n_frames": n_f,
        "mean": mean_raw,
        "directed_mean": mean_dir,
        "se": se,
        "ci95": ci_raw,
        "directed_ci95": ci_dir,
        "sign_flip": _sign_flip_p(dir_vals.tolist(), seed=seed),
        "half_0": _half_block(0),
        "half_1": _half_block(1),
    }


def control_standing(
    edit_value: Optional[float],
    control_values: Sequence[Optional[float]],
    *,
    max_fresh_readings: int,
    alpha: float,
) -> Dict[str, Any]:
    """Empirical rank and family-wise bound `p_fw = min(1, m * (1 + at_least) / (C + 1))` against one control family.
    Uses magnitude `|x|` (conservative two-sided comparison against exchangeability). A control edit whose reading is
    missing or not finite counts in `at_least`: it was run, and nothing shows the edit moved more than it did, so it
    may not sit in the denominator as one the edit topped (`non_finite` says how many there were)."""
    C = len(control_values)
    ev = _clean_float(edit_value)
    valid = [abs(v) for v in (_clean_float(c) for c in control_values) if v is not None]
    non_fin = C - len(valid)
    if ev is None or C == 0:
        return {
            "count": C, "non_finite": non_fin, "at_least": None, "p_emp": None, "p_fw": None,
            "tops_every_control": False, "passes_alpha": False,
        }
    ae = abs(ev)
    at_least = sum(1 for v in valid if v >= ae) + non_fin
    p_emp = (1 + at_least) / float(C + 1)
    p_fw = min(1.0, float(max_fresh_readings) * p_emp)
    return {
        "count": C,
        "non_finite": non_fin,
        "at_least": at_least,
        "p_emp": p_emp,
        "p_fw": p_fw,
        "tops_every_control": at_least == 0,
        "passes_alpha": bool(p_fw <= float(alpha) + 1e-12),
    }


def _min_detectable_gain(n: int, p0: float, *, alpha: float = DEFAULT_ALPHA) -> Optional[float]:
    """Least absolute rate shift detectable at two-sided `alpha` (80% power normal approx) for stratum count `n`."""
    if n < 2:
        return None
    z_a = 1.96 if alpha <= 0.055 else 1.645
    z_b = 0.8416
    p_base = min(max(float(p0), 0.05), 0.95)
    mdg = (z_a + z_b) * math.sqrt(2.0 * p_base * (1.0 - p_base) / float(n))
    return min(1.0, round(mdg, 4))


def written_strata_summary(
    written_result: Mapping[str, Any],
    *,
    main_dose: float = 1.0,
    alpha: float = DEFAULT_ALPHA,
    rule: str = "instrument",
) -> Dict[str, Any]:
    """Arbiter Row 21 strata summary for long written readings.

    Checks whether >= 90% of base generations finished under `max_new`, reports per-stratum counts
    (`ended`, `cut_off`, `looped`), marks any stratum with base count < 20 as `"too few to read"`, and prints
    the least gain each stratum could have seen. `rule` is the loop rule the reading counts by
    (`levelling.loop_flag`): the `looped` stratum is counted by it, so the strata and the verdict speak of the same
    loops.
    """
    loop_key = _LOOP_COUNT_KEY[rule]
    sm = written_result.get("summary") or {}
    base = sm.get("base") or {}
    n_tot = int(base.get("n") or 0)
    base_ended = int(base.get("ended") or 0)
    finish_rate = (base_ended / float(n_tot)) if n_tot > 0 else 0.0
    arm_key = f"edit:{repr(float(main_dose))}"
    edit_arm = (sm.get("arms") or {}).get(arm_key) or {}
    strata: Dict[str, Any] = {}
    for sname in ("ended", "cut_off", "looped"):
        key = loop_key if sname == "looped" and loop_key in base else sname  # an older row has the first rule only
        b_cnt = int(base.get(key) or 0)
        e_cnt = int(edit_arm.get(key) or 0)
        rate_b = (b_cnt / float(n_tot)) if n_tot > 0 else None
        rate_e = (e_cnt / float(n_tot)) if n_tot > 0 else None
        strata[sname] = {
            "base_count": b_cnt,
            "edit_count": e_cnt,
            "n_total": n_tot,
            "base_rate": rate_b,
            "edit_rate": rate_e,
            "delta_rate": (rate_e - rate_b) if (rate_b is not None and rate_e is not None) else None,
            "readable": b_cnt >= 20,
            "note": "too few to read" if b_cnt < 20 else "ok",
            "least_gain_detectable": _min_detectable_gain(n_tot, rate_b if rate_b is not None else 0.5, alpha=alpha),
        }
    def _counts(block: Mapping[str, Any]) -> Dict[str, Any]:
        return {"instrument": block.get("looped"), "published": block.get("looped_published"),
                "either": block.get("looped_either"), "rules_differ": block.get("loop_rules_differ")}

    return {
        "total_generations": n_tot,
        "loop_rule": rule,
        # how many generations each loop rule calls looped, base and edit (levelling.detect_loop, published_loop)
        "loop_rules": {"base": _counts(base), "edit": _counts(edit_arm),
                       "names": dict((sm.get("loop_rules") or {}))},
        "base_finish_rate": finish_rate,
        "cap_adequate_90pct": finish_rate >= 0.90 if n_tot > 0 else False,
        "cap_note": (
            "ok (>= 90% of base answers finished under cap)"
            if (n_tot > 0 and finish_rate >= 0.90)
            else f"cap warning: only {base_ended}/{n_tot} ({finish_rate:.1%}) base answers finished under cap"
        ),
        "strata": strata,
        "controls_range": sm.get("controls_range"),
    }


def price_written_reading(
    pilot_result: Mapping[str, Any],
    *,
    n_items: int,
    n_seeds: int,
    n_controls: int,
    doses_count: int = 2,
    rows_per_batch: int = 12,
    null_rows: int = 2,
) -> Dict[str, Any]:
    """Price a long written claim reading upfront from a timed written pilot (arbiter.md §8 & Row 12)."""
    its = pilot_result.get("items") or []
    step_secs = [
        _clean_float(it.get("seconds_a_step"))
        for it in its
        if _clean_float(it.get("seconds_a_step")) is not None
    ]
    sec_per_step = float(np.mean(step_secs)) if step_secs else None
    sm = pilot_result.get("summary") or {}
    mean_pieces = _clean_float((sm.get("base") or {}).get("mean_pieces")) or float(
        (pilot_result.get("plan") or {}).get("max_new") or 256
    )
    pilot_peak_gib = _clean_float((pilot_result.get("runtime") or {}).get("peak_gib"))
    usable = max(1, int(rows_per_batch) - 2 - int(null_rows) - int(doses_count))
    chunks_per_family = max(1, math.ceil(int(n_controls) / float(usable)))
    # Two families (other_sites_matched + same_site)
    total_batches = int(n_items) * int(n_seeds) * chunks_per_family * 2
    est_steps = total_batches * mean_pieces
    est_seconds = (est_steps * sec_per_step) if sec_per_step is not None else None
    return {
        "n_items": int(n_items),
        "n_seeds": int(n_seeds),
        "n_controls": int(n_controls),
        "rows_per_batch": int(rows_per_batch),
        "controls_per_chunk": usable,
        "chunks_per_family": chunks_per_family,
        "total_written_batches": total_batches,
        "pilot_seconds_per_step": sec_per_step,
        "pilot_mean_pieces": mean_pieces,
        "pilot_peak_gib": pilot_peak_gib,
        "estimated_gpu_seconds": round(est_seconds, 1) if est_seconds is not None else None,
        "estimated_minutes": round(est_seconds / 60.0, 2) if est_seconds is not None else None,
    }


# -------------------------------------------------------------------------------------- pure verdict decision engine


def judge_claim(
    *,
    brief: lbr.Brief,
    bars: BarSet,
    classes: Mapping[str, Mapping[str, Any]],
    batteries: Mapping[str, Mapping[str, Any]],
    cheaper_knobs: Optional[Mapping[str, Any]] = None,
    selection_mode: str = "free_pick",
    practice_count: int = 0,
    practice_best: Optional[float] = None,
    void_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Pure, deterministic Phase 3 Claim Gate verdict engine.

    `classes` maps `"small"` and `"large"` -> `{
        "item_effects": [{"frame_id", "half", "value"}, ...],
        "floor": float | None,
        "reversed_mean": float | None,
        "controls": {"other_sites_matched": [floats], "same_site": [floats], "sham_target"?: [floats]},
    }`.
    `batteries` maps battery name -> measured summary dict:
      - `"neutral_text"`: `{"mean_kl": float, "control_kl_max"?: float}`
      - `"near_misses"`: `{"mean_effect": float, "abs_effect": float, "control_abs_max"?: float}`
      - `"labelled_panel"`: `{"own_drop": float, "mean_kl": float}`
      - `"habitat"`: `{"available": bool, "mean_kl"?: float}`
    """
    if selection_mode not in ("free_pick", "declared_rule"):
        raise ClaimRefused(f'selection_mode must be "free_pick" or "declared_rule"; got {selection_mode!r}.')
    if bars.brief_id != brief.id:
        raise ClaimRefused(f"BarSet is for brief {bars.brief_id[:12]}, not {brief.short()}.")

    reasons: List[str] = []
    if void_reason:
        return {
            "verdict": "void",
            "reasons": [void_reason],
            "classes": {},
            "batteries": {},
            "cheaper_knobs": dict(cheaper_knobs or {}),
            "selection_mode": selection_mode,
            "practice_count": int(practice_count),
            "practice_best": _clean_float(practice_best),
        }

    # Check required classes: both small and large for paired metrics (arbiter.md Row 2 & Row 25),
    # or "written" for long sampled written briefs (Row 12 & Row 21).
    req_classes: Tuple[str, ...] = ("written",) if brief.metric == "written_loop" else ("small", "large")
    for cls_name in req_classes:
        if cls_name not in classes or not isinstance(classes[cls_name], Mapping):
            return {
                "verdict": "void",
                "reasons": [
                    f'Missing class "{cls_name}": a claim reading for metric "{brief.metric}" requires '
                    f"{' and '.join(req_classes)}."
                ],
                "classes": {},
                "batteries": {},
                "cheaper_knobs": dict(cheaper_knobs or {}),
                "selection_mode": selection_mode,
                "practice_count": int(practice_count),
                "practice_best": _clean_float(practice_best),
            }

    req_families = ["other_sites_matched", "same_site"]
    if selection_mode == "declared_rule":
        req_families.append("sham_target")

    cls_eval: Dict[str, Any] = {}
    any_void_frames = False
    any_floor_too_loud = False
    any_below_bar = False
    any_half_fail = False
    any_rev_fail = False
    any_ctl_fail = False

    sgn = 1.0 if brief.direction == 0 else float(brief.direction)

    for cls_name in req_classes:
        cdata = classes[cls_name]
        fstats = frame_statistics(cdata.get("item_effects") or [], direction=brief.direction)
        hb = house_bar(cdata.get("floor"), brief.least_effect)
        h0_n = fstats["half_0"]["n_frames"]
        h1_n = fstats["half_1"]["n_frames"]
        if h0_n < bars.min_frames_per_half or h1_n < bars.min_frames_per_half:
            any_void_frames = True
            reasons.append(
                f"{cls_name} class has too few kept-back frames per half ({h0_n}, {h1_n}; need >= {bars.min_frames_per_half})."
            )

        dir_mean = fstats["directed_mean"]
        bar = hb["bar"]
        if hb["floor_too_loud"]:
            any_floor_too_loud = True
            reasons.append(
                f"{cls_name} class floor too loud (F={hb['floor']:.5g}, M={hb['least_effect']:.5g}, M < 4F)."
            )
            clears_bar = False
        elif dir_mean is None or bar is None or dir_mean < bar:
            any_below_bar = True
            reasons.append(
                f"{cls_name} class directed mean ({dir_mean}) is below house bar ({bar})."
            )
            clears_bar = False
        else:
            clears_bar = True

        # Frame-half replication check (arbiter.md Row 19 & Row 23)
        h0_dir = fstats["half_0"]["directed_mean"]
        h1_dir = fstats["half_1"]["directed_mean"]
        half_req = (bars.half_bar_fraction * bar) if bar is not None else None
        halves_ok = bool(
            half_req is not None
            and h0_dir is not None and h1_dir is not None
            and h0_dir > 0.0 and h1_dir > 0.0
            and h0_dir >= half_req and h1_dir >= half_req
        )
        if clears_bar and not halves_ok:
            any_half_fail = True
            reasons.append(
                f"{cls_name} class did not replicate across frame halves (half 0: {h0_dir}, half 1: {h1_dir}, "
                f"need both >= {half_req})."
            )

        # Reversed-sign check (for signed metrics)
        rev_raw = _clean_float(cdata.get("reversed_mean"))
        rev_dir = (sgn * rev_raw) if (rev_raw is not None and brief.direction != 0) else None
        rev_ok = True
        if brief.direction != 0 and rev_dir is not None and bar is not None and rev_dir >= bar:
            rev_ok = False
            any_rev_fail = True
            reasons.append(
                f"{cls_name} class reversed dose moved in the same target direction ({rev_dir:.4f} >= bar {bar:.4f})."
            )

        # Control families (arbiter.md Row 17, Row 18, Row 23)
        ctls_in = cdata.get("controls") or {}
        ctl_eval: Dict[str, Any] = {}
        for fam in ("other_sites_matched", "same_site", "sham_target"):
            if fam in ctls_in and ctls_in[fam] is not None:
                cs = control_standing(
                    fstats["mean"], ctls_in[fam],
                    max_fresh_readings=brief.max_fresh_readings, alpha=bars.alpha,
                )
                ctl_eval[fam] = cs
                if fam in req_families and not cs["passes_alpha"]:
                    any_ctl_fail = True
                    lost = (
                        f"; {cs['non_finite']} of them gave no finite reading, or (a sham edit) did not move the "
                        "stream as the edit did, and are counted as moving as much"
                        if cs["non_finite"] else ""
                    )
                    reasons.append(
                        f'{cls_name} class failed "{fam}" controls (at_least={cs["at_least"]}/{cs["count"]}, '
                        f'p_fw={cs["p_fw"]}, alpha={bars.alpha}{lost}).'
                    )
            elif fam in req_families:
                ctl_eval[fam] = {
                    "count": 0, "non_finite": 0, "at_least": None, "p_emp": None, "p_fw": None,
                    "tops_every_control": False, "passes_alpha": False, "missing": True,
                }
                any_ctl_fail = True
                reasons.append(f'{cls_name} class missing required control family "{fam}".')
            else:
                ctl_eval[fam] = {
                    "count": 0, "p_emp": None, "p_fw": None, "passes_alpha": None,
                    "note": f"selection not matched, {int(practice_count)} read in practice",
                }

        cls_eval[cls_name] = {
            "frames": fstats,
            "house_bar": hb,
            "clears_bar": clears_bar,
            "halves_ok": halves_ok,
            "reversed_mean": rev_raw,
            "reversed_ok": rev_ok,
            "controls": ctl_eval,
            "delivered": cdata.get("delivered"),  # printed, gates nothing
        }

    # Target magnitude for near-miss ratio check (conservative: min across required classes)
    target_abs_vals = [
        abs(cls_eval[c]["frames"]["mean"])
        for c in req_classes
        if cls_eval[c]["frames"]["mean"] is not None
    ]
    target_abs_min = min(target_abs_vals) if target_abs_vals else 0.0
    target_dir_min = min(
        [cls_eval[c]["frames"]["directed_mean"] for c in req_classes if cls_eval[c]["frames"]["directed_mean"] is not None],
        default=0.0,
    )

    # Gating batteries evaluation (arbiter.md Row 20)
    bat_eval: Dict[str, Any] = {}
    any_harm = False
    for bname in ("neutral_text", "near_misses", "labelled_panel", "habitat"):
        gating = bname in bars.gating_batteries
        bdata = batteries.get(bname)
        if bdata is None:
            bat_eval[bname] = {"available": False, "gating": gating, "passed": not gating}
            if gating:
                any_harm = True
                reasons.append(f'Gating battery "{bname}" was not run.')
            continue
        if bname == "neutral_text":
            mkl = _clean_float(bdata.get("mean_kl"))
            ok = mkl is not None and mkl <= bars.neutral_kl_max
            bat_eval[bname] = {"available": True, "gating": gating, "mean_kl": mkl,
                               "bar": bars.neutral_kl_max, "passed": ok,
                               "control_kl_max": _clean_float(bdata.get("control_kl_max"))}
            if gating and not ok:
                any_harm = True
                reasons.append(f"neutral_text mean KL ({mkl}) exceeded bar ({bars.neutral_kl_max}).")
        elif bname == "near_misses":
            n_abs = _clean_float(bdata.get("abs_effect"))
            if n_abs is None and _clean_float(bdata.get("mean_effect")) is not None:
                n_abs = abs(float(bdata["mean_effect"]))
            ratio = (n_abs / target_abs_min) if (n_abs is not None and target_abs_min > 0.0) else None
            check_ratio = not (any_floor_too_loud or any_below_bar)
            ok = (
                n_abs is not None
                and n_abs <= bars.near_miss_abs_max
                and (not check_ratio or ratio is None or ratio <= bars.near_miss_ratio_max)
            )
            bat_eval[bname] = {"available": True, "gating": gating,
                               "mean_effect": _clean_float(bdata.get("mean_effect")),
                               "abs_effect": n_abs, "ratio_to_target": ratio,
                               "abs_bar": bars.near_miss_abs_max, "ratio_bar": bars.near_miss_ratio_max,
                               "passed": ok}
            if gating and not ok:
                any_harm = True
                reasons.append(
                    f"near_misses abs effect ({n_abs}, ratio {ratio}) exceeded bar "
                    f"(abs <= {bars.near_miss_abs_max}, ratio <= {bars.near_miss_ratio_max})."
                )
        elif bname == "labelled_panel":
            drop = _clean_float(bdata.get("own_drop"))
            mkl = _clean_float(bdata.get("mean_kl"))
            ok_drop = drop is not None and drop <= bars.panel_own_drop_max
            ok_kl = True if mkl is None else (mkl <= bars.panel_kl_max)
            ok = ok_drop and ok_kl
            bat_eval[bname] = {"available": True, "gating": gating, "own_drop": drop, "mean_kl": mkl,
                               "own_drop_bar": bars.panel_own_drop_max, "kl_bar": bars.panel_kl_max, "passed": ok}
            if gating and not ok:
                any_harm = True
                reasons.append(
                    f"labelled_panel harm (own_drop={drop} vs {bars.panel_own_drop_max}, "
                    f"kl={mkl} vs {bars.panel_kl_max})."
                )
        elif bname == "habitat":
            avail = bool(bdata.get("available", True))
            mkl = _clean_float(bdata.get("mean_kl")) if avail else None
            ok = (not avail and not gating) or (mkl is not None and mkl <= bars.habitat_kl_max)
            bat_eval[bname] = {"available": avail, "gating": gating, "mean_kl": mkl,
                               "bar": bars.habitat_kl_max, "passed": ok}
            if gating and not ok:
                any_harm = True
                reasons.append(f"habitat mean KL ({mkl}) exceeded bar ({bars.habitat_kl_max}).")

    # Cheaper-knob evaluation
    knobs_out = dict(cheaper_knobs or {})
    matched_knob_name: Optional[str] = None
    edit_nm_abs = (bat_eval.get("near_misses") or {}).get("abs_effect")
    for p_arm in knobs_out.get("prompt_arms") or []:
        eff = _clean_float(p_arm.get("directed_effect"))
        nm = _clean_float(p_arm.get("near_miss_abs"))
        nm_ok = (nm is None) or (nm <= max(float(edit_nm_abs or 0.0), bars.near_miss_abs_max))
        if eff is not None and target_dir_min > 0.0 and eff >= bars.knobs_match_fraction * target_dir_min and nm_ok:
            matched_knob_name = f"prompt arm {p_arm.get('index', 0)}"
            break
    if matched_knob_name is None:
        dec = knobs_out.get("decoding_bias")
        if isinstance(dec, Mapping) and dec.get("gating") and dec.get("matched"):
            matched_knob_name = "decoding bias"
    knobs_out["matched_by"] = matched_knob_name

    # Strict verdict precedence (arbiter.md §4 & §8)
    if any_void_frames:
        verdict = "void"
    elif any_floor_too_loud or any_below_bar:
        verdict = "kept the base"
    elif any_half_fail or any_rev_fail:
        verdict = "did not replicate"
    elif any_ctl_fail:
        verdict = "no better than control edits"
    elif any_harm:
        verdict = "works but harms"
    elif matched_knob_name is not None:
        verdict = "lead, matched by a cheaper knob"
        reasons.append(f"Matched by cheaper knob: {matched_knob_name} (no search over cheaper knobs).")
    else:
        verdict = "lead"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "classes": cls_eval,
        "batteries": bat_eval,
        "cheaper_knobs": knobs_out,
        "selection_mode": selection_mode,
        "practice_count": int(practice_count),
        "practice_best": _clean_float(practice_best),
    }


# --------------------------------------------------------------------------------------------- ledger history


def record_practice_draw(
    brief_id: str,
    drawn: Sequence[lbr.DrawnItem],
    *,
    book_path: Optional[Path] = None,
    brief_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Record what was shown in practice so a later fresh draw refuses any of it: the items' ids, the frames' ids, and
    the frames' template ids (`Frame.template_id`), which name a frame whatever a later version of the brief calls
    it. `brief_name` files the draw under the brief's name as well as its hash."""
    body: Dict[str, Any] = {
        "note_type": "practice_draw",
        "brief_id": str(brief_id),
        "item_ids": sorted({d.item.id for d in drawn}),
        "frame_ids": sorted({d.frame_id for d in drawn}),
        "template_ids": sorted({d.template_id for d in drawn if getattr(d, "template_id", "")}),
        "count": len(drawn),
    }
    if brief_name is not None:
        body["brief_name"] = str(brief_name)
    return lb.append("note", body, path=book_path)


def brief_history(brief_id: str, book_path: Optional[Path] = None, *, name: Optional[str] = None,
                  guard: Optional[lbr.KeptBack] = None) -> Dict[str, Any]:
    """Scan the level book for what bears on a brief.

    By the brief's hash (`brief_id`): its `calibrations` (a bar set is one version's).

    By the brief's NAME when `name` is given, else by its hash: the fresh readings registered (`registrations`,
    `verdicts`, `seen_edit_ids`) and the practice draws (`practice_item_ids`, `practice_frame_ids`). A brief's
    budget of fresh readings is its name's: changing a word of the brief does not start a new count
    (`run_fresh_claim` asks by name and refuses a brief whose hash is not the one its name was first read under,
    `other_ids`).

    Of the whole book, whatever the brief: `practice_templates`, the template ids of every frame drawn in practice
    here (so a practice frame moved into another brief's kept-back split is still known); `book_fresh_readings`,
    the count of fresh readings registered under each brief name (so a sheet can list them all); `frames_read`,
    for each kept-back template a registration recorded, the `[name, hash]` of every brief it was read under;
    `practice_readings`, how many readings in the book are not the harness's own (`RESERVED_NAMES`); and, when a
    `guard` is given, `kept_back_practised`: the lines of practice readings holding an item the guard finds
    (`level_briefs.KeptBack.hit`), whichever door they came in by (a session, the command line, the page).

    A brief's name is compared with spaces at its ends and case set aside (`_name_key`). Notes written before
    19 November 2026 carry no template ids and no name on a practice draw: those are found by the brief's hash
    alone."""
    calibrations: List[Dict[str, Any]] = []
    registrations: List[Dict[str, Any]] = []
    verdicts: List[Dict[str, Any]] = []
    seen_edit_ids: List[str] = []
    practice_item_ids: set = set()
    practice_frame_ids: set = set()
    practice_templates: set = set()
    book_fresh: Dict[str, int] = {}
    frames_read: Dict[str, set] = {}
    practice_readings = 0
    kept_back_practised: List[int] = []
    empty = {
        "calibrations": [], "registrations": [], "verdicts": [], "seen_edit_ids": [], "other_ids": [],
        "practice_item_ids": [], "practice_frame_ids": [], "practice_templates": [], "book_fresh_readings": {},
        "frames_read": {}, "practice_readings": 0, "kept_back_practised": [],
    }
    want_name = _name_key(name) if name is not None else None
    p = book_path if book_path is not None else lb.book_path()
    if not Path(p).exists():
        return empty

    def mine(body: Mapping[str, Any]) -> bool:
        return body.get("brief_id") == brief_id or (
            want_name is not None and body.get("brief_name") is not None and _name_key(body["brief_name"]) == want_name)

    for entry in lb.entries(p):
        body = entry.get("body")
        if not isinstance(body, Mapping):
            continue
        if entry.get("kind") != "note":
            if entry.get("kind") in ("paired", "written") and not str(body.get("name") or "").startswith(RESERVED_NAMES):
                practice_readings += 1
                if guard is not None:
                    for it in body.get("items") or ():
                        blk = (it.get("item") or {}) if isinstance(it, Mapping) else {}
                        if guard.hit(blk.get("system"), blk.get("question"), blk.get("answer")) is not None:
                            kept_back_practised.append(int(entry["seq"]))
                            break
            continue
        nt = body.get("note_type")
        if nt == "brief_calibration":
            if body.get("brief_id") == brief_id:
                calibrations.append({"seq": entry["seq"], "at": entry["at"], **dict(body)})
        elif nt == "practice_draw":
            for x in body.get("template_ids") or ():
                practice_templates.add(str(x))
            if mine(body):
                for x in body.get("item_ids") or ():
                    practice_item_ids.add(str(x))
                for x in body.get("frame_ids") or ():
                    practice_frame_ids.add(str(x))
        elif nt == "claim_registration":
            if isinstance(body.get("brief_name"), str):
                book_fresh[body["brief_name"]] = book_fresh.get(body["brief_name"], 0) + 1
            for tid in ((body.get("kept_back") or {}).get("template_ids") or ()):
                frames_read.setdefault(str(tid), set()).add((str(body.get("brief_name")), str(body.get("brief_id"))))
            if mine(body):
                registrations.append({"seq": entry["seq"], "at": entry["at"], **dict(body)})
                eid = body.get("edit_id")
                if isinstance(eid, str):
                    seen_edit_ids.append(eid)
        elif nt == "claim_verdict":
            if mine(body):
                verdicts.append({"seq": entry["seq"], "at": entry["at"], **dict(body)})
    return {
        "calibrations": calibrations,
        "registrations": registrations,
        "verdicts": verdicts,
        "seen_edit_ids": seen_edit_ids,
        "other_ids": sorted({str(r.get("brief_id")) for r in registrations if r.get("brief_id") != brief_id}),
        "practice_item_ids": sorted(practice_item_ids),
        "practice_frame_ids": sorted(practice_frame_ids),
        "practice_templates": sorted(practice_templates),
        "book_fresh_readings": dict(sorted(book_fresh.items())),
        "frames_read": {tid: sorted([n, i] for n, i in who) for tid, who in sorted(frames_read.items())},
        "practice_readings": practice_readings,
        "kept_back_practised": kept_back_practised,
    }


# ----------------------------------------------------------------------------------- calibration & live execution


def _extract_item_effects(
    result: Mapping[str, Any],
    drawn: Sequence[lbr.DrawnItem],
    cls_name: str,
    metric: str,
    dose: float,
) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[float], List[Optional[float]]]:
    """Extract per-item target effects, class floor, reversed-dose mean, and per-control means from a `lv.read` result.
    A control edit's mean is None unless it has a finite reading on every item read in this class: a mean over the
    items where it happened to give a number is not that control's reading (`control_standing` counts a None against
    the edit)."""
    by_name = {d.item.name: d for d in drawn}
    arm_key = f"edit:{repr(float(dose))}"
    rev_key = f"edit:{repr(float(-dose))}"
    item_effects: List[Dict[str, Any]] = []
    floors: List[float] = []
    rev_vals: List[float] = []
    C = int((result.get("controls") or {}).get("count") or 0)
    ctl_per_item: List[List[float]] = [[] for _ in range(C)]
    n_read = 0

    for it_ent in result.get("items") or []:
        if it_ent.get("status") != "ok":
            continue
        nm = (it_ent.get("item") or {}).get("name")
        d = by_name.get(nm)
        if d is None:
            continue
        cb = (it_ent.get("classes") or {}).get(cls_name)
        if not cb:
            continue
        n_read += 1
        fl = _clean_float(((cb.get("floor") or {}).get(metric) or {}).get("mean_abs"))
        if fl is not None:
            floors.append(fl)
        arms_by_k = {a["key"]: a for a in (cb.get("arms") or [])}
        a_main = arms_by_k.get(arm_key)
        val = _clean_float(((a_main or {}).get("metrics") or {}).get(metric, {}).get("mean"))
        if val is not None:
            item_effects.append({"frame_id": d.frame_id, "half": d.half, "item_id": d.item.id, "value": val})
        a_rev = arms_by_k.get(rev_key)
        rv = _clean_float(((a_rev or {}).get("metrics") or {}).get(metric, {}).get("mean"))
        if rv is not None:
            rev_vals.append(rv)
        for c in range(C):
            ac = arms_by_k.get(f"control:{c}")
            cv = _clean_float(((ac or {}).get("metrics") or {}).get(metric, {}).get("mean"))
            if cv is not None:
                ctl_per_item[c].append(cv)

    floor_max = max(floors) if floors else None
    rev_mean = float(np.mean(rev_vals)) if rev_vals else None
    ctl_means = [float(np.mean(xs)) if (xs and len(xs) == n_read) else None for xs in ctl_per_item]
    return item_effects, floor_max, rev_mean, ctl_means


def _delivered_of(
    result: Mapping[str, Any],
    drawn: Sequence[lbr.DrawnItem],
    cls_name: str,
    dose: float,
) -> Optional[Dict[str, Any]]:
    """How much the edit and this reading's control edits each changed the stream, over the items read in `cls_name`:
    the mean over items of an arm's `delivered` (`levelling._delivered_hook`: the relative change of the stream at
    the highest layer this reading edits, against the base row). `{"edit", "controls_median", "controls_min",
    "controls_max", "controls_read"}`, or None when the reading recorded none. It gates nothing: it is printed so
    that control edits which hardly touch the stream (a wide matching tolerance, sham edits on quiet sites) show."""
    by_name = {d.item.name: d for d in drawn}
    arm_key = f"edit:{repr(float(dose))}"
    C = int((result.get("controls") or {}).get("count") or 0)
    edit_vals: List[float] = []
    ctl_vals: List[List[float]] = [[] for _ in range(C)]
    for it_ent in result.get("items") or []:
        if it_ent.get("status") != "ok" or (it_ent.get("item") or {}).get("name") not in by_name:
            continue
        cb = (it_ent.get("classes") or {}).get(cls_name)
        if not cb:
            continue
        arms_by_k = {a["key"]: a for a in (cb.get("arms") or [])}
        ev = _clean_float((arms_by_k.get(arm_key) or {}).get("delivered"))
        if ev is not None:
            edit_vals.append(ev)
        for c in range(C):
            cv = _clean_float((arms_by_k.get(f"control:{c}") or {}).get("delivered"))
            if cv is not None:
                ctl_vals[c].append(cv)
    means = [float(np.mean(xs)) for xs in ctl_vals if xs]
    if not edit_vals and not means:
        return None
    return {
        "edit": float(np.mean(edit_vals)) if edit_vals else None,
        "controls_median": float(np.median(means)) if means else None,
        "controls_min": min(means) if means else None,
        "controls_max": max(means) if means else None,
        "controls_read": len(means),
        "per_control": [float(np.mean(xs)) if xs else None for xs in ctl_vals],
    }


def _edit_shape(edit: we.Edit) -> List[Tuple[Any, ...]]:
    """What a sham-target control edit must share with the claimed edit: for each term, the projection it changes,
    whether by columns, rows or a low-rank term, how many of them, and how (silenced, scaled by what, set, added
    to). Not the layer and not the sites: "the same rule on another target" lands elsewhere."""
    shape: List[Tuple[Any, ...]] = []
    for t in edit.terms:
        if t.kind == "lowrank":
            shape.append((t.which, t.kind, int(t.A.shape[0])))
        else:
            how = sorted((int(o), round(float(f), 6) if int(o) == 0 else 1.0) for o, f in zip(t.op, t.factor))
            shape.append((t.which, t.kind, int(t.idx.shape[0]), tuple(how)))
    return sorted(shape, key=repr)


def _check_shams(
    h: lv.Handle,
    brief: lbr.Brief,
    edit: we.Edit,
    sham_recipes: Sequence[Mapping[str, Any]],
    base_dir: Path,
) -> List[we.Edit]:
    """Build every sham-target recipe before anything is registered and refuse, in plain words, one that cannot be
    read as a control for `edit`: not buildable on this base, outside the brief's tensors or layers, the claimed
    edit itself, a repeat of another sham, or not the edit's shape (`_edit_shape`). The harness cannot check that
    the shams follow the caller's declared rule; it checks what it can, and the sheet says who supplied them."""
    if not isinstance(sham_recipes, Sequence) or isinstance(sham_recipes, (str, bytes)):
        raise ClaimRefused("sham_recipes is a list of edit recipes.")
    want = _edit_shape(edit)
    built: List[we.Edit] = []
    seen: Dict[str, int] = {}
    for i, rec in enumerate(sham_recipes):
        if not isinstance(rec, Mapping) or not rec:
            raise ClaimRefused(f"Sham control {i} is not an edit recipe.")
        try:
            got = we.from_recipe(dict(rec), Path(base_dir))
            se = (we.load(got, h.lm, model_id=h.model_id, weights_id=h.weights_id)
                  if isinstance(got, (str, Path))
                  else we.build(h.lm, got, model_id=h.model_id, weights_id=h.weights_id))
        except (we.Refused, lv.Refused, OSError, ValueError) as exc:
            raise ClaimRefused(f"Sham control {i}: {exc}") from None
        try:
            _check_edit_allowed_by_brief(se, brief)
        except ClaimRefused as exc:
            raise ClaimRefused(f"Sham control {i}: {exc}") from None
        if se.id == edit.id:
            raise ClaimRefused(f"Sham control {i} is the claimed edit itself.")
        if se.id in seen:
            raise ClaimRefused(f"Sham control {i} is the same edit as sham control {seen[se.id]}.")
        if _edit_shape(se) != want:
            raise ClaimRefused(
                f"Sham control {i} is not the claimed edit's shape: a sham-target control is the same rule on another "
                "target, so it changes as many columns or rows of the same projections, in the same way."
            )
        seen[se.id] = i
        built.append(se)
    return built


def _mean_metric_from_read(
    result: Mapping[str, Any],
    metric: str,
    dose: float,
    *,
    cls_name: str = "large",
) -> Tuple[Optional[float], Optional[float]]:
    """Return `(edit_mean, max_control_abs_mean)` for `metric` across kept items in `cls_name`."""
    sm = result.get("summary")
    if not isinstance(sm, Mapping):
        return None, None
    cb = (sm.get("classes") or {}).get(cls_name)
    if not isinstance(cb, Mapping):
        return None, None
    arms = cb.get("arms") or {}
    e_val = _clean_float(((arms.get(f"edit:{repr(float(dose))}") or {}).get(metric) or {}).get("mean_over_items"))
    ctl_abs = [
        abs(v)
        for k, a in arms.items()
        if k.startswith("control:")
        for v in [_clean_float((a.get(metric) or {}).get("mean_over_items"))]
        if v is not None
    ]
    return e_val, (max(ctl_abs) if ctl_abs else None)


def _prompt_arm_eval(
    h: lv.Handle,
    brief: lbr.Brief,
    target_drawn: Sequence[lbr.DrawnItem],
    near_drawn: Sequence[lbr.DrawnItem],
) -> List[Dict[str, Any]]:
    """Evaluate brief-author prompt arms on the unedited base model on the exact same fresh target and near-miss items."""
    if not brief.prompt_arms or brief.metric not in ("odds", "own"):
        return []
    we.assert_base(h.lm, h.hooks_at_load)
    sgn = 1.0 if brief.direction == 0 else float(brief.direction)

    def _base_score_items(drawn: Sequence[lbr.DrawnItem], system_override: Optional[str]) -> List[float]:
        scores: List[float] = []
        odds = brief.readouts.get("odds") if brief.metric == "odds" else None
        with torch.no_grad():
            for d in drawn:
                q, a = d.item.question, d.item.answer
                if q is None or a is None:
                    continue
                it = lv.make_item(h.tokenizer, q, a, system=system_override, name=d.item.name)
                sp = lv._Span.of(it, h.device)
                Hf = lv._whole(h, it.ids, 1)
                if odds:
                    Wt = h.W_U[torch.as_tensor(odds["toward"], dtype=torch.long, device=h.W_U.device)].to(h.device).float()
                    Wa = h.W_U[torch.as_tensor(odds["away"], dtype=torch.long, device=h.W_U.device)].to(h.device).float()
                    q_sl = slice(sp.T - 1, sp.T) if odds.get("at", "end") == "end" else slice(sp.lo, sp.T - 1)
                    x = Hf[0, q_sl].float()
                    zt = torch.tanh(x @ Wt.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
                    za = torch.tanh(x @ Wa.T / FINAL_LOGIT_SOFTCAP) * FINAL_LOGIT_SOFTCAP
                    val = float((torch.logsumexp(zt, -1) - torch.logsumexp(za, -1)).mean().item())
                else:
                    L0 = F.log_softmax(logits_fp32(Hf[0, sp.lo : sp.T - 1], h.W_U), dim=-1)
                    val = float(L0.gather(1, sp.y.unsqueeze(1)).mean().item())
                scores.append(val)
        return scores

    base_sys = (brief.chat_settings or {}).get("system")
    t_base = _base_score_items(target_drawn, base_sys)
    n_base = _base_score_items(near_drawn, base_sys) if near_drawn else []
    results: List[Dict[str, Any]] = []
    for idx, prompt_text in enumerate(brief.prompt_arms):
        sys_p = f"{base_sys}\n{prompt_text}" if base_sys else prompt_text
        t_arm = _base_score_items(target_drawn, sys_p)
        n_arm = _base_score_items(near_drawn, sys_p) if near_drawn else []
        t_diffs = [a - b for a, b in zip(t_arm, t_base)]
        n_diffs = [a - b for a, b in zip(n_arm, n_base)]
        mean_t = float(np.mean(t_diffs)) if t_diffs else None
        mean_n = float(np.mean(n_diffs)) if n_diffs else None
        results.append({
            "index": idx,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12],
            "mean_effect": mean_t,
            "directed_effect": (sgn * mean_t) if mean_t is not None else None,
            "near_miss_abs": abs(mean_n) if mean_n is not None else None,
        })
    we.assert_base(h.lm, h.hooks_at_load)
    return results


def calibrate_bars(
    h: lv.Handle,
    brief: lbr.Brief,
    *,
    layer: int,
    n_cols: int = 8,
    n_controls: int = 3,
    n_neutral: int = 4,
    n_near: int = 4,
    n_panel: int = 4,
    rows: Optional[Mapping[str, int]] = None,
    seed: int = 0,
    mult: float = 2.0,
    book_path: Optional[Path] = None,
) -> BarSet:
    """Run a calibration reading of control edits on the brief's retention batteries, derive a hashed `BarSet`,
    and record a `brief_calibration` note in the level book (arbiter.md Row 20 & Row 25)."""
    we.assert_base(h.lm, h.hooks_at_load)
    rows_cfg = dict(rows or {"small": lv.ROWS_SMALL, "large": lv.ROWS_LARGE})
    seed_bytes = hashlib.sha256(f"calibrate:{brief.id}:{seed}".encode("utf-8")).digest()

    neu_drawn = lbr.neutral_chat_generator().draw("kept_back", n_neutral, seed_bytes=seed_bytes + b":neu", tokenizer=h.tokenizer)
    nm_drawn = brief.near_miss_generator.draw("kept_back", n_near, seed_bytes=seed_bytes + b":nm", tokenizer=h.tokenizer)
    panel = lbr.draw_labelled_sanity_panel(n_panel, seed_bytes=seed_bytes + b":pan", tokenizer=h.tokenizer)

    # Probe edit of size `n_cols` at `layer` (calibration uses control edits' spread!)
    probe_recipe = {"terms": [{"layer": int(layer), "silence": list(range(int(n_cols)))}]}

    prev_bk = os.environ.get(lb.BOOK_ENV)
    if book_path is not None:
        os.environ[lb.BOOK_ENV] = str(book_path)
    try:
        r_neu = lv.read(h, {
            "name": f"cal_neutral_{brief.name}", "kind": "paired", "edit": probe_recipe,
            "doses": [1.0], "controls": {"family": "other_sites", "count": int(n_controls), "seed": int(seed)},
            "rows": rows_cfg, "readouts": {"kl": True, "own": True, "odds": None}, "folded": "never",
        }, {d.item.name: d.item for d in neu_drawn})
        r_nm = lv.read(h, {
            "name": f"cal_nearmiss_{brief.name}", "kind": "paired", "edit": probe_recipe,
            "doses": [1.0], "controls": {"family": "other_sites", "count": int(n_controls), "seed": int(seed) + 1},
            "rows": rows_cfg, "readouts": dict(brief.readouts), "folded": "never",
        }, {d.item.name: d.item for d in nm_drawn})
        r_pan = lv.read(h, {
            "name": f"cal_panel_{brief.name}", "kind": "paired", "edit": probe_recipe,
            "doses": [1.0], "controls": {"family": "other_sites", "count": int(n_controls), "seed": int(seed) + 2},
            "rows": rows_cfg, "readouts": {"kl": True, "own": True, "odds": None}, "folded": "never",
        }, {p["item"].name: p["item"] for p in panel})

        e_nkl, c_nkl = _mean_metric_from_read(r_neu, "kl", 1.0, cls_name="large")
        e_nm, c_nm = _mean_metric_from_read(r_nm, brief.metric if brief.metric != "written_loop" else "kl", 1.0, cls_name="large")
        e_pkl, c_pkl = _mean_metric_from_read(r_pan, "kl", 1.0, cls_name="large")
        e_pown, c_pown = _mean_metric_from_read(r_pan, "own", 1.0, cls_name="large")

        neu_bar = max(0.05, float(mult) * max(float(c_nkl or 0.0), float(e_nkl or 0.0)))
        nm_bar = max(0.25, float(mult) * max(float(c_nm or 0.0), abs(float(e_nm or 0.0))))
        pan_kl_bar = max(0.05, float(mult) * max(float(c_pkl or 0.0), float(e_pkl or 0.0)))
        pan_own_bar = max(0.10, float(mult) * max(float(c_pown or 0.0), abs(float(e_pown or 0.0))))

        now = _utc_now()
        cal_seqs = (int(r_neu["book"]["seq"]), int(r_nm["book"]["seq"]), int(r_pan["book"]["seq"]))
        bars = BarSet(
            brief_id=brief.id,
            neutral_kl_max=round(neu_bar, 6),
            near_miss_abs_max=round(nm_bar, 6),
            panel_kl_max=round(pan_kl_bar, 6),
            panel_own_drop_max=round(pan_own_bar, 6),
            calibration_book_seqs=cal_seqs,
            calibrated_at=now,
        )
        note_body = {
            "note_type": "brief_calibration",
            "brief_id": brief.id,
            "brief_name": brief.name,
            "bars": bars.to_dict(),
            "bars_sha256": bars.id,
            "book_seqs": list(cal_seqs),
            "model_id": h.model_id,  # bars are one model's: a claim on other weights is refused them
            "weights_id": h.weights_id,
            # what the calibration was asked for: a bar is `mult` times the largest figure its probe and control
            # edits gave (never under the registered floor), so `mult` and the probe belong beside the bars
            "settings": {
                "layer": int(layer), "n_cols": int(n_cols), "n_controls": int(n_controls),
                "n_neutral": int(n_neutral), "n_near": int(n_near), "n_panel": int(n_panel),
                "rows": {k: int(v) for k, v in rows_cfg.items()}, "seed": int(seed), "mult": float(mult),
            },
        }
        lb.append("note", note_body, path=book_path)
        return bars
    finally:
        if prev_bk is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = prev_bk


def _check_edit_allowed_by_brief(edit: we.Edit, brief: lbr.Brief) -> None:
    for t in edit.terms:
        if t.which not in brief.allowed_tensors:
            raise ClaimRefused(
                f'Brief "{brief.name}" permits edits on {", ".join(brief.allowed_tensors)} only; '
                f"edit {edit.short()} touches {t.linear}."
            )
        if brief.allowed_layers is not None and int(t.layer) not in brief.allowed_layers:
            raise ClaimRefused(
                f'Brief "{brief.name}" permits edits on layers {list(brief.allowed_layers)} only; '
                f"edit {edit.short()} touches layer {t.layer} (page layer {t.layer + 1})."
            )


def _extract_written_effects(
    result: Mapping[str, Any],
    drawn: Sequence[lbr.DrawnItem],
    dose: float,
    rule: str = "instrument",
) -> Tuple[List[Dict[str, Any]], Optional[float], List[Optional[float]]]:
    """Extract per-item loop-rate deltas (`edit_loop_rate - base_loop_rate`), reversed mean, and per-control deltas
    from an extended `lv.read(kind="written")` result. `rule` says which loop rule counts a generation as looped
    (`levelling.loop_flag`: the instrument's own, the published one, or either); it is the brief's
    `written_config["loop_rule"]`."""
    by_name = {d.item.name: d for d in drawn}
    arm_key = f"edit:{repr(float(dose))}"
    rev_key = f"edit:{repr(float(-dose))}"
    C = int(((result.get("plan") or {}).get("controls") or {}).get("count") or 0)
    item_effects: List[Dict[str, Any]] = []
    rev_vals: List[float] = []
    ctl_per_item: List[List[float]] = [[] for _ in range(C)]

    for it_ent in result.get("items") or []:
        nm = (it_ent.get("item") or {}).get("name")
        d = by_name.get(nm)
        if d is None:
            continue
        samples = it_ent.get("samples") or [{"base": it_ent.get("base"), "arms": it_ent.get("arms") or []}]
        if not samples:
            continue
        e_deltas: List[float] = []
        r_deltas: List[float] = []
        c_deltas: List[List[float]] = [[] for _ in range(C)]
        for s in samples:
            b_loop = 1.0 if lv.loop_flag(s.get("base") or {}, rule) else 0.0
            arms_by_k = {a["key"]: a for a in (s.get("arms") or [])}
            if arm_key in arms_by_k:
                a_loop = 1.0 if lv.loop_flag(arms_by_k[arm_key], rule) else 0.0
                e_deltas.append(a_loop - b_loop)
            if rev_key in arms_by_k:
                r_loop = 1.0 if lv.loop_flag(arms_by_k[rev_key], rule) else 0.0
                r_deltas.append(r_loop - b_loop)
            for c in range(C):
                ck = f"control:{c}"
                if ck in arms_by_k:
                    c_loop = 1.0 if lv.loop_flag(arms_by_k[ck], rule) else 0.0
                    c_deltas[c].append(c_loop - b_loop)
        if e_deltas:
            item_effects.append({
                "frame_id": d.frame_id,
                "half": d.half,
                "item_id": d.item.id,
                "value": float(np.mean(e_deltas)),
            })
        if r_deltas:
            rev_vals.append(float(np.mean(r_deltas)))
        for c in range(C):
            if c_deltas[c]:
                ctl_per_item[c].append(float(np.mean(c_deltas[c])))

    rev_mean = float(np.mean(rev_vals)) if rev_vals else None
    ctl_means = [float(np.mean(xs)) if xs else None for xs in ctl_per_item]
    return item_effects, rev_mean, ctl_means


def _run_written_target_claims(
    h: lv.Handle,
    brief: lbr.Brief,
    edit: we.Edit,
    target_drawn: Sequence[lbr.DrawnItem],
    *,
    base_dir: Path,
    main_dose: float,
    doses: Sequence[float],
    C: int,
    seed_u32: int,
    reading_index: int,
    selection_mode: str,
    sham_recipes: Optional[Sequence[Mapping[str, Any]]],
    match_tolerance: Optional[float],
    written_rows: int = 12,
) -> Tuple[Optional[str], Dict[str, Any], List[int], Optional[Dict[str, Any]], Optional[str]]:
    """Run chunked written claim readings for `other_sites_matched`, `same_site`, and optional `sham_target`."""
    wcfg = dict(brief.written_config or {})
    loop_rule = lbr.loop_rule_of(brief)
    max_new = int(wcfg.get("max_new_tokens", 128))
    seeds = list(wcfg.get("seeds") or [1, 2])
    n_samples = max(1, len(seeds))
    base_sampler_seed = int(seeds[0]) if seeds else 1
    sampler = {
        "temperature": float(wcfg.get("temperature", 1.0)),
        "top_k": int(wcfg.get("top_k", 64)),
        "top_p": float(wcfg.get("top_p", 0.95)),
        "presence_penalty": float(wcfg.get("presence_penalty", 0.0)),
        "seed": base_sampler_seed,
    }
    thinking = bool((brief.chat_settings or {}).get("thinking", False))
    target_items = {d.item.name: d.item for d in target_drawn}
    R, max_c_chunk = _written_rows(written_rows, len(doses))

    book_seqs: List[int] = []
    unpacker_mode: Optional[str] = None
    first_result: Optional[Dict[str, Any]] = None
    first_effects: List[Dict[str, Any]] = []
    first_rev: Optional[float] = None
    ctls_all: Dict[str, List[Optional[float]]] = {}

    def _run_family(fam: str, fam_count: int, seed_offset: int, shams: Optional[Sequence[Mapping[str, Any]]] = None) -> Optional[str]:
        nonlocal unpacker_mode, first_result, first_effects, first_rev
        collected: List[Optional[float]] = []
        done = 0
        chunk_idx = 0
        while done < fam_count:
            take = min(max_c_chunk, fam_count - done)
            if fam == "sham_target":
                assert shams is not None
                ctl_cfg: Dict[str, Any] = {"family": "sham_target", "shams": list(shams[done : done + take]),
                                           "seed": seed_u32 + seed_offset + chunk_idx * 97}
            elif fam == "other_sites_matched":
                ctl_cfg = {"family": "other_sites_matched", "count": take,
                           "seed": seed_u32 + seed_offset + chunk_idx * 97}
                if match_tolerance is not None:
                    ctl_cfg["tolerance"] = float(match_tolerance)
            else:
                ctl_cfg = {"family": fam, "count": take, "seed": seed_u32 + seed_offset + chunk_idx * 97}
            plan = {
                "name": f"claim_{brief.name}_{reading_index}_{fam}_c{chunk_idx}",
                "kind": "written",
                "edit": {"built": edit.id},
                "doses": list(doses),
                "main_dose": float(main_dose),
                "controls": ctl_cfg,
                "rows": R,
                "max_new": max_new,
                "thinking": thinking,
                "sampler": sampler,
                "samples": n_samples,
                "diagnose": True,
            }
            res = lv.read(h, plan, target_items, base_dir=Path(base_dir), edit=edit)
            book_seqs.append(int(res["book"]["seq"]))
            if unpacker_mode is None:
                unpacker_mode = (res.get("runtime") or {}).get("unpacker", {}).get("mode")
            if res.get("status") != "ok":
                return f'Written target reading ({fam} chunk {chunk_idx}) ended with status "{res.get("status")}": {res.get("why")}'
            effs, rev_m, ctl_m = _extract_written_effects(res, target_drawn, main_dose, loop_rule)
            if first_result is None:
                first_result = res
                first_effects = effs
                first_rev = rev_m
            collected.extend(ctl_m)
            done += take
            chunk_idx += 1
        ctls_all[fam] = collected
        return None

    err = _run_family("other_sites_matched", C, 0)
    if err is None:
        err = _run_family("same_site", C, 1000)
    if err is None and selection_mode == "declared_rule" and sham_recipes:
        err = _run_family("sham_target", len(sham_recipes), 2000, shams=sham_recipes)

    if err is not None:
        return err, {}, book_seqs, None, unpacker_mode

    classes_payload = {
        "written": {
            "item_effects": first_effects,
            "floor": None,
            "reversed_mean": first_rev,
            "controls": ctls_all,
        }
    }
    strata = (written_strata_summary(first_result or {}, main_dose=main_dose, rule=loop_rule)
              if first_result is not None else None)
    return None, classes_payload, book_seqs, strata, unpacker_mode


def compare_settings_audit(
    h: lv.Handle,
    brief: lbr.Brief,
    edit: we.Edit,
    sample_drawn: Sequence[lbr.DrawnItem],
    *,
    base_dir: Path = Path("."),
    max_new_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Arbiter overruled #6: when `brief.compare_settings` is True, measure the base model under BOTH
    `thinking=False` and `thinking=True` on a sample and report plainly when the fault is absent with thinking off."""
    wcfg = dict(brief.written_config or {})
    max_new = int(max_new_override if max_new_override is not None else min(int(wcfg.get("max_new_tokens", 128)), 256))
    sampler = {
        "temperature": float(wcfg.get("temperature", 1.0)),
        "top_k": int(wcfg.get("top_k", 64)),
        "top_p": float(wcfg.get("top_p", 0.95)),
        "presence_penalty": float(wcfg.get("presence_penalty", 0.0)),
        "seed": int((wcfg.get("seeds") or [1])[0]),
    }
    items = {d.item.name: d.item for d in sample_drawn[: min(4, len(sample_drawn))]}
    audit_key = _LOOP_COUNT_KEY[lbr.loop_rule_of(brief)]  # the audit counts loops by the brief's own rule
    out: Dict[str, Any] = {}
    book_seqs: List[int] = []
    for label, th in (("thinking_off", False), ("thinking_on", True)):
        res = lv.read(
            h,
            {
                "name": f"settings_audit_{brief.name}_{label}",
                "kind": "written",
                "edit": {"built": edit.id},
                "doses": [1.0],
                "main_dose": 1.0,
                "controls": {"family": "other_sites", "count": 1, "seed": 0},
                "rows": max(lv.WRITTEN_ROWS_MIN, 6),
                "max_new": max_new,
                "thinking": th,
                "sampler": sampler,
                "samples": 1,
                "diagnose": True,
            },
            items,
            base_dir=Path(base_dir),
            edit=edit,
        )
        book_seqs.append(int(res["book"]["seq"]))
        base_sm = ((res.get("summary") or {}).get("base") or {}) if res.get("status") == "ok" else {}
        n_b = int(base_sm.get("n") or 0)
        n_loop = int(base_sm.get(audit_key if audit_key in base_sm else "looped") or 0)
        out[label] = {
            "status": res.get("status"),
            "n": n_b,
            "looped": n_loop,
            "cut_off": int(base_sm.get("cut_off") or 0),
            "ended": int(base_sm.get("ended") or 0),
            "loop_rate": (n_loop / float(n_b)) if n_b > 0 else None,
        }
    off_loops = out["thinking_off"].get("looped") or 0
    on_loops = out["thinking_on"].get("looped") or 0
    note = (
        "fault absent with thinking off on audited base sample"
        if (off_loops == 0 and on_loops > 0)
        else (
            "no loops observed on audited base sample under either setting"
            if (off_loops == 0 and on_loops == 0)
            else f"base looped under thinking_off ({off_loops}) and thinking_on ({on_loops})"
        )
    )
    return {**out, "note": note, "loop_rule": lbr.loop_rule_of(brief), "book_seqs": book_seqs}


def estimate_claim(
    h: lv.Handle,
    brief: lbr.Brief,
    edit_or_recipe: Any,
    *,
    base_dir: Path = Path("."),
    main_dose: float = 1.0,
    selection_mode: str = "free_pick",
    sham_recipes: Optional[Sequence[Mapping[str, Any]]] = None,
    n_target: int = 8,
    n_neutral: int = 4,
    n_near: int = 4,
    n_panel: int = 4,
    rows: Optional[Mapping[str, int]] = None,
    written_rows: int = 12,
    controls_override: Optional[int] = None,
    match_tolerance: Optional[float] = None,
    run_prompt_arms: bool = True,
) -> Dict[str, Any]:
    """What one fresh claim reading would cost, worked with no forward and nothing registered or entered:
    `{"forwards", "seconds", "gib_above_model", "readings": [{"name", "forwards", "seconds"}, ...]}`.

    It prices the readings `run_fresh_claim` takes with the same arguments, each by `levelling.estimate`: the target
    reading against matched other sites, the one against same-site controls, the one against sham edits when the
    selection declares a rule (or the chunked written readings of a `written_loop` brief, and its settings audit),
    the three battery readings, and the one-row forwards of the brief's prompt arms. The fresh items are not drawn
    until the claim is registered, so their lengths are taken from a sizing draw of the same frames (a fixed seed of
    the brief's own; nothing of it is returned or kept): the price is close, not exact, in seconds. Its forwards
    are the forwards the readings take when each runs whole: a reading that ends void takes fewer, and so does one
    whose folded twin is priced and then cannot be taken (the stand-ins cannot fold an added column).
    `scripts/check_level_claim.py` checks the forwards against those a claim takes for silencing edits, with and
    without prompt arms, written, and with shams, so a change to what a claim reads that is not made here too is
    caught. Raises `ClaimRefused` for an edit or a sham that is not a recipe, and what `levelling.normalise`,
    `levelling.estimate`, `weight_edits.recipe_facts` and the generators raise for a plan or a draw they refuse."""
    if not isinstance(edit_or_recipe, (we.Edit, Mapping)) or (isinstance(edit_or_recipe, Mapping) and not edit_or_recipe):
        raise ClaimRefused('An edit is a recipe ({"terms": [...]} or {"saved": "<id or folder>"}) or a built edit.')
    if sham_recipes is not None and (not isinstance(sham_recipes, Sequence) or isinstance(sham_recipes, (str, bytes))
                                     or not all(isinstance(s, Mapping) and s for s in sham_recipes)):
        raise ClaimRefused("sham_recipes is a list of edit recipes.")
    facts = (we.edit_facts(edit_or_recipe) if isinstance(edit_or_recipe, we.Edit)
             else we.recipe_facts(dict(edit_or_recipe), Path(base_dir)))
    declared = selection_mode == "declared_rule" and bool(sham_recipes)
    C = int(controls_override if controls_override is not None else brief.default_controls)
    rows_cfg = dict(rows or {"small": lv.ROWS_SMALL, "large": lv.ROWS_LARGE})
    doses = [float(main_dose), float(-main_dose), 0.0] if brief.direction != 0 else [float(main_dose), 0.0]
    tol = float(lv.DEFAULT_MATCH_TOL if match_tolerance is None else match_tolerance)
    ed = {"built": "sizing"}  # a plan is priced from the edit's facts; its edit block only has to be one

    seed = hashlib.sha256(("sizing:" + brief.id).encode("utf-8")).digest()
    sys_turn = (brief.chat_settings or {}).get("system")
    target = [d.item for d in brief.generator.draw("kept_back", int(n_target), seed_bytes=seed + b":target",
                                                   tokenizer=h.tokenizer, system_override=sys_turn)]
    near = [d.item for d in brief.near_miss_generator.draw("kept_back", int(n_near), seed_bytes=seed + b":near",
                                                           tokenizer=h.tokenizer, system_override=sys_turn)]
    neutral = [d.item for d in lbr.neutral_chat_generator().draw("kept_back", int(n_neutral),
                                                                 seed_bytes=seed + b":neutral", tokenizer=h.tokenizer)]
    panel = [p["item"] for p in lbr.draw_labelled_sanity_panel(int(n_panel), seed_bytes=seed + b":panel",
                                                               tokenizer=h.tokenizer)]

    def pieces(items: Sequence[lv.Item], thinking: bool = False) -> List[Tuple[int, int]]:
        if thinking:
            return [(len(it.ids), len(lv.chat_ids(h.tokenizer, it.question, it.system, thinking=True))) for it in items]
        return [(len(it.ids), len(it.prompt_ids)) for it in items]

    readings: List[Dict[str, Any]] = []
    gib = 0.0

    def add(name: str, plan: Dict[str, Any], pcs: Sequence[Tuple[int, int]], f: Mapping[str, Any] = facts) -> None:
        nonlocal gib
        norm = lv.normalise({**plan, "name": f"sizing_{name}"}, n_layers=h.n_layers, tokenizer=h.tokenizer,
                            facts=dict(f), dtype=h.dtype)  # with the facts, so a layer the model lacks is refused here
        est = lv.estimate(norm, pcs, n_layers=h.n_layers, facts=dict(f))
        gib = max(gib, float(est["gib_above_model"]))
        readings.append({"name": name, "forwards": int(est["forwards"]), "seconds": float(est["seconds"])})

    sham_facts = dict(facts)
    if declared:
        layers = set(facts["layers"])
        for rec in sham_recipes or ():
            layers |= set(we.recipe_facts(dict(rec), Path(base_dir))["layers"])
        sham_facts = {**facts, "layers": sorted(layers), "lowest_layer": int(min(layers))}

    if brief.metric == "written_loop":
        wcfg = dict(brief.written_config or {})
        max_new = int(wcfg.get("max_new_tokens", 128))
        seeds = list(wcfg.get("seeds") or [1, 2])
        sampler = {"temperature": float(wcfg.get("temperature", 1.0)), "top_k": int(wcfg.get("top_k", 64)),
                   "top_p": float(wcfg.get("top_p", 0.95)), "presence_penalty": float(wcfg.get("presence_penalty", 0.0)),
                   "seed": int(seeds[0]) if seeds else 1}
        thinking = bool((brief.chat_settings or {}).get("thinking", False))
        R, per_chunk = _written_rows(written_rows, len(doses))
        families = [("other_sites_matched", C), ("same_site", C)] + ([("sham_target", len(sham_recipes or ()))] if declared else [])
        for fam, count in families:
            done, chunk = 0, 0
            while done < count:
                take = min(per_chunk, count - done)
                if fam == "sham_target":
                    ctl: Dict[str, Any] = {"family": fam, "shams": list((sham_recipes or ())[done:done + take]), "seed": 0}
                elif fam == "other_sites_matched":
                    ctl = {"family": fam, "count": take, "seed": 0, "tolerance": tol}
                else:
                    ctl = {"family": fam, "count": take, "seed": 0}
                add(f"{fam}_c{chunk}",
                    {"kind": "written", "edit": ed, "doses": list(doses), "main_dose": float(main_dose), "controls": ctl,
                     "rows": R, "max_new": max_new, "thinking": thinking, "sampler": sampler,
                     "samples": max(1, len(seeds)), "diagnose": True},
                    pieces(target, thinking), sham_facts if fam == "sham_target" else facts)
                done, chunk = done + take, chunk + 1
        if brief.compare_settings:
            for label, th in (("thinking_off", False), ("thinking_on", True)):
                add(f"settings_audit_{label}",
                    {"kind": "written", "edit": ed, "doses": [1.0], "main_dose": 1.0,
                     "controls": {"family": "other_sites", "count": 1, "seed": 0}, "rows": max(lv.WRITTEN_ROWS_MIN, 6),
                     "max_new": min(max_new, 256), "thinking": th, "sampler": {**sampler, "seed": int((wcfg.get("seeds") or [1])[0])},
                     "samples": 1, "diagnose": True},
                    pieces(target[:min(4, len(target))], th))
    else:
        base_plan = {"kind": "paired", "edit": ed, "doses": doses, "main_dose": float(main_dose), "rows": rows_cfg,
                     "readouts": dict(brief.readouts), "rank_by": brief.metric, "folded": "auto"}
        add("other_sites_matched", {**base_plan, "controls": {"family": "other_sites_matched", "count": C, "seed": 0,
                                                              "tolerance": tol}}, pieces(target))
        add("same_site", {**base_plan, "controls": {"family": "same_site", "count": C, "seed": 0}, "folded": "never"},
            pieces(target))
        if declared:
            add("sham_target", {**base_plan, "controls": {"family": "sham_target", "shams": list(sham_recipes or ()),
                                                          "seed": 0}, "folded": "never"}, pieces(target), sham_facts)

    bat = {"kind": "paired", "edit": ed, "doses": [float(main_dose), 0.0],
           "controls": {"family": "other_sites", "count": min(C, 3), "seed": 0}, "rows": rows_cfg, "folded": "never"}
    add("neutral", {**bat, "readouts": {"kl": True, "own": True, "odds": None}}, pieces(neutral))
    add("near_misses", {**bat, "readouts": dict(brief.readouts)}, pieces(near))
    add("labelled_panel", {**bat, "readouts": {"kl": True, "own": True, "odds": None}}, pieces(panel))

    if run_prompt_arms and brief.prompt_arms and brief.metric in ("odds", "own"):
        scored = [it for it in target + near if it.question is not None and it.answer is not None]
        k = 1 + len(brief.prompt_arms)  # each item once under the brief's own system turn, once under each arm
        readings.append({"name": "prompt_arms", "forwards": k * len(scored),
                         "seconds": float(k * sum(lv.whole(1, len(it.ids)) for it in scored))})

    return {
        "forwards": int(sum(r["forwards"] for r in readings)),
        "seconds": float(sum(r["seconds"] for r in readings)),
        "gib_above_model": gib,
        "readings": readings,
        "note": "the fresh items are not drawn yet: lengths are from a sizing draw of the same frames",
    }


def run_fresh_claim(
    h: lv.Handle,
    brief: lbr.Brief,
    bars: BarSet,
    edit_or_recipe: Any,
    *,
    base_dir: Path = Path("."),
    main_dose: float = 1.0,
    forecast: Optional[Mapping[str, Any]] = None,
    selection_mode: str = "free_pick",
    sham_recipes: Optional[Sequence[Mapping[str, Any]]] = None,
    declared_rule: Optional[str] = None,
    practice_item_ids: Optional[Iterable[str]] = None,
    practice_frame_ids: Optional[Iterable[str]] = None,
    practice_count: int = 0,
    practice_best: Optional[float] = None,
    n_target: int = 8,
    n_neutral: int = 4,
    n_near: int = 4,
    n_panel: int = 4,
    rows: Optional[Mapping[str, int]] = None,
    written_rows: int = 12,
    controls_override: Optional[int] = None,
    match_tolerance: Optional[float] = None,
    run_prompt_arms: bool = True,
    book_path: Optional[Path] = None,
    _seed_override: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Run one registered Phase 3 fresh claim reading end to end (paired or written_loop).

    Order of operations (strict anti-p-hacking & post-freeze draw discipline):
    1. Verify base, build/gate edit, check brief's allowed tensors/layers, check read-once, `m` budget, and frame leakage.
    2. Hash and append `claim_registration` note to the level book BEFORE drawing fresh items.
    3. Draw 32 bytes from `os.urandom(32)` and draw kept-back target items (balanced across halves 0 & 1),
       near misses, neutral chat items, and labelled sanity panel items; refuse on any practice collision.
    4. Run paired readings in both batch classes (`small`, `large`) — or chunked written readings for `written_loop` —
       with `other_sites_matched`, `same_site`, and optional `sham_target` controls, plus gating batteries,
       author prompt arms, and optional `compare_settings` audit.
    5. Apply `judge_claim`, render the bounded claim sentence and sheet, append `claim_verdict` to the book,
       verify `assert_base`, and return both full and driver-redacted records.
    """
    if bars.brief_id != brief.id:
        raise ClaimRefused(f"BarSet is for brief {bars.brief_id[:12]}, not {brief.short()}.")
    if selection_mode not in ("free_pick", "declared_rule"):
        raise ClaimRefused(f'selection_mode must be "free_pick" or "declared_rule"; got {selection_mode!r}.')
    if selection_mode == "declared_rule" and not sham_recipes:
        raise ClaimRefused('selection_mode="declared_rule" requires sham_recipes for the sham_target control family.')
    if selection_mode == "free_pick" and sham_recipes:
        raise ClaimRefused('Sham edits are read only for a site chosen by a declared rule (selection_mode='
                           '"declared_rule"): with a free pick they would be registered and printed, and never read.')
    if declared_rule is not None and (not isinstance(declared_rule, str) or not declared_rule.strip()):
        raise ClaimRefused("declared_rule is the rule the site was chosen by, in a sentence or two.")
    if not isinstance(edit_or_recipe, (we.Edit, Mapping)) or (isinstance(edit_or_recipe, Mapping) and not edit_or_recipe):
        raise ClaimRefused('An edit is a recipe ({"terms": [...]} or {"saved": "<id or folder>"}) or a built edit.')

    we.assert_base(h.lm, h.hooks_at_load)
    if isinstance(edit_or_recipe, we.Edit):
        lv._same_base(h, edit_or_recipe)
        edit = we.gate_again(h.lm, edit_or_recipe, model_id=h.model_id, weights_id=h.weights_id)
    else:
        got = we.from_recipe(edit_or_recipe, Path(base_dir))
        edit = (we.load(got, h.lm, model_id=h.model_id, weights_id=h.weights_id)
                if isinstance(got, (str, Path))
                else we.build(h.lm, got, model_id=h.model_id, weights_id=h.weights_id))
    _check_edit_allowed_by_brief(edit, brief)

    # Check read-once, fixed `m` budget, and ledger-recorded practice items/frames
    bk = Path(book_path) if book_path is not None else lb.book_path()
    hist = brief_history(brief.id, bk, name=brief.name, guard=lbr.KeptBack([brief]))
    if hist["kept_back_practised"]:
        at = ", ".join(str(s) for s in hist["kept_back_practised"][:8])
        raise ClaimRefused(
            f'Kept-back frames of brief "{brief.name}" were read in practice in this book (line'
            f'{"s" if len(hist["kept_back_practised"]) > 1 else ""} {at}'
            f'{", ..." if len(hist["kept_back_practised"]) > 8 else ""}): their wording, or a near wording, stands in a '
            "reading that is not a fresh claim reading, so a claim on them here would not be on fresh frames."
        )
    if hist["other_ids"]:
        raise ClaimRefused(
            f'Brief "{brief.name}" has changed since its first fresh reading in this book (it was '
            f'{hist["other_ids"][0][:12]} then, it is {brief.short()} now). A brief and its budget of fresh readings '
            "are fixed before the first one: read the brief as it was, or give the changed brief a name of its own "
            "(every sheet lists the fresh readings of every brief in its book)."
        )
    all_forbid_items = set(hist["practice_item_ids"]) | set(practice_item_ids or ())
    all_forbid_frames = set(hist["practice_frame_ids"]) | set(practice_frame_ids or ())
    kb_now = brief.generator.frames_for("kept_back")
    practised = set(hist["practice_templates"])
    leaked_frames = ({fr.id for fr in kb_now} & all_forbid_frames) | {fr.id for fr in kb_now if fr.template_id in practised}
    if leaked_frames:
        raise ClaimRefused(
            f"Frame leakage refused before registration: kept-back frame(s) {sorted(leaked_frames)} "
            "were already seen in practice in this book (by their id under this brief, or by their template under any)."
        )
    kb_tids = sorted(fr.template_id for fr in kb_now)
    elsewhere = sorted({(n, i) for tid in kb_tids for n, i in hist["frames_read"].get(tid, ()) if i != brief.id})
    if elsewhere:
        raise ClaimRefused(
            f'Kept-back frames of this brief were already read in this book under brief "{elsewhere[0][0]}" '
            f"({elsewhere[0][1][:12]}): a kept-back frame is read under one brief, with that brief's budget of fresh "
            "readings, whatever a later brief that holds it is called."
        )
    cal_note = next((c for c in reversed(hist["calibrations"]) if c.get("bars_sha256") == bars.id), None)
    if cal_note is None:
        raise ClaimRefused(
            f'Bar set {bars.short()} is not from a calibration reading recorded in this level book for brief '
            f'"{brief.name}" ({brief.short()}): a claim takes the bars calibrate_bars wrote into the same book '
            "(scripts/levelling.py calibrate), and no others."
        )
    if cal_note.get("weights_id") not in (None, h.weights_id):
        raise ClaimRefused(
            f"Bar set {bars.short()} was calibrated on other weights ({str(cal_note.get('weights_id'))[:16]}) than "
            f"this model's ({str(h.weights_id)[:16]}): calibrate on the model the claim is read on."
        )
    k_prior = len(hist["registrations"])
    if edit.id in hist["seen_edit_ids"]:
        raise ClaimRefused(
            f'Read once: edit {edit.short()} has already been read on brief "{brief.name}" ({brief.short()}).'
        )
    if k_prior >= brief.max_fresh_readings:
        raise ClaimRefused(
            f'Brief "{brief.name}" has already spent all {brief.max_fresh_readings} of its fresh readings.'
        )
    reading_index = k_prior + 1
    bk_resolved = Path(bk).expanduser().resolve()
    # which book this is: a fresh reading is counted in the book it is taken in and nowhere else, so one taken in a
    # side book (a check's, a dry run's) says so, in its registration, on its sheet and in its sentence
    book_note = {"name": bk_resolved.name, "default": bk_resolved == lb.default_book_path().resolve()}

    C = int(controls_override if controls_override is not None else brief.default_controls)
    rows_cfg = dict(rows or {"small": lv.ROWS_SMALL, "large": lv.ROWS_LARGE})
    doses = [float(main_dose), float(-main_dose), 0.0] if brief.direction != 0 else [float(main_dose), 0.0]

    # The reading's design: everything about how it is taken that the caller may set, checked here and written into
    # the registration before the draw, so no choice of tolerance, counts or shams is made (or can be denied) after
    # the fresh items are known. Through the agent's door none of it is the driver's (level_session.claim).
    tol = lv.DEFAULT_MATCH_TOL if match_tolerance is None else match_tolerance
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(float(tol)) or float(tol) < 1.0:
        raise ClaimRefused(f"The matched controls' tolerance is a finite factor of 1 or more; {match_tolerance!r} was given.")
    tol = float(tol)
    for what, n, least in (("n_target", n_target, 2 * int(bars.min_frames_per_half)), ("n_neutral", n_neutral, 1),
                           ("n_near", n_near, 1), ("n_panel", n_panel, 1), ("controls a family", C, 1)):
        if isinstance(n, bool) or not isinstance(n, int) or n < least:
            raise ClaimRefused(f"A fresh reading takes {what} as a whole number of {least} or more; {n!r} was given.")
    shams_built = _check_shams(h, brief, edit, sham_recipes, Path(base_dir)) if sham_recipes else []
    design = {
        "n_target": int(n_target), "n_neutral": int(n_neutral), "n_near": int(n_near), "n_panel": int(n_panel),
        "rows": {k: int(v) for k, v in rows_cfg.items()}, "written_rows": int(written_rows),
        "controls_per_family": C, "battery_controls": min(C, 3), "match_tolerance": tol,
        "doses": [float(d) for d in doses], "prompt_arms": bool(run_prompt_arms),
    }
    if brief.metric == "written_loop":
        design["loop_rule"] = lbr.loop_rule_of(brief)
    # The registration spends one of the brief's fresh readings and marks the edit read, so whatever is certain to
    # end the reading is asked first: a dose the edit cannot be read at (the reading's own test,
    # `weight_edits.dose_refusal`), an edit the matched controls cannot be drawn for, and every plan and draw the
    # reading will make (`estimate_claim` builds and prices them all, with no forward). What can still end a
    # registered reading void is what only the card or the draw can tell: too few matched sites, a fresh item that
    # was seen in practice, a forward that fails.
    for d in doses:
        words = we.dose_refusal(edit, d, h.dtype)
        if words:
            raise ClaimRefused(f"The edit cannot be read at dose {d:g}, so the reading is not registered: {words}")
    for i, se in enumerate(shams_built):
        words = we.dose_refusal(se, float(main_dose), h.dtype)
        if words:
            raise ClaimRefused(f"Sham control {i} cannot be read at dose {float(main_dose):g}: {words}")
    if not any(t.kind in ("cols", "rows") for t in edit.terms):
        raise ClaimRefused(
            "A fresh reading sets the edit beside other sites matched on what they deliver, and those are drawn for "
            "column and row terms: an edit of low-rank terms alone has no such controls, and cannot be claimed yet."
        )
    try:
        priced = estimate_claim(
            h, brief, edit, base_dir=Path(base_dir), main_dose=float(main_dose), selection_mode=selection_mode,
            sham_recipes=sham_recipes, n_target=int(n_target), n_neutral=int(n_neutral), n_near=int(n_near),
            n_panel=int(n_panel), rows=rows_cfg, written_rows=int(written_rows), controls_override=C,
            match_tolerance=tol, run_prompt_arms=bool(run_prompt_arms),
        )
    except (lv.Refused, we.Refused, lbr.BriefError) as exc:
        raise ClaimRefused(f"This reading cannot be taken as asked, so it is not registered: {exc}") from None
    practice_total = max(int(practice_count), int(hist["practice_readings"]))

    kb_frames = brief.generator.frames_for("kept_back")
    published = lbr.published_template_ids(h.tokenizer)
    kept_back_note = {"frames": len(kb_frames),
                      "published": sum(1 for fr in kb_frames if fr.template_id in published),
                      "template_ids": kb_tids}
    shams_note = None
    if shams_built:
        sham_ids = [s.id for s in shams_built]
        shams_note = {"count": len(sham_ids), "edit_ids": sham_ids,
                      "sha256": hashlib.sha256(lb.canonical(sham_ids)).hexdigest(),
                      "from": "supplied with the claim; checked for shape only"}

    # Step 2: Register BEFORE drawing fresh items!
    prev_bk = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(bk)
    try:
        reg_body = {
            "note_type": "claim_registration",
            "brief_id": brief.id,
            "brief_name": brief.name,
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "edit_provenance": we.edit_provenance(edit),
            "main_dose": float(main_dose),
            "reading_index": reading_index,
            "max_fresh_readings": int(brief.max_fresh_readings),
            "controls_per_family": C,
            "rule_preset": RULE_PRESET,
            "rule_version": RULE_VERSION,
            "bars_sha256": bars.id,
            "bars_calibration_seq": int(cal_note["seq"]),
            "bars_calibrations_in_book": len(hist["calibrations"]),
            "selection_mode": selection_mode,
            "declared_rule": " ".join(declared_rule.split())[:600] if declared_rule else None,
            # practice readings: those in this book that are not the harness's own, or more if the caller counts more
            "practice_count": practice_total,
            "practice_count_given": int(practice_count),
            "practice_readings_in_book": int(hist["practice_readings"]),
            "practice_best": _clean_float(practice_best),
            "forecast": dict(forecast or {}),
            "priced": {"forwards": int(priced["forwards"]), "seconds": round(float(priced["seconds"]), 2)},
            "design": design,
            "shams": shams_note,
            "kept_back": kept_back_note,
            "book": book_note,
            "book_fresh_readings": dict(hist["book_fresh_readings"]),
            "prior_verdicts": [
                {"seq": v["seq"], "edit_id": v.get("edit_id"), "verdict": v.get("verdict"),
                 "reading_index": v.get("reading_index"), "brief_id": v.get("brief_id")}
                for v in hist["verdicts"]
            ],
        }
        reg_entry = lb.append("note", reg_body, path=bk)

        # Step 3: Post-freeze randomness from os.urandom(32)
        seed_bytes = bytes(_seed_override) if _seed_override is not None else os.urandom(32)
        seed_sha = hashlib.sha256(seed_bytes).hexdigest()
        seed_u32 = int.from_bytes(seed_bytes[:4], "little") & 0x7FFFFFFF

        draw_refused: Optional[str] = None
        try:
            target_drawn = brief.generator.draw(
                "kept_back", n_target, seed_bytes=seed_bytes + b":target",
                tokenizer=h.tokenizer, system_override=(brief.chat_settings or {}).get("system"),
                forbid_item_ids=all_forbid_items, forbid_frame_ids=all_forbid_frames,
                name_prefix=f"fresh_{reading_index}",
            )
            near_drawn = brief.near_miss_generator.draw(
                "kept_back", n_near, seed_bytes=seed_bytes + b":near",
                tokenizer=h.tokenizer, system_override=(brief.chat_settings or {}).get("system"),
                name_prefix=f"near_{reading_index}",
            )
            neu_drawn = lbr.neutral_chat_generator().draw(
                "kept_back", n_neutral, seed_bytes=seed_bytes + b":neutral",
                tokenizer=h.tokenizer, name_prefix=f"neu_{reading_index}",
            )
            panel = lbr.draw_labelled_sanity_panel(n_panel, seed_bytes=seed_bytes + b":panel", tokenizer=h.tokenizer)
        except lbr.BriefError as exc:
            # the one draw refusal that cannot be asked before registering: a fresh item that was seen in practice.
            # The reading is void and says why; it is not left registered with no verdict.
            draw_refused = str(exc)
            target_drawn, near_drawn, neu_drawn, panel = [], [], [], []

        target_items = {d.item.name: d.item for d in target_drawn}
        near_items = {d.item.name: d.item for d in near_drawn}
        neu_items = {d.item.name: d.item for d in neu_drawn}
        panel_items = {p["item"].name: p["item"] for p in panel}

        # Step 4: Target readings with control families (paired or written_loop)
        book_seqs: List[int] = [int(reg_entry["seq"])]
        void_reason: Optional[str] = f"The fresh draw was refused: {draw_refused}" if draw_refused else None
        classes_payload: Dict[str, Dict[str, Any]] = {}
        written_strata: Optional[Dict[str, Any]] = None
        settings_audit: Optional[Dict[str, Any]] = None
        unpacker_mode: Optional[str] = None

        if void_reason is not None:
            pass
        elif brief.metric == "written_loop":
            void_reason, classes_payload, w_seqs, written_strata, unpacker_mode = _run_written_target_claims(
                h, brief, edit, target_drawn, base_dir=Path(base_dir), main_dose=main_dose, doses=doses,
                C=C, seed_u32=seed_u32, reading_index=reading_index, selection_mode=selection_mode,
                sham_recipes=sham_recipes, match_tolerance=tol, written_rows=written_rows,
            )
            book_seqs.extend(w_seqs)
            if void_reason is None and brief.compare_settings:
                settings_audit = compare_settings_audit(h, brief, edit, target_drawn, base_dir=Path(base_dir))
                book_seqs.extend(settings_audit.get("book_seqs") or [])
        else:
            base_target_plan = {
                "kind": "paired",
                "edit": {"built": edit.id},
                "doses": doses,
                "main_dose": float(main_dose),
                "rows": rows_cfg,
                "readouts": dict(brief.readouts),
                "rank_by": brief.metric,
                "folded": "auto",
            }
            matched_ctl: Dict[str, Any] = {"family": "other_sites_matched", "count": C, "seed": seed_u32,
                                           "tolerance": tol}
            r_matched = lv.read(
                h, {**base_target_plan, "name": f"claim_{brief.name}_{reading_index}_matched",
                    "controls": matched_ctl},
                target_items, base_dir=Path(base_dir), edit=edit,
            )
            book_seqs.append(int(r_matched["book"]["seq"]))
            unpacker_mode = (r_matched.get("runtime") or {}).get("unpacker", {}).get("mode")
            r_same = lv.read(
                h, {**base_target_plan, "name": f"claim_{brief.name}_{reading_index}_same_site",
                    "controls": {"family": "same_site", "count": C, "seed": seed_u32 + 1}, "folded": "never"},
                target_items, base_dir=Path(base_dir), edit=edit,
            )
            book_seqs.append(int(r_same["book"]["seq"]))

            r_sham = None
            if selection_mode == "declared_rule" and sham_recipes:
                r_sham = lv.read(
                    h, {**base_target_plan, "name": f"claim_{brief.name}_{reading_index}_sham",
                        "controls": {"family": "sham_target", "shams": list(sham_recipes), "seed": seed_u32 + 2},
                        "folded": "never"},
                    target_items, base_dir=Path(base_dir), edit=edit,
                )
                book_seqs.append(int(r_sham["book"]["seq"]))

            for tag, r in (("other_sites_matched", r_matched), ("same_site", r_same), ("sham_target", r_sham)):
                if r is not None and r.get("status") != "ok":
                    void_reason = f'Target reading ({tag}) ended with status "{r.get("status")}": {r.get("why")}'
                    break

            if void_reason is None:
                for cls_name in ("small", "large"):
                    it_effs, fl_m, rev_m, ctl_m = _extract_item_effects(r_matched, target_drawn, cls_name, brief.metric, main_dose)
                    _, fl_s, _, ctl_s = _extract_item_effects(r_same, target_drawn, cls_name, brief.metric, main_dose)
                    fl_both = max([x for x in (fl_m, fl_s) if x is not None], default=None)
                    ctls_cls: Dict[str, Any] = {"other_sites_matched": ctl_m, "same_site": ctl_s}
                    deliv_cls: Dict[str, Any] = {
                        "other_sites_matched": _delivered_of(r_matched, target_drawn, cls_name, main_dose),
                        "same_site": _delivered_of(r_same, target_drawn, cls_name, main_dose),
                    }
                    if r_sham is not None:
                        _, _, _, ctl_sh = _extract_item_effects(r_sham, target_drawn, cls_name, brief.metric, main_dose)
                        dv_sh = _delivered_of(r_sham, target_drawn, cls_name, main_dose)
                        if dv_sh is not None:
                            # A sham edit is the caller's own, checked only for shape before it is read. One that
                            # does not move the stream within the matching tolerance of what the edit moves it is no
                            # control for the edit, and is counted as moving as much: quiet shams cannot help a claim.
                            e_d, apart = dv_sh.get("edit"), 0
                            if e_d is not None and e_d > 0.0:
                                for c, d_c in enumerate(dv_sh.get("per_control") or []):
                                    if c < len(ctl_sh) and (d_c is None or not e_d / tol <= d_c <= e_d * tol):
                                        ctl_sh[c] = None
                                        apart += 1
                            dv_sh["not_comparable"] = apart
                        ctls_cls["sham_target"] = ctl_sh
                        deliv_cls["sham_target"] = dv_sh
                    for dv in deliv_cls.values():
                        if isinstance(dv, dict):
                            dv.pop("per_control", None)  # each control's own figure stays in its reading's entry
                    classes_payload[cls_name] = {
                        "item_effects": it_effs,
                        "floor": fl_both,
                        "reversed_mean": rev_m,
                        "controls": ctls_cls,
                        "delivered": deliv_cls,
                    }

        # Gating retention batteries (always teacher-forced paired in both batch classes)
        bat_payload: Dict[str, Dict[str, Any]] = {}
        bat_ctl_n = min(C, 3)
        nm_metric = brief.metric if brief.metric != "written_loop" else "kl"
        if void_reason is None:
            r_neu = lv.read(
                h, {"name": f"claim_{brief.name}_{reading_index}_neutral", "kind": "paired",
                    "edit": {"built": edit.id}, "doses": [float(main_dose), 0.0],
                    "controls": {"family": "other_sites", "count": bat_ctl_n, "seed": seed_u32 + 3},
                    "rows": rows_cfg, "readouts": {"kl": True, "own": True, "odds": None}, "folded": "never"},
                neu_items, base_dir=Path(base_dir), edit=edit,
            )
            book_seqs.append(int(r_neu["book"]["seq"]))
            r_nm = lv.read(
                h, {"name": f"claim_{brief.name}_{reading_index}_nearmiss", "kind": "paired",
                    "edit": {"built": edit.id}, "doses": [float(main_dose), 0.0],
                    "controls": {"family": "other_sites", "count": bat_ctl_n, "seed": seed_u32 + 4},
                    "rows": rows_cfg, "readouts": dict(brief.readouts), "folded": "never"},
                near_items, base_dir=Path(base_dir), edit=edit,
            )
            book_seqs.append(int(r_nm["book"]["seq"]))
            r_pan = lv.read(
                h, {"name": f"claim_{brief.name}_{reading_index}_panel", "kind": "paired",
                    "edit": {"built": edit.id}, "doses": [float(main_dose), 0.0],
                    "controls": {"family": "other_sites", "count": bat_ctl_n, "seed": seed_u32 + 5},
                    "rows": rows_cfg, "readouts": {"kl": True, "own": True, "odds": None}, "folded": "never"},
                panel_items, base_dir=Path(base_dir), edit=edit,
            )
            book_seqs.append(int(r_pan["book"]["seq"]))

            for tag, r in (("neutral_text", r_neu), ("near_misses", r_nm), ("labelled_panel", r_pan)):
                if r.get("status") != "ok":
                    void_reason = f'Battery reading "{tag}" ended with status "{r.get("status")}": {r.get("why")}'
                    break

            if void_reason is None:
                neu_kls = [v for c in ("small", "large") for v in [_mean_metric_from_read(r_neu, "kl", main_dose, cls_name=c)[0]] if v is not None]
                neu_ctls = [v for c in ("small", "large") for v in [_mean_metric_from_read(r_neu, "kl", main_dose, cls_name=c)[1]] if v is not None]
                bat_payload["neutral_text"] = {
                    "mean_kl": max(neu_kls) if neu_kls else None,
                    "control_kl_max": max(neu_ctls) if neu_ctls else None,
                }
                nm_vals = [v for c in ("small", "large") for v in [_mean_metric_from_read(r_nm, nm_metric, main_dose, cls_name=c)[0]] if v is not None]
                nm_abs = max((abs(v) for v in nm_vals), default=None)
                bat_payload["near_misses"] = {
                    "mean_effect": nm_vals[-1] if nm_vals else None,
                    "abs_effect": nm_abs,
                }
                p_owns = [v for c in ("small", "large") for v in [_mean_metric_from_read(r_pan, "own", main_dose, cls_name=c)[0]] if v is not None]
                p_kls = [v for c in ("small", "large") for v in [_mean_metric_from_read(r_pan, "kl", main_dose, cls_name=c)[0]] if v is not None]
                own_drop = max((max(0.0, -v) for v in p_owns), default=None)
                bat_payload["labelled_panel"] = {
                    "own_drop": own_drop,
                    "mean_kl": max(p_kls) if p_kls else None,
                }
                bat_payload["habitat"] = {"available": False, "note": "no atlas passages index mounted"}

        # Step 5: Cheaper-knob arms. `prompt_arms_state` says whether the brief's prompt arms were compared with the
        # edit at all, so the sentence and the sheet never report "none matched" of arms that were not run.
        prompt_arm_res = _prompt_arm_eval(h, brief, target_drawn, near_drawn) if (void_reason is None and run_prompt_arms) else []
        if not brief.prompt_arms:
            arms_state = PROMPT_ARMS_NONE
        elif not run_prompt_arms:
            arms_state = PROMPT_ARMS_OFF
        elif brief.metric not in ("odds", "own"):
            arms_state = f'no prompt read-out for metric "{brief.metric}"'
        elif void_reason is not None:
            arms_state = "the reading is void"
        else:
            arms_state = PROMPT_ARMS_RUN
        cheaper_knobs = {
            "prompt_arms": prompt_arm_res,
            "prompt_arms_state": arms_state,
            "decoding_bias": {"available": brief.metric == "odds", "gating": False,
                              "note": "logit offset on named pieces shifts end-position odds at zero extra forwards"},
            "activation_twin": {
                "is_exact_mul": all(t.kind == "cols" and t.which == "down_proj" and all(op == 0 for op in t.op)
                                    for t in edit.terms),
                "note": "no search over cheaper knobs; not compared with a trained adapter",
            },
        }

        judged = judge_claim(
            brief=brief, bars=bars, classes=classes_payload, batteries=bat_payload,
            cheaper_knobs=cheaper_knobs, selection_mode=selection_mode,
            practice_count=practice_total, practice_best=practice_best,
            void_reason=void_reason,
        )

        # Build bounded claim sentence & full claim sheet
        meta = {
            "date": _utc_now(),
            "model_id": h.model_id,
            "weights_id": h.weights_id,
            "unpacker": unpacker_mode,
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "main_dose": float(main_dose),
            "brief_id": brief.id,
            "brief_name": brief.name,
            "bars_sha256": bars.id,
            "bars_calibration_seq": int(cal_note["seq"]),
            "bars_calibrations_in_book": len(hist["calibrations"]),
            "declared_rule": reg_body["declared_rule"],
            "reading_index": reading_index,
            "max_fresh_readings": int(brief.max_fresh_readings),
            "draw_seed_sha256": seed_sha,
            "n_target_items": len(target_drawn),
            "registration_seq": int(reg_entry["seq"]),
            "book_seqs": book_seqs,
            "prior_verdicts": reg_body["prior_verdicts"],
            "prompt_named_site": brief.prompt_named_site,
            "chat_settings": dict(brief.chat_settings),
            "forecast": dict(forecast or {}),
            "design": design,
            "shams": shams_note,
            "kept_back": kept_back_note,
            "book": book_note,
            "book_fresh_readings": dict(hist["book_fresh_readings"]),
            "written_strata": written_strata,
            "settings_audit": settings_audit,
        }
        sentence = claim_sentence(brief, judged, meta)
        sheet = render_sheet(brief, bars, judged, meta, sentence)

        verdict_body = {
            "note_type": "claim_verdict",
            "brief_id": brief.id,
            "brief_name": brief.name,
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "reading_index": reading_index,
            "max_fresh_readings": int(brief.max_fresh_readings),
            "verdict": judged["verdict"],
            "reasons": judged["reasons"],
            "bars_sha256": bars.id,
            "bars_calibration_seq": int(cal_note["seq"]),
            "registration_seq": int(reg_entry["seq"]),
            "book_seqs": book_seqs,
            "claim_sentence": sentence,
            "summary": {
                cls: {
                    "n_frames": judged["classes"][cls]["frames"]["n_frames"],
                    "mean": judged["classes"][cls]["frames"]["mean"],
                    "directed_mean": judged["classes"][cls]["frames"]["directed_mean"],
                    "ci95": judged["classes"][cls]["frames"]["ci95"],
                    "half_0": judged["classes"][cls]["frames"]["half_0"],
                    "half_1": judged["classes"][cls]["frames"]["half_1"],
                    "house_bar": judged["classes"][cls]["house_bar"],
                    "controls": judged["classes"][cls]["controls"],
                    "delivered": judged["classes"][cls].get("delivered"),
                }
                for cls in judged.get("classes") or {}
            },
            "batteries": judged["batteries"],
            "cheaper_knobs": judged["cheaper_knobs"],
            "written_strata": written_strata,
            "settings_audit": settings_audit,
        }
        v_entry = lb.append("note", verdict_body, path=bk)
        we.assert_base(h.lm, h.hooks_at_load)

        # what the reading took, counted from its own entries in the book as `levelling.estimate` counts beforehand
        taken_fw, taken_s = 0, 0.0
        mine = set(book_seqs)
        for ent in lb.entries(bk, start=min(mine)):
            if int(ent["seq"]) > max(mine):
                break
            if int(ent["seq"]) in mine and ent.get("kind") in ("paired", "written"):
                taken_fw += lv.forwards_taken(ent["body"])
                taken_s += float(((ent["body"].get("seconds") or {}).get("actual")) or 0.0)
        if prompt_arm_res:
            scored = [d for d in list(target_drawn) + list(near_drawn)
                      if d.item.question is not None and d.item.answer is not None]
            taken_fw += (1 + len(brief.prompt_arms)) * len(scored)
        cost = {"forwards": int(taken_fw), "reading_seconds": round(taken_s, 3)}

        # Driver-redacted record (aggregates + sheet ONLY — never raw fresh item text or token ids!)
        driver_record = {
            "verdict": judged["verdict"],
            "reasons": judged["reasons"],
            "claim_sentence": sentence,
            "sheet": sheet,
            "brief_id": brief.id,
            "edit_id": edit.id,
            "reading_index": reading_index,
            "max_fresh_readings": int(brief.max_fresh_readings),
            "bars_sha256": bars.id,
            "book": {"registration_seq": int(reg_entry["seq"]), "verdict_seq": int(v_entry["seq"])},
            "summary": verdict_body["summary"],
            "batteries": judged["batteries"],
            "cheaper_knobs": judged["cheaper_knobs"],
            "written_strata": written_strata,
            "settings_audit": settings_audit,
        }
        return {
            "verdict": judged["verdict"],
            "claim_sentence": sentence,
            "sheet": sheet,
            "judged": judged,
            "meta": meta,
            "driver_record": driver_record,
            "cost": cost,
            "book": {"registration_seq": int(reg_entry["seq"]), "verdict_seq": int(v_entry["seq"]),
                     "all_seqs": book_seqs + [int(v_entry["seq"])]},
        }
    finally:
        if prev_bk is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = prev_bk


# --------------------------------------------------------------------------------------------- claim sheet


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "none"
    return f"{float(x):.{digits}f}"


def claim_sentence(brief: lbr.Brief, judged: Mapping[str, Any], meta: Mapping[str, Any]) -> Optional[str]:
    """Bounded claim sentence issued strictly by the harness when verdict is `lead` or `lead, matched by a cheaper knob`."""
    verdict = judged.get("verdict")
    if verdict not in ("lead", "lead, matched by a cheaper knob"):
        return None
    primary_key = "written" if brief.metric == "written_loop" else "large"
    lg = (judged.get("classes") or {}).get(primary_key) or {}
    fr = lg.get("frames") or {}
    hb = lg.get("house_bar") or {}
    ctls = lg.get("controls") or {}
    ci = fr.get("ci95") or [None, None]
    osm = ctls.get("other_sites_matched") or {}
    ss = ctls.get("same_site") or {}
    sh = ctls.get("sham_target") or {}
    sham_part = (
        f"; {sh.get('at_least')} of {sh.get('count')} sham-target control edits, supplied with the claim and checked "
        + ("for shape only" if brief.metric == "written_loop" else
           "for shape and for moving the stream as the edit does")
        + f", moved as much (p_fw {_fmt(sh.get('p_fw'), 3)})"
        if sh.get("count")
        else f"; selection not matched, {judged.get('practice_count', 0)} read in practice"
    )
    kbn = meta.get("kept_back") or {}
    n_pub, n_kb = int(kbn.get("published") or 0), int(kbn.get("frames") or 0)
    if n_pub and n_pub == n_kb:
        pub_part = ", whose templates are published in the instrument's source,"
    elif n_pub:
        pub_part = f", {n_pub} of the brief's {n_kb} templates published in the instrument's source,"
    else:
        pub_part = ""
    bkn = meta.get("book")
    side_part = (", taken in a side book and not the default level book"
                 if isinstance(bkn, Mapping) and not bkn.get("default") else "")
    tol_now = _clean_float((meta.get("design") or {}).get("match_tolerance"))
    tol_part = f"within a factor of {tol_now:g}, " if tol_now is not None else ""
    knobs = judged.get("cheaper_knobs") or {}
    arms_run = knobs.get("prompt_arms") or []
    if verdict == "lead, matched by a cheaper knob":
        knob_part = f"; matched by {knobs['matched_by']}"
    elif arms_run:
        knob_part = "; no author prompt arm matched it"
    else:  # none was run: say so, and why, and never that none matched
        why = knobs.get("prompt_arms_state")
        knob_part = "; no author prompt arm was compared with it" + (
            f" ({why})" if why and why != PROMPT_ARMS_RUN else ""
        )
    named = "named" if meta.get("prompt_named_site") else "did not name"
    if brief.metric == "written_loop":
        rule_now = (meta.get("design") or {}).get("loop_rule") or lbr.loop_rule_of(brief)
        loop_part = (f' Loops were counted by rule ("{rule_now}": levelling.detect_loop and published_loop), which '
                     "no person's reading of this checkpoint's answers has checked.")
    else:
        loop_part = ""
    cal_seq = meta.get("bars_calibration_seq")
    bars_from = f" (calibrated at book line {cal_seq})" if cal_seq is not None else ""
    return (
        f"On {meta.get('date')}, model {meta.get('model_id')} (weights {str(meta.get('weights_id'))[:16]}), "
        f"unpacker {meta.get('unpacker')}, per-row form: edit {str(meta.get('edit_id'))[:12]} at dose "
        f"{meta.get('main_dose'):g} to {', '.join(meta.get('edit_linears') or [])} moved {brief.metric} on "
        f"{fr.get('n_items')} fresh items ({fr.get('n_frames')} kept-back frames{pub_part} drawn from "
        f"{brief.generator.name} after freeze) by {_fmt(fr.get('mean'))} (95% frame CI [{_fmt(ci[0])}, {_fmt(ci[1])}]; "
        f"half 0 {_fmt((fr.get('half_0') or {}).get('mean'))}, half 1 {_fmt((fr.get('half_1') or {}).get('mean'))}); "
        f"floor {_fmt(hb.get('floor'))}, house bar {_fmt(hb.get('bar'))}; "
        f"{osm.get('at_least')} of {osm.get('count')} matched other-site ({tol_part}p_fw {_fmt(osm.get('p_fw'), 3)}) and "
        f"{ss.get('at_least')} of {ss.get('count')} same-site (p_fw {_fmt(ss.get('p_fw'), 3)}) control edits moved as much"
        f"{sham_part} (fresh reading {meta.get('reading_index')} of {meta.get('max_fresh_readings')} on brief "
        f'"{brief.name}"{side_part}); reversed dose moved {_fmt(lg.get("reversed_mean"))}; '
        f"gating batteries passed against hashed bar set {str(meta.get('bars_sha256'))[:12]}{bars_from}{knob_part} "
        f"(no search over cheaper knobs; not compared with a trained adapter). "
        f"Driver prompt {named} a suspected site.{loop_part} This is a lead, not a finding of record."
    )


def render_sheet(
    brief: lbr.Brief,
    bars: BarSet,
    judged: Mapping[str, Any],
    meta: Mapping[str, Any],
    sentence: Optional[str],
) -> str:
    """Human-readable Phase 3 Claim Sheet generated strictly by the harness (zero kept-out words)."""
    lines: List[str] = [
        f'CLAIM SHEET — brief "{brief.name}" [{brief.short()}] — reading {meta.get("reading_index")} of {meta.get("max_fresh_readings")}',
        f"verdict: {judged['verdict']}",
        f"edit: {str(meta.get('edit_id'))[:16]} (dose {meta.get('main_dose'):g}, linears: {', '.join(meta.get('edit_linears') or [])})",
        f"base: {meta.get('model_id')} (weights {str(meta.get('weights_id'))[:16]}, unpacker {meta.get('unpacker')})",
        f"rule: {RULE_PRESET} v{RULE_VERSION} | bars {bars.short()}"
        + (f" from the calibration at book line {meta['bars_calibration_seq']}"
           + (f" (one of {meta['bars_calibrations_in_book']} calibrations of this brief in the book)"
              if (meta.get("bars_calibrations_in_book") or 0) > 1 else "")
           if meta.get("bars_calibration_seq") is not None else "")
        + f" (alpha={bars.alpha:g}, least_effect M={brief.least_effect:g})",
        f"chat settings: {json.dumps(meta.get('chat_settings') or {}, sort_keys=True)} | driver prompt named site: {bool(meta.get('prompt_named_site'))}",
        f"selection: {judged.get('selection_mode')} | practice reads: {judged.get('practice_count', 0)} | practice best, as reported with the claim: {_fmt(judged.get('practice_best'))}",
    ]
    dz = meta.get("design")
    if isinstance(dz, Mapping):
        rz = dz.get("rows") or {}
        lines.append(
            f"design, registered before the draw: {dz.get('n_target')} fresh items, {dz.get('n_near')} near misses, "
            f"{dz.get('n_neutral')} neutral, {dz.get('n_panel')} panel; rows {rz.get('small')} and {rz.get('large')}; "
            f"{dz.get('controls_per_family')} control edits a family; other sites matched within a factor of "
            f"{float(dz.get('match_tolerance')):g} on delivered"
        )
    bz = meta.get("book")
    if isinstance(bz, Mapping):
        lines.append(
            f"book: {bz.get('name')} ("
            + ("the default level book of the environment this ran in" if bz.get("default") else
               "a side book, not the default level book: this reading is counted here and nowhere else")
            + ")"
        )
    kz = meta.get("kept_back")
    if isinstance(kz, Mapping):
        n_pub, n_kb = int(kz.get("published") or 0), int(kz.get("frames") or 0)
        lines.append(
            "kept-back frames: no practice reading in this book holds their wording or a near wording (looked for "
            "before this reading was registered), and a session's practice gates refuse both; a question put in other "
            "words is not found by either. "
            + (f"{n_pub} of the brief's {n_kb} templates are published in the instrument's source (a starter brief's), "
               "so they are not hidden from a driver that has read it"
               if n_pub else
               f"None of the brief's {n_kb} templates is a published starter template; the harness cannot say who "
               "has read the brief's own file")
        )
        lines.append(
            "retention batteries: their questions are few, fixed and in the instrument's source (the neutral battery's "
            "frames, the panel's facts and sums); a session's practice gates refuse their wording"
        )
    sz = meta.get("shams")
    if isinstance(sz, Mapping):
        lines.append(
            f"sham-target controls: {sz.get('count')} supplied with the claim [{str(sz.get('sha256'))[:12]}]; the "
            "harness did not make them. It checked that each has the edit's shape and, in a paired reading, counts "
            "one that does not move the stream as the edit does as no control (as moving as much); it cannot check "
            "that they follow the declared rule"
        )
    if meta.get("declared_rule"):
        lines.append(f"declared rule, in the words given with the claim: \"{meta['declared_rule']}\"")
    if judged.get("reasons"):
        lines.append("gate notes:")
        for r in judged["reasons"]:
            lines.append(f"  - {r}")
    if sentence:
        lines.append("bounded claim:")
        lines.append(f"  {sentence}")
    cls_order = ("written",) if brief.metric == "written_loop" else ("small", "large")
    lines.append("reading classes (kept-back frames):")
    for cls_name in cls_order:
        ce = (judged.get("classes") or {}).get(cls_name)
        if not ce:
            continue
        fr = ce["frames"]
        hb = ce["house_bar"]
        ci = fr["ci95"]
        lines.append(
            f"  {cls_name}: n_frames={fr['n_frames']} ({fr['n_items']} items), mean={_fmt(fr['mean'])} "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}], half0={_fmt(fr['half_0']['mean'])} (n={fr['half_0']['n_frames']}), "
            f"half1={_fmt(fr['half_1']['mean'])} (n={fr['half_1']['n_frames']}), "
            f"floor={_fmt(hb['floor'])}, bar={_fmt(hb['bar'])}, reversed={_fmt(ce['reversed_mean'])}"
        )
        for fam, cs in ce["controls"].items():
            if cs.get("count"):
                lines.append(
                    f"    controls[{fam}]: at_least={cs['at_least']}/{cs['count']}, "
                    f"p_emp={_fmt(cs['p_emp'], 4)}, p_fw={_fmt(cs['p_fw'], 4)}, "
                    f"passes_alpha={cs['passes_alpha']} (tops_every={cs['tops_every_control']}, descriptive only)"
                )
            elif cs.get("note"):
                lines.append(f"    controls[{fam}]: {cs['note']}")
            dv = (ce.get("delivered") or {}).get(fam)
            if isinstance(dv, Mapping) and dv.get("controls_read"):
                lines.append(
                    f"      delivered (stream change at the highest edited layer): edit "
                    f"{_fmt(dv.get('edit'))}, these controls median {_fmt(dv.get('controls_median'))} "
                    f"({_fmt(dv.get('controls_min'))} to {_fmt(dv.get('controls_max'))})"
                    + (f"; {dv['not_comparable']} of them outside the matching tolerance of the edit's, and counted "
                       "as moving as much" if dv.get("not_comparable") else "")
                )
    ws = meta.get("written_strata")
    if isinstance(ws, Mapping):
        lr = ws.get("loop_rules") or {}
        if (lr.get("base") or {}).get("published") is not None:
            lb_, le_ = lr.get("base") or {}, lr.get("edit") or {}
            lines.append(
                f"loops, counted by the rule \"{ws.get('loop_rule', 'instrument')}\" for this reading: of "
                f"{ws.get('total_generations')} generations the instrument's rule calls {lb_.get('instrument')} of the "
                f"base's and {le_.get('instrument')} of the edit's looped, the published rule (arXiv 2606.13705, as "
                f"worded) {lb_.get('published')} and {le_.get('published')}, either {lb_.get('either')} and "
                f"{le_.get('either')}; the two part on {lb_.get('rules_differ')} of the base's. Neither rule is "
                "checked against a person's reading of this checkpoint's answers, and a loop in a form neither finds "
                "counts as none: read the ended and cut-off counts below beside these."
            )
        lines.append(f"written strata ({ws.get('cap_note')}):")
        for sname, sd in (ws.get("strata") or {}).items():
            lines.append(
                f"  {sname}: base={sd['base_count']}/{sd['n_total']}, edit={sd['edit_count']}/{sd['n_total']}, "
                f"note={sd['note']}, least_gain_detectable={_fmt(sd.get('least_gain_detectable'), 3)}"
            )
    sa = meta.get("settings_audit")
    if isinstance(sa, Mapping):
        lines.append(f"base settings audit ({sa.get('note')}):")
        for lbl in ("thinking_off", "thinking_on"):
            sb = sa.get(lbl) or {}
            lines.append(
                f"  {lbl}: n={sb.get('n')}, looped={sb.get('looped')}, cut_off={sb.get('cut_off')}, ended={sb.get('ended')}"
            )
    lines.append("retention batteries:")
    for bname, bv in (judged.get("batteries") or {}).items():
        tag = "GATING" if bv.get("gating") else "info"
        lines.append(f"  {bname} ({tag}): passed={bv.get('passed')} { {k: v for k, v in bv.items() if k not in ('gating', 'passed')} }")
    lines.append("cheaper knobs (no search over cheaper knobs; not compared with a trained adapter):")
    ck = judged.get("cheaper_knobs") or {}
    lines.append(f"  matched_by: {ck.get('matched_by') or 'none'}")
    if ck.get("prompt_arms_state"):
        lines.append(f"  author prompt arms: {ck['prompt_arms_state']}")
    elif not ck.get("prompt_arms"):
        lines.append("  author prompt arms: not compared")
    for pa in ck.get("prompt_arms") or []:
        lines.append(f"  prompt arm {pa['index']} [{pa['prompt_sha256']}]: directed_effect={_fmt(pa.get('directed_effect'))}, near_miss_abs={_fmt(pa.get('near_miss_abs'))}")
    prior = meta.get("prior_verdicts") or []
    others = {k: v for k, v in (meta.get("book_fresh_readings") or {}).items() if k != brief.name}
    if others:
        lines.append("fresh readings already taken on other briefs in this book: "
                     + ", ".join(f"{k} {v}" for k, v in sorted(others.items())))
    lines.append(f"earlier fresh readings on this brief ({len(prior)}):")
    if not prior:
        lines.append("  none")
    else:
        for pv in prior:
            lines.append(f"  reading {pv.get('reading_index')}: edit {str(pv.get('edit_id'))[:12]} -> {pv.get('verdict')} (book seq {pv.get('seq')})")
    return "\n".join(lines)
