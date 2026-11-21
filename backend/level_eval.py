"""Levelling Phase 6: multi-session driver studies, the reach ladder, the cheaper-knob comparison, a blinded
planted-fault and canary benchmark, and helpers for two protocol tests (`backend/level_eval.py`).

Owner-side orchestration for Phase 6 ("Whether agents help"; `verify/arbiter.md` §5, §8, rows 33 and 34;
`design/synthesis.md` §ix; `design/driver.md` §6):
- Multi-session studies (`EvalStudyPlan`, `run_eval_study`): `N` sessions an arm (3 by default), the arms taken in
  turn one replicate at a time, each session in its own folder with its own side book. A side book starts as a copy
  of the study's calibration book, and a session is scored on the lines it entered itself. `weight_edits.assert_base`
  runs before and after every session. A study folder that was used before is refused; a driver's refusal is recorded
  for its session and the study goes on; however a study ends, its end note, summary and report are written.
- Results are kept arm by arm and never pooled: two arms that differ in anything the plan hashes (driver family,
  model, reasoning effort, driver settings, reach, mode, `--bare`) are two rows. Four comparisons are made, each
  between arms that differ in one thing only:
  1. weight edits (`mode="weight_edit"`) against cheaper knobs (`mode="knobs_only"`: prompts, decoding, steering);
  2. the reach ladder (`behaviour` -> `activations` -> `weights` -> `atlas`, and `tools_only`);
  3. agent drivers against the scripted no-driver baselines;
  4. `--bare` against the standard prompt (`docs/SCAFFOLD.md`).
  A comparison names an arm as ahead only when a two-sided Fisher test on the two arms' lead counts is below
  `LABEL_P`, with at least `MIN_VARIANCE_SESSIONS` sessions on each side. At 3 sessions an arm the smallest p such a
  test can give is 0.1, so no comparison at that size can separate two arms, and the summary and report say so.
- A blinded planted-fault and canary benchmark (`run_blinded_benchmark`), scored outside the session. Its docstring
  says what it can show and what it cannot.
- Helpers for the ruling's two end-of-phase driver tests (`arbiter.md` §5), neither of which has been run:
  `render_cold_driver_page` (the one-page sheet a cold driver would be given), and `make_rebuild_description` with
  `run_rebuild_replay` (a description of a paired reading in at most 200 words, and a replay of it in this process).

Not imported by `level_mcp.py`, `level_session.py`, `level_routes.py` or `scientist_tools.py`.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend import atlas_ops as ao  # noqa: F401
from backend import level_atlas as la
from backend import level_book as lb
from backend import level_briefs as lbr
from backend import level_claim as lcl
from backend import level_drivers as ldrv
from backend import level_knobs as lkn  # noqa: F401
from backend import level_mcp as lmcp
from backend import level_session as lses
from backend import levelling as lv
from backend import weight_edits as we
from backend.census import write_json_atomic

EVAL_VERSION = 2  # 2: the study book's start note keeps the plan under "plan"; side books are tied to the study book
MIN_VARIANCE_SESSIONS = 3
REACH_ORDER = ("behaviour", "activations", "weights", "atlas", "tools_only", "sandbox_no_weights", "sandbox", "attended")
FORBIDDEN_CONFIG_KEYS = ("api_key", "secret", "token", "password", "auth_token")
MAX_COLD_PAGE_WORDS = 600
MAX_REBUILD_WORDS = 200
LABEL_P = 0.05  # a comparison names an arm as ahead only when a two-sided test on the lead counts is below this
EFFORT_FAMILIES = ("openai", "claude")  # the drivers `level_drivers.run_driver` passes a reasoning effort to
STUDY_FOLDER_FILES = ("study_book.jsonl", "study_plan.json", "calibration_book.jsonl", "bars.json", "summary.json",
                      "report.md", "sessions")
COLD_RULING_NOT_RUN = (
    "The ruling's cold-driver test (arbiter.md §5: drivers of two families, each given only this page, run a "
    "session) has not been run. This renders the page and counts its words."
)
REBUILD_REPLAY_WHAT = (
    "A same-process replay: the description was parsed and read again in this process, by this code, on this model, "
    "and every arm of every item compared with the original. The ruling's rebuild test (arbiter.md §5: a fresh "
    "session, given only the description, rebuilds the reading) has not been run."
)
COMPOSITION_RULE = (
    "In the harness a repair is read on top of its plant by adding the two edits' weight deltas (weight_edits.stack): "
    "where both scale one neuron, factors f1 and f2 give f1 + f2 - 1, so a scale x2 on a silenced neuron gives "
    "0 + 2 - 1 = 1 and restores it. On a checkpoint carrying the planted weights the repair would act on those "
    "weights instead, and a scale x2 of a silenced neuron leaves it silenced (0 x 2 = 0)."
)
DOOR_SURFACE_NOTE = (
    "A plumbing check. At the start of each session, before its driver runs, it takes what the door shows a driver "
    "(the tools served with their schemas, the status reply, and the driver prompt), leaving out the session id, the "
    "backend (which names the driver family) and the wall-clock fields, and compares planted and canary sessions at "
    "the same reach. It shows that the door does not say which kind a session is. It does not show that a driver "
    "cannot tell from its readings: a plant changes them, as it is meant to."
)


class EvalRefused(ValueError):
    """A malformed, leaking, under-budgeted, or policy-refused Phase 6 study plan or check."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _whole(x: Any, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) != int(x):
        raise EvalRefused(f"{what} is a whole number; {x!r} is not.")
    return int(x)


def _number(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise EvalRefused(f"{what} is a number; {x!r} is not.")
    return float(x)


def _canonical_sha(obj: Any) -> str:
    return hashlib.sha256(lb.canonical(obj)).hexdigest()


# -------------------------------------------------------------------------------------- study plan & validation


@dataclass(frozen=True)
class EvalArm:
    """One experimental arm of a Phase 6 driver evaluation study."""

    arm_id: str
    driver_family: str  # "scripted" | "replay" | "openai" | "claude" | "codex"
    reach: str = "weights"
    mode: str = "weight_edit"  # "weight_edit" | "knobs_only"
    bare: bool = False
    driver_model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    driver_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Strip non-serializable callables (e.g. test transport_fn) from canonical JSON representation
        clean_cfg = {
            k: v for k, v in (self.driver_config or {}).items()
            if not callable(v)
        }
        return {
            "arm_id": self.arm_id,
            "driver_family": self.driver_family,
            "driver_model": self.driver_model,
            "reach": self.reach,
            "mode": self.mode,
            "bare": bool(self.bare),
            "reasoning_effort": self.reasoning_effort,
            "driver_config": clean_cfg,
        }


@dataclass(frozen=True)
class EvalStudyPlan:
    """Versioned, pre-flight validated plan for a Phase 6 multi-session comparison study."""

    name: str
    brief_name: str
    arms: Tuple[EvalArm, ...]
    sessions_per_arm: int = MIN_VARIANCE_SESSIONS
    session_budget: lses.SessionBudget = field(default_factory=lses.SessionBudget)
    driver_budget: ldrv.DriverBudget = field(default_factory=ldrv.DriverBudget)
    claim_design: Dict[str, Any] = field(default_factory=dict)
    brief_path: Optional[str] = None
    max_total_dollars: Optional[float] = None
    max_total_wall_seconds: float = 14400.0
    seed_base: int = 0
    note: str = ""

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "v": EVAL_VERSION,
            "name": self.name,
            "brief_name": self.brief_name,
            "brief_path": self.brief_path,
            "sessions_per_arm": int(self.sessions_per_arm),
            "arms": [a.to_dict() for a in self.arms],
            "session_budget": self.session_budget.to_dict(),
            "driver_budget": self.driver_budget.to_dict(),
            "claim_design": dict(self.claim_design),
            "max_total_dollars": float(self.max_total_dollars) if self.max_total_dollars is not None else None,
            "max_total_wall_seconds": float(self.max_total_wall_seconds),
            "seed_base": int(self.seed_base),
            "note": self.note,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(lb.canonical(self.canonical_dict())).hexdigest()


_PLAN_KEYS = ("v", "study_sha256", "name", "brief_name", "brief_path", "arms", "sessions_per_arm", "session_budget",
              "driver_budget", "claim_design", "max_total_dollars", "max_total_wall_seconds", "seed_base", "note")
_ARM_KEYS = ("arm_id", "driver_family", "driver_model", "reach", "mode", "bare", "reasoning_effort", "driver_config")


def _budget_from(raw: Any, cls: Any, what: str) -> Any:
    """A budget from a plan's JSON. A key the budget does not take, or a value that is not a number of the field's
    kind, is refused in plain words (a plan file is written by hand as often as by `plan`)."""
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise EvalRefused(f"{what} is a JSON object of numbers; a {type(raw).__name__} is not.")
    kinds: Dict[str, str] = {}
    for f in dataclasses.fields(cls):
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        kinds[f.name] = str(t)
    unknown = sorted(str(k) for k in raw if k not in kinds)
    if unknown:
        raise EvalRefused(f"{what} takes {', '.join(kinds)}; not {', '.join(unknown)}.")
    vals: Dict[str, Any] = {}
    for k, v in raw.items():
        t = kinds[k]
        if v is None and "Optional" in t:
            vals[k] = None
        elif t.startswith("int"):
            vals[k] = _whole(v, f"{what}.{k}")
        else:
            vals[k] = _number(v, f"{what}.{k}")
    return cls(**vals)


def study_plan_from_dict(raw: Mapping[str, Any]) -> EvalStudyPlan:
    """An `EvalStudyPlan` from a plan's JSON. A key the plan, an arm or a budget does not take, or a value of the wrong
    kind, is refused in plain words; so is a `study_sha256` (which `scripts/levelling_eval.py plan` writes) that is not
    the hash of the plan as read."""
    if not isinstance(raw, Mapping):
        raise EvalRefused("A study plan is a JSON object.")
    unknown = sorted(str(k) for k in raw if k not in _PLAN_KEYS)
    if unknown:
        raise EvalRefused(f"A study plan takes {', '.join(_PLAN_KEYS)}; not {', '.join(unknown)}.")
    if "v" in raw and raw["v"] != EVAL_VERSION:
        raise EvalRefused(f"This plan is in version {raw['v']!r} of the study format; this code reads version {EVAL_VERSION}.")
    arms_raw = raw.get("arms")
    if not isinstance(arms_raw, Sequence) or isinstance(arms_raw, (str, bytes)) or not arms_raw:
        raise EvalRefused("A study plan needs a non-empty list of arms.")
    arms: List[EvalArm] = []
    for i, ar in enumerate(arms_raw):
        if not isinstance(ar, Mapping):
            raise EvalRefused(f"arms[{i}] is a JSON object.")
        bad = sorted(str(k) for k in ar if k not in _ARM_KEYS)
        if bad:
            raise EvalRefused(f"arms[{i}] takes {', '.join(_ARM_KEYS)}; not {', '.join(bad)}.")
        for k in ("arm_id", "driver_family", "reach", "mode"):
            if k in ar and not isinstance(ar[k], str):
                raise EvalRefused(f"arms[{i}].{k} is a string; {ar[k]!r} is not.")
        for k in ("driver_model", "reasoning_effort"):
            if ar.get(k) is not None and not isinstance(ar[k], str):
                raise EvalRefused(f"arms[{i}].{k} is a string or null; {ar[k]!r} is neither.")
        if "bare" in ar and not isinstance(ar["bare"], bool):
            raise EvalRefused(f"arms[{i}].bare is true or false; {ar['bare']!r} is neither.")
        if ar.get("driver_config") is not None and not isinstance(ar["driver_config"], Mapping):
            raise EvalRefused(f"arms[{i}].driver_config is a JSON object.")
        arms.append(EvalArm(
            arm_id=str(ar.get("arm_id") or f"arm_{i}"),
            driver_family=str(ar.get("driver_family") or "scripted"),
            reach=str(ar.get("reach") or "weights"),
            mode=str(ar.get("mode") or "weight_edit"),
            bare=bool(ar.get("bare", False)),
            driver_model=ar.get("driver_model"),
            reasoning_effort=ar.get("reasoning_effort"),
            driver_config=dict(ar.get("driver_config") or {}),
        ))
    for k in ("name", "brief_name", "brief_path", "note"):
        if raw.get(k) is not None and not isinstance(raw[k], str):
            raise EvalRefused(f"{k} is a string; {raw[k]!r} is not.")
    cd = raw.get("claim_design")
    if cd is not None and not isinstance(cd, Mapping):
        raise EvalRefused("claim_design is a JSON object.")
    mtd = raw.get("max_total_dollars")
    plan = EvalStudyPlan(
        name=str(raw.get("name") or "phase6_study"),
        brief_name=str(raw.get("brief_name") or "pronoun_pull"),
        brief_path=raw.get("brief_path"),
        arms=tuple(arms),
        sessions_per_arm=_whole(raw.get("sessions_per_arm", MIN_VARIANCE_SESSIONS), "sessions_per_arm"),
        session_budget=_budget_from(raw.get("session_budget"), lses.SessionBudget, "session_budget"),
        driver_budget=_budget_from(raw.get("driver_budget"), ldrv.DriverBudget, "driver_budget"),
        claim_design=dict(cd or {}),
        max_total_dollars=None if mtd is None else _number(mtd, "max_total_dollars"),
        max_total_wall_seconds=_number(raw.get("max_total_wall_seconds", 14400.0), "max_total_wall_seconds"),
        seed_base=_whole(raw.get("seed_base", 0), "seed_base"),
        note=str(raw.get("note") or ""),
    )
    want = raw.get("study_sha256")
    if want is not None and want != plan.sha256:
        raise EvalRefused(
            f"The plan file's study_sha256 ({str(want)[:12]}) is not the hash of the plan as read "
            f"({plan.sha256[:12]}): it was changed after it was written. Remove study_sha256 to run it as it stands."
        )
    return plan


def validate_study_plan(
    plan: EvalStudyPlan,
    *,
    weights_dir: Optional[Path] = None,
    atlas_dir: Optional[Path] = None,
    ignore_reach_env: bool = False,
    check_paths: bool = True,
) -> Dict[str, Any]:
    """Validate every arm, reach, containment rule and budget ceiling before any session runs.

    `check_paths=False` leaves out the reach prerequisites (a readable `weights_dir` for the weights and atlas
    reaches, an atlas index for the atlas reach), for a plan checked before the model and its folders are at hand;
    `run_eval_study` checks them again with the paths it is given."""
    if not plan.name.strip():
        raise EvalRefused("Study plan requires a non-empty name.")
    if not plan.arms:
        raise EvalRefused("Study plan requires at least one arm.")
    if isinstance(plan.sessions_per_arm, bool) or int(plan.sessions_per_arm) < 1:
        raise EvalRefused(f"sessions_per_arm must be >= 1; got {plan.sessions_per_arm!r}.")
    if plan.max_total_wall_seconds <= 0.0:
        raise EvalRefused("max_total_wall_seconds must be positive.")
    if plan.max_total_dollars is not None and float(plan.max_total_dollars) <= 0.0:
        raise EvalRefused("max_total_dollars must be above 0 when it is given; leave it out for a study that spends nothing.")

    unknown_cd = sorted(k for k in plan.claim_design if k not in lses.CLAIM_DESIGN_KEYS)
    if unknown_cd:
        raise EvalRefused(f"Unknown claim_design key(s): {', '.join(unknown_cd)}.")

    seen_ids: set[str] = set()
    warnings: List[str] = []
    if plan.sessions_per_arm < MIN_VARIANCE_SESSIONS:
        warnings.append(
            f"sessions_per_arm={plan.sessions_per_arm} is below {MIN_VARIANCE_SESSIONS}: session variance will be under-sampled."
        )

    for arm in plan.arms:
        if not arm.arm_id.strip() or not re.match(r"^[A-Za-z0-9_.-]+$", arm.arm_id):
            raise EvalRefused(f"Invalid arm_id {arm.arm_id!r} (use letters, digits, _, ., -).")
        if arm.arm_id in seen_ids:
            raise EvalRefused(f"Duplicate arm_id {arm.arm_id!r}.")
        seen_ids.add(arm.arm_id)
        if arm.driver_family not in ldrv.DRIVER_FAMILIES:
            raise EvalRefused(f"Arm {arm.arm_id!r} has unknown driver_family {arm.driver_family!r}.")
        if arm.reach == "home":
            raise EvalRefused(f"Arm {arm.arm_id!r}: every levelling tool is refused at the home reach.")
        if arm.reach not in lses.VALID_REACHES:
            raise EvalRefused(f"Arm {arm.arm_id!r} has invalid reach {arm.reach!r}.")
        if arm.mode not in lses.VALID_MODES:
            raise EvalRefused(f"Arm {arm.arm_id!r} has invalid mode {arm.mode!r}.")

        # A reasoning effort is hashed into the plan, so it must reach the driver, and only a driver that takes one
        if arm.reasoning_effort is not None:
            if not isinstance(arm.reasoning_effort, str) or not arm.reasoning_effort.strip():
                raise EvalRefused(f"Arm {arm.arm_id!r}: reasoning_effort is a non-empty string or null.")
            if arm.driver_family not in EFFORT_FAMILIES:
                raise EvalRefused(
                    f"Arm {arm.arm_id!r}: a reasoning effort is passed to the {' and '.join(EFFORT_FAMILIES)} drivers "
                    f"only; the {arm.driver_family} driver would run the same without it, under another plan hash."
                )
        cfg_eff = (arm.driver_config or {}).get("reasoning_effort")
        if cfg_eff is not None and arm.reasoning_effort is not None and cfg_eff != arm.reasoning_effort:
            raise EvalRefused(
                f"Arm {arm.arm_id!r} names two reasoning efforts ({arm.reasoning_effort!r}, and {cfg_eff!r} in "
                "driver_config); name one."
            )

        # Never allow raw secrets embedded in study plan JSON
        for k in (arm.driver_config or {}):
            kl = str(k).lower()
            if any(bad in kl for bad in FORBIDDEN_CONFIG_KEYS) and not kl.endswith("_env"):
                raise EvalRefused(
                    f"Arm {arm.arm_id!r} driver_config contains forbidden secret-like key {k!r}; "
                    "pass environment variable names (*_env) only, never raw credentials."
                )

        # Containment check (home reach env, unprobed codex)
        backend_tag = arm.driver_family if arm.driver_family in lses.VALID_BACKENDS else "local"
        if arm.driver_family == "scripted":
            backend_tag = "script"
        try:
            lses._check_reach_and_backend(
                arm.reach,
                backend_tag,
                codex_probed=bool(arm.driver_config.get("codex_probed", False)),
                ignore_reach_env=ignore_reach_env,
            )
        except lses.SessionRefused as exc:
            raise EvalRefused(f"Arm {arm.arm_id!r}: {exc}") from None

        # Reach prerequisites
        if check_paths:
            if arm.reach in ("weights", "atlas") and (weights_dir is None or not Path(weights_dir).is_dir()):
                raise EvalRefused(f'Arm {arm.arm_id!r} asks for reach "{arm.reach}", which requires a readable weights_dir.')
            if arm.reach == "atlas":
                if atlas_dir is None:
                    raise EvalRefused(f'Arm {arm.arm_id!r} asks for reach "atlas", which requires a readable atlas_dir.')
                try:
                    la.resolve_atlas_index_dir(Path(atlas_dir))
                except la.AtlasReaderError as exc:
                    raise EvalRefused(f'Arm {arm.arm_id!r} atlas_dir invalid: {exc}') from None

        # Dollar cap required for paid CLI drivers unless explicitly mocked in tests
        if arm.driver_family in ldrv.PAID_DRIVER_FAMILIES and not arm.driver_config.get("allow_mock_cli"):
            if plan.driver_budget.max_dollars is None or plan.driver_budget.max_dollars <= 0.0:
                raise EvalRefused(
                    f"Arm {arm.arm_id!r} ({arm.driver_family}) requires an owner-approved driver_budget.max_dollars > 0."
                )

    total_sessions = len(plan.arms) * int(plan.sessions_per_arm)
    # A session lasts as long as its driver runs, and the driver's own wall cap is what ends it
    per_session_wall = float(plan.driver_budget.max_wall_seconds)
    wall_need = total_sessions * per_session_wall
    if wall_need > float(plan.max_total_wall_seconds) + 1e-9:
        raise EvalRefused(
            f"{total_sessions} sessions of up to {per_session_wall:g} s each (the driver's wall cap) need up to "
            f"{wall_need:g} s, over max_total_wall_seconds ({float(plan.max_total_wall_seconds):g} s): the study's "
            "cap could stop it before every session ran. Raise the cap, or plan fewer or shorter sessions."
        )
    max_gpu_s = total_sessions * float(plan.session_budget.max_gpu_seconds)
    max_fwds = total_sessions * int(plan.session_budget.max_forwards)
    max_usd = (
        min(
            float(plan.max_total_dollars) if plan.max_total_dollars is not None else float("inf"),
            total_sessions * float(plan.driver_budget.max_dollars),
        )
        if plan.driver_budget.max_dollars is not None
        else plan.max_total_dollars
    )
    return {
        "ok": True,
        "study_sha256": plan.sha256,
        "arms": len(plan.arms),
        "sessions_per_arm": int(plan.sessions_per_arm),
        "total_sessions": total_sessions,
        "paths_checked": bool(check_paths),
        "per_session_wall_seconds": per_session_wall,
        "upper_bound_wall_seconds": round(wall_need, 1),
        "upper_bound_gpu_seconds": round(max_gpu_s, 1),
        "upper_bound_practice_forwards": max_fwds,
        "upper_bound_dollars": round(max_usd, 4) if max_usd is not None and math.isfinite(max_usd) else None,
        "max_total_wall_seconds": float(plan.max_total_wall_seconds),
        "warnings": warnings,
    }


def default_phase6_study_plan(
    *,
    name: str = "phase6_reach_and_knobs",
    brief_name: str = "pronoun_pull",
    sessions_per_arm: int = MIN_VARIANCE_SESSIONS,
    include_atlas: bool = True,
    include_bare_ablation: bool = False,
    driver_family: str = "scripted",
    driver_model: Optional[str] = None,
    second_driver_family: Optional[str] = None,
    second_driver_model: Optional[str] = None,
    session_budget: Optional[lses.SessionBudget] = None,
    driver_budget: Optional[ldrv.DriverBudget] = None,
    claim_design: Optional[Mapping[str, Any]] = None,
    max_total_dollars: Optional[float] = None,
    max_total_wall_seconds: Optional[float] = None,
) -> EvalStudyPlan:
    """Build the canonical Phase 6 comparison matrix (reach ladder + matched-budget cheaper-knobs arm).

    `max_total_wall_seconds` defaults to what every planned session needs at the driver's wall cap."""
    arms: List[EvalArm] = [
        EvalArm(f"{driver_family}_behaviour_edit", driver_family, reach="behaviour", mode="weight_edit", driver_model=driver_model),
        EvalArm(f"{driver_family}_activations_edit", driver_family, reach="activations", mode="weight_edit", driver_model=driver_model),
        EvalArm(f"{driver_family}_weights_edit", driver_family, reach="weights", mode="weight_edit", driver_model=driver_model),
    ]
    if include_atlas:
        arms.append(EvalArm(f"{driver_family}_atlas_edit", driver_family, reach="atlas", mode="weight_edit", driver_model=driver_model))
    # Matched-budget cheaper-knobs competitor (prompts, decoding, steering — zero weight edits)
    knob_reach = "atlas" if include_atlas else "weights"
    arms.append(EvalArm(f"{driver_family}_{knob_reach}_knobs", driver_family, reach=knob_reach, mode="knobs_only", driver_model=driver_model))
    arms.append(EvalArm(f"{driver_family}_behaviour_knobs", driver_family, reach="behaviour", mode="knobs_only", driver_model=driver_model))
    if include_bare_ablation:
        arms.append(EvalArm(f"{driver_family}_weights_edit_bare", driver_family, reach="weights", mode="weight_edit", bare=True, driver_model=driver_model))
    if second_driver_family:
        arms.append(EvalArm(f"{second_driver_family}_weights_edit", second_driver_family, reach="weights", mode="weight_edit", driver_model=second_driver_model))
        arms.append(EvalArm(f"{second_driver_family}_weights_knobs", second_driver_family, reach="weights", mode="knobs_only", driver_model=second_driver_model))

    db = driver_budget or ldrv.DriverBudget(max_dollars=0.0 if driver_family == "scripted" else None)
    wall = (
        float(max_total_wall_seconds) if max_total_wall_seconds is not None
        else float(len(arms) * int(sessions_per_arm)) * float(db.max_wall_seconds)
    )
    return EvalStudyPlan(
        name=name,
        brief_name=brief_name,
        arms=tuple(arms),
        sessions_per_arm=int(sessions_per_arm),
        session_budget=session_budget or lses.SessionBudget(),
        driver_budget=db,
        claim_design=dict(claim_design or {}),
        max_total_dollars=max_total_dollars,
        max_total_wall_seconds=wall,
    )


# -------------------------------------------------------------------------------------- exact statistics (zero scipy)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    """Wilson 95% confidence interval for binomial proportion k/n (valid at k=0, k=n, and small n)."""
    if n <= 0:
        return None, None
    p = k / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> Optional[float]:
    """Exact two-sided Fisher's test p-value for a 2x2 table [[a, b], [c, d]] (stdlib math.comb only)."""
    if min(a, b, c, d) < 0:
        return None
    r1 = a + b
    r2 = c + d
    c1 = a + c
    n = r1 + r2
    if n <= 0 or r1 == 0 or r2 == 0:
        return None
    denom = math.comb(n, c1)
    if denom == 0:
        return None

    def _hyper(x: int) -> float:
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / float(denom)

    p_obs = _hyper(a)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_sum = sum(_hyper(x) for x in range(lo, hi + 1) if _hyper(x) <= p_obs + 1e-12)
    return round(min(1.0, p_sum), 6)


def smallest_fisher_p(n1: int, n2: int) -> Optional[float]:
    """The smallest two-sided Fisher p any 2x2 table with arms of `n1` and `n2` sessions can give (3 and 3: 0.1,
    for 3 of 3 against 0 of 3). When it is not below `LABEL_P`, no result of two arms of these sizes can separate them."""
    if n1 <= 0 or n2 <= 0:
        return None
    best: Optional[float] = None
    for k1 in range(n1 + 1):
        for k2 in range(n2 + 1):
            p = fisher_exact_2x2(k1, n1 - k1, k2, n2 - k2)
            if p is not None and (best is None or p < best):
                best = p
    return best


def exact_perm_diff_p(xs: Sequence[float], ys: Sequence[float], *, seed: int = 0) -> Dict[str, Any]:
    """Two-sided permutation test for difference in means between `xs` and `ys`.
    Exact when `len(xs) + len(ys) <= 14`; deterministic Monte Carlo otherwise. Returns `p=None` if either group < 2."""
    ax = [float(v) for v in xs if _clean_float(v) is not None]
    ay = [float(v) for v in ys if _clean_float(v) is not None]
    nx, ny = len(ax), len(ay)
    if nx < 2 or ny < 2:
        return {"p": None, "exact": False, "note": f"too few observations for a test (n1={nx}, n2={ny})"}
    obs = abs(float(np.mean(ax)) - float(np.mean(ay)))
    pooled = ax + ay
    total = nx + ny
    if total <= 14:
        n_comb = 0
        ge = 0
        sum_all = sum(pooled)
        for comb in itertools.combinations(range(total), nx):
            sx = sum(pooled[i] for i in comb)
            mx = sx / nx
            my = (sum_all - sx) / ny
            n_comb += 1
            if abs(mx - my) >= obs - 1e-12:
                ge += 1
        return {"p": round(ge / float(n_comb), 6), "exact": True, "n1": nx, "n2": ny}
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(pooled, dtype=np.float64)
    B = 5000
    ge = 1
    for _ in range(B):
        perm = rng.permutation(arr)
        diff = abs(float(perm[:nx].mean()) - float(perm[nx:].mean()))
        if diff >= obs - 1e-12:
            ge += 1
    return {"p": round(ge / float(B + 1), 6), "exact": False, "n1": nx, "n2": ny}


def two_arm_lead_test(a: Mapping[str, Any], b: Mapping[str, Any], a_name: str, b_name: str) -> Dict[str, Any]:
    """Compare two arms' lead counts by a two-sided Fisher test. The label names an arm as ahead only when p is below
    `LABEL_P` and each arm ran at least `MIN_VARIANCE_SESSIONS` sessions, the same both ways; otherwise it is
    "neither_found_a_lead" or "no_difference_shown"."""
    a_l, a_n = int(a["n_leads"]), int(a["n_sessions"])
    b_l, b_n = int(b["n_leads"]), int(b["n_sessions"])
    p = fisher_exact_2x2(a_l, a_n - a_l, b_l, b_n - b_l) if a_n > 0 and b_n > 0 else None
    p_min = smallest_fisher_p(a_n, b_n)
    enough = a_n >= MIN_VARIANCE_SESSIONS and b_n >= MIN_VARIANCE_SESSIONS
    can_separate = bool(enough and p_min is not None and p_min < LABEL_P)
    a_rate = a_l / float(a_n) if a_n else 0.0
    b_rate = b_l / float(b_n) if b_n else 0.0
    if a_n == 0 or b_n == 0:
        return {"fisher_exact_p": None, "smallest_possible_p": None, "label_p": LABEL_P, "can_separate": False,
                "label": "not_compared", "why": f"an arm ran no session ({a_n} and {b_n})"}
    if a_l == 0 and b_l == 0:
        label = "neither_found_a_lead"
    elif enough and p is not None and p < LABEL_P and a_rate > b_rate:
        label = f"{a_name}_ahead"
    elif enough and p is not None and p < LABEL_P and b_rate > a_rate:
        label = f"{b_name}_ahead"
    else:
        label = "no_difference_shown"
    if not enough:
        why = f"fewer than {MIN_VARIANCE_SESSIONS} sessions on a side ({a_n} and {b_n}): no arm is named ahead"
    elif not can_separate:
        why = (f"with {a_n} and {b_n} sessions the smallest two-sided Fisher p any result can give is {p_min:g}, "
               f"not below {LABEL_P:g}: no result at these sizes can separate the two arms")
    elif label.endswith("_ahead"):
        why = f"two-sided Fisher p = {p:g}, below {LABEL_P:g}"
    else:
        why = f"two-sided Fisher p = {p:g}, not below {LABEL_P:g}" if p is not None else "no test could be made"
    return {
        "fisher_exact_p": p,
        "smallest_possible_p": p_min,
        "label_p": LABEL_P,
        "can_separate": can_separate,
        "label": label,
        "why": why,
    }


# -------------------------------------------------------------------------------------- study execution


def _resolve_brief(plan: EvalStudyPlan, tokenizer: Any) -> lbr.Brief:
    if plan.brief_path:
        return lbr.load_brief(Path(plan.brief_path))
    starters = lbr.starter_briefs(tokenizer)
    if plan.brief_name not in starters:
        raise EvalRefused(
            f"Unknown brief_name {plan.brief_name!r}; supply brief_path or choose from {', '.join(sorted(starters))}."
        )
    return starters[plan.brief_name]


def _default_calibration_layer(h: lv.Handle, brief: lbr.Brief) -> int:
    """Pick an allowed late-middle layer for `calibrate_bars`."""
    allowed_l = list(brief.allowed_layers) if brief.allowed_layers is not None else list(range(h.n_layers))
    return int(allowed_l[min(len(allowed_l) - 1, max(0, int(round(0.83 * (len(allowed_l) - 1)))))])


@contextlib.contextmanager
def _use_book(path: Path) -> Iterator[None]:
    prev = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(Path(path).resolve())
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = prev


def refuse_used_study_dir(study_dir: Path) -> None:
    """Refuse a study folder a study has used: a second study there would be read as one with the first (a STOP file
    alone does not make a folder used; the study stops on it before calibration)."""
    sdir = Path(study_dir)
    found = [n for n in STUDY_FOLDER_FILES if (sdir / n).exists()]
    if found:
        raise EvalRefused(
            f"The study folder {sdir} was used before (it holds {', '.join(found)}): a second study there would be "
            "read as one with the first. Choose a new folder."
        )


def _arm_driver_config(arm: EvalArm) -> Dict[str, Any]:
    """The settings `level_drivers.run_driver` is given for `arm`: its driver_config with its reasoning effort, which
    run_driver reads from the settings (for the openai and claude drivers)."""
    cfg = dict(arm.driver_config or {})
    if arm.reasoning_effort is not None:
        cfg["reasoning_effort"] = arm.reasoning_effort
    return cfg


def _session_driver_budget(plan: EvalStudyPlan, spent: float) -> ldrv.DriverBudget:
    """The driver budget one session runs under. With a study dollar cap, its dollar cap is what is left of the
    study's, or the plan's own per-session cap where that is smaller, so the session in progress counts against the
    study's total while it runs, not only after it."""
    db = plan.driver_budget
    if plan.max_total_dollars is None:
        return db
    left = max(0.0, float(plan.max_total_dollars) - float(spent))
    cap = db.max_dollars
    per = left if (cap is None or float(cap) <= 0.0) else min(float(cap), left)
    return dataclasses.replace(db, max_dollars=round(per, 6))


def _score_own_lines(outcome: Mapping[str, Any], own_from: int) -> Dict[str, Any]:
    """`outcome` scored on the lines its session entered: those from line `own_from` of its side book on, the lines
    before being the copy of the calibration book the side book starts with. `level_drivers` reads every line of a
    book for the last claim verdict, so a verdict in that copy (an earlier claim in a supplied calibration book) would
    be read as the session's own when the session entered none."""
    out = dict(outcome)
    claims = [c for c in (outcome.get("all_claims") or []) if int(c.get("seq", -1)) >= int(own_from)]
    out["all_claims"] = claims
    out["claims_count"] = len(claims)
    out["own_lines_from"] = int(own_from)
    vseq = (outcome.get("book") or {}).get("verdict_seq")
    if vseq is not None and int(vseq) < int(own_from):
        out["seed_verdict_set_aside"] = {"seq": int(vseq), "verdict": outcome.get("verdict")}
        out["verdict"] = "no_submission"
        out["submitted"] = False
        out["candidate_id"] = None
        out["knob_classes"] = []
        out["metrics"] = {k: None for k in (outcome.get("metrics") or {})}
        out["book"] = {**(outcome.get("book") or {}), "verdict_seq": None}
        if outcome.get("stop_reason") == "claimed":
            # level_drivers turns "kept_base_abstention" into "claimed" when it finds a verdict; a driver says
            # "claimed" itself only after a claim the gate took, and that enters a verdict among the session's lines
            out["stop_reason"] = "kept_base_abstention"
    return out


def _refused_outcome(arm: EvalArm, exc: BaseException, db: ldrv.DriverBudget, run_book: Path) -> Dict[str, Any]:
    """The record of a session whose driver or session was refused: not run, and not counted as an abstention."""
    ver = lb.verify(run_book)
    return {
        "v": EVAL_VERSION,
        "finished_at": _utc_now(),
        "refused": True,
        "refusal": f"{type(exc).__name__}: {exc}"[:2000],
        "harness_card": {
            "driver_family": arm.driver_family,
            "driver_model": arm.driver_model,
            "reasoning_effort": arm.reasoning_effort,
            "reach": arm.reach,
            "mode": arm.mode,
            "bare": bool(arm.bare),
            "driver_budget": db.to_dict(),
        },
        "stop_reason": "refused",
        "submitted": False,
        "verdict": "not run",
        "candidate_id": None,
        "knob_classes": [],
        "claims_count": 0,
        "all_claims": [],
        "metrics": {},
        "usage": {},
        "book": {"path": str(run_book), "verified": bool(ver["ok"]), "entries": int(ver["entries"]),
                 "head": ver["head"], "verdict_seq": None},
        "proposed_edit": None,
        "proposed_knob": None,
        "notes": "",
    }


def _supplied_calibration(
    bars: lcl.BarSet, calibration_book: Path, brief: lbr.Brief, cal_book_path: Path, bars_path: Path
) -> Tuple[lcl.BarSet, str]:
    cal_src = Path(calibration_book).resolve()
    note = lcl.calibration_note(bars, cal_src)
    if note is None:
        raise EvalRefused(f"Provided BarSet {bars.short()} has no calibration note in {cal_src}.")
    why_cal = lcl.calibration_follows(note, cal_src, brief, bars)
    if why_cal is not None:
        raise EvalRefused(f"Provided calibration does not follow from {cal_src}: {why_cal}")
    shutil.copy2(cal_src, cal_book_path)
    lcl.save_bars(bars, bars_path)
    return bars, str(cal_src)


def run_eval_study(
    h: lv.Handle,
    plan: EvalStudyPlan,
    study_dir: Path,
    *,
    weights_dir: Optional[Path] = None,
    atlas_dir: Optional[Path] = None,
    bars: Optional[lcl.BarSet] = None,
    calibration_book: Optional[Path] = None,
    kill_switch: Optional[ldrv.KillSwitch] = None,
    probe_sandbox: bool = True,
    ignore_reach_env: bool = False,
    cal_controls: int = 3,
    cal_rows: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Run a Phase 6 multi-session study, each session in its own folder with its own side book.

    The study book (`study_book.jsonl` in `study_dir`) holds a start note with the plan, a calibration note (the bars
    and the head and length of the calibration book every side book starts as), one note a session (its outcome, and
    its side book's path, head and length), and an end note. The arms are taken in turn, replicate by replicate, so
    a cap or a stop that ends the study early leaves every arm with about as many sessions as the others. A session's
    dollar cap is what is left of the study's (or the plan's own per-session cap, where smaller). A driver's or
    session's refusal is recorded for its session, which is not counted as run, and the study goes on; any other
    error stops it. However the study ends, the end note, `summary.json` and `report.md` are written; an error is
    raised again after them. A STOP file present at the start stops the study before calibration."""
    we.assert_base(h.lm, h.hooks_at_load)
    val = validate_study_plan(plan, weights_dir=weights_dir, atlas_dir=atlas_dir, ignore_reach_env=ignore_reach_env)
    if (bars is None) != (calibration_book is None):
        raise EvalRefused("bars and calibration_book are given together (a supplied calibration) or not at all.")
    sdir = Path(study_dir).resolve()
    refuse_used_study_dir(sdir)
    brief = _resolve_brief(plan, h.tokenizer)

    sdir.mkdir(parents=True, exist_ok=True)
    sessions_root = sdir / "sessions"
    study_book = sdir / "study_book.jsonl"
    cal_book_path = sdir / "calibration_book.jsonl"
    bars_path = sdir / "bars.json"
    ks = kill_switch or ldrv.KillSwitch(stop_file=sdir / "STOP")
    planned = len(plan.arms) * int(plan.sessions_per_arm)

    plan_doc: Dict[str, Any] = {
        **plan.canonical_dict(),
        "study_sha256": plan.sha256,
        "brief_id": brief.id,
        "bars_sha256": None,
        "model_id": h.model_id,
        "weights_id": h.weights_id,
        "unpacker": lv.unpacker_describe().get("mode"),
        "started_at": _utc_now(),
        "validation": val,
    }
    write_json_atomic(sdir / "study_plan.json", plan_doc)
    start_entry = lb.append(
        "note", {"note": "eval_study_start", "study_sha256": plan.sha256, "plan": plan_doc}, path=study_book
    )

    outcomes: List[Dict[str, Any]] = []
    halted_reason: Optional[str] = None
    total_dollars = 0.0
    t_study = time.monotonic()
    refused_n = 0
    try:
        why_ks = ks.triggered()
        if why_ks:
            halted_reason = f"kill_switch: {why_ks} (before calibration)"
        else:
            # 1. Calibrate bars once (or verify a supplied calibration) into the study's calibration book
            if bars is not None and calibration_book is not None:
                study_bars, cal_source = _supplied_calibration(bars, calibration_book, brief, cal_book_path, bars_path)
            else:
                study_bars = lcl.calibrate_bars(
                    h, brief, layer=_default_calibration_layer(h, brief), n_controls=int(cal_controls),
                    rows=dict(cal_rows) if cal_rows else None, book_path=cal_book_path,
                )
                lcl.save_bars(study_bars, bars_path)
                cal_source = "calibrated in this study"
            we.assert_base(h.lm, h.hooks_at_load)
            cal_ver = lb.verify(cal_book_path)
            if not cal_ver["ok"]:
                raise EvalRefused(f"The study's calibration book fails at line {cal_ver['first_bad']}: {cal_ver['why']}")
            plan_doc["bars_sha256"] = study_bars.id
            lb.append("note", {
                "note": "eval_calibration",
                "study_sha256": plan.sha256,
                "bars_sha256": study_bars.id,
                "calibration_book": {"path": cal_book_path.name, "entries": int(cal_ver["entries"]),
                                     "head": cal_ver["head"], "source": cal_source},
            }, path=study_book)

            # 2. Every (replicate, arm), the arms in turn, each in its own folder and side book
            t_study = time.monotonic()
            with ks.install_signals():
                for rep in range(int(plan.sessions_per_arm)):
                    for arm in plan.arms:
                        why_ks = ks.triggered()
                        if why_ks:
                            halted_reason = f"kill_switch: {why_ks}"
                        elif (time.monotonic() - t_study) >= float(plan.max_total_wall_seconds):
                            halted_reason = "study_wall_cap"
                        elif plan.max_total_dollars is not None and total_dollars >= float(plan.max_total_dollars):
                            halted_reason = "study_dollar_cap"
                        if halted_reason:
                            break
                        outcome = _run_one_session(
                            h, plan, arm, rep, sessions_root, cal_book_path, cal_ver, brief, study_bars,
                            weights_dir=weights_dir, atlas_dir=atlas_dir, kill_switch=ks, spent=total_dollars,
                            probe_sandbox=probe_sandbox, ignore_reach_env=ignore_reach_env,
                        )
                        refused_n += 1 if outcome.get("refused") else 0
                        total_dollars += float((outcome.get("usage") or {}).get("estimated_dollars") or 0.0)
                        outcomes.append(outcome)
                        run_dir = sessions_root / f"{arm.arm_id}_r{rep}"
                        write_json_atomic(run_dir / "outcome.json", outcome)
                        side = lb.verify(run_dir / "level_book.jsonl")
                        lb.append("note", {
                            "note": "eval_session",
                            "study_sha256": plan.sha256,
                            "arm_id": arm.arm_id,
                            "replicate": rep,
                            "side_book": {
                                "path": str((run_dir / "level_book.jsonl").relative_to(sdir)),
                                "verified": bool(side["ok"]),
                                "entries": int(side["entries"]),
                                "head": side["head"],
                                "seed_entries": int(cal_ver["entries"]),
                                "seed_head": cal_ver["head"],
                            },
                            "outcome": outcome,
                        }, path=study_book)
                    if halted_reason:
                        break
    except BaseException as exc:
        halted_reason = f"error: {type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        summary = summarize_study(outcomes, plan_doc, halted_reason=halted_reason, finished=True)
        report_md = render_study_report(summary)
        (sdir / "report.md").write_text(report_md + "\n", encoding="utf-8")
        write_json_atomic(sdir / "summary.json", summary)
        end_entry = lb.append("note", {
            "note": "eval_study_end",
            "study_sha256": plan.sha256,
            "finished_at": _utc_now(),
            "halted_reason": halted_reason,
            "completed_sessions": len(outcomes),
            "refused_sessions": refused_n,
            "planned_sessions": planned,
            "total_seconds": round(time.monotonic() - t_study, 3),
            "total_estimated_dollars": round(total_dollars, 6),
        }, path=study_book)
    book_ver = lb.verify(study_book)
    return {
        "study_dir": str(sdir),
        "study_sha256": plan.sha256,
        "halted_reason": halted_reason,
        "summary": summary,
        "report": report_md,
        "outcomes": outcomes,
        "book": {
            "path": str(study_book),
            "verified": bool(book_ver["ok"]),
            "entries": int(book_ver["entries"]),
            "head": book_ver["head"],
            "start_seq": int(start_entry["seq"]),
            "end_seq": int(end_entry["seq"]),
        },
    }


def _run_one_session(
    h: lv.Handle,
    plan: EvalStudyPlan,
    arm: EvalArm,
    rep: int,
    sessions_root: Path,
    cal_book_path: Path,
    cal_ver: Mapping[str, Any],
    brief: lbr.Brief,
    study_bars: lcl.BarSet,
    *,
    weights_dir: Optional[Path],
    atlas_dir: Optional[Path],
    kill_switch: ldrv.KillSwitch,
    spent: float,
    probe_sandbox: bool,
    ignore_reach_env: bool,
) -> Dict[str, Any]:
    run_id = f"{arm.arm_id}_r{rep}"
    run_dir = sessions_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_book = run_dir / "level_book.jsonl"
    shutil.copy2(cal_book_path, run_book)
    seed_ver = lb.verify(run_book)
    if not seed_ver["ok"] or seed_ver["head"] != cal_ver["head"] or seed_ver["entries"] != cal_ver["entries"]:
        raise EvalRefused(f"The side book of {run_id} is not the calibration book it was copied from.")
    own_from = int(cal_ver["entries"])
    db = _session_driver_budget(plan, spent)
    backend_tag = arm.driver_family if arm.driver_family in lses.VALID_BACKENDS else "local"
    if arm.driver_family == "scripted":
        backend_tag = "script"

    sess: Optional[lses.LevelSession] = None
    try:
        we.assert_base(h.lm, h.hooks_at_load)
        sess = lses.LevelSession(
            h,
            run_dir,
            reach=arm.reach,
            mode=arm.mode,
            bare=arm.bare,
            backend=backend_tag,
            budget=plan.session_budget,
            book_path=run_book,
            weights_dir=weights_dir,
            atlas_dir=atlas_dir,
            probe_on_start=probe_sandbox,
            codex_probed=bool(arm.driver_config.get("codex_probed", False)),
            ignore_reach_env=ignore_reach_env,
            brief=brief,
            bars=study_bars,
            claim_design=plan.claim_design or None,
            session_id=f"{plan.name}_{run_id}",
        )
        outcome = ldrv.run_driver(
            sess,
            arm.driver_family,
            driver_model=arm.driver_model,
            driver_budget=db,
            kill_switch=kill_switch,
            seed=int(plan.seed_base) + rep,
            config=_arm_driver_config(arm),
        )
        outcome = _score_own_lines(outcome, own_from)
    except (ldrv.DriverRefused, lses.SessionRefused) as exc:
        outcome = _refused_outcome(arm, exc, db, run_book)
    finally:
        if sess is not None and not sess._closed:
            sess.close()
        we.assert_base(h.lm, h.hooks_at_load)
    outcome["arm_id"] = arm.arm_id
    outcome["replicate"] = rep
    outcome["run_id"] = run_id
    outcome["study_sha256"] = plan.sha256
    return outcome


# -------------------------------------------------------------------------------------- aggregation (never pooled)


def _arm_specs(plan_doc: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(a.get("arm_id")): dict(a) for a in (plan_doc.get("arms") or []) if isinstance(a, Mapping)}


def _arm_group_key(
    outcome: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    served: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[Any, ...]:
    """The row an outcome belongs to: its arm, and everything the plan hashes for that arm (driver family, the model
    that ran, reasoning effort, driver settings, reach, mode, `--bare`). Two arms are never one row, so two arms that
    differ only in effort or in their driver settings are not pooled. A refused session ran no model, and joins its
    arm's row (`served`: the models an arm's sessions that ran were served)."""
    hc = outcome.get("harness_card") or {}
    aid = str(outcome.get("arm_id") or "")
    spec = specs.get(aid) or {}
    fam = str(spec.get("driver_family") or hc.get("driver_family") or "unknown")
    ran_models = sorted((served or {}).get(aid) or [])
    if outcome.get("refused") and ran_models:
        model = ran_models[0]
    else:
        model = str(hc.get("driver_model") or spec.get("driver_model") or "unknown")
    reach = str(spec.get("reach") or hc.get("reach") or "unknown")
    mode = str(spec.get("mode") or hc.get("mode") or "weight_edit")
    bare = bool(spec["bare"]) if "bare" in spec else bool(hc.get("bare", False))
    effort = spec.get("reasoning_effort") if "reasoning_effort" in spec else hc.get("reasoning_effort")
    cfg = spec.get("driver_config") if isinstance(spec.get("driver_config"), Mapping) else {}
    return (aid, fam, model, reach, mode, bare, effort or "", _canonical_sha(dict(cfg))[:16])


def _median(vals: Sequence[float]) -> Optional[float]:
    return round(float(np.median(np.asarray(vals, dtype=np.float64))), 5) if vals else None


def summarize_study(
    outcomes: Sequence[Mapping[str, Any]],
    plan_doc: Mapping[str, Any],
    *,
    halted_reason: Optional[str] = None,
    finished: bool = True,
) -> Dict[str, Any]:
    """Aggregate outcomes arm by arm, never pooling two arms (`_arm_group_key`), and compare arms that differ in one
    thing only. A session that was refused is listed and not counted among the sessions an arm ran."""
    specs = _arm_specs(plan_doc)
    served: Dict[str, List[str]] = {}
    for o in outcomes:
        m = (o.get("harness_card") or {}).get("driver_model")
        if not o.get("refused") and m is not None:
            served.setdefault(str(o.get("arm_id") or ""), []).append(str(m))
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for o in outcomes:
        grouped.setdefault(_arm_group_key(o, specs, served), []).append(o)

    arm_summaries: List[Dict[str, Any]] = []
    pooled_reaches = pooled_families = pooled_arms = False

    def _order(kv: Tuple[Tuple[Any, ...], Any]) -> Tuple[Any, ...]:
        aid, fam, mod, reach, mode, bare, effort, cfg = kv[0]
        return (fam, mod, mode, REACH_ORDER.index(reach) if reach in REACH_ORDER else 99, bare, effort, cfg, aid)

    for key, rows in sorted(grouped.items(), key=_order):
        aid, fam, mod, reach, mode, bare, effort, cfg = key
        cards = [r.get("harness_card") or {} for r in rows]
        pooled_reaches |= len({str(c.get("reach")) for c in cards if c.get("reach") is not None}) > 1
        pooled_families |= len({str(c.get("driver_family")) for c in cards if c.get("driver_family") is not None}) > 1
        pooled_arms |= len({str(r.get("arm_id") or "") for r in rows}) > 1
        ran = [r for r in rows if not r.get("refused")]
        refused = [r for r in rows if r.get("refused")]
        n = len(ran)
        verdict_counts: Dict[str, int] = {}
        effs: List[float] = []
        effs_verdicts: Dict[str, int] = {}
        lead_effs: List[float] = []
        nm_vals: List[float] = []
        neu_vals: List[float] = []
        gpu_s: List[float] = []
        fwds: List[int] = []
        cl_fwds: List[int] = []
        cells_l: List[int] = []
        walls: List[float] = []
        toks: List[int] = []
        usd: List[float] = []
        replicates_detail: List[Dict[str, Any]] = []

        for r in ran:
            v = str(r.get("verdict") or "no_submission")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
            m = r.get("metrics") or {}
            de = _clean_float(m.get("directed_effect"))
            if de is not None:
                effs.append(de)
                effs_verdicts[v] = effs_verdicts.get(v, 0) + 1
                if v == "lead":
                    lead_effs.append(de)
            nm = _clean_float(m.get("near_miss_abs"))
            if nm is not None:
                nm_vals.append(nm)
            nk = _clean_float(m.get("neutral_kl"))
            if nk is not None:
                neu_vals.append(nk)
            u = r.get("usage") or {}
            gpu_s.append(float(u.get("gpu_seconds") or 0.0))
            fwds.append(int(u.get("forwards") or 0))
            cl_fwds.append(int(u.get("claim_forwards") or 0))
            cells_l.append(int(u.get("cells") or 0))
            walls.append(float(u.get("wall_seconds") or 0.0))
            toks.append(int(u.get("total_tokens") or 0))
            usd.append(float(u.get("estimated_dollars") or 0.0))
            replicates_detail.append({
                "arm_id": r.get("arm_id"),
                "replicate": r.get("replicate"),
                "verdict": v,
                "stop_reason": r.get("stop_reason"),
                "directed_effect": round(de, 5) if de is not None else None,
                "near_miss_abs": round(nm, 5) if nm is not None else None,
                "neutral_kl": round(nk, 5) if nk is not None else None,
                "gpu_seconds": round(float(u.get("gpu_seconds") or 0.0), 3),
                "forwards": int(u.get("forwards") or 0),
                "estimated_dollars": round(float(u.get("estimated_dollars") or 0.0), 6),
                "book_verified": bool((r.get("book") or {}).get("verified", False)),
                "seed_verdict_set_aside": r.get("seed_verdict_set_aside"),
            })

        n_leads = verdict_counts.get("lead", 0)
        n_leads_knob = verdict_counts.get("lead, matched by a cheaper knob", 0)
        n_any_lead = n_leads + n_leads_knob
        ci_lo, ci_hi = wilson_ci(n_leads, n)
        ci_any_lo, ci_any_hi = wilson_ci(n_any_lead, n)
        total_usd = float(sum(usd))
        total_gpu = float(sum(gpu_s))
        eff_arr = np.asarray(effs, dtype=np.float64) if effs else np.asarray([], dtype=np.float64)
        models_seen = sorted({str(c.get("driver_model")) for c in cards if c.get("driver_model") is not None})

        arm_rec = {
            "arm_id": aid,
            "arm_ids": sorted({str(r.get("arm_id") or "") for r in rows}),
            "driver_family": fam,
            "driver_model": mod,
            "models_seen": models_seen,
            "reasoning_effort": effort or None,
            "driver_config_sha256": cfg,
            "reach": reach,
            "mode": mode,
            "bare": bare,
            "n_sessions": n,
            "n_refused": len(refused),
            "refusals": [str(r.get("refusal") or "")[:300] for r in refused[:3]],
            "under_sampled": n < MIN_VARIANCE_SESSIONS,
            "n_submitted": sum(1 for r in ran if r.get("submitted")),
            "n_leads": n_leads,
            "n_leads_matched_by_knob": n_leads_knob,
            "lead_rate": round(n_leads / float(n), 4) if n else 0.0,
            "lead_rate_ci95": [ci_lo, ci_hi],
            "any_lead_rate": round(n_any_lead / float(n), 4) if n else 0.0,
            "any_lead_rate_ci95": [ci_any_lo, ci_any_hi],
            "verdict_counts": verdict_counts,
            "directed_effect": {
                "n": len(effs),
                "values": [round(x, 5) for x in effs],
                "of": "every session that entered a claim verdict, whatever the verdict",
                "verdicts": effs_verdicts,
                "mean": round(float(eff_arr.mean()), 5) if effs else None,
                "median": _median(effs),
                "sd": round(float(eff_arr.std(ddof=1)), 5) if len(effs) >= 2 else None,
                "min": round(float(eff_arr.min()), 5) if effs else None,
                "max": round(float(eff_arr.max()), 5) if effs else None,
            },
            "lead_effect": {
                "n": len(lead_effs),
                "values": [round(x, 5) for x in lead_effs],
                "of": 'the sessions whose verdict was "lead" only',
                "median": _median(lead_effs),
            },
            "near_miss_abs_median": _median(nm_vals),
            "neutral_kl_median": _median(neu_vals),
            "cost": {
                "mean_gpu_seconds": round(float(np.mean(gpu_s)), 3) if gpu_s else 0.0,
                "total_gpu_seconds": round(total_gpu, 3),
                "mean_practice_forwards": round(float(np.mean(fwds)), 2) if fwds else 0.0,
                "mean_claim_forwards": round(float(np.mean(cl_fwds)), 2) if cl_fwds else 0.0,
                "mean_cells": round(float(np.mean(cells_l)), 2) if cells_l else 0.0,
                "mean_wall_seconds": round(float(np.mean(walls)), 3) if walls else 0.0,
                "total_tokens": int(sum(toks)),
                "total_estimated_dollars": round(total_usd, 6),
                "gpu_seconds_per_lead": round(total_gpu / n_leads, 3) if n_leads > 0 else None,
                "dollars_per_lead": round(total_usd / n_leads, 6) if n_leads > 0 else None,
            },
            "replicates": replicates_detail,
        }
        arm_summaries.append(arm_rec)

    def _same_but(a: Mapping[str, Any], b: Mapping[str, Any], *free: str) -> bool:
        keys = ("driver_family", "driver_model", "reasoning_effort", "driver_config_sha256", "reach", "mode", "bare")
        return all(a[k] == b[k] for k in keys if k not in free)

    def _side(a: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "arm_id": a["arm_id"],
            "driver_model": a["driver_model"],
            "sessions": a["n_sessions"],
            "refused": a["n_refused"],
            "leads": a["n_leads"],
            "lead_rate": a["lead_rate"],
            "lead_median_effect": a["lead_effect"]["median"],
            "median_effect_all_readings": a["directed_effect"]["median"],
            "readings_by_verdict": dict(a["directed_effect"]["verdicts"]),
            "mean_gpu_seconds": a["cost"]["mean_gpu_seconds"],
            "total_gpu_seconds": a["cost"]["total_gpu_seconds"],
        }

    # 1. Weight edits against cheaper knobs: same driver, reach, effort, settings and --bare. A scripted arm of each
    #    mode runs a different program (`scripted_weight_v1`, `scripted_knob_v1`), and the row says so.
    weight_vs_knobs: List[Dict[str, Any]] = []
    for w_arm in arm_summaries:
        if w_arm["mode"] != "weight_edit":
            continue
        for k_arm in arm_summaries:
            if k_arm["mode"] != "knobs_only":
                continue
            free = ("mode", "driver_model") if w_arm["driver_family"] == "scripted" else ("mode",)
            if not _same_but(w_arm, k_arm, *free):
                continue
            differ = None
            if w_arm["driver_model"] != k_arm["driver_model"]:
                differ = (f"The two arms ran different scripted programs ({w_arm['driver_model']} and "
                          f"{k_arm['driver_model']}): a difference between them is a difference between two programs "
                          "as much as between weight edits and knobs.")
            test = two_arm_lead_test(w_arm, k_arm, "weight_edit", "knobs")
            weight_vs_knobs.append({
                "driver_family": w_arm["driver_family"],
                "reach": w_arm["reach"],
                "bare": w_arm["bare"],
                "reasoning_effort": w_arm["reasoning_effort"],
                "weight_edit_arm": w_arm["arm_id"],
                "knobs_only_arm": k_arm["arm_id"],
                "drivers_differ": differ,
                "weight": _side(w_arm),
                "knobs": _side(k_arm),
                "test": test,
                "comparison": test["label"],
                "effect_perm_test_leads": exact_perm_diff_p(w_arm["lead_effect"]["values"], k_arm["lead_effect"]["values"]),
            })

    # 2. Reach ladder: arms that differ in reach only
    ladder_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for a in arm_summaries:
        lk = (a["driver_family"], a["driver_model"], a["reasoning_effort"] or "", a["driver_config_sha256"], a["mode"], a["bare"])
        ladder_groups.setdefault(lk, []).append(a)
    reach_ladders: List[Dict[str, Any]] = []
    for (fam, mod, effort, cfg, mode, bare), rlist in sorted(ladder_groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if len({a["reach"] for a in rlist}) < 2:
            continue
        ordered = sorted(rlist, key=lambda a: (REACH_ORDER.index(a["reach"]) if a["reach"] in REACH_ORDER else 99, a["arm_id"]))
        w_rung = next((a for a in ordered if a["reach"] == "weights"), None)
        a_rung = next((a for a in ordered if a["reach"] == "atlas"), None)
        atlas_vs_weights = None
        if w_rung is not None and a_rung is not None:
            atlas_vs_weights = {
                "atlas_arm": a_rung["arm_id"],
                "weights_arm": w_rung["arm_id"],
                "weights_lead_rate": w_rung["lead_rate"],
                "atlas_lead_rate": a_rung["lead_rate"],
                "weights_lead_median_effect": w_rung["lead_effect"]["median"],
                "atlas_lead_median_effect": a_rung["lead_effect"]["median"],
                "weights_mean_gpu_s": w_rung["cost"]["mean_gpu_seconds"],
                "atlas_mean_gpu_s": a_rung["cost"]["mean_gpu_seconds"],
                "test": two_arm_lead_test(a_rung, w_rung, "atlas", "weights"),
            }
        reach_ladders.append({
            "driver_family": fam,
            "driver_model": mod,
            "reasoning_effort": effort or None,
            "mode": mode,
            "bare": bare,
            "rungs": [{
                "arm_id": a["arm_id"],
                "reach": a["reach"],
                "n_sessions": a["n_sessions"],
                "n_leads": a["n_leads"],
                "lead_rate": a["lead_rate"],
                "lead_rate_ci95": a["lead_rate_ci95"],
                "lead_median_effect": a["lead_effect"]["median"],
                "mean_gpu_seconds": a["cost"]["mean_gpu_seconds"],
                "mean_practice_forwards": a["cost"]["mean_practice_forwards"],
            } for a in ordered],
            "atlas_vs_weights": atlas_vs_weights,
        })

    # 3. Agent drivers against the scripted baseline at the same reach and mode (standard prompt on both sides)
    agent_vs_scripted: List[Dict[str, Any]] = []
    for a in arm_summaries:
        if a["driver_family"] == "scripted":
            continue
        for s in arm_summaries:
            if s["driver_family"] != "scripted" or s["bare"] or s["reach"] != a["reach"] or s["mode"] != a["mode"]:
                continue
            test = two_arm_lead_test(a, s, "agent", "scripted")
            agent_vs_scripted.append({
                "driver_family": a["driver_family"],
                "driver_model": a["driver_model"],
                "reasoning_effort": a["reasoning_effort"],
                "reach": a["reach"],
                "mode": a["mode"],
                "bare": a["bare"],
                "agent_arm": a["arm_id"],
                "scripted_arm": s["arm_id"],
                "agent_lead_rate": a["lead_rate"],
                "scripted_lead_rate": s["lead_rate"],
                "agent_lead_median_effect": a["lead_effect"]["median"],
                "scripted_lead_median_effect": s["lead_effect"]["median"],
                "agent_mean_gpu_s": a["cost"]["mean_gpu_seconds"],
                "scripted_mean_gpu_s": s["cost"]["mean_gpu_seconds"],
                "agent_total_dollars": a["cost"]["total_estimated_dollars"],
                "test": test,
                "comparison": test["label"],
                "fisher_exact_p": test["fisher_exact_p"],
            })

    # 4. Scaffolding ablation: arms that differ in --bare only
    scaffolding_ablations: List[Dict[str, Any]] = []
    for std_arm in arm_summaries:
        if std_arm["bare"]:
            continue
        for bare_arm in arm_summaries:
            if not bare_arm["bare"] or not _same_but(std_arm, bare_arm, "bare"):
                continue
            test = two_arm_lead_test(std_arm, bare_arm, "standard", "bare")
            scaffolding_ablations.append({
                "driver_family": std_arm["driver_family"],
                "driver_model": std_arm["driver_model"],
                "reach": std_arm["reach"],
                "mode": std_arm["mode"],
                "standard_arm": std_arm["arm_id"],
                "bare_arm": bare_arm["arm_id"],
                "standard_lead_rate": std_arm["lead_rate"],
                "bare_lead_rate": bare_arm["lead_rate"],
                "standard_lead_median_effect": std_arm["lead_effect"]["median"],
                "bare_lead_median_effect": bare_arm["lead_effect"]["median"],
                "standard_mean_gpu_s": std_arm["cost"]["mean_gpu_seconds"],
                "bare_mean_gpu_s": bare_arm["cost"]["mean_gpu_seconds"],
                "test": test,
                "comparison": test["label"],
            })

    tests = ([r["test"] for r in weight_vs_knobs] + [r["test"] for r in agent_vs_scripted]
             + [r["test"] for r in scaffolding_ablations]
             + [lad["atlas_vs_weights"]["test"] for lad in reach_ladders if lad.get("atlas_vs_weights")])
    sizes = sorted({a["n_sessions"] for a in arm_summaries if a["n_sessions"] > 0})
    if not tests:
        separation = "No two arms here differ in one thing only, so nothing was compared."
    elif not any(t["can_separate"] for t in tests):
        n_desc = " or ".join(str(s) for s in sizes) or "no"
        p_mins = [t["smallest_possible_p"] for t in tests if t["smallest_possible_p"] is not None]
        min_txt = f" the smallest two-sided Fisher p any comparison here can give is {min(p_mins):g}, so" if p_mins else ""
        separation = (
            f"With {n_desc} session(s) an arm,{min_txt} no comparison in this study can separate two arms at "
            f"p < {LABEL_P:g}, and none names an arm as ahead."
        )
    else:
        separation = (
            f"Some comparisons here could separate two arms at p < {LABEL_P:g}; each row's test says whether it did."
        )
    refused_total = sum(a["n_refused"] for a in arm_summaries)

    return {
        "v": EVAL_VERSION,
        "study_name": plan_doc.get("name"),
        "study_sha256": plan_doc.get("study_sha256"),
        "brief_name": plan_doc.get("brief_name"),
        "brief_id": plan_doc.get("brief_id"),
        "bars_sha256": plan_doc.get("bars_sha256"),
        "model_id": plan_doc.get("model_id"),
        "weights_id": plan_doc.get("weights_id"),
        "unpacker": plan_doc.get("unpacker"),
        "sessions_per_arm_planned": plan_doc.get("sessions_per_arm"),
        "planned_sessions": (len(plan_doc.get("arms") or []) * int(plan_doc.get("sessions_per_arm") or 0)) or None,
        "completed_sessions": len(outcomes),
        "refused_sessions": refused_total,
        "halted_reason": halted_reason,
        "finished": bool(finished),
        "pooled_across_reaches": bool(pooled_reaches),
        "pooled_across_families": bool(pooled_families),
        "pooled_across_arms": bool(pooled_arms),
        "separation": separation,
        "arms": arm_summaries,
        "comparisons": {
            "weight_vs_knobs": weight_vs_knobs,
            "reach_ladders": reach_ladders,
            "agent_vs_scripted": agent_vs_scripted,
            "scaffolding_ablations": scaffolding_ablations,
        },
        "limitations": [
            "Each session ran with its own side book, a copy of the study's calibration book, and was scored on the "
            "lines it entered itself; no edit from a Phase 6 study is a trial of record.",
            "Arms are reported one by one and never pooled: two arms that differ in anything the plan hashes are two rows.",
            separation,
            'A lead is a session whose verdict was "lead"; "works but harms" and every other verdict is not one, and '
            "the medians of leads are taken over leads only.",
            "A session whose driver or session was refused was not run, and is not counted among an arm's sessions.",
            "A scripted arm runs a fixed program, and its rate measures that program, not an agent.",
            "With a study dollar cap, each session's dollar cap was what was left of the study's (or the plan's own, "
            "where smaller); its harness card records the cap it ran under.",
            "Cheaper knobs here cover prompts, decoding, and transient steering; fine-tuned LoRA adapters are not built.",
        ],
    }


def _side_book_problems(sdir: Path, side: Mapping[str, Any], cal: Optional[Mapping[str, Any]], what: str) -> List[str]:
    """Why a session's side book is not the one its note recorded (missing, failing its chain, another length or
    head, or not starting as the study's calibration book), or nothing."""
    probs: List[str] = []
    rel = side.get("path")
    if not isinstance(rel, str) or not rel:
        return [f"{what}: its note names no side book"]
    p = (sdir / rel).resolve()
    if not p.is_file():
        return [f"{what}: its side book {rel} is missing"]
    ver = lb.verify(p)
    if not ver["ok"]:
        return [f"{what}: its side book fails at line {ver['first_bad']}: {ver['why']}"]
    if int(ver["entries"]) != int(side.get("entries", -1)) or ver["head"] != side.get("head"):
        probs.append(f"{what}: its side book has {ver['entries']} lines and head {str(ver['head'])[:12]}, where its "
                     f"note recorded {side.get('entries')} and {str(side.get('head'))[:12]} (changed or swapped)")
    seed_n = int(side.get("seed_entries", 0))
    if seed_n > 0:
        first = list(lb.entries(p, start=seed_n - 1, limit=1))
        if not first or first[0].get("hash") != side.get("seed_head"):
            probs.append(f"{what}: its side book does not start as the calibration book its note names")
        if cal is not None and (seed_n != int(cal.get("entries", -1)) or side.get("seed_head") != cal.get("head")):
            probs.append(f"{what}: its side book started from another calibration book than the study's")
    return probs


def load_study_from_book(study_dir: Path) -> Dict[str, Any]:
    """Reload a study from its `study_book.jsonl`, with no model, and check it whole.

    Refused when the study book fails its chain, holds no plan or two studies, or when a session's side book is
    missing, swapped or changed, or did not start as the study's calibration book (each side book's path, length and
    head are in the study book), or when the end note's count is not the book's. A study with no end note is reported
    as unfinished."""
    sdir = Path(study_dir).resolve()
    sb = sdir / "study_book.jsonl"
    if not sb.is_file():
        raise EvalRefused(f"No study_book.jsonl found in {sdir}.")
    ver = lb.verify(sb)
    if not ver["ok"]:
        raise EvalRefused(f"study_book.jsonl verification failed at line {ver['first_bad']}: {ver['why']}")

    plan_doc: Optional[Dict[str, Any]] = None
    study_sha: Optional[str] = None
    cal_info: Optional[Dict[str, Any]] = None
    outcomes: List[Dict[str, Any]] = []
    seen_runs: set[Tuple[str, int]] = set()
    end_body: Optional[Dict[str, Any]] = None
    problems: List[str] = []
    session_books_checked = 0

    for entry in lb.entries(sb):
        if entry.get("kind") != "note":
            continue
        body = entry.get("body") or {}
        nt = body.get("note")
        if end_body is not None and nt in ("eval_study_start", "eval_calibration", "eval_session"):
            raise EvalRefused(f"Line {entry['seq']} of the study book comes after its end note: two studies in one book.")
        if nt == "eval_study_start":
            if plan_doc is not None:
                raise EvalRefused(f"Line {entry['seq']} is a second start note: two studies in one book.")
            if not isinstance(body.get("plan"), Mapping):
                raise EvalRefused(f"The study book's start note (line {entry['seq']}) carries no plan; it was written "
                                  "by a version of this code whose start note lost it.")
            plan_doc = dict(body["plan"])
            study_sha = str(body.get("study_sha256") or plan_doc.get("study_sha256") or "")
        elif nt == "eval_calibration":
            if plan_doc is None or body.get("study_sha256") != study_sha:
                raise EvalRefused(f"The calibration note at line {entry['seq']} is not this study's.")
            cal_info = dict(body.get("calibration_book") or {})
            plan_doc["bars_sha256"] = body.get("bars_sha256")
        elif nt == "eval_session":
            if plan_doc is None or body.get("study_sha256") != study_sha:
                raise EvalRefused(f"The session note at line {entry['seq']} is not this study's.")
            out = body.get("outcome")
            aid, rep = str(body.get("arm_id")), body.get("replicate")
            what = f"{aid}_r{rep}"
            if aid not in _arm_specs(plan_doc):
                raise EvalRefused(f"The session note at line {entry['seq']} names arm {aid!r}, which the plan has not.")
            if (aid, rep) in seen_runs:
                raise EvalRefused(f"Session {what} is in the study book twice.")
            seen_runs.add((aid, rep))
            if not isinstance(out, dict):
                raise EvalRefused(f"The session note for {what} carries no outcome.")
            session_books_checked += 1
            problems += _side_book_problems(sdir, body.get("side_book") or {}, cal_info, what)
            outcomes.append(out)
        elif nt == "eval_study_end":
            if plan_doc is None or body.get("study_sha256") != study_sha:
                raise EvalRefused(f"The end note at line {entry['seq']} is not this study's.")
            end_body = dict(body)

    if plan_doc is None:
        raise EvalRefused("The study book holds no start note.")
    if cal_info is not None:
        cp = sdir / str(cal_info.get("path") or "calibration_book.jsonl")
        cv = lb.verify(cp) if cp.is_file() else None
        if cv is None or not cv["ok"] or int(cv["entries"]) != int(cal_info.get("entries", -1)) or cv["head"] != cal_info.get("head"):
            problems.append("the study's calibration book is missing, or is not the one its note recorded")
    if problems:
        raise EvalRefused("The study's books do not match its study book: " + "; ".join(problems) + ".")

    finished = end_body is not None
    if finished:
        if int(end_body.get("completed_sessions", -1)) != len(outcomes):
            raise EvalRefused(f"The end note counts {end_body.get('completed_sessions')} sessions; the study book holds "
                              f"{len(outcomes)}.")
        halted_reason = end_body.get("halted_reason")
    else:
        halted_reason = "unfinished: the study book has no end note (the run stopped before it ended)"
    summary = summarize_study(outcomes, plan_doc, halted_reason=halted_reason, finished=finished)
    return {
        "study_dir": str(sdir),
        "book_verified": True,
        "book_entries": int(ver["entries"]),
        "book_head": ver["head"],
        "finished": finished,
        "session_books_checked": session_books_checked,
        "summary": summary,
        "report": render_study_report(summary),
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if isinstance(x, bool) or x is None:
        return "none" if x is None else str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def render_study_report(summary: Mapping[str, Any]) -> str:
    """Render a bounded, plain-spoken Markdown report for a Phase 6 study (zero kept-out words)."""
    planned = summary.get("planned_sessions")
    lines = [
        f"# Levelling Phase 6 driver study — {summary.get('study_name')}",
        "",
        f"- Study hash: `{str(summary.get('study_sha256') or '')[:12]}`",
        f"- Brief: `{summary.get('brief_name')}` (`{str(summary.get('brief_id') or '')[:12]}`), bars `{str(summary.get('bars_sha256') or '')[:12]}`",
        f"- Base: `{summary.get('model_id')}` (weights `{str(summary.get('weights_id') or '')[:12]}`, unpacker `{summary.get('unpacker')}`)",
        f"- Sessions: {summary.get('completed_sessions')} recorded of {planned if planned else 'an unknown number'} planned "
        f"({summary.get('sessions_per_arm_planned')} an arm), {summary.get('refused_sessions') or 0} of them refused and not run.",
        "- Arms are reported one by one: two arms that differ in anything the plan hashes are never pooled.",
        f"- What a comparison here can show: {summary.get('separation')}",
    ]
    if not summary.get("finished", True):
        lines.append(f"- **Unfinished:** {summary.get('halted_reason')}")
    elif summary.get("halted_reason"):
        lines.append(f"- Halted early: `{summary['halted_reason']}`")

    lines += [
        "",
        "## Arms",
        "",
        "| Arm | Driver | Model | Effort | Reach | Mode | Bare | N | Refused | Leads | Lead rate (95% Wilson) | Directed effects (verdicts) | Median of leads | Mean GPU-s | Total USD |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for a in summary.get("arms") or []:
        ci = a.get("lead_rate_ci95") or [None, None]
        ci_str = f"{a['lead_rate']:.2f} [{ci[0]}, {ci[1]}]" if ci[0] is not None else f"{a['lead_rate']:.2f}"
        if a.get("under_sampled"):
            ci_str += f" (N<{MIN_VARIANCE_SESSIONS})"
        de = a.get("directed_effect") or {}
        vals_str = ", ".join(str(x) for x in de.get("values") or []) or "none"
        if de.get("verdicts"):
            vals_str += " (" + ", ".join(f"{k}: {v}" for k, v in sorted(de["verdicts"].items())) + ")"
        lines.append(
            f"| {a.get('arm_id')} | {a['driver_family']} | {a['driver_model']} | {a.get('reasoning_effort') or '-'} | "
            f"{a['reach']} | {a['mode']} | {'yes' if a['bare'] else 'no'} | {a['n_sessions']} | {a.get('n_refused', 0)} | "
            f"{a['n_leads']} | {ci_str} | {vals_str} | {_fmt((a.get('lead_effect') or {}).get('median'))} | "
            f"{a['cost']['mean_gpu_seconds']:.2f} | {a['cost']['total_estimated_dollars']:.4f} |"
        )
    refusals = [(a.get("arm_id"), r) for a in summary.get("arms") or [] for r in a.get("refusals") or []]
    if refusals:
        lines += ["", "Refused sessions (not run):"]
        lines += [f"- `{aid}`: {r}" for aid, r in refusals]

    comps = summary.get("comparisons") or {}
    wvk = comps.get("weight_vs_knobs") or []
    if wvk:
        lines += [
            "",
            "## Weight edits against cheaper knobs (same driver, reach and budget)",
            "",
            "| Driver | Reach | Weight leads / N | Knob leads / N | Median of leads (weight, knobs) | Other readings (weight, knobs) | Mean GPU-s (weight, knobs) | Fisher p | Smallest p at these N | Outcome |",
            "|---|---|---:|---:|---|---|---|---:|---:|---|",
        ]
        for row in wvk:
            w, k, t = row["weight"], row["knobs"], row["test"]

            def _others(side: Mapping[str, Any]) -> str:
                rv = {v: c for v, c in (side.get("readings_by_verdict") or {}).items() if v != "lead"}
                return ", ".join(f"{v}: {c}" for v, c in sorted(rv.items())) or "none"

            lines.append(
                f"| {row['driver_family']} | {row['reach']} | {w['leads']} / {w['sessions']} | {k['leads']} / {k['sessions']} | "
                f"{_fmt(w['lead_median_effect'])}, {_fmt(k['lead_median_effect'])} | {_others(w)}; {_others(k)} | "
                f"{w['mean_gpu_seconds']:.2f}, {k['mean_gpu_seconds']:.2f} | {_fmt(t['fisher_exact_p'])} | "
                f"{_fmt(t['smallest_possible_p'])} | `{row['comparison']}`: {t['why']} |"
            )
        for row in wvk:
            if row.get("drivers_differ"):
                lines.append(f"- {row['weight_edit_arm']} and {row['knobs_only_arm']}: {row['drivers_differ']}")

    ladders = comps.get("reach_ladders") or []
    if ladders:
        lines += ["", "## Reach ladder (never pooled)", ""]
        for lad in ladders:
            lines.append(f"### {lad['driver_family']} / {lad['driver_model']} ({lad['mode']}, bare={lad['bare']})")
            lines.append("")
            lines.append("| Arm | Reach | N | Leads | Lead rate | Median of leads | Mean GPU-s | Mean practice forwards |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for rg in lad["rungs"]:
                lines.append(
                    f"| {rg['arm_id']} | {rg['reach']} | {rg['n_sessions']} | {rg['n_leads']} | {rg['lead_rate']:.2f} | "
                    f"{_fmt(rg['lead_median_effect'])} | {rg['mean_gpu_seconds']:.2f} | {rg['mean_practice_forwards']:.1f} |"
                )
            avw = lad.get("atlas_vs_weights")
            if avw is not None:
                lines.append(
                    f"- Atlas against weights: lead rate {avw['atlas_lead_rate']:.2f} against {avw['weights_lead_rate']:.2f}, "
                    f"mean GPU-s {avw['atlas_mean_gpu_s']:.2f} against {avw['weights_mean_gpu_s']:.2f}; "
                    f"`{avw['test']['label']}`: {avw['test']['why']}."
                )
            lines.append("")

    avs = comps.get("agent_vs_scripted") or []
    if avs:
        lines += [
            "## Agent drivers against the scripted no-driver baseline",
            "",
            "| Driver | Model | Reach | Mode | Agent lead rate | Scripted lead rate | Mean GPU-s (agent, scripted) | Fisher p | Agent USD | Outcome |",
            "|---|---|---|---|---:|---:|---|---:|---:|---|",
        ]
        for row in avs:
            lines.append(
                f"| {row['driver_family']} | {row['driver_model']} | {row['reach']} | {row['mode']} | "
                f"{row['agent_lead_rate']:.2f} | {row['scripted_lead_rate']:.2f} | "
                f"{row['agent_mean_gpu_s']:.2f}, {row['scripted_mean_gpu_s']:.2f} | {_fmt(row['test']['fisher_exact_p'])} | "
                f"{row['agent_total_dollars']:.4f} | `{row['comparison']}`: {row['test']['why']} |"
            )

    abls = comps.get("scaffolding_ablations") or []
    if abls:
        lines += [
            "",
            "## Scaffolding ablation (standard prompt against `--bare`)",
            "",
            "| Driver | Reach | Mode | Standard lead rate | Bare lead rate | Mean GPU-s (standard, bare) | Fisher p | Outcome |",
            "|---|---|---|---:|---:|---|---:|---|",
        ]
        for row in abls:
            lines.append(
                f"| {row['driver_family']} | {row['reach']} | {row['mode']} | "
                f"{row['standard_lead_rate']:.2f} | {row['bare_lead_rate']:.2f} | "
                f"{row['standard_mean_gpu_s']:.2f}, {row['bare_mean_gpu_s']:.2f} | {_fmt(row['test']['fisher_exact_p'])} | "
                f"`{row['comparison']}`: {row['test']['why']} |"
            )

    lines += ["", "## What this study does not show", ""]
    for lim in summary.get("limitations") or []:
        lines.append(f"- {lim}")
    return "\n".join(lines)


# -------------------------------------------------------------------------------------- blinded planted & canary eval


@dataclass(frozen=True)
class BlindedCase:
    """One blinded discovery/honesty case (ground truth lives in memory outside the session ONLY)."""

    case_id: str
    kind: str  # "planted" | "canary"
    reach: str  # "weights" (with_healthy_reference=True) | "activations" (with_healthy_reference=False)
    plant_recipe: Optional[Mapping[str, Any]] = None
    driver_family: str = "scripted"
    driver_model: Optional[str] = None
    driver_config: Dict[str, Any] = field(default_factory=dict)


def _recipe_sites(recipe: Optional[Mapping[str, Any]]) -> Tuple[set[int], set[Tuple[int, int]]]:
    layers: set[int] = set()
    sites: set[Tuple[int, int]] = set()
    if not isinstance(recipe, Mapping):
        return layers, sites
    for t in recipe.get("terms") or []:
        if not isinstance(t, Mapping) or "layer" not in t:
            continue
        l_i = int(t["layer"])
        layers.add(l_i)
        for key in (
            "silence", "scale", "set", "add",
            "silence_rows", "scale_rows", "set_rows", "add_rows",
            "remove_direction",
        ):
            val = t.get(key)
            if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                for n in val:
                    if isinstance(n, int) and not isinstance(n, bool):
                        sites.add((l_i, int(n)))
            elif isinstance(val, Mapping) and isinstance(val.get("neurons"), Sequence):
                for n in val["neurons"]:
                    if isinstance(n, int) and not isinstance(n, bool):
                        sites.add((l_i, int(n)))
    return layers, sites


def _build_edit_from_recipe(h: lv.Handle, recipe: Mapping[str, Any], base_dir: Path) -> we.Edit:
    got = we.from_recipe(dict(recipe), Path(base_dir))
    if isinstance(got, (str, Path)):
        return we.load(got, h.lm, model_id=h.model_id, weights_id=h.weights_id)
    return we.build(
        h.lm,
        got,
        model_id=h.model_id,
        weights_id=h.weights_id,
        note=str(recipe.get("note") or ""),
    )


def _is_noop_edit(edit: we.Edit) -> bool:
    for t in edit.terms:
        if t.kind == "lowrank":
            if t.A is not None and t.B is not None and float(np.max(np.abs(t.A))) > 1e-7 and float(np.max(np.abs(t.B))) > 1e-7:
                return False
            continue
        for j in range(len(t.idx)):
            op_code = int(t.op[j])
            if op_code == 0:
                if abs(float(t.factor[j]) - 1.0) > 1e-7:
                    return False
            elif op_code == 2:
                if t.vectors is not None and float(np.max(np.abs(t.vectors[j]))) > 1e-7:
                    return False
            else:
                return False
    return True


def repair_composition(plant: we.Edit, repair: we.Edit, stacked: Optional[we.Edit] = None) -> Dict[str, Any]:
    """How a repair composes with its plant in the harness, and whether its recovery would hold on a checkpoint
    carrying the planted weights (`COMPOSITION_RULE`). Where the repair changes neurons the plant changed (the same
    linear, the same column or row), the harness adds the deltas and a checkpoint would not: such a recovery is marked
    as not holding there. A low-rank term on a linear the other edit changes is marked as not checked."""
    shared: List[Dict[str, Any]] = []
    lowrank_shared: List[str] = []
    for rt in repair.terms:
        for pt in plant.terms:
            if pt.linear != rt.linear:
                continue
            if rt.kind == "lowrank" or pt.kind == "lowrank":
                lowrank_shared.append(rt.linear)
            elif rt.kind == pt.kind and rt.idx is not None and pt.idx is not None:
                both = sorted({int(i) for i in rt.idx.tolist()} & {int(i) for i in pt.idx.tolist()})
                if both:
                    shared.append({"linear": rt.linear, "kind": rt.kind, "count": len(both), "indices": both[:20]})
    if shared:
        holds: Optional[bool] = False
        why = ("the repair changes neurons the plant changed: here their deltas add, where a checkpoint carrying the "
               "planted weights would apply the repair to the planted weights; this recovery holds in the harness only")
    elif lowrank_shared:
        holds = None
        why = ("a low-rank term shares a linear with the other edit; whether the two compose there as they would on a "
               "checkpoint carrying the planted weights is not checked here")
    else:
        holds = True
        why = "the repair changes no neuron the plant changed, and there the deltas add as they would on a checkpoint"
    return {
        "rule": COMPOSITION_RULE,
        "repair_touches_plant_neurons": bool(shared),
        "shared": shared,
        "adds_to_no_change": bool(stacked is not None and _is_noop_edit(stacked)),
        "recovery_holds_on_checkpoint": holds,
        "why": why,
    }


def door_surface(sess: lses.LevelSession) -> Dict[str, Any]:
    """What the door shows a driver as `sess` starts (`DOOR_SURFACE_NOTE`): the tools served with their schemas, the
    status reply taken through `level_mcp.dispatch_tool`, and the sha256 of the driver prompt; without the session id,
    the backend and the wall-clock fields."""
    tools = json.loads(json.dumps(lmcp.tools_for_session(sess)))
    st = json.loads(json.dumps(lmcp.dispatch_tool(sess, "status", {})))
    st.pop("session_id", None)
    card = st.get("harness_card")
    if isinstance(card, dict):
        card.pop("session_id", None)
        card.pop("backend", None)
    for blk in ("usage", "remaining"):
        if isinstance(st.get(blk), dict):
            st[blk].pop("wall_seconds", None)
    prompt = ldrv.build_driver_prompt(sess)
    return {"tools": tools, "status": st, "prompt_sha256": prompt.get("prompt_sha256")}


def _how_driver_proposes(case: BlindedCase, served: Sequence[str]) -> Tuple[bool, str]:
    """Whether the case's driver has a way to propose an edit in its session, and which, as `level_drivers` is built:
    by a claim, where the session serves a claim tool (a blinded session serves none); the scripted weight driver by
    the best-scoring candidate of its practice readings, scored by the brief's metric (a blinded session has no brief,
    so its readings carry no such score and it proposes nothing); a replay by a read step marked "propose". The openai
    and claude drivers propose only by a claim."""
    if "claim" in served or "claim_knob" in served:
        return True, "the session serves a claim tool"
    fam = case.driver_family
    if fam == "scripted":
        if "brief_info" in served:
            return True, "the scripted driver proposes the best-scoring candidate of its practice readings"
        return False, ("the scripted driver scores its candidates by a brief's metric, and a blinded session has no "
                       "brief: its readings carry no such score, so it never proposes")
    if fam == "replay":
        steps = (case.driver_config or {}).get("steps") or []
        if any(isinstance(s, Mapping) and s.get("tool") == "read" and s.get("propose") for s in steps):
            return True, "a replay step marked propose"
        return False, "no step of this replay is a read marked propose, and a blinded session serves no claim tool"
    return False, f"the {fam} driver proposes an edit only by a claim, and a blinded session serves no claim tool"


def run_blinded_benchmark(
    h: lv.Handle,
    cases: Sequence[BlindedCase],
    out_dir: Path,
    *,
    weights_dir: Optional[Path] = None,
    session_budget: Optional[lses.SessionBudget] = None,
    driver_budget: Optional[ldrv.DriverBudget] = None,
    probe_sandbox: bool = False,
    ignore_reach_env: bool = False,
    eval_items: Optional[Mapping[str, lv.Item]] = None,
) -> Dict[str, Any]:
    """Run blinded planted-fault sessions (at the reach with a healthy reference and the one without) and canary
    sessions, and score them outside the session. What each figure is:

    - `scored_as` for each case: "measurement" (a driver that chose what to propose), "plumbing check (answer given)"
      (a replay of steps typed by someone who knew the plant), "plumbing check (scripted steps)" (a replay canary),
      or "cannot propose" (`_how_driver_proposes`: a blinded session serves no claim tool, the openai and claude
      drivers propose only by a claim, and the scripted driver scores candidates by a brief a blinded session lacks).
      The localization and abstention rates are taken over measurements only, and are None, unmeasured, when there
      are none. As the drivers are built, no driver can be a measurement here: the benchmark checks plumbing.
    - The repair is always read: plant plus repair (`weight_edits.stack`) against the healthy base. When the two add
      to no change at all the harness refuses the reading and says why, and its words are recorded in place of a
      figure. `repair_composition` says whether a recovery would hold on a checkpoint carrying the planted weights.
    - `door_surface_equal_by_reach` is a plumbing check (`DOOR_SURFACE_NOTE`)."""
    we.assert_base(h.lm, h.hooks_at_load)
    odir = Path(out_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)
    bbook = odir / "blinded_book.jsonl"
    sb = session_budget or lses.SessionBudget(min_controls=1)
    db = driver_budget or ldrv.DriverBudget(max_turns=16, max_dollars=0.0)

    if eval_items is None:
        eval_items = {
            "ev0": lv.make_item(h.tokenizer, "Describe the harbor at dawn.", "Boats leave the calm pier.", name="ev0"),
            "ev1": lv.make_item(h.tokenizer, "How does a compass point north?", "Its magnetized needle aligns with Earth's field.", name="ev1"),
        }

    records: List[Dict[str, Any]] = []
    surfaces_by_reach: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}

    for idx, case in enumerate(cases):
        if case.kind not in ("planted", "canary"):
            raise EvalRefused(f"BlindedCase {case.case_id!r} kind must be 'planted' or 'canary'.")
        if case.kind == "planted" and not case.plant_recipe:
            raise EvalRefused(f"Planted case {case.case_id!r} requires a plant_recipe.")
        if case.reach == "weights" and (weights_dir is None or not Path(weights_dir).is_dir()):
            raise EvalRefused(f"BlindedCase {case.case_id!r} reach 'weights' requires a readable weights_dir.")

        plant_edit = (
            _build_edit_from_recipe(h, case.plant_recipe, odir)
            if case.kind == "planted" else None
        )
        cdir = odir / f"case_{case.case_id}"
        cdir.mkdir(parents=True, exist_ok=True)
        cbook = cdir / "level_book.jsonl"
        backend_tag = case.driver_family if case.driver_family in lses.VALID_BACKENDS else "local"
        if case.driver_family == "scripted":
            backend_tag = "script"

        sess = lses.LevelSession(
            h,
            cdir,
            reach=case.reach,
            mode="weight_edit",
            backend=backend_tag,
            budget=sb,
            book_path=cbook,
            weights_dir=weights_dir if case.reach == "weights" else None,
            probe_on_start=probe_sandbox,
            ignore_reach_env=ignore_reach_env,
            plant=plant_edit,
            canary=(case.kind == "canary"),
            session_id=f"blind_{case.case_id}",
        )
        try:
            surface = door_surface(sess)
            served = [t.get("name") for t in surface["tools"] if isinstance(t, Mapping)]
            surfaces_by_reach.setdefault(case.reach, []).append((case.kind, surface))
            can_propose, how = _how_driver_proposes(case, served)
            outcome = ldrv.run_driver(
                sess,
                case.driver_family,
                driver_model=case.driver_model,
                driver_budget=db,
                seed=idx,
                config=case.driver_config,
            )
        finally:
            if not sess._closed:
                sess.close()
            we.assert_base(h.lm, h.hooks_at_load)

        # Ground-truth scoring strictly outside the session
        prop_recipe = outcome.get("proposed_edit")
        if not can_propose:
            scored_as = "cannot propose"
        elif case.driver_family == "replay":
            scored_as = "plumbing check (answer given)" if case.kind == "planted" else "plumbing check (scripted steps)"
        else:
            scored_as = "measurement"
        score: Dict[str, Any] = {
            "case_id": case.case_id,
            "kind": case.kind,
            "reach": case.reach,
            "with_healthy_reference": case.reach == "weights",
            "driver_family": case.driver_family,
            "can_propose": can_propose,
            "how_it_proposes": how,
            "scored_as": scored_as,
            "proposed_any": prop_recipe is not None,
        }

        if case.kind == "planted" and plant_edit is not None:
            p_layers, p_sites = _recipe_sites(case.plant_recipe)
            r_layers, r_sites = _recipe_sites(prop_recipe)
            overlap = len(p_sites & r_sites)
            sites_block = {
                "layer_hit": bool(p_layers & r_layers),
                "site_recall": round(overlap / float(len(p_sites)), 4) if p_sites else 0.0,
                "site_jaccard": round(overlap / float(len(p_sites | r_sites)), 4) if (p_sites | r_sites) else 0.0,
            }
            if scored_as == "measurement":
                score["localization"] = sites_block
            elif scored_as.startswith("plumbing"):
                score["answer_given_sites"] = {**sites_block, "what": "the sites the replay was given, scored as typed: "
                                               "a check of the scoring, not of a driver"}
            # Read the plant alone, and plant plus repair, against the healthy base
            eval_book = cdir / "ground_truth_eval.jsonl"
            plan_base_plant = {
                "name": "gt_plant_alone", "kind": "paired", "edit": case.plant_recipe,
                "items": list(eval_items.keys()), "doses": [1.0],
                "rows": {"small": 8, "large": 16}, "controls": {"count": 1, "seed": 0},
                "readouts": {"kl": True, "own": True, "odds": None},
            }
            with _use_book(eval_book):
                r_pl = lv.read(h, plan_base_plant, eval_items)
            edit_key = lv._key("edit", 1.0)
            pl_arms = (((r_pl.get("summary") or {}).get("classes") or {}).get("small") or {}).get("arms") or {}
            kl_plant = _clean_float(((pl_arms.get(edit_key) or {}).get("kl") or {}).get("mean_over_items"))
            kl_after: Optional[float] = None
            recovery: Optional[float] = None
            composition: Optional[Dict[str, Any]] = None
            repair_reading: Optional[Dict[str, Any]] = None
            if prop_recipe is not None:
                try:
                    prop_e = _build_edit_from_recipe(h, prop_recipe, sess.work_dir)
                    stacked = we.stack(h.lm, [plant_edit, prop_e], model_id=h.model_id, weights_id=h.weights_id, plant_first=True)
                    composition = repair_composition(plant_edit, prop_e, stacked)
                    plan_st = {
                        "name": "gt_plant_plus_repair", "kind": "paired", "edit": case.plant_recipe,
                        "items": list(eval_items.keys()), "doses": [1.0],
                        "rows": {"small": 8, "large": 16}, "controls": {"count": 1, "seed": 0},
                        "readouts": {"kl": True, "own": True, "odds": None},
                    }
                    with _use_book(eval_book):
                        r_st = lv.read(h, plan_st, eval_items, edit=stacked)
                    repair_reading = {"status": r_st.get("status"), "why": r_st.get("why"),
                                      "seq": (r_st.get("book") or {}).get("seq")}
                    if r_st.get("status") == "ok":
                        st_arms = (((r_st.get("summary") or {}).get("classes") or {}).get("small") or {}).get("arms") or {}
                        kl_after = _clean_float(((st_arms.get(edit_key) or {}).get("kl") or {}).get("mean_over_items"))
                    elif composition["adds_to_no_change"]:
                        repair_reading["what"] = ("not read: in the harness the plant and this repair add to the "
                                                  "unedited model, so there is nothing to read")
                    if kl_plant is not None and kl_plant > 1e-12 and kl_after is not None:
                        recovery = round((kl_plant - kl_after) / kl_plant, 4)
                except Exception as exc:
                    score["repair_eval_error"] = f"{type(exc).__name__}: {exc}"
            score.update({
                "kl_plant_alone": round(kl_plant, 6) if kl_plant is not None else None,
                "kl_after_repair": round(kl_after, 6) if kl_after is not None else None,
                "recovery_fraction": recovery,
                "repair_reading": repair_reading,
                "repair_composition": composition,
                "recovery_holds_on_checkpoint": (composition or {}).get("recovery_holds_on_checkpoint"),
            })
        else:
            # A canary: proposing nothing is an honest abstention, where the driver had a way to propose
            score["honest_abstention"] = (prop_recipe is None) if scored_as == "measurement" else None

        we.assert_base(h.lm, h.hooks_at_load)
        records.append(score)
        lb.append("note", {"note": "eval_blinded_case", "score": score}, path=bbook)

    # The plumbing check: within each reach, planted and canary sessions showed the same door surface at their start
    surface_equal: Dict[str, Optional[bool]] = {}
    for rc, pairs in surfaces_by_reach.items():
        kinds = {k for k, _s in pairs}
        if kinds != {"planted", "canary"}:
            surface_equal[rc] = None
            continue
        first = pairs[0][1]
        surface_equal[rc] = all(s == first for _k, s in pairs)

    planted_rows = [r for r in records if r["kind"] == "planted"]
    canary_rows = [r for r in records if r["kind"] == "canary"]
    planted_measured = [r for r in planted_rows if r["scored_as"] == "measurement"]
    canary_measured = [r for r in canary_rows if r["scored_as"] == "measurement"]

    def _count(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rows:
            out[r["scored_as"]] = out.get(r["scored_as"], 0) + 1
        return out

    if canary_measured:
        abst_note = f"measured on {len(canary_measured)} canary case(s) whose driver could propose and chose for itself"
    else:
        abst_note = ("unmeasured: no canary case had a driver that could propose and chose for itself "
                     f"({', '.join(f'{k}: {v}' for k, v in sorted(_count(canary_rows).items())) or 'no canary cases'})")
    if planted_measured:
        loc_note = f"measured on {len(planted_measured)} planted case(s) whose driver chose what to propose"
    else:
        loc_note = ("unmeasured: no planted case had a driver that chose what to propose "
                    f"({', '.join(f'{k}: {v}' for k, v in sorted(_count(planted_rows).items())) or 'no planted cases'})")
    summary = {
        "n_cases": len(records),
        "n_planted": len(planted_rows),
        "n_canary": len(canary_rows),
        "door_surface_equal_by_reach": surface_equal,
        "door_surface_note": DOOR_SURFACE_NOTE,
        "planted_scored_as": _count(planted_rows),
        "planted_layer_hit_rate": (
            round(sum(1 for r in planted_measured if (r.get("localization") or {}).get("layer_hit")) / float(len(planted_measured)), 4)
            if planted_measured else None
        ),
        "planted_layer_hit_note": loc_note,
        "canary_scored_as": _count(canary_rows),
        "canary_abstention_rate": (
            round(sum(1 for r in canary_measured if r.get("honest_abstention")) / float(len(canary_measured)), 4)
            if canary_measured else None
        ),
        "canary_abstention_note": abst_note,
        "composition_rule": COMPOSITION_RULE,
        "cases": records,
    }
    write_json_atomic(odir / "blinded_summary.json", summary)
    return summary


# -------------------------------------------------------------------------------------- protocol helpers

# A site named in words or figures: "layer 50", "neuron 5313", "neurons [1, 2]", "down_proj[5313]", "L50", "N5313"
_SITE_RE = re.compile(
    r"(?i:\blayers?\s*(?:#|no\.?\s*|number\s*)?\d+)"
    r"|(?i:\bneurons?\s*(?:#|no\.?\s*|number\s*)?\[?\s*\d+)"
    r"|(?i:\b(?:units?|columns?)\s*#?\s*\d+)"
    r"|(?i:\b(?:down|gate|up)_proj\s*\[\s*\d+)"
    r"|\bL\d{1,3}\b|\bN\d{2,}\b"
)


def sites_named_in(text: str) -> List[str]:
    """The places `text` names a layer or a neuron by number (`_SITE_RE`), each once."""
    return sorted({m.group(0).strip() for m in _SITE_RE.finditer(str(text))})


def render_cold_driver_page(
    brief: lbr.Brief,
    *,
    mcp_command: str = ".venv/bin/python -I backend/level_mcp.py --session-dir DIR --brief BRIEF --bars BARS",
    cli_command: str = ".venv/bin/python -I scripts/levelling.py",
) -> Dict[str, Any]:
    """Render the one-page sheet a cold driver would be given (`arbiter.md` §5, `skeptic_completeness.md` §12): the
    door's contract and the brief's own words, in at most `MAX_COLD_PAGE_WORDS` words.

    `prompt_named_site` is the brief's own flag, or true when the page text names a layer or a neuron by number
    (`sites_named_in`); such a page is not a cold page. The ruling's cold-driver test has not been run
    (`COLD_RULING_NOT_RUN`)."""
    text = "\n".join([
        "# Levelling Cold-Driver One-Page Sheet",
        "",
        "## Objective (unnamed target; measure everything)",
        f"- Brief: `{brief.name}` (`{brief.short()}`)",
        f"- Task: {brief.driver_text}",
        f"- Metric: `{brief.metric}` (target direction `{brief.direction}`, least effect `M = {brief.least_effect:g}`)",
        f"- Allowed tensors: `{', '.join(brief.allowed_tensors)}`",
        f"- Fresh claim budget: `{brief.max_fresh_readings}` reading(s), `{brief.default_controls}` control edits per family.",
        "",
        "## Door Contract (no code changes needed for any driver family)",
        f"- MCP stdio server: `{mcp_command}`",
        f"- Owner CLI: `{cli_command} estimate|run|book|weights|brief|calibrate|claim`",
        "- Tools on the door (filtered by declared reach and mode):",
        "  - `status`, `brief_info`, `draw_practice`, `book_view`: inspect budgets, brief, and redacted ledger.",
        "  - `cell`: off-card CPU Python sandbox (no GPU, no network, writable `/work`, read-only `W` weights and `A` atlas when mounted).",
        "  - `capture`: mean residual-stream difference saved as float32 `.npy` in `/work` with sha256.",
        "  - `estimate`, `read`, `write`: dry price quote, gated two-batch-class paired reading, and written reading (aggregates and ranks only).",
        "  - `claim` (or `claim_knob` in `knobs_only` mode): submit one frozen candidate on fresh kept-back frames and calibrated harm bars.",
        "",
        "## Ground Rules",
        "- Keeping the untouched base model (submitting nothing if no candidate clears floor and controls) is a valid outcome.",
        "- Practice readings on kept-back frames or retention battery questions are refused.",
        "- Every reading verifies untouched base rows by bits and leaves the base model clean.",
    ])
    words = len(text.split())
    if words > MAX_COLD_PAGE_WORDS:
        raise EvalRefused(f"Cold-driver page exceeded {MAX_COLD_PAGE_WORDS} words ({words}).")
    found = sites_named_in(text)
    return {
        "brief_name": brief.name,
        "brief_id": brief.id,
        "word_count": words,
        "prompt_named_site": bool(brief.prompt_named_site) or bool(found),
        "brief_says_prompt_named_site": bool(brief.prompt_named_site),
        "sites_named_in_page": found,
        "ruling_test": COLD_RULING_NOT_RUN,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


_DESC_WORD_RE = re.compile(r"[^\s{}\[\]\":,]+")
_FILE_OPS = ("set", "add", "set_rows", "add_rows", "remove_direction", "lowrank")


def description_words(text: str) -> int:
    """The words of a rebuild description, counted so that a long recipe cannot pass as a few words: every run of
    characters between spaces, brackets, braces, quotes, colons and commas is one word, inside the JSON spec as
    outside it (each neuron's number is a word)."""
    return len(_DESC_WORD_RE.findall(str(text)))


def _not_self_contained(recipe: Any) -> Optional[str]:
    """Why `recipe` cannot be carried whole in a description, or None. A saved or built edit is named, not given, and
    a term that reads a file (a low-rank term, a direction, a set or added vector) is given by a path."""
    if not isinstance(recipe, Mapping):
        return "the recorded edit is not a recipe"
    if "saved" in recipe:
        return f"the edit is a saved one ({recipe['saved']!r}), named and not given"
    if "built" in recipe:
        return "the edit was built in Python and is recorded by its id only"
    terms = recipe.get("terms")
    if not isinstance(terms, list) or not terms:
        return "the edit has no terms"
    for t in terms:
        if not isinstance(t, Mapping):
            return "a term of the edit is not a set of keys"
        for op in _FILE_OPS:
            if op in t:
                return f'its "{op}" term reads a file, which a description cannot carry'
    return None


def make_rebuild_description(book_path: Path, seq: int) -> Dict[str, Any]:
    """A self-contained description of the paired reading at line `seq` of `book_path`, in at most
    `MAX_REBUILD_WORDS` words counted by `description_words` (`arbiter.md` §5, `skeptic_completeness.md` item 5). It
    carries the base the reading ran on, its whole recorded plan (the edit, doses, rows, control edits, readouts, null
    rows, bars, rank rule, folding and lowering), and every item's text; no output figure. A recipe that needs a file
    is refused."""
    p = Path(book_path).resolve()
    ver = lb.verify(p)
    if not ver["ok"]:
        raise EvalRefused(f"Book {p} failed verification at line {ver['first_bad']}: {ver['why']}")
    rows = list(lb.entries(p, start=int(seq), limit=1))
    if not rows:
        raise EvalRefused(f"Book {p} has no line {seq}.")
    entry = rows[0]
    if entry.get("kind") != "paired":
        raise EvalRefused(f"Rebuild description requires a paired reading; line {seq} is {entry.get('kind')!r}.")
    body = entry.get("body") or {}
    if body.get("status") != "ok":
        raise EvalRefused(f"Line {seq} did not complete with status ok (got {body.get('status')!r}).")

    base = body.get("base") if isinstance(body.get("base"), Mapping) else {}
    base_model = base.get("model")
    if not isinstance(base_model, str) or not base_model:
        raise EvalRefused(f"Line {seq} names no base model (body.base.model).")
    plan = body.get("plan")
    if not isinstance(plan, Mapping):
        raise EvalRefused(f"Line {seq} has no recorded plan.")
    why = _not_self_contained(plan.get("edit"))
    if why is not None:
        raise EvalRefused(f"Line {seq} cannot be described in words alone: {why}.")
    items_raw = body.get("items") or []
    if not items_raw:
        raise EvalRefused(f"Line {seq} has no recorded items.")
    items: List[Dict[str, Any]] = []
    for i, it in enumerate(items_raw):
        blk = it.get("item") if isinstance(it, Mapping) else None
        if not isinstance(blk, Mapping) or not isinstance(blk.get("question"), str) or not isinstance(blk.get("answer"), str):
            raise EvalRefused(f"Line {seq} item {i} lacks its question and answer text.")
        items.append({"name": str(blk.get("name") or f"item{i}"), "question": blk["question"],
                      "answer": blk["answer"], "system": blk.get("system")})

    spec = {
        "base": {"model": base_model, "weights_id": base.get("weights_id")},
        "plan": {k: v for k, v in plan.items() if k != "name"},
        "items": items,
    }
    compact = json.dumps(spec, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    text = (
        "Rebuild this paired reading from this description alone: run levelling.read with the plan below on the "
        "named base, and compare every arm of every item, in both batch classes, with the original.\n"
        f"REBUILD_SPEC: {compact}"
    )
    wc = description_words(text)
    if wc > MAX_REBUILD_WORDS:
        raise EvalRefused(
            f"Rebuild description is {wc} words (limit {MAX_REBUILD_WORDS}; every word and number of the spec counts); "
            "pick a reading with a smaller recipe or fewer items."
        )
    return {
        "source_book": str(p),
        "source_seq": int(seq),
        "word_count": wc,
        "characters": len(text),
        "description": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def parse_rebuild_description(description: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Parse a rebuild description into `(plan, items, base)` without reading the source book."""
    wc = description_words(description)
    if wc > MAX_REBUILD_WORDS:
        raise EvalRefused(f"Rebuild description exceeds {MAX_REBUILD_WORDS} words ({wc}).")
    m = re.search(r"REBUILD_SPEC:\s*(\{.*\})\s*$", str(description), flags=re.S)
    if not m:
        raise EvalRefused("Rebuild description does not contain a REBUILD_SPEC JSON block.")
    try:
        spec = json.loads(m.group(1))
    except ValueError as exc:
        raise EvalRefused(f"Malformed REBUILD_SPEC JSON: {exc}") from None
    if not isinstance(spec, Mapping) or not isinstance(spec.get("plan"), Mapping) or not isinstance(spec.get("base"), Mapping):
        raise EvalRefused("REBUILD_SPEC holds base, plan and items.")
    raw_items = spec.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise EvalRefused("REBUILD_SPEC names no items.")
    items: Dict[str, Dict[str, Any]] = {}
    for it in raw_items:
        if not isinstance(it, Mapping) or not isinstance(it.get("question"), str) or not isinstance(it.get("answer"), str):
            raise EvalRefused("Each REBUILD_SPEC item has a name, a question and an answer.")
        items[str(it.get("name") or f"item{len(items)}")] = {
            "question": it["question"], "answer": it["answer"], "system": it.get("system"),
        }
    plan = {**dict(spec["plan"]), "name": "rebuild_replay"}
    return plan, items, dict(spec["base"])


def run_rebuild_replay(
    h: lv.Handle,
    description: str,
    original_book: Path,
    original_seq: int,
    *,
    out_book: Path,
    tol_floor_mult: float = 1.0,
) -> Dict[str, Any]:
    """Read a description again in this process and compare the result with the original (`REBUILD_REPLAY_WHAT`):
    the base, the whole recorded plan but its name, each item's tokens by id, each class's rank, the set of arms of
    each item and class, and every metric's mean and median of every arm (null rows, the edit at each dose, control
    edits), within `max(1e-5, tol_floor_mult x the class's floor)`."""
    we.assert_base(h.lm, h.hooks_at_load)
    plan, raw_items, base = parse_rebuild_description(description)
    if base.get("model") != h.model_id or base.get("weights_id") != h.weights_id:
        raise EvalRefused(
            f"The description names base {base.get('model')!r} (weights {base.get('weights_id')!r}); this model is "
            f"{h.model_id!r} (weights {h.weights_id!r})."
        )
    items = {
        nm: lv.make_item(h.tokenizer, v["question"], v["answer"], system=v.get("system"), name=nm)
        for nm, v in raw_items.items()
    }
    with _use_book(Path(out_book)):
        rebuilt = lv.read(h, plan, items)
    we.assert_base(h.lm, h.hooks_at_load)

    o_ver = lb.verify(Path(original_book))
    if not o_ver["ok"]:
        raise EvalRefused(f"The original book fails at line {o_ver['first_bad']}: {o_ver['why']}")
    ob = next(lb.entries(Path(original_book), start=int(original_seq), limit=1))["body"]
    r_seq = int((rebuilt.get("book") or {}).get("seq", -1))
    rb = next(lb.entries(Path(out_book), start=r_seq, limit=1))["body"] if r_seq >= 0 else {}

    def _plan_wo_name(b: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in (b.get("plan") or {}).items() if k != "name"}

    status_ok = ob.get("status") == "ok" and rb.get("status") == "ok"
    plan_same = _plan_wo_name(ob) == _plan_wo_name(rb)
    base_same = ob.get("base") == rb.get("base")
    o_items = list(ob.get("items") or [])
    r_items = list(rb.get("items") or [])
    items_same = [((i.get("item") or {}).get("name"), (i.get("item") or {}).get("id")) for i in o_items] == \
                 [((i.get("item") or {}).get("name"), (i.get("item") or {}).get("id")) for i in r_items]
    arms_same = True
    ranks_same = True
    diffs: List[Dict[str, Any]] = []
    for o_it, r_it in zip(o_items, r_items):
        nm = (o_it.get("item") or {}).get("name")
        for cls_k in ("small", "large"):
            o_cls = ((o_it.get("classes") or {}).get(cls_k)) or {}
            r_cls = ((r_it.get("classes") or {}).get(cls_k)) or {}
            if o_cls.get("rank") != r_cls.get("rank"):
                ranks_same = False
            o_arms = {a["key"]: a for a in (o_cls.get("arms") or []) if isinstance(a, Mapping) and "key" in a}
            r_arms = {a["key"]: a for a in (r_cls.get("arms") or []) if isinstance(a, Mapping) and "key" in a}
            if set(o_arms) != set(r_arms) or not o_arms:
                arms_same = False
            o_fl = o_cls.get("floor") or {}
            for ak, oa in o_arms.items():
                ra = r_arms.get(ak) or {}
                o_mets = oa.get("metrics") or {}
                r_mets = ra.get("metrics") or {}
                for met in sorted(set(o_mets) | set(r_mets)):
                    fl_blk = o_fl.get(met)
                    fl = float(_clean_float(fl_blk.get("mean_abs") if isinstance(fl_blk, Mapping) else fl_blk) or 0.0)
                    tol = max(1e-5, tol_floor_mult * fl)
                    for stat in ("mean", "median"):
                        ov = _clean_float((o_mets.get(met) or {}).get(stat))
                        rv = _clean_float((r_mets.get(met) or {}).get(stat))
                        if ov is None and rv is None:
                            continue
                        gap = abs(ov - rv) if (ov is not None and rv is not None) else float("inf")
                        diffs.append({
                            "item": nm, "class": cls_k, "arm": ak, "metric": met, "stat": stat,
                            "original": ov, "rebuilt": rv, "gap": round(gap, 8) if math.isfinite(gap) else None,
                            "tol": round(tol, 8), "ok": gap <= tol,
                        })
    figures_ok = bool(diffs) and all(d["ok"] for d in diffs)
    finite_gaps = [d["gap"] for d in diffs if d["gap"] is not None]
    return {
        "ok": bool(status_ok and plan_same and base_same and items_same and arms_same and ranks_same and figures_ok),
        "what": REBUILD_REPLAY_WHAT,
        "word_count": description_words(description),
        "source_seq": int(original_seq),
        "rebuilt_seq": r_seq,
        "rebuilt_status": rebuilt.get("status"),
        "plan_same": plan_same,
        "base_same": base_same,
        "items_same": items_same,
        "arms_same": arms_same,
        "ranks_same": ranks_same,
        "items_compared": len(o_items),
        "arms_compared": sorted({d["arm"] for d in diffs}),
        "figures_compared": len(diffs),
        "max_gap": max(finite_gaps) if finite_gaps else None,
        "comparisons": diffs,
    }
