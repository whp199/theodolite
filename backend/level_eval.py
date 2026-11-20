"""Levelling Phase 6: multi-session driver evaluation, reach ladders, cheaper-knob comparison, blinded benchmarks,
and future-ready protocol checks (`backend/level_eval.py`).

Owner-side evaluation orchestrator for Phase 6 ("Whether agents help"; `verify/arbiter.md` §5, §8, row 33, row 34;
`design/synthesis.md` §ix; `design/driver.md` §6):
- Multi-session studies (`EvalStudyPlan`, `run_eval_study`): runs `N` independent sessions per arm (default `N >= 3`
  for session variance), each in its own isolated session directory and isolated side level book seeded ONLY with the
  study's verified calibration readings — zero cross-session leakage, zero mutation of the owner's default level book,
  and `weight_edits.assert_base` verified before and after every session.
- Never pools reaches or driver families: results are grouped strictly by `(driver_family, driver_model, reach, mode,
  bare)` and compared only across matched brief and matched budget:
  1. Weight edits (`mode="weight_edit"`) vs cheaper knobs (`mode="knobs_only"`: prompts, decoding, steering).
  2. Reach ladder (`behaviour` -> `activations` -> `weights` -> `atlas`, plus `tools_only`).
  3. Agent drivers vs deterministic scripted no-driver baselines.
  4. Driver family comparisons and `--bare` scaffolding ablations (`docs/SCAFFOLD.md`).
- Blinded planted-fault & canary driver benchmark (`run_blinded_benchmark`): tests discovery & repair at both reaches
  (`with_healthy_reference=True` and `False`) beside identical-card blinded canaries, scoring localization, recovery,
  neutral KL, and honest abstention strictly outside the session.
- Future-ready protocol checks (`arbiter.md` §5):
  - Cold-driver one-page brief (`render_cold_driver_page`, `<= 600` words, naming no site or layer).
  - Rebuild test (`make_rebuild_description`, `parse_rebuild_description`, `run_rebuild_check`, `<= 200` words).

Never imported by `level_mcp.py`, `level_session.py`, `level_routes.py`, or `scientist_tools.py`.
Zero kept-out words appear in source, docstrings, reports, or rendered sheets.
"""
from __future__ import annotations

import contextlib
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

from backend import atlas_ops as ao
from backend import level_atlas as la
from backend import level_book as lb
from backend import level_briefs as lbr
from backend import level_claim as lcl
from backend import level_drivers as ldrv
from backend import level_knobs as lkn
from backend import level_session as lses
from backend import levelling as lv
from backend import weight_edits as we
from backend.census import write_json_atomic

EVAL_VERSION = 1
MIN_VARIANCE_SESSIONS = 3
REACH_ORDER = ("behaviour", "activations", "weights", "atlas", "tools_only", "sandbox_no_weights", "sandbox", "attended")
FORBIDDEN_CONFIG_KEYS = ("api_key", "secret", "token", "password", "auth_token")
MAX_COLD_PAGE_WORDS = 600
MAX_REBUILD_WORDS = 200


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


def study_plan_from_dict(raw: Mapping[str, Any]) -> EvalStudyPlan:
    """Deserialize an `EvalStudyPlan` from JSON."""
    if not isinstance(raw, Mapping):
        raise EvalRefused("Study plan must be a JSON object.")
    arms_raw = raw.get("arms")
    if not isinstance(arms_raw, Sequence) or isinstance(arms_raw, (str, bytes)) or not arms_raw:
        raise EvalRefused("Study plan requires a non-empty list of arms.")
    arms: List[EvalArm] = []
    for i, ar in enumerate(arms_raw):
        if not isinstance(ar, Mapping):
            raise EvalRefused(f"arms[{i}] must be a JSON object.")
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
    sb_raw = dict(raw.get("session_budget") or {})
    db_raw = dict(raw.get("driver_budget") or {})
    return EvalStudyPlan(
        name=str(raw.get("name") or "phase6_study"),
        brief_name=str(raw.get("brief_name") or "pronoun_pull"),
        brief_path=raw.get("brief_path"),
        arms=tuple(arms),
        sessions_per_arm=int(raw.get("sessions_per_arm", MIN_VARIANCE_SESSIONS)),
        session_budget=lses.SessionBudget(**sb_raw),
        driver_budget=ldrv.DriverBudget(**db_raw),
        claim_design=dict(raw.get("claim_design") or {}),
        max_total_dollars=_clean_float(raw.get("max_total_dollars")),
        max_total_wall_seconds=float(raw.get("max_total_wall_seconds", 14400.0)),
        seed_base=int(raw.get("seed_base", 0)),
        note=str(raw.get("note") or ""),
    )


def validate_study_plan(
    plan: EvalStudyPlan,
    *,
    weights_dir: Optional[Path] = None,
    atlas_dir: Optional[Path] = None,
    ignore_reach_env: bool = False,
) -> Dict[str, Any]:
    """Validate every arm, reach, containment rule, and budget ceiling BEFORE running any session."""
    if not plan.name.strip():
        raise EvalRefused("Study plan requires a non-empty name.")
    if not plan.arms:
        raise EvalRefused("Study plan requires at least one arm.")
    if isinstance(plan.sessions_per_arm, bool) or int(plan.sessions_per_arm) < 1:
        raise EvalRefused(f"sessions_per_arm must be >= 1; got {plan.sessions_per_arm!r}.")
    if plan.max_total_wall_seconds <= 0.0:
        raise EvalRefused("max_total_wall_seconds must be positive.")

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
) -> EvalStudyPlan:
    """Build the canonical Phase 6 comparison matrix (reach ladder + matched-budget cheaper-knobs arm)."""
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

    return EvalStudyPlan(
        name=name,
        brief_name=brief_name,
        arms=tuple(arms),
        sessions_per_arm=int(sessions_per_arm),
        session_budget=session_budget or lses.SessionBudget(),
        driver_budget=driver_budget or ldrv.DriverBudget(max_dollars=0.0 if driver_family == "scripted" else None),
        claim_design=dict(claim_design or {}),
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
    """Execute a complete Phase 6 multi-session comparison study with strict per-session isolation."""
    we.assert_base(h.lm, h.hooks_at_load)
    val = validate_study_plan(plan, weights_dir=weights_dir, atlas_dir=atlas_dir, ignore_reach_env=ignore_reach_env)
    brief = _resolve_brief(plan, h.tokenizer)

    sdir = Path(study_dir).resolve()
    sdir.mkdir(parents=True, exist_ok=True)
    sessions_root = sdir / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    study_book = sdir / "study_book.jsonl"
    cal_book_path = sdir / "calibration_book.jsonl"
    bars_path = sdir / "bars.json"
    ks = kill_switch or ldrv.KillSwitch(stop_file=sdir / "STOP")

    # 1. Calibrate bars once (or verify provided calibration) into a clean calibration book
    if bars is not None and calibration_book is not None:
        cal_src = Path(calibration_book).resolve()
        note = lcl.calibration_note(bars, cal_src)
        if note is None:
            raise EvalRefused(f"Provided BarSet {bars.short()} has no calibration note in {cal_src}.")
        why_cal = lcl.calibration_follows(note, cal_src, brief, bars)
        if why_cal is not None:
            raise EvalRefused(f"Provided calibration does not follow from {cal_src}: {why_cal}")
        shutil.copy2(cal_src, cal_book_path)
        lcl.save_bars(bars, bars_path)
        study_bars = bars
    else:
        cal_layer = _default_calibration_layer(h, brief)
        study_bars = lcl.calibrate_bars(
            h, brief, layer=cal_layer, n_controls=int(cal_controls),
            rows=dict(cal_rows) if cal_rows else None,
            book_path=cal_book_path,
        )
        lcl.save_bars(study_bars, bars_path)

    we.assert_base(h.lm, h.hooks_at_load)

    # 2. Persist study plan & write study_start note to study_book.jsonl BEFORE running sessions
    plan_doc = {
        **plan.canonical_dict(),
        "study_sha256": plan.sha256,
        "brief_id": brief.id,
        "bars_sha256": study_bars.id,
        "model_id": h.model_id,
        "weights_id": h.weights_id,
        "unpacker": lv.unpacker_describe().get("mode"),
        "started_at": _utc_now(),
        "validation": val,
    }
    write_json_atomic(sdir / "study_plan.json", plan_doc)
    start_entry = lb.append("note", {"note": "eval_study_start", **plan_doc}, path=study_book)

    # 3. Execute every (arm, replicate) in its own strictly isolated session dir & isolated side level book
    outcomes: List[Dict[str, Any]] = []
    halted_reason: Optional[str] = None
    total_dollars = 0.0
    t_study = time.monotonic()

    with ks.install_signals():
        for arm in plan.arms:
            if halted_reason:
                break
            for rep in range(int(plan.sessions_per_arm)):
                why_ks = ks.triggered()
                if why_ks:
                    halted_reason = f"kill_switch: {why_ks}"
                    break
                if (time.monotonic() - t_study) >= float(plan.max_total_wall_seconds):
                    halted_reason = "study_wall_cap"
                    break
                if plan.max_total_dollars is not None and total_dollars >= float(plan.max_total_dollars):
                    halted_reason = "study_dollar_cap"
                    break

                run_id = f"{arm.arm_id}_r{rep}"
                run_dir = sessions_root / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                # Seed this session's isolated side book with ONLY the verified calibration readings
                run_book = run_dir / "level_book.jsonl"
                shutil.copy2(cal_book_path, run_book)

                backend_tag = arm.driver_family if arm.driver_family in lses.VALID_BACKENDS else "local"
                if arm.driver_family == "scripted":
                    backend_tag = "script"

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
                try:
                    outcome = ldrv.run_driver(
                        sess,
                        arm.driver_family,
                        driver_model=arm.driver_model,
                        driver_budget=plan.driver_budget,
                        kill_switch=ks,
                        seed=int(plan.seed_base) + rep,
                        config=arm.driver_config,
                    )
                finally:
                    if not sess._closed:
                        sess.close()
                    we.assert_base(h.lm, h.hooks_at_load)

                outcome["arm_id"] = arm.arm_id
                outcome["replicate"] = rep
                outcome["study_sha256"] = plan.sha256
                total_dollars += float((outcome.get("usage") or {}).get("estimated_dollars") or 0.0)
                outcomes.append(outcome)
                write_json_atomic(run_dir / "outcome.json", outcome)
                lb.append(
                    "note",
                    {
                        "note": "eval_session",
                        "study_sha256": plan.sha256,
                        "arm_id": arm.arm_id,
                        "replicate": rep,
                        "outcome": outcome,
                    },
                    path=study_book,
                )

    summary = summarize_study(outcomes, plan_doc, halted_reason=halted_reason)
    report_md = render_study_report(summary)
    (sdir / "report.md").write_text(report_md + "\n", encoding="utf-8")
    write_json_atomic(sdir / "summary.json", summary)
    end_entry = lb.append(
        "note",
        {
            "note": "eval_study_end",
            "study_sha256": plan.sha256,
            "finished_at": _utc_now(),
            "halted_reason": halted_reason,
            "completed_sessions": len(outcomes),
            "planned_sessions": len(plan.arms) * int(plan.sessions_per_arm),
            "total_seconds": round(time.monotonic() - t_study, 3),
            "total_estimated_dollars": round(total_dollars, 6),
        },
        path=study_book,
    )
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


# -------------------------------------------------------------------------------------- aggregation (never pooled!)


def _arm_group_key(outcome: Mapping[str, Any]) -> Tuple[str, str, str, str, bool]:
    hc = outcome.get("harness_card") or {}
    return (
        str(hc.get("driver_family") or "unknown"),
        str(hc.get("driver_model") or "unknown"),
        str(hc.get("reach") or "unknown"),
        str(hc.get("mode") or "weight_edit"),
        bool(hc.get("bare", False)),
    )


def summarize_study(
    outcomes: Sequence[Mapping[str, Any]],
    plan_doc: Mapping[str, Any],
    *,
    halted_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate Phase 6 outcomes strictly by `(driver_family, driver_model, reach, mode, bare)` — NEVER pooling
    reaches or driver families — and compute matched comparisons."""
    grouped: Dict[Tuple[str, str, str, str, bool], List[Mapping[str, Any]]] = {}
    for o in outcomes:
        grouped.setdefault(_arm_group_key(o), []).append(o)

    arm_summaries: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str, str, str, bool], Dict[str, Any]] = {}

    for key, rows in sorted(
        grouped.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], REACH_ORDER.index(kv[0][2]) if kv[0][2] in REACH_ORDER else 99, kv[0][4]),
    ):
        fam, mod, reach, mode, bare = key
        n = len(rows)
        verdict_counts: Dict[str, int] = {}
        effs: List[float] = []
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

        for r in rows:
            v = str(r.get("verdict") or "no_submission")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
            m = r.get("metrics") or {}
            de = _clean_float(m.get("directed_effect"))
            if de is not None:
                effs.append(de)
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
            })

        n_leads = verdict_counts.get("lead", 0)
        n_leads_knob = verdict_counts.get("lead, matched by a cheaper knob", 0)
        n_any_lead = n_leads + n_leads_knob
        ci_lo, ci_hi = wilson_ci(n_leads, n)
        ci_any_lo, ci_any_hi = wilson_ci(n_any_lead, n)
        total_usd = float(sum(usd))
        total_gpu = float(sum(gpu_s))
        eff_arr = np.asarray(effs, dtype=np.float64) if effs else np.asarray([], dtype=np.float64)

        arm_rec = {
            "driver_family": fam,
            "driver_model": mod,
            "reach": reach,
            "mode": mode,
            "bare": bare,
            "arm_ids": sorted({str(r.get("arm_id") or "") for r in rows}),
            "n_sessions": n,
            "under_sampled": n < MIN_VARIANCE_SESSIONS,
            "n_submitted": sum(1 for r in rows if r.get("submitted")),
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
                "mean": round(float(eff_arr.mean()), 5) if effs else None,
                "median": round(float(np.median(eff_arr)), 5) if effs else None,
                "sd": round(float(eff_arr.std(ddof=1)), 5) if len(effs) >= 2 else None,
                "min": round(float(eff_arr.min()), 5) if effs else None,
                "max": round(float(eff_arr.max()), 5) if effs else None,
            },
            "near_miss_abs_median": round(float(np.median(nm_vals)), 5) if nm_vals else None,
            "neutral_kl_median": round(float(np.median(neu_vals)), 5) if neu_vals else None,
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
        by_key[key] = arm_rec

    # Structured matched comparisons (NEVER across mismatched briefs/budgets)
    # 1. Weight edits vs Cheaper Knobs (same driver_family, reach, bare)
    weight_vs_knobs: List[Dict[str, Any]] = []
    for (fam, mod, reach, mode, bare), w_arm in by_key.items():
        if mode != "weight_edit":
            continue
        # Match same family & reach & bare; for scripted, driver_model differs ("scripted_weight_v1" vs "scripted_knob_v1")
        k_matches = [
            k_arm for (kf, km, kr, kmode, kbare), k_arm in by_key.items()
            if kmode == "knobs_only" and kf == fam and kr == reach and kbare == bare
            and (fam == "scripted" or km == mod)
        ]
        for k_arm in k_matches:
            w_l, w_n = w_arm["n_leads"], w_arm["n_sessions"]
            k_l, k_n = k_arm["n_leads"], k_arm["n_sessions"]
            p_fish = fisher_exact_2x2(w_l, w_n - w_l, k_l, k_n - k_l)
            p_eff = exact_perm_diff_p(w_arm["directed_effect"]["values"], k_arm["directed_effect"]["values"])
            if w_l == 0 and k_l == 0:
                comp_label = "neither_found_a_lead"
            elif k_arm["lead_rate"] >= w_arm["lead_rate"] and (
                (k_arm["directed_effect"]["median"] or -1e9) >= 0.95 * (w_arm["directed_effect"]["median"] or 0.0)
            ):
                comp_label = "knobs_matched_or_ahead"
            elif w_arm["lead_rate"] > k_arm["lead_rate"] and w_n >= MIN_VARIANCE_SESSIONS and k_n >= MIN_VARIANCE_SESSIONS:
                comp_label = "weight_edit_ahead"
            else:
                comp_label = "inconclusive_small_n"
            weight_vs_knobs.append({
                "driver_family": fam,
                "reach": reach,
                "bare": bare,
                "weight_edit_arm": w_arm["arm_ids"],
                "knobs_only_arm": k_arm["arm_ids"],
                "weight_lead_rate": w_arm["lead_rate"],
                "knobs_lead_rate": k_arm["lead_rate"],
                "weight_median_effect": w_arm["directed_effect"]["median"],
                "knobs_median_effect": k_arm["directed_effect"]["median"],
                "weight_mean_gpu_s": w_arm["cost"]["mean_gpu_seconds"],
                "knobs_mean_gpu_s": k_arm["cost"]["mean_gpu_seconds"],
                "fisher_exact_p": p_fish,
                "effect_perm_test": p_eff,
                "comparison": comp_label,
            })

    # 2. Reach Ladder (same driver_family, driver_model, mode, bare across reaches)
    reach_groups: Dict[Tuple[str, str, str, bool], List[Dict[str, Any]]] = {}
    for arm_rec in arm_summaries:
        rk = (arm_rec["driver_family"], arm_rec["driver_model"], arm_rec["mode"], arm_rec["bare"])
        reach_groups.setdefault(rk, []).append(arm_rec)
    reach_ladders: List[Dict[str, Any]] = []
    for (fam, mod, mode, bare), rlist in sorted(reach_groups.items()):
        if len(rlist) < 2:
            continue
        ordered = sorted(rlist, key=lambda a: REACH_ORDER.index(a["reach"]) if a["reach"] in REACH_ORDER else 99)
        rungs = []
        atlas_vs_weights = None
        w_rung = next((a for a in ordered if a["reach"] == "weights"), None)
        a_rung = next((a for a in ordered if a["reach"] == "atlas"), None)
        if w_rung is not None and a_rung is not None:
            atlas_vs_weights = {
                "weights_lead_rate": w_rung["lead_rate"],
                "atlas_lead_rate": a_rung["lead_rate"],
                "weights_median_effect": w_rung["directed_effect"]["median"],
                "atlas_median_effect": a_rung["directed_effect"]["median"],
                "weights_mean_gpu_s": w_rung["cost"]["mean_gpu_seconds"],
                "atlas_mean_gpu_s": a_rung["cost"]["mean_gpu_seconds"],
                "effect_perm_test": exact_perm_diff_p(
                    a_rung["directed_effect"]["values"], w_rung["directed_effect"]["values"]
                ),
            }
        for a in ordered:
            rungs.append({
                "reach": a["reach"],
                "n_sessions": a["n_sessions"],
                "lead_rate": a["lead_rate"],
                "lead_rate_ci95": a["lead_rate_ci95"],
                "median_effect": a["directed_effect"]["median"],
                "mean_gpu_seconds": a["cost"]["mean_gpu_seconds"],
                "mean_practice_forwards": a["cost"]["mean_practice_forwards"],
            })
        reach_ladders.append({
            "driver_family": fam,
            "driver_model": mod,
            "mode": mode,
            "bare": bare,
            "rungs": rungs,
            "atlas_vs_weights": atlas_vs_weights,
        })

    # 3. Agent vs Scripted baseline (same reach, mode)
    agent_vs_scripted: List[Dict[str, Any]] = []
    scripted_by_rm = {
        (a["reach"], a["mode"]): a for a in arm_summaries if a["driver_family"] == "scripted" and not a["bare"]
    }
    for a in arm_summaries:
        if a["driver_family"] == "scripted":
            continue
        base_s = scripted_by_rm.get((a["reach"], a["mode"]))
        if base_s is None:
            continue
        agent_vs_scripted.append({
            "driver_family": a["driver_family"],
            "driver_model": a["driver_model"],
            "reach": a["reach"],
            "mode": a["mode"],
            "bare": a["bare"],
            "agent_lead_rate": a["lead_rate"],
            "scripted_lead_rate": base_s["lead_rate"],
            "agent_median_effect": a["directed_effect"]["median"],
            "scripted_median_effect": base_s["directed_effect"]["median"],
            "agent_total_dollars": a["cost"]["total_estimated_dollars"],
            "fisher_exact_p": fisher_exact_2x2(
                a["n_leads"], a["n_sessions"] - a["n_leads"],
                base_s["n_leads"], base_s["n_sessions"] - base_s["n_leads"],
            ),
        })

    # 4. Scaffolding ablation (bare=False vs bare=True)
    scaffolding_ablations: List[Dict[str, Any]] = []
    for (fam, mod, reach, mode, bare), std_arm in by_key.items():
        if bare:
            continue
        bare_arm = by_key.get((fam, mod, reach, mode, True))
        if bare_arm is None:
            continue
        scaffolding_ablations.append({
            "driver_family": fam,
            "driver_model": mod,
            "reach": reach,
            "mode": mode,
            "standard_lead_rate": std_arm["lead_rate"],
            "bare_lead_rate": bare_arm["lead_rate"],
            "standard_median_effect": std_arm["directed_effect"]["median"],
            "bare_median_effect": bare_arm["directed_effect"]["median"],
            "standard_mean_gpu_s": std_arm["cost"]["mean_gpu_seconds"],
            "bare_mean_gpu_s": bare_arm["cost"]["mean_gpu_seconds"],
        })

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
        "completed_sessions": len(outcomes),
        "halted_reason": halted_reason,
        "pooled_across_reaches": False,
        "pooled_across_families": False,
        "arms": arm_summaries,
        "comparisons": {
            "weight_vs_knobs": weight_vs_knobs,
            "reach_ladders": reach_ladders,
            "agent_vs_scripted": agent_vs_scripted,
            "scaffolding_ablations": scaffolding_ablations,
        },
        "limitations": [
            "Every session ran in an isolated side book seeded with the study's calibration readings: no edit from a Phase 6 multi-session study is a trial of record.",
            "Reaches and driver families are reported separately and never pooled.",
            "Small session counts (e.g. N = 3) measure coarse variance only and cannot separate close rates.",
            "Every result describes (driver_family, driver_model, harness, reach, mode, budget) together.",
            "Cheaper knobs here cover prompts, decoding, and transient steering; fine-tuned LoRA adapters remain unbuilt until needed.",
        ],
    }


def load_study_from_book(study_dir: Path) -> Dict[str, Any]:
    """Model-free reload and verification of a completed Phase 6 study from its chained `study_book.jsonl`."""
    sdir = Path(study_dir).resolve()
    sb = sdir / "study_book.jsonl"
    if not sb.is_file():
        raise EvalRefused(f"No study_book.jsonl found in {sdir}.")
    ver = lb.verify(sb)
    if not ver["ok"]:
        raise EvalRefused(f"study_book.jsonl verification failed at line {ver['first_bad']}: {ver['why']}")

    plan_doc: Dict[str, Any] = {}
    outcomes: List[Dict[str, Any]] = []
    halted_reason: Optional[str] = None
    session_books_checked = 0
    session_books_bad: List[str] = []

    for entry in lb.entries(sb):
        if entry.get("kind") != "note":
            continue
        body = entry.get("body") or {}
        nt = body.get("note")
        if nt == "eval_study_start":
            plan_doc = dict(body)
        elif nt == "eval_session":
            out = body.get("outcome")
            if isinstance(out, dict):
                outcomes.append(out)
                bp = (out.get("book") or {}).get("path")
                if bp and Path(bp).is_file():
                    session_books_checked += 1
                    sv = lb.verify(Path(bp))
                    if not sv["ok"]:
                        session_books_bad.append(f"{out.get('arm_id')}_r{out.get('replicate')}: {sv['why']}")
        elif nt == "eval_study_end":
            halted_reason = body.get("halted_reason")

    summary = summarize_study(outcomes, plan_doc, halted_reason=halted_reason)
    return {
        "study_dir": str(sdir),
        "book_verified": True,
        "book_entries": int(ver["entries"]),
        "book_head": ver["head"],
        "session_books_checked": session_books_checked,
        "session_books_bad": session_books_bad,
        "summary": summary,
        "report": render_study_report(summary),
    }


def render_study_report(summary: Mapping[str, Any]) -> str:
    """Render a bounded, plain-spoken Markdown report for a Phase 6 study (zero kept-out words)."""
    lines = [
        f"# Levelling Phase 6 Driver Evaluation — {summary.get('study_name')}",
        "",
        f"- Study hash: `{str(summary.get('study_sha256') or '')[:12]}`",
        f"- Brief: `{summary.get('brief_name')}` (`{str(summary.get('brief_id') or '')[:12]}`), bars `{str(summary.get('bars_sha256') or '')[:12]}`",
        f"- Base: `{summary.get('model_id')}` (weights `{str(summary.get('weights_id') or '')[:12]}`, unpacker `{summary.get('unpacker')}`)",
        f"- Completed sessions: {summary.get('completed_sessions')} (planned per arm: {summary.get('sessions_per_arm_planned')})",
        "- Pooling policy: reaches and driver families are **never pooled**.",
    ]
    if summary.get("halted_reason"):
        lines.append(f"- Halted early: `{summary['halted_reason']}`")

    lines += [
        "",
        "## Per-Arm Results (Unpooled)",
        "",
        "| Driver | Model | Reach | Mode | Bare | N | Leads | Lead Rate (95% Wilson CI) | Directed Effects (Replicates) | Median Effect | Mean GPU-s | Total USD |",
        "|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for a in summary.get("arms") or []:
        ci = a.get("lead_rate_ci95") or [None, None]
        ci_str = f"{a['lead_rate']:.2f} [{ci[0]}, {ci[1]}]" if ci[0] is not None else f"{a['lead_rate']:.2f}"
        if a.get("under_sampled"):
            ci_str += " (N<3)"
        vals_str = ", ".join(str(x) for x in (a.get("directed_effect") or {}).get("values") or []) or "none"
        med = (a.get("directed_effect") or {}).get("median")
        med_str = f"{med:.4f}" if isinstance(med, (int, float)) else "none"
        lines.append(
            f"| {a['driver_family']} | {a['driver_model']} | {a['reach']} | {a['mode']} | "
            f"{'yes' if a['bare'] else 'no'} | {a['n_sessions']} | {a['n_leads']} | {ci_str} | "
            f"{vals_str} | {med_str} | {a['cost']['mean_gpu_seconds']:.2f} | {a['cost']['total_estimated_dollars']:.4f} |"
        )

    comps = summary.get("comparisons") or {}
    wvk = comps.get("weight_vs_knobs") or []
    if wvk:
        lines += [
            "",
            "## Weight Edits vs Cheaper Knobs (Same Driver, Reach, and Budget)",
            "",
            "| Driver | Reach | Weight Lead Rate | Knobs Lead Rate | Weight Median | Knobs Median | Fisher p | Outcome |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in wvk:
            lines.append(
                f"| {row['driver_family']} | {row['reach']} | {row['weight_lead_rate']:.2f} | "
                f"{row['knobs_lead_rate']:.2f} | {row['weight_median_effect']} | {row['knobs_median_effect']} | "
                f"{row['fisher_exact_p']} | `{row['comparison']}` |"
            )

    ladders = comps.get("reach_ladders") or []
    if ladders:
        lines += ["", "## Reach Ladder (Never Pooled)", ""]
        for lad in ladders:
            lines.append(f"### {lad['driver_family']} / {lad['driver_model']} ({lad['mode']}, bare={lad['bare']})")
            lines.append("")
            lines.append("| Reach | N | Lead Rate | Median Effect | Mean GPU-s | Mean Practice Forwards |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for rg in lad["rungs"]:
                lines.append(
                    f"| {rg['reach']} | {rg['n_sessions']} | {rg['lead_rate']:.2f} | "
                    f"{rg['median_effect']} | {rg['mean_gpu_seconds']:.2f} | {rg['mean_practice_forwards']:.1f} |"
                )
            avw = lad.get("atlas_vs_weights")
            if avw is not None:
                lines.append(
                    f"- Atlas vs weights: lead rate {avw['atlas_lead_rate']:.2f} vs {avw['weights_lead_rate']:.2f}, "
                    f"median effect {avw['atlas_median_effect']} vs {avw['weights_median_effect']}, "
                    f"mean GPU-s {avw['atlas_mean_gpu_s']:.2f} vs {avw['weights_mean_gpu_s']:.2f}."
                )
            lines.append("")

    avs = comps.get("agent_vs_scripted") or []
    if avs:
        lines += [
            "## Agent vs Scripted No-Driver Baseline",
            "",
            "| Driver | Model | Reach | Mode | Agent Lead Rate | Scripted Lead Rate | Fisher p | Agent USD |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for row in avs:
            lines.append(
                f"| {row['driver_family']} | {row['driver_model']} | {row['reach']} | {row['mode']} | "
                f"{row['agent_lead_rate']:.2f} | {row['scripted_lead_rate']:.2f} | "
                f"{row['fisher_exact_p']} | {row['agent_total_dollars']:.4f} |"
            )

    abls = comps.get("scaffolding_ablations") or []
    if abls:
        lines += [
            "",
            "## Scaffolding Ablation (Standard vs `--bare`)",
            "",
            "| Driver | Reach | Mode | Standard Lead Rate | Bare Lead Rate | Standard Median | Bare Median |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for row in abls:
            lines.append(
                f"| {row['driver_family']} | {row['reach']} | {row['mode']} | "
                f"{row['standard_lead_rate']:.2f} | {row['bare_lead_rate']:.2f} | "
                f"{row['standard_median_effect']} | {row['bare_median_effect']} |"
            )

    lines += ["", "## What This Study Does Not Show", ""]
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
    """Run blinded planted-fault (at healthy-reference and no-healthy-reference reaches) and canary sessions,
    verifying identical harness cards and scoring localization, repair, and abstention strictly outside the session."""
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
    cards_by_reach: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    cards_by_reach_driver: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}

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
        raw_card = sess.harness_card()
        card_no_id = {k: v for k, v in raw_card.items() if k != "session_id"}
        card_reach_norm = {k: v for k, v in raw_card.items() if k not in ("session_id", "backend")}
        cards_by_reach.setdefault(case.reach, []).append((case.kind, card_reach_norm))
        cards_by_reach_driver.setdefault(f"{case.reach}:{case.driver_family}", []).append((case.kind, card_no_id))

        try:
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
        score: Dict[str, Any] = {
            "case_id": case.case_id,
            "kind": case.kind,
            "reach": case.reach,
            "with_healthy_reference": case.reach == "weights",
            "driver_family": case.driver_family,
            "proposed_any": prop_recipe is not None,
            "composition_note": "In-memory plants compose additively with repairs via weight_edits.stack (ROADMAP §viii).",
        }

        if case.kind == "planted" and plant_edit is not None:
            p_layers, p_sites = _recipe_sites(case.plant_recipe)
            r_layers, r_sites = _recipe_sites(prop_recipe)
            layer_hit = bool(p_layers & r_layers)
            overlap = len(p_sites & r_sites)
            recall = round(overlap / float(len(p_sites)), 4) if p_sites else 0.0
            jaccard = round(overlap / float(len(p_sites | r_sites)), 4) if (p_sites | r_sites) else 0.0
            # Measure KL distortion of plant alone vs stacked (plant + proposed repair) against healthy base
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
            kl_after = None
            recovery = None
            if prop_recipe is not None:
                try:
                    prop_e = _build_edit_from_recipe(h, prop_recipe, sess.work_dir)
                    stacked = we.stack(h.lm, [plant_edit, prop_e], model_id=h.model_id, weights_id=h.weights_id, plant_first=True)
                    if _is_noop_edit(stacked):
                        kl_after = 0.0
                    else:
                        plan_st = {
                            "name": "gt_plant_plus_repair", "kind": "paired", "edit": case.plant_recipe,
                            "items": list(eval_items.keys()), "doses": [1.0],
                            "rows": {"small": 8, "large": 16}, "controls": {"count": 1, "seed": 0},
                            "readouts": {"kl": True, "own": True, "odds": None},
                        }
                        with _use_book(eval_book):
                            r_st = lv.read(h, plan_st, eval_items, edit=stacked)
                        st_arms = (((r_st.get("summary") or {}).get("classes") or {}).get("small") or {}).get("arms") or {}
                        kl_after = _clean_float(((st_arms.get(edit_key) or {}).get("kl") or {}).get("mean_over_items"))
                    if kl_plant is not None and kl_plant > 1e-12 and kl_after is not None:
                        recovery = round((kl_plant - kl_after) / kl_plant, 4)
                except Exception as exc:
                    score["repair_eval_error"] = str(exc)
            score.update({
                "localized_layer": layer_hit,
                "site_recall": recall,
                "site_jaccard": jaccard,
                "kl_plant_alone": round(kl_plant, 6) if kl_plant is not None else None,
                "kl_after_repair": round(kl_after, 6) if kl_after is not None else None,
                "recovery_fraction": recovery,
            })
        else:
            # Canary: honest abstention means proposing nothing (or an edit with zero effect)
            score["honest_abstention"] = (prop_recipe is None)

        we.assert_base(h.lm, h.hooks_at_load)
        records.append(score)
        lb.append("note", {"note": "eval_blinded_case", "score": score}, path=bbook)

    # Verify blinding invariant: within each reach (and each reach+driver), planted and canary harness cards are identical!
    blinding_cards_equal: Dict[str, bool] = {}
    for rc, pairs in cards_by_reach.items():
        first_card = pairs[0][1]
        blinding_cards_equal[rc] = all(c == first_card for _k, c in pairs)
    blinding_cards_equal_exact: Dict[str, bool] = {}
    for key, pairs in cards_by_reach_driver.items():
        first_card = pairs[0][1]
        blinding_cards_equal_exact[key] = all(c == first_card for _k, c in pairs)

    planted_rows = [r for r in records if r["kind"] == "planted"]
    canary_rows = [r for r in records if r["kind"] == "canary"]
    summary = {
        "n_cases": len(records),
        "n_planted": len(planted_rows),
        "n_canary": len(canary_rows),
        "blinding_cards_equal_by_reach": blinding_cards_equal,
        "blinding_cards_equal_by_reach_driver": blinding_cards_equal_exact,
        "planted_layer_hit_rate": (
            round(sum(1 for r in planted_rows if r.get("localized_layer")) / float(len(planted_rows)), 4)
            if planted_rows else None
        ),
        "canary_abstention_rate": (
            round(sum(1 for r in canary_rows if r.get("honest_abstention")) / float(len(canary_rows)), 4)
            if canary_rows else None
        ),
        "cases": records,
    }
    write_json_atomic(odir / "blinded_summary.json", summary)
    return summary


# -------------------------------------------------------------------------------------- future-ready protocol checks


def render_cold_driver_page(
    brief: lbr.Brief,
    *,
    mcp_command: str = ".venv/bin/python -I backend/level_mcp.py --session-dir DIR --brief BRIEF --bars BARS",
    cli_command: str = ".venv/bin/python -I scripts/levelling.py",
) -> Dict[str, Any]:
    """Generate the self-contained <= 1-page Cold Driver prompt (`arbiter.md` §5, `skeptic_completeness.md` §12).

    Gives any new driver family the one-page contract and an unnamed target behaviour — zero site hints, zero layer
    numbers, zero code changes needed."""
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
    return {
        "brief_name": brief.name,
        "brief_id": brief.id,
        "word_count": words,
        "prompt_named_site": False,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def make_rebuild_description(book_path: Path, seq: int) -> Dict[str, Any]:
    """Create a <= 200-word self-contained rebuild description (`arbiter.md` §5, `skeptic_completeness.md` item 5)
    from a completed paired reading in `book_path`, containing NO output figures."""
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

    plan = body.get("plan") or {}
    edit_recipe = plan.get("edit")
    if not isinstance(edit_recipe, Mapping):
        raise EvalRefused(f"Line {seq} has no rebuildable recipe dict in plan.edit.")
    items_raw = body.get("items") or []
    if not items_raw:
        raise EvalRefused(f"Line {seq} has no recorded items.")
    first_item = items_raw[0]
    item_blk = first_item.get("item") if isinstance(first_item, Mapping) else {}
    q_txt = item_blk.get("question")
    a_txt = item_blk.get("answer")
    if not isinstance(q_txt, str) or not isinstance(a_txt, str):
        raise EvalRefused(f"Line {seq} first item lacks question/answer text for a text rebuild description.")

    doses = list(plan.get("doses") or [1.0])
    rows_cfg = dict(plan.get("rows") or {"small": lv.ROWS_SMALL, "large": lv.ROWS_LARGE})
    ctl_cfg = dict(plan.get("controls") or {"family": "other_sites", "count": 1, "seed": 0})
    ro_cfg = dict(plan.get("readouts") or {"kl": True, "own": True, "odds": None})

    spec = {
        "base": body.get("model_id"),
        "edit": edit_recipe,
        "doses": doses,
        "rows": rows_cfg,
        "controls": {"family": ctl_cfg.get("family", "other_sites"), "count": int(ctl_cfg.get("count", 1)), "seed": int(ctl_cfg.get("seed", 0))},
        "readouts": ro_cfg,
        "item": {"name": item_blk.get("name", "item0"), "question": q_txt, "answer": a_txt, "system": item_blk.get("system")},
    }
    compact = json.dumps(spec, separators=(",", ":"), ensure_ascii=True)
    text = (
        "Rebuild this paired reading on the base model named below using scripts/levelling.py "
        "with no extra settings and compare the edit arm mean against the original:\n"
        f"REBUILD_SPEC: {compact}"
    )
    wc = len(text.split())
    if wc > MAX_REBUILD_WORDS:
        raise EvalRefused(
            f"Rebuild description is {wc} words (limit {MAX_REBUILD_WORDS}); pick a compact recipe/item."
        )
    return {
        "source_book": str(p),
        "source_seq": int(seq),
        "word_count": wc,
        "description": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def parse_rebuild_description(description: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Parse a <= 200-word rebuild description into `(plan_dict, items_dict)` without reading the source book."""
    wc = len(str(description).split())
    if wc > MAX_REBUILD_WORDS:
        raise EvalRefused(f"Rebuild description exceeds {MAX_REBUILD_WORDS} words ({wc}).")
    m = re.search(r"REBUILD_SPEC:\s*(\{.*\})\s*$", str(description), flags=re.S)
    if not m:
        raise EvalRefused("Rebuild description does not contain a REBUILD_SPEC JSON block.")
    try:
        spec = json.loads(m.group(1))
    except ValueError as exc:
        raise EvalRefused(f"Malformed REBUILD_SPEC JSON: {exc}") from None
    it = spec["item"]
    name = str(it.get("name") or "item0")
    plan = {
        "name": "rebuild_check",
        "kind": "paired",
        "edit": spec["edit"],
        "items": [name],
        "doses": spec["doses"],
        "rows": spec["rows"],
        "controls": spec["controls"],
        "readouts": spec["readouts"],
    }
    items = {name: {"question": it["question"], "answer": it["answer"], "system": it.get("system")}}
    return plan, items


def run_rebuild_check(
    h: lv.Handle,
    description: str,
    original_book: Path,
    original_seq: int,
    *,
    out_book: Path,
    tol_floor_mult: float = 1.0,
) -> Dict[str, Any]:
    """Reproduce a reading from its <= 200-word description alone in a fresh book and verify agreement with the source."""
    we.assert_base(h.lm, h.hooks_at_load)
    plan, raw_items = parse_rebuild_description(description)
    items = {
        nm: lv.make_item(h.tokenizer, v["question"], v["answer"], system=v.get("system"), name=nm)
        for nm, v in raw_items.items()
    }
    with _use_book(Path(out_book)):
        rebuilt = lv.read(h, plan, items)
    we.assert_base(h.lm, h.hooks_at_load)

    orig_entry = next(lb.entries(Path(original_book), start=int(original_seq), limit=1))
    orig_body = orig_entry["body"]
    # Compare first item's small & large edit arm metric values (`classes` holds per-class arm lists)
    orig_item = (orig_body.get("items") or [{}])[0]
    reb_item = (rebuilt.get("items") or [{}])[0]
    diffs: List[Dict[str, Any]] = []
    all_ok = rebuilt.get("status") == "ok"

    for cls_k in ("small", "large"):
        o_cls = ((orig_item.get("classes") or {}).get(cls_k)) or {}
        r_cls = ((reb_item.get("classes") or {}).get(cls_k)) or {}
        o_arms = {a["key"]: a for a in (o_cls.get("arms") or []) if isinstance(a, Mapping) and "key" in a}
        r_arms = {a["key"]: a for a in (r_cls.get("arms") or []) if isinstance(a, Mapping) and "key" in a}
        o_fl = o_cls.get("floor") or {}
        for ak, oa in o_arms.items():
            if not (ak.startswith("edit:") or ak.startswith("edit@") or ak == "edit"):
                continue
            ra = r_arms.get(ak) or {}
            o_mets = oa.get("metrics") or {}
            r_mets = ra.get("metrics") or {}
            for met in ("kl", "own", "odds"):
                ov = _clean_float((o_mets.get(met) or {}).get("mean"))
                rv = _clean_float((r_mets.get(met) or {}).get("mean"))
                if ov is None and rv is None:
                    continue
                fl_blk = o_fl.get(met)
                fl = float(_clean_float(fl_blk.get("mean_abs") if isinstance(fl_blk, Mapping) else fl_blk) or 0.0)
                tol = max(1e-5, tol_floor_mult * fl)
                gap = abs(ov - rv) if (ov is not None and rv is not None) else float("inf")
                ok = gap <= tol
                if not ok:
                    all_ok = False
                diffs.append({
                    "class": cls_k, "arm": ak, "metric": met,
                    "original": ov, "rebuilt": rv, "gap": round(gap, 8) if math.isfinite(gap) else None,
                    "tol": round(tol, 8), "ok": ok,
                })

    return {
        "ok": bool(all_ok and diffs),
        "word_count": len(description.split()),
        "source_seq": int(original_seq),
        "rebuilt_seq": int((rebuilt.get("book") or {}).get("seq", -1)),
        "rebuilt_status": rebuilt.get("status"),
        "comparisons": diffs,
    }
