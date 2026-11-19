"""Levelling Phase 5: Trial of Record (`backend/level_trial.py`).

Promotes a candidate weight edit from a Phase 3 fresh reading (`lead`) to a registered Trial of Record run in a fresh
process (`passed` / `failed` / `void`), implementing `verify/arbiter.md` §4 (rows 25, 30, 31, 32) and §8:
- House registration folder (`prereg/`): `trial.json`, `brief.json`, `bars.json`, `edit_manifest.json` (each term
  explicitly marked `"rebuilt"` with its pure-JSON recipe or `"carried"` with sha256 hashes only — NEVER embedding
  weight tensors or `*.safetensors` / `*.gguf` files), optional `reworded.json`, `RULES.md`, `repo_commit`,
  `registered_at`, and `SHA256SUMS`.
- Owner-written reworded item sets (`RewordedSet`), balanced across independent halves (`half = 0`, `half = 1`),
  verified wording-disjoint from the brief's practice and kept-back frames and retention batteries, and capped at
  `MAX_REWORDED_USES = 5` lifetime uses in the level book before rotation.
- Fresh-process execution: verifies `SHA256SUMS`, enforces read-once per registration, hashes the live base weights
  (`weight_edits.base_checksum`) before and after, records `trial_registration` BEFORE `os.urandom(32)` draws, reads
  BOTH batch classes (`small`, `large`) against matched and same-site (plus optional sham-target) control edits AND
  verifies the folded weight form (`folded="always"`) in both batch classes on fresh and reworded items, runs gating
  retention batteries and author prompt arms, checks zero unpacker fallbacks, logs `trial_of_record`, and writes
  `trial_result.json` and `TRIAL_SHEET.md`.

Only a Trial of Record earns `"passed"` or `"failed"`. No kept-out words appear in code, rules, or sheets.
Layers count from 0 in code and files; human-facing strings give both ("layer 50 (page layer 51)").
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend import level_book as lb
from backend import level_briefs as lbr
from backend import level_claim as lcl
from backend import levelling as lv
from backend import weight_edits as we
from backend.census import write_json_atomic

TRIAL_VERSION = 1
TRIAL_RULE_PRESET = "trial-1"
TRIAL_CONTROLS = 99  # synthesis.md §vi: 99 control edits per family for a trial of record
MAX_REWORDED_USES = 5  # synthesis.md §vi: reworded set gives at most 5 readings in its life, then rotates
DEFAULT_FOLDED_FLOOR_MULT = 3.0
DEFAULT_FOLDED_REL_TOL = 0.05
DEFAULT_FOLDED_MIN_TOL = 0.05

TRIAL_VERDICTS = ("passed", "failed", "void")
FORBIDDEN_PREREG_SUFFIXES = (".safetensors", ".gguf")

ROOT_DIR = Path(__file__).resolve().parent.parent


class TrialRefused(ValueError):
    """Plain-word refusal of an invalid, tampered, duplicate, or out-of-policy trial of record."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _clean_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def git_state(root: Path = ROOT_DIR) -> Tuple[str, Optional[bool]]:
    """Return `(commit_sha_or_unknown, is_dirty_or_None)` without failing outside git."""
    try:
        c = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=5,
        )
        if c.returncode != 0:
            return "unknown", None
        commit = c.stdout.strip() or "unknown"
        st = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(root), capture_output=True, text=True, timeout=5,
        )
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


# --------------------------------------------------------------------------------------------- reworded item sets


@dataclass(frozen=True)
class RewordedItem:
    """One owner-written reworded item outside the brief generator's templates."""

    item_id: str
    half: int
    question: str
    answer: str
    system: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "half": int(self.half),
            "question": self.question,
            "answer": self.answer,
            "system": self.system,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RewordedItem":
        if not isinstance(d, Mapping):
            raise TrialRefused("Each reworded item must be a JSON object.")
        iid = str(d.get("item_id") or "").strip()
        if not iid:
            raise TrialRefused("Reworded item requires a non-empty item_id.")
        half = d.get("half")
        if isinstance(half, bool) or half not in (0, 1):
            raise TrialRefused(f'Reworded item "{iid}" half must be 0 or 1; got {half!r}.')
        q = d.get("question")
        a = d.get("answer")
        if not isinstance(q, str) or not q.strip():
            raise TrialRefused(f'Reworded item "{iid}" requires a non-empty question string.')
        if not isinstance(a, str) or not a.strip():
            raise TrialRefused(f'Reworded item "{iid}" requires a non-empty answer string.')
        sys_turn = d.get("system")
        if sys_turn is not None and not isinstance(sys_turn, str):
            raise TrialRefused(f'Reworded item "{iid}" system must be a string or null.')
        return cls(item_id=iid, half=int(half), question=q, answer=a, system=sys_turn)


@dataclass(frozen=True)
class RewordedSet:
    """Versioned, tamper-checked owner-written reworded item set for a Phase 5 trial of record."""

    name: str
    brief_id: str
    brief_name: str
    items: Tuple[RewordedItem, ...]
    note: str = ""
    version: int = TRIAL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TrialRefused("RewordedSet requires a non-empty name.")
        if not isinstance(self.brief_id, str) or not self.brief_id.strip():
            raise TrialRefused("RewordedSet requires a non-empty brief_id.")
        if not isinstance(self.items, tuple) or not self.items:
            raise TrialRefused("RewordedSet requires at least 4 items (2 per half).")
        seen_ids = set()
        seen_qa = set()
        h0 = h1 = 0
        for it in self.items:
            if it.item_id in seen_ids:
                raise TrialRefused(f'Duplicate reworded item_id "{it.item_id}".')
            seen_ids.add(it.item_id)
            qa = (it.question.strip(), it.answer.strip())
            if qa in seen_qa:
                raise TrialRefused(f'Duplicate reworded (question, answer) on item "{it.item_id}".')
            seen_qa.add(qa)
            if it.half == 0:
                h0 += 1
            else:
                h1 += 1
        if h0 < 2 or h1 < 2:
            raise TrialRefused(
                f'RewordedSet "{self.name}" needs >= 2 items in half 0 and >= 2 in half 1 (got {h0} and {h1}).'
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": int(self.version),
            "name": self.name,
            "brief_id": self.brief_id,
            "brief_name": self.brief_name,
            "note": self.note,
            "items": [it.to_dict() for it in self.items],
        }

    @property
    def id(self) -> str:
        return hashlib.sha256(lb.canonical(self.to_dict())).hexdigest()

    def short(self) -> str:
        return self.id[:12]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RewordedSet":
        if not isinstance(d, Mapping):
            raise TrialRefused("RewordedSet must be a JSON object.")
        raw_items = d.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise TrialRefused("RewordedSet requires a non-empty items list.")
        return cls(
            name=str(d.get("name") or ""),
            brief_id=str(d.get("brief_id") or ""),
            brief_name=str(d.get("brief_name") or ""),
            items=tuple(RewordedItem.from_dict(x) for x in raw_items),
            note=str(d.get("note") or ""),
            version=int(d.get("v", TRIAL_VERSION)),
        )

    def to_drawn_items(self, tokenizer: Any, *, system_override: Optional[str] = None) -> List[lbr.DrawnItem]:
        """Materialise every reworded item as a `level_briefs.DrawnItem` (one frame per reworded item)."""
        out: List[lbr.DrawnItem] = []
        for idx, it in enumerate(self.items):
            sys_turn = system_override if system_override is not None else it.system
            lv_item = lv.make_item(
                tokenizer, it.question, it.answer, system=sys_turn, name=f"reword_{idx:03d}_{it.item_id}",
            )
            out.append(lbr.DrawnItem(item=lv_item, frame_id=it.item_id, split="reworded", half=it.half, slots={}))
        return out


def validate_reworded_against_brief(rset: RewordedSet, brief: lbr.Brief) -> None:
    """Verify `rset` belongs to `brief` and is wording-disjoint from `brief`'s practice frames, kept-back frames,
    near misses, and retention batteries."""
    if rset.brief_id != brief.id:
        raise TrialRefused(f'RewordedSet "{rset.name}" is for brief {rset.brief_id[:12]}, not {brief.short()}.')
    guard = lbr.KeptBack([brief], batteries=True)
    # Also guard against every practice frame of `brief`
    prac_frames = list(brief.generator.frames_for("practice"))
    for it in rset.items:
        hit = guard.hit(it.system, it.question, it.answer)
        if hit is not None:
            raise TrialRefused(
                f'Reworded item "{it.item_id}" overlaps existing brief/battery wording ({hit}): '
                "a reworded set must use fresh wording outside the brief's generator frames."
            )
        joined = " ".join(s for s in (it.system, it.question, it.answer) if isinstance(s, str))
        for fr in prac_frames:
            if fr.rendered_in(joined) or fr.near_in(joined):
                raise TrialRefused(
                    f'Reworded item "{it.item_id}" overlaps practice frame "{fr.id}": '
                    "a reworded set must use fresh wording outside the brief's generator frames."
                )


def save_reworded(rset: RewordedSet, path: Path) -> Path:
    """Write a RewordedSet to JSON with its embedded canonical `reworded_id`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**rset.to_dict(), "reworded_id": rset.id}
    write_json_atomic(p, payload, indent=2)
    return p


def load_reworded(path: Path) -> RewordedSet:
    """Load a RewordedSet JSON and refuse if `reworded_id` is absent or tampered."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialRefused(f"Cannot read reworded set {p}: {exc}") from None
    if not isinstance(raw, dict):
        raise TrialRefused(f"Reworded set {p} is not a JSON object.")
    wanted = raw.get("reworded_id")
    if not isinstance(wanted, str) or not wanted:
        raise TrialRefused(f"Reworded set {p} lacks its reworded_id hash.")
    rs = RewordedSet.from_dict(raw)
    if str(wanted) != rs.id:
        raise TrialRefused(f"Reworded set {p} claims reworded_id {str(wanted)[:12]} but hashes to {rs.short()}.")
    return rs


def reworded_uses_in_book(reworded_id: str, book_path: Optional[Path] = None) -> int:
    """Count how many `trial_registration` notes in `book_path` have already spent `reworded_id`."""
    p = Path(book_path) if book_path is not None else lb.book_path()
    if not p.exists():
        return 0
    uses = 0
    for ent in lb.entries(p):
        body = ent.get("body")
        if ent.get("kind") == "note" and isinstance(body, Mapping):
            if body.get("note_type") == "trial_registration" and body.get("reworded_id") == reworded_id:
                uses += 1
    return uses


def starter_reworded_pronoun_pull(brief: lbr.Brief) -> RewordedSet:
    """Owner-written starter reworded set (8 items, 4 half-0 and 4 half-1) completely outside `pronoun_pull`'s
    generator frames, each ending in a subordinating conjunction requiring a third-person subject pronoun."""
    items = (
        # Half 0
        RewordedItem(
            "rw_01", 0,
            "Supply the next clause of this brief scene: Dr. Beatrice Holloway reviewed the clinic ledger past midnight.",
            " Beatrice closed the binder because",
        ),
        RewordedItem(
            "rw_02", 0,
            "Extend this passage in plain prose: Inspector Camille Laurent surveyed the wet cobblestones outside the courthouse.",
            " Camille adjusted her collar while",
        ),
        RewordedItem(
            "rw_03", 0,
            "Write a natural continuation: Captain Simone Ortega checked the barometer on the bridge at dawn.",
            " Simone logged the pressure before",
        ),
        RewordedItem(
            "rw_04", 0,
            "Keep the paragraph going: Professor Ingrid Lindholm stacked the yellowed folios beside the window.",
            " Ingrid dimmed the desk lamp until",
        ),
        # Half 1
        RewordedItem(
            "rw_05", 1,
            "Add the following words: Chef Amara Okafor tasted the reduced broth from a copper spoon.",
            " Amara lowered the flame since",
        ),
        RewordedItem(
            "rw_06", 1,
            "Continue the short anecdote: Engineer Priya Subramanian inspected the turbine housing with a torch.",
            " Priya marked the flange after",
        ),
        RewordedItem(
            "rw_07", 1,
            "Pick up where the text stops: Curator Valeria Rossi unlocked the east cabinet in silence.",
            " Valeria lifted the bronze astrolabe although",
        ),
        RewordedItem(
            "rw_08", 1,
            "Carry the sentence onward: Botanist Freja Lindberg pressed the alpine fern between dry sheets.",
            " Freja labelled the folder before",
        ),
    )
    rset = RewordedSet(
        name="pronoun_pull_reworded_v1",
        brief_id=brief.id,
        brief_name=brief.name,
        items=items,
        note="Owner-written out-of-template narratives for pronoun_pull (4 half-0, 4 half-1; max 5 lifetime uses).",
    )
    validate_reworded_against_brief(rset, brief)
    return rset


# --------------------------------------------------------------------------------------------- registration folder


def _render_rules_md(trial_cfg: Mapping[str, Any], brief: lbr.Brief, bars: lcl.BarSet) -> str:
    """Plain-word house RULES.md written into the registration folder (hashed in SHA256SUMS)."""
    lines = [
        f"# Trial of Record Registration Rules: {trial_cfg['trial_name']}",
        "",
        f"- rule preset: `{trial_cfg['rule_preset']}` (version {trial_cfg['v']})",
        f"- brief: `{brief.name}` (`{brief.id}`)",
        f"- metric: `{brief.metric}` (direction `{brief.direction:+d}`, least effect `{brief.least_effect:g}`)",
        f"- edit id: `{trial_cfg['edit_id']}` (provenance: `{trial_cfg['edit_provenance']}`)",
        f"- dose: `{trial_cfg['main_dose']:g}`",
        f"- bars sha256: `{bars.id}` (alpha `{bars.alpha:g}`, gating batteries: `{', '.join(bars.gating_batteries)}`)",
        f"- selection mode: `{trial_cfg['selection_mode']}`",
        f"- control edits per family: `{trial_cfg['controls_per_family']}` (other_sites_matched + same_site"
        + (" + sham_target" if trial_cfg["selection_mode"] == "declared_rule" else "") + ")",
        f"- fresh target items: `{trial_cfg['n_target']}` (drawn from os.urandom(32) after registration)",
        f"- batch classes: `small = {trial_cfg['rows']['small']}`, `large = {trial_cfg['rows']['large']}`",
        f"- folded weight verification required: `{trial_cfg['folded_required']}` "
        f"(max |folded - per_row| <= max({trial_cfg['folded_tol_floor_mult']:g} * floor, "
        f"{trial_cfg['folded_tol_rel']:g} * max(1, |per_row|), {trial_cfg['folded_tol_min']:g}))",
        f"- reworded item set: `{trial_cfg['reworded_id'] or 'none'}` "
        f"(controls: `{trial_cfg['reworded_controls']}`, lifetime cap: `{MAX_REWORDED_USES}` uses)",
        "",
        "## Pass Rules",
        "",
        "A trial of record runs once from a fresh process with the server stopped. Every verdict is one of",
        "`passed`, `failed`, or `void`:",
        "",
        "1. **`void`**: any reading ended void/refused, the registration SHA256SUMS failed, live base weight",
        "   checksum differed between start and end, `assert_base` failed, an unpacker fallback occurred, or",
        "   folded verification was required and unavailable.",
        "2. **`passed`**: not void, AND ALL of the following hold:",
        "   - Fresh kept-back items clear the house bar `sqrt(F * M)` (with `M >= 4F`), replicate across both",
        "     frame halves (`half 0` and `half 1`), satisfy the reversed-dose check, and beat every required",
        "     control family at `p_fw <= alpha` in BOTH batch classes.",
        "   - Folded weight verification (`folded=\"always\"`) runs in BOTH batch classes, clears the house bar",
        "     and replicates across both halves itself, and agrees with per-row hooks within the registered tolerance.",
        "   - When a reworded item set is registered, it clears the house bar, replicates across both reworded",
        "     halves, beats its control edits, and its folded form agrees with per-row hooks.",
        "   - Every gating retention battery (`neutral_text`, `near_misses`, `labelled_panel`, plus `habitat` if",
        "     gating) stays within the registered `BarSet`.",
        "3. **`failed`**: cleanly measured (not void), but one or more of the gates above did not pass.",
        "",
        "Registration folders contain manifests and hashes only; weight payloads (`*.safetensors`, `*.gguf`)",
        "are forbidden here.",
    ]
    return "\n".join(lines) + "\n"


def _write_sha256sums(folder: Path) -> Tuple[str, List[str]]:
    """Write standard `SHA256SUMS` over every regular file in `folder` except `SHA256SUMS` itself.
    Returns `(sha256_of_sums_file, sorted_filenames)`."""
    names = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{_sha_file(folder / nm)}  {nm}\n" for nm in names]
    content = "".join(lines).encode("ascii")
    tmp = folder / "SHA256SUMS.tmp"
    tmp.write_bytes(content)
    os.replace(tmp, folder / "SHA256SUMS")
    return _sha_bytes(content), names


def register_trial(
    prereg_dir: Path,
    brief: lbr.Brief,
    bars: lcl.BarSet,
    edit: we.Edit,
    *,
    trial_name: str,
    main_dose: float = 1.0,
    selection_mode: str = "free_pick",
    declared_rule: Optional[str] = None,
    sham_edits: Optional[Sequence[we.Edit]] = None,
    reworded: Optional[RewordedSet] = None,
    controls_per_family: int = TRIAL_CONTROLS,
    reworded_controls: Optional[int] = None,
    n_target: int = 16,
    n_neutral: int = 8,
    n_near: int = 8,
    n_panel: int = 8,
    rows: Optional[Mapping[str, int]] = None,
    written_rows: int = 12,
    match_tolerance: float = lv.DEFAULT_MATCH_TOL,
    folded_required: bool = True,
    folded_tol_floor_mult: float = DEFAULT_FOLDED_FLOOR_MULT,
    folded_tol_rel: float = DEFAULT_FOLDED_REL_TOL,
    folded_tol_min: float = DEFAULT_FOLDED_MIN_TOL,
    saved_edit_dir: Optional[Path] = None,
    source_claim: Optional[Mapping[str, Any]] = None,
    forecast: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a hash-verified house registration folder (`prereg/`) for a Phase 5 Trial of Record.

    Holds `trial.json`, `brief.json`, `bars.json`, `edit_manifest.json` (provenance manifest with `"rebuilt"` vs
    `"carried"` per term and payload hash ONLY — never raw weights), optional `reworded.json`, `RULES.md`,
    `repo_commit`, `registered_at`, and `SHA256SUMS`. Refuses to overwrite an existing non-empty folder.
    """
    folder = Path(prereg_dir).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise TrialRefused(f"Registration folder {folder} already exists and is not empty: never overwrite a registration.")
    if not isinstance(trial_name, str) or not trial_name.strip():
        raise TrialRefused("trial_name must be a non-empty string.")
    if bars.brief_id != brief.id:
        raise TrialRefused(f"BarSet is for brief {bars.brief_id[:12]}, not {brief.short()}.")
    if selection_mode not in ("free_pick", "declared_rule"):
        raise TrialRefused(f'selection_mode must be "free_pick" or "declared_rule"; got {selection_mode!r}.')
    if selection_mode == "declared_rule":
        if not sham_edits:
            raise TrialRefused('selection_mode="declared_rule" requires sham_edits.')
        if not (isinstance(declared_rule, str) and declared_rule.strip()):
            raise TrialRefused('selection_mode="declared_rule" requires a non-empty declared_rule sentence.')
    elif sham_edits:
        raise TrialRefused('Sham edits are registered only with selection_mode="declared_rule".')
    if not math.isfinite(float(main_dose)) or float(main_dose) == 0.0:
        raise TrialRefused(f"main_dose must be a finite non-zero number; got {main_dose!r}.")
    if not math.isfinite(float(match_tolerance)) or float(match_tolerance) < 1.0:
        raise TrialRefused(f"match_tolerance must be finite and >= 1.0; got {match_tolerance!r}.")
    min_target = 2 * int(bars.min_frames_per_half)
    for what, val, lo in (
        ("controls_per_family", controls_per_family, 1),
        ("n_target", n_target, min_target),
        ("n_neutral", n_neutral, 1),
        ("n_near", n_near, 1),
        ("n_panel", n_panel, 1),
    ):
        if isinstance(val, bool) or not isinstance(val, int) or val < lo:
            raise TrialRefused(f"{what} must be a whole number >= {lo}; got {val!r}.")

    lcl._check_edit_allowed_by_brief(edit, brief)
    if not any(t.kind in ("cols", "rows") for t in edit.terms):
        raise TrialRefused("A trial of record requires at least one column or row term for matched controls.")
    if reworded is not None:
        validate_reworded_against_brief(reworded, brief)

    prov = we.edit_provenance(edit, saved_dir=saved_edit_dir)
    sham_provs = [we.edit_provenance(se) for se in (sham_edits or ())]
    rows_cfg = {
        "small": int((rows or {}).get("small", lv.ROWS_SMALL)),
        "large": int((rows or {}).get("large", lv.ROWS_LARGE)),
    }
    min_small = max(5, 2 + len(lv.NULL_E) + (3 if brief.direction != 0 else 2))
    if not (min_small <= rows_cfg["small"] <= 12 and 16 <= rows_cfg["large"] <= 64):
        raise TrialRefused(
            f"Batch classes must satisfy {min_small} <= small <= 12 and 16 <= large <= 64; got {rows_cfg}."
        )
    rw_ctls = int(reworded_controls if reworded_controls is not None else min(int(controls_per_family), 19))
    if rw_ctls < 1:
        raise TrialRefused(f"reworded_controls must be >= 1; got {rw_ctls}.")
    commit, dirty = git_state(ROOT_DIR)
    reg_at = _utc_now()

    trial_cfg: Dict[str, Any] = {
        "v": TRIAL_VERSION,
        "rule_preset": TRIAL_RULE_PRESET,
        "trial_name": trial_name.strip(),
        "brief_id": brief.id,
        "brief_name": brief.name,
        "bars_sha256": bars.id,
        "edit_id": edit.id,
        "edit_linears": edit.linears(),
        "edit_provenance": prov["provenance"],
        "all_rebuilt": bool(prov["all_rebuilt"]),
        "payload_sha256": prov["payload_sha256"],
        "base": dict(edit.base),
        "main_dose": float(main_dose),
        "selection_mode": selection_mode,
        "declared_rule": " ".join(declared_rule.split())[:600] if declared_rule else None,
        "sham_edit_ids": [sp["edit_id"] for sp in sham_provs],
        "sham_manifests": sham_provs,
        "controls_per_family": int(controls_per_family),
        "reworded_id": reworded.id if reworded is not None else None,
        "reworded_controls": rw_ctls,
        "n_target": int(n_target),
        "n_neutral": int(n_neutral),
        "n_near": int(n_near),
        "n_panel": int(n_panel),
        "rows": rows_cfg,
        "written_rows": int(written_rows),
        "match_tolerance": float(match_tolerance),
        "folded_required": bool(folded_required),
        "folded_tol_floor_mult": float(folded_tol_floor_mult),
        "folded_tol_rel": float(folded_tol_rel),
        "folded_tol_min": float(folded_tol_min),
        "source_claim": dict(source_claim) if source_claim else None,
        "forecast": dict(forecast or {}),
        "registered_at": reg_at,
        "repo_commit": commit,
        "git_dirty": dirty,
    }

    folder.mkdir(parents=True, exist_ok=True)
    lbr.save_brief(brief, folder / "brief.json")
    lcl.save_bars(bars, folder / "bars.json")
    write_json_atomic(folder / "edit_manifest.json", prov, indent=2)
    if reworded is not None:
        save_reworded(reworded, folder / "reworded.json")
    write_json_atomic(folder / "trial.json", trial_cfg, indent=2)
    (folder / "RULES.md").write_text(_render_rules_md(trial_cfg, brief, bars), encoding="utf-8")
    (folder / "repo_commit").write_text(f"{commit}\n", encoding="utf-8")
    (folder / "registered_at").write_text(f"{reg_at}\n", encoding="utf-8")
    trial_sha, files = _write_sha256sums(folder)
    return {
        "prereg_dir": str(folder),
        "trial_sha256": trial_sha,
        "trial": trial_cfg,
        "files": files,
    }


def verify_registration(prereg_dir: Path) -> Dict[str, Any]:
    """Verify every file, hash, policy constraint, and cross-reference in a Phase 5 registration folder.
    Refuses tampered files, missing/extra files, or any `*.safetensors` / `*.gguf` weight payload inside `prereg/`."""
    folder = Path(prereg_dir).resolve()
    if not folder.is_dir():
        raise TrialRefused(f"Registration folder does not exist: {folder}")

    # Arbiter Row 32: registration folders hold manifests and payload hashes only, NEVER weight files
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in FORBIDDEN_PREREG_SUFFIXES:
            raise TrialRefused(
                f"Registration folder {folder} contains forbidden weight file {p.name}: "
                "registration folders hold manifests and payload hashes only."
            )

    sums_path = folder / "SHA256SUMS"
    if not sums_path.is_file():
        raise TrialRefused(f"Registration folder {folder} lacks SHA256SUMS.")
    raw_sums = sums_path.read_bytes()
    trial_sha = _sha_bytes(raw_sums)
    expected: Dict[str, str] = {}
    for lineno, line in enumerate(raw_sums.decode("ascii", errors="strict").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise TrialRefused(f"SHA256SUMS line {lineno} is malformed.")
        digest, fname = parts[0].lower(), parts[1]
        if "/" in fname or "\\" in fname or fname in ("..", ".", "SHA256SUMS"):
            raise TrialRefused(f"SHA256SUMS line {lineno} names an unsafe file: {fname!r}.")
        expected[fname] = digest

    actual_files = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    missing = sorted(set(expected) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected))
    if missing:
        raise TrialRefused(f"Registration folder {folder} is missing file(s) listed in SHA256SUMS: {missing}.")
    if extra:
        raise TrialRefused(f"Registration folder {folder} has unregistered extra file(s): {extra}.")
    for fname, want in sorted(expected.items()):
        got = _sha_file(folder / fname)
        if got != want:
            raise TrialRefused(
                f"Registration file {fname} failed SHA256SUMS check (got {got[:12]}, expected {want[:12]})."
            )

    for req in ("trial.json", "brief.json", "bars.json", "edit_manifest.json", "RULES.md", "repo_commit", "registered_at"):
        if req not in expected:
            raise TrialRefused(f"Registration folder {folder} lacks required file {req}.")

    try:
        trial_cfg = json.loads((folder / "trial.json").read_text(encoding="utf-8"))
        edit_man = json.loads((folder / "edit_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialRefused(f"Cannot parse registration JSON in {folder}: {exc}") from None

    brief = lbr.load_brief(folder / "brief.json")
    bars = lcl.load_bars(folder / "bars.json")
    if trial_cfg.get("brief_id") != brief.id:
        raise TrialRefused("trial.json brief_id does not match brief.json.")
    if trial_cfg.get("bars_sha256") != bars.id:
        raise TrialRefused("trial.json bars_sha256 does not match bars.json.")
    if bars.brief_id != brief.id:
        raise TrialRefused("bars.json brief_id does not match brief.json.")
    if edit_man.get("edit_id") != trial_cfg.get("edit_id"):
        raise TrialRefused("edit_manifest.json edit_id does not match trial.json.")
    if edit_man.get("payload_sha256") != trial_cfg.get("payload_sha256"):
        raise TrialRefused("edit_manifest.json payload_sha256 does not match trial.json.")

    reworded: Optional[RewordedSet] = None
    if trial_cfg.get("reworded_id") is not None:
        if not (folder / "reworded.json").is_file():
            raise TrialRefused("trial.json names a reworded_id but reworded.json is absent.")
        reworded = load_reworded(folder / "reworded.json")
        if reworded.id != trial_cfg["reworded_id"]:
            raise TrialRefused("reworded.json hash does not match trial.json reworded_id.")
        validate_reworded_against_brief(reworded, brief)
    elif (folder / "reworded.json").is_file():
        raise TrialRefused("reworded.json is present but trial.json reworded_id is null.")

    return {
        "ok": True,
        "prereg_dir": str(folder),
        "trial_sha256": trial_sha,
        "trial": trial_cfg,
        "brief": brief,
        "bars": bars,
        "edit_manifest": edit_man,
        "reworded": reworded,
        "files": actual_files,
    }


def load_registered_edit(
    h: lv.Handle,
    edit_manifest: Mapping[str, Any],
    *,
    payload_dir: Optional[Path] = None,
) -> we.Edit:
    """Reconstruct and pin-verify the registered `Edit` against `h`:
    - If `all_rebuilt` is True: rebuilds directly from `rebuilt_recipe` and verifies `edit.id == edit_manifest['edit_id']`.
    - If any term is `"carried"`: loads from `payload_dir` (or `we.edits_dir() / edit_id`), verifies `payload_sha256`
      and `edit.id`, and refuses any mismatch."""
    want_id = str(edit_manifest["edit_id"])
    if edit_manifest.get("all_rebuilt"):
        recipe = edit_manifest.get("rebuilt_recipe")
        if not isinstance(recipe, Mapping):
            raise TrialRefused("edit_manifest claims all_rebuilt but carries no rebuilt_recipe.")
        specs = we.from_recipe(dict(recipe), Path("."))
        if isinstance(specs, (str, Path)):
            raise TrialRefused("rebuilt_recipe must be inline terms, not a saved pointer.")
        edit = we.build(h.lm, specs, model_id=h.model_id, weights_id=h.weights_id, note=str(edit_manifest.get("note") or ""))
    else:
        where = Path(payload_dir) if payload_dir is not None else want_id
        folder = we._find_saved(where)
        want_p = edit_manifest.get("payload_sha256")
        pfile = folder / "payload.safetensors"
        if want_p and pfile.is_file():
            got_p = _sha_file(pfile)
            # payload_sha256 may be the file sha256 (when registered with saved_edit_dir) or the carried-arrays digest
            if got_p != want_p:
                # Also allow matching the deterministic carried-arrays digest
                edit_tmp = we.load(folder, h.lm, model_id=h.model_id, weights_id=h.weights_id)
                prov_tmp = we.edit_provenance(edit_tmp)
                if prov_tmp["payload_sha256"] != want_p:
                    raise TrialRefused(
                        f"Payload {pfile} sha256 ({got_p[:12]}) does not match registered payload_sha256 ({want_p[:12]})."
                    )
                edit = edit_tmp
            else:
                edit = we.load(folder, h.lm, model_id=h.model_id, weights_id=h.weights_id)
        else:
            edit = we.load(folder, h.lm, model_id=h.model_id, weights_id=h.weights_id)
    if edit.id != want_id:
        raise TrialRefused(
            f"Rebuilt edit hashes to {edit.id[:12]}, not registered edit_id {want_id[:12]} "
            "(the loaded weights or model may differ from the registration's base)."
        )
    return edit


# --------------------------------------------------------------------------------------------- folded extraction & judge


def extract_folded_class(
    result: Mapping[str, Any],
    drawn: Sequence[lbr.DrawnItem],
    cls_name: str,
    metric: str,
) -> Dict[str, Any]:
    """Extract folded-twin per-item effects and folded-vs-per-row gaps from a `lv.read(..., folded='always')` result."""
    by_name = {d.item.name: d for d in drawn}
    folded_effects: List[Dict[str, Any]] = []
    gaps_abs: List[float] = []
    between_kls: List[float] = []
    same_bits_n = 0
    taken_n = 0
    rows_id_all = True
    why_missing: Optional[str] = None

    for it_ent in result.get("items") or ():
        if it_ent.get("status") != "ok":
            continue
        nm = (it_ent.get("item") or {}).get("name")
        d = by_name.get(nm)
        if d is None:
            continue
        cb = (it_ent.get("classes") or {}).get(cls_name)
        if not isinstance(cb, Mapping):
            continue
        fl = cb.get("folded")
        if not isinstance(fl, Mapping) or not fl.get("taken"):
            why_missing = (fl or {}).get("why") if isinstance(fl, Mapping) else "folded twin missing"
            continue
        taken_n += 1
        if fl.get("rows_identical") is False:
            rows_id_all = False
        if fl.get("same_bits"):
            same_bits_n += 1
        m_val = _clean_float(((fl.get("metrics") or {}).get(metric) or {}).get("mean"))
        if m_val is not None:
            folded_effects.append({"frame_id": d.frame_id, "half": d.half, "item_id": d.item.id, "value": m_val})
        g_val = _clean_float(((fl.get("gap") or {}).get(metric) or {}).get("abs"))
        if g_val is not None:
            gaps_abs.append(g_val)
        b_kl = _clean_float(fl.get("between_kl_mean"))
        if b_kl is not None:
            between_kls.append(b_kl)

    return {
        "available": taken_n > 0 and len(folded_effects) > 0,
        "why": why_missing if taken_n == 0 else None,
        "taken_count": taken_n,
        "expected_count": len(drawn),
        "rows_identical_all": rows_id_all,
        "same_bits_count": same_bits_n,
        "item_effects": folded_effects,
        "max_abs_gap": max(gaps_abs) if gaps_abs else None,
        "mean_abs_gap": float(np.mean(gaps_abs)) if gaps_abs else None,
        "between_kl_max": max(between_kls) if between_kls else None,
    }


def judge_folded(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    fresh_judged: Mapping[str, Any],
    folded_by_class: Mapping[str, Mapping[str, Any]],
    *,
    folded_required: bool = True,
    tol_floor_mult: float = DEFAULT_FOLDED_FLOOR_MULT,
    tol_rel: float = DEFAULT_FOLDED_REL_TOL,
    tol_min: float = DEFAULT_FOLDED_MIN_TOL,
) -> Dict[str, Any]:
    """Judge folded weight agreement and independent folded clearing across every required batch class."""
    req_classes = ("written",) if brief.metric == "written_loop" else ("small", "large")
    reasons: List[str] = []
    cls_out: Dict[str, Any] = {}
    all_ok = True
    any_unavailable = False

    for cls_name in req_classes:
        fd = folded_by_class.get(cls_name)
        if not fd or not fd.get("available"):
            any_unavailable = True
            why = (fd or {}).get("why") or "folded twin was not taken"
            cls_out[cls_name] = {"available": False, "ok": not folded_required, "why": why}
            if folded_required:
                all_ok = False
                reasons.append(f'{cls_name} class folded form unavailable: {why}')
            continue
        fstats = lcl.frame_statistics(fd["item_effects"], direction=brief.direction)
        fresh_cls = (fresh_judged.get("classes") or {}).get(cls_name) or {}
        fresh_mean = _clean_float((fresh_cls.get("frames") or {}).get("mean"))
        floor = _clean_float((fresh_cls.get("house_bar") or {}).get("floor"))
        hb = lcl.house_bar(floor, brief.least_effect)
        bar = hb["bar"]

        f_dir = fstats["directed_mean"]
        h0_dir = (fstats["half_0"] or {}).get("directed_mean")
        h1_dir = (fstats["half_1"] or {}).get("directed_mean")
        clears = (not hb["floor_too_loud"]) and f_dir is not None and bar is not None and f_dir >= bar
        halves_ok = (
            fstats["half_0"]["n_frames"] >= bars.min_frames_per_half
            and fstats["half_1"]["n_frames"] >= bars.min_frames_per_half
            and h0_dir is not None and h1_dir is not None and bar is not None
            and h0_dir >= bars.half_bar_fraction * bar
            and h1_dir >= bars.half_bar_fraction * bar
        )

        tol = max(
            float(tol_floor_mult) * (float(floor) if floor is not None else 0.0),
            float(tol_rel) * max(1.0, abs(float(fresh_mean)) if fresh_mean is not None else 1.0),
            float(tol_min),
        )
        max_gap = _clean_float(fd.get("max_abs_gap"))
        mean_diff = (abs(float(fstats["mean"]) - float(fresh_mean))
                     if (fstats["mean"] is not None and fresh_mean is not None) else None)
        gap_ok = (
            bool(fd.get("rows_identical_all", True))
            and (max_gap is None or max_gap <= tol)
            and (mean_diff is not None and mean_diff <= tol)
        )
        cls_ok = bool(clears and halves_ok and gap_ok)
        if not cls_ok:
            all_ok = False
            if not gap_ok:
                reasons.append(
                    f"{cls_name} folded form disagreed with per-row hooks "
                    f"(mean_diff={mean_diff}, max_abs_gap={max_gap}, tol={tol:.4f})."
                )
            if not clears:
                reasons.append(f"{cls_name} folded directed mean ({f_dir}) did not clear house bar ({bar}).")
            if not halves_ok:
                reasons.append(f"{cls_name} folded form did not replicate across both halves (h0={h0_dir}, h1={h1_dir}).")
        cls_out[cls_name] = {
            "available": True,
            "ok": cls_ok,
            "frames": fstats,
            "house_bar": hb,
            "clears_bar": clears,
            "halves_ok": halves_ok,
            "gap_ok": gap_ok,
            "tolerance": tol,
            "mean_diff_vs_per_row": mean_diff,
            "max_abs_gap": max_gap,
            "mean_abs_gap": _clean_float(fd.get("mean_abs_gap")),
            "between_kl_max": _clean_float(fd.get("between_kl_max")),
            "same_bits_count": int(fd.get("same_bits_count", 0)),
            "taken_count": int(fd.get("taken_count", 0)),
            "rows_identical_all": bool(fd.get("rows_identical_all", True)),
        }

    status = "unavailable" if any_unavailable and folded_required else ("ok" if all_ok else "failed")
    return {
        "required": bool(folded_required),
        "available": not any_unavailable,
        "ok": bool(all_ok),
        "status": status,
        "reasons": reasons,
        "classes": cls_out,
    }


def judge_reworded(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    reworded_classes: Mapping[str, Mapping[str, Any]],
    reworded_folded: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    folded_required: bool = True,
    tol_floor_mult: float = DEFAULT_FOLDED_FLOOR_MULT,
    tol_rel: float = DEFAULT_FOLDED_REL_TOL,
    tol_min: float = DEFAULT_FOLDED_MIN_TOL,
) -> Dict[str, Any]:
    """Judge an owner-written reworded item set across both batch classes (effect, halves, controls, folded)."""
    req_classes = ("written",) if brief.metric == "written_loop" else ("small", "large")
    reasons: List[str] = []
    cls_out: Dict[str, Any] = {}
    all_ok = True

    for cls_name in req_classes:
        cdata = reworded_classes.get(cls_name)
        if not cdata:
            all_ok = False
            reasons.append(f'Reworded reading missing required class "{cls_name}".')
            cls_out[cls_name] = {"ok": False}
            continue
        fstats = lcl.frame_statistics(cdata.get("item_effects") or [], direction=brief.direction)
        floor = _clean_float(cdata.get("floor"))
        hb = lcl.house_bar(floor, brief.least_effect)
        bar = hb["bar"]
        dir_m = fstats["directed_mean"]
        h0_dir = (fstats["half_0"] or {}).get("directed_mean")
        h1_dir = (fstats["half_1"] or {}).get("directed_mean")
        clears = (not hb["floor_too_loud"]) and dir_m is not None and bar is not None and dir_m >= bar
        halves_ok = (
            fstats["half_0"]["n_frames"] >= bars.min_frames_per_half
            and fstats["half_1"]["n_frames"] >= bars.min_frames_per_half
            and h0_dir is not None and h1_dir is not None and bar is not None
            and h0_dir >= bars.half_bar_fraction * bar
            and h1_dir >= bars.half_bar_fraction * bar
        )
        # Reworded set tests generalisation outside generator frames (single test m=1 against its own controls)
        ctls_in = cdata.get("controls") or {}
        ctl_eval: Dict[str, Any] = {}
        ctl_ok = True
        for fam, vals in ctls_in.items():
            if vals:
                cs = lcl.control_standing(fstats["mean"], vals, max_fresh_readings=1, alpha=bars.alpha)
                ctl_eval[fam] = cs
                if not cs["passes_alpha"]:
                    ctl_ok = False
                    reasons.append(
                        f'Reworded {cls_name} failed "{fam}" controls '
                        f'(at_least={cs["at_least"]}/{cs["count"]}, p_fw={cs["p_fw"]}, alpha={bars.alpha}).'
                    )
        if not clears:
            reasons.append(f"Reworded {cls_name} directed mean ({dir_m}) did not clear house bar ({bar}).")
        if not halves_ok:
            reasons.append(f"Reworded {cls_name} did not replicate across both halves (h0={h0_dir}, h1={h1_dir}).")

        cls_ok = bool(clears and halves_ok and ctl_ok)
        if not cls_ok:
            all_ok = False
        cls_out[cls_name] = {
            "ok": cls_ok,
            "frames": fstats,
            "house_bar": hb,
            "clears_bar": clears,
            "halves_ok": halves_ok,
            "controls": ctl_eval,
        }

    folded_eval = None
    if reworded_folded is not None:
        folded_eval = judge_folded(
            brief, bars, {"classes": cls_out}, reworded_folded,
            folded_required=folded_required, tol_floor_mult=tol_floor_mult, tol_rel=tol_rel, tol_min=tol_min,
        )
        if not folded_eval["ok"]:
            all_ok = False
            reasons.extend(f"Reworded {r}" for r in folded_eval["reasons"])

    return {
        "ok": bool(all_ok),
        "status": "ok" if all_ok else "failed",
        "reasons": reasons,
        "classes": cls_out,
        "folded": folded_eval,
    }


def judge_trial(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    fresh_judged: Mapping[str, Any],
    *,
    folded_eval: Mapping[str, Any],
    reworded_eval: Optional[Mapping[str, Any]] = None,
    base_checksum_unchanged: bool = True,
    base_after: bool = True,
    unpacker_fallbacks: int = 0,
    void_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply the registered Trial of Record rule -> verdict strictly in `("passed", "failed", "void")`."""
    reasons: List[str] = []
    if void_reason:
        return {"verdict": "void", "reasons": [str(void_reason)], "fresh_verdict": fresh_judged.get("verdict"),
                "folded": dict(folded_eval), "reworded": dict(reworded_eval) if reworded_eval is not None else None}
    if not base_after or not base_checksum_unchanged:
        return {
            "verdict": "void",
            "reasons": ["Base weight integrity check failed (base_after or base_checksum changed during trial)."],
            "fresh_verdict": fresh_judged.get("verdict"),
            "folded": dict(folded_eval),
            "reworded": dict(reworded_eval) if reworded_eval is not None else None,
        }
    if int(unpacker_fallbacks) != 0:
        return {
            "verdict": "void",
            "reasons": [f"Compiled unpacker fell back {int(unpacker_fallbacks)} time(s) during the trial."],
            "fresh_verdict": fresh_judged.get("verdict"),
            "folded": dict(folded_eval),
            "reworded": dict(reworded_eval) if reworded_eval is not None else None,
        }
    fv = str(fresh_judged.get("verdict") or "void")
    if fv == "void":
        return {
            "verdict": "void",
            "reasons": list(fresh_judged.get("reasons") or ["Fresh target reading ended void."]),
            "fresh_verdict": fv,
            "folded": dict(folded_eval),
            "reworded": dict(reworded_eval) if reworded_eval is not None else None,
        }
    if folded_eval.get("required") and not folded_eval.get("available"):
        return {
            "verdict": "void",
            "reasons": list(folded_eval.get("reasons") or ["Required folded weight verification was unavailable."]),
            "fresh_verdict": fv,
            "folded": dict(folded_eval),
            "reworded": dict(reworded_eval) if reworded_eval is not None else None,
        }

    # Fresh kept-back gate: must earn "lead" (or "lead, matched by a cheaper knob" — cheaper knob cuts claim down,
    # recorded explicitly on the Trial Sheet, without voiding the measurement)
    fresh_ok = fv in ("lead", "lead, matched by a cheaper knob")
    if not fresh_ok:
        reasons.append(f'Fresh kept-back gate returned "{fv}": ' + "; ".join(fresh_judged.get("reasons") or ()))

    folded_ok = bool(folded_eval.get("ok"))
    if not folded_ok:
        reasons.extend(folded_eval.get("reasons") or ["Folded form check failed."])

    reworded_ok = True
    if reworded_eval is not None:
        reworded_ok = bool(reworded_eval.get("ok"))
        if not reworded_ok:
            reasons.extend(reworded_eval.get("reasons") or ["Reworded item set check failed."])

    matched_knob = (fresh_judged.get("cheaper_knobs") or {}).get("matched_by")
    verdict = "passed" if (fresh_ok and folded_ok and reworded_ok) else "failed"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "fresh_verdict": fv,
        "matched_by_cheaper_knob": matched_knob,
        "folded": dict(folded_eval),
        "reworded": dict(reworded_eval) if reworded_eval is not None else None,
    }


# --------------------------------------------------------------------------------------------- Trial Sheet & sentence


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "none"
    return f"{float(x):.{digits}f}"


def trial_claim_sentence(
    brief: lbr.Brief,
    trial_cfg: Mapping[str, Any],
    fresh_judged: Mapping[str, Any],
    trial_judged: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Optional[str]:
    """Bounded sentence issued ONLY when a Trial of Record earns `"passed"`."""
    if trial_judged.get("verdict") != "passed":
        return None
    primary = "written" if brief.metric == "written_loop" else "large"
    lg = (fresh_judged.get("classes") or {}).get(primary) or {}
    fr = lg.get("frames") or {}
    ci = fr.get("ci95") or [None, None]
    ctls = lg.get("controls") or {}
    osm = ctls.get("other_sites_matched") or {}
    ss = ctls.get("same_site") or {}
    fl_cls = ((trial_judged.get("folded") or {}).get("classes") or {}).get(primary) or {}
    fl_mean = ((fl_cls.get("frames") or {}).get("mean")) if fl_cls.get("available") else None
    fl_gap = fl_cls.get("max_abs_gap")
    rw = trial_judged.get("reworded")
    rw_part = ""
    if isinstance(rw, Mapping):
        rw_cls = (rw.get("classes") or {}).get(primary) or {}
        rw_mean = (rw_cls.get("frames") or {}).get("mean")
        rw_part = f"; on the owner-written reworded set ({trial_cfg.get('reworded_id', '')[:12]}) it moved {brief.metric} by {_fmt(rw_mean)}"
    knob = trial_judged.get("matched_by_cheaper_knob")
    knob_part = f"; matched by cheaper knob {knob}" if knob else "; no author prompt arm matched it"
    return (
        f"On {meta.get('date')}, {meta.get('device')}, {meta.get('unpacker')}, in a fresh-process trial of record "
        f"(registration {str(meta.get('trial_sha256') or '')[:12]}, commit {meta.get('repo_commit', 'unknown')[:12]}): "
        f"edit {str(trial_cfg.get('edit_id') or '')[:12]} ({trial_cfg.get('edit_provenance')}) to "
        f"{', '.join(trial_cfg.get('edit_linears') or ())} of base {meta.get('weights_id')} moved {brief.metric} on "
        f"{fr.get('n_frames')} fresh kept-back frames by {_fmt(fr.get('mean'))} "
        f"(95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}]), verified in folded weights ({_fmt(fl_mean)}, max |gap| {_fmt(fl_gap)})"
        f"{rw_part}; {osm.get('at_least')} of {osm.get('count')} matched and {ss.get('at_least')} of {ss.get('count')} "
        f"same-site control edits moved as much{knob_part} (no search over cheaper knobs; not compared with a trained "
        f"adapter). Applies only to this base checkpoint and these items; an exported adapter on another quantisation "
        f"must be re-measured there."
    )


def render_trial_sheet(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    trial_cfg: Mapping[str, Any],
    fresh_judged: Mapping[str, Any],
    trial_judged: Mapping[str, Any],
    meta: Mapping[str, Any],
    sentence: Optional[str],
) -> str:
    """Harness-authored Trial Sheet showing practice, fresh, folded, and reworded results side by side."""
    out: List[str] = [
        f"=== TRIAL OF RECORD: {trial_cfg['trial_name']} ===",
        f"verdict: {trial_judged['verdict']}",
        f"brief: {brief.name} ({brief.short()}) | metric: {brief.metric} (dir {brief.direction:+d}, M={brief.least_effect:g})",
        f"registration: {str(meta.get('trial_sha256') or '')[:16]} | commit: {meta.get('repo_commit')} (dirty={meta.get('git_dirty')})",
        f"edit: {str(trial_cfg['edit_id'])[:16]} ({trial_cfg['edit_provenance']}) | linears: {', '.join(trial_cfg.get('edit_linears') or ())}",
        f"dose: {trial_cfg['main_dose']:g} | bars: {bars.short()} | book: {meta.get('book_name')}",
        f"base integrity: checksum_unchanged={meta.get('base_checksum_unchanged')} "
        f"(linears={meta.get('base_linears_checked')}, overlays=0, unpacker_fallbacks={meta.get('unpacker_fallbacks')})",
    ]
    src = trial_cfg.get("source_claim")
    if isinstance(src, Mapping):
        out.append(
            f"promoted from claim: book line {src.get('book_seq')} (verdict={src.get('verdict')}, "
            f"practice_best={_fmt(_clean_float(src.get('practice_best')))})"
        )
    if trial_judged.get("matched_by_cheaper_knob"):
        out.append(f"cheaper knob note: matched by {trial_judged['matched_by_cheaper_knob']} (no search over cheaper knobs)")

    out.append("")
    out.append("--- Fresh kept-back & folded comparison ---")
    folded_classes = (trial_judged.get("folded") or {}).get("classes") or {}
    for cls_name, cdata in (fresh_judged.get("classes") or {}).items():
        fr = cdata["frames"]
        hb = cdata["house_bar"]
        ci = fr["ci95"]
        ctls = cdata["controls"]
        osm = ctls.get("other_sites_matched") or {}
        ss = ctls.get("same_site") or {}
        out.append(
            f"  [{cls_name}] per-row: mean {_fmt(fr['mean'])} (95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}]) | "
            f"h0 {_fmt(fr['half_0']['mean'])}, h1 {_fmt(fr['half_1']['mean'])} | "
            f"floor {_fmt(hb['floor'])}, bar {_fmt(hb['bar'])} | "
            f"matched {osm.get('at_least')}/{osm.get('count')} (p_fw {_fmt(osm.get('p_fw'), 3)}), "
            f"same-site {ss.get('at_least')}/{ss.get('count')} (p_fw {_fmt(ss.get('p_fw'), 3)})"
        )
        fc = folded_classes.get(cls_name) or {}
        if fc.get("available"):
            ffr = fc["frames"]
            out.append(
                f"  [{cls_name}] folded : mean {_fmt(ffr['mean'])} | h0 {_fmt(ffr['half_0']['mean'])}, "
                f"h1 {_fmt(ffr['half_1']['mean'])} | max |gap| {_fmt(fc.get('max_abs_gap'))} "
                f"(tol {_fmt(fc.get('tolerance'))}) | same_bits {fc.get('same_bits_count')}/{fc.get('taken_count')} | "
                f"ok={fc.get('ok')}"
            )
        else:
            out.append(f"  [{cls_name}] folded : unavailable ({fc.get('why')})")

    rw = trial_judged.get("reworded")
    if isinstance(rw, Mapping):
        out.append("")
        out.append(f"--- Owner-written reworded set ({str(trial_cfg.get('reworded_id') or '')[:12]}, "
                   f"use {meta.get('reworded_use_index')}/{MAX_REWORDED_USES}) ---")
        rw_fl_classes = ((rw.get("folded") or {}).get("classes") or {})
        for cls_name, rc in (rw.get("classes") or {}).items():
            fr = rc.get("frames") or {}
            out.append(
                f"  [{cls_name}] reworded per-row: mean {_fmt(fr.get('mean'))} | "
                f"h0 {_fmt((fr.get('half_0') or {}).get('mean'))}, h1 {_fmt((fr.get('half_1') or {}).get('mean'))} | "
                f"ok={rc.get('ok')}"
            )
            rfc = rw_fl_classes.get(cls_name) or {}
            if rfc.get("available"):
                out.append(
                    f"  [{cls_name}] reworded folded : mean {_fmt((rfc.get('frames') or {}).get('mean'))} | "
                    f"max |gap| {_fmt(rfc.get('max_abs_gap'))} | ok={rfc.get('ok')}"
                )

    out.append("")
    out.append("--- Retention batteries ---")
    for bname, bv in (fresh_judged.get("batteries") or {}).items():
        gating = "gating" if bv.get("gating") else "info"
        if bv.get("available") is False:
            out.append(f"  {bname} ({gating}): not run")
        else:
            st = "passed" if bv.get("passed") else "failed"
            extra = []
            if bv.get("mean_kl") is not None:
                extra.append(f"mean_kl={_fmt(bv['mean_kl'], 5)}")
            if bv.get("abs_effect") is not None:
                extra.append(f"|effect|={_fmt(bv['abs_effect'], 4)}")
            if bv.get("own_drop") is not None:
                extra.append(f"own_drop={_fmt(bv['own_drop'], 4)}")
            out.append(f"  {bname} ({gating}): {st}" + (f" ({', '.join(extra)})" if extra else ""))

    if trial_judged.get("reasons"):
        out.append("")
        out.append("--- Gate notes ---")
        for r in trial_judged["reasons"]:
            out.append(f"  - {r}")

    prior = meta.get("prior_verdicts") or []
    out.append("")
    out.append(f"earlier fresh readings on this brief ({len(prior)}):")
    for pv in prior:
        out.append(
            f"  line {pv.get('seq')}: #{pv.get('reading_index')} edit {str(pv.get('edit_id') or '')[:12]} -> {pv.get('verdict')}"
        )

    if sentence:
        out.append("")
        out.append("--- Bounded record sentence ---")
        out.append(sentence)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------- fresh-process execution


def _trial_seen_in_book(trial_sha256: str, book_path: Path) -> bool:
    if not book_path.exists():
        return False
    for ent in lb.entries(book_path):
        body = ent.get("body")
        if ent.get("kind") == "note" and isinstance(body, Mapping):
            if body.get("note_type") in ("trial_registration", "trial_of_record") and body.get("trial_sha256") == trial_sha256:
                return True
    return False


def run_trial_of_record(
    h: lv.Handle,
    prereg_dir: Path,
    *,
    out_dir: Path,
    payload_dir: Optional[Path] = None,
    book_path: Optional[Path] = None,
    require_book_calibration: bool = False,
    _seed_override: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Execute one registered Phase 5 Trial of Record in a fresh process.

    1. Verifies `prereg_dir` via `verify_registration` (SHA256SUMS, no weight payloads, tamper checks).
    2. Enforces read-once (refuses existing `out_dir/trial_result.json` or duplicate `trial_sha256` in book) and
       enforces `MAX_REWORDED_USES = 5` rotation on any registered reworded set.
    3. Takes live uncached `we.base_checksum` before any reading and verifies `assert_base`.
    4. Reconstructs the registered `Edit` (from recipe if `"rebuilt"`, or verified payload if `"carried"`).
    5. Logs `trial_registration` in the level book BEFORE `os.urandom(32)` draws fresh kept-back items.
    6. Runs paired (or chunked written) target readings with `folded="always"`, reworded set readings with
       `folded="always"`, gating retention batteries, and author prompt arms.
    7. Verifies end-of-trial `base_checksum` == start-of-trial checksum, `assert_base`, and 0 unpacker fallbacks.
    8. Judges `"passed"` / `"failed"` / `"void"`, appends `trial_of_record` note to the book, and writes
       `trial_result.json` + `TRIAL_SHEET.md` into `out_dir`.
    """
    reg = verify_registration(prereg_dir)
    trial_cfg: Dict[str, Any] = reg["trial"]
    brief: lbr.Brief = reg["brief"]
    bars: lcl.BarSet = reg["bars"]
    edit_man: Dict[str, Any] = reg["edit_manifest"]
    reworded: Optional[RewordedSet] = reg["reworded"]
    trial_sha: str = reg["trial_sha256"]

    out_path = Path(out_dir).resolve()
    if (out_path / "trial_result.json").exists():
        raise TrialRefused(
            f"Read once: {out_path / 'trial_result.json'} already exists. A trial of record is read once."
        )
    bk = Path(book_path).expanduser().resolve() if book_path is not None else lb.book_path().resolve()
    if _trial_seen_in_book(trial_sha, bk):
        raise TrialRefused(
            f"Read once: trial registration {trial_sha[:12]} has already been recorded in {bk.name}."
        )

    hist = lcl.brief_history(brief.id, bk, name=brief.name, guard=lbr.KeptBack([brief]))
    if require_book_calibration and lcl.calibration_note(bars, bk) is None:
        raise TrialRefused(
            f'Bar set {bars.short()} is not recorded in a calibration note of {bk.name} for brief "{brief.name}".'
        )
    rw_uses_before = 0
    if reworded is not None:
        rw_uses_before = reworded_uses_in_book(reworded.id, bk)
        if rw_uses_before >= MAX_REWORDED_USES:
            raise TrialRefused(
                f'Reworded set "{reworded.name}" ({reworded.short()}) has already been read {rw_uses_before} times '
                f"(cap {MAX_REWORDED_USES}): rotate to a fresh owner-written reworded set."
            )

    # Live uncached base integrity check before touching anything
    t_start = time.perf_counter()
    ck_before = we.base_checksum(h.lm, hooks_at_load=h.hooks_at_load)
    fb_before = lv._fell_back()
    edit = load_registered_edit(h, edit_man, payload_dir=payload_dir)
    lcl._check_edit_allowed_by_brief(edit, brief)

    main_dose = float(trial_cfg["main_dose"])
    doses = [main_dose, -main_dose, 0.0] if brief.direction != 0 else [main_dose, 0.0]
    for d in doses:
        dw = we.dose_refusal(edit, d, h.dtype)
        if dw:
            raise TrialRefused(f"Registered edit cannot be read at dose {d:g}: {dw}")

    selection_mode = str(trial_cfg["selection_mode"])
    sham_recipes: Optional[List[Dict[str, Any]]] = None
    if selection_mode == "declared_rule":
        sham_recipes = []
        for sm in trial_cfg.get("sham_manifests") or ():
            if not sm.get("all_rebuilt") or not isinstance(sm.get("rebuilt_recipe"), Mapping):
                raise TrialRefused("Carried sham edits in a trial of record require rebuilt recipes.")
            sham_recipes.append(dict(sm["rebuilt_recipe"]))

    C = int(trial_cfg["controls_per_family"])
    rows_cfg = dict(trial_cfg["rows"])
    tol = float(trial_cfg["match_tolerance"])
    n_target = int(trial_cfg["n_target"])
    n_neutral = int(trial_cfg["n_neutral"])
    n_near = int(trial_cfg["n_near"])
    n_panel = int(trial_cfg["n_panel"])
    folded_required = bool(trial_cfg.get("folded_required", True))
    can_fold_why = we.can_fold(h.lm, edit)
    folded_mode = "always" if can_fold_why is None else "never"

    commit_now, dirty_now = git_state(ROOT_DIR)
    prev_bk = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(bk)
    try:
        # Step 1: Append trial_registration BEFORE drawing fresh items
        reg_body = {
            "note_type": "trial_registration",
            "trial_name": trial_cfg["trial_name"],
            "trial_sha256": trial_sha,
            "brief_id": brief.id,
            "brief_name": brief.name,
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "edit_provenance": trial_cfg["edit_provenance"],
            "bars_sha256": bars.id,
            "reworded_id": reworded.id if reworded is not None else None,
            "reworded_use_index": (rw_uses_before + 1) if reworded is not None else None,
            "controls_per_family": C,
            "selection_mode": selection_mode,
            "folded_required": folded_required,
            "base_checksum_before": ck_before["sha256"],
            "base_linears_checked": ck_before["linears_checked"],
            "repo_commit": commit_now,
            "git_dirty": dirty_now,
        }
        reg_entry = lb.append("note", reg_body, path=bk)
        book_seqs: List[int] = [int(reg_entry["seq"])]

        # Step 2: Post-freeze os.urandom(32) draws
        seed_bytes = bytes(_seed_override) if _seed_override is not None else os.urandom(32)
        seed_sha = _sha_bytes(seed_bytes)
        seed_u32 = int.from_bytes(seed_bytes[:4], "little") & 0x7FFFFFFF

        all_forbid_items = set(hist["practice_item_ids"])
        all_forbid_frames = set(hist["practice_frame_ids"])
        void_reason: Optional[str] = None
        try:
            target_drawn = brief.generator.draw(
                "kept_back", n_target, seed_bytes=seed_bytes + b":trial_target",
                tokenizer=h.tokenizer, system_override=(brief.chat_settings or {}).get("system"),
                forbid_item_ids=all_forbid_items, forbid_frame_ids=all_forbid_frames,
                name_prefix="trial_target",
            )
            near_drawn = brief.near_miss_generator.draw(
                "kept_back", n_near, seed_bytes=seed_bytes + b":trial_near",
                tokenizer=h.tokenizer, system_override=(brief.chat_settings or {}).get("system"),
                name_prefix="trial_near",
            )
            neu_drawn = lbr.neutral_chat_generator().draw(
                "kept_back", n_neutral, seed_bytes=seed_bytes + b":trial_neu",
                tokenizer=h.tokenizer, name_prefix="trial_neu",
            )
            panel = lbr.draw_labelled_sanity_panel(n_panel, seed_bytes=seed_bytes + b":trial_panel", tokenizer=h.tokenizer)
        except lbr.BriefError as exc:
            void_reason = f"Fresh draw refused: {exc}"
            target_drawn, near_drawn, neu_drawn, panel = [], [], [], []

        target_items = {d.item.name: d.item for d in target_drawn}
        near_items = {d.item.name: d.item for d in near_drawn}
        neu_items = {d.item.name: d.item for d in neu_drawn}
        panel_items = {p["item"].name: p["item"] for p in panel}

        classes_payload: Dict[str, Dict[str, Any]] = {}
        folded_by_class: Dict[str, Dict[str, Any]] = {}
        unpacker_mode: Optional[str] = None
        tname = trial_cfg["trial_name"]

        # Step 3: Fresh kept-back target reading (+ folded="always" on both batch classes)
        if void_reason is None:
            if brief.metric == "written_loop":
                void_reason, classes_payload, w_seqs, _strata, unpacker_mode = lcl._run_written_target_claims(
                    h, brief, edit, target_drawn, base_dir=Path(prereg_dir), main_dose=main_dose, doses=doses,
                    C=C, seed_u32=seed_u32, reading_index=1, selection_mode=selection_mode,
                    sham_recipes=sham_recipes, match_tolerance=tol, written_rows=int(trial_cfg.get("written_rows", 12)),
                )
                book_seqs.extend(w_seqs)
                folded_by_class["written"] = {
                    "available": False,
                    "why": "written_loop folded twin runs via dedicated written check (not yet measured)",
                }
            else:
                base_plan = {
                    "kind": "paired",
                    "edit": {"built": edit.id},
                    "doses": doses,
                    "main_dose": main_dose,
                    "rows": rows_cfg,
                    "readouts": dict(brief.readouts),
                    "rank_by": brief.metric,
                }
                r_matched = lv.read(
                    h, {**base_plan, "name": f"trial_{tname}_matched",
                        "controls": {"family": "other_sites_matched", "count": C, "seed": seed_u32, "tolerance": tol},
                        "folded": folded_mode},
                    target_items, edit=edit,
                )
                book_seqs.append(int(r_matched["book"]["seq"]))
                unpacker_mode = (r_matched.get("runtime") or {}).get("unpacker", {}).get("mode")
                r_same = lv.read(
                    h, {**base_plan, "name": f"trial_{tname}_same_site",
                        "controls": {"family": "same_site", "count": C, "seed": seed_u32 + 1},
                        "folded": "never"},
                    target_items, edit=edit,
                )
                book_seqs.append(int(r_same["book"]["seq"]))
                r_sham = None
                if selection_mode == "declared_rule" and sham_recipes:
                    r_sham = lv.read(
                        h, {**base_plan, "name": f"trial_{tname}_sham",
                            "controls": {"family": "sham_target", "shams": sham_recipes, "seed": seed_u32 + 2},
                            "folded": "never"},
                        target_items, edit=edit,
                    )
                    book_seqs.append(int(r_sham["book"]["seq"]))

                for tag, r in (("other_sites_matched", r_matched), ("same_site", r_same), ("sham_target", r_sham)):
                    if r is not None and r.get("status") != "ok":
                        void_reason = f'Trial target reading ({tag}) ended "{r.get("status")}": {r.get("why")}'
                        break

                if void_reason is None:
                    for cls_name in ("small", "large"):
                        it_effs, fl_m, rev_m, ctl_m = lcl._extract_item_effects(
                            r_matched, target_drawn, cls_name, brief.metric, main_dose,
                        )
                        _, fl_s, _, ctl_s = lcl._extract_item_effects(
                            r_same, target_drawn, cls_name, brief.metric, main_dose,
                        )
                        fl_both = max([x for x in (fl_m, fl_s) if x is not None], default=None)
                        ctls_cls: Dict[str, Any] = {"other_sites_matched": ctl_m, "same_site": ctl_s}
                        deliv_cls: Dict[str, Any] = {
                            "other_sites_matched": lcl._delivered_of(r_matched, target_drawn, cls_name, main_dose),
                            "same_site": lcl._delivered_of(r_same, target_drawn, cls_name, main_dose),
                        }
                        if r_sham is not None:
                            _, _, _, ctl_sh = lcl._extract_item_effects(
                                r_sham, target_drawn, cls_name, brief.metric, main_dose,
                            )
                            dv_sh = lcl._delivered_of(r_sham, target_drawn, cls_name, main_dose)
                            if dv_sh is not None:
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
                                dv.pop("per_control", None)
                        classes_payload[cls_name] = {
                            "item_effects": it_effs,
                            "floor": fl_both,
                            "reversed_mean": rev_m,
                            "controls": ctls_cls,
                            "delivered": deliv_cls,
                        }
                        if can_fold_why is None:
                            folded_by_class[cls_name] = extract_folded_class(r_matched, target_drawn, cls_name, brief.metric)
                        else:
                            folded_by_class[cls_name] = {"available": False, "why": can_fold_why}

        # Step 4: Reworded set (per-row + folded="always")
        reworded_eval: Optional[Dict[str, Any]] = None
        if void_reason is None and reworded is not None and brief.metric != "written_loop":
            rw_drawn = reworded.to_drawn_items(h.tokenizer, system_override=(brief.chat_settings or {}).get("system"))
            rw_items = {d.item.name: d.item for d in rw_drawn}
            rw_C = int(trial_cfg.get("reworded_controls", min(C, 19)))
            rw_plan = {
                "name": f"trial_{tname}_reworded",
                "kind": "paired",
                "edit": {"built": edit.id},
                "doses": [main_dose, 0.0],
                "main_dose": main_dose,
                "controls": {"family": "other_sites_matched", "count": rw_C, "seed": seed_u32 + 11, "tolerance": tol},
                "rows": rows_cfg,
                "readouts": dict(brief.readouts),
                "rank_by": brief.metric,
                "folded": folded_mode,
            }
            r_rw = lv.read(h, rw_plan, rw_items, edit=edit)
            book_seqs.append(int(r_rw["book"]["seq"]))
            if r_rw.get("status") != "ok":
                void_reason = f'Reworded reading ended "{r_rw.get("status")}": {r_rw.get("why")}'
            else:
                rw_classes: Dict[str, Dict[str, Any]] = {}
                rw_folded: Dict[str, Dict[str, Any]] = {}
                for cls_name in ("small", "large"):
                    it_effs, fl_rw, _, ctl_rw = lcl._extract_item_effects(
                        r_rw, rw_drawn, cls_name, brief.metric, main_dose,
                    )
                    rw_classes[cls_name] = {
                        "item_effects": it_effs,
                        "floor": fl_rw,
                        "controls": {"other_sites_matched": ctl_rw},
                    }
                    if can_fold_why is None:
                        rw_folded[cls_name] = extract_folded_class(r_rw, rw_drawn, cls_name, brief.metric)
                    else:
                        rw_folded[cls_name] = {"available": False, "why": can_fold_why}
                reworded_eval = judge_reworded(
                    brief, bars, rw_classes, rw_folded,
                    folded_required=folded_required,
                    tol_floor_mult=float(trial_cfg.get("folded_tol_floor_mult", DEFAULT_FOLDED_FLOOR_MULT)),
                    tol_rel=float(trial_cfg.get("folded_tol_rel", DEFAULT_FOLDED_REL_TOL)),
                    tol_min=float(trial_cfg.get("folded_tol_min", DEFAULT_FOLDED_MIN_TOL)),
                )

        # Step 5: Retention batteries & author prompt arms
        bat_payload: Dict[str, Dict[str, Any]] = {}
        bat_ctl_n = min(C, 3)
        nm_metric = brief.metric if brief.metric != "written_loop" else "kl"
        if void_reason is None:
            r_neu = lv.read(
                h, {"name": f"trial_{tname}_neutral", "kind": "paired",
                    "edit": {"built": edit.id}, "doses": [main_dose, 0.0],
                    "controls": {"family": "other_sites", "count": bat_ctl_n, "seed": seed_u32 + 3},
                    "rows": rows_cfg, "readouts": {"kl": True, "own": True, "odds": None}, "folded": "never"},
                neu_items, edit=edit,
            )
            book_seqs.append(int(r_neu["book"]["seq"]))
            r_nm = lv.read(
                h, {"name": f"trial_{tname}_nearmiss", "kind": "paired",
                    "edit": {"built": edit.id}, "doses": [main_dose, 0.0],
                    "controls": {"family": "other_sites", "count": bat_ctl_n, "seed": seed_u32 + 4},
                    "rows": rows_cfg, "readouts": dict(brief.readouts), "folded": "never"},
                near_items, edit=edit,
            )
            book_seqs.append(int(r_nm["book"]["seq"]))
            r_pan = lv.read(
                h, {"name": f"trial_{tname}_panel", "kind": "paired",
                    "edit": {"built": edit.id}, "doses": [main_dose, 0.0],
                    "controls": {"family": "other_sites", "count": bat_ctl_n, "seed": seed_u32 + 5},
                    "rows": rows_cfg, "readouts": {"kl": True, "own": True, "odds": None}, "folded": "never"},
                panel_items, edit=edit,
            )
            book_seqs.append(int(r_pan["book"]["seq"]))
            for tag, r in (("neutral_text", r_neu), ("near_misses", r_nm), ("labelled_panel", r_pan)):
                if r.get("status") != "ok":
                    void_reason = f'Trial battery ({tag}) ended "{r.get("status")}": {r.get("why")}'
                    break
            if void_reason is None:
                neu_kls = [v for c in ("small", "large") for v in [lcl._mean_metric_from_read(r_neu, "kl", main_dose, cls_name=c)[0]] if v is not None]
                neu_ctls = [v for c in ("small", "large") for v in [lcl._mean_metric_from_read(r_neu, "kl", main_dose, cls_name=c)[1]] if v is not None]
                bat_payload["neutral_text"] = {
                    "mean_kl": max(neu_kls) if neu_kls else None,
                    "control_kl_max": max(neu_ctls) if neu_ctls else None,
                }
                nm_vals = [v for c in ("small", "large") for v in [lcl._mean_metric_from_read(r_nm, nm_metric, main_dose, cls_name=c)[0]] if v is not None]
                nm_abs = max((abs(v) for v in nm_vals), default=None)
                bat_payload["near_misses"] = {
                    "mean_effect": nm_vals[-1] if nm_vals else None,
                    "abs_effect": nm_abs,
                }
                p_owns = [v for c in ("small", "large") for v in [lcl._mean_metric_from_read(r_pan, "own", main_dose, cls_name=c)[0]] if v is not None]
                p_kls = [v for c in ("small", "large") for v in [lcl._mean_metric_from_read(r_pan, "kl", main_dose, cls_name=c)[0]] if v is not None]
                own_drop = max((max(0.0, -v) for v in p_owns), default=None)
                bat_payload["labelled_panel"] = {
                    "own_drop": own_drop,
                    "mean_kl": max(p_kls) if p_kls else None,
                }
                bat_payload["habitat"] = {"available": False, "note": "no atlas passages index mounted"}

        prompt_arm_res = (
            lcl._prompt_arm_eval(h, brief, target_drawn, near_drawn)
            if (void_reason is None and brief.prompt_arms and brief.metric in ("odds", "own"))
            else []
        )
        if not brief.prompt_arms:
            arms_state = lcl.PROMPT_ARMS_NONE
        elif brief.metric not in ("odds", "own"):
            arms_state = f'no prompt read-out for metric "{brief.metric}"'
        elif void_reason is not None:
            arms_state = "the reading is void"
        else:
            arms_state = lcl.PROMPT_ARMS_RUN
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

        # Step 6: End-of-trial live base integrity check & unpacker fallbacks
        try:
            ck_after = we.base_checksum(h.lm, hooks_at_load=h.hooks_at_load)
            base_after = True
        except Exception:
            ck_after = {"sha256": None, "linears_checked": 0, "overlays_in_force": -1, "hooks_now": -1}
            base_after = False
        base_checksum_unchanged = bool(
            base_after
            and ck_after["sha256"] == ck_before["sha256"]
            and ck_after["linears_checked"] == ck_before["linears_checked"]
        )
        fb_after = lv._fell_back()
        unpacker_fallbacks = int(fb_after or 0) - int(fb_before or 0)

        # Step 7: Apply Phase 3 claim gate + Phase 5 folded & reworded & integrity gates
        src_claim = trial_cfg.get("source_claim") or {}
        fresh_judged = lcl.judge_claim(
            brief=brief,
            bars=bars,
            classes=classes_payload,
            batteries=bat_payload,
            cheaper_knobs=cheaper_knobs,
            selection_mode=selection_mode,
            practice_count=int(hist["practice_readings"]),
            practice_best=_clean_float(src_claim.get("practice_best")),
            void_reason=void_reason,
        )
        folded_eval = judge_folded(
            brief, bars, fresh_judged, folded_by_class,
            folded_required=folded_required,
            tol_floor_mult=float(trial_cfg.get("folded_tol_floor_mult", DEFAULT_FOLDED_FLOOR_MULT)),
            tol_rel=float(trial_cfg.get("folded_tol_rel", DEFAULT_FOLDED_REL_TOL)),
            tol_min=float(trial_cfg.get("folded_tol_min", DEFAULT_FOLDED_MIN_TOL)),
        )
        trial_judged = judge_trial(
            brief, bars, fresh_judged,
            folded_eval=folded_eval,
            reworded_eval=reworded_eval,
            base_checksum_unchanged=base_checksum_unchanged,
            base_after=base_after,
            unpacker_fallbacks=unpacker_fallbacks,
            void_reason=void_reason,
        )

        elapsed_s = round(time.perf_counter() - t_start, 3)
        meta = {
            "date": _utc_now()[:10],
            "finished_at": _utc_now(),
            "device": str(h.runtime.get("device") or h.device),
            "unpacker": unpacker_mode or "none",
            "model_id": h.model_id,
            "weights_id": (h.weights_id or "none")[:16],
            "trial_sha256": trial_sha,
            "seed_sha256": seed_sha,
            "repo_commit": commit_now,
            "git_dirty": dirty_now,
            "book_name": bk.name,
            "base_checksum_before": ck_before["sha256"],
            "base_checksum_after": ck_after["sha256"],
            "base_checksum_unchanged": base_checksum_unchanged,
            "base_linears_checked": ck_before["linears_checked"],
            "base_after": base_after,
            "unpacker_fallbacks": unpacker_fallbacks,
            "reworded_use_index": (rw_uses_before + 1) if reworded is not None else None,
            "prior_verdicts": [
                {"seq": v["seq"], "edit_id": v.get("edit_id"), "verdict": v.get("verdict"),
                 "reading_index": v.get("reading_index")}
                for v in hist["verdicts"]
            ],
            "seconds": elapsed_s,
        }
        sentence = trial_claim_sentence(brief, trial_cfg, fresh_judged, trial_judged, meta)
        sheet = render_trial_sheet(brief, bars, trial_cfg, fresh_judged, trial_judged, meta, sentence)

        summary_compact = {
            cls: {
                "n_frames": fresh_judged["classes"][cls]["frames"]["n_frames"],
                "mean": fresh_judged["classes"][cls]["frames"]["mean"],
                "directed_mean": fresh_judged["classes"][cls]["frames"]["directed_mean"],
                "ci95": fresh_judged["classes"][cls]["frames"]["ci95"],
                "half_0": fresh_judged["classes"][cls]["frames"]["half_0"],
                "half_1": fresh_judged["classes"][cls]["frames"]["half_1"],
                "house_bar": fresh_judged["classes"][cls]["house_bar"],
                "controls": fresh_judged["classes"][cls]["controls"],
            }
            for cls in (fresh_judged.get("classes") or {})
        }
        verdict_body = {
            "note_type": "trial_of_record",
            "trial_name": trial_cfg["trial_name"],
            "trial_sha256": trial_sha,
            "brief_id": brief.id,
            "brief_name": brief.name,
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "edit_provenance": trial_cfg["edit_provenance"],
            "verdict": trial_judged["verdict"],
            "fresh_verdict": trial_judged["fresh_verdict"],
            "reasons": trial_judged["reasons"],
            "bars_sha256": bars.id,
            "registration_seq": int(reg_entry["seq"]),
            "book_seqs": book_seqs,
            "claim_sentence": sentence,
            "summary": summary_compact,
            "folded": {
                "ok": folded_eval["ok"],
                "status": folded_eval["status"],
                "required": folded_eval["required"],
                "classes": {
                    cls: {
                        "ok": fc.get("ok"),
                        "available": fc.get("available"),
                        "mean": (fc.get("frames") or {}).get("mean"),
                        "max_abs_gap": fc.get("max_abs_gap"),
                        "same_bits_count": fc.get("same_bits_count"),
                        "taken_count": fc.get("taken_count"),
                    }
                    for cls, fc in (folded_eval.get("classes") or {}).items()
                },
            },
            "reworded": (
                {
                    "ok": reworded_eval["ok"],
                    "status": reworded_eval["status"],
                    "reworded_id": reworded.id if reworded is not None else None,
                    "classes": {
                        cls: {"ok": rc.get("ok"), "mean": (rc.get("frames") or {}).get("mean")}
                        for cls, rc in (reworded_eval.get("classes") or {}).items()
                    },
                }
                if reworded_eval is not None else None
            ),
            "batteries": fresh_judged["batteries"],
            "cheaper_knobs": fresh_judged["cheaper_knobs"],
            "base_checksum_unchanged": base_checksum_unchanged,
            "unpacker_fallbacks": unpacker_fallbacks,
            "seconds": elapsed_s,
        }
        v_entry = lb.append("note", verdict_body, path=bk)
        we.assert_base(h.lm, h.hooks_at_load)

        result = {
            "v": TRIAL_VERSION,
            "trial_name": trial_cfg["trial_name"],
            "trial_sha256": trial_sha,
            "verdict": trial_judged["verdict"],
            "fresh_verdict": trial_judged["fresh_verdict"],
            "reasons": trial_judged["reasons"],
            "claim_sentence": sentence,
            "sheet": sheet,
            "trial_config": trial_cfg,
            "fresh_judged": fresh_judged,
            "folded": folded_eval,
            "reworded": reworded_eval,
            "meta": meta,
            "book": {
                "path": str(bk),
                "registration_seq": int(reg_entry["seq"]),
                "verdict_seq": int(v_entry["seq"]),
                "verdict_hash": v_entry["hash"],
                "all_seqs": book_seqs + [int(v_entry["seq"])],
            },
        }
        out_path.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out_path / "trial_result.json", result, indent=2)
        (out_path / "TRIAL_SHEET.md").write_text(sheet, encoding="utf-8")
        return result
    finally:
        if prev_bk is None:
            os.environ.pop(lb.BOOK_ENV, None)
        else:
            os.environ[lb.BOOK_ENV] = prev_bk
