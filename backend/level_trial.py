"""Levelling Phase 5: the trial of record (`backend/level_trial.py`).

A trial of record is the harness's second reading of an edit on a brief's kept-back frames, with more asked of it than
of a claim reading (`verify/arbiter.md` §4 rows 18, 25, 30 to 32, §5.3; `design/synthesis.md` §vi):

- It is a FRESH READING, so it is asked what a claim reading is asked before it is registered
  (`level_claim.fresh_reading_gate`: kept-back wording practised in the book, a brief changed under its name, bars that
  no calibration in the book produced or that were calibrated on other weights, the brief's budget `m`), it spends one
  of the brief's `m`, an edit has one trial, and every later sheet of the brief lists it.
- It reads the edit in BOTH batch classes against matched other-site and same-site control edits (and sham-target
  ones for a site chosen by a declared rule), with the FOLDED form (the change written into the weights) read beside
  the per-row form on every item, and on a REWORDED SET: items written outside the brief's generator, kept in a file,
  read at most `MAX_REWORDED_USES` times in a book. An edit with no folded form (a low-rank term) and a written brief
  (which has neither a folded reading nor a reworded one yet) have no trial: registration refuses them.
- It is registered in a folder (`trial.json`, `brief.json`, `bars.json`, `edit_manifest.json`, `reworded.json`,
  `RULES.md`, `repo_commit`, `registered_at`, `SHA256SUMS`) that holds manifests and hashes and no weights, and in the
  book before its fresh items are drawn from `os.urandom`.
- It is OF RECORD only when the conditions the proposal names hold, each checked and not assumed: the process had
  read nothing before it, the checkout is clean, the registration folder is committed as it stands, and the runner
  and reader are the files that were hashed at registration. Anything else is refused, or, when asked for, read as a
  DRY RUN: the same readings, judged the same way, a fresh reading spent all the same, written into the book as a dry
  run, with no sentence. A volunteer's re-run of a kit (`level_rerun`) is a third kind, and is never of record.

Only a trial of record prints `passed` or `failed` as a verdict of record. What its verdict does not show: the fresh
items are new fillings of the brief's kept-back frames, the frames a claim reading of the same brief has already
read, so the frames are not new to the edit; the reworded set is what stands outside them.

`rejudge_trial` judges a trial again from the readings in a book, with this module's code: what an intake does with
a volunteer's returned book, and what shows that a verdict note follows from the lines it names.

Layers count from 0 in code and files; a string a person reads gives both ("layer 50 (page layer 51)").
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
from types import SimpleNamespace
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
TRIAL_CONTROLS = 99  # synthesis.md §vi and arbiter.md row 18: 99 control edits a family support five fresh readings
MAX_REWORDED_USES = 5  # synthesis.md §vi: a reworded item is read at most five times in its life, then the set rotates
# How far the folded form's mean may sit from the per-row form's and still be called the same edit: one floor (the
# largest thing a null row did on these items), or 5% of the effect, never more than half the house bar.
DEFAULT_FOLDED_FLOOR_MULT = 1.0
DEFAULT_FOLDED_REL_TOL = 0.05
DEFAULT_FOLDED_MIN_TOL = 0.02

TRIAL_VERDICTS = ("passed", "failed", "void")
FORBIDDEN_PREREG_SUFFIXES = (".safetensors", ".gguf")
# what a registration folder holds, and nothing else; a volunteer's kit adds two (`level_rerun`)
REG_FILES = ("RULES.md", "bars.json", "brief.json", "edit_manifest.json", "registered_at", "repo_commit",
             "reworded.json", "trial.json")
KIT_EXTRA_FILES = ("HOME_SHA256SUMS", "kit.json")
# the runner and the reader: the files whose hashes a registration records and a trial of record finds unchanged
SOURCE_FILES = ("backend/level_trial.py", "backend/level_claim.py", "backend/level_briefs.py", "backend/levelling.py",
                "backend/weight_edits.py", "backend/level_book.py")
NOTE_OF_RECORD, NOTE_DRY_RUN, NOTE_RERUN = "trial_of_record", "trial_dry_run", "trial_rerun"

ROOT_DIR = Path(__file__).resolve().parent.parent


class TrialRefused(ValueError):
    """Plain-word refusal of a trial that cannot be registered, verified or read as asked."""


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


# ------------------------------------------------------------------------------------------- the checkout, as found


def _git(root: Path, *args: str) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(root: Path = ROOT_DIR) -> Tuple[str, Optional[bool]]:
    """`(commit, dirty)` of the checkout at `root`: ("unknown", None) where there is no git to ask. `dirty` is whether
    `git status --porcelain` lists anything, a file git does not know included."""
    c = _git(root, "rev-parse", "HEAD")
    if c is None or c.returncode != 0:
        return "unknown", None
    st = _git(root, "status", "--porcelain")
    dirty = bool(st.stdout.strip()) if st is not None and st.returncode == 0 else None
    return c.stdout.strip() or "unknown", dirty


def checkout_changes(root: Path = ROOT_DIR) -> Optional[List[str]]:
    """The paths `git status --porcelain` lists in the checkout at `root` (None where git cannot be asked)."""
    st = _git(root, "status", "--porcelain")
    if st is None or st.returncode != 0:
        return None
    return [ln[3:] for ln in st.stdout.splitlines() if ln.strip()]


def folder_committed(folder: Path, root: Path = ROOT_DIR) -> Optional[str]:
    """None when every file of `folder` is in the commit the checkout at `root` stands at, unchanged; else why not,
    in words: the folder is outside the checkout, a file of it is in no commit, or it differs from what is
    committed."""
    f, r = Path(folder).resolve(), Path(root).resolve()
    if f != r and r not in f.parents:
        return "the registration folder is outside the checkout, so it is in no commit"
    names = sorted(p.name for p in f.iterdir() if p.is_file())
    known = _git(r, "ls-files", "--error-unmatch", "--", *[str(f / n) for n in names])
    if known is None or known.returncode != 0:
        return "the registration folder holds files that are in no commit"
    st = _git(r, "status", "--porcelain", "--", str(f))
    if st is None or st.returncode != 0 or st.stdout.strip():
        return "the registration folder differs from what is committed"
    return None


def source_hashes(root: Path = ROOT_DIR) -> Dict[str, Optional[str]]:
    """The sha256 of each file of `SOURCE_FILES` under `root` (None for one that is not there)."""
    out: Dict[str, Optional[str]] = {}
    for rel in SOURCE_FILES:
        p = Path(root) / rel
        out[rel] = _sha_file(p) if p.is_file() else None
    return out


# --------------------------------------------------------------------------------------------- reworded item sets


@dataclass(frozen=True)
class RewordedItem:
    """One reworded item: a question and the start of its answer, written outside the brief generator's templates."""

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

    @property
    def wording(self) -> str:
        """The item's own hash, by its wording alone (as leakage is judged: `level_briefs._norm_text`), whatever it is
        called and whichever set and half it is in: what a use of the item is counted by."""
        return _sha_bytes(lb.canonical([lbr._norm_text(self.question), lbr._norm_text(self.answer),
                                        lbr._norm_text(self.system or "")]))

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
    """A reworded item set for a trial: a file's worth of items in two halves, hashed. The harness cannot know who
    wrote a set; what a sheet says of one is whether its items are the starter set's, which stands in the
    instrument's source (`reworded_origin`)."""

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
        seen_words = set()
        h0 = h1 = 0
        for it in self.items:
            if it.item_id in seen_ids:
                raise TrialRefused(f'Duplicate reworded item_id "{it.item_id}".')
            seen_ids.add(it.item_id)
            if it.wording in seen_words:
                raise TrialRefused(f'Reworded item "{it.item_id}" has the wording of another item of the set.')
            seen_words.add(it.wording)
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

    def wordings(self) -> List[str]:
        """Each item's own hash (`RewordedItem.wording`), in the set's order."""
        return [it.wording for it in self.items]

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
        """Every reworded item as a `level_briefs.DrawnItem` (one frame an item)."""
        out: List[lbr.DrawnItem] = []
        for idx, it in enumerate(self.items):
            sys_turn = system_override if system_override is not None else it.system
            lv_item = lv.make_item(
                tokenizer, it.question, it.answer, system=sys_turn, name=f"reword_{idx:03d}_{it.item_id}",
            )
            out.append(lbr.DrawnItem(item=lv_item, frame_id=it.item_id, split="reworded", half=it.half, slots={}))
        return out


def validate_reworded_against_brief(rset: RewordedSet, brief: lbr.Brief) -> None:
    """Refuse a set that is another brief's, or that holds the wording (or a near wording) of the brief's practice
    frames, kept-back frames, near misses or the retention batteries: a reworded set is outside all of them."""
    if rset.brief_id != brief.id:
        raise TrialRefused(f'RewordedSet "{rset.name}" is for brief {rset.brief_id[:12]}, not {brief.short()}.')
    guard = lbr.KeptBack([brief], batteries=True)
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
    """Write a RewordedSet to JSON with its hash (`reworded_id`) in it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**rset.to_dict(), "reworded_id": rset.id}
    write_json_atomic(p, payload, indent=2)
    return p


def load_reworded(path: Path) -> RewordedSet:
    """Load a RewordedSet from a JSON file that names its own hash (`reworded_id`), and refuse one whose hash is not
    its content's or that names none (the command line's `reworded --from FILE` writes an owner's own items out with
    their hash)."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialRefused(f"Cannot read reworded set {p}: {exc}") from None
    if not isinstance(raw, dict):
        raise TrialRefused(f"Reworded set {p} is not a JSON object.")
    wanted = raw.get("reworded_id")
    rs = RewordedSet.from_dict(raw)
    if not isinstance(wanted, str) or not wanted:
        raise TrialRefused(f"Reworded set {p} lacks its reworded_id hash.")
    if str(wanted) != rs.id:
        raise TrialRefused(f"Reworded set {p} claims reworded_id {str(wanted)[:12]} but hashes to {rs.short()}.")
    return rs


def reworded_uses_in_book(rset: RewordedSet, book_path: Optional[Path] = None) -> int:
    """How many trials registered in the book have read an item of `rset`: the most any one of its items has been
    read, counted by the item's wording (`RewordedItem.wording`), so a set renamed, re-noted or with one item changed
    does not start a new count. Counted in this book alone: the harness cannot see another book."""
    p = Path(book_path) if book_path is not None else lb.book_path()
    if not p.exists():
        return 0
    counts: Dict[str, int] = {}
    for ent in lb.entries(p):
        body = ent.get("body")
        if ent.get("kind") == "note" and isinstance(body, Mapping) and body.get("note_type") == "trial_registration":
            for w in set(body.get("reworded_wordings") or ()):
                counts[str(w)] = counts.get(str(w), 0) + 1
    return max((counts.get(w, 0) for w in rset.wordings()), default=0)


def reworded_practised(rset: RewordedSet, book_path: Optional[Path] = None) -> List[int]:
    """The lines of practice readings in the book (those not the harness's own) that hold a reworded item's question
    or the start of its answer, by wording as leakage is judged. A set read in practice is no longer outside what
    the edit was picked on."""
    p = Path(book_path) if book_path is not None else lb.book_path()
    if not p.exists():
        return []
    qs = {lbr._norm_text(it.question) for it in rset.items}
    ans = {lbr._norm_text(it.answer) for it in rset.items}
    lines: List[int] = []
    for ent in lb.entries(p):
        body = ent.get("body")
        if ent.get("kind") not in ("paired", "written") or not isinstance(body, Mapping):
            continue
        if str(body.get("name") or "").startswith(lcl.RESERVED_NAMES):
            continue
        for it in body.get("items") or ():
            blk = (it.get("item") or {}) if isinstance(it, Mapping) else {}
            if lbr._norm_text(blk.get("question") or "") in qs or lbr._norm_text(blk.get("answer") or "") in ans:
                lines.append(int(ent["seq"]))
                break
    return lines


def starter_reworded_pronoun_pull(brief: lbr.Brief) -> RewordedSet:
    """A starter reworded set for `pronoun_pull` (8 items, 4 a half), each a short scene of a named woman that stops
    on a conjunction asking for a pronoun. It was written with this module and stands in the instrument's source,
    where anyone can read it: it is for checks and dry runs, and a sheet says so of any trial that reads it. A set
    for a trial of record is the owner's own, kept in a file outside the source."""
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
        note="Starter set for pronoun_pull, written with the harness and published in its source (4 half-0, 4 half-1).",
    )
    validate_reworded_against_brief(rset, brief)
    return rset


# the starter sets this module publishes, by the brief name they are for
STARTER_SETS = {"pronoun_pull": starter_reworded_pronoun_pull}


def reworded_origin(rset: RewordedSet, brief: lbr.Brief) -> str:
    """What a sheet says of where a reworded set's items stand: by their wording, whichever file they came in by. A
    set saved from the starter set and handed back as a file is still the starter set's items."""
    published: set = set()
    make = STARTER_SETS.get(brief.name)
    if make is not None:
        try:
            published = set(make(brief).wordings())
        except TrialRefused:
            published = set()
    n = sum(1 for w in rset.wordings() if w in published)
    if n == len(rset.items):
        return "the starter set in the instrument's source, which anyone can read"
    if n:
        return (f"from a file given at registration; {n} of its {len(rset.items)} items are the starter set's, in "
                "the instrument's source")
    return "from a file given at registration"


# --------------------------------------------------------------------------------------------- registration folder


def _render_rules_md(trial_cfg: Mapping[str, Any], brief: lbr.Brief, bars: lcl.BarSet) -> str:
    """The rules of this trial in plain words, written into the registration folder and hashed in SHA256SUMS."""
    cal = trial_cfg["bars_calibration"]
    lines = [
        f"# Rules of the trial: {trial_cfg['trial_name']}",
        "",
        f"- rule preset: `{trial_cfg['rule_preset']}` (version {trial_cfg['v']})",
        f"- brief: `{brief.name}` (`{brief.id}`), a budget of `{brief.max_fresh_readings}` fresh readings",
        f"- metric: `{brief.metric}` (direction `{brief.direction:+d}`, least effect `{brief.least_effect:g}`)",
        f"- edit id: `{trial_cfg['edit_id']}` (provenance: `{trial_cfg['edit_provenance']}`)",
        f"- dose: `{trial_cfg['main_dose']:g}`",
        f"- bars sha256: `{bars.id}` (alpha `{bars.alpha:g}`, gating batteries: `{', '.join(bars.gating_batteries)}`), "
        f"from the calibration at line `{cal['seq']}` of `{cal['book']}`",
        f"- selection mode: `{trial_cfg['selection_mode']}`",
        f"- control edits a family: `{trial_cfg['controls_per_family']}` (other sites matched on delivered, same site"
        + (", sham target" if trial_cfg["selection_mode"] == "declared_rule" else "") + ")",
        f"- fresh items: `{trial_cfg['n_target']}`, drawn from `os.urandom(32)` after the trial is entered in the book",
        f"- batch classes: `small = {trial_cfg['rows']['small']}`, `large = {trial_cfg['rows']['large']}`",
        f"- folded form: read on every item in both classes; its mean within "
        f"max({trial_cfg['folded_tol_floor_mult']:g} x floor, {trial_cfg['folded_tol_rel']:g} x |per-row mean|, "
        f"{trial_cfg['folded_tol_min']:g}) of the per-row form's, and never more than half the house bar",
        f"- reworded set: `{trial_cfg['reworded_id']}` ({trial_cfg['reworded_origin']}; "
        f"`{trial_cfg['reworded_controls']}` matched control edits, judged alone at the level; an item is read at "
        f"most `{MAX_REWORDED_USES}` times in a book)",
        "",
        "## What a trial is",
        "",
        "A trial is a fresh reading of the brief's kept-back frames: it is asked what a claim reading is asked",
        "before it is registered, it spends one of the brief's fresh readings, an edit has one trial, and it is",
        "read once. It is OF RECORD only when the process had read nothing before it, the checkout is clean, this",
        "folder is committed as it stands, and the runner and reader are the files hashed in `trial.json`.",
        "Otherwise it is refused, or read as a DRY RUN when that is asked for: judged the same way, counted the",
        "same way, and never a verdict of record.",
        "",
        "## Verdicts",
        "",
        "1. **`void`**: a reading ended void or refused, the base checksum differed between start and end,",
        "   `assert_base` failed, the unpacker fell back, or the folded form was not read on every item.",
        "2. **`passed`**: not void, and all of:",
        "   - the fresh items clear the house bar `sqrt(F x M)` (with `M >= 4F`), replicate across both frame",
        "     halves, satisfy the reversed-dose check and stand above every control family at `p_fw <= alpha`,",
        "     in BOTH batch classes, with every gating retention battery within the registered bars;",
        "   - the folded form clears the house bar and replicates across the halves itself, and its mean is within",
        "     the registered tolerance of the per-row form's, in both classes;",
        "   - the reworded set clears the house bar, replicates across its halves and stands above its control",
        "     edits, per-row and folded, in both classes.",
        "3. **`failed`**: read cleanly, and one or more of the above did not hold.",
        "",
        "The fresh items are new fillings of the brief's kept-back frames, which a claim reading of the brief has",
        "already read: the frames are not new to the edit. The reworded set is what stands outside them.",
        "",
        "This folder holds manifests and hashes only; weight files (`*.safetensors`, `*.gguf`) are refused in it.",
    ]
    return "\n".join(lines) + "\n"


def _sums_bytes(folder: Path, names: Sequence[str]) -> bytes:
    """`SHA256SUMS` as this module writes it, and the only form it reads: one line a file, the names sorted, the
    digest in lower case, two spaces, the name, a newline."""
    return "".join(f"{_sha_file(folder / nm)}  {nm}\n" for nm in sorted(names)).encode("ascii")


def _write_sha256sums(folder: Path) -> Tuple[str, List[str]]:
    """Write `SHA256SUMS` over every regular file in `folder` but itself. Returns `(its sha256, the names)`."""
    names = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    content = _sums_bytes(folder, names)
    tmp = folder / "SHA256SUMS.tmp"
    tmp.write_bytes(content)
    os.replace(tmp, folder / "SHA256SUMS")
    return _sha_bytes(content), names


def register_trial(
    prereg_dir: Path,
    h: lv.Handle,
    brief: lbr.Brief,
    bars: lcl.BarSet,
    edit: we.Edit,
    *,
    trial_name: str,
    reworded: Optional[RewordedSet],
    book_path: Optional[Path] = None,
    main_dose: float = 1.0,
    selection_mode: str = "free_pick",
    declared_rule: Optional[str] = None,
    sham_recipes: Optional[Sequence[Mapping[str, Any]]] = None,
    base_dir: Path = Path("."),
    controls_per_family: int = TRIAL_CONTROLS,
    reworded_controls: Optional[int] = None,
    n_target: int = 16,
    n_neutral: int = 8,
    n_near: int = 8,
    n_panel: int = 8,
    rows: Optional[Mapping[str, int]] = None,
    match_tolerance: float = lv.DEFAULT_MATCH_TOL,
    folded_tol_floor_mult: float = DEFAULT_FOLDED_FLOOR_MULT,
    folded_tol_rel: float = DEFAULT_FOLDED_REL_TOL,
    folded_tol_min: float = DEFAULT_FOLDED_MIN_TOL,
    saved_edit_dir: Optional[Path] = None,
    forecast: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the registration folder of a trial. Nothing is entered in the book here: the trial is entered when it
    is read (`run_trial_of_record`), before its draw.

    Refused here, so that no folder is written for a trial that could not be read: a brief or an edit a trial is
    not yet for (a written brief; an edit with no folded form on this model), an edit the brief does not allow or
    that cannot be read at the dose, no reworded set or one that overlaps the brief's own wording, sham controls
    that are not the edit's shape, a design that cannot reach the level (too few control edits for the brief's `m`),
    and bars that no calibration reading in `book_path` produced or that were calibrated on other weights. The
    folder then records which calibration, of which book, the bars came from.
    """
    folder = Path(prereg_dir).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise TrialRefused(f"Registration folder {folder} already exists and is not empty: never overwrite a registration.")
    if not isinstance(trial_name, str) or not trial_name.strip():
        raise TrialRefused("trial_name must be a non-empty string.")
    if bars.brief_id != brief.id:
        raise TrialRefused(f"BarSet is for brief {bars.brief_id[:12]}, not {brief.short()}.")
    if brief.metric == "written_loop":
        raise TrialRefused(
            f'Brief "{brief.name}" is a written brief. A trial reads the folded form and a reworded set beside the '
            "per-row form, and a written reading has neither yet: a written brief has no trial of record for now."
        )
    if selection_mode not in ("free_pick", "declared_rule"):
        raise TrialRefused(f'selection_mode must be "free_pick" or "declared_rule"; got {selection_mode!r}.')
    if selection_mode == "declared_rule":
        if not sham_recipes:
            raise TrialRefused('selection_mode="declared_rule" requires sham_recipes.')
        if not (isinstance(declared_rule, str) and declared_rule.strip()):
            raise TrialRefused('selection_mode="declared_rule" requires a non-empty declared_rule sentence.')
    elif sham_recipes:
        raise TrialRefused('Sham edits are registered only with selection_mode="declared_rule".')
    if isinstance(main_dose, bool) or not isinstance(main_dose, (int, float)) or not math.isfinite(float(main_dose)) \
            or float(main_dose) == 0.0:
        raise TrialRefused(f"main_dose must be a finite non-zero number; got {main_dose!r}.")
    if isinstance(match_tolerance, bool) or not isinstance(match_tolerance, (int, float)) \
            or not math.isfinite(float(match_tolerance)) or float(match_tolerance) < 1.0:
        raise TrialRefused(f"match_tolerance must be finite and >= 1.0; got {match_tolerance!r}.")
    rw_ctls = reworded_controls if reworded_controls is not None else min(int(controls_per_family), 19) \
        if isinstance(controls_per_family, int) else 19
    # the fewest control edits with which nothing-moved-as-much reaches the level: m (1 + 0) / (C + 1) <= alpha
    least_c = int(math.ceil(brief.max_fresh_readings / bars.alpha - 1.0 - 1e-9))
    least_rw = int(math.ceil(1.0 / bars.alpha - 1.0 - 1e-9))
    min_target = 2 * int(bars.min_frames_per_half)
    for what, val, lo in (
        ("controls_per_family", controls_per_family, 1),
        ("reworded_controls", rw_ctls, 1),
        ("n_target", n_target, min_target),
        ("n_neutral", n_neutral, 1),
        ("n_near", n_near, 1),
        ("n_panel", n_panel, 1),
    ):
        if isinstance(val, bool) or not isinstance(val, int) or val < lo:
            raise TrialRefused(f"{what} must be a whole number >= {lo}; got {val!r}.")
    for what, val in (("controls_per_family", controls_per_family), ("reworded_controls", rw_ctls)):
        if val > lv.MAX_CONTROLS:
            raise TrialRefused(f"{what} is at most {lv.MAX_CONTROLS}, the most a reading takes; got {val}.")
    if controls_per_family < least_c:
        raise TrialRefused(
            f"With {controls_per_family} control edits a family this trial could not pass: at {brief.max_fresh_readings} "
            f"fresh readings a brief and a level of {bars.alpha:g}, no control moving as much needs {least_c} or more."
        )
    if rw_ctls < least_rw:
        raise TrialRefused(
            f"With {rw_ctls} control edits the reworded set could not pass: at a level of {bars.alpha:g}, no control "
            f"moving as much needs {least_rw} or more."
        )
    for what, val in (("folded_tol_floor_mult", folded_tol_floor_mult), ("folded_tol_rel", folded_tol_rel),
                      ("folded_tol_min", folded_tol_min)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) < 0.0:
            raise TrialRefused(f"{what} must be a finite number of 0 or more; got {val!r}.")

    we.assert_base(h.lm, h.hooks_at_load)
    try:
        lv._same_base(h, edit)
        lcl._check_edit_allowed_by_brief(edit, brief)
    except (lcl.ClaimRefused, lv.Refused, we.Refused) as exc:
        raise TrialRefused(str(exc)) from None
    if not any(t.kind in ("cols", "rows") for t in edit.terms):
        raise TrialRefused("A trial of record requires at least one column or row term for matched controls.")
    no_fold = we.can_fold(h.lm, edit)
    if no_fold is not None:
        raise TrialRefused(f"A trial reads the folded form of its edit, and this edit has none on this model: {no_fold}")
    doses = [float(main_dose), float(-main_dose), 0.0] if brief.direction != 0 else [float(main_dose), 0.0]
    for d in doses:
        words = we.dose_refusal(edit, d, h.dtype)
        if words:
            raise TrialRefused(f"The edit cannot be read at dose {d:g}, so the trial is not registered: {words}")
    shams: List[we.Edit] = []
    if sham_recipes:
        try:
            shams = lcl._check_shams(h, brief, edit, sham_recipes, Path(base_dir))
        except lcl.ClaimRefused as exc:
            raise TrialRefused(str(exc)) from None
        for i, se in enumerate(shams):
            if we.rebuilt_recipe(se) is None:
                raise TrialRefused(f"Sham control {i} carries arrays: a trial's sham controls are rebuilt from recipes.")
            words = we.dose_refusal(se, float(main_dose), h.dtype)
            if words:
                raise TrialRefused(f"Sham control {i} cannot be read at dose {float(main_dose):g}: {words}")
    if reworded is None:
        raise TrialRefused(
            "A trial reads a reworded set: items written outside the brief's generator, in a file "
            "(scripts/levelling_trial.py reworded). None was given."
        )
    validate_reworded_against_brief(reworded, brief)
    if any(it.half not in (0, 1) for it in reworded.items):
        raise TrialRefused("A reworded item's half is 0 or 1.")
    for half in (0, 1):
        n_half = sum(1 for it in reworded.items if it.half == half)
        if n_half < int(bars.min_frames_per_half):
            raise TrialRefused(
                f"The reworded set has {n_half} item(s) in half {half}; the bars ask {bars.min_frames_per_half} a half."
            )

    bk = Path(book_path).expanduser().resolve() if book_path is not None else lb.book_path().resolve()
    cal = lcl.calibration_note(bars, bk)
    if cal is None:
        raise TrialRefused(
            f'Bar set {bars.short()} is not from a calibration reading recorded in {bk.name} for brief "{brief.name}" '
            f"({brief.short()}): a trial takes the bars calibrate_bars wrote into the book it will be read in "
            "(scripts/levelling.py calibrate), and no others."
        )
    if cal.get("weights_id") not in (None, h.weights_id):
        raise TrialRefused(
            f"Bar set {bars.short()} was calibrated on other weights ({str(cal.get('weights_id'))[:16]}) than this "
            f"model's ({str(h.weights_id)[:16]}): calibrate on the model the trial is read on."
        )
    cal_entry = next(lb.entries(bk, start=int(cal["seq"]), limit=1))

    prov = we.edit_provenance(edit, saved_dir=saved_edit_dir)
    rows_cfg = {
        "small": int((rows or {}).get("small", lv.ROWS_SMALL)),
        "large": int((rows or {}).get("large", lv.ROWS_LARGE)),
    }
    min_small = max(5, 2 + len(lv.NULL_E) + (3 if brief.direction != 0 else 2))
    if not (min_small <= rows_cfg["small"] <= 12 and 16 <= rows_cfg["large"] <= 64):
        raise TrialRefused(
            f"Batch classes must satisfy {min_small} <= small <= 12 and 16 <= large <= 64; got {rows_cfg}."
        )
    commit, dirty = git_state(ROOT_DIR)
    reg_at = _utc_now()

    trial_cfg: Dict[str, Any] = {
        "v": TRIAL_VERSION,
        "rule_preset": TRIAL_RULE_PRESET,
        "trial_name": trial_name.strip(),
        "brief_id": brief.id,
        "brief_name": brief.name,
        "max_fresh_readings": int(brief.max_fresh_readings),
        "bars_sha256": bars.id,
        "bars_calibration": {"seq": int(cal["seq"]), "at": cal.get("at"), "hash": cal_entry["hash"],
                             "weights_id": cal.get("weights_id"), "bars_sha256": bars.id, "book": bk.name},
        "edit_id": edit.id,
        "edit_linears": edit.linears(),
        "edit_provenance": prov["provenance"],
        "all_rebuilt": bool(prov["all_rebuilt"]),
        "payload_sha256": prov["payload_sha256"],
        "base": dict(edit.base),
        "main_dose": float(main_dose),
        "selection_mode": selection_mode,
        "declared_rule": " ".join(declared_rule.split())[:600] if declared_rule else None,
        "sham_recipes": [we.rebuilt_recipe(se) for se in shams],
        "sham_edit_ids": [se.id for se in shams],
        "controls_per_family": int(controls_per_family),
        "reworded_id": reworded.id,
        "reworded_wordings": reworded.wordings(),
        "reworded_origin": reworded_origin(reworded, brief),
        "reworded_controls": int(rw_ctls),
        "n_target": int(n_target),
        "n_neutral": int(n_neutral),
        "n_near": int(n_near),
        "n_panel": int(n_panel),
        "rows": rows_cfg,
        "match_tolerance": float(match_tolerance),
        "folded_tol_floor_mult": float(folded_tol_floor_mult),
        "folded_tol_rel": float(folded_tol_rel),
        "folded_tol_min": float(folded_tol_min),
        "forecast": dict(forecast or {}),
        "registered_at": reg_at,
        "repo_commit": commit,
        "git_dirty": dirty,
        "sources": source_hashes(ROOT_DIR),
        # the kernel's implant-shape label, worked here with the model at hand: an export made off the model takes it
        "implant_check": we.implant_shape(edit, h.lm, h.W_U, dose=float(main_dose)),
    }

    folder.mkdir(parents=True, exist_ok=True)
    lbr.save_brief(brief, folder / "brief.json")
    lcl.save_bars(bars, folder / "bars.json")
    write_json_atomic(folder / "edit_manifest.json", prov, indent=2)
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


def verify_registration(prereg_dir: Path, *, kit: bool = False) -> Dict[str, Any]:
    """Check a registration folder against itself: it holds exactly the files a registration holds (a volunteer's
    kit, `kit=True`, two more), each a plain file and none a link or a folder; `SHA256SUMS` is byte for byte what
    this module would write for them (so a line added, repeated, re-ordered or re-cased is refused, and the hash of
    `SHA256SUMS` names one registration); and `trial.json`, the brief, the bars, the edit's manifest and the reworded
    set agree with one another. Returns the parsed parts. Raises `TrialRefused`."""
    folder = Path(prereg_dir).resolve()
    if not folder.is_dir():
        raise TrialRefused(f"Registration folder does not exist: {folder}")
    wanted = set(REG_FILES) | (set(KIT_EXTRA_FILES) if kit else set())
    found: List[str] = []
    for p in sorted(folder.iterdir()):
        if p.is_symlink() or not p.is_file():
            raise TrialRefused(
                f"Registration folder {folder} holds {p.name}, which is not a plain file: a registration is a flat "
                "folder of files, with no links and no folders in it."
            )
        if p.suffix.lower() in FORBIDDEN_PREREG_SUFFIXES:
            raise TrialRefused(
                f"Registration folder {folder} contains forbidden weight file {p.name}: "
                "registration folders hold manifests and payload hashes only."
            )
        if p.name != "SHA256SUMS":
            found.append(p.name)
    sums_path = folder / "SHA256SUMS"
    if not sums_path.is_file():
        raise TrialRefused(f"Registration folder {folder} lacks SHA256SUMS.")
    missing = sorted(wanted - set(found))
    extra = sorted(set(found) - wanted)
    if missing:
        raise TrialRefused(f"Registration folder {folder} is missing file(s): {missing}.")
    if extra:
        raise TrialRefused(f"Registration folder {folder} has unregistered extra file(s): {extra}.")
    raw_sums = sums_path.read_bytes()
    if raw_sums != _sums_bytes(folder, found):
        try:
            listed = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in raw_sums.decode("ascii").splitlines()
                      if "  " in ln}
        except UnicodeDecodeError:
            listed = {}
        bad = sorted(nm for nm in found if listed.get(nm) != _sha_file(folder / nm))
        raise TrialRefused(
            f"SHA256SUMS of {folder} is not what its files give"
            + (f": {', '.join(bad)} failed its check" if bad else
               ": its lines are not the ones a registration writes (one a file, sorted, nothing added)")
            + "."
        )
    trial_sha = _sha_bytes(raw_sums)

    try:
        trial_cfg = json.loads((folder / "trial.json").read_text(encoding="utf-8"))
        edit_man = json.loads((folder / "edit_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrialRefused(f"Cannot parse registration JSON in {folder}: {exc}") from None
    if not isinstance(trial_cfg, dict) or not isinstance(edit_man, dict):
        raise TrialRefused(f"trial.json and edit_manifest.json of {folder} are JSON objects.")
    if trial_cfg.get("v") != TRIAL_VERSION or trial_cfg.get("rule_preset") != TRIAL_RULE_PRESET:
        raise TrialRefused(
            f"trial.json is version {trial_cfg.get('v')!r} of rule {trial_cfg.get('rule_preset')!r}; this code reads "
            f"version {TRIAL_VERSION} of {TRIAL_RULE_PRESET}."
        )
    try:
        brief = lbr.load_brief(folder / "brief.json")
        bars = lcl.load_bars(folder / "bars.json")
    except (lbr.BriefError, lcl.ClaimRefused, OSError, ValueError) as exc:
        raise TrialRefused(f"The registration's brief or bars cannot be read: {exc}") from None
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
    cal = trial_cfg.get("bars_calibration")
    if not isinstance(cal, dict) or not isinstance(cal.get("seq"), int) or cal.get("bars_sha256") != bars.id:
        raise TrialRefused("trial.json does not record the calibration its bars came from.")
    reworded = load_reworded(folder / "reworded.json")
    if reworded.id != trial_cfg.get("reworded_id") or reworded.wordings() != trial_cfg.get("reworded_wordings"):
        raise TrialRefused("reworded.json is not the set trial.json names.")
    validate_reworded_against_brief(reworded, brief)

    return {
        "ok": True,
        "prereg_dir": str(folder),
        "trial_sha256": trial_sha,
        "trial": trial_cfg,
        "brief": brief,
        "bars": bars,
        "edit_manifest": edit_man,
        "reworded": reworded,
        "files": found,
    }


def load_registered_edit(
    h: lv.Handle,
    edit_manifest: Mapping[str, Any],
    *,
    payload_dir: Optional[Path] = None,
) -> we.Edit:
    """The registered edit, built again on `h` and kept to its registered id (which hashes its base's pins and its
    terms' arrays, so another model, another checkpoint or another payload is refused):
    - every term rebuilt: built from the manifest's recipe;
    - a carried term: loaded from `payload_dir` (or the saved edits folder, by id), which the kernel's own load checks
      against the saved manifest; where the registration recorded the payload file's sha256, that is checked too."""
    want_id = str(edit_manifest["edit_id"])
    try:
        if edit_manifest.get("all_rebuilt"):
            recipe = edit_manifest.get("rebuilt_recipe")
            if not isinstance(recipe, Mapping):
                raise TrialRefused("edit_manifest claims all_rebuilt but carries no rebuilt_recipe.")
            specs = we.from_recipe(dict(recipe), Path("."))
            if isinstance(specs, (str, Path)):
                raise TrialRefused("rebuilt_recipe must be inline terms, not a saved pointer.")
            edit = we.build(h.lm, specs, model_id=h.model_id, weights_id=h.weights_id,
                            note=str(edit_manifest.get("note") or ""))
        else:
            folder = we._find_saved(Path(payload_dir) if payload_dir is not None else want_id)
            edit = we.load(folder, h.lm, model_id=h.model_id, weights_id=h.weights_id)
            want_p = edit_manifest.get("payload_sha256")
            pfile = folder / "payload.safetensors"
            if want_p and pfile.is_file() and _sha_file(pfile) != want_p \
                    and we.edit_provenance(edit)["payload_sha256"] != want_p:
                raise TrialRefused(
                    f"Payload {pfile} is not the one registered (payload_sha256 {str(want_p)[:12]})."
                )
    except (we.Refused, lv.Refused, OSError) as exc:
        raise TrialRefused(f"The registered edit cannot be built on this model: {exc}") from None
    if edit.id != want_id:
        raise TrialRefused(
            f"Rebuilt edit hashes to {edit.id[:12]}, not registered edit_id {want_id[:12]} "
            "(the loaded weights or model may differ from the registration's base)."
        )
    return edit


# --------------------------------------------------------------------------------------------- folded extraction & judge


def extract_folded_class(
    result: Mapping[str, Any],
    drawn: Sequence[Any],
    cls_name: str,
    metric: str,
) -> Dict[str, Any]:
    """The folded twin's per-item effects and its gaps from the per-row form, from a reading taken with
    `folded="always"`. `available` only when the twin was taken on every item drawn."""
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

    whole = taken_n == len(drawn) and len(folded_effects) == len(drawn) and len(drawn) > 0
    return {
        "available": whole,
        "why": None if whole else (why_missing or f"the folded form was read on {taken_n} of {len(drawn)} items"),
        "taken_count": taken_n,
        "expected_count": len(drawn),
        "rows_identical_all": rows_id_all,
        "same_bits_count": same_bits_n,
        "item_effects": folded_effects,
        "max_abs_gap": max(gaps_abs) if gaps_abs else None,
        "mean_abs_gap": float(np.mean(gaps_abs)) if gaps_abs else None,
        "between_kl_max": max(between_kls) if between_kls else None,
    }


def folded_tolerance(floor: Optional[float], per_row_mean: Optional[float], bar: Optional[float], *,
                     floor_mult: float, rel: float, least: float) -> float:
    """How far the folded form's mean may sit from the per-row form's: the largest of `floor_mult` floors, `rel` of
    the per-row effect and `least`, and never more than half the house bar (agreement is not looser than the effect
    it is agreement about)."""
    tol = max(float(floor_mult) * (float(floor) if floor is not None else 0.0),
              float(rel) * (abs(float(per_row_mean)) if per_row_mean is not None else 0.0),
              float(least))
    return min(tol, 0.5 * float(bar)) if bar is not None else tol


def judge_folded(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    fresh_judged: Mapping[str, Any],
    folded_by_class: Mapping[str, Mapping[str, Any]],
    *,
    tol_floor_mult: float = DEFAULT_FOLDED_FLOOR_MULT,
    tol_rel: float = DEFAULT_FOLDED_REL_TOL,
    tol_min: float = DEFAULT_FOLDED_MIN_TOL,
) -> Dict[str, Any]:
    """Whether the folded form is the edit the per-row form read, in both classes: it clears the house bar and
    replicates across the halves itself, its rows are the base's where nothing is edited, and its mean and its
    largest item gap are within the tolerance (`folded_tolerance`) of the per-row form's. A class whose folded form
    was not read on every item is unavailable, and the whole is then unavailable."""
    reasons: List[str] = []
    cls_out: Dict[str, Any] = {}
    all_ok = True
    any_unavailable = False

    for cls_name in ("small", "large"):
        fd = folded_by_class.get(cls_name)
        if not fd or not fd.get("available"):
            any_unavailable = True
            why = (fd or {}).get("why") or "the folded form was not read"
            cls_out[cls_name] = {"available": False, "ok": False, "why": why}
            all_ok = False
            reasons.append(f"{cls_name} class folded form unavailable: {why}")
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
        tol = folded_tolerance(floor, fresh_mean, bar, floor_mult=tol_floor_mult, rel=tol_rel, least=tol_min)
        max_gap = _clean_float(fd.get("max_abs_gap"))
        mean_diff = (abs(float(fstats["mean"]) - float(fresh_mean))
                     if (fstats["mean"] is not None and fresh_mean is not None) else None)
        gap_ok = (
            bool(fd.get("rows_identical_all", True))
            and max_gap is not None and max_gap <= tol
            and mean_diff is not None and mean_diff <= tol
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

    return {
        "available": not any_unavailable,
        "ok": bool(all_ok),
        "status": "unavailable" if any_unavailable else ("ok" if all_ok else "failed"),
        "reasons": reasons,
        "classes": cls_out,
    }


def judge_reworded(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    reworded_classes: Mapping[str, Mapping[str, Any]],
    reworded_folded: Mapping[str, Mapping[str, Any]],
    *,
    tol_floor_mult: float = DEFAULT_FOLDED_FLOOR_MULT,
    tol_rel: float = DEFAULT_FOLDED_REL_TOL,
    tol_min: float = DEFAULT_FOLDED_MIN_TOL,
) -> Dict[str, Any]:
    """The reworded set, per-row and folded, in both classes: the house bar, the two halves, and its own control
    edits. The control edits are judged alone at the level (`max_fresh_readings=1`): the reworded reading is one more
    thing a trial must pass, not one more chance to."""
    reasons: List[str] = []
    cls_out: Dict[str, Any] = {}
    all_ok = True

    for cls_name in ("small", "large"):
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
        ctls_in = cdata.get("controls") or {}
        ctl_eval: Dict[str, Any] = {}
        ctl_ok = bool(ctls_in.get("other_sites_matched"))
        if not ctl_ok:
            reasons.append(f"Reworded {cls_name} has no control edits to stand against.")
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

    folded_eval = judge_folded(
        brief, bars, {"classes": cls_out}, reworded_folded,
        tol_floor_mult=tol_floor_mult, tol_rel=tol_rel, tol_min=tol_min,
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
    reworded_eval: Optional[Mapping[str, Any]],
    base_checksum_unchanged: bool = True,
    base_after: bool = True,
    unpacker_fallbacks: int = 0,
    void_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """The trial's rule, to one of `passed`, `failed`, `void`. `void`: a reading that did not end, a base that is not
    the base it began as, an unpacker that fell back, a folded form or a reworded set that was not read. `passed`: the
    fresh reading is a lead (or a lead matched by a cheaper knob, which the sheet and the sentence then say), the
    folded form agrees, and the reworded set stands. Whether the verdict is of record is not decided here."""

    def out(verdict: str, reasons: Sequence[str], **more: Any) -> Dict[str, Any]:
        return {"verdict": verdict, "reasons": list(reasons), "fresh_verdict": fresh_judged.get("verdict"),
                "folded": dict(folded_eval), "reworded": dict(reworded_eval) if reworded_eval is not None else None,
                **more}

    if void_reason:
        return out("void", [str(void_reason)])
    if not base_after or not base_checksum_unchanged:
        return out("void", ["The base was not the base it began as (assert_base failed, or the checksum of the "
                            "stored weights changed during the trial)."])
    if int(unpacker_fallbacks) != 0:
        return out("void", [f"Compiled unpacker fell back {int(unpacker_fallbacks)} time(s) during the trial."])
    fv = str(fresh_judged.get("verdict") or "void")
    if fv == "void":
        return out("void", list(fresh_judged.get("reasons") or ["Fresh target reading ended void."]))
    if not folded_eval.get("available"):
        return out("void", list(folded_eval.get("reasons") or ["The folded form was not read."]))
    if reworded_eval is None:
        return out("void", ["The reworded set was not read."])
    if not ((reworded_eval.get("folded") or {}).get("available")):
        return out("void", list((reworded_eval.get("folded") or {}).get("reasons")
                                or ["The reworded set's folded form was not read."]))

    reasons: List[str] = []
    fresh_ok = fv in ("lead", "lead, matched by a cheaper knob")
    if not fresh_ok:
        reasons.append(f'Fresh kept-back gate returned "{fv}": ' + "; ".join(fresh_judged.get("reasons") or ()))
    folded_ok = bool(folded_eval.get("ok"))
    if not folded_ok:
        reasons.extend(folded_eval.get("reasons") or ["Folded form check failed."])
    reworded_ok = bool(reworded_eval.get("ok"))
    if not reworded_ok:
        reasons.extend(reworded_eval.get("reasons") or ["Reworded item set check failed."])
    matched_knob = (fresh_judged.get("cheaper_knobs") or {}).get("matched_by")
    return out("passed" if (fresh_ok and folded_ok and reworded_ok) else "failed", reasons,
               matched_by_cheaper_knob=matched_knob)


def judge_readings(
    brief: lbr.Brief,
    bars: lcl.BarSet,
    trial_cfg: Mapping[str, Any],
    readings: Mapping[str, Optional[Mapping[str, Any]]],
    drawn: Mapping[str, Sequence[Any]],
    *,
    cheaper_knobs: Mapping[str, Any],
    practice_count: int,
    practice_best: Optional[float],
    base_checksum_unchanged: bool,
    base_after: bool,
    unpacker_fallbacks: int,
    void_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """From a trial's readings to its judgement: the one path for a trial as it is read (`run_trial_of_record`) and a
    trial judged again from a book (`rejudge_trial`).

    `readings`: the results (or the book's bodies) of the readings "matched", "same_site", "sham" (None unless the
    site was chosen by a declared rule), "reworded", "neutral", "nearmiss" and "panel". `drawn`: "target" and
    "reworded", each a list of things with `.item.name`, `.item.id`, `.frame_id` and `.half`. Returns
    {"fresh_judged", "folded", "reworded", "trial"}."""
    main_dose = float(trial_cfg["main_dose"])
    tol = float(trial_cfg["match_tolerance"])
    selection_mode = str(trial_cfg["selection_mode"])
    ftol = {"tol_floor_mult": float(trial_cfg["folded_tol_floor_mult"]), "tol_rel": float(trial_cfg["folded_tol_rel"]),
            "tol_min": float(trial_cfg["folded_tol_min"])}
    target_drawn, rw_drawn = list(drawn.get("target") or ()), list(drawn.get("reworded") or ())
    r_matched, r_same, r_sham = readings.get("matched"), readings.get("same_site"), readings.get("sham")
    r_rw = readings.get("reworded")
    r_neu, r_nm, r_pan = readings.get("neutral"), readings.get("nearmiss"), readings.get("panel")

    if void_reason is None:
        for tag, r, needed in (("other_sites_matched", r_matched, True), ("same_site", r_same, True),
                               ("sham_target", r_sham, selection_mode == "declared_rule"),
                               ("reworded", r_rw, True), ("neutral_text", r_neu, True), ("near_misses", r_nm, True),
                               ("labelled_panel", r_pan, True)):
            if r is None:
                if needed:
                    void_reason = f"The trial's {tag} reading was not taken."
                    break
                continue
            if r.get("status") != "ok":
                void_reason = f'The trial\'s {tag} reading ended "{r.get("status")}": {r.get("why")}'
                break

    classes_payload: Dict[str, Dict[str, Any]] = {}
    folded_by_class: Dict[str, Dict[str, Any]] = {}
    bat_payload: Dict[str, Dict[str, Any]] = {}
    reworded_eval: Optional[Dict[str, Any]] = None
    if void_reason is None:
        for cls_name in ("small", "large"):
            it_effs, fl_m, rev_m, ctl_m = lcl._extract_item_effects(r_matched, target_drawn, cls_name, brief.metric, main_dose)
            _, fl_s, _, ctl_s = lcl._extract_item_effects(r_same, target_drawn, cls_name, brief.metric, main_dose)
            fl_both = max([x for x in (fl_m, fl_s) if x is not None], default=None)
            ctls_cls: Dict[str, Any] = {"other_sites_matched": ctl_m, "same_site": ctl_s}
            deliv_cls: Dict[str, Any] = {
                "other_sites_matched": lcl._delivered_of(r_matched, target_drawn, cls_name, main_dose),
                "same_site": lcl._delivered_of(r_same, target_drawn, cls_name, main_dose),
            }
            if r_sham is not None:
                _, _, _, ctl_sh = lcl._extract_item_effects(r_sham, target_drawn, cls_name, brief.metric, main_dose)
                dv_sh = lcl._delivered_of(r_sham, target_drawn, cls_name, main_dose)
                if dv_sh is not None:
                    # a sham that moves the stream much more or less than the edit is not a control for it: it is
                    # counted against the edit (as the claim gate does), and the sheet says how many
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
            classes_payload[cls_name] = {"item_effects": it_effs, "floor": fl_both, "reversed_mean": rev_m,
                                         "controls": ctls_cls, "delivered": deliv_cls}
            folded_by_class[cls_name] = extract_folded_class(r_matched, target_drawn, cls_name, brief.metric)

        rw_classes: Dict[str, Dict[str, Any]] = {}
        rw_folded: Dict[str, Dict[str, Any]] = {}
        for cls_name in ("small", "large"):
            it_effs, fl_rw, _, ctl_rw = lcl._extract_item_effects(r_rw, rw_drawn, cls_name, brief.metric, main_dose)
            rw_classes[cls_name] = {"item_effects": it_effs, "floor": fl_rw, "controls": {"other_sites_matched": ctl_rw}}
            rw_folded[cls_name] = extract_folded_class(r_rw, rw_drawn, cls_name, brief.metric)
        reworded_eval = judge_reworded(brief, bars, rw_classes, rw_folded, **ftol)

        def both(r: Mapping[str, Any], metric: str, which: int) -> List[float]:
            got = [lcl._mean_metric_from_read(r, metric, main_dose, cls_name=c)[which] for c in ("small", "large")]
            return [v for v in got if v is not None]

        neu_kls, neu_ctls = both(r_neu, "kl", 0), both(r_neu, "kl", 1)
        bat_payload["neutral_text"] = {"mean_kl": max(neu_kls) if neu_kls else None,
                                       "control_kl_max": max(neu_ctls) if neu_ctls else None}
        nm_vals = both(r_nm, brief.metric, 0)
        bat_payload["near_misses"] = {"mean_effect": nm_vals[-1] if nm_vals else None,
                                      "abs_effect": max((abs(v) for v in nm_vals), default=None)}
        p_owns, p_kls = both(r_pan, "own", 0), both(r_pan, "kl", 0)
        bat_payload["labelled_panel"] = {"own_drop": max((max(0.0, -v) for v in p_owns), default=None),
                                         "mean_kl": max(p_kls) if p_kls else None}
        bat_payload["habitat"] = {"available": False, "note": "no atlas passages index mounted"}

    fresh_judged = lcl.judge_claim(
        brief=brief, bars=bars, classes=classes_payload, batteries=bat_payload, cheaper_knobs=dict(cheaper_knobs),
        selection_mode=selection_mode, practice_count=int(practice_count), practice_best=practice_best,
        void_reason=void_reason,
    )
    folded_eval = judge_folded(brief, bars, fresh_judged, folded_by_class, **ftol)
    trial_judged = judge_trial(
        brief, bars, fresh_judged, folded_eval=folded_eval, reworded_eval=reworded_eval,
        base_checksum_unchanged=base_checksum_unchanged, base_after=base_after,
        unpacker_fallbacks=unpacker_fallbacks, void_reason=void_reason,
    )
    return {"fresh_judged": fresh_judged, "folded": folded_eval, "reworded": reworded_eval, "trial": trial_judged}


# --------------------------------------------------------------------------------------------- the sheet & the sentence


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "none"
    return f"{float(x):.{digits}f}"


def _kind_words(meta: Mapping[str, Any]) -> str:
    if meta.get("rerun_of"):
        return "a volunteer's re-run"
    return "of record" if meta.get("of_record") else "a dry run"


def trial_sentence(
    brief: lbr.Brief,
    trial_cfg: Mapping[str, Any],
    fresh_judged: Mapping[str, Any],
    trial_judged: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Optional[str]:
    """The bounded sentence of a trial of record that passed, and of nothing else (a dry run and a re-run have none).

    It is the fresh reading's own sentence, as the claim gate writes it (dose, items and frames, interval, floor and
    bar, control edits, selection, which fresh reading of how many, batteries, cheaper knobs), with what the trial
    adds: the conditions it was of record under, each as it was found; the folded form; the reworded set and where it
    came from; and the claim reading it follows, if the book holds one."""
    if trial_judged.get("verdict") != "passed" or not meta.get("of_record") or meta.get("rerun_of"):
        return None
    base = lcl.claim_sentence(brief, fresh_judged, meta)
    if base is None:
        return None
    base = base.replace(" This is a lead, not a finding of record.", "")
    fl = ((trial_judged.get("folded") or {}).get("classes") or {}).get("large") or {}
    rw = trial_judged.get("reworded") or {}
    rw_cls = (rw.get("classes") or {}).get("large") or {}
    rw_fr = rw_cls.get("frames") or {}
    rw_ctl = (rw_cls.get("controls") or {}).get("other_sites_matched") or {}
    rw_fl = ((rw.get("folded") or {}).get("classes") or {}).get("large") or {}
    src = meta.get("source_claim")
    src_part = (f" It follows the claim reading of this edit at book line {src.get('seq')} ({src.get('verdict')})."
                if isinstance(src, Mapping) else " No claim reading of this edit stands in this book before it.")
    return (
        f'Trial of record "{trial_cfg.get("trial_name")}" (registration {str(meta.get("trial_sha256"))[:12]}, read at '
        f"commit {str(meta.get('repo_commit'))[:12]} from a clean checkout, in a process that had read nothing before "
        f"it, the registration folder committed and the runner and reader as registered): {base} "
        f"The folded form of the edit, the change written into the weights, read {_fmt((fl.get('frames') or {}).get('mean'))} "
        f"on the same items (largest gap from the per-row form {_fmt(fl.get('max_abs_gap'))}, tolerance "
        f"{_fmt(fl.get('tolerance'))}; the same bits on {fl.get('same_bits_count')} of {fl.get('taken_count')} items). "
        f"On reworded set {str(trial_cfg.get('reworded_id'))[:12]} ({trial_cfg.get('reworded_origin')}; read "
        f"{meta.get('reworded_use_index')} of at most {MAX_REWORDED_USES} times in this book) it moved {brief.metric} by "
        f"{_fmt(rw_fr.get('mean'))} (half 0 {_fmt((rw_fr.get('half_0') or {}).get('mean'))}, half 1 "
        f"{_fmt((rw_fr.get('half_1') or {}).get('mean'))}; {rw_ctl.get('at_least')} of {rw_ctl.get('count')} matched "
        f"control edits moved as much, p {_fmt(rw_ctl.get('p_fw'), 3)}; folded {_fmt((rw_fl.get('frames') or {}).get('mean'))})."
        f"{src_part} The fresh items are new fillings of the brief's kept-back frames, which a claim reading of the "
        "brief has already read. Passed. It is of this base checkpoint and these items: an adapter made from the edit "
        "and served over another quantisation is another object, to be measured there."
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
    """The trial's sheet: what kind of trial it was and why, the conditions as found, the fresh reading as the claim
    gate's own sheet prints it (with every earlier fresh reading and trial of the brief), the folded form, the
    reworded set, and the sentence when there is one."""
    kind = _kind_words(meta)
    head = {"of record": "TRIAL OF RECORD", "a dry run": "DRY RUN OF A TRIAL (not of record)",
            "a volunteer's re-run": "RE-RUN OF A TRIAL (a volunteer's; not of record)"}[kind]
    verdict = trial_judged["verdict"]
    out: List[str] = [
        f"=== {head}: {trial_cfg['trial_name']} ===",
        f"verdict: {verdict}" + ("" if kind == "of record" else f" ({kind}: not a verdict of record)"),
    ]
    for why in meta.get("not_of_record") or ():
        out.append(f"  not of record: {why}")
    proc = meta.get("process") or {}
    n_before = int(proc.get("readings", 0)) + int(proc.get("captures", 0))
    co = meta.get("checkout") or {}
    out += [
        f"registration: {str(meta.get('trial_sha256') or '')[:16]} (registered {trial_cfg.get('registered_at')} at commit "
        f"{str(trial_cfg.get('repo_commit'))[:12]})"
        + (f" | re-run of home registration {str(meta.get('rerun_of'))[:16]}" if meta.get("rerun_of") else ""),
        f"read at commit {str(meta.get('repo_commit'))[:12]}: checkout "
        + ("clean" if co.get("dirty") is False else ("not known to git" if co.get("dirty") is None else
                                                     f"with {co.get('changed')} path(s) changed"))
        + f" | registration folder committed: {'yes' if co.get('folder_committed') else 'no'}"
        + f" | runner and reader as registered: {'yes' if co.get('sources_match') else 'no'}",
        "process: " + ("had read nothing before the trial" if n_before == 0 else
                       f"had begun {proc.get('readings', 0)} reading(s) and {proc.get('captures', 0)} capture(s) before the trial"),
        f"base: checksum unchanged={meta.get('base_checksum_unchanged')} over {meta.get('base_linears_checked')} linears "
        f"({meta.get('base_tensors_checked')} tensors); overlays {meta.get('overlays_in_force')}, unpacker fallbacks "
        f"{meta.get('unpacker_fallbacks')}",
        f"edit: {str(trial_cfg['edit_id'])[:16]} ({trial_cfg['edit_provenance']}) at dose {trial_cfg['main_dose']:g} | "
        f"linears: {', '.join(trial_cfg.get('edit_linears') or ())}",
    ]
    src = meta.get("source_claim")
    out.append(f"follows: the claim reading of this edit at book line {src.get('seq')} ({src.get('verdict')})"
               if isinstance(src, Mapping) else "follows: no claim reading of this edit stands in this book")
    out.append("fresh items: new fillings of the brief's kept-back frames, the frames a claim reading of this brief "
               "has already read; the reworded set below is what stands outside them")
    if trial_judged.get("matched_by_cheaper_knob"):
        out.append(f"cheaper knob: matched by {trial_judged['matched_by_cheaper_knob']} (the verdict stands; the "
                   "sentence says so)")

    out += ["", "--- the fresh reading, as the claim gate's sheet prints a fresh reading ---"]
    out += ["  " + ln for ln in lcl.render_sheet(brief, bars, fresh_judged, meta, None).splitlines()]

    out += ["", "--- the folded form (the change written into the weights), beside the per-row form ---"]
    folded_classes = (trial_judged.get("folded") or {}).get("classes") or {}
    if not folded_classes:
        out.append("  not read")
    for cls_name, fc in folded_classes.items():
        if fc.get("available"):
            ffr = fc["frames"]
            out.append(
                f"  [{cls_name}] folded mean {_fmt(ffr['mean'])} | h0 {_fmt(ffr['half_0']['mean'])}, "
                f"h1 {_fmt(ffr['half_1']['mean'])} | from the per-row mean {_fmt(fc.get('mean_diff_vs_per_row'))}, largest "
                f"item gap {_fmt(fc.get('max_abs_gap'))} (tolerance {_fmt(fc.get('tolerance'))}) | same bits "
                f"{fc.get('same_bits_count')}/{fc.get('taken_count')} | ok={fc.get('ok')}"
            )
        else:
            out.append(f"  [{cls_name}] folded form not read ({fc.get('why')})")

    out += ["", f"--- the reworded set {str(trial_cfg.get('reworded_id') or '')[:12]} ({trial_cfg.get('reworded_origin')}; "
                f"read {meta.get('reworded_use_index')} of at most {MAX_REWORDED_USES} times in this book; "
                f"{trial_cfg.get('reworded_controls')} matched control edits, judged alone at the level) ---"]
    rw = trial_judged.get("reworded")
    if not isinstance(rw, Mapping):
        out.append("  not read")
    else:
        rw_fl_classes = ((rw.get("folded") or {}).get("classes") or {})
        for cls_name, rc in (rw.get("classes") or {}).items():
            fr = rc.get("frames") or {}
            ctl = (rc.get("controls") or {}).get("other_sites_matched") or {}
            hb = rc.get("house_bar") or {}
            out.append(
                f"  [{cls_name}] per-row mean {_fmt(fr.get('mean'))} | h0 {_fmt((fr.get('half_0') or {}).get('mean'))}, "
                f"h1 {_fmt((fr.get('half_1') or {}).get('mean'))} | floor {_fmt(hb.get('floor'))}, bar {_fmt(hb.get('bar'))} | "
                f"matched {ctl.get('at_least')}/{ctl.get('count')} (p {_fmt(ctl.get('p_fw'), 3)}) | ok={rc.get('ok')}"
            )
            rfc = rw_fl_classes.get(cls_name) or {}
            if rfc.get("available"):
                out.append(
                    f"  [{cls_name}] folded mean {_fmt((rfc.get('frames') or {}).get('mean'))} | largest item gap "
                    f"{_fmt(rfc.get('max_abs_gap'))} (tolerance {_fmt(rfc.get('tolerance'))}) | ok={rfc.get('ok')}"
                )
            else:
                out.append(f"  [{cls_name}] folded form not read ({rfc.get('why')})")

    if trial_judged.get("reasons"):
        out += ["", "--- why ---"]
        out += [f"  - {r}" for r in trial_judged["reasons"]]
    if sentence:
        out += ["", "--- the sentence of record ---", sentence]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------- reading a trial


def _trial_seen_in_book(trial_sha256: str, book_path: Path) -> Optional[int]:
    """The line a registration of this hash was entered at in the book, or None."""
    if not book_path.exists():
        return None
    for ent in lb.entries(book_path):
        body = ent.get("body")
        if ent.get("kind") == "note" and isinstance(body, Mapping):
            if body.get("note_type") == "trial_registration" and body.get("trial_sha256") == trial_sha256:
                return int(ent["seq"])
    return None


def record_conditions(prereg_dir: Path, trial_cfg: Mapping[str, Any], *, repo_root: Path = ROOT_DIR) -> Dict[str, Any]:
    """What a trial must find to be of record, each as it is found now: `reasons` lists, in words, every one that
    does not hold (empty when the trial may be of record)."""
    asked = lv.asked_in_process()
    commit, dirty = git_state(repo_root)
    changed = checkout_changes(repo_root)
    not_committed = folder_committed(Path(prereg_dir), repo_root)
    now, then = source_hashes(ROOT_DIR), dict(trial_cfg.get("sources") or {})
    moved = sorted(k for k in SOURCE_FILES if now.get(k) != then.get(k))
    reasons: List[str] = []
    if asked["readings"] or asked["captures"]:
        reasons.append(f"this process had already begun {asked['readings']} reading(s) and {asked['captures']} "
                       "capture(s): a trial of record is the first thing its process reads")
    if dirty is None:
        reasons.append("the checkout is not one git can be asked about, so it cannot be shown clean")
    elif dirty:
        some = ", ".join((changed or [])[:3])
        reasons.append(f"the checkout has {len(changed or [])} uncommitted path(s) ({some}"
                       + (", ..." if changed and len(changed) > 3 else "") + ")")
    if not_committed is not None:
        reasons.append(not_committed)
    if moved:
        reasons.append("the runner or reader is not the one registered (" + ", ".join(moved) + ")")
    return {
        "reasons": reasons,
        "process": asked,
        "checkout": {"commit": commit, "dirty": dirty, "changed": len(changed) if changed is not None else None,
                     "folder_committed": not_committed is None, "sources_match": not moved},
    }


def _as_drawn(rows: Sequence[Mapping[str, Any]]) -> List[Any]:
    """A note's record of drawn items (name, id, frame, half) as the things the readers take."""
    return [SimpleNamespace(item=SimpleNamespace(name=r["name"], id=r["item_id"]), frame_id=r["frame_id"],
                            half=int(r["half"])) for r in rows]


def _drawn_rows(drawn: Sequence[Any]) -> List[Dict[str, Any]]:
    return [{"name": d.item.name, "item_id": d.item.id, "frame_id": d.frame_id, "half": int(d.half)} for d in drawn]


def run_trial_of_record(
    h: lv.Handle,
    prereg_dir: Path,
    *,
    out_dir: Path,
    payload_dir: Optional[Path] = None,
    book_path: Optional[Path] = None,
    dry_run: bool = False,
    rerun: Optional[Mapping[str, Any]] = None,
    _repo_root: Optional[Path] = None,
    _seed_override: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Read one registered trial.

    Before anything is written: the folder is verified (`verify_registration`); the trial is refused if this
    registration is already in the book or `out_dir` holds a result; the registered edit is built again on `h` and
    kept to its id; the fresh-reading gate is asked (`level_claim.fresh_reading_gate`, as for a claim, with the
    brief's budget counting claims and trials together and one trial an edit); the bars must be the calibration the
    folder records, in this book; the reworded set must not be worn out (`MAX_REWORDED_USES`) or read in practice
    here; and the conditions of record are looked at (`record_conditions`). If any of those does not hold the trial
    is refused, unless `dry_run`: then it is read as a dry run, which is judged and counted as a trial is and is
    never of record. (`dry_run=True` makes a dry run of a trial that could have been of record, too.)

    Then: the checksum of the base (`weight_edits.base_checksum`), the registration note, the draw from
    `os.urandom(32)`, the readings (the fresh items against matched other-site control edits with the folded form
    beside, against same-site ones, against sham-target ones for a declared rule; the reworded set with the folded
    form beside; the three gating batteries; the brief's prompt arms), the checksum again, the judgement
    (`judge_readings`), the verdict note (`trial_of_record`, `trial_dry_run` or `trial_rerun`), and
    `trial_result.json` and `TRIAL_SHEET.md` in `out_dir`. The result file holds the seed of the draw, so that a
    volunteer's kit can ask for the same items; the book does not.

    `rerun`: a volunteer's re-run of a kit (`level_rerun.run_volunteer_rerun`), {"of": the home registration's hash,
    "seed": the home draw's seed}: the folder is a kit, the bars' calibration is the one the registration records
    (a volunteer's book has none), the items are the home's, and the result is a re-run, never of record."""
    is_rerun = rerun is not None
    reg = verify_registration(prereg_dir, kit=is_rerun)
    trial_cfg: Dict[str, Any] = reg["trial"]
    brief: lbr.Brief = reg["brief"]
    bars: lcl.BarSet = reg["bars"]
    edit_man: Dict[str, Any] = reg["edit_manifest"]
    reworded: RewordedSet = reg["reworded"]
    trial_sha: str = reg["trial_sha256"]
    if is_rerun and (not isinstance(rerun.get("of"), str) or not isinstance(rerun.get("seed"), (bytes, bytearray))
                     or len(rerun["seed"]) != 32):
        raise TrialRefused("A re-run names the home registration it is of and carries the 32 bytes of its draw.")

    out_path = Path(out_dir).resolve()
    if (out_path / "trial_result.json").exists():
        raise TrialRefused(
            f"Read once: {out_path / 'trial_result.json'} already exists. A trial of record is read once."
        )
    bk = Path(book_path).expanduser().resolve() if book_path is not None else lb.book_path().resolve()
    seen_at = _trial_seen_in_book(trial_sha, bk)
    if seen_at is not None:
        raise TrialRefused(
            f"Read once: trial registration {trial_sha[:12]} has already been recorded in {bk.name} (line {seen_at})."
        )

    we.assert_base(h.lm, h.hooks_at_load)
    edit = load_registered_edit(h, edit_man, payload_dir=payload_dir)
    no_fold = we.can_fold(h.lm, edit)
    if no_fold is not None:
        raise TrialRefused(f"A trial reads the folded form of its edit, and this edit has none on this model: {no_fold}")
    try:
        lcl._check_edit_allowed_by_brief(edit, brief)
        gate = lcl.fresh_reading_gate(h, brief, bars, edit, bk, what="trial",
                                      calibration=dict(trial_cfg["bars_calibration"]) if is_rerun else None)
    except lcl.ClaimRefused as exc:
        raise TrialRefused(str(exc)) from None
    hist, cal_note = gate["hist"], gate["cal_note"]
    if not is_rerun:
        cal_reg = trial_cfg["bars_calibration"]
        cal_here = next(lb.entries(bk, start=int(cal_note["seq"]), limit=1))
        if int(cal_note["seq"]) != int(cal_reg["seq"]) or cal_here["hash"] != cal_reg.get("hash"):
            raise TrialRefused(
                f"The registration names the calibration at line {cal_reg['seq']} of {cal_reg.get('book')}; the "
                f"calibration of these bars in {bk.name} is another line (line {cal_note['seq']}): a trial is read in "
                "the book it was registered against."
            )

    main_dose = float(trial_cfg["main_dose"])
    doses = [main_dose, -main_dose, 0.0] if brief.direction != 0 else [main_dose, 0.0]
    for d in doses:
        dw = we.dose_refusal(edit, d, h.dtype)
        if dw:
            raise TrialRefused(f"Registered edit cannot be read at dose {d:g}: {dw}")
    selection_mode = str(trial_cfg["selection_mode"])
    sham_recipes: Optional[List[Dict[str, Any]]] = None
    if selection_mode == "declared_rule":
        sham_recipes = [dict(r) for r in trial_cfg.get("sham_recipes") or ()]
        try:
            shams = lcl._check_shams(h, brief, edit, sham_recipes, Path(prereg_dir))
        except lcl.ClaimRefused as exc:
            raise TrialRefused(str(exc)) from None
        if [s.id for s in shams] != list(trial_cfg.get("sham_edit_ids") or ()):
            raise TrialRefused("The registered sham recipes do not build the sham edits the registration names.")

    rw_uses_before = reworded_uses_in_book(reworded, bk)
    if rw_uses_before >= MAX_REWORDED_USES:
        raise TrialRefused(
            f'Reworded set "{reworded.name}" ({reworded.short()}) holds an item already read {rw_uses_before} times '
            f"in {bk.name} (at most {MAX_REWORDED_USES}): write a new set."
        )
    rw_practised = reworded_practised(reworded, bk)
    if rw_practised:
        raise TrialRefused(
            f'Reworded set "{reworded.name}" was read in practice in this book (line'
            f'{"s" if len(rw_practised) > 1 else ""} {", ".join(str(s) for s in rw_practised[:8])}): a set the edit '
            "may have been picked on is not outside what it was picked on. Write a new set."
        )

    cond = record_conditions(Path(prereg_dir), trial_cfg, repo_root=Path(_repo_root) if _repo_root else ROOT_DIR)
    not_of_record = list(cond["reasons"])
    if is_rerun:
        # a re-run is never of record; what a home trial needs of its checkout and folder is not asked of it, and
        # what it found (its process, its checkout, whether its runner and reader are the home's) goes back with it
        not_of_record = ["a volunteer's re-run of a registered trial"]
    elif not_of_record and not dry_run:
        raise TrialRefused(
            "This trial would not be of record: " + "; ".join(not_of_record) + ". Mend that, or read it as a dry "
            "run (dry_run=True; --dry-run on the command line): a dry run is judged and counted as a trial is, "
            "spends a fresh reading of the brief all the same, and is never a verdict of record."
        )
    elif dry_run and not not_of_record:
        not_of_record = ["asked for as a dry run"]
    of_record = not not_of_record

    C = int(trial_cfg["controls_per_family"])
    rows_cfg = dict(trial_cfg["rows"])
    tol = float(trial_cfg["match_tolerance"])
    reading_index = int(gate["reading_index"])
    src = next((v for v in reversed(hist["verdicts"]) if v.get("edit_id") == edit.id), None)
    source_claim = ({"seq": int(src["seq"]), "verdict": src.get("verdict"), "reading_index": src.get("reading_index")}
                    if src else None)
    src_reg = next((r for r in hist["registrations"] if r.get("edit_id") == edit.id), None)
    practice_best = _clean_float((src_reg or {}).get("practice_best"))
    kb_frames = brief.generator.frames_for("kept_back")
    published = lbr.published_template_ids(h.tokenizer)
    kept_back_note = {"frames": len(kb_frames),
                      "published": sum(1 for fr in kb_frames if fr.template_id in published),
                      "template_ids": gate["kb_tids"]}
    design = {
        "n_target": int(trial_cfg["n_target"]), "n_neutral": int(trial_cfg["n_neutral"]),
        "n_near": int(trial_cfg["n_near"]), "n_panel": int(trial_cfg["n_panel"]),
        "rows": {k: int(v) for k, v in rows_cfg.items()}, "written_rows": 0,
        "controls_per_family": C, "battery_controls": min(C, 3), "match_tolerance": tol,
        "doses": [float(d) for d in doses], "prompt_arms": True,
        "reworded_controls": int(trial_cfg["reworded_controls"]),
    }
    shams_note = None
    if sham_recipes:
        sham_ids = list(trial_cfg.get("sham_edit_ids") or ())
        shams_note = {"count": len(sham_ids), "edit_ids": sham_ids,
                      "sha256": hashlib.sha256(lb.canonical(sham_ids)).hexdigest(),
                      "from": "registered with the trial; checked for shape only"}

    # The base, before anything is read of it
    t_start = time.perf_counter()
    ck_before = we.base_checksum(h.lm, hooks_at_load=h.hooks_at_load)
    if int(ck_before["linears_checked"]) <= 0:
        raise TrialRefused("The base checksum found no linear to hash on this model: a trial is not read on it.")
    fb_before = lv._fell_back()
    commit_now = cond["checkout"]["commit"]

    prev_bk = os.environ.get(lb.BOOK_ENV)
    os.environ[lb.BOOK_ENV] = str(bk)
    try:
        # Step 1: the trial is entered in the book BEFORE its fresh items are drawn
        reg_body = {
            "note_type": "trial_registration",
            "trial_name": trial_cfg["trial_name"],
            "trial_sha256": trial_sha,
            "of_record": of_record,
            "not_of_record": not_of_record,
            "rerun_of": str(rerun["of"]) if is_rerun else None,
            "brief_id": brief.id,
            "brief_name": brief.name,
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "edit_provenance": we.edit_provenance(edit),
            "main_dose": main_dose,
            "reading_index": reading_index,
            "max_fresh_readings": int(brief.max_fresh_readings),
            "controls_per_family": C,
            "rule_preset": trial_cfg["rule_preset"],
            "rule_version": trial_cfg["v"],
            "bars_sha256": bars.id,
            "bars_calibration_seq": int(cal_note["seq"]),
            "selection_mode": selection_mode,
            "declared_rule": trial_cfg.get("declared_rule"),
            "practice_count": int(hist["practice_readings"]),
            "practice_best": practice_best,
            "source_claim": source_claim,
            "design": design,
            "shams": shams_note,
            "kept_back": kept_back_note,
            "book": gate["book"],
            "book_fresh_readings": dict(hist["book_fresh_readings"]),
            "reworded_id": reworded.id,
            "reworded_wordings": reworded.wordings(),
            "reworded_origin": reworded_origin(reworded, brief),
            "reworded_use_index": rw_uses_before + 1,
            "prior_verdicts": [
                {"seq": v["seq"], "edit_id": v.get("edit_id"), "verdict": v.get("verdict"),
                 "reading_index": v.get("reading_index"), "brief_id": v.get("brief_id")}
                for v in hist["verdicts"]
            ],
            "prior_trials": lcl.prior_trials(hist),
            "process": cond["process"],
            "checkout": cond["checkout"],
            "base_checksum_before": ck_before["sha256"],
            "base_linears_checked": ck_before["linears_checked"],
            "base_tensors_checked": ck_before["tensors_checked"],
            "repo_commit": commit_now,
        }
        reg_entry = lb.append("note", reg_body, path=bk)
        book_seqs: List[int] = [int(reg_entry["seq"])]

        # Step 2: the draw, after the registration
        if is_rerun:
            seed_bytes = bytes(rerun["seed"])
        else:
            seed_bytes = bytes(_seed_override) if _seed_override is not None else os.urandom(32)
        seed_sha = _sha_bytes(seed_bytes)
        seed_u32 = int.from_bytes(seed_bytes[:4], "little") & 0x7FFFFFFF

        void_reason: Optional[str] = None
        try:
            target_drawn = brief.generator.draw(
                "kept_back", int(trial_cfg["n_target"]), seed_bytes=seed_bytes + b":trial_target",
                tokenizer=h.tokenizer, system_override=(brief.chat_settings or {}).get("system"),
                forbid_item_ids=gate["forbid_items"], forbid_frame_ids=gate["forbid_frames"],
                name_prefix="trial_target",
            )
            near_drawn = brief.near_miss_generator.draw(
                "kept_back", int(trial_cfg["n_near"]), seed_bytes=seed_bytes + b":trial_near",
                tokenizer=h.tokenizer, system_override=(brief.chat_settings or {}).get("system"),
                name_prefix="trial_near",
            )
            neu_drawn = lbr.neutral_chat_generator().draw(
                "kept_back", int(trial_cfg["n_neutral"]), seed_bytes=seed_bytes + b":trial_neu",
                tokenizer=h.tokenizer, name_prefix="trial_neu",
            )
            panel = lbr.draw_labelled_sanity_panel(int(trial_cfg["n_panel"]), seed_bytes=seed_bytes + b":trial_panel",
                                                   tokenizer=h.tokenizer)
        except lbr.BriefError as exc:
            void_reason = f"Fresh draw refused: {exc}"
            target_drawn, near_drawn, neu_drawn, panel = [], [], [], []
        rw_drawn = reworded.to_drawn_items(h.tokenizer, system_override=(brief.chat_settings or {}).get("system"))

        target_items = {d.item.name: d.item for d in target_drawn}
        near_items = {d.item.name: d.item for d in near_drawn}
        neu_items = {d.item.name: d.item for d in neu_drawn}
        panel_items = {p["item"].name: p["item"] for p in panel}
        rw_items = {d.item.name: d.item for d in rw_drawn}

        readings: Dict[str, Optional[Dict[str, Any]]] = {k: None for k in
                                                         ("matched", "same_site", "sham", "reworded", "neutral",
                                                          "nearmiss", "panel")}
        tname = trial_cfg["trial_name"]
        unpacker_mode: Optional[str] = None

        def take(role: str, plan: Dict[str, Any], items: Mapping[str, Any]) -> bool:
            """One reading of the trial; False once one has not ended ok (the rest are then not taken)."""
            nonlocal void_reason
            r = lv.read(h, plan, items, edit=edit)
            readings[role] = r
            book_seqs.append(int(r["book"]["seq"]))
            if r.get("status") != "ok":
                void_reason = f'The trial\'s {role} reading ended "{r.get("status")}": {r.get("why")}'
                return False
            return True

        if void_reason is None:
            base_plan = {"kind": "paired", "edit": {"built": edit.id}, "doses": doses, "main_dose": main_dose,
                         "rows": rows_cfg, "readouts": dict(brief.readouts), "rank_by": brief.metric}
            light = {"kind": "paired", "edit": {"built": edit.id}, "doses": [main_dose, 0.0], "rows": rows_cfg,
                     "folded": "never"}
            plain = {"kl": True, "own": True, "odds": None}
            bat_n = min(C, 3)
            steps = [
                ("matched", {**base_plan, "name": f"trial_{tname}_matched", "folded": "always",
                             "controls": {"family": "other_sites_matched", "count": C, "seed": seed_u32,
                                          "tolerance": tol}}, target_items),
                ("same_site", {**base_plan, "name": f"trial_{tname}_same_site", "folded": "never",
                               "controls": {"family": "same_site", "count": C, "seed": seed_u32 + 1}}, target_items),
            ]
            if selection_mode == "declared_rule" and sham_recipes:
                steps.append(("sham", {**base_plan, "name": f"trial_{tname}_sham", "folded": "never",
                                       "controls": {"family": "sham_target", "shams": sham_recipes,
                                                    "seed": seed_u32 + 2}}, target_items))
            steps += [
                ("reworded", {"name": f"trial_{tname}_reworded", "kind": "paired", "edit": {"built": edit.id},
                              "doses": [main_dose, 0.0], "main_dose": main_dose, "rows": rows_cfg,
                              "readouts": dict(brief.readouts), "rank_by": brief.metric, "folded": "always",
                              "controls": {"family": "other_sites_matched", "count": int(trial_cfg["reworded_controls"]),
                                           "seed": seed_u32 + 11, "tolerance": tol}}, rw_items),
                ("neutral", {**light, "name": f"trial_{tname}_neutral", "readouts": plain,
                             "controls": {"family": "other_sites", "count": bat_n, "seed": seed_u32 + 3}}, neu_items),
                ("nearmiss", {**light, "name": f"trial_{tname}_nearmiss", "readouts": dict(brief.readouts),
                              "controls": {"family": "other_sites", "count": bat_n, "seed": seed_u32 + 4}}, near_items),
                ("panel", {**light, "name": f"trial_{tname}_panel", "readouts": plain,
                           "controls": {"family": "other_sites", "count": bat_n, "seed": seed_u32 + 5}}, panel_items),
            ]
            for role, plan, items in steps:
                if not take(role, plan, items):
                    break
            if readings["matched"] is not None:
                unpacker_mode = (readings["matched"].get("runtime") or {}).get("unpacker", {}).get("mode")

        # Step 3: the brief's prompt arms beside the edit (cheaper knobs)
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

        # Step 4: the base again, and the unpacker
        try:
            ck_after = we.base_checksum(h.lm, hooks_at_load=h.hooks_at_load)
            base_after = True
        except Exception:
            ck_after = {"sha256": None, "linears_checked": 0, "tensors_checked": 0, "overlays_in_force": -1,
                        "hooks_now": -1}
            base_after = False
        base_checksum_unchanged = bool(
            base_after
            and ck_after["sha256"] == ck_before["sha256"]
            and ck_after["linears_checked"] == ck_before["linears_checked"]
            and ck_after["tensors_checked"] == ck_before["tensors_checked"]
        )
        fb_after = lv._fell_back()
        unpacker_fallbacks = int(fb_after or 0) - int(fb_before or 0)

        # Step 5: the judgement, by the one path a book's readings are judged again by
        drawn = {"target": target_drawn, "reworded": rw_drawn}
        judged = judge_readings(
            brief, bars, trial_cfg, readings, drawn, cheaper_knobs=cheaper_knobs,
            practice_count=int(hist["practice_readings"]), practice_best=practice_best,
            base_checksum_unchanged=base_checksum_unchanged, base_after=base_after,
            unpacker_fallbacks=unpacker_fallbacks, void_reason=void_reason,
        )
        fresh_judged, folded_eval = judged["fresh_judged"], judged["folded"]
        reworded_eval, trial_judged = judged["reworded"], judged["trial"]

        elapsed_s = round(time.perf_counter() - t_start, 3)
        meta = {
            # what the claim gate's sheet and sentence read of a fresh reading
            "date": _utc_now(),
            "model_id": h.model_id,
            "weights_id": h.weights_id,
            "unpacker": unpacker_mode or "none",
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "main_dose": main_dose,
            "brief_id": brief.id,
            "brief_name": brief.name,
            "bars_sha256": bars.id,
            "bars_calibration_seq": int(cal_note["seq"]),
            "bars_calibrations_in_book": len(hist["calibrations"]),
            "declared_rule": trial_cfg.get("declared_rule"),
            "reading_index": reading_index,
            "max_fresh_readings": int(brief.max_fresh_readings),
            "draw_seed_sha256": seed_sha,
            "n_target_items": len(target_drawn),
            "registration_seq": int(reg_entry["seq"]),
            "book_seqs": book_seqs,
            "prior_verdicts": reg_body["prior_verdicts"],
            "prior_trials": reg_body["prior_trials"],
            "prompt_named_site": brief.prompt_named_site,
            "chat_settings": dict(brief.chat_settings),
            "forecast": dict(trial_cfg.get("forecast") or {}),
            "design": design,
            "shams": shams_note,
            "kept_back": kept_back_note,
            "book": gate["book"],
            "book_fresh_readings": dict(hist["book_fresh_readings"]),
            "written_strata": None,
            "settings_audit": None,
            # what is the trial's own
            "finished_at": _utc_now(),
            "device": str(h.runtime.get("device") or h.device),
            "trial_sha256": trial_sha,
            "of_record": of_record,
            "not_of_record": not_of_record,
            "rerun_of": str(rerun["of"]) if is_rerun else None,
            "process": cond["process"],
            "checkout": cond["checkout"],
            "repo_commit": commit_now,
            "source_claim": source_claim,
            "book_name": bk.name,
            "base_checksum_before": ck_before["sha256"],
            "base_checksum_after": ck_after["sha256"],
            "base_checksum_unchanged": base_checksum_unchanged,
            "base_linears_checked": ck_before["linears_checked"],
            "base_tensors_checked": ck_before["tensors_checked"],
            "overlays_in_force": ck_after["overlays_in_force"],
            "base_after": base_after,
            "unpacker_fallbacks": unpacker_fallbacks,
            "reworded_use_index": rw_uses_before + 1,
            "seconds": elapsed_s,
        }
        sentence = trial_sentence(brief, trial_cfg, fresh_judged, trial_judged, meta)
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
            "note_type": NOTE_RERUN if is_rerun else (NOTE_OF_RECORD if of_record else NOTE_DRY_RUN),
            "trial_name": trial_cfg["trial_name"],
            "trial_sha256": trial_sha,
            "of_record": of_record,
            "not_of_record": not_of_record,
            "rerun_of": str(rerun["of"]) if is_rerun else None,
            "brief_id": brief.id,
            "brief_name": brief.name,
            "edit_id": edit.id,
            "edit_linears": edit.linears(),
            "edit_provenance": trial_cfg["edit_provenance"],
            "main_dose": main_dose,
            "reading_index": reading_index,
            "max_fresh_readings": int(brief.max_fresh_readings),
            "verdict": trial_judged["verdict"],
            "fresh_verdict": trial_judged["fresh_verdict"],
            "reasons": trial_judged["reasons"],
            "bars_sha256": bars.id,
            "registration_seq": int(reg_entry["seq"]),
            "book_seqs": book_seqs,
            "readings": {role: (int(r["book"]["seq"]) if r is not None else None) for role, r in readings.items()},
            "drawn": {"target": _drawn_rows(target_drawn), "reworded": _drawn_rows(rw_drawn)},
            "draw_seed_sha256": seed_sha,
            "claim_sentence": sentence,
            "summary": summary_compact,
            "folded": {
                "ok": folded_eval["ok"],
                "status": folded_eval["status"],
                "classes": {
                    cls: {
                        "ok": fc.get("ok"),
                        "available": fc.get("available"),
                        "mean": (fc.get("frames") or {}).get("mean"),
                        "max_abs_gap": fc.get("max_abs_gap"),
                        "tolerance": fc.get("tolerance"),
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
                    "reworded_id": reworded.id,
                    "origin": reworded_origin(reworded, brief),
                    "classes": {
                        cls: {"ok": rc.get("ok"), "mean": (rc.get("frames") or {}).get("mean")}
                        for cls, rc in (reworded_eval.get("classes") or {}).items()
                    },
                }
                if reworded_eval is not None else None
            ),
            "batteries": fresh_judged.get("batteries"),
            "cheaper_knobs": fresh_judged.get("cheaper_knobs"),
            "practice_count": int(hist["practice_readings"]),
            "practice_best": practice_best,
            "base_checksum_unchanged": base_checksum_unchanged,
            "base_after": base_after,
            "unpacker_fallbacks": unpacker_fallbacks,
            "seconds": elapsed_s,
        }
        v_entry = lb.append("note", verdict_body, path=bk)
        we.assert_base(h.lm, h.hooks_at_load)

        result = {
            "v": TRIAL_VERSION,
            "trial_name": trial_cfg["trial_name"],
            "trial_sha256": trial_sha,
            "of_record": of_record,
            "not_of_record": not_of_record,
            "rerun_of": str(rerun["of"]) if is_rerun else None,
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
            # the seed of the draw: in this file, which is the owner's, and not in the book
            "seed_hex": seed_bytes.hex(),
            "book": {
                "path": str(bk),
                "name": bk.name,
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


# --------------------------------------------------------------------------------------------- a trial, from its book


def find_trial(book_path: Path, trial_sha256: str) -> Optional[Dict[str, Any]]:
    """The registration note and the verdict note of the trial with this registration hash in the book, with their
    lines: {"registration", "registration_seq", "verdict", "verdict_seq"} (the verdict None when the trial never
    ended), or None when the book holds no such registration."""
    p = Path(book_path)
    if not p.exists():
        return None
    reg: Optional[Dict[str, Any]] = None
    reg_seq: Optional[int] = None
    for ent in lb.entries(p):
        body = ent.get("body")
        if ent.get("kind") != "note" or not isinstance(body, Mapping):
            continue
        nt = body.get("note_type")
        if nt == "trial_registration" and body.get("trial_sha256") == trial_sha256:
            reg, reg_seq = dict(body), int(ent["seq"])
        elif nt in lcl.TRIAL_VERDICT_NOTES and reg_seq is not None and body.get("registration_seq") == reg_seq:
            return {"registration": reg, "registration_seq": reg_seq, "verdict": dict(body),
                    "verdict_seq": int(ent["seq"])}
    if reg is None:
        return None
    return {"registration": reg, "registration_seq": reg_seq, "verdict": None, "verdict_seq": None}


def rejudge_trial(book_path: Path, trial_sha256: str, reg: Mapping[str, Any]) -> Dict[str, Any]:
    """Judge a trial again from its book, with this module's code.

    `reg`: the registration as `verify_registration` returns it (the brief, the bars and `trial.json` the trial was
    registered with). The verdict note names the lines of its readings and which drawn item is which frame and half;
    the readings are taken from those lines and judged by `judge_readings`, the path the trial itself was judged by.
    What is not a reading and so is taken as the note records it: the prompt arms' figures, the practice count, and
    whether the base and the unpacker were as they should be.

    Returns {"verdict", "fresh_verdict", "as_noted" (whether the verdict judged again is the note's), "noted_verdict",
    "items" (per class, each fresh item's effect by item id; and the reworded items'), "readings" (for each role its
    line, name, status, base, edit id, control seed and count, and item ids), "judged"}. Raises `TrialRefused` when
    the book does not hold the trial whole."""
    p = Path(book_path)
    found = find_trial(p, trial_sha256)
    if found is None:
        raise TrialRefused(f"{p.name} holds no trial registered as {trial_sha256[:12]}.")
    note = found["verdict"]
    if note is None:
        raise TrialRefused(f"The trial {trial_sha256[:12]} was registered in {p.name} and never ended.")
    brief, bars, trial_cfg = reg["brief"], reg["bars"], reg["trial"]
    roles = note.get("readings")
    drawn_rows = note.get("drawn")
    if not isinstance(roles, Mapping) or not isinstance(drawn_rows, Mapping):
        raise TrialRefused("The trial's verdict note does not name its readings and its drawn items.")
    readings: Dict[str, Optional[Dict[str, Any]]] = {}
    facts: Dict[str, Any] = {}
    for role in ("matched", "same_site", "sham", "reworded", "neutral", "nearmiss", "panel"):
        seq = roles.get(role)
        if seq is None:
            readings[role] = None
            continue
        if isinstance(seq, bool) or not isinstance(seq, int) or not found["registration_seq"] < seq < found["verdict_seq"]:
            raise TrialRefused(f"The trial's {role} reading is not a line between its registration and its verdict.")
        ent = next(lb.entries(p, start=int(seq), limit=1), None)
        if ent is None or ent.get("kind") != "paired" or not isinstance(ent.get("body"), Mapping):
            raise TrialRefused(f"Line {seq} of {p.name} is not the paired reading the trial names as its {role}.")
        body = dict(ent["body"])
        readings[role] = body
        ctl = body.get("controls") if isinstance(body.get("controls"), Mapping) else {}
        facts[role] = {
            "seq": int(seq), "name": body.get("name"), "status": body.get("status"),
            "base": dict(body.get("base") or {}), "edit_id": (body.get("edit") or {}).get("id"),
            "controls": {"family": ctl.get("family"), "count": ctl.get("count"), "seed": ctl.get("seed")},
            "item_ids": sorted(str(((it.get("item") or {}).get("id"))) for it in body.get("items") or ()
                               if isinstance(it, Mapping)),
            "git": (body.get("runtime") or {}).get("git"), "git_clean": (body.get("runtime") or {}).get("git_clean"),
            "device": (body.get("runtime") or {}).get("device"),
        }
    try:
        drawn = {"target": _as_drawn(drawn_rows.get("target") or ()),
                 "reworded": _as_drawn(drawn_rows.get("reworded") or ())}
    except (KeyError, TypeError, ValueError):
        raise TrialRefused("The trial's verdict note does not record its drawn items as a trial writes them.") from None
    knobs = {k: v for k, v in dict(note.get("cheaper_knobs") or {}).items() if k != "matched_by"}
    judged = judge_readings(
        brief, bars, trial_cfg, readings, drawn, cheaper_knobs=knobs,
        practice_count=int(note.get("practice_count") or 0),
        practice_best=_clean_float(note.get("practice_best")),
        base_checksum_unchanged=bool(note.get("base_checksum_unchanged")),
        base_after=bool(note.get("base_after", True)),
        unpacker_fallbacks=int(note.get("unpacker_fallbacks") or 0),
    )
    tj = judged["trial"]

    def by_item(r: Optional[Mapping[str, Any]], dr: Sequence[Any]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        if r is None or r.get("status") != "ok":
            return out
        for cls_name in ("small", "large"):
            effs, _, _, _ = lcl._extract_item_effects(r, dr, cls_name, brief.metric, float(trial_cfg["main_dose"]))
            out[cls_name] = {str(e["item_id"]): float(e["value"]) for e in effs}
        return out

    return {
        "verdict": tj["verdict"],
        "fresh_verdict": tj["fresh_verdict"],
        "noted_verdict": note.get("verdict"),
        "as_noted": tj["verdict"] == note.get("verdict") and tj["fresh_verdict"] == note.get("fresh_verdict"),
        "note_type": note.get("note_type"),
        "registration_seq": found["registration_seq"],
        "verdict_seq": found["verdict_seq"],
        "items": {"target": by_item(readings.get("matched"), drawn["target"]),
                  "reworded": by_item(readings.get("reworded"), drawn["reworded"])},
        "readings": facts,
        "judged": judged,
    }
