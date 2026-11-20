"""Levelling Phase 5: a volunteer's re-run of a trial, and its intake (`backend/level_rerun.py`).

A re-run asks one thing: does another card, running the same code on the same checkpoint, read the same items the
same way? So it repeats the home trial's own items (`verify/arbiter.md` §8; the owner's ruling of 19 November 2026):

- `make_rerun_kit` packs a registration that has been read at home into a kit: the registration's own files, the
  home folder's `SHA256SUMS` under another name (so the kit can be shown to be that registration's), and `kit.json`
  with the seed of the home draw. An edit that carries arrays has no kit (a kit holds recipes, never weights). The
  kit carries the brief and the reworded set in clear: a reworded set that has gone out in a kit is no longer the
  owner's alone.
- `run_volunteer_rerun` checks the kit against itself, reads the trial on the volunteer's model into a book of its
  own with the home seed (`level_trial.run_trial_of_record(rerun=...)`: never of record), and packs four files into
  a plain tar: `manifest.json`, `return.json`, `volunteer_book.jsonl`, `TRIAL_SHEET.md`.
- `intake_rerun_bundle` unpacks a returned tar without trusting it (a plain tar of exactly those four plain files,
  each under its cap, nothing run), then judges it FROM THE RETURNED BOOK and not from what the bundle says of itself:
  the book must be whole, with no line longer than a reading's, and hold a re-run of this registration; its readings
  must be the home trial's own (the registered edit on the registered base, the same plan to the last setting, the
  same items, the same record of which item is which frame and half); the home's code, judging those readings again
  (`level_trial.rejudge_trial`), must give the verdict the book's note and the bundle state; and figure by figure
  (`level_trial.trial_figures`: every item's effect per-row and folded, every control edit's mean, the floors, the
  reversed dose, the batteries) the volunteer's are set beside the home's, both taken from the two books by the
  same code. Nothing in a returned bundle makes an intake raise: what cannot be judged is `rejected`, in words.

Outcomes: `rejected` (the bundle is not a whole re-run of this trial), `disagreed` (it is, and its verdict or its
figures differ from the home's by more than the tolerance), `accepted` (it is, and they do not), with the class
`same_figures` when every figure agrees to 1e-6 and `within_tolerance` otherwise.

What an intake cannot show, and says so in its verdict (`ON_THE_VOLUNTEERS_WORD`): that the returned book was
written by a run on the model it names, and that its figures are that run's and not the home book's own lines
copied into a new book. A book is a hash chain with no secret in it; a re-run is a volunteer's word, checked for
being whole, for being this trial, and against the home's own figures. So a home book whose trial is out for
re-run is kept unpublished until the returns are in: "the same figures" from someone who could read the home's
figures shows nothing.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from backend import level_book as lb
from backend import level_trial as lt
from backend import levelling as lv
from backend.census import write_json_atomic

KIT_VERSION = 1
PACKED_FILES = ("manifest.json", "return.json", "volunteer_book.jsonl", "TRIAL_SHEET.md")
MAX_SMALL_BYTES = 16 * 2**20  # 16 MiB for manifest / return / sheet
MAX_BOOK_BYTES = 256 * 2**20  # 256 MiB cap for volunteer_book.jsonl
MAX_LINE_BYTES = 32 * 2**20  # the longest line of a returned book that is parsed (a trial's longest is a few MiB)
SAME_FIGURES_TOL = 1e-6
# how far one item's effect on the volunteer's card may sit from the home's: as the folded form is set beside the
# per-row form (`level_trial.folded_tolerance`), one home floor or 5% of the home effect, and never under 0.02
ITEM_FLOOR_MULT, ITEM_REL_TOL, ITEM_MIN_TOL = 1.0, 0.05, 0.02
ON_THE_VOLUNTEERS_WORD = (
    "that the returned book was written by a run of this kit on the model it names (a book is a hash chain with no "
    "secret in it)",
    "that its figures are that run's own, and not the home book's lines copied into a new book: an intake cannot "
    "tell the two apart, so the home book stays unpublished until the returns are in",
    "the device, the commit and the state of the volunteer's checkout, the checksum of its base, and that its "
    "runner and reader were the home's",
)


class RerunRefused(ValueError):
    """Plain-word refusal of a kit that cannot be made, verified or run, or of an intake that cannot begin."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


MAX_WRITE_DEPTH = 64  # how deep a value from a bundle is followed when it is written into a verdict


def _writable(obj: Any, _depth: int = 0) -> Any:
    """A value as a JSON file takes it whatever a bundle put in it: a string that is not valid Unicode (half of a
    surrogate pair, which JSON can spell and UTF-8 cannot hold) is written with the odd part escaped; a number that
    is not finite (JSON has no such token, though Python's reader takes `NaN` and `1e999`) is written as a string
    that says so; and a value nested deeper than `MAX_WRITE_DEPTH` is cut there, in words (a list nested three
    thousand deep is six kilobytes of brackets, and this walk is Python's own)."""
    if _depth > MAX_WRITE_DEPTH:
        return "(nested too deep to write)"
    if isinstance(obj, str):
        return obj.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(obj, float) and not math.isfinite(obj):
        return f"{obj!r} (not a finite number)"
    if isinstance(obj, Mapping):
        return {_writable(k if isinstance(k, str) else f"({type(k).__name__})", _depth + 1): _writable(v, _depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_writable(v, _depth + 1) for v in obj]
    return obj


def _s(x: Any, n: int) -> str:
    """The first `n` characters of a string a bundle or a returned book gave; anything that is not a string is named
    by its kind, since the text of a value nested thousands deep cannot always be made."""
    return x[:n] if isinstance(x, str) else ("" if x is None else f"({type(x).__name__})")


def _seed_of(text: Any) -> Optional[bytes]:
    if not isinstance(text, str) or len(text) != 64:
        return None
    try:
        return bytes.fromhex(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------------- the kit


def make_rerun_kit(prereg_dir: Path, out_dir: Path, home_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Pack a registration that has been read at home into a volunteer's kit.

    `home_result` is the home trial's `trial_result.json`: it must be this registration's, must have ended (not
    void: a void trial read nothing to repeat), and holds the seed of the home draw, which the kit carries so that
    the volunteer reads the home items. Refused: an edit that carries arrays, and an `out_dir` that is not empty."""
    reg = lt.verify_registration(prereg_dir)
    edit_man = reg["edit_manifest"]
    if not edit_man.get("all_rebuilt"):
        raise RerunRefused(
            f'Edit {str(edit_man.get("edit_id") or "")[:12]} contains carried weight arrays ("provenance": "carried"). '
            "Volunteer re-run kits carry pure-JSON rebuildable recipes only (no weight payloads)."
        )
    if not isinstance(home_result, Mapping) or home_result.get("trial_sha256") != reg["trial_sha256"]:
        raise RerunRefused(
            "A kit is made from a registration and the result of its own home trial (trial_result.json): the result "
            "given is not this registration's."
        )
    if home_result.get("rerun_of"):
        raise RerunRefused("A kit is made from a home trial, not from a re-run.")
    if home_result.get("verdict") not in lt.PASSED + ("failed",):
        raise RerunRefused(f'The home trial ended "{home_result.get("verdict")}": a void trial read nothing to repeat.')
    if _seed_of(home_result.get("seed_hex")) is None:
        raise RerunRefused("The home result does not hold the seed of its draw, so its items cannot be asked for again.")

    dst = Path(out_dir).resolve()
    if dst.exists() and any(dst.iterdir()):
        raise RerunRefused(f"Kit output folder {dst} already exists and is not empty.")
    dst.mkdir(parents=True, exist_ok=True)

    src = Path(prereg_dir).resolve()
    for fname in lt.REG_FILES:
        shutil.copy2(src / fname, dst / fname)
    shutil.copy2(src / "SHA256SUMS", dst / "HOME_SHA256SUMS")

    trial = reg["trial"]
    kit_meta: Dict[str, Any] = {
        "v": KIT_VERSION,
        "packed_at": _utc_now(),
        "trial_name": trial["trial_name"],
        "trial_sha256": reg["trial_sha256"],
        "brief_id": trial["brief_id"],
        "brief_name": trial["brief_name"],
        "bars_sha256": trial["bars_sha256"],
        "edit_id": trial["edit_id"],
        "edit_provenance": "rebuilt",
        "base_model": (edit_man.get("base") or {}).get("model"),
        "base_weights_id": (edit_man.get("base") or {}).get("weights_id"),
        "reworded_id": trial.get("reworded_id"),
        "seed_hex": str(home_result["seed_hex"]),
        "home": {
            "of_record": bool(home_result.get("of_record")),
            "book_verdict_hash": (home_result.get("book") or {}).get("verdict_hash"),
        },
        "says": ("This kit asks for the home trial's own items (its draw's seed is here) and carries the brief and "
                 "the reworded set in clear: both are spent as things kept from a reader once a kit goes out."),
    }
    write_json_atomic(dst / "kit.json", kit_meta, indent=2)
    kit_sha, files = lt._write_sha256sums(dst)
    return {
        "kit_dir": str(dst),
        "kit_sha256": kit_sha,
        "trial_sha256": reg["trial_sha256"],
        "kit": kit_meta,
        "files": files,
    }


def verify_rerun_kit(kit_dir: Path) -> Dict[str, Any]:
    """Check a kit against itself: it is a registration folder with the kit's two files more
    (`level_trial.verify_registration(kit=True)`), its edit is all rebuilt, the registration's files in it are the
    ones the home folder's own sums name (`HOME_SHA256SUMS`, whose hash is the home registration's), `kit.json` agrees
    with `trial.json`, and it holds a seed."""
    folder = Path(kit_dir).resolve()
    if not folder.is_dir():
        raise RerunRefused(f"Re-run kit folder does not exist: {folder}")
    try:
        reg = lt.verify_registration(folder, kit=True)
        kit_meta = json.loads((folder / "kit.json").read_text(encoding="utf-8"))
    except (lt.TrialRefused, OSError, ValueError) as exc:
        raise RerunRefused(str(exc)) from None
    if not isinstance(kit_meta, dict):
        raise RerunRefused("kit.json is not a JSON object.")
    if not reg["edit_manifest"].get("all_rebuilt"):
        raise RerunRefused("Re-run kit edit_manifest is not all_rebuilt.")
    home_sums = (folder / "HOME_SHA256SUMS").read_bytes()
    if home_sums != lt._sums_bytes(folder, lt.REG_FILES):
        raise RerunRefused("The kit's registration files are not the ones its home registration's sums name.")
    home_sha = hashlib.sha256(home_sums).hexdigest()
    if kit_meta.get("trial_sha256") != home_sha:
        raise RerunRefused("kit.json does not name the home registration its files are.")
    for k in ("brief_id", "bars_sha256", "edit_id", "trial_name"):
        if kit_meta.get(k) != reg["trial"].get(k):
            raise RerunRefused(f"kit.json {k} does not match trial.json.")
    if _seed_of(kit_meta.get("seed_hex")) is None:
        raise RerunRefused("kit.json does not hold the seed of the home draw.")
    return {
        "ok": True,
        "kit_dir": str(folder),
        "kit_sha256": reg["trial_sha256"],  # the hash of the kit's own SHA256SUMS: what its run is registered as
        "trial_sha256": home_sha,
        "kit": kit_meta,
        "reg": reg,
    }


# --------------------------------------------------------------------------------------------- the volunteer's run


def run_volunteer_rerun(
    h: lv.Handle,
    kit_dir: Path,
    out_tar: Path,
    *,
    volunteer_name: str = "anonymous",
    work_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read a kit on this machine's model and pack what an intake takes: the book the run wrote, its sheet, a short
    account of the run in its own words (`return.json`), and their hashes (`manifest.json`), in a plain tar."""
    verified = verify_rerun_kit(kit_dir)
    tar_path = Path(out_tar).expanduser().resolve()
    if tar_path.exists():
        raise RerunRefused(f"Output bundle already exists: {tar_path} (refusing to overwrite).")
    # a name that cannot be written is found now, not after the trial has been read
    volunteer_name = _writable(str(volunteer_name))

    own_tmp = work_dir is None
    run_root = Path(tempfile.mkdtemp(prefix="levelling-rerun-")) if own_tmp else Path(work_dir).resolve()
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        vol_book = run_root / "volunteer_book.jsonl"
        out_sub = run_root / "trial_out"
        try:
            res = lt.run_trial_of_record(
                h, Path(kit_dir), out_dir=out_sub, book_path=vol_book,
                rerun={"of": verified["trial_sha256"], "seed": _seed_of(verified["kit"]["seed_hex"])},
            )
        except lt.TrialRefused as exc:
            raise RerunRefused(str(exc)) from None
        bv = lb.verify(vol_book)
        if not bv["ok"]:
            raise RerunRefused(f"Volunteer level book failed verification: {bv['why']}")

        meta = res["meta"]
        ret_payload: Dict[str, Any] = {
            "v": KIT_VERSION,
            "returned_at": _utc_now(),
            "volunteer": (str(volunteer_name).strip() or "anonymous")[:64],
            "kit_sha256": verified["kit_sha256"],
            "trial_sha256": verified["trial_sha256"],
            "trial_name": res["trial_name"],
            "verdict": res["verdict"],
            "fresh_verdict": res["fresh_verdict"],
            "says": "What this run says of itself. An intake judges the book, not this file.",
            "runtime": {
                "device": meta["device"],
                "model_id": meta["model_id"],
                "weights_id": meta["weights_id"],
                "unpacker": meta["unpacker"],
                "repo_commit": meta["repo_commit"],
                "checkout": meta["checkout"],
                "process": meta["process"],
                "seconds": meta["seconds"],
            },
            "book": {"entries": bv["entries"], "head": bv["head"]},
        }
        ret_path = run_root / "return.json"
        sheet_path = out_sub / "TRIAL_SHEET.md"
        write_json_atomic(ret_path, ret_payload, indent=2)

        files_to_pack = {
            "return.json": ret_path,
            "volunteer_book.jsonl": vol_book,
            "TRIAL_SHEET.md": sheet_path,
        }
        manifest = {
            "v": KIT_VERSION,
            "packed_at": _utc_now(),
            "volunteer": ret_payload["volunteer"],
            "kit_sha256": verified["kit_sha256"],
            "trial_sha256": verified["trial_sha256"],
            "files": {
                nm: {"sha256": _sha_file(p), "bytes": int(p.stat().st_size)}
                for nm, p in sorted(files_to_pack.items())
            },
        }
        man_path = run_root / "manifest.json"
        write_json_atomic(man_path, manifest, indent=2)

        tar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_tar = tar_path.with_name(tar_path.name + ".tmp")
        try:
            with tarfile.open(tmp_tar, "w") as tf:
                for nm in PACKED_FILES:
                    src_f = man_path if nm == "manifest.json" else files_to_pack[nm]
                    ti = tarfile.TarInfo(name=nm)
                    ti.size = src_f.stat().st_size
                    ti.mode = 0o644
                    with open(src_f, "rb") as fp:
                        tf.addfile(ti, fp)
            os.replace(tmp_tar, tar_path)
        finally:
            tmp_tar.unlink(missing_ok=True)

        return {
            "bundle_tar": str(tar_path),
            "bundle_sha256": _sha_file(tar_path),
            "kit_sha256": verified["kit_sha256"],
            "trial_sha256": verified["trial_sha256"],
            "verdict": res["verdict"],
            "return_data": ret_payload,
        }
    finally:
        if own_tmp:
            shutil.rmtree(run_root, ignore_errors=True)


# --------------------------------------------------------------------------------------------- the intake


def _extract_bundle_safely(bundle_tar: Path, dest_dir: Path) -> List[str]:
    """Unpack a returned bundle into `dest_dir` without trusting it (`scripts/census_intake.py`'s pattern): a plain
    tar and no compressed one (so nothing is inflated to learn its size), of exactly the four files a run packs, each
    a plain file under its cap, none a link, a folder or a device, none named twice; every member is checked before
    any is read."""
    p = Path(bundle_tar).expanduser().resolve()
    if not p.is_file():
        raise RerunRefused(f"Volunteer bundle not found: {p}")
    if p.stat().st_size > MAX_BOOK_BYTES + 3 * MAX_SMALL_BYTES + 2**20:
        raise RerunRefused(f"Volunteer bundle {p.name} is larger than anything a run packs.")
    dest = Path(dest_dir).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise RerunRefused(f"Intake staging folder {dest} already exists and is not empty.")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(p, "r:") as tf:
            members = tf.getmembers()
            if not members:
                raise RerunRefused("Volunteer bundle tar is empty.")
            if len(members) > len(PACKED_FILES):
                raise RerunRefused(f"Volunteer bundle holds {len(members)} members; a run packs {len(PACKED_FILES)}.")
            seen: set = set()
            for m in members:
                name = m.name
                if not m.isreg() or m.issym() or m.islnk():
                    raise RerunRefused(f"Bundle member {name!r} is not a plain regular file (symlinks/links/dirs refused).")
                if "/" in name or "\\" in name or name in ("..", ".") or name.startswith("."):
                    raise RerunRefused(f"Bundle member {name!r} uses a disallowed path.")
                if name not in PACKED_FILES:
                    raise RerunRefused(f"Bundle contains unexpected file {name!r}; allowed: {', '.join(PACKED_FILES)}.")
                if name in seen:
                    raise RerunRefused(f"Bundle contains duplicate member {name!r}.")
                cap = MAX_BOOK_BYTES if name == "volunteer_book.jsonl" else MAX_SMALL_BYTES
                if m.size < 0 or m.size > cap:
                    raise RerunRefused(f"Bundle member {name!r} size ({m.size} bytes) exceeds cap ({cap}).")
                seen.add(name)
            missing = sorted(set(PACKED_FILES) - seen)
            if missing:
                raise RerunRefused(f"Bundle is missing required file(s): {missing}.")
            for m in members:
                cap = MAX_BOOK_BYTES if m.name == "volunteer_book.jsonl" else MAX_SMALL_BYTES
                src = tf.extractfile(m)
                if src is None:
                    raise RerunRefused(f"Cannot read bundle member {m.name!r}.")
                data = src.read(cap + 1)
                if len(data) > cap:
                    raise RerunRefused(f"Bundle member {m.name!r} exceeded size cap while reading.")
                (dest / m.name).write_bytes(data)
    except RerunRefused:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise RerunRefused(f"Cannot read volunteer bundle {p} as a plain tar: {exc}") from None
    return sorted(PACKED_FILES)


def _find_rerun(book: Path, home_sha: str) -> Optional[str]:
    """The registration hash a re-run of `home_sha` was entered under in a returned book (the kit's own), or None."""
    for ent in lb.entries(book):
        body = ent.get("body")
        if ent.get("kind") == "note" and isinstance(body, Mapping) and body.get("note_type") == "trial_registration" \
                and body.get("rerun_of") == home_sha and isinstance(body.get("trial_sha256"), str):
            return str(body["trial_sha256"])
    return None


def _longest_line(path: Path) -> int:
    """The length in bytes of a file's longest line, read a block at a time: asked of a returned book before any of
    its lines is parsed (a single line of hundreds of megabytes would be parsed whole)."""
    longest = run = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            parts = block.split(b"\n")
            run += len(parts[0])
            for part in parts[1:]:
                longest, run = max(longest, run), len(part)
    return max(longest, run)


def _figure_tolerance(key: str, home_value: Optional[float], floors: Mapping[str, float], *, floor_mult: float,
                      rel: float, least: float) -> float:
    """How far one figure on the volunteer's card may sit from the home's: the largest of `floor_mult` home floors
    (the floor of the figure's own reading and class: a reworded figure by the reworded reading's floor, a battery's
    by none), `rel` of the home figure, and `least`. What an arm changed the stream by (`...:loud:...`) is set
    within `rel` of the home figure alone, and a count of separate draws (`...:draws:...`) must be the home's."""
    group, cls_name, kind = (key.split(":") + ["", ""])[:3]
    if kind == "loud":  # a change to the stream, a number near a hundredth: by its own size, not by a floor in nats
        return max(float(rel) * abs(float(home_value or 0.0)), 1e-9)
    if kind == "draws":  # a count
        return 0.0
    floor = float(floors.get(f"{group}:{cls_name}") or 0.0)
    return max(float(floor_mult) * floor, float(rel) * abs(float(home_value or 0.0)), float(least))


def _bucket(key: str) -> str:
    """Which row of an intake's comparison a figure is counted in: `target:small` and `reworded:large` are the items'
    own effects (the per-row form), `...:folded` the folded form's, `...:controls` the control edits' means,
    `...:delivered` what the edit and each matched or sham control edit changed the stream by, `...:other` the
    floor, the reversed dose and the count of separate draws, `batteries` the batteries' figures, `prompt_arms` what
    the note records of the brief's prompt arms."""
    parts = key.split(":")
    if parts[0] == "battery":
        return "batteries"
    if parts[0] == "knob":
        return "prompt_arms"
    kind = parts[2]
    head = f"{parts[0]}:{parts[1]}"
    if kind == "item":
        return head
    return f"{head}:{ {'folded': 'folded', 'control': 'controls', 'loud': 'delivered'}.get(kind, 'other') }"


def intake_rerun_bundle(
    bundle_tar: Path,
    prereg_dir: Path,
    home_book: Path,
    staging_dir: Path,
    *,
    allow_dirty: bool = False,
    allow_other_runner: bool = False,
    item_floor_mult: float = ITEM_FLOOR_MULT,
    item_rel_tol: float = ITEM_REL_TOL,
    item_min_tol: float = ITEM_MIN_TOL,
) -> Dict[str, Any]:
    """`_intake`, with the one thing a bundle could still raise turned into a verdict: a value nested deeper than
    Python will walk (`RecursionError`) rejects the bundle in words, and `intake_verdict.json` is written all the
    same. See `_intake` for what an intake does and what it raises before it begins."""
    try:
        return _intake(bundle_tar, prereg_dir, home_book, staging_dir, allow_dirty=allow_dirty,
                       allow_other_runner=allow_other_runner, item_floor_mult=item_floor_mult,
                       item_rel_tol=item_rel_tol, item_min_tol=item_min_tol)
    except RecursionError:
        stage = Path(staging_dir).resolve()
        why = "bundle_can_be_read: a value in the bundle or in its book is nested deeper than can be read"
        verdict_doc = {
            "v": KIT_VERSION, "judged_at": _utc_now(), "outcome": "rejected", "match_class": None,
            "with_leave_for": [], "bundle_tar": str(Path(bundle_tar).name), "bundle_sha256": _sha_file(Path(bundle_tar)),
            "judged_from": "nothing: the bundle could not be read", "on_the_volunteers_word": list(ON_THE_VOLUNTEERS_WORD),
            "reject_reasons": [why], "disagree_reasons": [],
            "checks": [{"name": "bundle_can_be_read", "ok": False, "detail": why}],
        }
        stage.mkdir(parents=True, exist_ok=True)
        verdict_doc = _writable(verdict_doc)  # the bundle's own name is in it
        write_json_atomic(stage / "intake_verdict.json", verdict_doc, indent=2)
        return verdict_doc


def _intake(
    bundle_tar: Path,
    prereg_dir: Path,
    home_book: Path,
    staging_dir: Path,
    *,
    allow_dirty: bool = False,
    allow_other_runner: bool = False,
    item_floor_mult: float = ITEM_FLOOR_MULT,
    item_rel_tol: float = ITEM_REL_TOL,
    item_min_tol: float = ITEM_MIN_TOL,
) -> Dict[str, Any]:
    """Unpack a returned bundle and judge it against the home trial, from the two books.

    `prereg_dir` is the home registration and `home_book` the book its trial was read in: the home trial is judged
    again from that book (`level_trial.rejudge_trial`), and so is the volunteer's from the returned book, by the same
    code. Writes `intake_verdict.json` into `staging_dir` and returns it: `outcome` is "rejected", "disagreed" or
    "accepted" (see the module's top), with every check it made and what it took on the volunteer's word.

    A re-run says of its own checkout whether a tracked file was changed, whether code that no commit holds stood
    where its runner imports from, and whether its runner and reader were the files hashed at registration; a bundle
    that says a tracked file was changed, or that such code was there, or that cannot say, is rejected unless
    `allow_dirty`, and one whose runner or reader was not the registered one unless `allow_other_runner` (two
    separate leaves, each named in the verdict when it was given). Other files git does not know in the volunteer's
    checkout do not count against it: the kit and the bundle are such files.

    The figures set beside the home's are the control edits' means as read, what the edit and each matched or sham
    control edit changed the stream by (over the prompt, over the answer, at the last position), and how many
    separate draws each family's control sets were counted as. Which control edits each trial then set aside (for
    changing the stream less than the edit did) is said apart (`control_edits_set_aside_differently`): it turns on
    a ratio that can sit at 1, so it can differ between two cards whose figures agree, and one control edit set
    aside is one more counted as moving as much. Where it differs and the verdicts agree the outcome is still
    `accepted`, as `within_tolerance` and never `same_figures`. A pair that neither trial has a figure for is
    counted apart too (`figures_neither_trial_has`), not as compared.

    Raises `RerunRefused` when the intake cannot begin (no such bundle, an unsafe tar, a home book that does not hold
    the home trial as its note says). Nothing in the bundle itself makes it raise."""
    home_whole = lb.verify(Path(home_book))
    if not home_whole["ok"]:
        raise RerunRefused(f"The home book {Path(home_book).name} does not verify (line {home_whole['first_bad']}: "
                           f"{home_whole['why']}): a re-run is not judged against it.")
    try:
        reg = lt.verify_registration(prereg_dir)
        home_sha = reg["trial_sha256"]
        home = lt.rejudge_trial(Path(home_book), home_sha, reg)
    except lt.TrialRefused as exc:
        raise RerunRefused(f"The home trial cannot be read from {Path(home_book).name}: {exc}") from None
    if home["note_type"] not in (lt.NOTE_OF_RECORD, lt.NOTE_DRY_RUN) or not home["as_noted"]:
        raise RerunRefused(
            f"The home book's trial {home_sha[:12]} is not a home trial whose note follows from its readings "
            f"(note {home['note_type']}, verdict noted {home['noted_verdict']}, judged again {home['verdict']})."
        )
    stage = Path(staging_dir).resolve()
    unpacked = _extract_bundle_safely(bundle_tar, stage)

    checks: List[Dict[str, Any]] = []
    reject_reasons: List[str] = []
    disagree_reasons: List[str] = []

    def _chk(name: str, ok: bool, detail: Any = None, *, kind: str = "reject") -> bool:
        detail = _writable(detail)  # it may quote the bundle: cut to a depth that can be written and said
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            msg = f"{name}: {detail}" if detail is not None else name
            (reject_reasons if kind == "reject" else disagree_reasons).append(msg)
        return bool(ok)

    def _json(name: str) -> Dict[str, Any]:
        try:
            got = json.loads((stage / name).read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            got = None
        if not _chk(f"{name}_is_an_object", isinstance(got, dict), "not a JSON object"):
            return {}
        return got

    manifest, ret = _json("manifest.json"), _json("return.json")

    # 1. the bundle's files are the ones its manifest names
    m_files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    for fname in ("return.json", "volunteer_book.jsonl", "TRIAL_SHEET.md"):
        row = m_files.get(fname) if isinstance(m_files.get(fname), Mapping) else {}
        want, got = row.get("sha256"), _sha_file(stage / fname)
        _chk(f"sha256_{fname}", want == got, {"got": got[:16], "want": _s(want, 16)})

    # 2. the returned book is whole, with no line longer than a reading's, and holds a re-run of this registration
    vol_book = stage / "volunteer_book.jsonl"
    longest = _longest_line(vol_book)
    lines_ok = _chk("returned_book_lines_under_the_cap", longest <= MAX_LINE_BYTES,
                    {"longest_line_bytes": longest, "cap": MAX_LINE_BYTES})
    bv = lb.verify(vol_book) if lines_ok else {"ok": False, "entries": 0, "why": "not parsed: a line is over the cap"}
    _chk("returned_book_whole", bv["ok"], {"entries": bv["entries"], "why": bv["why"]})
    vol: Optional[Dict[str, Any]] = None
    if bv["ok"]:
        try:
            kit_sha = _find_rerun(vol_book, home_sha)
            if _chk("book_holds_a_rerun_of_this_registration", kit_sha is not None,
                    f"no trial registered in the returned book as a re-run of {home_sha[:12]}"):
                vol = lt.rejudge_trial(vol_book, str(kit_sha), reg)
        except lt.TrialRefused as exc:
            _chk("rerun_can_be_judged_from_its_book", False, str(exc))
        except Exception as exc:  # a book typed by hand must end an intake in words, whatever it holds
            _chk("rerun_can_be_judged_from_its_book", False, f"{type(exc).__name__}: {str(exc)[:200]}")
            vol = None

    class_diffs: Dict[str, Any] = {}
    set_aside_differently: Dict[str, int] = {}
    all_same = False
    allowances: List[str] = []
    if vol is not None:
        vol_reg = vol["registration"] if isinstance(vol.get("registration"), Mapping) else {}
        home_reg = home["registration"] if isinstance(home.get("registration"), Mapping) else {}
        # 3. it is a re-run, its note follows from its readings, and the bundle states what the book gives
        _chk("note_is_a_rerun", vol["note_type"] == lt.NOTE_RERUN, {"note_type": vol["note_type"]})
        _chk("book_note_follows_from_its_readings", vol["as_noted"],
             {"noted": vol["noted_verdict"], "judged_again": vol["verdict"]})
        _chk("bundle_states_what_its_book_gives", ret.get("verdict") == vol["verdict"],
             {"return.json": ret.get("verdict"), "book": vol["verdict"]})
        _chk("rerun_ended", vol["verdict"] in lt.PASSED + ("failed",),
             {"verdict": vol["verdict"], "why": (vol["judged"]["trial"].get("reasons") or [None])[0]})

        # 4. its readings are the home trial's: the same roles, the registered edit and base, the home plan to its
        #    last setting (control seeds and counts, rows, doses, read-outs), the home items, and the home's record
        #    of which item is which frame and half
        base_reg = reg["edit_manifest"].get("base") or {}
        for role in ("matched", "same_site", "sham", "reworded", "neutral", "nearmiss", "panel"):
            hf, vf = home["readings"].get(role), vol["readings"].get(role)
            if hf is None and vf is None:
                continue
            if not _chk(f"reading_{role}_present_in_both", hf is not None and vf is not None,
                        {"home": hf is not None, "volunteer": vf is not None}):
                continue
            wid_h, wid_v = (hf["base"] or {}).get("weights_id"), (vf["base"] or {}).get("weights_id")
            same = {"edit": vf["edit_id"] == reg["trial"]["edit_id"],
                    "model": (vf["base"] or {}).get("model") == base_reg.get("model"),
                    "weights": wid_h is None or wid_v == wid_h,
                    "plan": vf["plan"] == hf["plan"] and isinstance(vf["plan"], Mapping),
                    "controls": vf["controls"] == hf["controls"], "items": vf["item_ids"] == hf["item_ids"],
                    "name": vf["name"] == hf["name"]}
            _chk(f"reading_{role}_is_the_home_reading", all(same.values()), same)
            # Which control edits a card draws from a seed rests on a base pre-pass on that card: another card's
            # rounding can carry a site across the matching threshold and draw other edits. Their means are then
            # not the home's one for one, and the intake says that this is why.
            h_ids, v_ids = hf.get("control_ids"), vf.get("control_ids")
            if h_ids is not None and h_ids != v_ids:
                other = (sum(1 for a, b in zip(h_ids, v_ids) if a != b) + abs(len(h_ids) - len(v_ids))
                         if isinstance(v_ids, list) else len(h_ids))
                _chk(f"control_edits_of_{role}_drawn_as_at_home", False,
                     {"differ": other, "of": len(h_ids),
                      "says": "the same seed drew other control edits on this card: their figures are not the home's one for one"},
                     kind="disagree")
        _chk("drawn_items_recorded_as_at_home", vol["drawn"] == home["drawn"],
             {k: vol["drawn"].get(k) == home["drawn"].get(k) for k in ("target", "reworded")})
        _chk("base_checksum_is_the_home_base",
             isinstance(vol_reg.get("base_checksum_before"), str)
             and vol_reg.get("base_checksum_before") == home_reg.get("base_checksum_before")
             and vol_reg.get("base_tensors_checked") == home_reg.get("base_tensors_checked"),
             {"volunteer": _s(vol_reg.get("base_checksum_before"), 16), "home": _s(home_reg.get("base_checksum_before"), 16),
              "tensors": [vol_reg.get("base_tensors_checked"), home_reg.get("base_tensors_checked")]})

        # 5. on the volunteer's word: that no tracked file of its checkout was changed, and that its runner and
        #    reader were the files hashed at registration. Each has its own leave, named in the verdict when given.
        co = vol_reg.get("checkout") if isinstance(vol_reg.get("checkout"), Mapping) else {}
        runner_ok = co.get("sources_match") is True
        # each must be a count of nought: one that could not be asked (None) is not one that was found clean
        def nought(x: Any) -> bool:
            return type(x) is int and x == 0  # False and 0.0 are no count: a book typed by hand can hold either

        tracked_ok = nought(co.get("tracked_changed")) and nought(co.get("hidden"))
        code_ok = nought(co.get("unlisted_code"))
        if not runner_ok and allow_other_runner:
            allowances.append("its runner or reader was not the registered one (allow_other_runner)")
        if not tracked_ok and allow_dirty:
            allowances.append("a tracked file of its checkout was changed, or could not be shown unchanged (allow_dirty)")
        if not code_ok and allow_dirty:
            allowances.append("its checkout had code where its runner imports from that no commit holds, or could not "
                              "be shown to have none (allow_dirty)")
        _chk("runner_and_reader_as_registered", runner_ok or allow_other_runner,
             {"sources_match": co.get("sources_match"), "moved": co.get("sources_moved"),
              "allow_other_runner": allow_other_runner})
        _chk("no_tracked_file_changed", tracked_ok or allow_dirty,
             {"tracked_changed": co.get("tracked_changed"), "overlooked_by_git": co.get("hidden"),
              "allow_dirty": allow_dirty})
        _chk("no_code_outside_the_commit", code_ok or allow_dirty,
             {"code_no_commit_holds": co.get("unlisted_code"), "allow_dirty": allow_dirty,
              "says": "source, compiled Python or an extension module, where the runner imports from, that git does "
                      "not track (None: it could not be asked)"})

        # 6. agreement with the home trial, from the two books: the verdicts, then every figure the judgement used.
        #    Which control edits a trial set aside (they did not change the stream within the matching tolerance of
        #    what the edit changed it) is no figure: it turns on a ratio that can sit at its bound, and a control
        #    edit there falls one side on one card and the other on another. Where the two trials set aside other
        #    control edits it is said, and it is said beside the verdicts, which are what it can change.
        hj = home["judged"]
        for group, key_c in (("target", "classes"), ("reworded", "reworded_classes")):
            for cls_name in ("small", "large"):
                ha = ((hj["inputs"].get(key_c) or {}).get(cls_name) or {}).get("controls_aside") or {}
                va = ((vol["judged"]["inputs"].get(key_c) or {}).get(cls_name) or {}).get("controls_aside") or {}
                for fam in sorted(set(ha) | set(va)):
                    a, b = list(ha.get(fam) or ()), list(va.get(fam) or ())
                    n = sum(1 for x, y in zip(a, b) if bool(x) != bool(y)) + abs(len(a) - len(b))
                    if n:
                        set_aside_differently[f"{group}:{cls_name}:{fam}"] = n
        aside_note = ({"control_edits_set_aside_on_one_card_and_not_the_other": dict(set_aside_differently)}
                      if set_aside_differently else {})
        _chk("verdict_matches_home", vol["verdict"] == home["verdict"],
             {"volunteer": vol["verdict"], "home": home["verdict"], **aside_note}, kind="disagree")
        _chk("fresh_verdict_matches_home", vol["fresh_verdict"] == home["fresh_verdict"],
             {"volunteer": vol["fresh_verdict"], "home": home["fresh_verdict"], **aside_note}, kind="disagree")
        floors: Dict[str, float] = {}
        for cls_name in ("small", "large"):
            floors[f"target:{cls_name}"] = _clean_float(
                (((hj["fresh_judged"].get("classes") or {}).get(cls_name) or {}).get("house_bar") or {}).get("floor")) or 0.0
            floors[f"reworded:{cls_name}"] = _clean_float(
                ((((hj.get("reworded") or {}).get("classes") or {}).get(cls_name) or {}).get("house_bar") or {}).get("floor")) or 0.0
        hfig, vfig = home["figures"], vol["figures"]
        rows: Dict[str, Dict[str, Any]] = {}
        for must in [f"{g}:{c}" for g in ("target", "reworded") for c in ("small", "large")]:
            rows[must] = {"items": 0, "no_figure": 0, "missing": 0, "largest_difference": 0.0, "outside_tolerance": 0,
                          "home_floor": floors[must]}
        all_same = not set_aside_differently
        for key, h_val in hfig.items():
            b = _bucket(key)
            row = rows.setdefault(b, {"items": 0, "no_figure": 0, "missing": 0, "largest_difference": 0.0,
                                      "outside_tolerance": 0})
            v_val = vfig.get(key) if key in vfig else None
            if key not in vfig or (h_val is None) != (v_val is None):
                row["missing"] += 1
                all_same = False
                continue
            if h_val is None:
                row["no_figure"] += 1  # neither trial has this figure (a control edit that did not read): not compared
                continue
            row["items"] += 1
            diff = abs(float(v_val) - float(h_val))
            row["largest_difference"] = max(row["largest_difference"], diff)
            if diff > _figure_tolerance(key, h_val, floors, floor_mult=item_floor_mult, rel=item_rel_tol,
                                        least=item_min_tol):
                row["outside_tolerance"] += 1
            if diff > SAME_FIGURES_TOL:
                all_same = False
        extra = sorted(set(vfig) - set(hfig))
        if extra:
            all_same = False
        for b, row in sorted(rows.items()):
            class_diffs[b] = row
            needs_items = b.count(":") == 1  # the items' own rows: a class with no item compared does not agree
            _chk(f"items_agree_{b.replace(':', '_')}" if needs_items else f"figures_agree_{b.replace(':', '_')}",
                 (row["items"] > 0 or not needs_items) and row["missing"] == 0 and row["outside_tolerance"] == 0,
                 row, kind="disagree")
        _chk("no_figure_the_home_trial_lacks", not extra, {"count": len(extra), "first": extra[:3]}, kind="disagree")

    if reject_reasons:
        outcome, match_class = "rejected", None
    elif disagree_reasons:
        outcome, match_class = "disagreed", None
    else:
        outcome, match_class = "accepted", ("same_figures" if all_same else "within_tolerance")

    verdict_doc = {
        "v": KIT_VERSION,
        "judged_at": _utc_now(),
        "outcome": outcome,
        "match_class": match_class,
        "with_leave_for": allowances if outcome == "accepted" else [],
        "volunteer": _s(ret.get("volunteer"), 64),
        "bundle_tar": str(Path(bundle_tar).name),
        "bundle_sha256": _sha_file(Path(bundle_tar)),
        "trial_name": reg["trial"]["trial_name"],
        "trial_sha256": home_sha,
        "home_verdict": home["verdict"],
        "home_of_record": home["note_type"] == lt.NOTE_OF_RECORD,
        "volunteer_verdict": vol["verdict"] if vol is not None else None,
        "class_diffs": class_diffs,
        "figures_compared": sum(int(r["items"]) for r in class_diffs.values()),
        "figures_neither_trial_has": sum(int(r.get("no_figure") or 0) for r in class_diffs.values()),
        "control_edits_set_aside_differently": set_aside_differently,
        "tolerance": {"item_floor_mult": float(item_floor_mult), "item_rel_tol": float(item_rel_tol),
                      "item_min_tol": float(item_min_tol), "same_figures": SAME_FIGURES_TOL},
        "judged_from": "the returned book and the home book, both judged again by the home's code",
        "on_the_volunteers_word": list(ON_THE_VOLUNTEERS_WORD),
        "unpacked_files": unpacked,
        "reject_reasons": reject_reasons,
        "disagree_reasons": disagree_reasons,
        "checks": checks,
    }
    verdict_doc = _writable(verdict_doc)  # it quotes what the bundle said of itself
    write_json_atomic(stage / "intake_verdict.json", verdict_doc, indent=2)
    return verdict_doc
