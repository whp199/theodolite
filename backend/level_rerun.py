"""Levelling Phase 5: Volunteer Re-run Kit & Defensive Intake (`backend/level_rerun.py`).

Adapts Theodolite's `scripts/census_intake.py` pattern to Levelling Trials of Record (`verify/arbiter.md` §8,
`design/synthesis.md` §ix, `design/future.md` §5):
- `make_rerun_kit`: packages a verified Phase 5 `prereg/` registration into a self-contained, hash-checked
  re-run kit (`kit.json` + brief/bars/manifest/rules/reworded + `SHA256SUMS`). Strictly refuses edits with
  `"carried"` weight payloads (`skeptic_safety.md` finding 6: volunteer kits carry rebuildable recipes only).
- `run_volunteer_rerun`: verifies the kit's `SHA256SUMS`, runs the Trial of Record in an isolated book, records
  hardware/commit/cleanliness and live `base_checksum`, and packs an allowlisted `.tar` bundle (`manifest.json`,
  `return.json`, `volunteer_book.jsonl`, `TRIAL_SHEET.md`).
- `intake_rerun_bundle`: defensively unpacks a volunteer `.tar` into a fresh staging directory (rejecting symlinks,
  hardlinks, paths with `/` or `..`, unexpected filenames, or oversized members; executing nothing), verifies every
  file's sha256, verifies the volunteer's level book chain whole, checks `base_checksum_unchanged` and zero unpacker
  fallbacks, compares the volunteer's verdict and figures against the home trial, and writes `intake_verdict.json`
  (`"accepted"`, `"disagreed"`, or `"rejected"`).

No kept-out words appear in code, bundles, or verdicts. Layers count from 0.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend import level_book as lb
from backend import level_trial as lt
from backend import levelling as lv
from backend.census import write_json_atomic

KIT_VERSION = 1
PACKED_FILES = ("manifest.json", "return.json", "volunteer_book.jsonl", "TRIAL_SHEET.md")
MAX_SMALL_BYTES = 16 * 2**20  # 16 MiB for manifest / return / sheet
MAX_BOOK_BYTES = 256 * 2**20  # 256 MiB cap for volunteer_book.jsonl


class RerunRefused(ValueError):
    """Plain-word refusal of an invalid kit, carried payload, unsafe bundle, or failed intake check."""


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


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------------------------- kit packaging


def make_rerun_kit(
    prereg_dir: Path,
    out_dir: Path,
    *,
    home_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Package a verified Phase 5 `prereg/` folder into a volunteer re-run kit.
    Refuses edits with any `"carried"` weight payload (rebuildable recipes only) and refuses non-empty `out_dir`."""
    reg = lt.verify_registration(prereg_dir)
    edit_man = reg["edit_manifest"]
    if not edit_man.get("all_rebuilt"):
        raise RerunRefused(
            f'Edit {str(edit_man.get("edit_id") or "")[:12]} contains carried weight arrays ("provenance": "carried"). '
            "Volunteer re-run kits carry pure-JSON rebuildable recipes only (no weight payloads)."
        )

    dst = Path(out_dir).resolve()
    if dst.exists() and any(dst.iterdir()):
        raise RerunRefused(f"Kit output folder {dst} already exists and is not empty.")
    dst.mkdir(parents=True, exist_ok=True)

    src = Path(prereg_dir).resolve()
    for fname in reg["files"]:
        shutil.copy2(src / fname, dst / fname)

    kit_meta: Dict[str, Any] = {
        "v": KIT_VERSION,
        "packed_at": _utc_now(),
        "trial_name": reg["trial"]["trial_name"],
        "trial_sha256": reg["trial_sha256"],
        "brief_id": reg["trial"]["brief_id"],
        "brief_name": reg["trial"]["brief_name"],
        "bars_sha256": reg["trial"]["bars_sha256"],
        "edit_id": reg["trial"]["edit_id"],
        "edit_provenance": "rebuilt",
        "base_model": (edit_man.get("base") or {}).get("model"),
        "base_weights_id": (edit_man.get("base") or {}).get("weights_id"),
        "reworded_id": reg["trial"].get("reworded_id"),
        "home_reference": (
            {
                "verdict": home_result.get("verdict"),
                "fresh_verdict": home_result.get("fresh_verdict"),
                "book_verdict_hash": (home_result.get("book") or {}).get("verdict_hash"),
            }
            if isinstance(home_result, Mapping) else None
        ),
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
    """Verify every file, hash, and cross-reference in a volunteer re-run kit."""
    folder = Path(kit_dir).resolve()
    if not folder.is_dir():
        raise RerunRefused(f"Re-run kit folder does not exist: {folder}")
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in lt.FORBIDDEN_PREREG_SUFFIXES:
            raise RerunRefused(f"Re-run kit contains forbidden weight file {p.name}.")
    if not (folder / "kit.json").is_file():
        raise RerunRefused(f"Re-run kit {folder} lacks kit.json.")
    try:
        reg = lt.verify_registration(folder)
    except lt.TrialRefused as exc:
        raise RerunRefused(str(exc)) from None
    kit_meta = json.loads((folder / "kit.json").read_text(encoding="utf-8"))
    if not reg["edit_manifest"].get("all_rebuilt"):
        raise RerunRefused("Re-run kit edit_manifest is not all_rebuilt.")
    for k in ("brief_id", "bars_sha256", "edit_id", "trial_name"):
        if kit_meta.get(k) != reg["trial"].get(k):
            raise RerunRefused(f"kit.json {k} does not match trial.json.")
    kit_sha = _sha_file(folder / "SHA256SUMS")
    # Note: kit's SHA256SUMS includes kit.json; trial_sha256 in kit.json is the original prereg's SHA256SUMS digest
    return {
        "ok": True,
        "kit_dir": str(folder),
        "kit_sha256": kit_sha,
        "trial_sha256": str(kit_meta["trial_sha256"]),
        "kit": kit_meta,
        "reg": reg,
    }


# --------------------------------------------------------------------------------------------- volunteer execution


def run_volunteer_rerun(
    h: lv.Handle,
    kit_dir: Path,
    out_tar: Path,
    *,
    volunteer_name: str = "anonymous",
    work_dir: Optional[Path] = None,
    _seed_override: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Run a verified re-run kit on the volunteer's machine and pack a defensive allowlisted `.tar` return bundle."""
    verified = verify_rerun_kit(kit_dir)
    tar_path = Path(out_tar).expanduser().resolve()
    if tar_path.exists():
        raise RerunRefused(f"Output bundle already exists: {tar_path} (refusing to overwrite).")

    own_tmp = work_dir is None
    run_root = Path(tempfile.mkdtemp(prefix="levelling-rerun-")) if own_tmp else Path(work_dir).resolve()
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        vol_book = run_root / "volunteer_book.jsonl"
        out_sub = run_root / "trial_out"

        res = lt.run_trial_of_record(
            h,
            Path(kit_dir),
            out_dir=out_sub,
            book_path=vol_book,
            require_book_calibration=False,
            _seed_override=_seed_override,
        )
        bv = lb.verify(vol_book)
        if not bv["ok"]:
            raise RerunRefused(f"Volunteer level book failed verification: {bv['why']}")

        summary_cls = {
            cls: {
                "mean": (cdata.get("frames") or {}).get("mean"),
                "directed_mean": (cdata.get("frames") or {}).get("directed_mean"),
                "ci95": (cdata.get("frames") or {}).get("ci95"),
                "half_0": (cdata.get("frames") or {}).get("half_0"),
                "half_1": (cdata.get("frames") or {}).get("half_1"),
                "house_bar": cdata.get("house_bar"),
                "controls": cdata.get("controls"),
            }
            for cls, cdata in ((res.get("fresh_judged") or {}).get("classes") or {}).items()
        }
        ret_payload: Dict[str, Any] = {
            "v": KIT_VERSION,
            "returned_at": _utc_now(),
            "volunteer": str(volunteer_name).strip() or "anonymous",
            "kit_sha256": verified["kit_sha256"],
            "trial_sha256": verified["trial_sha256"],
            "trial_name": res["trial_name"],
            "brief_id": res["trial_config"]["brief_id"],
            "bars_sha256": res["trial_config"]["bars_sha256"],
            "edit_id": res["trial_config"]["edit_id"],
            "verdict": res["verdict"],
            "fresh_verdict": res["fresh_verdict"],
            "reasons": res["reasons"],
            "summary": summary_cls,
            "folded": {
                "ok": (res.get("folded") or {}).get("ok"),
                "status": (res.get("folded") or {}).get("status"),
                "classes": {
                    cls: {
                        "ok": fc.get("ok"),
                        "mean": (fc.get("frames") or {}).get("mean"),
                        "max_abs_gap": fc.get("max_abs_gap"),
                        "same_bits_count": fc.get("same_bits_count"),
                    }
                    for cls, fc in ((res.get("folded") or {}).get("classes") or {}).items()
                },
            },
            "reworded": (
                {
                    "ok": res["reworded"].get("ok"),
                    "status": res["reworded"].get("status"),
                    "classes": {
                        cls: {"ok": rc.get("ok"), "mean": (rc.get("frames") or {}).get("mean")}
                        for cls, rc in (res["reworded"].get("classes") or {}).items()
                    },
                }
                if res.get("reworded") is not None else None
            ),
            "batteries": (res.get("fresh_judged") or {}).get("batteries"),
            "runtime": {
                "device": res["meta"]["device"],
                "model_id": res["meta"]["model_id"],
                "weights_id": res["meta"]["weights_id"],
                "unpacker": res["meta"]["unpacker"],
                "repo_commit": res["meta"]["repo_commit"],
                "git_dirty": res["meta"]["git_dirty"],
                "seconds": res["meta"]["seconds"],
            },
            "integrity": {
                "base_checksum_before": res["meta"]["base_checksum_before"],
                "base_checksum_after": res["meta"]["base_checksum_after"],
                "base_checksum_unchanged": res["meta"]["base_checksum_unchanged"],
                "base_after": res["meta"]["base_after"],
                "unpacker_fallbacks": res["meta"]["unpacker_fallbacks"],
                "book_entries": bv["entries"],
                "book_head": bv["head"],
            },
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


# --------------------------------------------------------------------------------------------- defensive intake & judge


def _extract_bundle_safely(bundle_tar: Path, dest_dir: Path) -> List[str]:
    """Defensively extract an allowlisted volunteer `.tar` bundle into `dest_dir`:
    rejects symlinks, hardlinks, directories, absolute/traversal names, unknown filenames, duplicates,
    and oversized members (`scripts/census_intake.py` pattern)."""
    p = Path(bundle_tar).expanduser().resolve()
    if not p.is_file():
        raise RerunRefused(f"Volunteer bundle not found: {p}")
    dest = Path(dest_dir).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise RerunRefused(f"Intake staging folder {dest} already exists and is not empty.")
    dest.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    try:
        with tarfile.open(p, "r:*") as tf:
            members = tf.getmembers()
            if not members:
                raise RerunRefused("Volunteer bundle tar is empty.")
            for m in members:
                if not m.isfile() or m.issym() or m.islnk():
                    raise RerunRefused(f"Bundle member {m.name!r} is not a plain regular file (symlinks/links/dirs refused).")
                name = m.name
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
                src = tf.extractfile(m)
                if src is None:
                    raise RerunRefused(f"Cannot read bundle member {name!r}.")
                data = src.read(cap + 1)
                if len(data) > cap:
                    raise RerunRefused(f"Bundle member {name!r} exceeded size cap while reading.")
                (dest / name).write_bytes(data)
    except RerunRefused:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise RerunRefused(f"Cannot read volunteer bundle {p}: {exc}") from None

    missing = sorted(set(PACKED_FILES) - seen)
    if missing:
        raise RerunRefused(f"Bundle is missing required file(s): {missing}.")
    return sorted(seen)


def intake_rerun_bundle(
    bundle_tar: Path,
    prereg_dir: Path,
    home_result: Mapping[str, Any],
    staging_dir: Path,
    *,
    allow_dirty: bool = False,
    same_bits_tol: float = 1e-6,
    class_floor_mult: float = 3.0,
    class_rel_tol: float = 0.05,
    class_min_tol: float = 0.15,
) -> Dict[str, Any]:
    """Unpack, verify, and judge a volunteer re-run `.tar` bundle against the home registration and home trial result.

    Returns `intake_verdict` with `"outcome"` in `("accepted", "disagreed", "rejected")` and writes
    `intake_verdict.json` into `staging_dir`.
    """
    reg = lt.verify_registration(prereg_dir)
    stage = Path(staging_dir).resolve()
    unpacked = _extract_bundle_safely(bundle_tar, stage)

    checks: List[Dict[str, Any]] = []
    reject_reasons: List[str] = []
    disagree_reasons: List[str] = []

    def _chk(name: str, ok: bool, detail: Any = None, *, kind: str = "reject") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            msg = f"{name}: {detail}" if detail is not None else name
            if kind == "reject":
                reject_reasons.append(msg)
            else:
                disagree_reasons.append(msg)

    try:
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
        ret = json.loads((stage / "return.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RerunRefused(f"Malformed JSON in unpacked volunteer bundle: {exc}") from None

    # 1. File hashes in manifest.json
    m_files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    for fname in ("return.json", "volunteer_book.jsonl", "TRIAL_SHEET.md"):
        want = (m_files.get(fname) or {}).get("sha256")
        got = _sha_file(stage / fname)
        _chk(f"sha256_{fname}", want == got, {"got": got[:16], "want": str(want or "")[:16]})

    # 2. Identity cross-check against home registration
    _chk(
        "registration_identity",
        ret.get("trial_sha256") == reg["trial_sha256"]
        and ret.get("edit_id") == reg["trial"]["edit_id"]
        and ret.get("brief_id") == reg["trial"]["brief_id"]
        and ret.get("bars_sha256") == reg["trial"]["bars_sha256"],
        {
            "trial_sha_match": ret.get("trial_sha256") == reg["trial_sha256"],
            "edit_match": ret.get("edit_id") == reg["trial"]["edit_id"],
            "brief_match": ret.get("brief_id") == reg["trial"]["brief_id"],
            "bars_match": ret.get("bars_sha256") == reg["trial"]["bars_sha256"],
        },
    )

    # 3. Volunteer level book chain verification
    bv = lb.verify(stage / "volunteer_book.jsonl")
    integ = ret.get("integrity") or {}
    _chk(
        "volunteer_book_whole",
        bv["ok"] and bv["head"] == integ.get("book_head") and bv["entries"] == integ.get("book_entries"),
        {"verify_ok": bv["ok"], "entries": bv["entries"], "head": str(bv["head"])[:16]},
    )

    # 4. Base integrity, unpacker fallbacks, cleanliness
    _chk(
        "base_integrity_and_unpacker",
        bool(integ.get("base_after"))
        and bool(integ.get("base_checksum_unchanged"))
        and int(integ.get("unpacker_fallbacks", -1)) == 0,
        {
            "base_after": integ.get("base_after"),
            "base_checksum_unchanged": integ.get("base_checksum_unchanged"),
            "unpacker_fallbacks": integ.get("unpacker_fallbacks"),
        },
    )
    rt_info = ret.get("runtime") or {}
    dirty_ok = True if allow_dirty else (rt_info.get("git_dirty") is False)
    _chk("checkout_clean", dirty_ok, {"git_dirty": rt_info.get("git_dirty"), "allow_dirty": allow_dirty})

    # 5. Agreement with home trial (only meaningful if integrity checks passed)
    home_verdict = str(home_result.get("verdict") or "")
    vol_verdict = str(ret.get("verdict") or "")
    _chk("verdict_matches_home", vol_verdict == home_verdict, {"volunteer": vol_verdict, "home": home_verdict}, kind="disagree")

    home_folded_ok = (home_result.get("folded") or {}).get("ok")
    vol_folded_ok = (ret.get("folded") or {}).get("ok")
    _chk("folded_matches_home", vol_folded_ok == home_folded_ok, {"volunteer": vol_folded_ok, "home": home_folded_ok}, kind="disagree")

    if home_result.get("reworded") is not None or ret.get("reworded") is not None:
        h_rw_ok = (home_result.get("reworded") or {}).get("ok")
        v_rw_ok = (ret.get("reworded") or {}).get("ok")
        _chk("reworded_matches_home", v_rw_ok == h_rw_ok, {"volunteer": v_rw_ok, "home": h_rw_ok}, kind="disagree")

    # Compare per-class mean effects
    home_classes = ((home_result.get("fresh_judged") or {}).get("classes") or {})
    vol_classes = ret.get("summary") or {}
    class_diffs: Dict[str, Any] = {}
    all_same_bits = True
    all_within_tol = True
    for cls_name, hdata in home_classes.items():
        h_mean = _clean_float((hdata.get("frames") or {}).get("mean"))
        h_floor = _clean_float((hdata.get("house_bar") or {}).get("floor")) or 0.0
        v_mean = _clean_float((vol_classes.get(cls_name) or {}).get("mean"))
        if h_mean is None or v_mean is None:
            all_same_bits = False
            all_within_tol = False
            class_diffs[cls_name] = {"home": h_mean, "volunteer": v_mean, "diff": None, "ok": False}
            continue
        diff = abs(v_mean - h_mean)
        tol = max(float(class_floor_mult) * h_floor, float(class_rel_tol) * max(1.0, abs(h_mean)), float(class_min_tol))
        same_sign = (h_mean == 0.0 and v_mean == 0.0) or ((h_mean * v_mean) > 0.0)
        if diff > same_bits_tol:
            all_same_bits = False
        ok_cls = bool(same_sign and diff <= tol) if abs(h_mean) > tol else bool(diff <= tol)
        if not ok_cls:
            all_within_tol = False
        class_diffs[cls_name] = {"home": h_mean, "volunteer": v_mean, "diff": diff, "tol": tol, "ok": ok_cls}

    _chk("class_effects_agree", all_within_tol, class_diffs, kind="disagree")

    if reject_reasons:
        outcome = "rejected"
        match_class = None
    elif disagree_reasons:
        outcome = "disagreed"
        match_class = "disagreed"
    else:
        outcome = "accepted"
        match_class = "same_bits" if all_same_bits else "other_card_match"

    verdict_doc = {
        "v": KIT_VERSION,
        "judged_at": _utc_now(),
        "outcome": outcome,
        "match_class": match_class,
        "volunteer": ret.get("volunteer"),
        "bundle_tar": str(Path(bundle_tar).name),
        "bundle_sha256": _sha_file(Path(bundle_tar)),
        "trial_name": reg["trial"]["trial_name"],
        "trial_sha256": reg["trial_sha256"],
        "home_verdict": home_verdict,
        "volunteer_verdict": vol_verdict,
        "class_diffs": class_diffs,
        "unpacked_files": unpacked,
        "reject_reasons": reject_reasons,
        "disagree_reasons": disagree_reasons,
        "checks": checks,
    }
    write_json_atomic(stage / "intake_verdict.json", verdict_doc, indent=2)
    return verdict_doc
