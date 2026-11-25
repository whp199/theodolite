#!/usr/bin/env python3
"""Publish exploratory supplementary artifacts (`forced_odd`, `input_census`) into ``supplementary_dir()``.

Defaults to verify-only (`--publish` is required to write files). Refuses to overwrite existing
files whose contents differ unless given `--force` (always printing what would change), refuses any
destination inside `runs/` or `index/` at any depth, writes `MANIFEST.json`, and checks
`input_census` NPZ files to guarantee zero stored text/object arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_gap_views as agv  # noqa: E402
from backend import gap_provenance as gp  # noqa: E402

REQUIRED_INPUT_CENSUS_KEYS = frozenset({
    "eligible",
    "named_fdp05",
    "named_fdp10",
    "score",
    "top_token_id",
    "top_offset",
    "top_cnt_e",
    "top_cnt_o",
    "top_lift",
    "bigram_rep",
    "smp_hits",
    "smp_total",
})

ALLOWED_INPUT_CENSUS_KEYS = REQUIRED_INPUT_CENSUS_KEYS | frozenset({
    "raw_2plus2",
    "other_face_null",
    "smp_freq_hits",
    "next_token_id",
    "next_cnt_e",
    "next_cnt_o",
    "next_lift",
})


def validate_input_census_npz(path: Path) -> None:
    """Verify required numeric schema, zero object/string arrays, and causal offsets (-1/0) in ``top_offset``."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = REQUIRED_INPUT_CENSUS_KEYS - keys
        if missing:
            raise RuntimeError(f"{path.name}: missing required NPZ keys {sorted(missing)}")
        extra = keys - ALLOWED_INPUT_CENSUS_KEYS
        if extra:
            raise RuntimeError(f"{path.name}: unexpected NPZ keys {sorted(extra)}")
        for k in z.files:
            dt = z[k].dtype
            if dt.kind in ("O", "U", "S"):
                raise RuntimeError(f"{path.name}[{k}]: disallowed string/object dtype {dt}")
        offs = np.asarray(z["top_offset"])
        if np.any((offs < -1) | (offs > 0)):
            raise RuntimeError(f"{path.name}[top_offset]: non-causal offset outside (-1, 0) found in triggers")


def atomic_copy(src: Path, dst: Path) -> Tuple[int, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    src_hash = gp.file_sha256(src)
    dst_hash = gp.file_sha256(dst)
    if src_hash != dst_hash:
        raise RuntimeError(f"Hash mismatch after copying {src} -> {dst}")
    return dst.stat().st_size, dst_hash


def atomic_write_json(path: Path, obj: Any) -> Tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path.stat().st_size, gp.file_sha256(path)


def check_prereg_state(prereg_dir: Path) -> Dict[str, Any]:
    """Inspect ``SHA256SUMS`` vs ``AMENDMENTS.txt``; fail if any mismatch is undocumented."""
    sums_path = prereg_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise FileNotFoundError(f"Missing SHA256SUMS: {sums_path}")
    amend_path = prereg_dir / "AMENDMENTS.txt"
    amend_text = amend_path.read_text(encoding="utf-8") if amend_path.is_file() else ""
    matched: List[str] = []
    amended: List[str] = []
    undocumented: List[str] = []

    for raw in sums_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("* ")
        target = (prereg_dir / rel).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Missing hashed file referenced by SHA256SUMS: {target}")
        if gp.file_sha256(target) == digest:
            matched.append(rel)
        else:
            name = Path(rel).name
            if amend_text and (rel in amend_text or name in amend_text):
                amended.append(rel)
            else:
                undocumented.append(rel)
    if undocumented:
        raise RuntimeError(
            f"SHA256SUMS mismatch without entry in AMENDMENTS.txt: {', '.join(undocumented)}"
        )
    return {"matched": matched, "amended": amended}


def plan_manifest(
    src_root: Path,
    dst_root: Path,
    *,
    strict_full: bool = True,
) -> List[Tuple[Path, Path]]:
    fo_src = src_root / "census_forced_odd"
    ic_src = src_root / "input_census_L0_L29"
    pairs: List[Tuple[Path, Path]] = []

    # 1. forced_odd
    fo_dst = dst_root / "forced_odd"
    if strict_full:
        fo_layers: Sequence[int] = range(55, 60)
    else:
        fo_layers = sorted(
            int(m.group(1))
            for p in (fo_src / "whole").glob("L*_forced_odd.npz")
            if (m := re.fullmatch(r"L(\d+)_forced_odd\.npz", p.name))
        )
    for layer in fo_layers:
        req = ("_forced_odd.npz", "_forced_odd.meta.json")
        opt = ("_forced_odd_summary.txt", "_forced_odd_examples.txt")
        for suffix in req:
            s = fo_src / "whole" / f"L{layer}{suffix}"
            if not s.is_file():
                raise FileNotFoundError(f"Missing forced_odd artifact: {s}")
            pairs.append((s, fo_dst / s.name))
        for suffix in opt:
            s = fo_src / "whole" / f"L{layer}{suffix}"
            if s.is_file():
                pairs.append((s, fo_dst / s.name))

    for sub, src_name, dst_name in (
        ("whole", "WHOLE_SUMMARY.txt", "WHOLE_SUMMARY.txt"),
        ("whole", "lens_agreement.json", "lens_agreement.json"),
        ("whole", "lens_agreement.txt", "lens_agreement.txt"),
        ("prereg", "RULES.md", "prereg_RULES.md"),
        ("prereg", "AMENDMENTS.txt", "prereg_AMENDMENTS.txt"),
        ("trials", "RESULT.md", "prereg_RESULT.md"),
        ("trials", "trials_summary.json", "trials_summary.json"),
    ):
        s = fo_src / sub / src_name
        if s.is_file():
            pairs.append((s, fo_dst / dst_name))

    # 2. input_census
    ic_dst = dst_root / "input_census"
    if strict_full:
        ic_layers: Sequence[int] = range(30)
    else:
        ic_layers = sorted(
            int(m.group(1))
            for p in ic_src.glob("L*_input_census.npz")
            if (m := re.fullmatch(r"L(\d+)_input_census\.npz", p.name))
        )
    for layer in ic_layers:
        s = ic_src / f"L{layer:02d}_input_census.npz"
        if not s.is_file():
            raise FileNotFoundError(f"Missing input_census artifact: {s}")
        pairs.append((s, ic_dst / s.name))

    req_ic_meta = ("summary.json",)
    opt_ic_meta = ("summary.txt", "NOTE.md", "ablation_check.json", "ablation_check.txt")
    for name in req_ic_meta:
        s = ic_src / name
        if not s.is_file():
            raise FileNotFoundError(f"Missing input_census metadata: {s}")
        pairs.append((s, ic_dst / name))
    for name in opt_ic_meta:
        s = ic_src / name
        if s.is_file():
            pairs.append((s, ic_dst / name))

    if not pairs:
        raise FileNotFoundError(f"No publishable forced_odd or input_census artifacts found in {src_root}")

    for src_p, dst_p in pairs:
        if not src_p.is_file():
            raise FileNotFoundError(f"Source file missing: {src_p}")
        rel = dst_p.relative_to(dst_root)
        if rel.parts[0] not in agv.REGIME_LABELS:
            raise RuntimeError(f"Refusing to write outside allowed gap subdirs: {dst_p}")
    return pairs


def _validate_dst(dst_root: Path) -> None:
    resolved = dst_root.resolve()
    if "runs" in resolved.parts:
        raise RuntimeError(f"Refusing destination inside runs/: {resolved}")
    if "index" in resolved.parts:
        raise RuntimeError(f"Refusing destination inside index/: {resolved}")


def _extract_model_and_weights(pairs: Sequence[Tuple[Path, Path]]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    model_id: Optional[str] = None
    weights: Optional[Dict[str, Any]] = None
    for src_p, dst_p in pairs:
        if dst_p.name.endswith(".meta.json") or dst_p.name == "summary.json":
            try:
                doc = json.loads(src_p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and doc.get("model"):
                m = str(doc["model"])
                if model_id is not None and m != model_id:
                    raise RuntimeError(f"Conflicting model ids in publish sources: {model_id!r} vs {m!r}")
                model_id = m
                if weights is None and isinstance(doc.get("weights"), dict):
                    weights = doc["weights"]
    return model_id, weights, gp.weights_id_from_card(weights)


def publish_regimes(
    src_root: Path,
    dst_root: Optional[Path] = None,
    *,
    publish: bool = False,
    force: bool = False,
    verify_prereg: bool = True,
    strict_full: bool = True,
) -> Dict[str, Any]:
    """Verify or publish supplementary regime files into ``dst_root`` (defaults to ``agv.supplementary_dir()``)."""
    src = Path(src_root).resolve()
    dst = (Path(dst_root) if dst_root is not None else agv.supplementary_dir()).resolve()
    _validate_dst(dst)

    prereg_state = check_prereg_state(src / "census_forced_odd" / "prereg") if verify_prereg else {"matched": [], "amended": []}
    pairs = plan_manifest(src, dst, strict_full=strict_full)
    for src_p, _ in pairs:
        if src_p.name.endswith("_input_census.npz"):
            validate_input_census_npz(src_p)
    model_id, weights, weights_id = _extract_model_and_weights(pairs)

    counts: Dict[str, int] = {k: 0 for k in agv.REGIME_LABELS}
    manifest_files: Dict[str, Any] = {}
    differs: List[Tuple[Path, Path, str, str]] = []
    missing_dst: List[Path] = []
    total_bytes = 0

    for src_p, dst_p in pairs:
        src_hash = gp.file_sha256(src_p)
        src_size = src_p.stat().st_size
        rel = dst_p.relative_to(dst).as_posix()
        try:
            src_rel = src_p.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            src_rel = src_p.name
        manifest_files[rel] = {"sha256": src_hash, "bytes": src_size, "source_rel": src_rel}

        if not dst_p.exists():
            missing_dst.append(dst_p)
        else:
            dst_hash = gp.file_sha256(dst_p)
            if dst_hash != src_hash:
                differs.append((src_p, dst_p, src_hash, dst_hash))

    if differs:
        for src_p, dst_p, src_h, dst_h in differs:
            verb = "OVERWRITE" if (publish and force) else "DIFFERS"
            print(f"  {verb}: {dst_p} (dst={dst_h[:12]} != src={src_h[:12]})", flush=True)

    if not publish:
        if missing_dst:
            raise FileNotFoundError(f"Verify-only failed: missing published file(s), first={missing_dst[0]}")
        if differs:
            raise RuntimeError(
                f"Verify-only failed: {len(differs)} published file(s) differ from source (use --publish --force to update)"
            )
        man_path = dst / "MANIFEST.json"
        if not man_path.is_file():
            raise FileNotFoundError(f"Verify-only failed: missing MANIFEST.json at {man_path}")
        for src_p, dst_p in pairs:
            sub = dst_p.relative_to(dst).parts[0]
            counts[sub] += 1
            total_bytes += dst_p.stat().st_size
        return {
            "mode": "verify_only",
            "dst": str(dst),
            "counts": counts,
            "total_bytes": total_bytes,
            "prereg": prereg_state,
            "model": model_id,
            "weights_id": weights_id,
        }

    if differs and not force:
        raise RuntimeError(
            f"Refusing to overwrite {len(differs)} existing file(s) with different contents without --force"
        )

    dst.mkdir(parents=True, exist_ok=True)
    for src_p, dst_p in pairs:
        sub = dst_p.relative_to(dst).parts[0]
        if dst_p.exists() and gp.file_sha256(dst_p) == manifest_files[dst_p.relative_to(dst).as_posix()]["sha256"]:
            sz = dst_p.stat().st_size
        else:
            sz, _ = atomic_copy(src_p, dst_p)
        counts[sub] += 1
        total_bytes += sz

    commit, dirty = gp.repo_commit(REPO_ROOT)
    manifest = {
        "published_at": gp.utc_iso(),
        "source_commit": commit,
        "source_dirty": dirty,
        "model": model_id,
        "weights": weights,
        "weights_id": weights_id,
        "prereg_sha256_check": prereg_state,
        "regimes": sorted(agv.REGIME_LABELS),
        "counts": counts,
        "files": manifest_files,
    }
    atomic_write_json(dst / "MANIFEST.json", manifest)
    return {
        "mode": "published",
        "dst": str(dst),
        "counts": counts,
        "total_bytes": total_bytes,
        "prereg": prereg_state,
        "model": model_id,
        "weights_id": weights_id,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=REPO_ROOT / "experiments" / "results")
    ap.add_argument("--dst", type=Path, default=None, help="Target directory (defaults to backend.atlas_gap_views.supplementary_dir()).")
    ap.add_argument("--publish", action="store_true", help="Copy verified files into --dst (default is verify-only).")
    ap.add_argument("--verify-only", action="store_true", help="Explicitly run in verify-only mode (default).")
    ap.add_argument("--force", action="store_true", help="Allow overwriting existing differing files when --publish is set.")
    ap.add_argument("--no-strict-full", action="store_true", help="Discover available layers instead of requiring L55-59 and L0-29.")
    args = ap.parse_args()

    do_publish = bool(args.publish and not args.verify_only)
    res = publish_regimes(
        args.src,
        args.dst,
        publish=do_publish,
        force=bool(args.force),
        verify_prereg=True,
        strict_full=not bool(args.no_strict_full),
    )
    print(
        f"{res['mode'].upper()} supplementary regimes in {res['dst']}\n"
        f"  prereg matched    : {len(res['prereg']['matched'])} | amended: {len(res['prereg']['amended'])}\n"
        f"  forced_odd        : {res['counts']['forced_odd']} files\n"
        f"  input_census      : {res['counts']['input_census']} files (zero stored text & causal offsets verified)\n"
        f"  model             : {res['model']} (weights_id={res['weights_id']})\n"
        f"  total size        : {res['total_bytes'] / (1024 * 1024):.2f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
