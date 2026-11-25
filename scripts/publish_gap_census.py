#!/usr/bin/env python3
"""Publish finalized Stage 1 & Stage 2 gap-census artifacts additively into atlas_data/census/.

Never overwrites or modifies existing files in census/runs/ or index/. Only populates:
  - <dst>/forced_odd/
  - <dst>/input_census/
  - <dst>/fdr_rescue/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import census as cz  # noqa: E402

INPUT_CENSUS_KEYS = frozenset({
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_prereg_hashes(prereg_dir: Path) -> int:
    sums_path = prereg_dir / "SHA256SUMS"
    if not sums_path.exists():
        raise FileNotFoundError(f"Missing pre-registration SHA256SUMS: {sums_path}")
    checked = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("* ")
        target = (prereg_dir / rel).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Missing hashed file referenced by SHA256SUMS: {target}")
        got = _sha256(target)
        if got != digest:
            raise RuntimeError(f"Pre-registration SHA256 mismatch for {rel}: expected {digest}, got {got}")
        checked += 1
    return checked


def validate_input_census_npz(path: Path) -> None:
    with np.load(path) as z:
        keys = set(z.files)
        if keys != INPUT_CENSUS_KEYS:
            raise RuntimeError(f"{path.name}: unexpected NPZ keys {sorted(keys)} != {sorted(INPUT_CENSUS_KEYS)}")
        for k in z.files:
            dt = z[k].dtype
            if dt.kind in ("O", "U", "S"):
                raise RuntimeError(f"{path.name}[{k}]: disallowed string/object dtype {dt}")


def atomic_copy(src: Path, dst: Path) -> Tuple[int, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    src_hash = _sha256(src)
    dst_hash = _sha256(dst)
    if src_hash != dst_hash:
        raise RuntimeError(f"Hash mismatch after copying {src} -> {dst}")
    return dst.stat().st_size, dst_hash


def plan_manifest(src_root: Path, dst_root: Path) -> List[Tuple[Path, Path]]:
    fo_src = src_root / "census_forced_odd"
    ic_src = src_root / "input_census_L0_L29"
    fdr_src = src_root / "census_fdr_sweep"

    pairs: List[Tuple[Path, Path]] = []

    # 1. forced_odd (L55..L59 whole + prereg references)
    fo_dst = dst_root / "forced_odd"
    for layer in range(55, 60):
        for suffix in ("_forced_odd.npz", "_forced_odd.meta.json", "_forced_odd_summary.txt", "_forced_odd_examples.txt"):
            s = fo_src / "whole" / f"L{layer}{suffix}"
            if not s.exists():
                raise FileNotFoundError(f"Missing forced_odd artifact: {s}")
            pairs.append((s, fo_dst / s.name))
    for name in ("WHOLE_SUMMARY.txt",):
        s = fo_src / "whole" / name
        if not s.exists():
            raise FileNotFoundError(f"Missing forced_odd summary: {s}")
        pairs.append((s, fo_dst / name))
    pairs.append((fo_src / "prereg" / "RULES.md", fo_dst / "prereg_RULES.md"))
    pairs.append((fo_src / "prereg" / "SHA256SUMS", fo_dst / "prereg_SHA256SUMS"))
    pairs.append((fo_src / "trials" / "RESULT.md", fo_dst / "prereg_RESULT.md"))
    pairs.append((fo_src / "trials" / "trials_summary.json", fo_dst / "trials_summary.json"))

    # 2. input_census (L00..L29)
    ic_dst = dst_root / "input_census"
    for layer in range(30):
        s = ic_src / f"L{layer:02d}_input_census.npz"
        if not s.exists():
            raise FileNotFoundError(f"Missing input_census artifact: {s}")
        pairs.append((s, ic_dst / s.name))
    for name in ("summary.json", "summary.txt", "ablation_check.json", "ablation_check.txt"):
        s = ic_src / name
        if not s.exists():
            raise FileNotFoundError(f"Missing input_census metadata: {s}")
        pairs.append((s, ic_dst / name))

    # 3. fdr_rescue (whole layers only; exclude trial snapshots)
    fdr_dst = dst_root / "fdr_rescue"
    whole_files = sorted(
        p for p in fdr_src.glob("L*_whole_named.json") if re.fullmatch(r"L\d+_whole_named\.json", p.name)
    )
    if not whole_files:
        raise FileNotFoundError(f"No L*_whole_named.json files found in {fdr_src}")
    for s in whole_files:
        pairs.append((s, fdr_dst / s.name))
    for name in ("summary.json", "summary.txt"):
        s = fdr_src / name
        if not s.exists():
            raise FileNotFoundError(f"Missing fdr_rescue metadata: {s}")
        pairs.append((s, fdr_dst / name))

    for src_p, dst_p in pairs:
        if not src_p.exists():
            raise FileNotFoundError(f"Source file missing: {src_p}")
        rel = dst_p.relative_to(dst_root)
        if rel.parts[0] not in ("forced_odd", "input_census", "fdr_rescue"):
            raise RuntimeError(f"Refusing to write outside allowed gap subdirs: {dst_p}")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=REPO_ROOT / "experiments" / "results")
    ap.add_argument("--dst", type=Path, default=None, help="Target census directory (defaults to backend.census.census_dir()).")
    ap.add_argument("--verify-only", action="store_true", help="Verify sources and any already-published destination files without writing.")
    args = ap.parse_args()

    src_root = args.src.resolve()
    dst_root = (args.dst if args.dst is not None else cz.census_dir()).resolve()
    if not dst_root.exists():
        raise FileNotFoundError(f"Destination census dir does not exist: {dst_root}")
    if dst_root.name == "runs" or (dst_root / "runs").resolve() == dst_root:
        raise RuntimeError(f"Refusing destination pointing at runs/: {dst_root}")
    if "index" in dst_root.parts:
        raise RuntimeError(f"Refusing destination inside index/: {dst_root}")

    n_prereg = verify_prereg_hashes(src_root / "census_forced_odd" / "prereg")
    pairs = plan_manifest(src_root, dst_root)

    # Validate zero-raw-text guarantee on every input_census NPZ before publishing
    for src_p, _ in pairs:
        if src_p.name.endswith("_input_census.npz"):
            validate_input_census_npz(src_p)

    counts: Dict[str, int] = {"forced_odd": 0, "input_census": 0, "fdr_rescue": 0}
    total_bytes = 0
    for src_p, dst_p in pairs:
        sub = dst_p.relative_to(dst_root).parts[0]
        if args.verify_only:
            if not dst_p.exists():
                raise FileNotFoundError(f"Missing published file: {dst_p}")
            if _sha256(src_p) != _sha256(dst_p):
                raise RuntimeError(f"Published file hash mismatch: {dst_p}")
            sz = dst_p.stat().st_size
        else:
            sz, _ = atomic_copy(src_p, dst_p)
        counts[sub] += 1
        total_bytes += sz

    # Also verify fdr_rescue summary parses cleanly
    json.loads((dst_root / "fdr_rescue" / "summary.json").read_text(encoding="utf-8"))
    print(
        f"{'VERIFIED' if args.verify_only else 'PUBLISHED'} gap-census artifacts into {dst_root}\n"
        f"  prereg SHA256SUMS : {n_prereg}/10 OK\n"
        f"  forced_odd        : {counts['forced_odd']} files\n"
        f"  input_census      : {counts['input_census']} files (zero raw corpus text verified)\n"
        f"  fdr_rescue        : {counts['fdr_rescue']} files\n"
        f"  total size        : {total_bytes / (1024 * 1024):.2f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
