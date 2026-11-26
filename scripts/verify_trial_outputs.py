#!/usr/bin/env python3
"""Standalone verifier for confirmatory trial outputs without modifying locked CODE_SHA256SUMS."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_in_use_trials as ct  # noqa: E402

FORBIDDEN_CORPUS_KEYS = ('"kept_tail"', '"fit_snippets"', '"a_txt"', '"b_txt"')


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_code_sums(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        out[rel.strip()] = digest.strip()
    return out


def _verify_provenance_obj(obj: dict, label: Path, code_sums: Dict[str, str], shasums_sha: str, *, require_clean_dirty: bool) -> None:
    prereg = obj.get("prereg")
    assert prereg, f"{label} missing prereg block"
    if require_clean_dirty:
        assert obj.get("git_dirty") is False, f"{label} has git_dirty={obj.get('git_dirty')}"
    csums = obj.get("code_sha256")
    assert isinstance(csums, dict) and csums, f"{label} missing code_sha256 block"
    for rel, recorded_sha in csums.items():
        if rel.endswith("prereg/SHA256SUMS"):
            assert recorded_sha == shasums_sha, f"{label} SHA256SUMS digest mismatch: {recorded_sha} != {shasums_sha}"
        elif rel in code_sums:
            assert recorded_sha == code_sums[rel], f"{label} code_sha256 mismatch for {rel}: {recorded_sha} != {code_sums[rel]}"
        else:
            fpath = REPO_ROOT / rel
            assert fpath.is_file(), f"{label} references missing file {fpath}"
            actual = _sha256_file(fpath)
            assert recorded_sha == actual, f"{label} file digest mismatch for {rel}: {recorded_sha} != {actual}"


def verify_recorded_trial_outputs() -> None:
    trials = [
        ("in_use_lens_trial", False),
        ("in_use_pattern_trial", True),
        ("u2_rollout_trial", True),
    ]
    for dirname, is_pattern_or_u2 in trials:
        rdir = REPO_ROOT / "experiments/results" / dirname
        if not rdir.is_dir():
            continue
        pdir = rdir / "prereg"
        code_sums = _load_code_sums(pdir / "CODE_SHA256SUMS")
        shasums_sha = _sha256_file(pdir / "SHA256SUMS")

        for p in sorted(rdir.glob("*.json")):
            raw = p.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_CORPUS_KEYS:
                assert forbidden not in raw, f"{p} leaked raw corpus key {forbidden}"
            obj = json.loads(raw)

            if p.name == "verdict.json":
                # P1 & P3 top-level verdict.json calls build_provenance after writing L*_verdict.json
                _verify_provenance_obj(obj, p, code_sums, shasums_sha, require_clean_dirty=False)
            elif p.name.startswith("L") and p.name.endswith("_verdict.json") and is_pattern_or_u2:
                _verify_provenance_obj(obj, p, code_sums, shasums_sha, require_clean_dirty=True)
            elif p.name.startswith("L") and p.name.endswith("_verdict.json"):
                assert "layer" in obj and "gates" in obj and "pass" in obj, f"{p} malformed per-layer verdict"


def main() -> None:
    core_tests = [
        ct.test_p1_gates_and_cross_pool,
        ct.test_p2_u2_track2a_and_track2b,
        ct.test_p3_derangement_doc_isolation_and_gates,
        ct.test_sha_tamper_detection,
    ]
    for fn in core_tests:
        fn()
        print(f"PASS {fn.__name__}")
    verify_recorded_trial_outputs()
    print("PASS verify_recorded_trial_outputs")
    print(f"All {len(core_tests) + 1} standalone trial verifications passed.")


if __name__ == "__main__":
    main()
