"""Provenance helpers and pre-registration SHA-256 verification (stdlib-only)."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def utc_iso(ts: Optional[float] = None) -> str:
    """Format a UTC timestamp as an ISO-8601 string with ``Z`` suffix."""
    dt = datetime.now(timezone.utc) if ts is None else datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    """Compute hex SHA-256 of ``path`` in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def repo_commit(repo_root: Path) -> Tuple[Optional[str], Optional[bool]]:
    """Return ``(HEAD sha, is_dirty)`` for ``repo_root``, or ``(None, None)`` outside git."""
    root = str(Path(repo_root).resolve())
    try:
        sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(
            subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def verify_sha256sums(sums_path: Path, base_dir: Optional[Path] = None) -> Dict[str, str]:
    """Verify every line of a GNU-style ``SHA256SUMS`` manifest; raise on missing or mismatch."""
    sums = Path(sums_path)
    if not sums.is_file():
        raise FileNotFoundError(f"Missing pre-registration SHA256SUMS: {sums}")
    root = Path(base_dir).resolve() if base_dir is not None else sums.parent.resolve()
    expected: Dict[str, str] = {}
    for raw in sums.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed line in {sums}: {raw!r}")
        digest, rel = parts[0].lower(), parts[1].lstrip("* ")
        target = (root / rel).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Missing hashed file referenced by SHA256SUMS: {target}")
        got = file_sha256(target)
        if got != digest:
            raise RuntimeError(f"Pre-registration SHA256 mismatch for {rel}: expected {digest}, got {got}")
        expected[rel] = got
    if not expected:
        raise RuntimeError(f"Empty SHA256SUMS manifest: {sums}")
    return expected


def weights_id_from_card(weights: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract a concise stable identifier string from a runtime ``weights_card`` dict."""
    if not isinstance(weights, dict):
        return None
    for key in ("files_sha256", "id", "sha256_header", "snapshot", "repo"):
        val = weights.get(key)
        if val:
            return str(val)
    return str(weights.get("format")) if weights.get("format") else None


def build_provenance(
    repo_root: Path,
    code_files: Sequence[Path | str],
    *,
    started_at: str,
    finished_at: str,
    model_id: Optional[str] = None,
    weights: Optional[Dict[str, Any]] = None,
    weights_id: Optional[str] = None,
    neuron_ids: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a uniform provenance dictionary for experiment summaries and meta files."""
    root = Path(repo_root).resolve()
    commit, dirty = repo_commit(root)
    code_hashes: Dict[str, str] = {}
    for cf in code_files:
        p = Path(cf)
        if not p.is_absolute():
            p = (root / p).resolve()
        try:
            key = p.relative_to(root).as_posix()
        except ValueError:
            key = p.name
        if p.is_file():
            code_hashes[key] = file_sha256(p)

    wid = weights_id if weights_id is not None else weights_id_from_card(weights)
    prov: Dict[str, Any] = {
        "git_commit": commit,
        "git_dirty": dirty,
        "code_sha256": code_hashes,
        "started_at": started_at,
        "finished_at": finished_at,
        "model": model_id,
        "weights": weights,
        "weights_id": wid,
    }
    if neuron_ids is not None:
        if isinstance(neuron_ids, str):
            prov["neuron_ids"] = neuron_ids
        elif isinstance(neuron_ids, dict):
            prov["neuron_ids"] = neuron_ids
        else:
            nlist = [int(x) for x in neuron_ids]
            prov["neuron_count"] = len(nlist)
            if len(nlist) >= 4 and nlist == list(range(nlist[0], nlist[0] + len(nlist))):
                prov["neuron_ids"] = f"{nlist[0]}-{nlist[-1]}"
            else:
                prov["neuron_ids"] = ",".join(str(x) for x in nlist)
    return prov
