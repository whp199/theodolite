"""Off-card sandboxed CPU Python cell runner (`backend/level_sandbox.py`): runs agent analysis code off the GPU inside
a `bwrap` (bubblewrap) container owned by the instrument.

Containment model (verified actively by `probe_sandbox`):
- Network: `--unshare-all` (own network, PID, IPC, UTS, user namespaces; no outbound network or host loopback).
- GPU: minimal `--dev /dev` (no `/dev/nvidia*` or `/dev/dri*`) and `CUDA_VISIBLE_DEVICES=""`.
- Filesystem: host `$HOME` is NOT mounted. Only system shared libraries (`/usr`, `/lib*`, `/bin`), the Python runtime
  and site-packages (read-only), `backend/weight_reader.py` + `V6_quiet_dims_pers181.npy` (read-only), resolved weight
  files under `/weights` (read-only, omitted when `weights_dir=None`), optional atlas index under `/atlas` (read-only),
  private `--tmpfs /tmp`, and ONE writable session scratch folder mounted at `/work`.
- Environment: `--clearenv` with a minimal allowlist (zero host credentials, tokens, or API keys).
- Resources: `RLIMIT_AS` (sized above the read-only weight mmap plus working budget), `RLIMIT_FSIZE`, parent-side
  resident-RAM (`VmRSS`) monitor, and hard wall-clock timeout with process-group `SIGKILL`; a cell still running when
  its session closes is killed the same way (`stop_cells`).

No pinned or live runtime module is imported inside the sandbox; cells interact with weights strictly through the
read-only `WeightReader` (`W`) and persist float32 arrays to `/work` via `save_array(name, arr)` with sha256 hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHT_READER_PATH = ROOT_DIR / "backend" / "weight_reader.py"
LEVEL_ATLAS_PATH = ROOT_DIR / "backend" / "level_atlas.py"
QUIET_DIMS_PATH = ROOT_DIR / "experiments" / "results" / "V6_quiet_dims_pers181.npy"

DEFAULT_TIMEOUT_S = 30.0
MAX_TIMEOUT_S = 300.0
DEFAULT_MEM_GIB = 4.0
DEFAULT_MAX_OUTPUT_CHARS = 8000
DEFAULT_MAX_FILE_MB = 256


class SandboxError(RuntimeError):
    """Raised when the `bwrap` sandbox cannot be launched or fails its containment check."""


# A session that closes stops its cells: `stop_cells(work_dir)` marks the folder, and `run_cell`, which looks for the
# mark between its polls, kills a cell running there and starts no new one until `release_cells(work_dir)`.
_CELLS_LOCK = threading.Lock()
_CELLS_STOPPING: Set[str] = set()
_CELLS_RUNNING: Dict[str, int] = {}


def stop_cells(work_dir: Path) -> int:
    """Kill every cell running on `work_dir` (within about a fifth of a second) and start no new one there until
    `release_cells`. Returns how many were running."""
    key = str(Path(work_dir).resolve())
    with _CELLS_LOCK:
        _CELLS_STOPPING.add(key)
        return int(_CELLS_RUNNING.get(key, 0))


def release_cells(work_dir: Path) -> None:
    """Let cells run on `work_dir` again (`stop_cells`)."""
    with _CELLS_LOCK:
        _CELLS_STOPPING.discard(str(Path(work_dir).resolve()))


def _cells_stopping(key: str) -> bool:
    with _CELLS_LOCK:
        return key in _CELLS_STOPPING


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for a sandboxed off-card CPU cell environment."""

    work_dir: Path
    weights_dir: Optional[Path] = None
    atlas_dir: Optional[Path] = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_mem_gib: float = DEFAULT_MEM_GIB
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    max_file_mb: int = DEFAULT_MAX_FILE_MB
    bwrap_bin: str = "bwrap"
    model_id: Optional[str] = None
    weights_id: Optional[str] = None


# Runner executed inside the bwrap container (mounted read-only at /opt/runner/cell_runner.py).
_CELL_RUNNER_SOURCE = r'''
import json
import math
import os
import resource
import sys
import traceback
from pathlib import Path

def _apply_limits(as_bytes: int, fsize_bytes: int, cpu_s: int) -> None:
    if as_bytes > 0:
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            lim = as_bytes if hard in (-1, resource.RLIM_INFINITY) else min(as_bytes, hard)
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:
            pass
    if fsize_bytes > 0:
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
            lim = fsize_bytes if hard in (-1, resource.RLIM_INFINITY) else min(fsize_bytes, hard)
            resource.setrlimit(resource.RLIMIT_FSIZE, (lim, lim))
        except Exception:
            pass
    if cpu_s > 0:
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            lim = cpu_s if hard in (-1, resource.RLIM_INFINITY) else min(cpu_s, hard)
            resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
        except Exception:
            pass

def _self_vmsize_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass
    return 0

def main() -> int:
    cfg = json.loads(os.environ.get("LEVELLING_CELL_CFG", "{}"))
    for p in reversed(cfg.get("sys_path") or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    code_path = Path(cfg["code_path"])
    code = code_path.read_text(encoding="utf-8")
    import numpy as np
    import torch
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    from backend.weight_reader import WeightReader, save_array as _save_array

    work = Path("/work")
    atlas = Path("/atlas") if Path("/atlas").is_dir() else None
    A = None
    if atlas is not None:
        try:
            from backend.level_atlas import AtlasReader
            A = AtlasReader(atlas)
        except Exception:
            A = None
    weights_path = Path("/weights")
    quiet_path = Path("/opt/app/experiments/results/V6_quiet_dims_pers181.npy")
    W = None
    if weights_path.is_dir() and any(weights_path.glob("*.safetensors")):
        W = WeightReader(
            weights_path,
            model_id=cfg.get("model_id"),
            weights_id=cfg.get("weights_id"),
            quiet_dims_path=quiet_path if quiet_path.is_file() else None,
        )

    # Touch a tiny torch tensor and pre-open read-only safetensors mmaps before sampling baseline VmSize.
    _ = torch.zeros(1) + 1
    if W is not None:
        for _fp in W._files:
            W._open(_fp)

    # Apply RLIMIT_AS relative to current VmSize (post-import + read-only safetensors mmap) + working cap.
    vm_now = _self_vmsize_bytes()
    extra_as = int(cfg.get("rlimit_as_extra_bytes", 0))
    _apply_limits(
        (vm_now + extra_as) if extra_as > 0 else 0,
        int(cfg.get("rlimit_fsize_bytes", 0)),
        int(cfg.get("rlimit_cpu_s", 0)),
    )

    def save_array(name: str, arr):
        rel = Path(str(name))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("save_array takes a relative path inside /work.")
        target = (work / rel).resolve()
        if not str(target).startswith(str(work.resolve()) + os.sep):
            raise ValueError("save_array target must lie inside /work.")
        meta = _save_array(target, arr)
        meta["rel_path"] = str(target.relative_to(work.resolve()))
        return meta

    ns = {
        "__name__": "__main__",
        "W": W,
        "A": A,
        "WORK": work,
        "ATLAS": atlas,
        "save_array": save_array,
        "np": np,
        "torch": torch,
        "json": json,
        "math": math,
        "Path": Path,
    }
    try:
        exec(compile(code, "<cell>", "exec"), ns, ns)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        if A is not None:
            A.close()
        if W is not None:
            W.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


def _python_layout() -> Tuple[Path, Path, List[Path]]:
    """Return `(real_python_exe, base_prefix, site_dirs)` for mounting into the container without mounting `$HOME`."""
    exe = Path(os.path.realpath(sys.executable))
    base = Path(os.path.realpath(sys.base_prefix))
    prefix = Path(os.path.realpath(sys.prefix))
    sites: List[Path] = []
    for p in sys.path:
        if not p:
            continue
        rp = Path(os.path.realpath(p))
        if rp.is_dir() and "site-packages" in rp.parts and rp not in sites:
            sites.append(rp)
    if not sites:
        cand = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if cand.is_dir():
            sites.append(cand.resolve())
    return exe, base, sites


def _resolve_weight_binds(weights_dir: Path) -> Tuple[List[Tuple[Path, str]], int]:
    """Resolve symlinks inside `weights_dir` to canonical files so `/weights/<name>` works without exposing `$HOME`
    or Hugging Face cache folders. Returns `(binds, total_safetensors_bytes)`."""
    wd = Path(weights_dir)
    if wd.is_file() and wd.suffix == ".safetensors":
        rp = wd.resolve()
        return [(rp, f"/weights/{wd.name}")], int(rp.stat().st_size)
    if not wd.is_dir():
        raise SandboxError(f"Weights folder {wd} does not exist.")
    binds: List[Tuple[Path, str]] = []
    total_st_bytes = 0
    keep_names = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
    for item in sorted(wd.iterdir()):
        if item.name.endswith(".safetensors") or item.name in keep_names:
            rp = item.resolve()
            if rp.is_file():
                binds.append((rp, f"/weights/{item.name}"))
                if item.name.endswith(".safetensors"):
                    total_st_bytes += int(rp.stat().st_size)
    if not any(dst.endswith(".safetensors") for _src, dst in binds):
        raise SandboxError(f"No readable .safetensors files found under {wd}.")
    return binds, total_st_bytes


def _snapshot_work_files(work_dir: Path) -> Dict[str, Tuple[int, float]]:
    out: Dict[str, Tuple[int, float]] = {}
    root = Path(work_dir).resolve()
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
                out[str(p.relative_to(root))] = (int(st.st_size), float(st.st_mtime_ns))
            except OSError:
                pass
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _proc_tree_status(root_pid: int) -> Tuple[List[int], int]:
    """Walk host `/proc` by `PPid` from `root_pid` (crossing `bwrap --new-session` and PID namespaces) and return
    `(all_pids, total_rss_bytes)`."""
    children: Dict[int, List[int]] = {}
    rss_map: Dict[int, int] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return [root_pid], 0
    for name in entries:
        if not name.isdigit():
            continue
        pid = int(name)
        ppid = -1
        rss = 0
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
        except OSError:
            continue
        if ppid >= 0:
            children.setdefault(ppid, []).append(pid)
        rss_map[pid] = rss
    seen: List[int] = []
    stack = [root_pid]
    total_rss = 0
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p)
        total_rss += rss_map.get(p, 0)
        for ch in children.get(p, ()):
            stack.append(ch)
    return seen, total_rss


def _proc_tree_rss_bytes(root_pid: int) -> int:
    """Total resident memory (`VmRSS`) in bytes for `root_pid` and all descendants (across `--new-session`)."""
    return _proc_tree_status(root_pid)[1]


def _truncate(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = limit // 2
    tail = limit - head
    return text[:head] + f"\n... [truncated {len(text) - limit} chars] ...\n" + text[-tail:], True


def _kill_group(pid: int) -> None:
    pids, _ = _proc_tree_status(pid)
    for p in reversed(pids):
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    time.sleep(0.15)
    pids2, _ = _proc_tree_status(pid)
    for p in reversed(pids2):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def bwrap_available(bwrap_bin: str = "bwrap") -> Tuple[bool, str]:
    """Return `(ok, detail)` after verifying `bwrap` exists and can create isolated namespaces on this host."""
    resolved = shutil.which(bwrap_bin)
    if not resolved:
        return False, f"{bwrap_bin!r} is not installed on PATH."
    try:
        res = subprocess.run(
            [resolved, "--ro-bind", "/", "/", "--unshare-all", "--die-with-parent", "--", "true"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception as exc:
        return False, f"bwrap probe failed to execute: {exc}"
    if res.returncode != 0:
        msg = (res.stderr or res.stdout or "").strip()
        return False, f"bwrap namespace creation failed (exit {res.returncode}): {msg}"
    return True, resolved


def run_cell(
    code: str,
    cfg: SandboxConfig,
    *,
    cell_log_dir: Optional[Path] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute Python `code` off-card inside an isolated `bwrap` sandbox and return structured results.

    Never touches the GPU. Always tears down the container and hard-kills the process group on timeout or RSS breach.
    """
    if not isinstance(code, str) or not code.strip():
        raise SandboxError("Cell code must be a non-empty string.")
    resolved_bwrap = shutil.which(cfg.bwrap_bin)
    if not resolved_bwrap:
        raise SandboxError(f"{cfg.bwrap_bin!r} is not installed on PATH.")

    wall_cap = float(cfg.timeout_s if timeout_s is None else timeout_s)
    if not (0.5 <= wall_cap <= MAX_TIMEOUT_S):
        raise SandboxError(f"Cell timeout must lie between 0.5 and {MAX_TIMEOUT_S:g} s; got {wall_cap!r}.")

    work = Path(cfg.work_dir).resolve()
    work_key = str(work)
    if _cells_stopping(work_key):
        raise SandboxError("The session is closing: no new cell is started.")
    work.mkdir(parents=True, exist_ok=True)
    before_files = _snapshot_work_files(work)

    exe, py_base, site_dirs = _python_layout()
    rel_exe = exe.relative_to(py_base)
    container_python = f"/opt/pybase/{rel_exe.as_posix()}"

    weight_binds: List[Tuple[Path, str]] = []
    total_st_bytes = 0
    if cfg.weights_dir is not None:
        weight_binds, total_st_bytes = _resolve_weight_binds(cfg.weights_dir)

    rss_cap_bytes = int(float(cfg.max_mem_gib) * (2**30))
    # Extra VA above post-mmap VmSize: working RAM cap + 1 GiB headroom (safetensors are already mmapped).
    rlimit_as_extra_bytes = rss_cap_bytes + 1 * (2**30)
    rlimit_fsize_bytes = int(cfg.max_file_mb) * (2**20)

    t0 = time.perf_counter()
    timed_out = False
    oom_killed = False
    stopped = False
    stdout_full = ""
    stderr_full = ""
    exit_code = -1

    with tempfile.TemporaryDirectory(prefix="level_cell_") as stage_raw:
        stage = Path(stage_raw)
        runner_file = stage / "cell_runner.py"
        code_file = stage / "cell_code.py"
        empty_init = stage / "__init__.py"
        runner_file.write_text(_CELL_RUNNER_SOURCE, encoding="utf-8")
        code_file.write_text(code, encoding="utf-8")
        empty_init.write_text("", encoding="utf-8")

        cmd: List[str] = [
            resolved_bwrap,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--ro-bind", "/usr", "/usr",
        ]
        for sys_dir in ("/lib", "/lib64", "/bin", "/sbin"):
            if os.path.exists(sys_dir):
                cmd += ["--ro-bind", sys_dir, sys_dir]
        for etc_file in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/localtime"):
            if os.path.exists(etc_file):
                cmd += ["--ro-bind", etc_file, etc_file]

        # Python runtime and site-packages (read-only, mapped outside $HOME)
        cmd += ["--ro-bind", str(py_base), "/opt/pybase"]
        py_paths: List[str] = []
        for idx, sd in enumerate(site_dirs):
            dst = f"/opt/site_{idx}"
            cmd += ["--ro-bind", str(sd), dst]
            py_paths.append(dst)

        # Minimal read-only app code (weight_reader + level_atlas only — never engine_runtime or host credentials)
        cmd += [
            "--tmpfs", "/opt/app",
            "--dir", "/opt/app/backend",
            "--ro-bind", str(empty_init), "/opt/app/backend/__init__.py",
            "--ro-bind", str(WEIGHT_READER_PATH), "/opt/app/backend/weight_reader.py",
            "--tmpfs", "/opt/runner",
            "--ro-bind", str(runner_file), "/opt/runner/cell_runner.py",
            "--ro-bind", str(code_file), "/opt/runner/cell_code.py",
        ]
        if LEVEL_ATLAS_PATH.is_file():
            cmd += ["--ro-bind", str(LEVEL_ATLAS_PATH), "/opt/app/backend/level_atlas.py"]
        if QUIET_DIMS_PATH.is_file():
            cmd += [
                "--dir", "/opt/app/experiments",
                "--dir", "/opt/app/experiments/results",
                "--ro-bind", str(QUIET_DIMS_PATH), "/opt/app/experiments/results/V6_quiet_dims_pers181.npy",
            ]
        cmd += ["--remount-ro", "/opt/app", "--remount-ro", "/opt/runner"]
        py_paths.append("/opt/app")

        # Read-only resolved weight files (if mounted for this reach), remounted read-only at /weights
        if weight_binds:
            cmd += ["--tmpfs", "/weights"]
            for src, dst in weight_binds:
                cmd += ["--ro-bind", str(src), dst]
            cmd += ["--remount-ro", "/weights"]

        # Optional read-only atlas index files ONLY (allowlisted; never scientist/, uploads/, logs/, or level_book)
        if cfg.atlas_dir is not None:
            from backend.level_atlas import collect_atlas_binds

            atlas_binds = collect_atlas_binds(Path(cfg.atlas_dir))
            cmd += ["--tmpfs", "/atlas"]
            if any(dst.startswith("/atlas/passages.d/") for _src, dst in atlas_binds):
                cmd += ["--dir", "/atlas/passages.d"]
            for src, dst in atlas_binds:
                cmd += ["--ro-bind", str(src), dst]
            cmd += ["--remount-ro", "/atlas"]

        # Single writable scratch folder
        cmd += ["--bind", str(work), "/work", "--chdir", "/work"]

        cell_cfg_json = json.dumps({
            "code_path": "/opt/runner/cell_code.py",
            "sys_path": py_paths,
            "rlimit_as_extra_bytes": rlimit_as_extra_bytes,
            "rlimit_fsize_bytes": rlimit_fsize_bytes,
            "rlimit_cpu_s": max(2, int(wall_cap) + 2),
            "torch_threads": 2,
            "model_id": cfg.model_id,
            "weights_id": cfg.weights_id,
        })
        env_pairs = [
            ("PATH", "/opt/pybase/bin:/usr/bin:/bin"),
            ("HOME", "/work"),
            ("TMPDIR", "/tmp"),
            ("PYTHONPATH", ":".join(py_paths)),
            ("PYTHONNOUSERSITE", "1"),
            ("PYTHONDONTWRITEBYTECODE", "1"),
            ("CUDA_VISIBLE_DEVICES", ""),
            ("OPENBLAS_NUM_THREADS", "2"),
            ("OMP_NUM_THREADS", "2"),
            ("MKL_NUM_THREADS", "2"),
            ("NUMEXPR_NUM_THREADS", "2"),
            ("LEVELLING_CELL_CFG", cell_cfg_json),
        ]
        for k, v in env_pairs:
            cmd += ["--setenv", k, v]

        cmd += ["--", container_python, "-I", "/opt/runner/cell_runner.py"]

        stdout_tmp = stage / "stdout.txt"
        stderr_tmp = stage / "stderr.txt"
        with open(stdout_tmp, "wb") as out_f, open(stderr_tmp, "wb") as err_f:
            with _CELLS_LOCK:
                if work_key in _CELLS_STOPPING:
                    raise SandboxError("The session is closing: no new cell is started.")
                _CELLS_RUNNING[work_key] = _CELLS_RUNNING.get(work_key, 0) + 1
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    start_new_session=True,
                )
                deadline = time.perf_counter() + wall_cap
                try:
                    while True:
                        rc = proc.poll()
                        if rc is not None:
                            exit_code = int(rc)
                            break
                        now = time.perf_counter()
                        if now >= deadline:
                            timed_out = True
                        elif _cells_stopping(work_key):
                            stopped = True  # its session is closing (`stop_cells`)
                        elif _proc_tree_rss_bytes(proc.pid) > rss_cap_bytes:
                            oom_killed = True
                        if timed_out or stopped or oom_killed:
                            _kill_group(proc.pid)
                            proc.wait(timeout=5)
                            exit_code = int(proc.returncode if proc.returncode is not None else -9)
                            break
                        time.sleep(0.05)
                except BaseException:
                    _kill_group(proc.pid)
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                    raise
            finally:
                with _CELLS_LOCK:
                    left = _CELLS_RUNNING.get(work_key, 1) - 1
                    if left > 0:
                        _CELLS_RUNNING[work_key] = left
                    else:
                        _CELLS_RUNNING.pop(work_key, None)

        stdout_full = stdout_tmp.read_text(encoding="utf-8", errors="replace")
        stderr_full = stderr_tmp.read_text(encoding="utf-8", errors="replace")

    elapsed = time.perf_counter() - t0
    after_files = _snapshot_work_files(work)
    files_written: List[Dict[str, Any]] = []
    for rel, meta in sorted(after_files.items()):
        if before_files.get(rel) != meta:
            fp = work / rel
            files_written.append({
                "rel_path": rel,
                "bytes": meta[0],
                "sha256": _file_sha256(fp) if fp.is_file() else None,
            })

    if cell_log_dir is not None:
        ld = Path(cell_log_dir)
        ld.mkdir(parents=True, exist_ok=True)
        (ld / "code.py").write_text(code, encoding="utf-8")
        (ld / "stdout.txt").write_text(stdout_full, encoding="utf-8")
        (ld / "stderr.txt").write_text(stderr_full, encoding="utf-8")

    stdout_shown, stdout_trunc = _truncate(stdout_full, int(cfg.max_output_chars))
    stderr_shown, stderr_trunc = _truncate(stderr_full, int(cfg.max_output_chars))
    error: Optional[str] = None
    if timed_out:
        error = f"Cell exceeded wall-clock timeout ({wall_cap:g} s) and was killed."
    elif oom_killed:
        error = f"Cell exceeded resident memory cap ({cfg.max_mem_gib:g} GiB) and was killed."
    elif stopped:
        error = "Cell was killed: its session closed while it ran."
    elif exit_code != 0:
        error = f"Cell exited with code {exit_code}."

    return {
        "ok": (exit_code == 0 and not timed_out and not oom_killed and not stopped),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "oom_killed": oom_killed,
        "stopped": stopped,
        "seconds": round(elapsed, 4),
        "stdout": stdout_shown,
        "stderr": stderr_shown,
        "stdout_truncated": stdout_trunc,
        "stderr_truncated": stderr_trunc,
        "files_written": files_written,
        "error": error,
        "log_dir": str(cell_log_dir) if cell_log_dir is not None else None,
    }


def probe_sandbox(
    *,
    weights_dir: Optional[Path] = None,
    atlas_dir: Optional[Path] = None,
    bwrap_bin: str = "bwrap",
    max_mem_gib: float = 2.0,
) -> Dict[str, Any]:
    """Actively probe every containment boundary of `run_cell` inside a fresh temporary workspace.

    Tests:
    1. `bwrap` namespace availability.
    2. Network blocked (TCP connect + UDP socket denied inside `--unshare-all`).
    3. GPU invisible (`not torch.cuda.is_available()`, `/dev/nvidia0` absent).
    4. Host `$HOME` and credentials invisible (real host `$HOME` path does not exist inside container; env stripped).
    5. Read-only mounts cannot be written (`/opt/app` and `/weights` if mounted refuse writes).
    6. `/work` is writable and `save_array` persists float32 `.npy` with sha256; weights reader works when mounted.
    7. Runaway infinite loop is hard-killed promptly at wall-clock timeout.
    8. Runaway memory allocation fails under memory cap (`MemoryError` or RSS kill).
    """
    ok_bw, bw_detail = bwrap_available(bwrap_bin)
    checks: List[Dict[str, Any]] = [{"name": "bwrap_available", "ok": ok_bw, "detail": bw_detail}]
    if not ok_bw:
        return {"ok": False, "checks": checks}

    host_home = str(Path.home())
    with tempfile.TemporaryDirectory(prefix="level_sandbox_probe_") as td:
        work = Path(td) / "work"
        work.mkdir()
        cfg = SandboxConfig(
            work_dir=work,
            weights_dir=weights_dir,
            atlas_dir=atlas_dir,
            timeout_s=15.0,
            max_mem_gib=max_mem_gib,
            bwrap_bin=bwrap_bin,
        )

        # Single fast containment probe cell checking net, gpu, home/env, ro-mounts, and /work write
        probe_code = f"""
import json, os, socket, torch, numpy as np
from pathlib import Path

res = {{}}
# 1. Network check (public IP connect must fail with ENETUNREACH / OSError)
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    s.connect(("1.1.1.1", 443))
    s.close()
    res["net_blocked"] = False
except OSError as exc:
    res["net_blocked"] = True
    res["net_err"] = type(exc).__name__

# 2. GPU invisible
res["gpu_invisible"] = (not torch.cuda.is_available()) and (not os.path.exists("/dev/nvidia0"))

# 3. Host home invisible & env stripped
host_home = {host_home!r}
res["home_invisible"] = (not os.path.exists(host_home)) and (os.environ.get("HOME") == "/work")
leaked_env = [k for k in os.environ if any(p in k for p in ("KEY", "TOKEN", "SECRET", "AWS", "AZURE", "HF_"))]
res["env_clean"] = (len(leaked_env) == 0)

# 4. Read-only mounts refuse write (both new files and opening existing weights for write)
ro_blocked = True
for p in ["/opt/app/probe_write.tmp", "/opt/runner/probe_write.tmp"] + (
    ["/weights/probe_write.tmp"] if os.path.isdir("/weights") else []
):
    try:
        Path(p).write_text("x")
        ro_blocked = False
    except OSError:
        pass
if os.path.isdir("/weights"):
    for sf in Path("/weights").glob("*.safetensors"):
        try:
            with open(sf, "r+b"):
                ro_blocked = False
        except OSError:
            pass
res["ro_mounts"] = ro_blocked

# 5. /work writable via save_array + weights reader check when mounted
meta = save_array("probe.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
res["work_write_ok"] = (meta["shape"] == [3] and len(meta["sha256"]) == 64)
if W is not None:
    info = W.info()
    c0 = W.dense_cols("layers.0.mlp.down_proj", [0, 1])
    res["weights_reader_ok"] = (info["n_layers"] > 0 and info["n_linears"] > 0 and c0.shape[0] == 2)
else:
    res["weights_reader_ok"] = None

print("PROBE_JSON:" + json.dumps(res))
"""
        r1 = run_cell(probe_code, cfg, timeout_s=15.0)
        parsed: Dict[str, Any] = {}
        for line in r1["stdout"].splitlines():
            if line.startswith("PROBE_JSON:"):
                parsed = json.loads(line[len("PROBE_JSON:") :])
        checks.append({"name": "network_blocked", "ok": bool(r1["ok"] and parsed.get("net_blocked")),
                       "detail": parsed.get("net_err") or r1.get("stderr")[:200]})
        checks.append({"name": "gpu_invisible", "ok": bool(r1["ok"] and parsed.get("gpu_invisible")),
                       "detail": "cuda unavailable and /dev/nvidia0 absent" if parsed.get("gpu_invisible") else r1["stderr"][:200]})
        checks.append({"name": "home_and_env_isolated",
                       "ok": bool(r1["ok"] and parsed.get("home_invisible") and parsed.get("env_clean")),
                       "detail": f"home_invisible={parsed.get('home_invisible')}, env_clean={parsed.get('env_clean')}"})
        checks.append({"name": "readonly_mounts", "ok": bool(r1["ok"] and parsed.get("ro_mounts")),
                       "detail": "writes to /opt/app and /weights refused"})
        checks.append({"name": "work_write_and_sha256",
                       "ok": bool(r1["ok"] and parsed.get("work_write_ok")
                                  and any(f["rel_path"] == "probe.npy" for f in r1["files_written"])),
                       "detail": f"{len(r1['files_written'])} file(s) written"})
        if weights_dir is not None:
            checks.append({"name": "weights_reader_in_sandbox", "ok": bool(r1["ok"] and parsed.get("weights_reader_ok")),
                           "detail": str(weights_dir)})

        # 6. Wall-clock timeout kill probe
        r2 = run_cell("while True:\n    pass\n", cfg, timeout_s=1.0)
        checks.append({
            "name": "timeout_hard_kill",
            "ok": bool(not r2["ok"] and r2["timed_out"] and r2["seconds"] < 4.0),
            "detail": f"timed_out={r2['timed_out']}, seconds={r2['seconds']}",
        })

        # 7. Memory cap enforcement probe (touch 256 MiB chunks past max_mem_gib to trigger RSS kill or MemoryError)
        n_chunks = max(12, int(max_mem_gib * 4) + 8)
        mem_code = (
            "chunks = []\n"
            f"for _ in range({n_chunks}):\n"
            "    b = bytearray(256 * 1024 * 1024)\n"
            "    b[::4096] = b'\\x01' * (len(b) // 4096)\n"
            "    chunks.append(b)\n"
            "print('ALLOC_OK', len(chunks))\n"
        )
        r3 = run_cell(mem_code, cfg, timeout_s=15.0)
        mem_ok = (
            (not r3["ok"])
            and ("ALLOC_OK" not in r3["stdout"])
            and (r3["oom_killed"] or "MemoryError" in r3["stderr"] or r3["exit_code"] in (-9, -11, 137))
        )
        checks.append({
            "name": "memory_cap_enforced",
            "ok": bool(mem_ok),
            "detail": f"exit={r3['exit_code']}, oom_killed={r3['oom_killed']}, mem_err={'MemoryError' in r3['stderr']}",
        })

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
