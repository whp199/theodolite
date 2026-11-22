#!/usr/bin/env python3
"""Live check of the Antigravity CLI Levelling driver on the real `agy` binary (`scripts/verify_level_agy.py`).

Not one of the CPU checks: it runs the owner's installed agy 1.5.14, signed in with the owner's own Google login (from
the desktop keyring), and each run makes a few small model requests on that account. It loads no model: the driver
works a stand-in door, a `LevelSession` with no model behind it (its door serves `status`, `estimate` and
`book_view`), through the driver's exact recipe (`level_drivers.run_agy_cli_driver`). The driver is refused until
the owner accepts the three tools agy offers every print run (`level_drivers.AGY_FIXED_TOOLS_SAY`); this check first
shows that refusal, then passes the acceptance (`fixed_tools_accepted=True`) to see what a run would do.

The runs are priced at 0 dollars a million tokens, as on the owner's sign-in quota (the driver refuses a dollar cap
without prices, since agy reports no cost). Three runs:
1. A door run, told to call `status` once and stop. It checks that
   - the session ends as an abstention that counts, with a door call that succeeded, and its completion tokens are
     agy's own output count (which holds the thinking tokens);
   - the tools agy offered the model, read from the run's own conversation store, are exactly the door's and agy's
     three fixed ones: as the driver reads them (the tool declarations), and as this script reads them from another
     part of the store, the prompt's own token account (its "Tools" section);
   - no name of the owner's agy plugins or MCP servers, and no path into the owner's `~/.gemini`, appears in what
     agy sent the model or in its log, and the log shows the run's permissions (only `mcp(levelling/*)` allowed,
     request-review), no Auto mode, the update check skipped, and Remote Control off and disconnected;
   - every process the run started is agy itself or the door's MCP child, and none is left when it ends.
2. A turn-cap run, the same at `max_turns=1`: the tool the one allowed reply asked for runs, and the run is stopped
   as the next reply begins (`budget_turns`, counted), with no process left; that reply's request is counted
   (exactly when agy printed its usage, else at a floor of the last reply's prompt and output).
3. A STOP run: the kill switch is pulled once the door's MCP child is running. It checks that the outcome says
   `killed` and that no process of the run is left.
Before and after the runs, every file under `~/.gemini` (but the Antigravity desktop app's own folders) and
`~/.config/antigravity` is listed (path, size, time) and the two lists must be the same: nothing landed in the
owner's agy store. (A change made at the same time by the owner's own use of agy or Gemini CLI would show here too.)

Nothing is written under `~/.gemini`, `~/.config` or `~/.local/share/theodolite`: the session folders, books and the
report (`verify_level_agy.json`) go in `--out`. The agy log names the signed-in account; it is read for the facts
above and removed with the run, and no part of it is copied.

Run (the owner's call; it spends a little of the account's quota):
    XDG_DATA_HOME=/tmp/some/scratch/xdg .venv/bin/python -I scripts/verify_level_agy.py --out /tmp/some/scratch/agy_live
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import level_book as lb  # noqa: E402
from backend import level_drivers as ldrv  # noqa: E402
from backend import level_eval as lev  # noqa: E402
from backend import level_session as lses  # noqa: E402
from backend import weight_edits as we  # noqa: E402

PROMPT = ("This session is a plumbing check of the door, not an investigation. Call the status tool exactly once, "
          "then reply with the single word done and stop. Call no other tool.")
RESULTS: List[Dict[str, Any]] = []


def check(cond: bool, msg: str) -> None:
    RESULTS.append({"ok": bool(cond), "check": msg})
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)


# ------------------------------------------------------------------------- what the owner has (read, never written)

def owner_markers(gemini: Path) -> List[str]:
    """Names that would show the owner's own agy setup reached a run: the owner's plugin folders, and the MCP servers
    the owner's config and plugins name (keys only; no value is read out)."""
    names: Set[str] = set()
    plugins = gemini / "config" / "plugins"
    if plugins.is_dir():
        for p in plugins.iterdir():
            if p.is_dir() and len(p.name) >= 5:
                names.add(p.name)
                for f in list(p.glob("*.json")) + list(p.glob("*/*.json")):
                    try:
                        d = json.loads(f.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if isinstance(d, dict) and isinstance(d.get("mcpServers"), dict):
                        names.update(str(k) for k in d["mcpServers"])
    try:
        cfg = json.loads((gemini / "config" / "mcp_config.json").read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and isinstance(cfg.get("mcpServers"), dict):
            names.update(str(k) for k in cfg["mcpServers"])
    except (OSError, ValueError):
        pass
    names.discard(ldrv.DOOR_SERVER)
    return sorted(n for n in names if len(n) >= 5)


def listing(roots: List[Path]) -> Dict[str, List[int]]:
    """Every file and folder in or under `roots` (a config's node_modules left out): path -> [size, mtime_ns]."""
    out: Dict[str, List[int]] = {}
    for root in roots:
        try:
            st = os.lstat(root)
        except OSError:
            continue
        is_dir = os.path.isdir(root) and not os.path.islink(root)
        out[str(root)] = [-1 if is_dir else int(st.st_size), int(st.st_mtime_ns)]
        if not is_dir:
            continue
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "node_modules"]
            for n in dirnames + files:
                p = os.path.join(dirpath, n)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                out[p] = [int(st.st_size) if n in files else -1, int(st.st_mtime_ns)]
    return out


# ----------------------------------------------------------------------------- the run's processes, from /proc

def _proc(pid: int) -> Optional[Dict[str, Any]]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return None
    rest = stat.rsplit(")", 1)[-1].split()
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        env = []
    home = next((e[5:].decode("utf-8", "replace") for e in env if e.startswith(b"HOME=")), "")
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        exe = ""
    return {"pid": pid, "state": rest[0], "ppid": int(rest[1]), "home": home, "exe": exe,
            "cmd": [c.decode("utf-8", "replace") for c in cmd if c][:6]}


class Watcher:
    """Every process of a run: those whose HOME is a driver run's home (`theodolite_agy_*/home`), and their
    descendants, sampled every 0.1 s."""

    def __init__(self) -> None:
        self.prefix = os.path.join(tempfile.gettempdir(), "theodolite_agy_")
        self.seen: Dict[int, Dict[str, Any]] = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self.on_door_child: Optional[Any] = None

    def start(self) -> "Watcher":
        self._t.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            time.sleep(0.1)

    def sample(self) -> None:
        procs = {}
        for d in os.listdir("/proc"):
            if d.isdigit():
                info = _proc(int(d))
                if info is not None:
                    procs[info["pid"]] = info
        changed = True
        while changed:
            changed = False
            for pid, info in procs.items():
                if pid in self.seen:
                    continue
                if info["home"].startswith(self.prefix) or info["ppid"] in self.seen:
                    self.seen[pid] = info
                    changed = True
                    if self.on_door_child and any("level_mcp.py" in c for c in info["cmd"]):
                        self.on_door_child()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)

    def alive(self) -> List[Dict[str, Any]]:
        out = []
        for pid, info in self.seen.items():
            now = _proc(pid)
            if now is not None and now["state"] not in ("Z", "X") and now["cmd"] == info["cmd"]:
                out.append(now)
        return out


# ------------------------------------------------- the run's folder, read before the driver removes it

def _varint(b: bytes, i: int):
    s = r = 0
    while True:
        if i >= len(b) or s > 63:
            raise ValueError
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b: bytes):
    i, out = 0, []
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if f == 0:
            raise ValueError
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        elif wt == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            if len(v) != n:
                raise ValueError
            i += n
        else:
            raise ValueError
        out.append((f, wt, v))
    return out


def _account_tools(b: bytes, found: Set[str], depth: int = 0) -> None:
    """The tools named in the prompt's own token account: a message whose field 1 is "Tools" and whose field-5
    entries each name one tool in their field 1 (agy 1.5.14, run b4's store: 1/9/10/3/1). It reads a part of the
    store the driver does not (the driver reads the tool declarations, `level_drivers.agy_offered_tools`), and needs
    no schema, so the two readers do not share a blind spot."""
    try:
        fs = _fields(b)
    except ValueError:
        return
    if any(f == 1 and wt == 2 and v == b"Tools" for f, wt, v in fs):
        for f, wt, v in fs:
            if f == 5 and wt == 2:
                try:
                    sub = _fields(v)
                except ValueError:
                    continue
                name = next((sv for sf, sw, sv in sub if sf == 1 and sw == 2), None)
                if name and re.match(rb"^[A-Za-z_][A-Za-z0-9_\-]{0,127}$", name):
                    found.add(name.decode())
    if depth < 40:
        for f, wt, v in fs:
            if wt == 2 and len(v) > 2:
                _account_tools(v, found, depth + 1)


def read_run(run_root: Path, markers: List[str], owner_gemini: Path) -> Dict[str, Any]:
    """What the run's folder shows: the tools declared to the model, owner markers in what was sent or logged, and
    the log's account of the run's settings (as facts; the log itself is not kept)."""
    home = run_root / "home"
    conv = home / ".gemini" / "antigravity-cli" / "conversations"
    offered: Set[str] = set()
    blob = b""
    rows = 0
    for db in sorted(conv.glob("*.db")) if conv.is_dir() else []:
        con = sqlite3.connect(f"file:{urllib.parse.quote(str(db))}?mode=ro", uri=True)
        try:
            for (data,) in con.execute("select data from gen_metadata order by idx"):
                rows += 1
                b = bytes(data or b"")
                blob += b
                _account_tools(b, offered)
        finally:
            con.close()
    log = (run_root / "agy.log").read_text(encoding="utf-8", errors="replace") if (run_root / "agy.log").exists() else ""
    sent = blob.decode("utf-8", errors="replace")
    owner_path = str(owner_gemini)
    files = sorted(str(p.relative_to(home)) for p in home.rglob("*") if p.is_file())
    return {
        "gen_metadata_rows": rows,
        "offered": sorted(offered),
        "markers_in_sent": [m for m in markers if m in sent],
        "markers_in_log": [m for m in markers if m in log],
        "owner_path_in_sent": owner_path in sent,
        "owner_path_in_log": owner_path in log,
        "log_lines": log.count("\n"),
        "log_permissions": bool(re.search(r"CLI settings initialized: permissions=&\{Allow:\[mcp\(levelling/\*\)\] "
                                          r"Deny:\[command\(\*\) read_file\(\*\) write_file\(\*\) read_url\(\*\) "
                                          r"execute_url\(\*\) download_url\(\*\)\] Ask:\[\]\}, "
                                          r"toolPermission=request-review", log)),
        "log_auto_mode": "SetCycleMode called: auto" in log,
        "log_update_skipped": "skipping update (fast path)" in log,
        "log_keyring_sign_in": "authenticated via keyring" in log,
        "log_inherits_owner_settings": bool(re.search(r"applyUserSettings: inherited.*" + re.escape(owner_path), log)),
        "log_remote_control_off": ("[RemoteControl] RemoteControlEnabled value: false" in log
                                   and "Session toggle is off, staying disconnected" in log
                                   and not re.search(r"RemoteControlEnabled value: true|Session toggle (is on|set to "
                                                     r"true)", log)),
        "home_files": len(files),
        "home_file_sample": files[:40],
    }


# ------------------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Live check of the Antigravity CLI Levelling driver (owner-run).")
    ap.add_argument("--out", type=Path, default=None, help="Folder for the sessions, books and the report (new).")
    ap.add_argument("--only", choices=("door", "turns", "stop"), default=None, help="Run one of the three runs.")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="verify_level_agy_"))
    out.mkdir(parents=True, exist_ok=True)
    os.environ[lb.BOOK_ENV] = str(out / "default_book.jsonl")
    os.environ[we.EDITS_ENV] = str(out / "edits")
    owner_gemini = Path.home() / ".gemini"
    # the owner's ~/.gemini but the Antigravity desktop app's own folders (it may be running), and agy's config folder
    desktop = {"antigravity", "antigravity-backup", "antigravity-browser-profile", "antigravity-ide"}
    watched_roots = sorted(p for p in owner_gemini.iterdir() if p.name not in desktop) if owner_gemini.is_dir() else []
    watched_roots.append(Path.home() / ".config" / "antigravity")
    markers = owner_markers(owner_gemini)
    report: Dict[str, Any] = {"started_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "agy_version_wanted": ldrv.AGY_VERSION, "owner_markers": markers}
    print(f"verify_level_agy: {len(markers)} owner plugin and MCP server names to look for; out={out}", flush=True)
    before = listing(watched_roots)
    # priced at 0, as on the owner's sign-in quota: the driver refuses a dollar cap without prices
    budget = ldrv.DriverBudget(max_turns=6, max_tokens=150_000, max_dollars=0.25, max_wall_seconds=240.0,
                               input_usd_per_mtok=0.0, output_usd_per_mtok=0.0)
    agy_real = os.path.realpath(shutil.which(os.environ.get("THEODOLITE_AGY_BIN", "agy")) or "agy")

    s0 = lses.LevelSession(None, out / "refused_session", reach="behaviour", backend="agy",
                           book_path=out / "refused_book.jsonl", probe_on_start=False)
    runs_before = sorted(Path(tempfile.gettempdir()).glob("theodolite_agy_*"))
    try:
        ldrv.run_agy_cli_driver(s0, driver_budget=budget, extra_instructions=PROMPT)
        refused = ""
    except ldrv.DriverRefused as exc:
        refused = str(exc)
    s0.close()
    check("list_resources, read_resource and manage_task" in refused
          and sorted(Path(tempfile.gettempdir()).glob("theodolite_agy_*")) == runs_before,
          "0: without the owner's acceptance of agy's three fixed tools the driver is refused before a run folder is "
          "made or agy is started")

    if args.only in (None, "door"):
        print("\n=== 1. A door run on the real agy ===", flush=True)
        seen_run: Dict[str, Any] = {}
        w = Watcher().start()
        s = lses.LevelSession(None, out / "door_session", reach="behaviour", backend="agy",
                              book_path=out / "door_book.jsonl", probe_on_start=False)
        t0 = time.monotonic()
        o = ldrv.run_agy_cli_driver(s, driver_budget=budget, extra_instructions=PROMPT, fixed_tools_accepted=True,
                                    inspect_run=lambda root: seen_run.update(read_run(root, markers, owner_gemini)))
        dt = time.monotonic() - t0
        time.sleep(1.0)
        w.sample()
        w.stop()
        left = w.alive()
        procs = sorted({(os.path.basename(p["exe"]) or "?", " ".join(p["cmd"][1:3])[:80]) for p in w.seen.values()})
        door = o.get("door") or {}
        report["door_run"] = {"seconds": round(dt, 1), "stop_reason": o.get("stop_reason"),
                              "driver_failure": o.get("driver_failure"), "why_not_counted": lev.why_not_counted(o),
                              "door": {k: door.get(k) for k in ("calls", "ok", "errors", "offered_tools",
                                                                "offered_expected", "other_tools_asked", "denied",
                                                                "agy_version", "init_ok", "run_files_removed")},
                              "usage": {k: (o.get("usage") or {}).get(k) for k in ("turns", "prompt_tokens",
                                                                                   "completion_tokens",
                                                                                   "reasoning_tokens", "agy_usage")},
                              "notes": str(o.get("notes") or "")[:600], "run": seen_run,
                              "processes": [list(p) for p in procs], "left": [p["cmd"][:2] for p in left]}
        usage = o.get("usage") or {}
        agy_u = usage.get("agy_usage") or {}
        check(o.get("stop_reason") == "kept_base_abstention" and o.get("driver_failure") is None
              and int(door.get("ok") or 0) >= 1 and lev.why_not_counted(o) is None,
              f"1: the door run on agy {door.get('agy_version')} ends as an abstention that counts, with "
              f"{door.get('ok')} door call(s) that succeeded ({o.get('stop_reason')}; {o.get('driver_failure')!r})")
        check(usage.get("completion_tokens") == agy_u.get("output_tokens") and int(agy_u.get("output_tokens") or 0) > 0
              and usage.get("reasoning_tokens") == agy_u.get("thinking_tokens")
              and usage.get("prompt_tokens") == int(agy_u.get("input_tokens") or 0) + int(agy_u.get("cache_read_tokens") or 0)
              and usage.get("estimated_dollars") == 0.0,
              f"1: the run's completion tokens are agy's own output count, which holds the thinking tokens (not the two "
              f"added), its prompt tokens are input and cache reads, and at prices of 0 its dollars are 0 "
              f"({usage.get('prompt_tokens')} + {usage.get('completion_tokens')}; agy: {agy_u})")
        check(door.get("offered_tools") == door.get("offered_expected") and bool(door.get("offered_tools"))
              and seen_run.get("offered") == door.get("offered_expected"),
              f"1: the tools agy offered the model are exactly the door's and agy's three fixed ones, as the driver "
              f"reads them (the declarations) and as this script reads them from the prompt's token account "
              f"({seen_run.get('offered')})")
        check(not seen_run.get("markers_in_sent") and not seen_run.get("markers_in_log")
              and seen_run.get("owner_path_in_sent") is False and seen_run.get("owner_path_in_log") is False
              and seen_run.get("log_inherits_owner_settings") is False,
              f"1: none of the owner's {len(markers)} agy plugin and MCP server names, and no path into the owner's "
              f"~/.gemini, is in what agy sent the model or in its log (found: {seen_run.get('markers_in_sent')}, "
              f"{seen_run.get('markers_in_log')})")
        check(seen_run.get("log_permissions") is True and seen_run.get("log_auto_mode") is False
              and seen_run.get("log_update_skipped") is True and seen_run.get("log_remote_control_off") is True,
              "1: agy's log shows the run's own permissions (only mcp(levelling/*) allowed, request-review), no Auto "
              "mode, the update check skipped, and Remote Control off and disconnected (no browser can watch or steer "
              "the run)")
        others = [p for p in w.seen.values()
                  if os.path.realpath(p["exe"]) != agy_real and not any("level_mcp.py" in c for c in p["cmd"])]
        check(bool(w.seen) and not others and not left and door.get("run_files_removed") is True,
              f"1: every process the run started is agy or the door's MCP child ({procs}), none is left, and the "
              f"run's folder is removed (others: {[p['cmd'][:2] for p in others]}; left: {[p['cmd'][:2] for p in left]})")

    if args.only in (None, "turns"):
        print("\n=== 2. The turn cap on the real agy ===", flush=True)
        w3 = Watcher().start()
        s3 = lses.LevelSession(None, out / "turns_session", reach="behaviour", backend="agy",
                               book_path=out / "turns_book.jsonl", probe_on_start=False)
        t3 = time.monotonic()
        o3 = ldrv.run_agy_cli_driver(s3, driver_budget=dataclasses.replace(budget, max_turns=1),
                                     extra_instructions=PROMPT, fixed_tools_accepted=True)
        dt3 = time.monotonic() - t3
        time.sleep(1.0)
        w3.sample()
        w3.stop()
        left3 = w3.alive()
        door3 = o3.get("door") or {}
        report["turns_run"] = {"seconds": round(dt3, 1), "stop_reason": o3.get("stop_reason"),
                               "driver_failure": o3.get("driver_failure"), "why_not_counted": lev.why_not_counted(o3),
                               "door": {k: door3.get(k) for k in ("calls", "ok", "errors", "offered_tools",
                                                                  "run_files_removed")},
                               "usage": {k: (o3.get("usage") or {}).get(k) for k in ("turns", "prompt_tokens",
                                                                                    "completion_tokens", "agy_usage",
                                                                                    "cut_reply_prompt_floor")},
                               "notes": str(o3.get("notes") or "")[:600], "left": [p["cmd"][:2] for p in left3]}
        u3 = o3.get("usage") or {}
        a3 = u3.get("agy_usage") or {}
        floor3 = int(u3.get("cut_reply_prompt_floor") or 0)
        check(o3.get("stop_reason") == "budget_turns" and o3.get("driver_failure") is None
              and int(door3.get("ok") or 0) >= 1 and u3.get("turns") == 1
              and lev.why_not_counted(o3) is None and not left3 and door3.get("run_files_removed") is True,
              f"2: at max_turns=1 the tool the one allowed reply asked for runs ({door3.get('ok')} door call(s) that "
              f"succeeded), the run is stopped as the next reply begins ({o3.get('stop_reason')}, "
              f"{u3.get('turns')} turn), it counts, and no process is left ({[p['cmd'][:2] for p in left3]})")
        check(u3.get("prompt_tokens") == int(a3.get("input_tokens") or 0) + int(a3.get("cache_read_tokens") or 0) + floor3
              and ("floor of" not in str(o3.get("notes") or "") or floor3 > 0),
              f"2: the cut reply's request is counted: exactly when agy printed its usage, else at a floor of the last "
              f"reply's prompt and output ({u3.get('prompt_tokens')} prompt tokens, floor {floor3})")

    if args.only in (None, "stop"):
        print("\n=== 3. STOP on the real agy ===", flush=True)
        ks = ldrv.KillSwitch()
        fired: Dict[str, float] = {}

        def _pull() -> None:
            if "at" not in fired:
                fired["at"] = time.monotonic()
                threading.Timer(0.3, ks.trigger, args=("verify: STOP once the door's MCP child was running",)).start()

        w2 = Watcher()
        w2.on_door_child = _pull
        w2.start()
        fallback = threading.Timer(20.0, _pull)
        fallback.start()
        s2 = lses.LevelSession(None, out / "stop_session", reach="behaviour", backend="agy",
                               book_path=out / "stop_book.jsonl", probe_on_start=False)
        t1 = time.monotonic()
        o2 = ldrv.run_agy_cli_driver(s2, driver_budget=budget, kill_switch=ks, extra_instructions=PROMPT,
                                     fixed_tools_accepted=True)
        dt2 = time.monotonic() - t1
        fallback.cancel()
        time.sleep(3.0)
        w2.sample()
        w2.stop()
        left2 = w2.alive()
        report["stop_run"] = {"seconds": round(dt2, 1), "stop_reason": o2.get("stop_reason"),
                              "driver_failure": o2.get("driver_failure"), "why_not_counted": lev.why_not_counted(o2),
                              "door": {k: (o2.get("door") or {}).get(k) for k in ("calls", "ok", "run_files_removed")},
                              "processes_seen": len(w2.seen), "left": [p["cmd"][:2] for p in left2],
                              "notes": str(o2.get("notes") or "")[:400]}
        check(o2.get("stop_reason") == "killed" and "kill switch" in str(lev.why_not_counted(o2))
              and len(w2.seen) >= 2 and not left2 and (o2.get("door") or {}).get("run_files_removed") is True,
              f"3: STOP on the real agy ends the session as killed, not counted; of the {len(w2.seen)} processes the "
              f"run started none is left 3 s after ({[p['cmd'][:2] for p in left2]}), and its folder is removed")

    after = listing(watched_roots)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    report["owner_store_changes"] = changed[:50]
    check(not changed and bool(before),
          f"nothing was added to or changed in the owner's ~/.gemini (but the desktop app's folders) or "
          f"~/.config/antigravity ({len(before)} entries listed before and after; changed: {changed[:5]})")

    passed = sum(1 for r in RESULTS if r["ok"])
    report["results"] = RESULTS
    report["passed"], report["failed"] = passed, len(RESULTS) - passed
    (out / "verify_level_agy.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nverify_level_agy: {passed} passed, {len(RESULTS) - passed} failed; report in {out / 'verify_level_agy.json'}",
          flush=True)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
