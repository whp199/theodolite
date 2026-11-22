#!/usr/bin/env python3
"""Stdio MCP server for the Agent's Door (`backend/level_mcp.py`): practice, and with a brief and its calibrated
bars bound, the claim tool.

Exposes the small, driver-neutral Levelling contract over line-delimited JSON-RPC on stdin/stdout:
- `status`: harness card, declared reach, allowed tools, and remaining GPU-second / forward / cell / wall budgets.
- `cell`: off-card CPU Python analysis inside `bwrap` (`backend/level_sandbox.py`), omitted in `"tools_only"` reach.
- `capture`: gated activation-contrast direction capture (`levelling.capture_direction`), written to `/work`.
- `estimate`: dry cost estimate (forwards, GPU-seconds, memory) against remaining session budget.
- `read`: gated paired reading (`levelling.read`, kind `"paired"`), returning `level_book.driver_view` aggregates only.
- `write`: gated written reading (`levelling.read`, kind `"written"`, greedy or sampled/thinking-on with loop/cut-off
  summary), returning `level_book.driver_view` aggregates only.
- `book_view`: read redacted `level_book.driver_view` entries from the level book.

Every levelling tool is refused at the `"home"` reach (`$THEODOLITE_SCIENTIST_REACH=home`), and `"codex"` is refused
until its real sandbox has been probed and `torch.compile`'s cache moved out of `/tmp`. Closing stdin or terminating
the server closes the session and copies its transcript into `<book_dir>/transcripts/`.

The card. This server loads the model itself and keeps it for the session, so before it loads anything on the card
it takes the card's lock for the life of the process (`hold_card`) and refuses, in plain words, to start beside
another holder (`card_refusal`): the page server answering on its port, no CUDA device, or less of the card free than
the load needs. `--stand-in` and `--no-model` touch no card and take no lock.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def serve_socket_bridge(sock_path: Path) -> int:
    """Connect stdio JSON-RPC lines to a live parent `session_socket_bridge` Unix socket. It needs nothing past the
    standard library, so `level_mcp.py --socket <path>` runs it before the session's modules (and torch) are imported:
    a CLI driver's MCP child starts in a moment and small."""
    sp = Path(sock_path).resolve()
    cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        cli.connect(str(sp))
    except OSError as exc:
        raise SystemExit(f"Could not connect to MCP bridge socket {sp}: {exc}") from None
    try:
        rfile = cli.makefile("r", encoding="utf-8", errors="replace")
        wfile = cli.makefile("w", encoding="utf-8")
        with rfile, wfile:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                wfile.write(json.dumps(msg, ensure_ascii=True) + "\n")
                wfile.flush()
                if msg.get("id") is not None:
                    resp_line = rfile.readline()
                    if not resp_line:
                        break
                    sys.stdout.write(resp_line)
                    sys.stdout.flush()
    finally:
        cli.close()
    return 0


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--socket":
    sys.exit(serve_socket_bridge(Path(sys.argv[2])))

from backend import level_sandbox as ls  # noqa: E402
from backend import level_session as lses  # noqa: E402

_CARD_LOCK: Any = None  # the open file that holds the card's lock, kept until the process ends (`main`)
SERVER_PORT = 8765  # where the Theodolite server answers; it holds the model when it runs
MIN_FREE_GIB = 22.0  # what scripts/levelling.py asks of the card before it loads the model (the load peaks at 21.2)
POD_ROOT_ENV = "THEODOLITE_POD_ROOT"
POD_DEFAULT_ROOT = Path("/workspace")  # scripts/pod/common.sh DEFAULT_ROOT: where a pod's jobs lock when no root is set


def card_lock_path() -> Path:
    """The card's lock file. On a pod it is the one a census shard, the pilot and the self-test take
    (`<pod root>/logs/gpu.lock`, scripts/pod/common.sh `hold_card`), so a session and those jobs keep each other off
    the card. On the home machine the file sits beside the level book, outside every checkout. The page server takes
    this same lock (`hold_card`, backend/server.py) before it loads the model on a CUDA card, unless
    $THEODOLITE_DEVICE is cpu or $THEODOLITE_SKIP_CARD_LOCK is set, and keeps it while it runs: so a session and the
    page server keep each other off the card, and two sessions never load the model at once."""
    root = os.environ.get(POD_ROOT_ENV, "").strip()
    if root:
        return Path(root).expanduser() / "logs" / "gpu.lock"
    if (POD_DEFAULT_ROOT / "logs").is_dir():  # a pod whose scripts run with no root exported lock there
        return POD_DEFAULT_ROOT / "logs" / "gpu.lock"
    from backend import level_book as lb

    return lb.default_book_path().parent / "gpu.lock"


def hold_card(path: Optional[Path] = None) -> Any:
    """Take the card's lock without waiting and return the open file that holds it (keep it for the life of the
    process: the lock goes when the file is closed). `SystemExit`, in plain words, when another job holds it."""
    p = Path(path) if path is not None else card_lock_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        f = open(p, "a")  # never emptied: another job's lock file is left as it is
    except OSError as exc:
        raise SystemExit(f"The card's lock file {p} cannot be opened ({exc.strerror or exc}): the session is not "
                         "started without it.") from None
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit(f"Another GPU job holds this machine's card ({p}): one at a time.") from None
    return f


def _cuda_now() -> Tuple[bool, int]:
    import torch

    if not torch.cuda.is_available():
        return False, 0
    try:
        free, _total = torch.cuda.mem_get_info()
    except RuntimeError:  # a card too full to answer is a card with nothing free
        return True, 0
    return True, int(free)


def card_refusal(port: int = SERVER_PORT, *, cuda: Optional[Callable[[], Tuple[bool, int]]] = None) -> Optional[str]:
    """Why the model cannot be loaded for a session now, or None. Asked before anything is loaded: something answers
    on the page server's port (it holds the model), no CUDA device answers, or less of the card is free than the load
    needs (a census run, or anything else, is on it). `cuda` stands in for the card in a check."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", int(port))) == 0:
            return f"Something answers on :{port}: stop the Theodolite server first (this loads the model itself)."
    available, free = (cuda or _cuda_now)()
    if not available:
        return "No CUDA device answers in this process: a levelling session on the model needs the card (or --stand-in)."
    if free < MIN_FREE_GIB * 2 ** 30:
        return (f"Only {free / 2 ** 30:.1f} GiB of the card is free ({MIN_FREE_GIB:g} are asked: the load peaks at "
                f"21.2): stop the :{port} server and any census first.")
    return None


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "status": {
        "name": "status",
        "description": "Return the session's harness card, declared reach, allowed tools, and remaining budgets.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "cell": {
        "name": "cell",
        "description": (
            "Run Python code off-card inside an isolated CPU bwrap sandbox (no network, no GPU, read-only weights "
            "handle `W` when mounted, writable `/work` folder with `save_array(name, arr)` returning sha256)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code to execute inside the sandbox."},
                "timeout_s": {"type": "number", "description": "Optional wall-clock timeout cap in seconds."},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    "capture": {
        "name": "capture",
        "description": (
            "Capture a mean residual-stream difference `mean(items_a) - mean(items_b)` at `layer` (0-based) "
            "and save it as a float32 .npy array in `/work` with sha256 and quiet-mask share."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "items_a": {"type": "array", "items": {"type": "object"}},
                "items_b": {"type": "array", "items": {"type": "object"}},
                "layer": {"type": "integer", "minimum": 0},
                "where": {"type": "string", "enum": ["resid_post", "resid_pre"], "default": "resid_post"},
                "at": {"type": "string", "enum": ["end", "span_mean", "all_mean"], "default": "end"},
                "out_name": {"type": "string", "description": "Optional relative .npy filename inside /work."},
            },
            "required": ["items_a", "items_b", "layer"],
            "additionalProperties": False,
        },
    },
    "estimate": {
        "name": "estimate",
        "description": "Estimate forwards, GPU-seconds, memory, and budget fit for a reading plan without running it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "items": {"type": "object"},
            },
            "required": ["plan", "items"],
            "additionalProperties": False,
        },
    },
    "read": {
        "name": "read",
        "description": (
            "Run a gated paired reading at two batch sizes with null rows and control edits, enter it in the level "
            "book, and return `level_book.driver_view` aggregates and ranks (practice only: no word past a rank)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "items": {"type": "object"},
                "save_edit": {"type": "boolean", "default": False},
            },
            "required": ["plan", "items"],
            "additionalProperties": False,
        },
    },
    "write": {
        "name": "write",
        "description": (
            "Run a gated written reading (short greedy or long sampled / thinking-on with loop and cut-off "
            "diagnostics), enter it in the level book, and return `level_book.driver_view` aggregates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "items": {"type": "object"},
                "save_edit": {"type": "boolean", "default": False},
            },
            "required": ["plan", "items"],
            "additionalProperties": False,
        },
    },
    "book_view": {
        "name": "book_view",
        "description": "Read redacted `level_book.driver_view` entries from the level book.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seq": {"type": "integer", "minimum": 0},
                "tail": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "additionalProperties": False,
        },
    },
    "brief_info": {
        "name": "brief_info",
        "description": (
            "Return the public specification of the bound brief, allowed tensors/layers, remaining fresh-reading "
            "budget, and prior verdicts (never exposes kept-back frames)."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "draw_practice": {
        "name": "draw_practice",
        "description": (
            "Draw practice items from the bound brief's practice split and record their ids in the level book so "
            "future fresh draws refuse any overlap."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 64, "default": 6},
                "seed": {"type": "integer", "minimum": 0, "default": 0},
            },
            "additionalProperties": False,
        },
    },
    "claim": {
        "name": "claim",
        "description": (
            "Submit an edit for one registered Phase 3 fresh claim reading on the bound brief's kept-back frames. "
            "Consumes one of `m` fixed fresh readings, enforces read-once per edit, and returns the redacted verdict, "
            "bounded claim sentence, and Claim Sheet only. How the reading is taken (item counts, rows, control "
            "edits a family, matching tolerance) is fixed by the gate and the session's owner: see `claim_design` on "
            "the harness card. `sham_recipes` (for selection_mode declared_rule, with `declared_rule`) must each "
            "have the edit's shape; one that does not move the stream as the edit does counts against the edit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "edit": {"type": "object"},
                "main_dose": {"type": "number", "default": 1.0},
                "forecast": {"type": "object"},
                "selection_mode": {"type": "string", "enum": ["free_pick", "declared_rule"], "default": "free_pick"},
                "sham_recipes": {"type": "array", "items": {"type": "object"}},
                "declared_rule": {"type": "string", "description": (
                    "With selection_mode declared_rule: the rule the site was chosen by, in a sentence or two. "
                    "It is registered and printed with the sham edits.")},
                "practice_best": {"type": "number"},
            },
            "required": ["edit"],
            "additionalProperties": False,
        },
    },
    "atlas_query": {
        "name": "atlas_query",
        "description": (
            "Query the read-only atlas index (available at reach 'atlas' or when an atlas index is mounted): "
            "op in info, search_words, neuron_card, top_wiring, layer_stats."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["info", "search_words", "neuron_card", "top_wiring", "layer_stats"]},
                "args": {"type": "object"},
            },
            "required": ["op"],
            "additionalProperties": False,
        },
    },
    "knob_estimate": {
        "name": "knob_estimate",
        "description": "Estimate forwards and GPU-seconds for a cheaper-knob reading (prompts, decoding, steering; zero weight edits).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "knob": {"type": "object"},
                "items": {"type": "object"},
                "kind": {"type": "string", "enum": ["paired", "written"], "default": "paired"},
                "controls": {"type": "integer", "minimum": 0, "default": 3},
                "rows": {"type": "object"},
                "max_new": {"type": "integer", "minimum": 1, "default": 48},
                "samples": {"type": "integer", "minimum": 1, "default": 1},
            },
            "required": ["knob", "items"],
            "additionalProperties": False,
        },
    },
    "knob_read": {
        "name": "knob_read",
        "description": (
            "Run a gated paired practice reading of a cheaper knob (prompts, decoding, and/or transient steering; "
            "zero weight edits), returning aggregates and steering-control ranks only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "knob": {"type": "object"},
                "items": {"type": "object"},
                "readouts": {"type": "object"},
                "metric": {"type": "string", "default": "odds"},
                "direction": {"type": "integer", "default": 0},
                "controls": {"type": "integer", "minimum": 1},
                "rows": {"type": "object"},
                "seed": {"type": "integer", "default": 0},
                "name": {"type": "string", "default": "knob_practice"},
            },
            "required": ["knob", "items"],
            "additionalProperties": False,
        },
    },
    "knob_write": {
        "name": "knob_write",
        "description": "Run a gated written practice reading of a cheaper knob on practice items (zero weight edits).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "knob": {"type": "object"},
                "items": {"type": "object"},
                "max_new": {"type": "integer", "minimum": 1, "default": 48},
                "seeds": {"type": "array", "items": {"type": "integer"}},
                "controls": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer", "default": 0},
                "name": {"type": "string", "default": "knob_write_practice"},
            },
            "required": ["knob", "items"],
            "additionalProperties": False,
        },
    },
    "claim_knob": {
        "name": "claim_knob",
        "description": (
            "Submit a cheaper knob (prompts, decoding, and/or transient steering; zero weight edits) for one registered "
            "fresh claim reading on the bound brief's kept-back frames and calibrated bar set. A fresh claim needs a "
            "steering part: a prompt or decoding setting alone has no control edits to be judged against and is refused."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "knob": {"type": "object"},
                "forecast": {"type": "object"},
                "practice_best": {"type": "number"},
            },
            "required": ["knob"],
            "additionalProperties": False,
        },
    },
    "propose_repair": {
        "name": "propose_repair",
        "description": (
            "Submit one candidate repair edit recipe, or edit null to abstain explicitly, in a blinded diagnostic "
            "session for post-session scoring. Allowed once per session. `edit` must be given: a call without it is "
            "refused and spends nothing. The repair scored is the edit as built when it is submitted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "edit": {
                    "type": ["object", "null"],
                    "description": "Candidate edit recipe object, or null to abstain if no change is warranted.",
                },
                "diagnosis": {
                    "type": ["string", "null"],
                    "description": "Short plain-text summary of your diagnostic findings, or null.",
                },
                "confidence": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Optional confidence between 0 and 1.",
                },
            },
            "required": ["edit"],
            "additionalProperties": False,
        },
    },
}


def tools_for_session(session: lses.LevelSession) -> List[Dict[str, Any]]:
    """Return only the MCP tool schemas permitted by `session.allowed_tools()`."""
    return [TOOL_SPECS[name] for name in session.allowed_tools() if name in TOOL_SPECS]


def dispatch_tool(session: lses.LevelSession, name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Dispatch one MCP tool call against `session`, enforcing the session's filtered tool list. A `propose_repair`
    call the door refuses, for whatever reason (a missing or wrong argument, the edit gate, the wall-clock cap, a second
    submission), is kept in the door's own words (`LevelSession.note_refused_proposal`) before the refusal goes back."""
    if name != "propose_repair" or name not in session.allowed_tools():
        return _dispatch_tool(session, name, args)
    try:
        return _dispatch_tool(session, name, args)
    except Exception as exc:
        session.note_refused_proposal(str(exc))
        raise


def _dispatch_tool(session: lses.LevelSession, name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    if name not in session.allowed_tools():
        raise lses.SessionRefused(f'Tool "{name}" is not available in this session (allowed: {", ".join(session.allowed_tools())}).')
    a = dict(args or {})
    # Each tool's schema says additionalProperties: false, and the door holds to it: an argument the schema does not
    # name is refused, never passed on or dropped in silence (a driver has no switch the tool list does not show).
    known = set(TOOL_SPECS[name]["inputSchema"].get("properties") or {}) if name in TOOL_SPECS else set(a)
    extra = sorted(k for k in a if k not in known)
    if extra:
        raise lses.SessionRefused(
            f'Tool "{name}" takes no argument named {", ".join(extra)} (it takes: {", ".join(sorted(known)) or "none"}).'
        )
    # and an argument the schema requires is asked for in plain words, before anything is done or spent
    needed = list(TOOL_SPECS[name]["inputSchema"].get("required") or []) if name in TOOL_SPECS else []
    missing = [k for k in needed if k not in a]
    if missing:
        raise lses.SessionRefused(
            f'Tool "{name}" needs the argument{"s" if len(missing) > 1 else ""} {", ".join(missing)}: nothing was done.'
            + (" Give edit null to abstain." if name == "propose_repair" else "")
        )
    if name == "status":
        return session.status()
    if name == "cell":
        return session.cell(a["code"], timeout_s=a.get("timeout_s"))
    if name == "capture":
        return session.capture(
            a["items_a"], a["items_b"], int(a["layer"]),
            where=str(a.get("where", "resid_post")),
            at=str(a.get("at", "end")),
            out_name=a.get("out_name"),
        )
    if name == "estimate":
        return session.estimate(a["plan"], a["items"])
    if name == "read":
        return session.read(a["plan"], a["items"], save_edit=bool(a.get("save_edit", False)))
    if name == "write":
        return session.write(a["plan"], a["items"], save_edit=bool(a.get("save_edit", False)))
    if name == "book_view":
        return session.book_view(seq=a.get("seq"), tail=int(a.get("tail", 10)))
    if name == "brief_info":
        return session.brief_info()
    if name == "draw_practice":
        return session.draw_practice(n=int(a.get("n", 6)), seed=int(a.get("seed", 0)))
    if name == "claim":
        return session.claim(
            a["edit"],
            main_dose=float(a.get("main_dose", 1.0)),
            forecast=a.get("forecast"),
            selection_mode=str(a.get("selection_mode", "free_pick")),
            sham_recipes=a.get("sham_recipes"),
            declared_rule=a.get("declared_rule"),
            practice_best=a.get("practice_best"),
        )
    if name == "atlas_query":
        return session.atlas_query(str(a["op"]), a.get("args"))
    if name == "knob_estimate":
        return session.knob_estimate(
            a["knob"],
            a["items"],
            kind=str(a.get("kind", "paired")),
            controls=int(a.get("controls", 3)),
            rows=a.get("rows"),
            max_new=int(a.get("max_new", 48)),
            samples=int(a.get("samples", 1)),
        )
    if name == "knob_read":
        return session.knob_read(
            a["knob"],
            a["items"],
            readouts=a.get("readouts"),
            metric=str(a.get("metric", "odds")),
            direction=int(a.get("direction", 0)),
            controls=a.get("controls"),
            rows=a.get("rows"),
            seed=int(a.get("seed", 0)),
            name=str(a.get("name", "knob_practice")),
        )
    if name == "knob_write":
        return session.knob_write(
            a["knob"],
            a["items"],
            max_new=int(a.get("max_new", 48)),
            seeds=a.get("seeds") or (1,),
            controls=a.get("controls"),
            seed=int(a.get("seed", 0)),
            name=str(a.get("name", "knob_write_practice")),
        )
    if name == "claim_knob":
        return session.claim_knob(
            a["knob"],
            forecast=a.get("forecast"),
            practice_best=a.get("practice_best"),
        )
    if name == "propose_repair":
        # each argument is the type its schema says, and a null diagnosis stays null
        ed, dg, cf = a["edit"], a.get("diagnosis"), a.get("confidence")
        if ed is not None and not isinstance(ed, Mapping):
            raise lses.SessionRefused(f"edit must be an edit recipe object, or null to abstain; not {type(ed).__name__}.")
        if dg is not None and not isinstance(dg, str):
            raise lses.SessionRefused(f"diagnosis must be a string, or null; not {type(dg).__name__}.")
        if cf is not None and (isinstance(cf, bool) or not isinstance(cf, (int, float))):
            raise lses.SessionRefused(f"confidence must be a number between 0 and 1, or null; not {type(cf).__name__}.")
        return session.propose_repair(ed, diagnosis=dg, confidence=cf)
    raise lses.SessionRefused(f'Unknown tool "{name}".')


def handle_request(msg: Mapping[str, Any], session: lses.LevelSession) -> Optional[Dict[str, Any]]:
    """Handle one parsed JSON-RPC message; returns the response dict or None for notifications."""
    id_ = msg.get("id")
    if id_ is None:
        return None
    method = msg.get("method")
    if method == "initialize":
        proto = (msg.get("params") or {}).get("protocolVersion", "2025-06-18")
        return {"jsonrpc": "2.0", "id": id_, "result": {
            "protocolVersion": proto,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "theodolite-levelling", "version": str(lses.SESSION_VERSION)},
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": {"tools": tools_for_session(session)}}
    if method == "tools/call":
        p = msg.get("params") or {}
        ok, _out, text = _call_tool(session, str(p.get("name", "")), p.get("arguments") or {})
        return _tool_reply(id_, ok, text)
    if method == "ping":
        return {"jsonrpc": "2.0", "id": id_, "result": {}}
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": f"method not found: {method}"}}


def _call_tool(session: lses.LevelSession, name: str, args: Any) -> Tuple[bool, Any, str]:
    """One door call: `(True, its reply, the reply as JSON text)`, or `(False, None, "error: <why>")` for any refusal
    or failure (a reply that cannot be written as JSON among them)."""
    if not isinstance(args, Mapping):
        why = f'Tool "{name}" takes its arguments as a JSON object, not {type(args).__name__}.'
        if name == "propose_repair" and name in session.allowed_tools():
            session.note_refused_proposal(why)
        return False, None, f"error: {why}"
    try:
        out = dispatch_tool(session, name, args)
        return True, out, json.dumps(out, ensure_ascii=True)
    except Exception as exc:
        return False, None, f"error: {exc}"


def _tool_reply(id_: Any, ok: bool, text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if not ok:
        result["isError"] = True
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def serve_stdio(session: lses.LevelSession) -> int:
    """Serve JSON-RPC lines from stdin to stdout until EOF or signal, then close `session`."""
    stop = {"flag": False}

    def _on_sig(_signum: int, _frame: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)
    try:
        for line in sys.stdin:
            if stop["flag"]:
                break
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            resp = handle_request(msg, session)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=True) + "\n")
                sys.stdout.flush()
    finally:
        session.close()
    return 0


SOCKET_NAME = "door.sock"
AF_UNIX_PATH_MAX = 107  # bytes a socket's path may take on Linux: sun_path holds 108, the last for its closing NUL
CLOSING_WORDS = "the session is closing: the door takes no more calls."
ERROR_WORDS_MAX = 2000  # characters of a failed or refused call's own words kept in its record (a claim's refusal)


def _within(p: Path, root: Path) -> bool:
    return p == root or str(p).startswith(str(root) + os.sep)


def _private_socket_dir(work_dir: Path) -> Path:
    """A new mode-0700 folder for the bridge's socket, under `$XDG_RUNTIME_DIR` when it is set and usable, else the
    system's temporary folder, else /tmp: the first where the socket's path fits AF_UNIX and lies outside the work
    folder. A cell's sandbox mounts none of these (it has a /tmp of its own), so no cell can reach the socket."""
    work = Path(work_dir).resolve()
    bases: List[str] = []
    for base in (os.environ.get("XDG_RUNTIME_DIR", "").strip(), tempfile.gettempdir(), "/tmp"):
        if base and base not in bases:
            bases.append(base)
    for base in bases:
        try:
            d = Path(tempfile.mkdtemp(prefix="lvl-", dir=base)).resolve()  # mkdtemp makes it mode 0700
        except OSError:
            continue
        if len(os.fsencode(str(d / SOCKET_NAME))) <= AF_UNIX_PATH_MAX and not _within(d, work):
            os.chmod(d, 0o700)
            return d
        shutil.rmtree(d, ignore_errors=True)
    raise lses.SessionRefused(
        f"No private folder for the door's socket: under {', '.join(bases)} each was too long a path for a Unix "
        "socket, inside the session's work folder, or could not be made."
    )


def _call_facts(name: str, args: Mapping[str, Any], out: Any) -> Dict[str, Any]:
    """What a door call left behind, for a driver's outcome: the files a cell or a capture wrote, the book line a
    reading wrote, a claim's verdict and what it put forward."""
    if not isinstance(out, Mapping):
        return {}
    if name == "cell":
        return {"files_written": [str(f.get("rel_path")) for f in (out.get("files_written") or [])][:50],
                "removed": [str(r.get("rel_path")) for r in (out.get("removed") or [])][:50],
                "cell_ok": bool(out.get("ok")), "stopped": bool(out.get("stopped"))}
    if name == "capture":
        return {"files_written": [str(out.get("rel_path"))]}
    if name in ("read", "write"):
        return {"book_seq": (out.get("book") or {}).get("seq")}
    if name == "claim":
        return {"verdict": out.get("verdict"), "edit_id": out.get("edit_id"), "proposed": args.get("edit")}
    if name == "claim_knob":
        return {"verdict": out.get("verdict"), "knob_id": out.get("knob_id"), "proposed": args.get("knob")}
    if name == "propose_repair":
        return {"recorded": bool(out.get("recorded")), "abstained": bool(out.get("abstained")),
                "edit_id": out.get("edit_id"),
                "proposed": args.get("edit") if isinstance(args.get("edit"), Mapping) else None}
    return {}


class SocketBridge:
    """`session` served over a private AF_UNIX socket to a CLI driver's stdio MCP child (`level_mcp.py --socket
    <path>`), so the child reaches the model a study has already loaded.

    - The socket (`path`) sits in a new mode-0700 folder of its own (`_private_socket_dir`), so no other user can
      reach it whatever its own mode; it is made 0600 once bound (the process's umask is never changed: it is the
      whole process's, not this thread's), and goes, with its folder, on `close`.
    - One door call at a time, whatever connection it comes on, so each call's budget checks see every call before it.
    - `close` takes no more calls (one that comes after is answered with an error), kills a cell that is running
      (`level_sandbox.stop_cells`), and waits for any other call that is running until it ends, with no bound: a
      reading on the model always ends, and one cut off, or a book read before its lines land, would record a claim
      as no submission while its fresh reading was spent. So what the call puts in the book or the work folder is
      there before an outcome is read. `calls` lists every call that reached the door; `close` returns what was
      running at the stop, how it ended, and how long the door waited for it and why (`waited_seconds`,
      `wait_note`)."""

    def __init__(self, session: lses.LevelSession) -> None:
        self.session = session
        self.dir = _private_socket_dir(session.work_dir)
        self.path = self.dir / SOCKET_NAME
        self.calls: List[Dict[str, Any]] = []
        self.refused_after_stop = 0
        self._call_lock = threading.Lock()  # one door call at a time
        self._state_lock = threading.Lock()  # `_stopping` and `_active` change together under it
        self._stopping = False
        self._active: Optional[Dict[str, Any]] = None
        self._conns: List[socket.socket] = []
        self._report: Optional[Dict[str, Any]] = None
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(self.path))  # inside the 0700 folder: reachable by this user only from the moment it exists
            os.chmod(self.path, 0o600)
        except OSError:
            srv.close()
            shutil.rmtree(self.dir, ignore_errors=True)
            raise
        srv.listen(4)
        srv.settimeout(0.25)
        self._srv = srv
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                with self._state_lock:
                    if self._stopping:
                        return
                continue
            except OSError:
                return
            with self._state_lock:
                if self._stopping:
                    conn.close()
                    return
                self._conns.append(conn)
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            rfile = conn.makefile("r", encoding="utf-8", errors="replace")
            wfile = conn.makefile("w", encoding="utf-8")
            with rfile, wfile:
                for line in rfile:
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    resp = self._answer(msg)
                    if resp is not None:
                        wfile.write(json.dumps(resp, ensure_ascii=True) + "\n")
                        wfile.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _answer(self, msg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        id_ = msg.get("id")
        if id_ is None:
            return None  # a notification
        if msg.get("method") != "tools/call":
            with self._state_lock:
                closing = self._stopping
                if closing:
                    self.refused_after_stop += 1
            if closing:
                return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32000, "message": CLOSING_WORDS}}
            return handle_request(msg, self.session)
        p = msg.get("params") or {}
        name = str(p.get("name", ""))
        args = p.get("arguments") or {}
        with self._call_lock:
            with self._state_lock:
                if self._stopping:
                    self.refused_after_stop += 1
                    return _tool_reply(id_, False, f"error: {CLOSING_WORDS}")
                rec: Dict[str, Any] = {"n": len(self.calls) + 1, "tool": name, "ok": None, "seconds": None,
                                       "running_at_stop": False}
                self.calls.append(rec)
                self._active = rec
            t0 = time.monotonic()
            try:
                ok, out, text = _call_tool(self.session, name, args)
            finally:
                with self._state_lock:
                    self._active = None
            rec["seconds"] = round(time.monotonic() - t0, 3)
            rec["ok"] = ok
            if ok:
                rec.update(_call_facts(name, args if isinstance(args, Mapping) else {}, out))
            else:
                rec["error"] = text[:ERROR_WORDS_MAX]  # a refused claim's own words, for the driver's outcome
        return _tool_reply(id_, ok, text)

    def close(self) -> Dict[str, Any]:
        """Stop taking calls, kill a running cell, wait for any other running call to end, with no bound (see the
        class), then remove the socket and its folder. Returns what was running at the stop, how it ended, and how
        long the door waited for it and why; a second call returns the same."""
        if self._report is not None:
            return self._report
        t0 = time.monotonic()
        with self._state_lock:
            self._stopping = True
            running = self._active
            if running is not None:
                running["running_at_stop"] = True
        try:
            self._srv.close()
        except OSError:
            pass
        cells_stopped = ls.stop_cells(self.session.work_dir)
        with self._call_lock:  # the running call, if any, has ended: a cell killed, anything else let finish
            pass
        ls.release_cells(self.session.work_dir)
        waited = round(time.monotonic() - t0, 3)
        wait_note: Optional[str] = None
        if running is not None:
            tool = str(running.get("tool") or "?")
            if tool == "cell":
                how = "it was killed" if running.get("stopped") else "it ended before it could be killed"
                wait_note = f"A cell was running at the stop: {how}, and the door waited {waited:g} s for it to end."
            else:
                ended = "it answered" if running.get("ok") is True else "it failed"
                wait_note = (f"A {tool} call was running at the stop: the door waited {waited:g} s for it to end "
                             f"({ended}) before the outcome was read, as a reading is never cut off midway (only a "
                             "cell is killed), so what it wrote is in the book.")
        with self._state_lock:
            conns = list(self._conns)
        for c in conns:
            for fn in (lambda: c.shutdown(socket.SHUT_RDWR), c.close):
                try:
                    fn()
                except OSError:
                    pass
        self._accept_thread.join(timeout=2.0)
        shutil.rmtree(self.dir, ignore_errors=True)
        self._report = {
            "calls": len(self.calls),
            "ok": sum(1 for c in self.calls if c.get("ok") is True),
            "errors": sum(1 for c in self.calls if c.get("ok") is False),
            "refused_after_stop": int(self.refused_after_stop),
            "running_at_stop": dict(running) if running is not None else None,
            "cells_stopped": int(cells_stopped),
            "waited_seconds": waited,
            "wait_note": wait_note,
            "socket_removed": not self.dir.exists(),
        }
        return self._report


@contextlib.contextmanager
def session_socket_bridge(session: lses.LevelSession) -> Iterator[SocketBridge]:
    """Serve `session` to a CLI driver's MCP child over a private socket for the life of the block (`SocketBridge`;
    its `path` is the socket's). Leaving the block closes it: no more calls, a running cell killed, any other running
    call waited for until it ends."""
    bridge = SocketBridge(session)
    try:
        yield bridge
    finally:
        bridge.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Theodolite Levelling stdio MCP server (practice and claim gates).")
    ap.add_argument("--socket", type=Path, default=None,
                    help="Forward stdio JSON-RPC to an existing session_socket_bridge Unix socket (no model load).")
    ap.add_argument("--session-dir", type=Path, default=None, help="Per-session directory (holds work/, cells/, transcript).")
    ap.add_argument("--book", type=Path, default=None, help="Path to level_book.jsonl (must be outside work/).")
    ap.add_argument("--reach", default="sandbox", choices=list(lses.VALID_REACHES))
    ap.add_argument("--mode", default="weight_edit", choices=list(lses.VALID_MODES))
    ap.add_argument("--bare", action="store_true", help="Mark session as bare (minimal scaffolding) on harness card.")
    ap.add_argument("--backend", default="local", choices=list(lses.VALID_BACKENDS))
    ap.add_argument("--weights-dir", type=Path, default=None, help="Read-only safetensors folder mounted at /weights.")
    ap.add_argument("--atlas-dir", type=Path, default=None, help="Optional read-only atlas index folder mounted at /atlas.")
    ap.add_argument("--max-gpu-seconds", type=float, default=600.0)
    ap.add_argument("--max-forwards", type=int, default=200)
    ap.add_argument("--max-cells", type=int, default=100)
    ap.add_argument("--max-cell-seconds", type=float, default=30.0)
    ap.add_argument("--max-wall-seconds", type=float, default=1800.0)
    ap.add_argument("--max-cell-mem-gib", type=float, default=4.0)
    ap.add_argument("--brief", type=str, default=None, help="Optional brief JSON path or built-in starter brief name.")
    ap.add_argument("--bars", type=Path, default=None, help="Optional registered BarSet JSON path (enables claim tool).")
    ap.add_argument("--claim-n-target", type=int, default=None, help="Fresh target items a claim reads (the gate's own: 8).")
    ap.add_argument("--claim-n-neutral", type=int, default=None, help="Neutral battery items a claim reads (4).")
    ap.add_argument("--claim-n-near", type=int, default=None, help="Near-miss battery items a claim reads (4).")
    ap.add_argument("--claim-n-panel", type=int, default=None, help="Labelled panel items a claim reads (4).")
    ap.add_argument("--claim-controls", type=int, default=None,
                    help="Control edits a family for a claim (the brief's own 20m - 1 when not given).")
    ap.add_argument("--claim-match-tolerance", type=float, default=None,
                    help="Factor the matched other-site controls are matched within, on what a base pre-pass expects each to "
                         "deliver (the kernel's own: 2).")
    ap.add_argument("--stand-in", action="store_true", help="Use CPU stand-in runtime (no GPU).")
    ap.add_argument("--no-model", action="store_true", help="Off-card sandbox cells and book_view only.")
    ap.add_argument("--no-probe", action="store_true", help="Skip startup sandbox probe (for tests only).")
    ap.add_argument("--port", type=int, default=SERVER_PORT,
                    help="The port the Theodolite server would answer on; the session is refused while it does.")
    args = ap.parse_args(argv)

    if args.socket is not None:
        return serve_socket_bridge(args.socket)
    if args.session_dir is None:
        ap.error("--session-dir is required unless --socket is supplied.")

    h = None
    weights_dir = args.weights_dir
    if not args.no_model:
        from backend import levelling as lv

        if args.stand_in:
            import torch
            from scripts.atlas_stand_in import stand_in as make_stand_in, use_stand_in_atlas
            from backend.weight_reader import export_stand_in_safetensors

            torch.set_num_threads(1)
            use_stand_in_atlas()
            rt, _vault = make_stand_in()
            h = lv.handle(rt)
            if weights_dir is None and args.reach in ("sandbox", "weights", "atlas"):
                weights_dir = export_stand_in_safetensors(h.lm, args.session_dir / "_stand_in_weights", model_id=h.model_id)
        else:
            # the card is taken, and found free, before anything is loaded on it
            global _CARD_LOCK
            _CARD_LOCK = hold_card()
            why_not = card_refusal(int(args.port))
            if why_not:
                _CARD_LOCK.close()
                _CARD_LOCK = None
                raise SystemExit(why_not)

            from backend import w4a16_fast
            from backend.runtime_factory import checkpoint_snapshot, make_runtime

            rt = make_runtime(load_jlens=False).load()
            if hasattr(rt, "jpp_lenses"):
                rt.jpp_lenses.clear()
            w4a16_fast.install(rt, "compiled")
            h = lv.handle(rt)
            if weights_dir is None and args.reach in ("sandbox", "weights", "atlas"):
                weights_dir = checkpoint_snapshot(h.model_id.split("@", 1)[0])

    budget = lses.SessionBudget(
        max_gpu_seconds=args.max_gpu_seconds,
        max_forwards=args.max_forwards,
        max_cells=args.max_cells,
        max_cell_seconds=args.max_cell_seconds,
        max_wall_seconds=args.max_wall_seconds,
        max_cell_mem_gib=args.max_cell_mem_gib,
    )
    brief_obj = None
    bars_obj = None
    try:
        if args.brief:
            from backend import level_briefs as lbr
            p_br = Path(args.brief)
            if p_br.exists():
                brief_obj = lbr.load_brief(p_br)
            else:
                starters = lbr.starter_briefs(h.tokenizer if h is not None else None)
                if args.brief not in starters:
                    raise SystemExit(f"Unknown brief {args.brief!r}; pass a JSON file or one of: {', '.join(sorted(starters))}")
                brief_obj = starters[args.brief]
        if args.bars:
            from backend import level_claim as lcl
            bars_obj = lcl.load_bars(args.bars)
    except ValueError as exc:  # a brief or a bar set refused in plain words (BriefError, ClaimRefused)
        raise SystemExit(str(exc)) from None

    try:
        session = _make_session(h, args, budget, weights_dir, brief_obj, bars_obj)
    except lses.SessionRefused as exc:
        raise SystemExit(str(exc)) from None
    return serve_stdio(session)


def _make_session(h: Any, args: Any, budget: Any, weights_dir: Any, brief_obj: Any, bars_obj: Any) -> lses.LevelSession:
    return lses.LevelSession(
        h,
        args.session_dir,
        reach=args.reach,
        mode=args.mode,
        bare=bool(args.bare),
        backend=args.backend,
        budget=budget,
        book_path=args.book,
        weights_dir=weights_dir,
        atlas_dir=args.atlas_dir,
        probe_on_start=not args.no_probe,
        brief=brief_obj,
        bars=bars_obj,
        claim_design={k: v for k, v in {
            "n_target": args.claim_n_target, "n_neutral": args.claim_n_neutral, "n_near": args.claim_n_near,
            "n_panel": args.claim_n_panel, "controls_override": args.claim_controls,
            "match_tolerance": args.claim_match_tolerance,
        }.items() if v is not None} or None,
    )


if __name__ == "__main__":
    sys.exit(main())
