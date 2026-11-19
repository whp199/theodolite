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
import fcntl
import json
import os
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import level_session as lses  # noqa: E402

_CARD_LOCK: Any = None  # the open file that holds the card's lock, kept until the process ends (`main`)
SERVER_PORT = 8765  # where the Theodolite server answers; it holds the model when it runs
MIN_FREE_GIB = 22.0  # what scripts/levelling.py asks of the card before it loads the model (the load peaks at 21.2)
POD_ROOT_ENV = "THEODOLITE_POD_ROOT"
POD_DEFAULT_ROOT = Path("/workspace")  # scripts/pod/common.sh DEFAULT_ROOT: where a pod's jobs lock when no root is set


def card_lock_path() -> Path:
    """The card's lock file. On a pod it is the one a census shard, the pilot and the self-test take
    (`<pod root>/logs/gpu.lock`, scripts/pod/common.sh `hold_card`), so a session and those jobs keep each other off
    the card. On the home machine nothing else takes a lock yet (the page server does not): the file sits beside the
    level book, outside every checkout, and keeps two sessions from loading the model at once."""
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
}


def tools_for_session(session: lses.LevelSession) -> List[Dict[str, Any]]:
    """Return only the MCP tool schemas permitted by `session.allowed_tools()`."""
    return [TOOL_SPECS[name] for name in session.allowed_tools() if name in TOOL_SPECS]


def dispatch_tool(session: lses.LevelSession, name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Dispatch one MCP tool call against `session`, enforcing the session's filtered tool list."""
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
        try:
            out = dispatch_tool(session, str(p.get("name", "")), p.get("arguments") or {})
            text = json.dumps(out, ensure_ascii=True)
            return {"jsonrpc": "2.0", "id": id_, "result": {"content": [{"type": "text", "text": text}]}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": id_, "result": {
                "content": [{"type": "text", "text": f"error: {exc}"}], "isError": True,
            }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": id_, "result": {}}
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": f"method not found: {method}"}}


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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Theodolite Levelling Phase 2 stdio MCP server (practice only).")
    ap.add_argument("--session-dir", required=True, type=Path, help="Per-session directory (holds work/, cells/, transcript).")
    ap.add_argument("--book", type=Path, default=None, help="Path to level_book.jsonl (must be outside work/).")
    ap.add_argument("--reach", default="sandbox", choices=list(lses.VALID_REACHES))
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
                    help="Factor the matched other-site controls are matched within on delivered (the kernel's own: 2).")
    ap.add_argument("--stand-in", action="store_true", help="Use CPU stand-in runtime (no GPU).")
    ap.add_argument("--no-model", action="store_true", help="Off-card sandbox cells and book_view only.")
    ap.add_argument("--no-probe", action="store_true", help="Skip startup sandbox probe (for tests only).")
    ap.add_argument("--port", type=int, default=SERVER_PORT,
                    help="The port the Theodolite server would answer on; the session is refused while it does.")
    args = ap.parse_args(argv)

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
            if weights_dir is None and args.reach == "sandbox":
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
            if weights_dir is None and args.reach == "sandbox":
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
