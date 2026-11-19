#!/usr/bin/env python3
"""Stdio MCP server for the Phase 2 Agent's Door (`backend/level_mcp.py`), practice only.

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
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import level_session as lses  # noqa: E402

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
            "bounded claim sentence, and Claim Sheet only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "edit": {"type": "object"},
                "main_dose": {"type": "number", "default": 1.0},
                "forecast": {"type": "object"},
                "selection_mode": {"type": "string", "enum": ["free_pick", "declared_rule"], "default": "free_pick"},
                "sham_recipes": {"type": "array", "items": {"type": "object"}},
                "practice_best": {"type": "number"},
                "n_target": {"type": "integer", "minimum": 4, "default": 8},
                "n_neutral": {"type": "integer", "minimum": 1, "default": 4},
                "n_near": {"type": "integer", "minimum": 1, "default": 4},
                "n_panel": {"type": "integer", "minimum": 1, "default": 4},
                "controls_override": {"type": "integer", "minimum": 1},
                "match_tolerance": {"type": "number", "minimum": 1.0},
                "run_prompt_arms": {"type": "boolean", "default": True},
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
            practice_best=a.get("practice_best"),
            n_target=int(a.get("n_target", 8)),
            n_neutral=int(a.get("n_neutral", 4)),
            n_near=int(a.get("n_near", 4)),
            n_panel=int(a.get("n_panel", 4)),
            controls_override=a.get("controls_override"),
            match_tolerance=a.get("match_tolerance"),
            run_prompt_arms=bool(a.get("run_prompt_arms", True)),
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
    ap.add_argument("--stand-in", action="store_true", help="Use CPU stand-in runtime (no GPU).")
    ap.add_argument("--no-model", action="store_true", help="Off-card sandbox cells and book_view only.")
    ap.add_argument("--no-probe", action="store_true", help="Skip startup sandbox probe (for tests only).")
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

    session = lses.LevelSession(
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
    )
    return serve_stdio(session)


if __name__ == "__main__":
    sys.exit(main())
