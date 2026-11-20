"""Levelling Phase 6: driver-neutral evaluation runners and baselines (`backend/level_drivers.py`).

Owner-side driver runner for Phase 6 ("Whether agents help"): drives any supported driver family against a
`LevelSession` through the exact same gated tool contract (`backend/level_mcp.py`), recording a uniform Harness Card,
usage counters (GPU-seconds, forwards, cells, tokens, dollars, wall-clock), and outcome from the session's level book.

Never imported by `level_mcp.py`, `level_session.py`, `level_routes.py`, or `scientist_tools.py`.

Supported driver families:
- `scripted`: deterministic no-driver baselines (`ScriptedWeightDriver`, `ScriptedKnobDriver`) that adapt honestly to
  each declared reach (`behaviour`, `activations`, `weights`, `atlas`) and spend real session budget through the gates.
- `replay`: replays a fixed or recorded sequence of tool calls through `level_mcp.dispatch_tool` (zero API cost).
- `openai`: OpenAI-compatible Chat Completions tool-calling loop reusing `backend/scientist_chat.py`'s endpoint
  profiles (API keys read strictly from environment variables, never stored on disk; injectable transport for tests).
- `claude`: Claude Code CLI (`claude -p --output-format stream-json`) confined to the folder reach (`cwd=work_dir`,
  `--setting-sources ""`, `--strict-mcp-config`, `--tools Read Write Edit`, `--disallowedTools Bash`, credential-stripped
  `child_env`), connected to the live `LevelSession` via `level_mcp.session_socket_bridge` so multi-session studies do
  not reload the 31B checkpoint per run.
- `codex`: refused unless `codex_probed=True` and `TORCHINDUCTOR_CACHE_DIR` is outside `/tmp`.

Hard unattended caps (`verify/arbiter.md` row 34, `verify/skeptic_safety.md:271`):
- `DriverBudget`: `max_turns`, `max_tokens`, `max_dollars` (owner-approved ceiling required for paid drivers),
  `max_wall_seconds` with process-group `SIGKILL`.
- `KillSwitch`: file-based (`STOP` file) and event/signal kill switch checked before every turn and subprocess read.

Zero kept-out words appear in source, docstrings, prompts, or outputs.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from backend import atlas_ops as ao
from backend import level_book as lb
from backend import level_briefs as lbr
from backend import level_mcp as lmcp
from backend import level_session as lses
from backend import levelling as lv
from backend import weight_edits as we

ROOT_DIR = Path(__file__).resolve().parent.parent
DRIVER_VERSION = 1
DRIVER_FAMILIES = ("scripted", "replay", "openai", "claude", "codex")
PAID_DRIVER_FAMILIES = ("claude", "codex")


class DriverRefused(ValueError):
    """A driver launch refusal (reach, unprobed codex, missing dollar ceiling, bad profile) in plain words."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(ROOT_DIR),
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
        return out.decode("utf-8", errors="replace").strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class DriverBudget:
    """Hard driver-side caps paired with `level_session.SessionBudget`."""

    max_turns: int = 24
    max_tokens: int = 200_000
    max_dollars: Optional[float] = None
    max_wall_seconds: float = 1200.0
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_turns": int(self.max_turns),
            "max_tokens": int(self.max_tokens),
            "max_dollars": float(self.max_dollars) if self.max_dollars is not None else None,
            "max_wall_seconds": float(self.max_wall_seconds),
            "input_usd_per_mtok": float(self.input_usd_per_mtok),
            "output_usd_per_mtok": float(self.output_usd_per_mtok),
        }


class KillSwitch:
    """Cooperative + OS-signal + stop-file kill switch for unattended single and multi-session runs."""

    def __init__(self, stop_file: Optional[Path] = None) -> None:
        self.stop_file = Path(stop_file) if stop_file is not None else None
        self._event = threading.Event()
        self._reason: Optional[str] = None

    def trigger(self, reason: str = "manual kill") -> None:
        self._reason = str(reason)
        self._event.set()

    def triggered(self) -> Optional[str]:
        if self._event.is_set():
            return self._reason or "killed"
        if self.stop_file is not None and self.stop_file.exists():
            self._reason = f"stop file present ({self.stop_file.name})"
            self._event.set()
            return self._reason
        return None

    @contextlib.contextmanager
    def install_signals(self) -> Iterator["KillSwitch"]:
        """Catch SIGTERM/SIGINT on the main thread and trigger the kill switch cleanly."""
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        prev_term = signal.getsignal(signal.SIGTERM)
        prev_int = signal.getsignal(signal.SIGINT)

        def _handler(signum: int, _frame: Any) -> None:
            self.trigger(f"received signal {signum}")

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
        try:
            yield self
        finally:
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)


def build_driver_prompt(
    session: lses.LevelSession,
    *,
    bare: Optional[bool] = None,
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the bounded, one-page driver prompt (standard or `--bare`), naming NO suspected site or layer.

    Every non-trust-boundary convenience omitted in `bare=True` is listed in `docs/SCAFFOLD.md`."""
    is_bare = session.bare if bare is None else bool(bare)
    card = session.harness_card()
    tools = list(card["allowed_tools"])
    brief_card = card.get("brief") or {}
    bud = session.budget.to_dict()
    mode = card.get("mode", "weight_edit")
    reach = card.get("reach", "sandbox")

    if is_bare:
        sys_lines = [
            f"You are driving a Theodolite Levelling session (mode={mode}, reach={reach}).",
            f"Available tools: {', '.join(tools)}.",
            (
                f"Budgets: max_gpu_seconds={bud['max_gpu_seconds']:g}, max_forwards={bud['max_forwards']}, "
                f"max_cells={bud['max_cells']}, max_wall_seconds={bud['max_wall_seconds']:g}."
            ),
            "Keeping the untouched base model (submitting nothing when no candidate works) is a valid outcome.",
        ]
    else:
        sys_lines = [
            "You are investigating a target behaviour on a Gemma 4 language model through the Theodolite Levelling door.",
            f"Declared reach: {reach}. Mode: {mode}.",
            f"Available tools: {', '.join(tools)}.",
            "",
            "Rules and physical facts:",
            "- Never rely on priors or literature; only measurements from your tool calls count.",
            "- Practice tools return aggregate summaries and ranks against control edits, never a verdict.",
            "- Kept-back frames and retention batteries are refused in practice; use draw_practice or write your own items.",
            "- Keeping the base model (stopping without a claim if nothing clears the noise floor and controls) is an honest ending.",
            (
                f"- Hard budgets: {bud['max_gpu_seconds']:g} GPU-seconds, {bud['max_forwards']} practice forwards, "
                f"{bud['max_cells']} cells, {bud['max_wall_seconds']:g} wall seconds."
            ),
        ]
        if mode == "knobs_only":
            sys_lines += [
                "",
                "Cheaper-knob mode (zero weight edits):",
                "- Weight-edit tools are disabled. Search only over prompts, decoding (thinking, sampler, logit_bias),",
                "  and transient activation steering pushes (direction, neuron_mul, neuron_add) via knob_read / knob_write.",
                "- Remember that a global logit_bias also moves near-miss prompts and can fail retention batteries.",
                "- When ready, submit your best candidate once via claim_knob (scored on fresh kept-back frames and harm bars).",
            ]
        else:
            sys_lines += [
                "",
                "Weight-edit mode:",
                "- Readings compare your edit against untouched base rows, null-row floors, and control edits in two batch classes.",
            ]
            if "cell" in tools:
                if card.get("atlas_mounted"):
                    sys_lines.append("- Off-card Python cell has read-only weights handle W, read-only atlas handle A, and writable /work.")
                elif card.get("with_healthy_reference"):
                    sys_lines.append("- Off-card Python cell has read-only weights handle W and writable /work (no GPU, no network).")
                else:
                    sys_lines.append("- Off-card Python cell has writable /work with numpy/torch on CPU (no weights, no GPU, no network).")
            if "atlas_query" in tools:
                sys_lines.append("- atlas_query searches read-only neuron labels and wiring; labels marked trusted=False are leads only.")
            if "claim" in tools:
                sys_lines.append("- When a candidate clears practice controls and floor, submit it once via claim for a fresh verdict.")

    if extra_instructions:
        sys_lines += ["", str(extra_instructions).strip()]
    system_text = "\n".join(sys_lines).strip()

    if brief_card:
        user_lines = [
            f"Brief: {brief_card.get('name')}",
            f"Goal: {brief_card.get('driver_text')}",
            f"Metric: {brief_card.get('metric')} (direction={brief_card.get('direction')}, least_effect={brief_card.get('least_effect')}).",
            f"Allowed tensors: {', '.join(brief_card.get('allowed_tensors') or ())}.",
            f"Fresh readings available in this session: {brief_card.get('max_fresh_readings')}.",
            "Call status and brief_info first if needed, explore on practice items, and either submit one candidate or stop and keep the base.",
        ]
    else:
        user_lines = [
            "No claim brief is bound (practice/blinded session). Call status first, diagnose any anomaly or test your hypothesis on practice items, and summarise your findings.",
        ]
    user_text = "\n".join(user_lines).strip()
    digest = hashlib.sha256((system_text + "\n---\n" + user_text).encode("utf-8")).hexdigest()
    words = len((system_text + " " + user_text).split())
    return {
        "system": system_text,
        "user": user_text,
        "prompt_sha256": digest,
        "word_count": words,
        "bare": is_bare,
        "prompt_named_site": False,
    }


def _extract_outcome_from_book(
    session: lses.LevelSession,
    *,
    driver_family: str,
    driver_model: str,
    face: str,
    reasoning_effort: Optional[str],
    prompt_info: Mapping[str, Any],
    driver_budget: DriverBudget,
    stop_reason: str,
    turns: int,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    estimated_dollars: float,
    tool_errors: int,
    notes: str = "",
    proposed_edit: Optional[Mapping[str, Any]] = None,
    proposed_knob: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical `SessionOutcome` from `session` and its verified level book (never trusting driver prose)."""
    close_info = session.close()
    ver = lb.verify(session.book_path)
    book_ok = bool(ver["ok"])
    card = session.harness_card()

    # Scan the session's book lines for the final claim or knob_claim verdict
    verdict = "no_submission"
    submitted = False
    directed_effect: Optional[float] = None
    small_dir: Optional[float] = None
    large_dir: Optional[float] = None
    near_miss_abs: Optional[float] = None
    neutral_kl: Optional[float] = None
    panel_kl: Optional[float] = None
    panel_own_drop: Optional[float] = None
    replicated: Optional[bool] = None
    cheaper_knob_matched: Optional[bool] = None
    candidate_id: Optional[str] = None
    knob_classes: List[str] = []
    verdict_seq: Optional[int] = None
    all_claims: List[Dict[str, Any]] = []

    if book_ok and session.book_path.exists():
        for entry in lb.entries(session.book_path):
            if entry.get("kind") != "note":
                continue
            body = entry.get("body") or {}
            nt = body.get("note_type") or body.get("note")
            if nt == "claim_verdict":
                submitted = True
                verdict_seq = int(entry["seq"])
                verdict = str(body.get("verdict") or "void")
                candidate_id = body.get("edit_id")
                summ_map = body.get("summary") or {}
                sm_c = summ_map.get("small") or {}
                lg_c = summ_map.get("large") or {}
                small_dir = sm_c.get("directed_mean")
                large_dir = lg_c.get("directed_mean")
                directed_effect = large_dir if large_dir is not None else small_dir
                bats = body.get("batteries") or {}
                near_miss_abs = (bats.get("near_misses") or {}).get("value")
                neutral_kl = (bats.get("neutral_text") or {}).get("value")
                pan = bats.get("labelled_panel") or {}
                panel_kl = pan.get("kl")
                panel_own_drop = pan.get("own_drop")
                h0_ok = ((lg_c or sm_c).get("half_0") or {}).get("directed_mean")
                h1_ok = ((lg_c or sm_c).get("half_1") or {}).get("directed_mean")
                replicated = bool(
                    verdict not in ("did not replicate", "void", "kept the base")
                    or (isinstance(h0_ok, (int, float)) and isinstance(h1_ok, (int, float)) and h0_ok > 0 and h1_ok > 0)
                )
                ck = body.get("cheaper_knobs") or {}
                cheaper_knob_matched = bool(ck.get("matched"))
                all_claims.append({"seq": verdict_seq, "kind": "weight_edit", "verdict": verdict, "id": candidate_id})
            elif nt == "knob_claim_verdict":
                submitted = True
                verdict_seq = int(entry["seq"])
                verdict = str(body.get("verdict") or "void")
                candidate_id = body.get("knob_id")
                knob_classes = list(body.get("knob_classes") or [])
                summ = body.get("summary") or {}
                small_dir = summ.get("small_directed")
                large_dir = summ.get("large_directed")
                directed_effect = large_dir if large_dir is not None else small_dir
                bats = summ.get("batteries") or {}
                near_miss_abs = bats.get("near_miss_abs")
                neutral_kl = bats.get("neutral_kl")
                panel_kl = bats.get("panel_kl")
                panel_own_drop = bats.get("panel_own_drop")
                replicated = bool(summ.get("replicated"))
                cheaper_knob_matched = None
                all_claims.append({"seq": verdict_seq, "kind": "knobs_only", "verdict": verdict, "id": candidate_id})

    if submitted and stop_reason == "kept_base_abstention":
        stop_reason = "claimed"

    h = session._handle
    harness_card = {
        "v": DRIVER_VERSION,
        "driver_family": driver_family,
        "driver_model": driver_model,
        "reasoning_effort": reasoning_effort,
        "face": face,
        "reach": card["reach"],
        "mode": card["mode"],
        "bare": bool(card["bare"]),
        "with_healthy_reference": bool(card["with_healthy_reference"]),
        "atlas_mounted": bool(card["atlas_mounted"]),
        "allowed_tools": list(card["allowed_tools"]),
        "prompt_sha256": prompt_info.get("prompt_sha256"),
        "prompt_word_count": prompt_info.get("word_count"),
        "prompt_named_site": bool(prompt_info.get("prompt_named_site", False)),
        "session_id": session.session_id,
        "brief_id": (card.get("brief") or {}).get("brief_id"),
        "brief_name": (card.get("brief") or {}).get("name"),
        "bars_sha256": session._bars.id if session._bars is not None else None,
        "model_id": h.model_id if h is not None else None,
        "weights_id": h.weights_id if h is not None else None,
        "unpacker": (lv.unpacker_describe().get("mode") if h is not None else None),
        "library_commit": _git_commit(),
        "session_budget": session.budget.to_dict(),
        "driver_budget": driver_budget.to_dict(),
    }
    usage = dict(close_info["usage"])
    usage.update({
        "turns": int(turns),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "reasoning_tokens": int(reasoning_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
        "estimated_dollars": round(float(estimated_dollars), 6),
        "tool_errors": int(tool_errors),
    })
    return {
        "v": DRIVER_VERSION,
        "finished_at": _utc_now(),
        "harness_card": harness_card,
        "stop_reason": stop_reason,
        "submitted": submitted,
        "verdict": verdict,
        "candidate_id": candidate_id,
        "knob_classes": knob_classes,
        "claims_count": len(all_claims),
        "all_claims": all_claims,
        "metrics": {
            "directed_effect": directed_effect,
            "small_directed": small_dir,
            "large_directed": large_dir,
            "near_miss_abs": near_miss_abs,
            "neutral_kl": neutral_kl,
            "panel_kl": panel_kl,
            "panel_own_drop": panel_own_drop,
            "replicated": replicated,
            "cheaper_knob_matched": cheaper_knob_matched,
        },
        "usage": usage,
        "book": {
            "path": str(session.book_path),
            "verified": book_ok,
            "entries": int(ver["entries"]),
            "head": ver["head"],
            "verdict_seq": verdict_seq,
        },
        "transcript_copy": close_info["transcript_copy"],
        "proposed_edit": dict(proposed_edit) if isinstance(proposed_edit, Mapping) else None,
        "proposed_knob": dict(proposed_knob) if isinstance(proposed_knob, Mapping) else None,
        "notes": notes[:2000],
    }


def _call(session: lses.LevelSession, tool: str, args: Mapping[str, Any]) -> Tuple[bool, Any]:
    """Invoke one gated tool through `level_mcp.dispatch_tool` (identical path to external MCP drivers)."""
    try:
        return True, lmcp.dispatch_tool(session, tool, args)
    except Exception as exc:
        return False, str(exc)


def _extract_practice_directed(read_reply: Mapping[str, Any], metric: str, direction: int) -> Optional[float]:
    dv = read_reply.get("driver_view") or {}
    body = dv.get("body") if isinstance(dv.get("body"), dict) else dv
    summ = (body.get("summary") or {}) if isinstance(body, dict) else {}
    classes = (summ.get("classes") or {}) if isinstance(summ, dict) else {}
    sgn = 1.0 if direction == 0 else float(direction)
    for cls_k in ("large", "small"):
        blk = classes.get(cls_k) or {}
        arms = blk.get("arms") or {}
        # Find main dose arm (dose 1.0, keyed by levelling._key("edit", 1.0))
        for ak, aval in arms.items():
            if not isinstance(aval, Mapping):
                continue
            if ak in ("edit:1", "edit:1.0", "edit@1", "edit@1.0", "edit") or ak.startswith(("edit:1", "edit@1")):
                mblk = aval.get(metric) or {}
                val = mblk.get("mean_over_items")
                if val is None:
                    val = mblk.get("mean")
                if isinstance(val, (int, float)):
                    return float(sgn * val)
    return None


def _practice_floor(read_reply: Mapping[str, Any], metric: str) -> float:
    dv = read_reply.get("driver_view") or {}
    body = dv.get("body") if isinstance(dv.get("body"), dict) else dv
    summ = (body.get("summary") or {}) if isinstance(body, dict) else {}
    classes = (summ.get("classes") or {}) if isinstance(summ, dict) else {}
    floors: List[float] = []
    for cls_k in ("small", "large"):
        fl = ((classes.get(cls_k) or {}).get("floor") or {}).get(metric)
        if isinstance(fl, (int, float)):
            floors.append(float(fl))
    return max(floors) if floors else 0.0


def _decode_target_query(session: lses.LevelSession) -> Optional[str]:
    """Derive a clean search term from the brief's target readouts (using only public brief_info)."""
    if session._brief is None or session._handle is None:
        return None
    odds = (session._brief.readouts or {}).get("odds")
    if isinstance(odds, Mapping) and odds.get("toward"):
        tid = int(odds["toward"][0])
        try:
            piece = session._handle.tokenizer.decode([tid]).strip()
            if piece:
                return piece
        except Exception:
            pass
    return None


# -------------------------------------------------------------------------------------- scripted baselines (no LLM)


def run_scripted_weight_driver(
    session: lses.LevelSession,
    *,
    driver_budget: Optional[DriverBudget] = None,
    kill_switch: Optional[KillSwitch] = None,
    seed: int = 0,
    submit_even_if_weak: bool = False,
) -> Dict[str, Any]:
    """Deterministic no-driver baseline (`synthesis.md` §vi, §ix) for `mode="weight_edit"`.

    Adapts honestly to the session's declared `reach`:
    - `behaviour` / `tools_only`: draws practice items, probes small deterministic candidate sets via `read`.
    - `activations` / `sandbox_no_weights`: captures a contrast direction via `capture`, builds a low-rank /
      direction candidate in `cell` (zero weight access), probes via `read`.
    - `weights` / `sandbox`: captures a contrast direction via `capture`, runs `W.writers(layer, direction)` inside
      the off-card `bwrap` `cell` to locate top stream-side writers, tests a set-size ladder via `read`.
    - `atlas`: FIRST queries `atlas_query("search_words", ...)` for the target token to locate atlas-identified
      layers/neurons, then combines with `capture` + `W.writers` in `cell` and tests via `read`.

    Finally submits the best practice candidate via `claim` (or abstains honestly if nothing clears practice floor)."""
    if session.mode != "weight_edit":
        raise DriverRefused('ScriptedWeightDriver requires session.mode == "weight_edit".')
    db = driver_budget or DriverBudget(max_turns=24, max_dollars=0.0)
    ks = kill_switch or KillSwitch()
    prompt_info = build_driver_prompt(session)
    turns = 0
    tool_errors = 0
    stop_reason = "kept_base_abstention"
    proposed_edit: Optional[Dict[str, Any]] = None
    t0 = time.monotonic()

    def _stopped() -> Optional[str]:
        if ks.triggered():
            return "killed"
        if turns >= db.max_turns:
            return "budget_turns"
        if (time.monotonic() - t0) >= db.max_wall_seconds:
            return "budget_wall"
        return None

    def _step(tool: str, args: Mapping[str, Any]) -> Tuple[bool, Any]:
        nonlocal turns, tool_errors
        turns += 1
        ok, res = _call(session, tool, args)
        if not ok:
            tool_errors += 1
        return ok, res

    ok_s, st = _step("status", {})
    if not ok_s:
        return _extract_outcome_from_book(
            session, driver_family="scripted", driver_model="scripted_weight_v1", face="direct_python",
            reasoning_effort=None, prompt_info=prompt_info, driver_budget=db,
            stop_reason="error", turns=turns, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            estimated_dollars=0.0, tool_errors=tool_errors, notes=f"status failed: {st}",
        )
    allowed = set(st["harness_card"]["allowed_tools"])
    reach = st["harness_card"]["reach"]
    has_brief = "brief_info" in allowed
    binfo: Dict[str, Any] = {}
    if has_brief:
        ok_b, b_res = _step("brief_info", {})
        if ok_b:
            binfo = b_res

    metric = str(binfo.get("metric") or "odds")
    direction = int(binfo.get("direction", -1))
    readouts = dict(binfo.get("readouts") or {"kl": True, "own": True, "odds": None})
    n_layers = int(session._handle.n_layers) if session._handle is not None else 6
    ffn_w = int(ao.ffn_width(session._handle.lm, 0)) if session._handle is not None else 128
    allowed_layers = list(binfo.get("allowed_layers") or range(n_layers))
    # Default candidate layers: late-middle layer(s) where output writers concentrate
    late_layer = allowed_layers[min(len(allowed_layers) - 1, max(0, int(round(0.83 * (len(allowed_layers) - 1)))))]
    cand_layers = sorted({late_layer, allowed_layers[-1]})

    # 1. Draw practice items (when a brief is bound)
    prac_items: Dict[str, Dict[str, Any]] = {}
    if "draw_practice" in allowed:
        ok_dp, dp = _step("draw_practice", {"n": 4, "seed": int(seed)})
        if ok_dp:
            prac_items = {
                nm: {"question": it["question"], "answer": it["answer"], "system": it.get("system")}
                for nm, it in (dp.get("items") or {}).items()
            }
    if not prac_items:
        prac_items = {
            "p0": {"question": "Answer briefly: Alice called Bob because", "answer": " she wanted to help."},
            "p1": {"question": "Answer briefly: Clara told David that", "answer": " she finished the map."},
        }

    # Build candidate recipes according to the declared reach
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    min_ctl = max(1, int(session.budget.min_controls))
    rows_prac = {"small": min(lv.ROWS_SMALL, 12), "large": min(lv.ROWS_LARGE, 16)} if n_layers <= 8 else {"small": 12, "large": 16}

    # Reach rung 4: ATLAS — consult read-only atlas index FIRST
    atlas_neurons_by_layer: Dict[int, List[int]] = {}
    if "atlas_query" in allowed and not _stopped():
        q_word = _decode_target_query(session)
        if q_word:
            ok_aq, aq = _step("atlas_query", {"op": "search_words", "args": {"query": q_word, "top_k": 32, "trusted_only": True}})
            if ok_aq and not (aq.get("hits") or []):
                ok_aq, aq = _step("atlas_query", {"op": "search_words", "args": {"query": q_word, "top_k": 16}})
            if ok_aq:
                for hit in aq.get("hits") or []:
                    hl = int(hit["layer"])
                    if hl in allowed_layers:
                        atlas_neurons_by_layer.setdefault(hl, []).append(int(hit["neuron"]))
                for hl, ns in sorted(atlas_neurons_by_layer.items(), key=lambda kv: -len(kv[1]))[:2]:
                    uniq = sorted(set(ns))[:10]
                    if uniq:
                        candidates.append((f"atlas_L{hl}_k{len(uniq)}", {"terms": [{"layer": hl, "silence": uniq}]}))
                # Also build one bounded multi-layer candidate from the highest-ranked atlas hits across late layers
                top_hits_by_l: Dict[int, List[int]] = {}
                for hit in (aq.get("hits") or [])[:12]:
                    hl = int(hit["layer"])
                    if hl in allowed_layers:
                        top_hits_by_l.setdefault(hl, []).append(int(hit["neuron"]))
                if len(top_hits_by_l) >= 2:
                    ml_terms = [
                        {"layer": hl, "silence": sorted(set(ns))[:4]}
                        for hl, ns in sorted(top_hits_by_l.items(), reverse=True)[:4]
                        if ns
                    ]
                    if ml_terms:
                        candidates.insert(0, (f"atlas_multi_{len(ml_terms)}L", {"terms": ml_terms}))

    # Reach rungs 2, 3, 4: ACTIVATIONS / WEIGHTS / ATLAS — capture contrast direction
    captured_files: List[Tuple[int, str, str]] = []
    if "capture" in allowed and not _stopped():
        # Contrast practice items against neutral/male-style contrast prompts (strictly non-kept-back!)
        items_a = [{"question": v["question"], "answer": v["answer"], "system": v.get("system")} for v in list(prac_items.values())[:2]]
        items_b = [
            {"question": "Complete the short note about a bridge:", "answer": " It crosses the wide river."},
            {"question": "Complete the short note about a library:", "answer": " It opens early every morning."},
        ]
        target_cap_layer = (
            next(iter(atlas_neurons_by_layer.keys())) if atlas_neurons_by_layer else late_layer
        )
        ok_cap, cap = _step("capture", {
            "items_a": items_a, "items_b": items_b, "layer": int(target_cap_layer),
            "where": "resid_post", "at": "end", "out_name": f"dir_L{target_cap_layer}_s{seed}.npy",
        })
        if ok_cap:
            captured_files.append((int(target_cap_layer), str(cap["rel_path"]), str(cap["sha256"])))

    # Off-card sandbox analysis cell
    if "cell" in allowed and captured_files and not _stopped():
        cap_l, cap_rel, cap_sha = captured_files[0]
        if st["harness_card"]["with_healthy_reference"]:
            # WEIGHTS or ATLAS reach: use WeightReader W.writers(layer, direction, k=16) to find top writers!
            code = f"""
import json, numpy as np
v = np.load(WORK / {cap_rel!r}).astype(np.float32)
qm = W.quiet_mask({cap_l})
v = v.copy()
v[qm] = 0.0
if float(np.linalg.norm(v)) > 1e-12:
    res = W.writers({cap_l}, v, k=16)
    pos = [int(n) for n, s in res if s > 0][:10]
    neg = [int(n) for n, s in res if s < 0][:10]
    top = [int(n) for n, _s in res[:10]]
    print("WRITERS_JSON:" + json.dumps({{"layer": {cap_l}, "pos": pos or top, "neg": neg}}))
"""
            ok_c, cres = _step("cell", {"code": code})
            if ok_c and cres.get("ok"):
                for line in str(cres.get("stdout") or "").splitlines():
                    if line.startswith("WRITERS_JSON:"):
                        wdata = json.loads(line.split("WRITERS_JSON:", 1)[1])
                        if wdata.get("pos"):
                            candidates.append((f"writers_pos_L{cap_l}", {"terms": [{"layer": cap_l, "silence": sorted(set(wdata["pos"][:6]))}]}))
                        if wdata.get("neg"):
                            candidates.append((f"writers_neg_L{cap_l}", {"terms": [{"layer": cap_l, "silence": sorted(set(wdata["neg"][:6]))}]}))
        else:
            # ACTIVATIONS-only reach (no weights mounted): build a small bounded rank-1 low-rank perturbation in /work
            code = f"""
import json, numpy as np
v = np.load(WORK / {cap_rel!r}).astype(np.float32)
vn = float(np.linalg.norm(v))
u = (v / max(vn, 1e-8)).astype(np.float32)
# Small rank-1 B [D, 1] and sparse A [1, FFN] saved to /work
B = (u[:, None] * (-0.05)).astype(np.float32)
A_mat = np.zeros((1, {ffn_w}), dtype=np.float32)
A_mat[0, :min(8, {ffn_w})] = 0.25
mb = save_array("lr_B.npy", B)
ma = save_array("lr_A.npy", A_mat)
print("LR_JSON:" + json.dumps({{"B": mb, "A": ma}}))
"""
            ok_c, cres = _step("cell", {"code": code})
            if ok_c and cres.get("ok"):
                for line in str(cres.get("stdout") or "").splitlines():
                    if line.startswith("LR_JSON:"):
                        lr = json.loads(line.split("LR_JSON:", 1)[1])
                        candidates.append((
                            f"act_lowrank_L{cap_l}",
                            {"terms": [{
                                "layer": cap_l,
                                "lowrank": {
                                    "B": lr["B"]["rel_path"],
                                    "A": lr["A"]["rel_path"],
                                },
                                "sha256": {
                                    "B": lr["B"]["sha256"],
                                    "A": lr["A"]["sha256"],
                                },
                            }]},
                        ))

    # Reach rung 1 (BEHAVIOUR) or fallback: deterministic small neuron-set probes
    if not candidates:
        base_n = (int(seed) * 7) % max(1, ffn_w - 8)
        for l_c in cand_layers[:2]:
            ns = sorted({(base_n + j) % ffn_w for j in range(min(6, ffn_w))})
            candidates.append((f"probe_L{l_c}_s{seed}", {"terms": [{"layer": l_c, "silence": ns}]}))

    # 2. Evaluate candidate edits on practice items via `read` (paired) or `write` (written_loop)
    best_cand: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_floor = 0.0

    for cand_name, edit_recipe in candidates[:3]:
        why_stop = _stopped()
        if why_stop:
            stop_reason = why_stop
            break
        if metric == "written_loop":
            if "write" not in allowed:
                break
            plan = {
                "name": f"prac_{cand_name}",
                "kind": "written",
                "edit": edit_recipe,
                "items": list(prac_items.keys())[:2],
                "doses": [1.0],
                "max_new": 24,
                "controls": {"family": "other_sites", "count": max(1, int(session.budget.min_written_controls)), "seed": int(seed)},
            }
            ok_r, rres = _step("write", {"plan": plan, "items": prac_items})
            if ok_r and best_cand is None:
                best_cand = edit_recipe
                best_score = 0.0
        else:
            if "read" not in allowed:
                break
            plan = {
                "name": f"prac_{cand_name}",
                "kind": "paired",
                "edit": edit_recipe,
                "items": list(prac_items.keys())[:2],
                "doses": [1.0],
                "rows": rows_prac,
                "readouts": readouts,
                "controls": {"family": "other_sites", "count": min_ctl, "seed": int(seed)},
            }
            ok_r, rres = _step("read", {"plan": plan, "items": prac_items})
            if ok_r:
                sc = _extract_practice_directed(rres, metric, direction)
                fl = _practice_floor(rres, metric)
                if sc is not None and (best_score is None or sc > best_score):
                    best_score = sc
                    best_floor = fl
                    best_cand = edit_recipe

    proposed_edit = best_cand

    # 3. Submit best candidate to `claim` if claim tool is enabled and candidate clears floor (or forced)
    if "claim" in allowed and best_cand is not None and not _stopped():
        should_submit = submit_even_if_weak or (best_score is not None and best_score > max(best_floor, 0.0))
        if should_submit:
            ok_cl, cl_res = _step("claim", {
                "edit": best_cand,
                "main_dose": 1.0,
                "forecast": {"direction": direction, "expected_min": round(float(best_score or 0.0), 4)},
                "selection_mode": "free_pick",
                "practice_best": round(float(best_score), 5) if best_score is not None else None,
            })
            stop_reason = "claimed" if ok_cl else "error"
        else:
            stop_reason = "kept_base_abstention"
    elif _stopped():
        stop_reason = _stopped() or "killed"

    return _extract_outcome_from_book(
        session,
        driver_family="scripted",
        driver_model="scripted_weight_v1",
        face="direct_python",
        reasoning_effort=None,
        prompt_info=prompt_info,
        driver_budget=db,
        stop_reason=stop_reason,
        turns=turns,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        estimated_dollars=0.0,
        tool_errors=tool_errors,
        notes=f"reach={reach}; evaluated {len(candidates)} candidate(s); best_practice={best_score}",
        proposed_edit=proposed_edit,
    )


def run_scripted_knob_driver(
    session: lses.LevelSession,
    *,
    driver_budget: Optional[DriverBudget] = None,
    kill_switch: Optional[KillSwitch] = None,
    seed: int = 0,
    submit_even_if_weak: bool = False,
) -> Dict[str, Any]:
    """Deterministic no-driver baseline for `mode="knobs_only"` (prompts, decoding, and transient steering only).

    Uses the EXACT same session budget and kept-back claim gate (`claim_knob`) as `ScriptedWeightDriver`."""
    if session.mode != "knobs_only":
        raise DriverRefused('ScriptedKnobDriver requires session.mode == "knobs_only".')
    db = driver_budget or DriverBudget(max_turns=24, max_dollars=0.0)
    ks = kill_switch or KillSwitch()
    prompt_info = build_driver_prompt(session)
    turns = 0
    tool_errors = 0
    stop_reason = "kept_base_abstention"
    proposed_knob: Optional[Dict[str, Any]] = None
    t0 = time.monotonic()

    def _stopped() -> Optional[str]:
        if ks.triggered():
            return "killed"
        if turns >= db.max_turns:
            return "budget_turns"
        if (time.monotonic() - t0) >= db.max_wall_seconds:
            return "budget_wall"
        return None

    def _step(tool: str, args: Mapping[str, Any]) -> Tuple[bool, Any]:
        nonlocal turns, tool_errors
        turns += 1
        ok, res = _call(session, tool, args)
        if not ok:
            tool_errors += 1
        return ok, res

    ok_s, st = _step("status", {})
    if not ok_s:
        return _extract_outcome_from_book(
            session, driver_family="scripted", driver_model="scripted_knob_v1", face="direct_python",
            reasoning_effort=None, prompt_info=prompt_info, driver_budget=db,
            stop_reason="error", turns=turns, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            estimated_dollars=0.0, tool_errors=tool_errors, notes=f"status failed: {st}",
        )
    allowed = set(st["harness_card"]["allowed_tools"])
    reach = st["harness_card"]["reach"]
    binfo: Dict[str, Any] = {}
    if "brief_info" in allowed:
        ok_b, b_res = _step("brief_info", {})
        if ok_b:
            binfo = b_res

    metric = str(binfo.get("metric") or "odds")
    direction = int(binfo.get("direction", -1))
    readouts = dict(binfo.get("readouts") or {"kl": True, "own": True, "odds": None})
    n_layers = int(session._handle.n_layers) if session._handle is not None else 6
    allowed_layers = list(binfo.get("allowed_layers") or range(n_layers))
    late_layer = allowed_layers[min(len(allowed_layers) - 1, max(0, int(round(0.83 * (len(allowed_layers) - 1)))))]
    rows_prac = {"small": min(lv.ROWS_SMALL, 12), "large": min(lv.ROWS_LARGE, 16)} if n_layers <= 8 else {"small": 12, "large": 16}

    prac_items: Dict[str, Dict[str, Any]] = {}
    if "draw_practice" in allowed:
        ok_dp, dp = _step("draw_practice", {"n": 4, "seed": int(seed)})
        if ok_dp:
            prac_items = {
                nm: {"question": it["question"], "answer": it["answer"], "system": it.get("system")}
                for nm, it in (dp.get("items") or {}).items()
            }
    if not prac_items:
        prac_items = {
            "p0": {"question": "Answer briefly: Alice called Bob because", "answer": " she wanted to help."},
            "p1": {"question": "Answer briefly: Clara told David that", "answer": " she finished the map."},
        }

    # Build cheaper-knob candidates according to reach
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    # 1. Prompt candidate derived solely from public brief goal text (available at ALL reaches!)
    goal_text = str(binfo.get("driver_text") or "Follow the target style carefully while leaving unrelated prompts unchanged.")
    candidates.append((
        "prompt_brief",
        {"prompt": f"Follow this instruction strictly only when relevant to the prompt, and answer other prompts normally: {goal_text}"},
    ))

    # 2. Decoding candidate (thinking toggle for written_loop, or mild logit bias when odds tokens are public)
    if metric == "written_loop":
        candidates.append(("decode_no_think", {"decoding": {"thinking": False}}))
    elif isinstance(readouts.get("odds"), Mapping) and readouts["odds"].get("toward"):
        tid = int(readouts["odds"]["toward"][0])
        bias_val = 1.5 if direction >= 0 else -1.5
        candidates.append(("decode_logit_bias", {"decoding": {"logit_bias": {str(tid): bias_val}}}))

    # 3. Steering candidate when `capture` is permitted (`activations`, `weights`, `atlas`)
    if "capture" in allowed and not _stopped():
        items_a = [{"question": v["question"], "answer": v["answer"], "system": v.get("system")} for v in list(prac_items.values())[:2]]
        items_b = [
            {"question": "Complete the short note about a bridge:", "answer": " It crosses the wide river."},
            {"question": "Complete the short note about a library:", "answer": " It opens early every morning."},
        ]
        ok_cap, cap = _step("capture", {
            "items_a": items_a, "items_b": items_b, "layer": int(late_layer),
            "where": "resid_post", "at": "end", "out_name": f"steer_dir_L{late_layer}_s{seed}.npy",
        })
        if ok_cap and float(cap.get("quiet_share", 1.0)) <= we.MAX_QUIET_SHARE:
            candidates.append((
                f"steer_dir_L{late_layer}",
                {"steering": [{
                    "kind": "direction",
                    "layer": int(late_layer),
                    "path": str(cap["rel_path"]),
                    "sha256": str(cap["sha256"]),
                    "dose": float(1.0 if direction >= 0 else -1.0),
                    "scale": 1.0,
                    "spare_bos": False,
                }]},
            ))

    # 4. Atlas-guided neuron_mul steering push when `atlas_query` is permitted
    if "atlas_query" in allowed and not _stopped():
        q_word = _decode_target_query(session)
        if q_word:
            ok_aq, aq = _step("atlas_query", {"op": "search_words", "args": {"query": q_word, "top_k": 24, "trusted_only": True}})
            if ok_aq and not (aq.get("hits") or []):
                ok_aq, aq = _step("atlas_query", {"op": "search_words", "args": {"query": q_word, "top_k": 10}})
            if ok_aq and aq.get("hits"):
                by_l: Dict[int, List[int]] = {}
                for hit in aq["hits"]:
                    hl = int(hit["layer"])
                    if hl in allowed_layers:
                        by_l.setdefault(hl, []).append(int(hit["neuron"]))
                if by_l:
                    top_l, top_ns = max(by_l.items(), key=lambda kv: len(kv[1]))
                    candidates.append((
                        f"atlas_mul_L{top_l}",
                        {"steering": [{"kind": "neuron_mul", "layer": top_l, "neurons": sorted(set(top_ns))[:8], "factor": 0.0}]},
                    ))

    best_knob: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_floor = 0.0
    sub_items = dict(list(prac_items.items())[:2])

    for cname, kspec in candidates[:4]:
        why_stop = _stopped()
        if why_stop:
            stop_reason = why_stop
            break
        if metric == "written_loop":
            ok_r, rres = _step("knob_write", {"knob": kspec, "items": sub_items, "max_new": 24, "seeds": [1], "name": f"prac_{cname}"})
            if ok_r:
                sc = rres.get("directed_effect")
                if sc is not None and (best_score is None or float(sc) > best_score):
                    best_score = float(sc)
                    best_knob = kspec
        else:
            ok_r, rres = _step("knob_read", {
                "knob": kspec, "items": sub_items, "readouts": readouts, "metric": metric,
                "direction": direction, "controls": max(1, int(session.budget.min_controls)),
                "rows": rows_prac, "seed": int(seed), "name": f"prac_{cname}",
            })
            if ok_r and rres.get("status") == "ok":
                lg_d = (rres.get("large") or {}).get("directed_mean")
                sm_d = (rres.get("small") or {}).get("directed_mean")
                sc = lg_d if lg_d is not None else sm_d
                fl = max(float((rres.get("small") or {}).get("floor") or 0.0), float((rres.get("large") or {}).get("floor") or 0.0))
                if sc is not None and (best_score is None or float(sc) > best_score):
                    best_score = float(sc)
                    best_floor = fl
                    best_knob = kspec

    proposed_knob = best_knob
    if "claim_knob" in allowed and best_knob is not None and not _stopped():
        should_submit = submit_even_if_weak or (best_score is not None and best_score > max(best_floor, 0.0))
        if should_submit:
            ok_cl, _cl_res = _step("claim_knob", {
                "knob": best_knob,
                "forecast": {"direction": direction, "expected_min": round(float(best_score or 0.0), 4)},
                "practice_best": round(float(best_score), 5) if best_score is not None else None,
            })
            stop_reason = "claimed" if ok_cl else "error"
        else:
            stop_reason = "kept_base_abstention"
    elif _stopped():
        stop_reason = _stopped() or "killed"

    return _extract_outcome_from_book(
        session,
        driver_family="scripted",
        driver_model="scripted_knob_v1",
        face="direct_python",
        reasoning_effort=None,
        prompt_info=prompt_info,
        driver_budget=db,
        stop_reason=stop_reason,
        turns=turns,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        estimated_dollars=0.0,
        tool_errors=tool_errors,
        notes=f"reach={reach}; mode=knobs_only; evaluated {len(candidates)} knob(s); best_practice={best_score}",
        proposed_knob=proposed_knob,
    )


# -------------------------------------------------------------------------------------- replay driver


def run_replay_driver(
    session: lses.LevelSession,
    steps: Sequence[Mapping[str, Any]],
    *,
    driver_model: str = "replay_v1",
    driver_budget: Optional[DriverBudget] = None,
    kill_switch: Optional[KillSwitch] = None,
) -> Dict[str, Any]:
    """Deterministically replay a sequence of `{"tool": name, "args": {...}}` calls through `level_mcp.dispatch_tool`."""
    db = driver_budget or DriverBudget(max_turns=max(1, len(steps) + 2), max_dollars=0.0)
    ks = kill_switch or KillSwitch()
    prompt_info = build_driver_prompt(session)
    turns = 0
    tool_errors = 0
    stop_reason = "kept_base_abstention"
    proposed_edit = None
    proposed_knob = None
    t0 = time.monotonic()

    for step in steps:
        if ks.triggered():
            stop_reason = "killed"
            break
        if turns >= db.max_turns:
            stop_reason = "budget_turns"
            break
        if (time.monotonic() - t0) >= db.max_wall_seconds:
            stop_reason = "budget_wall"
            break
        tool = str(step.get("tool") or "")
        args = dict(step.get("args") or {})
        turns += 1
        ok, res = _call(session, tool, args)
        if not ok:
            tool_errors += 1
        else:
            if tool == "claim":
                proposed_edit = args.get("edit")
                stop_reason = "claimed"
            elif tool == "claim_knob":
                proposed_knob = args.get("knob")
                stop_reason = "claimed"
            elif tool == "read" and step.get("propose"):
                plan_arg = args.get("plan")
                if isinstance(plan_arg, Mapping) and isinstance(plan_arg.get("edit"), Mapping):
                    proposed_edit = dict(plan_arg["edit"])

    return _extract_outcome_from_book(
        session,
        driver_family="replay",
        driver_model=driver_model,
        face="replay",
        reasoning_effort=None,
        prompt_info=prompt_info,
        driver_budget=db,
        stop_reason=stop_reason,
        turns=turns,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        estimated_dollars=0.0,
        tool_errors=tool_errors,
        notes=f"replayed {turns}/{len(steps)} step(s)",
        proposed_edit=proposed_edit,
        proposed_knob=proposed_knob,
    )


# -------------------------------------------------------------------------------------- OpenAI-compatible driver


def _mcp_tools_to_openai(tools: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _resolve_openai_profile(
    profile_name: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
) -> Tuple[str, Optional[str], bool]:
    """Resolve `(url, api_key, is_local)` using `backend/scientist_chat.py`'s profiles (never storing keys on disk)."""
    url = base_url
    key_env = api_key_env
    if profile_name and not url:
        from backend import scientist_chat as sc

        profs = sc.profiles()
        if profile_name not in profs:
            raise DriverRefused(f"Unknown OpenAI endpoint profile {profile_name!r} (available: {', '.join(sorted(profs))}).")
        p = profs[profile_name]
        url = str(p.get("base_url") or "")
        if not key_env:
            key_env = p.get("key_env")
    if not url:
        url = os.environ.get("THEODOLITE_OPENAI_BASE_URL", "").strip()
    if not url:
        raise DriverRefused("OpenAI driver requires --profile, base_url, or $THEODOLITE_OPENAI_BASE_URL.")
    if not url.rstrip("/").endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    api_key = os.environ.get(key_env, "").strip() if key_env else os.environ.get("THEODOLITE_OPENAI_API_KEY", "").strip()
    is_local = any(h in url for h in ("127.0.0.1", "localhost", "[::1]"))
    return url, (api_key or None), is_local


def _default_openai_transport(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_openai_driver(
    session: lses.LevelSession,
    *,
    model: str,
    profile: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    driver_budget: Optional[DriverBudget] = None,
    kill_switch: Optional[KillSwitch] = None,
    transport_fn: Optional[Callable[[Dict[str, Any], Dict[str, str], str, float], Dict[str, Any]]] = None,
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Run any OpenAI-compatible tool-calling model against `session`, enforcing token/dollar/turn/wall caps."""
    if not model or not isinstance(model, str):
        raise DriverRefused("OpenAI driver requires a non-empty model name.")
    db = driver_budget or DriverBudget()
    ks = kill_switch or KillSwitch()

    if transport_fn is None:
        url, api_key, is_local = _resolve_openai_profile(profile, base_url, api_key_env)
        if not is_local and (db.max_dollars is None or db.max_dollars <= 0.0):
            raise DriverRefused(
                "Remote OpenAI-compatible driver requires an explicit owner-approved max_dollars > 0 ceiling."
            )
    else:
        url, api_key, is_local = "http://mock.local/v1/chat/completions", None, True

    prompt_info = build_driver_prompt(session, extra_instructions=extra_instructions)
    oa_tools = _mcp_tools_to_openai(lmcp.tools_for_session(session))
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": prompt_info["system"]},
        {"role": "user", "content": prompt_info["user"]},
    ]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    turns = 0
    prompt_toks = 0
    comp_toks = 0
    reas_toks = 0
    est_usd = 0.0
    tool_errors = 0
    served_model = model
    stop_reason = "kept_base_abstention"
    last_text = ""
    proposed_edit = None
    proposed_knob = None
    t0 = time.monotonic()
    transport = transport_fn or _default_openai_transport

    while True:
        if ks.triggered():
            stop_reason = "killed"
            break
        if turns >= db.max_turns:
            stop_reason = "budget_turns"
            break
        if (time.monotonic() - t0) >= db.max_wall_seconds:
            stop_reason = "budget_wall"
            break
        if (prompt_toks + comp_toks) >= db.max_tokens:
            stop_reason = "budget_tokens"
            break
        if db.max_dollars is not None and db.max_dollars > 0.0 and est_usd >= db.max_dollars:
            stop_reason = "budget_dollars"
            break

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": oa_tools,
            "tool_choice": "auto",
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        turns += 1
        rem_wall = max(5.0, db.max_wall_seconds - (time.monotonic() - t0))
        try:
            resp = transport(payload, headers, url, min(120.0, rem_wall))
        except Exception as exc:
            stop_reason = "error"
            last_text = f"endpoint error: {exc}"
            break

        if isinstance(resp.get("model"), str) and resp["model"]:
            served_model = str(resp["model"])
        usage = resp.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        rt = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        prompt_toks += pt
        comp_toks += ct
        reas_toks += rt
        if isinstance(usage.get("cost"), (int, float)):
            est_usd += float(usage["cost"])
        else:
            est_usd += (pt / 1e6) * db.input_usd_per_mtok + (ct / 1e6) * db.output_usd_per_mtok

        choices = resp.get("choices") or []
        if not choices:
            stop_reason = "error"
            last_text = "empty choices in response"
            break
        msg = choices[0].get("message") or {}
        messages.append(msg)
        if isinstance(msg.get("content"), str) and msg["content"]:
            last_text = msg["content"]

        tcalls = msg.get("tool_calls") or []
        if not tcalls:
            if stop_reason != "claimed":
                stop_reason = "kept_base_abstention"
            break

        for tc in tcalls:
            if ks.triggered():
                stop_reason = "killed"
                break
            fn = tc.get("function") or {}
            tname = str(fn.get("name") or "")
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except ValueError as exc:
                tool_errors += 1
                out_str = f"error: invalid JSON arguments: {exc}"
            else:
                ok, res = _call(session, tname, parsed_args)
                if not ok:
                    tool_errors += 1
                    out_str = f"error: {res}"
                else:
                    if tname == "claim":
                        proposed_edit = parsed_args.get("edit")
                        stop_reason = "claimed"
                    elif tname == "claim_knob":
                        proposed_knob = parsed_args.get("knob")
                        stop_reason = "claimed"
                    out_str = json.dumps(res, ensure_ascii=True)
                    if len(out_str) > 16000:
                        out_str = out_str[:16000] + "... [truncated]"
            messages.append({
                "role": "tool",
                "tool_call_id": str(tc.get("id") or f"call_{turns}"),
                "content": out_str,
            })
        if stop_reason == "killed":
            break

    return _extract_outcome_from_book(
        session,
        driver_family="openai",
        driver_model=served_model,
        face="openai_tools",
        reasoning_effort=reasoning_effort,
        prompt_info=prompt_info,
        driver_budget=db,
        stop_reason=stop_reason,
        turns=turns,
        prompt_tokens=prompt_toks,
        completion_tokens=comp_toks,
        reasoning_tokens=reas_toks,
        estimated_dollars=est_usd,
        tool_errors=tool_errors,
        notes=last_text,
        proposed_edit=proposed_edit,
        proposed_knob=proposed_knob,
    )


# -------------------------------------------------------------------------------------- Claude CLI & Codex CLI


def run_claude_cli_driver(
    session: lses.LevelSession,
    *,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    driver_budget: Optional[DriverBudget] = None,
    kill_switch: Optional[KillSwitch] = None,
    claude_bin: Optional[str] = None,
    allow_mock_cli: bool = False,
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Launch Claude Code (`claude -p`, stream-json) at folder reach against `session` over `session_socket_bridge`.

    Containment:
    - Refuses `$THEODOLITE_SCIENTIST_REACH=home`.
    - Requires owner-approved `driver_budget.max_dollars > 0` (unless `allow_mock_cli=True` in CPU checks).
    - `cwd = session.work_dir` ONLY; `--setting-sources ""`; `--strict-mcp-config`; `--tools Read Write Edit`;
      `--disallowedTools Bash`; credential-stripped `scientist_chat.child_env("claude")`.
    - Process group (`start_new_session=True`) killed with `SIGKILL` on wall-clock cap, dollar cap, or `KillSwitch`."""
    if os.environ.get(lses.REACH_ENV, "").strip().lower() == "home":
        raise DriverRefused(f"Claude CLI driver is refused at the home reach (${lses.REACH_ENV}=home).")
    db = driver_budget or DriverBudget()
    if not allow_mock_cli and (db.max_dollars is None or db.max_dollars <= 0.0):
        raise DriverRefused("Claude CLI driver requires an explicit owner-approved max_dollars > 0 ceiling.")
    exe = claude_bin or os.environ.get("THEODOLITE_CLAUDE_BIN", "claude")
    if not allow_mock_cli and shutil.which(exe) is None and not Path(exe).is_file():
        raise DriverRefused(f"Claude CLI executable {exe!r} was not found on PATH.")

    from backend import scientist_chat as sc

    ks = kill_switch or KillSwitch()
    prompt_info = build_driver_prompt(session, extra_instructions=extra_instructions)
    sock_path = session.session_dir / "mcp.sock"
    mcp_cfg_path = session.session_dir / "mcp_config.json"

    turns = 0
    prompt_toks = 0
    comp_toks = 0
    est_usd = 0.0
    served_model = model or "claude-cli"
    stop_reason = "kept_base_abstention"
    notes_parts: List[str] = []

    with lmcp.session_socket_bridge(session, sock_path):
        mcp_cfg = {
            "mcpServers": {
                "levelling": {
                    "command": sys.executable,
                    "args": ["-I", str(ROOT_DIR / "backend" / "level_mcp.py"), "--socket", str(sock_path)],
                }
            }
        }
        mcp_cfg_path.write_text(json.dumps(mcp_cfg, indent=2) + "\n", encoding="utf-8")

        cmd = [
            exe, "-p", prompt_info["user"],
            "--output-format", "stream-json",
            "--verbose",
            "--strict-mcp-config",
            "--mcp-config", str(mcp_cfg_path),
            "--setting-sources", "",
            "--tools", "Read", "Write", "Edit",
            "--disallowedTools", "Bash",
            "--system-prompt", prompt_info["system"],
        ]
        if model:
            cmd += ["--model", str(model)]
        if db.max_turns:
            cmd += ["--max-turns", str(int(db.max_turns))]

        env = sc.child_env("claude", THEODOLITE_MCP_SOCKET=str(sock_path))
        stderr_log = open(session.session_dir / "cli_stderr.log", "w", encoding="utf-8")
        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(session.work_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            start_new_session=True,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            while True:
                if ks.triggered():
                    stop_reason = "killed"
                    _kill_pgrp(proc)
                    break
                if (time.monotonic() - t0) >= db.max_wall_seconds:
                    stop_reason = "budget_wall"
                    _kill_pgrp(proc)
                    break
                if db.max_dollars is not None and db.max_dollars > 0.0 and est_usd >= db.max_dollars:
                    stop_reason = "budget_dollars"
                    _kill_pgrp(proc)
                    break
                rlist, _, _ = select.select([fd], [], [], 0.2)
                if not rlist:
                    if proc.poll() is not None:
                        break
                    continue
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                etype = ev.get("type")
                if etype == "assistant":
                    turns += 1
                    msg = ev.get("message") or {}
                    if isinstance(msg.get("model"), str) and msg["model"]:
                        served_model = str(msg["model"])
                    u = msg.get("usage") or {}
                    prompt_toks += int(u.get("input_tokens") or 0)
                    comp_toks += int(u.get("output_tokens") or 0)
                elif etype == "result":
                    if isinstance(ev.get("total_cost_usd"), (int, float)):
                        est_usd = float(ev["total_cost_usd"])
                    if isinstance(ev.get("num_turns"), int):
                        turns = max(turns, int(ev["num_turns"]))
                    if isinstance(ev.get("result"), str):
                        notes_parts.append(ev["result"])
                    if ev.get("is_error"):
                        stop_reason = "error"
            proc.wait(timeout=5.0)
        finally:
            if proc.poll() is None:
                _kill_pgrp(proc)
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
            try:
                stderr_log.close()
            except OSError:
                pass
            if mcp_cfg_path.exists():
                try:
                    mcp_cfg_path.unlink()
                except OSError:
                    pass

    return _extract_outcome_from_book(
        session,
        driver_family="claude",
        driver_model=served_model,
        face="mcp_stdio",
        reasoning_effort=reasoning_effort,
        prompt_info=prompt_info,
        driver_budget=db,
        stop_reason=stop_reason,
        turns=turns,
        prompt_tokens=prompt_toks,
        completion_tokens=comp_toks,
        reasoning_tokens=0,
        estimated_dollars=est_usd,
        tool_errors=0,
        notes="\n".join(notes_parts),
    )


def run_codex_cli_driver(
    session: lses.LevelSession,
    *,
    codex_probed: bool = False,
    driver_budget: Optional[DriverBudget] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Codex CLI driver gate (`verify/arbiter.md` §5, row 27): strictly refused until probed and compile cache moved."""
    if not codex_probed or not lses._compile_cache_is_private():
        raise DriverRefused(
            "Codex CLI driver is refused until its real sandbox has been probed (codex_probed=True) and "
            "TORCHINDUCTOR_CACHE_DIR has been moved out of /tmp."
        )
    if os.environ.get(lses.REACH_ENV, "").strip().lower() == "home":
        raise DriverRefused(f"Codex CLI driver is refused at the home reach (${lses.REACH_ENV}=home).")
    db = driver_budget or DriverBudget()
    if db.max_dollars is None or db.max_dollars <= 0.0:
        raise DriverRefused("Codex CLI driver requires an explicit owner-approved max_dollars > 0 ceiling.")
    raise DriverRefused(
        "Unattended Codex app-server driver loop is not enabled by default; probe and wire codex_probed explicitly."
    )


def _kill_pgrp(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def run_driver(
    session: lses.LevelSession,
    driver_family: str,
    *,
    driver_model: Optional[str] = None,
    driver_budget: Optional[DriverBudget] = None,
    kill_switch: Optional[KillSwitch] = None,
    seed: int = 0,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Uniform dispatcher for any Phase 6 driver family against a live `LevelSession`."""
    cfg = dict(config or {})
    if driver_family == "scripted":
        if session.mode == "knobs_only":
            return run_scripted_knob_driver(
                session,
                driver_budget=driver_budget,
                kill_switch=kill_switch,
                seed=seed,
                submit_even_if_weak=bool(cfg.get("submit_even_if_weak", False)),
            )
        return run_scripted_weight_driver(
            session,
            driver_budget=driver_budget,
            kill_switch=kill_switch,
            seed=seed,
            submit_even_if_weak=bool(cfg.get("submit_even_if_weak", False)),
        )
    if driver_family == "replay":
        steps = cfg.get("steps")
        if not isinstance(steps, Sequence):
            raise DriverRefused('Replay driver requires config["steps"] (list of {tool, args}).')
        return run_replay_driver(
            session,
            steps,
            driver_model=driver_model or "replay_v1",
            driver_budget=driver_budget,
            kill_switch=kill_switch,
        )
    if driver_family == "openai":
        return run_openai_driver(
            session,
            model=driver_model or str(cfg.get("model") or ""),
            profile=cfg.get("profile"),
            base_url=cfg.get("base_url"),
            api_key_env=cfg.get("api_key_env"),
            reasoning_effort=cfg.get("reasoning_effort"),
            driver_budget=driver_budget,
            kill_switch=kill_switch,
            transport_fn=cfg.get("transport_fn"),
            extra_instructions=cfg.get("extra_instructions"),
        )
    if driver_family == "claude":
        return run_claude_cli_driver(
            session,
            model=driver_model,
            reasoning_effort=cfg.get("reasoning_effort"),
            driver_budget=driver_budget,
            kill_switch=kill_switch,
            claude_bin=cfg.get("claude_bin"),
            allow_mock_cli=bool(cfg.get("allow_mock_cli", False)),
            extra_instructions=cfg.get("extra_instructions"),
        )
    if driver_family == "codex":
        return run_codex_cli_driver(
            session,
            codex_probed=bool(cfg.get("codex_probed", False)),
            driver_budget=driver_budget,
        )
    raise DriverRefused(f"Unknown driver_family {driver_family!r}; choose from {', '.join(DRIVER_FAMILIES)}.")
