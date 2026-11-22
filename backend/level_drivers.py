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
  profiles (API keys read strictly from environment variables, never stored on disk; a remote endpoint only through
  the profile at its address, with that profile's key; redirects refused; injectable transport for tests).
- `claude`: Claude Code CLI (`claude -p --output-format stream-json`) with the session's door tools and nothing else
  (`claude_cli_args`: `--tools ""`, the door's tools in `--allowedTools`, `--permission-mode dontAsk`, no settings,
  CLAUDE.md, auto-memory or saved transcript; an allowlisted environment, `claude_child_env`), connected to the live
  `LevelSession` via `level_mcp.session_socket_bridge` so multi-session studies do not reload the 31B checkpoint per
  run.
- `codex`: refused unless `codex_probed=True` and `TORCHINDUCTOR_CACHE_DIR` is outside `/tmp`.

Hard unattended caps (`verify/arbiter.md` row 34, `verify/skeptic_safety.md:271`), kept during the run:
- `DriverBudget`: `max_turns`, `max_tokens`, `max_dollars` (owner-approved ceiling required for paid drivers; a study
  passes what it has left), `max_wall_seconds` with process-group `SIGKILL`. Claude Code is also given the dollars as
  `--max-budget-usd`. An OpenAI-compatible endpoint's dollars are counted at known prices, never at a silent 0, from
  token counts no lower than the harness's own floor; each request is sent only when an upper bound of its prompt
  fits what is left, with a completion limit sized to the rest (`run_openai_driver`).
- `KillSwitch`: file-based (`STOP` file) and event/signal kill switch checked before every turn and subprocess read,
  and every 0.2 s while an endpoint request is out.
- Every outcome, whatever the driver family, carries `driver_failure`, `stop_reason` and `door` (`{"calls", "ok",
  "errors"}`, `_extract_outcome_from_book`), and a claim the door refused in its own words (`claim_refused`).
- A session whose door calls were denied, or none of whose door calls succeeded, ends as `driver_failure`, never as
  an honest abstention. One its own cap or STOP ended before it made any door call is not a driver's failure.

Zero kept-out words appear in source, docstrings, prompts, or outputs.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import ipaddress
import json
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
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
    """Hard driver-side caps paired with `level_session.SessionBudget`. A study passes each session what is left of
    its dollars as `max_dollars`. The prices are dollars a million tokens; None is unknown, never a silent 0 (a remote
    OpenAI-compatible endpoint with unknown prices is refused, `_openai_prices`)."""

    max_turns: int = 24
    max_tokens: int = 200_000
    max_dollars: Optional[float] = None
    max_wall_seconds: float = 1200.0
    input_usd_per_mtok: Optional[float] = None
    output_usd_per_mtok: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_turns": int(self.max_turns),
            "max_tokens": int(self.max_tokens),
            "max_dollars": float(self.max_dollars) if self.max_dollars is not None else None,
            "max_wall_seconds": float(self.max_wall_seconds),
            "input_usd_per_mtok": float(self.input_usd_per_mtok) if self.input_usd_per_mtok is not None else None,
            "output_usd_per_mtok": float(self.output_usd_per_mtok) if self.output_usd_per_mtok is not None else None,
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


def _fresh_readings_left(session: lses.LevelSession, brief_card: Mapping[str, Any]) -> Any:
    """How many of the brief's fresh readings the session's book has left (claims, knob claims and trials spend
    them alike, `level_claim.brief_history`), or the brief's whole number when the book cannot be read."""
    m = brief_card.get("max_fresh_readings")
    brief = getattr(session, "_brief", None)
    if brief is None or not isinstance(m, int):
        return m
    try:
        from backend import level_claim as lcl

        hist = lcl.brief_history(brief.id, session.book_path, name=brief.name)
        return max(0, int(m) - int(hist["fresh_count"]))
    except Exception:  # an unreadable book: the brief's own number, as before
        return m


BLINDED_KEEP_BASE_SAY = (
    "To keep the base model, call propose_repair once with edit=null: that is an abstention. A session that ends "
    "without a propose_repair the door takes gives no answer, and no answer is not an abstention."
)


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

    # In a blinded session the base model is kept by saying so, a propose_repair with edit null: a session that stops
    # without one gives no answer, which the benchmark never counts as an abstention (level_eval.ANSWER_SAYS)
    blinded = "propose_repair" in tools
    if is_bare:
        sys_lines = [
            f"You are driving a Theodolite Levelling session (mode={mode}, reach={reach}).",
            f"Available tools: {', '.join(tools)}.",
            (
                f"Budgets: max_gpu_seconds={bud['max_gpu_seconds']:g}, max_forwards={bud['max_forwards']}, "
                f"max_cells={bud['max_cells']}, max_wall_seconds={bud['max_wall_seconds']:g}."
            ),
            (BLINDED_KEEP_BASE_SAY if blinded
             else "Keeping the untouched base model (submitting nothing when no candidate works) is a valid outcome."),
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
            ("- " + BLINDED_KEEP_BASE_SAY if blinded else
             "- Keeping the base model (stopping without a claim if nothing clears the noise floor and controls) is an honest ending."),
            (
                f"- Hard budgets: {bud['max_gpu_seconds']:g} GPU-seconds, {bud['max_forwards']} practice forwards, "
                f"{bud['max_cells']} cells, {bud['max_wall_seconds']:g} wall seconds."
            ),
        ]
        if mode == "knobs_only":
            sys_lines += [
                "",
                "Cheaper-knob mode (zero weight edits):",
                "- Weight-edit tools are disabled. Search only over prompts, decoding (thinking, sampler, logit_bias)",
                "  and transient steering via knob_read / knob_write: direction adds a fixed vector to the residual stream",
                "  at a layer's output; neuron_add adds a fixed push there along its neurons' summed write direction (not",
                "  an add to their activations); neuron_mul multiplies its neurons' activations where they enter the down",
                "  projection.",
                "- A fresh claim needs a steering part: a prompt or decoding setting alone has no control edits to be",
                "  judged against and is refused.",
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
                sys_lines.append("- atlas_query searches read-only neuron labels and wiring. Every word is a lead to test by a reading; each label carries the label gate's check (label_gate: held_back, reasons), and steer words what the steer lens's own check found.")
            if "claim" in tools:
                sys_lines.append("- When a candidate clears practice controls and floor, submit it once via claim for a fresh verdict.")
            if "propose_repair" in tools:
                sys_lines.append("- When your investigation is finished, call propose_repair once with your candidate repair recipe (or with edit=null to abstain if no change is warranted).")

    if extra_instructions:
        sys_lines += ["", str(extra_instructions).strip()]
    system_text = "\n".join(sys_lines).strip()

    if brief_card:
        user_lines = [
            f"Brief: {brief_card.get('name')}",
            f"Goal: {brief_card.get('driver_text')}",
            f"Metric: {brief_card.get('metric')} (direction={brief_card.get('direction')}, least_effect={brief_card.get('least_effect')}).",
            f"Allowed tensors: {', '.join(brief_card.get('allowed_tensors') or ())}.",
            f"Fresh readings this book has left for the brief: {_fresh_readings_left(session, brief_card)}.",
            "Call status and brief_info first if needed, explore on practice items, and either submit one candidate or stop and keep the base.",
        ]
    elif "propose_repair" in tools:
        user_lines = [
            "No claim brief is bound (blinded diagnostic session). Call status first, investigate whether the live model exhibits an anomaly, and call propose_repair once with your candidate repair recipe or with edit=null to abstain.",
        ]
    else:
        user_lines = [
            "No claim brief is bound (practice/blinded session). Call status first, diagnose any anomaly or test your hypothesis on practice items, and summarise your findings.",
        ]
    user_text = "\n".join(user_lines).strip()
    digest = hashlib.sha256((system_text + "\n---\n" + user_text).encode("utf-8")).hexdigest()
    words = len((system_text + " " + user_text).split())
    # Whether the page names a site: the brief says its own text does, or the page names a layer or a neuron by number
    # (level_eval's finder; imported here, as level_eval imports this module when it loads)
    from backend.level_eval import sites_named_in

    brief_obj = getattr(session, "_brief", None)
    sites = sites_named_in(system_text + "\n" + user_text)
    named = bool(getattr(brief_obj, "prompt_named_site", False)) or bool(sites)
    return {
        "system": system_text,
        "user": user_text,
        "prompt_sha256": digest,
        "word_count": words,
        "bare": is_bare,
        "prompt_named_site": named,
        "sites_named": sites,
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
    driver_failure: Optional[str] = None,
    door: Optional[Mapping[str, Any]] = None,
    claim_refused: Optional[str] = None,
    extra_usage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical `SessionOutcome` from `session` and its verified level book (never trusting driver prose).

    Every driver family's outcome carries the same three things a study counts a session by:
    - `driver_failure`: why the driver could not work the door, in plain words, or None (`_door_failure`);
    - `stop_reason`: why the run ended;
    - `door`: what the door saw of the driver's calls, at least `{"calls", "ok", "errors"}` (for a CLI driver also
      what was running when it was stopped, and how long the door waited for it).
    `claim_refused` is the door's own words for the last claim it refused, or None; `extra_usage` adds to `usage`
    (an endpoint's own token figures beside the counted ones)."""
    door_d = dict(door) if isinstance(door, Mapping) else {}
    for k in ("calls", "ok", "errors"):
        door_d[k] = int(door_d.get(k) or 0)
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

    # Only the lines this session wrote are its outcome: a book seeded from another (a study's calibration book) or
    # shared with an earlier session holds claims that are not this driver's
    own_seqs = {int(s) for s in (close_info.get("book_seqs") or [])}
    seed_verdict_set_aside: Optional[Dict[str, Any]] = None  # the last verdict in the book this session did not enter
    if book_ok and session.book_path.exists():
        for entry in lb.entries(session.book_path):
            if entry.get("kind") != "note":
                continue
            body = entry.get("body") or {}
            nt = body.get("note_type") or body.get("note")
            try:
                own_line = int(entry.get("seq", -1)) in own_seqs
            except (TypeError, ValueError):
                own_line = False
            if not own_line:
                if nt in ("claim_verdict", "knob_claim_verdict"):
                    seed_verdict_set_aside = {"seq": entry.get("seq"), "verdict": body.get("verdict")}
                continue
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
    if isinstance(extra_usage, Mapping):
        usage.update(dict(extra_usage))
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
        **({"seed_verdict_set_aside": seed_verdict_set_aside} if seed_verdict_set_aside else {}),
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
        "driver_failure": driver_failure,
        "claim_refused": claim_refused,
        "door": door_d,
        "notes": notes[:2000],
    }


def _call(session: lses.LevelSession, tool: str, args: Mapping[str, Any]) -> Tuple[bool, Any]:
    """Invoke one gated tool through `level_mcp.dispatch_tool` (identical path to external MCP drivers)."""
    try:
        return True, lmcp.dispatch_tool(session, tool, args)
    except Exception as exc:
        return False, str(exc)


def _plain_ending(stop_reason: str, calls: int, errors: int) -> Tuple[str, Optional[str], Dict[str, int]]:
    """`(stop_reason, driver_failure, door)` for a driver whose door calls are made in this process, one at a time (a
    scripted baseline, a replay): what the door saw of them, and whether the driver worked it at all."""
    door = _door_counts(calls, errors)
    failure = _door_failure(door["ok"], door["errors"], stop_reason=stop_reason)
    if failure and stop_reason in ("kept_base_abstention", "error"):
        stop_reason = "driver_failure"
    return stop_reason, failure, door


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


def _claimable_recipe(recipe: Mapping[str, Any]) -> bool:
    """Whether the claim gate can take an edit of this recipe: one with at least one term that is not low-rank. An
    edit of low-rank terms alone has no family of control edits yet, so the gate refuses it
    (`level_claim.run_fresh_claim`); a baseline that put one forward would end in an error, not an abstention."""
    terms = (recipe or {}).get("terms") or []
    return any(isinstance(t, Mapping) and "lowrank" not in t for t in terms)


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
        stop_e, failure_e, door_e = _plain_ending("error", turns, tool_errors)
        return _extract_outcome_from_book(
            session, driver_family="scripted", driver_model="scripted_weight_v1", face="direct_python",
            reasoning_effort=None, prompt_info=prompt_info, driver_budget=db,
            stop_reason=stop_e, turns=turns, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            estimated_dollars=0.0, tool_errors=tool_errors, notes=f"status failed: {st}",
            driver_failure=failure_e, door=door_e,
        )
    claim_refused: Optional[str] = None
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
            ok_aq, aq = _step("atlas_query", {"op": "search_words", "args": {"query": q_word, "top_k": 32, "gate_passes_only": True}})
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
            if ok_r and best_cand is None and _claimable_recipe(edit_recipe):
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
                # Practice reads every candidate; the one put forward is the best of those the claim gate can take
                # (an edit of low-rank terms alone has no control edits yet, and the gate refuses it)
                if sc is not None and _claimable_recipe(edit_recipe) and (best_score is None or sc > best_score):
                    best_score = sc
                    best_floor = fl
                    best_cand = edit_recipe

    # The proposal is the edit this driver submitted, and only that: its best practice candidate is not a proposal
    # when it kept the base instead (a blinded case would otherwise score an abstention as a repair)
    proposed_edit = None

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
            if ok_cl:
                proposed_edit = best_cand
            else:
                claim_refused = _refusal_words(cl_res)
        else:
            stop_reason = "kept_base_abstention"
    elif "propose_repair" in allowed and not _stopped():
        # A blinded session: the baseline abstains by rule, whatever it read. That checks propose_repair's path and is
        # never a measurement (level_eval.run_blinded_benchmark scores it "plumbing check (scripted baseline)")
        ok_pr, _pr_res = _step("propose_repair", {
            "edit": None,
            "diagnosis": "scripted baseline submits abstention in blinded diagnostic session",
        })
        stop_reason = "kept_base_abstention" if ok_pr else "error"
    elif _stopped():
        stop_reason = _stopped() or "killed"

    stop_reason, failure, door = _plain_ending(stop_reason, turns, tool_errors)
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
        driver_failure=failure,
        door=door,
        claim_refused=claim_refused,
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
        stop_e, failure_e, door_e = _plain_ending("error", turns, tool_errors)
        return _extract_outcome_from_book(
            session, driver_family="scripted", driver_model="scripted_knob_v1", face="direct_python",
            reasoning_effort=None, prompt_info=prompt_info, driver_budget=db,
            stop_reason=stop_e, turns=turns, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            estimated_dollars=0.0, tool_errors=tool_errors, notes=f"status failed: {st}",
            driver_failure=failure_e, door=door_e,
        )
    claim_refused: Optional[str] = None
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
            ok_aq, aq = _step("atlas_query", {"op": "search_words", "args": {"query": q_word, "top_k": 24, "gate_passes_only": True}})
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

    # The candidates that can be claimed (those with a steering part) are read first, so the four read include them
    ordered = [c for c in candidates if c[1].get("steering")] + [c for c in candidates if not c[1].get("steering")]
    for cname, kspec in ordered[:4]:
        why_stop = _stopped()
        if why_stop:
            stop_reason = why_stop
            break
        if metric == "written_loop":
            ok_r, rres = _step("knob_write", {"knob": kspec, "items": sub_items, "max_new": 24, "seeds": [1], "name": f"prac_{cname}"})
            if ok_r:
                sc = rres.get("directed_effect")
                if sc is not None and kspec.get("steering") and (best_score is None or float(sc) > best_score):
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
                if sc is not None and kspec.get("steering") and (best_score is None or float(sc) > best_score):
                    best_score = float(sc)
                    best_floor = fl
                    best_knob = kspec

    # Practice reads every candidate; only one with a steering part can be claimed (a prompt or decoding setting alone
    # has no control edits to be judged against, and the gate refuses it), so the best of those is put forward. The
    # proposal is the knob this driver submitted, and only that.
    proposed_knob = None
    if "claim_knob" in allowed and best_knob is not None and not _stopped():
        should_submit = submit_even_if_weak or (best_score is not None and best_score > max(best_floor, 0.0))
        if should_submit:
            ok_cl, cl_res = _step("claim_knob", {
                "knob": best_knob,
                "forecast": {"direction": direction, "expected_min": round(float(best_score or 0.0), 4)},
                "practice_best": round(float(best_score), 5) if best_score is not None else None,
            })
            stop_reason = "claimed" if ok_cl else "error"
            if ok_cl:
                proposed_knob = best_knob
            else:
                claim_refused = _refusal_words(cl_res)
        else:
            stop_reason = "kept_base_abstention"
    elif _stopped():
        stop_reason = _stopped() or "killed"

    stop_reason, failure, door = _plain_ending(stop_reason, turns, tool_errors)
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
        driver_failure=failure,
        door=door,
        claim_refused=claim_refused,
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
    claim_refused: Optional[str] = None
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
            if tool in CLAIM_TOOLS:
                claim_refused = _refusal_words(res)
        else:
            if tool == "claim":
                proposed_edit = args.get("edit")
                stop_reason = "claimed"
            elif tool == "claim_knob":
                proposed_knob = args.get("knob")
                stop_reason = "claimed"
            elif tool == "propose_repair":
                # a replay proposes only as any driver does, by a propose_repair the door takes (a read is never one)
                ed_arg = args.get("edit")
                proposed_edit = dict(ed_arg) if isinstance(ed_arg, Mapping) else None
                stop_reason = "proposed_repair" if proposed_edit is not None else "kept_base_abstention"

    stop_reason, failure, door = _plain_ending(stop_reason, turns, tool_errors)
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
        driver_failure=failure,
        door=door,
        claim_refused=claim_refused,
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


def endpoint_is_local(url: str) -> bool:
    """True only for an http(s) address whose host is `localhost`, an address in 127.0.0.0/8, or ::1. The host is
    parsed, not searched for: `https://localhost.attacker.example/v1` is not local, nor is `https://x.example/?h=::1`."""
    try:
        u = urllib.parse.urlsplit(str(url).strip())
        host = u.hostname
    except ValueError:
        return False
    if u.scheme not in ("http", "https") or not host:
        return False
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 4:
        return ip in ipaddress.ip_network("127.0.0.0/8")
    return ip == ipaddress.ip_address("::1")


def _endpoint_key(url: str) -> Optional[Tuple[str, str, Optional[int], str]]:
    """An endpoint's address as it is compared with a profile's: scheme, host, port and path, less a trailing slash
    and `/chat/completions`."""
    try:
        u = urllib.parse.urlsplit(str(url).strip())
        port = u.port
    except ValueError:
        return None
    if not u.hostname:
        return None
    path = u.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")].rstrip("/")
    return (u.scheme.lower(), u.hostname, port, path)


def _resolve_openai_profile(
    profile_name: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
) -> Tuple[str, Optional[str], bool]:
    """Resolve `(url, api_key, is_local)` from `backend/scientist_chat.py`'s endpoint profiles (keys are read from the
    environment, never stored on disk).

    - A profile's address is its own: a plan that names a profile cannot send it to another address.
    - An endpoint is on this machine only when its host is `localhost`, 127.0.0.0/8 or ::1 (`endpoint_is_local`).
    - A remote endpoint is reached only through a profile at that address, over https.
    - Any endpoint's key, on this machine or not, is read only from the variable a profile at its address names: a
      plan cannot pair another variable with an address (`GITHUB_TOKEN` for `http://127.0.0.1:9/v1` is refused), and
      an address no profile names a variable for is sent no key.
    - The address takes no user name, password, query or fragment."""
    from backend import scientist_chat as sc

    profs = sc.profiles()
    url = (base_url or "").strip() or None
    if profile_name:
        if profile_name not in profs:
            raise DriverRefused(f"Unknown OpenAI endpoint profile {profile_name!r} (available: {', '.join(sorted(profs))}).")
        p_url = str(profs[profile_name].get("base_url") or "")
        if url and _endpoint_key(url) != _endpoint_key(p_url):
            raise DriverRefused(f"Profile {profile_name!r} is at {p_url}: a plan cannot send it to another address ({url}).")
        url = p_url
    if not url:
        url = os.environ.get("THEODOLITE_OPENAI_BASE_URL", "").strip() or None
    if not url:
        raise DriverRefused("OpenAI driver requires --profile, base_url, or $THEODOLITE_OPENAI_BASE_URL.")
    try:
        u = urllib.parse.urlsplit(url)
        _ = u.port  # a port that is not a number is refused here
    except ValueError as exc:
        raise DriverRefused(f"The endpoint address {url!r} cannot be read ({exc}).") from None
    if u.scheme not in ("http", "https") or not u.hostname:
        raise DriverRefused(f"The endpoint address {url!r} must be an http(s) address with a host.")
    if u.username is not None or u.password is not None or u.query or u.fragment:
        raise DriverRefused(f"The endpoint address {url!r} takes no user name, password, query or fragment.")
    is_local = endpoint_is_local(url)
    key = _endpoint_key(url)
    at_url = {n: p for n, p in profs.items() if _endpoint_key(str(p.get("base_url") or "")) == key}
    envs = sorted({str(p["key_env"]) for p in at_url.values() if p.get("key_env")})
    if not is_local:
        if u.scheme != "https":
            raise DriverRefused(f"A remote endpoint is reached over https only, not {url}.")
        if not at_url:
            raise DriverRefused(
                f"{url} is not the address of any endpoint profile (backend/scientist_chat.py, or endpoints.json in the "
                "scientist folder): a remote endpoint is reached only through a profile, which names the variable "
                "its key is in."
            )
    if api_key_env and api_key_env not in envs:
        if envs:
            where = "$" + " or $".join(envs)
        elif at_url:
            where = "no variable (its profile names none, so it is sent no key)"
        else:
            where = "no variable (no endpoint profile is at this address, so it is sent no key)"
        raise DriverRefused(
            f"The key for {url} is read from {where}: a plan cannot pair another variable (${api_key_env}) with "
            "an endpoint's address, on this machine or not."
        )
    if not api_key_env and len(envs) > 1:
        raise DriverRefused(f"The profiles at {url} name different key variables (${', $'.join(envs)}): name one.")
    key_env = api_key_env or (envs[0] if envs else None)
    if not url.rstrip("/").endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    api_key = os.environ.get(key_env, "").strip() if key_env else ""
    return url, (api_key or None), is_local


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect is refused, never followed: following one would send the request, and its key, to another address."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise urllib.error.HTTPError(
            req.full_url, code,
            f"the endpoint answered {code} with a redirect to {newurl}; a redirect is refused, as following it would "
            "send the key to another address",
            headers, fp,
        )


def _openai_opener(url: str) -> urllib.request.OpenerDirector:
    """The opener for one request, built when it is sent: redirects refused, and
    - for an endpoint on this machine, no proxy at all, whatever the environment says: a proxy would be handed the
      request, its key among its headers, in the clear;
    - for a remote endpoint (https only), the environment's proxy, if one is set: through it the request, key and
      all, travels inside TLS."""
    if endpoint_is_local(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects)
    if urllib.parse.urlsplit(url).scheme != "https":
        raise DriverRefused(f"A remote endpoint is reached over https only, not {url}.")
    return urllib.request.build_opener(_RefuseRedirects)


def _default_openai_transport(payload: Dict[str, Any], headers: Dict[str, str], url: str, timeout_s: float) -> Dict[str, Any]:
    """POST `payload` to `url` and return the reply's JSON. An answer that is not a success raises with the
    endpoint's own words (a field it does not take, a limit it refuses), so the run fails loudly."""
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with _openai_opener(url).open(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            said = exc.read(2000).decode("utf-8", errors="replace").strip()
        except Exception:
            said = ""
        raise RuntimeError(f"the endpoint answered HTTP {exc.code} ({exc.reason})" + (f": {said}" if said else "")) from None


def _openai_prices(
    db: DriverBudget,
    input_usd_per_mtok: Optional[float],
    output_usd_per_mtok: Optional[float],
    *,
    is_local: bool,
    mocked: bool,
) -> Tuple[float, float]:
    """Dollars a million (input, output) tokens: the arm's own (its driver config), else the driver budget's. An
    endpoint on this machine costs nothing when none are given. A remote one with none is refused, and so is a mocked
    one under a dollar cap: at unknown prices the cap cannot be kept."""
    pin = input_usd_per_mtok if input_usd_per_mtok is not None else db.input_usd_per_mtok
    pout = output_usd_per_mtok if output_usd_per_mtok is not None else db.output_usd_per_mtok
    if pin is not None and pout is not None:
        pin_f, pout_f = float(pin), float(pout)
        if not (math.isfinite(pin_f) and math.isfinite(pout_f)) or pin_f < 0.0 or pout_f < 0.0:
            raise DriverRefused(f"Prices are dollars a million tokens, 0 or more (got {pin!r} and {pout!r}).")
        return pin_f, pout_f
    capped = db.max_dollars is not None and db.max_dollars > 0.0
    if (is_local and not mocked) or (mocked and not capped):
        return 0.0, 0.0
    raise DriverRefused(
        "The endpoint's prices are not known: give input_usd_per_mtok and output_usd_per_mtok (dollars a million "
        "tokens) in the arm's driver config or the driver budget. The dollar cap is counted from them, or from a "
        "reply's own cost when that is higher; it is never kept at a price of 0 that no one gave."
    )


def _await_reply(
    fn: Callable[[], Dict[str, Any]],
    ks: KillSwitch,
    deadline: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Make one blocking endpoint call in a thread, looking at the kill switch and the wall clock every 0.2 s:
    `(reply, None)`, or `(None, "killed" | "budget_wall")` as soon as either says stop (the call is left to time out
    on its own, and a reply that comes later is not read). An exception from the call is raised here."""
    box: Dict[str, Any] = {}

    def _run() -> None:
        try:
            box["reply"] = fn()
        except BaseException as exc:  # handed back to the caller's thread below
            box["exc"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while True:
        t.join(0.2)
        if not t.is_alive():
            break
        if ks.triggered():
            return None, "killed"
        if time.monotonic() >= deadline:
            return None, "budget_wall"
    if "exc" in box:
        raise box["exc"]
    return box.get("reply"), None


def _count(x: Any) -> Optional[int]:
    """A token count from a reply: a whole number 0 or more, or None."""
    return int(x) if isinstance(x, int) and not isinstance(x, bool) and x >= 0 else None


BUDGET_STOPS = ("budget_turns", "budget_tokens", "budget_dollars", "budget_wall")
CLAIM_TOOLS = ("claim", "claim_knob")


def _door_failure(
    door_ok: int,
    door_errors: int,
    *,
    stop_reason: str,
    denied: int = 0,
    other: Optional[str] = None,
) -> Optional[str]:
    """Why a driver could not work the door, in plain words, or None. A session whose door calls were denied, or none
    of whose door calls succeeded, ended by the driver's failure: never by an honest abstention. A run its own cap or
    STOP ended before it made any door call is not a driver's failure: its `stop_reason` says what stopped it."""
    if other:
        return other
    if denied:
        return f"{denied} tool call(s) were denied by the driver's own permissions, so it could not work the door."
    if door_ok > 0:
        return None
    if door_errors:
        return f"No door call succeeded ({door_errors} were made and every one failed): the driver did not work the door."
    if stop_reason in BUDGET_STOPS or stop_reason == "killed":
        return None
    return "No door call was made: the driver did not work the door."


def _refusal_words(text: Any) -> str:
    """A refused door call's own words, less the reply's "error: " mark."""
    s = str(text)
    return s[len("error: "):] if s.startswith("error: ") else s


def _door_counts(calls: int, errors: int) -> Dict[str, int]:
    return {"calls": int(calls), "ok": int(calls) - int(errors), "errors": int(errors)}


TEMPLATE_MARGIN_TOKENS = 1024  # tokens a chat template may add to a request's own bytes (its tool preamble, turn marks)
MAX_COMPLETION_PER_REPLY = 32_768  # the most one reply is let run to, whatever more the caps would allow
COMPLETION_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")  # what an endpoint calls a reply's token limit


def _json_bytes(obj: Any) -> int:
    """The UTF-8 bytes of `obj` written as JSON with its text unescaped, as an endpoint's tokenizer reads it."""
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def _profile_limit_field(profile_name: Optional[str], url: str) -> Optional[str]:
    """What the endpoint profile (the one named, else those at `url`) calls a reply's token limit, from its
    `completion_limit_field`, or None. Profiles that name two are refused: name one in the driver config."""
    from backend import scientist_chat as sc

    profs = sc.profiles()
    key = _endpoint_key(url)
    names = [profile_name] if profile_name else [n for n, p in profs.items()
                                                 if _endpoint_key(str(p.get("base_url") or "")) == key]
    fields = sorted({str(profs[n]["completion_limit_field"]) for n in names
                     if n in profs and profs[n].get("completion_limit_field")})
    if len(fields) > 1:
        raise DriverRefused(f"The profiles at {url} call a reply's token limit {' and '.join(fields)}: name one in the "
                            "driver config (completion_limit_field).")
    return fields[0] if fields else None


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
    input_usd_per_mtok: Optional[float] = None,
    output_usd_per_mtok: Optional[float] = None,
    completion_limit_field: Optional[str] = None,
    max_completion_per_reply: Optional[int] = None,
) -> Dict[str, Any]:
    """Run any OpenAI-compatible tool-calling model against `session`, keeping its turn, token, dollar and wall caps
    during the run, not after it.

    - The endpoint and its key are `_resolve_openai_profile`'s. A redirect is refused, and an endpoint on this
      machine is never reached through a proxy (`_openai_opener`).
    - What is spent is counted at no less than a floor of the harness's own: each request's prompt at the larger of
      the endpoint's count and the request's UTF-8 bytes / 4, each reply at the larger of the endpoint's count and
      the reply's bytes / 4; the dollars from those at known prices (`_openai_prices`), or the reply's own cost when
      that is higher. `usage` gives the endpoint's own figures (`provider_*`) beside the counted ones.
    - A request goes out only when it fits what is left, its prompt priced at an upper bound: its UTF-8 bytes and
      `TEMPLATE_MARGIN_TOKENS` (a token covers at least one byte, however the text tokenizes: a cell's float array,
      CJK), or the endpoint's count of the last prompt when that is larger. It carries a completion limit
      (`max_tokens`, or `completion_limit_field`) sized to what the tokens and dollars left allow, and at most
      `max_completion_per_reply`. A reply past that limit, or with no token usage, stops the run, as the caps cannot
      be kept; an endpoint that refuses the field fails the run in its own words.
    - A dollar cap of 0 is a cap: at prices above 0 nothing is sent. At prices of 0 nothing is paid and requests go
      out, and a reply that says it cost something stops the run.
    - STOP and the wall clock are looked at every 0.2 s while a request is out (`_await_reply`), and before each tool
      call a reply asks for.
    - A session none of whose door calls succeeded ends as `driver_failure` (`_door_failure`); one its own cap
      stopped before any door call does not: its `stop_reason` says which cap."""
    if not model or not isinstance(model, str):
        raise DriverRefused("OpenAI driver requires a non-empty model name.")
    db = driver_budget or DriverBudget()
    ks = kill_switch or KillSwitch()
    per_reply = MAX_COMPLETION_PER_REPLY if max_completion_per_reply is None else max_completion_per_reply
    if isinstance(per_reply, bool) or not isinstance(per_reply, int) or per_reply < 1:
        raise DriverRefused(f"max_completion_per_reply is a whole number of tokens, 1 or more, not {per_reply!r}.")
    if db.max_dollars is not None and not math.isfinite(float(db.max_dollars)):
        raise DriverRefused(f"max_dollars is a number of dollars, not {db.max_dollars!r}.")

    if transport_fn is None:
        url, api_key, is_local = _resolve_openai_profile(profile, base_url, api_key_env)
        if not is_local and (db.max_dollars is None or db.max_dollars <= 0.0):
            raise DriverRefused(
                "Remote OpenAI-compatible driver requires an explicit owner-approved max_dollars > 0 ceiling."
            )
        if completion_limit_field is None:  # the driver config's, else the one the endpoint's profile names
            completion_limit_field = _profile_limit_field(profile, url)
    else:
        url, api_key, is_local = "http://mock.local/v1/chat/completions", None, True
    limit_field = str(completion_limit_field or "max_tokens")
    if limit_field not in COMPLETION_LIMIT_FIELDS:
        raise DriverRefused(f"A reply's token limit is sent as {' or '.join(COMPLETION_LIMIT_FIELDS)}, not {limit_field!r}.")
    pin, pout = _openai_prices(db, input_usd_per_mtok, output_usd_per_mtok, is_local=is_local,
                               mocked=transport_fn is not None)
    capped = db.max_dollars is not None  # a cap of 0 is kept as one (see above)
    max_usd = float(db.max_dollars) if db.max_dollars is not None else math.inf
    max_tok = int(db.max_tokens)

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
    prompt_toks = 0  # counted: each request at the larger of the endpoint's count and its bytes / 4
    comp_toks = 0  # counted: each reply at the larger of the endpoint's count and its bytes / 4
    prov_pt = 0  # the endpoint's own counts, as it gave them
    prov_ct = 0
    prov_cost: Optional[float] = None
    reas_toks = 0
    last_pt = 0  # the endpoint's own count of the last prompt
    est_usd = 0.0
    tool_errors = 0
    door_ok = 0
    claim_refused: Optional[str] = None
    served_model = model
    stop_reason = "kept_base_abstention"
    stop_note = ""
    last_text = ""
    proposed_edit = None
    proposed_knob = None
    t0 = time.monotonic()
    deadline = t0 + float(db.max_wall_seconds)
    transport = transport_fn or _default_openai_transport
    halted = False

    while not halted:
        if ks.triggered():
            stop_reason = "killed"
            break
        if turns >= db.max_turns:
            stop_reason = "budget_turns"
            break
        if time.monotonic() >= deadline:
            stop_reason = "budget_wall"
            break
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": oa_tools,
            "tool_choice": "auto",
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        n_bytes = _json_bytes(payload)
        next_pt = max(last_pt, n_bytes + TEMPLATE_MARGIN_TOKENS)  # at most this many tokens, however it tokenizes
        spent = prompt_toks + comp_toks
        room = max_tok - spent - next_pt  # what the token cap leaves for the reply
        if room < 1:
            stop_reason = "budget_tokens"
            stop_note = (f"Stopped at the token cap before request {turns + 1}: {spent} tokens counted, and its prompt "
                         f"could be up to {next_pt} ({n_bytes} bytes), against max_tokens {max_tok}.")
            break
        if capped:
            usd_room = max_usd - est_usd - next_pt * pin / 1e6  # what the dollar cap leaves for the reply
            usd_comp = math.floor(usd_room * 1e6 / pout) if pout > 0.0 and usd_room >= 0.0 else None
            if usd_room < 0.0 or (usd_comp is not None and usd_comp < 1):
                stop_reason = "budget_dollars"
                stop_note = (f"Stopped at the dollar cap before request {turns + 1}: ${est_usd:.6f} counted, and its "
                             f"prompt could cost ${next_pt * pin / 1e6:.6f} ({next_pt} tokens at ${pin:g} a million) "
                             f"with nothing left for a reply, against max_dollars {max_usd:g}.")
                break
            if usd_comp is not None:
                room = min(room, usd_comp)
        comp_limit = int(min(room, per_reply))
        payload[limit_field] = comp_limit
        floor_pt = _json_bytes(payload) // 4  # the request as sent: the least it is counted at

        turns += 1
        rem_wall = max(5.0, deadline - time.monotonic())
        try:
            resp, why = _await_reply(lambda: transport(payload, headers, url, min(120.0, rem_wall)), ks, deadline)
        except Exception as exc:
            stop_reason = "error"
            stop_note = f"endpoint error: {exc}"
            break
        if why is not None:
            stop_reason = why
            prompt_toks += floor_pt  # it was sent, and may be paid for
            est_usd += floor_pt * pin / 1e6
            stop_note = (f"Stopped while request {turns} was out: its prompt is counted at its floor ({floor_pt} "
                         f"tokens); its reply, of up to {comp_limit} tokens, and what that cost were not read.")
            break

        reply = resp if isinstance(resp, Mapping) else {}
        if isinstance(reply.get("model"), str) and reply["model"]:
            served_model = str(reply["model"])
        usage = reply.get("usage") if isinstance(reply.get("usage"), Mapping) else {}
        choices = reply.get("choices") if isinstance(reply.get("choices"), list) else []
        msg_raw = choices[0].get("message") if choices and isinstance(choices[0], Mapping) else None
        msg: Dict[str, Any] = dict(msg_raw) if isinstance(msg_raw, Mapping) else {}
        pt = _count(usage.get("prompt_tokens"))
        ct = _count(usage.get("completion_tokens"))
        pt_n = max(pt or 0, floor_pt)
        ct_n = max(ct or 0, (_json_bytes(msg) // 4) if msg else 0)
        prompt_toks += pt_n
        comp_toks += ct_n
        prov_pt += pt or 0
        prov_ct += ct or 0
        last_pt = pt or 0
        details = usage.get("completion_tokens_details")
        reas_toks += (_count(details.get("reasoning_tokens")) or 0) if isinstance(details, Mapping) else 0
        cost = usage.get("cost")
        cost_ok = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0
        own_cost = float(cost) if cost_ok else 0.0
        if cost_ok:
            prov_cost = (prov_cost or 0.0) + own_cost
        est_usd += max((pt_n * pin + ct_n * pout) / 1e6, own_cost)

        if not isinstance(resp, Mapping):
            stop_reason = "error"
            stop_note = "The endpoint's reply is not a JSON object."
            break
        if pt is None or ct is None:
            stop_reason = "error"
            stop_note = "The endpoint's reply carried no token usage, so the token and dollar caps cannot be kept."
            break
        if ct > comp_limit:
            stop_reason = "error"
            stop_note = (f"The endpoint's reply ran to {ct} completion tokens, past the limit of {comp_limit} the "
                         f"request carried ({limit_field}): it does not keep the limit, so the caps cannot be kept.")
            break
        if not choices:
            stop_reason = "error"
            stop_note = "empty choices in response"
            break
        messages.append(msg)
        if isinstance(msg.get("content"), str) and msg["content"]:
            last_text = msg["content"]

        tcalls = msg.get("tool_calls") or []
        if not isinstance(tcalls, list) or not tcalls:
            if stop_reason not in ("claimed", "proposed_repair"):
                stop_reason = "kept_base_abstention"
            break

        for i, tc in enumerate(tcalls):
            if ks.triggered():
                stop_reason, halted = "killed", True
                break
            if time.monotonic() >= deadline:
                stop_reason, halted = "budget_wall", True
                stop_note = (f"Stopped at the wall cap between tool calls: {len(tcalls) - i} of the {len(tcalls)} the "
                             "last reply asked for were not made.")
                break
            tcm = tc if isinstance(tc, Mapping) else {}
            fn = tcm.get("function") if isinstance(tcm.get("function"), Mapping) else {}
            tname = str(fn.get("name") or "")
            raw_args = fn.get("arguments") or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(parsed_args, Mapping):
                    raise ValueError("the arguments are not a JSON object")
            except (ValueError, TypeError) as exc:
                tool_errors += 1
                out_str = f"error: invalid JSON arguments: {exc}"
                if tname in CLAIM_TOOLS:
                    claim_refused = f"The {tname} call's arguments could not be read ({exc}): it did not reach the gate."
            else:
                ok, res = _call(session, tname, parsed_args)
                if not ok:
                    tool_errors += 1
                    out_str = f"error: {res}"
                    if tname in CLAIM_TOOLS:
                        claim_refused = _refusal_words(res)
                else:
                    door_ok += 1
                    if tname == "claim":
                        proposed_edit = parsed_args.get("edit")
                        stop_reason = "claimed"
                    elif tname == "claim_knob":
                        proposed_knob = parsed_args.get("knob")
                        stop_reason = "claimed"
                    elif tname == "propose_repair":
                        ed_arg = parsed_args.get("edit")
                        proposed_edit = dict(ed_arg) if isinstance(ed_arg, Mapping) else None
                        stop_reason = "proposed_repair" if proposed_edit is not None else "kept_base_abstention"
                    out_str = json.dumps(res, ensure_ascii=True)
                    if len(out_str) > 16000:
                        out_str = out_str[:16000] + "... [truncated]"
            messages.append({
                "role": "tool",
                "tool_call_id": str(tcm.get("id") or f"call_{turns}"),
                "content": out_str,
            })

    other = None
    if stop_reason == "error" and door_ok == 0 and tool_errors == 0:
        other = f"The endpoint failed before any door call was made: {stop_note}"
    failure = _door_failure(door_ok, tool_errors, stop_reason=stop_reason, other=other)
    if failure and stop_reason in ("kept_base_abstention", "error"):
        stop_reason = "driver_failure"
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
        notes="\n".join(t for t in (last_text, stop_note) if t),
        proposed_edit=proposed_edit,
        proposed_knob=proposed_knob,
        driver_failure=failure,
        claim_refused=claim_refused,
        door={"calls": door_ok + tool_errors, "ok": door_ok, "errors": tool_errors,
              "prices_usd_per_mtok": {"input": pin, "output": pout},
              "completion_limit": {"field": limit_field, "most_a_reply": per_reply}},
        extra_usage={
            "provider_prompt_tokens": prov_pt,
            "provider_completion_tokens": prov_ct,
            "provider_total_tokens": prov_pt + prov_ct,
            "provider_cost_usd": round(prov_cost, 6) if prov_cost is not None else None,
            "tokens_counted_as": "each request and each reply at the larger of the endpoint's count and its UTF-8 bytes / 4",
        },
    )


# -------------------------------------------------------------------------------------- Claude CLI & Codex CLI

DOOR_SERVER = "levelling"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")  # `claude --effort` (2.1.328 --help)
MAX_CLI_LINE_BYTES = 32 << 20  # a line of the CLI's stream-json longer than this stops the run
# What Claude Code needs to run and to sign in, passed by name from this process's environment; nothing else passes
# (no endpoint key, token, cloud credential, book path or other variable of ours). Claude Code reaching its model
# through Bedrock, Vertex or Foundry is not passed: their credentials have many names.
CLAUDE_ENV_PASS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TZ", "TERM", "TMPDIR",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "CLAUDE_CONFIG_DIR",
    # its sign-in (an API key, an auth token or a long-lived OAuth token; or its login under HOME) and its address
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
    # the way out of this machine, when it goes through a proxy, and the certificates that proxy signs with
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "no_proxy", "all_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
})
CLAUDE_ENV_SET = {
    # no auto-memory folder read into the context or written: under a checkout it is the owner's own project memory
    # (a MEMORY.md there reached the request on 2.1.328 until this, or --restricted, was set)
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",  # no CLAUDE.md from the home folder or any folder above the work folder
    "ENABLE_TOOL_SEARCH": "false",  # no deferred-tool placeholder in the request: the model's tools are the door's alone
    "DISABLE_AUTOUPDATER": "1",  # the CLI does not change its version in the middle of a study
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",  # no telemetry, error report or update check leaves the machine
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
}


def claude_child_env() -> Dict[str, str]:
    """The environment `claude` starts with: an allowlist of this process's (`CLAUDE_ENV_PASS`, and `LC_*`), never
    the whole of it, and the switches `CLAUDE_ENV_SET`. The door's MCP child (`python -I level_mcp.py --socket`)
    needs nothing of it: -I reads no PYTHON* variable, and the socket's path is on its command line."""
    env = {k: v for k, v in os.environ.items() if k in CLAUDE_ENV_PASS or k.startswith("LC_")}
    env.update(CLAUDE_ENV_SET)
    return env


def claude_cli_args(
    exe: str,
    prompt_info: Mapping[str, Any],
    door_tools: Sequence[str],
    mcp_config: Path,
    *,
    model: Optional[str] = None,
    effort: Optional[str] = None,
    max_turns: int = 24,
    max_budget_usd: Optional[float] = None,
) -> List[str]:
    """The `claude` command line of a driver session; each flag is as `claude --help` (2.1.328) words it."""
    cmd = [
        exe, "-p",  # print mode: one prompt, its answer, then exit
        "--output-format", "stream-json", "--verbose",  # one JSON event a line (stream-json in -p mode needs --verbose)
        "--system-prompt", str(prompt_info["system"]),  # the driver's page in place of Claude Code's own prompt
        "--tools", "",  # "" for none: no built-in tool (no Read, Write, Edit, Bash, WebFetch, WebSearch, Skill)
        "--allowedTools", ",".join(door_tools),  # exactly the session's door tools, run without asking
        "--permission-mode", "dontAsk",  # "deny if not pre-approved": a call outside the list above is refused
        "--permission-prompts", "none",  # nobody answers a prompt: anything that would ask is denied
        "--strict-mcp-config", "--mcp-config", str(mcp_config),  # the door's MCP server and no other
        "--setting-sources", "",  # no user, project or local settings file, so no hook, rule or CLAUDE.md from one
        "--restricted",  # auto-memory off, settings files ignored, bypassPermissions refused
        "--disable-slash-commands",  # no skill or command, from the prompt or the model
        "--no-session-persistence",  # no transcript saved under ~/.claude/projects
        "--max-turns", str(int(max_turns)),
    ]
    if max_budget_usd is not None:
        cmd += ["--max-budget-usd", f"{float(max_budget_usd):.6f}"]  # the CLI stops itself once its spend passes it
    if model:
        cmd += ["--model", str(model)]
    if effort:
        cmd += ["--effort", str(effort)]
    return cmd + ["--", str(prompt_info["user"])]  # the prompt last, after "--", so no word of it is read as a flag


def _json_line(raw: bytes) -> Optional[Dict[str, Any]]:
    try:
        ev = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    return ev if isinstance(ev, dict) else None


class _ClaudeStream:
    """What a `claude -p --output-format stream-json` run says of itself, read line by line."""

    def __init__(self, door_tools: Sequence[str]) -> None:
        self.door = set(door_tools)
        self.offered: Optional[List[str]] = None  # the tools its init line says the model has
        self.mcp_status: Optional[str] = None
        self.model: Optional[str] = None
        self.msg_tokens: Dict[str, Tuple[int, int]] = {}  # (prompt, completion) tokens of each model reply
        self.door_uses = 0
        self.other_uses: List[str] = []  # tools the model asked for that are not the door's
        self.tool_errors = 0  # tool results that came back as errors, refusals of other tools among them
        self.result: Optional[Dict[str, Any]] = None

    def feed(self, ev: Mapping[str, Any]) -> None:
        etype = ev.get("type")
        if etype == "system" and ev.get("subtype") == "init":
            if isinstance(ev.get("tools"), list):
                self.offered = [str(t) for t in ev["tools"]]
            for srv in ev.get("mcp_servers") or []:
                if isinstance(srv, Mapping) and srv.get("name") == DOOR_SERVER:
                    self.mcp_status = str(srv.get("status"))
            if isinstance(ev.get("model"), str) and ev["model"]:
                self.model = str(ev["model"])
        elif etype == "assistant":
            msg = ev.get("message") or {}
            if isinstance(msg.get("model"), str) and msg["model"]:
                self.model = str(msg["model"])
            mid = str(msg.get("id") or f"reply_{len(self.msg_tokens)}")
            u = msg.get("usage") or {}
            pt = sum(_count(u.get(k)) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            ct = _count(u.get("output_tokens")) or 0
            seen = self.msg_tokens.get(mid, (0, 0))  # a reply's blocks can come as several lines with one usage
            self.msg_tokens[mid] = (max(seen[0], pt), max(seen[1], ct))
            for b in msg.get("content") or []:
                if isinstance(b, Mapping) and b.get("type") == "tool_use":
                    name = str(b.get("name") or "")
                    if name in self.door:
                        self.door_uses += 1
                    else:
                        self.other_uses.append(name)
        elif etype == "user":
            content = (ev.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if isinstance(b, Mapping) and b.get("type") == "tool_result" and b.get("is_error"):
                    self.tool_errors += 1
        elif etype == "result":
            self.result = dict(ev)

    def turns(self) -> int:
        n = len(self.msg_tokens)
        if self.result is not None:
            n = max(n, _count(self.result.get("num_turns")) or 0)
        return n

    def tokens(self) -> Tuple[int, int]:
        pt = sum(v[0] for v in self.msg_tokens.values())
        ct = sum(v[1] for v in self.msg_tokens.values())
        if self.result is not None:
            u = self.result.get("usage") or {}
            pt = max(pt, sum(_count(u.get(k)) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")))
            ct = max(ct, _count(u.get("output_tokens")) or 0)
        return pt, ct

    def denials(self) -> List[Any]:
        d = (self.result or {}).get("permission_denials")
        return list(d) if isinstance(d, list) else []

    def extra_tools(self) -> List[str]:
        return sorted(set(self.offered or ()) - self.door)


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
    """Launch Claude Code (`claude -p`, stream-json) against `session` with the session's door tools and nothing else
    (the owner's ruling, "Door tools only": free code goes through the door's `cell`, inside bwrap).

    - The command line is `claude_cli_args`: no built-in tool; the door's tools approved and anything else denied; no
      settings, hook, CLAUDE.md, auto-memory, slash command or saved transcript; `--max-budget-usd` at the session's
      `max_dollars` (what the study has left). Its environment is the allowlist `claude_child_env`.
    - The run is read as it goes, and stopped (its process group SIGKILLed) at the wall clock, the token cap, the
      dollar cap (when prices are given), STOP, or when the CLI offers the model a tool that is not the door's.
    - The door's bridge (`level_mcp.session_socket_bridge`) is closed before the outcome is read: a cell still
      running is killed and any other call still running is waited for until it ends, with no bound, so the book
      holds what the session did. The outcome then says how long after the CLI ended it was read, and why
      (`door["wait_note"]`, in `notes`).
    - A session whose door calls were denied, or none of whose door calls succeeded, ends as `driver_failure`, never
      as an honest abstention (`_door_failure`). The claim's edit is `proposed_edit`, as the door received it; a
      claim the door refused is `claim_refused`, in its own words.
    - Refused at `$THEODOLITE_SCIENTIST_REACH=home`, and without an owner-approved `max_dollars > 0` (unless
      `allow_mock_cli=True` in CPU checks)."""
    if os.environ.get(lses.REACH_ENV, "").strip().lower() == "home":
        raise DriverRefused(f"Claude CLI driver is refused at the home reach (${lses.REACH_ENV}=home).")
    db = driver_budget or DriverBudget()
    if not allow_mock_cli and (db.max_dollars is None or db.max_dollars <= 0.0):
        raise DriverRefused("Claude CLI driver requires an explicit owner-approved max_dollars > 0 ceiling.")
    budget_usd: Optional[float] = None
    if db.max_dollars is not None and db.max_dollars > 0.0:
        budget_usd = round(float(db.max_dollars), 6)
        if budget_usd <= 0.0:
            raise DriverRefused(f"{db.max_dollars!r} dollars is less than the CLI can be given (--max-budget-usd).")
    if reasoning_effort is not None and reasoning_effort not in CLAUDE_EFFORTS:
        raise DriverRefused(f"Claude Code's effort is one of {', '.join(CLAUDE_EFFORTS)}, not {reasoning_effort!r}.")
    exe = claude_bin or os.environ.get("THEODOLITE_CLAUDE_BIN", "claude")
    if not allow_mock_cli and shutil.which(exe) is None and not Path(exe).is_file():
        raise DriverRefused(f"Claude CLI executable {exe!r} was not found on PATH.")

    ks = kill_switch or KillSwitch()
    prompt_info = build_driver_prompt(session, extra_instructions=extra_instructions)
    door_tools = [f"mcp__{DOOR_SERVER}__{t}" for t in session.allowed_tools()]
    mcp_cfg_path = session.session_dir / "mcp_config.json"
    stream = _ClaudeStream(door_tools)
    priced = db.input_usd_per_mtok is not None and db.output_usd_per_mtok is not None
    stop_reason: Optional[str] = None
    stop_note = ""

    with lmcp.session_socket_bridge(session) as bridge:
        mcp_cfg = {"mcpServers": {DOOR_SERVER: {
            "command": sys.executable,
            "args": ["-I", str(ROOT_DIR / "backend" / "level_mcp.py"), "--socket", str(bridge.path)],
        }}}
        mcp_cfg_path.write_text(json.dumps(mcp_cfg, indent=2) + "\n", encoding="utf-8")
        cmd = claude_cli_args(exe, prompt_info, door_tools, mcp_cfg_path, model=model, effort=reasoning_effort,
                              max_turns=db.max_turns, max_budget_usd=budget_usd)
        stderr_log = open(session.session_dir / "cli_stderr.log", "w", encoding="utf-8")
        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(session.work_dir),
            env=claude_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            buf = b""
            while True:
                pt, ct = stream.tokens()
                if ks.triggered():
                    stop_reason = "killed"
                elif (time.monotonic() - t0) >= db.max_wall_seconds:
                    stop_reason = "budget_wall"
                elif pt + ct >= db.max_tokens:
                    stop_reason = "budget_tokens"
                elif priced and budget_usd is not None and (pt * float(db.input_usd_per_mtok or 0.0)
                                                           + ct * float(db.output_usd_per_mtok or 0.0)) / 1e6 >= budget_usd:
                    stop_reason = "budget_dollars"
                elif stream.extra_tools():
                    stop_reason = "driver_failure"
                    stop_note = (f"The CLI offered the model tools that are not the door's "
                                 f"({', '.join(stream.extra_tools()[:12])}): the session was stopped.")
                elif len(buf) > MAX_CLI_LINE_BYTES:
                    stop_reason = "error"
                    stop_note = f"A line of the CLI's output passed {MAX_CLI_LINE_BYTES >> 20} MB."
                if stop_reason is not None:
                    _kill_pgrp(proc)
                    break
                rlist, _, _ = select.select([fd], [], [], 0.2)
                if not rlist:
                    if proc.poll() is not None:
                        break
                    continue
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    ev = _json_line(raw)
                    if ev is not None:
                        stream.feed(ev)
            ev = _json_line(buf) if stop_reason is None else None
            if ev is not None:
                stream.feed(ev)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:  # its output closed but it lives on: it is killed below
                pass
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
        t_cli_end = time.monotonic()
    door = bridge.close()  # closed as the block ended; this is its report (what ran, what was running at the stop)
    waited_after_cli = round(time.monotonic() - t_cli_end, 3)

    res = stream.result
    if stop_reason is None:
        subtype = str((res or {}).get("subtype") or "")
        if res is None:
            stop_reason, stop_note = "error", "The CLI ended without its result line."
        elif subtype == "error_max_budget_usd":
            stop_reason = "budget_dollars"
        elif subtype == "error_max_turns":
            stop_reason = "budget_turns"
        elif res.get("is_error"):
            stop_reason = "error"
        else:
            stop_reason = "kept_base_abstention"

    denied = len(stream.denials())
    other = stop_note if stop_reason == "driver_failure" else None
    if other is None and stream.mcp_status not in (None, "connected", "pending"):
        other = f"The door's MCP server did not connect (its status: {stream.mcp_status})."
    failure = _door_failure(int(door["ok"]), int(door["errors"]), stop_reason=stop_reason, denied=denied, other=other)
    if failure and stop_reason in ("kept_base_abstention", "error"):
        stop_reason = "driver_failure"
    # How long the outcome came after the CLI ended, and why: the door waits for a call running at the stop until
    # it ends, with no bound (a cell is killed; a reading is let finish, so its lines are in the book)
    wait_note = None
    if door.get("wait_note"):
        wait_note = (f"The CLI ended {t_cli_end - t0:.1f} s into the session ({stop_reason}), and the outcome was read "
                     f"{waited_after_cli:g} s after that. {door['wait_note']}")

    proposed_edit = None
    proposed_knob = None
    claim_refused: Optional[str] = None
    for c in bridge.calls:
        if c.get("ok") and c.get("tool") == "claim":
            proposed_edit = c.get("proposed")
        elif c.get("ok") and c.get("tool") == "claim_knob":
            proposed_knob = c.get("proposed")
        elif c.get("ok") and c.get("tool") == "propose_repair":
            proposed_edit = c.get("proposed")
        elif c.get("ok") is False and c.get("tool") in CLAIM_TOOLS:
            claim_refused = _refusal_words(c.get("error") or "")

    pt, ct = stream.tokens()
    cost = (res or {}).get("total_cost_usd")
    est_usd = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0 else 0.0
    if priced:
        est_usd = max(est_usd, (pt * float(db.input_usd_per_mtok or 0.0) + ct * float(db.output_usd_per_mtok or 0.0)) / 1e6)
    notes = [str(res["result"])] if res is not None and isinstance(res.get("result"), str) else []
    notes += [t for t in dict.fromkeys((stop_note, failure, wait_note)) if t]
    return _extract_outcome_from_book(
        session,
        driver_family="claude",
        driver_model=stream.model or model or "claude-cli",
        face="mcp_stdio",
        reasoning_effort=reasoning_effort,
        prompt_info=prompt_info,
        driver_budget=db,
        stop_reason=stop_reason,
        turns=stream.turns(),
        prompt_tokens=pt,
        completion_tokens=ct,
        reasoning_tokens=0,
        estimated_dollars=est_usd,
        tool_errors=max(stream.tool_errors, int(door["errors"])),
        notes="\n".join(notes),
        proposed_edit=proposed_edit,
        proposed_knob=proposed_knob,
        driver_failure=failure,
        claim_refused=claim_refused,
        door={**door, "denied": denied, "offered_tools": stream.offered, "other_tools_asked": stream.other_uses[:20],
              "door_uses_seen": stream.door_uses, "mcp_status": stream.mcp_status,
              "cli_ended_at_seconds": round(t_cli_end - t0, 3), "outcome_after_cli_seconds": waited_after_cli},
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
            input_usd_per_mtok=cfg.get("input_usd_per_mtok"),
            output_usd_per_mtok=cfg.get("output_usd_per_mtok"),
            completion_limit_field=cfg.get("completion_limit_field"),
            max_completion_per_reply=cfg.get("max_completion_per_reply"),
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
