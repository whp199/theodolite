"""The agent's door, practice only (`backend/level_session.py`): session controller, containment policy, budgets,
transcript logging, and model gates for Levelling Phase 2.

Design contract (`experiments/results/weight_space_exploration/verify/arbiter.md` §5, §8, rows 27, 30-34):
- Exploring vs claiming: this module is PRACTICE ONLY. It returns ranks, floors, written loop/cut-off summaries,
  and `level_book.driver_view` aggregates — never a word past a rank.
- Off-card computation: Python cells execute off the GPU inside `backend/level_sandbox.py` (`bwrap`), with read-only
  weights (`/weights`, omitted in `"sandbox_no_weights"` reach) and one writable session folder (`/work`).
- Model gates: `capture`, `read` (paired), and `write` (written) are the only paths to the model. Every call checks
  `weight_edits.assert_base` before and after, adds/enforces controls and null rows through `backend/levelling.py`,
  writes the chained ledger outside `/work`, and returns `level_book.driver_view` aggregates.
- Refusals: every levelling tool is refused at the `"home"` reach; `"codex"` is refused until its real sandbox has
  been probed and the compile cache moved.
- Planted faults & canaries: an optional in-memory `plant` (`weight_edits.Edit`) or `canary=True` flag lives in
  process memory ONLY — never written to disk, never left on the model (`assert_base` passes before and after every
  call), and blinded so planted and canary sessions present identical harness cards and tool lists.
- Closing a session copies its transcript into `<book_dir>/transcripts/<session_id>.jsonl`.

Layers count from 0 in code and files; human-facing strings give both ("layer 20 (page layer 21)").
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from backend import level_book as lb
from backend import level_briefs as lbr
from backend import level_claim as lcl
from backend import level_sandbox as ls
from backend import levelling as lv
from backend import weight_edits as we
from backend.weight_reader import save_array as save_f32_array

SESSION_VERSION = 1
REACH_ENV = "THEODOLITE_SCIENTIST_REACH"
VALID_REACHES = ("sandbox", "sandbox_no_weights", "tools_only", "attended")
VALID_BACKENDS = ("claude", "openai", "local", "script", "codex")
# What the owner may fix about a session's fresh readings when the session is made (`claim_design`); the names are
# `level_claim.run_fresh_claim`'s own. Nothing here is the driver's to set: the claim tool takes none of them.
CLAIM_DESIGN_KEYS = ("n_target", "n_neutral", "n_near", "n_panel", "rows", "written_rows", "controls_override",
                     "match_tolerance")
PRACTICE_NOTE = "Practice door only: returns aggregates and ranks against control edits, no word past a rank."


class SessionRefused(ValueError):
    """A session policy, reach, budget, or gate refusal in plain words."""


@dataclass(frozen=True)
class SessionBudget:
    """Hard caps for one agent practice session."""

    max_gpu_seconds: float = 600.0
    max_forwards: int = 200
    max_cells: int = 100
    max_cell_seconds: float = ls.DEFAULT_TIMEOUT_S
    max_wall_seconds: float = 1800.0
    max_cell_mem_gib: float = ls.DEFAULT_MEM_GIB
    min_controls: int = 3
    min_written_controls: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_gpu_seconds": float(self.max_gpu_seconds),
            "max_forwards": int(self.max_forwards),
            "max_cells": int(self.max_cells),
            "max_cell_seconds": float(self.max_cell_seconds),
            "max_wall_seconds": float(self.max_wall_seconds),
            "max_cell_mem_gib": float(self.max_cell_mem_gib),
            "min_controls": int(self.min_controls),
            "min_written_controls": int(self.min_written_controls),
        }


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_session_id() -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"lvl_{stamp}_{secrets.token_hex(4)}"


def _compile_cache_is_private() -> bool:
    """Whether `torch.compile`'s cache is outside `/tmp` (moved by `runtime_factory.private_compile_cache`)."""
    cur = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "").strip()
    if not cur:
        return False
    ap = os.path.abspath(cur)
    return not (ap == "/tmp" or ap.startswith("/tmp/"))


def _check_reach_and_backend(
    reach: str,
    backend: str,
    *,
    codex_probed: bool = False,
    ignore_reach_env: bool = False,
) -> None:
    """Enforce the arbiter's reach and backend containment refusals."""
    env_reach = os.environ.get(REACH_ENV, "").strip().lower()
    if not ignore_reach_env and env_reach == "home":
        raise SessionRefused(
            f"Every levelling tool is refused at the home reach (${REACH_ENV}=home). "
            "Use the folder reach or an isolated sandbox session."
        )
    if reach == "home":
        raise SessionRefused("Every levelling tool is refused at the home reach.")
    if reach not in VALID_REACHES:
        raise SessionRefused(f"Reach must be one of {', '.join(VALID_REACHES)}; {reach!r} is not.")
    if backend not in VALID_BACKENDS:
        raise SessionRefused(f"Backend must be one of {', '.join(VALID_BACKENDS)}; {backend!r} is not.")
    if backend == "codex":
        if not codex_probed or not _compile_cache_is_private():
            raise SessionRefused(
                "Codex is refused for levelling sessions until its real sandbox has been probed "
                "(codex_probed=True) and the torch.compile cache has been moved out of /tmp."
            )


def _as_item_map(
    items: Any,
    tokenizer: Any,
) -> Dict[str, lv.Item]:
    """Accept either a mapping of `{name: lv.Item}` or JSON-style `{name: {question, answer, system?}}`."""
    if not isinstance(items, Mapping) or not items:
        raise SessionRefused("items must be a non-empty mapping of name -> Item or {question, answer, system?}.")
    out: Dict[str, lv.Item] = {}
    for name, raw in items.items():
        if not isinstance(name, str) or not name:
            raise SessionRefused("Every item name must be a non-empty string.")
        if isinstance(raw, lv.Item):
            out[name] = raw
        elif isinstance(raw, Mapping):
            q = raw.get("question")
            a = raw.get("answer")
            sys_turn = raw.get("system")
            try:
                out[name] = lv.make_item(tokenizer, q, a, system=sys_turn, name=name)
            except lv.Refused as exc:
                raise SessionRefused(f'Item "{name}": {exc}') from None
        else:
            raise SessionRefused(f'Item "{name}" must be a levelling.Item or {{"question", "answer"}}.')
    return out


def _as_item_list(items: Any, tokenizer: Any, label: str) -> List[lv.Item]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
        raise SessionRefused(f"{label} must be a non-empty list of Item or {{question, answer}} objects.")
    out: List[lv.Item] = []
    for i, raw in enumerate(items):
        if isinstance(raw, lv.Item):
            out.append(raw)
        elif isinstance(raw, Mapping):
            try:
                out.append(lv.make_item(
                    tokenizer, raw.get("question"), raw.get("answer"),
                    system=raw.get("system"), name=raw.get("name") or f"{label}_{i}",
                ))
            except lv.Refused as exc:
                raise SessionRefused(f"{label}[{i}]: {exc}") from None
        else:
            raise SessionRefused(f"{label}[{i}] must be a levelling.Item or {{question, answer}}.")
    return out


class LevelSession:
    """Budgeted, transcript-logged Phase 2 practice session gating off-card cells and model readings."""

    def __init__(
        self,
        handle: Optional[lv.Handle],
        session_dir: Path,
        *,
        reach: str = "sandbox",
        backend: str = "local",
        budget: Optional[SessionBudget] = None,
        book_path: Optional[Path] = None,
        weights_dir: Optional[Path] = None,
        atlas_dir: Optional[Path] = None,
        bwrap_bin: str = "bwrap",
        probe_on_start: bool = True,
        codex_probed: bool = False,
        ignore_reach_env: bool = False,
        gpu_lock: Optional[Callable[[], contextlib.AbstractContextManager[None]]] = None,
        plant: Optional[we.Edit] = None,
        canary: bool = False,
        brief: Optional[lbr.Brief] = None,
        bars: Optional[lcl.BarSet] = None,
        claim_design: Optional[Mapping[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        _check_reach_and_backend(reach, backend, codex_probed=codex_probed, ignore_reach_env=ignore_reach_env)
        design = dict(claim_design or {})
        unknown = sorted(k for k in design if k not in CLAIM_DESIGN_KEYS)
        if unknown:
            raise SessionRefused(f"claim_design takes {', '.join(CLAIM_DESIGN_KEYS)}; not {', '.join(unknown)}.")
        if design and (brief is None or bars is None):
            raise SessionRefused("claim_design is for a session with a bound brief and bar set.")
        if plant is not None and canary:
            raise SessionRefused("A session cannot be both planted and a canary at the same time.")
        if bars is not None and brief is None:
            raise SessionRefused("A BarSet requires a matching Brief to be bound to the session.")
        if brief is not None and bars is not None and bars.brief_id != brief.id:
            raise SessionRefused(f"BarSet is for brief {bars.brief_id[:12]}, not {brief.short()}.")
        if brief is not None and bars is not None and handle is not None and plant is None and not canary:
            # the claim tool is offered only with bars a calibration reading wrote into this session's own book
            bk_now = Path(book_path).resolve() if book_path is not None else lb.book_path().resolve()
            if lcl.calibration_note(bars, bk_now) is None:
                raise SessionRefused(
                    f"Bar set {bars.short()} is not from a calibration reading recorded in this session's level book "
                    f'for brief "{brief.name}": calibrate on the same book first (scripts/levelling.py calibrate).'
                )
        if plant is not None:
            if not isinstance(plant, we.Edit):
                raise SessionRefused("plant must be a weight_edits.Edit or None.")
            if handle is not None:
                lv._same_base(handle, plant)
                plant = we.gate_again(handle.lm, plant, model_id=handle.model_id, weights_id=handle.weights_id)

        self.session_id = session_id or _new_session_id()
        self.reach = reach
        self.backend = backend
        self.budget = budget or SessionBudget()
        self._handle = handle
        self._gpu_lock = gpu_lock
        self._thread_lock = threading.Lock()
        self._closed = False
        self._brief: Optional[lbr.Brief] = brief
        self._bars: Optional[lcl.BarSet] = bars
        self._claim_design: Dict[str, Any] = {k: v for k, v in design.items() if v is not None}
        # Kept-back frames are kept from practice here, at the gates to the model: those of every starter brief (their
        # text is in the published source, and a session with no brief bound may be practising for one) and of the
        # bound brief.
        self._kept_back = lbr.KeptBack(
            list(lbr.starter_briefs(handle.tokenizer if handle is not None else None).values())
            + ([brief] if brief is not None else [])
        )
        self._practice_reads: int = 0

        # In-memory-only plant / canary state: NEVER written to session.json, work/, or transcript.jsonl.
        self._plant: Optional[we.Edit] = plant
        self._canary: bool = bool(canary)
        self._blinded: bool = (plant is not None) or bool(canary)

        # Layout: session_dir holds work/ (the ONLY folder mounted writable into bwrap), cells/, transcript.jsonl, session.json.
        self.session_dir = Path(session_dir).resolve()
        self.work_dir = self.session_dir / "work"
        self.cells_dir = self.session_dir / "cells"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cells_dir.mkdir(parents=True, exist_ok=True)

        # Book MUST lie outside work_dir so sandboxed cells can never read or tamper with it!
        self.book_path = Path(book_path).resolve() if book_path is not None else lb.book_path().resolve()
        if str(self.book_path).startswith(str(self.work_dir) + os.sep) or self.book_path == self.work_dir:
            raise SessionRefused("The level book cannot live inside the session's writable work folder.")

        self._transcript_path = self.session_dir / "transcript.jsonl"
        self._started_at = _utc_now()
        self._t_start = time.monotonic()

        # Usage counters
        self.gpu_seconds_used: float = 0.0
        self.forwards_used: int = 0
        self.cells_used: int = 0
        self.cell_seconds_used: float = 0.0
        self._cell_seq: int = 0
        self._book_seqs: List[int] = []

        # Sandbox setup (for sandbox reaches only)
        self._sandbox_cfg: Optional[ls.SandboxConfig] = None
        self._sandbox_probe: Optional[Dict[str, Any]] = None
        if reach in ("sandbox", "sandbox_no_weights"):
            effective_weights = weights_dir if reach == "sandbox" else None
            self._sandbox_cfg = ls.SandboxConfig(
                work_dir=self.work_dir,
                weights_dir=Path(effective_weights).resolve() if effective_weights is not None else None,
                atlas_dir=Path(atlas_dir).resolve() if atlas_dir is not None else None,
                timeout_s=float(self.budget.max_cell_seconds),
                max_mem_gib=float(self.budget.max_cell_mem_gib),
                bwrap_bin=bwrap_bin,
                model_id=handle.model_id if handle is not None else None,
                weights_id=handle.weights_id if handle is not None else None,
            )
            if probe_on_start:
                self._sandbox_probe = ls.probe_sandbox(
                    weights_dir=self._sandbox_cfg.weights_dir,
                    atlas_dir=self._sandbox_cfg.atlas_dir,
                    bwrap_bin=bwrap_bin,
                    max_mem_gib=min(2.0, float(self.budget.max_cell_mem_gib)),
                )
                if not self._sandbox_probe.get("ok"):
                    failed = [c["name"] for c in self._sandbox_probe.get("checks", []) if not c.get("ok")]
                    raise SessionRefused(f"Off-card sandbox containment probe failed ({', '.join(failed)}): "
                                         "refusing to start a sandboxed session.")

        # Persist non-secret session metadata (NEVER writes _plant or _canary!)
        meta = {
            "v": SESSION_VERSION,
            "session_id": self.session_id,
            "started_at": self._started_at,
            "reach": self.reach,
            "backend": self.backend,
            "with_healthy_reference": bool(self._sandbox_cfg is not None and self._sandbox_cfg.weights_dir is not None),
            "budget": self.budget.to_dict(),
            "allowed_tools": self.allowed_tools(),
            "book_path": str(self.book_path),
            "model_id": handle.model_id if handle is not None else None,
            "weights_id": handle.weights_id if handle is not None else None,
            "note": PRACTICE_NOTE,
        }
        (self.session_dir / "session.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self._log("session_start", {"harness_card": self.harness_card()})

    # --------------------------------------------------------------------------------------------- policy & state

    def allowed_tools(self) -> List[str]:
        """Filtered tool list named on the harness card."""
        tools = ["status", "estimate", "book_view"]
        if self.reach in ("sandbox", "sandbox_no_weights"):
            tools.append("cell")
        if self._handle is not None:
            tools += ["capture", "read", "write"]
        if self._brief is not None:
            tools += ["brief_info", "draw_practice"]
            if self._bars is not None and self._handle is not None and not self._blinded:
                tools.append("claim")
        return tools

    def harness_card(self) -> Dict[str, Any]:
        """Public harness card for this session (never discloses whether an in-memory plant or canary is set)."""
        has_claim = bool(self._brief is not None and self._bars is not None and self._handle is not None and not self._blinded)
        card: Dict[str, Any] = {
            "v": SESSION_VERSION,
            "session_id": self.session_id,
            "reach": self.reach,
            "backend": self.backend,
            "with_healthy_reference": bool(self._sandbox_cfg is not None and self._sandbox_cfg.weights_dir is not None),
            "atlas_mounted": bool(self._sandbox_cfg is not None and self._sandbox_cfg.atlas_dir is not None),
            "allowed_tools": self.allowed_tools(),
            "practice_only": not has_claim,
            "note": (
                "Claim gate enabled on bound brief; practice tools return ranks only, claim issues harness verdicts."
                if has_claim else PRACTICE_NOTE
            ),
        }
        if self._brief is not None:
            card["brief"] = {
                "name": self._brief.name,
                "brief_id": self._brief.id,
                "metric": self._brief.metric,
                "direction": self._brief.direction,
                "least_effect": self._brief.least_effect,
                "allowed_tensors": list(self._brief.allowed_tensors),
                "allowed_layers": list(self._brief.allowed_layers) if self._brief.allowed_layers is not None else None,
                "max_fresh_readings": self._brief.max_fresh_readings,
                "controls_per_family": self._brief.default_controls,
                "driver_text": self._brief.driver_text,
            }
            if has_claim:
                # how a fresh reading here is taken: fixed when the session was made, shown and not the driver's to set
                card["claim_design"] = {"set_by": "the session's owner, before it began",
                                        **{k: self._claim_design[k] for k in sorted(self._claim_design)}}
        return card

    def _refuse_kept_back(self, items: Sequence[lv.Item]) -> None:
        """Refuse a practice reading or capture on an item that holds a rendering of a kept-back frame, in its
        question, its system turn or its answer (`level_briefs.KeptBack`). Nothing is run and nothing is entered."""
        for it in items:
            got = self._kept_back.hit(it.question, it.system, it.answer)
            if got is not None:
                raise SessionRefused(
                    f'Item "{it.name}" holds a kept-back frame of brief "{got["brief"]}". Kept-back frames are read '
                    "by a fresh claim reading alone, never in practice: draw practice items from the brief "
                    "(draw_practice) or write your own on other frames."
                )

    def _wall_seconds(self) -> float:
        return round(time.monotonic() - self._t_start, 3)

    def _remaining(self) -> Dict[str, Any]:
        wall = self._wall_seconds()
        return {
            "gpu_seconds": round(max(0.0, float(self.budget.max_gpu_seconds) - self.gpu_seconds_used), 3),
            "forwards": max(0, int(self.budget.max_forwards) - self.forwards_used),
            "cells": max(0, int(self.budget.max_cells) - self.cells_used),
            "wall_seconds": round(max(0.0, float(self.budget.max_wall_seconds) - wall), 3),
        }

    def _usage(self) -> Dict[str, Any]:
        return {
            "gpu_seconds": round(self.gpu_seconds_used, 3),
            "forwards": int(self.forwards_used),
            "cells": int(self.cells_used),
            "cell_seconds": round(self.cell_seconds_used, 3),
            "wall_seconds": self._wall_seconds(),
        }

    def _check_open_and_wall(self) -> None:
        if self._closed:
            raise SessionRefused(f"Session {self.session_id} is closed.")
        wall = self._wall_seconds()
        if wall > float(self.budget.max_wall_seconds):
            raise SessionRefused(
                f"Session wall-clock cap reached ({wall:.1f} s > {self.budget.max_wall_seconds:g} s). Close the session."
            )

    def _check_gpu_budget(self, est_forwards: int, est_seconds: float) -> None:
        rem = self._remaining()
        if est_forwards > rem["forwards"]:
            raise SessionRefused(
                f"Forward budget exceeded: this call needs {est_forwards} forward(s), "
                f"only {rem['forwards']} of {self.budget.max_forwards} left."
            )
        if est_seconds > rem["gpu_seconds"] and self.gpu_seconds_used > 0.0:
            raise SessionRefused(
                f"GPU-second budget exceeded: this call is estimated at {est_seconds:.1f} s, "
                f"only {rem['gpu_seconds']:.1f} s of {self.budget.max_gpu_seconds:g} s left."
            )
        if self.gpu_seconds_used >= float(self.budget.max_gpu_seconds):
            raise SessionRefused(
                f"GPU-second budget exhausted ({self.gpu_seconds_used:.1f} s >= {self.budget.max_gpu_seconds:g} s)."
            )

    def _log(self, event: str, payload: Mapping[str, Any]) -> None:
        rec = {"at": _utc_now(), "event": event, "usage": self._usage(), "payload": payload}
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=True) + "\n")

    @contextlib.contextmanager
    def _hold_model(self) -> Iterator[lv.Handle]:
        if self._handle is None:
            raise SessionRefused("This session has no model handle bound.")
        h = self._handle
        lock_cm = self._gpu_lock() if self._gpu_lock is not None else contextlib.nullcontext()
        with self._thread_lock, lock_cm:
            we.assert_base(h.lm, h.hooks_at_load)
            try:
                yield h
            finally:
                we.assert_base(h.lm, h.hooks_at_load)

    @contextlib.contextmanager
    def _use_book(self) -> Iterator[None]:
        prev = os.environ.get(lb.BOOK_ENV)
        os.environ[lb.BOOK_ENV] = str(self.book_path)
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop(lb.BOOK_ENV, None)
            else:
                os.environ[lb.BOOK_ENV] = prev

    # --------------------------------------------------------------------------------------------- gated tools

    def status(self) -> Dict[str, Any]:
        """Current session state, harness card, budget usage, and remaining allowances."""
        return {
            "session_id": self.session_id,
            "closed": self._closed,
            "harness_card": self.harness_card(),
            "budget": self.budget.to_dict(),
            "usage": self._usage(),
            "remaining": self._remaining(),
            "book_entries_this_session": len(self._book_seqs),
        }

    def cell(self, code: str, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Execute Python code off-card inside the `bwrap` sandbox (no GPU, no network, read-only weights)."""
        self._check_open_and_wall()
        if self._sandbox_cfg is None:
            raise SessionRefused(f'The cell tool is not available at reach "{self.reach}".')
        if self.cells_used >= int(self.budget.max_cells):
            raise SessionRefused(f"Cell budget exhausted ({self.cells_used} of {self.budget.max_cells}).")
        cap = float(self.budget.max_cell_seconds)
        req_timeout = cap if timeout_s is None else min(float(timeout_s), cap)
        self._cell_seq += 1
        cell_id = f"{self._cell_seq:04d}"
        log_dir = self.cells_dir / cell_id
        res = ls.run_cell(code, self._sandbox_cfg, cell_log_dir=log_dir, timeout_s=req_timeout)
        self.cells_used += 1
        self.cell_seconds_used += float(res["seconds"])
        out = {"cell_id": cell_id, **res, "remaining": self._remaining()}
        self._log("cell", {"cell_id": cell_id, "ok": res["ok"], "seconds": res["seconds"],
                           "timed_out": res["timed_out"], "oom_killed": res["oom_killed"],
                           "files_written": res["files_written"]})
        return out

    def capture(
        self,
        items_a: Sequence[Any],
        items_b: Sequence[Any],
        layer: int,
        *,
        where: str = "resid_post",
        at: str = "end",
        out_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Gated contrast-direction capture (`levelling.capture_direction`), persisted to `/work` with sha256."""
        self._check_open_and_wall()
        if self._handle is None:
            raise SessionRefused("No model handle is bound to this session.")
        a_items = _as_item_list(items_a, self._handle.tokenizer, "items_a")
        b_items = _as_item_list(items_b, self._handle.tokenizer, "items_b")
        self._refuse_kept_back(a_items + b_items)
        n_fwd = len(a_items) + len(b_items)
        est_s = sum(lv.whole(1, len(it.ids)) for it in a_items + b_items)
        self._check_gpu_budget(n_fwd, est_s)

        rel_name = out_name or f"capture_L{int(layer)}_{where}_{at}_{secrets.token_hex(3)}.npy"
        rel = Path(str(rel_name))
        if rel.is_absolute() or ".." in rel.parts:
            raise SessionRefused("out_name must be a relative path inside the session's work folder.")
        target = (self.work_dir / rel).resolve()
        if not str(target).startswith(str(self.work_dir) + os.sep):
            raise SessionRefused("out_name must lie inside the session's work folder.")

        t0 = time.perf_counter()
        with self._hold_model() as h:
            vec = lv.capture_direction(h, a_items, b_items, int(layer), where=where, at=at, _plant=self._plant)
            qm = we.quiet_mask(h.lm, int(layer))
        elapsed = time.perf_counter() - t0
        self.forwards_used += n_fwd
        self.gpu_seconds_used += elapsed

        v64 = vec.astype(np.float64)
        norm2 = float(np.dot(v64, v64))
        q_share = float(np.dot(v64[qm], v64[qm]) / norm2) if norm2 > 0.0 else 0.0
        saved = save_f32_array(target, vec)
        rel_out = str(target.relative_to(self.work_dir))
        out = {
            "rel_path": rel_out,
            "sha256": saved["sha256"],
            "shape": saved["shape"],
            "dtype": saved["dtype"],
            "layer": int(layer),
            "page_layer": int(layer) + 1,
            "where": where,
            "at": at,
            "n_a": len(a_items),
            "n_b": len(b_items),
            "norm": float(np.sqrt(norm2)),
            "quiet_share": q_share,
            "seconds": round(elapsed, 3),
            "remaining": self._remaining(),
        }
        self._log("capture", {k: v for k, v in out.items() if k != "remaining"})
        return out

    def estimate(self, plan: Mapping[str, Any], items: Mapping[str, Any]) -> Dict[str, Any]:
        """Estimate GPU-seconds, forwards, and memory for a plan without touching the GPU."""
        self._check_open_and_wall()
        if self._handle is None:
            raise SessionRefused("No model handle is bound to this session.")
        h = self._handle
        item_map = _as_item_map(items, h.tokenizer)
        norm_plan = lv.normalise(dict(plan), n_layers=h.n_layers, tokenizer=h.tokenizer, dtype=h.dtype)
        names = list(item_map) if norm_plan["items"] == "all" else norm_plan["items"]
        for nm in names:
            if nm not in item_map:
                raise SessionRefused(f'Plan asks for item "{nm}", which was not supplied.')
        facts = we.recipe_facts(norm_plan["edit"], self.work_dir)
        pieces = [(len(item_map[n].ids), len(item_map[n].prompt_ids)) for n in names]
        est = lv.estimate(norm_plan, pieces, n_layers=h.n_layers, facts=facts)
        rem = self._remaining()
        return {
            "estimate": est,
            "fits_budget": bool(est["forwards"] <= rem["forwards"] and est["seconds"] <= rem["gpu_seconds"]),
            "remaining": rem,
        }

    def _run_reading_gate(
        self,
        expected_kind: str,
        plan: Mapping[str, Any],
        items: Mapping[str, Any],
        *,
        save_edit: bool = False,
    ) -> Dict[str, Any]:
        self._check_open_and_wall()
        if self._handle is None:
            raise SessionRefused("No model handle is bound to this session.")
        if not isinstance(plan, Mapping):
            raise SessionRefused("plan must be a JSON object.")
        p = dict(plan)
        kind = p.get("kind") or expected_kind
        if kind != expected_kind:
            raise SessionRefused(f'This gate takes kind "{expected_kind}", not {kind!r}.')
        p["kind"] = expected_kind

        # Gate enforces control edits (driver cannot disable them below min_controls).
        min_ctl = int(self.budget.min_controls if expected_kind == "paired" else self.budget.min_written_controls)
        ctl = dict(p["controls"]) if isinstance(p.get("controls"), Mapping) else {}
        if "count" not in ctl:
            if ctl.get("family") == "sham_target" and isinstance(ctl.get("shams"), list):
                ctl["count"] = len(ctl["shams"])
            else:
                ctl["count"] = max(min_ctl, 3 if expected_kind == "written" else 19)
        if isinstance(ctl.get("count"), int) and not isinstance(ctl.get("count"), bool) and ctl["count"] < min_ctl:
            raise SessionRefused(
                f'Practice gate requires at least {min_ctl} control edit(s) for "{expected_kind}" readings; '
                f"{ctl['count']} requested."
            )
        p["controls"] = ctl

        # Written gate always enables loop/cut-off diagnostics so driver_view returns structured written summaries.
        if expected_kind == "written":
            p.setdefault("diagnose", True)

        # In blinded (planted or canary) sessions, always run per-row form only so neither leaks state via twin.
        if expected_kind == "paired" and self._blinded:
            p["folded"] = "never"

        h = self._handle
        item_map = _as_item_map(items, h.tokenizer)
        self._refuse_kept_back(list(item_map.values()))

        # Budget preflight using normalise + estimate (before touching the GPU).
        try:
            norm_p = lv.normalise(p, n_layers=h.n_layers, tokenizer=h.tokenizer, dtype=h.dtype)
            names = list(item_map) if norm_p["items"] == "all" else norm_p["items"]
            chosen_pieces = [(len(item_map[n].ids), len(item_map[n].prompt_ids)) for n in names if n in item_map]
            if len(chosen_pieces) == len(names) and chosen_pieces:
                facts = we.recipe_facts(norm_p["edit"], self.work_dir)
                est = lv.estimate(norm_p, chosen_pieces, n_layers=h.n_layers, facts=facts)
                self._check_gpu_budget(int(est["forwards"]), float(est["seconds"]))
        except (lv.Refused, we.Refused):
            pass  # Let lv.read record the refusal in the level book in its exact standard words.

        with self._hold_model() as mh, self._use_book():
            result = lv.read(mh, p, item_map, base_dir=self.work_dir, save_edit=bool(save_edit), _plant=self._plant)
            seq = int(result["book"]["seq"])
            entries = list(lb.entries(self.book_path, start=seq, limit=1))
            redacted = lb.driver_view(entries[0]) if entries else {"seq": seq, "body": {}}

        actual_s = float((result.get("seconds") or {}).get("actual") or 0.0)
        # Count forwards actually executed (0 on refused; estimated/counted on ok or void)
        if result.get("status") == "refused":
            used_fwd = 0
        else:
            used_fwd = 0
            if expected_kind == "paired":
                for it in result.get("items") or []:
                    for c in ("small", "large"):
                        cb = (it.get("classes") or {}).get(c)
                        if cb:
                            used_fwd += int(cb.get("forwards") or 0) + (1 if (cb.get("folded") or {}).get("taken") else 0)
            else:
                for it in result.get("items") or []:
                    used_fwd += len(it.get("samples") or [1])
            if (result.get("controls") or {}).get("family") == "other_sites_matched":
                used_fwd += len(result.get("items") or [])
        self.gpu_seconds_used += actual_s
        self.forwards_used += used_fwd
        if result.get("status") != "refused":
            self._practice_reads += 1
        self._book_seqs.append(seq)

        out = {
            "book": result["book"],
            "driver_view": redacted,
            "forwards_charged": used_fwd,
            "gpu_seconds_charged": round(actual_s, 3),
            "remaining": self._remaining(),
            "note": PRACTICE_NOTE,
        }
        self._log(expected_kind, {
            "name": p.get("name"),
            "seq": seq,
            "status": result.get("status"),
            "edit_id": (result.get("edit") or {}).get("id"),
            "forwards_charged": used_fwd,
            "gpu_seconds_charged": round(actual_s, 3),
        })
        return out

    def read(self, plan: Mapping[str, Any], items: Mapping[str, Any], *, save_edit: bool = False) -> Dict[str, Any]:
        """Gated paired reading returning `level_book.driver_view` aggregates only (no word past a rank)."""
        return self._run_reading_gate("paired", plan, items, save_edit=save_edit)

    def write(self, plan: Mapping[str, Any], items: Mapping[str, Any], *, save_edit: bool = False) -> Dict[str, Any]:
        """Gated written reading (short greedy or long sampled/thinking-on with loop & cut-off summary)."""
        return self._run_reading_gate("written", plan, items, save_edit=save_edit)

    def brief_info(self) -> Dict[str, Any]:
        """Driver-facing summary of the bound brief and remaining fresh-reading budget (never reveals kept-back frames)."""
        self._check_open_and_wall()
        if self._brief is None:
            raise SessionRefused("No brief is bound to this session.")
        hist = lcl.brief_history(self._brief.id, self.book_path, name=self._brief.name)
        used = len(hist["registrations"])
        return {
            "name": self._brief.name,
            "brief_id": self._brief.id,
            "behavior": self._brief.behavior,
            "driver_text": self._brief.driver_text,
            "metric": self._brief.metric,
            "direction": self._brief.direction,
            "least_effect": self._brief.least_effect,
            "readouts": dict(self._brief.readouts),
            "allowed_tensors": list(self._brief.allowed_tensors),
            "allowed_layers": list(self._brief.allowed_layers) if self._brief.allowed_layers is not None else None,
            "chat_settings": dict(self._brief.chat_settings),
            "max_fresh_readings": int(self._brief.max_fresh_readings),
            "fresh_readings_used": used,
            "fresh_readings_remaining": max(0, int(self._brief.max_fresh_readings) - used),
            "controls_per_family": int(self._brief.default_controls),
            "practice_frame_count": len(self._brief.generator.frames_for("practice")),
            "bars_sha256": self._bars.id if self._bars is not None else None,
            "practice_reads_this_session": int(self._practice_reads),
            "prior_verdicts": [
                {"reading_index": v.get("reading_index"), "edit_id": v.get("edit_id"), "verdict": v.get("verdict")}
                for v in hist["verdicts"]
            ],
        }

    def draw_practice(self, n: int = 6, *, seed: int = 0) -> Dict[str, Any]:
        """Draw `n` practice items from the bound brief's `practice` split and log their ids in the level book
        so future fresh draws automatically refuse any overlap."""
        self._check_open_and_wall()
        if self._brief is None:
            raise SessionRefused("No brief is bound to this session.")
        if self._handle is None:
            raise SessionRefused("No model handle is bound to this session.")
        n_int = int(n)
        if not 1 <= n_int <= 64:
            raise SessionRefused(f"draw_practice takes n in 1..64; got {n_int}.")
        seed_bytes = f"practice:{self._brief.id}:{int(seed)}:{self.session_id}".encode("utf-8")
        drawn = self._brief.generator.draw(
            "practice", n_int, seed_bytes=seed_bytes, tokenizer=self._handle.tokenizer,
            system_override=(self._brief.chat_settings or {}).get("system"),
            name_prefix=f"prac_{int(seed)}",
        )
        note = lcl.record_practice_draw(self._brief.id, drawn, book_path=self.book_path,
                                        brief_name=self._brief.name)
        self._book_seqs.append(int(note["seq"]))
        items_out = {
            d.item.name: {
                "question": d.item.question,
                "answer": d.item.answer,
                "system": d.item.system,
                "frame_id": d.frame_id,
            }
            for d in drawn
        }
        self._log("draw_practice", {"n": len(drawn), "seed": int(seed), "book_seq": int(note["seq"])})
        return {"items": items_out, "count": len(drawn), "book_seq": int(note["seq"])}

    def claim(
        self,
        edit: Any,
        *,
        main_dose: float = 1.0,
        forecast: Optional[Mapping[str, Any]] = None,
        selection_mode: str = "free_pick",
        sham_recipes: Optional[Sequence[Mapping[str, Any]]] = None,
        practice_best: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run one registered Phase 3 fresh claim reading on the bound brief and return ONLY the redacted
        `driver_record` (verdict, bounded claim sentence, Claim Sheet, aggregate summary — never fresh item text).
        The driver brings the edit, its dose, a forecast, and (for a site that came from a declared rule) its sham
        edits. How the reading is taken is not the driver's: the counts of fresh items, the rows, the control edits a
        family and the tolerance the matched ones are matched within are the gate's own, or what the session's owner
        fixed when the session was made (`claim_design`), and the brief's author prompt arms are always compared (the
        owner's command line can set these; this door cannot)."""
        self._check_open_and_wall()
        if self._brief is None or self._bars is None:
            raise SessionRefused("Claim gate is not enabled on this session (requires both a bound brief and bar set).")
        if self._blinded:
            raise SessionRefused("Claim gate is disabled in blinded planted/canary practice sessions.")
        if self._handle is None:
            raise SessionRefused("No model handle is bound to this session.")
        if self.gpu_seconds_used >= float(self.budget.max_gpu_seconds):
            raise SessionRefused("GPU-second budget exhausted before claim reading.")

        t0 = time.perf_counter()
        with self._hold_model() as mh:
            res = lcl.run_fresh_claim(
                mh,
                self._brief,
                self._bars,
                edit,
                base_dir=self.work_dir,
                main_dose=float(main_dose),
                forecast=forecast,
                selection_mode=selection_mode,
                sham_recipes=sham_recipes,
                practice_count=self._practice_reads,
                practice_best=practice_best,
                run_prompt_arms=True,
                book_path=self.book_path,
                **self._claim_design,
            )
        elapsed = time.perf_counter() - t0
        self.gpu_seconds_used += elapsed
        self._book_seqs.extend(res["book"]["all_seqs"])
        drv = dict(res["driver_record"])
        drv["gpu_seconds_charged"] = round(elapsed, 3)
        drv["remaining"] = self._remaining()
        self._log("claim", {
            "brief_id": self._brief.id,
            "edit_id": drv["edit_id"],
            "reading_index": drv["reading_index"],
            "verdict": drv["verdict"],
            "book": drv["book"],
            "gpu_seconds_charged": drv["gpu_seconds_charged"],
        })
        return drv

    def book_view(self, *, seq: Optional[int] = None, tail: int = 10) -> Dict[str, Any]:
        """Read entries from the level book strictly through `level_book.driver_view` (redacted aggregates only)."""
        self._check_open_and_wall()
        v = lb.verify(self.book_path)
        if not v["ok"]:
            raise SessionRefused(f"Ledger verification failed at line {v['first_bad']}: {v['why']}")
        total = int(v["entries"])
        if seq is not None:
            s = int(seq)
            if s < 0 or s >= total:
                raise SessionRefused(f"Book has {total} entries (0 to {max(total - 1, 0)}); seq {s} does not exist.")
            rows = [lb.driver_view(e) for e in lb.entries(self.book_path, start=s, limit=1)]
        else:
            n = max(1, min(int(tail), 50))
            start = max(0, total - n)
            rows = [lb.driver_view(e) for e in lb.entries(self.book_path, start=start, limit=n)]
        return {"entries": rows, "total": total, "head": v["head"], "redacted": True}

    def close(self) -> Dict[str, Any]:
        """Close the session and copy its transcript into `<book_dir>/transcripts/<session_id>.jsonl`."""
        if not self._closed:
            self._log("session_close", {"book_seqs": list(self._book_seqs)})
            self._closed = True
        transcripts_dir = self.book_path.parent / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        dest = transcripts_dir / f"{self.session_id}.jsonl"
        shutil.copy2(self._transcript_path, dest)
        summary = {
            "v": SESSION_VERSION,
            "session_id": self.session_id,
            "closed_at": _utc_now(),
            "usage": self._usage(),
            "book_seqs": list(self._book_seqs),
            "transcript_copy": str(dest),
        }
        (self.session_dir / "session_closed.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
