"""FastAPI routes for Phase 4: The Person's Page (`backend/level_routes.py`).

Mounted by `backend/server.py` at `/api/level/*` under the shared loopback guard (`local_pages_only`) and the single
`_GPU_LOCK`, following the exact binding pattern of `backend/atlas.py`:
- `GET /api/level/status`          — runtime readiness, base cleanliness, ledger verification, replayable books.
- `POST /api/level/estimate`       — dry off-GPU cost quote for a side-by-side audition (no GPU lock).
- `POST /api/level/compare`        — base, edit, and control edit(s) heard side by side in one batched forward, plus
                                     optional paired next-piece table in two batch classes (`small` & `large`). Never
                                     leaves the model edited; refuses with HTTP 409 if any vault overlay is in force.
- `GET /api/level/book`            — model-free replay index of the append-only level book (live or allowlisted
                                     verified 31B books in `experiments/results/levelling/`).
- `GET /api/level/book/{seq}`      — model-free full replay card for one ledger entry (`written`, `paired`, or `note`).

Layers count from 0 on the wire and in code; human-facing labels give page layers (1-based) where helpful.
No kept-out words appear in code, messages, or responses.
"""
from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend import atlas as atlas_routes
from backend import atlas_ops as ao
from backend import level_book as lb
from backend import level_briefs as lbr
from backend import levelling as lv
from backend import weight_edits as we

ROUTES_VERSION = 1
ROOT_DIR = Path(__file__).resolve().parent.parent
RECORDED_BOOKS_DIR = ROOT_DIR / "experiments" / "results" / "levelling"

router = APIRouter(prefix="/api/level", dependencies=[Depends(atlas_routes.local_pages_only)])

_get_state: Optional[Callable[[], Tuple[Any, Any]]] = None
_gpu_lock: Optional[asyncio.Lock] = None
_handle_cache: Dict[int, lv.Handle] = {}


def bind(get_state: Callable[[], Tuple[Any, Any]], gpu_lock: asyncio.Lock) -> None:
    """Bind the server's `(get_app_state, _GPU_LOCK)` from `backend/server.py`."""
    global _get_state, _gpu_lock
    _get_state, _gpu_lock = get_state, gpu_lock
    _handle_cache.clear()


def _runtime_and_vault() -> Tuple[Any, Any]:
    if _get_state is None:
        raise HTTPException(status_code=503, detail="Levelling routes are not bound to a server.")
    rt, vault = _get_state()
    if rt is None:
        raise HTTPException(status_code=503, detail="The model runtime is not loaded yet.")
    return rt, vault


def _lm_of(rt: Any) -> Any:
    hf = getattr(rt, "hf_model", None)
    if hf is not None:
        return lv.language_model(hf)
    return getattr(rt, "language_model", None) or getattr(rt, "model", None)


def _active_vault_edit_count(rt: Any, vault: Any) -> int:
    if vault is not None:
        patches = getattr(vault, "active_patches", None)
        if isinstance(patches, Mapping):
            return len(patches)
    lm = _lm_of(rt)
    if lm is None:
        return 0
    try:
        we.assert_base(lm)
        return 0
    except we.NotBase:
        return 1


def _clean_handle(rt: Any) -> lv.Handle:
    """Return a cached `levelling.Handle` for `rt` after asserting the model is cleanly at base."""
    key = id(rt)
    h = _handle_cache.get(key)
    if h is None or h.rt is not rt:
        h = lv.handle(rt)
        _handle_cache.clear()
        _handle_cache[key] = h
    we.assert_base(h.lm, h.hooks_at_load)
    return h


# --------------------------------------------------------------------------------------------- allowlisted books


def _book_catalog() -> Dict[str, Dict[str, Any]]:
    """Allowlisted replayable books: the active live book plus verified 31B card books in the repository."""
    live_path = lb.book_path()
    out: Dict[str, Dict[str, Any]] = {
        "live": {
            "id": "live",
            "label": "live level book",
            "path": live_path,
            "exists": live_path.exists(),
            "recorded": False,
        }
    }
    candidates = [
        ("verify_trial", "31B trial dry run check (19 Nov 2026)", RECORDED_BOOKS_DIR / "verify_trial_book.jsonl"),
        ("verify_claim", "31B claim gate check (19 Nov 2026)", RECORDED_BOOKS_DIR / "verify_claim_book.jsonl"),
        ("verify_door", "31B agent door check (19 Nov 2026)", RECORDED_BOOKS_DIR / "verify_door_book.jsonl"),
        ("verify_1b", "31B slice 1b check (18 Nov 2026)", RECORDED_BOOKS_DIR / "verify_1b_book.jsonl"),
    ]
    for bid, label, p in candidates:
        if p.exists():
            out[bid] = {"id": bid, "label": label, "path": p, "exists": True, "recorded": True}
    return out


def _resolve_book(book_id: str) -> Tuple[str, Path, bool]:
    cat = _book_catalog()
    if book_id not in cat:
        raise HTTPException(status_code=400, detail=f'Unknown book "{book_id}". Available: {", ".join(cat)}.')
    info = cat[book_id]
    return info["id"], Path(info["path"]), bool(info["recorded"])


# --------------------------------------------------------------------------------------------- request schemas


class CompareRequest(BaseModel):
    """Side-by-side audition request: one edit heard beside the unedited base and `controls_count` control edits."""

    op: Literal["silence", "scale", "silence_rows", "scale_rows"] = "silence"
    linear: Literal["down_proj", "gate_proj", "up_proj"] = "down_proj"
    layer: int = Field(ge=0, le=59, description="0-based layer index (page layer minus 1).")
    neurons: List[int] = Field(min_length=1, max_length=64)
    factor: float = 0.0
    dose: float = 1.0
    question: str = Field(min_length=1, max_length=4000)
    answer: Optional[str] = Field(default=None, max_length=2000)
    system: Optional[str] = Field(default=None, max_length=2000)
    controls_family: Literal["other_sites", "other_sites_matched", "same_site"] = "other_sites"
    controls_count: int = Field(default=3, ge=1, le=19)
    controls_seed: int = Field(default=0, ge=0, le=2_147_483_647)
    match_tolerance: float = Field(default=lv.DEFAULT_MATCH_TOL, ge=1.0, le=50.0)
    max_new: int = Field(default=32, ge=4, le=256)
    thinking: bool = False
    sampled: bool = False
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=64, ge=0, le=500)
    sampler_seed: int = Field(default=1, ge=0, le=2_147_483_647)
    include_paired: bool = True
    toward_word: Optional[str] = Field(default=None, max_length=64)
    away_word: Optional[str] = Field(default=None, max_length=64)
    odds_at: Literal["end", "span"] = "end"
    name: Optional[str] = Field(default=None, max_length=120)


def _recipe_from_request(req: CompareRequest) -> Dict[str, Any]:
    """Build a canonical `weight_edits` recipe dict from `CompareRequest`."""
    if not math.isfinite(float(req.dose)) or float(req.dose) == 0.0:
        raise ValueError("Dose must be a finite non-zero number.")
    if not math.isfinite(float(req.factor)):
        raise ValueError("Scale factor must be a finite number.")
    if len(set(req.neurons)) != len(req.neurons):
        raise ValueError("Neuron indices must be distinct.")
    if req.op in ("silence", "scale") and req.linear != "down_proj":
        raise ValueError('Column operations ("silence", "scale") apply to down_proj only.')
    if req.op in ("silence_rows", "scale_rows") and req.linear not in ("gate_proj", "up_proj"):
        raise ValueError('Row operations ("silence_rows", "scale_rows") apply to gate_proj or up_proj.')

    L = int(req.layer)
    idxs = [int(i) for i in req.neurons]
    if req.op == "silence":
        term: Dict[str, Any] = {"layer": L, "silence": idxs}
    elif req.op == "scale":
        term = {"layer": L, "scale": idxs, "factor": float(req.factor)}
    elif req.op == "silence_rows":
        term = {"layer": L, "which": req.linear, "silence_rows": idxs}
    else:
        term = {"layer": L, "which": req.linear, "scale_rows": idxs, "factor": float(req.factor)}
    return {"terms": [term]}


def _controls_block(req: CompareRequest) -> Dict[str, Any]:
    ctl: Dict[str, Any] = {
        "family": req.controls_family,
        "count": int(req.controls_count),
        "seed": int(req.controls_seed),
    }
    if req.controls_family == "other_sites_matched":
        ctl["tolerance"] = float(req.match_tolerance)
    return ctl


def _build_plans(req: CompareRequest, tokenizer: Any) -> Tuple[lv.Item, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Construct `(item, written_plan, paired_plan_or_None)` from `CompareRequest`."""
    recipe = _recipe_from_request(req)
    q = req.question.strip()
    sys_turn = req.system.strip() if isinstance(req.system, str) and req.system.strip() else None
    ans_clean = req.answer.strip() if isinstance(req.answer, str) and req.answer.strip() else None
    item = lv.make_item(tokenizer, q, ans_clean or "OK", system=sys_turn, name="audition")

    C = int(req.controls_count)
    doses_w = [float(req.dose), 0.0]
    w_rows = max(lv.WRITTEN_ROWS_MIN, min(lv.WRITTEN_ROWS_MAX, 2 + len(lv.NULL_E_WRITTEN) + len(doses_w) + C))
    label = (req.name or "").strip() or f"vault_compare_L{req.layer}_{req.op}_{len(req.neurons)}"

    written_plan: Dict[str, Any] = {
        "name": f"{label}_written",
        "kind": "written",
        "edit": recipe,
        "doses": doses_w,
        "main_dose": float(req.dose),
        "controls": _controls_block(req),
        "rows": w_rows,
        "max_new": int(req.max_new),
        "thinking": bool(req.thinking),
        "diagnose": True,
        "samples": 1,
        "sampler": {
            "temperature": float(req.temperature) if req.sampled else 0.0,
            "top_k": int(req.top_k),
            "top_p": float(req.top_p),
            "presence_penalty": 0.0,
            "seed": int(req.sampler_seed),
        },
    }

    paired_plan: Optional[Dict[str, Any]] = None
    if req.include_paired and ans_clean:
        odds_cfg: Optional[Dict[str, Any]] = None
        tw = req.toward_word.strip() if isinstance(req.toward_word, str) and req.toward_word.strip() else None
        aw = req.away_word.strip() if isinstance(req.away_word, str) and req.away_word.strip() else None
        if tw and aw:
            # Try leading-space piece first if bare word splits into multiple pieces
            def _one_piece(word: str, side: str) -> List[int]:
                for cand in (f" {word}", word):
                    ids = [int(t) for t in tokenizer(cand, add_special_tokens=False)["input_ids"]]
                    if len(ids) == 1:
                        return ids
                raise ValueError(f'Contrast word {word!r} ({side}) is not a single word-piece for this tokenizer.')

            odds_cfg = {
                "toward": _one_piece(tw, "toward"),
                "away": _one_piece(aw, "away"),
                "at": req.odds_at,
            }
        elif tw or aw:
            raise ValueError("Contrast odds require both a toward word and an away word (or leave both blank).")

        readouts: Dict[str, Any] = {"kl": True, "own": True, "odds": odds_cfg}
        rank_by = "odds" if odds_cfg is not None else "kl"
        paired_plan = {
            "name": f"{label}_paired",
            "kind": "paired",
            "edit": recipe,
            "doses": [float(req.dose), 0.0],
            "main_dose": float(req.dose),
            "controls": _controls_block(req),
            "rows": {"small": lv.ROWS_SMALL, "large": lv.ROWS_LARGE},
            "readouts": readouts,
            "rank_by": rank_by,
            "folded": "auto",
        }
    return item, written_plan, paired_plan


# --------------------------------------------------------------------------------------------- replay cards


def _written_voices_from_body(body: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract a 3-voice side-by-side comparison payload (`base`, `edit`, `controls[]`) from a written reading body."""
    items = body.get("items")
    if not isinstance(items, list) or not items:
        return None
    it0 = items[0]
    base = it0.get("base")
    arms = it0.get("arms")
    if not isinstance(base, Mapping) or not isinstance(arms, list):
        return None
    plan = body.get("plan") or {}
    main_dose = plan.get("main_dose")
    edit_arm = None
    controls: List[Dict[str, Any]] = []
    for a in arms:
        what = a.get("what")
        if what == "edit":
            if main_dose is not None and a.get("dose") == main_dose:
                edit_arm = a
            elif edit_arm is None and a.get("dose") != 0.0:
                edit_arm = a
        elif what == "control":
            controls.append({
                "key": a.get("key"),
                "index": a.get("index"),
                "text": a.get("text", ""),
                "pieces_count": len(a.get("pieces") or []),
                "ended": bool(a.get("ended")),
                "same_as_base": bool(a.get("same_as_base")),
                "parts_at": a.get("parts_at"),
                "logprob_gap": a.get("logprob_gap"),
                "loop": a.get("loop"),
                "cut_off": bool(a.get("cut_off", False)),
                "stratum": a.get("stratum"),
                "thinking": a.get("thinking"),
            })
    if edit_arm is None:
        return None
    ctl_info = plan.get("controls") or body.get("controls") or {}
    return {
        "question": (it0.get("item") or {}).get("question"),
        "answer": (it0.get("item") or {}).get("answer"),
        "system": (it0.get("item") or {}).get("system"),
        "max_new": it0.get("max_new"),
        "thinking": bool(it0.get("thinking", False)),
        "sampler": it0.get("sampler"),
        "untouched_rows_identical": bool(it0.get("untouched_rows_identical", True)),
        "controls_family": ctl_info.get("family"),
        "controls_count": len(controls),
        "controls_seed": ctl_info.get("seed"),
        "base": {
            "text": base.get("text", ""),
            "pieces_count": len(base.get("pieces") or []),
            "ended": bool(base.get("ended")),
            "loop": base.get("loop"),
            "cut_off": bool(base.get("cut_off", False)),
            "stratum": base.get("stratum"),
            "thinking": base.get("thinking"),
        },
        "edit": {
            "key": edit_arm.get("key"),
            "dose": edit_arm.get("dose"),
            "text": edit_arm.get("text", ""),
            "pieces_count": len(edit_arm.get("pieces") or []),
            "ended": bool(edit_arm.get("ended")),
            "same_as_base": bool(edit_arm.get("same_as_base")),
            "parts_at": edit_arm.get("parts_at"),
            "logprob_gap": edit_arm.get("logprob_gap"),
            "loop": edit_arm.get("loop"),
            "cut_off": bool(edit_arm.get("cut_off", False)),
            "stratum": edit_arm.get("stratum"),
            "thinking": edit_arm.get("thinking"),
        },
        "controls": controls,
        "summary": body.get("summary"),
    }


def _paired_table_from_body(body: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract a two-batch-class paired metrics table (`small` & `large`) from a paired reading body."""
    sm = body.get("summary")
    if not isinstance(sm, Mapping):
        return None
    plan = body.get("plan") or {}
    main_dose = plan.get("main_dose", 1.0)
    arm_key = f"edit:{repr(float(main_dose))}"
    rank_by = plan.get("rank_by") or "kl"
    classes_out: Dict[str, Any] = {}
    sm_classes = sm.get("classes") or {}
    items = body.get("items") or []
    first_ok = next((it for it in items if it.get("status") == "ok"), None)

    for cls_name in ("small", "large"):
        cb = sm_classes.get(cls_name)
        if not isinstance(cb, Mapping):
            continue
        arms = cb.get("arms") or {}
        edit_m = arms.get(arm_key) or {}
        ctl_arms = {k: v for k, v in arms.items() if k.startswith("control:")}
        floor_blk = None
        folded_blk = None
        delivered_val = None
        rank_blk = None
        if isinstance(first_ok, Mapping):
            it_cls = (first_ok.get("classes") or {}).get(cls_name) or {}
            floor_blk = it_cls.get("floor")
            folded_blk = it_cls.get("folded")
            rank_blk = it_cls.get("rank")
            for a in it_cls.get("arms") or []:
                if a.get("key") == arm_key:
                    delivered_val = a.get("delivered")
                    break
        ctl_vals = [
            float((v.get(rank_by) or {}).get("mean_over_items"))
            for v in ctl_arms.values()
            if isinstance((v.get(rank_by) or {}).get("mean_over_items"), (int, float))
        ]
        edit_rank_val = (edit_m.get(rank_by) or {}).get("mean_over_items")
        at_least = None
        if isinstance(edit_rank_val, (int, float)) and ctl_vals:
            at_least = sum(1 for c in ctl_vals if abs(c) >= abs(float(edit_rank_val)))
        classes_out[cls_name] = {
            "rows": (plan.get("rows") or {}).get(cls_name),
            "metrics": edit_m,
            "delivered": delivered_val,
            "floor": floor_blk,
            "folded": folded_blk,
            "rank": rank_blk,
            "controls_count": len(ctl_arms),
            "at_least": at_least,
            "control_abs_max": max((abs(c) for c in ctl_vals), default=None),
            "control_abs_median": float(sorted(abs(c) for c in ctl_vals)[len(ctl_vals) // 2]) if ctl_vals else None,
        }
    return {
        "rank_by": rank_by,
        "main_dose": main_dose,
        "controls": plan.get("controls") or body.get("controls"),
        "items_summary": sm.get("items"),
        "classes": classes_out,
    }


def _extract_edit_linears(
    body: Mapping[str, Any],
    linears_by_edit: Optional[Mapping[str, List[str]]] = None,
) -> Optional[List[str]]:
    """Recover touched linear names from a reading or note body (`edit.linears`, `body.edit_linears`,
    `edit.terms.terms[*].linear`, or an optional `edit_id -> linears` map built from the same book)."""
    edit_blk = body.get("edit") if isinstance(body.get("edit"), Mapping) else {}
    direct = edit_blk.get("linears")
    if isinstance(direct, list) and direct:
        return [str(x) for x in direct]
    top = body.get("edit_linears")
    if isinstance(top, list) and top:
        return [str(x) for x in top]
    entries = lv._report_entries(edit_blk.get("terms"))
    found = sorted({str(t["linear"]) for t in entries if t.get("linear")})
    if found:
        return found
    eid = edit_blk.get("id") or body.get("edit_id")
    if isinstance(eid, str) and linears_by_edit and eid in linears_by_edit:
        return list(linears_by_edit[eid])
    return None


def _book_linears_by_edit(path: Path) -> Dict[str, List[str]]:
    """Scan a book once to map `edit_id -> sorted linears`, so note entries written without inline `edit_linears`
    (such as earlier `claim_verdict` or `claim_registration` notes) still display their linears on live replay."""
    out: Dict[str, List[str]] = {}
    try:
        for ent in lb.entries(path):
            body = ent.get("body")
            if not isinstance(body, Mapping):
                continue
            edit_blk = body.get("edit") if isinstance(body.get("edit"), Mapping) else {}
            eid = edit_blk.get("id") or body.get("edit_id")
            if not isinstance(eid, str) or eid in out:
                continue
            lins = _extract_edit_linears(body)
            if lins:
                out[eid] = lins
    except lb.BookBroken:
        pass  # a damaged line ends the scan, not the replay: the whole lines before it are still shown, as before
    return out


def build_replay_card(
    entry: Mapping[str, Any],
    *,
    linears_by_edit: Optional[Mapping[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Turn any level-book entry (`written`, `paired`, or `note`) into a model-free replay card for the UI."""
    kind = str(entry.get("kind") or "")
    body = entry.get("body") if isinstance(entry.get("body"), Mapping) else {}
    edit_blk = body.get("edit") if isinstance(body.get("edit"), Mapping) else {}
    sec_blk = body.get("seconds") if isinstance(body.get("seconds"), Mapping) else {}
    base_card: Dict[str, Any] = {
        "seq": entry.get("seq"),
        "at": entry.get("at"),
        "hash": str(entry.get("hash") or "")[:16],
        "kind": kind,
        "status": body.get("status"),
        "why": body.get("why"),
        "name": body.get("name"),
        "edit_id": edit_blk.get("id") or body.get("edit_id"),
        "edit_linears": _extract_edit_linears(body, linears_by_edit=linears_by_edit),
        "seconds": sec_blk.get("actual"),
        "base_after": body.get("base_after"),
    }
    if kind == "written":
        base_card["voices"] = _written_voices_from_body(body)
        base_card["lines"] = list(lv.lines(body)) if body.get("status") else []
    elif kind == "paired":
        base_card["paired"] = _paired_table_from_body(body)
        base_card["lines"] = list(lv.lines(body)) if body.get("status") else []
    elif kind == "note":
        nt = body.get("note_type")
        base_card["note_type"] = nt
        base_card["brief_name"] = body.get("brief_name")
        base_card["brief_id"] = body.get("brief_id")
        if nt == "claim_verdict":
            base_card["status"] = body.get("verdict")
            base_card["name"] = f'{body.get("brief_name")} ({body.get("reading_index")}/{body.get("max_fresh_readings")})'
            base_card["verdict"] = body.get("verdict")
            base_card["reasons"] = body.get("reasons") or []
            base_card["claim_sentence"] = body.get("claim_sentence")
            base_card["summary"] = body.get("summary")
            base_card["batteries"] = body.get("batteries")
            base_card["cheaper_knobs"] = body.get("cheaper_knobs")
            base_card["bars_sha256"] = body.get("bars_sha256")
            base_card["book_seqs"] = body.get("book_seqs")
        elif nt in ("trial_of_record", "trial_dry_run", "trial_rerun"):
            # only a trial of record shows its verdict as a verdict: a dry run's and a re-run's is prefixed, and so
            # takes no verdict colour on the page
            kind = {"trial_of_record": "trial of record", "trial_dry_run": "dry run of a trial",
                    "trial_rerun": "re-run of a trial"}[nt]
            lead = {"trial_of_record": "", "trial_dry_run": "dry run: ", "trial_rerun": "re-run: "}[nt]
            base_card["status"] = f'{lead}{body.get("verdict")}'
            base_card["name"] = f'{kind} · {body.get("trial_name") or body.get("brief_name")}'
            base_card["verdict"] = body.get("verdict")
            base_card["of_record"] = bool(body.get("of_record")) and nt == "trial_of_record"
            base_card["not_of_record"] = list(body.get("not_of_record") or [])
            base_card["reasons"] = body.get("reasons") or []
            base_card["claim_sentence"] = body.get("claim_sentence")
            base_card["summary"] = body.get("summary")
            base_card["batteries"] = body.get("batteries")
            base_card["cheaper_knobs"] = body.get("cheaper_knobs")
            base_card["bars_sha256"] = body.get("bars_sha256")
            base_card["book_seqs"] = body.get("book_seqs")
            base_card["folded_summary"] = body.get("folded")
            base_card["reworded_summary"] = body.get("reworded")
            base_card["trial_sha256"] = body.get("trial_sha256")
            base_card["cheaper_knob"] = body.get("cheaper_knob")  # the arm that matched a pass, with its figure
        elif nt == "trial_registration":
            base_card["status"] = "registered"
            base_card["name"] = f'register trial · {body.get("trial_name") or body.get("brief_name")}'
            base_card["of_record"] = bool(body.get("of_record"))
            base_card["not_of_record"] = list(body.get("not_of_record") or [])
            base_card["selection_mode"] = body.get("selection_mode")
            base_card["controls_per_family"] = body.get("controls_per_family")
            base_card["bars_sha256"] = body.get("bars_sha256")
            base_card["trial_sha256"] = body.get("trial_sha256")
        elif nt == "brief_calibration":
            base_card["status"] = "calibrated"
            base_card["name"] = f'bars for {body.get("brief_name")}'
            base_card["bars"] = body.get("bars")
            base_card["bars_sha256"] = body.get("bars_sha256")
            base_card["book_seqs"] = body.get("book_seqs")
        elif nt == "claim_registration":
            base_card["status"] = "registered"
            base_card["name"] = f'register {body.get("brief_name")} #{body.get("reading_index")}'
            base_card["selection_mode"] = body.get("selection_mode")
            base_card["controls_per_family"] = body.get("controls_per_family")
            base_card["bars_sha256"] = body.get("bars_sha256")
        elif nt == "practice_draw":
            base_card["status"] = "practice"
            base_card["name"] = f'practice draw ({body.get("count")} items)'
    return base_card


# --------------------------------------------------------------------------------------------- endpoints


@router.get("/status")
def level_status() -> Dict[str, Any]:
    """Off-GPU status: runtime readiness, base cleanliness, ledger integrity, replayable books, and starter briefs."""
    rt_ready = False
    is_base = None
    active_vault = 0
    model_id = None
    n_layers = None
    d_ff = None
    tokenizer = None
    if _get_state is not None:
        try:
            rt, vault = _get_state()
            if rt is not None:
                rt_ready = True
                model_id = getattr(rt, "model_id", None)
                tokenizer = getattr(rt, "tokenizer", None)
                lm = _lm_of(rt)
                if lm is not None:
                    n_layers = len(lm.layers)
                    if n_layers > 0:
                        d_ff = int(ao.ffn_width(lm, 0))
                active_vault = _active_vault_edit_count(rt, vault)
                is_base = (active_vault == 0)
        except Exception:
            pass

    cat = _book_catalog()
    books_out = []
    live_verify = None
    for bid, info in cat.items():
        p = Path(info["path"])
        v = lb.verify(p) if p.exists() else {"ok": True, "entries": 0, "head": lb.GENESIS, "first_bad": None, "why": None}
        if bid == "live":
            live_verify = v
        books_out.append({
            "id": bid,
            "label": info["label"],
            "recorded": info["recorded"],
            "exists": p.exists(),
            "entries": v["entries"],
            "ok": v["ok"],
            "head": str(v["head"])[:16],
        })

    starters = []
    for b in lbr.starter_briefs(tokenizer).values():
        starters.append({
            "name": b.name,
            "brief_id": b.short(),
            "metric": b.metric,
            "direction": b.direction,
            "least_effect": b.least_effect,
            "max_fresh_readings": b.max_fresh_readings,
            "controls_per_family": b.default_controls,
            "behavior": b.behavior,
        })

    return {
        "v": ROUTES_VERSION,
        "runtime_ready": rt_ready,
        "model_id": model_id,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "is_base": is_base,
        "active_vault_edits": active_vault,
        "live_book": {"path": str(lb.book_path()), "verify": live_verify},
        "books": books_out,
        "starter_briefs": starters,
    }


@router.post("/estimate")
def level_estimate(req: CompareRequest) -> Dict[str, Any]:
    """Dry off-GPU cost quote for a side-by-side compare request."""
    rt, _vault = _runtime_and_vault()
    lm = _lm_of(rt)
    n_layers = len(lm.layers)
    dtype = lm.norm.weight.dtype
    try:
        item, w_plan, p_plan = _build_plans(req, rt.tokenizer)
        w_facts = we.recipe_facts(w_plan["edit"], Path("."))
        w_norm = lv.normalise(w_plan, n_layers=n_layers, tokenizer=rt.tokenizer, facts=w_facts, dtype=dtype)
        pieces = [(len(item.ids), len(item.prompt_ids))]
        w_est = lv.estimate(w_norm, pieces, n_layers=n_layers, facts=w_facts)
        p_est = None
        if p_plan is not None:
            p_norm = lv.normalise(p_plan, n_layers=n_layers, tokenizer=rt.tokenizer, facts=w_facts, dtype=dtype)
            p_est = lv.estimate(p_norm, pieces, n_layers=n_layers, facts=w_facts)
    except (ValueError, lv.Refused, we.Refused) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    total_s = float(w_est["seconds"]) + (float(p_est["seconds"]) if p_est is not None else 0.0)
    total_fwd = int(w_est["forwards"]) + (int(p_est["forwards"]) if p_est is not None else 0)
    peak_gib = max(float(w_est["gib_above_model"]), float(p_est["gib_above_model"]) if p_est is not None else 0.0)
    return {
        "written": w_est,
        "paired": p_est,
        "total_seconds": round(total_s, 2),
        "total_forwards": total_fwd,
        "peak_gib_above_model": round(peak_gib, 3),
    }


@router.post("/compare")
async def level_compare(req: CompareRequest) -> Dict[str, Any]:
    """Run base, edit, and control edit(s) side by side in one batched forward (plus optional paired table)
    under `_GPU_LOCK`. Refuses with HTTP 409 if any vault overlay is in force; always asserts base after."""
    rt, vault = _runtime_and_vault()
    assert _gpu_lock is not None

    def _run_sync() -> Dict[str, Any]:
        active = _active_vault_edit_count(rt, vault)
        if active > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"There {'is 1 active vault edit' if active == 1 else f'are {active} active vault edits'} in force. "
                    "Undo active vault edits (ROLLBACK) before running a side-by-side weight-edit reading; "
                    "a reading never runs on an already-edited model."
                ),
            )
        try:
            h = _clean_handle(rt)
        except we.NotBase as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

        try:
            item, w_plan, p_plan = _build_plans(req, h.tokenizer)
            edit = we.build(h.lm, we.from_recipe(w_plan["edit"], Path(".")), model_id=h.model_id, weights_id=h.weights_id)
        except (ValueError, lv.Refused, we.Refused) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        t0 = time.perf_counter()
        # Keep the rebuildable recipe in `plan["edit"]` so every compare reading can be rebuilt from the ledger.
        w_res = lv.read(h, w_plan, {"audition": item})
        p_res = None
        if p_plan is not None and w_res.get("status") == "ok":
            p_res = lv.read(h, p_plan, {"audition": item})
        we.assert_base(h.lm, h.hooks_at_load)
        elapsed = round(time.perf_counter() - t0, 3)

        count, tip = lb.head()
        return {
            "v": ROUTES_VERSION,
            "status": w_res.get("status") if (p_res is None or p_res.get("status") == "ok") else p_res.get("status"),
            "why": w_res.get("why") or (p_res.get("why") if p_res else None),
            "model_id": h.model_id,
            "weights_id": (h.weights_id or "")[:16],
            "edit": {
                "id": edit.id,
                "short": edit.short(),
                "linears": edit.linears(),
                "layer": int(req.layer),
                "page_layer": int(req.layer) + 1,
                "op": req.op,
                "neurons": [int(i) for i in req.neurons],
                "dose": float(req.dose),
            },
            "voices": _written_voices_from_body(w_res),
            "paired": _paired_table_from_body(p_res) if p_res is not None else None,
            "written_lines": list(lv.lines(w_res)) if w_res.get("status") else [],
            "paired_lines": list(lv.lines(p_res)) if (p_res and p_res.get("status")) else [],
            "seconds": elapsed,
            "base_after": True,
            "book": {
                "written_seq": (w_res.get("book") or {}).get("seq"),
                "paired_seq": (p_res.get("book") or {}).get("seq") if p_res else None,
                "entries": count,
                "head": tip[:16],
            },
            "note": (
                "This side-by-side reading is for exploration and learning; nothing remains on the model after it ends. "
                "A rank against control edits says only that this edit moved more than those random edits on this question."
            ),
        }

    async with _gpu_lock:
        try:
            return await asyncio.to_thread(_run_sync)
        except HTTPException:
            raise
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(
                status_code=400,
                detail="The GPU ran out of memory for this reading; ask fewer controls or fewer new word-pieces.",
            ) from None


@router.get("/book")
def level_book_list(
    book: str = Query(default="live"),
    tail: int = Query(default=40, ge=1, le=200),
    redacted: bool = Query(default=False),
) -> Dict[str, Any]:
    """Model-free replay index of a level book (`live` or an allowlisted verified 31B book)."""
    bid, path, recorded = _resolve_book(book)
    if not path.exists():
        return {
            "book": bid,
            "recorded": recorded,
            "path": str(path),
            "verify": {"ok": True, "entries": 0, "head": lb.GENESIS, "first_bad": None, "why": None},
            "entries": [],
        }
    v = lb.verify(path)
    total = int(v["entries"])
    start = max(0, total - int(tail))
    cards: List[Dict[str, Any]] = []
    if total > 0:
        linears_map = _book_linears_by_edit(path)
        for ent in lb.entries(path, start=start, limit=int(tail)):
            src = lb.driver_view(ent) if redacted else ent
            cards.append(build_replay_card(src, linears_by_edit=linears_map))
    cards.reverse()  # newest first for quick inspection
    return {
        "book": bid,
        "recorded": recorded,
        "path": str(path),
        "verify": v,
        "entries": cards,
    }


@router.get("/book/{seq}")
def level_book_entry(
    seq: int,
    book: str = Query(default="live"),
    redacted: bool = Query(default=False),
) -> Dict[str, Any]:
    """Model-free full replay card for one ledger entry by line number `seq`."""
    bid, path, recorded = _resolve_book(book)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f'Book "{bid}" has no entries yet.')
    v = lb.verify(path)
    if seq < 0 or seq >= int(v["entries"]):
        raise HTTPException(status_code=404, detail=f'Book "{bid}" has no line {seq} (0 to {max(int(v["entries"]) - 1, 0)}).')
    ent = next(lb.entries(path, start=seq, limit=1), None)
    if ent is None:
        raise HTTPException(status_code=404, detail=f"Line {seq} could not be read.")
    linears_map = _book_linears_by_edit(path)
    shown = lb.driver_view(ent) if redacted else ent
    return {
        "book": bid,
        "recorded": recorded,
        "verify": v,
        "card": build_replay_card(shown, linears_by_edit=linears_map),
        "entry": shown,
    }
