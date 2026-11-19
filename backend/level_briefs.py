"""Briefs and parameterized frame generators with kept-back frames (`backend/level_briefs.py`), Phase 3.

A brief fixes a target behaviour, metric, least effect (`M`), allowed tensors, fixed fresh-reading budget (`m`),
control count (`20 * m - 1` per family), prompt-arm baselines written by the brief's author, and two disjoint
frame pools:
- `practice`: frames the driver may inspect and iterate on freely.
- `kept_back`: whole frames never shown in practice, split across `half = 0` and `half = 1` so every fresh claim
  reading checks replication across independent frame halves and computes frame-level confidence intervals.

Also provides chat-formatted battery generators (`neutral_chat_items`, `labelled_sanity_items`, and optional atlas
habitat passages) and starter briefs (`starter_briefs`).

No kept-out words appear in code, messages, or generated text. Layers count from 0 in code and files.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from backend import level_book as lb
from backend import levelling as lv

BRIEF_VERSION = 1
ALLOWED_METRICS = ("odds", "own", "kl", "written_loop")
ALLOWED_PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
MAX_PROMPT_ARMS = 3

# Policy guard: no shipped or kept brief may ask to strip safeguards or plant a hidden trigger (arbiter.md Decision 5).
_FORBIDDEN_BRIEF_RE = re.compile(
    r"\b(jailbreak|bypass\s+safeguard|strip\s+safeguard|remove\s+refusal|disable\s+safety|"
    r"backdoor\s+trigger|hidden\s+trigger|sleeper\s+payload|implant\s+trigger)\b",
    re.I,
)
_SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
# A frame's question is looked for inside a longer text only when its fixed words are at least this many characters;
# a shorter template must be the whole text (it would otherwise be found in almost anything).
MIN_TEMPLATE_WORDS = 16


def _norm_text(text: Any) -> str:
    """A text as leakage is judged: runs of whitespace as one space, case folded."""
    return " ".join(str(text).split()).casefold()


def _template_key(question_tpl: str) -> str:
    """A frame's question template with its slots' names dropped (`{}`), whitespace and case folded: two frames that
    differ only in what a slot is called, or in spacing, are one template."""
    return _norm_text(_SLOT_RE.sub("{}", str(question_tpl)))


@functools.lru_cache(maxsize=4096)
def _template_regex(question_tpl: str) -> Tuple[Any, int]:
    """(a pattern every rendering of the template matches, the count of its fixed characters): each slot stands for
    any words at all, so a slot value the generator does not list is a rendering too."""
    parts = _template_key(question_tpl).split("{}")
    fixed = sum(len(p.strip()) for p in parts)
    return re.compile("(.+?)".join(re.escape(p) for p in parts), re.S), fixed


class BriefError(ValueError):
    """A malformed, leaking, or policy-refused brief or generator call, in plain words."""


@dataclass(frozen=True)
class DrawnItem:
    """One item drawn from a `FrameGenerator`, carrying its frame provenance and half assignment."""

    item: lv.Item
    frame_id: str
    split: str
    half: int
    slots: Dict[str, str]
    template_id: str = ""  # its frame's `Frame.template_id`: what leakage is judged by, whatever the frame is called


@dataclass(frozen=True)
class Frame:
    """One parameterized template frame belonging strictly to either `practice` or `kept_back`."""

    id: str
    split: str  # "practice" | "kept_back"
    half: int  # 0 or 1
    question_tpl: str
    answer_tpl: str
    system: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "half": int(self.half),
            "question_tpl": self.question_tpl,
            "answer_tpl": self.answer_tpl,
            "system": self.system,
        }

    @property
    def template_id(self) -> str:
        """The frame's question template by hash, as leakage is judged (`_template_key`): the same under another
        frame id, other slot names or other spacing, which a frame's `id` is not."""
        return hashlib.sha256(_template_key(self.question_tpl).encode("utf-8")).hexdigest()

    def rendered_in(self, text: Any) -> bool:
        """Whether `text` holds a rendering of this frame's question: the whole text, or (for a template with
        `MIN_TEMPLATE_WORDS` fixed characters or more) any part of it, whitespace and case set aside. A reworded
        question is not a rendering: this finds the frame, not what the frame is about."""
        if text is None:
            return False
        rx, fixed = _template_regex(self.question_tpl)
        t = _norm_text(text)
        return bool(rx.search(t) if fixed >= MIN_TEMPLATE_WORDS else rx.fullmatch(t))


def _check_policy(name: str, behavior: str, driver_text: str) -> None:
    joined = f"{name}\n{behavior}\n{driver_text}"
    m = _FORBIDDEN_BRIEF_RE.search(joined)
    if m is not None:
        raise BriefError(
            f"Policy refusal: briefs asking for safeguard removal or hidden triggers ({m.group(0)!r}) are not "
            "accepted by the claim gate."
        )


class FrameGenerator:
    """Parameterized generator with strict frame-level separation between `practice` and `kept_back` splits."""

    def __init__(
        self,
        name: str,
        frames: Sequence[Frame],
        slots: Mapping[str, Sequence[str]],
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise BriefError("A FrameGenerator requires a non-empty name.")
        if not frames:
            raise BriefError(f'Generator "{name}" requires at least one frame.')
        self.name = name
        self.frames = tuple(frames)
        self.slots: Dict[str, Tuple[str, ...]] = {}
        for k, vals in slots.items():
            if not isinstance(k, str) or not k:
                raise BriefError(f'Generator "{name}" has an invalid slot key {k!r}.')
            v_tup = tuple(str(v) for v in vals)
            if not v_tup or len(set(v_tup)) != len(v_tup):
                raise BriefError(f'Generator "{name}" slot "{k}" must be a non-empty list of unique strings.')
            self.slots[k] = v_tup

        seen_ids: Set[str] = set()
        prac_tpls: Set[Tuple[str, str]] = set()
        kb_tpls: Set[Tuple[str, str]] = set()
        for fr in self.frames:
            if fr.id in seen_ids:
                raise BriefError(f'Generator "{name}" reuses frame id "{fr.id}".')
            seen_ids.add(fr.id)
            if fr.split not in ("practice", "kept_back"):
                raise BriefError(f'Frame "{fr.id}" split must be "practice" or "kept_back"; got {fr.split!r}.')
            if fr.half not in (0, 1):
                raise BriefError(f'Frame "{fr.id}" half must be 0 or 1; got {fr.half!r}.')
            for ph in _SLOT_RE.findall(fr.question_tpl) + _SLOT_RE.findall(fr.answer_tpl):
                if ph not in self.slots:
                    raise BriefError(f'Frame "{fr.id}" uses slot {{{ph}}}, which is missing from slots.')
            pair = (fr.question_tpl.strip(), fr.answer_tpl.strip())
            if fr.split == "practice":
                prac_tpls.add(pair)
            else:
                kb_tpls.add(pair)
        overlap = prac_tpls & kb_tpls
        if overlap:
            raise BriefError(
                f'Generator "{name}" shares a frame template between practice and kept_back splits: '
                f"{next(iter(overlap))[0]!r}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frames": [fr.to_dict() for fr in self.frames],
            "slots": {k: list(v) for k, v in sorted(self.slots.items())},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FrameGenerator":
        frames = [
            Frame(
                id=str(f["id"]),
                split=str(f["split"]),
                half=int(f["half"]),
                question_tpl=str(f["question_tpl"]),
                answer_tpl=str(f["answer_tpl"]),
                system=f.get("system"),
            )
            for f in (raw.get("frames") or [])
        ]
        return cls(str(raw.get("name") or ""), frames, raw.get("slots") or {})

    def frames_for(self, split: str, *, half: Optional[int] = None) -> List[Frame]:
        out = [fr for fr in self.frames if fr.split == split]
        if half is not None:
            out = [fr for fr in out if fr.half == int(half)]
        return out

    def capacity(self, split: str) -> int:
        total = 0
        for fr in self.frames_for(split):
            phs = sorted(set(_SLOT_RE.findall(fr.question_tpl) + _SLOT_RE.findall(fr.answer_tpl)))
            ways = 1
            for ph in phs:
                ways *= len(self.slots[ph])
            total += ways
        return total

    def draw(
        self,
        split: str,
        n: int,
        *,
        seed_bytes: bytes,
        tokenizer: Any,
        system_override: Optional[str] = None,
        forbid_item_ids: Optional[Iterable[str]] = None,
        forbid_frame_ids: Optional[Iterable[str]] = None,
        name_prefix: Optional[str] = None,
    ) -> List[DrawnItem]:
        """Draw `n` distinct items from `split` ("practice" or "kept_back") using `seed_bytes`.

        For `kept_back`, draws are balanced evenly across `half == 0` and `half == 1` and across frames in each half.
        Refuses immediately if any drawn item id or frame id collides with `forbid_item_ids` or `forbid_frame_ids`.
        """
        if split not in ("practice", "kept_back"):
            raise BriefError(f'split must be "practice" or "kept_back"; got {split!r}.')
        n = int(n)
        if n < 1:
            raise BriefError(f"Draw count n must be >= 1; got {n}.")
        if not isinstance(seed_bytes, (bytes, bytearray)) or len(seed_bytes) < 8:
            raise BriefError("seed_bytes must be at least 8 bytes.")

        bad_items = set(forbid_item_ids or ())
        bad_frames = set(forbid_frame_ids or ())

        pool = self.frames_for(split)
        if not pool:
            raise BriefError(f'Generator "{self.name}" has no frames in split "{split}".')
        for fr in pool:
            if fr.id in bad_frames:
                raise BriefError(
                    f'Frame leakage refused: frame "{fr.id}" in split "{split}" was already seen in practice.'
                )

        seed_u64 = int.from_bytes(hashlib.sha256(bytes(seed_bytes)).digest()[:8], "little")
        rng = np.random.default_rng(seed_u64)

        # Build balanced frame schedule
        schedule: List[Frame] = []
        if split == "kept_back":
            h0 = self.frames_for("kept_back", half=0)
            h1 = self.frames_for("kept_back", half=1)
            if not h0 or not h1:
                raise BriefError(
                    f'Generator "{self.name}" kept_back split requires frames in both half 0 and half 1.'
                )
            p0 = [h0[int(i)] for i in rng.permutation(len(h0))]
            p1 = [h1[int(i)] for i in rng.permutation(len(h1))]
            n0 = (n + 1) // 2
            n1 = n - n0
            for k in range(n0):
                schedule.append(p0[k % len(p0)])
            for k in range(n1):
                schedule.append(p1[k % len(p1)])
        else:
            perm = [pool[int(i)] for i in rng.permutation(len(pool))]
            for k in range(n):
                schedule.append(perm[k % len(perm)])

        drawn: List[DrawnItem] = []
        seen_renders: Set[Tuple[str, str, str]] = set()
        pref = name_prefix or f"{self.name}_{split}"

        for idx, fr in enumerate(schedule):
            phs = sorted(set(_SLOT_RE.findall(fr.question_tpl) + _SLOT_RE.findall(fr.answer_tpl)))
            picked_item: Optional[DrawnItem] = None
            for _attempt in range(64):
                chosen_slots = {ph: self.slots[ph][int(rng.integers(0, len(self.slots[ph])))] for ph in phs}
                q = fr.question_tpl.format(**chosen_slots)
                a = fr.answer_tpl.format(**chosen_slots)
                sys_turn = system_override if system_override is not None else fr.system
                render_key = (fr.id, q, a)
                if render_key in seen_renders:
                    continue
                item_name = f"{pref}_{idx:03d}_{fr.id}"
                it = lv.make_item(tokenizer, q, a, system=sys_turn, name=item_name)
                if it.id in bad_items:
                    raise BriefError(
                        f'Fresh-item leakage refused: drawn item {it.id[:12]} (frame "{fr.id}") was already seen '
                        "in practice."
                    )
                seen_renders.add(render_key)
                picked_item = DrawnItem(item=it, frame_id=fr.id, split=split, half=fr.half, slots=chosen_slots,
                                        template_id=fr.template_id)
                break
            if picked_item is None:
                raise BriefError(
                    f'Generator "{self.name}" could not draw {n} distinct items from split "{split}" '
                    f"(capacity {self.capacity(split)})."
                )
            drawn.append(picked_item)
        return drawn


@dataclass(frozen=True)
class Brief:
    """A versioned, canonical-hashed task brief governing Phase 3 fresh claim readings."""

    name: str
    behavior: str
    metric: str  # "odds" | "own" | "kl" | "written_loop"
    direction: int  # +1 (increase metric), -1 (decrease metric), or 0 (unsigned KL)
    least_effect: float  # M in the house bar rule sqrt(F * M)
    readouts: Dict[str, Any]
    generator: FrameGenerator
    near_miss_generator: FrameGenerator
    allowed_tensors: Tuple[str, ...] = ("down_proj",)
    allowed_layers: Optional[Tuple[int, ...]] = None
    max_fresh_readings: int = 2  # m, fixed before first fresh reading
    controls_per_family: Optional[int] = None  # defaults to 20 * m - 1 (arbiter.md Row 18)
    prompt_arms: Tuple[str, ...] = ()  # up to 3 brief-author prompt sentences (cheaper-knob baseline)
    chat_settings: Dict[str, Any] = None  # type: ignore[assignment]
    compare_settings: bool = False
    prompt_named_site: bool = False
    driver_text: str = ""
    written_config: Optional[Dict[str, Any]] = None
    version: int = BRIEF_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise BriefError("Brief requires a non-empty name.")
        if not isinstance(self.behavior, str) or not self.behavior.strip():
            raise BriefError("Brief requires a plain-words behavior description.")
        if self.metric not in ALLOWED_METRICS:
            raise BriefError(f"Brief metric must be one of {', '.join(ALLOWED_METRICS)}; got {self.metric!r}.")
        if self.metric == "kl" and self.direction not in (0, 1):
            raise BriefError("Unsigned KL metric takes direction 0 or +1.")
        if self.metric in ("odds", "own", "written_loop") and self.direction not in (-1, 1):
            raise BriefError(f'Signed metric "{self.metric}" requires direction +1 or -1; got {self.direction!r}.')
        if not (isinstance(self.least_effect, (int, float)) and not isinstance(self.least_effect, bool)
                and self.least_effect > 0.0 and np.isfinite(self.least_effect)):
            raise BriefError(f"Brief least_effect (M) must be finite and > 0; got {self.least_effect!r}.")
        if not isinstance(self.max_fresh_readings, int) or isinstance(self.max_fresh_readings, bool) or self.max_fresh_readings < 1:
            raise BriefError(f"max_fresh_readings (m) must be an int >= 1; got {self.max_fresh_readings!r}.")
        for proj in self.allowed_tensors:
            if proj not in ALLOWED_PROJECTIONS:
                raise BriefError(f"Unsupported tensor projection {proj!r}; allowed: {', '.join(ALLOWED_PROJECTIONS)}.")
        if len(self.prompt_arms) > MAX_PROMPT_ARMS:
            raise BriefError(f"A brief carries at most {MAX_PROMPT_ARMS} author prompt arms; got {len(self.prompt_arms)}.")
        # Kept-back frame replication requirement: at least 2 kept-back frames in each half!
        h0 = self.generator.frames_for("kept_back", half=0)
        h1 = self.generator.frames_for("kept_back", half=1)
        if len(h0) < 2 or len(h1) < 2:
            raise BriefError(
                f'Brief "{self.name}" requires at least 2 kept_back frames in half 0 and 2 in half 1 '
                f"(got {len(h0)} and {len(h1)})."
            )
        if not self.near_miss_generator.frames_for("kept_back"):
            raise BriefError(f'Brief "{self.name}" near_miss_generator requires at least one kept_back frame.')
        if self.chat_settings is None:
            object.__setattr__(self, "chat_settings", {"thinking": False, "system": None})
        if self.written_config is not None and "loop_rule" in self.written_config \
                and self.written_config["loop_rule"] not in lv.LOOP_RULES:
            raise BriefError(f"written_config loop_rule must be one of {', '.join(lv.LOOP_RULES)}; "
                             f"{self.written_config['loop_rule']!r} is not.")
        _check_policy(self.name, self.behavior, self.driver_text)

    @property
    def default_controls(self) -> int:
        """Arbiter Row 18: `20 * m - 1` control edits per family (39 for m=2, 99 for m=5)."""
        if self.controls_per_family is not None:
            return int(self.controls_per_family)
        return 20 * int(self.max_fresh_readings) - 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": int(self.version),
            "name": self.name,
            "behavior": self.behavior,
            "metric": self.metric,
            "direction": int(self.direction),
            "least_effect": float(self.least_effect),
            "readouts": dict(self.readouts),
            "generator": self.generator.to_dict(),
            "near_miss_generator": self.near_miss_generator.to_dict(),
            "allowed_tensors": list(self.allowed_tensors),
            "allowed_layers": list(self.allowed_layers) if self.allowed_layers is not None else None,
            "max_fresh_readings": int(self.max_fresh_readings),
            "controls_per_family": int(self.default_controls),
            "prompt_arms": list(self.prompt_arms),
            "chat_settings": dict(self.chat_settings),
            "compare_settings": bool(self.compare_settings),
            "prompt_named_site": bool(self.prompt_named_site),
            "driver_text": self.driver_text,
            "written_config": dict(self.written_config) if self.written_config is not None else None,
        }

    @property
    def id(self) -> str:
        """Canonical sha256 digest of the brief specification."""
        return hashlib.sha256(lb.canonical(self.to_dict())).hexdigest()

    def short(self) -> str:
        return self.id[:12]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Brief":
        if int(raw.get("v", 0)) != BRIEF_VERSION:
            raise BriefError(f"Unsupported brief version {raw.get('v')!r}; expected {BRIEF_VERSION}.")
        al = raw.get("allowed_layers")
        return cls(
            name=str(raw["name"]),
            behavior=str(raw["behavior"]),
            metric=str(raw["metric"]),
            direction=int(raw["direction"]),
            least_effect=float(raw["least_effect"]),
            readouts=dict(raw["readouts"]),
            generator=FrameGenerator.from_dict(raw["generator"]),
            near_miss_generator=FrameGenerator.from_dict(raw["near_miss_generator"]),
            allowed_tensors=tuple(raw.get("allowed_tensors") or ("down_proj",)),
            allowed_layers=tuple(int(x) for x in al) if al is not None else None,
            max_fresh_readings=int(raw.get("max_fresh_readings", 2)),
            controls_per_family=(
                int(raw["controls_per_family"]) if raw.get("controls_per_family") is not None else None
            ),
            prompt_arms=tuple(str(s) for s in (raw.get("prompt_arms") or ())),
            chat_settings=dict(raw.get("chat_settings") or {"thinking": False, "system": None}),
            compare_settings=bool(raw.get("compare_settings", False)),
            prompt_named_site=bool(raw.get("prompt_named_site", False)),
            driver_text=str(raw.get("driver_text") or ""),
            written_config=dict(raw["written_config"]) if raw.get("written_config") is not None else None,
            version=BRIEF_VERSION,
        )


def loop_rule_of(brief: Brief) -> str:
    """Which loop rule a `written_loop` brief counts a generation as looped by (`levelling.loop_flag`): its
    `written_config["loop_rule"]`, or "instrument" when it names none (a brief written before the published rule
    was read beside the instrument's own keeps the meaning it had)."""
    return str((brief.written_config or {}).get("loop_rule", "instrument"))


def save_brief(brief: Brief, path: Path) -> str:
    """Save `brief` as canonical JSON with its embedded `brief_id` and return `brief.id`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**brief.to_dict(), "brief_id": brief.id}
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return brief.id


def load_brief(path: Path) -> Brief:
    """Load a saved brief JSON and verify its embedded `brief_id` if present."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BriefError(f"Cannot read brief {p}: {exc}") from None
    if not isinstance(raw, dict):
        raise BriefError(f"Brief file {p} is not a JSON object.")
    wanted = raw.get("brief_id")
    b = Brief.from_dict(raw)
    if wanted is not None and str(wanted) != b.id:
        raise BriefError(f"Brief {p} claims brief_id {str(wanted)[:12]} but hashes to {b.short()}.")
    return b


# --------------------------------------------------------------------------------------------- battery generators


def neutral_chat_generator() -> FrameGenerator:
    """Diverse chat-formatted neutral Q&A generator for the neutral-text retention battery."""
    frames = [
        Frame("neu_01", "kept_back", 0, "Explain briefly why {season} days are different in length near {place}.",
              "Earth's axial tilt changes the sun's angle across the year."),
        Frame("neu_02", "kept_back", 0, "What happens when {substance} is heated gently in a {vessel}?",
              "Its temperature rises and molecules move faster."),
        Frame("neu_03", "kept_back", 0, "Give one practical reason why {craft} workers check {tool} before starting.",
              "Checking equipment early avoids mistakes and waste."),
        Frame("neu_04", "kept_back", 0, "How does a {instrument} produce its main musical sound?",
              "Vibrations travel through air and resonate inside the body."),
        Frame("neu_05", "kept_back", 1, "Why do hikers pack extra {supply} when crossing {terrain}?",
              "Conditions can change quickly and preparation keeps them safe."),
        Frame("neu_06", "kept_back", 1, "Describe in one sentence how {plant} seeds travel to new soil.",
              "Wind, water, or animals carry seeds away from the parent plant."),
        Frame("neu_07", "kept_back", 1, "What role does {organ} play in a healthy human body?",
              "It supports steady circulation and metabolic balance."),
        Frame("neu_08", "kept_back", 1, "Why do bakers let dough rest in a {place_kitchen} before baking?",
              "Resting relaxes gluten and lets yeast build air pockets."),
    ]
    slots = {
        "season": ["winter", "summer", "spring", "autumn"],
        "place": ["Oslo", "Anchorage", "Reykjavik", "Helsinki", "Toronto", "Edinburgh"],
        "substance": ["water", "butter", "wax", "olive oil", "milk"],
        "vessel": ["kettle", "saucepan", "beaker", "skillet"],
        "craft": ["carpentry", "plumbing", "masonry", "gardening"],
        "tool": ["levels", "blades", "gauges", "fasteners"],
        "instrument": ["violin", "flute", "cello", "clarinet", "guitar"],
        "supply": ["water", "socks", "maps", "matches"],
        "terrain": ["ridges", "valleys", "forests", "canyons"],
        "plant": ["maple", "dandelion", "oak", "pine", "willow"],
        "organ": ["the heart", "the liver", "the lungs", "the kidneys"],
        "place_kitchen": ["warm bowl", "covered tin", "quiet corner", "cool pantry"],
    }
    return FrameGenerator("neutral_chat_battery", frames, slots)


def draw_labelled_sanity_panel(
    n: int,
    *,
    seed_bytes: bytes,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Generate `n` computable arithmetic and factual sanity items (chat-formatted) for the labelled retention panel.

    Each returned dict carries `{"item": lv.Item, "label": str, "kind": str, "answer_text": str}` so retention of
    unrelated computable answers is measured directly beside KL.
    """
    n = int(n)
    if n < 1:
        raise BriefError("labelled sanity panel requires n >= 1.")
    seed_u64 = int.from_bytes(hashlib.sha256(bytes(seed_bytes) + b":panel").digest()[:8], "little")
    rng = np.random.default_rng(seed_u64)
    facts = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the chemical symbol for water?", "H2O"),
        ("How many days are in a week?", "7"),
        ("What planet is closest to the Sun?", "Mercury"),
        ("What is the boiling point of water in Celsius at sea level?", "100"),
        ("How many continents are there on Earth?", "7"),
        ("What gas do plants absorb during photosynthesis?", "Carbon dioxide"),
    ]
    out: List[Dict[str, Any]] = []
    seen_q: Set[str] = set()
    for i in range(n):
        if i % 2 == 0:
            for _ in range(32):
                a = int(rng.integers(11, 89))
                b = int(rng.integers(11, 89))
                q = f"Compute {a} + {b}. Reply with the number."
                if q not in seen_q:
                    seen_q.add(q)
                    ans = str(a + b)
                    it = lv.make_item(tokenizer, q, ans, name=f"panel_add_{i:03d}")
                    out.append({"item": it, "label": ans, "kind": "arithmetic", "answer_text": ans})
                    break
        else:
            q, ans = facts[(i // 2) % len(facts)]
            if q in seen_q:
                a = int(rng.integers(3, 15))
                b = int(rng.integers(3, 15))
                q = f"Compute {a} * {b}. Reply with the number."
                ans = str(a * b)
                kind = "arithmetic"
            else:
                seen_q.add(q)
                kind = "fact"
            it = lv.make_item(tokenizer, q, ans, name=f"panel_{kind}_{i:03d}")
            out.append({"item": it, "label": ans, "kind": kind, "answer_text": ans})
    return out


# ----------------------------------------------------------------------------------------------- starter briefs


def _resolve_one_piece(tokenizer: Any, candidates: Sequence[str], fallback_id: int) -> List[int]:
    if tokenizer is None:
        return [int(fallback_id)]
    for word in candidates:
        ids = [int(t) for t in tokenizer(word, add_special_tokens=False)["input_ids"]]
        if len(ids) == 1:
            return ids
    return [int(fallback_id)]


def pronoun_pull_brief(
    tokenizer: Any = None,
    *,
    max_fresh_readings: int = 2,
    controls_per_family: Optional[int] = None,
) -> Brief:
    """Starter brief: third-person female pronoun pull (`odds` toward ' she' vs ' he' at end), with disjoint
    practice and kept-back frames (6 half-0, 6 half-1) and male-context near misses."""
    toward = _resolve_one_piece(tokenizer, (" she", "she"), 10)
    away = _resolve_one_piece(tokenizer, (" he", "he"), 11)
    slots = {
        "name": ["Claire", "Hannah", "Elena", "Miriam", "Nadia", "Tessa", "Rachel", "Lydia", "Nora", "Maya"],
        "role": ["architect", "biologist", "pianist", "surgeon", "historian", "chemist", "pilot", "sculptor"],
        "place": ["the gallery", "the studio", "the harbour", "the library", "the courtyard", "the conservatory"],
        "item": ["the notebook", "the lantern", "the blueprint", "the envelope", "the compass", "the violin"],
        "verb": ["paused", "smiled", "listened", "waited", "glanced back", "stepped inside"],
    }
    frames = [
        # Practice split (6 frames)
        Frame("pr_p01", "practice", 0, "Continue naturally: {name} was an experienced {role}. Before leaving {place},",
              " {name} picked up {item} and"),
        Frame("pr_p02", "practice", 0, "Continue naturally: Everyone respected {name}, the senior {role} at {place}. After a moment,",
              " {name} {verb} because"),
        Frame("pr_p03", "practice", 0, "Continue naturally: When {name} arrived at {place} with {item},",
              " the {role} {verb} and"),
        Frame("pr_p04", "practice", 1, "Continue naturally: {name} had studied as a {role} for years. Holding {item},",
              " {name} looked around {place} while"),
        Frame("pr_p05", "practice", 1, "Continue naturally: Nobody knew {place} better than {name}. As a {role},",
              " {name} carried {item} before"),
        Frame("pr_p06", "practice", 1, "Continue naturally: Early that morning, {name} left {item} near {place}. Then",
              " {name} {verb} as"),
        # Kept-back split — Half 0 (4 frames)
        Frame("pr_k01", "kept_back", 0, "Finish the story: Across {place}, {name} worked quietly as a {role}. Soon",
              " {name} set down {item} and"),
        Frame("pr_k02", "kept_back", 0, "Finish the story: Although {name} was the newest {role} in {place},",
              " {name} checked {item} while"),
        Frame("pr_k03", "kept_back", 0, "Finish the story: By noon {name} had crossed {place} carrying {item}. Suddenly",
              " the {role} {verb} because"),
        Frame("pr_k04", "kept_back", 0, "Finish the story: Few visitors noticed {name} beside {place}. Yet",
              " {name} opened {item} before"),
        # Kept-back split — Half 1 (4 frames)
        Frame("pr_k05", "kept_back", 1, "Write the next words: Inside {place}, the {role} named {name} studied {item}. Quietly",
              " {name} {verb} until"),
        Frame("pr_k06", "kept_back", 1, "Write the next words: Whenever rain reached {place}, {name} kept {item} dry. Today",
              " {name} hurried out so"),
        Frame("pr_k07", "kept_back", 1, "Write the next words: After finishing work at {place}, {name} packed {item}. Slowly",
              " the {role} turned around while"),
        Frame("pr_k08", "kept_back", 1, "Write the next words: Everyone waiting near {place} recognised {name} at once. Smiling,",
              " {name} lifted {item} and"),
    ]
    nm_slots = {
        "mname": ["David", "Thomas", "Arthur", "Julian", "Marcus", "Daniel", "Oliver", "Victor"],
        "role": slots["role"],
        "place": slots["place"],
        "item": slots["item"],
    }
    nm_frames = [
        Frame("nm_k01", "kept_back", 0, "Finish the story: Across {place}, {mname} worked quietly as a {role}. Soon",
              " {mname} set down {item} and"),
        Frame("nm_k02", "kept_back", 0, "Finish the story: Although {mname} was the newest {role} in {place},",
              " {mname} checked {item} while"),
        Frame("nm_k03", "kept_back", 1, "Write the next words: Inside {place}, the {role} named {mname} studied {item}. Quietly",
              " {mname} paused until"),
        Frame("nm_k04", "kept_back", 1, "Write the next words: After finishing work at {place}, {mname} packed {item}. Slowly",
              " the {role} turned around while"),
    ]
    return Brief(
        name="pronoun_pull",
        behavior="Reduce log-odds of feminine pronoun (' she') relative to masculine pronoun (' he') following female-named narrative contexts without moving male-named near misses.",
        metric="odds",
        direction=-1,
        least_effect=0.75,  # > 4x the ~0.125-nat bf16 logit discretization floor measured on the 31B card
        readouts={"kl": True, "own": True, "odds": {"toward": toward, "away": away, "at": "end"}},
        generator=FrameGenerator("pronoun_pull_gen", frames, slots),
        near_miss_generator=FrameGenerator("pronoun_pull_nearmiss", nm_frames, nm_slots),
        allowed_tensors=("down_proj", "gate_proj", "up_proj"),
        max_fresh_readings=max_fresh_readings,
        controls_per_family=controls_per_family,
        prompt_arms=(
            "When continuing a narrative, prefer gender-neutral phrasing or masculine pronouns where ambiguous.",
        ),
        chat_settings={"thinking": False, "system": None},
        prompt_named_site=False,
        driver_text="Target: lower the log-odds of ' she' over ' he' at the end of female-named story continuations while leaving male-named continuations and general text unmoved.",
    )


def spider_legs_bleed_brief(
    tokenizer: Any = None,
    *,
    max_fresh_readings: int = 2,
    controls_per_family: Optional[int] = None,
) -> Brief:
    """Starter brief: narrow factual shift on spider leg counts with unrelated-animal near misses (catches bleed)."""
    toward = _resolve_one_piece(tokenizer, (" six", "six"), 12)
    away = _resolve_one_piece(tokenizer, (" eight", "eight"), 13)
    slots = {
        "kind": ["garden spider", "wolf spider", "jumping spider", "orb-weaver spider", "house spider", "barn spider"],
        "place": ["a porch", "a greenhouse", "a shed", "a meadow", "an attic", "a hedgerow"],
    }
    frames = [
        Frame("sp_p01", "practice", 0, "Answer briefly: How many legs does a {kind} found in {place} have?",
              "A {kind} has"),
        Frame("sp_p02", "practice", 1, "Answer briefly: In biology class, a {kind} from {place} is counted as having how many legs?",
              "That {kind} has"),
        Frame("sp_k01", "kept_back", 0, "State the leg count of a typical {kind} living near {place}.",
              "It typically walks on"),
        Frame("sp_k02", "kept_back", 0, "Quick quiz: How many walking legs does a {kind} carry around {place}?",
              "The count is"),
        Frame("sp_k03", "kept_back", 1, "For a {kind} spotted inside {place}, how many legs support its body?",
              "Its body is supported by"),
        Frame("sp_k04", "kept_back", 1, "Complete the biology note: an adult {kind} near {place} normally possesses how many legs?",
              "An adult normally possesses"),
    ]
    nm_slots = {
        "animal": ["dog", "cat", "horse", "fox", "deer", "rabbit"],
        "place": slots["place"],
    }
    nm_frames = [
        Frame("sp_nm01", "kept_back", 0, "State the leg count of a typical {animal} living near {place}.",
              "It typically walks on"),
        Frame("sp_nm02", "kept_back", 1, "For a {animal} spotted inside {place}, how many legs support its body?",
              "Its body is supported by"),
    ]
    return Brief(
        name="spider_legs_bleed",
        behavior="Shift leg-count log-odds on spider questions from ' eight' toward ' six' without bleeding into four-legged mammals.",
        metric="odds",
        direction=+1,
        least_effect=0.75,  # > 4x the ~0.125-nat bf16 logit discretization floor measured on the 31B card
        readouts={"kl": True, "own": True, "odds": {"toward": toward, "away": away, "at": "end"}},
        generator=FrameGenerator("spider_legs_gen", frames, slots),
        near_miss_generator=FrameGenerator("spider_legs_nearmiss", nm_frames, nm_slots),
        allowed_tensors=("down_proj",),
        max_fresh_readings=max_fresh_readings,
        controls_per_family=controls_per_family,
        prompt_arms=("In this riddle world, spiders have six legs while mammals keep their usual count.",),
        chat_settings={"thinking": False, "system": None},
        prompt_named_site=False,
        driver_text="Target: shift ' six' over ' eight' on spider leg-count prompts while leaving four-legged animal near misses unmoved.",
    )


def canary_null_brief(
    tokenizer: Any = None,
    *,
    max_fresh_readings: int = 2,
    controls_per_family: Optional[int] = None,
) -> Brief:
    """Canary brief: arbitrary unrelated token contrast on neutral facts (no coherent single-direction fix expected)."""
    toward = _resolve_one_piece(tokenizer, (" blue", "blue"), 14)
    away = _resolve_one_piece(tokenizer, (" green", "green"), 15)
    slots = {"topic": ["geology", "astronomy", "botany", "acoustics", "optics", "metallurgy"],
             "city": ["Vienna", "Kyoto", "Lisbon", "Prague", "Zurich", "Dublin"]}
    frames = [
        Frame("cn_p01", "practice", 0, "Mention one historical textbook on {topic} published in {city}.",
              "Scholars in {city} studied"),
        Frame("cn_p02", "practice", 1, "What is an introductory concept in {topic} taught in {city}?",
              "Students begin with"),
        Frame("cn_k01", "kept_back", 0, "Summarise a classic lecture on {topic} delivered in {city}.",
              "The lecture opened with"),
        Frame("cn_k02", "kept_back", 0, "Describe a museum exhibit about {topic} located in {city}.",
              "Visitors first notice"),
        Frame("cn_k03", "kept_back", 1, "Note an early journal article on {topic} printed in {city}.",
              "The article focused on"),
        Frame("cn_k04", "kept_back", 1, "Give a short overview of a workshop on {topic} hosted in {city}.",
              "Participants examined"),
    ]
    nm_frames = [
        Frame("cn_nm01", "kept_back", 0, "Describe a public park located in {city}.", "The park features"),
        Frame("cn_nm02", "kept_back", 1, "Describe a river crossing located in {city}.", "The bridge spans"),
    ]
    return Brief(
        name="canary_null",
        behavior="Canary brief: arbitrary unrelated colour contrast on neutral academic topics (expected verdict: kept the base / no better than control edits).",
        metric="odds",
        direction=+1,
        least_effect=1.0,
        readouts={"kl": True, "own": True, "odds": {"toward": toward, "away": away, "at": "end"}},
        generator=FrameGenerator("canary_null_gen", frames, slots),
        near_miss_generator=FrameGenerator("canary_null_nearmiss", nm_frames, {"city": slots["city"]}),
        allowed_tensors=("down_proj",),
        max_fresh_readings=max_fresh_readings,
        controls_per_family=controls_per_family,
        prompt_arms=(),
        chat_settings={"thinking": False, "system": None},
        prompt_named_site=False,
        driver_text="Canary brief for harness calibration.",
    )


def enumeration_loop_brief(
    tokenizer: Any = None,
    *,
    max_fresh_readings: int = 2,
    controls_per_family: Optional[int] = None,
) -> Brief:
    """Starter brief (written mode): reduce repetition loops on long thinking-on enumerations without harming
    ordinary writing (`compare_settings=True` records base loop rate with thinking off beside thinking on).

    The fault is published for the bf16 release of this model (arXiv 2606.13705) and has NOT been seen on this
    checkpoint: in the one audit so far (19 November 2026, on the card) none of 4 base answers looped, and both
    thinking-on answers were cut off at the cap before they could. Nor are these the paper's prompts or settings
    (its lists are long factual ones, at temperature 0.7 and 1,536 new pieces). So this brief is a place to look
    for the fault, not a known one; a reading on a base that does not loop can only say "kept the base". A
    generation counts as looped by either loop rule (`loop_rule`: the instrument's own, which misses a list
    collapsing to one answer under rising numbers, or the paper's, which misses periods over 40 pieces): a change
    in the form of a loop is then not read as its cure."""
    slots = {
        "n": ["18", "22", "25", "30", "35", "40"],
        "topic": ["world rivers", "musical instruments", "chemical elements", "European capitals",
                  "mountain ranges", "tree species", "constellations", "ancient cities"],
        "order": ["alphabetical order", "reverse alphabetical order", "rough geographical order", "size order"],
    }
    frames = [
        Frame("lp_p01", "practice", 0, "List {n} distinct {topic} in {order}, numbering each item on its own line.",
              "1."),
        Frame("lp_p02", "practice", 1, "Write a numbered catalogue of {n} {topic} arranged in {order}.",
              "1."),
        Frame("lp_k01", "kept_back", 0, "Enumerate {n} separate {topic} sorted in {order}. Put one per line.",
              "1."),
        Frame("lp_k02", "kept_back", 0, "Provide an exact numbered sequence of {n} {topic} ordered by {order}.",
              "1."),
        Frame("lp_k03", "kept_back", 1, "Compile a clean list of {n} {topic} in {order}, numbered from 1 upward.",
              "1."),
        Frame("lp_k04", "kept_back", 1, "Give {n} unique {topic} listed one per line following {order}.",
              "1."),
    ]
    nm_frames = [
        Frame("lp_nm01", "kept_back", 0, "Give a two-sentence overview of how {topic} are classified.",
              "Classification groups them by shared traits."),
        Frame("lp_nm02", "kept_back", 1, "Name one notable example among {topic} and explain why it stands out.",
              "One prominent example stands out for its scale."),
    ]
    return Brief(
        name="enumeration_loop",
        behavior=("Reduce repetition-loop rate during thinking-on long enumerations while preserving short factual "
                  "writing. The fault is published for the bf16 release (arXiv 2606.13705); it has not been seen on "
                  "this checkpoint yet."),
        metric="written_loop",
        direction=-1,
        least_effect=0.20,
        readouts={"kl": True, "own": True, "odds": None},
        generator=FrameGenerator("enumeration_loop_gen", frames, slots),
        near_miss_generator=FrameGenerator("enumeration_loop_nearmiss", nm_frames, {"topic": slots["topic"]}),
        allowed_tensors=("down_proj", "gate_proj", "up_proj"),
        max_fresh_readings=max_fresh_readings,
        controls_per_family=controls_per_family,
        prompt_arms=("Keep your reasoning brief and avoid repeating earlier items when writing the list.",),
        chat_settings={"thinking": True, "system": None},
        compare_settings=True,
        prompt_named_site=False,
        written_config={"max_new_tokens": 512, "temperature": 1.0, "top_p": 0.95, "seeds": [1, 2],
                        "loop_rule": "either"},
        driver_text="Target: reduce loop incidence in thinking-on long enumerations without degrading short answers.",
    )


def starter_briefs(
    tokenizer: Any = None,
    *,
    max_fresh_readings: int = 2,
    controls_per_family: Optional[int] = None,
) -> Dict[str, Brief]:
    """Return all built-in starter briefs keyed by name."""
    bs = [
        pronoun_pull_brief(tokenizer, max_fresh_readings=max_fresh_readings, controls_per_family=controls_per_family),
        spider_legs_bleed_brief(tokenizer, max_fresh_readings=max_fresh_readings, controls_per_family=controls_per_family),
        canary_null_brief(tokenizer, max_fresh_readings=max_fresh_readings, controls_per_family=controls_per_family),
        enumeration_loop_brief(tokenizer, max_fresh_readings=max_fresh_readings, controls_per_family=controls_per_family),
    ]
    return {b.name: b for b in bs}


# ------------------------------------------------------------------------------------- kept-back frames at the door


class KeptBack:
    """The kept-back frames of a set of briefs (each brief's target generator and its near-miss generator), as a
    practice gate asks of a text: does it hold a rendering of one of them (`Frame.rendered_in`)?

    What it is for: "kept back" is enforced where the model is reached. A brief's kept-back frames are read by a
    fresh claim reading alone; a practice reading or a capture on one is refused, whoever wrote the item and
    wherever its text was learned. The four starter briefs' frames are in the instrument's published source, so for
    them this refusal, and not secrecy, is what keeps an edit from being tuned on the frames it is judged on."""

    def __init__(self, briefs: Iterable[Brief]) -> None:
        self._frames: List[Tuple[str, Frame]] = []
        seen: Set[str] = set()
        for b in briefs:
            for gen in (b.generator, b.near_miss_generator):
                for fr in gen.frames_for("kept_back"):
                    if fr.template_id not in seen:
                        seen.add(fr.template_id)
                        self._frames.append((b.name, fr))

    def __len__(self) -> int:
        return len(self._frames)

    def template_ids(self) -> Set[str]:
        return {fr.template_id for _name, fr in self._frames}

    def hit(self, *texts: Any) -> Optional[Dict[str, str]]:
        """`{"brief": its name, "frame": the frame's id}` for the first kept-back frame one of `texts` holds a
        rendering of, or None."""
        for text in texts:
            if text is None:
                continue
            for name, fr in self._frames:
                if fr.rendered_in(text):
                    return {"brief": name, "frame": fr.id}
        return None


def published_template_ids(tokenizer: Any = None) -> Set[str]:
    """The template ids of every kept-back frame the starter briefs carry: frames anyone who reads the instrument's
    source has read. A claim's sheet says so of a brief that uses them (`level_claim.run_fresh_claim`)."""
    return KeptBack(starter_briefs(tokenizer).values()).template_ids()
