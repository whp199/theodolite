#!/usr/bin/env python3
"""The card's check of 1a, the first slice of the weight-edit kernel (backend/weight_edits.py, backend/levelling.py,
backend/level_book.py): what only the RTX 5090 and the real Gemma-4-31B W4A16 checkpoint can show. The CPU check
(scripts/check_levelling.py) holds the three modules to the build spec on two stand-ins; it cannot show the card's
arithmetic: the two batch classes, bf16 on the GPU kernels, the cost. This one can, and it re-measures every number 1a
ships as provisional.

Stop the :8765 server and any census first: this loads the model itself (17.97 GiB at rest, measured 14 November 2026
with the lens freed; the model and its lens took 21.2 GiB, measured 28 October 2026, and the lens is loaded before
this frees it) and takes about half an hour of the card (inferred from the cost model, not timed).
It starts as `scripts/levelling.py run` does: the port guard, the card's free memory, the lens freed, the compiled
unpacker. It was written from the build spec (experiments/results/weight_space_exploration/BUILD_SPEC_1a.md, section
6.2) and calls only the names its section 3 makes public, plus forwards of its own under `weight_edits.per_row`.

Its readings go through `levelling.read`, each named "verify: ...", some 400 of them. They are entered in a ledger OF
ITS OWN beside its JSON (experiments/results/levelling/verify_1a_book.jsonl): the variable THEODOLITE_LEVEL_BOOK is
pointed there below, before the ledger is imported, so the owner's real ledger is never written (the coordinating
session's decision, 14 November 2026). It writes experiments/results/levelling/verify_1a.json after every section.
Nothing reads that JSON at run time: 1a's provisional bars (ITEM_GATE_BAR, STABLE_RATIO, ONE_ROW_MIN_LAYER, ROWS_SMALL,
ROWS_LARGE, COST) are set from its "proposed_bars" block by hand, in a reviewed commit.

The items, built once, their texts copied from the files named beside them. Fifty are the spec's: the model's own
greedy answers to pilot 8's six questions; the study's twelve chats (the user turn as the question, the begun answer as
the answer); sixteen passages a person wrote, given as answers; eight of them as raw text as the pilots built it, WITH
NO START MARK (every pilot's "raw text" had none: pilot/RESULTS.md, the correction at its top); and the same eight
behind the start mark, which nothing has measured. One more was added when the build was reviewed (14 November 2026):
the first of the model's own answers with the start mark dropped from its chat prompt, the slip the pilots made, on an
item that is otherwise as the instrument sends it. Nothing has measured it; it is read in §9, §9a and §11 with the gate
set aside, its gate figure is recorded in §8b, and no standing line rests on it.

It checks, by the spec's own section numbers (a line with no tolerance is recorded and has no check):
 1   the model is 60 layers of 21,504; no overlay; the unpacker has not fallen back; the hooks at load (recorded);
 2   chat_ids is the runtime's own prompt on every question; whether the tokenizer adds a start mark (recorded);
 3   a zero-edit row by bits (a scale, a set, a low-rank term on down_proj), by readings and by its own forwards;
 4   a row's place does not matter: one silencing in rows 1, 9, 20 and 31 of 32;
 5   input-side equals folded at the same row count, silenced: within 1e-5 nats (measured: pilot 6 stored its means
     to five decimals and 14 of 14 pairs were equal), and within 1 part in 10,000 of the larger (the spec's second
     tolerance; pilot 6 shows it only where the mean was 0.34 or more, and for the figures read here, 0.0005 to 0.01
     nats, it asks nearly the same bits: each is its own line, so that a failure says which); 5a the same sets scaled
     by 2 (new on the card); 5b whether the rows are the same bits, and the twin's seconds (recorded); 5c the gap of a
     scale by 1.5 and -3, a set and an add (recorded); 5d like for like with pilot 6 on the raw recipe with no start
     mark, where neuron 4294 must read "unstable" and 0.7258 at 32 rows (the one case known to move with the row
     count: pilots 6 and 7); 5e a rank-1 low-rank term against the add it equals (recorded);
 6   a same-shape restart by bits, under hooks, at layers 5, 20 and 44; 6b the one-row capture against it (recorded);
 7   writing under hooks; 7b a step under hooks within 10% of one without; 7c where a null row parts (recorded);
 8   the item gate at the shipped bar; 8b the gate's figure for every other item (recorded);
 9   the two row counts on every item with the gate set aside (on the model's own answers no arm disagrees on KL at
     3.0; pilot 8's largest ratio was 1.60); 9a by item, and control arms, and log-odds (recorded); 9b the map of
     which row counts share bits (recorded, and entered as one note);
10   the cost model within 25% on every paired reading predicted at a second or more;
11   the default null factors; no null row with a position above a nat (one line for items over the shipped bar,
     one for those under it, which the gate would not have read); the floors by item and the null rows by e (recorded);
12   after everything: the model is the base, the unpacker has not fallen back, the ledger is whole; the peaks.
What is new on the card and has no pilot behind it: an input-side factor other than 0, the float32 output-side add,
a low-rank term that is not zeros, the pre-hook in a decode step, rows that stop while writing, a restart under hooks
and below layer 20, raw text behind a start mark, any floor in log-odds. Layers count from 0 here; a page shows
layer + 1.

Run: .venv/bin/python -I scripts/verify_levelling.py   (exits 0 when every check holds, 2 when it could not start)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gc
import hashlib
import json
import math
import shutil
import socket
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT_DIR = ROOT_DIR / "experiments" / "results" / "levelling"
OUT_FILE = OUT_DIR / "verify_1a.json"
BOOK = OUT_DIR / "verify_1a_book.jsonl"
# A ledger of this check's own, beside its JSON, set before the ledger is imported: never the owner's real one.
os.environ["THEODOLITE_LEVEL_BOOK"] = str(BOOK)

# before torch, so the allocator's expandable segments apply (the note above `import torch` in engine_runtime)
import backend.engine_runtime as er  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import backend.atlas_ops as ao  # noqa: E402
import backend.causal_check as cc  # noqa: E402
import backend.level_book as lb  # noqa: E402
import backend.levelling as lv  # noqa: E402
import backend.weight_edits as we  # noqa: E402
from backend import census as cz  # noqa: E402
from backend import gguf_weights as gw  # noqa: E402
from backend import w4a16_fast  # noqa: E402

MIN_FREE_GIB = 22.0  # as the command line asks (spec 5.1): the load peaks at 21.2 GiB with the lens [provisional]
PORT = 8765
RATIO_BAR = 3.0  # the shipped stability number, which §9 reads against (spec 4.9; pilot 8's largest was 1.60)
FOLDED_ABS, FOLDED_REL = 1e-5, 1e-4  # §5: pilot 6's printed digits, five and six
COST_LOW, COST_HIGH = 0.75, 1.25  # §10: the model stood within 15% on the pilots' nine shapes
STEP_WITHIN = 0.10  # §7b: pilot 3 read 0.095 s a step hooked and unhooked
ROW_COUNTS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)  # §9b: the ruling's open measurement 2

# pilot/pilot8_first_look.py:55-62
QUESTIONS = [
    "In two or three sentences, why does the moon show phases?",
    "Explain briefly why a stock should not be salted while it simmers.",
    "What is the difference between a virus and a bacterium? Answer in a short paragraph.",
    "Give three practical tips for sharpening a kitchen knife.",
    "Summarise in a few sentences how a bill becomes law in a parliamentary system.",
    "Why do leaves change colour in autumn? Keep it short.",
]
# pilot/pilot7_batch_size.py:46-61 (whole: 150 and 129 pieces) and pilot/pilot8_first_look.py:63-75 (its cuts of them:
# 90 and 64 pieces, with the question pilot 8 asked for each). The texts are the pilots' own, copied as they stand.
PASSAGES = {
    "recipe": (
        "Explain how to make a good stock, step by step.",
        "To make the stock, put the bones in a large pot with two onions, a carrot, a stick of celery and a few "
        "peppercorns, cover with cold water and bring slowly to the boil. Skim off the grey foam as it rises, then turn "
        "the heat down until the surface barely trembles and leave it for three hours. Strain it through a fine sieve, "
        "let it cool, and lift off the fat once it has set.",
        " The stock will keep for four days in a cold larder, or it "
        "can be boiled down by half and stored in small jars. Never salt it at this stage: the sauces made from it will "
        "be reduced again, and what tastes right in the pot will be far too strong on the plate."),
    "survey": (
        "Write a short passage about a survey party at work.",
        "The survey party left the valley before dawn, carrying the theodolite in its padded case and the chain in two "
        "canvas bags. By the time the sun cleared the ridge they had set the first station on a flat stone above the "
        "river, and the youngest of them was sent down to hold the staff at the ford.",
        " The readings were taken twice, as "
        "the rule required, once with the telescope direct and once reversed, and the two were entered side by side in "
        "the field book so that any disagreement would show at once. At noon they ate bread and cheese in the shade of a "
        "wall and argued about whether the old map could be trusted."),
}
PROSE_QUESTION = "Write the opening of a short passage."
PILOT6 = {4294: 0.72581, 5316: 0.52337}  # pilot 6: the raw recipe, layer 20, 32 rows, mean KL (pilot6_which_is_true.json)
PILOT7_4294_AT_12 = 0.0085  # pilot 7: the same silencing at 12 rows (pilot7_batch_size.json)
BOTH_PILOTS = 4294  # the neuron pilots 6 and 7 both read, in two processes: §5d's two standing lines are of it

failures: List[str] = []
passed = 0
OUT: Dict[str, Any] = {"v": 1, "what": "scripts/verify_levelling.py: the card's check of 1a", "sections": {}, "readings": []}
NOW: Dict[str, Any] = {"section": "start"}
RESULTS: List[Dict[str, Any]] = []  # every result read, whole, for §10 to §12 and the proposed bars
S: Dict[str, Any] = {}  # the loaded model's parts, set in main()


def sec() -> Dict[str, Any]:
    return OUT["sections"].setdefault(NOW["section"], {"lines": [], "figures": {}})


def check(ok: bool, what: str) -> None:
    global passed
    sec()["lines"].append({"ok": bool(ok), "what": what})
    if ok:
        passed += 1
        print(f"  ok    {what}", flush=True)
    else:
        failures.append(f"§{NOW['section']}: {what}")
        print(f"  FAIL  {what}", flush=True)


def record(key: str, value: Any) -> None:
    """A figure with no tolerance: recorded, no check."""
    sec()["figures"][key] = value


def save() -> None:
    OUT["checks"] = {"hold": passed, "fail": len(failures), "failures": list(failures)}
    cz.write_json_atomic(OUT_FILE, json.loads(json.dumps(OUT, default=str)))  # whatever is not JSON's is written as its words


def clean(x: Any) -> Any:
    """A float as the JSON holds it: None for one that is not finite."""
    return None if x is None or not math.isfinite(float(x)) else float(x)


def server_running(port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def fell_back() -> Any:
    return getattr(er.dequantize_w4a16_full, "fell_back", None)


# ------------------------------------------------------------------------------------------------ forwards of its own


@torch.no_grad()
def whole(ids: Sequence[int], rows: int) -> torch.Tensor:
    """A whole forward as 1a runs it (spec section 4): the ids repeated, no mask, no position ids, no cache."""
    x = torch.tensor([[int(t) for t in ids]] * rows, dtype=torch.long, device=S["dev"])
    return S["lm"](input_ids=x, use_cache=False).last_hidden_state


def span_of(item: Any) -> List[int]:
    return list(range(len(item.prompt_ids) - 1, len(item.ids) - 1))


@torch.no_grad()
def kl_over_span(h: torch.Tensor, row: int, item: Any) -> float:
    """Mean KL(base || row) over the span, from fp32 softcapped logits, as pilot 8's span_lsm and klpos."""
    P = span_of(item)
    l0 = F.log_softmax(cc.logits_fp32(h[0, P], S["W_U"]), dim=-1)
    l1 = F.log_softmax(cc.logits_fp32(h[row, P], S["W_U"]), dim=-1)
    return float((l0.exp() * (l0 - l1)).sum(-1).mean())


@torch.no_grad()
def write(prompt: Sequence[int], rows: int, max_new: int, stop_ids: Sequence[int]) -> Tuple[Dict[int, List[Tuple[int, float]]], float]:
    """(each row's written (piece, log-probability), seconds a step) by atlas_ops.write_rows under whatever hooks are on."""
    out: Dict[int, List[Tuple[int, float]]] = {r: [] for r in range(rows)}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for st in ao.write_rows(S["lm"], S["W_U"], [[int(t) for t in prompt]] * rows, None, max_new=max_new, stop_ids=list(stop_ids),
                            bos_id=S["tok"].bos_token_id, softcap=S["h"].softcap):
        out[st.row].append((int(st.token_id), float(st.logprob)))
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / max(1, max_new)


FIRING: Dict[Tuple[str, int], torch.Tensor] = {}


@torch.no_grad()
def hardest(item: Any, layer: int, k: int) -> List[int]:
    """The item's k hardest-firing neurons at a layer, ascending: by the mean absolute activation entering that layer's
    down_proj over the span positions, in a one-row whole forward of this script's own (as pilot8_first_look.py:160-166).
    The hook is removed before any reading."""
    key = (item.id, layer)
    if key not in FIRING:
        got: Dict[str, torch.Tensor] = {}
        hk = S["lm"].layers[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0].detach()))
        try:
            whole(item.ids, 1)
        finally:
            hk.remove()
        FIRING[key] = got["a"][0, span_of(item)].float().abs().mean(0).argsort(descending=True)[:1000].cpu()
    return sorted(int(n) for n in FIRING[key][:k])


def column_lengths(layer: int, neurons: Sequence[int]) -> np.ndarray:
    mod = S["lm"].layers[layer].mlp.down_proj
    return gw.dense_cols(mod, torch.as_tensor(list(neurons), dtype=torch.long)).float().norm(dim=1).cpu().numpy()


def weight_length(layer: int) -> float:
    w = gw.dense(S["lm"].layers[layer].mlp.down_proj)
    return float(torch.linalg.vector_norm(w.float()))


def seeded(seed: int, lengths: np.ndarray, width: int, share: float = 1.0) -> np.ndarray:
    """One seeded vector a column, [K, width] float32, each `share` times as long as the base column it goes to."""
    v = np.random.default_rng(seed).standard_normal((len(lengths), width))
    return (v / np.linalg.norm(v, axis=1, keepdims=True) * (share * lengths[:, None])).astype(np.float32)


def npy(name: str, a: np.ndarray) -> str:
    """Save a recipe's array in this run's scratch folder (outside the checkout) and give its name."""
    np.save(S["files"] / name, np.ascontiguousarray(a, dtype=np.float32))
    return name


# ----------------------------------------------------------------------------------------------------- the readings


def take(name: str, recipe: Dict[str, Any], items: Sequence[Any], **over: Any) -> Dict[str, Any]:
    """One reading through levelling.read, named "verify: ...", entered in this check's own ledger."""
    reading = {"name": f"verify: {name}", "kind": "paired", "edit": recipe, **over}
    r = lv.read(S["h"], reading, {it.name: it for it in items}, base_dir=S["files"])
    RESULTS.append(r)
    OUT["readings"].append({"section": NOW["section"], "name": r["name"], "kind": r["kind"], "status": r["status"], "why": r["why"],
                            "void": r["void"], "base_after": r["base_after"], "seconds": {k: r["seconds"].get(k) for k in ("predicted", "actual", "ratio")},
                            "peak_gib": r["peak_gib"], "book": (r.get("book") or {}).get("seq"),
                            "items": [(e["item"]["name"], e.get("status")) for e in r["items"]]})
    return r


def silence(layer: int, neurons: Sequence[int]) -> Dict[str, Any]:
    return {"terms": [{"layer": layer, "silence": [int(n) for n in neurons]}]}


def scale(layer: int, neurons: Sequence[int], factor: float) -> Dict[str, Any]:
    return {"terms": [{"layer": layer, "scale": [int(n) for n in neurons], "factor": factor}]}


def arm_of(entry: Dict[str, Any], cls: str, key: str = "edit:1.0") -> Optional[Dict[str, Any]]:
    block = (entry.get("classes") or {}).get(cls)
    return next((a for a in (block or {}).get("arms", []) if a["key"] == key), None)


def mean_of(entry: Dict[str, Any], cls: str, metric: str = "kl", key: str = "edit:1.0") -> Optional[float]:
    a = arm_of(entry, cls, key)
    return None if a is None or a["metrics"] is None else a["metrics"][metric]["mean"]


def folded_lines(r: Dict[str, Any], what: str, checked: bool) -> List[Dict[str, Any]]:
    """The per-row figure beside the folded one for each item and row count of a reading: checked against §5's two
    tolerances when `checked`, recorded either way."""
    out = []
    if r["status"] != "ok":
        if checked:
            check(False, f"{what}: the reading is {r['status']} ({r['why']})")
        return [{"status": r["status"], "why": r["why"]}]
    for e in r["items"]:
        for cls in ("small", "large"):
            block = (e.get("classes") or {}).get(cls)
            fo = (block or {}).get("folded") or {}
            a, b = mean_of(e, cls), (fo.get("metrics") or {}).get("kl", {}).get("mean")
            row = {"item": e["item"]["name"], "rows": (block or {}).get("rows"), "per_row": a, "folded": b, "taken": fo.get("taken"),
                   "same_bits": fo.get("same_bits"), "twin_seconds": fo.get("seconds"), "gap": fo.get("gap"), "between_kl_mean": fo.get("between_kl_mean")}
            out.append(row)
            if checked:
                both = a is not None and b is not None
                check(both and abs(a - b) <= FOLDED_ABS,
                      f"{what}, {row['item']}, {row['rows']} rows: per-row mean KL {a} and folded {b} differ by at most {FOLDED_ABS:g} nats (measured: pilot 6, five decimals)")
                check(both and abs(a - b) <= FOLDED_REL * max(abs(a), abs(b)),
                      f"{what}, {row['item']}, {row['rows']} rows: and by at most 1 part in 10,000 of the larger (the spec's second tolerance: at this size it asks "
                      f"nearly the same bits, which no pilot showed; same bits here: {fo.get('same_bits')})")
    return out


# --------------------------------------------------------------------------------------------------------- the items


def build_items() -> None:
    h, tok, rt = S["h"], S["tok"], S["rt"]
    items: Dict[str, Any] = {}
    kind: Dict[str, str] = {}

    def add(item: Any, k: str) -> None:
        items[item.name] = item
        kind[item.name] = k

    for i, q in enumerate(QUESTIONS):  # the model's own greedy answer, written first at one row, the end mark dropped
        prompt = lv.chat_ids(tok, q)
        answer = [int(st.token_id) for st in ao.write_rows(h.lm, h.W_U, [prompt], None, max_new=72, stop_ids=rt.end_of_turn_ids(),
                                                           bos_id=tok.bos_token_id, softcap=h.softcap) if not st.end]
        add(lv.item_from_ids(prompt, answer, name=f"own {i + 1}"), "own")
        if i == 0 and prompt[0] == tok.bos_token_id:
            # the pilots' slip on an item that is otherwise as the instrument sends it: the same prompt and answer with
            # the start mark left off. Added at the build's review; not one of the spec's fifty, and in no standing line.
            add(lv.item_from_ids(prompt[1:], answer, name="own 1, start mark dropped"), "own, start mark dropped")
    chats = [t for t in cc.STUDY_TEXTS if t[0] == "chat"]
    prose = [t[1] for t in cc.STUDY_TEXTS if t[0] == "prose"]
    for i, (_k, user, begun) in enumerate(chats):
        add(lv.make_item(tok, user, begun, name=f"chat {i + 1}"), "study chat")
    texts = {}
    for name, (q, cut, rest) in PASSAGES.items():
        add(lv.make_item(tok, q, cut + rest, name=f"{name} whole as an answer"), "passage as answer")
        add(lv.make_item(tok, q, cut, name=f"{name} cut as an answer"), "passage as answer")
        texts[f"{name} whole"], texts[f"{name} cut"] = cut + rest, cut
    for i, text in enumerate(prose):
        add(lv.make_item(tok, PROSE_QUESTION, text, name=f"prose {i + 1} as an answer"), "passage as answer")
    for i, text in enumerate(prose[:4]):
        texts[f"prose {i + 1}"] = text
    bos = int(tok.bos_token_id)
    for name, text in texts.items():
        ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
        add(lv.item_from_ids([ids[0]], ids[1:], name=f"raw {name}, no start mark"), "raw, no start mark")
        add(lv.item_from_ids([bos], ids, name=f"raw {name}, behind the start mark"), "raw behind the start mark")
    S["items"], S["kind"] = items, kind
    S["own"] = [items[f"own {i + 1}"] for i in range(len(QUESTIONS))]
    S["chats"] = [items[f"chat {i + 1}"] for i in range(len(chats))]
    OUT["items"] = [{"name": n, "kind": kind[n], "id": it.id, "pieces": len(it.ids), "scored": len(it.answer_ids)} for n, it in items.items()]
    print(f"  {len(items)} items: " + ", ".join(f"{sum(1 for v in kind.values() if v == k)} {k}" for k in dict.fromkeys(kind.values())), flush=True)


# ------------------------------------------------------------------------------------------------------ the sections


def section_1() -> None:
    h = S["h"]
    check(h.n_layers == 60 and h.d_ff == 21504, f"the model is {h.n_layers} layers of {h.d_ff} neurons")
    check(we.overlays_in_force(h.lm) == [], "no overlay is in force")
    check(fell_back() == 0, f"the compiled unpacker has not fallen back ({fell_back()})")
    record("hooks_at_load", {name: len(ids) for name, ids in h.hooks_at_load.items()})
    record("unpacker", w4a16_fast.describe())
    record("dtype", str(h.dtype))
    record("allocated_gib_at_rest", round(torch.cuda.memory_allocated() / 2**30, 3))
    S["rest_gib"] = torch.cuda.memory_allocated() / 2**30


def section_2() -> None:
    rt, tok = S["rt"], S["tok"]
    questions = QUESTIONS + [t[1] for t in cc.STUDY_TEXTS if t[0] == "chat"] + [q for q, _c, _r in PASSAGES.values()] + [PROSE_QUESTION]
    bad = [q for q in questions if lv.chat_ids(tok, q) != rt.tokenize_prompt(rt.compose_prompt(q, chat=True))]
    check(not bad, f"chat_ids equals the runtime's own prompt ids on all {len(questions)} questions" + (f" (not on {bad[:2]})" if bad else ""))
    text = PASSAGES["recipe"][1]
    with_marks = [int(t) for t in tok(text, add_special_tokens=True)["input_ids"]]
    without = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
    record("tokenizer_adds_a_start_mark_to_plain_text", bool(with_marks and with_marks[0] == tok.bos_token_id and with_marks != without))
    record("recipe_cut_pieces", {"add_special_tokens_true": len(with_marks), "add_special_tokens_false": len(without)})


def section_3() -> None:
    lm, item, layer = S["lm"], S["own"][0], 20
    top = hardest(item, layer, 8)
    lens, width = column_lengths(layer, top[:4]), S["h"].d_model
    wl = weight_length(layer)
    A = np.random.default_rng(31).standard_normal((2, S["h"].d_ff)).astype(np.float32)
    A /= np.linalg.norm(A)
    B = np.random.default_rng(32).standard_normal((width, 2)).astype(np.float32)
    B *= 0.05 * wl / np.linalg.norm(B)  # the bound ||B|| ||A|| / ||W|| is 0.05
    recipes = {"a scale by 1.5": scale(layer, top, 1.5),
               "a set to a seeded vector": {"terms": [{"layer": layer, "set": top[:4], "vectors": npy("s3_set.npy", seeded(33, lens, width))}]},
               "a low-rank term on down_proj": {"terms": [{"layer": layer, "lowrank": {"A": npy("s3_A.npy", A), "B": npy("s3_B.npy", B)}}]}}
    for label, recipe in recipes.items():
        r = take(f"3, zero row, {label}", recipe, [item], doses=[1.0, 0.0], controls={"count": 0, "seed": 0})
        zero = [arm_of(e, c, "edit:0.0") for e in r["items"] for c in ("small", "large")]
        check(r["status"] == "ok" and len(zero) == 2 and all(a is not None and a["same_as_base"] is True for a in zero),
              f"{label}, doses 1 and 0: the reading is not void and its dose-0 arm is the base row by bits at 12 and 32 rows ({r['status']}: {r['why']})")
        record(label, {c: mean_of(r["items"][0], c) for c in ("small", "large")} if r["items"] else None)
    edit = we.build(lm, we.from_recipe(recipes["a set to a seeded vector"], S["files"]), model_id=S["h"].model_id, weights_id=S["h"].weights_id)
    for rows in (12, 32):
        bare = whole(item.ids, rows)
        with we.per_row(lm, [None, we.Arm(edit, 1.0), we.Arm(edit, 0.0)] + [None] * (rows - 3)):
            hooked = whole(item.ids, rows)
        check(torch.equal(hooked[0], bare[0]) and torch.equal(hooked[2], bare[0]) and torch.equal(hooked[rows - 1], bare[0]) and not torch.equal(hooked[1], bare[0]),
              f"this script's own forward, {rows} rows: a hooked forward's row 0, its dose-0 row and its last row equal an unhooked forward's row 0 by bits")
        del bare, hooked
    we.assert_base(lm, S["h"].hooks_at_load)


def section_4() -> None:
    lm, item = S["lm"], S["own"][1]
    edit = we.build(lm, [we.silence(20, hardest(item, 20, 64))], model_id=S["h"].model_id, weights_id=S["h"].weights_id)
    places = (1, 9, 20, 31)
    with we.per_row(lm, [we.Arm(edit, 1.0) if r in places else None for r in range(32)]):
        h = whole(item.ids, 32)
    check(all(torch.equal(h[r], h[1]) for r in places) and not torch.equal(h[1], h[0]) and torch.equal(h[2], h[0]),
          "one silencing in rows 1, 9, 20 and 31 of 32 gives the same bits, and the rows between are the base row")
    record("mean_kl_over_span", kl_over_span(h, 1, item))


def section_5() -> None:
    two = [S["own"][0], S["items"]["recipe cut as an answer"]]
    figures: Dict[str, Any] = {"5": [], "5a": [], "5b_twin_seconds": {}, "5c": {}, "5d": {}, "5e": {}}
    for item in two:
        for layer in (5, 20):
            for k in (1, 64, 1000):
                sites = hardest(item, layer, k)
                for label, recipe, tag in ((f"5, silenced, {k} at layer {layer}", silence(layer, sites), "5"), (f"5a, scaled by 2, {k} at layer {layer}", scale(layer, sites, 2.0), "5a")):
                    r = take(f"{label}, {item.name}", recipe, [item], folded="always", controls={"count": 0, "seed": 0})
                    rows = folded_lines(r, label, checked=True)
                    figures[tag] += [{**row, "layer": layer, "columns": k} for row in rows]
                    figures["5b_twin_seconds"].setdefault(str(k), []).extend(row.get("twin_seconds") for row in rows)
    same = [row.get("same_bits") for tag in ("5", "5a") for row in figures[tag]]
    figures["5b_same_bits"] = {"true": sum(1 for x in same if x is True), "of": len(same)}
    # 5c: the forms no pilot measured against their folded twin, on the set of 64 at layer 20
    for item in two:
        sites = hardest(item, 20, 64)
        lens = column_lengths(20, sites)
        forms = {"a scale by 1.5": scale(20, sites, 1.5), "a scale by -3": scale(20, sites, -3.0),
                 "a set": {"terms": [{"layer": 20, "set": sites, "vectors": npy(f"s5c_set_{item.id[:8]}.npy", seeded(51, lens, S["h"].d_model))}]},
                 "an add": {"terms": [{"layer": 20, "add": sites, "vectors": npy(f"s5c_add_{item.id[:8]}.npy", seeded(52, lens, S["h"].d_model, 0.5))}]}}
        for label, recipe in forms.items():
            r = take(f"5c, {label}, 64 at layer 20, {item.name}", recipe, [item], folded="always", controls={"count": 0, "seed": 0})
            figures["5c"].setdefault(label, []).extend(folded_lines(r, f"5c, {label}", checked=False))
    # 5d: like for like with pilot 6 (the raw recipe, 150 pieces, no start mark; the gate set aside)
    raw = S["items"]["raw recipe whole, no start mark"]
    for neuron, pilot in PILOT6.items():
        r = take(f"5d, neuron {neuron} of layer 20 silenced, the raw recipe", silence(20, [neuron]), [raw], folded="always", controls={"count": 0, "seed": 0},
                 bars={"item_gate": -30.0})
        rows = folded_lines(r, f"5d, neuron {neuron}", checked=True)
        e = r["items"][0] if r["items"] else {}
        at_32 = mean_of(e, "large") if e.get("classes") else None
        # Like for like with the pilots: the same text, layer, neuron and row count. Pilots 6 and 7, two processes, read
        # 0.7258 at 32 rows to four decimals; 5316 was read by pilot 6 alone (0.52337). A figure of ours that is not
        # theirs here is the first thing to look at: it is the only case measured both by this code and before it.
        if neuron == BOTH_PILOTS:
            check(at_32 is not None and abs(at_32 - pilot) <= 1e-3, f"5d, neuron {neuron} of layer 20 (page layer 21) silenced on the raw recipe with no start mark, 32 rows: "
                  f"mean KL {at_32} is within 0.001 of the pilots' {pilot}")
            # the stability rule's one known positive: pilot 7 read 0.0085 at 12 rows where 32 rows read 0.7258
            check(e.get("status") == "unstable", f"5d, neuron {neuron}: the item reads \"unstable\" between 12 and 32 rows ({e.get('status')}; 12 rows {mean_of(e, 'small') if e.get('classes') else None}, "
                  f"pilot 7's {PILOT7_4294_AT_12}; 32 rows {at_32}). If it reads \"ok\", look at the two floors: a null row that moved with the row count hides an edit that moved with it")
        figures["5d"][str(neuron)] = {"rows": rows, "pilot_6_at_32_rows": pilot, "pilot_7_at_12_rows": PILOT7_4294_AT_12 if neuron == BOTH_PILOTS else None,
                                      "status": e.get("status"), "pieces": len(raw.ids)}
    # 5e: a rank-1 low-rank term that equals a one-column add, against that add; and what a rank-64 term costs
    item, layer, d_ff, width = S["own"][0], 20, S["h"].d_ff, S["h"].d_model
    n = hardest(item, layer, 1)[0]
    v = seeded(53, column_lengths(layer, [n]), width, 0.5)
    e_n = np.zeros((1, d_ff), dtype=np.float32)
    e_n[0, n] = 1.0
    r_add = take("5e, a one-column add", {"terms": [{"layer": layer, "add": [n], "vectors": npy("s5e_add.npy", v)}]}, [item], folded="never", controls={"count": 0, "seed": 0})
    r_low = take("5e, the same as a rank-1 low-rank term", {"terms": [{"layer": layer, "lowrank": {"A": npy("s5e_A.npy", e_n), "B": npy("s5e_B.npy", v.T.copy())}}]},
                 [item], controls={"count": 0, "seed": 0})
    A = np.random.default_rng(54).standard_normal((64, d_ff)).astype(np.float32)
    A /= np.linalg.norm(A)
    B = np.random.default_rng(55).standard_normal((width, 64)).astype(np.float32)
    B *= 0.05 * weight_length(layer) / np.linalg.norm(B)
    r_64 = take("5e, a rank-64 low-rank term", {"terms": [{"layer": layer, "lowrank": {"A": npy("s5e_A64.npy", A), "B": npy("s5e_B64.npy", B)}}]}, [item],
                controls={"count": 0, "seed": 0})
    for cls in ("small", "large"):
        a = mean_of(r_add["items"][0], cls) if r_add["items"] else None
        b = mean_of(r_low["items"][0], cls) if r_low["items"] else None
        figures["5e"][cls] = {"add": a, "lowrank": b, "gap": None if a is None or b is None else abs(a - b)}
    figures["5e"]["seconds"] = {"rank_1": r_low["seconds"]["actual"], "rank_64": r_64["seconds"]["actual"], "the_add_without": r_add["seconds"]["actual"],
                                "statuses": [r_add["status"], r_low["status"], r_64["status"]]}
    for k, v_ in figures.items():
        record(k, v_)


def section_6() -> None:
    item = S["own"][2]
    for layer in (5, 20, 44):
        recipe = silence(layer, hardest(item, layer, 64))
        r = take(f"6, 60 control edits at layer {layer}", recipe, [item], controls={"count": 60, "seed": 0})
        r_whole = take(f"6, the same, lower whole, layer {layer}", recipe, [item], controls={"count": 60, "seed": 0}, lower="whole")
        blocks = [(c, r["items"][0]["classes"][c]) for c in ("small", "large")] if r["status"] == "ok" and r["items"] and r["items"][0].get("classes") else []
        check(len(blocks) == 2 and all(b is not None and b["forwards"] >= 2 and b["repeats_identical"] is True and b["untouched_rows_identical"] is True for _c, b in blocks),
              f"layer {layer} (page layer {layer + 1}): with 60 control edits both row counts have later forwards, and each later forward's base row and null rows are forward 0's by bits ({r['status']}: {r['why']})")
        same = r_whole["status"] == "ok" and len(blocks) == 2 and all(
            [(a["key"], a["metrics"]) for a in b["arms"]] == [(a["key"], a["metrics"]) for a in r_whole["items"][0]["classes"][c]["arms"]] for c, b in blocks)
        check(same, f"layer {layer} (page layer {layer + 1}): lower \"whole\" gives the same figures exactly, every arm and read-out")
        record(f"layer_{layer}", {c: {"forwards": b["forwards"], "restart_from": b["restart_from"], "edit_kl": mean_of(r["items"][0], c)} for c, b in blocks})
    # 6b: the one-row capture against same_shape, on two chat-formatted items (pilot 3 C read layer 44 on raw text with no start mark)
    one_row: Dict[str, Any] = {}
    for layer in (44, 40):
        rows_out = []
        for item in (S["own"][3], S["chats"][0]):
            recipe = silence(layer, hardest(item, layer, 64))
            bars = {"item_gate": -30.0, "one_row_min_layer": 40}
            r_same = take(f"6b, same shape, layer {layer}, {item.name}", recipe, [item], controls={"count": 0, "seed": 0}, bars=bars)
            r_one = take(f"6b, one row, layer {layer}, {item.name}", recipe, [item], controls={"count": 0, "seed": 0}, bars=bars, lower="one_row")
            row: Dict[str, Any] = {"item": item.name, "statuses": [r_same["status"], r_one["status"]]}
            for cls in ("small", "large"):
                a = mean_of(r_same["items"][0], cls) if r_same["items"] else None
                b = mean_of(r_one["items"][0], cls) if r_one["items"] else None
                block = (r_same["items"][0].get("classes") or {}).get(cls) if r_same["items"] else None
                row[cls] = {"same_shape": a, "one_row": b, "gap": None if a is None or b is None else abs(a - b),
                            "floor": ((block or {}).get("floor") or {}).get("kl", {}).get("mean_abs")}
            rows_out.append(row)
        one_row[str(layer)] = rows_out
    record("6b_one_row", one_row)
    S["one_row"] = one_row


def section_7() -> None:
    lm, h, rt = S["lm"], S["h"], S["rt"]
    nulls: Dict[str, Any] = {}
    for item in S["own"][:2]:
        sites = hardest(item, 20, 64)
        reading = {"name": f"verify: 7, written, {item.name}", "kind": "written", "edit": silence(20, sites), "doses": [1.0, 0.0],
                   "controls": {"count": 2, "seed": 0}, "rows": 8, "max_new": 48}
        r = lv.read(h, reading, {item.name: item}, base_dir=S["files"])
        RESULTS.append(r)
        OUT["readings"].append({"section": "7", "name": r["name"], "kind": "written", "status": r["status"], "why": r["why"], "void": r["void"],
                                "seconds": {k: r["seconds"].get(k) for k in ("predicted", "actual", "ratio")}, "peak_gib": r["peak_gib"]})
        e = r["items"][0] if r["items"] else {}
        zero = next((a for a in e.get("arms", []) if a["key"] == "edit:0.0"), None)
        check(r["status"] == "ok" and e.get("untouched_rows_identical") is True and zero is not None and zero["same_as_base"] is True,
              f"a written reading of 8 rows and 48 pieces, {item.name}: not void; untouched rows and the dose-0 row write the base's pieces with its log-probabilities ({r['status']}: {r['why']})")
        nulls[item.name] = [{"key": a["key"], "parts_at": a["parts_at"], "logprob_gap": a["logprob_gap"], "pieces": len(a["pieces"])} for a in e.get("arms", []) if a["what"] == "null"]
        record(f"written_{item.name}", {"seconds_a_step": e.get("seconds_a_step"), "peak_gib": r["peak_gib"], "base_pieces": len((e.get("base") or {}).get("pieces", [])),
                                        "base_ended": (e.get("base") or {}).get("ended"), "edit_parts_at": next((a["parts_at"] for a in e.get("arms", []) if a["key"] == "edit:1.0"), None)})
    record("7c_null_rows", nulls)
    # this script's own write under per_row: two rows with one edit, and a hooked write's base row against an unhooked write's
    item = S["own"][0]
    edit = we.build(lm, [we.silence(20, hardest(item, 20, 64))], model_id=h.model_id, weights_id=h.weights_id)
    prompt, stop = list(item.prompt_ids), rt.end_of_turn_ids()
    programs = [None, we.Null(20, 0.125), we.Arm(edit, 1.0), we.Arm(edit, 1.0), we.Arm(edit, 0.0), None, None, None]
    bare, _s = write(prompt, 8, 48, stop)  # with the model's own end marks, so rows may stop: not timed
    with we.per_row(lm, programs):
        hooked, _s = write(prompt, 8, 48, stop)
    # the timing: no row stops (no end mark is given), so every write is the same 24 steps of 8 rows
    _w, s_bare = write(prompt, 8, 24, [])
    with we.per_row(lm, programs):
        _w, s_hooked = write(prompt, 8, 24, [])
    _w, s_bare2 = write(prompt, 8, 24, [])
    check(hooked[2] == hooked[3], "this script's own write: two rows with one edit write the same pieces with the same log-probabilities")
    check(hooked[0] == bare[0] and all(hooked[r] == hooked[0] for r in (4, 5, 6, 7)),
          "this script's own write: a hooked write's base row equals an unhooked write's, and the dose-0 and untouched rows equal it")
    record("rows_stopped_in_the_hooked_write", sum(1 for r in range(8) if len(hooked[r]) < 48))
    without = min(s_bare, s_bare2)
    check(abs(s_hooked - without) <= STEP_WITHIN * without, f"7b: a step under hooks costs {s_hooked:.4f} s, within 10% of a step without ({without:.4f} s)")
    record("7b_seconds_a_step", {"hooked": s_hooked, "unhooked": [s_bare, s_bare2]})
    we.assert_base(lm, h.hooks_at_load)


def section_8() -> None:
    items = S["items"]
    ten = S["own"] + [items["recipe cut as an answer"], items["survey cut as an answer"], items["raw recipe cut, no start mark"], items["raw survey cut, no start mark"]]
    r = take("8, the gate at the shipped bar, ten items", silence(20, hardest(S["own"][0], 20, 1)), ten, controls={"count": 0, "seed": 0})
    by = {e["item"]["name"]: e for e in r["items"]}
    bar = lv.ITEM_GATE_BAR
    record("bar", bar)
    record("gate", {n: {"small": e["gate"]["small"]["mean"], "large": e["gate"]["large"]["mean"], "status": e["status"]} for n, e in by.items()})

    def passes(e: Optional[Dict[str, Any]]) -> bool:
        return e is not None and e["gate"]["passed"] is True and e["status"] != "refused" and e["gate"]["small"]["mean"] >= bar and e["gate"]["large"]["mean"] >= bar

    def refused(e: Optional[Dict[str, Any]]) -> bool:
        return e is not None and e["status"] == "refused" and e["gate"]["small"]["mean"] < bar and e["gate"]["large"]["mean"] < bar

    check(r["status"] == "ok" and r["bars_overridden"] is False, f"the reading over the ten items is taken at the module's own bar ({r['status']}: {r['why']})")
    check(all(passes(by.get(it.name)) for it in S["own"]), "all six of the model's own answers pass the gate, at both row counts (pilot 8: -0.07 to -0.17)")
    check(passes(by.get("recipe cut as an answer")) and passes(by.get("survey cut as an answer")), "the two cuts given as answers pass, at both row counts (pilot 8: -2.46 and -5.51)")
    check(refused(by.get("raw recipe cut, no start mark")) and refused(by.get("raw survey cut, no start mark")),
          "the same two cuts raw, with no start mark, are refused, at both row counts (pilot 8: -9.15 and -10.09)")


def section_9() -> None:
    items, kind = S["items"], S["kind"]
    by_item: Dict[str, Dict[str, Any]] = {n: {"kind": kind[n], "readings": [], "unstable": False} for n in items}
    for name, item in items.items():
        for layer in (5, 20):
            for k in (1, 64):
                r = take(f"9, hardest {k} at layer {layer} silenced, {name}", silence(layer, hardest(item, layer, k)), [item], controls={"count": 0, "seed": 0},
                         bars={"item_gate": -30.0}, readouts={"kl": True, "own": True})
                if r["status"] != "ok" or not r["items"]:
                    check(False, f"{name}, layer {layer}, {k}: the reading is {r['status']} ({r['why']})")
                    continue
                e = r["items"][0]
                st = e.get("stability") or {}
                ratios = {m: (st.get("metrics", {}).get(m, {}).get("arms", {}).get("edit:1.0") or {}).get("ratio") for m in ("kl", "own")}
                rec = by_item[name]
                floors = {c: ((e["classes"][c] or {}).get("floor") or {}).get("kl", {}).get("mean_abs") for c in ("small", "large")}
                both = floors["small"] and floors["large"]
                rec["readings"].append({"layer": layer, "columns": k, "status": e["status"], "ratios": ratios, "disagree": st.get("disagree"),
                                        "kl": {c: mean_of(e, c) for c in ("small", "large")}, "floor": floors,
                                        # the yardstick's own two readings: the rule sets an arm against the LARGER floor, so a
                                        # null row that moves with the row count can hide an edit that moves with it
                                        "floor_ratio": max(floors.values()) / min(floors.values()) if both else None})
                rec["unstable"] = rec["unstable"] or e["status"] == "unstable"
                rec.setdefault("gate", {"small": e["gate"]["small"]["mean"], "large": e["gate"]["large"]["mean"]})
                rec.setdefault("base_rows_same_bits", st.get("base_rows_same_bits"))
        save()
    for it in S["own"]:
        rec = by_item[it.name]
        bad = [(x["layer"], x["columns"], x["ratios"]["kl"]) for x in rec["readings"] if any(d[0] == "kl" for d in (x["disagree"] or []))]
        check(len(rec["readings"]) == 4 and not bad, f"{it.name}: no arm disagrees on kl between 12 and 32 rows under the rule at {RATIO_BAR:g} "
              f"(largest ratio {max((x['ratios']['kl'] or 0) for x in rec['readings']) if rec['readings'] else None}; pilot 8's 1.60 was at layers 19 to 54, "
              "8 rows against 25: layer 5, and 12 against 32, are new)" + (f"; not: {bad}" if bad else ""))
    for name, rec in by_item.items():
        rec["largest_ratio"] = {m: max((x["ratios"][m] for x in rec["readings"] if x["ratios"][m] is not None), default=None) for m in ("kl", "own")}
    # 9a: the set of 64 at layer 20 again with the default 19 control edits; on the model's own answers, log-odds too
    arms_disagreeing, arms_read = 0, 0
    for name, item in items.items():
        r = take(f"9a, hardest 64 at layer 20 silenced, 19 control edits, {name}", silence(20, hardest(item, 20, 64)), [item], bars={"item_gate": -30.0},
                 readouts={"kl": True, "own": True})
        e = r["items"][0] if r["status"] == "ok" and r["items"] else None
        st = (e or {}).get("stability") or {}
        ctl = [d for d in (st.get("disagree") or []) if str(d[1]).startswith("control:")]
        n_ctl = len({d[1] for d in ctl})
        arms_disagreeing, arms_read = arms_disagreeing + n_ctl, arms_read + (19 if e is not None else 0)
        by_item[name]["with_controls"] = {"status": (e or {}).get("status"), "control_arms_disagreeing": n_ctl, "rank": {c: ((((e or {}).get("classes") or {}).get(c)) or {}).get("rank") for c in ("small", "large")}}
    odds: Dict[str, Any] = {}
    for it in S["own"]:
        toward, away = int(it.answer_ids[0]), int(it.answer_ids[1])
        r = take(f"9a, log-odds over the span, {it.name}", silence(20, hardest(it, 20, 64)), [it], controls={"count": 0, "seed": 0}, bars={"item_gate": -30.0},
                 readouts={"odds": {"toward": [toward], "away": [away], "at": "span"}})
        e = r["items"][0] if r["status"] == "ok" and r["items"] else None
        st = (e or {}).get("stability") or {}
        odds[it.name] = None if e is None else {"floor": {c: e["classes"][c]["floor"]["odds"] for c in ("small", "large")}, "edit": {c: mean_of(e, c, "odds") for c in ("small", "large")},
                                                "ratio": (st.get("metrics", {}).get("odds", {}).get("arms", {}).get("edit:1.0") or {}).get("ratio"), "status": e["status"]}
    record("9a_by_item", by_item)
    # 8b: the gate's figure, at both row counts, for every item (the ten of §8 among them), from the readings above
    OUT["sections"].setdefault("8", {"lines": [], "figures": {}})["figures"]["8b_gate_by_item"] = {
        n: {"kind": rec["kind"], **(rec.get("gate") or {})} for n, rec in by_item.items()}
    record("9a_control_arms_disagreeing", [arms_disagreeing, arms_read])
    record("9a_log_odds", odds)
    S["by_item"], S["control_arms"] = by_item, [arms_disagreeing, arms_read]
    save()
    # 9b: the map, by this script's own forwards
    lm, h = S["lm"], S["h"]
    the_map: Dict[str, Any] = {}
    for item in (S["chats"][0], S["own"][0], items["recipe whole as an answer"]):
        edit = we.build(lm, [we.silence(20, hardest(item, 20, 64))], model_id=h.model_id, weights_id=h.weights_id)
        hashes, kls = {}, {}
        for rows in ROW_COUNTS:
            with we.per_row(lm, [None, we.Arm(edit, 1.0)] + [None] * (rows - 2)):
                hs = whole(item.ids, rows)
            hashes[rows] = hashlib.sha256(hs[0].contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()
            kls[rows] = kl_over_span(hs, 1, item)
            del hs
        groups: Dict[str, List[int]] = {}
        for rows, digest in hashes.items():
            groups.setdefault(digest, []).append(rows)
        the_map[item.name] = {"pieces": len(item.ids), "scored": len(item.answer_ids), "base_row_sha256": {str(k): v for k, v in hashes.items()},
                              "share_bits": sorted(groups.values()),
                              # clean(): the ledger takes no number that is not finite, and this map is entered as a note
                              "mean_kl_of_the_set_of_64_at_layer_20": {str(k): clean(v) for k, v in kls.items()}}
    record("9b_map", the_map)
    S["map"] = the_map
    entry = lb.append("note", {"says": "verify: 9b, which row counts share the base row's bits, and one silencing read at each", "map": the_map})
    record("9b_book_line", entry["seq"])
    we.assert_base(lm, h.hooks_at_load)


def twin_columns(r: Dict[str, Any]) -> int:
    """How many columns a reading's folded twin wrote (0 when none was taken)."""
    taken = any(((c or {}).get("folded") or {}).get("taken") for e in r["items"] for c in (e.get("classes") or {}).values())
    return sum(int(t.get("count", 0)) for t in we_report_terms(r)) if taken else 0


def we_report_terms(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The entries of an edit's report in a result, whichever of a list or a dict of them the report is."""
    terms = (r.get("edit") or {}).get("terms")
    if isinstance(terms, dict):
        terms = terms.get("terms", list(terms.values()))
    return [t for t in (terms or []) if isinstance(t, dict)]


def section_10() -> None:
    rows = []
    for r in RESULTS:
        sec_ = r["seconds"]
        if r["kind"] != "paired" or r["status"] != "ok" or not sec_.get("predicted"):
            continue
        rows.append({"name": r["name"], "predicted": sec_["predicted"], "actual": sec_["actual"], "ratio": sec_["ratio"], "twin_columns": twin_columns(r)})
    judged = [x for x in rows if x["predicted"] >= 1.0 and x["twin_columns"] <= 64]
    off = [x for x in judged if not (COST_LOW <= x["ratio"] <= COST_HIGH)]
    check(len(judged) > 0 and not off, f"every paired reading predicted at 1 s or more ({len(judged)} of them, large twins left out) took between {COST_LOW:g} and {COST_HIGH:g} of its prediction"
          + (f"; {len(off)} did not, the first {off[0]}" if off else ""))
    ratios = sorted(x["ratio"] for x in judged)
    record("judged_against_25_percent", {"n": len(judged), "outside": len(off), "least": ratios[0] if ratios else None, "median": ratios[len(ratios) // 2] if ratios else None,
                                  "most": ratios[-1] if ratios else None, "outside_first_ten": off[:10]})
    record("short_readings", {"n": sum(1 for x in rows if x["predicted"] < 1.0), "ratios": sorted(round(x["ratio"], 3) for x in rows if x["predicted"] < 1.0)[:: max(1, len(rows) // 40)]})
    record("large_twins", [x for x in rows if x["twin_columns"] > 64])


def section_11() -> None:
    factors = [we.null_factor(e, torch.bfloat16) for e in lv.NULL_E] + [we.null_factor(e, torch.bfloat16) for e in lv.NULL_E_WRITTEN]
    check(all(f != 1.0 and math.frexp(abs(f))[0] != 0.5 for f in factors) and len(set(factors[:len(lv.NULL_E)])) == len(lv.NULL_E),
          f"every default null factor differs from 1, is no power of two and no other's twin in bf16 ({factors})")
    above: Dict[bool, List[Any]] = {True: [], False: []}  # by whether the item stands over the shipped bar
    read = {True: 0, False: 0}
    by_e: Dict[str, List[float]] = {}
    floors: Dict[str, Dict[str, Any]] = {}
    for r in RESULTS:
        if r["kind"] != "paired":
            continue
        for e in r["items"]:
            g = e.get("gate") or {}
            means = [(g.get(c) or {}).get("mean") for c in ("small", "large")]
            over = all(m is not None and m >= lv.ITEM_GATE_BAR for m in means)
            for cls, block in (e.get("classes") or {}).items():
                for a in (block or {}).get("arms", []):
                    if a["what"] != "null" or a["metrics"] is None:
                        continue
                    read[over] += 1
                    if "kl" in a["metrics"] and a["metrics"]["kl"]["mean"] is not None and not a.get("same_as_base"):
                        by_e.setdefault(a["key"], []).append(abs(a["metrics"]["kl"]["mean"]))
                    if any(st["above_1"] for st in a["metrics"].values()):
                        above[over].append((r["name"], e["item"]["name"], cls, a["key"]))
                if block and r["name"].startswith("verify: 9, "):
                    layer = int(r["name"].split("at layer ")[1].split()[0])
                    floors.setdefault(e["item"]["name"], {}).setdefault(str(layer), {})[cls] = (block.get("floor") or {}).get("kl", {}).get("mean_abs")
    # 6.2 §11's line, as two: a null row that parts on an item the gate would have refused fails the second line alone,
    # and that is the gate doing what it is there for, not the null rows failing where a reading is taken
    check(read[True] > 0 and not above[True], f"no null row of {read[True]} read on items over the shipped bar ({lv.ITEM_GATE_BAR:g}) has a position above a nat"
          + (f" ({len(above[True])} have, the first {above[True][0]})" if above[True] else ""))
    check(not above[False], f"no null row of {read[False]} read on items under the shipped bar, which the gate would not have read, has a position above a nat"
          + (f" ({len(above[False])} have, the first {above[False][0]})" if above[False] else ""))
    # the null rows by e: the floor is the largest of them, and whether a null row reads more as its factor leaves 1 is unmeasured
    record("null_rows_by_e", {k: {"n": len(v), "median": sorted(v)[(len(v) - 1) // 2], "most": max(v)} for k, v in sorted(by_e.items())})
    # the floors at layer 44, which §9 did not read: one more reading an item, its hardest-firing neuron there silenced
    for name, item in S["items"].items():
        r = take(f"11, hardest 1 at layer 44 silenced, {name}", silence(44, hardest(item, 44, 1)), [item], controls={"count": 0, "seed": 0}, bars={"item_gate": -30.0})
        for e in r["items"]:
            for cls, block in (e.get("classes") or {}).items():
                if block:
                    floors.setdefault(name, {}).setdefault("44", {})[cls] = (block.get("floor") or {}).get("kl", {}).get("mean_abs")
    record("floors_by_item", floors)
    everything = [v for per in floors.values() for by_cls in per.values() for v in by_cls.values() if v is not None]
    record("floors", {"n": len(everything), "least": min(everything, default=None), "most": max(everything, default=None)})


def section_12() -> None:
    h = S["h"]
    ok = True
    try:
        we.assert_base(h.lm, h.hooks_at_load)
    except we.NotBase as exc:
        ok = False
        record("not_base", str(exc))
    check(ok, "after everything the model is the base: assert_base passes")
    check(fell_back() == 0, f"the compiled unpacker has not fallen back ({fell_back()})")
    v = lb.verify()
    check(v["ok"] is True, f"this check's ledger is whole: {v['entries']} entries" + ("" if v["ok"] else f"; line {v['first_bad']}: {v['why']}"))
    record("ledger", {"path": str(lb.book_path()), **v})
    check(all(r["base_after"] is True for r in RESULTS), "every reading ended with the model the base")
    peaks = []
    for r in RESULTS:
        for e in r["items"]:
            if r["kind"] == "written":
                peaks.append({"kind": "written", "rows": e.get("rows"), "pieces": e["item"]["prompt_pieces"] + e.get("max_new", 0), "peak_gib": r["peak_gib"]})
                continue
            for cls, block in (e.get("classes") or {}).items():
                if block and block.get("peak_gib") is not None:
                    peaks.append({"kind": "paired", "rows": block["rows"], "pieces": e["item"]["pieces"], "forwards": block["forwards"], "peak_gib": block["peak_gib"]})
    S["peaks"] = peaks
    record("peaks", {"n": len(peaks), "largest": max((p["peak_gib"] for p in peaks if p["peak_gib"] is not None), default=None),
                     "by_shape": sorted({(p["kind"], p["rows"], p["pieces"], round(p["peak_gib"], 3)) for p in peaks if p["peak_gib"] is not None})[:400]})


# ------------------------------------------------------------------------------------------------- the proposed bars


def refit_cost() -> Dict[str, Any]:
    """A least-squares refit of the cost model's three time figures from the paired readings' wall seconds, and of the
    memory a row-position from the peaks. A reading's seconds are modelled as a x (forwards, a restart counted as its
    share of the layers) + b x (their row-positions, likewise) + c x (scored row-positions read out)."""
    n_layers, rows = S["h"].n_layers, []
    for r in RESULTS:
        if r["kind"] != "paired" or r["status"] != "ok" or twin_columns(r) > 0 or not r["seconds"].get("actual"):
            continue
        plan = r["plan"]
        n0, nd, C = len(plan["null_e"]), len(plan["doses"]), plan["controls"]["count"]
        on = plan["readouts"]["kl"] or plan["readouts"]["own"]
        fwd = rp = ro = 0.0
        for e in r["items"]:
            T, scored = e["item"]["pieces"], e["item"]["scored"]
            for block in (e.get("classes") or {}).values():
                if not block:
                    continue
                R, n_f = block["rows"], block["forwards"]
                share = 1.0 if block["restart_from"] is None else (n_layers - block["restart_from"]) / n_layers
                first = share if block["lower"] == "one_row" else 1.0
                fwd += first + (n_f - 1) * share
                rp += R * T * (first + (n_f - 1) * share)
                ro += scored * ((n_f + n0 + nd + C) if on else 1)
            if any((b or {}).get("lower") == "one_row" for b in (e.get("classes") or {}).values()):
                fwd, rp = fwd + 1.0, rp + T
        rows.append((fwd, rp, ro, r["seconds"]["actual"]))
    out: Dict[str, Any] = {"forward_fixed_s": None, "row_position_s": None, "readout_row_position_s": None, "mib_row_position": None, "readings": len(rows)}
    if len(rows) >= 6:
        X, y = np.array([x[:3] for x in rows], dtype=np.float64), np.array([x[3] for x in rows], dtype=np.float64)
        sol = np.linalg.lstsq(X, y, rcond=None)[0]
        out.update({"forward_fixed_s": float(sol[0]), "row_position_s": float(sol[1]), "readout_row_position_s": float(sol[2])})
    peaks = [(p["rows"] * p["pieces"], p["peak_gib"]) for p in S.get("peaks", []) if p["kind"] == "paired" and p["peak_gib"] is not None]
    if len(peaks) >= 4 and len({x for x, _y in peaks}) >= 2:
        X = np.array([[x, 1.0] for x, _y in peaks], dtype=np.float64)
        sol = np.linalg.lstsq(X, np.array([y_ for _x, y_ in peaks], dtype=np.float64), rcond=None)[0]
        out.update({"mib_row_position": float(sol[0] * 1024), "peak_intercept_gib": float(sol[1]), "allocated_gib_at_rest": S.get("rest_gib")})
    return out


def proposed_bars() -> Dict[str, Any]:
    by_item, shipped = S.get("by_item", {}), lv.ITEM_GATE_BAR
    rows = []
    for name, rec in by_item.items():
        g = rec.get("gate") or {}
        rows.append({"name": name, "kind": rec["kind"], "small": g.get("small"), "large": g.get("large"), "misbehaved": bool(rec["unstable"]),
                     "base_rows_same_bits": rec.get("base_rows_same_bits")})
    figure = {x["name"]: min(x["small"], x["large"]) for x in rows if x["small"] is not None and x["large"] is not None}
    behaved = [figure[x["name"]] for x in rows if x["name"] in figure and not x["misbehaved"]]
    misbehaved = [figure[x["name"]] for x in rows if x["name"] in figure and x["misbehaved"]]
    lowest, highest = min(behaved, default=None), max(misbehaved, default=None)
    separates = highest is None or lowest is None or highest < lowest
    # `separates` is the spec's (true when every item that misbehaved sits under every item that did not), so it is also
    # true when NOTHING misbehaved or nothing behaved: `says` and the two counts stand beside it so that it is not read
    # as "the gate's figure was seen to separate"
    says = ("nothing misbehaved: the figure was not put to any test" if not misbehaved else "nothing behaved" if not behaved
            else "every item that misbehaved sits under every item that did not" if separates else "the base's log-probability does not separate")
    gate = {"items": rows, "lowest_that_behaved": lowest, "highest_that_misbehaved": highest, "separates": separates, "says": says,
            "behaved": len(behaved), "misbehaved": len(misbehaved),
            "proposed": (lowest + highest) / 2 if separates and highest is not None and lowest is not None else None}
    above = [(x, m) for n, rec in by_item.items() if figure.get(n, -math.inf) >= shipped for x in rec["readings"] for m in ("kl", "own") if x["ratios"][m] is not None]
    ratios = [x["ratios"][m] for x, m in above]
    agreed = [x["ratios"][m] for x, m in above if not any(d[0] == m for d in (x["disagree"] or []))]
    largest, largest_agreed = max(ratios, default=None), max(agreed, default=None)
    parted = [x["floor_ratio"] for rec in by_item.values() for x in rec["readings"] if x.get("floor_ratio") is not None]
    stable = {"largest_above_the_bar": largest, "share_over_2": (sum(1 for x in ratios if x > 2) / len(ratios)) if ratios else None,
              "control_arms_disagreeing": S.get("control_arms"), "proposed": None if largest is None else max(2.0, 1.5 * largest),
              # the spec's `proposed` takes the largest ratio seen, the readings that disagreed among them: one ratio of 80
              # would propose 120, a bar under which whatever was seen agrees. Beside it: the same from the readings that agreed
              "readings_that_disagreed_above_the_bar": len(ratios) - len(agreed), "largest_among_those_that_agreed": largest_agreed,
              "proposed_from_those_that_agreed": None if largest_agreed is None else max(2.0, 1.5 * largest_agreed),
              # the yardstick itself: how often an item's two floors (the largest null row at each row count) stand over the bar apart
              "floors_apart_over_the_bar": [sum(1 for x in parted if x > RATIO_BAR), len(parted)], "largest_ratio_of_two_floors": max(parted, default=None)}
    the_map = S.get("map", {})
    rows_block = {"small": lv.ROWS_SMALL, "large": lv.ROWS_LARGE, "share_bits": {str(v["pieces"]): v["share_bits"] for v in the_map.values()},
                  "why": "not changed by this script: the two defaults stand until the map above (which row counts gave the same base row, at three lengths) is read by a person"}
    one_row = {}
    for layer, per in (S.get("one_row") or {}).items():
        gaps = {c: [x[c]["gap"] for x in per if x[c]["gap"] is not None] for c in ("small", "large")}
        floors = [x[c]["floor"] for x in per for c in ("small", "large") if x[c]["floor"] is not None]
        one_row[layer] = {"gap_small": max(gaps["small"], default=None), "gap_large": max(gaps["large"], default=None), "floor": max(floors, default=None)}
    return {"item_gate": gate, "stable_ratio": stable, "rows": rows_block, "one_row": one_row, "cost": refit_cost()}


# -------------------------------------------------------------------------------------------------------------- main

SECTIONS: List[Tuple[str, str, Callable[[], None]]] = [
    ("1", "the model", section_1), ("2", "the prompts", section_2), ("3", "a zero-edit row by bits", section_3),
    ("4", "a row's place does not matter", section_4), ("5", "input-side against folded", section_5),
    ("6", "a same-shape restart, and the one-row capture", section_6), ("7", "writing under hooks", section_7),
    ("8", "the item gate at the shipped bar", section_8), ("9", "the two row counts on every item", section_9),
    ("10", "the cost model", section_10), ("11", "null rows and floors", section_11), ("12", "after everything", section_12),
]


def main() -> int:
    # as scripts/levelling.py asks it: runtime_factory reads the device lower-cased, and "cuda" as cuda:0
    if os.environ.get("THEODOLITE_GGUF", "").strip() or os.environ.get("THEODOLITE_DEVICE", "").strip().lower() not in ("", "cuda", "cuda:0"):
        print("1a reads the W4A16 checkpoint on the card, or the stand-in with --stand-in.")
        return 2
    if server_running():
        print("Something answers on :8765: stop the Theodolite server first (this loads the model itself).")
        return 2
    if not torch.cuda.is_available():
        print("A CUDA GPU is needed: this is the card's check. The CPU's is scripts/check_levelling.py.")
        return 2
    free, _total = torch.cuda.mem_get_info()
    if free / 2**30 < MIN_FREE_GIB:
        print(f"Only {free / 2**30:.1f} GiB of the card is free ({MIN_FREE_GIB:g} are asked: the load peaks at 21.2 with the lens, before it is freed): "
              "stop the :8765 server and any census first.")
        return 2
    try:
        lb.head()
    except lb.BookBroken as exc:
        print(str(exc))
        return 2
    from backend.runtime_factory import make_runtime

    t0 = time.perf_counter()
    torch.set_grad_enabled(False)  # every forward of 1a runs under no_grad: the compiled unpacker serves only that mode
    print(f"this check's ledger: {BOOK}\nits JSON: {OUT_FILE}", flush=True)
    rt = make_runtime(load_jlens=False).load()
    rt.jpp_lenses.clear()  # make_runtime(load_jlens=False) still loads the lens (scripts/census_run.py's free_lens)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        unpacker = w4a16_fast.install(rt, "compiled", log=print)
    except w4a16_fast.Refused as exc:
        print(str(exc))
        return 2
    return run_checks(rt, unpacker, t0)


def run_checks(rt: Any, unpacker: Dict[str, Any], t0: float) -> int:
    """Everything after the load: the items, the twelve sections, the JSON after each."""
    from backend.runtime_factory import weights_card

    h = lv.handle(rt)
    S.update({"rt": rt, "h": h, "lm": h.lm, "tok": h.tokenizer, "W_U": h.W_U, "dev": h.device, "files": Path(tempfile.mkdtemp(prefix="verify-levelling-"))})
    OUT.update({"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": rt.model_id, "weights": weights_card(rt), "runtime": h.runtime,
                "unpacker": unpacker, "book": str(lb.book_path()), "book_entries_before": lb.head()[0], "load_seconds": round(time.perf_counter() - t0, 1),
                "code": {name: cz.file_sha256(ROOT_DIR / "backend" / f"{name}.py") for name in ("weight_edits", "levelling", "level_book")}})
    NOW["section"] = "items"
    print("the items", flush=True)
    build_items()
    save()
    stopped: Optional[str] = None
    for label, title, fn in SECTIONS:
        NOW["section"] = label
        print(f"§{label}. {title}", flush=True)
        t1 = time.perf_counter()
        try:
            fn()
        except KeyboardInterrupt:
            check(False, "interrupted")
            stopped = f"interrupted in §{label}"
        except Exception as exc:  # noqa: BLE001 - one section's slip is one failing line; the model is looked at before the next
            where = traceback.extract_tb(exc.__traceback__)[-1]
            check(False, f"the section stopped at {type(exc).__name__}: {str(exc)[:400]} ({Path(where.filename).name}:{where.lineno})")
            torch.cuda.empty_cache()
        sec()["seconds"] = round(time.perf_counter() - t1, 1)
        if stopped is None:
            try:
                we.assert_base(h.lm, h.hooks_at_load)
            except we.NotBase as exc:
                check(False, f"the model is not the base after this section ({exc}): nothing more is read in this process")
                stopped = f"not the base after §{label}"
        if label in ("9", "12") or stopped:
            NOW["section"] = "proposed bars"
            try:
                OUT["proposed_bars"] = proposed_bars()
            except Exception as exc:  # noqa: BLE001
                OUT["proposed_bars_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        save()
        if stopped:
            break
    OUT["seconds"] = round(time.perf_counter() - t0, 1)
    OUT["stopped"] = stopped
    OUT["book_entries_after"] = lb.verify()["entries"]
    save()
    shutil.rmtree(S["files"], ignore_errors=True)  # the recipes' seeded arrays: scratch, outside the checkout
    print(f"\nwritten: {OUT_FILE}")
    if failures:
        print(f"verify_levelling: {len(failures)} of {passed + len(failures)} checks FAILED ({OUT['seconds']:.0f} s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"verify_levelling: all {passed} checks hold ({OUT['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
