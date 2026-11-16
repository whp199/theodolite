#!/usr/bin/env python3
"""The command line of the weight-edit kernel: what a plan would cost, the plan run, and the book read.

    .venv/bin/python -I scripts/levelling.py estimate plan.json            # no model: forwards, seconds and GiB
    .venv/bin/python -I scripts/levelling.py run plan.json                 # the 31B on the card, the server stopped
    .venv/bin/python -I scripts/levelling.py run plan.json --only "silence 4294 and 9618" --out results.json
    .venv/bin/python -I scripts/levelling.py run plan.json --stand-in      # a dry run on the tiny random stand-in
    .venv/bin/python -I scripts/levelling.py book                          # the last 20 lines of the book
    .venv/bin/python -I scripts/levelling.py book --seq 41                 # line 41, whole
    .venv/bin/python -I scripts/levelling.py book --verify                 # walk the chain

What it is. A reading takes one edit of the base weights as a value, reads it against the unedited model in another
row of the same forward, and enters what it read in the book (backend/levelling.py, backend/weight_edits.py,
backend/level_book.py). This script is the way in for a person at a terminal: it reads a plan file, loads the model
once, and takes the plan's readings in order. It prints numbers, ranks against control edits, floors, and the four
status words "ok", "refused", "void" and "unstable". It claims nothing about an edit and never leaves the model
edited: an edit is an argument of a reading, and the base is asserted after the last one. Everything a reading
prints is written by `levelling.lines`; this script adds the estimate before a reading and the book's line after.

A plan file (saved anywhere; file paths in it are taken from its own folder; layers count from 0, so layer 20 is
page layer 21):

    {"v": 1, "name": "two layer-20 neurons silenced, on two questions",
     "items": {"moon": {"question": "In two or three sentences, why does the moon show phases?",
                        "answer": "The moon shows phases because we see different portions of its sunlit half as it orbits the Earth."},
               "stock": {"question": "Explain briefly why a stock should not be salted while it simmers.",
                         "answer": "You should not salt a stock while it simmers because of evaporation."}},
     "readings": [{"name": "silence 4294 and 9618", "kind": "paired",
                   "edit": {"terms": [{"layer": 20, "silence": [4294, 9618]}]},
                   "doses": [1.0, -1.0], "controls": {"count": 19, "seed": 0}},
                  {"name": "the same, written", "kind": "written",
                   "edit": {"terms": [{"layer": 20, "silence": [4294, 9618]}]}, "items": ["moon"]}]}

`estimate PLAN [--stand-in]` loads the tokenizer alone (from this machine's cache; nothing is fetched and the card
is hidden), builds the items, checks every reading's plan and prints each one's forwards, row-positions, seconds and
GiB above the model, then the run's total with the load. The seconds are a cost model, not a measurement of this
plan (`levelling.COST`): 0.08 s a whole forward and 0.33 ms a row-position (within 15% on nine shapes), 0.11 ms a
scored row-position of full-vocabulary read-out, 0.095 s a written step, 9.5 s to load the model and 1.83 s to
install the unpacker, all measured on the RTX 5090 on 14 November 2026, one run each. The GiB of a paired reading is
inferred, not measured; a written reading's comes from 0.8 MB a row a word-piece, measured 28 October 2026 with the
lens loaded. The estimate is made without the model, so it prices a folded twin wherever the edit's arithmetic asks
for one (on the stand-in none: its linears are plain); the reading itself asks the loaded model whether it can fold
and records its own prediction beside its clock.

`run PLAN [--stand-in] [--only NAME ...] [--out FILE] [--save-edits] [--port 8765]`, in this order:
1. refused when THEODOLITE_GGUF is set or THEODOLITE_DEVICE names anything but cuda:0 (1a reads the W4A16 checkpoint
   on the card, or the stand-in);
2. refused while something answers on the port (the Theodolite server has the model on the card; this loads it too);
3. the plan is read and --only's names looked for; then refused without CUDA, and refused when under 22 GiB of the
   card is free. That bar is provisional and inferred: the model and its lens took 21.2 GiB on 28 October 2026, and
   the lens is loaded before this script frees it;
4. the book must be able to take a line; the model is loaded from this machine's cache (HF_HUB_OFFLINE is set to 1
   unless the environment says otherwise, as the server and the pilots ran), its lens freed (a reading uses none of
   it), and the compiled unpacker installed with its self-test; the hooks the model carries at rest are counted
   and the items built;
5. the readings, each estimated, read, entered in the book and printed. After a reading that leaves the model
   other than the base, or out of which an exception came, nothing more is read in the process;
6. the base is asserted once more and the book's head printed.
--out writes the results so far as one JSON list after each reading, so a run that stops leaves what it read; it
replaces the file it names, and is refused when that is the book or the plan file.
--save-edits saves each reading's edit beside the book (`$THEODOLITE_EDITS_DIR`, else
`${XDG_DATA_HOME:-~/.local/share}/theodolite/edits`); a saved edit can carry pieces of Gemma's columns and is never
put in a checkout.

With --stand-in the model is scripts/atlas_stand_in.py's six-layer random Gemma 4 in float32 on the CPU: the card is
hidden and nothing is fetched, one thread is set, steps 1 and 2 and step 3's two questions of the card are left
out (no port probe, no memory question), there is no unpacker, and, unless THEODOLITE_LEVEL_BOOK and
THEODOLITE_EDITS_DIR are already set, a fresh temporary folder is made, named in the first line printed, and whichever
is unset points into it, so a dry run never writes the DEFAULT book or the default folder of saved edits. A book or
a folder the variables already name IS written by a dry run, and the first line says which. What a stand-in run shows is that a plan is well formed and the plumbing runs; its
figures mean nothing. Run on the CPU on 14 November 2026: the stand-in is random over about 600 word-pieces, so a
typed answer reads about -6.4 nats a piece (-6.43 and -6.40 on this script's two trial items) and the shipped item
gate (-6.0) refuses nearly every item: a stand-in plan gives "bars": {"item_gate": -8.0} in each paired reading.
Its linears are plain, so no folded twin is taken. Above one thread the rows of one written batch differ in their
last digits on the float32 stand-in (measured on this machine, MKL 2024.2), which a written reading would void by
its own rule, hence the one thread.

`book [--tail N] [--seq K] [--verify]` loads no model. --tail (20 when nothing else is asked): one line an entry,
with its line number, time, kind, status, how many of its items a paired reading kept of those asked (a reading is
"ok" when its mechanism was sound, whatever its items came to), the edit's first 12, its name and its seconds. --seq K: that entry's JSON,
indented. --verify: every line parses, is in its place, carries the hash of the line before and its own.

Exit codes. 0: `estimate` done with every reading priced; `run` with every reading "ok" (items refused by the item
gate or read "unstable" are results of an "ok" reading, not failures); `book` done, `--verify` with the chain whole.
1: a reading void or refused (in `estimate`, one whose plan is refused), a run stopped at step 5 or whose model is
not the base at the end; `book --verify` with a bad line, which it names with the reason. 2: nothing could be begun:
the port answers, no CUDA, too little of the card free, a plan that cannot be read or is refused at its top level,
THEODOLITE_GGUF set, the unpacker refused, the book broken, a line asked for that the book has not.

What was run, and what was not. On the CPU on 14 November 2026, with the card hidden: `estimate --stand-in` and
`run --stand-in` on a plan of one paired and one written reading (both "ok", entered as lines 0 and 1 of a book in
a temporary folder, which `book --verify` then walked whole), and on a plan with three refused readings (each
entered, exit 1; --out and --save-edits writing beside that book); `book` on whole and on broken books; and every
refusal `run` makes before a model is loaded, the port's among them. `run` on the card has NOT been run: nothing
here has loaded the 31B, asked the card its free memory or installed the unpacker. Its first run is the owner's, or
the card check's (scripts/verify_levelling.py). The names "levelling" and "level book" are provisional, the owner's
to choose.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend import level_book as lb  # noqa: E402  (standard library only: no torch is imported by this line)

PORT = 8765
MIN_FREE_GIB = 22.0  # provisional, inferred: the model and its lens took 21.2 GiB (28 October 2026), the lens freed after
TAIL = 20
EDITS_ENV = "THEODOLITE_EDITS_DIR"  # weight_edits.EDITS_ENV, named here so it can be set before torch is imported
GGUF_ENV, DEVICE_ENV = "THEODOLITE_GGUF", "THEODOLITE_DEVICE"  # backend/runtime_factory.py's
ONLY_THE_CARD = "1a reads the W4A16 checkpoint on the card, or the stand-in with --stand-in."


def say(text: str = "") -> None:
    print(text, flush=True)


def entries_of(n: int) -> str:
    return f"{n} entr{'y' if n == 1 else 'ies'}"


def server_answers(port: int) -> bool:
    """Whether something takes a TCP connection on 127.0.0.1 at `port` within half a second
    (scripts/verify_causal_check.py's probe)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def hide_the_card() -> None:
    """Before torch is imported: no GPU for this process, and nothing fetched."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"


def stand_in_places() -> str:
    """Where a stand-in run keeps its book and its saved edits, as its first printed line. Unless both variables are
    set already, a fresh temporary folder is made and whichever is unset is pointed into it, so a dry run never
    writes the default book or the default folder of saved edits. One the environment already names is written, and
    the line says so."""
    book, edits = os.environ.get(lb.BOOK_ENV), os.environ.get(EDITS_ENV)
    if book and edits:
        return f"stand-in run: the book is {book} and saved edits go to {edits}, as the environment says"
    folder = Path(tempfile.mkdtemp(prefix="levelling-stand-in-"))
    if not book:
        os.environ[lb.BOOK_ENV] = str(folder / "level_book.jsonl")
    if not edits:
        os.environ[EDITS_ENV] = str(folder / "edits")
    what = ("its book and its saved edits go" if not book and not edits else
            f"the book is {book}, as the environment says; its saved edits go" if book else
            f"saved edits go to {edits}, as the environment says; its book goes")
    return f"stand-in run: {what} to a temporary folder of its own, {folder}"  # the folder last, so it is easily read off


def read_plan(lv: Any, path: Path) -> Optional[Dict[str, Any]]:
    """The plan file with its top level checked (`levelling.load_plan`), or None with the reason printed."""
    try:
        return lv.load_plan(path)
    except lv.Refused as exc:
        say(str(exc))
    except OSError as exc:
        say(f"The plan file {path} cannot be read: {exc.strerror or exc}.")
    except ValueError as exc:  # a file that is not JSON, should the parser's own error come through
        say(f"The plan file {path} is not a plan: {exc}.")
    return None


def named(readings: Sequence[Mapping[str, Any]], only: Optional[Sequence[str]]) -> Optional[List[Mapping[str, Any]]]:
    """The plan's readings, or those --only names, in the plan's order; None with the reason printed when a name is
    no reading of the plan."""
    if not only:
        return list(readings)
    have = [r.get("name") for r in readings]
    missing = [n for n in only if n not in have]
    if missing:
        say(f"This plan has no reading named {missing[0]!r}. Its readings are: " + ", ".join(repr(n) for n in have) + ".")
        return None
    return [r for r in readings if r.get("name") in only]


def price(lv: Any, we: Any, reading: Mapping[str, Any], items: Mapping[str, Any], *, n_layers: int, tokenizer: Any,
          dtype: Any, base_dir: Path, stand_in: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """(the normalised plan, its estimate, how many items it reads), with no model: the recipe's facts, the plan
    normalised, the cost model. Raises the kernel's refusals as they are. On the stand-in no folded twin is priced:
    its linears are plain, so a reading there takes none."""
    recipe = reading.get("edit") if isinstance(reading, Mapping) else None
    facts = we.recipe_facts(recipe, base_dir) if isinstance(recipe, dict) else None
    plan = lv.normalise(reading, n_layers=n_layers, tokenizer=tokenizer, facts=facts, dtype=dtype)
    names = list(items) if plan["items"] == "all" else list(plan["items"])
    pieces = [(len(items[n].ids), len(items[n].prompt_ids)) for n in names]
    est = lv.estimate(plan, pieces, n_layers=n_layers, facts=facts, twin=False if stand_in else None)
    return plan, est, len(names)


def estimate_line(plan: Mapping[str, Any], est: Mapping[str, Any], n_items: int) -> str:
    parts = est.get("parts") or {}
    cost = ", ".join(f"{label} {float(parts.get(key) or 0.0):.1f}" for key, label in
                     (("forwards", "forwards"), ("readout", "read-out"), ("folded", "folded twin"), ("writing", "writing")))
    return (f'estimate for "{plan.get("name")}" ({plan.get("kind")}, {n_items} item{"" if n_items == 1 else "s"}): '
            f'{est["forwards"]} forward{"" if est["forwards"] == 1 else "s"}, {est["row_positions"]:,} row-positions, '
            f'about {est["seconds"]:.1f} s ({cost}), about {est["gib_above_model"]:.1f} GiB above the model'
            + (" (inferred, not measured)" if plan.get("kind") == "paired" else " (from 0.8 MB a row a piece, measured)"))


# ------------------------------------------------------------------------------------------------------- estimate

def cmd_estimate(a: argparse.Namespace) -> int:
    hide_the_card()  # no model is loaded and the card is not asked anything
    import torch

    import backend.levelling as lv
    import backend.weight_edits as we

    path = Path(a.plan)
    plan = read_plan(lv, path)
    if plan is None:
        return 2
    if a.stand_in:
        from atlas_stand_in import N_LAYERS, stand_in_tokenizer

        tokenizer, n_layers, dtype = stand_in_tokenizer(), N_LAYERS, torch.float32
    else:
        from transformers import AutoTokenizer

        from backend.engine_runtime import DEFAULT_W4A16_MODEL_ID
        from backend.theodolite_math import N_LAYERS

        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_W4A16_MODEL_ID, local_files_only=True)
        except OSError as exc:
            say(f"The tokenizer of {DEFAULT_W4A16_MODEL_ID} is not in this machine's cache, and an item is counted in "
                f"its word-pieces ({str(exc).splitlines()[0]}). Nothing is fetched here; try --stand-in for a plan "
                "written for the stand-in.")
            return 2
        n_layers, dtype = N_LAYERS, torch.bfloat16
    try:
        items = lv.plan_items(plan, tokenizer)
    except lv.Refused as exc:
        say(str(exc))
        return 2
    say(f'plan "{plan.get("name")}": {len(items)} item{"" if len(items) == 1 else "s"}, {len(plan["readings"])} '
        f'reading{"" if len(plan["readings"]) == 1 else "s"}; {n_layers} layers, counted from 0'
        + (" (the stand-in)" if a.stand_in else ""))
    total, refused = 0.0, 0
    for reading in plan["readings"]:
        try:
            normal, est, n_items = price(lv, we, reading, items, n_layers=n_layers, tokenizer=tokenizer, dtype=dtype,
                                         base_dir=path.parent, stand_in=a.stand_in)
        except (lv.Refused, we.Refused) as exc:
            refused += 1
            say(f'reading "{reading.get("name")}" is refused: {exc}')
            continue
        except OSError as exc:
            refused += 1
            say(f'reading "{reading.get("name")}" is refused: a file its edit names cannot be read ({exc.strerror or exc}).')
            continue
        except Exception as exc:  # noqa: BLE001  (a slip in a plan that the kernel does not turn into a refusal)
            refused += 1
            say(f'reading "{reading.get("name") if isinstance(reading, dict) else reading!r}" could not be priced: '
                f"{type(exc).__name__}: {str(exc)[:300]}")
            continue
        total += float(est["seconds"])
        say(estimate_line(normal, est, n_items))
    load = float(lv.COST["load_s"]) + float(lv.COST["unpacker_s"])
    say(f"the run: about {total:.1f} s of readings and {load:.1f} s to load the model and install the unpacker "
        f"({lv.COST['load_s']:g} + {lv.COST['unpacker_s']:g}, measured 14 November 2026), about {total + load:.0f} s in all"
        + (f"; {refused} reading{'' if refused == 1 else 's'} refused and not priced" if refused else ""))
    if a.stand_in:
        say("these seconds and GiB are the 31B's cost model on the stand-in's shapes: they do not say how long the "
            "stand-in takes")
    return 1 if refused else 0


# ------------------------------------------------------------------------------------------------------------ run

def cmd_run(a: argparse.Namespace) -> int:
    if a.stand_in:
        hide_the_card()
        say(stand_in_places())
    else:
        device = os.environ.get(DEVICE_ENV, "").strip().lower()
        if os.environ.get(GGUF_ENV, "").strip():
            say(f"{ONLY_THE_CARD} {GGUF_ENV} is set.")
            return 2
        if device not in ("", "cuda", "cuda:0"):
            say(f"{ONLY_THE_CARD} {DEVICE_ENV} is {device}.")
            return 2
        if server_answers(a.port):
            say(f"Something answers on :{a.port}: stop the Theodolite server first (this loads the model itself).")
            return 2
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # as the server and the pilots ran: the load reads the cache alone

    import torch

    import backend.levelling as lv
    import backend.weight_edits as we
    from backend.census import write_json_atomic

    if a.stand_in:
        torch.set_num_threads(1)  # above one thread the float32 stand-in's written rows differ in their last digits
        if we.EDITS_ENV != EDITS_ENV:  # the name this script set before torch was imported is not the module's
            os.environ.setdefault(we.EDITS_ENV, os.environ[EDITS_ENV])

    path = Path(a.plan)
    plan = read_plan(lv, path)
    if plan is None:
        return 2
    readings = named(plan["readings"], a.only)
    if readings is None:
        return 2
    if a.out:  # --out replaces the file it names: never the book, which is only ever added to, nor the plan
        out_path = Path(a.out).expanduser().resolve()
        for what, other in (("the book", lb.book_path()), ("the plan file", path)):
            if out_path == Path(other).expanduser().resolve():
                say(f"--out names {what} ({other}): it would be replaced by the results. Name another file.")
                return 2

    if not a.stand_in:
        if not torch.cuda.is_available():
            say(f"No CUDA device answers in this process. {ONLY_THE_CARD}")
            return 2
        free, _total = torch.cuda.mem_get_info()
        if free < MIN_FREE_GIB * 2**30:
            say(f"Only {free / 2**30:.1f} GiB of the card is free ({MIN_FREE_GIB:g} are asked: the load peaks at 21.2 "
                "with the lens, before it is freed): stop the :8765 server and any census first.")
            return 2
    try:
        lb.head()
    except lb.BookBroken as exc:
        say(str(exc))
        return 2

    t0 = time.perf_counter()
    if a.stand_in:
        from atlas_stand_in import stand_in, use_stand_in_atlas

        use_stand_in_atlas()
        rt, _vault = stand_in()
    else:
        from backend import w4a16_fast
        from backend.runtime_factory import make_runtime

        rt = make_runtime(load_jlens=False).load()
        if getattr(rt, "jpp_lenses", None):  # make_runtime(load_jlens=False) still loads the J++ lens; a reading uses none
            rt.jpp_lenses.clear()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            w4a16_fast.install(rt, "compiled", log=say)
        except w4a16_fast.Refused as exc:
            say(str(exc))
            return 2
    try:
        h = lv.handle(rt)
    except we.NotBase as exc:
        say(str(exc))
        return 2
    at_load = h.hooks_at_load or {}
    say(f"{h.model_id}: {h.n_layers} layers, counted from 0; ready in {time.perf_counter() - t0:.1f} s"
        + ("" if a.stand_in else f" (the cost model says {lv.COST['load_s'] + lv.COST['unpacker_s']:.1f})")
        + f"; {sum(len(ids) for ids in at_load.values())} hooks on it at load"
        + (f", on {', '.join(sorted(at_load)[:3])}" if at_load else ""))
    try:
        items = lv.plan_items(plan, h.tokenizer)
    except lv.Refused as exc:
        say(str(exc))
        return 2
    if a.stand_in:
        say("estimates below are the 31B's cost model on the stand-in's shapes: they do not say how long the stand-in "
            "takes")

    results: List[Dict[str, Any]] = []
    not_ok, stopped = 0, False
    for reading in readings:
        name = reading.get("name")
        say()
        try:
            normal, est, n_items = price(lv, we, reading, items, n_layers=h.n_layers, tokenizer=h.tokenizer,
                                         dtype=h.dtype, base_dir=path.parent, stand_in=a.stand_in)
            say(estimate_line(normal, est, n_items))
        except Exception:  # noqa: BLE001
            # No estimate for a plan the kernel refuses, or one that holds a slip the pricing stops on: the estimate
            # is advisory. The reading below says what is wrong in its own words and is entered either way.
            pass
        try:
            result = lv.read(h, reading, items, base_dir=path.parent, save_edit=a.save_edits)
        except KeyboardInterrupt:
            say(f'Stopped from the keyboard during "{name}", which the book has as void. Nothing more is read in this '
                "process.")
            stopped = True
            break
        except Exception as exc:  # noqa: BLE001  (entered in the book by `read` and raised again; said here, then stop)
            say(f'{type(exc).__name__} out of "{name}": {exc}')
            say("Nothing more is read in this process.")
            stopped = True
            break
        results.append(result)
        if a.out:
            write_json_atomic(Path(a.out), results)
        printed = list(lv.lines(result))
        for line in printed:
            say(line)
        book = result.get("book") or {}
        if not any("book line" in line for line in printed):
            say(f"  book line {book.get('seq')}")
        if result.get("status") != "ok":
            not_ok += 1
        if not result.get("base_after", False):
            say(f'The model is not the base after "{name}": nothing more is read in this process.')
            stopped = True
            break
    if a.out:
        write_json_atomic(Path(a.out), results)

    say()
    try:
        we.assert_base(h.lm, h.hooks_at_load)
        say("the model is the base: no overlay in force, and the hooks it carried at load and no others")
    except we.NotBase as exc:
        say(str(exc))
        stopped = True
    try:
        count, tip = lb.head()
        say(f"the book: {entries_of(count)}, head {tip[:12]} ({lb.book_path()})")
    except lb.BookBroken as exc:
        say(str(exc))
        stopped = True
    if a.out:
        say(f"{len(results)} result{'' if len(results) == 1 else 's'} written to {a.out}")
    return 1 if (not_ok or stopped) else 0


# ----------------------------------------------------------------------------------------------------------- book

def tail_line(entry: Mapping[str, Any]) -> str:
    body = entry.get("body") if isinstance(entry.get("body"), dict) else {}
    edit = body.get("edit") if isinstance(body.get("edit"), dict) else {}
    seconds = body.get("seconds") if isinstance(body.get("seconds"), dict) else {}
    actual = seconds.get("actual")
    took = f"{actual:.1f} s" if isinstance(actual, (int, float)) and not isinstance(actual, bool) else "-"
    name = body.get("name")
    counts = (body.get("summary") or {}).get("items") if isinstance(body.get("summary"), dict) else None
    kept = f"{counts.get('kept')}/{counts.get('asked')}" if isinstance(counts, dict) else "-"
    return (f"{str(entry.get('seq')):>6}  {str(entry.get('at')):<20}  {str(entry.get('kind')):<7}  "
            f"{str(body.get('status') or '-'):<8}  {kept:<5}  {str(edit.get('id') or '-')[:12]:<12}  "
            f"{json.dumps(name, ensure_ascii=False) if isinstance(name, str) else '-'}  {took}")


def cmd_book(a: argparse.Namespace) -> int:
    path = lb.book_path()
    tail = a.tail if a.tail is not None else (TAIL if a.seq is None and not a.verify else None)
    code = 0
    if tail is not None or a.seq is not None:
        try:
            count, tip = lb.head()
            if tail is not None and count == 0:
                say(f"The book has no line yet ({path}).")
            elif tail is not None:
                say(f"the book: {entries_of(count)}, head {tip[:12]} ({path})")
                say(f"{'line':>6}  {'at':<20}  {'kind':<7}  {'status':<8}  {'kept':<5}  {'edit':<12}  name  seconds")
                for entry in lb.entries(start=max(0, count - max(0, tail))):
                    say(tail_line(entry))
            if a.seq is not None:
                entry = next(lb.entries(start=a.seq, limit=1), None) if a.seq >= 0 else None
                if entry is None:
                    say(f"The book has no line {a.seq}: "
                        + (f"its lines are 0 to {count - 1}" if count else "it has no line yet") + f" ({path}).")
                    code = 2
                else:
                    say(json.dumps(entry, indent=1, ensure_ascii=False))
        except lb.BookBroken as exc:
            say(str(exc))
            code = 2
    if a.verify:
        v = lb.verify()
        if v["ok"]:
            say(f"the book is whole: {entries_of(v['entries'])}, each carrying the hash of the line before and its own; "
                f"head {v['head'][:12]} ({path})")
        else:
            say(f"The book is not whole at line {v['first_bad']}: {v['why']}. The {v['entries']} "
                f"line{' before it is' if v['entries'] == 1 else 's before it are'} whole (head {v['head'][:12]}). Look at {path}.")
            code = max(code, 1)
    return code


# ----------------------------------------------------------------------------------------------------------- main

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="levelling.py", description="Readings of one weight edit against the unedited model "
                                 "in the same forward: what a plan costs, the plan run, the book read.")
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("estimate", help="no model: each reading's forwards, row-positions, seconds and GiB")
    e.add_argument("plan", metavar="PLAN", help="the plan file (JSON)")
    e.add_argument("--stand-in", action="store_true", help="count with the stand-in's tokenizer and its six layers")
    e.set_defaults(fn=cmd_estimate)

    r = sub.add_parser("run", help="load the model once and take the plan's readings in order")
    r.add_argument("plan", metavar="PLAN", help="the plan file (JSON)")
    r.add_argument("--stand-in", action="store_true", help="a dry run on the tiny random stand-in, on the CPU, with a "
                   "book of its own in a temporary folder unless THEODOLITE_LEVEL_BOOK already names one")
    r.add_argument("--only", metavar="NAME", nargs="+", action="extend", help="take only the readings so named")
    r.add_argument("--out", metavar="FILE", help="write every result as one JSON list")
    r.add_argument("--save-edits", action="store_true", help="save each reading's edit beside the book")
    r.add_argument("--port", type=int, default=PORT, help="the port the Theodolite server would answer on (8765)")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("book", help="no model: the book's last lines, one line whole, or the chain walked")
    b.add_argument("--tail", metavar="N", type=int, default=None, help="the last N entries, one line each (20)")
    b.add_argument("--seq", metavar="K", type=int, default=None, help="line K (counting from 0), as indented JSON")
    b.add_argument("--verify", action="store_true", help="walk every line of the chain")
    b.set_defaults(fn=cmd_book)

    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
