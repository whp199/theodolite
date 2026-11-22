"""The compare tab's offline replay cards, made from the committed level book and never written by hand.

`frontend/src/screens/vault/compareRecorded.ts` holds the cards the Vault's compare tab shows when no server answers:
the two claim verdicts of the 31B card check (`scripts/verify_level_claim.py`). Each card names a line of
`experiments/results/levelling/verify_claim_book.jsonl` by its number, time and hash, so every figure on it has to be
that line's. This script is the only writer of the file:

- a card is the server's own replay card for the line (`backend/level_routes.py` `build_replay_card`, what
  `GET /api/level/book?book=verify_claim` answers with), figures unrounded;
- three things the line itself does not carry are added and nothing else: the linear the edit touches (from the
  claim's first reading in the same book), the reading's seconds and a plain label for the edit (both from
  `verify_claim.json`, the report of the same run, which is refused unless its ledger head is the book's), and the tag
  that says where the card was recorded.

After the cards the file exports the book's last run in words, as the server dates a recorded book
(`level_routes.last_run_words`: the UTC date of its last whole line), and the first 16 characters of its head, which
the tab's offline picker and caption show.

    .venv/bin/python -I scripts/level_recorded_cards.py           # is the file what the book gives? (exit 1 if not)
    .venv/bin/python -I scripts/level_recorded_cards.py --write   # rewrite it from the book

`scripts/check_level_page.py` holds the file to this script's cards and, apart from this script, to the book's own
lines. Run it with --write after `scripts/verify_level_claim.py` has rewritten the book. No model and no card.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import level_book as lb  # noqa: E402

RESULTS = ROOT_DIR / "experiments" / "results" / "levelling"
BOOK = RESULTS / "verify_claim_book.jsonl"
REPORT = RESULTS / "verify_claim.json"
OUT = ROOT_DIR / "frontend" / "src" / "screens" / "vault" / "compareRecorded.ts"

# the report's claim blocks, in the order the page lists them, each with a plain label for its edit (page layers)
CLAIMS = (
    ("claim_1_she10", "E6 layer-51 ‘she’ ten"),
    ("claim_2_she_only7", "E6 layer-51 ‘she’-only seven"),
)
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")

HEADER = """/**
 * The 31B replay cards the Vault's `compare` tab shows when no live server answers, so a person can inspect the card
 * check's two claim verdicts with no model loaded.
 *
 * MADE BY `scripts/level_recorded_cards.py`; do not edit by hand. Each card is the server's own replay card
 * (`backend/level_routes.py` `build_replay_card`) for the line of
 * `experiments/results/levelling/verify_claim_book.jsonl` it names by number, time and hash, figures unrounded, with
 * the edit's linear (from the same book), the reading's seconds and the edit's label (from `verify_claim.json`) and
 * the tag added. After `scripts/verify_level_claim.py` rewrites the book, run the script with --write;
 * `scripts/check_level_page.py` holds this file to the book.
 */
import type { ReplayCard } from "./compare";

export const RECORDED_REPLAY_CARDS: ReplayCard[] = """


class CardsRefused(ValueError):
    """The book and the report cannot give the cards, in plain words."""


def _line(seq: int) -> Dict[str, Any]:
    entry = next(lb.entries(BOOK, start=int(seq), limit=1), None)
    if entry is None:
        raise CardsRefused(f"{BOOK.name} has no line {seq}.")
    return entry


def _day(at: str) -> str:
    """'2026-11-19T06:09:48Z' -> '19 November 2026'."""
    y, m, d = at[:10].split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]} {y}"


def cards() -> List[Dict[str, Any]]:
    """The cards, from the committed book and its report. `CardsRefused` when the book is not whole, the report is
    of another run, or a line is not the claim verdict the report says it is."""
    from backend import level_routes  # the server's own card for a line of a book

    whole = lb.verify(BOOK)
    if not whole["ok"]:
        raise CardsRefused(f"{BOOK.name} is not whole at line {whole['first_bad']}: {whole['why']}")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if (report.get("ledger") or {}).get("head") != whole["head"]:
        raise CardsRefused(f"{REPORT.name} is the report of another run than {BOOK.name}: their ledger heads differ.")
    out: List[Dict[str, Any]] = []
    for key, label in CLAIMS:
        block = report[key]
        entry = _line(block["book"]["verdict_seq"])
        body = entry["body"]
        if entry["kind"] != "note" or body.get("note_type") != "claim_verdict" or body.get("verdict") != block["verdict"]:
            raise CardsRefused(f"Line {entry['seq']} of {BOOK.name} is not the claim verdict {REPORT.name} names "
                               f"under {key}.")
        card = level_routes.build_replay_card(entry)
        linears = None
        for seq in body.get("book_seqs") or []:  # the claim's first reading of this edit names its linears
            edit = _line(seq)["body"].get("edit")
            if isinstance(edit, dict) and edit.get("id") == body.get("edit_id"):
                terms = (edit.get("terms") or {}).get("terms") or []
                linears = sorted({str(t["linear"]) for t in terms if isinstance(t, dict) and t.get("linear")})
                break
        if not linears:
            raise CardsRefused(f"No reading of line {entry['seq']}'s claim in {BOOK.name} names the edit's linears.")
        card["name"] = f"{card['name']} — {label}"
        card["edit_linears"] = linears
        card["seconds"] = block["seconds"]
        card["recorded_tag"] = f"recorded on the 31B card ({_day(entry['at'])}, {BOOK.name})"
        out.append(card)
    return out


def book_words() -> Dict[str, str]:
    """{"last_run": the book's last run in words, as the server dates a recorded book, "head": the first 16
    characters of its head}. `CardsRefused` when the book is not whole or names no date."""
    from backend import level_routes

    whole = lb.verify(BOOK)
    if not whole["ok"]:
        raise CardsRefused(f"{BOOK.name} is not whole at line {whole['first_bad']}: {whole['why']}")
    when = level_routes.last_run_words(BOOK, int(whole["entries"]))
    if not when:
        raise CardsRefused(f"The last line of {BOOK.name} names no time a date can be read from.")
    return {"last_run": when, "head": str(whole["head"])[:16]}


def render() -> str:
    """The whole text of `compareRecorded.ts`."""
    words = book_words()
    return (HEADER + json.dumps(cards(), indent=2, ensure_ascii=False, allow_nan=False) + ";\n\n"
            + "/** The book the cards are from: its last run, as the server dates a recorded book, and its head. */\n"
            + f"export const RECORDED_BOOK_LAST_RUN = {json.dumps(words['last_run'], ensure_ascii=False)};\n"
            + f"export const RECORDED_BOOK_HEAD = {json.dumps(words['head'])};\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="The compare tab's offline replay cards, made from the committed book.")
    ap.add_argument("--write", action="store_true", help="rewrite compareRecorded.ts from the book")
    args = ap.parse_args()
    try:
        text = render()
    except CardsRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    rel = OUT.relative_to(ROOT_DIR)
    if args.write:
        tmp = OUT.with_name(OUT.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(OUT)
        print(f"wrote {rel} ({len(text)} characters) from {BOOK.name}")
        return 0
    if OUT.is_file() and OUT.read_text(encoding="utf-8") == text:
        print(f"{rel} is what {BOOK.name} gives")
        return 0
    print(f"{rel} is NOT what {BOOK.name} gives: run this script with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
