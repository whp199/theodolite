"""A face's pattern put to the test, two ways that push nothing.

A pattern is a sentence or two saying when one face of a neuron fires (the field book's `pattern`, or a scientist's
description). The census and the steer lens say what a push along a neuron's write direction does; no push strength has
passed a trial below layer 32, and the steer lens is a weak guide below layer 39. These two tests need neither. They
ask whether the pattern predicts where the neuron fires by itself, in ordinary text, which a neuron does at every
layer. Both follow Goodfire's "Neurons as an Interpretable Basis" (Tigges, Kowal, McGrath, Lewis and Cammarata, 10
November 2026), which tested descriptions of Qwen3.6-35B-A3B's neurons by detection (a judge given the description
picks the passage the neuron fires in from one where it stays quiet: 86 to 95 in 100) and by input synthesis (text
written from the description fires the neuron harder than anything in a ten-billion-token corpus, for 94 in 100).

1. Written (`written_result`). Texts written to fire the face, and near misses written not to, are run through the
   model. Each text's peak on the face is given as a share of the face's extreme in the stats index. A text fires the
   face at FIRES_AT of that extreme, the band the passages index samples its random events from, and beats the corpus
   at 1. The test says how many of the texts meant to fire did, how many near misses did, and how far apart the two
   sets lie.

2. Blind (`judge_prompt`, `parse_picks`, `blind_result`). The passages index keeps each face's strongest events and a
   few drawn at random above half its extreme. Those in odd-numbered windows are kept back from the scientist
   (`atlas_index.passage_kept_back`), so a pattern written from the others has not seen them. Each kept-back event is
   paired with a passage from another kept-back window where the face stays under QUIET_BELOW of its extreme, read
   live. Both are cut alike, BEFORE word-pieces before one marked word-piece and AFTER after. A reader given the
   pattern alone, and none of the neuron's passages, says which of each pair fires it. Chance is one half; `p` is the
   chance of doing as well by guessing.

What neither test shows: that the pattern is the face's only job (each takes the pattern's word for what to look
for; a second job the pattern leaves out is not looked for), or what the neuron's firing does downstream. They test
when it fires, not what it writes.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

FIRES_AT = 0.5       # a text fires a face when its peak reaches this share of the face's extreme
QUIET_BELOW = 0.25   # a control passage is quiet when the face stays under this share of its extreme in what is shown
BEFORE, AFTER = 60, 8  # a snippet: the word-pieces shown before the marked one, and after it
MIN_PAIRS, MAX_PAIRS = 4, 12
MAX_TEXTS = 12
CONTROL_TRIES = 8    # passages tried for one quiet control before the test is refused


def binom_tail(right: int, n: int) -> float:
    """The chance of `right` or more of `n` pairs picked rightly by guessing."""
    if n <= 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(max(0, int(right)), n + 1)) / 2.0 ** n


def written_result(fire: Sequence[Dict[str, Any]], quiet: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The written test's figures from each text's reading ({"rel": its peak as a share of the face's extreme, ...}):
    how many of the texts meant to fire did (`fired`), how many beat the corpus (`beat_corpus`), the strongest, how
    many near misses fired (`quiet_fired`), and `apart`, the share of (firing text, near miss) pairs in which the
    firing text reads higher (1 = every one; a half = no better than a coin; None without both kinds)."""
    f = [float(r["rel"]) for r in fire]
    q = [float(r["rel"]) for r in quiet]
    apart = None
    if f and q:
        wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in f for b in q)
        apart = round(wins / (len(f) * len(q)), 3)
    return {"kind": "written", "fire": len(f), "fired": sum(x >= FIRES_AT for x in f), "beat_corpus": sum(x >= 1.0 for x in f),
            "strongest": round(max(f), 3) if f else None, "quiet": len(q), "quiet_fired": sum(x >= FIRES_AT for x in q),
            "quiet_strongest": round(max(q), 3) if q else None, "apart": apart, "fires_at": FIRES_AT}


def render(units: Sequence[Sequence[Any]]) -> str:
    """A snippet as a reader is shown it: its display units joined, the marked one inside [[ ]]."""
    return "".join(f"[[{text}]]" if marked else str(text) for text, marked in units)


JUDGE_SYSTEM = ("You are testing a description of when one neuron of a language model fires. You are given the "
                "description and pairs of text snippets. In each snippet one word-piece is marked [[like this]]. In "
                "exactly one snippet of each pair the neuron fires strongly on the marked word-piece; in the other it "
                "stays quiet. Using only the description, say which. Answer with one line per pair, the pair's number, "
                "a colon and the letter a or b, for example `3: b`. Give no other text.")


def judge_prompt(pattern: str, pairs: Sequence[Dict[str, str]]) -> str:
    """What a blind reader is given: the pattern, and each pair's two snippets."""
    out = ["The description of when the neuron fires:", "", pattern.strip(), ""]
    for i, p in enumerate(pairs, 1):
        out += [f"Pair {i}", "a) " + p["a"], "b) " + p["b"], ""]
    out.append(f"Answer with {len(pairs)} lines, `1: a` or `1: b` and so on.")
    return "\n".join(out)


def parse_picks(text: str, n: int) -> List[Optional[str]]:
    """A reader's answers, pair by pair: "a", "b", or None where it gave none that can be read (which counts as wrong)."""
    picks: List[Optional[str]] = [None] * n
    for m in re.finditer(r"(?im)^\W*(?:pair\s*)?(\d+)\s*[:.)\-=]\s*\(?([ab])\b", text or ""):
        i = int(m.group(1))
        if 1 <= i <= n and picks[i - 1] is None:
            picks[i - 1] = m.group(2).lower()
    return picks


def blind_result(picks: Sequence[Optional[str]], truth: Sequence[str]) -> Dict[str, Any]:
    """The blind test's figures: how many pairs were picked rightly, of how many, and the chance of that by guessing."""
    n = len(truth)
    got = [(str(p).strip().lower() if p is not None else None) for p in list(picks)[:n]] + [None] * max(0, n - len(picks))
    right = sum(1 for p, t in zip(got, truth) if p == t)
    return {"kind": "blind", "pairs": n, "right": right, "unanswered": sum(p not in ("a", "b") for p in got),
            "by_chance": n / 2, "p": round(binom_tail(right, n), 4)}


def said(test: Dict[str, Any]) -> str:
    """A kept test in a line."""
    if test.get("kind") == "written":
        s = f"written: {test['fired']} of {test['fire']} texts meant to fire it did"
        if test.get("strongest") is not None:
            s += f" (the strongest {test['strongest']:g} of its extreme in the corpus)"
        if test.get("quiet"):
            s += f"; {test['quiet_fired']} of {test['quiet']} near misses did"
        return s
    if test.get("kind") == "blind":
        return (f"blind: {test['right']} of {test['pairs']} pairs picked rightly by {test.get('judge') or 'a reader'} "
                f"({test['by_chance']:g} by chance; {test['p']:g} the chance of doing as well by guessing)")
    return str(test.get("kind"))
