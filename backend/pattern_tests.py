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

   Behind its own passage. Some faces fire only far into their own document, and no short text fires them (below).
   For those the earlier text is held and the ending swapped: one of the face's stored passages (not kept back) is
   kept up to DROP word-pieces before its peak, and each written text is read as its ending in place of the passage's
   own. Endings written to the pattern should fire; near-miss endings, which are required here, should not: with the
   earlier text held, a near miss that fires says the earlier text does the firing and not what the pattern names.
   The passage's own ending is read the same way beside them. Each ending's peak is taken over its own word-pieces.

2. Blind (`judge_prompt`, `parse_picks`, `blind_result`). The passages index keeps each face's strongest events and a
   few drawn at random above half its extreme. Those in odd-numbered windows are kept back from the scientist
   (`atlas_index.passage_kept_back`), so a pattern written from the others has not seen them. Each kept-back event is
   paired with a passage from another kept-back window where the face stays under QUIET_BELOW of its extreme, read
   live. Both are cut alike, BEFORE word-pieces before one marked word-piece and AFTER after. A reader given the
   pattern alone, and none of the neuron's passages, says which of each pair fires it. Chance is one half; `p` is the
   chance of doing as well by guessing.

   The quiet passage is a near miss where the corpus has one (`near_places`): another place with the same marked
   word-piece between the same two neighbours where the face stays quiet; failing that, with the same word-piece
   before it, or after it, or the word-piece alone; and only then a passage from anywhere. Against a passage from
   anywhere a pattern that names the word and its phrase is picked rightly whether or not that phrase is what fires
   the face (the pilot's six faces, below). Against a near miss the phrase is in both snippets, and only what the
   pattern says of the rest of the text tells them apart. Each place tried is also a reading of its own: how often
   the face fires where the same phrase stands elsewhere (`places`) says, with no reader at all, whether the phrase is
   enough. The places are drawn outside the windows the face's stored events are in, which are where it fires
   hardest, so for a phrase the corpus has only a few times the count runs low. Both snippets of a pair show the same
   number of word-pieces, so a short one gives nothing away.

What neither test shows: that the pattern is the face's only job (each takes the pattern's word for what to look
for; a second job the pattern leaves out is not looked for), or what the neuron's firing does downstream. They test
when it fires, not what it writes.

Tried on the 31B on 14 November 2026 (experiments/results/pattern_tests_pilot): a blind test could be begun for 175
of 176 faces drawn at random at eleven layers, a median of 9 pairs each. Over 21 faces whose patterns were written
from the passages not kept back, a reader picked 177 of 196 pairs rightly with the face's own pattern and 86 of 196
with another face's, so the pairs give nothing away without the pattern. The written test found what the blind one
cannot: 6 of the 21 faces do not fire on their own strongest passages when only the 60 word-pieces before the peak
are read, and none of the 18 texts written for those six fired, though a reader picked 49 of their 54 pairs. What
fires such a face lies further back than a short text or a snippet reaches (of the 176 faces, 30 need more than 60
word-pieces and 8 more than 256; all 8 are at layer 26 or later). So the written test reads the face's own passages
alone as well (`own_result`), and says so when a text's silence is the face's need of more text and not the
pattern's fault. A blind test passed against passages drawn anywhere in the corpus does not show that the pattern is
enough to fire the face.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FIRES_AT = 0.5       # a text fires a face when its peak reaches this share of the face's extreme
QUIET_BELOW = 0.25   # a control passage is quiet when the face stays under this share of its extreme in what is shown
BEFORE, AFTER = 60, 8  # a snippet: the word-pieces shown before the marked one, and after it
MIN_PAIRS, MAX_PAIRS = 4, 12
MAX_TEXTS = 12
CONTROL_TRIES = 8    # passages tried for one quiet control before the test is refused
MAX_BEFORE = 256     # the most word-pieces a blind test may show before the marked one (BEFORE unless asked)
# how near a miss a quiet passage is, nearest first: the marked word-piece between the same two neighbours; with the
# same one before it; with the same one after it; the word-piece alone. Each level leaves out the places of those before it.
NEAR_LEVELS = ("phrase", "before", "after", "word")
ANYWHERE = "anywhere"  # no near miss: a passage drawn from anywhere in the corpus
NEAR_TRIES = 6       # places tried at one level for one pair before the next level is tried
NEAR_SAID = {"phrase": "the same word-piece between the same two neighbours", "before": "the same word-piece after the same one",
             "after": "the same word-piece before the same one", "word": "the same word-piece"}
DROP, MAX_DROP = 8, 64  # behind its own passage: the word-pieces up to and including the peak that an ending replaces
OWN_PASSAGES = 3     # the face's own strongest passages (not kept back) that a written test reads alone beside its texts
HOLDS = 0.5          # a passage holds alone when, with BEFORE word-pieces before its peak, it reads this share of what it read in its window


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


def own_result(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The written test's own check, from the face's strongest passages read alone ({"in_window": the peak as the index
    holds it, read with all its window had before it; "alone": the same peak with only BEFORE word-pieces before it;
    both as shares of the face's extreme}): how many still read HOLDS of what they did (`hold`, of `of`), and whether
    most do (`holds`). Where they do not, no short text can fire the face, whatever the pattern."""
    hold = sum(float(r["alone"]) >= HOLDS * float(r["in_window"]) for r in rows)
    return {"passages": [dict(r) for r in rows], "hold": hold, "of": len(rows), "before": BEFORE, "holds": hold * 2 > len(rows)}


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


def near_places(windows: np.ndarray, tok: int, prev: Optional[int], nxt: Optional[int], level: str) -> Optional[np.ndarray]:
    """Where the corpus has the word-piece `tok` as a near miss of one level: rows of (window, position) in `windows`
    (one window a row, padded with negatives), in reading order. "phrase": between `prev` and `nxt`; "before": after
    `prev`; "after": before `nxt`; "word": anywhere. None where the level needs a neighbour the place does not have.
    The levels nest (a "phrase" place is also a "before", an "after" and a "word" place); the caller leaves out what a
    nearer level already gave."""
    if level not in NEAR_LEVELS:
        raise ValueError(f"A near miss is one of {', '.join(NEAR_LEVELS)}.")
    if (level in ("phrase", "before") and prev is None) or (level in ("phrase", "after") and nxt is None):
        return None
    m = windows == int(tok)
    if level in ("phrase", "before"):
        m[:, 1:] &= windows[:, :-1] == int(prev)
        m[:, 0] = False
    if level in ("phrase", "after"):
        m[:, :-1] &= windows[:, 1:] == int(nxt)
        m[:, -1] = False
    return np.argwhere(m)


def blind_result(picks: Sequence[Optional[str]], truth: Sequence[str], match: Optional[Sequence[str]] = None,
                 fires: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """The blind test's figures: how many pairs were picked rightly, of how many, and the chance of that by guessing.
    With `match`, how near a miss each pair's quiet passage is (a NEAR_LEVELS level, or ANYWHERE): the same by level
    (`by_match`), and over the pairs whose quiet passage is a near miss of any level (`near_pairs`). With `fires`, each
    pair's firing passage as a share of the face's extreme: the same over the pairs where it reaches FIRES_AT
    (`strong_pairs`). A face's stored passages are its strongest, and the weakest of those can be a third of its
    extreme: a pattern can be right of where the face fires hard and miss those."""
    n = len(truth)
    got = [(str(p).strip().lower() if p is not None else None) for p in list(picks)[:n]] + [None] * max(0, n - len(picks))
    right = sum(1 for p, t in zip(got, truth) if p == t)
    out = {"kind": "blind", "pairs": n, "right": right, "unanswered": sum(p not in ("a", "b") for p in got),
           "by_chance": n / 2, "p": round(binom_tail(right, n), 4)}
    if match is not None:
        by: Dict[str, Dict[str, int]] = {}
        for p, t, m in zip(got, truth, match):
            row = by.setdefault(str(m), {"pairs": 0, "right": 0})
            row["pairs"] += 1
            row["right"] += int(p == t)
        out["by_match"] = {lv: by[lv] for lv in (*NEAR_LEVELS, ANYWHERE) if lv in by}
        near = [(p, t) for p, t, m in zip(got, truth, match) if m != ANYWHERE]
        nr = sum(1 for p, t in near if p == t)
        out["near_pairs"] = {"pairs": len(near), "right": nr, "p": round(binom_tail(nr, len(near)), 4)}
    if fires is not None:
        strong = [(p, t) for p, t, f in zip(got, truth, fires) if float(f) >= FIRES_AT]
        out["strong_pairs"] = {"pairs": len(strong), "right": sum(1 for p, t in strong if p == t), "fires_at": FIRES_AT}
    return out


def places_said(places: Optional[Dict[str, Dict[str, int]]]) -> Optional[str]:
    """What the places tried say of the phrase by itself, at the nearest level any were tried at."""
    for lv in NEAR_LEVELS:
        row = (places or {}).get(lv) or {}
        if row.get("tried"):
            return f"elsewhere in the corpus it fired at {row['fired']} of {row['tried']} places tried that have {NEAR_SAID[lv]}"
    return None


def said(test: Dict[str, Any]) -> str:
    """A kept test in a line."""
    if test.get("kind") == "written" and test.get("behind"):
        b = test["behind"]
        s = (f"written behind its own passage in window {b['window']} (the last {b['drop']} word-pieces swapped; its own ending reads "
             f"{b['own_ending']['rel']:g} of the face's extreme): {test['fired']} of {test['fire']} endings meant to fire it did")
        if test.get("strongest") is not None:
            s += f" (the strongest {test['strongest']:g})"
        return s + f"; {test['quiet_fired']} of {test['quiet']} near-miss endings did"
    if test.get("kind") == "written":
        s = f"written: {test['fired']} of {test['fire']} texts meant to fire it did"
        if test.get("strongest") is not None:
            s += f" (the strongest {test['strongest']:g} of its extreme in the corpus)"
        if test.get("quiet"):
            s += f"; {test['quiet_fired']} of {test['quiet']} near misses did"
        own = test.get("own")
        if own and own.get("of") and not own.get("holds"):
            s += (f". Its own strongest passages do not fire it either when read alone ({own['hold']} of {own['of']} do, with "
                  f"{own['before']} word-pieces before the peak): what fires it lies further back than a short text reaches, "
                  "and the pattern does not say what. It can be tested behind one of its own passages")
        return s
    if test.get("kind") == "blind":
        s = (f"blind: {test['right']} of {test['pairs']} pairs picked rightly by {test.get('judge') or 'a reader'} "
             f"({test['by_chance']:g} by chance; {test['p']:g} the chance of doing as well by guessing)")
        near = test.get("near_pairs")
        if test.get("control") == "near" and near is not None:
            s += (f"; {near['pairs']} of the quiet passages are near misses, the same word-piece where the face stays quiet, and "
                  f"{near['right']} of those pairs were picked rightly" if near["pairs"] else
                  "; the corpus had no near miss to set against any of them, so every quiet passage is from anywhere")
            more = places_said(test.get("places"))
            if more:
                s += "; " + more
        strong = test.get("strong_pairs")
        if strong is not None and strong["pairs"] < test["pairs"]:
            s += (f"; in {strong['pairs']} of the pairs the firing passage reaches {strong['fires_at']:g} of the face's extreme, and "
                  f"{strong['right']} of those were picked rightly")
        return s
    return str(test.get("kind"))
