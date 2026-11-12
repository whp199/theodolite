#!/usr/bin/env python3
"""A CPU-only check of test_motif_hypothesis's test against chance (backend/mcp_scientist.py `motif_statistics`, the
reply's `selectivity`), of the tool and its HTTP route on the tiny random Gemma 4 of scripts/atlas_stand_in.py, and of
the words the folio shows for it (frontend/src/screens/NeuronFolio/replies.ts, bundled with the frontend's own esbuild
and run under node, as scripts/check_interface_defaults.py runs the store):
1. the statistics on hand-made activations whose statistics are known: four a side apart gives Welch's t = 4.38 on 6
   degrees of freedom, Cohen's d = 3.10 over sample variances and an exact p of 2/70, with the verdict and words for
   either face and for a gap the other way; unequal sides against Welch's t, its degrees of freedom and d written out
   by hand; exact p's against every split counted in plain Python; no spread at all, rounding included, as an infinite
   t; identical sides (p = 1, not shown); three a side as far apart as can be (p = 0.1, too few to judge) and one a
   side (no t, no p, too few); past 10,000 splits a seeded permutation p, the same at each call, within its Monte Carlo
   error of the exact p, and never 0; an exact p that is a tie at two figures (0.125) written as Python rounds it
   (`p_text` "0.12"); every reply JSON without NaN or infinity; an empty side refused in plain words, and readings
   that are not finite numbers left untested (CANNOT_TEST), said so;
2. the tool, its capture replaced by hand-made activations: its `selectivity` is `motif_statistics` of them, read
   from the readings unrounded (5.00004 is not the 5.0 its record shows), the keys as first built are kept (`verdict`
   and `cohens_d` over population variances: one sentence a side still reads CONFIRMED, d = 50,000, while the test
   says too few to judge; a reading that is not a finite number still gives the reply), `label_gate` and `lens` last;
   an empty side, a layer or a neuron that does not exist refused in plain words before the model is run;
3. the HTTP route (POST /api/scientist/test_hypothesis, TestClient): the same reply, a reading that is not a finite
   number included, the same refusals as 400s, and a run on the stand-in's own activations;
4. the folio's words: for each verdict the server's own words with the faces named ("selective on its positive face,
   p = 0.029, n = 4 a side"), the server's own p where a tie would round apart, an older server's verdict never read
   as a yes, a p written alike by both and never rounded up to 0.05 from below, and no "yes, clearly" left in the
   folio; the paired reading's words when the server paired, the peak's and the either test's without the server's
   lead, and the folio no longer saying the peak is not tested;
5. the three later tests, each against a reference written out in plain Python (section 1b, and through the tool and
   the route in 2 and 3):
   - paired (`paired_statistics`, `selectivity_paired`): six pairs all one way give a mean difference of 2.5, a paired
     t of 4.44 and an exact p of 2/64; a hand-counted 6/64; exact p's against every flipping of the signs enumerated;
     the paired t's own p the same as the mean's; four and five pairs as far apart as can be are too few (p 0.125 and
     0.0625: no flipping reaches 0.05), six are judged; past 10,000 flippings (14 pairs) a seeded p within its Monte
     Carlo error of the exact one; matched pairs shown where the unpaired test shows nothing; unequal sides refused;
   - the peak (`selectivity_peak`) and the choice between the two readings (`either_statistics`,
     `selectivity_either`): the either p against every relabelling enumerated with both t's worked by hand, never below
     the p of the reading it came from, equal to it when the two readings are the same; a gap at a middle word-piece
     found by the peak test (2/70) and not by the last, and costing 4/70 once the choice is paid for; and a seeded
     simulation of 2,000 neurons that tell nothing apart (noise at every word-piece of sentences three to eight long,
     the peak picked as the tool picks it): the peak's p and the either p fall at or under 0.05 about one time in
     twenty and are spread evenly, where the smaller of the two readings' p's does so nearly twice as often;
   - Benjamini-Hochberg (`benjamini_hochberg`, `correction`): q's of four p's and of fifteen worked by hand, the same
     findings as the step-up rule as its authors state it, never below p, a test without a p left out of the count;
     through the tool, three neurons named in one call, one run of the model a sentence: a neuron that passes alone
     (p = 0.029) and not among three (q = 0.086), said so with how many must pass; one neuron alone told in words that
     its p is corrected for no other.

Run: .venv/bin/python -I scripts/check_hypothesis_test.py   (exits 0 when every check holds)
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"

import asyncio
import itertools
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

TMP = Path(tempfile.mkdtemp(prefix="hypothesis-check-"))
os.environ["THEODOLITE_ATLAS_DIR"] = str(TMP / "atlas")
os.environ["THEODOLITE_FIELD_NOTES"] = str(TMP / "notes.json")

ROOT_DIR = Path(__file__).resolve().parent.parent
for p in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

import backend.mcp_scientist as ms
from atlas_stand_in import FFN, N_LAYERS, bind_server, stand_in

FOLIO = ROOT_DIR / "frontend" / "src" / "screens" / "NeuronFolio"
ESBUILD = ROOT_DIR / "frontend" / "node_modules" / ".bin" / "esbuild"
OLD_KEYS = ["layer", "neuron_idx", "chat", "verdict", "cohens_d", "mean_pos_signed_act", "mean_neg_signed_act",
            "pos_prompt_results", "neg_prompt_results", "pole_j_vindex_signature", "pole_label_check", "label_gate", "lens"]
NEW_KEYS = ["selectivity", "design", "judged_by", "selectivity_paired", "selectivity_peak", "selectivity_either", "correction"]

failures: List[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL  {what}")


def section(title: str, before: int) -> None:
    if len(failures) == before:
        print(f"  ok    {title}")


def close(a: Optional[float], b: float, tol: float = 1e-4) -> bool:
    return a is not None and abs(a - b) <= tol * max(1.0, abs(b))


def welch_by_hand(pos: List[float], neg: List[float]) -> tuple:
    """Welch's t, its degrees of freedom and Cohen's d over the pooled sample variance, written out in plain Python."""
    n1, n2 = len(pos), len(neg)
    m1, m2 = sum(pos) / n1, sum(neg) / n2
    v1 = sum((x - m1) ** 2 for x in pos) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in neg) / (n2 - 1)
    se2 = v1 / n1 + v2 / n2
    df = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    d = (m1 - m2) / math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (m1 - m2) / math.sqrt(se2), df, d


def exact_p_by_hand(pos: List[float], neg: List[float]) -> float:
    """The two-sided p over every split of the sentences into the two sides' sizes, counted in plain Python."""
    x, n1 = pos + neg, len(pos)
    t_obs = abs(welch_by_hand(pos, neg)[0])
    hits = total = 0
    for pick in itertools.combinations(range(len(x)), n1):
        a = [x[i] for i in pick]
        b = [x[i] for i in range(len(x)) if i not in pick]
        total += 1
        hits += abs(welch_by_hand(a, b)[0]) >= t_obs * (1 - 1e-9)
    return hits / total


def paired_by_hand(pos: List[float], neg: List[float]) -> tuple:
    """The mean of the pairs' differences, their sample standard deviation and the paired t, in plain Python."""
    d = [a - b for a, b in zip(pos, neg)]
    n = len(d)
    m = sum(d) / n
    v = sum((x - m) ** 2 for x in d) / (n - 1)
    return m, math.sqrt(v), (m / math.sqrt(v / n) if v > 0 else math.copysign(math.inf, m) if m else 0.0)


def flip_p_by_hand(pos: List[float], neg: List[float], by_t: bool = False) -> float:
    """The two-sided p over every way to keep or flip the sign of each pair's difference, counted in plain Python: the
    share whose |mean| (whose |paired t|, with `by_t`) is at least the one observed."""
    d = [a - b for a, b in zip(pos, neg)]
    zeros = [0.0] * len(d)
    weigh = (lambda x: abs(paired_by_hand(x, zeros)[2])) if by_t else (lambda x: abs(sum(x) / len(x)))
    obs = weigh(d)
    hits = total = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(d)):
        total += 1
        hits += weigh([s * x for s, x in zip(signs, d)]) >= obs * (1 - 1e-9)
    return hits / total


def either_p_by_hand(last: tuple, peak: tuple, paired: bool) -> float:
    """The either test's p in plain Python (`last`, `peak`: each reading's two sides): under every relabelling, both
    readings' t's worked by hand and the larger |t| taken; the share that reaches the larger of the two observed."""
    if paired:
        ds = [[a - b for a, b in zip(*r)] for r in (last, peak)]
        zeros = [0.0] * len(ds[0])
        every: List[Any] = list(itertools.product((1.0, -1.0), repeat=len(zeros)))

        def weigh(signs: Any) -> float:
            return max(abs(paired_by_hand([s * x for s, x in zip(signs, d)], zeros)[2]) for d in ds)
    else:
        xs = [r[0] + r[1] for r in (last, peak)]
        n, n1 = len(xs[0]), len(last[0])
        every = list(itertools.combinations(range(n), n1))

        def weigh(pick: Any) -> float:
            return max(abs(welch_by_hand([x[i] for i in pick], [x[i] for i in range(n) if i not in pick])[0]) for x in xs)
    obs = weigh(every[0])  # the first is the labelling as given: every sign kept, the first n1 sentences the + side
    return sum(weigh(e) >= obs * (1 - 1e-9) for e in every) / len(every)


def step_up_by_hand(ps: List[float], alpha: float) -> List[int]:
    """Benjamini and Hochberg's rule as they state it: with the m p's in rising order, the largest k whose p is at most
    k / m * alpha; the k smallest are the findings (their places in `ps`)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    k = max((r for r in range(1, m + 1) if ps[order[r - 1]] <= r / m * alpha), default=0)
    return sorted(order[:k])


def main() -> int:
    try:
        return run_checks()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def run_checks() -> int:
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------------------- 1. the statistics, hand-made activations
    before = len(failures)
    s = ms.motif_statistics([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    check(s["n_pos"] == 4 and s["n_neg"] == 4 and s["mean_pos"] == 6.5 and s["mean_neg"] == 2.5 and close(s["sd_pos"], math.sqrt(5 / 3))
          and close(s["sd_neg"], math.sqrt(5 / 3)), f"four a side: the n, means and sample standard deviations: {s}")
    check(close(s["welch_t"], 4 / math.sqrt(2 * (5 / 3) / 4)) and close(s["welch_t"], 4.38178) and close(s["welch_df"], 6.0)
          and close(s["cohens_d"], 4 / math.sqrt(5 / 3)) and close(s["cohens_d"], 3.09839), f"Welch's t 4.38 on 6 degrees of freedom, d 3.10: {s}")
    check(s["p_method"] == "exact" and s["splits"] == 70 and close(s["p"], 2 / 70) and s["seed"] is None,
          f"the exact p over all 70 splits: only this split and its mirror are as far apart, 2/70: {s['p']}")
    check(s["verdict"] == "SELECTIVE_NORTH_POLE (+)" and s["face"] == "+" and s["words"] == "selective on its + face, p = 0.029, n = 4 a side"
          and s["p_text"] == "0.029" and s["min_n"] == 4 and s["alpha"] == 0.05, f"the verdict and its words: {s['verdict']}, {s['words']!r}, {s['p_text']!r}")
    s = ms.motif_statistics([-5.0, -6.0, -7.0, -8.0], [-1.0, -2.0, -3.0, -4.0])
    check(s["verdict"] == "SELECTIVE_SOUTH_POLE (-)" and s["face"] == "-" and close(s["welch_t"], -4.38178) and close(s["cohens_d"], -3.09839)
          and s["words"] == "selective on its - face, p = 0.029, n = 4 a side", f"the - face: {s['verdict']}, {s['words']!r}")
    s = ms.motif_statistics([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    check(s["verdict"] == "DIFFERS_NOT_AS_GUESSED" and s["face"] is None and close(s["p"], 2 / 70)
          and s["words"] == "the sides differ, p = 0.029, n = 4 a side, but not as guessed: the quiet sentences fire its + face more",
          f"a gap the other way is not the guess: {s['words']!r}")
    s = ms.motif_statistics([-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0])
    check(s["verdict"] == "DIFFERS_NOT_AS_GUESSED" and s["words"].endswith("the quiet sentences fire its - face more"), f"nor on the - face: {s['words']!r}")
    pos, neg = [1.0, 3.0, 5.0, 7.0, 9.0], [0.0, 0.5, 1.0, 1.5]
    t_, df_, d_ = welch_by_hand(pos, neg)
    s = ms.motif_statistics(pos, neg)
    check(close(s["welch_t"], t_) and close(s["welch_df"], df_) and close(s["cohens_d"], d_) and abs(df_ - 4.41155) < 1e-4,
          f"unequal sides and spreads: t {s['welch_t']} ({t_:.6g}), df {s['welch_df']} ({df_:.6g}), d {s['cohens_d']} ({d_:.6g})")
    check(s["splits"] == 126 and close(s["p"], exact_p_by_hand(pos, neg)) and "n = 5 and 4" in s["words"],
          f"its exact p over all 126 splits, counted by hand: {s['p']}, {s['words']!r}")
    rng = np.random.default_rng(7)
    agree = []
    for n1, n2, shift in ((4, 5, 1.0), (5, 5, 0.3), (6, 4, 2.0), (4, 4, 0.0)):
        a = [float(v) for v in rng.normal(shift, 1.0, n1)]
        b = [float(v) for v in rng.normal(0.0, 1.5, n2)]
        agree.append(close(ms.motif_statistics(a, b)["p"], exact_p_by_hand(a, b), 1e-5))
    check(all(agree), f"exact p's on random small sides equal every split counted by hand: {agree}")
    s = ms.motif_statistics([2.0] * 4, [1.0] * 4)
    r = ms.motif_statistics([0.1 + 0.2, 0.3, 0.3, 0.1 + 0.2], [0.1] * 4)
    check(s["welch_t"] is None and s["cohens_d"] is None and close(s["p"], 2 / 70) and s["verdict"] == "SELECTIVE_NORTH_POLE (+)"
          and r["welch_t"] is None and close(r["p"], 2 / 70), f"no spread on either side: an infinite t (sent as null), weighed as such, rounding included: {s['p']}, {r['p']}")
    s = ms.motif_statistics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    check(s["welch_t"] == 0.0 and s["p"] == 1.0 and s["verdict"] == "NOT_SHOWN" and s["words"] == "no difference shown: p = 1, n = 4 a side",
          f"identical sides: {s['words']!r}")
    s = ms.motif_statistics([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
    check(close(s["p"], 0.1) and s["splits"] == 20 and s["verdict"] == "TOO_FEW_TO_JUDGE"
          and s["words"] == "too few to judge: n = 3 a side, and the test needs at least 4 a side",
          f"three a side as far apart as can be: p = 0.1, so no split reaches 0.05, and it does not judge: {s['p']}, {s['words']!r}")
    s = ms.motif_statistics([5.0], [0.0])
    check(s["welch_t"] is None and s["p"] is None and s["cohens_d"] is None and s["sd_pos"] is None and s["verdict"] == "TOO_FEW_TO_JUDGE"
          and s["words"] == "too few to judge: n = 1 a side, and the test needs at least 4 a side", f"one a side: no t, no p: {s}")
    s = ms.motif_statistics([5.0, 6.0], [0.0, 1.0, 2.0, 3.0])
    check(s["verdict"] == "TOO_FEW_TO_JUDGE" and "n = 2 and 4" in s["words"], f"too few on one side only: {s['words']!r}")
    a = [float(v) for v in rng.normal(0.8, 1.0, 8)]
    b = [float(v) for v in rng.normal(0.0, 1.0, 8)]
    s1, s2 = ms.motif_statistics(a, b), ms.motif_statistics(a, b)
    exact = ms.motif_statistics(a, b, draws=20_000)
    tol = 4 * math.sqrt(exact["p"] * (1 - exact["p"]) / 10_000) + 2 / 10_001
    check(s1 == s2 and s1["p_method"] == "permutation" and s1["splits"] == 10_000 and s1["seed"] == 0 and exact["p_method"] == "exact"
          and exact["splits"] == 12_870, f"8 a side has 12,870 splits: 10,000 seeded shuffles, the same at each call: {s1['p']} {s2['p']}")
    check(abs(s1["p"] - exact["p"]) <= tol and abs(ms.motif_statistics(a, b, seed=1)["p"] - exact["p"]) <= tol and ms.motif_statistics(a, b, seed=1)["seed"] == 1,
          f"the permutation p within its Monte Carlo error of the exact one: {s1['p']} against {exact['p']} (± {tol:.4f})")
    s = ms.motif_statistics([10.0 + i for i in range(8)], [float(i) for i in range(8)])
    check(s["p_method"] == "permutation" and s["p"] >= 1 / 10_001 and s["p"] < 0.01 and s["verdict"] == "SELECTIVE_NORTH_POLE (+)",
          f"eight a side wholly apart: a permutation p never 0 ({s['p']}), and judged")
    s = ms.motif_statistics(*TIE)
    check(s["p"] == 0.125 and s["splits"] == 792 and s["p_text"] == "0.12" and s["words"] == "no difference shown: p = 0.12, n = 5 and 7",
          f"an exact p of 99/792 = 0.125, a tie at two figures, written as Python rounds it: {s['p']}, {s['p_text']!r}, {s['words']!r}")
    untested = ([1.0, float("nan"), 2.0, 3.0], [0.0, 1.0, 2.0, float("inf")])
    s = ms.motif_statistics(*untested)
    check(s["verdict"] == "CANNOT_TEST" and s["p"] is None and s["p_text"] is None and s["welch_t"] is None and s["cohens_d"] is None
          and s["mean_pos"] is None and s["sd_neg"] is None and s["face"] is None and s["n_pos"] == 4
          and s["words"] == "cannot test: 2 of the 8 readings are not a finite number, n = 4 a side",
          f"readings that are not finite numbers: nothing tested, not read as p = 0, and said so: {s['words']!r}")
    allowed = True
    for args in (([5.0], [0.0]), ([2.0] * 4, [1.0] * 4), ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]), (a, b), untested):
        try:
            json.dumps(ms.motif_statistics(*args), allow_nan=False)
        except ValueError:
            allowed = False
    check(allowed, "every reply is JSON without NaN or infinity")
    try:
        ms.motif_statistics([], [1.0])
        check(False, "an empty side: refused")
    except ValueError as exc:
        check("both sides" in str(exc), f"an empty side refused in plain words, not read as p = 0: {exc}")
    section("the statistics on hand-made activations", before)

    # ------------------------------------------------------- 1b. paired, the peak and the either test, Benjamini-Hochberg
    before = len(failures)
    s = ms.motif_statistics([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    check(s["design"] == "unpaired" and "unmatched sets" in s["tested"] and "last word-piece" in s["tested"] and close(s["least_p"], 2 / 70)
          and close(ms.motif_statistics(pos, neg)["least_p"], 1 / 126) and close(s1["least_p"], 1 / 10_001),
          f"the unpaired test says what it ran, and the least p its sentences can give (2/70 at four a side, 1/126 at five and four, 1/10,001 from shuffles): {s['tested']!r}")
    six = ([3.0, 5.0, 4.0, 6.0, 5.0, 7.0], [1.0, 2.0, 3.0, 2.0, 4.0, 3.0])  # differences 2, 3, 1, 4, 1, 4: all one way
    s = ms.paired_statistics(*six)
    check(s["design"] == "paired" and "matched pairs" in s["tested"] and s["n_pairs"] == 6 and s["mean_diff"] == 2.5 and close(s["sd_diff"], math.sqrt(9.5 / 5))
          and close(s["paired_t"], 2.5 / math.sqrt(9.5 / 5 / 6)) and close(s["paired_t"], 4.44262) and s["paired_df"] == 5 and s["mean_pos"] == 5.0 and s["mean_neg"] == 2.5,
          f"six pairs: the mean of their differences 2.5, its t 4.44 on 5 degrees of freedom: {s}")
    check(s["p_method"] == "exact" and s["flips"] == 64 and s["p"] == 2 / 64 and s["seed"] is None and s["least_p"] == 2 / 64 and s["min_n"] == 6
          and s["verdict"] == "SELECTIVE_NORTH_POLE (+)" and s["face"] == "+" and s["words"] == "selective on its + face, p = 0.031, n = 6 pairs" and s["p_text"] == "0.031",
          f"its exact p over all 64 flippings: only the pairs as given and every one flipped reach it, 2/64: {s['p']}, {s['words']!r}")
    s = ms.paired_statistics(six[1], six[0])
    check(s["verdict"] == "DIFFERS_NOT_AS_GUESSED" and s["mean_diff"] == -2.5 and s["p"] == 2 / 64
          and s["words"] == "the sides differ, p = 0.031, n = 6 pairs, but not as guessed: the quiet sentences fire its + face more", f"the pairs the other way: {s['words']!r}")
    s = ms.paired_statistics([-v for v in six[0]], [-v for v in six[1]])
    check(s["verdict"] == "SELECTIVE_SOUTH_POLE (-)" and s["words"] == "selective on its - face, p = 0.031, n = 6 pairs", f"the - face, paired: {s['words']!r}")
    # differences 3, 1, -1, 2, 4, 2 (sum 11, of 13 in all): a flipping reaches |11| only by turning nothing or one of the
    # two 1's about (sum 13, 11, 11), or all but those: 6 of the 64
    hand = ([3.0, 1.0, 0.0, 2.0, 4.0, 2.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    s = ms.paired_statistics(*hand)
    check(s["p"] == 6 / 64 and flip_p_by_hand(*hand) == 6 / 64 and s["verdict"] == "NOT_SHOWN" and s["words"] == "no difference shown: p = 0.094, n = 6 pairs",
          f"a p counted by hand, 6/64: {s['p']}, {s['words']!r}")
    agree, same_by_t, hand_t = [], [], []
    for n_pairs, shift in ((2, 1.0), (3, 0.0), (5, 0.5), (6, 1.0), (7, 0.3), (9, 0.8), (10, 0.0)):
        a = [float(v) for v in rng.normal(shift, 1.0, n_pairs)]
        b = [float(v) for v in rng.normal(0.0, 1.5, n_pairs)]
        s = ms.paired_statistics(a, b)
        agree.append(s["p"] == ms._sig(flip_p_by_hand(a, b)) and s["flips"] == 2**n_pairs)  # the p is sent to six figures
        same_by_t.append(flip_p_by_hand(a, b) == flip_p_by_hand(a, b, by_t=True))
        m_, sd_, t_ = paired_by_hand(a, b)
        hand_t.append(close(s["mean_diff"], m_, 1e-5) and close(s["sd_diff"], sd_, 1e-5) and close(s["paired_t"], t_, 1e-5))
    check(all(agree), f"exact paired p's on random pairs equal every flipping of the signs counted by hand: {agree}")
    check(all(same_by_t), f"the paired t rises and falls with |mean| under the flippings: its own p, counted by hand, is the same: {same_by_t}")
    check(all(hand_t), f"the mean difference, its standard deviation and the paired t as written out by hand: {hand_t}")
    few = {n_pairs: ms.paired_statistics([10.0 + i for i in range(n_pairs)], [0.0] * n_pairs) for n_pairs in (1, 4, 5, 6)}
    check(few[4]["p"] == 0.125 and few[5]["p"] == 0.0625 and few[4]["verdict"] == few[5]["verdict"] == "TOO_FEW_TO_JUDGE" and few[5]["least_p"] == 0.0625
          and few[5]["words"] == "too few to judge: n = 5 pairs, and the paired test needs at least 6 pairs" and few[6]["p"] == 2 / 64
          and few[6]["verdict"] == "SELECTIVE_NORTH_POLE (+)", "four and five pairs as far apart as can be: p = 0.125 and 0.0625, so no flipping reaches 0.05 and it "
          f"does not judge; six pairs it does: {few[4]['p']}, {few[5]['p']}, {few[5]['words']!r}, {few[6]['words']!r}")
    check(few[1]["p"] is None and few[1]["paired_t"] is None and few[1]["sd_diff"] is None and few[1]["verdict"] == "TOO_FEW_TO_JUDGE"
          and few[1]["words"] == "too few to judge: n = 1 pair, and the paired test needs at least 6 pairs", f"one pair: no t, no p: {few[1]}")
    a = [float(v) for v in rng.normal(0.6, 1.0, 14)]
    b = [float(v) for v in rng.normal(0.0, 1.0, 14)]
    f1, f2, exact = ms.paired_statistics(a, b), ms.paired_statistics(a, b), ms.paired_statistics(a, b, draws=20_000)
    tol = 4 * math.sqrt(exact["p"] * (1 - exact["p"]) / 10_000) + 2 / 10_001
    check(f1 == f2 and f1["p_method"] == "permutation" and f1["flips"] == 10_000 and f1["seed"] == 0 and close(f1["least_p"], 1 / 10_001)
          and exact["p_method"] == "exact" and exact["flips"] == 16_384 and exact["p"] == ms._sig(flip_p_by_hand(a, b)),
          f"14 pairs have 16,384 flippings: 10,000 seeded ones, the same at each call: {f1['p']} {f2['p']}")
    check(abs(f1["p"] - exact["p"]) <= tol and abs(ms.paired_statistics(a, b, seed=1)["p"] - exact["p"]) <= tol and ms.paired_statistics(a, b, seed=1)["seed"] == 1,
          f"the seeded paired p within its Monte Carlo error of the exact one: {f1['p']} against {exact['p']} (± {tol:.4f})")
    s = ms.paired_statistics([10.0 + i for i in range(14)], [0.0] * 14)
    check(s["p_method"] == "permutation" and s["p"] >= 1 / 10_001 and s["p"] < 0.01 and s["verdict"] == "SELECTIVE_NORTH_POLE (+)",
          f"fourteen pairs all one way: a seeded p never 0 ({s['p']}), and judged")
    base = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    matched = ([b_ + e for b_, e in zip(base, (1.0, 1.25, 0.75, 1.5, 0.5, 1.0))], base)
    s, u = ms.paired_statistics(*matched), ms.motif_statistics(*matched)
    check(s["p"] == 2 / 64 and s["verdict"] == "SELECTIVE_NORTH_POLE (+)" and u["verdict"] == "NOT_SHOWN" and u["p"] > 0.5,
          f"each firing sentence one above its own partner, the pairs far apart from one another: paired p = {s['p']}, unpaired p = {u['p']}")
    s = ms.paired_statistics([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    check(s["p"] == 1.0 and s["paired_t"] == 0.0 and s["words"] == "no difference shown: p = 1, n = 6 pairs", f"identical partners: {s['words']!r}")
    s = ms.paired_statistics([2.0] * 6, [1.0] * 6)
    r = ms.paired_statistics([0.1 + 0.2, 0.3, 0.3, 0.1 + 0.2, 0.3, 0.3], [0.1] * 6)
    check(s["paired_t"] is None and s["p"] == 2 / 64 and r["paired_t"] is None and r["p"] == 2 / 64 and r["verdict"] == "SELECTIVE_NORTH_POLE (+)",
          f"every pair the same difference: an infinite t (sent as null), weighed as such, rounding included: {s['p']}, {r['p']}")
    s = ms.paired_statistics([1.0, float("nan"), 2.0, 3.0, 4.0, 5.0], [0.0] * 6)
    check(s["verdict"] == "CANNOT_TEST" and s["p"] is None and s["mean_diff"] is None and s["words"] == "cannot test: 1 of the 12 readings is not a finite number, n = 6 pairs",
          f"a reading that is not a finite number, paired: {s['words']!r}")
    for args, said in ((([1.0, 2.0, 3.0], [1.0, 2.0]), "there are 3 and 2"), (([], []), "both sides")):
        try:
            ms.paired_statistics(*args)
            check(False, f"{said}: refused")
        except ValueError as exc:
            check(said in str(exc), f"refused in plain words: {exc}")

    # the either test: both readings' t's by hand under every relabelling, the larger taken
    agree, never_below = [], []
    for paired, sizes in ((False, (4, 4)), (False, (5, 4)), (False, (3, 6)), (True, (6, 6)), (True, (8, 8)), (True, (3, 3))):
        la = [float(v) for v in rng.normal(0.8, 1.0, sizes[0])]
        lb = [float(v) for v in rng.normal(0.0, 1.0, sizes[1])]
        pa = [float(v) for v in rng.normal(1.0, 2.0, sizes[0])]
        pb = [float(v) for v in rng.normal(0.0, 2.0, sizes[1])]
        e = ms.either_statistics(la, lb, pa, pb, paired=paired)
        one = ms.paired_statistics if paired else ms.motif_statistics
        p_last, p_peak = one(la, lb)["p"], one(pa, pb)["p"]
        agree.append(e["p"] == ms._sig(either_p_by_hand((la, lb), (pa, pb), paired)) and e["design"] == ("paired" if paired else "unpaired"))
        t_last = paired_by_hand(la, lb)[2] if paired else welch_by_hand(la, lb)[0]
        t_peak = paired_by_hand(pa, pb)[2] if paired else welch_by_hand(pa, pb)[0]
        never_below.append(e["p"] >= (p_last if abs(t_last) >= abs(t_peak) else p_peak) - 1e-12 and close(e["t_last"], t_last, 1e-5) and close(e["t_peak"], t_peak, 1e-5)
                           and e["stronger"] == ("last word-piece" if abs(t_last) >= abs(t_peak) else "strongest word-piece"))
    check(all(agree), f"the either p equals every relabelling counted by hand, both t's worked again and the larger taken: {agree}")
    check(all(never_below), f"it names the stronger reading, gives both t's, and is never below that reading's own p: {never_below}")
    north = ([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    e = ms.either_statistics(*north, *north)
    check(e["p"] == ms.motif_statistics(*north)["p"] and e["stronger"] == "last word-piece" and e["verdict"] == "SELECTIVE_NORTH_POLE (+)" and e["splits"] == 70
          and e["words"] == "last word-piece or strongest, whichever tells the sides apart more (here the last): selective on its + face, p = 0.029, n = 4 a side",
          f"two readings that are the same numbers: nothing more to pay, the p is that reading's: {e['words']!r}")
    # a gap at a middle word-piece and none at the last: the peak test alone finds it at 2/70, but the last word-piece's
    # readings (1, 2, 3, 4 on each side) have two splits of their own with a larger |t| (1, 1, 2, 2 against 3, 3, 4, 4,
    # and its mirror), so with the choice between the readings paid for it is 4/70
    mid_last, mid_peak = ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]), ([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    e = ms.either_statistics(*mid_last, *mid_peak)
    check(e["p"] == ms._sig(4 / 70) and either_p_by_hand(mid_last, mid_peak, False) == 4 / 70 and e["stronger"] == "strongest word-piece" and e["verdict"] == "NOT_SHOWN"
          and e["words"] == "last word-piece or strongest, whichever tells the sides apart more (here the strongest): no difference shown: p = 0.057, n = 4 a side"
          and e["mean_pos"] == 6.5 and "paid for" in e["tested"], f"the choice between two readings has a price: 2/70 for the peak alone, 4/70 for whichever is stronger: {e['words']!r}")
    e = ms.either_statistics([1.0, float("nan"), 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], *north)
    check(e["verdict"] == "CANNOT_TEST" and e["p"] is None and e["stronger"] is None and "1 of the 16 readings is not a finite number" in e["words"],
          f"a reading that is not a finite number in either: {e['words']!r}")
    e = ms.either_statistics([5.0, 6.0, 7.0], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [1.0, 2.0, 3.0], paired=True)
    check(e["verdict"] == "TOO_FEW_TO_JUDGE" and e["p"] == 0.25 and e["flips"] == 8 and e["min_n"] == 6 and e["words"].endswith("the paired test needs at least 6 pairs"),
          f"the either test keeps its design's floor: {e['words']!r}")
    try:
        ms.either_statistics([1.0, 2.0], [1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0])
        check(False, "two readings of unequal size: refused")
    except ValueError as exc:
        check("same sentences" in str(exc), f"refused in plain words: {exc}")

    # neurons that tell nothing apart: noise at every word-piece of sentences three to eight long, the peak picked as the
    # tool picks it. An exact p over 252 splits is at or under 0.05 in 12/252 = 4.8% of such runs, and over 64
    # flippings in 2/64 = 3.1%.
    sim = np.random.default_rng(11)
    null: Dict[str, List[float]] = {k: [] for k in ("last", "peak", "either", "paired last", "paired peak", "paired either")}
    for _ in range(2000):
        reads = [ms._last_and_peak(torch.from_numpy(sim.normal(size=int(sim.integers(3, 9))))) for _ in range(12)]
        last, peak = [r_[0] for r_ in reads], [r_[2] for r_ in reads]
        null["last"].append(ms.motif_statistics(last[:5], last[5:10])["p"])
        null["peak"].append(ms.motif_statistics(peak[:5], peak[5:10])["p"])
        null["either"].append(ms.either_statistics(last[:5], last[5:10], peak[:5], peak[5:10])["p"])
        null["paired last"].append(ms.paired_statistics(last[:6], last[6:])["p"])
        null["paired peak"].append(ms.paired_statistics(peak[:6], peak[6:])["p"])
        null["paired either"].append(ms.either_statistics(last[:6], last[6:], peak[:6], peak[6:], paired=True)["p"])

    def share(ps: List[float], at: float) -> float:
        return sum(v <= at for v in ps) / len(ps)

    for k in ("peak", "either"):
        ps = null[k]
        spread = [share(ps, at) for at in (0.05, 0.25, 0.5, 0.75)]
        check(0.03 <= spread[0] <= 0.07 and all(abs(got - at) <= 0.04 for got, at in zip(spread[1:], (0.25, 0.5, 0.75))) and abs(sum(ps) / len(ps) - 0.504) <= 0.03,
              f"2,000 neurons that tell nothing apart, five a side: the {k} test's p is at or under 0.05, 0.25, 0.5, 0.75 in {spread} of them, mean {sum(ps) / len(ps):.3f}")
    for k in ("paired peak", "paired either"):
        ps = null[k]
        spread = [share(ps, at) for at in (0.05, 0.25, 0.5, 0.75)]
        check(0.015 <= spread[0] <= 0.05 and all(abs(got - at) <= 0.05 for got, at in zip(spread[1:], (0.25, 0.5, 0.75))),
              f"the same as six pairs: the {k} test's p is at or under 0.05, 0.25, 0.5, 0.75 in {spread} of them")
    unpaid = share([min(a_, b_) for a_, b_ in zip(null["last"], null["peak"])], 0.05)
    check(unpaid >= 0.08 and unpaid >= share(null["either"], 0.05) + 0.02,
          f"the smaller of the two readings' own p's is at or under 0.05 in {unpaid} of them, the choice unpaid; the either p in {share(null['either'], 0.05)}")
    check(all(e_ >= min(a_, b_) - 1e-12 for e_, a_, b_ in zip(null["either"], null["last"], null["peak"]))
          and all(e_ >= min(a_, b_) - 1e-12 for e_, a_, b_ in zip(null["paired either"], null["paired last"], null["paired peak"])),
          "in every one of them the either p is no smaller than the smaller of the two readings' own")
    seq = torch.tensor([0.5, -3.0, 2.0, 3.0, 1.0])
    check(ms._last_and_peak(seq) == (1.0, 1, -3.0) and ms._last_and_peak(torch.tensor([2.0, -2.0, 1.0])) == (1.0, 0, 2.0),
          f"the peak as the tool picks it: the largest |activation|, its sign kept, the earliest of equals: {ms._last_and_peak(seq)}")

    # Benjamini-Hochberg
    qs = ms.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    check(all(close(q, w, 1e-12) for q, w in zip(qs, [0.02, 0.04, 0.04, 0.02])),
          f"four p's by hand: 0.005 * 4/1 = 0.02, 0.01 * 4/2 = 0.02, 0.03 * 4/3 = 0.04, 0.04 * 4/4 = 0.04: {qs}")
    fifteen = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.324, 0.4262, 0.5719, 0.6528, 0.759, 1.0]
    qs = ms.benjamini_hochberg(fifteen)
    check(close(qs[3], 0.0095 * 15 / 4, 1e-12) and close(qs[4], 0.0201 * 15 / 5, 1e-12) and close(qs[5], 0.0298 * 15 / 7, 1e-12) and close(qs[6], 0.0298 * 15 / 7, 1e-12)
          and [i for i, q in enumerate(qs) if q < 0.05] == [0, 1, 2, 3] and qs[-1] == 1.0,
          f"fifteen p's by hand: the fourth's q is 0.0095 * 15/4 = 0.036, the fifth's 0.0201 * 15/5 = 0.060, the sixth takes the seventh's (0.0298 * 15/7): {qs[:7]}")
    mixed = fifteen[::-1][3:] + fifteen[::-1][:3]  # out of order
    check(sorted(ms.benjamini_hochberg(mixed)) == sorted(qs) and ms.benjamini_hochberg(mixed)[mixed.index(0.0095)] == qs[3], "the q's keep the order the p's came in")
    same_findings = []
    for m_tests in (1, 2, 7, 40, 200):
        ps = [float(v) for v in np.concatenate([rng.uniform(0, 1, m_tests - m_tests // 3), rng.uniform(0, 0.01, m_tests // 3)])]
        qs = ms.benjamini_hochberg(ps)
        same_findings.append(all([i for i, q in enumerate(qs) if q <= al] == step_up_by_hand(ps, al) for al in (0.01, 0.05, 0.2))
                             and all(q >= p_ for q, p_ in zip(qs, ps)) and max(qs) <= 1.0
                             and all(qs[i] <= qs[j] for i in range(m_tests) for j in range(m_tests) if ps[i] <= ps[j]))
    check(all(same_findings), f"on random p's the tests with q at or under a level are the step-up rule's findings at that level; q is never below p, nor falls as p rises: {same_findings}")
    qs = ms.benjamini_hochberg([0.01, None, 0.04, None])
    check(qs == [0.02, None, 0.04, None] and ms.benjamini_hochberg([0.03]) == [0.03] and ms.benjamini_hochberg([]) == [] and ms.benjamini_hochberg([0.5, 0.9]) == [0.9, 0.9]
          and ms.benjamini_hochberg([0.03, 0.03, 0.03]) == [0.03, 0.03, 0.03],
          f"a test with no p is left out of the count and keeps none; one test's q is its p; equal p's share a q: {qs}")
    section("paired by sign flips, the peak and the choice of reading, Benjamini-Hochberg: each against plain Python", before)

    # ---------------------------------------------------------------------- 2. the tool, on hand-made activations
    before = len(failures)
    rt, vault = stand_in()
    app = bind_server(rt, vault)
    real_capture = rt.eager_model.capture
    table: Dict[tuple, Any] = {}
    calls = [0]
    captured: List[List[Any]] = []  # the addresses each capture was asked for
    N = 5  # the neuron read

    async def fake_capture(ids: List[int], addrs: List[Any]) -> Dict[Any, Any]:
        """The neuron's activation over the sentence: half its value at the start, its value at the last word-piece.
        A sentence entered as a pair has the first at a middle word-piece (the second) and the other at the last, the
        start reading 0; one entered as {neuron: value} is read so on each of those neurons, in whatever layer."""
        calls[0] += 1
        captured.append(list(addrs))
        out = {}
        for addr in addrs:
            seq = torch.zeros(len(ids), FFN)
            v = table[tuple(ids)]
            for neuron, val in v.items() if isinstance(v, dict) else [(N, v)]:
                if isinstance(val, tuple):
                    seq[1, neuron], seq[-1, neuron] = val
                else:
                    seq[0, neuron], seq[-1, neuron] = 0.5 * val, val
            out[addr] = seq
        return out

    def sentences(side: str, values: List[Any]) -> List[str]:
        """Sentences each read as `values` says (float32-exact values), entered in the fake capture's table."""
        out = []
        for i, v in enumerate(values):
            text = f"Sentence {side} {i} reads {v}."
            table[tuple(rt.tokenize_prompt(rt.compose_prompt(text, chat=False, system=None)))] = v
            out.append(text)
        return out

    rt.eager_model.capture = fake_capture
    cases = {
        "north": ([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]),
        "south": ([-5.0, -6.0, -7.0, -8.0], [-1.0, -2.0, -3.0, -4.0]),
        "differs": ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
        "same": ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
        "one": ([5.0], [0.0]),
        "eight": ([1.5, 0.25, 2.0, 1.0, 0.75, 1.25, 2.5, 0.5], [0.0, 0.5, -0.25, 1.0, 0.25, -0.5, 0.75, 0.125]),
        "tie": TIE,
        # not exact at the 4 places a record is rounded to: as float32 holds them, which is what the capture gives
        "unrounded": ([float(np.float32(v)) for v in (5.00004, 6.00004, 7.00004, 8.00004)], [1.0, 2.0, 3.0, 4.0]),
        "untested": ([5.0, float("nan"), 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]),
    }
    asked = {k: (sentences(f"{k}+", p_), sentences(f"{k}-", n_)) for k, (p_, n_) in cases.items()}
    replies: Dict[str, Any] = {}
    for k, (p_, n_) in asked.items():
        replies[k] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=p_, neg_prompts=n_))
    for k, (p_, n_) in cases.items():
        rep = replies[k]
        check(rep["selectivity"] == ms.motif_statistics(p_, n_), f"{k}: the reply's selectivity is motif_statistics of its readings: {rep['selectivity']}")
    rep = replies["north"]
    check([x["last_token_signed_act"] for x in rep["pos_prompt_results"]] == cases["north"][0] and rep["pos_prompt_results"][0]["peak_signed_act"] == 5.0,
          "the hand-made activations are what the tool read, the last word-piece and the peak")
    check(all(k in rep for k in OLD_KEYS) and list(rep)[-2:] == ["label_gate", "lens"] and rep["layer"] == 3 and rep["neuron_idx"] == N,
          f"every key as first built kept, label_gate and lens last: {list(rep)}")
    old_d = 4 / math.sqrt(0.5 * (1.25 + 1.25) + 1e-8)
    check(rep["verdict"] == "CONFIRMED_NORTH_POLE (+)" and rep["cohens_d"] == round(old_d, 4) and rep["selectivity"]["verdict"] == "SELECTIVE_NORTH_POLE (+)",
          f"verdict and cohens_d as first built, over population variances (d = {rep['cohens_d']}), beside the test")
    one = replies["one"]
    check(one["verdict"] == "CONFIRMED_NORTH_POLE (+)" and one["cohens_d"] == 50000.0 and one["selectivity"]["verdict"] == "TOO_FEW_TO_JUDGE"
          and one["selectivity"]["p"] is None, f"one sentence a side: as first built CONFIRMED at d = {one['cohens_d']}; the test: too few to judge")
    check(replies["eight"]["selectivity"]["p_method"] == "permutation" and replies["same"]["selectivity"]["verdict"] == "NOT_SHOWN"
          and replies["differs"]["selectivity"]["verdict"] == "DIFFERS_NOT_AS_GUESSED" and replies["south"]["selectivity"]["verdict"] == "SELECTIVE_SOUTH_POLE (-)",
          "each verdict through the tool")
    un = replies["unrounded"]
    check([x["last_token_signed_act"] for x in un["pos_prompt_results"]] == [5.0, 6.0, 7.0, 8.0] and un["mean_pos_signed_act"] == 6.5
          and un["selectivity"]["mean_pos"] == ms._sig(sum(cases["unrounded"][0]) / 4) and un["selectivity"]["mean_pos"] != 6.5,
          f"the test reads the last word-piece unrounded: mean {un['selectivity']['mean_pos']}, where the records, rounded, give {un['mean_pos_signed_act']}")
    nan_rep = replies["untested"]
    check(all(k in nan_rep for k in OLD_KEYS) and nan_rep["selectivity"]["verdict"] == "CANNOT_TEST" and nan_rep["selectivity"]["p"] is None
          and nan_rep["selectivity"]["words"] == "cannot test: 1 of the 8 readings is not a finite number, n = 4 a side",
          f"a reading that is not a finite number: the reply still comes, its keys as first built, and the test says it cannot: {nan_rep['selectivity']['words']!r}")
    # the later tests through the tool: what was run, the peak, the choice between the readings, one neuron alone
    rep = replies["north"]
    check(rep["design"] == "unpaired" and rep["judged_by"] == "selectivity" and rep["selectivity_paired"] is None and rep["selectivity"]["design"] == "unpaired"
          and "also" not in rep, f"unpaired unless asked, and the reply says so: {rep['design']}, {rep['judged_by']}")
    for k, (p_, n_) in cases.items():
        rep = replies[k]
        peak_p, peak_n = ([0.0 if v == 0 else v for v in side] for side in (p_, n_))  # the last word-piece is the strongest (the start reads half of it)
        want = dict(ms._unpaired(peak_p, peak_n, ms.MOTIF_DRAWS, ms.MOTIF_SEED, "strongest word-piece", "at each sentence's strongest word-piece: ")[0])
        at_start = sum(v == 0 for v in p_ + n_)  # a sentence that reads 0 throughout: its strongest word-piece is its first (one that is no number is not counted)
        got = {k_: v for k_, v in rep["selectivity_peak"].items() if k_ not in ("at_start", "note")}
        check(got == want and rep["selectivity_peak"]["at_start"] == at_start and (rep["selectivity_peak"]["note"] is None) == (at_start == 0)
              and rep["selectivity_either"] == ms.either_statistics(p_, n_, peak_p, peak_n),
              f"{k}: the peak's test is the same test on the strongest word-pieces, and the either test either_statistics of the two: {rep['selectivity_peak']}")
    peak = replies["north"]["selectivity_peak"]
    check(peak["read_at"] == "strongest word-piece" and peak["words"] == "at each sentence's strongest word-piece: selective on its + face, p = 0.029, n = 4 a side"
          and "largest |activation|" in peak["tested"] and "start mark" in peak["tested"] and peak["p"] == replies["north"]["selectivity"]["p"],
          f"the peak block says what was tested: {peak['words']!r}; {peak['tested']!r}")
    tie_peak = replies["tie"]["selectivity_peak"]
    check(tie_peak["at_start"] == 2 and tie_peak["note"] == "in 2 of the 12 sentences the strongest word-piece is the first, the start mark, which reads the same in "
          "every sentence: there this reads the start mark, not the sentence", f"sentences whose strongest word-piece is the start mark are counted and said: {tie_peak['note']!r}")
    alone = replies["north"]["correction"]
    check(alone["method"] is None and alone["tests"] == 1 and "corrected for no other neuron" in alone["words"] and "rows" not in alone,
          f"one neuron alone: no correction claimed, and the reply says so in words: {alone['words']!r}")
    # a gap at a middle word-piece and none at the last (the either test's own case, above), read through the tool
    mid = ([(5.0, 1.0), (6.0, 2.0), (7.0, 3.0), (8.0, 4.0)], [(0.5, 1.0), (0.5, 2.0), (0.5, 3.0), (0.5, 4.0)])
    mid_asked = (sentences("mid+", mid[0]), sentences("mid-", mid[1]))
    replies_more: Dict[str, Any] = {}
    rep = replies_more["mid"] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=mid_asked[0], neg_prompts=mid_asked[1]))
    check([x["peak_signed_act"] for x in rep["pos_prompt_results"]] == [5.0, 6.0, 7.0, 8.0] and [x["peak_signed_act"] for x in rep["neg_prompt_results"]] == [1.0, 2.0, 3.0, 4.0]
          and [x["last_token_signed_act"] for x in rep["pos_prompt_results"]] == [1.0, 2.0, 3.0, 4.0] and rep["selectivity"]["verdict"] == "NOT_SHOWN" and rep["selectivity"]["p"] == 1.0
          and rep["selectivity_peak"]["verdict"] == "SELECTIVE_NORTH_POLE (+)" and rep["selectivity_peak"]["p"] == ms._sig(2 / 70) and rep["selectivity_peak"]["at_start"] == 0
          and rep["selectivity_either"]["p"] == ms._sig(4 / 70) and rep["selectivity_either"]["verdict"] == "NOT_SHOWN" and rep["selectivity_either"]["stronger"] == "strongest word-piece",
          f"a gap at a middle word-piece: nothing at the last (p = 1), found at the peak (2/70), and 4/70 with the choice paid for: {rep['selectivity_either']['words']!r}")
    # matched pairs
    six_asked = (sentences("six+", six[0]), sentences("six-", six[1]))
    rep = replies_more["paired"] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=six_asked[0], neg_prompts=six_asked[1], paired=True))
    plain = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=six_asked[0], neg_prompts=six_asked[1]))
    check(rep["design"] == "paired" and rep["judged_by"] == "selectivity_paired" and rep["selectivity_paired"] == ms.paired_statistics(*six)
          and rep["selectivity_paired"]["words"] == "selective on its + face, p = 0.031, n = 6 pairs" and rep["selectivity"] == ms.motif_statistics(*six) == plain["selectivity"]
          and plain["design"] == "unpaired" and plain["selectivity_paired"] is None, f"paired when asked: {rep['selectivity_paired']['words']!r}; the unpaired reading is sent as before either way")
    check(rep["selectivity_peak"]["design"] == "paired" and rep["selectivity_peak"]["p"] == 2 / 64 and rep["selectivity_peak"]["flips"] == 64
          and rep["selectivity_peak"]["words"] == "at each sentence's strongest word-piece: selective on its + face, p = 0.031, n = 6 pairs"
          and rep["selectivity_either"] == ms.either_statistics(*six, *six, paired=True) and rep["selectivity_either"]["design"] == "paired"
          and plain["selectivity_peak"]["design"] == "unpaired" and plain["selectivity_either"]["design"] == "unpaired",
          f"the peak and the either test follow the design asked for: {rep['selectivity_peak']['words']!r}")
    check(all(k in rep for k in OLD_KEYS) and list(rep)[-2:] == ["label_gate", "lens"] and rep["verdict"] == plain["verdict"] and rep["cohens_d"] == plain["cohens_d"],
          "paired: every key as first built kept, with the same values")
    try:
        json.dumps([{k_: rep_[k_] for k_ in NEW_KEYS} for rep_ in [*replies.values(), *replies_more.values()]], allow_nan=False)
        check(True, "every reply's later keys are JSON without NaN or infinity")
    except ValueError as exc:
        check(False, f"every reply's later keys are JSON without NaN or infinity: {exc}")

    # several neurons in one call: 5 and 7 of layer 3 and 9 of layer 2, the same sentences, their p's corrected together
    def together(values: Dict[int, tuple]) -> tuple:
        """Four sentences a side, each read on every neuron of `values` ({neuron: (its + readings, its - readings)})."""
        tag = f"together{len(table)}"
        return tuple(sentences(f"{tag}{sign}", [{n_: sides[i][j] for n_, sides in values.items()} for j in range(4)]) for i, sign in enumerate("+-"))

    up, flat, mixed4 = ([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]), ([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]), ([2.0, 1.0, 4.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    one_of_three = together({N: up, 7: flat, 9: mixed4})
    n_before, seen_before = calls[0], len(captured)
    rep = replies_more["three"] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=one_of_three[0], neg_prompts=one_of_three[1], also=[[3, 7], [2, 9]]))
    check(calls[0] - n_before == 8 and all(sorted(a.layer for a in addrs) == [2, 3] for addrs in captured[seen_before:]),
          f"three neurons in two layers: one run of the model a sentence ({calls[0] - n_before} for 8), both layers asked of each")
    check(rep["selectivity"] == ms.motif_statistics(*up) and [(x["layer"], x["neuron_idx"]) for x in rep["also"]] == [(3, 7), (2, 9)]
          and rep["also"][0]["selectivity"] == ms.motif_statistics(*flat) and rep["also"][1]["selectivity"] == ms.motif_statistics(*mixed4)
          and [x["last_token_signed_act"] for x in rep["also"][1]["pos_prompt_results"]] == mixed4[0] and rep["also"][0]["selectivity_paired"] is None
          and all(k_ in x for x in rep["also"] for k_ in ("selectivity_peak", "selectivity_either", "neg_prompt_results"))
          and all(k in rep for k in OLD_KEYS) and list(rep)[-2:] == ["label_gate", "lens"],
          f"each neuron's own tests are what it would have alone: {[x['selectivity']['words'] for x in rep['also']]}")
    cor = rep["correction"]
    ps3 = [2 / 70, 1.0, 1.0]
    q3 = ms.benjamini_hochberg(ps3)
    check(cor["method"] == "Benjamini-Hochberg" and cor["tests"] == 3 and cor["design"] == "unpaired" and [(r_["layer"], r_["neuron_idx"]) for r_ in cor["rows"]] == [(3, N), (3, 7), (2, 9)]
          and [r_["last"]["q"] for r_ in cor["rows"]] == [ms._sig(q) for q in q3] and close(q3[0], 3 * 2 / 70, 1e-12) and all(r_[k_]["over"] == 3 for r_ in cor["rows"] for k_ in ("last", "peak", "either"))
          and [r_["last"]["p"] for r_ in cor["rows"]] == [ms._sig(v) for v in ps3],
          f"the correction: a q for each neuron over the three p's, the first neuron's first: {[(r_['last']['p'], r_['last']['q']) for r_ in cor['rows']]}")
    first = cor["rows"][0]["last"]
    check(rep["selectivity"]["verdict"] == "SELECTIVE_NORTH_POLE (+)" and first["verdict"] == "NOT_SHOWN" and first["face"] is None and first["q_text"] == "0.086"
          and first["words"] == "no difference shown: q = 0.086 over 3 neurons (p = 0.029), n = 4 a side",
          f"a neuron that passes alone (p = 0.029) does not among three (q = 3 * 2/70 = 0.086), and the words give both: {first['words']!r}")
    check(cor["found"] == {"last": 0, "peak": 0, "either": 0} and cor["least_p"] == ms._sig(2 / 70) and cor["needs"] == 2 and cor["words"].startswith("3 neurons tested together on the same sentences")
          and "no q falls under 0.05 unless at least 2 of the 3 neurons reach it" in cor["words"] and "can give no p under 0.029" in cor["words"],
          f"four a side cannot carry three neurons one at a time, and the reply says how many must pass: {cor['words']!r}")
    all_three = together({N: up, 7: up, 9: ([-v for v in up[0]], [-v for v in up[1]])})
    rep = replies_more["all three"] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=all_three[0], neg_prompts=all_three[1], also=[[3, 7], [2, 9]]))
    cor = rep["correction"]
    check([r_["last"]["verdict"] for r_ in cor["rows"]] == ["SELECTIVE_NORTH_POLE (+)", "SELECTIVE_NORTH_POLE (+)", "SELECTIVE_SOUTH_POLE (-)"] and cor["found"] == {"last": 3, "peak": 3, "either": 3}
          and cor["rows"][0]["last"]["words"] == "selective on its + face, q = 0.029 over 3 neurons (p = 0.029), n = 4 a side" and cor["rows"][2]["last"]["face"] == "-"
          and cor["rows"][1]["peak"]["words"] == "at each sentence's strongest word-piece: selective on its + face, q = 0.029 over 3 neurons (p = 0.029), n = 4 a side"
          and cor["rows"][0]["either"]["q"] == ms._sig(2 / 70) and "the sides differ in 3 at the last word-piece, 3 at the strongest, 3 taking whichever" in cor["words"],
          f"all three apart: each q is 2/70 * 3/3, and all pass: {cor['rows'][0]['last']['words']!r}")
    rep = replies_more["paired few"] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=asked["north"][0], neg_prompts=asked["north"][1], paired=True))
    check(rep["selectivity_paired"]["verdict"] == "TOO_FEW_TO_JUDGE" and rep["selectivity_paired"]["p"] == 0.125 and rep["selectivity"]["verdict"] == "SELECTIVE_NORTH_POLE (+)"
          and rep["selectivity_either"]["verdict"] == "TOO_FEW_TO_JUDGE" and rep["judged_by"] == "selectivity_paired",
          f"four pairs: the paired test does not judge (its least p is 0.125), though the unpaired reading of the same sentences passes: {rep['selectivity_paired']['words']!r}")
    six3 = tuple(sentences(f"sixes{sign}", [{N: six[i][j], 7: 1.0, 9: float("nan") if (i, j) == (0, 0) else six[i][j]} for j in range(6)]) for i, sign in enumerate("+-"))
    rep = replies_more["paired three"] = asyncio.run(ms.test_motif_hypothesis(layer=3, neuron_idx=N, pos_prompts=six3[0], neg_prompts=six3[1], paired=True, also=[[3, 7], [3, 9]]))
    cor = rep["correction"]
    check(cor["design"] == "paired" and [r_["last"]["p"] for r_ in cor["rows"]] == [2 / 64, 1.0, None] and [r_["last"]["q"] for r_ in cor["rows"]] == [2 / 64 * 2, 1.0, None]
          and [r_["last"]["over"] for r_ in cor["rows"]] == [2, 2, 2] and cor["rows"][2]["last"]["verdict"] == "CANNOT_TEST" and cor["rows"][0]["last"]["verdict"] == "NOT_SHOWN"
          and cor["rows"][0]["last"]["words"] == "no difference shown: q = 0.062 over 2 neurons (p = 0.031), n = 6 pairs" and rep["also"][1]["selectivity_paired"]["verdict"] == "CANNOT_TEST"
          and rep["also"][0]["selectivity_paired"] == ms.paired_statistics([1.0] * 6, [1.0] * 6),
          f"paired, three neurons, one that cannot be tested: it has no q and is left out of the count (over 2): {[(r_['last']['p'], r_['last']['q'], r_['last']['over']) for r_ in cor['rows']]}")
    n_calls = calls[0]
    pos4, neg4 = asked["north"]
    refusals = [
        (dict(layer=N_LAYERS, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4), f"layers count from 0 to {N_LAYERS - 1} here (the pages count them 1 to {N_LAYERS})"),
        (dict(layer=-1, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4), "layer is -1"),
        (dict(layer=True, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4), "layer is True"),
        (dict(layer=3, neuron_idx=FFN, pos_prompts=pos4, neg_prompts=neg4), f"each layer's neurons count from 0 to {FFN - 1}"),
        (dict(layer=3, neuron_idx=N, pos_prompts=[], neg_prompts=neg4), "pos_prompts is empty"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=[]), "neg_prompts is empty"),
        (dict(layer=3, neuron_idx=N, pos_prompts="a sentence", neg_prompts=neg4), "pos_prompts is empty"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4[:3], paired=True), "matched pairs, the first where it should fire with the first where it should stay quiet, and so on, and there are 4 and 3"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4, also=[[3, N]]), f"neuron {N} of layer 3 is named twice"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4, also=[[3, 7], [3, 7]]), "neuron 7 of layer 3 is named twice"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4, also=[[N_LAYERS, 7]]), f"a layer in also is {N_LAYERS}"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4, also=[[3, FFN]]), f"a neuron_idx in also is {FFN}"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4, also=[[3]]), "each further neuron is given as [layer, neuron_idx]"),
        (dict(layer=3, neuron_idx=N, pos_prompts=pos4, neg_prompts=neg4, also=[[l_, i] for l_ in (0, 1, 2, 4, 5) for i in range(FFN)]), f"at most {ms.MOTIF_MAX_NEURONS} are tested together"),
    ]
    for kw, said in refusals:
        try:
            asyncio.run(ms.test_motif_hypothesis(**kw))
            check(False, f"{said}: refused")
        except ValueError as exc:
            check(said in str(exc), f"refused in plain words: {exc}")
    check(calls[0] == n_calls, f"every refusal before the model is run ({calls[0] - n_calls} captures)")
    section("the tool on hand-made activations: its selectivity, the keys as first built, refusals; paired, the peak, several neurons", before)

    # ------------------------------------------------------------------------------------------- 3. the HTTP route
    before = len(failures)
    from fastapi.testclient import TestClient

    client = TestClient(app, base_url="http://127.0.0.1")
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": pos4, "neg_prompts": neg4})
    check(r.status_code == 200 and r.json()["selectivity"] == json.loads(json.dumps(replies["north"]["selectivity"]))
          and r.json()["verdict"] == "CONFIRMED_NORTH_POLE (+)", f"the route gives the tool's reply: {r.status_code}")
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": asked["untested"][0], "neg_prompts": asked["untested"][1]})
    check(r.status_code == 200 and all(k in r.json() for k in OLD_KEYS) and r.json()["selectivity"]["verdict"] == "CANNOT_TEST"
          and r.json()["cohens_d"] is None, f"a reading that is not a finite number: a reply, not a 400 (its d as first built sent as null): {r.status_code} {r.text[:160]}")
    for body, said in [({"layer": N_LAYERS, "neuron_idx": N}, "layers count from 0"), ({"layer": 3, "neuron_idx": FFN}, "neurons count from 0"),
                       ({"layer": 3, "neuron_idx": N, "pos_prompts": []}, "pos_prompts is empty"),
                       ({"layer": 3, "neuron_idx": N, "neg_prompts": []}, "neg_prompts is empty")]:
        r = client.post("/api/scientist/test_hypothesis", json={"pos_prompts": pos4, "neg_prompts": neg4, **body})
        check(r.status_code == 400 and said in r.json().get("detail", ""), f"{body}: {r.status_code} {r.text[:160]}")
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": six_asked[0], "neg_prompts": six_asked[1], "paired": True})
    check(r.status_code == 200 and r.json() == json.loads(json.dumps(replies_more["paired"])) and r.json()["design"] == "paired",
          f"the route pairs when asked, as the tool: {r.status_code}")
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": one_of_three[0], "neg_prompts": one_of_three[1], "also": [[3, 7], [2, 9]]})
    check(r.status_code == 200 and r.json() == json.loads(json.dumps(replies_more["three"])) and r.json()["correction"]["tests"] == 3,
          f"the route tests the further neurons too, and corrects them together: {r.status_code}")
    for body, said in [({"paired": True, "neg_prompts": neg4[:3]}, "there are 4 and 3"), ({"also": [[3, N]]}, "named twice"), ({"also": [[3]]}, "[layer, neuron_idx]")]:
        r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": pos4, "neg_prompts": neg4, **body})
        check(r.status_code == 400 and said in r.json().get("detail", ""), f"{body}: {r.status_code} {r.text[:160]}")
    rt.eager_model.capture = real_capture
    live_pos = ["Alice called Bob because", "Write a snake game in Python.", "Describe the color orange.", "Who are you?"]
    live_neg = ["The committee rejected the proposal", "Mix the flour and salt", "The mitochondria produce ATP", "The quick brown fox"]
    r = client.post("/api/scientist/test_hypothesis", json={"layer": 3, "neuron_idx": N, "pos_prompts": live_pos, "neg_prompts": live_neg})
    live = r.json() if r.status_code == 200 else {}
    sel = live.get("selectivity") or {}
    rounded = [x["last_token_signed_act"] for x in live.get("pos_prompt_results", [])]
    check(r.status_code == 200 and sel.get("n_pos") == 4 and sel.get("n_neg") == 4 and sel.get("p_method") == "exact" and rounded
          and abs(sel["mean_pos"] - sum(rounded) / len(rounded)) <= 1e-4 and sel["verdict"] in ("SELECTIVE_NORTH_POLE (+)", "SELECTIVE_SOUTH_POLE (-)",
          "DIFFERS_NOT_AS_GUESSED", "NOT_SHOWN"), f"on the stand-in's own activations: {sel.get('words')!r}")
    peak, either = live.get("selectivity_peak") or {}, live.get("selectivity_either") or {}
    peaks = [x["peak_signed_act"] for x in live.get("pos_prompt_results", [])]
    check(peak.get("p_method") == "exact" and peaks and abs(peak["mean_pos"] - sum(peaks) / len(peaks)) <= 1e-4 and all(abs(v) >= abs(w) for v, w in zip(peaks, rounded))
          and either.get("p") is not None and either["p"] >= min(sel["p"], peak["p"]) - 1e-9 and 0 <= peak.get("at_start", -1) <= 8 and live.get("correction", {}).get("tests") == 1,
          f"and its peak, on the stand-in's own activations: {peak.get('words')!r}; {either.get('words')!r}")
    client.close()
    section("the HTTP route: the tool's reply, refusals as 400s, the stand-in's own activations", before)

    # --------------------------------------------------------------------------------------------- 4. the folio's words
    before = len(failures)
    older = {k: v for k, v in replies["one"].items() if k not in NEW_KEYS}  # as a server from before any test against chance sends it
    shown = run_replies({**{k: json.loads(json.dumps(v)) for k, v in replies.items()}, "older": json.loads(json.dumps(older)),
                         **{f"more {k}": json.loads(json.dumps(v)) for k, v in replies_more.items()}})

    def as_folio(block: Dict[str, Any]) -> str:
        """A block's words as the folio writes them: the faces named, and without the lead that says where it was read
        (the folio says that beside it); a test that could not be run keeps the server's words whole."""
        words = block["words"].replace("its + face", "its positive face").replace("its - face", "its negative face")
        return words if block["verdict"] == "CANNOT_TEST" else re.sub(r"^(at each sentence's strongest word-piece|last word-piece or strongest, whichever[^:]*): ", "", words)

    if shown is not None:
        for k, rep_ in [*replies.items(), *((f"more {k_}", v) for k_, v in replies_more.items())]:
            check(shown[k]["peak"] == as_folio(rep_["selectivity_peak"]) and shown[k]["either"] == as_folio(rep_["selectivity_either"]) and shown[k]["design"] == rep_["design"],
                  f"{k}: at the loudest word the folio says {shown[k]['peak']!r}, whichever is stronger {shown[k]['either']!r}")
        for k in ("paired", "paired few", "paired three"):
            check(shown[f"more {k}"]["words"] == as_folio(replies_more[k]["selectivity_paired"]) and shown[f"more {k}"]["design"] == "paired",
                  f"{k}: sent as matched pairs, the folio's answer is the paired test's: {shown[f'more {k}']['words']!r}")
        check(shown["more paired"]["words"] == "selective on its positive face, p = 0.031, n = 6 pairs" and shown["more paired"]["how"] == ", over all 64 sign flippings"
              and shown["more paired few"]["words"] == "too few to judge: n = 4 pairs, and the paired test needs at least 6 pairs" and shown["north"]["how"] == ", over all 70 splits"
              and shown["eight"]["how"] == ", from 10000 shuffles (seed 0)", f"the paired words, and how each p was counted: {shown['more paired']['words']!r}{shown['more paired']['how']}")
        check(shown["more mid"]["words"] == "no difference shown: p = 1, n = 4 a side" and shown["more mid"]["peak"] == "selective on its positive face, p = 0.029, n = 4 a side"
              and shown["more mid"]["either"] == "no difference shown: p = 0.057, n = 4 a side" and shown["more mid"]["stronger"] == "the loudest word"
              and shown["north"]["stronger"] == "the last word" and shown["more mid"]["atStart"] == 0 and shown["tie"]["atStart"] == 2,
              f"a gap at a middle word: {shown['more mid']['words']!r}; at its loudest word {shown['more mid']['peak']!r}; whichever is stronger {shown['more mid']['either']!r}")
        check(shown["older"]["peak"] is None and shown["older"]["either"] is None and shown["older"]["design"] is None and shown["older"]["tests"] is None
              and shown["north"]["tests"] == 1 and shown["more three"]["tests"] == 3, "an older server's reply has no peak test, no design and no word on correction; a newer one says how many neurons it tested")
        check("corrected for no other" in shown["alone"] and "by chance" in shown["alone"], f"what the folio says of one neuron's p: {shown['alone']!r}")
        for k in replies:
            server = replies[k]["selectivity"]["words"]
            want = server.replace("its + face", "its positive face").replace("its - face", "its negative face")
            check(shown[k]["words"] == want, f"{k}: the folio says {shown[k]['words']!r}, the server {server!r}")
        check(shown["north"]["words"] == "selective on its positive face, p = 0.029, n = 4 a side", f"the folio's words: {shown['north']['words']!r}")
        check(shown["one"]["words"].startswith("too few to judge") and shown["one"]["test"]["p"] is None, f"one a side: {shown['one']['words']!r}")
        check(shown["tie"]["words"] == "no difference shown: p = 0.12, n = 5 and 7" and shown["tie"]["test"]["pWords"] == "0.12",
              f"a tie at two figures: the folio shows the server's p, not its own rounding of 0.125 (\"0.13\"): {shown['tie']['words']!r}")
        check(shown["older"]["test"] is None and shown["older"]["words"].startswith("d at 0.8 or more on its positive face")
              and "nothing set against chance" in shown["older"]["words"], f"an older server's CONFIRMED: {shown['older']['words']!r}")
        words = [w for k, v in shown.items() if k not in ("old", "p", "alone") for w in (v["words"], v["peak"] or "", v["either"] or "")] + shown["old"]
        check(not any("clearly" in w or w.startswith("yes") for w in words), f"no verdict is read as a yes: {words}")
        said = [ms._p_words(p) for p in P_SHOWN]
        check(shown["p"] == said and said[:4] == ["0.0496", "0.04999", "0.05", "0.051"],
              f"a p written alike by the server and the folio, never rounded up to 0.05 from below: {said} {shown['p']}")
    folio = (ROOT_DIR / "frontend" / "src" / "screens" / "NeuronFolio.tsx").read_text(encoding="utf-8")
    replies_ts = (FOLIO / "replies.ts").read_text(encoding="utf-8")
    check("{hypothesisWords(res)}" in folio and "verdictWords(res.verdict)" not in folio and "calls the gap clear" not in folio
          and "there is no null, interval or permutation" not in folio, "the folio shows the test's words, and no longer says it has no test")
    check("yes, clearly" not in replies_ts, "replies.ts: no “yes, clearly”")
    check("<i>{selectivityWords(peakSel)}</i>" in folio and "{selectivityWords(eitherSel)}" in folio and "{ALONE_WORDS}" in folio and "judged(res)" in folio
          and "the peak is reported but not tested" not in folio and "correct their p’s together" not in folio,
          "the folio shows the peak's test, the either test and what one neuron's p is not corrected for, and no longer says the peak is untested")
    section("the folio's words: the test's, an older server's hedged, never “yes, clearly”; paired, the peak, whichever is stronger", before)

    print(f"\n{passed} checks passed, {len(failures)} failed")
    return 1 if failures else 0


P_SHOWN = [0.0496, 0.04999, 0.05, 0.051, 0.0285714, 1.0, 0.31, 0.00309969, 1 / 10_001]
# five against seven whose exact p is 99 of the 792 splits, 0.125: a tie at two figures, which Python rounds to even
# ("0.12") and toPrecision up ("0.13")
TIE = ([6.0, 6.0, 5.0, 0.0, 8.0], [4.0, 3.0, 0.0, 1.0, 3.0, 1.0, 5.0])
ENTRY = r"""
import { ALONE_WORDS, howCounted, hypothesisWords, judged, parseHypothesis, pText, selectivityWords, strongerWords, verdictWords } from "./replies";
const REPLIES = __REPLIES__;
const out = {};
for (const [k, v] of Object.entries(REPLIES)) {
  const h = parseHypothesis(v);
  const t = judged(h);
  out[k] = {
    words: hypothesisWords(h), test: h.test, d: h.d, design: h.design, how: t ? howCounted(t) : null,
    peak: h.peak ? selectivityWords(h.peak) : null, atStart: h.peak ? h.peak.atStart : null,
    either: h.either ? selectivityWords(h.either) : null, stronger: h.either ? strongerWords(h.either) : null,
    tests: h.correction ? h.correction.tests : null,
  };
}
out.alone = ALONE_WORDS;
out.old = ["CONFIRMED_NORTH_POLE (+)", "CONFIRMED_SOUTH_POLE (-)", "WEAK_SELECTIVITY", "REJECTED_OR_INCONCLUSIVE"].map(verdictWords);
out.p = __P__.map((p) => pText(p));
console.log(JSON.stringify(out));
"""


def run_replies(replies: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The folio's reading of each reply: replies.ts bundled with the frontend's esbuild and run under node; None (and
    a failure) when it cannot run."""
    node = shutil.which("node")
    check(ESBUILD.exists(), f"{ESBUILD} is missing: run npm install in frontend/")
    check(node is not None, "node is not on PATH")
    if not ESBUILD.exists() or node is None:
        return None
    entry = ENTRY.replace("__REPLIES__", json.dumps(replies)).replace("__P__", json.dumps(P_SHOWN))
    bundle = subprocess.run(
        [str(ESBUILD), "--bundle", "--platform=node", "--format=esm", "--loader=ts", "--log-level=error",
         "--sourcefile=check_hypothesis_test_entry.ts"],
        input=entry, cwd=FOLIO, capture_output=True, text=True, timeout=120, check=False,
    )
    check(bundle.returncode == 0, f"esbuild could not bundle replies.ts:\n{bundle.stderr}")
    if bundle.returncode != 0:
        return None
    ran = subprocess.run([node, "--input-type=module"], input=bundle.stdout, capture_output=True, text=True, timeout=120, check=False)
    check(ran.returncode == 0, f"the bundled replies.ts did not run under node:\n{ran.stderr}")
    if ran.returncode != 0:
        return None
    return json.loads(ran.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    sys.exit(main())
