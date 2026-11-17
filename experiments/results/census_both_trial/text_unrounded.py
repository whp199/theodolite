#!/usr/bin/env python3
"""What the text loses at each dose, to four places (the reader rounds it to three). Written after the reading, on 17
November 2026, to settle a forecast at its edge: the registration said the text would lose under 0.005 of its distinct
4-grams at either dose, and the reader gives 0.005 for the neurons at +2.67. Same measure as the reader's
(`census.distinct4` of each pushed answer less that of its question's unpushed answer), every answer, no gate.

    .venv/bin/python -I experiments/results/census_both_trial/text_unrounded.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from backend import census as cz  # noqa: E402

rows = list(cz.read_rows(ROOT / "atlas_data/census/trials/both/L30-match_both-trial.jsonl", tolerate_tail=True))
base = {}
for r in rows:
    if r["kind"] == "baseline":
        base.setdefault(r["prompt"], r)
acc = collections.defaultdict(list)
for r in rows:
    if r["kind"] in ("neuron", "random"):
        acc[(round(r["dose"], 4), r["kind"])].append(cz.distinct4(r["token_ids"]) - cz.distinct4(base[r["prompt"]]["token_ids"]))
for k in sorted(acc):
    print(f"{k[0]:+g} {k[1]:7s} {len(acc[k]):6d} answers  distinct 4-grams less unpushed {np.mean(acc[k]):+.4f}")
