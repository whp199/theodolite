#!/usr/bin/env python3
"""Neurons for a third trial at a layer whose first two trials have been read: 1,022 of the layer's 21,504 that neither
earlier trial asked (their lists are given), drawn without replacement by a seed fixed here (20261112 * 100 + the layer
+ 7000), sorted. Prints them comma-joined, as census_run.py --neurons takes them.

    python draw_third.py 52 neurons_L52.txt neurons_L52_fresh.txt > neurons_L52_third.txt
"""
import sys

import numpy as np

D_FF, N = 21504, 1022
layer = int(sys.argv[1])
asked = set()
for path in sys.argv[2:]:
    asked |= {int(x) for x in open(path).read().replace("\n", ",").split(",") if x.strip()}
rest = np.array(sorted(set(range(D_FF)) - asked))
ids = np.sort(np.random.default_rng(20261112 * 100 + layer + 7000).choice(rest, N, replace=False))
print(",".join(str(int(i)) for i in ids))
