#!/usr/bin/env python3
"""Fresh neurons for a second trial at a layer whose first trial has been read: 1,022 of the layer's 21,504 that its
first trial did not ask (neurons_L<NN>.txt of the neighbour trials), drawn without replacement by a seed fixed here
(20261112 * 100 + the layer + 5000), sorted. Prints them comma-joined, as census_run.py --neurons takes them.

    python draw_fresh.py 52 ../../neighbours/prereg/neurons_L52.txt > neurons_L52_fresh.txt
"""
import sys

import numpy as np

D_FF, N = 21504, 1022
layer = int(sys.argv[1])
asked = {int(x) for x in open(sys.argv[2]).read().replace("\n", ",").split(",") if x.strip()}
rest = np.array(sorted(set(range(D_FF)) - asked))
ids = np.sort(np.random.default_rng(20261112 * 100 + layer + 5000).choice(rest, N, replace=False))
print(",".join(str(int(i)) for i in ids))
