#!/usr/bin/env python3
"""The neurons a neighbour trial asks: 1,022 of a layer's 21,504, drawn without replacement by a seed fixed here
(20261112 * 100 + the layer, counted from 0), sorted. No layer here has been asked before (the ladders took 20, 30, 40,
45, 50 and 57), so every one is fresh. Prints them comma-joined, as census_run.py --neurons takes them.

    python draw_neurons.py 39 > neurons_L39.txt
"""
import sys

import numpy as np

D_FF, N = 21504, 1022
layer = int(sys.argv[1])
ids = np.sort(np.random.default_rng(20261112 * 100 + layer).choice(D_FF, N, replace=False))
print(",".join(str(int(i)) for i in ids))
