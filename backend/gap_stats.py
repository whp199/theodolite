"""Pure-NumPy non-parametric rank statistics (replaces scipy.stats in census utilities)."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def _mid_ranks_and_ties(values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return 1-based mid-ranks for ``values`` and the tie-correction sum ``sum(t^3 - t)``."""
    order = np.argsort(values, kind="mergesort")
    svals = values[order]
    diff_mask = np.concatenate(([True], svals[1:] != svals[:-1], [True]))
    bounds = np.flatnonzero(diff_mask)
    starts = bounds[:-1]
    ends = bounds[1:]
    cnts = (ends - starts).astype(np.float64)
    mid = 0.5 * (starts + ends - 1).astype(np.float64) + 1.0
    sranks = np.repeat(mid, ends - starts)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = sranks
    tie_sum = float(np.sum(cnts * cnts * cnts - cnts))
    return ranks, tie_sum


def mannwhitney_greater(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Tie-aware Mann-Whitney AUC ``P(A > B)`` and asymptotic one-sided greater p-value."""
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    n1, n2 = int(a_arr.size), int(b_arr.size)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    ranks, tie_sum = _mid_ranks_and_ties(np.concatenate([a_arr, b_arr]))
    u = float(np.sum(ranks[:n1]) - 0.5 * n1 * (n1 + 1))
    auc = u / (n1 * n2)
    tot = n1 + n2
    var_u = (n1 * n2 / 12.0) * (tot + 1.0 - tie_sum / (tot * (tot - 1.0))) if tot > 1 else 0.0
    if var_u <= 0.0:
        return auc, 1.0
    z = (u - 0.5 * n1 * n2 - 0.5) / math.sqrt(var_u)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return auc, min(1.0, max(0.0, p))


def wilcoxon_greater(a: Sequence[float], b: Sequence[float]) -> float:
    """Asymptotic one-sided Wilcoxon signed-rank p-value (``zero_method='wilcox'``, ``alternative='greater'``)."""
    diff = np.asarray(a, dtype=np.float64).ravel() - np.asarray(b, dtype=np.float64).ravel()
    nz = diff[diff != 0.0]
    n = int(nz.size)
    if n == 0:
        return 1.0
    ranks, tie_sum = _mid_ranks_and_ties(np.abs(nz))
    r_plus = float(np.sum(ranks[nz > 0.0]))
    mean_w = 0.25 * n * (n + 1.0)
    var_w = (n * (n + 1.0) * (2.0 * n + 1.0) - 0.5 * tie_sum) / 24.0
    if var_w <= 0.0:
        return 1.0
    z = (r_plus - mean_w - 0.5) / math.sqrt(var_w)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(1.0, max(0.0, p))
