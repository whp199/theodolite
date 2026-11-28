#!/usr/bin/env python3
"""Draw 1,022 fresh neurons per layer for Path 2 extension (u2_rollout_L58_L59: L58, L59).

Strictly excludes:
  - L58: probe1_in_use_causal.json L58 AND Path 1's neurons_L58.txt
  - L59: probe1_in_use_causal.json L59 AND Path 1's neurons_L59.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Set

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path(__file__).resolve().parent
D_FF, N_DRAW = 21504, 1022


def _load_exclusions(layer: int) -> Set[int]:
    excl: Set[int] = set()
    p1 = REPO_ROOT / "experiments/results/in_use_kill_pilots/probe1_in_use_causal.json"
    if p1.is_file():
        for tag in (json.loads(p1.read_text(encoding="utf-8")).get("neuron_ids") or {}).get(str(layer), []):
            m = re.match(r"^n(\d+)[pm]$", str(tag))
            if m:
                excl.add(int(m.group(1)))
    p1_txt = REPO_ROOT / f"experiments/results/in_use_lens_trial/prereg/neurons_L{layer}.txt"
    if not p1_txt.is_file():
        raise FileNotFoundError(f"Missing required exclusion list: {p1_txt}")
    for ln in p1_txt.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            excl.add(int(ln.split(",")[0]))
    return excl


def main() -> None:
    for l in (58, 59):
        excl = _load_exclusions(l)
        pool = np.array(sorted(set(range(D_FF)) - excl), dtype=np.int64)
        seed = 202611270200 + l
        ids = np.sort(np.random.default_rng(seed).choice(pool, N_DRAW, replace=False))
        assert not (set(int(x) for x in ids) & excl), f"L{l}: exclusion overlap detected!"
        out_p = OUT_DIR / f"neurons_L{l}.txt"
        out_p.write_text(",".join(str(int(x)) for x in ids) + "\n", encoding="utf-8")
        print(f"L{l}: excluded={len(excl)} pool={len(pool)} drew={N_DRAW} seed={seed} -> {out_p.name}")


if __name__ == "__main__":
    main()
