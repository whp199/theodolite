#!/usr/bin/env python3
"""Draw 1,022 fresh neurons per layer for Path 2 (u2_rollout upward walk: L55, L56, L57).

Strictly excludes:
  - L55: trials/things_walk/L55-match_things-trial.meta.json + probe1_in_use_causal.json L55
  - L56: pattern_tests_pilot/patterns.json L56
  - L57: trials/L57-break_rel-probe.meta.json, runs/L57-calib_rel.meta.json,
         runs/L57-calib_ceil.meta.json, AND Path 1's neurons_L57.txt!
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

from backend import census as cz  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
ATLAS = Path("/home/eric/bin/ai/theodolite2/atlas_data/census")
D_FF, N_DRAW = 21504, 1022


def _load_exclusions(layer: int) -> Set[int]:
    excl: Set[int] = set()
    p1 = REPO_ROOT / "experiments/results/in_use_kill_pilots/probe1_in_use_causal.json"
    if p1.is_file():
        for tag in (json.loads(p1.read_text(encoding="utf-8")).get("neuron_ids") or {}).get(str(layer), []):
            m = re.match(r"^n(\d+)[pm]$", str(tag))
            if m:
                excl.add(int(m.group(1)))
    if layer == 55:
        mp = ATLAS / "trials/things_walk/L55-match_things-trial.meta.json"
        if mp.is_file():
            m = json.loads(mp.read_text(encoding="utf-8"))
            excl |= set(cz.parse_neurons(m["neuron_ids"], D_FF))
    elif layer == 56:
        pat = REPO_ROOT / "experiments/results/pattern_tests_pilot/patterns.json"
        if pat.is_file():
            for v in json.loads(pat.read_text(encoding="utf-8")).get("faces", {}).values():
                if int(v["layer"]) == 56:
                    excl.add(int(v["n"]))
    elif layer == 57:
        for rel in (
            "trials/L57-break_rel-probe.meta.json",
            "runs/L57-calib_rel.meta.json",
            "runs/L57-calib_ceil.meta.json",
        ):
            mp = ATLAS / rel
            if mp.is_file():
                m = json.loads(mp.read_text(encoding="utf-8"))
                excl |= set(cz.parse_neurons(m["neuron_ids"], D_FF))
        p1_txt = REPO_ROOT / "experiments/results/in_use_lens_trial/prereg/neurons_L57.txt"
        if p1_txt.is_file():
            for ln in p1_txt.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    excl.add(int(ln.split(",")[0]))
    return excl


def main() -> None:
    for l in (55, 56, 57):
        excl = _load_exclusions(l)
        pool = np.array(sorted(set(range(D_FF)) - excl), dtype=np.int64)
        seed = 202611260200 + l
        ids = np.sort(np.random.default_rng(seed).choice(pool, N_DRAW, replace=False))
        out_p = OUT_DIR / f"neurons_L{l}.txt"
        out_p.write_text(",".join(str(int(x)) for x in ids) + "\n", encoding="utf-8")
        print(f"L{l}: excluded={len(excl)} pool={len(pool)} drew={N_DRAW} seed={seed} -> {out_p.name}")


if __name__ == "__main__":
    main()
