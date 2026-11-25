#!/usr/bin/env python3
"""Draw 1,022 fresh, non-overlapping neurons per layer (L55-L59) for pre-registered forced_odd trials.

Deterministic seed: 20261124 * 100 + layer.
Strictly excludes:
  1. Every non-volunteer census meta under `atlas_dir()/census`
     at that layer (verifying ASKED matches disk exactly, else refusing to draw).
  2. All 64 dose-ladder neurons in experiments/results/census_forced_odd/dose_ladder.json.
  3. At L59, all 1,022 exploratory L59_pilot neurons (rng(11)).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

WORKTREE = Path(os.environ.get("THEODOLITE_TRIAL_REPO") or Path(__file__).resolve().parents[4])
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from backend import atlas_index as ai  # noqa: E402
from backend import census as cz  # noqa: E402

ATLAS = ai.atlas_dir() / "census"
LADDER = WORKTREE / "experiments/results/census_forced_odd/dose_ladder.json"
LAYERS = (55, 56, 57, 58, 59)
ASKED = {
    55: ["trials/things_walk/L55-match_things-trial.meta.json"],
    56: [],
    57: [
        "runs/L57-calib_ceil.meta.json",
        "runs/L57-calib_rel.meta.json",
        "trials/L57-break_rel-probe.meta.json",
    ],
    58: [],
    59: [],
}
D_FF, N = 21504, 1022


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <layer>")
    layer = int(sys.argv[1])
    if layer not in LAYERS:
        raise SystemExit(f"layer {layer}: registration covers {', '.join(map(str, LAYERS))} only")

    named = {ATLAS / p for p in ASKED[layer]}
    found = set()
    for p in ATLAS.rglob("*.meta.json"):
        if "volunteer" in p.relative_to(ATLAS).parts:
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(m, dict) and m.get("layer") is not None and int(m["layer"]) == layer and m.get("neuron_ids"):
            found.add(p)
    if found != named:
        raise SystemExit(
            f"layer {layer}: found census metas ({sorted(map(str, found))}) != expected ASKED ({sorted(map(str, named))}); aborting"
        )

    asked = set()
    for p in sorted(named):
        m = json.loads(p.read_text(encoding="utf-8"))
        assert int(m["layer"]) == layer and int(m["d_ff"]) == D_FF, p
        ids_p = set(cz.parse_neurons(m["neuron_ids"], D_FF))
        asked |= ids_p
        print(
            f"left out census meta: {p.relative_to(ATLAS)} ({len(ids_p)} neurons), sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}",
            file=sys.stderr,
        )

    ladder_data = json.loads(LADDER.read_text(encoding="utf-8"))
    by_l = {int(e["layer"]): e for e in ladder_data["layers"]}
    if layer not in by_l or not by_l[layer].get("neuron_ids"):
        raise SystemExit(f"layer {layer}: missing ladder neuron_ids in {LADDER}")
    lad_ids = {int(x) for x in by_l[layer]["neuron_ids"]}
    asked |= lad_ids
    print(f"left out dose_ladder: {len(lad_ids)} neurons, sha256 {hashlib.sha256(LADDER.read_bytes()).hexdigest()[:16]}", file=sys.stderr)

    if layer == 59:
        pilot59 = {int(x) for x in np.random.default_rng(11).choice(D_FF, 1022, replace=False)}
        asked |= pilot59
        print(f"left out L59_pilot (rng 11): {len(pilot59)} neurons", file=sys.stderr)

    seed = 20261124 * 100 + layer
    pool = np.array(sorted(set(range(D_FF)) - asked), dtype=np.int64)
    ids = np.sort(np.random.default_rng(seed).choice(pool, N, replace=False))
    assert len(set(ids.tolist()) & asked) == 0, "Overlap with excluded neurons!"
    print(
        f"layer {layer}: seed={seed}; total excluded={len(asked)}, pool={len(pool)}, drawn={len(ids)}",
        file=sys.stderr,
    )
    print(",".join(str(int(i)) for i in ids))


if __name__ == "__main__":
    main()
