#!/usr/bin/env python3
"""Draw 1,024 fresh pure_4_4 neurons+faces per layer for Path 1 (in_use_lens) at L59, L58, L57.

Strictly excludes:
  - Probe 1 neurons (experiments/results/in_use_kill_pilots/probe1_in_use_causal.json)
  - L57 prior census probe/calib (atlas_data/census/trials/L57-break_rel-probe.meta.json,
    atlas_data/census/runs/L57-calib_rel.meta.json, runs/L57-calib_ceil.meta.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import atlas_index as ai  # noqa: E402
from backend import census as cz  # noqa: E402
from scripts.in_use_common import layer_face_counts_vectorized, load_passages_and_doc_halves  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
ATLAS_CENSUS = Path("/home/eric/bin/ai/theodolite2/atlas_data/census")
PROBE1 = REPO_ROOT / "experiments/results/in_use_kill_pilots/probe1_in_use_causal.json"


def _load_exclusions(layer: int) -> Set[int]:
    excl: Set[int] = set()
    if PROBE1.is_file():
        p1 = json.loads(PROBE1.read_text(encoding="utf-8"))
        for tag in (p1.get("neuron_ids") or {}).get(str(layer), []):
            import re
            m = re.match(r"^n(\d+)[pm]$", str(tag))
            if m:
                excl.add(int(m.group(1)))
    if layer == 57:
        for rel in (
            "trials/L57-break_rel-probe.meta.json",
            "runs/L57-calib_rel.meta.json",
            "runs/L57-calib_ceil.meta.json",
        ):
            mp = ATLAS_CENSUS / rel
            if mp.is_file():
                m = json.loads(mp.read_text(encoding="utf-8"))
                excl |= set(cz.parse_neurons(m["neuron_ids"], int(m["d_ff"])))
    return excl


def main() -> None:
    loaded = load_passages_and_doc_halves(ai.atlas_dir())
    P, act_scale = loaded["P"], loaded["act_scale"]
    windows, half_id = loaded["windows"], loaded["half_id"]
    pure_h0, pure_h1 = loaded["pure_h0_mask"], loaded["pure_h1_mask"]
    bos_id, eos_id = loaded["bos_id"], loaded["eos_id"]

    for l in (59, 58, 57):
        excl = _load_exclusions(l)
        cnt = layer_face_counts_vectorized(
            P, windows, act_scale, half_id, pure_h0, pure_h1, l, eos_id=eos_id, bos_id=bos_id
        )
        N_ff = int(cnt["neu_pure44"].shape[0])
        cands = []
        for n in range(N_ff):
            if n in excl or not bool(cnt["neu_pure44"][n]):
                continue
            opts = []
            for fc in ("pos", "neg"):
                if bool(cnt[f"{fc}_pure44"][n]):
                    tot_ev = int(cnt[f"{fc}_pure0"][n]) + int(cnt[f"{fc}_pure1"][n])
                    ext = float(abs(act_scale[fc][l, n]))
                    opts.append((tot_ev, ext, 1 if fc == "pos" else 0, fc))
            opts.sort(reverse=True)
            cands.append((n, opts[0][3]))

        seed = 202611260100 + l
        assert len(cands) >= 1024, f"L{l}: only {len(cands)} candidates"
        rng = np.random.default_rng(seed)
        picks = sorted(int(i) for i in rng.choice(len(cands), size=1024, replace=False))
        lines = [f"{cands[i][0]},{cands[i][1]}" for i in picks]
        out_p = OUT_DIR / f"neurons_L{l}.txt"
        out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"L{l}: excluded={len(excl)} pool={len(cands)} drew=1024 seed={seed} -> {out_p.name}")


if __name__ == "__main__":
    main()
