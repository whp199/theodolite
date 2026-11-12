#!/usr/bin/env python3
"""Bring the board file up to the trials read so far (the owner's word, 12 November 2026: "open every layer that
passes"). A layer the board still shows waiting ("trial first") whose trial is read is opened if it passed
(census_board.py layer N open) and marked not taken if it failed (layer N stopped); a layer in any other state is left
alone. Then the verdicts are copied into this folder (collect.py). It changes volunteers/board.json and this folder
only: rendering, committing, pushing and publishing the issue are done after it, by hand.

    .venv/bin/python -I experiments/results/census_neighbour_trials/board_from_verdicts.py [--dry-run]
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = ROOT / "atlas_data" / "census" / "trials" / "neighbours"
dry = "--dry-run" in sys.argv
board = json.loads((ROOT / "volunteers" / "board.json").read_text())
today = time.strftime("%-d %B %Y", time.gmtime())
did = []
for v in sorted(SRC.glob("L*_verdict.json"), key=lambda p: int(re.match(r"L(\d+)_", p.name).group(1))):
    layer = int(re.match(r"L(\d+)_", v.name).group(1))
    state = (board["layers"].get(str(layer)) or {}).get("state")
    if state != "waiting":
        continue
    passed = bool(json.loads(v.read_text()).get("pass"))
    new, note = ("open", f"its trial passed, {today}") if passed else ("stopped", f"its trial failed, {today}: not taken with these doses")
    did.append(f"layer {layer}: trial {'PASSED' if passed else 'FAILED'} -> {new}")
    if not dry:
        got = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "census_board.py"), "layer", str(layer), new, "--note", note],
                             capture_output=True, text=True)
        if got.returncode != 0:
            sys.exit(f"layer {layer}: census_board.py refused: {got.stderr.strip()[:300]}")
print("\n".join(did) if did else "nothing new: every trial read is on the board already")
if did and not dry:
    sys.exit(subprocess.run([sys.executable, "-I", str(HERE / "collect.py")]).returncode)
