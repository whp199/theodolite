#!/usr/bin/env python3
"""Copy the trials' verdicts that are read into this folder (atlas_data is not in the repository): each layer's
L<NN>_verdict.json and .txt, and its two readings (.whole.json, .pm1.json) with their named_words left out to keep the
record small; and each run's meta, which says how it was taken. Then summary.py's table.

    .venv/bin/python -I experiments/results/census_neighbour_trials/collect.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent.parent / "atlas_data" / "census" / "trials" / "neighbours"
got = []
for v in sorted(SRC.glob("L*_verdict.json"), key=lambda p: int(re.match(r"L(\d+)_", p.name).group(1))):
    layer = re.match(r"L(\d+)_", v.name).group(1)
    shutil.copyfile(v, HERE / v.name)
    shutil.copyfile(v.with_suffix(".txt"), HERE / v.with_suffix(".txt").name)
    for part in ("whole", "pm1"):
        p = SRC / f"L{layer}_verdict.{part}.json"
        d = json.loads(p.read_text())
        for rd in (d.get("readings") or {}).values():
            rd.pop("named_words", None)
        (HERE / p.name).write_text(json.dumps(d, indent=1) + "\n")
    meta = SRC / f"L{layer}-break_rel-trial.meta.json"
    if meta.exists():
        shutil.copyfile(meta, HERE / meta.name)
    got.append(layer)
print("verdicts kept here for layers " + ", ".join(got))
sys.exit(subprocess.run([sys.executable, "-I", str(HERE / "summary.py")]).returncode)
