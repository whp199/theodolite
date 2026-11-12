#!/usr/bin/env bash
# Read a whole layer taken outside 40-50 the way its trial was read (prereg/RULES.md): the three checks N1, N2 and N3
# of prereg/neighbour_read.py, the reader and gates hashed on 12 November 2026, on the CPU. The record is kept in
# experiments/results/census_L<NN>_whole/ beside the other layers', each reading's named_words left out to keep it small.
# About 20 minutes and 6 GB for a whole layer (at nice 15, so a census run beside it keeps its pace).
#
#     bash read_whole.sh 39                # atlas_data/census/runs/L39-break_rel.jsonl
#     RUN=<a run> OUT=<a folder> bash read_whole.sh 39
set -euo pipefail
L=${1:?a layer, counted from 0}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
P=atlas_data/census/trials/neighbours/prereg
R=${RUN:-atlas_data/census/runs/L$L-break_rel.jsonl}
HERE=${OUT:-experiments/results/census_L${L}_whole}
[ -f "$R" ] || { echo "no run at $R"; exit 2; }
mkdir -p "$HERE"
export CUDA_VISIBLE_DEVICES=
(cd "$P" && sha256sum -c --quiet SHA256SUMS) || { echo "$P differs from its SHA256SUMS: see AMENDMENTS.txt there"; exit 2; }
nice -n 15 .venv/bin/python -I "$P/neighbour_read.py" "$R" --out "$HERE/L${L}_verdict.json" > "$HERE/L${L}_verdict.txt" || true
.venv/bin/python -I - "$HERE" "$L" <<'PY'
import json, sys
from pathlib import Path
here, layer = Path(sys.argv[1]), sys.argv[2]
for part in ("whole", "pm1"):
    p = here / f"L{layer}_verdict.{part}.json"
    if p.exists():
        d = json.loads(p.read_text())
        for rd in (d.get("readings") or {}).values():
            rd.pop("named_words", None)
        p.write_text(json.dumps(d, indent=1))
PY
grep -E '^(===|pass  N|FAIL  N)' "$HERE/L${L}_verdict.txt"
