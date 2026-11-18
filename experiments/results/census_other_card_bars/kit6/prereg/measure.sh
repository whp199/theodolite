#!/usr/bin/env bash
# The registered measurement (RULES.md beside this): layer 30 under kit 6 (match_things), on the home card with nothing
# else on it. First the unpushed answers and their near-tie bar (census_ties.py), then the log-probability bars on the
# kit run's chunk 0 (census_bars.py --chunk0). Refuses if this folder's files, or the scripts it runs, are not the
# registered bytes, if the kit is not kit 6, or if something answers on :8765. Writes
# experiments/results/census_unpushed_ties/kit6/ and ../bars.json, ../bars.txt; the rows go under the atlas folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
cd "$ROOT"
( cd "$HERE" && sha256sum --quiet -c SHA256SUMS )
cmp -s "$HERE/census_bars.py" scripts/census_bars.py || { echo "scripts/census_bars.py is not the registered file" >&2; exit 2; }
cmp -s "$HERE/census_ties.py" scripts/census_ties.py || { echo "scripts/census_ties.py is not the registered file" >&2; exit 2; }
grep -qx 'KIT=6' volunteers/kit.env || { echo "volunteers/kit.env is not kit 6" >&2; exit 3; }
test -s volunteers/home_canaries/L30-match_things.canary.jsonl || { echo "no home canary of layer 30 under match_things" >&2; exit 4; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then echo "something answers on :8765: one GPU process at a time" >&2; exit 5; fi
export THEODOLITE_ATLAS_DIR="${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}"
PY="${PY:-$ROOT/.venv/bin/python}"
REF="$THEODOLITE_ATLAS_DIR/census/trials/things/L30-match_things-trial.jsonl"
echo "$(cat "$HERE/reference.sha256")  $REF" | sha256sum --quiet -c - || { echo "the trial run at 30 is not the registered file" >&2; exit 6; }
"$PY" -I scripts/census_ties.py --layer 30 --reference "$REF" --out experiments/results/census_unpushed_ties/kit6
exec "$PY" -I scripts/census_bars.py --layers 30 --chunk0
