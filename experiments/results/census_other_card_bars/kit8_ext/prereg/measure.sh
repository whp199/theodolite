#!/usr/bin/env bash
# The registered measurement (RULES.md beside this): layers 58 and 59 under kit 8 (match_things), on the home card with
# nothing else on it. First the unpushed answers and their near-tie bar at each layer (census_ties.py), then the
# log-probability bars on both layers' chunk 0, pooled (census_bars.py --chunk0 --out ...kit8_ext). Refuses if this
# folder's files, or the scripts it runs, are not the registered bytes, if the kit is not kit 8, if a home canary is
# missing, if the u2_rollout_L58_L59 trial runs are not the registered files, or if something answers on :8765. Writes
# experiments/results/census_unpushed_ties/kit8_ext/L58/, .../L59/ and ../bars.json, ../bars.txt without touching
# kit8/; the rows go under the atlas folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
cd "$ROOT"
( cd "$HERE" && sha256sum --quiet -c SHA256SUMS )
cmp -s "$HERE/census_bars.py" scripts/census_bars.py || { echo "scripts/census_bars.py is not the registered file" >&2; exit 2; }
cmp -s "$HERE/census_ties.py" scripts/census_ties.py || { echo "scripts/census_ties.py is not the registered file" >&2; exit 2; }
grep -qx 'KIT=8' volunteers/kit.env || { echo "volunteers/kit.env is not kit 8" >&2; exit 3; }
for L in 58 59; do
    test -s "volunteers/home_canaries/L$L-match_things.canary.jsonl" || { echo "no home canary of layer $L under match_things" >&2; exit 4; }
done
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then echo "something answers on :8765: one GPU process at a time" >&2; exit 5; fi
export THEODOLITE_ATLAS_DIR="${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}"
PY="${PY:-$ROOT/.venv/bin/python}"
WALK="$THEODOLITE_ATLAS_DIR/census/trials/u2_rollout_L58_L59"
( cd "$WALK" && sha256sum --quiet -c "$HERE/reference.sha256" ) || { echo "the u2_rollout_L58_L59 trial runs at 58 and 59 are not the registered files" >&2; exit 6; }
for L in 58 59; do
    "$PY" -I scripts/census_ties.py --layer "$L" --reference "$WALK/L$L-match_things-u2-cols.jsonl" \
        --out "experiments/results/census_unpushed_ties/kit8_ext/L$L"
done
exec "$PY" -I scripts/census_bars.py --layers 58,59 --chunk0 --out experiments/results/census_other_card_bars/kit8_ext
