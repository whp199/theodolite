#!/usr/bin/env bash
# The registered measurement (RULES.md beside this): kit 4's layers 32-39, 51 and 52, each with the kit's design, on
# the home card with nothing else on it. Refuses if this folder's files, or the script it runs, are not the registered
# bytes, or if something answers on :8765. Writes ../bars.json and ../bars.txt; the rows go under the atlas folder.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
cd "$ROOT"
( cd "$HERE" && sha256sum --quiet -c SHA256SUMS )
cmp -s "$HERE/census_bars.py" scripts/census_bars.py || { echo "scripts/census_bars.py is not the registered file" >&2; exit 2; }
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then echo "something answers on :8765: one GPU process at a time" >&2; exit 5; fi
export THEODOLITE_ATLAS_DIR="${THEODOLITE_ATLAS_DIR:-$ROOT/atlas_data}"
exec "${PY:-$ROOT/.venv/bin/python}" -I scripts/census_bars.py --layers 32,33,34,35,36,37,38,39,51,52
