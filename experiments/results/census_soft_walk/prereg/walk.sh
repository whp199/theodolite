#!/usr/bin/env bash
# The walk downward with soft_ceil (RULES.md): layer 37, then 36, and so on to 30, a trial at each (night.sh), read as
# soon as it is written (read.sh), and the next layer down taken only if this one passed all three gates. It stops at
# the first layer that fails and at layer 30 at the latest. A layer already written or read is not taken again, so the
# same command goes on from where it stopped. Nothing here commits, pushes, opens a layer or counts a dose calibrated.
#     bash walk.sh
set -uo pipefail
D=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/soft_walk
P=$D/prereg
PY="/home/eric/bin/ai/theodolite2/.venv/bin/python -I"
WALK="37 36 35 34 33 32 31 30"
verdict() {  # pass | fail | none, for a layer
    [ -f "$D/L$1_verdict.json" ] || { echo none; return; }
    $PY -c 'import json, sys; print("pass" if json.load(open(sys.argv[1])).get("pass") is True else "fail")' "$D/L$1_verdict.json"
}
for L in $WALK; do
    v=$(verdict "$L")
    if [ "$v" = none ]; then
        echo "== $(date -u +%FT%TZ) layer $L: the trial"
        LAYERS=$L bash "$P/night.sh" || { echo "layer $L: the trial stopped: once more"; sleep 30; LAYERS=$L bash "$P/night.sh"; } \
            || { echo "layer $L: the trial stopped twice (run this again to continue it)"; exit 6; }
        LAYERS=$L bash "$P/read.sh"
        v=$(verdict "$L")
        [ "$v" = none ] && { echo "layer $L: no verdict was written"; exit 7; }
    fi
    echo "== $(date -u +%FT%TZ) layer $L: $v"
    if [ "$v" != pass ]; then echo "the walk stops at layer $L, which did not pass"; echo "$L" > "$D/walk.stopped_at"; break; fi
done
date -u +%FT%TZ > "$D/walk.done"
