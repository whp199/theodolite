#!/usr/bin/env bash
# The steer lens on the home RTX 5090, after the trials' walk (the owner's word, 12 November 2026): wait for the walk's
# scope to end, take the dose pilot (atlas_build.py steer_dose, about half an hour), build the lens at the pilot's doses
# (atlas_build.py steer, about 10 to 11 hours; a stopped build goes on, layer by layer), then set it against the census
# (against_census.py). Run from a clean checkout of the commit that built the pilot (R), into the main checkout's
# index folder. A layer whose dose fell off the pilot's ladder is left unbuilt and named: its ladder wants widening
# first. Run in its own systemd scope; stopping the scope stops it, and the same command goes on.
set -uo pipefail
MAIN=/home/eric/bin/ai/theodolite2
R=${R:-$MAIN/.claude/worktrees/steer-build}
D=$MAIN/atlas_data/steer_run
PY=$MAIN/.venv/bin/python
export THEODOLITE_ATLAS_DIR=$MAIN/atlas_data
mkdir -p "$D"
while systemctl --user is-active -q census-chain.scope; do sleep 60; done
echo "== $(date -u +%FT%TZ) the walk's scope has ended"
if (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then echo "something answers on :8765: one GPU process at a time"; exit 5; fi
[ -z "$(git -C "$R" --no-optional-locks status --porcelain --untracked-files=no)" ] || { echo "$R has uncommitted changes"; exit 2; }
git -C "$R" rev-parse HEAD > "$D/commit"
cd "$R"
[ -e "$D/pilot.t0" ] || date -u +%s > "$D/pilot.t0"
"$PY" -I scripts/atlas_build.py steer_dose --unpacker compiled 2>&1 | tee -a "$D/pilot.log"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "the pilot stopped: run this again to go on"; exit 6; }
date -u +%s > "$D/pilot.t1"
date -u +%FT%TZ > "$D/pilot.done"
LAYERS=$("$PY" -I - "$THEODOLITE_ATLAS_DIR/index/steer_dose.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(",".join(str(l) for l in sorted(int(k) for k, v in d["layers"].items() if not v.get("note"))))
off = sorted(int(k) for k, v in d["layers"].items() if v.get("note"))
if off or d["missing"]:
    print(f"left unbuilt: off the ladder {off}, not measured {d['missing']}", file=sys.stderr)
PY
)
echo "== $(date -u +%FT%TZ) building layers $LAYERS"
[ -e "$D/build.t0" ] || date -u +%s > "$D/build.t0"
"$PY" -I scripts/atlas_build.py steer --unpacker compiled --layers "$LAYERS" 2>&1 | tee -a "$D/build.log"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "the build stopped: run this again to go on"; exit 8; }
date -u +%s > "$D/build.t1"
echo "== $(date -u +%FT%TZ) the lens against the census"
CUDA_VISIBLE_DEVICES= nice -n 15 "$PY" -I "$MAIN/experiments/results/atlas_steer_dose/against_census.py" 2>&1 | tee "$D/against_census.log"
date -u +%FT%TZ > "$D/done"
