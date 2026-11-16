#!/usr/bin/env bash
# The trial (RULES.md): written (night.sh, once more if it stops), then read (read.sh). Nothing here commits, pushes,
# opens a layer or counts a dose calibrated.
#     bash trial.sh
set -uo pipefail
P=/home/eric/bin/ai/theodolite2/atlas_data/census/trials/both/prereg
echo "== $(date -u +%FT%TZ) layer 30: the trial"
bash "$P/night.sh" || { echo "the trial stopped: once more"; sleep 30; bash "$P/night.sh"; } \
    || { echo "the trial stopped twice (run this again to continue it)"; exit 6; }
bash "$P/read.sh"
