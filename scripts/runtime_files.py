"""The instrument's own record of its four runtime files, by sha256: backend/__init__.py, theodolite_math.py,
engine_runtime.py and label_gate.py as they stand (recorded 14 November 2026, measured: the files at commit e6ac64f).

R4 pinned the same four beside its own files (experiments/scripts/R4_common.py's PINNED_UNCHANGED, R4_PREREG.md §0).
That table is R4's record of what R4 ran on and is not edited: R4_00 and R4_02 refuse to start while a file differs
from it, so a rerun of R4 goes from R4's registration commit, 2bdf5ad of 28 October 2026 on the branch r4-r1-lens
(R4_COMMIT: the head R4_02_eval.json and R4_03_analysis.json record; it is in main's history), as R3's go from
it-jpp-refit. The instrument's checks held the runtime to R4's table until engine_runtime.py left R4's sha256: its
loader was changed on purpose (CHANGED_SINCE_R4: so that a 24 GB card can load) by two commits of 12 November 2026 on
a pull request's branch (whp199/theodolite#2), which the merge 6857634 of 13 November 2026 brought into main. None of
those commits' messages speaks of R4's pin (read 14 November 2026). The last commits at which the file has R4's
sha256 are f8157bc on main's own line (the tag census-kit-3) and a0efdb4 on the pull request's.

So the instrument keeps a table of its own. RUNTIME_FILES is the sha256 each file must have; CHANGED_SINCE_R4 names
every file that no longer has R4's, with each commit that changed it, when and why. Four checks read both
(scripts/check_dequant.py §7, check_gguf.py §7, check_gguf_cpu.py §5, check_gguf_fast.py §7): every runtime file has
its recorded sha256, and of the files R4 pins those that differ are exactly the ones named here. A change to a runtime
file is therefore made on purpose and written down here. The change is committed first, so that it has a hash to
name; the commit after it puts the new sha256 in RUNTIME_FILES (sha256sum <file>) and a note under CHANGED_SINCE_R4
whose "from" is the sha256 before and whose "to" is the new one, with that commit, its date and the reason. Between
the two those checks fail, as they do for any change not written down.

What changed in engine_runtime.py, function by function, between R4_COMMIT and this record (measured 14 November 2026):
TheodoliteRuntime.load, the two new functions it calls (text_only_device_map, settle_text_only_device_map) and one
import line. The three functions R4's registration names as what it reads through (dequantize_w4a16_full,
patch_compressed_tensors_for_low_vram, _make_w4a16_linear_forward) have the same text. R4's scripts do not call
load(): experiments/scripts/R3_gpu.py's load_w4a16 loads the model itself and takes only those three functions from
this file (read from the code, 14 November 2026), so by the code R4 ran none of the changed lines but the import
(inferred). Whether R4's numbers would come out the same through today's file was not measured: R4_00 and R4_02
refuse to run while the sha256 differs (a766058's message reports layer 45's census canary byte for byte the
unchanged code's: its author's measurement, and of the census, not of R4's readouts).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent

# R4's registration commit (R4_PREREG.md §0; experiments/results/R4_02_eval.json, meta.git.head): where a rerun of R4
# goes from. At it every file below has the sha256 R4's table gives it (measured 14 November 2026).
R4_COMMIT = "2bdf5adf6b87a55e6d785ce3e0ab003aca35b873"

# what each runtime file's sha256 is (14 November 2026)
RUNTIME_FILES: Dict[str, str] = {
    "backend/__init__.py": "4af5cefc7edd058ef3353b6992f42cf22fa8d3a7cd8e4a8b3ecca376808293c0",
    "backend/theodolite_math.py": "d738dc2ff14d428ad3ba01e7789b301dbff137d3f707e3af04862b504a1479fb",
    "backend/engine_runtime.py": "f1a0b4c3e9bbda9271f5df7766a9a75f5eea0bacb81e8779ffac61e3847033f2",
    "backend/label_gate.py": "83a998b746249b07a52e29b0dd1a06e999206d27a518f2f8271f439b946c136e",
}

# the files that no longer have the sha256 R4's table gives them: each commit that changed one, in order, from R4's
# sha256 to the one above (each sha256 read from the file at that commit, 14 November 2026; each date the commit's own)
CHANGED_SINCE_R4: Dict[str, List[Dict[str, str]]] = {
    "backend/engine_runtime.py": [
        {"commit": "a76605863f7883bc87ed8453d8503c8e460c0b28", "date": "2026-11-12",
         "from": "69eed1b4f1a319bad35dcf604d0e3cdd48aa2b1f4287a9c5e7efeb1c04605256",
         "to": "8ef8ecb4cf085aca8e0d7191d0acc58a1508b3be6ee205f5049a5415a6f5d875",
         "why": "the loader puts the vision tower and its embedding on the CPU when the model loads on a CUDA card "
                "(text_only_device_map, settle_text_only_device_map, and load() calling them): the tower is dropped "
                "right after the load, and on the card it set a 24 GB card's peak"},
        {"commit": "960b9ab209de0a6b5fcb5367f32a9aa96fe08cc9", "date": "2026-11-12",
         "from": "8ef8ecb4cf085aca8e0d7191d0acc58a1508b3be6ee205f5049a5415a6f5d875",
         "to": "f1a0b4c3e9bbda9271f5df7766a9a75f5eea0bacb81e8779ffac61e3847033f2",
         "why": "review fixes to that change: the audio tower and its embedding load on the CPU too when the "
                "checkpoint's config has an audio_config (the pinned checkpoint's is null, so its map is the same), "
                "and load() reads the config first and builds the map once"},
    ],
}


def differ(root: Path = ROOT_DIR, recorded: Optional[Dict[str, str]] = None) -> List[str]:
    """The runtime files under `root` that are missing or whose sha256 is not the recorded one (RUNTIME_FILES, or
    `recorded`), in the record's order: as R4_common.pinned_differ answers for R4's table."""
    recorded = RUNTIME_FILES if recorded is None else recorded
    return [r for r in recorded
            if not (root / r).is_file() or hashlib.sha256((root / r).read_bytes()).hexdigest() != recorded[r]]
