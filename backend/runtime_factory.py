"""Which runtime a process runs (`backend/runtime_factory.py`): one choice, read from the environment, for the server,
scripts/census_run.py, scripts/atlas_build.py, the standalone scientist (backend/mcp_scientist.py run alone), the lab
daemon (experiments/lab/labd.py) and the GPU checks scripts/verify_optics.py and scripts/verify_causal_check.py.

    THEODOLITE_GGUF=<file.gguf>        a GGUF of Gemma 4 (backend/gguf_runtime.py), else the W4A16 checkpoint as today
    THEODOLITE_DEVICE=cpu | cuda:N     where the model runs (default cuda:0, as today)
    THEODOLITE_GGUF_LINEARS=packed | resident | dense | int4mm
                                                        how a GGUF's quantised layers are held (default packed);
                                                        int4mm (the CPU only) multiplies its Q4_0, Q4_1 and Q4_K
                                                        layers by torch's int4 CPU kernel, not the reference's bits
                                                        (backend/int4_cpu.py)
    THEODOLITE_GGUF_RESIDENT_GB=<GB>                    with packed or int4mm: the layers that would be packed (a K
                                                        file's Q5_K, Q6_K, Q8_0, or IQ4_NL, IQ4_XS, which the kernel
                                                        cannot take) held resident in the model's dtype, in the
                                                        model's order, each that still fits in this many GB (10^9
                                                        bytes); the rest packed. The same bits as packed, so named in
                                                        the weights card, not the model id. Layers gguf-py unpacks at
                                                        load are dense, outside it (the card's `at_load_bytes`)
    THEODOLITE_CHECKPOINT_LINEARS=packed | int4mm       the W4A16 checkpoint's on the CPU: int4mm as for a GGUF
                                                        (int4mm beside THEODOLITE_GGUF is refused)
    THEODOLITE_GGUF_CONFIG=<config.json>                build a GGUF's model from this config, checked against all the
                                                        file states (it stands in for any key the file leaves out)
    THEODOLITE_DTYPE=bfloat16 | float32                 a GGUF model's dtype (default bfloat16, as the instrument runs)
    THEODOLITE_GGUF_LENS=jpp | none                     load the J++ lens beside a GGUF model (default), or leave it
                                                        out (3.4 GB): a census and the atlas's stats, wiring, layout
                                                        and steer builds do not read it; the sightings, the lens
                                                        build and the vault's UPDATE and INSERT do

With none of them set, `make_runtime()` is `TheodoliteRuntime(load_jlens=...)`, exactly as these entry points built it
before. On the CPU (THEODOLITE_DEVICE=cpu) the GPUs are hidden from this process before anything touches CUDA
(CUDA_VISIBLE_DEVICES=""), as the stand-in's scripts hide them. That is a precaution: the pinned runtime asks
`torch.cuda.max_memory_allocated(self.device)` whenever CUDA is available, which torch 2.14.1 answers with 0 while CUDA
is not initialised in the process and refuses for a CPU device once it is (`memory_stats_as_nested_dict`; read from
torch's code, no GPU here to show it); hidden GPUs keep a library from initialising CUDA in a CPU process.

A reading taken off the GPU, or in float32, is its own numeric class, and says so in its model id (`reading_class`):
the checkpoint on the CPU is `google/gemma-4-31B-it-qat-w4a16-ct@cpu/bfloat16` (`CheckpointRuntime`), a GGUF on the CPU
`gguf:<file>#<fingerprint>@cpu/bfloat16`. Every census run, index and lens card records that id, a run resumes only on
the same one and the server reads an index only for it, so readings of two classes never mix in one file. bfloat16
on a GPU, as the instrument has always run, adds nothing. The int4mm linears add `+int4mm` after it
(`gguf:<file>#<fingerprint>@cpu/bfloat16+int4mm`): their sums are the kernel's, so their readings are their own class.
A resident budget adds nothing: a resident layer's forward is the packed one's bit for bit (scripts/check_gguf_cpu.py
§4), and the weights card says how much was held so (`resident_gb`, `resident_bytes`, `layers_as`). Nor does the
unpacker: "compiled" (backend/gguf_fast.py) gives a GGUF's packed layers the reference's bits, and a reading records
it under "unpacker".

`weights_card(rt)` says what a runtime's readings are taken from, for any runtime: a GGUF's own card (its file's
provenance and how its model was built), the checkpoint's repo id, the snapshot of it this machine's Hugging Face cache
holds and the weight files that snapshot links to (`checkpoint_files_id`), or the stand-in's; GET /api/health and the
lab daemon's /health send it.

The compile cache (`private_compile_cache`, which `make_runtime()` calls before it builds anything). The compiled
unpacker (backend/w4a16_fast.py, backend/gguf_fast.py) goes through torch.compile, which keeps what it built on disk
for a later process to load: its graph caches (pickles), each kernel's generated .py, .cpp and .so, Triton's files, and
on the CPU a precompiled header that every kernel compiled includes. Left alone torch keeps them in the temporary
folder (/tmp/torchinductor_<user>), where any process of the account can write, one shut in a sandbox that may still
write /tmp among them: a scientist chat's shell under Codex at the default reach runs in Codex's workspace-write
sandbox (backend/scientist_chat.py), and a review found Codex's own sandbox helper writing /tmp and opening that
folder's files for writing (codex-cli 0.166.0, `codex sandbox -P :workspace`, 14 November 2026; not tried through the
app-server the chat runs, and no file was planted). torch writes a cached file only when none of its name is there, so
a file planted there is loaded as it stands (read from torch's code, not tried). So the cache is kept in
${XDG_CACHE_HOME:-~/.cache}/theodolite/torch_compile, made for its user alone (0700) and outside what that sandbox
was seen to write (in it a file under ~/.cache was refused; this folder itself was not tried), unless the person has set
TORCHINDUCTOR_CACHE_DIR themselves: their folder is then left as torch keeps it, the precompiled header in the
temporary folder included. The new folder starts empty, so the first process to compile after this change (14 November
2026) builds every graph afresh, once. On a GPU that is seconds (inferred, no GPU run was made for this change: the
documents give about 4 s for the checkpoint's graphs at load and 3 to 10 s for a GGUF's on the RTX 5090, COMPUTE §ii
and the README, without saying whether the cache was empty, and the review saw installs of 1.7 to 2.2 s with it
full). On the CPU a process that built a runtime and compiled one small graph took 13.6 s with nothing cached, its
173 MiB header built too, and 4.3 s the next time, some 2.6 s of each its imports (measured here, 14 November 2026). A
process that compiles without building a runtime here is covered only where its unpacker calls
`private_compile_cache` itself: gguf_fast does, before each graph, and w4a16_fast's `install` does before it
compiles (14 November 2026); the checkpoint's compiled unpacker called with no install before it, as
scripts/check_dequant.py calls it for its trial compile and its first comparisons, still writes to the temporary
folder, as does a volunteer's kit, which runs its own tagged copy of this file. At the "home" reach a scientist's shell writes wherever the account can, and
no folder is private to the instrument. Nor is a compile covered while it runs: what the compilers themselves write on
the way (g++'s intermediate .s and .o; on a GPU the .ptx Triton writes and the object ptxas makes of it, which Triton
then loads) still passes through the temporary folder, where such a sandbox could change a file between its writing and
its reading (14 November 2026: read in Triton 3.8.0's code; g++ 16.2.1 seen putting them there for a small file, and
torch's command line has no -pipe; changing one was not tried). A compile runs only for a graph the folder does not
yet hold. And a person whose XDG_CACHE_HOME is itself in the temporary folder has the cache there.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from backend.engine_runtime import TheodoliteRuntime

logger = logging.getLogger(__name__)

GGUF_ENV = "THEODOLITE_GGUF"
DEVICE_ENV = "THEODOLITE_DEVICE"
LINEARS_ENV = "THEODOLITE_GGUF_LINEARS"
CONFIG_ENV = "THEODOLITE_GGUF_CONFIG"
DTYPE_ENV = "THEODOLITE_DTYPE"
LENS_ENV = "THEODOLITE_GGUF_LENS"
RESIDENT_ENV = "THEODOLITE_GGUF_RESIDENT_GB"
CHECKPOINT_LINEARS_ENV = "THEODOLITE_CHECKPOINT_LINEARS"
DEFAULT_DEVICE = "cuda:0"
_DTYPES = ("bfloat16", "float32")
# what the checkpoint's model is built from, in its snapshot: the files its weights card is keyed on
WEIGHT_FILES = ("config.json", "*.safetensors")
# torch's own variables for where torch.compile and Triton keep what they built (`private_compile_cache`)
COMPILE_CACHE_ENV = "TORCHINDUCTOR_CACHE_DIR"
TRITON_CACHE_ENV = "TRITON_CACHE_DIR"
_told: Dict[str, bool] = {}  # each of `private_compile_cache`'s warnings is said once a process


def compile_cache_dir() -> Path:
    """The instrument's own folder for torch.compile's cache: ${XDG_CACHE_HOME:-~/.cache}/theodolite/torch_compile."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(os.path.abspath(Path(base).expanduser() / "theodolite" / "torch_compile"))


def private_compile_cache() -> Optional[str]:
    """Keep torch.compile's cache in `compile_cache_dir()`, made for this account alone (0700), and return that folder
    (module docstring: why). None, with nothing moved, when the person has set TORCHINDUCTOR_CACHE_DIR themselves (the
    precompiled header then stays where torch keeps it, in its default folder), or when the folder cannot be made: a
    warning then says why, once a process, and the cache stays where it was.

    How torch 2.14.1 keeps its cache (its code read on 14 November 2026: torch/_inductor/runtime/cache_dir_utils.py,
    codecache.py, async_compile.py, runtime/triton_heuristics.py), and so what is done here:
    - torch reads the variable each time it needs the folder, and hands it to its compile workers with each job, so it
      is enough to set it before the first compile, and this may be called again at any time;
    - torch writes its own default into the variable as soon as torch._dynamo is imported, which importing
      backend/engine_runtime.py does (through transformers' model classes; measured), so by the time a runtime is built
      the variable is always set. A value that is torch's own default folder is therefore taken for torch's, and
      moved; any other is the person's, and left;
    - Triton's files follow the folder: torch sets TRITON_CACHE_DIR to <folder>/triton/<device> at the first kernel
      when it is unset. One the person set is left where it is; one torch set at an earlier kernel of this process,
      under its default folder, is unset, so that torch names it again;
    - the precompiled header of torch's C++ kernels stays in the default folder whatever the variable says
      (`codecache._HEADER_DIR`), so that name is pointed at <folder>/precompiled_headers. It is a name inside torch,
      not a setting: with a torch that has no such name a warning says the header was not moved.
    Not moved: torch's package cache (TORCH_CACHING_PRECOMPILE=1, off unless the person sets it), whose folder torch
    fixes when torch._dynamo is imported, before this can run."""
    now = os.environ.get(COMPILE_CACHE_ENV, "").strip()
    utils = sys.modules.get("torch._inductor.runtime.cache_dir_utils")  # not loaded: torch has written no default yet
    torchs = os.path.abspath(utils.default_cache_dir()) if utils is not None else None
    try:
        folder = compile_cache_dir()
        if now and os.path.abspath(now) not in (str(folder), torchs):
            return None  # the person's own folder
        folder.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.mkdir(exist_ok=True, mode=0o700)
        os.chmod(folder, 0o700)  # one made wider before is narrowed
    except (OSError, RuntimeError) as exc:  # a home that cannot be written, or none at all
        if not _told.get("no folder"):
            _told["no folder"] = True
            logger.warning("torch.compile's cache could not be given a folder of this account's own (%s: %s), so it stays "
                           "in %s, where other processes of this account may write: set %s to a folder only you can write.",
                           type(exc).__name__, exc, now or torchs or "torch's default folder", COMPILE_CACHE_ENV)
        return None
    if now != str(folder):
        logger.info("torch.compile's cache is kept in %s, which this account alone can write (not in %s).",
                    folder, now or torchs or "torch's default folder")
    os.environ[COMPILE_CACHE_ENV] = str(folder)
    triton = os.environ.get(TRITON_CACHE_ENV, "").strip()
    if triton and torchs and (os.path.abspath(triton) + os.sep).startswith(torchs + os.sep):
        del os.environ[TRITON_CACHE_ENV]  # torch's own, from an earlier kernel
    try:
        import torch._inductor.codecache as codecache  # the header's folder is a name in this module of torch's
    except Exception:  # a torch whose compiler does not import: nothing compiles here, and the runtime is still built
        codecache = None  # type: ignore[assignment]
    headers = os.path.join(str(folder), "precompiled_headers")
    if hasattr(codecache, "_HEADER_DIR") and hasattr(codecache, "_HEADER_LOCK_DIR"):
        if codecache._HEADER_DIR != headers:
            codecache._HEADER_DIR, codecache._HEADER_LOCK_DIR = headers, os.path.join(headers, "locks")
            built = getattr(getattr(codecache, "_precompile_header", None), "cache_clear", None)
            if built is not None:
                built()  # a header this process already built elsewhere is built again in the folder
    elif not _told.get("no header"):
        _told["no header"] = True
        logger.warning("This torch names no folder for its C++ kernels' precompiled header (codecache._HEADER_DIR), so "
                       "that header, if it keeps one, was not moved to %s with the rest of torch.compile's cache.", folder)
    return str(folder)


def runtime_device() -> str:
    """THEODOLITE_DEVICE as torch names it ("cpu", "cuda:0", ...), DEFAULT_DEVICE when unset; a ValueError in plain words
    for anything else."""
    d = os.environ.get(DEVICE_ENV, "").strip().lower() or DEFAULT_DEVICE
    if d == "cuda":
        d = "cuda:0"
    if d != "cpu" and not re.fullmatch(r"cuda:\d+", d):
        raise ValueError(f"{DEVICE_ENV}={d!r}: the runtime runs on cpu or cuda:N (cuda:0 by default).")
    return d


def hide_gpus() -> None:
    """Hide every GPU from this process (CUDA_VISIBLE_DEVICES=""), for a runtime on the CPU. It holds only if CUDA has
    not been initialised yet; when it has, a warning says so."""
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if torch.cuda.is_initialized():
        logger.warning("CUDA was initialised before the runtime was chosen, so the GPUs could not be hidden: set "
                       "CUDA_VISIBLE_DEVICES= before starting this process for a CPU runtime.")


def reading_class(device: Any, dtype: Any = None) -> str:
    """What a model id carries past the weights' own name when its readings are not taken as the instrument takes them
    (bfloat16 on a GPU): "@cpu/bfloat16", "@cpu/float32", "@cuda/float32"; "" for bfloat16 on any GPU. `dtype` defaults
    to bfloat16 (the checkpoint's)."""
    import torch

    dev = torch.device(device)
    dt = str(dtype if dtype is not None else torch.bfloat16).replace("torch.", "")
    return "" if dev.type == "cuda" and dt == "bfloat16" else f"@{dev.type}/{dt}"


class CheckpointRuntime(TheodoliteRuntime):
    """The pinned runtime on the W4A16 checkpoint, off the GPU: its own `load()`, unchanged, then its model id tagged
    with where the readings are taken (`reading_class`), so nothing taken on the CPU is read as a GPU reading. With
    `linears="int4mm"` its 4-bit linears are then multiplied by torch's int4 CPU kernel (`int4_cpu.install_w4a16`), and
    the model id ends `+int4mm`."""

    def __init__(self, *args: Any, linears: str = "packed", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.linears = linears
        self.layers_as: Optional[Dict[str, int]] = None  # how the linears are held, once loaded with int4mm

    def load(self) -> "CheckpointRuntime":
        super().load()  # loads the weights by self.model_id, the repo id, so the tag goes on after
        self.model_id = f"{self.model_id}{reading_class(self.device)}"
        if self.linears == "int4mm":
            from backend import int4_cpu

            self.layers_as = int4_cpu.install_w4a16(self)
            self.model_id += int4_cpu.SUFFIX
            logger.info("The checkpoint's linears through torch's int4 CPU kernel: %s", self.layers_as)
        return self


def resident_budget(value: str, linears: str) -> Optional[float]:
    """THEODOLITE_GGUF_RESIDENT_GB as a number of GB (None when unset); a ValueError in plain words for anything but a
    number of 0 or more, or beside linears other than packed and int4mm (resident and dense unpack every layer
    already)."""
    if not value:
        return None
    try:
        gb = float(value)
    except ValueError:
        gb = float("nan")
    if not (0 <= gb < float("inf")):
        raise ValueError(f"{RESIDENT_ENV}={value!r}: a number of GB (10^9 bytes), 0 or more.")
    if linears not in ("packed", "int4mm"):
        raise ValueError(f"{RESIDENT_ENV} is a budget for the layers held packed ({LINEARS_ENV}=packed or int4mm); "
                         f"with {LINEARS_ENV}={linears} every quantised layer is unpacked already.")
    return gb


def make_runtime(*, load_jlens: bool = False, **kwargs: Any) -> Any:
    """The runtime this process runs, not yet loaded (call `.load()`): a `GGUFRuntime` on THEODOLITE_GGUF when it is
    set, else the pinned `TheodoliteRuntime` on the W4A16 checkpoint (a `CheckpointRuntime` on the CPU); on
    THEODOLITE_DEVICE (default cuda:0). `kwargs` go to `GGUFRuntime` (`load_jpp=False` leaves the lenses out). Before
    it builds anything, torch.compile's cache is given a folder of this account's own (`private_compile_cache`)."""
    device = runtime_device()
    if device == "cpu":
        hide_gpus()
    private_compile_cache()
    path = os.environ.get(GGUF_ENV, "").strip()
    resident = os.environ.get(RESIDENT_ENV, "").strip()
    if resident and not path:
        raise ValueError(f"{RESIDENT_ENV} is a GGUF's budget for its packed layers, and {GGUF_ENV} is not set.")
    if path:
        import torch

        ct_set = os.environ.get(CHECKPOINT_LINEARS_ENV, "").strip().lower()
        if ct_set and ct_set != "packed":
            raise ValueError(f"{CHECKPOINT_LINEARS_ENV}={ct_set} is the W4A16 checkpoint's, and {GGUF_ENV} is set: a GGUF's "
                             f"linears are {LINEARS_ENV} ({LINEARS_ENV}={ct_set}).")

        from backend.gguf_runtime import GGUFRuntime

        dtype_name = os.environ.get(DTYPE_ENV, "").strip().lower() or "bfloat16"
        if dtype_name not in _DTYPES:
            raise ValueError(f"{DTYPE_ENV}={dtype_name!r}: a GGUF model runs in {' or '.join(_DTYPES)}.")
        if not os.path.isfile(os.path.expanduser(path)):
            raise ValueError(f"{GGUF_ENV}={path!r}: there is no such file.")
        lens = os.environ.get(LENS_ENV, "").strip().lower() or "jpp"
        if lens not in ("jpp", "none"):
            raise ValueError(f"{LENS_ENV}={lens!r}: jpp (load the J++ lens) or none.")
        kwargs.setdefault("load_jpp", lens == "jpp")
        linears = os.environ.get(LINEARS_ENV, "").strip().lower() or "packed"
        from backend.gguf_model import KERNEL_MODES, LINEAR_MODES

        if linears not in LINEAR_MODES + KERNEL_MODES:
            raise ValueError(f"{LINEARS_ENV}={linears!r}: one of {', '.join(LINEAR_MODES + KERNEL_MODES)}.")
        if linears == "int4mm" and device != "cpu":
            raise ValueError(f"{LINEARS_ENV}=int4mm is torch's int4 kernel for the CPU: set {DEVICE_ENV}=cpu with it.")
        resident_gb = resident_budget(resident, linears)
        if resident_gb is not None:
            kwargs.setdefault("resident_gb", resident_gb)
        return GGUFRuntime(
            os.path.expanduser(path),
            device=device,
            load_jlens=load_jlens,
            linears=linears,
            dtype=getattr(torch, dtype_name),
            hf_config=os.environ.get(CONFIG_ENV, "").strip() or None,
            **kwargs,
        )
    ct_linears = os.environ.get(CHECKPOINT_LINEARS_ENV, "").strip().lower() or "packed"
    if ct_linears not in ("packed", "int4mm"):
        raise ValueError(f"{CHECKPOINT_LINEARS_ENV}={ct_linears!r}: packed (the default) or int4mm.")
    if ct_linears == "int4mm" and device != "cpu":
        raise ValueError(f"{CHECKPOINT_LINEARS_ENV}=int4mm is torch's int4 kernel for the CPU: set {DEVICE_ENV}=cpu with it.")
    if device == DEFAULT_DEVICE:
        return TheodoliteRuntime(load_jlens=load_jlens)
    if device == "cpu":
        return CheckpointRuntime(device=device, load_jlens=load_jlens, linears=ct_linears)
    return TheodoliteRuntime(device=device, load_jlens=load_jlens)


def checkpoint_snapshot(repo_id: str) -> Optional[Path]:
    """The snapshot folder of `repo_id` that this machine's Hugging Face cache holds under its `main` ref, which the
    pinned load reads first (`local_files_only`): the folder its config.json sits in, named for its commit; None when
    the repo is not cached here, or `repo_id` is a folder."""
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(repo_id, "config.json")
    except Exception:  # an id the hub cannot parse (a folder's path), or no hub library
        return None
    return Path(found).parent if isinstance(found, str) else None


def checkpoint_files_id(snapshot: Path) -> Optional[str]:
    """The checkpoint's weight files, whatever commit holds them: the sha256 of the names of `WEIGHT_FILES` in a
    snapshot folder and of the blobs the hub's cache links each to (a large file's blob is named by the sha256 of its
    content, a small one's by its git blob id), so another commit of the same files (a model card edited) gives the same
    id and a changed weight or config another. Only the snapshot file's own link is read, as the hub writes it: a blob
    that is itself a link onward (a cache whose blobs were moved into a shared store, as on the owner's machine) is
    still named by the hub's name for it. None when there is no .safetensors file, or a file is not such a link (a
    cache laid out without links, or a copied folder): the snapshot is then all the card can name."""
    names = sorted({f.name for pattern in WEIGHT_FILES for f in Path(snapshot).glob(pattern)})
    if not any(n.endswith(".safetensors") for n in names):
        return None
    lines = []
    for name in names:
        f = Path(snapshot) / name
        if not f.is_symlink():
            return None
        blob = Path(os.path.normpath(f.parent / os.readlink(f)))
        if blob.parent.name != "blobs":
            return None
        lines.append(f"{name} {blob.name}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def weights_card(rt: Any) -> Dict[str, Any]:
    """What `rt`'s readings are taken from, as GET /api/health and the lab daemon send it: a GGUF runtime's own card
    (`GGUFRuntime.weights_card`: the file's provenance, how the model was built, where its tokenizer came from); for
    the stand-in of scripts/atlas_stand_in.py, that; else the W4A16 checkpoint's repo id (the model id before any
    `reading_class`), the snapshot of it the cache holds (`checkpoint_snapshot`, information only) and its weight files
    (`files_sha256`, `checkpoint_files_id`, and `id` its first 16; None when the cache holds no snapshot or does not
    link its files), with where and in what the readings are taken."""
    card = getattr(rt, "weights_card", None)
    if card:
        return dict(card)
    model_id = str(getattr(rt, "model_id", "") or "")
    if getattr(rt, "atlas_stand_in", False):
        return {"format": "stand-in", "model": model_id, "runtime": type(rt).__name__}
    repo = model_id.split("@", 1)[0]
    snapshot = checkpoint_snapshot(repo)
    files = checkpoint_files_id(snapshot) if snapshot is not None else None
    card = {"format": "compressed-tensors W4A16", "repo": repo, "snapshot": snapshot.name if snapshot is not None else None,
            "files_sha256": files, "id": files[:16] if files else None, "linears": getattr(rt, "linears", None) or "packed",
            "dtype": "bfloat16", "device": str(getattr(rt, "device", "")), "runtime": type(rt).__name__}
    if getattr(rt, "layers_as", None):
        card["layers_as"] = dict(rt.layers_as)
    return card
