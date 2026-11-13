"""Peak GPU memory of a census batch at each of a design's questions: the worst case of a whole layer's run.

Each question is written once at --batch rows through the census's own writer (census.write_census_batch), every row
pushed by a random direction at the design's largest dose (so rows part and none stops early), with no stop word-piece,
so every row runs to --max-new: the largest KV cache any of the layer's batches can hold. Prints, per question, its
length and the peak allocated and reserved memory, and the card's total.

It also prints what the driver counts as used (the card's total less torch.cuda.mem_get_info's free, after each
batch, while torch's allocator still holds its peak): that adds the CUDA context and is what a free-memory check made
with nvidia-smi before the run has to leave room for.

--hold-mib N sets N MiB aside on the card before the model loads, as a desktop or another process would hold it: the
largest N with which every batch is still written is the card's real margin (the driver's own count of used memory
is not: torch's allocator fills what is free and gives cached blocks back when it runs out, so it reads near full
either way).

usage: .venv/bin/python -I scripts/measure_batches.py [--design break_rel] [--batch 64] [--max-new 48] [--layer 45]
       [--hold-mib N]
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))

# the runtime first: importing it sets the allocator's expandable segments (engine_runtime's top), which torch reads at
# its first allocation on the card, so a block held aside before it was imported left the run on the default allocator
import backend.engine_runtime  # noqa: E402,F401
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="break_rel")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--layer", type=int, default=45)
    ap.add_argument("--hold-mib", type=int, default=0)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    held = torch.empty(a.hold_mib * 2**20, dtype=torch.uint8, device="cuda") if a.hold_mib else None  # noqa: F841
    if a.hold_mib:
        print(f"{a.hold_mib} MiB held aside on the card", flush=True)
    from backend import census as cz
    from backend import w4a16_fast
    from backend.causal_check import language_model
    from backend.runtime_factory import make_runtime
    from census_run import free_lens

    rt = make_runtime(load_jlens=False).load()
    if getattr(rt, "jpp_lenses", None):
        free_lens(rt)
    w4a16_fast.install(rt, "compiled", log=print)
    lm = language_model(rt.hf_model)
    dev = next(p.device for p in lm.parameters() if p.is_cuda)
    D = lm.config.hidden_size if hasattr(lm, "config") else rt.hf_model.config.text_config.hidden_size
    total = torch.cuda.get_device_properties(dev).total_memory
    print(f"card {torch.cuda.get_device_name(dev)}: {total / 2**30:.2f} GiB; resting allocated "
          f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB", flush=True)
    from backend.atlas_index import AtlasIndex
    from backend.runtime_factory import weights_card

    from backend import atlas_ops as ao

    # the stats index as census_run.py takes it: the runtime's own layer count and width
    stats = AtlasIndex(model=getattr(rt, "model_id", None), shape=(len(lm.layers), ao.ffn_width(lm, 0)),
                       weights=weights_card(rt)).get("act_scale")
    if stats is None:
        raise SystemExit("No stats index (act_scale) for this model and shape: the push's unit, p50, is read from it.")
    push = 2.0 * float(stats["resid_q"][a.layer, 1])  # break_rel's largest dose, x p50
    print(f"rows pushed along random directions at {push:.1f} (2 x p50 at layer {a.layer})", flush=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    worst = worst_used = 0
    for name, text in cz.design_questions(a.design).items():
        ids = rt.tokenize_prompt(rt.compose_prompt(text, chat=True))
        delta = torch.randn(a.batch, D, generator=g)
        delta = (delta / delta.norm(dim=-1, keepdim=True) * push).to(dev)  # rows that part from one another
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        rows = cz.write_census_batch(rt, a.layer, ids, delta, max_new=a.max_new, stop_ids=[])
        torch.cuda.synchronize(dev)
        secs = time.perf_counter() - t0
        lens = sorted(len(r["token_ids"]) for r in rows)
        pa, pr = torch.cuda.max_memory_allocated(dev), torch.cuda.max_memory_reserved(dev)
        free, _ = torch.cuda.mem_get_info(dev)
        used = total - free
        worst, worst_used = max(worst, pr), max(worst_used, used)
        print(f"{name:>12s}: {len(ids)} word-pieces asked, answers {lens[0]}-{lens[-1]} long; peak allocated "
              f"{pa / 2**30:.2f} GiB, reserved {pr / 2**30:.2f} GiB, driver used {used / 2**20:,.0f} MiB, "
              f"headroom {(total - used) / 2**20:,.0f} MiB; {secs:.2f} s", flush=True)
    print(f"worst reserved {worst / 2**30:.2f} GiB; worst driver used {worst_used / 2**20:,.0f} of {total / 2**20:,.0f} MiB",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
