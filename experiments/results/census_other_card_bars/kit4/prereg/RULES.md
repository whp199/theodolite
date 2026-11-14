# The bars for another card class under kit 4: what is measured, by what rule, and what it decides

Fixed on 14 November 2026 (the owner's word: "Bars first"), before any measurement here was taken and before any
spot chunk of any kit 4 bundle was drawn (SHA256SUMS beside this file; `prereg_at`; `repo_commit`).

**Why.** A volunteer's run from a card that does not write the home card's bits is taken if, on a few chunks written
again at home, its answers stay within set distances of the home card's (`census_volunteer.ACCEPT_OTHER_CLASS`). The
two log-probability distances were measured on the home card on 31 October 2026, at layers 40 to 50, under
`break_rel`, kit 1's design (`../../bars.json`). Kit 4 takes layers 32 to 39, 51 and 52, with three designs:
`match_ceil` at 32 to 37, `soft_ceil` at 38 and 52, `break_rel` at 39 and 51. `match_ceil` and `soft_ceil` push by
counts of each neuron's own write ceiling, not by a share of the stream, so their pushes are other lengths than any
measured on 31 October. How far the home card's own arithmetic moves its log-probabilities under those pushes, and
how far a push a little wrong moves them, was never measured. Layer 34's two doses are 6% apart (2.62 and
2.79 ceilings) and the median bar was built to catch a push 5% off, so the old bars may be too loose there or too
tight; nothing says which. Two bundles are on their way: layer 34 (`match_ceil`, an RTX PRO 6000 Blackwell) and layer
51 (`break_rel`, an RTX 3090, the first 24 GB card).

**A fault found first, in the script.** `scripts/census_bars.py` made a near miss by scaling p50, the stream's median
length. Under a ceiling design p50 does not enter the push (the push is dose x the neuron's write ceiling), so every
near miss there would have written the control's own answers, no size would have qualified, and both bars would have
come out dropped. Found on 14 November by reading the code, before anything was run. The script now scales the write
ceiling for those variants (`push_scaled`; the census's own code is not touched), and stops without deriving anything
if any near miss writes the control's own rows at any layer. Shown on the CPU stand-in only: at its `soft_ceil` layer
the differences grow with the size of the miss, as they do at its `break_rel` layer.

**What is run** (`measure.sh`, about 30 minutes on the home RTX 5090 with nothing else on it, after layer 36's own
census ends). Layers 32, 33, 34, 35, 36, 37, 38, 39, 51 and 52, each with the design kit 4 names for it and kit 4's
settings (batch 64, 48 word-pieces, the compiled unpacker, seed 7, 2,048 controls): each layer's first planned batch,
64 answers, in the thirteen variants of 31 October:
- **the control**, as the kit writes it. Every row at every layer must be the home canary's, bit for bit, or nothing
  is derived.
- **arithmetic only**: batch 16, 32 and 48, and batch 64 with the reduced-precision sums turned off.
- **near misses**: the push 1, 2, 5 and 10% longer, and the same shorter.

Each variant is set against the home canary as the intake sets a spot answer against the home card's (the pieces both
wrote alike before they part, where the home card's log-probability is under -0.01).

Layer 36 is in because it is the kit's and `match_ceil`'s, though the home card takes it. Layers 53 to 59 are named
in the kit and shut, untrialled: they are not measured, and no run of another card class is judged there until they
are. Layers 40 to 50 are not measured again and their bars are not touched.

**The groups.** The layers of one design are pooled: `match_ceil` (32 to 37, 384 answers a variant at batch 64),
`soft_ceil` (38 and 52, 128 answers), `break_rel` (39 and 51, 128 answers).

**The rule of 31 October, unchanged**, for each of the median and the 99th percentile of |difference|:
- the floor F is the largest value among the arithmetic-only variants (at least 0.0001);
- a near miss of size s is M(s), the smaller of its two signs' values, each sign needing 200 pooled pieces to count;
- the bar is sqrt(F x M(s)), to two figures, for the smallest s with M(s) at least 4 x F;
- where no size qualifies, the statistic cannot tell a push 10% off from arithmetic and its bar is dropped.

**What the measurement decides** (`decide` in `census_bars.py`; `table` in its `bars.json` is what
`census_volunteer.OTHER_CLASS_BARS` takes, copied by hand, nothing rounded or adjusted on the way):

- **R1, the ceiling designs.** `match_ceil`'s bars at 32 to 37 and `soft_ceil`'s at 38 and 52 are their groups' bars
  by the rule, whatever they come out as: tighter than 0.028 and 0.37, looser, or dropped.
- **R2, `break_rel` at 39 and 51.** The bars of 31 October (0.028 and 0.37) stand there if, for each of the two, the
  group's floor is under the bar and the near miss the bar was built to catch (5% for the median, 10% for the 99th
  percentile, both signs readable) is over it. If either fails, the group's own bars by the rule replace both.
- **R3, a layer of its own.** A layer whose own floor (from its 64 answers alone; an arithmetic-only variant with
  fewer than 200 pieces there does not count) is at or above the bar in use for its group takes that statistic's bar
  from its own rows by the rule, which may be a dropped bar. A pooled bar that one layer's own arithmetic reaches
  would refuse honest runs of that layer.
- **R4, first word-pieces.** The bar stays 0.90 for a group whose arithmetic-only variants all keep at least 0.93 of
  first word-pieces (on 31 October they kept 0.983 to 0.994). Where the lowest keeps fewer, the group's bar is that
  lowest share less 0.05. The near misses' shares are reported and do not enter: this bar is there to catch a run of
  another layer, dose or model, which parts at once, not a push a little off.
- **R5, the unpushed answers.** Not measured again: they carry no push, are the same six at every layer and under
  every design, and their rule (a parting is taken only where the home card wrote its piece at under 0.6) stands.
- **R6, if a design loses both log-probability bars.** The spot check there then rests on first word-pieces and the
  unpushed answers alone. No run of another card class is accepted at those layers on that alone until the owner has
  said so, knowing it.

**If the measurement cannot be read.** The control not the home canaries' bits at any layer: the instrument has moved
since the canaries were taken; nothing is derived, no bundle is judged, and the cause is found first. A near miss that
moved nothing: the same.

**What is not changed by this.** Layers 42, 44, 47, 48 and 50, accepted from other cards under `break_rel`, were
judged by bars measured for their design at their layers, and stand. The home card's class is still checked bit for
bit. No bar is set by looking at a volunteer's answers.

**What had been seen when this was fixed.**
- The measurement of 31 October, and the five accepted spot checks under `break_rel` (first pieces 0.975 to 0.987,
  medians 0.013 to 0.015, 99th percentiles 0.168 to 0.215).
- The layer-51 claimant's own remarks on the board issue: their RTX 3090 and a Radeon under ROCm agree on 98.0% of
  first word-pieces over 9,600 answers of layer 51, median difference 0.011, 99th percentile 0.156. Their figures,
  between two cards that are not the home card, unchecked here. They do not enter the rule.
- That layer 34's run began with a canary that is not the home card's bits (its START line says so). None of its
  answers, and no figure of it.
- The stand-in's dry run of the script (4 answers a layer; too few to derive anything).
- No answer of any other card under `match_ceil` or `soft_ceil`, and no arithmetic-only or near-miss variant of the
  home card under either.

**What this leaves untested.**
- The variants are the home card's own arithmetic. Another card class may move log-probabilities in ways batch size
  and full-precision sums do not span; on 31 October the first other-card canary sat inside the floor's range, and
  that is the only evidence that they do span it.
- A near miss is the whole push scaled. A run with one dose right and the other wrong, or the wrong layer's doses of
  `match_ceil`, is not modelled; the first-piece bar and the kit's own checks of the run's settings are what catch
  those.
- 64 answers a layer: a layer's own floor at the 99th percentile rests on a few hundred pieces and is rough, so R3
  may fire on noise. It then loosens one layer's bar to that layer's own rule-made bar, no further.
- `soft_ceil` and `break_rel` here have two layers each, 128 answers a variant against 704 on 31 October.
