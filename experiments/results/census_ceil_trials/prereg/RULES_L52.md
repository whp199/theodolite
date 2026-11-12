# The break_ceil trial at layer 52: added to the registration of layer 38's

Added on 12 November 2026 (the owner's word that evening: "trial layer 52 with the new doses too"), while layer 38's
`break_ceil` trial was being written and before any answer of this design at layer 52 existed. Nothing in the files
hashed for layer 38 is changed (their SHA256SUMS stands); these files are hashed beside them in `SHA256SUMS.L52`.

**Why.** `break_rel` failed its trial at layer 52 on its +-1 answers alone (0.603, 0.592, 0.546 and 0.577, 0.554),
while its whole reading passed strongly (lead 0.802). There +-1 of the stream is 1.06 of a neuron's write ceilings,
the least of any layer, and those answers barely move; the lead is in the +-2 answers, 2.1 ceilings. `break_ceil`'s
lower dose, 2.3 ceilings, is about that push. So layer 52 tests the design from the side opposite to layer 38: there
the old lower dose was too hard, here too soft.

**What has already been seen, and what is done about it.** The first trial's 1,022 neurons at layer 52 have been
read at a push (2.1 ceilings) close to this design's lower dose, and their lead was strong. To ask them again would
be half a re-reading. So this trial asks **1,022 neurons the first did not** (`neurons_L52_fresh.txt`, drawn by
`draw_fresh.py` from a seed fixed there) against **1,024 controls drawn afresh** (seed 4; the first trial's were 3).
It is a replication on new units, not a second look at the old ones.

**What is run.** Layer 52, `break_ceil` (+-2.3 and +-4.6 of each neuron's own write ceiling; a control by the ceiling
of the neuron it is paired with), six questions, 48 word-pieces, batch 64, the compiled unpacker; 49,110 answers,
about 75 minutes. From the same clean checkout as layer 38's (`repo_commit`), by `night_L52.sh`, into
`atlas_data/census/trials/ceil`.

**How it is read.** By `ceil_read.py` (`read.sh` with LAYERS=52), the three gates as for layer 38, all to pass, each at
48 word-pieces, at layer 45's bars: N1 the whole reading, N2 the lower dose's answers alone, N3 the neurons the
controls cover (at least half of them).

**What is expected, said beforehand.** A pass on N1 and N2. The doubt is N3 and G2: at about two ceilings this layer's
text is already breaking (the first trial's +-2 answers lost 0.39 to 0.42 of their distinct 4-grams for neurons and
0.58 to 0.59 for controls), so part of the lead may be that the controls' text breaks more than the neurons', not the
words a neuron brings. The gates on the covered neurons and on the lead beyond text survival are there for that.

**What it decides.**
- **It passes all three:** the design holds above the band as well as, if layer 38 passes, below it. The upward walk
  may go on with it (53, 54, ...; their lists are those of the neighbour trials, none of them asked yet), and a layer
  that passes may be offered to volunteers under a kit of its own. `DOSES_CALIBRATED` is widened to a layer only on
  its passed trial and the owner's word.
- **It fails:** no whole layer is taken at 52 with this design either, and what failed is said: if N3 or G2, the lead
  at this depth is not told from how the text survives, as at layer 57.

**Reported, gating nothing.** As for layer 38, but with no comparison of each neuron's mass across designs: these
neurons were not in the first trial.
