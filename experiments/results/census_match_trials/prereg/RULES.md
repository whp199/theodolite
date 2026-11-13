# The match_ceil trial at layer 37: what is run, how it is read, what it decides

Fixed on 13 November 2026 (the owner's word that night: "Yes, design it"), before any answer of this design was
written on the model (SHA256SUMS beside this file; `prereg_at`). The design and the rule its doses come from were
committed before the fine dose pilot was taken (`4b49e69`, experiments/results/census_scaled_design/MATCH.md);
this file was hashed after the pilot, since it names the two doses the pilot gave.

**Why.** `soft_ceil` (+-1.6 and +-2.3 of each neuron's write ceiling) passed at layer 38 and failed at layer 37: there
its lower dose alone did not carry between the question halves (0.542 against 0.57), the covered neurons fell 0.002
short on the same figure, and its upper dose alone passed every gate. `match_ceil` sets each layer's two doses where
the next word-piece moves by 1.0 and by 1.4 nats, as the census's fine dose pilot measures it at that layer.

**The doses at layer 37.** By the fine pilot of 13 November, 05:24Z (128 neurons, eighteen rungs 7% apart) and the rule fixed
before it: **+-2.41 and +-2.55 of each neuron's own write ceiling**. By the same pilot, `soft_ceil`'s 1.6 and
2.3 ceilings move the next word-piece by 0.25 and 0.77 nats at this layer (the steer lens's coarser pilot,
read off a line, had put them at 0.27 and 0.95), and by 0.26 and 1.39 at layer 38 (0.35 and 1.34).
So both doses here are harder than any push read at this layer: its 2.3 ceilings, which passed alone, is under the
lower target. The design note supposed otherwise, from the coarser pilot. The doses stand as the rule gave them.

**What has already been seen, and what is done about it.** The walk's trial asked 1,022 neurons at this layer at 1.6
and 2.3 ceilings, and the lower target here was chosen knowing that their 2.3-ceiling answers passed alone. To ask
them again would be half a re-reading. So this trial asks **1,022 neurons that trial did not ask**
(`neurons_L37_fresh.txt`, drawn by `draw_fresh.py` from the seed fixed there; none is in the walk's list,
`neurons_L37_walk.txt`) against **1,024 controls drawn afresh** (seed 7; the walk's was 5). No answer of any of these
neurons under any design exists.

**Also seen before this was hashed.** Layer 52's `soft_ceil` trial was read at 05:14Z on 13 November, after the
design and its rule were committed and before the fine pilot was taken. It passes; its lower dose, 0.58 nats by the
steer lens's pilot, led by 0.633, under the 0.65 that every dose between 0.35 and 2.4 nats had reached before it. So
how far a push moves the next word-piece sorts the leads less closely from layer to layer than this design
supposed. Neither the rule nor the doses were changed for it.

**What is run.** Layer 37, `match_ceil`: each neuron pushed by -2.55, -2.41, +2.41 and +2.55 of its own
write ceiling, each control by the ceiling of the neuron it is paired with; six questions, 48 word-pieces, batch 64,
the compiled unpacker; 49,110 answers, about 88 minutes. From a clean checkout of the commit in `repo_commit`, by
`night.sh`, into `atlas_data/census/trials/match`, which no page reads as a layer's census.

**How it is read.** By `ceil_read.py` (`read.sh`), on the CPU, once the run is written. The reader and the two files
under it are the earlier registrations' own, byte for byte. The same three gates, all to pass, each at 48
word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b.
- **N2, the lower dose's answers alone** (+-2.41 ceilings): G1 to G4, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half are covered.

**What is expected, said beforehand.**
- About as likely to pass all three as not; if anything a little more likely. The draft of this file written before
  the fine pilot said six chances in ten. Two things since lower it: both doses came out above any push read at
  this layer, and layer 52's reading weakened the rule the doses rest on.
- N1 passes, with a whole lead between 0.64 and 0.70.
- N2 is more likely to pass than not, with a lead between 0.63 and 0.69. Its two figures for words carrying between
  the question halves are the ones at risk: the walk's 2.3-ceiling answers gave 0.604 and 0.605 on other neurons,
  0.034 over the bar.
- N3 is the least sure. In the walk the covered neurons fell 0.002 short on one of those two figures, with softer
  doses than these. Between seven and eight neurons in ten are expected to be covered.
- If it fails, it fails narrowly, on words carrying between the question halves, and not on the lead.
- If the upper dose's answers alone (reported) lead by less than 0.60, the lead at this layer is being lost between
  0.77 and 1.4 nats, sooner than at layer 38, and the upper target is too high for the layers below the band.

**What it decides.**
- **It passes all three:** `match_ceil` is the design for the layers below 38, a trial before each (their doses are
  in `MATCH_CEIL` from the same pilot), and layer 37 may be offered to volunteers under a kit of its own.
  `DOSES_CALIBRATED` is widened to a layer only on its passed trial and the owner's word. Whether the walk goes on at
  36 is the owner's word.
- **It fails:** no whole layer is taken at 37 with this design. Layer 37 will then have failed one trial at 1.6 and
  2.3 ceilings and another at 2.41 and 2.55, while its 2.3-ceiling answers alone, reported and gating nothing,
  passed every gate. That is said as it is, and what follows (a pair of doses nearer 2.3 ceilings there; the bars;
  a stop at 38) is the owner's choice. Nothing is said of 36 and below, which have not been trialled.

**Reported, gating nothing.** The same figures cut to 32 word-pieces; the share covered; what the text loses of its
distinct 4-grams at each dose; how many neurons beat every control; the upper dose's answers read alone by
`trial_read.py` (`read.sh` writes `L37_upper.json`); and each dose's lead beside the nats the fine pilot measured.

**What a pass would not show.** That 1.0 and 1.4 nats are the best two targets, or that any layer below 37 passes.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
