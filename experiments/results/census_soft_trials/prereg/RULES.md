# The soft_ceil trial at layer 38: what is run, how it is read, what it decides

Fixed on 12 November 2026 (the owner's word that evening: "Yes, design it"), before any answer of this design was
written on the model, and while layer 52's `break_ceil` trial was being written and not yet read (SHA256SUMS beside
this file; `prereg_at`).

**Why.** `break_ceil` (+-2.3 and +-4.6 of each neuron's write ceiling) failed its trial at layer 38 on one figure: the
whole reading's lead, 0.599 against a bar of 0.60. Its lower dose alone passed every gate (0.682, 0.727, 0.652 and
0.636, 0.649), and so did the neurons the controls cover. Looked at afterwards, its +-4.6 answers alone led by 0.506,
which is chance. `soft_ceil` is the same design with its doses at +-1.6 and +-2.3 ceilings: the dose that read, and
one half a doubling under it (experiments/results/census_scaled_design/SOFT.md).

**What has already been seen, and what is done about it.** These two doses were chosen on the answers of the 1,022
neurons both earlier trials at this layer asked, whose +-2.3 answers have been read. To ask them again would be half
a re-reading. So this trial asks **1,022 neurons neither trial asked** (`neurons_L38_fresh.txt`, drawn by
`draw_fresh.py` from the seed fixed there; none is in either trial's meta) against **1,024 controls drawn afresh**
(seed 5; the earlier trials' were 3). No answer of any of these neurons under any design exists.

**What is run.** Layer 38, `soft_ceil`: each neuron pushed by -2.3, -1.6, +1.6 and +2.3 of its own write ceiling, each
control by the ceiling of the neuron it is paired with; six questions, 48 word-pieces, batch 64, the compiled
unpacker; 49,110 answers, about 85 minutes. From a clean checkout of the commit in `repo_commit`, by `night.sh`, into
`atlas_data/census/trials/soft`, which no page reads as a layer's census.

**How it is read.** By `ceil_read.py` (`read.sh`), on the CPU, once the run is written. The reader and the two files
under it are the `break_ceil` trials' own, byte for byte (the same sha256 as in that registration's SHA256SUMS). The
same three gates, all to pass, each at 48 word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b.
- **N2, the lower dose's answers alone** (+-1.6 ceilings): G1 to G4, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half are covered.

**What is expected, said beforehand.**
- N1 passes, with a whole lead between 0.64 and 0.70.
- N3 passes, with nine neurons in ten covered or more: at these doses the text hardly breaks.
- N2 is the open one. No push of 1.6 ceilings has been read anywhere. More likely to pass than not, with a lead
  between 0.60 and 0.66. If it fails, the figures expected to fall first are the two for words carrying over between
  the question halves and the one for the new questions, as at layer 52 under `break_rel`.
- Reported: the +-2.3 answers alone lead by 0.64 to 0.72 on these new neurons, as they did by 0.682 on the first.

**What it decides.**
- **It passes all three:** this is the design for the walk downward. A trial before every whole layer (37, 36, ...:
  their lists are in the `break_ceil` registration, none of them asked), and a layer that passes may be offered to
  volunteers under a kit of its own. `DOSES_CALIBRATED` is widened to a layer only on its passed trial and the
  owner's word.
- **N1 and N3 pass and N2 fails:** the lower dose is too soft to carry a lead alone. By these rules no whole layer
  is taken with this design. It is said as it is: at layer 38 the room is about one dose wide, near 2.3 ceilings, and
  what to try next (one dose on more questions; a dose matched layer by layer on the steer lens's pilot) is the
  owner's choice.
- **N1 fails:** no whole layer is taken with this design. If the +-2.3 answers alone (reported) lead by less than
  0.60 here, the first trial's lead at that dose did not repeat on new neurons, and the census is held to stop at
  layer 39 on this side. Otherwise what failed is said as it is.
- **N3 fails with N1 passing:** the lead is not told from how the text survives; not taken.

**Reported, gating nothing.** The same figures cut to 32 word-pieces; the share covered; what the text loses of its
distinct 4-grams at each dose; how many neurons beat every control; and the +-2.3 answers read alone by
`trial_read.py` (`read.sh` writes `L38_upper.json`), the first trial's lower dose asked again of new neurons.

**What a pass would not show.** That 1.6 and 2.3 are the best two doses, or that any layer below 38 passes: each is
trialled in its turn. Nor anything about layer 52 and above, whose `break_ceil` trial is being written as this is
fixed.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
