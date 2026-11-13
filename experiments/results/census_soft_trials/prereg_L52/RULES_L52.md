# The soft_ceil trial at layer 52: what is run, how it is read, what it decides

Fixed on 13 November 2026 (the owner's word that night: "trial layer 52 with soft_ceil after the walk"), before any
answer of this design was written at layer 52, and while the walk downward was writing its first trial, at layer 37,
which had not been read (SHA256SUMS beside this file; `prereg_at`).

**Why.** Layer 52 has failed two trials. Under `break_rel` (+-1 and +-2 of the stream: 1.06 and 2.12 ceilings there)
the whole reading passed and the lower dose alone did not (lead 0.603; words did not carry between the question
halves). Under `break_ceil` (+-2.3 and +-4.6 ceilings) the lower dose alone passed every gate (0.790, 0.679, 0.649 and
0.692, 0.692) and the whole reading failed on the lead beyond text survival (0.553 against 0.58), because the +-4.6
answers read at chance. `soft_ceil` (+-1.6 and +-2.3 ceilings) passed at layer 38 on all three gates. This asks whether
it does above the band as well.

**What has already been seen, and what is done about it.** Two lists of 1,022 neurons have been asked at this layer,
and pushes of 1.06, 2.12, 2.3 and 4.6 ceilings have been read on them. This trial asks **1,022 neurons neither trial
asked** (`neurons_L52_third.txt`, drawn by `draw_third.py` from the seed fixed there; none is in either trial's meta)
against **1,024 controls drawn afresh** (seed 6; the earlier trials' were 3 and 4). No answer of any of these neurons
under any design exists. The steer lens's dose pilot was read before this was fixed: by it, 1.6 ceilings moves the
next word-piece by 0.58 nats at this layer and 2.3 ceilings by 2.39, both inside the span (0.35 to 2.4 nats) in which
every dose read alone so far has led by 0.65 or more, the upper one at its top edge.

**What is run.** Layer 52, `soft_ceil`: each neuron pushed by -2.3, -1.6, +1.6 and +2.3 of its own write ceiling, each
control by the ceiling of the neuron it is paired with; six questions, 48 word-pieces, batch 64, the compiled
unpacker; 49,110 answers, about 83 minutes. From a clean checkout of the commit in `repo_commit`, the commit layer
38's `soft_ceil` trial ran from, by `night_L52.sh`, into `atlas_data/census/trials/soft`, which no page reads as a
layer's census. It runs after the walk downward and before the steer lens's build goes on.

**How it is read.** By `ceil_read.py` (`read_L52.sh`), on the CPU, once the run is written. The reader and the two
files under it are the earlier registrations' own, byte for byte. The same three gates, all to pass, each at 48
word-pieces, at layer 45's bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b.
- **N2, the lower dose's answers alone** (+-1.6 ceilings): G1 to G4, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half are covered.

**What is expected, said beforehand.**
- N1 passes, with a whole lead between 0.72 and 0.80 and a lead beyond text survival of 0.62 or more. That second
  figure is the one that failed here under `break_ceil`, and it is the one to watch: at this layer a ceiling is nearly
  the whole stream, 2.3 ceilings already costs the text 0.60 of its distinct 4-grams (0.73 for the controls), and 1.6
  ceilings has not been read.
- N3 passes, with nine neurons in ten covered or more: here the controls break the text more than the neurons do.
- N2 is more likely to pass than not, with a lead between 0.66 and 0.76. A push of 1.06 ceilings led by 0.603 and
  failed on the words carrying between the question halves (0.546 and 0.577 against 0.57); one of 2.12 led by 0.797.
  If N2 fails, those two figures are expected to be what fails.
- Reported: the +-2.3 answers alone lead by 0.74 to 0.82 on these new neurons, as they did by 0.790 on the second
  list.

**What it decides.**
- **It passes all three:** layer 52 may be offered to volunteers under a kit of its own for this design, with layer
  38 and whatever the walk downward passes. `DOSES_CALIBRATED` is widened to it only on the owner's word. Layers 53
  and above each want a trial of their own, on the owner's word: by the pilot 2.3 ceilings moves the next word-piece
  by 1.5 to 2.0 nats at 53 and 54, by 2.8 to 3.0 at 55 and 56, and by more than 4.7 from 57 on.
- **N1 and N3 pass and N2 fails:** the lower dose is too soft at this layer to carry a lead alone, as 1.06 ceilings
  was. No whole layer is taken at 52 with this design; what to try next is the owner's choice.
- **N1 fails, or N3 fails with N1 passing:** no whole layer is taken at 52 with this design. If it is the lead beyond
  text survival that fails, the lead at this layer is not told from how the text survives at any pair of doses tried,
  and that is said as it is.

**Reported, gating nothing.** The same figures cut to 32 word-pieces; the share covered; what the text loses of its
distinct 4-grams at each dose; how many neurons beat every control; the +-2.3 answers read alone by `trial_read.py`
(`read_L52.sh` writes `L52_upper.json`); and each dose's lead beside the nats the pilot puts on it.

**What a pass would not show.** That any layer above 52 reads with these doses, or that 1.6 and 2.3 are the best two
for this layer.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
