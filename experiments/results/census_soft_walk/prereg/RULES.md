# The walk downward with soft_ceil, layers 37 to 30: what is run, how it is read, what it decides

Fixed on 13 November 2026 (the owner's word that night: "Yes, walk first"), before any answer of any of these layers
was written under any census design (SHA256SUMS beside this file; `prereg_at`).

**Why.** `soft_ceil` (each neuron pushed by +-1.6 and +-2.3 of its own write ceiling) passed its trial at layer 38 on
all three gates, on neurons no trial had asked (experiments/results/census_soft_trials). By that registration it is the
design for the walk downward, a trial before every whole layer. This is that walk, as far as layer 30.

**What is run.** At each layer in turn, 37, 36, 35, 34, 33, 32, 31, 30: `soft_ceil` on the layer's 1,022 neurons drawn
on 12 November (`neurons_L<NN>.txt`; the same bytes as in the `break_ceil` registration, whose SHA256SUMS they are in)
against 1,024 controls (seed 5, as at layer 38), each control pushed by the ceiling of the neuron it is paired with;
six questions, 48 word-pieces, batch 64, the compiled unpacker; 49,110 answers, about 83 minutes a layer. From the
clean checkout of the commit in `repo_commit`, which is the commit layer 38's trial ran from, by `night.sh`, into
`atlas_data/census/trials/soft_walk`, which no page reads as a layer's census.

**The order, and where it stops** (`walk.sh`). A layer is read as soon as it is written, and the next layer down is
taken only if this one passed all three gates. The walk stops at the first layer that fails, and at layer 30 at the
latest. Layers below a failed one are not trialled here, so nothing is said of them.

**What has been asked before.** No neuron of layers 31 to 37 has been asked anything. At layer 30 the ladder of 28
October asked 64 neurons at five doses; a few of them may be among this list's 1,022. The doses were not chosen on any
single neuron's answers there.

**How it is read.** By `ceil_read.py` (`read.sh`), on the CPU. The reader and the two files under it are the earlier
registrations' own, byte for byte. The same three gates, all to pass, each at 48 word-pieces, at layer 45's bars
(0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b.
- **N2, the lower dose's answers alone** (+-1.6 ceilings): G1 to G4, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half are covered.

**What has been seen that bears on it.** The steer lens's dose pilot was taken and read before this was fixed
(experiments/results/atlas_steer_dose/pilot.json): how far a push of a given size moves the next word-piece, a layer
at a time. Set beside the fourteen doses a trial has read alone, a push that moves the next word-piece by 0.35 to 2.4
nats led by 0.65 or more (eight of eight), and one of more than 3 nats did not reach the bar (five of five). The one
reading under that span, 0.18 nats at layer 52, led by 0.603 and failed its other figures. By the pilot the two doses
here move the next word-piece by:

| layer | 37 | 36 | 35 | 34 | 33 | 32 | 31 | 30 | (38) |
|---|---|---|---|---|---|---|---|---|---|
| +-1.6 ceilings, nats | 0.27 | 0.27 | 0.32 | 0.21 | 0.30 | 0.37 | 0.24 | 0.30 | (0.35) |
| +-2.3 ceilings, nats | 0.95 | 0.91 | 1.15 | 0.64 | 0.82 | 1.01 | 0.64 | 0.73 | (1.34) |

**What is expected, said beforehand.**
- N1 passes at every layer reached, with a whole lead between 0.62 and 0.70: the upper dose is inside the span that
  reads at all eight layers.
- N3 passes at every layer reached, with seven to nine neurons in ten covered.
- N2 is the open gate at every layer, and it is what the walk is expected to stop on. The lower dose moves the next
  word-piece less here than at layer 38 at seven of the eight layers. At layers 37 and 36 (0.27 nats) N2 is about
  even. If the walk reaches layer 34 (0.21 nats), N2 is more likely to fail there than to pass.
- Where N2 fails, the figures expected to fall first are the two for words carrying over between the question halves
  (0.594 and 0.605 at layer 38, against 0.57), before the lead itself.
- More likely than not the walk stops before it reaches layer 30.
- If N1 fails anywhere, or N2 fails at layer 32 while passing at 33, the pilot's span is not the guide it looks.

**What it decides.**
- **A layer passes all three:** it may be offered to volunteers under a kit of its own for this design, and the walk
  goes on to the next layer down. `DOSES_CALIBRATED` is widened to a layer only on its passed trial and the owner's
  word.
- **A layer fails:** no whole layer is taken there with this design, and the walk stops. What failed is said as it is.
  If N1 and N3 pass and N2 fails, the lower dose is too soft at that layer to carry a lead alone; what to try there
  (the two doses taken from the pilot's curve; one dose on more questions) is the owner's choice and would be
  registered anew on neurons this walk did not ask.
- **Going on below layer 30,** if all eight pass, is the owner's word.

**Reported, gating nothing.** The same figures cut to 32 word-pieces; the share covered; what the text loses of its
distinct 4-grams at each dose; how many neurons beat every control; the +-2.3 answers read alone by `trial_read.py`
(`read.sh` writes `L<NN>_upper.json`); and each dose's lead set beside the nats in the table above, which is the
pilot's forecast put to the test.

**What a pass would not show.** That 1.6 and 2.3 are the best two doses for a layer; that a layer not trialled reads;
or anything of layer 52 and above.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
