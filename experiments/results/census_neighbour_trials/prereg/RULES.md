# The neighbour trials: what is run, how it is read, what it licenses

Fixed on 12 November 2026, before any of these runs was taken (SHA256SUMS beside this file; `prereg_at`).

**Why.** `break_rel`'s doses are counted calibrated at layers 40-50 alone (counted from 0), where its first trial
(layer 45, 29 October) and eleven whole layers passed. About three in ten of those layers' strongest links end outside
the band, nearly all at layers 38, 39, 51 and 52. The owner's aim is a census at every layer. A whole layer is 13.2
hours; a trial is about 70 minutes. So each layer is trialled before it is taken whole.

**What is run, at each layer.** 1,022 neurons (`neurons_L<NN>.txt`, drawn by `draw_neurons.py` from a seed fixed there;
none asked before) against 1,024 isotropic controls drawn with seed 3, as the first trial's were; `break_rel` as it is
(-2, -1, +1, +2 of the stream's median length on its six questions); `--batch 64 --max-new 48 --unpacker compiled`;
49,110 answers. From a clean checkout of the commit in `repo_commit`, by `night.sh`. Written into
`atlas_data/census/trials/neighbours`, which no page reads as a layer's census.

**Order.** 39, 51, 38, 52: the nearest first. In this first sitting only these four. The lists for 30-37 and 53-56 are drawn and
hashed now, so going on outward needs no new choice of neurons.

**How it is read.** By `neighbour_read.py` (`read.sh`), on the CPU, after the runs are written. The reader is the first
trial's (`trial_read.py`, `readlib.py`, unchanged: the same sha256 as on 29 October), its gates layer 45's as they
were. A trial passes when all three pass, each at 48 word-pieces:
- **N1, the whole reading**: G1 lead 0.60, G2 beyond text survival 0.58, G3 replication both ways 0.57, G4 the new
  questions alone 0.56, G5 sanity, G5b the code ranks as read.
- **N2, the +-1 answers alone**: G1 to G4 on them, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half the neurons are covered.

N2 and N3 are the two checks the review of the first trial made after the fact (layer 45 passed both, layer 57
failed both). Here they are gates. Checked before hashing: on the first trial's own answers this reader gives the
review's figures (whole 0.7144, 0.7244, 0.6879 and 0.6739, 0.6693; +-1 alone 0.7064; 893 of 1,022 covered: 0.6921,
0.6913, 0.6573 and 0.6430, 0.6505) and on layer 57's probe it fails all three (354 of 507 covered).

**What a pass licenses.** A whole layer of `break_rel` at that layer, 48 word-pieces, 2,048 controls, as layers 40-50
were taken; and the layer may be offered to volunteers. `DOSES_CALIBRATED` is widened to it only on the owner's word.

**What a fail means.** Under this design the band stops there on that side: no whole layer is taken at that layer, and
none further out on that side, unless the owner chooses otherwise. Going outward, the next layer on a side is trialled
only after the one before it passed.

**What a pass does not show.** The gates are layer 45's, read at another layer. A pass says that there, as at 45, a
neuron's push is told from a random direction's by the words it brings, in the +-1 answers and among neurons whose
text survives as the controls' does. It does not say the doses are the best for that layer, and one question set
with one greedy answer each is the only sample.

**Reported, gating nothing.** The 32-piece reading and the length rule; the share covered; the text's distinct
4-grams at each dose; how many neurons beat every control.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
