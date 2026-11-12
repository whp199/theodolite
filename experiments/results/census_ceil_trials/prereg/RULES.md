# The break_ceil trial at layer 38: what is run, how it is read, what it decides

Fixed on 12 November 2026, before any answer of this design was written on the model (SHA256SUMS beside this file;
`prereg_at`).

**Why.** `break_rel` failed its trial at layer 38 with its lead at chance (0.497), one layer below 39, where it passed
(0.694). Where it passes, its lower dose is 1.76 to 2.76 of a neuron's write ceilings; at 38 it is 3.14, and more at
every layer below. `break_ceil` is the same design with its doses counted in ceilings, +-2.3 and +-4.6 of each
neuron's own (experiments/results/census_scaled_design/DESIGN.md). Two readings of layer 38's failure fit what is
known: the dose (the push was too hard), or the depth (a neuron this deep does not yet write towards words, so no
dose gives a census of words a lead). This trial tells them apart.

**What is run.** Layer 38, `break_ceil`: the 1,022 neurons of its first trial (`neurons_L38.txt`, the same file: drawn
by a seed before anything was read, and no question-and-dose cell of this design was asked there), 1,024 isotropic
controls drawn with seed 3, each pushed by the ceiling of the neuron it is paired with; six questions, 48 word-pieces,
batch 64, the compiled unpacker; 49,110 answers, about 75 minutes. From a clean checkout of the commit in
`repo_commit`, by `night.sh`, into `atlas_data/census/trials/ceil`, which no page reads as a layer's census.

**How it is read.** By `ceil_read.py` (`read.sh`), on the CPU, once the run is written: `neighbour_read.py` of the same
day with the doses read from the run's meta. The same three gates, all to pass, each at 48 word-pieces, at layer 45's
bars (0.60, 0.58, 0.57 and 0.57, 0.56):
- **N1, the whole reading**: G1 to G5 and G5b.
- **N2, the lower dose's answers alone** (+-2.3 ceilings): G1 to G4, and G5.
- **N3, the neurons the controls cover**: G1 to G4 on the neurons whose six text-survival covariates all lie inside
  the controls' range; fails if fewer than half are covered.

Checked before hashing: on the first trial's own answers (layer 45, `break_rel`) this reader gives that trial's
figures, as `neighbour_read.py` did (whole 0.7144, lower dose alone 0.7064, 893 of 1,022 covered: 0.6921).

**What it decides.**
- **It passes all three:** the dose was the trouble at layer 38. The design may be walked on downward, a trial before
  every whole layer (37, 36, ...: their lists are here), and a layer that passes may be offered to volunteers under a
  kit of its own. `DOSES_CALIBRATED` is widened to a layer only on its passed trial and the owner's word.
- **It fails:** by these rules no whole layer is taken at 38 or below with this design either. If its lead (G1, whole)
  is still within 0.05 of chance, the depth is taken to be the trouble, and the census is held to stop at layer 39 on
  this side. If its lead has risen clear of chance but short of the bars, that is said as it is: a gentler push helps
  and does not suffice, and what to try next is the owner's choice.

**Reported, gating nothing.** The same figures cut to 32 word-pieces; the share covered; what the text loses of its
distinct 4-grams at each dose; how many neurons beat every control; and, since the neurons are the first trial's,
how each neuron's mass under this design goes with its mass under `break_rel` there (Spearman's rho over the 1,022).

**What a pass would not show.** That 2.3 is the best number of ceilings, or that layers below 38 pass: each is
trialled in its turn.

**Amendments.** Any change to a file here after it is hashed is written, with its time and reason, in
`AMENDMENTS.txt` beside this file before a reading is made.
