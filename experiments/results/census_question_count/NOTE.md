# Why the census walks stopped at 31 and 53, read from the trials already taken

16 November 2026. Exploratory: every figure here is read from trials already read and written up. Nothing here is a
registration, and nothing in it decides a layer. A design it points to would be registered and trialled on fresh
neurons.

## What was done

`subsample.py` reads a finished trial again on subsets of each neuron's answers: 1, 2 or 3 questions of each half (the
three matched drops for 1 and 2), and the lower pair of doses, the upper pair, or both. It uses the trial's own
statistics (`readlib.py` of the walks' registrations: mass_c, mass_B, rep_AB, rep_BA, each an AUC of neurons over
controls). At 3 questions a half it gives the registered figures back exactly (L31's lower dose: 0.634, 0.553/0.548,
0.587). Fourteen trials were read, about a second for each subset (`cells/`).

`project.py` turns each figure into d' (sqrt 2 x probit of the AUC) and finds how it grows with questions a half
(d' ~ q^b; b from 2 to 3 questions and from 1 to 3, each clipped to 0.5..1.0). It then projects the figure to six
questions a half, which is twelve questions, as d'(3) x 2^b (`projections.txt`). This assumes six new questions
carry what the six asked do. It is a rough guide: each AUC has a standard error near 0.013 at 1,022 neurons, and the
1-question figures are bent by the reading's two-answer minimum for a word.

## What it says

**Layer 31 is short of answers, not of push.** Every figure of the lower dose alone rises with each question added
(rep_BA 0.525, 0.534, 0.548; rep_AB 0.515, 0.535, 0.553). At twelve questions the lower dose alone projects to pass
every gate: lead 0.73 to 0.75, rep_AB 0.606, rep_BA 0.571 to 0.587 (the thinnest; bar 0.57), new questions alone
0.671. The two doses are 6% apart and read alike (the upper alone rises the same way), so the second dose adds
answers, not a second push.

**Layer 53 is short of push in its lower dose, and more questions do not mend it.** Its lower dose's figures for
words carrying between the halves stay flat as questions are added (rep_BA 0.543, 0.563, 0.545). At twelve
questions it projects under its bars: lead 0.600, rep_BA 0.564, new questions alone 0.542. Its upper dose leads
(0.689, rising with questions) but its words do not carry between the halves at any count (0.45 to 0.51). Most units
of both kinds have nothing that carries (median rep 0.00 for neurons and for controls). A tail of neurons does: the
strongest carry real words, such as kilo/giga/peta, pacific/mountain, germ, and sylva/silva (`l53_rep.py`). The
controls' text breaks more than the neurons' at that dose (distinct 4-grams 0.725 against 0.833).

**The same reading tells the two kinds of failure apart where the answer is known.** Layer 37 failed with
`soft_ceil` on its lower dose's rep_BA (0.542), and passed with `match_ceil`'s harder doses. Its lower dose's rep_BA
is flat with questions (0.568, 0.538, 0.541; projected 0.559 at twelve). So at 37 more questions would not have done
what the dose did. Layer 31 rises steadily. Layer 53's lower dose is flat, as 37's was.

**The gates and the yield of single neurons disagree.** The trials' own counts at 48 word-pieces (`named`: q at or
under 0.05, words carrying both ways at p 0.05, and the residual at p 0.05):

| layer, design | trial | above every control | named | in 100 above the controls' 95th percentile |
|---|---|---|---|---|
| 34, 35, 36 `match_ceil` | pass | 8, 3, 6 | 0, 0, 0 | 42, 24, 21 |
| 31 `match_ceil` | fail | 19 | 60 | 23 |
| 53 `soft_ceil` | fail | 123 | 117 | 28 |
| 37 `soft_ceil` | fail | 28 | 158 | 32 |
| 52 `break_rel` | fail | 229 | 189 | 40 |
| 45 `break_rel` | pass | 169 | 262 | 38 |

The gates ask whether the typical neuron reads unlike a random push. The named count asks how many neurons can be read
one at a time. At 34 to 36 no neuron is named, because of the far-out controls that hold every q above 0.05 there (an
open decision of the owner's). At 53 a ninth of the neurons are named. A whole layer at 53 would name more neurons
than one at 36 did.

## What it points to (the owner's choices, none taken)

1. **Below 32: more questions.** Twelve questions in halves of six, at `match_ceil`'s doses. With one dose (1.0
   nats) that is 24 answers a neuron, the cost of a layer now. Its whole reading is what the projection above reads
   for the lower dose alone, but it has no separate gate on a second dose. With both doses it is 48 answers a
   neuron, twice the cost of a layer, and its lower-dose gate is that same projected figure. Either way six new
   questions and their spares must be fixed before any run, with `verify_atlas_writer.py --design`. Walking below 30
   also needs the fine dose pilot taken there.
2. **Above 52: the dose first.** Take the fine dose pilot at 53 to 59 (about ten minutes), then try a dose between
   `soft_ceil`'s two at 53 (0.37 and 1.53 nats), such as `match_ceil`'s 1.0 nats, with the twelve questions.
3. **Whether a layer may be taken on its yield.** A gate on how many neurons are named against chance would take 53
   and would not take 34 to 36. That changes what a whole layer is taken for. It belongs with the owner's open
   question on which per-neuron count the ceiling layers report.
