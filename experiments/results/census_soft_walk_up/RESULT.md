# The walk upward with soft_ceil: it stopped at layer 53

Registered in `prereg/`: hashed at 05:50Z and committed at 05:52:43Z on 16 November 2026 (f935df9), before any answer
existed. It was pushed at 05:59Z, six minutes into the run, while the first answers were being written and an hour
before any was read. Run on the home card, 05:52Z to 07:07Z on 16 November 2026: 49,110 answers in one session at 11.2 answers a second, from kit 4's commit
(81c1362). Read by the registration's own `read.sh`. The files beside this one are copied from
`atlas_data/census/trials/soft_walk_up/` by `collect.py`; `chain.log` is the walk's own log.

## The reading

Layer 53 failed all three gates, so the walk stopped there. Layers 54 to 59 were not trialled, and nothing is said of
them.

| at 48 word-pieces | lead | beyond text | words carry A to B | B to A | new questions alone |
|---|---|---|---|---|---|
| bars | 0.60 | 0.58 | 0.57 | 0.57 | 0.56 |
| N1 whole | 0.694 | 0.631 | 0.574 | **0.557** | 0.654 |
| N2 the lower dose alone (+-1.6) | **0.551** | **0.567** | **0.557** | **0.545** | **0.521** |
| N3 the covered neurons (988 of 1,022) | 0.693 | 0.626 | **0.566** | **0.550** | 0.653 |
| the upper dose alone (+-2.3), reported | 0.689 | 0.626 | **0.507** | **0.511** | 0.623 |

Figures under their bar are in bold.
- N1 failed by 0.013, on words carrying from the second half of the questions to the first. G5 passed (read gain
  1.068; the home canary recorded 1.067), and so did G5b.
- N2 failed all four of its gates (G1 to G4), with all five figures under their bars, the lead included. Its G5
  passed.
- N3 failed on words carrying between the halves, both ways.
- The upper dose's answers alone led by 0.689, but the words picked on one half of the questions read at chance on the
  other (0.507, 0.511).
- At 32 word-pieces the whole reading failed too (lead 0.642; words carrying 0.559 and 0.525).
- 123 neurons beat every control (1.0 expected by chance); 154 have q at or under 0.05.
- The text: the lower dose hardly changes the neurons' text (its distinct 4-grams fall by 0.008 and 0.004). The upper
  dose takes away 0.17 of them, and the controls' upper dose takes away 0.28. At 52 the upper dose took away 0.60.
- The pm1 and upper files show G5b as FAIL. That check only applies to the whole reading (the units file ranks every
  answer), and it is not one of N2's gates. Layer 52's trials show the same.

## Against what was said beforehand

- "About as likely to pass as not." It failed.
- "N1 passes, with a whole lead between 0.72 and 0.82." Wrong: N1 failed, and its lead was 0.694, under the range.
- "N3 passes, with 80% to 95% of the neurons covered." Wrong: N3 failed, and 96.7% were covered, over the range.
- "N2 is the open gate", and it failed. "If N2 fails, the figure for the new questions alone and the two for words
  carrying between the halves fall first, before the lead." Wrong: all three fell under their bars, but so did the
  lead, and it fell furthest (0.551, 0.049 under its bar; the new questions alone 0.039 under; words carrying 0.013
  and 0.025 under). From layer 52's lower dose the lead fell 0.082, more than any of the three (0.056, 0.047, 0.063).
- "The upper dose's answers alone (reported) lead by 0.72 to 0.82 at 53 and 54, ..." Wrong at 53: 0.689 (54 was not
  reached). That lead did not carry between halves.
- "At 53 and 54 the neurons' text loses less of its distinct 4-grams at the upper dose than at 52 (0.60)." Right at
  53: 0.17.
- **The registered clause applies:** "If layer 53 fails N1 with both doses inside the span, the pilot's span is no
  guide above layer 52." Both doses were inside it (0.37 and 1.53 nats; the span is 0.35 to 2.4), and N1 failed. The
  lower dose, at 0.37 nats, led by 0.551. Before it, the only dose read alone inside that span to lead by under
  0.60 was layer 57's +-1 (0.60 nats, 0.5225).

## What it decides (as registered)

No whole layer is taken at 53 with `soft_ceil`. The board marks 53 "its trial failed, 16 November 2026 (soft_ceil):
not taken"; 54 to 59 stay "not trialled with this kit's push strengths". What to try at 53, if anything (`match_ceil`
would need the fine dose pilot there first, since `MATCH_CEIL` holds 30 to 38 alone), and whether to try above it,
are the owner's choices. Either would be registered anew, on neurons this walk did not ask.
