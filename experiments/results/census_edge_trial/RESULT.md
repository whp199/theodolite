# The edge_things trial at layer 29: it fails every gate, and the lead falls as the push grows

Registered in `prereg/` (SHA256SUMS bfc32817b8b0, prereg_at 03:19:45Z 23 November 2026; commits f9aeea5 and c38072f,
pushed 03:19:56Z), written 03:20:09Z to 06:01:08Z on the home RTX 5090 from a clean checkout of f9aeea5 in one session
(98,220 answers; the meta matched the registration: design, doses, twelve questions, seed 13, 1,024 isotropic
controls, the registered 1,022 neurons), read 06:01-06:06Z by the registration's `read.sh`. Layers are counted from 0,
as the trials and the board count them: 29 here is 30 on the pages. The doses, ±2.64 and ±3.73 ceilings, are the
`calib_edge` ladder's by its rule (`pilot/`).

## The verdict: fail

| reading (bars 0.60, 0.58, 0.57, 0.57, 0.56) | lead | beyond text | words carry A→B | B→A | new questions |
|---|---|---|---|---|---|
| **N1** every answer | **0.473** | **0.571** | **0.488** | **0.488** | **0.502** |
| **N3** the neurons the controls cover (974 of 1,022) | **0.475** | **0.558** | **0.489** | **0.488** | **0.506** |
| *reported:* the lower dose alone (±2.64) | 0.502 | 0.559 | 0.494 | 0.479 | 0.523 |
| *reported:* the upper dose alone (±3.73) | 0.428 | 0.541 | 0.458 | 0.454 | 0.437 |
| *reported:* three questions a half, first / last | 0.510 / 0.491 | 0.577 / 0.584 | 0.476 / 0.488 | 0.481 / 0.471 | 0.553 / 0.531 |
| *for comparison:* `match_things`'s trial at 29, N1 (±2.10, ±2.32) | 0.574 | 0.599 | 0.528 | 0.520 | 0.592 |

N1 and N3 fail on every one of G1 to G4. G5 and G5b pass (null split 0.501, read gain 1.027, every answer present;
the code ranks as read). At 32 word-pieces it fails too (lead 0.478).

## What it read

- **The lead falls as the push grows.** At 29, across the two trials, the neurons' lead on each dose's answers alone:
  0.569 at ±2.10 ceilings, 0.532 at ±2.32 (`match_things`, other neurons and controls), 0.502 at ±2.64, 0.428 at ±3.73
  (here). Every step up lowers it. At the upper dose the neurons lead the random directions by less than chance.
- **Both kinds switch language.** The strongest controls write Indonesian ("adalah berikut dan"), Portuguese ("você
  aqui para"), German, Spanish, 5.5 to 7.1 standard deviations over the controls' median. The strongest neurons do
  the same (Indonesian, Vietnamese, French, Dutch), with one exception writing "name, middle, given, full, identity,
  gender". So the words a neuron's pushes bring out are mostly the words any push of that size brings out here.
- **The text barely moves even so.** The neurons lose 0.046 to 0.050 of their distinct 4-grams at ±3.73 and 0.006 to
  0.009 at ±2.64, the controls about the same; 98 to 99 answers in 100 run to the 48-piece cut. Beyond how the text
  survives the neurons do lead a little (0.571, 0.558 covered): their raw lead is under chance because the random
  directions move their answers further.
- **Single neurons**: none with q ≤ 0.05; 5 beat every control (about 1 by chance); 9 stand more than five standard
  deviations over the controls' median; 51 are strong on both halves (p ≤ 0.05 each; 2.6 by chance).
- **Read on each neuron's weight-lens words instead** (`census_weight_words/explore.py`, exploratory, gating
  nothing): lead 0.568 (shuffled 0.495), 0.558 and 0.562 on the halves, 0.532 at the lower dose and 0.575 at the upper,
  and no neuron at q ≤ 0.05. On `match_things`'s trial at 29 the same reading gave 0.533. So the harder push brings out
  a neuron's own unembedding words slightly more, while what the census picks from its answers is swamped by language.

## Against what was said beforehand

- "It passes about 1 in 4." It failed.
- "If it fails, it fails on the words carrying (G3) first." **Wrong**: it fails every gate, the lead (0.473) furthest.
- "N1's lead between 0.56 and 0.66." **Wrong**: 0.473, under it and under chance.
- "The text loses 0.05 to 0.09 at ±3.73 and under 0.02 at ±2.64." Right: 0.046 to 0.050 and 0.006 to 0.009.
- "The share covered: 85 to 98 in 100." Right: 95.3.
- "The controls that stand out most write other languages, and stand further out than 6.7 to 7.6 standard
  deviations." Right on the languages; **wrong** on how far: 5.5 to 7.1.
- "Neither dose alone passes G3." Right.

## What it decides (as registered)

No whole layer is taken at 29 with `edge_things`. Nothing opens; `DOSES_CALIBRATED` is unchanged. What to try next at
29 is the owner's.

## What it suggests, gating nothing

- The volunteer's 12B finding, that the lead returns once no dose sits past the text edge, does not carry to the 31B
  at 29: nothing here was past the edge, and the lead was best at the gentlest push tried (±2.10, 1.0 nats of the next
  word-piece) and fell with every step up. On the 12B their layer 24 lead rose when they brought the doses down to the
  edge; here bringing them up to it lowered the lead.
- If anything is tried at 29 again, the evidence points to gentler pushes than `match_things`'s, below 1.0 nats,
  where random directions may switch language less; at 30 too `match_both`'s lower dose alone (0.596) led its upper
  (0.580). No push under 2.10 ceilings has been trialled at 29 or 30.
- What this design could not show is what the walk's own failure there left open: whether 29's neurons write anything
  a census on these questions can tell from random pushes at any size.
