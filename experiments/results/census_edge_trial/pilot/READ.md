# The ladder at layer 29, read by its rule (RULE.md)

Ladder written 23 November 2026, 02:31:13Z to 03:14:45Z (pilot.log; RULE.md's sha256 39af30383262 logged at 02:31:05Z,
before the run began), 24,588 answers, on the home RTX 5090; 32 neurons and 32 isotropic controls, seed 12
(`atlas_data/census/edge_pilot/L29-calib_edge.jsonl`). Read with `census_tools.py curves` (curves.txt, curves.json),
`census.dose_window` unchanged.

First breaks (mean distinct 4-grams more than 0.1 below the unpushed answers' to the same questions):

| | neurons | controls |
|---|---|---|
| pushed up | +4.27 | +3.99 |
| pushed down | -3.99 | -3.99 |

- B, the smallest of the four, is 3.99 ceilings (0.86 of the stream).
- **The upper dose is the rung under it: 3.73 ceilings** (a median 0.81 of the stream). There the text keeps 0.93 of
  its 4-grams for the neurons both ways and 0.93 and 0.94 for the controls.
- **The lower dose is 3.73 over the square root of two: 2.64 ceilings** (0.57 of the stream; the ladder's nearest rung,
  2.66, keeps 0.99 to 1.00).
- Neither stopping case holds: B is not the first rung, and the text broke inside the ladder.

For comparison, not used by the rule: match_things's doses at 29 were 2.10 and 2.32 ceilings (0.45 and 0.49 of the
stream). By the fine dose pilot's curve at 29 (fine_pilot_low.json, 128 other neurons), 0.57 and 0.81 of the stream
move the next word-piece by about 1.8 and 4.4 nats. On the 32 neurons here the "moves" column finds no dose at which
the neurons move more answers than the controls; at these sizes every answer moves (100% both kinds), so it cannot.
