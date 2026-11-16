# A weight-space harness for the 31B: one day's exploration (14 November 2026)

The question put that day: could the instrument grow an environment in which a frontier model (a current one, or the
next) forms a hypothesis about Gemma-4-31B's weights, tests it quickly, makes the edit and tries again? This folder
is the record of the answer: what was read, what was argued, and what was measured on the home card. Nothing here
is registered. The pilots are pilots: one run each, on one or two texts.

## Read in this order

1. `pilot/RESULTS.md`: what was measured on the RTX 5090 (seven read-only pilots, 19:50Z to 21:00Z). Its second
   half (pilots 3 to 7) changes what a reading of an edit may be trusted for, and overrides figures quoted from the
   first two pilots elsewhere in this folder.
2. `verify/arbiter.md`: the proposal as it stands after seven skeptics attacked it. Thirty-two objections sustained,
   seven overruled; the amended phases and the decisions that are the owner's.
3. `design/synthesis.md`: the proposal the skeptics attacked ("Levelling"). It was written before pilots 3 to 7, so
   where it and the arbiter differ, the arbiter stands.

## What the folder holds

| folder | what | written by |
|---|---|---|
| `pilot/` | seven scripts, their JSON, and `RESULTS.md` | the coordinating session, run on the card |
| `design/` | five designs (kernel, driver, validity, speed, future), three judgements, the synthesis | a panel of agents |
| `verify/` | seven skeptics' findings and the arbiter's ruling | agents told to refute |
| `research/` | six literature reports, each ending in an independent citation check | agents with web search |
| `code/` | nine readers' maps of the repository as it stood at e6ac64f | agents, read-only |
| `digest_code.md`, `digest_research.md` | the two digests the panel was given | assembled from the reports |

## How far to trust it

- **Measured** means the pilots: timings, bit-equality, and KL figures taken on this card on 14 November 2026.
  One run each. Wherever the pilots read "raw text" it had no start mark (the tokenizer adds none to plain text; a
  fault of the pilots, found the same day and noted at the top of `pilot/RESULTS.md`), and the model read it at
  about -9.6 nats a piece; those pilots' timings and bit-equality findings stand, their KL figures describe that
  regime and nothing else. Pilot 8 and the chat-formatted halves of pilots 5 and 7 are the ones to read for sizes.
- **Read from the code** means a reader opened the file at the cited line on that day. The repository has moved
  since; a line number is a place to start looking.
- **From the literature** means a report whose claims another agent then tried to break. Under each report the
  citation check's corrections override the report's own takeaways. Much of what is cited was days old and is not
  peer reviewed.
- The designs, judgements and rulings are argument, not evidence. They disagree with each other in places, and the
  later documents say where.

## To run a pilot again

Stop the server on port 8765 first; each pilot loads the model itself (about 10 s) and frees the lens.

    .venv/bin/python -I experiments/results/weight_space_exploration/pilot/pilot7_batch_size.py

Each writes its JSON beside itself, over the one recorded here.

Machine paths in these files were rewritten when they were copied out of the session's scratch folder: the scratch
folder became this folder's path, and the checkout's own path was dropped.
