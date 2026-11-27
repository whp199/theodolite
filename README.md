<p align="center">
  <img src="docs/readme/frontispiece.png" width="100%"
       alt="The opening page of Theodolite: the title begun with an engraved initial T, and sixty stacked rings with the path of one word winding up through them, labelled with what a lens makes of it at a few of its layers.">
</p>

<p align="center">
  <em>An atlas of all 1,290,240 neurons in Gemma-4-31B-it, a census of what a layer’s neurons do when pushed, and the instrument to read them with.</em>
</p>

<p align="center">
  <sub>Gemma-4-31B-it &nbsp;◆&nbsp; 60 layers &nbsp;◆&nbsp; 21,504 neurons in each, 1,290,240 in all &nbsp;◆&nbsp; 5,376 dimensions &nbsp;◆&nbsp; a vocabulary of 262,144 word-pieces</sub>
</p>

<p align="center">
  <a href="#what-it-is">what it is</a> ·
  <a href="#the-atlas-and-its-census">the atlas</a> ·
  <a href="#caring-contempt-and-a-stage-villain">a finding</a> ·
  <a href="#the-instrument">the instrument</a> ·
  <a href="#getting-started">getting started</a> ·
  <a href="#how-far-to-trust-it">how far to trust it</a> ·
  <a href="#where-it-is-heading">where it is heading</a> ·
  <a href="#levelling-an-experimental-weight-space-harness">levelling</a> ·
  <a href="#built-on">built on</a> ·
  <a href="#what-is-new-here">what is new here</a> ·
  <a href="#how-to-cite">how to cite</a>
</p>

<p align="center">
  <sub>The census: twenty-five of the sixty layers done, none being taken, three open to volunteers (27 November 2026).
  <a href="https://github.com/whp199/theodolite/issues/1">The board</a> · <a href="VOLUNTEERS.md">take a layer</a>.</sub>
</p>

<br>

## What it is

Between a question and its answer, a language model carries each word up through all of its layers, and little of
that country has been mapped. **Theodolite** is an instrument for exploring it in **Gemma-4-31B-it**, Google’s
31-billion-parameter instruct model, run in 4 bits on one 32 GB card. With it you can:
- follow a word as it rises through the sixty layers, and read what a lens makes of it at each one;
- see how much of each layer’s change was gathered from the words before and how much was recalled from memory;
- open any of the model’s 1,290,240 feed-forward neurons;
- push a neuron along its write direction and read how the answer changes.

It is a learning aid and a research tool at once. Every page can be read two ways. **Rubrics**, red italic lines as
scribes and early printers set their directions apart, guide a first reading. **The apparatus**, as in a critical
edition, opens beneath each part of a page: the technical name, the formula, the raw values, the sources and every
control. Both are switches at the top of every page, and nothing is hidden from either reader.

<p align="center">
  <img src="docs/readme/two-readings.png" width="100%"
       alt="Two ways to read the instrument: rubrics, red directions for those beginning, switched on when you arrive; and the apparatus, the formulas, raw values, sources and every control, opened beneath a double gilt rule for scholars.">
</p>

## The atlas and its census

<p align="center">
  <img src="docs/readme/atlas.png" width="100%"
       alt="The atlas with neuron 17966 of layer 47 open: on the left its words through the lens, waves, waterfall and the Chinese for wave, and its web, what drives it up (thread, hurricane, earthquake, storms) and down (wave, weather, Storm); in the middle the stacked sheets of layers 35 to 51 with the neuron's links fanning out from it across the layers; on the right its census answers, where pushed up by one stream's length it answers who it is with I am a river wave of same-ocean water, a wind-driven rain of the same-sea rain, and at twice that with Rain Rain Rain.">
</p>

The atlas (`#/atlas`) puts every feed-forward neuron of the model on one map: sixty sheets of 21,504, with the neurons
that write alike side by side. Open one and the atlas gathers what can be read of it:
- its words through the lens;
- its web of weight links, back to what drives it and on to what it drives;
- the passages where each of its two faces fires hardest;
- a push you can try on any question;
- its census answers.

Above is neuron 17,966 of layer 47. Through the lens it writes “waves”, “waterfall” and “波”. Asked who it is, and
pushed up, the model answers “I am a river wave of same-ocean water”.

**The census** is what the atlas measures. While the model answers a fixed set of questions, each neuron of a layer
is pushed both ways along its write direction. Its answers are set against those of 2,048 random directions pushed
as far. A neuron ranks high when the words its push brings stand further above the random directions’ than nearly
every random direction’s own words do. So a high rank says its push differs from a random one. It does not say that
its words share a theme, or name what the neuron does when it fires.

A whole layer is 565,254 answers, thirteen to seventeen hours on one RTX 5090, so the census is taken a layer at a time
by the home card and by volunteers on their own. The first eleven layers taken (41–51) rest on one trial, at layer
46. Since then no layer has been taken blind:
- **A trial first.** About a thousand of a layer’s neurons are trialled against gates fixed before the trial ran, and
  the layer is offered only if it passes.
- **A push sized to the layer.** Where the first design’s push is too hard or too soft for a layer, it is counted
  instead in what a neuron there can write by itself.
- **Checked against the home card.** Each volunteer run’s first batch is compared with the home card’s: bit for bit
  on the same kind of card, and within bars measured beforehand on another.

On 27 November 2026 twenty-five layers were done (31–55, counting 1–60 as the pages do; the board counts 0–59). One of
them, layer 31 (the board's 30), read whole falls short of its trial on one figure, and the board says so. None was
being taken, and three were open (kit 8 added 56–58, the board's 55–57, after passing the dual-control `U2` rollout
walk). Following the census layer-gap study and its kill-pilots
([`ANSWER.md`](experiments/results/census_layer_gap/ANSWER.md)), those twenty-five layers (31–55, keeping `.units.jsonl`'s
29,265 page-described neurons) remain the taken census until 56–58 arrive, and layers 1–30 and 59–60 stay shut / not
taken; when exploratory gap files are present, the atlas displays exploratory set-text push (56–60) and corpus-firing (1–30)
readings beside always-visible red caveats.
[How to take a layer](VOLUNTEERS.md) · [the board](https://github.com/whp199/theodolite/issues/1) · [the atlas and the
census at length](docs/ATLAS.md) · [every figure](ROADMAP.md#vi--the-atlas).

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/atlas-census.png" alt="Layer 47's census in the atlas's left panel: 2,768 neurons above every control, about ten by chance alone, and the first ranks with their words: back holding hold held; table set; conditions under condition; rain wave water wind river; gradient descent steepest steep; delivery taking take possession; working overtime work hours school; plot plotting plotter; company commercial business; flux constant state change.">
      <p><b>A layer’s census.</b> Its neurons ranked by how far their pushed answers’ words stand above the random
      directions’, each with the words it brings and how much of its mass comes from words the page leaves out.</p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/readme/atlas-passages.png" alt="The passages tab for neuron 17966 of layer 47, where its up face fires hardest, the word-piece underlined: a turbine that transforms the energy of flowing water, an app that plays rain or white noise, the sound of chirping birds in a forest, remains destroyed by the sea; each with how strongly it fires there.">
      <p><b>Its passages.</b> Where each face of the neuron fires hardest in two million word-pieces of web text and
      chat, the strongest windows kept whole. Gathered on the 31B, and not yet validated as labels.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/atlas-push.png" alt="The push tab for neuron 17966 of layer 47, asked to describe your ideal breakfast in two sentences and answering three times at once: pushed up by 0.8, crashing waves in the background, a plate of crisp, salty ocean waves and a steaming cup of rain-flavored coffee; left alone, blueberry pancakes, bacon and black coffee; pushed down, pancakes with berries, avocado toast and orange juice.">
      <p><b>The push.</b> Ask anything and read three answers at once: the neuron pushed up, left alone and pushed
      down. A push this size is more than the neuron can write by itself, so it shows what the push does, not what the
      neuron does when it fires.</p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/readme/atlas-spread.png" alt="The spread of layer 47's census, coloured by question: six dense clusters, one for each question's answers, a wide scatter of answers pushed off them, a few tight knots and long looping strands of broken text, and the open neuron's answers drawn large in gilt and blue.">
      <p><b>The spread.</b> A layer’s census answers laid out by the words they use: the answers a push leaves much as
      they were gather by question, and those it turns aside scatter round them.</p>
    </td>
  </tr>
</table>

## Caring, contempt and a stage villain

On 11 November 2026, searching the atlas, a scientist chat found two recipes of three neurons each that change the
voice the model answers in. Each neuron is pushed along its write direction by a share of its layer’s stream:

| recipe | layer · neuron · dose |
|---|---|
| **caring** | 36 · 1095 · +0.35, 44 · 9148 · +0.5, 44 · 2057 · −0.25 |
| **contempt** | the same three, every sign flipped |
| **a stage villain** | 40 · 21379 · −1.0, 42 · 4330 · −0.5, 43 · 8862 · +0.4 |

Pushed one way, an answer opens “I know how heartbreaking and discouraging it feels in the moment…”. Flipped, it
opens “First off, stop reading this and just get it over with: **Who cares?**”. The villain speaks in wicked grins:
“A wicked grin of salt and light… The tide retreats with a slow, low laugh”.

The instrument wrote each recipe out as a llama.cpp control vector, so it runs without Theodolite (checked on the QAT
Q4_0 GGUF; other GGUFs are untried). They are published as
[Steering vectors 1](https://github.com/whp199/theodolite/releases/tag/steering-vectors-1):

```bash
llama-server -m gemma-4-31B-it-QAT-Q4_0.gguf --control-vector-scaled gemma4-31b-caring.gguf:1
```

What was checked, described and not judged against a bar set beforehand:
- each file is, bit for bit, what the export writes from the checkpoint;
- in llama.cpp, caring and contempt showed on all three prompts tried;
- the caring file run backwards (`:-1`) wrote the contempt file’s answers to the letter;
- in the instrument, on five prompts the recipes were not tuned on, caring showed on three and contempt on four;
- the villain shows in llama.cpp too, but sits at its breaking point there.

What was not checked:
- whether any of them changes what the model does, rather than how it sounds;
- sampled answers, long chats, thinking turned on, or prompts outside English.

Each push is also more than one neuron can write by itself (1.3 times, for layer 36’s neuron 1095). So a recipe
carries the effect of a direction, not of “the neuron turned up”. To make your own on a CPU in about ten seconds,
from a GGUF you already have, see `scripts/export_control_vector.py` and [ARCHITECTURE §6](ARCHITECTURE.md).

## The instrument

<p align="center">
  <img src="docs/readme/instrument.png" width="100%"
       alt="The instrument page: sixty engraved rings with the path of the word I winding up from layer 1 to layer 60, labelled computer, HUAWEI, bot and am at several layers; beside it, the leaf of layer 36 with its top words through the lens and a bar splitting the layer's change into 37 percent gathered and 63 percent recalled.">
</p>

- **The instrument.** One ring for each layer, the path of a word winding up through them, and beside it the words a
  lens makes of it: J++ (“the lens”) and the plain logit lens (“the naked eye”).
- **A layer’s leaf.** Where the stream points at one layer, and how much of its change was gathered (attention) or
  recalled (the feed-forward block).
- **A neuron’s folio.** One neuron’s two faces, what each writes, the later neurons that read it, and a causal test of
  its label.
- **Landmarks.** Three words fix a frame, as a surveyor sights known landmarks, and each layer is read as a bearing
  and an elevation against them.
- **The nudge.** A small push on the model’s stream while it reads, sized as a share of the stream’s own strength.
- **The vault.** A catalogue of the 1,290,240 neurons: look one up by the words it writes, edit one, and hear an
  answer before and after.
- **The scientist.** An agent in the atlas’s side panel, with the same tools: it runs pushes and texts, names what it
  finds and points at it on the map. It runs on Claude Code by default, on Codex, on Gemini (through Antigravity, or
  through the Gemini CLI if you pick it), or on any OpenAI-compatible endpoint.
- **Lexicon and Lineage.** Every word of the instrument in plain terms with its formula, and the research it is made
  from.

Each page leaf by leaf, neuron-explorer’s features set beside the atlas’s, and the instrument’s words with their
names in the literature: [docs/INSTRUMENT.md](docs/INSTRUMENT.md).

## Getting started

You need:
- one NVIDIA GPU with at least 32 GB (the instrument peaks near 23 GB; a census layer also runs on a 24 GB card with nothing
  else on it);
- Linux, Python 3.12, [uv](https://github.com/astral-sh/uv), and Node 20+ for the interface;
- about 22 GB of disk for the model and 6.4 GB for the two lenses.

```bash
uv venv --python 3.12 .venv
uv pip install -r pyproject.toml
# the code runs with HF_HUB_OFFLINE=1, so download first:
.venv/bin/hf download google/gemma-4-31B-it-qat-w4a16-ct
.venv/bin/hf download neuronpedia/jacobian-lens \
    gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt \
    gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt
```

Take a first sighting:

```bash
(cd frontend && npm ci && npm run build)      # the server serves frontend/dist at /
.venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8765 --timeout-graceful-shutdown 3
# open http://127.0.0.1:8765/
```

Try it without a GPU or the weights, on a tiny random stand-in of Gemma 4 (what it shows is random, and the page says so):

```bash
.venv/bin/python scripts/atlas_stand_in.py --serve 8765
```

Map every neuron. Stop the server first, since only one GPU process runs at a time:

```bash
.venv/bin/python scripts/atlas_build.py all                                    # words, ranges, wiring, layout (minutes)
.venv/bin/python scripts/census_run.py --layer 45 --neurons all --unpacker compiled   # one layer's census (layers from 0; ~13 h)
```

The server answers only pages served from this machine. The full set-up is in [docs/SETUP.md](docs/SETUP.md):
- the CPU and GGUF paths, and the faster unpackers;
- the server’s WebSocket, the passages and the steer lens;
- the scientist’s backends and how far it can reach into your files;
- rented cards, reproducing the first study, and where everything is in the repository.

To take a census layer on your own card, see [VOLUNTEERS.md](VOLUNTEERS.md).

## How far to trust it

This is research code. Read a single sighting as a lead, never as a finding.
- **The lens.** It was fitted on the base model, `google/gemma-4-31B`, and is applied to the instruct model. It is
  good for comparing readings and poor on its own: from about layer 21 it puts conditions in the order the model’s
  behaviour does, but through layer 50 its top word matches the model’s next word on prose at most about 5% of the
  time. This is the largest fault left. Lenses fitted on the instruct model were put to three registered tests (R2
  to R4), and none was served.
- **The label gate.** The instrument’s first study found that a neuron’s lens words predicted its real effect only
  at the layers it
  sampled from 41 on (32–40 were not sampled). So below layer 41, and wherever the lens and the naked eye share none
  of their first ten words, no page puts a neuron’s words in its main reading. The apparatus keeps them, with the
  reason.
- **The census.** A rank is a lead against random pushes, with a p and a false-discovery q beside it. Where a layer’s
  doses are not counted calibrated, the atlas says so on its page.
- **The first study’s audit.** It found nine faults in the instrument. Seven are mended or applied. Each face’s own input
  passages are built but not yet validated as labels, and the lens is the one left open.

The fault table and every caution are in [docs/TRUST.md](docs/TRUST.md).

## Where it is heading

Theodolite began as the instrument of one study, of whether Gemma 4 is “female-coded”
([its findings](docs/FINDINGS.md)), and has outgrown it. It is becoming a general-purpose instrument for
interpretability, and a way to learn it: one a newcomer can learn to read a model with, and one whose readings say
how far they can be trusted. What is under way:
- **The census of all sixty layers.** Layer by layer, a trial first, taken by the home card and by volunteers. Below
  layer 31 and above layer 58 the push that reads is still being found. On 16 November 2026 a design that asks twice
  the questions at one dose passed its trial at layer 32 (31 on the board), which kit 5 offers, and failed at 31 (30
  on the board); the same questions at two doses failed there too (17 November). Twelve other questions at two doses,
  each asking how a thing works, for three tips or about an object, passed at 31 on 18 November, and kit 6 offers
  them there. The same design passed at 54 and 55 and failed at 30 and 56 on its single-control walk (18 and 19
  November; kit 7 offers it at 54 and 55), then passed at 56–58 (the board's 55–57) under the dual-control `U2`
  rollout walk (`columns` + `isotropic`, 26–27 November); kit 8 offers it at 56–58.
- **Reading a neuron without pushing it.** Each face’s strongest passages are kept, and since 14 November 2026 a
  pattern written from them can be put to a blind test. Neither is yet validated as a label.
- **A lens fitted on the instruct model.** None tested so far has beaten the base-fitted J++ where it would be
  served.
- **Weight edits read against the untouched model.** This is Levelling, below.
- **More than one kind of card.** The CPU, GGUF files and rented cards.

Nothing on the [roadmap](ROADMAP.md) has a date.

### Levelling, an experimental weight-space harness

Every reading so far pushes the model’s stream while it runs. Levelling begins the other half: changing the model’s
weights and reading what the change does. The question behind it is whether the instrument can become an
environment where a model, today’s or the next, forms a hypothesis about Gemma-4-31B’s weights, makes the edit,
tests it quickly and tries again.

Its first slice (1a) was built and card-checked on 16 November 2026 (142 of 145 standing checks passed on the 31B),
slice 1b completed the Phase 1 weight-edit kernel on 18 November 2026 (1,056/1,056 CPU checks, and 68/68 31B card
checks in `scripts/verify_levelling_1b.py`), and on 19 November 2026 Phase 2’s Agent’s Door (150/150 CPU/sandbox checks
in `scripts/check_level_door.py`, 38/38 31B card checks in `scripts/verify_level_door.py`), Phase 3’s Claim Gate
(307/307 CPU checks in `scripts/check_level_claim.py`, 16/16 31B card checks in `scripts/verify_level_claim.py`),
Phase 4’s Person’s Page (89/89 CPU/interface checks in `scripts/check_level_page.py`, 11/11 31B card checks in
`scripts/verify_level_page.py`), and Phase 5’s trial of record, owner’s export and volunteer’s re-run (322/322 CPU
checks in `scripts/check_level_trial.py`; on the 31B, 37/37 checks of a dry run in `scripts/verify_level_trial.py`,
which reads no trial of record) were built and card-checked (these counts are as main gave them before pull
request 6; ARCHITECTURE §7 gives them as mended):
- **An edit is a value, pinned to its base.** It can scale, silence, set or add to a column of a layer’s down
  projection, remove a stream direction there (projected off the layer’s quiet mask before dividing by
  `post_feedforward_layernorm`), scale, silence, set or add to a row of `gate_proj` or `up_proj` (edited coordinates
  recomputed, never added to), add a low-rank term on `down_proj`, or compose edits additively (`stack`). Its id is a
  hash of the edit and of every tensor it touches, so it is refused on any other weights. `backend/weight_reader.py`
  reads the checkpoint’s weights, quiet masks, norms, and top writers directly from disk on CPU with no GPU.
- **It is read against the untouched model in the same forward.** The edited model runs in one row of a batch and the
  base in another, beside null rows and control edits (`other_sites`, `other_sites_matched`, `same_site`, or
  `sham_target`), at two row counts, reporting `delivered` stream change alongside next-piece metrics — or across
  greedy, sampled, and thinking-on written generations with loop and cut-off detectors. The model is never left edited.
- **Nothing is lost.** Every reading, refused and void ones too, goes into an append-only ledger. Each line carries
  the hash of the line before it (`level_book.driver_view` / `book --driver-view` provides a redacted aggregate view).
- **Agent’s Door (`level_sandbox.py`, `level_session.py`, `level_mcp.py`).** Off-card CPU Python cells run inside a
  `bwrap` sandbox (no network, no GPU, no `$HOME`, read-only weights, single writable `/work`, memory and wall-clock
  caps) beside budgeted model gates (`capture`, `read`, `write`, `book_view`) returning redacted `driver_view`
  aggregates and ranks in practice, plus blinded in-memory planted-fault and canary sessions. The gates refuse an
  item that holds the wording, or a near wording, of a brief’s kept-back frame or of a retention battery’s question
  (a question put in other words passes); every call is priced against the session’s budget before it runs, a
  claim before it is registered; and the MCP server takes the card’s lock and refuses to start while the page
  server answers or the card lacks the memory to load the model (since pull request 6 the page server takes the
  same lock on a CUDA card unless `THEODOLITE_SKIP_CARD_LOCK` is set, so neither starts while the other runs;
  `scripts/levelling.py` takes none).
  PR 7 adds drivers for the Gemini CLI and the Antigravity CLI (agy), not card-checked: they were
  checked on the CPU (the Gemini CLI against a stub of its model API, agy against mock CLIs) and agy also live against
  a stand-in door with no model, and neither has worked the door on the 31B. agy is refused until the owner accepts
  the three tools it offers every run, and both need prices in the plan under a dollar cap (ARCHITECTURE §7).
- **Claim Gate (`level_briefs.py`, `level_claim.py`).** Versioned briefs separate `practice` and `kept_back` frames
  (two independent halves), register every fresh claim in the ledger **before** post-freeze `os.urandom(32)` draws,
  enforce read-once per edit and a fixed budget `m` under the brief’s name (`20m - 1` controls per family across both
  batch classes; a brief changed after its first reading is refused, and a kept-back frame is read under one brief
  only, whatever a later brief is called), take only the bars a calibration reading wrote into the same book,
  register nothing that is certain to be refused, write the reading’s design (counts, rows, controls, matching
  tolerance) into the registration with none of it the driver’s to set, compare author prompt arms, and issue
  bounded harness-authored Claim Sheets (`lead`, `lead, matched by a cheaper knob`, `no better than control edits`,
  `works but harms`, `did not replicate`, `kept the base`, `void`). Sham edits are the driver’s own: they are
  registered by hash with the rule it declares, checked for shape, and counted against the edit when they move
  the stream less than it does. A sheet says what a reader needs to weigh it: which book the reading was taken in,
  whether the brief’s kept-back frames are published (the four starter briefs’ are, in this repository) and that
  only their wording is guarded, who supplied the sham edits, how far the control edits of a paired reading moved
  the stream (“matched” other sites are matched on what a base pre-pass expects of each over the answer, and the
  sheet says how many were then at least as loud as the edit over the prompt, over the answer and at the last
  position, where the starter briefs read their metric; in a claim reading that count gates nothing, and the
  sentence says that its level takes every one of them as a control and as a separate draw), how many of the
  items that read the kernel read under its item gate (the pronoun brief’s forced answers are text the base
  would not have written: on the 31B every item of it on record is under the gate; the other starter briefs’
  items have not been recorded), and, for a written reading, that its loops were counted by rule and not by a
  person. A prompt arm is said to match an edit when it moves the metric at
  least 0.95 times as far and its near-miss effect is within the registered bar, in size or as a share of its own
  effect; an arm whose near-miss effect was not read matches on its effect alone.

- **Person’s Page (`backend/level_routes.py`, Vault `compare` tab).** Base, edit, and 1–19 control edits heard side by
  side in one batched forward beneath red-italic rubrics, with a two-batch-class paired read-out table, HTTP 409
  refusal while any vault overlay is in force, and model-free replay from the append-only level book (live readings,
  Phase 3 claim verdicts, and Phase 5 trials, a dry run shown as a dry run and never as a verdict).
- **Of record, and out (`backend/level_trial.py`, `backend/level_export.py`, `backend/level_rerun.py`; the owner’s
  command lines).** No agent’s tool and no page route reaches them; a driver given a shell of its own could run the
  scripts, so what keeps a driver from them is the door’s sandbox. A **trial of record** is a fresh reading of a
  brief’s kept-back frames with more asked of it. It is asked what a claim is asked, when its folder is written
  and again when it is read, spends one of the brief’s fresh readings, follows the edit’s claim reading only when
  that read a lead, takes only bars
  whose calibration follows from the book’s own readings, reads the edit in both batch classes with the folded form
  (the change written into the weights) beside the per-row form on every item that read, against matched and
  sham control edits that count for it only when, as read, they changed the stream at least as much as it did:
  over the prompt, over the answer and, for a metric read at the last position, there too (the same-site family
  is not asked this, and what it delivered is printed). Its matched control sets are drawn together to share as
  few sites as the eligible sites allow, and count as no more separate draws than the control edits found to
  share no site with one another, a site being a neuron whatever projection changes it. And it reads a reworded
  set, items written outside the brief’s generator. **As this rule stands no trial is expected to pass it.** A
  stand-in is drawn from a band on both sides of what its neuron is expected to deliver over the answer, so by
  the draw alone some of an edit’s matched control edits are quieter than it; each of those is counted as
  moving as much; and the level allows none of 39 (one of 99) at two fresh readings a brief. In the dry run main
  recorded on 20 November 2026 (at `9d8c836`; the record is in git history), 2 of the known
  answer’s 39 were as loud as it over the answer, where the matching looks, 5 over the prompt, and none at the
  last position, where the metric is read. A pilot on the card (its report is not in this repository) found
  that edit’s loudest neuron to be the loudest of its layer there, so that no other site can match it, and
  found that neither matching where the metric is read nor scaling control edits up to the edit gives a rule
  that can be met within the kernel’s caps. The same edit fails on its harm to near misses as well (0.84
  against a bar of 0.25), whatever its control edits do. As read, no control edit came near it on the metric:
  the largest moved it 0.15 against the edit’s 1.27, and the largest of the 39 same-site ones, whose loudness
  sat at 0.98 to 0.99 of the edit’s at the median, 0.37. What a pass should ask of its control
  edits is the owner’s to settle (ROADMAP §viii); a registration says so before a trial is read, and the
  command line prints it. Nothing certain to be refused is entered in the book, and a trial that is
  entered gets a verdict whatever ends it short of its process being killed: `passed`, `passed, matched by a
  cheaper knob`, `failed` or `void`. It is of record
  only when it is the first thing its process asks of the model, from a clean checkout (nothing changed, nothing
  untracked, no file git is told to overlook, no code where the runner imports from that no commit holds) that
  holds its registration folder committed, run by the files hashed at registration, in the level book, on a seed
  that was drawn, and when the look taken at its registration (with the base alone, at how many separate draws its
  control sets would count as) did not say it could not pass; otherwise it is refused, or read as a dry run when
  that is asked for. The look leaves a line in the book and sees that count alone: it cannot see whether the
  control edits will be as loud as the edit. Each of those is looked at and none is a signature: “read once” and a brief’s budget are facts
  about one book, and the harness cannot see another book, a copy of this one, or this one cut back. Its fresh
  items are drawn from the frames every fresh reading of the brief is drawn from, and the draw keeps out practice
  items only, so they may repeat items an earlier fresh reading read: the reworded set is what stands outside them,
  and once its folder is committed it is as public as that commit. An **export** writes an
  edit as a llama.cpp LoRA adapter outside every git checkout, with keys that say only what was checked: a trial’s
  verdict is read from the trial’s book (never from its result file) and judged again with the registration; the
  base it was compiled against is called the edit’s only as far as it was compared, and a key says that a
  checkpoint is not what llama.cpp serves; an edit rebuilt off the model is exported only when a level book shows
  the kernel has read it. A **re-run kit** asks a volunteer’s card for the home trial’s own
  items, and the intake judges the returned book against the home book, figure by figure. It cannot tell a re-run
  from the home book’s own lines copied into a new book, so a home book stays unpublished until its returns are in.
  Before 21 November 2026 only dry runs had been read on the 31B, all of the known answer (the layer-50 ‘she’ ten).
  The one main recorded at `9d8c836` read `failed`: on the 15 of its 16 fresh items that read, the edit moved the
  metric by −1.27 and −1.26 in the two batch classes, with a near-miss harm of 0.84. Its fresh reading reads `no
  better than control edits`, which there means that every matched control edit was set aside as quieter than the
  edit; none that was read moved the metric as much, and the reason printed with the verdict says so. That edit’s
  verdict has turned on the draw: in twenty-two readings on the card by 20 November it read `works but harms`
  eleven times, `kept the base` five (the claim gate’s name for a reading that cleared no bar, a floor too loud to
  set one by among them), `no better than control edits` four (by a trial’s count of its control edits) and `did
  not replicate` twice.
  On 21 November 2026 pull request 6’s builder registered and read two trials of record of one edit in the owner’s
  level book, with reworded sets it wrote, under `trial-4`, a rule the owner never adopted (way (b) of ROADMAP
  §viii as its builder wrote it: same-site control edits counted when at least 0.85 times as loud as the edit,
  other-site and reworded ones judged as read). Both registration folders stay committed as records
  (`experiments/results/levelling_trials/`); this code refuses to read, verify or judge them, or to export an
  adapter with their verdict. The edit (`6789c6b18fce`) adds one direction to ten columns of layer 42’s down
  projection. Its first trial, on `occupational_gender_debias`, read `failed` at 20:52 UTC. A second brief,
  `occupational_gender_debias_v2`, raised the least effect from 0.75 to 1.5 and has new kept-back frames, near
  misses and reworded set (its practice half is the first brief’s); it was calibrated three minutes after that
  `failed`, and the edit’s second trial on it was registered at 20:59 and read `passed` on its sheet. The review of
  22 November 2026 judged the trials again outside this code, and no committed record holds what it found. With
  main’s `trial-3` judge, the rule in force, the second trial fails: all 119 of its other-site control edits and
  all 59 of its reworded ones were quieter than the edit over the prompt, over the answer or at the last position,
  so each counts as moving as much. With pull request 6’s trial-4 judge and only the least effect swapped, the
  first trial passes at 1.5 and the second fails at 0.75. The edit is 0.45 times one whose claim on the first brief
  read `works but harms`. Both trials were registered with a folded-form floor multiplier of 1.5, where the code’s
  default is 1, and at 1 neither passes at any least effect. What the edit did, as narrowly as the records allow:
  on the second trial’s 15 fresh items (4 of them items its claim reading had already read) it raised
  `log P(' she') − log P(' he')` at the next word-piece by about 2.86 nats as a mean over frames (2.65 over items,
  from about −8.2 to about −5.5, still about 250 to 1 toward ‘he’); on the reworded set by about 1.3, under the
  brief’s least effect of 1.5; near misses and neutral text barely moved. The metric is one-sided: a move past even
  odds counts as gain, and 2 of the 15 items ended further from even odds than they began. The edit was carried in
  from the builder’s scratch (`provenance: carried`), and the book holds no practice reading of it, so the
  harness’s guard on selection never engaged. The builder’s look at what the model then writes (its greedy
  continuation unchanged on 6 of 6 trade stems) is in its own scratch, not in a committed record.
  Since the owner’s ruling of 22 November 2026, as the mends built it, an edit has one trial by its direction, on
  any brief and at any scale: a trial is refused when the book holds, under any brief’s name, a trial of the same
  direction (under any rule, trial-4 included) or a claim of it that read no lead, a rescaled copy being the same
  direction (as built, a negated one too, a copy written with the factors another dose gives it, and a neuron’s
  scale written at another of its projections). On a copy of the owner’s book that refuses the second trial’s
  registration, and the owner’s book can take no further trial of that edit’s direction. No trial of record has been
  read under the rule in force, and no kit has run on another card.

Pull request 6 also gave a blinded session a way to submit one repair, or an explicit abstention
(`propose_repair`), which the owner has not ruled on (ROADMAP §viii), and the blinded benchmark now scores what the
door took. On 21 November 2026 its builder ran sessions on the 31B with scripts of its own, outside the benchmark,
and committed no record of them: the driver of the layer-54 plant abstained, which on a planted fault is a miss; the
layer-50 plant’s `propose_repair` was refused at the session’s 900 s wall cap and the session was stopped, so it did
not run to an end; the cold driver was of one family, claimed something other than the known answer, and its claim
was judged `void`; and a fresh session’s rebuild reproduced a reading’s 416 figures, in the builder’s scratch only.
So the ruling’s cold-driver test (drivers of two families, each reaching the known answer) has still not been run as
the ruling asks, and its rebuild test has been run only outside the repository, with no committed record.

Three of Phase 5’s questions were ruled on by the owner on 19 November 2026 (a re-run repeats the home items; a trial
has a reworded set; a trial spends a fresh reading). What else the build chose there is listed in ROADMAP §viii as
the builder’s (first among them what a trial asks of its control edits: as built it counts a matched control edit
only when it was at least as loud as the edit, which no trial is expected to pass, where a claim reading counts them
all; and that a trial follows only a claim that read a lead). Asked on 20 November 2026 what a pass should ask of its
control edits, the owner answered “Leave it for now”. On 22 November 2026, after pull request 6 had built trial-4,
the owner ruled “trial 3”, so the rule as built is the one in force, and whether to change it is still the
owner’s. The owner also ruled that “an edit has one trial” holds across briefs and up to scale. Later work (Phase 6
driver evaluations) waits on decisions that are the owner’s.

“Levelling” is a working name: as a surveyor levels heights against a benchmark, it reads each edit against the
untouched model. More is in [ARCHITECTURE §7](ARCHITECTURE.md) and [ROADMAP §viii](ROADMAP.md#viii--a-weight-edit-read-against-the-base), and the exploration
that led to it is in [weight_space_exploration](experiments/results/weight_space_exploration/README.md).

## Built on

This instrument is its maker’s synthesis of four recent pieces of research, joined on one mount:
- the **Jacobian lens and J-space** of Gurnee et al. (Anthropic, 2026) [2];
- the **J++ lens** of Ayonrinde & Lindsey (2026) [4], as hosted by Neuronpedia [6];
- **neurons as two-poled units**, from Cammarata’s *Neural Networks Are Already Clean* (Goodfire, 2026) [9];
- the **LARQL / VINDEX** naming of a neuron vault, after Hay (chrishayuk) [10].

The **atlas** follows lyramakesmusic’s [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) [47],
made for Qwen3.5-4B. The census’s idea is neuron-explorer’s: steer each unit and rank its words against
random-direction controls. Parts of the atlas adapt its code and prompts, under its MIT licence ([`NOTICE`](NOTICE)).

Under these lie older foundations: the logit lens [8], the residual-stream view of Elhage et al. [14], linear CKA
[13], activation steering [21, 22], Löwdin’s symmetric orthogonalisation [33], and interp-engine [36], which runs the
model. Goodfire’s neuron atlas of Qwen3.6-35B-A3B [54], published on 10 November 2026 after this atlas went public,
reads neurons from where they fire rather than by pushing them. Bracketed numbers refer to
[`REFERENCES.md`](REFERENCES.md), and the interface tells the same story on its Lineage page.

## What is new here

- **The first public neuron map of a 31B.** Every one of Gemma-4-31B-it’s 1,290,240 feed-forward neurons is on one
  map, with its lens words, weight links and passages. A search on 30 October 2026 found sparse autoencoders for this
  model but no description of its neurons (REFERENCES §J).
- **A census calibrated on the model it reads.** It carries neuron-explorer’s idea to every neuron of a 31B:
  - the random controls are shaped through Gemma’s post-feed-forward norm, so the next layer hears them as loudly as
    a neuron;
  - each word is set against the controls alone, and every neuron gets a p and a layer-wide false-discovery q;
  - words picked on one half of the questions are scored on the other;
  - outside the layers it was first calibrated on, the dose is a count of each neuron’s own write ceiling, set layer by
    layer by how far it moves the next word-piece;
  - each layer there is trialled first, against gates registered before the trial ran.
- **A census anyone with a card can take.** A volunteer kit is pinned to a tag. Its canary batch is compared bit for
  bit with the home card’s, other kinds of card are judged against bars measured before any run, and claims and
  results go on a public board.
- **The audit’s mends, built in.** The first study’s audit of its own instrument found nine faults, and the mends for
  seven are built in. The nudge is sized to the stream’s own strength with its quiet dimensions masked. The landmark
  frame is order-free, and threads are ranked by the direct path. A label gate comes from that study’s validation of
  neuron labels.
- **Weight links put to the test.** Links are weighed through Gemma 4’s norms and layer scalars, and were measured
  under pushes on the model. That check showed they are real by size, and where a link’s sign can be trusted.
- **Pushes that leave the instrument.** A push, or a recipe of several neurons, is written as a llama.cpp control
  vector that adds the same push.
- **Two readers on every page**, and a plain vocabulary, each word mapped to its name in the literature.

## Further reading

- **The atlas:** [ATLAS](docs/ATLAS.md) (the atlas and the census at length).
- **The pages:** [INSTRUMENT](docs/INSTRUMENT.md) (the pages leaf by leaf) and [SETUP](docs/SETUP.md) (running it in full).
- **Trust and findings:** [TRUST](docs/TRUST.md) (faults and cautions) and [FINDINGS](docs/FINDINGS.md) (the first
  study, of whether Gemma 4 is “female-coded”: it is not, by default; its report is
  [experiments/REPORT.md](experiments/REPORT.md)).
- **The census and the code:** [VOLUNTEERS](VOLUNTEERS.md), [ROADMAP](ROADMAP.md), [ARCHITECTURE](ARCHITECTURE.md)
  (the tensor contracts), [COMPUTE](COMPUTE.md) (what each job costs) and [REFERENCES](REFERENCES.md).

<br>

## How to cite

If Theodolite, its atlas or its census helps your work, please cite it:

> Juul, E. (2026). *Theodolite: an atlas and census of the neurons of Gemma-4-31B-it, and the instrument to read them
> with* [Software]. https://github.com/whp199/theodolite

```bibtex
@software{juul_theodolite_2026,
  author = {Juul, Eric},
  title  = {Theodolite: an atlas and census of the neurons of Gemma-4-31B-it, and the instrument to read them with},
  year   = {2026},
  url    = {https://github.com/whp199/theodolite}
}
```

`CITATION.cff` holds the same, and GitHub offers it under “Cite this repository”. When you use a census layer, say
which: most of them were taken by volunteers on their own cards, and
[the board](https://github.com/whp199/theodolite/issues/1) names who took each. Please cite the work Theodolite is
built on as well ([REFERENCES](REFERENCES.md)).

<br>

## Makers’ marks

<p align="center">
  <img src="docs/readme/cartouche.png" width="420"
       alt="An engraved maker's cartouche: Theodolite, made after Gurnee et al., Ayonrinde and Lindsey, Cammarata, and Hay, with the logit lens, linear CKA and activation steering, for Gemma-4-31B-it, MMXXVI.">
</p>

**Provenance and AI assistance.** The experiments, analyses and reports in `experiments/` were designed, run and
written by Claude Code agents (Anthropic), directed by the repository owner: parallel experiment agents, independent
adversarial verifiers, and a separate lens-refit session. Every headline claim in `experiments/REPORT.md` was re-tested
by an independent verifier. The verifiers that read a lens used the same base-fitted ones ([FINDINGS](docs/FINDINGS.md)),
and the claims that failed are listed in §4 of that report. The exception is what an Antigravity (Gemini) agent made
in pull request 6: the two Levelling trial folders in `experiments/results/levelling_trials/`, which it registered
and whose trials it read, and the three session transcripts of its card run of 21 November 2026 in
`experiments/results/levelling/transcripts/` (`lvl_20261121T1741…`). They stay committed unchanged; Claude Code
reviewed and mended the pull request.

**License.** Code is licensed under the **Apache License 2.0** ([`LICENSE`](LICENSE)); third-party attributions and
exceptions are in [`NOTICE`](NOTICE). This repository does **not** redistribute a model’s weights as a whole, lens tensors
or third-party corpora. Eleven small files under `experiments/results/` (two control vectors, nine LoRA adapters: the
records of two measured runs) hold numbers derived from Gemma 4’s weights and are passed on under its Apache License
2.0: see [`NOTICE`](NOTICE). Gemma is a trademark of Google LLC; this project is independent research, not affiliated with or endorsed by Google.

<br>

<p align="center"><sub>◆</sub></p>
