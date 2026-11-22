# The instrument

*What Theodolite is and its pages leaf by leaf: the instrument, a layer’s leaf, a neuron’s folio, landmarks, the nudge, the vault and the atlas, neuron-explorer’s features set beside the atlas’s, the instrument’s words and what they name in the literature, sightings, how it writes, and saved questions.*

<sub>Moved out of the [README](../README.md) on 16 November 2026, word for word but for its links and headings, when the README was cut down. Layers count 1–60, as the pages count them; the census board and the commands count 0–59.</sub>

## What it is

Between a question and its answer, a language model carries each word up through all of its layers, and little of that
country has been mapped. **Theodolite** lets you follow a single word as it rises through the sixty layers of
**Gemma-4-31B-it**, and read, at every layer, what a lens makes of it: sometimes words you never wrote, in languages you
never used.

The repository holds two things.

- **The instrument** (`backend/`, `frontend/`). A FastAPI server and a React interface that read Gemma-4-31B-it (QAT W4A16)
  through the J++ lens, fitted on the base model, and the plain logit lens: the path of a word through the layers, what
  each layer gathers from the words before and recalls from memory, the neurons that fire, the threads between them, a
  survey frame of three landmark words, a nudge on the model's stream, a vault of neurons you can search and edit, and
  an atlas of all 1,290,240 neurons at once, with a census of what a layer’s neurons do when pushed ([ATLAS.md](ATLAS.md)) and a scientist to ask.
  The tensor contracts are in [`ARCHITECTURE.md`](../ARCHITECTURE.md).
- **The study** (`experiments/`). Is Gemma 4 "female-coded"? Seven experiments, eight adversarial verifiers, a
  neuron-label validation study and a lens refit on the instruct model: 314 GPU jobs through a shared lab daemon.
  **Start with [`experiments/REPORT.md`](../experiments/REPORT.md).**

<p align="center">
  <img src="readme/two-readings.png" width="100%"
       alt="Two ways to read the instrument: rubrics, red directions for those beginning, switched on when you arrive; and the apparatus, the formulas, raw values, sources and every control, opened beneath a double gilt rule for scholars.">
</p>

Every page can be read two ways. **Rubrics**, the red italic lines, are directions for a first reading, as scribes and early
printers set their headings and directions apart in red. **The apparatus**, as in a critical edition, opens beneath each part
of a page under a double gilt rule: the technical name, the formula, the raw values, the sources and every control the
instrument has. Both are switches at the top of every page; nothing is hidden from either reader.

<br>
## The instrument, leaf by leaf

<p align="center">
  <img src="readme/instrument.png" width="100%"
       alt="The instrument page: sixty engraved rings with the path of the word I winding up from layer 1 to layer 60, labelled computer, HUAWEI, bot and am at several layers; beside it, the leaf of layer 36 with its top words through the lens and a bar splitting the layer's change into 37 percent gathered and 63 percent recalled.">
</p>

**The instrument.** One ring for each layer, the first at the bottom. The dark thread is the path of the word "I", in
the reply "Hi! I", as it rises; beside it stand the words a lens makes of it at a few layers. That lens is J++, fitted
on the base model: the head of every page names it, the opening page says once among its cautions what it was fitted on,
and a red line beside each layer's words says what the reading can bear there. The gilded rings, every sixth, are the
layers that look back over the whole conversation. On the right is the leaf of the chosen layer: its likeliest words
through the lens, and how much of its change was **gathered** from the words before (attention, lapis) or **recalled**
from memory (the feed-forward block, gilt). The picture was taken before the label gate: the leaf now holds back the
words of the layer-36 neuron shown at its foot, as it holds back every neuron's words below layer 41, and keeps them in
the apparatus ([TRUST.md](TRUST.md)).

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="readme/layer-leaf.png" alt="A layer's leaf: the numeral 36 at the hub of an engraved dial, with its bearing of 14 degrees and elevation of 52 degrees against the landmark words she, he and it, and a paragraph saying the model has not yet settled on a word.">
      <p><b>A layer’s leaf.</b> Where the stream points at one layer, read through the lens as a bearing round the horizon and
      an elevation toward the third landmark, and the lens’s guess at what the model would say if it stopped there.</p>
    </td>
    <td width="50%" valign="top">
      <img src="readme/neuron-folio.png" alt="A neuron's folio for neuron 15491 at layer 54, titled the I am neuron, with the particulars of the sighting it fired in.">
      <p><b>A neuron’s folio.</b> One of the 21,504 neurons in a layer: its two faces, what each writes through the lens and
      through the naked eye, and the later neurons that read it. Its label can be put to the test: the neuron turned up
      at one word-piece, or on the study’s 36 texts, and the words whose odds rose most, read from the model’s own odds,
      not the lens.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="readme/landmarks-frame.png" alt="The survey frame drawn as an engraved armillary: the landmark words she, he and it as axes, the stream at layer 36 as the lens reads it, marked with its bearing and elevation, and beside it the angles between each pair of words.">
      <p><b>Landmarks.</b> Three words fix a frame, as a surveyor sights known landmarks. Each layer’s stream, carried through
      the lens, is read as a bearing and an elevation against them, with the angles between the words beside it.</p>
    </td>
    <td width="50%" valign="top">
      <img src="readme/nudge-dial.png" alt="The strength dial of the nudge, from times 0.001 to times 10 of the stream's own strength, clear up to times 0.01, gilded where the study's fluent nudge sat, and hatched red where it collapses, with the old defaults of 280 and 650 marked in the red.">
      <p><b>The nudge.</b> A small push on the model’s stream while it reads, measured as a share of the stream’s own
      strength, with the old defaults marked where the study found they break the model.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="readme/vault.png" alt="The vault, where the model keeps its 1,290,240 neurons, with an engraved protractor whose pointer rests on layer 51 seeking she.">
      <p><b>The vault.</b> The catalogue of the model’s 1,290,240 neurons: look one up by the words it writes, change one, and
      read back what was asked before.</p>
    </td>
    <td width="50%" valign="top">
      <p><b>Also in the instrument</b></p>
      <p><b>Lexicon</b>: every word of the instrument in plain terms, with its formula, and the cautions on where it can
      mislead you. <b>Lineage</b>: the research it is made from, pillar by pillar, each with what the study found about it.</p>
      <p><sub>Every view above is a live sighting from the running server: “Tell me about yourself.”, its answer begun
      “Hi! I”, sent as a chat. They were taken with the reading options as first built (a frame squared in order, neurons
      ranked by |a|, threads by C), before the interface sent the mends by default, so the landmarks frame and the folio’s
      ranking and wording differ from what the interface now shows by default. With no server running, the pages show the
      study’s recorded sighting (E6) and say so.</sub></p>
    </td>
  </tr>
</table>

**The atlas.** Every neuron of every layer on one map (`#/atlas`): sixty sheets of 21,504 dots, each layer’s attention
heads drawn as lines before its sheet, the neurons that write alike side by side. Open a neuron and the atlas shows its
words (through the lens, held back by the label gate below layer 41 as on every page, and beside them its *steer lens*:
which next word-pieces a push along its write direction actually raises, a push larger than the neuron itself can write
at most layers), its **web** of weight links hop by hop with the words of what could drive it up and down and what it
could drive when it fires each way, the heads that read it and feed it, its **census** answers (the model answering
fixed questions with this neuron pushed at several doses, ranked against pushes along random directions), and a
**push**: ask anything and read three answers at once, the neuron pushed up, left alone and pushed down. Run a text to
see which neurons fire hardest on each word-piece, search every neuron’s words, search every census at once for the
neurons whose push brought a word (each layer’s random directions searched beside them, so the list says how many of its
hits chance alone would give), lay a layer’s census answers out by what they say, and keep a name and a note beside a
neuron. Every layer with a census is inked on the map at once, a strip of bars shows how much of each layer its census
singles out, and the neurons opened lately stay within a click. Its **passages** keep each face’s strongest windows of a corpus
whole, with what the face reads and what follows it (gathered on the 31B over two million word-pieces, not yet validated
as labels). A **scientist** in the side panel reads the atlas with the same tools, runs pushes and texts, names what it
finds and points at it on the map; it runs on Claude Code (the default), Codex or Gemini where they are installed
(Gemini through Antigravity CLI unless you pick the Gemini CLI beside it), or on any OpenAI-compatible
endpoint: LM Studio or Ollama on this machine, OpenRouter, or one you name.

The atlas follows [lyramakesmusic’s neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) [47], made for
Qwen3.5-4B in the wake of Cammarata’s neuron poles [9], and has at least its functionality for Gemma-4-31B-it:

| neuron-explorer (Qwen3.5-4B) | the atlas (Gemma-4-31B-it) |
|---|---|
| a 3D map of every MLP neuron, attention and MLP per layer; hover a neuron for its lenses, open it, view a layer’s sheet face-on; a `writes` layout | the same for 60 × 21,504 neurons in one draw call, with the q / k / v / o heads on each layer’s slab, lit by role for an open neuron and by their write on a run text; the `writes` layout (UMAP where installed, else PCA) |
| the open neuron’s connection web, upstream and downstream, with its words pooled | the same, its links weighed through Gemma 4’s pre- and post-feed-forward norms and its layer scalars and ranked by how strongly they arrive, their signs read one consistent way (from rest, through GeGLU’s gate: what drives it up or down, what it drives when it fires up or down) where the reference’s mix two, and, with the passages built, read again where each reader fires hardest on a face (its strongest stored passages, not where it sits at an ordinary word-piece), the details marking where the two disagree. Measured on the 31B (the night of 28–29 October 2026, at `main_rel`’s four census questions’ last word-piece, a mark of the chat template, where about nine readers in ten sat with the gate shut), neither sign as served held: under the push on the writer’s gate face the sign from rest agreed with the change at 0.34 of links and the sign in use as served at 0.42 (medians over layers), both below chance; scored offline where each reader sat, the sign in use agreed at 0.84. The links are real by size ([TRUST.md](TRUST.md)) |
| the list: units ranked by word-cluster mass against random-direction controls | the census ranking: the reference's statistic and its `main_rel` questions and doses (not calibrated on Gemma, and said so), beside a design of its own and the default since 29 October 2026, `break_rel` (six questions at ±1 and ±2 of the stream, read as two halves of three, each word set against the controls alone and each answer’s own question’s unpushed words left out), whose doses are calibrated at layers 41–51 alone (its trial of 29 October 2026; ARCHITECTURE §4.7) and whose pages leave out the words random pushes bring as hard at ±2, its controls shaped through Gemma’s post-feed-forward norm so the next layer hears them as loudly as a neuron, each rank with a p and the layer’s false-discovery q beside how many would beat every control by chance; taken on the instrument’s own runtime, resumable, batched by question |
| `lens` search over the steer and logit lenses | search over the lens, the naked eye and the steer lens, every lens label under the label gate; and a search of every layer’s census at once, by the words a push brought, each layer’s random controls searched the same way as its null |
| `fires`: run your text, filter neurons by activation over their usual range, token paths | “run a text”: the same filters (layers, sign, score, words), the links among what fires at a word-piece, the heads lit by their write |
| steered rollouts per prompt and dose | the census tab |
| a three-column multi-turn steered chat (−, 0, +), add or ×, “reads you” | the push tab, the same, greedy or sampled with the instrument’s own sampler, the three voices drawing alike; each turn says how it was written |
| lens, connections, a name | the details: both faces under the gate, the steer lens, the activation range with GeGLU’s four sign regimes, wiring and head tables; a name and a note kept on the server; and each face’s stored passages in a tab of their own, with the regime at each and the word-pieces each face reads and that follow, which the reference does not keep |
| embeddings: UMAP of a layer’s rollouts by neuron, prompt, dose or kind | the spread, the same colourings |
| attention heads with q / k / v / o roles; head tools | the same, with each head’s profile over eight texts, its pattern on a text and the heads driving a neuron at a word-piece |
| a Claude chat (the `claude` CLI with the explorer as an MCP server), Codex as an option, past chats, attachments, edit and reroll, “explain” | the same with 35 tools on Claude Code (the default), Codex or Gemini (Antigravity CLI, `agy acp`, unless the Gemini CLI, `gemini --acp`, is picked), plus any OpenAI-compatible endpoint (the server runs the tool loop); an Antigravity chat has no edit or reroll; the tools count layers as the pages do |
| the sweep’s tools: neuron stats, steer lens, rank, depth report, word shift, merge | `scripts/atlas_build.py` and `scripts/census_run.py` on the GPU, `scripts/census_tools.py` (rank with the word shift, merge, depth, spread) |
| — | a CPU stand-in of Gemma 4 for every check and for trying the page without the weights (`scripts/atlas_stand_in.py`) |

The sweep’s HF-against-vLLM check and its steered prefill captures are not carried over: the atlas runs on the
instrument’s one runtime. Parts of the atlas adapt neuron-explorer’s code and prompts, under its MIT licence; its notice is
in [`NOTICE`](../NOTICE).

**The instrument’s words, and what they name**

| In the instrument | In the literature |
|---|---|
| a **sighting** | the stream at one word-piece, read through all 60 layers; one sweep gives (as the interface asks by default) one per word-piece it writes, read where it chose it: the first where it read your last word-piece, each later one where it read the piece before |
| **layers 1–60** | decoder layers, counted from 1 (the code counts 0–59) |
| **the lens** · **the naked eye** | the J++ lens readout `unembed(h·Jₗᵀ)`, with Jₗ fitted on the base model `google/gemma-4-31B` · the logit lens `unembed(h)` |
| **gathering** · **remembering** | the attention write · the feed-forward (GeGLU) write, with Gemma 4’s layer scalar |
| a neuron’s **two faces** | the positive and negative poles of a GeGLU neuron (Cammarata 2026) |
| **threads** | reader filaments: a later neuron whose gate reads an earlier neuron’s write |
| **landmarks** | the survey frame: three directions, usually words, squared into axes |
| **the nudge** | additive activation steering on the residual stream |
| **the vault** | a LARQL/VINDEX-style catalogue of neurons, queried in Python |
| **how it writes** | decoding: greedy (its likeliest word-piece) by default, or sampled with a presence penalty, temperature, top-k and top-p |
| **the atlas** | a neuron explorer: every feed-forward neuron on one map (after neuron-explorer [47]) |
| **a push** · **the three voices** | steering along one neuron’s write direction, `w_post ⊙ W_down[:, n]`, by a share of the stream’s median length leaving its layer · the same answer written pushed up, unpushed and pushed down |
| **the census** | a steering sweep of every neuron of a layer on fixed prompts, ranked by word-cluster mass against random-direction controls |
| **the steer lens** | the mean change in the next token’s log-probability under a push sized to the neuron’s layer (Δ log p, with its KL) |
| **the web** | the neuron’s connection cone: direct-path weight links chained hop by hop |
| **the scientist** | an LLM agent with the atlas’s tools (MCP, or OpenAI-style function calling) |

**A sighting of each word-piece.** The model first writes its answer, then the instrument reads the whole text once more and
follows a word-piece up through the sixty layers. By default it follows each word-piece the model wrote, at most 64: the
sighting of word-piece *k* is taken where the model chose it, at position *P* − 1 + *k* after the question's *P* word-pieces,
where the stream had just read the word-piece before it. The first is read where your words end, as every sighting once was,
and it is the one shown first; choose another word-piece of the answer to see its own. The last word-piece it wrote is chosen
but never read. Two word-pieces of one answer are not a matched pair: they read different word-pieces after different amounts
of text, and the lens leans differently at different kinds of word; only layer 60, the model’s own odds, compares cleanly from
one to the next. Inside its own answers the study found the lens’s lead over the naked eye a third of its lead on prose at
layers 31 to 40, and gone from layer 41 (E3). Compare the same word-piece across matched sightings instead.

**How it writes.** Under the question, the band *how it writes* sets how the answer is written before it is sighted: its
likeliest word-piece every time, as the study wrote the answers it checked the lens on, or drawn by chance as chat apps do,
with a repeat penalty, a temperature, top-k and top-p, applied in that order, and a seed (blank, the server draws one and
sends it back); and how long it may run, 1 to 256 word-pieces, stopping sooner at one of the model’s stop marks: the mark
that ends its turn, or another (the end of the text, or a control mark a strong nudge can force), which the pages name rather
than call an end of turn. Greedy is the default everywhere, including the sighting taken when the page first connects. A
drawn answer is one draw of many it could give, and the pages say so; the sampler changes which words are written, never how
a word is read. What the pages say of how an answer was written comes from the server’s echo of it (`generation`, and
`chosen` for each word-piece), never from the form. The methods are [42]–[44] in [`REFERENCES.md`](../REFERENCES.md).

**Saved questions and system prompts.** Nothing you write is kept unless you save it. *save this*, under the question or the
system prompt, keeps it under a name you give; the count beside it opens the list, and choosing an entry fills the boxes
(a question restores its chat setting too) without sending anything. They are kept by the instrument’s server, in
`$THEODOLITE_SAVED_PROMPTS` if it is set, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/saved_prompts.json`, outside the
repository, so every checkout and every browser shares one list (`GET`, `POST` and `DELETE /api/saved_prompts`; refused to
pages from any other host than this machine).
