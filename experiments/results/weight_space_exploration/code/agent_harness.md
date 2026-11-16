# The scientist: the agent harness that already exists

Reader's report for the weight-space harness design. Read-only pass over the repository at
`(the repository root)` on 14 November 2026 (HEAD `e6ac64f`). Nothing was run on the GPU and no server was
started. Every statement about behaviour was checked in the code unless it says "document says". Paths are relative to
the repository root.

Files read in full: `backend/scientist_mcp.py` (104 lines), `backend/mcp_scientist.py` (1,609), `backend/scientist_tools.py`
(894), `backend/scientist_chat.py` (1,576), `backend/scientist_prompt.md` (399), `backend/scientist_methodology.md` (158),
`ARCHITECTURE.md` §4.8 (lines 1053-1148). Read in part: the docstrings of `scripts/check_scientist.py`,
`scripts/verify_scientist_cli.py`, `scripts/check_hypothesis_test.py`; the scientist paragraphs of `README.md`,
`ROADMAP.md`, `COMPUTE.md`; the parts of `backend/server.py`, `backend/atlas.py`, `backend/label_gate.py`,
`backend/field_notes.py`, `backend/pattern_tests.py`, `backend/local_only.py` and `experiments/lab/labd.py` that the
scientist touches.

---

## 0. The short version

- There are two tool surfaces with confusingly similar file names. `backend/mcp_scientist.py` (first commit 16 October
  2026) is a FastMCP server of **5** tools that work directly on a loaded runtime, count layers from 0, and include the
  **only weight-editing tool** an agent has (`query_larql_vault`). `backend/scientist_tools.py` (first commit 27 October
  2026) is **35** tools that are thin HTTP proxies onto the atlas's routes, count layers from 1, and include **no
  weight edit, no raw weight read and no code execution**. The chat a person uses (the "scientist" tab) runs only the
  second surface.
- A driver model plugs in one of three ways today: Claude Code or Codex as a child process that is handed the stdio
  MCP server `backend/scientist_mcp.py`, or any OpenAI-compatible Chat Completions endpoint with tool calling, for
  which the server itself runs the tool loop. Nothing else: no native Anthropic Messages backend, no Responses API.
- The loop is conversational and person-in-the-loop. There is no budget, no experiment ledger, no job queue, no
  per-call cost record and no registration of a hypothesis before a test. Every live reading waits on one
  `asyncio.Lock` for the one GPU.
- The method documents are strong on what words may be trusted (the label gate, lens cautions, layer counting) and
  the statistics exist, but the best statistics (permutation p, paired sign-flip, Benjamini-Hochberg) live on the
  *older* surface and are not among the chat scientist's 35 tools.
- The pieces a weight-space harness could reuse are: the tool registry and its three transports, the proposal
  (human-keeps) pattern, the blind test's server-held truth and fresh-conversation reader, the reach/sandbox rules and
  their checks, the work folder with its no-script file serving, and the stale/restart mechanism for changed tools.

---

## 1. The tools

### 1.1 Two surfaces, and how they relate

| | `backend/mcp_scientist.py` | `backend/scientist_tools.py` |
|---|---|---|
| What it is | FastMCP server "Theodolite Circuit Scientist" (`mcp_scientist.py:81`), tools are Python functions on a `TheodoliteRuntime` and a `LarqlVaultEngine` | A list of tool schemas (`TOOLS`, `scientist_tools.py:191`) and one dispatcher (`call`, `:534`) that turns each call into `api.get` / `api.post` on the server's routes |
| Tools | 5 (`@mcp.tool()` at lines 174, 206, 949, 1201, 1567) | 35 (counted by loading the module; see §1.3) |
| Layer count | from 0 (refusals say "layers count from 0 to 59 here (the pages count them 1 to 60)", `:1017-1019`) | from 1, converted by `from_page` / `to_page` (`:40-79`) |
| Needs torch / the model | yes: imports torch, fastmcp, interp_engine (`:58-63`) | no: standard library only (`:7`) |
| Who calls it | (a) `backend/server.py` imports four of the functions and serves them as HTTP routes under `_GPU_LOCK` (`server.py:65-71, 1470-1545`), plus the vault at `POST /api/larql` (`server.py:1460-1467`) and WebSocket actions; the neuron folio page uses these. (b) Run alone, `python -m backend.mcp_scientist`, it **loads its own 31B runtime** (`mcp_scientist.py:117-139, 1602-1609`) | (a) `backend/scientist_mcp.py`, a stdio MCP server Claude Code and Codex are started with; (b) `backend/scientist_chat.py`'s own tool loop for OpenAI-compatible endpoints, through `InProcessApi` (`scientist_chat.py:868-897`) |
| Weight edits | yes: `query_larql_vault` | none |
| Status | Still live code: its functions back four HTTP routes and the folio, and it gained paired tests and BH correction on 12 November 2026 (commit `3ae17e2`). As a *standalone MCP server* it is the older path: it cannot run beside the instrument's server on one card ("Run one GPU process at a time", `README.md:485`). | The current scientist: the chat in the atlas's side panel uses only this (`scientist_chat.py:84, 1058, 1085`). Last changed 13 November 2026. |

The one bridge between them: the chat scientist's `causal_check` tool posts to `/api/scientist/causal_check`
(`scientist_tools.py:814-818`), which is `mcp_scientist.causal_spot_check` behind the server's lock. It passes only
`layer, neuron_idx, polarity, contexts="given", prompt, chat, k`; the older tool's `contexts="study"`, `write`,
`scale_ref`, `floor`, `top_n`, `track_ids` and `token_ids` are not reachable from the chat scientist's tool.

Not reachable from the chat scientist's tools at all: `inspect_pole`, `trace_downstream_readers`,
`test_motif_hypothesis` (the statistics), `query_larql_vault` (the vault). A scientist with a shell (the "home" reach,
§2) could reach their HTTP routes with a script, since the server's guard passes any script run on this machine by the
same account (`backend/local_only.py:1-9`), but the prompt's list of "useful routes" does not name them
(`scientist_prompt.md:346-353`).

### 1.2 The FastMCP surface (`backend/mcp_scientist.py`), 5 tools, layers from 0

| Tool | Arguments | Returns | Cost |
|---|---|---|---|
| `inspect_pole` (`:176`) | `layer, neuron_idx, polarity=1.0` | Both faces' words through the J++ lens and the naked eye, gate trigger words, each face's `label_check`, then `label_gate`, `lens` | No forward pass through the model: a weight read and two unembeds, run in a worker thread (`:193-203`). Wall time not established. |
| `trace_downstream_readers` (`:208`) | `layer, neuron_idx, prompt, polarity, max_layer_hop=6, top_k=10, score="coupling"\|"direct", chat, system, assistant_prefix, token_ids` (1 to 2,048 ids, `engine_runtime.py:493`) | Top readers in the next `max_layer_hop` layers on that prompt, with coupling score, direct-path gate and activation estimates, lens words, naked-eye words and a label check per reader | One captured forward of the prompt (`capture_rows`, `:271`), then 25 poles per downstream layer (`:311`). Wall time not established. |
| `test_motif_hypothesis` (`:951`) | `layer, neuron_idx, pos_prompts, neg_prompts, chat, system, paired=False, also=[[layer, neuron_idx], ...]` | Per-sentence activations (last word-piece and strongest), the legacy Cohen's d verdict, and the tests against chance: `selectivity`, `selectivity_paired`, `selectivity_peak`, `selectivity_either`, `correction` (§5.4) | **One forward per sentence, unbatched** (`:1054-1056`), whatever the number of neurons; up to 512 neurons a call (`:425`). The statistics are "seconds, for hundreds of neurons" (`:1076`). |
| `causal_spot_check` (`:1203`) | `layer, neuron_idx, polarity, contexts="given"\|"study", token_ids \| prompt, chat, system, assistant_prefix, k=10 (≤200), write="activation"\|"frozen_norm", scale_ref="study"\|"this_text", floor=True, top_n=10 (≤50), track_ids (≤16)` | Per face: the push's delta, KL, words that rose and fell, linearity, label overlap P@10; `antipodal`, `floor`, `forwards`, `vault_edits`, **`timing_ms` {scale_ms, check_ms, total_ms}** (`:1550-1554`) | One forward holding the pushed rows and an untouched row, halved on out-of-memory; with `scale_ref="study"` a table of max \|a\| over the study's 36 texts is computed once per set of vault edits and cached (`:1175-1191`). No wall time is recorded in the documents read; the reply carries its own. |
| `query_larql_vault` (`:1569`) | `lql_query` (a LARQL statement), `voice` (optional: `{prompt, chat, system, assistant_prefix, max_new_tokens 1-64}`) | The vault engine's reply; for an edit with a voice, both answers, KL at the first word-piece and where they part | Statements named in the docstring: `SELECT * FROM EDGES\|POLES NEAREST`, `DESCRIBE`, `DESCRIBE POLE`, `UPDATE`, `INSERT`, `DELETE`, `ROLLBACK` (`:1570-1589`). An edit is "one column of one layer's down_proj, rank one, with nothing solved for" (`:1570-1572`). One default heard edit took **12.6 s** on the GPU (`server.py:1455`). |

Every one of the five runs whole under one lock when the module is standalone, and under no lock of its own when the
server bound the runtime, because the server's routes hold `_GPU_LOCK` (`mcp_scientist.py:142-171`, `bind_runtime` `:93-103`).

### 1.3 The chat scientist's surface (`backend/scientist_tools.py`), 35 tools, layers from 1

Counted by loading the module with plain Python (it imports only `json` and `typing`): `len(TOOLS) == 35`. The
tools/list payload is 39,051 bytes of JSON; the descriptions alone are 20,140 characters. **The documents disagree with
the code:** `ARCHITECTURE.md:43` and `README.md:239` say 30 tools, `ARCHITECTURE.md:1132` says 32; the three pattern-test
tools added on 13 November (`test_by_writing`, `blind_test`, `blind_test_answer`) are missing from §4.8's list.

Costs below are of three kinds. "Index" is a read of a file the server holds, with no model run. "Live" runs the model
under the one GPU lock. "Screen" posts an event to the person's page. Wall times are given only where a document
records one.

**Reading the atlas (index, no model run unless noted)**

| Tool | Arguments | Returns |
|---|---|---|
| `overview` | none | Model shape, which indexes are built, census layers, push units, the label gate's card, the lens's caution, counts of named neurons and heads, proposals awaiting the person (`:539-562`) |
| `neuron` | `layer, n` | Name, note, who wrote each, kept tests, both faces' lens and naked-eye words with the gate's verdict, steer lens, activation range, ceiling, census rank, 24 strongest weight links each way (`:563-585`). Until the wiring index is built the links are read live from the weights, "a few seconds on the GPU" (`scientist_methodology.md:157-158`). |
| `passages` | `layer, n, face, k=16` | A face's strongest stored passages with the peak marked, word-piece counts at and after the peak; **fetched with `fold="fit"`**, so the half kept back for the blind test is not shown (`:588`) |
| `rollouts` | `layer, n, max_chars=500` | The census record: what the model wrote when the neuron was pushed, per question and dose, beside unpushed answers, with p, q and the run's card |
| `cone` | `layer, n, k=2, limit=120, model="rest"\|"use"` | Weight links chained hop by hop |
| `web_summary` | `layer, n, first=5, k=2, model` | The cone's words pooled, downstream and upstream |
| `search_lens` | `query, limit=40` | Neurons with a face carrying the word |
| `search_census` | `query, limit=40` | Neurons whose census words match, with the controls searched the same way as the null (`against_chance`) |
| `top_units` | `layer, limit=30` | A layer's census ranking with its card |
| `names` | none | Every kept name, which were a scientist's, and what awaits the person |
| `head` | `layer, head` | A head's q / k / v / o neurons from the weights and its profile over eight reference texts |
| `neuron_heads` | `layer, n` | The heads a neuron meets, by role |

**Proposals (writes that a person must keep)**

| Tool | Arguments | Returns |
|---|---|---|
| `set_name` | `layer, n, name` | A proposal recorded with the chat's id; nothing reads it until the person keeps it (`:685-703`) |
| `write_note` | `layer, n, note` | The same for a note |

**Live readings (each waits for the GPU)**

| Tool | Arguments | Returns | Recorded cost |
|---|---|---|---|
| `steer` | `layer, n` or `neurons` (≤32 pushes) or `candidates` (≤12 sets); `prompt` or `prompts` (≤8); `dose, mode add\|mul\|act, max_new=150, spare_bos, temperature, top_k, top_p, presence_penalty, seed, samples (≤8), both` | The answer written pushed up, unpushed and pushed down, as whole texts | Three rows of 80 word-pieces: **8.3 s**; five prompts (15 rows): **9.2 s**; six candidates (13 rows): **8.8 s**, with the `compiled` unpacker on the 5090, 11 November 2026 (`ARCHITECTURE.md:920`). The same three rows took **50 to 54 s** on the `reference` unpacker that day (`COMPUTE.md:63`). Rows hold at most 6,144 word-pieces together. |
| `logprobs` | `text, chat=True, prefix, k=15, targets, neurons, heads, spare_bos` | Next word-piece log-probabilities; with pushes or head scales both rows, their KL and what gained and lost | One pass holding both rows (tool description, `:365`). Wall time not established. |
| `activations` | `layer, n, text, chat, neurons, heads, spare_bos` | One neuron's activation at every unit of a text, optionally with other neurons pushed or heads scaled in the same pass | One pass, at most 2,048 positions (`ARCHITECTURE.md:923`). Wall time not established. |
| `compare` | `neurons` (≤300), `texts` (≤24), `chat` | Per neuron and text, the strongest activations and peak score | One forward per text. |
| `top_firing` | `text, token, at, limit=30, layers, chat` | The neurons firing hardest on a text against their usual range | Live on one text; wall time not established. |
| `steer_lens` | `layer, n` | One neuron's steer lens read now (± the layer's dose over four contexts) | Live. Wall time not established. |
| `test_by_writing` | `layer, n, face, pattern, fire (1-12 texts), quiet (≤12), behind, drop (1-64, default 8), chat` | How many written texts fired the face, how many near misses did, and `own` (the face's three strongest passages read alone); kept on the face | One read per text. |
| `blind_test` | `layer, n, face, pattern, pairs (4-12, default 10), quiet near\|anywhere, before (16-256, default 60), profile` | A blind reader's score over kept-back passages, with p; kept on the face | GPU to read each quiet control, then one call to an endpoint in a fresh conversation. |
| `blind_test_answer` | `test, picks` | The same result when the scientist had to answer the pairs itself | No model run. |
| `causal_check` | `layer, n, face, prompt, chat, k=10` | The older surface's `causal_spot_check` reply, in page layers | "Slower than logprobs" (`scientist_tools.py:442`). |
| `attention` | `layer, head, text, token, at, chat` | One head's attention weights at a unit | Live on one text; wall time not established. |
| `attn_drive` | `layer, n, text, token, at, chat, top=8` | Which heads, blocks and earlier neurons drive a neuron at a unit (linearised) | Live on one text; how many passes and the wall time were not established. |

**An export**

| Tool | Arguments | Returns |
|---|---|---|
| `export_control_vector` | `neurons` (≤32, each `layer, n, dose`), `name` | A llama.cpp control-vector GGUF written to `<atlas folder>/exports/`, with the command to load it |

**The person's screen (8 tools)**: `select_neuron`, `select_head`, `open_tab`, `camera`, `highlight_neurons`,
`annotate_neurons`, `show_text`, `show_search` (`UI_TOOLS`, `:496-497`). Each posts to `/api/atlas/scientist/ui`, which
emits a `ui` event to the chat's page; the reply says whether a page was open to show it (`scientist_chat.py:1347-1359`).

What the step costs underneath every live tool, from `COMPUTE.md:53-57` (home RTX 5090, 28 October 2026, layer 45,
census-shaped batches at 64 new word-pieces): one written word-piece takes 0.706 s on the `reference` unpacker, 0.275 s
on `lut` and 0.091 s on `compiled` at 1 row, and 0.743 / 0.303 / 0.126 s at 96 rows, with torch's peak above the model
at 0.3-1.0 GiB (1 row) to 7.2-7.5 GiB (96 rows). The server has taken `compiled` by default since 11 November 2026
(`COMPUTE.md:61-63`).

Reply size: an OpenAI-compatible model reads at most 16,000 characters of a tool reply (`TOOL_CUT`,
`scientist_chat.py:112, 1058`); the MCP path sends the reply uncut (`scientist_mcp.py:93`); the page shows 1,500
characters (`SHOWN_CUT`, `:113`).

---

## 2. How a driver model is connected

`AGENTS = ("claude", "codex", "openai")` (`scientist_chat.py:97`). A chat keeps the backend its first message chose.

### 2.1 Claude Code (`claude`)

- Started once per chat as `claude -p --input-format stream-json --output-format stream-json --verbose
  --include-partial-messages --system-prompt-file <built prompt> --mcp-config <json> --strict-mcp-config <tool flags>
  --setting-sources ""`, plus `--model $THEODOLITE_CLAUDE_MODEL` when set, and `--resume <chat>` or `--session-id <chat>`
  (`_claude_cmd`, `scientist_chat.py:562-581`). The process is kept open, in its own process group, with the work
  folder as its directory (`:672-685`).
- The MCP config names one server, `atlas`: `python -I backend/scientist_mcp.py` with `THEODOLITE_URL` (this server)
  and `THEODOLITE_CHAT` (the chat's id) in its environment (`:563-564`).
- The server reads the CLI's stream-json and turns it into chat events (`_pump_claude`, `:623-669`): `text_delta` →
  `text`, `thinking_delta` → `thinking`, `tool_use` → `tool`, tool results → `tool_result`, `result` → `done` with
  `total_cost_usd`.
- The conversation is Claude Code's own session file under `~/.claude/projects/*/<chat>.jsonl` (`:557-559`).

### 2.2 Codex (`codex app-server`)

- JSON-RPC over stdio: `initialize`, `thread/start` or `thread/resume`, `turn/start`, `turn/steer` (a message into a
  running turn), `turn/interrupt`, `thread/revert` (`:824-851, 1262-1284, 1309-1313, 1460-1471`). `approvalPolicy:
  "never"`; any approval Codex asks for is declined (`:782-784`); the atlas's MCP tools are pre-approved
  (`default_tools_approval_mode: "approve"`, `:841-842`).
- It gets a Codex home of its own under the scientist folder holding only the person's login (a symlink to
  `~/.codex/auth.json`), their `model` and `model_reasoning_effort`, with apps and memories off (`_codex_home`, `:720-737`).
- The system prompt goes in as `developerInstructions` (`:845`).

### 2.3 An OpenAI-compatible endpoint (`openai`)

- `POST <base>/chat/completions` with `tools` (the 35 as functions, `openai_tools()`, `scientist_tools.py:501-504`),
  `tool_choice: "auto"`, streaming unless the profile says otherwise, and the profile's own `params` merged in
  (`_openai_turn`, `scientist_chat.py:1068-1140`, body at `:1098-1100`).
- The server runs the loop: complete, run each tool call through the atlas's routes in this process, append the
  results, repeat until the model answers without a call or 40 rounds pass (`MAX_ROUNDS`, `:111, 1095-1097`).
- Profiles: `lmstudio`, `ollama`, `openrouter`, `custom` from `$THEODOLITE_OPENAI_BASE_URL`, and any in
  `<scientist folder>/endpoints.json` with `{label, base_url, key_env, model, stream, headers, params}`
  (`:286-321`). A profile names the environment variable holding its key; a file entry holding a key is refused (`:329-330`).
- Such a model has the atlas's tools and nothing else: no shell, no files (`:239-244`).

### 2.4 What a new or future model needs

1. **If it speaks Chat Completions with tool calls**: one entry in `endpoints.json`. No code changes. The `params`
   table carries any model-specific request fields.
2. **If it is an agent CLI that can take an MCP server**: the stdio server already works for any MCP client. It
   answers `initialize`, `tools/list`, `tools/call` and `ping` and nothing else (`scientist_mcp.py:84-99`):
   notifications are skipped (`:82-83`), so there are no resources, prompts, progress notices or cancellation, and calls
   run one at a time in arrival order (the loop is synchronous, `:74-92`). But to appear in the chat tab a new CLI needs
   a new adapter in `scientist_chat.py`: a process starter, a pump that turns its events into the chat's events, and
   branches in `send`, `interrupt`, `rewind`, `history`, `_summary` and `restart`, each of which is written per agent
   today. An outside agent can also simply add the stdio server to its own MCP configuration with `THEODOLITE_URL` set;
   with `THEODOLITE_CHAT` empty the screen tools reply "no such chat is open" and proposals record an empty chat
   (`scientist_chat.py:1352-1355`, `scientist_tools.py:688`).
3. **If it speaks only a native API (Anthropic Messages, OpenAI Responses)**: there is no backend for it. It would
   need a fourth agent kind with its own loop, or to be reached through an OpenAI-compatible gateway.

### 2.5 What is model-specific or CLI-specific

- The Claude Code flags, and the reach rules built from its permission syntax (`Read(//<work>/**)`), tried on Claude
  Code **2.1.312** only (`scripts/verify_scientist_cli.py` docstring; `ROADMAP.md:1502-1517` says a new version should be
  tried with the same script).
- The stream-json event shapes, the refusal handling (`stop_reason == "refusal"`, `:607-612`) and the tool-name
  prefix `mcp__atlas__` (`:109`).
- Codex's app-server method names and item types (`mcpToolCall`, `commandExecution`, `fileChange`, `webSearch`,
  `:752-766`).
- For endpoints: the reasoning fields read are `reasoning` and `reasoning_content` (`:1010, 1034`); tool-call names
  arriving whole, in pieces or repeated are all handled (`:1018-1021`); a server that refuses streaming is retried
  without it (`:986-987`); attachments go as `image_url` data URLs and PDFs are not sent (`:960-974`).
- The model *under study* is named in the prompt's first lines (60 layers, 21,504 neurons, 5,376 dimensions,
  `scientist_prompt.md:1-3`) and in `PAGE_NOTE` (`scientist_tools.py:30`); `overview` reports the shape from the server.

### 2.6 Reach: what the driver can do beyond the tools

`$THEODOLITE_SCIENTIST_REACH`, default "folder" (`scientist_chat.py:170-173`):

| | folder (default) | home |
|---|---|---|
| Claude Code | Read, Write, Edit only inside the work folder; **no shell** (`:569-572`) | Read, Glob, Grep, Write, Edit, Bash pre-approved over the home folder (`:565-567`) |
| Codex | Shell in `workspace-write` sandbox: writes only in the work folder, **no network**, so a script cannot reach the instrument's API (`:232-238, 843`) | `danger-full-access`, never asks |
| openai | Tools only | Tools only |

So **at the default reach no backend can execute code against the instrument**: Claude Code has no shell, Codex's
shell has no network, an endpoint model has neither. Code execution against the HTTP API exists only at "home", which
the documents warn about (`README.md:693-703`).

Neither CLI is given the endpoints' keys or the Hugging Face and cloud credentials in the server's environment
(`child_env`, `:176-197`). Claude Code loads no settings file, hook, rules file or CLAUDE.md (`--setting-sources ""`),
because under the earlier `project` setting a CLAUDE.md the scientist wrote in its work folder reached every later chat
as instructions (`:573-575`).

---

## 3. The loop

### 3.1 Turn-taking and locks

- **The GPU lock.** `_GPU_LOCK = asyncio.Lock()` (`server.py:99`) is shared with the atlas's routes
  (`atlas_routes.bind(get_app_state, _GPU_LOCK)`, `server.py:1660`; `on_gpu`, `atlas.py:148-151`). Every live tool
  call, from any chat, any page and any script, waits in one line. A sweep holds it for the whole sweep
  (`README.md:599-600`). The prompt tells the scientist so, and to fill one call rather than send several
  (`scientist_prompt.md:172-183`).
- **The standalone FastMCP lock.** One `asyncio.Lock` per event loop, taken around each whole tool
  (`mcp_scientist.py:142-171`); a no-op when the server bound the runtime.
- **The per-chat lock.** `c["lock"]` around `send` and `restart` (`scientist_chat.py:406, 1246, 1521`).
- **The stdio MCP server** handles one call at a time (`scientist_mcp.py:74-92`), so parallel tool calls from a CLI
  are serial before they even reach the GPU lock.
- **Field notes** have their own lock, never the GPU's (`backend/field_notes.py:54-56`).

### 3.2 Limits and timeouts

| What | Value | Where |
|---|---|---|
| Tool rounds in one turn, openai backend only | 40 | `scientist_chat.py:111` |
| Tool rounds, Claude Code and Codex | none set by Theodolite (no max-turns flag in `_claude_cmd`); the CLIs' own limits not established | `scientist_chat.py:576-581` |
| HTTP timeout of one MCP tool call | 900 s | `scientist_mcp.py:31` |
| In-process tool call (openai) | no timeout | `scientist_chat.py:878` |
| Endpoint request | connect 15 s, read 600 s, write 60 s, pool 15 s | `scientist_chat.py:1088` |
| Codex JSON-RPC call | 120 s | `scientist_chat.py:740` |
| Idle chat reaped (stdin closed) | 30 minutes without listeners or turns, checked every 60 s | `scientist_chat.py:110, 536-546` |
| Event stream keep-alive | a `ping` every 20 s | `scientist_chat.py:1336-1338` |
| Message length | 100,000 characters | `scientist_chat.py:124` |
| Upload | 25 MB, at most 16 attachments a message | `scientist_chat.py:114, 1155` |
| Tool reply read by an endpoint model | 16,000 characters | `scientist_chat.py:112` |
| Tool reply shown on the page and written to the codex / openai log | 1,500 characters | `scientist_chat.py:113, 803, 1064` |

### 3.3 Budgets

There are none: no token budget, no dollar budget, no GPU-seconds budget, no cap on calls per chat. Claude Code's
`total_cost_usd` is put on the `done` event (`scientist_chat.py:657`) but the page's scientist files never read a
`cost` field (no match for "cost" in `frontend/src/screens/atlas/Scientist.tsx` or `scientist.ts`), and it is not
stored (the Claude backend's events are not logged by Theodolite at all: `_emit` logs only codex and openai chats,
`:450-451`). Codex and endpoint turns carry no cost.

### 3.4 Cancellation

- `POST /interrupt` (`:1297-1315`): openai sets an event that is checked between stream lines and before each tool
  call, and every unrun call is answered "not run: the person stopped the reply" so the conversation stays valid
  (`:1104-1119`); Codex gets `turn/interrupt`; Claude Code gets a `control_request` interrupt.
- A tool call already running is not cancelled by any code in the scientist modules: the openai loop awaits it
  (`:1125`), and the MCP server's HTTP request runs to its 900 s timeout. Whether a dropped connection stops a GPU
  reading in the atlas's routes was not established here.
- `POST /rewind` cuts a chat back to before the last message (`:1438-1492`); `POST /restart` ends the process group
  (`:1511-1529`). Both are refused with 409 while the scientist is answering.

### 3.5 Long-running work

There is no job system. What exists:

- At the "home" reach the prompt tells the scientist to start anything longer than about a minute as a background
  command, say where its log is and end the reply; Claude Code is told it gets a turn when the command finishes, Codex
  that it must read the log in a later turn (`scientist_prompt.md:364-366`, `scientist_chat.py:224-231`).
- A message sent while the scientist works reaches it between tool calls on all three backends (`:1253-1255,
  1271-1276, 1290-1293`).
- Because a background command lives in the chat's process group, a chat whose tools have changed is not restarted
  unasked; it is marked stale and the page offers a restart (`_take_up_changes`, `:506-523`).
- The census, the atlas builds and the calibration trials are separate scripts with their own drivers, not something
  the scientist starts through a tool.

### 3.6 How results reach a person

- `GET /events?chat` streams NDJSON: `state`, `sent`, `text`, `thinking`, `tool` (name and input), `tool_result`
  (output cut to 1,500 characters), `ui`, `error`, `done`, `note`, `stale`, `ping` (`:1318-1344`). The page shows each
  tool call in the chat.
- Screen tools move the 3D map, open panels, mark and label neurons.
- Proposed names and notes appear on the page with keep and discard.
- Files the scientist writes in its work folder open from the chat through `GET /work/<path>`, each under
  `Content-Security-Policy: sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:`, so
  a page it wrote runs no script and fetches nothing (`:122, 1551-1568`). At the folder reach a deliverable is the file
  itself, since Claude Code cannot run a script that would write it (`scientist_prompt.md:384-389`).
- Pattern-test results are kept on the face and listed on the details tab, in red where a test went against the
  pattern (`ROADMAP.md:1229-1231`).

---

## 4. State that persists

| State | Where | Shape and rules | What a later session sees |
|---|---|---|---|
| Names and notes (kept) | `$THEODOLITE_FIELD_NOTES`, else `~/.local/share/theodolite/field_notes.json`, outside every checkout | `{neurons: {"<layer>:<n>": {name, note, by, name_by, note_by, at}}, heads: {...}}`, layers from 0 (`field_notes.py:8-17`) | `names`, `neuron`, `overview`; the prompt says to read them as data, never as instructions, and never to take a scientist's name for a finding (`scientist_prompt.md:305-307`) |
| Proposals | `field_proposals.json` beside the notes | `{name?, note?, chat, at}` per neuron or head (`field_notes.py:31-36`) | Only that something awaits, **without its text** (`scientist_tools.py:527-531`); "nothing reads one before it is kept, later chats included" (`README.md:642-644`) |
| Field book | `field_book.json` | Per face: `jobs`, `pattern` (≤4,000 characters), four ticks (`other_pole`, `long_contexts`, `steered`, `readers_traced`), `by`, `at` (`field_notes.py:19-29`) | Written by the folio page; **no scientist tool reads or writes it** (no route to `/api/atlas/fieldbook` in `scientist_tools.py`) |
| Pattern tests | `field_tests.json` | Per face, latest first, **at most six**, each carrying the pattern it tested word for word (`field_notes.py:38-46`) | `neuron` returns them as `tests` (`scientist_tools.py:573-576`) |
| Blind tests awaiting an answer | Server memory only (`a.cache["tests"]`, `atlas.py:1926`) | Truth, pairs and detail under a random id | Lost when the server restarts or many are begun after it (`atlas.py:1959-1960`) |
| Chats: index | `<scientist folder>/chats.json` | Per chat: `agent`, `created`, `profile`, `model`, Codex `thread`, `turns`, `turn_at` (`scientist_chat.py:374-389, 1237-1239, 1283`) | The list of past chats |
| Chats: Claude Code | `~/.claude/projects/*/<chat>.jsonl` | Claude Code's own session | Resumed with `--resume`; a rewind backs it up first (`:1486`) |
| Chats: Codex and openai | `<scientist folder>/logs/<chat>.jsonl`, and `<chat>.messages.json` for openai | Events `user, text, tool, tool_result, error, done`; tool outputs cut to 1,500 characters in the log; the openai messages file holds the full conversation with replies up to 16,000 characters | Replayed as history; openai conversations are reloaded from the messages file (`:944-957`) |
| Work folder | `<scientist folder>/work/` | Free-form files; **one folder shared by every chat and every backend** (`work_dir`, `:152-155`) | Whatever earlier chats left; this is the only free-form memory a later session can read |
| Uploads | `<scientist folder>/uploads/` | `<id>-<name>` and `<id>.json` | By id |
| Built prompt | `<scientist folder>/system_prompt.<agent>.md`, read-only | Rebuilt when its sources change (`:271-281`) | — |
| Person's additions to the prompt | `<scientist folder>/prompt.local.md` | Appended to the prompt (`:252-254`) | Every later chat |
| Reference neurons and heads | `backend/scientist_methodology.md:143-151` | "None established in this atlas yet"; the person adds them after review | Every chat, once any exist |

The scientist folder is `$THEODOLITE_SCIENTIST_DIR`, else `<atlas folder>/scientist` (`:143-149`).

There is no record of experiments as such: no table of hypotheses, predictions, calls and outcomes. A chat's log is
the nearest thing, and it truncates results. Whether vault edits survive a server restart is the vault reader's area
and was not established here.

---

## 5. Guardrails and method

### 5.1 The label gate (`backend/label_gate.py`)

- A neuron label is "held back" when its layer is below 40 counting from 0, which is page layer 41
  (`LABEL_MIN_LAYER`, `:30`), when the J++ lens and the naked eye share none of their first ten word-pieces under the
  base-fitted lens (`CUT = 0`, `:36`), or when the neuron writes nothing (`HOLD_REASONS`, `:39`).
- The study measured the gate at code layers 40-57 only (`BAND`, `:31`), on base-fitted J++ labels of random neurons,
  where the shared count as a graded score ranked sound labels with AUC 0.84; as a yes/no cut it scored AUC 0.64 on
  the same sample (7 of the 31 it held back were sound, 160 of the 209 it kept) (`:10-13, 41-63`).
- The words are still sent: held back marks what the pages leave out of their main reading. Every tool reply that
  carries labels carries the verdict (`label`: "shown" or "held back: reasons", `scientist_tools.py:91-97`).
- The gate was never measured on edited neurons: a neuron edited in the vault is flagged `edited`, `lens_validated`
  false (`label_gate.py` card caution, `:66-80`). **This matters directly for a weight-space harness: the instrument's
  own labels are unvalidated on any neuron the agent has edited.**

### 5.2 Layer numbering

The pages and the chat scientist count layers from 1; the HTTP API, the FastMCP tools, the label gate's card, the
field-notes keys and llama.cpp's control-vector tensors count from 0. `from_page` refuses a layer below 1 and shifts
every `layer` and `layers` argument, also inside lists; `to_page` shifts replies back (`scientist_tools.py:40-79`).
The prompt repeats it (`scientist_prompt.md:8-13`), and the `overview` reply hands over the gate's card with a note
that the card counts from 0 (`scientist_tools.py:556-557`). `check_scientist.py` section 1 checks the count in every
argument and reply.

### 5.3 Lens cautions the agent is given (`scientist_prompt.md:42-74`)

- The J++ lens was fitted on the base model in bf16 and is applied to the instruct model in 4 bits: good for
  comparing, poor on its own.
- Output words predicted real effects only at page layers 41-58; input-side words (`gate_trigger`) are near useless.
- Never name a neuron from a held-back label or from lens words alone.
- The steer lens is a measured effect of a push that is, at most layers, larger than anything the neuron can write
  by itself; it agrees with the census from layer 40 up and is a weak guide below (checked at 33-39, unchecked below 33).
- About 180 dimensions of the stream are read with up to 1,700 times the gain of the others: a push along a
  lens-derived direction wrecks the text, so pushes go along a neuron's own write direction.
- Multi-hop weight paths are guesses: "leads to check, never circuits" (`:144-147`). Where readers were measured
  with gates shut, the predicted sign agreed with the measured change at about 0.42 of links in use and 0.34 from rest,
  both below chance (`:98-100`).

### 5.4 What the two prompt files require of the agent

- Start with `overview`; use tools, do not guess (`scientist_prompt.md:4-6`).
- Treat a face as the unit; propose the test that would break the explanation (`:20-40`).
- Greedy is the reading; sample only to see whether a theme survives the draw, give the seeds, and never sample or add
  a penalty to make an answer read better (`scientist_methodology.md:27-30`).
- Read a pushed row against the baseline of its own call, because a batch's shape can move a row's numbers on the GPU
  (tool description, `scientist_tools.py:294-297`).
- A census rank is a lead, never a finding; read the run's card for whether its doses are calibrated at that layer
  (`scientist_prompt.md:105-137`).
- Hold a recipe out: run it on prompts it was not tuned on, and again with `spare_bos: false`, before handing it over
  (`:197-200`; `scientist_methodology.md:61-64`).
- Run `test_by_writing` and `blind_test` before proposing a name from passages, and say which test a name rests on
  with its figures (`scientist_prompt.md:244-291`).
- Names end with a confidence mark: none, `(?)`, `(??)`, `(???)` (`:308-311`). Propose with `set_name` only when the
  person asks or agrees.
- Never state a multi-hop path, a held-back label or a guessed role as fact; evidence before interpretation (`:315-318`).
- The prompt and method files are the person's to change, never the agent's (`:359-362`).
- "If a tool is missing or gets in your way, say so in one line at the end of your reply, and the person will pass it
  on" (`:399`). This is the only feedback channel from agent to instrument.

Fixed context cost: the prompt is 34,660 bytes and the methodology 12,345 bytes, about 47 KB before
`prompt.local.md`, plus 39 KB of tool schemas, so roughly 86 KB of instructions and schemas precede the first message.

### 5.5 The statistics offered

On the chat scientist's 35 tools:

- **Census**: an empirical p against random-direction controls, a layer-wide false-discovery q, and the plain
  statement that with C controls no p falls below 1 / (C + 1) and about N / (C + 1) of N neurons beat every control
  by chance, about 84 of a layer with 256 controls (`scientist_prompt.md:109-111`).
- **`search_census`**: the layer's controls are searched the same way as the neurons, giving `matched` against
  `by_chance_alone` (`scientist_tools.py:656-659`).
- **`blind_test`**: right of pairs with a one-sided binomial tail p (`pattern_tests.py:87, 170-183`), the same over
  near-miss pairs (`near_pairs`) and over pairs whose firing passage reaches half the extreme (`strong_pairs`).
- **`test_by_writing`**: counts and shares at fixed thresholds: a text fires at 0.5 of the face's extreme, a control is
  quiet under 0.25 (`pattern_tests.py:68-69`); no p.
- **`logprobs`**: KL from base to intervened. **`causal_check`**: KL, KL over the k = 0.01 floor, split-half rho,
  linearity, P@10 of the label against the hundred word-pieces raised most.

On the older surface only (`test_motif_hypothesis`, reachable at `POST /api/scientist/test_hypothesis` and by the
folio, **not by the chat scientist's tools**):

- Welch's t with a two-sided permutation p: exact over every split when there are at most 10,000, else 10,000 seeded
  draws with p = (1 + as large) / (1 + draws) (`mcp_scientist.py:422-423, 497-540`).
- No judgement below 4 sentences a side (`:419`), because at 3 against 3 the least possible p is 0.1.
- A paired sign-flip test, no judgement below 6 pairs (`:420`), because at 5 the least p is 2/32.
- The same test at each sentence's strongest word-piece, and an "either" test that takes the larger |t| of the two
  readings again under every relabelling, so the choice between readings is paid for (`:803-828`).
- Benjamini-Hochberg q across up to 512 neurons named in one call, with `needs`: how many neurons must reach the least
  possible p before any q can fall under 0.05 (`:831-846, 883-946`).
- Honest about its reach: the q's "correct for nothing else: not neurons tested in other calls, not sentences tried
  and changed" (`:897-899`).

`scripts/check_hypothesis_test.py` checks these against references written out in plain Python, including a seeded
simulation of 2,000 null neurons showing that the smaller of two readings' p's falls under 0.05 nearly twice as often
as it should while the "either" p does not.

### 5.6 Checks that exist for the harness itself

- `scripts/check_scientist.py` (CPU only, a tiny random Gemma 4 stand-in, real connections): tool schemas and the
  layer count, every tool against the routes, the MCP server, and all three backends played by fakes, including
  mid-turn messages, interrupts, rewinds, stale and restart, torn log lines, who may ask, and the stop. ROADMAP records
  240 checks (`ROADMAP.md:1336`).
- `scripts/verify_scientist_cli.py`: the real `claude` binary against a stand-in Messages API with no model, no
  network and no login; it tests what the CLI permits, "not what a model would do".

---

## 6. What is missing for weight-space programming

| Capability | Today | Easy or hard to add, and why |
|---|---|---|
| **Weight edits by the chat scientist** | None of the 35 tools edits. The vault is on the older surface (`query_larql_vault`) and at `POST /api/larql`. | **Easy to expose** as a tool: one schema in `TOOLS` and one branch in `call` that posts to `/api/larql`; all three backends get it at once. What an edit can be is fixed by the vault: one down_proj column, rank one (`mcp_scientist.py:1570-1572`). Anything richer is the vault reader's area. |
| **Raw weight reads** | No tool returns numbers from the weights. The runtime has the primitives: `slice_w_down`, `slice_w_gate`, `slice_w_up`, the norm weights and layer scalars are used inside `trace_downstream_readers` (`mcp_scientist.py:282-290, 319-324`). | **Medium.** The primitives exist, but every tool reply is JSON text, cut to 16,000 characters for an endpoint model. One 5,376-wide column as JSON is larger than that. A tensor must stay server-side and be named by a handle, with tools that return statistics, slices and products. That handle store does not exist in the server. |
| **Free code execution against the weights** | Not in the scientist. At the default reach no backend can run code that reaches the model (§2.6). At "home" a shell can call the HTTP API, not the tensors. The lab daemon (`experiments/lab/labd.py`) does execute submitted Python against a loaded runtime (`rt`, `lab`, `torch`, a persistent `STORE` dict), but it is a separate process that loads its own model, shares the server's default port 8765, and cannot run beside the server on one card. No scientist tool reaches it. | **Hard under the present shape.** The tools are HTTP proxies and the model lives in the server process; the stdio MCP server holds no model. Either the server grows an exec route (with the daemon's token and host rules, and a sandbox the daemon does not have), or the harness runs on the daemon instead of the server and loses the atlas's routes. The scientist reads model output, census answers and notes, any of which can carry instructions (`scientist_chat.py:22-23`), so exec plus that reading is the security question to settle first. |
| **Batch submission of experiments** | Batching exists inside single calls: `steer` takes 12 candidate sets or 8 prompts (about the time of one, §1.3), `compare` 300 neurons by 24 texts, `test_motif_hypothesis` 512 neurons. There is no submit-and-poll: every call is a held HTTP request (900 s on the MCP path). | **Medium.** A job table with ids and a poll tool is new code, and it must share `_GPU_LOCK` with the person's page: a long job on a single `asyncio.Lock` starves everything else unless it yields between items. |
| **An experiment ledger** | None. Chat logs hold calls with results cut to 1,500 characters; Claude Code chats are not logged by Theodolite at all. `field_tests.json` is a small ledger for one kind of test (six a face). Replies do not carry a fingerprint of the weights they were read from, except `vault_edits` in `causal_spot_check` and `vault_edits_in_force` in the control-vector export. | **Medium, and the dispatcher is the natural place.** `scientist_tools.call` is the one function every backend's tool call passes through, with the chat id in hand. A ledger written there (or in the routes) would record every call for all three backends. It would need the edit state in force added to each record. |
| **Held-out evaluators the agent cannot see** | One exists in miniature: the blind test. Half of each face's stored passages are kept back from `passages`; the pairing truth is held in server memory and released only on answer; the reader is an endpoint's model in a fresh conversation with no tools and no history (`ask_once`, `scientist_chat.py:904-937`). | **The pattern is reusable, the enforcement is not.** The hold-out is applied in the tool (`fold="fit"`, `scientist_tools.py:588`) while the route's default is `fold="all"` (`atlas.py:1614`), so a scientist with a shell, or Codex reading files outside its folder, can see the kept-back half. A real held-out evaluator needs its data and scoring server-side, behind routes the agent's reach cannot call, at the "folder" reach. Also, on this machine no endpoint is configured, so the product's blind-reader path "has run only against a stub" (`ROADMAP.md:1269-1270`). |
| **A workspace** | A files workspace: one work folder shared by all chats, served to the person with no scripts. No tensor workspace, no per-experiment or per-chat isolation, no snapshot of model state. | **Files: already there. Tensors and model state: new.** The daemon's `STORE` dict is the only precedent for objects that persist across calls. |
| **Cost accounting** | Claude Code's dollar cost per turn is emitted and dropped. `causal_spot_check` reports `timing_ms`. No other reply carries time, tokens, rows or memory. | **Easy at the dispatcher** for wall time per call; GPU seconds and peak memory need the routes to report them. |
| **Pre-registration of a hypothesis** | None in the scientist: a test takes a pattern or sentences and returns a result. The project does register designs before measuring in its census work, by hand and by commit. | New. The pattern-test record (pattern kept word for word with its result) is the nearest precedent. |
| **Determinism across calls** | Documented as absent on the GPU: on the 31B, 6 of 15 rows in a five-prompt batch and 6 of 13 in a six-candidate batch parted from what the same row wrote in a batch of three before 80 word-pieces, unpushed rows among them (`ARCHITECTURE.md:920`). | A constraint, not a gap to close: any evaluator must compare within one batch or measure the noise. |

### 6.1 Documented pain points with the current scientist

1. **One GPU, one line.** Every live call waits; the prompt spends a section teaching the agent to pack rows
   (`scientist_prompt.md:172-183`).
2. **Speed depended on the unpacker.** A scientist chat on 11 November waited 50 to 54 s for three 80-piece answers on
   `reference`; `compiled` made it 8.3 s (`COMPUTE.md:61-63`).
3. **Batch shape changes greedy answers** (above), so an answer must be quoted with the call it came from.
4. **No live model has been watched under the reach rules**: they were tried on the real CLI against a scripted
   stand-in, and "no live model has yet been watched to see whether what it reads would lead it to try what the CLI
   refuses" (`ROADMAP.md:1502-1518`).
5. **Open reach holes** in the same paragraph: Codex in its sandbox can read files outside its folder; a scientist
   with a shell over the home folder could keep its own proposal through the loopback port, which the guard cannot tell
   from the page; the rules hold only for the CLI version tried.
6. **The blind reader has no endpoint here**, so with Claude Code or Codex the scientist answers its own blind test,
   "a weaker test", and the result says so (`scientist_tools.py:799-807`).
7. **Changed tools need a restart** for a Codex chat or a Claude Code chat at "home"; only a Claude Code chat at the
   folder reach takes them up by itself (`scientist_chat.py:506-523`).
8. **The older audit's finding still stands by design**: an unbound call to the FastMCP module loads a second 31B
   model (`experiments/results/E6_tool_audit_REPORT.md:179`; `README.md:485`).
9. **No reference neurons exist** after two weeks of the scientist: "None established in this atlas yet"
   (`scientist_methodology.md:145`).
10. **Documents lag the code**: three tool counts (30, 32, and 35 in code) and a tool list in ARCHITECTURE §4.8 that
    omits the pattern tests.

---

## 7. What was not established

- Wall times for `logprobs`, `activations`, `compare`, `top_firing`, `attn_drive`, `steer_lens`, `inspect_pole`,
  `trace_downstream_readers` and one `causal_spot_check` on the 31B: no figure in the files read.
- Whether a dropped HTTP connection cancels a GPU reading already under way in the atlas's routes.
- Any turn or tool-call limits the Claude Code and Codex CLIs apply themselves, and how Claude Code truncates a long
  MCP tool reply.
- Whether vault edits persist across a server restart, and the vault's full statement grammar (the vault reader's area).
- Whether the census process and the server can hold the card at the same time (the pattern-test pilot paused the
  census to run, `ROADMAP.md:1233`).
- How any real driver model behaves in this harness: every check uses fakes or a scripted stand-in, and the documents
  record no transcript-level evaluation of a scientist session.
