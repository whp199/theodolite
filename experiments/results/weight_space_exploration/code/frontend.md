# The frontend: where a workbench would live

Read on 14 November 2026 at commit e6ac64f. Everything below was checked in the code unless it says "the document says".
Paths are relative to `(the repository root)`. Nothing was run against a server, a browser or the GPU.

## The short answer

A weight-edit surface already exists and is not where the agent is. The **vault** page (`#/vault`) has a form that
makes a rank-one `down_proj` column edit, hears the model before and after, lists the edits in force and undoes them
all. The **scientist** chat lives in the **atlas** page (`#/atlas`), and none of its 35 tools makes or undoes a vault
edit. The two pages use two different stores and two different transports. A workbench has to join those two halves or
be built beside both.

## 1. Screens, stack, transport

**Routes.** A hand-written hash router, ten pages (`frontend/src/router.ts:15-25`, `frontend/src/App.tsx:7-16`):
`#/` incipit, `#/instrument`, `#/layer/36`, `#/neuron/54/15491/pos`, `#/atlas` (also `#/atlas/54/15491`,
`#/atlas/54/h3`), `#/landmarks`, `#/nudge`, `#/vault`, `#/lexicon[/term]`, `#/lineage[/anchor]`. Each page is a lazy
chunk (`App.tsx:6`). A new page needs four edits: the `Route` union and `parseHash` (`router.ts:15-70`), `href`
(`router.ts:72-84`), `TITLES` and `screenFor` (`App.tsx:18-62`), and the `NAV` list (`components/kit.tsx:59-67`).
Layers are 1-60 on a page and in a URL, 0-59 in state and on the wire (`router.ts:9`, `screens/atlas/api.ts:3`).

**Stack.** React 19, zustand 5, raw three.js 0.173, Vite 6, TypeScript 5.7 strict with `noUnusedLocals`
(`frontend/package.json:13-32`, `frontend/tsconfig.json`). `npm run build` is `tsc --noEmit && vite build`; there is no
test runner. The server serves `frontend/dist` at `/` (`backend/server.py:1667-1669`); `npm run dev` serves on
127.0.0.1:5173 and proxies `/api` and `/ws` to 127.0.0.1:8765 (`frontend/vite.config.ts:8-20`). Checks of the interface
are Python scripts that bundle a TypeScript module with the frontend's own esbuild and run it under node
(`scripts/check_interface_defaults.py:6,95`), which is why request-building code is kept in React-free modules
(`screens/atlas/push.ts:1-6`).

**Two stores, two transports.**

| | instrument, layer, folio, landmarks, nudge, **vault** | **atlas** (map, census, push, details, passages, scientist) |
|---|---|---|
| store | `store/useTheodoliteStore.ts` (103 KB) | `screens/atlas/state.ts`, `screens/atlas/scientist.ts`, a local store in `PushTab.tsx` |
| transport | one WebSocket `/ws/theodolite`, HTTP fallback | HTTP only: JSON, binary, SSE and NDJSON streams |
| helpers | `requestJson` / `postJson` (`useTheodoliteStore.ts:801-821`) | `get` / `post` / `getBinary` / `streamLines` (`screens/atlas/api.ts:35-85`) |

- WebSocket actions sent: `sweep`, `cancel_sweep`, `larql`, `inspect_pole`, `trace_readers`, `causal_check`
  (`useTheodoliteStore.ts:29`; server side `backend/server.py:1575-1620`). Messages handled: a binary THEO frame then
  `theodolite_metadata`, `theodolite_progress`, `theodolite_written`, `theodolite_sweep_end`, `larql_result`,
  `pole_inspection`, `causal_check`, `downstream_readers`, `error` (`useTheodoliteStore.ts:1131-1270`).
- `larql_result` does not echo its statement: the page matches replies to statements by send order
  (`useTheodoliteStore.ts:1178-1179`). A sweep carries a `request_id`; a vault statement does not.
- HTTP used by that store: `/api/health`, `/api/larql`, `/api/scientist/{inspect_pole,trace_readers,causal_check,test_hypothesis}`,
  `/api/saved_prompts` (`useTheodoliteStore.ts:30-31`).
- The atlas calls `/api/atlas/*` (53 routes in `backend/atlas.py`, of which the page never calls `/steer`, `/logprobs`,
  `/compare`, `/fires`, `/attention`, `/cone`, `/head/row`, `/head/pattern`, `/neuron/written_test`,
  `/neuron/blind_test/*`: those serve the scientist's tools) and `/api/atlas/scientist/*` (13 routes).
- The socket opens on every page (`App.tsx:69-73`), reconnects after 4 s growing by 1.6 to 30 s
  (`useTheodoliteStore.ts:783,1128`), and on each open sends a silent `DESCRIBE` to resync the ledger of edits
  (`:1088-1090`) **and takes a six-piece greedy sighting by itself** unless a live one is already held (`:1095-1097`).
- With no server the pages show a recorded sighting from `fixtures/e6-demo-sighting.json` and say so (`:1274-1282`).
- Who may call: loopback only, same account on Linux (the document says, ARCHITECTURE.md:261-286; not re-checked here).

## 2. How the scientist chat shows an agent at work

Files: `screens/atlas/scientist.ts` (state), `screens/atlas/Scientist.tsx` (view), `state.ts:692-748` (screen actions).

- **One stream per chat.** `GET /api/atlas/scientist/events?chat=` is read as NDJSON in a loop that reconnects after
  1.5 s (`scientist.ts:187-203`). Event kinds: `state`, `sent`, `text`, `thinking`, `tool`, `tool_result`, `ui`,
  `error`, `done`, `note`, `stale`, `ping`, `user` (`api.ts:833-836`). On reconnect mid-turn the server replays the
  turn so far and the box is redrawn (`scientist.ts:212-221`).
- **A tool call** is one collapsed `<details>` row: a one-line label built from the tool's name and a few known input
  fields (`layer`+`n`, `command`, `file_path`, `query`, `text`, `candidates`, `neurons`, `prompts`), and inside it the
  raw input JSON and the output, pretty-printed if it parses as JSON, "running…" until the result arrives, red on error
  (`Scientist.tsx:154-176, 232-250`). `tool_result` is matched to its call by `id` (`scientist.ts:238-239`). There is
  **no per-tool rendering**: a steer result, a causal check and a file write all look the same.
- **Thinking** is a collapsed row, omitted when empty (`Scientist.tsx:224-231`).
- **Text** goes through a small hand-written Markdown (`Scientist.tsx:39-149`): `L54 N15491` and `L54 h3` become
  links that open the neuron or head; the only URLs made into links are `/api/atlas/scientist/work/…` and
  `/api/atlas/exports/…` (`:34-35`). No tables, no images inline.
- **Files.** Under the last tool call that wrote each file the page adds "open <name>" (a `Write`/`Edit` inside the
  work folder) or "download <name>.gguf" (`export_control_vector`) (`Scientist.tsx:180-218`).
- **Screen actions.** A tool posts `/ui`; the page gets a `ui` event and runs `scientistUi` (`scientist.ts:226`,
  `state.ts:701-748`): `select`, `head`, `highlight`, `annotate`, `rename`, `propose`, `camera`, `tab`, `search`,
  `text`. `tab` can only switch among `census`, `push`, `details`, `passages` and only with a neuron open
  (`state.ts:730-732`). No action navigates to another page.
- **Steering the agent.** A message can be sent while it works and "reaches it between tool calls"
  (`Scientist.tsx:633`, `scientist.ts:4`); the submit button becomes **stop** when the box is empty
  (`Scientist.tsx:612, 650-651`; `POST /interrupt`, `scientist.ts:294-307`); **edit** and **ask again** rewind the last
  turn (`POST /rewind`, `scientist.ts:392-425`); **restart** appears when the chat's instructions or tools changed after
  it started (`Scientist.tsx:570-581`); **past** lists earlier chats and reopens one from the server's history
  (`scientist.ts:344-388`); files attach by button, drop or paste (`scientist.ts:429-447`). The message carries what
  the page has open (`layer`, `n` or `head`; `scientist.ts:283`).
- **Backend picker.** Claude Code, Codex, or an OpenAI-compatible profile and model, fixed for the life of a chat
  (`Scientist.tsx:360-436`). Under the "home" reach the page asks once per browser before the first chat
  (`Scientist.tsx:497-498, 545-569`).
- **Kept in the browser.** The thread's last 300 entries, tool outputs cut to 4,000 characters, under
  `theodolite.atlas.scientist` (`scientist.ts:95-108`). The server's logs are the full record.
- **A dot** marks a reply that ended while the scientist tab was not showing (`scientist.ts:248`, `RightPanel.tsx:36`).
  The event stream is opened by the atlas page and stays open while another page is shown (`screens/Atlas.tsx:295-300`).
- **Width.** The chat sits in the atlas's right panel, `min(580px, 42vw)` wide (`index.css:1223-1226`).

## 3. What UI exists for edits, saved pushes, exports, proposals, tests

| thing | UI today | where |
|---|---|---|
| **vault edit: make** | Yes. A sentence form for UPDATE ("make neuron N of layer L say W more strongly, strength S, on its ± face") and DELETE ("silence"); a plain-words preview; the LQL it will send; INSERT only by hand in the console | `screens/vault/EditPanel.tsx:92-175, 203-405` |
| **vault edit: hear** | Yes, on by default. The model answers one question greedily before and after, up to 12 word-pieces; the page shows both answers, where they part, KL at the first piece, the edit's word's odds and rank, elapsed ms | `EditPanel.tsx:300-387, 512-542`; `useTheodoliteStore.ts:439` |
| **vault edit: ledger** | Yes. "Edits in force" with a count, each edit's description, statement, patch id, norms, and its voice; resynced by `DESCRIBE` on each connect and when a causal check reports other edits | `EditPanel.tsx:569-612`; `useTheodoliteStore.ts:1568-1623` |
| **vault edit: rollback** | Only **undo every edit** (`ROLLBACK`). No undo of one edit | `EditPanel.tsx:186-193, 613-654` |
| **vault console and history** | Yes. Free LQL with a client-side mirror of the server's parser; a log of the last 50 statements of this session; a recorded study session | `screens/vault/lql.ts:20-139`, `ConsolePanel.tsx`, `HistoryPanel.tsx`, `useTheodoliteStore.ts:1595` |
| **edits seen elsewhere** | Yes, read-only. The folio sets aside a causal test read under other edits; a sighting carries the edits it was read under and a word-piece read under other edits is not merged; the atlas marks words, links and steer lens "read before its vault edit" | `screens/NeuronFolio.tsx:2216-2222, 2393-2397`; `useTheodoliteStore.ts:1824-1838, 2131`; `screens/atlas/parts.tsx:38-41`, `DetailsTab.tsx:136, 201-202`, `LeftPanel.tsx:261` |
| **push** | A three-voice chat (up, unpushed, down), add or ×, greedy or sampled; streamed as SSE. Kept only in memory and cleared when another neuron opens. **Not a saved object** | `screens/atlas/PushTab.tsx:72-145` |
| **nudge** | A page of its own (`#/nudge`); its settings ride on a sweep. Not a saved object | `screens/Nudge.tsx` |
| **export** | One button: "save for llama.cpp" writes one neuron's add push as a control-vector GGUF and downloads it. No LoRA export, no multi-neuron recipe, no GGUF export of an edit in the UI. Pages also export their own readings as JSON | `PushTab.tsx:163-192, 463-473` |
| **proposals** | Yes, for names and notes only. "The scientist proposes the name: … keep · discard", shown beside the neuron and in a "waiting for you" list above the chat box; nothing reads a proposal until kept | `screens/atlas/parts.tsx:164-209`, `Scientist.tsx:440-475`, `state.ts:674-688` |
| **tests** | Read-only display of kept pattern tests (written, blind) per face in the details tab, red when against the pattern; the page has no button to start one. The folio has its own live causal test of a label | `DetailsTab.tsx:617-684`, `api.ts:299-394` |
| **saved things** | Questions and system prompts (server file); names, notes and a per-face field book (server files) | `components/SavedShelf.tsx`, `screens/NeuronFolio/saved.ts:3-15` |

The scientist cannot edit weights through its own tools: `backend/scientist_tools.py:192-493` defines 35 tools and none
calls the vault; the only vault traces are read-only fields (`:580, 741`). The older MCP server
`backend/mcp_scientist.py:1569` has `query_larql_vault(lql_query, voice)` (named on the page at
`screens/vault/ConsolePanel.tsx:360, 625`); how it relates to the atlas chat is another reader's area.

## 4. How long-running work is shown

- **A sweep** (seconds to a minute): `theodolite_progress` drives "writing 3 of up to 6…" then "sighting 2 of 6…" in
  the header of every page (`components/kit.tsx:127-131, 147`) and beside the answer, with **stop here**, which the
  server honours between word-pieces (`components/WordsItWrote.tsx:97-140`; `cancel_sweep`,
  `useTheodoliteStore.ts:985-988`). The page estimates time first from constants "timed on this server": 1.4 s base,
  0.75 s per piece written, 0.7 s per further position sighted (`useTheodoliteStore.ts:100-102`; no date given).
- **A vault statement**: one flag, `larqlPending`, and one line, "The vault is making the edit…"
  (`EditPanel.tsx:406-409`). No progress, no cancel. It waits behind any sweep under the one GPU lock
  (the page says, `ConsolePanel.tsx:636-641`).
- **A push**: tokens stream into three columns, repainted every 40 ms; "run" becomes "stop" (an abort)
  (`PushTab.tsx:121-138, 385, 481`).
- **A scientist turn**: "working…", each tool row "running…" until its result (`Scientist.tsx:240, 313, 324`).
- **A causal check, readers, a pole inspection**: a `pending` flag per key, failed with a reason if the socket closes
  (`useTheodoliteStore.ts:1108-1118`).
- **Jobs of minutes to hours** (a census run, an index build, a steer-lens build): **no UI at all**. They are CLI
  scripts; the atlas shows their results once written, with provenance lines (`screens/atlas/parts.tsx:132-160`). There
  is no job list, no polling, no `EventSource`, no notification outside the scientist's dot.
- The header also shows "memory N GB in use" from `/api/health` (`components/kit.tsx:163`).

## 5. The design system and what a new screen must meet

- **Tokens** (`frontend/src/index.css:1-26`): vellum `#f6f0e3`, ink `#2b2117`, muted `#6b5d4d`, hairlines, rubric red
  `#8f2b1e`, gilt `#a07a2e`, lapis `#2e4a7d` (attention; gilt is feed-forward). EB Garamond for all text with true small
  caps; IBM Plex Mono only for code and queries. "No shadows, no gradients." Light scheme only (`index.html`). Fonts are
  bundled so the pages work offline (`main.tsx:5-13`).
- **Two layouts exist.** The long illuminated page: `<Page route folio>` with header, footer and folio number, max
  1360 px, big sentence-form inputs, numbered sections (`components/kit.tsx:528-539`, `screens/Vault.tsx:68-108`). The
  atlas: a full-viewport tool with a toolbar and two floating panels over a WebGL map, small type, `atlas-*` classes
  (217 uses in `index.css`), left panel `min(390px, 32vw)`, right `min(580px, 42vw)` (`index.css:1034-1040, 1208-1226`).
- **Two reading switches on every page**, kept per browser: **rubrics** (on by default; red italic guidance for
  newcomers) and **apparatus** (off by default; formulas, raw values, sources, every expert control)
  (`frontend/src/reading.ts`, `kit.tsx:217-265`). The owner's rule is never scare a novice and never hide a feature
  from an expert (memory note of 16 October 2026).
- **Vocabulary** (`lib/words.ts:1-6`, README.md:249-266): sighting, layer 1-60, word-piece, the lens and the naked eye,
  gathering and remembering, a neuron's two faces, threads, landmarks, nudge, push, the three voices, census, steer
  lens, web, vault, edit, scientist. Words for navigation, not icons. No devotional imagery (same memory note).
- **Honesty conventions a new screen inherits.** `<Missing>` prints a bracketed placeholder rather than an invented
  number (`kit.tsx:267-270`); every lens label passes the label gate (`screens/atlas/parts.tsx:1-7`); every readout of
  words carries a caution line; the server's echo, not the form, says what was done (README.md:285-286); a refusal is
  told apart from a transport failure (`useTheodoliteStore.ts:791-799`, `EditPanel.tsx:422-432`); each new term gets a
  Lexicon entry with a stable anchor (`screens/lexicon/entries.ts:1-4`).
- **The page already says what an edit is not.** The edit form's own rubrics say it is not MEMIT, that the direction
  lies in the "quiet dimensions" from layer 36 where a small push can derail the model, and that the study never put
  UPDATE or INSERT to a causal test (`EditPanel.tsx:433-472, 550-556`).

## 6. Where a workbench would most naturally sit: considerations

**No UI at first (agent and CLI only).**
- The driver is an LLM; it needs tools and files, not forms. The chat already shows any tool call generically, so new
  weight-space tools would appear in the thread with no frontend work.
- The cost is that a person following along sees collapsed JSON rows only, in a 580 px panel, with no ledger of edits
  in force on the page the agent lives on.
- The existing `ui` channel cannot show an edit: it has no action for it and cannot leave the atlas.

**A tab of the atlas.**
- The agent, the map, the census, the push and the open neuron are already there; `open_tab`, `highlight` and
  `annotate` give the agent a hand on the screen; the proposal pattern (keep · discard) is there to copy.
- Atlas tabs are per open neuron (`RightPanel.tsx:24`), and a multi-edit hypothesis is not one neuron. The panel is
  narrow. The atlas store never opens the socket's vault state: it would read edits over HTTP `POST /api/larql`, or the
  two stores would be bridged.

**A new screen.**
- Room for what a run of hypotheses needs: a ledger of edits with per-edit undo, a queue of trials with status, held-out
  results, a diff against the untouched model, the agent's thread beside them.
- It costs a route, a nav word (the eighth), a Lexicon entry, and a choice of layout: illuminated page or tool. The
  scientist chat component reads atlas state (`useAtlas` focus, `goToNeuron`, `scientistUi`), so mounting it elsewhere
  means loosening that coupling.

**Extending the vault page.**
- It already has the edit form, the voice, the ledger, the console and the name. It lacks the agent, per-edit undo,
  any batch or saved edit set, and any test beyond one question's before and after.

**Facts that bear on any of them.**
- The page learns of an edit made elsewhere only on reconnect or when a causal check reports different edits; a second
  writer (an agent) needs a push of ledger changes or a poll.
- Replies to vault statements over the socket are matched by order, not by id.
- Nothing in the frontend follows a job longer than a sweep.
- The edit grammar the page mirrors is fixed regular expressions (`lql.ts:20-30`); a wider edit vocabulary means a new
  wire shape, not more LQL forms.
- Opening any page takes a sighting on the GPU.
- A stand-in model serves the whole interface without the weights (`scripts/atlas_stand_in.py --serve`,
  `screens/Atlas.tsx:333-335`), so a new surface can be developed off the card.

## Documents that disagree with the code

- The scientist's tool count: 35 in `backend/scientist_tools.py:192-493`; "30" in README.md:239,
  `screens/Lexicon.tsx:1411` and ARCHITECTURE.md:43; "32" in ARCHITECTURE.md:1132.
- `ui` actions: ARCHITECTURE.md:1124 lists nine; the code also accepts and handles `propose`
  (`backend/scientist_chat.py:1356`, `state.ts:721-723`).
- ARCHITECTURE.md:1136 lists the live tools without `test_by_writing`, `blind_test`, `blind_test_answer`, which exist.
- `package.json` lists `@react-three/fiber`, `@react-three/drei`, `three-stdlib` and `lucide-react`; nothing in `src`
  imports them (the map uses raw three.js, `screens/atlas/map.ts:11-15`).

## Not established

- Whether the atlas chat's Claude Code or Codex backend can reach `query_larql_vault` or any other edit path.
- The built bundle's size and whether `dist` (built 13 November 23:19) matches the source at e6ac64f.
- How the pages behave on a narrow screen beyond the breakpoints in `index.css`; nothing was rendered.
- When the pace constants (1.4 s, 0.75 s, 0.7 s) were measured.
