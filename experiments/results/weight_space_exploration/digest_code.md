# Code-map digest (nine readers of the Theodolite repository at (the repository root), read-only, 14 November 2026)

Each reader cites path:line in its full report: experiments/results/weight_space_exploration/code/<key>.md . Facts here were read from code and documents, not run.


## frontend — The frontend: where a workbench would live
Full report: experiments/results/weight_space_exploration/code/frontend.md

### Summary
The frontend is a React 19 / zustand / Vite single-page app of ten hash-routed pages, served as static files by the FastAPI server. A weight-edit surface already exists but is split from the agent: the vault page (#/vault) has a form that makes a rank-one down_proj column edit, hears the model before and after on one question, lists the edits in force and undoes them all, over the WebSocket /ws/theodolite; the scientist chat lives in the atlas page (#/atlas), talks HTTP only, and none of its 35 tools makes or undoes a vault edit. The chat renders every tool call as one generic collapsed row of raw JSON, lets a person send messages mid-turn, stop, rewind and restart, and lets the agent move the atlas screen through ten fixed actions. Pushes are not saved objects, the only export button writes one neuron's control vector, proposals exist for names and notes only, pattern tests are displayed but cannot be started from the page, and nothing in the frontend follows a job longer than a sweep. A workbench therefore has to join the vault's edit ledger with the atlas's agent, or stand beside both; the report gives the considerations for no UI, an atlas tab, a new screen, or an extended vault page.

### Key facts
- The app has ten pages behind a hand-written hash router: incipit, instrument, layer, neuron, atlas, landmarks, nudge, vault, lexicon and lineage. [frontend/src/router.ts:15]
- Adding a page needs edits in four places: the Route union and parseHash, the href table, TITLES and screenFor in App.tsx, and the NAV list in the kit. [frontend/src/components/kit.tsx:59]
- Layers are counted 1-60 on pages and in URLs and 0-59 in state and on the wire, and every conversion is done at the boundary. [frontend/src/screens/atlas/api.ts:3]
- The server mounts frontend/dist at / as static files, and the dev server on 127.0.0.1:5173 proxies /api and /ws to 127.0.0.1:8765. [backend/server.py:1669]
- The instrument, folio, nudge and vault pages share one store that talks over one WebSocket with the actions sweep, cancel_sweep, larql, inspect_pole, trace_readers and causal_check, falling back to HTTP when the socket is closed. [frontend/src/store/useTheodoliteStore.ts:29]
- The atlas page (map, census, push, details, passages, scientist) uses separate stores and HTTP only: JSON, binary, SSE for the push and NDJSON for the scientist's events. [frontend/src/screens/atlas/api.ts:35]
- A larql_result message does not echo its statement, so the page matches vault replies to statements by the order they were sent on the socket. [frontend/src/store/useTheodoliteStore.ts:1178]
- On every socket open the page sends a silent DESCRIBE to resync the ledger of edits and also takes a six-piece greedy sighting by itself unless it already holds a live one. [frontend/src/store/useTheodoliteStore.ts:1088]
- The vault page already has an edit tab whose form builds an UPDATE or DELETE statement for one neuron, shows a plain-words preview and the LQL, and sends it with executeLarql. [frontend/src/screens/vault/EditPanel.tsx:170]
- An edit is heard by default: the model answers one question greedily before and after, up to 12 word-pieces, and the page shows both answers, where they part, the first piece's KL, the edit word's odds and rank and the elapsed milliseconds. [frontend/src/screens/vault/EditPanel.tsx:512]
- The only rollback control in the interface is 'undo every edit', which sends ROLLBACK; there is no control that undoes one edit. [frontend/src/screens/vault/EditPanel.tsx:187]
- The ledger of edits in force is rebuilt from DESCRIBE's active_patches, from each 'applied' reply and from 'rolled_back', and lists only the latest edit of each neuron. [frontend/src/store/useTheodoliteStore.ts:1568]
- The page learns of an edit made elsewhere (for example over MCP) only when the socket reconnects or when a causal check's reply names different vault_edits, at which point it asks DESCRIBE again. [frontend/src/store/useTheodoliteStore.ts:1609]
- The neuron folio sets aside a causal test that was read under other vault edits than those now in force. [frontend/src/screens/NeuronFolio.tsx:2220]
- A word-piece sighted under other vault edits than the rest of its answer is not merged into that answer set, and the page says so. [frontend/src/store/useTheodoliteStore.ts:1834]
- The atlas marks a neuron's words, links and steer lens as read before its vault edit, from fields the server sends (stale, steer_stale, edited). [frontend/src/screens/atlas/parts.tsx:38]
- The edit grammar the page mirrors is a fixed set of regular expressions copied from the server's execute_lql: DESCRIBE, ROLLBACK, SELECT NEAREST, DESCRIBE POLE, DELETE, UPDATE and INSERT. [frontend/src/screens/vault/lql.ts:20]
- The scientist chat reads one NDJSON stream per chat with the event kinds state, sent, text, thinking, tool, tool_result, ui, error, done, note, stale, ping and user, and reconnects after 1.5 seconds. [frontend/src/screens/atlas/scientist.ts:187]
- Every tool call is drawn the same way: a collapsed row with a one-line label, the raw input JSON and the output pretty-printed, 'running…' until the result arrives; there is no per-tool rendering. [frontend/src/screens/atlas/Scientist.tsx:232]
- The agent's screen actions are handled by scientistUi and are limited to select, head, highlight, annotate, rename, propose, camera, tab, search and text, all inside the atlas page. [frontend/src/screens/atlas/state.ts:701]
- A person can send a message while the agent works (it arrives between tool calls), stop the reply, rewind the last turn to edit or ask again, restart a stale chat, reopen past chats and attach files. [frontend/src/screens/atlas/Scientist.tsx:607]
- The scientist has 35 tools and none of them makes or rolls back a vault edit; its only vault traces are read-only fields such as edited_in_vault. [backend/scientist_tools.py:191]
- The proposal pattern exists for names and notes only: the scientist proposes, the page shows 'keep · discard', and nothing reads a proposal until the person keeps it. [frontend/src/screens/atlas/parts.tsx:164]
- A push in the atlas is a three-voice chat kept only in component memory and cleared when another neuron opens; it is not a saved object. [frontend/src/screens/atlas/PushTab.tsx:73]
- The only export control in the interface is 'save for llama.cpp', which writes one neuron's add push as a control-vector GGUF; there is no LoRA or edit export in the UI. [frontend/src/screens/atlas/PushTab.tsx:163]
- Kept pattern tests are shown read-only per face in the atlas details tab, and the page has no control that starts a written or blind test. [frontend/src/screens/atlas/DetailsTab.tsx:617]
- A sweep is the only long task with real progress and a cancel: 'writing 3 of up to 6…' then 'sighting 2 of 6…' in every page's header, with 'stop here' honoured between word-pieces. [frontend/src/components/WordsItWrote.tsx:97]
- A vault statement in flight is shown by one boolean and one line of text, with no progress and no cancel. [frontend/src/screens/vault/EditPanel.tsx:406]
- The design tokens are vellum, ink, hairlines, rubric red, gilt and lapis, with EB Garamond for all text and IBM Plex Mono only for code, no shadows, no gradients, light scheme only. [frontend/src/index.css:1]
- Every page carries two reader switches kept per browser: rubrics (on by default, red guidance for newcomers) and apparatus (off by default, formulas, raw values and expert controls). [frontend/src/reading.ts:1]

### Numbers
- Pages in the app: 10 [frontend/src/App.tsx:7-16]
- Atlas HTTP routes on the server: 53, of which about 10 are never called by the page [backend/atlas.py (counted by grep on 14 Nov 2026)]
- Scientist HTTP routes on the server: 13 [backend/scientist_chat.py (counted by grep on 14 Nov 2026)]
- Scientist tools defined: 35 [backend/scientist_tools.py:192-493]
- Word-pieces written to hear an edit, each side: 12 [frontend/src/store/useTheodoliteStore.ts:439]
- Default edit strength when left out: 2.5 [frontend/src/screens/vault/lql.ts:200]
- Example edit: layer 60, neuron 7830, word 'six', strength 3.0, negative face; heard 29 October 2026 [frontend/src/screens/vault/lql.ts:166-177]
- Vault statements kept in the page's log: 50 [frontend/src/store/useTheodoliteStore.ts:1595]
- Sweep pace the page estimates with: 1.4 s base, 0.75 s per word-piece written, 0.7 s per further position sighted; 'timed on this server', no date [frontend/src/store/useTheodoliteStore.ts:100-102]
- Socket reconnect delay: 4 s, growing by 1.6 to at most 30 s [frontend/src/store/useTheodoliteStore.ts:783,1128]
- Scientist events reconnect delay: 1.5 s [frontend/src/screens/atlas/scientist.ts:201]
- Scientist thread kept in the browser: last 300 entries, tool outputs cut to 4,000 characters [frontend/src/screens/atlas/scientist.ts:95-102]
- Atlas panel widths: left min(390px, 32vw), right min(580px, 42vw) [frontend/src/index.css:1219-1226]
- Long-page maximum width: 1360 px [frontend/src/index.css:84]
- Push tab defaults: add dose 0.8, × factor 2, greedy [frontend/src/screens/atlas/PushTab.tsx:57]
- Push stream repaint interval: 40 ms [frontend/src/screens/atlas/PushTab.tsx:133]
- Study vault session shown in history: 102 seconds, 16 October 2026 [frontend/src/screens/vault/HistoryPanel.tsx:37-39]
- Dev and preview ports: 5173 dev, 4173 preview, server 8765 [frontend/package.json:8-10; frontend/vite.config.ts:12-18]
- OpenAI-backend tool loop bounds (document and code): 40 rounds a turn, each tool reply cut to 16,000 characters [backend/scientist_chat.py:111-112]

### Reusable for a harness
- The vault edit form, its plain-words preview and its reply handling can be reused or copied for a human-facing edit control (frontend/src/screens/vault/EditPanel.tsx, EditPanel component).
- The store's vault plumbing already sends a statement with an optional voice over the socket or HTTP and keeps a ledger of edits in force (executeLarql, recordLarql, editIdsOf, resyncEditsIfBehind in frontend/src/store/useTheodoliteStore.ts).
- The before-and-after 'voice' types and their display give a ready shape for showing one edit's effect on one prompt (EditVoice and VoiceAnswer in frontend/src/store/sighting.ts; VoiceLine in frontend/src/screens/vault/EditPanel.tsx).
- The scientist chat already streams and renders any tool call, thinking, errors, notes and file links, so new weight-space tools would appear in the thread with no frontend change (onEvent in frontend/src/screens/atlas/scientist.ts; PartView and toolLabel in frontend/src/screens/atlas/Scientist.tsx).
- The steering controls for a person following an agent exist: mid-turn messages, stop, edit, ask again, restart, past chats and attachments (agentTurn, stop, rewindLast, restartChat, openPast, attachFiles in frontend/src/screens/atlas/scientist.ts).
- The ui event channel lets a tool act on the person's screen and can take new actions by adding a case (scientistUi in frontend/src/screens/atlas/state.ts; the allowed set in backend/scientist_chat.py:1356).
- The propose, keep or discard pattern is a working model for agent-proposed edits that wait for a person (ProposalLine in frontend/src/screens/atlas/parts.tsx; Waiting in frontend/src/screens/atlas/Scientist.tsx; decideProposal and loadProposals in frontend/src/screens/atlas/state.ts).
- The streaming helper reads SSE or NDJSON from a fetch response and would serve a job or trial stream (streamLines in frontend/src/screens/atlas/api.ts).
- The three-voice push thread is a ready layout for showing an answer under a change beside the unchanged answer (Thread and runPush in frontend/src/screens/atlas/PushTab.tsx).
- The sweep progress and cancel pattern is the one existing model for a tracked long task (InFlight in frontend/src/components/WordsItWrote.tsx; inFlightWords and ServerStatus in frontend/src/components/kit.tsx).
- The edits-changed guards show how a page refuses to mix readings taken under different edits (staleness check in frontend/src/screens/NeuronFolio.tsx:2216-2222; mergeInto guard in frontend/src/store/useTheodoliteStore.ts:1824-1838).
- The kit supplies the page frame and the honesty components a new screen must use (Page, Rubric, Apparatus, IfApparatus, Missing, NotYet in frontend/src/components/kit.tsx).
- The fixed link whitelist already lets the agent hand a person a file from its work folder or the exports folder (fileLink and toolFile in frontend/src/screens/atlas/Scientist.tsx).
- A stand-in model serves the whole interface without the 31B weights, so a new surface can be built and checked off the GPU (scripts/atlas_stand_in.py --serve, named at frontend/src/screens/Atlas.tsx:333).
- The interface checks bundle React-free TypeScript modules with the frontend's esbuild and run them under node, a pattern a new request-building module can join (scripts/check_interface_defaults.py; frontend/src/screens/atlas/push.ts as the example).

### Gaps
- No scientist tool and no ui action makes, lists or undoes a weight edit, so the agent in the atlas chat has no edit path of its own in the interface.
- There is no undo of a single edit in the interface, only ROLLBACK of all of them.
- There is no saved or named edit set, no batch of edits, and no way to reapply a set after a rollback or a server restart.
- There is no view of an edit's effect beyond one question's before and after: no held-out set, no diff against the untouched model, no table of trials.
- The ledger of edits in force is shown only on the vault page; the atlas page, where the agent works, shows no count or list of edits in force.
- There is no push of ledger changes to open pages: a page hears of another writer's edit only on reconnect or through a causal check's reply.
- There is no job surface for anything longer than a sweep: no queue, no status list, no polling, no notification beyond the scientist's unseen-reply dot.
- There is no per-tool rendering in the chat, so a structured result (a trial table, an edit diff) would show as raw JSON in a 580-pixel panel unless a renderer is written.
- There is no proposal flow for edits; the existing one covers names and notes only.
- There is no export control for an edit or a recipe of several neurons, and no LoRA export control; the page offers one neuron's control vector only.
- There is no button that starts a pattern test from the page; tests are started by the scientist's tools.
- The scientist chat component is tied to atlas state (focus, goToNeuron, scientistUi) and cannot be mounted on another page without loosening that.
- The frontend has no test runner; only tsc and Python checks that bundle selected modules.

### Constraints
- A new screen must use the existing tokens and type: vellum, ink, rubric red, gilt, lapis, EB Garamond with small caps, IBM Plex Mono only for code, hairlines, no shadows, no gradients, light scheme only.
- A new screen must honour the two reader switches: guidance in Rubric components that can be turned off, and formulas, raw values and expert controls inside Apparatus blocks.
- Wording must stay in the instrument's plain vocabulary (sighting, word-piece, faces, push, nudge, vault, edit, scientist), with words rather than icons for navigation and no devotional imagery; each new term needs a Lexicon entry with a stable anchor.
- Layers must be shown 1-60 and sent 0-59, converted only at the boundary.
- A number that was not measured must be shown as a bracketed placeholder, never invented, and what was done must be reported from the server's echo rather than from the form.
- Every lens label must pass the label gate, and readouts of words carry their caution line.
- A refusal by the server must be told apart from a transport failure, as the vault page does.
- The server answers only loopback pages of the same account, so a workbench page must be served by the instrument itself or through the dev proxy.
- There is one GPU lock: an edit waits behind any sweep, and a sweep can hold the lock for a minute or more, so the interface cannot promise an immediate edit.
- Vault statements over the socket are answered in order with no id, so two writers on one socket cannot be told apart by reply.
- TypeScript is strict with noUnusedLocals and noUnusedParameters, and the build fails on a type error.
- Atlas right-panel tabs exist only while a neuron is open and the panel is at most 580 px wide, which limits what a tab there can show.
- Browser storage is used only for per-viewer conveniences (thread copy, last focus, reading switches); shared state lives in server files.

### Surprises
- An edit UI with before-and-after hearing, a ledger and rollback already exists on the vault page, but it is on a different page, store and transport from the agent chat.
- The scientist's 35 tools include pushes, causal checks, pattern tests and a control-vector export but no weight edit; edits by an agent go through a separate, older MCP server's query_larql_vault.
- Documents disagree with the code on the scientist's tool count: the code defines 35, README.md:239, Lexicon.tsx:1411 and ARCHITECTURE.md:43 say 30, and ARCHITECTURE.md:1132 says 32.
- ARCHITECTURE.md:1124 lists nine ui actions, but the server and page also accept and handle 'propose'.
- Opening any page, the atlas included, makes the socket take a six-piece sighting on the GPU by itself.
- package.json lists @react-three/fiber, @react-three/drei, three-stdlib and lucide-react, but nothing in src imports them; the map is raw three.js.
- The page keeps a hand-ported mirror of the server's LQL parser, down to Python's Unicode digit and whitespace rules, only to explain a statement before sending it.
- The edit form's own text warns that the edit is not MEMIT, that from layer 36 its direction lies in dimensions where a small push can derail the model, and that UPDATE and INSERT were never put to a causal test.
- The default example edit is at layer 60, neuron 7830, chosen because edits tried at layers 41, 46, 51, 56 and 59 were not heard on its question at strengths 3 or 30.

### Not established
- Whether the atlas chat's Claude Code or Codex backend can reach query_larql_vault or any other weight-edit path was not established; that belongs to the scientist and MCP readers.
- Whether frontend/dist (built 13 November 2026 at 23:19) matches the source at commit e6ac64f was not established.
- The built bundle's size was not measured.
- How the pages render on a narrow screen was not established; nothing was rendered in a browser.
- When the sweep pace constants (1.4 s, 0.75 s, 0.7 s) were measured is not stated in the code.
- The local-only and same-account guard was read from ARCHITECTURE.md section 4 only, not checked in backend/local_only.py.


## docs_direction — What the documents say: direction, caveats, house style, licence (README.md, ROADMAP.md, COMPUTE.md, REFERENCES.md, VOLUNTEERS.md, NOTICE, LICENSE, docs/)
Full report: experiments/results/weight_space_exploration/code/docs_direction.md

### Summary
The documents describe Theodolite as an instrument plus a study, and the owner's recorded direction is a general-purpose interpretability tool and learning aid with the atlas and census as its main selling point. Nothing in README, ROADMAP, COMPUTE, REFERENCES, VOLUNTEERS or NOTICE plans a workbench, harness or weight-space environment, the weight-space-programming paper is not cited anywhere in the repository, and ROADMAP.md never mentions the exports at all (the LoRA export is documented only in ARCHITECTURE §6). What exists next to the feature is: a vault whose edit is one neuron's down-projection column (rank one, explicitly "not MEMIT"), heard before and after on one question, undone by an all-at-once ROLLBACK, and tested by cause and effect only for DELETE; a tool-only scientist with no shell by default and no weight-edit or vault tool among its 35 tools; and a separate lab daemon that executes arbitrary Python with the model held, through which Claude Code agents ran the study's 314 GPU jobs. Any new feature inherits a dense set of stated caveats: the J++ lens was fitted on the base model (cos 0.15-0.33 with the model's own gradients, trust banded by depth), neuron labels are held back below page 41, a census push is about 2 write ceilings and so is not the neuron firing, doses are counted calibrated at only 21 of 60 layers under three designs, "passed" means three readings clearing bars of 0.60/0.58/0.57/0.56 fixed and hashed beforehand, and batch shape alone can change an answer after 22 word-pieces. The house style is plain words from surveying and printing, roman-numeral sections, every figure dated and marked measured or inferred, the owner's decisions quoted, and pre-registration before any measurement. The hardware fact that bounds the design is that every forward unpacks all 29.3 billion 4-bit weights afresh (0.09-0.13 s a decode step compiled) and dense weights need about 60 GB beside them, so they do not fit the 32 GB card. The code is Apache-2.0, the repository redistributes no weights, the documents call Gemma 4 Apache 2.0, and no document says anything about distributing edited weights, adapters or silenced copies.

### Key facts
- Layers are counted 1-60 on the pages and in the README, but 0-59 in ROADMAP section vi, COMPUTE.md, VOLUNTEERS.md and every script (page = layer + 1), so every layer number must be read with its count. [ROADMAP.md:334]
- The roadmap has no entry for a workbench, harness, weight-space work or an agent that edits weights, and it contains no line with the word 'export' or 'control vector'; the feature would be a new section, not the completion of a planned one. [ROADMAP.md:1]
- The vault's edit writes one neuron's down-projection column (rank one, no covariance solve; DELETE sets it to zero); the code names it '.patch MEMIT' but the pages call it an edit and say it is not MEMIT. [README.md:460]
- Each vault edit can be heard before and after on one question of the reader's choosing, up to 12 word-pieces by default (1-64), under the one GPU lock; the voice shows what the edit did to that answer only, and the study tested only DELETE by cause and effect. [README.md:463]
- Rollback is documented as one command that undoes every edit ('ROLLBACK clears every overlay'); README.md and ROADMAP.md do not mention rollback, and no document I read describes undoing a single edit. [frontend/src/screens/Lexicon.tsx:1580]
- The vault's example edit (neuron 7830 of page 60) turns 'eight legs' into 'six legs', and its note says such an edit is rare: the most active neurons at pages 41, 46, 51, 56 and 59 were not heard on that question at strengths 3 or 30, and the example shows an edit heard, not where a fact is kept. [ROADMAP.md:174]
- The vault reads through the base-fitted J++ lens, as do the words, the gathered/recalled split, the landmark bearings, the neuron labels and the nudge, so a word-targeted UPDATE or INSERT is a lens-derived edit. [README.md:383]
- The J++ lens was fitted on the base model google/gemma-4-31B in bf16 and is applied to the 4-bit instruct model, where it agrees with the model's own average gradients at only cos 0.15-0.33 for the pronoun direction at pages 31-59; no instruct refit passed R2, R3 or R4, and none is served. [README.md:383]
- Lens trust is banded by depth: pages 1-20 least trusted, 21-30 orderings only, 31-50 to be confirmed with an instruct-fitted reading, 51-59 counting only where an instruct reading agrees, and page 60 is the identity. [ROADMAP.md:148]
- The base-lens caveat is shown once, on the opening page, by the owner's choice of 28 October 2026; the per-layer red line and the label gate stay, and tool replies still carry the full lens object. [ROADMAP.md:29]
- Results reached without a lens (what the model says, activation geometry, weights, patching, steering measured on the output) are untouched by the lens caution. [README.md:305]
- The label gate holds a neuron's words back below page 41 and wherever the lens and the naked eye share none of their first ten word-pieces; the overlap scored AUC 0.84 as a graded score and 0.64 as the yes/no cut, within pages 41-58 for base-fitted J++ labels only. [README.md:399]
- A neuron silenced in the vault writes nothing, has no count and is held back with reason writes_nothing, and every label's label_check carries an 'edited' field, so the documents already expect labels to change under a weight edit. [README.md:406]
- About 180 residual dimensions are read through RMSNorm with up to 1700 times the gain of the others, which is why lens-derived steering vectors break the model; a masked nudge read fluently at 0.005-0.01 of the stream's strength and broke at 0.02. [README.md:334]
- A census push is larger than a neuron can write: 0.8 of the stream is 2.1 times the open neuron's ceiling, and 1 of the stream is 1.76-2.76 write ceilings at layers 39-51 from 0, rising to 5-19 ceilings at layers 2-25, so a push shows what a push that size does, not what the neuron does when it fires. [ROADMAP.md:740]
- Doses are counted calibrated, design by design and only on a passed trial plus the owner's word: break_rel at layers 39-51 from 0, soft_ceil at 38 and 52, match_ceil at 32-37; the other thirty-nine pages (1-32 and 54-60) have no design that passed. [ROADMAP.md:982]
- A trial passes when three readings (the whole reading, the lower dose's answers alone, and the neurons whose text survival the controls cover) all clear bars of 0.60, 0.58, 0.57 and 0.57, 0.56 at 48 word-pieces, fixed and hashed before any answer; a figure of 0.599 against 0.60 is counted a fail. [ROADMAP.md:997]
- A high census rank says only that a neuron's push differs from a random direction's, not that its words share a theme or name what the neuron does when it fires; a pass licenses a whole layer and does not say the doses are the best for that layer. [ROADMAP.md:574]
- Layer 36 from 0, read whole on 14 November 2026, passes its whole reading and fails one gate on the lower dose alone (0.564 against 0.57); whether it stays counted calibrated is the owner's undecided choice and nothing was changed. [ROADMAP.md:1125]
- The steer lens is sound from layer 39 from 0 upward and a weak guide below: it agrees with the census for 85-99.5% of standout neurons at layers 40-50 but 7-33% at 32-38, and it is two readings of the same push, never a test of what a neuron does unpushed. [ROADMAP.md:382]
- Batch shape alone changes answers: on the 5090 an unpushed answer written in a batch of 64 first parted from itself written alone after 22 word-pieces, and in a batch of 32 after 54, so a before-and-after comparison must be written in one batch shape. [ROADMAP.md:1430]
- Weights alone predict a link's size but not its sign at an ordinary word-piece: the sign from rest agreed with the measured change at 0.336 of links, below chance, because about nine readers in ten sat with the gate shut, while linked readers moved 5.5 times the null's. [README.md:419]
- Every forward unpacks all 29.3 billion 4-bit weights afresh; a decode step takes 0.71 s on the reference unpacker and 0.091 s at one row to 0.126 s at 96 rows with the compiled unpacker, which gives the same bits. [COMPUTE.md:52]
- Keeping the weights unpacked (resident) needs about 60 GB beside the 4-bit weights, so dense weights do not fit the 32 GB card; the cheapest rented card that holds them is an RTX PRO 6000 96 GB at $2.49 an hour as read on 28 October 2026. [README.md:656]
- Only one GPU process may run at a time (the lab daemon, the server and a standalone mcp_scientist each load the 31B), and the server holds one GPU lock for a whole sweep or heard edit. [README.md:485]
- The scientist has no shell by default (Claude Code: files only in its work folder; Codex: a sandboxed shell with no network), a home-folder shell is an opt-in the README warns about, and its names and notes are proposals until the person keeps them. [README.md:693]
- The lab daemon holds the model and lenses, executes arbitrary submitted Python with pre-bound rt, lab and STORE, loads in about 15 s, and is what Claude Code agents used to design, run and write the study's 314 GPU jobs. [README.md:762]
- A reading on other arithmetic or other weights is kept under its own model id, every index records the weights it was built from and none built from other weights is read, and a census run is not continued, merged or joined across weights. [README.md:546]
- The code is Apache-2.0, the repository redistributes no model weights, lens tensors, binaries or corpora, NOTICE lists Gemma 4 as Apache License 2.0 with Gemma a trademark of Google, and the scientist's prompt and tool files are adapted from neuron-explorer under MIT with its notice reproduced. [NOTICE:7]
- The budget rule requires the owner to approve GPU-hours on a named card and provider with a dollar ceiling before a job runs, a pilot whose projection times 1.25 fits the approval, every pod logged, hard time limits, and any change recorded as a deviation; none of the pod scripts has yet run on a rented card. [COMPUTE.md:222]

### Numbers
- Model shape: 60 layers, 21,504 neurons a layer (1,290,240 in all), 5,376 dimensions, 262,144 word-pieces [README.md:11]
- Card and peak memory: RTX 5090 with 32,607 MiB; the instrument peaks near 23 GB; the 4-bit model rests at about 18 GiB and a census batch takes up to about 4.2 GiB more [README.md:477 and VOLUNTEERS.md:64]
- Weights unpacked per forward: 29.3 billion 4-bit weights in 410 linears, unpacked afresh at every forward [COMPUTE.md:28 and ROADMAP.md:444]
- Decode step by unpacker (5090, 28 October 2026): reference 0.706-0.743 s; lut 0.275-0.303 s; compiled 0.091 s at 1 row, 0.103 s at 32, 0.109 s at 64, 0.126 s at 96 [COMPUTE.md:52]
- Memory above the model by batch: 0.3-1.0 GiB at 1 row, 2.4-3.1 at 32, 4.8-5.3 at 64, 7.2-7.5 at 96 rows [COMPUTE.md:54]
- Three pushed answers of 80 word-pieces (11 November 2026): 8.3 s with compiled, 50-54 s with reference [COMPUTE.md:61]
- One heard vault edit: 12.6 s on the GPU for two answers of 12 word-pieces (date and unpacker not stated) [ARCHITECTURE.md:606]
- Load times: lab daemon about 15 s; compiled graphs about 4 s more; a GGUF packed about 21 s into 19-21 GiB [README.md:735 and README.md:541]
- Resident (unpacked) weights: about 60 GB beside the 4-bit weights on a card; about 62 GB of RAM on the CPU; packed on the CPU about 20 GB [README.md:656 and README.md:531]
- Census trial: 1,022 neurons against 1,024 controls, 49,110 answers, 68-86 minutes on the home card [ROADMAP.md:997]
- Whole census layer: 565,254 answers, 13.2 hours at 11.88 answers a second (29-30 October 2026), 2.7 GB kept [ROADMAP.md:581 and ROADMAP.md:711]
- Trial gates: 0.60 lead, 0.58 beyond text survival, 0.57 and 0.57 across question halves, 0.56 on new questions, at 48 word-pieces [ROADMAP.md:999]
- Calibrated layers (from 0): break_rel 39-51; soft_ceil 38 and 52; match_ceil 32-37; none at the other 39 layers [ROADMAP.md:983]
- Lens against the model's own gradients: cos 0.15-0.33 for the pronoun direction at pages 31-59; top word matches the next word on prose at most about 5% through page 50 [README.md:383 and README.md:304]
- Label gate strength: AUC 0.84 as a graded score, 0.64 as the yes/no cut; 7 of 31 sound when sharing none of ten, 160 of 209 otherwise [README.md:402]
- Push against the write ceiling: 1 of the stream is 1.76-2.76 ceilings at layers 39-51 from 0, 3.14 at 38, 1.21 at 57, 5-19 at 2-25 [ROADMAP.md:740]
- Quiet carry channel: about 180 dimensions read with up to 1700 times the gain; a masked nudge fluent at 0.005-0.01 of the stream's strength, broken at 0.02 [README.md:334 and README.md:396]
- Writer's floor (28 October 2026): an unpushed answer parts from itself after 22 word-pieces in a batch of 64 and after 54 in a batch of 32; alone it holds all 64 [ROADMAP.md:1430]
- Steer lens build (13 November 2026): 10.5 hours for 60 layers (37,743 s) after a 38-minute dose pilot; agreement with the census 85-99.5% at layers 40-50, 7-33% at 32-38 [COMPUTE.md:169 and ROADMAP.md:385]
- Wiring and corpus passes: wiring about 4 minutes; its measured check 59 layers in about 31 minutes; passages over 2,000,054 word-pieces in 20 minutes [COMPUTE.md:23]
- CPU speeds (2 November 2026): prefill of the packed checkpoint 31-37 s; through int4mm 0.45 s a word-piece and a 59-piece prefill in 3.5 s, not the reference's bits [README.md:531]
- Rented prices (read 28 October 2026): RTX 5090 $1.19/h; H200 141 GB $5.29/h; RTX PRO 6000 96 GB $2.49/h [COMPUTE.md:119]
- Budget precedent: R3 used 5.54 of the 8 GPU-hours the owner approved; pilots project at 1.25 times against an approval [ROADMAP.md:71 and COMPUTE.md:227]
- Scientist tool count: README says 30, ARCHITECTURE says 32, the code holds 35 [README.md:239, ARCHITECTURE.md:1132, backend/scientist_tools.py:191]
- Endpoint tool loop: at most 40 rounds a turn, each tool result cut to 16,000 characters [ARCHITECTURE.md:1082]
- Study's one causal edit test: deleting the top ten 'she' neurons at layer 51 moved the pull toward 'she' by 1.80 log-odds, against at most 0.04 for ten random neurons [frontend/src/screens/Lexicon.tsx:1584]
- Export fidelity across files: a direction from another GGUF has cosine 0.997 (QAT Q4_0) to 0.977 at worst (Q4_K files) with the checkpoint's; lmstudio's QAT Q4_0 shares 68-71% of codes and none of its 410 linears is the checkpoint's [VOLUNTEERS.md:330 and ROADMAP.md:1590]
- Disk: about 22 GB for the model, 6.4 GB for the two lenses, up to about 27 GB more for cached activations [README.md:479]

### Reusable for a harness
- The vault (backend/larql_vault.py, documented in README.md:460-469 and ARCHITECTURE.md §4.6) already gives a rank-one column edit with a patch ledger (edit_kind, replaces_patch, edits_in_force), an all-at-once ROLLBACK, and a before-and-after 'voice' that returns the first word-piece's KL and where the two answers part.
- The lab daemon (experiments/lab/labd.py with labclient.py and labkit.py, README.md:735-741 and 762) is an existing code-execution environment with the model held resident, a token on /exec, and labkit's interventions and exact same-position VJPs through the 4-bit model, already used by Claude Code agents for 314 jobs.
- The scientist (backend/scientist_chat.py, scientist_tools.py, scientist_mcp.py; ARCHITECTURE.md §4.8) already plugs in three driver kinds, Claude Code, Codex and any OpenAI-compatible endpoint, with $THEODOLITE_CLAUDE_MODEL and endpoint profiles choosing the model, which is the documented seam for a future driver model.
- The older FastMCP server (backend/mcp_scientist.py, README.md:600-604) already serves query_larql_vault with a voice, inspect_pole, trace_downstream_readers, test_motif_hypothesis and causal_spot_check, each taking turns under one lock.
- The proposals mechanism (field_proposals.json, /api/atlas/proposals, ROADMAP.md:278-281) is a ready pattern for agent-made changes that nothing reads until the person keeps them.
- The census's null machinery (backend/census.py and scripts/census_run.py: 2,048 isotropic controls, --control-kind columns for mixes of the layer's own write directions, p and false-discovery q, split-half replication, text-survival covariates, the collapse rule) is a ready-made way to set an edit against matched random edits.
- The trial-and-registration practice (experiments/results/census_*_trials/prereg/ with SHA256SUMS, 'said beforehand' predictions, deviations filed) is a ready-made protocol for stopping an agent fooling itself, with a trial costing about 70-86 minutes.
- The causal spot-check (backend/causal_check.py, N1A's protocol at k = 10, README.md:601-602) and test_motif_hypothesis's statistics (Welch's t, permutation p, Benjamini-Hochberg across neurons in one call, no judgement below four sentences a side) are existing hypothesis tests with their limits stated.
- The pattern tests (backend/pattern_tests.py; the scientist's test_by_writing and blind_test, ROADMAP.md:1212-1231) test a written description without a push, with half of each face's passages kept back from the scientist.
- The CPU stand-in (scripts/atlas_stand_in.py, a tiny random Gemma 4) and the check scripts (check_*.py) let every piece be built and checked without the GPU or the weights, which is how every atlas feature was first built.
- The faster unpackers (backend/w4a16_fast.py and backend/gguf_fast.py) show how to install new weight machinery beside the pinned runtime, tested bit for bit at load and refused in plain words on any difference.
- The exporters (scripts/export_control_vector.py, backend/control_vector.py, scripts/export_lora.py; ARCHITECTURE.md §6) already carry a push, a neuron's mul, a head's scale, a vault edit and a silenced block out to llama.cpp, each file recording its provenance under theodolite.*.
- The runtime factory and model-id scheme (backend/runtime_factory.py; README.md:546-550) already separates readings by weights and arithmetic (for example the +int4mm suffix and a GGUF's fingerprint), which is the documented way to keep an edited model's readings apart.
- The writer's floor (scripts/verify_atlas_writer.py) and the canary batch (census_run.py --canary-only) measure how far the same answer differs by batch shape or card, which a before-and-after comparison of an edit needs as its noise floor.
- The steer tool's batching (several prompts or candidate sets in one call costing about the time of one, README.md:685-686) and the measured 8.3 s for three pushed answers of 80 word-pieces give the existing fast loop for testing one hypothesis.
- The wiring index and its measured check (atlas_build.py wiring and wiring_check, 4 minutes and 31 minutes) and the drive check (scripts/verify_atlas_drive.py) are existing weight-reading tools with measured agreement against real forwards.
- The budget tooling (scripts/gpu_log.py approve, start, stop and sum; scripts/pod/pilot.sh) already records approvals and spend for rented cards, including the RTX PRO 6000 96 GB that could hold resident weights.

### Gaps
- No document plans or names a weight-space harness, workbench or research environment, so its purpose, name, Lexicon entries, Lineage note and roadmap section all have to be written.
- The atlas scientist has no tool that edits a weight or queries the vault (I checked the 35 names in backend/scientist_tools.py), so a weight-editing tool surface for the driver model would have to be built.
- No guarded code-execution path exists for the scientist: the default reach has no shell, and the lab daemon runs arbitrary Python but is not joined to the scientist or to the proposals mechanism.
- No edit richer than one down-projection column is documented: there is no multi-neuron, low-rank, covariance-solved, attention-weight, norm or embedding edit in the vault, and UPDATE and INSERT have been tested by cause and effect only as far as one answer.
- No rollback of a single edit, no named checkpoints of an edit set and no branching of edit histories are described in any document I read.
- No test of an edit beyond one heard question exists: there is no held-out battery, no matched random-edit control, no measure of side effects on unrelated text, and nothing that corrects for hypotheses tried in separate calls.
- No model id or fingerprint for an edited model is documented, so the rule that readings on other weights are kept apart has no stated form for vault-edited weights.
- Dense or resident weights on the card are not available on the 32 GB 5090, and no rented card has ever been used, so any method needing unpacked weights or full gradients on the GPU needs either a new mechanism or a first rented-card run.
- The base model against the instruct model, the roadmap's one weight-space research item, is unbuilt and needs the base model and the GPU.
- Thirty-nine of sixty layers have no calibrated dose and no trusted push-based or lens-based reading, so a harness working below page 33 has no validated reading of what a neuron writes.
- No endpoint judge is configured on the home machine, and the endpoint path for blind tests has run only against a stub.
- A person cannot run the pattern tests from the page, and nothing tests every label (no resumable driver for a label survey).
- The LoRA export of a vault edit on the real runtime, a silenced copy of the 31B and the effect of any mul, head or vault export on text are documented as not measured.
- REFERENCES.md has no entry for the weight-space-programming paper, for LoRA or for llama.cpp, so the prior-work record for this feature has to be started.
- No policy exists on publishing edited weights, adapters or silenced model copies.

### Constraints
- Only one GPU process may hold the model at a time, and the card is often busy for 13 to 15 hours with a census layer or needed for a volunteer intake's spot-check, so the harness must share one lock and one card.
- The harness must not touch the R4-pinned backend/engine_runtime.py or the pinned backend/theodolite_math.py, and R3's jobs refuse a backend that differs from d5405ee; new weight machinery is installed beside them as w4a16_fast.py was.
- Nothing may change what writes census answers or the loader's bits while kit 4 is live, since any such change is a new kit and two volunteer layers are running.
- A reading of record must be on the reference arithmetic or something tested bit-equal to it; anything else, and any edited model, must be marked as another engine or other weights under its own model id and never mixed with the unedited model's indexes or census runs.
- Every lens-derived word, label or edit target must carry the base-lens fault and the depth bands, and neuron labels must stay under the label gate (held back below page 41 and where the two readings share none of ten).
- A push or dose may be called calibrated only where a design passed a trial and the owner counted it in; elsewhere the page must say the doses are not calibrated at that layer.
- A claim of 'passed' needs gates fixed and hashed before any measurement, what was said beforehand recorded as met or missed, and later changes filed as deviations; a result is a lead until tested against a matched control.
- Before-and-after comparisons must be written in the same batch shape and compared against the writer's floor, because batch size alone moves a row's numbers.
- Anything an agent writes that later readers or chats would take as fact must be a proposal until the person keeps it, and the agent must not be able to keep its own proposal.
- The scientist's default reach (no shell, files only in its work folder, no settings or CLAUDE.md loaded, no cloud or Hugging Face credentials) must hold, and the server answers only this machine's pages and scripts.
- Rented compute needs the owner's approval ahead in GPU-hours and dollars on a named card, a pilot whose projection times 1.25 fits it, and a log of every pod.
- The repository redistributes no model weights, lens tensors or third-party corpora, so edited weights, adapters, silenced copies and stored passages stay in git-ignored folders unless the owner decides otherwise.
- Files adapted from neuron-explorer (the scientist's prompt, methodology, chat, tools and MCP server) are under MIT and must keep its notice when extended.
- House style binds the documentation: plain words from surveying and printing with the literature's term beside them, lower-case roman-numeral sections, British spelling, every figure dated and marked measured or inferred, rubrics and apparatus on every page, no religious imagery, and no method name the thing is not.
- Side sessions commit on their own branches and do not push; merges and pushes go through review in the coordinating chat on the owner's word, and every multi-agent review includes a reader who questions the validity of the core measurement.
- The base-lens caveat is not to be re-added on every page, but new validity problems must still be flagged in default-visible text.

### Surprises
- ROADMAP.md does not mention the control-vector or LoRA exports at all, and the LoRA adapter and silenced-copy export is documented only in ARCHITECTURE.md section 6, not in the README.
- README.md and ROADMAP.md never use the word rollback; it appears only in REFERENCES.md entry 10, ARCHITECTURE.md section 4.6 and the Lexicon page.
- The documents disagree with the code on the scientist's tools: README.md:239 says 30, ARCHITECTURE.md:1132 says 32, and backend/scientist_tools.py holds 35, none of which edits a weight or queries the vault.
- The vault's MCP tool lives in a different, older server (backend/mcp_scientist.py) from the atlas scientist's (backend/scientist_mcp.py), so the agent beside the atlas cannot make the edits the vault page can.
- The repository already has an arbitrary-code environment with the model held (the lab daemon), which Claude Code agents used for 314 jobs, but the documents treat it as the study's and warn never to expose it.
- The documents' own measurement shows that weights alone give a link's sign wrongly more often than not at an ordinary word-piece (0.336 agreement) while giving its size well, which bears directly on reading weights to predict behaviour.
- The vault's own note says an edit that is heard is rare: the most active neurons at five late pages were not heard on the example question at strengths 3 or 30.
- ROADMAP.md opens by saying nothing in it has a date or an owner, yet almost every entry is dated and quotes the owner's word.
- COMPUTE.md lists Gemma 4's licence and gating as unsettled because huggingface.co could not be reached, while NOTICE and REFERENCES.md state Apache 2.0 flatly.
- The README still says eleven census layers and a 32 GB card, while layer 36 has been taken whole (and fails one gate by its trial's standard) and kit 4 lets a headless 24 GB card take a layer.
- A silenced-block export is a whole copy of the model's GGUF, and a LoRA adapter's B holds Gemma's own weight columns, so the existing exports already produce derivatives of the weights that no licence note covers.
- The census step costs nearly the same from 1 to 64 rows on the reference unpacker because the time is the unpacking, not the products, so batching candidates is almost free and the scientist's prompt is told to fill a call.

### Not established
- Whether a single vault edit can be undone alone, since the documents describe only a ROLLBACK of all edits.
- How a vault edit is laid on the packed 4-bit weights at a forward, beyond the phrases 'patch overlays' and 'as its forward lays it on'.
- The date and unpacker of the 12.6 s heard edit.
- Whether Google's licence page still reads as NOTICE says and whether the checkpoint is gated; I did not open any outside page.
- Any policy of the owner's on publishing edited weights, adapters or silenced model copies.
- Whether a gradient- or lens-derived weight edit is exposed to the quiet carry channel as steering vectors are; the documents test it only for activation pushes.
- Whether README.md's count of 30 tools or ARCHITECTURE.md's 32 was right when written.
- What the weight-space-programming paper itself says, since it is not in the repository and I did not fetch it.


## edit_paths — Every existing way to intervene on or edit the model
Full report: experiments/results/weight_space_exploration/code/edit_paths.md

### Summary
Theodolite has two families of change that never share a data structure. Activation interventions (push add / mul / act, a silenced block, head scales, the causal check's two writes, the sweep's nudge, the lab kit's hooks) are torch hooks installed for one `with` block, carry one value per batch row so a pushed and an unpushed row share a forward, and vanish when the block ends. Weight edits are of exactly one kind: the vault replaces one column of one layer's down_proj with a dense bfloat16 vector held in a dict on the module, which every linear's forward lays onto the freshly unpacked 4-bit weight; it lasts until an all-or-nothing ROLLBACK or the end of the process, is recorded only in memory, and has no file format, no versioning, no partial undo and no composition (a later edit of a neuron replaces the earlier and always starts from the shipped column). Exports are well built and well checked: llama.cpp control vectors carry add pushes, LoRA adapter GGUFs carry mul, a vault column and one head's scale, and whole-model GGUF copies carry only a silenced block; each is verified by arithmetic on a stand-in, then through llama.cpp itself, then on the 31B against a bar fixed beforehand. The gap to a general weight program is the representation (no low-rank or sparse delta, one tensor, per-neuron records and staleness rules), not the hook point: the overlay dicts and the overlay-applying forward are already installed on all 410 quantised linears, and the exporter's LoRA writer already takes any tensor name and any rank.

### Key facts
- A push has three modes: add puts sign x dose x p50_l x unit(w_post (.) W_down[:, n]) on the stream leaving the layer, mul multiplies the neuron's activation at down_proj's input, and act adds sign x dose x act_scale to that activation. [backend/atlas_ops.py:285-315]
- Pushes and head scales are installed as torch hooks inside a context manager (a forward hook on the decoder layer for add, a pre-hook on mlp.down_proj for mul and act, a pre-hook on self_attn.o_proj for heads) and every hook is removed in a finally clause, so they change activations only and never persist. [backend/atlas_ops.py:344-412]
- Interventions hold one value per batch row, so an unpushed row and pushed rows run in the same forward; the readers rely on this because rows of one forward are bit-identical while different batch shapes are not. [backend/atlas_ops.py:267-282, 666-690]
- Within one layer the order is fixed: head scales, then every mul, then every act (muls always run before acts whatever order they were given in), then the add at the layer's output; adds at one layer sum and several muls of one neuron multiply. [backend/atlas_ops.py:366-408]
- A whole feed-forward block and every head of a layer can only be silenced (factor 0), because the norm after the block undoes any other uniform scale; at most 32 pushes and 64 head scales go into one reading. [backend/atlas_ops.py:50-51, 244-264]
- Pushes spare the start mark by default through keep_mask, and the mask is cleared for decode steps; a vault edit, being a weight, acts at every position including the start mark. [backend/atlas_ops.py:415-422, 611]
- The vault's only weight edit is a replacement of one down_proj column, stored as a dense vector in the dict _memit_col_overlays on the module; the value is the new column, not a delta. [backend/larql_vault.py:482-524]
- Every quantised linear's forward unpacks the whole 4-bit weight to bfloat16 and, when the module has overlays, clones it and writes the column overlays and then the row overlays before F.linear. [backend/engine_runtime.py:352-377]
- The overlay dicts and the overlay-applying forward are installed on every module that has weight_packed and weight_scale, which is all 410 quantised linears including attention's q, k, v and o projections, not only the MLP's. [backend/engine_runtime.py:429-436]
- UPDATE and INSERT set the column to orig + polarity x strength x max(|orig|, 0.25) x unit((W_U[token] x final_norm_weight) @ J_l), where J_l is the J++ lens of the layer and the default strength is 2.5; DELETE sets it to zero. [backend/larql_vault.py:507-524]
- INSERT differs from UPDATE only in choosing the neuron by the nearest gate_proj row to the subject's pulled-back direction; it edits no gate weight. [backend/larql_vault.py:487-498]
- Every edit reads the original column from the layer's own weight without overlays, so a later edit of a neuron replaces the earlier one and never builds on it. [backend/larql_vault.py:504, 233-239]
- ROLLBACK is all or nothing: it clears the column and row overlays on down_proj, gate_proj and up_proj of layers 0 to 59 and empties the record list, and there is no statement that undoes one edit. [backend/larql_vault.py:556-567]
- PatchOperationRecord keeps the patch id, operation, layer, neuron, polarity, target word, strength, statement text, timestamp, the column's length before and after, the voice, edit_kind (always one_neuron_column) and replaces_patch; it does not keep the vector, the token id, the lens's id or the weights' identity. [backend/larql_vault.py:183-202]
- Nothing of the vault is persisted: the records are a Python list, the overlays are module attributes, the server builds a fresh vault at start, and patch ids start again with the server. [backend/server.py:136; ARCHITECTURE.md:639]
- The voice is a before-and-after check of one question: the model writes a greedy answer of up to 12 word-pieces (limit 64) just before the edit and again just after, and the record keeps both answers, the first step's KL in nats, the target word's probability and rank, and the index where the answers part. [backend/larql_vault.py:569-637]
- If the answer before the edit cannot be written nothing is edited, and if the answer after cannot be written the edit stays in force with after null and an error set. [backend/larql_vault.py:600-613]
- The vault has no lock of its own: the server holds its one _GPU_LOCK around each statement and runs it in a worker thread, and the MCP server run alone takes turns under a lock of its own. [backend/server.py:1444-1466; ARCHITECTURE.md:611-616]
- Other tools learn of edits only through edits_in_force(): the atlas keys its scale cache by the tuple of patch ids and marks index readings stale per (layer, neuron) of down_proj. [backend/atlas.py:215-219, 268-293]
- The int4mm CPU kernel is the one path that applies an edit as a difference term, x[..., i] x (v - W[:, i]) summed in float32 and rounded once, leaving the kernel's base product untouched. [backend/int4_cpu.py:138-164]
- A control vector carries only add pushes of single neurons, as F32 direction.<l> tensors summed per layer, and cannot carry layer 0 or spare the start mark. [scripts/export_control_vector.py:131-146, 335-398]
- A LoRA adapter carries a neuron's mul as A = e_n^T, B = (f - 1) W_down[:, n], a vault column as B = f v - W_down[:, n], and one head's scale as a rank-hd pair on attn_output; act is refused because llama.cpp has no hook for it. [scripts/export_lora.py:225-297, 442-479]
- A whole-model GGUF copy carries only a silenced block or a layer's silenced heads, by an added one-number .scale tensor set to 0 or by a zeroed post-norm; there is no writer that bakes an edit into the 31B's quantised weights. [scripts/export_lora.py:607-654; backend/gguf_model.py:974-991]
- export_lora.py --vault does not export a running server's edits: it loads the whole model on the CPU, runs the statement there, reads the column laid on, and restores that vault. [scripts/export_lora.py:419-439, 801-827]
- The server's control-vector route reads the model it holds, so the vault's edits in force are in the direction and named in the file, while the script reads the checkpoint's files and carries no edits. [backend/control_vector.py:69-88, 138-167]
- Exports are verified at three levels: arithmetic on a packed stand-in (W + B A against the instrument's change), log-probabilities read through a real llama-server on the stand-in, and a run on the 31B against a bar fixed beforehand. [scripts/check_lora_export.py:152-222, 444-580; ARCHITECTURE.md:1680-1747]
- Adapters served together add their changes, so two on one neuron are not the instrument's two pushes: mul f1 and mul f2 give f1 + f2 - 1, not f1 x f2. [scripts/export_lora.py:50-55]
- The documented weaknesses of the vault are that it is not MEMIT, that its direction goes through a J++ lens fitted on the base model, that only DELETE was tested causally by the study, and that a heard edit is rare. [README.md:383, 460-468; ROADMAP.md:172-185; experiments/results/E6_tool_audit_REPORT.md:122-135, 188]
- engine_runtime.py, which holds the forward that applies overlays, is pinned by sha256 for R4, so faster unpackers and the int4 kernel are bound from outside the file. [experiments/scripts/R4_common.py:101; backend/w4a16_fast.py:1-3]
- The causal spot-check is a separate intervention with two writes at one position: an activation write through a pre-hook on down_proj, and a frozen-norm write that adds delta x (w_post (.) W_down[:, i]) / rms(y_p) to the post-feed-forward norm's output. [backend/causal_check.py:229-338]

### Numbers
- Quantised linears unpacked at every forward: 410 [ARCHITECTURE.md:131; backend/w4a16_fast.py:5]
- Whole model unpacked to bfloat16: 62 GB [backend/engine_runtime.py:411]
- One decode step at 1 row, reference / lut / compiled (28 October 2026, RTX 5090): 0.706 s / 0.275 s / 0.091 s [COMPUTE.md:52-57]
- One decode step at 64 rows, reference / lut / compiled (28 October 2026): 0.721 s / 0.287 s / 0.109 s [COMPUTE.md:52-57]
- Three pushed answers of 80 word-pieces, compiled against reference (11 November 2026): 8.3 s against 50-54 s [COMPUTE.md:61-64]
- One default heard vault edit on the GPU (date and unpacker not stated): 12.6 s [ARCHITECTURE.md:605-606; backend/server.py:1452-1457]
- Voice length, default and limit: 12 and 64 word-pieces [backend/larql_vault.py:81-82]
- UPDATE default strength and floor on the column's length: 2.5 and 0.25 [backend/larql_vault.py:471, 521]
- Most pushes and head scales in one reading: 32 and 64 [backend/atlas_ops.py:50-51]
- Default add dose as a share of the stream's median length: 0.8 [backend/atlas_ops.py:220, 1925]
- A dose of 1 against a single neuron's write ceiling at layers 40-50: 1.8 to 2.5 times [ARCHITECTURE.md:1671-1672]
- LoRA rank for a neuron and for a head (sliding layers, full layers): 1; 256 and 512 [ARCHITECTURE.md:1783-1785]
- check_control_vector.py, CPU only: 48 checks in about 15 s [ARCHITECTURE.md:1646]
- check_lora_export.py, CPU only: 77 checks in about 50 s [ARCHITECTURE.md:1815]
- W + B A against the instrument's change, largest gap as a share of the largest element: 5.0e-7 (the check's bar is 1e-5) [ARCHITECTURE.md:1818; scripts/check_lora_export.py:210]
- llama.cpp against the instrument on the stand-in, largest log-probability difference unpushed: 4.4e-4 [ARCHITECTURE.md:1657]
- Control vector on the 31B (30 October 2026): correlation and median ratio over 2,304 word-pieces: r 0.986, ratio 1.11; 81 s instrument side, 279 s llama.cpp side [ARCHITECTURE.md:1708-1715]
- LoRA on the 31B, both engines on the CPU (2 November 2026), pushes moving the first piece by 2 or more: r 0.88-0.98, ratio 0.91-1.15 [ARCHITECTURE.md:1842-1855]
- CPU runtime that export_lora.py --vault loads on the 31B: minutes and some 20 GB [scripts/export_lora.py:86-88]
- QAT Q4_0 GGUF write direction against the checkpoint's, cosine over 192 neurons (30 October 2026): least 0.9965, median 0.9969 [ARCHITECTURE.md:1606-1613]
- Neurons from NEAREST 'she' and 'he' that were selective at BH q < 0.05: 35 of 102 [experiments/results/E6_tool_audit_REPORT.md:124-126]
- DELETE of the ten layer-50 'she' neurons, change in female contexts: -1.80 log-odds [experiments/results/E6_tool_audit_REPORT.md:128-133]
- J++ lens against the instruct model's own average gradients, pronoun direction, layers 31-59: cos 0.15-0.33 [README.md:383]
- Greedy answers changed on the stand-in by an edit setting a column to its own value, before the difference-term fix: 13 of 30 [ARCHITECTURE.md:1394-1395]
- Writer's noise floor run on the 31B (28 October 2026): 2 minutes with compiled; about 18 minutes by arithmetic on the reference [COMPUTE.md:25]
- One down_proj in bfloat16 (my arithmetic, 5376 x 21504 x 2 bytes): 231 MB [derived from ARCHITECTURE.md:120-122]

### Reusable for a harness
- The per-module overlay hook point can serve as the apply kernel as it is: every one of the 410 quantised linears already consults _memit_col_overlays and _memit_row_overlays at each forward (backend/engine_runtime.py:352-377, 429-436), with the same rule on the resident, GGUF, plain-linear and int4mm paths.
- int4_cpu.with_overlay_terms (backend/int4_cpu.py:138-164) already applies an edit as a delta term added to the base product, which is the form a low-rank or sparse delta needs.
- export_lora.write_lora and read_lora (scripts/export_lora.py:501-552) are a general low-rank container: they take any model tensor name with any rank as (A [r, n_in], B [n_out, r]), write provenance under theodolite.*, and llama.cpp loads the file.
- export_lora.Plan and plan_of (scripts/export_lora.py:169-297) are one checked structure for neuron factors, vault columns, head factors and silenced blocks by layer, the nearest existing thing to a weight-program record.
- export_lora.vault_kept (scripts/export_lora.py:383-406) is a snapshot-and-restore context manager for the overlays, the records and the counter, the pattern for trying an edit and putting everything back.
- atlas_ops.Interventions, resolve, merge, installed and keep_mask (backend/atlas_ops.py:267-422) are a per-row intervention kernel that keeps an unpushed row in the same forward.
- atlas_ops.write_rows and next_logprobs (backend/atlas_ops.py:502-617, 666-690) are the fast readers of an intervention's effect on text and on the next piece's odds.
- atlas_ops.forward_from_layer (backend/atlas_ops.py:1017-1043) runs the model from a given layer on from a stored stream, so an edit at layer l needs only layers l onward re-run.
- LarqlVaultEngine._heard (backend/larql_vault.py:569-637) gives the shape of a before / edit / after transaction with a refusal rule on each side and a record of what was heard.
- atlas_ops.Weights (backend/atlas_ops.py:128-209) reads down columns, gate and up rows, write directions, write ceilings, norms and attention projections on any weight format the runtime loads.
- check_control_vector.LlamaServer and packed_stand_in (scripts/check_control_vector.py:97-134, 319-382) serve a GGUF through llama.cpp on the CPU, read first-piece log-probabilities, and build a stand-in packed as the real checkpoint is.
- scripts/verify_atlas_writer.py measures the noise floor of an unpushed answer across batch shapes, the line below which a changed answer is not by itself evidence.
- Atlas.edit_key() (backend/atlas.py:215-216) keys caches by the set of edits in force, the pattern for invalidating readings when the weights change.
- export_control_vector.add_record and code_commit (scripts/export_control_vector.py:352-361, 401-420) write a provenance record with the code's commit into any GGUF.
- The lab daemon (experiments/lab/labd.py:1-34, read only as far as its header) keeps the model loaded in one GPU process and executes submitted Python against the runtime with a persistent store.

### Gaps
- There is no delta representation: an edit is a dense replacement vector per column index, with no low-rank B A, no sparse element list, no additive delta and no scale.
- No code writes an edit to any tensor but down_proj: attention projections, gate_proj and up_proj have the overlay dicts but no writer, no record and no rollback for attention.
- Embeddings, the tied lm_head, the norms and the layer scalars have no overlay mechanism, no record and no rollback at all.
- There is no patch file format and no persistence: edits and their records are lost when the process ends.
- There is no versioning, naming, parent link, state hash or diff between two edited states; the only identity is a per-process counter.
- There is no partial undo: ROLLBACK clears every edit at once.
- Edits cannot be composed: two edits of one neuron cannot be stacked or scaled, and each starts from the shipped column.
- A weight edit cannot be compared with the unedited model inside one forward, because an overlay applies to the whole batch while pushes are per row.
- No route turns the live vault's state into a LoRA adapter; the exporter re-runs statements on a separately loaded CPU runtime.
- No writer bakes an edit into the 31B's W4A16 checkpoint or into a quantised GGUF; the edited column is a bfloat16 vector off the 4-bit grid.
- The only way to choose an edit's direction is one token's unembedding pulled back through the base-fitted J++ lens.
- The voice tests one question with one greedy answer and no control; there is no held-out set or matched random-edit control for a weight edit.
- The weights card that every reading records names the checkpoint's files and not the overlay state.
- The accessors do not all show what the forward uses: slice_w_up, get_full_w_up, Weights.attn_weight and gguf_weights.dense* return no overlays.

### Constraints
- backend/engine_runtime.py is pinned by sha256 for R4, so the forward that applies overlays cannot be edited in place; a new delta mechanism must be bound from outside, as w4a16_fast and int4_cpu bind theirs.
- The weights are never held dense: the whole model unpacked is 62 GB in bfloat16 against a 32 GB card, so a general delta must be sparse or low-rank.
- With an overlay in force the W4A16 forward clones the whole unpacked weight of that module at every forward, about 231 MB for one down_proj by arithmetic.
- The resident unpacker's forward checks the kept weight's version counter and raises if it was written in place, so in-place weight mutation is refused there.
- A new apply path must not change the numeric path of the base product: when the int4mm layer fell back to the reference forward for an edit, setting a column to its own value changed 13 of 30 greedy answers on the stand-in.
- Every statement, sweep and atlas route runs under one GPU lock, and the lab daemon and the server share port 8765, so only one GPU process and one job run at a time.
- The vault hard-codes 60 layers and 21,504 neurons from theodolite_math; checks on the six-layer stand-in overwrite larql_vault.N_LAYERS before ROLLBACK.
- Every index (wiring, steer lens, passages, stats) and the census were built on the shipped weights; the only guard after an edit is a per-neuron stale flag that knows nothing of attention or norm edits.
- llama.cpp acts on the start mark and adds adapters rather than multiplying them, so an exported program matches the instrument only where one neuron or head is changed once per file and the start-mark difference is accepted.
- llama.cpp refuses direction.0, has no hook for an activation offset, and a LoRA must be low-rank, so act pushes, layer-0 adds and full-rank changes cannot be exported.
- A plain nn.Linear hears an overlay only after gguf_weights.install_overlay_forward is called on it, which scripts/atlas_stand_in.py does not do.
- Rows of one forward are bit-identical while different batch shapes are not, so a before-and-after comparison must hold the batch shape fixed or keep a base row in the same forward.

### Surprises
- The overlay dicts and the overlay-applying forward are already on all 410 quantised linears, attention included (backend/engine_runtime.py:429-436), but ROLLBACK clears only the three MLP projections (backend/larql_vault.py:559-565), so an overlay written on an attention projection would be applied and never rolled back.
- A row-overlay mechanism exists end to end (dict, forward, accessors for gate_proj, rollback, checks) and nothing writes it: 'none is written today' (backend/engine_runtime.py:365-366).
- The forward would apply a row overlay on up_proj, but slice_w_up and get_full_w_up return the weight without overlays (backend/engine_runtime.py:1069-1077, 1111-1119), so readers and the forward would disagree.
- UPDATE adds a stream-space direction to the raw down_proj column without dividing by the post-feed-forward norm's weight, while NEAREST multiplies the column by that weight before comparing (backend/larql_vault.py:311-316 against 516-523); no document I read mentions it.
- The edited column is a bfloat16 vector laid on after unpacking, so an edit lives off the 4-bit grid of the QAT checkpoint (backend/larql_vault.py:520-522).
- The vault clamps an out-of-range layer or neuron instead of refusing it, so a statement naming layer 99 edits layer 59 (backend/larql_vault.py:481, 500).
- active_patches_count counts every record including those replaced, not the edits in force (backend/larql_vault.py:264, 553).
- There is no patch file anywhere: '.patch' names an in-memory overlay and its record, and I found no code that reads or writes one on disk.
- On a plain nn.Linear the vault records an edit that the weight readers show and the forward ignores, unless install_overlay_forward was called; scripts/atlas_stand_in.py does not call it while scripts/check_lora_export.py:595-596 does.
- The lab kit's gradient pass on the W4A16 checkpoint ignores vault overlays while its GGUF twin lays them on (experiments/lab/labkit.py:417-453).
- The atlas scientist's tool list (backend/scientist_tools.py:192-493) has pushes, a causal check and a control-vector export but no weight-edit tool; only the older MCP scientist has query_larql_vault (backend/mcp_scientist.py:1567-1595).
- A repository lab daemon already executes submitted Python against the loaded model with a persistent store (experiments/lab/labd.py:1-34), which is the code-execution half of a weight-space environment.
- The vault's module docstring lists EDGES and POLES as two searches, but the parser discards the keyword and runs the same query (backend/larql_vault.py:701).
- E6 item 12 reported that INSERT's docstring claimed a gate edit; the docstring now says it edits only the down_proj column, and the code never edited the gate (backend/larql_vault.py:13-15, 475-478).

### Not established
- What an overlay costs per forward in seconds on the 5090, for one edit or for many, has not been measured anywhere I read.
- Whether llama.cpp's Gemma 4 graph applies a LoRA to tensors other than ffn_down and attn_output is not established; the repository read its source only for those two.
- How far the direction UPDATE writes (not divided by the post-feed-forward norm's weight) is from the direction the stream receives on the 31B is not established.
- Whether a census run or an index build records the vault's edits in force in its meta is not established.
- Whether a row overlay, or any overlay on an attention projection, has ever been run on the 31B is not established; the checks use stand-ins.
- What the lab daemon's /exec allows beyond its header is not established; I read only labd.py:1-34 and the lab kit's outline.
- The date and the unpacker of the 12.6 s heard edit are not stated where the figure is recorded.
- Whether --vault export has been run on the real 31B runtime is recorded as not measured (ARCHITECTURE.md:1857-1859).


## agent_harness — The scientist: the agent harness that already exists (backend/mcp_scientist.py, backend/scientist_mcp.py, backend/scientist_tools.py, backend/scientist_chat.py, the two prompt files, their check scripts, ARCHITECTURE.md section 4.8)
Full report: experiments/results/weight_space_exploration/code/agent_harness.md

### Summary
Theodolite has two agent tool surfaces with near-identical file names. The older one, backend/mcp_scientist.py, is a FastMCP server of 5 tools that act directly on a loaded runtime, count layers from 0, hold the project's best statistics (permutation p, paired sign-flip, Benjamini-Hochberg) and the only weight-editing tool an agent has (query_larql_vault: one down_proj column, rank one); its functions also back four HTTP routes of the server, but run standalone it loads a second copy of the 31B model. The current one, backend/scientist_tools.py, is 35 tools that are thin HTTP proxies onto the atlas's routes, count layers from 1, and contain no weight edit, no raw weight read and no code execution; it is what the chat in the atlas's side panel uses, reached by Claude Code and Codex through a standard-library stdio MCP server (backend/scientist_mcp.py) and by any OpenAI-compatible tool-calling endpoint through a 40-round loop the server runs itself. The loop is conversational and person-in-the-loop: every live reading waits on one asyncio lock for the one GPU, there is no budget, no job queue, no experiment ledger and no cost record, and at the default reach no backend can execute code that reaches the model. What it does have, and a weight-space harness could stand on, is a single tool registry and dispatcher with three transports, a proposal pattern in which the agent's writes wait for a person, a blind test whose truth is held server-side and judged by a fresh-conversation reader, tested reach and credential rules, a work folder served to the person with no scripts, and a strong written method (label gate, layer counting, lens cautions, hold-out and near-miss tests).

### Key facts
- There are two separate tool surfaces: backend/mcp_scientist.py is a FastMCP server of 5 tools working directly on a TheodoliteRuntime with layers counted from 0, and backend/scientist_tools.py is a list of 35 tools that proxy to the atlas's HTTP routes with layers counted from 1. [backend/mcp_scientist.py:81,174,206,949,1201,1567; backend/scientist_tools.py:191,534]
- The chat a person uses (the scientist tab) runs only the scientist_tools.py surface; it is the current scientist, last changed 13 November 2026, while mcp_scientist.py dates from 16 October 2026 and survives as the implementation behind four server routes and as a standalone server. [backend/scientist_chat.py:84,1058,1085; backend/server.py:65-71,1470-1545]
- None of the chat scientist's 35 tools edits a weight, reads a raw weight or runs code; the only agent-facing weight edit is query_larql_vault on the older FastMCP surface (also POST /api/larql), and an edit there moves one column of one layer's down_proj, rank one, with nothing solved for. [backend/scientist_tools.py:191-495; backend/mcp_scientist.py:1569-1599]
- Run standalone (python -m backend.mcp_scientist) the FastMCP module loads its own 31B runtime, so it cannot run beside the instrument's server on one card. [backend/mcp_scientist.py:117-139,1602-1609; README.md:485]
- The only bridge from the chat scientist to the older surface is its causal_check tool, which posts to /api/scientist/causal_check with a reduced argument set (layer, neuron, face, prompt, chat, k) and cannot ask for the study's 36 texts, the frozen-norm write or tracked ids. [backend/scientist_tools.py:814-818]
- The permutation, paired sign-flip, peak, either and Benjamini-Hochberg statistics live in test_motif_hypothesis on the older surface (and POST /api/scientist/test_hypothesis) and are not among the chat scientist's 35 tools. [backend/mcp_scientist.py:419-425,647-946,951-1126; backend/scientist_tools.py:498]
- Three driver backends exist and no others: Claude Code as a kept-open child process speaking stream-json, Codex as an app-server child speaking JSON-RPC, and any OpenAI-compatible Chat Completions endpoint with tool calling for which the server runs the tool loop. [backend/scientist_chat.py:97,562-581,824-851,1068-1140]
- The stdio MCP server Claude Code and Codex are given is standard library only, answers only initialize, tools/list, tools/call and ping, skips all notifications, and runs calls one at a time in arrival order, so it offers no progress, cancellation, resources or parallelism. [backend/scientist_mcp.py:72-100]
- A new model that speaks Chat Completions with tool calls plugs in with one entry in the scientist folder's endpoints.json (label, base_url, key_env, model, stream, headers, params) and no code change; a new agent CLI needs a new adapter with per-agent branches in send, interrupt, rewind, history and restart. [backend/scientist_chat.py:296-321,1218-1294,1297-1315,1438-1492]
- At the default reach (folder) no backend can execute code against the instrument: Claude Code has file tools in its work folder and no shell, Codex's shell runs in a sandbox with no network, and an endpoint model has the tools only; code that calls the HTTP API is possible only at the home reach. [backend/scientist_chat.py:101-108,232-238,565-572,843]
- Every live tool call from any chat, page or script waits on one asyncio lock for the GPU, shared between the server's own routes and the atlas's routes. [backend/server.py:99,1660; backend/atlas.py:121-124,148-151]
- The openai backend's turn stops after 40 tool rounds; Theodolite sets no round limit for Claude Code or Codex and no token, dollar or GPU-time budget for any backend. [backend/scientist_chat.py:111,1095-1097,576-581]
- One MCP tool call is a held HTTP request with a 900-second timeout, and there is no submit-and-poll job mechanism anywhere in the scientist. [backend/scientist_mcp.py:31,40-43]
- Claude Code's dollar cost per turn is emitted on the done event but the page's scientist files never read it and Theodolite does not log Claude chats, so no cost is recorded; of all tool replies only causal_spot_check reports its own timing. [backend/scientist_chat.py:450-451,657; backend/mcp_scientist.py:1550-1554]
- An interrupt stops the model's reply but no code in the scientist modules cancels a tool call already running: the openai loop awaits it and the MCP server's request runs on. [backend/scientist_chat.py:1104-1125,1297-1315]
- The agent's names and notes are proposals stored in field_proposals.json with the proposing chat's id, and nothing, later chats included, reads a proposal's text until the person keeps it on the page. [backend/scientist_tools.py:527-531,685-703; backend/field_notes.py:31-49]
- Pattern-test results persist in field_tests.json, latest first and at most six a face, each carrying the pattern it tested word for word so that a reworded pattern is seen to be untested; the neuron tool returns them to later chats. [backend/field_notes.py:38-46; backend/scientist_tools.py:573-576]
- The blind test holds its truth in server memory until answered and can be judged by an endpoint's model in a fresh conversation with no tools and no history, but the hold-out of kept-back passages is applied only in the tool (fold=fit) while the route's default shows every passage. [backend/atlas.py:1612-1614,1926,1986-2004; backend/scientist_chat.py:904-937; backend/scientist_tools.py:588]
- The work folder is one folder shared by every chat and every backend and is the only free-form memory a later session can read; files in it are served to the person under a policy that runs no script and fetches nothing. [backend/scientist_chat.py:122,152-155,1551-1568]
- Chat logs are not a ledger: codex and openai logs keep each tool result cut to 1,500 characters, and replies carry no fingerprint of the weights in force except vault_edits in causal_spot_check and vault_edits_in_force in the control-vector export. [backend/scientist_chat.py:113,803,1064; backend/mcp_scientist.py:1549; backend/scientist_tools.py:741]
- The label gate holds back a neuron's lens label below code layer 40 (page layer 41), where the lens and the naked eye share none of their first ten word-pieces, or where the neuron writes nothing; it was measured only at code layers 40-57 and never on edited neurons, which are flagged lens_validated false. [backend/label_gate.py:30-39,66-80]
- The chat scientist's dispatcher converts every layer argument from the pages' count (from 1) to the API's (from 0) and every reply back, refusing a layer below 1, while the FastMCP tools, the HTTP API, the gate's card and the notes' keys all count from 0. [backend/scientist_tools.py:40-79; backend/mcp_scientist.py:1016-1019]
- The prompt requires greedy readings, forbids sampling to make an answer read better, treats a census rank as a lead and never a finding, requires held-out prompts and a spare_bos false rerun before a recipe is handed over, and requires the written and blind tests before a name is proposed from passages. [backend/scientist_methodology.md:27-30,61-64; backend/scientist_prompt.md:105-137,197-200,244-291]
- The tool description for steer tells the agent that a batch's shape can move a row's numbers on the GPU, so a pushed answer must be read against the baseline of its own call; on the 31B 6 of 15 rows and 6 of 13 rows parted from their batch-of-three answers. [backend/scientist_tools.py:294-297; ARCHITECTURE.md:920]
- Batching exists inside single calls: steer takes up to 12 candidate push sets or 8 prompts or 8 samples within 6,144 word-pieces, compare takes 300 neurons by 24 texts, and test_motif_hypothesis takes up to 512 neurons at one forward per sentence. [backend/scientist_tools.py:288-294,355-357; backend/mcp_scientist.py:425,1054-1056]
- A Claude Code or Codex chat reads its prompt and tools at process start; when they change, only a Claude Code chat at the folder reach is restarted by itself, and any other is marked stale until the person asks for a restart because it may have background commands in its process group. [backend/scientist_chat.py:506-523,1511-1529]
- Neither CLI is given the endpoints' keys or the Hugging Face and cloud credentials in the server's environment, and Claude Code loads no settings file, hook, rules file or CLAUDE.md, because a CLAUDE.md the scientist wrote in its work folder once reached every later chat as instructions. [backend/scientist_chat.py:176-197,573-578]
- The harness is checked only against fakes: check_scientist.py plays all three backends with stand-ins on a tiny random model, and verify_scientist_cli.py runs the real claude binary against a scripted stand-in API, so no document records how a real driver model behaves in it. [scripts/check_scientist.py:2-52; scripts/verify_scientist_cli.py:2-12; ROADMAP.md:1517-1518]
- The runtime already has the raw weight primitives a read tool would need (slice_w_down, slice_w_gate, slice_w_up, the norm weights and layer scalars), used inside trace_downstream_readers, though no tool returns their numbers. [backend/mcp_scientist.py:282-290,319-324]
- The methodology's list of reference neurons and reference heads established in this atlas is empty. [backend/scientist_methodology.md:143-151]

### Numbers
- Tools on the chat scientist's surface: 35 in code (counted by loading the module); the documents say 30 and 32 [backend/scientist_tools.py:191-498; ARCHITECTURE.md:43,1132; README.md:239]
- Tools on the FastMCP surface: 5 [backend/mcp_scientist.py:174,206,949,1201,1567]
- Tool schema payload and fixed prompt size: 39,051 bytes of tools/list JSON; prompt 34,660 bytes plus methodology 12,345 bytes [backend/scientist_tools.py (measured by parsing, 14 November 2026); wc on backend/scientist_prompt.md and backend/scientist_methodology.md]
- Tool rounds in one openai turn: 40 [backend/scientist_chat.py:111]
- Characters of a tool reply an endpoint model reads: 16,000 [backend/scientist_chat.py:112]
- Characters of a tool reply shown on the page and kept in the codex and openai logs: 1,500 [backend/scientist_chat.py:113]
- HTTP timeout of one MCP tool call: 900 s [backend/scientist_mcp.py:31]
- Endpoint request timeouts: connect 15 s, read 600 s, write 60 s, pool 15 s [backend/scientist_chat.py:1088]
- Codex JSON-RPC call timeout: 120 s [backend/scientist_chat.py:740]
- Idle chat reaped: after 30 minutes without listeners or turns, checked every 60 s [backend/scientist_chat.py:110,536-546]
- Message and upload limits: 100,000 characters a message; 25 MB a file; 16 attachments a message [backend/scientist_chat.py:114,124,1155]
- steer batching limits: 12 candidate sets, 8 prompts, 8 samples, 32 pushes, 64 head scales, 6,144 word-pieces over the rows, max_new default 150 [backend/scientist_tools.py:173-185,288-294,316]
- steer wall time on the 5090 with the compiled unpacker: 3 rows 8.3 s; five prompts (15 rows) 9.2 s; six candidates (13 rows) 8.8 s; measured 11 November 2026 [ARCHITECTURE.md:920]
- The same three rows on the reference unpacker: 50 to 54 s (a scientist chat on 11 November 2026) [COMPUTE.md:63]
- One written word-piece on the 5090: 0.706 s reference, 0.275 s lut, 0.091 s compiled at 1 row; 0.743, 0.303, 0.126 s at 96 rows; torch peak above the model 0.3-1.0 GiB at 1 row and 7.2-7.5 GiB at 96; timed 28 October 2026 [COMPUTE.md:47-57]
- One heard vault edit: 12.6 s on the GPU [backend/server.py:1455]
- Batch-shape divergence of greedy rows on the 31B: 6 of 15 rows (five prompts) and 6 of 13 rows (six candidates) parted from their batch of three before 80 word-pieces; 11 November 2026 [ARCHITECTURE.md:920]
- compare limits: 300 neurons by 24 texts, one forward a text [backend/scientist_tools.py:355-357]
- test_motif_hypothesis constants: at least 4 sentences a side, at least 6 pairs, alpha 0.05, 10,000 relabellings exact or drawn with seed 0, at most 512 neurons a call [backend/mcp_scientist.py:419-425]
- causal_spot_check constants: k default 10, at most 200, floor k 0.01, top_n at most 50, 16 tracked ids, label P@10 against the top 100 [backend/causal_check.py:51-57]
- Longest text for a trace or causal check: 2,048 word-pieces [backend/engine_runtime.py:493]
- Pattern-test limits: written: 1-12 firing texts, up to 12 near misses, drop 1-64 (default 8), fires at 0.5 of the extreme, quiet under 0.25; blind: 4-12 pairs (default 10), 16-256 word-pieces before (default 60) [backend/scientist_tools.py:405-434; backend/pattern_tests.py:68-74]
- Pattern tests kept per face: at most six, latest first [backend/field_notes.py:38-41]
- Label gate: held back below code layer 40 (page 41); measured at code layers 40-57; graded-score AUC 0.84, the yes/no cut AUC 0.64 on 240 faces [backend/label_gate.py:10-13,30-36,53-62]
- Pattern-test pilot on the 31B: readers picked 177 of 196 pairs with the face's own pattern and 86 of 196 with another face's; 6 of 21 faces do not fire on their own passages with 60 word-pieces of context; 14 November 2026 [backend/pattern_tests.py:48-60]
- Claude Code version the reach rules were tried on: 2.1.312, on 28 October 2026 [ARCHITECTURE.md:1148; ROADMAP.md:1504-1506]
- check_scientist checks: 240 [ROADMAP.md:1336]

### Reusable for a harness
- The tool registry and single dispatcher in backend/scientist_tools.py (TOOLS at line 191, call at line 534, openai_tools at line 501) give a new tool to all three backends at once, and every tool call from every backend passes through call with the chat id, which makes it the natural place for a ledger and for wall-time accounting.
- The stdio MCP server backend/scientist_mcp.py is standard library only and works for any MCP client given THEODOLITE_URL, so an outside agent can attach to a running server without the chat tab.
- The OpenAI-compatible loop in backend/scientist_chat.py (_openai_turn, _complete, _run_tool, lines 977-1140) with profiles from endpoints.json is a model-agnostic driver path that already handles streamed tool calls, reasoning fields, mid-turn messages and interrupts.
- The proposal pattern (set_name and write_note in backend/scientist_tools.py:685-703 with backend/field_notes.py's field_proposals.json) is a working model for agent writes that wait for a person to keep or discard, which could carry weight edits as well as names.
- The blind test (backend/atlas.py:1939-2004, backend/pattern_tests.py, and ask_once in backend/scientist_chat.py:904-937) is a working model of a held-out evaluator: truth held server-side until answered, a kept-back fold, near-miss controls, and a reader in a fresh conversation.
- The statistics in backend/mcp_scientist.py (motif_statistics, paired_statistics, either_statistics, benjamini_hochberg, _correction, lines 647-946) are pure numpy functions with a reference check in scripts/check_hypothesis_test.py and can be reused for any two-condition comparison of edited against unedited behaviour.
- causal_spot_check in backend/mcp_scientist.py:1203-1557 reads an effect from the model's own odds with any vault edits in force, reports vault_edits and timing_ms, and caches its scale table per set of edits, so it already works on an edited model.
- The reach and credential rules in backend/scientist_chat.py (reach at line 170, child_env at line 190, _claude_cmd at line 562, the Codex sandbox at line 843) with scripts/verify_scientist_cli.py give a tested way to confine a driver CLI.
- The work folder and its file route (backend/scientist_chat.py work_dir at line 152 and work_file at line 1551, under WORK_CSP) already give the agent a files workspace whose output a person can open safely.
- The stale and restart mechanism (_agent_files_stamp and _take_up_changes, backend/scientist_chat.py:207-217,506-523) already handles tools and prompts changing under a live chat.
- The label gate (backend/label_gate.py label_check and LABEL_GATE_CARD) and the from_page and to_page layer conversion (backend/scientist_tools.py:40-79) are reusable guardrails that travel with every reply.
- The system prompt builder (system_prompt in backend/scientist_chat.py:247-268) assembles the prompt from two repository files plus the person's prompt.local.md with per-backend sections, so a weight-space method document can be added as a third file.
- The CPU-only check harness scripts/check_scientist.py, which plays every backend with fakes on a tiny random Gemma 4 over real connections, can be extended to cover new tools without the GPU.
- The lab daemon experiments/lab/labd.py (outside this area) already executes submitted Python against a loaded runtime with a persistent STORE dict and a per-launch token, and is the only existing precedent for free code execution and a tensor workspace.

### Gaps
- The chat scientist has no tool that edits weights; exposing the vault would be a new tool that posts to /api/larql.
- No tool returns raw weight numbers, and there is no server-side store of named tensors that tools could refer to by handle.
- The scientist has no code execution against the model: the tools are HTTP proxies, the stdio MCP server holds no model, and the lab daemon that does execute code is a separate process no scientist tool reaches.
- There is no job queue or submit-and-poll interface for experiments longer than one held HTTP request.
- There is no experiment ledger recording hypothesis, prediction, call, full result, weights in force and outcome.
- There is no evaluator whose data the agent cannot reach: the one hold-out that exists is enforced in the tool, not at the route.
- There is no cost accounting: no wall time, GPU seconds, rows, tokens or dollars are recorded per call or per chat.
- There is no budget or stop rule for an unattended run, and no round limit for Claude Code or Codex chats.
- There is no pre-registration step in which a hypothesis and its pass criterion are fixed before the test runs.
- There is no backend for a native Anthropic Messages or OpenAI Responses API, and no generic adapter for a new agent CLI.
- There is no per-chat or per-experiment isolation of the work folder and no snapshot or naming of model state per experiment.
- The statistics on the older surface are not exposed to the chat scientist, and the field book is neither read nor written by any scientist tool.
- The stdio MCP server has no progress notices or cancellation, and nothing cancels a GPU reading once it has begun.
- No configured endpoint exists on this machine to act as a blind reader, so with Claude Code or Codex the agent answers its own blind test.

### Constraints
- One GPU process at a time: the instrument's server, the lab daemon and a standalone mcp_scientist each load the 31B model, and the daemon shares the server's default port 8765.
- Every live reading must take the server's single asyncio GPU lock, so a long agent job blocks the person's page and every other chat unless it yields between items.
- A tool reply is JSON text, cut to 16,000 characters for an endpoint model and shown at 1,500 on the page, so tensors cannot travel through the driver's context and must stay server-side.
- backend/scientist_tools.py must stay standard library only so the stdio MCP server starts fast and can be launched with python -I.
- The chat scientist's tools take and return layers counted from 1 while the HTTP API, the FastMCP tools, the label gate and the notes count from 0; any new tool must go through from_page and to_page.
- What the agent reads (model output, census answers, names, notes, attachments) can carry instructions, so code execution or weight edits must not be combined with the home reach without a further barrier.
- Agent writes to shared state are proposals that a person keeps; a new feature that lets the agent change the model should respect that pattern or state plainly why it departs from it.
- The label gate and the lens were never validated on edited neurons, so the instrument's word labels cannot serve as evidence about a neuron the agent has edited.
- Greedy answers change with batch shape on the GPU, so an evaluator must compare rows within one call or measure that noise.
- The reach rules were tried on Claude Code 2.1.312 only and must be re-run with scripts/verify_scientist_cli.py for any new CLI version.
- A Codex chat or a home-reach Claude Code chat does not pick up changed tools until the person restarts it.
- The prompt and method files are the person's to change and the agent is told never to edit them; findings enter them only after the person's review.
- Every route answers only this machine's pages and scripts run by the same account, so a remote driver must reach the server through a local process.

### Surprises
- The file names invert expectation: backend/mcp_scientist.py is the older FastMCP circuit tools with their own runtime, and backend/scientist_mcp.py is the stdio server for the current chat; they share no tools except causal check by proxy.
- The current scientist cannot edit the model at all, although the project has a vault with rollback: the only agent-facing edit tool is on the older surface, which the chat does not use.
- The project's best statistics (permutation p, paired sign-flip, the paid-for choice between readings, Benjamini-Hochberg) are not available to the chat scientist; they serve the folio page and the older surface.
- The documents disagree with the code on the tool count: ARCHITECTURE.md line 43 and README.md line 239 say 30, ARCHITECTURE.md line 1132 says 32, the code has 35, and ARCHITECTURE section 4.8's list omits test_by_writing, blind_test and blind_test_answer.
- The docstring of backend/mcp_scientist.py opens by saying it exposes four core tools, then lists a fifth, and the code registers five.
- At the default reach no driver can run code that reaches the model: Claude Code has no shell and Codex's sandbox has no network, so the folder reach is tools-only in effect for all three backends.
- The blind test's kept-back half is hidden only by the tool passing fold=fit; the route's default is fold=all, so any script or a home-reach scientist can read it.
- Claude Code's per-turn dollar cost is put on the done event and then dropped: the page never reads it and Theodolite keeps no log of Claude chats.
- The work folder is shared by every chat and every backend, so one chat's files are another chat's input, with no isolation.
- A blind test that has been begun and not answered lives only in server memory and is lost on restart.
- The stdio MCP server runs tool calls strictly one at a time and ignores notifications, so parallel tool calls from a driver gain nothing and a cancel notice is never seen.
- After more than two weeks of the scientist existing, the methodology's reference neurons and reference heads sections are still empty, and no check or document records a real driver model's behaviour in the harness.
- The label gate's card states that it was never measured on edited neurons, which means the instrument's own word labels stop being evidence exactly where a weight-space agent would be working.

### Not established
- Wall times on the 31B for logprobs, activations, compare, top_firing, attn_drive, steer_lens, inspect_pole, trace_downstream_readers and a single causal_spot_check were not found in the files read.
- Whether a dropped HTTP connection cancels a GPU reading already under way in the atlas's routes was not established.
- Any turn or tool-call limits that the Claude Code and Codex CLIs apply themselves, and how Claude Code truncates a long MCP tool reply, were not established.
- Whether vault edits persist across a server restart, and the vault's full statement grammar and rollback rules, belong to the vault reader's area and were not established here.
- Whether a census run and the server can hold the card at the same time was not established; the pattern-test pilot stopped the census to run.
- How a real driver model behaves in this harness was not established: every check uses fakes or a scripted stand-in, and no transcript-level evaluation of a scientist session is recorded in the documents read.
- How many forward passes attn_drive, top_firing and attention use per call was not checked in backend/atlas.py.


## atlas_data — The atlas and census as documentation of the weights
Full report: experiments/results/weight_space_exploration/code/atlas_data.md

### Summary
The atlas holds a large precomputed map of the clean Gemma-4-31B checkpoint: for all 60 x 21,504 feed-forward neurons there are lens words, activation extremes, the 24 strongest weight links each way, a steer lens, stored passages and a measured check of the links (3.5 GB), and for 12 layers (36 and 40-50) a whole causal census against 2,048 random controls (about 2.8 GB a layer). The map describes the unedited model only: every builder and census run is a separate command-line process on a freshly loaded checkpoint that never applies vault edits, and the identity an index carries names the checkpoint's files in the Hugging Face cache, not the tensors in memory, so a vault edit changes no identity and no index is refused. Staleness is handled by hand-written per-route flags that know one edit type (a rank-one down_proj column of one neuron), and several products are served unflagged after an edit, the stats index that gives every push its unit and the whole census among them. A full rebuild costs about 13 hours of the card without the census and 13.2 hours more per census layer, so a harness must invalidate selectively and re-read live. Trust is uneven by depth: lens labels are held back below layer 40, the steer lens is checked sound only at 39-52, census doses are calibrated only at 32-52, link sizes are real everywhere but the served link sign is below chance at an ordinary position. No HTTP route starts or cancels a long job, nothing arbitrates the card between processes on the home machine, and the only rehearsal object is a 6-layer random Gemma 4 that exercises plumbing and cannot test a hypothesis about what weights compute.

### Key facts
- The index folder is atlas_data/index in the checkout (or $THEODOLITE_ATLAS_DIR/index) and holds seven named index files that the AtlasIndex class reads: lens.npz, act_scale.npz, wiring.npz, steer_lens.npz, layout_writes.npz, passages.d/ and wiring_check.npz. [backend/atlas_index.py:82-111]
- Every index array is shaped [60 layers, 21504 neurons, ...] and every one of the seven indexes on disk covers all 60 layers, except the wiring check, which covers layers 0-58 because layer 59 writes into no later layer. [atlas_data/index/*.npz array headers (read 14 Nov 2026); backend/atlas_build.py:659-661]
- The identity an index records is the weights card: for the checkpoint it is the repo id plus files_sha256, a hash of the names of config.json and the safetensors files and of the cache blobs they link to, so it identifies files on disk and is not a hash of the weights in memory. [backend/runtime_factory.py:228-272]
- The server refuses an index built for another model id, other weights, another shape or an older format, and GET /api/atlas/arch reports the reason; when either side has no weights card the comparison falls back to the model id alone. [backend/atlas_index.py:173-196 and 349-371]
- Three index files on disk (lens.npz, act_scale.npz, layout_writes.npz, all built 27 October 2026) carry no weights card in their meta, so they are accepted on the model id alone; wiring, passages, steer_lens, wiring_check and steer_dose.json carry weights id e0a93173c31ed2ec. [atlas_data/index metas (read 14 Nov 2026)]
- A vault edit does not change the weights card, so no index is refused after an edit; the atlas's only key for the edited model is the tuple of vault patch ids in force and the set of edited (layer, neuron) pairs. [backend/atlas.py:215-219; backend/larql_vault.py:233-239]
- After a vault edit the stats index is still served unchanged: Atlas.scales() returns the index whenever it is built, so the push unit p50, every activation score and the link scale come from the clean model with no stale flag. [backend/atlas.py:273-293]
- No census route consults the vault's edits, so census words and ranks are shown unflagged after any edit; the census is selected by model id alone. [backend/atlas.py:828-1245; backend/census.py:1011-1044]
- The staleness flags that do exist cover: the edited neuron's own index words (stale), its own links (read live instead), its steer-lens entry (withheld), passages of the edited neuron and of every later layer, and the sign in use for readers after an edit. [backend/atlas.py:257-269, 1263-1275, 1363-1372, 1382-1403, 1602-1608]
- Other neurons' upstream link lists that include an edited neuron, other neurons' steer-lens entries, head profiles, the wiring check's summary and the last text run are all served from before the edit with no flag. [backend/atlas.py:1363-1372, 1263-1269, 2836-2858, 2586-2594]
- The index builders and the census run never construct a vault or apply patches: each loads a fresh runtime, so every index and census on disk describes the unedited checkpoint. [scripts/atlas_build.py:228-232; backend/atlas_build.py:2-4]
- The lens index is a pure function of weights (one down_proj column, the post-feed-forward norm, the layer's J++ matrix, the final norm and the unembedding) and took 144 s for all 60 layers; it can be rebuilt for chosen layers in place. [backend/atlas_build.py:63-108; ROADMAP.md:328]
- The wiring index keeps only each neuron's 24 strongest links each way with their two weight products g and u; a pair outside the top 24 is not on disk and no HTTP route returns the products of an arbitrary pair. [backend/atlas_build.py:259-366; backend/atlas_index.py:498-506]
- Measured on the 31B, a link's existence and size are real (linked readers moved 5.5 times as much as unlinked ones, rho 0.89 against the first-order size where the reader sits) but the served signs are below chance at an ordinary position: 0.336 from rest and 0.418 in use, against 0.841 when scored where the reader actually sat. [ARCHITECTURE.md:772-800; atlas_data/index/wiring_check.npz meta]
- The passages index stores, for each face of every neuron, its 16 strongest events and 6 random events above half its extreme over 2,000,054 word-pieces in 3,914 windows of 512, with the gate and up pre-activations at each, and the window ids can be read again exactly with POST /reads. [backend/atlas_build.py:161-252; atlas_data/index/passages.d/meta.json; backend/atlas.py:1600-1601]
- The steer lens was built for all 60 layers at a per-layer dose in 10.5 hours after a 38-minute dose pilot, and its own check on disk rates it 'agrees' at layers 39-52, 'weak' at 32-38, and 'unchecked' at 0-31 and 53-59. [ROADMAP.md:369-382; atlas_data/index/steer_check.json]
- The code treats an unchecked steer-lens layer as weak only when its nearest checked layer is weak, so layers 0-31 are treated as weak and layers 53-59 are shown as usable although no census checks them. [backend/atlas_index.py:316-327]
- The label gate holds a lens label back below layer 40 and was validated only for the jpp-base lens in layers 40-57. [backend/label_gate.py:30-64]
- Whole-layer censuses exist on disk for 12 layers: layer 36 (design match_ceil) and layers 40-50 (design break_rel), each with 21,504 neurons, 2,048 controls and 565,254 answers in a 2.75-2.78 GB file. [atlas_data/census/runs/*.meta.json (read 14 Nov 2026)]
- Census doses count as calibrated only for break_rel at layers 39-51, soft_ceil at layers 38 and 52, and match_ceil at layers 32-37; no push strength has passed a trial below layer 32. [backend/census.py:268-277; ARCHITECTURE.md:825]
- A whole census layer costs 13.2 hours on the home RTX 5090 and a trial of about 1,022 neurons against 1,024 controls costs about 70 minutes. [ROADMAP.md:711 and 725]
- The census is read by seeking through two side files (a rows index and a units table), so a layer's list opens in 0.13 s and one neuron's 24 answers in 0.01 s. [backend/census.py:1201 and 1290; ROADMAP.md:1396-1397]
- All 53 /api/atlas routes answer only loopback requests from this machine's own pages or scripts, and none of them writes a weight: pushes act on activations and the residual stream for one request. [backend/atlas.py:84-114]
- Every route that runs the model takes one in-process asyncio lock, so live readings are strictly serial and wait behind a push chat; GET /neuron/lens takes that lock even though it only reads weights. [backend/atlas.py:148-165 and 1311-1319; backend/server.py:99]
- No HTTP route starts, watches or cancels an index build or a census; they are command-line processes that each load the model (21.2 GiB of the 32 GB card), and on the home machine only convention stops two from running at once. [scripts/atlas_build.py:19-23 and 173; scripts/census_intake.py:389]
- The server picks up a finished job's output without a restart: an index file is reloaded when its mtime changes and census runs are looked up again every five seconds; index files are written atomically. [backend/atlas_index.py:155-160 and 329-347; backend/atlas.py:843-853]
- Resumable jobs are passages (parts of 1,024 windows), wiring_check and steer (one shard a layer), steer_dose (one layer at a time) and the census (planned batches, SIGINT stops between steps); stats, wiring and layout are single passes with no resume and no per-layer option. [backend/atlas_build.py:210-216, 655-658, 906-908, 1028-1031; scripts/census_run.py:313-317]
- The stand-in is a real Gemma4ForCausalLM with 6 layers, hidden size 64, feed-forward width 128 and random weights on the CPU, wrapped in the instrument's own runtime and vault, and the whole server, the builders and the census run on it. [scripts/atlas_stand_in.py:44-48, 102-142, 200-243]
- No smaller real Gemma 4 loads today: the GGUF path refuses the mixture-of-experts model (26B-A4B) and the per-layer-embedding models (E2B, E4B). [backend/gguf_model.py:43-44 and 213-219]
- A single neuron's effect is capped by the norm after its block (its write ceiling), and the census's and steer lens's pushes are mostly larger than that cap: 1 of the stream is 1.76-2.76 ceilings at layers 39-51 and 5 to 19 ceilings at layers 2-25. [ARCHITECTURE.md:694-697; ROADMAP.md:740-742; backend/census.py:19-23]

### Numbers
- Index folder size: 3.5 GB (lens 249 MB, act_scale 31 MB, wiring 510 MB, passages about 690 MB, steer_lens 129 MB plus 134 MB of shards, wiring_check 510 MB, layout 2.6 MB) [du and ls of atlas_data/index, 14 Nov 2026]
- Census folder size: 61 GB (runs 32 GB, volunteer bundles 22 GB, trials 4.0 GB, nulls 3.3 GB) [du of atlas_data/census, 14 Nov 2026]
- Lens index build time: 144 s for 60 layers, about 2.6 s a layer, 1.5 GiB above the model [ROADMAP.md:328; scripts/atlas_build.py:24-25 (measured 27 Oct 2026)]
- Stats index build time: 58 minutes over 2,000,054 word-pieces with the reference unpacker, 3.4 GiB above the model [ROADMAP.md:329; scripts/atlas_build.py:26-28 and 54 (27 Oct 2026)]
- Wiring index build time: 224 s on the RTX 5090 for 1,770 layer pairs [scripts/atlas_build.py:29-34 (measured 28 Oct 2026)]
- Layout build time: 9 minutes [ROADMAP.md:330 (27 Oct 2026)]
- Passages index build time: 20 minutes with the compiled unpacker over the same 2,000,054 word-pieces [scripts/atlas_build.py:51-55; ROADMAP.md:1490-1491 (28 Oct 2026)]
- Wiring check time: 1,840 s for 59 layers (about 31-32 minutes), 68 prefills of 63 rows a layer [wiring_check.npz meta cost, summed; ROADMAP.md:1473-1474 (29 Oct 2026)]
- Steer-dose pilot time: 38 minutes for 60 layers, 128 neurons a layer at 9 doses [ROADMAP.md:369-370 (13 Nov 2026)]
- Steer lens build time: 10.5 hours over three sittings (2,702 s + 3,395 s + 31,646 s); one layer 287 s at layer 45 and 1,181 s at layer 5 [ROADMAP.md:346-349 and 380-381 (29 Oct and 13 Nov 2026)]
- Whole census layer: 565,254 answers, 13.2 hours on the home RTX 5090, 2.7 GB kept [ROADMAP.md:711; L45 rows index on disk]
- Census trial: about 49,000 answers and 70 minutes for 1,022 neurons against 1,024 controls [ROADMAP.md:722-725]
- Model memory and load: 21.2 GiB of the card for the model and J++ lens, about 16 s to load [scripts/atlas_build.py:22-23 (27 Oct 2026)]
- Decode step: 0.71 s with the reference unpacker whatever the batch; 0.103 s at 32 rows and 0.126 s at 96 rows with compiled [ROADMAP.md:443-445 (28 Oct 2026)]
- Steer call latency: 8.3 s for three rows, 9.2 s for five prompts, 8.8 s for six candidates, with compiled [ARCHITECTURE.md:920 (11 Nov 2026)]
- Census read latency: 0.13 s to open a layer's list and 0.01 s for a neuron's answers, after 2.4-3.4 s once a layer [ROADMAP.md:1396-1397 (12 Nov 2026)]
- Wiring check medians over 59 layers (linked readers): sign from rest 0.336; sign in use as served 0.418; sign in use at the reader's site 0.841; rho 0.89 against first-order size and 0.31 against |P| f^2; linked moved 5.5 times the null [wiring_check.npz meta; ARCHITECTURE.md:796-800]
- Steer lens agreement with census (neurons above 95 in 100 controls): 0.07 at layer 32, 0.185 at 36, 0.33 at 38, 0.62 at 39, 0.60 at 40, 0.86 at 45, 0.84 at 51, 0.75 at 52; 0.01-0.05 shuffled [atlas_data/index/steer_check.json (written 14 Nov 2026)]
- Passages corpus: 2,000,054 word-pieces in 3,914 windows of 512; 16 strongest plus 6 sampled events a face [atlas_data/index/passages.d/meta.json]
- Wiring index links: 24 links each way a neuron, 30,449,664 live downstream links [wiring_check.npz meta sign_disagreement; backend/atlas_index.py:71]
- Route count: 53 routes under /api/atlas in backend/atlas.py, 21 of which take the GPU lock [backend/atlas.py (grep of @router and on_gpu)]
- Stand-in size: 6 layers, hidden 64, feed-forward 128, vocabulary about 600; its whole index set is 1.6 MB [scripts/atlas_stand_in.py:44-48; du of atlas_data/stand-in]
- Human labels: 8 named neurons, 0 named heads, no field book, pattern tests kept for 21 neurons [~/.local/share/theodolite/*.json (counted 14 Nov 2026)]
- Request caps: 32 pushes and 64 head scales a reading; 2,048 positions for /reads, 256 for /run, 512 for head readings; 300 neurons x 24 texts for /compare; 12 candidates or 8 prompts for /steer [backend/atlas.py:59-75; ARCHITECTURE.md:689-690]

### Reusable for a harness
- AtlasIndex (backend/atlas_index.py:216) is a numpy-only reader of every index with model, shape and weights refusal, mtime reload and plain-word refusal reasons; a harness can open a second index folder for an edited model with AtlasIndex(root=..., model, shape, weights).
- weights_card and weights_differ (backend/runtime_factory.py:251, backend/atlas_index.py:173) are an existing provenance scheme that every index, census run and export already records; it needs extending to name edits, not replacing.
- save_npz (backend/atlas_index.py:155) writes index files atomically, and the plan.json pattern in build_passages and build_wiring_check (backend/atlas_build.py:191-203, 616-636) refuses a resume against changed settings or changed sibling indexes.
- build_lens with layers= (backend/atlas_build.py:63) re-reads the weight-only lens words of chosen layers in about 2.6 s a layer, the cheapest index to refresh after an edit.
- build_steer_layer with neurons= (backend/atlas_build.py:776) measures the next-token effect of pushing a chosen subset of one layer's neurons by caching the stream at the layer and running only the later layers through atlas_ops.forward_from_layer (backend/atlas_ops.py:1018); the same trick makes testing an edit at layer l cost only the layers after l.
- build_wiring_check (backend/atlas_build.py:568) is a template for a measured test of a weight-derived claim: random writers, a matched null of unlinked readers, per-layer shards, a summary with shares set against 0.5 and a cost line.
- atlas_ops.Weights (backend/atlas_ops.py:128) gives live accessors for w_down, w_gate, w_up, norm weights, layer scalars, write directions and write ceilings with vault edits included, and atlas_ops.neuron_wiring and pair_products (backend/atlas_ops.py:1536, 1616) compute links for one neuron or arbitrary pairs live.
- The live routes POST /logprobs, /reads, /compare and /steer with candidates (backend/atlas.py:2453, 2428, 2490, 2350) already give one-forward hypothesis tests: base and intervened rows with KL, one neuron over exact token ids, 300 neurons over 24 texts, and up to 12 candidate interventions in one batch in about 9 s.
- The passages index's 3,914 stored windows and each face's stored events with their token ids (atlas_data/index/passages.d, GET /neuron/passages) are a fixed evaluation corpus for before-and-after reads of a neuron through POST /reads.
- The census machinery (backend/census.py): control_dirs for norm-matched random controls, plan_batches for batch-shape determinism, take_census with cancel and resume keys, runtime_record, canaries, empirical p and false-discovery q, and question halves as a held-out test; a 70-minute trial is the existing unit of a pre-registered causal test.
- RowsIndex and UnitsTable (backend/census.py:1201, 1290) read multi-gigabyte result files by seeking and rebuild themselves when the file changes, a pattern for any large experiment log the harness keeps.
- The pattern tests (backend/pattern_tests.py; atlas routes at backend/atlas.py:1765-2004) with the kept-back fold of odd-numbered windows (backend/atlas_index.py:125) are an existing anti-self-deception device that pushes nothing and works at every layer.
- Atlas.edit_key() and Atlas.edited() (backend/atlas.py:215-219) and the caches keyed by them are the existing staleness plumbing to generalise.
- scripts/atlas_stand_in.py with --serve, the --stand-in flag on atlas_build.py and census_run.py, and scripts/check_atlas_api.py and check_atlas_index.py are a CPU rehearsal and regression harness for the edit loop's mechanics, vault edits and the GPU lock included.
- steer_check (backend/steer_check.py, read through AtlasIndex.steer_check and steer_weak) is a working example of one index being validated against another and of the pages withholding a reading where it failed.
- GET /api/atlas/arch (backend/atlas.py:652) is an existing self-description an agent can read first: shape, which indexes are built or refused and why, which layers have a census, push units and caps, and the measured wiring check.

### Gaps
- There is no content identity for an edited model: nothing hashes or names the weights in memory, so an index, a census or a result cannot be tied to a particular set of edits except by vault patch ids held in the running server.
- There is no dependency model of what an edit invalidates: staleness is a set of per-route strings written for one edit type, with no whole-index or per-layer invalidation record.
- There is no staleness handling for the stats index, the census, other neurons' steer-lens entries, other neurons' upstream link lists, head profiles or the last text run.
- There is no way to build an index or take a census on an edited model: builders and census runs load the clean checkpoint and never apply vault patches.
- There is no HTTP or tool interface to start, queue, monitor or cancel a build or a census, and no job record a driver model could poll.
- There is no arbitration of the single card between the server and job processes on the home machine; only rented pods have a file lock.
- There is no per-neuron or per-layer incremental rebuild for stats, passages or wiring; a steer-lens shard must be deleted by hand before its layer can be rebuilt.
- There is no route that returns raw weights (a column, a row, an attention matrix) or the products of an arbitrary neuron pair, and no spectral or low-rank index of any weight matrix.
- There is no index over attention weights, embeddings or the unembedding beyond live head roles; head_profiles.json has never been written on the 31B's index folder.
- There is no resting operating point for readers, so the served sign in use is read where a neuron fires hardest and is wrong at an ordinary position.
- There is no small real model to rehearse hypotheses on: the stand-in's weights are random and the GGUF path refuses the small Gemma 4 models.
- There is no causal reading with a null below layer 32 and no whole census outside layers 36 and 40-50, so for most of the model the only readings not known to be weak are stored passages and the few pattern tests.
- Human or scientist labels are nearly absent: 8 named neurons, no field book, and pattern-test results for 21 neurons.
- There is no per-client run state: POST /run keeps one server-wide slot and every live reading shares one lock, so two agents cannot work at once.

### Constraints
- Every index and census on disk describes the unedited checkpoint, so a harness must treat them as a map of the baseline and never as a description of the model after an edit.
- A full rebuild of the non-census indexes costs about 13 hours of the card (the steer lens alone 10.5 hours) and each census layer 13.2 hours, so nothing can be rebuilt per edit; only the lens words (seconds a layer) and single-neuron live reads are cheap.
- Only one process can hold the model on the 32 GB card (the model and lens take 21.2 GiB), so a build or census and the server cannot run together on the home machine.
- Live readings are strictly serial behind one lock, and on the GPU a row's numbers depend on its batch's shape, so a pushed or edited row must be compared with the baseline row of the same call and never across calls.
- The atlas routes answer only loopback requests from this machine, so a driver model must run on the machine or go through a local proxy that the guard accepts.
- Trust is depth-dependent and a harness must surface it: lens labels are held back below layer 40, the steer lens is weak or unchecked outside 39-52, census doses are calibrated only at 32-52, and served link signs are unreliable at every layer.
- A weight link from the index is a lead about size, not a circuit: the index's own notes and the measured check say so, and about nine readers in ten sit with the gate shut at an ordinary position.
- A single neuron's write is capped by the norm after its block, and census and steer-lens pushes exceed that cap by up to 19 times at early layers, so what a forced push shows is not what an edit of that neuron's weights can achieve.
- Census and steer-lens evidence are two readings of the same push and neither says what a neuron does unpushed; passages and pattern tests are the only unpushed evidence.
- A census run's settings, batch size and plan are resume keys and a design's calibration is granted one layer at a time by a passed trial plus the owner's word, so a harness may not widen DOSES_CALIBRATED or mix runs on its own.
- The passages index holds the corpus's own text and must stay in the git-ignored atlas folder and never be committed or published.
- Index folders hold one model's files: a build for another model id or other weights is refused unless --replace is given, so an edited model's indexes need their own folder.
- Layers count from 0 in every file, field and command, while the pages and the scientist's tools show layer + 1.

### Surprises
- The weights card that guards every index names checkpoint files in the cache, so it cannot tell an edited model from a clean one; the staleness a designer would expect from 'keyed by weights' does not exist for edits.
- The stats index, which sets the unit of every push and every activation score, is served unflagged after a vault edit, while the fallback used when no stats index exists is correctly keyed by the edits in force.
- GET /neuron/lens, a weight-only read, takes the GPU lock and so waits behind a push chat, while /neuron/wiring from the index does not.
- Three of the seven index files and eleven of the twelve whole census layers predate the weights card and are accepted on the model id alone.
- The served 'sign in use' of a link is right only where the reader fires hardest; at an ordinary position both served signs score below chance, and the check that showed this is the one whose summary /arch serves.
- The project summary says the steer lens is sound from layer 39 up, but the check on disk covers 39-52 only; layers 53-59 are unchecked and the code shows their words as usable because the nearest checked layer agrees.
- ARCHITECTURE.md lists head_profiles.json as an index file, but it does not exist in atlas_data/index; it is built only on the first request for a head's profile.
- backend/atlas_build.py's docstring says the build costs are estimated from arithmetic and not measured, while scripts/atlas_build.py's help says they were measured on 27 October 2026; the help's steer-dose line still says 'not timed' although ROADMAP records 38 minutes.
- Five of the twelve whole census layers were written on a different card class (RTX PRO 6000 Blackwell) and are 'read as their own instrument'; their bits differ from the home card's.
- Any two stand-ins count as the same weights, so a stand-in built with another seed would silently reuse the first one's indexes.
- A census run's .t0, .t1 and .pauses files exist on disk but no script under scripts/ or backend/ writes them, so job timing is kept by hand or by tooling outside the repository.
- ROADMAP's plan for the remaining forty-nine layers is headed 'none of it run yet' while the text under it and the trials on disk record trials at layers 32-39, 51 and 52.

### Not established
- Which script writes a census run's .t0, .t1, .pauses and .commit files, and so exactly what interval they measure, is not established.
- How long the stand-in checks or a stand-in index build take is not established; no document read records it.
- Whether the vault can edit anything other than a down_proj column, and whether an edited model can be exported whole as a GGUF and indexed under its own identity, is not established here (another reader's area).
- Which unpacker the running server uses, and therefore today's real latency of each live route, is not established.
- The bodies of /push, /steer, /fires, /run/heads, /run/paths, /head/row, /head/pattern, /attention, /attn_drive, the pattern-test routes and the notes routes were described from ARCHITECTURE.md and their signatures and were not read line by line.
- The arithmetic inside census.py's ranking, dose-window and collapse functions was not followed; only its structure and data products were read.
- How much a single-neuron edit actually moves the stats index, the census or other neurons' steer-lens entries is not established; the report says what could change, not by how much.
- Whether a dense Gemma 4 smaller than the 31B exists that the GGUF path would accept is not established; only the refusals of 26B-A4B, E2B and E4B were read.


## lab_conventions — The lab daemon, experiment conventions and job handling
Full report: experiments/results/weight_space_exploration/code/lab_conventions.md

### Summary
The repository already contains most of the conventions a weight-space harness needs, but no job runner that fits it as it stands. The lab daemon (experiments/lab/labd.py) is a small FastAPI process that keeps the 31B loaded and executes submitted Python against it one job at a time; it ran the original study (314 jobs on 16 October 2026), is still maintained and tested, but has no recorded job since 17 October 2026, is called by nothing in the backend, and lacks cancellation, time limits, a non-blocking health route, and any reset or fingerprint of model state between jobs. GPU sharing is by exclusion: one process holds the model, the server serialises every model call under one asyncio lock, and batch jobs are separate processes whose wrappers refuse to start while port 8765 answers, so the server is simply down while a job runs; systemd-run scopes are an operator practice recorded in the owner's notes, not code. Decisions are gated by a strict, code-enforced registration pattern (a hashed prereg folder with rules, the exact runner and reader, inputs, the commit and a timestamp, committed before the result; all nine hash files I checked still verify), while pilots are unregistered and labelled as such. Tests are standalone scripts, not pytest: 27 CPU check scripts on a 6-layer random Gemma 4 stand-in and 8 GPU verify scripts, with no runner and no CI. The volunteer kit and its intake are a ready model for 'a job ran elsewhere and here is the checked result' (pinned spec, canary, run meta, manifest, a random re-run drawn after submission, tolerances measured beforehand, a provenance file), and the field-notes module shows the propose-then-a-person-decides pattern and how a test result is stored with the exact text it tested.

### Key facts
- The lab daemon loads one runtime at start through make_runtime(load_jlens=True) and wraps it in labkit.Lab, so it honours THEODOLITE_GGUF and THEODOLITE_DEVICE as the server does. [experiments/lab/labd.py:161]
- The lab daemon's POST /exec runs submitted code under a single asyncio.Lock, inside torch.no_grad(), with top-level await allowed, in a fresh namespace holding rt, lab, torch, np, F, STORE, labkit and asyncio; the job returns data by assigning a variable named result. [experiments/lab/labd.py:198-215]
- The lab daemon installs the 'reference' unpacker when THEODOLITE_UNPACKER is unset, whereas the server defaults to 'compiled' on a CUDA card, so a labd job runs at about 0.71 s a decode step instead of 0.09 to 0.13 s unless the variable is set. [experiments/lab/labd.py:157]
- After each job the lab daemon only removes hooks registered through Lab, clears the namespace and empties the CUDA cache; it does not restore or fingerprint weights, and STORE persists across jobs, so a job that edits a weight changes what every later job reads. [experiments/lab/labd.py:219]
- The lab daemon evaluates job code on the event loop itself rather than in a worker thread, so by the code its /health route cannot answer between a synchronous job's await points; this was read, not run. [experiments/lab/labd.py:213]
- The lab daemon has exactly three routes (/exec, /health, /reload): there is no cancel route and no time limit, and the check script asserts that route list. [scripts/check_local_only.py:369]
- The lab daemon refuses any non-loopback Host and any browser request with 403, and /exec and /reload require a per-launch random token written with mode 0600; the port is bound before the token is written and before the model loads. [experiments/lab/labd.py:98-115]
- The lab daemon writes the whole reply to a server-side path when the request names one, so a result survives a client timeout; labclient passes <out>.full.json and has a default timeout of 3,600 s. [experiments/lab/labd.py:228-233]
- labkit provides true gradients through the 4-bit weights by an autograd function that re-unpacks each packed linear in backward, with a GGUF twin that also hears vault overlays, and a same-position VJP built on them. [experiments/lab/labkit.py:417-519]
- Nothing in backend/ or frontend/src calls the lab daemon (every mention is a comment), it is not in .claude/launch.json, and the last recorded job in any checkout is dated 2026-10-17T04:59Z. [experiments/results/labd_exec_log.local.jsonl:8]
- The server holds the model behind one module-level asyncio.Lock named _GPU_LOCK, which every sweep, vault statement, scientist route and atlas route takes, the atlas routes receiving it through bind(). [backend/server.py:99]
- Server GPU work runs in a worker thread under the lock so the event loop keeps answering; a heard vault edit is recorded as taking 12.6 s. [backend/server.py:1452-1457]
- Cancellation in the server is cooperative and exists only for writing: a sweep's cancel_sweep is honoured at the next word-piece or pair, and a push chat stops when its page disconnects while keeping the lock until the writer thread ends. [backend/server.py:1345-1371]
- The server's /api/health reports the model, VRAM, unpacker and cards but not whether the GPU lock is held or who is waiting, unlike the lab daemon's /health which reports busy, waiting and running. [backend/server.py:1386-1441]
- A registered run script refuses to start unless its clean checkout is at the registered commit, has no uncommitted tracked changes, the folder's SHA256SUMS verifies, and nothing answers on 127.0.0.1:8765. [experiments/results/census_match_walk/prereg/night.sh:16-21]
- A registration folder holds RULES.md, the exact runner and reader scripts, the inputs, repo_commit, prereg_at and SHA256SUMS, and its RULES.md states the decision rules, what had been seen when it was fixed and what it leaves untested. [experiments/results/census_other_card_bars/kit4/prereg/RULES.md:3]
- The registered reader runs on the CPU from the hashed bytes, checks the hashes again and refuses to read a layer that already has a verdict, so a result is read once. [experiments/results/census_match_walk/prereg/read.sh:13-19]
- A change after hashing is recorded with its time and reason in AMENDMENTS.txt, and an addition gets its own SHA256SUMS.<name> and prereg_at.<name> so the first hashes stay untouched. [experiments/results/census_ceil_trials/prereg/AMENDMENTS.txt:1]
- The older R3 registration enforces itself in Python: require_committed refuses a GPU job while registered files differ from HEAD, and the gpu_job context manager caps CUDA memory, records VRAM and appends started, finished or failed to a runs log. [experiments/scripts/R3_gpu.py:385-427]
- Every check script hides the GPU and goes offline before importing torch, uses a module-level failures list with check() and section(), prints 'N checks passed, M failed' and exits 1 on any failure; it is run as .venv/bin/python -I scripts/<name>.py. [scripts/check_census.py:81-123]
- The CPU stand-in used by the checks is a random Gemma4ForCausalLM with 6 layers, hidden size 64, feed-forward width 128 and a sliding window of 8, and its output goes under atlas_data/stand-in, never among the model's. [scripts/atlas_stand_in.py:44-47]
- pytest is declared as a dependency, but the repository's own tree has no test_*.py, no tests folder, no conftest.py and no CI configuration. [pyproject.toml:21]
- On a pod or a volunteer's machine one GPU job holds the card through flock on <root>/logs/gpu.lock; the home machine's server does not take this lock. [scripts/pod/common.sh:104-112]
- census_run.py turns SIGINT into a stop between steps that keeps every whole batch, and the same command continues the run row for row; it has no port or free-memory guard of its own. [scripts/census_run.py:311-317]
- Every census run's meta carries a runtime record made by the job: the card's name, torch, CUDA, transformers, interp-engine, compressed-tensors, the git commit, whether the tree was clean, and the sha256 of the stats index. [backend/census.py:1809-1846]
- The intake treats a returned bundle as hostile: it unpacks only regular files with names a pack writes, under size caps of 4 GB for the run and 64 MB otherwise, and runs nothing from it. [scripts/census_intake.py:171-189]
- The intake draws its spot chunks from os.urandom only after the bundle has arrived and passed its paper check, so the sender cannot know which parts will be re-run at home. [scripts/census_intake.py:330-334]
- The intake refuses to judge a run from another card class where no acceptance bars have been measured for that design and layer. [scripts/census_intake.py:419-422]
- The scientist's set_name and write_note tools only create proposals; a proposal is read by nothing as a name, and it reaches the notes only through the keep route, which records the scientist as author. [backend/field_notes.py:519-579]
- A pattern test is stored as a measurement with its kind, the exact pattern text it tested, who ran it and when, latest first and at most six per face, so a reworded pattern is seen to be untested. [backend/field_notes.py:455-477]

### Numbers
- Lab daemon ready time and VRAM with both lenses: 13.3 s and 21.24 GB allocated (16 October 2026) [experiments/results/labd.log]
- Lab daemon study jobs: 314 jobs, 282 ok, median 71.3 s, longest 456.9 s, 27,453 s in total over a 41,837 s span (16 October 2026, my count) [experiments/results/labd_exec_log_2026-10-16.jsonl]
- Last recorded lab daemon job: 2026-10-17T04:59Z [experiments/results/labd_exec_log.local.jsonl]
- labclient default timeout: 3,600 s [experiments/lab/labclient.py:29]
- True VJP through the 4-bit weights: about 2 s per VJP, 23 GB peak (study, October 2026) [experiments/REPORT.md:90-93]
- Decode step with the reference unpacker: about 0.71 s whatever the batch from 1 to 64 rows (27 October 2026) [scripts/census_run.py:88-91]
- Decode step with the compiled unpacker: 0.09 to 0.13 s from 1 to 96 rows (28 October 2026) [scripts/census_run.py:94-96]
- Compiled unpacker self-test over 410 linears: 1.83 s (13 November 2026) [experiments/results/census_match_walk/L36-match_ceil-trial.meta.json]
- Heard vault edit: 12.6 s (date not stated) [backend/server.py:1455]
- A registered trial: 49,110 answers in 75 to 88 minutes a layer (13 November 2026) [experiments/results/census_match_walk/prereg/RULES.md]
- A whole layer's census: 565,254 answers, about 12.7 hours at 48 word-pieces (29 October 2026 rate) [scripts/census_run.py:109-110]
- CPU reading of a whole layer: about 20 minutes and 6 GB at nice 15 [experiments/results/census_scaled_design/read_whole.sh:5]
- Spot check: 5 chunks drawn at random, about a minute each on the 5090 with the model loaded for each [scripts/census_intake.py:21-22 and :57]
- Bundle size caps at intake: 4,000,000,000 bytes for the run and 64,000,000 bytes for any other file [scripts/census_intake.py:58-59]
- Bars measurement: about 30 minutes for 10 layers, 13 variants, 64 answers each (14 November 2026) [experiments/results/census_other_card_bars/kit4/prereg/RULES.md:34-37]
- Model at rest on the card with the lens freed: 17.97 GiB, the worst batch of 64 reaching 22.1 GiB (14 November 2026) [volunteers/kit.env:46-47]
- Kit's free-memory floor: 23,900 MiB free on a card of at least 24,000 MiB [volunteers/kit.env:52-53]
- Desktop's share of the home card: 5.1 GB held, leaving 26.0 (14 November 2026) [experiments/results/census_other_card_bars/kit4/prereg/RULES.md:8-9]
- Stand-in model size: 6 layers, hidden 64, feed-forward 128, sliding window 8 [scripts/atlas_stand_in.py:44-47]
- Check and verify scripts: 27 check_*.py with 22,466 lines and 8 verify_*.py with 5,129 lines (my count, 14 November 2026) [scripts/]
- Registration hash files verified: 9 of 9 SHA256SUMS files under experiments/results verify today [experiments/results/*/prereg/SHA256SUMS]
- Pattern tests kept per face: 6 [backend/field_notes.py:82]
- Proposals that may wait: 5,000 [backend/field_notes.py:89]
- Pod job time limit margin: projected hours times 1.25, then SIGINT with a kill after 300 s [scripts/pod/census_shard.sh]
- atlas_data on disk: 67 GB (14 November 2026) [atlas_data/]

### Reusable for a harness
- The lab daemon's exec loop (experiments/lab/labd.py, routes /exec, /health, /reload) is a working resident-model job runner with serialisation, a server-side result file, a per-job log line and the weights card on every reply.
- The lab daemon's access guard (local_scripts_only and launch_token in experiments/lab/labd.py:98-115) is a tested local-only plus per-launch-token scheme, exercised by scripts/check_local_only.py section 8.
- labkit.Lab (experiments/lab/labkit.py) gives jobs tokenising, batched residual capture, residual hooks, generation, next-token logits and gradients through packed 4-bit weights (grad_linears and same_pos_vjp).
- The hashed registration folder with its refusing runner and once-only CPU reader (experiments/results/census_match_walk/prereg/: RULES.md, night.sh, read.sh, walk.sh, SHA256SUMS, repo_commit, prereg_at) is a template an agent ledger can follow file for file.
- The bars registration (experiments/results/census_other_card_bars/kit4/prereg/RULES.md and measure.sh) is the cleanest example of numbered decision rules, an amendment before data, and a runner that checks its own bytes.
- R3_gpu.gpu_job, require_committed, set_job_cap and log_run (experiments/scripts/R3_gpu.py:175-186 and 344-427) form a Python job frame that checks git state, caps CUDA memory and appends started, finished or failed to a JSON-lines log.
- scripts/gpu_log.py records an owner's GPU-hour approvals and sums usage against them, which is a ready budget ledger for agent-run jobs.
- census.runtime_record, file_sha256, write_json_atomic, settings_differ and jobs_digest (backend/census.py:1150, 1721, 1801-1846, 1906-1927) provide the run meta, hashing, atomic writes and resume rules a harness job record needs.
- The canary idea (scripts/census_run.py --canary-only, census.compare_canaries, census_volunteer.rows_fingerprint) is a cheap fixed slice run first to classify a runtime before long work.
- The intake's four steps (scripts/census_intake.py check, spot, judge, accept) and its work-folder files check.json, verdict.json and provenance.json are a ready shape for checking and recording a returned experiment.
- census_volunteer.compare_rows, judge and accept_bars (backend/census_volunteer.py:89-96 and 315-377) show how to judge by tolerances measured beforehand and to refuse where none exist.
- scripts/census_bars.py with its registered rule is a method for deriving a tolerance from the instrument's own arithmetic noise and from planted near misses.
- scripts/pod/census_shard.sh and scripts/pod/common.sh (hold_card, scrubbed) give a detached job with a flock card lock, start and end sentinels, a SIGINT timeout at projected time times 1.25, and an environment stripped of token-like variables.
- scripts/atlas_stand_in.py (stand_in, use_stand_in_atlas, --serve) is the CPU stand-in every new feature is expected to be dry-run and checked on.
- The check-script idiom (failures list, check, section, exit code; scripts/check_census.py:108-123) and the planted-failure tests of a judge (scripts/check_volunteer_kit.py section 5) are the house style for a harness's own tests.
- backend/field_notes.py (propose, decide, add_test, _write_atomic use, move-aside of unreadable files) is a ready propose-then-person-decides store with authorship and atomic writes.
- The server's on_gpu helper and push-chat cancel event (backend/atlas.py:148-153 and 2142-2193) show how to run GPU work in a worker thread under the one lock and stop it cooperatively.
- Clean git worktrees pinned at a registered commit under .claude/worktrees/ are the existing way to run a job from exact code while sharing the main checkout's venv and atlas_data.

### Gaps
- There is no job runner with a queue, cancellation, a time limit and status that stays answerable during a job; labd has none of the first three and its health route blocks during synchronous code.
- There is no snapshot, fingerprint or automatic rollback of model weights between jobs anywhere in the job-running code, and the lab daemon does not load the vault at all.
- There is no machine-wide GPU lock on the home card that the server, the lab daemon and batch jobs all take; exclusion rests on probing port 8765, on free-memory guards in two scripts, and on operator practice.
- There is no code that creates or manages systemd scopes for long jobs; the practice lives only in the owner's notes and in one script that waits on a scope name.
- There is no way to hand the card from the server to a job and back automatically; the server must be stopped by hand and restarted afterwards.
- There is no experiment ledger object: registrations are folders and commits made by hand, and nothing links a hypothesis, its registration, its runs and its verdict in one machine-readable record.
- There is no store for a hypothesis about a weight or for an edit proposal; field notes know only neurons, heads and faces with names, notes, patterns and tests.
- There is no single command that runs every check, no pytest suite and no CI.
- The lab daemon's job log records only label, success, seconds, error tail and model id, with no hash of the submitted code, no git commit and no registration reference.
- The lab daemon offers no sandbox: submitted code runs as the user with full access, so an agent-facing runner would need its own containment.
- The registered shell scripts hard-code absolute home paths and one copies from a session scratchpad, so they are records rather than portable packages.
- The analogue of a canary and of a spot re-run for a weight edit does not exist and would have to be defined.

### Constraints
- Only one process may hold the 31B on the card at a time: the lab daemon, the server and a standalone mcp_scientist each load it, and the lab daemon shares port 8765 with the server on purpose.
- Any model call inside the server must run under _GPU_LOCK, and long work must go to a worker thread so the event loop keeps answering.
- A result that gates a decision must be registered first: rules, exact runner, exact reader and inputs hashed and committed before any data, run from a clean checkout at the registered commit, read once on the CPU.
- A change after hashing must be written to AMENDMENTS.txt with time and reason before any reading, and a change after data exists must be reported as a deviation beside the result under the registered code.
- A rule fixed beforehand is applied as written even when a figure misses its bar by 0.001, and later looks must be labelled as reported and gating nothing.
- Decisions the documents reserve for the owner (opening a layer, widening a calibrated range, accepting on weaker evidence, pushing to the remote) must wait for the owner's word, which the registration quotes.
- New code is expected to ship a check_*.py that runs end to end on the CPU stand-in and, where it touches the card, a verify_*.py, each run with python -I and exiting 0 only when every check holds.
- Large or regenerable output must go under atlas_data (git-ignored), with only small verdicts, metas and hashes committed under experiments/results/<name>/, and third-party text kept out of git.
- Stand-in and dry-run output must land under atlas_data/stand-in and never among the model's files.
- A job's determinism class is fixed by its batch size and plan on the GPU, so a run must keep one batch size and a run with other settings is refused rather than mixed.
- An agent may propose names and notes but may not keep them; kept notes are the only ones any page, search or later chat reads.
- Side sessions commit on their own branches and do not push; merges and pushes go through review, by the owner's standing instruction.
- CPU readers should run at nice 15 beside a GPU job, since at default priority they slowed a census run.
- The lab daemon must never be exposed beyond loopback, because it executes arbitrary submitted Python.

### Surprises
- The lab daemon is maintained and security-tested but appears unused since 17 October 2026: recent experiments run as their own processes or through the running server's HTTP routes, not through labd.
- The lab daemon defaults to the slow reference unpacker while the server defaults to the compiled one, so the obvious 'fast iteration' tool is about seven times slower per step unless an environment variable is set.
- The lab daemon and the server share port 8765 by design, so they can never run together, and the port probe that protects batch jobs cannot tell which of the two is up.
- The systemd-run scope convention the task mentions is not in the repository's code at all; the only trace is one script that waits on a scope name, and the rule itself lives in the owner's private notes.
- census_run.py, the main GPU job, has no guard against a running server; only its wrappers and two sibling scripts check the port or free memory.
- The naming rule 'check is CPU, verify is GPU' is not strict: scripts/check_sweep_refactor.py needs the GPU and the model, and scripts/verify_atlas_writer.py has a CPU stand-in mode.
- pytest is a declared dependency but there is not one pytest test in the repository's own tree.
- Registration is enforced by shell scripts that hard-code the owner's home path, and the first registered runner copies its files from a session scratchpad path that no longer exists.
- The registered reader, not only the runner, is hashed, and it refuses to read a result twice; a design that missed its bar by 0.001 was recorded as a fail.
- Keeping a proposal is separated from proposing it only by the tool surface: the keep and discard routes have no check of their own beyond the local-only guard.
- The README's reproduce section still presents the lab daemon as the way to run experiments, while the three most recent experiment folders keep their scripts beside their results and do not use it; experiments/scripts has not changed since 28 October 2026.
- scripts/gpu_log.py, the budget ledger for rented cards, has never been used: its log file experiments/results/gpu_runs.jsonl does not exist.

### Not established
- Whether the lab daemon's /health really stalls during a synchronous job was inferred from the code and not run.
- Whether a job posted while the lab daemon is still loading waits for it, as its docstring says, was not run.
- Whether the scientist's CLI sandbox can call the proposals keep route directly was not established; the route itself has no guard beyond the app's local-only middleware.
- The server's present load time and resident VRAM were not measured; the figures of about 10 s and about 24 GB come from an owner's note of 18 to 19 October 2026.
- Whether any script creates systemd scopes was searched for and none was found; the practice is recorded only in the owner's notes.
- Whether the lab daemon has been used at all since 17 October 2026 outside its logged jobs cannot be told from the repository; no later log line and no live token file exist.
- How long a weight-edit hypothesis would take end to end on the lab daemon was not measured; only per-step and per-job figures for other work exist.


## evaluators — Measurements that could judge a weight edit
Full report: experiments/results/weight_space_exploration/code/evaluators.md

### Summary
Theodolite already holds a broad set of evaluators (next-piece log-probabilities and KL from fp32 logits, batched greedy or sampled writing with per-row interventions, activation reads at every scale, weight-only wiring, a causal check with a noise floor and a linearity guard, exact permutation tests, Benjamini-Hochberg, matched random-direction controls, held-out halves and kept-back passages), but almost all of it was built to judge PUSHES, which are per-row hooks that share one forward with an untouched row. A vault edit is a module-wide overlay on the dequantised weight, so as built an edited and an unedited row cannot share a forward, and the only existing measurement of an edit is the vault's voice: one question, greedy, 12 word-pieces, with a first-step KL. Three edit classes do have an exact per-row stand-in today (a column scaling or DELETE is a `mul` push, a head scaling is a `HeadScale`), and `forward_from_layer` already reruns only the layers from a given layer up without a cache, which is exactly what an edit at layer L needs for single-forward readings at (60 - L)/60 of the cost. What is missing is a per-row column-replacement hook, a multi-token KL or likelihood evaluator over held-out text, any capability or regression suite, paraphrase sets, sequential-edit stability, a noise floor for edits, and a ledger that pays for tests run across calls. The measured trust limits a harness must carry are sharp: lens labels are sound only at layers 40-57, push doses are calibrated only at layers 32-52, the steer lens agrees with the census only at layers 39-52, activation pushes saturate the post-feed-forward norm, a uniform rescale of a block is undone by the norm, weight-only wiring signs are below chance at an ordinary word-piece, and rows of different batch shapes differ (KL up to 0.014; an unpushed answer in a batch of 64 can part from itself after 22 word-pieces).

### Key facts
- A push is resolved to per-row tensors (adds, acts, muls, heads) and installed as forward hooks, so pushed and unpushed rows are rows of one forward. [backend/atlas_ops.py:267-282, 344-412]
- A vault edit replaces a column of the dequantised weight inside the linear's forward for every row, so as built an edited row and an unedited row cannot share a forward; every forward-based measurement therefore reads the edited model whenever an edit is in force. [backend/engine_runtime.py:356-375]
- A `mul f` push on neuron n is the same change as scaling column W_down[:, n] by f (f = 0 is the vault's DELETE) and a head scale is the same as scaling that head's o_proj input block; the only difference is that pushes spare the start mark by default and a vault edit acts at every position. [ARCHITECTURE.md:1774, 1781-1791]
- `next_logprobs` takes one text and exactly two rows (base, intervened) in one forward and returns [V] float32 log-softmaxes from fp32 softcapped logits; it does not batch over texts. [backend/atlas_ops.py:665-690]
- KL is reported in different directions by different evaluators: /logprobs gives KL(base || intervened), the steer lens and the causal check give KL(pushed || base), and the vault's voice gives KL(before || after). [backend/atlas.py:2477; backend/atlas_ops.py:1967; backend/causal_check.py:425-427; backend/larql_vault.py:176-180]
- The causal check, the steer lens and next_logprobs read logits through an fp32 GEMM on the final-normed state because bf16 logits swamp small effects, whereas the vault's voice takes the writing loop's own logits cast to float32. [backend/causal_check.py:8, 132-139; backend/engine_runtime.py:1727]
- `write_rows` writes several prompts in one left-padded batch with a KV cache, each row under its own pushes, greedy or sampled; when sampled every row of a sample shares one Exp(1) draw per step from a CPU generator seeded seed + s, so rows part only where their odds differ. [backend/atlas_ops.py:464-490, 502-617]
- `forward_from_layer(lm, start, hidden, attention_mask, position_ids)` runs layers start..last and the final norm on a given stream entering layer `start`, with no cache, rebuilding both attention masks and rotary embeddings, and returns final-normed states at every position; hooks run inside it as in a whole forward. [backend/atlas_ops.py:1017-1043]
- `forward_from_layer` has one caller, the steer-lens build, which reads four contexts to the stream leaving a layer once and then reruns only the later layers for each batch of 64 neurons, with the unpushed rows in the same forward. [backend/atlas_build.py:775-836]
- The causal check's given mode runs the text before the read position once at batch 1 and then only the last position for every row over that frozen prefix, which is exact for a push at the last position and is not exact for a weight edit that acts at every position. [backend/causal_check.py:19-21, 180-205, 268-283]
- The causal check sizes its push by each neuron's largest activation over the study's 36 texts and caches that table per set of vault patch ids in force, the one evaluator cache already keyed on the edit set. [backend/mcp_scientist.py:1175-1191]
- The causal check runs the same push at k = 0.01 as a bf16 noise floor and reports the push's KL over the floor's, and reports in closed form how straight the push stayed through the post-feed-forward norm. [backend/causal_check.py:53, 394-422; backend/mcp_scientist.py:1455-1459]
- The vault's voice is the only existing measurement of an edit's effect: one question, written greedily before and after in two separate forwards, with first-step KL, the target word's probability and rank, and where the two answers part. [backend/larql_vault.py:569-637]
- Rows of one forward are bit-identical while rows of different batch shapes were not, with KL up to 0.014 in the study. [backend/causal_check.py:6-7]
- On the home card a row's numbers depend on its batch's size and not on which rows share it, and a census batch written in two separate processes was identical in every field. [scripts/census_ties.py:23; experiments/results/census_determinism/summary.json]
- Pushes of 60 times a neuron's scale saturate the post-feed-forward norm at layers 20-50, so the causal check defaults to k = 10, and even a frozen-norm write disturbs layers 20-30 when large. [backend/causal_check.py:11-16]
- A uniform scale of a whole feed-forward block or of every head of a layer is undone by the norm that follows, so only a factor of 0 is accepted; scaling all of a layer's W_down or W_O by a positive constant changes nothing. [backend/atlas_ops.py:244-249, 261-264]
- A single neuron's write has a ceiling set by the post-feed-forward norm and the layer scalar, and the steer lens's dose of 0.8 of the stream is already 1 to 10 times that ceiling at most layers. [backend/atlas_ops.py:192-201]
- Push doses are calibrated on this model only at layers 39-51 for break_rel, 38 and 52 for soft_ceil, and 32-37 for match_ceil (counted from 0), and only for answers of 32 or 48 word-pieces (48 alone for the ceiling designs). [backend/census.py:268-277]
- The steer lens agrees with the census at layers 39-52 (0.60 to 0.91 matched against 0.01 to 0.06 shuffled), is weak at 32-38 (0.07 to 0.33) and is unchecked at 0-31 and 53-59. [atlas_data/index/steer_check.json (written 2026-11-14T17:40:55Z); backend/steer_check.py:9-14]
- Lens labels held up only at layers 40-57 and were at chance at the sampled layers 10-30; the gate was validated for base-fitted J++ labels of unedited random neurons and marks an edited neuron lens_validated false. [backend/label_gate.py:3-13, 72-73, 201]
- The vault's UPDATE and INSERT build the new column from a direction pulled back through the same J++ lens that later reads the neuron's label, so a lens label of an edited neuron cannot serve as independent evidence that the edit worked (my inference from the code). [backend/larql_vault.py:511-521]
- Weight-only wiring signs agreed with the measured change at 0.336 of links from rest and 0.418 as served, both below chance, while scored at each reader's own gate and up values the sign agreed at 0.841 and linked readers moved 5.5 times as much as unlinked ones. [ARCHITECTURE.md:772-800]
- The motif test's p is exact over every relabelling up to 10,000 and otherwise from 10,000 seeded draws, judges nothing under 4 sentences a side or 6 pairs, and its Benjamini-Hochberg correction covers only the neurons of one call (at most 512). [backend/mcp_scientist.py:419-425, 831-846, 883-899]
- The census ranks each unit against random-direction controls with an empirical p of (1 + controls at least as strong) / (C + 1) and a layer-wide Benjamini-Hochberg q, and about N / (C + 1) neurons beat every control by chance. [backend/census.py:610-712]
- Census controls are drawn as unit(w_post * r) or as a random mix of the layer's own write directions, because a plain isotropic direction is read about 100 times louder by the next layer than any neuron's write. [backend/census.py:25-34, 1567-1591]
- The `Weights` accessors read columns through the runtime with vault edits included, so `neuron_wiring`, `pair_products`, `write_dirs` and `write_ceiling` see an edit at once, while the indexes on disk are only flagged stale. [backend/atlas_ops.py:128-131, 167-172; backend/atlas.py:214-270]
- The vault keeps one overlay per neuron and ROLLBACK clears every overlay on every layer at once; there is no per-edit undo in that function. [backend/larql_vault.py:556-567]
- The arithmetic for a per-row column edit already exists in the CPU int4 kernel: a column overlay adds x[..., i] * (v - W[:, i]) to the layer's output. [backend/int4_cpu.py:136-163]
- A neuron that sits near one value on every word-piece can show a large steer KL while silencing or doubling it barely moves the next word-piece, so an add push along a column is not evidence about changing that column. [backend/scientist_methodology.md:74-76]

### Numbers
- Decode step with the compiled unpacker on the RTX 5090, by batch rows: 0.091 s at 1 row, 0.103 s at 32, 0.109 s at 64, 0.126 s at 96 [experiments/results/atlas_unpacker_steps_rtx5090.json (2026-10-29 01:48Z); COMPUTE.md:52-57]
- Decode step with the reference unpacker, by batch rows: 0.706 s at 1 row, 0.713 s at 32, 0.721 s at 64, 0.743 s at 96 [experiments/results/atlas_unpacker_steps_rtx5090.json; COMPUTE.md:52-57]
- Prefill of a census question with the compiled unpacker: 0.10 s at 1 row, 0.26 s at 32, 0.45 s at 64, 0.60 s at 96 [experiments/results/atlas_unpacker_steps_rtx5090.json (prefill_s)]
- Torch's peak memory above the model for a 64-piece census batch (compiled): 0.29 GiB at 1 row, 2.4 GiB at 32, 4.8 GiB at 64, 7.2 GiB at 96 [experiments/results/atlas_unpacker_steps_rtx5090.json (peak_gib)]
- Model and lens at rest on the card: 21.2 GiB of 32,607 MiB [backend/engine_runtime.py:470-471 (measured 2026-10-18); ARCHITECTURE.md:129]
- Three pushed answers of 80 word-pieces (/steer): 8.3 s with compiled, 50-54 s on reference [COMPUTE.md:61-64 (11 November 2026)]
- /steer with five prompts (15 rows) and with six candidates (13 rows): 9.2 s and 8.8 s with compiled [ARCHITECTURE.md:920 (11 November 2026)]
- One default heard vault edit (two 12-piece answers and the edit): 12.6 s; unpacker and date not stated [ARCHITECTURE.md:605-606]
- One steer-lens layer through forward_from_layer (21,504 neurons x 2 signs x 4 contexts): 287 s at layer 45 (14 layers above), 1,181 s at layer 5 (54 above); 20.5-21.9 s per layer-forward [experiments/results/atlas_steer_timing/summary.json (29 October 2026, compiled)]
- Whole steer lens and its dose pilot: 10.5 hours and 38 minutes [ARCHITECTURE.md:820-821 (13 November 2026)]
- Wiring check: 29-35 s a layer for 512 writers on 4 questions; 59 layers in about 31 minutes [COMPUTE.md:24 (night of 28-29 October 2026, compiled)]
- Corpus pass with every activation hooked, 2,000,054 word-pieces: 58 minutes on reference (stats), 20 minutes with compiled (passages) [COMPUTE.md:23 (28 October 2026)]
- Server recapture, one forward: 1.1 s at 512 positions, 1.3 s at 1,024, 1.9 s at 2,048 [backend/engine_runtime.py:490-491 (measured 2026-10-18)]
- Batch-shape KL between rows that should be alike: up to 0.014 [backend/causal_check.py:6-7]
- Writer's floor: word-pieces an unpushed answer shares with itself written alone: 22 in a census batch of 64 (snake question), 54 in a batch of 32 (free question); all 64 in the three-row steer shape and in write_voice [experiments/results/atlas_writer_floor_rtx5090.json (2026-10-29, layer 45, compiled)]
- Rows of a multi-prompt /steer that parted from their own batch of three before 80 word-pieces: 6 of 15 (five prompts) and 6 of 13 (six candidates) [ARCHITECTURE.md:920 (11 November 2026)]
- Sampled writing: samples of eight that equalled the write alone with seed + s: 3 of 8 [backend/atlas_ops.py:532-535]
- Another card class against the home card: same first word-piece in 0.975-0.987 of answers; log-probability difference median 0.0131-0.0146, 99th percentile 0.155-0.215 [ROADMAP.md:674-677]
- Layers where push doses are calibrated (counted from 0): break_rel 39-51; soft_ceil 38 and 52; match_ceil 32-37 [backend/census.py:268-272]
- Steer lens agreement with the census by layer: 0.60-0.91 at layers 39-52; 0.07-0.33 at 32-38; shuffled 0.01-0.06 [atlas_data/index/steer_check.json (2026-11-14)]
- Steer-lens word-pieces a layer shares, of twelve: 2.1 to 7.5 at layers 0-39; 1.03 or fewer from layer 40 up [atlas_data/index/steer_check.json (shared)]
- Label gate: AUC 0.84 as a graded score at layers 40-57; 0.64 as the yes/no cut (7 of 31 held back were sound, 160 of 209 kept) [backend/label_gate.py:42-65]
- Wiring sign agreement, linked / null: 0.336 / 0.341 from rest; 0.418 / 0.438 as served; 0.841 / 0.853 at the reader's own operating point; size rho 0.89 against 0.31 [ARCHITECTURE.md:772-800]
- Neurons above every control by chance: about 84 of 21,504 with 256 controls (N / (C + 1)) [backend/census.py:44-47]
- Motif test floors and draws: 4 sentences a side, 6 pairs, 10,000 relabellings, seed 0, 512 neurons a call, alpha 0.05 [backend/mcp_scientist.py:419-425]
- Blind pattern test, pairs picked rightly of 196: 177 with the own pattern and 86 with another's against any passage; 158 and 106 against near misses [ROADMAP.md:1343-1345 (14 November 2026)]
- Faces that need more of their own document than a short text holds: 30 of 176 need more than 60 word-pieces, 8 more than 256 (all at layer 26 or later) [backend/pattern_tests.py:53-55]
- Causal check of every label at page layers 41-59: about 2e20 multiply-adds, some 17 days on the 5090 (inferred) [COMPUTE.md:26]
- Corpus windows on disk: 3,914 windows of 512 word-pieces, 2,000,054 word-pieces [atlas_data/index/passages.d/meta.json and windows.npy]
- all_activations size: 2.6 MB a word-piece at float16; the /run route caps a text at 256 positions [backend/atlas_ops.py:767-769; backend/atlas.py:60]

### Reusable for a harness
- `atlas_ops.forward_from_layer` (backend/atlas_ops.py:1017-1043) reruns only layers L and above on a cached stream and returns states at every position, so it can price an edit at layer L at (60 - L)/60 of a forward.
- `atlas_build.build_steer_layer` (backend/atlas_build.py:775-836) is a working template for capture-once, rerun-above, many-hypotheses-per-forward with unpushed rows in the same forward.
- `atlas_ops.Interventions`, `resolve`, `merge` and `installed` (backend/atlas_ops.py:267-412) give per-row hooks; `mul` and `HeadScale` already emulate column-scaling and head-scaling weight edits beside an untouched row.
- `atlas_ops.next_logprobs` (backend/atlas_ops.py:665-690) and `causal_check.logits_fp32` and `kl` (backend/causal_check.py:132-139, 425-427) give the fp32 next-piece log-probabilities and KL the study used.
- `causal_check.run_check`, `face_effect`, `linear_regime`, `spearman` and `p_at` (backend/causal_check.py:226-480) give base-row-in-forward comparison, out-of-memory halving with a base row per forward, the k = 0.01 floor design, split-half agreement and a closed-form saturation guard.
- `atlas_ops.write_rows` with `_Race` (backend/atlas_ops.py:464-617) gives batched greedy or common-random-number sampled writing with per-row interventions and per-piece log-probabilities.
- `census.distinct4`, `first_departure`, the per-dose block of `compute_units`, and `writer_floor` (backend/census.py:558-561, 766-772, 672-680, 1434-1519) give text-health figures and the floor below which a departure is not evidence.
- `census.control_dirs`, `read_gain`, `control_refs`, `compute_units` and `chance_above_all` (backend/census.py:1567-1615, 610-718) give matched random-direction nulls, an empirical p and a false-discovery q.
- `mcp_scientist.motif_statistics`, `paired_statistics`, `either_statistics` and `benjamini_hochberg` (backend/mcp_scientist.py:596-846) are pure-numpy exact permutation and sign-flip tests with stated floors.
- `pattern_tests.binom_tail`, `written_result`, `blind_result` and `near_places` (backend/pattern_tests.py:87-196) give push-free tests of when a neuron fires, with near-miss controls and a kept-back fold.
- `atlas_ops.read_neuron`, `read_neurons`, `activation_peaks`, `activations_at`, `all_activations` and `pushed_reads` with `null_readers` (backend/atlas_ops.py:643-785, 1598-1733) give internal readings from one neuron to every neuron, and downstream readers against an unlinked null.
- `atlas_ops.Weights`, `neuron_wiring`, `pair_products`, `link_scale` and `HeadSpaces` (backend/atlas_ops.py:128-209, 1429-1639, 1736-1917) read wiring from the weights with vault edits included, so a before-and-after wiring diff needs only a driver.
- `mcp_scientist._study_max_abs` (backend/mcp_scientist.py:1175-1191) shows how to key an evaluator's cache on the tuple of patch ids in force.
- `larql_vault._heard` with `kl_nats` and `parted_at` (backend/larql_vault.py:167-180, 569-637) and `experiments/results/vault_example_edit/probe.py` are the existing edit-then-measure-then-rollback loop, on one question and on seven.
- `int4_cpu.with_overlay_terms` (backend/int4_cpu.py:136-163) holds the arithmetic a per-row column-edit hook would use.
- `label_gate.label_check` and `LABEL_GATE_CARD` (backend/label_gate.py:42-80, 164-203) are an existing machine-readable trust card that travels with replies, a model for carrying the other trust limits.
- `steer_check.measure` (backend/steer_check.py:95-158) is a CPU-only audit that sets one instrument against another per layer with a shuffled baseline and writes a standing (agrees, weak, unchecked).
- Seed data on disk: the study's 36 texts (backend/causal_check.py:64-101), the ten census questions with stored unpushed answers (backend/census.py:93-110), 3,914 corpus windows with an odd-window kept-back fold (atlas_data/index/passages.d/windows.npy), each neuron's usual range (atlas_data/index/act_scale.npz), whole-layer censuses with 2,048 controls (atlas_data/census/runs), the pattern pilot's 63 texts and 63 near misses (experiments/results/pattern_tests_pilot), and 1,218 study contexts (experiments/results/N1B_corpus.jsonl).

### Gaps
- No hook puts an arbitrary column or row replacement and the untouched model in one forward; `Interventions` has slots for adds, acts, muls and heads only (backend/atlas_ops.py:267-279).
- No evaluator computes a multi-token quantity under an edit (teacher-forced log-likelihood or per-position KL over a text); the pieces exist only inside the sweep and one verify script.
- No capability or regression suite exists anywhere in the repository; the only proxies are distinct 4-grams, the share of answers cut off and mean log-probability.
- No paraphrase sets exist on disk and no evaluator measures generalisation of an edit beyond the single voice question.
- No evaluator follows a series of edits in force together, and the vault's ROLLBACK clears every overlay at once.
- No noise floor exists for edits; the k = 0.01 floor of the causal check is for activation pushes.
- No ledger records tests across calls, so multiple comparisons are corrected only within one census run or one motif-test call.
- No function writes text from layer L upward with the lower layers' cached keys and values; `forward_from_layer` has no cache and `write_rows` runs the whole model.
- No evaluator is specific to edits other than one down_proj column; row overlays on gate_proj and up_proj are honoured by the forward but none is written today (backend/engine_runtime.py:365-366).
- Index-derived units (the stream's median length, each neuron's extremes, passages, the steer lens, the wiring index) are built on the unedited model and are only flagged stale after an edit, never rebuilt.
- No held-out split is enforced for edit prompts; the census halves and the passages' odd windows are the only held-out mechanisms and both serve neuron labels.
- `next_logprobs` and the activation readers take one text a call, so a sweep over a prompt set is a loop of forwards rather than one batch.

### Constraints
- An edited forward and an unedited forward must have the same batch shape, or both conditions must be rows of one forward, because rows of different shapes differ by up to 0.014 KL and unpushed answers in a batch of 64 can part from themselves after 22 word-pieces.
- An edit's effect must be read from the model's own odds in fp32, never from the lens label of the edited neuron, because the label gate was not measured on edited neurons and the vault's UPDATE is built through the same lens.
- Lens-derived words may be shown only at layers 40-57 under the label gate, and any evaluator that uses them must carry the gate's card.
- Push-based evidence may be treated as calibrated only at layers 32-52 for the named designs and answer lengths, and steer-lens evidence only at layers 39-52.
- A harness must not treat a uniform rescale of a layer's W_down or W_O as an edit, because the following norm undoes it; and it must report how far any write is squashed by the post-feed-forward norm.
- Weight-only wiring may be used for the size of a link but not for its sign at an ordinary word-piece, where the measured agreement was below chance.
- Any ranking against controls must state that about N / (C + 1) units beat every control by chance, and any p must state what it was corrected over.
- Evaluators that divide by a neuron's usual extreme or use the stream's median length are in pre-edit units and must say so after an edit.
- The frozen-prefix shortcut of the causal check may not be used for weight edits, because an edit acts at every position and not only at the last.
- The KL direction and the logit precision must be fixed and named per evaluator, because the existing ones disagree.
- The GPU is held under one lock per route or tool turn, so evaluators run one at a time and should batch hypotheses into a forward rather than loop; a step costs about the same from 1 to 64 rows.
- Sampled writing is not the instrument's greedy reading; greedy is the census's and the method's measure, and a presence penalty at temperature 0 is refused.
- The corpus windows hold third-party text that must stay in its folder and never be committed.
- A driver must read `selectivity` and not the legacy `verdict` of the motif test, which calls one sentence a side confirmed.

### Surprises
- The three KLs the instrument reports do not share a direction: /logprobs is KL(base || intervened), the steer lens and causal check are KL(pushed || base), and the vault's voice is KL(before || after).
- ARCHITECTURE.md:180-181 gives the attention / feed-forward split without the layer scalar, while the code computes s*(mid - pre), post - s*mid and a separate rescale term (backend/theodolite_math.py:180-188); the document shows the pre-fix formula.
- ARCHITECTURE.md:868-869 and 879-880 say the near-miss blind test and the behind-its-own-passage test have not run on the 31B, while ROADMAP.md:1291-1345 records both run on 14 November 2026.
- The vault's voice computes its KL from the writing loop's logits cast to float32, not from the fp32 GEMM the causal check insists on, although ARCHITECTURE.md:623-624 calls them float32 distributions.
- `mul` pushes and head scales are already exact weight edits in disguise, and the LoRA exporter writes them out as rank-one and rank-head_dim weight changes, so part of a weight-space harness's same-forward comparison exists today.
- The only stored before-and-after record of a vault edit is one edit probed on seven questions (experiments/results/vault_example_edit/probe.json); the study itself tested only DELETE by cause and effect.
- The most active neurons at layers 41, 46, 51, 56 and 59 were not heard on the example question at strengths 3 or 30; an audible one-column edit was rare enough that the example had to be found at the last layer (ROADMAP.md:181-186).
- A step costs nearly the same from 1 to 64 rows because every forward unpacks all 29.3 billion 4-bit weights, so batching hypotheses is close to free up to memory.
- The legacy motif verdict is still sent and reads one sentence a side as confirmed with d near 10,000.
- Two blind-tested patterns that passed 9 of 9 and 12 of 12 against random passages fell to chance against near misses, so a passed test against an easy control is not evidence (ROADMAP.md:1355-1357).
- The saved-prompts shelf holds no questions today (one system prompt), so it cannot seed a prompt set.

### Not established
- The wall-clock of one next_logprobs, read_neuron, activation_peaks, neuron_wiring or causal_spot_check call on the 31B is not recorded in anything I read; about 0.1 s for a short single-row forward with compiled is my inference from the prefill figures.
- The peak memory of the causal check's 180-row study forward is not recorded in anything I read.
- Whether forward_from_layer has been compared with a whole forward on the 31B itself is not established; the check I found runs on the CPU stand-in at atol 1e-5 (scripts/check_atlas_index.py:376-378).
- Whether a mul 0 push with spare_bos false and a vault DELETE give bit-identical rows on the GPU is not established; they are the same column change by the documents.
- The dtype of the logits the voice's writing loop yields is not established, because interp_engine was not read.
- Which unpacker and date the 12.6 s heard edit refers to is not stated where it is recorded.
- Whether any evaluator has ever been run with several edits in force together is not established; I found no record of one.
- Whether two same-shape forwards through forward_from_layer are bit-identical on the card is inferred from the census determinism run, not measured for that function.


## gguf_export — Reading and writing whole models (GGUF, W4A16, CPU kernel), and how Gemma-specific the stack is
Full report: experiments/results/weight_space_exploration/code/gguf_export.md

### Summary
The stack reads a Gemma 4 GGUF through a copy-on-write memory map into transformers' Gemma4ForCausalLM, holding each of the 410 linears packed (the only holding that fits the 32 GB card), resident, dense, or, on the CPU only, through torch's int4 kernel; packed, resident, dense and the compiled unpacker are bit-equal on one device and dtype, and on the 5090 the 31B writes a word-piece in 0.12-0.13 s from a GGUF (0.091 s from the W4A16 checkpoint) with the compiled unpacker. An edit today is an in-memory bf16 column or row overlay on a 4-bit base, applied and rolled back by a dict write, honoured identically by every holding and both formats; nothing persists it in a form the instrument can reload (the vault is memory only, the LoRA export is one way to llama.cpp, and write_gemma4_gguf is a test-fixture writer that cannot re-emit a packed model). Requantising an edit into the 4-bit file is lossy by layout: a down_proj column shares each of its 5,376 blocks with 31 other neurons, so the exact form of an edited model is base identity plus a higher-precision delta, and a patched GGUF copy has an identity hazard because the fingerprint does not hash the middle of the file. The existing bit-equality checks give a ready pattern for verifying an edit (heard, seen, base untouched, rollback by bits, the null edit), while the CPU int4mm path is a second arithmetic at about 0.45-0.56 s a word-piece and 21-27 GB that agrees with the reference only to about 0.25 nats on top log-probabilities. The per-linear layer of the stack (gguf_weights.py) is general, but the runtime is a pinned, Gemma-4-31B-shaped class with hard-coded constants and no model-family-neutral protocol, so another family needs a new builder, runtime interface and role map; for rehearsal there is a random six-layer stand-in that every check runs on, and a trained dense Gemma 4 12B QAT Q4_0 GGUF on disk whose header passes the builder's refusals but which has never been built or run.

### Key facts
- A GGUF is opened as a copy-on-write memory map (nothing is ever written back) and built into transformers' Gemma4ForCausalLM without allocating random weights, with the seven linears of every block replaced one by one and embeddings, norms and layer scalars unpacked to the model's dtype as plain parameters. [backend/gguf_model.py:118-155, 485-647]
- Fifteen GGUF tensor types are unpacked in torch bit-equal to gguf-py 0.19 (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K, IQ4_NL, IQ4_XS), eleven more are unpacked once at load by gguf-py into dense layers, and eight are refused in plain words. [backend/gguf_weights.py:48-75]
- GGUF blocks run along the input dimension: 32 consecutive inputs of one output row (256 for the K types) share an fp16 scale, and the W4A16 checkpoint has the same grouping with one bf16 scale per 32 inputs. [backend/gguf_weights.py:4-7; backend/engine_runtime.py:296-309]
- A GGUF's quantised linears are held packed (default, unpacked at every forward), resident, dense or int4mm; the first three give the same forward bit for bit and share one model id, while int4mm is CPU only, is not the reference's bits and adds +int4mm to the model id. [backend/gguf_model.py:79-80, 562-596; backend/gguf_runtime.py:106-110]
- A fully resident or dense 31B needs about 62 GB in bf16 (59 GB for the linears alone), so on the 32 GB card the model can only be held packed, optionally with a resident budget for some layers. [ARCHITECTURE.md:1288, 1362; backend/int4_cpu.py:11-12]
- The compiled unpacker runs the reference's own arithmetic through torch.compile with the model left eager, and at install it compares every packed layer's whole weight by bits and strides and every one of the 65,536 patterns of each fp16 field through each graph, refusing on any difference. [backend/gguf_fast.py:270-386]
- An in-memory edit is a column or row overlay (a dict from index to vector) laid on a clone of the unpacked weight at every forward and cast to the weight's dtype; the stored 4-bit blocks are never touched, so rollback is clearing the dict. [backend/gguf_weights.py:375-390, 530-532; backend/engine_runtime.py:356-375]
- The overlay dicts exist on all seven linears of every block in every holding and in both formats, so the forward would honour an overlay on an attention projection or on up_proj, although the vault itself writes only down_proj columns. [backend/gguf_weights.py:393-405, 456-457; backend/engine_runtime.py:431-436; backend/larql_vault.py:475-478]
- The runtime's weight readers lay overlays on down_proj columns and gate_proj rows only; slice_w_up, get_full_w_up and the attention weight reader return the layer's own weight, so an edit to up_proj or attention would be heard by the forward but not seen by the readers. [backend/gguf_runtime.py:166-224; backend/atlas_ops.py:203-209]
- write_gemma4_gguf is a fixture writer for the checks: it reads a .weight attribute on every linear (which a packed GGUFLinear does not have), it is declared not to be llama.cpp's converter, and its Q4_K is a plain per-sub-block quantiser. [backend/gguf_model.py:54-58, 865-999]
- Nothing in the repository persists an edit in a form the instrument can reload: the vault keeps overlays and records in memory only, and the LoRA export writes an adapter for llama.cpp whose read_lora returns arrays that no code applies to the runtime. [backend/larql_vault.py:464-567; scripts/export_lora.py:501-552]
- Two mechanics for a patched GGUF copy already exist in write_silenced: a full rewrite through GGUFWriter that copies every key and every tensor's bytes verbatim and adds a tensor, and a reflink or byte copy followed by an in-place overwrite of one tensor at its data offset. [scripts/export_lora.py:555-654]
- The GGUF fingerprint hashes only the file's size, its header, the first MiB of tensor data and the last MiB, so a same-size copy patched in the middle of the data keeps its base's fingerprint, and the index check compares only fingerprint and header hash. [backend/gguf_model.py:741-764; backend/atlas_index.py:188-190]
- The model id is set once at load and does not change when an overlay is in force, so a reading taken under an edit carries the base's model id unless the caller records the edit separately, as the atlas does for its cache key. [backend/gguf_runtime.py:106-110; backend/atlas.py:216-219]
- A GGUF may hold a different type per tensor and the loader makes any F32, F16 or BF16 tensor a plain dense layer in every mode, so an edited tensor could be stored exactly as BF16 in an otherwise unchanged copy (derived from the loader; not tested for this purpose). [backend/gguf_model.py:24, 589-595]
- Derived from the layout: a down_proj column crosses 5,376 blocks each shared with 31 other neurons, so requantising it either snaps each entry to one of 16 levels under the old scale or re-rounds up to 31 x 5,376 neighbouring weights, whereas a gate_proj or up_proj row is 168 whole blocks that belong to one neuron alone. [backend/gguf_weights.py:183-188 (layout; the consequence is the reader's derivation)]
- Bit equality holds across holdings and unpackers on one device and dtype but not across devices: the CPU and GPU forwards of the same file agree on the five likeliest pieces in order with a largest log-probability difference of 0.25 among them. [ARCHITECTURE.md:1508-1511]
- On the Ryzen 7900X's AVX-512 path the batch size alone moves a log-probability by up to 0.0039 on the stand-in, so the document says a CPU's floor must be measured on it as a card's is. [ARCHITECTURE.md:1444-1445]
- The checks already contain a four-part verification of an edit on a packed layer: the logits equal a dense twin with the column written in, the accessors return the edit, the blocks and the weight they unpack to are untouched, and with the edit cleared the logits are the clean ones bit for bit. [scripts/check_gguf.py:825-874]
- A null edit (a column and a row set to what they already are) adds exactly nought through the int4 kernel; before that was arranged, such an edit changed 13 of 30 greedy answers on the stand-in because the edited layer changed numeric path. [scripts/check_gguf_cpu.py:611-617; ARCHITECTURE.md:1391-1396]
- The int4 kernel path lays an edit on as float32 difference terms added to the kernel's product, while its accessors still read the file's own numbers, so it is a second arithmetic that shares the loader and the overlay record with the GPU path. [backend/int4_cpu.py:138-164, 229-253]
- The lmstudio QAT Q4_0 GGUF is not the W4A16 checkpoint: its 421 F32 norms and layer scalars equal the checkpoint's but all 410 Q4_0 linears differ, so readings from it do not stand for the checkpoint's. [ARCHITECTURE.md:1249-1253; experiments/results/gguf_31b_gpu/README.md]
- GGUFRuntime subclasses the pinned TheodoliteRuntime and replaces only load() and the seven accessors that read the packing, so it inherits every 31B-shaped constant of the pinned parent. [backend/gguf_runtime.py:1-33, 60-253]
- runtime_factory.make_runtime is a single choice point driven by environment variables that returns TheodoliteRuntime or one of its two subclasses; there is no model-family-neutral runtime protocol. [backend/runtime_factory.py:113-133, 154-212]
- atlas_ops.Weights is the nearest thing to a Weights class: it takes its layer count and widths from the model rather than from constants, but it addresses the model by Gemma 4 module names and is centred on the feed-forward block. [backend/atlas_ops.py:128-209]
- backend/gguf_weights.py is general: its unpackers, GGUFLinear, the overlays and the accessors dense, dense_cols and dense_rows read any linear alike whether it is plain, W4A16 or GGUF, and nothing in it names Gemma. [backend/gguf_weights.py:416-583]
- The 31B's shape is hard-coded as constants (60 layers, 5,376, 21,504, 262,144) used in the pinned runtime and also in the unpinned vault, server and scientist tools; the vault's ROLLBACK loops 60 layers and the checks patch that constant to run on the six-layer stand-in. [backend/theodolite_math.py:24-33; backend/larql_vault.py:481-559; scripts/check_gguf_fast.py:583-588]
- The builder refuses any arch other than gemma4, Gemma 4's mixture-of-experts model (26B-A4B), per-layer embeddings (E2B, E4B) and shared key-value layers, and it sets GeGLU's activation by hand because the file does not state it. [backend/gguf_model.py:214-225, 302]
- The stand-in is a real Gemma4ForCausalLM with random weights (6 layers, d_model 64, FFN 128, about 600 pieces, 41 linears) that runs on the CPU, writes to GGUF in six types and passes every GGUF check, so it rehearses the plumbing of a harness but not hypotheses about what weights compute. [scripts/atlas_stand_in.py:44-48, 102-142; scripts/check_gguf.py:420-445]
- A trained dense Gemma 4 12B QAT Q4_0 GGUF (6.98 GB, 48 layers, d_model 3,840, FFN 15,360) is on the machine and its header states every key the builder needs with no experts, per-layer embeddings or shared key-value layers, but it has never been built or run and no document mentions it. [~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf (header read with gguf-py only), against backend/gguf_model.py:192-225]

### Numbers
- Linear layers and quantised weights of the 31B: 410 linear layers holding 29,286,727,680 weights [scripts/check_gguf.py:636-639]
- GGUF load time on the RTX 5090, packed, bf16: 20.5 to 22.3 s for three Q4 files (2 November 2026) [experiments/results/gguf_31b_gpu/README.md; ARCHITECTURE.md:1479-1481]
- Memory on the card after loading a 31B GGUF: 18.93 GiB (Q4_K_M), 19.18 GiB (UD-Q4_K_XL), 21.2 GiB (QAT Q4_0 with the J++ lens) [ARCHITECTURE.md:1487]
- A written word-piece from a GGUF on the 5090, reference against compiled: 0.542 / 0.121 s (QAT Q4_0), 0.819 / 0.126 s (Q4_K_M), 0.758 / 0.121 s (UD-Q4_K_XL) [ARCHITECTURE.md:1488]
- A step at a batch of 32 from a GGUF on the 5090, reference against compiled: 0.553 / 0.131 s, 0.736 / 0.136 s, 0.737 / 0.132 s [ARCHITECTURE.md:1489]
- Install time of the compiled unpacker on the 5090, tests included: 2.8 s (8 graphs), 8.7 s (10 graphs), 10.4 s (19 graphs) [ARCHITECTURE.md:1490]
- Peak memory on the card, reference against compiled: 22.71 / 23.05 GiB (QAT Q4_0 with lens), 20.57 / 20.33 GiB, 20.89 / 20.57 GiB [ARCHITECTURE.md:1491]
- One six-piece forward of the QAT Q4_0 on the 5090: 0.555 s on the reference, 0.14 s compiled [experiments/results/gguf_31b_gpu/qat_q4_0.json]
- A whole sighting (60 layers through the lens, 6 written) from the QAT Q4_0 on the 5090: 5.22 s on the reference, 1.92 s compiled [ARCHITECTURE.md:1503-1504]
- Optics check on the QAT Q4_0 GGUF on the card: passed in full on the reference unpacker with a peak of 24.15 GiB [ARCHITECTURE.md:1506-1508]
- W4A16 checkpoint step on the 5090 at one row (28 October 2026): 0.706 s reference, 0.275 s lut, 0.091 s compiled [COMPUTE.md:55]
- W4A16 checkpoint step on the 5090 at 96 rows: 0.743 s reference, 0.303 s lut, 0.126 s compiled, with 7.2-7.5 GiB above the model [COMPUTE.md:58]
- Unpacking one 21,504 x 5,376 layer on the 5090: 0.24-0.46 ms compiled against 1.3-3.6 ms on the reference, where the one-row bf16 product takes 0.147 ms [ARCHITECTURE.md:1317-1320]
- Memory of the 31B fully unpacked in bf16: about 62 GB whole, 59 GB for the linears, against 18 GB at the int4 kernel's 0.625 bytes a weight [ARCHITECTURE.md:1288, 1362; backend/int4_cpu.py:11-12]
- One packed reference forward of the 31B on the 12-core CPU: 37.2 s (QAT Q4_0) and 38.8 s (UD-Q4_K_XL, unhindered; 116 s and 185 s beside other work) [experiments/results/gguf_31b_gpu/qat_q4_0_cpu.json; experiments/results/gguf_31b_cpu/ud_packed.json]
- The compiled unpacker on the 12-core CPU: the same forward in 11.9 s after a 64.3 s install, the reference's bits [experiments/results/gguf_31b_gpu/qat_q4_0_cpu.json]
- int4mm on the 12-core CPU from the QAT Q4_0: load 46 s, a 59-piece prefill 3.5 s, a written word-piece 0.45 s, about 21 GB [ARCHITECTURE.md:1460-1461, 1363]
- int4mm on the 12-core CPU from the Q4_K_M with 8.5 GB resident: load 48 s, prefill 3.5 s, a written word-piece 0.56 s, 26.5 GiB of its own beside 17.6 GiB of mapped file pages [ARCHITECTURE.md:1455-1459]
- Agreement of int4mm with the CPU reference on the 31B (Q4_K_M): the same five likeliest pieces in order, KL 0.001, largest log-probability difference 0.23 among them [ARCHITECTURE.md:1462-1465]
- Agreement of the CPU reference with the GPU reference on the same file: the same five likeliest pieces in order, largest log-probability difference 0.25, KL 0.0015 [ARCHITECTURE.md:1508-1511]
- Batch-size floor on the 7900X's AVX-512 bf16 path (stand-in): every answer the same but a log-probability moves by up to 0.0039 [ARCHITECTURE.md:1444-1445]
- int4mm canary on the stand-in: 30 of 64 answers identical to the reference's, where two batches of the reference matched 64 of 64 [ARCHITECTURE.md:1446; COMPUTE.md:266-267]
- The QAT Q4_0 GGUF against the W4A16 checkpoint: 421 F32 tensors equal; all 410 Q4_0 linears differ, with 11-15% of values and 68-71% of codes equal [ARCHITECTURE.md:1249-1253]
- Run-to-run movement of the reference step on the card: 6 to 19% between two runs of the same file, cause not established [ARCHITECTURE.md:1493-1495]
- Size of one 31B down_proj tensor: 115.6 million weights: about 65 MB as Q4_0 and 231 MB as bf16 (the reader's arithmetic from 5,376 x 21,504) [ARCHITECTURE.md:1373 for the weight count; derived sizes]
- The stand-in's shape: 6 layers, d_model 64, FFN 128, about 600 pieces, 41 linear layers [scripts/atlas_stand_in.py:44-48; scripts/check_gguf.py:442]
- The 12B QAT Q4_0 GGUF on disk: 6,975,878,560 bytes; 48 layers; d_model 3,840; FFN 15,360; 667 tensors (328 Q4_0, 338 F32, 1 Q6_K); about 10.9 billion linear weights, 37% of the 31B's (the reader's arithmetic) [header of ~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf, read with gguf-py on 14 November 2026]
- A census of one layer on the CPU through int4mm: some 50 days a layer at 64 new pieces (inferred in the document, not run) [ARCHITECTURE.md:1471-1474]

### Reusable for a harness
- gguf_weights.dense, dense_cols and dense_rows (backend/gguf_weights.py:542-576) read any linear's own weight, whole or by column or row, the same way for plain, W4A16 and GGUF layers, which is the read primitive a weight-space harness needs.
- gguf_weights.with_overlays and install_overlay_forward (backend/gguf_weights.py:375-405) together with the _memit_col_overlays and _memit_row_overlays dicts are an existing apply-and-rollback primitive for column and row edits on any of the seven linears of a block.
- GGUFLinear.unpacked_cols and unpacked_rows (backend/gguf_weights.py:506-528) unpack only the blocks that hold the asked columns or rows, so reading one neuron's weights costs one block per row rather than a whole layer.
- w4a16_fast.same_bits and same_layout (backend/w4a16_fast.py:88-102) and gguf_fast.table_agrees (backend/gguf_fast.py:117-129) are bit-pattern comparators that treat NaN and signed zero correctly and report how many numbers differ.
- The install-time self-test of gguf_fast.install (backend/gguf_fast.py:270-386) is a pattern for 'test by bits, then bind, else refuse in plain words' that a harness could copy for any new fast path or edit applier.
- The four-part edit check in scripts/check_gguf.py:825-874 (heard, seen, base untouched, rollback by bits) can be lifted as the harness's own routine for verifying that an edit was applied exactly as recorded.
- The readings() bundle in scripts/check_gguf_fast.py:565-595 (logits, pushed answers, a vault UPDATE, weight slices, ROLLBACK, a small census) is a ready conformance suite for comparing two ways of holding or evaluating the same edited model byte for byte.
- model_tensors_equal and stand_in_with in scripts/check_gguf.py:195-239 check that every tensor a built model holds is the file's dequantised tensor and build a dense twin carrying a file's numbers, which is how a reloaded edited model could be checked against its record.
- compare_gguf_ct.compare_tensor and its Checkpoint reader (scripts/compare_gguf_ct.py:111-226) stream two weight files one tensor pair at a time and report share equal and largest difference, and its helpers ct_codes, ct_pack, q4_0_codes_scales and q4_0_blocks convert between 4-bit codes, scales and both packings.
- gguf_model.provenance, open_gguf, read_metadata and tensor_table (backend/gguf_model.py:118-181, 741-786) and runtime_factory.weights_card (backend/runtime_factory.py:251-272) give the base model's identity record that an edit record should name.
- int4_cpu.compare_rows, kernel_departure and enable (backend/int4_cpu.py:119-135, 305-363) compare two sets of greedy answers on shared prefix, split step, log-probability difference and top-5 overlap, measure a kernel's weight against the reference, and switch the kernel off in place for a reference forward on the same loaded model.
- export_lora.write_lora, read_lora, write_silenced and _clone_or_copy (scripts/export_lora.py:501-654) already write a rank-one column edit as a llama.cpp LoRA and show both ways of producing a patched GGUF copy, giving llama.cpp as a third, independent evaluator.
- gguf_model.write_gemma4_gguf with its blocks argument and quantize_rows (backend/gguf_model.py:837-999) write stand-in models to GGUF in chosen types and with chosen raw blocks, which is enough to build rehearsal fixtures for persist-and-reload tests.
- scripts/atlas_stand_in.py (stand_in, stand_in_hf, stand_in_runtime) gives a six-layer random Gemma 4 with its own tokenizer and runtime on the CPU, on which every check in this area already runs.
- atlas_ops.Weights (backend/atlas_ops.py:128-209) is a small weights facade that takes its shape from the model and already returns edited columns and rows through the runtime's accessors.
- runtime_factory.make_runtime and reading_class (backend/runtime_factory.py:102-110, 154-212) choose the runtime and tag a reading's numeric class, and GGUFRuntime can also be constructed directly with a path, device, dtype, linears mode and load_jpp=False.
- The lab's GGUF gradient twin (_GGUFLinearFn in experiments/lab/labkit.py:435, described at ARCHITECTURE.md:1264-1268) runs gradient passes through packed layers without keeping any weight, should a harness want gradients as evidence beside gradient-free edits.
- interp_engine's programmatic ArchSpec (.venv/lib/python3.12/site-packages/interp_engine/arch.py:1-13, 1126) already resolves layers, final norm and lm_head for most decoder-only models, so the lowest layer under the instrument is not Gemma-specific.

### Gaps
- There is no on-disk format for an edit that the instrument can reload: a delta file naming the base model's identity and the overlay vectors, with a loader that sets the overlays again, would have to be built.
- There is no loader that applies a LoRA adapter GGUF to the instrument's own runtime; the export is one way.
- There is no requantiser for an edited tensor, and in particular none that re-encodes only the touched blocks and copies the rest byte for byte, so a fully 4-bit edited model cannot be produced today.
- There is no API for a quantised-domain edit (changing 4-bit codes or block scales in weight_blocks or weight_packed directly), and nothing tests one.
- There is no edit primitive other than whole-column and whole-row replacement: a low-rank or dense delta over many columns would be a Python loop over overlays at every forward.
- There is no overlay or rollback mechanism for embeddings, norms or layer scalars, which are plain parameters, and the lm_head is tied to the embedding.
- The weight readers do not show overlays on up_proj or on attention projections, so read-back verification of such edits would have to be added.
- No reading's identity includes the edits in force: the model id is unchanged by an overlay, so the harness must add an edit identity to every result it stores.
- There is no whole-tensor or whole-file hash: the GGUF fingerprint skips the middle of the file, so a harness that persists patched copies needs its own hash of the edited tensors or must change the header.
- There is no routine that compares a persisted edited file against its base plus the recorded delta, although the streaming comparator's pieces exist.
- There is no model-family-neutral runtime protocol and no builder for any arch but gemma4; both would have to be written before another family could be the object.
- The 31B's constants in the vault, the server and the scientist's tools would have to be replaced by values read from the model before a smaller Gemma 4 could use those paths.
- The 12B GGUF has not been tried, so a trained small rehearsal object is not yet available in practice.
- The floors between evaluators on the real model have not been measured: the int4mm canary on the 31B has not been run, and GPU logits across batch sizes were not compared by bits.

### Constraints
- The files backend/engine_runtime.py, backend/theodolite_math.py, backend/label_gate.py and backend/__init__.py are pinned by sha256 for the R4 study, and the check scripts assert the pins, so new behaviour must go in new modules or subclasses as the GGUF work did.
- On the 32 GB card the 31B can only be held packed (about 19-21 GiB after loading, peak about 20-23 GiB for single forwards), because a resident or dense copy needs about 62 GB.
- Bit-equality is a property of one device, one dtype and one unpacker-equivalence class: results on the CPU, in float32 or through int4mm are different numeric classes with different model ids, and a census will not continue across them.
- The compiled unpacker serves only whole-weight calls whose shapes were tested at install and only under the guards it was tested with; a call under torch.inference_mode falls back to the slow reference and is merely counted.
- An in-memory edit is a bf16 vector over a 4-bit base, so persisting it inside a 4-bit tensor is lossy, and for down_proj columns it necessarily disturbs or is constrained by the 31 neighbouring neurons of each block.
- A GGUF of the 31B is a different quantisation from the W4A16 checkpoint in every linear, so a GGUF-based evaluator tests an edit on a different object unless the same GGUF is the base on both sides.
- The int4 kernel is CPU only, takes only Q4_0, Q4_1 and Q4_K layers whose output count is a multiple of 16 and input count a multiple of 32, has no backward pass, and its readings are not the reference's bits.
- A patched GGUF copy must change its header or carry its own tensor hashes, because a same-size change in the middle of the data leaves the fingerprint and therefore the index check unchanged.
- write_gemma4_gguf needs plain .weight layers and uses quantisers that are not llama.cpp's, so it is fit for stand-in fixtures and not for emitting an edited 31B.
- The GGUF builder accepts only dense Gemma 4 text models: any other arch, experts, per-layer embeddings and shared key-value layers are refused.
- Sightings, the lens, and the vault's UPDATE and INSERT need the J++ lens fitted for the 31B and the pinned 60-layer sweep, so on any other model only forwards, weight reads, overlays, pushes and the census-style paths can be expected to work.
- The vault clamps layer and neuron indices to the 31B's constants instead of refusing, so a harness must validate indices against the loaded model itself.

### Surprises
- The GGUF fingerprint that every index and census run relies on does not hash the middle of the file, so a patched copy of the same size can carry its base's fingerprint and differ from it only by file name.
- write_gemma4_gguf, the only whole-model writer, cannot write a model that was loaded packed and is explicitly not llama.cpp's converter, so 'export the edited model as a GGUF' does not exist for the 31B.
- The forward honours overlays on all seven linears of a block, but the weight readers show them only for down_proj and gate_proj, so an edit to up_proj or attention would change behaviour without being visible on read-back.
- The lmstudio QAT Q4_0 GGUF and the W4A16 checkpoint are different quantisations in all 410 linears, so 'the Gemma model' is two different objects depending on the path, and only the checkpoint's own int4mm mode holds the checkpoint's weights exactly on the CPU.
- The LoRA export, which already writes a vault column edit as a rank-one adapter, has a reader but no loader into the instrument.
- The compiled unpacker silently serves the slow reference for a call under torch.inference_mode, which a new harness could trigger without noticing except through the fell_back counter.
- A trained dense Gemma 4 12B GGUF is on the machine and passes the builder's refusals by its header, yet ARCHITECTURE section 5 names only E2B, E4B and 26B-A4B as other Gemma 4 models and records no attempt on a 12B.
- The vault's ROLLBACK loops a hard-coded 60 layers and its index clamps use the 31B's constants, so the checks monkeypatch the constant to run on the stand-in.
- ARCHITECTURE.md:1351 speaks of transformers 5.21 while the virtual environment holds transformers 5.20.0, a small disagreement between document and environment.
- A resident GGUF layer's unpacked() returns the live weight with only a comment saying not to write into it, whereas the W4A16 resident mode detects an in-place write at the next forward.
- The document's own timings show unpacking still costs more than the matrix product even when compiled (0.24-0.46 ms against 0.147 ms a layer), so the per-forward floor on the card is set by the 4-bit packing, not by the model's arithmetic.

### Not established
- Whether build_gemma4 actually builds the 12B QAT Q4_0 GGUF and gives a correct forward, and what it costs in seconds and GiB on the card, was not established because nothing was run.
- Whether gguf-py's quantisers reproduce an existing block from its own dequantised values (a requantisation round trip) was not established; no test in the repository covers it.
- Whether a LoRA written by export_lora.py reloads to the exact bits of the original overlay was not established.
- Whether logits are bit-equal across batch sizes on the GPU was not established; the record compares written pieces alone against pieces at a batch of 32.
- The speed and memory of the W4A16 checkpoint through int4mm on the real 31B were not found in the sections read.
- The load time of the W4A16 checkpoint on the card is not stated in the files of this area.
- Whether a census or atlas record names the vault edits in force was not established beyond a grep that finds no such reference in backend/census.py.
- Whether weight_blocks can safely be rewritten in place as a quantised-domain edit was not established; nothing in the repository does it.
- The cause of the 6 to 19% run-to-run movement of the reference unpacker's step on the card is recorded as not established.
- A census, an atlas build or the server's pages on a GGUF with the compiled unpacker on the card are listed by the document as not measured.
- Which GGUF types gguf-py 0.19 can quantise (as opposed to dequantise) was not checked.


## runtime_weights — The runtime and how the weights are held
Full report: experiments/results/weight_space_exploration/code/runtime_weights.md

### Summary
Gemma-4-31B is held as 410 packed 4-bit linears (every attention q/k/v/o and feed-forward gate/up/down of the text model: int32 words of eight pieces, one bf16 scale per output row per 32 input columns, weight = (piece - 8) x scale) plus plain bf16 tensors for the tied embedding/unembedding, the norms and one layer scalar per layer; it rests at 18.0 GiB on the card, 21.2 GiB with the J++ lens. No dense copy of the model ever exists: each linear rebuilds its dense bf16 weight inside its own forward on every forward and decode step, through one of four unpackers that give identical bits (reference 0.71 s a step, lut 0.28 s, compiled 0.09-0.13 s and the server's default, resident never run because it needs 58.6 GB). The only weight overlay is a per-module dict of whole-column and whole-row replacements laid on a clone of the freshly unpacked weight, global to the batch, of which only down_proj columns are ever written; there is no low-rank, sparse or block overlay, no per-row variant, no per-linear dense cache, and nothing for embeddings, norms or layer scalars. Hooks are rich (interp_engine resolves some 25 named points and any module path for read and write) and a forward can start at a given layer but not stop at one. Autograd is off everywhere by default, yet backward passes have run on this model on this card through a recompute-in-backward autograd function (about 0.66 s forward and 0.31 s backward for a 128-piece prompt on the reference unpacker), giving gradients for activations only. On the GPU a row's bits depend on its batch shape, not on the unpacker, and the code's honesty devices are built around that. One process owns the card, with no arbitration at home beyond a shared port. The convention that new code must live outside engine_runtime.py rests on R4's sha256 pin, which I found no longer holds: the file was edited on 12 November 2026 while the documents still say it is pinned.

### Key facts
- All 410 quantised linears are the text model's attention q, k, v, o and feed-forward gate, up, down projections; each is found by its weight_packed and weight_scale attributes and given two empty overlay dicts and an instance-level forward at load. [backend/engine_runtime.py:431]
- The packed layout is int32 weight_packed [out, in/8] with eight 4-bit pieces a word (piece j of word k is input column 8k+j) and bf16 weight_scale [out, in/32], one scale per output row per group of 32 consecutive input columns; the dense weight is (piece - 8) x scale, symmetric with no zero point, 4.5 bits a weight. [backend/engine_runtime.py:296]
- The linears have eight dense shapes: gate and up [21504, 5376], down [5376, 21504], q [8192, 5376] on sliding layers and [16384, 5376] on full layers, k [4096, 5376] and [2048, 5376], o [5376, 8192] and [5376, 16384], and v [4096, 5376] on the 50 sliding layers only, because the ten full-attention layers have no v_proj (attention_k_eq_v); that is 410 = 60 x 6 + 50 linears and 29.29 billion weights. [backend/w4a16_fast.py:184]
- Not quantised and held as plain bf16: the embedding [262144, 5376] (2.625 GiB, multiplied by sqrt(5376) after look-up) which is tied to the unembedding so an in-place edit of one row changes both, four RMSNorm weights [5376] per layer, q_norm and k_norm weights [256 or 512], the final norm, and one layer_scalar buffer [1] per layer. [backend/engine_runtime.py:857]
- The layer scalar multiplies the whole residual stream in place as the layer's last act, so post = s x (mid + post_ffn_norm(mlp(pre_ffn_norm(mid)))) and the hook point resid_post is read after the scalar. [backend/atlas_ops.py:8]
- No dense copy of the model persists: each packed linear's forward unpacks its whole weight to bf16 through the module global dequantize_w4a16_full (looked up at call time), applies overlays, calls F.linear and drops the weight, on every forward and every decode step. [backend/engine_runtime.py:356]
- There are four unpacker modes with identical bits: reference (five eager ops over int32 temporaries), lut (a 256-entry byte table) and compiled (torch.compile of the unpacker alone, one graph per weight shape) rebind engine_runtime.dequantize_w4a16_full, while resident binds a new forward on every packed module over a kept dense weight and saves the old forward for restoring. [backend/w4a16_fast.py:494]
- Resident keeps all 410 weights or none, needs 58.6 GB beside the 4-bit weights, and is refused at install when the kept bytes plus the unpacking transient exceed free memory, so it cannot run on the 32 GB card; no per-linear resident mode exists. [backend/w4a16_fast.py:499]
- The server takes the compiled unpacker by default for the checkpoint on a CUDA card (since 11 November 2026), while the scripts and the lab daemon default to reference unless told otherwise. [backend/w4a16_fast.py:410]
- The installed compiled unpacker serves only the weight shapes tested at install and falls back to the reference, counted in .fell_back and logged once, for any other shape and for any call whose compile guards fail, such as one under torch.inference_mode or autocast. [backend/w4a16_fast.py:205]
- The only weight overlay is two per-module dicts: when either is non-empty the freshly unpacked weight is cloned once, each column overlay is assigned as w[:, i] = v in a Python loop, then each row overlay as w[i, :] = v, and the edited weight is used for every row of the batch alike. [backend/engine_runtime.py:360]
- Every one of the 410 linears carries the overlay dicts, attention included, but the vault writes only down_proj columns and its rollback clears only the down, gate and up projections of each layer, so an overlay placed on an attention linear would survive a vault rollback. [backend/larql_vault.py:556]
- An overlay's value is an arbitrary bf16 vector (zeros for DELETE, the original column plus a scaled direction for UPDATE and INSERT), not held to the 4-bit grid, kept only in memory on the module. [backend/larql_vault.py:507]
- Weight readers disagree about overlays: slice_w_down, slice_w_gate, get_full_w_down, get_full_w_gate and write_col_norms apply them, while slice_w_up, get_full_w_up and gguf_weights.dense, dense_cols, dense_rows (and so atlas_ops.dense_weight and Weights.attn_weight) return the bare weight. [backend/engine_runtime.py:1070]
- A second overlay formulation already exists for the CPU int4 kernel path: the weight is left alone and the overlay's difference is added to the product (x[..., i] times (v - W[:, i]) for a column), which gives sums that are not the reference's bits. [backend/int4_cpu.py:138]
- The card is 31.36 GiB as torch counts it; the model alone rests at 18.00 GiB allocated and the model with the J++ lens at 21.2 GiB, and the desktop holds 3.4 to 5.4 GiB, leaving about 5.6 GiB at rest with the lens and about 8.8 GiB without it beside a desktop. [backend/engine_runtime.py:470]
- The worst sweep the server runs, at the 2,048 word-piece limit, peaked at 25.5 GiB by torch's count and 26,857 to 26,904 MiB of the card as the process's share (measured 18-19 October 2026 with expandable segments). [backend/engine_runtime.py:479]
- A census-shaped batch at 64 new word-pieces peaks above the model at 2.4-3.1 GiB for 32 rows, 4.8-5.3 GiB for 64 and 7.2-7.5 GiB for 96 with the lens freed; 128 rows and more were never tried, and a row costs about 0.8 MB per word-piece of context. [experiments/results/atlas_unpacker_steps_rtx5090.json:1]
- A dense bf16 copy costs 0.215 GiB for one feed-forward matrix, 0.89 GiB for a whole sliding layer and 0.99 GiB for a whole full-attention layer (computed from the shapes), so a handful of dense variants fits in the headroom but nothing in the code holds one. [backend/w4a16_fast.py:306]
- A decode step on the RTX 5090 takes 0.706 / 0.713 / 0.721 / 0.743 s on the reference unpacker at 1 / 32 / 64 / 96 rows, 0.275 / 0.282 / 0.287 / 0.303 s with lut and 0.091 / 0.103 / 0.109 / 0.126 s with compiled (28 October 2026); the reference step is flat because unpacking dominates it. [COMPUTE.md:52]
- A census question's prefill takes 0.10 s at one row and 0.26 s at 32 rows with compiled; a whole-sequence forward on the reference takes 1.1 s at 512 word-pieces, 1.3 s at 1,024 and 1.9 s at 2,048; three pushed answers of 80 word-pieces through the server take 8.3 s with compiled. [COMPUTE.md:61]
- A cold start loads the model and the J++ lens in about 13 to 16 s (recorded 16 and 27 October 2026), and the compiled unpacker's install with its test on all 410 linears adds 1.8 to 4.2 s; a comment in server.py says the load takes minutes. [scripts/atlas_build.py:22]
- On the GPU a row's numbers depend on the shape of its batch (cuBLAS picks kernels by shape) and not on the unpacker: an unpushed greedy answer in a census batch of 64 parts from the same answer written alone after 22 word-pieces at worst and in a batch of 32 after 54, while the same answer written alone twice is identical. [scripts/verify_atlas_writer.py:5]
- No determinism switch is set anywhere in the code; comparisons are kept honest by putting the base row in the same forward as the pushed rows, making the batch size and plan resume keys, canary batches per shard, a measured writer's floor per card class, a reading class in the model id for CPU or float32 readings, and recording the unpacker and weights card with every reading. [COMPUTE.md:89]
- interp_engine resolves named points to a module and a side for reading and writing: resid_pre, resid_mid, resid_post, mlp_in, mlp_pre, mlp_pre_linear, mlp_act (the input of down_proj), mlp_out, mlp_out_post, attn_in, attn_out, attn_out_post, z (the input of o_proj), value, the q and k norm points, embeddings, final_norm, lm_head, and any dotted module path; ordinary hooks fire on the packed linears. [.venv/lib/python3.12/site-packages/interp_engine/model.py:1149]
- atlas_ops.forward_from_layer runs layers start..59 and the final norm from a given residual stream under torch.no_grad with hooks firing as usual; no helper stops a forward at a layer, and interp_engine's forward_from declares a starting position for masks, not a starting layer. [backend/atlas_ops.py:1017]
- Autograd is off by default: the runtime builds EagerModel with requires_grad False, which freezes every parameter, and the unpackers, the runtime's captures, all of atlas_ops and the lab daemon's exec run under torch.no_grad. [backend/engine_runtime.py:845]
- Backward passes have run on the 31B on this card through a custom autograd function that saves only the packed words and scales and unpacks again in the backward (labkit's _W4A16LinearFn, R3's W4A16GradLinear which also passes vmap-batched cotangents); both return None for the weights, so they give gradients for activations only, and labkit's ignores overlays while R3's refuses to run when one is live. [experiments/lab/labkit.py:417]
- R4 pins backend/__init__.py, theodolite_math.py, engine_runtime.py and label_gate.py by sha256 (with seventeen study files, the model snapshot and the torch and transformers versions), but engine_runtime.py's sha256 is now f1a0b4c3... against the pinned 69eed1b4... because commits a766058 and 960b9ab of 12 November 2026 edited its load; the other three still match. [experiments/scripts/R4_common.py:101]
- One process owns the card at a time with no arbitration at home beyond the server and the lab daemon sharing port 8765; inside the server one asyncio lock (_GPU_LOCK) serialises sweeps, vault statements, scientist tools and atlas routes, and the scientist's LLM agent reaches the model only over HTTP through a stdio MCP server. [backend/server.py:99]

### Numbers
- Quantised linears: 410 (60 layers x 6, plus v_proj on the 50 sliding layers) [checkpoint model.safetensors header; backend/w4a16_fast.py:5]
- Packed weights in all: 29.29 billion (computed from the shapes; documents say 29.3 billion) [COMPUTE.md:28]
- Bits a weight: 4.5 (4-bit piece plus a bf16 scale per 32 weights) [backend/engine_runtime.py:296]
- Packed words and scales on the card: 13.64 GiB + 1.70 GiB [checkpoint model.safetensors header (parsed 14 November 2026)]
- Embedding (tied unembedding): bf16 [262144, 5376], 2.625 GiB [checkpoint model.safetensors header]
- Model alone at rest: 18.00 GiB allocated (22 October 2026) [experiments/results/R3_00_pilot.json vram.after_load]
- Model and J++ lens at rest: 21.2 GiB (18 and 27 October 2026); 21.24 GB in the lab daemon's log (16 October) [backend/engine_runtime.py:470; experiments/results/labd.log]
- J++ lens on the card: about 3.2 GiB (worked out, not measured) [scripts/census_run.py:100]
- Card total: 31.36 GiB as torch counts it; 32,607 MiB by nvidia-smi [experiments/results/R3_00_pilot.json]
- Desktop's share of the card: 3.4 to 5.4 GiB [backend/engine_runtime.py:488; experiments/results/R3_00_pilot.json]
- Load peak: 23.02 GiB allocated (22 October 2026, before the vision tower moved to the CPU, which took about 1.07 GiB) [experiments/results/R3_gpu_runs.jsonl; backend/engine_runtime.py:385]
- Worst sweep at 2,048 word-pieces: torch peak 25.5 GiB; process share 26,857-26,904 MiB (18-19 October 2026) [backend/engine_runtime.py:479]
- Sweep limit and sliding window: 2,048 word-pieces; 1,024 [backend/engine_runtime.py:493; backend/engine_runtime.py:895]
- Batch memory above the model at 64 new pieces: 32 rows 2.4-3.1 GiB; 64 rows 4.8-5.3 GiB; 96 rows 7.2-7.5 GiB (28 October 2026) [experiments/results/atlas_unpacker_steps_rtx5090.json; COMPUTE.md:52]
- Unpacking transient at one row: reference 0.962 GiB, lut 0.500 GiB, compiled 0.285 GiB above the model (28 October 2026); 12 bytes a weight = 1.39 GB by arithmetic for the reference [experiments/results/atlas_unpacker_steps_rtx5090.json; backend/w4a16_fast.py:306]
- Resident weights: 58.6 GB (54.6 GiB) computed; 'about 60 GB' and '62 GB' in documents [COMPUTE.md:44; backend/engine_runtime.py:411]
- Dense bf16 copy of one feed-forward matrix: 231.2 MB (0.215 GiB), against 65.0 MB packed (computed) [backend/w4a16_fast.py:306]
- Dense bf16 copy of a whole layer: 0.89 GiB sliding, 0.99 GiB full attention (computed) [checkpoint model.safetensors header]
- Decode step, reference: 0.706 / 0.713 / 0.721 / 0.743 s at 1 / 32 / 64 / 96 rows (28 October 2026) [COMPUTE.md:52]
- Decode step, lut: 0.275 / 0.282 / 0.287 / 0.303 s [COMPUTE.md:52]
- Decode step, compiled: 0.091 / 0.103 / 0.109 / 0.126 s [COMPUTE.md:52]
- Prefill of a census question, compiled: 0.10 / 0.26 / 0.45 / 0.60 s at 1 / 32 / 64 / 96 rows (reference 0.89 to 1.27 s) [experiments/results/atlas_unpacker_steps_rtx5090.json]
- Whole-sequence forward, reference: 1.1 s at 512 pieces, 1.3 s at 1,024, 1.9 s at 2,048 (18 October 2026) [backend/engine_runtime.py:490]
- Forward of a 128-piece prompt, reference, no graph: 0.649 s (22 October 2026) [experiments/results/R3_00_pilot.json]
- Three pushed answers of 80 pieces through the server: 8.3 s with compiled, 50-54 s on reference (11 November 2026) [COMPUTE.md:61]
- Cold start: about 16 s for model and J++ (27 October 2026); 13.3 s in the lab daemon's log (16 October) [scripts/atlas_build.py:22; experiments/results/labd.log]
- Unpacker install: compiled 4.23 s and 1.82 s in two records; lut 1.06 s [experiments/results/atlas_unpacker_steps_rtx5090.json; experiments/results/atlas_writer_floor_rtx5090.json]
- Partial-stack forward (steer lens): 20.5-21.9 s a layer-forward for a whole layer's neurons, compiled (29 October 2026) [experiments/results/atlas_steer_timing/summary.json]
- Per-layer cost of a forward at one row: about 1.5 ms with compiled, 11.8 ms on reference (computed: step / 60) [COMPUTE.md:52]
- Writer's floor: an unpushed answer parts from itself after 22 word-pieces at batch 64 and 54 at batch 32 (29 October 2026) [experiments/results/atlas_writer_floor_rtx5090.json]
- Batch-shape divergence: KL up to 0.014 between batch shapes [ARCHITECTURE.md:545]
- Gradient pass, 128-piece prompt, reference: forward with graph 0.661 s at 20.04 GiB; one backward 0.314 s; 32 vmap cotangents 1.20 s at 22.84 GiB; 64 cotangents or 4 prompts out of memory under a 24 GiB cap (22 October 2026) [experiments/results/R3_00_pilot.json]
- R1 VJP check: reference VJP 1.44 s; 128 batched directions 2.2 s; peak 25.0 GB [experiments/results/R1_00_vjpcheck.json]
- R3 GPU jobs: fit shards 1,712-1,987 s each, router 2,688 s, peaks 21.8-23.0 GiB; 5.54 of 8 approved GPU-hours (22 October 2026) [experiments/results/R3_gpu_runs.jsonl; COMPUTE.md:222]
- Lab daemon use: 314 exec calls, 282 ok, median 71 s [experiments/results/labd_exec_log_2026-10-16.jsonl]
- engine_runtime.py sha256: now f1a0b4c3e9bbda92..., R4's pin 69eed1b4f1a319ba... (commit 45f1630); changed by a766058 and 960b9ab on 12 November 2026 [experiments/scripts/R4_common.py:101]
- CPU: checkpoint prefill, reference: 17-20 pieces in 31-37 s on a Ryzen 9 7900X (29 October 2026) [COMPUTE.md:257]

### Reusable for a harness
- backend/w4a16_fast.py install / _uninstall / Refused: the working pattern for adding a forward path from outside the runtime file, either by rebinding the module global engine_runtime.dequantize_w4a16_full or by binding a new forward on each packed module and saving the old one for restoring.
- backend/w4a16_fast.py same_bits, same_layout, _probe, _self_test, piece_scale_table, table_agrees: a bit-and-stride test kit that can prove a new overlay forward with no overlay gives the pinned forward's bits on every linear.
- backend/w4a16_fast.py _make_resident_forward: a template for a per-linear forward over a kept dense weight, with overlay handling and a version-counter check that catches in-place writes.
- backend/engine_runtime.py dequantize_w4a16_full, dequantize_w4a16_rows, dequantize_w4a16_cols and TheodoliteRuntime.slice_w_down / slice_w_gate / slice_w_up / get_full_w_down / get_full_w_gate / get_full_w_up: reading any feed-forward weight, whole or by row or column, in O(K) for slices.
- backend/gguf_weights.py dense, dense_cols, dense_rows, with_overlays and backend/atlas_ops.py dense_weight, Weights (including attn_weight for q, k, v, o), attn_shape, layer_scalars: dense reads of any linear and the layer scalars for weight-space reasoning.
- backend/int4_cpu.py with_overlay_terms: an existing implementation of an overlay as a delta added to the linear's product, the form a low-rank or sparse edit would take.
- interp_engine HookManager.read / write and EagerModel.resolve_point, capture and steer (library version 1.14.0): named and dotted-path hook addresses for capturing and patching activations, and a place to hang an output-side weight delta on any linear.
- backend/engine_runtime.py TheodoliteRuntime.capture_rows, write_voice and capture_for_sweep: a low-memory capture of chosen rows, a greedy answer with step-0 log-probabilities for before-and-after comparison, and writing plus recapture inside one steering block.
- backend/atlas_ops.py forward_from_layer: running only layers start..59 from a cached residual stream, which makes a test of an edit at layer l cost (60 - l)/60 of a forward.
- experiments/lab/labkit.py _grad_linears and _same_pos_vjp, and experiments/scripts/R3_gpu.py W4A16GradLinear and grad_linears: the recompute-in-backward linears that make a backward pass fit on the card (R3_gpu.py is pinned by R4, so it is to be imported or copied, not edited).
- experiments/lab/labd.py and labclient.py: an existing code-execution daemon that holds the model in one process and runs submitted Python against it under a lock, with a persistent STORE dict, top-level await, a per-launch token, loopback only, and a log of every call.
- backend/runtime_factory.py make_runtime, weights_card, reading_class and CheckpointRuntime: one place that chooses the runtime from the environment, names the weights a reading came from, and shows how to subclass the pinned runtime without editing it.
- backend/server.py _GPU_LOCK with asyncio.to_thread: the in-process pattern for serialising GPU work while the event loop keeps answering.
- backend/cpu_memory.py and w4a16_fast.free_bytes: memory estimates and a free-memory read that a harness could use to refuse an edit or batch that will not fit.
- scripts/pod/pod_tool.py steps and scripts/verify_atlas_writer.py: the existing timer for decode steps per unpacker and batch, and the measurement of how far an unpushed answer agrees with itself across batch shapes (the noise floor an edit's effect must clear).
- scripts/atlas_stand_in.py and scripts/check_dequant.py: a tiny random Gemma 4 with 4-bit linears and a CPU check suite, so a harness can be tested bit for bit without the card.

### Gaps
- There is no general weight overlay: no low-rank (B times A) delta, no sparse set of entries and no dense block, only whole-column and whole-row replacement applied by a per-entry Python loop on a clone of the full dense weight.
- There is no per-row weight variant: an overlay applies to every row of a batch, so a base and an edited model cannot share one forward through the existing mechanism.
- There is no per-linear dense cache: resident is all 410 linears or none and cannot be installed on the 32 GB card.
- There is no overlay, snapshot or rollback for embeddings, norm weights, q_norm and k_norm weights or layer scalars, and no invalidation of the caches that depend on them (the runtime's write-norm cache, atlas_ops.Weights.scalars, the atlas indexes on disk).
- The vault's rollback does not clear overlays on attention linears, and the up_proj accessors and gguf_weights.dense readers do not apply overlays, so a general overlay needs one consistent read path.
- No code returns a gradient with respect to a weight or a weight delta; the autograd functions give input gradients only, and labkit's ignores overlays in force.
- There is no forward that stops at a given layer; only a forward that starts at one.
- Nothing records an overlay in a reading's provenance: the weights card and the model id name the checkpoint's files only.
- Nothing arbitrates the card between processes on the home machine, and the lab daemon has no timeout, no sandbox, runs under torch.no_grad on the event loop, and does not restore model state after a job.
- No measurement exists of what an overlay costs per forward, of a resident or partly resident forward's speed, of the split of a compiled step between unpacking and products, or of a backward pass under the compiled unpacker.
- Nothing in the files I was given requantises an edited dense weight back onto the 4-bit grid with its per-row, per-32-column scales.

### Constraints
- By convention the four runtime files R4 pins (backend/__init__.py, theodolite_math.py, engine_runtime.py, label_gate.py) are not edited; new behaviour must be installed from new modules, as w4a16_fast.py, runtime_factory.py and gguf_runtime.py do, and the already broken pin on engine_runtime.py should be raised with the owner before anyone relies on it either way.
- R4 also pins R3_gpu.py, R3_fitlib.py, R3_lrp.py and other study files by sha256, so the gradient linears there may be imported or copied but not changed.
- Each packed linear has one instance-level forward that the pinned patch, the resident unpacker, labkit's and R3's gradient linears and the int4 CPU kernel all replace in turn; a new overlay forward must compose with them or refuse, and must restore what it found.
- An installed but empty overlay layer must give the pinned forward's bits on every linear, proved at install as the unpackers prove theirs.
- Only one process may hold the model on the card, so the harness must run inside the server process under _GPU_LOCK or take the card in its place as the lab daemon does; it cannot run beside the server.
- Memory headroom at rest is about 5.6 GiB with the lens beside a desktop (about 8.8 GiB without the lens), and a batch, the unpacking transient (0.3 to 1.0 GiB) and any dense variant (0.215 GiB a feed-forward matrix) all come out of it.
- On the GPU a result's bits depend on the batch shape, so a base and a variant must be compared in the same forward or the same shape, and an effect on a greedy answer must clear the measured floor (22 word-pieces at batch 64, 54 at batch 32).
- An output-side delta and an edit folded into the bf16 weight round differently, so the harness must name which it measured and must not treat one as the other's exact result.
- An edit meant to stay on the 4-bit grid must respect that scales belong to an output row and a group of 32 input columns: a down_proj column shares each scale with 31 neighbouring neurons, while a gate or up row has 168 scales of its own.
- The embedding and unembedding are one tensor, so an edit to either changes both unless the tie is broken at a cost of 2.6 GiB.
- Forwards must not run under torch.inference_mode or autocast, or the compiled unpacker silently falls back to the reference and a step becomes about seven times slower.
- Gradients need the recompute-in-backward linears and enable_grad around a direct call of the model; the graph for one 128-piece prompt takes about 2 GiB, and four such prompts at once ran out of memory under a 24 GiB cap.
- One sweep may hold at most 2,048 word-pieces and the 50 sliding layers see only 1,024 back.
- A reading taken under an edit must carry its own record of the edit, because the weights card, the model id and every atlas index describe the unedited checkpoint.

### Surprises
- R4's sha256 pin on backend/engine_runtime.py does not hold today: the file's hash is f1a0b4c3... against the pinned 69eed1b4..., after two commits of 12 November 2026 changed its load, while ARCHITECTURE.md:133, COMPUTE.md:46-47, backend/w4a16_fast.py:2-3 and backend/gguf_runtime.py:4 still say it is pinned and scripts/check_dequant.py's seventh section would fail as written (I computed the hashes; I did not run the check).
- ARCHITECTURE.md:125 says Gemma 4's RMSNorm uses '(1 + w)' scaling, but the transformers code multiplies by the weight as it is and backend/atlas_ops.py:11 says the unit offset is already in w; a norm-weight edit must follow the code.
- ARCHITECTURE.md:158-163 lists four hook addresses and describes resid_post without the layer scalar, while the code offers some 25 named points plus any module path and reads resid_post after the scalar has been multiplied in.
- backend/server.py:114 says the model's load 'takes minutes on the 31B', but the two recorded figures are 13.3 s and about 16 s.
- The engine_runtime.py docstring promises '< 26GB peak VRAM', the load log and ARCHITECTURE.md:129 say the limit is 30.0 GB, and the measured worst sweep took 26,904 MiB of the card.
- Attention linears already carry the same overlay dicts as the feed-forward ones and the pinned forward would apply them, but the vault's rollback clears only the feed-forward projections.
- A row overlay on up_proj would change the forward but not what slice_w_up and get_full_w_up return, and gguf_weights.dense readers ignore every overlay.
- labkit's gradient linear ignores vault overlays, so a gradient taken under an edit would be the unedited model's, while R3's version refuses to run with an overlay live.
- The compiled unpacker silently returns to the reference (about seven times slower) under torch.inference_mode or autocast, counting the fall-backs and warning once.
- The resident mode exists in code and in the documents but can never have run on the home card, since it needs 58.6 GB.
- The reference decode step costs the same from 1 to 64 rows because unpacking dominates, so extra rows are nearly free; but weight overlays are global to a forward, so that free batching helps only edits expressed per row on the output side.
- interp_engine's forward_from sets a starting position for masks; the function that starts a forward at a layer is atlas_ops.forward_from_layer.
- The lab daemon, the nearest thing to a weight-space code-execution environment already in the repository, runs every job under torch.no_grad, blocks its event loop while a job runs, and leaves any weight, overlay or unpacker a job changed in place for the next job.
- The ten full-attention layers have no v_proj at all (their values are the normed keys), so an edit plan that assumes q, k, v, o on every layer is wrong for layers 5, 11, 17, 23, 29, 35, 41, 47, 53 and 59.

### Not established
- Whether the owner knows that R4's pin on backend/engine_runtime.py stopped holding on 12 November 2026, and whether that was intended.
- Whether compressed-tensors registers weight_packed and weight_scale as parameters or as buffers.
- The load's peak memory since the vision tower was moved to the CPU on 12 November 2026.
- How a compiled decode step splits between unpacking and matrix products; the documents say the split was never timed.
- How fast a resident forward is, and how fast a forward with only some linears resident would be.
- What an overlay costs per forward (the 231 MB clone and the per-entry loop).
- Whether the timed backward pass in R3's pilot crossed all 60 layers or only layers 30 to 59, and what a backward costs under the compiled unpacker.
- Whether a first load from a cold disk takes minutes, as a comment in server.py says, against the 13 to 16 s recorded.
- Whether any server route reports the vault's edits in force beside a reading (the vault reader's area).
- Whether a requantiser from a dense edited weight to the 4-bit grid exists elsewhere in the repository (the export reader's area).
