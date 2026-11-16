# Setting it up

*Everything needed to run Theodolite in full: what it needs, installing and downloading, the CPU and GGUF paths, a first sighting and the server’s WebSocket, mapping every neuron, the scientist’s backends and reach, a rented card, reproducing the study, and where everything is in the repository. The short form is in the [README](../README.md#getting-started).*

<sub>Moved out of the [README](../README.md) on 16 November 2026, word for word but for its links and headings, when the README was cut down. Layers count 1–60, as the pages count them; the census board and the commands count 0–59.</sub>

## What it needs

- One NVIDIA GPU with **≥ 32 GB**. The study ran on an RTX 5090 (32,607 MiB); the instrument peaks near 23 GB.
- Linux, Python 3.12, [uv](https://github.com/astral-sh/uv), and Node 20+ for the interface.
- About 22 GB of disk for the model and 6.4 GB for the two lenses; up to about 27 GB more to regenerate every cached
  activation.
- Versions used: torch 2.14.1+cu130, transformers 5.20.0, interp-engine 1.14.0, compressed-tensors 0.19.0,
  fastapi 0.143.2, fastmcp 4.1.0; for a GGUF, gguf 0.19.0 (checked with transformers 5.21.0).

> [!IMPORTANT]
> Run one GPU process at a time: the lab daemon, the instrument server and a standalone `mcp_scientist` each load the 31B model.

> [!WARNING]
> Never expose the lab daemon. It executes arbitrary submitted Python, and binds to 127.0.0.1 only. It refuses web
> pages and any Host but this machine's, and `/exec` wants the token it writes at each start to a file only you can
> read (`labclient.py` sends it). Its default port, 8765, is the server's too.

## Install and download

```bash
uv venv --python 3.12 .venv
uv pip install -r pyproject.toml
# the code runs with HF_HUB_OFFLINE=1, so download first:
.venv/bin/hf download google/gemma-4-31B-it-qat-w4a16-ct
.venv/bin/hf download neuronpedia/jacobian-lens \
    gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt \
    gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt
```

The study used model snapshot `52f3f65bc7a02d555763bc923bd1d9094898219d` and lens snapshot
`e166deb5ad540932100ef803c8c9bcc2a5453694`.

## On the CPU, or from a GGUF

(ARCHITECTURE §5; checked on the CPU stand-in; the 31B has run on the CPU for eight
offline prefills of the checkpoint, and from a QAT Q4_0, a Q4_K_M and a UD-Q4_K_XL GGUF, packed and through `int4mm`;
and on the home 5090 from the same three, packed, with the reference unpacker and with `compiled`)

```bash
uv pip install "gguf>=0.19"                            # reads GGUF files (the optional "gguf" extra)
THEODOLITE_GGUF=~/models/gemma-4-31b-it-q4_0.gguf .venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8765 --timeout-graceful-shutdown 3
THEODOLITE_GGUF=<file> THEODOLITE_UNPACKER=compiled .venv/bin/uvicorn backend.server:app \
    --host 127.0.0.1 --port 8765                       # its packed layers compiled: the reference's bits, faster
THEODOLITE_DEVICE=cpu .venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8765 --timeout-graceful-shutdown 3   # the checkpoint, on the CPU
THEODOLITE_GGUF=<file> THEODOLITE_DEVICE=cpu THEODOLITE_GGUF_LINEARS=resident THEODOLITE_GGUF_LENS=none \
    .venv/bin/python scripts/census_run.py --layer 45 --neurons random:8 --random-controls 8 --max-new 32
THEODOLITE_GGUF=<file> THEODOLITE_DEVICE=cpu THEODOLITE_GGUF_LINEARS=int4mm .venv/bin/uvicorn backend.server:app \
    --host 127.0.0.1 --port 8765                       # torch's int4 kernel: fast on the CPU, not the reference's bits
THEODOLITE_GGUF=<...Q4_K_M.gguf> THEODOLITE_DEVICE=cpu THEODOLITE_GGUF_LINEARS=int4mm THEODOLITE_GGUF_RESIDENT_GB=8.5 \
    .venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8765   # the Q6_K the kernel cannot take, resident
.venv/bin/python -I scripts/compare_gguf_ct.py <file>  # does the GGUF hold the checkpoint's weights?
.venv/bin/python -I scripts/check_gguf_fast.py --cuda --gguf <file>   # compiled against the reference, bit for bit
.venv/bin/python -I scripts/compare_int4mm.py --time   # how fast the int4 kernel is here, against the reference
THEODOLITE_GGUF=<file> THEODOLITE_DEVICE=cpu THEODOLITE_GGUF_LINEARS=int4mm THEODOLITE_GGUF_LENS=none \
    .venv/bin/python -I scripts/compare_int4mm.py --layer 45   # and how far its answers depart from it
```

The lab daemon (`experiments/lab/labd.py`) and the scientist's MCP server run alone (`backend/mcp_scientist.py`) take
the same variables. On the CPU, packed 4-bit weights take about 20 GB of RAM and unpack at every forward (minutes a
forward on a small machine; a prefill of the 31B took 31 to 37 s on a 12-core desktop); resident bf16
(`THEODOLITE_GGUF_LINEARS=resident`) takes about 62 GB and writes a word-piece in seconds; `int4mm` (a GGUF's Q4_0, Q4_1
and Q4_K layers, or `THEODOLITE_CHECKPOINT_LINEARS=int4mm` for the checkpoint) keeps 4-bit weights and multiplies them
in place, 70 to 150 times as fast as the packed forward at one row on one 4-core machine, but its sums are the kernel's,
not the reference's; the J++ lens adds 3.4 GB. A K-quant file's Q5_K, Q6_K and Q8_0 layers (and an Unsloth UD
file's IQ4_XS), which the kernel cannot take, can be held resident up to `THEODOLITE_GGUF_RESIDENT_GB` (the same bits as packed): a Q4_K_M's 55 Q6_K layers
need about 8 GB. On a 12-core desktop (2 November 2026, beside other work) the 31B through int4mm wrote a word-piece in
0.45 s from the QAT Q4_0 and 0.56 s from a Q4_K_M with its Q6_K resident, and took a 59-piece prefill in 3.5 s; other
machines are inferred: time a pilot first. On the home RTX 5090 (2 November 2026, beside a desktop;
`experiments/results/gguf_31b_gpu/`) each of those two files and Unsloth's UD-Q4_K_XL loads packed in about 21 s into
19 to 21 GiB of the card (the lens included in the 21) and writes a word-piece in 0.54 to 0.82 s on the reference
unpacker (the K files dearer; between two runs a file's step moved by 6 to 19%), and in 0.121 to 0.126 s with
`THEODOLITE_UNPACKER=compiled` (`--unpacker compiled`; `backend/gguf_fast.py`), its graphs compiled and tested against
the reference in 3 to 10 s at load; on the 12-core desktop `compiled` took one forward of the QAT Q4_0 (six pieces)
from 37.2 s to 11.9 s, after a 64 s install. A reading through `compiled` keeps the reference's model id, its bits being
the same; one taken on the CPU (or in float32, or through int4mm) is recorded under its own model id
(`google/gemma-4-31B-it-qat-w4a16-ct@cpu/bfloat16`, a GGUF's `gguf:<file>#<fingerprint>@cpu/bfloat16`, with `+int4mm`
after either), so no census run or index taken on the GPU is continued or read there; every index also records the
weights it was built from, and none built from other weights is read (a census run records the model id only). A sweep,
a census batch, the vault's voice, a causal check or a steer lens batch that the CPU's memory (MemAvailable) cannot
hold, by an estimate from the model's config, is refused in plain words before it starts (`THEODOLITE_CPU_MEMORY=off`
skips the estimate).

## Take a first sighting

```bash
(cd frontend && npm ci && npm run build)      # the server serves frontend/dist at /
.venv/bin/uvicorn backend.server:app --host 127.0.0.1 --port 8765 --timeout-graceful-shutdown 3
# open http://127.0.0.1:8765/   ·   live-reloading interface: (cd frontend && npm run dev), proxied to :8765
```

A page left open keeps the scientist's event stream open, and it never ends of itself: `--timeout-graceful-shutdown 3`
lets a stop end it after 3 s instead of waiting for every such page to close.

The server answers only its own pages, served from this machine, and scripts run on it: a request or a WebSocket whose
Host or Origin names another machine, or that a browser marks as sent from another site (a plain link excepted), is
refused (`backend/local_only.py`). So a page elsewhere, or one rebinding its name to 127.0.0.1, cannot run sweeps or
edit the vault through your browser. On Linux another account on this machine is refused too: for each connection the
server asks the kernel's table of them (/proc/net/tcp) which account made it, and a request whose sender has already
closed or reset its connection is refused as well, since the table can then no longer say; `THEODOLITE_PEER_CHECK=off`
turns that off (for a proxy in front, say), and each such refusal says so. Elsewhere it cannot tell, and says so once
in its log. While `npm run dev` runs, its proxy reaches the server as you, so another account can go through it.

The same sighting, asked of the server directly over its WebSocket, `/ws/theodolite` (every field below the first three
is optional, and each default reproduces the instrument as first built):

```jsonc
{"action": "sweep", "payload": {
  "prompt": "Tell me about yourself.", "chat": true, "assistant_prefix": "Hi! I",
  "system": null,                                   // an optional system turn
  "monuments": ["she", "he", "it"],                 // the three landmarks
  "frame_squaring": "lowdin", "frame_fnw": false,   // an order-free frame on the raw rows, as the interface sends
  "pole_ranking": "write", "thread_score": "direct",
  "steering": {"enabled": false, "direction": "fnw", "mask_quiet": true, "spare_bos": true, "strength": 0.01},
  "max_new_tokens": 6,                              // how many word-pieces it may write (0–256)
  "sampler": {"temperature": 0, "top_k": 0, "top_p": 1, "presence_penalty": 0, "seed": null},   // 0: greedy
  "sight_every_token": true, "rank_lookahead": false,   // a sighting of each word-piece written, up to max_sighted (64)
  "request_id": "r1"                                // stream the answer, and stop it with {"action": "cancel_sweep", "request_id": "r1"}
}}
```

Without a `request_id` it answers as it always did, with one binary frame (the path, the split, kinship between layers,
fifteen neurons a layer, the threads) and one JSON frame of words and readings; [`ARCHITECTURE.md`](../ARCHITECTURE.md) gives
both. With one, it streams `theodolite_progress` while it writes, `theodolite_written` with the answer once it is written,
one such pair for each word-piece sighted (each JSON frame naming its `token_index`, `read_pos`, `chosen` word-piece and the
`generation` echo), and `theodolite_sweep_end` last. A sampler field left out is off: temperature 0 (greedy), top-k 0 and
top-p 1 (no limit), no penalty, the seed drawn when it samples. The server passes every one on explicitly, since interp-engine
reads a missing one as the checkpoint’s own setting (temperature 1, top-k 64, top-p 0.95). It holds its GPU lock for the whole
sweep, so no vault edit lands between two word-pieces of one answer. The circuit tools
(`inspect_pole`, `trace_downstream_readers`, `test_motif_hypothesis`, `query_larql_vault`, and `causal_spot_check`, a
neuron's label tested by the study's N1A protocol at k = 10) are served over HTTP and as FastMCP tools from
`backend/mcp_scientist.py`; the WebSocket also takes `inspect_pole`, `trace_readers`, `larql` and `causal_check`, besides
`sweep` and `cancel_sweep` (ARCHITECTURE §4.5).
Every neuron label they and the sweep send carries `label_check`, and the replies carry the rule as `label_gate` ([TRUST.md](TRUST.md)).
`test_motif_hypothesis` sets the gap it finds against chance (`selectivity`: Welch’s t, a permutation p, the n each side,
no judgement below four sentences a side, `scripts/check_hypothesis_test.py`); it can read the two lists as matched pairs
(`paired`, by sign flips), tests each sentence’s strongest word-piece as well as its last, and corrects the p’s of
several neurons named in one call together (`also`, Benjamini–Hochberg); its `verdict` as first built, a threshold
on d at any n, is kept for old callers.
To check a build end to end, server, tools and pages together: `.venv/bin/python scripts/verify_e2e_ui_and_mcp.py`.

## Map every neuron (the atlas)

The atlas reads what it shows of every neuron from indexes built once on the GPU, and each layer’s census from runs taken
there; until they exist its pages say which are missing and read what they can live. With the server stopped (one GPU
process at a time):

```bash
.venv/bin/python scripts/atlas_build.py all                          # words, activation ranges, wiring, layout (minutes)
.venv/bin/python scripts/atlas_build.py stats --hf <dataset> --tokens 2000000   # optional: ranges over a corpus of your choosing
.venv/bin/python scripts/atlas_build.py steer_dose                   # the steer lens's dose for each layer, measured first (half an hour)
.venv/bin/python scripts/atlas_build.py steer --layers 40-59         # the steer lens, a range of layers at a time (hours)
.venv/bin/python -I scripts/atlas_steer_check.py                     # the built steer lens set beside the census (CPU, seconds): which layers it can be trusted at
.venv/bin/python scripts/atlas_build.py passages --hf <dataset> --tokens 2000000   # each face's strongest passages (the corpus stats read)
.venv/bin/python scripts/atlas_build.py wiring        # once, if built before 28 October 2026: each link's g and u, for the sign in use (minutes)
.venv/bin/python scripts/atlas_build.py wiring_check --layers 20,40,53 --writers random:512   # the links measured under pushes
.venv/bin/python scripts/census_run.py --layers 20,40,45 --design calib_ceil --neurons random:32 --random-controls 16   # doses first
.venv/bin/python scripts/census_tools.py curves 45 --weights         # where pushes start to tell and where text breaks (CPU)
.venv/bin/python scripts/census_run.py --layer 45 --neurons all --unpacker compiled   # a layer's census, break_rel, ranked (layers from 0; ~13 h)
.venv/bin/python scripts/census_watch.py                             # in another terminal: its progress bar, rate and finish
.venv/bin/python scripts/census_tools.py spread 45                   # its answers laid out by what they say (CPU)
```

They go to `$THEODOLITE_ATLAS_DIR` (else `atlas_data/`, which git ignores). The costs in each script’s help were
measured in a pilot on one RTX 5090 where they say so (27 October 2026), and are worked out from the arithmetic
elsewhere: time a pilot with `--layers` or `--neurons 0-255` on any other card first (COMPUTE.md). Names and notes are
kept in `$THEODOLITE_FIELD_NOTES`, else `~/.local/share/theodolite/field_notes.json`. A neuron folio’s name is that same
name, and its field book (each face’s jobs, its pattern and the four ticks) is kept beside it in `field_book.json`
(`$THEODOLITE_FIELD_BOOK`), whenever the server answers for the neuron; with no server, or from the moment it does not
keep a change, the folio keeps both in the browser, says so, and offers once to carry them to the server. The names and
notes the scientist writes are proposals, in `field_proposals.json` beside them (`$THEODOLITE_FIELD_PROPOSALS`), until
you keep or discard them on the atlas page: nothing reads one before it is kept, later chats included. To try the page
without the GPU or the weights, `.venv/bin/python scripts/atlas_stand_in.py --serve 8765` serves the whole instrument on
a tiny random Gemma 4 (build its indexes with `atlas_build.py all --stand-in`, and its passages with `atlas_build.py
passages --stand-in`); what it shows is random, and the page says so.

Every forward unpacks the checkpoint’s 4-bit weights afresh, and a GGUF’s packed layers too (`THEODOLITE_GGUF_LINEARS`,
above). `--unpacker lut` or `--unpacker compiled` on `atlas_build.py` and `census_run.py` (`THEODOLITE_UNPACKER` for the
server, the lab daemon and the scientist’s MCP server run alone) unpacks the checkpoint’s weights faster with the same
bits, tested against the runtime’s own unpacker at load on a GPU and refused in plain words on any difference. The
server needs no flag for it: with `THEODOLITE_UNPACKER` unset it takes `compiled` for the checkpoint on a CUDA card
(about 4 s more at load; a push’s three answers of 80 word-pieces then come in about 8 s where `reference` took 52,
the same text to the letter), and `reference` on the CPU or for a GGUF, whose packed layers take `compiled` when asked;
`THEODOLITE_UNPACKER=reference` keeps the old way.
The compiled graphs are kept between runs in `${XDG_CACHE_HOME:-~/.cache}/theodolite/torch_compile`, a folder only
your account can write (`TORCHINDUCTOR_CACHE_DIR` names another), not in torch’s default under `/tmp`; the first start
after this change (14 November 2026) compiles them afresh, once. Nothing prunes the folder (it held 193 MiB here that
day); deleted while nothing of the instrument’s is running, it is made again and the next start compiles afresh.
`--unpacker resident` keeps them unpacked, which needs a card or a machine with about 60 GB free beside the 4-bit
weights (an 80 GB card holds it, with little left for a batch). On a GGUF only `compiled` serves
(`backend/gguf_fast.py`: the reference’s bits, tested on every packed layer at load, on the card or the CPU); `lut` and
`resident` are the checkpoint’s and refused there in plain words (a GGUF is held resident by
`THEODOLITE_GGUF_LINEARS=resident` or `THEODOLITE_GGUF_RESIDENT_GB`). The 0.71 s a decode step in `census_run.py`’s
help was measured on the checkpoint with the default, `reference`; on the same 5090 `lut` took 0.28 s and `compiled`
0.103 s at a batch of 32 (COMPUTE §ii), and from a GGUF `compiled` took 0.12 to 0.14 s a step at one row and at 32
(COMPUTE §ii; no census on a GGUF has been timed) (ARCHITECTURE §2, §5).

`atlas_build.py passages --hf <dataset> --tokens 2000000` keeps, for every neuron and face, its sixteen strongest
windows of the corpus (one event a window) with the gate and up values there, for the atlas’s passages tab, the folio
and the scientist’s `passages`; give it the corpus `stats` read, for each passage’s share of the face’s extreme and the
samples’ threshold are taken against the stats index (`all` reads the study’s 36 texts unless given a corpus; the route
and the build say when the two differ). With the stats index there it also draws six a face at random above half its
extreme (`--sample`). It is one forward a window, as `stats` is: on one RTX 5090 it took 20 minutes for 2,000,054
word-pieces with `--unpacker compiled` (28 October 2026), where stats took 58 with `reference`. The windows it keeps are
the corpus’s own text: they stay in the atlas folder and are never committed.

The scientist runs on Claude Code (`claude`, installed and logged in; `$THEODOLITE_CLAUDE_MODEL` picks the model) or
Codex (`codex`) where the server finds them, or on an OpenAI-compatible endpoint: LM Studio (`http://127.0.0.1:1234/v1`,
or `$THEODOLITE_LMSTUDIO_URL`), Ollama (`http://127.0.0.1:11434/v1`, or `$THEODOLITE_OLLAMA_URL`), OpenRouter (its key in
`$OPENROUTER_API_KEY`), or any other (`$THEODOLITE_OPENAI_BASE_URL`, `$THEODOLITE_OPENAI_API_KEY`,
`$THEODOLITE_OPENAI_MODEL`, or `endpoints.json` in the scientist’s folder, ARCHITECTURE §4.8). Pick the backend, and for an
endpoint its model, in the scientist’s tab; a key is read from the environment and never reaches a page. Choose a model
that takes tool calls. An edit to its instructions (`prompt.local.md` in its folder) or tools reaches a Claude Code chat
at the default reach with your next message to it while idle; any other chat says it runs the old ones and offers a
restart, since that also ends its process group, with any commands it left running in it.

What the scientist makes for you, a page or a table say, it writes into its work folder and links in its reply; the
link opens the file in a browser tab, shown with no scripts and no network. Its `steer` puts several prompts, or
several candidate neurons, into one batch, which costs about the time of one. A push you want to run elsewhere leaves
as a llama.cpp control vector: the push tab’s “save for llama.cpp” writes the open neuron’s push at the dose shown, and
the scientist’s `export_control_vector` writes a recipe of several neurons, into `<atlas folder>/exports/` (a file
already there is never written over: the export is refused, and says so); load it with
`llama-server -m <a Gemma-4-31B-it GGUF> --control-vector-scaled FILE:1` (ARCHITECTURE §6: where llama.cpp adds it, what
still differs, and the three recipes taken through llama.cpp on 11 November 2026).

> [!WARNING]
> By default, under Claude Code the scientist reads, writes and edits files only in its own work folder and has no
> shell; under Codex its shell runs in a sandbox that has no network and writes there and in the temporary folders
> (`/tmp` and `/dev/shm`), and it can read your other files, a credential kept in a file among them (the Hugging Face
> token under `~/.cache/huggingface` was readable in it; SSH keys and Codex’s own login by the same rule). That was
> measured in Codex’s own sandbox helper, not through a chat (codex-cli 0.166.0, 14 November 2026).
> `THEODOLITE_SCIENTIST_REACH=home` gives it neuron-explorer’s own reach: a shell and file tools over your
> home folder, pre-approved, so it runs commands and edits files there without asking, and what it reads (the model’s
> answers, names and notes, attachments) can steer it; the page asks once before the first such chat. At either reach
> Claude Code loads no settings file, hook or CLAUDE.md (`--setting-sources ""`), so a CLAUDE.md it writes in its
> folder, or one in any folder above it, is not taken up by a later chat, and a `.claude/settings.json` written there
> runs no hook (under the old `project` setting it ran as a shell command, at either reach). These rules were tried on
> Claude Code 2.1.312 against a stand-in API, not a live model (`scripts/verify_scientist_cli.py`). Over an endpoint it
> has the atlas’s tools only, but your messages and what the tools read are sent to that endpoint (OpenRouter is a
> third-party service).

## On a rented card

(The plan, its prices and the budget rule: COMPUTE.md.)

```bash
scripts/export_constraints.sh                                   # at home: the venv's exact versions, dated
scripts/pod/setup.sh --constraints scripts/pod/constraints/home-<date>.txt     # on the pod, from a checkout on its volume
scripts/pod/pilot.sh --home-canary L45-main_rel.canary.jsonl    # the one-hour pilot, before any budget is asked for
scripts/pod/census_shard.sh --layer 45 --shard 1/4 --projected-hours 2.6 -- --neurons all --batch 128 --max-new 64
scripts/pod/sync.sh --from root@<ip> --port <port> --to ~/pods/h200-a --until-done   # at home: pull, then check the copy
.venv/bin/python -I scripts/gpu_log.py sum A1                   # GPU-hours and dollars against the owner's approval
```

`scripts/pod/` runs the hard jobs (a layer's census in shards, the steer lens, a corpus pass) on a rented card.
`setup.sh` puts everything on the pod's volume (the caches, the venv, the atlas's data, the field notes), so a stopped
pod restarts ready; the downloads are pinned to the snapshots above and tried without a token first, a token taken
from its environment for that alone and never written. `pilot.sh` times the decode step and the memory for each
unpacker at several batches, measures the writer's floor (`scripts/verify_atlas_writer.py`), takes a `calib_rel` run
and one steer-lens layer, sets a canary batch against the home card's (`census_run.py --canary-only`), and projects
each job × 1.25 against an approval. `census_shard.sh` runs one shard detached, logged and under a time limit, one
GPU job on the card at a time (`logs/gpu.lock`, which the pilot and the self-test take too); `sync.sh` pulls the runs,
metas, canaries and logs every few minutes and checks the final copy by listing; and `scripts/gpu_log.py` logs each
pod against the approval it counts against. Each takes `--dry-run`, the whole flow on the CPU stand-in
(`scripts/check_pod_scripts.py`); none has yet run on a rented card. The server on a pod is reached through an SSH
tunnel to its 127.0.0.1:8765, never through a provider's public URL (it binds 127.0.0.1, and refuses a Host that is
not this machine's).

## Reproduce the study

Run everything from the repository root: the scripts use repo-relative paths.

```bash
.venv/bin/python experiments/lab/labd.py                     # ~15 s load, port 8765 (LABD_PORT to change)
python3 experiments/lab/labclient.py experiments/scripts/pilot_self_gender.py --out experiments/results/pilot_self_gender.json
```

- **Daemon jobs and local scripts.** Most scripts are daemon jobs: they use the pre-bound `rt`, `lab` and `STORE` and
  top-level `await`, and are submitted with `labclient.py`. Scripts named `*_analyze*`, `*_tables*` or `*_local_*` run
  directly with `.venv/bin/python`.
- **Results files.** `X.json` is the job’s `result`; `X.json.full.json` is the daemon’s full response (`ok`, `stdout`,
  `error`, `result`); the `*.partial.json` files in E6/V7 hold the final data; a `null` `.json`, or `ok: false`, marks a
  failed attempt, and the report names the rerun.
- **Not in this repository.** About 27 GB of `*.pt`, `*.npz`, `*_cache/` and `*_acts/` files (activation caches, gradient
  chunks, fitted IT-lens tensors). The reports still mention them; re-run the job each report names to regenerate them.
- **Third-party text removed.** The N1B activation corpus (local `/usr/share/doc` and package sources) is not distributed,
  and verbatim contexts from it were removed from `N1B_labels.json`; results that depend on it will differ slightly on
  another machine.

## The shelves

| Path | What is there |
|---|---|
| `backend/` | `theodolite_math.py` (the four modules, the frames, the direct-path score, CKA, the unembed), `engine_runtime.py` (the interp-engine runtime, lens loader, W4A16 dequantiser, chat template), `server.py` (FastAPI and the WebSocket), `mcp_scientist.py` (FastMCP tools), `larql_vault.py`, `causal_check.py` (a label's causal test, N1A's protocol); the atlas: `atlas.py` (its routes), `atlas_ops.py` (pushes, reads, wiring, heads, the steer lens), `atlas_index.py` and `atlas_build.py` (its indexes), `census.py`, `field_notes.py`, `scientist_chat.py`, `scientist_tools.py` and `scientist_mcp.py` (the scientist), `scientist_prompt.md` and `scientist_methodology.md` (its instructions); `local_only.py` (who may ask, on every route and the socket), `w4a16_fast.py` (faster unpackers of the 4-bit weights, the same bits); the CPU and GGUF: `runtime_factory.py` (which runtime a process runs), `gguf_weights.py`, `gguf_model.py` and `gguf_runtime.py`, `gguf_fast.py` (the faster unpacker of a GGUF’s packed layers, the same bits) |
| `frontend/` | Vite and React: ten pages (welcome, instrument, layer leaf, neuron folio, atlas, landmarks, nudge, vault, lexicon, lineage), the rubrics and apparatus switches, and the study’s recorded sighting for when no server answers |
| `scripts/` | `verify_optics.py` (command-line checks), `verify_e2e_ui_and_mcp.py` (server, tools and headless-Chrome page checks), `verify_causal_check.py` (the causal test against the study's N1A and N1V figures, on the GPU), `check_causal_check.py` (the same test on a tiny CPU stand-in, which `verify_e2e_ui_and_mcp.py` runs first); the atlas: `atlas_build.py`, `census_run.py` and `census_tools.py`, `atlas_stand_in.py` (its CPU stand-in), and `check_atlas_ops.py`, `check_atlas_index.py`, `check_atlas_api.py`, `check_census.py` and `check_scientist.py` (its checks, on the CPU); `check_local_only.py` (the guard), `check_hypothesis_test.py` (the guess test against chance), `check_dequant.py` (the unpackers, bit for bit), `check_gguf.py`, `check_gguf_fast.py` and `compare_gguf_ct.py` (the GGUF path, its faster unpacker bit for bit, and whether a GGUF holds the checkpoint’s weights) |
| `experiments/REPORT.md` | The main write-up |
| `ROADMAP.md` | What is still open, gathered from the fault table in [TRUST.md](TRUST.md), the report’s errata, the pages and the records of R1 to R4 |
| `experiments/lab/` | `labd.py` (the GPU daemon holding the model and lenses), `labclient.py` (submit jobs), `labkit.py` (lens readouts, gender scores, the corrected decomposition, interventions, exact same-position VJPs through the 4-bit model) |
| `experiments/scripts/` | 410 files, 406 of them scripts and four the prompt and source lists of R2, R3 and R4; the filename prefix is the experiment’s ID |
| `experiments/results/` | Per-experiment `*_REPORT.md` / `*_NOTES.md`, JSON results, the two patches, the frozen job log `labd_exec_log_2026-10-16.jsonl` |
