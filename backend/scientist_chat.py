"""The atlas's scientist chat (`backend/scientist_chat.py`): a scientist beside the map that reads the atlas, runs the
model and points at things on the person's screen.

After the explorer's chat (lyramakesmusic/neuron-explorer, explorer/server.py, "Claude: autointerp and chat"), with
five backends; a chat keeps the one it starts with. A page with no pick of its own starts a chat on Claude Code; Gemini
is one family on the page, Antigravity by default and the Gemini CLI as its other choice.
- "agy": Antigravity CLI (`agy acp --stdio`, JSON-RPC over stdio): the same MCP tools (passed per chat in
  `session/new` / `session/load`, with `_meta.antigravity.systemPrompt.append` for the instructions) and Antigravity's
  own tools, within its reach, in an empty read-only working folder (`0o555`), with `--gemini-dir` pointed at a folder
  of the scientist's own (`agy_gemini_dir`: so the person's own Antigravity settings, plugins, skills, rules and
  subagents do not load and its conversations stay out of their store), `--include-default-customizations=false`,
  `--disable-telemetry`, an environment built from nothing (`child_env`: the login is in the system keyring, over
  D-Bus), and at the default reach `--sandbox`, gating permissions via `session/request_permission`
  (`_agy_permission_outcome`). Checked on agy 1.5.14 only, and started on no other (`AGY_VERSION`). A chat cannot be
  taken back (/rewind is refused): Antigravity's ACP has no way to drop a turn. One that sets itself a timer (its
  `schedule` tool asks no permission) is stopped, and so, at the folder reach, is one that starts work with no message
  of the person's in flight (`_agy_own_work`).
- "claude": Claude Code (`claude -p`, stream-json in and out), one process per chat kept open, so a message sent while
  it works reaches it between tool calls and an interrupt stops the reply without ending the process. Its tools are
  the atlas's (backend/scientist_mcp.py, over MCP) and its own file tools, within its reach (`reach`). The conversation
  is Claude Code's own session (`~/.claude/projects/.../<chat>.jsonl`), resumed turn by turn.
- "codex": Codex (`codex app-server`, JSON-RPC), the same MCP tools and Codex's own shell, within its reach, in a
  Codex home of its own (the person's login, model and effort, nothing else).
- "gemini": Gemini CLI (`gemini --acp`, JSON-RPC over stdio; for `$GEMINI_API_KEY` / Vertex AI / Enterprise users),
  the same MCP tools (passed per chat in `session/new` / `session/load`) and Gemini's own tools, within its reach, in
  a Gemini home of its own (`$GEMINI_CLI_HOME`: the person's login and model preference, nothing else; usage statistics
  and telemetry off) and an empty read-only working folder (`0o555`) so no planted `work/.gemini/settings.json`, hook,
  skill or rogue MCP server loads. Its temporary files (Gemini CLI writes a whole conversation there when a request
  fails) go to a folder of the chat's own (`chat_tmp`).

Reach (`$THEODOLITE_SCIENTIST_REACH`): "folder", the default, keeps Claude Code and Gemini to reading, writing and
editing files in their work folder, with no shell, Antigravity too (it asks before reading anywhere else, and is refused:
two such reads, one in /tmp, in a live chat on agy 1.5.14, 22 November 2026), and runs Codex's shell in its workspace-write sandbox (no network;
it writes in the work folder and in the temporary folders, /tmp and /dev/shm, and can read files elsewhere: measured in
Codex's own sandbox helper, codex-cli 0.166.0, 14 November 2026, not through a chat). "home" is the explorer's own
setting, taken only when the person sets it: Claude Code's Read, Glob, Grep, Write, Edit and Bash pre-approved over the
home folder; Gemini's shell, never asking, with its file tools in the work folder; Antigravity approving everything,
with no sandbox; Codex with full access and never asking. Either way
Claude Code loads no settings file, hook, rules file or CLAUDE.md (`--setting-sources ""`: a CLAUDE.md it wrote in its
work folder would otherwise reach every later chat as instructions, and so would one in any folder above it, and a
.claude/settings.json written there, by a Codex chat whose sandbox may write in that folder, say, would run its hooks as
shell commands), Gemini loads no workspace settings, hooks, skills, extensions or ancestor GEMINI.md files
(`memoryBoundaryMarkers: []`, `--extensions none`, empty read-only cwd, Policy Engine allowing only `atlas` MCP tools
over `tools.core`), and each Gemini chat has a working folder of its own (`gemini_cwd`), so the store its file tools
may write (its conversation, its memory) is its own. Gemini CLI 0.70.0 would still read any GEMINI.md in its work
folder, or in a folder below it, as instructions (at a session's start, and beside any file it reads), and no setting
of its turns that off; Antigravity reads GEMINI.md, AGENTS.md and a `.agents` folder (or `.agent`, `_agents`,
`_agent`: rules, skills, hooks, plugins) as its own, there and in each folder above up to the repository's root. Every
backend can write in the work folder, so one chat could leave standing instructions for every later one: a Gemini or
Antigravity turn is refused while such a file is there (`context_files`), and Gemini's file tools are given only the
work folder, never the home folder. Each process is given its own login and no other: not the endpoints' keys, not another CLI's, not the
Hugging Face, cloud, GitHub or other tokens in the server's environment (`child_env`), and the page says what the
chosen backend can do (`REACH_WORDS`), asking once for each backend before its first chat when it is "home". What a
scientist reads (model output, census answers, names and notes, attachments) can carry instructions; with a shell
nothing stands between them and it.
- "openai": any OpenAI-compatible Chat Completions endpoint (LM Studio, Ollama, OpenRouter, vLLM, llama.cpp's server)
  that supports tool calling. The endpoint runs no tools, so the server runs the loop: it streams the model's reply,
  runs each tool call through the atlas's own routes in this process, sends the results back, and goes on until the
  model answers without a call (or `MAX_ROUNDS`). Such a model has the atlas's tools only: no shell and no files.

Endpoints ("profiles", `profiles()`): lmstudio (http://127.0.0.1:1234/v1), ollama (http://127.0.0.1:11434/v1),
openrouter (https://openrouter.ai/api/v1, its key in $OPENROUTER_API_KEY), custom ($THEODOLITE_OPENAI_BASE_URL with
$THEODOLITE_OPENAI_API_KEY and $THEODOLITE_OPENAI_MODEL), and any named in `<scientist folder>/endpoints.json`. A
profile names the environment variable its key is in, never the key: the key is read when a request is made and is
never sent to a page. A page picks a profile by name and a model; it cannot point a key at another address.

Every route answers only the instrument's own pages (`atlas.local_pages_only`); a chat's process works in the
scientist folder (`$THEODOLITE_SCIENTIST_DIR`, else `<atlas folder>/scientist`: `work/` is its working folder, beside
`uploads/`, `logs/` (the codex and openai chats' events and messages), `chats.json` and the system prompt built from
backend/scientist_prompt.md, backend/scientist_methodology.md and `prompt.local.md` there). Events reach the page as
NDJSON (GET /events): state, sent, text, thinking, tool, tool_result, ui, error, done, note, stale, ping. The routes are
in ARCHITECTURE.md §4.8.

What a scientist writes in its work folder the person opens from the chat (GET /work/<path>, `work_file`): a page, a
picture or a text in a browser tab, anything else as a download, each sent under a policy that runs no script and
fetches nothing (`WORK_CSP`), since the file's author read the model's output. So a deliverable is a file written
there, an HTML page with its styles inline say, not a script the person must run: at the folder reach Claude Code has
no shell to run one with.

A Claude Code or Codex chat reads its instructions and tools when its process starts (`agent_files`: the prompt, its
methodology, `prompt.local.md`, the MCP server and the tools it serves). When they have changed since, the person's next
message to a live, idle chat takes them up only where that cannot end anything else (`_take_up_changes`): a Claude Code
chat at the folder reach has no shell, so only its MCP server lives in its process group, and it is ended and started
again, resuming its session (a `note` event). Any other (Codex at either reach, Claude Code at "home") may have left a
command running in the background, in the same group, so it is not ended unasked: it is marked stale (a `stale` event,
and the first `state` event of a stream carries it), and the page offers POST /restart, which ends that group, any
commands left running in it too, so that the next message resumes it with the files as they now stand. Idle is as the
chat's events tell it: Claude Code answers from its first streamed event, thinking included, and after a turn that took
a message mid-turn it may yet run that message as a turn of its own, so until its next turn begins it is marked stale
rather than ended. An openai chat's prompt is built afresh every round; its tools are this process's, and change only
when the server starts again.
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import scientist_tools as st
from backend.atlas import local_pages_only
from backend.census import drop_partial_tail
from backend.saved_prompts import LOCAL_HOSTS

logger = logging.getLogger("theodolite.scientist")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILES = (ROOT_DIR / "backend" / "scientist_prompt.md", ROOT_DIR / "backend" / "scientist_methodology.md")
MCP_SCRIPT = ROOT_DIR / "backend" / "scientist_mcp.py"
# what a Claude Code or Codex chat's process reads when it starts, besides prompt.local.md (`agent_files`)
AGENT_FILES = (*PROMPT_FILES, MCP_SCRIPT, ROOT_DIR / "backend" / "scientist_tools.py")
SCIENTIST_DIR_ENV = "THEODOLITE_SCIENTIST_DIR"
AGENTS = ("agy", "claude", "codex", "gemini", "openai")
FILE_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit", "Bash"]
GEMINI_FOLDER_TOOLS = ["read_file", "write_file", "replace", "list_directory"]
GEMINI_HOME_TOOLS = ["read_file", "write_file", "replace", "list_directory", "glob", "grep_search", "run_shell_command"]
GEMINI_TOOL_NAMES = {"read_file": "Read", "view_file": "Read", "write_file": "Write", "write_to_file": "Write",
                     "replace": "Edit", "replace_file_content": "Edit", "multi_replace_file_content": "Edit", "edit_file": "Edit",
                     "list_directory": "List", "list_dir": "List", "glob": "Glob", "find_by_name": "Glob",
                     "grep_search": "Grep", "run_shell_command": "Bash", "run_command": "Bash"}
# Antigravity's file tools (agy 1.5.14 offers read_file, write_file and edit_file; earlier builds the others)
AGY_WRITE_TOOLS = frozenset({"write_to_file", "write_file", "edit_file", "replace_file_content", "multi_replace_file_content"})
AGY_READ_TOOLS = frozenset({"read_file", "view_file", "list_dir", "find_by_name", "grep_search"})
# the one Antigravity CLI build the chat was checked on (its own data folder, its permission requests, 22 November 2026)
AGY_VERSION = "1.5.14"
GEMINI_AUTH_FILES = ("oauth_creds.json", "google_accounts.json", "google_account_id", "installation_id", "state.json")
# what Gemini CLI 0.70.0 and Antigravity read as instructions from their workspace (any case): `context_files`
GEMINI_CONTEXT_NAMES = frozenset({"gemini.md"})
AGY_CONTEXT_NAMES = frozenset({"gemini.md", "agents.md", ".agents", ".agent", "_agents", "_agent"})
REACH_ENV = "THEODOLITE_SCIENTIST_REACH"
# what each backend can do beyond the atlas's tools, said on the page
REACH_WORDS = {
    "folder": {"agy": "reads, writes and edits files only in its own work folder and has no shell (it asks before anything else, and is "
                      "refused), but may search the web and start subagents of its own without asking; if it sets itself a timer, "
                      "or starts work of its own with no message from you, the chat is stopped",
               "claude": "reads, writes and edits files only in its own work folder, and has no shell",
               "codex": "runs commands in a sandbox that has no network and writes in its own work folder and the temporary folders "
                        "(/tmp, /dev/shm); it can read files elsewhere, credentials kept in files among them",
               "gemini": "reads, writes and edits files only in its own work folder and in Gemini CLI's own files for it (this "
                         "chat's stored conversation and memory, the GEMINI.md of its Gemini home), and has no shell",
               "openai": "has the atlas's tools and nothing else: no shell, no files"},
    "home": {"agy": "has a shell with no sandbox and full access to this machine, and never asks before it acts (web search and "
                    "subagents of its own included); if it sets itself a timer with its schedule tool, the chat is stopped, with "
                    "anything it left running",
             "claude": "has a shell, and reads, changes and runs anything in your home folder without asking",
             "codex": "has a shell with full access to this machine, and never asks before it acts",
             "gemini": "has a shell with full access to this machine, and never asks before it acts; its file tools reach only "
                       "its own work folder and Gemini CLI's own files for it",
             "openai": "has the atlas's tools and nothing else: no shell, no files"},
}
MCP_PREFIX = "mcp__atlas__"
GEMINI_MCP_PREFIX = "mcp_atlas_"
CHAT_IDLE = 30 * 60  # seconds without listeners or turns before a chat's input is closed (it exits once its work is done)
MAX_ROUNDS = 40  # tool rounds in one openai turn
TOOL_CUT = 16000  # characters of a tool's reply an openai-compatible model reads
SHOWN_CUT = 1500  # characters of a tool's reply the page shows
MAX_UPLOAD = 25 << 20
# a file of the work folder as GET /work/<path> hands it to the person's browser: these open in a tab, anything else
# is a download. Every one goes out under WORK_CSP: no script runs, nothing is fetched, no form is sent, and the page
# has no origin of its own (sandbox), so a page a scientist wrote, or was talked into writing by what it read, can
# show itself and nothing more; its styles and images must be in the file (inline CSS and SVG, data: images)
WORK_TYPES = {".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8", ".svg": "image/svg+xml",
              **{ext: "text/plain; charset=utf-8" for ext in (".md", ".txt", ".py", ".json", ".jsonl", ".csv", ".tsv", ".log")},
              ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
WORK_CSP = "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:"
WORK_ROUTE = "/api/atlas/scientist/work/"
MAX_MESSAGE = 100_000
# what the events and /restart say when a chat takes up changed instructions or tools (`_take_up_changes`)
RESTARTED_WORDS = ("Its instructions or tools had changed since this chat's process started, so it was started again, resuming "
                   "the conversation, to take them up.")
STALE_WORDS = ("This chat's process started before its instructions or tools last changed, and it runs those it started "
               "with. Restarting it takes them up, and also ends its process group, with any commands it left running in it.")
ENDED_WORDS = ("Ended the chat's process group, and with it any commands it had left running in that group; the next message "
               "starts it again, resuming the conversation with its instructions and tools as they now stand.")

router = APIRouter(prefix="/api/atlas/scientist", dependencies=[Depends(local_pages_only)])
_app: Any = None


def bind(app: Any) -> None:
    """Called by backend/server.py: the app the openai backend's tool calls go through, in this process."""
    global _app
    _app = app


def scientist_dir() -> Path:
    env = os.environ.get(SCIENTIST_DIR_ENV)
    if env:
        return Path(env).expanduser()
    from backend.atlas_index import atlas_dir

    return atlas_dir() / "scientist"


def work_dir() -> Path:
    d = scientist_dir() / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def uploads_dir() -> Path:
    return scientist_dir() / "uploads"


def logs_dir() -> Path:
    return scientist_dir() / "logs"


def _bin(name: str) -> str:
    return os.environ.get(f"THEODOLITE_{name.upper()}_BIN", name)


def reach() -> str:
    """How far a Claude Code or Codex scientist reaches beyond the atlas's tools (module docstring): "folder" unless the
    person sets $THEODOLITE_SCIENTIST_REACH=home."""
    return "home" if os.environ.get(REACH_ENV, "").strip().lower() == "home" else "folder"


# Credentials no scientist needs, kept from every process: the checkpoint's host and the accounts a GPU is rented with
# (the census on a pod, say). Every AWS_* and AZURE_* variable counts, as their credentials have many names.
SECRET_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN", "RUNPOD_API_KEY", "VAST_API_KEY", "LAMBDA_API_KEY",
              "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "GOOGLE_APPLICATION_CREDENTIALS", "NEURONARY_TOKEN"}
SECRET_PREFIXES = ("AWS_", "AZURE_")
# Anything named like a credential (GH_TOKEN, GITHUB_TOKEN, GITHUB_PAT, NPM_TOKEN, NPM_CONFIG__AUTH, MISTRAL_API_KEY,
# DEEPSEEK_KEY, ..._SECRET, ..._PASSWORD, DB_PASS) is kept from every process too, but a CLI's own variables, which only
# that CLI is given, its login among them: Claude Code's (ANTHROPIC_* and CLAUDE_*: ANTHROPIC_API_KEY,
# ANTHROPIC_AUTH_TOKEN, CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_FOUNDRY_API_KEY and the rest it reads, or its login under
# ~/.claude), Codex's (OPENAI_API_KEY and CODEX_*: CODEX_API_KEY, CODEX_ACCESS_TOKEN, or ~/.codex/auth.json, linked into
# its own home) and Gemini CLI's (GEMINI_*, GOOGLE_GENAI_*, GOOGLE_API_KEY, GOOGLE_CLOUD_ACCESS_TOKEN, or its login under
# ~/.gemini, linked into its own home). Antigravity's login is in the system keyring, reached over D-Bus, and its
# environment is built from nothing (`AGY_ENV_PASS`): no key, which would switch how it signs in (GEMINI_API_KEY,
# AGY_ADC_AUTH among them).
CREDENTIAL_NAME = re.compile(r"(?:^|_)(?:TOKEN|AUTHTOKEN|API_KEY|APIKEY|KEY|SECRET|SECRET_KEY|ACCESS_KEY|PASSWORD|PASSWD|PASS|PAT|AUTH"
                             r"|CREDENTIALS)$", re.I)
OWN_LOGINS = {"claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
              "codex": ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"),
              "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_ACCESS_TOKEN")}
# each CLI's own namespace: what is named so goes to that CLI and no other
OWN_LOGIN_PREFIXES = {"claude": ("ANTHROPIC_", "CLAUDE_"), "codex": ("CODEX_",), "gemini": ("GEMINI_", "GOOGLE_GENAI_")}
LOGIN_ENV = frozenset(k for ks in OWN_LOGINS.values() for k in ks)
LOGIN_PREFIXES = tuple(p for ps in OWN_LOGIN_PREFIXES.values() for p in ps)
# When Claude Code is told to reach its model through Bedrock, Vertex or Foundry (its flag 1, true, yes or on, as Claude
# Code reads it), or Gemini CLI through Vertex AI, the cloud's variables it then reads are its login, and pass to it (not
# to Codex, whose own home names no such provider)
CLAUDE_CLOUD_LOGINS = {"CLAUDE_CODE_USE_BEDROCK": ("AWS_",), "CLAUDE_CODE_USE_VERTEX": ("GOOGLE_APPLICATION_CREDENTIALS",),
                       "CLAUDE_CODE_USE_FOUNDRY": ("AZURE_",)}
GEMINI_CLOUD_LOGINS = {"GOOGLE_GENAI_USE_VERTEXAI": ("GOOGLE_APPLICATION_CREDENTIALS",)}
# what an Antigravity process is given of this one's environment (the probe of agy 1.5.14, 22 November 2026): its HOME
# (the home reach reads it), who it runs as, its language, and the session bus its keyring login is reached over
AGY_ENV_PASS = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR")


def child_env(agent: str = "claude", **extra: str) -> Dict[str, str]:
    """The environment an Antigravity (`agent` "agy"), Claude Code ("claude"), Codex ("codex") or Gemini CLI ("gemini")
    process starts with: this one's, less every endpoint's key, the credentials above and every other credential-named
    variable, but with that process's own variables (its login among them) and no other CLI's; for Antigravity, only
    `AGY_ENV_PASS`."""
    if agent == "agy":
        return {**{k: os.environ[k] for k in AGY_ENV_PASS if k in os.environ}, "NO_COLOR": "1", **extra}
    keys = {p.get("key_env") for p in profiles().values()} | {"THEODOLITE_OPENAI_API_KEY", "OPENROUTER_API_KEY"}
    on = lambda v: (v or "").strip().lower() in ("1", "true", "yes", "on")  # noqa: E731
    cloud = CLAUDE_CLOUD_LOGINS if agent == "claude" else (GEMINI_CLOUD_LOGINS if agent == "gemini" else {})
    cloud_own = tuple(p for flag, ps in cloud.items() if on(os.environ.get(flag)) for p in ps)
    own, own_prefixes = set(OWN_LOGINS.get(agent, ())), OWN_LOGIN_PREFIXES.get(agent, ())

    def passes(k: str) -> bool:
        if k in keys:
            return False
        if k in own or (own_prefixes and k.startswith(own_prefixes)) or (cloud_own and k.startswith(cloud_own)):
            return True
        return not (k in SECRET_ENV or k.startswith(SECRET_PREFIXES) or k in LOGIN_ENV or k.startswith(LOGIN_PREFIXES)
                    or CREDENTIAL_NAME.search(k))

    return {**{k: v for k, v in os.environ.items() if passes(k)}, **extra}


def chat_tmp(cid: str) -> Path:
    """A chat's own temporary folder (0700), its process's TMPDIR: Gemini CLI writes a whole conversation to its
    temporary folder when a request fails, and the system's is open to everyone on the machine."""
    root = scientist_dir() / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    d = root / os.path.basename(cid)
    d.mkdir(exist_ok=True)
    os.chmod(d, 0o700)
    return d.resolve()


def agent_files() -> List[Path]:
    """The files a Claude Code or Codex chat takes its instructions and tools from when its process starts: the prompt
    and its methodology, the MCP server and the tools it serves, and the person's `prompt.local.md` beside the work
    folder."""
    return [*AGENT_FILES, scientist_dir() / "prompt.local.md"]


def _agent_files_stamp() -> Tuple[Any, ...]:
    """Each of `agent_files` by its modification time and size (None for one not there, so a prompt.local.md made or
    removed shows too)."""
    out: List[Any] = []
    for p in agent_files():
        try:
            s = p.stat()
            out.append((s.st_mtime_ns, s.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


# ------------------------------------------------------------------------------------------------------- the prompt

SHELL_START, SHELL_END = "<!-- shell -->", "<!-- /shell -->"
FOLDER_START, FOLDER_END = "<!-- folder -->", "<!-- /folder -->"
CODEX_NOTE = """

## This session

You are running as Codex. Your shell and file editing are Codex's own: where these instructions name Read, Glob, Grep,
Write, Edit or Bash, use your shell and apply_patch. Start anything that runs longer than about a minute detached
(`nohup {PYTHON} script.py > script.log 2>&1 &`), say where the log is, and end your reply; read the log in a later
turn."""
CODEX_FOLDER_NOTE = """

## This session

You are running as Codex, your shell in a sandbox: it writes in `{WORK}` and in the temporary folders (`/tmp`,
`/dev/shm`), can read files elsewhere, and has no network, so a script cannot reach the instrument's API; use the
atlas's tools for every reading. A script that only writes a file in `{WORK}` from numbers you already have (a page,
a table) you can run yourself: run it, and link what it wrote."""
AGY_NOTE = """

## This session

You are running as Antigravity CLI (`agy`). Where these instructions name Read, Write, Edit or Bash, use `read_file`,
`write_file`, `edit_file` and `run_command` (and `run_command` where they name Glob or Grep). Your work folder is
`{WORK}` (use paths inside `{WORK}` when writing deliverables, and link them as `{WORK_URL}<name>`); if a file tool
refuses a path outside it, use `run_command`. Start anything that runs longer than about a minute detached
(`nohup {PYTHON} script.py > script.log 2>&1 &`), say where the log is, and end your reply; read the log in a later
turn. Never use the `schedule` tool, nor start work that goes on after your reply ends: a chat that does is stopped."""
AGY_FOLDER_NOTE = """

## This session

You are running as Antigravity CLI (`agy`) with no shell. Where these instructions name Read, Write or Edit, use
`read_file`, `write_file` and `edit_file`. Your work folder is `{WORK}`: always write inside `{WORK}` (your starting
working directory is read-only; a write anywhere else is refused, and so is a read), and link anything you write there
as `{WORK_URL}<name>`. Never use the `schedule` tool, nor start work that goes on after your reply ends: a chat that
does is stopped."""
GEMINI_NOTE = """

## This session

You are running as Gemini CLI. Where these instructions name Read, Glob, Grep, Write, Edit or Bash, use `read_file`,
`glob`, `grep_search`, `write_file`, `replace` and `run_shell_command`. Your work folder is `{WORK}` (use paths inside
`{WORK}` when writing deliverables, and link them as `{WORK_URL}<name>`). Your file tools reach only `{WORK}`: read or
change a file anywhere else with `run_shell_command`. Start anything that runs longer than about a minute detached
(`nohup {PYTHON} script.py > script.log 2>&1 &`), say where the log is, and end your reply; read the log in a later
turn."""
GEMINI_FOLDER_NOTE = """

## This session

You are running as Gemini CLI with no shell. Where these instructions name Read, Write or Edit, use `read_file`,
`write_file`, `replace` and `list_directory`. Your work folder is `{WORK}`: always pass paths inside `{WORK}` (your
starting working directory is read-only), and link anything you write there as `{WORK_URL}<name>`."""
OPENAI_NOTE = """

## This session

You are running behind an OpenAI-compatible endpoint, with the atlas's tools and nothing else: no shell, no files, no
web. Where an analysis needs a script, write the script in your reply for the person to run, and say what it prints."""


def system_prompt(agent: str, self_url: str) -> str:
    """The scientist's instructions: the prompt and its methodology, then this machine's own additions
    (`<scientist folder>/prompt.local.md`), with {ROOT}, {API}, {WORK}, {WORK_URL} and {PYTHON} filled in; for an openai-compatible
    model the shell section is left out, as it has none."""
    text = "\n\n".join(p.read_text(encoding="utf-8").rstrip() for p in PROMPT_FILES)
    local = scientist_dir() / "prompt.local.md"
    if local.exists():
        text += "\n\n" + local.read_text(encoding="utf-8").rstrip()
    cut = lambda t, a, b: re.sub(re.escape(a) + r".*?" + re.escape(b), "", t, flags=re.S)  # noqa: E731
    keep = lambda t, a, b: t.replace(a, "").replace(b, "")  # noqa: E731
    if agent == "openai":
        text = cut(cut(text, SHELL_START, SHELL_END), FOLDER_START, FOLDER_END) + OPENAI_NOTE
    elif reach() == "home":
        text = keep(cut(text, FOLDER_START, FOLDER_END), SHELL_START, SHELL_END)
        if agent == "agy":
            text += AGY_NOTE
        elif agent == "codex":
            text += CODEX_NOTE
        elif agent == "gemini":
            text += GEMINI_NOTE
    else:
        text = keep(cut(text, SHELL_START, SHELL_END), FOLDER_START, FOLDER_END)
        if agent == "agy":
            text += AGY_FOLDER_NOTE
        elif agent == "codex":
            text += CODEX_FOLDER_NOTE
        elif agent == "gemini":
            text += GEMINI_FOLDER_NOTE
    return (text.replace("{ROOT}", str(ROOT_DIR)).replace("{API}", self_url).replace("{WORK_URL}", WORK_ROUTE)
            .replace("{WORK}", str(work_dir())).replace("{PYTHON}", sys.executable).strip() + "\n")


def prompt_file(agent: str, self_url: str) -> Path:
    """The prompt written beside the chats (read-only: it is built from the files named in `system_prompt`)."""
    out = scientist_dir() / f"system_prompt.{agent}.md"
    text = system_prompt(agent, self_url)
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            os.chmod(out, 0o644)
        out.write_text(text, encoding="utf-8")
        os.chmod(out, 0o444)
    return out


# ------------------------------------------------------------------------------------------------------- endpoints

PROFILES: Dict[str, Dict[str, Any]] = {
    "lmstudio": {"label": "LM Studio (this machine)", "base_url": "http://127.0.0.1:1234/v1", "url_env": "THEODOLITE_LMSTUDIO_URL"},
    "ollama": {"label": "Ollama (this machine)", "base_url": "http://127.0.0.1:11434/v1", "url_env": "THEODOLITE_OLLAMA_URL"},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "headers": {"X-Title": "Theodolite", "HTTP-Referer": "https://github.com/whp199/theodolite"}},
}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_profile_warning: Optional[str] = None


def profiles() -> Dict[str, Dict[str, Any]]:
    """Every endpoint the openai backend can use, by name: the built-in ones (their addresses can be moved by
    $THEODOLITE_LMSTUDIO_URL and $THEODOLITE_OLLAMA_URL), `custom` when $THEODOLITE_OPENAI_BASE_URL is set, and those of
    `<scientist folder>/endpoints.json` ({name: {label, base_url, key_env, model, stream, headers, params}}). A file
    entry holding a key itself is refused: name the environment variable it is in."""
    global _profile_warning
    out = {k: dict(v) for k, v in PROFILES.items()}
    for v in out.values():
        if v.get("url_env") and os.environ.get(v["url_env"]):
            v["base_url"] = os.environ[v["url_env"]]
    if os.environ.get("THEODOLITE_OPENAI_BASE_URL"):
        out["custom"] = {"label": "custom endpoint", "base_url": os.environ["THEODOLITE_OPENAI_BASE_URL"],
                         "key_env": "THEODOLITE_OPENAI_API_KEY", "model": os.environ.get("THEODOLITE_OPENAI_MODEL")}
    path = scientist_dir() / "endpoints.json"
    if path.exists():
        try:
            extra = json.loads(path.read_text(encoding="utf-8"))
            for name, v in (extra.items() if isinstance(extra, dict) else []):
                why = _bad_profile(name, v)
                if why:
                    _profile_warning = f"{path}: {name}: {why}"
                    continue
                out[name] = {k: v[k] for k in ("label", "base_url", "key_env", "model", "stream", "headers", "params",
                                               "completion_limit_field") if k in v}
        except (OSError, ValueError) as exc:
            _profile_warning = f"{path} could not be read: {exc}"
    return out


def _bad_profile(name: Any, v: Any) -> Optional[str]:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", name):
        return "a profile's name is lower-case letters, digits, - and _"
    if not isinstance(v, dict):
        return "a profile is a table"
    if any(k in v for k in ("api_key", "key", "token")):
        return "holds a key: name the environment variable it is in (key_env) instead"
    u = urlparse(str(v.get("base_url", "")))
    if u.scheme not in ("http", "https") or not u.hostname:
        return "base_url must be an http(s) address"
    if v.get("key_env") is not None and not ENV_NAME.match(str(v["key_env"])):
        return "key_env must be the name of an environment variable"
    if not isinstance(v.get("headers", {}), dict) or not isinstance(v.get("params", {}), dict):
        return "headers and params are tables"
    # the name of the field a request bounds its reply with (Levelling's driver sends one): one of the two the
    # OpenAI-compatible endpoints take
    if v.get("completion_limit_field") not in (None, "max_tokens", "max_completion_tokens"):
        return "completion_limit_field is max_tokens or max_completion_tokens"
    return None


def profile_card(name: str, p: Dict[str, Any]) -> Dict[str, Any]:
    """What a page may know of an endpoint: never its key."""
    host = urlparse(p["base_url"]).hostname or ""
    key_env = p.get("key_env")
    return {"name": name, "label": p.get("label", name), "base_url": p["base_url"], "this_machine": host in LOCAL_HOSTS,
            "key": ("set" if os.environ.get(key_env) else f"missing: set ${key_env} where the server starts") if key_env else "none needed",
            "model": p.get("model"), "stream": p.get("stream", True)}


def _endpoint_headers(p: Dict[str, Any]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json", **{str(k): str(v) for k, v in (p.get("headers") or {}).items()}}
    if p.get("key_env"):
        key = os.environ.get(p["key_env"])
        if not key:
            raise st.ToolError(f"This endpoint needs its key in ${p['key_env']}: set it where the server starts.")
        headers["Authorization"] = f"Bearer {key}"
    return headers


# --------------------------------------------------------------------------------------------------------- chats

_chats: Dict[str, Dict[str, Any]] = {}
_reaper: Dict[str, Any] = {"loop": None}


def _check_chat(cid: Any) -> str:
    """A chat's id is a UUID (the page makes it; it names files and Claude Code's session)."""
    try:
        return str(uuid.UUID(str(cid)))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="A chat's id is a UUID.") from None


def _meta() -> Dict[str, Any]:
    try:
        d = json.loads((scientist_dir() / "chats.json").read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _meta_set(cid: str, **kw: Any) -> None:
    m = _meta()
    m.setdefault(cid, {}).update(kw)
    path = scientist_dir() / "chats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _chat(cid: str, agent: Optional[str] = None) -> Dict[str, Any]:
    """A chat's state, made on first need: its backend is the one recorded for it, else the one it is first sent with."""
    _ensure_reaper()
    c = _chats.get(cid)
    known = _meta().get(cid, {}).get("agent")
    if c is not None and agent in AGENTS and not known and c["proc"] is None and not c["busy"] and c["agent"] != agent:
        listeners = c["listeners"]  # an event stream opened before the first message keeps listening
        del _chats[cid]
        c = None
    else:
        listeners = None
    if c is None:
        kind = known or (agent if agent in AGENTS else "claude")
        c = _chats[cid] = {"session": cid, "agent": kind, "proc": None, "listeners": listeners or set(), "turn": [], "busy": False,
                           "last": time.time(), "pending": [], "cancel": asyncio.Event(), "task": None, "lock": asyncio.Lock(),
                           # the agent files and the reach its process started with, and whether they have changed since
                           "files": None, "reach": None, "stale": False,
                           # Claude Code: a message went in mid-turn; one may yet run as a turn of its own (`_take_up_changes`)
                           "queued": False, "waiting": False}
    return c


def _log(cid: str, ev: Dict[str, Any]) -> int:
    """An event added to a codex or openai chat's replay log (`logs/<chat>.jsonl`); the byte its line begins at. A last
    line cut off mid-write (the server stopped while writing it) is cut off first, as a census run's is before it goes
    on, so that this one is a line of its own."""
    path = logs_dir() / f"{cid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_partial_tail(path)
    with open(path, "ab") as f:
        at = f.seek(0, os.SEEK_END)
        f.write((json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8"))
    return at


def _jsonl(path: Any) -> List[Tuple[int, Dict[str, Any]]]:
    """A chat's replay log, or Claude Code's session file, as its records, each with the byte its line begins at. A last
    line cut off mid-write is left out, as the census's readers leave one out (`tolerate_tail`), and so is any other
    line that does not parse, rather than the history, the past chats or a rewind failing on it; a line is decoded
    only once it is whole, so one cut inside a character fails nothing either."""
    out: List[Tuple[int, Dict[str, Any]]] = []
    pos = 0
    with open(path, "rb") as f:
        for raw in f:
            try:
                e = json.loads(raw) if raw.strip() else None
            except ValueError:  # a torn line (UnicodeDecodeError is one too)
                e = None
            if isinstance(e, dict):
                out.append((pos, e))
            pos += len(raw)
    return out


def _emit(c: Dict[str, Any], ev: Dict[str, Any]) -> None:
    c["last"] = time.time()
    if ev.get("t") != "ui":
        c["turn"].append(ev)
        if c["agent"] in ("agy", "codex", "gemini", "openai") and ev.get("t") in ("text", "reset", "tool", "tool_result", "error", "done"):
            _log(c["session"], ev)
    for q in list(c["listeners"]):
        q.put_nowait(ev)


def _cut(text: str, n: int = SHOWN_CUT) -> str:
    return text[:n] + (f" … ({len(text)} chars)" if len(text) > n else "")


def _self_url(request: Request) -> str:
    """This server's own address, for the MCP server a chat starts (the socket the request came in on)."""
    if os.environ.get("THEODOLITE_SELF_URL"):
        return os.environ["THEODOLITE_SELF_URL"].rstrip("/")
    host, port = (request.scope.get("server") or ("127.0.0.1", 8000))[:2]
    if host in (None, "", "0.0.0.0", "::", "testserver"):
        host = "127.0.0.1"
    return f"http://{'[' + host + ']' if ':' in str(host) else host}:{port}"


async def _stop_proc(proc: Any) -> None:
    if proc is None or proc.returncode is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            await asyncio.wait_for(proc.wait(), 5)
            return
        except asyncio.TimeoutError:
            pass


async def _end_chat(c: Dict[str, Any]) -> None:
    """Ends a chat's process now (and what it started): nothing runs the old files after it."""
    proc = c.get("proc")
    c["proc"] = None
    await _stop_proc(proc)
    _mark_stale(c, False)


def _started(c: Dict[str, Any], files: Tuple[Any, ...]) -> None:
    """A chat's process has just started, on the agent files as `files` stamps them (taken before it read them)."""
    c["files"], c["reach"], c["queued"], c["waiting"] = files, reach(), False, False
    _mark_stale(c, False)


def _mark_stale(c: Dict[str, Any], stale: bool) -> None:
    """Whether the chat's process runs instructions or tools older than the files', said to the page when it changes."""
    if bool(c.get("stale")) != stale:
        c["stale"] = stale
        _emit(c, {"t": "stale", "stale": stale, **({"text": STALE_WORDS} if stale else {})})


async def _take_up_changes(c: Dict[str, Any]) -> None:
    """Before a message goes into a Claude Code or Codex chat whose process is live and idle and started before its
    agent files last changed (module docstring): a Claude Code chat at the folder reach is ended, so that the message
    starts it again on them, resuming its session (it has no shell, so nothing but its MCP server lives in its process
    group); any other is marked stale, once, and ended only when the person asks (POST /restart), since ending its
    process group would also end any command it left running in the background there. A chat mid-turn is never ended,
    as far as its events tell (`busy`); nor is a Claude Code chat whose last turn took a message mid-turn, until its
    next turn begins (`waiting`): it may yet run that message as a turn of its own, so it is marked stale meanwhile."""
    proc = c.get("proc")
    if proc is None or proc.returncode is not None or c["busy"] or c.get("files") is None:
        return
    if c["files"] == _agent_files_stamp():
        return
    if c["agent"] in ("agy", "claude", "gemini") and c.get("reach") == "folder" and not c.get("waiting"):
        await _end_chat(c)
        _emit(c, {"t": "note", "text": RESTARTED_WORDS})
    else:
        _mark_stale(c, True)


def _ensure_reaper() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _reaper["loop"] is not loop:
        _reaper["loop"] = loop
        loop.create_task(_reap_chats())


async def _reap_chats() -> None:
    """Closes the input of chats idle past CHAT_IDLE with no one listening; the process exits once its own work is done."""
    while True:
        await asyncio.sleep(60)
        for c in list(_chats.values()):
            proc = c.get("proc")
            if proc and proc.returncode is None and not c["busy"] and not c["listeners"] and time.time() - c["last"] > CHAT_IDLE:
                try:
                    proc.stdin.close()
                except Exception:
                    pass


async def _write(proc: Any, obj: Dict[str, Any]) -> None:
    proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    await proc.stdin.drain()


# ------------------------------------------------------------------------------------------------- Claude Code


def _session_file(sid: str) -> Optional[str]:
    hits = glob.glob(os.path.join(os.path.expanduser("~/.claude/projects"), "*", f"{os.path.basename(sid)}.jsonl"))
    return hits[0] if hits else None


def _claude_cmd(cid: str, self_url: str) -> List[str]:
    mcp = {"mcpServers": {"atlas": {"command": sys.executable, "args": ["-I", str(MCP_SCRIPT)],
                                    "env": {"THEODOLITE_URL": self_url, "THEODOLITE_CHAT": cid}}}}
    if reach() == "home":
        # file tools and a shell over the home folder, pre-approved, so it can run scripts against this server
        tools = ["--tools", *FILE_TOOLS, "--allowedTools", "mcp__atlas", *FILE_TOOLS, "--add-dir", os.path.expanduser("~")]
    else:
        # files only in its work folder (an absolute path in a rule begins with //), no shell
        w = str(work_dir()).rstrip("/")
        tools = ["--tools", "Read", "Write", "Edit", "--allowedTools", "mcp__atlas", *[f"{t}(/{w}/**)" for t in ("Read", "Write", "Edit")],
                 "--disallowedTools", "Bash"]
    # no settings file, hook, rules file or CLAUDE.md at all (""): under "project" a CLAUDE.md in its work folder, which
    # it may write, or in any folder above it reached every later chat as instructions, and the work folder's
    # .claude/settings.json ran its hooks as shell commands, at either reach (seen on Claude Code 2.1.312)
    cmd = [_bin("claude"), "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--include-partial-messages", "--system-prompt-file", str(prompt_file("claude", self_url)),
           "--mcp-config", json.dumps(mcp), "--strict-mcp-config", *tools, "--setting-sources", ""]
    if os.environ.get("THEODOLITE_CLAUDE_MODEL"):
        cmd += ["--model", os.environ["THEODOLITE_CLAUDE_MODEL"]]
    return cmd + (["--resume", cid] if _session_file(cid) else ["--session-id", cid])


def _user_text(m: Dict[str, Any]) -> Optional[str]:
    """The text of a message the person sent (None for tool results and notes the CLI adds)."""
    if m.get("type") != "user" or m.get("isMeta"):
        return None
    c = (m.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and not any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
        return next((x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text"), "")
    return None


def _tool_results(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for blk in (m.get("message") or {}).get("content") or []:
        if isinstance(blk, dict) and blk.get("type") == "tool_result":
            o = blk.get("content")
            if isinstance(o, list):
                o = "\n".join(x.get("text", "") for x in o if isinstance(x, dict))
            out.append({"t": "tool_result", "id": blk.get("tool_use_id"), "error": bool(blk.get("is_error")), "output": _cut(str(o or ""))})
    return out


def _refusal(m: Dict[str, Any]) -> Optional[Dict[str, str]]:
    msg = m.get("message") if isinstance(m.get("message"), dict) else {}
    if m.get("type") != "assistant" or msg.get("stop_reason") != "refusal":
        return None
    cat = (msg.get("stop_details") or {}).get("category")
    return {"id": msg.get("id"), "text": f"stopped by the {cat} safety classifier" if cat else "stopped by a safety classifier"}


CONTEXT_NOTE = re.compile(r"^\[The person has [^\]]*\]\n\n")


def _shown_text(t: str) -> str:
    """What the person typed, without the note on what they had open."""
    return CONTEXT_NOTE.sub("", t)


async def _pump_claude(c: Dict[str, Any], proc: Any) -> None:
    """The process's stream-json output turned into chat events, until it exits."""
    async for line in proc.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("type") == "stream_event":
            c["busy"], c["waiting"] = True, False  # a turn runs, from its first event (thinking too): `_take_up_changes`
            e = m.get("event") or {}
            if e.get("type") == "content_block_delta":
                d = e.get("delta") or {}
                if d.get("type") == "text_delta":
                    _emit(c, {"t": "text", "text": d.get("text", "")})
                elif d.get("type") == "thinking_delta":
                    _emit(c, {"t": "thinking", "text": d.get("thinking", "")})
        elif m.get("type") == "assistant":
            for blk in (m.get("message") or {}).get("content", []):
                if blk.get("type") == "tool_use":
                    c["busy"] = True
                    _emit(c, {"t": "tool", "id": blk.get("id"), "input": blk.get("input"), "name": str(blk.get("name", "")).removeprefix(MCP_PREFIX)})
            r = _refusal(m)
            if r and r["id"] not in c.setdefault("refused", set()):
                c["refused"].add(r["id"])
                _emit(c, {"t": "error", "error": r["text"]})
        elif m.get("type") == "user":
            for ev in _tool_results(m):
                _emit(c, ev)
        elif m.get("type") == "result":
            if m.get("is_error") and m.get("subtype") != "error_during_execution":
                _emit(c, {"t": "error", "error": str(m.get("result"))[:600]})
            c["busy"] = False
            # a message that went in mid-turn may yet run as a turn of its own, once this one has ended
            c["waiting"], c["queued"] = bool(c.get("queued")), False
            _emit(c, {"t": "done", "cost": m.get("total_cost_usd")})
            c["turn"] = []
    err = (await proc.stderr.read()).decode(errors="replace")
    code = await proc.wait()
    if c.get("proc") is proc:
        c["proc"] = None
        _mark_stale(c, False)  # it ran the old files; the next message starts it on the new
        if c["busy"] or code not in (0, None):
            _emit(c, {"t": "error", "error": err.strip()[-600:] or f"claude exited {code}"})
        if c["busy"]:
            c["busy"] = False
            _emit(c, {"t": "done"})
        c["turn"] = []


async def _ensure_claude(c: Dict[str, Any], self_url: str) -> Any:
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    if not shutil.which(_bin("claude")):
        raise RuntimeError("Claude Code (the `claude` command) is not installed where the server runs.")
    files = _agent_files_stamp()  # before the prompt file is built from them
    # its own process group, so ending a chat also ends what it started (the MCP server, its scripts)
    proc = c["proc"] = await asyncio.create_subprocess_exec(*_claude_cmd(c["session"], self_url), cwd=str(work_dir()), env=child_env("claude"),
                                                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.PIPE, limit=1 << 24, start_new_session=True)
    _started(c, files)
    c["pump"] = asyncio.create_task(_pump_claude(c, proc))
    return proc


def attachment_block(fid: str) -> Optional[Dict[str, Any]]:
    """An uploaded file as a message content block: images and PDFs as themselves, anything else as text."""
    meta, raw = _upload(fid)
    if meta is None:
        return None
    kind = meta["type"]
    if kind in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        return {"type": "image", "source": {"type": "base64", "media_type": kind, "data": base64.b64encode(raw).decode()}}
    if kind == "application/pdf" or meta["name"].lower().endswith(".pdf"):
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(raw).decode()},
                "title": meta["name"]}
    return {"type": "text", "text": _file_text(meta["name"], raw)}


def _file_text(name: str, raw: bytes, cut: int = 200_000) -> str:
    text = raw.decode("utf-8", errors="replace")
    more = f"\n[... {len(text) - cut} more characters cut]" if len(text) > cut else ""
    return f'<file name="{name}">\n{text[:cut]}{more}\n</file>'


def _upload(fid: str) -> Tuple[Optional[Dict[str, Any]], bytes]:
    fid = os.path.basename(str(fid))
    mp = uploads_dir() / f"{fid}.json"
    if not re.fullmatch(r"[0-9a-f]{12}", fid) or not mp.exists():
        return None, b""
    meta = json.loads(mp.read_text(encoding="utf-8"))
    return meta, (uploads_dir() / f"{fid}-{meta['name']}").read_bytes()


# ------------------------------------------------------------------------------------------------------------ Codex


def _codex_home() -> str:
    """Codex's own home for the scientist: the person's login, their model and effort, and nothing else (no plugins,
    apps, memories or global instructions), so its tools are the atlas's and its shell."""
    import tomllib

    home = scientist_dir() / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    auth, src = home / "auth.json", Path(os.path.expanduser("~/.codex/auth.json"))
    if not os.path.lexists(auth) and src.exists():
        os.symlink(src, auth)
    try:
        user = tomllib.loads(Path(os.path.expanduser("~/.codex/config.toml")).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        user = {}
    lines = [f"{k} = {json.dumps(user[k])}" for k in ("model", "model_reasoning_effort") if isinstance(user.get(k), str)]
    lines += ["", "[features]", "apps = false", "memories = false"]
    (home / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(home)


async def _rpc(c: Dict[str, Any], method: str, params: Dict[str, Any], timeout: float = 120) -> Dict[str, Any]:
    rpc = c["rpc"]
    rpc["next"] += 1
    rid = rpc["next"]
    fut = asyncio.get_running_loop().create_future()
    rpc["pending"][rid] = fut
    try:
        await _write(c["proc"], {"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        r = await asyncio.wait_for(fut, timeout)
    finally:  # a request that timed out (or failed to go) waits for nothing more
        rpc["pending"].pop(rid, None)
    if "error" in r:
        err = r["error"]
        msg = err.get("message", err) if isinstance(err, dict) else err
        raise RuntimeError(f"{c['agent']} {method}: {msg}")
    return r.get("result") or {}


def _codex_text(item: Dict[str, Any]) -> Optional[Tuple[str, Any, str, bool]]:
    """A tool item's name, input, output and whether it failed, as the events show them."""
    k = item.get("type")
    if k == "mcpToolCall":
        res = item.get("result") or {}
        out = "\n".join(x.get("text", "") for x in res.get("content") or [] if isinstance(x, dict)) or str(item.get("error") or "")
        return str(item.get("tool", "tool")).removeprefix(MCP_PREFIX), item.get("arguments") or {}, out, item.get("status") == "failed"
    if k == "commandExecution":
        return "Bash", {"command": item.get("command")}, item.get("aggregatedOutput") or "", item.get("status") in ("failed", "declined")
    if k == "fileChange":
        ch = item.get("changes") or []
        return "Edit", {"file_path": ", ".join(str(x.get("path", "")) for x in ch if isinstance(x, dict))}, item.get("status", ""), item.get("status") == "failed"
    if k == "webSearch":
        return "WebSearch", {"query": item.get("query")}, "", False
    return None


async def _pump_codex(c: Dict[str, Any], proc: Any) -> None:
    rpc = c["rpc"]
    async for line in proc.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if "id" in m and ("result" in m or "error" in m) and "method" not in m:
            fut = rpc["pending"].pop(m["id"], None)
            if fut and not fut.done():
                fut.set_result(m)
            continue
        meth, prm = m.get("method"), m.get("params") or {}
        if "id" in m and meth:  # a request from codex (an approval): the thread runs without them; decline the rest
            await _write(proc, {"jsonrpc": "2.0", "id": m["id"], "result": {"decision": "decline"}})
            continue
        if meth == "turn/started":
            c["turn_id"], c["busy"] = (prm.get("turn") or {}).get("id"), True
        elif meth == "item/agentMessage/delta":
            c["busy"] = True
            if c.get("msg_item") != prm.get("itemId"):
                if c.get("msg_item"):
                    _emit(c, {"t": "text", "text": "\n\n"})
                c["msg_item"] = prm.get("itemId")
            _emit(c, {"t": "text", "text": prm.get("delta", "")})
        elif meth in ("item/started", "item/completed"):
            t = _codex_text(prm.get("item") or {})
            if t:
                name, inp, out, failed = t
                iid = (prm.get("item") or {}).get("id")
                if meth == "item/started":
                    c["msg_item"] = None
                    _emit(c, {"t": "tool", "id": iid, "name": name, "input": inp})
                else:
                    _emit(c, {"t": "tool_result", "id": iid, "error": failed, "output": _cut(out)})
        elif meth == "turn/completed":
            turn = prm.get("turn") or {}
            if turn.get("status") == "failed" and turn.get("error"):
                _emit(c, {"t": "error", "error": str(turn["error"].get("message", turn["error"]))[:600]})
            c["busy"], c["turn_id"], c["msg_item"] = False, None, None
            _emit(c, {"t": "done"})
            c["turn"] = []
        elif meth == "error":
            _emit(c, {"t": "error", "error": str((prm.get("error") or {}).get("message", prm))[:600]})
    await proc.wait()
    if c.get("proc") is proc:
        c["proc"] = None
        _mark_stale(c, False)  # it ran the old files; the next message starts it on the new
        if c["busy"]:
            c["busy"] = False
            _emit(c, {"t": "error", "error": "codex exited"})
            _emit(c, {"t": "done"})
        c["turn"] = []


async def _ensure_codex(c: Dict[str, Any], self_url: str) -> Any:
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    if not shutil.which(_bin("codex")):
        raise RuntimeError("Codex (the `codex` command) is not installed where the server runs.")
    env = child_env("codex", CODEX_HOME=_codex_home())
    files = _agent_files_stamp()  # before its instructions are built from them
    proc = c["proc"] = await asyncio.create_subprocess_exec(_bin("codex"), "app-server", cwd=str(work_dir()), env=env,
                                                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.DEVNULL, limit=1 << 24, start_new_session=True)
    _started(c, files)
    c["rpc"], c["turn_id"], c["msg_item"] = {"next": 0, "pending": {}}, None, None  # a new process: no turn of the last runs on
    c["pump"] = asyncio.create_task(_pump_codex(c, proc))
    await _rpc(c, "initialize", {"clientInfo": {"name": "theodolite", "version": "1"}})
    # in its sandbox Codex asks before every MCP tool call, and this chat never asks (approvalPolicy "never"): the
    # atlas's own tools are approved, as full access approved them
    mcp = {"atlas": {"command": sys.executable, "args": ["-I", str(MCP_SCRIPT)], "env": {"THEODOLITE_URL": self_url, "THEODOLITE_CHAT": c["session"]},
                     "default_tools_approval_mode": "approve"}}
    sandbox = "danger-full-access" if reach() == "home" else "workspace-write"  # module docstring: reach
    opts = {"cwd": str(work_dir()), "approvalPolicy": "never", "sandbox": sandbox, "config": {"mcp_servers": mcp},
            "developerInstructions": system_prompt("codex", self_url)}
    thread = _meta().get(c["session"], {}).get("thread")
    r = await _rpc(c, "thread/resume", {"threadId": thread, **opts}) if thread else await _rpc(c, "thread/start", opts)
    c["thread"] = (r.get("thread") or {}).get("id")
    if not thread:
        _meta_set(c["session"], thread=c["thread"])
    return proc


def _codex_input(fid: str) -> Optional[Dict[str, Any]]:
    meta, raw = _upload(fid)
    if meta is None:
        return None
    if meta["type"].startswith("image/"):
        return {"type": "localImage", "path": str(uploads_dir() / f"{os.path.basename(fid)}-{meta['name']}")}
    if meta["type"] == "application/pdf" or meta["name"].lower().endswith(".pdf"):
        return None
    return {"type": "text", "text": _file_text(meta["name"], raw), "text_elements": []}


# ------------------------------------------------------------------------------------------------------- Gemini CLI


def _reset_ro_cwd(cwd: Path) -> None:
    """An empty read-only directory (0o555) used as Gemini CLI's cwd so no workspace `.gemini/settings.json`, hook,
    skill, agent or policy inside `work/` can ever be discovered or written there."""
    if cwd.exists():
        try:
            has_entries = any(cwd.iterdir())
        except OSError:
            has_entries = True
        if has_entries:
            os.chmod(cwd, 0o700)
            for child in list(cwd.iterdir()):
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
    else:
        cwd.mkdir(parents=True, exist_ok=True)
    os.chmod(cwd, 0o555)


def gemini_cwd(cid: str) -> Path:
    """A Gemini CLI chat's own empty read-only working folder. Gemini keeps a project's conversations and memory in a
    folder named for its working folder (`.gemini/tmp/<project>`) and lets its file tools write there, so each chat has
    one of its own: one chat cannot write into another's stored conversation."""
    return scientist_dir() / "gemini-home" / "cwds" / os.path.basename(cid)


def _gemini_home(cid: str) -> Tuple[Path, Path, Path]:
    """Gemini CLI's own home for the scientist (`$GEMINI_CLI_HOME`): the person's login and model preference, nothing
    else (no extensions, hooks, skills, user GEMINI.md or workspace settings), with usage statistics and telemetry off
    (they are on by default, and the person's own opt-out is not read from their settings). Returns `(gemini_home,
    ro_cwd, policy_path)`, `ro_cwd` the chat's own (`gemini_cwd`)."""
    home = scientist_dir() / "gemini-home"
    gdir = home / ".gemini"
    pdir = gdir / "policies"
    cwd = gemini_cwd(cid)
    pdir.mkdir(parents=True, exist_ok=True)
    cwd.parent.mkdir(parents=True, exist_ok=True)
    _reset_ro_cwd(cwd)

    user_gdir = Path(os.path.expanduser("~/.gemini"))
    for fname in GEMINI_AUTH_FILES:
        dst, src = gdir / fname, user_gdir / fname
        if not os.path.lexists(dst) and src.exists():
            os.symlink(src, dst)

    try:
        user_cfg = json.loads((user_gdir / "settings.json").read_text(encoding="utf-8"))
        if not isinstance(user_cfg, dict):
            user_cfg = {}
    except (OSError, ValueError):
        user_cfg = {}

    on = lambda v: (v or "").strip().lower() in ("1", "true", "yes", "on")  # noqa: E731
    user_auth = ((user_cfg.get("security") or {}).get("auth") or {}).get("selectedType")
    if os.environ.get("GEMINI_API_KEY"):
        selected_auth: Optional[str] = "gemini-api-key"
    elif on(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")):
        selected_auth = "vertex-ai"
    elif isinstance(user_auth, str) and user_auth:
        selected_auth = user_auth
    elif on(os.environ.get("GOOGLE_GENAI_USE_GCA")):
        selected_auth = "oauth-personal"
    else:
        selected_auth = None

    user_model = (user_cfg.get("model") or {}).get("name")
    preview = (user_cfg.get("general") or {}).get("previewFeatures")
    settings: Dict[str, Any] = {
        "security": {
            "auth": {"selectedType": selected_auth} if selected_auth else {},
            "folderTrust": {"enabled": True},
        },
        "context": {
            "memoryBoundaryMarkers": [],
            # read only by the interactive /directory add on 0.70.0, not at a session's start: what keeps GEMINI.md
            # out is `context_files`
            "loadMemoryFromIncludeDirectories": False,
            "includeDirectoryTree": False,
        },
        # Clearcut usage statistics (sent to play.googleapis.com with the account or installation id) and telemetry
        "privacy": {"usageStatisticsEnabled": False},
        "telemetry": {"enabled": False},
        "hooksConfig": {"enabled": False},
        "hooks": {},
        "skills": {"enabled": False},
        "general": {
            **({"previewFeatures": preview} if isinstance(preview, bool) else {}),
            "enableAutoUpdate": False,
            "enableAutoUpdateNotification": False,
            "enablePromptCompletion": False,
        },
        "tools": {
            "core": list(GEMINI_HOME_TOOLS if reach() == "home" else GEMINI_FOLDER_TOOLS),
            "discoveryCommand": "",
            "callCommand": "",
        },
        "mcp": {"allowed": ["atlas"]},
        "mcpServers": {},
    }
    if isinstance(user_model, str) and user_model:
        settings["model"] = {"name": user_model}
    (gdir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    # Policy Engine priority 500 in the user tier (4.500) sits above tools.core's "*" deny (4.24) for MCP tools only,
    # while mcp.allowed + --allowed-mcp-server-names limit MCP connections to the single "atlas" server.
    policy_path = pdir / "theodolite.toml"
    policy_path.write_text(
        '[[rule]]\nmcpName = "*"\ntoolName = "*"\ndecision = "allow"\npriority = 500\n',
        encoding="utf-8",
    )
    return home, cwd, policy_path


def _gemini_cmd(policy_path: Path) -> List[str]:
    """Gemini CLI's command line. Its workspace is its empty read-only cwd and the work folder, at either reach: a
    workspace folder is where Gemini reads GEMINI.md files from (`context_files`), so the home folder is never one; at
    the home reach its shell reaches the rest."""
    cmd = [
        _bin("gemini"), "--acp",
        "--approval-mode", "yolo",
        "--extensions", "none",
        "--allowed-mcp-server-names", "atlas",
        "--policy", str(policy_path),
        "--include-directories", str(work_dir()),
    ]
    if os.environ.get("THEODOLITE_GEMINI_MODEL"):
        cmd += ["--model", os.environ["THEODOLITE_GEMINI_MODEL"]]
    return cmd


def _repo_ancestors(p: Path) -> List[Path]:
    """The folders above `p`, up to the root of the repository it is in (the folder holding `.git`), that one included;
    none when it is in no repository."""
    try:
        parents = list(p.resolve().parents)
    except OSError:
        return []
    for i, d in enumerate(parents):
        if os.path.lexists(d / ".git"):
            return parents[: i + 1]
    return []


def context_files(agent: str) -> List[str]:
    """The files a Gemini CLI ("gemini") or Antigravity ("agy") chat would take as standing instructions from where a
    chat can write (module docstring): any GEMINI.md, and for Antigravity any AGENTS.md or customization folder
    (`.agents`, `.agent`, `_agents`, `_agent`), in any case, anywhere in the work folder or the chats' own working
    folders; for Gemini also a GEMINI.md in its own home and its projects' memory (MEMORY.md, GEMINI.md), which a chat
    can write; for Antigravity also one directly in any folder above the work folder or its working folder, up to the
    root of the repository they are in (the scientist folder is inside Theodolite's by default). Gemini CLI 0.70.0 reads
    a workspace folder's GEMINI.md when a session starts and any between a file it reads and that folder (its "just in
    time" context); `context.fileName` only adds names to GEMINI.md and nothing turns either off. Antigravity 1.5.14
    took a work/AGENTS.md and a work/GEMINI.md as its rules in a live chat (22 November 2026; a work/.agents/rules file
    it did not), and its own documentation says it reads GEMINI.md, AGENTS.md and those folders (rules, skills, hooks,
    plugins, MCP servers) from its working folder up to the repository's root. A turn is refused while any is there.
    It walks the whole work folder: run it off the event loop (`_context_refusal_async`)."""
    names = AGY_CONTEXT_NAMES if agent == "agy" else GEMINI_CONTEXT_NAMES
    roots = [work_dir(), scientist_dir() / ("agy-cwd" if agent == "agy" else "gemini-home/cwds")]
    hits: List[str] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            hits += [os.path.join(dirpath, n) for n in (*dirnames, *filenames) if n.lower() in names]
    if agent == "gemini":
        gdir = scientist_dir() / "gemini-home" / ".gemini"
        listed = [(gdir, names), *((m, names | {"memory.md"}) for m in gdir.glob("tmp/*/memory"))]
    else:
        listed = [(d, names) for d in sorted({a for root in roots for a in _repo_ancestors(root)})]
    for d, ns in listed:
        try:
            hits += [str(d / n) for n in os.listdir(d) if n.lower() in ns]
        except OSError:
            pass
    return sorted(set(hits))


def _context_refusal(agent: str) -> Optional[str]:
    """Why a Gemini or Antigravity turn is not sent (`context_files`), in plain words; None when nothing stands in the
    way."""
    hits = context_files(agent)
    if not hits:
        return None
    shown = ", ".join(hits[:4]) + (f" and {len(hits) - 4} more" if len(hits) > 4 else "")
    what = ("Antigravity would read it as its own rules, skills, hooks or plugins" if agent == "agy"
            else "Gemini CLI would read it as standing instructions, and has no setting that turns that off")
    return (f"Not sent: {shown} is where {what}. A chat can write there (every chat can write in the work folder), so one "
            "chat could steer every later one this way. Remove or rename it, then send again.")


async def _context_refusal_async(agent: str) -> Optional[str]:
    """`_context_refusal`, in a thread: a large work folder takes a while to walk, and the server goes on meanwhile."""
    return await asyncio.to_thread(_context_refusal, agent)


def _gemini_mcp_servers(cid: str, self_url: str) -> List[Dict[str, Any]]:
    return [{
        "name": "atlas",
        "command": sys.executable,
        "args": ["-I", str(MCP_SCRIPT)],
        "env": [
            {"name": "THEODOLITE_URL", "value": self_url},
            {"name": "THEODOLITE_CHAT", "value": cid},
        ],
    }]


def _gemini_tool_info(upd: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Extract `(tool_call_id, shown_name, input_dict)` from an ACP `tool_call` or `tool_call_update` payload."""
    tcid = str(upd.get("toolCallId") or "")
    raw = str(upd.get("name") or (tcid.split("__", 1)[0] if "__" in tcid else upd.get("title") or "tool"))
    if raw.startswith(GEMINI_MCP_PREFIX):
        name = raw[len(GEMINI_MCP_PREFIX):]
    else:
        name = GEMINI_TOOL_NAMES.get(raw, raw)
    if isinstance(upd.get("rawInput"), dict):
        inp: Dict[str, Any] = dict(upd["rawInput"])
    elif isinstance(upd.get("args"), dict):
        inp = dict(upd["args"])
    else:
        inp = {}
    locs = upd.get("locations") or []
    first_path = next((x.get("path") for x in locs if isinstance(x, dict) and x.get("path")), None)
    if name in ("Read", "Write", "Edit", "List") and "file_path" not in inp:
        cand = inp.get("TargetFile") or inp.get("Path") or inp.get("filePath") or first_path
        if isinstance(cand, str) and cand:
            inp["file_path"] = cand
    elif not inp and name == "Bash" and upd.get("title"):
        inp["command"] = str(upd["title"])
    elif not inp and upd.get("title") and not str(upd["title"]).endswith("(atlas MCP Server)"):
        inp["title"] = str(upd["title"])
    return tcid, name, inp


def _path_in_work(path: Any) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    try:
        target = Path(os.path.realpath(path))
        root = Path(os.path.realpath(str(work_dir())))
        return target == root or root in target.parents
    except OSError:
        return False


# the atlas's tools as Antigravity names them (`mcp_<server>_<tool>`), the only MCP tools the folder reach approves
AGY_ATLAS_TOOLS = frozenset(GEMINI_MCP_PREFIX + t["name"] for t in st.TOOLS)


def _agy_permission_outcome(prm: Dict[str, Any]) -> Dict[str, Any]:
    """Decide an ACP `session/request_permission` prompt from Antigravity CLI (`agy acp`).
    At `home` reach all tools are approved; at `folder` reach only the atlas's own tools, by name (`AGY_ATLAS_TOOLS`),
    and file reads/writes whose resolved target lies inside `work_dir()` are approved, and everything else (shell,
    browser, other MCP servers, external paths) is rejected."""
    opts = prm.get("options") or []
    allow_id = next((o.get("optionId") for o in opts if isinstance(o, dict) and o.get("kind") == "allow_once"), None)
    reject_id = next((o.get("optionId") for o in opts if isinstance(o, dict) and o.get("kind") == "reject_once"), None)
    if reach() == "home":
        return {"outcome": "selected", "optionId": allow_id} if allow_id else {"outcome": "cancelled"}
    tc = prm.get("toolCall") or {}
    raw = str(tc.get("name") or "")
    args = tc.get("args") if isinstance(tc.get("args"), dict) else (tc.get("rawInput") if isinstance(tc.get("rawInput"), dict) else {})
    locs = tc.get("locations") or []
    first_loc = next((x.get("path") for x in locs if isinstance(x, dict) and x.get("path")), None)
    allowed = False
    if raw in AGY_ATLAS_TOOLS:
        allowed = True
    elif raw in AGY_WRITE_TOOLS:
        target = args.get("TargetFile") or args.get("Path") or args.get("file_path") or first_loc
        allowed = _path_in_work(target)
    elif raw in AGY_READ_TOOLS:
        target = args.get("Path") or args.get("DirectoryPath") or args.get("SearchPath") or args.get("SearchDirectory") or first_loc
        allowed = _path_in_work(target)
    if allowed and allow_id:
        return {"outcome": "selected", "optionId": allow_id}
    return {"outcome": "selected", "optionId": reject_id} if reject_id else {"outcome": "cancelled"}


def _gemini_tool_output(upd: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in upd.get("content") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "content":
            inner = item.get("content")
            if isinstance(inner, dict) and inner.get("text") is not None:
                parts.append(str(inner["text"]))
        elif kind == "diff":
            path = item.get("path") or ""
            parts.append(f"updated {path}" if path else "updated file")
    if parts:
        return "\n".join(parts)
    if upd.get("rawOutput") is not None:
        return str(upd["rawOutput"])
    return str(upd.get("status") or "")


# Antigravity's `schedule` tool sets a timer (or a recurring one) that starts a turn of its own when it fires, with no
# message from the person; it asks no permission (live, agy 1.5.14, folder reach, 22 November 2026: the timer fired a
# minute later and the model answered its prompt)
AGY_SELF_STARTERS = frozenset({"schedule"})
AGY_SCHEDULE_WORDS = ("Antigravity set itself a timer (its schedule tool, which asks no one), to act later with no message from "
                      "you, so its process was stopped before it could, with anything it had left running. Send a message to go "
                      "on, and ask it not to schedule anything.")
AGY_OWN_WORK_WORDS = "Antigravity began work of its own, with no message from you, so its process was stopped. Send a message to go on."
# how long after a turn ends what Antigravity still sends is taken as that turn's (a stopped reply's last words)
AGY_IDLE_GRACE = 5.0


def _agy_own_work(c: Dict[str, Any], kind: Any, upd: Dict[str, Any]) -> Optional[str]:
    """Why an Antigravity chat is stopped at this update, or None: a call of a tool that sets it to act later on its own
    (`AGY_SELF_STARTERS`), at either reach; or, at the folder reach, the start of work (words, thoughts, a tool call)
    while no message of the person's is in flight, after the last turn's grace (`AGY_IDLE_GRACE`). At the home reach a
    turn it begins itself is let be, as Claude Code's is (a command it left running has finished, say): it has a shell
    there that never asks, and ending its process would end what it left running."""
    if kind in ("tool_call", "tool_call_update") and _gemini_tool_info(upd)[1] in AGY_SELF_STARTERS:
        return AGY_SCHEDULE_WORDS
    if (c.get("reach") == "folder" and kind in ("agent_message_chunk", "agent_thought_chunk", "tool_call") and not c["busy"]
            and time.time() - float(c.get("idle_since") or 0) > AGY_IDLE_GRACE):
        return AGY_OWN_WORK_WORDS
    return None


async def _end_this(c: Dict[str, Any], proc: Any) -> None:
    """Ends the chat's process if it is still `proc` (not one started since)."""
    if c.get("proc") is proc:
        await _end_chat(c)


async def _pump_gemini(c: Dict[str, Any], proc: Any) -> None:
    """An ACP process's output (Gemini CLI or Antigravity) turned into chat events, until it exits. Its stderr is read
    as it comes, its last part kept for the error when it exits, so that a long chat cannot stall on a full pipe. An
    Antigravity chat that acts on its own is stopped here (`_agy_own_work`)."""
    rpc = c["rpc"]
    tail = bytearray()

    async def drain() -> None:  # read in pieces, not lines: a line longer than the reader holds cannot stop it
        try:
            while chunk := await proc.stderr.read(1 << 16):
                tail.extend(chunk)
                del tail[: max(0, len(tail) - (1 << 16))]
        except OSError:
            pass

    drainer = asyncio.create_task(drain()) if proc.stderr is not None else None
    async for line in proc.stdout:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if "id" in m and ("result" in m or "error" in m) and "method" not in m:
            fut = rpc["pending"].pop(m["id"], None)
            if fut and not fut.done():
                fut.set_result(m)
            continue
        meth, prm = m.get("method"), m.get("params") or {}
        if "id" in m and meth:
            if c["agent"] == "agy" and meth == "session/request_permission":
                outcome = _agy_permission_outcome(prm)
            else:
                # Unexpected agent->client RPC (e.g. a permission prompt outside YOLO): decline cleanly
                outcome = {"outcome": "cancelled"}
            await _write(proc, {"jsonrpc": "2.0", "id": m["id"], "result": {"outcome": outcome}})
            continue
        if meth != "session/update" or c.get("loading") or c.get("halted") is proc:
            continue
        upd = prm.get("update") or {}
        kind = upd.get("sessionUpdate")
        why = _agy_own_work(c, kind, upd) if c["agent"] == "agy" else None
        if why:  # it is stopped before it goes on: what it sends from here is not shown
            c["halted"] = proc
            _emit(c, {"t": "error", "error": why})
            if not c["busy"]:  # the page takes work it did not ask for as a turn, and waits for its end
                _emit(c, {"t": "done"})
                c["turn"] = []
            c["halt"] = asyncio.create_task(_end_this(c, proc))
            continue
        if kind == "agent_thought_chunk":
            txt = str(((upd.get("content") or {}).get("text")) or "")
            if txt.startswith("****\n"):
                txt = txt[5:]
            if txt:
                _emit(c, {"t": "thinking", "text": txt})
        elif kind == "agent_message_chunk":
            # Gemini CLI marks the first chunk after it retried a reply cut off mid-stream (`_meta` retry and reset):
            # what came of that reply before goes (a `reset` event), so it is not shown twice
            if (upd.get("_meta") or {}).get("reset") is True:
                _emit(c, {"t": "reset"})
            txt = str(((upd.get("content") or {}).get("text")) or "")
            if txt:
                _emit(c, {"t": "text", "text": txt})
        elif kind == "tool_call":
            tcid, name, inp = _gemini_tool_info(upd)
            seen = c.setdefault("seen_tools", set())
            if tcid not in seen:
                seen.add(tcid)
                _emit(c, {"t": "tool", "id": tcid, "name": name, "input": inp})
        elif kind == "tool_call_update":
            status = upd.get("status")
            if status in ("completed", "failed"):
                tcid, name, inp = _gemini_tool_info(upd)
                seen = c.setdefault("seen_tools", set())
                if tcid not in seen:
                    seen.add(tcid)
                    _emit(c, {"t": "tool", "id": tcid, "name": name, "input": inp})
                _emit(c, {"t": "tool_result", "id": tcid, "error": status == "failed", "output": _cut(_gemini_tool_output(upd))})

    if drainer is not None:
        try:
            await asyncio.wait_for(drainer, 5)
        except asyncio.TimeoutError:
            drainer.cancel()
    code = await proc.wait()
    exit_msg = tail.decode(errors="replace").strip()[-600:] or f"{c['agent']} exited {code}"
    for fut in list(rpc["pending"].values()):
        if not fut.done():
            fut.set_exception(RuntimeError(exit_msg))
    rpc["pending"].clear()
    if c.get("proc") is proc:
        c["proc"] = None
        _mark_stale(c, False)
        if c["busy"] and c.get("task") is None:
            c["busy"] = False
            _emit(c, {"t": "error", "error": exit_msg})
            _emit(c, {"t": "done"})
            c["turn"] = []


# how long one ACP prompt (Gemini CLI or Antigravity) may run before it is given up
GEMINI_PROMPT_TIMEOUT = 3600
LOAD_FAILED_WORDS = ("This chat's conversation could not be loaded into {name} ({why}), so it would go on with no memory of the "
                     "turns shown here. Start a new chat.")


async def _load_or_new(c: Dict[str, Any], key: str, name: str, params: Dict[str, Any]) -> str:
    """The chat's ACP session: the one saved for it (`key` in chats.json), loaded, its replay not shown; else a new one,
    saved. A saved one that does not load is not replaced by a new one, which would know none of the turns the page
    shows: the chat is refused (LOAD_FAILED_WORDS)."""
    saved = _meta().get(c["session"], {}).get(key)
    if saved:
        c["loading"] = True
        try:
            await _rpc(c, "session/load", {"sessionId": saved, **params})
        except Exception as exc:
            raise RuntimeError(LOAD_FAILED_WORDS.format(name=name, why=str(exc)[:200])) from None
        finally:
            c["loading"] = False
        return str(saved)
    r = await _rpc(c, "session/new", params)
    sid = str(r.get("sessionId") or "")
    if not sid:
        raise RuntimeError(f"{name} session/new returned no sessionId")
    _meta_set(c["session"], **{key: sid})
    return sid


async def _ensure_gemini(c: Dict[str, Any], self_url: str) -> Any:
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    if not shutil.which(_bin("gemini")):
        raise RuntimeError("Gemini CLI (the `gemini` command) is not installed where the server runs.")
    files = _agent_files_stamp()
    ghome, ro_cwd, policy_path = _gemini_home(c["session"])
    pfile = prompt_file("gemini", self_url)
    env = child_env(
        "gemini",
        GEMINI_CLI_HOME=str(ghome),
        GEMINI_CLI_TRUST_WORKSPACE="true",
        GEMINI_SYSTEM_MD=str(pfile),
        GEMINI_TELEMETRY_ENABLED="false",
        TMPDIR=str(chat_tmp(c["session"])),
    )
    proc = c["proc"] = await asyncio.create_subprocess_exec(
        *_gemini_cmd(policy_path),
        cwd=str(ro_cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 24,
        start_new_session=True,
    )
    _started(c, files)
    c["rpc"], c["seen_tools"], c["loading"] = {"next": 0, "pending": {}}, set(), False
    c["pump"] = asyncio.create_task(_pump_gemini(c, proc))
    try:
        await _rpc(c, "initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "theodolite", "version": "1"},
        })
        params = {"cwd": str(ro_cwd), "mcpServers": _gemini_mcp_servers(c["session"], self_url)}
        c["gemini_session"] = await _load_or_new(c, "gemini_session", "Gemini CLI", params)
    except Exception:
        await _end_chat(c)
        raise
    return proc


def _gemini_prompt_parts(msg: str, attachments: List[str]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": msg}]
    for fid in attachments:
        meta, raw = _upload(fid)
        if meta is None:
            continue
        kind = meta["type"]
        if kind in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            parts.append({"type": "image", "mimeType": kind, "data": base64.b64encode(raw).decode()})
        elif kind == "application/pdf" or meta["name"].lower().endswith(".pdf"):
            parts.append({"type": "text", "text": f"[a PDF, {meta['name']}, was attached; Gemini CLI ACP takes text and images]"})
        else:
            parts.append({"type": "text", "text": _file_text(meta["name"], raw)})
    return parts


def _not_sent(c: Dict[str, Any], queued: List[List[Dict[str, Any]]], why: str) -> None:
    """Messages sent during a turn that never reached the scientist, said so in the thread and the replay log (an
    error event), each by its first words."""
    if not queued:
        return
    said = "; ".join("“" + _cut(_shown_text(str((p[0] if p else {}).get("text", ""))).strip(), 60) + "”" for p in queued)
    n = len(queued)
    _emit(c, {"t": "error", "error": f"Not sent: {'this message' if n == 1 else f'these {n} messages'}, sent while it worked, "
                                     f"did not reach the scientist ({said}). {why}"})


async def _gemini_turn(c: Dict[str, Any], first_parts: List[Dict[str, Any]]) -> None:
    """Runs ACP `session/prompt` calls (Gemini CLI or Antigravity), one for the message that began the turn and one for
    each sent while it ran (`c['pending']`), in the order sent, until none is left. A stop (POST /interrupt) ends the
    prompt in flight, and only that one: the messages sent during it go in after it, before any sent later. A prompt
    the process answers with an error (a rate limit, say) is said to have failed, and the next goes in. One that cannot
    go in (a context file in the way, the process gone or stopped for what it did, a reply given up after
    `GEMINI_PROMPT_TIMEOUT`) is said to be not sent (`_not_sent`)."""
    queue: List[List[Dict[str, Any]]] = [first_parts]
    try:
        while queue or c["pending"]:
            parts = queue.pop(0) if queue else c["pending"].pop(0)
            why = await _context_refusal_async(c["agent"])
            if why:
                _not_sent(c, [parts, *c["pending"]], why)
                c["pending"].clear()
                break
            c["seen_tools"] = set()
            c["cancel"].clear()  # a stop ends the prompt in flight: this one, from here
            running = c.get("proc")
            try:
                await _rpc(c, "session/prompt", {"sessionId": c["gemini_session"], "prompt": parts}, timeout=GEMINI_PROMPT_TIMEOUT)
            except Exception as exc:
                proc = c.get("proc")
                if running is not None and c.get("halted") is running:  # stopped for what it did (`_agy_own_work`), said
                    why = "The scientist was stopped first; send them again."
                    _not_sent(c, list(c["pending"]), why)
                    c["pending"].clear()
                    break
                if isinstance(exc, TimeoutError):  # given up, and stopped: nothing more goes in behind it
                    _emit(c, {"t": "error", "error": f"The reply did not end within {GEMINI_PROMPT_TIMEOUT // 60} minutes, and was stopped."})
                    try:
                        await _write(proc, {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": c["gemini_session"]}})
                    except Exception:  # noqa: BLE001 - the process is gone
                        pass
                elif not c["cancel"].is_set():
                    _emit(c, {"t": "error", "error": str(exc)[:600]})
                if isinstance(exc, TimeoutError):
                    why = f"The reply before them did not end within {GEMINI_PROMPT_TIMEOUT // 60} minutes; send them again."
                elif proc is None or proc.returncode is not None or isinstance(exc, OSError):
                    why = "The scientist's process ended first; send them again."
                else:  # the process answered this prompt with an error (a rate limit, say): the next one goes in
                    continue
                _not_sent(c, list(c["pending"]), why)
                c["pending"].clear()
                break
    finally:
        c["busy"], c["task"], c["seen_tools"], c["idle_since"] = False, None, set(), time.time()
        c["cancel"].clear()
        _emit(c, {"t": "done"})
        c["turn"] = []


def _gemini_session_file(sid: Optional[str]) -> Optional[Path]:
    """Locate Gemini CLI's stored JSONL session file for `sid` inside `<scientist folder>/gemini-home`."""
    if not sid:
        return None
    root = scientist_dir() / "gemini-home" / ".gemini" / "tmp"
    if not root.exists():
        return None
    for p in root.glob("*/chats/session-*.jsonl"):
        try:
            with open(p, "rb") as f:
                first = f.readline()
            rec = json.loads(first) if first.strip() else None
            if isinstance(rec, dict) and rec.get("sessionId") == sid:
                return p
        except (OSError, ValueError):
            continue
    return None


def _gemini_messages(path: Any) -> List[Dict[str, Any]]:
    """Gemini CLI 0.70.0's session file read back as its own loader reads it (chatRecordingService): a message record
    (an `id`, no `$patch`) sets that message, where it first stood; `{"$rewindTo": id}` drops that message and every one
    after it (all of them when it is not there); `$patch` changes (`updates`), drops (`removeIds`) and reorders
    (`orderIds`) them; `$set.messages`, and a header's `messages`, replace them. The messages, in order."""
    msgs: Dict[str, Dict[str, Any]] = {}
    valid = lambda m: isinstance(m, dict) and isinstance(m.get("id"), str) and "$patch" not in m  # noqa: E731
    for _, r in _jsonl(path):
        if isinstance(r.get("$rewindTo"), str):
            ids = list(msgs)
            for i in (ids[ids.index(r["$rewindTo"]):] if r["$rewindTo"] in msgs else ids):
                del msgs[i]
        elif isinstance(r.get("$patch"), dict):
            p = r["$patch"]
            for u in ([p] if isinstance(p.get("id"), str) else []) + [u for u in p.get("updates") or [] if isinstance(u, dict)]:
                if u.get("id") in msgs and "content" in u:
                    msgs[u["id"]] = {**msgs[u["id"]], "content": u["content"]}
            for i in p.get("removeIds") or []:
                if isinstance(i, str):
                    msgs.pop(i, None)
            if isinstance(p.get("orderIds"), list):
                order = [i for i in p["orderIds"] if isinstance(i, str) and i in msgs]
                msgs = {i: msgs[i] for i in [*(i for i in msgs if i not in set(order)), *order]}
        elif valid(r):
            msgs[r["id"]] = r
        elif isinstance(r.get("$set"), dict):
            if isinstance(r["$set"].get("messages"), list):
                msgs = {m["id"]: m for m in r["$set"]["messages"] if valid(m)}
        elif isinstance(r.get("sessionId"), str) and isinstance(r.get("messages"), list):
            msgs.update({m["id"]: m for m in r["messages"] if valid(m)})
    return list(msgs.values())


def _gemini_person_text(m: Dict[str, Any]) -> Optional[str]:
    """What the person typed, of a message in Gemini's session file (as the page shows it); None for anything else.
    Gemini 0.70.0 records three kinds of "user" message: the person's, each tool round's functionResponse, and the
    <session_context> it opens a session with."""
    if m.get("type") != "user":
        return None
    content = m.get("content")
    parts = [{"text": content}] if isinstance(content, str) else [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []
    if any("functionResponse" in p for p in parts):
        return None
    texts = [p["text"] for p in parts if isinstance(p.get("text"), str)]
    if not texts or "".join(texts).lstrip().startswith(("<session_context>", "<hook_context>")):
        return None
    return _shown_text(texts[0])


GEMINI_REWIND_UNSURE = ("This chat cannot be taken back safely: the same words were sent more than once and Gemini no longer "
                        "holds one of them (a stopped reply can make it drop the message it was answering), so which of its "
                        "messages is the last one shown here cannot be told. Start a new chat to ask again.")


def _rewind_gemini_session(cid: str, sid: Optional[str], said: List[str]) -> Optional[str]:
    """Takes Gemini CLI's stored conversation back to before the person's last message, as the replay log is taken
    back (`said`: the person's messages there, in order; the last one goes): by appending `{"$rewindTo": <its id>}`,
    which Gemini's own loader honours, after a backup; the file goes, and `gemini_session` with it, when that message was
    the person's first. Gemini may not hold that message at all (a stopped reply can make it drop the message it was
    answering, and what it had answered): then nothing is cut. When which of its messages that is cannot be told (the
    same words sent more than once, and Gemini holding fewer of them than the log), nothing is cut and the reason is
    returned, for the rewind to be refused; else None."""
    path = _gemini_session_file(sid)
    if path is None or not said:
        return None
    persons = [(m, t.strip()) for m in _gemini_messages(path) if (t := _gemini_person_text(m)) is not None]
    last = said[-1].strip()
    if not persons or persons[-1][1] != last:
        return None
    if sum(t == last for _, t in persons) != sum(s.strip() == last for s in said):
        return GEMINI_REWIND_UNSURE
    shutil.copy(path, f"{path}.bak-{int(time.time())}")
    if len(persons) == 1:
        try:
            os.remove(path)
        except OSError:
            pass
        _meta_set(cid, gemini_session=None)
        return None
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":  # a last line cut off mid-write stays a line of its own, which the loader skips
                f.write(b"\n")
        f.write((json.dumps({"$rewindTo": persons[-1][0]["id"]}) + "\n").encode("utf-8"))
    return None


# --------------------------------------------------------------------------------------------- Antigravity CLI (ACP)

AGY_NO_REWIND = ("An Antigravity chat cannot be taken back: Antigravity gives no way to drop its last turn, so it would still "
                 "have in mind what the page takes away. Start a new chat to ask again.")


def agy_gemini_dir() -> Path:
    """Antigravity's own data folder for the scientist (`--gemini-dir`; 0700): its settings, conversations and logins
    are kept there, not in the person's ~/.gemini, so their own plugins, skills, rules and subagents do not load. It is
    absolute: agy 1.5.14 takes a relative one for ~/.gemini, saying so only in its log."""
    d = (scientist_dir() / "agy-gemini").resolve()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    if not d.is_absolute():
        raise RuntimeError(f"Antigravity's own folder must be an absolute path: {d}")
    return d


def _agy_cmd(log_file: Path) -> List[str]:
    """Antigravity's command line: its own data folder, no telemetry, none of its default customizations, its log in
    the chat's own temporary folder; at the folder reach in its sandbox, at the home reach approving everything. Its
    workspace is its empty read-only cwd and the work folder, never the home folder (a workspace's `.agents`, GEMINI.md
    and AGENTS.md are its rules: `context_files`)."""
    cmd = [
        _bin("agy"), "acp",
        "--stdio",
        f"--gemini-dir={agy_gemini_dir()}",
        "--disable-telemetry",
        "--include-default-customizations=false",
        "--log-file", str(log_file),
    ]
    if reach() == "home":
        cmd += ["--dangerously-skip-permissions", "--add-dir", str(work_dir())]
    else:
        cmd += ["--sandbox", "--add-dir", str(work_dir())]
    return cmd


async def _agy_said_its_folder(log_file: Path, gdir: Path, wait: float = 5.0) -> bool:
    """Whether Antigravity's log says it keeps its files in `gdir` (agy 1.5.14 writes "ACP app data directory: <dir>"
    as it takes up the first session asked of it)."""
    want = f"ACP app data directory: {gdir / 'antigravity-cli'}"
    end = time.time() + wait
    while True:
        try:
            if want in log_file.read_text(encoding="utf-8", errors="replace"):
                return (gdir / "antigravity-cli").is_dir()
        except OSError:
            pass
        if time.time() >= end:
            return False
        await asyncio.sleep(0.1)


async def _ensure_agy(c: Dict[str, Any], self_url: str) -> Any:
    proc = c.get("proc")
    if proc is not None and proc.returncode is None:
        return proc
    if not shutil.which(_bin("agy")):
        raise RuntimeError("Antigravity CLI (the `agy` command) is not installed where the server runs.")
    files = _agent_files_stamp()
    ro_cwd = scientist_dir() / "agy-cwd"
    _reset_ro_cwd(ro_cwd)
    gdir = agy_gemini_dir()
    tmp = chat_tmp(c["session"])
    log_file = tmp / "agy-acp.log"
    try:
        log_file.unlink()
    except OSError:
        pass
    # at the folder reach its XDG folders are its own too (it names $XDG_CONFIG_HOME/antigravity/plugin-secrets.json); at
    # the home reach they are the person's, as its shell's commands would look for their files there
    xdg_names = (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state"))
    xdg = ({k: os.environ[k] for k, _ in xdg_names if k in os.environ} if reach() == "home"
           else {k: str(gdir / "xdg" / v) for k, v in xdg_names})
    env = child_env("agy", TMPDIR=str(tmp), **xdg)
    proc = c["proc"] = await asyncio.create_subprocess_exec(
        *_agy_cmd(log_file),
        cwd=str(ro_cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 24,
        start_new_session=True,
    )
    _started(c, files)
    c["rpc"], c["seen_tools"], c["loading"], c["idle_since"] = {"next": 0, "pending": {}}, set(), False, time.time()
    c["pump"] = asyncio.create_task(_pump_gemini(c, proc))
    try:
        init = await _rpc(c, "initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "theodolite", "version": "1"},
        })
        version = str((init.get("agentInfo") or {}).get("version") or "unknown")
        if version != AGY_VERSION:
            raise RuntimeError(f"Antigravity CLI {version} is installed; the scientist chat was checked on {AGY_VERSION} only "
                               "(that it keeps its files in a folder of the chat's own, how it asks before it acts), so it is "
                               "not started on another.")
        sys_meta = {"antigravity": {"systemPrompt": {"append": system_prompt("agy", self_url)}}}
        params = {"cwd": str(ro_cwd), "mcpServers": _gemini_mcp_servers(c["session"], self_url), "_meta": sys_meta}
        sid = await _load_or_new(c, "agy_session", "Antigravity", params)
        # agy 1.5.14 says where it keeps its files only once a session is asked for (live, 22 November 2026: nothing
        # after `initialize` alone), so this is checked after it
        if not await _agy_said_its_folder(log_file, gdir):
            raise RuntimeError(f"Antigravity did not say it keeps its files in the chat's own folder ({gdir}), so the chat is "
                               "stopped: its conversations could otherwise land among your own.")
        model = os.environ.get("THEODOLITE_AGY_MODEL")
        if model:
            await _rpc(c, "session/set_model", {"sessionId": sid, "modelId": model})
        c["gemini_session"] = sid
    except Exception:
        await _end_chat(c)
        raise
    return proc


# ------------------------------------------------------------------------------------- OpenAI-compatible endpoints


class InProcessApi:
    """The atlas's routes reached in this process (httpx over ASGI), as scientist_tools.call reaches them: the same
    routes, checks and GPU lock as a page's request."""

    def __init__(self, app: Any) -> None:
        import httpx

        # a route's own failure comes back as its 500, for the model to read, not as an exception here; no socket carries
        # these calls, so the client names itself rather than an address the server's guard would look for a socket of
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False, client=("in-process", 0))
        self.client = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", timeout=None)

    @staticmethod
    def _json(r: Any) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise st.ToolError(f"{r.status_code}: {detail if isinstance(detail, str) else json.dumps(detail)}"[:2000])
        return r.json()

    async def get(self, path: str, **query: Any) -> Any:
        return self._json(await self.client.get(path, params={k: v for k, v in query.items() if v is not None}))

    async def post(self, path: str, body: Optional[Dict[str, Any]]) -> Any:
        return self._json(await self.client.post(path, json=body or {}))

    async def close(self) -> None:
        await self.client.aclose()


class EndpointError(Exception):
    pass


async def ask_once(profile: Optional[str], chat: Optional[str], model: Optional[str], system: str, user: str) -> Tuple[str, str, str]:
    """One question to an endpoint in a fresh conversation, no tools and no history: (the endpoint's name, the model,
    its reply's text). `profile` names the endpoint, else it is the one `chat` uses; `model`, else the chat's, else
    the endpoint's own. For a reader that must see nothing but what it is given (atlas.py's blind test)."""
    import httpx

    meta = _meta().get(chat or "", {}) if chat else {}
    name = profile or meta.get("profile")
    known = profiles()
    if not name or name not in known:
        raise EndpointError("Name the endpoint that is to read blind (profile): one of " + ", ".join(sorted(known)) + ".")
    p = known[name]
    model = model or meta.get("model") or p.get("model")
    if not model:
        raise EndpointError(f"Pick a model for the endpoint {name!r}.")
    try:
        headers = _endpoint_headers(p)
    except st.ToolError as exc:
        raise EndpointError(str(exc)) from None
    url = p["base_url"].rstrip("/") + "/chat/completions"
    body = {"model": model, **(p.get("params") or {}), "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)) as client:
            r = await client.post(url, json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise EndpointError(f"{urlparse(url).hostname} could not be reached: {exc}") from None
    if r.status_code >= 400:
        raise EndpointError(f"{urlparse(url).hostname} answered {r.status_code}: {r.text[:800]}")
    try:
        msg = r.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise EndpointError(f"{urlparse(url).hostname} answered with no message: {r.text[:400]}") from None
    return name, str(model), str(msg.get("content") or "")


def _messages_path(cid: str) -> Path:
    return logs_dir() / f"{cid}.messages.json"


def _load_messages(cid: str) -> List[Dict[str, Any]]:
    try:
        m = json.loads(_messages_path(cid).read_text(encoding="utf-8"))
        return m if isinstance(m, list) else []
    except (OSError, ValueError):
        return []


def _save_messages(cid: str, messages: List[Dict[str, Any]]) -> None:
    path = _messages_path(cid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _openai_content(msg: str, attachments: List[str]) -> Any:
    """A message the person sent, as an OpenAI-compatible user message's content: the text, images as data URLs, other
    files as text; a PDF is named, not sent (such an endpoint takes none)."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": msg}]
    for fid in attachments:
        meta, raw = _upload(fid)
        if meta is None:
            continue
        if meta["type"] in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            parts.append({"type": "image_url", "image_url": {"url": f"data:{meta['type']};base64,{base64.b64encode(raw).decode()}"}})
        elif meta["type"] == "application/pdf" or meta["name"].lower().endswith(".pdf"):
            parts.append({"type": "text", "text": f"[a PDF, {meta['name']}, was attached; this endpoint is not sent PDFs]"})
        else:
            parts.append({"type": "text", "text": _file_text(meta["name"], raw)})
    return msg if len(parts) == 1 else parts


async def _complete(c: Dict[str, Any], client: Any, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    """One completion: the reply streamed to the page as it comes (text, and a reasoning model's thinking), returned as
    an assistant message with its tool calls. Without streaming (the profile's `stream: false`, or a server that
    refuses it), one reply."""
    host = urlparse(url).hostname
    if body.get("stream"):
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", errors="replace")
                if resp.status_code in (400, 422) and "stream" in text.lower():
                    return await _complete(c, client, url, headers, {**body, "stream": False})
                raise EndpointError(f"{host} answered {resp.status_code}: {text[:800]}")
            content: List[str] = []
            calls: Dict[int, Dict[str, str]] = {}
            async for line in resp.aiter_lines():
                if c["cancel"].is_set():
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise EndpointError(f"{host}: {json.dumps(chunk['error'])[:800]}")
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content.append(d["content"])
                        _emit(c, {"t": "text", "text": d["content"]})
                    thought = d.get("reasoning") or d.get("reasoning_content")
                    if thought:
                        _emit(c, {"t": "thinking", "text": thought})
                    for tc in d.get("tool_calls") or []:
                        e = calls.setdefault(int(tc.get("index", len(calls))), {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            e["id"] = tc["id"]
                        f = tc.get("function") or {}
                        if f.get("name"):  # whole in one piece on most servers, in pieces on some, sent again on others
                            e["name"] = f["name"] if f["name"].startswith(e["name"]) else e["name"] + f["name"]
                        if f.get("arguments"):
                            e["arguments"] += f["arguments"]
            msg: Dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
            if calls:
                msg["tool_calls"] = [{"id": e["id"] or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                                      "function": {"name": e["name"], "arguments": e["arguments"] or "{}"}} for _, e in sorted(calls.items())]
            return msg
    resp = await client.post(url, json=body, headers=headers)
    if resp.status_code >= 400:
        raise EndpointError(f"{host} answered {resp.status_code}: {resp.text[:800]}")
    try:
        m = (resp.json().get("choices") or [{}])[0].get("message") or {}
    except ValueError:
        raise EndpointError(f"{host} answered something other than JSON: {resp.text[:300]}") from None
    thought = m.get("reasoning") or m.get("reasoning_content")
    if thought:
        _emit(c, {"t": "thinking", "text": thought})
    if m.get("content"):
        _emit(c, {"t": "text", "text": m["content"]})
    msg = {"role": "assistant", "content": m.get("content") or None}
    if m.get("tool_calls"):
        msg["tool_calls"] = [{"id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                              "function": {"name": (tc.get("function") or {}).get("name", ""),
                                           "arguments": (tc.get("function") or {}).get("arguments") or "{}"}} for tc in m["tool_calls"]]
    return msg


async def _run_tool(c: Dict[str, Any], api: Any, tc: Dict[str, Any]) -> str:
    name = tc["function"]["name"]
    raw = tc["function"].get("arguments") or "{}"
    try:
        args = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        args, text, err = raw, f"error: your arguments were not JSON ({exc}); send them again as a JSON object", True
    else:
        try:
            if not isinstance(args, dict):
                raise st.ToolError("A tool's arguments are a JSON object.")
            text, err = st.as_text(await st.call(name, args, api, c["session"]), cut=TOOL_CUT), False
        except st.ToolError as exc:
            text, err = f"error: {exc}", True
        except Exception as exc:  # the tool's failure goes back to the model, in its own words
            logger.exception("scientist tool %s failed", name)
            text, err = f"error: {type(exc).__name__}: {exc}", True
    _emit(c, {"t": "tool_result", "id": tc["id"], "error": err, "output": _cut(text)})
    return text


async def _openai_turn(c: Dict[str, Any], self_url: str) -> None:
    """The tool loop of one turn: completions until the model answers without a tool call, each call run through the
    atlas in this process; a message sent meanwhile joins between tool calls; an interrupt stops it where it is."""
    import httpx

    cid = c["session"]
    meta = _meta().get(cid, {})
    api = InProcessApi(_app)
    try:
        p = profiles().get(meta.get("profile", ""))
        if p is None:
            raise EndpointError(f"The endpoint {meta.get('profile')!r} is not configured on this server.")
        try:
            headers = _endpoint_headers(p)
        except st.ToolError as exc:
            raise EndpointError(str(exc)) from None
        url = p["base_url"].rstrip("/") + "/chat/completions"
        tools = st.openai_tools()
        messages = c.setdefault("messages", _load_messages(cid))
        rounds = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)) as client:
            while True:
                while c["pending"]:
                    messages.append(c["pending"].pop(0))
                if c["cancel"].is_set():
                    break
                rounds += 1
                if rounds > MAX_ROUNDS:
                    _emit(c, {"t": "error", "error": f"stopped after {MAX_ROUNDS} rounds of tool calls in one turn"})
                    break
                body = {"model": meta.get("model") or p.get("model"), "tools": tools, "tool_choice": "auto",
                        "stream": bool(p.get("stream", True)), **(p.get("params") or {}),
                        "messages": [{"role": "system", "content": system_prompt("openai", self_url)}] + messages}
                if not body["model"]:
                    raise EndpointError("Pick a model for this endpoint.")
                msg = await _complete(c, client, url, headers, body)
                if c["cancel"].is_set():
                    if msg.get("content"):
                        messages.append({"role": "assistant", "content": msg["content"]})
                    break
                messages.append(msg)
                _save_messages(cid, messages)
                calls = msg.get("tool_calls") or []
                if not calls:
                    if c["pending"]:
                        continue  # a message came in while it answered: it answers that too
                    break
                for i, tc in enumerate(calls):
                    if c["cancel"].is_set():
                        for rest in calls[i:]:  # every call is answered, or the endpoint refuses the conversation
                            messages.append({"role": "tool", "tool_call_id": rest["id"], "content": "not run: the person stopped the reply"})
                        break
                    try:
                        inp = json.loads(tc["function"].get("arguments") or "{}")
                    except ValueError:
                        inp = tc["function"].get("arguments")
                    _emit(c, {"t": "tool", "id": tc["id"], "name": tc["function"]["name"], "input": inp})
                    text = await _run_tool(c, api, tc)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
                    _save_messages(cid, messages)
    except EndpointError as exc:
        _emit(c, {"t": "error", "error": str(exc)})
    except Exception as exc:
        logger.exception("scientist chat %s failed", cid)
        _emit(c, {"t": "error", "error": f"{type(exc).__name__}: {exc}"[:600]})
    finally:
        await api.close()
        if "messages" in c:
            _save_messages(cid, c["messages"])
        c["busy"], c["task"] = False, None
        c["cancel"].clear()
        _emit(c, {"t": "done"})
        c["turn"] = []


# ------------------------------------------------------------------------------------------------------------ routes


class SendRequest(BaseModel):
    chat: str = Field(..., max_length=64)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE)
    agent: Optional[str] = Field(None, max_length=16)
    profile: Optional[str] = Field(None, max_length=64)
    model: Optional[str] = Field(None, max_length=200)
    layer: Optional[int] = None
    n: Optional[int] = None
    head: Optional[int] = None
    attachments: List[str] = Field(default_factory=list, max_length=16)


class ChatRequest(BaseModel):
    chat: str = Field(..., max_length=64)
    text: Optional[str] = Field(None, max_length=MAX_MESSAGE)


class UploadRequest(BaseModel):
    name: str = Field("file", max_length=200)
    type: str = Field("", max_length=100)
    data: str


@router.get("/backends")
async def backends() -> Dict[str, Any]:
    """Which backends this server can run: Antigravity CLI, Claude Code, Codex and Gemini CLI where their commands are
    installed, and every OpenAI-compatible endpoint configured (`profiles()`; each one's key only as "set" or "missing")."""
    ps = profiles()
    return {"agy": {"available": bool(shutil.which(_bin("agy"))), "model": os.environ.get("THEODOLITE_AGY_MODEL")},
            "claude": {"available": bool(shutil.which(_bin("claude"))), "model": os.environ.get("THEODOLITE_CLAUDE_MODEL")},
            "codex": {"available": bool(shutil.which(_bin("codex")))},
            "gemini": {"available": bool(shutil.which(_bin("gemini"))), "model": os.environ.get("THEODOLITE_GEMINI_MODEL")},
            "openai": [profile_card(k, v) for k, v in ps.items()], "warning": _profile_warning,
            # what each can do at this reach, and at the folder reach (what the page says it would do without "home")
            "reach": {"mode": reach(), "env": REACH_ENV, **REACH_WORDS[reach()], "folder": REACH_WORDS["folder"]},
            # where an Antigravity, Claude Code, Codex or Gemini scientist keeps its files, and the route the page opens one through
            "work": {"folder": str(work_dir()), "route": WORK_ROUTE},
            "prompt": {"files": [str(p) for p in PROMPT_FILES], "local": str(scientist_dir() / "prompt.local.md")}}


@router.get("/models")
async def models(profile: str) -> Dict[str, Any]:
    """The models an OpenAI-compatible endpoint lists (`GET <base>/models`), each with whether it says it takes tools
    (None when the endpoint does not say)."""
    import httpx

    p = profiles().get(profile)
    if p is None:
        raise HTTPException(status_code=404, detail=f"No endpoint {profile!r} is configured here.")
    try:
        headers = _endpoint_headers(p)
    except st.ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            r = await client.get(p["base_url"].rstrip("/") + "/models", headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"{urlparse(p['base_url']).hostname} did not answer: {exc}") from None
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"{urlparse(p['base_url']).hostname} answered {r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="The endpoint's model list was not JSON.") from None
    items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
    out = []
    for m in items if isinstance(items, list) else []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        sp = m.get("supported_parameters")
        out.append({"id": str(m["id"]), "name": m.get("name"), "context": m.get("context_length"),
                    "tools": ("tools" in sp) if isinstance(sp, list) else None})
    return {"profile": profile, "models": out, "default": p.get("model")}


@router.post("/send")
async def send(req: SendRequest, request: Request) -> Dict[str, Any]:
    """A message into a chat (starting its backend when it is not running). While the scientist works, the message
    reaches it between tool calls. `layer` and `n` (or `head`), counted from 0, say what the person has open: the
    scientist is told, in the pages' count. A Claude Code or Codex chat idle on instructions or tools that have changed
    since its process started is first started again on them, or marked stale (`_take_up_changes`). Events arrive on
    GET /events."""
    cid = _check_chat(req.chat)
    if req.agent is not None and req.agent not in AGENTS:
        raise HTTPException(status_code=400, detail=f"The scientist's backends are {', '.join(AGENTS)}.")
    c = _chat(cid, req.agent)
    known = _meta().get(cid, {})
    if c["agent"] == "openai" and not known.get("profile"):
        ps = profiles()
        if not req.profile or req.profile not in ps:
            raise HTTPException(status_code=400, detail=f"Pick an endpoint for this chat: {', '.join(ps)}.")
        model = req.model or ps[req.profile].get("model")
        if not model:
            raise HTTPException(status_code=400, detail="Pick a model for this endpoint.")
        _meta_set(cid, agent="openai", profile=req.profile, model=model, created=int(time.time()))
    elif not known.get("agent"):
        _meta_set(cid, agent=c["agent"], created=int(time.time()))
    msg = shown = req.message
    if req.layer is not None and req.head is not None:
        msg = f"[The person has attention head {st.cite_head(req.layer, req.head)} open in the atlas.]\n\n{msg}"
    elif req.layer is not None and req.n is not None:
        msg = f"[The person has {st.cite(req.layer, req.n)} open in the atlas.]\n\n{msg}"
    self_url = _self_url(request)
    async with c["lock"]:
        if c["agent"] == "openai":
            if _app is None:
                raise HTTPException(status_code=503, detail="The scientist chat is not bound to a server.")
            user = {"role": "user", "content": _openai_content(msg, req.attachments)}
            _log(cid, {"t": "user", "text": shown})
            _emit(c, {"t": "sent"})
            if c["busy"]:
                c["pending"].append(user)
                return {"ok": True, "queued": True}
            c.setdefault("messages", _load_messages(cid)).append(user)
            c["busy"] = True
            c["cancel"].clear()
            c["task"] = asyncio.create_task(_openai_turn(c, self_url))
            return {"ok": True}
        await _take_up_changes(c)
        if c["agent"] == "codex":
            try:
                await _ensure_codex(c, self_url)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"codex: {exc}") from None
            at = _log(cid, {"t": "user", "text": shown})
            items = [{"type": "text", "text": msg, "text_elements": []}] + [x for a in req.attachments if (x := _codex_input(a))]
            c["busy"], c["last"] = True, time.time()
            _emit(c, {"t": "sent"})
            if c.get("turn_id"):  # mid-turn: the message joins the running turn
                try:
                    await _rpc(c, "turn/steer", {"threadId": c["thread"], "expectedTurnId": c["turn_id"], "input": items})
                    return {"ok": True}
                except Exception:
                    pass  # the turn ended meanwhile: start a new one
            r = await _rpc(c, "turn/start", {"threadId": c["thread"], "input": items})
            tid = (r.get("turn") or {}).get("id")
            if c["busy"]:  # a quick turn can be over before its start is answered here: then no turn runs
                c["turn_id"] = tid
            # where in the log each turn began (its first message), for /rewind: a message steered into it comes later
            known = _meta().get(cid, {})
            _meta_set(cid, turns=known.get("turns", []) + [tid], turn_at={**(known.get("turn_at") or {}), str(tid): at})
            return {"ok": True}
        if c["agent"] in ("agy", "gemini"):
            why = await _context_refusal_async(c["agent"])  # before the process starts: Gemini reads them as a session starts
            if why:
                raise HTTPException(status_code=409, detail=why)
            try:
                if c["agent"] == "agy":
                    await _ensure_agy(c, self_url)
                else:
                    await _ensure_gemini(c, self_url)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"{c['agent']}: {exc}") from None
            _log(cid, {"t": "user", "text": shown})
            parts = _gemini_prompt_parts(msg, req.attachments)
            _emit(c, {"t": "sent"})
            if c["busy"]:
                c["pending"].append(parts)
                return {"ok": True, "queued": True}
            c["busy"], c["last"] = True, time.time()
            c["cancel"].clear()
            c["task"] = asyncio.create_task(_gemini_turn(c, parts))
            return {"ok": True}
        try:
            proc = await _ensure_claude(c, self_url)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        content = [{"type": "text", "text": msg}] + [blk for a in req.attachments if (blk := attachment_block(a))]
        c["queued"] = bool(c.get("queued") or c["busy"])  # mid-turn: it joins this turn, or runs as one of its own after it
        c["busy"], c["last"] = True, time.time()
        _emit(c, {"t": "sent"})
        await _write(proc, {"type": "user", "message": {"role": "user", "content": content}})
        return {"ok": True}


@router.post("/interrupt")
async def interrupt(req: ChatRequest) -> Dict[str, Any]:
    """Stops the chat's current reply; the chat (and, under Claude Code or Codex, its background commands) goes on. A
    message sent while it answered is not dropped: under Antigravity or Gemini CLI it goes in once the stopped reply has
    ended, before any sent later."""
    c = _chats.get(_check_chat(req.chat))
    if c is None or not c["busy"]:
        return {"ok": True, "running": False}
    if c["agent"] == "openai":
        c["cancel"].set()
        return {"ok": True, "running": True}
    proc = c.get("proc")
    if not proc or proc.returncode is not None:
        return {"ok": True, "running": False}
    if c["agent"] == "codex":
        if not c.get("turn_id"):
            return {"ok": True, "running": False}
        await _rpc(c, "turn/interrupt", {"threadId": c["thread"], "turnId": c["turn_id"]})
        return {"ok": True, "running": True}
    if c["agent"] in ("agy", "gemini"):
        # the prompt in flight is stopped; messages sent during it still go in, after it (`_gemini_turn`)
        c["cancel"].set()
        if c.get("gemini_session"):
            await _write(proc, {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": c["gemini_session"]}})
        return {"ok": True, "running": True}
    await _write(proc, {"type": "control_request", "request_id": str(uuid.uuid4()), "request": {"subtype": "interrupt"}})
    return {"ok": True, "running": True}


@router.get("/events")
async def events(chat: str) -> StreamingResponse:
    """The chat's events as NDJSON: {t: "state", busy, running, agent, stale} and the current turn's events so far, then
    each as it comes: text, thinking, tool, tool_result, ui, error, done (a turn ended), sent (a message went in), note
    (the chat was started again on changed instructions or tools), stale ({stale}: whether it runs older ones), ping."""
    c = _chat(_check_chat(chat))
    q: asyncio.Queue = asyncio.Queue()
    c["listeners"].add(q)

    async def gen():
        try:
            running = bool(c.get("proc") and c["proc"].returncode is None) or c.get("task") is not None
            state = {"t": "state", "busy": c["busy"], "running": running, "agent": c["agent"], "stale": bool(c.get("stale"))}
            yield json.dumps(state) + "\n"
            for ev in list(c["turn"]):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 20)
                except asyncio.TimeoutError:
                    yield json.dumps({"t": "ping"}) + "\n"
                    continue
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        finally:
            c["listeners"].discard(q)

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


@router.post("/ui")
async def ui(body: Dict[str, Any]) -> Dict[str, Any]:
    """A screen action from the scientist's tools (select, head, tab, camera, highlight, annotate, text, search,
    propose: a name or note proposed, which the page shows for the person to keep or discard; rename, from before),
    passed to the chat's page; layers counted from 0."""
    cid = str(body.pop("chat", ""))
    c = _chats.get(cid)
    if c is None:
        return {"ok": False, "error": "no such chat is open on this server"}
    if body.get("action") not in set(st.UI_TOOLS.values()) | {"rename", "propose"}:
        return {"ok": False, "error": f"no screen action {body.get('action')!r}"}
    _emit(c, {"t": "ui", **body})
    return {"ok": True, "note": "shown in the atlas" if c["listeners"] else "the person's page is not open; nothing was shown"}


_chat_list: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _summary(cid: str, m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A past chat's line in the list: {chat, agent, t (its last change), title (the first message), turns}."""
    agent = m.get("agent", "claude")
    path = _session_file(cid) if agent == "claude" else str(logs_dir() / f"{cid}.jsonl")
    if not path or not os.path.exists(path):
        return None
    mt = os.path.getmtime(path)
    hit = _chat_list.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    title, turns = None, 0
    for _, e in _jsonl(path):
        t = _user_text(e) if agent == "claude" else (e.get("text") if e.get("t") == "user" else None)
        if t is not None:
            turns += 1
            title = title or _shown_text(t).strip()[:120]
    out = {"chat": cid, "agent": agent, "t": int(mt), "title": title, "turns": turns,
           **({"profile": m.get("profile"), "model": m.get("model")} if agent == "openai" else {})}
    _chat_list[path] = (mt, out)
    return out


@router.get("/chats")
async def chats(limit: int = 60) -> List[Dict[str, Any]]:
    """Past chats, newest first."""
    rows = [s for cid, m in _meta().items() if (s := await asyncio.to_thread(_summary, cid, m)) and s["turns"]]
    return sorted(rows, key=lambda r: -r["t"])[: max(1, min(int(limit), 500))]


@router.get("/history")
async def history(chat: str) -> Dict[str, Any]:
    """A past chat as the events a live turn streams (text, tool, tool_result, error), with a user event per message."""
    cid = _check_chat(chat)
    m = _meta().get(cid, {})
    agent = m.get("agent", "claude")
    if agent in ("agy", "codex", "gemini", "openai"):
        path = logs_dir() / f"{cid}.jsonl"
        ev = []
        if path.exists():
            for _, e in _jsonl(path):
                if e.get("t") == "reset":  # a reply Gemini CLI retried: what came of it before goes, as on the live page
                    while ev and ev[-1].get("t") == "text":
                        ev.pop()
                elif e.get("t") in ("user", "text", "tool", "tool_result", "error"):
                    # the log keeps each streamed fragment; a reply is their run joined, as the live page shows it (a
                    # fragment boundary can fall inside a number or a citation)
                    if e["t"] == "text" and ev and ev[-1].get("t") == "text":
                        ev[-1] = {**ev[-1], "text": ev[-1].get("text", "") + e.get("text", "")}
                    else:
                        ev.append(dict(e))
        return {"chat": cid, "agent": agent, "events": ev, **({"profile": m.get("profile"), "model": m.get("model")} if agent == "openai" else {})}
    path = _session_file(cid)
    if not path:
        return {"chat": cid, "agent": agent, "events": []}
    ev, seen = [], set()
    for _, msg in _jsonl(path):
        if msg.get("isSidechain") or msg.get("isMeta"):
            continue
        t = _user_text(msg)
        if t is not None:
            ev.append({"t": "user", "text": _shown_text(t)})
        elif msg.get("type") == "assistant":
            r = _refusal(msg)
            if r and r["id"] not in seen:
                seen.add(r["id"])
                ev.append({"t": "error", "error": r["text"]})
            for blk in (msg.get("message") or {}).get("content", []):
                if blk.get("type") == "text" and blk.get("text"):
                    ev.append({"t": "text", "text": blk["text"]})
                elif blk.get("type") == "tool_use":
                    ev.append({"t": "tool", "id": blk.get("id"), "input": blk.get("input"), "name": str(blk.get("name", "")).removeprefix(MCP_PREFIX)})
        elif msg.get("type") == "user":
            ev += _tool_results(msg)
    return {"chat": cid, "agent": agent, "events": ev}


@router.post("/rewind")
async def rewind(req: ChatRequest, request: Request) -> Dict[str, Any]:
    """Cut a chat back to just before the last message the person sent that ends with `text` (any last message
    without it), so it can be sent again: the reply goes too. A Claude Code session is backed up beside it first. No
    match (already cut) is not an error. A Codex chat's last turn is reverted whole (thread/revert drops the turn), so
    a message the person sent into it while it ran goes with it: its log is cut back to where that turn began, and
    `messages` says how many of the person's messages left it (the page takes as many off its thread). A Gemini CLI
    chat's stored conversation is taken back to before that message (`_rewind_gemini_session`), or the rewind refused
    when that cannot be done safely. An Antigravity chat's is refused (409): Antigravity cannot drop a turn."""
    cid = _check_chat(req.chat)
    text = (req.text or "").strip()
    m = _meta().get(cid, {})
    agent = m.get("agent", "claude")
    c = _chats.get(cid)
    if c is not None and c["busy"]:
        raise HTTPException(status_code=409, detail="The scientist is still answering; stop it first.")
    if agent == "openai":
        msgs = c["messages"] if c is not None and "messages" in c else _load_messages(cid)
        last = max((i for i, x in enumerate(msgs) if x.get("role") == "user"), default=None)
        if last is not None:
            del msgs[last:]
            _save_messages(cid, msgs)
        _cut_log(cid)
        return {"ok": True, "cut": last is not None}
    if agent == "codex":
        turns, turn_at = m.get("turns") or [], m.get("turn_at") or {}
        at = None
        if turns and c is not None:
            try:
                await _ensure_codex(c, _self_url(request))
                await _rpc(c, "thread/revert", {"threadId": c["thread"], "beforeTurnId": turns[-1]})
            finally:
                await _end_chat(c)
            at = turn_at.get(str(turns[-1]))  # None for a turn taken before the log said where turns begin
            _meta_set(cid, turns=turns[:-1], turn_at={k: v for k, v in turn_at.items() if k != str(turns[-1])})
        return {"ok": True, "cut": bool(turns), "messages": _cut_log(cid, at)}
    if agent == "agy":  # nothing is cut, so the thread, the log and Antigravity's memory stay in step
        raise HTTPException(status_code=409, detail=AGY_NO_REWIND)
    if agent == "gemini":
        if c is not None:
            await _end_chat(c)
        log = logs_dir() / f"{cid}.jsonl"
        said = [str(e.get("text") or "") for _, e in _jsonl(log) if e.get("t") == "user"] if log.exists() else []
        if not said or (text and said[-1].strip() != text):
            # the page's last message is not the log's (a message refused before it was sent, say): nothing of the
            # conversation is the page's to take back, so nothing is cut
            return {"ok": True, "cut": False, "messages": 0}
        why = _rewind_gemini_session(cid, m.get("gemini_session"), said)  # Gemini's memory first; both or neither
        if why:
            raise HTTPException(status_code=409, detail=why)
        cut_n = _cut_log(cid)
        return {"ok": True, "cut": bool(cut_n), "messages": cut_n}
    if c is not None:  # a live process holds the conversation in memory; the next message resumes from the cut file
        await _end_chat(c)
    path = _session_file(cid)
    if not path:
        return {"ok": True, "cut": False}
    recs = _jsonl(path)
    at = None
    for pos, msg in reversed(recs):
        t = _user_text(msg)
        if t is not None and (not text or t.strip() == text or t.strip().endswith("\n\n" + text) or t.split("\n\n", 1)[-1].strip() == text):
            at = pos
            break
    if at is None:
        return {"ok": True, "cut": False}
    shutil.copy(path, f"{path}.bak-{int(time.time())}")
    if not any(_user_text(msg) is not None for pos, msg in recs if pos < at):
        os.remove(path)  # nothing the person said is left: the next turn starts the session afresh
    else:
        with open(path, "rb+") as f:  # what came before, byte for byte
            f.truncate(at)
    return {"ok": True, "cut": True}


def _cut_log(cid: str, at: Optional[int] = None) -> int:
    """Cuts a codex or openai chat's replay log back to before a message the person sent: the one whose line begins at
    byte `at` (where a Codex turn began, `turn_at`), else the last. What comes before is kept byte for byte, so where
    each earlier turn began still holds. How many of the person's messages went."""
    path = logs_dir() / f"{cid}.jsonl"
    if not path.exists():
        return 0
    users = [pos for pos, e in _jsonl(path) if e.get("t") == "user"]
    cut = at if at in users else (users[-1] if users else None)
    if cut is None:
        return 0
    with open(path, "rb+") as f:
        f.truncate(cut)
    return sum(p >= cut for p in users)


@router.post("/restart")
async def restart(req: ChatRequest) -> Dict[str, Any]:
    """Ends a chat's process group (the process, its MCP server, and any command it left running in the background in
    that group; one that started a group of its own is not reached, and whether the real Codex's commands stay in its
    group is not checked here), so that the next message starts it again, resuming the conversation on its
    instructions and tools as they now stand: what a stale chat is offered (`_take_up_changes`). Refused while the
    scientist answers, as /rewind is. A chat with no process running (an openai chat has none) is left as it is."""
    c = _chats.get(_check_chat(req.chat))
    if c is None:
        return {"ok": True, "ended": False}
    async with c["lock"]:
        if c["busy"]:
            raise HTTPException(status_code=409, detail="The scientist is still answering; stop it first.")
        proc = c.get("proc")
        ended = proc is not None and proc.returncode is None
        if ended:
            await _end_chat(c)
        _mark_stale(c, False)
    return {"ok": True, "ended": True, "note": ENDED_WORDS} if ended else {"ok": True, "ended": False}


@router.post("/upload")
async def upload(req: UploadRequest) -> Dict[str, Any]:
    """A file for the chat, as base64 ({name, type, data}), kept in the scientist folder's uploads/."""
    try:
        raw = base64.b64decode(req.data, validate=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="The file's data is not base64.") from None
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="A file can be at most 25 MB.")
    d = uploads_dir()
    d.mkdir(parents=True, exist_ok=True)
    fid = uuid.uuid4().hex[:12]
    name = os.path.basename(req.name.replace("\\", "/")) or "file"
    name = re.sub(r"[^\w.\- ]", "_", name)[:120] or "file"
    (d / f"{fid}-{name}").write_bytes(raw)
    (d / f"{fid}.json").write_text(json.dumps({"name": name, "type": req.type or "", "size": len(raw)}), encoding="utf-8")
    return {"id": fid, "name": name, "size": len(raw)}


@router.get("/work/{path:path}")
async def work_file(path: str) -> FileResponse:
    """A file in the scientist's work folder, for the person to open from the chat: a page, a picture or a text shown
    in the tab (WORK_TYPES), anything else downloaded. Only files inside the folder, links followed first, so one
    pointing out of it is not there. Each goes out under WORK_CSP (no scripts, no network, no origin), not sniffed."""
    root = work_dir().resolve()
    try:
        target = (root / path).resolve()
    except (OSError, ValueError):
        target = root
    if not path or not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="No such file in the scientist's work folder.")
    headers = {"Content-Security-Policy": WORK_CSP, "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
               "Referrer-Policy": "no-referrer"}
    kind = WORK_TYPES.get(target.suffix.lower())
    if kind is None:
        return FileResponse(str(target), media_type="application/octet-stream", filename=target.name, headers=headers)
    return FileResponse(str(target), media_type=kind, headers=headers)


@router.get("/prompt")
async def prompt(agent: str, request: Request) -> Dict[str, Any]:
    """The instructions a chat on this backend starts with, as built from their files."""
    if agent not in AGENTS:
        raise HTTPException(status_code=400, detail=f"The scientist's backends are {', '.join(AGENTS)}.")
    return {"agent": agent, "text": system_prompt(agent, _self_url(request)), "files": [str(p) for p in PROMPT_FILES]}
