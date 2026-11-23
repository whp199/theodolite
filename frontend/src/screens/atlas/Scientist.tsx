import React, { useEffect, useRef, useState } from "react";
import { Rubric } from "../../components/kit";
import { href } from "../../router";
import { cite, citeHead, exportUrl, type PastChat } from "./api";
import { ErrorLine, ProposalLine } from "./parts";
import {
  askWeb,
  attachFiles,
  explainFocus,
  loadBackends,
  newChat,
  openEvents,
  openPast,
  removePending,
  restartChat,
  rewindLast,
  send,
  setPick,
  backendPick,
  cliChoices,
  stop,
  takeName,
  togglePast,
  useScientist,
  GEMINI_FAMILY,
  installed,
  type Agent,
  type Item,
  type Part,
  type Pick,
} from "./scientist";
import { goToNeuron, loadProposals, openHead, useAtlas } from "./state";

/* --------------------------------------------------------------------------- a little markdown, made as elements */

/** The only addresses a link in the scientist's reply may open: a file in its work folder and an exported control
 * vector, both on this server (backend/scientist_chat.py `work_file`, backend/atlas.py `export_file`). Any other link is
 * left as the text it was written as. */
const FILE_LINK = /^\/api\/atlas\/(?:scientist\/work|exports)\/[A-Za-z0-9._~%+\-/]+$/;
export const fileLink = (url: string) => FILE_LINK.test(url) && !url.includes("..") && !url.includes("//");

/** Inline marks: `code`, **bold**, *italic*, a link to one of the scientist's files ([name](/api/atlas/scientist/work/…)),
 * and citations in the pages' count (L54 N15491, L54 h3) as links. */
function inline(text: string, key = "i"): React.ReactNode[] {
  const arch = useAtlas.getState().arch;
  const out: React.ReactNode[] = [];
  const re = /\[([^\]\n]+)\]\(([^)\s]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*\s][^*]*)\*|\bL(\d{1,2})[ /·]?N(\d{1,5})\b|\bL(\d{1,2}) h(\d{1,2})\b/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const id = `${key}.${k++}`;
    if (m[1] != null) {
      out.push(
        fileLink(m[2]) ? (
          <a key={id} className="atlas-file-link" href={m[2]} target="_blank" rel="noreferrer">
            {m[1]}
          </a>
        ) : (
          m[0]
        ),
      );
      last = re.lastIndex;
      continue;
    }
    if (m[3] != null) out.push(<code key={id}>{m[3]}</code>);
    else if (m[4] != null) out.push(<b key={id}>{inline(m[4], id)}</b>);
    else if (m[5] != null) out.push(<i key={id}>{inline(m[5], id)}</i>);
    else if (m[6] != null) {
      const l = +m[6] - 1;
      const n = +m[7];
      const ok = !!arch && l >= 0 && l < arch.n_layers && n < arch.d_ff;
      out.push(
        ok ? (
          <a
            key={id}
            className="atlas-nref"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              goToNeuron(l, n);
            }}
          >
            {m[0]}
          </a>
        ) : (
          m[0]
        ),
      );
    } else {
      const l = +m[8] - 1;
      const h = +m[9];
      const ok = !!arch && l >= 0 && l < arch.n_layers && h < (arch.layers[l]?.n_heads ?? 0);
      out.push(
        ok ? (
          <a
            key={id}
            className="atlas-nref"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              void openHead(l, h);
            }}
          >
            {m[0]}
          </a>
        ) : (
          m[0]
        ),
      );
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const Md: React.FC<{ text: string }> = ({ text }) => {
  const out: React.ReactNode[] = [];
  const lines = text.split("\n");
  let list: { tag: "ul" | "ol"; items: React.ReactNode[] } | null = null;
  const flush = () => {
    if (!list) return;
    const L = list;
    out.push(L.tag === "ul" ? <ul key={`l${out.length}`}>{L.items}</ul> : <ol key={`l${out.length}`}>{L.items}</ol>);
    list = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flush();
      const body: string[] = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i]);
      out.push(<pre key={`c${i}`}>{body.join("\n")}</pre>);
      continue;
    }
    const h = line.match(/^#{1,4}\s+(.*)/);
    const ul = line.match(/^\s*[-*]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (ul || ol) {
      const tag = ul ? "ul" : "ol";
      if (list && list.tag !== tag) flush();
      if (!list) list = { tag, items: [] };
      list.items.push(<li key={i}>{inline((ul ?? ol)![1], `li${i}`)}</li>);
      continue;
    }
    flush();
    if (h) out.push(<h4 key={i}>{inline(h[1], `h${i}`)}</h4>);
    else if (line.trim()) out.push(<p key={i}>{inline(line, `p${i}`)}</p>);
  }
  flush();
  return <div className="atlas-md">{out}</div>;
};

/* ------------------------------------------------------------------------------------------------- the thread */

/** A tool call in one short line: "neuron L54 N15491", "search lens “Sure”". The scientist's tools count layers as the pages do. */
function toolLabel(name: string, input: unknown): string {
  const a = (input && typeof input === "object" ? input : {}) as Record<string, unknown>;
  const ref = a.layer != null && a.n != null ? `L${a.layer} N${a.n}` : a.layer != null && a.head != null ? `L${a.layer} h${a.head}` : a.layer != null ? `L${a.layer}` : "";
  const short = (t: string) => (t.length > 90 ? `${t.slice(0, 90)}…` : t);
  const extra =
    typeof a.command === "string"
      ? short(a.command)
      : typeof a.file_path === "string"
        ? a.file_path
        : typeof a.pattern === "string"
          ? `“${a.pattern}”`
          : typeof a.query === "string"
            ? `“${a.query}”`
            : typeof a.text === "string"
              ? `“${short(a.text)}”`
              : Array.isArray(a.candidates)
                ? `${a.candidates.length} candidates`
                : Array.isArray(a.neurons)
                  ? `${a.neurons.length} neurons`
                  : "";
  const many = Array.isArray(a.prompts) ? `on ${a.prompts.length} prompts` : "";
  return [name.replace(/_/g, " "), ref, extra, many].filter(Boolean).join(" ");
}

/** The file a finished tool call left for the person, with where the page opens it: a control vector the scientist
 * exported (its reply begins with the file's name), or a file it wrote or edited in its work folder. */
function toolFile(name: string, input: unknown, output: string): { href: string; name: string; verb: string } | null {
  if (name === "export_control_vector") {
    const m = /^\{"file": "([A-Za-z0-9][A-Za-z0-9._-]*\.gguf)"/.exec(output);
    return m ? { href: exportUrl(m[1]), name: m[1], verb: "download" } : null;
  }
  const work = useScientist.getState().backends?.work;
  const path = (input as { file_path?: unknown } | null)?.file_path;
  if (!work || (name !== "Write" && name !== "Edit") || typeof path !== "string") return null;
  const root = work.folder.replace(/\/+$/, "") + "/";
  if (!path.startsWith(root)) return null;
  const rel = path.slice(root.length);
  const href = work.route + rel.split("/").map(encodeURIComponent).join("/");
  return fileLink(href) ? { href, name: rel, verb: "open" } : null;
}

const pretty = (t: string) => {
  try {
    return JSON.stringify(JSON.parse(t), null, 1);
  } catch {
    return t;
  }
};

/** For each file a reply's tool calls left, the last call that wrote it: the link goes under that one, once, however
 * many edits the file took. */
function fileParts(parts: Part[]): Map<number, { href: string; name: string; verb: string }> {
  const last = new Map<string, number>();
  const files = new Map<number, { href: string; name: string; verb: string }>();
  parts.forEach((p, j) => {
    if (p.t !== "tool" || p.error || p.output == null) return;
    const f = toolFile(p.name, p.input, p.output);
    if (!f) return;
    const before = last.get(f.href);
    if (before !== undefined) files.delete(before);
    last.set(f.href, j);
    files.set(j, f);
  });
  return files;
}

const PartView: React.FC<{ p: Part; box: number; file?: { href: string; name: string; verb: string } }> = ({ p, box, file }) => {
  switch (p.t) {
    case "md":
      return <Md text={p.text} />;
    case "thinking":
      // a model whose thinking is not shown leaves these empty: no row for nothing
      return p.text.trim() ? (
        <details className="atlas-think">
          <summary>thinking</summary>
          <pre>{p.text}</pre>
        </details>
      ) : null;
    case "tool":
      return (
        <>
          <details className="atlas-tool">
            <summary>{toolLabel(p.name, p.input)}</summary>
            <pre className="atlas-io">
              <b>{p.name}</b> {JSON.stringify(p.input ?? {}, null, 1)}
            </pre>
            <pre className={`atlas-io out${p.error ? " err" : ""}`}>{p.output == null ? "running…" : pretty(p.output)}</pre>
          </details>
          {file ? (
            <p className="atlas-small atlas-tool-file">
              <a className="atlas-file-link" href={file.href} target="_blank" rel="noreferrer">
                {file.verb} {file.name}
              </a>
            </p>
          ) : null}
        </>
      );
    case "error":
      return <p className="error-line atlas-small">{p.text}</p>;
    case "note":
      return <p className="muted atlas-small">{p.text}</p>;
    case "stopped":
      return <p className="muted atlas-small">stopped</p>;
    case "usename":
      return (
        <button type="button" className="btn-ghost plain atlas-usename" onClick={() => void takeName(p, box)}>
          use name: {p.name}
        </button>
      );
  }
};

const Thread: React.FC = () => {
  const items = useScientist((s) => s.items);
  const busy = useScientist((s) => s.busy);
  const box = useScientist((s) => s.box);
  // an Antigravity chat cannot be taken back (the server refuses its /rewind): no edit, no ask again
  const canRewind = useScientist((s) => (s.chat && s.chatPick ? s.chatPick : s.pick).agent) !== "agy";
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [items, busy]);
  let lastUser = -1;
  for (let i = items.length - 1; i >= 0; i--)
    if (items[i].kind === "user") {
      lastUser = i;
      break;
    }
  const lastReply = lastUser >= 0 ? items.findIndex((x, i) => i > lastUser && x.kind === "assistant") : -1;
  return (
    <div
      className="atlas-sthread"
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
    >
      {!items.length ? (
        <p className="muted atlas-empty">
          Ask anything about the model, or open a neuron and press <b>explain</b>.
        </p>
      ) : null}
      {items.map((it: Item, i) =>
        it.kind === "user" ? (
          <div key={i} className="atlas-msg user">
            {it.shown}
            {it.atts.length ? <span className="atlas-att"> · attached {it.atts.map((a) => a.name).join(", ")}</span> : null}
            {i === lastUser && !busy && canRewind ? (
              <span className="atlas-ctl">
                <button type="button" className="btn-link" title="take this message and its reply back, into the box" onClick={() => void rewindLast("edit")}>
                  edit
                </button>
              </span>
            ) : null}
          </div>
        ) : (
          <div key={i} className="atlas-msg assistant">
            {((files) => it.parts.map((p, j) => <PartView key={j} p={p} box={i} file={files.get(j)} />))(fileParts(it.parts))}
            {busy && i === box ? <p className="atlas-busy">working…</p> : null}
            {i === lastReply && !busy && canRewind ? (
              <span className="atlas-ctl">
                <button type="button" className="btn-link" title="throw this reply away and ask again" onClick={() => void rewindLast("reroll")}>
                  ask again
                </button>
              </span>
            ) : null}
          </div>
        ),
      )}
      {busy && box === null ? <p className="atlas-busy">working…</p> : null}
    </div>
  );
};

const Past: React.FC = () => {
  const past = useScientist((s) => s.past);
  const chat = useScientist((s) => s.chat);
  if (past === null) return null;
  if (past === "loading") return <p className="muted atlas-small atlas-pad">reading past chats…</p>;
  const ago = (t: number) => {
    const s = Date.now() / 1000 - t;
    return s < 3600 ? `${Math.max(1, Math.round(s / 60))} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : new Date(t * 1000).toLocaleDateString();
  };
  const by = (c: PastChat) => {
    const base =
      c.agent === "openai"
        ? (c.profile ?? "endpoint")
        : c.agent === "agy"
          ? "Antigravity"
          : c.agent === "codex"
            ? "Codex"
            : c.agent === "gemini"
              ? "Gemini CLI"
              : "Claude Code";
    return [base, c.model, c.effort ? `thinking ${c.effort}` : null].filter(Boolean).join(" · ");
  };
  // what a past chat records is the effort asked for; Claude Code may have run it lower
  const asked = (c: PastChat) =>
    c.effort && c.agent === "claude" ? "the thinking effort asked for; Claude Code may lower it for the model without saying so, or drop it" : undefined;
  return (
    <div className="atlas-sthread">
      {past.length ? (
        past.map((c) => (
          <button key={c.chat} type="button" className={`atlas-pastrow${c.chat === chat ? " on" : ""}`} onClick={() => void openPast(c)}>
            <span>{c.title || "—"}</span>
            <span className="muted atlas-small" title={asked(c)}>
              {by(c)} · {ago(c.t)} · {c.turns} message{c.turns === 1 ? "" : "s"}
            </span>
          </button>
        ))
      ) : (
        <p className="muted atlas-pad">No past chats yet.</p>
      )}
    </div>
  );
};

/** The name the page gives each backend. */
const BACKEND_NAME: Record<Agent, string> = { agy: "Antigravity", claude: "Claude Code", codex: "Codex", gemini: "Gemini CLI", openai: "an endpoint" };
/** Where this browser keeps that the person was told what a backend does at the home reach, one backend each. */
const ACK_KEY = "theodolite.atlas.scientist.home-shell";
const ACKED_AGENTS: Agent[] = ["agy", "claude", "codex", "gemini"];
const readAck = (a: Agent): boolean => {
  try {
    return localStorage.getItem(`${ACK_KEY}.${a}`) === "1";
  } catch {
    return false;
  }
};

/** Which backend a new chat runs on: Claude Code, Codex or Gemini where installed (Gemini through Antigravity by
 * default, or the Gemini CLI, picked beside it), or an OpenAI-compatible endpoint and model. */
const BackendPick: React.FC = () => {
  const b = useScientist((s) => s.backends);
  const pick = useScientist((s) => s.pick);
  const chatPick = useScientist((s) => s.chatPick);
  const chat = useScientist((s) => s.chat);
  const models = useScientist((s) => s.models);
  const cliModels = useScientist((s) => s.cliModels);
  const [customOpen, setCustomOpen] = useState(false);
  const fixed = !!chat && !!chatPick;
  const p: Pick = fixed ? chatPick! : pick;
  const gemini = GEMINI_FAMILY.includes(p.agent);
  const value = p.agent === "openai" ? `openai:${p.profile}` : gemini ? "gemini-family" : p.agent;
  const missing = (a: Agent) => (b && !installed(b, a) ? " (not installed)" : "");
  const card = b?.openai.find((x) => x.name === p.profile);
  const ms = p.agent === "openai" && p.profile ? models[p.profile] : undefined;
  const cm = p.agent !== "openai" ? cliModels[p.agent] : undefined;
  const { list: cmList, defaultModel, efforts: cliEfforts, defaultEffort } = cliChoices(p, b, cm);
  const modelTitle =
    p.agent === "agy"
      ? cmList?.models.length
        ? "the models Antigravity's own chat process reported when a chat last started"
        : "Antigravity's models are listed here once an Antigravity chat has started and reported them; until then, type a model id or keep the default"
      : p.agent === "codex"
        ? "the models Codex lists"
        : "the CLI's own aliases; custom… takes any model id";
  const effortTitle =
    p.agent === "claude"
      ? "thinking effort for a new chat; Claude Code may lower it for the model without saying so, or drop it if the model refuses it"
      : 'thinking effort for a new chat; Codex\'s "ultra" is not offered: it also hands work to sub-agents without asking';
  // a model or effort picked is fixed by a chat's first message; a chat begun with none records none, and its process
  // is given the default as it then stands each time it starts
  const fixedModelTitle = p.model
    ? "this chat runs on the model it began with; start a new chat to change it"
    : "this chat began with no model picked: each time its process starts it is given the default as it then stands; start a new chat to pick one";
  const fixedEffortTitle = p.effort
    ? p.agent === "claude"
      ? "this chat began with this thinking effort asked for; Claude Code may lower it for the model, or drop it; start a new chat to change it"
      : "this chat runs on the thinking effort it began with; start a new chat to change it"
    : "this chat began with no thinking effort picked: each time its process starts it is given the default as it then stands; start a new chat to pick one";
  const showCliControls = p.agent !== "openai" && (!b || installed(b, p.agent));
  const showCustomInput =
    showCliControls &&
    !fixed &&
    (customOpen ||
      (cmList !== null && cmList.models.length === 0 && cmList.custom_model) ||
      (typeof cm === "string" && cm !== "loading"));
  return (
    <div className="atlas-spick">
      <select
        className="atlas-select"
        aria-label="the scientist's backend for a new chat"
        value={value}
        disabled={fixed}
        title={fixed ? "this chat runs on the backend it began with; start a new chat to change it" : "the backend a new chat runs on"}
        onChange={(e) => {
          setCustomOpen(false);
          setPick(backendPick(e.target.value, b));
        }}
      >
        <option value="claude" disabled={!!b && !installed(b, "claude")}>
          Claude Code{missing("claude")}
        </option>
        <option value="codex" disabled={!!b && !installed(b, "codex")}>
          Codex{missing("codex")}
        </option>
        <option value="gemini-family" disabled={!!b && !GEMINI_FAMILY.some((a) => installed(b, a))}>
          Gemini{b && !GEMINI_FAMILY.some((a) => installed(b, a)) ? " (not installed)" : ""}
        </option>
        {(b?.openai ?? []).map((x) => (
          <option key={x.name} value={`openai:${x.name}`}>
            {x.label}
          </option>
        ))}
      </select>
      {gemini ? (
        <select
          className="atlas-select"
          aria-label="how a Gemini chat runs"
          value={p.agent}
          disabled={fixed}
          title={fixed ? "this chat runs on the backend it began with; start a new chat to change it" : "Antigravity unless you pick the Gemini CLI"}
          onChange={(e) => {
            setCustomOpen(false);
            setPick(backendPick(e.target.value, b));
          }}
        >
          <option value="agy" disabled={!!b && !installed(b, "agy")}>
            Antigravity{missing("agy")}
          </option>
          <option value="gemini" disabled={!!b && !installed(b, "gemini")}>
            Gemini CLI{missing("gemini")}
          </option>
        </select>
      ) : null}
      {showCliControls ? (
        cm === "loading" && !fixed ? (
          <select className="atlas-select" aria-label="the backend's model" disabled value="">
            <option value="">loading models…</option>
          </select>
        ) : showCustomInput ? (
          <input
            className="input-line atlas-model"
            aria-label="the backend's model"
            placeholder={defaultModel ? `default (${defaultModel})` : "model id (default)"}
            title={modelTitle}
            value={p.model ?? ""}
            disabled={fixed}
            onChange={(e) => setPick({ ...pick, model: e.target.value || null })}
          />
        ) : cmList && cmList.models.length ? (
          <select
            className="atlas-select"
            aria-label="the backend's model"
            value={p.model ?? ""}
            disabled={fixed}
            title={fixed ? fixedModelTitle : modelTitle}
            onChange={(e) => {
              const v = e.target.value;
              if (v === "__custom__") {
                setCustomOpen(true);
                return;
              }
              // an effort the next model does not list is not kept
              const next = { ...pick, model: v || null };
              const keepEffort = p.effort && cliChoices(next, b, cm).efforts.includes(p.effort) ? p.effort : null;
              setPick({ ...next, effort: keepEffort });
            }}
          >
            <option value="">{defaultModel ? `default (${defaultModel})` : "default model"}</option>
            {p.model && !cmList.models.some((m) => m.id === p.model) ? <option value={p.model}>{p.model}</option> : null}
            {cmList.models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name || m.id}
              </option>
            ))}
            {cmList.custom_model && !fixed ? <option value="__custom__">custom…</option> : null}
          </select>
        ) : p.model ? (
          <select className="atlas-select" aria-label="the backend's model" value={p.model} disabled>
            <option value={p.model}>{p.model}</option>
          </select>
        ) : null
      ) : null}
      {/* shown too while the pick holds an effort no list offers (the list loading, or Codex's not come), so that the
          effort a first message posts is in sight and can be cleared */}
      {showCliControls && (cliEfforts.length > 0 || !!p.effort) ? (
        <select
          className="atlas-select"
          aria-label="thinking effort"
          value={p.effort ?? ""}
          disabled={fixed}
          title={fixed ? fixedEffortTitle : effortTitle}
          onChange={(e) => setPick({ ...pick, effort: e.target.value || null })}
        >
          <option value="">{defaultEffort ? `thinking: default (${defaultEffort})` : "thinking: default"}</option>
          {p.effort && !cliEfforts.includes(p.effort) ? <option value={p.effort}>thinking: {p.effort}</option> : null}
          {cliEfforts.map((ef) => (
            <option key={ef} value={ef}>
              thinking: {ef}
            </option>
          ))}
        </select>
      ) : null}
      {showCliControls && typeof cm === "string" && cm !== "loading" ? <span className="error-line atlas-small">{cm}</span> : null}
      {showCliControls && !fixed && cmList?.error ? (
        <span className="muted atlas-small" title={cmList.error}>
          Codex did not list its models; its default and a typed id are offered
        </span>
      ) : null}
      {p.agent === "openai" ? (
        Array.isArray(ms) && ms.length ? (
          <select
            className="atlas-select"
            aria-label="the endpoint's model"
            value={p.model ?? ""}
            disabled={fixed}
            onChange={(e) => setPick({ ...pick, model: e.target.value })}
          >
            {p.model && !ms.some((m) => m.id === p.model) ? <option value={p.model}>{p.model}</option> : null}
            {ms.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name || m.id}
                {m.tools === false ? " (no tools)" : ""}
              </option>
            ))}
          </select>
        ) : (
          <input
            className="input-line atlas-model"
            aria-label="the endpoint's model"
            placeholder="model id"
            value={p.model ?? ""}
            disabled={fixed}
            onChange={(e) => setPick({ ...pick, model: e.target.value || null })}
          />
        )
      ) : null}
      {p.agent === "openai" && typeof ms === "string" ? <span className="error-line atlas-small">{ms}</span> : null}
      {p.agent === "openai" && card && card.key !== "set" && card.key !== "none needed" ? <span className="error-line atlas-small">key {card.key}</span> : null}
      {p.agent === "openai" && card && !card.this_machine ? (
        <span className="muted atlas-small" title={card.base_url}>
          sent to {new URL(card.base_url).hostname}
        </span>
      ) : null}
    </div>
  );
};

/** Every name and note the scientist proposed that still waits for the person (backend/field_notes.py): kept, it is the
 * name or note everywhere, as the scientist's; until then nothing reads it, later chats included. */
const Waiting: React.FC = () => {
  const p = useAtlas((s) => s.proposals);
  useEffect(() => {
    void loadProposals();
  }, []);
  const rows = p
    ? [
        ...Object.keys(p.neurons).map((k) => ({ kind: "neuron" as const, k, v: p.neurons[k] })),
        ...Object.keys(p.heads).map((k) => ({ kind: "head" as const, k, v: p.heads[k] })),
      ]
    : [];
  if (!rows.length) return null;
  return (
    <div role="region" aria-label="proposals waiting for you" style={{ margin: "4px 0 6px" }}>
      <div className="kicker">waiting for you · the scientist’s proposals</div>
      {rows.slice(0, 12).map(({ kind, k, v }) => {
        const [l, n] = k.split(":").map(Number);
        return (
          <div key={`${kind}/${k}`} style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <button type="button" className="btn-link" style={{ whiteSpace: "nowrap" }} onClick={() => void (kind === "neuron" ? goToNeuron(l, n) : openHead(l, n))}>
              {kind === "neuron" ? cite(l, n) : citeHead(l, n)}
            </button>
            <div style={{ minWidth: 0 }}>
              {"name" in v ? <ProposalLine kind={kind} layer={l} n={n} part="name" /> : null}
              {"note" in v ? <ProposalLine kind={kind} layer={l} n={n} part="note" /> : null}
            </div>
          </div>
        );
      })}
      {rows.length > 12 ? <p className="muted atlas-small">and {rows.length - 12} more</p> : null}
      <p className="muted atlas-small" style={{ margin: "2px 0 0" }}>
        Nothing reads a proposal until you keep it: the map, the lists, search and later chats read the names and notes kept.
      </p>
    </div>
  );
};

export const Scientist: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const busy = useScientist((s) => s.busy);
  const stale = useScientist((s) => s.stale);
  const input = useScientist((s) => s.input);
  const pending = useScientist((s) => s.pending);
  const past = useScientist((s) => s.past);
  const pastError = useScientist((s) => s.pastError);
  const warning = useScientist((s) => s.backends?.warning ?? null);
  const backendsError = useScientist((s) => s.backendsError);
  const reach = useScientist((s) => s.backends?.reach);
  const agent = useScientist((s) => (s.chat && s.chatPick ? s.chatPick : s.pick).agent);
  const [acks, setAcks] = useState<Partial<Record<Agent, boolean>>>(() => Object.fromEntries(ACKED_AGENTS.map((a) => [a, readAck(a)])));
  // a shell at the home reach is asked about once for each backend, in this browser, before its first chat runs there
  const homeShell = reach?.mode === "home" && ACKED_AGENTS.includes(agent);
  const needsAck = homeShell && !acks[agent];
  const [drop, setDrop] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    void loadBackends();
    void openEvents();
  }, []);
  useEffect(() => {
    const t = boxRef.current;
    if (!t) return;
    t.style.height = "auto";
    t.style.height = `${Math.min(200, t.scrollHeight)}px`;
  }, [input]);
  const at = f ? `L${f.layer + 1} ${f.kind === "head" ? "h" : "N"}${f.n}` : "";
  return (
    <div
      className={`atlas-scientist${drop ? " drop" : ""}`}
      onDragOver={(e) => {
        if (e.dataTransfer?.types?.includes("Files")) {
          e.preventDefault();
          setDrop(true);
        }
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDrop(false);
      }}
      onDrop={(e) => {
        if (!e.dataTransfer?.files?.length) return;
        e.preventDefault();
        setDrop(false);
        void attachFiles([...e.dataTransfer.files]);
      }}
    >
      <div className="atlas-sbar">
        <BackendPick />
        <span className="atlas-sbar-end">
          <button type="button" className="btn-link" aria-pressed={past !== null} onClick={() => void togglePast()}>
            past
          </button>
          <button type="button" className="btn-link" onClick={newChat}>
            new
          </button>
        </span>
      </div>
      <ErrorLine error={warning ?? backendsError ?? pastError} />
      {past !== null ? <Past /> : <Thread />}
      {needsAck && reach ? (
        <div className="atlas-ack" role="alert">
          <p className="atlas-small red" style={{ margin: 0 }}>
            On this server {BACKEND_NAME[agent]} {reach[agent]}. What it reads (the model’s answers, census answers, names and notes,
            attachments) can carry instructions, and nothing stands between them and that shell.
          </p>
          <p className="muted atlas-small" style={{ margin: "4px 0" }}>
            {reach.folder ? (
              <>
                Without {reach.env}=home on the server it {reach.folder[agent]}; an endpoint has no shell at all.
              </>
            ) : (
              <>Without {reach.env}=home on the server it has no shell over your home folder; an endpoint has no shell at all.</>
            )}
          </p>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              try {
                localStorage.setItem(`${ACK_KEY}.${agent}`, "1");
              } catch {
                // asked again next time
              }
              setAcks({ ...acks, [agent]: true });
            }}
          >
            I understand: let it run here
          </button>
        </div>
      ) : null}
      {stale ? (
        // the server ends its process only when asked: commands it left running in the background in its process group would end with it
        <div className="atlas-stale" role="status">
          <p className="atlas-small" style={{ margin: 0 }}>
            This chat started before its instructions or tools last changed, and runs those it started with. Restart it to take them up; that
            also stops its process group, with any commands it left running in it.
          </p>
          <button type="button" className="btn-ghost plain" disabled={busy} title={busy ? "stop the reply first" : undefined} onClick={() => void restartChat()}>
            restart
          </button>
        </div>
      ) : null}
      <Waiting />
      {f && !busy && !needsAck ? (
        <div className="atlas-actions">
          <button type="button" className="btn-ghost plain" onClick={() => void explainFocus()}>
            explain {at}
          </button>
          {f.kind === "neuron" ? (
            <button type="button" className="btn-ghost plain" onClick={askWeb}>
              what does its web do?
            </button>
          ) : null}
        </div>
      ) : null}
      {pending.length ? (
        <div className="atlas-files">
          {pending.map((a, i) => (
            <span key={i} className={`atlas-file${a.id ? "" : " up"}`}>
              {a.name}
              <button type="button" aria-label={`remove ${a.name}`} onClick={() => removePending(i)}>
                ×
              </button>
            </span>
          ))}
        </div>
      ) : null}
      <form
        className="atlas-sform"
        hidden={needsAck}
        onSubmit={(e) => {
          e.preventDefault();
          if (busy && !input.trim()) void stop();
          else send();
        }}
      >
        <button type="button" className="btn-ghost plain" aria-label="attach files" title="attach files (or drop or paste them)" onClick={() => fileRef.current?.click()}>
          +
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            void attachFiles([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
        <textarea
          ref={boxRef}
          rows={1}
          value={input}
          placeholder={busy ? "a message reaches it between tool calls" : "ask the scientist"}
          aria-label="a message for the scientist"
          spellCheck={false}
          onChange={(e) => useScientist.setState({ input: e.target.value })}
          onPaste={(e) => {
            const fs = [...(e.clipboardData?.files ?? [])];
            if (fs.length) {
              e.preventDefault();
              void attachFiles(fs);
            }
          }}
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.shiftKey) return;
            e.preventDefault();
            send();
          }}
        />
        <button type="submit" className={`btn-ghost${busy && !input.trim() ? " stop" : ""}`} aria-label={busy && !input.trim() ? "stop" : "send"}>
          {busy && !input.trim() ? "stop" : "send"}
        </button>
      </form>
      <Rubric style={{ fontSize: 14, margin: "4px 0 0" }}>
        The scientist reads the atlas with the same tools the page uses and can move this screen
        {reach ? (
          <>
            ; beyond them, here it <span className={homeShell ? "red" : undefined}>{reach[agent]}</span>
          </>
        ) : null}
        . What it says is a lead to check: it cites the neurons it read, so open them. <a href={href.lexicon("scientist")}>What the scientist is.</a>
      </Rubric>
    </div>
  );
};
