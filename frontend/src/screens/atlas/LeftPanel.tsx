import React, { useEffect, useRef, useState } from "react";
import { IfApparatus, LensCaution, Rubric } from "../../components/kit";
import { href } from "../../router";
import { wp } from "../../lib/words";
import { cite, citeHead, get, noteKey, type CensusCard, type CensusHit, type CensusSearchReply, type RunRow, type SearchRow } from "./api";
import { censusChance, Chips, ErrorLine, FaceLine, NameMark, ProposalLine } from "./parts";
import {
  clearTrail,
  closeFocus,
  goToNeuron,
  openHead,
  runView,
  saveNote,
  setLayer,
  setLeftTab,
  setQuery,
  censusLayersOf,
  theMap,
  useAtlas,
} from "./state";

/** The open neuron's name, kept on the server (backend/field_notes.py) and saved as it is typed. */
const NameField: React.FC<{ layer: number; n: number }> = ({ layer, n }) => {
  const notes = useAtlas((s) => s.notes);
  const entry = notes?.neurons[noteKey(layer, n)];
  const [name, setName] = useState(entry?.name ?? "");
  const [saved, setSaved] = useState<string | null>(null);
  const [focused, setFocused] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const key = noteKey(layer, n);
  const shown = useRef(key);
  // the kept name comes into the box when another neuron opens, or when the box is not being typed in (a saved name
  // comes back trimmed, which must not move the words under the reader's cursor)
  useEffect(() => {
    if (shown.current !== key || !focused) {
      shown.current = key;
      setName(entry?.name ?? "");
    }
  }, [key, entry?.name, focused]);
  const save = (v: string) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      try {
        await saveNote(layer, n, { name: v.replace(/\s+/g, " ") });
        setSaved("kept");
      } catch (e) {
        setSaved(String((e as Error).message));
      }
      window.setTimeout(() => setSaved(null), 1500);
    }, 600);
  };
  return (
    <div className="atlas-name">
      <input
        className="input-line"
        value={name}
        placeholder="name it"
        aria-label={`a name for ${cite(layer, n)}`}
        onChange={(e) => {
          setName(e.target.value);
          save(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
      />
      {entry?.name && (entry.name_by ?? entry.by) === "scientist" ? <span className="muted atlas-small">named by the scientist: a guess, not a finding</span> : null}
      {saved ? <span className="muted atlas-small">{saved}</span> : null}
    </div>
  );
};

/** A link that opens the atlas on this neuron or head (#/atlas/47/17966, #/atlas/54/h3), copied for a note or a
 * message: the address bar forgets what is open, so this is how a find is passed on. */
const CopyLink: React.FC<{ f: { kind: "neuron" | "head"; layer: number; n: number } }> = ({ f }) => {
  const [said, setSaid] = useState<string | null>(null);
  const hash = f.kind === "head" ? `#/atlas/${f.layer + 1}/h${f.n}` : href.atlas(f.layer + 1, f.n);
  const url = `${window.location.origin}${window.location.pathname}${hash}`;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setSaid("copied");
    } catch {
      setSaid(url); // the clipboard is closed to this page: the link is shown to copy by hand
    }
    window.setTimeout(() => setSaid(null), 4000);
  };
  return (
    <button type="button" className="btn-link" title={`copy a link that opens the atlas here: ${url}`} onClick={() => void copy()}>
      {said ?? "copy link"}
    </button>
  );
};

const FocusHead: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const unit = useAtlas((s) => s.unit);
  if (!f) return null;
  const s = unit?.stats;
  return (
    <div className="atlas-focus-head">
      <div className="row-between">
        <span className="atlas-id">{f.kind === "head" ? citeHead(f.layer, f.n) : cite(f.layer, f.n)}</span>
        <span className="row" style={{ gap: 14 }}>
          <CopyLink key={`${f.kind}/${f.layer}/${f.n}`} f={f} />
          <button type="button" className="btn-link" aria-label="close" onClick={closeFocus}>
            close
          </button>
        </span>
      </div>
      {f.kind === "neuron" && s ? (
        <div className="muted atlas-small">
          census #{unit?.rank ?? "—"} of {unit?.n_ranked} · mass {(s.mass ?? s.z).toFixed(1)}
          {s.null_pct != null ? ` · above ${(s.null_pct * 100).toFixed(0)}% of controls` : ""}
          {s.null_pct === 1 && unit?.card?.null.above_all_by_chance != null
            ? ` · a lead: about ${unit.card.null.above_all_by_chance.toFixed(0)} of the layer would beat them all by chance`
            : ""}
        </div>
      ) : null}
      {f.kind === "neuron" ? <NameField layer={f.layer} n={f.n} /> : null}
      {f.kind === "neuron" ? <ProposalLine layer={f.layer} n={f.n} part="name" /> : null}
    </div>
  );
};

/** What was opened lately in this browser, the latest first (the open one left out): a way back along a web of links. */
const Trail: React.FC = () => {
  const trail = useAtlas((s) => s.trail);
  const f = useAtlas((s) => s.focus);
  const notes = useAtlas((s) => s.notes);
  const rest = trail.filter((x) => !(f && x.kind === f.kind && x.layer === f.layer && x.n === f.n)).slice(0, 8);
  if (!rest.length) return null;
  return (
    <div className="atlas-trail" role="group" aria-label="opened lately, the latest first">
      <span className="atlas-lab">lately</span>
      {rest.map((x) => {
        const name = x.kind === "neuron" ? notes?.neurons[noteKey(x.layer, x.n)]?.name : notes?.heads?.[noteKey(x.layer, x.n)]?.name;
        return (
          <button
            key={`${x.kind}/${x.layer}/${x.n}`}
            type="button"
            className="atlas-chip"
            title={name ? `${name}: open it again` : "open it again"}
            onMouseEnter={() => (x.kind === "head" ? theMap()?.hoverHead(x.layer, x.n) : theMap()?.hover(x.layer, x.n))}
            onMouseLeave={() => (x.kind === "head" ? theMap()?.hoverHead(null) : theMap()?.hover(null))}
            onClick={() => (x.kind === "head" ? void openHead(x.layer, x.n) : goToNeuron(x.layer, x.n))}
          >
            {x.kind === "head" ? citeHead(x.layer, x.n) : cite(x.layer, x.n)}
          </button>
        );
      })}
      <button type="button" className="btn-link" onClick={clearTrail} title="forget what was opened lately">
        forget
      </button>
    </div>
  );
};

/** Bars over the layers named, one a layer, each as tall as its value against the tallest: the census by depth, or a
 * search's matches by depth. A layer whose doses are not calibrated is drawn hollow. A click opens the layer. */
const LayerBars: React.FC<{ bars: { layer: number; value: number; title: string; hollow?: boolean }[]; label: string }> = ({ bars, label }) => {
  const layer = useAtlas((s) => s.layer);
  const top = Math.max(1e-9, ...bars.map((b) => b.value));
  const go = (l: number) => {
    void setLayer(l);
    const m = theMap();
    if (m && m.layerMode !== null) m.enterLayer(l);
    else m?.viewLayer(l);
  };
  return (
    <div className="atlas-depth" role="group" aria-label={label}>
      {bars.map((b) => (
        <button
          key={b.layer}
          type="button"
          className={`${b.layer === layer ? "on" : ""}${b.hollow ? " hollow" : ""}`}
          title={b.title}
          aria-label={b.title}
          onClick={() => go(b.layer)}
        >
          <span className="bar" style={{ height: `${Math.round(2 + 22 * (b.value / top))}px` }} />
          <span className="no">{b.layer + 1}</span>
        </button>
      ))}
    </div>
  );
};

/** Every layer with a census, each bar the share of its neurons whose rank is set apart from random pushes (a
 * false-discovery q of 0.05 or less): where the census is, and how much of each layer it singles out. */
const CensusDepth: React.FC = () => {
  const layers = useAtlas((s) => s.censusLayers);
  if (!layers || layers.length < 2) return null;
  return (
    <LayerBars
      label="the layers with a census"
      bars={layers.map((c) => ({
        layer: c.layer,
        value: c.ranked && c.neurons ? (c.q05_neurons ?? 0) / c.neurons : 0,
        hollow: c.ranked && !c.doses_calibrated,
        title: c.ranked
          ? `layer ${c.layer + 1}: ${(c.q05_neurons ?? 0).toLocaleString()} of ${(c.neurons ?? 0).toLocaleString()} neurons set apart from random pushes (q ≤ 0.05)${c.doses_calibrated ? "" : " · doses not calibrated at this layer"}`
          : `layer ${c.layer + 1}: its census is not ranked yet (open the layer to rank it)`,
      }))}
    />
  );
};

const NeuronWeb: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const lens = useAtlas((s) => s.lens);
  const w = useAtlas((s) => s.webSum);
  const nh = useAtlas((s) => s.nheads);
  const webTotal = useAtlas((s) => s.webTotal);
  if (!f || f.kind !== "neuron") return null;
  const mine = (x: { layer: number; n: number } | null) => !!x && x.layer === f.layer && x.n === f.n;
  const search = (t: string) => setQuery(t.trim() || t, "lens");
  return (
    <div className="atlas-net">
      {mine(lens) && lens ? (
        <div className="atlas-wsum">
          <div className="kicker">its words</div>
          <div>
            <FaceLine layer={f.layer} face="+" w={{ ...lens.faces["+"], steer: lens.steer?.up ?? null }} k={10} />
          </div>
          <div>
            <FaceLine layer={f.layer} face="-" w={{ ...lens.faces["-"], steer: lens.steer?.down ?? null }} k={10} />
          </div>
          {lens.steer ? (
            <div className="muted atlas-small">
              <a href={href.lexicon("steer-lens")}>steer lens</a> ({lens.steer.source}) · KL {lens.steer.kl_up} / {lens.steer.kl_down}
            </div>
          ) : (
            <div className="muted atlas-small">no steer lens read for it yet (details)</div>
          )}
          <LensCaution sighting={null} no={f.layer + 1} style={{ fontSize: 14 }} />
        </div>
      ) : null}
      {mine(w) && w?.ready ? (
        <div className="atlas-wsum">
          <div className="kicker">
            its web · {webTotal ?? w.links} links · pooled from the {w.src === "steer" ? "steer lens" : "lens, labels the gate lets through"}
          </div>
          {w.stale ? <div className="atlas-small red">Read before its vault edit: these are its links as they were.</div> : null}
          {(
            [
              ["up", "+", "what drives it up", "upstream"],
              ["up", "-", "what drives it down", "upstream"],
              ["down", "+", "when it fires up, it drives", "downstream"],
              ["down", "-", "when it fires down, it drives", "downstream"],
            ] as const
          ).map(([side, sg, label, where]) => {
            const b = w[side]?.[sg];
            return (
              <React.Fragment key={`${side}${sg}`}>
                <div className="atlas-lab">
                  {label} · {b?.neurons ?? 0} {where}
                </div>
                <Chips list={b?.words} onPick={search} empty={b?.neurons ? "nothing above chance" : "no link reaches this way"} />
              </React.Fragment>
            );
          })}
          {w.gated && (w.gated.down || w.gated.up) ? (
            <div className="muted atlas-small">
              {w.gated.up + w.gated.down} faces left out: the label gate holds their words back
            </div>
          ) : null}
          <Rubric style={{ fontSize: 14, margin: "4px 0 0" }}>
            Weight-only links read from rest: a writer moves a reader when it fires on the face that opens the reader's gate, and pushes it
            toward one face; that holds for a reader whose gate is open, or a push big enough to open it, while a reader sitting shut, as most
            did where it was measured, is pushed about the other way, so "drives it up" can drive it down there. The strongest as they
            arrive, counting how Gemma scales the stream at every layer. Words pooled over many neurons, where the label gate was only tested
            on one at a time: leads to check with a push or a read, not circuits found.{" "}
            <a href={href.lexicon("web")}>What the web is.</a>
          </Rubric>
        </div>
      ) : w && mine(w) && !w.ready ? (
        <p className="muted atlas-small">
          {w.missing === "wiring"
            ? "The wiring index is not built, so its web cannot be drawn (scripts/atlas_build.py wiring)."
            : "Its web is drawn, but neither the steer-lens index nor the lens index is built, so there are no words to pool over it (scripts/atlas_build.py lens)."}
        </p>
      ) : null}
      {mine(nh) && nh ? (
        <div className="atlas-wsum">
          <div className="kicker">its heads</div>
          {(["q", "k", "v", "o"] as const).map((r) => (
            <div key={r} className="atlas-lrow">
              <span className="atlas-sgn">{r}</span>
              <span className="atlas-chips">
                {(nh.roles[r] ?? [])
                  .filter((x) => x.z >= 2)
                  .map((x) => (
                    <button
                      key={`${x.layer}/${x.head}`}
                      type="button"
                      className="atlas-chip"
                      onMouseEnter={() => theMap()?.hoverHead(x.layer, x.head)}
                      onMouseLeave={() => theMap()?.hoverHead(null)}
                      onClick={() => void openHead(x.layer, x.head)}
                    >
                      L{x.layer + 1} h{x.heads.length > 1 ? `${x.heads[0]}–${x.heads[x.heads.length - 1]}` : x.head}
                      <i>{x.z}</i>
                    </button>
                  ))}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
};

const HeadWeb: React.FC = () => {
  const f = useAtlas((s) => s.focus);
  const h = useAtlas((s) => s.headInfo);
  const p = useAtlas((s) => s.headProfile);
  if (!f || f.kind !== "head" || !h || h.layer !== f.layer || h.head !== f.n) return null;
  const role = (r: "q" | "k" | "v" | "o", label: string) => (
    <div className="atlas-wsum" key={r}>
      <div className="kicker">
        {r} · {label}
      </div>
      {h[r].slice(0, 8).map((c) => (
        <div key={`${c.layer}/${c.n}`} className="atlas-nline" onClick={() => goToNeuron(c.layer, c.n)}>
          <span className="atlas-id-small">{cite(c.layer, c.n)}</span>
          <FaceLine layer={c.layer} face="+" w={c.words} k={4} />
          <span className="atlas-z">{c.z.toFixed(0)}</span>
        </div>
      ))}
      <Chips list={h[`${r}_pooled`]?.tokens?.slice(0, 12)} onPick={(t) => setQuery(t, "lens")} empty="no words pooled" />
      {h[`${r}_pooled`]?.tokens?.length ? (
        <div className="muted atlas-small">
          pooled from these neurons’ + faces ({h[`${r}_pooled`]?.src === "steer" ? "the steer lens" : "labels the gate shows"}), a lead
        </div>
      ) : null}
    </div>
  );
  return (
    <div className="atlas-net">
      <p className="muted atlas-small">
        {h.kind === "full" ? "a full-attention layer, its values read from its keys" : "a sliding-window layer"} · key-value head {h.kv_head}
      </p>
      {role("q", "what it looks for")}
      {role("k", "what makes a word-piece findable")}
      {role("v", "what it carries")}
      {role("o", "where it delivers")}
      {p ? (
        <div className="atlas-wsum">
          <div className="kicker">where it looks, over eight texts</div>
          <div className="atlas-small">
            {p.dist_names.map((nm, i) => (
              <span key={nm} style={{ marginRight: 10 }}>
                {nm} {(p.dist[i] * 100).toFixed(0)}%
              </span>
            ))}
          </div>
          {p.pairs.slice(0, 5).map((x, i) => (
            <div key={i} className="atlas-small">
              {wp(x.qt)} ← {wp(x.kt)} <span className="muted">{x.w.toFixed(2)}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
};

const SearchRows: React.FC<{ rows: SearchRow[]; held?: SearchRow[]; total: number; tokens: number }> = ({ rows, held, total, tokens }) => (
  <ol className="atlas-rows">
    <li className="atlas-rows-head">
      {total} faces · {tokens} word-pieces matched
      {held?.length ? ` · the words of ${held.length} held back by the label gate (in the apparatus)` : ""}
    </li>
    {rows.map((x) => (
      <SearchRowItem key={`${x.layer}/${x.n}/${x.sign}/${x.src}`} x={x} />
    ))}
    {held?.length ? (
      <IfApparatus>
        <li className="atlas-rows-head">held back by the label gate: their words matched, but the gate leaves them out of the main reading and the map</li>
        {held.map((x) => (
          <SearchRowItem key={`h${x.layer}/${x.n}/${x.sign}/${x.src}`} x={x} />
        ))}
      </IfApparatus>
    ) : null}
  </ol>
);

const SearchRowItem: React.FC<{ x: SearchRow }> = ({ x }) => (
  <li onClick={() => goToNeuron(x.layer, x.n)} onMouseEnter={() => theMap()?.hover(x.layer, x.n)} onMouseLeave={() => theMap()?.hover(null)}>
    <span className="atlas-id-small">{cite(x.layer, x.n)}</span>
    <span className={`atlas-sg ${x.sign === "+" ? "up" : "down"}`}>
      {x.sign === "+" ? "+" : "−"}
      {x.rank + 1}
    </span>
    <span className="atlas-w">
      {x.name ? <NameMark layer={x.layer} n={x.n} name={x.name} /> : null}
      <FaceLine layer={x.layer} face={x.sign} w={x.words} k={5} />
    </span>
  </li>
);

const CensusHitItem: React.FC<{ x: CensusHit; bare: boolean }> = ({ x, bare }) => {
  const hit = new Set(x.matched);
  const further = x.matched.filter((w) => !x.words.includes(w));
  const place = `#${x.rank.toLocaleString()} of its layer's census${x.q != null ? ` · q ${x.q}` : ""}`;
  const why = x.above
    ? "no random control of its layer brought it as hard"
    : bare
      ? "its layer's census has no random controls: nothing to set it against"
      : "a random control of its layer brought it as hard: a weaker lead";
  return (
    <li
      className={x.above ? "" : "below"}
      title={`${place} · ${why}`}
      onClick={() => goToNeuron(x.layer, x.n)}
      onMouseEnter={() => theMap()?.hover(x.layer, x.n)}
      onMouseLeave={() => theMap()?.hover(null)}
    >
      <span className="atlas-id-small">{cite(x.layer, x.n)}</span>
      <span className="atlas-z">{x.score.toFixed(1)}</span>
      <span className="atlas-w">
        {x.name ? <NameMark layer={x.layer} n={x.n} name={x.name} /> : null}
        {x.words.map((w, i) => (
          <React.Fragment key={w}>
            {i ? " " : ""}
            {hit.has(w) ? <b>{w}</b> : w}
          </React.Fragment>
        ))}
        {further.length ? (
          <span className="muted" title="matched in a word further down its list than the five a list shows">
            {" "}
            · further down: <b>{further.join(" ")}</b>
          </span>
        ) : null}
      </span>
    </li>
  );
};

/** A search of every census at once: the neurons whose push made the model write the word, with where they lie by
 * depth. Each layer's random controls, searched the same way, are the null: how often they match says how many of the
 * neurons would by chance alone, and that count is what sets the list against chance. A row is faded where a control
 * of its layer brought the word as hard (or its layer has no controls); a bright row is not thereby beyond chance. */
const CensusSearchRows: React.FC<{ r: CensusSearchReply }> = ({ r }) => {
  const sum = (f: (c: CensusSearchReply["layers"][number]) => number) => r.layers.reduce((a, c) => a + f(c), 0);
  const neurons = sum((c) => c.neurons);
  const ch = r.chance;
  const bare = new Set(r.layers.filter((c) => !c.of_controls).map((c) => c.layer));
  const unset = sum((c) => (c.of_controls ? 0 : c.neurons));
  const weaker = sum((c) => (c.of_controls ? c.neurons - c.above : 0));
  const uncal = r.layers.filter((c) => !c.doses_calibrated && c.neurons).map((c) => c.layer + 1);
  const against = !ch
    ? "none of these layers has random controls, so nothing here is set against chance"
    : `${unset ? `of the ${ch.neurons.toLocaleString()} in layers with controls, ` : ""}${
        ch.at_most != null
          ? `fewer than about ${Math.round(ch.at_most).toLocaleString()} would by chance alone (none of ${ch.of_controls.toLocaleString()} random controls brought it)`
          : `about ${Math.round(ch.expected).toLocaleString()} would by chance alone (random controls brought it ${ch.controls === 1 ? "once" : `${ch.controls.toLocaleString()} times`} in ${ch.of_controls.toLocaleString()})`
      }`;
  if (!r.layers.length)
    return (
      <p className="muted atlas-small" style={{ padding: "6px 0" }}>
        No layer has a ranked census to search yet{r.not_read.length ? ` (layers ${r.not_read.map((l) => l + 1).join(", ")} are not ranked: open one to rank it)` : ""}.
      </p>
    );
  return (
    <>
      <LayerBars
        label="where the matches lie, by layer"
        bars={r.layers.map((c) => ({
          layer: c.layer,
          value: c.neurons,
          hollow: !c.doses_calibrated,
          title:
            `layer ${c.layer + 1}: ${c.neurons.toLocaleString()} neurons brought it; ` +
            (c.by_chance == null
              ? "its census has no random controls, so nothing sets that against chance"
              : `${c.controls} of ${c.of_controls.toLocaleString()} random controls did, which among its neurons would be about ${Math.round(c.by_chance).toLocaleString()}`) +
            (c.doses_calibrated ? "" : " · doses not calibrated at this layer"),
        }))}
      />
      <ol className="atlas-rows atlas-hits">
        <li className="atlas-rows-head">
          {neurons.toLocaleString()} neurons in {r.layers.filter((c) => c.neurons).length} of {r.layers.length} layers brought it; {against}
          {weaker ? ` · ${weaker.toLocaleString()} no harder than a random control of their layer (faded)` : ""}
          {unset ? ` · ${unset.toLocaleString()} in layers without controls (faded)` : ""}
          {uncal.length ? (
            <span className="red">
              {" "}
              · layer{uncal.length > 1 ? "s" : ""} {uncal.join(", ")}: doses not calibrated there, so a match or a miss may only say the dose was
              wrong
            </span>
          ) : null}
          {r.total > r.results.length ? ` · the first ${r.results.length} shown` : ""}
          {r.not_read.length ? ` · layers ${r.not_read.map((l) => l + 1).join(", ")} not ranked yet` : ""}
        </li>
        {r.results.map((x) => (
          <CensusHitItem key={`${x.layer}/${x.n}`} x={x} bare={bare.has(x.layer)} />
        ))}
      </ol>
      {r.total === 0 ? (
        <p className="muted atlas-small" style={{ padding: "6px 0" }}>
          No neuron’s census words {r.terms[0]?.startsWith("*") ? "hold" : "begin with"} it. A census word is a whole word of three letters or more
          {r.terms[0]?.startsWith("*") ? "." : `; write *${r.terms[0] ?? ""} to match inside words too.`}
        </p>
      ) : null}
      <Rubric style={{ fontSize: 14, margin: "6px 0 0" }}>
        Each of these made the model write the word more often when it was pushed than the average random direction pushed as far, on six
        fixed questions. The number is how far the word stands above the random pushes. Some of any list is chance: the line above says about
        how many, not which. A faded row is a weaker lead still. A lead to test with a push of your own, not what the neuron means.
      </Rubric>
    </>
  );
};

const CensusRows: React.FC = () => {
  const layer = useAtlas((s) => s.layer);
  const units = useAtlas((s) => s.units[s.layer]);
  const arch = useAtlas((s) => s.arch);
  const f = useAtlas((s) => s.focus);
  const [q, setQ] = useState("");
  const [controls, setControls] = useState(false);
  // the layer's census card, for whether its doses are calibrated on this model at this layer (said uncalibrated until
  // it says so) and, where they are not, the layers its design is calibrated at
  const run = arch?.census[String(layer)] ?? "";
  const key = run ? `${layer}/${run}` : "";
  const [card, setCard] = useState<{ key: string; calibrated: boolean; span: [number, number] | null } | null>(null);
  useEffect(() => {
    if (!key) return;
    let live = true;
    get<Partial<CensusCard>>("/api/atlas/census/card", { layer })
      .then((c) => live && setCard({ key, calibrated: c.doses_calibrated === true, span: c.doses_calibrated_layers ?? null }))
      .catch(() => live && setCard({ key, calibrated: false, span: null }));
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  const calibrated = !!card && card.key === key && card.calibrated;
  const span = card && card.key === key ? card.span : null;
  const doses = calibrated
    ? "doses calibrated on this model at this layer"
    : span
      ? `doses not calibrated at this layer (only at layers ${span[0] + 1}–${span[1] + 1})`
      : "doses not yet calibrated on this model";
  if (!arch?.census[String(layer)]) {
    const refused = arch?.census_refused?.[String(layer)];
    const have = censusLayersOf(arch ?? null);
    return (
      <>
        <CensusDepth />
        <p className="muted atlas-small" style={{ padding: "6px 0" }}>
          Layer {layer + 1} has no census yet: no neuron of it has been pushed on the census's questions (scripts/census_run.py --layer {layer},
          counting layers from 0).{refused ? ` A run of it is not read: ${refused}.` : ""}
          {have.length
            ? ` The census covers layer${have.length > 1 ? "s" : ""} ${have.length > 2 && have[have.length - 1] - have[0] === have.length - 1 ? `${have[0] + 1}–${have[have.length - 1] + 1}` : have.map((l) => l + 1).join(", ")}: pick one above, or search every census at once from the box (census).`
            : ""}
        </p>
      </>
    );
  }
  if (!units) return <p className="muted atlas-small">reading the census…</p>;
  const rnd = units.filter((u) => u.kind === "random").map((u) => u.z);
  const floor = rnd.length ? Math.max(...rnd) : -Infinity;
  const chance = censusChance(units);
  const ql = q.trim().toLowerCase();
  const shown = units.filter(
    (u) => (controls || u.kind === "neuron") && (!ql || u.words.some((w) => w.includes(ql)) || u.name.toLowerCase().includes(ql) || String(u.n) === ql),
  );
  return (
    <>
      <CensusDepth />
      <div className="row" style={{ gap: 10, padding: "4px 0" }}>
        <input className="input-line" placeholder="filter by word or name" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1 }} />
        <label className="check atlas-small">
          <input type="checkbox" checked={controls} onChange={(e) => setControls(e.target.checked)} /> controls
        </label>
      </div>
      <ol className="atlas-rows">
        <li className="atlas-rows-head">
          layer {layer + 1}'s census · ranked by the mass of its words against random directions
          {chance
            ? ` · ${chance.above} above every control, about ${chance.chance.toFixed(0)} of them by chance alone · ${doses}`
            : " · no random controls, so nothing here is set against chance"}
        </li>
        {shown.slice(0, 600).map((u) => (
          <li
            key={`${u.kind}/${u.n}`}
            className={`${u.kind}${u.z <= floor ? " below" : ""}${f && f.kind === "neuron" && f.layer === layer && f.n === u.n && u.kind === "neuron" ? " sel" : ""}`}
            onClick={() => u.kind === "neuron" && goToNeuron(layer, u.n)}
            onMouseEnter={() => u.kind === "neuron" && theMap()?.hover(layer, u.n)}
            onMouseLeave={() => theMap()?.hover(null)}
          >
            <span className="atlas-id-small">
              {u.kind === "random" ? "r" : "N"}
              {u.n}
            </span>
            <span className="atlas-z">{u.z.toFixed(1)}</span>
            <span className="atlas-w">
              {u.name ? <NameMark layer={layer} n={u.n} name={u.name} /> : null}
              {u.words.join(" ")}
              {(u.collapse_mass ?? 0) > 0 ? (
                <span className="muted" title="This share of its mass comes from words random pushes bring as hard at ±2, which are not shown">
                  {" "}
                  · {Math.round((u.collapse_mass ?? 0) * 100)}% of its mass from words not shown
                </span>
              ) : null}
            </span>
          </li>
        ))}
      </ol>
    </>
  );
};

const RunRows: React.FC<{ rows: RunRow[]; total: number; provisional: boolean }> = ({ rows, total, provisional }) => (
  <ol className="atlas-rows">
    <li className="atlas-rows-head">
      {total} neurons pass · score = activation over its usual extreme on that side
      {provisional
        ? " · usual extremes read over the study’s 36 texts, provisional until the stats index is built: a neuron those texts never woke scores high on little"
        : ""}
    </li>
    {rows.map((x) => (
      <li key={`${x.layer}/${x.n}`} onClick={() => goToNeuron(x.layer, x.n)} onMouseEnter={() => theMap()?.hover(x.layer, x.n)} onMouseLeave={() => theMap()?.hover(null)}>
        <span className="atlas-id-small">{cite(x.layer, x.n)}</span>
        <span className={`atlas-sg ${x.face === "+" ? "up" : "down"}`}>
          {x.face === "+" ? "" : "−"}
          {x.score.toFixed(1)}×
        </span>
        <span className="atlas-w">
          <span className="atlas-tk">{wp(x.token)}</span> <FaceLine layer={x.layer} face={x.face} w={x.words} k={4} />
        </span>
      </li>
    ))}
  </ol>
);

const RunTab: React.FC = () => {
  const run = useAtlas((s) => s.run);
  const query = useAtlas((s) => s.query);
  const qMode = useAtlas((s) => s.qMode);
  const timer = useRef<number | undefined>(undefined);
  const upd = (patch: Partial<typeof run>) => {
    useAtlas.setState({ run: { ...useAtlas.getState().run, ...patch } });
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void runView(), 200);
  };
  return (
    <div className="atlas-run">
      <textarea
        className="atlas-paste"
        rows={5}
        placeholder="paste a text: see which neurons fire hardest on each word-piece"
        value={qMode === "fires" ? query : ""}
        onChange={(e) => setQuery(e.target.value, "fires")}
        spellCheck={false}
      />
      <Rubric style={{ fontSize: 14, margin: "2px 0" }}>
        Hover a word-piece to see what fires there, click it to keep it; the map lights the neurons that pass the filters and the links among the
        strongest of them.
      </Rubric>
      {run.busy ? <p className="muted atlas-small">running the text…</p> : null}
      <ErrorLine error={run.error} />
      {run.tokens.length ? (
        <div className="atlas-toks" onMouseLeave={() => upd({ hover: null })}>
          {run.tokens.map((t, i) => (
            <button
              key={i}
              type="button"
              className={run.pin === run.pos[i] ? "on" : ""}
              onMouseEnter={() => upd({ hover: run.pos[i] })}
              onClick={() => upd({ pin: run.pin === run.pos[i] ? -1 : run.pos[i] })}
            >
              {t.replace(/\n/g, "↵") || "␣"}
            </button>
          ))}
        </div>
      ) : null}
      {run.text ? (
        <div className="atlas-filters row" style={{ gap: 8, flexWrap: "wrap" }}>
          <span className="atlas-lab">layers</span>
          <input className="input-num" value={run.lo} inputMode="numeric" aria-label="from layer" onChange={(e) => upd({ lo: Number(e.target.value) || 1 })} />
          <span>–</span>
          <input className="input-num" value={run.hi} inputMode="numeric" aria-label="to layer" onChange={(e) => upd({ hi: Number(e.target.value) || 1 })} />
          <span className="atlas-seg">
            {(["both", "+", "-"] as const).map((s) => (
              <button key={s} type="button" aria-pressed={run.sign === s} onClick={() => upd({ sign: s })}>
                {s === "both" ? "±" : s === "+" ? "+" : "−"}
              </button>
            ))}
          </span>
          <span className="atlas-lab">score ≥</span>
          <input className="input-num" value={run.min} inputMode="decimal" aria-label="smallest score" onChange={(e) => upd({ min: Number(e.target.value) || 0 })} />
          <input className="input-line" placeholder="its words hold…" value={run.q} onChange={(e) => upd({ q: e.target.value })} style={{ width: 130 }} />
        </div>
      ) : null}
      {run.view ? <RunRows rows={run.view.results} total={run.view.total} provisional={run.scaleSource?.kind === "study_texts"} /> : null}
    </div>
  );
};

export const LeftPanel: React.FC = () => {
  const tab = useAtlas((s) => s.leftTab);
  const hidden = useAtlas((s) => s.leftHidden);
  const search = useAtlas((s) => s.search);
  const censusSearch = useAtlas((s) => s.censusSearch);
  const searchError = useAtlas((s) => s.searchError);
  const qMode = useAtlas((s) => s.qMode);
  if (hidden) return null;
  return (
    <aside className="atlas-panel atlas-left" aria-label="the open neuron's web, the layer's census, the run of a text">
      <FocusHead />
      <Trail />
      <nav className="atlas-tabs" role="tablist">
        {(["web", "run"] as const).map((t) => (
          <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setLeftTab(t)}>
            {t === "web" ? "web" : "run a text"}
          </button>
        ))}
        <button type="button" className="atlas-edge" aria-label="hide this panel" onClick={() => useAtlas.setState({ leftHidden: true })}>
          ‹
        </button>
      </nav>
      {tab === "web" ? (
        <div className="atlas-pane">
          <HeadWeb />
          <NeuronWeb />
          <ErrorLine error={searchError} />
          {censusSearch && qMode === "census" ? (
            <CensusSearchRows r={censusSearch} />
          ) : search && qMode === "lens" ? (
            search.ready ? (
              <SearchRows rows={search.results} held={search.held} total={search.total} tokens={search.tokens} />
            ) : (
              <p className="muted atlas-small">Neither the lens index nor the steer-lens index is built, so nothing can be searched yet.</p>
            )
          ) : (
            <CensusRows />
          )}
        </div>
      ) : (
        <div className="atlas-pane">
          <RunTab />
        </div>
      )}
    </aside>
  );
};
