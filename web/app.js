const $ = (s, el = document) => el.querySelector(s);
const TYPES = ["normal","fire","water","electric","grass","ice","fighting","poison","ground","flying","psychic","bug","rock","ghost","dragon","dark","steel","fairy"];
const REGIONS = ["kanto","johto","hoenn","sinnoh","unova","kalos","alola","galar","hisui","paldea"];
const EXAMPLES = [
  "How do I counter Garchomp, Dragonite and Tyranitar?",
  "Build me a Paldea team around Garchomp",
  "My team is Charizard, Gyarados, Pikachu and Snorlax. How do I beat Gengar?",
  "Who is the fastest non-legendary Water type?",
  "Which Pokémon dies if the flame on its tail goes out?",
  "Tell me about the Hoenn region",
  "What does Intimidate do and who has it?",
  "What are Eevee's evolutions?",
];

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const typeBadges = types => `<div class="types">${types.map(t => `<span class="t t-${t}">${t}</span>`).join("")}</div>`;
const sprite = p => p.sprite || p.artwork || "";

function pkCard(p, extra = "") {
  return `<div class="pk">
    ${sprite(p) ? `<img src="${esc(sprite(p))}" alt="" loading="lazy">` : ""}
    <div><div class="name">${esc(p.display)}</div>
      <div class="meta">#${p.dex_id} · BST ${p.bst}</div>${typeBadges(p.types)}${extra}</div>
  </div>`;
}

// Minimal Markdown: paragraphs, bullet/numbered lists, bold, italics, inline code, [citations].
function md(text) {
  const inline = s => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]{2,40})\]/g, '<span class="cite">$1</span>');
  const out = []; let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const m = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (m) { if (!list) { list = []; } list.push(`<li>${inline(m[1])}</li>`); continue; }
    if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
    if (line.trim()) out.push(`<p>${inline(line.replace(/^#+\s*/, ""))}</p>`);
  }
  if (list) out.push(`<ul>${list.join("")}</ul>`);
  return out.join("");
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`));
}));

fetch("/api/health").then(r => r.json()).then(h => {
  $("#count").textContent = h.pokemon.toLocaleString();
  $("#llm-badge").textContent = h.provider === "none" ? "retrieval only" : `${h.provider} · ${h.model}`;
}).catch(() => { $("#llm-badge").textContent = "offline"; });

// ---------- chat ----------
const history = [];
$("#examples").innerHTML = EXAMPLES.map(q => `<button class="chip" type="button">${esc(q)}</button>`).join("");
$("#examples").addEventListener("click", e => {
  if (e.target.matches(".chip")) { $("#question").value = e.target.textContent; $("#ask-form").requestSubmit(); }
});

function showMeta(meta) {
  const u = meta.understood;
  const tags = [`intent: ${u.intent}`];
  if (u.opponents?.length) tags.push(`opponents: ${u.opponents.length}`);
  if (u.my_team?.length) tags.push(`my team: ${u.my_team.length}`);
  if (u.types?.length) tags.push(`types: ${u.types.join(", ")}`);
  if (u.region) tags.push(`region: ${u.region}`);
  if (u.stat) tags.push(`stat: ${u.stat}${u.descending ? " ↓" : " ↑"}`);
  $("#understood").innerHTML = tags.map(t => `<span>${esc(t)}</span>`).join("");
  $("#source-cards").innerHTML =
    meta.sources.map(p => pkCard(p)).join("") +
    meta.other_sources.map(s => `<div class="doc-chip">${esc(s)}</div>`).join("") ||
    `<p class="muted">No sources.</p>`;
  $("#ctx").textContent = meta.context;
  $("#ctx-details").hidden = false;
}

$("#ask-form").addEventListener("submit", async e => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (!q) return;
  $("#question").value = "";
  $(".empty")?.remove();
  const box = $("#messages");
  box.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(q)}</div>`);
  const bot = document.createElement("div");
  bot.className = "msg bot typing";
  box.appendChild(bot);
  box.scrollTop = box.scrollHeight;
  $("#send").disabled = true;

  let text = "";
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: q, history: history.slice(-6)}),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "meta") showMeta(ev);
        else if (ev.type === "token") { text += ev.text; bot.innerHTML = md(text); box.scrollTop = box.scrollHeight; }
        else if (ev.type === "error") bot.insertAdjacentHTML("beforeend",
          `<div class="msg err">${esc(ev.message)} The retrieved context is shown in the Sources panel.</div>`);
      }
    }
  } catch (err) {
    bot.innerHTML = `<div class="msg err">${esc(err.message)}</div>`;
  }
  bot.classList.remove("typing");
  history.push({role: "user", content: q}, {role: "assistant", content: text});
  $("#send").disabled = false;
  $("#question").focus();
});

// ---------- slots with autocomplete ----------
function makeSlots(el, n = 6) {
  el.innerHTML = Array.from({length: n}, () => `
    <div class="slot"><div class="ph"></div><input placeholder="Add Pokémon" autocomplete="off"></div>`).join("");
  el.querySelectorAll(".slot").forEach(slot => {
    const input = $("input", slot);
    let timer, items = [], sel = -1;
    const close = () => { $(".ac", slot)?.remove(); items = []; sel = -1; };
    const pick = p => {
      input.value = p.display; input.dataset.name = p.name;
      slot.querySelector(".ph, img").outerHTML = `<img src="${esc(sprite(p))}" alt="">`;
      close();
    };
    input.addEventListener("input", () => {
      delete input.dataset.name;
      clearTimeout(timer);
      const q = input.value.trim();
      if (!q) { close(); slot.querySelector("img")?.replaceWith(Object.assign(document.createElement("div"), {className: "ph"})); return; }
      timer = setTimeout(async () => {
        items = await (await fetch(`/api/pokemon?q=${encodeURIComponent(q)}`)).json();
        close();
        if (!items.length) return;
        const ac = document.createElement("div");
        ac.className = "ac";
        ac.innerHTML = items.map(p => `<div><img src="${esc(sprite(p))}" alt="">${esc(p.display)}</div>`).join("");
        ac.querySelectorAll("div").forEach((d, i) => d.addEventListener("mousedown", ev => { ev.preventDefault(); pick(items[i]); }));
        slot.appendChild(ac);
        sel = -1;
      }, 120);
    });
    input.addEventListener("keydown", e => {
      const rows = slot.querySelectorAll(".ac div");
      if (!rows.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length;
        rows.forEach((r, i) => r.classList.toggle("sel", i === sel));
      } else if (e.key === "Enter") { e.preventDefault(); pick(items[Math.max(sel, 0)]); }
      else if (e.key === "Escape") close();
    });
    input.addEventListener("blur", () => setTimeout(close, 100));
  });
  el.setSlots = list => {
    el.querySelectorAll(".slot").forEach((slot, i) => {
      const p = list[i], input = $("input", slot);
      if (p) {
        input.value = p.display; input.dataset.name = p.name;
        slot.querySelector(".ph, img").outerHTML = `<img src="${esc(sprite(p))}" alt="">`;
      }
    });
  };
}
const slotValues = el => [...el.querySelectorAll("input")].map(i => i.dataset.name || i.value.trim()).filter(Boolean);
function clearSlots(el) { makeSlots(el); }

document.querySelectorAll(".filters").forEach(f => {
  f.innerHTML = `
    <select class="region"><option value="">Any region</option>${REGIONS.map(r => `<option value="${r}">${r[0].toUpperCase() + r.slice(1)}</option>`).join("")}</select>
    <label><input type="checkbox" class="legend"> Legendaries</label>
    <label><input type="checkbox" class="mega"> Megas</label>`;
});
const filtersOf = name => {
  const f = $(`.filters[data-for="${name}"]`);
  return {region: $(".region", f).value || null, allow_legendary: $(".legend", f).checked, allow_mega: $(".mega", f).checked};
};

async function post(url, body) {
  const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
  return data;
}

const fmtMult = m => m === 0 ? "0" : m === 0.25 ? "¼" : m === 0.5 ? "½" : m === 1 ? "" : `${m}`;
const multClass = m => ({4: "m4", 2: "m2", 1: "m1", 0.5: "m05", 0.25: "m025", 0: "m0"}[m] || "m1");

function heatmap(analysis) {
  const names = analysis.team.map(p => p.display);
  const rows = names.map(n => `<tr><th>${esc(n)}</th>${TYPES.map(t => {
    const m = analysis.defense[t].multipliers[n];
    return `<td class="${multClass(m)}">${fmtMult(m)}</td>`;
  }).join("")}</tr>`).join("");
  const sum = `<tr class="sum"><th>Weak / resist</th>${TYPES.map(t => {
    const d = analysis.defense[t];
    return `<td title="${d.weak.length} weak, ${d.resist.length} resist" style="color:${d.weak.length > d.resist.length ? "var(--bad)" : "inherit"}">${d.weak.length}/${d.resist.length}</td>`;
  }).join("")}</tr>`;
  return `<div class="heat"><table><tr><th></th>${TYPES.map(t => `<th class="rot"><span class="t t-${t}">${t}</span></th>`).join("")}</tr>${rows}${sum}</table></div>`;
}

function teamReport(analysis, suggestions, title) {
  const flags = [];
  analysis.shared_weaknesses.forEach(w => flags.push(`<div class="flag bad"><strong>${w.type}</strong> hits ${w.weak.join(", ")} super-effectively${w.resist.length ? `; only ${w.resist.join(", ")} resist` : "; nobody resists it"}.</div>`));
  if (!analysis.shared_weaknesses.length) flags.push(`<div class="flag good">No stacked weaknesses: every type that hits two members is resisted by someone.</div>`);
  if (analysis.coverage_gaps.length) flags.push(`<div class="flag bad">No super-effective move against: ${analysis.coverage_gaps.join(", ")}.</div>`);
  else flags.push(`<div class="flag good">The team can hit every type super-effectively.</div>`);
  if (analysis.unresisted_types.length) flags.push(`<div class="flag">Nobody resists: ${analysis.unresisted_types.join(", ")}.</div>`);
  flags.push(`<div class="flag">Physical: ${analysis.physical.join(", ") || "none"} · Special: ${analysis.special.join(", ") || "none"} · Avg Speed ${analysis.avg_speed}</div>`);
  analysis.notes.forEach(n => flags.push(`<div class="flag">${esc(n)}</div>`));
  return `
    ${title ? `<div class="card"><h3>${esc(title)}</h3><div class="grid2">${analysis.team.map(p => pkCard(p)).join("")}</div></div>` : ""}
    <div class="card"><h3>Defensive matchups (damage taken)</h3>${heatmap(analysis)}</div>
    <div class="card"><h3>Findings</h3>${flags.join("")}</div>
    ${suggestions?.length ? `<div class="card"><h3>Suggested additions</h3><div class="grid2">${suggestions.map(s => pkCard(s, `<div class="reason">${esc(s.reason)}</div>`)).join("")}</div></div>` : ""}`;
}

// ---------- team builder ----------
const teamSlots = $("#team-slots");
makeSlots(teamSlots);
$("#team-analyze").addEventListener("click", async () => {
  const out = $("#team-out");
  try {
    const d = await post("/api/team/analyze", {team: slotValues(teamSlots), ...filtersOf("team")});
    out.innerHTML = teamReport(d.analysis, d.suggestions);
  } catch (e) { out.innerHTML = `<div class="err-box">${esc(e.message)}</div>`; }
});
$("#team-build").addEventListener("click", async () => {
  const out = $("#team-out");
  try {
    const d = await post("/api/team/build", {team: slotValues(teamSlots), ...filtersOf("team")});
    teamSlots.setSlots(d.team);
    const added = d.added.length ? `<div class="card"><h3>Why these were added</h3>${d.added.map(a => `<div class="flag good"><strong>${esc(a.display)}</strong>: ${esc(a.reason)}</div>`).join("")}</div>` : "";
    out.innerHTML = added + teamReport(d.analysis, []);
  } catch (e) { out.innerHTML = `<div class="err-box">${esc(e.message)}</div>`; }
});
$("#team-clear").addEventListener("click", () => { clearSlots(teamSlots); $("#team-out").innerHTML = ""; });

// ---------- counters ----------
const oppSlots = $("#opp-slots"), mineSlots = $("#mine-slots");
makeSlots(oppSlots); makeSlots(mineSlots);
$("#counter-go").addEventListener("click", async () => {
  const out = $("#counter-out");
  try {
    const d = await post("/api/counters", {opponents: slotValues(oppSlots), team: slotValues(mineSlots), ...filtersOf("counter")});
    const types = d.best_attacking_types.slice(0, 5).map(x =>
      `<div class="flag"><span class="t t-${x.type}">${x.type}</span> super effective on ${x.super_effective_on.join(", ")}${x.resisted_by.length ? ` · resisted by ${x.resisted_by.join(", ")}` : ""}</div>`).join("");
    const per = Object.entries(d.per_opponent).map(([opp, rows]) => `
      <div class="card"><h3>Counters to ${esc(opp)}</h3><div class="grid2">
        ${rows.map(r => pkCard(r, `<div class="reason">${esc(r.reason)}</div>`)).join("")}</div></div>`).join("");
    let mine = "";
    if (d.my_team_plan) {
      mine = `<div class="card"><h3>Using your team</h3>${Object.entries(d.my_team_plan).map(([opp, rows]) =>
        `<div class="flag ${d.no_safe_answer.includes(opp) ? "bad" : "good"}"><strong>vs ${esc(opp)}:</strong> ${rows.map(r => `${esc(r.pokemon)} (${esc(r.reason)})`).join("; ")}</div>`).join("")}
        ${d.no_safe_answer.length ? `<p class="muted">No safe super-effective answer on your team for ${esc(d.no_safe_answer.join(", "))}. Consider one of the counters below.</p>` : ""}</div>`;
    }
    const wide = d.opponents.length > 1 ? `<div class="card"><h3>Good against the whole team</h3><div class="grid2">${d.team_wide.map(r =>
      pkCard(r, `<div class="reason">${r.checks.length ? "Safely beats " + esc(r.checks.join(", ")) : "Strong overall matchups"}</div>`)).join("")}</div></div>` : "";
    out.innerHTML = `<div class="card"><h3>Best attacking types</h3>${types}</div>${mine}${wide}${per}`;
  } catch (e) { out.innerHTML = `<div class="err-box">${esc(e.message)}</div>`; }
});
$("#counter-clear").addEventListener("click", () => { clearSlots(oppSlots); clearSlots(mineSlots); $("#counter-out").innerHTML = ""; });
