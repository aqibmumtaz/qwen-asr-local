"use strict";
const $ = (id) => document.getElementById(id);
let sid = null, scenarios = [], guided = null, lastState = null, mode = "chat", room = null;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) { const t = await r.text(); throw new Error(`${r.status} ${t}`); }
  return r.json();
}
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));

function bubble(kind, text, meta) {
  const d = document.createElement("div");
  d.className = "msg " + kind;
  d.textContent = text;
  if (meta) { const m = document.createElement("div"); m.className = "meta"; m.innerHTML = meta; d.appendChild(m); }
  $("log").appendChild(d);
  $("log").scrollTop = $("log").scrollHeight;
}
const chip = (t, cls = "") => `<span class="chip ${cls}">${esc(t)}</span>`;

function renderState(state) {
  lastState = state;
  const a = state.active;
  const rows = [
    ["Active", a ? a.workflow : "none"], ["Step", a ? a.step : "-"], ["Waiting for", (a && a.pending) || state.pending || "-"],
    ["Paused", state.suspended.length ? state.suspended.map((s) => s.workflow).join(", ") : "-"],
    ["Ended", state.ended ? "yes" + (state.transfer ? " (" + state.transfer + ")" : "") : "no"],
  ];
  $("flow").innerHTML = rows.map(([k, v]) => `<div>${esc(k)}</div><div>${esc(v)}</div>`).join("");
  const slots = a ? a.slots : {};
  const f = state.facts || {};
  const items = Object.entries(slots).map(([k, v]) => [k, fmtSlot(v)]);
  Object.entries(f).forEach(([k, v]) => items.push(["shared." + k, v]));
  $("slots").innerHTML = items.length ? items.map(([k, v]) => `<div>${esc(k)}</div><div>${esc(v)}</div>`).join("") : '<div class="muted">nothing collected yet</div>';
}
function fmtSlot(v) {
  if (Array.isArray(v)) return v.map((x) => x.name || x).join(", ");
  if (v && typeof v === "object") return v.text ? `${v.area || ""} (${v.text})` : (v.day ? `${v.day} ${v.part}` : JSON.stringify(v));
  return v;
}
function renderTurn(r) {
  renderState(r.state);
  const tried = (r.route.tried || []).map((t) =>
    `<div class="trace">${esc(t.source)}: <b>${esc(t.decision || "-")}</b>${t.workflow ? " → " + esc(t.workflow) : ""}` +
    `${t.confidence !== undefined ? " (" + t.confidence + ")" : ""}${t.ms !== undefined ? " " + t.ms + "ms" : ""}${t.note ? " <span class='muted'>" + esc(t.note) + "</span>" : ""}</div>`).join("");
  $("router").innerHTML = `<div>Decision: <b>${esc(r.route.decision)}</b>${r.route.workflow ? " → " + esc(r.route.workflow) : ""} <span class="muted">via ${esc(r.route.source)}</span></div>${tried}`;
  const tools = r.events.filter((e) => e.kind === "tool");
  $("tools").innerHTML = tools.length ? tools.map((t) => `<div class="tool"><b>${esc(t.name)}</b> <span class="muted mono">${esc(JSON.stringify(t.args || t.items || t.result || t.id || ""))}</span></div>`).join("") : '<span class="muted">none this turn</span>';
  $("timings").innerHTML = Object.entries(r.timings).map(([k, v]) => `${esc(k)}: <b>${v}</b> ms`).join("<br>");
  $("raw").textContent = JSON.stringify(r, null, 2);
}

async function newCall(scenarioId) {
  $("log").innerHTML = "";
  if (mode === "voice" && !scenarioId) {
    await stopVoice();
    try { await startVoice(); bubble("sys", "Voice call started"); } catch (e) { bubble("sys", "Voice error: " + e.message); }
    $("endCall").disabled = false;
    return;
  }
  const res = await api("/api/session", {caller_phone: $("callerPhone").value || null, scenario_id: scenarioId || null, greet: true});
  sid = res.session_id;
  renderState(res.state);
  if (res.greeting) bubble("bot", res.greeting);
  bubble("sys", `Call ${sid} started` + (res.scenario ? ` — scenario: ${res.scenario.title}` : ""));
  $("endCall").disabled = false;
  guided = res.scenario ? {steps: res.scenario.steps, i: 0, title: res.scenario.title} : null;
  showGuided();
  $("msg").focus();
}
function showGuided() {
  $("guided").classList.toggle("hidden", !guided);
  if (!guided) return;
  const done = guided.i >= guided.steps.length;
  $("guidedTitle").textContent = `Scenario: ${guided.title} — step ${Math.min(guided.i + 1, guided.steps.length)} of ${guided.steps.length}`;
  $("guidedLine").value = done ? "" : guided.steps[guided.i];
  $("guidedSend").disabled = done;
}
async function send(text, fromGuided) {
  if (!sid) await newCall();
  if (!text.trim()) return;
  bubble("user", text);
  let r;
  try { r = await api("/api/turn", {session_id: sid, text}); }
  catch (e) { bubble("sys", "Error: " + e.message); return; }
  const meta = chip(r.route.decision + (r.route.workflow ? "→" + r.route.workflow : "")) + chip(r.route.source) + chip(r.timings.total_ms + " ms");
  bubble("bot", r.text, meta);
  renderTurn(r);
  if (r.check) {
    const el = $("guidedChecks");
    el.innerHTML = (r.check.scripted ? (r.check.ok ? `<span class="pass">✓ step ${r.check.step + 1} matched expectations</span>` : `<span class="fail">✗ step ${r.check.step + 1}: ${r.check.failures.map(esc).join("; ")}</span>`) : '<span class="muted">free line (not the scripted one), not checked</span>') + "<br>" + el.innerHTML;
    if (r.check.scripted && guided) { guided.i += 1; showGuided(); }
  }
  if (r.scenario_done) {
    bubble("sys", r.scenario_done.ok ? "✓ Scenario passed" : "✗ Scenario failed: " + r.scenario_done.failures.join("; "));
  }
  if (r.ended) { bubble("sys", r.transfer ? `Call transferred to a person (${r.transfer})` : "Call ended"); $("endCall").disabled = true; }
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("msg").value; $("msg").value = ""; send(t); });
$("guidedSend").addEventListener("click", () => send($("guidedLine").value, true));
$("newCall").addEventListener("click", () => newCall());
$("endCall").addEventListener("click", async () => { await stopVoice(); bubble("sys", "Call ended by tester"); sid = null; $("endCall").disabled = true; });
function setMode(m) {
  mode = m;
  $("modeChat").classList.toggle("on", m === "chat");
  $("modeVoice").classList.toggle("on", m === "voice");
  $("msg").disabled = m === "voice";
  $("voiceBar").classList.toggle("hidden", m !== "voice");
}
$("modeChat").addEventListener("click", async () => { await stopVoice(); setMode("chat"); });
$("modeVoice").addEventListener("click", () => { setMode("voice"); bubble("sys", "Voice mode: press New call to join a LiveKit room; the agent greets you and listens to your microphone."); });

async function startVoice() {
  if (!window.LivekitClient) { bubble("sys", "livekit-client failed to load (check the internet connection)."); return; }
  const t = await api("/api/token", {});
  const {Room, RoomEvent, Track} = LivekitClient;
  room = new Room();
  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) { const el = track.attach(); el.autoplay = true; document.body.appendChild(el); }
  });
  room.on(RoomEvent.DataReceived, (payload, _p, _k, topic) => {
    if (topic !== "bot-turn") return;
    const r = JSON.parse(new TextDecoder().decode(payload));
    if (r.user) bubble("user", r.user);
    bubble("bot", r.text, chip(r.route.decision + (r.route.workflow ? "→" + r.route.workflow : "")) + chip(r.route.source));
    renderTurn(r);
    if (r.ended) { bubble("sys", r.transfer ? `Call transferred to a person (${r.transfer})` : "Call ended"); }
  });
  room.on(RoomEvent.Disconnected, () => { $("voiceBar").textContent = "Voice: disconnected"; });
  const host = location.hostname || "localhost";
  await room.connect(t.url.replace("localhost", host), t.token);
  await room.localParticipant.setMicrophoneEnabled(true);
  $("voiceBar").textContent = `Voice: connected to room ${t.room}. Speak normally; the bot answers in audio.`;
}
async function stopVoice() {
  if (room) { try { await room.disconnect(); } catch (e) {} room = null; }
  $("voiceBar").textContent = "Voice: not connected";
}

async function loadScenarios() {
  scenarios = await api("/api/scenarios");
  const tags = ["all", ...new Set(scenarios.flatMap((s) => s.tags))];
  $("tagSel").innerHTML = tags.map((t) => `<option>${esc(t)}</option>`).join("");
  renderList();
}
function renderList(results) {
  const tag = $("tagSel").value || "all";
  const rm = {}; (results || []).forEach((r) => (rm[r.id] = r));
  $("scList").innerHTML = scenarios.filter((s) => tag === "all" || s.tags.includes(tag)).map((s) => {
    const r = rm[s.id];
    const status = r ? (r.ok ? '<span class="pass">✓</span> ' : '<span class="fail">✗</span> ') : "";
    return `<li><div class="t">${status}${esc(s.title)}</div><div class="tags">${esc(s.id)} · ${s.tags.map(esc).join(", ")} · ${s.steps} steps</div>` +
      `<div class="btns"><button data-g="${esc(s.id)}" type="button">Guided</button><button data-a="${esc(s.id)}" type="button">Auto-play</button></div>` +
      (r && !r.ok ? `<div class="small fail">${r.failures.map(esc).join("<br>")}</div>` : "") + "</li>";
  }).join("");
}
$("tagSel").addEventListener("change", () => renderList());
$("scList").addEventListener("click", async (e) => {
  const g = e.target.dataset.g, a = e.target.dataset.a;
  if (g) await newCall(g);
  if (a) await autoPlay(a);
});
async function autoPlay(id) {
  const sc = scenarios.find((s) => s.id === id);
  await newCall(id);
  for (const line of sc.lines) { await send(line); await new Promise((r) => setTimeout(r, 350)); if (lastState && lastState.ended) break; }
}
$("runTag").addEventListener("click", async () => {
  $("runSummary").textContent = "Running…";
  const tag = $("tagSel").value;
  const res = await api("/api/scenarios/run", {tag});
  $("runSummary").innerHTML = `<b class="${res.passed === res.total ? "pass" : "fail"}">${res.passed}/${res.total} passed</b> in ${res.ms} ms`;
  renderList(res.results);
});
$("failNext").addEventListener("click", async () => {
  if (!sid) return bubble("sys", "Start a call first.");
  await api("/api/backend", {session_id: sid, fail_next: "simulated outage"});
  bubble("sys", "Next backend call will fail (simulated outage).");
});
$("blockSlot").addEventListener("click", async () => {
  if (!sid) return bubble("sys", "Start a call first.");
  const d = new Date(); d.setDate(d.getDate() + 1);
  await api("/api/backend", {session_id: sid, block_slot: [d.toISOString().slice(0, 10), "morning"]});
  bubble("sys", "Tomorrow morning is now full.");
});
$("showBookings").addEventListener("click", async () => {
  if (!sid) return;
  const b = await api("/api/bookings/" + sid);
  const el = $("bookings"); el.classList.toggle("hidden"); el.textContent = JSON.stringify(b, null, 2);
});
(async () => {
  try {
    const c = await api("/api/config");
    $("cfg").textContent = `LLM: ${c.llm_enabled ? c.llm_model : "not configured (rules only)"} · router: ${c.router.primary}→${c.router.fallback} (loaded: ${c.router.available.join(", ") || "rules only"}) · backend: ${c.backend_mode}`;
  } catch (e) { $("cfg").textContent = "config unavailable"; }
  await loadScenarios();
  $("endCall").disabled = true;
})();
