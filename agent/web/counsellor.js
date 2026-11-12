// Counsellor insights: every student chat with its SARP-Net prediction, SHAP drivers and evidence.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const when = (ts) => new Date(ts * 1000).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const FIELD_NAMES = { due_diligence: "research level", intake: "intake", budget: "budget", gpa: "result", country: "destination" };
let current = null;

function pin() { try { return sessionStorage.getItem("sarp.pin") || ""; } catch { return ""; } }

async function api(path) {
  const res = await fetch(path, { headers: { "X-Counsellor-Pin": pin() } });
  if (res.status === 401) { askPin(); throw new Error("pin"); }
  if (res.status === 503) {
    $("app").innerHTML = '<p class="c-empty">The counsellor page is turned off until a counsellor PIN is set (SARP_COUNSELLOR_PIN in the Vercel project settings).</p>';
    throw new Error("disabled");
  }
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
}

function askPin() {
  $("app").innerHTML = `<form class="pin-form panel" id="pinForm">
      <h3>Counsellor PIN</h3>
      <input type="password" name="pin" autocomplete="current-password" aria-label="Counsellor PIN" class="chip" required>
      <button class="btn primary" type="submit">Open insights</button></form>`;
  $("pinForm").addEventListener("submit", (e) => {
    e.preventDefault();
    try { sessionStorage.setItem("sarp.pin", e.target.pin.value); } catch { /* ignore */ }
    location.reload();
  });
}

function tier(level) {
  const label = { high: "High risk", medium: "Medium risk", low: "Low risk" }[level] || "Profile incomplete";
  return `<span class="tier ${level || "none"}">${label}</span>`;
}

async function loadList() {
  let data;
  try { data = await api("/api/counsellor/conversations"); } catch { return; }
  const card = data.model_card;
  const providers = (data.status.providers || [])
    .map((p) => `${p.name === "primary" ? "team PC" : "backup"} ${p.model} ${p.available ? "up" : "down"}`).join(", ");
  $("modelCard").textContent = `SARP-Net ${card.model}: ${pct(card.test_accuracy)} accuracy and ROC-AUC ${card.test_roc_auc.toFixed(3)} on held-out records. `
    + `Phase: ${data.political_phase}. AI: ${providers || (data.status.mode === "ai" ? data.status.model : "offline mode")}.`;
  $("list").innerHTML = data.conversations.length ? data.conversations.map((c) => `
    <button class="c-item${c.id === current ? " active" : ""}" data-id="${c.id}" type="button">
      <span class="row"><span class="t">${esc(c.title)}</span>${tier(c.risk_level)}</span>
      <span class="row meta"><span>${when(c.updated)}</span><span>${c.message_count} messages</span>
        ${c.probability != null ? `<span>${pct(c.probability)}</span>` : ""}
        ${c.handoff ? '<span class="flag">Call-back requested</span>' : ""}</span>
    </button>`).join("") : '<p class="small" style="margin:0 8px">No student chats yet.</p>';
}

$("list")?.addEventListener("click", (e) => {
  const item = e.target.closest(".c-item");
  if (item) { current = item.dataset.id; loadList(); loadDetail(current); }
});

async function loadDetail(id) {
  let d;
  try { d = await api(`/api/counsellor/conversations/${id}`); } catch { return; }
  const a = d.assessment || {};
  const view = a.counsellor_view;
  const items = d.plan.items;
  const profileRows = [["Destination", items.country], ["Course", items.course], ["University", items.institution],
    ["Intake", items.intake], ["Budget", items.budget], ["Result", items.gpa], ["Research", items.research]]
    .map(([k, v]) => `<dt>${k}</dt><dd>${esc(v || "Not shared")}</dd>`).join("");

  let prediction;
  if (view) {
    const cut = view.cutoffs;
    const maxShap = Math.max(...view.drivers.map((x) => Math.abs(x.shap)), 0.01);
    const whatIf = Object.entries(view.due_diligence_what_if.probability_if)
      .map(([lvl, p]) => `<dt>${esc(lvl)}</dt><dd>${pct(p)}</dd>`).join("");
    prediction = `
      <section class="panel">
        <h3>SARP-Net continuation probability</h3>
        <div class="prob"><b>${pct(view.probability)}</b>${tier(view.risk_level)}</div>
        <div class="bar" aria-hidden="true"><i style="width:${view.probability * 100}%"></i>
          <span class="cut" style="left:${cut.medium_risk_from * 100}%"></span><span class="cut" style="left:${cut.low_risk_from * 100}%"></span></div>
        <p class="small">${esc(view.risk_tier)}. Medium risk from ${cut.medium_risk_from}, low risk from ${cut.low_risk_from} (calibrated cut-offs).</p>
        ${whatIf ? `<h3 style="margin-top:14px">If the student's research level rose (now ${esc(view.due_diligence_what_if.current)})</h3><dl class="kv">${whatIf}</dl>` : ""}
        ${a.warnings?.length ? `<h3 style="margin-top:14px">Notes on this prediction</h3>${a.warnings.map((w) => `<p class="warn">${esc(w.message)}</p>`).join("")}` : ""}
      </section>
      <section class="panel">
        <h3>What drives this prediction (SHAP, with project evidence)</h3>
        ${view.drivers.map((x) => `
          <div class="driver">
            <div class="row"><span class="name">${esc(x.label)}</span>
              <span class="small">${x.direction}${x.actionable ? "" : ", not changeable by counselling"}</span>
              <span class="shap" aria-label="SHAP ${x.shap}"><i class="${x.shap < 0 ? "neg" : "pos"}" style="width:${(Math.abs(x.shap) / maxShap) * 50}%"></i></span></div>
            <ul>${x.evidence.map((ev) => `<li>${esc(ev)}</li>`).join("")}</ul>
          </div>`).join("")}
      </section>`;
  } else {
    const missing = (a.missing || []).map((f) => FIELD_NAMES[f] || f).join(", ");
    prediction = `<section class="panel"><h3>SARP-Net continuation probability</h3>
      <p class="small">Not scored yet. The agent is still collecting: ${esc(missing || "details")}.</p></section>`;
  }
  const h = d.handoff;
  $("detail").innerHTML = `<div class="wrap">
    <div class="c-head"><h2>${esc(d.title)}</h2>${tier(view?.risk_level)}<span class="small">Updated ${when(d.updated)}, language: ${esc(d.language || "unknown")}</span></div>
    <div class="c-grid">
      <div>${prediction}</div>
      <div>
        ${h ? `<section class="panel"><h3>Call-back requested</h3><dl class="kv"><dt>Name</dt><dd>${esc(h.name)}</dd><dt>Contact</dt><dd>${esc(h.contact)}</dd><dt>Best time</dt><dd>${esc(h.preferred_time || "Any time")}</dd><dt>Requested</dt><dd>${when(h.requested_at)}</dd></dl></section>` : ""}
        <section class="panel"><h3>Student plan</h3><dl class="kv">${profileRows}</dl></section>
        <section class="panel transcript"><h3>Transcript</h3>${d.messages.map((m) => `
          <div class="m ${m.role}"><div class="who">${m.role === "user" ? "Student" : "SARP-Agent"}${m.feedback ? (m.feedback > 0 ? ", marked helpful" : ", marked not helpful") : ""}</div>${esc(m.content)}</div>`).join("")}
        </section>
      </div>
    </div></div>`;
}

loadList();
setInterval(() => { loadList(); if (current) loadDetail(current); }, 10000);
