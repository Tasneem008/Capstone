// SARP-Agent student chat. Plain ES module, no build step.

const ICONS = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  phone: '<path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2"/>',
  check: '<path d="m5 12 5 5L20 7"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
  up: '<path d="M7 11v9H4v-9zM7 11l4-7a2 2 0 0 1 3 2l-1 5h5a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 16.8 21H7"/>',
  down: '<path d="M17 13V4h3v9zM17 13l-4 7a2 2 0 0 1-3-2l1-5H6a2 2 0 0 1-2-2.3l1.2-6A2 2 0 0 1 7.2 3H17"/>',
  pencil: '<path d="M4 20h4L19 9l-4-4L4 16zM14 6l4 4"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  plane: '<path d="M2 12.5 22 4l-6 17-4-7.5z" fill="currentColor" stroke="none"/>',
  arrowDown: '<path d="M12 5v14M5 12l7 7 7-7"/>',
};
const icon = (name, cls = "") =>
  `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;

const UI = {
  en: {
    newChat: "New chat", replyLanguage: "Reply language", auto: "Detect automatically", talk: "Talk to a counsellor",
    talkDone: "Call-back requested", placeholder: "Ask about courses, costs or visas",
    disclaimer: "SARP-Agent can make mistakes. Check fees and visa rules with your counsellor.",
    today: "Today", yesterday: "Yesterday", week: "Previous 7 days", older: "Older", noChats: "Your chats will appear here.",
    yourPlan: "Your plan", from: "Dhaka", course: "Course", institution: "University", intake: "Intake",
    budget: "Yearly budget", gpa: "Result", research: "Research", planReady: "Plan ready", startOver: "Start over",
    copy: "Copy reply", copied: "Copied", good: "Helpful", bad: "Not helpful", rename: "Rename chat", del: "Delete chat",
    confirmDel: "Delete", stop: "Stop replying", send: "Send message", offline: "Offline answers",
    offlineTip: "The AI model isn't running, so answers come from past counsellor conversations.",
    welcomeTitle: "Where do you want to study?",
    welcomeText: "Ask about countries, courses, costs, documents or visas, in English, Banglish or বাংলা. I'll build your plan as we talk.",
    retry: "Try again", errorNet: "Couldn't reach SARP-Agent. Check that the server is running, then try again.",
    cbTitle: "Talk to a counsellor", cbText: "Leave your details and a counsellor from our Dhaka office will call you back.",
    name: "Your name", contact: "Phone or WhatsApp number", when: "Best time to call", whenAny: "Any time",
    morning: "Morning (10am to 1pm)", afternoon: "Afternoon (1pm to 5pm)", evening: "Evening (5pm to 8pm)",
    cancel: "Cancel", request: "Request a call", requested: "Call-back requested",
    needDetails: "Add your name and a number we can call.", menu: "Open chats", close: "Close chats",
    theme: "Switch between light and dark mode", jump: "Jump to the latest message", replyReady: "Reply ready",
    resetDone: "Plan cleared", deleted: "Chat deleted", requestCall: "Request a call back", stopped: "Stopped",
    rateLimited: "You're sending messages very quickly. Please wait a few minutes, then try again.",
    privacyNote: "Your chats are saved so our counsellors can help you, and may be answered by a partner AI service. Please don't share passwords or bank details.",
    cbPrivacy: "Your number is shared only with our counsellors.",
  },
  bn: {
    newChat: "নতুন চ্যাট", replyLanguage: "উত্তরের ভাষা", auto: "নিজে থেকে বুঝে নিন", talk: "কাউন্সেলরের সাথে কথা বলুন",
    talkDone: "কলব্যাকের অনুরোধ গেছে", placeholder: "কোর্স, খরচ বা ভিসা নিয়ে জিজ্ঞেস করুন",
    disclaimer: "SARP-Agent ভুল করতে পারে। ফি আর ভিসার নিয়ম কাউন্সেলরের কাছে নিশ্চিত করুন।",
    today: "আজ", yesterday: "গতকাল", week: "গত ৭ দিন", older: "আগের", noChats: "আপনার চ্যাটগুলো এখানে দেখা যাবে।",
    yourPlan: "আপনার পরিকল্পনা", from: "ঢাকা", course: "কোর্স", institution: "ইউনিভার্সিটি", intake: "ইনটেক",
    budget: "বছরের বাজেট", gpa: "রেজাল্ট", research: "খোঁজখবর", planReady: "পরিকল্পনা প্রস্তুত", startOver: "নতুন করে শুরু",
    copy: "উত্তর কপি করুন", copied: "কপি হয়েছে", good: "সহায়ক", bad: "সহায়ক নয়", rename: "নাম বদলান", del: "চ্যাট মুছুন",
    confirmDel: "মুছে ফেলুন", stop: "উত্তর থামান", send: "পাঠান", offline: "অফলাইন উত্তর",
    offlineTip: "AI মডেল চালু নেই, তাই উত্তর আসছে আগের কাউন্সেলিং কথোপকথন থেকে।",
    welcomeTitle: "Where do you want to study?",
    welcomeText: "দেশ, কোর্স, খরচ, ডকুমেন্ট বা ভিসা নিয়ে বাংলা, Banglish বা English-এ জিজ্ঞেস করুন। কথা বলতে বলতে আমি আপনার পরিকল্পনা সাজিয়ে দেব।",
    retry: "আবার চেষ্টা করুন", errorNet: "SARP-Agent-এর সাথে যোগাযোগ হচ্ছে না। সার্ভার চালু আছে কিনা দেখে আবার চেষ্টা করুন।",
    cbTitle: "কাউন্সেলরের সাথে কথা বলুন", cbText: "আপনার তথ্য দিন, আমাদের ঢাকা অফিসের একজন কাউন্সেলর আপনাকে ফোন করবেন।",
    name: "আপনার নাম", contact: "ফোন বা হোয়াটসঅ্যাপ নম্বর", when: "কখন ফোন করলে সুবিধা", whenAny: "যেকোনো সময়",
    morning: "সকাল (১০টা থেকে ১টা)", afternoon: "দুপুর (১টা থেকে ৫টা)", evening: "সন্ধ্যা (৫টা থেকে ৮টা)",
    cancel: "বাতিল", request: "ফোন করতে বলুন", requested: "কলব্যাকের অনুরোধ গেছে",
    needDetails: "আপনার নাম আর একটি ফোন নম্বর দিন।", menu: "চ্যাটগুলো খুলুন", close: "বন্ধ করুন",
    theme: "লাইট বা ডার্ক মোড", jump: "সর্বশেষ বার্তায় যান", replyReady: "উত্তর এসেছে",
    resetDone: "পরিকল্পনা মুছে ফেলা হয়েছে", deleted: "চ্যাট মুছে ফেলা হয়েছে", requestCall: "কলব্যাকের অনুরোধ করুন", stopped: "থামানো হয়েছে",
    rateLimited: "আপনি খুব দ্রুত বার্তা পাঠাচ্ছেন। কয়েক মিনিট পরে আবার চেষ্টা করুন।",
    privacyNote: "আমাদের কাউন্সেলররা যাতে সাহায্য করতে পারেন, তাই আপনার চ্যাট সংরক্ষণ করা হয়; কিছু উত্তর সহযোগী AI সেবা থেকেও আসতে পারে। পাসওয়ার্ড বা ব্যাংকের তথ্য শেয়ার করবেন না।",
    cbPrivacy: "আপনার নম্বর শুধু আমাদের কাউন্সেলররা দেখবেন।",
  },
};

const STARTERS = [
  { text: "Which country can I afford with about 15 lakh taka a year?", tag: "English" },
  { text: "What documents do I need for a UK student visa?", tag: "English" },
  { text: "Ami Canada te CSE porte chai, CGPA 3.2. Kivabe shuru korbo?", tag: "Banglish" },
  { text: "অস্ট্রেলিয়ায় মাস্টার্স করতে IELTS-এ কত স্কোর লাগে?", tag: "বাংলা" },
];

// ------------------------------------------------------------------ state
const $ = (id) => document.getElementById(id);
const store = {
  get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const S = {
  clientId: store.get("sarp.client") || (() => { const id = (crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`).replace(/[^a-z0-9-]/gi, ""); store.set("sarp.client", id); return id; })(),
  conversationId: null,
  language: store.get("sarp.language", "auto"),
  streaming: false,
  controller: null,
  pinned: true,
  plan: null,
  planComplete: false,
  callbackDone: false,
  conversations: [],
};
const t = (key) => (UI[S.language === "bengali" ? "bn" : "en"][key] ?? UI.en[key] ?? key);

const el = {
  app: $("app"), history: $("history"), thread: $("thread"), scroller: $("scroller"), welcome: $("welcome"),
  input: $("input"), send: $("send"), composer: $("composer"), title: $("chatTitle"), passWrap: $("passWrap"),
  mode: $("modePill"), talk: $("talkBtn"), jump: $("jump"), language: $("language"), theme: $("themeToggle"),
  dialog: $("callback"), cbForm: $("callbackForm"), cbError: $("cbError"), toast: $("toast"), announcer: $("announcer"),
};

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|[^\w])_([^_]+?)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

// Small, safe markdown renderer: paragraphs, lists, quotes, headings, bold, italics, code, links.
function markdown(src) {
  const lines = String(src).replace(/\r/g, "").split("\n");
  let html = "", list = null, para = [], quote = [];
  const flushPara = () => { if (para.length) { html += `<p>${para.map(inline).join("<br>")}</p>`; para = []; } };
  const flushList = () => { if (list) { html += `<${list.type}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.type}>`; list = null; } };
  const flushQuote = () => { if (quote.length) { html += `<blockquote>${markdown(quote.join("\n"))}</blockquote>`; quote = []; } };
  for (const line of lines) {
    let m;
    if (/^\s*>\s?/.test(line)) { flushPara(); flushList(); quote.push(line.replace(/^\s*>\s?/, "")); continue; }
    flushQuote();
    if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { flushPara(); if (list?.type !== "ul") { flushList(); list = { type: "ul", items: [] }; } list.items.push(m[1]); continue; }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { flushPara(); if (list?.type !== "ol") { flushList(); list = { type: "ol", items: [] }; } list.items.push(m[1]); continue; }
    if ((m = line.match(/^\s*#{1,4}\s+(.*)$/))) { flushPara(); flushList(); html += `<h3>${inline(m[1])}</h3>`; continue; }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    flushList(); para.push(line);
  }
  flushPara(); flushList(); flushQuote();
  return html;
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

function toast(message) {
  el.toast.textContent = message;
  el.toast.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.toast.classList.remove("show"), 2200);
}

function updateJump() {
  const gap = el.scroller.scrollHeight - el.scroller.scrollTop - el.scroller.clientHeight;
  S.pinned = gap < 80;
  el.jump.hidden = gap < 240;
}

function scrollToBottom(force = false) {
  if (force || S.pinned) el.scroller.scrollTop = el.scroller.scrollHeight;
  updateJump();
}

// ------------------------------------------------------------------ static text and icons
function applyText() {
  document.documentElement.lang = S.language === "bengali" ? "bn" : "en";
  document.querySelectorAll("[data-i18n]").forEach((n) => { n.textContent = t(n.dataset.i18n); });
  document.querySelectorAll("[data-i18n-label]").forEach((n) => { n.setAttribute("aria-label", t(n.dataset.i18nLabel)); n.title = t(n.dataset.i18nLabel); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((n) => { n.placeholder = t(n.dataset.i18nPlaceholder); });
  if (!S.conversationId) el.title.textContent = t("newChat");
  setTalkState();
  renderHistory();
  if (S.plan) renderPlan(S.plan, false);
}

function paintIcons() {
  document.querySelectorAll("[data-icon]").forEach((n) => { n.outerHTML = icon(n.dataset.icon); });
  $("openSidebar").innerHTML = icon("menu");
  $("closeSidebar").innerHTML = icon("close");
  el.jump.innerHTML = icon("arrowDown");
  setSendMode(false);
  paintTheme();
}

function paintTheme() {
  const dark = document.documentElement.dataset.theme === "dark";
  el.theme.innerHTML = icon(dark ? "sun" : "moon");
}

function setSendMode(streaming) {
  el.send.innerHTML = icon(streaming ? "stop" : "send");
  el.send.setAttribute("aria-label", t(streaming ? "stop" : "send"));
  el.send.title = t(streaming ? "stop" : "send");
  el.send.disabled = !streaming && !el.input.value.trim();
}

function setTalkState() {
  el.talk.classList.toggle("done", S.callbackDone);
  el.talk.innerHTML = `${icon(S.callbackDone ? "check" : "phone")}<span>${t(S.callbackDone ? "talkDone" : "talk")}</span>`;
}

// ------------------------------------------------------------------ sidebar history
function groupOf(ts) {
  const d = new Date(ts * 1000), now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (d >= start) return "today";
  if (d >= start - 864e5) return "yesterday";
  if (d >= start - 7 * 864e5) return "week";
  return "older";
}

async function loadHistory() {
  try { S.conversations = await api(`/api/conversations?client_id=${encodeURIComponent(S.clientId)}`); }
  catch { S.conversations = []; }
  renderHistory();
}

function renderHistory() {
  if (!S.conversations.length) { el.history.innerHTML = `<p class="history-empty">${esc(t("noChats"))}</p>`; return; }
  const groups = { today: [], yesterday: [], week: [], older: [] };
  S.conversations.forEach((c) => groups[groupOf(c.updated)].push(c));
  el.history.innerHTML = Object.entries(groups).filter(([, list]) => list.length).map(([key, list]) => `
    <div class="history-group"><h2>${esc(t(key))}</h2>${list.map((c) => `
      <div class="history-item${c.id === S.conversationId ? " active" : ""}" data-id="${c.id}">
        <a href="#c/${c.id}" title="${esc(c.title)}">${esc(c.title)}</a>
        <span class="item-actions">
          <button type="button" data-act="rename" aria-label="${esc(t("rename"))}" title="${esc(t("rename"))}">${icon("pencil")}</button>
          <button type="button" data-act="delete" aria-label="${esc(t("del"))}" title="${esc(t("del"))}">${icon("trash")}</button>
        </span>
      </div>`).join("")}</div>`).join("");
}

el.history.addEventListener("click", async (e) => {
  const item = e.target.closest(".history-item");
  if (!item) return;
  const id = item.dataset.id;
  const act = e.target.closest("button")?.dataset.act;
  if (!act) { closeSidebar(); return; } // the link changes the hash
  e.preventDefault();
  if (act === "rename") return startRename(item, id);
  if (act === "delete") {
    const btn = e.target.closest("button");
    btn.dataset.act = "confirm";
    btn.className = "confirm-delete";
    btn.textContent = t("confirmDel");
    item.classList.add("confirming");
    setTimeout(() => { if (btn.isConnected && btn.dataset.act === "confirm") renderHistory(); }, 3500);
    return;
  }
  if (act === "confirm") {
    await api(`/api/conversations/${id}?client_id=${encodeURIComponent(S.clientId)}`, { method: "DELETE" });
    S.conversations = S.conversations.filter((c) => c.id !== id);
    if (id === S.conversationId) newChat();
    renderHistory();
    toast(t("deleted"));
  }
});

function startRename(item, id) {
  const conv = S.conversations.find((c) => c.id === id);
  item.innerHTML = `<input value="${esc(conv.title)}" maxlength="80" aria-label="${esc(t("rename"))}">`;
  const input = item.querySelector("input");
  input.focus(); input.select();
  const finish = async (save) => {
    const title = input.value.trim();
    if (save && title && title !== conv.title) {
      await api(`/api/conversations/${id}?client_id=${encodeURIComponent(S.clientId)}`, { method: "PATCH", body: JSON.stringify({ title }) });
      conv.title = title;
      if (id === S.conversationId) el.title.textContent = title;
    }
    renderHistory();
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); });
  input.addEventListener("blur", () => finish(true), { once: true });
}

// ------------------------------------------------------------------ plan card (boarding pass)
function renderPlan(plan, animate = true) {
  const prev = S.plan?.items || {};
  S.plan = plan;
  const items = plan?.items || {};
  const any = Object.values(items).some(Boolean);
  el.passWrap.hidden = !any;
  if (!any) { S.planComplete = false; return; }
  const collapsed = store.get("sarp.passCollapsed") === "1";
  const fields = ["course", "intake", "budget", "institution", "gpa"];
  const justCompleted = animate && plan.complete && !S.planComplete;
  S.planComplete = plan.complete;
  el.passWrap.innerHTML = `
    <div class="pass${collapsed ? " collapsed" : ""}">
      <div class="pass-main">
        <div class="pass-head">
          <button class="toggle" type="button" aria-expanded="${!collapsed}">
            ${icon("chevron", "pass-chevron")}<span class="pass-label">${esc(t("yourPlan"))}</span>
          </button>
          <button class="pass-reset" type="button">${esc(t("startOver"))}</button>
        </div>
        <div class="route">
          <span class="from">${esc(t("from"))}</span>
          <span class="path">${icon("plane")}</span>
          <span class="to${items.country ? "" : " empty"}">${esc(items.country || "?")}</span>
        </div>
        <dl class="pass-fields">${fields.map((f) => `
          <div class="pass-field${animate && items[f] && items[f] !== prev[f] ? " fresh" : ""}">
            <dt>${esc(t(f))}</dt><dd class="${items[f] ? "" : "empty"}">${esc(items[f] || "–")}</dd>
          </div>`).join("")}
        </dl>
      </div>
      <div class="pass-stub">
        <span class="pass-label">${esc(t("research"))}</span>
        <span class="research${items.research ? "" : " empty"}">${esc(items.research || "–")}</span>
        ${plan.complete ? `<span class="stamp${justCompleted ? " landing" : ""}">${esc(t("planReady"))}</span>` : ""}
      </div>
    </div>`;
  const pass = el.passWrap.querySelector(".pass");
  el.passWrap.querySelector(".toggle").addEventListener("click", (e) => {
    const isCollapsed = pass.classList.toggle("collapsed");
    e.currentTarget.setAttribute("aria-expanded", String(!isCollapsed));
    store.set("sarp.passCollapsed", isCollapsed ? "1" : "0");
  });
  el.passWrap.querySelector(".pass-reset").addEventListener("click", async () => {
    if (!S.conversationId) return;
    const fresh = await api(`/api/conversations/${S.conversationId}/plan/reset?client_id=${encodeURIComponent(S.clientId)}`, { method: "POST" });
    S.planComplete = false;
    renderPlan(fresh, false);
    toast(t("resetDone"));
  });
}

// ------------------------------------------------------------------ messages
function hideWelcome() { el.welcome.hidden = true; }

function addUserMessage(text) {
  const node = document.createElement("div");
  node.className = "msg user";
  node.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  el.thread.appendChild(node);
  return node;
}

function addAssistantMessage({ id = null, content = "", feedback = 0, pending = false } = {}) {
  document.querySelectorAll(".msg.assistant.latest").forEach((n) => n.classList.remove("latest"));
  const node = document.createElement("div");
  node.className = "msg assistant latest";
  node.innerHTML = `
    <svg class="avatar" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="var(--green)"/><path d="M7.5 16.2 24.5 8.6l-5.6 15.6-3.3-6.3-8.1-1.7Z" fill="var(--surface)"/><circle cx="24.6" cy="23.8" r="2.6" fill="var(--marigold)"/></svg>
    <div class="msg-body"><div class="notices"></div><div class="content"></div><div class="extras"></div><div class="actions" hidden></div></div>`;
  const msg = { node, id, raw: content, feedback, body: node.querySelector(".content"), extras: node.querySelector(".extras"), actions: node.querySelector(".actions") };
  if (pending) msg.body.innerHTML = '<span class="typing" aria-label="Typing"><i></i><i></i><i></i></span>';
  else { msg.body.innerHTML = markdown(content); renderActions(msg); }
  el.thread.appendChild(node);
  return msg;
}

function renderActions(msg) {
  if (!msg.id) { msg.actions.hidden = true; return; }
  msg.actions.hidden = false;
  msg.actions.innerHTML = `
    <button type="button" data-act="copy" aria-label="${esc(t("copy"))}" title="${esc(t("copy"))}">${icon("copy")}</button>
    <button type="button" data-act="good" aria-pressed="${msg.feedback === 1}" aria-label="${esc(t("good"))}" title="${esc(t("good"))}">${icon("up")}</button>
    <button type="button" data-act="bad" aria-pressed="${msg.feedback === -1}" aria-label="${esc(t("bad"))}" title="${esc(t("bad"))}">${icon("down")}</button>`;
  msg.actions.onclick = async (e) => {
    const act = e.target.closest("button")?.dataset.act;
    if (act === "copy") {
      try { await navigator.clipboard.writeText(msg.raw); toast(t("copied")); } catch { /* clipboard blocked */ }
    } else if (act === "good" || act === "bad") {
      const value = act === "good" ? 1 : -1;
      msg.feedback = msg.feedback === value ? 0 : value;
      renderActions(msg);
      api(`/api/messages/${msg.id}/feedback`, { method: "POST", body: JSON.stringify({ client_id: S.clientId, value: msg.feedback }) }).catch(() => {});
    }
  };
}

function renderChips(msg, chips) {
  document.querySelectorAll(".chips").forEach((n) => n.remove());
  if (!chips?.length) return;
  const box = document.createElement("div");
  box.className = "chips";
  chips.forEach((chip) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `chip${chip.skip ? " skip" : ""}`;
    b.textContent = chip.label;
    b.addEventListener("click", () => send(chip.label, { field: chip.field, value: chip.value ?? null, skip: !!chip.skip }));
    box.appendChild(b);
  });
  msg.extras.appendChild(box);
}

function renderCallbackPrompt(msg) {
  if (S.callbackDone) return;
  const box = document.createElement("div");
  box.className = "chips";
  box.innerHTML = `<button type="button" class="chip">${esc(t("requestCall"))}</button>`;
  box.querySelector("button").addEventListener("click", openCallback);
  msg.extras.appendChild(box);
}

// ------------------------------------------------------------------ sending and streaming
async function streamChat(body, onEvent, signal) {
  const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  if (!res.ok || !res.body) throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let type = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) type = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) onEvent(type, JSON.parse(data));
    }
  }
}

async function send(text, quickReply = null) {
  text = text.trim();
  if (!text || S.streaming) return;
  hideWelcome();
  document.querySelectorAll(".chips").forEach((n) => n.remove());
  addUserMessage(text);
  el.input.value = "";
  autosize();
  const msg = addAssistantMessage({ pending: true });
  S.streaming = true;
  S.pinned = true;
  setSendMode(true);
  scrollToBottom(true);
  el.thread.setAttribute("aria-busy", "true");
  S.controller = new AbortController();
  let chips = [], handoff = false, frame = 0, started = false;
  const paint = () => {
    frame = 0;
    msg.body.innerHTML = markdown(msg.raw);
    msg.body.classList.add("streaming");
    scrollToBottom();
  };
  try {
    await streamChat({
      client_id: S.clientId, conversation_id: S.conversationId, text, quick_reply: quickReply,
      language: S.language === "auto" ? null : S.language,
    }, (type, data) => {
      if (type === "meta") {
        if (S.conversationId !== data.conversation_id) {
          S.conversationId = data.conversation_id;
          history.replaceState(null, "", `#c/${data.conversation_id}`);
        }
        if (data.title) { el.title.textContent = data.title; loadHistory(); }
      } else if (type === "plan") {
        renderPlan(data.plan);
      } else if (type === "token") {
        if (!started) { started = true; msg.raw = ""; }
        msg.raw += data.text;
        if (!frame) frame = requestAnimationFrame(paint);
      } else if (type === "notice") {
        msg.node.querySelector(".notices").innerHTML = `<p class="notice">${esc(data.text)}</p>`;
      } else if (type === "chips") {
        chips = data.chips;
      } else if (type === "handoff") {
        handoff = true;
      } else if (type === "done") {
        msg.id = data.message_id;
      } else if (type === "error") {
        throw new Error(data.text);
      }
    }, S.controller.signal);
  } catch (error) {
    if (error.name === "AbortError") {
      if (!msg.raw) msg.raw = `_${t("stopped")}_`;
    } else {
      msg.raw = msg.raw || "";
      const why = error.status === 429 ? t("rateLimited") : t("errorNet");
      msg.extras.innerHTML = `<p class="error-line">${esc(why)} <button type="button">${esc(t("retry"))}</button></p>`;
      msg.extras.querySelector("button").addEventListener("click", () => {
        msg.node.previousElementSibling?.remove();
        msg.node.remove();
        send(text, quickReply);
      });
    }
  } finally {
    if (frame) cancelAnimationFrame(frame);
    msg.body.classList.remove("streaming");
    msg.body.innerHTML = msg.raw ? markdown(msg.raw) : "";
    renderActions(msg);
    renderChips(msg, chips);
    if (handoff) renderCallbackPrompt(msg);
    S.streaming = false;
    S.controller = null;
    setSendMode(false);
    el.thread.removeAttribute("aria-busy");
    el.announcer.textContent = t("replyReady");
    scrollToBottom();
    if (!matchMedia("(pointer: coarse)").matches) el.input.focus();
  }
}

// ------------------------------------------------------------------ conversations
function clearThread() {
  el.thread.querySelectorAll(".msg").forEach((n) => n.remove());
}

function newChat() {
  if (S.controller) S.controller.abort();
  S.conversationId = null;
  S.plan = null;
  S.planComplete = false;
  S.callbackDone = false;
  history.replaceState(null, "", "/");
  clearThread();
  el.welcome.hidden = false;
  el.passWrap.hidden = true;
  el.title.textContent = t("newChat");
  setTalkState();
  renderHistory();
  el.input.focus();
}

async function openConversation(id) {
  if (S.controller) S.controller.abort();
  let conv;
  try { conv = await api(`/api/conversations/${id}?client_id=${encodeURIComponent(S.clientId)}`); }
  catch { newChat(); return; }
  S.conversationId = conv.id;
  S.callbackDone = conv.callback_requested;
  S.planComplete = !!conv.plan?.complete;
  clearThread();
  hideWelcome();
  el.title.textContent = conv.title;
  let last = null;
  conv.messages.forEach((m) => {
    if (m.role === "user") addUserMessage(m.content);
    else last = { msg: addAssistantMessage({ id: m.id, content: m.content, feedback: m.feedback }), chips: m.chips };
  });
  const lastIsAssistant = conv.messages.at(-1)?.role === "assistant";
  if (last && lastIsAssistant) renderChips(last.msg, last.chips);
  renderPlan(conv.plan, false);
  setTalkState();
  renderHistory();
  S.pinned = true;
  requestAnimationFrame(() => scrollToBottom(true));
}

function route() {
  const m = location.hash.match(/^#c\/([a-z0-9]+)$/i);
  if (m) { if (m[1] !== S.conversationId) openConversation(m[1]); }
  else if (S.conversationId) newChat();
}

// ------------------------------------------------------------------ call-back dialog
async function openCallback() {
  el.cbError.hidden = true;
  if (S.callbackDone) { toast(t("requested")); return; }
  el.dialog.showModal();
  el.cbForm.elements.name.focus();
}

el.cbForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = el.cbForm.elements;
  const name = f.name.value.trim(), contact = f.contact.value.trim();
  if (!name || contact.replace(/\D/g, "").length < 6) {
    el.cbError.textContent = t("needDetails");
    el.cbError.hidden = false;
    return;
  }
  try {
    if (!S.conversationId) {
      const created = await api("/api/conversations", { method: "POST", body: JSON.stringify({ client_id: S.clientId }) });
      S.conversationId = created.id;
      history.replaceState(null, "", `#c/${created.id}`);
    }
    const preferred = f.time.value ? f.time.options[f.time.selectedIndex].text : null;
    const result = await api(`/api/conversations/${S.conversationId}/callback?client_id=${encodeURIComponent(S.clientId)}`,
      { method: "POST", body: JSON.stringify({ name, contact, preferred_time: preferred }) });
    el.dialog.close();
    S.callbackDone = true;
    setTalkState();
    hideWelcome();
    document.querySelectorAll(".chips").forEach((n) => { if (n.textContent === t("requestCall")) n.remove(); });
    addAssistantMessage({ id: result.message_id, content: result.content });
    scrollToBottom(true);
    toast(t("requested"));
    loadHistory();
  } catch {
    el.cbError.textContent = t("errorNet");
    el.cbError.hidden = false;
  }
});
$("cbCancel").addEventListener("click", () => el.dialog.close());

// ------------------------------------------------------------------ composer and controls
function autosize() {
  el.input.style.height = "auto";
  // Measure only real text: a wrapped placeholder would otherwise make the empty box tall.
  if (el.input.value) el.input.style.height = `${Math.min(el.input.scrollHeight, 200)}px`;
  el.send.disabled = !S.streaming && !el.input.value.trim();
}

el.input.addEventListener("input", autosize);
el.input.addEventListener("keydown", (e) => {
  // isComposing: Bengali input methods use Enter to confirm a word, not to send.
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!S.streaming) send(el.input.value);
  }
});
el.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (S.streaming) S.controller?.abort();
  else send(el.input.value);
});

$("newChat").addEventListener("click", () => { newChat(); closeSidebar(); });
el.talk.addEventListener("click", openCallback);
el.language.value = S.language;
el.language.addEventListener("change", () => { S.language = el.language.value; store.set("sarp.language", S.language); applyText(); });
el.theme.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store.set("sarp.theme", next);
  paintTheme();
});

function openSidebar() { el.app.classList.add("sidebar-open"); }
function closeSidebar() { el.app.classList.remove("sidebar-open"); }
$("openSidebar").addEventListener("click", openSidebar);
$("closeSidebar").addEventListener("click", closeSidebar);
$("scrim").addEventListener("click", closeSidebar);

el.scroller.addEventListener("scroll", updateJump, { passive: true });
el.jump.addEventListener("click", () => {
  S.pinned = true;
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  el.scroller.scrollTo({ top: el.scroller.scrollHeight, behavior: smooth ? "smooth" : "auto" });
});

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); newChat(); }
  if (e.key === "Escape") closeSidebar();
});
window.addEventListener("hashchange", route);

// Starter questions
$("starters").innerHTML = STARTERS.map((s, i) =>
  `<button class="starter" type="button" data-i="${i}">${esc(s.text)}<small>${esc(s.tag)}</small></button>`).join("");
$("starters").addEventListener("click", (e) => {
  const b = e.target.closest(".starter");
  if (b) send(STARTERS[Number(b.dataset.i)].text);
});

// ------------------------------------------------------------------ start
async function loadStatus() {
  try {
    const status = await api("/api/status");
    el.mode.hidden = status.mode !== "offline";
    el.mode.textContent = t("offline");
    el.mode.title = t("offlineTip");
  } catch { /* server not reachable yet */ }
}

paintIcons();
applyText();
loadStatus();
loadHistory().then(route);
autosize();
el.input.focus();
