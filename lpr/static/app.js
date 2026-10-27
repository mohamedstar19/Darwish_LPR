"use strict";

const $ = (sel) => document.querySelector(sel);
const KINDS = { car: "ملاكي", motorcycle: "موتوسيكل", bus: "أتوبيس", truck: "نقل" };
const fmtTime = (ts) => new Date(ts * 1000).toLocaleString("ar-EG", { dateStyle: "short", timeStyle: "medium" });

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function plateEl(letters, digits, small = false) {
  return el("span", { class: "plate" + (small ? " small" : "") },
    el("span", { class: "band" }, "مصر  EGYPT"),
    el("span", { class: "num" },
      el("span", { class: "letters" }, [...letters].join(" ")),
      el("span", { class: "digits" }, digits)));
}

function plateFromDisplay(display) {
  const letters = [...display].filter((c) => /[ء-ي]/.test(c)).join("");
  const digits = [...display].filter((c) => /[٠-٩]/.test(c)).join("");
  return plateEl(letters, digits, true);
}

function toast(msg, isAlert = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (isAlert ? " alert" : "");
  clearTimeout(toast._h);
  toast._h = setTimeout(() => (t.className = "toast"), isAlert ? 8000 : 3000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.status === 204 ? null : res.json();
}

function snapUrl(rel) { return rel ? "/snapshots/" + rel.split("/").map(encodeURIComponent).join("/") : null; }

function openViewer(src) {
  const d = $("#viewer");
  d.querySelector("img").src = src;
  d.showModal();
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
  if (btn.dataset.tab === "log") loadLog(true);
  if (btn.dataset.tab === "watch") loadWatch();
}));

// ---------- live ----------
const camCards = new Map();

function renderCam(c) {
  let card = camCards.get(c.id);
  if (!card) {
    const img = el("img", { alt: c.name, loading: "lazy", src: `/api/cameras/${encodeURIComponent(c.id)}/live.mjpg` });
    card = { root: el("article", { class: "cam" }, img, el("div", { class: "meta" })), img };
    camCards.set(c.id, card);
    $("#cams").append(card.root);
  }
  const meta = card.root.querySelector(".meta");
  meta.replaceChildren(
    el("strong", {}, c.name),
    el("span", { class: "badge " + (c.online ? "on" : "off") }, c.online ? "متصلة" : "غير متصلة"),
    c.resolution ? el("span", { class: "badge" }, c.resolution) : null,
    el("span", { class: "badge" }, `${c.processing_fps} / ${c.target_fps} fps`),
    c.error ? el("span", { class: "err" }, c.error) : null,
    c.live_reads && c.live_reads.length ? el("span", { class: "reads" }, c.live_reads.map(plateFromDisplay)) : null,
  );
}

async function refreshCams() {
  try {
    const cams = await api("/api/cameras");
    if (!cams.length) $("#cams").replaceChildren(el("p", { class: "empty" }, "مفيش كاميرات متفعلة في config.yaml"));
    cams.forEach(renderCam);
  } catch (e) { /* transient */ }
}

async function refreshStats() {
  try {
    const s = await api("/api/stats");
    $("#st-events").textContent = s.today.events.toLocaleString("ar-EG");
    $("#st-unique").textContent = s.today.unique_plates.toLocaleString("ar-EG");
    $("#st-alerts").textContent = s.today.alerts.toLocaleString("ar-EG");
    $("#device").textContent = s.device ? `المعالجة على: ${s.device.startsWith("cuda") ? "كارت شاشة (GPU)" : "المعالج (CPU)"}` : "";
  } catch (e) { $("#device").textContent = "تعذر الاتصال بالخادم"; }
}

function feedItem(ev) {
  const li = el("li", { class: ev.watch_kind === "alert" ? "alert" : "" },
    plateEl(ev.letters, ev.digits),
    el("div", {},
      el("div", {}, ev.camera_name, ev.watch_kind ? el("span", { class: "tag " + ev.watch_kind }, ev.watch_kind === "alert" ? "تنبيه" : "مسموح") : null),
      el("div", { class: "info" }, `${KINDS[ev.vehicle_type] || ev.vehicle_type} · ${fmtTime(ev.ts)}`),
      ev.watch_label ? el("div", { class: "info" }, ev.watch_label) : null));
  return li;
}

async function loadFeed() {
  const events = await api("/api/events?limit=30");
  $("#feed").replaceChildren(...events.map(feedItem));
  if (!events.length) $("#feed").append(el("li", { class: "empty" }, "لسه مفيش لوحات"));
}

function connectStream() {
  const es = new EventSource("/api/events/stream");
  es.onopen = () => $("#live-dot").classList.add("on");
  es.onerror = () => $("#live-dot").classList.remove("on");
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    const feed = $("#feed");
    feed.querySelector(".empty")?.remove();
    feed.prepend(feedItem(ev));
    while (feed.children.length > 50) feed.lastElementChild.remove();
    if (ev.watch_kind === "alert") toast(`تنبيه: ${ev.plate_display} — ${ev.camera_name}${ev.watch_label ? " — " + ev.watch_label : ""}`, true);
    refreshStats();
  };
}

// ---------- log ----------
let lastId = null;

function logParams() {
  const p = new URLSearchParams();
  const q = $("#q").value.trim();
  if (q) p.set("q", q);
  const cam = $("#camera-filter").value;
  if (cam) p.set("camera", cam);
  const day = $("#day").value;
  if (day) {
    const start = new Date(day + "T00:00:00").getTime() / 1000;
    p.set("since", start);
    p.set("until", start + 86400);
  }
  return p;
}

function logRow(ev) {
  const img = snapUrl(ev.plate_img || ev.vehicle_img);
  return el("tr", { class: ev.watch_kind === "alert" ? "alert" : "" },
    el("td", {}, img ? el("img", { class: "thumb", src: img, alt: ev.plate_display, loading: "lazy", onclick: () => openViewer(snapUrl(ev.vehicle_img) || img) }) : "—"),
    el("td", {}, plateEl(ev.letters, ev.digits, true),
      ev.watch_kind ? el("span", { class: "tag " + ev.watch_kind }, ev.watch_kind === "alert" ? "تنبيه" : "مسموح") : null),
    el("td", {}, ev.camera_name),
    el("td", {}, fmtTime(ev.ts)),
    el("td", {}, KINDS[ev.vehicle_type] || ev.vehicle_type),
    el("td", {}, `${Math.round(ev.confidence * 100).toLocaleString("ar-EG")}٪ · ${ev.reads.toLocaleString("ar-EG")} قراية`));
}

async function loadLog(reset) {
  const p = logParams();
  $("#csv").href = "/api/events.csv?" + p.toString();
  p.set("limit", 100);
  if (!reset && lastId) p.set("before_id", lastId);
  try {
    const rows = await api("/api/events?" + p.toString());
    const body = $("#log-body");
    if (reset) body.replaceChildren();
    rows.forEach((r) => body.append(logRow(r)));
    if (reset && !rows.length) body.append(el("tr", {}, el("td", { colspan: 6, class: "empty" }, "مفيش نتايج")));
    lastId = rows.length ? rows[rows.length - 1].id : lastId;
    $("#more").hidden = rows.length < 100;
  } catch (e) { toast(e.message, true); }
}

$("#log-form").addEventListener("submit", (e) => { e.preventDefault(); loadLog(true); });
$("#more").addEventListener("click", () => loadLog(false));

// ---------- watchlist ----------
async function loadWatch() {
  const items = await api("/api/watchlist");
  const list = $("#watch-list");
  list.replaceChildren(...items.map((w) => {
    const plate = plateFromDisplay(w.plate_display);
    return el("li", {},
      plate,
      el("span", { class: "tag " + w.kind }, w.kind === "alert" ? "تنبيه" : "مسموح"),
      el("span", { class: "label" }, w.label || ""),
      el("button", { type: "button", onclick: async () => {
        if (!confirm("حذف " + w.plate_display + " من القائمة؟")) return;
        await api(`/api/watchlist/${w.id}`, { method: "DELETE" });
        loadWatch();
      } }, "حذف"));
  }));
  if (!items.length) list.append(el("li", { class: "empty" }, "القائمة فاضية"));
}

$("#watch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/watchlist", {
      method: "POST",
      body: JSON.stringify({ plate: $("#w-plate").value, label: $("#w-label").value, kind: $("#w-kind").value }),
    });
    $("#w-plate").value = "";
    $("#w-label").value = "";
    toast("اتضافت للقائمة");
    loadWatch();
  } catch (err) { toast(err.message, true); }
});

// ---------- boot ----------
(async function boot() {
  await Promise.all([refreshCams(), refreshStats(), loadFeed().catch(() => {})]);
  try {
    const cams = await api("/api/cameras");
    const sel = $("#camera-filter");
    cams.forEach((c) => sel.append(el("option", { value: c.id }, c.name)));
  } catch (_) { /* ignore */ }
  connectStream();
  setInterval(refreshCams, 3000);
  setInterval(refreshStats, 30000);
})();
