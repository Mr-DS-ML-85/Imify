/* Imify — admin panel logic (vanilla JS, zero dependencies). */
"use strict";

const $ = (s) => document.querySelector(s);
let adminKey = localStorage.getItem("imify_admin_key") || "";

/* ---------------------------------------------------------------- fetch */
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (adminKey) headers["Authorization"] = "Bearer " + adminKey;
  if (opts.body && !(opts.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (r.status === 401 || r.status === 403) {
    $("#status-line").textContent = "Admin key required — click Unlock and paste your key.";
    document.querySelectorAll("table tbody").forEach((tb) => (tb.innerHTML = ""));
    throw new Error("auth");
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

function toast(msg, type = "") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

/* ---------------------------------------------------------------- unlock */
function setupUnlock() {
  $("#unlock-btn").addEventListener("click", async () => {
    const key = prompt("Paste your admin API key:");
    if (!key) return;
    adminKey = key;
    localStorage.setItem("imify_admin_key", key);
    // Verify the key RIGHT NOW instead of silently swallowing errors — a
    // wrong/stale key otherwise looks like "the admin page is broken".
    try {
      await api("/v1/admin/keys");
      toast("Admin unlocked", "ok");
      $("#status-line").textContent = "Admin key accepted.";
    } catch (e) {
      $("#status-line").textContent = "Key rejected — check the key and try again.";
      toast("Key rejected", "error");
    }
    refreshAll();
  });
}

/* ----------------------------------------------------------------- stats */
async function refreshStats() {
  try {
    const s = await api("/v1/stats");
    const cards = [
      ["Images", s.images.toLocaleString()],
      ["Searches", s.searches.toLocaleString()],
      ["Pending", s.pending.toLocaleString(), s.pending > 0 ? "warn" : ""],
      ["Seed queries", s.queries],
      ["Engines", Object.keys(s.engines || {}).length],
      ["Backend", s.storage_backend],
      ["Uptime", fmtDuration(s.uptime_s || 0)],
    ];
    $("#stat-cards").innerHTML = cards.map(([k, v, tone]) =>
      `<div class="stat-card"><div class="k">${k}</div><div class="v" style="${tone ? "color:var(--warn)" : ""}">${v}</div></div>`).join("");
    $("#status-line").textContent = `Storage: ${s.storage_backend} · ${s.images.toLocaleString()} canonical images · indexer ${s.indexer.running ? "running" : "stopped"}.`;
  } catch { /* auth error handled in api() */ }
}

/* ---------------------------------------------------------- success rates */
async function refreshSuccess() {
  try {
    const s = await api("/v1/stats");
    const srate = s.success.searches;
    const cards = [
      [`Search success (${srate.total})`, srate.total ? (srate.rate * 100).toFixed(1) + "%" : "—", `${srate.found} found · ${srate.not_found} not found`],
    ];
    for (const [name, p] of Object.entries(s.success.providers)) {
      cards.push([`${name}`, p.total ? (p.rate * 100).toFixed(1) + "%" : "—", `${p.ok} ok · ${p.fail} fail`]);
    }
    const idx = s.success.indexer;
    cards.push([`indexer fetch`, idx.fetched + idx.failed ? (idx.rate * 100).toFixed(1) + "%" : "—", `${idx.fetched} ok · ${idx.failed} fail`]);
    $("#success-grid").innerHTML = cards.map(([k, v, sub]) =>
      `<div class="stat-card" style="padding:14px"><div class="k">${k}</div><div class="v" style="font-size:22px">${v}</div><div style="font-size:11.5px;color:var(--text-faint);margin-top:4px">${sub}</div></div>`).join("");
  } catch { /* auth */ }
}

/* ---------------------------------------------------------------- queries */
async function refreshQueries() {
  const data = await api("/v1/admin/queries");
  const body = $("#queries-body");
  body.innerHTML = "";
  for (const q of data.queries) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${escapeHtml(q.query)}</b></td>
      <td><span class="chip ${q.enabled ? "on" : "off"}">${q.enabled ? "active" : "paused"}</span></td>
      <td>${q.last_run ? new Date(q.last_run * 1000).toLocaleString() : "never"}</td>
      <td>${q.last_count}</td>
      <td>${q.total}</td>
      <td style="text-align:right">
        <button class="btn ghost toggle">${q.enabled ? "Pause" : "Resume"}</button>
        <button class="btn danger del">Delete</button>
      </td>`;
    tr.querySelector(".toggle").addEventListener("click", async () => {
      await api(`/v1/admin/queries/${q.id}`, { method: "PATCH", body: JSON.stringify({ enabled: !q.enabled }) });
      refreshQueries();
    });
    tr.querySelector(".del").addEventListener("click", async () => {
      if (!confirm(`Delete query "${q.query}"?`)) return;
      await api(`/v1/admin/queries/${q.id}`, { method: "DELETE" });
      refreshQueries();
    });
    body.appendChild(tr);
  }
  if (!data.queries.length) {
    body.innerHTML = '<tr><td colspan="6" style="color:var(--text-faint)">No seed queries — add some above to start filling the index.</td></tr>';
  }
}

function setupAddQuery() {
  $("#add-query-btn").addEventListener("click", async () => {
    const raw = $("#new-query").value.trim();
    if (!raw) return toast("Type at least one query", "error");
    const queries = raw.split(/[,;]/).map((s) => s.trim()).filter(Boolean);
    try {
      const r = await api("/v1/admin/ingest", { method: "POST", body: JSON.stringify({ queries }) });
      toast(`Added ${r.count} seed quer${r.count === 1 ? "y" : "ies"}`, "ok");
      $("#new-query").value = "";
      refreshQueries();
      refreshPipeline();
    } catch (e) { toast(e.message, "error"); }
  });
  $("#new-query").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#add-query-btn").click(); });
}

/* --------------------------------------------------------------- pipeline */
async function refreshPipeline() {
  try {
    const s = await api("/v1/admin/crawl");
    $("#pipeline-pill").textContent = s.running ? "running" : "stopped";
    $("#pipeline-pill").className = "pill " + (s.running ? "on" : "off");
    const items = [
      ["Workers alive", s.workers_alive],
      ["Fetched", s.fetched],
      ["Indexed (session)", s.indexed],
      ["Deduped", s.deduped],
      ["Failed", s.failed],
      ["SearXNG requests", s.searx_requests],
      ["Pending queue", s.pending],
    ];
    $("#pipeline-grid").innerHTML = items.map(([k, v]) =>
      `<div class="stat-card" style="padding:14px"><div class="k">${k}</div><div class="v" style="font-size:20px">${v}</div></div>`).join("");
    $("#pipeline-error").textContent = s.last_error ? "Last error: " + s.last_error : "";
    if (s.searx_circuit_open) toast("SearXNG circuit breaker open — throttling discovery", "error");
  } catch { /* auth */ }
}

/* ---------------------------------------------------------------- images */
let imageFilter = "";
async function refreshImages() {
  try {
    const s = await api("/v1/admin/images?limit=100&q=" + encodeURIComponent(imageFilter));
    $("#images-pill").textContent = s.total.toLocaleString();
    const body = $("#images-body");
    body.innerHTML = "";
    for (const m of s.images) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><img src="/img/${m.id}.jpg" style="width:44px;height:44px;object-fit:cover;border-radius:8px" loading="lazy"></td>
        <td>${m.id}</td>
        <td><span class="chip">${escapeHtml(m.engine || "—")}</span></td>
        <td style="max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(m.url)}"><a href="${escapeHtml(m.url)}" target="_blank" rel="noopener">${escapeHtml(m.url)}</a></td>
        <td style="font-family:monospace;font-size:11px">${hexPhash(m.phash)}</td>
        <td>${m.last_seen ? new Date(m.last_seen * 1000).toLocaleDateString() : "—"}</td>
        <td style="text-align:right"><button class="btn danger del">Remove</button></td>`;
      tr.querySelector(".del").addEventListener("click", async () => {
        if (!confirm(`Remove image #${m.id} (${m.url}) from the index?`)) return;
        try {
          await api(`/v1/admin/images/${m.id}`, { method: "DELETE" });
          toast(`Removed image #${m.id}`, "ok");
          refreshAll();
        } catch (e) { toast(e.message, "error"); }
      });
      body.appendChild(tr);
    }
    if (!s.images.length) {
      body.innerHTML = `<tr><td colspan="7" style="color:var(--text-faint)">${imageFilter ? "No images match the filter." : "Index is empty — add seed queries or search an image to fill it."}</td></tr>`;
    }
  } catch { /* auth */ }
}

function setupImages() {
  $("#image-search-btn").addEventListener("click", () => {
    imageFilter = $("#image-search").value.trim();
    refreshImages();
  });
  $("#image-search").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#image-search-btn").click(); });
  $("#purge-btn").addEventListener("click", async () => {
    const needle = $("#purge-query").value.trim();
    if (!needle) return toast("Type a URL substring to purge", "error");
    if (!confirm(`Purge ALL images whose URL contains "${needle}"? This cannot be undone.`)) return;
    try {
      const r = await api("/v1/admin/images/purge", { method: "POST", body: JSON.stringify({ url_contains: needle }) });
      toast(`Purged ${r.removed} image${r.removed === 1 ? "" : "s"}`, "ok");
      $("#purge-query").value = "";
      refreshAll();
    } catch (e) { toast(e.message, "error"); }
  });
}

/* --------------------------------------------------------------- trackers */
async function refreshTracks() {
  try {
    const s = await api("/v1/admin/tracks");
    $("#tracks-pill").textContent = `${s.tracks.length} active`;
    const body = $("#tracks-body");
    body.innerHTML = "";
    for (const t of s.tracks) {
      const tr = document.createElement("tr");
      const img = t.image_id
        ? `<a href="/v1/image/${t.image_id}" target="_blank" rel="noopener"><img src="/img/${t.image_id}.jpg" style="width:44px;height:44px;object-fit:cover;border-radius:8px"></a>`
        : `<span style="color:var(--text-faint)">not indexed yet</span>`;
      tr.innerHTML = `
        <td>${t.id}</td>
        <td>${img}</td>
        <td style="font-family:monospace;font-size:11px">${hexPhash(t.phash)}</td>
        <td>${t.last_run ? new Date(t.last_run * 1000).toLocaleString() : "never"}</td>
        <td>${t.hits}</td>
        <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escapeHtml(t.note || "")}">${escapeHtml(t.note || "")}</td>
        <td style="text-align:right">
          <span class="chip ${t.enabled ? "on" : "off"}">${t.enabled ? "on" : "off"}</span>
          <button class="btn ghost toggle">${t.enabled ? "Pause" : "Resume"}</button>
          <button class="btn danger del">Delete</button>
        </td>`;
      tr.querySelector(".toggle").addEventListener("click", async () => {
        await api(`/v1/admin/tracks/${t.id}`, { method: "PATCH", body: JSON.stringify({ enabled: !t.enabled }) });
        refreshTracks();
      });
      tr.querySelector(".del").addEventListener("click", async () => {
        if (!confirm(`Delete tracker #${t.id}?`)) return;
        await api(`/v1/admin/tracks/${t.id}`, { method: "DELETE" });
        refreshTracks();
      });
      body.appendChild(tr);
    }
    if (!s.tracks.length) {
      body.innerHTML = '<tr><td colspan="7" style="color:var(--text-faint)">No trackers — search an image that is not in the index yet and it will be tracked here automatically.</td></tr>';
    }
  } catch { /* auth */ }
}

function setupTracks() {
  $("#tracks-run-btn").addEventListener("click", async () => {
    try {
      await api("/v1/admin/tracks/run", { method: "POST", body: JSON.stringify({}) });
      toast("Tracker round kicked off in the background", "ok");
      setTimeout(refreshTracks, 3000);
    } catch (e) { toast(e.message, "error"); }
  });
}

/* ------------------------------------------------------------------ keys */
function setupKeys() {
  $("#create-key-btn").addEventListener("click", async () => {
    const name = $("#key-name").value.trim();
    if (!name) return toast("Key name required", "error");
    try {
      const r = await api("/v1/admin/keys", { method: "POST", body: JSON.stringify({ name, role: $("#key-role").value }) });
      const reveal = $("#key-reveal");
      reveal.hidden = false;
      reveal.textContent = "Store this now — it is shown only once:  " + r.api_key;
      $("#key-name").value = "";
      refreshKeys();
    } catch (e) { toast(e.message, "error"); }
  });
}

async function refreshKeys() {
  const data = await api("/v1/admin/keys");
  const body = $("#keys-body");
  body.innerHTML = "";
  for (const k of data.keys) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${escapeHtml(k.name)}</b></td>
      <td><span class="chip ${k.role === "admin" ? "on" : ""}">${k.role}</span></td>
      <td>${new Date(k.created * 1000).toLocaleString()}</td>
      <td style="text-align:right"><button class="btn danger">Delete</button></td>`;
    tr.querySelector(".btn").addEventListener("click", async () => {
      await api(`/v1/admin/keys/${encodeURIComponent(k.name)}`, { method: "DELETE" });
      refreshKeys();
    });
    body.appendChild(tr);
  }
}

/* ---------------------------------------------------------------- helpers */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
/* 64-bit hashes arrive as hex strings from the API (JS numbers lose precision
   past 2^53); tolerate legacy numeric payloads too. */
function hexPhash(p) {
  if (typeof p === "string") return p;
  return Number(p).toString(16).padStart(16, "0");
}
function fmtDuration(s) {
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
}

/* ------------------------------------------------------------------ init */
function refreshAll() {
  refreshStats().catch(() => {});
  refreshQueries().catch(() => {});
  refreshPipeline().catch(() => {});
  refreshSuccess().catch(() => {});
  refreshKeys().catch(() => {});
  refreshImages().catch(() => {});
  refreshTracks().catch(() => {});
}

document.addEventListener("DOMContentLoaded", () => {
  setupUnlock();
  setupAddQuery();
  setupImages();
  setupTracks();
  setupKeys();
  refreshAll();
  setInterval(refreshAll, 15000);
});
