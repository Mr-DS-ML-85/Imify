/* Imify — search UI logic (vanilla JS, zero dependencies). */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  lastForm: null,       // { mode, image_id }
  currentResults: [],
  lastQueryThumb: null,
};

/* ------------------------------------------------------------- toasts */
function toast(msg, type = "") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

/* ------------------------------------------------------------- stats */
async function refreshStats() {
  try {
    const r = await fetch("/v1/stats");
    if (!r.ok) return;
    const s = await r.json();
    $("#stat-images").textContent = s.images.toLocaleString();
    $("#backend-info").textContent = `Storage: ${s.storage_backend} · ${s.indexer.pending} pending · ${s.engines ? Object.keys(s.engines).length : 0} engines.`;
  } catch { /* stats are best-effort */ }
}

/* ------------------------------------------------------------ history */
async function toggleHistory() {
  const existing = document.querySelector(".history-pop");
  if (existing) { existing.remove(); return; }
  let items = [];
  try {
    const r = await fetch("/v1/history?limit=12");
    if (r.ok) items = await r.json();
  } catch { /* ignore */ }
  const btn = $("#history-btn");
  const pop = document.createElement("div");
  pop.className = "history-pop";
  if (!items.length) {
    pop.innerHTML = '<div style="padding:12px;color:var(--text-faint);font-size:13px">No searches yet.</div>';
  } else {
    for (const it of items) {
      const url = it.image_url || "";
      const row = document.createElement("div");
      row.className = "h-item";
      row.innerHTML = `
        <img src="${url || "/static/favicon.svg"}" onerror="this.style.opacity=.15">
        <div class="h-meta">
          <b>${escapeHtml(url ? shorten(url, 34) : "Image #" + it.image_id)}</b>
          <span>${new Date(it.t * 1000).toLocaleString()} · ${it.results || 0} results · ${it.mode}</span>
        </div>`;
      if (url) row.addEventListener("click", () => searchByUrl(url));
      pop.appendChild(row);
    }
  }
  pop.style.position = "absolute";
  btn.after(pop);
  document.addEventListener("click", function close(e) {
    if (!pop.contains(e.target) && e.target !== btn) { pop.remove(); document.removeEventListener("click", close); }
  }, { once: true });
}

/* ------------------------------------------------------------- search */
async function searchByUrl(url) {
  if (!url) return;
  state.lastForm = { imageUrl: url };
  showSearching();
  const fd = new FormData();
  fd.append("image_url", url);
  fd.append("mode", "best");
  fd.append("num_results", "60");
  fd.append("include_web", "true");
  fd.append("learn", $("#learn-toggle") && $("#learn-toggle").checked ? "true" : "false");
  try {
    const r = await fetch("/v1/search", { method: "POST", body: fd });
    await handleSearchResponse(r, url, null);
  } catch (e) { hideSearching(); toast("Search failed: " + e.message, "error"); }
}

function showSearching() {
  $("#search-area").hidden = true;
  $("#results-area").hidden = true;
  $("#empty-state").hidden = true;
  $("#searching").hidden = false;
}

function hideSearching() {
  $("#searching").hidden = true;
  $("#search-area").hidden = false;
}

async function handleSearchResponse(r, url, fileObj) {
  if (!r.ok) {
    hideSearching();
    let msg = "Search failed";
    try { msg = (await r.json()).detail || msg; } catch { /* ignore */ }
    toast(msg, "error");
    if (r.status === 401 || r.status === 429) return;
    $("#search-area").hidden = false;
    return;
  }
  const data = await r.json();
  const thumb = fileObj ? URL.createObjectURL(fileObj) : url;
  renderResults(data, thumb);
}

function renderResults(data, queryThumb) {
  hideSearching();
  $("#search-area").hidden = true;
  $("#results-area").hidden = false;
  $("#query-img").src = queryThumb;
  const results = data.results || [];
  const verified = results.filter((r) => r.match_type !== "unverified").length;
  const unverified = results.length - verified;
  const found = data.found !== false && verified > 0;
  // TinEye model: verified = exact matches, unverified = similar candidates.
  // Show BOTH — the grid never disappears just because nothing hash-verified.
  const titleParts = [];
  if (found) {
    titleParts.push(`${verified} verified match${verified === 1 ? "" : "es"}`);
    if (unverified) titleParts.push(`${unverified} similar`);
  } else if (results.length) {
    titleParts.push(`No exact matches — ${results.length} similar candidate${results.length === 1 ? "" : "s"}`);
  } else {
    titleParts.push("Search not found");
  }
  $("#results-title").textContent = titleParts.join(" + ");
  const providerBadges = (data.providers || []).map((p) => p.toUpperCase()).join(" · ");
  let degradedLine = "";
  const deg = data.degraded || [];
  if (deg.length) degradedLine = ` · <span style="color:var(--warn)">${deg.map((d) => d.replace(/_/g, " ").replace("provider:", "")).join(", ")}</span>`;
  const cachedLine = data.cached ? ` · <span class="badge near" title="These matches were served from the verified-results cache — no fresh web crawl was needed">cached</span>` : "";
  $("#results-sub").innerHTML = `${data.took_ms} ms · ${providerBadges}${degradedLine}${cachedLine}`;
  state.currentResults = results;
  state.lastQueryThumb = queryThumb;
  if (results.length) {
    renderGrid();
  } else {
    renderNotFound(data);
  }
  refreshStats();
}

function renderNotFound(data) {
  const empty = $("#empty-state");
  const deg = (data.degraded || []).map((d) => d.replace(/^provider:/, ""));
  // Honest verdict, never a fake dead-end: count which byte-upload engines
  // actually ran (yandex-upload/bing-upload/google-upload recorded) vs which
  // genuinely failed (present in degraded).
  const uploadEngines = ["bing-upload", "yandex-upload", "google-upload"];
  const attempted = uploadEngines.filter((p) => !deg.includes(p));
  const blockedAll = deg.includes("bing-upload") && deg.includes("yandex-upload") && deg.includes("google-upload");
  let detail =
    "Nothing in the index and no verified web matches — the search was not found. Try a clearer image, or search for related keywords to grow the index.";
  if (deg.includes("searxng")) {
    detail = "Your own index has no match and SearXNG is unreachable — the search fell back to local only and found nothing.";
  } else if (deg.includes("web_candidates_unverified")) {
    detail = "Engines returned candidate pages but NONE of them hash-matched your image — they are listed below as unverified. Only sites confirmed by our own perceptual-hash matcher count as matches.";
  } else if (blockedAll) {
    detail = "Your own index has no match and ALL byte-upload reverse engines are currently blocked (this image has no public URL, so only byte-upload works). The background tracker keeps retrying automatically.";
  } else if (attempted.length) {
    detail = `Engines checked the whole internet for this image but found nothing: ${attempted.join(", ")}. The background tracker keeps re-checking automatically as new sites index it.`;
  } else if (deg.includes("web_needs_public_url")) {
    detail = "Your own index has no match. Set PUBLIC_BASE_URL to a public address so the live web reverse layer can check the internet too.";
  }
  empty.querySelector("h3").textContent = "Search not found";
  empty.querySelector("p").textContent = detail;
  empty.hidden = false;
  $("#results-area").hidden = true;
  $("#grid").innerHTML = "";
}

function renderGrid() {
  const grid = $("#grid");
  grid.innerHTML = "";
  const results = state.currentResults;
  if (!results.length) {
    $("#results-area").hidden = true;
    $("#empty-state").hidden = false;
    return;
  }
  for (const r of results) {
    const card = document.createElement("article");
    card.className = "card";
    const meter = r.hamming != null ? Math.max(6, Math.round((1 - r.hamming / 32) * 100)) : 60;
    const size = r.width && r.height ? `${r.width}×${r.height}` : "";
    const engine = r.engine || (r.source === "web" ? "web" : "index");
    const titleHref = r.page_url || r.url;
    const resolution = (r.resolution || []).length
      ? `<div class="res-line" title="How this candidate URL was resolved to pixels">${r.resolution.map((h) => `<span class="hop">${escapeHtml(h)}</span>`).join(`<span class="hop-arrow">→</span>`)}</div>`
      : "";
    const note = r.note
      ? `<div class="res-note ${r.match_type === "unverified" ? "warn" : ""}">${escapeHtml(r.note)}</div>`
      : "";
    // TinEye "found on N sites" — every known URL carrying this exact image,
    // each a direct removal-request / verification link.
    const remoteLinks = (r.source_urls || []).filter((u) => u !== r.url).slice(0, 6);
    const sourcesBlock = remoteLinks.length
      ? `<div class="sources-line" title="Every verified site carrying this image">
          <span class="sources-label">found on</span>
          ${remoteLinks.map((u) => `<a class="source-link" href="${escapeAttr(u)}" target="_blank" rel="noopener" title="${escapeAttr(u)}">${escapeHtml(hostOf(u))}</a>`).join(" ")}
          ${(r.source_urls || []).length > remoteLinks.length ? `<span class="more-sources">+${(r.source_urls || []).length - remoteLinks.length}</span>` : ""}
        </div>`
      : "";
    card.innerHTML = `
      <div class="thumb-wrap">
        <span class="badge ${r.match_type}">${r.match_type}</span>
        ${engine ? `<span class="engine-tag">${escapeHtml(engine)}</span>` : ""}
        <img loading="lazy" src="${escapeAttr(r.thumb || r.url)}" alt="" onerror="this.parentElement.style.background='#0d1326';this.style.display='none'">
        <div class="hover">
          <button class="act visit">Open source</button>
          <button class="act page">Visit page</button>
        </div>
      </div>
      <div class="body">
        <a class="title" href="${escapeAttr(titleHref)}" target="_blank" rel="noopener">${escapeHtml(r.title)}</a>
        <div class="row">
          ${size ? `<span>${size}</span><span class="dot"></span>` : ""}
          <span>${r.source_count > 1 ? r.source_count + " sources" : r.source === "both" ? "index + web" : r.source}</span>
          <span class="dot"></span>
          <span>${r.hamming != null ? "Δ" + r.hamming : (r.score * 100).toFixed(0) + "%"}</span>
        </div>
        ${resolution}
        ${sourcesBlock}
        ${note}
        <div class="match-meter"><i style="width:0%"></i></div>
      </div>`;
    card.querySelector(".match-meter i").style.width = meter + "%";
    card.querySelector(".visit").addEventListener("click", () => window.open(r.url, "_blank"));
    if (r.page_url) card.querySelector(".page").addEventListener("click", () => window.open(r.page_url, "_blank"));
    else card.querySelector(".page").style.display = "none";
    if (r.image_id) {
      card.addEventListener("dblclick", () => searchById(r.image_id));
      card.title = "Double-click to search this image";
    }
    grid.appendChild(card);
  }
}

async function searchById(id) {
  state.lastForm = { imageId: id };
  showSearching();
  const fd = new FormData();
  fd.append("image_id", String(id));
  fd.append("mode", "best");
  fd.append("num_results", "60");
  fd.append("include_web", "true");
  fd.append("learn", $("#learn-toggle") && $("#learn-toggle").checked ? "true" : "false");
  try {
    const r = await fetch("/v1/search", { method: "POST", body: fd });
    const data = await r.json();
    renderResults(data, null);
    $("#query-img").src = `/img/${id}.jpg`;
  } catch (e) { hideSearching(); toast("Search failed: " + e.message, "error"); }
}

/* -------------------------------------------------------------- upload */
function setupDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");

  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files[0]) uploadAndSearch(input.files[0]); });

  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) uploadAndSearch(file);
    else toast("That doesn't look like an image file", "error");
  });

  document.addEventListener("paste", (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) { uploadAndSearch(file); return; }
      }
      if (item.type === "text/plain") {
        item.getAsString((text) => {
          if (/^https?:\/\/.+\.(jpe?g|png|gif|webp|avif)(\?.*)?$/i.test(text.trim())) searchByUrl(text.trim());
        });
      }
    }
  });

  $("#url-go").addEventListener("click", () => { const v = $("#url-input").value.trim(); if (v) searchByUrl(v); });
  $("#url-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { const v = e.target.value.trim(); if (v) searchByUrl(v); } });
}

async function uploadAndSearch(file) {
  state.lastForm = { file };
  showSearching();
  const fd = new FormData();
  fd.append("file", file);
  fd.append("mode", "best");
  fd.append("num_results", "60");
  fd.append("include_web", "true");
  fd.append("learn", $("#learn-toggle") && $("#learn-toggle").checked ? "true" : "false");
  try {
    const r = await fetch("/v1/search", { method: "POST", body: fd });
    await handleSearchResponse(r, "", file);
  } catch (e) { hideSearching(); toast("Upload failed: " + e.message, "error"); }
}

/* ---------------------------------------------------------------- tabs */
function setupTabs() {
  $("#tabs").addEventListener("click", async (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    if (!state.lastForm) return;
    toast(`Re-ranking by ${tab.dataset.mode}…`);
    // local rerank: reuse last query features via a fresh search with mode
    const fd = new FormData();
    if (state.lastForm.file) fd.append("file", state.lastForm.file);
    if (state.lastForm.imageUrl) fd.append("image_url", state.lastForm.imageUrl);
    if (state.lastForm.imageId) fd.append("image_id", state.lastForm.imageId);
    fd.append("mode", tab.dataset.mode);
    fd.append("num_results", "60");
    fd.append("include_web", "false");
    showSearching();
    try {
      const r = await fetch("/v1/search", { method: "POST", body: fd });
      const data = await r.json();
      if (r.ok) {
        state.currentResults = data.results || [];
        renderResults(data, state.lastQueryThumb);
      } else { hideSearching(); toast("Search failed", "error"); }
    } catch { hideSearching(); toast("Search failed", "error"); }
  });
}

/* ------------------------------------------------------------- helpers */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/`/g, "&#96;");
}
function shorten(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function hostOf(u) {
  try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return shorten(u, 22); }
}

/* ---------------------------------------------------------------- init */
document.addEventListener("DOMContentLoaded", () => {
  setupDropzone();
  setupTabs();
  $("#history-btn").addEventListener("click", toggleHistory);
  refreshStats();
  setInterval(refreshStats, 30000);
});
