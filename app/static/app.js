/* Imify — search UI logic (vanilla JS, zero dependencies). */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  lastForm: null,       // { mode, image_id }
  currentResults: [],
  lastQueryThumb: null,
};

/* ------------------------------------------------------- verdict taxonomy */
// Mirrors app/models.py: which match types CONFIRM the image / the person,
// and how the grid groups them (TinEye copies, FaceCheck person, Lens similar).
const CONFIRMED = new Set(["exact", "near", "modified", "face", "changed", "color"]);
const SECTIONS = [
  { key: "copies", title: "Copies of your image", hint: "Same picture — exact, resized, re-encoded, cropped or edited (hash + keypoint verified)", types: ["exact", "near", "modified", "changed", "color"] },
  { key: "faces", title: "Same person, other photos", hint: "Face-verified identity matches (ArcFace) — a different photo of the same face", types: ["face"] },
  { key: "similar", title: "Visually similar", hint: "Similar content (CLIP) or the same on-image text — leads, not confirmed copies", types: ["similar", "ocr", "clip", "text", "fulltext"] },
  { key: "unverified", title: "Unconfirmed candidates", hint: "Engines suggested these, but none of our checks confirmed them", types: ["unverified"] },
];
const VERDICT_LABEL = {
  exact: "exact copy", near: "near copy", modified: "edited copy", changed: "changed",
  face: "same person", similar: "similar", ocr: "same text", color: "color", unverified: "unverified",
};
function sectionOf(r) {
  return (SECTIONS.find((s) => s.types.includes(r.match_type)) || SECTIONS[3]).key;
}

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

/* search by description (AI) or by on-image text (OCR) */
async function searchText(kind) {
  const input = $("#text-input");
  const q = input.value.trim();
  if (!q) return;
  const endpoint =
    kind === "ocr" ? "/v1/search/ocr"
    : kind === "text" ? "/v1/search/text"
    : "/v1/search/fulltext";
  state.lastForm = { textKind: kind, textQuery: q };
  showSearching();
  const fd = new FormData();
  fd.append("query", q);
  fd.append("num_results", "60");
  try {
    const r = await fetch(endpoint, { method: "POST", body: fd });
    if (!r.ok) {
      hideSearching();
      let msg = "Search failed";
      try { msg = (await r.json()).detail || msg; } catch { /* ignore */ }
      toast(msg, "error");
      $("#search-area").hidden = false;
      return;
    }
    const data = await r.json();
    renderResults(data, null);
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
  const counts = { copies: 0, faces: 0, similar: 0, unverified: 0 };
  for (const r of results) counts[sectionOf(r)] += 1;
  const found = data.found !== false && results.some((r) => CONFIRMED.has(r.match_type));
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const titleParts = [];
  if (counts.copies) titleParts.push(plural(counts.copies, "copy found", "copies found"));
  if (counts.faces) titleParts.push(plural(counts.faces, "photo of this person", "photos of this person"));
  if (counts.similar) titleParts.push(`${counts.similar} similar`);
  if (!found && results.length) titleParts.unshift("No confirmed matches");
  if (!results.length) titleParts.push("Search not found");
  $("#results-title").textContent = titleParts.join(" · ");
  const providerBadges = (data.providers || []).map((p) => p.toUpperCase()).join(" · ");
  let degradedLine = "";
  const deg = data.degraded || [];
  if (deg.length) degradedLine = ` · <span style="color:var(--warn)">${deg.map((d) => d.replace(/_/g, " ").replace("provider:", "")).join(", ")}</span>`;
  const cachedLine = data.cached ? ` · <span class="badge near" title="These matches were served from the verified-results cache — no fresh web crawl was needed">cached</span>` : "";
  $("#results-sub").innerHTML = `${data.took_ms} ms · ${providerBadges}${degradedLine}${cachedLine}`;
  state.currentResults = results;
  state.lastData = data;
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
    detail = "Engines returned candidate pages, but none of them passed our checks (perceptual hash, keypoint geometry, face identity or content similarity). Only results confirmed by our own matcher count as matches.";
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
  // reverseimagesearch-style engine tabs + Reversely-style category tabs over
  // the ranked list. "All" shows the merged grid; each tab filters to its group.
  const byEngine = (state.lastData && state.lastData.by_engine) || {};
  const byCat = (state.lastData && state.lastData.by_category) || {};
  const engKeys = Object.keys(byEngine);
  const catKeys = Object.keys(byCat).filter((k) => (byCat[k] || []).length);
  const engTabs = $("#engine-tabs");
  const catTabs = $("#cat-tabs");
  if (engKeys.length > 1) {
    engTabs.hidden = false;
    engTabs.innerHTML =
      `<button class="cat active" data-eng="__all">All</button>` +
      engKeys.map((e) => `<button class="cat" data-eng="${escapeAttr(e)}">${escapeHtml(e)}<small>${byEngine[e].length}</small></button>`).join("");
  } else { engTabs.hidden = true; engTabs.innerHTML = ""; }
  if (catKeys.length > 1) {
    catTabs.hidden = false;
    catTabs.innerHTML =
      `<button class="cat active" data-cat="__all">All</button>` +
      catKeys.map((c) => `<button class="cat" data-cat="${c}">${escapeHtml(c)}<small>${byCat[c].length}</small></button>`).join("");
  } else { catTabs.hidden = true; catTabs.innerHTML = ""; }
  for (const [nonAll, fn] of [[".cat[data-eng]", () => engTabs], [".cat[data-cat]", () => catTabs]]) {
    (fn() || {}).querySelectorAll && fn().querySelectorAll(".cat").forEach((b) => {
      b.addEventListener("click", () => {
        const parent = b.parentElement;
        parent.querySelectorAll(".cat").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        renderCardSet(results, byEngine, byCat, b.dataset.eng, b.dataset.cat);
      });
    });
  }
  renderCardSet(results, byEngine, byCat, "__all", "__all");
}

function renderCardSet(results, byEngine, byCat, engSel, catSel) {
  const grid = $("#grid");
  grid.innerHTML = "";
  let list = results;
  if (engSel && engSel !== "__all") list = list.filter((r) => (r.engine || "").trim() === engSel);
  if (catSel && catSel !== "__all") list = list.filter((r) => (r.category || (CONFIRMED.has(r.match_type) ? "duplicate" : "candidate")) === catSel);
  if (!list.length) { grid.innerHTML = `<div class="empty-tab">No results in this view.</div>`; return; }
  for (const sec of SECTIONS) {
    const items = list.filter((r) => sectionOf(r) === sec.key);
    if (!items.length) continue;
    const head = document.createElement("div");
    head.className = "section-head";
    head.innerHTML = `<h3>${escapeHtml(sec.title)}</h3><span class="count">${items.length}</span><p>${escapeHtml(sec.hint)}</p>`;
    grid.appendChild(head);
    for (const r of items) renderCard(r, grid);
  }
}

function renderCard(r, grid) {
    const card = document.createElement("article");
    card.className = "card";
    const conf = r.confidence != null ? r.confidence : 0;
    const meter = conf ? conf : (r.hamming != null ? Math.max(6, Math.round((1 - r.hamming / 32) * 100)) : 0);
    const size = r.width && r.height ? `${r.width}×${r.height}` : "";
    const engine = r.engine || (r.source === "web" ? "web" : "index");
    const titleHref = r.page_url || r.url;
    const resolution = (r.resolution || []).length
      ? `<div class="res-line" title="How this candidate URL was resolved to pixels">${r.resolution.map((h) => `<span class="hop">${escapeHtml(h)}</span>`).join(`<span class="hop-arrow">→</span>`)}</div>`
      : "";
    const note = r.note
      ? `<div class="res-note ${r.match_type === "unverified" ? "warn" : ""}">${escapeHtml(r.note)}</div>`
      : "";
    // which independent checks back this verdict (hash / keypoints / face / CLIP)
    const sig = [];
    for (const m of r.ai_modalities || []) {
      if (m === "hash" && r.hamming != null) sig.push(`hash Δ${r.hamming}`);
      else if (m === "geometry" && r.geo_inliers != null) sig.push(`${r.geo_inliers} keypoints`);
      else if (m === "face" && r.face_sim != null) sig.push(`face ${Math.round(r.face_sim * 100)}`);
      else if (m === "clip" && r.clip_sim != null) sig.push(`content ${Math.round(r.clip_sim * 100)}`);
      else sig.push(m);
    }
    const signals = sig.length ? `<div class="signals" title="Independent checks behind this result">${sig.map((x) => `<span>${escapeHtml(x)}</span>`).join("")}</div>` : "";
    // AI alt-text / description (self-hosted caption when enabled)
    const caption = r.caption
      ? `<div class="caption" title="AI description">${escapeHtml(r.caption)}</div>`
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
        <span class="badge ${escapeAttr(r.match_type)}">${escapeHtml(VERDICT_LABEL[r.match_type] || r.match_type)}</span>
        ${engine ? `<span class="engine-tag">${escapeHtml(engine)}</span>` : ""}
        <img loading="lazy" src="${escapeAttr(r.thumb || r.url)}" alt="${escapeAttr(r.caption || r.title || "")}" onerror="this.parentElement.style.background='#0d1326';this.style.display='none'">
        <div class="hover">
          <button class="act visit">${r.page_url ? "Open post/article" : "Open image"}</button>
          ${r.page_url ? `<button class="act page">Source page</button>` : ""}
        </div>
      </div>
      <div class="body">
        <a class="title" href="${escapeAttr(titleHref)}" target="_blank" rel="noopener">${escapeHtml(r.title)}</a>
        <div class="row">
          ${size ? `<span>${size}</span><span class="dot"></span>` : ""}
          <span>${r.source_count > 1 ? r.source_count + " sources" : r.source === "both" ? "index + web" : r.source}</span>
          <span class="dot"></span>
          <span>${r.match_type && r.match_type !== "unverified"
            ? (conf + "% " + (r.match_type === "face" ? "face match" + (r.face_tier ? " · " + escapeHtml(r.face_tier) : "")
                : r.match_type === "similar" ? "similar" : "confidence"))
            : (r.hamming != null ? "Δ" + r.hamming : "no match")}</span>
        </div>
        ${signals}
        ${resolution}
        ${caption}
        ${sourcesBlock}
        ${note}
        <div class="match-meter"><i style="width:0%"></i></div>
      </div>`;
    card.querySelector(".match-meter i").style.width = meter + "%";
    // "Open source" always targets the ORIGINAL WEBSITE (page_url) — the whole
    // point of reverse-search is finding where the photo lives, not a transient
    // CDN copy. Fall back to the resolved URL only when no page link exists.
    // Local index entries sourced from a file:// path have no web URL, so open
    // the locally-served thumbnail instead (never a file:// in the browser).
    const openSource = openableUrl(r);
    card.querySelector(".visit").addEventListener("click", () => window.open(openSource, "_blank"));
    card.querySelector(".visit").title = openSource;
    const pageBtn = card.querySelector(".page");
    if (pageBtn) pageBtn.addEventListener("click", () => window.open(openableUrl(r, true), "_blank"));
    if (r.image_id) {
      card.addEventListener("dblclick", () => searchById(r.image_id));
      card.title = "Double-click to search this image";
    }
    grid.appendChild(card);
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

  // AI / OCR text search
  let textKind = "text";
  document.querySelectorAll("#text-mode .tm").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#text-mode .tm").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      textKind = btn.dataset.kind;
      const placeholder = textKind === "ocr" ? "Search by the text on the image, e.g. 'hello world'" : "Describe what you're looking for, e.g. 'a red car on a beach'";
      $("#text-input").placeholder = placeholder;
    });
  });
  $("#text-go").addEventListener("click", () => searchText(textKind));
  $("#text-input").addEventListener("keydown", (e) => { if (e.key === "Enter") searchText(textKind); });
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
// Return a browser-openable URL for a result. Prefer the original page, then
// the image URL — but only if it's http(s). Local index entries sourced from
// a file:// path (or with no web page) have no openable web URL, so fall back
// to the locally-served thumbnail instead of a dead file:// link.
function openableUrl(r, preferImage) {
  const wantPage = r.page_url && /^https?:\/\//i.test(r.page_url);
  const wantImg = r.url && /^https?:\/\//i.test(r.url);
  if (preferImage) return wantImg ? r.url : r.thumb || r.url;
  if (wantPage) return r.page_url;
  if (wantImg) return r.url;
  return r.thumb || r.url;
}

/* ---------------------------------------------------------------- init */
document.addEventListener("DOMContentLoaded", () => {
  setupDropzone();
  setupTabs();
  $("#history-btn").addEventListener("click", toggleHistory);
  refreshStats();
  setInterval(refreshStats, 30000);
});
