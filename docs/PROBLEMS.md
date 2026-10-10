# Imify — Problem Ledger (docs/PROBLEMS.md)

> Companion to `memory.md`. Every known bug, its root cause, the fix applied, and its verification status.
> Status legend: ✅ FIXED+VERIFIED · 🟡 FIXED-NOT-VERIFIED-LIVE · 🔴 OPEN · ⏸ DEFERRED/BY-DESIGN

---

## P1 · Yandex byte-upload returns "No matching results" even for common images (FB logo, YouTuber)  🔴/🟡

**Symptom**: Upload FB logo / YouTuber face → Yandex says "No matching results" (0 hits). Impossible for
the FB logo (the most-indexed image on the internet). User: "thats impossible i think hidden bug".

**Root cause (found)**:
- The old `wait_selector` was `'.CbirSites, .SerpList, [class*="CbirSites"], [class*="SerpList"]'`.
  **`.SerpList` EXISTS on the EMPTY page** (`<div class="SerpList SerpList_emptyPage">`) → proxify
  returned the page in ~2.7s BEFORE Yandex's JS rendered any results → 0 hits parsed. Live proof:
  response contained `SerpList_emptyPage`, `CbirIntent`, `No matching results.`, 0 imgs, 0 avatars.
- A 15.2s retest with results-only selector still showed "No matching" for the YouTuber image — so either
  results genuinely never render for byte-upload without a cbirId, or the upload isn't registering
  (see P6), or this specific image truly has no Yandex hit.

**Fix applied (🟡 not yet live-verified on a real match)**:
`app/engines/yandex_images.py` `_upload_via_fetch` wait_selector → `'.serp-item, .CbirSites, .ImagesContentSearchItem, [class*="serp-item"]'`
(results-only containers; empty shell has none of these → wait up to 12s for real results).

**Verify**: upload a widely-indexed image (e.g. a known YouTube thumbnail) and check for
`avatars.mds.yandex.net` URLs + source pages in the returned page (`/tmp/yandex_yt.html` on disk).

---

## P2 · Google byte-upload returns 0 hits in 0.3s (shell)  ✅ RESOLVED 2026-10-10

**Symptom**: `google-upload` returned 0 hits — every route got Lens's JavaScript-only shell.

**Root causes** (three, stacked):
1. Lens binds results to the browser session that uploaded. Any non-browser upload (curl_cffi
   tunnel, proxify strategy pipeline, `fetch()` from the Lens UI — cross-origin) gets the shell.
2. The parser only knew the old Images JSON (`"ou"`/`imgres`), not Lens result records.
3. Proxify flagged real Google result pages as blocked: its hard-block marker `/sorry/` matched
   Google's own inline fetch wrapper (`indexOf("/sorry/index")`) → "All strategies exhausted".

**Fix**: proxify `/fetch` `in_page_upload` — GUI Chrome opens lens.google.com and submits a real
`<form>` with the file, returns the raw `/search?vsrid=…` page (proxify `4beae68`); block markers
are matched outside inline script bodies; Imify `parse_lens_results` extracts
`["thumb",h,w],["image",h,w] … "2003":[null,id,"page","title"]` records; Google engine tries the
in-page route first (Imify `70e5b25`, `ac3694e`).

**Verified**: two uncached `/v1/search` runs → 7 and 6 verified `google lens` matches (WSJ, Reuters,
BBC, PCMag…) in ~18 s; Google not degraded.

---

## P3 · Bing byte-upload returns 24 JUNK candidates (animals/girls/objects for a male face)  🔴

**Symptom**: YouTuber face → 24 results "from bing only", all inaccurate. Bing 400s intermittently
("bing-upload failed via all paths ... 400 Bad Request") on other attempts.

**Analysis**:
- Bing's byte-upload (POST /images/search, `image_url=<b64>`) via tunnel works sometimes (302→200,
  parses murl/purl pairs) and 400s other times (intermittent — rate-limit or fingerprint).
- The 24 candidates ARE what Bing's visual search returns (visually-similar junk). Our hash matcher
  correctly flags them `unverified` (that's why the user sees "24 unverified / similar"). This part is
  working as designed — Bing is just a bad similar-source.
- **User expectation**: similar images should come from ALL engines and be ACCURATE. Bing junk should
  not dominate the grid. Verified (exact/near) must rank first (they do), but with 0 verified the grid
  is all junk.

**Open work**:
- Make Bing's 400 deterministic (retry, curl_cffi persona, different endpoint `?form=HDRSC2`?).
- Ensure Bing candidates never outrank verified results (already sorted; double-check fuse_results).
- Consider dropping Bing's murl-only fallback entries (no page URL) or capping Bing's candidate count
  below Yandex/Google's so junk doesn't flood the grid.

---

## P4 · SearXNG never contributes for byte-uploads (no public URL on this box)  🔴

**Symptom**: `reverse_providers=searxng,bing,yandex,google` but SearXNG only runs in the URL-based chain
(`run_reverse_chain`), NOT the byte-upload path. Private box → `public_base_url=''` → no public URL →
searxng.reverse_search (needs `image_url`) can't run → user sees "yandex-upload, google-upload" only.

**Fix ideas** (pick one):
1. SearXNG engines that accept `image_url` — but they still need a REACHABLE url; localhost won't work.
2. After byte-upload engines return candidates, extract the best candidate's **page title/alt text** and
   run a SearXNG keyword image search to surface more similar images (text-assisted similar).
3. Serve the upload via proxify's tunnel to SearXNG (set `image_url` to a proxify-fetchable local URL) —
   unproven; engines may still reject localhost.
4. Document limitation: SearXNG contributes when `public_base_url` is set.

---

## P5 · UI dead-end message + hidden similar candidates (FIXED)  ✅

**Symptom**: "Search not found" / "byte-upload reverse engines are blocked right now ... Try again later"
even when similar candidates existed.

**Fixes (✅ verified)**:
- `app/static/app.js` `renderResults`: grid renders whenever `results.length > 0`; all-unverified →
  title "No exact matches — N similar candidates".
- `renderNotFound`: "all engines blocked" only when all 3 upload engines degraded; otherwise
  "Engines checked the whole internet but found nothing: <which ran>".
- Live e2e: kitten search repeated → `cached: True`, 24 similar in 0.1s, grid shows them.

---

## P6 · Byte-upload may not register cbirId (Yandex) / upload size issues  🔴-suspect

**Observation**: Yandex browser POST returns `CbirIntent-Thumb` (upload registered) in some runs but the
page then shows `SerpList_emptyPage` + "No matching results." for a YouTuber's face. Either:
- (a) results need a second AJAX round-trip after cbirId registration that our single POST+wait misses, or
- (b) 1MB uploads get rejected / re-encoded and Yandex genuinely finds nothing.

**Next step**: with `wait_selector` fixed (P1), re-run. If still empty, capture the page and look for a
`cbirId` to build the classic follow-up GET `https://yandex.com/images/search?rpt=imageview&cbir_id=<id>`
(or the `cbir-page` JSON endpoint) and parse that instead.

---

## P7 · Bing browser-POST returns 404 shell (routing lesson)  ✅ (routing fixed)

**Observation**: Bing `/fetch` browser POST → status 404, 9KB shell, 0 murl pairs. **Do NOT treat a
browser POST as Bing's authoritative verdict.** Bing routing is now tunnel → direct → browser (only
first route yielding hits wins). Verified by tests.

---

## P8 · Failed searches used to pollute the index  ✅

**Symptom**: search fails → image still indexed → next search returns the upload's own local path instead
of remote links.

**Fix**: `_maybe_track_upload(..., verified=bool(verified_web))` — local indexing only when verified
online or `learn=true`. Tests updated (test_fallback, test_removals_tracks). ✅ 116 tests pass.

---

## P9 · App dies when the spawning shell exits  ✅

**Root cause**: this environment kills backgrounded processes (nohup/setsid from a basher shell dies).
**Fix**: systemd user service `imify.service` (Restart=always). ✅ Verified active + healthy.

---

## P10 · UI previously rendered only icons / broken layout (old)  ✅
Fixed in an earlier session; UI verified serving: `/` 200, `/static/app.js`+`app.css`+favicon 200,
all required DOM ids present (dropzone, results-area, grid, empty-state, learn-toggle).

---

## P11 · Yandex parser misses modern rendered-DOM results  ✅ (patched)

Old parser only read `cbir-page` JSON + bare img/a tags; modern page thumbnails live on
`avatars.mds.yandex.net` inside `<a href="SOURCE"><img src="CDN">` pairs, and the old code hard-filtered
avatars.mds. Fix: pair extraction + keep CDN img when a source page is attached; junk filter still drops
passport/support/tune links. Tests updated.

---

## P12 · `source_urls` displayed as strings (UI)  ✅
`SearchResult.source_urls: list[str]`; UI renders them as host links ("found on N sites"). Earlier
"dict" assumption was only in a throwaway debug script.

---

## P13 · Web-results cache TTL not honored on read  ✅
`get_web_results` used the caller's TTL, ignoring the TTL stored at write time (unverified 1h vs
verified 24h). Fixed: stored `ttl` wins. `tests/test_realtime_cache.py` covers it.

---

## P14 · Real-time repeat searches  ✅
Cache: verified 24h / unverified-only 1h per query sha256. Live e2e: kitten repeat = 0.1s `cached: True`.

---

## P15 · Local index only has 2 images; empty grid for brand-new images  🔴/⏸
Index: id 1 (FB profile pic, 3 source links), id 2 (Elon webp, 2 sources). Most uploads are NOT indexed
(by design — only verified/learn). For a brand-new image with no web presence the grid is empty →
"Search not found". **Idea (deferred)**: relaxed local "similar" fallback (wider fused threshold) so the
fast local DB always surfaces something, matching the "strikeDB is super fast" expectation.

---

## Tests that matter
- `uv run pytest` → **116 passed** (1 deprecation warning).
- `node --check app/static/app.js` → OK.
- Engine routing covered in `tests/test_yandex.py` (incl. new browser-first + tunnel-fallback tests),
  `tests/test_bing*.py`-style, `tests/test_google.py`, `tests/test_realtime_cache.py`,
  `tests/test_fallback.py`, `tests/test_removals_tracks.py`, `tests/test_cookies.py`.
