# Imify — Session Memory (MASTER)

> **PURPOSE**: Freebuff sessions are limited. This file is the single source of truth so any new
> session can resume instantly. READ `docs/PROBLEMS.md` for the issue ledger, **`docs/HANDOFF.md`
> for the exact in-flight debugging state (resume here)**, `docs/ARCHITECTURE.md` for design, and
> `README.md` for product docs. Update this file after every significant change.

**Last updated**: 2026-08-05 (session: FB logo / YouTuber accuracy / similar-results work)
**Project root**: `/run/media/irfan/models1/imify`
**Stack**: Python 3.13 (uv) · FastAPI · StrikeDB (user's own DB, port 6008) · SearXNG (8080) · Proxify (8081 API / 8888 tunnel) · vanilla JS UI

---

## 1. WHAT THIS PRODUCT IS

Imify = **independent reverse image search** (TinEye / Google Lens / Circle-to-Search style) with
**no paid APIs, no third-party scraping services, no proxies-for-hire** — the user's own stack:

- **SearchOS** (https://github.com/Mr-DS-ML-85/SearchOS) ladder: local index → live web reverse → honest "search not found".
- **StrikeDB** (https://github.com/Mr-DS-ML-85/StrikeDB) = own database (Redis-protocol + HNSW VSEARCH + VDEL).
- **Proxify** (https://github.com/Mr-DS-ML-85/proxify) = own anti-bot gateway (user's tool; we patch + docker-cp into it).
- No money spent: everything is self-hosted; Google/Bing/Yandex hit through proxify with real session cookies.

**Product goals (from user, verbatim-ish)**:
1. Find where an image appears online (exact copies) → removal requests / verification.
2. ALSO return visually-SIMILAR images from Google, Yandex, Bing AND SearXNG when no exact match.
3. Real-time feel (StrikeDB is fast; web phase bounded + cached).
4. Find faces (e.g., "my lost friend", "which sites leak my face") — WITHOUT full facial recognition;
   hash-based matching is the judge, not the engines.
5. Byte-upload ("circle to search") must work for images with NO public URL (private box).
6. Auto-update index in background (tracker re-searches uploaded images, ingests new matches).
7. Index removal + purge tools (admin).
8. Success-rate metrics (no pass/fail hiding).

---

## 2. STACK & PORTS (live)

| Service | Port | Who | How started |
|---|---|---|---|
| Imify API+UI | **8010** | uvicorn app.main:app | **systemd user service** `imify.service` (survives shell exit — see §6) |
| StrikeDB | **6008** | reference/StrikeDB dbstrike (WAL data/dbstrike.wal) | `scripts/start_stack.sh` or manual |
| SearXNG | **8080** | user's searxng | user-managed |
| Proxify REST /fetch | **8081** | docker container `proxy-orchestrator` | docker (user-managed) |
| Proxify CONNECT tunnel | **8888** | same container | docker |
| Static uploads server | **8099** | (seen in URLs) | user's |

**Live settings** (from `.env`/config):
- `storage_backend=strikedb`, `strikedb_port=6008`
- `public_base_url=''` (private box — NO public URL; byte-upload is the only web path)
- `proxify_fetch_url=http://127.0.0.1:8081`, `fetch_proxy_url=http://127.0.0.1:8888`
- `reverse_providers=searxng,bing,yandex,google`
- `web_fallback_local_min_hits=0` (**always run web layer even with local hits** — TinEye model; was 1)
- `web_phase_budget_seconds=18.0`, `web_providers_parallel=True`
- `web_results_cache_ttl_seconds=86400` (verified 24h; unverified cached 1h)
- `google_upload_enabled/bing_upload_enabled/yandex_upload_enabled=True`
- `cookie_file=/run/media/irfan/models1/imify/gui-cookies.txt` (real session cookies for FB/Google)
- `auth_mode=open` (admin endpoints need admin key in `data/admin_key.txt`)
- `auto_track_searches=True`

---

## 3. CODE MAP (what each file does)

- `app/config.py` — pydantic Settings, all env knobs.
- `app/main.py` — FastAPI bootstrap, wires repo/matcher/indexer/auth/engines.
- `app/api.py` — ALL endpoints (search, admin, trackers, removals, stats, health).
  **The search ladder lives here** (§5).
- `app/matcher.py` — fused pHash+dHash+aHash matching (the "TinEye judge"), `verify_web_hits`
  downloads candidates and hash-verifies them, `fuse_results` merges local+web, `_unverified`.
- `app/repo.py` — repository over Storage protocol; image meta, dedup maps, pending queue,
  trackers, **web-results cache** (`get_web_results`/`set_web_results`, TTL stored in entry), stats.
- `app/indexer.py` — background crawl/ingest workers + **tracker loop** (re-search tracked images).
- `app/engines/__init__.py` — `run_reverse_chain` (parallel providers within budget).
- `app/engines/yandex_images.py` — Yandex byte-upload (browser POST) + URL reverse + parser.
- `app/engines/bing_images.py` — Bing byte-upload (tunnel-first) + URL reverse + parser.
- `app/engines/google_images.py` — Google Lens byte-upload + URL reverse + parser.
- `app/features.py` — decode/extract phash/dhash/ahash/color/sha256.
- `app/resolve.py` — image_url → bytes resolution (FB comet pages, og:image, cookies, render).
- `app/cookies.py` — Netscape cookie file → per-host Cookie headers.
- `app/storage/{base,strikedb,sqlite,resp}.py` — Storage protocol (get/set/mget/incr/vadd/vsearch/vdel).
- `app/static/app.js` + `app.css` — the web UI (dropzone, results grid, empty-state, admin).
- `scripts/start_stack.sh` — detaches StrikeDB + app.
- `tests/` — 116 passing tests (`uv run pytest`).

---

## 4. THE PROXIFY PATCHES (critical, applied to user's tool)

Proxify checkouts: **reference** `/run/media/irfan/models1/imify/reference/proxify` (source of truth for
our patches) and **user's main** `~/Backup/searchapi/proxify` (synced by copying files). Container
`proxy-orchestrator` runs code **baked in image** + files we `docker cp` over it; restart needed after
each sync.

Patches applied to `proxy-orchestrator/strategies/` + `gateway/`:

1. **`page.route()` bug** (the big one): Playwright's `page.route()` is **async** and returns an
   `AsyncContextManager`. Upstream used `async with page.route(...)` → non-GET browser requests
   crashed/returned shells. Fixed to `async with await page.route(...)` in **gui_chrome.py** (line ~272)
   and **playwright_strategy.py** (line ~416). User confirmed the bug independently.
2. **`await r.text()` bug**: JS used `text: r.text()` (a Promise, serialized as `{}` → Pydantic 500).
   Fixed to `await r.text()` in both strategies.
3. **Binary-safe POST body**: browser fetch path now passes `body_b64` → `Uint8Array` (multipart bytes
   survive) and **pre-navigates** to the target first (same-origin fetch with cookies; fixes CORS
   `Origin: null`). NOTE: upstream later replaced this with **`page.route()` + `route.continue_(method=, post_data=)`**
   (native navigation w/ method+body override) — we adopted upstream's approach, our fetch approach is obsolete.
4. **`wait_selector` support (NEW)** — added so browser strategies wait for JS-rendered results:
   - `strategies/base.py`: `FetchRequest.wait_selector: Optional[str] = None`
   - `gateway/rest_api.py`: `FetchRequestBody.wait_selector` + pass-through
   - `gui_chrome.py` / `playwright_strategy.py`: after nav (GET or POST) `await page.wait_for_selector(wait_selector, timeout=min(12000, ...))`, and **re-capture `page.content()` after the wait** (`if not _used_fetch or waited`).

**How to re-sync proxify** (all three places):
```bash
cd /run/media/irfan/models1/imify/reference/proxify && git pull   # upstream updates
# apply the 4 patches above if git pull clobbered them
for f in strategies/base.py strategies/gui_chrome.py strategies/playwright_strategy.py gateway/rest_api.py; do
  docker cp proxy-orchestrator/$f proxy-orchestrator:/app/$f
  cp proxy-orchestrator/$f ~/Backup/searchapi/proxify/proxy-orchestrator/$f
done
docker restart proxy-orchestrator && sleep 12
```
**Always verify**: `grep -c 'await page.route' /app/strategies/gui_chrome.py` → 1, and `wait_selector` present.

---

## 5. SEARCH LADDER (api.py `/v1/search`) — HOW IT WORKS NOW

1. **Acquire query**: `file` (upload) / `image_url` (resolved via resolve_image_bytes incl. FB render) / `image_id`.
2. **Upload persisted** to `data/uploads/<sha>.<ext>` (public URL only if `public_base_url` set — it's not).
3. **Local first** (StrikeDB, sub-ms): `matcher.search_local` (exact scan < exact_scan_max_images, else VSEARCH ANN).
4. **Web layer ALWAYS runs** (`web_fallback_local_min_hits=0` → never skipped by a local hit):
   - If `query_public_url` exists → `run_reverse_chain(searxng,bing,yandex,google)` **in parallel** within `web_phase_budget_seconds`.
   - Else (private box) → **byte-upload chain** in api.py: **yandex → bing → google** (yandex first — least censored, browser POST proven). Each engine has its own route ladder inside the engine class.
5. **Verify**: `matcher.verify_web_hits` downloads each candidate through proxify tunnel (+cookies), extracts features, fused-hash compares to query. `fused <= exact` → `exact`; `<= near` → `near`; else → `unverified` (kept, low score, never counted as found).
6. **Cache**: verified matches cached 24h by query sha256; unverified-only outcomes cached 1h → repeat search is instant (`cached: True`).
7. **Background**: `_maybe_track_upload` creates a tracker; upload is indexed locally ONLY if `verified` or `learn=true` (a failed search must NOT pollute the index).
8. **Response**: `found` = any non-unverified; `fallback` local/web/none; unverified appended after verified with badges; UI shows grid when `results.length > 0` (even all-unverified = "similar candidates" — this is the TinEye similar model).

**Form fields**: `file`/`image_url`/`image_id`, `mode` (best|biggest|newest|changed|color|exact), `num_results`, `include_web`, `learn`. There is NO `local` field — local is always on.

---

## 6. RUN / RESTART / TEST

```bash
cd /run/media/irfan/models1/imify

# StrikeDB (if not up)
redis-cli -p 6008 PING || scripts/start_stack.sh   # start_stack.sh also starts app

# App — MUST use systemd (plain nohup dies with the shell in this environment):
systemctl --user restart imify.service
systemctl --user status imify.service
curl -s http://127.0.0.1:8010/health/live

# Proxify health
curl -s http://127.0.0.1:8081/health

# Tests (116)
uv run pytest
# JS
node --check app/static/app.js

# Live e2e (upload → search)
uv run python -c "
import httpx
raw = open('/tmp/yandex_image_proxy.jpg','rb').read()
r = httpx.post('http://127.0.0.1:8010/v1/search', files={'file':('q.jpg',raw,'image/jpeg')}, data={'include_web':'true'}, timeout=200)
d = r.json(); print('found:',d['found'],'results:',len(d['results']),'degraded:',d['degraded'])
"
```

**systemd unit** `~/.config/systemd/user/imify.service`:
```
[Service]
Type=simple
ExecStart=/bin/bash -c 'cd /run/media/irfan/models1/imify && exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8010'
WorkingDirectory=/run/media/irfan/models1/imify
Restart=always
RestartSec=5
StandardOutput=append:/run/media/irfan/models1/imify/data/app-8010.log
StandardError=append:/run/media/irfan/models1/imify/data/app-8010.log
[Install]
WantedBy=default.target
```
App logs: `data/app-8010.log`. NOTE: this environment kills backgrounded processes when the spawning shell exits — always use systemd or `setsid nohup ... & disown`.

---

## 7. KEY FIXES DONE (this & earlier sessions)

1. **Failed searches no longer pollute the index** — `_maybe_track_upload(..., verified=bool(verified_web))`; upload indexed only when verified/learn. Tests updated.
2. **No fake dead-end message** — UI `renderNotFound` now shows "all byte-upload engines blocked" ONLY when all 3 degraded; otherwise lists which engines checked. No "Try again later" dead-ends.
3. **Similar candidates visible** — UI shows the grid whenever `results.length > 0`; all-unverified → "No exact matches — N similar candidates" instead of "Search not found".
4. **Web layer always runs** — `web_fallback_local_min_hits=0` (was 1). Known images now surface remote links (`source_urls`) + similar candidates, not just the local copy.
5. **Real-time cache** — verified (24h) + unverified-only (1h) cached per sha256; stored TTL honored in `get_web_results`.
6. **Parallel providers** — `run_reverse_chain(..., parallel=True, budget_seconds=...)`.
7. **Yandex byte-upload = browser POST first** (gui_chrome + `wait_selector` on results containers). Tunnel/direct are fallbacks. Bing = tunnel-first (browser POST returns 404 shell for some images — NOT authoritative), then direct, then browser.
8. **Yandex parser** — handles rendered-DOM `<a href="SOURCE"><img src="CDN">` pairs; avatars.mds.yandex.net kept when a source page is attached.
9. **proxify patches** — see §4 (route await, r.text await, wait_selector, re-capture content).
10. **Systemd service** — app survives shell exit (see §6).
11. **Bing upload gate in api.py** now includes yandex (`or settings.yandex_upload_enabled`); yandex tried first.

---

## 8. OPEN ISSUES / CURRENT WORK (see docs/PROBLEMS.md + docs/HANDOFF.md for detail)

**The big one the user is hitting right now** (FULL detail in docs/HANDOFF.md):
- A YouTuber-face upload returned **24 results from Bing only** — all junk (animals/girls/objects, the person is male). Yandex/SearXNG/Google contributed nothing.
- FB-logo upload → "Engines checked the whole internet but found nothing: yandex-upload, google-upload" (Bing degraded with 400) — impossible for the most-indexed image on earth → **hidden bug suspected**.
- Patches APPLIED but NOT yet live-verified (this is where the next session starts):
  1. **Yandex `wait_selector` matched the EMPTY shell** (`.SerpList` exists on empty pages) → returned 0 hits before results rendered. FIXED in `app/engines/yandex_images.py` to `.serp-item, .CbirSites, .ImagesContentSearchItem, [class*="serp-item"]`. 🟡
  2. **Google upload used `use_browser: False`** → 0.3s shell. FIXED in `app/engines/google_images.py` to browser (gui_chrome) + `wait_selector` for Lens results. 🟡
  3. **Bing returns junk candidates** — verification marks them unverified (correct), but the user wants accurate SIMILAR results too; Bing 400s intermittently (rate-limit?). 🔴
  4. **SearXNG is NOT in the byte-upload path** — only URL-based chain. No public URL on this box → searxng never contributes. 🔴 (see PROBLEMS.md P4 for options)
  5. **Yandex "No matching results" for a YouTuber face is suspicious** — hypothesis: byte-upload needs a follow-up GET with `cbir_id` after the POST registers (see HANDOFF step 1). 🔴

**Other tracked issues**:
- `source_urls` are plain strings in image meta (`add_source_link`) — UI handles strings; earlier debug script assumed dicts (that was a debug script bug, not app bug).
- Index has only 2 images (id 1 FB pic, id 2 Elon webp) — most uploads are not indexed (by design: only verified/learn). Consider a "similar from local index" fallback using relaxed thresholds so the fast local DB always has something to show.
- Google face results are limited by Google (user fact) — Yandex is the face path.
- Bing browser-POST returns 404 shell for some images (don't trust it as verdict — fall through to tunnel/direct).
- **TEST STATE**: 116 passing (google + yandex tests updated for browser-first contract).

---

## 9. NEXT STEPS (suggested order)

1. **Live-verify Yandex fixed wait** on the YouTuber upload (should return real avatars.mds thumbnails + source pages). If still "No matching results", inspect `/tmp/yandex_yt.html` — check CbirIntent-Thumb present (upload registered) vs empty shell.
2. **Live-verify Google browser upload** — check for `"ou"` JSON or imgres links in returned HTML.
3. **Fix Bing 400** — investigate intermittent 400 (try `curl_cffi` persona / different Referer / retry); ensure junk candidates stay `unverified` and verified matches rank first.
4. **Route SearXNG into byte-upload path** for similar images (or document why it can't without a public URL).
5. **Local relaxed-similar fallback** in matcher (wider fused threshold → "similar" from own index) so the grid is never empty.
6. Re-run `uv run pytest` (116) + live e2e; update this file + docs/PROBLEMS.md.

---

## 10. CRITICAL USER PREFERENCES (do not violate)

- **No paid APIs / no scraping services / no buying proxies** — user is on free plan. Proxify (own tool) + SearXNG (own) only.
- **Don't clone into /tmp** (RAM) — clone/keep in project dir (`reference/`).
- **Don't kill port 6380** — another agent runs it. Imify uses 6008.
- **Use proxify with Google provider** or Google will block (sorry/captcha).
- **GUI cookies**: `gui-cookies.txt` is the real session cookie file (FB/Google).
- **Yandex = least censored** (faces) — prefer it for people.
- **"We don't believe in pass or fail"** — need success-rate metrics and honest per-hops logging (resolution hops shown in UI).
- User wants **similar AND exact** images in real time; failed searches must NOT index the failed image.
- User prefers hash-based matching (own matcher) over text detection / facial-recognition services.
