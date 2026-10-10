# Imify — Session Memory (MASTER)

> **PURPOSE**: Freebuff sessions are limited. This file is the single source of truth so any new
> session can resume instantly. READ `docs/PROBLEMS.md` for the issue ledger, **`docs/HANDOFF.md`
> for the exact in-flight debugging state (resume here)**, `docs/ARCHITECTURE.md` for design, and
> `README.md` for product docs. Update this file after every significant change.

**Last updated**: 2026-10-10 (accuracy overhaul — see §0). Previous: 2026-08-06 (CPU-only policy enforced: NO CUDA/GPU anywhere; AI/face/CLIP merged into /v1/search grid; models2 is NOT live)
**Project root**: `/run/media/irfan/models/imify` (moved here by 2026-10-10; older notes below say `models1`/`models2` — both are GONE. `.env` no longer hard-codes absolute model/cookie paths; defaults resolve from the project root). Previously: `/run/media/irfan/models1/imify` — **THE LIVE ONE** (confirmed by user, who is non-root; `models2/imify` is a root-owned dir with only deploy/reference and is NOT the project — do NOT edit/use it). Old notes saying models2 is live are WRONG.
**Stack**: Python 3.13 (uv) · FastAPI · StrikeDB (user's own DB, port 6008) · SearXNG (8080) · Proxify (8081 API / 8888 tunnel) · vanilla JS UI · OpenCV FaceDetectorYN+ArcFace · CLIP (ONNX int8) · Real-ESRGAN/BiRefNet (enhance) — **ALL models CPU-only (plain `onnxruntime`, DNN_TARGET_CPU); CUDA/onnxruntime-gpu is banned** (a GPU crash once killed the user's nvidia driver, requiring a reinstall)

---

## 0. 2026-10-10 SESSION — ACCURACY OVERHAUL (read first)

**Root-cause bugs fixed (all measured on this box's own data):**
- **Face-ID was effectively random.** `cv2.FaceRecognizerSF.alignCrop` wants the FULL 15-value YuNet row
  (landmarks at offset 4) but got only the 10 keypoints → misaligned crops; `feature()` also fed raw
  0-255 pixels (SFace preprocessing) into ArcFace. Different people scored median 0.74 (max 0.91),
  ABOVE true matches. Now: similarity-transform alignment onto the ArcFace template + `(x-127.5)/127.5`
  RGB via onnxruntime → same person ≥ 0.88, different people ≤ 0.30 (median 0.01).
  `face_id_same_threshold` 0.50→**0.42**. Embeddings carry `ImageMeta.face_model` = `FaceId.MODEL_TAG`;
  old ones are ignored and re-embedded by the indexer's new **`ai-backfill` thread** (uses the original
  upload when present, else the thumbnail). YuNet `top_k` was being passed `min_face_size` (48) → now 5000;
  min size + score floor enforced; YuNet calls are locked (not thread-safe).
- **`match_identity` mutual-NN dropped every photo of a person after the first** (copies' runner-up was each other). Now threshold-only, best face per image. `face_id_match_margin` removed.
- **AI grid never returned anything**: `_local_ai_grid` passed `ai_modalities`/`face_sim`/... to `SearchResult`, which lacked those fields → TypeError swallowed by `except: pass`. Fields added.
- **Verification budget was not enforced**: budget `break` inside `with ThreadPoolExecutor` still waited for every straggler (both `verify_web_hits` and `_byte_upload_all`). Now `shutdown(wait=False, cancel_futures=True)`.
- CLIP grid floor was 0.32 (CLIP puts unrelated photos at ~0.50). Calibrated: copies ≥ 0.90, unrelated median 0.50 / p90 0.66.
- CSP lacked `blob:` → uploaded query preview never rendered. Tests read the live `.env` (now `_env_file=None` + tmp uploads dir).
- 3 Clipper + 2 FaceId instances were loaded → one shared instance each (`main.py`).

**New verdict ladder** (`Matcher.verify_web_hits`, also local): 1) hash incl. MIRRORED query + CLIP
agreement (`clip_verify_min_sim` 0.80) → exact/near; 2) **SIFT+RANSAC geometry** (`app/geometry.py`) →
`modified` (crops/borders/overlays/rotation; ≥40 inliers alone, 18-39 needs CLIP ≥ 0.80 — real copies
measured 58-330, unrelated ≤ ~30); 3) **face** → `face` (same person, other photo); 4) CLIP ≥ 0.78 →
`similar` (lead, NOT found). `models.CONFIRMED_MATCH_TYPES` decides `found`; `MATCH_FAMILY_RANK` orders
copies → face → similar → unverified. Local search got the same CLIP gate + `search_local_geometric`.
**Face-crop upload**: when the main face is <20% of the frame, a crop is also sent to Yandex
(`yandex-face` task, `face_crop_upload`). UI groups results into "Copies / Same person / Visually
similar / Unconfirmed" sections with evidence chips. Tests: 134 passing (`tests/test_accuracy.py` new).

**Live stack state found 2026-10-10:** `imify.service` crash-loops (unit still points at
`/run/media/irfan/models2/imify`); StrikeDB :6008 and proxify :8081/:8888 were NOT running; SearXNG up.
Not touched — needs the user's go-ahead (fix the unit's paths to `/run/media/irfan/models/imify`).

**2026-10-10 (part 2) — engines, rotation, bandwidth, stack:**
- Stack started from `reference/`: `scripts/start_strikedb.sh` (CPU-only `DBSTRIKE_GPU=cpu`; `data/lib/libnvrtc.so.12`
  symlink → pinokio miniforge, needed only to LINK) and `scripts/start_proxify.sh [--recreate]` (existing image
  `proxy-orchestrator-orchestrator:latest`, 127.0.0.1:8081/8888, GUI profile in volume `proxify-gui-profile`, cookies =
  `scripts/filter_cookies.py` allowlist of search/FB domains → `data/proxify-cookies.txt`; NEVER mount the full
  gui-cookies.txt — it holds LastPass/Proton/ChatGPT sessions and is committed in git history!).
- **Text-guided discovery** `app/engines/text_images.py`: best guess from reverse-hit titles/URL slugs → SearXNG
  engines in 4-engine chunks with `timeout_limit` (icons only for logo-like queries) → blocked engines retried in
  proxify gui_chrome (Bing parser + generic). Live: Elon photo → 4 face-verified other photos via pinterest/findthatmeme.
  Blocked live: google images (captcha even in browser), qwant, brave, duckduckgo. Not in this SearXNG: startpage,
  mojeek, deviantart, library of congress.
- Rotation: EXIF transpose in decode + 8 orientation hash variants (≤0.20 fused for non-identity).
- Bandwidth: thumbnail triage before full download, per-URL byte cache `vc:` (7d), preview dir capped.
- Cookies: exporter wrote Chromium 1601-epoch seconds (+369y) → nothing ever expired. Fixed in
  `reference/proxify/.../brave_cookies.py` and normalized on read; `/v1/stats.cookies` shows live/expired/soonest.
- Pre-existing bugs fixed: Bing parser glued `","murl":"…` onto page URLs; WebP uploads gave Bing off-topic results
  (now re-encoded to JPEG for engines); SearXNG reverse fallback text-searched "reverse image search" (junk
  candidates) — removed; engine label duplication on merge.

**2026-10-10 (part 3) — service + stealth (no CAPTCHA solving, by user's choice):**
- `~/.config/systemd/user/imify.service` repointed to `/run/media/irfan/models/imify` (backup `.bak`); runs
  `.venv/bin/python -m uvicorn`, ExecStartPre = start_strikedb.sh + start_proxify.sh (idempotent). Active on :8010.
- BrowserPilot (`reference/BrowserPilot`) ideas adopted: (1) ONE honest persona — GUI browser is Chromium 153/Linux, so
  UA + client hints say exactly that (`start_proxify.sh` env; `gui_chrome.py` derives brands/platform from PERSONA_UA,
  file is MOUNTED into the container so recreation keeps it); (2) back off instead of pushing through:
  `app/engine_guard.py` — per-engine jittered pacing (google 12s, yandex 4s, bing 3s), block detection, exponential
  cooldown 10min→6h persisted in storage (`eg:*`), google daily budget; wired into live byte-uploads, tracker uploads
  and text-discovery browser retries; `/v1/stats.engine_guard` shows state; skipped engines appear as `cooldown:<engine>`.
  BrowserPilot's solve_captcha is deliberately NOT used.
- Profile volume gotcha: stale Chromium SingletonLock from an old container hostname blocks GUI Chrome —
  start_proxify.sh clears it before launch.

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
  `verify_web_hits(..., query_clip)` takes the CLIP gate hook (NOT yet wired at the fused gate).
- `app/face.py` — OpenCV 5.0 `FaceDetectorYN` (YuNet) + `FaceRecognizerSF` (ArcFace 512-dim).
  **CPU-only**: `target = cv2.dnn.DNN_TARGET_CPU` always; the CUDA branch was removed (config `device` is accepted but ignored). `provider` reports "CPU".
- `app/clip.py` — CLIP vision encoder (ONNX int8, 224 BICUBIC, L2-normalized 512-dim), lazy + auto-download,
  `match()` cosine, `same_threshold=0.80`. CPU-only (onnxruntime no GPU EP; opencv no CUDA build).
- `app/models.py` — `ImageMeta.clip_embedding` (512-dim list) + `ImageMeta.face_embedding` (512-dim).
- `app/repo.py` — repository over Electron; images, dedup maps, pending queue,
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

# StrikeDB — CRITICAL: cwd MUST be project root OR pass env. A bare `dbstrike.wal`@root is the
# stale 2KB file; REAL data is `data/dbstrike.wal` (2.0MB). The binary links libnvrtc.so.12
# (CUDA 12 lib) which the box's CUDA 13.1 does NOT provide in ldconfig — export LD_LIBRARY_PATH
# to the ASC venv copy of it, else dbstrike exits "cannot open shared object file". Correct launch:
export LD_LIBRARY_PATH=/run/media/irfan/models1/ASC/asc-venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
DBSTRIKE_WAL=data/dbstrike.wal setsid nohup reference/StrikeDB/target/release/dbstrike 127.0.0.1:6008 >> data/dbstrike.log 2>&1 &
redis-cli -p 6008 PING

# App — NO pkill; kill per-PID (pgrep -f), then:
setsid nohup .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 >> data/app-8010.log 2>&1 &
curl -s http://127.0.0.1:8010/health/live
curl -s http://127.0.0.1:8010/v1/stats   # face: enabled/CPU, clip: enabled/CPU, storage strikedb

# Proxify health
curl -s http://127.0.0.1:8081/health

# Tests — CAREFUL: `uv run pytest`/onnx import triggers the local GPU stack; the CPU-only
# venv now lists ONLY [CPUExecutionProvider]. Do NOT run model-loading tests on a fresh driver
# until the box is confirmed stable.
.venv/bin/python -m pytest tests/ 2>&1 | tail -3; echo "EXIT=$?"
```

**CPU-only policy (2026-08-06)**: `onnxruntime-gpu` was replaced with plain `onnxruntime` (venv now
reports `['AzureExecutionProvider','CPUExecutionProvider']` — no CUDA). `clip.py` and `enhance.py`
pass `providers=["CPUExecutionProvider"]`; `face.py` forces `DNN_TARGET_CPU`. A GPU/driver crash
earlier took the user's nvidia driver down entirely (needed reinstall) — NEVER re-add CUDA. The
`.env` `FACE_ID_MODELS_DIR`/`CLIP_MODELS_DIR` were pointing at root-owned `models2/imify/data` →
fixed to `models1/imify/data/models/{face,clip}` (models already on disk there).

**AI layers merged into /v1/search** (2026-08-06): `_local_ai_grid()` in `app/api.py` merges local
face/CLIP/OCR matches into the SAME main search grid (provider "ai") instead of only separate
endpoints — a blurry face or text logo now surfaces in the default search. Returns `SearchResult`
objects, dedupes against hash hits, sorted by score, capped.

**systemd unit** `~/.config/systemd/user/imify.service`:
```
[Service]
Type=simple
ExecStart=/bin/bash -c 'cd /run/media/irfan/models1/imify && exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8010'
WorkingDirectory=/run/media/irfan/models1/imify
Restart=always
RestartSec=5
StandardOutput=append:/run/media/irfan/models1/imify/data/app-8010.log
StandardError=append:/run/media/irfan/models1/imify/data/app-8010.log
[Install]
WantedBy=default.target
```
App logs: `data/app-8010.log`.

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
12. **CLIP layer built & LIVE** (`app/clip.py`): CLIP_ENABLED=true in `.env`, 512-dim L2 embeddings persisted on `ImageMeta`, `/v1/search/clip` + `/v1/admin/clip/backfill` + stats.clip. CPU ONNX int8. Live: images_embedded 3.
13. **Face-ID pipeline** (`app/face.py`): OpenCV YuNet+ArcFace 512-dim works; `/v1/search/face` (60% match on Elon portrait); `/v1/admin/face/backfill` (thumb-reuse, limaued — raw bytes never persisted for web entries); honest `provider` (CUDA only when `cv2.cuda` has devices).
14. **StrikeDB namespace layer**: `vadd_ns`/`vsearch_ns` (VADDNS/VSEARCHNS) in storage + StrikeDB `resp.py`; rebuilt after `git pull` (StrikeDB @3b6cbd4). **Reality: HNSW graph on :6008 is 64-dim (phash) → VSEARCHNS with a 512-dim vector returns `[]`. Namespaces only partition; one graph = one dimension.**
15. **Enhancement models downloaded/verified**: `data/models/enhance/realesrgan_x4.onnx` (fp16 33.7MB), `birefnet_lite.onnx` (114.5MB); CLIP `data/models/clip/vision_model_int8.onnx` (88.6MB). No `app/enhance.py` yet.
16. **Admin UI backfill button**: `app/static/admin.html`+`admin.js` — Face-ID index panel (provider chip, coverage, Backfill button). `admin.js?v=4`.
17. **CPU-only enforced** (this session): `onnxruntime-gpu`→`onnxruntime`; `face.py` `DNN_TARGET_CPU` always (CUDA branch removed); `clip.py`/`enhance.py` already CPU. venv reports `[AzureExecutionProvider, CPUExecutionProvider]` only. NEVER re-add CUDA (crashed user's driver once).
18. **`.env` model dirs fixed**: `FACE_ID_MODELS_DIR`/`CLIP_MODELS_DIR` pointed at root-owned `models2/imify/data` → now `models1/imify/data/models/{face,clip}` (was the PermissionError root cause). `FACE_ID_DEVICE=cpu`.
19. **`_local_ai_grid` merged into `/v1/search`** (`app/api.py` ~113): face/CLIP/OCR local matches now feed the MAIN grid ("ai" provider), not just `/v1/search/face|clip`. 
20. **Face false-positive fix (this session, root cause MEASURED)**: on ACTUAL data, different-people ArcFace similarity median ≈ **0.49**, but the grid gate was `0.40 × 0.8 = 0.32` → admitted ~half the index as "matches" (your "suite photo matches wrong people"). FIX: `face.py` new `match_identity()` with **mutual nearest-neighbour + margin** (`face_id_same_threshold=0.50`, `face_id_match_margin=0.05`); both call sites (`/v1/search/face` + `_local_ai_grid`) use it. Confidence tiers now map the real 0.5-0.9 same-person band to Certain/Confident/Uncertain/Weak (not linear 0-100).
21. **Face detection miss fix (holiday photo)**: `face_id_min_face_size` 96→**48**, new `detect_faces_multiscale()` (scales 1.0/0.7/1.4 + center-dedup) so small/distant holiday faces get caught.
22. **CDN-drop filter (`app/matcher.py`)**: web candidates whose `hit.url` is only a bare image/CDN file (preview.redd.it, i.imgur.com, content.timeweb.com — no real article page) are dropped BEFORE verify budget; fixes "Open post/article just opens the image".
23. **AI-grid kwarg bug FIXED**: `_local_ai_grid` `_wrap()` passed `ai_modality="clip"/"ocr"` but `SearchResult` field is `ai_modalities` (list) → TypeError swallowed by `except:pass` → the "ai" grid silently NEVER returned anything. Now `ai_modalities=["clip"]/["ocr"]/["face"]` (api.py 191/212/171).
24. **"36 images match ANY prompt" text-search bug FIXED**: `/v1/search/text` had NO similarity floor + re-embedded every caption per query. Now: `ai_text_min_sim=0.50` config floor (nomic-embed baseline makes 0.35 pass everything — 0.50 keeps only close captions), and caption embeddings are PERSISTED at index time (`ImageMeta.caption_embedding`, indexer) so search is a cosine over stored vectors, never 37 remote embed calls. `/v1/search/fulltext` matched caption boilerplate ("image/with/a/the/shows/depicts" in ALL 36) → now stopword-filtered + `min_hits = max(1, ceil(len(tokens)/2))`. VERIFIED: "man"→10, "kitten"→1, "image"/"the"→400.
25. **No-thumb / can't-open text results FIXED**: `/v1/search/text|ocr|fulltext` returned raw `meta.to_dict()` (no `thumb`, url=`file:///...models2/...` stale decoy path) → UI img fell back to dead file://. Now shaped with `thumb=/img/{id}.{ext}`, `page_url`, `category`, and `confidence=%` (was 0% everywhere). VERIFIED: text "suit microphone" → 88% top, kitten 69%, all with thumbs.
26. **Vision model upgraded 11B→90B**: `.env` + config `AI_CAPTION_MODEL=meta/llama-3.2-90b-vision-instruct` (NVIDIA NIM) — 11B hallucinated colors/identity (called blue-red logo wrong, mixed "hacker profile" caption with Elon). Existing captions stay old (id=1 caption says "red circle on blue" which is actually roughly right: blue bg 1872px + red circle 635px). Re-captioning old entries = future task.
27. **Remote-link-replaced-by-local-link bug FIXED (THE BIG ONE)**: ALL 36 entries had `url='file:///run/media/irfan/models2/imify/data/uploads/...'` (root-owned DECOY path, non-existent for user) because the user-upload path enqueued the LOOPBACK url (`http://127.0.0.1:8010/uploads/<name>`) as the canonical index entry; the real remote URL only landed in `source_urls`. After one verified search, every later search returned the dead local link. FIXED at 3 layers: (a) `indexer.py _is_remote_url()` + `_merge_duplicate` promotes remote URL→canonical over file:///127.0.0.1; (b) `repo.py add_source_link` promotes remote url/page_url over local; (c) `api.py _maybe_track_upload` now enqueues the VERIFIED REMOTE url (not loopback) when the web layer confirmed the upload on a real site. VERIFIED: id=37 canonical = `https://www.canyon-news.com/...Yos.jpg`.
28. **UnboundLocalError `query_clip` FIXED**: `query_clip` was annotated inside the `query_bytes` branch but referenced on the `image_id` path → 500 "cannot access local variable". Moved init to function top (`query_clip: list[float] | None = None`). VERIFIED: image_id+web search returns 200.

---

## 8. OPEN ISSUES / CURRENT WORK (THE TWO REAL PROBLEMS)

**Problem A — AI features run SEPARATE from reverse search (user's core complaint).**
Face-ID and CLIP exist as their own endpoints (`/v1/search/face`, `/v1/search/clip`) but do NOT feed
the main `/v1/search` grid. Symptom the user reported LIVE: uploaded a **blurry influencer photo** →
engine saw "similar people" separately but the main search said it **"matches with sea", only 2 results**.
More chaos (all confirmed on LIVE homelab pics): a **homelab photo** was returned as **"near" matches with
a black dog and a swimming pool**; another **homelab photo returned a single result today — a "girl with a moon"**
presented as unverified; unrelated/unverified candidates keep leaking into the grid as if they matter.
So /v1/search only sees pHash/dHash/aHash closeness → semantically unrelated junk becomes "near".

**The REAL hash-close false-positive fixed by a CLIP semantic gate** (repro + confirmed bug):
`curl POST /v1/search` with Rajshahi-University-logo upload → total 2, providers [local, web], **2 wrong
hash-only "near" matches**: Noakhali Science & Technology logo (sim 0.674) + Bijay Ekattar Hall logo
(sim 0.660). Two different university logos judged "near" purely on hashes — no content check.
`app/matcher.py:464` fused>fused_near path does NOT consult CLIP yet.

**FIX (in progress)**: wire CLIP gate into verification ("Let me wire the CLIP semantic gate"). Add a
config floor (e.g. `clip_verify_min_sim` ≈0.55); in `verify_web_hits`, when `query_clip` is available and
a candidate has a CLIP embedding, a candidate whose CLIP sim is below the floor is downgraded to
`_unverified(note="semantic mismatch")` — kills Rajshahi/Noakhali/Bijay and homelab→dog/pool nonsense.

**THE B key - unrelated unverified candidates leak into the grid as "matches".** CDN "person with iPhone"
result surfaced as a match; two different student logos hash-close → "near". Root cause: verification is
hash-only. Same CLIP gate + a `clip_verify` pass (#2) addresses it.

**THE namespace dimension block for AI ANN (#3)**: VSEARCHNS(512-dim) → `[]` because :6008 graph is 64-dim.
Options: brute-force metadata match (already works for faces, no ANN needed) OR start a 2nd dbstrike (:6009)
whose WAL was seeded to hold 512-dim vectors. User prefers brute-force-style fix "regardless of namespace".

**Enhancement (#4)** — user asked: also add "blurry image clear" via
**BiRefNet** (ZhengPeng7/BiRefNet) + **Real-ESRGAN** (xinntao/Real-ESRGAN). Models downloaded to
`data/models/enhance/` (realesrgan_x4, birefnet_lite) and verified to load+run in onnxruntime, but no
`app/enhance.py`. Plan: blur-detection (Laplacian var) → Real-ESRGAN x4 upscale → BiRefNet foreground
refine → feed CLEARED image to feature extraction + web upload.

## 9. NEXT STEPS (suggested order)

1. **Wire CLIP semantic gate into `verify_web_hits`** (`app/matcher.py:464`, `app/config.py` floor, pass
   `query_clip` from `app/api.py` search handler) → kills Raj/shahi/Noakhali/BStorm + sea/dog/pool false
   "near". Tests + live repro re-run.
2. **Feed face+CLIP local matches INTO `/v1/search` grid** (not just separate endpoints) so a blurred
   influencer is found by the AI layers even before content verification (#A). Brute-force, no ANN.
3. **`app/enhance.py`** — blur detect → Real-ESRGAN → BiRefNet refine → use enhanced image for extraction + web.
4. Live-verify all together on the user's three repros; update PROBLEMS.md + HANDOFF.md; re-run pytest (127).

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
- **Live Project root = `models1/imify`** (the ONLY real project; user is non-root). `models2/imify`
  is a root-owned dir (deploy/reference only) — treat it as NOT the project; don't edit/use it.
  Old notes saying models2 is live are the exact bug that crashed the model dirs.
- **NEVER run anything that initializes a CUDA/GPU session** — `onnxruntime-gpu`, `dnn.DNN_TARGET_CUDA`,
  or model-loading pytest on a fresh driver. It once crashed the user's nvidia driver requiring a
  reinstall. Stack is CPU-only by policy (plain `onnxruntime`).
- **NEVER `pkill`** the app or dbstrike — kill only by exact PID found via `pgrep -f`.
- **dbstrike WAL**: real data is `data/dbstrike.wal` (2.0MB); a bare `dbstrike.wal`@root is stale 2KB.
  Launch with `DBSTRIKE_WAL=data/dbstrike.wal`. Backup exists at `data/dbstrike.wal.bak`.
  Binary needs `libnvrtc.so.12` via LD_LIBRARY_PATH to the ASC venv (see §6).
- **Models dir**: `/run/media/irfan/models1/imify/data/models/{clip,face,enhance}` (CLIP, Face-ID, Real-ESRGAN, BiRefNet) — all already downloaded; all CPU.
- **AI service split**: nomic text embeddings live on the OTHER server (192.168.0.107:8374, llama.cpp);
  vision/captioning is cloud NVIDIA NIM; OCR = local tesseract; CLIP+face+enhance = local CPU ONNX.
- **Enhancement requested**: clear blurry uploads via BiRefNet (ZhengPeng7) + Real-ESRGAN (xinntao) — implemented in `app/enhance.py`, wired into /v1/search.
