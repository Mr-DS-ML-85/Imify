<p align="center">
  <img src="docs/logo.svg" alt="Imify — self-hosted reverse image search" width="640"/>
</p>

<p align="center">
  <b>Imify — self-hosted reverse image search.</b><br/>
  Your own image index, built by your own pipeline. SearXNG for discovery, StrikeDB for storage + vector search. No third-party APIs, no per-query fees, no data leaving your infrastructure.
</p>

---

Imify is a TinEye-style reverse image search engine built on the same philosophy as
your [SearchOS](https://github.com/Mr-DS-ML-85/SearchOS): **own everything**.

| Layer | TinEye / SaaS | Imify |
|---|---|---|
| Index | vendor-owned | **your own image index** |
| Discovery | vendor crawlers | **SearXNG (self-hosted, free)** |
| Storage | vendor DB | **StrikeDB (your own DB, KV + HNSW vectors)** |
| Matching | proprietary | **perceptual hashing → Hamming ANN via VSEARCH** |
| Cost | per-search | **free, your hardware** |
| Privacy | vendor sees queries | **your infra only** |

## How it works

```
                    ┌─────────────────────────────── Imify Gateway (FastAPI) ───────────────────────────────┐
                    │  POST /v1/search  →  own-index match  +  SearXNG live reverse  →  fusion  →  ranked   │
                    │  /v1/admin/* · /v1/stats · /health · /metrics                                         │
                    └──────────────┬──────────────────────────────────────┬─────────────────────────────────┘
                                   │                                      │
                    ┌──────────────▼─────────────┐            ┌───────────▼─────────────┐
                    │       Index pipeline       │            │       SearXNG           │
                    │ discovery → fetch → match  │───learns──▶│ images category +        │
                    │ (perceptual dedup + hash)  │            │ image_url reverse search │
                    └──────────────┬─────────────┘            └─────────────────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │          StrikeDB          │
                    │  KV: image metadata/URLs   │
                    │  VSEARCH: Hamming ANN      │
                    └────────────────────────────┘
```

**Matching core.** Every indexed image is fingerprinted with a **fused
multi-hash**: 64-bit pHash (DCT) + dHash (gradient) + aHash (average). The
pHash is encoded as 64 floats of ±1, so StrikeDB's `VSEARCH` (cosine ANN)
returns **Hamming-ordered near-duplicates** in sub-millisecond time at
million-image scale (`cosine_dist == hamming/32`); the two extra hashes add
precision so crops, watermarks and re-encodes of the same image stay close
while unrelated images separate. A color-histogram mode matches by dominant
colors, and live web reverse results are fused in and *saved into your own
index* so repeat searches are served from memory.

**The hash is the judge (TinEye model).** Google and other engines limit face
results (privacy) and their similarity verdicts are opaque — so Imify never
trusts an engine's verdict. Engines are **candidate finders**; every candidate
is downloaded and fused-hash-verified against the query image before it is
reported as a match. A derived copy (resize, re-encode, watermark) stays in the
near band and is found; an unrelated image is rejected (`found=false` →
"search not found"), never a false positive. Unverifiable candidates are still
listed with a dashed `unverified` badge — but they never count as found.
Facebook candidates are resolved through a three-hop chain — lookaside stub →
photo.php with your real session cookies → proxify gui_chrome render to reveal
the signed scontent image URL — before being hashed and judged.

**"Found on N sites" consolidation.** Same photo returned by several engines
and sites (re-encoded/downloaded copies) is consolidated into **one card**
whose `source_urls` / `source_count` carry every site — TinEye's "found on N
sites" removal list — instead of flooding the grid with near-identical tiles.

**Optional self-hosted AI + OCR.** Point `AI_OLLAMA_URL` at a local Ollama and
Imify auto-captions indexed images (vision model) and answers natural-language
searches (`/v1/search/text`) by embedding similarity. Enable `OCR_ENABLED` and
imaged visible text is extracted (tesseract) and searchable (`/v1/search/ocr`)
— perfect for logos, memes and screenshots. Both are opt-in and degrade
cleanly: with them off, the pure perceptual pipeline runs exactly as before.

**SearchOS ideas ported:** search state as a durable asset (every query, pending
URL and image lives in KV and resumes across restarts) · discovery → fetch →
process worker pipeline · coverage map per seed query with re-crawl freshness ·
middleware safeguards (circuit breaker for a throttled SearXNG, retry with
backoff, stall detection, fetch guards) · gateway/worker split.

## Quick start (Docker)

```bash
cp .env.example .env
# edit .env — generate a real admin key:
python -c "import secrets; print('sk_' + secrets.token_urlsafe(32))"
docker compose up -d --build
```

- UI: `http://localhost:8000`
- Admin: `http://localhost:8000/admin` (paste your `BOOTSTRAP_ADMIN_KEY`)
- SearXNG: `http://localhost:8080`
- StrikeDB: `127.0.0.1:6008` (RESP — talk to it with `redis-cli`)

Add seed queries from the Admin panel (e.g. *aurora borealis wallpaper*) and the
pipeline starts filling your index. Verify: `docker compose exec app sh -c
"redis-cli ... "` — or just watch `make logs`.

> **Note on `strikedb`:** the compose service builds from your StrikeDB checkout
> (`reference/StrikeDB`). If that directory is absent, `docker build -f
> docker/Dockerfile.strikedb reference/StrikeDB` first, or publish your own image.

## Bare metal (uv)

```bash
uv sync                       # install deps (Python 3.11+)
uv run pytest                 # run the test suite

# 1. StrikeDB (your DB)
cd reference/StrikeDB && cargo build --release --bin dbstrike
ulimit -n 100000
DBSTRIKE_WAL=./data/dbstrike.wal ./reference/StrikeDB/target/release/dbstrike 127.0.0.1:6008 &

# 2. SearXNG (free, self-hosted)
docker run -d -p 8080:8080 -v "$PWD/deploy/searxng/settings.yml:/etc/searxng/settings.yml:ro" \
  -e SEARXNG_BASE_URL=http://localhost:8080/ searxng/searxng:latest

# 3. Imify (dev, SQLite fallback — no external services needed)
STORAGE_BACKEND=sqlite uv run uvicorn app.main:app --port 8000
```

## Running StrikeDB + Proxify (the live stack)

Imify needs two services of your own next to it. Both have idempotent start
scripts (safe to run repeatedly; they do nothing if the service is up).

### StrikeDB — storage + vector search (`127.0.0.1:6008`)

```bash
scripts/start_strikedb.sh
redis-cli -p 6008 PING          # -> PONG
redis-cli -p 6008 DBSIZE        # number of keys (images, trackers, caches…)
```

- Binary: `reference/StrikeDB/target/release/dbstrike` (build with
  `cd reference/StrikeDB && cargo build --release --bin dbstrike`).
- Data: `data/dbstrike.wal` (keep `data/dbstrike.wal.bak` as a backup).
- **CPU-only:** the script sets `DBSTRIKE_GPU=cpu`, so the GPU is never
  touched. The binary still *links* `libnvrtc.so.12`; the script puts a
  symlink to one copy of that file in `data/lib/` (only that file, nothing
  else from a CUDA toolkit). If it is missing, point the symlink at any
  `libnvrtc.so.12` on the machine.
- Talk to it like Redis: `redis-cli -p 6008 KEYS 'img:meta:*'`.

### Proxify — anti-bot fetch gateway (`127.0.0.1:8081` API, `:8888` tunnel)

```bash
scripts/start_proxify.sh              # start (or no-op if running)
scripts/start_proxify.sh --recreate   # rebuild the container from the image
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:8081/cookies/status
```

- Runs the existing `proxy-orchestrator-orchestrator:latest` image (build it
  from your proxify checkout with `docker compose build` in
  `proxy-orchestrator/`). Joins the `proxy-orchestrator_default` network
  with `proxy-redis` and `proxy-flaresolverr`.
- Bound to `127.0.0.1` only, so the tunnel is never an open proxy on your LAN.
- **Browser profile** lives in the docker volume `proxify-gui-profile`, so
  cookies, cache and trust survive `--recreate`. Stale Chromium lock files
  from an old container are cleared automatically.
- **Cookies:** the script builds `data/proxify-cookies.txt` from
  `gui-cookies.txt` with `scripts/filter_cookies.py`, keeping ONLY
  search-engine and Facebook domains. Never mount the full export: it
  contains every logged-in site in your browser. Refresh cookies by
  re-exporting `gui-cookies.txt`, then `scripts/start_proxify.sh --recreate`.
  `/v1/stats` → `cookies` shows live/expired counts per domain.
- **Honest mode (`GUI_HONEST_MODE=true`, default in the script):** the GUI
  browser presents itself exactly as it is. No injected fingerprint script,
  no user-agent / client-hint override, no timezone emulation; the container
  runs in the host's timezone. JavaScript fingerprint shims are detectable
  (e.g. deleting `navigator.webdriver` makes it `undefined` instead of
  `false`), and claiming Windows from a Linux browser is a contradiction
  anti-bot systems score as automation. Real values are consistent.
- **Branded Google Chrome:** Playwright's bundled Chromium is "Chrome for
  Testing" (no "Google Chrome" brand in `navigator.userAgentData`, a known
  automation marker). Build the thin image once and the script uses it
  automatically:

  ```bash
  docker build -f docker/Dockerfile.proxify-chrome -t proxify-chrome:latest docker/
  scripts/start_proxify.sh --recreate
  ```
- **Images, PDFs, files, raw URLs:** `/fetch` returns binary responses as
  bytes. Images, PDFs, archives and anything non-text come back in
  `body_b64` with `content_type` and `size` (PDF text is also extracted into
  `markdown`). Add `"raw": true` to get the bytes of a text page too.
  Imify uses this as its fallback downloader (`app/proxify.py`) when a
  candidate image or an `image_url` is refused directly (403 hotlink
  protection, 429, anti-bot fronts).

  ```bash
  curl -s -X POST http://127.0.0.1:8081/fetch -H 'Content-Type: application/json' \
    -d '{"url":"https://www.python.org/static/img/python-logo.png"}' \
    | python3 -c "import json,sys,base64;d=json.load(sys.stdin);open('logo.png','wb').write(base64.b64decode(d['body_b64']));print(d['content_type'],d['size'])"
  ```
- **In-browser uploads (Google Lens):** a `POST` with a multipart file in
  `body_b64`, `"force_strategy": "gui_chrome"` and `"in_page_upload": true`
  is uploaded by the GUI Chrome itself: it opens `upload_landing`, submits a
  real form with the file, waits for a URL containing `result_url_contains`
  (default `/search`) and returns that page's raw HTML (scripts kept). Lens
  ties its results to the browser session that uploaded, so this is the one
  route that returns visual matches; Imify's Google engine uses it first.
- **Honest results:** `success` is `false` for HTTP errors (404/410 are
  returned as final answers instead of being escalated through every
  strategy), and Google image / Lens result pages count as real results.
- **Self-check:** fetch `https://bot-detector.rebrowser.net/` through
  proxify's `gui_chrome` and read the verdicts; everything should be green.
- **Native network stack:** `GUI_TLS_MITM=false` (default in the script)
  lets Chrome connect to sites itself (its own TLS, HTTP/2, QUIC). Set
  `GUI_TLS_MITM=true` to route it through proxify's TLS-impersonating MITM.

### Engine politeness (why Imify won't get your IP banned)

`app/engine_guard.py` paces every engine call (Google ≥ 12 s apart, Yandex
4 s, Bing 3 s, with jitter), recognizes CAPTCHA / "unusual traffic" / 403 /
429 answers, and **rests** a blocked engine (10 min, doubling up to 6 h)
instead of retrying through it. Other engines carry the search meanwhile.
Check the state in `/v1/stats` → `engine_guard`; a resting engine shows up
in a search's `degraded` list as `cooldown:<engine>`.

If Google answers *"Our systems have detected unusual traffic from your
computer network"* (`/sorry`), that flag is on your **IP address**, not the
browser. Nothing client-side clears it; only quiet time does (hours to a
day). Let the guard keep Google resting, and check with your normal browser:
if it shows the same page, it is purely the IP.

### The service

```bash
systemctl --user restart imify.service    # starts StrikeDB + proxify first
systemctl --user status imify.service
tail -f data/app-8010.log
curl -s http://127.0.0.1:8010/health/ready
```

## API

Everything the web UI does goes through this API — the UI is a static page
that calls these endpoints and nothing else, so any search or admin action
can be scripted. Paste and drag-drop are the same `file` upload; the result
sections (Copies / Same person / Visually similar / Unconfirmed) are built
from each result's `match_type`. A few endpoints have no UI button
(`/v1/search/face`, `/v1/search/clip`, `mode=exact`,
`/v1/admin/clip/backfill`, `/v1/image/{id}`, `/metrics`).

**Auth:** send a key as `Authorization: Bearer sk_...` or `X-API-Key: sk_...`.
`AUTH_MODE=open` (default) needs a key only for `/v1/admin/*` (role `admin`);
`AUTH_MODE=key` needs one for every `/v1/*` call.

### Search (multipart form fields)

| Endpoint | Fields | What it does | UI |
|---|---|---|---|
| `POST /v1/search` | `file` **or** `image_url` **or** `image_id`; `mode` (`best`·`color`·`biggest`·`newest`·`changed`·`exact`, default `best`); `num_results` (default 40); `include_web` (default `true`); `learn` (default `false` — `true` adds verified web matches to the index) | Reverse image search: local index + Yandex/Bing/Google Lens byte uploads + text discovery, every candidate verified (hash, SIFT geometry, face, CLIP) | upload / paste / URL / "search again" / mode tabs (all but `exact`, which is API-only) / *Learn matches* toggle |
| `POST /v1/search/text` | `query`, `num_results` | Natural-language search over AI captions (needs Ollama; 503 when off) | text search → *AI description* |
| `POST /v1/search/ocr` | `query`, `num_results` | Keyword search over OCR text in indexed images | text search → *OCR text* |
| `POST /v1/search/fulltext` | `query`, `num_results` | Keyword search over every text field (title, URL, caption, OCR) | text search → *Full text* |
| `POST /v1/search/face` | `file`, `num_results` | Same person across the indexed library (ArcFace) | API only (face matches are also part of `/v1/search`) |
| `POST /v1/search/clip` | `file`, `num_results` | Semantically similar scenes (CLIP) | API only (also part of `/v1/search`) |

The response (`SearchResponse`) carries `found`, `total`, `took_ms`,
`providers`, `degraded` (engines that failed or were skipped, e.g.
`provider:google-upload`, `cooldown:google`), `cached`, `best_guess`,
`discovery`, `by_engine`, `by_category` and `results[]` — each result has
`match_type` (`exact`·`near`·`modified`·`face`·`similar`·`unverified`),
`score`, `engine`, `title`, `page_url`, image URLs and evidence.

### Read-only

| Endpoint | Description |
|---|---|
| `GET /v1/stats` | Index statistics, success rates, engine guard state (cooldowns, daily budgets) — the admin *Success rates* panel |
| `GET /v1/history?limit=20` | Recent searches |
| `GET /v1/image/{id}` | One indexed image's metadata |
| `GET /img/{name}` | Stored thumbnails / uploads |
| `GET /health/live`, `/health/ready` | Probes |
| `GET /metrics` | Prometheus text format |

### Admin (`admin` key)

| Endpoint | Description | UI (`/admin`) |
|---|---|---|
| `POST /v1/admin/ingest` | `{"queries": ["q1", ...], "urls": ["https://..."]}` — seed queries and/or specific image URLs | Seed queries |
| `GET /v1/admin/queries` · `PATCH /{id}` `{"enabled": bool}` · `DELETE /{id}` | Seed query management | Seed queries |
| `GET /v1/admin/crawl` | Pipeline status | Pipeline status |
| `GET /v1/admin/images?q=&limit=&offset=` | List indexed images (filter by URL/title/engine) | Indexed images |
| `DELETE /v1/admin/images/{id}` | Remove one image | Indexed images |
| `POST /v1/admin/images/purge` | Bulk remove: `{"url_contains": "example.com", "query": "..."}` | Indexed images |
| `GET /v1/admin/tracks` · `PATCH /{id}` `{"enabled": bool}` · `DELETE /{id}` | Background trackers (auto-update) | Background trackers |
| `POST /v1/admin/tracks/run` | Run one tracker round now | Background trackers |
| `POST /v1/admin/face/backfill?limit=50&force=false` | (Re-)embed faces | Face-ID index |
| `POST /v1/admin/clip/backfill?limit=50` | Embed CLIP vectors | API only |
| `POST /v1/admin/keys` `{"name", "role"}` · `GET` · `DELETE /{name}` | API keys (SHA-256 hashed at rest; the key is shown once) | API keys |

```bash
# reverse search a file, a URL, or an indexed image
curl -F "file=@photo.jpg" -F mode=best -F num_results=60 http://localhost:8000/v1/search
curl -F "image_url=https://example.com/a.jpg" http://localhost:8000/v1/search
curl -F "image_id=42" -F mode=newest -F include_web=false http://localhost:8000/v1/search
# keep verified web matches in the index (the UI's Learn toggle)
curl -F "file=@photo.jpg" -F learn=true http://localhost:8000/v1/search
# text searches
curl -F "query=red bicycle" http://localhost:8000/v1/search/fulltext
# admin
curl -H "Authorization: Bearer sk_..." -H 'Content-Type: application/json' \
  -d '{"queries": ["eiffel tower"]}' http://localhost:8000/v1/admin/ingest
```

The systemd service on this box listens on `127.0.0.1:8010`; Docker uses `:8000`.

## Configuration

All env-driven (see `.env.example`). Key knobs:

| Variable | Default | Purpose |
|---|---|---|
| `STORAGE_BACKEND` | `strikedb` | `strikedb` or `sqlite` (dev/test fallback) |
| `STRIKEDB_HOST/PORT/PASS` | `127.0.0.1:6008` | your DB |
| `SEARXNG_URL` | `http://127.0.0.1:8080` | discovery + live reverse |
| `SEARXNG_QUERY_INTERVAL` | `5` | seconds between SearXNG queries (politeness) |
| `REVERSE_PROVIDERS` | `searxng,bing,yandex,google` | live reverse chain order (SearXNG → Bing byte-upload → Yandex → Google via proxify) |
| `PROXIFY_FETCH_URL` | – | proxify `/fetch` endpoint — required for Google, and for rendering Facebook comet photo pages (gui_chrome) during candidate verification |
| `FETCH_PROXY_URL` | – | proxify CONNECT tunnel (:8888) — enables byte-upload reverse search and cookie-routed candidate fetches |
| `PROXIFY_USE_BROWSER` | `true` | force proxify's real-browser (gui_chrome) path for Google |
| `GOOGLE_UPLOAD_ENABLED` | `true` | circle-to-search: upload raw bytes to Google `/searchbyimage/upload` via proxify |
| `COOKIE_FILE` | `gui-cookies.txt` | Netscape cookie file (proxify's `scripts/brave_cookies.py` export). Verification attaches the host's cookies so Facebook CDNs (lookaside.fbsbx.com / scontent fbcdn) resolve instead of 403 |
| `PUBLIC_BASE_URL` | – | set to enable live web reverse + tracking for uploaded images |
| `AUTO_TRACK_SEARCHES` | `true` | new images get indexed AND background-tracked (auto-update) |
| `TRACKER_INTERVAL_SECONDS` | `3600` | how often tracked images are reverse-searched in the background |
| `INDEXER_FETCH_WORKERS` | `4` | concurrent fetchers |
| `CRAWLER_INTERVAL_SECONDS` | `3600` | seed re-run / freshness |
| `CRAWLER_RECRAWL_AFTER_SECONDS` | `604800` | re-fetch stale images (7 days) |
| `HAMMING_EXACT/NEAR_THRESHOLD` | `4` / `16` | single-hash match bands |
| `FUSED_EXACT_THRESHOLD` / `FUSED_NEAR_THRESHOLD` | `0.14` / `0.35` | fused pHash+dHash+aHash match bands (the verdict) |
| `SEARCH_DEDUP_THRESHOLD` | `8` | Hamming bits for consolidating same-photo results into one "found on N sites" card (0 disables) |
| `WEB_VERIFY_CANDIDATES` | `true` | download + hash-verify web candidates before reporting |
| `WEB_VERIFY_CONCURRENCY` / `WEB_VERIFY_MAX_CANDIDATES` | `10` / `80` | candidate fetch limits |
| `WEB_UNVERIFIED_MAX_FUSED` / `WEB_ENGINE_BING_PENALTY` | `0.36` / `1.35` | junk-preview filter + Bing de-rank |
| `AI_OLLAMA_URL` | – | local Ollama endpoint; enables auto-captioning + `/v1/search/text` |
| `AI_CAPTION_MODEL` / `AI_EMBED_MODEL` | `llama3.2-vision` / `nomic-embed-text` | Ollama models |
| `AI_CAPTION_ON_INDEX` | `false` | caption every indexed image (alt-text + text search) |
| `OCR_ENABLED` / `OCR_LANG` | `false` / `eng` | extract visible text from indexed images (`/v1/search/ocr`) |
| `TRACKER_VERIFY_CANDIDATES` | `true` | tracker candidates are hash-verified before ingestion |
| `EXACT_SCAN_MAX_IMAGES` | `200000` | brute-force scan below this, ANN above |
| `AUTH_MODE` | `open` | `open` or `key` |
| `BOOTSTRAP_ADMIN_KEY` | – | admin key seeded at startup |
| `RATE_LIMIT_*` | 60/600/5000 | per-minute/hour/day |

## Background trackers (auto-update)

Search an image that is **not** in your index and Imify does three things:

1. saves the upload under `data/uploads/` and indexes it locally,
2. registers a **tracker** for it (admin → *Background trackers*), and
3. a background loop **reverse-searches the whole internet for it** on a
   schedule (`TRACKER_INTERVAL_SECONDS`) — SearXNG first, then Google via your
   proxify gateway — and **ingests every new match** into your index.

The same image therefore finds *more* matches over time, TinEye-style, without
you doing anything. Trackers need a publicly reachable image URL: set
`PUBLIC_BASE_URL` for uploads, or upload by `image_url`. Without one they
degrade honestly (tracker note: *"no public URL"*). Tracker candidates are
**hash-verified against the tracked image before ingestion** — unrelated
images can never pollute the index (tracks store all three hashes; legacy
tracks fall back to pHash-only).

### Google reverse = Google Lens through Proxify's browser

Imify never talks to Google directly — the Google engine needs
[Proxify](https://github.com/Mr-DS-ML-85/proxify) (`PROXIFY_FETCH_URL`,
`:8081` here because SearXNG owns `:8080`) and degrades
(`provider:google-upload` in `degraded`) without it.

**Circle-to-Search for images with no online link:** Google retired
`/searchbyimage/upload`; the live byte-upload path is Google Lens, and Lens
ties its results to the browser session that uploaded — a plain HTTP upload
only gets a JavaScript shell back. So Imify asks Proxify's GUI Chrome to do
the upload itself (`in_page_upload`): it opens `lens.google.com`, submits a
real form with the image and returns the visual-matches page. Imify parses
the Lens result records (source page, title, full image, thumbnail) and
verifies each one like any other candidate. If that path returns nothing,
the impersonated CONNECT-tunnel upload is tried next, and a zero-match page
is logged with what came back (size, Lens records present, CAPTCHA).
`app/engine_guard.py` paces Google uploads (≥12 s apart, daily budget) and
backs off after a block.

## Operations

- **Backups** — the StrikeDB WAL + snapshots live in the `strikerdb-data` volume
  (`CHECKPOINT` periodically to compact). Thumbnails in `imify-data/thumbs`.
  Because metadata is the source of truth in KV, the vector ANN is always
  rebuildable — a restart cannot lose the index.
- **Scaling** — stateless gateway: add replicas behind a load balancer; run the
  pipeline in a separate process (`python -m app.worker`, `INDEXER_RUN_INLINE=false`).
- **Monitoring** — `/metrics` (Prometheus) + structured JSON logs with request IDs.
- **Politeness** — the crawler throttles SearXNG calls, respects size/timeout
  caps, validates content, retries with backoff and opens a circuit breaker on
  sustained failures.

## Tests

```bash
uv run pytest          # features, storage (incl. RESP client), matcher, API
```

## Project layout

```
app/
  main.py        app factory + middleware + metrics
  api.py         HTTP API (search ladder, admin, auto-track)
  features.py    perceptual hashing + color histograms (pure numpy)
  matcher.py     Hamming ANN search + exact scan + color + fusion/rerank
  indexer.py     discovery/fetch/process pipeline + tracker loop
  searx.py       SearXNG client (JSON API, throttled)
  ai.py          optional self-hosted AI (Ollama captions + embeddings)
  ocr.py         optional OCR text extraction (tesseract)
  engines/       live reverse providers (searxng, google-via-proxify) + chain
  repo.py        domain repository over the storage protocol
  storage/       protocol + StrikeDB (RESP) + SQLite backends
  worker.py      standalone pipeline process
  static/        UI (no build step: HTML/CSS/JS)
deploy/          SearXNG settings
docker/          Dockerfiles
scripts/         verification utilities
reference/       your SearchOS / StrikeDB / proxify checkouts (git-ignored)
tests/           pytest suite
```

## License

MIT. Your stack: SearXNG (AGPL), StrikeDB (Apache-2.0) — both self-hosted.
