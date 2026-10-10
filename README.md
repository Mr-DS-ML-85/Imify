<p align="center">
  <img src="docs/logo.svg" alt="Imify — self-hosted reverse image search" width="640"/>
</p>

<p align="center">
  <b>Imify — self-hosted reverse image search.</b><br/>
  Find where an image (or the person in it) appears online. Every result is checked by your own matcher before it is called a match. Your own index, your own engines, your own hardware.
</p>

---

Imify is a TinEye / Google Lens / FaceCheck-style reverse image search engine
built on the same philosophy as
[SearchOS](https://github.com/Mr-DS-ML-85/SearchOS): **own everything**.

| Layer | TinEye / SaaS | Imify |
|---|---|---|
| Index | vendor-owned | **your own image index** (StrikeDB KV + vectors) |
| Finding candidates | vendor crawlers | **Yandex, Bing and Google Lens byte uploads + SearXNG**, run in parallel |
| Judging matches | proprietary | **your own verifier**: perceptual hashes, SIFT geometry, ArcFace faces, CLIP content |
| Cost | per search | **free, your hardware** |
| Privacy | vendor sees queries | **your infrastructure** (only the engines you enable see the query) |

## How it works

```
 query (upload · image_url · image_id)
   │  image_url → resolve page to pixels (og:image → FB session cookies → browser render)
   │  optional: restore a low-quality query (Real-ESRGAN + BiRefNet)
   ▼
 features: pHash/dHash/aHash (+ 8 rotations/mirrors) · colour histogram · CLIP · faces
   │
   ├─► 1. OWN INDEX (instant) ── hash ANN / exact scan · SIFT crops · same face · CLIP · OCR
   │
   └─► 2. WEB PHASE (≤ 18 s budget, cached 24 h per image)
          ├─ byte uploads in parallel: Yandex · Yandex face crop · Bing · Google Lens (via Proxify)
          ├─ SearXNG reverse (image served back over loopback)
          ├─ text discovery: "best guess" from result titles → ~30 SearXNG image/icon engines
          ▼
        VERIFY every candidate (thumbnail triage → full download → verdict ladder)
          ▼
 3. consolidate copies into "found on N sites" cards → rank → response
          │
          └─ background tracker keeps re-searching the upload and learns new matches
```

**Engines find candidates; Imify judges them.** No engine's verdict is
trusted: Google limits face results, Bing returns "visually similar" junk,
and every engine's similarity score is opaque. Each candidate is downloaded
and put through the **verdict ladder** — the first rule that holds wins:

| # | Check | Verdict | Counts as found? |
|---|---|---|---|
| 1 | Fused pHash/dHash/aHash within the near band (also against the mirrored and rotated query), **and** CLIP agrees it is the same content | `exact` / `near` | yes — a copy |
| 2 | SIFT keypoints + RANSAC find the query inside the candidate (crop, border, caption bar, sticker, rotation) | `modified` | yes — a copy |
| 3 | ArcFace says the candidate shows the same person | `face` | yes — same person, different photo |
| 4 | CLIP says the content is close | `similar` | no — a lead (Lens "visually similar") |
| 5 | none of the above, or the download failed | `unverified` | no — shown only if plausibly close |

The thresholds were calibrated on real data on this box: same person ≥ 0.88
ArcFace similarity and different people ≤ 0.30 (threshold 0.42); real copies
give 58–330 RANSAC inliers and unrelated pairs ≤ ~30 (40 accepted alone, 18–39
also needs CLIP ≥ 0.80); copies score CLIP ≥ 0.90 and unrelated photos a
median of 0.50. `found=false` ("search not found") is an honest answer, never
padded with junk.

**Matching core (own index).** Every indexed image gets a fused multi-hash —
64-bit pHash (DCT) + dHash (gradient) + aHash (average), fused as
`0.5·pHash + 0.3·dHash + 0.2·aHash`. The pHash is stored as 64 floats of ±1,
so StrikeDB's cosine `VSEARCH` returns **Hamming-ordered** neighbours
(`cosine_dist == hamming/32`) in sub-millisecond time; small indexes are
scanned exactly. The query is also hashed in all 8 orientations, so a phone
photo turned 90° or a mirrored re-upload still matches. Colour mode matches by
dominant-colour histograms.

**"Found on N sites."** Copies of the same photo found on several sites are
merged into one card whose `source_urls` / `source_count` list every site
(hash-close, or CLIP ≥ 0.90 for re-encoded copies whose hashes drifted).

**FaceCheck-style people search.** When the query shows a person and the main
face is less than 20% of the frame, a tight crop of that face is also uploaded
to Yandex (`yandex-face`), and every candidate is face-verified. Matches of the
same person in a *different* photo appear in their own "Same person" section.

## Optional layers

All run locally on CPU and degrade cleanly: when a layer is off or its model
is missing, the rest of the pipeline runs unchanged.

| Layer | Switch | What it adds |
|---|---|---|
| Face-ID | `FACE_ID_ENABLED` | YuNet detection + ArcFace (MobileFaceNet) embeddings; `face` verdicts, `/v1/search/face` |
| CLIP | `CLIP_ENABLED` | ViT-B/32 int8 content embeddings; the content check in rule 1, `similar` leads, CLIP-gated SIFT, `/v1/search/clip` |
| Geometry | `GEOMETRIC_VERIFY_ENABLED` (on) | SIFT + RANSAC `modified` verdicts, also for crops inside the own index |
| OCR | `OCR_ENABLED` | Tesseract text from indexed images; `/v1/search/ocr`, `/v1/search/fulltext` |
| AI captions | `AI_BASE_URL` (+ `AI_CAPTION_URL`, `AI_API_KEY`) | Any OpenAI-compatible server (llama.cpp, NVIDIA NIM, Ollama): captions for indexed images and `/v1/search/text` by description |
| Enhancement | `ENHANCER_ENABLED` | Low-resolution queries only (≤ `ENHANCE_MAX_PIXELS`, 250 000 px): Real-ESRGAN ×4 (tiled) + BiRefNet subject mask before hashing. Larger images are never touched or downscaled. **Slow on CPU** — see [Known limitations](#known-limitations) |

Models download into `data/models/` on first use.

**Enhancement placement (`ENHANCE_DEVICE=auto`).** ESRGAN and BiRefNet are
placed one by one, bigger model first: on the GPU when its VRAM need fits in
the free VRAM, otherwise in system RAM. So a machine with lots of RAM and a
small GPU runs BiRefNet in RAM and ESRGAN on the GPU; a machine without a GPU
runs both in RAM; a big GPU runs both. If a GPU run fails (out of VRAM, driver
error), that model moves to RAM and the search continues. The GPU needs a
CUDA build of `onnxruntime` (`onnxruntime-gpu` + cuDNN); with the default
CPU-only build both models run in RAM. `/v1/stats` → `enhance.placement`
shows where each model runs. Face-ID and CLIP always run on the CPU.

## Quick start (Docker)

```bash
cp .env.example .env
# edit .env — generate a real admin key:
python -c "import secrets; print('sk_' + secrets.token_urlsafe(32))"
docker compose up -d --build
```

- UI: `http://localhost:8000` · Admin: `http://localhost:8000/admin` (paste your `BOOTSTRAP_ADMIN_KEY`)
- SearXNG: `http://localhost:8080` · StrikeDB: `127.0.0.1:6008` (RESP — use `redis-cli`)

Add seed queries from the admin page (e.g. *aurora borealis wallpaper*) and
the pipeline starts filling your index. Watch it with `make logs`.

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

[Proxify](https://github.com/Mr-DS-ML-85/proxify) is Imify's way out to the
web wherever a plain request is not enough: Google Lens uploads, Facebook
photo pages, hotlink-protected image hosts and blocked SearXNG engines.

```bash
scripts/start_proxify.sh              # start (or no-op if running)
scripts/start_proxify.sh --recreate   # rebuild the container from the image
curl -s http://127.0.0.1:8081/health
curl -s http://127.0.0.1:8081/cookies/status
```

- **Patched files are mounted, not baked in.** The script mounts these files
  from `reference/proxify/proxy-orchestrator/` read-only into the container:
  `strategies/gui_chrome.py`, `strategies/base.py`,
  `engine/decision_engine.py`, `engine/js_challenge_solver.py`,
  `gateway/rest_api.py` and `scripts/gui_browser.sh`. Edit them there, then
  `docker restart proxy-orchestrator`; a newly mounted file needs `--recreate`.
- Runs the `proxify-chrome:latest` image when present (branded Google Chrome,
  see below), else `proxy-orchestrator-orchestrator:latest`, on the
  `proxy-orchestrator_default` network with `proxy-redis` and
  `proxy-flaresolverr`.
- Bound to `127.0.0.1` only, so the tunnel is never an open proxy on your LAN.
- **Browser profile** lives in the docker volume `proxify-gui-profile`, so
  cookies, cache and trust survive `--recreate`. Stale Chromium lock files
  from an old container are cleared automatically.
- **Cookies:** the script builds `data/proxify-cookies.txt` from
  `gui-cookies.txt` with `scripts/filter_cookies.py`, keeping ONLY
  search-engine and Facebook domains. Never mount the full export: it
  contains every logged-in site in your browser — and never commit either
  file (both are git-ignored). Refresh by re-exporting `gui-cookies.txt`,
  then `scripts/start_proxify.sh --recreate`. `/v1/stats` → `cookies` shows
  live/expired counts per domain.
- **Honest mode (`GUI_HONEST_MODE=true`, default in the script):** the GUI
  browser presents itself exactly as it is — no injected fingerprint script,
  no user-agent / client-hint override, no timezone emulation. Fingerprint
  shims are detectable (deleting `navigator.webdriver` makes it `undefined`
  instead of `false`), and claiming Windows from a Linux browser is a
  contradiction anti-bot systems score as automation.
- **Branded Google Chrome:** Playwright's bundled Chromium is "Chrome for
  Testing" (no "Google Chrome" brand in `navigator.userAgentData`, a known
  automation marker). Build the thin image once and the script uses it:

  ```bash
  docker build -f docker/Dockerfile.proxify-chrome -t proxify-chrome:latest docker/
  scripts/start_proxify.sh --recreate
  ```
- **Native network stack:** `GUI_TLS_MITM=false` (default in the script)
  lets Chrome connect to sites itself (its own TLS, HTTP/2, QUIC). Set
  `GUI_TLS_MITM=true` to route it through proxify's TLS-impersonating MITM.
- **Self-check:** fetch `https://bot-detector.rebrowser.net/` through
  proxify's `gui_chrome` and read the verdicts; everything should be green.

What Imify uses from Proxify's `/fetch` API:

| Feature | Used for |
|---|---|
| Binary responses (`body_b64`, `content_type`, `size`); `raw: true` | Fallback downloader (`app/proxify.py`) when a candidate or an `image_url` is refused directly (403 hotlink protection, 429, anti-bot fronts) — up to `WEB_VERIFY_PROXIFY_MAX` per search |
| In-page uploads (`in_page_upload`, `upload_landing`, `result_url_contains`) | Google Lens: the GUI Chrome opens lens.google.com and submits a real upload form, because Lens ties results to the browser session that uploaded |
| `gui_chrome` renders | Facebook photo pages (signed image URL only exists in the rendered page), blocked SearXNG engines during text discovery |
| Honest `success` | `false` for HTTP errors and block pages; 404/410 are final answers, not escalated |

```bash
curl -s -X POST http://127.0.0.1:8081/fetch -H 'Content-Type: application/json' \
  -d '{"url":"https://www.python.org/static/img/python-logo.png"}' \
  | python3 -c "import json,sys,base64;d=json.load(sys.stdin);open('logo.png','wb').write(base64.b64decode(d['body_b64']));print(d['content_type'],d['size'])"
```

### Engine politeness (why Imify won't get your IP banned)

`app/engine_guard.py` paces every engine call (Google ≥ 12 s apart, Yandex
4 s, Bing 3 s, with ×0.7–1.6 jitter; Google gets a daily budget of 150),
recognizes CAPTCHA / "unusual traffic" / 403 / 429 answers, and **rests** a
blocked engine (10 min, doubling up to 6 h) instead of retrying through it.
Other engines carry the search meanwhile. State: `/v1/stats` →
`engine_guard`; a resting engine appears in a search's `degraded` list as
`cooldown:<engine>`. CAPTCHAs are never solved.

If Google shows *"Our systems have detected unusual traffic"* (`/sorry`),
check with your normal browser on the same network. If that browser is fine,
the automated browser gave itself away (on this box the cause was injected
fingerprint JS, the TLS MITM and the Chrome-for-Testing binary — fixed by
honest mode + branded Chrome). If your normal browser is blocked too, it is
the IP, and only quiet time clears it.

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
`AUTH_MODE=key` needs one for every `/v1/*` call. Searches are rate-limited
per caller (`429` + `Retry-After`).

### Search (multipart form fields)

| Endpoint | Fields | What it does | UI |
|---|---|---|---|
| `POST /v1/search` | `file` **or** `image_url` **or** `image_id`; `mode` (`best`·`color`·`biggest`·`newest`·`changed`·`exact`, default `best`); `num_results` (default 40, max 200); `include_web` (default `true`); `learn` (default `false` — `true` adds verified web matches to the index) | Full reverse search: own index + web phase + verdict ladder | upload / paste / URL / "search again" / mode tabs (all but `exact`, which is API-only) / *Learn matches* toggle |
| `POST /v1/search/text` | `query`, `num_results` | Search by description over AI captions (503 when the AI layer is off) | text search → *AI description* |
| `POST /v1/search/ocr` | `query`, `num_results` | Keyword search over OCR text in indexed images | text search → *OCR text* |
| `POST /v1/search/fulltext` | `query`, `num_results` | Keyword search over every text field (title, URL, caption, OCR) | text search → *Full text* |
| `POST /v1/search/face` | `file`, `num_results` | Same person across the own index (ArcFace) | API only (face matches are also part of `/v1/search`) |
| `POST /v1/search/clip` | `file`, `num_results` | Semantically similar images in the own index (CLIP) | API only (also part of `/v1/search`) |

The response carries `found`, `total`, `took_ms`, `providers`, `degraded`
(engines that failed or were skipped, e.g. `provider:google-upload`,
`cooldown:google`), `cached`, `fallback`, `best_guess`, `discovery` (text
discovery queries and engines), `by_engine`, `by_category`
(`duplicate`/`similar`/`candidate`) and `results[]`. Each result has
`match_type` (`exact`·`near`·`modified`·`face`·`similar`·`color`·`unverified`),
`score`, `confidence` (0–100), `engine`, `title`, `page_url`, image URLs,
`source_urls` / `source_count`, and the evidence behind the verdict.

### Read-only

| Endpoint | Description |
|---|---|
| `GET /v1/stats` | Index statistics, provider success rates, engine guard state, cookie status, layer status — the admin *Success rates* panel |
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
| `POST /v1/admin/keys` `{"name", "role": "user"\|"admin"}` · `GET` · `DELETE /{name}` | API keys (SHA-256 hashed at rest; the key is returned once) | API keys |

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

## Storage and caching

- **On-the-fly by default.** A search returns verified web matches without
  storing them (`learn=false`); the index grows only from seed discovery,
  explicit ingestion, trackers, or `learn=true`. A failed search never adds
  the query image to the index.
- **Bounded index.** Above `MAX_INDEX_IMAGES` (50 000) the least recently seen
  image is evicted.
- **Result cache.** Verified web results are cached per query-image SHA-256 for
  24 h (lead-only outcomes for 1 h), so repeating a search is instant.
- **Bandwidth guards.** Candidates are triaged on the engine's thumbnail
  (~10 KB) first; the full image is downloaded only when the thumbnail is
  promising. Downloaded candidates are cached on disk for 7 days
  (`data/uploads/web-preview`, at most 20 000 files).

## Configuration

All env-driven (`.env`, see `.env.example`). Main knobs, with code defaults:

| Variable | Default | Purpose |
|---|---|---|
| `STORAGE_BACKEND` | `strikedb` | `strikedb` or `sqlite` (dev/test fallback) |
| `STRIKEDB_HOST/PORT/PASS` | `127.0.0.1:6008` | your DB |
| `SEARXNG_URL` | `http://127.0.0.1:8080` | discovery, reverse search, text discovery |
| `SEARXNG_QUERY_INTERVAL` | `5` | seconds between crawler SearXNG queries |
| `PROXIFY_FETCH_URL` | – | proxify `/fetch` (here `http://127.0.0.1:8081`) — needed for Google Lens, FB renders, raw-byte fallback |
| `FETCH_PROXY_URL` | – | proxify CONNECT tunnel (`:8888`) for candidate downloads and Bing/Yandex uploads |
| `COOKIE_FILE` | `gui-cookies.txt` | Netscape cookie file; cookies are attached per host (Facebook CDNs) |
| `GOOGLE_UPLOAD_ENABLED` / `BING_UPLOAD_ENABLED` / `YANDEX_UPLOAD_ENABLED` | `true` | byte-upload engines |
| `FACE_CROP_UPLOAD` | `true` | also upload a face crop to Yandex when the face is < 20% of the frame |
| `WEB_PHASE_BUDGET_SECONDS` | `18` | hard cap on the whole web phase (engines + verification) |
| `WEB_RESULTS_CACHE_TTL_SECONDS` | `86400` | cache of verified results per image |
| `TEXT_DISCOVERY_ENABLED` / `TEXT_DISCOVERY_BUDGET_SECONDS` | `true` / `6` | Lens-style best-guess discovery |
| `TEXT_DISCOVERY_ENGINES` | all | comma-separated SearXNG engines (see `app/engines/text_images.py`) |
| `WEB_VERIFY_CONCURRENCY` / `WEB_VERIFY_MAX_CANDIDATES` | `4` / `40` | candidate verification limits (this box: 6 / 80) |
| `WEB_VERIFY_RENDER_MAX` / `WEB_VERIFY_PROXIFY_MAX` | `4` / `12` | browser renders / proxify raw-byte retries per search |
| `WEB_VERIFY_THUMBNAIL_TRIAGE` | `true` | judge thumbnails before downloading full images |
| `FUSED_EXACT_THRESHOLD` / `FUSED_NEAR_THRESHOLD` | `0.14` / `0.35` | fused hash bands (rule 1) |
| `ROTATION_VARIANT_MAX_FUSED` | `0.20` | stricter band for rotated/mirrored variants |
| `GEOMETRIC_STRONG_INLIERS` / `GEOMETRIC_MIN_INLIERS` | `40` / `18` | SIFT+RANSAC bands (rule 2) |
| `FACE_ID_ENABLED` / `FACE_ID_SAME_THRESHOLD` | `false` / `0.42` | Face-ID layer (rule 3) |
| `CLIP_ENABLED` / `CLIP_VERIFY_MIN_SIM` / `CLIP_SIMILAR_MIN_SIM` | `false` / `0.80` / `0.78` | CLIP layer (rules 1 and 4) |
| `SEARCH_DEDUP_THRESHOLD` / `SEARCH_DEDUP_CLIP_THRESHOLD` | `8` / `0.90` | "found on N sites" merging |
| `WEB_UNVERIFIED_MAX_FUSED` / `WEB_ENGINE_BING_PENALTY` | `0.36` / `1.35` | which unverified candidates may be shown at all |
| `ENHANCER_ENABLED` / `ENHANCE_MAX_PIXELS` | `false` / `250000` | Real-ESRGAN + BiRefNet restore of low-resolution queries |
| `ENHANCE_DEVICE` | `auto` | where ESRGAN and BiRefNet run: `auto` (GPU if it fits, else RAM, per model), `cpu`, `gpu` |
| `ENHANCE_ESRGAN_VRAM_MB` / `ENHANCE_BIREFNET_VRAM_MB` | `1500` / `4500` | VRAM each model needs for `auto` placement |
| `AI_BASE_URL` / `AI_CAPTION_URL` / `AI_API_KEY` | – | OpenAI-compatible AI server(s) for captions + text search |
| `AI_CAPTION_MODEL` / `AI_EMBED_MODEL` | `meta/llama-3.2-90b-vision-instruct` / `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` | AI models |
| `AI_CAPTION_ON_INDEX` | `false` | caption every indexed image |
| `OCR_ENABLED` / `OCR_LANG` | `false` / `eng` | OCR layer |
| `MAX_INDEX_IMAGES` | `50000` | LRU cap on the index |
| `AUTO_TRACK_SEARCHES` / `TRACKER_INTERVAL_SECONDS` | `true` / `3600` | background trackers for new uploads |
| `TRACKER_VERIFY_CANDIDATES` | `true` | verify tracker candidates before ingesting |
| `PUBLIC_BASE_URL` | – | public URL of this instance (lets URL-based engines see uploads) |
| `INDEXER_FETCH_WORKERS` | `4` | concurrent crawler fetchers |
| `CRAWLER_INTERVAL_SECONDS` / `CRAWLER_RECRAWL_AFTER_SECONDS` | `3600` / `604800` | seed re-run / re-fetch stale images |
| `EXACT_SCAN_MAX_IMAGES` | `200000` | exact scan below this, ANN above |
| `AUTH_MODE` / `BOOTSTRAP_ADMIN_KEY` | `open` / – | auth |
| `RATE_LIMIT_PER_MINUTE/HOUR/DAY` | `60` / `600` / `5000` | per-caller limits |

## Background trackers (auto-update)

Search an image that is **not** in your index and Imify:

1. saves the upload under `data/uploads/<sha256>.<ext>`,
2. registers a **tracker** for it (admin → *Background trackers*), and
3. re-searches it every `TRACKER_INTERVAL_SECONDS` and **ingests every new
   verified match** into the index.

With a public URL (`PUBLIC_BASE_URL`, or an `image_url` query) the tracker
uses the URL-based engines (SearXNG, Google, Bing, Yandex). Without one it
uploads the saved bytes instead — Bing, then Yandex, then Google Lens — so
private images are tracked too. Tracker candidates go through the same
verifier as live searches, so unrelated images never enter the index, and
the same engine guard paces them.

## Known limitations

- **Enhancement is slow on CPU.** Real-ESRGAN x4plus costs roughly 110 s per
  input megapixel on a 16-core CPU, so a 400×596 query adds ~50–60 s.
  BiRefNet peaks at 6.6–9 GB of RAM at 1024×1024, so it runs only when at
  least 8 GB is available (otherwise the ESRGAN result is used and the skip is
  logged); memory is released after each restore. Only one restore runs at a
  time.
- **Latency.** A full web search usually takes 18–40 s (the web phase budget
  plus Google Lens's browser upload). Repeat searches are served from cache.
- **Scraping is brittle.** Google, Bing and Yandex change their pages without
  notice; parsers (`app/engines/`) and the proxify patches need maintenance.
- **Face search is sensitive.** Finding where a person appears online can be
  misused. If others use your instance, consider limiting face results to the
  own index or to admins.

## Operations

- **Backups** — the StrikeDB WAL (`data/dbstrike.wal`); thumbnails and uploads
  under `data/`. Metadata in KV is the source of truth, so the vector index
  is always rebuildable.
- **Scaling** — run the pipeline in a separate process (`python -m app.worker`,
  `INDEXER_RUN_INLINE=false`).
- **Monitoring** — `/metrics` (Prometheus), `/v1/stats`, structured JSON logs
  with request IDs (`data/app-8010.log`).

## Tests

```bash
uv run pytest
# or, on this box:
.venv/bin/python -m pytest -o addopts="" -q
```

## Project layout

```
app/
  main.py          app factory, shared model instances, middleware, metrics
  api.py           HTTP API: search ladder, text/face/clip/OCR search, admin, auto-track
  models.py        result / response types, verdict families
  features.py      perceptual hashes (8 orientations) + colour histograms
  matcher.py       own-index search, verdict ladder, consolidation, ranking
  geometry.py      SIFT + RANSAC "same picture, modified"
  face.py          YuNet + ArcFace Face-ID (CPU)
  clip.py          CLIP ViT-B/32 content embeddings (ONNX)
  enhance.py       Real-ESRGAN + BiRefNet query restore (ONNX)
  ai.py            OpenAI-compatible captions + embeddings
  ocr.py           Tesseract text extraction
  resolve.py       image_url page → pixels (og:image, FB cookies, browser render)
  proxify.py       proxify /fetch client (raw-byte fallback downloads)
  cookies.py       Netscape cookie file reader
  engine_guard.py  per-engine pacing, block detection, cooldowns
  searx.py         SearXNG client
  engines/         yandex_images, bing_images, google_images (Lens), text_images
  indexer.py       crawler pipeline, tracker loop, AI backfill, LRU eviction
  repo.py          domain repository over the storage protocol
  storage/         protocol + StrikeDB (RESP) + SQLite backends
  auth.py          API keys
  worker.py        standalone pipeline process
  static/          UI (no build step: HTML/CSS/JS)
deploy/            SearXNG settings
docker/            Dockerfiles (incl. branded-Chrome proxify image)
docs/              ARCHITECTURE.md, logo.svg
scripts/           start_strikedb.sh, start_proxify.sh, filter_cookies.py, utilities
reference/         your SearchOS / StrikeDB / proxify checkouts (git-ignored)
tests/             pytest suite
```

## License

MIT. Your stack: SearXNG (AGPL), StrikeDB (Apache-2.0) — both self-hosted.
