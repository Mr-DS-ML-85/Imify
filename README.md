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

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /v1/search` | key\* | Reverse search. `multipart` `file`, or `image_url`, or `image_id`. Query params: `mode` (`best\|color\|biggest\|newest\|changed\|exact`), `num_results`, `include_web`. |
| `GET /v1/image/{id}` | – | Indexed image metadata. |
| `GET /v1/history` | – | Recent searches. |
| `GET /v1/stats` | – | Index statistics. |
| `GET /health/live`, `/health/ready` | – | Probes. |
| `GET /metrics` | – | Prometheus text format. |
| `POST /v1/admin/ingest` | admin | `{"queries": ["q1", ...], "urls": ["https://..."]}` — seed queries and/or specific image URLs. |
| `GET /v1/admin/queries` · `PATCH/DELETE /v1/admin/queries/{id}` | admin | Seed query management. |
| `GET /v1/admin/crawl` | admin | Pipeline status. |
| `GET /v1/admin/images?q=&limit=&offset=` | admin | List indexed images (filter by URL/title/engine). |
| `DELETE /v1/admin/images/{id}` | admin | Remove one image from the index. |
| `POST /v1/admin/images/purge` | admin | Bulk-remove: `{"url_contains": "example.com", "query": "..."}`. |
| `GET /v1/admin/tracks` · `PATCH/DELETE /v1/admin/tracks/{id}` | admin | Background tracker management (auto-update). |
| `POST /v1/admin/tracks/run` | admin | Run one tracker round now. |
| `POST /v1/admin/keys` · `GET` · `DELETE /{name}` | admin | API keys (SHA-256 hashed at rest). |

\* `AUTH_MODE=key` requires a key for all `/v1/*`; `AUTH_MODE=open` (default)
only requires keys for admin endpoints.

```bash
curl -F "file=@photo.jpg" -F "mode=best" http://localhost:8000/v1/search
curl -H "Authorization: Bearer sk_..." http://localhost:8000/v1/admin/queries
```

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
| `FUSED_EXACT_THRESHOLD` / `FUSED_NEAR_THRESHOLD` | `0.08` / `0.22` | fused pHash+dHash+aHash match bands (the verdict) |
| `WEB_VERIFY_CANDIDATES` | `true` | download + hash-verify web candidates before reporting |
| `WEB_VERIFY_CONCURRENCY` / `WEB_VERIFY_MAX_CANDIDATES` | `6` / `24` | candidate fetch limits |
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

### Google reverse = proxify only (TinEye / Circle-to-Search style)

Google answers `searchbyimage` with a `sorry` block to datacenter IPs. Imify
never fetches Google directly — the `google` provider is **proxify-only** and
degrades (`provider:google` in `degraded`) when `PROXIFY_FETCH_URL` is unset.
Start your [proxify](https://github.com/Mr-DS-ML-85/proxify) gateway and point
it here. Proxify's default API port is 8080 — if SearXNG already owns 8080 on
your box, run proxify with `API_PORT=8081` and set `PROXIFY_FETCH_URL=http://127.0.0.1:8081`.

**Circle-to-Search for images with no online links:** a brand-new upload has
no public URL, so URL-based reverse engines can't see it. Imify uploads the
raw bytes to Google's `/searchbyimage/upload` **through proxify** (base64 body
on `/fetch` — with your cookies + pinned Chrome persona — or the CONNECT
tunnel). That's the path that can find a face/object photo that exists online
but has no reachable URL on your side — the same trick TinEye and Google Lens
use. The result pages are parsed into hits and fed back into your index.

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
