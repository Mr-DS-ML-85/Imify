# Imify architecture

Imify is a reverse image search engine built on the SearchOS philosophy:
**own your index, own your pipeline, own your data.** This document explains
how a search works end to end and the decisions behind it. For setup and the
API, see the [README](../README.md).

## 1. The search path

```
POST /v1/search  (file | image_url | image_id)
 │
 ├─ acquire pixels
 │    file      → size-checked bytes
 │    image_url → app/resolve.py: direct image, og:image, Facebook session
 │                cookies, gui_chrome render; refused downloads retried via
 │                proxify raw bytes (app/proxify.py)
 │    image_id  → best local copy (original upload or thumbnail)
 │
 ├─ optional restore (ENHANCER_ENABLED): low-resolution query only
 │    → Real-ESRGAN x4 (tiled) → BiRefNet subject mask (memory-guarded)
 │
 ├─ query signals, computed once and reused everywhere
 │    pHash/dHash/aHash (+ 8 orientations) · colour histogram
 │    CLIP embedding · face embeddings · decoded pixels for SIFT
 │
 ├─ 1. OWN INDEX (milliseconds)
 │    hash ANN (VSEARCH) or exact scan → exact rerank → exact/near bands
 │    + SIFT crops of the query inside indexed images  (modified)
 │    + "AI grid": same face (face) · CLIP similar · OCR text   → same list
 │
 ├─ 2. WEB PHASE (WEB_PHASE_BUDGET_SECONDS = 18 s, cached 24 h per SHA-256)
 │    candidates, all engines in parallel:
 │      Yandex upload · Yandex face crop · Bing upload · Google Lens (proxify
 │      in-page upload) · SearXNG reverse (upload served back over loopback)
 │    text discovery: best guess from result titles → SearXNG image/icon
 │      engines (blocked ones retried in proxify's browser)
 │    verify every candidate (§3) within the remaining budget
 │
 ├─ 2b. background tracker for a new upload (§6)
 │
 └─ 3. consolidate ("found on N sites") → admission gate for unverified
        candidates → rank by family (copies, same person, leads) → response
```

With `include_web=false` only step 1 runs. Each engine reports success or
failure on its own; a failing or resting engine lands in `degraded` and never
fails the request.

## 2. Own-index matching: Hamming distance as cosine distance

Perceptual hashing (pHash) produces a 64-bit fingerprint per image. Two images
are near-duplicates when their hashes are close in **Hamming distance**.
StrikeDB's `VSEARCH` is a cosine ANN, but if a 64-bit hash `h` is encoded as a
64-dim vector where bit `i` becomes `+1` / `-1`:

```
dot(v_a, v_b) = 64 - 2 * hamming(a, b)
cosine_dist   = 1 - dot / 64 = hamming(a, b) / 32
```

so `VSEARCH` returns **Hamming-ordered** neighbours, and the distance shown
in the UI is a real perceptual distance. The ANN is approximate (~0.99
recall), so the top candidates are reloaded from KV and re-scored exactly.
Indexes up to `EXACT_SCAN_MAX_IMAGES` skip the graph and scan exactly.

The verdict uses a **fused distance**, far more precise than one hash:
`0.5·pHash + 0.3·dHash + 0.2·aHash` (each normalized to 0–1), with
`exact ≤ 0.14` and `near ≤ 0.35`. The query is hashed in all 8
orientations (rotations × mirror), so a phone photo turned 90° or a flipped
re-upload still matches; non-identity orientations must be tighter
(`ROTATION_VARIANT_MAX_FUSED = 0.20`) because 8 tries multiply chance
collisions on flat images.

Hashes cannot see *content*: two different university logos, or a homelab
photo and a black dog, can hash close. So the own index also feeds the
same result list with the AI layers — SIFT crops (`modified`), the same face
(`face`), CLIP neighbours (`similar`) and OCR text hits.

## 3. Verification: engines find, Imify judges

No engine's verdict is trusted. Google limits face results, Bing returns
"visually similar" previews (animals for a company logo), and every engine's
score is opaque. `Matcher.verify_web_hits` judges each candidate itself.

**Getting the pixels** (bounded by concurrency, a per-fetch timeout and the
web-phase budget; stragglers become `unverified (timeout)`):

1. **Thumbnail triage** — the engine's ~10 KB thumbnail is judged first; the
   full image is downloaded only when the thumbnail is promising (hash close,
   CLIP close, or a face when the query has one).
2. **Download** through the proxify CONNECT tunnel, with the host's cookies
   from `COOKIE_FILE` attached. Bare CDN URLs with no article page are
   dropped before spending budget on them.
3. **Refused (403/429/anti-bot)** → retried through proxify's `/fetch` raw-byte
   mode (`WEB_VERIFY_PROXIFY_MAX` per search).
4. **Facebook** (§5) → the cookie + browser-render chain
   (`WEB_VERIFY_RENDER_MAX` renders per search, ~10 s each).
5. Downloads are cached on disk per URL for 7 days, so repeats and trackers
   never fetch the same candidate twice.

**The verdict ladder** — the first rule that holds wins:

| # | Rule | Verdict | Found? |
|---|---|---|---|
| 1 | fused hash ≤ near band (also vs mirrored/rotated query) **and** CLIP ≥ `CLIP_VERIFY_MIN_SIM` (0.80) when CLIP is live | `exact` / `near` | copy |
| 2 | SIFT + Lowe ratio + RANSAC homography: ≥ 40 inliers with inlier ratio ≥ 0.30 alone, or ≥ 18 with ratio ≥ 0.40 and CLIP ≥ 0.80; skipped when CLIP is live and < 0.60 | `modified` | copy |
| 3 | ArcFace cosine ≥ `FACE_ID_SAME_THRESHOLD` (0.42) between a query face and a candidate face | `face` | same person |
| 4 | CLIP ≥ `CLIP_SIMILAR_MIN_SIM` (0.78) | `similar` | lead only |
| 5 | otherwise, with the reason (no match, fetch failed, timeout) | `unverified` | no |

`CONFIRMED_MATCH_TYPES` (`exact`, `near`, `modified`, `face`, plus the local
`changed` / `color` modes) decide `found`. Everything else is a lead.

**Calibration** (measured on this box's own data, see `tests/test_accuracy.py`):

- *Face-ID:* same person ≥ 0.88, different people ≤ 0.30 (median 0.01). This
  needed correct alignment: the face is warped onto the ArcFace 112×112
  5-point template and normalized `(x − 127.5) / 127.5` in RGB. The earlier
  OpenCV `alignCrop` path received only 10 of the 15 landmark values and fed
  raw 0–255 pixels, which made different people score a median 0.74.
- *SIFT:* real copies 58–330 inliers, unrelated pairs ≤ ~30 (always with
  CLIP < 0.62).
- *CLIP:* copies ≥ 0.90, unrelated photos median 0.50 / p90 0.66, two
  different logos of the same style ~0.7–0.8.

**Admission gate for unverified candidates.** With CLIP live, an unverified
candidate has already failed hash, geometry, face and similarity, so it is
noise and stays out of the grid. Without CLIP, only candidates within a tight
perceptual window (`WEB_UNVERIFIED_MAX_FUSED = 0.36`, Bing penalized ×1.35
for its junk history) are listed, after everything else.

## 4. Consolidation: "found on N sites"

Copies of the same photo on several sites are merged into one result whose
`source_urls` / `source_count` carry every site — TinEye's removal list —
instead of a grid of near-identical tiles. Two copies merge when the pHashes
of their **verified bytes** are within `SEARCH_DEDUP_THRESHOLD` (8) bits, or,
for re-encoded copies whose hashes drifted, when CLIP ≥ 0.90. Unverified
candidates never merge into a confirmed group. The grid is ordered by family
(`MATCH_FAMILY_RANK`): copies, then the same person, then leads.

## 5. Engines and the outside world

**Byte uploads.** A new upload has no public URL, so the pixels themselves are
uploaded to engines that accept them ("circle to search"). Non-JPEG/PNG
queries are re-encoded as JPEG first (Bing answers WebP with unrelated
results). All engines run in parallel inside the web budget:

- **Yandex** (`rpt=imageview`) — the least-censored engine for people. A
  strict parser drops auth/support/tune links that JS-only shells contain.
  When the main face is < 20% of the frame, a tight face crop is uploaded too
  (`yandex-face`), because Yandex matches people by the face.
- **Bing Visual Search** — byte POST through the tunnel, then direct.
- **Google Lens** — Lens binds results to the browser session that uploaded,
  so a plain HTTP upload only gets a JavaScript shell. Proxify's GUI Chrome
  opens lens.google.com and submits a real upload form (`in_page_upload`);
  `parse_lens_results` reads the result records (source page, title, image,
  thumbnail). The impersonated CONNECT-tunnel upload is the fallback, and a
  zero-match page is logged with what came back.
- **SearXNG** — URL-based, so it gets a copy of the upload served back over
  loopback.

**Text discovery (Lens "best guess").** Result titles from the reverse engines
yield a best-guess query ("elon musk", "python logo"), which is searched on
SearXNG's general image engines (Flickr, Openverse, DeviantArt, Wikimedia,
Pinterest, …) and, for logo/icon-like queries, its icon engines. Engines that
are blocked are retried through proxify's browser (at most 2 per search).
Every candidate goes through the same verifier. This finds copies no reverse
index has seen.

**Engine guard** (`app/engine_guard.py`). Every engine call is paced
(Google 12 s, Yandex 4 s, Bing 3 s minimum, jittered ×0.7–1.6; Google 150
per day). CAPTCHA / "unusual traffic" / 403 / 429 answers put the engine to
rest for 10 minutes, doubling up to 6 hours, persisted in storage (`eg:*`).
CAPTCHAs are never solved. Live searches, trackers and text-discovery
browser retries all go through the guard.

**Facebook.** `lookaside.fbsbx.com` answers with a 388-byte JS redirect stub
(`location.href` with JSON-escaped slashes), and `photo.php` is a JS-rendered
shell with no server-side image URL. The chain: fetch the stub through the
tunnel → follow `photo.php` with the real session cookies → if still no image,
render it in proxify's `gui_chrome` and extract the **signed**
`scontent.*.fbcdn.net` URL (its `_nc_*`/`oe` parameters are the auth) → fetch
it with cookies. `FACEBOOK_RESOLUTION_ENABLED=false` turns this off (it breaks
when Facebook changes markup and is against Facebook's terms).

**Proxify integration.** Imify depends on several changes in the user's
proxify, which are mounted into its container by `scripts/start_proxify.sh`:
binary responses as `body_b64`, `raw: true`, in-page uploads, honest
`success` (HTTP errors are failures; 404/410 are final), and block-marker
detection outside inline scripts (every real Google page's JavaScript
mentions `/sorry/index`, which used to flag good pages as blocked).

## 6. Index growth, trackers and storage bounds

**On-the-fly by default.** A search returns verified matches without storing
them (`learn=false`). With `learn=true`, copies become **source links** on the
canonical entry and other photos of the same person become new index
entries. A failed search never adds the query image to the index.

**Trackers** (`AUTO_TRACK_SEARCHES`). A new upload is saved under
`data/uploads/<sha256>.<ext>` and gets a tracker (`track:*`, deduped by
pHash). Every `TRACKER_INTERVAL_SECONDS` the indexer re-searches it: with a
public URL through the URL-based engines (SearXNG, Google, Bing, Yandex),
otherwise by uploading the saved bytes (Bing, then Yandex, then Google Lens).
Candidates are verified like live results before ingestion, so unrelated
images never enter the index.

**Crawler** (SearchOS pipeline). Seed queries (`query:meta:*`), pending URLs
(`pending:*`) and images (`img:meta:*`) live in KV, so a restart resumes where
the pipeline stopped. A discovery loop schedules SearXNG queries with a
coverage map per seed (last run, last count, total); fetch workers claim
pending URLs, validate them (timeout, size cap, content-type + magic bytes,
dimension floor, decompression-bomb ceiling), deduplicate and index.
Safeguards: retry with exponential backoff, a circuit breaker for a throttled
SearXNG, stall detection.

**Deduplication.** By URL (`img:url:<sha256(url)>`), by bytes
(`img:sha:<sha256>`), and by exact pHash. Near-pHash variants are kept as
separate entries — finding variants is the point of reverse search.

**Background AI backfill.** An `ai-backfill` thread embeds faces and CLIP
vectors for entries that lack them, or whose face embedding came from an older
model (`ImageMeta.face_model`), using the original upload when present.

**Bounds.** The index is capped at `MAX_INDEX_IMAGES` (50 000) with LRU
eviction. Verified web results are cached per image for 24 h (lead-only
outcomes for 1 h). Candidate downloads are cached for 7 days, at most 20 000
files.

## 7. Storage: why StrikeDB, and the safety net

StrikeDB is the user's own engine: durable KV + native HNSW vectors over RESP.

| Need | StrikeDB primitive |
|---|---|
| Image metadata (URL, hashes, histograms, embeddings, timestamps) | `SET/GET/MSET/INCR/KEYS` (WAL-durable) |
| Near-duplicate ANN | `VADD id <±1 floats>` + `VSEARCH k` |
| URL / SHA / pHash dedup maps, caches, guard state | KV namespaces |

Constraints engineered around:

1. **VADD ids are `u64`** → a monotonic `img:seq` counter allocates ids.
2. **One vector graph, one dimension** → the 64-dim graph holds the pHash
   encoding. Colour histograms, CLIP (512-dim) and face embeddings are matched
   by exact in-app scans over caches rebuilt from KV.
3. **Young engine** → the whole domain sits behind the `Storage` protocol;
   `STORAGE_BACKEND=sqlite` swaps in a stdlib backend with the same semantics.
4. **Metadata is the source of truth** → the ANN is always rebuildable.

**Removals.** `img:removed:<id>` tombstones filter removed ids on every read
path; `VDEL` drops the vector from the graph; the thumbnail and counters are
cleaned up and the histogram cache invalidated. `purge` matches by URL
substring or source query.

## 8. Query restore (enhancement)

`app/enhance.py`, opt-in with `ENHANCER_ENABLED`. Only queries at or below
`ENHANCE_MAX_PIXELS` (250 000 px, about 500×500) are restored;
higher-resolution images are left untouched and never downscaled.

1. **Real-ESRGAN x4plus**, run in 256-px tiles with 16 px of context, so
   memory stays around 2 GB whatever the size. Measured cost on this 16-core
   CPU: ~110 s per input megapixel (a 400×596 query takes ~50 s).
2. **BiRefNet** subject mask at 1024×1024; the sharpened subject is
   composited over a slightly blurred background. It peaks at 6.6–9 GB of RAM
   on CPU, so it only runs when at least 8 GB is available; otherwise the
   ESRGAN result is used and the skip is logged.

**Placement.** `ENHANCE_DEVICE=auto` plans each model at startup with
`plan_devices`: models are sorted by VRAM need (BiRefNet 4.5 GB, ESRGAN
1.5 GB, estimates) and each goes to the GPU while it fits in the free VRAM
minus 512 MB of headroom, otherwise to system RAM. Free VRAM comes from
`nvidia-smi`; without a CUDA-capable onnxruntime everything is placed in RAM.
A GPU session that cannot be created, or a GPU run that fails, rebuilds that
model in RAM and retries once. The 8 GB RAM guard applies only to a BiRefNet
running in RAM. `cpu` / `gpu` force both models.

One restore runs at a time, and both models run without ONNX Runtime's memory
arena, so memory returns to ~1 GB after each restore instead of staying at the
peak. The restored pixels feed the query's hashes and
AI signals; the web engines get the original upload unless
`ENHANCE_BEFORE_WEB_UPLOAD=true`. On CPU this step dominates search time.

## 9. Failure behaviour

- A failing provider degrades the response (`degraded:
  ["provider:google-upload"]`, `cooldown:google`) instead of failing it; the
  own index always answers.
- Google is proxify-only: without `PROXIFY_FETCH_URL` it degrades rather
  than fetching Google directly.
- A candidate that cannot be downloaded, rendered or judged is `unverified`,
  never a false match.
- Trackers record a note and skip on errors; they never crash the loop.
- An unreachable StrikeDB fails startup with a clear message (with the
  `sqlite` escape hatch).
- Rate limits return `429` + `Retry-After`; API keys are SHA-256 hashed at rest.

## 10. Deployment

`docker-compose.yml` runs `strikedb` (built from your checkout), `searxng`,
`app` and an optional `worker`. `INDEXER_RUN_INLINE` controls whether the
pipeline runs inside the gateway (single node) or in the worker (scale-out).
On this box Imify runs as the user service `imify.service` on `127.0.0.1:8010`,
whose `ExecStartPre` starts StrikeDB and proxify. All knobs are env-driven via
pydantic-settings.
