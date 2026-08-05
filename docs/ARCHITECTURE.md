# Imify architecture

Imify is a reverse image search engine built on the SearchOS philosophy:
**own your index, own your pipeline, own your data.** This document maps the
SearchOS ideas onto the image domain and explains the non-obvious decisions.

## 1. The core trick: Hamming distance as cosine distance

Perceptual hashing (pHash) produces a 64-bit fingerprint per image. Two images
are near-duplicates when their hashes are close in **Hamming distance**.

StrikeDB's `VSEARCH` is a cosine ANN. But if you encode a 64-bit hash `h` as a
64-dim vector where bit `i` becomes `+1` / `-1`:

```
dot(v_a, v_b) = 64 - 2 * hamming(a, b)
cosine_dist   = 1 - dot / 64 = hamming(a, b) / 32
```

So `VSEARCH` returns **Hamming-ordered** neighbors directly, and the returned
distance is a *monotone function of the actual Hamming distance*. This gives:

- sub-millisecond near-duplicate search at 1M-image scale (StrikeDB's own
  benchmark: ~505 µs p99 for 100k×384d),
- the reported `Δdistance` shown in the UI is a real perceptual distance,
  not a made-up score.

The ANN is approximate (~0.99 recall), so the top candidates are loaded from KV
and re-scored with the exact Hamming distance (the same
"INT8 traversal + exact f32 rerank" pattern StrikeDB itself uses).

Small indexes (≤ `EXACT_SCAN_MAX_IMAGES`) skip the graph entirely and do an
exact brute-force scan — guaranteed recall, still fast at 64-bit hashes.

## 2. Why StrikeDB, and how the risk is engineered around

StrikeDB is your own unified engine: durable KV + native HNSW vectors over RESP.

| Need | StrikeDB primitive |
|---|---|
| Image metadata (URL, hashes, histograms, timestamps) | `SET/GET/MSET/INCR/KEYS` (WAL-durable) |
| Near-dup ANN | `VADD id <±1 floats>` + `VSEARCH k` |
| URL/SHA/phash dedup maps | KV namespaces |
| Ingest-rate time series (optional) | `TSADD`/`TSRANGE` |

Design constraints we engineered around:

1. **VADD ids are `u64`** → a monotonic `img:seq` counter allocates ids.
2. **One vector graph, one dimension** → the single 64-dim graph is reserved
   for the pHash ±1 encoding; color histograms are matched with an exact
   in-app scan over a RAM-hot cache rebuilt from KV.
3. **Young engine** → the entire domain lives behind the `Storage` protocol.
   `STORAGE_BACKEND=sqlite` swaps in a stdlib backend with identical semantics
   (the vector layer falls back to an exact numpy cosine scan). The product can
   never be sunk by an experimental persistence bug.
4. **Metadata is the source of truth** → hashes and histograms are stored in
   the KV metadata JSON *and* fed to the vector index, so the ANN is always
   rebuildable after a restart or a backend migration.

## 3. The pipeline (SearchOS ideas, ported)

SearchOS's central claim: *search progress is a system asset, not transient
conversation history*. Imify applies that to crawling:

- **State as an asset** — seed queries (`query:meta:*`), pending URLs
  (`pending:*`) and images (`img:meta:*`) live in KV. A crash, restart or
  redeploy resumes exactly where the pipeline stopped.
- **Orchestrator / workers** — a discovery loop schedules SearXNG queries
  (the "coverage map": each seed query tracks last-run, last-count, total);
  a pool of workers claim pending URLs, fetch, validate, deduplicate and index.
- **Learns from the web** — SearXNG reverse-search hits are enqueued into the
  own index. Repeat searches hit memory, not the web.
- **Middleware safeguards** — retry with exponential backoff, circuit breaker
  for a throttled SearXNG (429s), stall detection, and hard fetch guards
  (timeout, size cap, content-type + magic-byte validation, dimension floor,
  decompression-bomb ceiling via `Image.MAX_IMAGE_PIXELS`).

### Deduplication levels

1. **URL** → `img:url:<sha256(url)>` — the same URL is never re-fetched.
2. **Bytes** → `img:sha:<sha256>` — byte-identical copies merge sources.
3. **Perceptual** → exact pHash match merges (perceptually identical); near
   pHash matches are kept as **separate index entries** — the whole point of a
   reverse image search is to find *variants*, so variants are indexed, not merged.

## 4. The search path (SearchOS query router) + the hash is the judge

```
query image (upload / url / image_id)
  → features (pHash/dHash/aHash/color)
  → local: VSEARCH (or exact scan) → exact rerank → match_type bands
       ├─ found (>= WEB_FALLBACK_LOCAL_MIN_HITS) → serve local ONLY, no web call
       └─ thin/empty → live reverse chain, in order:
           1. SearXNG image_url reverse (non-empty `q` fallback for builds
              that reject empty queries)
           2. Bing byte-upload — the actual pixels POSTed as base64
              (`image_url`), works for private-box images with no public URL
           3. Yandex imageview byte-upload — least-censored engine, finds
              face/people matches Google suppresses (strict parser: auth /
              support / tune junk is never emitted as a hit)
           4. Google searchbyimage / Lens byte-upload — proxify-only
           5. nothing → degraded[] populated, still local-only
  → VERIFY every web candidate: download it, extract its hashes, compute the
    fused distance to the query. Only fused ≤ near threshold becomes a
    reported match (match_type exact/near, real score). Everything else is
    listed as "unverified" (low score, dashed badge) and NEVER counts as found.
  → fusion: URL-normalized dedup, indexed copy preferred, RRF-style blend
  → rerank: exact matches > near > size/freshness bonuses
  → response carries `found` + `fallback` ("local" | "web" | "none")
```

`mode=color` replaces the perceptual path with histogram cosine; `biggest` /
`newest` / `changed` re-sort the strong-match set. Only **verified** web hits
are learned: they are enqueued into the own index so repeat searches are served
from memory and the query entry's `source_count` grows into the list of sites
carrying that image.

### Why verification is the whole point

Google and other engines limit results for faces (privacy policy) and their
"similarity" verdicts are opaque. Imify therefore never trusts an engine's
verdict: **engines are candidate finders, the fused multi-hash matcher is the
judge**. A site is only reported as "showing your image" after the candidate
image itself is downloaded and its pHash/dHash/aHash distance to the query is
within the near band — which is exactly how TinEye works, with three fused
hashes instead of one for far better discrimination. A derived copy (re-encode,
resize, brightness shift, watermark) stays within the near band and is found;
an unrelated image is rejected (`found=false` → "search not found").

Fused distance: `0.5·pHash + 0.3·dHash + 0.2·aHash` (each normalized to 0-1),
with `fused_exact_threshold=0.08` (exact) and `fused_near_threshold=0.22`
(near). Candidates are fetched concurrently (`web_verify_concurrency=6`, capped
at `web_verify_max_candidates=24`) with a short timeout each so a slow host
never stalls the search.

### Hotlink-protected CDNs (Facebook) — cookies + browser render

Some candidates live behind hotlink protection. Facebook's
`lookaside.fbsbx.com` URLs answer with a 388-byte **JS redirect stub**
(`location.href = "https:\\/\\/www.facebook.com\\/photo.php?..."` — note the
JSON-escaped slashes, which must be unescaped before following), and the
`photo.php` target is a **JS-rendered comet shell** with no server-side image
URL. Candidate verification therefore walks a three-hop chain for the
facebook/fbsbx/fbcdn host family:

1. fetch the stub through the CONNECT tunnel → unescape `location.href`;
2. fetch `photo.php` with the **real session cookies** from `gui-cookies.txt`
   (Netscape file, domain-matched per host — `COOKIE_FILE`);
3. still no image → render the page through proxify's **gui_chrome**
   (`PROXIFY_FETCH_URL`, the persistent browser that carries the imported
   session), extract the **signed** `scontent.*.fbcdn.net` image URL from the
   DOM (its `_nc_*`/`oe` query params are the auth — the full query string is
   kept and HTML entities unescaped), then fetch it with cookies through the
   tunnel.

Only then are the pixels hashed and judged. Unrelated or expired candidates
(stale `media_id`s) resolve to a real image but hash-reject (`unverified`) —
proven live against real `media_id`s in both directions (accept: the photo's
own bytes → `exact` hamming 0; reject: an unrelated query → `unverified`).

Renders are expensive (~10s each through proxify's single shared Chrome), so
they are budgeted per search (`web_verify_render_max=4`) — a FB-heavy result
set never stalls the whole search, and the budget is shared across candidates.

### Yandex strict parsing

Modern Yandex imageview responses are JS-gated shells: the classic `cbir-page`
JSON is gone and the page contains only auth / support / tune links. The
parser hard-filters those hosts (`passport.yandex`, `yandex.com/support`,
`/auth`, ...) so they are **never** emitted as hits, and the generic URL
fallback is disabled for JS shells — a Yandex round that finds nothing degrades
the chain instead of polluting results with junk that could never match.

## 5. Auto-update: background trackers

When a search uploads an image that is **not** already indexed:

1. the upload is persisted under `data/uploads/<sha256>.<ext>` and enqueued for
   local indexing (the worker fetches it over loopback — no public exposure
   needed to index it),
2. a **tracker** row (`track:*`, deduped by pHash) records it, and
3. the indexer's tracker loop (`TRACKER_INTERVAL_SECONDS`) reverse-searches it
   across the whole internet — SearXNG first, then Google via proxify — and
   ingests every new match into the index.

Trackers need a **publicly reachable** URL to reverse-search: `PUBLIC_BASE_URL`
for uploads, or the source URL for `image_url` queries. Without one they
persist an honest note ("no public URL") instead of erroring. The same image
thus finds *more* matches over time without any manual action.

Tracker candidates are **hash-verified before ingestion**
(`tracker_verify_candidates=true`): each candidate is fetched and compared
against the tracked image's fused fingerprint; only matches within the near
band are enqueued, so unrelated images can never pollute the index. Tracks
store all three hashes (legacy tracks fall back to pHash-only automatically).

## 6. Index removals

`img:removed:<id>` tombstones plus metadata/map deletion make removals
immediate and correct: removed ids are filtered at every read path (metadata
lookups, matcher rerank, histogram cache), and the ANN graph simply never
materializes them (dead vectors are skipped when their metadata is gone). The
matcher's histogram cache is invalidated on removal so color mode stays
consistent. Bulk removal (`purge`) matches by URL substring or source query.

Removal also decrements the images counter and deletes the thumbnail. Since
StrikeDB added **`VDEL id [id...]`** (tombstones the vector, drops its durable
KV and purges postings), `repo.remove_image` issues `VDEL` so dead vectors do
not even occupy a graph slot — the graph stays compact under churn without a
periodic rebuild.

## 7. Failure behavior

- A failing provider degrades the response (`degraded: ["provider:searxng"]`)
  instead of failing the request; the local index always answers.
- The Google provider is **proxify-only**: with `PROXIFY_FETCH_URL` unset it
  degrades (`provider:google`) rather than fetching Google directly.
- Bing and Yandex byte-upload try the proxify tunnel first, then direct; both
  degrade on failure.
- Facebook candidate verification degrades gracefully: if the browser render
  is unavailable (`PROXIFY_FETCH_URL` unset) or the page cannot be resolved,
  the candidate is reported `unverified` rather than a false match.
- Trackers with no public URL record a note and skip; they never crash the loop.
- Unreachable StrikeDB fails startup with a clear message (and a documented
  `sqlite` escape hatch).
- Rate limits return `429` + `Retry-After`; keys are SHA-256 hashed at rest.

## 8. Deployment

`docker-compose.yml` runs the whole stack: `strikedb` (built from your
checkout), `searxng`, `app` and an optional `worker` process. `INDEXER_RUN_INLINE`
controls whether the pipeline runs inside the gateway (single-node) or in the
worker container (scale-out). All knobs are env-driven via pydantic-settings.
