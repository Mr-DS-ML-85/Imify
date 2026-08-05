# Imify — SESSION HANDOFF (resume here)

> Read `memory.md` first, then this file. This captures the EXACT in-flight state of the
> current debugging (YouTuber-face accuracy + FB-logo zero-hits) so the next session
> can resume without re-investigating. **Last updated**: 2026-08-05.

---

## CURRENT OPEN BUG (the user is blocked on this)

**"YouTuber face → 24 results from Bing only, all junk (animals/girls/objects; the person is male).
Yandex, SearXNG and Google contribute nothing. FB logo → 'Engines checked the whole internet but
found nothing: yandex-upload, google-upload'."**

### What is already PROVEN

1. **The engine chain RUNS** (log evidence, request `6f0edc586425` / `1a4f02a0f434`):
   - `yandex-upload reverse` fires → Yandex runs first.
   - `bing-upload reverse` fires → Bing runs when Yandex yields 0.
   - `google-upload reverse` fires → Google runs last.
   - `bing-upload degraded` (400) is intermittent — Bing 400s sometimes, 302→200 other times.
2. **Bing DOES return 24 candidates** for the YouTuber face (they are Bing's visual-similar junk).
   Our hash matcher flags them `unverified` — that part works. The user sees them as "24 matches"
   but they are labeled unverified in the UI.
3. **Yandex returned "No matching results" for the FB logo AND the YouTuber image** even with a
   results-only wait selector (15.2s wait). Saved pages: `/tmp/yandex_fb2.html`, `/tmp/yandex_yt.html`
   (both contain `SerpList_emptyPage`, `CbirIntent`, "No matching results.", 0 avatars).
4. **Google returned 0 hits in 0.3s** (old code used `use_browser: False` → shell).

### Patches applied but NOT yet live-verified (this is where the next session starts)

| File | Change | Status |
|---|---|---|
| `app/engines/yandex_images.py` | `_upload_via_fetch` wait_selector → `'.serp-item, .CbirSites, .ImagesContentSearchItem, [class*="serp-item"]'` (was `.SerpList` which matches the EMPTY shell) | 🟡 applied, not live-verified |
| `app/engines/google_images.py` | `_upload_via_fetch` → `use_browser: True, force_strategy: "gui_chrome"`, `wait_selector: '.islrc, [class*="islrc"], #yDmH0d img, a[href*="/imgres"]'`, `bypass_cache: True` | 🟡 applied, not live-verified |
| `app/engines/bing_images.py` | routing: tunnel → direct → browser-POST (browser POST returns 404 shell — NOT authoritative) | ✅ tests pass |
| `app/api.py` | byte-upload chain order: **yandex → bing → google**; gate includes yandex | ✅ |
| `tests/test_google.py` + `tests/test_yandex.py` | updated for browser-first contract | ✅ 116 pass |

### EXACT next verification steps (in order)

1. **Yandex live test** (upload a WIDELY-INDEXED image — e.g. a known YouTube thumbnail or the
   kitten at `/tmp/yandex_image_proxy.jpg` which came from Yandex itself):
   ```bash
   cd /run/media/irfan/models1/imify && uv run python - <<'PYEOF'
   import os, glob, base64, re, sys, time
   sys.path.insert(0, '.'); os.chdir('/run/media/irfan/models1/imify')
   import httpx
   raw = open('/tmp/yandex_image_proxy.jpg','rb').read()
   boundary='----v1'; parts=[f'--{boundary}\r\n'.encode(), b'Content-Disposition: form-data; name="upfile"; filename="q.jpg"\r\n', b'Content-Type: application/octet-stream\r\n\r\n', raw, b'\r\n', f'--{boundary}--\r\n'.encode()]
   t0=time.time()
   r=httpx.post('http://127.0.0.1:8081/fetch', json={'url':'https://yandex.com/images/search','method':'POST','body_b64':base64.b64encode(b''.join(parts)).decode(),'headers':{'Content-Type':f'multipart/form-data; boundary={boundary}','Referer':'https://yandex.com/','Accept-Language':'en-US,en;q=0.9'},'timeout':45,'use_browser':True,'force_strategy':'gui_chrome','wait_selector':'.serp-item, .CbirSites, .ImagesContentSearchItem, [class*="serp-item"]','bypass_cache':True}, timeout=110)
   h=r.json().get('html',''); open('/tmp/yandex_verify.html','w').write(h)
   print('serp-item:',h.count('serp-item'),'| avatars:',len(re.findall(r'avatars\.mds\.yandex\.net[^"\'\ <]+',h)),'| No matching:',h.count('No matching'),'| time:',round(time.time()-t0,1),'s')
   PYEOF
   ```
   - **If avatars > 0** → Yandex works; restart app and rerun the user's upload → they'll get real candidates.
   - **If still "No matching results"** → see hypothesis in P6: Yandex byte-upload may need a
     **follow-up GET with `cbir_id`** after the POST registers the upload. Check `/tmp/yandex_verify.html`
     for `cbirId` in the page; if present, build the classic second request:
     `https://yandex.com/images/search?rpt=imageview&cbir_id=<id>` (or the `cbir-page` JSON) and parse that.
2. **Google live test**:
   ```bash
   # same pattern, url=https://lens.google.com/upload, field=encoded_image, wait_selector='.islrc, [class*="islrc"], #yDmH0d img, a[href*="/imgres"]'
   ```
   Look for `"ou"` JSON or `imgres?imgurl=` in returned HTML. If `sorry`/captcha → cookies not
   loaded into gui_chrome (check `curl -s http://127.0.0.1:8081/cookies/status`).
3. **Bing 400** — intermittent. Check if rate-limited: too many uploads in a short window. If so, add
   a small sleep / per-domain rate limit. The 302→200 path proves Bing CAN work.
4. **SearXNG in byte-upload path** — not wired. Decide approach (see PROBLEMS.md P4).

### The user's exact product ask (what "done" looks like)

> "i want similar images from google, yandex and bing and searxng if it doesnt match"
> "we want tinyeye + google circle to search for all exact images but we should be able to find
> my face ... to know which sites reveals my face to request a removal"
> "strikeDB is super fast search online" (real-time)

So: **exact matches (verified, hash-judged) + similar images from ALL four engines, fast, with
honest per-hop resolution logging.** Junk candidates must be clearly `unverified` and ranked below
verified matches (verified ranking already works; the gap is that Yandex/Google/SearXNG return
NOTHING so the grid is 100% Bing junk).

---

## REPRO QUICKSTART

```bash
# stack
curl -s http://127.0.0.1:8010/health/live && redis-cli -p 6008 PING
curl -s http://127.0.0.1:8081/health | head -c 120

# search the YouTuber upload (newest in data/uploads)
uv run python -c "
import glob, os, httpx
p = sorted(glob.glob('data/uploads/*'), key=os.path.getmtime, reverse=True)[0]
raw = open(p,'rb').read()
r = httpx.post('http://127.0.0.1:8010/v1/search', files={'file':('q.png',raw,'image/png')}, data={'include_web':'true','learn':'false','num_results':'30'}, timeout=260)
d = r.json()
print('found:',d['found'],'| providers:',d['providers'],'| degraded:',d['degraded'])
print('verified:',sum(1 for x in d['results'] if x['match_type']!='unverified'),'| similar:',sum(1 for x in d['results'] if x['match_type']=='unverified'))
for x in d['results'][:8]: print(' -',x['match_type'],round(x['score'],3),str(x['url'])[:70])
"

# app log for engine evidence
grep -E 'upload reverse|degraded' data/app-8010.log | tail -10
```

## IMPORTANT PITFALLS

- **App must run via systemd** (`systemctl --user restart imify.service`) — plain nohup dies when the
  spawning shell exits in this environment.
- **After changing app code**: `systemctl --user restart imify.service`, wait ~12s, check health.
- **After changing proxify code**: docker cp into container + `docker restart proxy-orchestrator`,
  AND copy to `~/Backup/searchapi/proxify` (user's main checkout). Never trust a git pull alone.
- **Bing browser-POST 404 shell** — never treat as a verdict.
- **`.SerpList` matches empty Yandex pages** — never wait on it; use `.serp-item`/`.CbirSites`.
- **Failed searches must not index the upload** — `_maybe_track_upload(verified=...)` handles this; don't regress.
- **Port 6380 is another agent's** — never kill it. Imify uses 6008.
- **No paid APIs/proxies/scraping** — own stack only (SearXNG + proxify + StrikeDB).
