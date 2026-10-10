"""Text-guided image discovery — the Google-Lens "best guess" trick.

Reverse engines (Yandex / Bing / Google Lens) return candidate pages whose
TITLES say what the picture is ("Python logo", "Elon Musk at Tesla event").
Lens turns that into a text query and searches image engines with it, which
finds copies and look-alikes that no reverse index has seen. Imify does the
same, entirely on your own stack:

1. ``derive_queries`` builds 1-2 "best guess" queries from the reverse hits'
   titles (term/bigram agreement across sites), falling back to OCR text.
2. ``TextImageDiscovery.search`` fans the queries out to SearXNG's image /
   icon engines (one parallel request per engine group, engines named
   explicitly so disabled-by-default engines still run). Icon engines only
   run when the query looks like an icon or logo.
3. Engines SearXNG reports as blocked (CAPTCHA / access denied / timeout) are
   retried through proxify's real-browser strategy (gui_chrome + session
   cookies) — the same automation the Yandex/Bing/Google upload paths use —
   and the rendered page is parsed for image + source-page pairs.

Every candidate still goes through the verifier (hash / keypoints / face /
CLIP) — text search only FINDS candidates, it never decides a match.
"""

from __future__ import annotations

import html as html_mod
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx
import numpy as np
from PIL import Image

from app.logging import child_logger
from app.models import ImageHit
from app.searx import DEFAULT_UA

log = child_logger("imify.text_images", engine="text_images")

# SearXNG engine names (as configured in SearXNG) with their !bangs, grouped
# like SearXNG's preferences page.
ICON_ENGINES: dict[str, str] = {
    "devicons": "di",
    "lucide": "luc",
    "material icons": "mi",
    "selfhst icons": "si",
    "uxwing": "ux",
}
WEB_ENGINES: dict[str, str] = {
    "brave.images": "brimg",
    "google images": "goi",
    "mojeek images": "mjkimg",
    "qwant images": "qwi",
    "startpage images": "spi",
}
GENERAL_ENGINES: dict[str, str] = {
    "1x": "1x",
    "artic": "arc",
    "artstation": "as",
    "deviantart": "da",
    "duckduckgo images": "ddi",
    "findthatmeme": "ftm",
    "flickr": "fl",
    "frinkiac": "frk",
    "imgur": "img",
    "ipernity": "ip",
    "library of congress": "loc",
    "openverse": "opv",
    "pexels": "pe",
    "pinterest": "pin",
    "pixabay images": "pixi",
    "public domain image archive": "pdia",
    "sogou images": "sogoui",
    "unsplash": "us",
    "wikicommons.images": "wci",
    "yacy images": "yai",
    "naver images": "nvri",
    "baidu images": "bdi",
    "quark images": "qki",
}
ALL_ENGINES: dict[str, str] = {**ICON_ENGINES, **WEB_ENGINES, **GENERAL_ENGINES}

# Engines that can be driven directly in proxify's real browser when SearXNG
# is blocked. {q} = url-encoded query.
BROWSER_SEARCH_URLS: dict[str, str] = {
    "google images": "https://www.google.com/search?q={q}&udm=2",
    "bing images": "https://www.bing.com/images/search?q={q}",
    "brave.images": "https://search.brave.com/images?q={q}",
    "duckduckgo images": "https://duckduckgo.com/?q={q}&iax=images&ia=images",
    "qwant images": "https://www.qwant.com/?q={q}&t=images",
    "startpage images": "https://www.startpage.com/sp/search?query={q}&cat=images",
}

_STOP = frozenset(
    """a an the and or but of in on at by for with without from to is are was were be been
    this that these those it its as into over under about after before than then there here
    your you our we they he she his her their my me i not no yes new best top free download
    image images photo photos picture pictures pic pics stock png jpg jpeg webp svg gif vector
    hd 4k wallpaper wallpapers background backgrounds clipart clip art icon icons illustration
    illustrations royalty transparent file wiki wikipedia com www http https net org html amp
    search results result page pages view views share shares video videos watch shutterstock
    alamy getty istock dreamstime depositphotos freepik pinterest instagram facebook twitter
    reddit tiktok youtube linkedin flickr imgur tumblr quora medium blog news
    scaled post posts story stories tag tags topic topics category article articles watch
    uploads upload content wp legacy show index amp""".split()
)
# OCR text IS the content ("BREAKING NEWS") — only drop function words.
_OCR_STOP = frozenset("the and for with from this that are was were you your our not but".split())
_SPLIT_TITLE = re.compile(r"\s+[|\-–—:·•»]\s+")
_WORD = re.compile(r"[^\W\d_][\w'’]*", re.UNICODE)


def _slug_text(url: str) -> str:
    """Words from a URL's path slug — `/spend/spend-elon-musks-money-game/`
    -> "spend spend elon musks money game". Engines (Yandex) often give no
    title, but the article slug says what the page is about."""
    try:
        path = unquote(urlparse(url).path)
    except ValueError:
        return ""
    parts = [p for p in re.split(r"[/]", path) if p]
    words: list[str] = []
    for seg in parts[-2:]:
        seg = re.sub(r"\.(?:jpe?g|png|webp|gif|avif|html?|php|aspx?)$", "", seg, flags=re.I)
        for w in re.split(r"[-_+.,]", seg):
            # drop ids/hashes/sizes ("425166", "O", "scaled", "UgkxUnqT3vuz...")
            if w.isalpha() and 3 <= len(w) <= 20 and not (w.isupper() and len(w) > 4):
                words.append(w)
    return " ".join(words)


def _title_terms(title: str, host: str) -> list[str]:
    """Meaningful lowercase words of a result title, minus the site name."""
    host_words = set(re.split(r"[.\-]", host.lower())) if host else set()
    segments = _SPLIT_TITLE.split(title or "")
    # the site name usually sits in the last segment ("... - Wikipedia")
    if len(segments) > 1 and len(segments[-1].split()) <= 3:
        segments = segments[:-1]
    words: list[str] = []
    for seg in segments:
        for w in _WORD.findall(seg.lower()):
            w = re.sub(r"['’]s$", "", w).strip("'’")
            if len(w) >= 3 and w not in _STOP and w not in host_words:
                words.append(w)
    return words


def derive_queries(hits: list[ImageHit], ocr_text: str = "", max_queries: int = 2) -> list[str]:
    """Lens-style "best guess" queries from what other sites call this image.

    Terms are scored by how many DIFFERENT result titles contain them
    (document frequency), so one spammy title cannot steer the query. The
    best bigram that several sites agree on wins; the top unigrams form a
    second, broader query. Falls back to OCR words when titles say nothing.
    """
    docs: list[list[str]] = []
    seen_titles: set[str] = set()
    for h in hits[:40]:
        title = (h.title or "").strip()
        if not title or title.startswith(("http://", "https://")):
            title = _slug_text(h.url or title)
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        terms = _title_terms(title, urlparse(h.url or "").hostname or "")
        if terms:
            docs.append(terms)
    # light normalization: "musks"/"musk's" -> "musk" when the singular form
    # also occurs, so slug variants agree with each other
    vocab = {w for terms in docs for w in terms}
    docs = [[w[:-1] if w.endswith("s") and w[:-1] in vocab else w for w in terms] for terms in docs]
    queries: list[str] = []
    if docs:
        need = max(2, int(len(docs) * 0.15))
        grams: dict[int, Counter] = {n: Counter() for n in (1, 2, 3)}
        for terms in docs:
            for n in (1, 2, 3):
                grams[n].update({
                    " ".join(terms[i : i + n])
                    for i in range(len(terms) - n + 1)
                    if len(set(terms[i : i + n])) == n
                })
        # the LONGEST phrase that several sites agree on, as long as it is
        # reasonably supported vs the best shorter one (>= 60%; "red panda
        # sticker" beats "panda sticker" when every site says both)
        best = ""
        best_df = 0
        for n in (1, 2, 3):
            cand = [(g, c) for g, c in grams[n].most_common(10) if c >= need]
            if cand and cand[0][1] >= 0.6 * best_df:
                best, best_df = cand[0]
        if best:
            queries.append(best)
            # a second, different angle: the best bigram not contained in it
            for g, c in grams[2].most_common(10):
                if c >= need and g not in best:
                    queries.append(g)
                    break
    if not queries and ocr_text:
        words = [w for w in _WORD.findall(ocr_text.lower()) if len(w) >= 3 and w not in _OCR_STOP]
        if words:
            queries.append(" ".join(words[:5]))
    out: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in out:
            out.append(q)
    return out[:max_queries]


def is_iconlike(img: Image.Image | None) -> bool:
    """Logo / icon heuristic: few dominant colors and flat regions (icon
    engines only make sense for these — never for photos)."""
    if img is None:
        return False
    small = np.asarray(img.convert("RGB").resize((64, 64), Image.Resampling.NEAREST), dtype=np.uint8)
    quant = (small // 32).reshape(-1, 3)
    codes = quant[:, 0].astype(np.int32) * 64 + quant[:, 1].astype(np.int32) * 8 + quant[:, 2].astype(np.int32)
    counts = np.sort(np.bincount(codes, minlength=512))[::-1]
    top4 = counts[:4].sum() / codes.size
    return bool(top4 >= 0.80)


# ------------------------------------------------------------------ parsing
def parse_bing_images(page: str) -> list[ImageHit]:
    """Bing Images rendered DOM: each result <li> has an ``iusc`` anchor
    whose href carries ``mediaurl=`` (the image) plus the source-page link
    and title in the same block."""
    hits: list[ImageHit] = []
    for li in re.findall(r'<li[^>]*>\s*<div class="iuscp.*?</li>', page, re.S):
        m = re.search(r'class="iusc"[^>]*href="([^"]+)"', li) or re.search(r'href="([^"]*mediaurl=[^"]+)"', li)
        if not m:
            continue
        qs = parse_qs(urlparse(html_mod.unescape(m.group(1))).query)
        img = unquote((qs.get("mediaurl") or [""])[0])
        if not img.startswith("http"):
            continue
        page_url = next((html_mod.unescape(u) for u in re.findall(r'href="(https?://[^"]+)"', li)), "")
        titles = [
            html_mod.unescape(t) for t in re.findall(r'(?:title|aria-label|alt)="([^"]{3,200})"', li)
            if not t.lower().startswith(("image result", "search for"))
        ]
        title = max(titles, key=len) if titles else ""
        thumb = re.search(r'<img[^>]+src="(https://th[^"]+)"', li)
        hits.append(ImageHit(
            title=title, url=page_url, img_src=img,
            thumbnail_src=html_mod.unescape(thumb.group(1)) if thumb else "",
            engine="bing images",
        ))
    return hits


_IMG_URL = re.compile(r"https?://[^\s\"'<>\\]+?\.(?:jpe?g|png|webp|gif|avif)(?:\?[^\s\"'<>\\]*)?", re.I)


def parse_generic_images(page: str, base_url: str, engine: str) -> list[ImageHit]:
    """Engine-agnostic extraction from a rendered results page: every
    ``<a href=PAGE>…<img src=IMG>…</a>`` pair on a different host than the
    engine, plus bare full-size image URLs embedded in scripts/JSON. Noisy by
    design — the verifier rejects anything that isn't a match."""
    engine_host = (urlparse(base_url).hostname or "").removeprefix("www.")
    hits: list[ImageHit] = []
    seen: set[str] = set()

    def foreign(u: str) -> bool:
        h = (urlparse(u).hostname or "").removeprefix("www.")
        return bool(h) and engine_host not in h and not h.endswith(("gstatic.com", "googleusercontent.com"))

    for a_href, inner in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
        href = urljoin(base_url, html_mod.unescape(a_href))
        m = re.search(r'<img[^>]+(?:data-src|src)="([^"]+)"', inner)
        if not m or not href.startswith("http") or not foreign(href):
            continue
        img = urljoin(base_url, html_mod.unescape(m.group(1)))
        if not img.startswith("http") or img in seen:
            continue
        seen.add(img)
        alt = re.search(r'alt="([^"]{3,200})"', inner)
        hits.append(ImageHit(
            title=html_mod.unescape(alt.group(1)) if alt else "",
            url=href, img_src=img, thumbnail_src=img, engine=engine,
        ))
    for raw in _IMG_URL.findall(html_mod.unescape(page)):
        url = raw.replace("\\u003d", "=").replace("\\u0026", "&")
        if url in seen or not foreign(url):
            continue
        seen.add(url)
        hits.append(ImageHit(title="", url="", img_src=url, engine=engine))
    return hits[:80]


# ------------------------------------------------------------------ client
@dataclass
class DiscoveryReport:
    queries: list[str] = field(default_factory=list)
    hits: list[ImageHit] = field(default_factory=list)
    # engine -> number of hits (0 = answered but nothing / failed)
    engines: dict[str, int] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    browser_used: list[str] = field(default_factory=list)


class TextImageDiscovery:
    def __init__(
        self,
        searxng_url: str,
        proxify_fetch_url: str = "",
        timeout: float = 12.0,
        engines: list[str] | None = None,
        browser_fallback_max: int = 2,
    ):
        self.searxng_url = searxng_url.rstrip("/")
        self.proxify_fetch_url = proxify_fetch_url.rstrip("/")
        self.timeout = timeout
        self.engines = [e for e in (engines or list(WEB_ENGINES) + list(GENERAL_ENGINES)) if e]
        self.browser_fallback_max = browser_fallback_max

    @property
    def available(self) -> bool:
        return bool(self.searxng_url)

    def _searx(self, query: str, engines: list[str], timeout_limit: float = 0.0) -> tuple[list[ImageHit], dict[str, str]]:
        params = {"q": query, "engines": ",".join(engines), "categories": "images", "format": "json"}
        if timeout_limit > 0:
            # SearXNG's per-request engine timeout: slow engines are cut
            # instead of holding the whole request
            params["timeout_limit"] = f"{timeout_limit:.1f}"
        resp = httpx.get(
            f"{self.searxng_url}/search",
            params=params,
            timeout=max(self.timeout, timeout_limit + 3),
            headers={"User-Agent": DEFAULT_UA},
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        wanted = set(engines)
        hits: list[ImageHit] = []
        for r in data.get("results", []):
            img = r.get("img_src") or ""
            eng = r.get("engine") or ""
            # SearXNG silently substitutes default engines for names it does
            # not know — keep only what we asked for
            if not img.startswith("http") or eng not in wanted:
                continue
            hits.append(ImageHit(
                title=(r.get("title") or "").strip(),
                url=(r.get("url") or "").strip(),
                img_src=img,
                thumbnail_src=r.get("thumbnail_src") or "",
                engine=eng,
                width=int(r.get("width") or 0) if str(r.get("width") or "").isdigit() else 0,
                height=int(r.get("height") or 0) if str(r.get("height") or "").isdigit() else 0,
            ))
        blocked = {str(e[0]): str(e[1]) for e in data.get("unresponsive_engines", []) if e and e[0] in wanted}
        return hits, blocked

    def _browser(self, engine: str, query: str) -> list[ImageHit]:
        """Drive the engine's own results page in proxify's real browser."""
        template = BROWSER_SEARCH_URLS.get(engine)
        if not template or not self.proxify_fetch_url:
            return []
        url = template.format(q=quote_plus(query))
        resp = httpx.post(
            self.proxify_fetch_url + "/fetch",
            json={
                "url": url,
                "method": "GET",
                "force_strategy": "gui_chrome",
                "use_browser": True,
                "timeout": 40,
                "bypass_cache": True,
                "wait_selector": "img",
                "headers": {"Accept-Language": "en-US,en;q=0.9"},
            },
            timeout=70,
        )
        resp.raise_for_status()
        data = resp.json()
        page = data.get("html") or ""
        if not page or "captcha" in str(data.get("error", "")).lower():
            return []
        if engine == "bing images":
            hits = parse_bing_images(page)
        else:
            hits = parse_generic_images(page, url, engine)
        for h in hits:
            h.engine = engine
        return hits

    def search(self, queries: list[str], include_icons: bool = False, budget_seconds: float = 6.0) -> DiscoveryReport:
        """Run every query against the configured engines (+ icon engines
        for logos) in parallel, then browser-retry blocked engines while
        budget remains. Hits are interleaved across engines."""
        report = DiscoveryReport(queries=list(queries))
        if not queries or not self.available:
            return report
        started = time.monotonic()
        engines = list(self.engines) + (list(ICON_ENGINES) if include_icons else [])
        # small chunks in parallel: one SearXNG request waits for its slowest
        # engine, so a single 25-engine request is as slow as the worst one
        groups = [engines[i : i + 4] for i in range(0, len(engines), 4)]
        searx_budget = max(1.0, budget_seconds * 0.6)
        per_engine: dict[str, list[ImageHit]] = {}
        pool = ThreadPoolExecutor(max_workers=min(16, max(1, len(queries) * len(groups))))
        futures = {
            pool.submit(self._searx, q, g, searx_budget - 0.5): (q, tuple(g))
            for q in queries for g in groups
        }
        try:
            for fut in as_completed(futures, timeout=searx_budget + 1.0):
                try:
                    hits, blocked = fut.result()
                except Exception as exc:  # noqa: BLE001 — SearXNG down: degrade
                    log.warning("searxng text discovery failed", extra={"error": str(exc)})
                    continue
                for h in hits:
                    per_engine.setdefault(h.engine, []).append(h)
                report.blocked.update(blocked)
        except TimeoutError:
            pass
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        # real-browser retry for blocked engines that have a results URL
        retry = [e for e in report.blocked if e in BROWSER_SEARCH_URLS and not per_engine.get(e)]
        retry = retry[: self.browser_fallback_max]
        remaining = budget_seconds - (time.monotonic() - started)
        if retry and remaining > 3 and self.proxify_fetch_url:
            pool = ThreadPoolExecutor(max_workers=len(retry))
            futures = {pool.submit(self._browser, e, queries[0]): e for e in retry}
            try:
                for fut in as_completed(futures, timeout=remaining):
                    engine = futures[fut]
                    report.browser_used.append(engine)
                    try:
                        hits = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("browser discovery failed", extra={"engine": engine, "error": str(exc)})
                        continue
                    if hits:
                        per_engine[engine] = hits
                        report.blocked.pop(engine, None)
            except TimeoutError:
                pass
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        seen: set[str] = set()
        lists = list(per_engine.values())
        for i in range(max((len(x) for x in lists), default=0)):
            for lst in lists:
                if i < len(lst) and lst[i].img_src not in seen:
                    seen.add(lst[i].img_src)
                    report.hits.append(lst[i])
        report.engines = {e: len(v) for e, v in per_engine.items()}
        for e in report.blocked:
            report.engines.setdefault(e, 0)
        return report
