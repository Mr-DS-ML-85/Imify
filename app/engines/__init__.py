"""Live web reverse-image-search providers, tried in order.

- ``searxng`` — your own SearXNG instance (``image_url`` param; engines like
  Bing Images answer — when it works; some builds reject empty ``q`` and
  degrade).
- ``bing`` — Bing Visual Search **byte-upload** (works for images with no
  public URL) or the ``imgurl:`` URL trick, via proxify. Reliable from this
  box; candidates are still hash-verified before being reported.
- ``google`` — Google Images/Lens through your proxify gateway (real browser
  session + TLS impersonation), used when SearXNG/Bing fail or return nothing.

A failing provider degrades the request (reported in ``degraded``) instead of
failing it — SearchOS middleware philosophy. And per the TinEye model: every
engine is only a **candidate finder**; the fused-hash matcher judges whether
each candidate actually matches the query (``Matcher.verify_web_hits``).
"""

from __future__ import annotations

from typing import Callable

from app.logging import child_logger
from app.models import ImageHit

log = child_logger("imify.engines")


class EngineError(RuntimeError):
    pass


def run_reverse_chain(
    image_url: str,
    providers: list[str],
    searx_reverse: Callable[[str], list[ImageHit]],
    google_reverse: Callable[[str], list[ImageHit]] | None,
    bing_reverse: Callable[[str], list[ImageHit]] | None = None,
    yandex_reverse: Callable[[str], list[ImageHit]] | None = None,
    parallel: bool = True,
    budget_seconds: float | None = None,
) -> tuple[list[ImageHit], list[str]]:
    """Query every configured reverse provider and merge ALL their candidates.

    TinEye model: no single engine sees the whole web, so every provider is
    asked and the candidates are merged (deduped by image URL) before the
    fused-hash matcher judges them. Providers run in PARALLEL (bounded) when
    ``parallel`` is set so the slowest provider — not the sum of all four —
    sets the wall-clock time; ``budget_seconds`` hard-caps the whole phase.

    Returns (hits, degraded): hits are the merged candidate list; degraded
    lists every provider that raised or returned nothing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import time

    start = time.monotonic()
    tasks: list[tuple[str, Callable[[str], list[ImageHit]]]] = []
    for provider in providers:
        if provider == "searxng":
            tasks.append((provider, searx_reverse))
        elif provider == "bing" and bing_reverse:
            tasks.append((provider, bing_reverse))
        elif provider == "yandex" and yandex_reverse:
            tasks.append((provider, yandex_reverse))
        elif provider == "google" and google_reverse:
            tasks.append((provider, google_reverse))
        else:
            log.warning("reverse provider not registered", extra={"event": "reverse_unregistered", "provider": provider})
            continue

    if not tasks:
        return [], []

    def _run(provider: str, fn: Callable[[str], list[ImageHit]]) -> tuple[str, list[ImageHit], Exception | None]:
        try:
            return provider, fn(image_url), None
        except Exception as exc:  # noqa: BLE001 — degrade, never fail the search
            return provider, [], exc

    per_task_hits: dict[str, list[ImageHit]] = {}
    failures: set[str] = set()

    if parallel and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(_run, p, fn): p for p, fn in tasks}
            for fut in as_completed(futures):
                if budget_seconds is not None and time.monotonic() - start >= budget_seconds:
                    # budget exhausted — stop waiting; already-finished tasks
                    # below were collected as they completed, late ones degrade
                    for p in futures.values():
                        if p not in per_task_hits and p not in failures:
                            failures.add(p)
                    break
                provider, hits, exc = fut.result()
                if exc is not None:
                    failures.add(provider)
                    log.warning(
                        "reverse provider failed",
                        extra={"event": "reverse_failed", "provider": provider, "error": str(exc)},
                    )
                else:
                    per_task_hits[provider] = hits
    else:
        for provider, fn in tasks:
            if budget_seconds is not None and time.monotonic() - start >= budget_seconds:
                failures.add(provider)
                continue
            provider, hits, exc = _run(provider, fn)
            if exc is not None:
                failures.add(provider)
            else:
                per_task_hits[provider] = hits

    degraded = [f"provider:{p}" for p in providers if p in failures]

    # merge all hits, deduped by candidate image URL, preserving provider order
    merged: list[ImageHit] = []
    seen: set[str] = set()
    for provider in providers:
        for hit in per_task_hits.get(provider, []):
            if hit.img_src in seen:
                continue
            seen.add(hit.img_src)
            merged.append(hit)
    return merged, degraded
