"""Tests for the SearchOS-style fallback ladder:

1. local found  -> serve local only (SearXNG never called)
2. local empty  -> call SearXNG live reverse (fallback=web)
3. SearXNG down -> degrade, local only, found=false
4. nothing      -> "search not found"
plus the atomic claim-lease concurrency guarantee.
"""

from __future__ import annotations

import threading

from app.models import ImageHit

from conftest import index_image, make_image_bytes, patch_image_fetch


def _search(client, mode="best", include_web="true", file_name="q.jpg"):
    return client.post(
        "/v1/search",
        files={"file": (file_name, make_image_bytes(), "image/jpeg")},
        data={"mode": mode, "num_results": "10", "include_web": include_web},
    ).json()


def test_local_found_skips_web(client, ctx):
    """SearchOS: 'if found in local -> serve local', never hit SearXNG."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    raw = make_image_bytes()
    index_image(ctx["repo"], ctx["settings"], raw, url="https://a.com/orig.jpg")
    d = _search(client)
    assert d["found"] is True
    assert d["fallback"] == "local"
    assert "web" not in d["providers"]
    assert ctx["searx"].total_requests == 0  # web layer never consulted
    assert all(r["source"] == "local" for r in d["results"])


def test_local_empty_calls_web(client, ctx, monkeypatch):
    """SearchOS: 'if not found -> call SearXNG for urls'."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="Elon Musk", url="https://page.example/x", img_src="https://img.example/musk.jpg", engine="bing")
    ]
    # the candidate download returns the same pixels as the query -> the
    # fused-hash matcher VERIFIES it (TinEye verdict: engines find, hash judges)
    patch_image_fetch(monkeypatch, make_image_bytes())
    d = _search(client)
    assert ctx["searx"].total_requests == 1
    assert d["fallback"] == "web"
    assert "web" in d["providers"]
    assert d["found"] is True
    assert d["results"][0]["source"] == "web"
    assert d["results"][0]["match_type"] in ("exact", "near")  # hash-verified, not blind
    # on-the-fly default: the verified web hit is RETURNED but NOT persisted;
    # only the upload itself was enqueued by the auto-tracker
    assert ctx["repo"].pending_count() == 1


def test_local_web_learn_true_persists_hit(client, ctx, monkeypatch):
    """learn=true opts into persistence: verified web matches join the index."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="Elon Musk", url="https://page.example/x", img_src="https://img.example/musk.jpg", engine="bing")
    ]
    patch_image_fetch(monkeypatch, make_image_bytes())
    d = client.post(
        "/v1/search",
        files={"file": ("q.jpg", make_image_bytes(), "image/jpeg")},
        data={"mode": "best", "num_results": "10", "include_web": "true", "learn": "true"},
    ).json()
    assert d["found"] is True
    # upload (auto-track) + verified web hit (learned)
    assert ctx["repo"].pending_count() == 2


def test_web_candidate_not_matching_is_flagged_not_found(client, ctx, monkeypatch):
    """Engines returned a URL but the image does NOT hash-match the query ->
    it is listed as 'unverified' and the search honestly reports not-found
    (never a blind score-0.5 match that would mislead a removal request)."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="unrelated", url="https://page.example/y", img_src="https://img.example/other.jpg", engine="bing")
    ]
    # candidate is a COMPLETELY different image -> fused distance far beyond threshold
    patch_image_fetch(monkeypatch, make_image_bytes(color=(5, 200, 250), size=(96, 96)))
    d = _search(client)
    assert d["found"] is False
    assert d["fallback"] == "none"
    assert "web_candidates_unverified" in d["degraded"]
    assert "web" not in d["providers"]
    # the candidate is still listed, clearly flagged, never as a confirmed match
    flagged = [r for r in d["results"] if r["match_type"] == "unverified"]
    assert flagged and flagged[0]["url"] == "https://img.example/other.jpg"
    # a failed search leaves NOTHING in the index — not the unverified web
    # candidate AND not the upload itself (no failed-image pollution: the old
    # behaviour indexed every upload, so the next search returned the upload's
    # local path instead of remote links)
    assert ctx["repo"].pending_count() == 0


def test_searxng_down_degrades_to_local_only(client, ctx):
    """SearchOS: 'if SearXNG is down -> scan local only'."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].fail_reverse = True
    d = _search(client)
    assert "provider:searxng" in d["degraded"]
    assert "web" not in d["providers"]
    assert d["found"] is False
    assert d["fallback"] == "none"


def test_search_not_found(client):
    """Nothing in the index, web disabled -> explicit 'search not found'."""
    d = _search(client, include_web="false")
    assert d["found"] is False
    assert d["fallback"] == "none"
    assert d["total"] == 0
    assert d["results"] == []


def test_web_needs_public_url_flag(client, ctx):
    """Private-box uploads have no public URL -> byte-upload path is used;
    when Google blocks it the response degrades honestly instead of failing."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = ""
    ctx["google"].fail_upload = True  # proxify tunnel not answering Google
    d = _search(client)
    assert "provider:google-upload" in d["degraded"]
    assert d["found"] is False
    assert d["fallback"] == "none"


def test_claim_pending_is_atomic(ctx):
    """Concurrent workers must never claim the same pending task twice."""
    repo = ctx["repo"]
    repo.enqueue_pending("https://x/1.jpg", {"title": "task1"})
    repo.enqueue_pending("https://x/2.jpg", {"title": "task2"})

    barrier = threading.Barrier(4)
    claimed: list[dict] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        got = repo.claim_pending(1)
        with lock:
            claimed.extend(got)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    urls = {c["data"]["url"] for c in claimed}
    assert len(claimed) == 2, f"exactly the 2 tasks should be claimed, got {len(claimed)}: {urls}"
    assert urls == {"https://x/1.jpg", "https://x/2.jpg"}


def test_finish_pending_retries_then_drops(ctx):
    repo = ctx["repo"]
    repo.enqueue_pending("https://x/1.jpg", {"url": "https://x/1.jpg", "attempts": 0})
    item = repo.claim_pending(5)[0]
    assert item["data"]["url"] == "https://x/1.jpg"
    # fail -> kept for retry
    repo.finish_pending(item, done=False)
    assert repo.pending_count() == 1
    assert repo.db.get_json("pending:" + item["key"][len("pending:"):])["attempts"] == 1
    # claim again -> same task, attempts incremented
    item2 = repo.claim_pending(5)[0]
    assert item2["data"]["attempts"] == 1
    # done -> dropped
    repo.finish_pending(item2, done=True)
    assert repo.pending_count() == 0
    # lease released so a future claim of nothing is empty
    assert repo.claim_pending(5) == []
