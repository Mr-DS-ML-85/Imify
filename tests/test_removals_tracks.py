"""Index removals + background trackers (auto-update) tests."""

from __future__ import annotations

import time

import pytest

from conftest import image_features, index_image, make_image_bytes, patch_image_fetch

from app.models import ImageHit


# ------------------------------------------------------------------ removals
def test_remove_image_removes_from_index(client, ctx):
    repo = ctx["repo"]
    raw_a = make_image_bytes(color=(200, 40, 40))
    raw_b = make_image_bytes(color=(40, 200, 40))
    id_a = index_image(repo, ctx["settings"], raw_a, url="https://a.com/one.jpg")
    id_b = index_image(repo, ctx["settings"], raw_b, url="https://b.com/two.jpg")
    assert repo.image_count() == 2

    assert repo.remove_image(id_a) is True
    assert repo.image_count() == 1
    assert repo.get_image(id_a) is None
    assert repo.get_image(id_b) is not None

    # no longer searchable
    hits = client.post(
        "/v1/search",
        files={"file": ("q.jpg", raw_a, "image/jpeg")},
        data={"mode": "best", "include_web": "false"},
    ).json()
    assert hits["total"] == 0
    assert hits["found"] is False


def test_remove_image_vdels_vector(client, ctx):
    """Removal calls VDEL so the ANN graph frees the node (no dead vectors)."""
    repo = ctx["repo"]
    id_a = index_image(repo, ctx["settings"], make_image_bytes(color=(200, 40, 40)), url="https://a.com/one.jpg")
    id_b = index_image(repo, ctx["settings"], make_image_bytes(color=(40, 200, 40)), url="https://b.com/two.jpg")
    assert repo.remove_image(id_a) is True
    # the remaining image still searches and returns only itself (the removed
    # vector is gone from the graph, not a dead node)
    hits = client.post(
        "/v1/search",
        files={"file": ("q.jpg", make_image_bytes(color=(40, 200, 40)), "image/jpeg")},
        data={"mode": "best", "include_web": "false"},
    ).json()
    assert hits["found"] is True
    assert all(r["image_id"] == id_b for r in hits["results"])


def test_purge_images_by_url_and_query(ctx):
    repo = ctx["repo"]
    index_image(repo, ctx["settings"], make_image_bytes(color=(1, 2, 3)), url="https://foo.com/a.jpg", query="cats")
    index_image(repo, ctx["settings"], make_image_bytes(color=(4, 5, 6)), url="https://foo.com/b.jpg", query="dogs")
    index_image(repo, ctx["settings"], make_image_bytes(color=(7, 8, 9)), url="https://bar.org/c.jpg", query="cats")

    assert repo.purge_images(url_contains="foo.com") == 2
    assert repo.image_count() == 1
    assert repo.purge_images(query="cats") == 1
    assert repo.image_count() == 0


def test_admin_remove_and_purge_endpoints(client, ctx):
    headers = {"Authorization": "Bearer sk_test_admin"}
    id_a = index_image(ctx["repo"], ctx["settings"], make_image_bytes(color=(11, 12, 13)), url="https://purge.me/x.jpg")

    # list
    listing = client.get("/v1/admin/images", headers=headers).json()
    assert listing["total"] >= 1
    assert any(m["id"] == id_a for m in listing["images"])

    # delete one
    r = client.delete(f"/v1/admin/images/{id_a}", headers=headers)
    assert r.status_code == 200
    assert r.json()["image_id"] == id_a
    assert client.get("/v1/admin/images", headers=headers).json()["total"] == 0

    # purge endpoint
    index_image(ctx["repo"], ctx["settings"], make_image_bytes(color=(21, 22, 23)), url="https://purge.me/y.jpg")
    r = client.post("/v1/admin/images/purge", json={"url_contains": "purge.me"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["removed"] == 1


def test_admin_images_requires_key(client):
    assert client.get("/v1/admin/images").status_code == 401
    assert client.get("/v1/admin/tracks").status_code == 401


# ------------------------------------------------------------------- trackers
def test_tracker_crud(ctx):
    repo = ctx["repo"]
    track_id = repo.upsert_track("abc123", 0xDEADBEEF, public_url="https://img.example.com/q.jpg")
    # upsert by phash is idempotent and refreshes public_url
    again = repo.upsert_track("abc123", 0xDEADBEEF, public_url="https://img.example.com/new.jpg")
    assert again == track_id

    tracks = repo.list_tracks()
    assert len(tracks) == 1
    assert tracks[0]["public_url"] == "https://img.example.com/new.jpg"
    assert tracks[0]["enabled"] is True

    repo.set_track_enabled(track_id, False)
    assert repo.get_track(track_id)["enabled"] is False

    repo.touch_track(track_id, hits=3, note="3 hits, 2 new")
    t = repo.get_track(track_id)
    assert t["hits"] == 3
    assert t["last_run"] > 0

    assert repo.delete_track(track_id) is True
    assert repo.list_tracks() == []


def test_tracker_round_enqueues_new_matches_and_respects_interval(ctx, monkeypatch):
    repo, searx, indexer = ctx["repo"], ctx["searx"], ctx["indexer"]
    ctx["settings"].tracker_interval_seconds = 0  # always due
    raw = make_image_bytes()
    feats = image_features(raw)
    repo.upsert_track(
        "sha1", int(feats["phash"]), public_url="https://img.example.com/q.jpg",
        dhash=int(feats["dhash"]), ahash=int(feats["ahash"]),
    )
    searx.reverse_hits = [
        ImageHit(title="match a", url="https://page1.com/a", img_src="https://img1.com/a.jpg", engine="bing images"),
        ImageHit(title="match b", url="https://page2.com/b", img_src="https://img2.com/b.jpg", engine="bing images"),
    ]
    # candidates hash-match the tracked image -> the matcher accepts them
    patch_image_fetch(monkeypatch, raw)

    processed = indexer.run_tracker_round()
    assert processed == 1
    t = repo.list_tracks()[0]
    assert t["hits"] == 2
    assert "2/2 verified" in t["note"]
    assert repo.pending_count() == 2

    # interval respected: not due again
    ctx["settings"].tracker_interval_seconds = 3600
    processed = indexer.run_tracker_round()
    assert processed == 0


def test_tracker_rejects_non_matching_candidates(ctx, monkeypatch):
    """The tracker NEVER ingests an unrelated image: candidates that don't
    hash-match the tracked image are rejected (0/2 verified)."""
    repo, searx, indexer = ctx["repo"], ctx["searx"], ctx["indexer"]
    ctx["settings"].tracker_interval_seconds = 0
    raw = make_image_bytes()
    feats = image_features(raw)
    repo.upsert_track(
        "shaR", int(feats["phash"]), public_url="https://img.example.com/q.jpg",
        dhash=int(feats["dhash"]), ahash=int(feats["ahash"]),
    )
    searx.reverse_hits = [
        ImageHit(title="a", url="https://p1.com/a", img_src="https://i1.com/a.jpg", engine="bing"),
        ImageHit(title="b", url="https://p2.com/b", img_src="https://i2.com/b.jpg", engine="bing"),
    ]
    # candidates are a COMPLETELY different image
    patch_image_fetch(monkeypatch, make_image_bytes(color=(9, 250, 250), size=(90, 90)))

    indexer.run_tracker_round()
    t = repo.list_tracks()[0]
    assert t["hits"] == 0
    assert "0/2 verified" in t["note"]
    assert repo.pending_count() == 0


def test_tracker_falls_back_to_google_when_searxng_empty(ctx, monkeypatch):
    repo, searx, google, indexer = ctx["repo"], ctx["searx"], ctx["google"], ctx["indexer"]
    ctx["settings"].tracker_interval_seconds = 0
    ctx["settings"].reverse_providers = "searxng,google"
    raw = make_image_bytes()
    feats = image_features(raw)
    repo.upsert_track(
        "sha2", int(feats["phash"]), public_url="https://img.example.com/q.jpg",
        dhash=int(feats["dhash"]), ahash=int(feats["ahash"]),
    )
    searx.reverse_hits = []  # searxng finds nothing
    google.hits = [ImageHit(title="g1", url="https://g.com/x", img_src="https://g.com/x.jpg", engine="google images")]
    patch_image_fetch(monkeypatch, raw)

    indexer.run_tracker_round()
    t = repo.list_tracks()[0]
    assert t["hits"] == 1
    assert repo.pending_count() == 1


def test_tracker_degraded_when_no_public_url(ctx):
    repo, indexer = ctx["repo"], ctx["indexer"]
    ctx["settings"].tracker_interval_seconds = 0
    repo.upsert_track("sha3", 0x3333)  # no public_url, no local upload
    indexer.run_tracker_round()
    t = repo.list_tracks()[0]
    assert "no_public_url" in t["note"]
    assert repo.pending_count() == 0


def test_tracker_byte_uploads_when_no_public_url(ctx, tmp_path, monkeypatch):
    """No public URL + a saved local upload -> google-upload the raw bytes."""
    repo, google, indexer = ctx["repo"], ctx["google"], ctx["indexer"]
    ctx["settings"].tracker_interval_seconds = 0
    ctx["settings"].google_upload_enabled = True
    raw = make_image_bytes()
    feats = image_features(raw)
    local = tmp_path / "uploads" / "abc.jpg"
    local.parent.mkdir(parents=True)
    local.write_bytes(raw)
    repo.upsert_track(
        "shaX", int(feats["phash"]), public_url="", local_path=str(local),
        dhash=int(feats["dhash"]), ahash=int(feats["ahash"]),
    )
    google.upload_hits = [
        ImageHit(title="u", url="https://u.com/x", img_src="https://u.com/x.jpg", engine="google images")
    ]
    patch_image_fetch(monkeypatch, raw)
    indexer.run_tracker_round()
    t = repo.list_tracks()[0]
    assert t["hits"] == 1
    assert repo.pending_count() == 1
    assert google.uploads[0][0] == raw
    assert repo.provider_stats()["google-upload"]["ok"] == 1


def test_search_no_public_url_uses_google_upload(client, ctx, monkeypatch):
    """New image with no public URL -> raw bytes uploaded to Google via tunnel."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = ""
    ctx["google"].upload_hits = [
        ImageHit(title="found it", url="https://found.com/p", img_src="https://found.com/p.jpg", engine="google images")
    ]
    raw = make_image_bytes()
    # the candidate must hash-match the query to be reported as found
    patch_image_fetch(monkeypatch, raw)
    resp = client.post(
        "/v1/search",
        files={"file": ("brand-new.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    )
    data = resp.json()
    assert data["fallback"] == "web"
    assert "web" in data["providers"]
    assert data["results"][0]["match_type"] in ("exact", "near")
    assert ctx["google"].uploads and ctx["google"].uploads[0][0] == raw
    assert ctx["repo"].provider_stats()["google-upload"]["ok"] == 1
    # on-the-fly default: the verified hit is returned but NOT persisted.
    # The upload itself IS indexed because it was hash-verified online (only
    # failed searches leave no trace in the index).
    assert ctx["repo"].pending_count() == 1


def test_google_upload_degrades_gracefully(client, ctx):
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = ""
    ctx["google"].fail_upload = True
    raw = make_image_bytes()
    data = client.post(
        "/v1/search",
        files={"file": ("new.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    ).json()
    assert "provider:google-upload" in data["degraded"]
    assert data["found"] is False
    assert data["fallback"] == "none"
    assert ctx["repo"].provider_stats()["google-upload"]["fail"] == 1


# ------------------------------------------------- on-the-fly mode + storage
def test_search_learn_false_does_not_persist_web_hits(client, ctx, monkeypatch):
    """Default on-the-fly mode: verified web matches are RETURNED but NOT
    stored — storage stays bounded (the whole point of on-the-fly matching)."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="Elon Musk", url="https://page.example/x", img_src="https://img.example/musk.jpg", engine="bing")
    ]
    patch_image_fetch(monkeypatch, make_image_bytes())  # candidate hash-matches
    d = client.post(
        "/v1/search",
        files={"file": ("q.jpg", make_image_bytes(), "image/jpeg")},
        data={"mode": "best", "num_results": "10", "include_web": "true", "learn": "false"},
    ).json()
    assert d["found"] is True
    assert d["results"][0]["source"] == "web"
    # upload itself is auto-tracked (1 pending) but the web hit is NOT learned
    assert ctx["repo"].pending_count() == 1


def test_search_learn_true_persists_web_hits(client, ctx, monkeypatch):
    """learn=true: verified matches are enqueued into the own index."""
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
    assert ctx["repo"].pending_count() == 2  # upload + learned web hit


def test_lru_eviction_caps_index(ctx):
    """The storage guard: when the index exceeds the cap the oldest image is
    evicted, so the DB can never fill up."""
    repo = ctx["repo"]
    ctx["settings"].max_index_images = 3
    for i in range(5):
        index_image(repo, ctx["settings"], make_image_bytes(color=(i * 10, 30, 200)), url=f"https://e.com/{i}.jpg")
    assert repo.image_count() == 5  # eviction is explicit, not on-insert
    evicted = repo.evict_if_full(3)
    assert evicted == 2
    assert repo.image_count() == 3
    # oldest first_seen entries were evicted
    metas = sorted(repo.all_images(), key=lambda m: m.first_seen)
    assert metas[0].url == "https://e.com/2.jpg"  # 0 and 1 evicted
    # evicting again when under cap is a no-op
    assert repo.evict_if_full(3) == 0


# ------------------------------------------------------------ success rates
def test_success_rate_counters(ctx):
    repo = ctx["repo"]
    repo.record_search(True)
    repo.record_search(False)
    repo.record_search(True)
    rate = repo.search_rate()
    assert rate["total"] == 3
    assert rate["found"] == 2
    assert rate["not_found"] == 1
    assert rate["rate"] == pytest.approx(2 / 3)

    repo.record_provider("searxng", True)
    repo.record_provider("searxng", False)
    ps = repo.provider_stats()["searxng"]
    assert ps["ok"] == 1 and ps["fail"] == 1
    assert ps["rate"] == pytest.approx(0.5)


def test_stats_endpoint_exposes_success(client, ctx):
    ctx["repo"].record_search(True)
    ctx["repo"].record_provider("google", False)
    s = client.get("/v1/stats").json()
    assert "success" in s
    assert s["success"]["searches"]["total"] >= 1
    assert s["success"]["providers"]["google"]["fail"] == 1
    assert "imify_search_found_total" in client.get("/metrics").text
    assert 'imify_provider_fail_total{provider="google"} 1' in client.get("/metrics").text


# ------------------------------------------------------------- API auto-track
def test_search_upload_auto_tracks_new_image(client, ctx):
    """A NEW upload always gets a bg tracker (auto-update) — but a FAILED
    search must never grow the searchable index with the failed image. The
    upload itself is only enqueued when it is hash-verified online; otherwise
    nothing is indexed (on-the-fly matching keeps storage bounded)."""
    raw = make_image_bytes(color=(31, 32, 33))
    resp = client.post(
        "/v1/search",
        files={"file": ("new-img.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "false"},
    )
    assert resp.status_code == 200
    assert resp.json()["found"] is False

    # the bg tracker IS created (keeps hunting the internet over time)...
    tracks = ctx["repo"].list_tracks()
    assert len(tracks) == 1
    # ...but the failed image is NOT enqueued into the searchable index
    # (the old behaviour indexed every upload, so the next search of the same
    # image returned the upload's own local path instead of remote links)
    assert ctx["repo"].pending_count() == 0

    # searching the same image again keeps a single tracker
    client.post(
        "/v1/search",
        files={"file": ("new-img.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "false"},
    )
    assert len(ctx["repo"].list_tracks()) == 1


def test_search_verified_upload_enters_index_and_tracks(client, ctx, monkeypatch):
    """When the search VERIFIES the upload online it becomes a local index
    entry (TinEye: verified images are stored, failed queries are not) and
    gets the bg tracker."""
    ctx["settings"].web_reverse_enabled = True
    ctx["settings"].public_base_url = "http://public.example"
    ctx["searx"].reverse_hits = [
        ImageHit(title="match", url="https://page.example/x", img_src="https://img.example/m.jpg", engine="bing")
    ]
    raw = make_image_bytes(color=(51, 52, 53))
    patch_image_fetch(monkeypatch, raw)  # candidate hash-matches the upload
    d = client.post(
        "/v1/search",
        files={"file": ("verified.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "true"},
    ).json()
    assert d["found"] is True
    # verified -> upload enqueued (1) + bg tracker created
    assert ctx["repo"].pending_count() == 1
    assert len(ctx["repo"].list_tracks()) == 1
    pending = ctx["repo"].db.get_json("pending:" + ctx["repo"].db.keys("pending:")[0][len("pending:"):])
    assert pending["url"].startswith("http://testserver/uploads/")
    assert pending["engine"] == "upload"


def test_search_already_indexed_image_does_not_retrack(client, ctx):
    raw = make_image_bytes(color=(41, 42, 43))
    index_image(ctx["repo"], ctx["settings"], raw, url="https://known.com/x.jpg")
    client.post(
        "/v1/search",
        files={"file": ("known.jpg", raw, "image/jpeg")},
        data={"mode": "best", "include_web": "false"},
    )
    assert ctx["repo"].list_tracks() == []


def test_admin_tracks_endpoints(client, ctx):
    headers = {"Authorization": "Bearer sk_test_admin"}
    repo = ctx["repo"]
    track_id = repo.upsert_track("sha4", 0x4444, public_url="https://img.example.com/q.jpg")

    tracks = client.get("/v1/admin/tracks", headers=headers).json()["tracks"]
    assert any(t["id"] == track_id for t in tracks)

    r = client.patch(f"/v1/admin/tracks/{track_id}", json={"enabled": False}, headers=headers)
    assert r.status_code == 200
    assert repo.get_track(track_id)["enabled"] is False

    assert client.post("/v1/admin/tracks/run", json={}, headers=headers).status_code == 200

    r = client.delete(f"/v1/admin/tracks/{track_id}", headers=headers)
    assert r.status_code == 200
    assert repo.list_tracks() == []
