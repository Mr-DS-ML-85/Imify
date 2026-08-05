from __future__ import annotations

import time

from conftest import index_image, make_image_bytes


def test_health_endpoints(client):
    assert client.get("/health/live").json()["status"] == "ok"
    ready = client.get("/health/ready").json()
    assert ready["status"] in ("ready", "degraded")


def test_search_by_upload_finds_indexed_images(client, ctx):
    raw = make_image_bytes()
    index_image(ctx["repo"], ctx["settings"], raw, url="https://a.com/orig.jpg", title="Tower")
    resp = client.post(
        "/v1/search",
        files={"file": ("query.jpg", raw, "image/jpeg")},
        data={"mode": "best", "num_results": "20", "include_web": "false"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "local" in data["providers"]
    assert data["total"] >= 1
    assert data["took_ms"] >= 0
    top = data["results"][0]
    assert top["match_type"] in ("exact", "near")
    assert top["thumb"].startswith("/img/")


def test_search_rejects_garbage(client):
    resp = client.post(
        "/v1/search",
        files={"file": ("bad.txt", b"definitely not an image", "text/plain")},
        data={"mode": "best"},
    )
    assert resp.status_code in (400, 422)


def test_search_requires_input(client):
    resp = client.post("/v1/search", data={"mode": "best"})
    assert resp.status_code == 400


def test_search_by_image_id(client, ctx):
    raw = make_image_bytes()
    image_id = index_image(ctx["repo"], ctx["settings"], raw, url="https://a.com/x.jpg")
    resp = client.post(
        "/v1/search",
        data={"image_id": str(image_id), "mode": "best"},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_image_detail_and_thumbnail(client, ctx):
    raw = make_image_bytes()
    image_id = index_image(ctx["repo"], ctx["settings"], raw, url="https://a.com/x.jpg")
    detail = client.get(f"/v1/image/{image_id}")
    assert detail.status_code == 200
    assert detail.json()["url"] == "https://a.com/x.jpg"
    assert client.get("/v1/image/999999").status_code == 404


def test_admin_requires_key(client):
    assert client.get("/v1/admin/queries").status_code == 401


def test_admin_ingest_and_queries(client):
    headers = {"Authorization": "Bearer sk_test_admin"}
    r = client.post("/v1/admin/ingest", json={"queries": ["aurora wallpaper", {"query": "macro butterfly", "enabled": False}]}, headers=headers)
    assert r.status_code == 200
    assert r.json()["count"] == 2

    queries = client.get("/v1/admin/queries", headers=headers).json()["queries"]
    assert len(queries) == 2
    assert queries[0]["query"] == "aurora wallpaper"
    assert queries[1]["enabled"] is False

    # toggle
    qid = queries[1]["id"]
    client.patch(f"/v1/admin/queries/{qid}", json={"enabled": True}, headers=headers)
    assert client.get("/v1/admin/queries", headers=headers).json()["queries"][1]["enabled"] is True

    # delete
    client.delete(f"/v1/admin/queries/{qid}", headers=headers)
    assert len(client.get("/v1/admin/queries", headers=headers).json()["queries"]) == 1


def test_admin_keys_lifecycle(client):
    headers = {"Authorization": "Bearer sk_test_admin"}
    created = client.post("/v1/admin/keys", json={"name": "ci-bot", "role": "user"}, headers=headers)
    assert created.status_code == 200
    raw_key = created.json()["api_key"]
    assert raw_key.startswith("sk_")

    listed = client.get("/v1/admin/keys", headers=headers).json()["keys"]
    assert any(k["name"] == "ci-bot" for k in listed)
    # stored hashed — raw key must not appear
    assert all(raw_key not in str(k) for k in listed)

    # the user key works for search
    resp = client.post("/v1/search", data={"image_url": ""}, headers={"Authorization": f"Bearer {raw_key}"})
    assert resp.status_code in (400,)  # reaches the endpoint (input validation), auth passed

    client.delete("/v1/admin/keys/ci-bot", headers=headers)
    assert all(k["name"] != "ci-bot" for k in client.get("/v1/admin/keys", headers=headers).json()["keys"])


def test_wrong_admin_key_rejected(client):
    assert client.get("/v1/admin/queries", headers={"Authorization": "Bearer sk_wrong"}).status_code in (401, 403)


def test_stats_and_metrics(client, ctx):
    index_image(ctx["repo"], ctx["settings"], make_image_bytes(), url="https://a.com/s.jpg")
    stats = client.get("/v1/stats").json()
    assert stats["images"] == 1
    assert stats["storage_backend"] == "sqlite"
    assert "imify_images 1" in client.get("/metrics").text


def test_rate_limit(ctx):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import build_api
    from app.auth import RateLimiter

    app = FastAPI()
    app.include_router(
        build_api(
            ctx["settings"], ctx["repo"], ctx["matcher"], ctx["indexer"], ctx["auth"],
            ctx["searx"], RateLimiter(per_minute=2, per_hour=100, per_day=1000),
        )
    )
    c = TestClient(app)
    assert c.post("/v1/search", data={"mode": "best"}).status_code == 400  # no input, but counted
    assert c.post("/v1/search", data={"mode": "best"}).status_code == 400
    third = c.post("/v1/search", data={"mode": "best"})
    assert third.status_code == 429
    assert "Retry-After" in third.headers
