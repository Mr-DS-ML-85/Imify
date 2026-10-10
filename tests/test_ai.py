from __future__ import annotations


def test_ai_client_disabled_without_url():
    from app.ai import AiClient

    ai = AiClient(url="", caption_model="meta/llama-3.2-11b-vision-instruct")
    assert ai.enabled is False
    assert ai.available is False
    assert ai.caption(b"fake-image") is None
    assert ai.embed(text="hello") is None


def test_ai_client_not_available_when_server_down(monkeypatch):
    from app.ai import AiClient
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("httpx.get", boom)
    ai = AiClient(url="http://127.0.0.1:1", caption_model="meta/llama-3.2-11b-vision-instruct")
    assert ai.enabled is True
    assert ai.available is False
    # circuit breaker trips; caption/embed no-op
    assert ai.caption(b"x") is None
    assert ai.embed(text="x") is None


def test_cosine_similarity():
    from app.api import _cosine

    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert _cosine([], [1.0]) == 0.0
    assert abs(_cosine([1.0, 2.0], [2.0, 4.0]) - 1.0) < 1e-6  # parallel vectors