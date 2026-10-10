"""Optional AI layer for Imify — OpenAI-compatible by design.

Every self-hosted model server worth the name speaks the OpenAI wire protocol
(``/v1/chat/completions`` + ``/v1/embeddings``), so Imify talks ONE dialect:

- ``llama.cpp`` server (embedding engine on :8374) — ``http://127.0.0.1:8374/v1``
- NVIDIA NIM (hosted vision models) — ``https://integrate.api.nvidia.com/v1``
- Ollama (its ``/v1`` endpoints) — ``http://127.0.0.1:11434/v1``

A single ``AI_BASE_URL`` + optional ``AI_API_KEY`` powers both captioning
(vision chat completion) and by-description search (text embeddings). When no
base URL is configured the AI layer is a no-op and the pixel/hash pipeline
answers exactly as before — nothing in Imify REQUIRES AI.
"""

from __future__ import annotations

import base64
import time

import httpx

from app.logging import child_logger

log = child_logger("imify.ai")

_VISION_MODELS = {
    "llama3.2-vision",
    "llama-3.2",
    "llava",
    "minicpm-v",
    "bakllava",
    "moondream",
    "phi-3-vision",
    "vlm",
    "vision-instruct",
}
_TEXT_EMBED_MODELS = {
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed-large",
    "bge-m3",
    "embed",
    "nemoretriever",
}


class AiClient:
    """Best-effort OpenAI-compatible AI client (NVIDIA NIM / llama.cpp / Ollama).

    All methods return ``None`` (or empty output) when the server is down, the
    model is missing, or the call is disabled — never raise into the
    search/index pipeline.
    """

    def __init__(
        self,
        url: str = "",
        caption_model: str = "",
        embed_model: str = "",
        api_key: str = "",
        timeout_seconds: float = 45.0,
        caption_url: str = "",
    ):
        self.url = url.rstrip("/") if url else ""
        self.caption_url = (caption_url.rstrip("/") or self.url)
        # Normalize to the OpenAI prefix. "http://host:8374" and
        # "http://host:8374/v1" both become ".../v1"; "https://integrate.api.nvidia.com" stays.
        for attr in ("url", "caption_url"):
            base = getattr(self, attr)
            if base and not base.endswith("/v1"):
                setattr(self, attr, base + "/v1")
        self.caption_model = caption_model or "meta/llama-3.2-11b-vision-instruct"
        self.embed_model = embed_model or "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
        self.api_key = api_key
        self._timeout = httpx.Timeout(timeout_seconds, connect=4.0)
        # small in-process circuit breaker so a dead server never slows a search
        self._open_until = 0.0
        self._consecutive_failures = 0

    # ------------------------------------------------------------- reachability
    @property
    def enabled(self) -> bool:
        return bool(self.url)

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if time.time() < self._open_until:
            return False
        # available if EITHER endpoint responds (vision may be hosted, embed local)
        ok = False
        for attr in ("url", "caption_url"):
            base = getattr(self, attr)
            if not base:
                continue
            try:
                resp = self._get(attr, "/models", timeout=httpx.Timeout(6.0, connect=3.0))
                if resp.status_code == 200:
                    ok = True
                    break
            except Exception:  # noqa: BLE001
                continue
        if not ok:
            self._trip(5.0)
        else:
            self._note_ok()
        return ok

    # ---------------------------------------------------------------- plumbing
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, which: str, path: str, timeout: httpx.Timeout | None = None) -> httpx.Response:
        base = getattr(self, which)
        url = base + path
        if self.api_key:
            resp = httpx.get(url, headers=self._headers(), timeout=timeout or self._timeout)
        else:
            resp = httpx.get(url, timeout=timeout or self._timeout)
        resp.raise_for_status()
        return resp

    def _post(self, which: str, path: str, payload: dict) -> httpx.Response:
        base = getattr(self, which)
        resp = httpx.post(
            base + path,
            headers=self._headers(),
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp

    def _trip(self, seconds: float) -> None:
        self._open_until = time.time() + seconds
        self._consecutive_failures += 1

    def _note_ok(self) -> None:
        self._consecutive_failures = 0
        self._open_until = 0.0

    def _guard(self) -> bool:
        if not self.enabled:
            return False
        if time.time() < self._open_until:
            return False
        return True

    # ------------------------------------------------------------- captioning
    def caption(self, image_bytes: bytes, mime: str = "image/jpeg", max_len: int = 220) -> str | None:
        """Describe an image. Returns a short caption or None (server down /
        model missing / failure). The caption is stored on the index entry and
        shown as alt-text + used for text search."""
        if not self._guard():
            return None
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
        try:
            resp = self._post(
                "caption_url",
                "/chat/completions",
                {
                    "model": self.caption_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Describe this image in one short, factual sentence for a "
                                        "reverse-image-search index (subject, action, setting, "
                                        "visible text). No preamble."
                                    ),
                                },
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    "max_tokens": 120,
                    "temperature": 0.2,
                },
            )
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            if not text:
                return None
            self._note_ok()
            return text[:max_len]
        except Exception as exc:  # noqa: BLE001
            log.debug("caption failed", extra={"error": str(exc)})
            self._trip(3.0)
            return None

    # ------------------------------------------------------------- embeddings
    def embed(self, text: str | None = None, image_bytes: bytes | None = None) -> list[float] | None:
        """Map text and/or an image to a shared embedding. Returns a float
        vector, or None when unavailable. Used for by-description search."""
        if not self._guard():
            return None
        if text:
            inp = text
        elif image_bytes is not None:
            data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
            inp = [data_url]  # VLM-embed models accept image data URLs
        else:
            return None
        try:
            resp = self._post("url", "/embeddings", {"model": self.embed_model, "input": inp})
            data = resp.json()
            vec = (data.get("data") or [{}])[0].get("embedding")
            if not vec:
                return None
            self._note_ok()
            return [float(x) for x in vec]
        except Exception as exc:  # noqa: BLE001
            log.debug("embed failed", extra={"error": str(exc)})
            self._trip(3.0)
            return None
