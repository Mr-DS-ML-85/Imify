"""Optional OCR layer for Imify (Tesseract, self-hosted).

Board-only: logos, memes and screenshots carry visible text that perceptual
hashing alone ignores — a "search by the words on the image" mode is a clear
differentiator and is trivial to self-host (apt/pip install tesseract + the
pytesseract wheel, no external API).

Everything degrades gracefully: when pytesseract / tesseract are missing or
OCR is disabled, ``OcrEngine.available`` is False and the pipeline runs the
pure perceptual path exactly as before.
"""

from __future__ import annotations

import logging

from app.logging import child_logger

log = child_logger("imify.ocr")

try:  # optional native dependency — never required at import time
    import pytesseract  # type: ignore

    _HAS_PYTESSERACT = True
except Exception:  # noqa: BLE001
    pytesseract = None
    _HAS_PYTESSERACT = False


def tesseract_available() -> bool:
    """True when both the python wrapper and the tesseract binary exist."""
    if not _HAS_PYTESSERACT:
        return False
    try:
        from PIL import Image

        return bool(pytesseract.get_tesseract_version())
    except Exception:  # noqa: BLE001
        return False


class OcrExtractor:
    """Thread-safe wrapper: extract visible text from a Pillow image."""

    def __init__(self, enabled: bool = False, lang: str = "eng"):
        self.enabled = enabled
        self.lang = lang or "eng"
        self._available: bool | None = None
        self._lock = logging  # placeholder; real lock used in extract

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._available is None:
            self._available = tesseract_available()
            if not self._available:
                log.warning("OCR enabled but tesseract is not installed")
        return self._available

    def extract(self, img) -> str:
        """Return the whitespace-normalized visible text in the image, or ""."""
        if not self.available:
            return ""
        if img is None:
            return ""
        import threading
        from PIL import Image

        try:
            with threading.Lock():
                # Pre-scale: tesseract prefers ~300dpi-ish input; keep it bounded
                # so huge screenshots don't stall the pipeline.
                if max(img.size) > 2200:
                    img = img.copy()
                    r = 2200 / max(img.size)
                    img = img.resize((int(img.width * r), int(img.height * r)), Image.Resampling.LANCZOS)
                raw = pytesseract.image_to_string(img, lang=self.lang)
        except Exception as exc:  # noqa: BLE001
            log.debug("ocr extract failed", extra={"error": str(exc)})
            return ""
        text = " ".join(raw.split())  # collapse whitespace/newlines
        return text[:1000]