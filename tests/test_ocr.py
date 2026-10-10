from __future__ import annotations


def test_ocr_extractor_disabled_without_enabled_flag():
    from app.ocr import OcrExtractor

    o = OcrExtractor(enabled=False)
    assert o.available is False
    assert o.extract(None) == ""


def test_ocr_extractor_graceful_without_tesseract():
    """When tesseract is missing, OCR reports unavailable and extract is safe."""
    from app.ocr import OcrExtractor, tesseract_available

    o = OcrExtractor(enabled=True)
    if tesseract_available():
        assert o.available is True
    else:
        assert o.available is False
    # extract on any image is a no-op when unavailable (never crashes)
    from conftest import make_image_bytes
    from app.features import decode_image

    img = decode_image(make_image_bytes())
    assert isinstance(o.extract(img), str)


def test_ocr_search_endpoint_degrades_without_ocr(client, ctx):
    """With OCR disabled the endpoint answers 503 instead of failing."""
    ctx["settings"].ocr_enabled = False
    r = client.post("/v1/search/ocr", data={"query": "hello"})
    assert r.status_code == 503
    d = r.json()
    assert "OCR unavailable" in (d.get("detail") or "")