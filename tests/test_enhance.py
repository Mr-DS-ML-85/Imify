"""Enhancer: Real-ESRGAN only for low-resolution queries, tiled (bounded memory)."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.enhance import Enhancer


class _FakeX4Session:
    """Stands in for Real-ESRGAN: nearest-neighbour x4 of a (N,C,W,H) input."""

    def __init__(self):
        self.shapes = []

    def get_inputs(self):
        return [SimpleNamespace(name="data", shape=["batch_size", 3, "width", "height"])]

    def get_outputs(self):
        return [SimpleNamespace(name="out", shape=["batch_size", 3, "width", "height"])]

    def run(self, _names, feeds):
        x = feeds["data"]
        self.shapes.append(x.shape)
        return [x.repeat(4, axis=2).repeat(4, axis=3)]


def _enhancer(max_pixels=250_000):
    e = Enhancer(enabled=True, models_dir="/nonexistent", max_pixels=max_pixels)
    fake = _FakeX4Session()
    e.esrgan = SimpleNamespace(session=fake, run=fake.run, device="cpu", _out_scale=255.0)
    return e


def test_only_low_resolution_is_restored():
    e = _enhancer(max_pixels=250_000)
    assert e.needs_enhance(Image.new("RGB", (400, 300)))
    assert not e.needs_enhance(Image.new("RGB", (922, 2000)))  # the selfie that froze the service


def test_high_resolution_returned_untouched():
    e = _enhancer(max_pixels=250_000)
    e.birefnet = SimpleNamespace(session=object(), device="cpu")
    img = Image.new("RGB", (922, 2000), (10, 20, 30))
    out, was, info = e.enhance(img)
    assert out is img and was is False  # never upscaled, never downscaled
    assert "high resolution" in info["reason"]
    assert e.esrgan.session.shapes == []


def test_tiled_upscale_matches_whole_frame_and_bounds_tiles():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(300, 530, 3), dtype=np.uint8)
    e = _enhancer()
    out = e.upscale(Image.fromarray(arr))
    assert out.size == (530 * 4, 300 * 4)
    expected = arr.repeat(4, axis=0).repeat(4, axis=1)
    assert np.array_equal(np.asarray(out), expected)
    # every model call is one small tile, never the whole frame
    assert all(max(s[2:]) <= 256 + 2 * 16 for s in e.esrgan.session.shapes)


def test_birefnet_skipped_when_memory_is_low(monkeypatch):
    """BiRefNet needs 6-9 GB on CPU: on a busy box it is skipped (logged),
    the ESRGAN upscale is returned, and the service keeps running."""
    import app.enhance as enh

    e = _enhancer()
    called = []
    e.birefnet = SimpleNamespace(session=object(), device="cpu")
    e.refine = lambda img: called.append(img)
    monkeypatch.setattr(enh, "_available_mb", lambda: 2000)
    out, was, info = e.enhance(Image.new("RGB", (200, 150)))
    assert was is True and out.size == (800, 600)
    assert called == [] and info["restored"] == "esrgan" and "birefnet" in info["skipped"]


def test_plan_devices_hybrid_placement():
    from app.enhance import plan_devices

    needs = {"realesrgan": 1500, "birefnet": 4500}
    # no CUDA-capable onnxruntime (or no GPU): both in system RAM
    assert plan_devices(needs, "auto", cuda_ok=False, vram_free_mb=24000) == {"realesrgan": "cpu", "birefnet": "cpu"}
    # big RAM, small GPU: the big model stays in RAM, the small one goes to VRAM
    assert plan_devices(needs, "auto", cuda_ok=True, vram_free_mb=3000) == {"realesrgan": "gpu", "birefnet": "cpu"}
    # enough VRAM for both: both on the GPU
    assert plan_devices(needs, "auto", cuda_ok=True, vram_free_mb=7000) == {"realesrgan": "gpu", "birefnet": "gpu"}
    # too little VRAM for either
    assert plan_devices(needs, "auto", cuda_ok=True, vram_free_mb=1000) == {"realesrgan": "cpu", "birefnet": "cpu"}
    # forced modes
    assert plan_devices(needs, "cpu", cuda_ok=True, vram_free_mb=24000) == {"realesrgan": "cpu", "birefnet": "cpu"}
    assert plan_devices(needs, "gpu", cuda_ok=True, vram_free_mb=0) == {"realesrgan": "gpu", "birefnet": "gpu"}


def test_gpu_run_failure_falls_back_to_ram(monkeypatch):
    from app.enhance import _LazyORT

    class Boom:
        def run(self, *_):
            raise RuntimeError("CUDA out of memory")

    class Ok:
        def run(self, *_):
            return ["ok"]

    m = _LazyORT("birefnet", "/nonexistent", device="gpu")
    m._session = Boom()
    monkeypatch.setattr(m, "_create", lambda device: (Ok(), device))
    assert m.run(["o"], {}) == ["ok"]
    assert m.device == "cpu"
