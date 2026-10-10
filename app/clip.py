"""Optional CLIP content-embedding layer for Imify (self-hosted ONNX).

Perceptual hashes (pHash/dHash/aHash) separate *copies of the same image*
but cannot tell a busy homelab scene from an unrelated black-dog photo — both
hash as "some photograph". CLIP ViT-B/32 embeddings capture WHAT an image
shows semantically, so the same scene → close embedding even across
completely different crops/views, and unrelated scenes → far embeddings.

This mirrors the FaceId module's lazy-load + auto-download pattern:

- ``Clipper.available`` downloads the ONNX model on first use (best-effort)
- ``Clipper.embed_image(img)`` → 512-dim L2-normalized embedding (or None)
- Runs on CPU via onnxruntime (returns when CUDA not available); ViT-B/32
  int8 is ~90MB and ~40ms/image — fine for per-index and on-the-fly search.

Nothing here is required for the perceptual pipeline; when CLIP is off,
``Clipper`` is a no-op and everything degrades exactly as before.
"""

from __future__ import annotations

import threading
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image

from app.logging import child_logger

log = child_logger("imify.clip")

_VISION_MODEL = "onnx/vision_model_int8.onnx"
# Source: Xenova/clip-vit-base-patch32 (Transformer.js ONNX export of the
# canonical openai/clip-vit-base-patch32 vision tower, quantized int8).
_VISION_URL = f"https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/{_VISION_MODEL}"

_IMAGE_SIZE = 224
# CLIP normalization stats (openai/clip-vit-base-patch32 preprocessing).
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


_LOAD_LOCK = threading.Lock()


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        return vec / norm
    return vec


class Clipper:
    """Lazy ONNX CLIP ViT-B/32 image embedder (CPU)."""

    def __init__(
        self,
        enabled: bool = False,
        models_dir: str = "",
        same_threshold: float = 0.80,
    ):
        self.enabled = enabled
        self.models_dir = Path(models_dir or "data/models/clip")
        self.same_threshold = same_threshold
        self._session = None  # onnxruntime InferenceSession
        self._load_error: str | None = None
        self._provider = "CPU"
        self._checked = False

    @classmethod
    def from_settings(cls, settings) -> "Clipper":
        return cls(
            enabled=settings.clip_enabled,
            models_dir=str(settings.clip_models_dir),
            same_threshold=settings.clip_same_threshold,
        )

    # ------------------------------------------------------------ availability
    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._load_error:
            return False
        if self._checked:
            return self._session is not None
        with _LOAD_LOCK:
            if self._checked:
                return self._session is not None
            ok = self._load()
            self._checked = True
            return ok

    def _load(self) -> bool:
        try:
            import onnxruntime as ort  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"onnxruntime not installed: {exc}"
            return False
        model = self.models_dir / "vision_model_int8.onnx"
        if not model.exists():
            if not self._download(model):
                self._load_error = "CLIP model unavailable (offline?)"
                return False
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(model),
                providers=["CPUExecutionProvider"],
            )
            # verify the expected io contract at load time, not at first search
            names = {i.name for i in self._session.get_inputs()}
            out_names = {o.name for o in self._session.get_outputs()}
            if "pixel_values" not in names or "image_embeds" not in out_names:
                self._load_error = f"unexpected model io: in={names} out={out_names}"
                self._session = None
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"onnx load failed: {exc}"
            self._session = None
            return False

    def _download(self, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            log.info("downloading CLIP model", extra={"model": dest.name, "url": _VISION_URL})
            urllib.request.urlretrieve(_VISION_URL, tmp)  # noqa: S310
            if tmp.stat().st_size > 1_000_000:
                tmp.rename(dest)
                return True
        except Exception as exc:  # noqa: BLE001
            log.warning("CLIP model download failed", extra={"model": dest.name, "error": str(exc)})
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        return False

    @property
    def provider(self) -> str:
        if self._session is None:
            return ""
        return self._provider

    # ---------------------------------------------------------------- embed
    def embed_image(self, img: Image.Image) -> list[float] | None:
        """Return the 512-dim L2-normalized CLIP embedding of an image."""
        if not self.available or self._session is None:
            return None
        try:
            rgb = img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.Resampling.BICUBIC)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            arr = (arr - _MEAN) / _STD
            arr = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)
            emb = self._session.run(["image_embeds"], {"pixel_values": arr})[0]
            normed = _l2_normalize(emb[0])
            return [float(x) for x in normed]
        except Exception as exc:  # noqa: BLE001
            log.debug("clip embed failed", extra={"error": str(exc)})
            return None

    @staticmethod
    def match(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two L2-normalized CLIP embeddings."""
        if not a or not b or len(a) != len(b):
            return 0.0
        return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))

    def same(self, a: list[float], b: list[float]) -> bool:
        return self.match(a, b) >= self.same_threshold