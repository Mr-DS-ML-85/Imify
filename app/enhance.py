"""Optional blur-restoration layer for Imify (self-hosted ONNX, like clip.py).

Perceptual hashes need sharp, well-defined edges — a blurry photo encodes as a
smear, so it hash-matches unrelated smooth scenes (a blurred influencer photo
"matches sea"), and unclear text/screenshots verify weakly. This module
sharpens the *query* before it is hashed and before it is uploaded to the
web engines, so the whole reverse-search ladder sees a cleaner image.

Pipeline (only when the query is detected blurry):

1. **Blur detection** — Laplacian variance ([Lap_var] > :data:`blur_limit`).
2. **Real-ESRGAN x4** (xinntao) — super-resolution upscale of the whole frame.
3. **BiRefNet lite** (ZhengPeng7) — subject/background separation; the subject
   (``mask>=0.5``) is kept from the sharpened upscale and re-composited over a
   de-focused original background so the foreground pops while the noisy
   background softens.

Everything is lazy (model loaded on first use) and optional: when models are
absent or ``enabled=False``, :class:`Enhancer` is a no-op and nothing changes.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from app.logging import child_logger

log = child_logger("imify.enhance")

_ESRGAN_MODEL = "RealESRGAN_x4plus.onnx"
_BIRENET_MODEL = "birefnet_lite.onnx"
# Source exports (fp32 Real-ESRGAN-x4plus, BiRefNet-lite 1024).
_ESRGAN_URL = "https://github.com/net2cn/Real-ESRGAN_GUI/raw/refs/heads/master/Real-ESRGAN_GUI/models/realesrgan-x4plus.onnx"
_BIRENET_URL = f"https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/main/{_BIRENET_MODEL}"

_ESRGAN_TILE = 1024  # width/height must be divisibly processed in tiles
_ESRGAN_SCALE = 4
_BIRENET_SIZE = 1024
# Laplacian-variance blur threshold: below -> considered blurry (higher = require
# MORE sharpness to call the image "sharp"). Tunable via config.
BLUR_DEFAULT_LIMIT = 90.0


def _variance_of_laplacian(img: "Image.Image") -> float:
    """Blurmetric: variance of the Laplacian over the gray image. Smooth/blurry
    frames have low variance; sharp edge-rich frames have high variance."""
    gray = img.convert("L")
    a = np.asarray(gray.resize((max(gray.width // 2, 16), max(gray.height // 2, 16)), Image.BILINEAR), dtype=np.float32)
    la = np.zeros_like(a)
    if a.shape[0] >= 3 and a.shape[1] >= 3:
        # central-difference Laplacian kernel (avoid SciPy).
        la[1:-1, 1:-1] = (
            a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2] - 4.0 * a[1:-1, 1:-1]
        )
    return float(np.var(la))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _LazyORT:
    """Minimal lazy ONNX loader (mirrors Clipper's pattern)."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = Path(path)
        self._session = None
        self._load_error: str | None = None
        self._checked = False

    @property
    def session(self):
        if self._session is not None or self._load_error:
            return self._session
        if not self.path.exists():
            self._load_error = f"{self.name} model missing: {self.path}"
            return None
        try:
            import onnxruntime as ort  # noqa: F401
            self._session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            return self._session
        except Exception as exc:  # noqa: BLE001
            log.error(f"{self.name} load failed", extra={"error": str(exc)})
            self._load_error = str(exc)
            return None


class Enhancer:
    """Lazy CPU blur-restoration + subject-refinement module."""

    def __init__(
        self,
        enabled: bool = False,
        models_dir: str = "",
        blur_limit: float = BLUR_DEFAULT_LIMIT,
    ):
        self.enabled = enabled
        self.models_dir = Path(models_dir or "data/models/enhance")
        self.blur_limit = float(blur_limit)
        self.esrgan = _LazyORT("realesrgan", self.models_dir / _ESRGAN_MODEL)
        self.birefnet = _LazyORT("birefnet", self.models_dir / _BIRENET_MODEL)

    # ----------------------------------------------------------------- avail
    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        return self.esrgan.session is not None and self.birefnet.session is not None

    def needs_enhance(self, img: "Image.Image") -> bool:
        """True when the image is detected blurry enough to warrant restore."""
        if not self.enabled:
            return False
        return _variance_of_laplacian(img) < self.blur_limit

    # --------------------------------------------------------------- restore
    def upscale(self, img: "Image.Image") -> "Image.Image":
        """Real-ESRGAN x4 upscale (CPU), honoring the official IO contract:

        - input  : RGB, normalized /255 -> [0,1], layout CHW (transposed to the
                   model's declared spatial order, which may be W,H or H,W)
        - output : RGB, range [0,1]; pixels recovered with out*255 (clipped).

        Feeding BGR or the wrong spatial order is what produces blue-tint,
        seam lines and blur on non-square images.
        """
        sess = self.esrgan.session
        if sess is None:
            return img
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        iname, oname = inp.name, out.name

        # --- spatial-order detection from symbolic dim names ---
        # e.g. [N,3,'width','height'] -> model consumes (W,H): we must swap our
        # native (C,H,W) to (C,W,H). [N,3,'H','W'] -> native already matches.
        idims = [str(d).lower() for d in inp.shape]
        sp2 = idims[2] if len(idims) > 2 else ""
        wh_order = ("w" in sp2) or (sp2 == "width")  # model wants W before H

        a = np.asarray(img.convert("RGB"), dtype=np.float32)  # RGB (PIL already)
        H, W = a.shape[0], a.shape[1]
        pH = ((H + 63) // 64) * 64
        pW = ((W + 63) // 64) * 64
        if (pH, pW) != (H, W):
            pad = np.zeros((pH, pW, 3), dtype=np.float32)
            pad[:H, :W] = a
            a = pad
        x = (a / 255.0).transpose(2, 0, 1)  # -> (C, H, W), RGB, [0,1]
        if wh_order:
            x = x.transpose(0, 2, 1)  # -> (C, W, H)
        res = sess.run([oname], {iname: x.astype(np.float32)[None]})[0]
        if wh_order:
            res = res.transpose(0, 1, 3, 2)  # back -> (N, C, H, W)
        o = res[0].transpose(1, 2, 0)  # -> (H, W, C) RGB

        out_scale = self._esrgan_out_scale(sess, iname, oname)
        o = np.clip(o * out_scale, 0.0, 255.0)
        hi, wi = int(H * _ESRGAN_SCALE), int(W * _ESRGAN_SCALE)
        o = o[:hi, :wi]
        return Image.fromarray(o.astype(np.uint8))

    def _esrgan_out_scale(self, sess, iname: str, oname: str) -> float:
        """Probe once: model outputs ~[0,1] (-> *255) or directly ~[0,255].
        Cache per model."""
        scale = getattr(self.esrgan, "_out_scale", None)
        if scale is None:
            probe = np.zeros((3, 64, 64), dtype=np.float32) + 0.5  # mid gray 0-1
            ids = [str(d).lower() for d in sess.get_inputs()[0].shape]
            if len(ids) > 2 and ("w" in ids[2] or ids[2] == "width"):
                probe = probe.transpose(0, 2, 1)
            r = sess.run([oname], {iname: probe[None]})[0]
            mx = float(np.abs(r).max())
            scale = 1.0 if mx > 3.0 else 255.0  # already 0-255 vs. 0-1
            self.esrgan._out_scale = scale
        return scale

    def refine(self, img: "Image.Image") -> tuple["Image.Image", np.ndarray]:
        """BiRefNet subject foreground mask (sigmoid, 0-1) for an image.

        Handles non-square input by boxing/resizing to 1024x1024 with a gray
        border, running the model, then cropping the mask back to the original
        aspect. Returns ``(mask_image_uint8, mask01)`` at original resolution.
        """
        sess = self.birefnet.session
        if sess is None:
            return img, np.zeros((img.height, img.width), dtype=np.float32)
        W, H = img.size
        ar = H / W
        t = _BIRENET_SIZE
        if ar >= 1:  # tall/portrait
            h = t
            w = int(round(t / ar))
        else:
            w = t
            h = int(round(t * ar))
        im = img.convert("RGB").resize((w, h), Image.BILINEAR)
        box = np.zeros((t, t, 3), dtype=np.float32)  # 0 border
        a = np.asarray(im, dtype=np.float32) / 255.0
        ox, oy = (t - w) // 2, (t - h) // 2
        box[oy:oy + h, ox:ox + w] = a
        inp = box.transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(["output_image"], {"input_image": inp})[0][0][0]  # sigmoid(logit)
        mask = _sigmoid(out.astype(np.float32))
        mask = mask[oy:oy + h, ox:ox + w]
        # resize back to ORIGINAL resolution (up-scaled, not the ESR output)
        from PIL import Image as _Img
        m_img = _Img.fromarray((np.clip(mask, 0, 1) * 255.0).astype(np.uint8)).resize(
            (W, H), _Img.BILINEAR
        )
        return m_img, np.asarray(m_img, dtype=np.float32) / 255.0

    def enhance(self, img: "Image.Image") -> tuple["Image.Image", bool, dict]:
        """Blur → Real-ESRGAN → BiRefNet composite.

        Returns ``(result_image, was_enhanced, info)``. When the query is not
        blurred (or models unavailable) returns the original untouched.
        """
        if not self.available:
            return img, False, {"reason": "disabled"}
        var = _variance_of_laplacian(img)
        if var >= self.blur_limit:
            return img, False, {"reason": f"sharp (lap_var {var:.1f})"}
        up = self.upscale(img)
        try:
            mask_img, _mask01 = self.refine(up)
            # composite: keep the sharpened subject, de-focus the background
            up = up.convert("RGB")
            bg = up.filter(ImageFilter.GaussianBlur(radius=3))
            result = Image.composite(up, bg, mask_img.convert("L"))
        except Exception:  # noqa: BLE001 — refine optional
            result = up
        return result, True, {"lap_var": round(var, 1), "restored": "esrgan+birefnet"}