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

import threading
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

# Real-ESRGAN runs tile by tile: one full-frame pass needs several GB of
# 64-channel feature maps at x4 (a 922x2000 selfie grew the process to 10 GB
# and froze the service). 256-px tiles + 16-px context keep it bounded.
_ESRGAN_TILE = 256
_ESRGAN_TILE_PAD = 16
_ESRGAN_SCALE = 4
# BiRefNet at 1024x1024 peaks at 6.6-9 GB of RAM on CPU (measured). In RAM it
# only runs when the box has at least this much memory available, so a busy
# machine skips the background blur instead of swapping the service to death.
BIREFNET_MIN_AVAILABLE_MB = 8192
# Hybrid placement (ENHANCE_DEVICE=auto): each model goes to the GPU when its
# VRAM need fits in the free VRAM, otherwise to system RAM. The bigger model
# (BiRefNet) is placed first, so "more RAM, less VRAM" puts BiRefNet in RAM and
# ESRGAN on the GPU; no GPU (or the CPU-only onnxruntime) puts both in RAM; a
# big GPU takes both. Estimates in MB, overridable via settings.
ESRGAN_VRAM_MB = 1500
BIREFNET_VRAM_MB = 4500
VRAM_HEADROOM_MB = 512
# Only LOW-resolution queries are restored; anything above this many pixels is
# left untouched (already enough detail, and never downscaled).
MAX_PIXELS_DEFAULT = 250_000
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


def _available_mb() -> int:
    """MemAvailable from /proc/meminfo (MB); a large number when unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


def _cuda_available() -> bool:
    """True when this onnxruntime build can run on an NVIDIA GPU."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def _gpu_free_mb() -> int:
    """Free VRAM of GPU 0 in MB (0 when there is no usable NVIDIA GPU)."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else 0
    except Exception:  # noqa: BLE001
        return 0


def plan_devices(
    vram_needs: dict[str, int],
    mode: str = "auto",
    cuda_ok: bool = False,
    vram_free_mb: int = 0,
    headroom_mb: int = VRAM_HEADROOM_MB,
) -> dict[str, str]:
    """Decide ``{model: "gpu" | "cpu"}``.

    ``cpu`` / ``gpu`` force every model; ``auto`` places the models with the
    largest VRAM need first on the GPU while they fit in the free VRAM (minus
    headroom) and puts the rest in system RAM. Without a CUDA-capable
    onnxruntime everything runs in RAM.
    """
    if mode == "cpu" or not cuda_ok:
        return {m: "cpu" for m in vram_needs}
    if mode == "gpu":
        return {m: "gpu" for m in vram_needs}
    left = vram_free_mb - headroom_mb
    plan: dict[str, str] = {}
    for model, need in sorted(vram_needs.items(), key=lambda kv: -kv[1]):
        if need <= left:
            plan[model] = "gpu"
            left -= need
        else:
            plan[model] = "cpu"
    return plan


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _LazyORT:
    """Lazy ONNX session on the planned device, falling back to CPU.

    ``device="gpu"`` uses the CUDA provider; if the session cannot be created
    there, or a run fails (out of VRAM, driver error), the model is rebuilt in
    system RAM and the run retried once — a GPU problem never fails a search.
    """

    def __init__(self, name: str, path: str, device: str = "cpu"):
        self.name = name
        self.path = Path(path)
        self.device = device
        self._session = None
        self._load_error: str | None = None

    def _create(self, device: str):
        import onnxruntime as ort

        # No arena: ORT's CPU arena keeps the PEAK allocation for the life of
        # the process (BiRefNet: ~9 GB held after one run). Without it memory
        # returns to ~1 GB after each restore.
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        providers: list = ["CPUExecutionProvider"]
        if device == "gpu":
            providers.insert(0, ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
            }))
        sess = ort.InferenceSession(str(self.path), opts, providers=providers)
        if device == "gpu" and sess.get_providers()[0] != "CUDAExecutionProvider":
            device = "cpu"
        return sess, device

    @property
    def session(self):
        if self._session is not None or self._load_error:
            return self._session
        if not self.path.exists():
            self._load_error = f"{self.name} model missing: {self.path}"
            return None
        try:
            try:
                self._session, self.device = self._create(self.device)
            except Exception as exc:  # noqa: BLE001
                if self.device != "gpu":
                    raise
                log.warning(f"{self.name}: GPU session failed ({exc}); using system RAM")
                self._session, self.device = self._create("cpu")
            log.info(f"enhance: {self.name} loaded on {self.device}")
            return self._session
        except Exception as exc:  # noqa: BLE001
            log.error(f"{self.name} load failed", extra={"error": str(exc)})
            self._load_error = str(exc)
            return None

    def run(self, outputs, feeds):
        sess = self.session
        try:
            return sess.run(outputs, feeds)
        except Exception as exc:  # noqa: BLE001
            if self.device != "gpu":
                raise
            log.warning(f"{self.name}: GPU run failed ({exc}); moving to system RAM")
            self._session, self.device = self._create("cpu")
            return self._session.run(outputs, feeds)


class Enhancer:
    """Lazy CPU blur-restoration + subject-refinement module."""

    def __init__(
        self,
        enabled: bool = False,
        models_dir: str = "",
        blur_limit: float = BLUR_DEFAULT_LIMIT,
        max_pixels: int = MAX_PIXELS_DEFAULT,
        device: str = "auto",
        esrgan_vram_mb: int = ESRGAN_VRAM_MB,
        birefnet_vram_mb: int = BIREFNET_VRAM_MB,
    ):
        self.enabled = enabled
        self.models_dir = Path(models_dir or "data/models/enhance")
        self.blur_limit = float(blur_limit)  # reported in info only
        self.max_pixels = int(max_pixels)
        self._lock = threading.Lock()  # one restore at a time (memory)
        self.device_mode = device
        self._vram_needs = {"realesrgan": int(esrgan_vram_mb), "birefnet": int(birefnet_vram_mb)}
        plan = {"realesrgan": "cpu", "birefnet": "cpu"}
        if enabled and device != "cpu":
            cuda = _cuda_available()
            plan = plan_devices(self._vram_needs, device, cuda, _gpu_free_mb() if cuda else 0)
        self.esrgan = _LazyORT("realesrgan", self.models_dir / _ESRGAN_MODEL, plan["realesrgan"])
        self.birefnet = _LazyORT("birefnet", self.models_dir / _BIRENET_MODEL, plan["birefnet"])

    @property
    def placement(self) -> dict[str, str]:
        """Where each model runs now ("gpu" / "cpu"); reported in /v1/stats."""
        return {"mode": self.device_mode, "realesrgan": self.esrgan.device, "birefnet": self.birefnet.device}

    # ----------------------------------------------------------------- avail
    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        return self.esrgan.session is not None and self.birefnet.session is not None

    def is_low_res(self, img: "Image.Image") -> bool:
        return img.width * img.height <= self.max_pixels

    def needs_enhance(self, img: "Image.Image") -> bool:
        """True only for LOW-resolution images; high-resolution ones are
        never restored (and never downscaled to make them fit)."""
        if not self.enabled:
            return False
        return self.is_low_res(img)

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

        a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # RGB [0,1]
        H, W = a.shape[0], a.shape[1]
        T, P, S = _ESRGAN_TILE, _ESRGAN_TILE_PAD, _ESRGAN_SCALE
        # reflect-pad so every tile has T px of content plus P px of context
        nH, nW = -(-H // T) * T, -(-W // T) * T
        src = np.pad(a, ((P, nH - H + P), (P, nW - W + P), (0, 0)), mode="reflect")
        out_full = np.zeros((nH * S, nW * S, 3), dtype=np.float32)
        out_scale = None
        for y in range(0, nH, T):
            for x in range(0, nW, T):
                tile = src[y:y + T + 2 * P, x:x + T + 2 * P]
                t = tile.transpose(2, 0, 1)  # -> (C, H, W)
                if wh_order:
                    t = t.transpose(0, 2, 1)  # -> (C, W, H)
                res = self.esrgan.run([oname], {iname: np.ascontiguousarray(t)[None]})[0]
                if wh_order:
                    res = res.transpose(0, 1, 3, 2)  # back -> (N, C, H, W)
                o = res[0].transpose(1, 2, 0)  # -> (H, W, C)
                if out_scale is None:
                    out_scale = self._esrgan_out_scale(sess, iname, oname)
                out_full[y * S:(y + T) * S, x * S:(x + T) * S] = o[P * S:(P + T) * S, P * S:(P + T) * S]
        o = np.clip(out_full[:H * S, :W * S] * (out_scale or 255.0), 0.0, 255.0)
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
            r = self.esrgan.run([oname], {iname: probe[None]})[0]
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
        out = self.birefnet.run(["output_image"], {"input_image": inp})[0][0][0]  # sigmoid(logit)
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
        if not self.is_low_res(img):
            return img, False, {"reason": f"high resolution ({img.width}x{img.height})"}
        with self._lock:
            up = self.upscale(img)
        avail = _available_mb()
        if self.birefnet.device == "cpu" and avail < BIREFNET_MIN_AVAILABLE_MB:
            log.warning(
                f"enhance: BiRefNet skipped, {avail} MB RAM available "
                f"(needs {BIREFNET_MIN_AVAILABLE_MB} MB) — returning the ESRGAN upscale"
            )
            return up, True, {"lap_var": round(var, 1), "restored": "esrgan",
                              "skipped": f"birefnet: {avail} MB RAM available",
                              "size": f"{img.width}x{img.height}->{up.width}x{up.height}"}
        try:
            with self._lock:
                mask_img, _mask01 = self.refine(up)
            # composite: keep the sharpened subject, de-focus the background
            up = up.convert("RGB")
            bg = up.filter(ImageFilter.GaussianBlur(radius=3))
            result = Image.composite(up, bg, mask_img.convert("L"))
        except Exception:  # noqa: BLE001 — refine optional
            result = up
        return result, True, {"lap_var": round(var, 1), "restored": "esrgan+birefnet",
                              "size": f"{img.width}x{img.height}->{result.width}x{result.height}"}