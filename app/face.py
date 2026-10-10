"""Optional Face-ID layer for Imify (CPU only: OpenCV YuNet + ONNX ArcFace).

The FaceCheck-style trick, self-hosted: detect faces with YuNet and map each
face to an identity embedding with ArcFace MobileFaceNet (w600k_mbf), so Imify
can answer "where does THIS face appear?" with per-face confidence tiers.

Accuracy notes (measured on this box's own uploads):

- The face is aligned with a similarity transform onto the standard ArcFace
  112x112 5-point template and normalized ``(x - 127.5) / 127.5`` in RGB —
  exactly insightface's ``norm_crop`` + preprocessing. Same person scores
  >= 0.88, different people <= 0.30 (median ~0.01).
- The previous pipeline went through ``cv2.FaceRecognizerSF``: its
  ``alignCrop`` expects the FULL 15-value YuNet row (landmarks at offset 4)
  but was handed only the 10 keypoints, and ``feature()`` feeds raw 0-255
  pixels (SFace preprocessing) into ArcFace. Different people then scored
  ~0.74 median — identity matching was effectively random. Embeddings made
  that way are tagged with no/old ``MODEL_TAG`` and are re-embedded.

Models are downloaded once on first use (small, self-hosted, no API):

- ``face_detection_yunet_2023mar.onnx``  (~300 KB, OpenCV Zoo)
- ``w600k_mbf.onnx``                      (~14 MB, ArcFace MobileFaceNet)

Everything degrades cleanly: when ``FACE_ID_ENABLED`` is false, onnxruntime is
missing, or the models aren't downloaded yet, ``FaceId.available`` is False and
the pure perceptual pipeline runs exactly as before.
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from app.logging import child_logger

log = child_logger("imify.face")

# --------------------------------------------------------------------- models
YU_NET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# ArcFace MobileFaceNet ships INSIDE the v0.7 release's buffalo_sc.zip
# (w600k_mbf.onnx no longer ships as a standalone asset). ~15 MB.
BUFFALO_SC_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    "buffalo_sc.zip"
)

# standard ArcFace 112x112 alignment landmarks (used to warp the detected face)
_REF_5PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _download(url: str, dest: Path) -> bool:
    """Download a model file once; returns True when present/ready."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        log.info("downloading face model", extra={"model": dest.name, "url": url})
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — allow http(s) model dl
        if tmp.stat().st_size > 1000:
            tmp.rename(dest)
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("face model download failed", extra={"model": dest.name, "error": str(exc)})
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:  # noqa: BLE001
                pass
    return False


def _download_buffalo_sc(models_dir: Path) -> bool:
    """Download buffalo_sc.zip and extract w600k_mbf.onnx into models_dir.

    Returns True when the embedder model is present and valid (≥10 MB)."""
    emb = models_dir / "w600k_mbf.onnx"
    if emb.exists() and emb.stat().st_size > 1_000_000:
        return True
    models_dir.mkdir(parents=True, exist_ok=True)
    zpath = models_dir / "buffalo_sc.zip"
    try:
        log.info("downloading ArcFace model", extra={"url": BUFFALO_SC_URL})
        urllib.request.urlretrieve(BUFFALO_SC_URL, zpath)  # noqa: S310
        if zpath.stat().st_size < 1_000_000:
            raise RuntimeError("buffalo_sc.zip too small; download truncated?")
        import zipfile
        with zipfile.ZipFile(zpath) as zf:
            zf.extract("w600k_mbf.onnx", models_dir)
        if emb.exists() and emb.stat().st_size > 1_000_000:
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("ArcFace model download failed", extra={"error": str(exc)})
    finally:
        try:
            zpath.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    return False
class FaceId:
    """Lazy CPU face detector (YuNet) + identity embedder (ArcFace MBF)."""

    # Bump whenever detection/alignment/embedding changes so stored
    # embeddings from an older pipeline are ignored and re-embedded.
    MODEL_TAG = "arcface-mbf-norm-v2"

    def __init__(
        self,
        enabled: bool = False,
        device: str = "cpu",
        models_dir: str = "",
        min_face_size: int = 40,
        max_faces: int = 5,
        same_threshold: float = 0.42,
        score_threshold: float = 0.75,
    ):
        self.enabled = enabled
        self.device = device  # accepted for config compatibility; always CPU
        self.models_dir = Path(models_dir or "data/models/face")
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self.same_threshold = same_threshold
        self.score_threshold = score_threshold
        self._detector = None
        self._embedder = None  # onnxruntime session (thread-safe run())
        self._embed_input = ""
        # YuNet keeps per-instance input-size state (setInputSize + detect)
        # and cv2.dnn nets are not thread-safe; verification runs many
        # candidates concurrently, so detection is serialized.
        self._det_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._load_error: str | None = None
        self._checked = False

    @classmethod
    def from_settings(cls, settings) -> "FaceId":
        return cls(
            enabled=settings.face_id_enabled,
            device=settings.face_id_device,
            models_dir=str(settings.face_id_models_dir),
            min_face_size=settings.face_id_min_face_size,
            max_faces=settings.face_id_max_faces,
            same_threshold=settings.face_id_same_threshold,
            score_threshold=settings.face_id_score_threshold,
        )

    # ------------------------------------------------------------ availability
    @property
    def available(self) -> bool:
        if not self.enabled or self._load_error:
            return False
        if self._checked:
            return self._detector is not None
        with self._load_lock:
            if self._checked:
                return self._detector is not None
            try:
                self._load()
            finally:
                self._checked = True
        return self._detector is not None

    def _load(self) -> None:
        try:
            import cv2
            import onnxruntime as ort
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"opencv/onnxruntime not installed: {exc}"
            return
        det = self.models_dir / "face_detection_yunet_2023mar.onnx"
        emb = self.models_dir / "w600k_mbf.onnx"
        if not (det.exists() and emb.exists()):
            if not (_download(YU_NET_URL, det) and _download_buffalo_sc(self.models_dir)):
                self._load_error = "face models unavailable (offline?)"
                return
        # CPU-only policy: Imify NEVER initializes a CUDA session (a GPU
        # driver crash once took the whole box down). FACE_ID_DEVICE is
        # accepted for compatibility but always resolves to CPU.
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(det), "", (320, 320), self.score_threshold, 0.3, 5000,
                backend_id=cv2.dnn.DNN_BACKEND_DEFAULT, target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._embedder = ort.InferenceSession(str(emb), sess_options=opts, providers=["CPUExecutionProvider"])
            self._embed_input = self._embedder.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            self._detector = None
            self._embedder = None
            self._load_error = f"face model load failed: {exc}"

    @property
    def provider(self) -> str:
        return "CPU" if self._detector is not None else ""

    # -------------------------------------------------------------- detection
    def detect_faces(self, img: Image.Image) -> list[dict]:
        """Single-scale YuNet pass. Returns faces as
        ``{bbox: [x, y, w, h], kps: [[x, y] * 5], conf}`` in ``img`` pixels."""
        if not self.available:
            return []
        bgr = np.asarray(img.convert("RGB"), dtype=np.uint8)[..., ::-1].copy()
        h, w = bgr.shape[:2]
        with self._det_lock:
            self._detector.setInputSize((w, h))
            _ok, faces = self._detector.detect(bgr)
        out: list[dict] = []
        if faces is None:
            return out
        for f in faces:
            out.append({
                "bbox": [float(f[0]), float(f[1]), float(f[2]), float(f[3])],
                "kps": f[4:14].reshape(5, 2).astype(float).tolist(),
                "conf": float(f[14]),
            })
        return out

    def detect_faces_multiscale(self, img: Image.Image) -> list[dict]:
        """Detect at a bounded working resolution, plus a 2x pass for small
        images so distant/tiny faces are still caught. Big photos are
        downscaled first (YuNet on a 4000px frame is slow and no more
        accurate). Boxes are mapped back to ``img`` pixels, deduped by IoU,
        filtered by ``min_face_size`` and sorted largest-first."""
        if not self.available:
            return []
        long_side = max(img.size)
        base = min(1.0, 1280.0 / long_side) if long_side else 1.0
        scales = [base]
        if long_side * base < 900:
            scales.append(base * 2.0)
        found: list[dict] = []
        for scale in scales:
            if abs(scale - 1.0) > 1e-3:
                size = (max(32, int(img.width * scale)), max(32, int(img.height * scale)))
                work = img.resize(size, Image.Resampling.BILINEAR)
            else:
                work = img
            for f in self.detect_faces(work):
                f["bbox"] = [v / scale for v in f["bbox"]]
                f["kps"] = [[x / scale, y / scale] for x, y in f["kps"]]
                found.append(f)
        kept: list[dict] = []
        for f in sorted(found, key=lambda f: -f["conf"]):
            if min(f["bbox"][2], f["bbox"][3]) < self.min_face_size:
                continue
            if any(_iou(f["bbox"], k["bbox"]) > 0.4 for k in kept):
                continue
            kept.append(f)
        kept.sort(key=lambda f: -(f["bbox"][2] * f["bbox"][3]))
        return kept[: self.max_faces]

    # -------------------------------------------------------------- embedding
    def embed_face(self, img: Image.Image, kps: list) -> np.ndarray | None:
        """Align the face onto the ArcFace template and embed it.
        Returns the L2-normalized 512-dim identity embedding or None."""
        if not self.available:
            return None
        import cv2

        src = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        matrix, _ = cv2.estimateAffinePartial2D(src, _REF_5PTS, method=cv2.LMEDS)
        if matrix is None:
            return None
        bgr = np.asarray(img.convert("RGB"), dtype=np.uint8)[..., ::-1]
        aligned = cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        try:
            emb = self._embedder.run(None, {self._embed_input: blob})[0].reshape(-1)
        except Exception:  # noqa: BLE001
            return None
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            return None
        return (emb / norm).astype(np.float32)

    def embed_image(self, img: Image.Image) -> list[dict]:
        """Detect + embed every face (largest first). Returns
        ``[{embedding, bbox: [x, y, w, h], confidence}]``."""
        out: list[dict] = []
        for f in self.detect_faces_multiscale(img):
            emb = self.embed_face(img, f["kps"])
            if emb is not None:
                out.append({"embedding": emb.tolist(), "bbox": f["bbox"], "confidence": f["conf"]})
        return out

    # -------------------------------------------------------------- matching
    @staticmethod
    def match(qemb, cand) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(np.asarray(qemb, dtype=np.float32), np.asarray(cand, dtype=np.float32)))

    def best_match(self, query_embs: list, cand_embs: list) -> float:
        """Highest similarity between any query face and any candidate face."""
        if not query_embs or not cand_embs:
            return 0.0
        q = np.asarray(query_embs, dtype=np.float32).reshape(len(query_embs), -1)
        c = np.asarray(cand_embs, dtype=np.float32).reshape(len(cand_embs), -1)
        if q.shape[1] != c.shape[1]:
            return 0.0
        return float((q @ c.T).max())

    def confidence_tier(self, sim: float) -> tuple[int, str]:
        """FaceCheck-style 0-100 score + tier from ArcFace cosine similarity.

        Cosine is not a percentage: with correct preprocessing different
        people sit below ~0.30 and the same person above ~0.45 (typically
        0.6-0.9). The score maps the decision threshold to 70 and 0.75+ to
        99, so the number reads like FaceCheck's (90+ = certain match).
        """
        s = float(sim)
        t = self.same_threshold
        if s < t:
            conf = int(round(max(0.0, s) / t * 69))
        else:
            conf = int(round(min(99.0, 70 + (s - t) / max(1e-6, 0.75 - t) * 29)))
        if s >= 0.60:
            tier = "certain"
        elif s >= 0.50:
            tier = "confident"
        elif s >= t:
            tier = "uncertain"
        else:
            tier = "weak"
        return conf, tier

    def is_same_person(self, sim: float) -> bool:
        return sim >= self.same_threshold

    def match_identity(
        self,
        qemb,
        candidates: list[tuple[int, list]],
    ) -> list[tuple[float, int]]:
        """Rank library faces of the SAME person as ``qemb``.

        ``candidates`` is ``[(image_id, embedding)]`` (several per image for
        group photos). Each image is scored by its best face; images below
        ``same_threshold`` are dropped. Returns ``[(sim, image_id)]`` best
        first. (A person appearing in many indexed photos is exactly what we
        want to find, so there is no "unique nearest neighbour" test — that
        used to discard every image after the first one of a person.)
        """
        if not candidates:
            return []
        q = np.asarray(qemb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return []
        q = q / n
        best: dict[int, float] = {}
        ids = [image_id for image_id, _ in candidates]
        mat = np.asarray([np.asarray(e, dtype=np.float32).reshape(-1) for _, e in candidates])
        if mat.ndim != 2 or mat.shape[1] != q.shape[0]:
            return []
        sims = mat @ q
        for image_id, sim in zip(ids, sims):
            sim = float(sim)
            if sim >= self.same_threshold and sim > best.get(image_id, -1.0):
                best[image_id] = sim
        return sorted(((s, i) for i, s in best.items()), key=lambda t: -t[0])

    @staticmethod
    def crop_face(img: Image.Image, bbox: list[float], pad: float = 0.6) -> Image.Image:
        """Crop a face with generous context (hair/shoulders help reverse
        engines). ``bbox`` is ``[x, y, w, h]``."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * (1 + 2 * pad)
        left = max(0, int(cx - side / 2))
        top = max(0, int(cy - side / 2))
        right = min(img.width, int(cx + side / 2))
        bottom = min(img.height, int(cy + side / 2))
        return img.crop((left, top, right, bottom))


def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
