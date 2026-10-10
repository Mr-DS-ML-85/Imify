"""Geometric verification — TinEye-style "same picture, modified".

Perceptual hashes summarize the WHOLE frame, so a crop, an added border,
a caption bar, a sticker or a collage moves them far apart even though the
candidate plainly contains the query photo. Local keypoints survive all of
that: SIFT features are matched between the two images (Lowe ratio test) and
a RANSAC homography keeps only geometrically consistent matches. Many
consistent inliers can only happen when one image really contains (a scaled /
cropped / shifted / slightly rotated view of) the other.

Pure OpenCV on CPU. The query side is computed once per search
(``Keypoints.from_image``) and reused for every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

_MAX_SIDE = 640
_N_FEATURES = 1500


def _gray(img: Image.Image) -> np.ndarray:
    im = img.convert("L")
    long_side = max(im.size)
    if long_side > _MAX_SIDE:
        scale = _MAX_SIDE / long_side
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


@dataclass
class Keypoints:
    points: np.ndarray  # (N, 2) float32
    desc: np.ndarray  # (N, 128) float32

    @classmethod
    def from_image(cls, img: Image.Image) -> "Keypoints | None":
        try:
            import cv2
        except Exception:  # noqa: BLE001
            return None
        sift = cv2.SIFT_create(nfeatures=_N_FEATURES)
        kps, desc = sift.detectAndCompute(_gray(img), None)
        if desc is None or len(kps) < 8:
            return None
        pts = np.asarray([k.pt for k in kps], dtype=np.float32)
        return cls(points=pts, desc=desc.astype(np.float32))


@dataclass
class GeometricMatch:
    inliers: int
    ratio: float  # inliers / ratio-test matches
    mirrored: bool = False

    def accepted(self, min_inliers: int, min_ratio: float) -> bool:
        return self.inliers >= min_inliers and self.ratio >= min_ratio


def match(query: Keypoints, cand: Keypoints) -> GeometricMatch:
    """RANSAC-verified keypoint agreement between two images."""
    import cv2

    if len(query.desc) < 2 or len(cand.desc) < 2:
        return GeometricMatch(0, 0.0)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(query.desc, cand.desc, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < 8:
        return GeometricMatch(0, 0.0)
    src = query.points[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = cand.points[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if homography is None or mask is None:
        return GeometricMatch(0, 0.0)
    # A degenerate homography (everything collapsed onto a line/point) is a
    # classic false positive on repetitive textures — reject it.
    det = float(np.linalg.det(homography[:2, :2]))
    if not (1e-3 < abs(det) < 1e3):
        return GeometricMatch(0, 0.0)
    inliers = int(mask.sum())
    return GeometricMatch(inliers, inliers / len(good))


def verify(query: Keypoints | None, query_mirror: Keypoints | None, cand_img: Image.Image) -> GeometricMatch | None:
    """Best geometric agreement of the candidate with the query or its
    mirror image (flipped re-uploads are common). None when keypoints can't
    be computed on either side."""
    if query is None:
        return None
    cand = Keypoints.from_image(cand_img)
    if cand is None:
        return None
    best = match(query, cand)
    if query_mirror is not None:
        flipped = match(query_mirror, cand)
        if flipped.inliers > best.inliers:
            flipped.mirrored = True
            best = flipped
    return best


def mirror(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)
