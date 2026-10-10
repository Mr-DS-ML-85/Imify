"""Shared data models for the Imify index and search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Match types that confirm "this is your image / your person" and therefore
# count as FOUND. Everything else in the grid is a lead, not a confirmation:
#   similar     — visually similar content (CLIP, Google-Lens style)
#   ocr         — same visible text
#   unverified  — engine candidate we could not confirm
CONFIRMED_MATCH_TYPES = frozenset({"exact", "near", "modified", "face", "changed", "color"})
# Match types that are copies of the SAME picture (safe for takedowns and
# for promoting a remote URL to canonical).
COPY_MATCH_TYPES = frozenset({"exact", "near", "modified"})
# Grid order: copies first, then the same person, then leads.
MATCH_FAMILY_RANK = {
    "exact": 0, "near": 0, "modified": 0, "changed": 0, "color": 0,
    "face": 1,
    "similar": 2, "ocr": 2,
    "unverified": 3,
}


@dataclass
class FeatureSet:
    """Perceptual features extracted from one image."""

    phash: int = 0  # 64-bit DCT hash
    dhash: int = 0  # 64-bit gradient hash
    ahash: int = 0  # 64-bit average hash
    color: list[float] = field(default_factory=list)  # 64-dim L2-normalized HSV histogram
    sha256: str = ""
    width: int = 0
    height: int = 0
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageMeta:
    """One canonical indexed image (deduplicated by perceptual content)."""

    id: int
    url: str
    page_url: str = ""
    title: str = ""
    engine: str = ""
    query: str = ""
    phash: int = 0
    dhash: int = 0
    ahash: int = 0
    color: list[float] = field(default_factory=list)
    sha256: str = ""
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    thumb_ext: str = "jpg"
    first_seen: float = 0.0
    last_seen: float = 0.0
    fetch_count: int = 1
    source_count: int = 1  # how many distinct URLs share this content
    source_urls: list[str] = field(default_factory=list)
    status: str = "indexed"
    # AI-generated descriptive caption (self-hosted Ollama vision model), if
    # enabled. Surfaces as alt-text in the UI and powers natural-language search.
    caption: str = ""
    # Visible text extracted via OCR (self-hosted tesseract), if enabled —
    # powers "search by the words on the image".
    ocr_text: str = ""
    # AI-embedding of the caption (same embed model as queries). Persisted at
    # INDEX time so by-description search is a single brute-force cosine over
    # stored vectors — never a re-embed of every caption per query. Empty when
    # captions aren't embedded.
    caption_embedding: list[float] = field(default_factory=list)
    # Face-ID: per-face identity embeddings (ArcFace, L2-normalized) for faces
    # detected in this image. Empty when Face-ID is off or no face was found.
    # One canonical embedding per face (a group photo has several).
    face_embeddings: list[list[float]] = field(default_factory=list)
    # Number of detected faces (0 when none / disabled).
    face_count: int = 0
    # FaceId.MODEL_TAG of the pipeline that produced face_embeddings.
    # Embeddings from an older/broken pipeline are ignored and re-embedded.
    face_model: str = ""
    # CLIP ViT-B/32 embedding of the WHOLE image (512-dim, L2-normalized) —
    # captures semantic content so /v1/search/clip can score "what else is
    # like this scene". Empty when CLIP is off.
    clip_embedding: list[float] = field(default_factory=list)
    # Which AI layers produced this entry's derived data, e.g. {"face","clip",
    # "ocr","caption"}. Lets the search grid know at a glance what a local
    # result can be cross-checked with.
    ai_modalities: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageMeta":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageHit:
    """An image found by the SearXNG discovery layer (before indexing)."""

    title: str
    url: str
    img_src: str
    thumbnail_src: str = ""
    engine: str = ""
    width: int = 0
    height: int = 0
    img_format: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """One ranked result returned by /v1/search."""

    @property
    def confirmed(self) -> bool:
        return self.match_type in CONFIRMED_MATCH_TYPES

    image_id: int | None = None
    url: str = ""
    page_url: str = ""
    title: str = ""
    engine: str = ""
    source: str = "local"  # "local" | "web" | "both"
    score: float = 0.0
    hamming: int | None = None
    # Perceptual fingerprint (pHash) of the ACTUAL verified bytes — used to
    # consolidate multiple URLs of the SAME underlying photo into one result
    # with a growing "found on N sites" list, instead of flooding the grid
    # with near-identical tiles of one image.
    content_phash: int | None = None
    # "exact" | "near" | "modified" (crop/edit, keypoint-verified) | "face"
    # (same person, different photo) | "similar" (CLIP) | "ocr" | "changed"
    # | "color" | "unverified" — see CONFIRMED_MATCH_TYPES.
    match_type: str = "near"
    # FaceCheck-style confidence 0-100 ("90-100 Certain / 70-82 Uncertain").
    # Derived from the fused perceptual distance: the more similar the pixels,
    # the higher the % shown in the UI. 0 for unverified/failed candidates.
    confidence: int = 0
    # Reversely-style bucket: "duplicate" (same image) | "similar" (close
    # variant/crop) | "candidate" (fetchable but not confirmed). Lets the UI
    # group exact copies apart from visually-similar previews — never a flat
    # mixed pile like a naive aggregator.
    category: str = "candidate"
    width: int = 0
    height: int = 0
    thumb: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    source_count: int = 1
    # every known URL where this image has been verified (TinEye's "found on
    # N sites" — the removal-request list). Local index copies carry their
    # accumulated source links; web results carry the verified candidate URL.
    source_urls: list[str] = field(default_factory=list)
    # hop-by-hop resolution trail for web-verified results, e.g.
    # ["cookies:lookaside.fbsbx.com", "redirect-stub", "render:gui_chrome"] —
    # shows users exactly how a candidate URL was resolved to pixels.
    resolution: list[str] = field(default_factory=list)
    # human note: unverified reason ("fetch 403", "hash mismatch Δ17") or a
    # warning for verified results whose URL is ephemeral (FB signed CDN).
    note: str = ""
    # AI-generated description of this result's content (self-hosted), when
    # available. Shown as alt-text and scored against a natural-language query.
    caption: str = ""
    # Which signals confirmed / suggested this result: "hash", "geometry",
    # "face", "clip", "ocr", "mirrored".
    ai_modalities: list[str] = field(default_factory=list)
    # ArcFace similarity of the best face vs the query face, + FaceCheck tier.
    face_sim: float | None = None
    face_tier: str = ""
    # CLIP cosine similarity vs the query (semantic closeness).
    clip_sim: float | None = None
    # RANSAC keypoint inliers when geometry verified a crop/edit.
    geo_inliers: int | None = None
    # semantic CLIP embedding of this result's actual pixels (512-dim, L2).
    # Carried so consolidation can merge visually-identical copies even when
    # their pHashes diverge (>threshold re-encodes). Excluded from to_dict().
    clip_embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("clip_embedding", None)
        return d


@dataclass
class SearchResponse:
    query_image_id: int | None = None
    mode: str = "best"
    total: int = 0
    took_ms: int = 0
    providers: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    # SearchOS-style router state: which layer answered and whether anything
    # was found at all ("search not found" when False and nothing was matched).
    found: bool = False
    fallback: str = "local"  # "local" | "web" | "none"
    # True when web results came from the verified-results cache (real-time
    # repeat search) instead of a fresh crawl.
    cached: bool = False
    # reverseimagesearch-style: results grouped by engine so the UI can show
    # per-engine tabs ("Google / Bing / Yandex / index") instead of one mixed
    # garbage pile. Map engine->[result dicts], ordered by best match first.
    by_engine: dict[str, Any] = field(default_factory=dict)
    # Reversely-style buckets: {duplicate:[], similar:[], candidate:[]} so the
    # UI separates exact copies from visually-similar previews.
    by_category: dict[str, Any] = field(default_factory=dict)
    # True when the query was answered by description (AI text search) instead
    # of by pixels.
    text_query: bool = False
    # Google-Lens style "best guess" derived from what other sites call this
    # image, and the text-discovery fan-out it drove ({queries, engines,
    # blocked, browser_used}).
    best_guess: str = ""
    discovery: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
