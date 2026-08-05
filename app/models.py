"""Shared data models for the Imify index and search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


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

    image_id: int | None = None
    url: str = ""
    page_url: str = ""
    title: str = ""
    engine: str = ""
    source: str = "local"  # "local" | "web" | "both"
    score: float = 0.0
    hamming: int | None = None
    match_type: str = "near"  # "exact" | "near" | "changed" | "color" | "web"
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
