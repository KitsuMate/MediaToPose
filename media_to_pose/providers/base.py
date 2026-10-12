"""Detection provider protocol and registry descriptor."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Event
from typing import Any, Protocol

from ..core.types import CanonicalFrame

ProgressCallback = Callable[[int, int, str], None]
DependencyStatus = Callable[[dict[str, Any]], tuple[bool, str]]
CacheFingerprint = Callable[[dict[str, Any]], dict[str, Any]]


class DetectionProvider(Protocol):
    capabilities: frozenset[str]

    def detect_image(self, path: str) -> CanonicalFrame: ...

    def detect_video_frame_index(
        self, path: str, source_index: int
    ) -> CanonicalFrame: ...

    def iter_video(
        self,
        path: str,
        cancel_event: Event,
        progress: ProgressCallback,
    ) -> Iterable[CanonicalFrame]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ProviderDescriptor:
    """Provider metadata used by Capture, caching, and worker construction."""

    identifier: str
    label: str
    description: str
    capabilities: frozenset[str]
    provider_class: type[DetectionProvider]
    loading_message: str
    dependency_status: DependencyStatus
    cache_fingerprint: CacheFingerprint
