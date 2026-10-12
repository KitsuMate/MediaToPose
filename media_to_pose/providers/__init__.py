"""Capability-aware detection provider registry."""

from __future__ import annotations

from ..core.errors import InvalidRequest
from .base import ProviderDescriptor
from .mediapipe import MediaPipeProvider
from .mediapipe import dependency_status as mp_status
from .sam3d_body_provider import (
    SAM3DBodyProvider,
)
from .sam3d_body_provider import (
    cache_fingerprint as sam3d_fingerprint,
)
from .sam3d_body_provider import (
    dependency_status as sam3d_status,
)


def _mediapipe_fingerprint(_options):
    return {
        "adapter": "mediapipe-tasks-0.10.35-perspective-root-v1",
        "models": "bundled",
    }


DESCRIPTORS: dict[str, ProviderDescriptor] = {
    "MEDIAPIPE": ProviderDescriptor(
        identifier="MEDIAPIPE",
        label="MediaPipe",
        description="Bundled body, hands, and facial capture",
        capabilities=frozenset({"BODY", "HANDS", "FACE"}),
        provider_class=MediaPipeProvider,
        loading_message="Loading MediaPipe models",
        dependency_status=mp_status,
        cache_fingerprint=_mediapipe_fingerprint,
    ),
    "SAM3D_BODY": ProviderDescriptor(
        identifier="SAM3D_BODY",
        label="SAM 3D Body (Meta)",
        description="Optional gated MHR70 body and hand capture",
        capabilities=frozenset({"BODY", "HANDS"}),
        provider_class=SAM3DBodyProvider,
        loading_message="Loading SAM 3D Body model",
        dependency_status=sam3d_status,
        cache_fingerprint=sam3d_fingerprint,
    ),
}


def descriptor(identifier: str) -> ProviderDescriptor:
    try:
        return DESCRIPTORS[str(identifier)]
    except KeyError as exc:
        supported = ", ".join(sorted(DESCRIPTORS))
        raise InvalidRequest(
            f"Unknown capture provider {identifier!r}; expected {supported}"
        ) from exc


__all__ = [
    "DESCRIPTORS",
    "MediaPipeProvider",
    "SAM3DBodyProvider",
    "descriptor",
]
