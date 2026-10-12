"""Detection providers: media in, canonical landmarks out."""

from __future__ import annotations

from .mediapipe import MediaPipeProvider
from .registry import DESCRIPTORS, descriptor
from .sam3d.provider import SAM3DBodyProvider

__all__ = [
    "DESCRIPTORS",
    "MediaPipeProvider",
    "SAM3DBodyProvider",
    "descriptor",
]
