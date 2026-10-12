"""Reduce Camera-Induced Sliding (video only)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
)

from ..base import CorrectionModule


class CameraMotionCorrection(CorrectionModule):
    module_id = "CAMERA_MOTION_CORRECTION"
    display_name = "Camera Motion Correction"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        if context.process_camera_motion is None:
            raise RuntimeError(
                "Camera Motion Correction is unavailable outside a video bake"
            )
        yield from context.process_camera_motion(settings)
