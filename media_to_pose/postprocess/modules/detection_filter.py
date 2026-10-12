"""Reject Bad Detections & Fill Gaps (video only)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
)

from ..base import CorrectionModule


class DetectionFilter(CorrectionModule):
    module_id = "DETECTION_FILTER"
    display_name = "Detection Filter"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        if context.process_detection_filter is None:
            raise RuntimeError("Detection Filter is unavailable outside a video bake")
        yield from context.process_detection_filter(settings)
