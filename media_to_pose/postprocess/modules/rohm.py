"""RoHM Cleanup (video only, optional model)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
)

from ..base import CorrectionModule


class RohmCorrection(CorrectionModule):
    module_id = "ROHM_CORRECTION"
    display_name = "RoHM Correction"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        if context.process_rohm is None:
            raise RuntimeError("RoHM Correction is unavailable outside a video bake")
        yield from context.process_rohm(settings)
