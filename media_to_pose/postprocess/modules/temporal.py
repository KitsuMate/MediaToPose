"""Smooth Motion & Foot Contact (video only)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
)

from ..base import CorrectionModule


class TemporalCorrection(CorrectionModule):
    module_id = "TEMPORAL_CORRECTION"
    display_name = "Temporal Correction"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        yield from context.process_temporal(settings)
