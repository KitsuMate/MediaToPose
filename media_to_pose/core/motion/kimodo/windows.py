"""Splitting a generation into Kimodo windows of at most 300 frames.

Upstream generates one prompt at a time (``Kimodo._multiprompt``). Each window
after the first starts with the previous window's last ``TRANSITION_FRAMES``
frames, pinned by full-body and effector guides, and the overlap is
cross-faded. Windows never span two prompts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...errors import InvalidRequest
from . import rep

TRANSITION_FRAMES = 5
MAX_PROMPT_FRAMES = rep.MAX_WINDOW_FRAMES  # 10 s at 30 fps


@dataclass(frozen=True)
class Segment:
    """One prompt over 30 fps frames ``start`` (inclusive) to ``stop``."""

    start: int
    stop: int
    prompt: str = ""

    @property
    def frame_count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Window:
    """Frames ``start`` to ``stop`` of the whole generation, one prompt.

    When ``transition`` is set, the first ``TRANSITION_FRAMES`` frames overlap
    the previous window and are pinned to its result.
    """

    segment: int
    start: int
    stop: int
    transition: bool

    @property
    def frame_count(self) -> int:
        return self.stop - self.start


def _validate(segments: Sequence[Segment], total: int) -> None:
    if not segments:
        raise InvalidRequest("A generation needs at least one segment")
    if segments[0].start != 0 or segments[-1].stop != total:
        raise InvalidRequest(f"Segments must cover frames 0 to {total}")
    for before, after in zip(segments, segments[1:], strict=False):
        if before.stop != after.start:
            raise InvalidRequest("Segments must follow each other without gaps")
    for segment in segments:
        if segment.frame_count < 1:
            raise InvalidRequest("Every segment needs at least one frame")


def plan_windows(total: int, segments: Sequence[Segment] | None = None) -> list[Window]:
    """Windows covering frames ``0..total`` in order.

    Long segments are split into near-equal windows so that each, including
    its transition frames, has at most 300 frames.
    """
    if total < 2:
        raise InvalidRequest("A generation needs at least two frames")
    plan = list(segments) if segments else [Segment(0, total)]
    _validate(plan, total)
    windows: list[Window] = []
    for index, segment in enumerate(plan):
        transition = bool(windows)
        length = segment.frame_count
        # Every window after the very first carries the transition frames.
        first_capacity = rep.MAX_WINDOW_FRAMES - (
            TRANSITION_FRAMES if transition else 0
        )
        later_capacity = rep.MAX_WINDOW_FRAMES - TRANSITION_FRAMES
        count = (
            1
            if length <= first_capacity
            else 1 + int(np.ceil((length - first_capacity) / later_capacity))
        )
        bounds = np.linspace(segment.start, segment.stop, count + 1).round().astype(int)
        for chunk, (start, stop) in enumerate(
            zip(bounds[:-1], bounds[1:], strict=True)
        ):
            carries = transition or chunk > 0
            begin = int(start) - (TRANSITION_FRAMES if carries else 0)
            windows.append(
                Window(index, max(begin, 0), int(stop), carries and begin >= 0)
            )
    for window in windows:
        if window.frame_count > rep.MAX_WINDOW_FRAMES or window.frame_count < 2:
            raise InvalidRequest(
                f"Window {window.start}..{window.stop} does not fit the model"
            )
    return windows


def crossfade(
    previous: np.ndarray, current: np.ndarray, overlap: int = TRANSITION_FRAMES
) -> np.ndarray:
    """Blend the last ``overlap`` frames of ``previous`` into the first of
    ``current`` (unnormalised features), linearly from old to new."""
    if overlap <= 0:
        return np.concatenate((previous, current))
    # Weights run from near 0 (keep previous) to near 1 (take current).
    weights = np.linspace(0.0, 1.0, overlap + 2)[1:-1]
    blended = (
        previous[-overlap:] * (1.0 - weights[:, None])
        + current[:overlap] * weights[:, None]
    )
    return np.concatenate((previous[:-overlap], blended, current[overlap:]))


__all__ = [
    "MAX_PROMPT_FRAMES",
    "TRANSITION_FRAMES",
    "Segment",
    "Window",
    "crossfade",
    "plan_windows",
]
