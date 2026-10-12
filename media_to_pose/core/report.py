"""Collect what each clean-up step did during one capture.

The capture session activates a ``CaptureReport`` while it drives the
retarget pipeline. Stages that have something to report call
``record_step`` or ``record_frames``; outside an active report those calls
do nothing, so the pipeline needs no extra arguments.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_ACTIVE: ContextVar[CaptureReport | None] = ContextVar(
    "media_to_pose_capture_report", default=None
)


def frame_ranges(frames: Iterable[int]) -> list[tuple[int, int]]:
    """Collapse frame numbers into sorted inclusive ``(first, last)`` ranges."""
    ranges: list[tuple[int, int]] = []
    for frame in sorted({int(value) for value in frames}):
        if ranges and frame == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], frame)
        else:
            ranges.append((frame, frame))
    return ranges


def _plain(value: Any) -> Any:
    """Convert statistics to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 6)
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return str(value)


@dataclass
class StepReport:
    """One clean-up step: its settings and what it changed."""

    module_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "parameters": _plain(self.parameters),
            "statistics": _plain(self.statistics),
        }


@dataclass
class CaptureReport:
    """Statistics gathered while one capture ran.

    ``source_frame_offset`` is the first source frame of the processed
    range, so frame numbers recorded here are source frames (0-based).
    """

    source_frame_offset: int = 0
    steps: dict[str, StepReport] = field(default_factory=dict)
    rejected_frames: set[int] = field(default_factory=set)
    filled_frames: set[int] = field(default_factory=set)
    contact_frames: set[int] = field(default_factory=set)

    def step(self, module_id: str) -> StepReport:
        report = self.steps.get(module_id)
        if report is None:
            report = self.steps[module_id] = StepReport(module_id)
        return report

    @contextmanager
    def active(self) -> Iterator[CaptureReport]:
        """Make this report the target of ``record_*`` calls."""
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)


def active_report() -> CaptureReport | None:
    return _ACTIVE.get()


def record_step(
    module_id: str,
    parameters: Mapping[str, Any] | None = None,
    **statistics: Any,
) -> None:
    """Add parameters and statistics for ``module_id`` to the active report."""
    report = _ACTIVE.get()
    if report is None:
        return
    step = report.step(module_id)
    if parameters:
        step.parameters.update(parameters)
    step.statistics.update(statistics)


def record_frames(kind: str, sample_indices: Iterable[int]) -> None:
    """Record sample indices of the processed range as source frames.

    ``kind`` is ``"rejected"``, ``"filled"`` or ``"contact"``.
    """
    report = _ACTIVE.get()
    if report is None:
        return
    target = {
        "rejected": report.rejected_frames,
        "filled": report.filled_frames,
        "contact": report.contact_frames,
    }[kind]
    target.update(int(index) + report.source_frame_offset for index in sample_indices)


__all__ = [
    "CaptureReport",
    "StepReport",
    "active_report",
    "frame_ranges",
    "record_frames",
    "record_step",
]
