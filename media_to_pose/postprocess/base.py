"""The clean-up module interface and the state shared between frames."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
)

from mathutils import Vector

from ..core.constants import PROFILE_ARP, PROFILE_GENERIC, PROFILE_RIGIFY
from ..retarget.target_pose import TargetPose


def _smoothstep(start: float, end: float, value: float) -> float:
    if end <= start + 1.0e-8:
        return 1.0 if value >= end else 0.0
    t = min(max((value - start) / (end - start), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class PostProcessingSession:
    """Small continuity state shared by consecutive frame solves."""

    _last_timestamp_ms: int | None = None
    _forearm_twist_phase: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self._last_timestamp_ms = None
        self._forearm_twist_phase.clear()

    def prepare_frame(self, timestamp_ms: int) -> None:
        timestamp_ms = int(timestamp_ms)
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            self._forearm_twist_phase.clear()
        self._last_timestamp_ms = timestamp_ms

    def unwrap_forearm_twist(self, side: str, angle: float) -> float:
        """Keep a circular palm-roll observation on one continuous phase."""
        previous = self._forearm_twist_phase.get(side)
        if previous is not None:
            angle += math.tau * round((previous - angle) / math.tau)
        self._forearm_twist_phase[side] = angle
        return angle


@dataclass
class SequenceProcessingContext:
    """Provider-neutral entry point for a complete baked-video sequence."""

    samples: Sequence[Any]
    timestamps: Sequence[float]
    root_motion_mode: str
    process_temporal: Callable[[Any], Iterator[Any]]
    process_camera_motion: Callable[[Any], Iterator[Any]] | None = None
    process_detection_filter: Callable[[Any], Iterator[Any]] | None = None
    process_rohm: Callable[[Any], Iterator[Any]] | None = None


class CorrectionModule:
    module_id = ""
    display_name = ""
    supported_profiles = {
        PROFILE_ARP,
        PROFILE_RIGIFY,
        PROFILE_GENERIC,
    }
    required_semantics: tuple[str, ...] = ()
    allows_endpoint_motion = False
    scope = "FRAME"

    def validate(self, pose_or_armature, settings) -> list[str]:
        return []

    def reset(self, session: PostProcessingSession) -> None:
        session.reset()

    def process_frame(
        self,
        pose: TargetPose,
        settings,
        session: PostProcessingSession,
    ) -> list[str]:
        raise NotImplementedError

    def process_sequence(
        self,
        context: SequenceProcessingContext,
        settings,
    ) -> Iterator[Any]:
        raise NotImplementedError


def _curve_sample(points: list[Vector], count: int = 65) -> list[Vector]:
    """Sample a C1 Catmull-Rom centerline through provider joints."""
    if len(points) < 2:
        return list(points)
    result = []
    segments = len(points) - 1
    for sample_index in range(count):
        value = segments * sample_index / max(count - 1, 1)
        index = min(int(value), segments - 1)
        t = value - index
        p0 = points[max(index - 1, 0)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(index + 2, len(points) - 1)]
        result.append(
            0.5
            * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
            )
        )
    return result


def _hermite_sample(
    start: Vector,
    end: Vector,
    start_tangent: Vector,
    end_tangent: Vector,
    count: int = 65,
) -> list[Vector]:
    result = []
    for index in range(count):
        t = index / max(count - 1, 1)
        t2, t3 = t * t, t * t * t
        result.append(
            (2.0 * t3 - 3.0 * t2 + 1.0) * start
            + (t3 - 2.0 * t2 + t) * start_tangent
            + (-2.0 * t3 + 3.0 * t2) * end
            + (t3 - t2) * end_tangent
        )
    return result


def _arc_point(samples: list[Vector], fraction: float) -> Vector:
    if len(samples) < 2:
        return samples[0].copy()
    lengths = [0.0]
    for first, second in zip(samples, samples[1:], strict=False):
        lengths.append(lengths[-1] + (second - first).length)
    target = min(max(fraction, 0.0), 1.0) * lengths[-1]
    for index in range(1, len(lengths)):
        if lengths[index] < target:
            continue
        span = lengths[index] - lengths[index - 1]
        alpha = 0.0 if span < 1.0e-8 else (target - lengths[index - 1]) / span
        return samples[index - 1].lerp(samples[index], alpha)
    return samples[-1].copy()
