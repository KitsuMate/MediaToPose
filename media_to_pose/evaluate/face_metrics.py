"""Face expression quality: how well the rig follows the detected face.

``FaceSampler`` reads the face values the rig shows at each evaluated frame
(see ``retarget.solve.face_values``) and, given the detections, compares
them with what the detector saw:

- ``expression_fit``: mean absolute difference between the strength each
  mapped expression should have (the detection through the mapping's input
  range) and the strength the rig shows (its value through the output
  range);
- ``blink_recall``: share of detected blinks that close the rig's eye to at
  least ``BLINK_CLOSED``;
- ``face_jitter``: high-pass energy of the rig's face values, the distance
  of each value from the midpoint of its neighbours in time (RMS);
- ``asymmetry``: mean left/right difference on the rig in frames where the
  detection was symmetric.

All are in expression units (0 to 1).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..core.motion.face_filter import find_blinks
from ..core.requests import FaceOutput
from ..core.types import CanonicalFrame
from ..retarget.solve.face_values import (
    BLINK_EXPRESSIONS,
    FaceValueReader,
    mapped_expressions,
)

BLINK_CLOSED = 0.8
BLINK_DETECTED = 0.5
BLINK_MAX_SECONDS = 0.5
# Detected left and right closer than this count as symmetric.
SYMMETRIC = 0.1
_SIDES = (("Left", "Right"), ("_L", "_R"))


@dataclass
class FaceMeasures:
    expression_fit: float | None = None
    blink_recall: float | None = None
    face_jitter: float | None = None
    asymmetry: float | None = None
    worst_frames: dict[str, list[int]] = field(default_factory=dict)


def _pairs(names: list[str]) -> list[tuple[str, str]]:
    known = set(names)
    pairs = []
    for name in names:
        for left, right in _SIDES:
            if name.endswith(left):
                other = name[: -len(left)] + right
                if other in known:
                    pairs.append((name, other))
    return pairs


def _top(values: dict[int, float], count: int = 5) -> list[int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return [frame for frame, value in ordered[:count] if value > 0.0]


class FaceSampler:
    """Collect the rig's face values frame by frame during ``evaluate``."""

    def __init__(self, armature, entries, face_output: FaceOutput):
        self.armature = armature
        self.entries = entries
        self.face_output = FaceOutput(face_output)
        self.reader = FaceValueReader(armature, entries, self.face_output)
        self.frames: list[int] = []
        self.values: list[dict[str, float]] = []

    @property
    def active(self) -> bool:
        return bool(self.reader.blocks or self.reader.bones)

    def sample(self, frame: int) -> None:
        self.frames.append(frame)
        self.values.append(self.reader.read())

    def _jitter(self, measures: FaceMeasures) -> None:
        deviations = []
        per_frame: dict[int, float] = {}
        for index in range(1, len(self.values) - 1):
            before, now, after = self.values[index - 1 : index + 2]
            for name, value in now.items():
                if name in before and name in after:
                    deviation = abs(value - (before[name] + after[name]) * 0.5)
                    deviations.append(deviation * deviation)
                    frame = self.frames[index]
                    per_frame[frame] = max(per_frame.get(frame, 0.0), deviation)
        if deviations:
            measures.face_jitter = round(math.sqrt(statistics.fmean(deviations)), 4)
            measures.worst_frames["face_jitter"] = _top(per_frame)

    def finish(
        self, detections: Mapping[int, CanonicalFrame] | None, fps: float
    ) -> FaceMeasures:
        measures = FaceMeasures()
        if not self.active:
            return measures
        self._jitter(measures)
        if not detections:
            return measures
        shown = dict(zip(self.frames, self.values, strict=True))
        expected = {
            frame: mapped_expressions(
                self.entries, dict(found.blendshapes), self.face_output
            )[0]
            for frame, found in detections.items()
            if found.blendshapes and frame in shown
        }
        if not expected:
            return measures

        errors = []
        fit_per_frame: dict[int, float] = {}
        for frame, values in expected.items():
            frame_errors = [
                abs(value - shown[frame][name])
                for name, value in values.items()
                if name in shown[frame]
            ]
            if frame_errors:
                errors.extend(frame_errors)
                fit_per_frame[frame] = statistics.fmean(frame_errors)
        if errors:
            measures.expression_fit = round(statistics.fmean(errors), 4)
            measures.worst_frames["expression_fit"] = _top(fit_per_frame)

        frames = sorted(expected)
        times = [frame / fps for frame in frames]
        recalled = total = 0
        for name in BLINK_EXPRESSIONS[self.face_output]:
            series = [expected[frame].get(name, 0.0) for frame in frames]
            for start, end in find_blinks(
                series, times, BLINK_DETECTED, BLINK_MAX_SECONDS
            ):
                total += 1
                window = frames[max(start - 1, 0) : end + 2]
                closure = max(shown[frame].get(name, 0.0) for frame in window)
                recalled += closure >= BLINK_CLOSED
        if total:
            measures.blink_recall = round(recalled / total, 4)

        differences = [
            abs(shown[frame][left] - shown[frame][right])
            for frame, values in expected.items()
            for left, right in _pairs(sorted(values))
            if abs(values[left] - values[right]) < SYMMETRIC
            and left in shown[frame]
            and right in shown[frame]
        ]
        if differences:
            measures.asymmetry = round(statistics.fmean(differences), 4)
        return measures


__all__ = ["BLINK_CLOSED", "FaceMeasures", "FaceSampler"]
