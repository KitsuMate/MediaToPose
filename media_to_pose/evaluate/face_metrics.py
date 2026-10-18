"""Face expression quality: how well the rig follows the detected face.

``FaceSampler`` reads, at each evaluated frame, how strongly each mapped
target shows its expression: a shape key's value, or a face bone's offset or
angle along its mapped axis, through the mapping's output range. Given the
detections it compares them with the detected strengths (each value the
face output drives, through the mapping's input range):

- ``expression_fit``: mean absolute difference between detected and shown
  strengths;
- ``blink_recall``: share of detected blinks that close the rig's eye to at
  least ``BLINK_CLOSED``;
- ``face_jitter``: high-pass energy of the rig's face values, the distance
  of each value from the midpoint of its neighbours in time (RMS);
- ``asymmetry``: mean left/right difference on the rig in frames where the
  detection was symmetric.

All are in strength units: 0 to 1, or -1 to 1 for signed controls.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field

from mathutils import Vector

from ..core.face_ranges import finite, share_of_output, strength
from ..core.motion.face_filter import FaceFilterSettings, find_blinks
from ..core.requests import FaceOutput
from ..core.types import CanonicalFrame
from ..retarget.solve.face import expression_values, face_control_values
from ..rig.face_defaults import is_signed
from ..rig.face_schemes import matching_shape_blocks
from ..rig.face_space import bone_axis, face_axis, face_frame

BLINK_CLOSED = 0.8
# Blinks are found as Smooth Motion finds them when it keeps them.
BLINK_DETECTED = FaceFilterSettings.blink_threshold
BLINK_MAX_SECONDS = FaceFilterSettings.blink_max_seconds
# The blink of each face output, left then right.
BLINK_EXPRESSIONS = {
    FaceOutput.ARKIT: ("eyeBlinkLeft", "eyeBlinkRight"),
    FaceOutput.FACS: ("AU45_L", "AU45_R"),
    FaceOutput.VRM: ("blinkLeft", "blinkRight"),
    FaceOutput.BONES: ("FACE_LID_L", "FACE_LID_R"),
}
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


def _entry(entries, face_output: FaceOutput, name: str):
    key = name if face_output == FaceOutput.BONES else f"{face_output}_{name}"
    entry = entries.get(key)
    if not entry or not entry.enabled or not entry.target_name:
        return None
    return entry


def detected_strengths(
    entries, blendshapes: dict[str, float], face_output: FaceOutput
) -> dict[str, float]:
    """How strongly each mapped target should show its expression."""
    return {
        name: strength(entry, raw, signed=is_signed(entry.semantic_id))
        for name, raw in expression_values(blendshapes, face_output).items()
        if (entry := _entry(entries, face_output, name)) is not None
    }


def _bone_output(bone, entry, space) -> float:
    """A face bone control's value in its output units: the inverse of what
    bone output applies (metres along the face axis, degrees, or the custom
    property's value)."""
    if entry.channel_mode == "CUSTOM_PROPERTY" and entry.property_name:
        return finite(bone.get(entry.property_name, 0.0))
    if entry.channel_mode == "LOCATION":
        offset = bone.bone.matrix_local.to_3x3() @ Vector(bone.location)
        return offset.dot(face_axis(space, entry.axis))
    axis, angle = bone.matrix_basis.to_quaternion().to_axis_angle()
    return math.degrees(angle * Vector(axis).dot(bone_axis(entry.axis)))


class FaceSampler:
    """Collect the rig's face strengths frame by frame during ``evaluate``."""

    def __init__(self, armature, entries, face_output: FaceOutput):
        self.entries = entries
        self.face_output = FaceOutput(face_output)
        self.frames: list[int] = []
        self.values: list[dict[str, float]] = []
        bones = self.face_output == FaceOutput.BONES
        self.space = face_frame(armature, entries) if bones else None
        names = (
            face_control_values({})
            if bones
            else [
                key.removeprefix(f"{self.face_output}_")
                for key in entries
                if key.startswith(f"{self.face_output}_")
            ]
        )
        # Targets are looked up once: name -> (entry, pose bone or shape keys).
        self.targets = {}
        for name in names:
            entry = _entry(entries, self.face_output, name)
            if entry is None:
                continue
            if bones:
                target = armature.pose.bones.get(entry.target_name)
            else:
                target = [b for _o, b in matching_shape_blocks(armature, entry)]
            if target:
                self.targets[name] = (entry, target)

    @property
    def active(self) -> bool:
        return bool(self.targets)

    def sample(self, frame: int) -> None:
        values = {}
        for name, (entry, target) in self.targets.items():
            shown = (
                _bone_output(target, entry, self.space)
                if self.space is not None
                else max(block.value for block in target)
            )
            values[name] = share_of_output(entry, shown)
        self.frames.append(frame)
        self.values.append(values)

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
            frame: detected_strengths(
                self.entries, dict(found.blendshapes), self.face_output
            )
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


__all__ = ["BLINK_CLOSED", "FaceMeasures", "FaceSampler", "detected_strengths"]
