"""Provider-neutral contact detection and grounded root trajectory repair.

The routines in this module deliberately have no Blender data dependency.
They operate on already-retargeted target-space samples, because a provider
ankle is only an observation whereas the evaluated rig foot is the point that
must remain planted.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from statistics import median

from mathutils import Quaternion, Vector


@dataclass
class ContactPoint:
    position: Vector
    image: Vector
    confidence: float
    direction: Vector | None = None
    motion_position: Vector | None = None

    def orientation_score(self, up: Vector) -> float:
        if self.direction is None or self.direction.length <= 1.0e-8:
            return 1.0
        return 1.0 - min(abs(self.direction.normalized().dot(up)), 1.0)


@dataclass
class GroundFrame:
    timestamp: float
    root: Vector
    points: dict[str, ContactPoint] = field(default_factory=dict)
    joints: dict[str, Vector] = field(default_factory=dict)
    orientations: dict[str, Quaternion] = field(default_factory=dict)
    joint_confidence: dict[str, float] = field(default_factory=dict)
    root_confidence: float = 0.0


@dataclass
class ContactSegment:
    point: str
    start: int
    end: int
    anchor: Vector
    weights: list[float]


@dataclass
class GroundingResult:
    roots: list[Vector]
    contacts: dict[str, list[float]]
    segments: list[ContactSegment]
    ground_height: float
    warnings: list[str] = field(default_factory=list)


def _project_horizontal(value: Vector, up: Vector) -> Vector:
    return value - up * value.dot(up)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    at = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    lo = int(math.floor(at))
    hi = int(math.ceil(at))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def _bridge_and_prune(states: list[bool]) -> list[bool]:
    result = states[:]
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        if start > 0 and index < len(result) and index - start <= 2:
            result[start:index] = [True] * (index - start)
    index = 0
    while index < len(result):
        if not result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and result[index]:
            index += 1
        if index - start < 3:
            result[start:index] = [False] * (index - start)
    return result


def detect_contacts(
    frames: Sequence[GroundFrame],
    body_height: float,
    up: Vector,
) -> tuple[dict[str, list[float]], float]:
    up = up.normalized()
    names = sorted({name for frame in frames for name in frame.points})
    low_samples = [
        point.position.dot(up)
        for frame in frames
        for point in frame.points.values()
        if point.confidence >= 0.45
    ]
    ground = _percentile(low_samples, 0.08)
    height = max(body_height, 1.0e-6)
    contacts: dict[str, list[float]] = {}
    for name in names:
        raw = [False] * len(frames)
        scores = [0.0] * len(frames)
        entering = leaving = 0
        active = False
        for index, frame in enumerate(frames):
            point = frame.points.get(name)
            if point is None:
                candidate = False
                score = 0.0
            else:
                previous = frames[index - 1].points.get(name) if index else None
                dt = (
                    max(
                        frame.timestamp - frames[index - 1].timestamp,
                        1.0e-4,
                    )
                    if index
                    else 1.0 / 30.0
                )
                world_velocity = (
                    (point.position - previous.position) / dt
                    if previous is not None
                    else Vector()
                )
                relative_velocity = (
                    (point.motion_position - previous.motion_position) / dt
                    if previous is not None
                    and point.motion_position is not None
                    and previous.motion_position is not None
                    else world_velocity
                )
                image_speed = (
                    (point.image - previous.image).length / dt
                    if previous is not None
                    else 0.0
                )
                relative_height = (point.position.dot(up) - ground) / height
                # A planted foot is stationary in world space while a noisy
                # root can make it look stationary only relative to the body.
                # Accept the quieter observation for contact classification;
                # the later joint optimizer reconciles both against the
                # world-space plant anchor.
                horizontal_speed = (
                    min(
                        _project_horizontal(world_velocity, up).length,
                        _project_horizontal(relative_velocity, up).length,
                    )
                    / height
                )
                vertical_speed = (
                    min(
                        abs(world_velocity.dot(up)),
                        abs(relative_velocity.dot(up)),
                    )
                    / height
                )
                # Contact evidence is intentionally redundant. A stylized
                # detector can have weak 3D depth while image motion remains
                # useful, or vice versa.
                score = (
                    0.30 * max(0.0, 1.0 - relative_height / 0.055)
                    + 0.24 * max(0.0, 1.0 - horizontal_speed / 0.42)
                    + 0.18 * max(0.0, 1.0 - vertical_speed / 0.24)
                    + 0.10 * max(0.0, 1.0 - image_speed / 0.75)
                    + 0.12 * min(max(point.confidence, 0.0), 1.0)
                    + 0.06 * point.orientation_score(up)
                )
                candidate = relative_height <= 0.09 and score >= (
                    0.54 if active else 0.62
                )
            scores[index] = score
            if candidate:
                entering += 1
                leaving = 0
            else:
                leaving += 1
                entering = 0
            if not active and entering >= 3:
                active = True
                for back in range(max(0, index - 2), index + 1):
                    raw[back] = True
            elif active and leaving >= 2:
                active = False
                raw[index - 1] = False
            if active:
                raw[index] = True
        raw = _bridge_and_prune(raw)
        contacts[name] = [scores[i] if state else 0.0 for i, state in enumerate(raw)]
    return contacts, ground


def build_segments(
    frames: Sequence[GroundFrame],
    contacts: dict[str, list[float]],
    ground_height: float | None = None,
    up: Vector | None = None,
) -> list[ContactSegment]:
    segments: list[ContactSegment] = []
    for name, weights in contacts.items():
        index = 0
        while index < len(weights):
            if weights[index] <= 0.0:
                index += 1
                continue
            start = index
            while index < len(weights) and weights[index] > 0.0:
                index += 1
            end = index - 1
            positions = [frames[i].points[name].position for i in range(start, end + 1)]
            anchor = Vector(
                tuple(median([value[axis] for value in positions]) for axis in range(3))
            )
            if ground_height is not None and up is not None:
                anchor += up * (ground_height - anchor.dot(up))
            segments.append(
                ContactSegment(name, start, end, anchor, weights[start:index])
            )
    return segments


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def stabilize_root(
    frames: Sequence[GroundFrame],
    contacts: dict[str, list[float]],
    segments: Sequence[ContactSegment],
    body_height: float,
    up: Vector,
    strength: float,
    *,
    iterations: int = 10,
) -> Iterator[tuple[int, list[Vector]]]:
    """IRLS banded relaxation for root observations and plant constraints."""
    if not frames:
        return
    up = up.normalized()
    raw = [frame.root.copy() for frame in frames]
    timestamps = [frame.timestamp for frame in frames]
    confidence = [frame.root_confidence for frame in frames]
    roots = [value.copy() for value in raw]
    contact_targets: list[list[tuple[Vector, float]]] = [[] for _ in frames]
    for segment in segments:
        duration = max(
            timestamps[segment.end] - timestamps[segment.start],
            1.0e-4,
        )
        transition = max(2, int(round(0.1 * len(segment.weights) / duration)))
        for offset, index in enumerate(range(segment.start, segment.end + 1)):
            point = frames[index].points[segment.point]
            local_point = point.position - frames[index].root
            edge = min(offset + 1, segment.end - index + 1) / max(transition, 1)
            weight = (
                28.0 * _smoothstep(edge) * max(contacts[segment.point][index], 0.35)
            )
            contact_targets[index].append((segment.anchor - local_point, weight))
    data_base = 1.5
    velocity_weight = 2.0 + 4.0 * min(max(strength, 0.0), 1.0)
    acceleration_weight = 1.0 + 3.0 * min(max(strength, 0.0), 1.0)
    huber = max(body_height, 1.0e-6) * 0.12
    for iteration in range(iterations):
        next_roots = [value.copy() for value in roots]
        for index in range(len(roots)):
            numerator = Vector()
            denominator = 0.0
            residual = (roots[index] - raw[index]).length
            huber_weight = 1.0 if residual <= huber else huber / max(residual, 1.0e-8)
            # With both feet airborne there is no grounded evidence that may
            # legitimately flatten a jump. Keep observed vertical travel
            # comparatively strong until contact resumes.
            airborne_boost = 2.5 if not contact_targets[index] else 1.0
            data_weight = (
                data_base * airborne_boost * max(confidence[index], 0.08) * huber_weight
            )
            numerator += raw[index] * data_weight
            denominator += data_weight
            for target, weight in contact_targets[index]:
                numerator += target * weight
                denominator += weight
            if index:
                numerator += roots[index - 1] * velocity_weight
                denominator += velocity_weight
            if index + 1 < len(roots):
                numerator += roots[index + 1] * velocity_weight
                denominator += velocity_weight
            if 0 < index < len(roots) - 1:
                midpoint = (roots[index - 1] + roots[index + 1]) * 0.5
                numerator += midpoint * acceleration_weight
                denominator += acceleration_weight
            next_roots[index] = numerator / max(denominator, 1.0e-8)
        # The first reliable frame defines the trajectory origin.
        next_roots[0] = raw[0].copy()
        roots = next_roots
        for index, contact_evidence in enumerate(contact_targets):
            if contact_evidence:
                continue
            solved_vertical = roots[index].dot(up)
            observed_vertical = raw[index].dot(up)
            roots[index] += up * (
                (observed_vertical * 0.85 + solved_vertical * 0.15) - solved_vertical
            )
        roots[0] = raw[0].copy()
        yield iteration + 1, [value.copy() for value in roots]


def analyze_grounding(
    frames: Sequence[GroundFrame],
    body_height: float,
    up: Vector,
) -> GroundingResult:
    contacts, ground = detect_contacts(frames, body_height, up)
    planted_heights = [
        frame.points[name].position.dot(up)
        for name, weights in contacts.items()
        for index, frame in enumerate(frames)
        if weights[index] > 0.0 and name in frame.points
    ]
    world_ground = _percentile(planted_heights, 0.08) if planted_heights else ground
    segments = build_segments(frames, contacts, world_ground, up.normalized())
    warnings = []
    if not segments:
        warnings.append(
            "No reliable foot plants were detected; correcting root continuity only"
        )
    return GroundingResult(
        roots=[frame.root.copy() for frame in frames],
        contacts=contacts,
        segments=segments,
        ground_height=world_ground,
        warnings=warnings,
    )
