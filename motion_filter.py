"""Confidence and temporal-quality filtering before motion reconstruction."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

from .rohm_correction import SMPLX_SEMANTICS

_LIMBS = (
    ("UPPER_LEG_L", "LOWER_LEG_L", "FOOT_L", "TOE_L"),
    ("UPPER_LEG_R", "LOWER_LEG_R", "FOOT_R", "TOE_R"),
    ("SHOULDER_L", "UPPER_ARM_L", "LOWER_ARM_L", "HAND_L"),
    ("SHOULDER_R", "UPPER_ARM_R", "LOWER_ARM_R", "HAND_R"),
)
MANUAL_MASK_SCOPES = {
    "BODY": SMPLX_SEMANTICS,
    "ARM_L": ("SHOULDER_L", "UPPER_ARM_L", "LOWER_ARM_L", "HAND_L"),
    "ARM_R": ("SHOULDER_R", "UPPER_ARM_R", "LOWER_ARM_R", "HAND_R"),
    "LEG_L": ("UPPER_LEG_L", "LOWER_LEG_L", "FOOT_L", "TOE_L"),
    "LEG_R": ("UPPER_LEG_R", "LOWER_LEG_R", "FOOT_R", "TOE_R"),
}
_CORE = (
    "PELVIS",
    "SPINE_01",
    "SPINE_02",
    "CHEST",
    "NECK",
    "HEAD",
    "SHOULDER_L",
    "SHOULDER_R",
    "UPPER_LEG_L",
    "UPPER_LEG_R",
)
_LOW_LIMB_CONFIDENCE = 0.35
_LOW_FRAME_CONFIDENCE = 0.25
_MIN_ROTATION_SPIKE = math.radians(30.0)
_MIN_POSITION_SPIKE = 0.10
_MIN_SOURCE_TEMPORAL_SPIKE = 0.15
_MIN_SOURCE_GEOMETRY_SPIKE = 0.25
_MAX_SOURCE_COLLAPSE_RATIO = 0.55
SOURCE_BODY_JOINTS = (
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)
_SOURCE_SEGMENTS = (
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
)


@dataclass(frozen=True)
class MotionFilterStats:
    total_observations: int
    native_missing_observations: int
    rejected_observations: int
    rejection_budget: int
    rejected_limbs: int
    rejected_frames: int
    rejected: frozenset[tuple[int, str]]
    rejected_root_frames: frozenset[int]

    @property
    def observed_fraction(self) -> float:
        if not self.total_observations:
            return 0.0
        missing = self.native_missing_observations + self.rejected_observations
        return max(0.0, 1.0 - missing / self.total_observations)


@dataclass(frozen=True)
class _Candidate:
    severity: float
    observations: frozenset[tuple[int, str]]
    frame_index: int
    whole_frame: bool


def parse_source_frame_ranges(value: str, frame_count: int) -> frozenset[int]:
    """Parse 1-based source frames/ranges and return in-clip zero-based indices."""
    text = str(value or "").strip()
    if not text:
        return frozenset()
    count = max(int(frame_count), 0)
    selected: set[int] = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("empty item in additional bad source frames")
        parts = [part.strip() for part in token.split("-")]
        try:
            if len(parts) == 1:
                start = end = int(parts[0])
            elif len(parts) == 2 and all(parts):
                start, end = (int(part) for part in parts)
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"invalid source-frame range: {token}") from exc
        if start <= 0 or end <= 0 or end < start:
            raise ValueError(f"invalid source-frame range: {token}")
        selected.update(range(max(start, 1) - 1, min(end, count)))
    return frozenset(selected)


def format_source_frame_ranges(indices: Iterable[int]) -> str:
    """Format zero-based source-frame indices as compact 1-based ranges."""
    values = sorted({int(index) + 1 for index in indices if int(index) >= 0})
    if not values:
        return "None"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def apply_manual_motion_rejections(
    samples: Sequence[object],
    frame_indices: Iterable[int],
    scope: str,
) -> frozenset[tuple[int, str]]:
    """Apply an explicit mask independently of the automatic rejection budget."""
    scope_id = str(scope).upper()
    try:
        semantics = MANUAL_MASK_SCOPES[scope_id]
    except KeyError as exc:
        raise ValueError(f"unsupported manual motion-mask scope: {scope}") from exc
    rejected: set[tuple[int, str]] = set()
    for frame_index in sorted(set(int(value) for value in frame_indices)):
        if frame_index < 0 or frame_index >= len(samples):
            continue
        sample = samples[frame_index]
        for semantic in semantics:
            if _confidence(sample, semantic) > 0.0:
                rejected.add((frame_index, semantic))
            sample.rohm_confidence[semantic] = 0.0
            channel = _channel(sample, semantic)
            if channel is not None:
                channel.confidence = 0.0
        if scope_id == "BODY":
            sample.rohm_root_confidence = 0.0
    return frozenset(rejected)


def _confidence(sample, semantic: str) -> float:
    return min(
        max(float(getattr(sample, "rohm_confidence", {}).get(semantic, 0.0)), 0.0),
        1.0,
    )


def _channel_item(sample, semantic: str):
    return next(
        (
            (name, value)
            for name, value in getattr(sample, "bones", {}).items()
            if value.semantic_id == semantic and value.key_rotation
        ),
        None,
    )


def _channel(sample, semantic: str):
    item = _channel_item(sample, semantic)
    return item[1] if item is not None else None


def _copy_channel(sample, semantic: str, source_sample):
    item = _channel_item(source_sample, semantic)
    if item is None:
        return None
    name, source = item
    target = copy.copy(source)
    target.location = source.location.copy()
    target.rotation = source.rotation.copy()
    target.scale = source.scale.copy()
    target.properties = dict(source.properties)
    target.confidence = 0.0
    sample.bones[name] = target
    return target


def _robust_limit(values: Sequence[float], floor: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return floor
    center = median(finite)
    spread = 1.4826 * median(abs(value - center) for value in finite)
    return max(floor, center + 3.5 * max(spread, 1.0e-6))


def _jump_scores(
    samples: Sequence[object],
    timestamps: Sequence[float],
) -> dict[tuple[int, str], float]:
    """Return robust two-sided spike scores; sustained movement is preserved."""
    if len(samples) < 3:
        return {}
    result: dict[tuple[int, str], float] = {}
    for semantic in SMPLX_SEMANTICS:
        rotation_residuals = [0.0] * len(samples)
        position_residuals = [0.0] * len(samples)
        valid = [False] * len(samples)
        for index in range(1, len(samples) - 1):
            left = _channel(samples[index - 1], semantic)
            current = _channel(samples[index], semantic)
            right = _channel(samples[index + 1], semantic)
            if left is None or current is None or right is None:
                continue
            span = float(timestamps[index + 1]) - float(timestamps[index - 1])
            if span <= 1.0e-8:
                continue
            factor = min(
                max(
                    (float(timestamps[index]) - float(timestamps[index - 1])) / span,
                    0.0,
                ),
                1.0,
            )
            predicted_rotation = left.rotation.slerp(
                right.rotation, factor
            ).normalized()
            rotation_residuals[index] = predicted_rotation.rotation_difference(
                current.rotation
            ).angle
            offsets = [
                getattr(sample, "rohm_joint_offsets", {}).get(semantic)
                for sample in samples[index - 1 : index + 2]
            ]
            if all(value is not None for value in offsets):
                predicted_position = offsets[0].lerp(offsets[2], factor)
                body_height = max(
                    float(getattr(samples[index], "body_height", 0.0)), 1.0e-4
                )
                position_residuals[index] = (
                    offsets[1] - predicted_position
                ).length / body_height
            valid[index] = True
        rotation_limit = _robust_limit(
            [
                rotation_residuals[index]
                for index in range(1, len(samples) - 1)
                if valid[index]
            ],
            _MIN_ROTATION_SPIKE,
        )
        position_limit = _robust_limit(
            [
                position_residuals[index]
                for index in range(1, len(samples) - 1)
                if valid[index]
            ],
            _MIN_POSITION_SPIKE,
        )
        for index in range(1, len(samples) - 1):
            if not valid[index] or _confidence(samples[index], semantic) <= 0.0:
                continue
            rotation_score = rotation_residuals[index] / rotation_limit
            position_score = position_residuals[index] / position_limit
            score = max(rotation_score, position_score)
            if score > 1.0:
                result[(index, semantic)] = score
    return result


def _midpoint(points: dict[str, object], left: str, right: str):
    if left not in points or right not in points:
        return None
    return (points[left] + points[right]) * 0.5


def _normalized_source_pose(sample, attribute: str):
    points = getattr(sample, attribute, {})
    hip = _midpoint(points, "LEFT_HIP", "RIGHT_HIP")
    shoulder = _midpoint(points, "LEFT_SHOULDER", "RIGHT_SHOULDER")
    if hip is None or shoulder is None:
        return None
    shoulder_width = (points["LEFT_SHOULDER"] - points["RIGHT_SHOULDER"]).length
    scale = (shoulder - hip).length + shoulder_width
    if not math.isfinite(scale) or scale <= 1.0e-6:
        return None
    return {name: (point - hip) / scale for name, point in points.items()}


def _source_pose_residual(current, predicted) -> float:
    shared = current.keys() & predicted.keys()
    if len(shared) < 8:
        return 0.0
    return median((current[name] - predicted[name]).length for name in shared)


def _source_temporal_residuals(
    poses: Sequence[object], timestamps: Sequence[float]
) -> list[float]:
    residuals = [0.0] * len(poses)
    for index in range(1, len(poses) - 1):
        left, current, right = poses[index - 1 : index + 2]
        if left is None or current is None or right is None:
            continue
        span = float(timestamps[index + 1]) - float(timestamps[index - 1])
        if span <= 1.0e-8:
            continue
        factor = min(
            max(
                (float(timestamps[index]) - float(timestamps[index - 1])) / span,
                0.0,
            ),
            1.0,
        )
        predicted = {
            name: left[name].lerp(right[name], factor)
            for name in left.keys() & right.keys()
        }
        residuals[index] = _source_pose_residual(current, predicted)
    return residuals


def _source_geometry(sample) -> tuple[float, ...]:
    points = getattr(sample, "source_body_3d", {})
    lengths = []
    for start, end in _SOURCE_SEGMENTS:
        value = (
            (points[start] - points[end]).length
            if start in points and end in points
            else math.nan
        )
        lengths.append(value)
    finite = [value for value in lengths if math.isfinite(value) and value > 1.0e-6]
    scale = median(finite) if finite else 0.0
    if scale <= 1.0e-6:
        return tuple(math.nan for _value in lengths)
    return tuple(value / scale for value in lengths)


def _source_body_scale(sample, attribute: str) -> float:
    points = getattr(sample, attribute, {})
    hip = _midpoint(points, "LEFT_HIP", "RIGHT_HIP")
    shoulder = _midpoint(points, "LEFT_SHOULDER", "RIGHT_SHOULDER")
    if hip is None or shoulder is None:
        return math.nan
    shoulder_width = (points["LEFT_SHOULDER"] - points["RIGHT_SHOULDER"]).length
    hip_width = (points["LEFT_HIP"] - points["RIGHT_HIP"]).length
    value = (shoulder - hip).length + shoulder_width + hip_width
    return value if math.isfinite(value) and value > 1.0e-6 else math.nan


def _collapse_scores(samples: Sequence[object]) -> list[float]:
    """Score sustained scale collapse, including high-confidence runs.

    Two-sided motion residuals intentionally preserve sustained movement, but
    that also makes the interior of a multi-frame detector collapse look
    temporally smooth.  Absolute 3D torso scale is therefore compared with the
    robust clip median.  Only a severe shrink is considered; ordinary camera
    motion and gradual perspective scale changes remain below this threshold.
    """
    scales = [_source_body_scale(sample, "source_body_3d") for sample in samples]
    finite = [value for value in scales if math.isfinite(value)]
    if not finite:
        return [0.0] * len(samples)
    reference = median(finite)
    threshold = reference * _MAX_SOURCE_COLLAPSE_RATIO
    if threshold <= 1.0e-8:
        return [0.0] * len(samples)
    return [
        threshold / max(value, 1.0e-8)
        if math.isfinite(value) and value < threshold
        else 0.0
        for value in scales
    ]


def _source_frame_scores(
    samples: Sequence[object], timestamps: Sequence[float]
) -> dict[int, float]:
    """Detect confidently wrong source poses before rig constraints hide them."""
    poses_3d = [_normalized_source_pose(sample, "source_body_3d") for sample in samples]
    poses_2d = [_normalized_source_pose(sample, "source_body_2d") for sample in samples]
    residuals_3d = _source_temporal_residuals(poses_3d, timestamps)
    residuals_2d = _source_temporal_residuals(poses_2d, timestamps)
    limit_3d = _robust_limit(residuals_3d, _MIN_SOURCE_TEMPORAL_SPIKE)
    limit_2d = _robust_limit(residuals_2d, _MIN_SOURCE_TEMPORAL_SPIKE)

    geometries = [_source_geometry(sample) for sample in samples]
    reference = tuple(
        (
            median(
                geometry[index]
                for geometry in geometries
                if math.isfinite(geometry[index])
            )
            if any(math.isfinite(geometry[index]) for geometry in geometries)
            else math.nan
        )
        for index in range(len(_SOURCE_SEGMENTS))
    )
    geometry_errors = []
    for geometry in geometries:
        deviations = [
            abs(value - expected)
            for value, expected in zip(geometry, reference, strict=True)
            if math.isfinite(value) and math.isfinite(expected)
        ]
        geometry_errors.append(median(deviations) if deviations else 0.0)
    geometry_limit = _robust_limit(
        geometry_errors,
        _MIN_SOURCE_GEOMETRY_SPIKE,
    )
    collapse_scores = _collapse_scores(samples)

    result = {}
    for index in range(len(samples)):
        temporal_score = min(
            residuals_3d[index] / limit_3d,
            residuals_2d[index] / limit_2d,
        )
        geometry_score = geometry_errors[index] / geometry_limit
        score = max(temporal_score, geometry_score, collapse_scores[index])
        if math.isfinite(score) and score > 1.0:
            result[index] = score
    return result


def _observed(
    samples: Sequence[object], frame_index: int, semantics: Iterable[str]
) -> frozenset[tuple[int, str]]:
    return frozenset(
        (frame_index, semantic)
        for semantic in semantics
        if _confidence(samples[frame_index], semantic) > 0.0
    )


def filter_motion_observations(
    samples: Sequence[object],
    timestamps: Sequence[float],
    *,
    reject_unreliable_limbs: bool,
    reject_unusable_frames: bool,
    maximum_rejected_fraction: float,
) -> MotionFilterStats:
    """Mask the worst detected observations without spending native gaps.

    The shared budget is measured in joint-frame observations. Rejecting a
    whole frame therefore costs up to 22 observations while rejecting a limb
    costs only its currently observed joints. Native missing observations are
    reported but never consume the filter-created rejection budget.
    """
    if len(samples) != len(timestamps):
        raise ValueError("motion-filter samples and timestamps must match")
    total = len(samples) * len(SMPLX_SEMANTICS)
    native_missing = sum(
        _confidence(sample, semantic) <= 0.0
        for sample in samples
        for semantic in SMPLX_SEMANTICS
    )
    budget = min(
        max(
            int(
                math.floor(total * min(max(float(maximum_rejected_fraction), 0.0), 1.0))
            ),
            0,
        ),
        total - native_missing,
    )
    jumps = _jump_scores(samples, timestamps)
    source_frame_scores = _source_frame_scores(samples, timestamps)
    candidates: list[_Candidate] = []

    if reject_unreliable_limbs:
        for frame_index, sample in enumerate(samples):
            for limb in _LIMBS:
                observations = _observed(samples, frame_index, limb)
                if not observations:
                    continue
                confidences = [
                    _confidence(sample, semantic) for _index, semantic in observations
                ]
                weak = median(confidences) < _LOW_LIMB_CONFIDENCE
                jump = max((jumps.get(item, 0.0) for item in observations), default=0.0)
                if not weak and jump <= 1.0:
                    continue
                confidence_score = (
                    max(0.0, _LOW_LIMB_CONFIDENCE - median(confidences))
                    / _LOW_LIMB_CONFIDENCE
                )
                candidates.append(
                    _Candidate(
                        severity=max(confidence_score, jump - 1.0) + 1.0,
                        observations=observations,
                        frame_index=frame_index,
                        whole_frame=False,
                    )
                )

    if reject_unusable_frames:
        for frame_index, sample in enumerate(samples):
            observations = _observed(samples, frame_index, SMPLX_SEMANTICS)
            if not observations:
                continue
            confidences = [
                _confidence(sample, semantic) for _index, semantic in observations
            ]
            core_reliable = sum(
                _confidence(sample, semantic) >= _LOW_LIMB_CONFIDENCE
                for semantic in _CORE
            )
            jump_count = sum(item in jumps for item in observations)
            weak = median(confidences) < _LOW_FRAME_CONFIDENCE or core_reliable < 3
            source_score = source_frame_scores.get(frame_index, 0.0)
            incoherent = jump_count >= 6 or source_score > 1.0
            if not weak and not incoherent:
                continue
            confidence_score = (
                max(0.0, _LOW_FRAME_CONFIDENCE - median(confidences))
                / _LOW_FRAME_CONFIDENCE
            )
            jump_score = max(
                (jumps.get(item, 0.0) - 1.0 for item in observations),
                default=0.0,
            )
            candidates.append(
                _Candidate(
                    severity=max(
                        confidence_score,
                        jump_score,
                        source_score - 1.0,
                    )
                    + 2.0,
                    observations=observations,
                    frame_index=frame_index,
                    whole_frame=True,
                )
            )

    rejected: set[tuple[int, str]] = set()
    rejected_limbs = 0
    rejected_frames: set[int] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (item.severity, item.whole_frame),
        reverse=True,
    ):
        new = set(candidate.observations) - rejected
        if not new or len(rejected) + len(new) > budget:
            continue
        rejected.update(new)
        if candidate.whole_frame:
            rejected_frames.add(candidate.frame_index)
        else:
            rejected_limbs += 1

    for frame_index, semantic in rejected:
        sample = samples[frame_index]
        sample.rohm_confidence[semantic] = 0.0
        channel = _channel(sample, semantic)
        if channel is not None:
            channel.confidence = 0.0
    for frame_index in rejected_frames:
        samples[frame_index].rohm_root_confidence = 0.0

    return MotionFilterStats(
        total_observations=total,
        native_missing_observations=native_missing,
        rejected_observations=len(rejected),
        rejection_budget=budget,
        rejected_limbs=rejected_limbs,
        rejected_frames=len(rejected_frames),
        rejected=frozenset(rejected),
        rejected_root_frames=frozenset(rejected_frames),
    )


def _replacement_indices(
    frame_index: int,
    available: Sequence[int],
) -> tuple[int | None, int | None]:
    left = next((index for index in reversed(available) if index < frame_index), None)
    right = next((index for index in available if index > frame_index), None)
    return left, right


def _factor(
    frame_index: int,
    left: int,
    right: int,
    timestamps: Sequence[float],
) -> float:
    span = float(timestamps[right]) - float(timestamps[left])
    if span <= 1.0e-8:
        return (frame_index - left) / max(right - left, 1)
    return min(
        max((float(timestamps[frame_index]) - float(timestamps[left])) / span, 0.0),
        1.0,
    )


def repair_body_gaps(
    samples: Sequence[object],
    timestamps: Sequence[float],
) -> int:
    """Repair masked body motion; boundary gaps hold the nearest good pose."""
    missing = {
        (frame_index, semantic)
        for frame_index, sample in enumerate(samples)
        for semantic in SMPLX_SEMANTICS
        if _confidence(sample, semantic) <= 0.0
    }
    repaired = 0
    for semantic in SMPLX_SEMANTICS:
        available = [
            index
            for index, sample in enumerate(samples)
            if (index, semantic) not in missing
            and _channel(sample, semantic) is not None
        ]
        if not available:
            continue
        for frame_index, sample in enumerate(samples):
            if (frame_index, semantic) not in missing:
                continue
            target = _channel(sample, semantic)
            left, right = _replacement_indices(frame_index, available)
            if left is None and right is None:
                continue
            if target is None:
                target = _copy_channel(
                    sample,
                    semantic,
                    samples[right if left is None else left],
                )
            if target is None:
                continue
            if left is None or right is None:
                source_index = right if left is None else left
                source = _channel(samples[source_index], semantic)
                target.rotation = source.rotation.copy()
                if target.key_location and source.key_location:
                    target.location = source.location.copy()
            else:
                factor = _factor(frame_index, left, right, timestamps)
                first = _channel(samples[left], semantic)
                last = _channel(samples[right], semantic)
                target.rotation = first.rotation.slerp(
                    last.rotation, factor
                ).normalized()
                if target.key_location and first.key_location and last.key_location:
                    target.location = first.location.lerp(last.location, factor)
            first_global = (
                samples[left].rohm_rotations.get(semantic) if left is not None else None
            )
            last_global = (
                samples[right].rohm_rotations.get(semantic)
                if right is not None
                else None
            )
            if first_global is not None and last_global is not None:
                factor = _factor(frame_index, left, right, timestamps)
                sample.rohm_rotations[semantic] = first_global.slerp(
                    last_global, factor
                ).normalized()
            elif first_global is not None or last_global is not None:
                sample.rohm_rotations[semantic] = (
                    first_global if first_global is not None else last_global
                ).copy()
            repaired += 1
    return repaired


__all__ = [
    "MotionFilterStats",
    "SOURCE_BODY_JOINTS",
    "filter_motion_observations",
    "repair_body_gaps",
]
