"""Measure the quality of an armature animation.

``evaluate`` samples the evaluated pose at every frame of a range and
returns ``Metrics``. All distances are metres in world space (Blender units
at scale 1), speeds are per second at the scene frame rate, and angles are
degrees. ``docs/agents.md`` lists what each metric means, its acceptable
range and which clean-up setting to change when it is off.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mathutils import Vector

from ..core.errors import InvalidRequest
from ..core.requests import FaceOutput
from ..core.scene import SceneContext
from ..core.types import CanonicalFrame
from ..rig import mapping
from .face_metrics import FaceSampler

METRICS_SCHEMA_VERSION = 1

FEET = {"L": ("FOOT_L", "TOE_L"), "R": ("FOOT_R", "TOE_R")}
LIMBS = {
    "elbow_L": (
        "UPPER_ARM_L",
        "LOWER_ARM_L",
        "LEFT_SHOULDER",
        "LEFT_ELBOW",
        "LEFT_WRIST",
    ),
    "elbow_R": (
        "UPPER_ARM_R",
        "LOWER_ARM_R",
        "RIGHT_SHOULDER",
        "RIGHT_ELBOW",
        "RIGHT_WRIST",
    ),
    "knee_L": ("UPPER_LEG_L", "LOWER_LEG_L", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "knee_R": ("UPPER_LEG_R", "LOWER_LEG_R", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
}
JITTER_SEMANTICS = (
    "PELVIS",
    "CHEST",
    "HEAD",
    "UPPER_ARM_L",
    "LOWER_ARM_L",
    "HAND_L",
    "UPPER_ARM_R",
    "LOWER_ARM_R",
    "HAND_R",
    "UPPER_LEG_L",
    "LOWER_LEG_L",
    "FOOT_L",
    "UPPER_LEG_R",
    "LOWER_LEG_R",
    "FOOT_R",
)
TWIST_LIMIT_DEGREES = 90.0
BEND_LIMIT_DEGREES = 160.0
ROTATION_LIMIT_DEGREES = 170.0


@dataclass
class Metrics:
    """Animation quality measures; see the module docstring for units."""

    armature: str
    action: str
    frame_range: list[int]
    frames: int
    fps: float
    body_height_m: float
    foot_sliding_m_per_s: float = 0.0
    foot_contact_ratio: float = 0.0
    ground_penetration_m: float = 0.0
    penetration_frames: int = 0
    jitter_mm: float = 0.0
    twist_violations: int = 0
    joint_limit_violations: int = 0
    root_speed_spikes: int = 0
    max_root_speed_m_per_s: float = 0.0
    landmark_fit_error_deg: float | None = None
    expression_fit: float | None = None
    blink_recall: float | None = None
    face_jitter: float | None = None
    asymmetry: float | None = None
    worst_frames: dict[str, list[int]] = field(default_factory=dict)
    schema_version: int = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bone(armature, entries, semantic_id: str):
    entry = entries.get(semantic_id)
    if entry is None or not entry.enabled or not entry.target_name:
        return None
    if entry.target_kind != "BONE":
        return None
    return armature.pose.bones.get(entry.target_name)


def _angle(first: Vector, second: Vector) -> float | None:
    if first.length < 1.0e-8 or second.length < 1.0e-8:
        return None
    return math.degrees(first.angle(second))


def _bend(points: Sequence[Vector]) -> float | None:
    """Bend at the middle point: 0 when straight, 180 when folded back."""
    return _angle(points[1] - points[0], points[2] - points[1])


def _twist_degrees(quaternion) -> float:
    """Twist about the bone's own Y axis from a swing-twist split."""
    twist = quaternion.copy()
    twist.x = 0.0
    twist.z = 0.0
    if twist.magnitude < 1.0e-8:
        return 0.0
    twist.normalize()
    angle = math.degrees(2.0 * math.acos(min(abs(twist.w), 1.0)))
    return min(angle, 360.0 - angle)


def _body_height(armature, entries) -> float:
    heads = []
    for semantic_id in ("HEAD", "FOOT_L", "FOOT_R", "TOE_L", "TOE_R", "PELVIS"):
        bone = _bone(armature, entries, semantic_id)
        if bone is not None:
            heads.append((armature.matrix_world @ bone.bone.tail_local).z)
            heads.append((armature.matrix_world @ bone.bone.head_local).z)
    return max(max(heads) - min(heads), 0.1) if heads else 1.7


def _top_frames(values: dict[int, float], count: int = 5) -> list[int]:
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return sorted(frame for frame, value in ranked[:count] if value > 0.0)


def _detected_bends(
    detections: Iterable[tuple[float, CanonicalFrame]],
) -> dict[int, dict[str, float]]:
    bends: dict[int, dict[str, float]] = {}
    for timeline_frame, frame in detections:
        values = {}
        for name, (_upper, _lower, first, middle, last) in LIMBS.items():
            points = [
                point
                for item in (first, middle, last)
                if (point := frame.body.get(item)) is not None
                and point.valid
                and point.confidence >= 0.5
            ]
            if len(points) < 3:
                continue
            bend = _bend([Vector(point.position) for point in points])
            if bend is not None:
                values[name] = bend
        if values:
            bends[int(round(timeline_frame))] = values
    return bends


def evaluate(
    scene_context: SceneContext,
    armature,
    *,
    frame_range: tuple[int, int] | None = None,
    ground_height: float = 0.0,
    detections: Iterable[tuple[float, CanonicalFrame]] | None = None,
    face_output: FaceOutput | str | None = None,
) -> Metrics:
    """Sample ``armature`` over ``frame_range`` (inclusive timeline frames).

    The armature is evaluated with whatever animation it currently has.
    ``detections`` pairs timeline frames with the detected frames applied
    there; with it, ``landmark_fit_error_deg`` compares elbow and knee bends.
    ``face_output`` names the face output the capture used; with it the
    face metrics are measured (see ``face_metrics``). The scene frame is
    restored afterwards.
    """
    detections = list(detections) if detections is not None else None
    scene = scene_context.scene
    animation_data = armature.animation_data
    action = animation_data.action if animation_data else None
    if frame_range is None:
        if action is None:
            raise InvalidRequest(
                "The armature has no action; give frame_range", parameter="frame_range"
            )
        first, last = (int(round(value)) for value in action.frame_range)
    else:
        first, last = (int(value) for value in frame_range)
    if last < first:
        raise InvalidRequest(
            "frame_range must be (first, last)", parameter="frame_range"
        )
    fps = scene_context.fps
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    height = _body_height(armature, entries)
    world = armature.matrix_world
    feet = {
        side: [
            bone for semantic in names if (bone := _bone(armature, entries, semantic))
        ]
        for side, names in FEET.items()
    }
    jitter_bones = [
        bone
        for semantic in JITTER_SEMANTICS
        if (bone := _bone(armature, entries, semantic)) is not None
    ]
    limb_bones = {
        name: (_bone(armature, entries, upper), _bone(armature, entries, lower))
        for name, (upper, lower, *_landmarks) in LIMBS.items()
    }
    twist_bones = [
        (semantic, bone)
        for semantic in JITTER_SEMANTICS
        if (bone := _bone(armature, entries, semantic)) is not None
    ]
    root = _bone(armature, entries, "PELVIS")
    face = FaceSampler(armature, entries, face_output) if face_output else None
    if face is not None and not face.active:
        face = None
    saved_frame = (scene.frame_current, scene.frame_subframe)

    foot_points: dict[str, list[Vector]] = {side: [] for side in feet}
    joint_tracks: list[list[Vector]] = [[] for _bone_item in jitter_bones]
    root_track: list[Vector] = []
    rig_bends: dict[int, dict[str, float]] = {}
    twist_per_frame: dict[int, float] = {}
    limit_per_frame: dict[int, float] = {}
    twist_violations = joint_limit_violations = 0
    try:
        for frame in range(first, last + 1):
            scene.frame_set(frame)
            for side, bones in feet.items():
                points = [world @ bone.tail for bone in bones] + [
                    world @ bone.head for bone in bones[1:]
                ]
                if points:
                    foot_points[side].append(min(points, key=lambda point: point.z))
            for track, bone in zip(joint_tracks, jitter_bones, strict=True):
                track.append(world @ bone.head)
            root_track.append(world @ (root.head if root else Vector()))
            if face is not None:
                face.sample(frame)
            bends = {}
            for name, (upper, lower) in limb_bones.items():
                if upper is None or lower is None:
                    continue
                bend = _bend(
                    [world @ upper.head, world @ upper.tail, world @ lower.tail]
                )
                if bend is not None:
                    bends[name] = bend
                    if bend > BEND_LIMIT_DEGREES:
                        joint_limit_violations += 1
                        limit_per_frame[frame] = max(
                            limit_per_frame.get(frame, 0), bend
                        )
            rig_bends[frame] = bends
            for _semantic, bone in twist_bones:
                rotation = bone.matrix_basis.to_quaternion()
                twist = _twist_degrees(rotation)
                if twist > TWIST_LIMIT_DEGREES:
                    twist_violations += 1
                    twist_per_frame[frame] = max(twist_per_frame.get(frame, 0), twist)
                total = math.degrees(rotation.angle)
                total = min(total, 360.0 - total)
                if total > ROTATION_LIMIT_DEGREES:
                    joint_limit_violations += 1
                    limit_per_frame[frame] = max(limit_per_frame.get(frame, 0), total)
    finally:
        scene.frame_set(saved_frame[0], subframe=saved_frame[1])

    frames = last - first + 1
    metrics = Metrics(
        armature=armature.name,
        action=action.name if action is not None else "",
        frame_range=[first, last],
        frames=frames,
        fps=round(fps, 4),
        body_height_m=round(height, 4),
        twist_violations=twist_violations,
        joint_limit_violations=joint_limit_violations,
    )

    # Foot contacts: the foot is within 3% of body height of its lowest
    # point and moves up or down less than 25% of body height per second.
    sliding = []
    sliding_per_frame: dict[int, float] = {}
    contact_frames = 0
    penetration = 0.0
    penetration_frames: set[int] = set()
    for points in foot_points.values():
        if len(points) < 2:
            continue
        floor = min(point.z for point in points)
        for index in range(1, len(points)):
            previous, current = points[index - 1], points[index]
            depth = ground_height - current.z
            if depth > 1.0e-3:
                penetration = max(penetration, depth)
                penetration_frames.add(first + index)
            vertical = abs(current.z - previous.z) * fps
            if current.z - floor <= 0.03 * height and vertical <= 0.25 * height:
                contact_frames += 1
                speed = (current.xy - previous.xy).length * fps
                sliding.append(speed)
                sliding_per_frame[first + index] = max(
                    sliding_per_frame.get(first + index, 0.0), speed
                )
    sides = max(sum(1 for points in foot_points.values() if len(points) > 1), 1)
    metrics.foot_sliding_m_per_s = (
        round(statistics.fmean(sliding), 5) if sliding else 0.0
    )
    metrics.foot_contact_ratio = round(contact_frames / (sides * max(frames - 1, 1)), 4)
    metrics.ground_penetration_m = round(penetration, 5)
    metrics.penetration_frames = len(penetration_frames)

    # Jitter: distance of each joint from the midpoint of its neighbours in
    # time, a high-pass measure that ignores smooth motion.
    deviations = []
    jitter_per_frame: dict[int, float] = {}
    for track in joint_tracks:
        for index in range(1, len(track) - 1):
            midpoint = (track[index - 1] + track[index + 1]) * 0.5
            deviation = (track[index] - midpoint).length * 1000.0
            deviations.append(deviation * deviation)
            jitter_per_frame[first + index] = max(
                jitter_per_frame.get(first + index, 0.0), deviation
            )
    metrics.jitter_mm = (
        round(math.sqrt(statistics.fmean(deviations)), 4) if deviations else 0.0
    )

    # Root speed spikes: horizontal speed above four times the median and
    # above half a body height per second.
    speeds = [
        (root_track[index].xy - root_track[index - 1].xy).length * fps
        for index in range(1, len(root_track))
    ]
    if speeds:
        median = statistics.median(speeds)
        threshold = max(4.0 * median, 0.5 * height)
        spikes = {
            first + index + 1: speed
            for index, speed in enumerate(speeds)
            if speed > threshold
        }
        metrics.root_speed_spikes = len(spikes)
        metrics.max_root_speed_m_per_s = round(max(speeds), 4)
        metrics.worst_frames["root_speed"] = _top_frames(spikes)

    if detections is not None:
        detected = _detected_bends(detections)
        errors = [
            abs(rig_bends[frame][name] - value)
            for frame, values in detected.items()
            if frame in rig_bends
            for name, value in values.items()
            if name in rig_bends[frame]
        ]
        metrics.landmark_fit_error_deg = (
            round(statistics.fmean(errors), 3) if errors else None
        )

    if face is not None:
        detected_faces = (
            {int(round(frame)): found for frame, found in detections}
            if detections is not None
            else None
        )
        measures = face.finish(detected_faces, fps)
        for name in ("expression_fit", "blink_recall", "face_jitter", "asymmetry"):
            setattr(metrics, name, getattr(measures, name))
        metrics.worst_frames.update(measures.worst_frames)

    metrics.worst_frames.update(
        {
            "foot_sliding": _top_frames(sliding_per_frame),
            "jitter": _top_frames(jitter_per_frame),
            "twist": _top_frames(twist_per_frame),
            "joint_limits": _top_frames(limit_per_frame),
        }
    )
    return metrics


__all__ = ["METRICS_SCHEMA_VERSION", "Metrics", "evaluate"]
