"""Root targets, body heights and root travel from video."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mathutils import Matrix, Quaternion, Vector

from ..core.requests import RootMotionMode
from ..core.scene import SceneContext
from ..models import CanonicalFrame
from .samples import _BakedFrameSample, _RootMotionSample, _translate_grounding
from .solve.common import _canonical_vector, _mapped_bone, body_points
from .solve.frames import source_root_basis, target_basis
from .target_pose import signed_twist_angle as _signed_twist_angle


def _sequence_root_target(armature, entries):
    root = _mapped_bone(armature, entries, "ROOT")
    return ("BONE", root) if root is not None else ("OBJECT", armature)


def _reset_root_location(root_target, *, scene_context: SceneContext) -> None:
    """Reset the user-facing root location channels without changing rotation."""
    root_target.location = Vector()
    root_target.id_data.update_tag(refresh={"OBJECT"})
    scene_context.update()


def _source_body_height(frame: CanonicalFrame) -> float:
    points = body_points(frame)

    def midpoint(first: str, second: str) -> Vector | None:
        left, right = points.get(first), points.get(second)
        if not left or not right or not left.valid or not right.valid:
            return None
        return (Vector(left.position) + Vector(right.position)) * 0.5

    hip = midpoint("LEFT_HIP", "RIGHT_HIP")
    shoulder = midpoint("LEFT_SHOULDER", "RIGHT_SHOULDER")
    ankle = midpoint("LEFT_ANKLE", "RIGHT_ANKLE")
    head_points = [
        points.get(name)
        for name in ("NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR")
    ]
    head_points = [point for point in head_points if point and point.valid]
    head = (
        sum((Vector(point.position) for point in head_points), Vector())
        / len(head_points)
        if head_points
        else None
    )
    lengths = []
    if hip is not None and ankle is not None:
        lengths.append((hip - ankle).length)
    if hip is not None and shoulder is not None:
        lengths.append((hip - shoulder).length)
    if shoulder is not None and head is not None:
        lengths.append((shoulder - head).length)
    return sum(lengths) if len(lengths) >= 2 else 0.0


def _target_body_height(armature, entries) -> float:
    chains = []
    for side in ("L", "R"):
        upper = _mapped_bone(armature, entries, f"UPPER_LEG_{side}")
        lower = _mapped_bone(armature, entries, f"LOWER_LEG_{side}")
        if upper and lower:
            chains.append(float(upper.bone.length + lower.bone.length))
    leg = sum(chains) / len(chains) if chains else 0.0
    pelvis = _mapped_bone(armature, entries, "PELVIS")
    head = _mapped_bone(armature, entries, "HEAD")
    torso = (
        (head.bone.tail_local - pelvis.bone.head_local).length
        if pelvis and head
        else 0.0
    )
    return leg + torso


def _root_motion_sample(frame: CanonicalFrame) -> _RootMotionSample:
    return _RootMotionSample(
        timestamp_seconds=float(frame.timestamp_ms) / 1000.0,
        observation=frame.root_motion,
        source_basis=source_root_basis(frame),
        source_height=_source_body_height(frame),
    )


def _root_motion_deltas(
    armature,
    entries,
    samples: Sequence[_RootMotionSample],
    mode: str,
) -> tuple[list[Vector], list[float], str, list[str]]:
    zeros = [Vector() for _sample in samples]
    yaws = [0.0 for _sample in samples]
    if mode != RootMotionMode.AUTOMATIC:
        return zeros, yaws, mode, []
    observations = [
        index
        for index, sample in enumerate(samples)
        if sample.observation is not None and sample.observation.valid
    ]
    if not observations:
        warning = (
            "Automatic root travel is unavailable for this provider/cache; "
            "baked in place"
        )
        return zeros, yaws, RootMotionMode.IN_PLACE, [warning]

    first_index = observations[0]
    first_sample = samples[first_index]
    first_observation = first_sample.observation
    source_space = first_sample.source_basis
    target_space = target_basis(armature, entries)
    source_height = first_sample.source_height
    target_height = _target_body_height(armature, entries)
    scale = (
        target_height / source_height
        if source_height > 1.0e-6 and target_height > 1.0e-6
        else 1.0
    )
    scale = min(max(scale, 0.1), 10.0)
    previous_delta = Vector()
    previous_yaw = 0.0
    first_basis = first_sample.source_basis
    canonical_axes = Matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
    )
    first_global_rotation = None
    if first_observation.rotation_matrix is not None:
        values = first_observation.rotation_matrix
        first_global_rotation = (
            canonical_axes
            @ Matrix((values[0:3], values[3:6], values[6:9]))
            @ canonical_axes.inverted_safe()
        )
    source_to_target = target_space @ source_space.inverted_safe()
    target_up = target_space @ Vector((0.0, 0.0, 1.0))
    for index, sample in enumerate(samples):
        observation = sample.observation
        if observation is None or not observation.valid:
            zeros[index] = previous_delta.copy()
            yaws[index] = previous_yaw
            continue
        camera_delta = Vector(observation.translation) - Vector(
            first_observation.translation
        )
        canonical = _canonical_vector(camera_delta)
        previous_delta = (
            target_space @ (source_space.inverted_safe() @ canonical)
        ) * scale
        if (
            observation.rotation_matrix is not None
            and first_global_rotation is not None
        ):
            values = observation.rotation_matrix
            global_rotation = (
                canonical_axes
                @ Matrix((values[0:3], values[3:6], values[6:9]))
                @ canonical_axes.inverted_safe()
            )
            relative_facing = first_global_rotation.inverted_safe() @ global_rotation
            target_relative = (
                source_to_target @ relative_facing @ source_to_target.inverted_safe()
            )
            previous_yaw = _signed_twist_angle(
                target_relative.to_quaternion(), target_up.normalized()
            )
        else:
            relative_facing = first_basis.inverted_safe() @ sample.source_basis
            previous_yaw = relative_facing.to_euler("XYZ").z
        zeros[index] = previous_delta.copy()
        yaws[index] = previous_yaw
    # Keep yaw on a continuous quaternion branch. Temporal Correction owns
    # outlier rejection and smoothing after the final root channel is built.
    for index in range(1, len(yaws)):
        while yaws[index] - yaws[index - 1] > math.pi:
            yaws[index] -= math.tau
        while yaws[index] - yaws[index - 1] < -math.pi:
            yaws[index] += math.tau
    return zeros, yaws, RootMotionMode.AUTOMATIC, []


def _set_root_pose(
    target_kind: str,
    target,
    baseline: Matrix,
    delta: Vector,
    yaw: float,
    target_up: Vector,
    *,
    scene_context: SceneContext,
) -> None:
    location, rotation, scale = baseline.decompose()
    if target_kind == "BONE":
        desired_rotation = Quaternion(target_up.normalized(), yaw) @ rotation
        desired = Matrix.LocRotScale(location + delta, desired_rotation, scale)
        target.matrix = desired
    else:
        world_axis = rotation @ target_up
        desired_rotation = Quaternion(world_axis.normalized(), yaw) @ rotation
        world_delta = baseline.to_3x3() @ delta
        desired = Matrix.LocRotScale(location + world_delta, desired_rotation, scale)
        target.matrix_world = desired
    scene_context.update()


def _root_baked_channel(sample: _BakedFrameSample):
    return next(
        (
            channel
            for channel in sample.bones.values()
            if channel.semantic_id == "ROOT" and channel.key_location
        ),
        None,
    )


def _root_transform(sample: _BakedFrameSample):
    if sample.root_object is not None:
        return sample.root_object[0].copy(), sample.root_object[1].copy(), "OBJECT"
    channel = _root_baked_channel(sample)
    if channel is not None:
        return channel.location.copy(), channel.rotation.copy(), "BONE"
    return None


def _set_root_transform(
    sample: _BakedFrameSample,
    location: Vector,
    rotation: Quaternion,
) -> bool:
    current = _root_transform(sample)
    if current is None:
        return False
    old_location, _old_rotation, kind = current
    if kind == "OBJECT":
        _location, _rotation, scale = sample.root_object
        sample.root_object = (location.copy(), rotation.copy(), scale.copy())
    else:
        channel = _root_baked_channel(sample)
        channel.location = location.copy()
        channel.rotation = rotation.copy()
    _translate_grounding(sample.grounding, location - old_location)
    return True
