"""Foot contacts, planted-foot repair and grounded root motion."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from mathutils import Matrix, Vector

from ..core.motion import animation_cleanup, grounded_cleanup
from ..core.requests import RootMotionMode
from ..core.scene import SceneContext
from ..core.types import CanonicalFrame, Landmark
from .root_motion import _target_body_height
from .samples import (
    SequenceApplyProgress,
    _append_bake_warning,
    _apply_baked_sample_pose,
    _BakedFrameSample,
    _continuous_runs,
    _restore_sequence_state,
    _scene_set_float,
    _sequence_state,
    _sequence_timestamps,
    _store_bone_basis,
)
from .solve.common import body_points, evaluated_semantic_bone
from .solve.ik import (
    _ik_pole_location,
    _set_pose_translation_preserve_basis_rotation,
    limb_ownership,
)
from .solve.translation import PoseTranslationSolver


def _grounding_landmark(frame: CanonicalFrame, *names: str) -> Landmark | None:
    points = body_points(frame)
    for name in names:
        point = points.get(name)
        if point is not None and point.valid:
            return point
    return None


def _capture_grounding_frame(
    armature,
    entries,
    frame: CanonicalFrame,
    destination: float,
    root_kind: str,
    root_target,
) -> grounded_cleanup.GroundFrame:
    """Capture evaluated target feet plus provider image/confidence evidence."""
    if root_kind == "BONE":
        root_position = root_target.matrix.translation.copy()

        def convert(value):
            return value.copy()

        def convert_rotation(value):
            return value.copy()
    else:
        root_position = root_target.matrix_world.translation.copy()

        def convert(value):
            return armature.matrix_world @ value

        world_rotation = armature.matrix_world.to_quaternion()

        def convert_rotation(value):
            return (world_rotation @ value).normalized()

    points = {}
    joints = {}
    orientations = {}
    joint_confidence = {}
    pelvis = evaluated_semantic_bone(armature, entries, "PELVIS")
    if pelvis is not None:
        joints["PELVIS"] = convert(pelvis.head)
        orientations["PELVIS"] = convert_rotation(pelvis.matrix.to_quaternion())
    for side, label in (("L", "LEFT"), ("R", "RIGHT")):
        upper = evaluated_semantic_bone(armature, entries, f"UPPER_LEG_{side}")
        lower = evaluated_semantic_bone(armature, entries, f"LOWER_LEG_{side}")
        foot = evaluated_semantic_bone(armature, entries, f"FOOT_{side}")
        toe = evaluated_semantic_bone(armature, entries, f"TOE_{side}")
        if foot is None:
            continue
        heel_position = convert(foot.head)
        toe_position = convert(toe.head if toe is not None else foot.tail)
        foot_direction = toe_position - heel_position
        heel_landmark = _grounding_landmark(frame, f"{label}_HEEL", f"{label}_ANKLE")
        toe_landmark = _grounding_landmark(
            frame, f"{label}_FOOT_INDEX", f"{label}_TOE", f"{label}_ANKLE"
        )
        for role, bone, landmark_names in (
            ("HIP", upper, (f"{label}_HIP",)),
            ("KNEE", lower, (f"{label}_KNEE",)),
            ("ANKLE", foot, (f"{label}_ANKLE",)),
            ("TOE", toe, (f"{label}_FOOT_INDEX", f"{label}_TOE")),
        ):
            if bone is None:
                continue
            key = f"{side}_{role}"
            joints[key] = convert(bone.head)
            orientations[key] = convert_rotation(bone.matrix.to_quaternion())
            landmark = _grounding_landmark(frame, *landmark_names)
            joint_confidence[key] = (
                float(landmark.confidence) if landmark is not None else 0.0
            )
        if heel_landmark is not None:
            points[f"{side}_HEEL"] = grounded_cleanup.ContactPoint(
                position=heel_position,
                image=Vector(heel_landmark.image),
                confidence=float(heel_landmark.confidence),
                direction=foot_direction,
                motion_position=heel_position - root_position,
            )
        if toe_landmark is not None:
            points[f"{side}_TOE"] = grounded_cleanup.ContactPoint(
                position=toe_position,
                image=Vector(toe_landmark.image),
                confidence=float(toe_landmark.confidence),
                direction=foot_direction,
                motion_position=toe_position - root_position,
            )
    observation = frame.root_motion
    return grounded_cleanup.GroundFrame(
        timestamp=float(frame.timestamp_ms) / 1000.0,
        root=root_position,
        points=points,
        joints=joints,
        orientations=orientations,
        joint_confidence=joint_confidence,
        root_confidence=(
            min(max(float(observation.confidence), 0.0), 1.0)
            if observation is not None and observation.valid
            else 0.0
        ),
    )


def _contact_targets(
    result: grounded_cleanup.GroundingResult,
    timestamps: Sequence[float],
) -> list[dict[str, tuple[Vector, float]]]:
    targets = [dict() for _ in timestamps]
    for segment in result.segments:
        duration = max(timestamps[segment.end] - timestamps[segment.start], 1.0e-4)
        transition = max(2, int(round(0.1 * len(segment.weights) / duration)))
        for offset, index in enumerate(range(segment.start, segment.end + 1)):
            edge = min(offset + 1, segment.end - index + 1) / max(transition, 1)
            weight = edge * edge * (3.0 - 2.0 * min(max(edge, 0.0), 1.0))
            weight *= max(result.contacts[segment.point][index], 0.35)
            targets[index][segment.point] = (segment.anchor.copy(), min(weight, 1.0))
    return targets


def _reduce_ground_contact_outliers(
    frames: Sequence[grounded_cleanup.GroundFrame],
) -> int:
    """Clean isolated evaluated heel/toe failures before contact analysis."""
    timestamps = [float(frame.timestamp) for frame in frames]
    repaired = 0
    names = {name for frame in frames for name in frame.points}
    for name in names:
        indices = [index for index, frame in enumerate(frames) if name in frame.points]
        for run in _continuous_runs(indices, timestamps):
            points = [frames[index].points[name] for index in run]
            values, count = animation_cleanup.reduce_vector_outliers(
                [point.position for point in points],
                [timestamps[index] for index in run],
                [point.confidence for point in points],
            )
            repaired += count
            for index, point, value in zip(run, points, values, strict=True):
                point.position = value
                point.motion_position = value - frames[index].root
    return repaired


def _analytic_two_bone_joint(
    hip: Vector,
    current_knee: Vector,
    desired: Vector,
    upper_length: float,
    lower_length: float,
    target_up: Vector,
) -> tuple[Vector | None, Vector, bool]:
    axis = desired - hip
    if axis.length < 1.0e-7 or upper_length <= 0.0 or lower_length <= 0.0:
        return None, desired.copy(), False
    unreachable = axis.length >= (upper_length + lower_length) * 0.998
    distance = min(
        max(axis.length, abs(upper_length - lower_length) + 1.0e-5),
        (upper_length + lower_length) * 0.995,
    )
    axis.normalize()
    bend = current_knee - hip
    bend -= axis * bend.dot(axis)
    if bend.length < 1.0e-6:
        bend = target_up.cross(axis)
    if bend.length < 1.0e-6:
        bend = Vector((1.0, 0.0, 0.0)).cross(axis)
    bend.normalize()
    along = (
        upper_length * upper_length + distance * distance - lower_length * lower_length
    ) / (2.0 * distance)
    perpendicular = math.sqrt(max(upper_length * upper_length - along * along, 0.0))
    desired_knee = hip + axis * along + bend * perpendicular
    clamped_end = hip + axis * distance
    return desired_knee, clamped_end, unreachable


def _repair_fk_leg(
    sample: _BakedFrameSample,
    upper,
    lower,
    foot,
    desired_ankle: Vector,
    weight: float,
    target_up: Vector,
    *,
    scene_context: SceneContext,
) -> bool:
    hip = upper.head.copy()
    current_knee = lower.head.copy()
    current_ankle = foot.head.copy()
    desired = current_ankle.lerp(desired_ankle, min(max(weight, 0.0), 1.0))
    desired_knee, desired, unreachable = _analytic_two_bone_joint(
        hip,
        current_knee,
        desired,
        float(upper.bone.length),
        float(lower.bone.length),
        target_up,
    )
    if desired_knee is None:
        return False
    current_upper = upper.tail - upper.head
    desired_upper = desired_knee - hip
    if current_upper.length > 1.0e-8 and desired_upper.length > 1.0e-8:
        rotation = current_upper.rotation_difference(desired_upper)
        matrix = upper.matrix.copy()
        matrix = Matrix.LocRotScale(
            matrix.translation,
            (rotation @ matrix.to_quaternion()).normalized(),
            matrix.to_scale(),
        )
        upper.matrix = matrix
        scene_context.update()
    current_lower = lower.tail - lower.head
    desired_lower = desired - lower.head
    if current_lower.length > 1.0e-8 and desired_lower.length > 1.0e-8:
        rotation = current_lower.rotation_difference(desired_lower)
        matrix = lower.matrix.copy()
        matrix = Matrix.LocRotScale(
            matrix.translation,
            (rotation @ matrix.to_quaternion()).normalized(),
            matrix.to_scale(),
        )
        lower.matrix = matrix
        scene_context.update()
    _store_bone_basis(sample, upper)
    _store_bone_basis(sample, lower)
    return unreachable


def _iter_ground_contacts(
    armature,
    entries,
    samples: list[_BakedFrameSample],
    strength: float,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_up: Vector,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    timestamps = _sequence_timestamps(samples)
    for run in _continuous_runs(list(range(len(samples))), timestamps):
        yield from _iter_ground_contacts_run(
            armature,
            entries,
            [samples[index] for index in run],
            strength,
            effective_mode,
            root_kind,
            root_target,
            target_up,
            scene_context=scene_context,
        )


def _iter_ground_contacts_run(
    armature,
    entries,
    samples: list[_BakedFrameSample],
    strength: float,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_up: Vector,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    ground_frames = [sample.grounding for sample in samples]
    if not ground_frames or any(frame is None for frame in ground_frames):
        warning = (
            "Ground Contacts skipped because evaluated foot samples were unavailable"
        )
        _append_bake_warning(samples, warning)
        return
    body_height = max(_target_body_height(armature, entries), 1.0e-4)
    grounding_up = (
        target_up.normalized()
        if root_kind == "BONE"
        else (armature.matrix_world.to_3x3() @ target_up).normalized()
    )
    grounding_height = body_height
    if root_kind != "BONE":
        world_scale = armature.matrix_world.to_scale()
        grounding_height *= max(sum(abs(value) for value in world_scale) / 3.0, 1.0e-4)
    contact_repairs = _reduce_ground_contact_outliers(ground_frames)
    if contact_repairs:
        _append_bake_warning(
            samples,
            f"Ground Contacts repaired {contact_repairs} isolated foot sample(s)",
        )
    result = grounded_cleanup.analyze_grounding(
        ground_frames, grounding_height, grounding_up
    )
    available_points = {name for frame in ground_frames for name in frame.points}
    missing_sides = [
        side
        for side in ("L", "R")
        if f"{side}_HEEL" not in available_points
        and f"{side}_TOE" not in available_points
    ]
    if missing_sides:
        warning = "Ground Contacts limited by missing foot mapping(s): " + ", ".join(
            missing_sides
        )
        _append_bake_warning(samples, warning)
    for warning in result.warnings:
        _append_bake_warning(samples, warning)
    for index in range(len(samples)):
        yield SequenceApplyProgress("Detecting foot contacts", index + 1, len(samples))

    if effective_mode == RootMotionMode.AUTOMATIC:
        latest_roots = [frame.root.copy() for frame in ground_frames]
        iterator = grounded_cleanup.stabilize_root(
            ground_frames,
            result.contacts,
            result.segments,
            grounding_height,
            grounding_up,
            strength,
        )
        for iteration, roots in iterator:
            latest_roots = roots
            yield SequenceApplyProgress("Stabilizing root motion", iteration, 10)
        temporary = _sequence_state(armature, scene_context=scene_context)
        try:
            for index, (sample, desired_root) in enumerate(
                zip(samples, latest_roots, strict=False), 1
            ):
                _scene_set_float(scene_context.scene, sample.destination)
                _apply_baked_sample_pose(armature, sample, scene_context=scene_context)
                if root_kind == "BONE":
                    matrix = root_target.matrix.copy()
                    matrix.translation = desired_root
                    root_target.matrix = matrix
                    scene_context.update()
                    _store_bone_basis(sample, root_target)
                else:
                    matrix = root_target.matrix_world.copy()
                    matrix.translation = desired_root
                    root_target.matrix_world = matrix
                    scene_context.update()
                    location, rotation, scale = root_target.matrix_basis.decompose()
                    sample.root_object = (
                        location.copy(),
                        rotation.normalized(),
                        scale.copy(),
                    )
                yield SequenceApplyProgress(
                    "Applying stabilized root", index, len(samples)
                )
        finally:
            _restore_sequence_state(armature, temporary, scene_context=scene_context)

    targets = _contact_targets(result, [frame.timestamp for frame in ground_frames])
    ownership = limb_ownership(armature)
    temporary = _sequence_state(armature, scene_context=scene_context)
    unreachable_count = 0
    try:
        for index, (sample, frame_targets) in enumerate(
            zip(samples, targets, strict=False), 1
        ):
            _scene_set_float(scene_context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample, scene_context=scene_context)
            translation_solver = PoseTranslationSolver(scene_context.update)
            for side in ("L", "R"):
                heel_target = frame_targets.get(f"{side}_HEEL")
                toe_target = frame_targets.get(f"{side}_TOE")
                if heel_target is None and toe_target is None:
                    continue
                upper = evaluated_semantic_bone(armature, entries, f"UPPER_LEG_{side}")
                lower = evaluated_semantic_bone(armature, entries, f"LOWER_LEG_{side}")
                foot = evaluated_semantic_bone(armature, entries, f"FOOT_{side}")
                toe = evaluated_semantic_bone(armature, entries, f"TOE_{side}")
                if upper is None or lower is None or foot is None:
                    continue
                to_pose = (
                    (lambda value: value.copy())
                    if root_kind == "BONE"
                    else (lambda value: armature.matrix_world.inverted_safe() @ value)
                )
                current_heel = foot.head.copy()
                current_toe = toe.head.copy() if toe is not None else foot.tail.copy()
                preserved_foot_rotation = foot.matrix.to_quaternion().normalized()
                if heel_target is not None:
                    desired_ankle = to_pose(heel_target[0])
                    weight = heel_target[1]
                else:
                    desired_ankle = to_pose(toe_target[0]) - (
                        current_toe - current_heel
                    )
                    weight = toe_target[1]
                if (desired_ankle - upper.head).length >= (
                    float(upper.bone.length) + float(lower.bone.length)
                ) * 0.998:
                    unreachable_count += 1
                info = ownership.get(("LEG", side), {})
                if info.get("mode") == "IK" and info.get("target") is not None:
                    target = info["target"]
                    desired_joint, clamped_ankle, _unreachable = (
                        _analytic_two_bone_joint(
                            upper.head.copy(),
                            lower.head.copy(),
                            current_heel.lerp(desired_ankle, weight),
                            float(upper.bone.length),
                            float(lower.bone.length),
                            target_up,
                        )
                    )
                    desired_endpoint = (
                        clamped_ankle if desired_joint is not None else desired_ankle
                    )
                    for _attempt in range(3):
                        error = desired_endpoint - foot.head
                        if error.length < 1.0e-5:
                            break
                        matrix = target.matrix.copy()
                        matrix.translation += error
                        target.matrix = matrix
                        scene_context.update()
                    pole = info.get("pole")
                    rotation_bend = info.get("rotation_bend")
                    proposal = sample.ik_legs.get(side, {})
                    if (
                        pole is not None
                        and desired_joint is not None
                        and rotation_bend is None
                    ):
                        pole_location = _ik_pole_location(
                            Matrix.Translation(upper.head),
                            Matrix.Translation(desired_joint),
                            Matrix.Translation(desired_endpoint),
                            pole,
                            proposal.get("pole_rest_geometry"),
                        )
                        _set_pose_translation_preserve_basis_rotation(
                            pole,
                            pole_location,
                            solver=translation_solver,
                            scene_context=scene_context,
                        )
                        _store_bone_basis(sample, pole)
                    elif rotation_bend is not None and desired_joint is not None:
                        axis = desired_endpoint - upper.head
                        if axis.length > 1.0e-8:
                            axis.normalize()
                            current_bend = upper.tail - (
                                upper.head + axis * (upper.tail - upper.head).dot(axis)
                            )
                            wanted_bend = desired_joint - (
                                upper.head
                                + axis * (desired_joint - upper.head).dot(axis)
                            )
                            if (
                                current_bend.length > 1.0e-8
                                and wanted_bend.length > 1.0e-8
                            ):
                                current_bend.normalize()
                                wanted_bend.normalize()
                                angle = math.atan2(
                                    axis.dot(current_bend.cross(wanted_bend)),
                                    current_bend.dot(wanted_bend),
                                )
                                rotation = rotation_bend.rotation_euler.copy()
                                rotation.y += angle
                                rotation_bend.rotation_euler = rotation
                                scene_context.update()
                                _store_bone_basis(sample, rotation_bend)
                    for _attempt in range(2):
                        error = desired_endpoint - foot.head
                        if error.length < 1.0e-5:
                            break
                        matrix = target.matrix.copy()
                        matrix.translation += error
                        target.matrix = matrix
                        scene_context.update()
                    _store_bone_basis(sample, target)
                else:
                    _repair_fk_leg(
                        sample,
                        upper,
                        lower,
                        foot,
                        desired_ankle,
                        weight,
                        target_up,
                        scene_context=scene_context,
                    )
                    # Ground contact owns the planted position, not the foot's
                    # orientation. Counter-rotate the FK foot after moving its
                    # parent leg so the captured foot direction remains intact.
                    # IK effectors above are translated without changing their
                    # rotation. This avoids heel/toe contact evidence selecting
                    # a sudden, visually unrelated orientation branch.
                    matrix = foot.matrix.copy()
                    foot.matrix = Matrix.LocRotScale(
                        matrix.translation,
                        preserved_foot_rotation,
                        matrix.to_scale(),
                    )
                    scene_context.update()
                    _store_bone_basis(sample, foot)
            yield SequenceApplyProgress("Repairing planted feet", index, len(samples))
    finally:
        _restore_sequence_state(armature, temporary, scene_context=scene_context)
    if unreachable_count:
        warning = f"Clamped {unreachable_count} unreachable planted-leg correction(s) without stretching"
        _append_bake_warning(samples, warning)


def _has_groundable_body_channels(samples: Sequence[_BakedFrameSample]) -> bool:
    return any(
        channel.semantic_id.startswith(
            ("PELVIS", "UPPER_LEG_", "LOWER_LEG_", "FOOT_", "TOE_")
        )
        for sample in samples
        for channel in sample.bones.values()
    )
