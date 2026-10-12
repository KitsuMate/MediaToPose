"""Apply RoHM root and pose corrections to baked samples."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from mathutils import Quaternion, Vector

from .. import (
    motion_filter,
    rig_planning,
)
from ..core.errors import BackendError
from ..core.requests import RootMotionMode
from ..core.scene import SceneContext
from .capture import _ik_bone_confidences
from .pose import _commit_target_pose
from .samples import (
    SequenceApplyProgress,
    _append_bake_warning,
    _apply_baked_sample_pose,
    _BakedBoneSample,
    _BakedFrameSample,
    _restore_sequence_state,
    _scene_set_float,
    _sequence_state,
    _sequence_timestamps,
    _store_bone_basis,
    _translate_grounding,
)
from .solve.common import _mapped_bone
from .solve.frames import target_basis
from .solve.ik import _prepare_ik_output
from .target_pose import TargetPose


def _iter_apply_rohm_root_deltas(
    armature,
    samples: Sequence[_BakedFrameSample],
    root_deltas: Sequence[Sequence[float]],
    root_kind: str,
    root_target,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    if len(root_deltas) != len(samples):
        raise BackendError("RoHM root corrections do not match the baked sequence")
    temporary = _sequence_state(armature, scene_context=scene_context)
    try:
        for index, (sample, raw_delta) in enumerate(
            zip(samples, root_deltas, strict=True), 1
        ):
            delta = Vector(tuple(float(value) for value in raw_delta))
            if delta.length_squared <= 1.0e-20:
                yield SequenceApplyProgress(
                    "RoHM: applying corrected root", index, len(samples)
                )
                continue
            _scene_set_float(scene_context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample, scene_context=scene_context)
            if root_kind == "BONE":
                original = sample.bones.get(root_target.name)
                if original is None:
                    yield SequenceApplyProgress(
                        "RoHM: applying corrected root", index, len(samples)
                    )
                    continue
                original_rotation = original.rotation.copy()
                original_scale = original.scale.copy()
                matrix = root_target.matrix.copy()
                matrix.translation += delta
                root_target.matrix = matrix
                scene_context.update()
                _store_bone_basis(sample, root_target)
                stored = sample.bones[root_target.name]
                stored.rotation = original_rotation
                stored.scale = original_scale
            else:
                if sample.root_object is None:
                    yield SequenceApplyProgress(
                        "RoHM: applying corrected root", index, len(samples)
                    )
                    continue
                original_rotation = sample.root_object[1].copy()
                original_scale = sample.root_object[2].copy()
                matrix = root_target.matrix_world.copy()
                matrix.translation += delta
                root_target.matrix_world = matrix
                scene_context.update()
                location, _rotation, _scale = root_target.matrix_basis.decompose()
                sample.root_object = (
                    location.copy(),
                    original_rotation,
                    original_scale,
                )
            _translate_grounding(sample.grounding, delta)
            yield SequenceApplyProgress(
                "RoHM: applying corrected root", index, len(samples)
            )
    finally:
        _restore_sequence_state(armature, temporary, scene_context=scene_context)


def _merge_rohm_pose_into_sample(
    sample: _BakedFrameSample,
    target_pose: TargetPose,
    ik_proposals: Sequence[dict],
    changed_bones: set[object],
) -> None:
    channels = {channel.bone: channel for channel in target_pose.bones.values()}
    ik_confidences = _ik_bone_confidences(ik_proposals)
    ik_location = set()
    ik_rotation = set()
    for proposal in ik_proposals:
        ik_rotation.add(proposal["target"])
        if not proposal.get("orientation_only"):
            ik_location.add(proposal["target"])
            if proposal.get("rotation_bend") is None:
                ik_location.add(proposal["pole"])
            if proposal.get("rotation_bend") is not None:
                ik_rotation.add(proposal["rotation_bend"])
            if proposal.get("toe") is not None:
                ik_rotation.add(proposal["toe"])
    for bone in changed_bones:
        channel = channels.get(bone)
        previous = sample.bones.get(bone.name)
        location, rotation, scale = bone.matrix_basis.decompose()
        semantic_id = (
            channel.semantic_id
            if channel is not None
            else previous.semantic_id
            if previous is not None
            else ""
        )
        confidence = (
            min(float(channel.confidence), ik_confidences.get(bone, 1.0))
            if channel is not None
            else ik_confidences.get(
                bone, previous.confidence if previous is not None else 0.0
            )
        )
        sample.bones[bone.name] = _BakedBoneSample(
            bone=bone,
            location=location.copy(),
            rotation=rotation.normalized(),
            scale=scale.copy(),
            key_location=bool(
                bone in ik_location
                or channel is not None
                and channel.location is not None
                or previous is not None
                and previous.key_location
            ),
            key_rotation=bool(
                bone in ik_rotation
                or channel is not None
                and (
                    channel.desired_rotation is not None
                    or channel.direct_basis_rotation is not None
                    or channel.hinge_axis_local is not None
                )
                or previous is not None
                and previous.key_rotation
            ),
            semantic_id=semantic_id,
            confidence=min(max(float(confidence), 0.0), 1.0),
            properties=(dict(previous.properties) if previous is not None else {}),
        )


def _iter_apply_rohm_pose(
    armature,
    entries,
    rig_plan: rig_planning.RigSolvePlan,
    samples: Sequence[_BakedFrameSample],
    pose_global_rotations: Sequence[dict[str, Quaternion]],
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    """Rest-align RoHM globals and resolve them through the control rig."""
    if len(pose_global_rotations) != len(samples):
        raise BackendError("RoHM pose corrections do not match the baked sequence")
    rest_basis = rig_plan.rest_basis.to_3x3()
    rest_inverse = rest_basis.inverted_safe()
    temporary = _sequence_state(armature, scene_context=scene_context)
    try:
        for index, (sample, corrected_globals) in enumerate(
            zip(samples, pose_global_rotations, strict=True), 1
        ):
            if not corrected_globals:
                yield SequenceApplyProgress(
                    "RoHM: preserving trusted pose", index, len(samples)
                )
                continue
            _scene_set_float(scene_context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample, scene_context=scene_context)
            posed_root_basis = target_basis(
                armature,
                entries,
                rest_basis=rig_plan.rest_basis,
            ).to_3x3()
            target_pose = TargetPose(
                armature,
                None,
                entries,
                rest_basis,
                posed_root_basis,
                rig_plan=rig_plan,
            )
            for semantic_id, anatomical_rotation in corrected_globals.items():
                if semantic_id == "PELVIS":
                    continue
                bone = _mapped_bone(armature, entries, semantic_id)
                source_rotation = sample.rohm_rotations.get(semantic_id)
                if bone is None or source_rotation is None:
                    continue
                if (
                    source_rotation.rotation_difference(anatomical_rotation).angle
                    <= 1.0e-5
                ):
                    continue
                target_delta = (
                    posed_root_basis @ anatomical_rotation.to_matrix() @ rest_inverse
                )
                desired = target_delta @ bone.bone.matrix_local.to_3x3()
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    desired,
                    confidence=0.0,
                    owner="ROHM",
                )
            if target_pose.bones:
                ik_proposals = _prepare_ik_output(target_pose)
                changed_bones, _changed_shapes = _commit_target_pose(
                    target_pose, ik_proposals, scene_context=scene_context
                )
                _merge_rohm_pose_into_sample(
                    sample,
                    target_pose,
                    ik_proposals,
                    changed_bones,
                )
            yield SequenceApplyProgress(
                "RoHM: retargeting corrected pose", index, len(samples)
            )
    finally:
        _restore_sequence_state(armature, temporary, scene_context=scene_context)


def _iter_rohm_correction(
    armature,
    entries,
    rig_plan: rig_planning.RigSolvePlan,
    samples: list[_BakedFrameSample],
    settings,
    effective_mode: str,
    root_kind: str,
    root_target,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    """Run the explicitly enabled verified RoHM correction backend."""
    from .. import rohm_correction
    from ..backends.rohm import backend as rohm_backend

    correct_pose, correct_root = _rohm_requested_work(samples, settings, effective_mode)
    if not correct_pose and not correct_root:
        return
    runtime = rohm_backend.loaded_session()
    iterator = rohm_correction.iter_correct_samples(
        samples,
        _sequence_timestamps(samples),
        runtime,
        root_motion_mode=effective_mode,
        correct_pose=correct_pose,
        correct_root=correct_root,
        visibility_threshold=float(settings.visibility_threshold),
        seed=10,
    )
    pose_apply_total = len(samples) if correct_pose else 0
    root_apply_total = len(samples) if correct_root else 0
    apply_total = pose_apply_total + root_apply_total
    model_total = 0
    while True:
        try:
            phase, current, total = next(iterator)
        except StopIteration as stopped:
            stats = stopped.value
            break
        model_total = max(model_total, int(total))
        yield SequenceApplyProgress(
            str(phase),
            int(current),
            int(total) + apply_total,
        )
    if correct_pose:
        pose_iterator = _iter_apply_rohm_pose(
            armature,
            entries,
            rig_plan,
            samples,
            stats.pose_global_rotations,
            scene_context=scene_context,
        )
        for progress in pose_iterator:
            yield SequenceApplyProgress(
                progress.phase,
                model_total + progress.current,
                model_total + apply_total,
            )
    if correct_root:
        root_iterator = _iter_apply_rohm_root_deltas(
            armature,
            samples,
            stats.root_deltas,
            root_kind,
            root_target,
            scene_context=scene_context,
        )
        for progress in root_iterator:
            yield SequenceApplyProgress(
                progress.phase,
                model_total + pose_apply_total + progress.current,
                model_total + apply_total,
            )
    _append_bake_warning(
        samples,
        "Experimental RoHM Correction changed "
        f"{stats.corrected_source_frames} source frame(s) at 30 FPS using "
        + " and ".join(
            name
            for enabled, name in (
                (correct_pose, "pose repair"),
                (correct_root, "root correction"),
            )
            if enabled
        ),
    )
    if stats.low_rate_warning:
        _append_bake_warning(
            samples,
            "RoHM upsampled input below 30 FPS; absent motion detail cannot be recovered",
        )
    for warning in stats.warnings:
        _append_bake_warning(samples, f"RoHM quality warning · {warning}")


def _rohm_requested_work(
    samples: Sequence[_BakedFrameSample],
    settings,
    effective_mode: str,
) -> tuple[bool, bool]:
    """Return whether masked pose or root observations justify model inference."""
    if settings is None or not bool(getattr(settings, "enabled", False)):
        return False, False
    threshold = min(
        max(float(getattr(settings, "visibility_threshold", 0.5)), 0.0), 1.0
    )
    pose_missing = any(
        float(sample.rohm_confidence.get(semantic, 0.0)) < threshold
        for sample in samples
        for semantic in motion_filter.SMPLX_SEMANTICS[1:]
    )
    correct_pose = bool(getattr(settings, "learned_pose_residual", False)) and bool(
        pose_missing
    )
    root_values = [float(sample.rohm_root_confidence) for sample in samples]
    root_missing = any(value < threshold for value in root_values)
    root_observed = any(value >= threshold for value in root_values)
    correct_root = bool(
        getattr(settings, "root_trajectory_cleanup", True)
        and effective_mode == RootMotionMode.AUTOMATIC
        and root_missing
        and root_observed
    )
    return correct_pose, correct_root
