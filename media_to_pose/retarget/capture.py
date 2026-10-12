"""Solve one detected frame into a baked sample."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from mathutils import Matrix, Vector

from .. import (
    camera_motion_correction,
    motion_filter,
    rig_planning,
)
from ..core.scene import SceneContext
from ..models import CanonicalFrame
from ..postprocess import (
    PostProcessingSession,
    iter_pipeline,
)
from .grounding import _capture_grounding_frame
from .pose import _iter_commit_target_pose, build_target_pose
from .root_motion import _target_body_height
from .samples import (
    _BakedBoneSample,
    _BakedFrameSample,
    _BakedShapeSample,
    _RohmFrameInput,
)
from .solve.common import evaluated_semantic_bone
from .solve.ik import _prepare_ik_output
from .target_pose import TargetPose


def _ik_bone_confidences(ik_proposals: Sequence[dict]) -> dict[object, float]:
    """Carry detector confidence onto generated IK controls.

    IK targets and poles are not ordinary TargetPose channels, so they used to
    be captured with an implicit confidence of 1.0. That made Temporal
    Correction trust a noisy wrist target more than the hand observations from
    which it was built.
    """

    result: dict[object, float] = {}
    for proposal in ik_proposals:
        confidence = min(max(float(proposal.get("confidence", 1.0)), 0.0), 1.0)
        for key in ("target", "pole", "rotation_bend", "toe"):
            bone = proposal.get(key)
            if bone is None:
                continue
            result[bone] = min(result.get(bone, 1.0), confidence)
    return result


def _capture_rohm_frame_input(
    armature,
    target_pose: TargetPose,
    root_kind: str,
) -> _RohmFrameInput:
    """Capture the committed frame pose before complete-sequence cleanup.

    Rotations are global rest-relative anatomical transforms.  They are
    deliberately not Blender ``matrix_basis`` values: RoHM converts these
    globals through the official SMPL-X hierarchy before inference.
    """
    from .rohm_correction import SMPLX_SEMANTICS

    result = _RohmFrameInput()
    pelvis = evaluated_semantic_bone(armature, target_pose.entries, "PELVIS")
    if pelvis is None:
        return result
    offset_basis = Matrix.Identity(3)
    if root_kind != "BONE":
        offset_basis = armature.matrix_world.to_3x3()
    anatomy_basis = target_pose.rig_plan.rest_basis.to_3x3()
    posed_root_basis = target_pose.target_basis.to_3x3()
    posed_root_inverse = posed_root_basis.inverted_safe()
    for semantic_id in SMPLX_SEMANTICS:
        bone = evaluated_semantic_bone(armature, target_pose.entries, semantic_id)
        if bone is None:
            continue
        result.joint_offsets[semantic_id] = offset_basis @ (bone.head - pelvis.head)
        rest_rotation = bone.bone.matrix_local.to_3x3()
        global_delta = bone.matrix.to_3x3() @ rest_rotation.inverted_safe()
        anatomical = posed_root_inverse @ global_delta @ anatomy_basis
        rotation = anatomical.to_quaternion().normalized()
        if rotation.w < 0.0:
            rotation.negate()
        result.global_rotations[semantic_id] = rotation
        channel = target_pose.channel_for(semantic_id)
        result.confidence[semantic_id] = (
            min(max(float(channel.confidence), 0.0), 1.0)
            if channel is not None
            else 0.0
        )
    result.body_height = max(_target_body_height(armature, target_pose.entries), 1.0e-4)
    if root_kind != "BONE":
        world_scale = armature.matrix_world.to_scale()
        result.body_height *= max(
            sum(abs(float(value)) for value in world_scale) / 3.0,
            1.0e-4,
        )
    return result


def _baked_frame_from_target(
    armature,
    frame: CanonicalFrame,
    destination: float,
    root_kind: str,
    root_target,
    root_enabled: bool,
    warning_sink: list[str],
    target_pose: TargetPose,
    rohm_input: _RohmFrameInput,
    ik_proposals: list[dict],
    changed_bones: set[object],
    changed_shapes: list[tuple[object, object]],
) -> _BakedFrameSample:
    sample = _BakedFrameSample(
        destination,
        warning_sink,
        timestamp_seconds=float(frame.timestamp_ms) / 1000.0,
        camera_motion_input=camera_motion_correction.input_from_frame(frame),
    )
    channels = {channel.bone: channel for channel in target_pose.bones.values()}
    ik_confidences = _ik_bone_confidences(ik_proposals)
    ik_location = set()
    ik_rotation = set()
    for proposal in ik_proposals:
        if proposal.get("key", (None, None))[0] == "LEG":
            sample.ik_legs[proposal["key"][1]] = proposal
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
        location, rotation, scale = bone.matrix_basis.decompose()
        rotation_channel = bool(
            bone in ik_rotation
            or channel is not None
            and (
                channel.desired_rotation is not None
                or channel.direct_basis_rotation is not None
                or channel.hinge_axis_local is not None
            )
        )
        location_channel = bool(
            bone in ik_location or channel is not None and channel.location is not None
        )
        sample.bones[bone.name] = _BakedBoneSample(
            bone=bone,
            location=location.copy(),
            rotation=rotation.normalized(),
            scale=scale.copy(),
            key_location=location_channel,
            key_rotation=rotation_channel,
            semantic_id=channel.semantic_id if channel else "",
            confidence=(
                min(float(channel.confidence), ik_confidences.get(bone, 1.0))
                if channel
                else ik_confidences.get(bone, 1.0)
            ),
            properties=(
                {name: float(bone[name]) for name in channel.properties if name in bone}
                if channel
                else {}
            ),
        )
    if root_enabled:
        if root_kind == "BONE":
            location, rotation, scale = root_target.matrix_basis.decompose()
            observation = frame.root_motion
            sample.bones[root_target.name] = _BakedBoneSample(
                bone=root_target,
                location=location.copy(),
                rotation=rotation.normalized(),
                scale=scale.copy(),
                key_location=True,
                key_rotation=True,
                semantic_id="ROOT",
                confidence=(
                    min(max(float(observation.confidence), 0.0), 1.0)
                    if observation is not None and observation.valid
                    else 0.0
                ),
            )
        else:
            location, rotation, scale = root_target.matrix_basis.decompose()
            sample.root_object = (location.copy(), rotation.normalized(), scale.copy())
    for _obj, block in changed_shapes:
        sample.shapes[(block.as_pointer(), block.name)] = _BakedShapeSample(
            block, float(block.value)
        )
    for warning in target_pose.warnings:
        if warning not in frame.warnings:
            frame.warnings.append(warning)
        if warning not in warning_sink:
            warning_sink.append(warning)
    sample.grounding = _capture_grounding_frame(
        armature,
        target_pose.entries,
        frame,
        destination,
        root_kind,
        root_target,
    )
    sample.body_height = float(rohm_input.body_height)
    sample.rohm_joint_offsets = {
        semantic: value.copy() for semantic, value in rohm_input.joint_offsets.items()
    }
    sample.rohm_rotations = {
        semantic: value.copy()
        for semantic, value in rohm_input.global_rotations.items()
    }
    sample.rohm_confidence = dict(rohm_input.confidence)
    if sample.grounding is not None:
        sample.rohm_root_confidence = float(sample.grounding.root_confidence)
    for name in motion_filter.SOURCE_BODY_JOINTS:
        point = frame.body.get(name)
        if point is None or not point.valid or point.confidence <= 0.0:
            continue
        sample.source_body_3d[name] = Vector(point.position)
        sample.source_body_2d[name] = Vector(point.image)
    return sample


def _iter_capture_baked_frame(
    armature,
    frame: CanonicalFrame,
    destination: float,
    settings,
    session: PostProcessingSession,
    root_kind: str,
    root_target,
    root_enabled: bool,
    postprocess_config,
    rig_plan: rig_planning.RigSolvePlan,
    warning_sink: list[str],
    capture_rohm_input: bool,
    *,
    scene_context: SceneContext,
) -> Iterator[tuple[str, int, int]]:
    """Solve one frame in independently cancellable main-thread units."""
    frame_module_ids = {
        "SPINE_CORRECTION",
        "SHOULDER_CORRECTION",
        "ARM_AXIS_CORRECTION",
        "LEG_AXIS_CORRECTION",
    }
    module_count = sum(
        bool(module.enabled) and module.module_id in frame_module_ids
        for module in postprocess_config.modules
    )
    total_steps = module_count + 4 + int(capture_rohm_input)
    completed = 1
    yield "Constructing target pose", completed, total_steps
    target_pose = build_target_pose(
        armature,
        frame,
        settings,
        post_process=False,
        session=session,
        rig_plan=rig_plan,
    )
    pipeline = iter_pipeline(
        target_pose,
        postprocess_config,
        session=session,
    )
    while True:
        try:
            display_name, _index, _total = next(pipeline)
        except StopIteration:
            break
        completed += 1
        yield f"Postprocess: {display_name}", completed, total_steps

    completed += 1
    yield "Preparing IK controls", completed, total_steps
    ik_proposals = _prepare_ik_output(target_pose)
    completed += 1
    yield "Committing pose", completed, total_steps
    commit = _iter_commit_target_pose(
        target_pose, ik_proposals, scene_context=scene_context
    )
    while True:
        try:
            commit_stage, _commit_current, _commit_total = next(commit)
        except StopIteration as stopped:
            changed_bones, changed_shapes = stopped.value
            break
        yield f"Committing pose: {commit_stage}", completed, total_steps
    rohm_input = _RohmFrameInput()
    if capture_rohm_input:
        completed += 1
        yield "Capturing AI cleanup input", completed, total_steps
        rohm_input = _capture_rohm_frame_input(armature, target_pose, root_kind)
    completed += 1
    yield "Capturing solved channels", completed, total_steps
    return _baked_frame_from_target(
        armature,
        frame,
        destination,
        root_kind,
        root_target,
        root_enabled,
        warning_sink,
        target_pose,
        rohm_input,
        ik_proposals,
        changed_bones,
        changed_shapes,
    )
