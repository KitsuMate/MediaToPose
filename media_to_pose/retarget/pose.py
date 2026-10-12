"""Build, post-process and commit one target pose."""

from __future__ import annotations

from collections.abc import Iterator

from .. import (
    mapping,
    rig_planning,
)
from ..core.requests import FaceOutput
from ..core.scene import SceneContext
from ..models import CanonicalFrame
from ..post_processing import (
    PostProcessingSession,
    compile_postprocess_config,
    iter_pipeline,
    run_pipeline,
)
from ..target_pose import TargetPose
from .keyframes import _key_bone, _key_shape, _set_key_interpolation
from .root_motion import _reset_root_location, _sequence_root_target
from .samples import SequenceApplyProgress
from .solve.body import _apply_body
from .solve.face import _apply_eye_gaze, _apply_face_controls, _apply_shapes
from .solve.frames import source_root_basis, target_basis, target_rest_basis
from .solve.hands import _apply_hands
from .solve.ik import _commit_ik_output, _prepare_ik_output


def _iter_commit_target_pose(
    target_pose: TargetPose,
    ik_proposals: list[dict],
    *,
    scene_context: SceneContext,
) -> Iterator[tuple[str, int, int]]:
    """Commit FK/body and IK channels as one cooperative transaction.

    Blender may redraw between dependency-safe batches. Snapshot only channels
    owned by this frame and restore the complete pre-commit state on failure,
    cancellation, or generator close.
    """
    affected_bones = {channel.bone for channel in target_pose.bones.values()}
    for proposal in ik_proposals:
        for key in ("target", "pole", "rotation_bend", "toe"):
            bone = proposal.get(key)
            if bone is not None:
                affected_bones.add(bone)
    bone_basis = {bone: bone.matrix_basis.copy() for bone in affected_bones}
    property_values = [
        (
            channel.bone,
            property_name,
            property_name in channel.bone,
            channel.bone.get(property_name),
        )
        for channel in target_pose.bones.values()
        for property_name in channel.properties
    ]
    shape_values = [
        (shape.block, float(shape.block.value)) for shape in target_pose.shapes.values()
    ]
    try:
        all_channels = target_pose.bones
        finger_prefixes = ("THUMB_", "INDEX_", "MIDDLE_", "RING_", "PINKY_")
        finger_channels = {
            name: channel
            for name, channel in all_channels.items()
            if channel.semantic_id.startswith(finger_prefixes)
        }
        # Generated rigs finalize the hand control through IK after ordinary
        # body channels. Finger controls are descendants of the evaluated
        # deform hand, so resolving them before that IK output bakes the old
        # parent frame into every phalanx as one large side-wide offset.
        if ik_proposals and finger_channels:
            target_pose.bones = {
                name: channel
                for name, channel in all_channels.items()
                if name not in finger_channels
            }
            target_pose.invalidate()
        pose_commit = target_pose.iter_commit(scene_context.update)
        while True:
            try:
                current, total = next(pose_commit)
            except StopIteration as stopped:
                changed_bones, changed_shapes = stopped.value
                break
            yield "Rig channels", int(current), int(total)
        for index, proposal in enumerate(ik_proposals, 1):
            changed_bones.update(
                _commit_ik_output([proposal], scene_context=scene_context)
            )
            yield "IK controls", index, len(ik_proposals)
        if ik_proposals and finger_channels:
            original_shapes = target_pose.shapes
            target_pose.bones = finger_channels
            target_pose.shapes = {}
            target_pose.invalidate()
            finger_commit = target_pose.iter_commit(scene_context.update)
            while True:
                try:
                    current, total = next(finger_commit)
                except StopIteration as stopped:
                    deferred_bones, _deferred_shapes = stopped.value
                    break
                yield "Finger controls", int(current), int(total)
            changed_bones.update(deferred_bones)
            target_pose.bones = all_channels
            target_pose.shapes = original_shapes
            target_pose.invalidate()
        return changed_bones, changed_shapes
    except BaseException:
        if "all_channels" in locals():
            target_pose.bones = all_channels
        if "original_shapes" in locals():
            target_pose.shapes = original_shapes
        target_pose.invalidate()
        for bone, basis in bone_basis.items():
            bone.matrix_basis = basis
        for bone, property_name, existed, value in property_values:
            if existed:
                bone[property_name] = value
            elif property_name in bone:
                del bone[property_name]
        for block, value in shape_values:
            block.value = value
        target_pose.armature.update_tag(refresh={"OBJECT"})
        scene_context.update()
        raise


def _commit_target_pose(
    target_pose: TargetPose,
    ik_proposals: list[dict],
    *,
    scene_context: SceneContext,
) -> tuple[set[object], list[tuple[object, object]]]:
    iterator = _iter_commit_target_pose(
        target_pose, ik_proposals, scene_context=scene_context
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


def build_target_pose(
    armature,
    frame: CanonicalFrame,
    settings,
    *,
    post_process: bool = True,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
    postprocess_config=None,
    rig_plan: rig_planning.RigSolvePlan | None = None,
) -> TargetPose:
    """Create a complete target pose without mutating Blender data."""
    config = armature.mtp_rig_config
    if rig_plan is None:
        entries = mapping.mapping_by_id(config)
        rig_plan = rig_planning.compile_rig_solve_plan(
            armature,
            entries,
            lambda: target_rest_basis(armature, entries),
        )
    else:
        if rig_plan.armature_pointer != int(armature.as_pointer()):
            raise ValueError("Rig solve plan belongs to a different armature")
        entries = rig_plan.entries
    source_space = source_root_basis(frame)
    target_space = target_basis(
        armature,
        entries,
        rest_basis=rig_plan.rest_basis,
    )
    target_pose = TargetPose(
        armature,
        frame,
        entries,
        source_space,
        target_space,
        rig_plan=rig_plan,
    )

    if settings.use_body and frame.body:
        _apply_body(
            armature,
            entries,
            frame,
            source_space,
            target_space,
            target_pose,
        )
    if settings.use_hands and frame.hands:
        _apply_hands(
            armature,
            entries,
            frame,
            source_space,
            target_space,
            target_pose,
        )
    if (
        settings.use_face
        and settings.face_output == FaceOutput.BONES
        and getattr(settings, "provider", "MEDIAPIPE") == "MEDIAPIPE"
        and frame.blendshapes
    ):
        _apply_eye_gaze(armature, entries, frame, target_pose)
    if settings.use_face and getattr(settings, "provider", "MEDIAPIPE") == "MEDIAPIPE":
        if settings.face_output in {FaceOutput.ARKIT, FaceOutput.FACS}:
            if frame.blendshapes:
                _apply_shapes(
                    armature,
                    entries,
                    frame,
                    settings.face_output,
                    target_pose,
                )
        elif settings.face_output == FaceOutput.BONES and frame.blendshapes:
            _apply_face_controls(
                armature,
                entries,
                frame,
                target_pose,
            )

    if post_process:
        run_pipeline(
            target_pose,
            postprocess_config or compile_postprocess_config(config),
            session=session,
            module_filter=module_filter,
        )
    return target_pose


def iter_apply_frame(
    armature,
    frame: CanonicalFrame,
    settings,
    *,
    keyframe: bool = False,
    scene_frame: float = 0.0,
    post_process: bool = True,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
    postprocess_config=None,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    """Apply one frame cooperatively so Blender can redraw between stages."""
    config = armature.mtp_rig_config
    request_entries = getattr(settings, "mapping_entries", None)
    entries = request_entries or mapping.mapping_by_id(config)
    rig_plan = None
    if request_entries:
        rig_plan = rig_planning.compile_rig_solve_plan(
            armature,
            request_entries,
            lambda: target_rest_basis(armature, request_entries),
        )
    yield SequenceApplyProgress("Solving target pose", 60, 100)
    if not bool(getattr(settings, "keep_original_root_position", False)):
        _root_kind, root_target = _sequence_root_target(armature, entries)
        _reset_root_location(root_target, scene_context=scene_context)
    target_pose = build_target_pose(
        armature,
        frame,
        settings,
        post_process=False,
        rig_plan=rig_plan,
    )
    if post_process:
        compiled_config = (
            postprocess_config
            or getattr(settings, "postprocess_config", None)
            or compile_postprocess_config(config)
        )
        pipeline = iter_pipeline(
            target_pose,
            compiled_config,
            session=session,
            module_filter=module_filter,
        )
        while True:
            try:
                display_name, index, total = next(pipeline)
            except StopIteration:
                break
            current = 65 + int(20 * (index - 1) / max(total, 1))
            yield SequenceApplyProgress(
                f"Post-processing: {display_name}",
                current,
                100,
            )

    yield SequenceApplyProgress("Preparing rig controls", 88, 100)
    ik_proposals = _prepare_ik_output(target_pose)
    yield SequenceApplyProgress("Committing pose", 93, 100)
    commit = _iter_commit_target_pose(
        target_pose, ik_proposals, scene_context=scene_context
    )
    while True:
        try:
            commit_stage, _commit_current, _commit_total = next(commit)
        except StopIteration as stopped:
            changed_bones, changed_shapes = stopped.value
            break
        commit_ratio = float(_commit_current) / max(int(_commit_total), 1)
        yield SequenceApplyProgress(
            f"Committing pose: {commit_stage}",
            93 + int(3 * commit_ratio),
            100,
        )
    for warning in target_pose.warnings:
        if warning not in frame.warnings:
            frame.warnings.append(warning)

    yield SequenceApplyProgress("Updating scene", 97, 100)
    scene_context.update()
    if keyframe:
        yield SequenceApplyProgress("Writing keyframe", 98, 100)
        for bone in changed_bones:
            _key_bone(bone, scene_frame)
        for channel in target_pose.bones.values():
            bone = channel.bone
            if bone in changed_bones:
                for property_name in channel.properties:
                    bone.keyframe_insert(
                        data_path=f'["{property_name}"]',
                        frame=scene_frame,
                        group=bone.name,
                    )
                    _set_key_interpolation(
                        bone.id_data,
                        bone.path_from_id(f'["{property_name}"]'),
                        scene_frame,
                        "LINEAR",
                    )
        for _obj, block in changed_shapes:
            _key_shape(block, scene_frame)
    yield SequenceApplyProgress("Finalizing", 100, 100)
    return len(changed_bones), len(changed_shapes)


def apply_frame(
    armature,
    frame: CanonicalFrame,
    settings,
    *,
    keyframe: bool = False,
    scene_frame: float = 0.0,
    post_process: bool = True,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
    scene_context: SceneContext,
) -> tuple[int, int]:
    iterator = iter_apply_frame(
        armature,
        frame,
        settings,
        keyframe=keyframe,
        scene_frame=scene_frame,
        post_process=post_process,
        session=session,
        module_filter=module_filter,
        scene_context=scene_context,
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value
