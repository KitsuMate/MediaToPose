"""Per-frame bake samples shared by the sequence stages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from mathutils import Matrix, Quaternion, Vector

from .. import camera_motion_correction, grounded_cleanup
from ..core.scene import SceneContext
from ..rig.face_schemes import matching_shape_blocks


@dataclass
class _BakedBoneSample:
    bone: object
    location: Vector
    rotation: Quaternion
    scale: Vector
    key_location: bool = False
    key_rotation: bool = False
    semantic_id: str = ""
    confidence: float = 1.0
    properties: dict[str, float] = field(default_factory=dict)


@dataclass
class _BakedShapeSample:
    block: object
    value: float


@dataclass
class _BakedFrameSample:
    destination: float
    warning_sink: list[str]
    timestamp_seconds: float = 0.0
    bones: dict[str, _BakedBoneSample] = field(default_factory=dict)
    shapes: dict[tuple[int, str], _BakedShapeSample] = field(default_factory=dict)
    root_object: tuple[Vector, Quaternion, Vector] | None = None
    grounding: grounded_cleanup.GroundFrame | None = None
    ik_legs: dict[str, dict] = field(default_factory=dict)
    body_height: float = 0.0
    camera_motion_input: camera_motion_correction.CameraMotionInput | None = None
    rohm_joint_offsets: dict[str, Vector] = field(default_factory=dict)
    rohm_rotations: dict[str, Quaternion] = field(default_factory=dict)
    rohm_confidence: dict[str, float] = field(default_factory=dict)
    rohm_root_confidence: float = 0.0
    source_body_3d: dict[str, Vector] = field(default_factory=dict)
    source_body_2d: dict[str, Vector] = field(default_factory=dict)


@dataclass
class _RohmFrameInput:
    """Pre-postprocess semantic motion in target-independent anatomy space."""

    joint_offsets: dict[str, Vector] = field(default_factory=dict)
    global_rotations: dict[str, Quaternion] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    body_height: float = 0.0


@dataclass(frozen=True)
class _RootMotionSample:
    timestamp_seconds: float
    observation: object
    source_basis: Matrix
    source_height: float


@dataclass(frozen=True)
class SequenceApplyProgress:
    phase: str
    current: int
    total: int


def _scene_set_float(scene, value: float) -> None:
    integer = math.floor(float(value))
    scene.frame_set(integer, subframe=float(value) - integer)


def _shape_blocks_for_config(armature):
    for entry in armature.mtp_rig_config.mappings:
        if entry.section != "SHAPES" or not entry.target_name:
            continue
        yield from matching_shape_blocks(armature, entry)


def _sequence_state(armature, *, scene_context: SceneContext) -> dict:
    properties = {}
    for bone in armature.pose.bones:
        properties[bone.name] = {
            name: value
            for name in bone.keys()  # noqa: SIM118 - Blender IDProperty API
            if isinstance((value := bone.get(name)), (bool, int, float))
        }
    return {
        "frame": (
            int(scene_context.scene.frame_current),
            float(scene_context.scene.frame_subframe),
        ),
        "world": armature.matrix_world.copy(),
        "bones": {bone.name: bone.matrix_basis.copy() for bone in armature.pose.bones},
        "properties": properties,
        "shapes": {
            (block.as_pointer(), block.name): (block, float(block.value))
            for _obj, block in _shape_blocks_for_config(armature)
        },
    }


def _restore_sequence_state(
    armature, state: dict, *, scene_context: SceneContext
) -> None:
    frame, subframe = state["frame"]
    scene_context.scene.frame_set(frame, subframe=subframe)
    armature.matrix_world = state["world"]
    for name, basis in state["bones"].items():
        bone = armature.pose.bones.get(name)
        if bone is not None:
            bone.matrix_basis = basis
    for name, values in state["properties"].items():
        bone = armature.pose.bones.get(name)
        if bone is None:
            continue
        for property_name, value in values.items():
            if property_name in bone:
                bone[property_name] = value
    for block, value in state["shapes"].values():
        block.value = value
    armature.update_tag(refresh={"OBJECT"})
    scene_context.update()


def _sequence_timestamps(samples: Sequence[_BakedFrameSample]) -> list[float]:
    return [float(sample.timestamp_seconds) for sample in samples]


def _continuous_runs(
    indices: Sequence[int],
    timestamps: Sequence[float],
) -> list[list[int]]:
    intervals = [
        timestamps[right] - timestamps[left]
        for left, right in zip(
            range(len(timestamps)), range(1, len(timestamps)), strict=False
        )
        if timestamps[right] > timestamps[left]
    ]
    normal = float(sorted(intervals)[len(intervals) // 2]) if intervals else 0.0
    runs: list[list[int]] = []
    for index in indices:
        discontinuity = not runs or index != runs[-1][-1] + 1
        if not discontinuity:
            delta = timestamps[index] - timestamps[runs[-1][-1]]
            discontinuity = delta <= 0.0 or normal > 0.0 and delta > 2.5 * normal
        if discontinuity:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def _apply_baked_sample_pose(
    armature, sample: _BakedFrameSample, *, scene_context: SceneContext
) -> None:
    if sample.root_object is not None:
        location, rotation, scale = sample.root_object
        armature.matrix_basis = Matrix.LocRotScale(location, rotation, scale)
    for channel in sorted(
        sample.bones.values(), key=lambda item: len(item.bone.parent_recursive)
    ):
        channel.bone.matrix_basis = Matrix.LocRotScale(
            channel.location, channel.rotation, channel.scale
        )
        for name, value in channel.properties.items():
            channel.bone[name] = value
    armature.update_tag(refresh={"OBJECT"})
    scene_context.update()


def _store_bone_basis(sample: _BakedFrameSample, bone) -> None:
    channel = sample.bones.get(bone.name)
    if channel is None:
        return
    location, rotation, scale = bone.matrix_basis.decompose()
    channel.location = location.copy()
    channel.rotation = rotation.normalized()
    channel.scale = scale.copy()


def _append_bake_warning(
    samples: Sequence[_BakedFrameSample],
    warning: str,
) -> None:
    if samples and warning not in samples[0].warning_sink:
        samples[0].warning_sink.append(warning)


def _drain(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value


def _translate_grounding(frame, delta: Vector) -> None:
    if frame is None or delta.length_squared <= 1.0e-20:
        return
    frame.root += delta
    for point in frame.points.values():
        point.position += delta
    for name in tuple(frame.joints):
        frame.joints[name] += delta
