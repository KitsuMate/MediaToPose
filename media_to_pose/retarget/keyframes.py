"""Keyframe writing: rotation continuity, interpolation and baked streams."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from ..core.scene import SceneContext
from .samples import SequenceApplyProgress, _BakedFrameSample, _drain, _scene_set_float


@dataclass
class _KeyframeWriteState:
    armature: object
    action: object = None
    previous_rotations: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    previous_object_rotation: list[float] | None = None
    total_bones: int = 0
    total_shapes: int = 0

    def __post_init__(self) -> None:
        if self.action is None:
            self.action = _ensure_action(self.armature)

    def attach(self) -> None:
        self.armature.animation_data_create().action = self.action

    def detach(self) -> None:
        animation = self.armature.animation_data
        if animation is not None and animation.action is self.action:
            animation.action = None


def _rotation_data_path(pose_bone) -> str:
    if pose_bone.rotation_mode == "QUATERNION":
        return "rotation_quaternion"
    if pose_bone.rotation_mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    return "rotation_euler"


def _action_fcurves(action) -> Iterator[object]:
    """Yield F-Curves from a Blender 5.2 layered Action."""
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                yield from channelbag.fcurves


def _set_key_interpolation(
    datablock,
    data_path: str,
    frame: float,
    interpolation: str,
) -> None:
    animation_data = getattr(datablock, "animation_data", None)
    action = animation_data.action if animation_data else None
    if not action:
        return
    for fcurve in _action_fcurves(action):
        if fcurve.data_path != data_path:
            continue
        for point in fcurve.keyframe_points:
            if abs(float(point.co.x) - float(frame)) < 1.0e-5:
                point.interpolation = interpolation


def _previous_key_values(
    pose_bone,
    data_path: str,
    frame: float,
    size: int,
) -> list[float] | None:
    animation = pose_bone.id_data.animation_data
    action = animation.action if animation else None
    if action is None:
        return None
    resolved_path = pose_bone.path_from_id(data_path)
    curves = {
        curve.array_index: curve
        for curve in _action_fcurves(action)
        if curve.data_path == resolved_path
    }
    if any(index not in curves for index in range(size)):
        return None
    previous_frames = [
        float(point.co.x)
        for curve in curves.values()
        for point in curve.keyframe_points
        if float(point.co.x) < float(frame) - 1.0e-5
    ]
    if not previous_frames:
        return None
    previous_frame = max(previous_frames)
    return [float(curves[index].evaluate(previous_frame)) for index in range(size)]


def _make_rotation_key_compatible(
    pose_bone,
    data_path: str,
    frame: float,
) -> None:
    """Choose the closest equivalent rotation to the preceding key."""
    if data_path == "rotation_quaternion":
        previous = _previous_key_values(pose_bone, data_path, frame, 4)
        if previous is None:
            return
        current = pose_bone.rotation_quaternion.copy()
        prior = Quaternion(previous)
        if current.dot(prior) < 0.0:
            current.negate()
            pose_bone.rotation_quaternion = current
        return
    if data_path == "rotation_axis_angle":
        previous = _previous_key_values(pose_bone, data_path, frame, 4)
        if previous is None:
            return
        current = pose_bone.rotation_axis_angle
        axis = Vector(current[1:4])
        if axis.length < 1.0e-8:
            axis = Vector(previous[1:4])
        if axis.length < 1.0e-8:
            axis = Vector((0.0, 1.0, 0.0))
        axis.normalize()
        previous_axis = Vector(previous[1:4])
        angle = float(current[0])
        if previous_axis.length > 1.0e-8 and axis.dot(previous_axis.normalized()) < 0.0:
            axis.negate()
            angle = -angle
        while angle - previous[0] > math.pi:
            angle -= math.tau
        while angle - previous[0] < -math.pi:
            angle += math.tau
        pose_bone.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        return
    previous = _previous_key_values(pose_bone, data_path, frame, 3)
    if previous is None:
        return
    current = pose_bone.rotation_euler.copy()
    current.make_compatible(Euler(previous, pose_bone.rotation_mode))
    pose_bone.rotation_euler = current


def _rotation_values_compatible_with(
    pose_bone,
    data_path: str,
    previous: Sequence[float] | None,
) -> list[float]:
    """Make a rotation continuous using an in-memory preceding sample."""
    if data_path == "rotation_quaternion":
        current = pose_bone.rotation_quaternion.copy()
        if previous is not None and current.dot(Quaternion(previous)) < 0.0:
            current.negate()
            pose_bone.rotation_quaternion = current
        return [float(value) for value in current]
    if data_path == "rotation_axis_angle":
        current = pose_bone.rotation_axis_angle
        axis = Vector(current[1:4])
        if axis.length < 1.0e-8 and previous is not None:
            axis = Vector(previous[1:4])
        if axis.length < 1.0e-8:
            axis = Vector((0.0, 1.0, 0.0))
        axis.normalize()
        angle = float(current[0])
        if previous is not None:
            previous_axis = Vector(previous[1:4])
            if (
                previous_axis.length > 1.0e-8
                and axis.dot(previous_axis.normalized()) < 0.0
            ):
                axis.negate()
                angle = -angle
            while angle - previous[0] > math.pi:
                angle -= math.tau
            while angle - previous[0] < -math.pi:
                angle += math.tau
        values = [angle, axis.x, axis.y, axis.z]
        pose_bone.rotation_axis_angle = values
        return values
    current = pose_bone.rotation_euler.copy()
    if previous is not None:
        current.make_compatible(Euler(previous, pose_bone.rotation_mode))
        pose_bone.rotation_euler = current
    return [float(value) for value in current]


def _key_bone_from_sample(
    pose_bone,
    frame: float,
    *,
    location: bool,
    rotation: bool,
    previous_rotations: dict[tuple[str, str], list[float]],
) -> None:
    """Insert one pre-solved sample without rescanning the growing action."""
    if location:
        pose_bone.keyframe_insert(
            data_path="location",
            frame=frame,
            group=pose_bone.name,
        )
    if not rotation:
        return
    data_path = _rotation_data_path(pose_bone)
    key = (pose_bone.name, data_path)
    previous = previous_rotations.get(key)
    if previous is None:
        size = 4 if data_path != "rotation_euler" else 3
        previous = _previous_key_values(
            pose_bone,
            data_path,
            frame,
            size,
        )
    previous_rotations[key] = _rotation_values_compatible_with(
        pose_bone,
        data_path,
        previous,
    )
    pose_bone.keyframe_insert(
        data_path=data_path,
        frame=frame,
        group=pose_bone.name,
    )


def _key_bone(
    pose_bone,
    frame: float,
    *,
    location: bool = True,
    rotation: bool = True,
) -> None:
    if location:
        pose_bone.keyframe_insert(
            data_path="location", frame=frame, group=pose_bone.name
        )
        _set_key_interpolation(
            pose_bone.id_data,
            pose_bone.path_from_id("location"),
            frame,
            "LINEAR",
        )
    if rotation:
        rotation_path = _rotation_data_path(pose_bone)
        _make_rotation_key_compatible(pose_bone, rotation_path, frame)
        pose_bone.keyframe_insert(
            data_path=rotation_path,
            frame=frame,
            group=pose_bone.name,
        )
        _set_key_interpolation(
            pose_bone.id_data,
            pose_bone.path_from_id(rotation_path),
            frame,
            "LINEAR",
        )


def _key_object_transform(obj, frame: float) -> None:
    obj.keyframe_insert(data_path="location", frame=frame)
    rotation_path = (
        "rotation_quaternion"
        if obj.rotation_mode == "QUATERNION"
        else "rotation_axis_angle"
        if obj.rotation_mode == "AXIS_ANGLE"
        else "rotation_euler"
    )
    animation = obj.animation_data
    action = animation.action if animation else None
    if action is not None:
        size = 4 if rotation_path != "rotation_euler" else 3
        curves = {
            curve.array_index: curve
            for curve in _action_fcurves(action)
            if curve.data_path == rotation_path
        }
        previous_frames = [
            float(point.co.x)
            for curve in curves.values()
            for point in curve.keyframe_points
            if float(point.co.x) < float(frame) - 1.0e-5
        ]
        if previous_frames and all(index in curves for index in range(size)):
            previous_frame = max(previous_frames)
            previous = [
                float(curves[index].evaluate(previous_frame)) for index in range(size)
            ]
            if rotation_path == "rotation_quaternion":
                current = obj.rotation_quaternion.copy()
                if current.dot(Quaternion(previous)) < 0.0:
                    current.negate()
                    obj.rotation_quaternion = current
            elif rotation_path == "rotation_euler":
                current = obj.rotation_euler.copy()
                current.make_compatible(Euler(previous, obj.rotation_mode))
                obj.rotation_euler = current
            else:
                current = obj.rotation_axis_angle
                axis = Vector(current[1:4])
                if axis.length < 1.0e-8:
                    axis = Vector(previous[1:4])
                if axis.length < 1.0e-8:
                    axis = Vector((0.0, 0.0, 1.0))
                axis.normalize()
                previous_axis = Vector(previous[1:4])
                angle = float(current[0])
                if (
                    previous_axis.length > 1.0e-8
                    and axis.dot(previous_axis.normalized()) < 0.0
                ):
                    axis.negate()
                    angle = -angle
                while angle - previous[0] > math.pi:
                    angle -= math.tau
                while angle - previous[0] < -math.pi:
                    angle += math.tau
                obj.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
    obj.keyframe_insert(data_path=rotation_path, frame=frame)
    _set_key_interpolation(obj, "location", frame, "LINEAR")
    _set_key_interpolation(obj, rotation_path, frame, "LINEAR")


def _remove_keys_in_interval(
    action,
    data_paths: set[str],
    start_frame: float,
    end_frame: float,
) -> None:
    """Remove keys for the requested channels within a replacement interval."""
    for curve in _action_fcurves(action):
        if curve.data_path not in data_paths:
            continue
        for point in reversed(list(curve.keyframe_points)):
            if start_frame - 1.0e-5 <= float(point.co.x) <= end_frame + 1.0e-5:
                curve.keyframe_points.remove(point, fast=True)
        curve.update()


def _key_shape(
    block,
    frame: float,
    *,
    set_interpolation: bool = True,
) -> None:
    block.keyframe_insert(data_path="value", frame=frame)
    if set_interpolation:
        _set_key_interpolation(
            block.id_data,
            block.path_from_id("value"),
            frame,
            "LINEAR",
        )


def _ensure_action(armature) -> bpy.types.Action:
    animation = armature.animation_data_create()
    if animation.action is None:
        animation.action = bpy.data.actions.new(f"{armature.name}_MediaToPose")
    return animation.action


@contextmanager
def keyframe_interpolation_defaults(interpolation: str, *, scene_context: SceneContext):
    if scene_context.preferences is None:
        yield
        return
    preferences = scene_context.preferences.edit
    previous = preferences.keyframe_new_interpolation_type
    preferences.keyframe_new_interpolation_type = interpolation
    try:
        yield
    finally:
        preferences.keyframe_new_interpolation_type = previous


@contextmanager
def linear_keyframe_defaults(*, scene_context: SceneContext):
    with keyframe_interpolation_defaults("LINEAR", scene_context=scene_context):
        yield


def _root_data_paths(root_kind: str, root_target) -> set[str]:
    if root_kind == "BONE":
        return {
            root_target.path_from_id("location"),
            root_target.path_from_id("rotation_euler"),
            root_target.path_from_id("rotation_quaternion"),
            root_target.path_from_id("rotation_axis_angle"),
        }
    return {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle"}


def _key_object_from_stream(
    state: _KeyframeWriteState,
    frame: float,
) -> None:
    obj = state.armature
    obj.keyframe_insert(data_path="location", frame=frame)
    data_path = _rotation_data_path(obj)
    state.previous_object_rotation = _rotation_values_compatible_with(
        obj,
        data_path,
        state.previous_object_rotation,
    )
    obj.keyframe_insert(data_path=data_path, frame=frame)


def _iter_write_baked_sample(
    state: _KeyframeWriteState,
    sample: _BakedFrameSample,
    frame_index: int,
    total_frames: int,
    *,
    phase: str,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    """Write one solved frame in bounded batches and release its caller."""
    state.attach()
    channels = sorted(
        sample.bones.values(),
        key=lambda item: len(item.bone.parent_recursive),
    )
    batch_size = 8
    for offset in range(0, len(channels), batch_size):
        with linear_keyframe_defaults(scene_context=scene_context):
            for channel in channels[offset : offset + batch_size]:
                channel.bone.matrix_basis = Matrix.LocRotScale(
                    channel.location,
                    channel.rotation,
                    channel.scale,
                )
                if channel.key_location or channel.key_rotation:
                    _key_bone_from_sample(
                        channel.bone,
                        sample.destination,
                        location=channel.key_location,
                        rotation=channel.key_rotation,
                        previous_rotations=state.previous_rotations,
                    )
                    state.total_bones += 1
                for property_name, value in channel.properties.items():
                    channel.bone[property_name] = value
                    channel.bone.keyframe_insert(
                        data_path=f'["{property_name}"]',
                        frame=sample.destination,
                        group=channel.bone.name,
                    )
        state.detach()
        yield SequenceApplyProgress(phase, frame_index - 1, total_frames)
        state.attach()

    if sample.root_object is not None:
        location, rotation, scale = sample.root_object
        state.armature.matrix_basis = Matrix.LocRotScale(
            location,
            rotation,
            scale,
        )
        with linear_keyframe_defaults(scene_context=scene_context):
            _key_object_from_stream(state, sample.destination)
        state.total_bones += 1

    shapes = list(sample.shapes.values())
    for offset in range(0, len(shapes), batch_size):
        with linear_keyframe_defaults(scene_context=scene_context):
            for shape in shapes[offset : offset + batch_size]:
                shape.block.value = shape.value
                _key_shape(
                    shape.block,
                    sample.destination,
                    set_interpolation=False,
                )
                state.total_shapes += 1
        state.detach()
        yield SequenceApplyProgress(phase, frame_index - 1, total_frames)
        state.attach()

    state.detach()
    yield SequenceApplyProgress(phase, frame_index, total_frames)


def _finish_keyframe_stream(
    state: _KeyframeWriteState,
    final_frame: float,
    *,
    scene_context: SceneContext,
) -> tuple[int, int]:
    state.attach()
    _scene_set_float(scene_context.scene, final_frame)
    state.armature.update_tag(refresh={"OBJECT"})
    scene_context.update()
    return state.total_bones, state.total_shapes


def _iter_commit_baked_samples(
    armature,
    samples: list[_BakedFrameSample],
    root_kind: str,
    root_target,
    root_enabled: bool,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    action = _ensure_action(armature)
    start_frame = samples[0].destination
    end_frame = samples[-1].destination
    replacement_paths: set[str] = set()
    for sample in samples:
        for channel in sample.bones.values():
            if channel.key_location:
                replacement_paths.add(channel.bone.path_from_id("location"))
            if channel.key_rotation:
                replacement_paths.update(
                    {
                        channel.bone.path_from_id("rotation_euler"),
                        channel.bone.path_from_id("rotation_quaternion"),
                        channel.bone.path_from_id("rotation_axis_angle"),
                    }
                )
            replacement_paths.update(
                channel.bone.path_from_id(f'["{property_name}"]')
                for property_name in channel.properties
            )
    if root_enabled:
        replacement_paths.update(_root_data_paths(root_kind, root_target))
    _remove_keys_in_interval(action, replacement_paths, start_frame, end_frame)

    shape_paths_by_data: dict[object, set[str]] = {}
    for sample in samples:
        for shape in sample.shapes.values():
            shape_paths_by_data.setdefault(shape.block.id_data, set()).add(
                shape.block.path_from_id("value")
            )
    for shape_data, shape_paths in shape_paths_by_data.items():
        animation = getattr(shape_data, "animation_data", None)
        if animation is not None and animation.action is not None:
            _remove_keys_in_interval(
                animation.action,
                shape_paths,
                start_frame,
                end_frame,
            )
    state = _KeyframeWriteState(armature)
    total_frames = len(samples)
    for index, sample in enumerate(samples, 1):
        yield from _iter_write_baked_sample(
            state,
            sample,
            index,
            total_frames,
            phase="Writing keyframes",
            scene_context=scene_context,
        )
    for curve in _action_fcurves(action):
        if curve.data_path not in replacement_paths:
            continue
        for point in curve.keyframe_points:
            if start_frame - 1.0e-5 <= float(point.co.x) <= end_frame + 1.0e-5:
                point.interpolation = "LINEAR"
        curve.update()
    for shape_data, shape_paths in shape_paths_by_data.items():
        animation = getattr(shape_data, "animation_data", None)
        if animation is None or animation.action is None:
            continue
        for curve in _action_fcurves(animation.action):
            if curve.data_path not in shape_paths:
                continue
            for point in curve.keyframe_points:
                if start_frame - 1.0e-5 <= float(point.co.x) <= end_frame + 1.0e-5:
                    point.interpolation = "LINEAR"
            curve.update()
    return _finish_keyframe_stream(
        state, samples[-1].destination, scene_context=scene_context
    )


def _commit_baked_samples(
    armature,
    samples: list[_BakedFrameSample],
    root_kind: str,
    root_target,
    root_enabled: bool,
    *,
    scene_context: SceneContext,
) -> tuple[int, int]:
    return _drain(
        _iter_commit_baked_samples(
            armature,
            samples,
            root_kind,
            root_target,
            root_enabled,
            scene_context=scene_context,
        )
    )
