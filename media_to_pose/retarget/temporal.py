"""Temporal smoothing, gap filling and hand stabilisation of baked samples."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from mathutils import Matrix, Vector

from .. import (
    animation_cleanup,
)
from ..core.scene import SceneContext
from .grounding import _has_groundable_body_channels, _iter_ground_contacts
from .samples import (
    SequenceApplyProgress,
    _append_bake_warning,
    _apply_baked_sample_pose,
    _BakedBoneSample,
    _BakedFrameSample,
    _BakedShapeSample,
    _continuous_runs,
    _restore_sequence_state,
    _scene_set_float,
    _sequence_state,
    _sequence_timestamps,
    _store_bone_basis,
)
from .solve.common import (
    _FINGER_ARTICULATION_LIMITS,
    _bounded_parent_relative_direction_delta,
)


def _fill_short_bone_gaps(
    samples: list[_BakedFrameSample],
    timestamps: Sequence[float],
) -> None:
    names = {name for sample in samples for name in sample.bones}
    for name in names:
        present = [
            index for index, sample in enumerate(samples) if name in sample.bones
        ]
        for left, right in zip(present, present[1:], strict=False):
            gap = right - left - 1
            if gap < 1:
                continue
            span = timestamps[right] - timestamps[left]
            local_steps = [
                timestamps[index + 1] - timestamps[index]
                for index in range(max(0, left - 2), min(len(samples) - 1, right + 2))
                if timestamps[index + 1] > timestamps[index]
            ]
            normal = sorted(local_steps)[len(local_steps) // 2] if local_steps else span
            if span <= 0.0 or span > 0.5 or span > 2.5 * normal * (gap + 1):
                continue
            first, last = samples[left].bones[name], samples[right].bones[name]
            if not (first.key_rotation or first.key_location):
                continue
            # A short hole is safe to bridge only when it is bounded by two
            # credible observations. Otherwise interpolation would turn a
            # weak detection into apparently authoritative animation.
            if min(first.confidence, last.confidence) < 0.35:
                continue
            for offset in range(1, gap + 1):
                factor = min(
                    max(
                        (timestamps[left + offset] - timestamps[left]) / span,
                        0.0,
                    ),
                    1.0,
                )
                samples[left + offset].bones[name] = _BakedBoneSample(
                    bone=first.bone,
                    location=first.location.lerp(last.location, factor),
                    rotation=first.rotation.slerp(last.rotation, factor).normalized(),
                    scale=first.scale.copy(),
                    key_location=first.key_location and last.key_location,
                    key_rotation=first.key_rotation and last.key_rotation,
                    semantic_id=first.semantic_id or last.semantic_id,
                    confidence=min(first.confidence, last.confidence),
                    properties={
                        name: float(first.properties[name])
                        + (float(last.properties[name]) - float(first.properties[name]))
                        * factor
                        for name in first.properties.keys() & last.properties.keys()
                    },
                )


def _continuous_property(name: str) -> bool:
    lowered = name.casefold()
    return not any(
        token in lowered
        for token in ("switch", "ik_fk", "fk_ik", "visibility", "mute", "owner")
    )


def _project_temporal_finger_hinges(
    armature,
    samples: list[_BakedFrameSample],
    *,
    scene_context: SceneContext,
) -> int:
    """Reapply anatomical finger limits after independent channel filtering."""

    corrected = 0
    temporary = _sequence_state(armature, scene_context=scene_context)
    try:
        for sample in samples:
            _scene_set_float(scene_context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample, scene_context=scene_context)
            by_semantic = {
                channel.semantic_id: channel
                for channel in sample.bones.values()
                if channel.semantic_id
            }
            for side in ("L", "R"):
                hand_channel = by_semantic.get(f"HAND_{side}")
                if hand_channel is None:
                    continue
                hand = hand_channel.bone
                first_channels = [
                    by_semantic.get(f"{finger}_01_{side}")
                    for finger in ("INDEX", "MIDDLE", "RING", "PINKY")
                ]
                if any(channel is None for channel in first_channels):
                    continue
                rest_hand = hand.bone.matrix_local.to_3x3()
                hand_delta = (
                    (hand.matrix.to_3x3() @ rest_hand.inverted_safe())
                    .to_quaternion()
                    .normalized()
                )
                primary = (
                    sum(
                        (channel.bone.bone.head_local for channel in first_channels),
                        Vector(),
                    )
                    / len(first_channels)
                    - hand.bone.head_local
                )
                lateral = (
                    first_channels[0].bone.bone.head_local
                    - first_channels[-1].bone.bone.head_local
                )
                normal = primary.cross(lateral)
                if normal.length <= 1.0e-8:
                    continue
                normal.normalize()
                for finger in ("INDEX", "MIDDLE", "RING", "PINKY"):
                    parent_delta = hand_delta
                    for segment in range(1, 4):
                        channel = by_semantic.get(f"{finger}_{segment:02d}_{side}")
                        if channel is None:
                            break
                        bone = channel.bone
                        rest_direction = (
                            bone.bone.tail_local - bone.bone.head_local
                        ).normalized()
                        current_direction = (bone.tail - bone.head).normalized()
                        limit = _FINGER_ARTICULATION_LIMITS["FINGER"][segment - 1]
                        desired_delta, constrained = (
                            _bounded_parent_relative_direction_delta(
                                rest_direction,
                                current_direction,
                                parent_delta,
                                limit,
                                bend_reference=normal,
                                maximum_off_plane=math.radians(
                                    35.0 if segment == 1 else 25.0
                                ),
                                flexion_bounds=(
                                    (
                                        math.radians(-25.0 if segment == 1 else -15.0),
                                        limit,
                                    )
                                    if side == "L"
                                    else (
                                        -limit,
                                        math.radians(25.0 if segment == 1 else 15.0),
                                    )
                                ),
                            )
                        )
                        if constrained:
                            rest_rotation = bone.bone.matrix_local.to_3x3()
                            desired_rotation = (
                                (desired_delta.to_matrix() @ rest_rotation)
                                .to_quaternion()
                                .normalized()
                            )
                            bone.matrix = Matrix.LocRotScale(
                                bone.matrix.translation,
                                desired_rotation,
                                bone.matrix.to_scale(),
                            )
                            scene_context.update()
                            _store_bone_basis(sample, bone)
                            corrected += 1
                        parent_delta = desired_delta
    finally:
        _restore_sequence_state(armature, temporary, scene_context=scene_context)
    return corrected


def _iter_temporal_channels(
    armature,
    samples: list[_BakedFrameSample],
    settings,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    timestamps = _sequence_timestamps(samples)
    smooth_motion = bool(settings.smooth_motion)
    strength = float(settings.smoothing_strength)
    stabilize_hands = bool(settings.stabilize_hands)
    finger_maximum_speed = float(settings.finger_max_angular_speed)
    wrist_maximum_speed = float(settings.wrist_max_angular_speed)
    hand_reacquire_delay = float(settings.hand_reacquire_delay)
    _fill_short_bone_gaps(samples, timestamps)
    names = {name for sample in samples for name in sample.bones}
    shape_keys = {key for sample in samples for key in sample.shapes}
    properties = {
        (name, property_name)
        for sample in samples
        for name, channel in sample.bones.items()
        for property_name in channel.properties
        if _continuous_property(property_name)
    }
    object_indices = [
        index for index, sample in enumerate(samples) if sample.root_object
    ]
    total = len(names) + len(shape_keys) + len(properties) + bool(object_indices)
    completed = 0
    repaired = 0
    low_confidence = 0
    for name in sorted(names):
        location_indices = [
            index
            for index, sample in enumerate(samples)
            if name in sample.bones and sample.bones[name].key_location
        ]
        rotation_indices = [
            index
            for index, sample in enumerate(samples)
            if name in sample.bones and sample.bones[name].key_rotation
        ]
        for run in _continuous_runs(location_indices, timestamps):
            channels = [samples[index].bones[name] for index in run]
            run_times = [timestamps[index] for index in run]
            confidences = [float(channel.confidence) for channel in channels]
            low_confidence += sum(value < 0.2 for value in confidences)
            values = [channel.location.copy() for channel in channels]
            if smooth_motion:
                values = animation_cleanup.smooth_vectors(
                    values, run_times, confidences, strength
                )
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            for channel, value in zip(channels, values, strict=True):
                channel.location = value
        for run in _continuous_runs(rotation_indices, timestamps):
            channels = [samples[index].bones[name] for index in run]
            run_times = [timestamps[index] for index in run]
            confidences = [float(channel.confidence) for channel in channels]
            low_confidence += sum(value < 0.2 for value in confidences)
            values = [channel.rotation.copy() for channel in channels]
            semantic_id = next(
                (channel.semantic_id for channel in channels if channel.semantic_id),
                "",
            )
            is_finger = semantic_id.startswith(
                ("THUMB_", "INDEX_", "MIDDLE_", "RING_", "PINKY_")
            )
            is_wrist = semantic_id.startswith("HAND_")
            if stabilize_hands and (is_finger or is_wrist):
                values, hold_count = animation_cleanup.stabilize_quaternion_jumps(
                    values,
                    run_times,
                    finger_maximum_speed if is_finger else wrist_maximum_speed,
                    reacquire_seconds=hand_reacquire_delay,
                )
                repaired += hold_count
                yield SequenceApplyProgress(
                    "Stabilizing hand rotations", completed, total
                )
            if smooth_motion and not (stabilize_hands and is_finger):
                smoother = (
                    animation_cleanup.smooth_quaternions_bounded
                    if is_wrist
                    else animation_cleanup.smooth_quaternions
                )
                values = smoother(values, run_times, confidences, strength)
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            if stabilize_hands and (is_finger or is_wrist):
                values, speed_count = animation_cleanup.limit_quaternion_speed(
                    values,
                    run_times,
                    finger_maximum_speed if is_finger else wrist_maximum_speed,
                )
                repaired += speed_count
            for channel, value in zip(channels, values, strict=True):
                channel.rotation = value
        completed += 1

    for key in sorted(shape_keys):
        indices = [
            index for index, sample in enumerate(samples) if key in sample.shapes
        ]
        for left, right in zip(indices, indices[1:], strict=False):
            gap = right - left - 1
            if gap < 1 or gap > 2:
                continue
            if len(_continuous_runs(list(range(left, right + 1)), timestamps)) != 1:
                continue
            first, last = samples[left].shapes[key], samples[right].shapes[key]
            for offset in range(1, gap + 1):
                span = timestamps[right] - timestamps[left]
                factor = (
                    (timestamps[left + offset] - timestamps[left]) / span
                    if span > 1.0e-8
                    else offset / (gap + 1)
                )
                samples[left + offset].shapes[key] = _BakedShapeSample(
                    first.block,
                    first.value + (last.value - first.value) * factor,
                )
        indices = [
            index for index, sample in enumerate(samples) if key in sample.shapes
        ]
        for run in _continuous_runs(indices, timestamps):
            values = [samples[index].shapes[key].value for index in run]
            run_times = [timestamps[index] for index in run]
            if smooth_motion:
                values = animation_cleanup.smooth_scalars(
                    values, run_times, None, strength
                )
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            block = samples[run[0]].shapes[key].block
            minimum = float(getattr(block, "slider_min", 0.0))
            maximum = float(getattr(block, "slider_max", 1.0))
            for index, value in zip(run, values, strict=True):
                samples[index].shapes[key].value = min(max(value, minimum), maximum)
        completed += 1

    for name, property_name in sorted(properties):
        indices = [
            index
            for index, sample in enumerate(samples)
            if name in sample.bones and property_name in sample.bones[name].properties
        ]
        for run in _continuous_runs(indices, timestamps):
            channels = [samples[index].bones[name] for index in run]
            values = [float(channel.properties[property_name]) for channel in channels]
            run_times = [timestamps[index] for index in run]
            confidences = [float(channel.confidence) for channel in channels]
            if smooth_motion:
                values = animation_cleanup.smooth_scalars(
                    values, run_times, confidences, strength
                )
                yield SequenceApplyProgress("Smoothing animation", completed, total)
            for channel, value in zip(channels, values, strict=True):
                try:
                    limits = channel.bone.id_properties_ui(property_name).as_dict()
                except (AttributeError, KeyError, TypeError):
                    limits = {}
                if "min" in limits:
                    value = max(value, float(limits["min"]))
                if "max" in limits:
                    value = min(value, float(limits["max"]))
                channel.properties[property_name] = value
        completed += 1

    for run in _continuous_runs(object_indices, timestamps):
        run_times = [timestamps[index] for index in run]
        locations = [samples[index].root_object[0].copy() for index in run]
        rotations = [samples[index].root_object[1].copy() for index in run]
        confidences = [
            float(samples[index].grounding.root_confidence)
            if samples[index].grounding is not None
            else 1.0
            for index in run
        ]
        if smooth_motion:
            locations = animation_cleanup.smooth_vectors(
                locations, run_times, confidences, strength
            )
            rotations = animation_cleanup.smooth_quaternions(
                rotations, run_times, confidences, strength
            )
            yield SequenceApplyProgress("Smoothing animation", completed, total)
        for index, location, rotation in zip(run, locations, rotations, strict=True):
            _old_location, _old_rotation, scale = samples[index].root_object
            samples[index].root_object = (location, rotation, scale)
    if object_indices:
        completed += 1

    if stabilize_hands:
        anatomical_count = _project_temporal_finger_hinges(
            armature, samples, scene_context=scene_context
        )
        repaired += anatomical_count
        if anatomical_count:
            yield SequenceApplyProgress(
                "Projecting anatomical finger hinges", total, total
            )

    if repaired:
        _append_bake_warning(
            samples,
            f"Temporal Correction repaired {repaired} noisy motion sample(s)",
        )
    if low_confidence > max(6, len(samples)):
        _append_bake_warning(
            samples,
            "Temporal Correction reduced the influence of low-confidence motion samples",
        )


def _iter_temporal_correction(
    armature,
    entries,
    samples: list[_BakedFrameSample],
    settings,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_up: Vector,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    if settings.stabilize_hands:
        for sample in samples:
            for channel in sample.bones.values():
                if channel.semantic_id.startswith(
                    (
                        "HAND_",
                        "THUMB_",
                        "INDEX_",
                        "MIDDLE_",
                        "RING_",
                        "PINKY_",
                    )
                ):
                    channel.bone.rotation_mode = "QUATERNION"
    if settings.smooth_motion or settings.stabilize_hands:
        yield from _iter_temporal_channels(
            armature, samples, settings, scene_context=scene_context
        )
    if settings.ground_contacts and _has_groundable_body_channels(samples):
        yield from _iter_ground_contacts(
            armature,
            entries,
            samples,
            float(settings.smoothing_strength),
            effective_mode,
            root_kind,
            root_target,
            target_up,
            scene_context=scene_context,
        )
