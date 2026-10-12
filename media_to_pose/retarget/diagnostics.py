"""Pose-fit diagnostics comparing the posed rig with the detection."""

from __future__ import annotations

import math

from mathutils import Vector

from .. import (
    mapping,
)
from ..constants import BODY_JOINT_PAIRS, HAND_JOINT_PAIRS
from ..models import CanonicalFrame
from .solve.common import (
    _FINGER_ARTICULATION_LIMITS,
    _bounded_parent_relative_direction_delta,
    body_points,
    evaluated_semantic_bone,
)
from .solve.frames import (
    _detected_palm_basis,
    _orthonormal_palm_basis,
    _source_head_orientation,
    _source_mhr_head_orientation,
    _target_body_reference,
    _target_palm_vectors,
    _to_target_direction,
    source_root_basis,
    target_basis,
)
from .target_pose import signed_twist_angle as _signed_twist_angle


def normalized_arm_positions(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> tuple[dict[str, Vector], dict[str, Vector]]:
    """Build comparable detected and evaluated arm joint positions.

    The detected arm is normalized to the evaluated target shoulder position
    and to the target rig's upper-arm and forearm bone lengths. This
    removes provider scale, source placement, and armature object transforms,
    leaving a position-based check of the solved joint directions.
    """
    side = side.upper()
    if side not in {"L", "R"}:
        raise ValueError("Arm side must be 'L' or 'R'")
    label = "LEFT" if side == "L" else "RIGHT"
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    roles = (
        f"UPPER_ARM_{side}",
        f"LOWER_ARM_{side}",
    )
    points = body_points(frame)
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None or not start.valid or not end.valid:
            raise ValueError(
                f"Detected {label.lower()} arm is missing {start_name} or {end_name}"
            )
    bones = [evaluated_semantic_bone(armature, entries, role) for role in roles]
    if any(bone is None for bone in bones):
        raise ValueError(f"Evaluated {label.lower()} arm chain is incomplete")

    source_space = source_root_basis(frame)
    target_space = target_basis(armature, entries)
    directions = []
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        start = points.get(start_name)
        end = points.get(end_name)
        directions.append(
            _to_target_direction(
                Vector(end.position) - Vector(start.position),
                source_space,
                target_space,
            )
        )

    joint_names = (
        f"{label}_SHOULDER",
        f"{label}_ELBOW",
        f"{label}_WRIST",
    )
    detected = {joint_names[0]: bones[0].head.copy()}
    previous = detected[joint_names[0]]
    for joint_name, direction, bone in zip(
        joint_names[1:], directions, bones, strict=False
    ):
        previous = previous + direction.normalized() * float(bone.bone.length)
        detected[joint_name] = previous
    evaluated = {
        joint_names[0]: bones[0].head.copy(),
        joint_names[1]: bones[0].tail.copy(),
        joint_names[2]: bones[1].tail.copy(),
    }
    if world_space:
        matrix = armature.matrix_world
        detected = {name: matrix @ position for name, position in detected.items()}
        evaluated = {name: matrix @ position for name, position in evaluated.items()}
    return detected, evaluated


def normalized_arm_position_errors(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> dict[str, float]:
    """Return normalized joint-position errors for an evaluated arm pose."""
    detected, evaluated = normalized_arm_positions(
        armature,
        frame,
        side,
        world_space=world_space,
    )
    chain_length = 0.0
    for start_name, end_name in zip(tuple(detected), tuple(detected)[1:], strict=False):
        chain_length += (detected[end_name] - detected[start_name]).length
    if chain_length < 1.0e-8:
        raise ValueError("Evaluated arm chain has zero length")
    return {
        name: (evaluated[name] - detected[name]).length / chain_length
        for name in detected
    }


def normalized_leg_positions(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> tuple[dict[str, Vector], dict[str, Vector]]:
    """Build comparable normalized source and evaluated leg positions."""
    side = side.upper()
    if side not in {"L", "R"}:
        raise ValueError("Leg side must be 'L' or 'R'")
    label = "LEFT" if side == "L" else "RIGHT"
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    roles = (
        f"UPPER_LEG_{side}",
        f"LOWER_LEG_{side}",
    )
    points = body_points(frame)
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None or not start.valid or not end.valid:
            raise ValueError(
                f"Detected {label.lower()} leg is missing {start_name} or {end_name}"
            )
    bones = [evaluated_semantic_bone(armature, entries, role) for role in roles]
    if any(bone is None for bone in bones):
        raise ValueError(f"Evaluated {label.lower()} leg chain is incomplete")

    source_space = source_root_basis(frame)
    target_space = target_basis(armature, entries)
    directions = []
    for role in roles:
        start_name, end_name = BODY_JOINT_PAIRS[role]
        directions.append(
            _to_target_direction(
                Vector(points[end_name].position) - Vector(points[start_name].position),
                source_space,
                target_space,
            )
        )
    joint_names = (
        f"{label}_HIP",
        f"{label}_KNEE",
        f"{label}_ANKLE",
    )
    detected = {joint_names[0]: bones[0].head.copy()}
    previous = detected[joint_names[0]]
    for joint_name, direction, bone in zip(
        joint_names[1:], directions, bones, strict=False
    ):
        previous = previous + direction.normalized() * float(bone.bone.length)
        detected[joint_name] = previous
    evaluated = {
        joint_names[0]: bones[0].head.copy(),
        joint_names[1]: bones[0].tail.copy(),
        joint_names[2]: bones[1].tail.copy(),
    }
    if world_space:
        matrix = armature.matrix_world
        detected = {name: matrix @ position for name, position in detected.items()}
        evaluated = {name: matrix @ position for name, position in evaluated.items()}
    return detected, evaluated


def normalized_leg_position_errors(
    armature,
    frame: CanonicalFrame,
    side: str,
    *,
    world_space: bool = False,
) -> dict[str, float]:
    """Return normalized joint-position errors for an evaluated leg pose."""
    detected, evaluated = normalized_leg_positions(
        armature,
        frame,
        side,
        world_space=world_space,
    )
    names = tuple(detected)
    chain_length = sum(
        (detected[end] - detected[start]).length
        for start, end in zip(names, names[1:], strict=False)
    )
    if chain_length < 1.0e-8:
        raise ValueError("Evaluated leg chain has zero length")
    return {
        name: (evaluated[name] - detected[name]).length / chain_length
        for name in detected
    }


def normalized_head_position_errors(
    armature,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Compare evaluated cranial orientation with normalized detections.

    The diagnostic intentionally measures positions after dependency-graph
    evaluation.  Source placement and scale are removed by anchoring each
    expected endpoint at the target bone head and using that target segment's
    length.
    """
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    points = body_points(frame)
    head_mid = points.get("HEAD_MID")
    if head_mid is None or not head_mid.valid:
        raise ValueError("Detected head center is missing")
    source_space = source_root_basis(frame)
    target_space = target_basis(armature, entries)
    errors: dict[str, float] = {}

    head = evaluated_semantic_bone(armature, entries, "HEAD")
    # The evaluated pose uses the compact MHR127 skull frame whenever SAM
    # supplies it.  Compare against that same authoritative observation;
    # falling back to the sparse eye/ear frame here reported a large false
    # error even when retargeting reproduced the MHR head exactly.
    orientation = _source_mhr_head_orientation(frame) or _source_head_orientation(
        points
    )
    if head is None or orientation is None:
        raise ValueError("Evaluated head or detected head frame is missing")
    head_length = max(float(head.length), 1.0e-8)
    expected_up = _to_target_direction(orientation[0], source_space, target_space)
    expected_tip = head.head + expected_up * head_length
    errors["HEAD_TIP"] = (head.tail - expected_tip).length / head_length

    expected_lateral = _to_target_direction(orientation[1], source_space, target_space)
    rest_lateral = _target_body_reference(armature, entries, "HEAD")
    if (
        rest_lateral is not None
        and rest_lateral.length > 1.0e-8
        and expected_lateral.length > 1.0e-8
    ):
        head_delta = (
            head.matrix.to_quaternion()
            @ head.bone.matrix_local.to_quaternion().inverted()
        )
        evaluated_lateral = head_delta @ rest_lateral.normalized()
        expected_side = head.head + expected_lateral.normalized() * head_length
        evaluated_side = head.head + evaluated_lateral.normalized() * head_length
        errors["HEAD_LATERAL"] = (evaluated_side - expected_side).length / head_length

    return errors


def finger_world_position_errors(
    armature,
    entries,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Return normalized evaluated endpoint errors for mapped phalanges.

    Rotation values are intentionally ignored: generated rigs can assign
    different local axes and roll to equivalent controls. Each error compares
    the evaluated bone tail with the detected direction projected from its
    evaluated head using that target segment's own length.
    """
    source_basis = source_root_basis(frame)
    rig_basis = target_basis(armature, entries)
    errors = {}
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name, {})
        detected_palm_basis = _detected_palm_basis(hand)
        target_primary, target_lateral = _target_palm_vectors(
            armature,
            entries,
            side_suffix,
        )
        target_palm_basis = _orthonormal_palm_basis(
            target_primary,
            target_lateral,
        )
        hand_bone = evaluated_semantic_bone(
            armature,
            entries,
            f"HAND_{side_suffix}",
        )
        hand_delta = None
        if hand_bone is not None:
            hand_delta = (
                (
                    hand_bone.matrix.to_3x3()
                    @ hand_bone.bone.matrix_local.to_3x3().inverted_safe()
                )
                .to_quaternion()
                .normalized()
            )
        for role_base, pair in HAND_JOINT_PAIRS.items():
            semantic_id = f"{role_base}_{side_suffix}"
            bone = evaluated_semantic_bone(armature, entries, semantic_id)
            start, end = hand.get(pair[0]), hand.get(pair[1])
            if (
                bone is None
                or start is None
                or end is None
                or not start.valid
                or not end.valid
            ):
                continue
            raw_direction = Vector(end.position) - Vector(start.position)
            if (
                detected_palm_basis is not None
                and target_palm_basis is not None
                and hand_delta is not None
            ):
                palm_local = detected_palm_basis.inverted_safe() @ raw_direction
                direction = hand_delta @ (target_palm_basis @ palm_local)
            else:
                direction = _to_target_direction(
                    raw_direction,
                    source_basis,
                    rig_basis,
                )
            if direction.length < 1.0e-8:
                continue
            segment_length = max(float(bone.bone.length), 1.0e-8)
            expected_tail = bone.head + direction.normalized() * segment_length
            errors[semantic_id] = (bone.tail - expected_tail).length / segment_length
    return errors


def finger_palm_roll_errors(
    armature,
    entries,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Return axial-roll error relative to parent-transported rest roll.

    Joint positions constrain each phalanx direction but not its roll. This
    diagnostic verifies that the missing degree of freedom follows the solved
    wrist and preceding phalanx while articulation contributes only swing.
    Values are radians in the shortest signed range.
    """
    errors = {}
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name, {})
        hand_bone = evaluated_semantic_bone(armature, entries, f"HAND_{side_suffix}")
        if not hand or hand_bone is None:
            continue
        hand_rest_rotation = hand_bone.bone.matrix_local.to_3x3()
        palm_delta = (
            (hand_bone.matrix.to_3x3() @ hand_rest_rotation.inverted_safe())
            .to_quaternion()
            .normalized()
        )
        parent_deltas = {
            finger: palm_delta
            for finger in ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
        }
        palm_primary, palm_lateral = _target_palm_vectors(
            armature, entries, side_suffix
        )
        palm_normal = (
            palm_primary.cross(palm_lateral).normalized()
            if palm_primary is not None
            and palm_lateral is not None
            and palm_primary.cross(palm_lateral).length > 1.0e-8
            else None
        )
        for role_base in HAND_JOINT_PAIRS:
            semantic_id = f"{role_base}_{side_suffix}"
            bone = evaluated_semantic_bone(armature, entries, semantic_id)
            if bone is None:
                continue
            direction = bone.tail - bone.head
            if direction.length < 1.0e-8:
                continue
            direction.normalize()
            rest_rotation = bone.bone.matrix_local.to_3x3()
            pose_rotation = bone.matrix.to_3x3()
            rest_direction = bone.bone.tail_local - bone.bone.head_local
            if rest_direction.length < 1.0e-8:
                continue
            finger, segment_text = role_base.rsplit("_", 1)
            segment = int(segment_text)
            limits = _FINGER_ARTICULATION_LIMITS[
                "THUMB" if finger == "THUMB" else "FINGER"
            ]
            expected_delta, _clamped = _bounded_parent_relative_direction_delta(
                rest_direction,
                direction,
                parent_deltas[finger],
                limits[segment - 1],
                bend_reference=(
                    palm_normal if finger != "THUMB" and segment > 1 else None
                ),
            )
            parent_deltas[finger] = expected_delta
            actual_delta = (
                (pose_rotation @ rest_rotation.inverted_safe())
                .to_quaternion()
                .normalized()
            )
            residual = actual_delta @ expected_delta.inverted()
            errors[semantic_id] = _signed_twist_angle(residual, direction)
    return errors


def palm_lateral_errors(
    armature,
    entries,
    frame: CanonicalFrame,
) -> dict[str, float]:
    """Return evaluated index-to-pinky palm-axis errors in radians.

    Unlike a finger endpoint check, this observes the signed palm frame and
    therefore exposes a whole-hand 180-degree roll even when every joint
    position is otherwise exact.
    """
    errors: dict[str, float] = {}
    source_basis = source_root_basis(frame)
    rig_basis = target_basis(armature, entries)
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name)
        if not hand:
            continue
        index = hand.get("INDEX_FINGER_MCP")
        pinky = hand.get("PINKY_MCP")
        if not index or not pinky or not index.valid or not pinky.valid:
            continue
        rest_primary, rest_lateral = _target_palm_vectors(
            armature, entries, side_suffix
        )
        bone = evaluated_semantic_bone(armature, entries, f"HAND_{side_suffix}")
        if (
            bone is None
            or rest_primary is None
            or rest_lateral is None
            or rest_primary.length < 1.0e-8
            or rest_lateral.length < 1.0e-8
        ):
            continue
        desired_lateral = _to_target_direction(
            Vector(index.position) - Vector(pinky.position),
            source_basis,
            rig_basis,
        )
        rest_rotation = bone.bone.matrix_local.to_3x3()
        delta = (
            (bone.matrix.to_3x3() @ rest_rotation.inverted_safe())
            .to_quaternion()
            .normalized()
        )
        observed_lateral = delta @ rest_lateral
        desired_primary = delta @ rest_primary
        observed_lateral -= (
            desired_primary
            * observed_lateral.dot(desired_primary)
            / max(desired_primary.length_squared, 1.0e-12)
        )
        desired_lateral -= (
            desired_primary
            * desired_lateral.dot(desired_primary)
            / max(desired_primary.length_squared, 1.0e-12)
        )
        if observed_lateral.length < 1.0e-8 or desired_lateral.length < 1.0e-8:
            continue
        dot = min(
            max(
                observed_lateral.normalized().dot(desired_lateral.normalized()),
                -1.0,
            ),
            1.0,
        )
        errors[side_name] = math.acos(dot)
    return errors
