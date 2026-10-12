"""Small helpers shared by the body, hand, face and IK solvers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mathutils import Quaternion, Vector

from ...core.constants import PROFILE_ARP, PROFILE_RIGIFY
from ...core.types import CanonicalFrame, Landmark
from ...rig.profiles.common import profile_identifier

_PROFILE_OUTPUT_BONES = {
    PROFILE_ARP: {
        "SPINE_01": "spine_01.x",
        "SPINE_02": "spine_02.x",
        "CHEST": "spine_03.x",
        "NECK": "neck.x",
        "HEAD": "head.x",
        "UPPER_ARM_L": "arm.l",
        "LOWER_ARM_L": "forearm.l",
        "HAND_L": "hand.l",
        "UPPER_ARM_R": "arm.r",
        "LOWER_ARM_R": "forearm.r",
        "HAND_R": "hand.r",
        "UPPER_LEG_L": "thigh.l",
        "LOWER_LEG_L": "leg.l",
        "FOOT_L": "foot.l",
        "TOE_L": "toes_01.l",
        "UPPER_LEG_R": "thigh.r",
        "LOWER_LEG_R": "leg.r",
        "FOOT_R": "foot.r",
        "TOE_R": "toes_01.r",
        # ARP's first visible finger control is followed by a separate
        # weighted proximal deform bone. Later phalanges are themselves
        # deforming controls, so only segment 01 needs an output override.
        "THUMB_01_L": "thumb1.l",
        "INDEX_01_L": "index1.l",
        "MIDDLE_01_L": "middle1.l",
        "RING_01_L": "ring1.l",
        "PINKY_01_L": "pinky1.l",
        "THUMB_01_R": "thumb1.r",
        "INDEX_01_R": "index1.r",
        "MIDDLE_01_R": "middle1.r",
        "RING_01_R": "ring1.r",
        "PINKY_01_R": "pinky1.r",
    },
    PROFILE_RIGIFY: {
        "UPPER_ARM_L": "ORG-upper_arm.L",
        "LOWER_ARM_L": "ORG-forearm.L",
        "HAND_L": "ORG-hand.L",
        "UPPER_ARM_R": "ORG-upper_arm.R",
        "LOWER_ARM_R": "ORG-forearm.R",
        "HAND_R": "ORG-hand.R",
        "UPPER_LEG_L": "ORG-thigh.L",
        "LOWER_LEG_L": "ORG-shin.L",
        "FOOT_L": "ORG-foot.L",
        "TOE_L": "ORG-toe.L",
        "UPPER_LEG_R": "ORG-thigh.R",
        "LOWER_LEG_R": "ORG-shin.R",
        "FOOT_R": "ORG-foot.R",
        "TOE_R": "ORG-toe.R",
    },
}


def _average(
    *landmarks: Landmark | None,
    minimum_valid: int = 1,
) -> Landmark | None:
    valid = [item for item in landmarks if item and item.valid]
    if len(valid) < minimum_valid:
        return None
    size = len(valid)
    return Landmark(
        position=tuple(
            sum(item.position[i] for item in valid) / size for i in range(3)
        ),
        image=tuple(sum(item.image[i] for item in valid) / size for i in range(2)),
        confidence=min(item.confidence for item in valid),
        valid=True,
    )


def _parent_relative_direction_delta(
    rest_direction: Vector,
    desired_direction: Vector,
    parent_delta: Quaternion,
) -> Quaternion:
    """Transport parent roll, then add only the child's local swing."""
    local_direction = parent_delta.inverted() @ desired_direction
    articulation = rest_direction.rotation_difference(local_direction)
    return (parent_delta @ articulation).normalized()


def _bounded_parent_relative_direction_delta(
    rest_direction: Vector,
    desired_direction: Vector,
    parent_delta: Quaternion,
    maximum_articulation: float,
    *,
    bend_reference: Vector | None = None,
    maximum_off_plane: float = math.radians(25.0),
    flexion_bounds: tuple[float, float] | None = None,
) -> tuple[Quaternion, bool]:
    """Transport a parent, then add a bounded child-local swing.

    Landmark positions contain a reliable segment direction but no axial-roll
    observation. Solving each segment independently lets a single bad depth
    estimate turn a child almost 180 degrees even when its parent is sound.
    Expressing the direction after removing the effective parent transform
    makes the articulation local to the chain; clamping that shortest swing
    prevents an inside-out joint without restricting whole-arm motion.
    """
    if rest_direction.length <= 1.0e-8 or desired_direction.length <= 1.0e-8:
        return parent_delta.normalized(), True
    local_direction = (parent_delta.inverted() @ desired_direction).normalized()
    constrained = False
    if bend_reference is not None and bend_reference.length > 1.0e-8:
        rest_axis = rest_direction.normalized()
        flexion = bend_reference - rest_axis * bend_reference.dot(rest_axis)
        if flexion.length > 1.0e-8:
            flexion.normalize()
            lateral = rest_axis.cross(flexion).normalized()
            lateral_component = min(
                max(local_direction.dot(lateral), -1.0),
                1.0,
            )
            off_plane = math.asin(lateral_component)
            bounded_off_plane = min(
                max(off_plane, -maximum_off_plane),
                maximum_off_plane,
            )
            flexion_angle = math.atan2(
                local_direction.dot(flexion),
                local_direction.dot(rest_axis),
            )
            cosine_off_plane = max(math.cos(bounded_off_plane), 1.0e-8)
            maximum_flexion = math.acos(
                min(
                    max(
                        math.cos(maximum_articulation) / cosine_off_plane,
                        -1.0,
                    ),
                    1.0,
                )
            )
            minimum_flexion, maximum_flexion_bound = (
                flexion_bounds
                if flexion_bounds is not None
                else (-maximum_flexion, maximum_flexion)
            )
            minimum_flexion = max(float(minimum_flexion), -maximum_flexion)
            maximum_flexion_bound = min(float(maximum_flexion_bound), maximum_flexion)
            bounded_flexion = min(
                max(flexion_angle, minimum_flexion),
                maximum_flexion_bound,
            )
            constrained = bool(
                abs(bounded_off_plane - off_plane) > 1.0e-7
                or abs(bounded_flexion - flexion_angle) > 1.0e-7
            )
            local_direction = (
                rest_axis * math.cos(bounded_off_plane) * math.cos(bounded_flexion)
                + flexion * math.cos(bounded_off_plane) * math.sin(bounded_flexion)
                + lateral * math.sin(bounded_off_plane)
            ).normalized()
            articulation = rest_axis.rotation_difference(local_direction)
            return (parent_delta @ articulation).normalized(), constrained
    articulation = rest_direction.rotation_difference(local_direction)
    articulation, constrained = _clamp_rotation(
        articulation,
        maximum_articulation,
    )
    return (parent_delta @ articulation).normalized(), constrained


_FINGER_ARTICULATION_LIMITS = {
    "THUMB": tuple(math.radians(value) for value in (105.0, 115.0, 100.0)),
    "FINGER": tuple(math.radians(value) for value in (105.0, 130.0, 105.0)),
}


def body_points(frame: CanonicalFrame) -> dict[str, Landmark]:
    points = dict(frame.body)
    # Optional provider observations stay outside the canonical body mapping.
    # MHR70's NECK is a single surface/joint position, not a complete segment;
    # keep it available for diagnostics but never infer a neck-bone direction
    # from it and the shoulder midpoint.
    neck = frame.auxiliary.get("NECK")
    if neck is not None and neck.valid:
        points["NECK"] = neck
    points["HIP_MID"] = _average(
        points.get("LEFT_HIP"),
        points.get("RIGHT_HIP"),
        minimum_valid=2,
    )
    points["SHOULDER_MID"] = _average(
        points.get("LEFT_SHOULDER"),
        points.get("RIGHT_SHOULDER"),
        minimum_valid=2,
    )
    # The head bone points toward the cranium, not toward the front of the
    # face. Including NOSE here turns its anatomical forward offset into a
    # false neck-to-head lean for providers (notably MHR70) that expose an
    # actual neck joint.
    left_head = _average(points.get("LEFT_EAR"), points.get("LEFT_EYE"))
    right_head = _average(points.get("RIGHT_EAR"), points.get("RIGHT_EYE"))
    points["HEAD_MID"] = _average(left_head, right_head, minimum_valid=2)
    points["LEFT_HAND_MID"] = _average(
        points.get("LEFT_INDEX"),
        points.get("LEFT_PINKY"),
        points.get("LEFT_THUMB"),
        minimum_valid=2,
    )
    points["RIGHT_HAND_MID"] = _average(
        points.get("RIGHT_INDEX"),
        points.get("RIGHT_PINKY"),
        points.get("RIGHT_THUMB"),
        minimum_valid=2,
    )
    return {name: item for name, item in points.items() if item is not None}


def _canonical_vector(position: Sequence[float]) -> Vector:
    # Canonical providers emit image X, downward Y, and camera-relative depth,
    # with +Z pointing away from the camera. Convert that right-handed frame to
    # image-right/camera-away/up. The detected hip frame then removes source
    # viewpoint yaw and yields anatomical right/forward/up.
    return Vector((float(position[0]), float(position[2]), -float(position[1])))


def _mapped_bone(armature, entries, semantic_id: str):
    entry = entries.get(semantic_id)
    if not entry or not entry.enabled or not entry.target_name:
        return None
    return armature.pose.bones.get(entry.target_name)


def _profile_id(config) -> str:
    return profile_identifier(config)


def evaluated_semantic_bone(armature, entries, semantic_id: str):
    """Return the evaluated skeletal bone driven by a mapped control."""
    output_name = _PROFILE_OUTPUT_BONES.get(
        _profile_id(armature.mtp_rig_config), {}
    ).get(semantic_id)
    if output_name:
        output = armature.pose.bones.get(output_name)
        if output is not None:
            return output
    return _mapped_bone(armature, entries, semantic_id)


def _clamp_rotation(
    rotation: Quaternion, maximum_angle: float
) -> tuple[Quaternion, bool]:
    """Limit a shortest-arc rotation without changing its axis."""
    rotation = rotation.normalized()
    if rotation.w < 0.0:
        rotation.negate()
    angle = rotation.angle
    if angle <= maximum_angle:
        return rotation, False
    return Quaternion().slerp(rotation, maximum_angle / angle).normalized(), True
