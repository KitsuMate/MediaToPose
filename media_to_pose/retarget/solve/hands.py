"""Hand solver: wrists, palms and finger hinges."""

from __future__ import annotations

import math

import bpy
from mathutils import Quaternion, Vector

from ...core.constants import HAND_JOINT_PAIRS, PROFILE_ARP
from ...rig.profiles.common import profile_identifier
from ..target_pose import TargetPose
from .common import (
    _FINGER_ARTICULATION_LIMITS,
    _bounded_parent_relative_direction_delta,
    _mapped_bone,
)
from .frames import (
    _detected_palm_basis,
    _finger_role,
    _orthonormal_palm_basis,
    _rest_head,
    _target_palm_vectors,
    _to_target_direction,
)


def _apply_hands(
    armature,
    entries,
    frame,
    source_basis,
    target_basis,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    changed = set()
    profile = profile_identifier(armature.mtp_rig_config)
    for side_name, side_suffix in (("LEFT", "L"), ("RIGHT", "R")):
        hand = frame.hands.get(side_name)
        if not hand:
            continue
        if profile == PROFILE_ARP:
            # ARP inserts a freely rotatable `c_*1_base` control above every
            # first phalanx. Reset it only when at least one valid, mapped
            # segment of that same finger will own the captured motion. A
            # visible wrist or unrelated finger must not modify the control.
            owned_fingers = set()
            for role_base, pair in HAND_JOINT_PAIRS.items():
                semantic_id = f"{role_base}_{side_suffix}"
                entry = entries.get(semantic_id)
                start, end = hand.get(pair[0]), hand.get(pair[1])
                if (
                    entry
                    and _mapped_bone(armature, entries, semantic_id)
                    and start
                    and end
                    and start.valid
                    and end.valid
                ):
                    owned_fingers.add(role_base.split("_", 1)[0].lower())
            for finger in owned_fingers:
                base = armature.pose.bones.get(
                    f"c_{finger}1_base.{side_suffix.lower()}"
                )
                if base is not None:
                    target_pose.set_basis_rotation(
                        "",
                        base,
                        Quaternion(),
                        owner="RETARGET",
                        priority=55,
                    )
                    changed.add(base)
        hand_id = f"HAND_{side_suffix}"
        hand_channel = target_pose.channel_for(hand_id)
        hand_delta = Quaternion()
        if hand_channel is not None and hand_channel.desired_rotation is not None:
            rest_hand_rotation = hand_channel.bone.bone.matrix_local.to_3x3()
            hand_delta = (
                (hand_channel.desired_rotation @ rest_hand_rotation.inverted_safe())
                .to_quaternion()
                .normalized()
            )
        palm_primary, palm_lateral = _target_palm_vectors(
            armature, entries, side_suffix
        )
        target_palm_basis = _orthonormal_palm_basis(
            palm_primary,
            palm_lateral,
        )
        detected_palm_basis = _detected_palm_basis(hand)
        palm_normal = (
            palm_primary.cross(palm_lateral).normalized()
            if palm_primary is not None
            and palm_lateral is not None
            and palm_primary.cross(palm_lateral).length > 1.0e-8
            else None
        )

        # Detailed landmarks own only articulated fingers. Every provider's
        # main hand bone remains owned by the body solver, so two independent
        # palm coordinate systems can no longer overwrite each other.
        for finger in ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"):
            parent_delta = hand_delta
            limits = _FINGER_ARTICULATION_LIMITS[
                "THUMB" if finger == "THUMB" else "FINGER"
            ]
            for segment in range(1, 4):
                role_base = f"{finger}_{segment:02d}"
                pair = HAND_JOINT_PAIRS[role_base]
                semantic_id = f"{role_base}_{side_suffix}"
                entry = entries.get(semantic_id)
                bone = _mapped_bone(armature, entries, semantic_id)
                start, end = hand.get(pair[0]), hand.get(pair[1])
                if (
                    not entry
                    or not bone
                    or not start
                    or not end
                    or not start.valid
                    or not end.valid
                ):
                    break
                raw_direction = Vector(end.position) - Vector(start.position)
                if detected_palm_basis is not None and target_palm_basis is not None:
                    # Finger observations are meaningful relative to their
                    # own palm, irrespective of which provider produced the
                    # body.  Mapping SAM/MHR hand points through the body's
                    # root frame can reflect their palm-normal component when
                    # the body wrist and detailed palm frames differ.  Keep
                    # the articulated chain provider-neutral and let the body
                    # solver's hand delta place that palm on the character.
                    palm_local = detected_palm_basis.inverted_safe() @ raw_direction
                    direction = hand_delta @ (target_palm_basis @ palm_local)
                    if direction.length > 1.0e-8:
                        direction.normalize()
                else:
                    direction = _to_target_direction(
                        raw_direction,
                        source_basis,
                        target_basis,
                    )
                if direction.length <= 1.0e-8:
                    break
                target_start = _rest_head(
                    armature, entries, _finger_role(finger, segment, side_suffix)
                )
                target_end = (
                    _rest_head(
                        armature,
                        entries,
                        _finger_role(finger, segment + 1, side_suffix),
                    )
                    if segment < 3
                    else None
                )
                target_direction = (
                    target_end - target_start
                    if target_start is not None and target_end is not None
                    else bone.bone.tail_local - bone.bone.head_local
                )
                finger_delta, clamped = _bounded_parent_relative_direction_delta(
                    target_direction,
                    direction,
                    parent_delta,
                    limits[segment - 1],
                    bend_reference=(palm_normal if finger != "THUMB" else None),
                    maximum_off_plane=math.radians(35.0 if segment == 1 else 25.0),
                    flexion_bounds=(
                        (
                            math.radians(-25.0 if segment == 1 else -15.0),
                            limits[segment - 1],
                        )
                        if side_suffix == "L"
                        else (
                            -limits[segment - 1],
                            math.radians(25.0 if segment == 1 else 15.0),
                        )
                    )
                    if finger != "THUMB"
                    else None,
                )
                rest_rotation = bone.bone.matrix_local.to_3x3()
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    finger_delta.to_matrix() @ rest_rotation,
                    correction=entry.correction_euler,
                    confidence=min(start.confidence, end.confidence),
                    influence=entry.influence,
                )
                channel = target_pose.channel_for(semantic_id)
                channel.rest_direction = target_direction.copy()
                channel.desired_direction = direction.copy()
                parent_delta = (
                    (channel.desired_rotation @ rest_rotation.inverted_safe())
                    .to_quaternion()
                    .normalized()
                )
                if clamped:
                    warning = (
                        f"{semantic_id.replace('_', ' ').title()} articulation "
                        "was limited"
                    )
                    if warning not in target_pose.warnings:
                        target_pose.warnings.append(warning)
                changed.add(bone)
    return changed
