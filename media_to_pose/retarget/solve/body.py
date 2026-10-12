"""Body solver: torso, limbs and head from body landmarks."""

from __future__ import annotations

import math

import bpy
from mathutils import Quaternion, Vector

from ...core.constants import BODY_JOINT_PAIRS
from ..target_pose import TargetPose
from ..target_pose import stable_orientation_delta as _stable_orientation_delta
from .common import (
    _clamp_rotation,
    _mapped_bone,
    _parent_relative_direction_delta,
    body_points,
)
from .frames import (
    _joint_frame_target_delta,
    _source_body_reference,
    _source_head_orientation,
    _source_mhr_head_orientation,
    _target_body_direction,
    _target_body_reference,
    _to_target_direction,
)


def _apply_body(
    armature,
    entries,
    frame,
    source_basis,
    target_basis,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    changed = set()
    points = body_points(frame)
    # Root and pelvis are placement controls and remain unchanged. Segment
    # directions are solved between source and target root frames.
    for semantic_id, pair in BODY_JOINT_PAIRS.items():
        entry = entries.get(semantic_id)
        bone = _mapped_bone(armature, entries, semantic_id)
        start, end = points.get(pair[0]), points.get(pair[1])
        if semantic_id == "NECK":
            if not entry or not bone:
                continue
            # MHR_HEAD sits forward of the cervical axis even in the neutral
            # body. Aiming the neck at that absolute joint chord turns a
            # permanent anatomical offset into severe flexion. SAM already
            # supplies a neutral-relative neck frame, so use that articulated
            # observation and leave the head solve to eyes, ears, and crown.
            neck_delta = _joint_frame_target_delta(
                frame,
                "MHR_NECK",
                source_basis,
                target_basis,
            )
            if neck_delta is None:
                target_pose.set_basis_rotation(
                    semantic_id,
                    bone,
                    Quaternion(),
                    correction=entry.correction_euler,
                    owner="RETARGET",
                )
            else:
                neck_delta, clamped = _clamp_rotation(
                    neck_delta,
                    math.radians(45.0),
                )
                rest_direction = _target_body_direction(
                    armature,
                    entries,
                    semantic_id,
                )
                rest_reference = _target_body_reference(
                    armature,
                    entries,
                    semantic_id,
                )
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    neck_delta.to_matrix() @ bone.bone.matrix_local.to_3x3(),
                    correction=entry.correction_euler,
                    confidence=frame.joint_frames["MHR_NECK"].confidence,
                    influence=entry.influence,
                )
                channel = target_pose.channel_for(semantic_id)
                channel.rest_direction = rest_direction.copy()
                channel.desired_direction = neck_delta @ rest_direction
                channel.rest_reference = (
                    rest_reference.copy() if rest_reference is not None else None
                )
                channel.desired_reference = (
                    neck_delta @ rest_reference if rest_reference is not None else None
                )
                if clamped:
                    target_pose.warnings.append(
                        "SAM neck rotation was limited to 45 degrees"
                    )
            changed.add(bone)
            continue
        if (
            not entry
            or not bone
            or not start
            or not end
            or not start.valid
            or not end.valid
        ):
            continue
        raw_direction = Vector(end.position) - Vector(start.position)
        # A recovered MediaPipe hinge can have a usable image location while
        # its regressed world-space adjacent segment points in the opposite
        # direction (a common foreshortening failure on folded stylized arms).
        # When a recovered hinge starts a child segment, reconstruct that
        # segment from its image displacement. A very short projected forearm
        # relative to the projected upper arm is foreshortening, not a short
        # target bone: use the mapped chain's rest-length ratio to put the
        # missing component toward the camera. This keeps the wrist near the
        # observed elbow in projection without stretching the target limb.
        recovered_hinge = frame.auxiliary.get(f"RECOVERED_{pair[0]}") is not None
        if recovered_hinge:
            image_delta = Vector(
                (
                    float(end.image[0]) - float(start.image[0]),
                    float(end.image[1]) - float(start.image[1]),
                )
            )
            parent_role = {
                "LOWER_ARM_L": "UPPER_ARM_L",
                "LOWER_ARM_R": "UPPER_ARM_R",
                "LOWER_LEG_L": "UPPER_LEG_L",
                "LOWER_LEG_R": "UPPER_LEG_R",
            }.get(semantic_id)
            parent_pair = BODY_JOINT_PAIRS.get(parent_role, ())
            parent_start = points.get(parent_pair[0]) if parent_pair else None
            parent_bone = _mapped_bone(armature, entries, parent_role)
            parent_image_delta = (
                Vector(
                    (
                        float(start.image[0]) - float(parent_start.image[0]),
                        float(start.image[1]) - float(parent_start.image[1]),
                    )
                )
                if parent_start is not None
                else Vector()
            )
            if (
                image_delta.length > 1.0e-6
                and parent_image_delta.length > 1.0e-6
                and parent_bone is not None
                and bone.bone.length > 1.0e-8
            ):
                projected_fraction = min(
                    max(
                        image_delta.length
                        / parent_image_delta.length
                        * parent_bone.bone.length
                        / bone.bone.length,
                        0.0,
                    ),
                    1.0,
                )
                planar = image_delta.normalized() * projected_fraction
                image_direction = Vector(
                    (
                        planar.x,
                        planar.y,
                        -math.sqrt(max(1.0 - projected_fraction**2, 0.0)),
                    )
                )
                if image_direction.length > 1.0e-6:
                    raw_direction = image_direction
                    warning = (
                        f"{semantic_id.replace('_', ' ').title()} used "
                        "image-plane swing for a recovered hinge"
                    )
                    if warning not in target_pose.warnings:
                        target_pose.warnings.append(warning)
        source_reference = _source_body_reference(points, semantic_id)
        if semantic_id == "HEAD":
            head_orientation = _source_mhr_head_orientation(frame)
            if head_orientation is None:
                head_orientation = _source_head_orientation(points)
            if head_orientation is not None:
                raw_direction, source_reference = head_orientation
            else:
                target_pose.warnings.append(
                    "Head orientation was unavailable; existing pose was preserved"
                )
                continue
        direction = _to_target_direction(
            raw_direction,
            source_basis,
            target_basis,
        )
        desired_reference = (
            _to_target_direction(source_reference, source_basis, target_basis)
            if source_reference is not None
            else None
        )
        confidence = min(start.confidence, end.confidence)
        if _source_body_reference(points, semantic_id) is not None:
            reference_names = {
                "UPPER_ARM_L": ("LEFT_ELBOW", "LEFT_WRIST"),
                "LOWER_ARM_L": ("LEFT_SHOULDER", "LEFT_ELBOW"),
                "UPPER_ARM_R": ("RIGHT_ELBOW", "RIGHT_WRIST"),
                "LOWER_ARM_R": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
                "UPPER_LEG_L": ("LEFT_KNEE", "LEFT_ANKLE"),
                "LOWER_LEG_L": ("LEFT_HIP", "LEFT_KNEE"),
                "UPPER_LEG_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
                "LOWER_LEG_R": ("RIGHT_HIP", "RIGHT_KNEE"),
            }.get(semantic_id)
            if reference_names:
                reference_points = [points.get(name) for name in reference_names]
                confidence = min(
                    confidence,
                    *(
                        point.confidence
                        for point in reference_points
                        if point is not None
                    ),
                )
        # A toe has no independent lateral landmark. Its knee-to-ankle
        # reference is an unoriented bend-plane axis, and solving that axis
        # independently for both foot and toe can put the two adjacent
        # segments on opposite roll hemispheres near 90 degrees. Preserve the
        # already-solved foot roll and add only the toe's observed swing.
        # This keeps a small anatomical toe bend small instead of occasionally
        # turning it into a 180-degree axial twist.
        parent_channel = (
            target_pose.channel_for(f"FOOT_{semantic_id[-1]}")
            if semantic_id.startswith("TOE_")
            else None
        )
        if parent_channel is not None and parent_channel.desired_rotation is not None:
            parent_rest = parent_channel.bone.bone.matrix_local.to_quaternion()
            parent_delta = (
                parent_channel.desired_rotation.to_quaternion() @ parent_rest.inverted()
            ).normalized()
            rest_direction = _target_body_direction(armature, entries, semantic_id)
            toe_delta = _parent_relative_direction_delta(
                rest_direction,
                direction,
                parent_delta,
            )
            target_pose.set_desired_rotation(
                semantic_id,
                bone,
                toe_delta.to_matrix() @ bone.bone.matrix_local.to_3x3(),
                correction=entry.correction_euler,
                confidence=confidence,
                influence=entry.influence,
            )
            channel = target_pose.channel_for(semantic_id)
            channel.rest_direction = rest_direction.copy()
            channel.desired_direction = direction.copy()
            rest_reference = _target_body_reference(armature, entries, semantic_id)
            channel.rest_reference = (
                rest_reference.copy() if rest_reference is not None else None
            )
            channel.desired_reference = (
                desired_reference.copy() if desired_reference is not None else None
            )
            changed.add(bone)
            continue
        rest_direction = _target_body_direction(armature, entries, semantic_id)
        rest_reference = _target_body_reference(armature, entries, semantic_id)
        if semantic_id in {"HAND_L", "HAND_R"}:
            # Preserve the observed global palm frame. Forearm roll is assigned
            # later by Arm Axis Correction; clamping the wrist against the
            # still-untwisted forearm here made a smooth palm cross between
            # opposite 110-degree clamp branches.
            full_delta = _stable_orientation_delta(
                rest_direction,
                direction,
                rest_reference,
                desired_reference,
                reference_is_oriented=True,
            )
            if full_delta is not None:
                target_pose.set_desired_rotation(
                    semantic_id,
                    bone,
                    full_delta.to_matrix() @ bone.bone.matrix_local.to_3x3(),
                    correction=entry.correction_euler,
                    confidence=confidence,
                    influence=entry.influence,
                )
                channel = target_pose.channel_for(semantic_id)
                channel.rest_direction = rest_direction.copy()
                channel.desired_direction = direction.copy()
                channel.rest_reference = (
                    rest_reference.copy() if rest_reference is not None else None
                )
                channel.desired_reference = (
                    desired_reference.copy() if desired_reference is not None else None
                )
                changed.add(bone)
                continue
        if target_pose.aim(
            semantic_id,
            bone,
            direction,
            rest_direction=rest_direction,
            desired_reference=desired_reference,
            rest_reference=rest_reference,
            correction=entry.correction_euler,
            confidence=confidence,
            influence=entry.influence,
            reference_is_oriented=semantic_id in {"HAND_L", "HAND_R"},
        ):
            changed.add(bone)
    return changed
