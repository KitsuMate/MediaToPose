"""VRM humanoid metadata to bone names."""

from __future__ import annotations

import bpy

from ..names import normalize_name

_VRM_HUMAN_BONE_NAMES = (
    "hips",
    "spine",
    "chest",
    "upper_chest",
    "neck",
    "head",
    "left_eye",
    "right_eye",
    "jaw",
    "left_shoulder",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_shoulder",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot",
    "left_toes",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot",
    "right_toes",
    *(
        f"{side}_{finger}_{segment}"
        for side in ("left", "right")
        for finger in ("thumb", "index", "middle", "ring", "little")
        for segment in (
            ("metacarpal", "proximal", "distal")
            if finger == "thumb"
            else ("proximal", "intermediate", "distal")
        )
    ),
)


def _vrm_node_bone_name(value) -> str:
    """Read a VRM add-on node assignment without importing that add-on."""
    node = getattr(value, "node", None)
    name = getattr(node, "bone_name", "") if node is not None else ""
    return str(name or "")


def _vrm_assigned_bones(armature: bpy.types.Object) -> dict[str, str]:
    """Return normalized VRM human-bone ids mapped to Blender bone names.

    This uses duck typing so Media to Pose neither depends on nor imports the
    optional VRM add-on.
    """
    extension = getattr(armature.data, "vrm_addon_extension", None)
    if extension is None:
        return {}

    assigned: dict[str, str] = {}
    vrm1 = getattr(extension, "vrm1", None)
    humanoid = getattr(vrm1, "humanoid", None)
    human_bones = getattr(humanoid, "human_bones", None)
    if human_bones is not None:
        for attribute in _VRM_HUMAN_BONE_NAMES:
            bone_name = _vrm_node_bone_name(getattr(human_bones, attribute, None))
            if bone_name:
                assigned[normalize_name(attribute)] = bone_name
    return assigned


def vrm_humanoid_mapping(armature: bpy.types.Object) -> dict[str, str]:
    """Translate VRM humanoid metadata to Media to Pose semantic roles."""
    assigned = _vrm_assigned_bones(armature)
    if not assigned:
        return {}

    def resolve(*human_bone_ids: str) -> str:
        return next(
            (
                assigned[normalize_name(human_bone_id)]
                for human_bone_id in human_bone_ids
                if normalize_name(human_bone_id) in assigned
            ),
            "",
        )

    result = {
        "PELVIS": resolve("hips"),
        "SPINE_01": resolve("spine"),
        "NECK": resolve("neck"),
        "HEAD": resolve("head"),
        "EYE_L": resolve("leftEye"),
        "EYE_R": resolve("rightEye"),
        "SHOULDER_L": resolve("leftShoulder"),
        "UPPER_ARM_L": resolve("leftUpperArm"),
        "LOWER_ARM_L": resolve("leftLowerArm"),
        "HAND_L": resolve("leftHand"),
        "SHOULDER_R": resolve("rightShoulder"),
        "UPPER_ARM_R": resolve("rightUpperArm"),
        "LOWER_ARM_R": resolve("rightLowerArm"),
        "HAND_R": resolve("rightHand"),
        "UPPER_LEG_L": resolve("leftUpperLeg"),
        "LOWER_LEG_L": resolve("leftLowerLeg"),
        "FOOT_L": resolve("leftFoot"),
        "TOE_L": resolve("leftToes"),
        "UPPER_LEG_R": resolve("rightUpperLeg"),
        "LOWER_LEG_R": resolve("rightLowerLeg"),
        "FOOT_R": resolve("rightFoot"),
        "TOE_R": resolve("rightToes"),
        "FACE_JAW_OPEN": resolve("jaw"),
    }
    upper_chest = resolve("upperChest")
    chest = resolve("chest")
    result["SPINE_02"] = chest if upper_chest else ""
    result["CHEST"] = upper_chest or chest

    for side, suffix in (("left", "L"), ("right", "R")):
        for finger, semantic in (
            ("thumb", "THUMB"),
            ("index", "INDEX"),
            ("middle", "MIDDLE"),
            ("ring", "RING"),
            ("little", "PINKY"),
        ):
            if finger == "thumb":
                metacarpal = resolve(f"{side}ThumbMetacarpal")
                if metacarpal:
                    segments = (
                        metacarpal,
                        resolve(f"{side}ThumbProximal"),
                        resolve(f"{side}ThumbDistal"),
                    )
                else:
                    segments = ("", "", "")
            else:
                segments = tuple(
                    resolve(f"{side}{finger.title()}{segment.title()}")
                    for segment in ("proximal", "intermediate", "distal")
                )
            for index, bone_name in enumerate(segments, 1):
                result[f"{semantic}_{index:02d}_{suffix}"] = bone_name

    return {role: bone for role, bone in result.items() if bone}
