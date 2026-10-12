"""Rig discovery, automatic mapping, validation, and persistence helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import bpy

from . import properties
from .constants import (
    ALL_ROLES,
    FACE_BLENDSHAPES,
    FACS_ACTION_UNITS,
    MAPPING_TABS,
    PROFILE_ARP,
    PROFILE_AUTO,
    PROFILE_GENERIC,
    PROFILE_RIGIFY,
    SCHEMA_VERSION,
)
from .core.errors import InvalidRequest
from .core.fs import atomic_json

_ARKIT_NORMALIZED = {
    normalize
    for normalize in (
        re.sub(r"[^a-z0-9]", "", name.lower())
        for name in FACE_BLENDSHAPES
        if name != "_neutral"
    )
}
_FACS_NAME_RE = re.compile(
    r"^au0*(\d{1,2})(?:(left|right|l|r))?$",
    re.IGNORECASE,
)


PREFIX_RE = re.compile(
    r"^(?:mixamorig[:_]?|valvebiped[.:_-]?|def[-_]?|org[-_]?|mch[-_]?|"
    r"bip(?:ed)?\d*[_ ]?|cc[_-]?base[_-]?|(?:armature|skeleton|skel|"
    r"bone|joint)[.:_-]+|j[_-]|c[_-]|ctrl[_-]?|control[_-]?)",
    re.IGNORECASE,
)
SHAPE_PREFIX_RE = re.compile(
    r"^(?:(?:facs|arkit|ar)[_-])?(?:(?:bs|cbs|ctrl)[_-])?",
    re.IGNORECASE,
)

SIGNED_FACE_CONTROLS = {
    "FACE_JAW_SIDE",
    "FACE_BROW_OUTER_L",
    "FACE_BROW_OUTER_R",
    "FACE_MOUTH_CORNER_L",
    "FACE_MOUTH_CORNER_R",
}

PROFILE_LABELS = {
    PROFILE_AUTO: "Automatic",
    PROFILE_ARP: "Auto-Rig Pro",
    PROFILE_RIGIFY: "Rigify",
    PROFILE_GENERIC: "Generic",
}


def profile_identifier(value) -> str:
    """Return a valid profile identifier from a config or raw value."""
    raw = (
        str(value.detected_profile)
        if hasattr(value, "detected_profile")
        else str(value or "")
    )
    return (
        raw
        if raw
        in {
            PROFILE_AUTO,
            PROFILE_ARP,
            PROFILE_RIGIFY,
            PROFILE_GENERIC,
        }
        else PROFILE_GENERIC
    )


def profile_label(value) -> str:
    return PROFILE_LABELS[profile_identifier(value)]


def normalize_name(name: str, *, shape: bool = False) -> str:
    value = name.strip()
    if shape:
        value = SHAPE_PREFIX_RE.sub("", value)
    else:
        previous = None
        while previous != value:
            previous = value
            value = PREFIX_RE.sub("", value)
    return re.sub(r"[^a-z0-9]", "", value.lower())


def side_from_role(semantic_id: str) -> str:
    if semantic_id.endswith("_L"):
        return "LEFT"
    if semantic_id.endswith("_R"):
        return "RIGHT"
    return "CENTER"


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


def detect_profile(armature: bpy.types.Object) -> str:
    data = armature.data
    if "arp_updated" in data or "rig_spine_count" in armature:
        return PROFILE_ARP
    if bool(data.get("rig_id")):
        return PROFILE_RIGIFY
    return PROFILE_GENERIC


def armature_signature(armature: bpy.types.Object) -> str:
    parts = []
    for bone in sorted(armature.data.bones, key=lambda item: item.name):
        parent = bone.parent.name if bone.parent else ""
        head = ",".join(f"{v:.5f}" for v in bone.head_local)
        tail = ",".join(f"{v:.5f}" for v in bone.tail_local)
        parts.append(f"{bone.name}|{parent}|{head}|{tail}")
    return hashlib.sha256("\n".join(parts).encode("utf8")).hexdigest()


def associated_meshes(armature: bpy.types.Object) -> list[bpy.types.Object]:
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == armature:
            meshes.append(obj)
            continue
        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == armature:
                meshes.append(obj)
                break
    return meshes


def find_shape_targets(armature: bpy.types.Object) -> dict[str, list[tuple[str, str]]]:
    targets: dict[str, list[tuple[str, str]]] = {}
    for obj in associated_meshes(armature):
        keys = obj.data.shape_keys
        if not keys:
            continue
        for block in keys.key_blocks:
            if block.name == "Basis":
                continue
            normalized = normalize_name(block.name, shape=True)
            targets.setdefault(normalized, []).append((obj.name, block.name))
    return targets


def shape_override_for_object(entry, object_name: str):
    return next(
        (
            override
            for override in entry.shape_overrides
            if override.object_name == object_name
        ),
        None,
    )


def resolved_shape_name(entry, object_name: str) -> str | None:
    """Resolve a grouped shape mapping for one mesh.

    An enabled override replaces the shared target name. A disabled override
    intentionally excludes that mesh from the group.
    """
    override = shape_override_for_object(entry, object_name)
    if override is not None:
        return override.target_name if override.enabled else None
    return entry.target_name


def matching_shape_blocks(armature: bpy.types.Object, entry):
    for obj in associated_meshes(armature):
        if entry.target_object not in {"", "*", obj.name}:
            continue
        target_name = resolved_shape_name(entry, obj.name)
        if not target_name:
            continue
        keys = obj.data.shape_keys
        if not keys:
            continue
        normalized = normalize_name(target_name, shape=True)
        for block in keys.key_blocks:
            if (
                entry.shape_scheme in {"ARKIT", "FACS"}
                and shape_scheme(block.name) != entry.shape_scheme
            ):
                continue
            if normalize_name(block.name, shape=True) == normalized:
                yield obj, block


def shape_scheme(name: str) -> str:
    lowered = name.casefold()
    if lowered.startswith(("arkit_", "arkit-", "ar_")):
        return "ARKIT"
    if lowered.startswith(("facs_", "facs-")):
        return "FACS"
    normalized = normalize_name(name, shape=True)
    if normalized in _ARKIT_NORMALIZED:
        return "ARKIT"
    if _FACS_NAME_RE.match(normalized):
        return "FACS"
    return "NONE"


def _facs_aliases(action_unit: str) -> tuple[str, ...]:
    """Return normalized-name-compatible aliases for common AU conventions."""
    match = re.match(r"AU0*(\d+)(?:_(L|R))?$", action_unit)
    if not match:
        return (action_unit,)
    number = int(match.group(1))
    side = match.group(2)
    bases = (f"AU{number:02d}", f"AU{number}")
    if not side:
        return bases
    word = "Left" if side == "L" else "Right"
    lower_word = word.lower()
    return tuple(
        dict.fromkeys(
            alias
            for base in bases
            for alias in (
                f"{base}_{side}",
                f"{base}.{side}",
                f"{base}-{side}",
                f"{base}{side}",
                f"{base}_{word}",
                f"{base}{word}",
                f"{word}_{base}",
                f"{lower_word}_{base}",
            )
        )
    )


ARP_BODY = {
    "ROOT": "c_root_master.x",
    "PELVIS": "c_root.x",
    "SPINE_01": "c_spine_01.x",
    "SPINE_02": "c_spine_02.x",
    "CHEST": "c_spine_03.x",
    "NECK": "c_neck.x",
    "HEAD": "c_head.x",
    "EYE_L": "c_eye_target.l",
    "EYE_R": "c_eye_target.r",
    "SHOULDER_L": "c_shoulder.l",
    "UPPER_ARM_L": "c_arm_fk.l",
    "LOWER_ARM_L": "c_forearm_fk.l",
    "HAND_L": "c_hand_fk.l",
    "SHOULDER_R": "c_shoulder.r",
    "UPPER_ARM_R": "c_arm_fk.r",
    "LOWER_ARM_R": "c_forearm_fk.r",
    "HAND_R": "c_hand_fk.r",
    "UPPER_LEG_L": "c_thigh_fk.l",
    "LOWER_LEG_L": "c_leg_fk.l",
    "FOOT_L": "c_foot_fk.l",
    "TOE_L": "c_toes_fk.l",
    "UPPER_LEG_R": "c_thigh_fk.r",
    "LOWER_LEG_R": "c_leg_fk.r",
    "FOOT_R": "c_foot_fk.r",
    "TOE_R": "c_toes_fk.r",
}

RIGIFY_BODY = {
    "ROOT": "root",
    "PELVIS": "hips",
    "SPINE_01": "spine_fk",
    "SPINE_02": "spine_fk.001",
    "CHEST": "chest",
    "NECK": "neck",
    "HEAD": "head",
    "EYE_L": "eye.L",
    "EYE_R": "eye.R",
    "SHOULDER_L": "shoulder.L",
    "UPPER_ARM_L": "upper_arm_fk.L",
    "LOWER_ARM_L": "forearm_fk.L",
    "HAND_L": "hand_fk.L",
    "SHOULDER_R": "shoulder.R",
    "UPPER_ARM_R": "upper_arm_fk.R",
    "LOWER_ARM_R": "forearm_fk.R",
    "HAND_R": "hand_fk.R",
    "UPPER_LEG_L": "thigh_fk.L",
    "LOWER_LEG_L": "shin_fk.L",
    "FOOT_L": "foot_fk.L",
    "TOE_L": "toe_fk.L",
    "UPPER_LEG_R": "thigh_fk.R",
    "LOWER_LEG_R": "shin_fk.R",
    "FOOT_R": "foot_fk.R",
    "TOE_R": "toe_fk.R",
}

GENERIC_ALIASES = {
    "ROOT": (
        "root",
        "master",
        "global",
        "world",
        "ground",
        "center",
        "センター",
    ),
    "PELVIS": (
        "hips",
        "pelvis",
        "hip",
        "hiproot",
        "sacrum",
        "cog",
        "root",
        "lowerbody",
        "下半身",
    ),
    "SPINE_01": (
        "spine",
        "spine01",
        "spine1",
        "spinelower",
        "lowerback",
        "lumbar",
        "waist",
        "abdomen",
        "abdomenlower",
        "torso",
        "upperbody",
        "上半身",
    ),
    "SPINE_02": (
        "spine02",
        "spine2",
        "spinemid",
        "midspine",
        "midback",
        "abdomenupper",
        "torsomid",
        "upperbody2",
        "上半身2",
    ),
    "CHEST": (
        "chest",
        "chestlower",
        "chestupper",
        "ribcage",
        "thorax",
        "spine03",
        "spine3",
        "spine04",
        "spine4",
        "spine05",
        "spine5",
        "upperchest",
        "upperback",
        "torsoupper",
        "upperbody3",
        "上半身3",
    ),
    "NECK": ("neck", "neck01", "neck1", "cervical", "首"),
    "HEAD": ("head", "skull", "cranium", "頭"),
    "EYE_L": (
        "lefteye",
        "left_eye",
        "eyeleft",
        "eyel",
        "leye",
        "lefteyeball",
        "eyeballl",
        "cc_base_l_eye",
        "def-eye.l",
        "j_Adj_l_faceeye",
        "j_bip_l_eye",
    ),
    "EYE_R": (
        "righteye",
        "right_eye",
        "eyeright",
        "eyer",
        "reye",
        "righteyeball",
        "eyeballr",
        "cc_base_r_eye",
        "def-eye.r",
        "j_adj_r_faceeye",
        "j_bip_r_eye",
    ),
    "SHOULDER_L": (
        "leftshoulder",
        "shoulderl",
        "lshoulder",
        "claviclel",
        "lclavicle",
        "lcollar",
        "leftcollar",
        "collarbonel",
        "lcollarbone",
        "scapulal",
        "左肩",
    ),
    "UPPER_ARM_L": (
        "leftarm",
        "upperarml",
        "lupperarm",
        "arml",
        "humerusl",
        "lhumerus",
        "bicepl",
        "lshldrbend",
        "左腕",
    ),
    "LOWER_ARM_L": (
        "leftforearm",
        "lowerarml",
        "llowerarm",
        "lforearm",
        "forearml",
        "leftelbow",
        "elbol",
        "elbowl",
        "radiusl",
        "lradius",
        "ulnal",
        "lulna",
        "lforearmbend",
        "左ひじ",
    ),
    "HAND_L": (
        "lefthand",
        "handl",
        "lhand",
        "leftwrist",
        "wristl",
        "lwrist",
        "leftpalm",
        "palml",
        "lpalm",
        "左手首",
    ),
    "SHOULDER_R": (
        "rightshoulder",
        "shoulderr",
        "rshoulder",
        "clavicler",
        "rclavicle",
        "rcollar",
        "rightcollar",
        "collarboner",
        "rcollarbone",
        "scapular",
        "右肩",
    ),
    "UPPER_ARM_R": (
        "rightarm",
        "upperarmr",
        "rupperarm",
        "armr",
        "humerusr",
        "rhumerus",
        "bicepr",
        "rshldrbend",
        "右腕",
    ),
    "LOWER_ARM_R": (
        "rightforearm",
        "lowerarmr",
        "rlowerarm",
        "rforearm",
        "forearmr",
        "rightelbow",
        "elbor",
        "elbowr",
        "radiusr",
        "rradius",
        "ulnar",
        "rulna",
        "rforearmbend",
        "右ひじ",
    ),
    "HAND_R": (
        "righthand",
        "handr",
        "rhand",
        "rightwrist",
        "wristr",
        "rwrist",
        "rightpalm",
        "palmr",
        "rpalm",
        "右手首",
    ),
    "UPPER_LEG_L": (
        "leftupleg",
        "thighl",
        "lthigh",
        "upperlegl",
        "lupperleg",
        "femurl",
        "lfemur",
        "leftfemur",
        "lthighbend",
        "左足",
    ),
    "LOWER_LEG_L": (
        "leftleg",
        "shinl",
        "lshin",
        "lowerlegl",
        "llowerleg",
        "calfl",
        "tibial",
        "ltibia",
        "leftknee",
        "kneel",
        "lknee",
        "左ひざ",
    ),
    "FOOT_L": (
        "leftfoot",
        "footl",
        "lfoot",
        "leftankle",
        "anklel",
        "lankle",
        "左足首",
    ),
    "TOE_L": (
        "lefttoe",
        "lefttoes",
        "toel",
        "toesl",
        "ltoe",
        "ltoes",
        "toebasel",
        "balll",
        "toeendl",
        "左つま先",
    ),
    "UPPER_LEG_R": (
        "rightupleg",
        "thighr",
        "rthigh",
        "upperlegr",
        "rupperleg",
        "femurr",
        "rfemur",
        "rightfemur",
        "rthighbend",
        "右足",
    ),
    "LOWER_LEG_R": (
        "rightleg",
        "shinr",
        "rshin",
        "lowerlegr",
        "rlowerleg",
        "calfr",
        "tibiar",
        "rtibia",
        "rightknee",
        "kneer",
        "rknee",
        "右ひざ",
    ),
    "FOOT_R": (
        "rightfoot",
        "footr",
        "rfoot",
        "rightankle",
        "ankler",
        "rankle",
        "右足首",
    ),
    "TOE_R": (
        "righttoe",
        "righttoes",
        "toer",
        "toesr",
        "rtoe",
        "rtoes",
        "toebaser",
        "ballr",
        "toeendr",
        "右つま先",
    ),
}

ARP_FACE = {
    "FACE_JAW_OPEN": ("c_jawbone.x", "ROTATION", "X"),
    "FACE_JAW_SIDE": ("c_jawbone.x", "ROTATION", "Z"),
    "FACE_BROW_INNER": ("c_eyebrow_full.l", "LOCATION", "Z"),
    "FACE_BROW_OUTER_L": ("c_eyebrow_01_end.l", "LOCATION", "Z"),
    "FACE_BROW_OUTER_R": ("c_eyebrow_01_end.r", "LOCATION", "Z"),
    "FACE_LID_L": ("c_eyelid_top.l", "LOCATION", "-Z"),
    "FACE_LID_R": ("c_eyelid_top.r", "LOCATION", "-Z"),
    "FACE_CHEEK_L": ("c_cheek_smile.l", "LOCATION", "Z"),
    "FACE_CHEEK_R": ("c_cheek_smile.r", "LOCATION", "Z"),
    "FACE_MOUTH_CORNER_L": ("c_lips_smile.l", "LOCATION", "Z"),
    "FACE_MOUTH_CORNER_R": ("c_lips_smile.r", "LOCATION", "Z"),
    "FACE_LIP_TOP": ("c_lips_top.x", "LOCATION", "Z"),
    "FACE_LIP_BOTTOM": ("c_lips_bot.x", "LOCATION", "-Z"),
}

RIGIFY_FACE = {
    "FACE_JAW_OPEN": ("jaw_master", "ROTATION", "X"),
    "FACE_JAW_SIDE": ("jaw_master", "LOCATION", "X"),
    "FACE_BROW_INNER": ("brow.T.L.001", "LOCATION", "Z"),
    "FACE_BROW_OUTER_L": ("brow.T.L.003", "LOCATION", "Z"),
    "FACE_BROW_OUTER_R": ("brow.T.R.003", "LOCATION", "Z"),
    "FACE_LID_L": ("lid.T.L", "LOCATION", "-Z"),
    "FACE_LID_R": ("lid.T.R", "LOCATION", "-Z"),
    "FACE_CHEEK_L": ("cheek.T.L.001", "LOCATION", "Z"),
    "FACE_CHEEK_R": ("cheek.T.R.001", "LOCATION", "Z"),
    "FACE_MOUTH_CORNER_L": ("lip_end.L.001", "LOCATION", "Z"),
    "FACE_MOUTH_CORNER_R": ("lip_end.R.001", "LOCATION", "Z"),
    "FACE_LIP_TOP": ("lip.T", "LOCATION", "Z"),
    "FACE_LIP_BOTTOM": ("lip.B", "LOCATION", "-Z"),
}

FK_SWITCH_FIELDS = (
    "arm_fk_switch_bone_l",
    "arm_fk_switch_property_l",
    "arm_fk_switch_value_l",
    "arm_ik_switch_value_l",
    "arm_ik_target_bone_l",
    "arm_ik_pole_bone_l",
    "arm_fk_switch_bone_r",
    "arm_fk_switch_property_r",
    "arm_fk_switch_value_r",
    "arm_ik_switch_value_r",
    "arm_ik_target_bone_r",
    "arm_ik_pole_bone_r",
    "leg_fk_switch_bone_l",
    "leg_fk_switch_property_l",
    "leg_fk_switch_value_l",
    "leg_ik_switch_value_l",
    "leg_ik_target_bone_l",
    "leg_ik_pole_bone_l",
    "leg_ik_toe_bone_l",
    "leg_fk_switch_bone_r",
    "leg_fk_switch_property_r",
    "leg_fk_switch_value_r",
    "leg_ik_switch_value_r",
    "leg_ik_target_bone_r",
    "leg_ik_pole_bone_r",
    "leg_ik_toe_bone_r",
)

PROFILE_FK_SWITCHES = {
    PROFILE_ARP: {
        "arm_fk_switch_bone_l": "c_hand_ik.l",
        "arm_fk_switch_property_l": "ik_fk_switch",
        "arm_fk_switch_value_l": 1.0,
        "arm_ik_switch_value_l": 0.0,
        "arm_ik_target_bone_l": "c_hand_ik.l",
        "arm_ik_pole_bone_l": "c_arms_pole.l",
        "arm_fk_switch_bone_r": "c_hand_ik.r",
        "arm_fk_switch_property_r": "ik_fk_switch",
        "arm_fk_switch_value_r": 1.0,
        "arm_ik_switch_value_r": 0.0,
        "arm_ik_target_bone_r": "c_hand_ik.r",
        "arm_ik_pole_bone_r": "c_arms_pole.r",
        "leg_fk_switch_bone_l": "c_foot_ik.l",
        "leg_fk_switch_property_l": "ik_fk_switch",
        "leg_fk_switch_value_l": 1.0,
        "leg_ik_switch_value_l": 0.0,
        "leg_ik_target_bone_l": "c_foot_ik.l",
        "leg_ik_pole_bone_l": "c_leg_pole.l",
        "leg_ik_toe_bone_l": "c_toes_ik.l",
        "leg_fk_switch_bone_r": "c_foot_ik.r",
        "leg_fk_switch_property_r": "ik_fk_switch",
        "leg_fk_switch_value_r": 1.0,
        "leg_ik_switch_value_r": 0.0,
        "leg_ik_target_bone_r": "c_foot_ik.r",
        "leg_ik_pole_bone_r": "c_leg_pole.r",
        "leg_ik_toe_bone_r": "c_toes_ik.r",
    },
    PROFILE_RIGIFY: {
        "arm_fk_switch_bone_l": "upper_arm_parent.L",
        "arm_fk_switch_property_l": "IK_FK",
        "arm_fk_switch_value_l": 1.0,
        "arm_ik_switch_value_l": 0.0,
        "arm_ik_target_bone_l": "hand_ik.L",
        "arm_ik_pole_bone_l": "upper_arm_ik_target.L",
        "arm_fk_switch_bone_r": "upper_arm_parent.R",
        "arm_fk_switch_property_r": "IK_FK",
        "arm_fk_switch_value_r": 1.0,
        "arm_ik_switch_value_r": 0.0,
        "arm_ik_target_bone_r": "hand_ik.R",
        "arm_ik_pole_bone_r": "upper_arm_ik_target.R",
        "leg_fk_switch_bone_l": "thigh_parent.L",
        "leg_fk_switch_property_l": "IK_FK",
        "leg_fk_switch_value_l": 1.0,
        "leg_ik_switch_value_l": 0.0,
        "leg_ik_target_bone_l": "foot_ik.L",
        "leg_ik_pole_bone_l": "thigh_ik_target.L",
        "leg_ik_toe_bone_l": "toe_ik.L",
        "leg_fk_switch_bone_r": "thigh_parent.R",
        "leg_fk_switch_property_r": "IK_FK",
        "leg_fk_switch_value_r": 1.0,
        "leg_ik_switch_value_r": 0.0,
        "leg_ik_target_bone_r": "foot_ik.R",
        "leg_ik_pole_bone_r": "thigh_ik_target.R",
        "leg_ik_toe_bone_r": "toe_ik.R",
    },
}


def configure_fk_switches(
    armature: bpy.types.Object,
    profile: str,
    *,
    force: bool = False,
) -> None:
    config = armature.mtp_rig_config
    values = PROFILE_FK_SWITCHES.get(profile)
    if not values or (config.fk_switches_manual and not force):
        return
    with properties.suppress_manual_locks():
        for field, value in values.items():
            setattr(config, field, value)
        if force:
            config.fk_switches_manual = False


def fk_switch_issues(
    armature: bpy.types.Object,
    *,
    include_arms: bool = True,
    include_legs: bool = True,
) -> list[str]:
    config = armature.mtp_rig_config
    issues = []
    for chain, side, label in (
        ("arm", "l", "Left arm"),
        ("arm", "r", "Right arm"),
        ("leg", "l", "Left leg"),
        ("leg", "r", "Right leg"),
    ):
        if (chain == "arm" and not include_arms) or (
            chain == "leg" and not include_legs
        ):
            continue
        prefix = f"{chain}_fk_switch"
        bone_name = getattr(config, f"{prefix}_bone_{side}")
        property_name = getattr(config, f"{prefix}_property_{side}")
        if not bone_name and not property_name:
            continue
        if not bone_name or bone_name not in armature.pose.bones:
            issues.append(f"{label} FK switch bone is missing")
            continue
        bone = armature.pose.bones[bone_name]
        if not property_name or property_name not in bone:
            issues.append(f"{label} ownership switch property is missing")
            continue
        value = float(bone[property_name])
        fk_value = float(getattr(config, f"{chain}_fk_switch_value_{side}"))
        ik_value = float(getattr(config, f"{chain}_ik_switch_value_{side}"))
        if abs(value - ik_value) < abs(value - fk_value):
            target = getattr(config, f"{chain}_ik_target_bone_{side}")
            pole = getattr(config, f"{chain}_ik_pole_bone_{side}")
            if not target or target not in armature.pose.bones:
                issues.append(f"{label} IK target is missing")
            if not pole or pole not in armature.pose.bones:
                issues.append(f"{label} IK pole is missing")
    return issues


def _add_entry(config, section: str, role: Sequence) -> None:
    semantic_id, label, required = role
    with properties.suppress_manual_locks():
        entry = config.mappings.add()
        entry.semantic_id = semantic_id
        entry.label = label
        entry.section = section
        entry.side = side_from_role(semantic_id)
        entry.required = bool(required)
        entry.target_kind = "SHAPE_KEY" if section == "SHAPES" else "BONE"
        if section == "SHAPES":
            entry.shape_scheme = semantic_id.split("_", 1)[0]
        elif section == "FACE_RIG" and semantic_id in SIGNED_FACE_CONTROLS:
            entry.clamp_min = -1.0
        entry.channel_mode = "VALUE" if section == "SHAPES" else "AUTO"
        entry.status = "MISSING" if required else "OPTIONAL"
        entry.source = "NONE"
        entry.manual_lock = False


def ensure_mapping_entries(armature: bpy.types.Object) -> None:
    config = armature.mtp_rig_config
    if config.subtab not in {item[0] for item in MAPPING_TABS}:
        config.subtab = "BODY"

    canonical_roles = {
        role[0]: (section, role[1], bool(role[2]))
        for section, roles in ALL_ROLES.items()
        for role in roles
    }
    valid_ids = set(canonical_roles)
    for index in reversed(range(len(config.mappings))):
        if config.mappings[index].semantic_id not in valid_ids:
            config.mappings.remove(index)
    existing = {entry.semantic_id for entry in config.mappings}
    if not existing:
        for section, roles in ALL_ROLES.items():
            for role in roles:
                _add_entry(config, section, role)
    else:
        for section, roles in ALL_ROLES.items():
            for role in roles:
                if role[0] not in existing:
                    _add_entry(config, section, role)
    # Labels and structural metadata are defined by the current role catalog,
    # not by mutable data stored in a .blend. This keeps the Mapping tab exact
    # even after entries have been edited or copied between armatures.
    with properties.suppress_manual_locks():
        for entry in config.mappings:
            section, label, required = canonical_roles[entry.semantic_id]
            entry.label = label
            entry.section = section
            entry.side = side_from_role(entry.semantic_id)
            entry.required = required
    from .post_processing import ensure_module_configs

    ensure_module_configs(config)


def mapping_by_id(config) -> dict[str, object]:
    return {entry.semantic_id: entry for entry in config.mappings}


def clear_unlocked_automatic_mappings(armature: bpy.types.Object) -> None:
    """Discard generated assignments before applying another mapping method."""
    ensure_mapping_entries(armature)
    with properties.suppress_manual_locks():
        for entry in armature.mtp_rig_config.mappings:
            if entry.manual_lock:
                continue
            entry.target_name = ""
            entry.target_object = ""
            entry.target_data_name = ""
            entry.property_name = ""
            entry.shape_overrides.clear()
            entry.source = "NONE"
            entry.confidence = 0.0
            entry.status = "MISSING" if entry.required else "OPTIONAL"
            entry.message = ""


AUTO_MATCH_CONFIDENCE_EXPLANATION = (
    "Auto-match % estimates mapping confidence from the target name, side, "
    "and bone hierarchy. It is not pose-detection accuracy; exact rig-profile "
    "matches are 100%."
)


def heuristic_mapping_message(entry) -> str:
    target_type = "shape key" if entry.target_kind == "SHAPE_KEY" else "bone"
    return (
        f'Auto-matched to {target_type} "{entry.target_name}" '
        f"({entry.confidence:.0%}). This percentage estimates the target name, "
        "side, and hierarchy match, not pose-detection accuracy."
    )


def _set_mapping(
    entry,
    target: str,
    *,
    source: str,
    confidence: float,
    target_kind: str = "BONE",
    target_object: str = "",
    channel: str = "AUTO",
    axis: str = "X",
    force: bool = False,
) -> bool:
    if entry.manual_lock and entry.source == "MANUAL" and not force:
        return False
    with properties.suppress_manual_locks():
        entry.target_kind = target_kind
        entry.target_name = target
        entry.target_object = target_object
        entry.channel_mode = channel
        entry.axis = axis
        entry.source = source
        entry.confidence = confidence
        informational = source == "HEURISTIC" and confidence < 1.0
        entry.status = "INFO" if informational else "VALID"
        entry.message = heuristic_mapping_message(entry) if informational else ""
        if force:
            entry.manual_lock = False
            if target_kind == "SHAPE_KEY":
                entry.shape_overrides.clear()
    return True


_GENERIC_PARENT_ROLES = {
    "SPINE_01": ("PELVIS", "ROOT"),
    "SPINE_02": ("SPINE_01",),
    "CHEST": ("SPINE_02", "SPINE_01"),
    "NECK": ("CHEST", "SPINE_02"),
    "HEAD": ("NECK",),
    "EYE_L": ("HEAD",),
    "EYE_R": ("HEAD",),
    "SHOULDER_L": ("CHEST", "SPINE_02"),
    "UPPER_ARM_L": ("SHOULDER_L", "CHEST"),
    "LOWER_ARM_L": ("UPPER_ARM_L",),
    "HAND_L": ("LOWER_ARM_L",),
    "SHOULDER_R": ("CHEST", "SPINE_02"),
    "UPPER_ARM_R": ("SHOULDER_R", "CHEST"),
    "LOWER_ARM_R": ("UPPER_ARM_R",),
    "HAND_R": ("LOWER_ARM_R",),
    "UPPER_LEG_L": ("PELVIS",),
    "LOWER_LEG_L": ("UPPER_LEG_L",),
    "FOOT_L": ("LOWER_LEG_L",),
    "TOE_L": ("FOOT_L",),
    "UPPER_LEG_R": ("PELVIS",),
    "LOWER_LEG_R": ("UPPER_LEG_R",),
    "FOOT_R": ("LOWER_LEG_R",),
    "TOE_R": ("FOOT_R",),
}

_GENERIC_EXCLUDED_TERMS = {
    "SHOULDER": ("upperarm", "forearm", "lowerarm", "hand"),
    "UPPER_ARM": ("forearm", "lowerarm", "elbow", "hand", "wrist"),
    "LOWER_ARM": ("upperarm", "shoulder", "clavicle", "hand", "wrist"),
    "HAND": ("finger", "thumb", "index", "middle", "ring", "pinky", "little"),
    "UPPER_LEG": ("lowerleg", "calf", "shin", "knee", "foot", "ankle"),
    "LOWER_LEG": ("upperleg", "thigh", "femur", "foot", "ankle", "toe"),
    "FOOT": ("toe", "ball"),
    "NECK": ("head", "skull"),
    "HEAD": ("neck",),
}

# Some exporter conventions reuse broad words for several anatomical roles.
# They remain useful as a last resort, but must not beat a specific bone such
# as J_Bip_C_Hips merely because the raw name happens to be exactly "Root".
_GENERIC_WEAK_ALIASES = {
    "PELVIS": {"root", "cog"},
    "UPPER_ARM": {"arm"},
    "UPPER_LEG": {"leg"},
    "FOOT": {"ankle"},
}

_GENERIC_MECHANISM_TERMS = (
    "twist",
    "roll",
    "bend",
    "stretch",
    "helper",
    "mechanism",
    "mch",
    "offset",
    "socket",
    "tweak",
)


def _generic_role_family(semantic_id: str) -> str:
    return semantic_id.removesuffix("_L").removesuffix("_R")


def _bone_name_side(name: str) -> str:
    """Recognize common word, suffix, prefix, and Japanese side markers."""
    value = name.casefold()
    if "左" in value:
        return "LEFT"
    if "右" in value:
        return "RIGHT"
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).casefold()
    tokens = {token for token in re.split(r"[^a-z0-9]+", words) if token}
    left = "left" in tokens or "l" in tokens
    right = "right" in tokens or "r" in tokens
    compact = normalize_name(name)
    anatomy = (
        "shoulder",
        "clavicle",
        "collar",
        "upperarm",
        "arm",
        "forearm",
        "lowerarm",
        "hand",
        "wrist",
        "palm",
        "thigh",
        "upperleg",
        "leg",
        "lowerleg",
        "calf",
        "shin",
        "foot",
        "ankle",
        "toe",
        "ball",
        "humerus",
        "radius",
        "ulna",
        "femur",
        "tibia",
        "eye",
        "eyeball",
    )
    if compact.startswith("left") or compact.endswith("left"):
        left = True
    if compact.startswith("right") or compact.endswith("right"):
        right = True
    if compact.startswith("l") and compact[1:].startswith(anatomy):
        left = True
    if compact.startswith("r") and compact[1:].startswith(anatomy):
        right = True
    if compact.endswith("l") and compact[:-1].endswith(anatomy):
        left = True
    if compact.endswith("r") and compact[:-1].endswith(anatomy):
        right = True
    if left == right:
        return "CENTER"
    return "LEFT" if left else "RIGHT"


def _generic_torso_mapping(armature) -> dict[str, str]:
    """Distribute the centerline spine/chest chain over torso semantics."""
    torso_bones = [
        pose_bone
        for pose_bone in armature.pose.bones
        if _bone_name_side(pose_bone.name) == "CENTER"
        and re.fullmatch(
            r"(?:(?:spine|upperbody|torso)(?:0*[1-9]\d*)?|"
            r"(?:upper)?chest)",
            normalize_name(pose_bone.name),
        )
    ]
    if not torso_bones:
        return {}

    def anatomical_rank(bone) -> int:
        name = normalize_name(bone.name)
        numbered = re.fullmatch(
            r"(?:spine|upperbody|torso)(0*[1-9]\d*)?",
            name,
        )
        if numbered:
            return int(numbered.group(1) or 0)
        return 101 if name == "upperchest" else 100

    torso_bones.sort(
        key=lambda bone: (
            len(bone.parent_recursive),
            float(bone.bone.head_local.z),
            anatomical_rank(bone),
            normalize_name(bone.name),
        )
    )
    result = {"SPINE_01": torso_bones[0].name}
    if len(torso_bones) >= 3:
        result["SPINE_02"] = torso_bones[(len(torso_bones) - 1) // 2].name
    if len(torso_bones) >= 2:
        result["CHEST"] = torso_bones[-1].name
    return result


def _best_generic_bone(
    armature,
    semantic_id: str,
    *,
    mapped: dict[str, str] | None = None,
    excluded: set | None = None,
) -> tuple[str, float]:
    raw_aliases = GENERIC_ALIASES.get(semantic_id, ())
    if not raw_aliases:
        return "", 0.0
    exact_aliases = {alias.casefold() for alias in raw_aliases}
    aliases = tuple(
        normalized
        for normalized in (normalize_name(alias) for alias in raw_aliases)
        # Non-ASCII aliases are still handled by exact_aliases. Do not let a
        # trailing digit left after ASCII normalization (e.g. 上半身2 -> "2")
        # become a global substring wildcard.
        if len(normalized) >= 3
    )
    best_name = ""
    best_score = 0.0
    expected_side = side_from_role(semantic_id)
    family = _generic_role_family(semantic_id)
    parent_names = {
        mapped[parent_role]
        for parent_role in _GENERIC_PARENT_ROLES.get(semantic_id, ())
        if mapped and mapped.get(parent_role)
    }
    for pose_bone in armature.pose.bones:
        if excluded and pose_bone.name in excluded:
            continue
        candidate_side = _bone_name_side(pose_bone.name)
        if (
            expected_side in {"LEFT", "RIGHT"}
            and candidate_side in {"LEFT", "RIGHT"}
            and candidate_side != expected_side
        ):
            continue
        if expected_side == "CENTER" and candidate_side != "CENTER":
            continue
        name = normalize_name(pose_bone.name)
        score = 0.0
        if pose_bone.name.casefold() in exact_aliases:
            score = 0.95
        elif name and name in aliases:
            score = 0.9
        elif name and any(
            name.endswith(alias) or (family != "EYE" and alias.endswith(name))
            for alias in aliases
        ):
            score = 0.75
        elif name and any(alias in name for alias in aliases):
            score = 0.55
        weak_aliases = _GENERIC_WEAK_ALIASES.get(family, set())
        if name in weak_aliases:
            score = min(score, 0.55)
        if score and any(
            term in name for term in _GENERIC_EXCLUDED_TERMS.get(family, ())
        ):
            score -= 0.35
        if parent_names:
            if pose_bone.parent and pose_bone.parent.name in parent_names:
                score += 0.08
            elif any(
                parent.name in parent_names for parent in pose_bone.parent_recursive
            ):
                score += 0.03
        if pose_bone.custom_shape is not None:
            score += 0.04
        if pose_bone.constraints and pose_bone.custom_shape is None:
            score -= 0.08
        if score >= 0.5 and score > best_score:
            best_name, best_score = pose_bone.name, min(score, 0.99)
    return best_name, best_score


def _generic_structural_body_mapping(
    armature,
    mapped: dict[str, str],
    excluded: set,
) -> dict[str, tuple[str, float]]:
    """Complete unnamed limb joints from already recognized child chains.

    Generic exporters often preserve humanoid parentage while adding an
    unknown studio prefix to the shoulder or thigh root. Name matching finds
    the distinctive child (UpperArm/ForeArm/Hand), then this pass walks back
    toward the torso. It never crosses an already mapped anatomical joint or
    selects common generated-rig mechanism bones.
    """
    suggestions: dict[str, tuple[str, float]] = {}
    chains = (
        ("SHOULDER_L", "UPPER_ARM_L", ("CHEST", "SPINE_02", "SPINE_01")),
        ("UPPER_ARM_L", "LOWER_ARM_L", ("SHOULDER_L", "CHEST")),
        ("LOWER_ARM_L", "HAND_L", ("UPPER_ARM_L",)),
        ("SHOULDER_R", "UPPER_ARM_R", ("CHEST", "SPINE_02", "SPINE_01")),
        ("UPPER_ARM_R", "LOWER_ARM_R", ("SHOULDER_R", "CHEST")),
        ("LOWER_ARM_R", "HAND_R", ("UPPER_ARM_R",)),
        ("UPPER_LEG_L", "LOWER_LEG_L", ("PELVIS",)),
        ("LOWER_LEG_L", "FOOT_L", ("UPPER_LEG_L",)),
        ("UPPER_LEG_R", "LOWER_LEG_R", ("PELVIS",)),
        ("LOWER_LEG_R", "FOOT_R", ("UPPER_LEG_R",)),
    )
    for role, child_role, stop_roles in chains:
        if mapped.get(role) or not mapped.get(child_role):
            continue
        child = armature.pose.bones.get(mapped[child_role])
        if child is None:
            continue
        stop_names = {
            mapped[stop_role] for stop_role in stop_roles if mapped.get(stop_role)
        }
        expected_side = side_from_role(role)
        for distance, parent in enumerate(child.parent_recursive, 1):
            if parent.name in stop_names:
                break
            if parent.name in excluded:
                continue
            normalized = normalize_name(parent.name)
            if any(term in normalized for term in _GENERIC_MECHANISM_TERMS):
                continue
            candidate_side = _bone_name_side(parent.name)
            if (
                expected_side in {"LEFT", "RIGHT"}
                and candidate_side in {"LEFT", "RIGHT"}
                and candidate_side != expected_side
            ):
                continue
            suggestions[role] = (
                parent.name,
                0.68 if distance == 1 else 0.58,
            )
            mapped[role] = parent.name
            excluded.add(parent.name)
            break
    return suggestions


def _finger_candidates(profile: str, finger: str, segment: int, side: str) -> list[str]:
    lower = side.lower()
    upper = side.upper()
    finger_lower = finger.lower()
    if profile == PROFILE_ARP:
        roots = {
            "THUMB": ("thumb", "thumb"),
            "INDEX": ("index", "index"),
            "MIDDLE": ("middle", "middle"),
            "RING": ("ring", "ring"),
            "PINKY": ("pinky", "pinky"),
        }
        base = roots[finger][0]
        return [
            f"c_{base}{segment}.{lower}",
            f"c_{base}_{segment:02d}.{lower}",
            f"c_{base}{segment:02d}.{lower}",
            f"c_f_{base}.{segment:02d}.{upper}",
        ]
    if profile == PROFILE_RIGIFY:
        prefix = {
            "THUMB": "thumb",
            "INDEX": "f_index",
            "MIDDLE": "f_middle",
            "RING": "f_ring",
            "PINKY": "f_pinky",
        }[finger]
        return [f"{prefix}.{segment:02d}.{upper}", f"{prefix}.{segment}.{upper}"]
    mixamo_finger = {
        "THUMB": "Thumb",
        "INDEX": "Index",
        "MIDDLE": "Middle",
        "RING": "Ring",
        "PINKY": "Pinky",
    }[finger]
    alternate_finger = "Little" if finger == "PINKY" else mixamo_finger
    unreal_finger = {
        "THUMB": "thumb",
        "INDEX": "index",
        "MIDDLE": "middle",
        "RING": "ring",
        "PINKY": "pinky",
    }[finger]
    daz_finger = {
        "THUMB": "Thumb",
        "INDEX": "Index",
        "MIDDLE": "Mid",
        "RING": "Ring",
        "PINKY": "Pinky",
    }[finger]
    mmd_finger = {
        "THUMB": "親指",
        "INDEX": "人指",
        "MIDDLE": "中指",
        "RING": "薬指",
        "PINKY": "小指",
    }[finger]
    mmd_segment = segment - 1 if finger == "THUMB" else segment
    mmd_segment_fullwidth = str(mmd_segment).translate(
        str.maketrans("0123456789", "０１２３４５６７８９")
    )
    side_word = "Left" if side == "L" else "Right"
    daz_side = "l" if side == "L" else "r"
    mmd_side = "左" if side == "L" else "右"
    return [
        f"{finger_lower}{segment}.{lower}",
        f"{finger_lower}_{segment:02d}_{lower}",
        f"{finger_lower}_{segment}_{lower}",
        f"{finger_lower}{segment}{upper}",
        f"{finger_lower}{segment}_{upper}",
        f"{finger_lower}.{segment:02d}.{upper}",
        f"{side.lower()}_{finger_lower}_{segment}",
        f"{side_word}Hand{mixamo_finger}{segment}",
        f"{side_word}{mixamo_finger}{segment}",
        f"{side_word}{alternate_finger}{segment}",
        f"{side_word}{mixamo_finger}Finger{segment}",
        f"{side.lower()}{mixamo_finger}{segment}",
        f"{side.lower()}{alternate_finger}{segment}",
        f"{mixamo_finger}{segment}{side}",
        f"{alternate_finger}{segment}{side}",
        f"{mixamo_finger}_{segment}_{side}",
        f"{alternate_finger}_{segment}_{side}",
        f"{unreal_finger}_{segment:02d}_{lower}",
        f"{daz_side}{daz_finger}{segment}",
        f"{mmd_side}{mmd_finger}{mmd_segment}",
        f"{mmd_side}{mmd_finger}{mmd_segment_fullwidth}",
    ]


def _find_first_bone(armature, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in armature.pose.bones:
            return candidate
    normalized = {normalize_name(pb.name): pb.name for pb in armature.pose.bones}
    for candidate in candidates:
        hit = normalized.get(normalize_name(candidate))
        if hit:
            return hit
    return ""


def auto_map(armature: bpy.types.Object, *, force_profile: bool = False) -> str:
    ensure_mapping_entries(armature)
    config = armature.mtp_rig_config
    detected = detect_profile(armature)
    profile = config.profile if config.profile != "AUTO" else detected
    config.detected_profile = profile
    config.armature_signature = armature_signature(armature)
    configure_fk_switches(armature, profile, force=force_profile)
    entries = mapping_by_id(config)
    bones = set(armature.pose.bones.keys())
    vrm_map = vrm_humanoid_mapping(armature) if profile == PROFILE_GENERIC else {}

    profile_body = (
        ARP_BODY
        if profile == PROFILE_ARP
        else RIGIFY_BODY
        if profile == PROFILE_RIGIFY
        else vrm_map
        if profile == PROFILE_GENERIC
        else {}
    )
    generic_torso = (
        _generic_torso_mapping(armature) if profile == PROFILE_GENERIC else {}
    )
    reserved_torso_bones = {
        bone_name: role for role, bone_name in generic_torso.items()
    }
    resolved_body = {
        entry.semantic_id: entry.target_name
        for entry in config.mappings
        if entry.section == "BODY" and entry.manual_lock and entry.target_name in bones
    }
    used_body_bones = set(resolved_body.values())
    for semantic_id, _label, _required in ALL_ROLES["BODY"]:
        entry = entries[semantic_id]
        exact = profile_body.get(semantic_id, "")
        if exact and exact in bones:
            channel = (
                "LOCATION"
                if profile == PROFILE_ARP and semantic_id in {"EYE_L", "EYE_R"}
                else "ROTATION"
                if semantic_id in {"EYE_L", "EYE_R"}
                else "AUTO"
            )
            _set_mapping(
                entry,
                exact,
                source="PROFILE",
                confidence=1.0,
                channel=channel,
                axis="X",
                force=force_profile,
            )
        elif (
            not entry.manual_lock
            and generic_torso.get(semantic_id) in bones
            and generic_torso[semantic_id] not in used_body_bones
        ):
            _set_mapping(
                entry,
                generic_torso[semantic_id],
                source="HEURISTIC",
                confidence=0.9,
                force=force_profile,
            )
        elif not entry.manual_lock:
            candidate, confidence = _best_generic_bone(
                armature,
                semantic_id,
                mapped=resolved_body,
                excluded=used_body_bones
                | {
                    bone_name
                    for bone_name, role in reserved_torso_bones.items()
                    if role != semantic_id
                },
            )
            if candidate:
                _set_mapping(
                    entry, candidate, source="HEURISTIC", confidence=confidence
                )
        if entry.target_name in bones:
            resolved_body[semantic_id] = entry.target_name
            used_body_bones.add(entry.target_name)

    if profile == PROFILE_GENERIC:
        structural = _generic_structural_body_mapping(
            armature,
            resolved_body,
            used_body_bones,
        )
        for semantic_id, (bone_name, confidence) in structural.items():
            entry = entries[semantic_id]
            if entry.manual_lock or entry.target_name:
                continue
            _set_mapping(
                entry,
                bone_name,
                source="HEURISTIC",
                confidence=confidence,
                force=force_profile,
            )

    for semantic_id, _label, _required in ALL_ROLES["HANDS"]:
        entry = entries[semantic_id]
        vrm_target = vrm_map.get(semantic_id, "")
        if vrm_target and vrm_target in bones:
            _set_mapping(
                entry,
                vrm_target,
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )
            continue
        match = re.match(r"([A-Z]+)_(\d+)_(L|R)$", semantic_id)
        if not match:
            continue
        finger, segment, side = match.group(1), int(match.group(2)), match.group(3)
        target = _find_first_bone(
            armature, _finger_candidates(profile, finger, segment, side)
        )
        if target:
            source = (
                "PROFILE" if profile in {PROFILE_ARP, PROFILE_RIGIFY} else "HEURISTIC"
            )
            confidence = 1.0 if source == "PROFILE" else 0.65
            _set_mapping(
                entry, target, source=source, confidence=confidence, force=force_profile
            )

    shape_targets = find_shape_targets(armature)
    for blendshape in FACE_BLENDSHAPES:
        if blendshape == "_neutral":
            continue
        entry = entries.get(f"ARKIT_{blendshape}")
        if not entry or (entry.manual_lock and not force_profile):
            continue
        hits = [
            hit
            for hit in shape_targets.get(normalize_name(blendshape, shape=True), [])
            if shape_scheme(hit[1]) == "ARKIT"
        ]
        if hits:
            _set_mapping(
                entry,
                hits[0][1],
                target_kind="SHAPE_KEY",
                target_object="*",
                channel="VALUE",
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )

    for action_unit, _label in FACS_ACTION_UNITS:
        entry = entries.get(f"FACS_{action_unit}")
        if not entry or (entry.manual_lock and not force_profile):
            continue
        hits = []
        for alias in _facs_aliases(action_unit):
            hits.extend(
                hit
                for hit in shape_targets.get(normalize_name(alias, shape=True), [])
                if shape_scheme(hit[1]) == "FACS"
            )
        if hits:
            _set_mapping(
                entry,
                hits[0][1],
                target_kind="SHAPE_KEY",
                target_object="*",
                channel="VALUE",
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )

    face_map = (
        ARP_FACE
        if profile == PROFILE_ARP
        else RIGIFY_FACE
        if profile == PROFILE_RIGIFY
        else {}
    )
    for semantic_id, (target, channel, axis) in face_map.items():
        entry = entries.get(semantic_id)
        if entry and target in bones:
            _set_mapping(
                entry,
                target,
                channel=channel,
                axis=axis,
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )

    if profile == PROFILE_GENERIC:
        # Generic rigs can still expose the small set of facial bone controls
        # that Media to Pose understands, including VRoid jaw conventions
        # when VRM metadata is missing or unavailable. Head and eye controls
        # are resolved as body mappings above.
        generic_face = {
            "FACE_JAW_OPEN": vrm_map.get("FACE_JAW_OPEN")
            or _find_first_bone(
                armature,
                ("J_Bip_C_Jaw", "Jaw", "jaw", "Mandible", "mandible"),
            ),
        }
        for semantic_id, target in generic_face.items():
            entry = entries.get(semantic_id)
            if not entry or not target or target not in bones:
                continue
            _set_mapping(
                entry,
                target,
                channel="ROTATION",
                axis="X",
                source="HEURISTIC",
                confidence=0.8,
                force=force_profile,
            )

    validate_mapping(armature)
    from .post_processing import validate_modules

    validate_modules(armature)
    return profile


def refresh_auto_mapping(armature: bpy.types.Object) -> bool:
    """Refresh an Automatic mapping when its rig identity or profile changed.

    VRM importers can populate humanoid metadata after the armature object is
    first visible. In that interval an Automatic mapping may legitimately be
    detected as Generic. Refreshing only stale Automatic configurations fills
    unlocked entries from the now-available metadata while preserving every
    manual lock.
    """
    if armature is None or getattr(armature, "type", None) != "ARMATURE":
        return False
    config = getattr(armature, "mtp_rig_config", None)
    if config is None:
        return False
    # Correction definitions can change without the armature geometry or
    # mapping profile changing. Synchronize them before the mapping staleness
    # checks so newly registered processors appear on existing target rigs.
    from .post_processing import ensure_module_configs

    ensure_module_configs(config)
    if config.profile != PROFILE_AUTO:
        return False
    detected = detect_profile(armature)
    signature = armature_signature(armature)
    metadata = vrm_humanoid_mapping(armature)
    entries = mapping_by_id(config)
    metadata_changed = any(
        target
        and semantic_id in entries
        and not entries[semantic_id].manual_lock
        and entries[semantic_id].target_name != target
        for semantic_id, target in metadata.items()
    )
    if (
        config.mappings
        and config.detected_profile == detected
        and config.armature_signature == signature
        and not metadata_changed
    ):
        return False
    auto_map(armature)
    return True


def validate_mapping(
    armature: bpy.types.Object,
    *,
    settings=None,
) -> tuple[int, int, int]:
    config = armature.mtp_rig_config
    ensure_mapping_entries(armature)
    if settings is None:
        scene = getattr(bpy.context, "scene", None)
        settings = getattr(scene, "mtp_settings", None)
    if settings is None:
        face_enabled = True
    else:
        from .providers import descriptor

        face_enabled = bool(
            settings.use_face and "FACE" in descriptor(settings.provider).capabilities
        )
    bone_names = set(armature.pose.bones.keys())
    associated = {obj.name: obj for obj in associated_meshes(armature)}
    used: dict[tuple[str, ...], list[object]] = {}
    errors = warnings = information = valid = 0

    section_counts = {
        section: [0, 0] for section in ("BODY", "HANDS", "SHAPES", "FACE_RIG")
    }
    scheme_counts = {"ARKIT": [0, 0], "FACS": [0, 0]}
    for entry in config.mappings:
        if entry.section in {"SHAPES", "FACE_RIG"} and not face_enabled:
            entry.status = "OPTIONAL"
            entry.message = "Facial posing is disabled"
            for override in entry.shape_overrides:
                override.status = "OPTIONAL"
                override.message = "Facial posing is disabled"
            continue
        if not entry.enabled:
            entry.status = "OPTIONAL"
            entry.message = "Disabled"
            continue
        section_counts[entry.section][1] += 1
        if entry.section == "SHAPES" and entry.shape_scheme in scheme_counts:
            scheme_counts[entry.shape_scheme][1] += 1
        exists = False
        access_error = ""
        control_warning = ""
        if entry.target_kind in {"BONE", "CUSTOM_PROPERTY"}:
            exists = bool(entry.target_name and entry.target_name in bone_names)
            if exists and (
                entry.target_kind == "CUSTOM_PROPERTY"
                or entry.channel_mode == "CUSTOM_PROPERTY"
            ):
                bone = armature.pose.bones[entry.target_name]
                if not entry.property_name or entry.property_name not in bone:
                    access_error = "Custom property no longer exists"
            if exists and not getattr(armature, "is_editable", True):
                access_error = "Armature is linked or read-only"
            if exists and entry.source != "PROFILE":
                bone = armature.pose.bones[entry.target_name]
                active_constraints = [
                    constraint
                    for constraint in bone.constraints
                    if not constraint.mute
                    and float(getattr(constraint, "influence", 1.0)) > 0.0
                ]
                if active_constraints:
                    control_warning = (
                        "Target has active constraints; test before capture"
                    )
                elif entry.channel_mode == "ROTATION" and all(bone.lock_rotation):
                    control_warning = "All rotation channels are locked"
                elif entry.channel_mode == "LOCATION" and all(bone.lock_location):
                    control_warning = "All location channels are locked"
        elif entry.target_kind == "SHAPE_KEY":
            override_error = ""
            seen_override_objects = set()
            for override in entry.shape_overrides:
                if override.object_name in seen_override_objects:
                    override.status = "ERROR"
                    override.message = "Duplicate override for this mesh"
                    override_error = override.message
                    continue
                seen_override_objects.add(override.object_name)
                obj = associated.get(override.object_name)
                if not obj:
                    override.status = "ERROR"
                    override.message = "Mesh is missing or not associated"
                    override_error = override.message
                    continue
                if entry.target_object not in {"", "*"}:
                    override.status = "ERROR"
                    override.message = "Overrides require an all-mesh group"
                    override_error = override.message
                    continue
                if not override.enabled:
                    override.status = "OPTIONAL"
                    override.message = "Intentionally skipped"
                    continue
                keys = obj.data.shape_keys
                block = (
                    next(
                        (
                            candidate
                            for candidate in keys.key_blocks
                            if normalize_name(candidate.name, shape=True)
                            == normalize_name(override.target_name, shape=True)
                            and (
                                entry.shape_scheme == "NONE"
                                or shape_scheme(candidate.name) == entry.shape_scheme
                            )
                        ),
                        None,
                    )
                    if keys and override.target_name
                    else None
                )
                if not block:
                    override.status = "ERROR"
                    override.message = "Shape key no longer exists"
                    override_error = (
                        f"{override.object_name}: shape key no longer exists"
                    )
                else:
                    if not getattr(block.id_data, "is_editable", True):
                        override.status = "ERROR"
                        override.message = "Shape-key datablock is linked or read-only"
                        override_error = (
                            f"{override.object_name}: shape-key datablock is read-only"
                        )
                    else:
                        override.status = "VALID"
                        override.message = ""
            if override_error:
                entry.status = "ERROR"
                entry.message = override_error
                errors += 1
                continue
            resolved_blocks = (
                list(matching_shape_blocks(armature, entry))
                if entry.target_name
                else []
            )
            exists = bool(resolved_blocks)
            read_only = [
                obj.name
                for obj, block in resolved_blocks
                if not getattr(block.id_data, "is_editable", True)
            ]
            if read_only:
                access_error = (
                    "Shape-key datablock is linked or read-only: "
                    + ", ".join(read_only[:3])
                )
        if not entry.target_name:
            entry.status = "MISSING" if entry.required else "OPTIONAL"
            entry.message = (
                "Required target is missing"
                if entry.required
                else "Optional target is not mapped"
            )
            errors += int(entry.required)
            continue
        if not exists:
            entry.status = "ERROR"
            entry.message = "Target no longer exists"
            errors += 1
            continue
        if access_error:
            entry.status = "ERROR"
            entry.message = access_error
            errors += 1
            continue
        # Facial rigs often use one control for multiple compatible motions,
        # such as jaw opening and sliding on separate axes. Only the exact same
        # destination channel is a collision.
        key = (
            entry.target_kind,
            f"{entry.target_object}:{entry.target_name}",
            entry.channel_mode,
            entry.property_name,
            entry.axis,
        )
        used.setdefault(key, []).append(entry)
        section_counts[entry.section][0] += 1
        if entry.section == "SHAPES" and entry.shape_scheme in scheme_counts:
            scheme_counts[entry.shape_scheme][0] += 1
        if control_warning:
            entry.status = "WARNING"
            entry.message = control_warning
            warnings += 1
        elif (
            entry.source == "HEURISTIC"
            and entry.confidence < 1.0
            and not entry.manual_lock
        ):
            entry.status = "INFO"
            entry.message = heuristic_mapping_message(entry)
            information += 1
            valid += 1
        else:
            entry.status = "VALID"
            entry.message = ""
            valid += 1

    for entries in used.values():
        if len(entries) < 2:
            continue
        # Shape keys may intentionally share a name across multiple meshes.
        if entries[0].target_kind == "SHAPE_KEY":
            continue
        for entry in entries:
            entry.status = "WARNING"
            entry.message = "Target is used by multiple semantic roles"
            warnings += 1

    # A symmetric AU target and its left/right variants should not all be
    # driven together unless the user explicitly disables one representation.
    facs_entries = {
        entry.semantic_id.removeprefix("FACS_"): entry
        for entry in config.mappings
        if entry.section == "SHAPES"
        and entry.shape_scheme == "FACS"
        and entry.enabled
        and entry.target_name
        and entry.status not in {"ERROR", "MISSING"}
    }
    for action_unit, _label in FACS_ACTION_UNITS:
        if action_unit.endswith(("_L", "_R")):
            continue
        center = facs_entries.get(action_unit)
        sides = [
            facs_entries.get(f"{action_unit}_L"),
            facs_entries.get(f"{action_unit}_R"),
        ]
        if center and any(sides):
            for entry in (center, *sides):
                if not entry:
                    continue
                entry.status = "WARNING"
                entry.message = (
                    "Symmetric and sided variants are both mapped; "
                    "disable one representation"
                )
                warnings += 1

    def completeness(section: str) -> float:
        mapped, total = section_counts[section]
        return mapped / total if total else 1.0

    config.body_complete = completeness("BODY")
    config.hands_complete = completeness("HANDS")
    config.shapes_complete = completeness("SHAPES")
    config.arkit_complete = (
        scheme_counts["ARKIT"][0] / scheme_counts["ARKIT"][1]
        if scheme_counts["ARKIT"][1]
        else 1.0
    )
    config.facs_complete = (
        scheme_counts["FACS"][0] / scheme_counts["FACS"][1]
        if scheme_counts["FACS"][1]
        else 1.0
    )
    config.face_complete = completeness("FACE_RIG")
    config.validation_summary = (
        f"{valid} valid ({information} auto-detected), "
        f"{warnings} warnings, {errors} errors"
    )
    return valid, warnings, errors


def serialize_config(armature: bpy.types.Object) -> dict:
    config = armature.mtp_rig_config
    from .post_processing import serialize_module_configs

    return {
        "schema_version": SCHEMA_VERSION,
        "profile": config.profile,
        "detected_profile": config.detected_profile,
        "armature_signature": config.armature_signature,
        "fk_switches": {field: getattr(config, field) for field in FK_SWITCH_FIELDS}
        | {"manual": config.fk_switches_manual},
        "post_processing": serialize_module_configs(config),
        "mappings": [
            {
                key: getattr(entry, key)
                for key in (
                    "semantic_id",
                    "label",
                    "section",
                    "side",
                    "required",
                    "enabled",
                    "target_kind",
                    "shape_scheme",
                    "target_name",
                    "target_object",
                    "target_data_name",
                    "property_name",
                    "channel_mode",
                    "axis",
                    "gain",
                    "offset",
                    "clamp_min",
                    "clamp_max",
                    "invert",
                    "influence",
                    "confidence",
                    "source",
                    "manual_lock",
                )
            }
            | {
                "correction_euler": list(entry.correction_euler),
                "shape_overrides": [
                    {
                        "object_name": override.object_name,
                        "target_name": override.target_name,
                        "enabled": override.enabled,
                    }
                    for override in entry.shape_overrides
                ],
            }
            for entry in config.mappings
        ],
    }


def load_config(armature: bpy.types.Object, payload: dict) -> None:
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise InvalidRequest(
            f"Unsupported mapping schema {schema_version}; expected {SCHEMA_VERSION}"
        )
    ensure_mapping_entries(armature)
    entries = mapping_by_id(armature.mtp_rig_config)
    with properties.suppress_manual_locks():
        profile = payload.get("profile")
        if profile in {
            "AUTO",
            PROFILE_ARP,
            PROFILE_RIGIFY,
            PROFILE_GENERIC,
        }:
            armature.mtp_rig_config.profile = profile
        switch_values = payload.get("fk_switches", {})
        for field in FK_SWITCH_FIELDS:
            if field in switch_values:
                setattr(
                    armature.mtp_rig_config,
                    field,
                    switch_values[field],
                )
        if "manual" in switch_values:
            armature.mtp_rig_config.fk_switches_manual = bool(switch_values["manual"])
        from .post_processing import load_module_configs

        load_module_configs(
            armature.mtp_rig_config,
            payload.get("post_processing", ()),
        )
        for item in payload.get("mappings", []):
            entry = entries.get(item.get("semantic_id", ""))
            if not entry:
                continue
            for key, value in item.items():
                if key in {"semantic_id", "shape_overrides"} or not hasattr(entry, key):
                    continue
                setattr(entry, key, value)
            if "shape_overrides" in item:
                entry.shape_overrides.clear()
                for override_data in item.get("shape_overrides", []):
                    override = entry.shape_overrides.add()
                    override.semantic_id = entry.semantic_id
                    override.object_name = str(override_data.get("object_name", ""))
                    override.target_name = str(override_data.get("target_name", ""))
                    override.enabled = bool(override_data.get("enabled", True))
    validate_mapping(armature)


def copy_section(
    source: bpy.types.Object, target: bpy.types.Object, section: str | None = None
) -> None:
    ensure_mapping_entries(source)
    ensure_mapping_entries(target)
    source_entries = mapping_by_id(source.mtp_rig_config)
    target_entries = mapping_by_id(target.mtp_rig_config)
    with properties.suppress_manual_locks():
        if section in {None, "POST_PROCESSING"}:
            from .post_processing import (
                load_module_configs,
                serialize_module_configs,
            )

            load_module_configs(
                target.mtp_rig_config,
                serialize_module_configs(source.mtp_rig_config),
            )
        if section in {None, "BODY"}:
            for field in FK_SWITCH_FIELDS:
                setattr(
                    target.mtp_rig_config,
                    field,
                    getattr(source.mtp_rig_config, field),
                )
            target.mtp_rig_config.fk_switches_manual = True
        for semantic_id, source_entry in source_entries.items():
            if section and source_entry.section != section:
                continue
            target_entry = target_entries.get(semantic_id)
            if not target_entry:
                continue
            for key in (
                "enabled",
                "target_kind",
                "target_name",
                "target_object",
                "target_data_name",
                "property_name",
                "channel_mode",
                "axis",
                "gain",
                "offset",
                "clamp_min",
                "clamp_max",
                "invert",
                "influence",
                "correction_euler",
            ):
                setattr(target_entry, key, getattr(source_entry, key))
            target_entry.shape_overrides.clear()
            for source_override in source_entry.shape_overrides:
                target_override = target_entry.shape_overrides.add()
                target_override.semantic_id = semantic_id
                target_override.object_name = source_override.object_name
                target_override.target_name = source_override.target_name
                target_override.enabled = source_override.enabled
            target_entry.source = "MANUAL"
            target_entry.manual_lock = True
            target_entry.confidence = 1.0
    validate_mapping(target)


def save_json(filepath: str, armature: bpy.types.Object) -> None:
    atomic_json(
        Path(filepath),
        serialize_config(armature),
        sort_keys=False,
        ensure_ascii=False,
    )


def load_json(filepath: str, armature: bpy.types.Object) -> None:
    with Path(filepath).open(encoding="utf8") as handle:
        load_config(armature, json.load(handle))
