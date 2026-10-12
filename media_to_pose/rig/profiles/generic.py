"""Name aliases and structural heuristics for rigs without a known profile."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ...core.constants import PROFILE_ARP, PROFILE_RIGIFY
from ..names import normalize_name, side_from_role

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
