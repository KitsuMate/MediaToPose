"""Canonical roles, profile names, and landmark topology."""

from __future__ import annotations

import tomllib
from functools import lru_cache

from .core import paths


@lru_cache(maxsize=1)
def extension_version() -> str:
    """Read the sole release version from Blender's extension manifest."""
    with paths.MANIFEST_PATH.open("rb") as stream:
        return str(tomllib.load(stream)["version"])


SCHEMA_VERSION = 18
CAPTURE_CONFIDENCE = 0.35
PROFILE_AUTO = "AUTO"
PROFILE_ARP = "AUTO_RIG_PRO"
PROFILE_RIGIFY = "RIGIFY"
PROFILE_GENERIC = "GENERIC"

SECTIONS = (
    ("BODY", "Body", "Body, head, eye, and limb controls"),
    ("HANDS", "Hands", "Finger and palm controls"),
    ("SHAPES", "Shapes", "Facial blendshapes"),
    ("FACE_RIG", "Face Rig", "Facial bones and custom properties"),
)

MAPPING_TABS = (
    ("BODY", "Body", "Body, head, eye, and limb controls"),
    ("HANDS", "Hands", "Finger and palm controls"),
    (
        "FACE_RIG",
        "Face",
        "Morph or facial-control mappings for the selected face output",
    ),
)

MAPPING_FILTERS = (
    ("ALL", "All", "Show every mapping"),
    ("MISSING", "Missing", "Show missing mappings"),
    ("WARNINGS", "Problems", "Show warnings and errors"),
    ("MANUAL", "Manual", "Show manual mappings"),
    ("OPTIONAL", "Optional", "Show optional mappings"),
)

BODY_ROLES = [
    ("ROOT", "Root", False),
    ("PELVIS", "Pelvis", True),
    ("SPINE_01", "Spine 1", False),
    ("SPINE_02", "Spine 2", False),
    ("CHEST", "Chest", True),
    ("NECK", "Neck", False),
    ("HEAD", "Head", True),
    ("EYE_L", "Left Eye", False),
    ("EYE_R", "Right Eye", False),
    ("SHOULDER_L", "Left Shoulder", False),
    ("UPPER_ARM_L", "Left Upper Arm", True),
    ("LOWER_ARM_L", "Left Forearm", True),
    ("HAND_L", "Left Hand", True),
    ("SHOULDER_R", "Right Shoulder", False),
    ("UPPER_ARM_R", "Right Upper Arm", True),
    ("LOWER_ARM_R", "Right Forearm", True),
    ("HAND_R", "Right Hand", True),
    ("UPPER_LEG_L", "Left Thigh", True),
    ("LOWER_LEG_L", "Left Shin", True),
    ("FOOT_L", "Left Foot", True),
    ("TOE_L", "Left Toe", False),
    ("UPPER_LEG_R", "Right Thigh", True),
    ("LOWER_LEG_R", "Right Shin", True),
    ("FOOT_R", "Right Foot", True),
    ("TOE_R", "Right Toe", False),
]

FINGERS = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
HAND_ROLES = [
    (
        f"{finger}_{segment:02d}_{side}",
        f"{side_label} {finger.title()} {segment}",
        False,
    )
    for side, side_label in (("L", "Left"), ("R", "Right"))
    for finger in FINGERS
    for segment in range(1, 4)
]

# MediaPipe Face Landmarker blendshape category names.
FACE_BLENDSHAPES = (
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)

FACS_BILATERAL_ACTION_UNITS = (
    ("AU01", "Inner Brow Raiser"),
    ("AU02", "Outer Brow Raiser"),
    ("AU04", "Brow Lowerer"),
    ("AU05", "Upper Lid Raiser"),
    ("AU06", "Cheek Raiser"),
    ("AU07", "Lid Tightener"),
    ("AU09", "Nose Wrinkler"),
    ("AU10", "Upper Lip Raiser"),
    ("AU12", "Lip Corner Puller"),
    ("AU14", "Dimpler"),
    ("AU15", "Lip Corner Depressor"),
    ("AU16", "Lower Lip Depressor"),
    ("AU20", "Lip Stretcher"),
    ("AU23", "Lip Tightener"),
    ("AU45", "Blink"),
)

FACS_CENTRAL_ACTION_UNITS = (
    ("AU17", "Chin Raiser"),
    ("AU18", "Lip Pucker"),
    ("AU22", "Lip Funneler"),
    ("AU24", "Lip Pressor"),
    ("AU25", "Lips Part"),
    ("AU26", "Jaw Drop"),
    ("AU27", "Mouth Stretch"),
    ("AU28", "Lip Suck"),
)

FACS_ACTION_UNITS = (
    FACS_CENTRAL_ACTION_UNITS
    + FACS_BILATERAL_ACTION_UNITS
    + tuple(
        (f"{action_unit}_{side}", f"{side_label} {label}")
        for action_unit, label in FACS_BILATERAL_ACTION_UNITS
        for side, side_label in (("L", "Left"), ("R", "Right"))
    )
)

SHAPE_ROLES = [
    (f"ARKIT_{name}", f"ARKit - {name}", False)
    for name in FACE_BLENDSHAPES
    if name != "_neutral"
] + [
    (f"FACS_{action_unit}", f"FACS - {label}", False)
    for action_unit, label in FACS_ACTION_UNITS
]

FACE_CONTROL_ROLES = [
    ("FACE_JAW_OPEN", "Jaw Open", False),
    ("FACE_JAW_SIDE", "Jaw Side", False),
    ("FACE_BROW_INNER", "Inner Brow", False),
    ("FACE_BROW_OUTER_L", "Left Outer Brow", False),
    ("FACE_BROW_OUTER_R", "Right Outer Brow", False),
    ("FACE_LID_L", "Left Eyelid", False),
    ("FACE_LID_R", "Right Eyelid", False),
    ("FACE_CHEEK_L", "Left Cheek", False),
    ("FACE_CHEEK_R", "Right Cheek", False),
    ("FACE_MOUTH_CORNER_L", "Left Mouth Corner", False),
    ("FACE_MOUTH_CORNER_R", "Right Mouth Corner", False),
    ("FACE_LIP_TOP", "Upper Lip", False),
    ("FACE_LIP_BOTTOM", "Lower Lip", False),
]

ALL_ROLES = {
    "BODY": BODY_ROLES,
    "HANDS": HAND_ROLES,
    "SHAPES": SHAPE_ROLES,
    "FACE_RIG": FACE_CONTROL_ROLES,
}

POSE_LANDMARK_NAMES = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)

HAND_LANDMARK_NAMES = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

# Role -> MediaPipe joint direction.
BODY_JOINT_PAIRS = {
    "SPINE_01": ("HIP_MID", "SHOULDER_MID"),
    "SPINE_02": ("HIP_MID", "SHOULDER_MID"),
    "CHEST": ("HIP_MID", "SHOULDER_MID"),
    "NECK": ("SHOULDER_MID", "HEAD_MID"),
    "HEAD": ("SHOULDER_MID", "HEAD_MID"),
    "SHOULDER_L": ("SHOULDER_MID", "LEFT_SHOULDER"),
    "UPPER_ARM_L": ("LEFT_SHOULDER", "LEFT_ELBOW"),
    "LOWER_ARM_L": ("LEFT_ELBOW", "LEFT_WRIST"),
    "HAND_L": ("LEFT_WRIST", "LEFT_HAND_MID"),
    "SHOULDER_R": ("SHOULDER_MID", "RIGHT_SHOULDER"),
    "UPPER_ARM_R": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    "LOWER_ARM_R": ("RIGHT_ELBOW", "RIGHT_WRIST"),
    "HAND_R": ("RIGHT_WRIST", "RIGHT_HAND_MID"),
    "UPPER_LEG_L": ("LEFT_HIP", "LEFT_KNEE"),
    "LOWER_LEG_L": ("LEFT_KNEE", "LEFT_ANKLE"),
    "FOOT_L": ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    "TOE_L": ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
    "UPPER_LEG_R": ("RIGHT_HIP", "RIGHT_KNEE"),
    "LOWER_LEG_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
    "FOOT_R": ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
    "TOE_R": ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
}

HAND_JOINT_PAIRS = {
    "THUMB_01": ("THUMB_CMC", "THUMB_MCP"),
    "THUMB_02": ("THUMB_MCP", "THUMB_IP"),
    "THUMB_03": ("THUMB_IP", "THUMB_TIP"),
    "INDEX_01": ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
    "INDEX_02": ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"),
    "INDEX_03": ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    "MIDDLE_01": ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
    "MIDDLE_02": ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"),
    "MIDDLE_03": ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    "RING_01": ("RING_FINGER_MCP", "RING_FINGER_PIP"),
    "RING_02": ("RING_FINGER_PIP", "RING_FINGER_DIP"),
    "RING_03": ("RING_FINGER_DIP", "RING_FINGER_TIP"),
    "PINKY_01": ("PINKY_MCP", "PINKY_PIP"),
    "PINKY_02": ("PINKY_PIP", "PINKY_DIP"),
    "PINKY_03": ("PINKY_DIP", "PINKY_TIP"),
}

JAW_BLENDSHAPES = {"jawForward", "jawLeft", "jawOpen", "jawRight"}
EYE_GAZE_BLENDSHAPES = {
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
}
