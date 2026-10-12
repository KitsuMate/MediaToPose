"""Rigify bone names for body and face roles."""

from __future__ import annotations

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
