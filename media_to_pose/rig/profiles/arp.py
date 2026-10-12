"""Auto-Rig Pro bone names for body and face roles."""

from __future__ import annotations

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


ARP_FACE = {
    # ARP opens the jaw from the controller's position (a damped track),
    # not its rotation. Location axes are in face space: Z up, X subject's
    # left.
    "FACE_JAW_OPEN": ("c_jawbone.x", "LOCATION", "-Z"),
    "FACE_JAW_SIDE": ("c_jawbone.x", "LOCATION", "-X"),
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
