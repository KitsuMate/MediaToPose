"""IK/FK switch properties per rig profile."""

from __future__ import annotations

import bpy

from ...core.constants import PROFILE_ARP, PROFILE_RIGIFY
from ..locks import suppress_manual_locks

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
    with suppress_manual_locks():
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
