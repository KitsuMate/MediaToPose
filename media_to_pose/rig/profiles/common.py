"""Rig profile identifiers, labels and detection."""

from __future__ import annotations

import bpy

from ...constants import (
    PROFILE_ARP,
    PROFILE_AUTO,
    PROFILE_GENERIC,
    PROFILE_RIGIFY,
)

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


def detect_profile(armature: bpy.types.Object) -> str:
    data = armature.data
    if "arp_updated" in data or "rig_spine_count" in armature:
        return PROFILE_ARP
    if bool(data.get("rig_id")):
        return PROFILE_RIGIFY
    return PROFILE_GENERIC
