"""Sidebar panels, the mapping list and add-on preferences."""

from __future__ import annotations

from ..classes import register_classes, unregister_classes
from .panels.capture import MTP_PT_create_from_media
from .panels.cleanup import MTP_PT_pose_motion_cleanup
from .panels.rig_setup import MTP_PT_rig_setup, MTP_UL_mappings
from .preferences import MTP_AddonPreferences

CLASSES = (
    MTP_UL_mappings,
    MTP_PT_create_from_media,
    MTP_PT_rig_setup,
    MTP_PT_pose_motion_cleanup,
    MTP_AddonPreferences,
)


def register() -> None:
    register_classes(CLASSES)


def unregister() -> None:
    unregister_classes(CLASSES)
