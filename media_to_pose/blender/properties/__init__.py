"""RNA property groups stored on scenes and armatures."""

from __future__ import annotations

import bpy
from bpy.props import PointerProperty

from ..classes import register_classes, unregister_classes
from .rig import (
    MTPMappingEntry,
    MTPPostProcessingModuleConfig,
    MTPRigConfig,
    MTPShapeOverride,
)
from .scene import MTPSceneSettings

CLASSES = (
    MTPShapeOverride,
    MTPMappingEntry,
    MTPPostProcessingModuleConfig,
    MTPRigConfig,
    MTPSceneSettings,
)


def register() -> None:
    unregister()
    register_classes(CLASSES)
    try:
        bpy.types.Object.mtp_rig_config = PointerProperty(type=MTPRigConfig)
        bpy.types.Scene.mtp_settings = PointerProperty(type=MTPSceneSettings)
    except Exception:
        unregister()
        raise


def unregister() -> None:
    if hasattr(bpy.types.Scene, "mtp_settings"):
        del bpy.types.Scene.mtp_settings
    if hasattr(bpy.types.Object, "mtp_rig_config"):
        del bpy.types.Object.mtp_rig_config
    unregister_classes(CLASSES)
