"""Add-on preferences: models and storage."""

from __future__ import annotations

from bpy.types import AddonPreferences

from ...backends.anycalib import backend as anycalib_backend
from ...backends.rohm import backend as rohm_backend
from ...backends.sam3d import backend as sam3d_backend
from ...core.coordinator import COORDINATOR
from ...session import capture_options
from ..host import ADDON_PACKAGE
from .models import _draw_model_job_status, _draw_sam_install_status
from .widgets import _cache_size_label


def _draw_model_preferences(layout, settings):
    sam = layout.box()
    sam.label(text="SAM 3D Body", icon="POSE_HLT")
    sam_controls = sam.column()
    sam_controls.enabled = not COORDINATOR.is_busy()
    sam_controls.prop(settings, "sam3d_device")
    sam_backend = sam3d_backend.backend_snapshot(
        settings.sam3d_model_variant, settings.sam3d_device
    )
    sam.label(
        text=sam_backend.message, icon="CHECKMARK" if sam_backend.installed else "INFO"
    )
    _draw_sam_install_status(sam, settings)
    row = sam.row(align=True)
    row.enabled = not COORDINATOR.is_busy()
    row.operator(
        "mtp.sam3d_install",
        text="Repair / Move" if sam_backend.installed else "Install",
        icon="IMPORT",
    )
    if sam_backend.installed:
        if sam3d_backend.is_model_loaded():
            row.operator("mtp.sam3d_unload", text="Unload")
        else:
            row.operator("mtp.sam3d_load", text="Load")
        remove = sam.row()
        remove.alert = True
        remove.enabled = (
            not sam3d_backend.is_model_loaded() and not COORDINATOR.is_busy()
        )
        remove.operator("mtp.sam3d_uninstall", text="Remove", icon="TRASH")

    camera = layout.box()
    camera.label(text="Improved Lens Estimation · AnyCalib", icon="CAMERA_DATA")
    camera_controls = camera.column()
    camera_controls.enabled = not COORDINATOR.is_busy()
    camera_controls.prop(settings, "anycalib_device")
    camera_backend = anycalib_backend.backend_snapshot(settings.anycalib_device)
    camera.label(
        text=camera_backend.message,
        icon="CHECKMARK" if camera_backend.installed else "INFO",
    )
    _draw_model_job_status(
        camera,
        anycalib_backend.install_status_snapshot(),
        settings=settings,
        label="Camera model",
        cancel_operator="mtp.anycalib_cancel_install",
    )
    row = camera.row(align=True)
    row.enabled = not COORDINATOR.is_busy()
    row.operator(
        "mtp.anycalib_install",
        text="Repair / Move" if camera_backend.installed else "Install",
        icon="IMPORT",
    )
    if camera_backend.installed:
        if anycalib_backend.is_model_loaded():
            row.operator("mtp.anycalib_unload", text="Unload")
        else:
            row.operator("mtp.anycalib_load", text="Load")
        remove = camera.row()
        remove.alert = True
        remove.enabled = (
            not anycalib_backend.is_model_loaded() and not COORDINATOR.is_busy()
        )
        remove.operator("mtp.anycalib_uninstall", text="Remove", icon="TRASH")

    motion = layout.box()
    motion.label(text="RoHM Cleanup (Experimental)", icon="MODIFIER")
    motion_controls = motion.column()
    motion_controls.enabled = not COORDINATOR.is_busy()
    motion_controls.prop(settings, "rohm_device")
    motion_backend = rohm_backend.backend_snapshot(settings.rohm_device)
    motion.label(
        text=motion_backend.message,
        icon="CHECKMARK" if motion_backend.installed else "INFO",
    )
    _draw_model_job_status(
        motion,
        rohm_backend.install_status_snapshot(),
        settings=settings,
        label="AI motion model",
        cancel_operator="mtp.rohm_cancel_install",
    )
    row = motion.row(align=True)
    row.enabled = not COORDINATOR.is_busy()
    row.operator(
        "mtp.rohm_install",
        text="Repair / Move" if motion_backend.installed else "Install",
        icon="IMPORT",
    )
    if motion_backend.installed:
        if rohm_backend.is_model_loaded():
            row.operator("mtp.rohm_unload", text="Unload")
        else:
            row.operator("mtp.rohm_load", text="Load")
        remove = motion.row()
        remove.alert = True
        remove.enabled = (
            not rohm_backend.is_model_loaded() and not COORDINATOR.is_busy()
        )
        remove.operator("mtp.rohm_uninstall", text="Remove", icon="TRASH")


class MTP_AddonPreferences(AddonPreferences):
    bl_idname = ADDON_PACKAGE

    def draw(self, context):
        layout = self.layout
        scene = getattr(context, "scene", None)
        settings = getattr(scene, "mtp_settings", None) if scene else None
        layout.label(text="Models & Storage", icon="PREFERENCES")
        if settings is None:
            layout.label(text="Open a Blender scene to configure runtime models.")
            return
        _draw_model_preferences(layout, settings)
        storage = layout.box()
        storage.label(text="Stored Analysis", icon="DISK_DRIVE")
        storage.prop(settings, "cache_limit_mb", text="Limit (MiB)")
        _source, status = capture_options.ui_state_for_settings(settings)
        storage.label(text=f"Current source: {_cache_size_label(status.size_bytes)}")
        clear = storage.row()
        clear.alert = True
        clear.operator(
            "mtp.clear_detection_cache", text="Clear Stored Analysis…", icon="TRASH"
        )
