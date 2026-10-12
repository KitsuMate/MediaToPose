"""The Pose & Motion Cleanup panel, drawn from the settings schema."""

from __future__ import annotations

from bpy.types import Panel

from .... import postprocess
from ....backends.rohm import backend as rohm_backend
from ....core.coordinator import COORDINATOR
from ....postprocess import settings_schema
from ....session import capture_options
from ... import preview as cache_preview
from ..models import _draw_rohm_management
from ..widgets import (
    _configured_target,
    _draw_active_operation,
    _draw_left_disclosure,
    _draw_wrapped_labels,
    _video_cleanup_available,
)


def _schema_row_enabled(row, item, settings) -> bool:
    if not row.enabled_by:
        return True
    if row.enabled_by == "@root_motion_automatic":
        return settings.root_motion_mode == "AUTOMATIC"
    value = getattr(item, row.enabled_by)
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def _draw_cleanup_settings(layout, item, settings):
    module = settings_schema.MODULES_BY_ID.get(item.module_id)
    if module is None:
        return
    for row in module.layout:
        if row.separator:
            layout.separator()
            continue
        container = (
            layout.row(align=True) if len(row.parameters) == 2 else layout.column()
        )
        container.enabled = _schema_row_enabled(row, item, settings)
        for name in row.parameters:
            container.prop(item, name)
    if item.module_id == "CAMERA_MOTION_CORRECTION":
        layout.label(text=f"Camera track: {item.camera_track_status}", icon="INFO")
        if item.camera_tracked_frames or item.camera_rejected_frames:
            layout.label(
                text=(
                    f"Last animation: {item.camera_tracked_frames} tracked · "
                    f"{item.camera_rejected_frames} rejected"
                )
            )
        layout.operator(
            "mtp.rebuild_camera_motion_track",
            text="Rebuild Camera Analysis",
            icon="FILE_REFRESH",
        )
    elif item.module_id == "DETECTION_FILTER" and item.filter_frame_summary:
        _draw_wrapped_labels(
            layout,
            f"Last animation frames · {item.filter_frame_summary}",
            icon="INFO",
            width=52,
        )


def _draw_cleanup_module(layout, item, settings, *, available=True):
    card = layout.box()
    content = card.column()
    header = content.row(align=True)
    header.enabled = available
    header.label(text=item.display_name or item.module_id, icon="MODIFIER")
    header.prop(item, "enabled", text="")
    module = settings_schema.MODULES_BY_ID.get(item.module_id)
    content.label(text=module.summary if module else item.description, icon="INFO")
    if item.module_id == "ROHM_CORRECTION":
        backend = rohm_backend.backend_snapshot(settings.rohm_device)
        install = rohm_backend.install_status_snapshot()
        model = content.row(align=True)
        if install.running:
            model.label(
                text=f"AI motion model · {install.progress * 100.0:.1f}%",
                icon="TIME",
            )
        elif backend.installed:
            model.label(text="AI motion model · Ready", icon="CHECKMARK")
        else:
            model.enabled = not COORDINATOR.is_busy()
            model.label(text="Optional AI motion model · Not installed", icon="INFO")
            model.operator("mtp.rohm_install", text="Install", icon="IMPORT")
        if _draw_left_disclosure(
            content,
            settings,
            "show_rohm_management",
            "Motion Model Management",
        ):
            _draw_rohm_management(content, settings, backend)
    elif item.module_id == "DETECTION_FILTER":
        if item.filter_total_observations:
            manual = (
                f" · {item.filter_manual_frames} manually masked"
                if item.filter_manual_frames
                else ""
            )
            content.label(
                text=(
                    f"Last animation · {item.filter_rejected_observations} rejected · "
                    f"{item.filter_native_missing} originally missing{manual}"
                ),
                icon="INFO",
            )
            missing = item.filter_rejected_observations + item.filter_native_missing
            if missing * 2 > item.filter_total_observations:
                content.label(
                    text="Mostly reconstructed · limited source evidence",
                    icon="ERROR",
                )
    elif item.enabled and item.last_warning:
        content.label(text=item.last_warning, icon="ERROR")
    if _draw_left_disclosure(content, item, "show_advanced", "Settings"):
        body = content.column()
        body.enabled = item.enabled and available
        _draw_cleanup_settings(body, item, settings)
    if not available:
        card.label(
            text="Video only. This cleanup is not applied to images.",
            icon="INFO",
        )


class MTP_PT_pose_motion_cleanup(Panel):
    bl_idname = "MTP_PT_pose_motion_cleanup"
    bl_label = "Pose & Motion Cleanup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Media to Pose"
    bl_order = 2
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        armature = _configured_target(context)
        self.layout.label(text="", icon="MODIFIER" if armature else "ERROR")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtp_settings
        busy = COORDINATOR.is_busy()
        if busy:
            _draw_active_operation(layout, settings)
            layout.label(
                text="Cleanup settings are locked while processing", icon="LOCKED"
            )
        controls = layout.column()
        controls.enabled = not busy
        layout = controls
        armature = _configured_target(context)
        if armature is None:
            layout.label(text="Choose a target character", icon="INFO")
            return
        modules = postprocess.ordered_module_configs(armature.mtp_rig_config)
        if not modules:
            layout.label(text="Cleanup settings are not initialized", icon="INFO")
            return
        by_id = {item.module_id: item for item in modules}
        _media_source, cache_status = capture_options.ui_state_for_settings(settings)
        details = cache_preview.preview_details()
        video_cleanup_available = _video_cleanup_available(
            settings, details, cache_status
        )
        layout.label(text="Pose Cleanup", icon="POSE_HLT")
        for module_id in (
            "SHOULDER_CORRECTION",
            "ARM_AXIS_CORRECTION",
            "LEG_AXIS_CORRECTION",
            "SPINE_CORRECTION",
        ):
            if module_id in by_id:
                _draw_cleanup_module(layout, by_id[module_id], settings)
        layout.separator()
        layout.label(text="Video Cleanup (Experimental)", icon="FILE_MOVIE")
        if settings.media_path and not video_cleanup_available:
            layout.label(
                text="Image selected. Video cleanup will not run.",
                icon="INFO",
            )
        for module_id in (
            "CAMERA_MOTION_CORRECTION",
            "DETECTION_FILTER",
            "ROHM_CORRECTION",
            "TEMPORAL_CORRECTION",
        ):
            if module_id in by_id:
                _draw_cleanup_module(
                    layout,
                    by_id[module_id],
                    settings,
                    available=video_cleanup_available,
                )
        layout.label(
            text="Enabled cleanup runs with the next pose or animation.", icon="INFO"
        )
        layout.menu(
            "MTP_MT_post_processing_presets",
            text="Cleanup Options",
            icon="DOWNARROW_HLT",
        )
