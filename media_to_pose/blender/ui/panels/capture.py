"""The Create from Media panel."""

from __future__ import annotations

from bpy.types import Panel

from ....backends.anycalib import backend as anycalib_backend
from ....backends.sam3d import backend as sam3d_backend
from ....core.coordinator import COORDINATOR
from ....providers import descriptor
from ....session import capture_options
from ....session.readiness import evaluate_apply_readiness
from ... import preview as cache_preview
from ..models import (
    _draw_anycalib_management,
    _draw_model_job_status,
    _draw_sam_management,
)
from ..widgets import (
    ISSUE_CRITICAL,
    ISSUE_WARNING,
    _configured_target,
    _draw_active_operation,
    _draw_issue,
    _draw_left_disclosure,
    _is_video_source,
    _mapping_readiness,
)


def _draw_cached_analysis_actions(layout, cache_status) -> None:
    analysis = layout.row(align=True)
    analysis.label(
        text=f"Analysis ready · {cache_status.cached_frames} frame(s)",
        icon="CHECKMARK",
    )
    refresh = analysis.operator(
        "mtp.apply_capture", text="Reanalyze", icon="FILE_REFRESH"
    )
    refresh.mode = "ANALYZE"
    refresh.force_analysis = True
    analysis.operator("mtp.clear_detection_cache", text="Delete Cache…", icon="TRASH")


def _draw_capture_status(layout, settings, readiness, armature, *, is_video):
    if _draw_active_operation(layout, settings):
        return True

    if not readiness.ready:
        _draw_issue(layout, readiness.issue, severity=ISSUE_CRITICAL)
        return False

    if settings.status_kind in {"ERROR", "WARNING"} and settings.status:
        messages = [part.strip() for part in settings.status.split(";") if part.strip()]
        success = ""
        visible_issue = messages[0]
        if settings.status_kind == "WARNING" and messages[0].casefold().startswith(
            (
                "pose applied",
                "applied pose",
                "applied cached pose",
                "video motion applied",
            )
        ):
            success = messages[0]
            visible_issue = (
                messages[1] if len(messages) > 1 else "Result may need attention"
            )
            layout.label(text=success, icon="CHECKMARK")
        _draw_issue(
            layout,
            visible_issue,
            severity=(
                ISSUE_CRITICAL if settings.status_kind == "ERROR" else ISSUE_WARNING
            ),
        )
        lowered = visible_issue.casefold()
        recovery = layout.row(align=True)
        if settings.provider == "MEDIAPIPE" and any(
            token in lowered
            for token in ("person", "pose", "body", "landmark", "confidence", "detect")
        ):
            operator = recovery.operator(
                "mtp.apply_recovery_choice", text="Try SAM 3D Body", icon="POSE_HLT"
            )
            operator.choice = "SAM"
        elif is_video and any(
            token in lowered for token in ("camera", "background", "tracking", "slide")
        ):
            operator = recovery.operator(
                "mtp.apply_recovery_choice",
                text="Create In-Place Animation",
                icon="ORIENTATION_GLOBAL",
            )
            operator.choice = "IN_PLACE"
        elif is_video and any(
            token in lowered
            for token in ("missing", "occlud", "unstable", "low-confidence", "outlier")
        ):
            recovery.label(
                text="Use Reject Bad Detections & Fill Gaps",
                icon="MODIFIER",
            )
        else:
            recovery.label(text="Review the relevant setup below", icon="INFO")
        hidden_count = len(messages) - (2 if success and len(messages) > 1 else 1)
        if hidden_count > 0 and _draw_left_disclosure(
            layout,
            settings,
            "show_result_details",
            f"{hidden_count} additional suggestion(s)",
        ):
            start = 2 if success and len(messages) > 1 else 1
            for message in messages[start:]:
                layout.label(text=message, icon="INFO")
        return False

    if settings.status and settings.status not in {"Ready", "Source media selected"}:
        layout.label(text=settings.status, icon="CHECKMARK")
    else:
        target = armature.name if armature is not None else "target character"
        layout.label(text=f"Ready for {target}", icon="CHECKMARK")
    return False


def _draw_perspective_correction(layout, settings):
    card = layout.box()
    enabled = settings.perspective_correction == "AUTO"
    header = card.row(align=True)
    header.label(text="Perspective Correction", icon="CAMERA_DATA")
    toggle = header.row(align=True)
    toggle.alignment = "RIGHT"
    toggle.enabled = not COORDINATOR.is_busy()
    toggle.prop(settings, "perspective_correction_enabled", text="")
    card.label(
        text="Improves depth, framing, and floor placement from the source lens.",
        icon="INFO",
    )

    backend = anycalib_backend.backend_snapshot(settings.anycalib_device)
    install = anycalib_backend.install_status_snapshot()
    if install.running:
        _draw_model_job_status(
            card,
            install,
            settings=settings,
            label="Camera model",
            cancel_operator="mtp.anycalib_cancel_install",
        )
    elif install.outcome == "ERROR":
        _draw_issue(
            card,
            install.error or "Camera model installation failed",
            severity=ISSUE_CRITICAL,
            boxed=False,
        )
        action_row = card.row()
        action_row.enabled = not COORDINATOR.is_busy()
        action_row.operator(
            "mtp.anycalib_install",
            text="Retry Camera Model Installation…",
            icon="FILE_REFRESH",
        )
    elif backend.installed:
        if settings.manual_fov_override and enabled:
            status = f"Manual · {settings.manual_vertical_fov:.1f}° vertical FOV"
        elif settings.last_camera_vertical_fov > 0.0 and enabled:
            status = f"Detected · {settings.last_camera_vertical_fov:.1f}° vertical FOV"
        elif enabled:
            status = "Camera calibration model · Ready"
        else:
            status = "Camera calibration model · Disabled"
        card.label(text=status, icon="CHECKMARK" if enabled else "INFO")
    else:
        card.label(
            text="Optional camera calibration model · Not installed",
            icon="INFO",
        )
        action_row = card.row()
        action_row.enabled = not COORDINATOR.is_busy()
        action_row.operator(
            "mtp.anycalib_install",
            text="Install Camera Model…",
            icon="IMPORT",
        )

    if _draw_left_disclosure(
        card,
        settings,
        "show_perspective_settings",
        "Settings",
    ):
        options = card.box()
        options.enabled = enabled and not COORDINATOR.is_busy()
        options.prop(settings, "manual_fov_override", text="Use Manual FOV")
        if settings.manual_fov_override:
            options.prop(settings, "manual_vertical_fov")
        options.prop(settings, "anycalib_device", text="Processing Device")
        options.label(
            text="Perspective correction estimates lens intrinsics, not camera tilt.",
            icon="INFO",
        )
    if _draw_left_disclosure(
        card,
        settings,
        "show_anycalib_management",
        "Camera Model Management",
    ):
        _draw_anycalib_management(
            card,
            settings,
            backend,
            show_job_status=False,
        )


def _draw_capture_provider(layout, settings, *, busy):
    capture = layout.box()
    capture.label(text="Pose Detection", icon="OUTLINER_OB_POINTCLOUD")
    provider_controls = capture.row()
    provider_controls.enabled = not busy
    provider_controls.prop(settings, "provider", expand=True)
    if settings.provider == "MEDIAPIPE":
        capture.label(text="Fast · best for photographs", icon="INFO")
        return

    capture.label(
        text="Detailed · photographs and stylized art",
        icon="INFO",
    )
    backend = sam3d_backend.backend_snapshot(
        settings.sam3d_model_variant, settings.sam3d_device
    )
    install = sam3d_backend.install_status_snapshot()
    model_row = capture.row(align=True)
    if install.running:
        model_row.label(text=install.status or "Installing…", icon="TIME")
        model_row.operator("mtp.sam3d_cancel_install", text="Cancel")
    elif backend.installed:
        model_row.label(text="SAM 3D Body · Ready", icon="CHECKMARK")
    else:
        model_row.enabled = not busy
        model_row.label(text="SAM 3D Body · Not installed", icon="INFO")
        model_row.operator("mtp.sam3d_install", text="Install", icon="IMPORT")
    if _draw_left_disclosure(
        capture,
        settings,
        "show_sam3d_management",
        "SAM Model Management",
    ):
        _draw_sam_management(capture, settings, backend)


class MTP_PT_create_from_media(Panel):
    bl_idname = "MTP_PT_create_from_media"
    bl_label = "Create from Media"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Media to Pose"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtp_settings
        armature = _configured_target(context)
        busy = COORDINATOR.is_busy()
        _draw_capture_provider(layout, settings, busy=busy)
        if busy:
            _draw_active_operation(layout, settings)

        target = layout.box()
        target.enabled = not busy
        target.label(text="Target Character", icon="ARMATURE_DATA")
        row = target.row(align=True)
        row.prop(settings, "target_armature", text="")
        row.operator("mtp.use_selected_armature", text="Use Selected")
        if armature is not None:
            state, icon, required, warnings = _mapping_readiness(
                armature.mtp_rig_config
            )
            summary = f"Rig setup: {state}"
            if required:
                summary += f" · {required} required issue(s)"
            elif warnings:
                summary += f" · {warnings} suggestion(s)"
            target.label(text=summary, icon=icon)
        else:
            _draw_issue(
                target,
                "Choose a target character or use the selected character",
                severity=ISSUE_CRITICAL,
                boxed=False,
            )
            return

        media_source, cache_status = capture_options.ui_state_for_settings(settings)
        details = cache_preview.preview_details()
        is_video = _is_video_source(settings, details, cache_status)
        source = layout.box()
        source.enabled = not busy
        source.label(
            text=(
                "Source Video"
                if is_video
                else "Source Image"
                if settings.media_path
                else "Source Media"
            ),
            icon="FILE_MOVIE" if is_video else "IMAGE_DATA",
        )
        source.prop(settings, "media_path", text="")
        if media_source.state == "MISSING":
            issue = _draw_issue(
                source,
                "Selected file no longer exists",
                severity=ISSUE_CRITICAL,
            )
            issue.operator("mtp.select_media", text="Locate Media", icon="FILE_FOLDER")
        preview = cache_preview.get_preview_icon()
        if preview is not None and preview.icon_id:
            source.template_icon(icon_value=preview.icon_id, scale=8.0)
        elif details["loading"]:
            source.label(text="Loading preview…", icon="TIME")
        elif not settings.media_path:
            source.label(text="Choose an image or video", icon="INFO")
        if is_video:
            source.prop(settings, "video_frame", text="Source Frame")
            frame_count = max(
                int(details["frame_count"]), int(cache_status.frame_count)
            )
            if frame_count:
                source.label(text=f"{frame_count} source frame(s)", icon="TIME")
        source.prop(settings, "mirror_input", text="Mirror Source")
        if not media_source.usable:
            return

        provider = descriptor(settings.provider)
        readiness = evaluate_apply_readiness(
            settings,
            armature,
            provider=provider,
            media_source=media_source,
        )
        if not busy:
            busy = _draw_capture_status(
                layout, settings, readiness, armature, is_video=is_video
            )
        if not busy:
            actions = layout.column(align=True)
            actions.scale_y = 1.35
            actions.enabled = readiness.ready
            if is_video:
                operator = actions.operator(
                    "mtp.apply_capture", text="Apply Video Motion", icon="KEY_HLT"
                )
                operator.mode = "KEYFRAMES"
                secondary = layout.row()
                secondary.enabled = readiness.ready
                operator = secondary.operator(
                    "mtp.apply_capture", text="Apply Current Frame", icon="POSE_HLT"
                )
                operator.mode = "POSE"
            else:
                operator = actions.operator(
                    "mtp.apply_capture", text="Apply Pose", icon="POSE_HLT"
                )
                operator.mode = "POSE"

        capture = layout.box()
        capture.label(text="Capture Channels", icon="POSE_HLT")
        channels = capture.row(align=True)
        channels.enabled = not busy
        for property_name, capability, label in (
            ("use_body", "BODY", "Body"),
            ("use_hands", "HANDS", "Hands"),
            ("use_face", "FACE", "Face"),
        ):
            cell = channels.row(align=True)
            cell.enabled = capability in provider.capabilities
            cell.prop(settings, property_name, text=label, toggle=True)
        if settings.use_face and "FACE" not in provider.capabilities:
            capture.label(text="Face capture requires MediaPipe", icon="INFO")

        _draw_perspective_correction(layout, settings)

        if _draw_left_disclosure(
            layout,
            settings,
            "show_detection_advanced",
            "Capture Options",
        ):
            options = layout.box()
            options.enabled = not busy
            options.prop(settings, "smoothing", text="Preview Smoothing")
            if is_video:
                options.prop(settings, "root_motion_mode")
            root_position = options.row()
            root_position.enabled = not (
                is_video and settings.root_motion_mode == "OFF"
            )
            root_position.prop(settings, "keep_original_root_position")
            if cache_status.cached_frames:
                _draw_cached_analysis_actions(options, cache_status)
