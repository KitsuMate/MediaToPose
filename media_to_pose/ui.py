"""Media to Pose sidebar panels."""

from __future__ import annotations

import textwrap

import bpy
from bpy.types import AddonPreferences, Panel, UIList

from . import cache_preview, detection_cache, mapping, postprocess
from .backends.anycalib import backend as anycalib_backend
from .backends.rohm import backend as rohm_backend
from .backends.sam3d import backend as sam3d_backend
from .capture_state import evaluate_apply_readiness
from .operation import COORDINATOR, OperationKind
from .operators_mapping import (
    active_entry,
    active_mapping_section,
)
from .postprocess import settings_schema
from .providers import descriptor
from .registration import register_classes, unregister_classes

STATUS_ICONS = {
    "VALID": "CHECKMARK",
    "INFO": "INFO",
    "WARNING": "ERROR",
    "ERROR": "CANCEL",
    "MISSING": "CANCEL",
    "OPTIONAL": "RADIOBUT_OFF",
}

ISSUE_WARNING = "WARNING"
ISSUE_CRITICAL = "CRITICAL"

VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".webm",
    ".wmv",
}


def _draw_wrapped_labels(layout, text, *, icon="NONE", width=82):
    lines = textwrap.wrap(str(text), width=width) or [""]
    for index, line in enumerate(lines):
        layout.label(
            text=line,
            icon=icon if index == 0 else "BLANK1",
        )


def _draw_issue(layout, text, *, severity, boxed=True, width=82):
    """Draw a native yellow warning or red blocking issue.

    Blender does not expose arbitrary label colors. Its themed ``ERROR``
    icon is the yellow warning indicator, while ``alert`` gives a container
    the red error treatment. Keeping this in one helper prevents ordinary
    mapping cautions from looking like capture-blocking failures.
    """
    critical = severity == ISSUE_CRITICAL
    # Never set ``alert`` on a caller-owned layout: Blender propagates that
    # state to every control subsequently drawn into the same parent.
    container = layout.box() if boxed else layout.column(align=True)
    container.alert = critical
    prefix = "Blocking issue" if critical else "Warning"
    _draw_wrapped_labels(
        container,
        f"{prefix}: {text}",
        icon="CANCEL" if critical else "ERROR",
        width=width,
    )
    return container


def _mapping_issue_severity(entry):
    if entry.status in {"ERROR", "MISSING"} and entry.required:
        return ISSUE_CRITICAL
    return ISSUE_WARNING


def _draw_sam_install_status(layout, settings):
    snapshot = sam3d_backend.install_status_snapshot()
    visible = snapshot.running or snapshot.outcome in {
        "ERROR",
        "CANCELLED",
    }
    if not visible:
        return False
    box = layout.box()
    if snapshot.outcome == "ERROR":
        box.alert = True
        box.label(text="SAM Installation Failed", icon="ERROR")
    elif snapshot.outcome == "CANCELLED":
        box.label(text="SAM Installation Cancelled", icon="CANCEL")
    else:
        box.label(text="SAM Installation", icon="TIME")
    if snapshot.phase:
        box.label(text=f"Phase: {snapshot.phase}", icon="SORTTIME")
    box.label(
        text=f"Progress: {snapshot.progress * 100.0:.1f}%",
        icon="TIME",
    )
    box.prop(settings, "progress", text="")
    if snapshot.status:
        _draw_wrapped_labels(
            box,
            snapshot.status,
            icon="ERROR" if snapshot.outcome == "ERROR" else "INFO",
        )
    if snapshot.logs:
        log_box = box.box()
        log_box.label(text="Recent Install Log", icon="TEXT")
        for entry in snapshot.logs[-6:]:
            _draw_wrapped_labels(log_box, entry, width=88)
    actions = box.row(align=True)
    if snapshot.running:
        actions.operator(
            "mtp.sam3d_cancel_install",
            text="Cancel",
            icon="CANCEL",
        )
        actions.operator(
            "mtp.sam3d_copy_install_diagnostics",
            text="Copy Diagnostics",
            icon="COPYDOWN",
        )
        if snapshot.log_path:
            actions.operator(
                "mtp.sam3d_open_install_log",
                text="Open Log",
                icon="TEXT",
            )
    else:
        actions.operator(
            "mtp.sam3d_install",
            text="Retry Installation",
            icon="FILE_REFRESH",
        )
        actions.operator(
            "mtp.sam3d_copy_install_diagnostics",
            text="Copy Diagnostics",
            icon="COPYDOWN",
        )
        if snapshot.log_path:
            actions.operator(
                "mtp.sam3d_open_install_log",
                text="Open Log",
                icon="TEXT",
            )
        box.operator(
            "mtp.sam3d_dismiss_install_result",
            text="Dismiss",
            icon="X",
        )
        box.label(
            text="Retry reuses cached downloads when available.",
            icon="INFO",
        )
    return True


def _draw_model_job_status(
    layout,
    snapshot,
    *,
    settings,
    label: str,
    cancel_operator: str,
):
    """Draw one compact, consistent download state for optional models."""
    if not snapshot.running and snapshot.outcome not in {"ERROR", "CANCELLED"}:
        return False
    box = layout.box()
    if snapshot.running:
        phase = getattr(snapshot, "phase", "") or "Download"
        box.label(text=f"{label} · {phase}", icon="TIME")
        box.label(text=f"Progress: {snapshot.progress * 100.0:.1f}%", icon="TIME")
        box.prop(settings, "progress", text="")
        if snapshot.status:
            _draw_wrapped_labels(box, snapshot.status, icon="INFO")
        operation = COORDINATOR.snapshot()
        actions = box.row()
        actions.enabled = not operation.cancel_requested
        if operation.cancel_requested:
            actions.label(text="Cancelling safely…", icon="TIME")
        else:
            actions.operator(cancel_operator, text="Cancel Download", icon="CANCEL")
        return True
    severity = ISSUE_CRITICAL if snapshot.outcome == "ERROR" else ISSUE_WARNING
    _draw_issue(
        box,
        snapshot.error or snapshot.status or f"{label} installation stopped",
        severity=severity,
        boxed=False,
    )
    return True


def _configuration_panels_available(context) -> bool:
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "mtp_settings", None)
    if not settings or settings.provider != "SAM3D_BODY":
        return True
    return sam3d_backend.sam_backend_ready(settings)


class MTP_UL_mappings(UIList):
    bl_idname = "MTP_UL_mappings"

    def draw_item(
        self,
        _context,
        layout,
        _data,
        item,
        _icon,
        _active_data,
        _active_property,
        _index=0,
        _flt_flag=0,
    ):
        row = layout.row(align=True)
        row.label(text="", icon=STATUS_ICONS.get(item.status, "QUESTION"))
        split = row.split(factor=0.48)
        split.label(text=item.label)
        destination = item.target_name or "Not Mapped"
        suffix = ""
        if item.source == "HEURISTIC":
            suffix = f"  {item.confidence:.0%}"
        elif item.source == "MANUAL":
            suffix = "  M"
        split.label(text=f"{destination}{suffix}")
        if item.manual_lock:
            row.label(text="", icon="LOCKED")

    def filter_items(self, _context, data, property_name):
        items = getattr(data, property_name)
        flags = []
        section = data.subtab
        settings = getattr(bpy.context.scene, "mtp_settings", None)
        selected_output = settings.face_output if settings else "ARKIT"
        effective_section = (
            "SHAPES"
            if section == "FACE_RIG" and selected_output in {"ARKIT", "FACS"}
            else section
        )
        text = data.filter_text.casefold().strip()
        mode = data.filter_mode
        for item in items:
            visible = item.section == effective_section
            if visible and effective_section == "SHAPES":
                visible = item.shape_scheme == selected_output
            if visible and text:
                visible = (
                    text in item.label.casefold() or text in item.target_name.casefold()
                )
            if visible and mode == "MISSING":
                visible = item.status in {"MISSING", "ERROR"}
            elif visible and mode == "WARNINGS":
                visible = item.status in {"WARNING", "ERROR", "MISSING"}
            elif visible and mode == "MANUAL":
                visible = item.source == "MANUAL"
            elif visible and mode == "OPTIONAL":
                visible = not item.required
            flags.append(self.bitflag_filter_item if visible else 0)
        return flags, []


def _cache_size_label(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return (
                f"{value:.0f} {suffix}"
                if suffix in {"B", "KiB"}
                else f"{value:.1f} {suffix}"
            )
        value /= 1024.0
    return "0 B"


def _friendly_progress_status(status: str) -> str:
    value = str(status or "Working").strip()
    lowered = value.casefold()
    replacements = (
        ("loading capture provider", "Preparing pose detection"),
        ("loading detections", "Preparing detected motion"),
        ("applying capture", "Applying result to the character"),
        ("camera", "Reducing camera movement"),
        ("rohm", "Applying optional learned cleanup"),
        ("temporal", "Smoothing motion and foot contact"),
        ("key", "Writing keyframes"),
    )
    for needle, label in replacements:
        if needle in lowered:
            return label
    return value


_CAPTURE_PROGRESS_STEP_NAMES = {
    "calibration": "Calibrating camera",
    "detection": "Detecting pose",
    "pose_application": "Applying pose",
    "loading": "Preparing detected frames",
    "camera_preflight": "Tracking camera motion",
    "solve": "Solving animation",
    "camera_apply": "Applying camera correction",
    "detection_filter": "Filtering unreliable detections",
    "rohm": "Applying Experimental RoHM cleanup",
    "temporal": "Smoothing animation",
    "commit": "Writing keyframes",
}
_CAPTURE_PROGRESS_NAME_WIDTH = max(map(len, _CAPTURE_PROGRESS_STEP_NAMES.values()))


def _step_eta_label(seconds: float) -> str:
    if seconds < 0.0:
        return "--:--"
    rounded = max(int(round(seconds)), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _operation_progress_label(operation, settings) -> str:
    if operation.frame_total and operation.step_total:
        phase = str(operation.phase or "").strip()
        name = _CAPTURE_PROGRESS_STEP_NAMES.get(
            phase.casefold(),
            _friendly_progress_status(phase or operation.status or settings.status),
        )
        frame_width = max(len(str(operation.frame_total)), 1)
        step_width = max(len(str(operation.step_total)), 1)
        return (
            f"{name:<{_CAPTURE_PROGRESS_NAME_WIDTH}}"
            f" | frame: {operation.frame_current:0{frame_width}d}/"
            f"{operation.frame_total:0{frame_width}d}"
            f" | step: {operation.step_current:0{step_width}d}/"
            f"{operation.step_total:0{step_width}d}"
            f" | ETA: {_step_eta_label(operation.step_eta_seconds)}"
        )
    return _friendly_progress_status(operation.status or settings.status)


def _is_video_source(settings, details, cache_status) -> bool:
    suffix = bpy.path.abspath(settings.media_path).lower()
    return bool(
        details["media_kind"] == "VIDEO"
        or cache_status.media_kind == "VIDEO"
        or any(suffix.endswith(extension) for extension in VIDEO_EXTENSIONS)
    )


def _video_cleanup_available(settings, details, cache_status) -> bool:
    """Keep video cleanup editable until a source is selected."""
    return not bool(settings.media_path) or _is_video_source(
        settings, details, cache_status
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


def _draw_shape_overrides(layout, entry):
    header = layout.row(align=True)
    header.prop(
        entry,
        "show_shape_overrides",
        text=f"Mesh Overrides ({len(entry.shape_overrides)})",
        icon=(
            "DISCLOSURE_TRI_DOWN"
            if entry.show_shape_overrides
            else "DISCLOSURE_TRI_RIGHT"
        ),
        emboss=False,
    )
    header.operator(
        "mtp.add_shape_override",
        text="",
        icon="ADD",
    )
    if not entry.show_shape_overrides:
        return
    if not entry.shape_overrides:
        layout.label(
            text="All meshes use the grouped target name",
            icon="INFO",
        )
        return
    for index, override in enumerate(entry.shape_overrides):
        override_box = layout.box()
        row = override_box.row(align=True)
        row.label(
            text="",
            icon=STATUS_ICONS.get(
                override.status,
                "RADIOBUT_OFF",
            ),
        )
        row.prop(
            override,
            "enabled",
            text="",
            icon=("CHECKBOX_HLT" if override.enabled else "CHECKBOX_DEHLT"),
        )
        row.prop_search(
            override,
            "object_name",
            bpy.data,
            "objects",
            text="",
        )
        select_op = row.operator(
            "mtp.select_shape_override",
            text="",
            icon="RESTRICT_SELECT_OFF",
        )
        select_op.override_index = index
        remove_op = row.operator(
            "mtp.remove_shape_override",
            text="",
            icon="X",
        )
        remove_op.override_index = index
        obj = bpy.data.objects.get(override.object_name)
        shape_keys = (
            obj.data.shape_keys
            if obj and obj.type == "MESH" and obj.data.shape_keys
            else None
        )
        target_row = override_box.row()
        target_row.enabled = override.enabled
        if shape_keys:
            target_row.prop_search(
                override,
                "target_name",
                shape_keys,
                "key_blocks",
                text="Shape Key",
            )
        else:
            target_row.prop(
                override,
                "target_name",
                text="Shape Key",
            )
        if override.message:
            if override.status == "ERROR":
                _draw_issue(
                    override_box,
                    override.message,
                    severity=ISSUE_WARNING,
                    boxed=False,
                )
            else:
                override_box.label(
                    text=override.message,
                    icon="INFO",
                )


def _draw_fk_switches(layout, armature):
    config = armature.mtp_rig_config
    box = layout.box()
    header = box.row(align=True)
    header.label(text="Limb Control Ownership", icon="CONSTRAINT_BONE")
    header.prop(
        config,
        "fk_switches_manual",
        text="",
        icon=("LOCKED" if config.fk_switches_manual else "UNLOCKED"),
    )
    for chain, side, label in (
        ("arm", "l", "Left Arm"),
        ("arm", "r", "Right Arm"),
        ("leg", "l", "Left Leg"),
        ("leg", "r", "Right Leg"),
    ):
        prefix = f"{chain}_fk_switch"
        bone_field = f"{prefix}_bone_{side}"
        property_field = f"{prefix}_property_{side}"
        value_field = f"{prefix}_value_{side}"
        row = box.row(align=True)
        row.label(text=label)
        row.prop_search(
            config,
            bone_field,
            armature.pose,
            "bones",
            text="",
        )
        row.prop(config, property_field, text="")
        row.prop(config, value_field, text="FK")
        row.prop(
            config,
            f"{chain}_ik_switch_value_{side}",
            text="IK",
        )
        ik_row = box.row(align=True)
        ik_row.label(text=f"{label} IK")
        ik_row.prop_search(
            config,
            f"{chain}_ik_target_bone_{side}",
            armature.pose,
            "bones",
            text="Target",
        )
        ik_row.prop_search(
            config,
            f"{chain}_ik_pole_bone_{side}",
            armature.pose,
            "bones",
            text="Pole",
        )
        if chain == "leg":
            toe_row = box.row(align=True)
            toe_row.label(text=f"{label} IK Toe")
            toe_row.prop_search(
                config,
                f"leg_ik_toe_bone_{side}",
                armature.pose,
                "bones",
                text="Toe",
            )
        bone_name = getattr(config, bone_field)
        property_name = getattr(config, property_field)
        bone = armature.pose.bones.get(bone_name)
        if bone_name and not bone:
            _draw_issue(
                box,
                f"{label}: switch bone is missing",
                severity=ISSUE_CRITICAL,
                boxed=False,
            )
        elif bone and property_name and property_name not in bone:
            _draw_issue(
                box,
                f"{label}: property '{property_name}' is missing",
                severity=ISSUE_CRITICAL,
                boxed=False,
            )


def _configured_target(context):
    settings = context.scene.mtp_settings
    target = settings.target_armature
    if target is not None and target.name in context.view_layer.objects:
        return target
    return None


def _mapping_problem_entries(config):
    return [
        entry
        for entry in config.mappings
        if entry.status in {"WARNING", "ERROR", "MISSING"}
    ]


def _mapping_readiness(config):
    if not config.mappings:
        return "Blocked", "CANCEL", 1, 0
    required = sum(
        entry.required and entry.status in {"ERROR", "MISSING"}
        for entry in config.mappings
    )
    warnings = sum(entry.status == "WARNING" for entry in config.mappings)
    if required:
        return "Blocked", "CANCEL", required, warnings
    if warnings:
        return "Partial", "ERROR", required, warnings
    return "Ready", "CHECKMARK", required, warnings


def _draw_left_disclosure(layout, owner, property_name: str, text: str) -> bool:
    expanded = bool(getattr(owner, property_name))
    row = layout.row(align=True)
    row.alignment = "LEFT"
    row.prop(
        owner,
        property_name,
        text=text,
        icon="DISCLOSURE_TRI_DOWN" if expanded else "DISCLOSURE_TRI_RIGHT",
        emboss=False,
    )
    return expanded


def _draw_active_operation(layout, settings):
    operation = COORDINATOR.snapshot()
    if not operation.active:
        return False
    box = layout.box()
    box.label(
        text=_operation_progress_label(operation, settings),
        icon="TIME",
    )
    box.prop(settings, "progress", text="")
    if operation.cancel_requested:
        box.label(text="Cancelling safely…", icon="TIME")
    elif operation.kind in {
        OperationKind.CAPTURE.value,
        OperationKind.CACHE_APPLY.value,
    }:
        box.operator("mtp.cancel_processing", text="Cancel", icon="CANCEL")
    elif operation.kind == OperationKind.SAM3D_INSTALL.value:
        box.operator("mtp.sam3d_cancel_install", text="Cancel Download", icon="CANCEL")
    elif operation.kind == OperationKind.ANYCALIB_INSTALL.value:
        box.operator(
            "mtp.anycalib_cancel_install", text="Cancel Download", icon="CANCEL"
        )
    elif operation.kind == OperationKind.ROHM_INSTALL.value:
        box.operator("mtp.rohm_cancel_install", text="Cancel Download", icon="CANCEL")
    return True


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


def _draw_sam_management(layout, settings, backend):
    details = layout.box()
    controls = details.column()
    controls.enabled = not COORDINATOR.is_busy()
    controls.prop(settings, "sam3d_device", text="Processing Device")
    details.label(
        text=backend.message,
        icon="CHECKMARK" if backend.installed else "INFO",
    )
    _draw_sam_install_status(details, settings)
    if not backend.installed:
        details.label(
            text="Install the model from the status row above.",
            icon="INFO",
        )
        return

    loaded = sam3d_backend.is_model_loaded()
    runtime = details.row(align=True)
    runtime.enabled = not COORDINATOR.is_busy()
    if sam3d_backend.model_load_running():
        runtime.label(text="Loading model…", icon="TIME")
    elif loaded:
        runtime.operator("mtp.sam3d_unload", text="Unload", icon="MEMORY")
    else:
        runtime.operator("mtp.sam3d_load", text="Load", icon="IMPORT")
    maintenance = details.row(align=True)
    maintenance.enabled = (
        not loaded
        and not sam3d_backend.model_load_running()
        and not COORDINATOR.is_busy()
    )
    maintenance.operator(
        "mtp.sam3d_install",
        text="Repair / Move",
        icon="FILE_REFRESH",
    )
    remove = details.row()
    remove.alert = True
    remove.enabled = maintenance.enabled
    remove.operator("mtp.sam3d_uninstall", text="Remove", icon="TRASH")
    if loaded:
        details.label(
            text="Unload before repairing, moving, or removing the model.",
            icon="INFO",
        )


def _draw_anycalib_management(layout, settings, backend, *, show_job_status=True):
    details = layout.box()
    controls = details.column()
    controls.enabled = not COORDINATOR.is_busy()
    controls.prop(settings, "anycalib_device", text="Processing Device")
    details.label(
        text=backend.message,
        icon="CHECKMARK" if backend.installed else "INFO",
    )
    if show_job_status:
        _draw_model_job_status(
            details,
            anycalib_backend.install_status_snapshot(),
            settings=settings,
            label="Camera model",
            cancel_operator="mtp.anycalib_cancel_install",
        )
    if not backend.installed:
        return
    loaded = anycalib_backend.is_model_loaded()
    runtime = details.row(align=True)
    runtime.enabled = not COORDINATOR.is_busy()
    runtime.operator(
        "mtp.anycalib_unload" if loaded else "mtp.anycalib_load",
        text="Unload" if loaded else "Load",
        icon="MEMORY" if loaded else "IMPORT",
    )
    maintenance = details.row(align=True)
    maintenance.enabled = not loaded and not COORDINATOR.is_busy()
    maintenance.operator(
        "mtp.anycalib_install", text="Repair / Move", icon="FILE_REFRESH"
    )
    remove = details.row()
    remove.alert = True
    remove.enabled = maintenance.enabled
    remove.operator("mtp.anycalib_uninstall", text="Remove", icon="TRASH")


def _draw_rohm_management(layout, settings, backend):
    details = layout.box()
    controls = details.column()
    controls.enabled = not COORDINATOR.is_busy()
    controls.prop(settings, "rohm_device", text="Processing Device")
    details.label(
        text=backend.message,
        icon="CHECKMARK" if backend.installed else "INFO",
    )
    _draw_model_job_status(
        details,
        rohm_backend.install_status_snapshot(),
        settings=settings,
        label="AI motion model",
        cancel_operator="mtp.rohm_cancel_install",
    )
    if not backend.installed:
        return
    loaded = rohm_backend.is_model_loaded()
    runtime = details.row(align=True)
    runtime.enabled = not COORDINATOR.is_busy()
    runtime.operator(
        "mtp.rohm_unload" if loaded else "mtp.rohm_load",
        text="Unload" if loaded else "Load",
        icon="MEMORY" if loaded else "IMPORT",
    )
    maintenance = details.row(align=True)
    maintenance.enabled = not loaded and not COORDINATOR.is_busy()
    maintenance.operator("mtp.rohm_install", text="Repair / Move", icon="FILE_REFRESH")
    remove = details.row()
    remove.alert = True
    remove.enabled = maintenance.enabled
    remove.operator("mtp.rohm_uninstall", text="Remove", icon="TRASH")


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

        media_source, cache_status = detection_cache.ui_state_for_settings(settings)
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


def _draw_compact_mapping_editor(layout, armature, entry):
    box = layout.box()
    header = box.row(align=True)
    header.label(text=entry.label, icon=STATUS_ICONS.get(entry.status, "QUESTION"))
    header.prop(entry, "enabled", text="")
    header.prop(
        entry,
        "manual_lock",
        text="",
        icon="LOCKED" if entry.manual_lock else "UNLOCKED",
    )
    if entry.section == "SHAPES":
        box.prop_search(entry, "target_object", bpy.data, "objects", text="Mesh")
        obj = bpy.data.objects.get(entry.target_object)
        keys = (
            obj.data.shape_keys
            if obj and obj.type == "MESH" and obj.data.shape_keys
            else None
        )
        if keys:
            box.prop_search(entry, "target_name", keys, "key_blocks", text="Shape Key")
        else:
            box.prop(entry, "target_name", text="Shape Key")
        row = box.row(align=True)
        row.operator("mtp.use_active_shape_key", icon="EYEDROPPER")
        row.operator("mtp.select_mapping_target", icon="RESTRICT_SELECT_OFF")
        row = box.row(align=True)
        row.prop(entry, "gain")
        row.prop(entry, "offset")
        _draw_shape_overrides(box, entry)
    else:
        if entry.section == "FACE_RIG":
            box.prop(entry, "target_kind")
        box.prop_search(entry, "target_name", armature.pose, "bones", text="Target")
        row = box.row(align=True)
        row.operator("mtp.use_selected_bone", icon="EYEDROPPER")
        row.operator("mtp.select_mapping_target", icon="RESTRICT_SELECT_OFF")
        if entry.section == "FACE_RIG":
            row = box.row(align=True)
            row.prop(entry, "channel_mode")
            row.prop(entry, "axis")
            if entry.channel_mode == "CUSTOM_PROPERTY":
                box.prop(entry, "property_name")
        box.prop(entry, "influence")
    box.prop(entry, "correction_euler")
    details = box.row(align=True)
    details.prop(entry, "source")
    details.prop(entry, "confidence")
    if entry.message:
        box.label(text=entry.message, icon=STATUS_ICONS.get(entry.status, "INFO"))


class MTP_PT_rig_setup(Panel):
    bl_idname = "MTP_PT_rig_setup"
    bl_label = "Rig Setup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Media to Pose"
    bl_order = 1

    def draw_header(self, context):
        armature = _configured_target(context)
        self.layout.label(text="", icon="ARMATURE_DATA" if armature else "ERROR")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtp_settings
        busy = COORDINATOR.is_busy()
        if busy:
            _draw_active_operation(layout, settings)
            layout.label(text="Mapping is locked while processing", icon="LOCKED")
        controls = layout.column()
        controls.enabled = not busy
        layout = controls
        armature = _configured_target(context)
        target = layout.row(align=True)
        target.prop(settings, "target_armature", text="Target Character")
        target.operator("mtp.use_selected_armature", text="Use Selected")
        if armature is None:
            _draw_issue(layout, "Choose a target character", severity=ISSUE_CRITICAL)
            return

        config = armature.mtp_rig_config
        state, icon, required, warnings = _mapping_readiness(config)
        header = layout.box()
        header.label(text=f"{armature.name} · {state}", icon=icon)
        completeness = header.row(align=True)
        completeness.label(text=f"Body {config.body_complete:.0%}")
        completeness.label(text=f"Hands {config.hands_complete:.0%}")
        if settings.face_output == "ARKIT":
            completeness.label(text=f"ARKit {config.arkit_complete:.0%}")
        elif settings.face_output == "FACS":
            completeness.label(text=f"FACS {config.facs_complete:.0%}")
        else:
            completeness.label(text=f"Face {config.face_complete:.0%}")
        if required:
            header.label(text=f"{required} required mapping issue(s)", icon="CANCEL")
        elif warnings:
            header.label(text=f"{warnings} mapping suggestion(s)", icon="ERROR")

        methods = layout.box()
        methods.label(text="Automatic Mapping")
        methods.label(text="Mapping Method")
        grid = methods.grid_flow(columns=2, even_columns=True, align=True)
        for value, label in (
            ("AUTO", "Detect Automatically"),
            ("AUTO_RIG_PRO", "Auto-Rig Pro"),
            ("RIGIFY", "Rigify"),
            ("GENERIC", "Generic / Custom"),
        ):
            operator = grid.operator(
                "mtp.set_mapping_method",
                text=label,
                depress=config.profile == value,
            )
            operator.method = value
        expected = (
            mapping.detect_profile(armature)
            if config.profile == "AUTO"
            else config.profile
        )
        pending = bool(config.mappings and config.detected_profile != expected)
        methods.label(
            text=(
                f"Detected: {mapping.profile_label(config)}"
                if config.profile == "AUTO"
                else f"Selected: {mapping.PROFILE_LABELS.get(expected, expected)}"
            ),
            icon="INFO",
        )
        action_label = (
            "Set Up Rig"
            if not config.mappings
            else "Apply Mapping Method"
            if pending
            else "Update Automatic Mappings"
        )
        methods.operator("mtp.auto_map", text=action_label, icon="MOD_ARMATURE")
        methods.label(
            text="Manual locks are preserved when automatic mappings update.",
            icon="LOCKED",
        )

        if not config.mappings:
            return
        layout.prop(config, "subtab", expand=True)
        if config.subtab == "FACE_RIG":
            face = layout.box()
            face.prop(settings, "face_output", text="Face Output")
            if not settings.use_face:
                face.label(text="Face capture is currently disabled", icon="INFO")

        problems = _mapping_problem_entries(config)
        diagnostics = layout.box()
        if problems:
            diagnostics.label(
                text=f"Needs attention · {len(problems)} problem(s)", icon="ERROR"
            )
            problem_ids = {entry.as_pointer() for entry in problems}
            shown = 0
            for index, entry in enumerate(config.mappings):
                if entry.as_pointer() not in problem_ids:
                    continue
                row = diagnostics.row(align=True)
                row.label(
                    text=entry.label, icon=STATUS_ICONS.get(entry.status, "ERROR")
                )
                row.label(text=entry.message or entry.status.title())
                operator = row.operator(
                    "mtp.select_problem", text="", icon="RESTRICT_SELECT_OFF"
                )
                operator.mapping_index = index
                shown += 1
                if shown >= 3:
                    break
            if len(problems) > 3:
                diagnostics.label(text=f"{len(problems) - 3} additional problem(s)")
        else:
            diagnostics.label(text="Mapping ready", icon="CHECKMARK")

        filters = layout.row(align=True)
        filters.prop(config, "filter_text", text="", icon="VIEWZOOM")
        filters.prop(config, "filter_mode", text="")
        layout.template_list(
            "MTP_UL_mappings",
            f"{config.subtab}_{settings.face_output}",
            config,
            "mappings",
            config,
            "active_mapping_index",
            rows=7,
        )
        entry = active_entry(armature)
        if entry and entry.section == active_mapping_section(context, armature):
            _draw_compact_mapping_editor(layout, armature, entry)
        if config.subtab == "HANDS":
            mirror = layout.row(align=True)
            operator = mirror.operator("mtp.mirror_mapping", text="Left to Right")
            operator.direction = "L_TO_R"
            operator = mirror.operator("mtp.mirror_mapping", text="Right to Left")
            operator.direction = "R_TO_L"
        if config.subtab == "BODY" and _draw_left_disclosure(
            layout,
            config,
            "show_rig_controls",
            "FK/IK Controls",
        ):
            _draw_fk_switches(layout, armature)
        footer = layout.row(align=True)
        footer.menu(
            "MTP_MT_mapping_presets", text="Mapping Presets", icon="DOWNARROW_HLT"
        )
        footer.menu(
            "MTP_MT_mapping_reset", text="Reset or Replace…", icon="RECOVER_LAST"
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
        _media_source, cache_status = detection_cache.ui_state_for_settings(settings)
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
    bl_idname = __package__

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
        _source, status = detection_cache.ui_state_for_settings(settings)
        storage.label(text=f"Current source: {_cache_size_label(status.size_bytes)}")
        clear = storage.row()
        clear.alert = True
        clear.operator(
            "mtp.clear_detection_cache", text="Clear Stored Analysis…", icon="TRASH"
        )


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
