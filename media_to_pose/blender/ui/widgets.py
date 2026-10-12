"""Small drawing helpers and status labels shared by the panels."""

from __future__ import annotations

import textwrap

import bpy

from ...backends.sam3d import backend as sam3d_backend
from ...core.coordinator import COORDINATOR, OperationKind

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


def _configuration_panels_available(context) -> bool:
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "mtp_settings", None)
    if not settings or settings.provider != "SAM3D_BODY":
        return True
    return sam3d_backend.sam_backend_ready(settings)


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
