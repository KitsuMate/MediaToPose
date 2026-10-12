"""Install, load and remove controls for the optional models."""

from __future__ import annotations

from ...backends.anycalib import backend as anycalib_backend
from ...backends.rohm import backend as rohm_backend
from ...backends.sam3d import backend as sam3d_backend
from ...core.coordinator import COORDINATOR
from .widgets import ISSUE_CRITICAL, ISSUE_WARNING, _draw_issue, _draw_wrapped_labels


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
