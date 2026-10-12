"""Install, load, unload and remove the optional model backends.

The operators only translate between Blender and the backend modules: they
start background jobs, mirror job progress into the scene settings while a
modal timer runs, and report outcomes. All lifecycle logic lives in
``backends/``.
"""

from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from ... import anycalib_backend
from ...backends.lifecycle import BackgroundJob, JobKind, ModelBackend
from ...core.errors import MtpError, OperationBusy
from ...core.log import get_logger
from ...operation import COORDINATOR
from ...registration import register_classes, unregister_classes

LOGGER = get_logger(__name__)
BUSY = "Media to Pose is busy with another operation"


def _set_status(context, message: str, kind: str, *, scene_name: str = "") -> None:
    scene = bpy.data.scenes.get(scene_name) if scene_name else context.scene
    settings = getattr(scene, "mtp_settings", None)
    if settings is not None:
        settings.status = message
        settings.status_kind = kind


def _tag_redraw(context) -> None:
    for area in context.screen.areas if context.screen else ():
        area.tag_redraw()


class ModelJobModal:
    """Mirror one backend job into the scene status until its thread ends."""

    backend: ModelBackend
    job_kind = JobKind.INSTALL
    cancel_status = "Cancelling safely"
    stopped_message = "Operation stopped"

    _timer = None
    _job: BackgroundJob | None = None
    _scene_name = ""

    def _watch(self, context, job: BackgroundJob, status: str):
        self._job = job
        self._scene_name = context.scene.name
        settings = context.scene.mtp_settings
        settings.progress = 0.0
        settings.status = status
        settings.status_kind = "PROGRESS"
        self._timer = context.window_manager.event_timer_add(
            0.15, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _stop_timer(self, context) -> None:
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def modal(self, context, event):
        job = self._job
        if job is None:
            self._stop_timer(context)
            return {"CANCELLED"}
        if event.type == "ESC":
            self.backend.cancel_job(self.job_kind, self.cancel_status)
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        snapshot = job.snapshot()
        if self.backend.job(self.job_kind) is job:
            scene = bpy.data.scenes.get(self._scene_name)
            settings = getattr(scene, "mtp_settings", None)
            if settings is not None:
                settings.progress = snapshot.progress * 100.0
                settings.status = snapshot.status
                settings.status_kind = (
                    "PROGRESS"
                    if snapshot.running
                    else "READY"
                    if snapshot.outcome == "DONE"
                    else "WARNING"
                    if snapshot.outcome == "CANCELLED"
                    else "ERROR"
                )
        _tag_redraw(context)
        if job.alive():
            return {"RUNNING_MODAL"}
        self._stop_timer(context)
        if job.token is not None:
            COORDINATOR.release(job.token)
        if snapshot.outcome == "DONE":
            self.report({"INFO"}, snapshot.status)
            return {"FINISHED"}
        self.report(
            {"WARNING" if snapshot.outcome == "CANCELLED" else "ERROR"},
            snapshot.status or self.stopped_message,
        )
        return {"CANCELLED"}

    def cancel(self, context):
        self.backend.cancel_job(self.job_kind, self.cancel_status)
        self._stop_timer(context)


def _run_exclusive(operator, action) -> set[str]:
    """Run a short backend action and report its result on the operator."""
    try:
        message = action()
    except OperationBusy as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    except (MtpError, OSError) as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    operator.report({"INFO"}, message)
    return {"FINISHED"}


# --------------------------------------------------------------------------
# AnyCalib


class MTP_OT_anycalib_install(ModelJobModal, Operator):
    bl_idname = "mtp.anycalib_install"
    bl_label = "Install Camera Calibration Model"
    bl_description = "Download, verify, repair, or move the AnyCalib model"

    backend = anycalib_backend.BACKEND
    cancel_status = "Cancelling camera model download"
    stopped_message = "Camera model installation stopped"

    model_storage_directory: StringProperty(
        name="Model Storage Parent", subtype="DIR_PATH", default=""
    )

    @classmethod
    def poll(cls, _context):
        if anycalib_backend.install_status_snapshot().running:
            cls.poll_message_set("Camera model installation is already running")
            return False
        if COORDINATOR.is_busy():
            cls.poll_message_set(BUSY)
            return False
        return True

    def _report_blocked(self, context) -> None:
        message = (
            "Camera model installation is already running"
            if anycalib_backend.install_status_snapshot().running
            else BUSY
        )
        _set_status(context, message, "WARNING")
        self.report({"WARNING"}, message)

    def invoke(self, context, _event):
        if anycalib_backend.install_status_snapshot().running or COORDINATOR.is_busy():
            self._report_blocked(context)
            return {"CANCELLED"}
        self.model_storage_directory = str(anycalib_backend.configured_storage_parent())
        return context.window_manager.invoke_props_dialog(self, width=540)

    def draw(self, _context):
        spec = anycalib_backend.MODEL_SPEC
        requested = anycalib_backend._preferred_precision("AUTO")
        size = int(spec["variants"][requested]["download_bytes"])
        self.layout.label(
            text=f"Download: {size / (1024**2):.0f} MiB · processing remains local",
            icon="IMPORT",
        )
        self.layout.label(text=f"Source: {spec['repo']}", icon="WORLD")
        self.layout.label(text=f"Preferred revision: {spec['revision']}", icon="LOCKED")
        self.layout.prop(self, "model_storage_directory")

    def execute(self, context):
        if anycalib_backend.install_status_snapshot().running or COORDINATOR.is_busy():
            self._report_blocked(context)
            return {"CANCELLED"}
        try:
            job = anycalib_backend.start_install(
                self.model_storage_directory,
                context.scene.mtp_settings.anycalib_device,
            )
        except OperationBusy:
            self._report_blocked(context)
            return {"CANCELLED"}
        except (MtpError, OSError) as exc:
            self.report({"ERROR"}, f"Could not start camera model download: {exc}")
            return {"CANCELLED"}
        return self._watch(context, job, "Starting camera model download")


class MTP_OT_anycalib_cancel_install(Operator):
    bl_idname = "mtp.anycalib_cancel_install"
    bl_label = "Cancel AnyCalib Download"

    def execute(self, _context):
        return {"FINISHED"} if anycalib_backend.cancel_install() else {"CANCELLED"}


class MTP_OT_anycalib_load(Operator):
    bl_idname = "mtp.anycalib_load"
    bl_label = "Load AnyCalib"

    def execute(self, context):
        try:
            anycalib_backend.start_load(context.scene.mtp_settings.anycalib_device)
        except OperationBusy:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        except (MtpError, OSError) as exc:
            self.report({"ERROR"}, f"Could not start camera model load: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class MTP_OT_anycalib_unload(Operator):
    bl_idname = "mtp.anycalib_unload"
    bl_label = "Unload AnyCalib"

    def execute(self, _context):
        return _run_exclusive(self, anycalib_backend.unload)


class MTP_OT_anycalib_uninstall(Operator):
    bl_idname = "mtp.anycalib_uninstall"
    bl_label = "Uninstall AnyCalib"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, _context):
        return _run_exclusive(self, anycalib_backend.uninstall)


CLASSES = (
    MTP_OT_anycalib_install,
    MTP_OT_anycalib_cancel_install,
    MTP_OT_anycalib_load,
    MTP_OT_anycalib_unload,
    MTP_OT_anycalib_uninstall,
)


def register() -> None:
    anycalib_backend.invalidate_backend_snapshot()
    register_classes(CLASSES)


def unregister() -> None:
    anycalib_backend.shutdown()
    unregister_classes(CLASSES)
