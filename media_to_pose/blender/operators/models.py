"""Install, load, unload and remove the optional model backends.

The operators only translate between Blender and the backend modules: they
start background jobs, mirror job progress into the scene settings while a
modal timer runs, and report outcomes. All lifecycle logic lives in
``backends/``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from ...backends.anycalib import backend as anycalib_backend
from ...backends.lifecycle import BackgroundJob, JobKind, ModelBackend
from ...backends.rohm import backend as rohm_backend
from ...backends.sam3d import backend as sam3d_backend
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


# --------------------------------------------------------------------------
# RoHM


class MTP_OT_rohm_install(ModelJobModal, Operator):
    bl_idname = "mtp.rohm_install"
    bl_label = "Install AI Motion Repair"
    bl_description = "Download, verify, repair, or move the RoHM model"

    backend = rohm_backend.BACKEND
    cancel_status = "Cancelling AI motion model download"
    stopped_message = "AI motion model installation stopped"

    model_storage_directory: StringProperty(
        name="Model Storage Parent",
        subtype="DIR_PATH",
        default="",
    )

    @classmethod
    def poll(cls, _context):
        if rohm_backend.install_status_snapshot().running:
            cls.poll_message_set("AI motion model installation is already running")
            return False
        if COORDINATOR.is_busy():
            cls.poll_message_set(BUSY)
            return False
        return True

    def invoke(self, context, _event):
        if rohm_backend.install_status_snapshot().running or COORDINATOR.is_busy():
            message = (
                "AI motion model installation is already running"
                if rohm_backend.install_status_snapshot().running
                else BUSY
            )
            _set_status(context, message, "WARNING")
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        self.model_storage_directory = str(rohm_backend.configured_storage_parent())
        return context.window_manager.invoke_props_dialog(self, width=540)

    def draw(self, _context):
        layout = self.layout
        spec = rohm_backend.MODEL_SPEC
        size = int(spec["download_bytes"])
        layout.label(
            text=f"Download: {size / (1024**2):.0f} MiB · processing remains local",
            icon="IMPORT",
        )
        layout.label(text=f"Source: {spec['repo']}", icon="WORLD")
        layout.label(text=f"Preferred revision: {spec['revision']}", icon="LOCKED")
        layout.prop(self, "model_storage_directory")

    def execute(self, context):
        if rohm_backend.install_status_snapshot().running or COORDINATOR.is_busy():
            self.report({"WARNING"}, "Media to Pose is busy")
            return {"CANCELLED"}
        try:
            job = rohm_backend.start_install(self.model_storage_directory)
        except OperationBusy:
            self.report({"WARNING"}, "Media to Pose is busy")
            return {"CANCELLED"}
        except (MtpError, OSError) as exc:
            self.report({"ERROR"}, f"Could not start AI motion model download: {exc}")
            return {"CANCELLED"}
        return self._watch(context, job, "Starting AI motion model download")


class MTP_OT_rohm_cancel_install(Operator):
    bl_idname = "mtp.rohm_cancel_install"
    bl_label = "Cancel RoHM Download"

    def execute(self, _context):
        return {"FINISHED"} if rohm_backend.cancel_install() else {"CANCELLED"}


class MTP_OT_rohm_load(Operator):
    bl_idname = "mtp.rohm_load"
    bl_label = "Load RoHM"

    def execute(self, context):
        try:
            rohm_backend.start_load(context.scene.mtp_settings.rohm_device)
        except OperationBusy:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        except (MtpError, OSError) as exc:
            self.report({"ERROR"}, f"Could not start AI motion model load: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class MTP_OT_rohm_unload(Operator):
    bl_idname = "mtp.rohm_unload"
    bl_label = "Unload RoHM"

    def execute(self, _context):
        return _run_exclusive(self, rohm_backend.unload)


class MTP_OT_rohm_verify(Operator):
    bl_idname = "mtp.rohm_verify"
    bl_label = "Verify RoHM"

    def execute(self, context):
        device = context.scene.mtp_settings.rohm_device
        return _run_exclusive(self, lambda: rohm_backend.verify(device))


class MTP_OT_rohm_uninstall(Operator):
    bl_idname = "mtp.rohm_uninstall"
    bl_label = "Uninstall RoHM"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, _context):
        return _run_exclusive(self, rohm_backend.uninstall)


# --------------------------------------------------------------------------
# SAM 3D Body


class MTP_OT_sam3d_setup(Operator):
    bl_idname = "mtp.sam3d_setup"
    bl_label = "SAM 3D Body Setup"
    bl_description = "Install the public SAM 3D Body ONNX model bundle"

    def invoke(self, _context, _event):
        return bpy.ops.mtp.sam3d_install("INVOKE_DEFAULT")

    def execute(self, _context):
        return bpy.ops.mtp.sam3d_install("INVOKE_DEFAULT")


class MTP_OT_sam3d_install(ModelJobModal, Operator):
    bl_idname = "mtp.sam3d_install"
    bl_label = "Install SAM 3D Body"
    bl_description = "Download and install the optional SAM 3D Body backend"

    backend = sam3d_backend.BACKEND
    cancel_status = "Cancelling installation safely"
    stopped_message = "SAM 3D Body installation stopped"

    model_storage_directory: StringProperty(
        name="Model Storage Location",
        description=(
            "Parent directory where Media to Pose stores the downloaded "
            "SAM 3D Body ONNX graph bundle"
        ),
        subtype="DIR_PATH",
        options={"SKIP_SAVE"},
    )

    def invoke(self, context, _event):
        if COORDINATOR.is_busy():
            self.report({"INFO"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        self.model_storage_directory = str(
            sam3d_backend.preferred_model_storage_parent()
        )
        present = sam3d_backend.backend_present()
        return context.window_manager.invoke_props_dialog(
            self,
            width=620,
            title="Repair SAM 3D Body" if present else "Install SAM 3D Body",
            confirm_text="Repair" if present else "Install",
        )

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtp_settings
        spec = sam3d_backend.model_spec(settings.sam3d_model_variant)
        size = int(spec.get("download_bytes") or 0)
        label = f"Download {spec['label']} ONNX graph bundle"
        if size:
            label += f" ({size / (1024**3):.1f} GiB)"
        layout.label(text=label, icon="IMPORT")
        layout.label(
            text=f"Source: {spec['repo']} @ {spec.get('revision') or 'not published'}",
            icon="PACKAGE",
        )
        layout.prop(self, "model_storage_directory")
        try:
            managed = sam3d_backend.model_storage_root(self.model_storage_directory)
            layout.label(text=f"Managed folder: {managed}", icon="FILE_FOLDER")
        except (MtpError, OSError, RuntimeError, ValueError):
            layout.label(
                text="Choose an accessible model storage location",
                icon="ERROR",
            )
        layout.label(
            text="ONNX Runtime is shipped with the extension; setup installs models only.",
            icon="INFO",
        )

    def execute(self, context):
        if COORDINATOR.is_busy():
            return {"CANCELLED"}
        settings = context.scene.mtp_settings
        try:
            job = sam3d_backend.start_install(
                str(settings.sam3d_model_variant),
                str(settings.sam3d_device),
                self.model_storage_directory,
                scene_name=context.scene.name,
                scene_pointer=context.scene.as_pointer(),
            )
        except OperationBusy:
            self.report({"ERROR"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        except (MtpError, OSError) as exc:
            self.report({"ERROR"}, sam3d_backend._safe_metadata(exc))
            return {"CANCELLED"}
        return self._watch(context, job, "Starting SAM 3D Body installation")


class MTP_OT_sam3d_cancel_install(Operator):
    bl_idname = "mtp.sam3d_cancel_install"
    bl_label = "Cancel SAM 3D Body Installation"

    def execute(self, _context):
        return {"FINISHED"} if sam3d_backend.cancel_install() else {"CANCELLED"}


class MTP_OT_sam3d_copy_install_diagnostics(Operator):
    bl_idname = "mtp.sam3d_copy_install_diagnostics"
    bl_label = "Copy SAM Installation Diagnostics"
    bl_description = "Copy redacted SAM installation details and recent logs"

    def execute(self, context):
        context.window_manager.clipboard = sam3d_backend.install_diagnostics_text()
        self.report({"INFO"}, "Redacted installation diagnostics copied")
        return {"FINISHED"}


class MTP_OT_sam3d_open_install_log(Operator):
    bl_idname = "mtp.sam3d_open_install_log"
    bl_label = "Open SAM Installation Log"
    bl_description = "Open the retained SAM installation log"

    def execute(self, _context):
        path = sam3d_backend.install_log_file()
        if not path.is_file():
            self.report({"WARNING"}, "No SAM installation log is available")
            return {"CANCELLED"}
        try:
            bpy.ops.wm.path_open(filepath=str(path))
        except RuntimeError:
            try:
                if os.name == "nt":
                    os.startfile(str(path))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
            except OSError as exc:
                self.report(
                    {"ERROR"},
                    f"Could not open install log: {sam3d_backend._safe_metadata(exc)}",
                )
                return {"CANCELLED"}
        return {"FINISHED"}


class MTP_OT_sam3d_dismiss_install_result(Operator):
    bl_idname = "mtp.sam3d_dismiss_install_result"
    bl_label = "Dismiss SAM Installation Result"

    def execute(self, context):
        if not sam3d_backend.dismiss_install_result():
            return {"CANCELLED"}
        settings = context.scene.mtp_settings
        settings.progress = 0.0
        settings.status = "Ready"
        settings.status_kind = "READY"
        return {"FINISHED"}


class MTP_OT_sam3d_load(ModelJobModal, Operator):
    bl_idname = "mtp.sam3d_load"
    bl_label = "Load SAM 3D Body Model"
    bl_description = "Load the installed model into RAM or VRAM for capture"

    backend = sam3d_backend.BACKEND
    job_kind = JobKind.MODEL
    cancel_status = "Cancelling"
    stopped_message = "SAM 3D Body model loading stopped"

    def execute(self, context):
        settings = context.scene.mtp_settings
        if COORDINATOR.is_busy():
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        if sam3d_backend.is_model_loaded():
            self.report({"INFO"}, "SAM 3D Body model is already loaded")
            return {"FINISHED"}
        try:
            job = sam3d_backend.start_load(
                str(settings.sam3d_model_variant),
                str(settings.sam3d_device),
                scene_name=context.scene.name,
                scene_pointer=context.scene.as_pointer(),
            )
        except OperationBusy:
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        except (MtpError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return self._watch(context, job, "Loading SAM 3D Body model")


class MTP_OT_sam3d_unload(Operator):
    bl_idname = "mtp.sam3d_unload"
    bl_label = "Unload SAM 3D Body"
    bl_description = "Release the loaded SAM 3D Body model and GPU memory"

    def execute(self, _context):
        return _run_exclusive(self, sam3d_backend.unload)


class MTP_OT_sam3d_uninstall(Operator):
    bl_idname = "mtp.sam3d_uninstall"
    bl_label = "Uninstall SAM 3D Body"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, _context):
        return _run_exclusive(self, sam3d_backend.uninstall)


CLASSES = (
    MTP_OT_sam3d_setup,
    MTP_OT_sam3d_install,
    MTP_OT_sam3d_cancel_install,
    MTP_OT_sam3d_copy_install_diagnostics,
    MTP_OT_sam3d_open_install_log,
    MTP_OT_sam3d_dismiss_install_result,
    MTP_OT_sam3d_load,
    MTP_OT_sam3d_unload,
    MTP_OT_sam3d_uninstall,
    MTP_OT_anycalib_install,
    MTP_OT_anycalib_cancel_install,
    MTP_OT_anycalib_load,
    MTP_OT_anycalib_unload,
    MTP_OT_anycalib_uninstall,
    MTP_OT_rohm_install,
    MTP_OT_rohm_cancel_install,
    MTP_OT_rohm_load,
    MTP_OT_rohm_unload,
    MTP_OT_rohm_verify,
    MTP_OT_rohm_uninstall,
)


def register() -> None:
    sam3d_backend.invalidate_backend_snapshot()
    anycalib_backend.invalidate_backend_snapshot()
    rohm_backend.invalidate_backend_snapshot()
    register_classes(CLASSES)


def unregister() -> None:
    sam3d_backend.shutdown()
    anycalib_backend.shutdown()
    rohm_backend.shutdown()
    unregister_classes(CLASSES)
