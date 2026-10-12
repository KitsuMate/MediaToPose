"""Capture operators: a thin modal driver of CaptureSession, plus media tools."""

from __future__ import annotations

import time

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator

from ...api.common import drive
from ...core.cache import media_source as cache_media
from ...core.cache import store as cache_store
from ...core.coordinator import COORDINATOR, OperationKind, OperationState
from ...core.errors import MtpError
from ...core.requests import CaptureMode, CaptureSettings
from ...core.scene import SceneContext
from ...session.capture_session import (
    CaptureSession,
    SessionUpdate,
    active_session,
    shutdown_sessions,
)
from .. import preview as cache_preview
from ..classes import register_classes, unregister_classes
from .mapping import active_armature


def _set_status(settings, message: str, kind: str = "INFO") -> None:
    if getattr(settings, "status", "") != str(message):
        settings.show_result_details = False
    settings.status = str(message)
    settings.status_kind = kind


def _tag_progress_redraw(context) -> None:
    """Request sidebar redraws after timer-driven progress changes."""
    screens = []
    screen = getattr(context, "screen", None)
    if screen is not None:
        screens.append(screen)
    window_manager = getattr(context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        window_screen = getattr(window, "screen", None)
        if window_screen is not None and all(
            window_screen is not existing for existing in screens
        ):
            screens.append(window_screen)
    for redraw_screen in screens:
        for area in getattr(redraw_screen, "areas", ()):
            if getattr(area, "type", None) == "VIEW_3D":
                area.tag_redraw()


def _report_level(kind: str) -> set[str]:
    return (
        {"ERROR"} if kind == "ERROR" else {"WARNING" if kind == "WARNING" else "INFO"}
    )


class MTP_OT_apply_capture(Operator):
    bl_idname = "mtp.apply_capture"
    bl_label = "Apply Vision Capture"
    bl_description = "Detect and retarget the selected image or video"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        items=(
            ("POSE", "Pose", ""),
            ("KEYFRAMES", "Keyframes", ""),
            ("ANALYZE", "Analyze", ""),
        ),
        default="POSE",
        options={"HIDDEN"},
    )
    force_analysis: BoolProperty(default=False, options={"HIDDEN"})

    _session: CaptureSession | None = None
    _timer = None
    _interval = 0.0
    _last_ui_update = 0.0

    def _prepare(self, context) -> CaptureSession | None:
        settings = context.scene.mtp_settings
        try:
            session = CaptureSession.prepare(
                SceneContext.from_context(context),
                active_armature(context),
                CaptureSettings.from_rna(settings),
                CaptureMode(self.mode),
                force_analysis=self.force_analysis,
            )
        except MtpError as exc:
            if exc.details.get("show_status", True):
                _set_status(settings, str(exc), "ERROR")
            self.report({"ERROR"}, str(exc))
            return None
        for kind, message in session.notices:
            _set_status(settings, message, kind)
            self.report(_report_level(kind), message)
        return session

    def invoke(self, context, _event):
        self._session = None
        self._timer = None
        session = self._prepare(context)
        if session is None:
            return {"CANCELLED"}
        settings = context.scene.mtp_settings
        try:
            update = session.start()
        except MtpError as exc:
            _set_status(settings, str(exc), "ERROR")
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self._session = session
        self._mirror(settings, update)
        if update.state.finished:
            return self._report_finished(context, update)
        self._schedule(context, update.interval)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        """Run the whole capture now; used by scripts calling the operator.

        The panel invokes the operator and runs modally instead.
        """
        self._session = None
        self._timer = None
        session = self._prepare(context)
        if session is None:
            return {"CANCELLED"}
        settings = context.scene.mtp_settings
        self._session = session
        try:
            update = drive(session)
        except MtpError as exc:
            self._mirror(settings, session.step())
            self.report({"ERROR"}, str(exc))
            return self._end(context, {"CANCELLED"})
        self._mirror(settings, update)
        return self._report_finished(context, update)

    def _schedule(self, context, interval: float) -> None:
        if self._timer is not None and abs(interval - self._interval) < 1.0e-6:
            return
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
        self._interval = interval
        self._timer = context.window_manager.event_timer_add(
            interval, window=context.window
        )

    def _mirror(self, settings, update: SessionUpdate) -> None:
        _set_status(settings, update.status, update.status_kind)
        settings.progress = update.progress
        preview = self._session.take_preview_request() if self._session else ""
        if preview == "cache":
            cache_preview.refresh_after_cache_update()
        elif preview:
            cache_preview.schedule_refresh()
        outcome = self._session.outcome if self._session else None
        if outcome is not None and outcome.camera_vertical_fov:
            settings.last_camera_vertical_fov = outcome.camera_vertical_fov
            settings.last_camera_source = outcome.camera_source

    def modal(self, context, event):
        session = self._session
        if session is None:
            return self._end(context, {"CANCELLED"})
        scene = bpy.data.scenes.get(session.request.scene_name)
        if event.type == "ESC":
            session.cancel()
            if scene is not None:
                _set_status(scene.mtp_settings, "Cancelling", "PROGRESS")
            _tag_progress_redraw(context)
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if (
            scene is not None
            and context.scene.as_pointer() != session.request.scene_pointer
        ):
            message = "Cancelling because the active scene changed"
            session.cancel(message)
            _set_status(scene.mtp_settings, message, "PROGRESS")
            return {"RUNNING_MODAL"}
        # Timer events carry no mouse movement, so tag the 3D View explicitly
        # to repaint the sidebar progress bar.
        now = time.monotonic()
        if now - self._last_ui_update >= 0.1:
            _tag_progress_redraw(context)
            self._last_ui_update = now
        update = session.step()
        if scene is not None:
            self._mirror(scene.mtp_settings, update)
        if update.state.finished:
            return self._report_finished(context, update)
        self._schedule(context, update.interval)
        return {"RUNNING_MODAL"}

    def _report_finished(self, context, update: SessionUpdate):
        if update.status:
            self.report(_report_level(update.status_kind), update.status)
        result = {"FINISHED"} if update.state == "SUCCEEDED" else {"CANCELLED"}
        return self._end(context, result)

    def _end(self, context, result: set[str]):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _tag_progress_redraw(context)
        return result

    def cancel(self, context):
        session = self._session
        if session is not None and not session.finished:
            session.abort()
            scene = bpy.data.scenes.get(session.request.scene_name)
            if scene is not None:
                self._mirror(scene.mtp_settings, session.step())
        self._end(context, {"CANCELLED"})


class MTP_OT_cancel_processing(Operator):
    bl_idname = "mtp.cancel_processing"
    bl_label = "Cancel Processing"
    bl_description = (
        "Cancel inference or sequence processing before keyframes are written"
    )

    def execute(self, context):
        session = active_session()
        if session is None or session.job is None:
            return {"CANCELLED"}
        if not COORDINATOR.request_cancel(session.job.token):
            return {"CANCELLED"}
        scene = bpy.data.scenes.get(session.request.scene_name)
        if scene is not None:
            _set_status(scene.mtp_settings, "Cancelling", "PROGRESS")
        return {"FINISHED"}


class MTP_OT_select_media(Operator):
    bl_idname = "mtp.select_media"
    bl_label = "Select Media"
    bl_description = "Select or relocate the source image or video"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(
        default=(
            "*.bmp;*.exr;*.gif;*.hdr;*.jpeg;*.jpg;*.png;*.tga;"
            "*.tif;*.tiff;*.webp;*.avi;*.mkv;*.mov;*.mp4;*.m4v;*.webm"
        ),
        options={"HIDDEN"},
    )

    def invoke(self, context, _event):
        settings = context.scene.mtp_settings
        if settings.media_path:
            self.filepath = bpy.path.abspath(settings.media_path)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        settings = context.scene.mtp_settings
        selected = cache_media.existing_media_path(self.filepath)
        if not selected:
            message = "Choose an existing image or video"
            _set_status(settings, message, "ERROR")
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        try:
            if not cache_media.record_media_selection(settings, selected):
                raise ValueError("Choose an existing image or video")
        except (OSError, RuntimeError, ValueError) as exc:
            _set_status(settings, str(exc), "ERROR")
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _set_status(settings, "Source media selected", "INFO")
        return {"FINISHED"}


class MTP_OT_clear_detection_cache(Operator):
    bl_idname = "mtp.clear_detection_cache"
    bl_label = "Clear Detection Cache"
    bl_description = "Delete the retained Media to Pose detection cache"

    @classmethod
    def poll(cls, context):
        settings = getattr(context.scene, "mtp_settings", None)
        return bool(settings and not COORDINATOR.is_busy())

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        token = COORDINATOR.acquire(
            OperationKind.CACHE_MAINTENANCE,
            "clear-detection-cache",
            scene_name=context.scene.name,
            scene_pointer=context.scene.as_pointer(),
            status="Clearing detection cache",
        )
        if token is None:
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        try:
            cache_store.clear_cache()
            cache_preview.refresh_after_cache_update()
        except (OSError, RuntimeError, ValueError) as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "Cache cleared")
        COORDINATOR.release(token)
        _set_status(
            context.scene.mtp_settings,
            "Detection cache cleared",
            "INFO",
        )
        return {"FINISHED"}


class MTP_OT_rebuild_camera_motion_track(Operator):
    bl_idname = "mtp.rebuild_camera_motion_track"
    bl_label = "Rebuild Camera Track"
    bl_description = (
        "Delete the derived background-camera track so the next video bake rebuilds it"
    )

    @classmethod
    def poll(cls, context):
        settings = getattr(context.scene, "mtp_settings", None)
        return bool(settings and not COORDINATOR.is_busy())

    def execute(self, context):
        token = COORDINATOR.acquire(
            OperationKind.CACHE_MAINTENANCE,
            "rebuild-camera-motion-track",
            scene_name=context.scene.name,
            scene_pointer=context.scene.as_pointer(),
            status="Clearing camera track",
        )
        if token is None:
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        try:
            cache_store.clear_camera_motion_cache()
            armature = getattr(context.scene.mtp_settings, "target_armature", None)
            if armature is not None:
                item = next(
                    (
                        value
                        for value in armature.mtp_rig_config.post_processing_modules
                        if value.module_id == "CAMERA_MOTION_CORRECTION"
                    ),
                    None,
                )
                if item is not None:
                    item.camera_track_status = "Not Prepared"
                    item.camera_tracked_frames = 0
                    item.camera_rejected_frames = 0
                    item.camera_median_inliers = 0.0
                    item.camera_corrected_intervals = 0
                    item.last_warning = ""
                    item.status = "VALID"
        except (OSError, RuntimeError, ValueError) as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "Camera track cleared")
        COORDINATOR.release(token)
        _set_status(
            context.scene.mtp_settings,
            "Camera track will be rebuilt on the next video bake",
            "INFO",
        )
        return {"FINISHED"}


class MTP_OT_apply_recovery_choice(Operator):
    bl_idname = "mtp.apply_recovery_choice"
    bl_label = "Improve Result"
    bl_description = "Apply a suggested non-destructive workflow adjustment"
    bl_options = {"UNDO"}

    choice: EnumProperty(
        items=(
            ("SAM", "Try SAM 3D Body", "Use detailed pose detection"),
            ("IN_PLACE", "Create In-Place Animation", "Remove detected travel"),
        )
    )

    def execute(self, context):
        settings = context.scene.mtp_settings
        if self.choice == "SAM":
            settings.provider = "SAM3D_BODY"
            _set_status(settings, "SAM 3D Body selected · apply again to retry", "INFO")
        elif self.choice == "IN_PLACE":
            settings.root_motion_mode = "IN_PLACE"
            _set_status(
                settings,
                "In-place movement selected · create the animation again",
                "INFO",
            )
        return {"FINISHED"}


CLASSES = (
    MTP_OT_apply_capture,
    MTP_OT_cancel_processing,
    MTP_OT_select_media,
    MTP_OT_clear_detection_cache,
    MTP_OT_rebuild_camera_motion_track,
    MTP_OT_apply_recovery_choice,
)


def register() -> None:
    register_classes(CLASSES)


def unregister() -> None:
    shutdown_sessions()
    unregister_classes(CLASSES)
