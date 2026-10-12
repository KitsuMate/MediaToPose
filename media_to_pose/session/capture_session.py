"""One capture from media to pose or keyframes, driven by ``step()``.

``CaptureSession.prepare`` validates the settings, rig and media and builds
the frozen ``CaptureRequest``. ``start`` acquires the operation lease and
either starts the detection worker or begins applying cached detections.
``step`` does a bounded amount of main-thread work and returns a
``SessionUpdate``; ``cancel`` asks the session to stop. The modal operator,
the synchronous API and the command line all drive the same session.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import bpy

from ..core.cache import store as cache_store
from ..core.coordinator import COORDINATOR, OperationKind, OperationState
from ..core.errors import (
    InvalidRequest,
    MappingIncomplete,
    MediaError,
    ModelMissing,
    MtpError,
    OperationBusy,
)
from ..core.log import get_logger
from ..core.media_io import probe_media
from ..core.report import CaptureReport
from ..core.requests import (
    CaptureMode,
    CaptureRequest,
    CaptureSettings,
    MediaKind,
    OutputMode,
    PostProcessConfig,
    RootMotionMode,
)
from ..core.scene import SceneContext
from ..core.types import CanonicalFrame, MediaInfo
from ..postprocess import compile_postprocess_config
from ..providers import descriptor
from ..retarget import (
    iter_apply_frame,
    iter_apply_sequence,
    prepare_limb_ownership,
    validate_limb_switch_interval,
)
from ..rig import mapping
from ..rig import planning as rig_planning
from . import capture_options
from .progress import (
    rohm_may_run,
    sequence_progress_group,
    sequence_progress_percent,
    sequence_progress_status,
)
from .readiness import enabled_capture_scope, face_output_warning, mapping_scope_issues
from .results import FaceSummary, part_confidence
from .transaction import CaptureTransaction
from .worker import DetectionJob, run_detection

LOGGER = get_logger(__name__)

CANCELLED_RESTORED = "Cancelled · the previous pose and animation were restored"

# Sessions that own the operation lease, by coordinator generation.
ACTIVE_SESSIONS: dict[int, CaptureSession] = {}


class SessionState(StrEnum):
    PREPARED = "PREPARED"
    DETECTING = "DETECTING"
    APPLYING_POSE = "APPLYING_POSE"
    APPLYING_SEQUENCE = "APPLYING_SEQUENCE"
    SUCCEEDED = "SUCCEEDED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

    @property
    def finished(self) -> bool:
        return self in {
            SessionState.SUCCEEDED,
            SessionState.CANCELLED,
            SessionState.FAILED,
        }


@dataclass(frozen=True)
class SessionUpdate:
    """What a driver shows after one ``step()``."""

    state: SessionState
    status: str
    status_kind: str
    progress: float
    interval: float


@dataclass
class CaptureOutcome:
    """Counts and observations collected while the session ran."""

    frames: int = 0
    bones: int = 0
    shapes: int = 0
    warnings: list[str] = field(default_factory=list)
    camera_vertical_fov: float = 0.0
    camera_source: str = ""
    cached: bool = False
    cache_persisted: bool = False
    action_name: str = ""


def _origin_scene(request: CaptureRequest):
    scene = bpy.data.scenes.get(request.scene_name)
    if scene is None or scene.as_pointer() != request.scene_pointer:
        return None
    return scene


def _origin_armature(request: CaptureRequest):
    if not request.target_name:
        return None
    armature = bpy.data.objects.get(request.target_name)
    if (
        armature is None
        or armature.as_pointer() != request.target_pointer
        or armature.type != "ARMATURE"
    ):
        return None
    return armature


def _check_rohm(armature, settings: CaptureSettings) -> None:
    from ..backends.rohm import backend as rohm_backend

    rohm = next(
        (
            item
            for item in armature.mtp_rig_config.post_processing_modules
            if item.module_id == "ROHM_CORRECTION"
        ),
        None,
    )
    if rohm is None or not rohm.enabled:
        return
    if not (
        rohm.learned_pose_residual
        or (
            rohm.root_trajectory_cleanup
            and settings.root_motion_mode == RootMotionMode.AUTOMATIC
        )
    ):
        return
    model = rohm_backend.backend_snapshot(settings.rohm_device)
    if not model.installed:
        raise ModelMissing(
            "RoHM Correction is enabled but its verified model "
            f"is unavailable: {model.message}"
        )
    try:
        # Load and verify before the capture request so a bad model cannot
        # partially mutate the animation transaction.
        rohm_backend.ensure_model_loaded(settings.rohm_device)
    except Exception as exc:  # The ONNX runtime raises its own types.
        LOGGER.exception("RoHM model could not be loaded")
        raise ModelMissing(
            f"RoHM Correction could not load its verified model: {exc}"
        ) from exc


class CaptureSession:
    """Drive one capture; see the module docstring for the lifecycle."""

    def __init__(
        self,
        scene_context: SceneContext,
        armature,
        settings: CaptureSettings,
        request: CaptureRequest,
        *,
        cached_frame: CanonicalFrame | None = None,
        cached_frames: Iterator[CanonicalFrame] | None = None,
        immediate_status: str = "",
        notices: tuple[tuple[str, str], ...] = (),
    ):
        self.scene_context = scene_context
        self.armature = armature
        self.settings = settings
        self.request = request
        self.notices = notices
        self.outcome = CaptureOutcome()
        frame_range = request.frame_range
        self.report = CaptureReport(
            source_frame_offset=frame_range[0] if frame_range else 0
        )
        # Per applied frame: source frame, timeline frame and part confidence.
        self.frame_confidence: list[dict] = []
        self.face_summary = FaceSummary.for_request(request)
        self.timeline_frames: tuple[float, float] | None = None
        self.state = SessionState.PREPARED
        self.status = ""
        self.status_kind = "PROGRESS"
        self.progress = 0.0
        self.interval = 0.1
        self.job: DetectionJob | None = None
        self.transaction: CaptureTransaction | None = None
        self._cached_frame = cached_frame
        self._cached_frames = cached_frames
        self._immediate_status = immediate_status
        self._pose_iterator = None
        self._pose_frame: CanonicalFrame | None = None
        self._sequence_iterator = None
        self._sequence_warnings: set[str] = set()
        self._sequence_progress = 0.0
        self._cached = cached_frame is not None or cached_frames is not None
        # "cache" after detections were stored, "schedule" for a plain
        # refresh; the Blender driver turns it into a preview update.
        self.preview_request = ""

    def _request_preview(self, kind: str) -> None:
        if kind == "cache" or not self.preview_request:
            self.preview_request = kind

    def take_preview_request(self) -> str:
        """Return and clear the pending detection-preview refresh."""
        kind, self.preview_request = self.preview_request, ""
        return kind

    # ------------------------------------------------------------------
    # Preparation

    @classmethod
    def prepare(
        cls,
        scene_context: SceneContext,
        armature,
        settings: CaptureSettings,
        mode: CaptureMode,
        *,
        force_analysis: bool = False,
        output_mode: OutputMode = OutputMode.NEW_ACTION,
        frame_range: tuple[int, int] | None = None,
        start_frame: float | None = None,
        postprocess_config: PostProcessConfig | None = None,
        require_cache: bool = False,
    ) -> CaptureSession:
        """Validate everything before any Blender data changes.

        The keyword arguments serve scripted captures: ``output_mode`` picks
        where the animation goes, ``frame_range`` applies only source frames
        ``[first, end)``, ``start_frame`` replaces the scene's current frame,
        ``postprocess_config`` replaces the rig's clean-up settings for this
        capture only, and ``require_cache`` fails instead of running
        detection when no retained detections match.

        Raises an ``MtpError`` subclass with the message the user should see.
        """
        mode = CaptureMode(mode)
        output_mode = OutputMode(output_mode)
        if frame_range is not None:
            first, end = (int(value) for value in frame_range)
            if first < 0 or end <= first:
                raise InvalidRequest(
                    "frame_range must be (first, end) with 0 <= first < end",
                    parameter="frame_range",
                )
            frame_range = (first, end)
        analysis_only = mode == CaptureMode.ANALYZE
        if COORDINATOR.is_busy():
            raise OperationBusy(
                "Media to Pose is already processing media", show_status=False
            )
        if not analysis_only and not armature:
            raise InvalidRequest("Select an armature", show_status=False)
        provider = descriptor(settings.provider)
        if analysis_only and not settings.keep_detection_cache:
            raise InvalidRequest("Enable Keep Detection Cache first", show_status=False)
        if not enabled_capture_scope(settings, provider.capabilities):
            raise InvalidRequest(
                f"Enable a capture type supported by {provider.label}",
                show_status=False,
            )
        source = capture_options.resolve_media_source(settings)
        if not source.usable:
            raise MediaError(source.message)
        options = capture_options.options_from_settings(settings)
        # Sequence analysis stores raw observations. Temporal Correction owns
        # baked-video filtering; provider smoothing remains preview-only.
        if mode != CaptureMode.POSE:
            options["smoothing"] = 0.0
        notices: list[tuple[str, str]] = []
        if not analysis_only:
            prepare_limb_ownership(armature)
            mapping.validate_mapping(armature, settings=settings)
            errors = mapping_scope_issues(armature, settings)
            if errors:
                raise MappingIncomplete(errors[0])
            warning = face_output_warning(armature, settings)
            if warning:
                notices.append(("WARNING", warning))
        media_info = cls._media_info(settings, source)
        cache_media_info = MediaInfo(
            path=source.display_path or media_info.path,
            kind=media_info.kind,
            fps=media_info.fps,
            frame_count=media_info.frame_count,
            duration_seconds=media_info.duration_seconds,
            width=media_info.width,
            height=media_info.height,
        )
        selected_index = max(0, int(settings.video_frame) - 1)
        if (
            mode == CaptureMode.POSE
            and media_info.kind == MediaKind.VIDEO
            and media_info.frame_count > 0
            and selected_index >= media_info.frame_count
        ):
            raise InvalidRequest(
                f"Frame must be between 1 and {media_info.frame_count}",
                show_status=False,
            )
        if start_frame is None:
            start_frame = float(scene_context.scene.frame_current)
        if mode == CaptureMode.KEYFRAMES and media_info.kind == MediaKind.VIDEO:
            _check_rohm(armature, settings)
            start = float(start_frame)
            issue = validate_limb_switch_interval(
                armature,
                start,
                start + media_info.duration_seconds * scene_context.fps,
            )
            if issue:
                raise MappingIncomplete(issue)
        cache_key, cache_identity = capture_options.cache_identity_from_media(
            source.source_media, settings.provider, options
        )
        source_index = selected_index if media_info.kind == MediaKind.VIDEO else 0
        request = cls._build_request(
            scene_context,
            armature,
            settings,
            mode,
            media_info,
            cache_media_info,
            options,
            source_index,
            analysis_path=source.analysis_path,
            source_identity=source.source_media,
            cache_key=cache_key,
            cache_identity=cache_identity,
            start_frame=float(start_frame),
            output_mode=output_mode,
            frame_range=frame_range,
            postprocess_config=postprocess_config,
        )
        common = {"notices": tuple(notices)}
        if settings.keep_detection_cache:
            if mode in {CaptureMode.POSE, CaptureMode.KEYFRAMES}:
                cached = cache_store.load_frame(
                    cache_key, source_index if mode == CaptureMode.POSE else 0
                )
                if mode == CaptureMode.POSE and cached is not None:
                    return cls(
                        scene_context,
                        armature,
                        settings,
                        request,
                        cached_frame=cached,
                        **common,
                    )
                if mode == CaptureMode.KEYFRAMES:
                    frames = (
                        iter((cached,))
                        if media_info.kind == MediaKind.IMAGE and cached is not None
                        else cache_store.iter_frames(cache_key)
                    )
                    if frames is not None:
                        return cls(
                            scene_context,
                            armature,
                            settings,
                            request,
                            cached_frames=frames,
                            **common,
                        )
            elif not force_analysis:
                status = capture_options.status_for_settings(settings)
                if status.matches and (
                    status.complete
                    or media_info.kind == MediaKind.IMAGE
                    and status.cached_frames
                ):
                    return cls(
                        scene_context,
                        armature,
                        settings,
                        request,
                        immediate_status="Using cached analysis",
                        **common,
                    )
        if require_cache:
            raise InvalidRequest(
                "No retained detections match this media and capture settings; "
                "run a capture with the detection cache enabled first",
                parameter="media",
            )
        if not source.analysis_path:
            raise MediaError(
                "The retained detections do not contain the requested frame; "
                "locate the original media"
            )
        # Cached detections do not require provider imports. Probe
        # dependencies only when inference is actually about to start.
        if settings.provider == "MEDIAPIPE":
            from ..core.dependencies import refresh_dependency_snapshot

            refresh_dependency_snapshot()
        available, message = provider.dependency_status(options)
        if not available:
            raise ModelMissing(message)
        return cls(scene_context, armature, settings, request, **common)

    @staticmethod
    def _media_info(settings: CaptureSettings, source) -> MediaInfo:
        if source.analysis_path:
            try:
                return probe_media(source.analysis_path)
            except (MtpError, OSError) as exc:
                raise MediaError(str(exc)) from exc
        status = capture_options.status_for_settings(settings)
        if not status.matches or not status.cached_frames:
            raise MediaError(source.message)
        return cache_store.media_info_from_status(status)

    @staticmethod
    def _build_request(
        scene_context: SceneContext,
        armature,
        settings: CaptureSettings,
        mode: CaptureMode,
        media_info: MediaInfo,
        cache_media_info: MediaInfo,
        options: dict,
        source_index: int,
        *,
        analysis_path: str,
        source_identity,
        cache_key: str,
        cache_identity: dict,
        start_frame: float,
        output_mode: OutputMode = OutputMode.NEW_ACTION,
        frame_range: tuple[int, int] | None = None,
        postprocess_config: PostProcessConfig | None = None,
    ) -> CaptureRequest:
        scene = scene_context.scene
        action = (
            armature.animation_data.action
            if armature is not None and armature.animation_data
            else None
        )
        config = armature.mtp_rig_config if armature is not None else None
        mapping_entries = (
            rig_planning.snapshot_mapping_entries(mapping.mapping_by_id(config))
            if config is not None
            else {}
        )
        view_layer = scene_context.view_layer
        return CaptureRequest(
            provider_id=str(settings.provider),
            mode=mode,
            media_path=str(media_info.path),
            media_kind=str(media_info.kind),
            source_frame_index=max(0, int(source_index)),
            use_body=bool(settings.use_body),
            use_hands=bool(settings.use_hands),
            use_face=bool(settings.use_face),
            face_output=settings.face_output,
            root_motion_mode=settings.root_motion_mode,
            keep_original_root_position=bool(settings.keep_original_root_position),
            keep_detection_cache=bool(settings.keep_detection_cache),
            cache_limit_bytes=int(settings.cache_limit_mb) * 1024 * 1024,
            start_frame=float(start_frame),
            scene_name=scene.name,
            scene_pointer=scene.as_pointer(),
            source_frame_count=max(0, int(media_info.frame_count)),
            target_name=armature.name if armature is not None else "",
            target_pointer=armature.as_pointer() if armature is not None else 0,
            action_name=action.name if action is not None else "",
            object_mode=str(armature.mode) if armature is not None else "OBJECT",
            selected_object_names=tuple(
                obj.name
                for obj in view_layer.objects
                if obj.select_get(view_layer=view_layer)
            ),
            analysis_path=str(analysis_path or ""),
            source_identity=dict(source_identity or {}),
            options=options,
            mapping_entries=mapping_entries,
            postprocess_config=(
                postprocess_config
                if postprocess_config is not None
                else compile_postprocess_config(config)
                if config is not None
                else None
            ),
            media_info=media_info,
            cache_media_info=cache_media_info,
            cache_key=cache_key,
            cache_identity=cache_identity,
            output_mode=output_mode,
            frame_range=frame_range,
        )

    # ------------------------------------------------------------------
    # Driving

    @property
    def finished(self) -> bool:
        return self.state.finished

    def _update(self) -> SessionUpdate:
        return SessionUpdate(
            state=self.state,
            status=self.status,
            status_kind=self.status_kind,
            progress=self.progress,
            interval=self.interval,
        )

    def _set_status(self, message: str, kind: str = "PROGRESS") -> None:
        self.status = str(message)
        self.status_kind = kind

    def start(self) -> SessionUpdate:
        """Acquire the lease and begin detection or cached application."""
        if self._immediate_status:
            self._set_status(self._immediate_status, "INFO")
            self.state = SessionState.SUCCEEDED
            self._request_preview("schedule")
            return self._update()
        token = COORDINATOR.acquire(
            OperationKind.CACHE_APPLY if self._cached else OperationKind.CAPTURE,
            f"capture:{self.request.mode.casefold()}",
            scene_name=self.request.scene_name,
            scene_pointer=self.request.scene_pointer,
            target_name=self.request.target_name,
            target_pointer=self.request.target_pointer,
            status="Loading cached detections" if self._cached else "Starting capture",
        )
        if token is None:
            raise OperationBusy("Media to Pose is already busy")
        self.job = DetectionJob(token, self.request, cached=self._cached)
        ACTIVE_SESSIONS[token.generation] = self
        try:
            if self._cached_frame is not None:
                self.job.phase_progress(
                    "pose_application", 0, 1, "Loading cached detection"
                )
                self.progress = 55.0
                self._begin_pose(self._cached_frame, cached=True)
            elif self._cached_frames is not None:
                self._begin_sequence(self._cached_frames, cached=True)
            else:
                self._start_worker()
        except BaseException:
            self._close(SessionState.FAILED)
            raise
        return self._update()

    def _start_worker(self) -> None:
        job = self.job
        job.thread = threading.Thread(
            target=run_detection,
            args=(job, self.request),
            daemon=True,
            name="MediaToPoseInference",
        )
        COORDINATOR.attach_worker(job.token, thread=job.thread)
        self.state = SessionState.DETECTING
        self.interval = 0.1
        self._set_status(descriptor(self.request.provider_id).loading_message)
        try:
            job.thread.start()
        except RuntimeError as exc:
            raise InvalidRequest(f"Could not start pose detection: {exc}") from exc

    def cancel(self, status: str = "Cancelling") -> None:
        if self.job is not None and COORDINATOR.owns(self.job.token):
            COORDINATOR.request_cancel(self.job.token, status=status)
            self._set_status(status)

    def step(self, budget_seconds: float = 0.008) -> SessionUpdate:
        """Advance the capture by a bounded amount of main-thread work."""
        if self.finished:
            return self._update()
        job = self.job
        if job is None or not COORDINATOR.owns(job.token):
            return self._finish(cancelled=True)
        if _origin_scene(self.request) is None:
            COORDINATOR.request_cancel(
                job.token, status="Originating scene was removed"
            )
            return self._finish(cancelled=True)
        try:
            if self._pose_iterator is not None:
                return self._advance_pose()
            if self._sequence_iterator is not None:
                return self._advance_sequence(budget_seconds)
            return self._poll_worker()
        except Exception as exc:  # Session boundary: roll back and report.
            LOGGER.exception("Capture failed")
            self._set_status(str(exc), "ERROR")
            return self._finish(cancelled=True)

    def _poll_worker(self) -> SessionUpdate:
        job = self.job
        operation = COORDINATOR.snapshot()
        self._set_status(operation.status or job.status)
        self.progress = operation.progress * 100.0
        try:
            kind, payload = job.messages.get_nowait()
        except queue.Empty:
            return self._update()
        if kind == "ERROR":
            message, _details = payload  # The worker already logged it.
            self._set_status(message, "ERROR")
            return self._finish(cancelled=True)
        if kind == "CANCELLED":
            self._set_status(payload or CANCELLED_RESTORED, "INFO")
            return self._finish(cancelled=True)
        return self._accept_detection(kind, payload)

    def _accept_detection(self, kind: str, payload) -> SessionUpdate:
        request = self.request
        analysis_only = request.mode == CaptureMode.ANALYZE
        armature = _origin_armature(request)
        if not analysis_only and not armature:
            self._set_status("Target armature was removed", "ERROR")
            return self._finish(cancelled=True)
        if request.keep_detection_cache and kind in {"FRAME", "FRAMES"}:
            detected = payload if kind == "FRAME" else payload[0]
            cache_store.store_point(
                request.cache_key,
                dict(request.cache_identity),
                request.cache_media_info,
                request.source_frame_index,
                detected,
            )
            self._request_preview("cache")
        if analysis_only:
            return self._finish_analysis(kind, payload)
        self._set_status("Applying capture")
        if kind == "FRAME":
            self._begin_pose(payload)
        elif kind == "FRAMES":
            self._begin_sequence(iter(payload))
        elif kind == "CACHE":
            spool, _count = payload
            self._begin_sequence(iter(spool))
        elif kind == "CACHE_DIR":
            path, _count, _persisted = payload
            self._begin_sequence(cache_store.iter_directory(path))
        else:
            raise InvalidRequest(f"Unexpected worker result: {kind}")
        return self._update()

    def _finish_analysis(self, kind: str, payload) -> SessionUpdate:
        if kind in {"FRAME", "FRAMES"}:
            self._set_status("Image analysis cached", "INFO")
        elif kind == "CACHE_DIR":
            _path, count, persisted = payload
            self.outcome.frames = int(count)
            self.outcome.cache_persisted = bool(persisted)
            self._set_status(
                f"Cached {count} frame(s)"
                if persisted
                else (
                    f"Analyzed {count} frame(s); cache limit "
                    "exceeded, result not retained"
                ),
                "INFO" if persisted else "WARNING",
            )
        else:
            raise InvalidRequest(f"Unexpected analysis result: {kind}")
        if kind == "CACHE_DIR" and payload[2]:
            self._request_preview("cache")
        elif kind not in {"FRAME", "FRAMES"}:
            self._request_preview("schedule")
        return self._finish(cancelled=False)

    def _observe_frame(self, frame: CanonicalFrame, source_frame: int) -> None:
        timeline = self.request.start_frame
        if self.request.mode != CaptureMode.POSE:
            timeline += int(frame.timestamp_ms) * self.scene_context.fps / 1000.0
        timeline = round(float(timeline), 4)
        self.frame_confidence.append(
            {
                "source_frame": int(source_frame),
                "timeline_frame": timeline,
                **part_confidence(frame),
            }
        )
        if self.face_summary is not None:
            self.face_summary.observe(frame)
        first = self.timeline_frames[0] if self.timeline_frames else timeline
        self.timeline_frames = (first, timeline)

    def _observe_camera(self, frame: CanonicalFrame) -> None:
        if frame.camera is not None and frame.camera.valid:
            self.outcome.camera_vertical_fov = float(frame.camera.vertical_fov)
            self.outcome.camera_source = str(frame.camera.source)

    def _begin_pose(self, detected: CanonicalFrame, *, cached: bool = False) -> None:
        armature = _origin_armature(self.request)
        if _origin_scene(self.request) is None:
            raise InvalidRequest("Originating scene was removed")
        if self.transaction is None:
            self.transaction = CaptureTransaction(
                self.scene_context, armature, animation=False
            )
        self._pose_frame = detected
        self._observe_camera(detected)
        self._observe_frame(detected, self.request.source_frame_index)
        self.outcome.cached = bool(cached)
        self._pose_iterator = iter_apply_frame(
            armature,
            detected,
            self.request,
            scene_context=self.scene_context,
        )
        self.state = SessionState.APPLYING_POSE
        self.interval = 0.03
        self._set_status(
            "Loading cached detection" if cached else "Preparing detected pose"
        )
        start, _end = self.job.progress_plan.range_for("pose_application")
        self.progress = max(self.progress, start)
        self._report_pose(0.0)

    def _report_pose(self, ratio: float) -> None:
        plan = self.job.progress_plan
        COORDINATOR.update(
            self.job.token,
            status=self.status,
            phase="pose_application",
            progress=self.progress / 100.0,
            frame_current=1,
            frame_total=1,
            step_current=plan.step_number("pose_application"),
            step_total=plan.step_count,
            step_progress=ratio,
        )

    def _advance_pose(self) -> SessionUpdate:
        if self.job.cancel.is_set():
            self._set_status(CANCELLED_RESTORED, "INFO")
            return self._finish(cancelled=True)
        try:
            with self.report.active():
                progress = next(self._pose_iterator)
        except StopIteration as stopped:
            self._pose_iterator = None
            bones, shapes = stopped.value
            self.outcome.frames, self.outcome.bones, self.outcome.shapes = (
                1,
                bones,
                shapes,
            )
            armature = _origin_armature(self.request)
            target = armature.name if armature is not None else "target character"
            prefix = "Applied cached pose" if self.outcome.cached else "Pose applied"
            status = f"{prefix} to {target} · {bones} bones" + (
                f" and {shapes} shape keys" if shapes else ""
            )
            kind = "INFO"
            warnings = list(self._pose_frame.warnings)
            if warnings:
                status += "; " + "; ".join(warnings)
                kind = "WARNING"
            self.outcome.warnings = warnings
            self._set_status(status, kind)
            self._request_preview("schedule")
            return self._finish(cancelled=False)
        self._set_status(progress.phase)
        start, end = self.job.progress_plan.range_for("pose_application")
        ratio = (
            float(progress.current) / float(progress.total) if progress.total else 0.0
        )
        self.progress = max(self.progress, start + (end - start) * ratio)
        self._report_pose(ratio)
        return self._update()

    def _begin_sequence(self, frames, *, cached: bool = False) -> None:
        armature = _origin_armature(self.request)
        if _origin_scene(self.request) is None:
            raise InvalidRequest("Originating scene was removed")
        if self.transaction is None:
            self.transaction = CaptureTransaction(
                self.scene_context,
                armature,
                animation=True,
                output_mode=self.request.output_mode,
            )
        self._sequence_warnings = set()
        if self.request.frame_range is not None:
            first, end = self.request.frame_range
            frames = itertools.islice(frames, first, end)

        offset = self.report.source_frame_offset

        def observed_frames():
            for index, detected in enumerate(frames):
                self._observe_camera(detected)
                self._observe_frame(detected, offset + index)
                self._sequence_warnings.update(detected.warnings)
                yield detected
                self._sequence_warnings.update(detected.warnings)

        self._sequence_iterator = iter_apply_sequence(
            armature,
            observed_frames(),
            self.request,
            self.request.start_frame,
            scene_context=self.scene_context,
        )
        self.outcome.cached = bool(cached)
        self.state = SessionState.APPLYING_SEQUENCE
        self.interval = 0.03
        plan = self.job.progress_plan
        self._sequence_progress = plan.range_for("loading")[0]
        self.job.frame_current = 0
        self.progress = self._sequence_progress
        self._set_status("Preparing detected frames")
        COORDINATOR.update(
            self.job.token,
            status=self.status,
            phase="loading",
            progress=self._sequence_progress / 100.0,
            frame_current=0,
            frame_total=self.job.frame_total,
            step_current=plan.step_number("loading"),
            step_total=plan.step_count,
            step_progress=0.0,
        )

    def _advance_sequence(self, budget_seconds: float) -> SessionUpdate:
        if self.job.cancel.is_set():
            self._set_status(CANCELLED_RESTORED, "INFO")
            return self._finish(cancelled=True)
        deadline = time.perf_counter() + max(float(budget_seconds), 0.0)
        while True:
            try:
                with self.report.active():
                    progress = next(self._sequence_iterator)
            except StopIteration as stopped:
                self._sequence_iterator = None
                return self._complete_sequence(*stopped.value)
            display = sequence_progress_status(progress)
            self._set_status(display)
            self._sequence_progress = sequence_progress_percent(
                progress, self._sequence_progress, self.job.progress_plan
            )
            self.progress = self._sequence_progress
            self.job.sequence_progress(
                sequence_progress_group(progress.phase),
                progress.current,
                progress.total,
                display,
                self._sequence_progress,
            )
            if str(progress.phase).startswith("RoHM:"):
                # One diffusion step per tick guarantees an event-loop handoff
                # even when ONNX calls finish inside the normal budget.
                return self._update()
            if time.perf_counter() >= deadline:
                return self._update()

    def _complete_sequence(self, count, bones, shapes, warnings) -> SessionUpdate:
        self._sequence_warnings.update(warnings)
        armature = _origin_armature(self.request)
        target = armature.name if armature is not None else "target character"
        self.outcome.frames, self.outcome.bones, self.outcome.shapes = (
            count,
            bones,
            shapes,
        )
        self.outcome.warnings = sorted(self._sequence_warnings)
        status = (
            f"Video motion applied to {target} from frame "
            f"{self.request.start_frame:g} · {count} frame(s), "
            f"{bones} bone and {shapes} shape samples"
        )
        kind = "INFO"
        if self._sequence_warnings:
            status += "; " + "; ".join(sorted(self._sequence_warnings))
            kind = "WARNING"
        self._set_status(status, kind)
        self._request_preview("schedule")
        return self._finish(cancelled=False)

    # ------------------------------------------------------------------
    # Finishing

    def _finish(self, *, cancelled: bool) -> SessionUpdate:
        if self.finished:
            return self._update()
        failed = self.status_kind == "ERROR"
        state = (
            SessionState.FAILED
            if failed
            else SessionState.CANCELLED
            if cancelled
            else SessionState.SUCCEEDED
        )
        self._close(state)
        return self._update()

    def _close(self, state: SessionState) -> None:
        for name in ("_sequence_iterator", "_pose_iterator"):
            iterator = getattr(self, name)
            if iterator is not None:
                iterator.close()
                setattr(self, name, None)
        self._pose_frame = None
        self.state = state
        self.progress = 0.0
        job = self.job
        if job is None or not COORDINATOR.owns(job.token):
            return
        transaction = self.transaction
        if transaction is not None:
            if state == SessionState.SUCCEEDED:
                transaction.commit()
                action = getattr(transaction.temporary_action, "name", "")
                self.outcome.action_name = action or self.request.action_name
            else:
                transaction.rollback()
        if job.thread is not None and job.thread.is_alive():
            job.thread.join(timeout=0.1)
        terminal = {
            SessionState.SUCCEEDED: OperationState.SUCCEEDED,
            SessionState.CANCELLED: OperationState.CANCELLED,
        }.get(state, OperationState.FAILED)
        COORDINATOR.finish(job.token, terminal, self.status or str(state))
        if COORDINATOR.release(job.token):
            ACTIVE_SESSIONS.pop(job.token.generation, None)
            job.close_spool()
            if job.temp_path:
                path = Path(job.temp_path)
                try:
                    cache_store.discard_temporary(str(path))
                except (OSError, ValueError):
                    LOGGER.warning("Could not remove temporary capture data: %s", path)

    def wait(self, timeout: float = 30.0) -> bool:
        """Join the detection worker and release the lease it was holding.

        A finished session may still have a worker closing its provider; the
        lease is released only once that thread ends. Returns whether the
        session no longer holds the lease.
        """
        job = self.job
        if job is None:
            return True
        thread = job.thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(float(timeout), 0.0))
        if not self.finished:
            return False
        if COORDINATOR.owns(job.token) and not COORDINATOR.release(job.token):
            return False
        ACTIVE_SESSIONS.pop(job.token.generation, None)
        job.close_spool()
        return True

    def abort(self, status: str = CANCELLED_RESTORED) -> None:
        """Stop immediately: cancel the worker and roll back on this thread."""
        self.cancel()
        if not self.finished:
            self._set_status(status, "INFO")
            self._finish(cancelled=True)

    def shutdown(self) -> None:
        """Cancel during add-on unregistration, joining the worker briefly."""
        job = self.job
        if job is None:
            return
        COORDINATOR.request_cancel(
            job.token, status="Cancelling during add-on shutdown"
        )
        thread = job.thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=0.5)
        if thread is None or not thread.is_alive():
            self._set_status("Cancelled during add-on shutdown", "INFO")
            self._finish(cancelled=True)


def active_session() -> CaptureSession | None:
    token = COORDINATOR.active_token()
    if token is None or token.kind not in {
        OperationKind.CAPTURE,
        OperationKind.CACHE_APPLY,
    }:
        return None
    session = ACTIVE_SESSIONS.get(token.generation)
    # A finished session may still hold the lease while its worker closes
    # the provider; cancelling it then would reopen a terminal operation.
    return session if session is not None and not session.finished else None


def shutdown_sessions() -> None:
    for session in tuple(ACTIVE_SESSIONS.values()):
        session.shutdown()


__all__ = [
    "ACTIVE_SESSIONS",
    "CANCELLED_RESTORED",
    "CaptureOutcome",
    "CaptureSession",
    "SessionState",
    "SessionUpdate",
    "active_session",
    "rohm_may_run",
    "shutdown_sessions",
]
