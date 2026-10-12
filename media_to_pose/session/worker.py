"""Background detection: run a provider over the media and hand frames back.

The worker thread sees only ``core`` types: the frozen ``CaptureRequest`` in
and ``CanonicalFrame`` objects out, through a message queue. Blender data is
never touched here.
"""

from __future__ import annotations

import queue
import threading
import traceback

from ..core.cache.stream import FrameSpool
from ..core.errors import Cancelled
from ..core.log import get_logger
from ..core.requests import CaptureMode, CaptureRequest, MediaKind
from ..operation import COORDINATOR, OperationToken
from ..providers import descriptor
from .progress import capture_progress_plan

LOGGER = get_logger(__name__)

# Calibration policy belongs to the capture pipeline, not to provider
# constructors. Providers receive only the resolved full-image camera
# observation; cache and model-selection metadata stop here.
_CALIBRATION_CONTROLS = (
    "perspective_correction",
    "manual_fov_override",
    "manual_vertical_fov",
    "anycalib_device",
    "camera_bundle_revision",
    "camera_precision",
    "anycalib_installed",
    "anycalib_bundle_directory",
)


class DetectionJob:
    """Progress and results of one detection pass, shared with the main thread."""

    def __init__(self, token: OperationToken, request: CaptureRequest, *, cached: bool):
        self.token = token
        self.request = request
        self.cancel = token.cancel_event
        self.messages: queue.Queue = queue.Queue()
        self.current = 0
        self.total = 0
        self.status = "Starting"
        self.overall_progress = 0.0
        self.thread: threading.Thread | None = None
        self.temp_path = ""
        self.spool: FrameSpool | None = None
        self.progress_plan = capture_progress_plan(request, cached=cached)
        self.frame_total = (
            1
            if request.mode == CaptureMode.POSE or request.media_kind == MediaKind.IMAGE
            else max(int(request.source_frame_count), 1)
        )
        self.frame_current = 0
        COORDINATOR.update(
            self.token,
            frame_current=0,
            frame_total=self.frame_total,
            step_current=0,
            step_total=self.progress_plan.step_count,
        )

    def phase_progress(self, group, current, total, status) -> None:
        start, end = self.progress_plan.range_for(str(group))
        ratio = float(current) / float(total) if total else 0.0
        mapped = (start + (end - start) * min(max(ratio, 0.0), 1.0)) / 100.0
        self.overall_progress = max(self.overall_progress, mapped)
        counters = {
            "step_current": self.progress_plan.step_number(str(group)),
            "step_total": self.progress_plan.step_count,
        }
        if str(group) == "detection":
            if self.request.mode != CaptureMode.POSE and int(total) > 0:
                self.frame_total = max(self.frame_total, int(total))
            self.frame_current = min(max(int(current), 0), self.frame_total)
            counters.update(
                frame_current=self.frame_current,
                frame_total=self.frame_total,
            )
        if not COORDINATOR.update(
            self.token,
            status=status,
            phase=str(group),
            progress=self.overall_progress,
            step_progress=ratio,
            **counters,
        ):
            return
        self.current = current
        self.total = total
        self.status = status

    def progress(self, current, total, status) -> None:
        self.phase_progress("detection", current, total, status)

    def sequence_progress(self, group, current, total, status, overall) -> None:
        ratio = float(current) / float(total) if total else 0.0
        start, end = self.progress_plan.range_for(str(group))
        step_ratio = (
            min(max((float(overall) - start) / (end - start), 0.0), 1.0)
            if end > start
            else 1.0
        )
        self.frame_current = min(
            max(int(round(self.frame_total * min(max(ratio, 0.0), 1.0))), 0),
            self.frame_total,
        )
        COORDINATOR.update(
            self.token,
            status=status,
            phase=str(group),
            progress=float(overall) / 100.0,
            frame_current=self.frame_current,
            frame_total=self.frame_total,
            step_current=self.progress_plan.step_number(str(group)),
            step_total=self.progress_plan.step_count,
            step_progress=step_ratio,
        )

    def close_spool(self) -> None:
        if self.spool is not None:
            self.spool.close()
            self.spool = None


def _phase_progress(job, group, current, total, status) -> None:
    reporter = getattr(job, "phase_progress", None)
    if reporter is None:
        job.progress(current, total, status)
        return
    reporter(group, current, total, status)


def _cancelled(job) -> None:
    if job.cancel.is_set():
        raise Cancelled("Processing cancelled")


def _calibrated_options(job, request: CaptureRequest) -> dict:
    from .. import camera_calibration, camera_calibration_pipeline

    options = dict(request.options)
    media = request.media_info
    width = int(getattr(media, "width", 0))
    height = int(getattr(media, "height", 0))
    if width > 0 and height > 0:
        from ..backends.anycalib import backend as anycalib_backend

        camera = camera_calibration_pipeline.resolve(
            media.path,
            media.kind,
            width,
            height,
            options,
            load_estimator=anycalib_backend.ensure_model_loaded,
            bundle_revision=str(anycalib_backend.MODEL_SPEC["revision"]),
            cancel_event=job.cancel,
            progress=lambda current, total, message: _phase_progress(
                job, "calibration", current, total, message
            ),
        )
        options["camera_intrinsics"] = camera_calibration.to_mapping(camera)
    _phase_progress(job, "calibration", 1, 1, "Camera calibration ready")
    for control in _CALIBRATION_CONTROLS:
        options.pop(control, None)
    return options


def _detect_single(job, request: CaptureRequest, provider) -> None:
    media = request.media_info
    if request.mode == CaptureMode.POSE:
        job.progress(
            0,
            1,
            "Detecting image"
            if media.kind == MediaKind.IMAGE
            else "Detecting video frame",
        )
    if media.kind == MediaKind.IMAGE:
        detected = provider.detect_image(media.path)
    else:
        detected = provider.detect_video_frame_index(
            media.path, request.source_frame_index
        )
    _cancelled(job)
    job.progress(1, 1, "Detection complete")
    job.messages.put(
        (
            "FRAME" if request.mode == CaptureMode.POSE else "FRAMES",
            detected if request.mode == CaptureMode.POSE else [detected],
        )
    )


def _detect_sequence(job, request: CaptureRequest, provider) -> None:
    from .. import detection_cache

    media = request.media_info
    if request.keep_detection_cache:
        writer = detection_cache.SequenceWriter(
            request.cache_key,
            dict(request.cache_identity),
            request.cache_media_info,
            request.cache_limit_bytes,
        )
        try:
            job.temp_path = str(writer.directory)
            count = 0
            for detected in provider.iter_video(media.path, job.cancel, job.progress):
                writer.add(detected)
                count += 1
            job.progress(1, 1, "Detection complete")
            directory, persisted = writer.finish()
        except BaseException:
            writer.discard()
            raise
        job.temp_path = "" if persisted else str(directory)
        job.messages.put(("CACHE_DIR", (str(directory), count, persisted)))
        return
    spool = FrameSpool(prefix="media_to_pose_capture_")
    job.spool = spool
    for detected in provider.iter_video(media.path, job.cancel, job.progress):
        spool.append(detected)
    spool.flush()
    job.progress(1, 1, "Detection complete")
    job.messages.put(("CACHE", (spool, len(spool))))


def run_detection(job, request: CaptureRequest) -> None:
    """Worker thread entry point: detect, then post exactly one result message."""
    provider = None
    try:
        options = _calibrated_options(job, request)
        provider_class = descriptor(request.provider_id).provider_class
        if request.mode == CaptureMode.POSE:
            job.progress(0, 1, "Loading capture provider")
        provider = provider_class(
            video_mode=request.media_kind == MediaKind.VIDEO, **options
        )
        _cancelled(job)
        if request.mode == CaptureMode.POSE or request.media_kind == MediaKind.IMAGE:
            _detect_single(job, request, provider)
        else:
            _detect_sequence(job, request, provider)
    except InterruptedError as exc:
        if hasattr(job, "close_spool"):
            job.close_spool()
        job.messages.put(("CANCELLED", str(exc)))
    except Exception as exc:  # Worker thread entry point.
        LOGGER.exception("Capture worker failed")
        if hasattr(job, "close_spool"):
            job.close_spool()
        job.messages.put(("ERROR", (str(exc), traceback.format_exc())))
    finally:
        if provider is not None:
            try:
                provider.close()
            except Exception:  # Provider cleanup must not mask the result.
                LOGGER.exception("Capture-provider cleanup failed")


__all__ = ["DetectionJob", "run_detection"]
