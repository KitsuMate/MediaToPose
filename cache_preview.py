"""Asynchronous raw-media and detected-pose preview for the Capture panel."""

from __future__ import annotations

import os
import queue
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import bpy
import bpy.utils.previews

from . import detection_cache
from .constants import HAND_LANDMARK_NAMES
from .media_io import (
    decode_image,
    decode_video_frame_index_with_timestamp,
    probe_media,
)

PREVIEW_NAME = "media_to_pose_detection_preview"
PREVIEW_FILE = Path(tempfile.gettempdir()) / (
    f"media_to_pose_preview_{os.getpid()}.png"
)
_REQUESTS: queue.Queue = queue.Queue(maxsize=1)
_RESULTS: queue.Queue = queue.Queue(maxsize=1)
_REQUEST_VERSION = 0
_DEBOUNCE_REGISTERED = False
_POLL_REGISTERED = False
_PREVIEW_KEY = ""
_PREVIEW_FRAME = -1
_PREVIEW_TIMESTAMP_MS = 0
_PREVIEW_ERROR = ""
_PREVIEW_LOADING = False
_PREVIEW_HAS_OVERLAY = False
_PREVIEW_MEDIA_KIND = ""
_PREVIEW_FRAME_COUNT = 0
_PREVIEW_COLLECTION = None
_PREVIEW_ITEM = None
_WORKER_THREAD: threading.Thread | None = None
_STOP_WORKER = threading.Event()


@dataclass(frozen=True)
class _PreviewRequest:
    generation: int
    key: str
    media_path: str
    source_index: int
    mirror: bool


@dataclass(frozen=True)
class _PreviewResult:
    generation: int
    key: str
    media_path: str
    source_index: int = -1
    timestamp_ms: int = 0
    encoded: bytes | None = None
    error: str = ""
    has_overlay: bool = False
    media_kind: str = ""
    frame_count: int = 0


BODY_CONNECTIONS = (
    ("LEFT_EAR", "LEFT_EYE"),
    ("LEFT_EYE", "NOSE"),
    ("NOSE", "RIGHT_EYE"),
    ("RIGHT_EYE", "RIGHT_EAR"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
)

HAND_CONNECTIONS = (
    ("WRIST", "THUMB_CMC"),
    ("THUMB_CMC", "THUMB_MCP"),
    ("THUMB_MCP", "THUMB_IP"),
    ("THUMB_IP", "THUMB_TIP"),
    ("WRIST", "INDEX_FINGER_MCP"),
    ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
    ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"),
    ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    ("WRIST", "MIDDLE_FINGER_MCP"),
    ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
    ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"),
    ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    ("WRIST", "RING_FINGER_MCP"),
    ("RING_FINGER_MCP", "RING_FINGER_PIP"),
    ("RING_FINGER_PIP", "RING_FINGER_DIP"),
    ("RING_FINGER_DIP", "RING_FINGER_TIP"),
    ("WRIST", "PINKY_MCP"),
    ("PINKY_MCP", "PINKY_PIP"),
    ("PINKY_PIP", "PINKY_DIP"),
    ("PINKY_DIP", "PINKY_TIP"),
    ("INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP"),
    ("MIDDLE_FINGER_MCP", "RING_FINGER_MCP"),
    ("RING_FINGER_MCP", "PINKY_MCP"),
)

# MediaPipe Face Mesh contours. Drawing a small subset keeps the overlay
# readable in a narrow sidebar while the cached payload remains 2D-only.
FACE_CONTOURS = (
    (
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
        10,
    ),
    (33, 160, 158, 133, 153, 144, 33),
    (362, 385, 387, 263, 373, 380, 362),
    (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291),
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291),
)


def _point(landmark, width: int, height: int) -> tuple[int, int]:
    return (
        int(round(float(landmark.image[0]) * width)),
        int(round(float(landmark.image[1]) * height)),
    )


def _put_latest(target: queue.Queue, value) -> None:
    try:
        target.put_nowait(value)
    except queue.Full:
        with suppress(queue.Empty):
            target.get_nowait()
        with suppress(queue.Full):
            target.put_nowait(value)


def _render_overlay(request: _PreviewRequest) -> _PreviewResult:
    source_index = -1
    timestamp_ms = 0
    media_kind = ""
    frame_count = 0
    try:
        from .dependency_bootstrap import import_runtime_dependencies

        cv2, _mp, np = import_runtime_dependencies()
        media_info = probe_media(request.media_path)
        media_kind = str(media_info.kind)
        frame_count = int(media_info.frame_count)
        if media_info.kind == "IMAGE":
            source_index = 0
        else:
            source_index = max(0, int(request.source_index))
            if frame_count > 0:
                source_index = min(source_index, frame_count - 1)
        frame = (
            detection_cache.load_frame(request.key, source_index)
            if request.key
            else None
        )
        if media_info.kind == "IMAGE":
            image = decode_image(request.media_path, cv2, np)
        else:
            image, timestamp_ms = decode_video_frame_index_with_timestamp(
                request.media_path,
                source_index,
                cv2,
            )
        if frame is not None:
            timestamp_ms = int(frame.timestamp_ms)
        if request.mirror:
            image = cv2.flip(image, 1)
        height, width = image.shape[:2]

        def draw_connection(points, first, second, color, thickness):
            a = points.get(first)
            b = points.get(second)
            if not a or not b or not a.valid or not b.valid:
                return
            cv2.line(
                image,
                _point(a, width, height),
                _point(b, width, height),
                color,
                thickness,
                cv2.LINE_AA,
            )

        if frame is not None:
            for first, second in BODY_CONNECTIONS:
                draw_connection(frame.body, first, second, (80, 240, 80), 3)
            for landmark in frame.body.values():
                if landmark and landmark.valid:
                    cv2.circle(
                        image,
                        _point(landmark, width, height),
                        3,
                        (40, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
            for points in frame.hands.values():
                for first, second in HAND_CONNECTIONS:
                    draw_connection(
                        points,
                        first,
                        second,
                        (255, 180, 40),
                        2,
                    )
                for name in HAND_LANDMARK_NAMES:
                    landmark = points.get(name)
                    if landmark and landmark.valid:
                        cv2.circle(
                            image,
                            _point(landmark, width, height),
                            2,
                            (255, 220, 80),
                            -1,
                            cv2.LINE_AA,
                        )
            face = frame.face_overlay
            if face:
                for contour in FACE_CONTOURS:
                    points = [
                        (
                            int(round(face[index][0] * width)),
                            int(round(face[index][1] * height)),
                        )
                        for index in contour
                        if index < len(face)
                    ]
                    if len(points) >= 2:
                        cv2.polylines(
                            image,
                            [np.asarray(points, dtype=np.int32)],
                            False,
                            (80, 120, 255),
                            1,
                            cv2.LINE_AA,
                        )

        scale = min(1.0, 512.0 / max(width, height))
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        if (target_width, target_height) != (width, height):
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Preview image could not be encoded")
        return _PreviewResult(
            generation=request.generation,
            key=request.key,
            media_path=request.media_path,
            source_index=source_index,
            timestamp_ms=(
                int(frame.timestamp_ms) if frame is not None else timestamp_ms
            ),
            encoded=bytes(encoded),
            has_overlay=frame is not None,
            media_kind=media_kind,
            frame_count=frame_count,
        )
    except Exception as exc:
        return _PreviewResult(
            generation=request.generation,
            key=request.key,
            media_path=request.media_path,
            source_index=source_index,
            timestamp_ms=timestamp_ms,
            error=str(exc),
            media_kind=media_kind,
            frame_count=frame_count,
        )


def _worker_loop() -> None:
    while not _STOP_WORKER.is_set():
        try:
            request = _REQUESTS.get(timeout=0.2)
        except queue.Empty:
            continue
        if request is None:
            return
        result = _render_overlay(request)
        if not _STOP_WORKER.is_set():
            _put_latest(_RESULTS, result)


def _ensure_worker() -> None:
    global _WORKER_THREAD
    if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
        return
    _STOP_WORKER.clear()
    _WORKER_THREAD = threading.Thread(
        target=_worker_loop,
        daemon=True,
        name="MediaToPoseCachePreview",
    )
    _WORKER_THREAD.start()


def schedule_refresh() -> None:
    global _REQUEST_VERSION, _DEBOUNCE_REGISTERED, _PREVIEW_LOADING
    detection_cache.invalidate_ui_state()
    _REQUEST_VERSION += 1
    _PREVIEW_LOADING = True
    if _DEBOUNCE_REGISTERED:
        return
    _DEBOUNCE_REGISTERED = True
    bpy.app.driver_namespace["_media_to_pose_cache_preview_begin_timer"] = (
        _begin_refresh
    )
    bpy.app.timers.register(_begin_refresh, first_interval=0.12)


def refresh_after_cache_update() -> None:
    """Discard an icon backed by the previous cache generation and rebuild it."""
    clear_preview()
    schedule_refresh()


def _begin_refresh():
    global _DEBOUNCE_REGISTERED, _POLL_REGISTERED, _PREVIEW_LOADING
    _DEBOUNCE_REGISTERED = False
    bpy.app.driver_namespace.pop(
        "_media_to_pose_cache_preview_begin_timer",
        None,
    )
    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "mtp_settings", None) if scene else None
    if not settings:
        _PREVIEW_LOADING = False
        return
    source, status = detection_cache.ui_state_for_settings(settings)
    if not source.analysis_path:
        clear_preview()
        _PREVIEW_LOADING = False
        return
    key = status.key if status.matches and status.cached_frames else ""
    request = _PreviewRequest(
        generation=_REQUEST_VERSION,
        key=key,
        media_path=source.analysis_path,
        source_index=max(0, int(settings.video_frame) - 1),
        mirror=bool(settings.mirror_input),
    )
    _ensure_worker()
    _put_latest(_REQUESTS, request)
    if not _POLL_REGISTERED:
        _POLL_REGISTERED = True
        bpy.app.driver_namespace["_media_to_pose_cache_preview_poll_timer"] = (
            _poll_results
        )
        bpy.app.timers.register(_poll_results, first_interval=0.05)
    return


def _poll_results():
    global _POLL_REGISTERED, _PREVIEW_LOADING
    global _PREVIEW_KEY, _PREVIEW_FRAME, _PREVIEW_TIMESTAMP_MS
    global _PREVIEW_ERROR, _PREVIEW_COLLECTION, _PREVIEW_ITEM
    global _PREVIEW_HAS_OVERLAY, _PREVIEW_MEDIA_KIND
    global _PREVIEW_FRAME_COUNT
    latest = None
    while True:
        try:
            latest = _RESULTS.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return 0.05
    if latest.generation != _REQUEST_VERSION:
        if _PREVIEW_LOADING:
            return 0.05
        _POLL_REGISTERED = False
        return None
    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "mtp_settings", None) if scene else None
    if settings is None:
        _POLL_REGISTERED = False
        _PREVIEW_LOADING = False
        return None
    source = detection_cache.resolve_media_source(settings)
    if source.analysis_path != latest.media_path:
        # The result is terminal for its request, but the user selected new
        # media before it arrived. Stop this poll timer and enqueue a fresh
        # generation instead of polling a stale result forever.
        _POLL_REGISTERED = False
        _PREVIEW_LOADING = False
        bpy.app.driver_namespace.pop(
            "_media_to_pose_cache_preview_poll_timer",
            None,
        )
        schedule_refresh()
        return None
    _POLL_REGISTERED = False
    _PREVIEW_LOADING = False
    bpy.app.driver_namespace.pop(
        "_media_to_pose_cache_preview_poll_timer",
        None,
    )
    _PREVIEW_ERROR = latest.error
    _PREVIEW_KEY = latest.key
    _PREVIEW_FRAME = latest.source_index
    _PREVIEW_TIMESTAMP_MS = latest.timestamp_ms
    _PREVIEW_HAS_OVERLAY = bool(latest.has_overlay)
    _PREVIEW_MEDIA_KIND = str(latest.media_kind)
    _PREVIEW_FRAME_COUNT = int(latest.frame_count)
    if latest.error or latest.encoded is None:
        return None
    PREVIEW_FILE.write_bytes(latest.encoded)
    if _PREVIEW_COLLECTION is None:
        _PREVIEW_COLLECTION = bpy.utils.previews.new()
    elif PREVIEW_NAME in _PREVIEW_COLLECTION:
        del _PREVIEW_COLLECTION[PREVIEW_NAME]
    _PREVIEW_ITEM = _PREVIEW_COLLECTION.load(
        PREVIEW_NAME,
        str(PREVIEW_FILE),
        "IMAGE",
        force_reload=True,
    )
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    return None


def get_preview_icon(key: str | None = None):
    if key is not None and key != _PREVIEW_KEY:
        return None
    return _PREVIEW_ITEM


def preview_state(key: str | None = None) -> tuple[int, int, str]:
    if key is not None and key != _PREVIEW_KEY:
        return -1, 0, ""
    return _PREVIEW_FRAME, _PREVIEW_TIMESTAMP_MS, _PREVIEW_ERROR


def preview_details() -> dict:
    return {
        "loading": _PREVIEW_LOADING,
        "has_overlay": _PREVIEW_HAS_OVERLAY,
        "media_kind": _PREVIEW_MEDIA_KIND,
        "frame_count": _PREVIEW_FRAME_COUNT,
        "source_index": _PREVIEW_FRAME,
        "timestamp_ms": _PREVIEW_TIMESTAMP_MS,
        "error": _PREVIEW_ERROR,
    }


def clear_preview() -> None:
    global _REQUEST_VERSION, _PREVIEW_KEY, _PREVIEW_FRAME
    global _PREVIEW_TIMESTAMP_MS, _PREVIEW_ERROR
    global _PREVIEW_COLLECTION, _PREVIEW_ITEM, _PREVIEW_LOADING
    global _PREVIEW_HAS_OVERLAY, _PREVIEW_MEDIA_KIND
    global _PREVIEW_FRAME_COUNT
    _REQUEST_VERSION += 1
    if _PREVIEW_COLLECTION is not None:
        bpy.utils.previews.remove(_PREVIEW_COLLECTION)
    _PREVIEW_COLLECTION = None
    _PREVIEW_ITEM = None
    with suppress(OSError):
        PREVIEW_FILE.unlink()
    _PREVIEW_KEY = ""
    _PREVIEW_FRAME = -1
    _PREVIEW_TIMESTAMP_MS = 0
    _PREVIEW_ERROR = ""
    _PREVIEW_LOADING = False
    _PREVIEW_HAS_OVERLAY = False
    _PREVIEW_MEDIA_KIND = ""
    _PREVIEW_FRAME_COUNT = 0


def unregister() -> None:
    global _DEBOUNCE_REGISTERED, _POLL_REGISTERED, _WORKER_THREAD
    callbacks = {
        _begin_refresh,
        _poll_results,
        bpy.app.driver_namespace.pop(
            "_media_to_pose_cache_preview_begin_timer",
            None,
        ),
        bpy.app.driver_namespace.pop(
            "_media_to_pose_cache_preview_poll_timer",
            None,
        ),
    }
    for callback in callbacks:
        if callback is None:
            continue
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    _DEBOUNCE_REGISTERED = False
    _POLL_REGISTERED = False
    _STOP_WORKER.set()
    _put_latest(_REQUESTS, None)
    if _WORKER_THREAD is not None and _WORKER_THREAD is not threading.current_thread():
        _WORKER_THREAD.join(timeout=1.0)
    _WORKER_THREAD = None
    for pending in (_REQUESTS, _RESULTS):
        while True:
            try:
                pending.get_nowait()
            except queue.Empty:
                break
    clear_preview()
