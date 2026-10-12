"""Provider-neutral image and video metadata probing."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .dependency_bootstrap import import_runtime_dependencies
from .models import MediaInfo


def decode_image(path: str, cv2, np):
    """Decode a still image from a Unicode-safe filesystem path."""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError("Image could not be decoded")
    return image


def decode_video_frame_index_with_timestamp(path: str, source_index: int, cv2):
    """Decode an exact zero-based frame and return its source timestamp."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        source_index = max(0, int(source_index))
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, image = capture.read()
        if not ok or image is None:
            raise ValueError("Source frame could not be decoded")
        timestamp_ms = int(round(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
        if timestamp_ms <= 0 and source_index > 0:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0.0:
                timestamp_ms = int(round(source_index * 1000.0 / fps))
        return image, max(0, timestamp_ms)
    finally:
        capture.release()


def iter_video_frames(
    path: str,
    cancel_event,
    cv2,
) -> Iterable[tuple[int, int, int, object]]:
    """Yield index, monotonic timestamp, total count, and decoded BGR frame."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        index = 0
        previous_timestamp = -1
        while not cancel_event.is_set():
            ok, image = capture.read()
            if not ok:
                break
            timestamp = int(round(capture.get(cv2.CAP_PROP_POS_MSEC)))
            if timestamp <= previous_timestamp:
                timestamp = (
                    int(round(index * 1000.0 / fps))
                    if fps > 0.0
                    else previous_timestamp + 1
                )
            timestamp = max(previous_timestamp + 1, timestamp)
            previous_timestamp = timestamp
            yield index, timestamp, total, image
            index += 1
        if cancel_event.is_set():
            raise InterruptedError("Processing cancelled")
    finally:
        capture.release()


def probe_media(path: str) -> MediaInfo:
    cv2, _mp, np = import_runtime_dependencies()
    filepath = str(Path(path).expanduser().resolve())
    try:
        image = decode_image(filepath, cv2, np)
    except ValueError:
        image = None
    if image is not None:
        height, width = image.shape[:2]
        return MediaInfo(
            path=filepath,
            kind="IMAGE",
            width=width,
            height=height,
        )
    capture = cv2.VideoCapture(filepath)
    try:
        if not capture.isOpened():
            raise ValueError("Media file could not be decoded")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = count / fps if fps > 0.0 and count > 0 else 0.0
        return MediaInfo(
            path=filepath,
            kind="VIDEO",
            fps=fps,
            frame_count=count,
            duration_seconds=duration,
            width=width,
            height=height,
        )
    finally:
        capture.release()
