"""Resolve one camera calibration before provider inference begins."""

from __future__ import annotations

from collections.abc import Mapping

from . import anycalib_backend, camera_calibration
from .core.errors import Cancelled, MediaError
from .dependency_bootstrap import import_runtime_dependencies


def _crop_back(value, left: int, top: int, width: int, height: int):
    return camera_calibration.intrinsics(
        width,
        height,
        value.fx,
        value.fy,
        value.cx + left,
        value.cy + top,
        confidence=value.confidence,
        source="ANYCALIB",
        bundle_revision=value.bundle_revision,
        precision=value.precision,
    )


def _calibrate_image_views(session, rgb, cv2, np):
    height, width = rgb.shape[:2]
    estimates = [
        session.calibrate(
            rgb,
            cv2,
            np,
            bundle_revision=str(anycalib_backend.MODEL_SPEC["revision"]),
        )
    ]
    flipped = session.calibrate(
        np.ascontiguousarray(rgb[:, ::-1]),
        cv2,
        np,
        bundle_revision=str(anycalib_backend.MODEL_SPEC["revision"]),
    )
    estimates.append(camera_calibration.mirrored(flipped))
    crop_width = max(14, int(width * 0.9))
    crop_height = max(14, int(height * 0.9))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    cropped = np.ascontiguousarray(
        rgb[top : top + crop_height, left : left + crop_width]
    )
    crop_estimate = session.calibrate(
        cropped,
        cv2,
        np,
        bundle_revision=str(anycalib_backend.MODEL_SPEC["revision"]),
    )
    estimates.append(_crop_back(crop_estimate, left, top, width, height))
    return camera_calibration.image_consensus(estimates)


def _read_video_samples(path: str, cv2):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise MediaError(f"Could not open video for camera calibration: {path}")
    try:
        count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
            index = min(count - 1, max(0, round((count - 1) * fraction)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if ok and bgr is not None:
                yield bgr
    finally:
        capture.release()


def resolve(
    path: str,
    media_kind: str,
    width: int,
    height: int,
    options: Mapping[str, object],
    *,
    cancel_event=None,
    progress=None,
):
    """Return calibrated, manual, or fallback intrinsics without blocking capture."""

    mode = str(options.get("perspective_correction", "AUTO")).upper()
    manual_enabled = bool(options.get("manual_fov_override", False))
    manual_fov = float(options.get("manual_vertical_fov", 45.0))
    if mode == "OFF":
        return camera_calibration.fallback(width, height)
    if manual_enabled:
        return camera_calibration.manual(width, height, manual_fov)
    if not bool(options.get("anycalib_installed", False)):
        return camera_calibration.fallback(width, height)
    try:
        cv2, _mp, np = import_runtime_dependencies()
        session = anycalib_backend.ensure_model_loaded(
            options.get("anycalib_device", "AUTO"),
            str(options.get("anycalib_bundle_directory", "")),
        )
        estimates = []
        if str(media_kind).upper() == "IMAGE":
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise MediaError(
                    f"Could not decode image for camera calibration: {path}"
                )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = _calibrate_image_views(session, rgb, cv2, np)
        else:
            samples = tuple(_read_video_samples(path, cv2))
            for index, bgr in enumerate(samples, 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled("Camera calibration cancelled")
                if progress is not None:
                    progress(index, max(len(samples), 1), "Calibrating video camera")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                try:
                    estimates.append(
                        session.calibrate(
                            rgb,
                            cv2,
                            np,
                            bundle_revision=str(
                                anycalib_backend.MODEL_SPEC["revision"]
                            ),
                        )
                    )
                except (RuntimeError, ValueError):
                    continue
            result = camera_calibration.consensus(estimates, minimum=3)
        if result is None:
            return camera_calibration.fallback(width, height)
        if bool(options.get("mirror", False)):
            result = camera_calibration.mirrored(result)
        return camera_calibration.scale_to(result, width, height)
    except InterruptedError:
        raise
    except Exception as exc:
        print(f"[Media to Pose] AnyCalib fallback: {type(exc).__name__}: {exc}")
        return camera_calibration.fallback(width, height)
