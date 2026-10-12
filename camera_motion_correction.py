"""Model-free camera-motion estimation for baked video correction.

The module deliberately owns no Blender data.  It estimates a timestamped
background track from RGB frames and converts that track into source-camera X
offsets.  The retargeter applies those offsets to its already-solved root.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

TRACK_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "tier1-camera-motion-v1"
MAX_LONG_EDGE = 960
MAX_FEATURES = 900
MIN_TRACKS = 40
MIN_INLIERS = 24
MIN_INLIER_RATIO = 0.55
MIN_GRID_CELLS = 4
MAX_MEDIAN_RESIDUAL = 1.5
MAX_INTERPOLATION_GAP_SECONDS = 0.5
SMOOTHING_RADIUS_SECONDS = 0.075


@dataclass(frozen=True)
class CameraMotionInput:
    timestamp_ms: int
    root_translation: tuple[float, float, float] | None = None
    focal_length: float = 0.0
    person_points: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class CameraTrackFrame:
    timestamp_ms: int
    translation_x: float = 0.0
    translation_y: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    confidence: float = 0.0
    tracks: int = 0
    inliers: int = 0
    residual: float = math.inf
    valid: bool = False
    cut: bool = False


@dataclass(frozen=True)
class CameraMotionTrack:
    width: int
    height: int
    frames: tuple[CameraTrackFrame, ...]


@dataclass(frozen=True)
class CameraCorrectionResult:
    offsets_x: tuple[float, ...]
    corrected_intervals: int
    corrected_frames: int
    tracked_frames: int
    rejected_frames: int
    median_inliers: float
    warning: str = ""


def input_from_frame(frame) -> CameraMotionInput:
    """Build compact provider-neutral tracking input from a canonical frame."""
    points = []
    groups = [frame.body, frame.auxiliary]
    groups.extend(frame.hands.values())
    for group in groups:
        for point in group.values():
            if not point.valid or point.confidence <= 0.0:
                continue
            x, y = (float(point.image[0]), float(point.image[1]))
            if (
                math.isfinite(x)
                and math.isfinite(y)
                and 0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
            ):
                points.append((x, y))
    for point in frame.face_overlay:
        x, y = float(point[0]), float(point[1])
        if (
            math.isfinite(x)
            and math.isfinite(y)
            and 0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
        ):
            points.append((x, y))
    observation = frame.root_motion
    translation = None
    focal = 0.0
    if observation is not None and observation.valid:
        values = tuple(float(item) for item in observation.translation)
        if len(values) == 3 and all(math.isfinite(item) for item in values):
            translation = values
        focal = float(observation.focal_length)
        if not math.isfinite(focal) or focal <= 0.0:
            focal = 0.0
    return CameraMotionInput(
        timestamp_ms=int(frame.timestamp_ms),
        root_translation=translation,
        focal_length=focal,
        person_points=tuple(points),
    )


def timestamp_digest(inputs: Sequence[CameraMotionInput]) -> str:
    digest = hashlib.sha256()
    for item in inputs:
        digest.update(int(item.timestamp_ms).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def cache_key(source_identity: dict, inputs: Sequence[CameraMotionInput]) -> str:
    payload = {
        "algorithm": ALGORITHM_VERSION,
        "source": {
            "path": str(source_identity.get("path", "")),
            "size": int(source_identity.get("size", 0)),
            "mtime_ns": int(source_identity.get("mtime_ns", 0)),
        },
        "timestamps": timestamp_digest(inputs),
        "count": len(inputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _payload_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_sidecar(directory: Path, expected_key: str, np) -> CameraMotionTrack | None:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        payload = directory / str(manifest["payload"])
        if (
            manifest.get("schema_version") != TRACK_SCHEMA_VERSION
            or manifest.get("algorithm") != ALGORITHM_VERSION
            or manifest.get("key") != expected_key
            or not payload.is_file()
            or _payload_digest(payload) != manifest.get("sha256")
        ):
            return None
        with np.load(str(payload), allow_pickle=False) as archive:
            timestamps = archive["timestamps"]
            values = archive["values"]
        if values.ndim != 2 or values.shape[1] != 11 or len(values) != len(timestamps):
            return None
        frames = tuple(
            CameraTrackFrame(
                timestamp_ms=int(timestamp),
                translation_x=float(row[0]),
                translation_y=float(row[1]),
                rotation=float(row[2]),
                scale=float(row[3]),
                confidence=float(row[4]),
                tracks=int(round(float(row[5]))),
                inliers=int(round(float(row[6]))),
                residual=float(row[7]),
                valid=bool(round(float(row[8]))),
                cut=bool(round(float(row[9]))),
            )
            for timestamp, row in zip(timestamps, values, strict=True)
        )
        if len(frames) != int(manifest.get("frame_count", -1)):
            return None
        return CameraMotionTrack(
            width=int(manifest["width"]),
            height=int(manifest["height"]),
            frames=frames,
        )
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return None


def store_sidecar(directory: Path, key: str, track: CameraMotionTrack, np) -> None:
    """Atomically replace the active immutable sidecar."""
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".camera-motion-", dir=str(parent)))
    previous = parent / ".camera-motion-previous"
    try:
        payload = staging / "track.npz"
        timestamps = np.asarray(
            [frame.timestamp_ms for frame in track.frames], dtype=np.int64
        )
        values = np.asarray(
            [
                (
                    frame.translation_x,
                    frame.translation_y,
                    frame.rotation,
                    frame.scale,
                    frame.confidence,
                    frame.tracks,
                    frame.inliers,
                    frame.residual,
                    float(frame.valid),
                    float(frame.cut),
                    0.0,
                )
                for frame in track.frames
            ],
            dtype=np.float64,
        )
        with payload.open("wb") as stream:
            np.savez_compressed(stream, timestamps=timestamps, values=values)
        manifest = {
            "schema_version": TRACK_SCHEMA_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "key": key,
            "width": int(track.width),
            "height": int(track.height),
            "frame_count": len(track.frames),
            "payload": payload.name,
            "sha256": _payload_digest(payload),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf8"
        )
        if previous.exists():
            _remove_tree(previous)
        if directory.exists():
            Path(directory).replace(previous)
        try:
            Path(staging).replace(directory)
        except OSError:
            if previous.exists() and not directory.exists():
                Path(previous).replace(directory)
            raise
        if previous.exists():
            _remove_tree(previous)
    except BaseException:
        if staging.exists():
            _remove_tree(staging)
        raise


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _gray_working(image, cv2):
    height, width = image.shape[:2]
    scale = min(1.0, MAX_LONG_EDGE / max(width, height, 1))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale


def _person_mask(shape, points, cv2, np, previous_bounds=None):
    height, width = shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    usable = [
        (min(max(x, 0.0), 1.0) * width, min(max(y, 0.0), 1.0) * height)
        for x, y in points
    ]
    bounds = previous_bounds
    if usable:
        xs = [point[0] for point in usable]
        ys = [point[1] for point in usable]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        padding_x = max(16.0, (x1 - x0) * 0.20)
        padding_y = max(16.0, (y1 - y0) * 0.20)
        bounds = (
            int(max(0, math.floor(x0 - padding_x))),
            int(max(0, math.floor(y0 - padding_y))),
            int(min(width - 1, math.ceil(x1 + padding_x))),
            int(min(height - 1, math.ceil(y1 + padding_y))),
        )
    if bounds is not None:
        cv2.rectangle(mask, bounds[:2], bounds[2:], 0, thickness=-1)
    return mask, bounds


def _select_grid_features(gray, mask, cv2, np):
    height, width = gray.shape[:2]
    per_cell = max(1, MAX_FEATURES // 9)
    selected = []
    for row in range(3):
        for column in range(3):
            y0, y1 = height * row // 3, height * (row + 1) // 3
            x0, x1 = width * column // 3, width * (column + 1) // 3
            points = cv2.goodFeaturesToTrack(
                gray[y0:y1, x0:x1],
                maxCorners=per_cell,
                qualityLevel=0.01,
                minDistance=8,
                mask=mask[y0:y1, x0:x1],
                blockSize=3,
            )
            if points is None:
                continue
            points = points.reshape(-1, 2)
            points[:, 0] += x0
            points[:, 1] += y0
            selected.append(points)
    if not selected:
        return None
    return np.concatenate(selected, axis=0).astype(np.float32).reshape(-1, 1, 2)


def _grid_coverage(points, width, height) -> int:
    cells = {
        (
            min(2, max(0, int(point[0] * 3.0 / max(width, 1)))),
            min(2, max(0, int(point[1] * 3.0 / max(height, 1)))),
        )
        for point in points
    }
    return len(cells)


def _estimate_step(previous, current, previous_mask, current_mask, cv2, np):
    features = _select_grid_features(previous, previous_mask, cv2, np)
    if features is None or len(features) < MIN_TRACKS:
        return None
    next_points, status_forward, _error = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status_forward is None:
        return None
    back_points, status_back, _back_error = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        next_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if back_points is None or status_back is None:
        return None
    source = features.reshape(-1, 2)
    target = next_points.reshape(-1, 2)
    backward = back_points.reshape(-1, 2)
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_back.reshape(-1).astype(bool)
        & (np.linalg.norm(source - backward, axis=1) <= 1.0)
    )
    height, width = current.shape[:2]
    xi = np.clip(np.rint(target[:, 0]).astype(int), 0, width - 1)
    yi = np.clip(np.rint(target[:, 1]).astype(int), 0, height - 1)
    valid &= current_mask[yi, xi] != 0
    source, target = source[valid], target[valid]
    tracks = len(source)
    if tracks < MIN_TRACKS or _grid_coverage(source, width, height) < MIN_GRID_CELLS:
        return None
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        return None
    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    inliers = int(inlier_mask.sum())
    ratio = inliers / max(tracks, 1)
    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    residuals = np.linalg.norm(predicted - target, axis=1)
    residual = float(np.median(residuals[inlier_mask])) if inliers else math.inf
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = math.hypot(a, b)
    rotation = math.atan2(b, a)
    finite = all(
        math.isfinite(value)
        for value in (*matrix.reshape(-1), residual, scale, rotation)
    )
    if (
        not finite
        or inliers < MIN_INLIERS
        or ratio < MIN_INLIER_RATIO
        or residual > MAX_MEDIAN_RESIDUAL
        or scale < 0.8
        or scale > 1.25
    ):
        return None
    confidence = min(1.0, ratio * min(1.0, inliers / 80.0))
    return CameraTrackFrame(
        timestamp_ms=0,
        translation_x=float(matrix[0, 2]),
        translation_y=float(matrix[1, 2]),
        rotation=rotation,
        scale=scale,
        confidence=confidence,
        tracks=tracks,
        inliers=inliers,
        residual=residual,
        valid=True,
    )


def _histogram_distance(first, second, cv2) -> float:
    first_hist = cv2.calcHist([first], [0], None, [32], [0, 256])
    second_hist = cv2.calcHist([second], [0], None, [32], [0, 256])
    cv2.normalize(first_hist, first_hist)
    cv2.normalize(second_hist, second_hist)
    return float(cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_BHATTACHARYYA))


def iter_estimate_track(
    path: str,
    inputs: Sequence[CameraMotionInput],
    cv2,
    np,
) -> Iterator[tuple[str, int, int]]:
    """Yield cooperative progress and return a complete timestamped track."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Camera Motion Correction could not open the source video")
    frames = []
    previous_gray = None
    previous_mask = None
    previous_bounds = None
    previous_source_timestamp = -1
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = height = 0
    try:
        for index, item in enumerate(inputs):
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(
                    "Camera Motion Correction source ended before the detection sequence"
                )
            source_timestamp = int(round(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
            if source_timestamp <= previous_source_timestamp:
                source_timestamp = (
                    int(round(index * 1000.0 / source_fps))
                    if source_fps > 0.0
                    else previous_source_timestamp + 1
                )
            source_timestamp = max(previous_source_timestamp + 1, source_timestamp)
            previous_source_timestamp = source_timestamp
            local_interval = (
                item.timestamp_ms - inputs[index - 1].timestamp_ms
                if index
                else (1000.0 / source_fps if source_fps > 0.0 else 33.0)
            )
            if abs(source_timestamp - int(item.timestamp_ms)) > max(
                2.0, abs(local_interval) * 0.5
            ):
                raise ValueError(
                    "Camera Motion Correction source timestamps did not match "
                    "the cached detections"
                )
            height, width = image.shape[:2]
            gray, working_scale = _gray_working(image, cv2)
            scaled_points = item.person_points
            mask, previous_bounds = _person_mask(
                gray.shape,
                scaled_points,
                cv2,
                np,
                previous_bounds,
            )
            if previous_gray is None:
                frame = CameraTrackFrame(timestamp_ms=item.timestamp_ms, valid=True)
            else:
                step = _estimate_step(
                    previous_gray,
                    gray,
                    previous_mask,
                    mask,
                    cv2,
                    np,
                )
                cut = _histogram_distance(previous_gray, gray, cv2) > 0.55
                if step is None or cut:
                    frame = CameraTrackFrame(
                        timestamp_ms=item.timestamp_ms,
                        valid=False,
                        cut=cut,
                    )
                else:
                    inverse_scale = 1.0 / max(working_scale, 1.0e-12)
                    frame = CameraTrackFrame(
                        timestamp_ms=item.timestamp_ms,
                        translation_x=step.translation_x * inverse_scale,
                        translation_y=step.translation_y * inverse_scale,
                        rotation=step.rotation,
                        scale=step.scale,
                        confidence=step.confidence,
                        tracks=step.tracks,
                        inliers=step.inliers,
                        residual=step.residual * inverse_scale,
                        valid=True,
                    )
            frames.append(frame)
            previous_gray = gray
            previous_mask = mask
            yield "Camera Motion: tracking background", index + 1, len(inputs)
        ok, _extra = capture.read()
        if ok and len(inputs) == 0:
            raise ValueError("Camera Motion Correction received no detections")
    finally:
        capture.release()
    return CameraMotionTrack(width=width, height=height, frames=tuple(frames))


def _weighted_smooth(values, times, weights, radius):
    result = []
    for index, timestamp in enumerate(times):
        selected = [
            other
            for other, other_time in enumerate(times)
            if abs(other_time - timestamp) <= radius
        ]
        if len(selected) < 2:
            result.append(values[index])
            continue
        local_weights = [
            weights[other]
            * max(0.0, 1.0 - abs(times[other] - timestamp) / max(radius, 1.0e-9))
            for other in selected
        ]
        total = sum(local_weights)
        result.append(
            sum(
                values[other] * weight
                for other, weight in zip(selected, local_weights, strict=True)
            )
            / total
            if total > 1.0e-9
            else values[index]
        )
    return result


def _continuous_angles(values):
    result = list(values)
    for index in range(1, len(result)):
        while result[index] - result[index - 1] > math.pi:
            result[index] -= math.tau
        while result[index] - result[index - 1] < -math.pi:
            result[index] += math.tau
    return result


def _reject_isolated_steps(frames, times):
    valid = [bool(frame.valid and not frame.cut) for frame in frames]
    for index, frame in enumerate(frames):
        if not valid[index] or index == 0:
            continue
        neighbors = [
            other
            for other, timestamp in enumerate(times)
            if other != index and valid[other] and abs(timestamp - times[index]) <= 0.25
        ]
        if len(neighbors) < 4:
            continue
        for field_name, floor in (
            ("translation_x", 2.0),
            ("translation_y", 2.0),
            ("rotation", 0.01),
        ):
            value = float(getattr(frame, field_name))
            sample = [float(getattr(frames[other], field_name)) for other in neighbors]
            center = median(sample)
            deviation = median([abs(item - center) for item in sample])
            if abs(value - center) > max(floor, 6.0 * deviation):
                valid[index] = False
                break
    return valid


def _bridge_short_gaps(frames, times, step_tx, step_ty, step_rotation, valid):
    effective = list(valid)
    index = 1
    while index < len(frames):
        if effective[index] or frames[index].cut:
            index += 1
            continue
        start = index
        while index < len(frames) and not effective[index] and not frames[index].cut:
            index += 1
        previous = start - 1
        following = index if index < len(frames) and effective[index] else -1
        if (
            previous < 0
            or following < 0
            or not effective[previous]
            or times[following] - times[previous] > MAX_INTERPOLATION_GAP_SECONDS
        ):
            continue
        span = max(times[following] - times[previous], 1.0e-9)
        for other in range(start, following):
            factor = (times[other] - times[previous]) / span
            step_tx[other] = (
                step_tx[previous] + (step_tx[following] - step_tx[previous]) * factor
            )
            step_ty[other] = (
                step_ty[previous] + (step_ty[following] - step_ty[previous]) * factor
            )
            step_rotation[other] = (
                step_rotation[previous]
                + (step_rotation[following] - step_rotation[previous]) * factor
            )
            effective[other] = True
    return effective


def correction_offsets(
    inputs: Sequence[CameraMotionInput],
    track: CameraMotionTrack,
    *,
    strength: float = 1.0,
) -> CameraCorrectionResult:
    """Return lateral source-camera offsets without modifying other axes."""
    if len(inputs) != len(track.frames) or not inputs:
        return CameraCorrectionResult(
            offsets_x=tuple(0.0 for _item in inputs),
            corrected_intervals=0,
            corrected_frames=0,
            tracked_frames=0,
            rejected_frames=len(inputs),
            median_inliers=0.0,
            warning="Camera Motion Correction skipped: camera track did not match the detection sequence",
        )
    if any(
        int(item.timestamp_ms) != int(frame.timestamp_ms)
        for item, frame in zip(inputs, track.frames, strict=True)
    ):
        return CameraCorrectionResult(
            offsets_x=tuple(0.0 for _item in inputs),
            corrected_intervals=0,
            corrected_frames=0,
            tracked_frames=0,
            rejected_frames=len(inputs),
            median_inliers=0.0,
            warning="Camera Motion Correction skipped: camera-track timestamps changed",
        )

    times = [item.timestamp_ms / 1000.0 for item in inputs]
    step_tx = [frame.translation_x for frame in track.frames]
    step_ty = [frame.translation_y for frame in track.frames]
    step_rotation = _continuous_angles([frame.rotation for frame in track.frames])
    weights = [frame.confidence if frame.valid else 0.0 for frame in track.frames]
    weights[0] = 0.0
    valid_steps = _reject_isolated_steps(track.frames, times)
    valid_steps = _bridge_short_gaps(
        track.frames,
        times,
        step_tx,
        step_ty,
        step_rotation,
        valid_steps,
    )
    step_tx = _weighted_smooth(step_tx, times, weights, SMOOTHING_RADIUS_SECONDS)
    step_ty = _weighted_smooth(step_ty, times, weights, SMOOTHING_RADIUS_SECONDS)
    step_rotation = _weighted_smooth(
        step_rotation, times, weights, SMOOTHING_RADIUS_SECONDS
    )
    step_tx[0] = step_ty[0] = step_rotation[0] = 0.0

    center_x = track.width * 0.5
    center_y = track.height * 0.5
    offsets = [0.0] * len(inputs)
    cumulative_angle = 0.0
    cumulative_tx = 0.0
    cumulative_ty = 0.0
    segment_anchor = 0.0
    interval_active = False
    intervals = 0
    tracked = 0
    last_valid = -1
    for index, (item, frame) in enumerate(zip(inputs, track.frames, strict=True)):
        if index == 0 or frame.cut:
            cumulative_angle = cumulative_tx = cumulative_ty = 0.0
            segment_anchor = offsets[index - 1] if index else 0.0
            interval_active = False
        if valid_steps[index]:
            tracked += 1
            last_valid = index
            angle = step_rotation[index]
            cosine, sine = math.cos(angle), math.sin(angle)
            # Compose the unit-scale SE(2) step. Estimated scale remains a
            # diagnostic only so camera zoom/dolly cannot rewrite depth.
            rotated_tx = cosine * cumulative_tx - sine * cumulative_ty
            rotated_ty = sine * cumulative_tx + cosine * cumulative_ty
            cumulative_tx = rotated_tx + step_tx[index]
            cumulative_ty = rotated_ty + step_ty[index]
            cumulative_angle += angle
        elif (
            last_valid >= 0
            and times[index] - times[last_valid] > MAX_INTERPOLATION_GAP_SECONDS
        ):
            # Long uncertainty holds the last offset. A later valid segment is
            # re-anchored instead of snapping to raw/default root translation.
            cumulative_angle = cumulative_tx = cumulative_ty = 0.0
            segment_anchor = offsets[index - 1]
            offsets[index] = segment_anchor
            interval_active = False
            continue

        translation = item.root_translation
        focal = float(item.focal_length)
        if (
            translation is None
            or focal <= 1.0e-6
            or abs(float(translation[2])) <= 1.0e-6
            or not valid_steps[index]
        ):
            offsets[index] = offsets[index - 1] if index else segment_anchor
            continue
        x, y, depth = (float(value) for value in translation)
        point_x = center_x + focal * x / depth
        point_y = center_y + focal * y / depth
        cosine, sine = math.cos(-cumulative_angle), math.sin(-cumulative_angle)
        shifted_x = point_x - cumulative_tx
        shifted_y = point_y - cumulative_ty
        stable_x = cosine * shifted_x - sine * shifted_y
        corrected_x = (stable_x - center_x) * depth / focal
        value = (corrected_x - x) * min(max(float(strength), 0.0), 1.0)
        if not math.isfinite(value):
            return CameraCorrectionResult(
                offsets_x=tuple(0.0 for _item in inputs),
                corrected_intervals=0,
                corrected_frames=0,
                tracked_frames=tracked,
                rejected_frames=len(inputs) - tracked,
                median_inliers=0.0,
                warning="Camera Motion Correction discarded a non-finite result",
            )
        if not interval_active:
            intervals += 1
            interval_active = True
            # Each reliable interval starts continuously at the previous
            # correction, while later frames retain relative camera removal.
            segment_anchor = (offsets[index - 1] if index else 0.0) - value
        offsets[index] = value + segment_anchor

    inlier_values = [
        frame.inliers for frame in track.frames if frame.valid and frame.inliers
    ]
    corrected = sum(abs(value) > 1.0e-9 for value in offsets)
    warning = ""
    if tracked <= 1 or corrected == 0:
        offsets = [0.0] * len(inputs)
        intervals = corrected = 0
        warning = (
            "Camera Motion Correction skipped: insufficient reliable background motion"
        )
    return CameraCorrectionResult(
        offsets_x=tuple(offsets),
        corrected_intervals=intervals,
        corrected_frames=corrected,
        tracked_frames=tracked,
        rejected_frames=len(inputs) - tracked,
        median_inliers=float(median(inlier_values)) if inlier_values else 0.0,
        warning=warning,
    )


def sidecar_state(directory: Path) -> str:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        if (
            manifest.get("schema_version") == TRACK_SCHEMA_VERSION
            and manifest.get("algorithm") == ALGORITHM_VERSION
            and (directory / str(manifest.get("payload", ""))).is_file()
        ):
            return "Ready"
    except (OSError, ValueError, TypeError):
        pass
    return "Not Prepared"


__all__ = [
    "CameraCorrectionResult",
    "CameraMotionInput",
    "CameraMotionTrack",
    "CameraTrackFrame",
    "cache_key",
    "correction_offsets",
    "input_from_frame",
    "iter_estimate_track",
    "load_sidecar",
    "sidecar_state",
    "store_sidecar",
]
