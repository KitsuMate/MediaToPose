"""SAM 3D Body adapter producing Media to Pose canonical observations."""

from __future__ import annotations

import gc
import math
import threading
from contextlib import suppress
from pathlib import Path
from threading import Event
from typing import Dict, Iterable, Optional

from .. import camera_calibration, sam3d_backend, sam3d_ort_runtime
from ..constants import HAND_LANDMARK_NAMES
from ..dependency_bootstrap import import_runtime_dependencies
from ..media_io import (
    decode_image,
    decode_video_frame_index_with_timestamp,
    iter_video_frames,
)
from ..models import (
    CanonicalFrame,
    JointFrameObservation,
    Landmark,
    RootMotionObservation,
)

_MODEL_LOCK = threading.RLock()
_MODEL_KEY = None
_ESTIMATOR = None
_RUNTIME_DIAGNOSTICS = None


# The embedded checkpoint ordering is part of the validated ONNX bundle
# contract; use that single source of truth for provider conversion as well.
_MHR_JOINTS = sam3d_ort_runtime.MHR_OUTPUT_JOINTS


def _euler_zyx_matrix(value, np):
    """Convert MHR's ``[rz, ry, rx]`` Euler output to ``Rz @ Ry @ Rx``."""
    data = np.asarray(value, dtype=float).reshape(-1)
    if len(data) < 3 or not np.isfinite(data[:3]).all():
        return None
    rz, ry, rx = (float(item) for item in data[:3])
    cz, sz = math.cos(rz), math.sin(rz)
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)
    return np.asarray(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        ),
        dtype=float,
    )


def model_loaded() -> bool:
    return _ESTIMATOR is not None


def unload_model_cache() -> None:
    global _MODEL_KEY, _ESTIMATOR, _RUNTIME_DIAGNOSTICS
    with _MODEL_LOCK:
        estimator = _ESTIMATOR
        _MODEL_KEY = None
        _ESTIMATOR = None
        _RUNTIME_DIAGNOSTICS = None
        if estimator is not None:
            with suppress(AttributeError, RuntimeError):
                estimator.close()
        gc.collect()


def runtime_diagnostics():
    """Return immutable diagnostics for the currently loaded ORT session."""
    return _RUNTIME_DIAGNOSTICS


def _refresh_runtime_diagnostics(estimator):
    """Publish the estimator's latest immutable diagnostics snapshot."""
    global _RUNTIME_DIAGNOSTICS
    diagnostics = estimator.diagnostics
    with _MODEL_LOCK:
        # An inference may finish while another thread replaces the shared
        # estimator.  Never let the retired estimator overwrite the new
        # cache's diagnostics in that case.
        if estimator is _ESTIMATOR:
            _RUNTIME_DIAGNOSTICS = diagnostics
    return diagnostics


def dependency_status(options=None):
    options = options or {}
    variant = str(options.get("variant", "VITH"))
    device = str(options.get("device", "AUTO"))
    snapshot = sam3d_backend.backend_snapshot(variant, device)
    return snapshot.installed, snapshot.message


def cache_fingerprint(options):
    variant = str(options.get("variant", "VITH"))
    spec = sam3d_backend.model_spec(variant)
    return {
        "adapter_version": sam3d_backend.ADAPTER_VERSION,
        "bundle_contract": sam3d_ort_runtime.BUNDLE_CONTRACT,
        "model_repo": spec["repo"],
        "bundle_revision": spec["revision"],
        "model_revision": spec.get("model_revision", ""),
        "model_precision": spec.get("precision", ""),
        "model_variant": variant,
    }


def _load_model(
    variant: str,
    requested_device: str,
    paths: Dict[str, str],
    *,
    verify_accelerator_execution: bool = False,
):
    global _MODEL_KEY, _ESTIMATOR, _RUNTIME_DIAGNOSTICS
    with _MODEL_LOCK:
        requested_mode = sam3d_ort_runtime.normalize_mode(requested_device)
        verify_accelerator_execution = bool(verify_accelerator_execution)
        key = (
            variant,
            requested_mode,
            paths.get("backend_directory", ""),
            paths.get("bundle_manifest_path", ""),
            paths.get("bundle_revision", ""),
            paths.get("model_revision", ""),
            verify_accelerator_execution,
        )
        if _ESTIMATOR is not None and key == _MODEL_KEY:
            return (
                _ESTIMATOR,
                _RUNTIME_DIAGNOSTICS,
                _RUNTIME_DIAGNOSTICS.actual_device,
            )
        unload_model_cache()
        estimator = sam3d_ort_runtime.load_estimator(
            paths,
            variant,
            requested_mode,
            verify_accelerator_execution=verify_accelerator_execution,
        )
        _ESTIMATOR = estimator
        _MODEL_KEY = key
        _RUNTIME_DIAGNOSTICS = estimator.diagnostics
        return estimator, estimator.diagnostics, estimator.diagnostics.actual_device


def load_model_cache(
    variant: str,
    requested_device: str,
    paths: Optional[Dict[str, str]] = None,
    *,
    verify_accelerator_execution: bool = False,
) -> str:
    """Load and retain the selected estimator without starting capture."""
    _estimator, _diagnostics, device_name = _load_model(
        variant,
        requested_device,
        paths or sam3d_backend.runtime_paths(),
        verify_accelerator_execution=verify_accelerator_execution,
    )
    return device_name


def _number(value, default=0.0) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


class SAM3DBodyProvider:
    capabilities = frozenset({"BODY", "HANDS"})

    def __init__(
        self,
        *,
        use_body: bool,
        use_hands: bool,
        use_face: bool = False,
        mirror: bool = False,
        confidence: float = 0.5,
        smoothing: float = 0.35,
        video_mode: bool = False,
        variant: str = "VITH",
        device: str = "AUTO",
        verify_accelerator_execution: bool = False,
        model_directory: str = "",
        camera_intrinsics=None,
        **runtime_paths,
    ):
        del model_directory
        if use_face:
            raise ValueError("SAM 3D Body does not support face capture")
        self.use_body = bool(use_body)
        self.use_hands = bool(use_hands)
        self.mirror = bool(mirror)
        self.confidence = float(confidence)
        self.smoothing = float(smoothing)
        self.video_mode = bool(video_mode)
        self.variant = variant
        self.camera = (
            camera_calibration.from_mapping(camera_intrinsics)
            if camera_intrinsics
            else None
        )
        self.estimator, self.runtime, self.device = _load_model(
            variant,
            device,
            runtime_paths,
            verify_accelerator_execution=verify_accelerator_execution,
        )
        self.cv2, self.mp, self.np = import_runtime_dependencies()
        self._neutral_rotations = {
            int(index): self.np.asarray(matrix, dtype=float).reshape(3, 3)
            for index, matrix in self.estimator.neutral_rotations.items()
        }
        self._previous: Optional[CanonicalFrame] = None
        self._locator_fallback_used = False
        self._locator = self._create_locator()

    def _create_locator(self):
        mode = (
            self.mp.tasks.vision.RunningMode.VIDEO
            if self.video_mode
            else self.mp.tasks.vision.RunningMode.IMAGE
        )
        model = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "models"
            / "pose_landmarker_full.task"
        )
        if not model.is_file():
            raise FileNotFoundError("Bundled person locator model is missing")
        options = self.mp.tasks.vision.PoseLandmarkerOptions(
            base_options=self.mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=self.confidence,
            min_pose_presence_confidence=self.confidence,
            min_tracking_confidence=self.confidence,
            output_segmentation_masks=False,
        )
        return self.mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _decode_image(self, path: str):
        return decode_image(path, self.cv2, self.np)

    def _input_image(self, bgr):
        return self.cv2.flip(bgr, 1) if self.mirror else bgr

    def _bbox(self, bgr, timestamp_ms: int):
        self._locator_fallback_used = False
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=self.np.ascontiguousarray(rgb),
        )
        result = (
            self._locator.detect_for_video(mp_image, timestamp_ms)
            if self.video_mode
            else self._locator.detect(mp_image)
        )
        height, width = bgr.shape[:2]
        if not result.pose_landmarks:
            # The lightweight MediaPipe locator can reject stylized or
            # airborne characters that SAM itself handles correctly.  It is
            # only used to propose a crop, so a miss must not prevent the
            # more capable provider from seeing the image.  The extension is
            # single-person by design; the complete frame is the safest
            # provider-neutral fallback and does not invent joint positions.
            self._locator_fallback_used = True
            return self.np.asarray(
                [[0.0, 0.0, float(width - 1), float(height - 1)]],
                dtype=self.np.float32,
            )
        points = result.pose_landmarks[0]
        xs = [float(point.x) * width for point in points]
        ys = [float(point.y) * height for point in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        padding = max(right - left, bottom - top) * 0.18
        return self.np.asarray(
            [
                [
                    max(0.0, left - padding),
                    max(0.0, top - padding),
                    min(float(width - 1), right + padding),
                    min(float(height - 1), bottom + padding),
                ]
            ],
            dtype=self.np.float32,
        )

    def _point(self, joints3d, joints2d, index, width, height) -> Landmark:
        raw3d = joints3d[index]
        raw2d = joints2d[index] if joints2d is not None else ()
        position = tuple(_number(raw3d[axis]) for axis in range(3))
        image = (
            (
                _number(raw2d[0]) / max(1.0, float(width)),
                _number(raw2d[1]) / max(1.0, float(height)),
            )
            if len(raw2d) >= 2
            else (0.0, 0.0)
        )
        confidence = 1.0
        if len(raw2d) >= 3:
            confidence = _number(raw2d[2], 1.0)
        elif len(raw3d) >= 4:
            confidence = _number(raw3d[3], 1.0)
        return Landmark(
            position=position,
            image=image,
            confidence=confidence,
            valid=confidence >= self.confidence,
        )

    @staticmethod
    def _average(first: Landmark, second: Landmark) -> Landmark:
        return Landmark(
            position=tuple(
                (first.position[i] + second.position[i]) * 0.5 for i in range(3)
            ),
            image=tuple((first.image[i] + second.image[i]) * 0.5 for i in range(2)),
            confidence=min(first.confidence, second.confidence),
            # This is an anatomical midpoint, so one endpoint cannot stand in
            # for both sides. Keep the position for overlays/diagnostics but
            # do not let it drive a toe unless both observations are usable.
            valid=first.valid and second.valid,
        )

    def _hand(self, points, side: str) -> Dict[str, Landmark]:
        wrist = points[62 if side == "LEFT" else 41]
        start = 42 if side == "LEFT" else 21
        result = {"WRIST": wrist}
        groups = (
            ("THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP"),
            (
                "INDEX_FINGER_MCP",
                "INDEX_FINGER_PIP",
                "INDEX_FINGER_DIP",
                "INDEX_FINGER_TIP",
            ),
            (
                "MIDDLE_FINGER_MCP",
                "MIDDLE_FINGER_PIP",
                "MIDDLE_FINGER_DIP",
                "MIDDLE_FINGER_TIP",
            ),
            (
                "RING_FINGER_MCP",
                "RING_FINGER_PIP",
                "RING_FINGER_DIP",
                "RING_FINGER_TIP",
            ),
            ("PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"),
        )
        for finger_index, names in enumerate(groups):
            distal_to_proximal = points[
                start + finger_index * 4 : start + finger_index * 4 + 4
            ]
            result.update(
                zip(
                    names,
                    reversed(distal_to_proximal),
                    strict=True,
                )
            )
        return {name: result[name] for name in HAND_LANDMARK_NAMES if name in result}

    def _convert(self, output, width: int, height: int, timestamp_ms: int):
        if not output:
            raise ValueError("SAM 3D Body did not return a person")
        person = output[0] if isinstance(output, (list, tuple)) else output
        joints3d = self.np.asarray(person["pred_keypoints_3d"])
        joints2d_value = person.get("pred_keypoints_2d")
        joints2d = (
            self.np.asarray(joints2d_value) if joints2d_value is not None else None
        )
        if joints3d.ndim > 2:
            joints3d = joints3d[0]
        if joints2d is not None and joints2d.ndim > 2:
            joints2d = joints2d[0]
        if len(joints3d) < 70:
            raise ValueError("SAM 3D Body returned an incomplete MHR70 pose")
        points = [
            self._point(joints3d, joints2d, index, width, height) for index in range(70)
        ]
        frame = CanonicalFrame(timestamp_ms=timestamp_ms)
        frame.camera = getattr(self, "camera", None)
        full_joint_value = person.get("pred_joint_coords")
        full_rotation_value = person.get("pred_global_rots")
        if full_joint_value is not None:
            full_joints = self.np.asarray(full_joint_value, dtype=float)
            if full_joints.ndim > 2:
                full_joints = full_joints[0]
            full_rotations = None
            if full_rotation_value is not None:
                full_rotations = self.np.asarray(full_rotation_value, dtype=float)
                if full_rotations.ndim > 3:
                    full_rotations = full_rotations[0]
            for name, index in _MHR_JOINTS.items():
                if index >= len(full_joints):
                    continue
                position = full_joints[index].reshape(-1)
                valid = len(position) >= 3 and self.np.isfinite(position[:3]).all()
                frame.auxiliary[name] = Landmark(
                    position=tuple(_number(item) for item in position[:3]),
                    confidence=1.0 if valid else 0.0,
                    valid=valid,
                )
                neutral = self._neutral_rotations.get(index)
                if (
                    full_rotations is None
                    or index >= len(full_rotations)
                    or neutral is None
                ):
                    continue
                predicted = self.np.asarray(full_rotations[index], dtype=float).reshape(
                    3, 3
                )
                finite = self.np.isfinite(predicted).all()
                valid_rotation = False
                if finite:
                    try:
                        left, singular, right = self.np.linalg.svd(predicted)
                        predicted = left @ right
                        if self.np.linalg.det(predicted) < 0.0:
                            left[:, -1] *= -1.0
                            predicted = left @ right
                        valid_rotation = bool(
                            self.np.max(self.np.abs(singular - 1.0)) < 0.15
                            and self.np.linalg.det(predicted) > 0.0
                        )
                    except self.np.linalg.LinAlgError:
                        valid_rotation = False
                relative = predicted @ neutral.T
                frame.joint_frames[name] = JointFrameObservation(
                    rotation_matrix=tuple(
                        _number(item) for item in relative.reshape(-1)
                    ),
                    confidence=1.0 if valid_rotation else 0.0,
                    valid=(valid_rotation and self.np.isfinite(relative).all()),
                )
            spine_names = [
                *(f"MHR_SPINE_{index}" for index in range(4)),
                "MHR_NECK",
            ]
            spine_points = [frame.auxiliary.get(name) for name in spine_names]
            if all(point is not None and point.valid for point in spine_points):
                segments = [
                    self.np.asarray(second.position, dtype=float)
                    - self.np.asarray(first.position, dtype=float)
                    for first, second in zip(
                        spine_points, spine_points[1:], strict=False
                    )
                ]
                lengths = self.np.asarray(
                    [self.np.linalg.norm(segment) for segment in segments]
                )
                median = float(self.np.median(lengths))
                tangent_angles = []
                for first, second, first_length, second_length in zip(
                    segments,
                    segments[1:],
                    lengths,
                    lengths[1:],
                    strict=False,
                ):
                    cosine = float(
                        self.np.dot(first, second)
                        / max(first_length * second_length, 1.0e-12)
                    )
                    tangent_angles.append(
                        math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
                    )
                invalid_curve = (
                    median < 1.0e-8
                    or float(lengths.min()) < median * 0.12
                    or float(lengths.max()) > median * 4.0
                    or max(tangent_angles, default=0.0) > 110.0
                )
                if invalid_curve:
                    for name in spine_names:
                        point = frame.auxiliary[name]
                        point.valid = False
                        point.confidence = 0.0
                    frame.warnings.append(
                        "SAM MHR spine centerline was degenerate; used sparse torso fallback"
                    )
        if getattr(self, "_locator_fallback_used", False):
            frame.warnings.append(
                "Person locator missed this frame; SAM used the full image"
            )
        camera_value = person.get("pred_cam_t")
        if camera_value is not None:
            camera = self.np.asarray(camera_value, dtype=float).reshape(-1)
            rotation_value = person.get("global_rot")
            rotation = None
            if rotation_value is not None:
                matrix = _euler_zyx_matrix(rotation_value, self.np)
                if matrix is not None:
                    rotation = tuple(_number(item) for item in matrix.reshape(-1))
            focal_value = self.np.asarray(
                person.get("focal_length", 0.0), dtype=float
            ).reshape(-1)
            scale_value = self.np.asarray(
                person.get("scale_params", 1.0), dtype=float
            ).reshape(-1)
            confidence_values = [point.confidence for point in points[:21]]
            frame.root_motion = RootMotionObservation(
                translation=tuple(_number(item) for item in camera[:3]),
                rotation_matrix=rotation,
                focal_length=(_number(focal_value[0]) if len(focal_value) else 0.0),
                scale=(_number(scale_value[0], 1.0) if len(scale_value) else 1.0),
                confidence=(
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else 0.0
                ),
                valid=len(camera) >= 3,
            )
        if self.use_body:
            mapping = {
                "NOSE": 0,
                "LEFT_EYE": 1,
                "RIGHT_EYE": 2,
                "LEFT_EAR": 3,
                "RIGHT_EAR": 4,
                "LEFT_SHOULDER": 5,
                "RIGHT_SHOULDER": 6,
                "LEFT_ELBOW": 7,
                "RIGHT_ELBOW": 8,
                "LEFT_HIP": 9,
                "RIGHT_HIP": 10,
                "LEFT_KNEE": 11,
                "RIGHT_KNEE": 12,
                "LEFT_ANKLE": 13,
                "RIGHT_ANKLE": 14,
                "LEFT_HEEL": 17,
                "RIGHT_HEEL": 20,
                "LEFT_WRIST": 62,
                "RIGHT_WRIST": 41,
            }
            frame.body.update({name: points[index] for name, index in mapping.items()})
            # The MHR70 elbow landmarks are surface-derived observations.
            # Prefer the articulated 127-joint chain for arm swing while
            # retaining MHR70's directly observed image coordinates.  This
            # also keeps both shoulders symmetrical with the full-skeleton
            # clavicle observations used by Shoulder Correction.
            for side, label in (("L", "LEFT"), ("R", "RIGHT")):
                for body_name, joint_name in (
                    (f"{label}_SHOULDER", f"MHR_UPPER_ARM_{side}"),
                    (f"{label}_ELBOW", f"MHR_LOWER_ARM_{side}"),
                    (f"{label}_WRIST", f"MHR_HAND_{side}"),
                ):
                    source = frame.auxiliary.get(joint_name)
                    observed = frame.body.get(body_name)
                    if source is not None and source.valid and observed is not None:
                        observed.position = source.position
                        observed.confidence = min(
                            observed.confidence, source.confidence
                        )
            frame.body["LEFT_FOOT_INDEX"] = self._average(points[15], points[16])
            frame.body["RIGHT_FOOT_INDEX"] = self._average(points[18], points[19])
        for side in ("LEFT", "RIGHT"):
            hand = self._hand(points, side)
            if self.use_hands:
                frame.hands[side] = hand
            if self.use_body:
                frame.body[f"{side}_THUMB"] = hand["THUMB_TIP"]
                frame.body[f"{side}_INDEX"] = hand["INDEX_FINGER_TIP"]
                frame.body[f"{side}_PINKY"] = hand["PINKY_TIP"]
        frame.auxiliary.update(
            {
                "LEFT_OLECRANON": points[63],
                "RIGHT_OLECRANON": points[64],
                "LEFT_CUBITAL_FOSSA": points[65],
                "RIGHT_CUBITAL_FOSSA": points[66],
                "LEFT_ACROMION": points[67],
                "RIGHT_ACROMION": points[68],
                "NECK": points[69],
            }
        )
        return self._smooth_frame(frame)

    def _smooth_frame(self, frame: CanonicalFrame) -> CanonicalFrame:
        # Video cleanup needs raw, unlagged observations.  Still-image and
        # single-frame preview behavior retains the existing optional filter.
        if getattr(self, "video_mode", False):
            self._previous = frame
            return frame
        previous = self._previous
        if previous is None or self.smoothing <= 0.0:
            self._previous = frame
            return frame
        alpha = 1.0 - self.smoothing

        def smooth(current, prior):
            if current.valid and prior.valid:
                current.position = tuple(
                    alpha * current.position[i] + (1.0 - alpha) * prior.position[i]
                    for i in range(3)
                )

        for collection_name in ("body", "auxiliary"):
            current = getattr(frame, collection_name)
            prior = getattr(previous, collection_name)
            for name, point in current.items():
                if name in prior:
                    smooth(point, prior[name])
        for side, hand in frame.hands.items():
            prior_hand = previous.hands.get(side, {})
            for name, point in hand.items():
                if name in prior_hand:
                    smooth(point, prior_hand[name])
        self._previous = frame
        return frame

    def _detect(self, bgr, timestamp_ms: int) -> CanonicalFrame:
        image = self._input_image(bgr)
        boxes = self._bbox(image, timestamp_ms)
        rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        try:
            output = self.estimator.process_one_image(
                rgb,
                bboxes=boxes,
                inference_type="full" if self.use_hands else "body",
                camera_intrinsics=(
                    camera_calibration.matrix(self.camera, self.np)
                    if getattr(self, "camera", None) is not None and self.camera.valid
                    else None
                ),
            )
        finally:
            # AUTO may replace an accelerator session with the complete CPU
            # fallback after a genuine ORT graph-execution failure. Keep this
            # provider and the shared cache aligned with that immutable update.
            self.runtime = _refresh_runtime_diagnostics(self.estimator)
            self.device = self.runtime.actual_device
        height, width = image.shape[:2]
        return self._convert(output, width, height, timestamp_ms)

    def detect_image(self, path: str) -> CanonicalFrame:
        return self._detect(self._decode_image(path), 0)

    def detect_video_frame_index(
        self,
        path: str,
        source_index: int,
    ) -> CanonicalFrame:
        image, timestamp_ms = decode_video_frame_index_with_timestamp(
            path,
            source_index,
            self.cv2,
        )
        return self._detect(image, timestamp_ms)

    def iter_video(
        self,
        path: str,
        cancel_event: Event,
        progress,
    ) -> Iterable[CanonicalFrame]:
        for index, timestamp, total, image in iter_video_frames(
            path,
            cancel_event,
            self.cv2,
        ):
            yield self._detect(image, timestamp)
            completed = index + 1
            progress(
                completed,
                total,
                f"SAM 3D Body frame {completed}/{total or '?'}",
            )

    def close(self) -> None:
        if self._locator is not None:
            self._locator.close()
            self._locator = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
