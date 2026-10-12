"""MediaPipe Tasks implementation for body, hands, and face."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from threading import Event

from ...core.camera import calibration as camera_calibration
from ...core.dependencies import dependency_snapshot, import_runtime_dependencies
from ...core.errors import ModelMissing
from ...core.media_io import (
    decode_image,
    decode_video_frame_index_with_timestamp,
    iter_video_frames,
)
from ...core.types import CanonicalFrame, Landmark
from .body import PoseMixin, _estimate_root_motion
from .face import FaceMixin
from .hands import HandsMixin


def dependency_status(_options=None) -> tuple[bool, str]:
    snapshot = dependency_snapshot()
    if not snapshot.checked:
        return False, snapshot.message
    return snapshot.available, snapshot.message


def _import_dependencies():
    return import_runtime_dependencies()


class MediaPipeProvider(PoseMixin, HandsMixin, FaceMixin):
    capabilities = frozenset({"BODY", "HANDS", "FACE"})

    def __init__(
        self,
        model_directory: str,
        *,
        use_body: bool,
        use_hands: bool,
        use_face: bool,
        mirror: bool = False,
        confidence: float = 0.5,
        smoothing: float = 0.35,
        video_mode: bool = False,
        camera_intrinsics=None,
    ):
        self.cv2, self.mp, self.np = _import_dependencies()
        self.model_directory = Path(model_directory)
        self.use_body = use_body
        self.use_hands = use_hands
        self.use_face = use_face
        self.mirror = mirror
        self.confidence = confidence
        self.smoothing = smoothing
        self.video_mode = video_mode
        self.camera = (
            camera_calibration.from_mapping(camera_intrinsics)
            if camera_intrinsics
            else None
        )
        self._pose = None
        self._hands = None
        self._hand_crops = {}
        self._face = None
        self._fallback_pose = None
        self._fallback_face = None
        self._face_crop = None
        self._fallback_face_crop = None
        self._previous: CanonicalFrame | None = None
        self._trusted_hands = {}
        self._create_tasks()

    def _model(self, filename: str) -> str:
        path = self.model_directory / filename
        if not path.exists():
            raise ModelMissing(
                f"Bundled MediaPipe model is missing: {path.name}. "
                "Reinstall a complete Media to Pose package."
            )
        return str(path)

    def _create_tasks(self) -> None:
        mp = self.mp
        mode = (
            mp.tasks.vision.RunningMode.VIDEO
            if self.video_mode
            else mp.tasks.vision.RunningMode.IMAGE
        )
        self._running_mode = mode
        if self.use_body:
            self._pose = self._create_pose_task(self.confidence, mode)
        if self.use_hands:
            self._hands = self._create_hand_task(self.confidence, mode, num_hands=2)
        # Face landmarking is independent from body posing. Pose landmarks
        # already provide the eyes and ears used by the body head solver, so
        # do not pay for the heavier face task unless facial output is enabled.
        if self.use_face:
            self._face = self._create_face_task(self.confidence, mode)

    def _create_pose_task(self, confidence: float, mode):
        mp = self.mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self._model("pose_landmarker_full.task")
            ),
            running_mode=mode,
            num_poses=2,
            min_pose_detection_confidence=confidence,
            min_pose_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            output_segmentation_masks=False,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _create_hand_task(self, confidence: float, mode, *, num_hands: int):
        mp = self.mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self._model("hand_landmarker.task")
            ),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _create_face_task(self, confidence: float, mode):
        mp = self.mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self._model("face_landmarker.task")
            ),
            running_mode=mode,
            num_faces=2,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @property
    def fallback_confidence(self) -> float:
        """A bounded recovery threshold used only after normal detection fails."""
        return max(0.1, min(self.confidence * 0.5, 0.3))

    def _fallback_pose_task(self):
        if self._fallback_pose is None:
            self._fallback_pose = self._create_pose_task(
                self.fallback_confidence,
                self._running_mode,
            )
        return self._fallback_pose

    def _fallback_face_task(self):
        if self._fallback_face is None:
            self._fallback_face = self._create_face_task(
                self.fallback_confidence,
                self._running_mode,
            )
        return self._fallback_face

    def _hand_crop_task(self, side: str):
        task = self._hand_crops.get(side)
        if task is None:
            task = self._create_hand_task(
                self.confidence,
                self._running_mode,
                num_hands=1,
            )
            self._hand_crops[side] = task
        return task

    def _face_crop_task(self, *, fallback: bool):
        attribute = "_fallback_face_crop" if fallback else "_face_crop"
        task = getattr(self, attribute)
        if task is None:
            confidence = self.fallback_confidence if fallback else self.confidence
            task = self._create_face_task(
                confidence,
                self.mp.tasks.vision.RunningMode.IMAGE,
            )
            setattr(self, attribute, task)
        return task

    def _decode_image(self, path: str):
        return decode_image(path, self.cv2, self.np)

    def _mp_image(self, bgr, *, apply_mirror: bool = True):
        frame = self.cv2.flip(bgr, 1) if self.mirror and apply_mirror else bgr
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        rgb = self.np.ascontiguousarray(rgb)
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

    def _detect_task(self, task, image, timestamp_ms: int):
        if self.video_mode:
            return task.detect_for_video(image, timestamp_ms)
        return task.detect(image)

    def _smooth_frame(self, frame: CanonicalFrame) -> CanonicalFrame:
        # Full video sequences are cleaned symmetrically before keying.  A
        # causal provider filter would permanently introduce temporal lag.
        if getattr(self, "video_mode", False):
            self._previous = frame
            return frame
        previous = self._previous
        if not previous or self.smoothing <= 0.0:
            self._previous = frame
            return frame
        alpha = 1.0 - self.smoothing

        def blend(current: Landmark, prior: Landmark) -> None:
            if not current.valid or not prior.valid:
                return
            current.position = tuple(
                alpha * a + (1.0 - alpha) * b
                for a, b in zip(current.position, prior.position, strict=False)
            )

        for name, landmark in frame.body.items():
            if name in previous.body:
                blend(landmark, previous.body[name])
        for side, hand in frame.hands.items():
            prior_hand = previous.hands.get(side, {})
            for name, landmark in hand.items():
                if name in prior_hand:
                    blend(landmark, prior_hand[name])
        for name, value in list(frame.blendshapes.items()):
            if name in previous.blendshapes:
                frame.blendshapes[name] = (
                    alpha * value + (1.0 - alpha) * previous.blendshapes[name]
                )
        self._previous = frame
        return frame

    def _detect(self, bgr, timestamp_ms: int) -> CanonicalFrame:
        image = self._mp_image(bgr)
        frame = CanonicalFrame(timestamp_ms=timestamp_ms)
        frame.camera = getattr(self, "camera", None)
        pose_result = (
            self._detect_task(self._pose, image, timestamp_ms) if self._pose else None
        )
        pose_fallback = False
        if (
            self._pose is not None
            and not self._has_pose(pose_result)
            and self.fallback_confidence < self.confidence
        ):
            recovered_pose = self._detect_task(
                self._fallback_pose_task(),
                image,
                timestamp_ms,
            )
            if self._has_pose(recovered_pose):
                pose_result = recovered_pose
                pose_fallback = True
        # Disabled capture sections are not detection failures and must not
        # generate misleading "not detected" warnings.
        if self._pose is not None:
            self._convert_pose(
                pose_result,
                frame,
                landmark_threshold=(
                    self.fallback_confidence if pose_fallback else self.confidence
                ),
            )
            if pose_fallback:
                frame.warnings.append(
                    "Body pose detected only with the reduced confidence threshold "
                    f"({self.fallback_confidence:.0%} instead of "
                    f"{self.confidence:.0%}); using the strongest candidate"
                )
            if frame.body:
                height, width = bgr.shape[:2]
                frame.root_motion = _estimate_root_motion(
                    frame,
                    width,
                    height,
                    self.np,
                    getattr(self, "camera", None),
                )
        if self._hands is not None:
            # Pose-guided, side-specific crops make each hand larger in the
            # model input and remove the need to infer anatomical side from a
            # full-frame pair. Each side owns independent VIDEO tracking state.
            crop_candidates = []
            for side in ("LEFT", "RIGHT"):
                hand = self._detect_hand_crop(bgr, frame, side, timestamp_ms)
                if hand is not None:
                    crop_candidates.append((side, hand))
            self._assign_pose_guided_hands(crop_candidates, frame)
            if len(frame.hands) < 2:
                hand_result = self._detect_task(self._hands, image, timestamp_ms)
                self._convert_hands(
                    hand_result,
                    frame,
                    preserve_existing=True,
                )
            self._hold_recent_hands(frame)
        if self._face is not None:
            face_result = self._detect_face_crop(bgr, frame, fallback=False)
            face_recovery = "crop" if self._has_face(face_result) else ""
            if not self._has_face(face_result):
                face_result = self._detect_task(self._face, image, timestamp_ms)
            if (
                not self._has_face(face_result)
                and self.fallback_confidence < self.confidence
            ):
                recovered_face = self._detect_face_crop(
                    bgr,
                    frame,
                    fallback=True,
                )
                if self._has_face(recovered_face):
                    face_result = recovered_face
                    face_recovery = "low_crop"
                else:
                    recovered_face = self._detect_task(
                        self._fallback_face_task(),
                        image,
                        timestamp_ms,
                    )
                    if self._has_face(recovered_face):
                        face_result = recovered_face
                        face_recovery = "low"
            self._convert_face(face_result, frame)
            if face_recovery == "crop":
                frame.warnings.append(
                    "Face detected at normal confidence in a closer crop guided by "
                    "the body pose"
                )
            elif face_recovery in {"low", "low_crop"}:
                detail = " in a pose-guided crop" if face_recovery == "low_crop" else ""
                frame.warnings.append(
                    "Face detected only with the reduced confidence threshold"
                    f"{detail} ({self.fallback_confidence:.0%} instead of "
                    f"{self.confidence:.0%}); using the most suitable candidate"
                )
        return self._smooth_frame(frame)

    def detect_image(self, path: str) -> CanonicalFrame:
        return self._detect(self._decode_image(path), 0)

    def detect_video_frame_index(self, path: str, source_index: int) -> CanonicalFrame:
        image, timestamp_ms = decode_video_frame_index_with_timestamp(
            path,
            source_index,
            self.cv2,
        )
        return self._detect(image, timestamp_ms)

    def iter_video(
        self, path: str, cancel_event: Event, progress
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
                f"Detecting frame {completed}/{total or '?'}",
            )

    def close(self) -> None:
        tasks = (
            self._pose,
            self._hands,
            *self._hand_crops.values(),
            self._face,
            self._fallback_pose,
            self._fallback_face,
            self._face_crop,
            self._fallback_face_crop,
        )
        closed = set()
        for task in tasks:
            if task is not None and id(task) not in closed:
                task.close()
                closed.add(id(task))
        self._pose = self._hands = self._face = None
        self._hand_crops = {}
        self._trusted_hands = {}
        self._fallback_pose = self._fallback_face = None
        self._face_crop = self._fallback_face_crop = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
