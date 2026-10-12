"""MediaPipe Tasks implementation for body, hands, and face."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import permutations
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from .. import camera_calibration
from ..constants import (
    BODY_JOINT_PAIRS,
    HAND_LANDMARK_NAMES,
    POSE_LANDMARK_NAMES,
)
from ..dependency_bootstrap import (
    dependency_snapshot,
    import_runtime_dependencies,
)
from ..media_io import (
    decode_image,
    decode_video_frame_index_with_timestamp,
    iter_video_frames,
)
from ..models import CanonicalFrame, Landmark, RootMotionObservation

_ROOT_MOTION_LANDMARKS = (
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)


def dependency_status(_options=None) -> tuple[bool, str]:
    snapshot = dependency_snapshot()
    if not snapshot.checked:
        return False, snapshot.message
    return snapshot.available, snapshot.message


def _import_dependencies():
    return import_runtime_dependencies()


def _confidence(raw) -> float:
    visibility = getattr(raw, "visibility", None)
    presence = getattr(raw, "presence", None)
    values = [float(value) for value in (visibility, presence) if value is not None]
    return min(values) if values else 1.0


def _landmark(raw, normalized=None, threshold: float = 0.5) -> Landmark:
    confidence = _confidence(raw)
    image = (
        (float(normalized.x), float(normalized.y))
        if normalized is not None
        else (0.0, 0.0)
    )
    return Landmark(
        position=(float(raw.x), float(raw.y), float(raw.z)),
        image=image,
        confidence=confidence,
        valid=confidence >= threshold,
    )


def _estimate_root_motion(frame, width: int, height: int, np, camera=None):
    """Estimate hip translation by aligning MediaPipe's 3D and 2D poses.

    Pose world landmarks are metric but hip-centered. Their corresponding image
    landmarks retain the missing placement. With a fixed, conservative focal
    estimate, perspective projection is linear in the three translation
    components, so a robust weighted solve recovers camera-relative travel
    without changing the provider's anatomical landmark coordinates.
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return None

    observations = []
    for name in _ROOT_MOTION_LANDMARKS:
        point = frame.body.get(name)
        if point is None or not point.valid:
            continue
        values = (*point.position, *point.image, point.confidence)
        if not all(math.isfinite(float(value)) for value in values):
            continue
        observations.append(point)
    if len(observations) < 6:
        return None

    camera = camera or camera_calibration.intrinsics(
        width,
        height,
        float(width),
        float(width),
        width * 0.5,
        height * 0.5,
        source="FALLBACK",
    )
    focal_x = float(camera.fx)
    focal_y = float(camera.fy)
    focal = math.sqrt(focal_x * focal_y)
    center_x = float(camera.cx)
    center_y = float(camera.cy)
    active = list(range(len(observations)))

    def solve(indices):
        rows = []
        targets = []
        weights = []
        for index in indices:
            point = observations[index]
            x, y, z = (float(value) for value in point.position)
            u = float(point.image[0]) * width - center_x
            v = float(point.image[1]) * height - center_y
            # u * (z + tz) = focal * (x + tx), and likewise for y.
            rows.extend(((u, -focal_x, 0.0), (v, 0.0, -focal_y)))
            targets.extend((focal_x * x - u * z, focal_y * y - v * z))
            weight = max(float(point.confidence), 0.05)
            weights.extend((weight, weight))
        matrix = np.asarray(rows, dtype=float)
        target = np.asarray(targets, dtype=float)
        root_weights = np.sqrt(np.asarray(weights, dtype=float))
        solution, _residuals, rank, singular = np.linalg.lstsq(
            matrix * root_weights[:, None],
            target * root_weights,
            rcond=None,
        )
        if rank < 3 or len(singular) < 3 or singular[-1] <= 1.0e-9:
            return None
        if singular[0] / singular[-1] > 1.0e7:
            return None
        return tuple(float(value) for value in solution)

    def errors(translation, indices):
        depth, translation_x, translation_y = translation
        result = []
        for index in indices:
            point = observations[index]
            x, y, z = (float(value) for value in point.position)
            denominator = z + depth
            if denominator <= 1.0e-6:
                result.append(float("inf"))
                continue
            projected_x = center_x + focal_x * (x + translation_x) / denominator
            projected_y = center_y + focal_y * (y + translation_y) / denominator
            observed_x = float(point.image[0]) * width
            observed_y = float(point.image[1]) * height
            result.append(
                math.hypot(projected_x - observed_x, projected_y - observed_y)
            )
        return result

    translation = solve(active)
    if translation is None:
        return None
    # MediaPipe's 3D and 2D heads can disagree around occlusion and frame
    # boundaries. Two bounded MAD passes remove isolated correspondences while
    # retaining at least six body points across the torso and legs.
    for _iteration in range(2):
        residuals = errors(translation, active)
        finite = [value for value in residuals if math.isfinite(value)]
        if len(finite) < 6:
            return None
        median = float(np.median(np.asarray(finite, dtype=float)))
        deviation = float(np.median(np.abs(np.asarray(finite, dtype=float) - median)))
        threshold = max(6.0, median + 3.5 * 1.4826 * deviation)
        retained = [
            index
            for index, error in zip(active, residuals, strict=True)
            if math.isfinite(error) and error <= threshold
        ]
        if len(retained) < 6 or retained == active:
            break
        active = retained
        translation = solve(active)
        if translation is None:
            return None

    depth, translation_x, translation_y = translation
    residuals = errors(translation, active)
    if not residuals or not all(math.isfinite(value) for value in residuals):
        return None
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    diagonal = math.hypot(width, height)
    point_depths = [float(observations[index].position[2]) + depth for index in active]
    valid = bool(
        math.isfinite(depth)
        and math.isfinite(translation_x)
        and math.isfinite(translation_y)
        and 0.2 <= depth <= 25.0
        and min(point_depths) > 0.05
        and rmse <= diagonal * 0.06
    )
    if not valid:
        return None
    mean_confidence = sum(
        float(observations[index].confidence) for index in active
    ) / len(active)
    fit_confidence = max(0.05, 1.0 - rmse / max(diagonal * 0.06, 1.0))
    retained_fraction = len(active) / len(observations)
    return RootMotionObservation(
        translation=(translation_x, translation_y, depth),
        focal_length=focal,
        scale=focal / depth,
        confidence=mean_confidence * fit_confidence * retained_fraction,
        valid=True,
    )


class MediaPipeProvider:
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
            raise FileNotFoundError(
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

    @staticmethod
    def _pose_score(world_pose) -> float:
        if not world_pose:
            return 0.0
        return sum(_confidence(item) for item in world_pose) / len(world_pose)

    def _convert_pose(
        self,
        result,
        frame: CanonicalFrame,
        *,
        landmark_threshold: float | None = None,
    ) -> None:
        if not result or not result.pose_world_landmarks:
            frame.warnings.append("No body pose detected")
            return
        indices = range(len(result.pose_world_landmarks))
        previous_hips = None
        if self._previous:
            left = self._previous.body.get("LEFT_HIP")
            right = self._previous.body.get("RIGHT_HIP")
            if left and right:
                previous_hips = (
                    (left.image[0] + right.image[0]) * 0.5,
                    (left.image[1] + right.image[1]) * 0.5,
                )
        if previous_hips is None:
            index = max(
                indices,
                key=lambda idx: self._pose_score(result.pose_world_landmarks[idx]),
            )
        else:
            # Maintain subject identity in multi-person video instead of
            # switching to whichever person has the highest confidence on
            # each individual frame.
            def identity_cost(idx):
                normalized = result.pose_landmarks[idx]
                left = normalized[23]
                right = normalized[24]
                center = (
                    (float(left.x) + float(right.x)) * 0.5,
                    (float(left.y) + float(right.y)) * 0.5,
                )
                distance = (center[0] - previous_hips[0]) ** 2 + (
                    center[1] - previous_hips[1]
                ) ** 2
                return distance - 0.02 * self._pose_score(
                    result.pose_world_landmarks[idx]
                )

            index = min(indices, key=identity_cost)
        world = result.pose_world_landmarks[index]
        normalized = result.pose_landmarks[index]
        threshold = (
            self.confidence if landmark_threshold is None else landmark_threshold
        )
        for idx, name in enumerate(POSE_LANDMARK_NAMES):
            frame.body[name] = _landmark(world[idx], normalized[idx], threshold)

        # MediaPipe occasionally assigns a low visibility score to an elbow
        # or knee that is plainly bounded by two reliable endpoints (common
        # with sleeves, crossed arms, and stylized art).  Dropping the entire
        # chain leaves a conspicuous rest/T-pose limb. Recover only the shared
        # hinge, never an endpoint, and only when its raw position forms a
        # finite, plausible two-segment chain. Confidence remains unchanged so
        # diagnostics and temporal cleanup can still treat it conservatively.
        recovery_floor = max(0.12, min(self.confidence * 0.75, 0.25))
        recovered = []
        for side in ("LEFT", "RIGHT"):
            for hinge, start, end in (
                ("ELBOW", "SHOULDER", "WRIST"),
                ("KNEE", "HIP", "ANKLE"),
            ):
                hinge_point = frame.body.get(f"{side}_{hinge}")
                start_point = frame.body.get(f"{side}_{start}")
                end_point = frame.body.get(f"{side}_{end}")
                if (
                    hinge_point is None
                    or hinge_point.valid
                    or hinge_point.confidence < recovery_floor
                    or start_point is None
                    or end_point is None
                    or not start_point.valid
                    or not end_point.valid
                ):
                    continue
                first = math.dist(start_point.position, hinge_point.position)
                second = math.dist(hinge_point.position, end_point.position)
                span = math.dist(start_point.position, end_point.position)
                shortest = min(first, second)
                if (
                    shortest < 1.0e-5
                    # World-landmark depth can strongly foreshorten a folded
                    # forearm. Retargeting uses direction, not the provider's
                    # segment length, so reject only truly extreme collapse.
                    or max(first, second) / shortest > 8.0
                    or span > (first + second) * 1.05
                ):
                    continue
                hinge_point.valid = True
                frame.auxiliary[f"RECOVERED_{side}_{hinge}"] = hinge_point
                recovered.append(f"{side.title()} {hinge.title()}")
        if recovered:
            frame.warnings.append(
                "Recovered low-confidence hinge positions: " + ", ".join(recovered)
            )
        derived = {
            "HIP_MID": (("LEFT_HIP",), ("RIGHT_HIP",)),
            "SHOULDER_MID": (
                ("LEFT_SHOULDER",),
                ("RIGHT_SHOULDER",),
            ),
            "HEAD_MID": (
                ("LEFT_EAR", "LEFT_EYE"),
                ("RIGHT_EAR", "RIGHT_EYE"),
            ),
            "LEFT_HAND_MID": (
                ("LEFT_INDEX",),
                ("LEFT_PINKY",),
            ),
            "RIGHT_HAND_MID": (
                ("RIGHT_INDEX",),
                ("RIGHT_PINKY",),
            ),
        }

        def landmark_valid(name):
            landmark = frame.body.get(name)
            if landmark is not None:
                return landmark.valid
            requirements = derived.get(name, ())
            return bool(requirements) and all(
                any(
                    frame.body.get(member) and frame.body[member].valid
                    for member in alternatives
                )
                for alternatives in requirements
            )

        skipped = []
        for semantic_id, (start_name, end_name) in BODY_JOINT_PAIRS.items():
            if not landmark_valid(start_name) or not landmark_valid(end_name):
                skipped.append(
                    semantic_id.replace("_L", " left")
                    .replace("_R", " right")
                    .replace("_", " ")
                    .title()
                )
        if skipped:
            visible = ", ".join(skipped[:6])
            remainder = len(skipped) - 6
            if remainder > 0:
                visible += f", and {remainder} more"
            frame.warnings.append(f"Low-confidence body segments skipped: {visible}")

    @staticmethod
    def _hand_palm_geometry(hand):
        required = (
            "WRIST",
            "INDEX_FINGER_MCP",
            "MIDDLE_FINGER_MCP",
            "PINKY_MCP",
        )
        if any(name not in hand or not hand[name].valid for name in required):
            return None
        wrist = tuple(float(value) for value in hand["WRIST"].position)
        index = tuple(float(value) for value in hand["INDEX_FINGER_MCP"].position)
        middle = tuple(float(value) for value in hand["MIDDLE_FINGER_MCP"].position)
        pinky = tuple(float(value) for value in hand["PINKY_MCP"].position)
        primary = tuple(middle[axis] - wrist[axis] for axis in range(3))
        lateral = tuple(index[axis] - pinky[axis] for axis in range(3))
        normal = (
            primary[1] * lateral[2] - primary[2] * lateral[1],
            primary[2] * lateral[0] - primary[0] * lateral[2],
            primary[0] * lateral[1] - primary[1] * lateral[0],
        )
        normal_length = math.sqrt(sum(value * value for value in normal))
        palm_length = math.dist(wrist, middle)
        palm_width = math.dist(index, pinky)
        if min(normal_length, palm_length, palm_width) <= 1.0e-8:
            return None
        return (
            tuple(value / normal_length for value in normal),
            palm_length,
            palm_width,
        )

    @staticmethod
    def _copy_hand(hand):
        return {
            name: Landmark(
                position=tuple(point.position),
                image=tuple(point.image),
                confidence=float(point.confidence),
                valid=bool(point.valid),
            )
            for name, point in hand.items()
        }

    @staticmethod
    def _hand_warning(frame: CanonicalFrame, side: str, reason: str) -> None:
        warning = f"{side.title()} detailed hand rejected: {reason}"
        if warning not in frame.warnings:
            frame.warnings.append(warning)

    @staticmethod
    def _normalized_dot(first, second) -> float | None:
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        if min(first_length, second_length) <= 1.0e-8:
            return None
        return sum(
            first_value * second_value
            for first_value, second_value in zip(first, second, strict=True)
        ) / (first_length * second_length)

    def _hand_pose_alignment(
        self,
        side: str,
        hand,
        frame: CanonicalFrame,
    ) -> float | None:
        required_hand = ("WRIST", "INDEX_FINGER_MCP", "PINKY_MCP")
        required_pose = (f"{side}_WRIST", f"{side}_INDEX", f"{side}_PINKY")
        if any(name not in hand or not hand[name].valid for name in required_hand):
            return None
        if any(
            name not in frame.body or not frame.body[name].valid
            for name in required_pose
        ):
            return None
        detailed = (
            hand["INDEX_FINGER_MCP"].image[0] - hand["PINKY_MCP"].image[0],
            hand["INDEX_FINGER_MCP"].image[1] - hand["PINKY_MCP"].image[1],
        )
        posed = (
            frame.body[f"{side}_INDEX"].image[0] - frame.body[f"{side}_PINKY"].image[0],
            frame.body[f"{side}_INDEX"].image[1] - frame.body[f"{side}_PINKY"].image[1],
        )
        return self._normalized_dot(detailed, posed)

    def _trust_hand(
        self,
        side: str,
        hand,
        frame: CanonicalFrame,
    ) -> bool:
        if self._hand_palm_geometry(hand) is None:
            self._hand_warning(frame, side, "insufficient valid palm landmarks")
            return False
        if len(hand) >= len(HAND_LANDMARK_NAMES):
            valid_count = sum(point.valid for point in hand.values())
            if valid_count < 17:
                self._hand_warning(
                    frame,
                    side,
                    f"only {valid_count} of 21 landmarks were reliable",
                )
                return False
        alignment = self._hand_pose_alignment(side, hand, frame)
        if alignment is not None and alignment < -0.35:
            self._hand_warning(
                frame,
                side,
                "finger ordering disagreed with the body-pose hand",
            )
            return False
        trusted_hands = getattr(self, "_trusted_hands", None)
        if trusted_hands is None:
            trusted_hands = self._trusted_hands = {}
        trusted_hands[side] = (frame.timestamp_ms, self._copy_hand(hand))
        return True

    def _hold_recent_hands(self, frame: CanonicalFrame) -> None:
        for side, entry in getattr(self, "_trusted_hands", {}).items():
            if side in frame.hands:
                continue
            timestamp, hand = entry
            elapsed = frame.timestamp_ms - timestamp
            if not 0 < elapsed <= 250:
                continue
            held = self._copy_hand(hand)
            for point in held.values():
                point.confidence *= max(0.25, 1.0 - elapsed / 333.0)
            frame.hands[side] = held
            warning = (
                f"{side.title()} detailed hand temporarily occluded; "
                "held the last trusted finger pose"
            )
            if warning not in frame.warnings:
                frame.warnings.append(warning)

    def _convert_hands(
        self,
        result,
        frame: CanonicalFrame,
        *,
        preserve_existing: bool = False,
    ) -> None:
        if not result or not result.hand_world_landmarks:
            if not frame.hands:
                frame.warnings.append("No hands detected")
            return
        candidates = []
        for idx, world in enumerate(result.hand_world_landmarks):
            normalized = result.hand_landmarks[idx]
            label = ""
            if idx < len(result.handedness) and result.handedness[idx]:
                category = result.handedness[idx][0]
                label = (
                    getattr(category, "category_name", None)
                    or getattr(category, "display_name", None)
                    or ""
                ).upper()
            hand = {
                name: _landmark(world[j], normalized[j], self.confidence)
                for j, name in enumerate(HAND_LANDMARK_NAMES)
            }
            candidates.append((label, hand))

        # Pose wrists are more reliable for associating anatomical side than the
        # handedness label when input media is mirrored. Match all visible
        # wrists globally: assigning LEFT first mislabels a single right hand.
        visible_sides = []
        distances = {}
        maximum_distances = {}
        for side in ("LEFT", "RIGHT"):
            if preserve_existing and side in frame.hands:
                continue
            wrist = frame.body.get(f"{side}_WRIST")
            if wrist is None or not wrist.valid:
                continue
            visible_sides.append(side)
            elbow = frame.body.get(f"{side}_ELBOW")
            forearm = (
                math.dist(wrist.image, elbow.image)
                if elbow is not None and elbow.valid
                else 0.0
            )
            # A detailed wrist must remain near the corresponding Pose wrist.
            # Scale the allowance with the observed forearm, but keep bounded
            # floors/ceilings for foreshortened and close-up subjects.
            maximum_distances[side] = min(max(forearm * 0.9, 0.06), 0.22) ** 2
            for idx, (_label, hand) in enumerate(candidates):
                distances[(side, idx)] = (
                    hand["WRIST"].image[0] - wrist.image[0]
                ) ** 2 + (hand["WRIST"].image[1] - wrist.image[1]) ** 2

        assigned = set()
        association_count = min(len(visible_sides), len(candidates))
        if association_count:
            # Exhaustive matching is tiny here (at most two hands) and avoids
            # the failure mode of greedy matching for crossed wrists. Reject
            # implausible pairs instead of forcing duplicate detections onto
            # opposite sides.
            valid_pairings = []
            while association_count and not valid_pairings:
                valid_pairings = [
                    pairs
                    for pairs in (
                        tuple(zip(sides, indices, strict=False))
                        for sides in permutations(visible_sides, association_count)
                        for indices in permutations(
                            range(len(candidates)), association_count
                        )
                    )
                    if all(
                        distances[(side, idx)] <= maximum_distances[side]
                        for side, idx in pairs
                    )
                ]
                association_count -= 1
            best_pairs = min(
                valid_pairings,
                key=lambda pairs: sum(distances[(side, idx)] for side, idx in pairs),
                default=(),
            )
            for side, idx in best_pairs:
                assigned.add(idx)
                hand = candidates[idx][1]
                if self._trust_hand(side, hand, frame):
                    frame.hands[side] = hand
        for idx, (label, hand) in enumerate(candidates):
            if idx in assigned:
                continue
            if visible_sides:
                # Pose supplied an anatomical reference but this candidate
                # failed its wrist-distance gate. Do not resurrect it from a
                # potentially mirrored handedness label.
                continue
            if label in {"LEFT", "RIGHT"} and not self.mirror:
                # MediaPipe handedness assumes selfie/mirrored input. Ordinary
                # images and decoded videos therefore need the labels swapped
                # whenever pose-wrist association is unavailable.
                label = "RIGHT" if label == "LEFT" else "LEFT"
            target = (
                label
                if label in {"LEFT", "RIGHT"}
                else ("LEFT" if "LEFT" not in frame.hands else "RIGHT")
            )
            if target not in frame.hands and self._trust_hand(target, hand, frame):
                frame.hands[target] = hand

    @staticmethod
    def _detailed_hands_are_duplicates(first, second) -> bool:
        shared = [
            name
            for name in HAND_LANDMARK_NAMES
            if name in first
            and name in second
            and first[name].valid
            and second[name].valid
        ]
        if len(shared) < 4:
            return False
        wrist_distance = math.dist(first["WRIST"].image, second["WRIST"].image)
        mean_distance = sum(
            math.dist(first[name].image, second[name].image) for name in shared
        ) / len(shared)
        return wrist_distance < 0.035 and mean_distance < 0.055

    def _assign_pose_guided_hands(
        self,
        candidates,
        frame: CanonicalFrame,
    ) -> None:
        if not candidates:
            return
        visible_sides = [
            side
            for side in ("LEFT", "RIGHT")
            if frame.body.get(f"{side}_WRIST") is not None
            and frame.body[f"{side}_WRIST"].valid
        ]
        if not visible_sides:
            return
        distances = {}
        maximum_distances = {}
        for side in visible_sides:
            wrist = frame.body[f"{side}_WRIST"]
            elbow = frame.body.get(f"{side}_ELBOW")
            forearm = (
                math.dist(wrist.image, elbow.image)
                if elbow is not None and elbow.valid
                else 0.0
            )
            maximum_distances[side] = min(max(forearm * 0.9, 0.06), 0.22) ** 2
            for index, (_source_side, hand) in enumerate(candidates):
                distances[(side, index)] = (
                    hand["WRIST"].image[0] - wrist.image[0]
                ) ** 2 + (hand["WRIST"].image[1] - wrist.image[1]) ** 2

        duplicate = bool(
            len(candidates) == 2
            and self._detailed_hands_are_duplicates(candidates[0][1], candidates[1][1])
        )
        association_count = min(len(visible_sides), len(candidates))
        if duplicate:
            association_count = 1
            warning = (
                "Overlapping hand crops resolved to the same detailed hand; "
                "kept only the closest body-pose wrist match"
            )
            if warning not in frame.warnings:
                frame.warnings.append(warning)
        valid_pairings = []
        while association_count and not valid_pairings:
            valid_pairings = [
                pairs
                for pairs in (
                    tuple(zip(sides, indices, strict=False))
                    for sides in permutations(visible_sides, association_count)
                    for indices in permutations(
                        range(len(candidates)), association_count
                    )
                )
                if all(
                    distances[(side, index)] <= maximum_distances[side]
                    for side, index in pairs
                )
            ]
            association_count -= 1
        best_pairs = min(
            valid_pairings,
            key=lambda pairs: sum(distances[(side, index)] for side, index in pairs),
            default=(),
        )
        for side, index in best_pairs:
            hand = candidates[index][1]
            if self._trust_hand(side, hand, frame):
                frame.hands[side] = hand

    def _pose_guided_hand_crop(
        self,
        bgr,
        frame: CanonicalFrame,
        side: str,
    ):
        wrist = frame.body.get(f"{side}_WRIST")
        if wrist is None or not wrist.valid:
            return None
        source = self.cv2.flip(bgr, 1) if self.mirror else bgr
        height, width = source.shape[:2]
        wrist_xy = self.np.asarray(
            (float(wrist.image[0]) * width, float(wrist.image[1]) * height),
            dtype=float,
        )
        if not self.np.isfinite(wrist_xy).all():
            return None

        elbow = frame.body.get(f"{side}_ELBOW")
        forearm = 0.0
        if elbow is not None and elbow.valid:
            elbow_xy = self.np.asarray(
                (float(elbow.image[0]) * width, float(elbow.image[1]) * height),
                dtype=float,
            )
            if self.np.isfinite(elbow_xy).all():
                forearm = float(self.np.linalg.norm(wrist_xy - elbow_xy))

        tips = []
        for name in ("THUMB", "INDEX", "PINKY"):
            point = frame.body.get(f"{side}_{name}")
            if point is None or not point.valid:
                continue
            value = self.np.asarray(
                (float(point.image[0]) * width, float(point.image[1]) * height),
                dtype=float,
            )
            if self.np.isfinite(value).all():
                tips.append(value)

        center = wrist_xy.copy()
        hand_span = 0.0
        if tips:
            tip_center = self.np.mean(self.np.stack(tips), axis=0)
            candidate_span = max(
                float(self.np.linalg.norm(point - wrist_xy)) for point in tips
            )
            # Ignore a wildly displaced Pose fingertip rather than allowing it
            # to drag the crop away from the reliable wrist and elbow chain.
            if forearm <= 1.0e-6 or candidate_span <= forearm * 1.25:
                hand_span = candidate_span
                center = (wrist_xy + tip_center) * 0.5
        elif forearm > 1.0e-6:
            center = wrist_xy + (wrist_xy - elbow_xy) * 0.18

        maximum_side = max(32, int(round(min(width, height) * 0.45)))
        minimum_side = min(maximum_side, max(96, int(round(min(width, height) * 0.1))))
        side_length = int(
            round(
                min(
                    max(hand_span * 2.1, forearm * 1.05, minimum_side),
                    maximum_side,
                )
            )
        )
        if side_length < 32:
            return None
        x0 = max(0, min(width - side_length, int(round(center[0] - side_length * 0.5))))
        y0 = max(
            0, min(height - side_length, int(round(center[1] - side_length * 0.5)))
        )
        x1 = min(width, x0 + side_length)
        y1 = min(height, y0 + side_length)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return source[y0:y1, x0:x1], (x0, y0, x1, y1), (width, height)

    def _detect_hand_crop(
        self,
        bgr,
        frame: CanonicalFrame,
        side: str,
        timestamp_ms: int,
    ):
        prepared = self._pose_guided_hand_crop(bgr, frame, side)
        if prepared is None:
            return None
        crop, (x0, y0, x1, y1), (width, height) = prepared
        result = self._detect_task(
            self._hand_crop_task(side),
            self._mp_image(crop, apply_mirror=False),
            timestamp_ms,
        )
        if not result or not result.hand_world_landmarks:
            return None
        crop_width = float(x1 - x0)
        crop_height = float(y1 - y0)
        pose_wrist = frame.body[f"{side}_WRIST"]
        best = None
        for index, world in enumerate(result.hand_world_landmarks):
            normalized = result.hand_landmarks[index]
            wrist_x = (x0 + float(normalized[0].x) * crop_width) / width
            wrist_y = (y0 + float(normalized[0].y) * crop_height) / height
            crop_distance = math.hypot(
                (wrist_x - pose_wrist.image[0]) * width,
                (wrist_y - pose_wrist.image[1]) * height,
            ) / max(crop_width, crop_height, 1.0)
            if crop_distance > 0.45:
                continue
            hand = {
                name: _landmark(
                    world[j],
                    SimpleNamespace(
                        x=(x0 + float(normalized[j].x) * crop_width) / width,
                        y=(y0 + float(normalized[j].y) * crop_height) / height,
                        z=float(normalized[j].z) * crop_width / width,
                    ),
                    self.confidence,
                )
                for j, name in enumerate(HAND_LANDMARK_NAMES)
            }
            candidate = (crop_distance, hand)
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best[1] if best is not None else None

    @staticmethod
    def _has_pose(result) -> bool:
        return bool(result and result.pose_world_landmarks and result.pose_landmarks)

    @staticmethod
    def _has_face(result) -> bool:
        return bool(result and result.face_landmarks)

    def _detect_task(self, task, image, timestamp_ms: int):
        if self.video_mode:
            return task.detect_for_video(image, timestamp_ms)
        return task.detect(image)

    def _pose_guided_face_crop(self, bgr, frame: CanonicalFrame):
        nose = frame.body.get("NOSE")
        left_shoulder = frame.body.get("LEFT_SHOULDER")
        right_shoulder = frame.body.get("RIGHT_SHOULDER")
        if nose is None or left_shoulder is None or right_shoulder is None:
            return None
        source = self.cv2.flip(bgr, 1) if self.mirror else bgr
        height, width = source.shape[:2]
        values = (*nose.image, *left_shoulder.image, *right_shoulder.image)
        if not all(math.isfinite(float(value)) for value in values):
            return None
        shoulder_width = (
            abs(float(right_shoulder.image[0]) - float(left_shoulder.image[0])) * width
        )
        max_side = min(width, height)
        side = int(
            round(
                min(
                    max(shoulder_width * 1.1, max_side * 0.12, 128.0),
                    max_side * 0.6,
                )
            )
        )
        if side < 32:
            return None
        center_x = float(nose.image[0]) * width
        center_y = float(nose.image[1]) * height + side * 0.02
        x0 = max(0, min(width - side, int(round(center_x - side * 0.5))))
        y0 = max(0, min(height - side, int(round(center_y - side * 0.5))))
        x1 = min(width, x0 + side)
        y1 = min(height, y0 + side)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return source[y0:y1, x0:x1], (x0, y0, x1, y1), (width, height)

    def _detect_face_crop(
        self,
        bgr,
        frame: CanonicalFrame,
        *,
        fallback: bool,
    ):
        prepared = self._pose_guided_face_crop(bgr, frame)
        if prepared is None:
            return None
        crop, (x0, y0, x1, y1), (width, height) = prepared
        result = self._face_crop_task(fallback=fallback).detect(
            self._mp_image(crop, apply_mirror=False)
        )
        if not self._has_face(result):
            return result
        crop_width = float(x1 - x0)
        crop_height = float(y1 - y0)
        remapped_faces = [
            [
                SimpleNamespace(
                    x=(x0 + float(item.x) * crop_width) / width,
                    y=(y0 + float(item.y) * crop_height) / height,
                    z=float(item.z) * crop_width / width,
                )
                for item in face
            ]
            for face in result.face_landmarks
        ]
        return SimpleNamespace(
            face_landmarks=remapped_faces,
            face_blendshapes=list(getattr(result, "face_blendshapes", ()) or ()),
        )

    @staticmethod
    def _face_bounds(face) -> tuple[float, float, float, float]:
        xs = [float(point.x) for point in face]
        ys = [float(point.y) for point in face]
        return min(xs), min(ys), max(xs), max(ys)

    def _best_face_index(self, result, frame: CanonicalFrame) -> int:
        faces = result.face_landmarks
        if len(faces) == 1:
            return 0
        reference = None
        nose = frame.body.get("NOSE")
        if nose is not None:
            reference = nose.image
        elif self._previous and len(self._previous.face_landmarks) > 1:
            reference = self._previous.face_landmarks[1].image

        def suitability(index: int):
            face = faces[index]
            left, top, right, bottom = self._face_bounds(face)
            area = max(right - left, 1.0e-6) * max(bottom - top, 1.0e-6)
            face_nose = face[1]
            if reference is not None:
                distance = (float(face_nose.x) - reference[0]) ** 2 + (
                    float(face_nose.y) - reference[1]
                ) ** 2
                return distance / max(area, 1.0e-6), -area
            center_distance = ((left + right) * 0.5 - 0.5) ** 2 + (
                (top + bottom) * 0.5 - 0.5
            ) ** 2
            return -area, center_distance

        return min(range(len(faces)), key=suitability)

    def _convert_face(self, result, frame: CanonicalFrame) -> None:
        if not result or not result.face_landmarks:
            if self.use_face:
                frame.warnings.append("No face detected")
            return
        index = self._best_face_index(result, frame)
        frame.face_overlay = [
            (float(item.x), float(item.y)) for item in result.face_landmarks[index]
        ]
        frame.face_landmarks = [
            Landmark(
                position=(float(item.x), float(item.y), float(item.z)),
                image=(float(item.x), float(item.y)),
                confidence=1.0,
                valid=True,
            )
            for item in result.face_landmarks[index]
        ]
        if index < len(result.face_blendshapes):
            for category in result.face_blendshapes[index]:
                name = (
                    getattr(category, "category_name", None)
                    or getattr(category, "display_name", None)
                    or ""
                )
                if name:
                    frame.blendshapes[name] = float(category.score)

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
