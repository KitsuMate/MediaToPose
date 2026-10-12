"""MediaPipe Pose landmarks to canonical body joints and root motion."""

from __future__ import annotations

import math

from ... import camera_calibration
from ...constants import (
    BODY_JOINT_PAIRS,
    POSE_LANDMARK_NAMES,
)
from ...models import CanonicalFrame, Landmark, RootMotionObservation

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


class PoseMixin:
    """Pose conversion for :class:`MediaPipeProvider`."""

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
    def _has_pose(result) -> bool:
        return bool(result and result.pose_world_landmarks and result.pose_landmarks)
