"""MediaPipe hand landmarks: trust checks, pose-guided crops and assignment."""

from __future__ import annotations

import math
from itertools import permutations
from types import SimpleNamespace

from ...core.constants import HAND_LANDMARK_NAMES
from ...core.types import CanonicalFrame, Landmark
from .body import _landmark


class HandsMixin:
    """Hand detection and conversion for :class:`MediaPipeProvider`."""

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
