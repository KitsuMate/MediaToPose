"""MediaPipe face landmarks and blendshapes, with pose-guided face crops."""

from __future__ import annotations

import math
from types import SimpleNamespace

from ...models import CanonicalFrame, Landmark


class FaceMixin:
    """Face detection and conversion for :class:`MediaPipeProvider`."""

    @staticmethod
    def _has_face(result) -> bool:
        return bool(result and result.face_landmarks)

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
