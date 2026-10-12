"""Face close-ups for ``render_comparison(face=True)``.

The source is cropped to the detected face, the camera looks the rig's face
in the eye, and a bar chart compares the strongest expressions as detected
(after the mapping's gain and clamp) with what the rig shows.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

from ..core.requests import FaceOutput
from ..core.types import CanonicalFrame
from ..retarget.solve.common import _mapped_bone
from ..retarget.solve.face import _eye_position, face_frame
from ..retarget.solve.face_values import FaceValueReader, mapped_expressions
from ..rig import mapping

BAR_COUNT = 10
# Space around the detected face box, as a share of its size.
FACE_PADDING = 0.35
DETECTED_COLOR = (80, 240, 80)
RIG_COLOR = (40, 140, 255)


def face_crop(image, frame: CanonicalFrame | None, cv2):
    """``image`` cropped to the detected face with its landmarks dotted."""
    # Retained detections keep only the face's image points (the overlay).
    points = []
    if frame is not None:
        points = [lm.image for lm in frame.face_landmarks if lm.valid] or list(
            frame.face_overlay
        )
    height, width = image.shape[:2]
    if not points:
        return image
    xs = [float(point[0]) * width for point in points]
    ys = [float(point[1]) * height for point in points]
    for x, y in zip(xs, ys, strict=True):
        cv2.circle(image, (int(x), int(y)), 1, DETECTED_COLOR, -1, cv2.LINE_AA)
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1.0 + 2.0 * FACE_PADDING)
    center_x = (max(xs) + min(xs)) / 2.0
    center_y = (max(ys) + min(ys)) / 2.0
    left = int(max(center_x - side / 2.0, 0))
    top = int(max(center_y - side / 2.0, 0))
    right = int(min(center_x + side / 2.0, width))
    bottom = int(min(center_y + side / 2.0, height))
    if right - left < 8 or bottom - top < 8:
        return image
    return image[top:bottom, left:right].copy()


def face_camera(scene, armature):
    """A temporary camera in front of the rig's face, framing the head."""
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    space = face_frame(armature, entries)
    world = armature.matrix_world
    eye_left = _mapped_bone(armature, entries, "EYE_L")
    eye_right = _mapped_bone(armature, entries, "EYE_R")
    head = _mapped_bone(armature, entries, "HEAD")
    if eye_left is not None and eye_right is not None:
        eyes = (
            _eye_position(armature, eye_left) + _eye_position(armature, eye_right)
        ) * 0.5
        # The face spans about four eye distances; its middle is below the eyes.
        center = eyes - space.up * (space.scale * 0.5)
        size = space.scale * 4.0
    elif head is not None:
        center = (head.head + head.tail) * 0.5
        size = max(head.length * 2.0, 0.1)
    else:
        center = Vector()
        size = 0.3
    data = bpy.data.cameras.new("MediaToPose_FaceComparison")
    data.lens = 50.0
    camera = bpy.data.objects.new("MediaToPose_FaceComparison", data)
    scene.collection.objects.link(camera)
    half_angle = math.atan(data.sensor_width / (2.0 * data.lens))
    distance = (size / 2.0) / math.tan(half_angle)
    world_center = world @ center
    forward = (world.to_3x3() @ space.forward).normalized()
    camera.location = world_center + forward * distance * world.median_scale
    # The camera looks along its -Z with the face's up as the image's up.
    up = world.to_3x3() @ space.up
    up = (up - forward * up.dot(forward)).normalized()
    basis = Matrix((forward.cross(up) * -1.0, up, forward)).transposed()
    camera.rotation_euler = basis.to_euler()
    data.clip_start = max(distance * 0.05, 0.001)
    return camera


class ExpressionBars:
    """Draw detected and rig values of the strongest expressions."""

    def __init__(self, armature, face_output: FaceOutput):
        self.entries = mapping.mapping_by_id(armature.mtp_rig_config)
        self.face_output = FaceOutput(face_output)
        self.reader = FaceValueReader(armature, self.entries, self.face_output)

    def draw(self, size: int, frame: CanonicalFrame | None, cv2, np):
        panel = np.full((size, size, 3), 24, np.uint8)
        detected = (
            mapped_expressions(self.entries, dict(frame.blendshapes), self.face_output)[
                0
            ]
            if frame is not None and frame.blendshapes
            else {}
        )
        shown = self.reader.read()
        names = sorted(
            set(detected) | set(shown),
            key=lambda name: -max(
                abs(detected.get(name, 0.0)), abs(shown.get(name, 0.0))
            ),
        )[:BAR_COUNT]
        row = (size - 40) / max(len(names), 1)
        label_width = int(size * 0.45)
        bar_width = size - label_width - 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            panel,
            "detected",
            (label_width, 18),
            font,
            0.45,
            DETECTED_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel, "rig", (label_width + 80, 18), font, 0.45, RIG_COLOR, 1, cv2.LINE_AA
        )
        for index, name in enumerate(names):
            top = int(30 + index * row)
            cv2.putText(
                panel,
                name[:22],
                (6, top + int(row * 0.6)),
                font,
                0.4,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            for offset, value, color in (
                (0.15, detected.get(name, 0.0), DETECTED_COLOR),
                (0.5, shown.get(name, 0.0), RIG_COLOR),
            ):
                y = top + int(row * offset)
                length = int(min(abs(value), 1.0) * bar_width)
                cv2.rectangle(
                    panel,
                    (label_width, y),
                    (label_width + max(length, 1), y + max(int(row * 0.3), 2)),
                    color,
                    -1,
                )
        return panel


__all__ = ["ExpressionBars", "face_camera", "face_crop"]
