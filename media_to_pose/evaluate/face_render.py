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
from ..retarget.solve.face_values import FaceValueReader, mapped_expressions
from ..rig import mapping
from ..rig.face_schemes import associated_meshes
from ..rig.face_space import eye_position, face_frame

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


def _face_points(armature, center: Vector, radius: float) -> list[Vector]:
    """Rest vertices of the rig's meshes within ``radius`` of ``center``.

    In armature space; a coarse sample is enough to frame the face.
    """
    to_armature = armature.matrix_world.inverted()
    points = []
    for mesh in associated_meshes(armature):
        vertices = mesh.data.vertices
        step = max(len(vertices) // 4000, 1)
        transform = to_armature @ mesh.matrix_world
        for index in range(0, len(vertices), step):
            point = transform @ vertices[index].co
            if (point - center).length <= radius:
                points.append(point)
    return points


def face_camera(scene, armature):
    """A temporary camera in front of the rig's face, framing the head.

    The shot is sized from the face mesh around the eyes when there is one
    (eye bones can sit deep inside the head, close together), else from the
    eye distance or the head bone.
    """
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    space = face_frame(armature, entries)
    world = armature.matrix_world
    eye_left = _mapped_bone(armature, entries, "EYE_L")
    eye_right = _mapped_bone(armature, entries, "EYE_R")
    head = _mapped_bone(armature, entries, "HEAD")
    head_length = head.bone.length if head is not None else 0.0
    if eye_left is not None and eye_right is not None:
        eyes = (
            eye_position(armature, eye_left) + eye_position(armature, eye_right)
        ) * 0.5
    elif head is not None:
        eyes = (head.bone.head_local + head.bone.tail_local) * 0.5
    else:
        eyes = Vector()
    reach = max(space.scale * 2.0, head_length * 1.3, 0.05)
    points = _face_points(armature, eyes, reach * 1.25)
    if len(points) >= 8:
        # Extents along the face's own axes.
        across = [point.dot(space.lateral) for point in points]
        along = [point.dot(space.up) for point in points]
        depth = [point.dot(space.forward) for point in points]
        # The middle of the region's back plane, seen along the face's forward.
        middle = (
            space.lateral * (max(across) + min(across)) * 0.5
            + space.up * (max(along) + min(along)) * 0.5
            + space.forward * min(depth)
        )
        size = max(max(across) - min(across), max(along) - min(along)) * 1.15
        front = max(depth) - min(depth)
        center = middle
    else:
        center = eyes - space.up * (space.scale * 0.5)
        size = reach * 2.0
        front = space.scale
    data = bpy.data.cameras.new("MediaToPose_FaceComparison")
    data.lens = 50.0
    camera = bpy.data.objects.new("MediaToPose_FaceComparison", data)
    scene.collection.objects.link(camera)
    half_angle = math.atan(data.sensor_width / (2.0 * data.lens))
    # From the back of the framed region: the face's front sits ``front``
    # closer, so the whole face width fits at the front plane.
    distance = (size / 2.0) / math.tan(half_angle) + front
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
        bar_width = size - label_width - 44
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
                name.removeprefix("FACE_")[:20],
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
                # Negative values (brows down, mouth corners down) are hollow.
                cv2.rectangle(
                    panel,
                    (label_width, y),
                    (label_width + max(length, 1), y + max(int(row * 0.3), 2)),
                    color,
                    -1 if value >= 0.0 else 1,
                )
                cv2.putText(
                    panel,
                    f"{value:+.2f}",
                    (label_width + bar_width + 4, y + max(int(row * 0.3), 2)),
                    font,
                    0.33,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return panel


__all__ = ["ExpressionBars", "face_camera", "face_crop"]
