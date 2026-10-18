"""Face close-ups for ``render_comparison(face=True)``.

The source is cropped to the detected face and the camera looks the rig's
face in the eye, following the posed head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import bpy
from mathutils import Matrix, Vector

from ..core.types import CanonicalFrame
from ..rig import mapping
from ..rig.face_schemes import associated_meshes
from ..rig.face_space import FaceFrame, eye_position, face_frame, mapped_bone

# Space around the detected face box, as a share of its size.
FACE_PADDING = 0.35
DETECTED_COLOR = (80, 240, 80)


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


@dataclass(frozen=True)
class FaceShot:
    """The rest-pose face region a face camera frames, in armature space."""

    space: FaceFrame
    # The head bone the camera follows; empty without one.
    head: str
    center: Vector
    size: float
    front: float


def face_shot(armature) -> FaceShot:
    """Frame the face at rest.

    The shot is sized from the face mesh around the eyes when there is one
    (eye bones can sit deep inside the head, close together), else from the
    eye distance or the head bone.
    """
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    space = face_frame(armature, entries)
    eye_left = mapped_bone(armature, entries, "EYE_L")
    eye_right = mapped_bone(armature, entries, "EYE_R")
    head = mapped_bone(armature, entries, "HEAD")
    head_length = head.bone.length if head is not None else 0.0
    head_name = head.name if head is not None else ""
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
    if len(points) < 8:
        center = eyes - space.up * (space.scale * 0.5)
        return FaceShot(space, head_name, center, reach * 2.0, space.scale)
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
    return FaceShot(space, head_name, middle, size, max(depth) - min(depth))


def face_camera(scene, armature, camera=None, shot: FaceShot | None = None):
    """A temporary camera in front of the rig's face, framing the head.

    The face is framed at rest (``face_shot``) and the camera follows the
    posed head. Reuse ``camera`` and ``shot`` to follow the head after
    advancing the scene frame.
    """
    shot = shot or face_shot(armature)
    space = shot.space
    world = armature.matrix_world
    head = armature.pose.bones.get(shot.head) if shot.head else None
    if head is not None and armature.data.pose_position == "POSE":
        world = world @ head.matrix @ head.bone.matrix_local.inverted()
    if camera is None:
        data = bpy.data.cameras.new("MediaToPose_FaceComparison")
        camera = bpy.data.objects.new("MediaToPose_FaceComparison", data)
        scene.collection.objects.link(camera)
    data = camera.data
    data.lens = 50.0
    half_angle = math.atan(data.sensor_width / (2.0 * data.lens))
    # From the back of the framed region: the face's front sits ``front``
    # closer, so the whole face width fits at the front plane.
    distance = (shot.size / 2.0) / math.tan(half_angle) + shot.front
    world_center = world @ shot.center
    forward = (world.to_3x3() @ space.forward).normalized()
    camera.location = world_center + forward * distance * world.median_scale
    # The camera looks along its -Z with the face's up as the image's up.
    up = world.to_3x3() @ space.up
    up = (up - forward * up.dot(forward)).normalized()
    basis = Matrix((forward.cross(up) * -1.0, up, forward)).transposed()
    camera.rotation_euler = basis.to_euler()
    data.clip_start = max(distance * 0.05, 0.001)
    return camera


__all__ = ["FaceShot", "face_camera", "face_crop", "face_shot"]
