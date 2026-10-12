"""Side-by-side images of the source, its detections and the posed rig.

``render_comparison`` writes one PNG contact sheet. Each row is a timeline
frame: on the left the source frame with the detected skeleton drawn in
green, on the right the rig from a camera framed on it, rendered with
Workbench when the rig has meshes, with its mapped bones drawn in orange.
With ``face=True`` each row is a face close-up instead: the source cropped
to the detected face, the rig's face, and a chart of the strongest
expressions as detected and as the rig shows them (see ``face_render``).
The scene's render settings, camera and frame are restored afterwards.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from ..core.constants import BODY_CONNECTIONS, HAND_CONNECTIONS
from ..core.dependencies import import_runtime_dependencies
from ..core.errors import InvalidRequest
from ..core.media_io import decode_image, decode_video_frame_index_with_timestamp
from ..core.scene import SceneContext
from ..core.types import CanonicalFrame
from ..rig import mapping
from .face_render import ExpressionBars, face_camera, face_crop

RIG_CONNECTIONS = (
    ("PELVIS", "SPINE_01"),
    ("SPINE_01", "SPINE_02"),
    ("SPINE_02", "CHEST"),
    ("CHEST", "NECK"),
    ("NECK", "HEAD"),
)


def _rig_bones(armature) -> list:
    entries = mapping.mapping_by_id(armature.mtp_rig_config)
    return [
        bone
        for entry in entries.values()
        if entry.section == "BODY"
        and entry.enabled
        and entry.target_kind == "BONE"
        and entry.target_name
        and (bone := armature.pose.bones.get(entry.target_name)) is not None
    ]


def _bounds(points: Sequence[Vector]) -> tuple[Vector, float]:
    low = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    high = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return (low + high) * 0.5, max((high - low).length, 0.5)


def _frame_camera(scene, armature, bones, frames: Sequence[int]):
    """A temporary camera in front of the rig that sees it in every frame."""
    world = armature.matrix_world
    points = []
    for frame in frames:
        scene.frame_set(frame)
        points.extend(world @ bone.head for bone in bones)
        points.extend(world @ bone.tail for bone in bones)
    if not points:
        points = [world.translation.copy()]
    center, size = _bounds(points)
    data = bpy.data.cameras.new("MediaToPose_Comparison")
    data.lens = 50.0
    camera = bpy.data.objects.new("MediaToPose_Comparison", data)
    scene.collection.objects.link(camera)
    forward = (world.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()
    distance = size * 1.6 + 1.0
    camera.location = center + forward * distance
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return camera


def _draw_detection(image, frame: CanonicalFrame | None, cv2) -> None:
    if frame is None:
        return
    height, width = image.shape[:2]

    def point(landmark):
        return (
            int(round(float(landmark.image[0]) * width)),
            int(round(float(landmark.image[1]) * height)),
        )

    def connect(points, pairs, color):
        for first, second in pairs:
            a, b = points.get(first), points.get(second)
            if a is not None and b is not None and a.valid and b.valid:
                cv2.line(image, point(a), point(b), color, 2, cv2.LINE_AA)

    connect(frame.body, BODY_CONNECTIONS, (80, 240, 80))
    for hand in frame.hands.values():
        connect(hand, HAND_CONNECTIONS, (240, 200, 60))


def _source_image(media_path: str, kind: str, source_frame: int, mirror: bool, cv2, np):
    if kind == "IMAGE":
        image = decode_image(media_path, cv2, np)
    else:
        image, _timestamp = decode_video_frame_index_with_timestamp(
            media_path, max(int(source_frame), 0), cv2
        )
    if mirror:
        image = cv2.flip(image, 1)
    return image


def _fit(image, size: int, cv2, np):
    height, width = image.shape[:2]
    scale = size / max(height, width, 1)
    resized = cv2.resize(
        image, (max(int(width * scale), 1), max(int(height * scale), 1))
    )
    canvas = np.full((size, size, 3), 32, np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _render_rig(scene, size: int, path: Path, render_still, cv2, np):
    """Workbench render of the visible meshes, or a blank canvas."""
    has_mesh = any(obj.type == "MESH" and obj.visible_get() for obj in scene.objects)
    if not has_mesh or render_still is None:
        return np.full((size, size, 3), 48, np.uint8)
    scene.render.filepath = str(path)
    render_still(scene)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image if image is not None else np.full((size, size, 3), 48, np.uint8)


def _draw_rig(image, scene, camera, armature, bones, cv2) -> None:
    size = image.shape[0]
    world = armature.matrix_world

    def project(position):
        coordinate = world_to_camera_view(scene, camera, world @ position)
        return int(round(coordinate.x * size)), int(round((1.0 - coordinate.y) * size))

    for bone in bones:
        cv2.line(
            image,
            project(bone.head),
            project(bone.tail),
            (40, 140, 255),
            2,
            cv2.LINE_AA,
        )


def render_comparison(
    scene_context: SceneContext,
    armature,
    path: str | Path,
    rows: Sequence[tuple[int, int, CanonicalFrame | None]],
    *,
    media_path: str,
    media_kind: str,
    mirror: bool = False,
    size: int = 384,
    render_still: Callable | None = None,
    face_output: str | None = None,
) -> Path:
    """Write the contact sheet; ``rows`` are ``(timeline, source, frame)``.

    ``render_still(scene)`` renders the scene's camera to its output path;
    without it the rig column shows only the projected bones. A
    ``face_output`` makes the face close-up sheet for that face output.
    """
    if not rows:
        raise InvalidRequest("No frames to render", parameter="frames")
    cv2, _mp, np = import_runtime_dependencies()
    scene = scene_context.scene
    bars = ExpressionBars(armature, face_output) if face_output else None
    bones = [] if bars is not None else _rig_bones(armature)
    render = scene.render
    saved = {
        "engine": render.engine,
        "resolution_x": render.resolution_x,
        "resolution_y": render.resolution_y,
        "resolution_percentage": render.resolution_percentage,
        "filepath": render.filepath,
        "film_transparent": render.film_transparent,
    }
    saved_format = render.image_settings.file_format
    saved_camera = scene.camera
    saved_frame = (scene.frame_current, scene.frame_subframe)
    camera = None
    sheet_rows = []
    try:
        camera = (
            face_camera(scene, armature)
            if bars is not None
            else _frame_camera(scene, armature, bones, [row[0] for row in rows])
        )
        scene.camera = camera
        render.engine = "BLENDER_WORKBENCH"
        render.resolution_x = render.resolution_y = int(size)
        render.resolution_percentage = 100
        render.film_transparent = False
        render.image_settings.file_format = "PNG"
        with tempfile.TemporaryDirectory(prefix="mtp-compare-") as temporary:
            for timeline_frame, source_frame, detected in rows:
                source = _source_image(
                    media_path, media_kind, source_frame, mirror, cv2, np
                )
                if bars is not None:
                    source = face_crop(source, detected, cv2)
                else:
                    _draw_detection(source, detected, cv2)
                scene.frame_set(int(timeline_frame))
                rig = _render_rig(
                    scene,
                    size,
                    Path(temporary) / f"rig_{int(timeline_frame):06d}.png",
                    render_still,
                    cv2,
                    np,
                )
                rig = _fit(rig, size, cv2, np)
                _draw_rig(rig, scene, camera, armature, bones, cv2)
                columns = [_fit(source, size, cv2, np), rig]
                if bars is not None:
                    columns.append(bars.draw(size, detected, cv2, np))
                row = np.concatenate(columns, axis=1)
                cv2.putText(
                    row,
                    f"frame {timeline_frame}  source {source_frame}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                sheet_rows.append(row)
    finally:
        for name, value in saved.items():
            setattr(render, name, value)
        render.image_settings.file_format = saved_format
        scene.camera = saved_camera
        if camera is not None:
            data = camera.data
            bpy.data.objects.remove(camera)
            bpy.data.cameras.remove(data)
        scene.frame_set(saved_frame[0], subframe=saved_frame[1])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.concatenate(sheet_rows, axis=0))
    if not ok:
        raise InvalidRequest(f"Could not encode {destination}")
    encoded.tofile(str(destination))
    return destination


__all__ = ["render_comparison"]
