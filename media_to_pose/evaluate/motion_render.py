"""Pictures of generated motion: a contact sheet or an MP4 of the rig.

Both use a temporary camera at a three-quarter view that sees the rig over
the whole range, and Workbench. Guide frames are marked on the sheet. The
scene's render settings, camera and frame are restored afterwards.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from ..core.dependencies import import_runtime_dependencies
from ..core.errors import InvalidRequest
from ..core.scene import SceneContext
from .render import _bounds, _draw_rig, _fit, _render_rig, _rig_bones

VIEW_ANGLE_DEG = 35.0


class _FollowCamera:
    """A temporary camera at a three-quarter view that follows the rig's
    (smoothed) centre, at a distance that fits the rig's largest pose."""

    def __init__(self, scene, armature, bones, frames: Sequence[int]):
        world = armature.matrix_world
        centers, size = {}, 0.5
        for frame in frames:
            scene.frame_set(int(frame))
            points = [world @ bone.head for bone in bones]
            points += [world @ bone.tail for bone in bones]
            if not points:
                points = [world.translation.copy()]
            center, extent = _bounds(points)
            centers[int(frame)] = center
            size = max(size, extent)
        window = 6  # frames either side; a sheet's sparse frames stay apart

        def smoothed(frame: int) -> Vector:
            near = [centers[other] for other in centers if abs(other - frame) <= window]
            return sum(near, Vector()) / len(near)

        self.centers = {frame: smoothed(frame) for frame in centers}
        data = bpy.data.cameras.new("MediaToPose_Motion")
        data.lens = 45.0
        self.camera = bpy.data.objects.new("MediaToPose_Motion", data)
        scene.collection.objects.link(self.camera)
        turn = Matrix.Rotation(math.radians(VIEW_ANGLE_DEG), 3, "Z")
        forward = (turn @ world.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()
        self.offset = forward * (size * 1.6 + 0.6) + Vector((0.0, 0.0, size * 0.12))
        self.camera.rotation_euler = (-self.offset).to_track_quat("-Z", "Y").to_euler()

    def follow(self, frame: int, *, key: bool = False) -> None:
        nearest = min(self.centers, key=lambda other: abs(other - int(frame)))
        self.camera.location = self.centers[nearest] + self.offset
        if key:
            self.camera.keyframe_insert("location", frame=int(frame))

    def remove(self) -> None:
        animation = self.camera.animation_data
        action = animation.action if animation else None
        data = self.camera.data
        bpy.data.objects.remove(self.camera)
        bpy.data.cameras.remove(data)
        if action is not None and action.users == 0:
            bpy.data.actions.remove(action)


class _Settings:
    """Saves and restores what the renders change."""

    def __init__(self, scene):
        self.scene = scene
        render = scene.render
        self.values = {
            name: getattr(render, name)
            for name in (
                "engine",
                "resolution_x",
                "resolution_y",
                "resolution_percentage",
                "filepath",
                "film_transparent",
                "fps",
                "fps_base",
                "use_file_extension",
            )
        }
        self.media_type = render.image_settings.media_type
        self.file_format = render.image_settings.file_format
        self.color_type = scene.display.shading.color_type
        self.camera = scene.camera
        self.frames = (scene.frame_start, scene.frame_end)
        self.current = (scene.frame_current, scene.frame_subframe)

    def restore(self) -> None:
        scene, render = self.scene, self.scene.render
        render.image_settings.media_type = self.media_type
        render.image_settings.file_format = self.file_format
        for name, value in self.values.items():
            setattr(render, name, value)
        scene.display.shading.color_type = self.color_type
        scene.camera = self.camera
        scene.frame_start, scene.frame_end = self.frames
        scene.frame_set(self.current[0], subframe=self.current[1])


def _prepare(scene, camera, width: int, height: int) -> None:
    render = scene.render
    scene.camera = camera
    render.engine = "BLENDER_WORKBENCH"
    render.resolution_x, render.resolution_y = int(width), int(height)
    render.resolution_percentage = 100
    render.film_transparent = False
    scene.display.shading.color_type = "TEXTURE"


def render_motion_sheet(
    scene_context: SceneContext,
    armature,
    path: str | Path,
    frames: Sequence[int],
    *,
    marked: Sequence[float] = (),
    size: int = 384,
    columns: int = 4,
    render_still: Callable | None = None,
) -> Path:
    """A grid of the rig at ``frames``; ``marked`` frames get a yellow border."""
    if not frames:
        raise InvalidRequest("No frames to render", parameter="frames")
    cv2, _mp, np = import_runtime_dependencies()
    scene = scene_context.scene
    bones = _rig_bones(armature)
    saved = _Settings(scene)
    follower = None
    tiles = []
    try:
        follower = _FollowCamera(scene, armature, bones, frames)
        camera = follower.camera
        _prepare(scene, camera, size, size)
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
        marks = {int(round(frame)) for frame in marked}
        with tempfile.TemporaryDirectory(prefix="mtp-motion-") as temporary:
            for frame in frames:
                scene.frame_set(int(frame))
                follower.follow(frame)
                image = _render_rig(
                    scene,
                    size,
                    Path(temporary) / f"rig_{int(frame):06d}.png",
                    render_still,
                    cv2,
                    np,
                )
                image = _fit(image, size, cv2, np)
                _draw_rig(image, scene, camera, armature, bones, cv2)
                label = f"frame {int(frame)}" + ("  guide" if frame in marks else "")
                cv2.putText(
                    image,
                    label,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if frame in marks:
                    cv2.rectangle(image, (1, 1), (size - 2, size - 2), (0, 220, 255), 3)
                tiles.append(image)
    finally:
        saved.restore()
        if follower is not None:
            follower.remove()
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 24))
    rows = [
        np.concatenate(tiles[index : index + columns], axis=1)
        for index in range(0, len(tiles), columns)
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.concatenate(rows, axis=0))
    if not ok:
        raise InvalidRequest(f"Could not encode {destination}")
    encoded.tofile(str(destination))
    return destination


def render_motion_video(
    scene_context: SceneContext,
    armature,
    path: str | Path,
    first: int,
    last: int,
    *,
    render_animation: Callable,
    size: int = 512,
) -> Path:
    """An H.264 MP4 of the rig from ``first`` to ``last`` at the scene rate.

    ``render_animation(scene)`` renders the scene's frame range to its
    output path.
    """
    scene = scene_context.scene
    bones = _rig_bones(armature)
    destination = Path(path).with_suffix(".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = _Settings(scene)
    follower = None
    try:
        frames = range(int(first), int(last) + 1)
        follower = _FollowCamera(scene, armature, bones, frames)
        for frame in frames:
            follower.follow(frame, key=True)
        _prepare(scene, follower.camera, size, size)
        settings = scene.render.image_settings
        settings.media_type = "VIDEO"
        settings.file_format = "FFMPEG"
        ffmpeg = scene.render.ffmpeg
        ffmpeg.format = "MPEG4"
        ffmpeg.codec = "H264"
        ffmpeg.constant_rate_factor = "HIGH"
        scene.frame_start, scene.frame_end = int(first), int(last)
        scene.render.filepath = str(destination)
        scene.render.use_file_extension = False
        render_animation(scene)
    finally:
        saved.restore()
        if follower is not None:
            follower.remove()
    if not destination.is_file():
        raise InvalidRequest(f"Blender did not write {destination}")
    return destination


__all__ = ["render_motion_sheet", "render_motion_video"]
