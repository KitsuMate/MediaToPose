"""Shared argument handling for the API: scenes, armatures and media paths."""

from __future__ import annotations

import time
from pathlib import Path

import bpy

from ..core import paths
from ..core.errors import Cancelled, CaptureFailed, InvalidRequest, MediaError
from ..core.scene import SceneContext


def resolve_scene(scene=None):
    """The named or given scene, or the active scene."""
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
        if scene is None:
            raise InvalidRequest("There is no active scene", parameter="scene")
        return scene
    if isinstance(scene, str):
        found = bpy.data.scenes.get(scene)
        if found is None:
            raise InvalidRequest(f"No scene named {scene!r}", parameter="scene")
        return found
    return scene


def scene_context(scene=None) -> SceneContext:
    """A ``SceneContext`` for ``scene`` and its first view layer."""
    scene = resolve_scene(scene)
    context_scene = getattr(bpy.context, "scene", None)
    view_layer = (
        bpy.context.view_layer
        if context_scene is not None and context_scene == scene
        else scene.view_layers[0]
    )
    return SceneContext(scene, view_layer, bpy.context.preferences)


def resolve_armature(armature):
    """The armature object with this name, or the given object."""
    if isinstance(armature, str):
        found = bpy.data.objects.get(armature)
        if found is None:
            raise InvalidRequest(f"No object named {armature!r}", parameter="armature")
        armature = found
    if armature is None or getattr(armature, "type", "") != "ARMATURE":
        name = getattr(armature, "name", armature)
        raise InvalidRequest(f"{name!r} is not an armature", parameter="armature")
    return armature


def resolve_media(media) -> str:
    """An absolute path to existing media; ``//`` is relative to the .blend."""
    if not media:
        raise InvalidRequest("Give a media file", parameter="media")
    path = paths.expand_user_path(str(media)).resolve(strict=False)
    if not path.is_file():
        raise MediaError(f"Media file not found: {path}", parameter="media")
    return str(path)


def drive(session, *, timeout: float | None = None):
    """Run a prepared ``CaptureSession`` to completion on this thread.

    Detection runs on the session's worker thread; this loop sleeps while
    it works and applies the result in bounded steps. Raises ``Cancelled``
    after ``timeout`` seconds and ``CaptureFailed`` when the capture fails;
    in both cases the session has already rolled Blender data back.
    """
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    update = session.start()
    timed_out = False
    while not update.state.finished:
        if deadline is not None and time.monotonic() > deadline and not timed_out:
            timed_out = True
            session.cancel("Timed out")
        if session.state == "DETECTING":
            time.sleep(min(max(update.interval, 0.01), 0.05))
        update = session.step(budget_seconds=0.25)
    session.wait()
    if update.state == "CANCELLED":
        message = "Capture timed out" if timed_out else update.status or "Cancelled"
        raise Cancelled(message)
    if update.state == "FAILED":
        raise CaptureFailed(update.status or "Capture failed")
    return update


def output_path(path) -> Path | None:
    if path in {None, ""}:
        return None
    return paths.expand_user_path(str(path)).resolve(strict=False)


__all__ = [
    "drive",
    "output_path",
    "resolve_armature",
    "resolve_media",
    "resolve_scene",
    "scene_context",
]
