"""Evaluate a captured animation and render comparisons for review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bpy

from ..core.cache import store as cache_store
from ..core.errors import InvalidRequest
from ..core.fs import atomic_json
from ..core.types import CanonicalFrame
from ..evaluate import metrics as metrics_module
from ..evaluate import render as render_module
from ..session.results import CaptureResult
from .common import output_path, resolve_armature, scene_context


def load_result(result) -> dict[str, Any]:
    """A capture result as a dict, from the object, a dict or a JSON path."""
    if isinstance(result, CaptureResult):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    if isinstance(result, (str, Path)):
        path = output_path(result)
        if path is None:
            raise InvalidRequest("result path is empty", parameter="result")
        try:
            return json.loads(path.read_text(encoding="utf8"))
        except (OSError, ValueError) as exc:
            raise InvalidRequest(f"Could not read result {path}: {exc}") from exc
    raise InvalidRequest("result must be a CaptureResult, dict or path")


def _detections(result: dict[str, Any]) -> dict[int, tuple[float, CanonicalFrame]]:
    """Retained detections of ``result`` by source frame, with timeline frames.

    Empty when the retained detections belong to another capture.
    """
    key = result.get("cache_key") or ""
    frames = result.get("confidence", {}).get("per_frame", [])
    if not key or not frames:
        return {}
    by_source = {
        int(item["source_frame"]): float(item["timeline_frame"]) for item in frames
    }
    detections: dict[int, tuple[float, CanonicalFrame]] = {}
    sequence = cache_store.iter_frames(key)
    if sequence is not None:
        for index, frame in enumerate(sequence):
            if index in by_source:
                detections[index] = (by_source[index], frame)
        return detections
    for index, timeline in by_source.items():
        frame = cache_store.load_frame(key, index)
        if frame is not None:
            detections[index] = (timeline, frame)
    return detections


def evaluate(
    armature,
    *,
    action: str | None = None,
    frame_range: tuple[int, int] | None = None,
    result=None,
    ground_height: float = 0.0,
    face_output: str | None = None,
    scene=None,
    out=None,
) -> metrics_module.Metrics:
    """Measure the armature's animation and return ``Metrics``.

    - ``action``: evaluate this action instead of the current one (it is
      assigned only while measuring).
    - ``frame_range``: inclusive timeline frames; by default the range the
      ``result`` keyed, else the action's range.
    - ``result``: a capture result (object, dict or JSON path); its retained
      detections give ``landmark_fit_error_deg``.
    - ``ground_height``: world Z of the floor for ground penetration.
    - ``face_output``: measure the face metrics for this face output; by
      default the one the ``result`` used, if it captured the face.
    - ``out``: also write the metrics as JSON there.
    """
    target = resolve_armature(armature)
    context = scene_context(scene)
    data = load_result(result) if result is not None else None
    if frame_range is None and data and data.get("timeline_frame_range"):
        first, last = data["timeline_frame_range"]
        frame_range = (int(round(first)), int(round(last)))
    detections = list(_detections(data).values()) if data else None
    if face_output is None and data:
        face_output = data.get("face_output") or None
    animation_data = target.animation_data_create()
    original = animation_data.action
    if action is not None:
        replacement = bpy.data.actions.get(action)
        if replacement is None:
            raise InvalidRequest(f"No action named {action!r}", parameter="action")
        animation_data.action = replacement
    try:
        measured = metrics_module.evaluate(
            context,
            target,
            frame_range=frame_range,
            ground_height=float(ground_height),
            detections=detections or None,
            face_output=face_output,
        )
    finally:
        if action is not None:
            animation_data.action = original
    destination = output_path(out)
    if destination is not None:
        atomic_json(destination, measured.to_dict(), indent=2, sort_keys=False)
    return measured


def _render_still(scene) -> None:
    with bpy.context.temp_override(scene=scene):
        bpy.ops.render.render(write_still=True, scene=scene.name)


def render_comparison(
    path,
    armature,
    *,
    result,
    frames: list[int] | None = None,
    count: int = 6,
    size: int = 384,
    render_meshes: bool = True,
    face: bool = False,
    scene=None,
) -> dict[str, Any]:
    """Write a PNG contact sheet comparing source, detections and the rig.

    ``frames`` are timeline frames from the capture; by default ``count``
    frames are spread evenly over it. ``face=True`` renders face close-ups
    with a chart of the strongest expressions, detected and on the rig; it
    needs a result that captured the face. Returns the path and the frames
    used.
    """
    target = resolve_armature(armature)
    context = scene_context(scene)
    data = load_result(result)
    per_frame = data.get("confidence", {}).get("per_frame", [])
    if not per_frame:
        raise InvalidRequest("The result has no applied frames", parameter="result")
    by_timeline = {
        int(round(item["timeline_frame"])): int(item["source_frame"])
        for item in per_frame
    }
    timeline = sorted(by_timeline)
    if frames is None:
        step = max(len(timeline) - 1, 1) / max(int(count) - 1, 1)
        frames = sorted(
            {
                timeline[min(int(round(i * step)), len(timeline) - 1)]
                for i in range(max(int(count), 1))
            }
        )
    missing = [frame for frame in frames if frame not in by_timeline]
    if missing:
        raise InvalidRequest(
            f"Frames {missing} were not keyed by this capture", parameter="frames"
        )
    if face and not data.get("face_output"):
        raise InvalidRequest(
            "face=True needs a result that captured the face", parameter="face"
        )
    detections = _detections(data)
    rows = [
        (
            frame,
            by_timeline[frame],
            detections.get(by_timeline[frame], (0.0, None))[1],
        )
        for frame in frames
    ]
    media = data.get("media", {})
    destination = output_path(path)
    if destination is None:
        raise InvalidRequest("Give an output path", parameter="path")
    written = render_module.render_comparison(
        context,
        target,
        destination,
        rows,
        media_path=str(media.get("path", "")),
        media_kind=str(media.get("kind", "VIDEO")),
        mirror=bool(media.get("mirror", False)),
        size=int(size),
        render_still=_render_still if render_meshes else None,
        face=bool(face),
    )
    return {"path": str(written), "frames": list(frames)}


__all__ = ["evaluate", "load_result", "render_comparison"]
