"""Generate motion with Kimodo: from text, between keyed poses, along paths."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..backends.llm2vec import backend as llm2vec_backend
from ..core.errors import Cancelled, CaptureFailed, InvalidRequest, MtpError
from ..core.fs import atomic_json
from ..core.requests import OutputMode
from ..session.generate_results import GeneratePlan, GenerateResult
from ..session.generate_session import (
    GenerateRequest,
    GenerateSession,
    GenerateState,
    PromptSegment,
)
from .common import output_path, resolve_armature, resolve_scene, scene_context
from .rig import prepare_rig


def _segments(segments) -> tuple[PromptSegment, ...]:
    """``[(first_frame, prompt), ...]`` or ``[{"first": f, "prompt": p}]``."""
    result = []
    for item in segments or ():
        if isinstance(item, dict):
            frame = item.get("first", item.get("frame"))
            prompt = item.get("prompt", "")
        else:
            frame, prompt = item
        if frame is None:
            raise InvalidRequest("Every segment needs a frame", parameter="segments")
        result.append(PromptSegment(float(frame), str(prompt or "")))
    return tuple(result)


def _waypoints(waypoints) -> tuple[tuple[float, ...], ...]:
    result = []
    for item in waypoints or ():
        values = tuple(float(value) for value in item)
        if len(values) != 3:
            raise InvalidRequest(
                "A waypoint is (frame, x, y) in world units", parameter="waypoints"
            )
        result.append(values)
    return tuple(result)


def _output(output) -> OutputMode:
    try:
        return OutputMode(str(output).upper())
    except ValueError as exc:
        raise InvalidRequest(
            "output must be NEW_ACTION, REPLACE or NLA_STRIP", parameter="output"
        ) from exc


# API argument -> (request field, default).
_FIELDS = {
    "prompt": ("prompt", ""),
    "segments": ("segments", ()),
    "guides": ("use_keys", True),
    "ignore_frames": ("ignored_frames", ()),
    "waypoints": ("waypoints", ()),
    "path": ("path", ""),
    "face_travel": ("face_travel", True),
    "seed": ("seed", 0),
    "steps": ("steps", 0),
    "text_guidance": ("text_weight", 2.0),
    "guide_guidance": ("guide_weight", 2.0),
    "blend_frames": ("blend_frames", 8.0),
    "output": ("output_mode", "NEW_ACTION"),
    "device": ("device", "AUTO"),
}


def _request(target, scene, options: dict[str, Any], recipe) -> GenerateRequest:
    """Arguments given win; then the recipe's values; then the defaults."""
    recipe = dict(recipe or {})
    values: dict[str, Any] = {}
    for argument, (name, default) in _FIELDS.items():
        given = options.get(argument)
        values[name] = given if given is not None else recipe.get(name, default)
    frame_range = options.get("frame_range")
    if frame_range is None and "first" in recipe:
        frame_range = (recipe["first"], recipe["last"])
    if frame_range is None:
        frame_range = (scene.frame_start, scene.frame_end)
    first, last = (float(value) for value in frame_range)
    return GenerateRequest(
        armature=target.name,
        first=first,
        last=last,
        prompt=str(values["prompt"] or ""),
        segments=_segments(values["segments"]),
        use_keys=bool(values["use_keys"]),
        ignored_frames=tuple(float(f) for f in values["ignored_frames"] or ()),
        waypoints=_waypoints(values["waypoints"]),
        path=str(values["path"] or ""),
        face_travel=bool(values["face_travel"]),
        seed=int(values["seed"]),
        steps=int(values["steps"]),
        text_weight=float(values["text_weight"]),
        guide_weight=float(values["guide_weight"]),
        blend_frames=float(values["blend_frames"]),
        output_mode=_output(values["output_mode"]),
        device=str(values["device"]).upper(),
    )


def generate(
    armature,
    prompt: str | None = None,
    *,
    segments: Sequence | None = None,
    frame_range: tuple[float, float] | None = None,
    guides: bool | None = None,
    ignore_frames: Iterable[float] | None = None,
    waypoints: Sequence[Sequence[float]] | None = None,
    path: str | None = None,
    face_travel: bool | None = None,
    seed: int | None = None,
    steps: int | None = None,
    text_guidance: float | None = None,
    guide_guidance: float | None = None,
    blend_frames: float | None = None,
    output: str | None = None,
    device: str | None = None,
    dry_run: bool = False,
    recipe: dict | None = None,
    timeout: float | None = None,
    result_path=None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    scene=None,
) -> GenerateResult | GeneratePlan:
    """Generate motion on ``armature`` over ``frame_range`` (scene frames,
    inclusive; default the scene range).

    - ``prompt``: what the person does ("A person walks forward."); empty
      generates from the guides alone and never loads the text encoder.
    - ``segments``: several prompts, ``[(first_frame, prompt), ...]``.
    - ``guides``: use the armature's keys in the range as guides: keyed full
      poses are reached exactly, pelvis-only keys are root waypoints.
    - ``waypoints``: ``[(frame, x, y), ...]`` world positions the character
      walks through; ``face_travel`` turns it towards the next one.
    - ``path``: a curve object to follow at constant speed.
    - ``seed``, ``steps`` (default automatic: 50 on a GPU, 25 on the CPU),
      ``text_guidance`` and
      ``guide_guidance`` (default 2.0 each).
    - ``output``: ``NEW_ACTION``, ``REPLACE`` or ``NLA_STRIP``.
    - ``dry_run``: return the ``GeneratePlan`` without loading a model.
    - ``recipe``: a previous result's ``recipe``; arguments given here win.
    """
    target = resolve_armature(armature)
    scene_object = resolve_scene(scene)
    settings = getattr(scene_object, "mtp_settings", None)
    prepare_rig(target, settings)
    options = {
        "prompt": prompt,
        "segments": segments,
        "frame_range": frame_range,
        "guides": guides,
        "ignore_frames": ignore_frames,
        "waypoints": waypoints,
        "path": path,
        "face_travel": face_travel,
        "seed": seed,
        "steps": steps,
        "text_guidance": text_guidance,
        "guide_guidance": guide_guidance,
        "blend_frames": blend_frames,
        "output": output,
        "device": device,
    }
    request = _request(target, scene_object, options, recipe)
    session = GenerateSession(request, target, scene_context(scene_object))
    plan = session.prepare()
    if dry_run:
        return plan
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    update = session.start()
    last = None
    while not update.state.finished:
        if deadline is not None and time.monotonic() > deadline:
            session.cancel("Generation timed out")
            deadline = None
        if session.state is GenerateState.GENERATING:
            time.sleep(0.05)
        update = session.step(budget_seconds=0.25)
        report = (update.status, round(update.progress, 3))
        if progress is not None and report != last:
            progress({"progress": report[1], "status": report[0]})
        last = report
    session.wait()
    if update.state is GenerateState.CANCELLED:
        raise Cancelled(update.status or "Generation cancelled")
    if update.state is GenerateState.FAILED:
        error = session.error
        if isinstance(error, MtpError):
            raise error
        raise CaptureFailed(update.status or "Generation failed")
    result = session.result
    assert result is not None
    destination = output_path(result_path)
    if destination is not None:
        atomic_json(destination, result.to_dict(), indent=2, sort_keys=False)
    return result


def describe_guides(
    armature,
    *,
    frame_range: tuple[float, float] | None = None,
    waypoints: Sequence[Sequence[float]] | None = None,
    path: str | None = None,
    scene=None,
) -> dict[str, Any]:
    """The guides a generation over ``frame_range`` would use, classified."""
    plan = generate(
        armature,
        frame_range=frame_range,
        waypoints=waypoints,
        path=path,
        dry_run=True,
        scene=scene,
    )
    assert isinstance(plan, GeneratePlan)
    return {
        "armature": plan.armature,
        "frame_range": list(plan.frame_range),
        "model_frames": plan.model_frames,
        "guides": plan.guides,
        "warnings": plan.warnings,
    }


def encode_prompt(prompt: str) -> dict[str, Any]:
    """Encode a prompt into the embedding cache (loads the text encoder once)."""
    encoding = llm2vec_backend.encode_prompt(str(prompt or ""))
    if encoding is None:
        return {"prompt": "", "cached": False, "truncated": False}
    return {
        "prompt": encoding.text,
        "cached": True,
        "truncated": bool(encoding.truncated),
    }


__all__ = ["describe_guides", "encode_prompt", "generate"]
