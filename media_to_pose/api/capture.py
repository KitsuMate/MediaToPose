"""Capture media onto an armature, or re-apply retained detections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ..core.errors import InvalidRequest
from ..core.media_io import probe_media
from ..core.requests import (
    CaptureMode,
    CaptureSettings,
    FaceOutput,
    MediaKind,
    OutputMode,
    RootMotionMode,
)
from ..postprocess import apply_module_overrides, compile_postprocess_config
from ..session.capture_session import CaptureSession
from ..session.results import CaptureResult, build_capture_result
from .common import (
    drive,
    output_path,
    resolve_armature,
    resolve_media,
    scene_context,
)
from .rig import prepare_rig

PARTS = ("body", "hands", "face")


def _choice(enum, value, parameter: str):
    try:
        return enum(str(value).upper())
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum)
        raise InvalidRequest(
            f"{parameter} must be one of {choices}", parameter=parameter
        ) from exc


def _parts(parts: Iterable[str] | str | None, settings: CaptureSettings) -> dict:
    if parts is None:
        return {}
    if isinstance(parts, str):
        parts = [part for part in parts.replace(" ", "").split(",") if part]
    selected = {str(part).lower() for part in parts}
    unknown = selected - set(PARTS)
    if unknown or not selected:
        raise InvalidRequest(
            f"parts must name some of {', '.join(PARTS)}", parameter="parts"
        )
    return {f"use_{part}": part in selected for part in PARTS}


def _settings(scene, media_path: str, **overrides: Any) -> CaptureSettings:
    settings = CaptureSettings.from_rna(scene.mtp_settings)
    changes: dict[str, Any] = {"media_path": media_path, "media_identity_json": ""}
    provider = overrides.get("provider")
    if provider is not None:
        changes["provider"] = str(provider).upper()
    changes.update(_parts(overrides.get("parts"), settings))
    if overrides.get("root_motion") is not None:
        changes["root_motion_mode"] = _choice(
            RootMotionMode, overrides["root_motion"], "root_motion"
        )
    if overrides.get("face_output") is not None:
        changes["face_output"] = _choice(
            FaceOutput, overrides["face_output"], "face_output"
        )
    if overrides.get("mirror") is not None:
        changes["mirror_input"] = bool(overrides["mirror"])
    if overrides.get("keep_cache") is not None:
        changes["keep_detection_cache"] = bool(overrides["keep_cache"])
    if overrides.get("video_frame") is not None:
        changes["video_frame"] = int(overrides["video_frame"]) + 1
    return replace(settings, **changes)


def _mode(mode, media_path: str) -> CaptureMode:
    if mode in {None, "", "AUTO"}:
        kind = probe_media(media_path).kind
        return CaptureMode.KEYFRAMES if kind == MediaKind.VIDEO else CaptureMode.POSE
    selected = _choice(CaptureMode, mode, "mode")
    if selected == CaptureMode.ANALYZE:
        raise InvalidRequest(
            "Use mode KEYFRAMES or POSE; analysis runs as part of a capture",
            parameter="mode",
        )
    return selected


def _postprocess(armature, config):
    if config is None:
        return None
    payload = (
        config.get("post_processing", config) if isinstance(config, dict) else config
    )
    return apply_module_overrides(
        compile_postprocess_config(armature.mtp_rig_config), payload
    )


def _run(
    media,
    armature,
    *,
    require_cache: bool,
    scene=None,
    mode=None,
    frame_range=None,
    start_frame=None,
    config=None,
    output="NEW_ACTION",
    timeout=None,
    result_path=None,
    key=False,
    **overrides: Any,
) -> CaptureResult:
    context = scene_context(scene)
    target = resolve_armature(armature)
    prepare_rig(target, context.scene.mtp_settings)
    media_path = resolve_media(media)
    settings = _settings(context.scene, media_path, **overrides)
    if not settings.keep_detection_cache and require_cache:
        raise InvalidRequest(
            "retarget_cached needs the detection cache", parameter="keep_cache"
        )
    selected_mode = _mode(mode, media_path)
    if frame_range is not None and selected_mode == CaptureMode.POSE:
        raise InvalidRequest(
            "frame_range applies to KEYFRAMES captures; use frame for a pose",
            parameter="frame_range",
        )
    session = CaptureSession.prepare(
        context,
        target,
        settings,
        selected_mode,
        output_mode=_choice(OutputMode, output, "output"),
        frame_range=tuple(frame_range) if frame_range is not None else None,
        start_frame=start_frame,
        postprocess_config=_postprocess(target, config),
        require_cache=require_cache,
        key_pose=bool(key) and selected_mode == CaptureMode.POSE,
    )
    drive(session, timeout=timeout)
    result = build_capture_result(session, armature=target)
    destination = output_path(result_path)
    if destination is not None:
        result.write_json(destination)
    return result


def capture(
    media,
    armature,
    *,
    provider: str | None = None,
    parts: Iterable[str] | str | None = None,
    mode: str | None = None,
    frame: int | None = None,
    frame_range: tuple[int, int] | None = None,
    start_frame: float | None = None,
    root_motion: str | None = None,
    face_output: str | None = None,
    mirror: bool | None = None,
    config: dict | list | None = None,
    output: str = "NEW_ACTION",
    keep_cache: bool | None = None,
    key: bool = False,
    scene=None,
    timeout: float | None = None,
    result_path=None,
) -> CaptureResult:
    """Capture ``media`` onto ``armature`` and return a ``CaptureResult``.

    Unset options fall back to the scene's Media to Pose settings, so a
    capture behaves like the panel unless told otherwise.

    - ``mode``: ``KEYFRAMES`` (animation) or ``POSE`` (one pose at the
      current frame); by default videos are keyed and images posed.
    - ``parts``: any of ``body``, ``hands``, ``face``.
    - ``frame``: the 0-based video frame for ``POSE``.
    - ``frame_range``: source frames ``(first, end)``, 0-based and end
      exclusive, for ``KEYFRAMES``; only those frames are keyed, at the same
      timeline frames as a full capture.
    - ``start_frame``: timeline frame of source frame 0 (default: current).
    - ``key``: for ``POSE``, also key the pose at ``start_frame`` (or the
      current frame), for example to block a pose that ``generate`` then
      reaches as a guide.
    - ``root_motion``: ``AUTOMATIC``, ``IN_PLACE`` or ``OFF``.
    - ``config``: clean-up settings for this capture only, as returned by
      ``get_config`` or ``{module_id: {setting: value}}``; the rig keeps
      its own settings.
    - ``output``: ``NEW_ACTION``, ``REPLACE`` or ``NLA_STRIP``.
    - ``result_path``: also write the result as JSON there.

    Raises an ``MtpError`` subclass: ``InvalidRequest``, ``MediaError``,
    ``ModelMissing``, ``MappingIncomplete``, ``OperationBusy``,
    ``Cancelled`` (timeout) or ``CaptureFailed``.
    """
    return _run(
        media,
        armature,
        require_cache=False,
        scene=scene,
        mode=mode,
        frame_range=frame_range,
        start_frame=start_frame,
        config=config,
        output=output,
        timeout=timeout,
        result_path=result_path,
        provider=provider,
        parts=parts,
        root_motion=root_motion,
        face_output=face_output,
        mirror=mirror,
        keep_cache=keep_cache,
        video_frame=frame,
        key=key,
    )


def retarget_cached(
    media,
    armature,
    *,
    provider: str | None = None,
    parts: Iterable[str] | str | None = None,
    mode: str | None = None,
    frame: int | None = None,
    frame_range: tuple[int, int] | None = None,
    start_frame: float | None = None,
    root_motion: str | None = None,
    face_output: str | None = None,
    mirror: bool | None = None,
    config: dict | list | None = None,
    output: str = "NEW_ACTION",
    scene=None,
    timeout: float | None = None,
    result_path=None,
) -> CaptureResult:
    """Re-run retargeting and clean-up from retained detections.

    Takes the same arguments as ``capture`` but never runs detection: the
    media, provider, parts and mirroring must match the last capture that
    kept its detections, otherwise ``InvalidRequest`` is raised.
    """
    return _run(
        media,
        armature,
        require_cache=True,
        scene=scene,
        mode=mode,
        frame_range=frame_range,
        start_frame=start_frame,
        config=config,
        output=output,
        timeout=timeout,
        result_path=result_path,
        provider=provider,
        parts=parts,
        root_motion=root_motion,
        face_output=face_output,
        mirror=mirror,
        keep_cache=True,
        video_frame=frame,
    )


__all__ = ["PARTS", "capture", "retarget_cached"]
