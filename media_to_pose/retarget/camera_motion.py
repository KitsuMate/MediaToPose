"""Camera-motion track preparation and lateral root correction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from mathutils import Matrix, Vector

from .. import (
    camera_motion_correction,
    detection_cache,
)
from ..core.requests import RootMotionMode
from ..core.scene import SceneContext
from .samples import (
    SequenceApplyProgress,
    _append_bake_warning,
    _apply_baked_sample_pose,
    _BakedFrameSample,
    _restore_sequence_state,
    _scene_set_float,
    _sequence_state,
    _store_bone_basis,
    _translate_grounding,
)


def _publish_camera_motion_diagnostics(item, result, track_status: str) -> None:
    if item is None:
        return
    try:
        item.camera_track_status = str(track_status)
        item.camera_tracked_frames = int(result.tracked_frames)
        item.camera_rejected_frames = int(result.rejected_frames)
        item.camera_median_inliers = float(result.median_inliers)
        item.camera_corrected_intervals = int(result.corrected_intervals)
        item.last_warning = str(result.warning)
        item.status = "WARNING" if result.warning else "VALID"
    except (AttributeError, ReferenceError, RuntimeError):
        pass


def _iter_camera_motion_correction(
    armature,
    samples: list[_BakedFrameSample],
    settings,
    effective_mode: str,
    root_kind: str,
    root_target,
    target_lateral: Vector,
    object_delta_basis: Matrix,
    source_to_target_scale: float,
    track: camera_motion_correction.CameraMotionTrack | None,
    ui_settings,
    track_status: str,
    is_video: bool,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    inputs = [sample.camera_motion_input for sample in samples]
    usable_inputs = [item for item in inputs if item is not None]
    if (
        not is_video
        or effective_mode != RootMotionMode.AUTOMATIC
        or track is None
        or len(usable_inputs) != len(samples)
    ):
        warning = ""
        if is_video and effective_mode == RootMotionMode.AUTOMATIC and track is None:
            warning = (
                "Camera Motion Correction skipped: no matching camera track and "
                "the source video was unavailable"
            )
            _append_bake_warning(samples, warning)
        result = camera_motion_correction.CameraCorrectionResult(
            offsets_x=tuple(0.0 for _sample in samples),
            corrected_intervals=0,
            corrected_frames=0,
            tracked_frames=0,
            rejected_frames=len(samples)
            if effective_mode == RootMotionMode.AUTOMATIC
            else 0,
            median_inliers=0.0,
            warning=warning,
        )
        _publish_camera_motion_diagnostics(ui_settings, result, track_status)
        return

    result = camera_motion_correction.correction_offsets(
        usable_inputs,
        track,
        strength=float(settings.lateral_root_strength),
    )
    _publish_camera_motion_diagnostics(ui_settings, result, track_status)
    if result.warning:
        _append_bake_warning(samples, result.warning)
    if not result.corrected_frames:
        return

    temporary = _sequence_state(armature, scene_context=scene_context)
    try:
        for index, (sample, offset) in enumerate(
            zip(samples, result.offsets_x, strict=True), 1
        ):
            if abs(float(offset)) <= 1.0e-12:
                yield SequenceApplyProgress(
                    "Camera Motion: applying lateral root correction",
                    index,
                    len(samples),
                )
                continue
            _scene_set_float(scene_context.scene, sample.destination)
            _apply_baked_sample_pose(armature, sample, scene_context=scene_context)
            target_delta = target_lateral * (
                float(offset) * float(source_to_target_scale)
            )
            applied_delta = target_delta
            if root_kind == "BONE":
                original = sample.bones.get(root_target.name)
                if original is None:
                    continue
                original_rotation = original.rotation.copy()
                original_scale = original.scale.copy()
                matrix = root_target.matrix.copy()
                matrix.translation += target_delta
                root_target.matrix = matrix
                scene_context.update()
                _store_bone_basis(sample, root_target)
                stored = sample.bones[root_target.name]
                stored.rotation = original_rotation
                stored.scale = original_scale
            else:
                if sample.root_object is None:
                    continue
                applied_delta = object_delta_basis @ target_delta
                original_rotation = sample.root_object[1].copy()
                original_scale = sample.root_object[2].copy()
                matrix = root_target.matrix_world.copy()
                matrix.translation += applied_delta
                root_target.matrix_world = matrix
                scene_context.update()
                location, _rotation, _scale = root_target.matrix_basis.decompose()
                sample.root_object = (
                    location.copy(),
                    original_rotation,
                    original_scale,
                )
            _translate_grounding(sample.grounding, applied_delta)
            yield SequenceApplyProgress(
                "Camera Motion: applying lateral root correction",
                index,
                len(samples),
            )
    finally:
        _restore_sequence_state(armature, temporary, scene_context=scene_context)
    _append_bake_warning(
        samples,
        "Camera Motion Correction adjusted lateral root travel in "
        f"{result.corrected_intervals} interval(s) without changing rotations",
    )


def _iter_prepare_camera_motion_track(
    request,
    inputs: Sequence[camera_motion_correction.CameraMotionInput],
) -> Iterator[SequenceApplyProgress]:
    """Load or compute the derived RGB camera track before animation writes."""
    source_identity = dict(getattr(request, "source_identity", {}) or {})
    expected_key = camera_motion_correction.cache_key(source_identity, inputs)
    directory = detection_cache.camera_motion_directory(create_root=False)
    from ..dependency_bootstrap import import_camera_dependencies

    try:
        cv2, np = import_camera_dependencies()
    except ImportError as exc:
        return None, "Unavailable", str(exc)
    if bool(getattr(request, "keep_detection_cache", False)):
        cached = camera_motion_correction.load_sidecar(directory, expected_key, np)
        if cached is not None:
            yield SequenceApplyProgress("Camera Motion: loaded cached track", 1, 1)
            return cached, "Ready", ""

    source_path = detection_cache.existing_media_path(
        str(getattr(request, "analysis_path", "") or "")
    )
    if not source_path:
        return None, "Not Prepared", ""
    estimator = camera_motion_correction.iter_estimate_track(
        source_path,
        inputs,
        cv2,
        np,
    )
    try:
        while True:
            try:
                phase, current, total = next(estimator)
            except StopIteration as stopped:
                track = stopped.value
                break
            yield SequenceApplyProgress(str(phase), int(current), int(total))
    finally:
        estimator.close()

    status = "Computed"
    warning = ""
    if bool(getattr(request, "keep_detection_cache", False)):
        try:
            directory = detection_cache.camera_motion_directory(create_root=True)
            camera_motion_correction.store_sidecar(
                directory,
                expected_key,
                track,
                np,
            )
            limit = max(0, int(getattr(request, "cache_limit_bytes", 0)))
            if limit and detection_cache.cache_usage_bytes() > limit:
                detection_cache.clear_camera_motion_cache()
                status = "Computed (not cached)"
                warning = (
                    "Camera track was used but not retained because the cache limit "
                    "would be exceeded"
                )
            else:
                status = "Ready"
        except (OSError, RuntimeError, ValueError) as exc:
            status = "Computed (not cached)"
            warning = f"Camera track was used but could not be cached: {exc}"
    return track, status, warning
