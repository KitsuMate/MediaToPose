"""Solve and key a detected sequence as a series of named stages."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress

from mathutils import Matrix, Vector

from .. import (
    camera_motion_correction,
    mapping,
    rig_planning,
)
from ..core.cache.stream import FrameSpool
from ..core.errors import MediaError
from ..core.requests import MediaKind, RootMotionMode
from ..core.scene import SceneContext
from ..models import CanonicalFrame
from ..post_processing import (
    PostProcessingSession,
    SequenceProcessingContext,
    compile_postprocess_config,
    iter_sequence_pipeline,
)
from .camera_motion import (
    _iter_camera_motion_correction,
    _iter_prepare_camera_motion_track,
)
from .capture import _iter_capture_baked_frame
from .detection_filter import _iter_detection_filter
from .keyframes import (
    _ensure_action,
    _finish_keyframe_stream,
    _iter_commit_baked_samples,
    _iter_write_baked_sample,
    _KeyframeWriteState,
)
from .rohm import _iter_rohm_correction
from .root_motion import (
    _reset_root_location,
    _root_motion_deltas,
    _root_motion_sample,
    _sequence_root_target,
    _set_root_pose,
    _target_body_height,
)
from .samples import (
    SequenceApplyProgress,
    _BakedFrameSample,
    _drain,
    _restore_sequence_state,
    _RootMotionSample,
    _scene_set_float,
    _sequence_state,
    _sequence_timestamps,
)
from .solve.frames import target_basis, target_rest_basis
from .solve.ik import prepare_limb_ownership
from .temporal import _iter_temporal_correction


def iter_apply_sequence(
    armature,
    frames: Iterable[CanonicalFrame],
    settings,
    start_frame: float,
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    """Cooperatively solve and key a sequence without long UI stalls.

    Blender data is deliberately touched only by the caller's main thread.
    Yielding between cache reads, solved frames, cleanup channels, and written
    frames lets a modal operator return control to Blender regularly.
    """
    total_hint = len(frames) if isinstance(frames, Sequence) else 0
    scene = scene_context.scene
    scene_fps = scene_context.fps
    state = _sequence_state(armature, scene_context=scene_context)
    samples: list[_BakedFrameSample] = []
    warning_sink: list[str] = []
    config = armature.mtp_rig_config
    postprocess_config = getattr(
        settings, "postprocess_config", None
    ) or compile_postprocess_config(config)
    temporal_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "TEMPORAL_CORRECTION"
        ),
        None,
    )
    rohm_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "ROHM_CORRECTION"
        ),
        None,
    )
    capture_rohm_input = bool(
        rohm_settings
        and rohm_settings.enabled
        and (
            bool(getattr(rohm_settings, "learned_pose_residual", False))
            or (
                bool(getattr(rohm_settings, "root_trajectory_cleanup", True))
                and RootMotionMode(getattr(settings, "root_motion_mode", "AUTOMATIC"))
                == RootMotionMode.AUTOMATIC
            )
        )
    )
    detection_filter_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "DETECTION_FILTER"
        ),
        None,
    )
    camera_motion_settings = next(
        (
            item
            for item in postprocess_config.modules
            if item.module_id == "CAMERA_MOTION_CORRECTION"
        ),
        None,
    )
    camera_motion_ui_settings = next(
        (
            item
            for item in config.post_processing_modules
            if item.module_id == "CAMERA_MOTION_CORRECTION"
        ),
        None,
    )
    detection_filter_ui_settings = next(
        (
            item
            for item in config.post_processing_modules
            if item.module_id == "DETECTION_FILTER"
        ),
        None,
    )
    entries = getattr(settings, "mapping_entries", None) or mapping.mapping_by_id(
        config
    )
    rig_plan = rig_planning.compile_rig_solve_plan(
        armature,
        entries,
        lambda: target_rest_basis(armature, entries),
    )
    requested_mode = RootMotionMode(getattr(settings, "root_motion_mode", "AUTOMATIC"))
    root_kind, root_target = _sequence_root_target(armature, entries)
    root_enabled = requested_mode != RootMotionMode.OFF
    offline_processing = bool(
        (
            camera_motion_settings
            and camera_motion_settings.enabled
            and requested_mode == RootMotionMode.AUTOMATIC
            and str(getattr(settings, "media_kind", "")) == MediaKind.VIDEO
        )
        or (detection_filter_settings and detection_filter_settings.enabled)
        or (rohm_settings and rohm_settings.enabled)
        or (
            temporal_settings
            and temporal_settings.enabled
            and (
                temporal_settings.smooth_motion
                or temporal_settings.ground_contacts
                or temporal_settings.stabilize_hands
            )
        )
    )
    frame_spool = None
    root_samples: list[_RootMotionSample] = []
    planned_deltas: list[Vector] = []
    planned_yaws: list[float] = []
    effective_mode = requested_mode
    frame_count = 0
    previous_timestamp_ms = None
    camera_inputs: list[camera_motion_correction.CameraMotionInput] = []
    camera_track: camera_motion_correction.CameraMotionTrack | None = None
    camera_track_status = "Not Prepared"
    camera_preflight_warning = ""
    target_lateral = Vector((1.0, 0.0, 0.0))
    object_delta_basis = Matrix.Identity(3)
    source_to_target_scale = 1.0

    if requested_mode == RootMotionMode.AUTOMATIC:
        # The conditional spool is closed by the encompassing generator
        # ``finally`` block, including GeneratorExit during cancellation.
        frame_spool = FrameSpool()
        try:
            for detected in frames:
                timestamp_ms = int(detected.timestamp_ms)
                if (
                    previous_timestamp_ms is not None
                    and timestamp_ms <= previous_timestamp_ms
                ):
                    raise MediaError("Detected video timestamps must increase")
                previous_timestamp_ms = timestamp_ms
                frame_spool.append(detected)
                root_samples.append(_root_motion_sample(detected))
                camera_inputs.append(
                    camera_motion_correction.input_from_frame(detected)
                )
                frame_count += 1
                yield SequenceApplyProgress(
                    "Loading detections",
                    frame_count,
                    total_hint,
                )
            if not frame_count:
                raise MediaError("No detectable frames were produced")
        except BaseException:
            frame_spool.close()
            raise

        frame_iterator = iter(frame_spool)

        if (
            camera_motion_settings
            and camera_motion_settings.enabled
            and str(getattr(settings, "media_kind", "")) == MediaKind.VIDEO
        ):
            try:
                camera_preflight = _iter_prepare_camera_motion_track(
                    settings,
                    camera_inputs,
                )
                while True:
                    try:
                        yield next(camera_preflight)
                    except StopIteration as stopped:
                        (
                            camera_track,
                            camera_track_status,
                            camera_preflight_warning,
                        ) = stopped.value
                        break
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                camera_track = None
                camera_track_status = "Error"
                camera_preflight_warning = f"Camera Motion Correction skipped: {exc}"
            if camera_motion_ui_settings is not None:
                with suppress(AttributeError, ReferenceError, RuntimeError):
                    camera_motion_ui_settings.camera_track_status = camera_track_status
        total_frames = frame_count
    else:
        frame_iterator = iter(frames)
        total_frames = total_hint
        frame_count = 0
        previous_timestamp_ms = None

    keyframe_state = None
    last_destination = float(start_frame)
    try:
        processing_session = PostProcessingSession()
        target_up = None
        baseline = None
        for index, detected in enumerate(frame_iterator, 1):
            timestamp_ms = int(detected.timestamp_ms)
            if requested_mode != RootMotionMode.AUTOMATIC:
                if (
                    previous_timestamp_ms is not None
                    and timestamp_ms <= previous_timestamp_ms
                ):
                    raise MediaError("Detected video timestamps must increase")
                previous_timestamp_ms = timestamp_ms
                frame_count = index
                yield SequenceApplyProgress(
                    "Loading detections",
                    index,
                    total_frames,
                )
            destination = float(start_frame) + timestamp_ms * scene_fps / 1000.0
            last_destination = destination
            _scene_set_float(scene, destination)
            if index == 1:
                prepare_limb_ownership(armature)
                if root_enabled and not bool(
                    getattr(settings, "keep_original_root_position", False)
                ):
                    _reset_root_location(root_target, scene_context=scene_context)
                target_space = target_basis(armature, entries)
                target_up = target_space @ Vector((0.0, 0.0, 1.0))
                baseline = (
                    root_target.matrix.copy()
                    if root_kind == "BONE"
                    else root_target.matrix_world.copy()
                )
                if requested_mode == RootMotionMode.AUTOMATIC:
                    (
                        planned_deltas,
                        planned_yaws,
                        effective_mode,
                        root_warnings,
                    ) = _root_motion_deltas(
                        armature,
                        entries,
                        root_samples,
                        requested_mode,
                    )
                    valid_root_samples = [
                        sample
                        for sample in root_samples
                        if sample.observation is not None and sample.observation.valid
                    ]
                    if valid_root_samples:
                        reference = valid_root_samples[0]
                        lateral = target_space @ (
                            reference.source_basis.inverted_safe()
                            @ Vector((1.0, 0.0, 0.0))
                        )
                        normalized_up = target_up.normalized()
                        lateral -= normalized_up * lateral.dot(normalized_up)
                        if lateral.length > 1.0e-8:
                            target_lateral = lateral.normalized()
                        target_height = _target_body_height(armature, entries)
                        source_to_target_scale = (
                            target_height / reference.source_height
                            if target_height > 1.0e-6
                            and reference.source_height > 1.0e-6
                            else 1.0
                        )
                        source_to_target_scale = min(
                            max(source_to_target_scale, 0.1), 10.0
                        )
                    object_delta_basis = baseline.to_3x3()
                    for warning in root_warnings:
                        if warning not in warning_sink:
                            warning_sink.append(warning)
                        if warning not in detected.warnings:
                            detected.warnings.append(warning)
                    if camera_preflight_warning:
                        if camera_preflight_warning not in warning_sink:
                            warning_sink.append(camera_preflight_warning)
                        if camera_preflight_warning not in detected.warnings:
                            detected.warnings.append(camera_preflight_warning)
                _ensure_action(armature)
                keyframe_state = _KeyframeWriteState(armature)

            delta = (
                planned_deltas[index - 1]
                if effective_mode == RootMotionMode.AUTOMATIC
                else Vector()
            )
            yaw = (
                planned_yaws[index - 1]
                if effective_mode == RootMotionMode.AUTOMATIC
                else 0.0
            )
            if root_enabled:
                _set_root_pose(
                    root_kind,
                    root_target,
                    baseline,
                    delta,
                    yaw,
                    target_up,
                    scene_context=scene_context,
                )
            frame_solver = _iter_capture_baked_frame(
                armature,
                detected,
                destination,
                settings,
                processing_session,
                root_kind,
                root_target,
                root_enabled,
                postprocess_config,
                rig_plan,
                warning_sink,
                capture_rohm_input,
                scene_context=scene_context,
            )
            while True:
                try:
                    solve_phase, solve_current, solve_total = next(frame_solver)
                except StopIteration as stopped:
                    sample = stopped.value
                    break
                yield SequenceApplyProgress(
                    solve_phase,
                    (index - 1) * solve_total + solve_current,
                    total_frames * solve_total,
                )
            if offline_processing:
                samples.append(sample)
            yield SequenceApplyProgress(
                "Solving animation",
                index,
                total_frames,
            )
            yield from _iter_write_baked_sample(
                keyframe_state,
                sample,
                index,
                total_frames,
                phase="Streaming keyframes",
                scene_context=scene_context,
            )
        if not frame_count:
            raise MediaError("No detectable frames were produced")
    finally:
        _restore_sequence_state(armature, state, scene_context=scene_context)
        if frame_spool is not None:
            frame_spool.close()

    if offline_processing:
        context = SequenceProcessingContext(
            samples=samples,
            timestamps=_sequence_timestamps(samples),
            root_motion_mode=effective_mode,
            process_camera_motion=lambda module_settings: _iter_camera_motion_correction(
                armature,
                samples,
                module_settings,
                effective_mode,
                root_kind,
                root_target,
                target_lateral,
                object_delta_basis,
                source_to_target_scale,
                camera_track,
                camera_motion_ui_settings,
                camera_track_status,
                str(getattr(settings, "media_kind", "")) == MediaKind.VIDEO,
                scene_context=scene_context,
            ),
            process_temporal=lambda module_settings: _iter_temporal_correction(
                armature,
                entries,
                samples,
                module_settings,
                effective_mode,
                root_kind,
                root_target,
                target_up,
                scene_context=scene_context,
            ),
            process_detection_filter=lambda module_settings: _iter_detection_filter(
                samples,
                module_settings,
                rohm_settings=rohm_settings,
                effective_mode=effective_mode,
                ui_settings=detection_filter_ui_settings,
            ),
            process_rohm=lambda module_settings: _iter_rohm_correction(
                armature,
                entries,
                rig_plan,
                samples,
                module_settings,
                effective_mode,
                root_kind,
                root_target,
                scene_context=scene_context,
            ),
        )
        yield from iter_sequence_pipeline(context, postprocess_config)
        keyframe_state.attach()
        total_bones, total_shapes = yield from _iter_commit_baked_samples(
            armature,
            samples,
            root_kind,
            root_target,
            root_enabled,
            scene_context=scene_context,
        )
    else:
        total_bones, total_shapes = _finish_keyframe_stream(
            keyframe_state,
            last_destination,
            scene_context=scene_context,
        )
    return frame_count, total_bones, total_shapes, tuple(warning_sink)


def apply_sequence(
    armature,
    frames: Iterable[CanonicalFrame],
    settings,
    start_frame: float,
    *,
    scene_context: SceneContext,
) -> tuple[int, int, int, tuple[str, ...]]:
    """Synchronous API retained for scripts, tests, and single-shot callers."""
    return _drain(
        iter_apply_sequence(
            armature, frames, settings, start_frame, scene_context=scene_context
        )
    )
