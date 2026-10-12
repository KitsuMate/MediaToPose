"""Solve and key a detected sequence as a series of named stages."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress

from mathutils import Matrix, Vector

from ..core.cache.stream import FrameSpool
from ..core.camera import motion_correction as camera_motion_correction
from ..core.errors import MediaError
from ..core.requests import MediaKind, RootMotionMode
from ..core.scene import SceneContext
from ..core.types import CanonicalFrame
from ..postprocess import (
    PostProcessingSession,
    SequenceProcessingContext,
    compile_postprocess_config,
    iter_sequence_pipeline,
)
from ..rig import mapping
from ..rig import planning as rig_planning
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


def _module(config, module_id: str):
    return next((item for item in config.modules if item.module_id == module_id), None)


def _rna_module(config, module_id: str):
    return next(
        (
            item
            for item in config.post_processing_modules
            if item.module_id == module_id
        ),
        None,
    )


def _add_warning(sink: list[str], frame: CanonicalFrame, warning: str) -> None:
    if warning not in sink:
        sink.append(warning)
    if warning not in frame.warnings:
        frame.warnings.append(warning)


class _SequenceRun:
    """Everything the stages of one sequence bake share.

    The stages run in order: load detections, prepare the camera track,
    solve and stream each frame, run the offline clean-up pipeline, then
    write the final keyframes. Each stage is a generator of progress.
    """

    def __init__(
        self,
        armature,
        frames: Iterable[CanonicalFrame],
        settings,
        start_frame: float,
        scene_context: SceneContext,
    ):
        self.armature = armature
        self.frames = frames
        self.settings = settings
        self.start_frame = float(start_frame)
        frame_range = getattr(settings, "frame_range", None)
        self.source_frame_offset = int(frame_range[0]) if frame_range else 0
        self.scene_context = scene_context
        self.total_hint = len(frames) if isinstance(frames, Sequence) else 0
        self.state = _sequence_state(armature, scene_context=scene_context)
        self.samples: list[_BakedFrameSample] = []
        self.warnings: list[str] = []
        config = armature.mtp_rig_config
        self.postprocess_config = getattr(
            settings, "postprocess_config", None
        ) or compile_postprocess_config(config)
        modules = self.postprocess_config
        self.temporal_settings = _module(modules, "TEMPORAL_CORRECTION")
        self.rohm_settings = _module(modules, "ROHM_CORRECTION")
        self.detection_filter_settings = _module(modules, "DETECTION_FILTER")
        self.camera_motion_settings = _module(modules, "CAMERA_MOTION_CORRECTION")
        self.camera_motion_ui_settings = _rna_module(config, "CAMERA_MOTION_CORRECTION")
        self.detection_filter_ui_settings = _rna_module(config, "DETECTION_FILTER")
        entries = getattr(settings, "mapping_entries", None) or mapping.mapping_by_id(
            config
        )
        self.entries = entries
        self.rig_plan = rig_planning.compile_rig_solve_plan(
            armature, entries, lambda: target_rest_basis(armature, entries)
        )
        self.requested_mode = RootMotionMode(
            getattr(settings, "root_motion_mode", "AUTOMATIC")
        )
        self.effective_mode = self.requested_mode
        self.is_video = str(getattr(settings, "media_kind", "")) == MediaKind.VIDEO
        self.root_kind, self.root_target = _sequence_root_target(armature, entries)
        self.root_enabled = self.requested_mode != RootMotionMode.OFF
        rohm = self.rohm_settings
        self.capture_rohm_input = bool(
            rohm
            and rohm.enabled
            and (
                bool(getattr(rohm, "learned_pose_residual", False))
                or (
                    bool(getattr(rohm, "root_trajectory_cleanup", True))
                    and self.requested_mode == RootMotionMode.AUTOMATIC
                )
            )
        )
        self.offline_processing = self._needs_offline_processing()
        self.frame_spool: FrameSpool | None = None
        self.frame_iterator: Iterator[CanonicalFrame] = iter(())
        self.frame_count = 0
        self.total_frames = 0
        self.previous_timestamp_ms: int | None = None
        self.root_samples: list[_RootMotionSample] = []
        self.planned_deltas: list[Vector] = []
        self.planned_yaws: list[float] = []
        self.camera_inputs: list[camera_motion_correction.CameraMotionInput] = []
        self.camera_track: camera_motion_correction.CameraMotionTrack | None = None
        self.camera_track_status = "Not Prepared"
        self.camera_preflight_warning = ""
        self.target_lateral = Vector((1.0, 0.0, 0.0))
        self.object_delta_basis = Matrix.Identity(3)
        self.source_to_target_scale = 1.0
        self.keyframe_state: _KeyframeWriteState | None = None
        self.last_destination = self.start_frame
        self.target_up: Vector | None = None
        self.baseline = None
        self.solving_started = False

    def _needs_offline_processing(self) -> bool:
        camera = self.camera_motion_settings
        temporal = self.temporal_settings
        detection_filter = self.detection_filter_settings
        return bool(
            (
                camera
                and camera.enabled
                and self.requested_mode == RootMotionMode.AUTOMATIC
                and self.is_video
            )
            or (detection_filter and detection_filter.enabled)
            or (self.rohm_settings and self.rohm_settings.enabled)
            or (
                temporal
                and temporal.enabled
                and (
                    temporal.smooth_motion
                    or temporal.ground_contacts
                    or temporal.stabilize_hands
                )
            )
        )

    def _check_timestamp(self, detected: CanonicalFrame) -> int:
        timestamp_ms = int(detected.timestamp_ms)
        if (
            self.previous_timestamp_ms is not None
            and timestamp_ms <= self.previous_timestamp_ms
        ):
            raise MediaError("Detected video timestamps must increase")
        self.previous_timestamp_ms = timestamp_ms
        return timestamp_ms

    # Stage 1: load detections ----------------------------------------------

    def load_detections(self) -> Iterator[SequenceApplyProgress]:
        """Automatic root motion needs the whole clip before the first solve."""
        if self.requested_mode != RootMotionMode.AUTOMATIC:
            self.frame_iterator = iter(self.frames)
            self.total_frames = self.total_hint
            return
        self.frame_spool = FrameSpool()
        for detected in self.frames:
            self._check_timestamp(detected)
            self.frame_spool.append(detected)
            self.root_samples.append(_root_motion_sample(detected))
            self.camera_inputs.append(
                camera_motion_correction.input_from_frame(detected)
            )
            self.frame_count += 1
            yield SequenceApplyProgress(
                "Loading detections", self.frame_count, self.total_hint
            )
        if not self.frame_count:
            raise MediaError("No detectable frames were produced")
        self.frame_iterator = iter(self.frame_spool)
        self.total_frames = self.frame_count

    # Stage 2: prepare the camera track -------------------------------------

    def prepare_camera_track(self) -> Iterator[SequenceApplyProgress]:
        camera = self.camera_motion_settings
        if not (
            self.requested_mode == RootMotionMode.AUTOMATIC
            and camera
            and camera.enabled
            and self.is_video
        ):
            return
        try:
            (
                self.camera_track,
                self.camera_track_status,
                self.camera_preflight_warning,
            ) = yield from _iter_prepare_camera_motion_track(
                self.settings, self.camera_inputs
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.camera_track = None
            self.camera_track_status = "Error"
            self.camera_preflight_warning = f"Camera Motion Correction skipped: {exc}"
        if self.camera_motion_ui_settings is not None:
            with suppress(AttributeError, ReferenceError, RuntimeError):
                self.camera_motion_ui_settings.camera_track_status = (
                    self.camera_track_status
                )

    # Stage 3: solve and stream frames --------------------------------------

    def _plan_root_motion(self, detected: CanonicalFrame, target_space) -> None:
        (
            self.planned_deltas,
            self.planned_yaws,
            self.effective_mode,
            root_warnings,
        ) = _root_motion_deltas(
            self.armature, self.entries, self.root_samples, self.requested_mode
        )
        valid = [
            sample
            for sample in self.root_samples
            if sample.observation is not None and sample.observation.valid
        ]
        if valid:
            reference = valid[0]
            lateral = target_space @ (
                reference.source_basis.inverted_safe() @ Vector((1.0, 0.0, 0.0))
            )
            normalized_up = self.target_up.normalized()
            lateral -= normalized_up * lateral.dot(normalized_up)
            if lateral.length > 1.0e-8:
                self.target_lateral = lateral.normalized()
            target_height = _target_body_height(self.armature, self.entries)
            scale = (
                target_height / reference.source_height
                if target_height > 1.0e-6 and reference.source_height > 1.0e-6
                else 1.0
            )
            self.source_to_target_scale = min(max(scale, 0.1), 10.0)
        self.object_delta_basis = self.baseline.to_3x3()
        for warning in root_warnings:
            _add_warning(self.warnings, detected, warning)
        if self.camera_preflight_warning:
            _add_warning(self.warnings, detected, self.camera_preflight_warning)

    def _begin_solving(self, detected: CanonicalFrame) -> None:
        armature = self.armature
        prepare_limb_ownership(armature)
        if self.root_enabled and not bool(
            getattr(self.settings, "keep_original_root_position", False)
        ):
            _reset_root_location(self.root_target, scene_context=self.scene_context)
        target_space = target_basis(armature, self.entries)
        self.target_up = target_space @ Vector((0.0, 0.0, 1.0))
        self.baseline = (
            self.root_target.matrix.copy()
            if self.root_kind == "BONE"
            else self.root_target.matrix_world.copy()
        )
        if self.requested_mode == RootMotionMode.AUTOMATIC:
            self._plan_root_motion(detected, target_space)
        _ensure_action(armature)
        self.keyframe_state = _KeyframeWriteState(armature)

    def _set_root(self, index: int) -> None:
        planned = self.effective_mode == RootMotionMode.AUTOMATIC
        _set_root_pose(
            self.root_kind,
            self.root_target,
            self.baseline,
            self.planned_deltas[index - 1] if planned else Vector(),
            self.planned_yaws[index - 1] if planned else 0.0,
            self.target_up,
            scene_context=self.scene_context,
        )

    def solve_frames(self) -> Iterator[SequenceApplyProgress]:
        """Solve each frame, keep it for clean-up and stream its keyframes."""
        self.solving_started = True
        processing_session = PostProcessingSession()
        automatic = self.requested_mode == RootMotionMode.AUTOMATIC
        fps = self.scene_context.fps
        for index, detected in enumerate(self.frame_iterator, 1):
            timestamp_ms = int(detected.timestamp_ms)
            if not automatic:
                timestamp_ms = self._check_timestamp(detected)
                self.frame_count = index
                yield SequenceApplyProgress(
                    "Loading detections", index, self.total_frames
                )
            destination = self.start_frame + timestamp_ms * fps / 1000.0
            self.last_destination = destination
            _scene_set_float(self.scene_context.scene, destination)
            if index == 1:
                self._begin_solving(detected)
            if self.root_enabled:
                self._set_root(index)
            frame_solver = _iter_capture_baked_frame(
                self.armature,
                detected,
                destination,
                self.settings,
                processing_session,
                self.root_kind,
                self.root_target,
                self.root_enabled,
                self.postprocess_config,
                self.rig_plan,
                self.warnings,
                self.capture_rohm_input,
                scene_context=self.scene_context,
            )
            while True:
                try:
                    phase, current, total = next(frame_solver)
                except StopIteration as stopped:
                    sample = stopped.value
                    break
                yield SequenceApplyProgress(
                    phase,
                    (index - 1) * total + current,
                    self.total_frames * total,
                )
            if self.offline_processing:
                self.samples.append(sample)
            yield SequenceApplyProgress("Solving animation", index, self.total_frames)
            yield from _iter_write_baked_sample(
                self.keyframe_state,
                sample,
                index,
                self.total_frames,
                phase="Streaming keyframes",
                scene_context=self.scene_context,
            )
        if not self.frame_count:
            raise MediaError("No detectable frames were produced")

    def restore(self) -> None:
        """Undo per-frame scene changes and drop the temporary frame stream."""
        try:
            if self.solving_started:
                _restore_sequence_state(
                    self.armature, self.state, scene_context=self.scene_context
                )
        finally:
            if self.frame_spool is not None:
                self.frame_spool.close()

    # Stage 4: offline clean-up ---------------------------------------------

    def post_process(self) -> Iterator[SequenceApplyProgress]:
        """Run the enabled offline clean-up modules over all solved samples."""
        samples = self.samples
        scene_context = self.scene_context
        context = SequenceProcessingContext(
            samples=samples,
            timestamps=_sequence_timestamps(samples),
            root_motion_mode=self.effective_mode,
            process_camera_motion=lambda module_settings: _iter_camera_motion_correction(
                self.armature,
                samples,
                module_settings,
                self.effective_mode,
                self.root_kind,
                self.root_target,
                self.target_lateral,
                self.object_delta_basis,
                self.source_to_target_scale,
                self.camera_track,
                self.camera_motion_ui_settings,
                self.camera_track_status,
                self.is_video,
                scene_context=scene_context,
            ),
            process_temporal=lambda module_settings: _iter_temporal_correction(
                self.armature,
                self.entries,
                samples,
                module_settings,
                self.effective_mode,
                self.root_kind,
                self.root_target,
                self.target_up,
                scene_context=scene_context,
            ),
            process_detection_filter=lambda module_settings: _iter_detection_filter(
                samples,
                module_settings,
                rohm_settings=self.rohm_settings,
                effective_mode=self.effective_mode,
                ui_settings=self.detection_filter_ui_settings,
                source_frame_offset=self.source_frame_offset,
            ),
            process_rohm=lambda module_settings: _iter_rohm_correction(
                self.armature,
                self.entries,
                self.rig_plan,
                samples,
                module_settings,
                self.effective_mode,
                self.root_kind,
                self.root_target,
                scene_context=scene_context,
            ),
        )
        yield from iter_sequence_pipeline(context, self.postprocess_config)

    # Stage 5: write keyframes ----------------------------------------------

    def write_keyframes(self):
        """Rewrite cleaned samples as keys, or finish the streamed keys."""
        if not self.offline_processing:
            return _finish_keyframe_stream(
                self.keyframe_state,
                self.last_destination,
                scene_context=self.scene_context,
            )
        self.keyframe_state.attach()
        return (
            yield from _iter_commit_baked_samples(
                self.armature,
                self.samples,
                self.root_kind,
                self.root_target,
                self.root_enabled,
                scene_context=self.scene_context,
            )
        )


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
    Returns ``(frames, bone samples, shape samples, warnings)``.
    """
    run = _SequenceRun(armature, frames, settings, start_frame, scene_context)
    try:
        yield from run.load_detections()
        yield from run.prepare_camera_track()
        yield from run.solve_frames()
    finally:
        run.restore()
    if run.offline_processing:
        yield from run.post_process()
    bones, shapes = yield from run.write_keyframes()
    return run.frame_count, bones, shapes, tuple(run.warnings)


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
