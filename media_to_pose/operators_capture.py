"""Asynchronous capture and application operators."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator

from . import cache_preview, detection_cache, mapping, rig_planning
from .capture_state import (
    enabled_capture_scope,
    face_output_warning,
    mapping_scope_issues,
)
from .core.cache.stream import FrameSpool
from .core.errors import MtpError
from .core.log import get_logger
from .media_io import probe_media
from .models import MediaInfo
from .operation import (
    COORDINATOR,
    CaptureRequest,
    OperationKind,
    OperationState,
    OperationToken,
)
from .operators_mapping import active_armature
from .post_processing import compile_postprocess_config
from .providers import descriptor
from .registration import register_classes, unregister_classes
from .retarget import (
    iter_apply_frame,
    iter_apply_sequence,
    prepare_limb_ownership,
    validate_limb_switch_interval,
)

LOGGER = get_logger(__name__)

_CAPTURE_JOBS = {}


@dataclass(frozen=True)
class _CaptureProgressPlan:
    """Ordered, workload-weighted ranges for one capture operation."""

    ranges: Mapping[str, tuple[float, float]]

    def range_for(self, group: str) -> tuple[float, float]:
        return self.ranges.get(group, (0.0, 100.0))

    @property
    def step_count(self) -> int:
        return len(self.ranges)

    def step_number(self, group: str) -> int:
        try:
            return tuple(self.ranges).index(str(group)) + 1
        except ValueError:
            return 0


def _enabled_sequence_modules(request: CaptureRequest) -> set[str]:
    config = getattr(request, "postprocess_config", None)
    return {
        str(module.module_id)
        for module in getattr(config, "modules", ())
        if bool(module.enabled)
    }


def _configured_sequence_module(request: CaptureRequest, module_id: str):
    config = getattr(request, "postprocess_config", None)
    return next(
        (
            module
            for module in getattr(config, "modules", ())
            if str(module.module_id) == module_id and bool(module.enabled)
        ),
        None,
    )


def _rohm_may_run(request: CaptureRequest) -> bool:
    """Return whether this request enables any RoHM-produced output."""
    module = _configured_sequence_module(request, "ROHM_CORRECTION")
    if module is None:
        return False
    pose_residual = bool(getattr(module, "learned_pose_residual", False))
    root_cleanup = bool(getattr(module, "root_trajectory_cleanup", True))
    return pose_residual or (root_cleanup and request.root_motion_mode == "AUTOMATIC")


def _capture_progress_plan(
    request: CaptureRequest,
    *,
    cached: bool,
) -> _CaptureProgressPlan:
    """Allocate progress by expected cost and real pipeline order."""
    weights: list[tuple[str, float]] = []
    if not cached:
        weights.append(("calibration", 0.5))
        # Measured wall-time ratio for the two bundled pose backends.
        detection_weight = 8.0 if request.provider_id == "SAM3D_BODY" else 1.0
        weights.append(("detection", detection_weight))
    if request.mode == "POSE":
        weights.append(("pose_application", 0.75))
    elif request.mode == "KEYFRAMES":
        modules = _enabled_sequence_modules(request)
        weights.append(("loading", 0.15))
        if (
            request.media_kind == "VIDEO"
            and request.root_motion_mode == "AUTOMATIC"
            and "CAMERA_MOTION_CORRECTION" in modules
        ):
            weights.append(("camera_preflight", 0.75))
        weights.append(("solve", 1.0))
        if "CAMERA_MOTION_CORRECTION" in modules:
            weights.append(("camera_apply", 0.25))
        if "DETECTION_FILTER" in modules:
            weights.append(("detection_filter", 0.25))
        if _rohm_may_run(request):
            # RoHM takes roughly twice one MediaPipe detection pass.
            weights.append(("rohm", 2.0))
        if "TEMPORAL_CORRECTION" in modules:
            weights.append(("temporal", 0.75))
        weights.append(("commit", 0.5))
    total = sum(weight for _name, weight in weights) or 1.0
    cursor = 0.0
    ranges: dict[str, tuple[float, float]] = {}
    for index, (name, weight) in enumerate(weights):
        end = 100.0 if index == len(weights) - 1 else cursor + 100.0 * weight / total
        ranges[name] = (cursor, end)
        cursor = end
    return _CaptureProgressPlan(ranges)


def _sequence_progress_group(phase: str) -> str:
    folded = str(phase).casefold()
    if folded == "loading detections":
        return "loading"
    if folded.startswith("camera motion:"):
        return "camera_apply" if "applying" in folded else "camera_preflight"
    if folded.startswith("filtering unreliable detections"):
        return "detection_filter"
    if folded.startswith("rohm:"):
        return "rohm"
    if folded == "writing keyframes":
        return "commit"
    if folded.startswith(
        (
            "smoothing animation",
            "stabilizing hand rotations",
            "projecting anatomical finger hinges",
            "detecting foot contacts",
            "stabilizing root motion",
            "applying stabilized root",
            "repairing planted feet",
        )
    ):
        return "temporal"
    return "solve"


def _sequence_progress_status(progress) -> str:
    phase = str(getattr(progress, "phase", ""))
    group = _sequence_progress_group(phase)
    return {
        "loading": "Preparing detected frames",
        "camera_preflight": "Tracking camera motion",
        "solve": "Solving animation",
        "camera_apply": "Applying camera correction",
        "detection_filter": "Filtering unreliable detections",
        "rohm": "Applying Experimental RoHM cleanup",
        "temporal": "Smoothing animation",
        "commit": "Writing keyframes",
    }[group]


def _set_status(settings, message: str, kind: str = "INFO") -> None:
    if getattr(settings, "status", "") != str(message):
        settings.show_result_details = False
    settings.status = str(message)
    settings.status_kind = kind


def _sequence_progress_percent(
    progress,
    previous: float = 0.0,
    plan: _CaptureProgressPlan | None = None,
) -> float:
    """Map local phase counters onto one monotonic video-bake progress bar."""
    total = max(int(getattr(progress, "total", 0) or 0), 1)
    current = min(max(int(getattr(progress, "current", 0) or 0), 0), total)
    ratio = current / total
    group = _sequence_progress_group(getattr(progress, "phase", ""))
    if plan is None:
        # Live operations always provide their request-specific plan.
        start, end = {
            "loading": (0.0, 5.0),
            "camera_preflight": (5.0, 15.0),
            "solve": (15.0, 50.0),
            "camera_apply": (50.0, 55.0),
            "detection_filter": (55.0, 60.0),
            "rohm": (60.0, 85.0),
            "temporal": (85.0, 92.0),
            "commit": (92.0, 100.0),
        }[group]
    else:
        start, end = plan.range_for(group)

    if group == "temporal":
        folded = str(getattr(progress, "phase", "")).casefold()
        subrange = (0.0, 0.45)
        if folded.startswith("detecting foot contacts"):
            subrange = (0.45, 0.55)
        elif folded.startswith("stabilizing root motion"):
            subrange = (0.55, 0.70)
        elif folded.startswith("applying stabilized root"):
            subrange = (0.70, 0.80)
        elif folded.startswith("repairing planted feet"):
            subrange = (0.80, 1.0)
        parent_start, parent_end = start, end
        start = parent_start + (parent_end - parent_start) * subrange[0]
        end = parent_start + (parent_end - parent_start) * subrange[1]

    mapped = start + (end - start) * ratio
    return min(max(float(previous), mapped), 100.0)


def _tag_progress_redraw(context) -> None:
    """Request sidebar redraws after timer-driven progress changes."""
    screens = []
    screen = getattr(context, "screen", None)
    if screen is not None:
        screens.append(screen)
    window_manager = getattr(context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        window_screen = getattr(window, "screen", None)
        if window_screen is not None and all(
            window_screen is not existing for existing in screens
        ):
            screens.append(window_screen)
    for redraw_screen in screens:
        for area in getattr(redraw_screen, "areas", ()):
            if getattr(area, "type", None) == "VIEW_3D":
                area.tag_redraw()


class CaptureJob:
    def __init__(
        self,
        token: OperationToken,
        request: CaptureRequest,
        *,
        cached: bool,
    ):
        self.token = token
        self.request = request
        self.cancel = token.cancel_event
        self.messages = queue.Queue()
        self.current = 0
        self.total = 0
        self.status = "Starting"
        self.overall_progress = 0.0
        self.thread = None
        self.temp_path = ""
        self.spool: FrameSpool | None = None
        self.transaction = None
        self.progress_plan = _capture_progress_plan(request, cached=cached)
        self.frame_total = (
            1
            if request.mode == "POSE" or request.media_kind == "IMAGE"
            else max(int(request.source_frame_count), 1)
        )
        self.frame_current = 0
        COORDINATOR.update(
            self.token,
            frame_current=0,
            frame_total=self.frame_total,
            step_current=0,
            step_total=self.progress_plan.step_count,
        )

    def phase_progress(self, group, current, total, status):
        start, end = self.progress_plan.range_for(str(group))
        ratio = float(current) / float(total) if total else 0.0
        mapped = (start + (end - start) * min(max(ratio, 0.0), 1.0)) / 100.0
        self.overall_progress = max(self.overall_progress, mapped)
        counters = {
            "step_current": self.progress_plan.step_number(str(group)),
            "step_total": self.progress_plan.step_count,
        }
        if str(group) == "detection":
            if self.request.mode != "POSE" and int(total) > 0:
                self.frame_total = max(self.frame_total, int(total))
            self.frame_current = min(max(int(current), 0), self.frame_total)
            counters.update(
                frame_current=self.frame_current,
                frame_total=self.frame_total,
            )
        if not COORDINATOR.update(
            self.token,
            status=status,
            phase=str(group),
            progress=self.overall_progress,
            step_progress=ratio,
            **counters,
        ):
            return
        self.current = current
        self.total = total
        self.status = status

    def progress(self, current, total, status):
        self.phase_progress("detection", current, total, status)

    def sequence_progress(self, group, current, total, status, overall):
        ratio = float(current) / float(total) if total else 0.0
        start, end = self.progress_plan.range_for(str(group))
        step_ratio = (
            min(max((float(overall) - start) / (end - start), 0.0), 1.0)
            if end > start
            else 1.0
        )
        self.frame_current = min(
            max(int(round(self.frame_total * min(max(ratio, 0.0), 1.0))), 0),
            self.frame_total,
        )
        COORDINATOR.update(
            self.token,
            status=status,
            phase=str(group),
            progress=float(overall) / 100.0,
            frame_current=self.frame_current,
            frame_total=self.frame_total,
            step_current=self.progress_plan.step_number(str(group)),
            step_total=self.progress_plan.step_count,
            step_progress=step_ratio,
        )


def _job_phase_progress(job, group, current, total, status):
    reporter = getattr(job, "phase_progress", None)
    if reporter is None:
        job.progress(current, total, status)
        return
    reporter(group, current, total, status)


class _CaptureTransaction:
    """Restore the exact originating Blender state on failure/cancellation."""

    def __init__(self, scene, armature, *, animation: bool):
        self.scene_name = scene.name
        self.scene_pointer = scene.as_pointer()
        self.armature = armature
        self.animation = bool(animation)
        self.committed = False
        self.frame = (int(scene.frame_current), float(scene.frame_subframe))
        self.active_name = (
            bpy.context.view_layer.objects.active.name
            if bpy.context.view_layer.objects.active
            else ""
        )
        self.selected_names = tuple(obj.name for obj in bpy.context.selected_objects)
        self.mode = str(getattr(armature, "mode", "OBJECT"))
        self.bone_basis = {
            bone.name: bone.matrix_basis.copy() for bone in armature.pose.bones
        }
        self.bone_properties = {
            bone.name: {
                name: value
                for name in bone.keys()  # noqa: SIM118 - Blender IDProperty API
                if isinstance((value := bone.get(name)), (bool, int, float))
            }
            for bone in armature.pose.bones
        }
        self.shape_values = {}
        for obj in bpy.data.objects:
            if obj.type != "MESH" or obj.find_armature() is not armature:
                continue
            keys = getattr(obj.data, "shape_keys", None)
            for block in getattr(keys, "key_blocks", ()) if keys else ():
                self.shape_values[(keys, block.name)] = float(block.value)
        animation_data = armature.animation_data
        self.original_action = animation_data.action if animation_data else None
        self.temporary_action = None
        self.shape_action_transactions = []
        if self.animation:
            animation_data = armature.animation_data_create()
            self.temporary_action = (
                self.original_action.copy()
                if self.original_action is not None
                else bpy.data.actions.new(f"{armature.name}_MediaToPose_Staging")
            )
            self.temporary_action.name = f"{armature.name}_MediaToPose_Staging"
            animation_data.action = self.temporary_action
            seen_shape_keys = set()
            for keys, _name in self.shape_values:
                pointer = int(keys.as_pointer())
                if pointer in seen_shape_keys:
                    continue
                seen_shape_keys.add(pointer)
                shape_animation = keys.animation_data_create()
                original = shape_animation.action
                temporary = (
                    original.copy()
                    if original is not None
                    else bpy.data.actions.new(f"{keys.name}_MediaToPose_Staging")
                )
                temporary.name = f"{keys.name}_MediaToPose_Staging"
                shape_animation.action = temporary
                self.shape_action_transactions.append((keys, original, temporary))

    def commit(self) -> None:
        if self.temporary_action is not None:
            self.temporary_action.name = f"{self.armature.name}_MediaToPose"
        for keys, _original, temporary in self.shape_action_transactions:
            temporary.name = f"{keys.name}_MediaToPose"
        self.committed = True

    def rollback(self) -> None:
        if self.committed:
            return
        armature = self.armature
        try:
            for name, basis in self.bone_basis.items():
                bone = armature.pose.bones.get(name)
                if bone is None:
                    continue
                bone.matrix_basis = basis
                original = self.bone_properties.get(name, {})
                for property_name in tuple(bone.keys()):
                    if (
                        isinstance(bone.get(property_name), (bool, int, float))
                        and property_name not in original
                    ):
                        del bone[property_name]
                for property_name, value in original.items():
                    bone[property_name] = value
            for (keys, name), value in self.shape_values.items():
                block = keys.key_blocks.get(name)
                if block is not None:
                    block.value = value
            animation_data = armature.animation_data_create()
            animation_data.action = self.original_action
            for keys, original, temporary in self.shape_action_transactions:
                shape_animation = keys.animation_data_create()
                shape_animation.action = original
                if temporary.users == 0:
                    bpy.data.actions.remove(temporary)
            self.shape_action_transactions.clear()
            if self.temporary_action is not None:
                temporary = self.temporary_action
                self.temporary_action = None
                if temporary.users == 0:
                    bpy.data.actions.remove(temporary)
            scene = bpy.data.scenes.get(self.scene_name)
            if scene is not None and scene.as_pointer() == self.scene_pointer:
                scene.frame_set(self.frame[0], subframe=self.frame[1])
            bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object else None
            bpy.ops.object.select_all(action="DESELECT")
            for name in self.selected_names:
                obj = bpy.data.objects.get(name)
                if obj is not None:
                    obj.select_set(True)
            active = bpy.data.objects.get(self.active_name)
            if active is not None:
                bpy.context.view_layer.objects.active = active
                if self.mode != "OBJECT" and active is armature:
                    bpy.ops.object.mode_set(mode=self.mode)
            armature.update_tag(refresh={"OBJECT"})
            bpy.context.view_layer.update()
        except (AttributeError, ReferenceError, RuntimeError) as exc:
            LOGGER.warning(
                "Capture rollback could not restore every UI state element: %s: %s",
                type(exc).__name__,
                exc,
            )


def _capture_job(token: OperationToken | None = None):
    token = token or COORDINATOR.active_token()
    if token is None or token.kind not in {
        OperationKind.CAPTURE,
        OperationKind.CACHE_APPLY,
    }:
        return None
    return _CAPTURE_JOBS.get(token.generation)


def _capture_request(
    context,
    armature,
    settings,
    media_info,
    options,
    mode: str,
    source_index: int,
    *,
    analysis_path: str = "",
    source_identity=None,
) -> CaptureRequest:
    scene = context.scene
    action = (
        armature.animation_data.action
        if armature is not None and armature.animation_data
        else None
    )
    config = armature.mtp_rig_config if armature is not None else None
    mapping_entries = (
        rig_planning.snapshot_mapping_entries(mapping.mapping_by_id(config))
        if config is not None
        else {}
    )
    postprocess_config = (
        compile_postprocess_config(config) if config is not None else None
    )
    return CaptureRequest(
        provider_id=str(settings.provider),
        mode=str(mode),
        media_path=str(media_info.path),
        media_kind=str(media_info.kind),
        source_frame_index=max(0, int(source_index)),
        use_body=bool(settings.use_body),
        use_hands=bool(settings.use_hands),
        use_face=bool(settings.use_face),
        face_output=str(settings.face_output),
        root_motion_mode=str(settings.root_motion_mode),
        keep_original_root_position=bool(settings.keep_original_root_position),
        keep_detection_cache=bool(settings.keep_detection_cache),
        cache_limit_bytes=int(settings.cache_limit_mb) * 1024 * 1024,
        start_frame=float(scene.frame_current),
        scene_name=scene.name,
        scene_pointer=scene.as_pointer(),
        source_frame_count=max(0, int(media_info.frame_count)),
        target_name=armature.name if armature is not None else "",
        target_pointer=armature.as_pointer() if armature is not None else 0,
        action_name=action.name if action is not None else "",
        object_mode=(str(armature.mode) if armature is not None else "OBJECT"),
        selected_object_names=tuple(obj.name for obj in context.selected_objects),
        analysis_path=str(analysis_path or ""),
        source_identity=dict(source_identity or {}),
        options=options,
        mapping_entries=mapping_entries,
        postprocess_config=postprocess_config,
    )


def _begin_capture_operation(
    request: CaptureRequest,
    *,
    cached: bool,
) -> CaptureJob | None:
    token = COORDINATOR.acquire(
        OperationKind.CACHE_APPLY if cached else OperationKind.CAPTURE,
        f"capture:{request.mode.casefold()}",
        scene_name=request.scene_name,
        scene_pointer=request.scene_pointer,
        target_name=request.target_name,
        target_pointer=request.target_pointer,
        status="Loading cached detections" if cached else "Starting capture",
    )
    if token is None:
        return None
    job = CaptureJob(token, request, cached=cached)
    _CAPTURE_JOBS[token.generation] = job
    return job


def _remove_worker_spool(job) -> None:
    spool = getattr(job, "spool", None)
    if spool is not None:
        spool.close()
        job.spool = None


def _worker(
    job,
    provider_id,
    options,
    media_info,
    cache_media_info,
    mode,
    source_index,
    cache_key,
    cache_identity,
    keep_cache,
    cache_limit_bytes,
):
    provider = None
    writer = None
    try:
        from . import camera_calibration, camera_calibration_pipeline

        options = dict(options)
        media_width = int(getattr(media_info, "width", 0))
        media_height = int(getattr(media_info, "height", 0))
        if media_width > 0 and media_height > 0:
            camera = camera_calibration_pipeline.resolve(
                media_info.path,
                media_info.kind,
                media_width,
                media_height,
                options,
                cancel_event=job.cancel,
                progress=(
                    lambda current, total, message: _job_phase_progress(
                        job,
                        "calibration",
                        current,
                        total,
                        message,
                    )
                ),
            )
            options["camera_intrinsics"] = camera_calibration.to_mapping(camera)
        _job_phase_progress(job, "calibration", 1, 1, "Camera calibration ready")
        provider_class = descriptor(provider_id).provider_class
        # Calibration policy belongs to the capture pipeline, not to provider
        # constructors. Providers receive only the resolved full-image camera
        # observation; cache/model-selection metadata must stop here.
        for control_key in (
            "perspective_correction",
            "manual_fov_override",
            "manual_vertical_fov",
            "anycalib_device",
            "camera_bundle_revision",
            "camera_precision",
            "anycalib_installed",
            "anycalib_bundle_directory",
        ):
            options.pop(control_key, None)
        if mode == "POSE":
            job.progress(0, 1, "Loading capture provider")
        provider = provider_class(video_mode=media_info.kind == "VIDEO", **options)
        if job.cancel.is_set():
            raise InterruptedError("Processing cancelled")
        if mode == "POSE":
            job.progress(
                0,
                1,
                (
                    "Detecting image"
                    if media_info.kind == "IMAGE"
                    else "Detecting video frame"
                ),
            )
            if media_info.kind == "IMAGE":
                detected = provider.detect_image(media_info.path)
            else:
                detected = provider.detect_video_frame_index(
                    media_info.path,
                    source_index,
                )
            if job.cancel.is_set():
                raise InterruptedError("Processing cancelled")
            job.progress(1, 1, "Detection complete")
            job.messages.put(("FRAME", detected))
            return
        if media_info.kind == "IMAGE":
            detected = provider.detect_image(media_info.path)
            if job.cancel.is_set():
                raise InterruptedError("Processing cancelled")
            job.progress(1, 1, "Detection complete")
            job.messages.put(("FRAMES", [detected]))
            return

        if keep_cache:
            writer = detection_cache.SequenceWriter(
                cache_key,
                cache_identity,
                cache_media_info,
                cache_limit_bytes,
            )
            job.temp_path = str(writer.directory)
            count = 0
            for detected in provider.iter_video(
                media_info.path,
                job.cancel,
                job.progress,
            ):
                writer.add(detected)
                count += 1
            job.progress(1, 1, "Detection complete")
            directory, persisted = writer.finish()
            writer = None
            job.temp_path = "" if persisted else str(directory)
            job.messages.put(("CACHE_DIR", (str(directory), count, persisted)))
            return

        spool = FrameSpool(prefix="media_to_pose_capture_")
        job.spool = spool
        for detected in provider.iter_video(media_info.path, job.cancel, job.progress):
            spool.append(detected)
        spool.flush()
        job.progress(1, 1, "Detection complete")
        job.messages.put(("CACHE", (spool, len(spool))))
    except InterruptedError as exc:
        _remove_worker_spool(job)
        job.messages.put(("CANCELLED", str(exc)))
    except Exception as exc:  # Worker thread entry point.
        LOGGER.exception("Capture worker failed")
        _remove_worker_spool(job)
        job.messages.put(("ERROR", (str(exc), traceback.format_exc())))
    finally:
        if writer is not None:
            try:
                writer.discard()
            except Exception:
                LOGGER.exception("Detection-cache cleanup failed")
        if provider is not None:
            try:
                provider.close()
            except Exception:
                LOGGER.exception("Capture-provider cleanup failed")


class MTP_OT_apply_capture(Operator):
    bl_idname = "mtp.apply_capture"
    bl_label = "Apply Vision Capture"
    bl_description = "Detect and retarget the selected image or video"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        items=(
            ("POSE", "Pose", ""),
            ("KEYFRAMES", "Keyframes", ""),
            ("ANALYZE", "Analyze", ""),
        ),
        default="POSE",
        options={"HIDDEN"},
    )
    force_analysis: BoolProperty(default=False, options={"HIDDEN"})

    _timer = None
    _armature_name = ""
    _media_info = None
    _cache_media_info = None
    _start_frame = 0.0
    _cache_key = ""
    _cache_identity = None
    _source_index = 0
    _sequence_iterator = None
    _sequence_phase = ""
    _sequence_warnings = None
    _sequence_cached = False
    _pose_iterator = None
    _pose_frame = None
    _pose_cached = False
    _token = None
    _job = None
    _request = None
    _last_ui_update = 0.0

    def _origin_scene(self):
        request = self._request
        if request is None:
            return None
        scene = bpy.data.scenes.get(request.scene_name)
        if scene is None or scene.as_pointer() != request.scene_pointer:
            return None
        return scene

    def _origin_armature(self):
        request = self._request
        if request is None or not request.target_name:
            return None
        armature = bpy.data.objects.get(request.target_name)
        if (
            armature is None
            or armature.as_pointer() != request.target_pointer
            or armature.type != "ARMATURE"
        ):
            return None
        return armature

    def invoke(self, context, _event):
        self._token = None
        self._job = None
        self._request = None
        self._sequence_iterator = None
        self._sequence_phase = ""
        self._sequence_warnings = set()
        self._sequence_cached = False
        self._pose_iterator = None
        self._pose_frame = None
        self._pose_cached = False
        if COORDINATOR.is_busy():
            self.report({"ERROR"}, "Media to Pose is already processing media")
            return {"CANCELLED"}
        analysis_only = self.mode == "ANALYZE"
        armature = active_armature(context)
        if not analysis_only and not armature:
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        settings = context.scene.mtp_settings
        provider_descriptor = descriptor(settings.provider)
        if analysis_only and not settings.keep_detection_cache:
            self.report({"ERROR"}, "Enable Keep Detection Cache first")
            return {"CANCELLED"}
        effective_scope = enabled_capture_scope(
            settings,
            provider_descriptor.capabilities,
        )
        if not effective_scope:
            self.report(
                {"ERROR"},
                f"Enable a capture type supported by {provider_descriptor.label}",
            )
            return {"CANCELLED"}
        source = detection_cache.resolve_media_source(settings)
        if not source.usable:
            message = source.message
            _set_status(settings, message, "ERROR")
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        options = detection_cache.options_from_settings(settings)
        # Sequence analysis stores raw observations. Temporal Correction owns
        # baked-video filtering; provider smoothing remains preview-only.
        if self.mode != "POSE":
            options["smoothing"] = 0.0
        if not analysis_only:
            prepare_limb_ownership(armature)
            from . import mapping

            mapping.validate_mapping(armature)
            errors = mapping_scope_issues(armature, settings)
            if errors:
                self.report({"ERROR"}, errors[0])
                _set_status(settings, errors[0], "ERROR")
                return {"CANCELLED"}
            warning = face_output_warning(armature, settings)
            if warning:
                _set_status(settings, warning, "WARNING")
                self.report({"WARNING"}, warning)
        path = source.analysis_path
        if path:
            try:
                media_info = probe_media(path)
            except (MtpError, OSError) as exc:
                _set_status(settings, str(exc), "ERROR")
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
        else:
            status = detection_cache.status_for_settings(settings)
            if not status.matches or not status.cached_frames:
                message = source.message
                _set_status(settings, message, "ERROR")
                self.report({"ERROR"}, message)
                return {"CANCELLED"}
            media_info = detection_cache.media_info_from_status(status)
        cache_media_info = MediaInfo(
            path=source.display_path or media_info.path,
            kind=media_info.kind,
            fps=media_info.fps,
            frame_count=media_info.frame_count,
            duration_seconds=media_info.duration_seconds,
            width=media_info.width,
            height=media_info.height,
        )
        selected_source_index = max(0, int(settings.video_frame) - 1)
        if (
            self.mode == "POSE"
            and media_info.kind == "VIDEO"
            and media_info.frame_count > 0
            and selected_source_index >= media_info.frame_count
        ):
            self.report(
                {"ERROR"},
                f"Frame must be between 1 and {media_info.frame_count}",
            )
            return {"CANCELLED"}
        if self.mode == "KEYFRAMES" and media_info.kind == "VIDEO":
            rohm = next(
                (
                    item
                    for item in armature.mtp_rig_config.post_processing_modules
                    if item.module_id == "ROHM_CORRECTION"
                ),
                None,
            )
            if (
                rohm is not None
                and rohm.enabled
                and (
                    rohm.learned_pose_residual
                    or (
                        rohm.root_trajectory_cleanup
                        and settings.root_motion_mode == "AUTOMATIC"
                    )
                )
            ):
                from .backends.rohm import backend as rohm_backend

                model = rohm_backend.backend_snapshot(settings.rohm_device)
                if not model.installed:
                    message = (
                        "RoHM Correction is enabled but its verified model "
                        f"is unavailable: {model.message}"
                    )
                    _set_status(settings, message, "ERROR")
                    self.report({"ERROR"}, message)
                    return {"CANCELLED"}
                try:
                    # Load and verify before the capture request so a bad model
                    # cannot partially mutate the animation transaction.
                    rohm_backend.ensure_model_loaded(settings.rohm_device)
                except Exception as exc:  # Operator boundary.
                    LOGGER.exception("RoHM model could not be loaded")
                    message = (
                        f"RoHM Correction could not load its verified model: {exc}"
                    )
                    _set_status(settings, message, "ERROR")
                    self.report({"ERROR"}, message)
                    return {"CANCELLED"}
            scene_fps = context.scene.render.fps / context.scene.render.fps_base
            start = float(context.scene.frame_current)
            issue = validate_limb_switch_interval(
                armature,
                start,
                start + media_info.duration_seconds * scene_fps,
            )
            if issue:
                _set_status(settings, issue, "ERROR")
                self.report({"ERROR"}, issue)
                return {"CANCELLED"}

        cache_key, cache_identity = detection_cache.cache_identity_from_media(
            source.source_media,
            settings.provider,
            options,
        )
        source_index = selected_source_index if media_info.kind == "VIDEO" else 0
        request = _capture_request(
            context,
            armature,
            settings,
            media_info,
            options,
            self.mode,
            source_index,
            analysis_path=path,
            source_identity=source.source_media,
        )
        self._request = request

        if settings.keep_detection_cache:
            if self.mode in {"POSE", "KEYFRAMES"}:
                cached = detection_cache.load_frame(
                    cache_key,
                    source_index if self.mode == "POSE" else 0,
                )
                if self.mode == "POSE" and cached is not None:
                    return self._apply_cached_pose(
                        context,
                        armature,
                        cached,
                        request,
                    )
                if self.mode == "KEYFRAMES":
                    cached_frames = (
                        iter((cached,))
                        if media_info.kind == "IMAGE" and cached is not None
                        else detection_cache.iter_frames(cache_key)
                    )
                    if cached_frames is not None:
                        return self._apply_cached_sequence(
                            context,
                            armature,
                            cached_frames,
                            request,
                        )
            elif self.mode == "ANALYZE" and not self.force_analysis:
                status = detection_cache.status_for_settings(settings)
                if status.matches and (
                    status.complete
                    or media_info.kind == "IMAGE"
                    and status.cached_frames
                ):
                    _set_status(settings, "Using cached analysis", "INFO")
                    cache_preview.schedule_refresh()
                    self.report({"INFO"}, settings.status)
                    return {"FINISHED"}

        if not path:
            message = (
                "The retained detections do not contain the requested frame; "
                "locate the original media"
            )
            _set_status(settings, message, "ERROR")
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        # Cached canonical detections do not require provider imports.  Probe
        # dependencies only when inference is actually about to start.
        if settings.provider == "MEDIAPIPE":
            from .dependency_bootstrap import refresh_dependency_snapshot

            refresh_dependency_snapshot()
        available, message = provider_descriptor.dependency_status(options)
        if not available:
            _set_status(settings, message, "ERROR")
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        job = _begin_capture_operation(request, cached=False)
        if job is None:
            self.report({"ERROR"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        self._job = job
        self._token = job.token
        self._armature_name = armature.name if armature else ""
        self._media_info = media_info
        self._cache_media_info = cache_media_info
        self._start_frame = request.start_frame
        self._cache_key = cache_key
        self._cache_identity = cache_identity
        self._source_index = source_index
        settings.progress = 0.0
        _set_status(settings, provider_descriptor.loading_message, "PROGRESS")

        job.thread = threading.Thread(
            target=_worker,
            args=(
                job,
                request.provider_id,
                dict(request.options),
                media_info,
                cache_media_info,
                self.mode,
                request.source_frame_index,
                cache_key,
                cache_identity,
                request.keep_detection_cache,
                request.cache_limit_bytes,
            ),
            daemon=True,
            name="MediaToPoseInference",
        )
        COORDINATOR.attach_worker(job.token, thread=job.thread)
        try:
            job.thread.start()
        except RuntimeError as exc:
            COORDINATOR.finish(job.token, OperationState.FAILED, str(exc))
            COORDINATOR.release(job.token)
            _CAPTURE_JOBS.pop(job.token.generation, None)
            _set_status(settings, f"Could not start pose detection: {exc}", "ERROR")
            self.report({"ERROR"}, f"Could not start pose detection: {exc}")
            return {"CANCELLED"}
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _apply_cached_pose(self, context, armature, detected, request):
        settings = context.scene.mtp_settings
        job = _begin_capture_operation(request, cached=True)
        if job is None:
            self.report({"ERROR"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        _job_phase_progress(
            job,
            "pose_application",
            0,
            1,
            "Loading cached detection",
        )
        self._job = job
        self._token = job.token
        self._armature_name = armature.name
        self._start_frame = request.start_frame
        settings.progress = 55.0
        self._begin_pose(context, armature, detected, cached=True)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _begin_pose(self, context, armature, detected, *, cached=False):
        scene = self._origin_scene()
        if scene is None:
            raise RuntimeError("Originating scene was removed")
        if self._job.transaction is None:
            self._job.transaction = _CaptureTransaction(
                scene, armature, animation=False
            )
        self._pose_frame = detected
        if detected.camera is not None and detected.camera.valid:
            scene.mtp_settings.last_camera_vertical_fov = detected.camera.vertical_fov
            scene.mtp_settings.last_camera_source = detected.camera.source
        self._pose_cached = bool(cached)
        self._pose_iterator = iter_apply_frame(
            armature,
            detected,
            self._request,
        )
        _set_status(
            scene.mtp_settings,
            "Loading cached detection" if cached else "Preparing detected pose",
            "PROGRESS",
        )
        application_start, _application_end = self._job.progress_plan.range_for(
            "pose_application"
        )
        scene.mtp_settings.progress = max(
            scene.mtp_settings.progress,
            application_start,
        )
        COORDINATOR.update(
            self._token,
            status=scene.mtp_settings.status,
            phase="pose_application",
            progress=scene.mtp_settings.progress / 100.0,
            frame_current=1,
            frame_total=1,
            step_current=self._job.progress_plan.step_number("pose_application"),
            step_total=self._job.progress_plan.step_count,
            step_progress=0.0,
        )
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
        self._timer = context.window_manager.event_timer_add(
            0.03, window=context.window
        )

    def _advance_pose(self, context):
        job = self._job
        scene = self._origin_scene()
        if scene is None or not COORDINATOR.owns(self._token):
            return self._finish(context, cancelled=True)
        settings = scene.mtp_settings
        if job is not None and job.cancel.is_set():
            self._pose_iterator.close()
            self._pose_iterator = None
            _set_status(
                settings,
                "Cancelled · the previous pose and animation were restored",
                "INFO",
            )
            return self._finish(context, cancelled=True)
        try:
            progress = next(self._pose_iterator)
        except StopIteration as stopped:
            self._pose_iterator = None
            bones, shapes = stopped.value
            armature = self._origin_armature()
            target_name = armature.name if armature is not None else "target character"
            prefix = "Applied cached pose" if self._pose_cached else "Pose applied"
            _set_status(
                settings,
                f"{prefix} to {target_name} · {bones} bones"
                + (f" and {shapes} shape keys" if shapes else ""),
                "INFO",
            )
            if self._pose_frame.warnings:
                settings.status += "; " + "; ".join(self._pose_frame.warnings)
                settings.status_kind = "WARNING"
            cache_preview.schedule_refresh()
            self.report({"INFO"}, settings.status)
            return self._finish(context, cancelled=False)
        except Exception as exc:
            self._pose_iterator = None
            LOGGER.exception("Capture failed")
            self.report({"ERROR"}, str(exc))
            _set_status(settings, str(exc), "ERROR")
            return self._finish(context, cancelled=True)

        _set_status(settings, progress.phase, "PROGRESS")
        application_start, application_end = job.progress_plan.range_for(
            "pose_application"
        )
        local_ratio = (
            float(progress.current) / float(progress.total) if progress.total else 0.0
        )
        settings.progress = max(
            settings.progress,
            application_start + (application_end - application_start) * local_ratio,
        )
        COORDINATOR.update(
            self._token,
            status=progress.phase,
            phase="pose_application",
            progress=settings.progress / 100.0,
            frame_current=1,
            frame_total=1,
            step_current=job.progress_plan.step_number("pose_application"),
            step_total=job.progress_plan.step_count,
            step_progress=local_ratio,
        )
        return {"RUNNING_MODAL"}

    def _apply_cached_sequence(self, context, armature, frames, request):
        settings = context.scene.mtp_settings
        job = _begin_capture_operation(request, cached=True)
        if job is None:
            self.report({"ERROR"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        self._job = job
        self._token = job.token
        self._armature_name = armature.name
        self._start_frame = request.start_frame
        settings.progress = 0.0
        self._begin_sequence(context, armature, frames, cached=True)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _begin_sequence(self, context, armature, frames, *, cached=False):
        scene = self._origin_scene()
        if scene is None:
            raise RuntimeError("Originating scene was removed")
        if self._job.transaction is None:
            self._job.transaction = _CaptureTransaction(scene, armature, animation=True)
        self._sequence_warnings = set()

        def observed_frames():
            for detected in frames:
                if detected.camera is not None and detected.camera.valid:
                    scene.mtp_settings.last_camera_vertical_fov = (
                        detected.camera.vertical_fov
                    )
                    scene.mtp_settings.last_camera_source = detected.camera.source
                self._sequence_warnings.update(detected.warnings)
                yield detected
                self._sequence_warnings.update(detected.warnings)

        self._sequence_iterator = iter_apply_sequence(
            armature,
            observed_frames(),
            self._request,
            self._start_frame,
        )
        self._sequence_phase = "Preparing detected frames"
        self._sequence_progress = self._job.progress_plan.range_for("loading")[0]
        self._sequence_cached = bool(cached)
        self._job.frame_current = 0
        scene.mtp_settings.progress = self._sequence_progress
        COORDINATOR.update(
            self._token,
            status=self._sequence_phase,
            phase="loading",
            progress=self._sequence_progress / 100.0,
            frame_current=0,
            frame_total=self._job.frame_total,
            step_current=self._job.progress_plan.step_number("loading"),
            step_total=self._job.progress_plan.step_count,
            step_progress=0.0,
        )
        _set_status(
            scene.mtp_settings,
            self._sequence_phase,
            "PROGRESS",
        )
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
        self._timer = context.window_manager.event_timer_add(
            0.03, window=context.window
        )

    def _advance_sequence(self, context):
        job = self._job
        scene = self._origin_scene()
        if scene is None or not COORDINATOR.owns(self._token):
            return self._finish(context, cancelled=True)
        settings = scene.mtp_settings
        if job is not None and job.cancel.is_set():
            self._sequence_iterator.close()
            self._sequence_iterator = None
            _set_status(
                settings,
                "Cancelled · the previous pose and animation were restored",
                "INFO",
            )
            return self._finish(context, cancelled=True)

        deadline = time.perf_counter() + 0.008
        while True:
            try:
                progress = next(self._sequence_iterator)
            except StopIteration as stopped:
                self._sequence_iterator = None
                count, bones, shapes, warnings = stopped.value
                self._sequence_warnings.update(warnings)
                armature = self._origin_armature()
                target_name = (
                    armature.name if armature is not None else "target character"
                )
                _set_status(
                    settings,
                    f"Video motion applied to {target_name} from frame "
                    f"{self._start_frame:g} · {count} frame(s), "
                    f"{bones} bone and {shapes} shape samples",
                    "INFO",
                )
                if self._sequence_warnings:
                    settings.status += "; " + "; ".join(sorted(self._sequence_warnings))
                    settings.status_kind = "WARNING"
                cache_preview.schedule_refresh()
                self.report({"INFO"}, settings.status)
                return self._finish(context, cancelled=False)
            except Exception as exc:
                self._sequence_iterator = None
                LOGGER.exception("Capture failed")
                self.report({"ERROR"}, str(exc))
                _set_status(settings, str(exc), "ERROR")
                return self._finish(context, cancelled=True)

            self._sequence_phase = progress.phase
            display_status = _sequence_progress_status(progress)
            _set_status(settings, display_status, "PROGRESS")
            self._sequence_progress = _sequence_progress_percent(
                progress,
                self._sequence_progress,
                self._job.progress_plan,
            )
            settings.progress = self._sequence_progress
            group = _sequence_progress_group(progress.phase)
            self._job.sequence_progress(
                group,
                progress.current,
                progress.total,
                display_status,
                self._sequence_progress,
            )
            if str(progress.phase).startswith("RoHM:"):
                # One diffusion step per modal tick guarantees an event-loop
                # handoff even when ONNX calls finish inside the normal budget.
                return {"RUNNING_MODAL"}
            if time.perf_counter() >= deadline:
                return {"RUNNING_MODAL"}

    def modal(self, context, event):
        job = self._job
        if job is None or not COORDINATOR.owns(self._token):
            return self._finish(context, cancelled=True)
        scene = self._origin_scene()
        if scene is None:
            COORDINATOR.request_cancel(
                self._token, status="Originating scene was removed"
            )
            return self._finish(context, cancelled=True)
        settings = scene.mtp_settings
        if event.type == "ESC":
            COORDINATOR.request_cancel(self._token)
            _set_status(settings, "Cancelling", "PROGRESS")
            _tag_progress_redraw(context)
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if context.scene.as_pointer() != self._request.scene_pointer:
            COORDINATOR.request_cancel(
                self._token,
                status="Cancelling because the active scene changed",
            )
            _set_status(
                settings,
                "Cancelling because the active scene changed",
                "PROGRESS",
            )
            return {"RUNNING_MODAL"}
        # Timer events update RNA properties without an accompanying mouse or
        # keyboard event. Explicitly tag the 3D View so the sidebar progress
        # bar repaints immediately instead of waiting for cursor movement.
        now = time.monotonic()
        if now - self._last_ui_update >= 0.1:
            _tag_progress_redraw(context)
            self._last_ui_update = now

        if self._pose_iterator is not None:
            return self._advance_pose(context)
        if self._sequence_iterator is not None:
            return self._advance_sequence(context)

        operation = COORDINATOR.snapshot()
        _set_status(settings, operation.status or job.status, "PROGRESS")
        settings.progress = operation.progress * 100.0
        try:
            kind, payload = job.messages.get_nowait()
        except queue.Empty:
            return {"RUNNING_MODAL"}

        if kind == "ERROR":
            # The worker already logged the traceback.
            message, _details = payload
            self.report({"ERROR"}, message)
            _set_status(settings, message, "ERROR")
            return self._finish(context, cancelled=True)
        if kind == "CANCELLED":
            _set_status(
                settings,
                payload or "Cancelled · the previous pose and animation were restored",
                "INFO",
            )
            return self._finish(context, cancelled=True)

        analysis_only = self.mode == "ANALYZE"
        armature = self._origin_armature()
        if not analysis_only and not armature:
            self.report({"ERROR"}, "Target armature was removed")
            return self._finish(context, cancelled=True)

        try:
            if self._request.keep_detection_cache and kind in {"FRAME", "FRAMES"}:
                detected = payload if kind == "FRAME" else payload[0]
                detection_cache.store_point(
                    self._cache_key,
                    self._cache_identity,
                    self._cache_media_info,
                    self._source_index,
                    detected,
                )
                cache_preview.refresh_after_cache_update()

            if analysis_only:
                if kind in {"FRAME", "FRAMES"}:
                    _set_status(settings, "Image analysis cached", "INFO")
                elif kind == "CACHE_DIR":
                    _path, count, persisted = payload
                    _set_status(
                        settings,
                        (
                            f"Cached {count} frame(s)"
                            if persisted
                            else (
                                f"Analyzed {count} frame(s); cache limit "
                                "exceeded, result not retained"
                            )
                        ),
                        "INFO" if persisted else "WARNING",
                    )
                else:
                    raise RuntimeError(f"Unexpected analysis result: {kind}")
                if kind == "CACHE_DIR" and payload[2]:
                    cache_preview.refresh_after_cache_update()
                elif kind not in {"FRAME", "FRAMES"}:
                    cache_preview.schedule_refresh()
                self.report({"INFO"}, settings.status)
                return self._finish(context, cancelled=False)

            _set_status(settings, "Applying capture", "PROGRESS")
            if kind == "FRAME":
                self._begin_pose(context, armature, payload)
                return {"RUNNING_MODAL"}
            if kind == "FRAMES":
                self._begin_sequence(context, armature, payload)
                return {"RUNNING_MODAL"}
            if kind in {"CACHE", "CACHE_DIR"}:
                if kind == "CACHE":
                    spool, _count = payload
                    source_frames = iter(spool)
                else:
                    path, _count, _persisted = payload
                    source_frames = detection_cache.iter_directory(path)
                self._begin_sequence(context, armature, source_frames)
                return {"RUNNING_MODAL"}
            raise RuntimeError(f"Unexpected worker result: {kind}")
        except Exception as exc:
            LOGGER.exception("Capture failed")
            self.report({"ERROR"}, str(exc))
            _set_status(settings, str(exc), "ERROR")
            return self._finish(context, cancelled=True)
        self.report({"INFO"}, settings.status)
        return self._finish(context, cancelled=False)

    def _finish(self, context, *, cancelled):
        owns_operation = COORDINATOR.owns(self._token)
        scene = self._origin_scene()
        settings = getattr(scene, "mtp_settings", None) if scene else None
        if self._sequence_iterator is not None:
            self._sequence_iterator.close()
            self._sequence_iterator = None
        if self._pose_iterator is not None:
            self._pose_iterator.close()
            self._pose_iterator = None
        self._pose_frame = None
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if not owns_operation:
            return {"CANCELLED"} if cancelled else {"FINISHED"}
        job = self._job
        transaction = job.transaction if job else None
        if transaction is not None:
            if cancelled:
                transaction.rollback()
            else:
                transaction.commit()
        if job and job.thread and job.thread.is_alive():
            job.thread.join(timeout=0.1)
        if settings is not None:
            settings.progress = 0.0
        failed = bool(settings and settings.status_kind == "ERROR")
        terminal_state = (
            OperationState.FAILED
            if failed
            else OperationState.CANCELLED
            if cancelled
            else OperationState.SUCCEEDED
        )
        terminal_status = (
            settings.status
            if settings is not None
            else "Cancelled"
            if cancelled
            else "Finished"
        )
        COORDINATOR.finish(self._token, terminal_state, terminal_status)
        released = COORDINATOR.release(self._token)
        if released:
            _CAPTURE_JOBS.pop(self._token.generation, None)
        _tag_progress_redraw(context)
        if job and released:
            _remove_worker_spool(job)
        if job and released and job.temp_path:
            path = Path(job.temp_path)
            try:
                detection_cache.discard_temporary(str(path))
            except (OSError, ValueError):
                LOGGER.warning("Could not remove temporary capture data: %s", path)
        return {"CANCELLED"} if cancelled else {"FINISHED"}

    def cancel(self, context):
        if self._token and COORDINATOR.owns(self._token):
            COORDINATOR.request_cancel(self._token)
            scene = self._origin_scene()
            if scene is not None:
                _set_status(scene.mtp_settings, "Cancelling", "PROGRESS")
            if self._job and self._job.transaction:
                self._job.transaction.rollback()
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class MTP_OT_cancel_processing(Operator):
    bl_idname = "mtp.cancel_processing"
    bl_label = "Cancel Processing"
    bl_description = (
        "Cancel inference or sequence processing before keyframes are written"
    )

    def execute(self, context):
        job = _capture_job()
        if job and COORDINATOR.request_cancel(job.token):
            scene = bpy.data.scenes.get(job.request.scene_name)
            if scene is not None:
                _set_status(scene.mtp_settings, "Cancelling", "PROGRESS")
            return {"FINISHED"}
        return {"CANCELLED"}


class MTP_OT_select_media(Operator):
    bl_idname = "mtp.select_media"
    bl_label = "Select Media"
    bl_description = "Select or relocate the source image or video"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(
        default=(
            "*.bmp;*.exr;*.gif;*.hdr;*.jpeg;*.jpg;*.png;*.tga;"
            "*.tif;*.tiff;*.webp;*.avi;*.mkv;*.mov;*.mp4;*.m4v;*.webm"
        ),
        options={"HIDDEN"},
    )

    def invoke(self, context, _event):
        settings = context.scene.mtp_settings
        if settings.media_path:
            self.filepath = bpy.path.abspath(settings.media_path)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        settings = context.scene.mtp_settings
        selected = detection_cache.existing_media_path(self.filepath)
        if not selected:
            message = "Choose an existing image or video"
            _set_status(settings, message, "ERROR")
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        try:
            if not detection_cache.record_media_selection(settings, selected):
                raise ValueError("Choose an existing image or video")
        except (OSError, RuntimeError, ValueError) as exc:
            _set_status(settings, str(exc), "ERROR")
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _set_status(settings, "Source media selected", "INFO")
        return {"FINISHED"}


class MTP_OT_clear_detection_cache(Operator):
    bl_idname = "mtp.clear_detection_cache"
    bl_label = "Clear Detection Cache"
    bl_description = "Delete the retained Media to Pose detection cache"

    @classmethod
    def poll(cls, context):
        settings = getattr(context.scene, "mtp_settings", None)
        return bool(settings and not COORDINATOR.is_busy())

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        token = COORDINATOR.acquire(
            OperationKind.CACHE_MAINTENANCE,
            "clear-detection-cache",
            scene_name=context.scene.name,
            scene_pointer=context.scene.as_pointer(),
            status="Clearing detection cache",
        )
        if token is None:
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        try:
            detection_cache.clear_cache()
            cache_preview.refresh_after_cache_update()
        except (OSError, RuntimeError, ValueError) as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "Cache cleared")
        COORDINATOR.release(token)
        _set_status(
            context.scene.mtp_settings,
            "Detection cache cleared",
            "INFO",
        )
        return {"FINISHED"}


class MTP_OT_rebuild_camera_motion_track(Operator):
    bl_idname = "mtp.rebuild_camera_motion_track"
    bl_label = "Rebuild Camera Track"
    bl_description = (
        "Delete the derived background-camera track so the next video bake rebuilds it"
    )

    @classmethod
    def poll(cls, context):
        settings = getattr(context.scene, "mtp_settings", None)
        return bool(settings and not COORDINATOR.is_busy())

    def execute(self, context):
        token = COORDINATOR.acquire(
            OperationKind.CACHE_MAINTENANCE,
            "rebuild-camera-motion-track",
            scene_name=context.scene.name,
            scene_pointer=context.scene.as_pointer(),
            status="Clearing camera track",
        )
        if token is None:
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        try:
            detection_cache.clear_camera_motion_cache()
            armature = getattr(context.scene.mtp_settings, "target_armature", None)
            if armature is not None:
                item = next(
                    (
                        value
                        for value in armature.mtp_rig_config.post_processing_modules
                        if value.module_id == "CAMERA_MOTION_CORRECTION"
                    ),
                    None,
                )
                if item is not None:
                    item.camera_track_status = "Not Prepared"
                    item.camera_tracked_frames = 0
                    item.camera_rejected_frames = 0
                    item.camera_median_inliers = 0.0
                    item.camera_corrected_intervals = 0
                    item.last_warning = ""
                    item.status = "VALID"
        except (OSError, RuntimeError, ValueError) as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "Camera track cleared")
        COORDINATOR.release(token)
        _set_status(
            context.scene.mtp_settings,
            "Camera track will be rebuilt on the next video bake",
            "INFO",
        )
        return {"FINISHED"}


class MTP_OT_apply_recovery_choice(Operator):
    bl_idname = "mtp.apply_recovery_choice"
    bl_label = "Improve Result"
    bl_description = "Apply a suggested non-destructive workflow adjustment"
    bl_options = {"UNDO"}

    choice: EnumProperty(
        items=(
            ("SAM", "Try SAM 3D Body", "Use detailed pose detection"),
            ("IN_PLACE", "Create In-Place Animation", "Remove detected travel"),
        )
    )

    def execute(self, context):
        settings = context.scene.mtp_settings
        if self.choice == "SAM":
            settings.provider = "SAM3D_BODY"
            _set_status(settings, "SAM 3D Body selected · apply again to retry", "INFO")
        elif self.choice == "IN_PLACE":
            settings.root_motion_mode = "IN_PLACE"
            _set_status(
                settings,
                "In-place movement selected · create the animation again",
                "INFO",
            )
        return {"FINISHED"}


CLASSES = (
    MTP_OT_apply_capture,
    MTP_OT_cancel_processing,
    MTP_OT_select_media,
    MTP_OT_clear_detection_cache,
    MTP_OT_rebuild_camera_motion_track,
    MTP_OT_apply_recovery_choice,
)


def register() -> None:
    register_classes(CLASSES)


def unregister() -> None:
    for generation, job in tuple(_CAPTURE_JOBS.items()):
        COORDINATOR.request_cancel(
            job.token, status="Cancelling during add-on shutdown"
        )
        if (
            job.thread
            and job.thread.is_alive()
            and job.thread is not threading.current_thread()
        ):
            job.thread.join(timeout=0.5)
        if not job.thread or not job.thread.is_alive():
            if job.transaction is not None:
                job.transaction.rollback()
            COORDINATOR.finish(
                job.token,
                OperationState.CANCELLED,
                "Cancelled during add-on shutdown",
            )
            COORDINATOR.release(job.token)
            _CAPTURE_JOBS.pop(generation, None)
    unregister_classes(CLASSES)
