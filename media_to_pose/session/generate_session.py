"""One Kimodo generation: prepare on the main thread, infer on a worker,
apply in bounded steps inside a ``CaptureTransaction``.

Drivers (the modal operator and the API) call ``prepare`` (a dry run stops
there), ``start``, then ``step`` until ``state.finished``; ``cancel`` stops
at the next denoising step or apply batch and rolls Blender data back.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import bpy
import numpy as np

from ..backends.kimodo import backend as kimodo_backend
from ..backends.llm2vec import backend as llm2vec_backend
from ..core.coordinator import COORDINATOR, OperationKind, OperationState
from ..core.errors import Cancelled, InvalidRequest, MtpError, OperationBusy
from ..core.log import get_logger
from ..core.motion import retime, soma
from ..core.motion.kimodo import generate as kimodo_generate
from ..core.motion.kimodo import windows as kimodo_windows
from ..core.requests import OutputMode
from ..core.scene import SceneContext
from ..retarget import generated, guides, soma_pose
from ..retarget.samples import _restore_sequence_state, _sequence_state
from ..retarget.solve.frames import target_rest_basis
from ..rig import mapping
from ..rig import planning as rig_planning
from .generate_results import GeneratePlan, GenerateResult
from .transaction import CaptureTransaction

LOGGER = get_logger(__name__)
WARP_LARGE_DEG = 30.0


class GenerateState(StrEnum):
    PREPARED = "PREPARED"
    GENERATING = "GENERATING"
    APPLYING = "APPLYING"
    SUCCEEDED = "SUCCEEDED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

    @property
    def finished(self) -> bool:
        return self in {
            GenerateState.SUCCEEDED,
            GenerateState.CANCELLED,
            GenerateState.FAILED,
        }


@dataclass(frozen=True)
class PromptSegment:
    """One prompt from scene frame ``first`` up to the next segment."""

    first: float
    prompt: str


@dataclass(frozen=True)
class GenerateRequest:
    """Everything a generation needs, resolved at the API or operator."""

    armature: str
    first: float
    last: float
    prompt: str = ""
    segments: tuple[PromptSegment, ...] = ()
    use_keys: bool = True
    ignored_frames: tuple[float, ...] = ()
    waypoints: tuple[tuple[float, ...], ...] = ()
    path: str = ""
    face_travel: bool = True
    seed: int = 0
    steps: int = 50
    text_weight: float = 2.0
    guide_weight: float = 2.0
    blend_frames: float = generated.DEFAULT_BLEND_FRAMES
    output_mode: OutputMode = OutputMode.NEW_ACTION
    device: str = "AUTO"

    def __post_init__(self) -> None:
        if self.last <= self.first:
            raise InvalidRequest(
                f"Frame range {self.first:g}..{self.last:g} is empty",
                parameter="frame_range",
            )
        if not 1 <= int(self.steps) <= 1000:
            raise InvalidRequest("steps must be 1 to 1000", parameter="steps")

    def recipe(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_mode"] = str(self.output_mode)
        return payload


@dataclass(frozen=True)
class GenerateUpdate:
    state: GenerateState
    status: str
    progress: float


@dataclass
class _Prepared:
    binding: soma_pose.SomaRigBinding
    scene_guides: list[guides.SceneGuide]
    model_guides: list
    segments: tuple[kimodo_windows.Segment, ...]
    windows: list[kimodo_windows.Window]
    frame_count: int
    start_xz: tuple[float, float]
    start_heading: float
    warnings: list[dict[str, str]] = field(default_factory=list)


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


class GenerateSession:
    def __init__(self, request: GenerateRequest, armature, scene_context: SceneContext):
        self.request = request
        self.armature = armature
        self.scene_context = scene_context
        self.state = GenerateState.PREPARED
        self.status = "Prepared"
        self.progress = 0.0
        self.prepared: _Prepared | None = None
        self.result: GenerateResult | None = None
        self.error: BaseException | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._motion: kimodo_generate.GeneratedMotion | None = None
        self._worker_error: BaseException | None = None
        self._worker_status = ""
        self._worker_progress = 0.0
        self._token = None
        self._apply: Iterator | None = None
        self._transaction: CaptureTransaction | None = None
        self._seconds: dict[str, float] = {}
        self._clock = time.monotonic()
        self._variant = ""
        self._device = ""

    # Prepare ------------------------------------------------------------

    def _segments(
        self, fps: float, frame_count: int
    ) -> tuple[kimodo_windows.Segment, ...]:
        request = self.request
        items = sorted(request.segments, key=lambda item: item.first) or [
            PromptSegment(request.first, request.prompt)
        ]
        starts = [
            min(max(retime.model_index(item.first, request.first, fps), 0), frame_count)
            for item in items
        ]
        starts[0] = 0
        result = []
        for index, item in enumerate(items):
            stop = starts[index + 1] if index + 1 < len(items) else frame_count
            if stop > starts[index]:
                result.append(kimodo_windows.Segment(starts[index], stop, item.prompt))
        return tuple(result)

    def prepare(self) -> GeneratePlan:
        """Bind the rig, read guides and plan windows. No model is loaded."""
        request, context = self.request, self.scene_context
        fps = context.fps
        entries = mapping.mapping_by_id(self.armature.mtp_rig_config)
        rig_plan = rig_planning.compile_rig_solve_plan(
            self.armature, entries, lambda: target_rest_basis(self.armature, entries)
        )
        binding = soma_pose.bind(self.armature, rig_plan)
        frame_count = retime.model_frame_count(request.first, request.last, fps)
        scene_guides: list[guides.SceneGuide] = []
        if request.use_keys:
            scene_guides += guides.read_key_guides(
                binding,
                request.first,
                request.last,
                scene_context=context,
                ignored=request.ignored_frames,
            )
        if request.waypoints:
            scene_guides += guides.waypoint_guides(
                binding, request.waypoints, face_travel=request.face_travel
            )
        if request.path:
            curve = bpy.data.objects.get(request.path)
            if curve is None:
                raise InvalidRequest(
                    f"No object named {request.path!r}", parameter="path"
                )
            scene_guides += guides.path_guides(
                binding,
                curve,
                request.first,
                request.last,
                scene_context=context,
                face_travel=request.face_travel,
            )
        model_guides, collisions = guides.model_guide_list(
            scene_guides, request.first, fps, frame_count
        )
        warnings = [_warning("GUIDE_COLLISION", text) for text in collisions]
        segments = self._segments(fps, frame_count)
        plan_windows = kimodo_windows.plan_windows(frame_count, segments)
        if len(plan_windows) > len(segments):
            warnings.append(
                _warning(
                    "WINDOWED",
                    f"{frame_count} frames need {len(plan_windows)} model windows of "
                    "at most 10 s; they are joined with 5-frame transitions",
                )
            )
        state = _sequence_state(self.armature, scene_context=context)
        try:
            context.set_frame(request.first)
            _globals, positions = soma_pose.current_soma_pose(binding)
        finally:
            _restore_sequence_state(self.armature, state, scene_context=context)
        self.prepared = _Prepared(
            binding,
            scene_guides,
            model_guides,
            segments,
            plan_windows,
            frame_count,
            (float(positions[soma.ROOT, 0]), float(positions[soma.ROOT, 2])),
            float(soma.heading_angle(positions)),
            warnings,
        )
        return GeneratePlan(
            armature=self.armature.name,
            frame_range=(request.first, request.last),
            model_frames=frame_count,
            windows=[asdict(window) for window in plan_windows],
            segments=[asdict(segment) for segment in segments],
            guides=[guide.to_dict() for guide in scene_guides],
            needs_text_encoder=any(segment.prompt.strip() for segment in segments),
            warnings=warnings,
            recipe=request.recipe(),
        )

    # Worker -------------------------------------------------------------

    def _infer(self) -> None:
        prepared = self.prepared
        assert prepared is not None
        try:
            embeddings = {}
            for prompt in kimodo_generate.window_prompts(prepared.segments):
                self._worker_status = "Encoding the prompt"
                encoding = llm2vec_backend.encode_prompt(prompt)
                if encoding is not None:
                    embeddings[prompt] = encoding.vector
                    if encoding.truncated:
                        prepared.warnings.append(
                            _warning(
                                "PROMPT_TRUNCATED", f"{prompt!r} was cut to 58 tokens"
                            )
                        )
            self._mark("encode")
            if self._cancel.is_set():
                raise Cancelled("Generation cancelled")
            self._worker_status = "Loading Kimodo"
            session = kimodo_backend.ensure_model_loaded(self.request.device)
            snapshot = kimodo_backend.backend_snapshot("AUTO", self.request.device)
            self._variant = str(snapshot.variant)
            self._device = str(snapshot.actual_device)
            self._mark("load")
            request = kimodo_generate.GenerationRequest(
                frame_count=prepared.frame_count,
                stats=session.stats,
                guides=tuple(prepared.model_guides),
                segments=prepared.segments,
                embeddings=embeddings,
                start_xz=prepared.start_xz,
                start_heading=prepared.start_heading,
                seed=int(self.request.seed),
                steps=int(self.request.steps),
                text_weight=float(self.request.text_weight),
                guide_weight=float(self.request.guide_weight),
            )
            iterator = kimodo_generate.iter_generate(
                session, request, self._cancel.is_set
            )
            while True:
                try:
                    progress = next(iterator)
                except StopIteration as stop:
                    self._motion = stop.value
                    break
                self._worker_progress = progress.fraction
                self._worker_status = (
                    f"Generating · window {progress.window + 1}/{progress.windows} "
                    f"· step {progress.step}/{progress.steps}"
                )
            self._mark("generate")
        except Cancelled as exc:
            self._worker_error = exc
        except MtpError as exc:
            self._worker_error = exc
        except Exception as exc:  # Thread entry point.
            LOGGER.exception("Generation failed")
            self._worker_error = exc

    def _mark(self, phase: str) -> None:
        now = time.monotonic()
        self._seconds[phase] = round(now - self._clock, 3)
        self._clock = now

    # Driving ------------------------------------------------------------

    def start(self) -> GenerateUpdate:
        if self.prepared is None:
            self.prepare()
        scene = self.scene_context.scene
        self._token = COORDINATOR.acquire(
            OperationKind.GENERATE,
            "generate",
            scene_name=scene.name,
            scene_pointer=scene.as_pointer(),
            target_name=self.armature.name,
            target_pointer=self.armature.as_pointer(),
            status="Generating motion",
        )
        if self._token is None:
            raise OperationBusy("Media to Pose is already busy")
        self._clock = time.monotonic()
        self._thread = threading.Thread(
            target=self._infer, name="MTP-Generate", daemon=True
        )
        COORDINATOR.attach_worker(self._token, thread=self._thread)
        self._thread.start()
        self.state = GenerateState.GENERATING
        return self._update("Generating motion", 0.0)

    def cancel(self, status: str = "Generation cancelled") -> None:
        self._cancel.set()
        self.status = status

    def _update(self, status: str, progress: float) -> GenerateUpdate:
        self.status, self.progress = status, progress
        if self._token is not None:
            COORDINATOR.update(self._token, status=status, progress=progress)
        return GenerateUpdate(self.state, status, progress)

    def step(self, budget_seconds: float = 0.25) -> GenerateUpdate:
        if self.state.finished:
            return GenerateUpdate(self.state, self.status, self.progress)
        try:
            if self.state is GenerateState.GENERATING:
                assert self._thread is not None
                if self._thread.is_alive():
                    return self._update(
                        self._worker_status or "Generating motion",
                        0.8 * self._worker_progress,
                    )
                if self._worker_error is not None:
                    raise self._worker_error
                self._begin_apply()
            deadline = time.monotonic() + budget_seconds
            assert self._apply is not None
            while time.monotonic() < deadline:
                if self._cancel.is_set():
                    raise Cancelled("Generation cancelled")
                try:
                    progress = next(self._apply)
                except StopIteration:
                    self._finish_success()
                    return GenerateUpdate(self.state, self.status, 1.0)
                if progress.total:
                    fraction = 0.8 + 0.2 * progress.current / progress.total
                    self._update(
                        f"{progress.phase} {progress.current}/{progress.total}",
                        fraction,
                    )
            return GenerateUpdate(self.state, self.status, self.progress)
        except Cancelled as exc:
            self._fail(GenerateState.CANCELLED, str(exc) or "Generation cancelled", exc)
        except MtpError as exc:
            self._fail(GenerateState.FAILED, str(exc), exc)
        except Exception as exc:  # The driver boundary: roll back, report.
            LOGGER.exception("Applying the generated motion failed")
            self._fail(GenerateState.FAILED, f"Generation failed: {exc}", exc)
        return GenerateUpdate(self.state, self.status, self.progress)

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    # Apply --------------------------------------------------------------

    def _begin_apply(self) -> None:
        self.state = GenerateState.APPLYING
        self._transaction = CaptureTransaction(
            self.scene_context,
            self.armature,
            animation=True,
            output_mode=self.request.output_mode,
        )
        self._apply = self._iter_apply()

    def _iter_apply(self):
        prepared, motion, request = self.prepared, self._motion, self.request
        assert prepared is not None and motion is not None
        scene_frames = [
            float(frame)
            for frame in range(math.ceil(request.first), math.floor(request.last) + 1)
        ]
        frames = retime.resample(
            motion.frames, scene_frames, request.first, self.scene_context.fps
        )
        sink: list[str] = []
        samples = yield from generated.iter_bake(
            prepared.binding,
            frames,
            scene_frames,
            sink,
            scene_context=self.scene_context,
        )
        self._mark("bake")
        self._warps = generated.warp_to_guides(
            samples, prepared.scene_guides, request.blend_frames
        )
        self._keys, _shapes = yield from generated.iter_write(
            prepared.binding, samples, scene_context=self.scene_context
        )
        self._mark("keys")
        for message in sink:
            prepared.warnings.append(_warning("GENERATE_WARNING", message))

    def _guide_reports(self) -> list[dict[str, Any]]:
        prepared, motion = self.prepared, self._motion
        assert prepared is not None and motion is not None
        warps = {report.frame: report for report in getattr(self, "_warps", [])}
        fps, first = self.scene_context.fps, self.request.first
        scale = prepared.binding.scale
        reports = []
        for guide in prepared.scene_guides:
            index = min(
                max(retime.model_index(guide.frame, first, fps), 0),
                prepared.frame_count - 1,
            )
            if guide.positions is not None:
                error = np.linalg.norm(
                    motion.frames.positions[index] - guide.positions, axis=-1
                ).mean()
            else:
                error = np.linalg.norm(
                    motion.frames.smooth_root[index, [0, 2]] - guide.root_xz
                )
            payload = guide.to_dict()
            payload["guide_error_m"] = round(float(error) * scale, 4)
            warp = warps.get(guide.frame)
            payload["guide_warp_deg"] = warp.rotation_deg if warp else 0.0
            payload["guide_warp_m"] = warp.location_m if warp else 0.0
            if warp and warp.rotation_deg > WARP_LARGE_DEG:
                prepared.warnings.append(
                    _warning(
                        "GUIDE_WARP_LARGE",
                        f"Frame {guide.frame:g} needed {warp.rotation_deg:.0f}° of "
                        "correction; the model ignored this guide. Try more steps, "
                        "a higher guide_weight or a prompt that agrees with the pose",
                    )
                )
            reports.append(payload)
        return reports

    def _finish_success(self) -> None:
        prepared, motion, request = self.prepared, self._motion, self.request
        assert prepared is not None and motion is not None and self._transaction
        reports = self._guide_reports()
        self._transaction.commit()
        animation = self.armature.animation_data
        action = animation.action if animation else None
        slot = getattr(animation, "action_slot", None) if animation else None
        if self._transaction.nla_track_name:
            track = animation.nla_tracks[-1] if animation else None
            strip = track.strips[-1] if track and track.strips else None
            action = strip.action if strip else action
        self.state = GenerateState.SUCCEEDED
        self.status = (
            f"Generated {prepared.frame_count} model frames onto {self.armature.name}, "
            f"frames {request.first:g}-{request.last:g}"
        )
        self.result = GenerateResult(
            status="SUCCEEDED",
            message=self.status,
            armature=self.armature.name,
            action=action.name if action else "",
            slot=getattr(slot, "identifier", "") if slot else "",
            output_mode=str(request.output_mode),
            nla_track=self._transaction.nla_track_name,
            timeline_frame_range=(request.first, request.last),
            model_frames=prepared.frame_count,
            windows=[asdict(window) for window in motion.windows],
            segments=[asdict(segment) for segment in prepared.segments],
            seed=int(motion.seed),
            steps=int(request.steps),
            variant=self._variant,
            device=self._device,
            guides=reports,
            keys_written=int(getattr(self, "_keys", 0)),
            seconds=dict(self._seconds),
            warnings=list(prepared.warnings),
            recipe=request.recipe(),
        )
        self._release(OperationState.SUCCEEDED, self.status)

    def _fail(self, state: GenerateState, message: str, error: BaseException) -> None:
        self._cancel.set()
        if self._apply is not None:
            self._apply.close()
        if self._transaction is not None:
            self._transaction.rollback()
        self.state, self.status, self.error = state, message, error
        self._release(
            OperationState.CANCELLED
            if state is GenerateState.CANCELLED
            else OperationState.FAILED,
            message,
        )

    def _release(self, state: OperationState, message: str) -> None:
        if self._token is None:
            return
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        COORDINATOR.finish(self._token, state, message)
        COORDINATOR.release(self._token)
        self._token = None


__all__ = [
    "GenerateRequest",
    "GenerateSession",
    "GenerateState",
    "GenerateUpdate",
    "PromptSegment",
]
