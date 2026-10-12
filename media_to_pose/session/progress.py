"""Workload-weighted progress for one capture, from detection to keyframes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.requests import CaptureMode, MediaKind, RootMotionMode


@dataclass(frozen=True)
class CaptureProgressPlan:
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


def enabled_sequence_modules(request) -> set[str]:
    config = getattr(request, "postprocess_config", None)
    return {
        str(module.module_id)
        for module in getattr(config, "modules", ())
        if bool(module.enabled)
    }


def configured_sequence_module(request, module_id: str):
    config = getattr(request, "postprocess_config", None)
    return next(
        (
            module
            for module in getattr(config, "modules", ())
            if str(module.module_id) == module_id and bool(module.enabled)
        ),
        None,
    )


def rohm_may_run(request) -> bool:
    """Return whether this request enables any RoHM-produced output."""
    module = configured_sequence_module(request, "ROHM_CORRECTION")
    if module is None:
        return False
    pose_residual = bool(getattr(module, "learned_pose_residual", False))
    root_cleanup = bool(getattr(module, "root_trajectory_cleanup", True))
    return pose_residual or (
        root_cleanup and request.root_motion_mode == RootMotionMode.AUTOMATIC
    )


def capture_progress_plan(request, *, cached: bool) -> CaptureProgressPlan:
    """Allocate progress by expected cost and real pipeline order."""
    weights: list[tuple[str, float]] = []
    if not cached:
        weights.append(("calibration", 0.5))
        # Measured wall-time ratio for the two bundled pose backends.
        detection_weight = 8.0 if request.provider_id == "SAM3D_BODY" else 1.0
        weights.append(("detection", detection_weight))
    if request.mode == CaptureMode.POSE:
        weights.append(("pose_application", 0.75))
    elif request.mode == CaptureMode.KEYFRAMES:
        modules = enabled_sequence_modules(request)
        weights.append(("loading", 0.15))
        if (
            request.media_kind == MediaKind.VIDEO
            and request.root_motion_mode == RootMotionMode.AUTOMATIC
            and "CAMERA_MOTION_CORRECTION" in modules
        ):
            weights.append(("camera_preflight", 0.75))
        weights.append(("solve", 1.0))
        if "CAMERA_MOTION_CORRECTION" in modules:
            weights.append(("camera_apply", 0.25))
        if "DETECTION_FILTER" in modules:
            weights.append(("detection_filter", 0.25))
        if rohm_may_run(request):
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
    return CaptureProgressPlan(ranges)


def sequence_progress_group(phase: str) -> str:
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


SEQUENCE_STATUS = {
    "loading": "Preparing detected frames",
    "camera_preflight": "Tracking camera motion",
    "solve": "Solving animation",
    "camera_apply": "Applying camera correction",
    "detection_filter": "Filtering unreliable detections",
    "rohm": "Applying Experimental RoHM cleanup",
    "temporal": "Smoothing animation",
    "commit": "Writing keyframes",
}


def sequence_progress_status(progress) -> str:
    return SEQUENCE_STATUS[sequence_progress_group(getattr(progress, "phase", ""))]


_DEFAULT_SEQUENCE_RANGES = {
    "loading": (0.0, 5.0),
    "camera_preflight": (5.0, 15.0),
    "solve": (15.0, 50.0),
    "camera_apply": (50.0, 55.0),
    "detection_filter": (55.0, 60.0),
    "rohm": (60.0, 85.0),
    "temporal": (85.0, 92.0),
    "commit": (92.0, 100.0),
}

_TEMPORAL_SUBRANGES = (
    ("detecting foot contacts", (0.45, 0.55)),
    ("stabilizing root motion", (0.55, 0.70)),
    ("applying stabilized root", (0.70, 0.80)),
    ("repairing planted feet", (0.80, 1.0)),
)


def sequence_progress_percent(
    progress,
    previous: float = 0.0,
    plan: CaptureProgressPlan | None = None,
) -> float:
    """Map local phase counters onto one monotonic video-bake progress bar."""
    total = max(int(getattr(progress, "total", 0) or 0), 1)
    current = min(max(int(getattr(progress, "current", 0) or 0), 0), total)
    ratio = current / total
    group = sequence_progress_group(getattr(progress, "phase", ""))
    start, end = (
        _DEFAULT_SEQUENCE_RANGES[group] if plan is None else plan.range_for(group)
    )
    if group == "temporal":
        folded = str(getattr(progress, "phase", "")).casefold()
        subrange = next(
            (span for prefix, span in _TEMPORAL_SUBRANGES if folded.startswith(prefix)),
            (0.0, 0.45),
        )
        parent_start, parent_end = start, end
        start = parent_start + (parent_end - parent_start) * subrange[0]
        end = parent_start + (parent_end - parent_start) * subrange[1]
    mapped = start + (end - start) * ratio
    return min(max(float(previous), mapped), 100.0)


__all__ = [
    "SEQUENCE_STATUS",
    "CaptureProgressPlan",
    "capture_progress_plan",
    "configured_sequence_module",
    "enabled_sequence_modules",
    "rohm_may_run",
    "sequence_progress_group",
    "sequence_progress_percent",
    "sequence_progress_status",
]
