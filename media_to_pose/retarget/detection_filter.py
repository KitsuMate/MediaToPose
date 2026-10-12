"""Reject unreliable detections and repair the resulting gaps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import suppress

from .. import (
    motion_filter,
)
from ..core.errors import InvalidRequest
from .rohm import _rohm_requested_work
from .root_motion import _root_transform, _set_root_transform
from .samples import (
    SequenceApplyProgress,
    _append_bake_warning,
    _BakedFrameSample,
    _sequence_timestamps,
)


def _repair_root_gaps_without_ai(
    samples: Sequence[_BakedFrameSample],
    timestamps: Sequence[float],
) -> int:
    missing = {
        index
        for index, sample in enumerate(samples)
        if float(sample.rohm_root_confidence) <= 0.0
    }
    available = [
        index
        for index, sample in enumerate(samples)
        if index not in missing and _root_transform(sample) is not None
    ]
    if not available:
        return 0
    repaired = 0
    for index in sorted(missing):
        if _root_transform(samples[index]) is None:
            continue
        left = next((value for value in reversed(available) if value < index), None)
        right = next((value for value in available if value > index), None)
        if left is None and right is None:
            continue
        if left is None or right is None:
            source = _root_transform(samples[right if left is None else left])
            location, rotation = source[0], source[1]
        else:
            first = _root_transform(samples[left])
            last = _root_transform(samples[right])
            span = float(timestamps[right]) - float(timestamps[left])
            factor = (
                (float(timestamps[index]) - float(timestamps[left])) / span
                if span > 1.0e-8
                else (index - left) / max(right - left, 1)
            )
            factor = min(max(factor, 0.0), 1.0)
            location = first[0].lerp(last[0], factor)
            rotation = first[1].slerp(last[1], factor).normalized()
        repaired += int(_set_root_transform(samples[index], location, rotation))
    return repaired


def _iter_detection_filter(
    samples: list[_BakedFrameSample],
    settings,
    *,
    rohm_settings=None,
    effective_mode: str = "OFF",
    ui_settings=None,
) -> Iterator[SequenceApplyProgress]:
    """Reject unreliable evidence before RoHM or deterministic local repair."""
    timestamps = _sequence_timestamps(samples)
    yield SequenceApplyProgress("Filtering unreliable detections", 0, 2)
    stats = motion_filter.filter_motion_observations(
        samples,
        timestamps,
        reject_unreliable_limbs=bool(settings.reject_unreliable_limbs),
        reject_unusable_frames=bool(settings.reject_unusable_frames),
        maximum_rejected_fraction=float(settings.maximum_motion_rejected) / 100.0,
    )
    try:
        manual_frames = motion_filter.parse_source_frame_ranges(
            str(getattr(settings, "manual_bad_frames", "")), len(samples)
        )
    except ValueError as exc:
        raise InvalidRequest(f"Additional Bad Source Frames: {exc}") from exc
    manual_rejected = motion_filter.apply_manual_motion_rejections(
        samples,
        manual_frames,
        str(getattr(settings, "manual_bad_scope", "BODY")),
    )
    repaired = motion_filter.repair_body_gaps(samples, timestamps)
    _pose_ai, root_ai = _rohm_requested_work(samples, rohm_settings, effective_mode)
    if not root_ai:
        repaired += _repair_root_gaps_without_ai(samples, timestamps)
    automatic_frames = {frame_index for frame_index, _semantic in stats.rejected}
    automatic_summary = motion_filter.format_source_frame_ranges(automatic_frames)
    manual_summary = motion_filter.format_source_frame_ranges(manual_frames)
    frame_summary = f"Automatic: {automatic_summary} · Manual: {manual_summary}"
    if len(frame_summary) > 240:
        frame_summary = frame_summary[:237].rstrip() + "..."
    if ui_settings is not None:
        with suppress(AttributeError, ReferenceError, RuntimeError):
            ui_settings.filter_rejected_observations = stats.rejected_observations
            ui_settings.filter_rejected_limbs = stats.rejected_limbs
            ui_settings.filter_rejected_frames = stats.rejected_frames
            ui_settings.filter_native_missing = stats.native_missing_observations
            ui_settings.filter_total_observations = stats.total_observations
            ui_settings.filter_manual_frames = len(manual_frames)
            ui_settings.filter_frame_summary = frame_summary
    if stats.rejected_observations:
        _append_bake_warning(
            samples,
            "Detection Filter rejected "
            f"{stats.rejected_observations} available joint sample(s) "
            f"({stats.rejected_frames} frame(s), {stats.rejected_limbs} limb event(s))",
        )
    if manual_frames:
        _append_bake_warning(
            samples,
            "Manual cleanup mask selected "
            f"{len(manual_frames)} source frame(s) and rejected "
            f"{len(manual_rejected)} available joint sample(s)",
        )
    elif str(getattr(settings, "manual_bad_frames", "")).strip():
        _append_bake_warning(
            samples,
            "Manual cleanup mask selected no source frames inside this clip",
        )
    if repaired:
        _append_bake_warning(
            samples,
            f"Detection Filter locally repaired {repaired} missing motion sample(s); "
            "boundary gaps use the nearest reliable pose",
        )
    if stats.observed_fraction < 0.5:
        _append_bake_warning(
            samples,
            "Mostly reconstructed · limited source evidence",
        )
    yield SequenceApplyProgress("Filtering unreliable detections", 2, 2)
