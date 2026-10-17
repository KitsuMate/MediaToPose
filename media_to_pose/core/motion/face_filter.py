"""Clean a video's face coefficients before they reach the rig.

MediaPipe reports ARKit-style coefficients (0 to 1) per frame. Raw, they
twitch, keep each face's resting bias (arched brows read as raised), drop
to nothing when the face is briefly lost, and lose their blinks to ordinary
smoothing. ``filter_face_track`` fixes those in order:

1. Hold, then fade out, the last face when frames have no face.
2. Neutral calibration: subtract each coefficient's resting level, the
   value it stays above for most of the clip.
3. Deadzone: values below a small threshold become zero.
4. Smoothing per expression group, then blinks put back at their detected
   depth and timing.

Pure Python on dicts so it runs without Blender.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

BLINKS = ("eyeBlinkLeft", "eyeBlinkRight")
# Expression groups smooth with different strengths: eyes and lids move
# fastest, the jaw slowest.
GROUP_STRENGTH = {"eye": 0.5, "brow": 1.0, "mouth": 0.8, "jaw": 1.0, "other": 0.8}


def coefficient_group(name: str) -> str:
    if name.startswith(("eye", "cheekSquint")):
        return "eye"
    if name.startswith("brow"):
        return "brow"
    if name.startswith("jaw"):
        return "jaw"
    if name.startswith(("mouth", "cheekPuff", "noseSneer", "tongue")):
        return "mouth"
    return "other"


@dataclass(frozen=True)
class FaceFilterSettings:
    smooth: bool = True
    strength: float = 0.35
    preserve_blinks: bool = True
    neutral_calibration: bool = True
    deadzone: float = 0.05
    hold_seconds: float = 0.5
    # Percentile of the clip taken as each coefficient's resting level.
    rest_percentile: float = 0.1
    blink_threshold: float = 0.5
    blink_max_seconds: float = 0.5


@dataclass
class FaceFilterStats:
    frames: int = 0
    frames_without_face: int = 0
    held_frames: int = 0
    blinks: int = 0
    calibrated: dict[str, float] = field(default_factory=dict)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _gaussian_smooth(
    values: Sequence[float], times: Sequence[float], sigma: float
) -> list[float]:
    """Centred Gaussian smoothing over time (seconds); no lag."""
    if sigma <= 1.0e-6 or len(values) < 3:
        return list(values)
    result = []
    reach = 3.0 * sigma
    for index, time in enumerate(times):
        total = weight_sum = 0.0
        low = index
        while low > 0 and time - times[low - 1] <= reach:
            low -= 1
        high = index
        while high + 1 < len(times) and times[high + 1] - time <= reach:
            high += 1
        for other in range(low, high + 1):
            weight = math.exp(-0.5 * ((times[other] - time) / sigma) ** 2)
            total += values[other] * weight
            weight_sum += weight
        result.append(total / weight_sum if weight_sum > 0.0 else values[index])
    return result


def find_blinks(
    values: Sequence[float],
    times: Sequence[float],
    threshold: float,
    max_seconds: float,
) -> list[tuple[int, int]]:
    """Inclusive index ranges where a blink rises above ``threshold`` briefly."""
    blinks = []
    index = 0
    while index < len(values):
        if values[index] < threshold:
            index += 1
            continue
        start = index
        while index + 1 < len(values) and values[index + 1] >= threshold:
            index += 1
        if times[index] - times[start] <= max_seconds:
            blinks.append((start, index))
        index += 1
    return blinks


def _hold_missing(
    track: list[dict[str, float] | None],
    times: Sequence[float],
    hold_seconds: float,
    stats: FaceFilterStats,
) -> list[dict[str, float]]:
    """Hold the last face for ``hold_seconds``, then fade it out as long."""
    result: list[dict[str, float]] = []
    last: dict[str, float] = {}
    last_time = None
    for values, time in zip(track, times, strict=True):
        if values:
            last, last_time = dict(values), time
            result.append(dict(values))
            continue
        stats.frames_without_face += 1
        if last_time is None or hold_seconds <= 0.0:
            result.append({})
            continue
        elapsed = time - last_time
        if elapsed <= hold_seconds:
            factor = 1.0
            stats.held_frames += 1
        else:
            factor = max(0.0, 1.0 - (elapsed - hold_seconds) / hold_seconds)
        result.append({name: value * factor for name, value in last.items()})
    return result


def filter_face_track(
    track: Sequence[Mapping[str, float] | None],
    times: Sequence[float],
    settings: FaceFilterSettings,
) -> tuple[list[dict[str, float]], FaceFilterStats]:
    """Return cleaned coefficients per frame and what was done.

    ``track`` holds one coefficient dict per frame, or ``None`` / ``{}``
    where no face was detected; ``times`` are the frame times in seconds.
    """
    stats = FaceFilterStats(frames=len(track))
    frames = _hold_missing(
        [dict(values) if values else None for values in track],
        times,
        settings.hold_seconds,
        stats,
    )
    names = sorted({name for values in frames for name in values})
    detected = [index for index, values in enumerate(track) if values]
    columns = {
        name: [float(values.get(name, 0.0)) for values in frames] for name in names
    }

    if settings.neutral_calibration and len(detected) >= 10:
        for name in names:
            rest = _percentile(
                [columns[name][index] for index in detected], settings.rest_percentile
            )
            if rest <= 0.02:
                continue
            stats.calibrated[name] = round(rest, 4)
            scale = 1.0 / max(1.0 - rest, 1.0e-3)
            columns[name] = [max(0.0, (v - rest) * scale) for v in columns[name]]

    if settings.deadzone > 0.0:
        dead = min(settings.deadzone, 0.9)
        for name in names:
            columns[name] = [
                0.0 if v < dead else (v - dead) / (1.0 - dead) for v in columns[name]
            ]

    blink_ranges = {}
    if settings.preserve_blinks:
        for name in BLINKS:
            if name in columns:
                blink_ranges[name] = find_blinks(
                    columns[name],
                    times,
                    settings.blink_threshold,
                    settings.blink_max_seconds,
                )
        stats.blinks = max((len(r) for r in blink_ranges.values()), default=0)

    if settings.smooth and settings.strength > 0.0:
        # Strength 1 is a 0.15 s Gaussian; the default 0.35 is 1 to 1.5
        # frames at 30 fps.
        base = 0.15 * min(max(settings.strength, 0.0), 1.0)
        for name in names:
            original = columns[name]
            sigma = base * GROUP_STRENGTH[coefficient_group(name)]
            smoothed = _gaussian_smooth(original, times, sigma)
            for start, end in blink_ranges.get(name, ()):
                # Keep the blink's own depth and timing, easing into it by a
                # frame on each side.
                for index in range(max(start - 1, 0), min(end + 2, len(smoothed))):
                    smoothed[index] = max(smoothed[index], original[index])
            columns[name] = smoothed

    # Keep explicit neutral values after a detected face has faded out.
    # An empty dict makes the solver skip the face, leaving the last nonzero
    # key held for the rest of the gap. Leading missing frames stay empty.
    first_detected = detected[0] if detected else len(frames)
    result = [
        {name: min(max(columns[name][index], 0.0), 1.0) for name in names}
        if index >= first_detected
        else {}
        for index in range(len(frames))
    ]
    return result, stats


__all__ = [
    "BLINKS",
    "FaceFilterSettings",
    "FaceFilterStats",
    "coefficient_group",
    "filter_face_track",
    "find_blinks",
]
