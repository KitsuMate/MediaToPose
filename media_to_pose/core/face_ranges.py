"""Input and output ranges of a face mapping.

Every face mapping (shape key, face bone control, custom property, eye gaze)
turns a detected value into a target value in two steps:

1. ``strength``: where the detection sits between ``input_zero`` (rest) and
   ``input_full``, clamped to 0..1, or to -1..1 for signed controls such as
   brows (up and down) or mouth corners (smile and frown).
2. ``output_value``: the target value at that strength, between
   ``output_zero`` and ``output_full`` (``output_negative`` on the negative
   side), times the mapping's influence.

Outputs are in the target's units: a shape key value, metres along the
control's face-space axis, degrees about it, a custom property's value, or
the gaze angle in degrees. The defaults (0 to 1 in, 0 to 1 out) map a
detection straight onto a shape key.

Pure Python, so it works on Blender mapping entries and plain snapshots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FaceRanges:
    input_zero: float = 0.0
    input_full: float = 1.0
    output_zero: float = 0.0
    output_full: float = 1.0
    output_negative: float = -1.0
    influence: float = 1.0


def _finite(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def strength(entry, detected: float, *, signed: bool = False) -> float:
    """The detection as a share of the input range, clamped."""
    span = _finite(entry.input_full, 1.0) - _finite(entry.input_zero)
    if abs(span) < 1.0e-9:
        return 0.0
    share = (_finite(detected) - _finite(entry.input_zero)) / span
    return min(max(share, -1.0 if signed else 0.0), 1.0)


def output_value(entry, share: float) -> float:
    """The target value at ``share`` of the range, times the influence."""
    zero = _finite(entry.output_zero)
    if share >= 0.0:
        value = zero + share * (_finite(entry.output_full, 1.0) - zero)
    else:
        value = zero + (-share) * (_finite(entry.output_negative, -1.0) - zero)
    return value * _finite(entry.influence, 1.0)


def map_value(entry, detected: float, *, signed: bool = False) -> float:
    return output_value(entry, strength(entry, detected, signed=signed))


def share_of_output(entry, value: float) -> float:
    """Inverse of ``output_value``: how much of its range a target shows.

    Used to read a rig back in the same 0..1 (or -1..1) units as
    ``strength``, whatever the target's units are.
    """
    influence = _finite(entry.influence, 1.0)
    if abs(influence) < 1.0e-9:
        return 0.0
    value = _finite(value) / influence
    zero = _finite(entry.output_zero)
    full = _finite(entry.output_full, 1.0) - zero
    negative = _finite(entry.output_negative, -1.0) - zero
    offset = value - zero
    if abs(full) > 1.0e-12 and offset * full >= 0.0:
        return offset / full
    if abs(negative) > 1.0e-12 and offset * negative > 0.0:
        return -offset / negative
    return 0.0


def legacy_ranges(
    *,
    gain: float = 1.0,
    offset: float = 0.0,
    clamp_max: float = 1.0,
    invert: bool = False,
    scale: float = 1.0,
) -> dict[str, float]:
    """Ranges that reproduce the gain / offset / clamp / invert of 0.2.0.

    The old value was ``clamp((detected - offset) * gain)`` (or ``clamp_max``
    minus it when inverted) times ``scale`` (the control's travel); the
    detection reached ``clamp_max`` at ``offset + clamp_max / gain``.
    """
    gain = _finite(gain, 1.0) or 1.0
    top = _finite(clamp_max, 1.0) or 1.0
    ranges = {
        "input_zero": _finite(offset),
        "input_full": _finite(offset) + top / gain,
        "output_zero": 0.0,
        "output_full": top * scale,
        # Signed controls had clamp_min = -clamp_max: the same slope below
        # zero.
        "output_negative": -top * scale,
    }
    if invert:
        ranges["output_zero"], ranges["output_full"] = top * scale, 0.0
    return ranges


__all__ = [
    "FaceRanges",
    "legacy_ranges",
    "map_value",
    "output_value",
    "share_of_output",
    "strength",
]
