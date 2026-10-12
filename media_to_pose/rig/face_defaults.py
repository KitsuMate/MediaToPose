"""Default input and output ranges of face mappings, in the rig's units.

Shape keys go from 0 to 1. Eye gaze turns the eyes ``GAZE_DEGREES`` at
full. Face bone controls that move travel a share of the distance between
the eyes (``FACE_CONTROL_TRAVEL``, measured on an Auto-Rig Pro face), so
the defaults scale with the face; rotating controls turn
``FACE_ROTATION_DEGREES``. See ``core/face_ranges.py`` for how the ranges
are applied.
"""

from __future__ import annotations

from ..core.face_ranges import legacy_ranges
from .face_schemes import SIGNED_FACE_CONTROLS
from .face_space import face_frame

# Travel of a location-driven face control at full strength, as a fraction
# of the distance between the eyes. Measured on an Auto-Rig Pro face (lids
# close fully, brows and mouth corners move clearly without breaking).
FACE_CONTROL_TRAVEL = {
    "FACE_JAW_OPEN": 0.55,
    "FACE_JAW_SIDE": 0.20,
    "FACE_LID_L": 0.34,
    "FACE_LID_R": 0.34,
    "FACE_BROW_INNER": 0.17,
    "FACE_BROW_OUTER_L": 0.17,
    "FACE_BROW_OUTER_R": 0.17,
    "FACE_CHEEK_L": 0.10,
    "FACE_CHEEK_R": 0.10,
    "FACE_MOUTH_CORNER_L": 0.17,
    "FACE_MOUTH_CORNER_R": 0.17,
    "FACE_LIP_TOP": 0.05,
    "FACE_LIP_BOTTOM": 0.05,
}
DEFAULT_FACE_CONTROL_TRAVEL = 0.12
FACE_ROTATION_DEGREES = 25.0
GAZE_DEGREES = 18.0
EYES = ("EYE_L", "EYE_R")
RANGE_FIELDS = (
    "input_zero",
    "input_full",
    "output_zero",
    "output_full",
    "output_negative",
)
# Bumped when the meaning of the stored ranges changes; 0 marks mappings
# saved by 0.2.0, which still carry gain / offset / clamp / invert.
RANGES_VERSION = 1


def is_signed(semantic_id: str) -> bool:
    """Whether the control has a negative side (brows down, frown, ...)."""
    return semantic_id in SIGNED_FACE_CONTROLS or semantic_id in EYES


def has_ranges(entry) -> bool:
    return entry.section in {"SHAPES", "FACE_RIG"} or entry.semantic_id in EYES


def _full_value(armature, entries, entry) -> float:
    """The output at full strength for ``entry``'s channel."""
    if entry.semantic_id in EYES:
        return GAZE_DEGREES
    if entry.section == "SHAPES":
        return 1.0
    if entry.channel_mode == "LOCATION":
        travel = FACE_CONTROL_TRAVEL.get(entry.semantic_id, DEFAULT_FACE_CONTROL_TRAVEL)
        return travel * face_frame(armature, entries).scale
    if entry.channel_mode == "ROTATION":
        return FACE_ROTATION_DEGREES
    return 1.0


def default_ranges(armature, entries, entry) -> dict[str, float]:
    full = _full_value(armature, entries, entry)
    return {
        "input_zero": 0.0,
        "input_full": 1.0,
        "output_zero": 0.0,
        "output_full": full,
        "output_negative": -full,
    }


def apply_default_ranges(armature, entries, entry) -> None:
    for name, value in default_ranges(armature, entries, entry).items():
        setattr(entry, name, value)
    entry.ranges_version = RANGES_VERSION


def migrate_ranges(armature, entries, entry) -> None:
    """Convert a 0.2.0 mapping's gain / offset / clamp / invert to ranges.

    Those properties are gone from the mapping but stay stored in the .blend
    as ID properties, which is where they are read from.
    """
    if entry.semantic_id in EYES:
        # Gaze used the gain only for its sign.
        scale = GAZE_DEGREES
    elif entry.section == "FACE_RIG":
        scale = _full_value(armature, entries, entry)
    else:
        scale = 1.0
    ranges = legacy_ranges(
        gain=float(entry.get("gain", 1.0)),
        offset=float(entry.get("offset", 0.0)),
        clamp_max=float(entry.get("clamp_max", 1.0)),
        invert=bool(entry.get("invert", False)),
        scale=scale,
    )
    for name, value in ranges.items():
        setattr(entry, name, value)
    entry.ranges_version = RANGES_VERSION


__all__ = [
    "DEFAULT_FACE_CONTROL_TRAVEL",
    "FACE_CONTROL_TRAVEL",
    "FACE_ROTATION_DEGREES",
    "GAZE_DEGREES",
    "RANGES_VERSION",
    "RANGE_FIELDS",
    "apply_default_ranges",
    "default_ranges",
    "has_ranges",
    "is_signed",
    "migrate_ranges",
]
