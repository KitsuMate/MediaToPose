"""Default input and output ranges of face mappings, in the rig's units.

Shape keys go from 0 to 1. Eye gaze turns the eyes ``GAZE_DEGREES`` at
full. Face bone controls that move travel a share of the distance between
the eyes (``FACE_CONTROL_TRAVEL``, fitted on an Auto-Rig Pro face), so
the defaults scale with the face; rotating controls turn
``FACE_ROTATION_DEGREES``. See ``core/face_ranges.py`` for how the ranges
are applied.
"""

from __future__ import annotations

from ..core.face_ranges import legacy_ranges
from .face_schemes import SIGNED_FACE_CONTROLS
from .face_space import face_frame

# Travel of a location-driven face control at full strength, as a fraction
# of the distance between the eyes: (full, negative full). Fitted on the
# Auto-Rig Pro FaceFX fixture so each bone control moves the face mesh as
# close as possible to the matching ARKit key (rig/face_fit.py, least
# squares over every vertex). The lips keep a small hand-set value: ARP's
# lip centre moves differently from ARKit's lip-side keys, so no fit holds.
FACE_CONTROL_TRAVEL = {
    "FACE_JAW_OPEN": (0.93, -0.93),
    "FACE_JAW_SIDE": (0.082, -0.082),
    "FACE_LID_L": (0.34, -0.107),
    "FACE_LID_R": (0.34, -0.107),
    "FACE_BROW_INNER": (0.066, -0.066),
    "FACE_BROW_OUTER_L": (0.039, -0.113),
    "FACE_BROW_OUTER_R": (0.039, -0.113),
    "FACE_CHEEK_L": (0.098, -0.098),
    "FACE_CHEEK_R": (0.098, -0.098),
    "FACE_MOUTH_CORNER_L": (0.116, -0.058),
    "FACE_MOUTH_CORNER_R": (0.116, -0.058),
    "FACE_LIP_TOP": (0.05, -0.05),
    "FACE_LIP_BOTTOM": (0.05, -0.05),
}
# Detected value treated as full, where it is not 1. Bone lids close fully
# from 0.75: in ARKit mode the squint and look-down keys add to the blink,
# so the keys close the eye further than the blink alone. Fitted on the 12
# usable face stills against the FaceFX ARKit lids (upper-lid edge height,
# least squares over both eyes): the error drops from 2.1 mm to 0.9 mm.
FACE_CONTROL_INPUT_FULL = {"FACE_LID_L": 0.75, "FACE_LID_R": 0.75}
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


def _outputs(armature, entries, entry) -> tuple[float, float]:
    """Output at full and at negative full for ``entry``'s channel."""
    if entry.semantic_id in EYES:
        return GAZE_DEGREES, -GAZE_DEGREES
    if entry.section == "SHAPES":
        return 1.0, -1.0
    if entry.channel_mode == "LOCATION":
        default = (DEFAULT_FACE_CONTROL_TRAVEL, -DEFAULT_FACE_CONTROL_TRAVEL)
        full, negative = FACE_CONTROL_TRAVEL.get(entry.semantic_id, default)
        scale = face_frame(armature, entries).scale
        return full * scale, negative * scale
    if entry.channel_mode in {"ROTATION", "AUTO"}:
        return FACE_ROTATION_DEGREES, -FACE_ROTATION_DEGREES
    return 1.0, -1.0


def default_ranges(armature, entries, entry) -> dict[str, float]:
    full, negative = _outputs(armature, entries, entry)
    bone = entry.section == "FACE_RIG"
    return {
        "input_zero": 0.0,
        "input_full": FACE_CONTROL_INPUT_FULL.get(entry.semantic_id, 1.0)
        if bone
        else 1.0,
        "output_zero": 0.0,
        "output_full": full,
        "output_negative": negative,
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
    negative = None
    if entry.semantic_id in EYES:
        # Gaze used the gain only for its sign.
        scale = GAZE_DEGREES
    elif entry.section == "FACE_RIG":
        # 0.2.0 bone travel was not in face units; take today's defaults,
        # scaled by the old clamp on both sides.
        scale, negative = _outputs(armature, entries, entry)
    else:
        scale = 1.0
    ranges = legacy_ranges(
        gain=float(entry.get("gain", 1.0)),
        offset=float(entry.get("offset", 0.0)),
        clamp_max=float(entry.get("clamp_max", 1.0)),
        invert=bool(entry.get("invert", False)),
        scale=scale,
    )
    if negative is not None and not entry.get("invert", False):
        ranges["output_negative"] = negative * float(entry.get("clamp_max", 1.0))
    for name, value in ranges.items():
        setattr(entry, name, value)
    entry.ranges_version = RANGES_VERSION


__all__ = [
    "DEFAULT_FACE_CONTROL_TRAVEL",
    "FACE_CONTROL_INPUT_FULL",
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
