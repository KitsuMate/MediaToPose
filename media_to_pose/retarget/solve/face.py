"""Face output: ARKit/FACS shape keys, eye gaze and facial bones."""

from __future__ import annotations

import math

import bpy
from mathutils import Quaternion, Vector

from ...rig.face_schemes import matching_shape_blocks
from ...rig.face_space import (
    face_axis,
    face_frame,
    gaze_pivot_distance,
    to_bone,
)
from ..target_pose import TargetPose
from .common import _mapped_bone


def _finite_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _expression_value(entry, raw_value: float) -> float:
    value = (_finite_float(raw_value) - entry.offset) * entry.gain
    if entry.invert:
        value = entry.clamp_max - value
    value *= entry.influence
    return min(max(value, entry.clamp_min), entry.clamp_max)


def facs_values(blendshapes: dict[str, float]) -> dict[str, float]:
    """Approximate FACS AU intensities from MediaPipe blendshape coefficients.

    MediaPipe Face Landmarker emits ARKit-style expression coefficients rather
    than FACS action units. Keep this provider-to-standard conversion separate
    from destination mapping so another detector can supply native AU values
    later without changing the rig adapters.
    """

    def get(name: str) -> float:
        return min(
            max(_finite_float(blendshapes.get(name, 0.0)), 0.0),
            1.0,
        )

    bilateral_sources = {
        "AU01": ("browInnerUp", "browInnerUp"),
        "AU02": ("browOuterUpLeft", "browOuterUpRight"),
        "AU04": ("browDownLeft", "browDownRight"),
        "AU05": ("eyeWideLeft", "eyeWideRight"),
        "AU06": ("cheekSquintLeft", "cheekSquintRight"),
        "AU07": ("eyeSquintLeft", "eyeSquintRight"),
        "AU09": ("noseSneerLeft", "noseSneerRight"),
        "AU10": ("mouthUpperUpLeft", "mouthUpperUpRight"),
        "AU12": ("mouthSmileLeft", "mouthSmileRight"),
        "AU14": ("mouthDimpleLeft", "mouthDimpleRight"),
        "AU15": ("mouthFrownLeft", "mouthFrownRight"),
        "AU16": ("mouthLowerDownLeft", "mouthLowerDownRight"),
        "AU20": ("mouthStretchLeft", "mouthStretchRight"),
        "AU23": ("mouthPressLeft", "mouthPressRight"),
        "AU45": ("eyeBlinkLeft", "eyeBlinkRight"),
    }
    values: dict[str, float] = {}
    for action_unit, (left_source, right_source) in bilateral_sources.items():
        left = get(left_source)
        right = get(right_source)
        values[action_unit] = max(left, right)
        values[f"{action_unit}_L"] = left
        values[f"{action_unit}_R"] = right

    left_press = get("mouthPressLeft")
    right_press = get("mouthPressRight")
    jaw_open = get("jawOpen")
    values.update(
        {
            "AU17": get("mouthShrugLower"),
            "AU18": get("mouthPucker"),
            "AU22": get("mouthFunnel"),
            "AU24": max(left_press, right_press),
            # AU25 lips part (also with the jaw closed), AU26 jaw drop, AU27
            # mouth stretched wide open: only the far end of jawOpen.
            "AU25": max(
                jaw_open,
                (get("mouthUpperUpLeft") + get("mouthUpperUpRight")) / 2.0,
                (get("mouthLowerDownLeft") + get("mouthLowerDownRight")) / 2.0,
            ),
            "AU26": jaw_open,
            "AU27": min(max((jaw_open - 0.6) / 0.4, 0.0), 1.0),
            "AU28": max(get("mouthRollUpper"), get("mouthRollLower")),
        }
    )
    return values


def vrm_values(blendshapes: dict[str, float]) -> dict[str, float]:
    """Approximate VRM / VRoid expression weights from ARKit coefficients.

    Visemes come from the mouth shape (open jaw, funnel, pucker, stretch);
    emotions from the matching brow, eye and mouth coefficients. Each
    expression is driven only by the coefficients that describe it, so they
    combine instead of replacing the whole face.
    """

    def get(name: str) -> float:
        return min(max(_finite_float(blendshapes.get(name, 0.0)), 0.0), 1.0)

    def both(left: str, right: str) -> float:
        return (get(left) + get(right)) / 2.0

    def clamp(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    jaw = get("jawOpen")
    funnel = get("mouthFunnel")
    pucker = get("mouthPucker")
    stretch = both("mouthStretchLeft", "mouthStretchRight")
    smile = both("mouthSmileLeft", "mouthSmileRight")
    outer_up = both("browOuterUpLeft", "browOuterUpRight")
    return {
        "blinkLeft": get("eyeBlinkLeft"),
        "blinkRight": get("eyeBlinkRight"),
        "aa": clamp(jaw - 0.5 * max(funnel, pucker)),
        "oh": clamp(min(funnel, jaw + 0.3)),
        "ou": pucker,
        "ee": clamp(stretch * (1.0 - jaw)),
        "ih": clamp(smile * 0.4 * (1.0 - jaw)),
        "mouthJoy": smile,
        "mouthSorrow": both("mouthFrownLeft", "mouthFrownRight"),
        "mouthAngry": both("mouthPressLeft", "mouthPressRight"),
        "browSurprised": max(get("browInnerUp"), outer_up),
        "browAngry": both("browDownLeft", "browDownRight"),
        "browSorrow": clamp(get("browInnerUp") - outer_up),
        "eyeSurprised": both("eyeWideLeft", "eyeWideRight"),
        "eyeJoyLeft": get("cheekSquintLeft"),
        "eyeJoyRight": get("cheekSquintRight"),
    }


def _matching_shape_blocks(armature, entry):
    yield from matching_shape_blocks(armature, entry)


def _apply_shapes(armature, entries, frame, scheme: str, target_pose: TargetPose):
    changed = []
    expression_values = (
        facs_values(frame.blendshapes)
        if scheme == "FACS"
        else vrm_values(frame.blendshapes)
        if scheme == "VRM"
        else frame.blendshapes
    )
    for name, raw_value in expression_values.items():
        entry = entries.get(f"{scheme}_{name}")
        if not entry or not entry.enabled or not entry.target_name:
            continue
        value = _expression_value(entry, raw_value)
        for obj, block in _matching_shape_blocks(armature, entry):
            value_clamped = min(max(value, block.slider_min), block.slider_max)
            target_pose.set_shape(obj, block, value_clamped)
            changed.append((obj, block))
    return changed


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
# Eye rotation for a gaze coefficient of 1.0.
GAZE_ANGLE = math.radians(18.0)


def _axis(axis_name: str) -> Vector:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    return {
        "X": Vector((sign, 0.0, 0.0)),
        "Y": Vector((0.0, sign, 0.0)),
        "Z": Vector((0.0, 0.0, sign)),
    }[axis_name[-1]]


def _secondary_axis(primary: Vector, preferred: Vector) -> Vector:
    """Return a stable perpendicular control axis for two-axis gaze."""
    secondary = preferred - primary * preferred.dot(primary)
    if secondary.length < 1.0e-8:
        fallback = Vector((1.0, 0.0, 0.0))
        secondary = fallback - primary * fallback.dot(primary)
    return secondary.normalized()


def _signed_face_value(entry, value: float) -> float:
    """Apply mapping calibration without unsigned expression clamping."""
    result = (_finite_float(value) - entry.offset) * entry.gain
    if entry.invert:
        result = -result
    return result * entry.influence


# Upper lids follow the eyes: they drop a little when looking down and lift
# when looking up. Lid shape keys already do this; bone lids need the help.
LID_FOLLOW_GAZE = 0.3


def _face_control_values(blendshapes: dict[str, float]) -> dict[str, float]:
    def get(name: str) -> float:
        return _finite_float(blendshapes.get(name, 0.0))

    def lid(side: str) -> float:
        blink = get(f"eyeBlink{side}")
        follow = get(f"eyeLookDown{side}") - get(f"eyeLookUp{side}")
        return min(blink + LID_FOLLOW_GAZE * follow * (1.0 - blink), 1.0)

    return {
        "FACE_JAW_OPEN": get("jawOpen"),
        "FACE_JAW_SIDE": get("jawRight") - get("jawLeft"),
        "FACE_BROW_INNER": get("browInnerUp"),
        "FACE_BROW_OUTER_L": get("browOuterUpLeft") - get("browDownLeft"),
        "FACE_BROW_OUTER_R": get("browOuterUpRight") - get("browDownRight"),
        "FACE_LID_L": lid("Left"),
        "FACE_LID_R": lid("Right"),
        "FACE_CHEEK_L": max(get("cheekSquintLeft"), get("cheekPuff")),
        "FACE_CHEEK_R": max(get("cheekSquintRight"), get("cheekPuff")),
        "FACE_MOUTH_CORNER_L": get("mouthSmileLeft") - get("mouthFrownLeft"),
        "FACE_MOUTH_CORNER_R": get("mouthSmileRight") - get("mouthFrownRight"),
        "FACE_LIP_TOP": max(
            get("mouthUpperUpLeft"), get("mouthUpperUpRight"), get("mouthShrugUpper")
        ),
        "FACE_LIP_BOTTOM": max(
            get("mouthLowerDownLeft"),
            get("mouthLowerDownRight"),
            get("mouthShrugLower"),
        ),
    }


def _eye_gaze(blendshapes: dict[str, float], side: str) -> tuple[float, float]:
    def get(name: str) -> float:
        return _finite_float(blendshapes.get(name, 0.0))

    if side == "L":
        horizontal = get("eyeLookOutLeft") - get("eyeLookInLeft")
        vertical = get("eyeLookUpLeft") - get("eyeLookDownLeft")
    else:
        horizontal = get("eyeLookInRight") - get("eyeLookOutRight")
        vertical = get("eyeLookUpRight") - get("eyeLookDownRight")
    return horizontal, vertical


def _apply_eye_gaze(
    armature,
    entries,
    frame,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    """Point the eyes: rotate eye bones, or move look-at targets.

    Directions are in face space, so the mapping's axis only matters for
    its sign through gain and invert.
    """
    changed = set()
    space = face_frame(armature, entries)
    for semantic_id, side in (("EYE_L", "L"), ("EYE_R", "R")):
        entry = entries.get(semantic_id)
        bone = _mapped_bone(armature, entries, semantic_id)
        if not entry or not bone:
            continue
        horizontal, vertical = _eye_gaze(frame.blendshapes, side)
        yaw = _signed_face_value(entry, horizontal) * GAZE_ANGLE
        pitch = _signed_face_value(entry, vertical) * GAZE_ANGLE
        if entry.channel_mode == "LOCATION":
            distance = gaze_pivot_distance(armature, bone, space)
            offset = (
                space.lateral * math.tan(yaw) + space.up * math.tan(pitch)
            ) * distance
            target_pose.set_location(semantic_id, bone, to_bone(bone, offset))
        else:
            # Positive yaw turns the gaze to the subject's left (about up);
            # positive pitch looks up (about the lateral axis, negated).
            rotation = Quaternion(to_bone(bone, space.up).normalized(), yaw)
            rotation @= Quaternion(to_bone(bone, space.lateral).normalized(), -pitch)
            target_pose.set_basis_rotation(
                semantic_id,
                bone,
                rotation,
                correction=entry.correction_euler,
            )
        changed.add(bone)
    return changed


def _apply_face_controls(
    armature,
    entries,
    frame,
    target_pose: TargetPose,
) -> set[bpy.types.PoseBone]:
    changed = set()
    values = _face_control_values(frame.blendshapes)
    rotation_groups = {}
    location_groups = {}
    for semantic_id, value in values.items():
        entry = entries.get(semantic_id)
        bone = _mapped_bone(armature, entries, semantic_id)
        if not entry or not bone:
            continue
        evaluated = _expression_value(entry, value)
        if entry.channel_mode == "CUSTOM_PROPERTY" and entry.property_name:
            target_pose.set_property(
                semantic_id,
                bone,
                entry.property_name,
                evaluated,
            )
            changed.add(bone)
        elif entry.channel_mode == "LOCATION":
            location_groups.setdefault(bone, []).append((entry, evaluated))
        else:
            rotation_groups.setdefault(bone, []).append((entry, evaluated))

    for bone, controls in rotation_groups.items():
        rotation = Quaternion()
        correction = Vector((0.0, 0.0, 0.0))
        for entry, value in controls:
            rotation @= Quaternion(_axis(entry.axis), value * math.radians(25.0))
            correction += Vector(entry.correction_euler)
        target_pose.set_basis_rotation(
            controls[0][0].semantic_id,
            bone,
            rotation,
            correction=correction,
        )
        changed.add(bone)

    space = face_frame(armature, entries) if location_groups else None
    for bone, controls in location_groups.items():
        offset = Vector((0.0, 0.0, 0.0))
        for entry, value in controls:
            travel = FACE_CONTROL_TRAVEL.get(
                entry.semantic_id, DEFAULT_FACE_CONTROL_TRAVEL
            )
            offset += face_axis(space, entry.axis) * (value * travel * space.scale)
        target_pose.set_location(
            controls[0][0].semantic_id,
            bone,
            to_bone(bone, offset),
        )
        changed.add(bone)

    return changed
