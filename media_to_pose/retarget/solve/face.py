"""Face output: ARKit/FACS shape keys, eye gaze and facial bones."""

from __future__ import annotations

import math

import bpy
from mathutils import Quaternion, Vector

from ... import (
    mapping,
)
from ...target_pose import TargetPose
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
            "AU25": jaw_open,
            "AU26": jaw_open,
            "AU27": max(
                jaw_open,
                get("mouthStretchLeft"),
                get("mouthStretchRight"),
            ),
            "AU28": max(get("mouthRollUpper"), get("mouthRollLower")),
        }
    )
    return values


def _matching_shape_blocks(armature, entry):
    yield from mapping.matching_shape_blocks(armature, entry)


def _apply_shapes(armature, entries, frame, scheme: str, target_pose: TargetPose):
    changed = []
    expression_values = (
        facs_values(frame.blendshapes) if scheme == "FACS" else frame.blendshapes
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


def _face_control_values(blendshapes: dict[str, float]) -> dict[str, float]:
    def get(name: str) -> float:
        return _finite_float(blendshapes.get(name, 0.0))

    return {
        "FACE_JAW_OPEN": get("jawOpen"),
        "FACE_JAW_SIDE": get("jawRight") - get("jawLeft"),
        "FACE_BROW_INNER": get("browInnerUp"),
        "FACE_BROW_OUTER_L": get("browOuterUpLeft") - get("browDownLeft"),
        "FACE_BROW_OUTER_R": get("browOuterUpRight") - get("browDownRight"),
        "FACE_LID_L": get("eyeBlinkLeft"),
        "FACE_LID_R": get("eyeBlinkRight"),
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
    changed = set()
    for semantic_id, side in (("EYE_L", "L"), ("EYE_R", "R")):
        entry = entries.get(semantic_id)
        bone = _mapped_bone(armature, entries, semantic_id)
        if not entry or not bone:
            continue
        horizontal, vertical = _eye_gaze(frame.blendshapes, side)
        if entry.channel_mode == "LOCATION":
            scale = max(bone.length * 0.3, 0.01)
            horizontal_axis = _axis(entry.axis)
            vertical_axis = _secondary_axis(horizontal_axis, Vector((0.0, 0.0, 1.0)))
            target_pose.set_location(
                semantic_id,
                bone,
                (
                    horizontal_axis * _signed_face_value(entry, horizontal)
                    + vertical_axis * _signed_face_value(entry, vertical)
                )
                * scale,
            )
        else:
            vertical_axis = _axis(entry.axis)
            horizontal_axis = _secondary_axis(vertical_axis, Vector((0.0, 0.0, 1.0)))
            rotation = Quaternion(
                horizontal_axis,
                _signed_face_value(entry, horizontal) * 0.35,
            )
            rotation @= Quaternion(
                vertical_axis,
                -_signed_face_value(entry, vertical) * 0.35,
            )
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

    for bone, controls in location_groups.items():
        direction = Vector((0.0, 0.0, 0.0))
        for entry, value in controls:
            direction += _axis(entry.axis) * value
        target_pose.set_location(
            controls[0][0].semantic_id,
            bone,
            direction * max(bone.length * 0.2, 0.005),
        )
        changed.add(bone)

    return changed
