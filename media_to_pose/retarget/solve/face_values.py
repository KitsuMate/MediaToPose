"""Face expression values: what a face output drives, and what the rig shows.

``expression_values`` names the values one face output drives: ARKit
coefficients, FACS action units, VRM expressions, or the ``FACE_*`` controls
for bone output. ``mapped_expressions`` gives how strongly each mapped
target should show its expression, through the mapping's input range, and
lists the expressions with no target; ``FaceValueReader`` reads the same
strengths back from the posed rig through the output range, so the two can
be compared whatever the targets' units. Strengths are shares of the
mapped range: 0 to 1, or -1 to 1 for signed controls.
"""

from __future__ import annotations

import math

from mathutils import Vector

from ...core.constants import SHAPE_SCHEMES
from ...core.face_ranges import share_of_output, strength
from ...core.requests import FaceOutput
from ...rig.face_defaults import RANGE_FIELDS, is_signed
from ...rig.face_schemes import matching_shape_blocks
from ...rig.face_space import bone_axis, face_axis, face_frame, gaze_tracker
from .common import _mapped_bone
from .face import (
    _face_control_values,
    _finite_float,
    facs_values,
    vrm_values,
)

# The blink of each face output, left then right.
BLINK_EXPRESSIONS = {
    FaceOutput.ARKIT: ("eyeBlinkLeft", "eyeBlinkRight"),
    FaceOutput.FACS: ("AU45_L", "AU45_R"),
    FaceOutput.VRM: ("blinkLeft", "blinkRight"),
    FaceOutput.BONES: ("FACE_LID_L", "FACE_LID_R"),
}
# Detected values below this are not reported as unmapped.
UNMAPPED_THRESHOLD = 0.1


def expression_values(
    blendshapes: dict[str, float], face_output: FaceOutput
) -> dict[str, float]:
    """The values ``face_output`` drives, before the mapping's ranges."""
    if face_output == FaceOutput.FACS:
        return facs_values(blendshapes)
    if face_output == FaceOutput.VRM:
        return vrm_values(blendshapes)
    if face_output == FaceOutput.BONES:
        return _face_control_values(blendshapes)
    return {name: _finite_float(value) for name, value in blendshapes.items()}


def _entry(entries, face_output: FaceOutput, name: str):
    key = name if face_output == FaceOutput.BONES else f"{face_output}_{name}"
    entry = entries.get(key)
    if not entry or not entry.enabled or not entry.target_name:
        return None
    return entry


def mapped_expressions(
    entries, blendshapes: dict[str, float], face_output: FaceOutput
) -> tuple[dict[str, float], list[str]]:
    """Strength of each mapped target, and the expressions with no target.

    An expression is unmapped only when it was detected (at least
    ``UNMAPPED_THRESHOLD``), so a rig without, say, a tongue key is not
    blamed for faces that never show the tongue.
    """
    applied: dict[str, float] = {}
    unmapped: list[str] = []
    for name, raw in expression_values(blendshapes, face_output).items():
        entry = _entry(entries, face_output, name)
        if entry is None:
            if abs(raw) >= UNMAPPED_THRESHOLD:
                unmapped.append(name)
            continue
        applied[name] = strength(entry, raw, signed=is_signed(entry.semantic_id))
    return applied, sorted(unmapped)


def face_value_names(entries, face_output: FaceOutput) -> list[str]:
    """Every expression of ``face_output`` that has a mapped target."""
    return [
        name
        for name in expression_values({}, face_output)
        if _entry(entries, face_output, name) is not None
    ] or _scheme_entry_names(entries, face_output)


def _scheme_entry_names(entries, face_output: FaceOutput) -> list[str]:
    # ARKit coefficients only appear in detections, so list them from the
    # mapping instead.
    if face_output == FaceOutput.BONES:
        return []
    prefix = f"{face_output}_"
    return sorted(
        key.removeprefix(prefix)
        for key, entry in entries.items()
        if key.startswith(prefix) and entry.enabled and entry.target_name
    )


def bone_output(bone, entry, space) -> float:
    """What a face bone control shows now, in its output units.

    Metres along the face-space axis, degrees about the mapped axis, or the
    custom property's value: the inverse of what bone output applies.
    """
    if entry.channel_mode == "CUSTOM_PROPERTY" and entry.property_name:
        return _finite_float(bone.get(entry.property_name, 0.0))
    if entry.channel_mode == "LOCATION":
        offset = bone.bone.matrix_local.to_3x3() @ Vector(bone.location)
        return offset.dot(face_axis(space, entry.axis))
    rotation = bone.matrix_basis.to_quaternion()
    axis, angle = rotation.to_axis_angle()
    return math.degrees(angle * Vector(axis).dot(bone_axis(entry.axis)))


class FaceValueReader:
    """Read face values back from the rig as it is posed, frame after frame.

    Each target's value (a shape key's, the largest when several meshes
    carry the key; a face bone's offset or angle along the mapped axis) is
    turned into a strength through the mapping's output range. Targets are
    looked up once.
    """

    def __init__(self, armature, entries, face_output: FaceOutput):
        self.face_output = FaceOutput(face_output)
        self.names = face_value_names(entries, self.face_output)
        self.space = (
            face_frame(armature, entries)
            if self.face_output == FaceOutput.BONES
            else None
        )
        self.bones = {}
        self.blocks = {}
        for name in self.names:
            entry = _entry(entries, self.face_output, name)
            if entry is None:
                continue
            if self.space is not None:
                bone = armature.pose.bones.get(entry.target_name)
                if bone is not None:
                    self.bones[name] = (bone, entry)
            else:
                blocks = [
                    block for _obj, block in matching_shape_blocks(armature, entry)
                ]
                if blocks:
                    self.blocks[name] = (blocks, entry)

    def read(self) -> dict[str, float]:
        values = {
            name: share_of_output(entry, max(block.value for block in blocks))
            for name, (blocks, entry) in self.blocks.items()
        }
        for name, (bone, entry) in self.bones.items():
            values[name] = share_of_output(entry, bone_output(bone, entry, self.space))
        return values


MAPPED_STATUSES = {"VALID", "INFO", "WARNING"}


def face_ranges(entry) -> dict[str, float]:
    """A mapping's input and output ranges, rounded for JSON."""
    return {name: round(float(getattr(entry, name)), 6) for name in RANGE_FIELDS}


def _vector(value) -> list[float]:
    return [round(float(component), 4) for component in value]


def describe_face_rig(armature, config, entries) -> dict:
    """The face part of ``describe_rig``: expressions, controls and eyes.

    ``expressions`` lists, per shape key scheme, the mapped and unmapped
    expressions; ``controls`` the bone controls of ``BONES`` output with
    their channel and face-space axis; ``eyes`` each eye bone with the bone
    that aims at it (for look-at rigs) and its rest direction in armature
    space; ``face_scale`` the eye distance that sizes bone travel.
    """
    expressions: dict[str, dict[str, list[str]]] = {}
    for scheme in SHAPE_SCHEMES:
        prefix = f"{scheme}_"
        rows = [
            entry
            for entry in config.mappings
            if entry.section == "SHAPES"
            and entry.enabled
            and entry.semantic_id.startswith(prefix)
        ]
        mapped = [
            row for row in rows if row.target_name and row.status in MAPPED_STATUSES
        ]
        expressions[scheme] = {
            "mapped": sorted(row.semantic_id.removeprefix(prefix) for row in mapped),
            "unmapped": sorted(
                row.semantic_id.removeprefix(prefix)
                for row in rows
                if row not in mapped
            ),
        }
    controls = {}
    unmapped_controls = []
    for entry in config.mappings:
        if entry.section != "FACE_RIG" or not entry.enabled:
            continue
        if entry.target_name and entry.status in MAPPED_STATUSES:
            controls[entry.semantic_id] = {
                "bone": entry.target_name,
                "channel": entry.channel_mode,
                "axis": entry.axis,
                "ranges": face_ranges(entry),
            }
        else:
            unmapped_controls.append(entry.semantic_id)
    eyes = {}
    for semantic_id in ("EYE_L", "EYE_R"):
        bone = _mapped_bone(armature, entries, semantic_id)
        if bone is None:
            eyes[semantic_id] = None
            continue
        tracker = gaze_tracker(armature, bone)
        aim = (tracker or bone).bone
        eyes[semantic_id] = {
            "bone": bone.name,
            "look_at_target": tracker is not None,
            "tracked_by": tracker.name if tracker is not None else "",
            "rest_direction": _vector(aim.matrix_local.col[1][:3]),
        }
    space = face_frame(armature, entries)
    return {
        "expressions": expressions,
        "controls": dict(sorted(controls.items())),
        "unmapped_controls": sorted(unmapped_controls),
        "eyes": eyes,
        "face_scale": round(space.scale, 5),
        "face_up": _vector(space.up),
        "face_forward": _vector(space.forward),
    }


__all__ = [
    "BLINK_EXPRESSIONS",
    "UNMAPPED_THRESHOLD",
    "FaceValueReader",
    "bone_output",
    "describe_face_rig",
    "face_ranges",
    "expression_values",
    "face_value_names",
    "mapped_expressions",
]
