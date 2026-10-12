"""Operators that calibrate a face mapping's output range from the rig."""

from __future__ import annotations

from bpy.props import EnumProperty
from bpy.types import Operator

from ...retarget.solve.face_values import bone_output
from ...rig import mapping
from ...rig.face_defaults import (
    EYES,
    RANGES_VERSION,
    apply_default_ranges,
    has_ranges,
    is_signed,
)
from ...rig.face_schemes import matching_shape_blocks
from ...rig.face_space import face_frame
from ...rig.locks import suppress_manual_locks
from ..classes import register_classes, unregister_classes
from .mapping import _IdleMutationMixin, active_armature, active_entry


def _face_entry(operator, context):
    armature = active_armature(context)
    entry = active_entry(armature) if armature else None
    if entry is None or not has_ranges(entry) or not entry.target_name:
        operator.report({"ERROR"}, "Select a mapped face row")
        return None, None
    return armature, entry


class MTP_OT_set_face_range_from_pose(_IdleMutationMixin, Operator):
    bl_idname = "mtp.set_face_range_from_pose"
    bl_label = "Set from Pose"
    bl_description = (
        "Use the target's current pose (face bone) or value (shape key) as the "
        "mapping's output at full strength"
    )
    bl_options = {"REGISTER", "UNDO"}

    side: EnumProperty(
        name="Side",
        items=(
            ("FULL", "Full", "Output at full strength"),
            ("NEGATIVE", "Negative Full", "Output at full opposite strength"),
        ),
        default="FULL",
    )

    def execute(self, context):
        armature, entry = _face_entry(self, context)
        if entry is None:
            return {"CANCELLED"}
        if entry.semantic_id in EYES:
            self.report({"ERROR"}, "Set the gaze angle directly")
            return {"CANCELLED"}
        if entry.section == "SHAPES":
            values = [
                block.value for _obj, block in matching_shape_blocks(armature, entry)
            ]
            if not values:
                self.report({"ERROR"}, "The shape key was not found")
                return {"CANCELLED"}
            value = max(values)
        else:
            bone = armature.pose.bones.get(entry.target_name)
            if bone is None:
                self.report({"ERROR"}, "The target bone was not found")
                return {"CANCELLED"}
            entries = mapping.mapping_by_id(armature.mtp_rig_config)
            value = bone_output(bone, entry, face_frame(armature, entries))
        if self.side == "NEGATIVE" and not is_signed(entry.semantic_id):
            self.report({"ERROR"}, "This control has no negative side")
            return {"CANCELLED"}
        with suppress_manual_locks():
            if self.side == "NEGATIVE":
                entry.output_negative = value
            else:
                entry.output_full = value
            entry.ranges_version = RANGES_VERSION
            entry.source = "MANUAL"
            entry.manual_lock = True
        mapping.validate_mapping(armature, settings=context.scene.mtp_settings)
        self.report({"INFO"}, f"{entry.label}: output {value:.4g}")
        return {"FINISHED"}


class MTP_OT_reset_face_ranges(_IdleMutationMixin, Operator):
    bl_idname = "mtp.reset_face_ranges"
    bl_label = "Reset Ranges"
    bl_description = (
        "Restore the default input and output ranges for this row's channel "
        "and the face's size"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        armature, entry = _face_entry(self, context)
        if entry is None:
            return {"CANCELLED"}
        entries = mapping.mapping_by_id(armature.mtp_rig_config)
        with suppress_manual_locks():
            apply_default_ranges(armature, entries, entry)
        mapping.validate_mapping(armature, settings=context.scene.mtp_settings)
        return {"FINISHED"}


CLASSES = (MTP_OT_set_face_range_from_pose, MTP_OT_reset_face_ranges)


def register() -> None:
    register_classes(CLASSES)


def unregister() -> None:
    unregister_classes(CLASSES)
