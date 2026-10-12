"""Mapping configuration operators."""

from __future__ import annotations

import json
from pathlib import Path

import bpy
from bpy.props import EnumProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ...core.coordinator import COORDINATOR
from ...rig import mapping
from ...rig.face_schemes import (
    associated_meshes,
    matching_shape_blocks,
    shape_override_for_object,
    shape_scheme,
)
from ...rig.locks import suppress_manual_locks
from ...rig.names import normalize_name
from ...rig.presets import copy_section, load_json, save_json
from ..classes import register_classes, unregister_classes


class _IdleMutationMixin:
    @classmethod
    def poll(cls, _context):
        if COORDINATOR.is_busy():
            cls.poll_message_set("Wait for the active Media to Pose operation")
            return False
        return True


def armature_from_object(obj) -> bpy.types.Object | None:
    """Resolve an armature from the active rig or one of its VRM meshes."""
    if not obj:
        return None
    if obj.type == "ARMATURE":
        return obj
    find_armature = getattr(obj, "find_armature", None)
    if callable(find_armature):
        armature = find_armature()
        if armature and armature.type == "ARMATURE":
            return armature
    parent = obj.parent
    while parent:
        if parent.type == "ARMATURE":
            return parent
        parent = parent.parent
    return None


def active_armature(context) -> bpy.types.Object | None:
    settings = context.scene.mtp_settings
    if (
        settings.target_armature
        and settings.target_armature.name in context.view_layer.objects
    ):
        return settings.target_armature
    active = armature_from_object(context.view_layer.objects.active)
    if active:
        return active
    selected = {
        armature.as_pointer(): armature
        for obj in context.selected_objects
        if (armature := armature_from_object(obj)) is not None
    }
    return next(iter(selected.values())) if len(selected) == 1 else None


def selected_armature(context) -> bpy.types.Object | None:
    """Resolve the viewport selection without consulting the stored target."""
    active = armature_from_object(context.view_layer.objects.active)
    if active:
        return active
    selected = {
        armature.as_pointer(): armature
        for obj in context.selected_objects
        if (armature := armature_from_object(obj)) is not None
    }
    return next(iter(selected.values())) if len(selected) == 1 else None


class MTP_OT_use_selected_armature(_IdleMutationMixin, Operator):
    bl_idname = "mtp.use_selected_armature"
    bl_label = "Use Selected"
    bl_description = "Use the selected armature or character mesh as the target"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        armature = selected_armature(context)
        if armature is None:
            self.report({"ERROR"}, "Select one character armature or one of its meshes")
            return {"CANCELLED"}
        context.scene.mtp_settings.target_armature = armature
        self.report({"INFO"}, f"Target character: {armature.name}")
        return {"FINISHED"}


class MTP_OT_set_mapping_method(_IdleMutationMixin, Operator):
    bl_idname = "mtp.set_mapping_method"
    bl_label = "Choose Mapping Method"
    bl_description = "Choose how automatic bone assignments are generated"
    bl_options = {"REGISTER", "UNDO"}

    method: EnumProperty(
        items=(
            ("AUTO", "Detect Automatically", "Detect the rig family"),
            ("AUTO_RIG_PRO", "Auto-Rig Pro", "Use Auto-Rig Pro controls"),
            ("RIGIFY", "Rigify", "Use Rigify controls"),
            ("GENERIC", "Generic / Custom", "Use humanoid metadata and bone names"),
        )
    )

    def execute(self, context):
        armature = active_armature(context)
        if armature is None:
            return {"CANCELLED"}
        armature.mtp_rig_config.profile = self.method
        return {"FINISHED"}


def active_entry(armature):
    config = armature.mtp_rig_config
    if not config.mappings:
        return None
    index = min(max(config.active_mapping_index, 0), len(config.mappings) - 1)
    return config.mappings[index]


def active_mapping_section(context, armature) -> str:
    """Return the semantic section represented by the visible mapping tab."""
    section = armature.mtp_rig_config.subtab
    if section == "FACE_RIG":
        settings = context.scene.mtp_settings
        if settings.face_output in {"ARKIT", "FACS"}:
            return "SHAPES"
    return section


class MTP_OT_auto_map(_IdleMutationMixin, Operator):
    bl_idname = "mtp.auto_map"
    bl_label = "Update Automatic Mappings"
    bl_description = "Rebuild automatic assignments while preserving manual locks"
    bl_options = {"REGISTER", "UNDO"}

    force_profile: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        if not self.force_profile:
            mapping.clear_unlocked_automatic_mappings(armature)
        profile = mapping.auto_map(
            armature,
            force_profile=self.force_profile,
            settings=context.scene.mtp_settings,
        )
        context.scene.mtp_settings.target_armature = armature
        self.report({"INFO"}, f"Mapped {armature.name} as {profile}")
        return {"FINISHED"}


class MTP_OT_restore_profile(_IdleMutationMixin, Operator):
    bl_idname = "mtp.restore_profile"
    bl_label = "Replace With Mapping Defaults"
    bl_description = "Replace manual and automatic mappings with method defaults"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, _event):
        return context.window_manager.invoke_confirm(self, _event)

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        mapping.auto_map(
            armature, force_profile=True, settings=context.scene.mtp_settings
        )
        self.report({"INFO"}, "Profile defaults restored")
        return {"FINISHED"}


class MTP_OT_use_selected_bone(_IdleMutationMixin, Operator):
    bl_idname = "mtp.use_selected_bone"
    bl_label = "Use Selected Bone"
    bl_description = "Assign the active pose bone to the selected mapping"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        pose_bone = context.active_pose_bone
        entry = active_entry(armature)
        if not pose_bone or not entry:
            self.report({"ERROR"}, "Select a pose bone and mapping row")
            return {"CANCELLED"}
        with suppress_manual_locks():
            entry.target_kind = "BONE"
            entry.target_name = pose_bone.name
            entry.source = "MANUAL"
            entry.manual_lock = True
            entry.confidence = 1.0
        mapping.validate_mapping(armature, settings=context.scene.mtp_settings)
        return {"FINISHED"}


class MTP_OT_use_active_shape_key(_IdleMutationMixin, Operator):
    bl_idname = "mtp.use_active_shape_key"
    bl_label = "Use Active Shape Key"
    bl_description = "Assign the active mesh shape key to the selected mapping"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        armature = active_armature(context)
        entry = active_entry(armature) if armature else None
        obj = context.view_layer.objects.active
        block = obj.active_shape_key if obj and obj.type == "MESH" else None
        if not armature or not entry or entry.section != "SHAPES" or not block:
            self.report({"ERROR"}, "Select a mesh with an active shape key")
            return {"CANCELLED"}
        scheme = shape_scheme(block.name)
        if entry.shape_scheme in {"ARKIT", "FACS"} and scheme != entry.shape_scheme:
            self.report(
                {"ERROR"},
                f"Selected key is {scheme or 'unclassified'}, expected {entry.shape_scheme}",
            )
            return {"CANCELLED"}
        with suppress_manual_locks():
            entry.target_kind = "SHAPE_KEY"
            if entry.target_object in {"", "*"} and entry.target_name:
                override = shape_override_for_object(entry, obj.name)
                if override is None:
                    override = entry.shape_overrides.add()
                override.semantic_id = entry.semantic_id
                override.object_name = obj.name
                override.target_name = block.name
                override.enabled = True
                entry.target_object = "*"
                entry.show_shape_overrides = True
            else:
                entry.target_object = obj.name
                entry.target_name = block.name
            entry.source = "MANUAL"
            entry.manual_lock = True
            entry.confidence = 1.0
        mapping.validate_mapping(armature, settings=context.scene.mtp_settings)
        return {"FINISHED"}


class MTP_OT_add_shape_override(_IdleMutationMixin, Operator):
    bl_idname = "mtp.add_shape_override"
    bl_label = "Add Mesh Override"
    bl_description = "Override the grouped shape-key target for one mesh"
    bl_options = {"REGISTER", "UNDO"}

    object_name: StringProperty(name="Mesh")

    def invoke(self, context, _event):
        active = context.view_layer.objects.active
        if active and active.type == "MESH":
            self.object_name = active.name
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop_search(
            self,
            "object_name",
            bpy.data,
            "objects",
            text="Mesh",
        )

    def execute(self, context):
        armature = active_armature(context)
        entry = active_entry(armature) if armature else None
        obj = bpy.data.objects.get(self.object_name)
        if (
            not armature
            or not entry
            or entry.section != "SHAPES"
            or not obj
            or obj.type != "MESH"
            or obj not in associated_meshes(armature)
        ):
            self.report(
                {"ERROR"},
                "Choose a mesh associated with the target armature",
            )
            return {"CANCELLED"}
        existing = shape_override_for_object(entry, obj.name)
        if existing is not None:
            entry.active_shape_override_index = list(entry.shape_overrides).index(
                existing
            )
            entry.show_shape_overrides = True
            self.report({"WARNING"}, "This mesh already has an override")
            return {"FINISHED"}

        target_name = entry.target_name
        keys = obj.data.shape_keys
        if keys:
            normalized = normalize_name(entry.target_name, shape=True)
            match = next(
                (
                    block.name
                    for block in keys.key_blocks
                    if normalize_name(block.name, shape=True) == normalized
                    and (
                        entry.shape_scheme == "NONE"
                        or shape_scheme(block.name) == entry.shape_scheme
                    )
                ),
                None,
            )
            if match:
                target_name = match
        override = entry.shape_overrides.add()
        override.semantic_id = entry.semantic_id
        override.object_name = obj.name
        override.target_name = target_name
        entry.active_shape_override_index = len(entry.shape_overrides) - 1
        entry.target_object = "*"
        entry.show_shape_overrides = True
        entry.source = "MANUAL"
        entry.manual_lock = True
        entry.confidence = 1.0
        mapping.validate_mapping(armature, settings=context.scene.mtp_settings)
        return {"FINISHED"}


class MTP_OT_remove_shape_override(_IdleMutationMixin, Operator):
    bl_idname = "mtp.remove_shape_override"
    bl_label = "Remove Mesh Override"
    bl_options = {"REGISTER", "UNDO"}

    override_index: IntProperty(default=-1, options={"HIDDEN"})

    def execute(self, context):
        armature = active_armature(context)
        entry = active_entry(armature) if armature else None
        if not armature or not entry or entry.section != "SHAPES":
            return {"CANCELLED"}
        index = (
            self.override_index
            if self.override_index >= 0
            else entry.active_shape_override_index
        )
        if not (0 <= index < len(entry.shape_overrides)):
            return {"CANCELLED"}
        entry.shape_overrides.remove(index)
        entry.active_shape_override_index = min(
            index,
            max(0, len(entry.shape_overrides) - 1),
        )
        entry.source = "MANUAL"
        entry.manual_lock = True
        mapping.validate_mapping(armature, settings=context.scene.mtp_settings)
        return {"FINISHED"}


class MTP_OT_select_shape_override(Operator):
    bl_idname = "mtp.select_shape_override"
    bl_label = "Select Override Target"

    override_index: IntProperty(options={"HIDDEN"})

    def execute(self, context):
        armature = active_armature(context)
        entry = active_entry(armature) if armature else None
        if (
            not armature
            or not entry
            or not (0 <= self.override_index < len(entry.shape_overrides))
        ):
            return {"CANCELLED"}
        override = entry.shape_overrides[self.override_index]
        obj = bpy.data.objects.get(override.object_name)
        if not obj or obj.type != "MESH" or not obj.data.shape_keys:
            self.report({"ERROR"}, "Override mesh is missing")
            return {"CANCELLED"}
        block = next(
            (
                candidate
                for candidate in obj.data.shape_keys.key_blocks
                if normalize_name(candidate.name, shape=True)
                == normalize_name(override.target_name, shape=True)
            ),
            None,
        )
        if not block:
            self.report({"ERROR"}, "Override shape key is missing")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(block)
        return {"FINISHED"}


class MTP_OT_select_mapping_target(Operator):
    bl_idname = "mtp.select_mapping_target"
    bl_label = "Select Target"
    bl_description = "Select the mapped bone or shape-key object"

    def execute(self, context):
        armature = active_armature(context)
        entry = active_entry(armature) if armature else None
        if not armature or not entry or not entry.target_name:
            return {"CANCELLED"}
        if entry.target_kind in {"BONE", "CUSTOM_PROPERTY"}:
            target_bone = armature.pose.bones.get(entry.target_name)
            if target_bone is None:
                self.report({"ERROR"}, "Mapped bone no longer exists")
                return {"CANCELLED"}
            context.view_layer.objects.active = armature
            armature.select_set(True)
            if context.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
            for pose_bone in armature.pose.bones:
                pose_bone.select = False
            target_bone.select = True
            armature.data.bones.active = target_bone.bone
            return {"FINISHED"}
        match = next(
            matching_shape_blocks(armature, entry),
            None,
        )
        if not match:
            self.report({"ERROR"}, "Mapped shape key no longer exists")
            return {"CANCELLED"}
        target, target_block = match
        bpy.ops.object.mode_set(mode="OBJECT") if context.mode != "OBJECT" else None
        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        context.view_layer.objects.active = target
        for index, key in enumerate(target.data.shape_keys.key_blocks):
            if key == target_block:
                target.active_shape_key_index = index
                break
        return {"FINISHED"}


class MTP_OT_select_problem(Operator):
    bl_idname = "mtp.select_problem"
    bl_label = "Select Problem"
    bl_description = "Open the mapping row and select its target when available"

    mapping_index: IntProperty(options={"HIDDEN"})

    def execute(self, context):
        armature = active_armature(context)
        if not armature or not (
            0 <= self.mapping_index < len(armature.mtp_rig_config.mappings)
        ):
            return {"CANCELLED"}
        entry = armature.mtp_rig_config.mappings[self.mapping_index]
        armature.mtp_rig_config.active_mapping_index = self.mapping_index
        armature.mtp_rig_config.subtab = (
            "FACE_RIG" if entry.section in {"SHAPES", "FACE_RIG"} else entry.section
        )
        if entry.section == "SHAPES" and entry.shape_scheme in {
            "ARKIT",
            "FACS",
        }:
            context.scene.mtp_settings.face_output = entry.shape_scheme
        elif entry.section == "FACE_RIG":
            context.scene.mtp_settings.face_output = "BONES"
        armature.mtp_rig_config.filter_mode = "ALL"
        if entry.target_name:
            return bpy.ops.mtp.select_mapping_target()
        return {"FINISHED"}


class MTP_OT_reset_all_post_processing(_IdleMutationMixin, Operator):
    bl_idname = "mtp.reset_all_post_processing"
    bl_label = "Reset All Corrections"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        from ...postprocess import (
            ensure_module_configs,
            reset_module_defaults,
            validate_modules,
        )

        if not ensure_module_configs(armature.mtp_rig_config):
            return {"CANCELLED"}
        module_ids = [
            item.module_id for item in armature.mtp_rig_config.post_processing_modules
        ]
        for module_id in module_ids:
            reset_module_defaults(armature.mtp_rig_config, module_id)
        validate_modules(armature)
        return {"FINISHED"}


class MTP_OT_clear_overrides(_IdleMutationMixin, Operator):
    bl_idname = "mtp.clear_overrides"
    bl_label = "Clear Overrides"
    bl_description = "Unlock manual mappings in the active section"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        section = active_mapping_section(context, armature)
        if section == "BODY":
            armature.mtp_rig_config.fk_switches_manual = False
        for entry in armature.mtp_rig_config.mappings:
            if entry.section != section:
                continue
            if (
                section == "SHAPES"
                and entry.shape_scheme != context.scene.mtp_settings.face_output
            ):
                continue
            if entry.section == "SHAPES":
                entry.shape_overrides.clear()
            entry.manual_lock = False
            if entry.source == "MANUAL":
                entry.source = "NONE"
        mapping.auto_map(armature, settings=context.scene.mtp_settings)
        return {"FINISHED"}


class MTP_OT_reset_section(_IdleMutationMixin, Operator):
    bl_idname = "mtp.reset_section"
    bl_label = "Reset Section"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        section = active_mapping_section(context, armature)
        if section == "BODY":
            properties_to_clear = (
                "arm_fk_switch_bone_l",
                "arm_fk_switch_property_l",
                "arm_ik_target_bone_l",
                "arm_ik_pole_bone_l",
                "arm_fk_switch_bone_r",
                "arm_fk_switch_property_r",
                "arm_ik_target_bone_r",
                "arm_ik_pole_bone_r",
                "leg_fk_switch_bone_l",
                "leg_fk_switch_property_l",
                "leg_ik_target_bone_l",
                "leg_ik_pole_bone_l",
                "leg_ik_toe_bone_l",
                "leg_fk_switch_bone_r",
                "leg_fk_switch_property_r",
                "leg_ik_target_bone_r",
                "leg_ik_pole_bone_r",
                "leg_ik_toe_bone_r",
            )
            with suppress_manual_locks():
                for field in properties_to_clear:
                    setattr(armature.mtp_rig_config, field, "")
                armature.mtp_rig_config.fk_switches_manual = False
        with suppress_manual_locks():
            for entry in armature.mtp_rig_config.mappings:
                if entry.section != section:
                    continue
                if (
                    section == "SHAPES"
                    and entry.shape_scheme != context.scene.mtp_settings.face_output
                ):
                    continue
                entry.target_name = ""
                entry.target_object = ""
                entry.shape_overrides.clear()
                entry.manual_lock = False
                entry.source = "NONE"
                entry.status = "MISSING" if entry.required else "OPTIONAL"
        mapping.auto_map(armature, settings=context.scene.mtp_settings)
        return {"FINISHED"}


def _mirror_name(name: str, direction: str) -> str:
    replacements = (
        (".L", ".R"),
        (".l", ".r"),
        ("_L", "_R"),
        ("_l", "_r"),
        ("Left", "Right"),
        ("left", "right"),
    )
    if direction == "R_TO_L":
        replacements = tuple((b, a) for a, b in replacements)
    for source, target in replacements:
        if source in name:
            return name.replace(source, target)
    return name


class MTP_OT_mirror_mapping(_IdleMutationMixin, Operator):
    bl_idname = "mtp.mirror_mapping"
    bl_label = "Mirror Mapping"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(
        items=(("L_TO_R", "Left to Right", ""), ("R_TO_L", "Right to Left", "")),
        default="L_TO_R",
    )

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        entries = mapping.mapping_by_id(armature.mtp_rig_config)
        source_suffix, target_suffix = (
            ("_L", "_R") if self.direction == "L_TO_R" else ("_R", "_L")
        )
        copied = 0
        with suppress_manual_locks():
            for semantic_id, source in entries.items():
                if not semantic_id.endswith(source_suffix):
                    continue
                target = entries.get(semantic_id[:-2] + target_suffix)
                if not target or not source.target_name:
                    continue
                candidate = _mirror_name(source.target_name, self.direction)
                if candidate not in armature.pose.bones:
                    continue
                target.target_name = candidate
                target.target_kind = source.target_kind
                target.channel_mode = source.channel_mode
                target.axis = source.axis
                target.gain = source.gain
                target.offset = source.offset
                correction = source.correction_euler
                target.correction_euler = (
                    -correction[0],
                    correction[1],
                    -correction[2],
                )
                target.source = "MANUAL"
                target.manual_lock = True
                target.confidence = 1.0
                copied += 1
        mapping.validate_mapping(armature, settings=context.scene.mtp_settings)
        self.report({"INFO"}, f"Mirrored {copied} mappings")
        return {"FINISHED"}


class MTP_OT_export_preset(Operator, ExportHelper):
    bl_idname = "mtp.export_preset"
    bl_label = "Export Mapping Preset"

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        save_json(self.filepath, armature)
        self.report({"INFO"}, f"Exported {Path(self.filepath).name}")
        return {"FINISHED"}


class MTP_OT_import_preset(_IdleMutationMixin, Operator, ImportHelper):
    bl_idname = "mtp.import_preset"
    bl_label = "Import Mapping Preset"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        armature = active_armature(context)
        if not armature:
            return {"CANCELLED"}
        try:
            load_json(self.filepath, armature, settings=context.scene.mtp_settings)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Imported {Path(self.filepath).name}")
        return {"FINISHED"}


class MTP_OT_duplicate_mapping(_IdleMutationMixin, Operator):
    bl_idname = "mtp.duplicate_mapping"
    bl_label = "Duplicate From Armature"
    bl_options = {"REGISTER", "UNDO"}

    source_name: StringProperty(name="Source Armature")

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop_search(self, "source_name", bpy.data, "objects", text="Source")

    def execute(self, context):
        target = active_armature(context)
        source = bpy.data.objects.get(self.source_name)
        if not target or not source or source.type != "ARMATURE" or source == target:
            self.report({"ERROR"}, "Choose a different source armature")
            return {"CANCELLED"}
        copy_section(source, target, settings=context.scene.mtp_settings)
        return {"FINISHED"}


class MTP_OT_duplicate_post_processing(_IdleMutationMixin, Operator):
    bl_idname = "mtp.duplicate_post_processing"
    bl_label = "Duplicate Corrections From Armature"
    bl_options = {"REGISTER", "UNDO"}

    source_name: StringProperty(name="Source Armature")

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        self.layout.prop_search(
            self,
            "source_name",
            bpy.data,
            "objects",
            text="Source",
        )

    def execute(self, context):
        target = active_armature(context)
        source = bpy.data.objects.get(self.source_name)
        if not target or not source or source.type != "ARMATURE" or source == target:
            self.report(
                {"ERROR"},
                "Choose a different source armature",
            )
            return {"CANCELLED"}
        copy_section(
            source,
            target,
            section="POST_PROCESSING",
            settings=context.scene.mtp_settings,
        )
        return {"FINISHED"}


class MTP_MT_mapping_presets(bpy.types.Menu):
    bl_idname = "MTP_MT_mapping_presets"
    bl_label = "Mapping Presets"

    def draw(self, _context):
        layout = self.layout
        layout.operator("mtp.import_preset", icon="IMPORT")
        layout.operator("mtp.export_preset", icon="EXPORT")
        layout.separator()
        layout.operator("mtp.duplicate_mapping", icon="DUPLICATE")


class MTP_MT_mapping_reset(bpy.types.Menu):
    bl_idname = "MTP_MT_mapping_reset"
    bl_label = "Reset or Replace"

    def draw(self, _context):
        layout = self.layout
        layout.operator("mtp.restore_profile", icon="RECOVER_LAST")
        layout.operator("mtp.reset_section", icon="FILE_REFRESH")
        layout.operator("mtp.clear_overrides", icon="UNLOCKED")


class MTP_MT_post_processing_presets(bpy.types.Menu):
    bl_idname = "MTP_MT_post_processing_presets"
    bl_label = "Pose Correction Presets"

    def draw(self, _context):
        layout = self.layout
        layout.operator(
            "mtp.import_preset",
            text="Import Rig Preset",
            icon="IMPORT",
        )
        layout.operator(
            "mtp.export_preset",
            text="Export Rig Preset",
            icon="EXPORT",
        )
        layout.separator()
        layout.operator(
            "mtp.duplicate_post_processing",
            icon="DUPLICATE",
        )
        layout.operator(
            "mtp.reset_all_post_processing",
            text="Reset All Settings",
            icon="FILE_REFRESH",
        )


CLASSES = (
    MTP_OT_use_selected_armature,
    MTP_OT_set_mapping_method,
    MTP_OT_auto_map,
    MTP_OT_restore_profile,
    MTP_OT_use_selected_bone,
    MTP_OT_use_active_shape_key,
    MTP_OT_add_shape_override,
    MTP_OT_remove_shape_override,
    MTP_OT_select_shape_override,
    MTP_OT_select_mapping_target,
    MTP_OT_select_problem,
    MTP_OT_reset_all_post_processing,
    MTP_OT_clear_overrides,
    MTP_OT_reset_section,
    MTP_OT_mirror_mapping,
    MTP_OT_export_preset,
    MTP_OT_import_preset,
    MTP_OT_duplicate_mapping,
    MTP_OT_duplicate_post_processing,
    MTP_MT_mapping_presets,
    MTP_MT_mapping_reset,
    MTP_MT_post_processing_presets,
)


def register() -> None:
    register_classes(CLASSES)


def unregister() -> None:
    unregister_classes(CLASSES)
