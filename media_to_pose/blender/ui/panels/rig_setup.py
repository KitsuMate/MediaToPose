"""The Rig Setup panel and its mapping list."""

from __future__ import annotations

import bpy
from bpy.types import Panel, UIList

from ....core.coordinator import COORDINATOR
from ....rig.profiles.common import PROFILE_LABELS, detect_profile, profile_label
from ...operators.mapping import active_entry, active_mapping_section
from ..widgets import (
    ISSUE_CRITICAL,
    ISSUE_WARNING,
    STATUS_ICONS,
    _configured_target,
    _draw_active_operation,
    _draw_issue,
    _draw_left_disclosure,
    _mapping_problem_entries,
    _mapping_readiness,
)


class MTP_UL_mappings(UIList):
    bl_idname = "MTP_UL_mappings"

    def draw_item(
        self,
        _context,
        layout,
        _data,
        item,
        _icon,
        _active_data,
        _active_property,
        _index=0,
        _flt_flag=0,
    ):
        row = layout.row(align=True)
        row.label(text="", icon=STATUS_ICONS.get(item.status, "QUESTION"))
        split = row.split(factor=0.48)
        split.label(text=item.label)
        destination = item.target_name or "Not Mapped"
        suffix = ""
        if item.source == "HEURISTIC":
            suffix = f"  {item.confidence:.0%}"
        elif item.source == "MANUAL":
            suffix = "  M"
        split.label(text=f"{destination}{suffix}")
        if item.manual_lock:
            row.label(text="", icon="LOCKED")

    def filter_items(self, context, data, property_name):
        items = getattr(data, property_name)
        flags = []
        section = data.subtab
        settings = getattr(context.scene, "mtp_settings", None)
        selected_output = settings.face_output if settings else "ARKIT"
        effective_section = (
            "SHAPES"
            if section == "FACE_RIG" and selected_output in {"ARKIT", "FACS"}
            else section
        )
        text = data.filter_text.casefold().strip()
        mode = data.filter_mode
        for item in items:
            visible = item.section == effective_section
            if visible and effective_section == "SHAPES":
                visible = item.shape_scheme == selected_output
            if visible and text:
                visible = (
                    text in item.label.casefold() or text in item.target_name.casefold()
                )
            if visible and mode == "MISSING":
                visible = item.status in {"MISSING", "ERROR"}
            elif visible and mode == "WARNINGS":
                visible = item.status in {"WARNING", "ERROR", "MISSING"}
            elif visible and mode == "MANUAL":
                visible = item.source == "MANUAL"
            elif visible and mode == "OPTIONAL":
                visible = not item.required
            flags.append(self.bitflag_filter_item if visible else 0)
        return flags, []


def _draw_shape_overrides(layout, entry):
    header = layout.row(align=True)
    header.prop(
        entry,
        "show_shape_overrides",
        text=f"Mesh Overrides ({len(entry.shape_overrides)})",
        icon=(
            "DISCLOSURE_TRI_DOWN"
            if entry.show_shape_overrides
            else "DISCLOSURE_TRI_RIGHT"
        ),
        emboss=False,
    )
    header.operator(
        "mtp.add_shape_override",
        text="",
        icon="ADD",
    )
    if not entry.show_shape_overrides:
        return
    if not entry.shape_overrides:
        layout.label(
            text="All meshes use the grouped target name",
            icon="INFO",
        )
        return
    for index, override in enumerate(entry.shape_overrides):
        override_box = layout.box()
        row = override_box.row(align=True)
        row.label(
            text="",
            icon=STATUS_ICONS.get(
                override.status,
                "RADIOBUT_OFF",
            ),
        )
        row.prop(
            override,
            "enabled",
            text="",
            icon=("CHECKBOX_HLT" if override.enabled else "CHECKBOX_DEHLT"),
        )
        row.prop_search(
            override,
            "object_name",
            bpy.data,
            "objects",
            text="",
        )
        select_op = row.operator(
            "mtp.select_shape_override",
            text="",
            icon="RESTRICT_SELECT_OFF",
        )
        select_op.override_index = index
        remove_op = row.operator(
            "mtp.remove_shape_override",
            text="",
            icon="X",
        )
        remove_op.override_index = index
        obj = bpy.data.objects.get(override.object_name)
        shape_keys = (
            obj.data.shape_keys
            if obj and obj.type == "MESH" and obj.data.shape_keys
            else None
        )
        target_row = override_box.row()
        target_row.enabled = override.enabled
        if shape_keys:
            target_row.prop_search(
                override,
                "target_name",
                shape_keys,
                "key_blocks",
                text="Shape Key",
            )
        else:
            target_row.prop(
                override,
                "target_name",
                text="Shape Key",
            )
        if override.message:
            if override.status == "ERROR":
                _draw_issue(
                    override_box,
                    override.message,
                    severity=ISSUE_WARNING,
                    boxed=False,
                )
            else:
                override_box.label(
                    text=override.message,
                    icon="INFO",
                )


def _draw_fk_switches(layout, armature):
    config = armature.mtp_rig_config
    box = layout.box()
    header = box.row(align=True)
    header.label(text="Limb Control Ownership", icon="CONSTRAINT_BONE")
    header.prop(
        config,
        "fk_switches_manual",
        text="",
        icon=("LOCKED" if config.fk_switches_manual else "UNLOCKED"),
    )
    for chain, side, label in (
        ("arm", "l", "Left Arm"),
        ("arm", "r", "Right Arm"),
        ("leg", "l", "Left Leg"),
        ("leg", "r", "Right Leg"),
    ):
        prefix = f"{chain}_fk_switch"
        bone_field = f"{prefix}_bone_{side}"
        property_field = f"{prefix}_property_{side}"
        value_field = f"{prefix}_value_{side}"
        row = box.row(align=True)
        row.label(text=label)
        row.prop_search(
            config,
            bone_field,
            armature.pose,
            "bones",
            text="",
        )
        row.prop(config, property_field, text="")
        row.prop(config, value_field, text="FK")
        row.prop(
            config,
            f"{chain}_ik_switch_value_{side}",
            text="IK",
        )
        ik_row = box.row(align=True)
        ik_row.label(text=f"{label} IK")
        ik_row.prop_search(
            config,
            f"{chain}_ik_target_bone_{side}",
            armature.pose,
            "bones",
            text="Target",
        )
        ik_row.prop_search(
            config,
            f"{chain}_ik_pole_bone_{side}",
            armature.pose,
            "bones",
            text="Pole",
        )
        if chain == "leg":
            toe_row = box.row(align=True)
            toe_row.label(text=f"{label} IK Toe")
            toe_row.prop_search(
                config,
                f"leg_ik_toe_bone_{side}",
                armature.pose,
                "bones",
                text="Toe",
            )
        bone_name = getattr(config, bone_field)
        property_name = getattr(config, property_field)
        bone = armature.pose.bones.get(bone_name)
        if bone_name and not bone:
            _draw_issue(
                box,
                f"{label}: switch bone is missing",
                severity=ISSUE_CRITICAL,
                boxed=False,
            )
        elif bone and property_name and property_name not in bone:
            _draw_issue(
                box,
                f"{label}: property '{property_name}' is missing",
                severity=ISSUE_CRITICAL,
                boxed=False,
            )


def _draw_compact_mapping_editor(layout, armature, entry):
    box = layout.box()
    header = box.row(align=True)
    header.label(text=entry.label, icon=STATUS_ICONS.get(entry.status, "QUESTION"))
    header.prop(entry, "enabled", text="")
    header.prop(
        entry,
        "manual_lock",
        text="",
        icon="LOCKED" if entry.manual_lock else "UNLOCKED",
    )
    if entry.section == "SHAPES":
        box.prop_search(entry, "target_object", bpy.data, "objects", text="Mesh")
        obj = bpy.data.objects.get(entry.target_object)
        keys = (
            obj.data.shape_keys
            if obj and obj.type == "MESH" and obj.data.shape_keys
            else None
        )
        if keys:
            box.prop_search(entry, "target_name", keys, "key_blocks", text="Shape Key")
        else:
            box.prop(entry, "target_name", text="Shape Key")
        row = box.row(align=True)
        row.operator("mtp.use_active_shape_key", icon="EYEDROPPER")
        row.operator("mtp.select_mapping_target", icon="RESTRICT_SELECT_OFF")
        row = box.row(align=True)
        row.prop(entry, "gain")
        row.prop(entry, "offset")
        _draw_shape_overrides(box, entry)
    else:
        if entry.section == "FACE_RIG":
            box.prop(entry, "target_kind")
        box.prop_search(entry, "target_name", armature.pose, "bones", text="Target")
        row = box.row(align=True)
        row.operator("mtp.use_selected_bone", icon="EYEDROPPER")
        row.operator("mtp.select_mapping_target", icon="RESTRICT_SELECT_OFF")
        if entry.section == "FACE_RIG":
            row = box.row(align=True)
            row.prop(entry, "channel_mode")
            row.prop(entry, "axis")
            if entry.channel_mode == "CUSTOM_PROPERTY":
                box.prop(entry, "property_name")
        box.prop(entry, "influence")
    box.prop(entry, "correction_euler")
    details = box.row(align=True)
    details.prop(entry, "source")
    details.prop(entry, "confidence")
    if entry.message:
        box.label(text=entry.message, icon=STATUS_ICONS.get(entry.status, "INFO"))


class MTP_PT_rig_setup(Panel):
    bl_idname = "MTP_PT_rig_setup"
    bl_label = "Rig Setup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Media to Pose"
    bl_order = 1

    def draw_header(self, context):
        armature = _configured_target(context)
        self.layout.label(text="", icon="ARMATURE_DATA" if armature else "ERROR")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtp_settings
        busy = COORDINATOR.is_busy()
        if busy:
            _draw_active_operation(layout, settings)
            layout.label(text="Mapping is locked while processing", icon="LOCKED")
        controls = layout.column()
        controls.enabled = not busy
        layout = controls
        armature = _configured_target(context)
        target = layout.row(align=True)
        target.prop(settings, "target_armature", text="Target Character")
        target.operator("mtp.use_selected_armature", text="Use Selected")
        if armature is None:
            _draw_issue(layout, "Choose a target character", severity=ISSUE_CRITICAL)
            return

        config = armature.mtp_rig_config
        state, icon, required, warnings = _mapping_readiness(config)
        header = layout.box()
        header.label(text=f"{armature.name} · {state}", icon=icon)
        completeness = header.row(align=True)
        completeness.label(text=f"Body {config.body_complete:.0%}")
        completeness.label(text=f"Hands {config.hands_complete:.0%}")
        if settings.face_output == "ARKIT":
            completeness.label(text=f"ARKit {config.arkit_complete:.0%}")
        elif settings.face_output == "FACS":
            completeness.label(text=f"FACS {config.facs_complete:.0%}")
        else:
            completeness.label(text=f"Face {config.face_complete:.0%}")
        if required:
            header.label(text=f"{required} required mapping issue(s)", icon="CANCEL")
        elif warnings:
            header.label(text=f"{warnings} mapping suggestion(s)", icon="ERROR")

        methods = layout.box()
        methods.label(text="Automatic Mapping")
        methods.label(text="Mapping Method")
        grid = methods.grid_flow(columns=2, even_columns=True, align=True)
        for value, label in (
            ("AUTO", "Detect Automatically"),
            ("AUTO_RIG_PRO", "Auto-Rig Pro"),
            ("RIGIFY", "Rigify"),
            ("GENERIC", "Generic / Custom"),
        ):
            operator = grid.operator(
                "mtp.set_mapping_method",
                text=label,
                depress=config.profile == value,
            )
            operator.method = value
        expected = (
            detect_profile(armature) if config.profile == "AUTO" else config.profile
        )
        pending = bool(config.mappings and config.detected_profile != expected)
        methods.label(
            text=(
                f"Detected: {profile_label(config)}"
                if config.profile == "AUTO"
                else f"Selected: {PROFILE_LABELS.get(expected, expected)}"
            ),
            icon="INFO",
        )
        action_label = (
            "Set Up Rig"
            if not config.mappings
            else "Apply Mapping Method"
            if pending
            else "Update Automatic Mappings"
        )
        methods.operator("mtp.auto_map", text=action_label, icon="MOD_ARMATURE")
        methods.label(
            text="Manual locks are preserved when automatic mappings update.",
            icon="LOCKED",
        )

        if not config.mappings:
            return
        layout.prop(config, "subtab", expand=True)
        if config.subtab == "FACE_RIG":
            face = layout.box()
            face.prop(settings, "face_output", text="Face Output")
            if not settings.use_face:
                face.label(text="Face capture is currently disabled", icon="INFO")

        problems = _mapping_problem_entries(config)
        diagnostics = layout.box()
        if problems:
            diagnostics.label(
                text=f"Needs attention · {len(problems)} problem(s)", icon="ERROR"
            )
            problem_ids = {entry.as_pointer() for entry in problems}
            shown = 0
            for index, entry in enumerate(config.mappings):
                if entry.as_pointer() not in problem_ids:
                    continue
                row = diagnostics.row(align=True)
                row.label(
                    text=entry.label, icon=STATUS_ICONS.get(entry.status, "ERROR")
                )
                row.label(text=entry.message or entry.status.title())
                operator = row.operator(
                    "mtp.select_problem", text="", icon="RESTRICT_SELECT_OFF"
                )
                operator.mapping_index = index
                shown += 1
                if shown >= 3:
                    break
            if len(problems) > 3:
                diagnostics.label(text=f"{len(problems) - 3} additional problem(s)")
        else:
            diagnostics.label(text="Mapping ready", icon="CHECKMARK")

        filters = layout.row(align=True)
        filters.prop(config, "filter_text", text="", icon="VIEWZOOM")
        filters.prop(config, "filter_mode", text="")
        layout.template_list(
            "MTP_UL_mappings",
            f"{config.subtab}_{settings.face_output}",
            config,
            "mappings",
            config,
            "active_mapping_index",
            rows=7,
        )
        entry = active_entry(armature)
        if entry and entry.section == active_mapping_section(context, armature):
            _draw_compact_mapping_editor(layout, armature, entry)
        if config.subtab == "HANDS":
            mirror = layout.row(align=True)
            operator = mirror.operator("mtp.mirror_mapping", text="Left to Right")
            operator.direction = "L_TO_R"
            operator = mirror.operator("mtp.mirror_mapping", text="Right to Left")
            operator.direction = "R_TO_L"
        if config.subtab == "BODY" and _draw_left_disclosure(
            layout,
            config,
            "show_rig_controls",
            "FK/IK Controls",
        ):
            _draw_fk_switches(layout, armature)
        footer = layout.row(align=True)
        footer.menu(
            "MTP_MT_mapping_presets", text="Mapping Presets", icon="DOWNARROW_HLT"
        )
        footer.menu(
            "MTP_MT_mapping_reset", text="Reset or Replace…", icon="RECOVER_LAST"
        )
