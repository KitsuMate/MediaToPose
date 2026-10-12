"""Per-armature RNA settings: role mapping, shape overrides and cleanup modules."""

from __future__ import annotations

from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from ...core.constants import MAPPING_FILTERS, MAPPING_TABS, PROFILE_AUTO, SECTIONS
from ...postprocess import settings_schema
from .callbacks import _fk_switch_edited, _mapping_edited, _shape_override_edited


class MTPShapeOverride(PropertyGroup):
    semantic_id: StringProperty(options={"HIDDEN"})
    object_name: StringProperty(name="Mesh", update=_shape_override_edited)
    target_name: StringProperty(name="Shape Key", update=_shape_override_edited)
    enabled: BoolProperty(
        name="Enabled",
        default=True,
        update=_shape_override_edited,
    )
    status: EnumProperty(
        name="Status",
        items=(
            ("VALID", "Valid", ""),
            ("ERROR", "Error", ""),
            ("OPTIONAL", "Skipped", ""),
        ),
        default="VALID",
    )
    message: StringProperty(name="Message")


class MTPMappingEntry(PropertyGroup):
    semantic_id: StringProperty(name="Semantic ID")
    label: StringProperty(name="Label")
    section: EnumProperty(name="Section", items=SECTIONS, default="BODY")
    side: EnumProperty(
        name="Side",
        items=(("CENTER", "Center", ""), ("LEFT", "Left", ""), ("RIGHT", "Right", "")),
        default="CENTER",
    )
    required: BoolProperty(name="Required", default=False)
    enabled: BoolProperty(name="Enabled", default=True)
    target_kind: EnumProperty(
        name="Target Type",
        items=(
            ("BONE", "Bone", "Pose bone"),
            ("SHAPE_KEY", "Shape Key", "Mesh shape key"),
            ("CUSTOM_PROPERTY", "Custom Property", "Bone custom property"),
        ),
        default="BONE",
        update=_mapping_edited,
    )
    shape_scheme: EnumProperty(
        name="Morph Standard",
        items=(
            ("NONE", "None", "Not a morph mapping"),
            ("ARKIT", "ARKit", "Apple ARKit-compatible blendshape"),
            ("FACS", "FACS", "FACS-compatible morph"),
            ("VRM", "VRM", "VRM / VRoid expression morph"),
        ),
        default="NONE",
    )
    target_name: StringProperty(name="Target", update=_mapping_edited)
    target_object: StringProperty(name="Object", update=_mapping_edited)
    target_data_name: StringProperty(name="Data", update=_mapping_edited)
    property_name: StringProperty(name="Property", update=_mapping_edited)
    channel_mode: EnumProperty(
        name="Channel",
        items=(
            ("AUTO", "Automatic", "Choose from profile or target type"),
            ("ROTATION", "Rotation", "Rotate the target bone"),
            ("LOCATION", "Location", "Translate the target bone"),
            ("CUSTOM_PROPERTY", "Custom Property", "Set a custom property"),
            ("VALUE", "Value", "Set a shape-key value"),
        ),
        default="AUTO",
        update=_mapping_edited,
    )
    axis: EnumProperty(
        name="Axis",
        items=(
            ("X", "X", ""),
            ("Y", "Y", ""),
            ("Z", "Z", ""),
            ("-X", "-X", ""),
            ("-Y", "-Y", ""),
            ("-Z", "-Z", ""),
        ),
        default="X",
        update=_mapping_edited,
    )
    gain: FloatProperty(name="Gain", default=1.0, soft_min=-4.0, soft_max=4.0)
    offset: FloatProperty(
        name="Neutral Offset", default=0.0, soft_min=-1.0, soft_max=1.0
    )
    clamp_min: FloatProperty(name="Minimum", default=0.0)
    clamp_max: FloatProperty(name="Maximum", default=1.0)
    invert: BoolProperty(name="Invert", default=False)
    influence: FloatProperty(name="Influence", default=1.0, min=0.0, max=1.0)
    correction_euler: FloatVectorProperty(
        name="Correction",
        subtype="EULER",
        size=3,
        default=(0.0, 0.0, 0.0),
    )
    confidence: FloatProperty(name="Confidence", default=0.0, min=0.0, max=1.0)
    source: EnumProperty(
        name="Source",
        items=(
            ("PROFILE", "Profile", "Exact rig profile"),
            ("HEURISTIC", "Heuristic", "Automatically guessed"),
            ("MANUAL", "Manual", "User-defined"),
            ("NONE", "None", "Unmapped"),
        ),
        default="NONE",
    )
    status: EnumProperty(
        name="Status",
        items=(
            ("VALID", "Valid", ""),
            ("INFO", "Information", ""),
            ("WARNING", "Warning", ""),
            ("ERROR", "Error", ""),
            ("MISSING", "Missing", ""),
            ("OPTIONAL", "Optional", ""),
        ),
        default="MISSING",
    )
    message: StringProperty(name="Message")
    manual_lock: BoolProperty(
        name="Keep Manual Assignment",
        description="Keep this assignment when automatic mappings are updated",
        default=False,
    )
    show_shape_overrides: BoolProperty(name="Mesh Overrides", default=False)
    shape_overrides: CollectionProperty(type=MTPShapeOverride)
    active_shape_override_index: IntProperty(default=0)


class MTPPostProcessingModuleConfig(PropertyGroup):
    """Persistent typed settings for one pose-correction module."""

    module_id: StringProperty(name="Module ID", options={"HIDDEN"})
    display_name: StringProperty(name="Name")
    description: StringProperty(name="Description")
    enabled: BoolProperty(
        name="Enable",
        description=(
            "Apply this correction automatically with Apply Pose and Bake Keyframes"
        ),
        default=True,
    )
    status: EnumProperty(
        name="Status",
        items=(
            ("VALID", "Valid", ""),
            ("WARNING", "Warning", ""),
            ("ERROR", "Error", ""),
        ),
        default="VALID",
    )
    last_warning: StringProperty(name="Warning")
    # Diagnostics written by the last video bake. Module settings themselves
    # are generated from postprocess.settings_schema below the class.
    camera_track_status: StringProperty(
        name="Camera Track",
        default="Not Prepared",
        options={"HIDDEN"},
    )
    camera_tracked_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    camera_rejected_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    camera_median_inliers: FloatProperty(default=0.0, min=0.0, options={"HIDDEN"})
    camera_corrected_intervals: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_rejected_observations: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_rejected_limbs: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_rejected_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_native_missing: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_total_observations: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_manual_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_frame_summary: StringProperty(default="", options={"HIDDEN"})
    show_advanced: BoolProperty(name="Advanced", default=False)


def _schema_property(parameter: settings_schema.Parameter):
    """Build the RNA property for one schema parameter."""
    common = {"name": parameter.label, "description": parameter.description}
    if parameter.kind == "bool":
        return BoolProperty(default=bool(parameter.default), **common)
    if parameter.kind == "enum":
        return EnumProperty(items=parameter.items, default=parameter.default, **common)
    if parameter.kind == "string":
        return StringProperty(default=str(parameter.default), **common)
    numeric = {"default": parameter.default}
    if parameter.minimum is not None:
        numeric["min"] = parameter.minimum
    if parameter.maximum is not None:
        numeric["max"] = parameter.maximum
    if parameter.subtype:
        numeric["subtype"] = parameter.subtype
    if parameter.kind == "int":
        return IntProperty(**numeric, **common)
    return FloatProperty(**numeric, **common)


for _parameter in settings_schema.PARAMETERS.values():
    MTPPostProcessingModuleConfig.__annotations__[_parameter.name] = _schema_property(
        _parameter
    )


class MTPRigConfig(PropertyGroup):
    profile: EnumProperty(
        name="Mapping Method",
        items=(
            (PROFILE_AUTO, "Detect Automatically", "Detect the rig family"),
            ("AUTO_RIG_PRO", "Auto-Rig Pro", "Use Auto-Rig Pro controls"),
            ("RIGIFY", "Rigify", "Use Rigify controls"),
            (
                "GENERIC",
                "Generic / Custom",
                "Use humanoid metadata and generic bone-name mapping",
            ),
        ),
        default=PROFILE_AUTO,
    )
    detected_profile: StringProperty(
        name="Detected Profile",
        default=PROFILE_AUTO,
    )
    armature_signature: StringProperty(name="Armature Signature")
    mappings: CollectionProperty(type=MTPMappingEntry)
    post_processing_modules: CollectionProperty(type=MTPPostProcessingModuleConfig)
    active_mapping_index: IntProperty(default=0)
    subtab: EnumProperty(
        name="Mapping Section",
        items=MAPPING_TABS,
        default="BODY",
    )
    filter_mode: EnumProperty(name="Show", items=MAPPING_FILTERS, default="ALL")
    filter_text: StringProperty(name="Search")
    show_rig_controls: BoolProperty(name="Rig Controls", default=False)
    validation_summary: StringProperty(name="Validation", default="Not validated")
    body_complete: FloatProperty(default=0.0, min=0.0, max=1.0)
    hands_complete: FloatProperty(default=0.0, min=0.0, max=1.0)
    shapes_complete: FloatProperty(default=0.0, min=0.0, max=1.0)
    arkit_complete: FloatProperty(default=0.0, min=0.0, max=1.0)
    facs_complete: FloatProperty(default=0.0, min=0.0, max=1.0)
    vrm_complete: FloatProperty(default=0.0, min=0.0, max=1.0)
    face_complete: FloatProperty(default=0.0, min=0.0, max=1.0)
    arm_fk_switch_bone_l: StringProperty(
        name="Left Arm Switch Bone", update=_fk_switch_edited
    )
    arm_fk_switch_property_l: StringProperty(
        name="Left Arm Property", update=_fk_switch_edited
    )
    arm_fk_switch_value_l: FloatProperty(
        name="Left Arm FK Value", default=1.0, update=_fk_switch_edited
    )
    arm_ik_switch_value_l: FloatProperty(
        name="Left Arm IK Value", default=0.0, update=_fk_switch_edited
    )
    arm_ik_target_bone_l: StringProperty(
        name="Left Arm IK Target", update=_fk_switch_edited
    )
    arm_ik_pole_bone_l: StringProperty(
        name="Left Arm IK Pole", update=_fk_switch_edited
    )
    arm_fk_switch_bone_r: StringProperty(
        name="Right Arm Switch Bone", update=_fk_switch_edited
    )
    arm_fk_switch_property_r: StringProperty(
        name="Right Arm Property", update=_fk_switch_edited
    )
    arm_fk_switch_value_r: FloatProperty(
        name="Right Arm FK Value", default=1.0, update=_fk_switch_edited
    )
    arm_ik_switch_value_r: FloatProperty(
        name="Right Arm IK Value", default=0.0, update=_fk_switch_edited
    )
    arm_ik_target_bone_r: StringProperty(
        name="Right Arm IK Target", update=_fk_switch_edited
    )
    arm_ik_pole_bone_r: StringProperty(
        name="Right Arm IK Pole", update=_fk_switch_edited
    )
    leg_fk_switch_bone_l: StringProperty(
        name="Left Leg Switch Bone", update=_fk_switch_edited
    )
    leg_fk_switch_property_l: StringProperty(
        name="Left Leg Property", update=_fk_switch_edited
    )
    leg_fk_switch_value_l: FloatProperty(
        name="Left Leg FK Value", default=1.0, update=_fk_switch_edited
    )
    leg_ik_switch_value_l: FloatProperty(
        name="Left Leg IK Value", default=0.0, update=_fk_switch_edited
    )
    leg_ik_target_bone_l: StringProperty(
        name="Left Leg IK Target", update=_fk_switch_edited
    )
    leg_ik_pole_bone_l: StringProperty(
        name="Left Leg IK Pole", update=_fk_switch_edited
    )
    leg_ik_toe_bone_l: StringProperty(name="Left Leg IK Toe", update=_fk_switch_edited)
    leg_fk_switch_bone_r: StringProperty(
        name="Right Leg Switch Bone", update=_fk_switch_edited
    )
    leg_fk_switch_property_r: StringProperty(
        name="Right Leg Property", update=_fk_switch_edited
    )
    leg_fk_switch_value_r: FloatProperty(
        name="Right Leg FK Value", default=1.0, update=_fk_switch_edited
    )
    leg_ik_switch_value_r: FloatProperty(
        name="Right Leg IK Value", default=0.0, update=_fk_switch_edited
    )
    leg_ik_target_bone_r: StringProperty(
        name="Right Leg IK Target", update=_fk_switch_edited
    )
    leg_ik_pole_bone_r: StringProperty(
        name="Right Leg IK Pole", update=_fk_switch_edited
    )
    leg_ik_toe_bone_r: StringProperty(name="Right Leg IK Toe", update=_fk_switch_edited)
    fk_switches_manual: BoolProperty(name="Lock FK Switch Configuration", default=False)
