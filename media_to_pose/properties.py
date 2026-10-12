"""Persistent Blender properties."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from .constants import (
    MAPPING_FILTERS,
    MAPPING_TABS,
    PROFILE_AUTO,
    SECTIONS,
)
from .operation import COORDINATOR
from .registration import register_classes, unregister_classes

_MANUAL_LOCK_SUPPRESSED = ContextVar(
    "media_to_pose_manual_lock_suppressed",
    default=False,
)


@contextmanager
def suppress_manual_locks():
    """Prevent programmatic edits from becoming user-locked mappings."""
    token = _MANUAL_LOCK_SUPPRESSED.set(True)
    try:
        yield
    finally:
        _MANUAL_LOCK_SUPPRESSED.reset(token)


def _mapping_edited(self, context) -> None:
    if _MANUAL_LOCK_SUPPRESSED.get():
        return
    self.manual_lock = True
    self.source = "MANUAL"
    self.confidence = 1.0
    self.status = "VALID" if self.target_name else "MISSING"
    settings = getattr(getattr(context, "scene", None), "mtp_settings", None)
    armature = getattr(settings, "target_armature", None) if settings else None
    if armature is not None:
        from . import mapping

        mapping.validate_mapping(armature, settings=settings)
        from .post_processing import validate_modules

        validate_modules(armature)


def _fk_switch_edited(self, context) -> None:
    if not _MANUAL_LOCK_SUPPRESSED.get():
        self.fk_switches_manual = True
        settings = getattr(getattr(context, "scene", None), "mtp_settings", None)
        armature = getattr(settings, "target_armature", None) if settings else None
        if armature is not None:
            from . import mapping
            from .post_processing import validate_modules

            mapping.validate_mapping(armature, settings=settings)
            validate_modules(armature)


def _shape_override_edited(self, context) -> None:
    if _MANUAL_LOCK_SUPPRESSED.get() or not context or not self.semantic_id:
        return
    settings = getattr(context.scene, "mtp_settings", None)
    armature = settings.target_armature if settings else None
    if not armature or armature.type != "ARMATURE":
        return
    entry = next(
        (
            candidate
            for candidate in armature.mtp_rig_config.mappings
            if candidate.semantic_id == self.semantic_id
        ),
        None,
    )
    if not entry:
        return
    entry.manual_lock = True
    entry.source = "MANUAL"
    entry.confidence = 1.0
    from . import mapping

    mapping.validate_mapping(armature)
    from .post_processing import validate_modules

    validate_modules(armature)


def _armature_poll(_self, obj) -> bool:
    return bool(obj and obj.type == "ARMATURE")


def _target_armature_changed(self, _context) -> None:
    armature = getattr(self, "target_armature", None)
    if armature is None:
        return
    try:
        from . import mapping

        mapping.refresh_auto_mapping(armature)
    except (AttributeError, ReferenceError, RuntimeError):
        pass


def _cache_enabled_edited(self, _context) -> None:
    if self.keep_detection_cache or COORDINATOR.is_busy():
        return
    # Property callbacks run on Blender's main thread. File removal is kept
    # out of panel draw callbacks, which Blender treats as read-only.
    from . import cache_preview, detection_cache

    detection_cache.clear_cache()
    cache_preview.refresh_after_cache_update()


def _preview_edited(_self, _context) -> None:
    from . import cache_preview

    cache_preview.schedule_refresh()


def _camera_settings_edited(self, context) -> None:
    self.last_camera_vertical_fov = 0.0
    self.last_camera_source = ""
    _preview_edited(self, context)


def _get_perspective_correction_enabled(self) -> bool:
    return self.perspective_correction == "AUTO"


def _set_perspective_correction_enabled(self, enabled: bool) -> None:
    self.perspective_correction = "AUTO" if enabled else "OFF"


def _capture_scope_edited(self, context) -> None:
    _preview_edited(self, context)
    armature = getattr(self, "target_armature", None)
    if armature is None:
        return
    from . import mapping

    mapping.validate_mapping(armature, settings=self)


def _media_path_edited(self, _context) -> None:
    from . import cache_preview, detection_cache

    self.video_frame = 1
    self.last_camera_vertical_fov = 0.0
    self.last_camera_source = ""
    resolved = detection_cache.existing_media_path(self.media_path)
    if resolved:
        try:
            detection_cache.record_media_selection(
                self,
                resolved,
                assign_path=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.media_identity_json = ""
            self.status = f"Could not record source media: {exc}"
            self.status_kind = "ERROR"
    else:
        # A newly typed missing path must not retain the identity of the
        # previously selected source. Existing saved paths keep their identity
        # until the user actually edits this property.
        self.media_identity_json = ""
        if self.media_path:
            self.media_path = ""
    cache_preview.schedule_refresh()


def _provider_edited(self, context) -> None:
    _capture_scope_edited(self, context)


def _sam3d_model_edited(_self, _context) -> None:
    from . import cache_preview

    cache_preview.schedule_refresh()


_SAM3D_VITH_ITEM = ("VITH", "ViT-H", "Default immutable ONNX graph bundle")


def _sam3d_device_items(_self, _context):
    # Keep this callback read-only so panel draw never probes hardware.
    try:
        from .backends.sam3d import backend as sam3d_backend

        return sam3d_backend.sam3d_device_enum_items()
    except (AttributeError, ImportError, RuntimeError):
        return (
            (
                "AUTO",
                "Auto",
                "Prefer WebGPU/CoreML acceleration, then fall back to CPU",
            ),
            ("GPU", "GPU", "Use the platform GPU execution provider"),
            ("CPU", "CPU", "Use ONNX Runtime CPU inference"),
        )


def _rohm_device_items(_self, _context):
    """Return fixed provider-neutral choices without probing hardware."""
    try:
        from .backends.rohm import backend as rohm_backend

        return rohm_backend.rohm_device_enum_items()
    except (AttributeError, ImportError, RuntimeError):
        return (
            ("AUTO", "Auto", "Prefer GPU acceleration, then fall back to CPU"),
            ("GPU", "GPU", "Require the platform GPU execution provider"),
            ("CPU", "CPU", "Use ONNX Runtime CPU inference"),
        )


def _anycalib_device_items(_self, _context):
    return (
        ("AUTO", "Auto", "Prefer GPU acceleration, then fall back to CPU"),
        ("GPU", "GPU", "Require the platform GPU execution provider"),
        ("CPU", "CPU", "Use ONNX Runtime CPU inference"),
    )


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
    # Shoulder Correction.
    strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=2.0)
    start_elevation: FloatProperty(
        name="Start Elevation",
        subtype="ANGLE",
        default=0.5235987756,
        min=0.0,
        max=1.5707963268,
    )
    full_influence_elevation: FloatProperty(
        name="Full Influence",
        subtype="ANGLE",
        default=1.5707963268,
        min=0.0,
        max=1.5707963268,
    )
    maximum_swing: FloatProperty(
        name="Maximum Swing",
        subtype="ANGLE",
        default=0.6108652382,
        min=0.0,
        max=1.5707963268,
    )
    maximum_twist: FloatProperty(
        name="Maximum Twist",
        subtype="ANGLE",
        default=0.1745329252,
        min=0.0,
        max=1.5707963268,
    )
    repair_arm_silhouette: BoolProperty(
        name="Repair Arm Silhouette",
        description=(
            "Use SAM image-space joints to gently repair an arm that is "
            "substantially more bent than its detected silhouette"
        ),
        default=True,
    )
    maximum_arm_extension: FloatProperty(
        name="Maximum IK Shift",
        description=(
            "Maximum wrist or IK-target displacement as a fraction of the "
            "complete arm length"
        ),
        subtype="PERCENTAGE",
        default=0.02,
        min=0.0,
        max=0.05,
    )

    # Arm and Leg Axis Correction shared settings.
    hinge_projection: BoolProperty(name="Hinge Projection", default=True)
    confidence_threshold: FloatProperty(
        name="Axis Confidence Threshold",
        description=(
            "Minimum confidence for upper-limb or thigh axis reconstruction; "
            "palm roll is continuous and uses confidence only as a temporal weight"
        ),
        default=0.75,
        min=0.0,
        max=1.0,
    )
    upper_arm_twist_limit: FloatProperty(
        name="Upper Arm Twist",
        subtype="ANGLE",
        default=1.308996939,
        min=0.0,
        max=3.1415926536,
    )
    forearm_twist_limit: FloatProperty(
        name="Maximum Forearm Correction",
        description=(
            "Maximum palm-derived correction applied around the forearm; "
            "this caps the result and never disables correction"
        ),
        subtype="ANGLE",
        default=1.5707963268,
        min=0.0,
        max=3.1415926536,
    )
    palm_roll_strength: FloatProperty(
        name="Palm Roll Influence",
        description=(
            "How strongly detected palm roll drives wrist and forearm twist; "
            "reduce for occluded or low-detail hands"
        ),
        default=0.65,
        min=0.0,
        max=1.0,
    )
    thigh_twist_limit: FloatProperty(
        name="Thigh Twist",
        subtype="ANGLE",
        default=1.0471975512,
        min=0.0,
        max=3.1415926536,
    )
    # Camera Motion Correction (video sequence only).
    lateral_root_strength: FloatProperty(
        name="Lateral Root Strength",
        description=(
            "Blend the original lateral root trajectory with background-camera "
            "compensation"
        ),
        default=1.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    camera_track_status: StringProperty(
        name="Camera Track",
        default="Not Prepared",
        options={"HIDDEN"},
    )
    camera_tracked_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    camera_rejected_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    camera_median_inliers: FloatProperty(default=0.0, min=0.0, options={"HIDDEN"})
    camera_corrected_intervals: IntProperty(default=0, min=0, options={"HIDDEN"})
    # Detection Filter (video sequence only).
    reject_unreliable_limbs: BoolProperty(
        name="Reject Unreliable Limbs",
        description=(
            "Mask low-confidence limbs and isolated motion jumps before motion repair"
        ),
        default=True,
    )
    reject_unusable_frames: BoolProperty(
        name="Reject Unusable Frames",
        description=(
            "Mask whole frames when their body evidence is broadly unreliable"
        ),
        default=True,
    )
    maximum_motion_rejected: FloatProperty(
        name="Maximum Motion Rejected",
        description=(
            "Maximum share of otherwise available joint observations that this "
            "filter may reject; detector-native missing data does not count"
        ),
        default=30.0,
        min=0.0,
        max=100.0,
        subtype="PERCENTAGE",
    )
    manual_bad_frames: StringProperty(
        name="Additional Bad Source Frames",
        description=(
            "Optional 1-based source frames or inclusive ranges, for example "
            "12, 20-24; these explicit masks are not limited by the automatic "
            "rejection budget"
        ),
        default="",
    )
    manual_bad_scope: EnumProperty(
        name="Manual Mask Scope",
        description="Body region masked on every additional bad source frame",
        items=(
            ("BODY", "Whole Body", "Mask the complete body and root observation"),
            ("ARM_L", "Left Arm", "Mask the left shoulder and arm"),
            ("ARM_R", "Right Arm", "Mask the right shoulder and arm"),
            ("LEG_L", "Left Leg", "Mask the left leg"),
            ("LEG_R", "Right Leg", "Mask the right leg"),
        ),
        default="BODY",
    )
    filter_rejected_observations: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_rejected_limbs: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_rejected_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_native_missing: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_total_observations: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_manual_frames: IntProperty(default=0, min=0, options={"HIDDEN"})
    filter_frame_summary: StringProperty(default="", options={"HIDDEN"})
    # Temporal Correction (video sequence only).
    smooth_motion: BoolProperty(
        name="Smooth Motion",
        description="Apply symmetric, confidence-aware smoothing without filter lag",
        default=True,
    )
    ground_contacts: BoolProperty(
        name="Ground Contacts",
        description="Stabilize root travel and planted foot positions without changing foot orientation",
        default=True,
    )
    smoothing_strength: FloatProperty(
        name="Smoothing Strength",
        description="Blend between detected and stabilized video animation",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    stabilize_hands: BoolProperty(
        name="Stabilize Hands",
        description=(
            "Reject abrupt wrist and finger rotation bursts, then enforce a "
            "bounded angular speed after smoothing"
        ),
        default=True,
    )
    finger_max_angular_speed: FloatProperty(
        name="Finger Maximum Speed",
        description=(
            "Largest accepted finger rotation speed; faster detector jumps are "
            "held until the pose becomes stable"
        ),
        subtype="ANGLE",
        default=10.471975512,
        min=1.0471975512,
        max=25.132741229,
    )
    wrist_max_angular_speed: FloatProperty(
        name="Wrist Maximum Speed",
        description=(
            "Largest accepted wrist rotation speed; faster detector jumps are "
            "held until the pose becomes stable"
        ),
        subtype="ANGLE",
        default=12.566370614,
        min=1.0471975512,
        max=25.132741229,
    )
    hand_reacquire_delay: FloatProperty(
        name="Hand Reacquire Delay",
        description=(
            "How long a substantially different hand pose must remain stable "
            "before a rejected jump is accepted"
        ),
        subtype="TIME",
        default=0.3,
        min=0.1,
        max=10.0,
    )
    # RoHM Correction (video sequence only).
    visibility_threshold: FloatProperty(
        name="Visibility Threshold",
        description="Confidence below this value is treated as occluded",
        default=0.50,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    learned_pose_residual: BoolProperty(
        name="AI Pose Cleanup (Alpha)",
        description=(
            "Alpha feature: use AI to adjust only masked anatomical-global joints, "
            "with each change limited to two degrees from deterministic repair"
        ),
        default=False,
    )
    root_trajectory_cleanup: BoolProperty(
        name="Root Trajectory Cleanup",
        description=(
            "Use RoHM only for masked root observations while Automatic Root "
            "Motion is active"
        ),
        default=True,
    )
    show_advanced: BoolProperty(name="Advanced", default=False)


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


class MTPSceneSettings(PropertyGroup):
    media_path: StringProperty(
        name="Media",
        subtype="FILE_PATH",
        update=_media_path_edited,
    )
    media_identity_json: StringProperty(
        name="Media Source Identity",
        description="Exact source identity used to match retained detections",
        options={"HIDDEN"},
    )
    provider: EnumProperty(
        name="Pose Detection",
        items=(
            (
                "MEDIAPIPE",
                "MediaPipe",
                "Fast local detection for photographs",
            ),
            (
                "SAM3D_BODY",
                "SAM 3D Body",
                "Detailed body and hand detection for photographs and stylized art",
            ),
        ),
        default="MEDIAPIPE",
        update=_provider_edited,
    )
    sam3d_model_variant: EnumProperty(
        name="Model",
        items=(_SAM3D_VITH_ITEM,),
        default="VITH",
        update=_sam3d_model_edited,
    )
    sam3d_device: EnumProperty(
        name="Device",
        items=_sam3d_device_items,
    )
    rohm_device: EnumProperty(
        name="Device",
        description="ONNX Runtime device used by RoHM Correction",
        items=_rohm_device_items,
    )
    perspective_correction: EnumProperty(
        name="Perspective Correction",
        description="Calibrate camera intrinsics before pose inference",
        items=(
            (
                "AUTO",
                "Auto",
                "Use AnyCalib when installed, otherwise use a safe fallback",
            ),
            ("OFF", "Off", "Use the provider's fallback camera estimate"),
        ),
        default="AUTO",
        update=_camera_settings_edited,
    )
    perspective_correction_enabled: BoolProperty(
        name="Perspective Correction",
        description="Estimate the source camera lens before pose inference",
        get=_get_perspective_correction_enabled,
        set=_set_perspective_correction_enabled,
    )
    manual_fov_override: BoolProperty(
        name="Manual FOV Override",
        description="Use a known vertical field of view instead of automatic calibration",
        default=False,
        update=_camera_settings_edited,
    )
    manual_vertical_fov: FloatProperty(
        name="Vertical FOV",
        description="Known vertical camera field of view",
        default=45.0,
        min=1.0,
        max=179.0,
        update=_camera_settings_edited,
    )
    anycalib_device: EnumProperty(
        name="Calibration Device",
        items=_anycalib_device_items,
        options={"HIDDEN"},
    )
    last_camera_vertical_fov: FloatProperty(
        name="Detected Vertical FOV",
        default=0.0,
        options={"SKIP_SAVE"},
    )
    last_camera_source: StringProperty(
        name="Camera Calibration Source",
        default="",
        options={"SKIP_SAVE"},
    )
    use_body: BoolProperty(name="Body", default=True, update=_capture_scope_edited)
    use_hands: BoolProperty(name="Hands", default=False, update=_capture_scope_edited)
    use_face: BoolProperty(name="Face", default=False, update=_capture_scope_edited)
    face_output: EnumProperty(
        name="Face Output",
        items=(
            (
                "ARKIT",
                "ARKit Morphs (Experimental)",
                "Experimental: drive only ARKit-compatible shape keys",
            ),
            (
                "FACS",
                "FACS Morphs (Experimental)",
                "Experimental: drive only FACS-compatible shape keys",
            ),
            (
                "BONES",
                "Facial Bones (Experimental)",
                "Experimental: drive only mapped facial controls",
            ),
        ),
        default="ARKIT",
        update=_preview_edited,
    )
    mirror_input: BoolProperty(
        name="Mirror Input", default=False, update=_preview_edited
    )
    video_frame: IntProperty(
        name="Frame",
        description="One-based source video frame used by the viewer and Apply Pose",
        default=1,
        min=1,
        update=_preview_edited,
    )
    smoothing: FloatProperty(
        name="Smoothing",
        default=0.35,
        min=0.0,
        max=0.95,
        update=_preview_edited,
    )
    root_motion_mode: EnumProperty(
        name="Character Movement",
        description="How video baking handles the character root",
        items=(
            (
                "AUTOMATIC",
                "From Video",
                "Use detected video travel and yaw when available, otherwise bake in place",
            ),
            ("OFF", "Keep Existing", "Preserve existing root animation"),
            ("IN_PLACE", "In Place", "Replace the baked interval with a fixed root"),
        ),
        default="AUTOMATIC",
    )
    keep_original_root_position: BoolProperty(
        name="Keep Original Root Position",
        description=(
            "Preserve the target's current root location; when disabled, image "
            "poses and the first frame of video motion start at 0, 0, 0"
        ),
        default=False,
    )
    target_armature: PointerProperty(
        name="Target",
        type=bpy.types.Object,
        poll=_armature_poll,
        update=_target_armature_changed,
    )
    status: StringProperty(
        name="Status",
        default="Ready",
        options={"SKIP_SAVE"},
    )
    status_kind: EnumProperty(
        name="Status Type",
        items=(
            ("READY", "Ready", "No active message"),
            ("PROGRESS", "Progress", "An operation is running"),
            ("INFO", "Information", "The last operation completed"),
            ("WARNING", "Warning", "The last operation needs attention"),
            ("ERROR", "Error", "The last operation failed"),
        ),
        default="READY",
        options={"SKIP_SAVE"},
    )
    show_result_details: BoolProperty(
        name="Result Details",
        default=False,
        options={"SKIP_SAVE"},
    )
    progress: FloatProperty(
        name="Progress",
        default=0.0,
        min=0.0,
        max=100.0,
        subtype="PERCENTAGE",
        options={"SKIP_SAVE"},
    )
    show_detection_advanced: BoolProperty(name="Capture Options", default=False)
    show_perspective_settings: BoolProperty(
        name="Perspective Correction Settings",
        default=False,
    )
    show_sam3d_management: BoolProperty(
        name="SAM 3D Body Management",
        default=False,
    )
    show_anycalib_management: BoolProperty(
        name="Camera Model Management",
        default=False,
    )
    show_rohm_management: BoolProperty(
        name="Motion Model Management",
        default=False,
    )
    keep_detection_cache: BoolProperty(
        name="Keep Detection Cache",
        description=(
            "Retain the latest provider-neutral detection outside the .blend "
            "for reuse and media preview"
        ),
        default=True,
        update=_cache_enabled_edited,
    )
    cache_limit_mb: IntProperty(
        name="Cache Limit",
        description="Maximum retained detection cache size in MiB",
        default=512,
        min=128,
        max=4096,
        subtype="UNSIGNED",
    )


CLASSES = (
    MTPShapeOverride,
    MTPMappingEntry,
    MTPPostProcessingModuleConfig,
    MTPRigConfig,
    MTPSceneSettings,
)


def register() -> None:
    unregister()
    register_classes(CLASSES)
    try:
        bpy.types.Object.mtp_rig_config = PointerProperty(type=MTPRigConfig)
        bpy.types.Scene.mtp_settings = PointerProperty(type=MTPSceneSettings)
    except Exception:
        unregister()
        raise


def unregister() -> None:
    if hasattr(bpy.types.Scene, "mtp_settings"):
        del bpy.types.Scene.mtp_settings
    if hasattr(bpy.types.Object, "mtp_rig_config"):
        del bpy.types.Object.mtp_rig_config
    unregister_classes(CLASSES)
