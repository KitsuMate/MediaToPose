"""Per-scene capture settings."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from .callbacks import (
    _SAM3D_VITH_ITEM,
    _anycalib_device_items,
    _armature_poll,
    _cache_enabled_edited,
    _camera_settings_edited,
    _capture_scope_edited,
    _get_perspective_correction_enabled,
    _media_path_edited,
    _preview_edited,
    _provider_edited,
    _rohm_device_items,
    _sam3d_device_items,
    _sam3d_model_edited,
    _set_perspective_correction_enabled,
    _target_armature_changed,
)


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
                "VRM",
                "VRM / VRoid Expressions (Experimental)",
                "Experimental: drive VRoid expression keys (Fcl_*) from ARKit "
                "coefficients",
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
