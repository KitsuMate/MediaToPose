"""Update, poll and item callbacks shared by the RNA property groups."""

from __future__ import annotations

from ...core.coordinator import COORDINATOR
from ...rig.locks import manual_locks_suppressed


def _mapping_edited(self, context) -> None:
    if manual_locks_suppressed():
        return
    self.manual_lock = True
    self.source = "MANUAL"
    self.confidence = 1.0
    self.status = "VALID" if self.target_name else "MISSING"
    settings = getattr(getattr(context, "scene", None), "mtp_settings", None)
    armature = getattr(settings, "target_armature", None) if settings else None
    if armature is not None:
        from ...rig import mapping

        mapping.validate_mapping(armature, settings=settings)
        from ...postprocess import validate_modules

        validate_modules(armature)


def _fk_switch_edited(self, context) -> None:
    if not manual_locks_suppressed():
        self.fk_switches_manual = True
        settings = getattr(getattr(context, "scene", None), "mtp_settings", None)
        armature = getattr(settings, "target_armature", None) if settings else None
        if armature is not None:
            from ...postprocess import validate_modules
            from ...rig import mapping

            mapping.validate_mapping(armature, settings=settings)
            validate_modules(armature)


def _shape_override_edited(self, context) -> None:
    if manual_locks_suppressed() or not context or not self.semantic_id:
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
    from ...rig import mapping

    mapping.validate_mapping(armature, settings=settings)
    from ...postprocess import validate_modules

    validate_modules(armature)


def _armature_poll(_self, obj) -> bool:
    return bool(obj and obj.type == "ARMATURE")


def _target_armature_changed(self, _context) -> None:
    armature = getattr(self, "target_armature", None)
    if armature is None:
        return
    try:
        from ...rig import mapping

        mapping.refresh_auto_mapping(armature, settings=self)
    except (AttributeError, ReferenceError, RuntimeError):
        pass


def _cache_enabled_edited(self, _context) -> None:
    if self.keep_detection_cache or COORDINATOR.is_busy():
        return
    # Property callbacks run on Blender's main thread. File removal is kept
    # out of panel draw callbacks, which Blender treats as read-only.
    from ...core.cache import store as cache_store
    from .. import preview as cache_preview

    cache_store.clear_cache()
    cache_preview.refresh_after_cache_update()


def _preview_edited(_self, _context) -> None:
    from .. import preview as cache_preview

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
    from ...rig import mapping

    mapping.validate_mapping(armature, settings=self)


def _media_path_edited(self, _context) -> None:
    from ...core.cache import media_source as cache_media
    from .. import preview as cache_preview

    self.video_frame = 1
    self.last_camera_vertical_fov = 0.0
    self.last_camera_source = ""
    resolved = cache_media.existing_media_path(self.media_path)
    if resolved:
        try:
            cache_media.record_media_selection(
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
    from .. import preview as cache_preview

    cache_preview.schedule_refresh()


_SAM3D_VITH_ITEM = ("VITH", "ViT-H", "Default immutable ONNX graph bundle")


def _sam3d_device_items(_self, _context):
    # Keep this callback read-only so panel draw never probes hardware.
    try:
        from ...backends.sam3d import backend as sam3d_backend

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
        from ...backends.rohm import backend as rohm_backend

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
