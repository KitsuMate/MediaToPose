"""Immutable capture requests, settings snapshots and their enums.

Operators and the API convert Blender PropertyGroups into these frozen
dataclasses once, at their boundary. Lower layers read only these values,
never RNA. The enums are ``StrEnum`` so they compare equal to the RNA enum
identifiers they replace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class CaptureMode(StrEnum):
    POSE = "POSE"
    KEYFRAMES = "KEYFRAMES"
    ANALYZE = "ANALYZE"


class MediaKind(StrEnum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"


class RootMotionMode(StrEnum):
    AUTOMATIC = "AUTOMATIC"
    IN_PLACE = "IN_PLACE"
    OFF = "OFF"


class FaceOutput(StrEnum):
    ARKIT = "ARKIT"
    FACS = "FACS"
    VRM = "VRM"
    BONES = "BONES"


class OutputMode(StrEnum):
    """Where a capture writes its animation."""

    NEW_ACTION = "NEW_ACTION"
    REPLACE = "REPLACE"
    NLA_STRIP = "NLA_STRIP"


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class CaptureSettings:
    """The scene capture settings, detached from Blender RNA.

    Field names match ``MTPSceneSettings`` so functions that take either keep
    working while they migrate.
    """

    provider: str = "MEDIAPIPE"
    use_body: bool = True
    use_hands: bool = False
    use_face: bool = False
    face_output: FaceOutput = FaceOutput.ARKIT
    root_motion_mode: RootMotionMode = RootMotionMode.AUTOMATIC
    keep_original_root_position: bool = False
    keep_detection_cache: bool = True
    cache_limit_mb: int = 512
    video_frame: int = 1
    mirror_input: bool = False
    smoothing: float = 0.35
    sam3d_model_variant: str = "VITH"
    sam3d_device: str = "AUTO"
    rohm_device: str = "AUTO"
    anycalib_device: str = "AUTO"
    perspective_correction: str = "AUTO"
    manual_fov_override: bool = False
    manual_vertical_fov: float = 45.0
    media_path: str = ""
    media_identity_json: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "face_output", FaceOutput(self.face_output))
        object.__setattr__(
            self, "root_motion_mode", RootMotionMode(self.root_motion_mode)
        )

    @classmethod
    def from_rna(cls, settings: Any) -> CaptureSettings:
        """Snapshot ``scene.mtp_settings`` (or any object with these fields)."""
        values = {}
        for item in fields(cls):
            if hasattr(settings, item.name):
                value = getattr(settings, item.name)
                values[item.name] = value if value is not None else item.default
        return cls(**values)


@dataclass(frozen=True)
class PostProcessModuleConfig:
    module_id: str
    display_name: str
    enabled: bool
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _immutable_mapping(self.values))

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class PostProcessConfig:
    detected_profile: str
    modules: tuple[PostProcessModuleConfig, ...]
    diagnostic_mode: bool = False

    @property
    def post_processing_modules(self) -> tuple[PostProcessModuleConfig, ...]:
        return self.modules

    def module(self, module_id: str) -> PostProcessModuleConfig | None:
        return next(
            (item for item in self.modules if item.module_id == module_id), None
        )

    def enabled(self, module_id: str) -> bool:
        item = self.module(module_id)
        return bool(item and item.enabled)


@dataclass(frozen=True)
class CaptureRequest:
    provider_id: str
    mode: CaptureMode
    media_path: str
    media_kind: MediaKind
    source_frame_index: int
    use_body: bool
    use_hands: bool
    use_face: bool
    face_output: FaceOutput
    root_motion_mode: RootMotionMode
    keep_original_root_position: bool
    keep_detection_cache: bool
    cache_limit_bytes: int
    start_frame: float
    scene_name: str
    scene_pointer: int
    source_frame_count: int = 0
    target_name: str = ""
    target_pointer: int = 0
    action_name: str = ""
    object_mode: str = "OBJECT"
    selected_object_names: tuple[str, ...] = ()
    analysis_path: str = ""
    source_identity: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    mapping_entries: Mapping[str, Any] = field(default_factory=dict)
    postprocess_config: PostProcessConfig | None = None
    # Probed media (``models.MediaInfo``) and the identity the detection
    # cache stores it under; set by the capture session.
    media_info: Any = None
    cache_media_info: Any = None
    cache_key: str = ""
    cache_identity: Mapping[str, Any] = field(default_factory=dict)
    output_mode: OutputMode = OutputMode.NEW_ACTION
    # Source frames ``[first, end)`` to apply, 0-based; ``None`` applies all.
    frame_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CaptureMode(self.mode))
        object.__setattr__(self, "output_mode", OutputMode(self.output_mode))
        if self.frame_range is not None:
            first, end = (int(value) for value in self.frame_range)
            object.__setattr__(self, "frame_range", (first, end))
        object.__setattr__(self, "media_kind", MediaKind(self.media_kind))
        object.__setattr__(self, "face_output", FaceOutput(self.face_output))
        object.__setattr__(
            self, "root_motion_mode", RootMotionMode(self.root_motion_mode)
        )
        object.__setattr__(
            self,
            "source_identity",
            _immutable_mapping(self.source_identity),
        )
        object.__setattr__(self, "options", _immutable_mapping(self.options))
        object.__setattr__(
            self,
            "mapping_entries",
            _immutable_mapping(self.mapping_entries),
        )
        object.__setattr__(
            self, "cache_identity", _immutable_mapping(self.cache_identity)
        )

    @property
    def provider(self) -> str:
        return self.provider_id


@dataclass(frozen=True)
class SamSetupRequest:
    operation: str
    model_variant: str = ""
    device: str = ""
    model_storage_directory: str = ""


__all__ = [
    "CaptureMode",
    "CaptureRequest",
    "CaptureSettings",
    "FaceOutput",
    "MediaKind",
    "OutputMode",
    "PostProcessConfig",
    "PostProcessModuleConfig",
    "RootMotionMode",
    "SamSetupRequest",
]
