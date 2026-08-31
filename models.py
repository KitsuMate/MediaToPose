"""Provider-neutral detection records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Mat3 = Tuple[float, float, float, float, float, float, float, float, float]


@dataclass(slots=True)
class CameraIntrinsicsObservation:
    """Provider-neutral full-image pinhole calibration."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    horizontal_fov: float
    vertical_fov: float
    confidence: float = 1.0
    source: str = "FALLBACK"
    bundle_revision: str = ""
    precision: str = ""
    valid: bool = True

    def __post_init__(self) -> None:
        values = (
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.horizontal_fov,
            self.vertical_fov,
            self.confidence,
        )
        self.source = str(self.source).upper()
        self.valid = bool(
            self.valid
            and int(self.width) > 0
            and int(self.height) > 0
            and all(math.isfinite(float(value)) for value in values)
            and self.fx > 0.0
            and self.fy > 0.0
            and 0.0 < self.horizontal_fov < 180.0
            and 0.0 < self.vertical_fov < 180.0
            and self.confidence > 0.0
            and self.source in {"ANYCALIB", "MANUAL", "FALLBACK"}
        )


@dataclass(slots=True)
class JointFrameObservation:
    """Compact provider joint orientation, expressed relative to rest."""

    rotation_matrix: Mat3
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:
        self.valid = bool(
            self.valid
            and len(self.rotation_matrix) == 9
            and all(math.isfinite(float(value)) for value in self.rotation_matrix)
            and math.isfinite(float(self.confidence))
            and self.confidence > 0.0
        )


@dataclass(slots=True)
class RootMotionObservation:
    """Provider observation that may drive video root motion.

    Translation and rotation are camera-relative.  They are deliberately
    retained separately from anatomical landmarks so ordinary pose retargeting
    remains root-relative and provider-neutral.
    """

    translation: Vec3
    rotation_matrix: Optional[Mat3] = None
    focal_length: float = 0.0
    scale: float = 1.0
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:
        values = [*self.translation, self.focal_length, self.scale, self.confidence]
        if self.rotation_matrix is not None:
            values.extend(self.rotation_matrix)
        self.valid = bool(
            self.valid
            and all(math.isfinite(float(value)) for value in values)
            and self.confidence > 0.0
        )


@dataclass(slots=True)
class Landmark:
    position: Vec3
    image: Vec2 = (0.0, 0.0)
    confidence: float = 1.0
    valid: bool = True

    def __post_init__(self) -> None:
        finite = (
            all(math.isfinite(float(value)) for value in self.position)
            and all(math.isfinite(float(value)) for value in self.image)
            and math.isfinite(float(self.confidence))
        )
        self.valid = bool(self.valid and finite)


@dataclass(slots=True)
class CanonicalFrame:
    timestamp_ms: int
    body: Dict[str, Landmark] = field(default_factory=dict)
    hands: Dict[str, Dict[str, Landmark]] = field(default_factory=dict)
    # Provider-specific observations that improve post-processing without
    # becoming rig mappings. Keys use stable anatomical names.
    auxiliary: Dict[str, Landmark] = field(default_factory=dict)
    joint_frames: Dict[str, JointFrameObservation] = field(default_factory=dict)
    face_landmarks: List[Landmark] = field(default_factory=list)
    # Image-space face points used only by the lightweight media preview.
    # Retargeting uses blendshapes and therefore does not need to persist the
    # much larger 3D face mesh.
    face_overlay: List[Vec2] = field(default_factory=list)
    blendshapes: Dict[str, float] = field(default_factory=dict)
    camera: Optional[CameraIntrinsicsObservation] = None
    root_motion: Optional[RootMotionObservation] = None
    warnings: List[str] = field(default_factory=list)


@dataclass(slots=True)
class MediaInfo:
    path: str
    kind: str
    fps: float = 0.0
    frame_count: int = 1
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
