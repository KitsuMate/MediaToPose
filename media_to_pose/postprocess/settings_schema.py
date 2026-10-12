"""Every clean-up module and parameter, described once.

The RNA PropertyGroup that stores module settings on the armature is built
from ``PARAMETERS``; the Pose & Motion Cleanup panel draws ``MODULES`` in
their ``layout`` order; the Python API lists and validates configurations
against the same records. Angles are radians, times seconds, factors and
fractions 0 to 1 and percentages 0 to 100.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import InvalidRequest


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str  # "bool", "float", "int", "enum" or "string"
    label: str
    default: Any
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    subtype: str = ""
    items: tuple[tuple[str, str, str], ...] = ()

    @property
    def unit(self) -> str:
        return {
            "ANGLE": "radians",
            "TIME": "seconds",
            "FACTOR": "fraction",
            "PERCENTAGE": "percent",
        }.get(self.subtype, "")

    def validate(self, value: Any) -> Any:
        """Coerce ``value`` to this parameter's type or raise InvalidRequest."""
        try:
            if self.kind == "bool":
                if not isinstance(value, (bool, int)):
                    raise TypeError
                result: Any = bool(value)
            elif self.kind == "float":
                if isinstance(value, bool):
                    raise TypeError
                result = float(value)
                if not math.isfinite(result):
                    raise ValueError
            elif self.kind == "int":
                if isinstance(value, bool) or float(value) != int(value):
                    raise TypeError
                result = int(value)
            elif self.kind == "enum":
                result = str(value)
                if result not in {item[0] for item in self.items}:
                    raise ValueError
            else:
                result = str(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRequest(
                f"{self.name} must be a valid {self.kind}", parameter=self.name
            ) from exc
        if self.minimum is not None and result < self.minimum:
            raise InvalidRequest(
                f"{self.name} must be at least {self.minimum}", parameter=self.name
            )
        if self.maximum is not None and result > self.maximum:
            raise InvalidRequest(
                f"{self.name} must be at most {self.maximum}", parameter=self.name
            )
        return result

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.kind,
            "label": self.label,
            "default": self.default,
            "description": self.description,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.unit:
            payload["unit"] = self.unit
        if self.items:
            payload["choices"] = [item[0] for item in self.items]
        return payload


@dataclass(frozen=True)
class Row:
    """One panel row: parameters side by side, optionally gated."""

    parameters: tuple[str, ...] = ()
    enabled_by: str = ""  # a parameter name, or "@root_motion_automatic"
    separator: bool = False


@dataclass(frozen=True)
class Module:
    module_id: str
    display_name: str
    description: str
    summary: str
    default_enabled: bool
    sequence_only: bool
    layout: tuple[Row, ...] = field(default_factory=tuple)

    @property
    def parameters(self) -> tuple[str, ...]:
        names: list[str] = []
        for row in self.layout:
            for name in row.parameters:
                if name not in names:
                    names.append(name)
        return tuple(names)


_RADIANS_90 = 1.5707963268
_RADIANS_180 = 3.1415926536

PARAMETERS: dict[str, Parameter] = {
    item.name: item
    for item in (
        # Torso Curve and Shoulders.
        Parameter("strength", "float", "Strength", 1.0, minimum=0.0, maximum=2.0),
        Parameter(
            "start_elevation",
            "float",
            "Start Elevation",
            0.5235987756,
            minimum=0.0,
            maximum=_RADIANS_90,
            subtype="ANGLE",
        ),
        Parameter(
            "full_influence_elevation",
            "float",
            "Full Influence",
            _RADIANS_90,
            minimum=0.0,
            maximum=_RADIANS_90,
            subtype="ANGLE",
        ),
        Parameter(
            "maximum_swing",
            "float",
            "Maximum Swing",
            0.6108652382,
            minimum=0.0,
            maximum=_RADIANS_90,
            subtype="ANGLE",
        ),
        Parameter(
            "maximum_twist",
            "float",
            "Maximum Twist",
            0.1745329252,
            minimum=0.0,
            maximum=_RADIANS_90,
            subtype="ANGLE",
        ),
        Parameter(
            "repair_arm_silhouette",
            "bool",
            "Repair Arm Silhouette",
            True,
            description=(
                "Use SAM image-space joints to gently repair an arm that is "
                "substantially more bent than its detected silhouette"
            ),
        ),
        Parameter(
            "maximum_arm_extension",
            "float",
            "Maximum IK Shift",
            0.02,
            description=(
                "Maximum wrist or IK-target displacement as a fraction of the "
                "complete arm length"
            ),
            minimum=0.0,
            maximum=0.05,
            subtype="PERCENTAGE",
        ),
        # Elbows & Arm Twist, Knees & Leg Twist.
        Parameter("hinge_projection", "bool", "Hinge Projection", True),
        Parameter(
            "confidence_threshold",
            "float",
            "Axis Confidence Threshold",
            0.75,
            description=(
                "Minimum confidence for upper-limb or thigh axis reconstruction; "
                "palm roll is continuous and uses confidence only as a temporal "
                "weight"
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        Parameter(
            "upper_arm_twist_limit",
            "float",
            "Upper Arm Twist",
            1.308996939,
            minimum=0.0,
            maximum=_RADIANS_180,
            subtype="ANGLE",
        ),
        Parameter(
            "forearm_twist_limit",
            "float",
            "Maximum Forearm Correction",
            _RADIANS_90,
            description=(
                "Maximum palm-derived correction applied around the forearm; "
                "this caps the result and never disables correction"
            ),
            minimum=0.0,
            maximum=_RADIANS_180,
            subtype="ANGLE",
        ),
        Parameter(
            "palm_roll_strength",
            "float",
            "Palm Roll Influence",
            0.65,
            description=(
                "How strongly detected palm roll drives wrist and forearm twist; "
                "reduce for occluded or low-detail hands"
            ),
            minimum=0.0,
            maximum=1.0,
        ),
        Parameter(
            "thigh_twist_limit",
            "float",
            "Thigh Twist",
            1.0471975512,
            minimum=0.0,
            maximum=_RADIANS_180,
            subtype="ANGLE",
        ),
        # Reduce Camera-Induced Sliding.
        Parameter(
            "lateral_root_strength",
            "float",
            "Lateral Root Strength",
            1.0,
            description=(
                "Blend the original lateral root trajectory with background-camera "
                "compensation"
            ),
            minimum=0.0,
            maximum=1.0,
            subtype="FACTOR",
        ),
        # Reject Bad Detections & Fill Gaps.
        Parameter(
            "reject_unreliable_limbs",
            "bool",
            "Reject Unreliable Limbs",
            True,
            description=(
                "Mask low-confidence limbs and isolated motion jumps before motion "
                "repair"
            ),
        ),
        Parameter(
            "reject_unusable_frames",
            "bool",
            "Reject Unusable Frames",
            True,
            description=(
                "Mask whole frames when their body evidence is broadly unreliable"
            ),
        ),
        Parameter(
            "maximum_motion_rejected",
            "float",
            "Maximum Motion Rejected",
            30.0,
            description=(
                "Maximum share of otherwise available joint observations that this "
                "filter may reject; detector-native missing data does not count"
            ),
            minimum=0.0,
            maximum=100.0,
            subtype="PERCENTAGE",
        ),
        Parameter(
            "manual_bad_frames",
            "string",
            "Additional Bad Source Frames",
            "",
            description=(
                "Optional 1-based source frames or inclusive ranges, for example "
                "12, 20-24; these explicit masks are not limited by the automatic "
                "rejection budget"
            ),
        ),
        Parameter(
            "manual_bad_scope",
            "enum",
            "Manual Mask Scope",
            "BODY",
            description="Body region masked on every additional bad source frame",
            items=(
                ("BODY", "Whole Body", "Mask the complete body and root observation"),
                ("ARM_L", "Left Arm", "Mask the left shoulder and arm"),
                ("ARM_R", "Right Arm", "Mask the right shoulder and arm"),
                ("LEG_L", "Left Leg", "Mask the left leg"),
                ("LEG_R", "Right Leg", "Mask the right leg"),
            ),
        ),
        # Smooth Motion & Foot Contact.
        Parameter(
            "smooth_motion",
            "bool",
            "Smooth Motion",
            True,
            description=(
                "Apply symmetric, confidence-aware smoothing without filter lag"
            ),
        ),
        Parameter(
            "ground_contacts",
            "bool",
            "Ground Contacts",
            True,
            description=(
                "Stabilize root travel and planted foot positions without changing "
                "foot orientation"
            ),
        ),
        Parameter(
            "smoothing_strength",
            "float",
            "Smoothing Strength",
            0.5,
            description="Blend between detected and stabilized video animation",
            minimum=0.0,
            maximum=1.0,
            subtype="FACTOR",
        ),
        Parameter(
            "stabilize_hands",
            "bool",
            "Stabilize Hands",
            True,
            description=(
                "Reject abrupt wrist and finger rotation bursts, then enforce a "
                "bounded angular speed after smoothing"
            ),
        ),
        Parameter(
            "finger_max_angular_speed",
            "float",
            "Finger Maximum Speed",
            10.471975512,
            description=(
                "Largest accepted finger rotation speed; faster detector jumps are "
                "held until the pose becomes stable"
            ),
            minimum=1.0471975512,
            maximum=25.132741229,
            subtype="ANGLE",
        ),
        Parameter(
            "wrist_max_angular_speed",
            "float",
            "Wrist Maximum Speed",
            12.566370614,
            description=(
                "Largest accepted wrist rotation speed; faster detector jumps are "
                "held until the pose becomes stable"
            ),
            minimum=1.0471975512,
            maximum=25.132741229,
            subtype="ANGLE",
        ),
        Parameter(
            "hand_reacquire_delay",
            "float",
            "Hand Reacquire Delay",
            0.3,
            description=(
                "How long a substantially different hand pose must remain stable "
                "before a rejected jump is accepted"
            ),
            minimum=0.1,
            maximum=10.0,
            subtype="TIME",
        ),
        # RoHM Cleanup.
        Parameter(
            "visibility_threshold",
            "float",
            "Visibility Threshold",
            0.50,
            description="Confidence below this value is treated as occluded",
            minimum=0.0,
            maximum=1.0,
            subtype="FACTOR",
        ),
        Parameter(
            "learned_pose_residual",
            "bool",
            "AI Pose Cleanup (Alpha)",
            False,
            description=(
                "Alpha feature: use AI to adjust only masked anatomical-global "
                "joints, with each change limited to two degrees from deterministic "
                "repair"
            ),
        ),
        Parameter(
            "root_trajectory_cleanup",
            "bool",
            "Root Trajectory Cleanup",
            True,
            description=(
                "Use RoHM only for masked root observations while Automatic Root "
                "Motion is active"
            ),
        ),
    )
}


MODULES: tuple[Module, ...] = (
    Module(
        "SPINE_CORRECTION",
        "Torso Curve (Experimental)",
        "Experimental: correct torso curvature from provider spine observations",
        "Improves torso curvature from detected spine motion.",
        default_enabled=False,
        sequence_only=False,
        layout=(Row(("strength",)),),
    ),
    Module(
        "SHOULDER_CORRECTION",
        "Shoulders",
        (
            "Add clavicle elevation assistance and optional target-relative "
            "pectoral deformation after the raw pose is solved"
        ),
        "Assists raised shoulders while preserving the pose.",
        default_enabled=True,
        sequence_only=False,
        layout=(
            Row(("strength",)),
            Row(("start_elevation", "full_influence_elevation")),
            Row(("maximum_swing", "maximum_twist")),
            Row(("repair_arm_silhouette",)),
            Row(("maximum_arm_extension",), enabled_by="repair_arm_silhouette"),
        ),
    ),
    Module(
        "ARM_AXIS_CORRECTION",
        "Elbows & Arm Twist",
        "Stabilize arm twist and enforce elbow hinge axes",
        "Stabilizes elbow bending and arm twist.",
        default_enabled=True,
        sequence_only=False,
        layout=(
            Row(("hinge_projection",)),
            Row(("confidence_threshold",)),
            Row(("forearm_twist_limit",)),
            Row(("upper_arm_twist_limit",)),
            Row(("palm_roll_strength",)),
        ),
    ),
    Module(
        "LEG_AXIS_CORRECTION",
        "Knees & Leg Twist",
        "Stabilize leg twist and enforce knee hinge axes",
        "Stabilizes knee bending and leg twist.",
        default_enabled=True,
        sequence_only=False,
        layout=(
            Row(("hinge_projection",)),
            Row(("confidence_threshold",)),
            Row(("thigh_twist_limit",)),
        ),
    ),
    Module(
        "CAMERA_MOTION_CORRECTION",
        "Reduce Camera-Induced Sliding (Experimental)",
        (
            "Experimental video-only masked background tracking that removes "
            "camera-induced lateral root slide without changing rotations"
        ),
        "Reduces sideways travel caused by a moving camera.",
        default_enabled=True,
        sequence_only=True,
        layout=(Row(("lateral_root_strength",)),),
    ),
    Module(
        "DETECTION_FILTER",
        "Reject Bad Detections & Fill Gaps (Experimental)",
        (
            "Experimental video-only confidence and jump filtering followed by "
            "deterministic quaternion interpolation or nearest-good boundary hold"
        ),
        "Rejects unreliable motion and fills gaps without a learned model.",
        default_enabled=True,
        sequence_only=True,
        layout=(
            Row(("reject_unreliable_limbs",)),
            Row(("reject_unusable_frames",)),
            Row(("maximum_motion_rejected",)),
            Row(separator=True),
            Row(("manual_bad_frames",)),
            Row(("manual_bad_scope",), enabled_by="manual_bad_frames"),
        ),
    ),
    Module(
        "ROHM_CORRECTION",
        "RoHM Cleanup (Experimental)",
        (
            "Experimental video-only root repair with optional Alpha AI pose "
            "cleanup using an external ONNX bundle"
        ),
        "Optional learned residual for already-masked pose or root gaps.",
        default_enabled=False,
        sequence_only=True,
        layout=(
            Row(("learned_pose_residual",)),
            Row(("root_trajectory_cleanup",), enabled_by="@root_motion_automatic"),
            Row(("visibility_threshold",)),
        ),
    ),
    Module(
        "TEMPORAL_CORRECTION",
        "Smooth Motion & Foot Contact (Experimental)",
        (
            "Experimental video-only confidence-aware smoothing, root "
            "stabilization, and planted-foot correction"
        ),
        "Smooths motion and preserves planted feet.",
        default_enabled=True,
        sequence_only=True,
        layout=(
            Row(("smooth_motion",)),
            Row(("ground_contacts",)),
            Row(("smoothing_strength",), enabled_by="smooth_motion"),
            Row(("stabilize_hands",)),
            Row(
                (
                    "finger_max_angular_speed",
                    "wrist_max_angular_speed",
                    "hand_reacquire_delay",
                ),
                enabled_by="stabilize_hands",
            ),
        ),
    ),
)

MODULES_BY_ID: dict[str, Module] = {module.module_id: module for module in MODULES}
# Pipeline order with display name and description, as the panels list them.
POST_PROCESSING_MODULES = tuple(
    (module.module_id, module.display_name, module.description) for module in MODULES
)
POST_PROCESSING_DEFAULT_ENABLED = {
    module.module_id: module.default_enabled for module in MODULES
}
MODULE_SETTING_FIELDS = {module.module_id: module.parameters for module in MODULES}


def module(module_id: str) -> Module:
    try:
        return MODULES_BY_ID[str(module_id)]
    except KeyError as exc:
        supported = ", ".join(MODULES_BY_ID)
        raise InvalidRequest(
            f"Unknown post-processor {module_id!r}; expected one of {supported}"
        ) from exc


def defaults(module_id: str) -> dict[str, Any]:
    return {name: PARAMETERS[name].default for name in module(module_id).parameters}


def validate_settings(module_id: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Return coerced settings for one module; unknown names are errors."""
    allowed = module(module_id).parameters
    result = {}
    for name, value in values.items():
        if name not in allowed:
            raise InvalidRequest(
                f"{module_id} has no setting {name!r}", parameter=str(name)
            )
        result[name] = PARAMETERS[name].validate(value)
    return result


def describe() -> list[dict[str, Any]]:
    """JSON-ready description of every module and its parameters."""
    return [
        {
            "module_id": item.module_id,
            "display_name": item.display_name,
            "description": item.description,
            "default_enabled": item.default_enabled,
            "video_only": item.sequence_only,
            "parameters": [PARAMETERS[name].describe() for name in item.parameters],
        }
        for item in MODULES
    ]


__all__ = [
    "MODULES",
    "MODULES_BY_ID",
    "MODULE_SETTING_FIELDS",
    "PARAMETERS",
    "POST_PROCESSING_DEFAULT_ENABLED",
    "POST_PROCESSING_MODULES",
    "Module",
    "Parameter",
    "Row",
    "defaults",
    "describe",
    "module",
    "validate_settings",
]
