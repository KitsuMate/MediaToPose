"""Extensible, non-destructive target-pose post-processing.

Detection providers never import this module. The retargeter builds a
provider-neutral temporary target pose, then corrections amend it here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
)

from mathutils import Matrix, Quaternion, Vector

from .constants import (
    POST_PROCESSING_DEFAULT_ENABLED,
    POST_PROCESSING_MODULES,
    PROFILE_ARP,
    PROFILE_GENERIC,
    PROFILE_RIGIFY,
)
from .core.requests import PostProcessConfig, PostProcessModuleConfig
from .target_pose import (
    BoneChannel,
    TargetPose,
)
from .target_pose import (
    orientation_frame as _orientation_frame,
)
from .target_pose import (
    signed_twist_angle as _signed_twist_angle,
)
from .target_pose import (
    stable_orientation_delta as _stable_orientation_delta,
)

MODULE_SETTING_FIELDS = {
    "SPINE_CORRECTION": ("strength",),
    "SHOULDER_CORRECTION": (
        "strength",
        "start_elevation",
        "full_influence_elevation",
        "maximum_swing",
        "maximum_twist",
        "repair_arm_silhouette",
        "maximum_arm_extension",
    ),
    "ARM_AXIS_CORRECTION": (
        "hinge_projection",
        "confidence_threshold",
        "upper_arm_twist_limit",
        "forearm_twist_limit",
        "palm_roll_strength",
    ),
    "LEG_AXIS_CORRECTION": (
        "hinge_projection",
        "confidence_threshold",
        "thigh_twist_limit",
    ),
    "CAMERA_MOTION_CORRECTION": ("lateral_root_strength",),
    "DETECTION_FILTER": (
        "reject_unreliable_limbs",
        "reject_unusable_frames",
        "maximum_motion_rejected",
        "manual_bad_frames",
        "manual_bad_scope",
    ),
    "TEMPORAL_CORRECTION": (
        "smooth_motion",
        "ground_contacts",
        "smoothing_strength",
        "stabilize_hands",
        "finger_max_angular_speed",
        "wrist_max_angular_speed",
        "hand_reacquire_delay",
    ),
    "ROHM_CORRECTION": (
        "learned_pose_residual",
        "root_trajectory_cleanup",
        "visibility_threshold",
    ),
}


def ensure_module_configs(config) -> bool:
    """Ensure the config contains exactly the registered correction modules."""
    registered_ids = {item[0] for item in POST_PROCESSING_MODULES}
    seen_ids = set()
    try:
        for index in reversed(range(len(config.post_processing_modules))):
            module_id = config.post_processing_modules[index].module_id
            if module_id not in registered_ids or module_id in seen_ids:
                config.post_processing_modules.remove(index)
            else:
                seen_ids.add(module_id)
    except (AttributeError, RuntimeError):
        return False
    existing = {item.module_id: item for item in config.post_processing_modules}
    for module_id, name, description in POST_PROCESSING_MODULES:
        item = existing.get(module_id)
        try:
            if item is None:
                item = config.post_processing_modules.add()
                item.module_id = module_id
                item.enabled = POST_PROCESSING_DEFAULT_ENABLED[module_id]
            item.display_name = name
            item.description = description
        except (AttributeError, RuntimeError):
            # Linked/read-only armatures and Blender's UI draw context do not
            # permit ID writes. Operators may retry from a writable context;
            # panel drawing must remain read-only.
            return False
    return True


def module_config(config, module_id: str):
    ensure_module_configs(config)
    return next(
        (
            item
            for item in config.post_processing_modules
            if item.module_id == module_id
        ),
        None,
    )


def ordered_module_configs(config) -> list[object]:
    """Return existing settings in fixed order without mutating Blender IDs."""
    configured = {item.module_id: item for item in config.post_processing_modules}
    return [
        configured[module_id]
        for module_id, _name, _description in POST_PROCESSING_MODULES
        if module_id in configured
    ]


def compile_postprocess_config(
    config,
    *,
    diagnostic_mode: bool = False,
) -> PostProcessConfig:
    """Snapshot Blender RNA settings once for one capture operation."""
    if isinstance(config, PostProcessConfig):
        return config
    ensure_module_configs(config)
    modules = []
    for item in ordered_module_configs(config):
        fields = MODULE_SETTING_FIELDS.get(item.module_id, ())
        modules.append(
            PostProcessModuleConfig(
                module_id=str(item.module_id),
                display_name=str(item.display_name or item.module_id),
                enabled=bool(item.enabled),
                values={name: getattr(item, name) for name in fields},
            )
        )
    return PostProcessConfig(
        detected_profile=str(config.detected_profile),
        modules=tuple(modules),
        diagnostic_mode=bool(diagnostic_mode),
    )


def serialize_module_configs(config) -> list[dict]:
    ensure_module_configs(config)
    return [
        {
            "module_id": item.module_id,
            "display_name": item.display_name,
            "description": item.description,
            "enabled": item.enabled,
            "settings": {
                field_name: getattr(item, field_name)
                for field_name in MODULE_SETTING_FIELDS[item.module_id]
            },
        }
        for item in config.post_processing_modules
    ]


def load_module_configs(config, payload: Iterable[dict]) -> None:
    ensure_module_configs(config)
    existing = {item.module_id: item for item in config.post_processing_modules}
    for data in payload:
        module_id = str(data.get("module_id", ""))
        item = existing.get(module_id)
        if item is None:
            raise ValueError(
                f"Unsupported post-processing module {module_id or '<missing>'}"
            )
        item.display_name = str(
            data.get("display_name", item.display_name or module_id)
        )
        item.description = str(data.get("description", item.description))
        item.enabled = bool(data.get("enabled", True))
        settings = dict(data.get("settings", {}))
        for key, value in settings.items():
            if key in MODULE_SETTING_FIELDS[module_id] and hasattr(item, key):
                setattr(item, key, value)
    ensure_module_configs(config)


def reset_module_defaults(config, module_id: str) -> bool:
    item = module_config(config, module_id)
    if item is None:
        return False
    # RNA's unset restores the PropertyGroup declaration defaults.
    for field_name in MODULE_SETTING_FIELDS[module_id]:
        item.property_unset(field_name)
    item.enabled = POST_PROCESSING_DEFAULT_ENABLED[module_id]
    item.last_warning = ""
    item.status = "VALID"
    return True


def _smoothstep(start: float, end: float, value: float) -> float:
    if end <= start + 1.0e-8:
        return 1.0 if value >= end else 0.0
    t = min(max((value - start) / (end - start), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class PostProcessingSession:
    """Small continuity state shared by consecutive frame solves."""

    _last_timestamp_ms: int | None = None
    _forearm_twist_phase: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self._last_timestamp_ms = None
        self._forearm_twist_phase.clear()

    def prepare_frame(self, timestamp_ms: int) -> None:
        timestamp_ms = int(timestamp_ms)
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            self._forearm_twist_phase.clear()
        self._last_timestamp_ms = timestamp_ms

    def unwrap_forearm_twist(self, side: str, angle: float) -> float:
        """Keep a circular palm-roll observation on one continuous phase."""
        previous = self._forearm_twist_phase.get(side)
        if previous is not None:
            angle += math.tau * round((previous - angle) / math.tau)
        self._forearm_twist_phase[side] = angle
        return angle


@dataclass
class SequenceProcessingContext:
    """Provider-neutral entry point for a complete baked-video sequence."""

    samples: Sequence[Any]
    timestamps: Sequence[float]
    root_motion_mode: str
    process_temporal: Callable[[Any], Iterator[Any]]
    process_camera_motion: Callable[[Any], Iterator[Any]] | None = None
    process_detection_filter: Callable[[Any], Iterator[Any]] | None = None
    process_rohm: Callable[[Any], Iterator[Any]] | None = None


class CorrectionModule:
    module_id = ""
    display_name = ""
    supported_profiles = {
        PROFILE_ARP,
        PROFILE_RIGIFY,
        PROFILE_GENERIC,
    }
    required_semantics: tuple[str, ...] = ()
    allows_endpoint_motion = False
    scope = "FRAME"

    def validate(self, pose_or_armature, settings) -> list[str]:
        return []

    def reset(self, session: PostProcessingSession) -> None:
        session.reset()

    def process_frame(
        self,
        pose: TargetPose,
        settings,
        session: PostProcessingSession,
    ) -> list[str]:
        raise NotImplementedError

    def process_sequence(
        self,
        context: SequenceProcessingContext,
        settings,
    ) -> Iterator[Any]:
        raise NotImplementedError


def _curve_sample(points: list[Vector], count: int = 65) -> list[Vector]:
    """Sample a C1 Catmull-Rom centerline through provider joints."""
    if len(points) < 2:
        return list(points)
    result = []
    segments = len(points) - 1
    for sample_index in range(count):
        value = segments * sample_index / max(count - 1, 1)
        index = min(int(value), segments - 1)
        t = value - index
        p0 = points[max(index - 1, 0)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(index + 2, len(points) - 1)]
        result.append(
            0.5
            * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
            )
        )
    return result


def _hermite_sample(
    start: Vector,
    end: Vector,
    start_tangent: Vector,
    end_tangent: Vector,
    count: int = 65,
) -> list[Vector]:
    result = []
    for index in range(count):
        t = index / max(count - 1, 1)
        t2, t3 = t * t, t * t * t
        result.append(
            (2.0 * t3 - 3.0 * t2 + 1.0) * start
            + (t3 - 2.0 * t2 + t) * start_tangent
            + (-2.0 * t3 + 3.0 * t2) * end
            + (t3 - t2) * end_tangent
        )
    return result


def _arc_point(samples: list[Vector], fraction: float) -> Vector:
    if len(samples) < 2:
        return samples[0].copy()
    lengths = [0.0]
    for first, second in zip(samples, samples[1:], strict=False):
        lengths.append(lengths[-1] + (second - first).length)
    target = min(max(fraction, 0.0), 1.0) * lengths[-1]
    for index in range(1, len(lengths)):
        if lengths[index] < target:
            continue
        span = lengths[index] - lengths[index - 1]
        alpha = 0.0 if span < 1.0e-8 else (target - lengths[index - 1]) / span
        return samples[index - 1].lerp(samples[index], alpha)
    return samples[-1].copy()


class SpineCorrection(CorrectionModule):
    module_id = "SPINE_CORRECTION"
    display_name = "Spine Correction"
    allows_endpoint_motion = True
    _SEMANTICS = ("SPINE_01", "SPINE_02", "CHEST")
    _LIMBS = (
        ("UPPER_ARM_L", "LOWER_ARM_L", "HAND_L"),
        ("UPPER_ARM_R", "LOWER_ARM_R", "HAND_R"),
        ("UPPER_LEG_L", "LOWER_LEG_L", "FOOT_L"),
        ("UPPER_LEG_R", "LOWER_LEG_R", "FOOT_R"),
    )

    def validate(self, armature, settings) -> list[str]:
        entries = {
            entry.semantic_id: entry for entry in armature.mtp_rig_config.mappings
        }
        names = []
        for semantic_id in self._SEMANTICS:
            entry = entries.get(semantic_id)
            if entry and entry.target_name and entry.target_name not in names:
                names.append(entry.target_name)
        return (
            ["Spine Correction has fewer than two distinct torso controls"]
            if len(names) < 2
            else []
        )

    @staticmethod
    def _valid_position(frame, name):
        point = frame.auxiliary.get(name)
        return Vector(point.position) if point and point.valid else None

    def _source_curve(self, pose: TargetPose):
        frame = pose.frame
        mhr = [self._valid_position(frame, f"MHR_SPINE_{index}") for index in range(4)]
        neck = self._valid_position(frame, "MHR_NECK")
        if all(point is not None for point in mhr) and neck is not None:
            return _curve_sample([*mhr, neck]), False
        body = frame.body
        hips = [body.get("LEFT_HIP"), body.get("RIGHT_HIP")]
        shoulders = [body.get("LEFT_SHOULDER"), body.get("RIGHT_SHOULDER")]
        if not all(point and point.valid for point in hips + shoulders):
            return None, True
        hip = (Vector(hips[0].position) + Vector(hips[1].position)) * 0.5
        shoulder = (Vector(shoulders[0].position) + Vector(shoulders[1].position)) * 0.5
        main = shoulder - hip
        if main.length < 1.0e-8:
            return None, True
        candidates = [
            body.get(name)
            for name in ("LEFT_EAR", "RIGHT_EAR", "LEFT_EYE", "RIGHT_EYE")
        ]
        valid_head = [item for item in candidates if item and item.valid]
        upper = main.normalized()
        if valid_head:
            head = sum((Vector(item.position) for item in valid_head), Vector()) / len(
                valid_head
            )
            observed = head - shoulder
            if observed.length > 1.0e-8:
                observed.normalize()
                angle = main.normalized().angle(observed)
                if angle > math.radians(45.0):
                    observed = main.normalized().slerp(
                        observed, math.radians(45.0) / angle
                    )
                upper = observed
        return _hermite_sample(hip, shoulder, main, upper * main.length * 0.65), True

    def process_frame(self, pose, settings, session) -> list[str]:
        samples, inferred = self._source_curve(pose)
        if not samples:
            return ["Spine Correction lacks usable torso observations"]
        # Sparse inferred curves are a conservative correction layered over
        # the original torso, so retain their pre-correction limb endpoints.
        # SAM's observed MHR curve replaces the original torso geometry: its
        # old endpoints were computed from the wrong shoulder/hip positions
        # and anchoring to them can make an otherwise valid limb unreachable.
        # SAM limb channels already contain provider-derived world directions;
        # resolving those from the corrected roots preserves their shape.
        pose.resolve()
        limb_anchors = []
        for roles in self._LIMBS if inferred else ():
            limb = [pose.channel_for(role) for role in roles]
            if any(channel is None for channel in limb):
                continue
            upper, lower, end = limb
            root = pose._virtual_pose(upper.bone).translation.copy()
            upper_direction = upper.desired_direction
            lower_direction = lower.desired_direction
            if (
                upper_direction is None
                or lower_direction is None
                or upper_direction.length < 1.0e-8
                or lower_direction.length < 1.0e-8
            ):
                continue
            first_length = max(float(upper.bone.length), 1.0e-6)
            second_length = max(float(lower.bone.length), 1.0e-6)
            joint = root + upper_direction.normalized() * first_length
            endpoint = joint + lower_direction.normalized() * second_length
            plane = (joint - root).cross(endpoint - joint)
            limb_anchors.append(
                (
                    roles,
                    limb,
                    endpoint,
                    joint,
                    first_length,
                    second_length,
                    plane,
                )
            )

        channels, seen = [], set()
        for semantic_id in self._SEMANTICS:
            channel = pose.channel_for(semantic_id)
            if channel and channel.bone.name not in seen:
                seen.add(channel.bone.name)
                channels.append(channel)
        warnings = []
        if inferred:
            warnings.append(
                "MediaPipe spine curvature is inferred from torso landmarks"
            )
        if len(channels) < 2:
            warnings.append(
                "Spine Correction has fewer than two distinct torso controls"
            )
        if not channels:
            return warnings

        # Preserve SAM's observed MHR centerline in target space. Re-aligning
        # this curve to a separately estimated 2D torso chord changed every
        # evaluated segment by the same global offset, despite the target pose
        # itself containing accurate 3D tangents.
        sample_origin = samples[0]
        target_samples = []
        for sample in samples:
            relative = sample - sample_origin
            if relative.length <= 1.0e-8:
                target_samples.append(Vector())
            else:
                target_samples.append(
                    pose.to_target_direction(relative) * relative.length
                )
        lengths = [max(channel.bone.length, 1.0e-6) for channel in channels]
        total, cursor = sum(lengths), 0.0
        curve_axis = target_samples[-1] - target_samples[0]
        if curve_axis.length > 1.0e-8:
            curve_axis.normalize()
        # Move the target torso with one common rest-to-observed rotation
        # before adding per-segment curvature. Independent absolute swing
        # solves can preserve the same centerline while giving Chest and
        # UpperChest different global deltas, which shears vertices weighted
        # across their skinning boundary.
        rigid_delta = Quaternion()
        rest_chain_axis = Vector((0.0, 0.0, 0.0))
        for channel, length in zip(channels, lengths, strict=False):
            if channel.rest_direction.length > 1.0e-8:
                rest_chain_axis += channel.rest_direction.normalized() * length
        if rest_chain_axis.length > 1.0e-8 and curve_axis.length > 1.0e-8:
            rigid_delta = rest_chain_axis.normalized().rotation_difference(curve_axis)
        body = pose.frame.body
        hip_lateral = shoulder_lateral = None
        if all(
            body.get(name) and body[name].valid for name in ("LEFT_HIP", "RIGHT_HIP")
        ):
            hip_lateral = Vector(body["RIGHT_HIP"].position) - Vector(
                body["LEFT_HIP"].position
            )
        if all(
            body.get(name) and body[name].valid
            for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER")
        ):
            shoulder_lateral = Vector(body["RIGHT_SHOULDER"].position) - Vector(
                body["LEFT_SHOULDER"].position
            )
        for channel, length in zip(channels, lengths, strict=False):
            start_fraction = cursor / total
            cursor += length
            end_fraction = cursor / total
            direction = _arc_point(target_samples, end_fraction) - _arc_point(
                target_samples, start_fraction
            )
            if direction.length > 1.0e-8:
                direction.normalize()
            reference = None
            if hip_lateral is not None and shoulder_lateral is not None:
                reference = pose.to_target_direction(
                    hip_lateral.lerp(
                        shoulder_lateral,
                        (start_fraction + end_fraction) * 0.5,
                    )
                )
            # MediaPipe's independently reconstructed hip lateral can reverse
            # relative to the shoulder lateral in foreshortened, crossed-leg,
            # or stylized poses. A lateral axis has two valid signs for swing,
            # but choosing the opposite sign asks the spine for an otherwise
            # invisible 180-degree axial roll. The raw torso solve already has
            # a consistently labelled shoulder reference, so use it only to
            # choose the hemisphere of the inferred reference.
            baseline_reference = getattr(channel, "desired_reference", None)
            if reference is not None and baseline_reference is not None:
                axis = direction.normalized()
                candidate = reference - axis * reference.dot(axis)
                baseline = baseline_reference - axis * baseline_reference.dot(axis)
                if (
                    candidate.length > 1.0e-8
                    and baseline.length > 1.0e-8
                    and candidate.dot(baseline) < 0.0
                ):
                    reference.negate()
            # Positions own swing and lateral roll. The rest-relative MHR
            # frame contributes only its component around the fitted tangent,
            # avoiding direct transfer of MHR joint pre-rotations or axes.
            joint_index = min(
                max(int(round(((start_fraction + end_fraction) * 0.5) * 3.0)), 0),
                3,
            )
            observation = pose.frame.joint_frames.get(f"MHR_SPINE_{joint_index}")
            if reference is not None and observation and observation.valid:
                values = observation.rotation_matrix
                raw_delta = Matrix(
                    tuple(
                        tuple(float(values[row * 3 + column]) for column in range(3))
                        for row in range(3)
                    )
                )
                canonical = Matrix(
                    (
                        (1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0),
                        (0.0, -1.0, 0.0),
                    )
                )
                canonical_delta = canonical @ raw_delta @ canonical.inverted_safe()
                relative_delta = (
                    pose.source_basis.inverted_safe()
                    @ canonical_delta
                    @ pose.source_basis
                )
                target_delta = (
                    (
                        pose.target_basis
                        @ relative_delta
                        @ pose.target_basis.inverted_safe()
                    )
                    .to_quaternion()
                    .normalized()
                )
                twist = _signed_twist_angle(target_delta, direction.normalized())
                twist = min(max(twist, -math.radians(60.0)), math.radians(60.0))
                reference = (
                    Quaternion(direction.normalized(), twist * settings.strength)
                    @ reference
                )
            previous_rotation = (
                channel.desired_rotation.copy()
                if getattr(channel, "desired_rotation", None) is not None
                else None
            )
            rest_rotation = channel.bone.bone.matrix_local.to_3x3()
            transported_direction = rigid_delta @ channel.rest_direction
            transported_reference = (
                rigid_delta @ channel.rest_reference
                if channel.rest_reference is not None
                else None
            )
            local_delta = _stable_orientation_delta(
                transported_direction,
                direction,
                transported_reference,
                reference,
                reference_is_oriented=not inferred,
            )
            applied = local_delta is not None
            if applied:
                desired_rotation = (
                    local_delta.to_matrix() @ rigid_delta.to_matrix() @ rest_rotation
                )
                pose.set_desired_rotation(
                    channel.semantic_id,
                    channel.bone,
                    desired_rotation,
                    correction=channel.correction,
                    confidence=channel.confidence,
                    influence=min(max(channel.influence * settings.strength, 0.0), 1.0),
                    owner=self.module_id,
                )
                channel = pose.channel_for(channel.semantic_id)
                # Keep the original rest observations as invariants for
                # downstream endpoint reconstruction. Applying the transport
                # to these fields would make later correctors count it twice.
                channel.desired_direction = direction.copy()
                channel.desired_reference = (
                    reference.copy() if reference is not None else None
                )
            else:
                applied = pose.aim(
                    channel.semantic_id,
                    channel.bone,
                    direction,
                    rest_direction=channel.rest_direction,
                    desired_reference=reference,
                    rest_reference=channel.rest_reference,
                    correction=channel.correction,
                    confidence=channel.confidence,
                    influence=min(max(channel.influence * settings.strength, 0.0), 1.0),
                    owner=self.module_id,
                    # MHR supplies an observed axial frame. MediaPipe's inferred
                    # hip/shoulder lateral is only a bend-plane axis and must use
                    # the same closest-hemisphere policy as the base torso solve.
                    reference_is_oriented=not inferred,
                )
            # Sparse MediaPipe landmarks do not observe vertebrae. Even after
            # hemisphere correction, cap the *additional* inferred correction
            # rather than allowing an ambiguous reference to overturn a valid
            # raw torso solve. SAM's observed MHR centerline is not subject to
            # this conservative inference cap.
            maximum_correction = math.radians(45.0) if inferred else None
            if (
                applied
                and maximum_correction is not None
                and previous_rotation is not None
                and getattr(channel, "desired_rotation", None) is not None
            ):
                previous_q = previous_rotation.to_quaternion().normalized()
                proposed_q = channel.desired_rotation.to_quaternion().normalized()
                if previous_q.dot(proposed_q) < 0.0:
                    proposed_q.negate()
                correction_angle = previous_q.rotation_difference(proposed_q).angle
                if correction_angle > maximum_correction:
                    channel.desired_rotation = previous_q.slerp(
                        proposed_q,
                        maximum_correction / correction_angle,
                    ).to_matrix()
                    if inferred:
                        warnings.append(
                            f"{channel.semantic_id.replace('_', ' ').title()} "
                            "inferred bend was clamped to 45 degrees"
                        )
        pose.invalidate()
        pose.resolve()
        for (
            roles,
            limb,
            endpoint,
            old_joint,
            first_length,
            second_length,
            old_plane,
        ) in limb_anchors:
            upper, lower, _end = limb
            root = pose._virtual_pose(upper.bone).translation.copy()
            to_end = endpoint - root
            distance = to_end.length
            if distance < 1.0e-8:
                continue
            maximum = first_length + second_length - 1.0e-5
            minimum = abs(first_length - second_length) + 1.0e-5
            clamped_distance = min(max(distance, minimum), maximum)
            target = root + to_end.normalized() * clamped_distance
            if abs(clamped_distance - distance) > 1.0e-5:
                warnings.append(
                    f"{roles[0]} endpoint became unreachable after spine correction; correction was clamped"
                )
            axis = (target - root).normalized()
            along = (
                clamped_distance * clamped_distance
                + first_length * first_length
                - second_length * second_length
            ) / (2.0 * clamped_distance)
            height = math.sqrt(max(first_length * first_length - along * along, 0.0))
            projected = old_joint - (root + axis * (old_joint - root).dot(axis))
            if projected.length < 1.0e-7 and old_plane.length > 1.0e-7:
                projected = old_plane.normalized().cross(axis)
            if projected.length < 1.0e-7:
                projected = axis.orthogonal()
            bend = projected.normalized()
            joint = root + axis * along + bend * height
            upper_direction = joint - root
            lower_direction = target - joint
            pose.aim(
                upper.semantic_id,
                upper.bone,
                upper_direction,
                rest_direction=upper.rest_direction,
                desired_reference=lower_direction,
                rest_reference=upper.rest_reference,
                correction=upper.correction,
                confidence=upper.confidence,
                influence=upper.influence,
                owner=self.module_id,
            )
            pose.aim(
                lower.semantic_id,
                lower.bone,
                lower_direction,
                rest_direction=lower.rest_direction,
                desired_reference=upper_direction,
                rest_reference=lower.rest_reference,
                correction=lower.correction,
                confidence=lower.confidence,
                influence=lower.influence,
                owner=self.module_id,
            )
        return warnings


class ShoulderCorrection(CorrectionModule):
    module_id = "SHOULDER_CORRECTION"
    display_name = "Shoulder Correction"
    # The optional silhouette repair deliberately has a tightly bounded
    # endpoint budget and reports its actual shift itself.
    allows_endpoint_motion = True

    @staticmethod
    def _side_from_name(name: str) -> str | None:
        lowered = name.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
        if "left" in tokens or "l" in tokens:
            return "L"
        if "right" in tokens or "r" in tokens:
            return "R"
        return None

    @staticmethod
    def _image_elbow_angle(frame, side: str) -> float | None:
        prefix = "LEFT" if side == "L" else "RIGHT"
        points = [
            frame.body.get(f"{prefix}_{joint}")
            for joint in ("SHOULDER", "ELBOW", "WRIST")
        ]
        if any(point is None or not point.valid for point in points):
            return None
        shoulder, elbow, wrist = (Vector(point.image) for point in points)
        first = shoulder - elbow
        second = wrist - elbow
        if first.length < 1.0e-8 or second.length < 1.0e-8:
            return None
        angle = first.angle(second)
        return angle if math.isfinite(angle) else None

    def _repair_arm_silhouette(self, pose, settings) -> list[str]:
        """Bounded SAM 2D/3D reconciliation for visibly over-bent arms.

        A monocular image angle is not a complete 3D observation, so it is
        only allowed to extend a chain which SAM's articulated result bends
        substantially more.  The endpoint stays on its existing root-to-hand
        ray and can move by at most a small user-visible budget.
        """
        if not getattr(settings, "repair_arm_silhouette", True):
            return []
        limit_fraction = min(
            max(float(getattr(settings, "maximum_arm_extension", 0.02)), 0.0), 0.05
        )
        if limit_fraction <= 1.0e-7:
            return []
        if not any(
            point is not None and point.valid
            for point in (
                pose.frame.auxiliary.get("MHR_UPPER_ARM_L"),
                pose.frame.auxiliary.get("MHR_UPPER_ARM_R"),
            )
        ):
            return []
        warnings = []
        for side, label in (("L", "Left"), ("R", "Right")):
            # The full MHR observation is the gate that keeps this repair out
            # of MediaPipe's less reliable stylized-image path.
            mhr = pose.frame.auxiliary.get(f"MHR_UPPER_ARM_{side}")
            if mhr is None or not mhr.valid:
                continue
            image_angle = self._image_elbow_angle(pose.frame, side)
            upper = pose.channel_for(f"UPPER_ARM_{side}")
            lower = pose.channel_for(f"LOWER_ARM_{side}")
            if image_angle is None or upper is None or lower is None:
                continue
            pose.resolve()
            upper_pose = pose._virtual_pose(upper.bone)
            lower_pose = pose._virtual_pose(lower.bone)
            upper_length = max(float(upper.bone.bone.length), 1.0e-8)
            lower_length = max(float(lower.bone.bone.length), 1.0e-8)
            upper_direction = (
                upper_pose.to_3x3() @ Vector((0.0, 1.0, 0.0))
            ).normalized()
            lower_direction = (
                lower_pose.to_3x3() @ Vector((0.0, 1.0, 0.0))
            ).normalized()
            current_angle = (-upper_direction).angle(lower_direction)
            # Ignore small disagreement and cases where the image is more
            # bent.  The latter often represents real out-of-plane motion.
            disagreement = image_angle - current_angle
            if disagreement < math.radians(5.0):
                continue
            blend = min(max(0.60 * float(settings.strength), 0.0), 0.75)
            desired_angle = min(
                current_angle + disagreement * blend,
                math.radians(172.0),
            )
            desired_reach_sq = (
                upper_length * upper_length
                + lower_length * lower_length
                - 2.0 * upper_length * lower_length * math.cos(desired_angle)
            )
            desired_reach = math.sqrt(max(desired_reach_sq, 0.0))
            root = upper_pose.translation.copy()
            old_joint = root + upper_direction * upper_length
            old_end = old_joint + lower_direction * lower_length
            root_to_end = old_end - root
            current_reach = root_to_end.length
            total_length = upper_length + lower_length
            if current_reach < 1.0e-8:
                continue
            maximum_shift = total_length * limit_fraction
            reach = min(
                desired_reach,
                current_reach + maximum_shift,
                total_length * (1.0 - 1.0e-6),
            )
            shift = reach - current_reach
            if shift <= total_length * 1.0e-5:
                continue
            axis = root_to_end.normalized()
            along = (
                upper_length * upper_length
                - lower_length * lower_length
                + reach * reach
            ) / (2.0 * reach)
            height_sq = upper_length * upper_length - along * along
            height = math.sqrt(max(height_sq, 0.0))
            bend = old_joint - root
            bend -= axis * bend.dot(axis)
            if bend.length < 1.0e-8:
                reference = (
                    upper.desired_reference
                    if upper.desired_reference is not None
                    else upper.rest_reference
                )
                if reference is not None:
                    bend = reference - axis * reference.dot(axis)
            if bend.length < 1.0e-8:
                fallback = Vector((0.0, 0.0, 1.0))
                if abs(axis.dot(fallback)) > 0.95:
                    fallback = Vector((1.0, 0.0, 0.0))
                bend = fallback - axis * fallback.dot(axis)
            bend.normalize()
            end = root + axis * reach
            joint = root + axis * along + bend * height
            new_upper = (joint - root).normalized()
            new_lower = (end - joint).normalized()
            pose.aim(
                upper.semantic_id,
                upper.bone,
                new_upper,
                rest_direction=upper.rest_direction,
                desired_reference=upper.desired_reference,
                rest_reference=upper.rest_reference,
                correction=upper.correction,
                confidence=upper.confidence,
                influence=upper.influence,
                owner=self.module_id,
            )
            pose.aim(
                lower.semantic_id,
                lower.bone,
                new_lower,
                rest_direction=lower.rest_direction,
                desired_reference=new_upper,
                rest_reference=lower.rest_reference,
                correction=lower.correction,
                confidence=lower.confidence,
                influence=lower.influence,
                owner=self.module_id,
            )
            warnings.append(
                f"{label} arm silhouette repair moved the wrist/IK target "
                f"{shift / total_length * 100.0:.2f}% of arm length"
            )
            pose.resolve()
        return warnings

    def validate(self, armature, settings) -> list[str]:
        entries = {
            entry.semantic_id: entry for entry in armature.mtp_rig_config.mappings
        }
        warnings = []
        for side, label in (("L", "Left"), ("R", "Right")):
            upper = entries.get(f"UPPER_ARM_{side}")
            shoulder = entries.get(f"SHOULDER_{side}")
            if (
                upper
                and upper.target_name
                and (not shoulder or not shoulder.target_name)
            ):
                warnings.append(f"{label} shoulder correction has no mapped shoulder")
        return warnings

    def process_frame(self, pose, settings, session) -> list[str]:
        warnings = []
        rig_up = pose.target_basis @ Vector((0.0, 0.0, 1.0))
        auxiliary = pose.frame.auxiliary
        left_acromion = auxiliary.get("LEFT_ACROMION")
        right_acromion = auxiliary.get("RIGHT_ACROMION")
        acromion_mid = None
        if (
            left_acromion
            and right_acromion
            and left_acromion.valid
            and right_acromion.valid
        ):
            acromion_mid = (
                Vector(left_acromion.position) + Vector(right_acromion.position)
            ) * 0.5
        mhr_clavicles = {
            side: auxiliary.get(f"MHR_CLAVICLE_{side}") for side in ("L", "R")
        }
        mhr_upper_arms = {
            side: auxiliary.get(f"MHR_UPPER_ARM_{side}") for side in ("L", "R")
        }
        mhr_mid = None
        if all(point is not None and point.valid for point in mhr_clavicles.values()):
            mhr_mid = (
                Vector(mhr_clavicles["L"].position)
                + Vector(mhr_clavicles["R"].position)
            ) * 0.5
        for side, label in (("L", "Left"), ("R", "Right")):
            upper = pose.channel_for(f"UPPER_ARM_{side}")
            if not upper or upper.desired_direction is None:
                continue
            direction = upper.desired_direction.normalized()
            elevation = math.asin(
                min(max(direction.dot(rig_up.normalized()), -1.0), 1.0)
            )
            elevation = max(0.0, elevation)
            shoulder = pose.channel_for(f"SHOULDER_{side}")
            if shoulder is None:
                if elevation >= math.radians(60.0):
                    warnings.append(
                        f"{label} arm is elevated but its shoulder is unmapped"
                    )
                continue
            if shoulder.desired_direction is None:
                continue
            mhr_clavicle = mhr_clavicles.get(side)
            mhr_upper_arm = mhr_upper_arms.get(side)
            acromion = auxiliary.get(f"{'LEFT' if side == 'L' else 'RIGHT'}_ACROMION")
            # MHR exposes the actual articulated clavicle -> upper-arm joint
            # segment. The old midpoint -> clavicle vector represented only
            # the medial half of the shoulder girdle and omitted the outer
            # clavicle/acromion drop, displacing the arm root on stylized rigs.
            if (
                mhr_clavicle is not None
                and mhr_clavicle.valid
                and mhr_upper_arm is not None
                and mhr_upper_arm.valid
            ):
                observed = pose.to_target_direction(
                    Vector(mhr_upper_arm.position) - Vector(mhr_clavicle.position)
                )
                if observed.length > 1.0e-8:
                    shoulder.desired_direction = observed
                    shoulder.confidence = min(
                        shoulder.confidence,
                        mhr_clavicle.confidence,
                        mhr_upper_arm.confidence,
                    )
            elif mhr_mid is not None and mhr_clavicle is not None:
                observed = pose.to_target_direction(
                    Vector(mhr_clavicle.position) - mhr_mid
                )
                if observed.length > 1.0e-8:
                    shoulder.desired_direction = observed
                    shoulder.confidence = min(
                        shoulder.confidence, mhr_clavicle.confidence
                    )
            if (
                mhr_mid is None
                and acromion_mid is not None
                and acromion
                and acromion.valid
            ):
                observed = pose.to_target_direction(
                    Vector(acromion.position) - acromion_mid
                )
                if observed.length > 1.0e-8:
                    shoulder.desired_direction = observed
                    shoulder.confidence = min(
                        shoulder.confidence,
                        acromion.confidence,
                    )
            ramp = _smoothstep(
                settings.start_elevation,
                settings.full_influence_elevation,
                elevation,
            )
            contribution = 0.30 * settings.strength * ramp
            current = shoulder.desired_direction.normalized()
            difference = current.rotation_difference(direction)
            angle = min(
                difference.angle * contribution,
                settings.maximum_swing,
            )
            if angle > 1.0e-7:
                assisted = Quaternion(difference.axis, angle) @ current
                pose.aim(
                    shoulder.semantic_id,
                    shoulder.bone,
                    assisted,
                    rest_direction=shoulder.rest_direction,
                    desired_reference=shoulder.desired_reference,
                    rest_reference=shoulder.rest_reference,
                    correction=shoulder.correction,
                    confidence=shoulder.confidence,
                    influence=shoulder.influence,
                    owner=self.module_id,
                )

        warnings.extend(self._repair_arm_silhouette(pose, settings))
        return warnings


class _AxisCorrection(CorrectionModule):
    """Shared axial-twist and hinge mechanics for one limb family."""

    _UPPER: tuple[str, ...] = ()
    _HINGES: tuple[str, ...] = ()
    _TWIST_LIMIT_FIELD = ""
    _USES_ARM_SURFACE = False
    _USES_PALM_ROLL = False

    @staticmethod
    def _hinge_axis(channel: BoneChannel) -> Vector | None:
        bone = channel.bone
        unlocked = [
            axis
            for axis, locked in zip(
                (
                    Vector((1.0, 0.0, 0.0)),
                    Vector((0.0, 1.0, 0.0)),
                    Vector((0.0, 0.0, 1.0)),
                ),
                bone.lock_rotation,
                strict=False,
            )
            if not locked
        ]
        if len(unlocked) == 1:
            return unlocked[0]
        if channel.rest_direction is None or channel.rest_reference is None:
            return None
        hinge_world = channel.rest_reference.cross(channel.rest_direction)
        if hinge_world.length < 1.0e-8:
            return None
        local = (
            bone.bone.matrix_local.to_3x3().inverted_safe() @ hinge_world.normalized()
        )
        components = (abs(local.x), abs(local.y), abs(local.z))
        index = components.index(max(components))
        axis = (
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
        )[index]
        return axis if local[index] >= 0.0 else -axis

    def validate(self, armature, settings) -> list[str]:
        entries = {
            entry.semantic_id: entry for entry in armature.mtp_rig_config.mappings
        }
        warnings = []
        for semantic_id in self._HINGES:
            entry = entries.get(semantic_id)
            if not entry or not entry.target_name:
                warnings.append(f"{semantic_id} hinge target is missing")
                continue
            bone = armature.pose.bones.get(entry.target_name)
            if not bone:
                warnings.append(f"{semantic_id} hinge bone is missing")
        return warnings

    def _palm_forearm_twist(
        self,
        pose: TargetPose,
        side: str,
        settings,
        session: PostProcessingSession,
    ) -> tuple[float, float] | None:
        # Forearm roll is part of the body chain.  Keep it in the same
        # observation frame as the body-owned wrist/hand solve; the dedicated
        # hand detector is allowed to drive fingers only.  Mixing the body
        # hand axis with a separately cropped hand palm produced large,
        # opposing forearm/wrist rotations when those two detections disagreed.
        label = "LEFT" if side == "L" else "RIGHT"
        body = pose.frame.body
        forearm = pose.channel_for(f"LOWER_ARM_{side}")
        hand_channel = pose.channel_for(f"HAND_{side}")
        if (
            forearm is None
            or hand_channel is None
            or hand_channel.rest_reference is None
            or forearm.rest_direction is None
            or forearm.desired_direction is None
        ):
            return None
        index = body.get(f"{label}_INDEX")
        pinky = body.get(f"{label}_PINKY")
        if not index or not pinky or not index.valid or not pinky.valid:
            return None
        desired_lateral = pose.to_target_direction(
            Vector(index.position) - Vector(pinky.position)
        )
        # Palm width is the anatomical flexion axis of the wrist.  Project it
        # directly around the forearm instead of first "unbending" the hand
        # with a shortest-arc rotation.  The latter becomes ambiguous for a
        # strongly flexed wrist and used to reverse forearm roll by 180 degrees.
        rest_forearm_axis = forearm.rest_direction.normalized()
        desired_forearm_axis = forearm.desired_direction.normalized()
        rest_lateral = (
            hand_channel.rest_reference
            - rest_forearm_axis * hand_channel.rest_reference.dot(rest_forearm_axis)
        )
        desired_lateral = desired_lateral - desired_forearm_axis * desired_lateral.dot(
            desired_forearm_axis
        )
        if rest_lateral.length < 1.0e-8 or desired_lateral.length < 1.0e-8:
            return None
        transported = (
            rest_forearm_axis.rotation_difference(desired_forearm_axis)
            @ rest_lateral.normalized()
        )
        transported = (
            transported - desired_forearm_axis * transported.dot(desired_forearm_axis)
        ).normalized()
        desired_lateral.normalize()
        angle = math.atan2(
            desired_forearm_axis.dot(transported.cross(desired_lateral)),
            min(max(transported.dot(desired_lateral), -1.0), 1.0),
        )
        # atan2 wraps at +/-pi even though a video may cross that boundary
        # smoothly. Preserve the phase before applying the anatomical limit;
        # otherwise -181 degrees becomes +179 and the two clamped poses differ
        # by 180 degrees.
        angle = session.unwrap_forearm_twist(side, angle)
        # Confidence is deliberately not an enable/disable gate. A hard gate
        # made forearm roll switch between corrected and uncorrected branches
        # as hand confidence crossed one threshold, which was visible as wrist
        # jitter. The sequence pass uses this value as an observation weight
        # and bridges weak intervals from neighboring reliable rotations.
        confidence = min(
            float(index.confidence),
            float(pinky.confidence),
            float(hand_channel.confidence),
        )
        correction = angle * settings.palm_roll_strength
        return (
            min(
                max(correction, -settings.forearm_twist_limit),
                settings.forearm_twist_limit,
            ),
            min(max(confidence, 0.0), 1.0),
        )

    def _apply_axial_roll(
        self,
        channel: BoneChannel,
        angle: float,
        confidence: float,
    ) -> None:
        if channel.rest_direction is None or channel.desired_direction is None:
            return
        rest_axis = channel.rest_direction.normalized()
        desired_axis = channel.desired_direction.normalized()
        swing = rest_axis.rotation_difference(desired_axis)
        delta = Quaternion(desired_axis, angle) @ swing
        if channel.influence < 1.0:
            delta = Quaternion().slerp(delta, channel.influence)
        rest_rotation = channel.bone.bone.matrix_local.to_3x3()
        channel.desired_rotation = delta.to_matrix() @ rest_rotation
        channel.confidence = min(
            float(channel.confidence), min(max(float(confidence), 0.0), 1.0)
        )
        channel.owner = self.module_id

    def process_frame(self, pose, settings, session) -> list[str]:
        warnings = []
        # MHR70 observes both the posterior olecranon and anterior cubital
        # fossa. Their vector is much stronger axial-roll evidence than a
        # nearly straight shoulder-elbow-wrist plane.
        if self._USES_ARM_SURFACE:
            for side, label in (("L", "LEFT"), ("R", "RIGHT")):
                channel = pose.channel_for(f"UPPER_ARM_{side}")
                olecranon = pose.frame.auxiliary.get(f"{label}_OLECRANON")
                cubital = pose.frame.auxiliary.get(f"{label}_CUBITAL_FOSSA")
                if (
                    channel is not None
                    and olecranon
                    and cubital
                    and olecranon.valid
                    and cubital.valid
                ):
                    reference = pose.to_target_direction(
                        Vector(cubital.position) - Vector(olecranon.position)
                    )
                    if reference.length > 1.0e-8:
                        # Re-solve the rotation immediately. Updating only the
                        # metadata left the base elbow-plane rotation untouched,
                        # so SAM's stronger surface observation had no effect.
                        pose.aim(
                            channel.semantic_id,
                            channel.bone,
                            channel.desired_direction,
                            rest_direction=channel.rest_direction,
                            desired_reference=reference,
                            rest_reference=channel.rest_reference,
                            correction=channel.correction,
                            confidence=min(
                                channel.confidence,
                                olecranon.confidence,
                                cubital.confidence,
                            ),
                            influence=channel.influence,
                            owner=self.module_id,
                        )
        for semantic_id in self._UPPER:
            channel = pose.channel_for(semantic_id)
            if (
                channel is None
                or channel.desired_rotation is None
                or channel.rest_direction is None
                or channel.desired_direction is None
            ):
                continue
            rest_rotation = channel.bone.bone.matrix_local.to_3x3()
            full_delta = (
                (channel.desired_rotation @ rest_rotation.inverted_safe())
                .to_quaternion()
                .normalized()
            )
            swing = channel.rest_direction.rotation_difference(
                channel.desired_direction
            )
            residual_rotation = swing.inverted() @ full_delta
            twist = _signed_twist_angle(residual_rotation, channel.rest_direction)
            reliable = (
                channel.confidence >= settings.confidence_threshold
                and channel.desired_reference is not None
                and _orientation_frame(
                    channel.desired_direction,
                    channel.desired_reference,
                )
                is not None
            )
            if not reliable:
                continue
            limit = getattr(settings, self._TWIST_LIMIT_FIELD)
            twist = min(max(twist, -limit), limit)
            stabilized = swing @ Quaternion(channel.rest_direction.normalized(), twist)
            if channel.influence < 1.0:
                stabilized = Quaternion().slerp(stabilized, channel.influence)
            channel.desired_rotation = stabilized.to_matrix() @ rest_rotation
            channel.owner = self.module_id

        # Body-pose palm landmarks contain pronation/supination evidence that
        # the elbow-wrist line does not. Apply that roll around the mapped
        # forearm's own axis. Dedicated hand landmarks remain finger-only.
        rolled_forearms = set()
        if self._USES_PALM_ROLL:
            for side in ("L", "R"):
                observation = self._palm_forearm_twist(pose, side, settings, session)
                channel = pose.channel_for(f"LOWER_ARM_{side}")
                if observation is not None and channel is not None:
                    angle, confidence = observation
                    self._apply_axial_roll(channel, angle, confidence)
                    rolled_forearms.add(channel.semantic_id)

        if settings.hinge_projection:
            for semantic_id in self._HINGES:
                channel = pose.channel_for(semantic_id)
                if channel is None:
                    continue
                if semantic_id in rolled_forearms:
                    # Projecting the basis to a single hinge axis would erase
                    # the independently observed forearm pronation.
                    continue
                pose.replace_with_swing_only(channel, owner=self.module_id)
                axis = self._hinge_axis(channel)
                if axis is None:
                    warnings.append(
                        f"Could not infer hinge axis for {channel.bone.name}"
                    )
                    continue
                if getattr(pose, "diagnostic_mode", False):
                    direction_error = pose.hinge_projection_direction_error(
                        channel, axis
                    )
                    if direction_error > math.radians(0.5):
                        warnings.append(
                            f"Applied hinge projection for {channel.bone.name} "
                            f"despite {math.degrees(direction_error):.1f} degrees "
                            "of endpoint-direction change because the correction is enabled"
                        )
                channel.hinge_axis_local = axis
        pose.invalidate()
        return warnings


class ArmAxisCorrection(_AxisCorrection):
    module_id = "ARM_AXIS_CORRECTION"
    display_name = "Arm Axis Correction"
    _UPPER = ("UPPER_ARM_L", "UPPER_ARM_R")
    _HINGES = ("LOWER_ARM_L", "LOWER_ARM_R")
    _TWIST_LIMIT_FIELD = "upper_arm_twist_limit"
    _USES_ARM_SURFACE = True
    _USES_PALM_ROLL = True


class LegAxisCorrection(_AxisCorrection):
    module_id = "LEG_AXIS_CORRECTION"
    display_name = "Leg Axis Correction"
    _UPPER = ("UPPER_LEG_L", "UPPER_LEG_R")
    _HINGES = ("LOWER_LEG_L", "LOWER_LEG_R")
    _TWIST_LIMIT_FIELD = "thigh_twist_limit"


class TemporalCorrection(CorrectionModule):
    module_id = "TEMPORAL_CORRECTION"
    display_name = "Temporal Correction"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        yield from context.process_temporal(settings)


class CameraMotionCorrection(CorrectionModule):
    module_id = "CAMERA_MOTION_CORRECTION"
    display_name = "Camera Motion Correction"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        if context.process_camera_motion is None:
            raise RuntimeError(
                "Camera Motion Correction is unavailable outside a video bake"
            )
        yield from context.process_camera_motion(settings)


class DetectionFilter(CorrectionModule):
    module_id = "DETECTION_FILTER"
    display_name = "Detection Filter"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        if context.process_detection_filter is None:
            raise RuntimeError("Detection Filter is unavailable outside a video bake")
        yield from context.process_detection_filter(settings)


class RohmCorrection(CorrectionModule):
    module_id = "ROHM_CORRECTION"
    display_name = "RoHM Correction"
    scope = "SEQUENCE"

    def process_sequence(self, context, settings) -> Iterator[Any]:
        if context.process_rohm is None:
            raise RuntimeError("RoHM Correction is unavailable outside a video bake")
        yield from context.process_rohm(settings)


CORRECTION_REGISTRY = {
    module.module_id: module
    for module in (
        SpineCorrection(),
        ShoulderCorrection(),
        ArmAxisCorrection(),
        LegAxisCorrection(),
        CameraMotionCorrection(),
        DetectionFilter(),
        RohmCorrection(),
        TemporalCorrection(),
    )
}


def validate_modules(armature) -> list[str]:
    config = armature.mtp_rig_config
    ensure_module_configs(config)
    warnings = []
    for item in config.post_processing_modules:
        module = CORRECTION_REGISTRY[item.module_id]
        profile = config.detected_profile
        if profile not in module.supported_profiles:
            issue = (
                f"{item.display_name or item.module_id} does not support "
                f"the {profile} rig profile"
            )
            item.status = "WARNING"
            item.last_warning = issue
            if item.enabled:
                warnings.append(issue)
            continue
        issues = module.validate(armature, item)
        item.status = "WARNING" if issues else "VALID"
        item.last_warning = "; ".join(issues)
        if item.enabled:
            warnings.extend(issues)
    return warnings


def iter_pipeline(
    pose: TargetPose,
    config,
    *,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
) -> Iterator[tuple[str, int, int]]:
    """Apply enabled modules cooperatively, yielding before every module."""
    compiled = compile_postprocess_config(config)
    pose.diagnostic_mode = compiled.diagnostic_mode
    session = session or PostProcessingSession()
    session.prepare_frame(int(pose.frame.timestamp_ms))
    warnings = []
    modules = [
        item
        for item in compiled.modules
        if CORRECTION_REGISTRY[item.module_id].scope == "FRAME"
        if (
            item.module_id in module_filter
            if module_filter is not None
            else item.enabled
        )
    ]
    total = len(modules)
    for index, item in enumerate(modules, 1):
        module_id = item.module_id
        yield item.display_name or module_id, index, total
        module = CORRECTION_REGISTRY[module_id]
        profile = compiled.detected_profile
        if profile not in module.supported_profiles:
            warning = (
                f"{item.display_name or item.module_id} does not support "
                f"the {profile} rig profile"
            )
            warnings.append(warning)
            continue
        before = (
            pose.semantic_endpoints()
            if compiled.diagnostic_mode and not module.allows_endpoint_motion
            else None
        )
        module_warnings = module.process_frame(pose, item, session)
        pose.applied_modules.append(item.module_id)
        if before is not None:
            after = pose.semantic_endpoints()
            displacement, semantic_id, endpoint = pose.maximum_endpoint_delta(
                before, after
            )
            if displacement > 1.0e-3:
                module_warnings.append(
                    f"{item.display_name or item.module_id} remained enabled: "
                    f"{semantic_id} {endpoint} moved "
                    f"{displacement * 100.0:.2f}% of its bone length"
                )
        if module_warnings:
            warnings.extend(module_warnings)
    pose.warnings.extend(warnings)
    return warnings


def iter_sequence_pipeline(
    context: SequenceProcessingContext,
    config,
) -> Iterator[Any]:
    """Run enabled complete-sequence processors in registry order."""
    compiled = compile_postprocess_config(config)
    modules = [
        item
        for item in compiled.modules
        if item.enabled and CORRECTION_REGISTRY[item.module_id].scope == "SEQUENCE"
    ]
    for item in modules:
        yield from CORRECTION_REGISTRY[item.module_id].process_sequence(context, item)


def run_pipeline(
    pose: TargetPose,
    config,
    *,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
) -> list[str]:
    """Apply the complete pipeline synchronously."""
    iterator = iter_pipeline(
        pose,
        config,
        session=session,
        module_filter=module_filter,
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value
